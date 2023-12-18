/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libutil/CLOptions.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages2/covariance_complete.cuh"
#include "libmycu/custages2/covariance_refn_complete.cuh"
#include "linear_scoring2_complete.cuh"

// =========================================================================
// ScoreFragmentBasedSuperpositionsLinearly2: perform and score 
// fragment-based superposition using index in linear time;
// // napiterations, #iterations to perform excluding the initial (tfm) and 
// // final scoring: 1, 2, or 3;
// // scoreachiteration, flag of whether to perform scoring in each nap_iteration;
// stacksize, dynamically determined stack size;
// depth, superposition depth for calculating query and reference positions;
// nqystrs, total number of queries;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db structure positions in the chunk;
// maxnsteps, max number of steps that can be performed in one pass;
// qryfragfct, argument 1 for calculating starting query position;
// rfnfragfct, argument 2 for calculating starting reference position;
// NOTE: memory pointers should be aligned!
// tmpdpalnpossbuffer, temporary buffers of aligned coordinates;
// tmpdpdiagbuffers, temporary diagonal buffers for positional scores and distances;
// wrkmemtmibest, working memory for best-performing transformation matrices;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// 
__global__ 
void ScoreFragmentBasedSuperpositionsLinearly2(
    const int stacksize,
    const int depth,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int qryfragfct,
    const int rfnfragfct,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux)
{
    const uint dbstrndx = blockIdx.x;//reference serial number
    const uint sfragfct = blockIdx.y;//fragment factor
    const uint qryndx = blockIdx.z;//query serial number
    const int fragndx = (sfragfct & 1);//fragment length index
    enum {neffds = twmvEndOfCCDataExt,//effective number of fields
        smidim = neffds+1,//total number to resolve bank conflicts
        maxszstack = 17,//max size for stack
        //hash table size:
        SZQNXCH = (maxszstack-1) * CUSF_TBSP_COMPLETE_INDEX_SCORE_XDIM
    };
    //NOTE: CUSF_TBSP_COMPLETE_INDEX_SCORE_XDIM==CUS1_TBINITSP_COMPLETEREFINE_XDIM
    //NOTE: CUSF_TBSP_COMPLETE_INDEX_SCORE_XDIM==CUS1_TBINITSP_COMPLETEREFINE_XDIM
    __shared__ float dSMEM[smidim + maxszstack * nStks_ * CUSF_TBSP_COMPLETE_INDEX_SCORE_XDIM];
    float* tfmCache = dSMEM;//cache for tfm: [smidim>nTTranformMatrix] for in-place comp.
    float* trtStack = tfmCache + smidim;//[maxszstack * nStks_ * XDIM]; smidim (+1) resolves bank conflicts
    float* ccmCache = trtStack;//[smidim * XDIM];

    int qrylen, dbstrlen;//query and reference lengths
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;
    int qrypos, rfnpos;//const


    if(threadIdx.x == 0) {
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        ccmCache[6] = ccmCache[7] = wrkmemaux[mloc0 + dbstrndx];
        if(sfragfct != 0) ccmCache[7] = wrkmemaux[mloc + dbstrndx];
    }

    __syncthreads();

    if((((int)ccmCache[6]) & (CONVERGED_LOWTMSC_bitval)) || ccmCache[7])
        //(NOTE:any type of convergence applies locally);
        //all threads in the block exit;
        return;

    //NOTE: no sync as long ccmCache cell for convergence is not overwritten;


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse ccmCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(dbstrndx, (int*)ccmCache);
        GetQueryLenDst(qryndx, (int*)ccmCache + 2);
    }

    __syncthreads();


    dbstrlen = ((int*)ccmCache)[0]; dbstrdst = ((int*)ccmCache)[1];
    qrylen = ((int*)ccmCache)[2]; qrydst = ((int*)ccmCache)[3];

    __syncthreads();

    GetQryRfnPos_frg2(
        depth,
        qrypos, rfnpos, qrylen, dbstrlen, sfragfct, qryfragfct, rfnfragfct, fragndx
    );

    const int nalnposs = 
        GetNAlnPoss_frg(qrylen, dbstrlen, qrypos, rfnpos, qryfragfct, rfnfragfct, fragndx);

    //if fragment is out of bounds (tfm not calculated): all threads in the block exit
    if(qrylen < qrypos + nalnposs || dbstrlen < rfnpos + nalnposs) return;


    //threshold calculated for the original lengths
    const float d02 = GetD02(qrylen, dbstrlen);
    float best = 0.0f;//best score obtained


    for(;;)
    {
        //INITIAL
        CalcCCMatrices64_Complete<smidim,neffds>(
            qrydst, dbstrdst, nalnposs, 
            qrypos + nalnposs, rfnpos + nalnposs, qrypos, rfnpos, ccmCache);

        //NOTE: synced above and below before ccmCache gets updated;
        if(ccmCache[twmvNalnposs] < 1.0f) break;//all exit

        CopyCCMDataToTFM_REVERSETfm_Complete(ccmCache, tfmCache);
        //NOTE: tfmCache updated by the first warp; 
        __syncwarp();

        CalcTfmMatrices_Complete(tfmCache, qrylen, dbstrlen);
        //sync all threads to see the REVERSE-mode tfm
        __syncthreads();


        int maxalnlen;//IT #0
        ProduceAlignmentUsingIndex2_Complete<true/*SECSTRFILT*/,false/*WRTNDX*/>(
            maxalnlen/*out*/, stacksize,
            qryndx, ndbCposs, maxnsteps, sfragfct,
            qrydst, dbstrdst, qrylen, dbstrlen, qrypos, rfnpos,
            tfmCache, trtStack, tmpdpalnpossbuffer, tmpdpdiagbuffers);
        //sync: trtStack and ccmCache share the same address
        __syncthreads();

        CalcCCMatrices64_SWFTscan_Complete<smidim,neffds>(
            qryndx, ndbCposs, maxnsteps, sfragfct, dbstrdst,
            maxalnlen/*qrylen*/, maxalnlen/*dbstrlen*/, 0/*qrypos*/, 0/*rfnpos*/,
            tmpdpalnpossbuffer, ccmCache);

        //NOTE: synced above and below before ccmCache gets updated;
        if(ccmCache[twmvNalnposs] < 1.0f) break;//all exit

        CopyCCMDataToTFM_REVERSETfm_Complete(ccmCache, tfmCache);
        //NOTE: tfmCache updated by the first warp; 
        __syncwarp();

        CalcTfmMatrices_Complete(tfmCache, qrylen, dbstrlen);
        //sync all threads to see the REVERSE-mode tfm
        __syncthreads();


        //IT #1
        ProduceAlignmentUsingIndex2_Complete<false/*SECSTRFILT*/,true/*WRTNDX*/>(
            maxalnlen/*out*/, stacksize,
            qryndx, ndbCposs, maxnsteps, sfragfct,
            qrydst, dbstrdst, qrylen, dbstrlen, qrypos, rfnpos,
            tfmCache, trtStack, tmpdpalnpossbuffer, tmpdpdiagbuffers);
        //sync: trtStack and ccmCache share the same address
        __syncthreads();

        CalcCCMatrices64_SWFTscan_Complete<smidim,neffds>(
            qryndx, ndbCposs, maxnsteps, sfragfct, dbstrdst,
            maxalnlen/*qrylen*/, maxalnlen/*dbstrlen*/, 0/*qrypos*/, 0/*rfnpos*/,
            tmpdpalnpossbuffer, ccmCache);

        //NOTE: synced above and below before ccmCache gets updated;
        if(ccmCache[twmvNalnposs] < 1.0f) break;//all exit

        CopyCCMDataToTFM_Complete(ccmCache, tfmCache);
        //NOTE: tfmCache updated by the first warp; 
        __syncwarp();

        CalcTfmMatrices_Complete(tfmCache, qrylen, dbstrlen);
        //sync all threads to see the tfm
        __syncthreads();


        //SCORE alignment:
        best = 
        CalcScoresUnrl_SWFTscanProgressive_Complete<SZQNXCH>(
            qryndx, ndbCposs, maxnsteps, sfragfct, dbstrdst,
            maxalnlen/*qrylen*/, maxalnlen/*dbstrlen*/, 0/*qrypos*/, 0/*rfnpos*/,
            d02,  tfmCache, tmpdpalnpossbuffer, tmpdpdiagbuffers,
            trtStack+1/*qnxCache*/, trtStack+1+SZQNXCH/*maxCache*/);
        break;
    }

    //NOTE: warp-synced above:
    SaveBestScoreAndTM_Complete<false/*WRITEFRAGINFO*/,true/*CONDITIONAL*/>(
        best,  qryndx, dbstrndx, ndbCstrs,
        maxnsteps, sfragfct, 0/*sfragndx(unused)*/, 0/*sfragpos(unused)*/,
        tfmCache, wrkmemtmibest, wrkmemaux);
}

// =========================================================================
