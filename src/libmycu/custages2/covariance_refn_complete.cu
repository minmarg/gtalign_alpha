/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/covariance.cuh"
#include "libmycu/custages/covariance_plus.cuh"
#include "libmycu/custages/covariance_refn.cuh"
#include "libmycu/custages/covariance_dp_refn.cuh"
#include "libmycu/custages2/covariance_complete.cuh"
#include "covariance_refn_complete.cuh"

// =========================================================================
// FragmentBasedAlignmentRefinement: refine alignment and its boundaries 
// within the single kernel's actions to obtain favorable superposition;
// WRITEFRAGINFO, template parameter, flag of whether refined fragment 
// boundaries should be saved;
// TFM_DINV, use doubly inverted transformation matrices under suitable conditions;
// nmaxconvit, maximum number of superposition iterations;
// nqystrs, total number of query structures in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of reference positions in the chunk;
// nmaxsubfrags, total number of fragment lengths to consider;
// maxnsteps, total number of steps that should be performed for each reference structure;
// sfragstep, step size to traverse subfragments;
// maxalnmax, maximum alignment length across all query-reference pairs;
// tmpdpdiagbuffers, temporary diagonal buffers for positional scores;
// wrkmemtmibest, working memory for best-performing transformation matrices;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// 
template<bool WRITEFRAGINFO, bool TFM_DINV>
__global__ 
void FragmentBasedAlignmentRefinement(
    const int nmaxconvit,
//     const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint nmaxsubfrags,
    const uint maxnsteps,
    const int sfragstep,
    const int /*maxalnmax*/,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux)
{
    uint dbstrndx = blockIdx.x;//reference serial number
    uint sfragfctxndx = blockIdx.y;//fragment factor x fragment length index
    uint sfragfct = sfragfctxndx / nmaxsubfrags;//fragment factor
    uint sfragndx = sfragfctxndx - sfragfct * nmaxsubfrags;//fragment length index
    uint qryndx = blockIdx.z;//query serial number
    //cache for the cross-covarinace matrix and related data: 
    //no bank conflicts as long as inner-most dim is odd
    enum {neffds = twmvEndOfCCDataExt,//effective number of fields
        smidim = neffds+1};
    __shared__ float ccmCache[
        smidim * CUS1_TBINITSP_COMPLETEREFINE_XDIM + twmvEndOfCCDataExt * 2 + nTTranformMatrix];
//     __shared__ float ccmLast[twmvEndOfCCDataExt];
//     __shared__ float tfmCache[twmvEndOfCCDataExt];//twmvEndOfCCDataExt>nTTranformMatrix
    float* ccmLast = ccmCache + smidim * CUS1_TBINITSP_COMPLETEREFINE_XDIM;
    float* tfmCache = ccmLast + twmvEndOfCCDataExt;//twmvEndOfCCDataExt>nTTranformMatrix
    float* tfmBest = tfmCache + twmvEndOfCCDataExt;//of size nTTranformMatrix

    int qrylen, dbstrlen;//query and reference lengths
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;
    int qrypos, rfnpos;
    int sfragpos, fraglen;


//     //NOTE: no convergence check as this refinement executes once in the beginning
//     if(threadIdx.x == 0) {
//         uint mloc = ((qryndx * maxnsteps + 0/*sfragfctxndx*/) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
//         ccmCache[6] = wrkmemaux[mloc + dbstrndx];
//     }
// 
//     __syncthreads();
// 
//     if(ccmCache[6])
//         //DP or finding rotation matrix converged already; 
//         //(NOTE:any type of convergence applies);
//         //all threads in the block exit;
//         return;
// 
//     //NOTE: no sync as long ccmCache cell for convergence is not overwritten;

    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse ccmCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(dbstrndx, (int*)ccmCache);
        GetQueryLenDst(qryndx, (int*)ccmCache + 2);
    }

    if(threadIdx.x == tawmvQRYpos + 8 || threadIdx.x == tawmvRFNpos + 8) {
        //NOTE: reuse ccmCache to read positions;
        //NOTE: written at sfragfct==0:
        uint mloc = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        ccmCache[threadIdx.x] = wrkmemaux[mloc + (threadIdx.x-8) * ndbCstrs + dbstrndx];
    }

    __syncthreads();


    dbstrlen = ((int*)ccmCache)[0]; dbstrdst = ((int*)ccmCache)[1];
    qrylen = ((int*)ccmCache)[2]; qrydst = ((int*)ccmCache)[3];
    qrypos = ccmCache[tawmvQRYpos+8]; rfnpos = ccmCache[tawmvRFNpos+8];
    sfragpos = sfragfct * sfragstep;

    __syncthreads();


    if(qrylen <= qrypos || dbstrlen <= rfnpos)
        return;//all threads in the block exit

    fraglen = GetFragLength(qrylen, dbstrlen, qrypos, rfnpos, sfragndx);
    if(fraglen < 1) 
        return;//all threads in the block exit

    if(qrylen + sfragstep <= qrypos + sfragpos + fraglen ||
       dbstrlen + sfragstep <= rfnpos + sfragpos + fraglen)
        return;//all threads in the block exit


    //threshold calculated for the original lengths
    const float d0 = GetD0(qrylen, dbstrlen);
    const float d02 = SQRD(d0);
    const float d82 = GetD82(qrylen, dbstrlen);
    float dst32 = CP_LARGEDST;
    float best = 0.0f;//best score obtained

    CalcCCMatrices64Refined_Complete<smidim,neffds>(
        qrydst, dbstrdst, fraglen,
        qrylen, dbstrlen,  qrypos + sfragpos, rfnpos + sfragpos,
        ccmCache);


    for(int cit = 0; cit < nmaxconvit + 2; cit++)
    {
        if(0 < cit) {
            CalcCCMatrices64RefinedExtended_Complete<smidim,neffds>(
                (cit < 2)? READCNST_CALC: READCNST_CALC2,
                qryndx, ndbCposs, maxnsteps, sfragfctxndx, qrydst, dbstrdst,
                qrylen, dbstrlen, qrypos, rfnpos,  d0, dst32,
                tmpdpdiagbuffers, ccmCache);

            CheckConvergence64Refined_Complete(ccmCache, ccmLast);
            if(ccmLast[0]) break;//converged
            __syncthreads();//prevent overwriting ccmLast[0]
        }

        //NOTE: synced above and below before ccmCache gets updated;
        if(ccmCache[twmvNalnposs] < 1.0f) break;

        SaveCCMData_Complete(ccmCache, tfmCache, ccmLast);
        //NOTE: tfmCache updated by the first warp; 
        //NOTE: CalcTfmMatrices_Complete uses only the first warp;
        //NOTE: ccmLast not used until the first syncthreads below;
        __syncwarp();

        CalcTfmMatrices_Complete<TFM_DINV>(tfmCache, qrylen, dbstrlen);
        //all threads synced and see the tfm

        CalcScoresUnrlRefined_Complete(
            (cit < 1)? READCNST_CALC: READCNST_CALC2,
            qryndx, ndbCposs, maxnsteps, sfragfctxndx, qrydst, dbstrdst,
            qrylen, dbstrlen, qrypos, rfnpos,  d0, d02, d82,
            tmpdpdiagbuffers, tfmCache, ccmCache+1);

        //distance threshold for at least three aligned pairs:
        dst32 = ccmCache[2];

        //NOTE: no sync inside:
        SaveLocalBestScoreAndTM(best, ccmCache[1]/*score*/, tfmCache, tfmBest);

        //sync all threads to see dst32 (and prevent overwriting the cache):
        __syncthreads();
    }

    //NOTE: synced either after the last cit or convergence check:
    SaveBestScoreAndTM_Complete<WRITEFRAGINFO,false/*CONDITIONAL*/>(
        best,  qryndx, dbstrndx, ndbCstrs, 
        maxnsteps, sfragfctxndx, sfragndx, sfragpos,
        tfmBest, wrkmemtmibest, wrkmemaux);
}

// -------------------------------------------------------------------------
// Instantiations
// 
#define INSTANTIATE_FragmentBasedAlignmentRefinement(tpWRITEFRAGINFO,tpTFM_DINV) \
    template __global__ void FragmentBasedAlignmentRefinement<tpWRITEFRAGINFO,tpTFM_DINV>( \
        const int nmaxconvit, /*const uint nqystrs,*/ \
        const uint ndbCstrs, const uint ndbCposs, \
        const uint nmaxsubfrags, const uint maxnsteps, \
        const int sfragstep, const int maxalnmax, \
        float* __restrict__ tmpdpdiagbuffers, \
        float* __restrict__ wrkmemtmibest, \
        float* __restrict__ wrkmemaux);

INSTANTIATE_FragmentBasedAlignmentRefinement(false/* true */,false);
INSTANTIATE_FragmentBasedAlignmentRefinement(false/* true */,true);

// =========================================================================
