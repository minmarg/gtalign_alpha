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
#include "covariance_complete.cuh"

// =========================================================================
// FindGaplessAlignedFragment: search for suboptimal superposition and 
// identify fragments for further refinement;
// TFM_DINV, use doubly inverted transformation matrices under suitable conditions;
// nqystrs, total number of query structures in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of reference positions in the chunk;
// maxnsteps, total number of steps that should be performed for each reference structure;
// arg1, arg2, arg3, arguments for length and position functors;
// tmpdpdiagbuffers, temporary diagonal buffers for positional scores;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// 
template<bool TFM_DINV>
__global__ 
void FindGaplessAlignedFragment(
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int arg1,//n1
    const int arg2,//step
    const int arg3,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux)
{
    uint dbstrndx = blockIdx.x;//reference serial number
    uint sfragfct = blockIdx.y;//fragment factor
    uint qryndx = blockIdx.z;//query serial number
    //cache for the cross-covarinace matrix and related data: 
    //no bank conflicts as long as inner-most dim is odd
    enum {neffds = twmvEndOfCCDataExt,//effective number of fields
        smidim = neffds+1};
    __shared__ float ccmCache[
        smidim * CUS1_TBINITSP_COMPLETEREFINE_XDIM + twmvEndOfCCDataExt];
//     __shared__ float tfmCache[twmvEndOfCCDataExt];//twmvEndOfCCDataExt>nTTranformMatrix
    float* tfmCache = ccmCache + smidim * CUS1_TBINITSP_COMPLETEREFINE_XDIM;

    int qrylen, dbstrlen;//query and reference lengths
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;
    int qrypos, rfnpos;//const


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


    GetQryRfnPos(0/*depth; unused*/, qrypos, rfnpos,  qrylen, dbstrlen, sfragfct, arg1, arg2, arg3);

    if(qrylen <= qrypos || dbstrlen <= rfnpos)
        return;//all threads in the block exit

    if(PositionsOutofBounds(qrylen, dbstrlen, qrypos, rfnpos, arg1, arg2, arg3))
        return;//all threads in the block exit


    const float nalnposs = (float)GetNAlnPoss(qrylen, dbstrlen, qrypos, rfnpos, arg1, arg2, arg3);

    //threshold calculated for the original lengths
    const float d0 = GetD0(qrylen, dbstrlen);
    const float d02 = SQRD(d0);
    float dst32 = CP_LARGEDST;
    float best = 0.0f;//best score obtained


    for(;;)
    {
        CalcCCMatrices64_Complete<smidim,neffds>(
            qrydst, dbstrdst, nalnposs, 
            qrylen, dbstrlen, qrypos, rfnpos, ccmCache);

        //NOTE: synced above and below before ccmCache gets updated;
        if(ccmCache[twmvNalnposs] < 1.0f) break;//all exit

        CopyCCMDataToTFM_Complete(ccmCache, tfmCache);
        //NOTE: tfmCache updated by the first warp; 
        //NOTE: CalcTfmMatrices_Complete uses only the first warp;
        __syncwarp();

        CalcTfmMatrices_Complete<TFM_DINV>(tfmCache, qrylen, dbstrlen);
        //all threads synced and see the tfm


        CalcScoresUnrl_Complete<SAVEPOS_SAVE>(
            READCNST_CALC,
            qryndx, ndbCposs, maxnsteps, sfragfct, qrydst, dbstrdst,
            nalnposs, qrylen, dbstrlen, qrypos, rfnpos,  d0, d02,
            tmpdpdiagbuffers, tfmCache, ccmCache+1);

        //distance threshold for at least three aligned pairs:
        dst32 = ccmCache[2];

        if(best < ccmCache[1]) best = ccmCache[1];//score
        //sync all threads to see dst32 (and prevent overwriting ccmCache):
        __syncthreads();

        CalcCCMatrices64Extended_Complete<smidim,neffds>(
            qryndx, ndbCposs, maxnsteps, sfragfct, qrydst, dbstrdst,
            qrylen, dbstrlen, qrypos, rfnpos, dst32,
            tmpdpdiagbuffers, ccmCache);

        //NOTE: synced above and below before ccmCache gets updated;
        if(ccmCache[twmvNalnposs] < 1.0f) break;//all exit
        if(ccmCache[twmvNalnposs] == nalnposs) break;//all exit

        CopyCCMDataToTFM_Complete(ccmCache, tfmCache);
        __syncwarp();

        CalcTfmMatrices_Complete<TFM_DINV>(tfmCache, qrylen, dbstrlen);
        //all threads synced and see the tfm


        CalcScoresUnrl_Complete<SAVEPOS_SAVE>(
            READCNST_CALC2,
            qryndx, ndbCposs, maxnsteps, sfragfct, qrydst, dbstrdst,
            nalnposs, qrylen, dbstrlen, qrypos, rfnpos,  d0, d02,
            tmpdpdiagbuffers, tfmCache, ccmCache+1);

        //distance threshold for at least three aligned pairs:
        dst32 = ccmCache[2];

        if(best < ccmCache[1]) best = ccmCache[1];//score
        //sync all threads to see dst32 (and prevent overwriting ccmCache):
        __syncthreads();

        CalcCCMatrices64Extended_Complete<smidim,neffds>(
            qryndx, ndbCposs, maxnsteps, sfragfct, qrydst, dbstrdst,
            qrylen, dbstrlen, qrypos, rfnpos, dst32,
            tmpdpdiagbuffers, ccmCache);

        //NOTE: synced above and below before ccmCache gets updated;
        if(ccmCache[twmvNalnposs] < 1.0f) break;//all exit
        if(ccmCache[twmvNalnposs] == nalnposs) break;//all exit

        CopyCCMDataToTFM_Complete(ccmCache, tfmCache);
        __syncwarp();

        CalcTfmMatrices_Complete<TFM_DINV>(tfmCache, qrylen, dbstrlen);
        //all threads synced and see the tfm

        CalcScoresUnrl_Complete<SAVEPOS_NOSAVE>(
            READCNST_CALC2,
            qryndx, ndbCposs, maxnsteps, sfragfct, qrydst, dbstrdst,
            nalnposs, qrylen, dbstrlen, qrypos, rfnpos,  d0, d02,
            tmpdpdiagbuffers, tfmCache, ccmCache+1);

        if(best < ccmCache[1]) best = ccmCache[1];//score
        break;
    }

    //NOTE: synced above:
    SaveBestScoreAndPositions_Complete(
        best,  qryndx, dbstrndx, ndbCstrs, 
        maxnsteps, sfragfct, qrypos, rfnpos, wrkmemaux);
}

// =========================================================================
// Instantiations
// 
#define INSTANTIATE_FindGaplessAlignedFragment(tpTFM_DINV) \
    template __global__ void FindGaplessAlignedFragment<tpTFM_DINV>( \
        const uint ndbCstrs, const uint ndbCposs, const uint maxnsteps, \
        const int arg1/* n1 */,const int arg2/* step */,const int arg3, \
        float* __restrict__ tmpdpdiagbuffers, \
        float* __restrict__ wrkmemaux);

INSTANTIATE_FindGaplessAlignedFragment(false);
INSTANTIATE_FindGaplessAlignedFragment(true);

// -------------------------------------------------------------------------
