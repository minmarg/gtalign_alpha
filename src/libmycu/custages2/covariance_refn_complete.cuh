/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __covariance_refn_complete_cuh__
#define __covariance_refn_complete_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fragment.cuh"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/covariance.cuh"
#include "libmycu/custages/covariance_plus.cuh"
#include "libmycu/custages/covariance_refn.cuh"
#include "libmycu/custages/covariance_dp_refn.cuh"
#include "libmycu/custages/transform.cuh"
#include "libmycu/custages/scoring.cuh"

// =========================================================================
// FragmentBasedAlignmentRefinement: refine alignment and its boundaries 
// within the single kernel's actions to obtain favorable superposition;
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
    const int maxalnmax,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux
);

// =========================================================================
// SaveCCMData_Complete: save cross-covariance data to two additional 
// buffers;
//
__device__ __forceinline__
void SaveCCMData_Complete(
    const float* __restrict__ ccmCache,
    float* __restrict__ tfmCache,
    float* __restrict__ ccmLast)
{
    if(threadIdx.x < twmvEndOfCCDataExt) tfmCache[threadIdx.x] = ccmCache[threadIdx.x];
    //use a different warp
#if (CUS1_TBINITSP_COMPLETEREFINE_XDIM >= 64)
    if(32 <= threadIdx.x && threadIdx.x < twmvEndOfCCDataExt + 32)
        ccmLast[threadIdx.x-32] = ccmCache[threadIdx.x-32];
#else
    if(threadIdx.x < twmvEndOfCCDataExt)
        ccmLast[threadIdx.x] = ccmCache[threadIdx.x];
#endif
}

// -------------------------------------------------------------------------
// CheckConvergence64Refined_Complete: check whether calculating rotation 
// matrices converged by verifying the absolute difference of two latest 
// cross-covariance matrix data between the query and reference structures;
// Complete version;
//
__device__ __forceinline__
void CheckConvergence64Refined_Complete(
    const float* __restrict__ ccmCache,
    float* __restrict__ ccmLast)
{
    //effective number of fields (16):
    enum {neffds = twmvEndOfCCDataExt};
    int fldval = 0;

    if(threadIdx.x < neffds) {
        float dat1 = ccmCache[threadIdx.x];
        float dat2 = ccmLast[threadIdx.x];
        //convergence criterion for all fields: |a-b| / min{|a|,|b|} < epsilon
        if(fabsf(dat1-dat2) < myhdmin(fabsf(dat1), fabsf(dat2)) * RM_CONVEPSILON)
            fldval = 1;
    }

    //NOTE: warp reduction within each section of neffds values in the warp!!
    fldval += __shfl_down_sync(0xffffffff, fldval, 8, neffds);
    fldval += __shfl_down_sync(0xffffffff, fldval, 4, neffds);
    fldval += __shfl_down_sync(0xffffffff, fldval, 2, neffds);
    fldval += __shfl_down_sync(0xffffffff, fldval, 1, neffds);

    //write the convergence flag back to ccmLast:
    if(threadIdx.x == 0) {
        ccmLast[0] = 0.0f;
        if(neffds <= fldval) ccmLast[0] = 1.0f;
    }

    //make all threads in the block see the convergence flag:
    __syncthreads();
}

// -------------------------------------------------------------------------
// CalcCCMatrices64Refined_Complete: calculate cross-covariance matrix 
// between the query and reference structures for refinement, i.e. 
// delineation of suboptimal fragment boundaries;
// Complete version for the refinement of fragment boundaries;
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// qrydst, distances in positions to the beginnings of the query structures;
// dbstrdst, distances in positions to the beginnings of the reference structures;
// fraglen, fragment length;
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// ccmCache, cache for the cross-covarinace matrix and related data;
// 
template<int SMIDIM, int NEFFDS>
__device__ __forceinline__
void CalcCCMatrices64Refined_Complete(
    const uint qrydst,
    const uint dbstrdst,
    const int fraglen,
    int qrylen, int dbstrlen,
    int qrypos, int rfnpos,
    float* __restrict__  ccmCache)
{
    //update positions and assign virtual query and reference lengths:
    UpdateLengths(qrylen, dbstrlen, qrypos, rfnpos, fraglen);

    //initialize cache:
    InitCCMCacheExtended<SMIDIM,0,NEFFDS>(ccmCache);
    
    //no sync as long as each thread works in its own memory space


    //manually unroll along data blocks:
    for(qrypos += threadIdx.x, rfnpos += threadIdx.x;
        qrypos < qrylen && rfnpos < dbstrlen;
        qrypos += blockDim.x, rfnpos += blockDim.x)
    {
        UpdateCCMOneAlnPos<SMIDIM>(//no sync.
            qrydst + qrypos, dbstrdst + rfnpos,
            ccmCache
        );
    }

    //sync now:
    __syncthreads();

    //unroll until reaching warpSize 
    //for(int xdim = (blockDim.x>>1); xdim >= 32; xdim >>= 1) {
    for(int xdim = (CUS1_TBINITSP_COMPLETEREFINE_XDIM>>1); xdim >= 32; xdim >>= 1) {
        if(threadIdx.x < xdim) {
//             #pragma unroll
            for(int i = 0; i < twmvEndOfCCData; i++)
                ccmCache[threadIdx.x * SMIDIM + i] +=
                    ccmCache[(threadIdx.x + xdim) * SMIDIM + i];
        }
        __syncthreads();
    }

    //unroll warp
    if(threadIdx.x < 32) {
        for(int i = 0; i < twmvEndOfCCData; i++) {
            float sum = ccmCache[threadIdx.x * SMIDIM + i];
            sum = mywarpreducesum(sum);
            //write to the first SMEM data slot
            if(threadIdx.x == 0) ccmCache[i] = sum;
        }
    }

    //one thread writes nalnposs
    if(threadIdx.x == 0)
        ccmCache[twmvNalnposs] = fraglen;

    //make all threads in the block see the changes
    __syncthreads();
}

// -------------------------------------------------------------------------
// CalcScoresUnrlRefined_Complete: calculate/reduce UNNORMALIZED scores for 
// obtained superpositions; complete version for fragments refinement; 
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// qryndx, query serial number;
// ndbCposs, total number of reference positions in the chunk;
// maxnsteps, max number of steps to perform for each reference structure;
// sfragfctxndx, current fragment factor x fragment length index, which is <maxnsteps;
// qrydst, distances in positions to the beginnings of the query structures;
// dbstrdst, distances in positions to the beginnings of the reference structures;
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// d0, d02, d82, distance thresholds;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary diagonal buffers reused here for saving positional scores;
// tfmCache, cached transformation matrix;
// scvCache, cache for scores;
//
__device__ __forceinline__
void CalcScoresUnrlRefined_Complete(
    const int READCNST,
    const uint qryndx,
    const uint ndbCposs,
    const uint maxnsteps,
    const uint sfragfctxndx,
    const uint qrydst,
    const uint dbstrdst,
    int qrylen, int dbstrlen,
    const int qrypos, const int rfnpos,
    const float d0, const float d02, const float d82,
    float* __restrict__ tmpdpdiagbuffers,
    const float* __restrict__ tfmCache,
    float* __restrict__  scvCache)
{
    enum {lsmidim = 3};//three minimum distance values
    //NOTE: scvCache assumed to have space for blockDim.x*(lsmidim+1) entries
    float* dstCache = scvCache + blockDim.x;

    //initialize cache
    scvCache[threadIdx.x] = 0.0f;

    //initialize cache of distances:
    #pragma unroll
    for(int i = 0; i < lsmidim; i++)
        dstCache[threadIdx.x * lsmidim + i] = CP_LARGEDST;

    __syncthreads();

    const int mloc = (qryndx * maxnsteps + sfragfctxndx) * ndbCposs;

    //manually unroll along data blocks:
    //pos0, position index starting from 0
    for(int pos0 = threadIdx.x; qrypos + pos0 < qrylen && rfnpos + pos0 < dbstrlen;
        pos0 += blockDim.x)
    {
        //calculated distance is written to to gmem: SAVEPOS_SAVE
        float dst = 
        UpdateOneAlnPosScore<SAVEPOS_SAVE,CHCKDST_CHECK>(//no sync;
            d02, d82,
            qrydst + qrypos + pos0,//query position
            dbstrdst + rfnpos + pos0,//reference position
            mloc + dbstrdst + pos0,//position for scores
            tfmCache,//tfm. mtx.
            scvCache,//score cache
            tmpdpdiagbuffers//scores/dsts written to gmem
        );
        //store three min distance values
#if (DO_FINDD02_DURING_REFINEFRAG == 0)
        if(READCNST == READCNST_CALC)
#endif
            StoreMinDst(dstCache + threadIdx.x * lsmidim, dst);
    }

    //sync now:
    __syncthreads();

    //unroll until reaching warpSize 
    //for(int xdim = (blockDim.x>>1); xdim >= 32; xdim >>= 1) {
    for(int xdim = (CUS1_TBINITSP_COMPLETEREFINE_XDIM>>1); xdim >= 32; xdim >>= 1) {
        int tslot = threadIdx.x * lsmidim;
        if(threadIdx.x < xdim) {
            scvCache[threadIdx.x] += scvCache[threadIdx.x + xdim];
            StoreMinDstSrc(dstCache + tslot, dstCache + tslot + xdim * lsmidim);
        }
        __syncthreads();
    }

    //unroll warp for the score
    if(threadIdx.x < 32/*warpSize*/) {
        float sum = scvCache[threadIdx.x];
        sum = mywarpreducesum(sum);
        //write to the first SMEM data slot
        if(threadIdx.x == 0) scvCache[0] = sum;
    }

    //unroll warp for min distances
#if (DO_FINDD02_DURING_REFINEFRAG == 0)
    if(READCNST == READCNST_CALC)
#endif
    if(threadIdx.x < 32/*warpSize*/) {
        for(int xdim = (32>>1); xdim >= 1; xdim >>= 1) {
            int tslot = threadIdx.x * lsmidim;
            if(threadIdx.x < xdim)
                StoreMinDstSrc(dstCache + tslot, dstCache + tslot + xdim * lsmidim);
            __syncwarp();
        }
    }

    //write the minimum score that ensures at least 3 aligned positions:
    //NOTE: process onced synced (above):
#if (DO_FINDD02_DURING_REFINEFRAG == 0)
    if(READCNST == READCNST_CALC)
#endif
    if(threadIdx.x == 2) {
        float d0s = GetD0s(d0) + ((READCNST == READCNST_CALC2)? 1.0f: -1.0f);
        float d02s = SQRD(d0s);

        float min3 = dstCache[threadIdx.x];

        if(CP_LARGEDST_cmp < min3 || min3 < d02s || 
           GetGplAlnLength(qrylen, dbstrlen, qrypos, rfnpos) <= 3)
            //max number of alignment positions (GetGplAlnLength) <3;
            //use the dfault threshold
            min3 = d02s;
        else {//round the 3rd minimum score according to the below:
            //obtained from (d0s + k*0.5)^2 >= min3 (squared distance)
            min3 = d0s + ceilf((sqrtf(min3) - d0s) * 2.0f) * 0.5f;
            min3 = SQRD(min3);
        }

        //write min3 to scvCache[1] as scvCache[0] is reserved for the score:
        scvCache[1] = min3;
    }

    //make the block's all threads see the reduced score scvCache[0] and 
    //distance scvCache[1]:
    __syncthreads();
}

// -------------------------------------------------------------------------
// SaveLocalBestScoreAndTM: save local best score with transformation matrix;
// best, best score so far;
// score, current score calculated;
// tfmCache, cached transformation matrix;
// tfmBest, locally best transformation matrix;
// 
__device__ __forceinline__
void SaveLocalBestScoreAndTM(
    float& best,
    const float score,
    const float* __restrict__ tfmCache,
    float* __restrict__ tfmBest)
{
    const bool bwrite = (best < score);

    if(bwrite) best = score;

    //save transformation matrix
    if(bwrite && threadIdx.x < nTTranformMatrix)
        tfmBest[threadIdx.x] = tfmCache[threadIdx.x];
}

// -------------------------------------------------------------------------
// CalcCCMatrices64RefinedExtended_Complete: calculate cross-covariance 
// matrix between the query and reference structures based on aligned 
// positions within given distance;
// Complete version for the refinement of fragment boundaries;
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// READCNST, flag indicating the stage of this local processing;
// qryndx, query serial number;
// ndbCposs, total number of reference positions in the chunk;
// maxnsteps, max number of steps to perform for each reference structure;
// sfragfctxndx, current fragment factor x fragment length index, which is <maxnsteps;
// qrydst, distances in positions to the beginnings of the query structures;
// dbstrdst, distances in positions to the beginnings of the reference structures;
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// d0, distance threshold;
// dst32, squared distance threshold at which at least three aligned pairs are observed;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary diagonal buffers reused for reading positional scores;
// ccmCache, cache for the cross-covariance matrix and related data;
// 
template<int SMIDIM, int NEFFDS>
__device__ __forceinline__
void CalcCCMatrices64RefinedExtended_Complete(
    const int READCNST,
    const uint qryndx,
    const uint ndbCposs,
    const uint maxnsteps,
    const uint sfragfctxndx,
    const uint qrydst,
    const uint dbstrdst,
    const int qrylen, const int dbstrlen,
    const int qrypos, const int rfnpos,
    const float d0, const float dst32,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__  ccmCache)
{
    InitCCMCacheExtended<SMIDIM,0,NEFFDS>(ccmCache);

    //no sync as long as each thread works in its own memory space

    float d02s = dst32;

#if (DO_FINDD02_DURING_REFINEFRAG == 0)
    if(READCNST != READCNST_CALC) {
        float d0s = GetD0s(d0) + 1.0f;
        d02s = SQRD(d0s);
    }
#endif

    const int mloc = (qryndx * maxnsteps + sfragfctxndx) * ndbCposs;

    //manually unroll along data blocks:
    //pos0, position index starting from 0
    for(int pos0 = threadIdx.x; qrypos + pos0 < qrylen && rfnpos + pos0 < dbstrlen;
        pos0 += blockDim.x)
    {
        UpdateCCMOneAlnPosExtended<SMIDIM>(//no sync;
            d02s,
            qrydst + qrypos + pos0,//query position
            dbstrdst + rfnpos + pos0,//reference position
            mloc + dbstrdst + pos0,//position for scores
            tmpdpdiagbuffers,//scores/dsts
            ccmCache//reduction output
        );
    }

    //sync now:
    __syncthreads();

    //unroll until reaching warpSize 
    //for(int xdim = (blockDim.x>>1); xdim >= 32; xdim >>= 1) {
    for(int xdim = (CUS1_TBINITSP_COMPLETEREFINE_XDIM>>1); xdim >= 32; xdim >>= 1) {
        if(threadIdx.x < xdim) {
//             #pragma unroll
            for(int i = 0; i < NEFFDS; i++)
                ccmCache[threadIdx.x * SMIDIM + i] +=
                    ccmCache[(threadIdx.x + xdim) * SMIDIM + i];
        }
        __syncthreads();
    }

    //unroll warp
    if(threadIdx.x < 32) {
        for(int i = 0; i < NEFFDS; i++) {
            float sum = ccmCache[threadIdx.x * SMIDIM + i];
            sum = mywarpreducesum(sum);
            //write to the first SMEM data slot
            if(threadIdx.x == 0) ccmCache[i] = sum;
        }
    }

    //make all threads in the block see the changes
    __syncthreads();
}

// -------------------------------------------------------------------------
// SaveBestScoreAndTM_Complete: complete version of saving the best 
// score along with transformation;
// save fragment indices and starting positions too;
// WRITEFRAGINFO, template parameter, flag of writing fragment attributes;
// CONDITIONAL, template parameter, flag of writing the score if it's greater at the same location;
// best, best score so far;
// qryndx, query serial number;
// dbstrndx, reference serial number;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps to perform for each reference structure;
// sfragfctxndx, current fragment factor x fragment length index, which is <maxnsteps;
// sfragndx, index defining fragment length;
// sfragpos, starting position within fragment;
// NOTE: memory pointers should be aligned!
// tfmCache, cached transformation matrix;
// wrkmemtmibest, working memory for iteration-best transformation matrices;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// 
template<bool WRITEFRAGINFO, bool CONDITIONAL>
__device__ __forceinline__
void SaveBestScoreAndTM_Complete(
    float best,
    const uint qryndx,
    const uint dbstrndx,
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint sfragfctxndx,
    const uint sfragndx,
    const int sfragpos,
    const float* __restrict__ tfmCache,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux)
{
    if(best <= 0.0f) return;

    float currentbest = 0.0f;

    //save best score
    if(threadIdx.x == 0)
    {
        const uint mloc = ((qryndx * maxnsteps + sfragfctxndx) * nTAuxWorkingMemoryVars) * ndbCstrs;

        if(CONDITIONAL) 
            currentbest = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx];

        if(currentbest < best) {
            wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx] = best;
            if(WRITEFRAGINFO) {
                wrkmemaux[mloc + tawmvSubFragNdxCurrent * ndbCstrs + dbstrndx] = sfragndx;
                wrkmemaux[mloc + tawmvSubFragPosCurrent * ndbCstrs + dbstrndx] = sfragpos;
            }
        }
    }

    currentbest = __shfl_sync(0xffffffff, currentbest, 0/*srcLane*/);

    //save transformation matrix
    if(currentbest < best && threadIdx.x < nTTranformMatrix) {
        const uint mloc = ((qryndx * maxnsteps + sfragfctxndx) * ndbCstrs + dbstrndx) * nTTranformMatrix;
        wrkmemtmibest[mloc + threadIdx.x] = tfmCache[threadIdx.x];
    }
}

// -------------------------------------------------------------------------

#endif//__covariance_refn_complete_cuh__
