/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __covariance_dp_refn_complete_cuh__
#define __covariance_dp_refn_complete_cuh__

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
// FragmentBasedDPAlignmentRefinement: refine alignment and its boundaries 
// within the single kernel's actions to obtain favorable superposition;
//
template<bool WRITEFRAGINFO, bool CONDITIONAL, bool TFM_DINV>
__global__ 
void FragmentBasedDPAlignmentRefinement(
    const bool readlocalconv,
    const int nmaxconvit,
//     const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint nmaxsubfrags,
    const uint maxnsteps,
    const int sfragstep,
    const int maxalnmax,
    const float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux
);

// =========================================================================
// GetSubfragFctAndNdx: get the fragment factor sfragfct and fragment length 
// index sfragndx given the total sum of fragment factors sfragfctxndx;
//
__device__ __forceinline__
void GetSubfragFctAndNdx(
    uint& sfragfct,
    uint& sfragndx,
    const uint sfragfctxndx,
    const uint nmaxsubfrags,
    const int sfragstep,
    const int maxalnmax)
{
    uint nlocsteps = 0;

    for(sfragndx = 0, sfragfct = sfragfctxndx; sfragndx < nmaxsubfrags; sfragndx++)
    {   //maximum alignment length for this frag index
        int maxfraglen = GetFragLength(maxalnmax, maxalnmax, 0, 0, sfragndx);
        if(maxfraglen < 1) break;
        nlocsteps += GetMaxNFragSteps(maxalnmax, sfragstep, maxfraglen);
        if(sfragfctxndx < nlocsteps) {
            sfragfct = nlocsteps - sfragfctxndx - 1;
            break;
        }
    }
}

// -------------------------------------------------------------------------
// CalcCCMatrices64_DPRefined_Complete: calculate cross-covariance matrix 
// between the query and reference structures for refinement, i.e. 
// delineation of suboptimal fragment boundaries;
// Complete version for the refinement of fragment boundaries obtained as a 
// result of the application of DP;
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// qryndx, query serial number;
// ndbCposs, total number of reference positions in the chunk;
// dbxpad, #pad positions along the dimension of reference structures;
// maxnsteps, max number of steps to perform for each reference structure;
// sfragfctxndx, current fragment factor x fragment length index, which is <maxnsteps;
// dbstrdst, distances in positions to the beginnings of the reference structures;
// fraglen, fragment length;
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// NOTE: memory pointers should be aligned!
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// ccmCache, cache for the cross-covarinace matrix and related data;
// 
template<int SMIDIM, int NEFFDS>
__device__ __forceinline__
void CalcCCMatrices64_DPRefined_Complete(
    const uint qryndx,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const uint /*sfragfctxndx*/,
    const uint dbstrdst,
    const int fraglen,
    int qrylen, int dbstrlen,
    int qrypos, int rfnpos,
    const float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__  ccmCache)
{
    //qrylen == dbstrlen; reuse qrylen for original alignment length;
    //update positions and assign virtual query and reference lengths:
    UpdateLengths(dbstrlen/*qrylen*/, dbstrlen, qrypos, rfnpos, fraglen);

    //initialize cache:
    InitCCMCacheExtended<SMIDIM,0,NEFFDS>(ccmCache);
    
    //no sync as long as each thread works in its own memory space


    const int dblen = ndbCposs + dbxpad;
    //offset to the beginning of the data along the y axis wrt query qryndx: 
    const int yofff = (qryndx * maxnsteps + 0/*sfragfctxndx*/) * dblen * nTDPAlignedPoss;

    //manually unroll along data blocks:
    for(int relpos = rfnpos + threadIdx.x; relpos < dbstrlen; relpos += blockDim.x)
    {
        //starting position in tmpdpalnpossbuffer for a pair:
        //NOTE: qrypos == rfnpos as well as lengths: use qrylen as the 
        //NOTE: original alignment length here;
        //NOTE: alignment written in reverse order:
        int pos = yofff + dbstrdst + qrylen-1 - (relpos);
        UpdateCCMOneAlnPos_DPRefined<SMIDIM>(//no sync.
            pos, dblen,
            tmpdpalnpossbuffer,
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
// CalcScoresUnrl_DPRefined: calculate/reduce UNNORMALIZED scores for 
// obtained superpositions; complete version for the refinement of fragments 
// obtained by DP; 
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// CHCKDST, template parameter to request accumulating scores within the 
// given threshold distance only;
// qryndx, query serial number;
// ndbCposs, total number of reference positions in the chunk;
// dbxpad, #pad positions along the dimension of reference structures;
// maxnsteps, max number of steps to perform for each reference structure;
// sfragfctxndx, current fragment factor x fragment length index, which is <maxnsteps;
// dbstrdst, distances in positions to the beginnings of the reference structures;
// qrylen, dbstrlen, query and reference lengths;
// qrylenorg, dbstrlenorg, original query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// d0, d02, d82, distance thresholds;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary diagonal buffers reused here for saving positional scores;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// tfmCache, cached transformation matrix;
// scvCache, cache for scores;
//
template<int CHCKDST = CHCKDST_CHECK>
__device__ __forceinline__
void CalcScoresUnrl_DPRefined_Complete(
    const int READCNST,
    const uint qryndx,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const uint sfragfctxndx,
    const uint dbstrdst,
    int qrylen, int dbstrlen,
    const int qrypos, const int rfnpos,
    const float d0, const float d02, const float d82,
    float* __restrict__ tmpdpdiagbuffers,
    const float* __restrict__ tmpdpalnpossbuffer,
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

    const int dblen = ndbCposs + dbxpad;
    //offset to the beginning of the data along the y axis wrt query qryndx: 
    const int yofff = (qryndx * maxnsteps + 0/*sfragfctxndx*/) * dblen * nTDPAlignedPoss;
    const int mloc = (qryndx * maxnsteps + sfragfctxndx) * ndbCposs;

    //manually unroll along data blocks:
    //pos0, position index starting from 0
    for(int pos0 = threadIdx.x; qrypos + pos0 < qrylen && rfnpos + pos0 < dbstrlen;
        pos0 += blockDim.x)
    {
        //starting position in tmpdpalnpossbuffer for a pair:
        //NOTE: aligned coordinates in tmpdpalnpossbuffer are in the reverse order!
        //NOTE: qrypos == rfnpos as well as qrylen == dbstrlen here
        int dppos = yofff + dbstrdst + dbstrlen-1 - (rfnpos + pos0);
        //calculated distance is written to to gmem: SAVEPOS_SAVE
        float dst = 
        UpdateOneAlnPosScore_DPRefined<SAVEPOS_SAVE,CHCKDST>(//no sync;
            d02, d82,
            dppos, dblen,
            mloc + dbstrdst + pos0,//position for scores
            tmpdpalnpossbuffer,//coordinates
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

        //write min3 to scvCache[1] as scvCache[1] is reserved for the score:
        scvCache[1] = min3;
    }

    //make the block's all threads see the reduced score scvCache[0] and 
    //distance scvCache[1]:
    __syncthreads();
}

// -------------------------------------------------------------------------
// CalcCCMatrices64_DPRefinedExtended_Complete: calculate cross-covariance 
// matrix between the query and reference structures based on aligned 
// positions within given distance;
// Complete version for the refinement of fragment boundaries and 
// superposition obtained as a result of the application of DP;
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// READCNST, flag indicating the stage of this local processing;
// qryndx, query serial number;
// ndbCposs, total number of reference positions in the chunk;
// dbxpad, #pad positions along the dimension of reference structures;
// maxnsteps, max number of steps to perform for each reference structure;
// sfragfctxndx, current fragment factor x fragment length index, which is <maxnsteps;
// dbstrdst, distances in positions to the beginnings of the reference structures;
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// d0, distance threshold;
// dst32, squared distance threshold at which at least three aligned pairs are observed;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary diagonal buffers reused for reading positional scores;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// ccmCache, cache for the cross-covariance matrix and related data;
// 
template<int SMIDIM, int NEFFDS>
__device__ __forceinline__
void CalcCCMatrices64_DPRefinedExtended_Complete(
    const int READCNST,
    const uint qryndx,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const uint sfragfctxndx,
    const uint dbstrdst,
    const int qrylen, const int dbstrlen,
    const int qrypos, const int rfnpos,
    const float d0, const float dst32,
    const float* __restrict__ tmpdpdiagbuffers,
    const float* __restrict__ tmpdpalnpossbuffer,
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


    const int dblen = ndbCposs + dbxpad;
    //offset to the beginning of the data along the y axis wrt query qryndx: 
    const int yofff = (qryndx * maxnsteps + 0/*sfragfctxndx*/) * dblen * nTDPAlignedPoss;
    const int mloc = (qryndx * maxnsteps + sfragfctxndx) * ndbCposs;

    //manually unroll along data blocks:
    //pos0, position index starting from 0
    for(int pos0 = threadIdx.x; qrypos + pos0 < qrylen && rfnpos + pos0 < dbstrlen;
        pos0 += blockDim.x)
    {
        //starting position in tmpdpalnpossbuffer for a pair:
        //NOTE: aligned coordinates in tmpdpalnpossbuffer are in the reverse order!
        //NOTE: qrypos == rfnpos as well as qrylen == dbstrlen here
        int dppos = yofff + dbstrdst + dbstrlen-1 - (rfnpos + pos0);
        UpdateCCMOneAlnPos_DPExtended<SMIDIM>(//no sync;
            d02s,
            dppos, dblen,
            mloc + dbstrdst + pos0,//position for scores
            tmpdpalnpossbuffer,//coordinates
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
#endif//__covariance_dp_refn_complete_cuh__
