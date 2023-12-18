/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __covariance_complete_cuh__
#define __covariance_complete_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fragment.cuh"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/covariance.cuh"
#include "libmycu/custages/covariance_plus.cuh"
#include "libmycu/custages/scoring.cuh"

// =========================================================================
// FindGaplessAlignedFragment: search for suboptimal superposition and 
// identify fragments for further refinement;
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
    float* __restrict__ wrkmemaux
);

// =========================================================================
// SaveCCMData_Complete: save cross-covariance data to two additional 
// buffers;
//
__device__ __forceinline__
void CopyCCMDataToTFM_Complete(
    const float* __restrict__ ccmCache,
    float* __restrict__ tfmCache)
{
    if(threadIdx.x < twmvEndOfCCDataExt)
        tfmCache[threadIdx.x] = ccmCache[threadIdx.x];
}

// -------------------------------------------------------------------------
// CalcCCMatrices64_Complete: calculate cross-covariance matrix 
// between the query and reference structures for a given gapless fragment;
// Complete version for identifying fragment boundaries;
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// qrydst, distances in positions to the beginnings of the query structures;
// dbstrdst, distances in positions to the beginnings of the reference structures;
// nalnposs, #aligned positions (fragment length);
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// ccmCache, cache for the cross-covarinace matrix and related data;
// 
template<int SMIDIM, int NEFFDS>
__device__ __forceinline__
void CalcCCMatrices64_Complete(
    const uint qrydst,
    const uint dbstrdst,
    const float nalnposs,
    int qrylen, int dbstrlen,
    int qrypos, int rfnpos,
    float* __restrict__  ccmCache)
{
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
        ccmCache[twmvNalnposs] = nalnposs;

    //make all threads in the block see the changes
    __syncthreads();
}

// -------------------------------------------------------------------------
// CalcTfmMatricesHelper_Complete: thread block calculates a tranformation 
// matrix;
// ccmCache, cache for the cross-covariance matrix and reuse;
// NOTE: Based on the original Kabsch algorithm:
/*
c**** CALCULATES A BEST ROTATION & TRANSLATION BETWEEN TWO VECTOR SETS
c**** SUCH THAT U*X+T IS THE CLOSEST APPROXIMATION TO Y.
c**** THE CALCULATED BEST SUPERPOSITION MAY NOT BE UNIQUE AS INDICATED
c**** BY A RESULT VALUE IER=-1. HOWEVER IT IS GARANTIED THAT WITHIN
c**** NUMERICAL TOLERANCES NO OTHER SUPERPOSITION EXISTS GIVING A
c**** SMALLER VALUE FOR RMS.
c**** THIS VERSION OF THE ALGORITHM IS OPTIMIZED FOR THREE-DIMENSIONAL
c**** REAL VECTOR SPACE.
c**** USE OF THIS ROUTINE IS RESTRICTED TO NON-PROFIT ACADEMIC
c**** APPLICATIONS.
c**** PLEASE REPORT ERRORS TO
c**** PROGRAMMER:  W.KABSCH   MAX-PLANCK-INSTITUTE FOR MEDICAL RESEARCH
c        JAHNSTRASSE 29, 6900 HEIDELBERG, FRG.
c**** REFERENCES:  W.KABSCH   ACTA CRYST.(1978).A34,827-828
c           W.KABSCH ACTA CRYST.(1976).A32,922-923
c
c  W    - W(M) IS WEIGHT FOR ATOM PAIR  # M           (GIVEN)
c  X    - X(I,M) ARE COORDINATES OF ATOM # M IN SET X       (GIVEN)
c  Y    - Y(I,M) ARE COORDINATES OF ATOM # M IN SET Y       (GIVEN)
c  N    - N IS number OF ATOM PAIRS             (GIVEN)
c  MODE  - 0:CALCULATE RMS ONLY              (GIVEN)
c      1:CALCULATE RMS,U,T   (TAKES LONGER)
c  RMS   - SUM OF W*(UX+T-Y)**2 OVER ALL ATOM PAIRS        (RESULT)
c  U    - U(I,J) IS   ROTATION  MATRIX FOR BEST SUPERPOSITION  (RESULT)
c  T    - T(I)   IS TRANSLATION VECTOR FOR BEST SUPERPOSITION  (RESULT)
c  IER   - 0: A UNIQUE OPTIMAL SUPERPOSITION HAS BEEN DETERMINED(RESULT)
c     -1: SUPERPOSITION IS NOT UNIQUE BUT OPTIMAL
c     -2: NO RESULT OBTAINED BECAUSE OF NEGATIVE WEIGHTS W
c      OR ALL WEIGHTS EQUAL TO ZERO.
c
c-----------------------------------------------------------------------
*/
__device__ __forceinline__
void CalcTfmMatricesHelper_Complete(
    float* __restrict__ ccmCache, float nalnposs)
{
    __shared__ float aCache[twmvEndOfCCMtx];
    __shared__ float rr[6];//rrCache[6];

    //initialize matrix a;
    //NOTE: only valid when indices start from 0
    if(threadIdx.x == 0) RotMtxToIdentity(aCache);

    //calculate query center vector in advance
    if(twmvCVq_0 <= threadIdx.x && threadIdx.x <= twmvCVq_2)
        ccmCache[threadIdx.x] = fdividef(ccmCache[threadIdx.x], nalnposs);

    __syncwarp();

    if(threadIdx.x == 0) CalcRmatrix(ccmCache);

    //calculate reference center vector now
    if(twmvCVr_0 <= threadIdx.x && threadIdx.x <= twmvCVr_2)
        ccmCache[threadIdx.x] = fdividef(ccmCache[threadIdx.x], nalnposs);

    __syncwarp();


    //NOTE: scale correlation matrix to enable rotation matrix 
    // calculation in single precision without overflow and underflow:
    //ScaleRmatrix(ccmCache);
    float scale = GetRScale(ccmCache);
    if(threadIdx.x < twmvEndOfCCMtx)
        ccmCache[threadIdx.x] = fdividef(ccmCache[threadIdx.x], scale);

    __syncwarp();


    //calculate determinant
    float det = CalcDet(ccmCache);

    //calculate the product transposed(R) * R
    if(threadIdx.x == 0) CalcRTR(ccmCache, rr);

    __syncwarp();

    //Kabsch:
    //eigenvalues: form characteristic cubic x**3-3*spur*x**2+3*cof*x-det=0
    float spur = (rr[0] + rr[2] + rr[5]) * oneTHIRDf;
    float cof = (((((rr[2] * rr[5] - SQRD(rr[4])) + rr[0] * rr[5]) -
                SQRD(rr[3])) + rr[0] * rr[2]) -
                SQRD(rr[1])) * oneTHIRDf;

    bool abok = (spur > 0.0f);

    if(abok && threadIdx.x == 0)
    {   //Kabsch:
        //reduce cubic to standard form y**3-3hy+2g=0 by putting x=y+spur

        //Kabsch: solve cubic: roots are e[0],e[1],e[2] in decreasing order
        //Kabsch: handle special case of 3 identical roots
        float e0, e1, e2;
        if(SolveCubic(det, spur, cof, e0, e1, e2))
        {
            //Kabsch: eigenvectors
            //almost always this branch gets executed
            CalcPartialA_Reg<0>(e0, rr, aCache);
            CalcPartialA_Reg<2>(e2, rr, aCache);
            abok = CalcCompleteA(e0, e1, e2, aCache);
        }
        if(abok) {
            //Kabsch: rotation matrix
            abok = CalcRotMtx(aCache, ccmCache);
        }
    }

    if(!abok && threadIdx.x == 0) RotMtxToIdentity(ccmCache);

    //Kabsch: translation vector
    //NOTE: scaling translation vector would be needed if the data 
    // vectors were scaled previously so that transformation is 
    // applied in the original coordinate space
    if(threadIdx.x == 0) CalcTrlVec(ccmCache);
}

// CalcTfmMatrices_Complete: calculate a tranformation matrix;
// DOUBLY_INVERTED, change places of query and reference sums so that a
// tranformation matrix is calculated wrt the query;
// then, revert back the transformation matrix to obtain it wrt the reference
// again; NOTE: for numerical stability (index) and symmetric results;
//
template<bool DOUBLY_INVERTED = false>
__device__ __forceinline__
void CalcTfmMatrices_Complete(
    float* __restrict__ ccmCache, int qrylen, int dbstrlen)
{
    //#positions used to calculate cross-covarinaces:
    float nalnposs = ccmCache[twmvNalnposs];

    if(nalnposs <= 0.0f)
        return;//all threads exit

    if(DOUBLY_INVERTED && (qrylen < dbstrlen)) {
        TransposeRotMtx(ccmCache);
        myhdswap(ccmCache[twmvCVq_0], ccmCache[twmvCVr_0]);
        myhdswap(ccmCache[twmvCVq_1], ccmCache[twmvCVr_1]);
        myhdswap(ccmCache[twmvCVq_2], ccmCache[twmvCVr_2]);
        __syncthreads();
    }

    CalcTfmMatricesHelper_Complete(ccmCache, nalnposs);
    __syncthreads();

    if(DOUBLY_INVERTED && (qrylen < dbstrlen)) {
        InvertRotMtx(ccmCache);
        InvertTrlVec(ccmCache);
        __syncthreads();
    }
}

// -------------------------------------------------------------------------
// CalcScoresUnrl_Complete: calculate/reduce UNNORMALIZED scores for 
// obtained superpositions; complete version for fragment identification; 
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// SAVEPOS, template parameter to request saving distances;
// qryndx, query serial number;
// ndbCposs, total number of reference positions in the chunk;
// maxnsteps, max number of steps to perform for each reference structure;
// sfragfct, current fragment factor;
// qrydst, distances in positions to the beginnings of the query structures;
// dbstrdst, distances in positions to the beginnings of the reference structures;
// maxnalnposs, #originally aligned positions (original fragment length);
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// d0, d02, distance thresholds;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary diagonal buffers reused here for saving positional scores/distances;
// tfmCache, cached transformation matrix;
// scvCache, cache for scores;
//
template<int SAVEPOS>
__device__ __forceinline__
void CalcScoresUnrl_Complete(
    const int READCNST,
    const uint qryndx,
    const uint ndbCposs,
    const uint maxnsteps,
    const uint sfragfct,
    const uint qrydst,
    const uint dbstrdst,
    const float maxnalnposs,
    int qrylen, int dbstrlen,
    const int qrypos, const int rfnpos,
    const float d0, const float d02,
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

    const int mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs;

    //manually unroll along data blocks:
    //pos0, position index starting from 0
    for(int pos0 = threadIdx.x; qrypos + pos0 < qrylen && rfnpos + pos0 < dbstrlen;
        pos0 += blockDim.x)
    {
        //calculated distance is written to to gmem: SAVEPOS_SAVE
        float dst = 
        UpdateOneAlnPosScore<SAVEPOS,CHCKDST_NOCHECK>(//no sync;
            d02, d02,
            qrydst + qrypos + pos0,//query position
            dbstrdst + rfnpos + pos0,//reference position
            mloc + dbstrdst + pos0,//position for scores
            tfmCache,//tfm. mtx.
            scvCache,//score cache
            tmpdpdiagbuffers//scores/dsts written to gmem
        );
        //store three min distance values
        if(SAVEPOS == SAVEPOS_SAVE)
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
    if(SAVEPOS == SAVEPOS_SAVE)
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
    if(SAVEPOS == SAVEPOS_SAVE)
    if(threadIdx.x == 2) {
        float d02s = GetD02s(d0);
        if(READCNST == READCNST_CALC2) d02s += D02s_PROC_INC;

        float min3 = dstCache[threadIdx.x];

        if(CP_LARGEDST_cmp < min3 || min3 < d02s || maxnalnposs <= 3.0f)
            //max number of alignment positions (maxnalnposs) <3;
            //use the dfault threshold
            min3 = d02s;
        else {//round the 3rd minimum score to the next multiple of 0.5:
            //obtained from d02s + k*0.5 >= min3
            min3 = d02s + ceilf((min3 - d02s) * 2.0f) * 0.5f;
        }

        //write min3 to scvCache[1] as scvCache[0] is reserved for the score:
        scvCache[1] = min3;
    }

    //make the block's all threads see the reduced score scvCache[0] and 
    //distance scvCache[1]:
    __syncthreads();
}

// -------------------------------------------------------------------------
// CalcCCMatrices64Extended_Complete: calculate cross-covariance 
// matrix between the query and reference structures based on aligned 
// positions within given distance;
// Complete version for the identification of fragment boundaries;
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// qryndx, query serial number;
// ndbCposs, total number of reference positions in the chunk;
// maxnsteps, max number of steps to perform for each reference structure;
// sfragfct, current fragment factor <maxnsteps;
// qrydst, distances in positions to the beginnings of the query structures;
// dbstrdst, distances in positions to the beginnings of the reference structures;
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// dst32, squared distance threshold at which at least three aligned pairs are observed;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary diagonal buffers reused for reading positional scores;
// ccmCache, cache for the cross-covariance matrix and related data;
// 
template<int SMIDIM, int NEFFDS>
__device__ __forceinline__
void CalcCCMatrices64Extended_Complete(
    const uint qryndx,
    const uint ndbCposs,
    const uint maxnsteps,
    const uint sfragfct,
    const uint qrydst,
    const uint dbstrdst,
    const int qrylen, const int dbstrlen,
    const int qrypos, const int rfnpos,
    const float dst32,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__  ccmCache)
{
    InitCCMCacheExtended<SMIDIM,0,NEFFDS>(ccmCache);

    //no sync as long as each thread works in its own memory space

    const float d02s = dst32;

    const int mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs;

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
// SaveBestScoreAndPositions_Complete: complete version of saving the best 
// score along with query and reference positions;
// best, best score obtained by the thread block;
// qryndx, query serial number;
// dbstrndx, reference serial number;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps to perform for each reference structure;
// sfragfct, current fragment factor <maxnsteps;
// qrypos, rfnpos, starting query and reference positions;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory (includes the section of scores);
// 
__device__ __forceinline__
void SaveBestScoreAndPositions_Complete(
    float best,
    const uint qryndx,
    const uint dbstrndx,
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint sfragfct,
    const int qrypos, const int rfnpos,
    float* __restrict__ wrkmemaux)
{
    if(best <= 0.0f) return;

    //save best score
    if(threadIdx.x == 0) {
        const uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        float currentbest = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx];
        if(currentbest < best) {
            wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx] = best;
            wrkmemaux[mloc + tawmvQRYpos * ndbCstrs + dbstrndx] = qrypos;
            wrkmemaux[mloc + tawmvRFNpos * ndbCstrs + dbstrndx] = rfnpos;
        }
    }
}

// -------------------------------------------------------------------------

#endif//__covariance_complete_cuh__
