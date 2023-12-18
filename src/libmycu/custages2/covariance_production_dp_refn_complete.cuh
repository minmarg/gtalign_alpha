/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __covariance_production_dp_refn_complete_cuh__
#define __covariance_production_dp_refn_complete_cuh__

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
// ProductionFragmentBasedDPAlignmentRefinementPhase1: phase 1 to perform 
// production-version tuning of the the best superposition obtained 
// within the single kernel's actions; write the final superposition scores;
//
template<bool TFM_DINV>
__global__ 
void ProductionFragmentBasedDPAlignmentRefinementPhase1(
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
    float* __restrict__ wrkmemaux,
    float* __restrict__ alndatamem
);

// =========================================================================
// ProductionFragmentBasedDPAlignmentRefinementPhase2: phase 2 to perform 
// production-version tuning of the the best superposition obtained 
// within the single kernel's actions; write the final superposition scores;
//
template<bool TFM_DINV>
__global__ 
void ProductionFragmentBasedDPAlignmentRefinementPhase2(
    const int nmaxconvit,
//     const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint nmaxsubfrags,
    const uint maxnfragfcts,
    const uint maxnsteps,
    const int sfragstep,
    const int /*maxalnmax*/,
    const float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux
);

// ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearch: phase 2 to
// perform production-version tuning of the the best superposition obtained 
// within the single kernel's actions; write the final superposition scores;
// this version performs full search of maxnfragfcts positions from the 
// identified one in phase 1 for each fragment length;
template<bool TFM_DINV>
__global__ 
void ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearch(
    const int nmaxconvit,
//     const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint nmaxsubfrags,
    const uint maxnfragfcts,
    const uint maxnsteps,
    const int sfragstep,
    const int /*maxalnmax*/,
    const float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux
);

// ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearch: phase 2 to
// perform production-version tuning of the the best superposition obtained 
// within the single kernel's actions; write the final superposition scores;
// NOTE: This version performs a log number of superposition evaluations;
template<bool TFM_DINV>
__global__ 
void ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearch(
    const int nmaxconvit,
//     const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint nmaxsubfrags,
    const uint /*maxnfragfcts*/,
    const uint maxnsteps,
    const int /*sfragstep*/,
    const int /*maxalnmax*/,
    const float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux,
    float* __restrict__ alndatamem,
    float* __restrict__ tfmmem
);



// =========================================================================
// CalcRMSD_Complete: thread block calculates RMSD;
// ccmCache, cache for the cross-covariance matrix and reuse;
// Based on the original Kabsch algorithm (see CalcTfmMatrices_Complete);
//
__device__ __forceinline__
float CalcRMSD_Complete(float* __restrict__ ccmCache)
{
    __shared__ float rr[6];//rrCache[6];

    //#positions used to calculate cross-covarinaces:
    float nalnposs = ccmCache[twmvNalnposs];

    if(nalnposs <= 0.0f) return 0.0f;

    //calculate query center vector in advance
    if((twmvCVq_0 <= threadIdx.x && threadIdx.x <= twmvCVq_2) ||
       (twmvCV2q_0 <= threadIdx.x && threadIdx.x <= twmvCV2q_2))
        ccmCache[threadIdx.x] = fdividef(ccmCache[threadIdx.x], nalnposs);

    __syncwarp();

    if(threadIdx.x == 0) CalcRmatrix(ccmCache);

    //calculate reference center vector now
    if((twmvCVr_0 <= threadIdx.x && threadIdx.x <= twmvCVr_2) ||
       (twmvCV2r_0 <= threadIdx.x && threadIdx.x <= twmvCV2r_2))
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

    float E_0;//E_0 in Kabsch's 1978 paper
    if(threadIdx.x == 0)
        E_0 = //two variances
            ccmCache[twmvCV2q_0] - SQRD(ccmCache[twmvCVq_0]) +
            ccmCache[twmvCV2q_1] - SQRD(ccmCache[twmvCVq_1]) +
            ccmCache[twmvCV2q_2] - SQRD(ccmCache[twmvCVq_2]) +
            ccmCache[twmvCV2r_0] - SQRD(ccmCache[twmvCVr_0]) +
            ccmCache[twmvCV2r_1] - SQRD(ccmCache[twmvCVr_1]) +
            ccmCache[twmvCV2r_2] - SQRD(ccmCache[twmvCVr_2]);

    __syncwarp();

    //Kabsch:
    //eigenvalues: form characteristic cubic x**3-3*spur*x**2+3*cof*x-det=0
    float spur = (rr[0] + rr[2] + rr[5]) * oneTHIRDf;
    float cof = (((((rr[2] * rr[5] - SQRD(rr[4])) + rr[0] * rr[5]) -
                SQRD(rr[3])) + rr[0] * rr[2]) -
                SQRD(rr[1])) * oneTHIRDf;

    bool abok = (spur > 0.0f);

    float e0, e1, e2;//polynomial roots (eigenvalues)
    e0 = e1 = e2 = spur;

    if(abok && threadIdx.x == 0)
    {   //Kabsch:
        //reduce cubic to standard form y**3-3hy+2g=0 by putting x=y+spur;
        //Kabsch: solve cubic: roots are e[0],e[1],e[2] in decreasing order
        //Kabsch: handle special case of 3 identical roots
        SolveCubic(det, spur, cof, e0, e1, e2);
    }

    e0 = (e0 <= 0.0f)? 0.0f: sqrtf(e0);
    e1 = (e1 <= 0.0f)? 0.0f: sqrtf(e1);
    e2 = (e2 <= 0.0f)? 0.0f: (sqrtf(e2) * ((det < 0.0f)? -1.0f: 1.0f));

    //write rmsd to E_0:
    //NOTE: scale the eigenvalues to get values for the unscaled RTR;
    E_0 -= __fdividef(2.0f * scale * (e0 + e1 + e2), nalnposs);
    E_0 = (E_0 <= 0.0f)? 0.0f: sqrtf(E_0);

    return E_0;
}

// -------------------------------------------------------------------------
// CalcExtCCMatrices64_DPRefined_Complete: calculate cross-covariance matrix 
// between the query and reference structures for refinement, i.e. 
// delineation of suboptimal fragment boundaries;
// This complete version complements CalcCCMatrices64_DPRefined_Complete by
// additionally calculating the sum of squares required for RMSD computation;
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
void CalcExtCCMatrices64_DPRefined_Complete(
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
    InitCCMCacheExtended<SMIDIM,0,SMIDIM>(ccmCache);

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
        UpdateExtCCMOneAlnPos_DPRefined<SMIDIM>(//no sync.
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
            for(int i = 0; i < SMIDIM; i++)
                ccmCache[threadIdx.x * SMIDIM + i] +=
                    ccmCache[(threadIdx.x + xdim) * SMIDIM + i];
        }
        __syncthreads();
    }

    //unroll warp
    if(threadIdx.x < 32) {
        for(int i = 0; i < SMIDIM; i++) {
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
// SaveBestQRScoresAndTM_Complete: complete version of saving transformation
// and the best scores calculated for the query and reference structures;
// save fragment indices and starting positions too;
// WRITEFRAGINFO, template parameter, flag of writing fragment attributes;
// CONDITIONAL, template parameter, flag of writing the score if it's greater at the same location;
// best, best score calculated for the smaller length;
// gbest, best score calculated for the greater length;
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
void SaveBestQRScoresAndTM_Complete(
    const float best,
    const float gbest,
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
    // if(best <= 0.0f) return;

    float currentbest = -1.0f;

    //save best score
    if(threadIdx.x == 0)
    {
        const uint mloc = ((qryndx * maxnsteps + sfragfctxndx) * nTAuxWorkingMemoryVars) * ndbCstrs;

        if(CONDITIONAL) 
            currentbest = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx];

        if(currentbest < best) {
            wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx] = best;
            //reuse the tawmvBest0 slot, which has been used only in DP-based refinement
            wrkmemaux[mloc + tawmvBest0 * ndbCstrs + dbstrndx] = gbest;
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
// SaveBestQRScoresAndTM_Phase2_logsearch_Complete: complete version of 
// saving transformation and the best scores calculated for the query and 
// reference structures directly to the production output memory region;
// see also ProductionSaveBestScoresAndTMAmongBests;
// best, best score calculated for the smaller length;
// gbest, best score calculated for the greater length;
// qryndx, query serial number;
// dbstrndx, reference serial number;
// ndbCstrs, total number of reference structures in the chunk;
// qrylenorg, dbstrlenorg, query and reference lengths;
// NOTE: memory pointers should be aligned!
// tfmCache, cached transformation matrix;
// tfmmem, output memory for best transformation matrices;
// alndatamem, memory for full alignment information, including scores;
// 
__device__ __forceinline__
void SaveBestQRScoresAndTM_Phase2_logsearch_Complete(
    float best,
    float gbest,
    const uint qryndx,
    const uint dbstrndx,
    const uint ndbCstrs,
    const int qrylenorg,
    const int dbstrlenorg,
    const float* __restrict__ tfmCache,
    float* __restrict__ tfmmem,
    float* __restrict__ alndatamem)
{
    //save best scores
    if(threadIdx.x == 0) {
        uint mloc = (qryndx * ndbCstrs + dbstrndx) * nTDP2OutputAlnData;
        //make best represent the query score:
        if(dbstrlenorg < qrylenorg) myhdswap(best, gbest);
        //NOTE: d0Q and d0R thresholds are assumed to be saved previously;
        alndatamem[mloc + dp2oadScoreQ] = __fdividef(best, qrylenorg);
        alndatamem[mloc + dp2oadScoreR] = __fdividef(gbest, dbstrlenorg);
    }

    //save transformation matrix
    if(threadIdx.x < nTTranformMatrix) {
        uint tfmloc = (qryndx * ndbCstrs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        tfmmem[tfmloc] = tfmCache[threadIdx.x];
    }
}

// -------------------------------------------------------------------------

#endif//__covariance_production_dp_refn_complete_cuh__
