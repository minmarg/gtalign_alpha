/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __covariance_plus_h__
#define __covariance_plus_h__

// #include "libutil/macros.h"
// #include "libgenp/gproc/gproc.h"
// #include "libgenp/gdats/PM2DVectorFields.h"
// #include "libmycu/cucom/cucommon.h"
// #include "libmycu/culayout/cuconstant.cuh"
#include "covariance.cuh"
#include "transform.cuh"
#include "scoring.cuh"

// efficiently find distance d02s thresholds for the 
// inclusion of aligned positions for CCM and rotation matrix 
// calculations;
// READCNST, template parameter controlling how constants are 
// defined, see above;
template<int READCNST>
__global__ void FindD02ThresholdsCCM(
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    int n1, int step,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux
);

// extended version of cross-covariance calculation, including only 
// pairs within given distance threshold;
// READCNST, template parameter controlling the verification of the 
// number of aligned positions;
template<int READCNST>
__global__ void CalcCCMatrices64Extended(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    int n1, int step,
    const float* __restrict__ tmpdpdiagbuffers,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem
);

// template constants for kernel CalcScoresUnrl:
// do not check the number of positions in the alignment:
#define CHCKALNLEN_NOCHECK 0
// check the number of positions in the alignment:
#define CHCKALNLEN_CHECK 1
// CalcScoreUnrl: calculate/reduce score for obtained superpositions; 
// save partial sums;
template<int SAVEPOS, int CHCKALNLEN>
__global__ void CalcScoresUnrl(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    int n1, int step,
    const float* __restrict__ wrkmemtm,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers
);

// CalcScoresUnrl_frg2: calculate/reduce initial scores for obtained 
// superpositions during extensive fragment-based search of optimal superpositions;
__global__ void CalcScoresUnrl_frg2(
    const float thrscorefactor,
    const bool dynamicorientation,
    const int depth,
    const uint ndbCstrs,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemaux
);




// -------------------------------------------------------------------------
// StoreMinDst: store minimum distances in three cache buffers
__device__ __forceinline__
void StoreMinDst(
    float* __restrict__ dstChe,
    float dst)
{
    if(dst < dstChe[0]) {
        dstChe[2] = dstChe[1];
        dstChe[1] = dstChe[0];
        dstChe[0] = dst;
    } else if(dst < dstChe[1]) {
        dstChe[2] = dstChe[1];
        dstChe[1] = dst;
    } else if(dst < dstChe[2])
        dstChe[2] = dst;
}

// -------------------------------------------------------------------------
// StoreMinDstSrc: store minimum distances from source in three cache 
// buffers
__device__ __forceinline__
void StoreMinDstSrc(
    float* __restrict__ dstChe,
    const float* __restrict__ dstSrc)
{
    StoreMinDst(dstChe, dstSrc[0]);
    StoreMinDst(dstChe, dstSrc[1]);
    StoreMinDst(dstChe, dstSrc[2]);
}

// -------------------------------------------------------------------------
// GetMinScoreOneAlnPos: keep track of top three minimum scores by 
// considering the given position;
// SMIDIM, template parameter: inner-most dimensions of the cache matrix;
//
template<int SMIDIM>
__device__ __forceinline__
void GetMinScoreOneAlnPos(
    int scrpos,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ ccmCache)
{
    float dst = tmpdpdiagbuffers[scrpos];
    int tslot = threadIdx.x * SMIDIM;

    //ccmCache will contain blockDim.x (or length-size) 
    // (possibly equal) minimum scores
    StoreMinDst(ccmCache + tslot, dst);
//     //TODO:REMOVE:
//     ccmCache[tslot] = myhdmin(ccmCache[tslot], dst);
}

// -------------------------------------------------------------------------
// UpdateCCMCacheExtended: update cross-covariance cache data given query 
// and reference coordinates, respectively;
//
template<int SMIDIM>
__device__ __forceinline__
void UpdateCCMCacheExtended(
    float* __restrict__ ccmCache,
    float qx, float qy, float qz,
    float rx, float ry, float rz)
{
    int tslot = threadIdx.x * SMIDIM;

    ccmCache[tslot + twmvCCM_0_0] += qx * rx;
    ccmCache[tslot + twmvCCM_0_1] += qx * ry;
    ccmCache[tslot + twmvCCM_0_2] += qx * rz;

    ccmCache[tslot + twmvCCM_1_0] += qy * rx;
    ccmCache[tslot + twmvCCM_1_1] += qy * ry;
    ccmCache[tslot + twmvCCM_1_2] += qy * rz;

    ccmCache[tslot + twmvCCM_2_0] += qz * rx;
    ccmCache[tslot + twmvCCM_2_1] += qz * ry;
    ccmCache[tslot + twmvCCM_2_2] += qz * rz;

    ccmCache[tslot + twmvCVq_0] += qx;
    ccmCache[tslot + twmvCVq_1] += qy;
    ccmCache[tslot + twmvCVq_2] += qz;

    ccmCache[tslot + twmvCVr_0] += rx;
    ccmCache[tslot + twmvCVr_1] += ry;
    ccmCache[tslot + twmvCVr_2] += rz;

    //update the number of positions
    ccmCache[tslot + twmvNalnposs] += 1.0f;
}

// -------------------------------------------------------------------------
// UpdateCCMOneAlnPosExtended: update one position contributing to the 
// cross-covariance matrix between the query and reference structures 
// only if transformed query is within the given distance from reference;
// SMIDIM, template parameter: inner-most dimensions of the cache matrix;
// d02s, d0 squared used for the inclusion of pairs in the alignment;
// scrpos, position index to read the score obtained at the alignment 
// position;
//
template<int SMIDIM>
__device__ __forceinline__
void UpdateCCMOneAlnPosExtended(
    float d02s,
    int qrypos,
    int rfnpos,
    int scrpos,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ ccmCache)
{
    float dst = tmpdpdiagbuffers[scrpos];

    if(d02s < dst)
        //distant positions do not contribute to cross-covariance
        return;

    float qx = GetQueryCoord<pmv2DX>(qrypos);
    float qy = GetQueryCoord<pmv2DY>(qrypos);
    float qz = GetQueryCoord<pmv2DZ>(qrypos);

    float rx = GetDbStrCoord<pmv2DX>(rfnpos);
    float ry = GetDbStrCoord<pmv2DY>(rfnpos);
    float rz = GetDbStrCoord<pmv2DZ>(rfnpos);

    UpdateCCMCacheExtended<SMIDIM>(ccmCache,  qx, qy, qz,  rx, ry, rz);
}

// -------------------------------------------------------------------------
// InitCCMCache: initialize cache;
//
template<int SMIDIM, int ndxFROM, int ndxTO>
__device__ __forceinline__
void InitCCMCacheExtended(float* __restrict__ ccmCache)
{
    int tslot = threadIdx.x * SMIDIM;

    #pragma unroll
    for(int i = ndxFROM; i < ndxTO; i++)
        ccmCache[tslot + i] = 0.0f;
}

#endif//__covariance_plus_h__
