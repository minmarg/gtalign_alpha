/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __covariance_dp_refn_cuh__
#define __covariance_dp_refn_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/covariance.cuh"
#include "libmycu/custages/covariance_plus.cuh"
#include "libmycu/custages/scoring.cuh"


// CopyCCDataToWrkMem2_DPRefined: copy cross-covariance matrices to 
// section-2 memory to enable efficient Kabsch 
// algorithm application for multiple structures simultaneously; 
__global__ 
void CopyCCDataToWrkMem2_DPRefined(
    const uint ndbCstrs,
    const uint maxnsteps,
    const int sfragstep,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmem2
);

// InitCopyCheckConvergence64_DPRefined: check whether calculating 
// rotation matrices converged by verifying the absolute 
// difference of two latest cross-covariance matrix data;
template<int CC64Action>
__global__
void InitCopyCheckConvergence64_DPRefined(
    const uint ndbCstrs,
    const uint maxnsteps,
    const int sfragstep,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemccd,
    float* __restrict__ wrkmemaux
);

// cross-covariance calculation for refinement
__global__ 
void CalcCCMatrices64_DPRefined(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const int sfragstep,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmem
);

// FindD02ThresholdsCCM_DPRefined: efficiently find distance thresholds 
// for the inclusion of aligned positions for CCM and rotation matrix 
// calculations during the boundaries refinement of fragments initially 
// identified by DP;
template<int READCNST>
__global__
void FindD02ThresholdsCCM_DPRefined(
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int sfragstep,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux
);

// CalcCCMatrices64_DPRefinedExtended: calculate cross-covariance matrix 
// between the query and reference structures based on aligned positions 
// within given distance;
// Version for the refinement of fragment boundaries obtained as a 
// result of the application of DP;
template<int READCNST>
__global__
void CalcCCMatrices64_DPRefinedExtended(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const int sfragstep,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ tmpdpdiagbuffers,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem
);

// CalcScoresUnrl_DPRefined: calculate/reduce UNNORMALIZED scores for 
// obtained superpositions; version for the refinement of fragments 
// obtained by DP; 
template<int SAVEPOS, int CHCKCONV>
__global__
void CalcScoresUnrl_DPRefined(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const int sfragstep,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers
);



// -------------------------------------------------------------------------
// UpdateCCMOneAlnPos_DPRefined: update one position of the alignment 
// obtained by DP, contributing to the cross-covariance matrix between the 
// query and reference structures
//
template<int SMIDIM = twmvEndOfCCData>
__device__ __forceinline__
void UpdateCCMOneAlnPos_DPRefined(
    int pos, int dblen,
    const float* __restrict__ tmpdpalnpossbuffer,
    FPTYPE* __restrict__ ccmCache)
{
    float qx = tmpdpalnpossbuffer[pos + dpapsQRYx * dblen];
    float qy = tmpdpalnpossbuffer[pos + dpapsQRYy * dblen];
    float qz = tmpdpalnpossbuffer[pos + dpapsQRYz * dblen];

    float rx = tmpdpalnpossbuffer[pos + dpapsRFNx * dblen];
    float ry = tmpdpalnpossbuffer[pos + dpapsRFNy * dblen];
    float rz = tmpdpalnpossbuffer[pos + dpapsRFNz * dblen];

    UpdateCCMCache<SMIDIM>(ccmCache,  qx, qy, qz,  rx, ry, rz);
}

// UpdateExtCCMOneAlnPos_DPRefined: extension to UpdateCCMOneAlnPos_DPRefined;
// coordinate squares are updated too;
//
template<int SMIDIM = twmvEndOfCCData>
__device__ __forceinline__
void UpdateExtCCMOneAlnPos_DPRefined(
    int pos, int dblen,
    const float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ ccmCache)
{
    float qx = tmpdpalnpossbuffer[pos + dpapsQRYx * dblen];
    float qy = tmpdpalnpossbuffer[pos + dpapsQRYy * dblen];
    float qz = tmpdpalnpossbuffer[pos + dpapsQRYz * dblen];

    float rx = tmpdpalnpossbuffer[pos + dpapsRFNx * dblen];
    float ry = tmpdpalnpossbuffer[pos + dpapsRFNy * dblen];
    float rz = tmpdpalnpossbuffer[pos + dpapsRFNz * dblen];

    UpdateExtCCMCache<SMIDIM>(ccmCache,  qx, qy, qz,  rx, ry, rz);
}

// -------------------------------------------------------------------------
// UpdateCCMOneAlnPos_DPExtended: update one position contributing to the 
// cross-covariance matrix between the query and reference structures 
// only if transformed query is within the given distance from reference;
// SMIDIM, template parameter: inner-most dimensions of the cache matrix;
// d02s, d0 squared used for the inclusion of pairs in the alignment;
// pos, position in alignment buffer tmpdpalnpossbuffer;
// dblen, step (db length) by which coordinates of different dimension 
// written in tmpdpalnpossbuffer;
// scrpos, position index to read the score obtained at the alignment 
// position;
//
template<int SMIDIM>
__device__ __forceinline__
void UpdateCCMOneAlnPos_DPExtended(
    float d02s,
    int pos, 
    int dblen,
    int scrpos,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ ccmCache)
{
    float dst = tmpdpdiagbuffers[scrpos];

    if(d02s < dst)
        //distant positions do not contribute to cross-covariance
        return;

    float qx = tmpdpalnpossbuffer[pos + dpapsQRYx * dblen];
    float qy = tmpdpalnpossbuffer[pos + dpapsQRYy * dblen];
    float qz = tmpdpalnpossbuffer[pos + dpapsQRYz * dblen];

    float rx = tmpdpalnpossbuffer[pos + dpapsRFNx * dblen];
    float ry = tmpdpalnpossbuffer[pos + dpapsRFNy * dblen];
    float rz = tmpdpalnpossbuffer[pos + dpapsRFNz * dblen];

    UpdateCCMCacheExtended<SMIDIM>(ccmCache,  qx, qy, qz,  rx, ry, rz);
}

// -------------------------------------------------------------------------
// UpdateOneAlnPosScore_DPRefined: update score unconditionally for one 
// position of the alignment obtained by DP;
// SAVEPOS, template parameter to request saving positional scores;
// CHCKDST, template parameter to request accumulating scores within the 
// given threshold distance only;
// d02, d0 squared used for calculating score;
// d82, distance threshold for reducing scores;
// pos, position in alignment buffer tmpdpalnpossbuffer;
// dblen, step (db length) by which coordinates of different dimension 
// written in tmpdpalnpossbuffer;
// scrpos, position index to write the score obtained at the alignment 
// position;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// tfm, address of the transformation matrix;
// scv, address of the vector of scores;
// tmpdpdiagbuffers, global memory address for saving positional scores;
//
template<int SAVEPOS, int CHCKDST>
__device__ __forceinline__
float UpdateOneAlnPosScore_DPRefined(
    float d02, float d82,
    int pos, int dblen, int scrpos,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ tfm,
    float* __restrict__ scv,
    float* __restrict__ tmpdpdiagbuffers)
{
    float qx = tmpdpalnpossbuffer[pos + dpapsQRYx * dblen];
    float qy = tmpdpalnpossbuffer[pos + dpapsQRYy * dblen];
    float qz = tmpdpalnpossbuffer[pos + dpapsQRYz * dblen];

    float rx = tmpdpalnpossbuffer[pos + dpapsRFNx * dblen];
    float ry = tmpdpalnpossbuffer[pos + dpapsRFNy * dblen];
    float rz = tmpdpalnpossbuffer[pos + dpapsRFNz * dblen];

    float dst = transform_and_distance2(tfm, qx, qy, qz,  rx, ry, rz);

    constexpr int reduce = (CHCKDST == CHCKDST_CHECK)? 0: 1;

    if(reduce || dst <= d82)
        //calculate score
        scv[threadIdx.x] += GetPairScore(d02, dst);

    if(SAVEPOS == SAVEPOS_SAVE)
        tmpdpdiagbuffers[scrpos] = dst;

    return dst;
}

// -------------------------------------------------------------------------

#endif//__covariance_dp_refn_cuh__
