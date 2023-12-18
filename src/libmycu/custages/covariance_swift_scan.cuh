/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __covariance_swift_scan_cuh__
#define __covariance_swift_scan_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/covariance.cuh"
#include "libmycu/custages/covariance_plus.cuh"
#include "libmycu/custages/scoring.cuh"


// CopyCCDataToWrkMem2_DPRefined: copy cross-covariance matrices to 
// section-2 memory to enable efficient Kabsch 
// algorithm application for multiple structures simultaneously; 
template<int READNPOS>
__global__ 
void CopyCCDataToWrkMem2_SWFTscan(
    const uint ndbCstrs,
    const uint maxnsteps,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmem2
);

// cross-covariance calculation for massive alignments
__global__ 
void CalcCCMatrices64_SWFTscan(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmem
);

// FindD02ThresholdsCCM_DPscan: efficiently find distance thresholds 
// for the inclusion of aligned positions for CCM and rotation matrix 
// calculations during extensive superposition search;
template<int READCNST>
__global__
void FindD02ThresholdsCCM_SWFTscan(
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const float* __restrict__ tmpdpdiagbuffers,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux
);

// CalcCCMatrices64_DPscanExtended: calculate cross-covariance matrix
// between the query and reference structures based on aligned positions
// within given distance;
// Version for alignments obtained as a result of the extensive 
// superposition search;
template<int READCNST>
__global__
void CalcCCMatrices64_SWFTscanExtended(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ tmpdpdiagbuffers,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem
);

// CalcScoresUnrl_SWFTscan: calculate/reduce scores for obtained 
// superpositions;
template<int SAVEPOS, int CHCKALNLEN>
__global__
void CalcScoresUnrl_SWFTscan(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ wrkmemtm,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers
);

// CalcScoresUnrl_SWFTscanProgressive: calculate/reduce scores for obtained 
// superpositions progressively so that alignment increasing in 
// positions is ensured; version for alignments obtained by exhaustively 
// applying a linear algorithm; 
template<int SAVEPOS, int CHCKALNLEN>
__global__
void CalcScoresUnrl_SWFTscanProgressive(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint /*dbxpad*/,
    const uint maxnsteps,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ wrkmemtm,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers
);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// UpdateCCMOneAlnPos_SWFTRefined: update one position contributing to the 
// cross-covariance matrix between the query and reference structures 
// only for non-masked coordinates;
// SMIDIM, template parameter: inner-most dimensions of the cache matrix;
// pos, position in alignment buffer tmpdpalnpossbuffer;
// dblen, step (db length) by which coordinates of different dimension 
// written in tmpdpalnpossbuffer;
//
template<int SMIDIM>
__device__ __forceinline__
void UpdateCCMOneAlnPos_SWFTRefined(
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

    //compare only the first coordinates:
    if(qx < SCNTS_COORD_MASK_cmp && rx < SCNTS_COORD_MASK_cmp)
        UpdateCCMCacheExtended<SMIDIM>(ccmCache,  qx, qy, qz,  rx, ry, rz);
}

// -------------------------------------------------------------------------
// UpdateCCMOneAlnPos_SWFTExtended: update one position contributing to the 
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
void UpdateCCMOneAlnPos_SWFTExtended(
    float d02s,
    int pos, 
    int dblen,
    int scrpos,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ ccmCache)
{
    float dst = tmpdpdiagbuffers[scrpos];

    //distant positions do not contribute to cross-covariance
    //(including those masked):
    if(d02s < dst) return;

    float qx = tmpdpalnpossbuffer[pos + dpapsQRYx * dblen];
    float qy = tmpdpalnpossbuffer[pos + dpapsQRYy * dblen];
    float qz = tmpdpalnpossbuffer[pos + dpapsQRYz * dblen];

    float rx = tmpdpalnpossbuffer[pos + dpapsRFNx * dblen];
    float ry = tmpdpalnpossbuffer[pos + dpapsRFNy * dblen];
    float rz = tmpdpalnpossbuffer[pos + dpapsRFNz * dblen];

    UpdateCCMCacheExtended<SMIDIM>(ccmCache,  qx, qy, qz,  rx, ry, rz);
}

// -------------------------------------------------------------------------
// UpdateOneAlnPosScore_SWFTRefined: update score for one position of the 
// alignment obtained from applying the linear algorithm;
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
void UpdateOneAlnPosScore_SWFTRefined(
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

    float dst = CP_LARGEDST;

    //compare only the first coordinates:
    bool validpair = (qx < SCNTS_COORD_MASK_cmp && rx < SCNTS_COORD_MASK_cmp);

    if(validpair) {
        dst = transform_and_distance2(tfm, qx, qy, qz,  rx, ry, rz);

        constexpr int reduce = (CHCKDST == CHCKDST_CHECK)? 0: 1;

        if(reduce || dst <= d82)
            //calculate score
            scv[threadIdx.x] += GetPairScore(d02, dst);
    }

    if(SAVEPOS == SAVEPOS_SAVE)
        tmpdpdiagbuffers[scrpos] = dst;
}

// -------------------------------------------------------------------------
// CacheAlnPosScore_SWFTProgressive: cache score for positions of the 
// alignment obtained from applying the linear algorithm;
// SAVEPOS, template parameter to request saving positional scores;
// d02, d0 squared used for calculating score;
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
template<int SAVEPOS>
__device__ __forceinline__
void CacheAlnPosScore_SWFTProgressive(
    float d02,
    int pos, int dblen, int scrpos,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ tfm,
    float* __restrict__ scvCache,
    float* __restrict__ qnxWrkch,
    float* __restrict__ tmpdpdiagbuffers)
{
    float qx = tmpdpalnpossbuffer[pos + dpapsQRYx * dblen];
    float rx = tmpdpalnpossbuffer[pos + dpapsRFNx * dblen];

    float dst = CP_LARGEDST;
    float sco = 0.0f;
    float qnx = -1.0f;

    //compare only the first coordinates:
    bool validpair = (qx < SCNTS_COORD_MASK_cmp && rx < SCNTS_COORD_MASK_cmp);

    if(validpair) {
        float qy = tmpdpalnpossbuffer[pos + dpapsQRYy * dblen];
        float qz = tmpdpalnpossbuffer[pos + dpapsQRYz * dblen];
        float ry = tmpdpalnpossbuffer[pos + dpapsRFNy * dblen];
        float rz = tmpdpalnpossbuffer[pos + dpapsRFNz * dblen];

        //read query position:
        qnx = tmpdpdiagbuffers[scrpos + dblen];

        dst = transform_and_distance2(tfm, qx, qy, qz,  rx, ry, rz);
        sco = GetPairScore(d02, dst);
    }

    scvCache[threadIdx.x] = sco;
    qnxWrkch[threadIdx.x] = qnx;

    if(SAVEPOS == SAVEPOS_SAVE)
        tmpdpdiagbuffers[scrpos] = dst;
}

// version for using registers instead of smem:
//
__device__ __forceinline__
void CacheAlnPosScore_SWFTProgressive_Reg(
    float d02,
    int pos, int dblen, int scrpos,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ tfm,
    float* __restrict__ sco,
    float* __restrict__ qnx,
    const float* __restrict__ tmpdpdiagbuffers)
{
    float qx = tmpdpalnpossbuffer[pos + dpapsQRYx * dblen];
    float rx = tmpdpalnpossbuffer[pos + dpapsRFNx * dblen];
    float dst = CP_LARGEDST;

    //compare only the first coordinates:
    bool validpair = (qx < SCNTS_COORD_MASK_cmp && rx < SCNTS_COORD_MASK_cmp);

    if(validpair) {
        float qy = tmpdpalnpossbuffer[pos + dpapsQRYy * dblen];
        float qz = tmpdpalnpossbuffer[pos + dpapsQRYz * dblen];
        float ry = tmpdpalnpossbuffer[pos + dpapsRFNy * dblen];
        float rz = tmpdpalnpossbuffer[pos + dpapsRFNz * dblen];

        //read query position:
        *qnx = tmpdpdiagbuffers[scrpos + dblen];

        dst = transform_and_distance2(tfm, qx, qy, qz,  rx, ry, rz);
        *sco = GetPairScore(d02, dst);
    }
}

// -------------------------------------------------------------------------
// FindMax_SWFTProgressive: find in the cache max score up to a given query 
// position qnx;
// UNCND, template parameter, unconditionally find max;
// 
template<int pad, int xdim, int szqnxch, int nwrpsdim, bool UNCND>
__device__ __forceinline__
float FindMax_SWFTProgressive(
    float qnx,
    const float* __restrict__ qnxCache,
    const float* __restrict__ maxCache,
    float* __restrict__ tmpSMbuf)
{
    float max = 0.0f;

    //save max in a block thread 
    for(int c = threadIdx.x; c < szqnxch; c += xdim) {
        if(max < maxCache[pad+c] && (UNCND || qnxCache[pad+c] < qnx))
            max = maxCache[pad+c];
    }

    //per-warp-reduce for max (for tid:tid%32==0):
    max = mywarpreducemax(max);
    if((threadIdx.x & 31) == 0) tmpSMbuf[threadIdx.x>>5] = max;
    __syncthreads();

    //NOTE: assuming nwrpsdim<=32!
    //NOTE: (1024 is currently max for the thread block; 
    //NOTE: works otherwise but inconsistently)
    if(threadIdx.x < nwrpsdim) max = tmpSMbuf[threadIdx.x];
    max = mywarpreducemax(max);

    return max;
}

// version for a single warp:
template<int xdim, int szqnxch, bool UNCND>
__device__ __forceinline__
float FindMax_SWFTProgressive_Warp(
    float qnx,
    const float* __restrict__ qnxCache,
    const float* __restrict__ maxCache)
{
    float max = 0.0f;

    //save max in a block thread 
    for(int c = threadIdx.x; c < szqnxch; c += xdim) {
        if(max < maxCache[c] && (UNCND || qnxCache[c] < qnx))
            max = maxCache[c];
    }

    //warp-reduce for max:
    max = mywarpreducemax(max);

    return max;
}

// -------------------------------------------------------------------------

#endif//__covariance_swift_scan_cuh__
