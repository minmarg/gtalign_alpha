/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __scoring_cuh__
#define __scoring_cuh__

#include "libmymp/mpstages/scoringbase.h"
#include "covariance.cuh"
#include "transform.cuh"


// SetCurrentFragSpecs: set the specifications of the current fragment 
// under process;
__global__ void SetCurrentFragSpecs(
    const uint ndbCstrs,
    const uint maxnsteps,
    const int sfragndx,
    float* __restrict__ wrkmemaux
);

// SetLowScoreConvergenceFlag: set the appropriate convergence flag for 
// the pairs for which the score is below the threshold;
__global__ void SetLowScoreConvergenceFlag(
    const float scorethld,
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux
);

// InitScores: initialize best and current scores to 0;
// INITOPT, template parameter controlling which scores are to be 
// initialized, see above;
template<int INITOPT>
__global__ void InitScores(
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint minfraglen,
    const bool checkfragos,
    float* __restrict__ wrkmemaux
);

// SaveLastScore0: save last calculated score;
__global__ void SaveLastScore0(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux
);

// SaveBestScore: save best score along with query and reference 
// positions for which this score is observed;
__global__ void SaveBestScore(
    const uint ndbCstrs,
    const uint maxnsteps,
    int n1, int step,
    float* __restrict__ wrkmemaux
);

// SaveBestScoreAmongBests: save best score along with query and reference 
// positions by considering all partial best scores calculated over all 
// fragment factors; write it to the location of fragment factor 0;
__global__ void SaveBestScoreAmongBests(
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint effnsteps,
    float* __restrict__ wrkmemaux
);

// CheckScoreConvergence: check whether the score of the last two 
// procedures converged, i.e., the difference is small;
__global__ void CheckScoreConvergence(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux
);

// CheckScoreProgression: check whether the difference between the maximum 
// score and the score of the last procedure is large enough; if not, set 
// the appropriate convergence flag;
__global__ void CheckScoreProgression(
    uint ndbCstrs,
    float maxscorefct,
    float* __restrict__ wrkmemaux
);

// SaveBestScoreAndTM: save best scores along with transformation matrices;
template<bool WRITEFRAGINFO>
__global__
void SaveBestScoreAndTM(
    const uint ndbCstrs,
    const uint maxnsteps,
    const int sfragstep,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux
);



// SaveBestScoreAndTMAmongBests: save best scores with transformation 
// matrices by considering all partial best scores calculated over all 
// fragment factors; write it to the location of fragment factor 0;
template<
    bool WRITEFRAGINFO,
    bool GRANDUPDATE = true,
    bool FORCEWRITEFRAGINFO = false,
    int SECONDARYUPDATE = SECONDARYUPDATE_NOUPDATE>
__global__
void SaveBestScoreAndTMAmongBests(
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint effnsteps,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ tfmmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmemtmibest2nd = NULL
);

// ProductionSaveBestScoresAndTMAmongBests: save best scores and respective 
// transformation matrices by considering all partial best scores 
// calculated over all fragment factors; production version;
template<bool WRITEFRAGINFO, bool CONDITIONAL>
__global__
void ProductionSaveBestScoresAndTMAmongBests(
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint effnsteps,
    const float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ alndatamem,
    float* __restrict__ tfmmem
);



// SaveTopNScoresAndTMsAmongBests: save top N scores and respective 
// transformation matrices by considering all partial best scores 
// calculated over all fragment factors;
__global__ void SaveTopNScoresAndTMsAmongBests(
    const uint ndbCstrs,
    const uint maxnsteps,
//     const uint effnsteps,
    const float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemaux
);

// SaveTopNScoresAndTMsAmongSecondaryBests: save secondary top N scores and 
// respective transformation matrices by considering all partial best scores 
// calculated over all fragment factors; write the information to the first
// N locations of fragment factors;
__global__ void SaveTopNScoresAndTMsAmongSecondaryBests(
    const int depth,
    const bool firstit,
    const bool twoconfs,
    const int rfnfragfctinit,
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint effnsteps,
    const float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemaux
);



#if 0
// SaveBestDPscoreAndTMAmongDPswifts: save best DP scores and respective 
// transformation matrices by considering all partial DP swift scores 
// calculated over all fragment factors;
// template<bool WRITEFRAGINFO, bool READSCORE, bool STEPx5>
__global__
void SaveBestDPscoreAndTMAmongDPswifts(
    bool WRITEFRAGINFO, bool READSCORE, bool STEPx5,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const uint effnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    const float* __restrict__ tmpdpdiagbuffers,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmtarget,
    float* __restrict__ wrkmemaux
);
#endif



// SortBestDPscoresAndTMsAmongDPswifts: sort best DP scores and then save 
// them along with respective transformation matrices by considering all 
// partial DP swift scores calculated over all fragment factors; 
__global__ void SortBestDPscoresAndTMsAmongDPswifts(
    const uint nbranches,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const float* __restrict__ tmpdpdiagbuffers,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmtarget,
    float* __restrict__ wrkmemaux
);



// -------------------------------------------------------------------------
// UpdateOneAlnPosScore: update score unconditionally for one alignment 
// position;
// SAVEPOS, template parameter to request saving positional scores;
// CHCKDST, template parameter to request accumulating scores within the 
// given threshold distance only;
// d02, d0 squared used for calculating score;
// d82, distance threshold for reducing scores;
// qrypos, starting query position;
// rfnpos, starting reference position;
// scrpos, position index to write the score obtained at the alignment 
// position;
// tfm, address of the transformation matrix;
// scv, address of the vector of scores;
// tmpdpdiagbuffers, global memory address for saving positional scores;
//
template<int SAVEPOS, int CHCKDST>
__device__ __forceinline__
float UpdateOneAlnPosScore(
    float d02, float d82,
    int qrypos, int rfnpos, int scrpos,
    const float* __restrict__ tfm,
    float* __restrict__ scv,
    float* __restrict__ tmpdpdiagbuffers)
{
    float qx = GetQueryCoord<pmv2DX>(qrypos);
    float qy = GetQueryCoord<pmv2DY>(qrypos);
    float qz = GetQueryCoord<pmv2DZ>(qrypos);

    float rx = GetDbStrCoord<pmv2DX>(rfnpos);
    float ry = GetDbStrCoord<pmv2DY>(rfnpos);
    float rz = GetDbStrCoord<pmv2DZ>(rfnpos);

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
// UpdateOneAlnPosScore_frg2: update score unconditionally for one alignment 
// position;
// CHCKDST, template parameter to request accumulating scores within the 
// given threshold distance only;
// REVERSE, flag of reverse transformation;
// d02, d0 squared used for calculating score;
// d82, distance threshold for reducing scores;
// qrypos, starting query position;
// rfnpos, starting reference position;
// tfm, address of the transformation matrix;
// scv, address of the vector of scores;
//
template<int CHCKDST>
__device__ __forceinline__
void UpdateOneAlnPosScore_frg2(
    const bool REVERSE,
    float d02, float d82,
    int qrypos, int rfnpos,
    const float* __restrict__ tfm,
    float* __restrict__ scv)
{
    float qx = GetQueryCoord<pmv2DX>(qrypos);
    float qy = GetQueryCoord<pmv2DY>(qrypos);
    float qz = GetQueryCoord<pmv2DZ>(qrypos);

    float rx = GetDbStrCoord<pmv2DX>(rfnpos);
    float ry = GetDbStrCoord<pmv2DY>(rfnpos);
    float rz = GetDbStrCoord<pmv2DZ>(rfnpos);

    float dst =
        REVERSE
        ? transform_and_distance2(tfm, rx, ry, rz,  qx, qy, qz)
        : transform_and_distance2(tfm, qx, qy, qz,  rx, ry, rz);

    constexpr int reduce = (CHCKDST == CHCKDST_CHECK)? 0: 1;

    if(reduce || dst <= d82)
        //calculate score
        scv[threadIdx.x] += GetPairScore(d02, dst);
}

#endif//__scoring_cuh__
