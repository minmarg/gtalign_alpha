/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __covariance_dp_scan_cuh__
#define __covariance_dp_scan_cuh__

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
template<int READNPOS>
__global__ 
void CopyCCDataToWrkMem2_DPscan(
    const uint ndbCstrs,
    const uint maxnsteps,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmem2
);

// cross-covariance calculation for massive alignments
__global__ 
void CalcCCMatrices64_DPscan(
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
// calculations during exhaustive application of DP;
template<int READCNST>
__global__
void FindD02ThresholdsCCM_DPscan(
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
// Version for alignments obtained as a result of the exhaustive
// application of DP;
template<int READCNST>
__global__
void CalcCCMatrices64_DPscanExtended(
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

// CalcScoresUnrl_DPscan: calculate/reduce scores for obtained 
// superpositions; version for alignments obtained by exhaustively 
// applying DP; 
template<int SAVEPOS, int CHCKALNLEN>
__global__
void CalcScoresUnrl_DPscan(
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

// -------------------------------------------------------------------------

#endif//__covariance_dp_scan_cuh__
