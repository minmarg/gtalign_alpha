/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __covariance_refn_cuh__
#define __covariance_refn_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fragment.cuh"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/covariance.cuh"


// CopyCCDataRefined: copy cross-covariance data between the query and 
// reference structures to additional memory location (save last value);
__global__ void CopyCCDataRefined(
    const uint ndbCstrs,
    const uint maxnsteps,
    const int sfragstep,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmemccd,
    float* __restrict__ wrkmemaux
);

// CopyCCDataToWrkMem2Refined: copy cross-covariance matrices to 
// section-2 memory to enable efficient Kabsch 
// algorithm application for multiple structures simultaneously; 
__global__ 
void CopyCCDataToWrkMem2Refined(
    const uint ndbCstrs,
    const uint maxnsteps,
    const int sfragstep,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmem2
);

// private constants for kernel InitCopyCheckConvergence64Refined
enum {
    covrfnccrLength,
    covrfnccrPosition,
    covrfnccrFragNdx,
    covrfnccrFragPos,
    covrfnccrTotal
};

//template values for kernel InitCopyCheckConvergence64Refined:
//0, check convergence;
//1, copy CCData only; substitute for CopyCCDataRefined;
//2, check convergence followed by CCData copy ;
//4, initialize CCData only; substitute for InitCCData;
#define CC64Action_Convergence 0
#define CC64Action_CopyCCData 1
#define CC64Action_Convergence_CopyCCData 2
#define CC64Action_InitCCData 4
// InitCopyCheckConvergence64Refined: check whether calculating rotation 
// matrices converged by verifying the absolute difference of two latest 
// cross-covariance matrix data;
template<int CC64Action>
__global__
void InitCopyCheckConvergence64Refined(
    const uint ndbCstrs,
    const uint maxnsteps,
    const int sfragstep,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemccd,
    float* __restrict__ wrkmemaux
);

// cross-covariance calculation for refinement
__global__ 
void CalcCCMatrices64Refined(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint maxnsteps,
    const int sfragstep,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem
);

// FindD02ThresholdsCCMRefined: efficiently find distance thresholds for the 
// inclusion of aligned positions for CCM and rotation matrix calculations 
// during the boundaries refinement of initially identified fragments;
template<int READCNST>
__global__
void FindD02ThresholdsCCMRefined(
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int sfragstep,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux
);

// CalcCCMatrices64RefinedExtended: calculate cross-covariance matrix 
// between the query and reference structures based on aligned positions 
// within given distance;
template<int READCNST>
__global__
void CalcCCMatrices64RefinedExtended(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int sfragstep,
    const float* __restrict__ tmpdpdiagbuffers,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem
);

// CalcScoresUnrlRefined: calculate/reduce scores for obtained 
// superpositions; version for fragment refinement; 
template<int SAVEPOS, int CHCKCONV>
__global__
void CalcScoresUnrlRefined(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int sfragstep,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers
);

// -------------------------------------------------------------------------

#endif//__covariance_refn_cuh__
