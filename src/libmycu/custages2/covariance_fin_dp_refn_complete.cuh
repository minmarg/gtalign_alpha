/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __covariance_fin_dp_refn_complete_cuh__
#define __covariance_fin_dp_refn_complete_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fragment.cuh"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/covariance.cuh"
#include "libmycu/custages/covariance_plus.cuh"
#include "libmycu/custages/covariance_dp_refn.cuh"
#include "libmycu/custages/transform.cuh"
#include "libmycu/custages/scoring.cuh"

// =========================================================================
// FinalFragmentBasedDPAlignmentRefinement: perform final alignment 
// refinement based on the best superposition obtained in the course of 
// complete superposition search within the single kernel's actions;
//
template<bool D0FINAL, int CHCKDST, bool TFM_DINV>
__global__ 
void FinalFragmentBasedDPAlignmentRefinementPhase1(
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
// FinalFragmentBasedDPAlignmentRefinementPhase2: phase 2 to perform 
// finer-scale refinement of the the best superposition obtained 
// within the single kernel's actions;
template<bool D0FINAL, int CHCKDST, bool TFM_DINV>
__global__ 
void FinalFragmentBasedDPAlignmentRefinementPhase2(
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

// FinalFragmentBasedDPAlignmentRefinementPhase2_fullsearch: phase 2 to
// perform finer-scale refinement of the the best superposition obtained 
// within the single kernel's actions;
// this version performs full search of maxnfragfcts positions from the 
// identified one in phase 1 for each fragment length;
template<bool D0FINAL, int CHCKDST, bool TFM_DINV>
__global__ 
void FinalFragmentBasedDPAlignmentRefinementPhase2_fullsearch(
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

// -------------------------------------------------------------------------

#endif//__covariance_fin_dp_refn_complete_cuh__
