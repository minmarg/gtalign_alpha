/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __dpw_score_cuh__
#define __dpw_score_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gproc/btckcoords.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fields.cuh"
#include "libmycu/cudp/dpw_btck.cuh"

// =========================================================================
// ExecDPScore3264x: kernel for executing dynamic programming to calculate 
// max score completely in linear space (without backtracking) using 32x or 
// 64x unrolling
template<bool ANCHOR, bool BANDED, bool GAP0, bool CHECKCONV>
__global__
void ExecDPScore3264x(
    const uint blkdiagnum,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const float gapopencost,
    const float* __restrict__ wrkmemtm,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer
);

// =========================================================================

#endif//__dpw_score_cuh__
