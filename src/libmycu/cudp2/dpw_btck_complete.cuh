/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __dpw_btck_complete_cuh__
#define __dpw_btck_complete_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gproc/btckcoords.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fields.cuh"

// =========================================================================
// =========================================================================
// ExecCompleteDPwBtck512x: kernel for executing complete dynamic 
// programming with backtracking using 512x unrolling
template<bool ANCHOR, bool BANDED, bool GAP0, int D02IND>
__global__
void ExecCompleteDPwBtck512x(
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const uint stepnumber,
    const float gapopencost,
    const float* __restrict__ wrkmemtmibest,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpbotbuffer,
//     uint* __restrict__ maxscoordsbuf,
    char* __restrict__ btckdata
);


// =========================================================================
// =========================================================================

#endif//__dpw_btck_complete_cuh__
