/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __dpagc_ssw_btck_specscore_cuh__
#define __dpagc_ssw_btck_specscore_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gproc/btckcoords.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fields.cuh"
#include "libmycu/cudp/dpssw_btck.cuh"
#include "libmycu/cudp/dpw_btck_specscore.cuh"

// =========================================================================
// =========================================================================
// ExecDPAGCSSwBtckSpecScores3264x: kernel for executing dynamic programming 
// with affine gap cost model, secondary structure and backtracking 
// information using 32x or 64x unrolling
template<bool USESS>
__global__
void ExecDPAGCSSwBtckSpecScores3264x(
    const uint blkdiagnum,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const float ssweight,
    const float gapopencost,
    const float gapextcost,
    const float* __restrict__ specscores,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
//     uint* __restrict__ maxscoordsbuf,
    char* __restrict__ btckdata
);


// =========================================================================
// =========================================================================

#endif//__dpagc_ssw_btck_specscore_cuh__
