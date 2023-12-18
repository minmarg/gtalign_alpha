/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __dpw_btck_specscore_cuh__
#define __dpw_btck_specscore_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gproc/btckcoords.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fields.cuh"

// =========================================================================
// score identifiers for DP: power spectral density scores, 
// spatial distance distribution scores
#define CUSA_DP_SCORES_PSD 0
#define CUSA_DP_SCORES_SPT 1
// =========================================================================
// =========================================================================
// ExecDPwBtckSpecScores3264x: kernel for executing dynamic programming with 
// backtracking using 32x or 64x unrolling
template<bool GAP0, int SCORES_TYPE>
__global__
void ExecDPwBtckSpecScores3264x(
    const uint blkdiagnum,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const float gapopencost,
    const float* __restrict__ specscores,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
//     uint* __restrict__ maxscoordsbuf,
    char* __restrict__ btckdata
);


// =========================================================================
// =========================================================================

#endif//__dpw_btck_specscore_cuh__
