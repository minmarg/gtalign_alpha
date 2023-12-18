/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __dpssw_tfm_btck_cuh__
#define __dpssw_tfm_btck_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fields.cuh"
#include "libmycu/cudp/dpw_btck.cuh"

// =========================================================================
// ExecDPTFMSSwBtck3264x: execute dynamic programming with secondary
// structure and backtracking information using shared memory and
// 32(64)-fold unrolling along the diagonal of dimension CUDP_2DCACHE_DIM;
template<bool GLOBTFM, bool GAP0, bool USESS, int D02IND = D02IND_SEARCH>
__global__
void ExecDPTFMSSwBtck3264x(
    const uint blkdiagnum,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const uint stepnumber,
    const float ssweight,
    const float gapopencost,
    const float* __restrict__ tfmmem,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
//     uint* __restrict__ maxscoordsbuf,
    char* __restrict__ btckdata
);

// =========================================================================

#endif//__dpssw_tfm_btck_cuh__
