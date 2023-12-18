/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __dpsslocal_cuh__
#define __dpsslocal_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"

// =========================================================================
// ExecDPSSLocal3264x: execute dynamic programming for local alignment using
// secondary structure information with 32(64)-fold unrolling along the 
// diagonal of dimension CUDP_2DCACHE_DIM;
__global__
void ExecDPSSLocal3264x(
    const uint blkdiagnum,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const float gapcost,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
    char* __restrict__ dpscoremtx
);

// =========================================================================
// =========================================================================

#endif//__dpsslocal_cuh__
