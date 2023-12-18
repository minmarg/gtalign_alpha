/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __dpssw_btck_cuh__
#define __dpssw_btck_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gproc/btckcoords.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmymp/mpdp/mpdpbase.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fields.cuh"

// =========================================================================
// ExecDPSSwBtck3264x: execute dynamic programming with secondary 
// structure and backtracking information using shared memory and 
// 32(64)-fold unrolling along the diagonal of dimension CUDP_2DCACHE_DIM;
template <bool USESEQSCORING>
__global__
void ExecDPSSwBtck3264x(
    const uint blkdiagnum,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const float weight4ss,
    const float weight4rr,
    const float gapopencost,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
//     uint* __restrict__ maxscoordsbuf,
    char* __restrict__ btckdata
);


// =========================================================================
// -------------------------------------------------------------------------
// DPLocInitSS: initialize secondary structure
//
template<unsigned int SHFT = 0, char VALUE = pmvLOOP>
__device__ __forceinline__
void DPLocInitSS(char* __restrict__ ssCache)
{
    ssCache[threadIdx.x+SHFT] = VALUE;
}

template<unsigned int SHFT = 0, char VALUE = pmvLOOP>
__device__ __forceinline__
void DPLocInitSS(char& qss)
{
    qss = VALUE;
}

// -------------------------------------------------------------------------
// DPLocInitSS: initialize residue letters
//
template<unsigned int SHFT = 0, char VALUE = 0>
__device__ __forceinline__
void DPLocInitRsd(char* __restrict__ reCache)
{
    reCache[threadIdx.x+SHFT] = VALUE;
}

template<unsigned int SHFT = 0, char VALUE = 0>
__device__ __forceinline__
void DPLocInitRsd(char& qre)
{
    qre = VALUE;
}

// -------------------------------------------------------------------------
// DPLocCacheQrySS/DPLocCacheRfnSS: cache secondary structure assignment to 
// smem at position pos
//
__device__ __forceinline__
void DPLocCacheQrySS(char* __restrict__ ssCache, int pos)
{
    ssCache[threadIdx.x] = GetQuerySS(pos);
}

__device__ __forceinline__
void DPLocCacheQrySS(char& qss, int pos)
{
    qss = GetQuerySS(pos);
}

template<unsigned int SHFT = 0>
__device__ __forceinline__
void DPLocCacheRfnSS(char* __restrict__ ssCache, int pos)
{
    ssCache[threadIdx.x+SHFT] = GetDbStrSS(pos);
}

// -------------------------------------------------------------------------
// DPLocCacheQryRsd/DPLocCacheRfnRsd: cache residue letter to smem at pos
//
__device__ __forceinline__
void DPLocCacheQryRsd(char& qre, int pos)
{
    qre = GetQueryRsd(pos);
}

template<unsigned int SHFT = 0>
__device__ __forceinline__
void DPLocCacheRfnRsd(char* __restrict__ reCache, int pos)
{
    reCache[threadIdx.x+SHFT] = GetDbStrRsd(pos);
}

// =========================================================================

#endif//__dpssw_btck_cuh__
