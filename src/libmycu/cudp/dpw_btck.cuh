/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __dpw_btck_cuh__
#define __dpw_btck_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gproc/btckcoords.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fields.cuh"
#include "libmymp/mpdp/mpdpbase.h"

// =========================================================================
// ExecDPwBtck3264x: kernel for executing dynamic programming with 
// backtracking using 32x or 64x unrolling
template<bool ANCHOR, bool BANDED, bool GAP0, int D02IND, bool ALTSCTMS = false>
__global__
void ExecDPwBtck3264x(
    const uint blkdiagnum,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const uint stepnumber,
    const float gapopencost,
    const float* __restrict__ wrkmemtmibest,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
//     uint* __restrict__ maxscoordsbuf,
    char* __restrict__ btckdata
);


// =========================================================================
// -------------------------------------------------------------------------
// GetCoordsNdx: get 1D index of logically 2D coordinates array
//
__device__ __forceinline__
uint GetCoordsNdx(uint i, uint j) {return pmv2DNoElems * j + i;}

// -------------------------------------------------------------------------
// DPLocInitCoords: initialize coordinates
//
template<unsigned int SHFT = 0, int VALUE = 0>
__device__ __forceinline__
void DPLocInitCoords(float* __restrict__ coordsCache)
{
    coordsCache[GetCoordsNdx(pmv2DX,threadIdx.x+SHFT)] = (float)(VALUE);
    coordsCache[GetCoordsNdx(pmv2DY,threadIdx.x+SHFT)] = (float)(VALUE);
    coordsCache[GetCoordsNdx(pmv2DZ,threadIdx.x+SHFT)] = (float)(VALUE);
}

__device__ __forceinline__
void DPLocInitCoords(
    unsigned int SHFT, int VALUE,
    float* __restrict__ coordsCache)
{
    coordsCache[GetCoordsNdx(pmv2DX,threadIdx.x+SHFT)] = (float)(VALUE);
    coordsCache[GetCoordsNdx(pmv2DY,threadIdx.x+SHFT)] = (float)(VALUE);
    coordsCache[GetCoordsNdx(pmv2DZ,threadIdx.x+SHFT)] = (float)(VALUE);
}

template<unsigned int SHFT = 0, int VALUE = 0>
__device__ __forceinline__
void DPLocInitCoords(float& qx, float& qy, float& qz)
{
    qx = (float)(VALUE);
    qy = (float)(VALUE);
    qz = (float)(VALUE);
}

__device__ __forceinline__
void DPLocAssignCoords(
    unsigned int SHFTTRG, unsigned int SHFTSRC,
    float* __restrict__ coordsCache)
{
    coordsCache[GetCoordsNdx(pmv2DX,threadIdx.x+SHFTTRG)] =
    coordsCache[GetCoordsNdx(pmv2DX,threadIdx.x+SHFTSRC)];

    coordsCache[GetCoordsNdx(pmv2DY,threadIdx.x+SHFTTRG)] = 
    coordsCache[GetCoordsNdx(pmv2DY,threadIdx.x+SHFTSRC)];

    coordsCache[GetCoordsNdx(pmv2DZ,threadIdx.x+SHFTTRG)] =
    coordsCache[GetCoordsNdx(pmv2DZ,threadIdx.x+SHFTSRC)];
}

template<unsigned int CRD>
__device__ __forceinline__
float DPLocGetCoord(unsigned int SHFT, float* __restrict__ coordsCache)
{
    return coordsCache[GetCoordsNdx(CRD, SHFT)];
}

// -------------------------------------------------------------------------
// DPLocCacheQryCoords/DPLocCacheRfnCoords: cache coordinates to smem at 
// position pos
//
__device__ __forceinline__
void DPLocCacheQryCoords(float* __restrict__ coordsCache, int pos)
{
    coordsCache[GetCoordsNdx(pmv2DX,threadIdx.x)] = GetQueryCoord<pmv2DX>(pos);
    coordsCache[GetCoordsNdx(pmv2DY,threadIdx.x)] = GetQueryCoord<pmv2DY>(pos);
    coordsCache[GetCoordsNdx(pmv2DZ,threadIdx.x)] = GetQueryCoord<pmv2DZ>(pos);
}

__device__ __forceinline__
void DPLocCacheQryCoords(float& qx, float& qy, float& qz, int pos)
{
    qx = GetQueryCoord<pmv2DX>(pos);
    qy = GetQueryCoord<pmv2DY>(pos);
    qz = GetQueryCoord<pmv2DZ>(pos);
}

template<unsigned int SHFT = 0>
__device__ __forceinline__
void DPLocCacheRfnCoords(float* __restrict__ coordsCache, int pos)
{
    coordsCache[GetCoordsNdx(pmv2DX,threadIdx.x+SHFT)] = GetDbStrCoord<pmv2DX>(pos);
    coordsCache[GetCoordsNdx(pmv2DY,threadIdx.x+SHFT)] = GetDbStrCoord<pmv2DY>(pos);
    coordsCache[GetCoordsNdx(pmv2DZ,threadIdx.x+SHFT)] = GetDbStrCoord<pmv2DZ>(pos);
}

__device__ __forceinline__
void DPLocCacheRfnCoords(
    unsigned int SHFT, float* __restrict__ coordsCache, int pos)
{
    coordsCache[GetCoordsNdx(pmv2DX,threadIdx.x+SHFT)] = GetDbStrCoord<pmv2DX>(pos);
    coordsCache[GetCoordsNdx(pmv2DY,threadIdx.x+SHFT)] = GetDbStrCoord<pmv2DY>(pos);
    coordsCache[GetCoordsNdx(pmv2DZ,threadIdx.x+SHFT)] = GetDbStrCoord<pmv2DZ>(pos);
}


// =========================================================================
// -------------------------------------------------------------------------
// GetBufferNdx: get 1D index of logically 2D buffer array
template<unsigned int _2DCACHE_DIM_D>
__device__ __forceinline__
uint GetBufferNdx(uint i, uint j) {return _2DCACHE_DIM_D * i + j;}

__device__ __forceinline__
uint GetBufferNdx(unsigned int _2DCACHE_DIM_D, uint i, uint j) {return _2DCACHE_DIM_D * i + j;}

// -------------------------------------------------------------------------
// DPLocGetCacheVal/DPLocSetCacheVal: Get/SET methods
//
template<unsigned int LOG2MINDDIM = 5>
__device__ __forceinline__
float DPLocGetCacheVal(
    unsigned int DIM_PD,
    unsigned int SBCT,
    unsigned int SHFT,
    const float* __restrict__ scmCache)
{
    return scmCache[GetBufferNdx(DIM_PD, SBCT, SHFT+(SHFT>>LOG2MINDDIM))];
}

template<unsigned int LOG2MINDDIM = 5>
__device__ __forceinline__
void DPLocSetCacheVal(
    unsigned int DIM_PD,
    unsigned int SBCT,
    unsigned int SHFT,
    float* __restrict__ scmCache,
    float value)
{
    scmCache[GetBufferNdx(DIM_PD, SBCT, SHFT+(SHFT>>LOG2MINDDIM))] = value;
}

// -------------------------------------------------------------------------
// DPLocInitCache: initialize a DP cache buffer;
// SHFT, shift of inner-most index for writing values;
//
template<
    unsigned int _2DCACHE_DIM_D,
    unsigned int SHFT = 0,
    int NSUBCTS = nTDPDiagScoreSubsections>
__device__ __forceinline__
void DPLocInitCache( 
    float* __restrict__ scmCache,
    float initmmval = 0.0f,
    float initval = 0.0f)
{
    scmCache[GetBufferNdx<_2DCACHE_DIM_D>(dpdsssStateMM, threadIdx.x+SHFT)] = initmmval;
    #pragma unroll
    for(int i = dpdsssStateMM+1; i < NSUBCTS; i++)
        scmCache[GetBufferNdx<_2DCACHE_DIM_D>(i, threadIdx.x+SHFT)] = initval;
}

template<int NSUBCTS = nTDPDiagScoreSubsections>
__device__ __forceinline__
void DPLocInitCache512x( 
    unsigned int DIM_PD,
    unsigned int SHFT,
    float* __restrict__ scmCache,
    float initmmval = 0.0f,
    float initval = 0.0f)
{
    DPLocSetCacheVal(DIM_PD, dpdsssStateMM, threadIdx.x+SHFT, scmCache, initmmval);
    for(int i = dpdsssStateMM+1; i < NSUBCTS; i++)
        DPLocSetCacheVal(DIM_PD, i, threadIdx.x+SHFT, scmCache, initval);
}

// -------------------------------------------------------------------------
// DPLocCacheBuffer: cache one of the DP buffers;
// SHFT, shift of inner-most index for writing values;
// scmCache, SMEM cache;
// gmemtmpbuffer, address of the buffer to read data from;
// x, x position at which the buffer has to be read;
// y, initial y position in the buffer;
// stride, stride to refer to the same x position in the buffer;
//
template<
    unsigned int _2DCACHE_DIM_D,
    int SHFT = 0,
    int NSUBCTS = nTDPDiagScoreSubsections>
__device__ __forceinline__
void DPLocCacheBuffer( 
    float* __restrict__ scmCache,
    const float* __restrict__ gmemtmpbuffer,
    int x, int y, int stride)
{
    #pragma unroll
    for(int i = 0; i < NSUBCTS; i++, y += stride)
        scmCache[GetBufferNdx<_2DCACHE_DIM_D>(i, threadIdx.x+SHFT)] =
            gmemtmpbuffer[y + x];
}

template<int NSUBCTS = nTDPDiagScoreSubsections>
__device__ __forceinline__
void DPLocCacheBuffer( 
    unsigned int DIM_PD,
    unsigned int SHFT,
    float* __restrict__ scmCache,
    const float* __restrict__ gmemtmpbuffer,
    int x, int y, int stride)
{
    for(int i = 0; i < NSUBCTS; i++, y += stride)
        DPLocSetCacheVal(DIM_PD, i, threadIdx.x+SHFT, scmCache, gmemtmpbuffer[y + x]);
}

// -------------------------------------------------------------------------
// DPLocWriteBuffer: write one of the cahced DP buffers back to GMEM;
// SHFT, shift of inner-most index of the cache;
// scmCache, SMEM cache;
// gmemtmpbuffer, address of the buffer to write data to;
// x, x position at which the data has to be written;
// y, initial y position in the buffer;
// stride, stride to refer to the same x position in the buffer;
//
template<
    unsigned int _2DCACHE_DIM_D,
    int SHFT = 0,
    int NSUBCTS = nTDPDiagScoreSubsections>
__device__ __forceinline__
void DPLocWriteBuffer( 
    const float* __restrict__ scmCache,
    float* __restrict__ gmemtmpbuffer,
    int x, int y, int stride)
{
    #pragma unroll
    for(int i = 0; i < NSUBCTS; i++, y += stride) {
        gmemtmpbuffer[y + x] =
            scmCache[GetBufferNdx<_2DCACHE_DIM_D>(i, threadIdx.x+SHFT)];
    }
}

template<int NSUBCTS = nTDPDiagScoreSubsections>
__device__ __forceinline__
void DPLocWriteBuffer( 
    unsigned int DIM_PD,
    unsigned int SHFT,
    const float* __restrict__ scmCache,
    float* __restrict__ gmemtmpbuffer,
    int x, int y, int stride)
{
    for(int i = 0; i < NSUBCTS; i++, y += stride) {
        gmemtmpbuffer[y + x] =
            DPLocGetCacheVal(DIM_PD, i, threadIdx.x+SHFT, scmCache);
    }
}

// =========================================================================
// -------------------------------------------------------------------------
// DPLocGetBtckCacheVal/DPLocSetBtckCacheVal: Get/SET methods for 
// backtracking cache;
// MINDDIM1, innermost dimension with padding;
//
__device__ __forceinline__
char DPLocGetBtckCacheVal(
    unsigned int MINDDIM1,
    unsigned int Y,
    unsigned int X,
    const char* __restrict__ btckCache)
{
    return btckCache[GetBufferNdx(MINDDIM1, Y, X)];
}

__device__ __forceinline__
void DPLocSetBtckCacheVal(
    unsigned int MINDDIM1,
    unsigned int Y,
    unsigned int X,
    char* __restrict__ btckCache,
    char value)
{
    btckCache[GetBufferNdx(MINDDIM1, Y, X)] = value;
}

// =========================================================================
// -------------------------------------------------------------------------
// dpmaxandcoords: update maximum value along with the coordinates of this 
// value
template<typename T>
__device__ __forceinline__ 
void dpmaxandcoords(T& a, T b, uint& xy, uint x, uint y)
{
    //NOTE: for (semi-)global alignment, record match scores unconditionally,
    //as opposed to local alignment where maximum scores are recorded:
//     if(a < b) {
        a = b;
        xy = CombineCoords(x,y);
//     }
}


// =========================================================================
// -------------------------------------------------------------------------
// CellXYInvalidLowerArea: return true if cell (y,x) (y, query; x, 
// reference pos.) is in the invalid lower region (not explored);
// ANCHORRGN, template parameter for considering the boundaries 
// implied by the anchor region;
// x, position along the reference (db) direction;
// y, position along the query direction;
// 
template<bool ANCHORRGN, bool BANDED>
__device__ __forceinline__ 
bool CellXYInvalidLowerArea(
    int x, int y,
    int qrylen, int dbstrlen,
    int qrypos, int rfnpos, int fraglen)
{
    if(ANCHORRGN)
        //bottom-left region
        if(qrypos + fraglen <= y && x < rfnpos)
            return true;

    if(BANDED) {
        int delta = rfnpos - qrypos;
        //lower triangle implied by the band (origin: upper-left corner);
        //equation of the lower line: x-delta=y-b;
        //b, bandwidth/2; delta=x0-y0, where (x0,y0) is the middle point;
        if(x - delta + CUDP_BANDWIDTH_HALF <= y)
            return true;
    }

    return false;
}

// CellXYInvalidUpperArea: return true if cell (y,x) (y, query; x, 
// reference pos.) is in the invalid upper region (not explored);
// ANCHORRGN, template parameter for considering the boundaries 
// implied by the anchor region;
// x, position along the reference (db) direction;
// y, position along the query direction;
// 
template<bool ANCHORRGN, bool BANDED>
__device__ __forceinline__ 
bool CellXYInvalidUpperArea(
    int x, int y,
    int qrylen, int dbstrlen,
    int qrypos, int rfnpos, int fraglen)
{
    if(ANCHORRGN)
        //upper-right region
        if(y < qrypos && rfnpos + fraglen <= x)
            return true;

    if(BANDED) {
        int delta = rfnpos - qrypos;
        //upper triangle implied by the band (origin: upper-left corner);
        //equation of the upper line: x-delta=y+b;
        //b, bandwidth/2; delta=x0-y0, where (x0,y0) is the middle point;
        if(y <= x - delta - CUDP_BANDWIDTH_HALF)
            return true;
    }

    return false;
}

// =========================================================================

#endif//__dpw_btck_cuh__
