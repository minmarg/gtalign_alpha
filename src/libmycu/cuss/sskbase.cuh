/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __sskbase_cuh__
#define __sskbase_cuh__

#include "libmymp/mpss/MpSecStrBase.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/custages/transform.cuh"
#include "libmycu/custages/fields.cuh"

// =========================================================================
// template parameter values of kernel CalcSecStrs: query and reference 
// structures
#define SSK_STRUCTS_QRIES 0
#define SSK_STRUCTS_REFNS 1

// =========================================================================
// -------------------------------------------------------------------------
// SSKCacheCoords: cache coordinates at structure position strpos to 
// destination position dstpos of strCoords;
// dstpos, destination position of strCoords;
// strpos, structure position to read the coordinates at;
//
template<int STRUCTS>
__device__ __forceinline__
void SSKCacheCoords(
    float (*__restrict__ strCoords)[pmv2DNoElems],
    int dstpos, int strpos)
{
    if(STRUCTS == SSK_STRUCTS_QRIES) {
        strCoords[dstpos][pmv2DX] = GetQueryCoord<pmv2DX>(strpos);
        strCoords[dstpos][pmv2DY] = GetQueryCoord<pmv2DY>(strpos);
        strCoords[dstpos][pmv2DZ] = GetQueryCoord<pmv2DZ>(strpos);
    } else {
        strCoords[dstpos][pmv2DX] = GetDbStrCoord<pmv2DX>(strpos);
        strCoords[dstpos][pmv2DY] = GetDbStrCoord<pmv2DY>(strpos);
        strCoords[dstpos][pmv2DZ] = GetDbStrCoord<pmv2DZ>(strpos);
    }
}

// SSKCacheCoordsYPrl: cache coordinates at structure position strpos to 
// destination position dstpos of strCoords using y dimension for parallelism;
// dstpos, destination position of strCoords;
// strpos, structure position to read the coordinates at;
template<int STRUCTS, int YDIMSHIFT>
__device__ __forceinline__
void SSKCacheCoordsYPrl(
    float (*__restrict__ strCoords)[pmv2DNoElems],
    int dstpos, int strpos)
{
    if(STRUCTS == SSK_STRUCTS_QRIES) {
        if(threadIdx.y == 0 + YDIMSHIFT) strCoords[dstpos][pmv2DX] = GetQueryCoord<pmv2DX>(strpos);
        if(threadIdx.y == 1 + YDIMSHIFT) strCoords[dstpos][pmv2DY] = GetQueryCoord<pmv2DY>(strpos);
        if(threadIdx.y == 2 + YDIMSHIFT) strCoords[dstpos][pmv2DZ] = GetQueryCoord<pmv2DZ>(strpos);
    } else {
        if(threadIdx.y == 0 + YDIMSHIFT) strCoords[dstpos][pmv2DX] = GetDbStrCoord<pmv2DX>(strpos);
        if(threadIdx.y == 1 + YDIMSHIFT) strCoords[dstpos][pmv2DY] = GetDbStrCoord<pmv2DY>(strpos);
        if(threadIdx.y == 2 + YDIMSHIFT) strCoords[dstpos][pmv2DZ] = GetDbStrCoord<pmv2DZ>(strpos);
    }
}

// -------------------------------------------------------------------------
// SSKCacheRsds: cache residues at structure position strpos to destination
// position dstpos of rsdCache;
// dstpos, destination position of rsdCache;
// strpos, structure position to read a residue;
template<int STRUCTS, int YDIMNDX>
__device__ __forceinline__
void SSKCacheRsds(
    char* __restrict__ rsdCache,
    int dstpos, int strpos)
{
    if(STRUCTS == SSK_STRUCTS_QRIES) {
        if(threadIdx.y == YDIMNDX) rsdCache[dstpos] = GetQueryRsd(strpos);
    } else {
        if(threadIdx.y == YDIMNDX) rsdCache[dstpos] = GetDbStrRsd(strpos);
    }
}

// -------------------------------------------------------------------------
// SSKGetDistance: calculate distances between two residues in the same 
// sequence;
// strCoords, cached coordinates;
// dstpos, destination position;
// seqpos, position several residues apart;
//
__device__ __forceinline__
float SSKGetDistance(
    const float (*__restrict__ strCoords)[pmv2DNoElems],
    int dstpos, int seqpos)
{
    return sqrtf(distance2(
        strCoords[dstpos][pmv2DX], strCoords[dstpos][pmv2DY], strCoords[dstpos][pmv2DZ],
        strCoords[seqpos][pmv2DX], strCoords[seqpos][pmv2DY], strCoords[seqpos][pmv2DZ]
    ));
}

// SSKGetDistance: calculate distances between two residues of the same 
// sequence represented by different buffers;
// strCoords1, strCoords2, cached coordinates;
// pos1, pos2, first/second positions;
//
__device__ __forceinline__
float SSKGetDistance(
    const float (*__restrict__ strCoords1)[pmv2DNoElems],
    const float (*__restrict__ strCoords2)[pmv2DNoElems],
    int pos1, int pos2)
{
    return sqrtf(distance2(
        strCoords1[pos1][pmv2DX], strCoords1[pos1][pmv2DY], strCoords1[pos1][pmv2DZ],
        strCoords2[pos2][pmv2DX], strCoords2[pos2][pmv2DY], strCoords2[pos2][pmv2DZ]
    ));
}

// =========================================================================

#endif//__sskbase_cuh__
