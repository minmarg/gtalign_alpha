/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __ssk_cuh__
#define __ssk_cuh__

#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/custages/transform.cuh"
#include "libmycu/custages/fields.cuh"

// =========================================================================
// template parameter values of kernel CalcSecStrs: query and reference 
// structures
#define SSK_STRUCTS_QRIES 0
#define SSK_STRUCTS_REFNS 1

//indices for distances along str. positions: two, three, four residues apart
enum {css2RESdst, css3RESdst, css4RESdst, cssTotal};

// =========================================================================
// CalcSecStrs: calculate secondary structures for all query OR reference 
// structures in the chunk;
template<int STRUCTS>
__global__
void CalcSecStrs();

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

// -------------------------------------------------------------------------
// SSKCalcDistances: calculate distances between residues several residues 
// apart in the sequence of structures;
// nMAXPOS, template parameter, outermost dimension of dstCache and strCoords;
// dstCache, cache to hold calculated distances;
// strCoords, cached coordinates;
// dstpos, destination position of arrays;
//
template<int nMAXPOS>
__device__ __forceinline__
void SSKCalcDistances(
    float (*__restrict__ dstCache)[cssTotal],
    const float (*__restrict__ strCoords)[pmv2DNoElems],
    int dstpos)
{
    if(dstpos+2 < nMAXPOS) dstCache[dstpos][css2RESdst] = SSKGetDistance(strCoords, dstpos, dstpos+2);
    if(dstpos+3 < nMAXPOS) dstCache[dstpos][css3RESdst] = SSKGetDistance(strCoords, dstpos, dstpos+3);
    if(dstpos+4 < nMAXPOS) dstCache[dstpos][css4RESdst] = SSKGetDistance(strCoords, dstpos, dstpos+4);
}

// -------------------------------------------------------------------------
// SSKAassignSecStr: assign secondary structure for given position once 
// the required distances have been calculated;
// nMRG0, template parameter, margin of positions for dstCache;
// dstCache, cache of calculated distances;
// trgpos, position to calculate secondary structure for;
// NOTE: trgpos corresponds to the structure position, hence
// NOTE: dstCache position is nMRG0 greater;
// NOTE: trgpos assumed to be valid;
// NOTE: constants as observed in tmalign;
//
template<int nMRG0>
__device__ __forceinline__
char SSKAassignSecStr(
    const float (*__restrict__ dstCache)[cssTotal],
    int trgpos)
{
    //NOTE: calculating for position 3 of 5 consecutive position
    float d13 = dstCache[trgpos][css2RESdst];
    float d14 = dstCache[trgpos][css3RESdst];
    float d15 = dstCache[trgpos][css4RESdst];
    float d24 = dstCache[trgpos+1][css2RESdst];
    float d25 = dstCache[trgpos+1][css3RESdst];
    float d35 = dstCache[trgpos+2][css2RESdst];

    float error = 2.1f;

    if(fabsf(d15-6.37f) < error && fabsf(d14-5.18f) < error &&
       fabsf(d25-5.18f) < error && fabsf(d13-5.45f) < error &&
       fabsf(d24-5.45f) < error && fabsf(d35-5.45f) < error)
        return pmvHELIX;//helix

    error = 1.42f;

    if(fabsf(d15-13.0f) < error && fabsf(d14-10.4f) < error &&
       fabsf(d25-10.4f) < error && fabsf(d13-6.1f) < error &&
       fabsf(d24-6.1f) < error && fabsf(d35-6.1f) < error)
        return pmvSTRND;//strand

    if(d15 < 8.0f) return pmvTURN;//turn

    return pmvLOOP;//loop/coil
}

// =========================================================================

#endif//__ssk_cuh__
