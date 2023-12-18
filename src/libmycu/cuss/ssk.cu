/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/custages/fields.cuh"
#include "ssk.cuh"

// #define CUSS_SECSTR_TESTPRINT 0

// -------------------------------------------------------------------------
// CalcSecStrs: calculate secondary structures for all query OR reference 
// structures in the chunk;
// NOTE: thread block is 1D and processes fragment along structure 
// positions;
// STRUCTS, template parameter indicating the use of either the queries or 
// reference structures;
// NOTE: memory pointers should be aligned!
// NOTE: keep #registers below 32!
// 
template<int STRUCTS>
__global__
void CalcSecStrs()
{
    // blockIdx.x is the block index of positions;
    // blockIdx.y is the serial number of query or reference structure;
    enum{   nMRG0 = 2,//margin from one end
            nMRG2 = 4,//total margin: two from both ends
            nMAXPOS = CUSS_CALCSTR_XDIM + nMRG2//#max positions in the cache
    };
    //cache for coordinates: 
    __shared__ float strCoords[nMAXPOS][pmv2DNoElems];
    //cache for distances along str. positions: two, three, four residues apart
    //no bank conflicts as long as inner-most dimension is appropriately odd
    __shared__ float dstCache[nMAXPOS][cssTotal];
    //relative position index:
    const int ndx0 = blockIdx.x * blockDim.x;
    const int ndx = ndx0 + threadIdx.x;
    int strlen;//structure length
    int strdst;//distance in positions to the beginning of structures

    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse ccmCache
    if(threadIdx.x < 2) {
        if(STRUCTS == SSK_STRUCTS_QRIES)
                GetQueryLenDst(blockIdx.y, (int*)(strCoords[0]));
        else    GetDbStrLenDst(blockIdx.y, (int*)(strCoords[0]));
    }

    __syncthreads();

    //NOTE: no bank conflict when two threads from the same warp access the same address;
    strlen = ((int*)(strCoords[0]))[0];
    strdst = ((int*)(strCoords[0]))[1];

    __syncthreads();

    //all threads in the block exit if thread 0 is out of bounds
    if(strlen <= ndx0) return;

    bool cond = //condition of whether the first margin is valid for calculation
        (threadIdx.x < nMRG0 &&
        0 <= ndx0-(int)threadIdx.x-1 && ndx0-(int)threadIdx.x-1 < strlen);


    //cache coordinates
    if(ndx < strlen)
        SSKCacheCoords<STRUCTS>(
            strCoords, threadIdx.x + nMRG0/*dstpos*/, strdst + ndx/*strpos*/);

    //cache additional coordinates for two positions at both ends
    if(cond)
        SSKCacheCoords<STRUCTS>(
            strCoords, nMRG0-threadIdx.x-1/*dstpos*/, strdst + ndx0-threadIdx.x-1);

    if(CUSS_CALCSTR_XDIM <= threadIdx.x+nMRG0 && ndx0+threadIdx.x+nMRG0 < strlen)
        SSKCacheCoords<STRUCTS>(
            strCoords, nMRG0+threadIdx.x+nMRG0/*dstpos*/, strdst + ndx0+threadIdx.x+nMRG0);

    __syncthreads();

    //calculate distances
    if(ndx < strlen)
        SSKCalcDistances<nMAXPOS>(dstCache, strCoords, threadIdx.x + nMRG0/*dstpos*/);

    //calculate distances at one of the boundaries (beginning)
    if(cond)
        SSKCalcDistances<nMAXPOS>(dstCache, strCoords, nMRG0-threadIdx.x-1/*dstpos*/);

    __syncthreads();

    //assign secondary structure
    char ss = pmvLOOP;//loop/coil

    cond = (ndx0+threadIdx.x+nMRG0 < strlen) && ((threadIdx.x < nMRG0)? cond: 1);

    if(cond) ss = SSKAassignSecStr<nMRG0>(dstCache, threadIdx.x);

    //write secondary structure to gmem
    if(ndx < strlen) {
        if(STRUCTS == SSK_STRUCTS_QRIES)
                SetQuerySecStr(strdst + ndx, ss);
        else    SetDbStrSecStr(strdst + ndx, ss);
    }

#ifdef CUSS_SECSTR_TESTPRINT
    if(blockIdx.y==CUSS_SECSTR_TESTPRINT && threadIdx.x==0) {
        char ssbuf[CUSS_CALCSTR_XDIM+1];
        ssbuf[CUSS_CALCSTR_XDIM] = 0;
        for(int ii=0;ii<CUSS_CALCSTR_XDIM;ii++)
            ssbuf[ii] = 
                (STRUCTS == SSK_STRUCTS_QRIES)
                ? GetQuerySS(strdst+ndx0+ii)
                : GetDbStrSS(strdst+ndx0+ii);
        printf(" ss_frag: %d - %d\n'%s'\n\n",
            ndx0, ndx0+blockDim.x-1, ssbuf);
    }
#endif
}

// =========================================================================
// Instantiations
//
#define INSTANTIATE_CalcSecStrs(tpSTRUCTS) \
    template \
    __global__ void CalcSecStrs<tpSTRUCTS>();

INSTANTIATE_CalcSecStrs(SSK_STRUCTS_QRIES);
INSTANTIATE_CalcSecStrs(SSK_STRUCTS_REFNS);

// -------------------------------------------------------------------------
