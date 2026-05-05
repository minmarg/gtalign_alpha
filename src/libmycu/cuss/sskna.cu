/***************************************************************************
 *   Copyright (C) 2021-2025 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cucom/mymutex.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/custages/fields.cuh"
#include "sskna.cuh"

// #define CUSS_NASECSTR_TESTPRINT 0

// -------------------------------------------------------------------------
// CalcNASecStrs: calculate secondary structures for all query OR reference 
// nucleic acid structures in the chunk;
// ndbCposs, total number of reference positions in the chunk;
// tmpdpdiagbuffers, temporary buffer for distances and positions;
// 
template<int STRUCTS>
__global__ void CalcSecStrs_NASS(
    const uint ndbCposs,
    float* __restrict__ tmpdpdiagbuffers)
{
    enum{   lXDIM = CUSS_NACALCSTR_XDIM
    };
    // blockIdx.x is the serial number of query or reference structure;
    const int strndx = blockIdx.x;
    __shared__ int strCache[3];
    int strlen;//structure length
    int strdst;//distance in positions to the beginning of structures

    //reuse ccmCache
    if(threadIdx.x == 0) {
        if(STRUCTS == SSK_STRUCTS_QRIES)
                strCache[0] =  GetQueryLength(strndx);
        else    strCache[0] =  GetDbStrLength(strndx);
    }

    if(threadIdx.x == 32) {//next warp
        int strdst0;
        if(STRUCTS == SSK_STRUCTS_QRIES) {
            strCache[1] = strdst0 = GetQueryDst(strndx);
            strCache[2] = GetQueryStrField<INTYPE,pmv2D_Ins_Ch_Ord>(strdst0);
        } else {
            strCache[1] = strdst0 = GetDbStrDst(strndx);
            strCache[2] = GetDbStrField<INTYPE,pmv2D_Ins_Ch_Ord>(strdst0);
        }
    }

    __syncthreads();

    //NOTE: no bank conflict when two threads from the same warp access the same address;
    strlen = strCache[0];
    strdst = strCache[1];

    {
        int type = strCache[2];
        if(GetMoleculeType(type) != gtmtNA) return;
    }

    //NOTE: no sync as long as the cache is not overwritten below!
    // __syncthreads();

    for(int pos = threadIdx.x; pos < strlen; pos += lXDIM) {
        if(STRUCTS == SSK_STRUCTS_QRIES)
                SetQuerySecStr(strdst + pos, pmnasUNPAIRED);
        else    SetDbStrSecStr(strdst + pos, pmnasUNPAIRED);
    }

    __syncthreads();

    for(int pos = threadIdx.x; pos < strlen; pos += lXDIM) {
        //position paired with:
        int paired = tmpdpdiagbuffers[strdst + pos + ndbCposs * lTMPBUFFNDX_POSIT];//int<-float
        if(0 <= paired && paired < strlen && pos < paired) {
            //verifying reciprocal pairing:
            int paired_rcp = tmpdpdiagbuffers[strdst + paired + ndbCposs * lTMPBUFFNDX_POSIT];
            //if pairing is mutually consistent:
            if(paired_rcp == pos) {
                if(STRUCTS == SSK_STRUCTS_QRIES) {
                    SetQuerySecStr(strdst + pos, pmnasOPEN);
                    SetQuerySecStr(strdst + paired, pmnasCLOSE);
                } else {
                    SetDbStrSecStr(strdst + pos, pmnasOPEN);
                    SetDbStrSecStr(strdst + paired, pmnasCLOSE);
                }
            }
        }
    }
}

// =========================================================================
// Instantiations
//
#define INSTANTIATE_CalcSecStrs_NASS(tpSTRUCTS) \
    template __global__ void CalcSecStrs_NASS<tpSTRUCTS>(\
        const uint ndbCposs, \
        float* __restrict__ tmpdpdiagbuffers);

INSTANTIATE_CalcSecStrs_NASS(SSK_STRUCTS_QRIES);
INSTANTIATE_CalcSecStrs_NASS(SSK_STRUCTS_REFNS);

// -------------------------------------------------------------------------
// Initialize_NASS: initialize temporary memory buffer for calculating
// nucleic acid secondary structures
// 
__global__
void Initialize_NASS(
    const uint ndbCposs,
    float* __restrict__ tmpdpdiagbuffers)
{
    const uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    if(pos < ndbCposs) {
        if(threadIdx.y == lTMPBUFFNDX_POSIT) tmpdpdiagbuffers[pos + ndbCposs * lTMPBUFFNDX_POSIT] = -1;
        if(threadIdx.y == lTMPBUFFNDX_DEVIA) tmpdpdiagbuffers[pos + ndbCposs * lTMPBUFFNDX_DEVIA] = 9999.9f;
        if(threadIdx.y == lTMPBUFFNDX_MUTEX) tmpdpdiagbuffers[pos + ndbCposs * lTMPBUFFNDX_MUTEX] = 0;
    }
}

// -------------------------------------------------------------------------
// CalcDistances_NASS: calculate relevant pairwise distances between
// residues for all query OR reference nucleic acid structures in the chunk;
// atomtype, nucleic acid atom type processed;
// ndbCposs, total number of reference positions in the chunk;
// tmpdpdiagbuffers, temporary buffer for distances and positions;
// 
template<int STRUCTS>
__global__
void CalcDistances_NASS(
    const int atomtype,
    const uint ndbCposs,
    float* __restrict__ tmpdpdiagbuffers)
{
    enum{lXYDIM = CUSS_NARESDST_XYDIM};
    // blockIdx.x/y are block indices of positions;
    // blockIdx.z is the serial number of query or reference structure;
    const int strndx = blockIdx.z;
    //cache for coordinates: 
    __shared__ float strCoordsydim[lXYDIM][pmv2DNoElems];
    __shared__ float strCoordsxdim[lXYDIM][pmv2DNoElems];
    //cache for most relevant distances and positions
    __shared__ float dstCache[lXYDIM+1];
    __shared__ int posCache[lXYDIM+1];
    __shared__ char rsdCacheydim[lXYDIM+1];
    __shared__ char rsdCachexdim[lXYDIM+1];
    //relative position indices:
    const int ndx0ydim = blockIdx.y * blockDim.y;
    int rowy = ndx0ydim + threadIdx.y;//row index
    int strlen;//structure length
    int strdst;//distance in positions to the beginning of structures

    //reuse ccmCache
    if(threadIdx.y == 0 && threadIdx.x == 0) {
        if(STRUCTS == SSK_STRUCTS_QRIES)
                ((int*)(strCoordsydim[0]))[0] =  GetQueryLength(strndx);
        else    ((int*)(strCoordsydim[0]))[0] =  GetDbStrLength(strndx);
    }

    if(threadIdx.y == 1 && threadIdx.x == 0) {//next warp
        int strdst0;
        if(STRUCTS == SSK_STRUCTS_QRIES) {
            ((int*)(strCoordsydim[0]))[1] = strdst0 = GetQueryDst(strndx);
            ((int*)(strCoordsydim[0]))[2] = GetQueryStrField<INTYPE,pmv2D_Ins_Ch_Ord>(strdst0);
        } else {
            ((int*)(strCoordsydim[0]))[1] = strdst0 = GetDbStrDst(strndx);
            ((int*)(strCoordsydim[0]))[2] = GetDbStrField<INTYPE,pmv2D_Ins_Ch_Ord>(strdst0);
        }
    }

    __syncthreads();

    //NOTE: no bank conflict when two threads from the same warp access the same address;
    strlen = ((int*)(strCoordsydim[0]))[0];
    strdst = ((int*)(strCoordsydim[0]))[1];

    {
        int type = ((int*)(strCoordsydim[0]))[2];
        if(GetMoleculeType(type) != gtmtNA) return;
    }

    __syncthreads();

    //exit upon out-of-bounds condition:
    if(strlen <= ndx0ydim) return;


    if(threadIdx.y == 12) dstCache[threadIdx.x] = 9999.9f;//too large deviation
    if(threadIdx.y == 13) posCache[threadIdx.x] = -1;

    //cache coordinates along y dim
    if(ndx0ydim + threadIdx.x < strlen)
        SSKCacheCoordsYPrl<STRUCTS,0/*YDIMSHIFT*/>(
            strCoordsydim, threadIdx.x, strdst + ndx0ydim + threadIdx.x/*strpos*/);

    //cache residues along y dim
    if(ndx0ydim + threadIdx.x < strlen)
        SSKCacheRsds<STRUCTS,TIMES3(pmv2DNoElems)/*YDIMNDX*/>(
            rsdCacheydim, threadIdx.x, strdst + ndx0ydim + threadIdx.x/*strpos*/);


    float dev_pair = 9999.9f;
    int col_pair = 0;
    const int lid = threadIdx.x & 0x1f;//lane id

    for(int col0 = 0; col0 < strlen; col0 += blockDim.x) {
        int col1 = col0 + threadIdx.x;

        if(col1 < strlen)//cache coordinates along x dim:
            SSKCacheCoordsYPrl<STRUCTS,pmv2DNoElems/*YDIMSHIFT*/>(
                strCoordsxdim, threadIdx.x, strdst + col1);

        if(col1 < strlen)//cache residues along x dim:
            SSKCacheRsds<STRUCTS,TIMES3(pmv2DNoElems) + 1/*YDIMNDX*/>(
                rsdCachexdim, threadIdx.x, strdst + col1);

        __syncthreads();

        char rsdy = rsdCacheydim[threadIdx.y];
        char rsdx = rsdCachexdim[threadIdx.x];
        bool cond = SSKNAGetPairingCondition(rsdy, rsdx);

        if(cond &&  rowy < strlen && col1 < strlen && 2 < abs(col1 - rowy)) {
            float dst = SSKGetDistance(strCoordsydim, strCoordsxdim, threadIdx.y, threadIdx.x);
            float dev = SSKNAGetDstDeviation(atomtype, dst);
            if(dev < dev_pair) { dev_pair = dev; col_pair = col1; }
        }

        __syncthreads();
    }


    //find the least deviation across col1 for each rowy (threadIdx.y)
    for(int s = (lXYDIM >> 1); s >= 1; s >>= 1) {
        float dev_s = __shfl_down_sync(0xffffffff, dev_pair, s/*delta*/);
        int col_s = __shfl_down_sync(0xffffffff, col_pair, s/*delta*/);
        if(lid < s && dev_s < dev_pair) {dev_pair = dev_s; col_pair = col_s;}
    }


    //minimums at threadIdx.x==0 across threadIdx.y
    if(threadIdx.x == 0) {
        dstCache[threadIdx.y] = dev_pair;
        posCache[threadIdx.y] = col_pair;
    }

    __syncthreads();

    //change indexing for rows:
    rowy = ndx0ydim + threadIdx.x;

    //write deviations and positions to GMEM
    if(threadIdx.y < 2 && rowy < strlen) {
        dev_pair = dstCache[threadIdx.x];
        col_pair = posCache[threadIdx.x];
        if(dev_pair < gtnaat_LUB_AVG_ERROR) {
            if(threadIdx.y == 0) tmpdpdiagbuffers[strdst + rowy + ndbCposs * lTMPBUFFNDX_POSIT] = col_pair;
            if(threadIdx.y == 1) tmpdpdiagbuffers[strdst + rowy + ndbCposs * lTMPBUFFNDX_DEVIA] = dev_pair;
        }
    }
}

// =========================================================================
// Instantiations
//
#define INSTANTIATE_CalcDistances_NASS(tpSTRUCTS) \
    template __global__ void CalcDistances_NASS<tpSTRUCTS>(\
        const int atomtype, const uint ndbCposs, \
        float* __restrict__ tmpdpdiagbuffers);

INSTANTIATE_CalcDistances_NASS(SSK_STRUCTS_QRIES);
INSTANTIATE_CalcDistances_NASS(SSK_STRUCTS_REFNS);

// -------------------------------------------------------------------------
// CalcDistances_NASS_CC7: calculate relevant pairwise distances between
// residues for all query OR reference nucleic acid structures in the chunk;
// atomtype, nucleic acid atom type processed;
// NOTE: version for compute capability starting with No. 7;
// ndbCposs, total number of reference positions in the chunk;
// tmpdpdiagbuffers, temporary buffer for distances and positions;
// 
template<int STRUCTS>
__global__
void CalcDistances_NASS_CC7(
    const int atomtype,
    const uint ndbCposs,
    float* __restrict__ tmpdpdiagbuffers)
{
    enum{   lXYDIM = CUSS_NARESDST_XYDIM,
            l2XYDIM = TIMES2(CUSS_NARESDST_XYDIM)
    };
    // blockIdx.x/y are block indices of positions;
    // blockIdx.z is the serial number of query or reference structure;
    const int strndx = blockIdx.z;
    //cache for coordinates: 
    __shared__ float strCoordsydim[lXYDIM][pmv2DNoElems];
    __shared__ float strCoordsxdim[l2XYDIM][pmv2DNoElems];
    //cache for most relevant distances and positions
    __shared__ float dstCache[lXYDIM+1];
    __shared__ int posCache[lXYDIM+1];
    __shared__ char rsdCacheydim[lXYDIM+1];
    __shared__ char rsdCachexdim[l2XYDIM+1];
    //relative position indices:
    const int ndx0ydim = blockIdx.y * blockDim.y;
    const int ndx0xdim = blockIdx.x * blockDim.x * 2;
    int rowy = ndx0ydim + threadIdx.y;//row index
    int col1 = ndx0xdim + threadIdx.x;//1st column
    int col2 = col1 + blockDim.x;//2nd column
    int strlen;//structure length
    int strdst;//distance in positions to the beginning of structures

    //reuse ccmCache
    if(threadIdx.y == 0 && threadIdx.x == 0) {
        if(STRUCTS == SSK_STRUCTS_QRIES)
                ((int*)(strCoordsydim[0]))[0] =  GetQueryLength(strndx);
        else    ((int*)(strCoordsydim[0]))[0] =  GetDbStrLength(strndx);
    }

    if(threadIdx.y == 1 && threadIdx.x == 0) {//next warp
        int strdst0;
        if(STRUCTS == SSK_STRUCTS_QRIES) {
            ((int*)(strCoordsydim[0]))[1] = strdst0 = GetQueryDst(strndx);
            ((int*)(strCoordsydim[0]))[2] = GetQueryStrField<INTYPE,pmv2D_Ins_Ch_Ord>(strdst0);
        } else {
            ((int*)(strCoordsydim[0]))[1] = strdst0 = GetDbStrDst(strndx);
            ((int*)(strCoordsydim[0]))[2] = GetDbStrField<INTYPE,pmv2D_Ins_Ch_Ord>(strdst0);
        }
    }

    __syncthreads();

    //NOTE: no bank conflict when two threads from the same warp access the same address;
    strlen = ((int*)(strCoordsydim[0]))[0];
    strdst = ((int*)(strCoordsydim[0]))[1];

    {
        int type = ((int*)(strCoordsydim[0]))[2];
        if(GetMoleculeType(type) != gtmtNA) return;
    }

    __syncthreads();

    //exit upon out-of-bounds condition:
    if(strlen <= ndx0ydim || strlen <= ndx0xdim) return;


    if(threadIdx.y == 12) dstCache[threadIdx.x] = 9999.9f;//too large deviation
    if(threadIdx.y == 13) posCache[threadIdx.x] = -1;

    //cache coordinates along y dim
    if(ndx0ydim + threadIdx.x < strlen)
        SSKCacheCoordsYPrl<STRUCTS,0/*YDIMSHIFT*/>(
            strCoordsydim, threadIdx.x, strdst + ndx0ydim + threadIdx.x/*strpos*/);

    //cache coordinates along x dim: 1st column block
    if(col1 < strlen)
        SSKCacheCoordsYPrl<STRUCTS,pmv2DNoElems/*YDIMSHIFT*/>(
            strCoordsxdim, threadIdx.x, strdst + col1);

    //cache coordinates along x dim: 2nd column block
    if(col2 < strlen)
        SSKCacheCoordsYPrl<STRUCTS,TIMES2(pmv2DNoElems)/*YDIMSHIFT*/>(
            strCoordsxdim, threadIdx.x + lXYDIM, strdst + col2);

    //cache residues along y dim
    if(ndx0ydim + threadIdx.x < strlen)
        SSKCacheRsds<STRUCTS,TIMES3(pmv2DNoElems)/*YDIMNDX*/>(
            rsdCacheydim, threadIdx.x, strdst + ndx0ydim + threadIdx.x/*strpos*/);

    //cache residues along x dim: 1st column block
    if(col1 < strlen)
        SSKCacheRsds<STRUCTS,TIMES3(pmv2DNoElems) + 1/*YDIMNDX*/>(
            rsdCachexdim, threadIdx.x, strdst + col1);

    //cache residues along x dim: 2nd column block
    if(col2 < strlen)
        SSKCacheRsds<STRUCTS,TIMES3(pmv2DNoElems) + 2/*YDIMNDX*/>(
            rsdCachexdim, threadIdx.x + lXYDIM, strdst + col2);

    __syncthreads();


    float dev_pair, dev = 9999.9f;
    int col_pair = col1;
    const int lid = threadIdx.x & 0x1f;//lane id

    if(rowy < strlen && col1 < strlen && 2 < abs(col1 - rowy)) {
        char rsdy = rsdCacheydim[threadIdx.y];
        char rsdx = rsdCachexdim[threadIdx.x];
        bool cond = SSKNAGetPairingCondition(rsdy, rsdx);
        if(cond) {
            float dst = SSKGetDistance(
                strCoordsydim, strCoordsxdim, threadIdx.y, threadIdx.x);
            dev = SSKNAGetDstDeviation(atomtype, dst);
        }
    }

    dev_pair = dev;

    //find the least deviation across col1 for each rowy (threadIdx.y)
    for(int s = (lXYDIM >> 1); s >= 1; s >>= 1) {
        float dev_s = __shfl_down_sync(0xffffffff, dev_pair, s/*delta*/);
        int col_s = __shfl_down_sync(0xffffffff, col_pair, s/*delta*/);
        if(lid < s && dev_s < dev_pair) {dev_pair = dev_s; col_pair = col_s;}
    }

    //repeat the procedure for the col2 block
    if(rowy < strlen && col2 < strlen && 2 < abs(col2 - rowy)) {
        char rsdy = rsdCacheydim[threadIdx.y];
        char rsdx = rsdCachexdim[threadIdx.x + lXYDIM];
        bool cond = SSKNAGetPairingCondition(rsdy, rsdx);
        if(cond) {
            float dst = SSKGetDistance(
                strCoordsydim, strCoordsxdim, threadIdx.y, threadIdx.x + lXYDIM);
            dev = SSKNAGetDstDeviation(atomtype, dst);
        }
    }

    if(dev < dev_pair) {dev_pair = dev; col_pair = col2;}

    //find the least deviation across col2 for each rowy (threadIdx.y)
    for(int s = (lXYDIM >> 1); s >= 1; s >>= 1) {
        float dev_s = __shfl_down_sync(0xffffffff, dev_pair, s/*delta*/);
        int col_s = __shfl_down_sync(0xffffffff, col_pair, s/*delta*/);
        if(lid < s && dev_s < dev_pair) {dev_pair = dev_s; col_pair = col_s;}
    }

    //minimums at threadIdx.x==0 across threadIdx.y
    if(threadIdx.x == 0) {
        dstCache[threadIdx.y] = dev_pair;
        posCache[threadIdx.y] = col_pair;
    }

    __syncthreads();

    //change indexing for rows:
    rowy = ndx0ydim + threadIdx.x;

    //write deviations and positions to GMEM
    if(threadIdx.y == 0 && rowy < strlen) {
        dev_pair = dstCache[threadIdx.x];
        col_pair = posCache[threadIdx.x];
        uint* mutex =
            (uint*)(&tmpdpdiagbuffers[strdst + rowy + ndbCposs * lTMPBUFFNDX_MUTEX]);
        if(dev_pair < gtnaat_LUB_AVG_ERROR) {
            LOCK(mutex);
                float stored = tmpdpdiagbuffers[strdst + rowy + ndbCposs * lTMPBUFFNDX_DEVIA];
                if(dev_pair < stored) {
                    tmpdpdiagbuffers[strdst + rowy + ndbCposs * lTMPBUFFNDX_POSIT] = col_pair;//float<-int
                    tmpdpdiagbuffers[strdst + rowy + ndbCposs * lTMPBUFFNDX_DEVIA] = dev_pair;
                }
                __threadfence();
            UNLOCK(mutex);
        }
    }
}

// =========================================================================
// Instantiations
//
#define INSTANTIATE_CalcDistances_NASS_CC7(tpSTRUCTS) \
    template __global__ void CalcDistances_NASS_CC7<tpSTRUCTS>(\
        const int atomtype, const uint ndbCposs, \
        float* __restrict__ tmpdpdiagbuffers);

INSTANTIATE_CalcDistances_NASS_CC7(SSK_STRUCTS_QRIES);
INSTANTIATE_CalcDistances_NASS_CC7(SSK_STRUCTS_REFNS);

// -------------------------------------------------------------------------
