/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libutil/CLOptions.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/custages/fields.cuh"
#include "covariance.cuh"

// -------------------------------------------------------------------------
// InitCCData0Helper: device helper code for initializing cross-covariance 
// data between the query and reference structures when unconditionally 
// calculating superpositions for fragments (initial phase);
// FQryRfnPosGetter, template parameter, function type, query and reference position getter;
// FPosInvalidator, template parameter, function type, position invalidator;
// depth, superposition depth for calculating query and reference positions;
// sfragfct, fragment factor;
// qryndx, query serial number;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// arg1, argument 1 for calculating starting query/reference position;
// arg2, argument 2 for calculating starting query/reference position;
// arg3, argument 3 for calculating starting query/reference position;
// alnlen, maximum alignment length which is less than the minimum 
// length of the structures being compared;
// NOTE: memory pointers should be aligned!
// wrkmem, working memory, including the section of CC data;
// NOTE: unroll by a factor of CUS1_TBINITSP_CCDINIT_XFCT: this number of 
// structures initialized by a thread block
// 
template<typename FQryRfnPosGetter, typename FPosInvalidator>
__device__ __forceinline__
void InitCCData0Helper(
    FQryRfnPosGetter fPosGetter,
    FPosInvalidator fInvalidator,
    const int depth,
    const uint sfragfct,
    const uint qryndx,
    const uint ndbCstrs,
    const uint maxnsteps,
    int arg1, int arg2, int arg3,
    float* __restrict__ wrkmem)
{
    //index of the first structure to start with (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * CUS1_TBINITSP_CCDINIT_XFCT;
    constexpr int qslot = CUS1_TBINITSP_CCDINIT_XFCT;//query slot in wrtCache
    __shared__ int wrtCache[CUS1_TBINITSP_CCDINIT_XFCT+1];
    uint ndx = 0;//relative reference index < CUS1_TBINITSP_CCDINIT_XFCT

    if(threadIdx.x < CUS1_TBINITSP_CCDINIT_XFCT) {
        wrtCache[threadIdx.x] = 0;
    }

    #pragma unroll
    for(int i = 1; i < CUS1_TBINITSP_CCDINIT_XFCT; i++)
        if(i * twmvEndOfCCDataExt <= threadIdx.x) ndx = i;

    if(threadIdx.x < CUS1_TBINITSP_CCDINIT_XFCT && dbstrndx + threadIdx.x < ndbCstrs) {
        wrtCache[threadIdx.x] = GetDbStrLength(dbstrndx+threadIdx.x);
        if(threadIdx.x == 0) wrtCache[qslot] = GetQueryLength(qryndx);
    }

    __syncthreads();

    if(threadIdx.x < CUS1_TBINITSP_CCDINIT_XFCT && dbstrndx + threadIdx.x < ndbCstrs) {
        int qrylen = wrtCache[qslot];
        int dbstrlen = wrtCache[threadIdx.x];
        int qrypos, rfnpos;
        fPosGetter(depth, qrypos, rfnpos,  qrylen, dbstrlen, sfragfct, arg1, arg2, arg3);
        wrtCache[threadIdx.x] = !fInvalidator(qrylen, dbstrlen, qrypos, rfnpos, arg1, arg2, arg3);
    }

    __syncthreads();

    if(wrtCache[ndx] && threadIdx.x < twmvEndOfCCDataExt * (ndx+1)) {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx + ndx) * nTWorkingMemoryVars;
        wrkmem[mloc + threadIdx.x - twmvEndOfCCDataExt * ndx] = 0.0f;
    }
}

// -------------------------------------------------------------------------
// InitCCData0: device code for initializing cross-covariance data 
// between the query and reference structures when unconditionally 
// calculating superpositions for fragments (initial phase);
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// n1, starting position that determines positions in query and reference;
// step, step size in positions used to traverse query and reference 
// ungapped alignments;
// NOTE: memory pointers should be aligned!
// wrkmem, working memory, including the section of CC data;
// 
__global__ void InitCCData0(
    const uint ndbCstrs,
    const uint maxnsteps,
    int n1, int step,
    float* __restrict__ wrkmem)
{
    uint qryndx = blockIdx.y;//query serial number
    uint sfragfct = blockIdx.z;//fragment factor

    InitCCData0Helper(
        GetQryRfnPos, PositionsOutofBounds, 0/*depth; unused*/,
        sfragfct, qryndx, ndbCstrs, maxnsteps,  n1, step, 0/*arg3(unused)*/,
        wrkmem);
}

#if 0
// -------------------------------------------------------------------------
// InitCCData0_var: device code for initializing cross-covariance data 
// between the query and reference structures when unconditionally 
// calculating superpositions for VARIABLE-LENGTH fragments;
// STEPx5, template parameter, multiply the step by 5 when calculating 
// query and reference positions;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// qryfragfct, fragment factor for query (to be multiplied by step dependent 
// upon lengths);
// rfnfragfct, fragment factor for reference (to be multiplied by step dependent 
// upon lengths);
// fragndx, fragment index determining the fragment size dependent 
// upon lengths;
// NOTE: memory pointers should be aligned!
// wrkmem, working memory, including the section of CC data;
// 
template<bool STEPx5>
__global__
void InitCCData0_var(
    const uint ndbCstrs,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    float* __restrict__ wrkmem)
{
    uint qryndx = blockIdx.y;//query serial number
    uint sfragfct = blockIdx.z;//fragment factor

    InitCCData0Helper(
        STEPx5? GetQryRfnPos_var5: GetQryRfnPos_var, PositionsOutofBounds_var,
        sfragfct, qryndx, ndbCstrs, maxnsteps,  qryfragfct, rfnfragfct, fragndx,
        wrkmem);
}
#endif

// -------------------------------------------------------------------------
// InitCCData0_frg: device code for initializing cross-covariance data 
// between the query and reference structures when unconditionally 
// calculating superpositions for VARIABLE-LENGTH fragments;
// similar to InitCCData0_var with more extensive parallelization;
// stepx5, multiply the step by 5 when calculating query and reference 
// positions;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// qryfragfct, fragment factor for query (to be multiplied by step dependent 
// upon lengths);
// rfnfragfct, fragment factor for reference (to be multiplied by step dependent 
// upon lengths);
// fragndx, fragment index determining the fragment size dependent 
// upon lengths;
// NOTE: memory pointers should be aligned!
// wrkmem, working memory, including the section of CC data;
// 
#if 0
__global__
void InitCCData0_frg(
    const bool stepx5,
    const uint ndbCstrs,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    float* __restrict__ wrkmem)
{
    uint qryndx = blockIdx.y;//query serial number
    uint sfragfct = blockIdx.z;//fragment factor

    InitCCData0Helper(
        stepx5? GetQryRfnPos_frg5: GetQryRfnPos_frg, PositionsOutofBounds_frg,
        sfragfct, qryndx, ndbCstrs, maxnsteps,  qryfragfct, rfnfragfct, fragndx,
        wrkmem);
}
#endif

// InitCCData0_frg2: device code for initializing cross-covariance data 
// between the query and reference structures when unconditionally 
// calculating superpositions for variable-length fragments;
// with more extensive parallelization;
// depth, superposition depth for calculating query and reference positions;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// qryfragfct, fragment factor for query (to be multiplied by step dependent 
// upon lengths);
// rfnfragfct, fragment factor for reference (to be multiplied by step dependent 
// upon lengths);
// fragndx, fragment index determining the fragment size dependent 
// upon lengths;
// NOTE: memory pointers should be aligned!
// wrkmem, working memory, including the section of CC data;
__global__
void InitCCData0_frg2(
    const int depth,
    const uint ndbCstrs,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    float* __restrict__ wrkmem)
{
    uint qryndx = blockIdx.y;//query serial number
    uint sfragfct = blockIdx.z;//fragment factor
    fragndx = (sfragfct & 1);

    InitCCData0Helper(
        GetQryRfnPos_frg2, PositionsOutofBounds_frg, depth,
        sfragfct, qryndx, ndbCstrs, maxnsteps,  qryfragfct, rfnfragfct, fragndx,
        wrkmem);
}

#if 0
// InitCCData0_frgbest: initialize cross-covariance data 
// between the query and reference structures for best fragment 
// identified before; fragment configuration read from memory section wrkmemaux;
// 
__global__
void InitCCData0_frgbest(
    const uint ndbCstrs,
    const uint maxnsteps,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem)
{
    //reference structure index (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * CUS1_TBINITSP_CCDINIT_XFCT + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number
    uint sfragfct = blockIdx.z;//fragment factor (=0)
    __shared__ int argsCache[3];
    int qrypos = 0, rfnpos = 0, fraglen = 3;

    uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;

    if(threadIdx.x < CUS1_TBINITSP_CCDINIT_XFCT && dbstrndx < ndbCstrs) {
        argsCache[0] = wrkmemaux[mloc + tawmvQRYpos * ndbCstrs + dbstrndx];
        argsCache[1] = wrkmemaux[mloc + tawmvRFNpos * ndbCstrs + dbstrndx];
        //NOTE: actual fragment length has been written:
        argsCache[2] = wrkmemaux[mloc + tawmvSubFragNdx * ndbCstrs + dbstrndx];
    }

    __syncthreads();

    qrypos = argsCache[0]; rfnpos = argsCache[1]; fraglen = argsCache[2];

    InitCCData0Helper(
        GetQryRfnPos_frgbest, PositionsOutofBounds_frgbest,
        sfragfct, qryndx, ndbCstrs, maxnsteps,  qrypos, rfnpos, fraglen,
        wrkmem);
}

// =========================================================================
// Instantiations:
// 
#define INSTANTIATE_InitCCData0_var(tpSTEPx5) \
    template __global__ void InitCCData0_var<tpSTEPx5>( \
        const uint ndbCstrs, \
        const uint maxnsteps, int qryfragfct, int rfnfragfct, int fragndx, \
        float* __restrict__ wrkmem);

INSTANTIATE_InitCCData0_var(false);
INSTANTIATE_InitCCData0_var(true);
#endif



// -------------------------------------------------------------------------
// InitCCData: device code for initializing cross-covariance data 
// between the query and reference structures;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// minfraglen, minimum fragment length for which maxnsteps is calculated;
// NOTE: memory pointers should be aligned!
// wrkmem, working memory, including the section of CC data;
// NOTE: unroll by a factor of CUS1_TBINITSP_CCDINIT_XFCT: this number of 
// structures initialized by a thread block
// 
template<int CHCKCONV>
__global__ void InitCCData(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux)
{
    //index of the first structure to start with (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * CUS1_TBINITSP_CCDINIT_XFCT;
    uint qryndx = blockIdx.y;//query serial number
    uint sfragfct = blockIdx.z;//fragment factor
    uint ndx = 0;//relative reference index < CUS1_TBINITSP_CCDINIT_XFCT
    __shared__ float cnvCache[CUS1_TBINITSP_CCDINIT_XFCT];

    if(CHCKCONV == CHCKCONV_CHECK) {
        //any type of convergence implies ignoring the function of this method
        if(threadIdx.x < CUS1_TBINITSP_CCDINIT_XFCT) cnvCache[threadIdx.x] = 1.0f;

        if(threadIdx.x < CUS1_TBINITSP_CCDINIT_XFCT && dbstrndx + threadIdx.x < ndbCstrs) {
            uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
            cnvCache[threadIdx.x] =
                wrkmemaux[mloc + tawmvConverged * ndbCstrs + dbstrndx + threadIdx.x];
        }

        __syncthreads();
    }

    //TODO: do not initialize for sfragfct implying the out-of-bounds condition;

    #pragma unroll
    for(int i = 1; i < CUS1_TBINITSP_CCDINIT_XFCT; i++)
        if(i * twmvEndOfCCDataExt <= threadIdx.x) ndx = i;

    if(threadIdx.x < twmvEndOfCCDataExt * (ndx+1) && 
      ((CHCKCONV == CHCKCONV_CHECK)? cnvCache[ndx] == 0.0f: dbstrndx + ndx < ndbCstrs)) {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx + ndx) * nTWorkingMemoryVars;
        wrkmem[mloc + threadIdx.x - twmvEndOfCCDataExt * ndx] = 0.0f;
    }
}

// Instantiations
//
#define INSTANTIATE_InitCCData(tpCHCKCONV) \
    template __global__ void InitCCData<tpCHCKCONV>( \
        const uint ndbCstrs, const uint maxnsteps, \
        float* __restrict__ wrkmem, \
        float* __restrict__ wrkmemaux);

INSTANTIATE_InitCCData(CHCKCONV_NOCHECK);
INSTANTIATE_InitCCData(CHCKCONV_CHECK);



// -------------------------------------------------------------------------
// CalcCCMatrices64Helper: helper function to calculate cross-covariance 
// matrix between the query and  reference structures;
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// NOTE: Version for CUS1_TBINITSP_CCMCALC_XDIM==64!
// FPosInvalidator, template parameter, function type, position invalidator;
// depth, superposition depth for calculating query and reference positions;
// sfragfct, fragment factor;
// qryndx, query serial number;
// nqystrs, total number of queries;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// arg1, argument 1 for calculating starting query/reference position;
// arg2, argument 2 for calculating starting query/reference position;
// arg3, argument 3 for calculating starting query/reference position;
// alnlen, maximum alignment length which is less than the minimum 
// length of the structures being compared;
// NOTE: memory pointers should be aligned!
// wrkmem, working memory, including the section of CC data;
// 
template<
    int CHCKCONV,
    typename FQryRfnPosGetter,
    typename FPosInvalidator>
__device__ __forceinline__
void CalcCCMatrices64Helper(
    FQryRfnPosGetter fPosGetter,
    FPosInvalidator fInvalidator,
    const int depth,
    const uint sfragfct,
    const uint qryndx,
    const uint ndbCstrs,
    const uint maxnsteps,
    int arg1, int arg2, int arg3,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem)
{
    // blockIdx.x is the block index of positions for query-reference pair;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number TIMES fragment factor;
    //cache for the cross-covarinace matrix and related data: 
    //no bank conflicts as long as twmvEndOfCCData is odd
    __shared__ float ccmCache[twmvEndOfCCData * CUS1_TBINITSP_CCMCALC_XDIM];
    //relative position index:
    const uint ndx0 = blockIdx.x * blockDim.x * CUS1_TBINITSP_CCMCALC_XFCT;
    const uint ndx = ndx0 + threadIdx.x;
    const uint dbstrndx = blockIdx.y;
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;


    if(CHCKCONV == CHCKCONV_CHECK) {
        if(threadIdx.x == 0) {
            //NOTE: reuse ccmCache to read convergence flag at both 0 and sfragfct:
            uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
            ccmCache[6] = wrkmemaux[mloc0 + dbstrndx];
        }
        if(threadIdx.x == 32) {//next warp
            uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
            ccmCache[7] = wrkmemaux[mloc + dbstrndx];
        }

        __syncthreads();

        if((((int)(ccmCache[6])) & (CONVERGED_LOWTMSC_bitval)) || ccmCache[7])
            //(NOTE:any type of convergence applies locally);
            //all threads in the block exit;
            return;

        //NOTE: no sync as long as ccmCache cells for convergence not overwritten;
    }


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse ccmCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(dbstrndx, (int*)ccmCache);
        GetQueryLenDst(qryndx, (int*)ccmCache + 2);
    }

    __syncthreads();

    //NOTE: no bank conflict when two threads from the same warp access the same address;
    //blockDim.x includes only several warps
    dbstrlen = ((int*)ccmCache)[0]; dbstrdst = ((int*)ccmCache)[1];
    qrylen = ((int*)ccmCache)[2]; qrydst = ((int*)ccmCache)[3];

    __syncthreads();


    int qrypos, rfnpos;

    fPosGetter(depth,  qrypos, rfnpos,  qrylen, dbstrlen, sfragfct, arg1, arg2, arg3);

    if(qrylen <= qrypos + ndx0 || dbstrlen <= rfnpos + ndx0)
        //all threads in the block exit if thread 0 is out of bounds
        return;

    if(fInvalidator(qrylen, dbstrlen, qrypos, rfnpos, arg1, arg2, arg3))
        //all threads in the block exit
        return;


    //initialize cache
    #pragma unroll
    for(int i = 0; i < twmvEndOfCCData; i++)
        ccmCache[threadIdx.x * twmvEndOfCCData +i] = 0.0f;

    #pragma unroll
    for(int i = 0; i < CUS1_TBINITSP_CCMCALC_XFCT; i++) {
        //manually unroll along data blocks by a factor of CUS1_TBINITSP_CCMCALC_XFCT
        if(!(qrypos + ndx + i * blockDim.x < qrylen &&
             rfnpos + ndx + i * blockDim.x < dbstrlen))
            break;
        UpdateCCMOneAlnPos(
            qrydst + qrypos + ndx + i * blockDim.x,//query position
            dbstrdst + rfnpos + ndx + i * blockDim.x,//reference position
            ccmCache
        );
        //no sync: every thread works in its own space (of ccmCache)
    }

    //sync now:
    __syncthreads();

    //unroll by a factor 2
    if(threadIdx.x < (CUS1_TBINITSP_CCMCALC_XDIM>>1)) {
        #pragma unroll
        for(int i = 0; i < twmvEndOfCCData; i++)
            ccmCache[threadIdx.x * twmvEndOfCCData +i] +=
                ccmCache[(threadIdx.x + (CUS1_TBINITSP_CCMCALC_XDIM>>1)) * twmvEndOfCCData +i];
    }

    __syncthreads();

    //unroll warp
    if(threadIdx.x < 32) {
        #pragma unroll
        for(int i = 0; i < twmvEndOfCCData; i++) {
            float sum = ccmCache[threadIdx.x * twmvEndOfCCData + i];
            sum = mywarpreducesum(sum);
            //write to the first data slot of SMEM
            if(threadIdx.x == 0) ccmCache[i] = sum;
        }
    }

    //in case of twmvEndOfCCData gets larger than warpSize
    __syncthreads();

    //add the result and write to global memory
    if(threadIdx.x < twmvEndOfCCData) {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTWorkingMemoryVars;
        atomicAdd(&wrkmem[mloc + threadIdx.x], ccmCache[threadIdx.x]);
    }
}

// -------------------------------------------------------------------------
// CalcCCMatrices: calculate cross-covariance matrix between the query and 
// reference structures;
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// NOTE: Version for CUS1_TBINITSP_CCMCALC_XDIM==64!
// nqystrs, total number of queries;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// n1, starting position that determines positions in query and reference;
// step, step size in positions used to traverse query and reference 
// ungapped alignments;
// alnlen, maximum alignment length which corresponds to the minimum 
// length of the structures being compared;
// NOTE: memory pointers should be aligned!
// wrkmem, working memory, including the section of CC data;
// 
__global__ void CalcCCMatrices64(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint maxnsteps,
    int n1, int step,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem)
{
    // blockIdx.x is the block index of positions for query-reference pair;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number TIMES fragment factor;
    uint sfragfct = blockIdx.z / nqystrs;//fragment factor
    uint qryndx = blockIdx.z - sfragfct * nqystrs;//query serial number

    CalcCCMatrices64Helper<CHCKCONV_NOCHECK>(
        GetQryRfnPos, PositionsOutofBounds, 0/*depth; unused*/,
        sfragfct, qryndx, ndbCstrs, maxnsteps,  n1, step, 0/*arg3(unused)*/,
        wrkmemaux, wrkmem);
}

#if 0
// -------------------------------------------------------------------------
// CalcCCMatrices64_var: calculate cross-covariance matrix between the 
// query and reference structures over a fragment whose length depends on 
// structure lengths;
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// NOTE: Version for CUS1_TBINITSP_CCMCALC_XDIM==64!
// STEPx5, template parameter, multiply the step by 5 when calculating 
// query and reference positions;
// nqystrs, total number of queries;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// qryfragfct, fragment factor for query (to be multiplied by step dependent 
// upon lengths);
// rfnfragfct, fragment factor for reference (to be multiplied by step dependent 
// upon lengths);
// fragndx, fragment index determining the fragment size dependent 
// upon lengths;
// alnlen, maximum alignment length which corresponds to the minimum 
// length of the structures being compared;
// NOTE: memory pointers should be aligned!
// wrkmem, working memory, including the section of CC data;
// 
template<bool STEPx5>
__global__
void CalcCCMatrices64_var(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    float* __restrict__ wrkmem)
{
    // blockIdx.x is the block index of positions for query-reference pair;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number TIMES fragment factor;
    uint sfragfct = blockIdx.z / nqystrs;//fragment factor
    uint qryndx = blockIdx.z - sfragfct * nqystrs;//query serial number

    CalcCCMatrices64Helper(
        STEPx5? GetQryRfnPos_var5: GetQryRfnPos_var, PositionsOutofBounds_var,
        sfragfct, qryndx, ndbCstrs, maxnsteps,  qryfragfct, rfnfragfct, fragndx,
        wrkmem);
}

// Instantiations:
// 
#define INSTANTIATE_CalcCCMatrices64_var(tpSTEPx5) \
    template __global__ void CalcCCMatrices64_var<tpSTEPx5>( \
        const uint nqystrs, const uint ndbCstrs, const uint maxnsteps, \
        int qryfragfct, int rfnfragfct, int fragndx, \
        float* __restrict__ wrkmem);

INSTANTIATE_CalcCCMatrices64_var(false);
INSTANTIATE_CalcCCMatrices64_var(true);
#endif

#if 0
// -------------------------------------------------------------------------
// CalcCCMatrices64_frg: calculate cross-covariance matrix between the 
// query and reference structures over a fragment whose length depends on 
// structure lengths;
// same as CalcCCMatrices64_var with more extensive parallelization;
// stepx5, multiply the step by 5 when calculating query and reference 
// positions;
// 
__global__
void CalcCCMatrices64_frg(
    const bool stepx5,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    float* __restrict__ wrkmem)
{
    // blockIdx.x is the block index of positions for query-reference pair;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number TIMES fragment factor;
    uint sfragfct = blockIdx.z / nqystrs;//fragment factor
    uint qryndx = blockIdx.z - sfragfct * nqystrs;//query serial number

    CalcCCMatrices64Helper(
        stepx5? GetQryRfnPos_frg5: GetQryRfnPos_frg, PositionsOutofBounds_frg,
        sfragfct, qryndx, ndbCstrs, maxnsteps,  qryfragfct, rfnfragfct, fragndx,
        wrkmem);
}
#endif

// CalcCCMatrices64_frg2: calculate cross-covariance matrix between the 
// query and reference structures over a fragment whose length depends on 
// structure lengths;
// same as CalcCCMatrices64_var with more extensive parallelization;
// depth, superposition depth for calculating query and reference positions;
__global__
void CalcCCMatrices64_frg2(
    const int depth,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem)
{
    // blockIdx.x is the block index of positions for query-reference pair;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number TIMES fragment factor;
    uint sfragfct = blockIdx.z / nqystrs;//fragment factor
    uint qryndx = blockIdx.z - sfragfct * nqystrs;//query serial number
    fragndx = (sfragfct & 1);

    CalcCCMatrices64Helper<CHCKCONV_CHECK>(
        GetQryRfnPos_frg2, PositionsOutofBounds_frg, depth,
        sfragfct, qryndx, ndbCstrs, maxnsteps,  qryfragfct, rfnfragfct, fragndx,
        wrkmemaux, wrkmem);
}

#if 0
// CalcCCMatrices64_frgbest: calculate cross-covariance matrix between the 
// query and reference structures for best-identified fragment;
// 
__global__
void CalcCCMatrices64_frgbest(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint maxnsteps,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem)
{
    // blockIdx.x is the block index of positions for query-reference pair;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number TIMES fragment factor;
    uint dbstrndx = blockIdx.y;
    uint sfragfct = blockIdx.z / nqystrs;//fragment factor
    uint qryndx = blockIdx.z - sfragfct * nqystrs;//query serial number
    __shared__ int argsCache[3];
    int qrypos = 0, rfnpos = 0, fraglen = 3;

    uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;

    if(threadIdx.x == 0) {
        argsCache[0] = wrkmemaux[mloc + tawmvQRYpos * ndbCstrs + dbstrndx];
        argsCache[1] = wrkmemaux[mloc + tawmvRFNpos * ndbCstrs + dbstrndx];
        //NOTE: actual fragment length has been written:
        argsCache[2] = wrkmemaux[mloc + tawmvSubFragNdx * ndbCstrs + dbstrndx];
    }

    __syncthreads();

    qrypos = argsCache[0]; rfnpos = argsCache[1]; fraglen = argsCache[2];

    CalcCCMatrices64Helper(
        GetQryRfnPos_frgbest, PositionsOutofBounds_frgbest,
        sfragfct, qryndx, ndbCstrs, maxnsteps,  qrypos, rfnpos, fraglen,
        wrkmem);
}
#endif

// =========================================================================



// -------------------------------------------------------------------------
// CopyCCDataToWrkMem2: helper function to copy cross-covariance matrix 
// between the query and reference structures to section 2 to enable 
// efficient Kabsch algorithm application for multiple structures 
// simultaneously;
// NOTE: thread block is 2D and copies structures' data: from:
// NOTE: | struct i          | struct i+1        | ...
// NOTE: | field1,dield2,... | field1,dield2,... | ...
// NOTE: to 
// NOTE: | struct i | struct i+1 | ... | struct i | ... 
// NOTE: | field1   | field1     | ... | field2   | ...
// READNPOS, template parameter indicating whether nalnposs should be read;
// FQryRfnPosGetter, template parameter, function type, query and reference position getter;
// FPosInvalidator, template parameter, function type, position invalidator;
// FAlnLengthGetter, template parameter, function type, alignment length getter;
// depth, superposition depth for calculating query and reference positions;
// sfragfct, fragment factor;
// qryndx, query serial number;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// arg1, argument 1 for calculating starting query/reference position;
// arg2, argument 2 for calculating starting query/reference position;
// arg3, argument 3 for calculating starting query/reference position;
// alnlen, maximum alignment length which is less than the minimum 
// length of the structures being compared;
// NOTE: memory pointers should be aligned!
// wrkmem, working memory, including the section of CC data (saved as 
// whole for each structure) to copy;
// wrkmem2, working memory, including the section of CC data to be written by 
// field;
// 
template<
    int CHCKCONV,
    int READNPOS,
    typename FQryRfnPosGetter,
    typename FPosInvalidator,
    typename FAlnLengthGetter>
__device__ __forceinline__
void CopyCCDataToWrkMem2Helper(
    FQryRfnPosGetter fPosGetter,
    FPosInvalidator fInvalidator,
    FAlnLengthGetter fAlnLenGetter,
    const int depth,
    const uint sfragfct,
    const uint qryndx,
    const uint ndbCstrs,
    const uint maxnsteps,
    int arg1, int arg2, int arg3,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmem2)
{
    //cache for cross-covarinace matrices and related data: 
    //bank conflicts resolved as long as innermost dim is odd
    __shared__ float ccmCache[CUS1_TBINITSP_CCMCOPY_N][twmvEndOfCCDataExt+1];
    //index of the first structure to start with (blockIdx.x, refn. serial number):
    int dbstrndx = blockIdx.x * CUS1_TBINITSP_CCMCOPY_N;
    int qrylen, dbstrlen;//query and reference length
    int absndx = dbstrndx + threadIdx.x;
    int nalnposs = 0;


    if(CHCKCONV == CHCKCONV_CHECK) {
        if(absndx < ndbCstrs && threadIdx.y == tawmvConverged)
        {
            uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + threadIdx.y) * ndbCstrs;
            uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + threadIdx.y) * ndbCstrs;
            ccmCache[threadIdx.x][threadIdx.y] = wrkmemaux[mloc0 + absndx/*dbstrndx*/];
            float convflag = ccmCache[threadIdx.x][threadIdx.y];
            if(sfragfct != 0 ) convflag = wrkmemaux[mloc + absndx];
            ccmCache[threadIdx.x][threadIdx.y] = //any convergence applies locally
                (((int)(ccmCache[threadIdx.x][threadIdx.y])) & (CONVERGED_LOWTMSC_bitval)) ||
                (convflag);
        }
    }

    if(absndx < ndbCstrs && threadIdx.x == 0 && threadIdx.y == 0)
        //reuse cache
        ccmCache[0][0] = GetQueryLength(qryndx);

    __syncthreads();

    if(absndx < ndbCstrs && threadIdx.y == 0)
        qrylen = ccmCache[0][0];

    __syncthreads();

    //calculate and write to smem #alignment positions
    if(absndx < ndbCstrs && threadIdx.y == 0) {
        if(CHCKCONV == CHCKCONV_CHECK && ccmCache[threadIdx.x][tawmvConverged]) {
            //assign 0 #aligned positions so that no memory and 
            //computing operations are executed
            ccmCache[threadIdx.x][twmvNalnposs] = 0.0f;
            ccmCache[threadIdx.x][twmvNalnposs+1] = 0.0f;
        }
        else {
            int qrypos, rfnpos, nlposs;
            dbstrlen = GetDbStrLength(absndx);
            fPosGetter(depth,  qrypos, rfnpos,  qrylen, dbstrlen, sfragfct, arg1, arg2, arg3);
            nlposs = fAlnLenGetter(qrylen, dbstrlen, qrypos, rfnpos, arg1, arg2, arg3);
            //NOTE: fInvalidator changes qrylen and dbstrlen by definition!
            if(!fInvalidator(qrylen, dbstrlen, qrypos, rfnpos, arg1, arg2, arg3))
                nalnposs = nlposs;
            //write nalnposs to smem; this slot won't be overwritten: no sync
            ccmCache[threadIdx.x][twmvNalnposs] = nalnposs;
            ccmCache[threadIdx.x][twmvNalnposs+1] = nalnposs;
        }
    }

    //used only if nalnposs value is verified below
    __syncthreads();

    //cache data: iterative coalesced read
    for(int reldbndx = threadIdx.y; reldbndx < CUS1_TBINITSP_CCMCOPY_N; reldbndx += blockDim.y) {
        int absndxloc = dbstrndx + reldbndx;
        if(absndxloc < ndbCstrs && 
           threadIdx.x < ((READNPOS==READNPOS_READ)? twmvEndOfCCDataExt: twmvEndOfCCData) && 
           ccmCache[reldbndx][twmvNalnposs+1])
        {
            //read only if nalnposs >0
            uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + absndxloc) * nTWorkingMemoryVars;
            ccmCache[reldbndx][threadIdx.x] = wrkmem[mloc + threadIdx.x];
            if(threadIdx.x == twmvNalnposs && 
               ccmCache[reldbndx][twmvNalnposs] == ccmCache[reldbndx][twmvNalnposs+1])
                //NOTE: if nalnposs equals maximum possible for given qrypos and rfnpos,
                //assign it to 0 so that the Kabsch algorithm is not applied to this 
                //particular query-reference pair:
                ccmCache[reldbndx][twmvNalnposs] = 0.0f;
        }
    }

    __syncthreads();

    //write data to gmem; coalesced write;
    //first write nalnposs 
    if(absndx < ndbCstrs && threadIdx.y == twmvNalnposs) {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTWorkingMemoryVars + threadIdx.y) * ndbCstrs;
        wrkmem2[mloc + absndx] = ccmCache[threadIdx.x][threadIdx.y];
    }

    if(absndx < ndbCstrs && threadIdx.y < twmvEndOfCCData &&
       ccmCache[threadIdx.x][twmvNalnposs]) {
        //write only if nalnposs >0;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTWorkingMemoryVars + threadIdx.y) * ndbCstrs;
        wrkmem2[mloc + absndx] = ccmCache[threadIdx.x][threadIdx.y];
    }
}

// -------------------------------------------------------------------------
// CopyCCDataToWrkMem2: copy cross-covariance matrix between the query and 
// reference structures to section 2 to enable efficient Kabsch 
// algorithm application for multiple structures simultaneously;
// NOTE: thread block is 2D and copies structures' data: from:
// NOTE: | struct i          | struct i+1        | ...
// NOTE: | field1,dield2,... | field1,dield2,... | ...
// NOTE: to 
// NOTE: | struct i | struct i+1 | ... | struct i | ... 
// NOTE: | field1   | field1     | ... | field2   | ...
// READNPOS, template parameter indicating whether nalnposs should be read;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// n1, starting position that determines positions in query and reference;
// step, step size in positions used to traverse query and reference;
// qrypos, query position starting with which CCM data have been calculated;
// rfnpos, reference position starting with which CCM data have been calculated;
// NOTE: memory pointers should be aligned!
// wrkmem, working memory, including the section of CC data (saved as 
// whole for each structure) to copy;
// wrkmem2, working memory, including the section of CC data to be written by 
// field;
// 
template<int READNPOS>
__global__ void CopyCCDataToWrkMem2(
    const uint ndbCstrs,
    const uint maxnsteps,
    int n1, int step,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmem2)
{
    int qryndx = blockIdx.y;//query index in the chunk
    int sfragfct = blockIdx.z;//fragment factor

    CopyCCDataToWrkMem2Helper<CHCKCONV_NOCHECK,READNPOS>(
        GetQryRfnPos, PositionsOutofBounds, GetNAlnPoss, 0/*depth; unused*/,
        sfragfct, qryndx, ndbCstrs, maxnsteps,  n1, step, 0/*arg3(unused)*/,
        wrkmemaux, wrkmem, wrkmem2);
}

// =========================================================================
// Instantiations:
// 
#define INSTANTIATE_CopyCCDataToWrkMem2(tpREADNPOS) \
    template __global__ void CopyCCDataToWrkMem2<tpREADNPOS>( \
        const uint ndbCstrs, \
        const uint maxnsteps, int n1, int step, \
        const float* __restrict__ wrkmemaux, \
        const float* __restrict__ wrkmem, \
        float* __restrict__ wrkmem2);

INSTANTIATE_CopyCCDataToWrkMem2(READNPOS_NOREAD);
INSTANTIATE_CopyCCDataToWrkMem2(READNPOS_READ);

#if 0
// -------------------------------------------------------------------------
// CopyCCDataToWrkMem2_var: copy cross-covariance matrix between the 
// query and reference structures to section 2 to enable efficient Kabsch 
// algorithm application for multiple structures simultaneously;
// this version differs from the previous one in determining the 
// out-of-bounds condition when the fragment length and position depends on 
// structure lengths;
// READNPOS, template parameter indicating whether nalnposs should be read;
// STEPx5, template parameter, multiply the step by 5 when calculating 
// query and reference positions;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// qryfragfct, fragment factor for query (to be multiplied by step dependent 
// upon lengths);
// rfnfragfct, fragment factor for reference (to be multiplied by step dependent 
// upon lengths);
// fragndx, fragment index determining the fragment size dependent 
// upon lengths;
// NOTE: memory pointers should be aligned!
// wrkmem, working memory, including the section of CC data (saved as 
// whole for each structure) to copy;
// wrkmem2, working memory, including the section of CC data to be written by 
// field;
// 
template<int READNPOS, bool STEPx5>
__global__
void CopyCCDataToWrkMem2_var(
    const uint ndbCstrs,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmem2)
{
    int qryndx = blockIdx.y;//query index in the chunk
    int sfragfct = blockIdx.z;//fragment factor

    CopyCCDataToWrkMem2Helper<READNPOS>(
        STEPx5? GetQryRfnPos_var5: GetQryRfnPos_var, 
        PositionsOutofBounds_var, GetNAlnPoss_var,
        sfragfct, qryndx, ndbCstrs, maxnsteps,  qryfragfct, rfnfragfct, fragndx,
        wrkmem, wrkmem2);
}

// =========================================================================
// Instantiations:
// 
#define INSTANTIATE_CopyCCDataToWrkMem2_var(tpREADNPOS,tpSTEPx5) \
    template __global__ void CopyCCDataToWrkMem2_var<tpREADNPOS,tpSTEPx5>( \
        const uint ndbCstrs, \
        const uint maxnsteps, int qryfragfct, int rfnfragfct, int fragndx, \
        const float* __restrict__ wrkmem, \
        float* __restrict__ wrkmem2);

INSTANTIATE_CopyCCDataToWrkMem2_var(READNPOS_NOREAD,false);
INSTANTIATE_CopyCCDataToWrkMem2_var(READNPOS_NOREAD,true);
#endif

#if 0
// -------------------------------------------------------------------------
// CopyCCDataToWrkMem2_frg: copy cross-covariance matrix between the 
// query and reference structures to section 2 to enable efficient Kabsch 
// algorithm application for multiple structures simultaneously;
// same as CopyCCDataToWrkMem2_var with more extensive parallelization;
// stepx5, multiply the step by 5 when calculating query and reference positions;
// 
__global__
void CopyCCDataToWrkMem2_frg(
    const bool stepx5,
    const uint ndbCstrs,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmem2)
{
    int qryndx = blockIdx.y;//query index in the chunk
    int sfragfct = blockIdx.z;//fragment factor

    CopyCCDataToWrkMem2Helper<READNPOS_NOREAD>(
        stepx5? GetQryRfnPos_frg5: GetQryRfnPos_frg, 
        PositionsOutofBounds_frg, GetNAlnPoss_frg,
        sfragfct, qryndx, ndbCstrs, maxnsteps,  qryfragfct, rfnfragfct, fragndx,
        wrkmem, wrkmem2);
}
#endif

// CopyCCDataToWrkMem2_frg2: copy cross-covariance matrix between the 
// query and reference structures to section 2 to enable efficient Kabsch 
// algorithm application for multiple structures simultaneously;
// same as CopyCCDataToWrkMem2_var with more extensive parallelization;
// depth, superposition depth for calculating query and reference positions;
__global__
void CopyCCDataToWrkMem2_frg2(
    const int depth,
    const uint ndbCstrs,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmem2)
{
    int qryndx = blockIdx.y;//query index in the chunk
    int sfragfct = blockIdx.z;//fragment factor
    fragndx = (sfragfct & 1);

    CopyCCDataToWrkMem2Helper<CHCKCONV_CHECK,READNPOS_NOREAD>(
        GetQryRfnPos_frg2, PositionsOutofBounds_frg, GetNAlnPoss_frg, depth,
        sfragfct, qryndx, ndbCstrs, maxnsteps,  qryfragfct, rfnfragfct, fragndx,
        wrkmemaux, wrkmem, wrkmem2);
}

#if 0
// CopyCCDataToWrkMem2_frgbest: copy cross-covariance matrix between the 
// query and reference structures to section 2 to enable efficient Kabsch 
// algorithm application for multiple structures simultaneously;
// 
__global__
void CopyCCDataToWrkMem2_frgbest(
    const uint ndbCstrs,
    const uint maxnsteps,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmem2)
{
    //reference structure index (blockIdx.x, refn. serial number):
    int dbstrndx = blockIdx.x * CUS1_TBINITSP_CCMCOPY_N + threadIdx.x;
    int qryndx = blockIdx.y;//query index in the chunk
    int sfragfct = blockIdx.z;//fragment factor
    __shared__ int argsCache[3];
    int qrypos = 0, rfnpos = 0, fraglen = 3;

    uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;

    if(dbstrndx < ndbCstrs && threadIdx.y == 0) {
        argsCache[0] = wrkmemaux[mloc + tawmvQRYpos * ndbCstrs + dbstrndx];
        argsCache[1] = wrkmemaux[mloc + tawmvRFNpos * ndbCstrs + dbstrndx];
        //NOTE: actual fragment length has been written:
        argsCache[2] = wrkmemaux[mloc + tawmvSubFragNdx * ndbCstrs + dbstrndx];
    }

    __syncthreads();

    qrypos = argsCache[0]; rfnpos = argsCache[1]; fraglen = argsCache[2];

    CopyCCDataToWrkMem2Helper<READNPOS_NOREAD>(
        GetQryRfnPos_frgbest,
        PositionsOutofBounds_frgbest, GetNAlnPoss_frgbest,
        sfragfct, qryndx, ndbCstrs, maxnsteps,  qrypos, rfnpos, fraglen,
        wrkmem, wrkmem2);
}
#endif



// -------------------------------------------------------------------------
// CopyTfmMtsFromWrkMem2: copy calculated transformation matrix between the 
// query and reference structures from section 2 to enable efficient 
// position-dependent structure processing for multiple structures 
// simultaneously;
// NOTE: data are copied to the area of memory used to hold transformation 
// matrices;
// NOTE: thread block is 2D and copies one structure's data: from:
// NOTE: | struct i | struct i+1 | ... | struct i | ... 
// NOTE: | field1   | field1     | ... | field2   | ...
// NOTE: to 
// NOTE: | struct i          | struct i+1        | ...
// NOTE: | field1,dield2,... | field1,dield2,... | ...
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// NOTE: memory pointers should be aligned!
// wrkmem2, working memory, the section of transformation data to copy;
// wrkmemtm, memory for transformation matrices;
// 
__global__ void CopyTfmMtsFromWrkMem2(
    const uint ndbCstrs,
    const uint maxnsteps,
    const float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemtm)
{
    //cache for transformation matrices: 
    //bank conflicts resolved as long as inner-most dimension is odd
    __shared__ float ccmCache[CUS1_TBINITSP_CCMCOPY_N][nTTranformMatrix+1];
    //index of the first structure to start with (blockIdx.x, refn. serial number):
    int dbstrndx = blockIdx.x * CUS1_TBINITSP_CCMCOPY_N;
    int qryndx = blockIdx.y;//query index in the chunk
    int sfragfct = blockIdx.z;//fragment factor

    int absndx = dbstrndx + threadIdx.x;

    //cache data from gmem: coalesced read
    //first, read nalnposs to slot nTTranformMatrix
    if(absndx < ndbCstrs && threadIdx.y == nTTranformMatrix) {
        //NOTE: block's dim y is at least 16>nTTranformMatrix
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTWorkingMemoryVars + twmvNalnposs) * ndbCstrs;
        ccmCache[threadIdx.x][threadIdx.y] = wrkmem2[mloc + absndx];
    }

    __syncthreads();

    //slot nTTranformMatrix will not be overwritten below: no sync;
    //threadIdx.x corresponds to refn. structure index
    int nalnposs = ccmCache[threadIdx.x][nTTranformMatrix];

    if(absndx < ndbCstrs && threadIdx.y < nTTranformMatrix && nalnposs) {
        //read tfm matrix only if nalnposs >0
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTWorkingMemoryVars + threadIdx.y) * ndbCstrs;
        ccmCache[threadIdx.x][threadIdx.y] = wrkmem2[mloc + absndx];
    }

    __syncthreads();

    //write data to gmem: iterative coalesced write
    for(int reldbndx = threadIdx.y; reldbndx < CUS1_TBINITSP_CCMCOPY_N; reldbndx += blockDim.y) {
        absndx = dbstrndx + reldbndx;
        if(absndx < ndbCstrs && threadIdx.x < nTTranformMatrix &&
           ccmCache[reldbndx][nTTranformMatrix]) {
            //write tfm matrix only if nalnposs >0;
            //NOTE: tfm matrices have been initialized before
            uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + absndx) * nTTranformMatrix;
            wrkmemtm[mloc + threadIdx.x] = ccmCache[reldbndx][threadIdx.x];
        }
    }
}
