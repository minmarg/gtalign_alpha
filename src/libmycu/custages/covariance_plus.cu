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
#include "stagecnsts.cuh"
#include "covariance.cuh"
#include "covariance_plus.cuh"

// -------------------------------------------------------------------------
// FindD02ThresholdsCCM: efficiently find distance d02s thresholds for the 
// inclusion of aligned positions for CCM and rotation matrix calculations;
// NOTE: thread block is 1D and processes alignment along structure
// positions;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db structure positions in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// n1, starting position that determines positions in query and reference;
// step, step size in positions used to traverse query and reference 
// ungapped alignments;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary diagonal buffers filled with positional 
// scores;
// wrkmem, working memory, including the section of CC data;
// wrkmemaux, auxiliary working memory;
// 
template<int READCNST>
__global__
void FindD02ThresholdsCCM(
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    int n1, int step,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux)
{
    // blockIdx.x is the reference serial number;
    // blockIdx.y is the query serial number;
    // blockIdx.z is the fragment factor;
    //cache for minimum scores: 
    //no bank conflicts as long as inner-most dim is odd
    constexpr int smidim = 3;//top three min scores
    __shared__ float ccmCache[smidim * CUS1_TBINITSP_FINDD02_ITRD_XDIM];
    uint qryndx = blockIdx.y;//query serial number
    uint sfragfct = blockIdx.z;//fragment factor
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    uint /* qrydst, */ dbstrdst;
    n1 += sfragfct * step;
    int qrypos = myhdmax(0,n1);//starting query position
    int rfnpos = myhdmax(-n1,0);//starting reference position

    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse ccmCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(blockIdx.x, (int*)ccmCache);
        GetQueryLenDst(qryndx, (int*)ccmCache + 2);
    }

    __syncthreads();

    //NOTE: no bank conflict when two threads from the same warp access the same address;
    dbstrlen = ((int*)ccmCache)[0]; dbstrdst = ((int*)ccmCache)[1];
    qrylen = ((int*)ccmCache)[2]; //qrydst = ((int*)ccmCache)[3];

    __syncthreads();


    if(PositionsOutofBounds(qrylen, dbstrlen, qrypos, rfnpos, 0, 0, 0/*args1-3(unused)*/))
        //all threads in the block exit;
        //nalnposs is left as initialized
        return;


    if(READCNST == READCNST_CALC2) {
        if(threadIdx.x == 0) {
            //NOTE: reuse ccmCache[0] to contain twmvLastD02s. ccmCache[1] twmvNalnposs
            uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + blockIdx.x/*dbstrndx*/) * nTWorkingMemoryVars;
            ccmCache[1] = wrkmem[mloc + twmvNalnposs];
        }
    }

    int maxnalnposs = GetNAlnPoss(qrylen, dbstrlen, qrypos, rfnpos, 0, 0, 0/*args1-3(unused)*/);

    if(READCNST == READCNST_CALC2) {
        __syncthreads();

        int nalnposs = ccmCache[1];
        if(nalnposs == maxnalnposs)
            //all threads in the block exit;
            return;

        //cache will be overwritten below, sync
        __syncthreads();
    }


    //initialize cache
    #pragma unroll
    for(int i = 0; i < smidim; i++)
        ccmCache[threadIdx.x * smidim + i] = CP_LARGEDST;

    for(int rpos = threadIdx.x; qrypos + rpos < qrylen && rfnpos + rpos < dbstrlen;
        rpos += blockDim.x)
    {
        //manually unroll along alignment
        int mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs;
        GetMinScoreOneAlnPos<smidim>(
            mloc + dbstrdst + rpos,//position for scores
            tmpdpdiagbuffers,
            ccmCache
        );
        //no sync: every thread works in its own space (of ccmCache)
    }
    //sync now:
    __syncthreads();

    //unroll until reaching warpSize; 
    for(int xdim = (CUS1_TBINITSP_FINDD02_ITRD_XDIM>>1); xdim >= 32; xdim >>= 1) {
        int tslot = threadIdx.x * smidim;
        //ccmCache will contain 3x32 (or length-size) (possibly equal) minimum scores 
        if(threadIdx.x < xdim &&
           qrypos + threadIdx.x + xdim < qrylen &&
           rfnpos + threadIdx.x + xdim < dbstrlen)
            StoreMinDstSrc(ccmCache + tslot, ccmCache + tslot + xdim * smidim);
//             //TODO:REMOVE:
//             ccmCache[tslot] = 
//                 myhdmin(ccmCache[tslot], ccmCache[tslot + xdim * smidim]);

        __syncthreads();
    }

    //unroll warp
    if(threadIdx.x < 32/*warpSize*/) {
        for(int xdim = (32>>1); xdim >= 1; xdim >>= 1) {
            int tslot = threadIdx.x * smidim;
            if(threadIdx.x < xdim)
                StoreMinDstSrc(ccmCache + tslot, ccmCache + tslot + xdim * smidim);
            __syncwarp();
        }
//         //TODO:REMOVE:
//         float min = ccmCache[threadIdx.x * smidim];
//         min = myhdmin(min, __shfl_down_sync(0xffffffff, min, 16));
//         min = myhdmin(min, __shfl_down_sync(0xffffffff, min, 8));
//         min = myhdmin(min, __shfl_down_sync(0xffffffff, min, 4));
//         //sort four minimum scores:
//         ccmCache[threadIdx.x] = min;
//         __syncwarp();
//         //sort within groups of size two:
//         if((threadIdx.x & 1) == 0 && ccmCache[threadIdx.x^1] < ccmCache[threadIdx.x])
//             //only threads 0 and 2 swap using different smem adresses: no race condition
//             myhdswap(ccmCache[threadIdx.x^1], ccmCache[threadIdx.x]);
//         __syncwarp();
//         //sort across groups of size two:
//         if(threadIdx.x < 2 && ccmCache[threadIdx.x+2] < ccmCache[threadIdx.x])
//             //no race condition: same as above
//             myhdswap(ccmCache[threadIdx.x+2] < ccmCache[threadIdx.x]);
//         __syncwarp();
//         //final comparison: thread 2 will contain 3rd minimum value
//         if(threadIdx.x == 1 && ccmCache[threadIdx.x+1] < ccmCache[threadIdx.x])
//             myhdswap(ccmCache[threadIdx.x+1], ccmCache[threadIdx.x]);
    }
//     //TODO:REMOVE:
//     __syncwarp();

    //write to gmem the minimum score that ensures at least 3 aligned positions:
    if(threadIdx.x == 2) {
        float d0 = GetD0(qrylen, dbstrlen);
        float d02s = GetD02s(d0);
        if(READCNST == READCNST_CALC2) d02s += D02s_PROC_INC;

        float min3 = ccmCache[threadIdx.x];

        //TODO: move the clause (maxnalnposs <= 3) along with the write to gmem up
        if(CP_LARGEDST_cmp < min3 || min3 < d02s || maxnalnposs <= 3)
            //max number of alignment positions (maxnalnposs) <3;
            //use the dfault threshold
            min3 = d02s;
        else {//round the 3rd minimum score to the next multiple of 0.5:
            //obtained from d02s + k*0.5 >= min3
            min3 = d02s + ceilf((min3 - d02s) * 2.0f) * 0.5f;
            //d0 = floorf(min3);
            //d02s = min3 - d0;
            //if(d02s) min3 = d0 + ((d02s <= 0.5f)? 0.5f: 1.0f);
        }

        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvLastD02s) * ndbCstrs;
        wrkmemaux[mloc + blockIdx.x/*dbstrndx*/] = min3;
    }
}

// =========================================================================
// Instantiations
//
#define INSTANTIATE_FindD02ThresholdsCCM(tpREADCNST) \
    template __global__ void FindD02ThresholdsCCM<tpREADCNST>( \
        const uint ndbCstrs, const uint ndbCposs, \
        const uint maxnsteps, int n1, int step, \
        const float* __restrict__ tmpdpdiagbuffers, \
        float* __restrict__ wrkmem, \
        float* __restrict__ wrkmemaux);

INSTANTIATE_FindD02ThresholdsCCM(READCNST_CALC);
INSTANTIATE_FindD02ThresholdsCCM(READCNST_CALC2);

// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// CalcCCMatrices64Extended: calculate cross-covariance matrix between the 
// query and reference structures based on aligned positions within 
// given distance;
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// NOTE: Version for CUS1_TBINITSP_CCMCALC_XDIM==64!
// nqystrs, total number of queries;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db structure positions in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// n1, starting position that determines positions in query and reference;
// step, step size in positions used to traverse query and reference 
// ungapped alignments;
// alnlen, maximum alignment length which corresponds to the minimum 
// length of the structures being compared;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary diagonal buffers filled with positional 
// scores;
// wrkmem, working memory, including the section of CC data;
// wrkmemaux, auxiliary working memory;
// 
template<int READCNST>
__global__
void CalcCCMatrices64Extended(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    int n1, int step,
    const float* __restrict__ tmpdpdiagbuffers,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem)
{
    // blockIdx.x is the block index of positions for query-reference pair;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number TIMES fragment factor;
    //cache for the cross-covarinace matrix and related data: 
    //no bank conflicts as long as inner-most dim is odd
    constexpr int neffds = twmvEndOfCCDataExt;//effective number of fields
    constexpr int smidim = neffds+1;
    __shared__ float ccmCache[smidim * CUS1_TBINITSP_CCMCALC_XDIM];
    //relative position index:
    const uint ndx0 = blockIdx.x * blockDim.x * CUS1_TBINITSP_CCMCALC_XFCT;
    const uint ndx = ndx0 + threadIdx.x;
    uint sfragfct = blockIdx.z / nqystrs;//fragment factor
    uint qryndx = blockIdx.z - sfragfct * nqystrs;//query serial number
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;
    n1 += sfragfct * step;
    int qrypos = myhdmax(0,n1);//starting query position
    int rfnpos = myhdmax(-n1,0);//starting reference position

    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse ccmCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(blockIdx.y, (int*)ccmCache);
        GetQueryLenDst(qryndx, (int*)ccmCache + 2);
    }

    __syncthreads();

    //NOTE: no bank conflict when two threads from the same warp access the same address;
    //blockDim.x includes only several warps
    dbstrlen = ((int*)ccmCache)[0]; dbstrdst = ((int*)ccmCache)[1];
    qrylen = ((int*)ccmCache)[2]; qrydst = ((int*)ccmCache)[3];

    __syncthreads();


    if(qrylen <= qrypos + ndx0 || dbstrlen <= rfnpos + ndx0)
        //all threads in the block exit if thread 0 is out of bounds
        return;

    if(PositionsOutofBounds(qrylen, dbstrlen, qrypos, rfnpos, 0, 0, 0/*args1-3(unused)*/))
        //all threads in the block exit
        return;


    if(threadIdx.x == 0) {
        //NOTE: reuse ccmCache[0] to contain twmvLastD02s. ccmCache[1] twmvNalnposs
        //structure-specific-formatted data
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvLastD02s) * ndbCstrs;
        ccmCache[0] = wrkmemaux[mloc + blockIdx.y/*dbstrndx*/];

        if(READCNST == READCNST_CALC2) {
            mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + blockIdx.y/*dbstrndx*/) * nTWorkingMemoryVars;
            ccmCache[1] = wrkmem[mloc + twmvNalnposs];
        }
    }

    __syncthreads();

   float d02s = ccmCache[0];
   int maxnalnposs = GetNAlnPoss(qrylen, dbstrlen, qrypos, rfnpos, 0, 0, 0/*args1-3(unused)*/);

    if(READCNST == READCNST_CALC2) {
        int nalnposs = ccmCache[1];
        if(nalnposs == maxnalnposs)
            //all threads in the block exit;
            return;
    }

    //cache will be overwritten below, sync
    __syncthreads();


    //initialize cache
    #pragma unroll
    for(int i = 0; i < neffds; i++)
        ccmCache[threadIdx.x * smidim + i] = 0.0f;

    #pragma unroll
    for(int i = 0; i < CUS1_TBINITSP_CCMCALC_XFCT; i++) {
        //manually unroll along data blocks by a factor of CUS1_TBINITSP_CCMCALC_XFCT
        int mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs;
        int pos0 = ndx + i * blockDim.x;//position index starting from 0
        if(!(qrypos + pos0 < qrylen && rfnpos + pos0 < dbstrlen))
            break;
        UpdateCCMOneAlnPosExtended<smidim>(
            d02s,
            qrydst + qrypos + pos0,//query position
            dbstrdst + rfnpos + pos0,//reference position
            mloc + dbstrdst + pos0,//position for scores
            tmpdpdiagbuffers,
            ccmCache
        );
        //no sync: every thread works in its own space (of ccmCache)
    }

    //sync now:
    __syncthreads();

    //unroll by a factor 2
    if(threadIdx.x < (CUS1_TBINITSP_CCMCALC_XDIM>>1)) {
        #pragma unroll
        for(int i = 0; i < neffds; i++)
            ccmCache[threadIdx.x * smidim + i] +=
                ccmCache[(threadIdx.x + (CUS1_TBINITSP_CCMCALC_XDIM>>1)) * smidim + i];
    }

    __syncthreads();

    //unroll warp
    if(threadIdx.x < 32) {
        #pragma unroll
        for(int i = 0; i < neffds; i++) {
            float sum = ccmCache[threadIdx.x * smidim + i];
            sum = mywarpreducesum(sum);
            //write to the first data slot of SMEM
            if(threadIdx.x == 0) ccmCache[i] = sum;
        }
    }

    //in case of neffds gets larger than warpSize
    __syncthreads();

    //add the result and write to global memory
    if(threadIdx.x < neffds) {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + blockIdx.y/*dbstrndx*/) * nTWorkingMemoryVars;
        atomicAdd(&wrkmem[mloc + threadIdx.x], ccmCache[threadIdx.x]);
    }
}

// =========================================================================
// Instantiations
//
#define INSTANTIATE_CalcCCMatrices64Extended(tpREADCNST) \
    template __global__ void CalcCCMatrices64Extended<tpREADCNST>( \
        const uint nqystrs, const uint ndbCstrs, const uint ndbCposs, \
        const uint maxnsteps, int n1, int step, \
        const float* __restrict__ tmpdpdiagbuffers, \
        const float* __restrict__ wrkmemaux, \
        float* __restrict__ wrkmem);

INSTANTIATE_CalcCCMatrices64Extended(READCNST_CALC);
INSTANTIATE_CalcCCMatrices64Extended(READCNST_CALC2);

// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// CalcScoresUnrl: calculate/reduce scores for obtained superpositions; 
// save partial sums;
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// NOTE: Universal version for any CUS1_TBSP_SCORE_XDIM multiple of 32;
// SAVEPOS, template parameter to request saving positional scores;
// CHCKALNLEN, template parameter to request checking whether alignment 
// length has changed;
// nqystrs, total number of queries;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db structure positions in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// n1, starting position that determines positions in query and reference;
// step, step size in positions used to traverse query and reference 
// ungapped alignments;
// NOTE: memory pointers should be aligned!
// wrkmemtm, memory for transformation matrices;
// wrkmemaux, auxiliary working memory;
// tmpdpdiagbuffers, temporary diagonal buffers reused here for saving 
// positional scores;
// NOTE: keep #registers <= 32
// 
template<int SAVEPOS, int CHCKALNLEN>
__global__
void CalcScoresUnrl(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    int n1, int step,
    const float* __restrict__ wrkmemtm,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers)
{
    // blockIdx.x is the block index of positions for query-reference pair;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number TIMES fragment factor;
    //no bank conflicts as long as inner-most dim is odd
    constexpr int pad = 1;//padding
    //cache for scores and transformation matrix: 
    __shared__ float scvCache[pad + CUS1_TBSP_SCORE_XDIM + nTTranformMatrix];
    //pointer to transformation matrix;
    float* tfmCache = scvCache + pad + CUS1_TBSP_SCORE_XDIM;
    //relative position index:
    const uint ndx0 = blockIdx.x * blockDim.x * CUS1_TBSP_SCORE_XFCT;
    const uint ndx = ndx0 + threadIdx.x;
    uint sfragfct = blockIdx.z / nqystrs;//fragment factor
    uint qryndx = blockIdx.z - sfragfct * nqystrs;//query serial number
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;
    n1 += sfragfct * step;
    int qrypos = myhdmax(0,n1);//starting query position
    int rfnpos = myhdmax(-n1,0);//starting reference position
    float d02;

    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse scvCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(blockIdx.y, (int*)scvCache);
        GetQueryLenDst(qryndx, (int*)scvCache + 2);
    }

    __syncthreads();

    //NOTE: no bank conflict when two threads from the same warp access the same address;
    //blockDim.x includes only several warps
    dbstrlen = ((int*)scvCache)[0]; dbstrdst = ((int*)scvCache)[1];
    qrylen = ((int*)scvCache)[2]; qrydst = ((int*)scvCache)[3];

    __syncthreads();


    if(qrylen <= qrypos + ndx0 || dbstrlen <= rfnpos + ndx0)
        //all threads in the block exit if thread 0 is out of bounds
        return;

    if(PositionsOutofBounds(qrylen, dbstrlen, qrypos, rfnpos, 0, 0, 0/*args1-3(unused)*/))
        //all threads in the block exit;
        return;


    if(CHCKALNLEN == CHCKALNLEN_CHECK) {
        if(threadIdx.x == 0) {
            //NOTE: reuse scvCache[0] to twmvNalnposs
            uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + blockIdx.y/*dbstrndx*/) * nTWorkingMemoryVars;
            scvCache[0] = wrkmem[mloc + twmvNalnposs];
        }

        __syncthreads();

        int nalnposs = scvCache[0];
        int maxnalnposs = GetNAlnPoss(qrylen, dbstrlen, qrypos, rfnpos, 0, 0, 0/*args1-3(unused)*/);
        if(nalnposs == maxnalnposs)
            //score has been calculated before; 
            //all threads in the block exit;
            return;
    }


    d02 = GetD02(qrylen, dbstrlen);

    //initialize cache
    scvCache[pad + threadIdx.x] = 0.0f;

    //read transformation matrix for query-reference pair
    if(threadIdx.x < nTTranformMatrix) {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + blockIdx.y/*dbstrndx*/) * nTTranformMatrix;
        tfmCache[threadIdx.x] = wrkmemtm[mloc + threadIdx.x];
    }

    __syncthreads();


    #pragma unroll
    for(int i = 0; i < CUS1_TBSP_SCORE_XFCT; i++) {
        //manually unroll along data blocks by a factor of CUS1_TBSP_SCORE_XFCT
        int mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs;
        int pos0 = ndx + i * blockDim.x;//position index starting from 0
        if(!(qrypos + pos0 < qrylen && rfnpos + pos0 < dbstrlen))
            break;
        UpdateOneAlnPosScore<SAVEPOS,CHCKDST_NOCHECK>(
            d02, d02,
            qrydst + qrypos + pos0,//query position
            dbstrdst + rfnpos + pos0,//reference position
            mloc + dbstrdst + pos0,//position for scores
            tfmCache,
            scvCache + pad,
            tmpdpdiagbuffers
        );
        //no sync: every thread works in its own space (of scvCache)
    }

    //sync now:
    __syncthreads();

    //unroll until reaching warpSize 
    #pragma unroll
    for(int xdim = (CUS1_TBSP_SCORE_XDIM>>1); xdim >= 32; xdim >>= 1) {
        if(threadIdx.x < xdim)
            scvCache[pad + threadIdx.x] +=
                scvCache[pad + threadIdx.x + xdim];

        __syncthreads();
    }

    //unroll warp
    if(threadIdx.x < 32/*warpSize*/) {
        float sum = scvCache[pad + threadIdx.x];
        sum = mywarpreducesum(sum);
        //write to the first data slot of SMEM
        if(threadIdx.x == 0) scvCache[0] = sum;
    }

    //add the score and write to global memory
    if(threadIdx.x == 0) {
        //structure-specific-formatted data; scvCache[0] is the reduced score
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvScore) * ndbCstrs;
        atomicAdd(&wrkmemaux[mloc + blockIdx.y/*dbstrndx*/], scvCache[0]);
    }
}

// -------------------------------------------------------------------------
// Instantiations
// 
#define INSTANTIATE_CalcScoresUnrl(tpSAVEPOS,tpCHCKALNLEN) \
    template __global__ void CalcScoresUnrl<tpSAVEPOS,tpCHCKALNLEN>( \
        const uint nqystrs, const uint ndbCstrs, const uint ndbCposs, \
        const uint maxnsteps, int n1, int step, \
        const float* __restrict__ wrkmemtm, \
        const float* __restrict__ wrkmem, \
        float* __restrict__ wrkmemaux, \
        float* __restrict__ tmpdpdiagbuffers);

INSTANTIATE_CalcScoresUnrl(SAVEPOS_SAVE,CHCKALNLEN_NOCHECK);
INSTANTIATE_CalcScoresUnrl(SAVEPOS_SAVE,CHCKALNLEN_CHECK);
INSTANTIATE_CalcScoresUnrl(SAVEPOS_NOSAVE,CHCKALNLEN_CHECK);

// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// CalcScoresUnrl_frg2: calculate/reduce initial scores for obtained 
// superpositions during extensive fragment-based search of optimal superpositions; 
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// NOTE: Universal version for any CUS1_TBSP_SCORE_XDIM multiple of 32;
// thrscorefactor, threshold score factor;
// dynamicorientation, flag controlling transformation orientation;
// depth, superposition depth for calculating query and reference positions;
// nqystrs, total number of queries;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db structure positions in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// qryfragfct, fragment factor for query (to be multiplied by step dependent upon lengths);
// rfnfragfct, fragment factor for reference (to be multiplied by step dependent upon lengths);
// fragndx, fragment index determining the fragment size dependent upon lengths;
// NOTE: memory pointers should be aligned!
// wrkmemtm, memory OF transformation matrices;
// wrkmemaux, auxiliary working memory;
// NOTE: keep #registers <= 32
// 
__global__
void CalcScoresUnrl_frg2(
    const float thrscorefactor,
    const bool dynamicorientation,
    const int depth,
    const uint ndbCstrs,
    // const uint ndbCposs,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemaux)
{
    // blockIdx.x is the reference serial number;
    // blockIdx.y is fragment factor;
    // blockIdx.z is the query serial number;
    //no bank conflicts as long as inner-most dim is odd
    constexpr int pad = 1;//padding
    //cache for scores and transformation matrix: 
    __shared__ float scvCache[pad + CUS1_TBSP_SCORE_FRG2_HALT_CHK_XDIM + nTTranformMatrix];
    //pointer to transformation matrix;
    float* tfmCache = scvCache + pad + CUS1_TBSP_SCORE_FRG2_HALT_CHK_XDIM;
    const uint dbstrndx = blockIdx.x;
    const uint sfragfct = blockIdx.y;//fragment factor
    const uint qryndx = blockIdx.z;//query serial number
    fragndx = (sfragfct & 1);
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;
    int qrypos, rfnpos;//starting query and reference position


    //check for global convergence first;
    //NOTE: tawmvConverged & tawmvInitialBest should be <nTTranformMatrix!
    if(threadIdx.x == tawmvConverged || threadIdx.x == tawmvInitialBest) {
        uint mloc = ((qryndx * maxnsteps + 0/*sfragfct*/) * nTAuxWorkingMemoryVars) * ndbCstrs;
        tfmCache[threadIdx.x] = wrkmemaux[mloc + threadIdx.x * ndbCstrs + dbstrndx];
    }

    __syncthreads();

    const int convflag = (int)tfmCache[tawmvConverged];

    if((convflag) & 
       (CONVERGED_SCOREDP_bitval | CONVERGED_NOTMPRG_bitval | CONVERGED_LOWTMSC_bitval))
        //(termination flag for this pair is set);
        //all threads in the block exit;
        return;

    float uibesthr = tfmCache[tawmvInitialBest];//unrefined initial best score


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse scvCache; do not overwrite tfmCache[0] used for reading conv flag above;
    if(threadIdx.x < 2) {
        GetDbStrLenDst(dbstrndx, (int*)scvCache);
        GetQueryLenDst(qryndx, (int*)scvCache + 2);
    }

    __syncthreads();

    //NOTE: no bank conflict when two threads from the same warp access the same address;
    //blockDim.x includes only several warps
    dbstrlen = ((int*)scvCache)[0]; dbstrdst = ((int*)scvCache)[1];
    qrylen = ((int*)scvCache)[2]; qrydst = ((int*)scvCache)[3];

    __syncthreads();


    //reverse transformation wrt query
    const bool reverse = dynamicorientation? !(qrylen < dbstrlen): true;

    GetQryRfnPos_frg2(
        depth,
        qrypos, rfnpos, qrylen, dbstrlen, sfragfct, qryfragfct, rfnfragfct, fragndx
    );

    int fraglen = GetNAlnPoss_frg(
            qrylen, dbstrlen, 0/*qrypos,unused*/, 0/*rfnpos,unused*/,
            qryfragfct/*unused*/, rfnfragfct/*unused*/, fragndx);

    //if fragment is out of bounds (tfm not calculated): all threads in the block exit
    if(qrylen < qrypos + fraglen || dbstrlen < rfnpos + fraglen) return;

    const int maxalnlen = myhdmin(qrylen, dbstrlen);
    fraglen = myhdmin(maxalnlen, CUSF_TBSP_INDEX_SCORE_POSLIMIT2);
    qrypos = myhdmax(0, qrypos - (fraglen>>1));
    rfnpos = myhdmax(0, rfnpos - (fraglen>>1));
    qrylen = myhdmin(qrylen, qrypos + fraglen);
    dbstrlen = myhdmin(dbstrlen, rfnpos + fraglen);
    qrypos = myhdmax(0, qrylen - fraglen);
    rfnpos = myhdmax(0, dbstrlen - fraglen);

    //NOTE: update the threshold:
    uibesthr = __fdividef(uibesthr * (float)fraglen, maxalnlen) * thrscorefactor;


    const float d02 = GetD02(qrylen, dbstrlen);//distance threshold

    //initialize cache
    scvCache[pad + threadIdx.x] = 0.0f;

    //read transformation matrix for query-reference pair
    if(threadIdx.x < nTTranformMatrix) {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix;
        tfmCache[threadIdx.x] = wrkmemtm[mloc + threadIdx.x];
    }

    __syncthreads();


    for(qrypos += threadIdx.x, rfnpos += threadIdx.x;
        qrypos < qrylen && rfnpos < dbstrlen;
        qrypos += blockDim.x, rfnpos += blockDim.x)
    {
        //manually unroll along data blocks
        UpdateOneAlnPosScore_frg2<CHCKDST_NOCHECK>(
            reverse,
            d02, d02,
            qrydst + qrypos,//query position
            dbstrdst + rfnpos,//reference position
            tfmCache,
            scvCache + pad
        );
        //no sync: every thread works in its own space (of scvCache)
    }

    //sync now:
    __syncthreads();

    #pragma unroll
    for(int xdim = (CUS1_TBSP_SCORE_FRG2_HALT_CHK_XDIM>>1); xdim >= 32; xdim >>= 1) {
        if(threadIdx.x < xdim)
            scvCache[pad + threadIdx.x] +=
                scvCache[pad + threadIdx.x + xdim];
        __syncthreads();
    }

    //unroll warp
    if(threadIdx.x < 32/*warpSize*/) {
        float sum = scvCache[pad + threadIdx.x];
        sum = mywarpreducesum(sum);
        //write to the first data slot of SMEM
        if(threadIdx.x == 0) scvCache[0] = sum;
    }

    //add the score and write to global memory
    if(threadIdx.x == 0) {
        //structure-specific-formatted data; scvCache[0] is the reduced score
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        if(scvCache[0] < uibesthr)
            wrkmemaux[mloc + dbstrndx] =
                (sfragfct == 0)? (convflag | CONVERGED_SCOREDP_bitval): CONVERGED_SCOREDP_bitval;
    }
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
