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
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/covariance.cuh"
#include "libmycu/custages/transform.cuh"
#include "libmycu/custages/scoring.cuh"
#include "libmycu/custages/fields.cuh"
#include "linear_scoring.cuh"

// -------------------------------------------------------------------------
// PositionalScoresFromIndexLinear: calculate scores at each reference 
// position for following reduction, using index; scores follow from 
// superpositions based on fragments;
// NOTE: thread block is 1D and processes reference fragment along structure
// positions;
// stacksize, dynamically determined stack size;
// stepx5, multiply the step by 5 when calculating query and reference positions;
// nqystrs, total number of queries;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db structure positions in the chunk;
// maxnsteps, max number of steps (blockIdx.z) that can be performed in one pass;
// qryfragfct, argument 1 for calculating starting query position;
// rfnfragfct, argument 2 for calculating starting reference position;
// fragndx, argument 3 for calculating fragment length;
// NOTE: memory pointers should be aligned!
// wrkmemtm, memory for transformation matrices;
// tmpdpdiagbuffers, temporary diagonal buffers reused here for saving 
// positional scores;
// 
template<int SECSTRFILT>
__global__
void PositionalScoresFromIndexLinear(
    const int stacksize,
    const bool stepx5,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int qryfragfct,
    const int rfnfragfct,
    const int fragndx,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ tmpdpdiagbuffers)
{
    // blockIdx.x is the block index of positions for query-reference pair;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number TIMES fragment factor;
    uint sfragfct = blockIdx.z / nqystrs;//fragment factor
    uint qryndx = blockIdx.z - sfragfct * nqystrs;//query serial number
    uint dbstrndx = blockIdx.y;//reference serial number

    //padding to resolve bank conflicts; stack size
    //enum {pad = 1, stacksize = 17};
    //use dynamically allocated SM in case when allocation exceeds 48KB:
    extern __shared__ float dSMEM[];
    //transformation matrix;
    //__shared__ float tfmCache[nTTranformMatrix];
    float* tfmCache = dSMEM;
    //cache for scores: 
    ///__shared__ float scvCache[pad + CUSF_TBSP_INDEX_SCORE_XDIM];
    //cache for query positional indices (to keep track of insertions wrt query later): 
    ///__shared__ float qnxCache[pad + CUSF_TBSP_INDEX_SCORE_XDIM];
    //stack for traversing the index tree:
    //__shared__ float trtStack[CUSF_TBSP_INDEX_SCORE_XDIM][stacksize * nStks_];
#if (CUSF_TBSP_INDEX_SCORE_XFCT == 1)
    float* trtStack = tfmCache;
#else
    float* trtStack = tfmCache + nTTranformMatrix;
#endif
    //relative position index:
    const uint ndx0 = blockIdx.x * blockDim.x * CUSF_TBSP_INDEX_SCORE_XFCT;
    const uint ndx = ndx0 + threadIdx.x;
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    int qrydst, dbstrdst;


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse tfmCache
    if(threadIdx.x < 2) GetDbStrLenDst(dbstrndx, (int*)tfmCache);
    if(threadIdx.x < 2) GetQueryLenDst(qryndx, (int*)tfmCache + 2);


    __syncthreads();

    //NOTE: no bank conflicts when accessing the same address;
    dbstrlen = ((int*)tfmCache)[0]; dbstrdst = ((int*)tfmCache)[1];
    qrylen = ((int*)tfmCache)[2]; qrydst = ((int*)tfmCache)[3];

    __syncthreads();


    int qrypos, rfnpos;

    if(stepx5)
        GetQryRfnPos_frg5(qrypos, rfnpos,  qrylen, dbstrlen, sfragfct, qryfragfct, rfnfragfct, fragndx);
    else GetQryRfnPos_frg(qrypos, rfnpos,  qrylen, dbstrlen, sfragfct, qryfragfct, rfnfragfct, fragndx);

    int fraglen = GetNAlnPoss_frg(
            qrylen, dbstrlen, qrypos, rfnpos, qryfragfct, rfnfragfct, fragndx);

    //if fragment is out of bounds (tfm not calculated): all threads in the block exit
    if(qrylen < qrypos + fraglen || dbstrlen < rfnpos + fraglen) return;

    //qrypos = myhdmax(0, qrypos - CUSF_TBSP_INDEX_SCORE_POSLIMIT);
    rfnpos = myhdmax(0, rfnpos - CUSF_TBSP_INDEX_SCORE_POSLIMIT);
    dbstrlen = myhdmin(dbstrlen, rfnpos + CUSF_TBSP_INDEX_SCORE_POSLIMIT2);
    rfnpos = myhdmax(0, dbstrlen - CUSF_TBSP_INDEX_SCORE_POSLIMIT2);

    //all threads in the block exit if thread 0 is out of reference bounds
    if(dbstrlen <= rfnpos + ndx0) return;


    //initialize cache
    ///scvCache[pad + threadIdx.x] = 0.0f;
    ///qnxCache[pad + threadIdx.x] = 0.0f;

    //read transformation matrix for query-reference pair
    if(threadIdx.x < nTTranformMatrix) {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix;
        tfmCache[threadIdx.x] = wrkmemtm[mloc + threadIdx.x];
    }

    __syncthreads();


    for(int i = 0; i < CUSF_TBSP_INDEX_SCORE_XFCT; i++) {
        //manually unroll along data blocks by a factor of CUSF_TBSP_INDEX_SCORE_XFCT
        char rss;
        float rx, ry, rz;
        int pos0 = ndx + i * blockDim.x;//position index starting from 0

        if(rfnpos + pos0 < dbstrlen)
        {
            int dpos = dbstrdst + rfnpos + pos0;

            rx = GetDbStrCoord<pmv2DX>(dpos);
            ry = GetDbStrCoord<pmv2DY>(dpos);
            rz = GetDbStrCoord<pmv2DZ>(dpos);
            if(SECSTRFILT == 1) rss = GetDbStrSS(dpos);
            transform_point(tfmCache, rx, ry, rz);
        }

#if (CUSF_TBSP_INDEX_SCORE_XFCT == 1)
        //same smem buffer used for both
        __syncthreads();
#endif

        if(rfnpos + pos0 < dbstrlen)
        {
            int bestqnx = -1;//query index of the position nearest to a reference atom
            float bestdst2 = 9.9e6f;//squared distance to the query atom at bestqnx

            //nearest neighbour using the index tree:
            NNByIndex<SECSTRFILT>(
                stacksize,
                bestqnx, bestdst2,  rx, ry, rz,
                rss,
                qrydst, (qrylen >> 1)/*root*/, 0/*dimndx*/,
                trtStack + stacksize * nStks_ * threadIdx.x);

            //WRITE best distances and corresponding query positional indices
            //starting with index 0 for tmpdpdiagbuffers
            //(x2 to account for distances and indices; since tmpdpdiagbuffers 
            //always contains at least two diagonals, no access error):
            int mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs * 2;
            tmpdpdiagbuffers[mloc + dbstrdst + pos0] = bestdst2;
            tmpdpdiagbuffers[mloc + dbstrdst + pos0 + ndbCposs] = bestqnx;
        }
    }
}

// Instantiations
// 
#define INSTANTIATE_PositionalScoresFromIndexLinear(tpSECSTRFILT) \
    template __global__ void PositionalScoresFromIndexLinear<tpSECSTRFILT>( \
        const int stacksize, const bool stepx5, const uint nqystrs, \
        const uint ndbCstrs, const uint ndbCposs, const uint maxnsteps, \
        const int qryfragfct, const int rfnfragfct, int fragndx, \
        const float* __restrict__ wrkmemtm, float* __restrict__ tmpdpalnpossbuffer);

INSTANTIATE_PositionalScoresFromIndexLinear(0);
INSTANTIATE_PositionalScoresFromIndexLinear(1);

// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// ReduceScoresLinear: reduce positional scores obtained previously by
// PositionalScoresFromIndexLinear; 
// NOTE: thread block is 1D and processes reference fragment along structure
// positions;
// stepx5, multiply the step by 5 when calculating query and reference positions;
// nqystrs, total number of queries;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db structure positions in the chunk;
// maxnsteps, max number of steps (blockIdx.z) that can be performed in one pass;
// qryfragfct, argument 1 for calculating starting query position;
// rfnfragfct, argument 2 for calculating starting reference position;
// fragndx, argument 3 for calculating fragment length;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary diagonal buffers reused here for reading 
// positional scores;
// wrkmemaux, auxiliary working memory;
// 
__global__
void ReduceScoresLinear(
    const bool stepx5,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int qryfragfct,
    const int rfnfragfct,
    const int fragndx,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux)
{
    // blockIdx.x is the block index of positions for query-reference pair;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number TIMES fragment factor;
    uint sfragfct = blockIdx.z / nqystrs;//fragment factor
    uint qryndx = blockIdx.z - sfragfct * nqystrs;//query serial number
    uint dbstrndx = blockIdx.y;//reference serial number

    //padding to resolve bank conflicts and include one extra position;
    enum {pad = 1};
    //cache for scores: 
    __shared__ float scvCache[pad + CUSF_TBSP_INDEX_SCORE_REDUCE_XDIM];
    //cache for query positional indices (to keep track of insertions wrt query later): 
    __shared__ int qnxCache[pad + CUSF_TBSP_INDEX_SCORE_REDUCE_XDIM];
    //relative position index:
    const uint ndx0 = blockIdx.x * blockDim.x * CUSF_TBSP_INDEX_SCORE_REDUCE_XFCT;
    const uint ndx = ndx0 + threadIdx.x;
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    int /*qrydst, */dbstrdst;


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    if(threadIdx.x < 2) GetDbStrLenDst(dbstrndx, (int*)scvCache);
    if(threadIdx.x < 2) GetQueryLenDst(qryndx, (int*)scvCache + 2);


    __syncthreads();

    //NOTE: no bank conflicts when accessing the same address;
    dbstrlen = ((int*)scvCache)[0]; dbstrdst = ((int*)scvCache)[1];
    qrylen = ((int*)scvCache)[2]; //qrydst = ((int*)scvCache)[3];

    __syncthreads();


    int qrypos, rfnpos;

    if(stepx5)
        GetQryRfnPos_frg5(qrypos, rfnpos,  qrylen, dbstrlen, sfragfct, qryfragfct, rfnfragfct, fragndx);
    else GetQryRfnPos_frg(qrypos, rfnpos,  qrylen, dbstrlen, sfragfct, qryfragfct, rfnfragfct, fragndx);

    int fraglen = GetNAlnPoss_frg(
            qrylen, dbstrlen, qrypos, rfnpos, qryfragfct, rfnfragfct, fragndx);

    //if fragment is out of bounds (tfm and scores not calculated): all threads exit
    if(qrylen < qrypos + fraglen || dbstrlen < rfnpos + fraglen) return;

    //calculate before the lengths get updated
    float d02 = GetD02(qrylen, dbstrlen);

    dbstrlen = myhdmin(dbstrlen, CUSF_TBSP_INDEX_SCORE_POSLIMIT2);

    //all threads in the block exit if thread 0 is out of reference bounds
    if(dbstrlen <= ndx0) return;


    //initialize cache
    scvCache[pad + threadIdx.x] = 0.0f;
    ///qnxCache[pad + threadIdx.x] = 0;

    __syncthreads();


    //(x2 to account for distances and indices; no access error):
    uint mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs * 2;
    int prevqnx = -1;//immediately previous query index

    if(threadIdx.x == 0 && ndx)
        prevqnx = tmpdpdiagbuffers[mloc + dbstrdst + ndx-1 + ndbCposs];

    for(int i = 0; i < CUSF_TBSP_INDEX_SCORE_REDUCE_XFCT; i++) {
        //manually unroll along data blocks by a factor of CUSF_TBSP_INDEX_SCORE_REDUCE_XFCT
        int pos0 = ndx + i * blockDim.x;//position index starting from 0

        int bestqnx;//query index of the position nearest to a reference atom
        float bestdst2;//squared distance to the query atom at bestqnx

        if(pos0 < dbstrlen) {
            //READ best distances and corresponding query positional indices
            //starting with index 0 for tmpdpdiagbuffers
            bestdst2 = tmpdpdiagbuffers[mloc + dbstrdst + pos0];
            bestqnx = tmpdpdiagbuffers[mloc + dbstrdst + pos0 + ndbCposs];
            if(threadIdx.x == 0 && i)
                prevqnx = qnxCache[pad + CUSF_TBSP_INDEX_SCORE_REDUCE_XDIM-1];
        }

        __syncthreads();

        if(pos0 < dbstrlen) {
            if(threadIdx.x == 0) qnxCache[pad-1] = prevqnx;
            qnxCache[pad + threadIdx.x] = bestqnx;
        }

        __syncthreads();

        if(pos0 < dbstrlen) {
            //update score if two consequetive query positions differ (no insertion wrt query):
            if(qnxCache[pad+threadIdx.x] > qnxCache[pad+threadIdx.x-1])
                ///if(bestdst2 <= d82)
                scvCache[pad + threadIdx.x] += GetPairScore(d02, bestdst2);
        }
    }

    //sync:
    __syncthreads();

    //unroll until reaching warpSize 
    for(int xdim = (CUSF_TBSP_INDEX_SCORE_REDUCE_XDIM>>1); xdim >= 32; xdim >>= 1) {
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
        mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvScore) * ndbCstrs;
        atomicAdd(&wrkmemaux[mloc + dbstrndx], scvCache[0]);
    }
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// SaveBestScoreAndConfigLinear: save best scores and associated 
// configuration of query and reference positions along with fragment length;
// stepx5, multiply the step by 5 when calculating query and reference positions;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db structure positions in the chunk;
// maxnsteps, max number of steps (blockIdx.z) that can be performed in one pass;
// qryfragfct, argument 1 for calculating starting query position;
// rfnfragfct, argument 2 for calculating starting reference position;
// fragndx, argument 3 for calculating fragment length;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
__global__
void SaveBestScoreAndConfigLinear(
    const bool stepx5,
    const uint ndbCstrs,
    const uint maxnsteps,
    const int qryfragfct,
    const int rfnfragfct,
    const int fragndx,
    float* __restrict__ wrkmemaux)
{
    //index of the structure processed by thread x (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number
    uint sfragfct = blockIdx.z;//fragment factor
    __shared__ int qrylenshd[1];
    int qrylen, dbstrlen;//query and reference length
    int qrypos, rfnpos;


    if(threadIdx.x == 0) qrylenshd[0] = GetQueryLength(qryndx);
    __syncthreads();
    qrylen = qrylenshd[0];


    //NOTE: a thread can exit if no sync below!
    if(ndbCstrs <= dbstrndx) return;


    dbstrlen = GetDbStrLength(dbstrndx);


    if(stepx5)
        GetQryRfnPos_frg5(qrypos, rfnpos,  qrylen, dbstrlen, sfragfct, qryfragfct, rfnfragfct, fragndx);
    else GetQryRfnPos_frg(qrypos, rfnpos,  qrylen, dbstrlen, sfragfct, qryfragfct, rfnfragfct, fragndx);

    int fraglen = GetNAlnPoss_frg(
            qrylen, dbstrlen, qrypos, rfnpos, qryfragfct, rfnfragfct, fragndx);

    //if fragment is out of bounds (tfm and scores not calculated): a thread exits;
    //NOTE: valid as long as no sync below!
    if(qrylen < qrypos + fraglen || dbstrlen < rfnpos + fraglen) return;


    uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;

    float current = wrkmemaux[mloc + tawmvScore * ndbCstrs + dbstrndx];
    float best = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx];

    if(best < current) {
        wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx] = current;
        wrkmemaux[mloc + tawmvQRYpos * ndbCstrs + dbstrndx] = qrypos;
        wrkmemaux[mloc + tawmvRFNpos * ndbCstrs + dbstrndx] = rfnpos;
        //NOTE: write actual fragment length instead of its index:
        wrkmemaux[mloc + tawmvSubFragNdxCurrent * ndbCstrs + dbstrndx] = fraglen;
    }
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// SaveBestScoreAndConfigAmongBestsLinear: save best scores and respective 
// fragment configuration for transformation matrices by considering all 
// partial best scores calculated over all fragment factors; write the 
// information to the location of fragment factor 0;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps performed for each reference structure;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
__global__
void SaveBestScoreAndConfigAmongBestsLinear(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux)
{
    //index of the structure (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number
    __shared__ float scvCache[CUSF_TBSP_INDEX_SCORE_MAX_YDIM][CUSF_TBSP_INDEX_SCORE_MAX_XDIM+1];
    __shared__ uint ndxCache[CUSF_TBSP_INDEX_SCORE_MAX_YDIM][CUSF_TBSP_INDEX_SCORE_MAX_XDIM+1];

    scvCache[threadIdx.y][threadIdx.x] = 0.0f;
    ndxCache[threadIdx.y][threadIdx.x] = 0;

    //no sync; threads do not access other cells below

    for(uint sfragfct = threadIdx.y; sfragfct < maxnsteps; sfragfct += blockDim.y) {
        float bscore = 0.0f;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(dbstrndx < ndbCstrs)//READ, coalesced for multiple references
            bscore = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx];
        if(scvCache[threadIdx.y][threadIdx.x] < bscore) {
            scvCache[threadIdx.y][threadIdx.x] = bscore;
            ndxCache[threadIdx.y][threadIdx.x] = sfragfct;
        }
        //no sync, every thread works in its own space
    }

    __syncthreads();

    //reduce/unroll for max best score over the fragment factors:
    for(int ydim = (CUSF_TBSP_INDEX_SCORE_MAX_YDIM>>1); ydim >= 1; ydim >>= 1) {
        if(threadIdx.y < ydim &&
            scvCache[threadIdx.y][threadIdx.x] <
            scvCache[threadIdx.y+ydim][threadIdx.x])
        {
            scvCache[threadIdx.y][threadIdx.x] = scvCache[threadIdx.y+ydim][threadIdx.x];
            ndxCache[threadIdx.y][threadIdx.x] = ndxCache[threadIdx.y+ydim][threadIdx.x];
        }

        __syncthreads();
    }

    //scvCache[0][...] now contains maximum
    uint sfragfct = ndxCache[0][threadIdx.x];

    //write scores and associated fragment data
    if(threadIdx.y == 0) {
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(sfragfct != 0 && dbstrndx < ndbCstrs) {
            float bscore = scvCache[0][threadIdx.x];
            //coalesced WRITE for multiple references
            wrkmemaux[mloc0 + tawmvBestScore * ndbCstrs + dbstrndx] = bscore;
            float qrypos = wrkmemaux[mloc + tawmvQRYpos * ndbCstrs + dbstrndx];
            float rfnpos = wrkmemaux[mloc + tawmvRFNpos * ndbCstrs + dbstrndx];
            //NOTE: actual fragment length:
            float fraglen = wrkmemaux[mloc + tawmvSubFragNdxCurrent * ndbCstrs + dbstrndx];
            wrkmemaux[mloc0 + tawmvQRYpos * ndbCstrs + dbstrndx] = qrypos;
            wrkmemaux[mloc0 + tawmvRFNpos * ndbCstrs + dbstrndx] = rfnpos;
            wrkmemaux[mloc0 + tawmvSubFragNdxCurrent * ndbCstrs + dbstrndx] = fraglen;
        }
    }
}

// -------------------------------------------------------------------------
