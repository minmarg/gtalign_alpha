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
#include "libmycu/custgfrg/linear_scoring.cuh"
#include "linear_scoring2.cuh"

// -------------------------------------------------------------------------
// ProduceAlignmentUsingIndex2ReferenceHelper: helper to find coordinates of 
// nearest reference atoms at each query position using index; the result 
// follows from superpositions based on fragments;
// write the coordinates of neighbors for each position processed;
// NOTE: thread block is 1D and processes reference fragment along structure
// positions;
// SECSTRFILT, flag of whether the secondary structure match is required for 
// building an alignment;
// dSMEM, dynamically allocated smem address;
// stacksize, dynamically determined stack size;
// sfragfct, fragment factor;
// qryndx, query serial number;
// dbstrndx, reference serial number;
// qrylen, dbstrlen, query and reference lengths;
// qrydst, dbstrdst, distances to the beginnings of query and reference structures;
// WRTNDX, flag of writing query indices participating in an alignment;
// depth, superposition depth for calculating query and reference positions;
// nqystrs, total number of queries;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db structure positions in the chunk;
// maxnsteps, max number of steps (blockIdx.z) that can be performed in one pass;
// qryfragfct, argument 1 for calculating starting query position;
// rfnfragfct, argument 2 for calculating starting reference position;
// NOTE: memory pointers should be aligned!
// wrkmemtm, memory for transformation matrices;
// tmpdpalnpossbuffer, temporary buffers of found coordinates;
// 
template<int SECSTRFILT>
__device__ __forceinline__
void ProduceAlignmentUsingIndex2ReferenceHelper(
    float* __restrict__ dSMEM,
    const int stacksize,
    const uint sfragfct,
    const uint qryndx,
    const uint dbstrndx,
    int qrylen, int dbstrlen,
    const int qrydst,
    const int dbstrdst,
    const bool WRTNDX,
    const int depth,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int qryfragfct,
    const int rfnfragfct,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux)
{
    //padding to resolve bank conflicts; stack size
    //enum {pad = 1, stacksize = 17};
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
    int fragndx = (sfragfct & 1);

    int qrypos, rfnpos;

    GetQryRfnPos_frg2(
        depth,
        qrypos, rfnpos,  qrylen, dbstrlen, sfragfct, qryfragfct, rfnfragfct, fragndx
    );

    int fraglen = GetNAlnPoss_frg(
            qrylen, dbstrlen, qrypos, rfnpos, qryfragfct, rfnfragfct, fragndx);

    //if fragment is out of bounds (tfm not calculated): all threads in the block exit
    if(qrylen < qrypos + fraglen || dbstrlen < rfnpos + fraglen) return;

    fraglen = myhdmin(dbstrlen, CUSF_TBSP_INDEX_SCORE_POSLIMIT2);
    qrypos = myhdmax(0, qrypos - (fraglen>>1));
    qrylen = myhdmin(qrylen, qrypos + fraglen);
    qrypos = myhdmax(0, qrylen - fraglen);

    //all threads in the block exit if thread 0 is out of query bounds
    if(qrylen <= qrypos + ndx0) return;


    //initialize cache
    ///scvCache[pad + threadIdx.x] = 0.0f;
    ///qnxCache[pad + threadIdx.x] = 0.0f;

    //read transformation matrix for query-reference pair
    if(threadIdx.x < nTTranformMatrix) {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix;
        tfmCache[threadIdx.x] = wrkmemtm[mloc + threadIdx.x];
    }

    //WRITE #matched (aligned) positions (including those masked);
    //NOTE: a block size of more than one warp expected
    if(blockIdx.x == 0 && threadIdx.x == 32) {
        //structure-specific-formatted data; 1st block writes:
        uint wloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvNAlnPoss) * ndbCstrs;
        wrkmemaux[wloc + dbstrndx] = (qrylen - qrypos);
    }

    __syncthreads();


    //(x2 to account for scores and indices; no access error):
    uint dloc = (qryndx * maxnsteps + sfragfct) * ndbCposs * 2;
    int mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs * nTDPAlignedPoss;


    for(int i = 0; i < CUSF_TBSP_INDEX_SCORE_XFCT; i++) {
        //manually unroll along data blocks by a factor of CUSF_TBSP_INDEX_SCORE_XFCT
        char qss;
        float qx, qy, qz;
        int pos0 = ndx + i * blockDim.x;//position index starting from 0

        if(qrypos + pos0 < qrylen)
        {
            int dpos = qrydst + qrypos + pos0;

            qx = GetQueryCoord<pmv2DX>(dpos);
            qy = GetQueryCoord<pmv2DY>(dpos);
            qz = GetQueryCoord<pmv2DZ>(dpos);
            if(SECSTRFILT == 1) qss = GetQuerySS(dpos);

            //WRITE the query coordinates of part of the alignment before transform:
            tmpdpalnpossbuffer[mloc + dbstrdst + pos0 + dpapsQRYx * ndbCposs] = qx;
            tmpdpalnpossbuffer[mloc + dbstrdst + pos0 + dpapsQRYy * ndbCposs] = qy;
            tmpdpalnpossbuffer[mloc + dbstrdst + pos0 + dpapsQRYz * ndbCposs] = qz;

            transform_point(tfmCache, qx, qy, qz);
        }

#if (CUSF_TBSP_INDEX_SCORE_XFCT == 1)
        //same smem buffer used for both
        __syncthreads();
#endif

        if(qrypos + pos0 < qrylen)
        {
            float rx, ry, rz;
            int bestrnx = -1;//reference index of the position nearest to a query atom

            //nearest neighbour using the index tree:
            NNByIndexReference<SECSTRFILT>(
                stacksize,
                bestrnx,//returned
                rx, ry, rz,//returned
                qx, qy, qz, qss,
                dbstrdst, (dbstrlen >> 1)/*root*/, 0/*dimndx*/,
                trtStack + stacksize * nStks_ * threadIdx.x);

            //mask aligned position for no contribution to the alignment:
            //TODO: bestqnx<0 since no difference in values is examined
            if(bestrnx <= 0) {rx = ry = rz = SCNTS_COORD_MASK;}

            //WRITE the query coordinates of part of the alignment:
            tmpdpalnpossbuffer[mloc + dbstrdst + pos0 + dpapsRFNx * ndbCposs] = rx;
            tmpdpalnpossbuffer[mloc + dbstrdst + pos0 + dpapsRFNy * ndbCposs] = ry;
            tmpdpalnpossbuffer[mloc + dbstrdst + pos0 + dpapsRFNz * ndbCposs] = rz;

            //WRITE reference position;
            //TODO: 0<=bestrnx since no difference in values is examined
            if(WRTNDX && 0 < bestrnx)
                tmpdpdiagbuffers[dloc + dbstrdst + pos0 + ndbCposs] = bestrnx;
        }
    }
}

// -------------------------------------------------------------------------
// ProduceAlignmentUsingIndex2QueryHelper: helper to find coordinates of 
// nearest query atoms at each reference position for following processing, 
// using index; the result follows from superpositions based on fragments;
// write the coordinates of neighbors for each position processed;
// NOTE: thread block is 1D and processes reference fragment along structure
// positions;
// SECSTRFILT, flag of whether the secondary structure match is required for 
// building an alignment;
// dSMEM, dynamically allocated smem address;
// stacksize, dynamically determined stack size;
// sfragfct, fragment factor;
// qryndx, query serial number;
// dbstrndx, reference serial number;
// qrylen, dbstrlen, query and reference lengths;
// qrydst, dbstrdst, distances to the beginnings of query and reference structures;
// WRTNDX, flag of writing query indices participating in an alignment;
// depth, superposition depth for calculating query and reference positions;
// nqystrs, total number of queries;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db structure positions in the chunk;
// maxnsteps, max number of steps (blockIdx.z) that can be performed in one pass;
// qryfragfct, argument 1 for calculating starting query position;
// rfnfragfct, argument 2 for calculating starting reference position;
// NOTE: memory pointers should be aligned!
// wrkmemtm, memory for transformation matrices;
// tmpdpalnpossbuffer, temporary buffers of found coordinates;
// 
template<int SECSTRFILT>
__device__ __forceinline__
void ProduceAlignmentUsingIndex2QueryHelper(
    float* __restrict__ dSMEM,
    const int stacksize,
    const uint sfragfct,
    const uint qryndx,
    const uint dbstrndx,
    int qrylen, int dbstrlen,
    const int qrydst,
    const int dbstrdst,
    const bool WRTNDX,
    const int depth,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int qryfragfct,
    const int rfnfragfct,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux)
{
    //padding to resolve bank conflicts; stack size
    //enum {pad = 1, stacksize = 17};
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
    int fragndx = (sfragfct & 1);

    int qrypos, rfnpos;

    GetQryRfnPos_frg2(
        depth,
        qrypos, rfnpos,  qrylen, dbstrlen, sfragfct, qryfragfct, rfnfragfct, fragndx
    );

    int fraglen = GetNAlnPoss_frg(
            qrylen, dbstrlen, qrypos, rfnpos, qryfragfct, rfnfragfct, fragndx);

    //if fragment is out of bounds (tfm not calculated): all threads in the block exit
    if(qrylen < qrypos + fraglen || dbstrlen < rfnpos + fraglen) return;

    fraglen = myhdmin(qrylen, CUSF_TBSP_INDEX_SCORE_POSLIMIT2);
    //qrypos = myhdmax(0, qrypos - (fraglen>>1));
    rfnpos = myhdmax(0, rfnpos - (fraglen>>1));
    dbstrlen = myhdmin(dbstrlen, rfnpos + fraglen);
    rfnpos = myhdmax(0, dbstrlen - fraglen);

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

    //WRITE #matched (aligned) positions (including those masked);
    //NOTE: a block size of more than one warp expected
    if(blockIdx.x == 0 && threadIdx.x == 32) {
        //structure-specific-formatted data; 1st block writes:
        uint wloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvNAlnPoss) * ndbCstrs;
        wrkmemaux[wloc + dbstrndx] = (dbstrlen - rfnpos);
    }

    __syncthreads();


    //(x2 to account for scores and indices; no access error):
    uint dloc = (qryndx * maxnsteps + sfragfct) * ndbCposs * 2;
    int mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs * nTDPAlignedPoss;


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

            //WRITE the reference coordinates of part of the alignment before transform:
            tmpdpalnpossbuffer[mloc + dbstrdst + pos0 + dpapsRFNx * ndbCposs] = rx;
            tmpdpalnpossbuffer[mloc + dbstrdst + pos0 + dpapsRFNy * ndbCposs] = ry;
            tmpdpalnpossbuffer[mloc + dbstrdst + pos0 + dpapsRFNz * ndbCposs] = rz;

            transform_point(tfmCache, rx, ry, rz);
        }

#if (CUSF_TBSP_INDEX_SCORE_XFCT == 1)
        //same smem buffer used for both
        __syncthreads();
#endif

        if(rfnpos + pos0 < dbstrlen)
        {
            float qx, qy, qz;
            int bestqnx = -1;//query index of the position nearest to a reference atom

            //nearest neighbour using the index tree:
            NNByIndex<SECSTRFILT>(
                stacksize,
                bestqnx,//returned
                qx, qy, qz,//returned
                rx, ry, rz, rss,
                qrydst, (qrylen >> 1)/*root*/, 0/*dimndx*/,
                trtStack + stacksize * nStks_ * threadIdx.x);

            //mask aligned position for no contribution to the alignment:
            //TODO: bestqnx<0 since no difference in values is examined
            if(bestqnx <= 0) {qx = qy = qz = SCNTS_COORD_MASK;}

            //WRITE the query coordinates of part of the alignment:
            tmpdpalnpossbuffer[mloc + dbstrdst + pos0 + dpapsQRYx * ndbCposs] = qx;
            tmpdpalnpossbuffer[mloc + dbstrdst + pos0 + dpapsQRYy * ndbCposs] = qy;
            tmpdpalnpossbuffer[mloc + dbstrdst + pos0 + dpapsQRYz * ndbCposs] = qz;

            //WRITE query position;
            //TODO: 0<=bestqnx since no difference in values is examined
            if(WRTNDX && 0 < bestqnx)
                tmpdpdiagbuffers[dloc + dbstrdst + pos0 + ndbCposs] = bestqnx;
        }
    }
}

// -------------------------------------------------------------------------
// ProduceAlignmentUsingIndex2: find coordinates of nearest query 
// atoms at each reference position for following processing, using 
// index; the result follows from superpositions based on fragments;
// write the coordinates of neighbors for each position processed;
// NOTE: thread block is 1D and processes reference fragment along structure
// positions;
// SECSTRFILT, flag of whether the secondary structure match is required for 
// building an alignment;
// WRTNDX, flag of writing query indices participating in an alignment;
// stacksize, dynamically determined stack size;
// depth, superposition depth for calculating query and reference positions;
// nqystrs, total number of queries;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db structure positions in the chunk;
// maxnsteps, max number of steps (blockIdx.z) that can be performed in one pass;
// qryfragfct, argument 1 for calculating starting query position;
// rfnfragfct, argument 2 for calculating starting reference position;
// NOTE: memory pointers should be aligned!
// wrkmemtm, memory for transformation matrices;
// tmpdpalnpossbuffer, temporary buffers of found coordinates;
// 
template<int SECSTRFILT>
__global__
void ProduceAlignmentUsingIndex2(
    const int stacksize,
    const bool WRTNDX,
    const int depth,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int qryfragfct,
    const int rfnfragfct,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux)
{
    // blockIdx.x is the block index of positions for query-reference pair;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number TIMES fragment factor;
    uint sfragfct = blockIdx.z / nqystrs;//fragment factor
    uint qryndx = blockIdx.z - sfragfct * nqystrs;//query serial number
    uint dbstrndx = blockIdx.y;//reference serial number

    //use dynamically allocated SM in case when allocation exceeds 48KB:
    extern __shared__ float dSMEM[];
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    int qrydst, dbstrdst;

    //READ convergence flag for stopping;
    //NOTE: a block size of more than one warp expected
    if(threadIdx.x == 32) {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        dSMEM[7] = wrkmemaux[mloc + dbstrndx];
    }

    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    if(threadIdx.x < 2) GetDbStrLenDst(dbstrndx, (int*)dSMEM);
    if(threadIdx.x < 2) GetQueryLenDst(qryndx, (int*)dSMEM + 2);

    __syncthreads();

    if((/*(int)*/dSMEM[7]) /*& (CONVERGED_SCOREDP_bitval)*/)
        //(any flag implies exit for all threads)
        return;

    //NOTE: no bank conflicts when accessing the same address;
    dbstrlen = ((int*)dSMEM)[0]; dbstrdst = ((int*)dSMEM)[1];
    qrylen = ((int*)dSMEM)[2]; qrydst = ((int*)dSMEM)[3];

    __syncthreads();

    ProduceAlignmentUsingIndex2QueryHelper<SECSTRFILT>(
        dSMEM, stacksize,
        sfragfct, qryndx, dbstrndx,
        qrylen, dbstrlen, qrydst, dbstrdst,
        WRTNDX, (depth), nqystrs, ndbCstrs, ndbCposs,
        maxnsteps, qryfragfct, rfnfragfct,
        wrkmemtm, tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux);
}

// Instantiations
// 
#define INSTANTIATE_ProduceAlignmentUsingIndex2(tpSECSTRFILT) \
    template __global__ void ProduceAlignmentUsingIndex2<tpSECSTRFILT>( \
        const int stacksize, const bool WRTNDX, const int depth, const uint nqystrs, \
        const uint ndbCstrs, const uint ndbCposs, const uint maxnsteps, \
        const int qryfragfct, const int rfnfragfct, \
        const float* __restrict__ wrkmemtm, \
        float* __restrict__ tmpdpalnpossbuffer, \
        float* __restrict__ tmpdpdiagbuffers, \
        float* __restrict__ wrkmemaux);

INSTANTIATE_ProduceAlignmentUsingIndex2(0);
INSTANTIATE_ProduceAlignmentUsingIndex2(1);

// -------------------------------------------------------------------------
// ProduceAlignmentUsingDynamicIndex2: same as ProduceAlignmentUsingIndex2 
// except that the alignments are produced using dynamically selected index
// 
template<int SECSTRFILT>
__global__
void ProduceAlignmentUsingDynamicIndex2(
    const int stacksize,
    const bool WRTNDX,
    const int depth,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int qryfragfct,
    const int rfnfragfct,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux)
{
    // blockIdx.x is the block index of positions for query-reference pair;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number TIMES fragment factor;
    uint sfragfct = blockIdx.z / nqystrs;//fragment factor
    uint qryndx = blockIdx.z - sfragfct * nqystrs;//query serial number
    uint dbstrndx = blockIdx.y;//reference serial number

    //use dynamically allocated SM in case when allocation exceeds 48KB:
    extern __shared__ float dSMEM[];
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    int qrydst, dbstrdst;

    //READ convergence flag for stopping;
    //NOTE: a block size of more than one warp expected
    if(threadIdx.x == 32) {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        dSMEM[7] = wrkmemaux[mloc + dbstrndx];
    }

    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    if(threadIdx.x < 2) GetDbStrLenDst(dbstrndx, (int*)dSMEM);
    if(threadIdx.x < 2) GetQueryLenDst(qryndx, (int*)dSMEM + 2);

    __syncthreads();

    if((/*(int)*/dSMEM[7]) /*& (CONVERGED_SCOREDP_bitval)*/)
        //(any flag implies exit for all threads)
        return;

    //NOTE: no bank conflicts when accessing the same address;
    dbstrlen = ((int*)dSMEM)[0]; dbstrdst = ((int*)dSMEM)[1];
    qrylen = ((int*)dSMEM)[2]; qrydst = ((int*)dSMEM)[3];

    __syncthreads();

    if(qrylen < dbstrlen)
        ProduceAlignmentUsingIndex2ReferenceHelper<SECSTRFILT>(
            dSMEM, stacksize,
            sfragfct, qryndx, dbstrndx,
            qrylen, dbstrlen, qrydst, dbstrdst,
            WRTNDX, (depth), nqystrs, ndbCstrs, ndbCposs,
            maxnsteps, qryfragfct, rfnfragfct,
            wrkmemtm, tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux);
    else
        ProduceAlignmentUsingIndex2QueryHelper<SECSTRFILT>(
            dSMEM, stacksize,
            sfragfct, qryndx, dbstrndx,
            qrylen, dbstrlen, qrydst, dbstrdst,
            WRTNDX, (depth), nqystrs, ndbCstrs, ndbCposs,
            maxnsteps, qryfragfct, rfnfragfct,
            wrkmemtm, tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux);
}

// Instantiations
// 
#define INSTANTIATE_ProduceAlignmentUsingDynamicIndex2(tpSECSTRFILT) \
    template __global__ void ProduceAlignmentUsingDynamicIndex2<tpSECSTRFILT>( \
        const int stacksize, const bool WRTNDX, const int depth, const uint nqystrs, \
        const uint ndbCstrs, const uint ndbCposs, const uint maxnsteps, \
        const int qryfragfct, const int rfnfragfct, \
        const float* __restrict__ wrkmemtm, \
        float* __restrict__ tmpdpalnpossbuffer, \
        float* __restrict__ tmpdpdiagbuffers, \
        float* __restrict__ wrkmemaux);

INSTANTIATE_ProduceAlignmentUsingDynamicIndex2(0);
INSTANTIATE_ProduceAlignmentUsingDynamicIndex2(1);

// -------------------------------------------------------------------------





// -------------------------------------------------------------------------
// PositionalCoordsFromIndexLinear2: find coordinates of nearest query 
// atoms at each reference position for following processing, using 
// index; the result follows from superpositions based on fragments;
// NOTE: thread block is 1D and processes reference fragment along structure
// positions;
// stacksize, dynamically determined stack size;
// depth, superposition depth for calculating query and reference positions;
// nqystrs, total number of queries;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db structure positions in the chunk;
// maxnsteps, max number of steps (blockIdx.z) that can be performed in one pass;
// qryfragfct, argument 1 for calculating starting query position;
// rfnfragfct, argument 2 for calculating starting reference position;
// fragndx, argument 3 for calculating fragment length;
// NOTE: memory pointers should be aligned!
// wrkmemtm, memory for transformation matrices;
// tmpdpalnpossbuffer, temporary buffers of found coordinates;
// 
template<int SECSTRFILT>
__global__
void PositionalCoordsFromIndexLinear2(
    const int stacksize,
    const int depth,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int qryfragfct,
    const int rfnfragfct,
    int fragndx,
    const float* __restrict__ wrkmemtm,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpalnpossbuffer)
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
    fragndx = (sfragfct & 1);


    //READ convergence flag for stopping;
    //NOTE: a block size of more than one warp expected
    if(threadIdx.x == 32) {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        dSMEM[7] = wrkmemaux[mloc + dbstrndx];
    }

    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse tfmCache
    if(threadIdx.x < 2) GetDbStrLenDst(dbstrndx, (int*)tfmCache);
    if(threadIdx.x < 2) GetQueryLenDst(qryndx, (int*)tfmCache + 2);


    __syncthreads();

    if((/*(int)*/dSMEM[7]) /*& (CONVERGED_SCOREDP_bitval)*/)
        //(any flag implies exit for all threads)
        return;

    //NOTE: no bank conflicts when accessing the same address;
    dbstrlen = ((int*)tfmCache)[0]; dbstrdst = ((int*)tfmCache)[1];
    qrylen = ((int*)tfmCache)[2]; qrydst = ((int*)tfmCache)[3];

    __syncthreads();


    int qrypos, rfnpos;

    GetQryRfnPos_frg2(
        depth,
        qrypos, rfnpos,  qrylen, dbstrlen, sfragfct, qryfragfct, rfnfragfct, fragndx
    );

    int fraglen = GetNAlnPoss_frg(
            qrylen, dbstrlen, qrypos, rfnpos, qryfragfct, rfnfragfct, fragndx);

    //if fragment is out of bounds (tfm not calculated): all threads in the block exit
    if(qrylen < qrypos + fraglen || dbstrlen < rfnpos + fraglen) return;

    fraglen = myhdmin(qrylen, CUSF_TBSP_INDEX_SCORE_POSLIMIT2);
    //qrypos = myhdmax(0, qrypos - (fraglen>>1));
    rfnpos = myhdmax(0, rfnpos - (fraglen>>1));
    dbstrlen = myhdmin(dbstrlen, rfnpos + fraglen);
    rfnpos = myhdmax(0, dbstrlen - fraglen);

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

            //WRITE indices of best matching query atoms starting with index 0 
            int mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs * nTDPAlignedPoss;
            tmpdpalnpossbuffer[mloc + dbstrdst + pos0] = bestqnx;
        }
    }
}

// Instantiations
// 
#define INSTANTIATE_PositionalCoordsFromIndexLinear2(tpSECSTRFILT) \
    template __global__ void PositionalCoordsFromIndexLinear2<tpSECSTRFILT>( \
        const int stacksize, const int depth, const uint nqystrs, \
        const uint ndbCstrs, const uint ndbCposs, const uint maxnsteps, \
        const int qryfragfct, const int rfnfragfct, int fragndx, \
        const float* __restrict__ wrkmemtm, \
        const float* __restrict__ wrkmemaux, \
        float* __restrict__ tmpdpalnpossbuffer);

INSTANTIATE_PositionalCoordsFromIndexLinear2(0);
INSTANTIATE_PositionalCoordsFromIndexLinear2(1);

// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// MakeAlignmentLinear2: make alignment from indices identified previously 
// by using the index tree; 
// NOTE: thread block is 2D and processes reference fragment along structure
// positions;
// complete, flag of complete alignment;
// depth, superposition depth for calculating query and reference positions;
// nqystrs, total number of queries;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db structure positions in the chunk;
// maxnsteps, max number of steps (blockIdx.z) that can be performed in one pass;
// qryfragfct, argument 1 for calculating starting query position;
// rfnfragfct, argument 2 for calculating starting reference position;
// fragndx, argument 3 for calculating fragment length;
// NOTE: memory pointers should be aligned!
// tmpdpalnpossbuffer, temporary buffers of aligned coordinates;
// wrkmemaux, auxiliary working memory;
// 
__global__
void MakeAlignmentLinear2(
    const bool complete,
    const int depth,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int qryfragfct,
    const int rfnfragfct,
    int fragndx,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux)
{
    // blockIdx.x is the block index of positions for query-reference pair;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number TIMES fragment factor;
    uint sfragfct = blockIdx.z / nqystrs;//fragment factor
    uint qryndx = blockIdx.z - sfragfct * nqystrs;//query serial number
    uint dbstrndx = blockIdx.y;//reference serial number

    enum {pad = 1};//padding to resolve bank conflicts
    //cache for query positional indices (to keep track of insertions wrt query later): 
    __shared__ int qnxCache[pad + CUSF2_TBSP_INDEX_ALIGNMENT_XDIM];
    //relative position index:
    const uint ndx0 = blockIdx.x * blockDim.x * CUSF2_TBSP_INDEX_ALIGNMENT_XFCT;
    const uint ndx = ndx0 + threadIdx.x;
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    int qrydst, dbstrdst;
    fragndx = (sfragfct & 1);


    //READ convergence flag for stopping;
    if(threadIdx.y == 2 && threadIdx.x == 0) {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        qnxCache[7] = wrkmemaux[mloc + dbstrndx];
    }

    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    if(threadIdx.y == 0 && threadIdx.x < 2) GetDbStrLenDst(dbstrndx, (int*)qnxCache);
    if(threadIdx.y == 1 && threadIdx.x < 2) GetQueryLenDst(qryndx, (int*)qnxCache + 2);


    __syncthreads();

    if((/*(int)*/qnxCache[7]) /*& (CONVERGED_SCOREDP_bitval)*/)
        //(any flag implies exit for all threads)
        return;

    //NOTE: no bank conflicts when accessing the same address;
    dbstrlen = ((int*)qnxCache)[0]; dbstrdst = ((int*)qnxCache)[1];
    qrylen = ((int*)qnxCache)[2]; qrydst = ((int*)qnxCache)[3];

    __syncthreads();


    int qrypos, rfnpos;

    GetQryRfnPos_frg2(
        depth,
        qrypos, rfnpos,  qrylen, dbstrlen, sfragfct, qryfragfct, rfnfragfct, fragndx
    );

    int fraglen = GetNAlnPoss_frg(
            qrylen, dbstrlen, qrypos, rfnpos, qryfragfct, rfnfragfct, fragndx);

    //if fragment is out of bounds (tfm and scores not calculated): all threads exit
    if(qrylen < qrypos + fraglen || dbstrlen < rfnpos + fraglen) return;

    fraglen = myhdmin(qrylen, CUSF_TBSP_INDEX_SCORE_POSLIMIT2);
    //qrypos = myhdmax(0, qrypos - (fraglen>>1));
    rfnpos = myhdmax(0, rfnpos - (fraglen>>1));
    dbstrlen = myhdmin(dbstrlen, rfnpos + fraglen);
    rfnpos = myhdmax(0, dbstrlen - fraglen);

    //all threads in the block exit if thread 0 is out of reference bounds
    if(dbstrlen <= rfnpos + ndx0) return;


    //initialize cache
    ///if(threadIdx.y == 0) qnxCache[pad + threadIdx.x] = 0;
    ///__syncthreads();


    //(x2 to account for scores and indices; no access error):
    uint dloc = (qryndx * maxnsteps + sfragfct) * ndbCposs * 2;
    uint mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs * nTDPAlignedPoss;
    int prevqnx = -1;//immediately previous query index

    if(threadIdx.x == 0 && ndx && threadIdx.y == 0)
        prevqnx = tmpdpalnpossbuffer[mloc + dbstrdst + ndx-1];


    //WRITE #matched (aligned) positions (including those masked)
    if(blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        //structure-specific-formatted data; 1st block writes:
        uint wloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvNAlnPoss) * ndbCstrs;
        wrkmemaux[wloc + dbstrndx] = (dbstrlen - rfnpos);
    }


#if (CUSF2_TBSP_INDEX_ALIGNMENT_YDIM != 6)
#error "INVALID EXECUTION CONFIGURATION: CUSF2_TBSP_INDEX_ALIGNMENT_YDIM should be 6."
#endif


    for(int i = 0; i < CUSF2_TBSP_INDEX_ALIGNMENT_XFCT; i++) {
        //manually unroll along data blocks by a factor of CUSF2_TBSP_INDEX_ALIGNMENT_XFCT
        int pos0 = ndx + i * blockDim.x;//position index starting from 0
        int pos00 = ndx0 + i * blockDim.x;//thread-0 position index

        int bestqnx;//query index of the position nearest to a reference atom

        if(dbstrlen <= rfnpos + pos00) break;

        if(rfnpos + pos0 < dbstrlen && threadIdx.y == 0) {
            //READ best-matching query position indices starting with index 0
            bestqnx = tmpdpalnpossbuffer[mloc + dbstrdst + pos0];
            if(threadIdx.x == 0 && i)
                prevqnx = qnxCache[pad + CUSF2_TBSP_INDEX_ALIGNMENT_XDIM-1];
        }

        __syncthreads();

        if(rfnpos + pos0 < dbstrlen && threadIdx.y == 0) {
            if(threadIdx.x == 0) qnxCache[pad-1] = prevqnx;
            qnxCache[pad + threadIdx.x] = bestqnx;
        }

        __syncthreads();

        if(rfnpos + pos0 < dbstrlen) {
            //initially, a pair is masked:
            float crd = SCNTS_COORD_MASK;
            int qndx = qnxCache[pad+threadIdx.x];//query index
            //aligned pair if two consequetive query positions differ (no insertion wrt query):
            if(qndx > 0 && (complete || (qndx > qnxCache[pad+threadIdx.x-1])))
            {
                int dpos = dbstrdst + rfnpos + pos0;
                int qpos = qrydst + qndx;
                if(threadIdx.y == dpapsRFNx) crd = GetDbStrCoord<pmv2DX>(dpos);
                if(threadIdx.y == dpapsRFNy) crd = GetDbStrCoord<pmv2DY>(dpos);
                if(threadIdx.y == dpapsRFNz) crd = GetDbStrCoord<pmv2DZ>(dpos);
                if(threadIdx.y == dpapsQRYx) crd = GetQueryCoord<pmv2DX>(qpos);
                if(threadIdx.y == dpapsQRYy) crd = GetQueryCoord<pmv2DY>(qpos);
                if(threadIdx.y == dpapsQRYz) crd = GetQueryCoord<pmv2DZ>(qpos);
            }

            tmpdpalnpossbuffer[mloc + dbstrdst + pos0 + threadIdx.y * ndbCposs] = crd;
            //write query position:
            if(threadIdx.y == 0 && crd < SCNTS_COORD_MASK_cmp)
                tmpdpdiagbuffers[dloc + dbstrdst + pos0 + ndbCposs] = qndx;
        }
    }
}

// -------------------------------------------------------------------------
