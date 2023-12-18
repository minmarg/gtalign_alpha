/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __linear_scoring2_complete_cuh__
#define __linear_scoring2_complete_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fields.cuh"
#include "libmycu/custages/fragment.cuh"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/covariance_swift_scan.cuh"
#include "libmycu/custages/scoring.cuh"
#include "libmycu/custgfrg/linear_scoring.cuh"

// =========================================================================
// ScoreFragmentBasedSuperpositionsLinearly2: perform and score 
// fragment-based superposition using index in linear time;
//
__global__ 
void ScoreFragmentBasedSuperpositionsLinearly2(
    const int stacksize,
    const int depth,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int qryfragfct,
    const int rfnfragfct,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux
);

// =========================================================================
// CopyCCMDataToTFM_REVERSETfm_Complete: copy cross-covariance data to an 
// additional buffer for REVERSE transformation;
//
__device__ __forceinline__
void CopyCCMDataToTFM_REVERSETfm_Complete(
    const float* __restrict__ ccmCache,
    float* __restrict__ tfmCache)
{
    int srcndx = threadIdx.x;
    switch(threadIdx.x) {
//         case twmvCCM_0_0: srcndx = twmvCCM_0_0; break;
        case twmvCCM_0_1: srcndx = twmvCCM_1_0; break;
        case twmvCCM_0_2: srcndx = twmvCCM_2_0; break;
        case twmvCCM_1_0: srcndx = twmvCCM_0_1; break;
//         case twmvCCM_1_1: srcndx = twmvCCM_1_1; break;
        case twmvCCM_1_2: srcndx = twmvCCM_2_1; break;
        case twmvCCM_2_0: srcndx = twmvCCM_0_2; break;
        case twmvCCM_2_1: srcndx = twmvCCM_1_2; break;
//         case twmvCCM_2_2: srcndx = twmvCCM_2_2; break;
        case twmvCVr_0: srcndx = twmvCVq_0; break; 
        case twmvCVr_1: srcndx = twmvCVq_1; break; 
        case twmvCVr_2: srcndx = twmvCVq_2; break; 
        case twmvCVq_0: srcndx = twmvCVr_0; break; 
        case twmvCVq_1: srcndx = twmvCVr_1; break; 
        case twmvCVq_2: srcndx = twmvCVr_2; break; 
    }
    if(threadIdx.x < twmvEndOfCCDataExt)
        tfmCache[threadIdx.x] = ccmCache[srcndx];
}

// -------------------------------------------------------------------------
// ProduceAlignmentUsingIndex2_Complete: find coordinates of nearest query 
// atoms at each reference position for following processing, using 
// index; the result follows from superpositions based on fragments;
// NOTE: thread block is 1D and processes reference fragment along structure
// positions;
// SECSTRFILT, flag of whether the secondary structure match is required for 
// building an alignment;
// WRTNDX, flag of writing query indices participating in an alignment;
// maxalnlen, maximum length of an alignment that can be produced;
// stacksize, dynamically determined stack size;
// qryndx, query serial number;
// ndbCposs, total number of db structure positions in the chunk;
// maxnsteps, max number of steps that can be performed in one pass;
// sfragfct, current fragment factor;
// qrydst, distances in positions to the beginnings of the query structures;
// dbstrdst, distances in positions to the beginnings of the reference structures;
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// NOTE: memory pointers should be aligned!
// tfmCache, cached transformation matrix;
// trtStack, cache for stack;
// tmpdpalnpossbuffer, coordinates of matched positions (alignment) obtained from index;
// tmpdpdiagbuffers, temporary diagonal buffers used here for saving query indices;
// 
template<int SECSTRFILT, bool WRTNDX = false>
__device__ __forceinline__
void ProduceAlignmentUsingIndex2_Complete(
    int& maxalnlen,
    const int stacksize,
    const uint qryndx,
    const uint ndbCposs,
    const uint maxnsteps,
    const uint sfragfct,
    const uint qrydst,
    const uint dbstrdst,
    int qrylen, int dbstrlen,
    int /*qrypos*/, int rfnpos,
    const float* __restrict__ tfmCache,
    float* __restrict__ trtStack,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers)
{
    int fraglen = myhdmin(qrylen, CUSF_TBSP_INDEX_SCORE_POSLIMIT2);
    //qrypos = myhdmax(0, qrypos - (fraglen>>1));
    rfnpos = myhdmax(0, rfnpos - (fraglen>>1));
    dbstrlen = myhdmin(dbstrlen, rfnpos + fraglen);
    rfnpos = myhdmax(0, dbstrlen - fraglen);

// //     //qrypos = myhdmax(0, qrypos - CUSF_TBSP_INDEX_SCORE_POSLIMIT);
// //     rfnpos = myhdmax(0, rfnpos - CUSF_TBSP_INDEX_SCORE_POSLIMIT);
// //     dbstrlen = myhdmin(dbstrlen, rfnpos + CUSF_TBSP_INDEX_SCORE_POSLIMIT2);
// //     rfnpos = myhdmax(0, dbstrlen - CUSF_TBSP_INDEX_SCORE_POSLIMIT2);

    //#matched (aligned) positions (including those masked)
    maxalnlen = (dbstrlen - rfnpos);


    //(x2 to account for scores and indices; no access error):
    uint dloc = (qryndx * maxnsteps + sfragfct) * ndbCposs * 2;
    uint mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs * nTDPAlignedPoss;


    //manually unroll along data blocks:
    //pos0, position index starting from 0
    for(int pos0 = threadIdx.x; rfnpos + pos0 < dbstrlen + threadIdx.x;
        pos0 += blockDim.x)
    {
        char rss;
        float rx, ry, rz;
        float qx, qy, qz;
        int bestqnx = -1;//query index of the position nearest to a reference atom

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
// CalcCCMatrices64_SWFTscan_Complete: calculate cross-covariance matrix 
// between the query and reference structures given their alignment;
// Complete version for alignments obtained by the application of linearity;
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// qryndx, query serial number;
// ndbCposs, total number of reference positions in the chunk;
// maxnsteps, max number of steps that can be performed in one pass;
// sfragfct, current fragment factor (<maxnsteps);
// dbstrdst, distances in positions to the beginnings of the reference structures;
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// NOTE: memory pointers should be aligned!
// tmpdpalnpossbuffer, coordinates of matched positions obtained from index;
// ccmCache, cache for the cross-covariance matrix and related data;
// 
template<int SMIDIM, int NEFFDS>
__device__ __forceinline__
void CalcCCMatrices64_SWFTscan_Complete(
    const uint qryndx,
    const uint ndbCposs,
    const uint maxnsteps,
    const uint sfragfct,
    const uint dbstrdst,
    const int qrylen, const int dbstrlen,
    const int qrypos, const int rfnpos,
    const float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__  ccmCache)
{
    InitCCMCacheExtended<SMIDIM,0,NEFFDS>(ccmCache);

    //no sync as long as each thread works in its own memory space

    const int dblen = ndbCposs;
    //offset to the beginning of the data along the y axis wrt query qryndx:
    const int yofff = (qryndx * maxnsteps + sfragfct) * dblen * nTDPAlignedPoss;

    //manually unroll along data blocks:
    //pos0, position index starting from 0
    for(int pos0 = threadIdx.x; qrypos + pos0 < qrylen && rfnpos + pos0 < dbstrlen;
        pos0 += blockDim.x)
    {
        //starting position in tmpdpalnpossbuffer for a pair:
        //NOTE: qrypos == rfnpos as well as qrylen == dbstrlen here
        int dbpos = yofff + dbstrdst + (rfnpos + pos0);
        UpdateCCMOneAlnPos_SWFTRefined<SMIDIM>(//no sync;
            dbpos, dblen,
            tmpdpalnpossbuffer,
            ccmCache
        );
    }

    //sync now:
    __syncthreads();

    //unroll until reaching warpSize 
    //for(int xdim = (blockDim.x>>1); xdim >= 32; xdim >>= 1) {
    for(int xdim = (CUSF_TBSP_COMPLETE_INDEX_SCORE_XDIM>>1); xdim >= 32; xdim >>= 1) {
        if(threadIdx.x < xdim) {
//             #pragma unroll
            for(int i = 0; i < NEFFDS; i++)
                ccmCache[threadIdx.x * SMIDIM + i] +=
                    ccmCache[(threadIdx.x + xdim) * SMIDIM + i];
        }
        __syncthreads();
    }

    //unroll warp
    if(threadIdx.x < 32) {
        for(int i = 0; i < NEFFDS; i++) {
            float sum = ccmCache[threadIdx.x * SMIDIM + i];
            sum = mywarpreducesum(sum);
            //write to the first SMEM data slot
            if(threadIdx.x == 0) ccmCache[i] = sum;
        }
    }

    //make all threads in the block see the changes
    __syncthreads();
}

// -------------------------------------------------------------------------
// CalcScoresUnrl_SWFTscanProgressive_Complete: calculate/reduce scores for 
// obtained superpositions progressively to ensure order-dependent alignment;
// Complete version for alignments obtained from the application of a linear
// algorithm; 
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// NOTE: WARP version: x-dimension of the thread block should be 32 for consistent results;
// SZQNXCH, template parameter, cache size for qnxCache and maxCache; 
// preferably it should be large since the cache is used as a hash table too;
// qryndx, query serial number;
// ndbCposs, total number of reference positions in the chunk;
// maxnsteps, max number of steps that can be performed in one pass;
// sfragfct, current fragment factor (<maxnsteps);
// dbstrdst, distances in positions to the beginnings of the reference structures;
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// d02, distance threshold;
// NOTE: memory pointers should be aligned!
// tfmCache, cached transformation matrix;
// tmpdpalnpossbuffer, coordinates of matched positions obtained from index;
// tmpdpdiagbuffers, diagonal buffers with query indices written within;
// qnxCache, cache for query positions;
// maxCache, cache for maximum scores over all query positions;
// 
template<int SZQNXCH>
__device__ __forceinline__
float CalcScoresUnrl_SWFTscanProgressive_Complete(
    const uint qryndx,
    const uint ndbCposs,
    const uint maxnsteps,
    const uint sfragfct,
    const uint dbstrdst,
    const int /*qrylen*/, const int dbstrlen,
    const int /*qrypos*/, const int rfnpos,
    const float d02,
    const float* __restrict__ tfmCache,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ qnxCache,
    float* __restrict__ maxCache)
{
    //initialize cache:
    for(int i = threadIdx.x; i < SZQNXCH; i += blockDim.x) {
        qnxCache[i] = -1.0f;
        maxCache[i] = 0.0f;
    }

    __syncthreads();

    const int dblen = ndbCposs;
    //offset to the beginning of the data along the y axis wrt query qryndx: 
    const int yofff = (qryndx * maxnsteps + sfragfct) * dblen * nTDPAlignedPoss;
    //(x2 to account for scores and query positions; no access error):
    uint dloc = (qryndx * maxnsteps + sfragfct) * dblen * 2;

    //manually unroll along data blocks:
    //pos0, position index starting from 0
    for(int pos0 = threadIdx.x;
        //qrypos + pos0 < qrylen + threadIdx.x && //same as below
        rfnpos + pos0 < dbstrlen + threadIdx.x;
        pos0 += blockDim.x)
    {
        //starting position in tmpdpalnpossbuffer for a pair:
        //NOTE: qrypos == rfnpos as well as qrylen == dbstrlen here
        int dppos = yofff + dbstrdst + (rfnpos + pos0);
        float sco_r = 0.0f, qnx_r = -1.0f;//score and query position

        if(rfnpos + pos0 < dbstrlen) {
            CacheAlnPosScore_SWFTProgressive_Reg(
                d02, dppos, dblen,
                dloc + dbstrdst + pos0,//tmpdpdiagbuffers address
                tmpdpalnpossbuffer,//coordinates
                tfmCache,//tfm.
                &sco_r, &qnx_r,//score and query position
                tmpdpdiagbuffers//gmem to read query poss. from
            );
        }

        //NOTE: process WARP-width reference positions to pregressively find max score
        for(int p = 0; 
            p < CUSF_TBSP_COMPLETE_INDEX_SCORE_XDIM && 
            //qrypos + pos0 + p < qrylen + threadIdx.x && //same as below
            rfnpos + pos0 + p < dbstrlen + threadIdx.x; p++)
        {
            float sco = __shfl_sync(0xffffffff, sco_r, p/*srcLane*/);
            float qnx = __shfl_sync(0xffffffff, qnx_r, p/*srcLane*/);

            if(sco <= 0.0f || qnx < 0.0f) continue;

            //find max score up to position qnx:
            float max =
                FindMax_SWFTProgressive_Warp
                <CUSF_TBSP_COMPLETE_INDEX_SCORE_XDIM,SZQNXCH,false/*uncnd*/>(
                    qnx, qnxCache, maxCache);

            //save max score to cache:
            if(threadIdx.x == 0) {
                //extremely simple hash function for the cache index:
                int c = (int)(qnx) & (SZQNXCH-1);
                float stqnx = qnxCache[c];//stored query position
                float stsco = maxCache[c];//stored score
                float newsco = max + sco;//new score
                //NOTE: rfnpos+pos0+p represents the position relative to thread 0:
                bool half2nd = (rfnpos + pos0 + p > (dbstrlen>>1));
                //heuristics: under hash collision, update position and 
                //score wrt to which reference half is under process:
                if(stqnx < 0.0f ||
                  (stqnx == qnx && stsco < newsco) ||
                  ((half2nd && stqnx < qnx) || (!half2nd && qnx < stqnx))) {
                    qnxCache[c] = qnx;
                    maxCache[c] = newsco;
                }
            }

            __syncwarp();
        }
    }

    //find max score over all query positions:
    float max = 
        FindMax_SWFTProgressive_Warp
        <CUSF_TBSP_COMPLETE_INDEX_SCORE_XDIM,SZQNXCH,true/*uncnd*/>(
            0.0f, qnxCache, maxCache);

    max = __shfl_sync(0xffffffff, max, 0/*srcLane*/);
    return max;
}

// -------------------------------------------------------------------------

#endif//__linear_scoring2_complete_cuh__
