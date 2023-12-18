/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fields.cuh"
#include "similarity.cuh"

// -------------------------------------------------------------------------
// VerifyAlignmentScore: calculate local sequence alignment score between the
// queries and reference structures and set the flag of low score if it is 
// below the threshold;
// seqsimthrscore, sequence similarity threshold score;
// nqystrs, total number of query structures in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// NOTE: thread block is 1D and processes ungapped fragments of aligned 
// NOTE: query-reference structures;
// NOTE: block size is warp size! appropriate synchronization is used!
// 
__global__ void VerifyAlignmentScore(
    const float seqsimthrscore,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux)
{
    enum {
        lXdim = CUFL_TBSP_SEQUENCE_SIMILARITY_XDIM,
        lEdge = CUFL_TBSP_SEQUENCE_SIMILARITY_EDGE,
        lStep = CUFL_TBSP_SEQUENCE_SIMILARITY_STEP,
        lStepLog2 = CUFL_TBSP_SEQUENCE_SIMILARITY_STEPLOG2
    };

    //cache for scores and sequences
    // __shared__ float scores[lXdim];
    // __shared__ float pxmins[lXdim];

    const uint dbstrndx = blockIdx.x;//reference serial number
    const uint diagnumb = blockIdx.y;//diagonal number determines qrypos, rfnpos
    const uint qryndx = blockIdx.z;//query serial number
    int qrylen, dbstrlen;//query and reference length
    int qrypos, rfnpos;//starting query and reference position
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;
    const uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;

    if(seqsimthrscore <= 0.0f) return;

    //read convergence first
    if(threadIdx.x == 0)
        qrypos = (int)(wrkmemaux[mloc0 + dbstrndx]);

    qrypos = __shfl_sync(0xffffffff/*mask*/, qrypos, 0/*srcLane*/);

    //all threads in the block exit upon unset convergence
    if(qrypos == 0) return;

    if(threadIdx.x == 0) dbstrlen = GetDbStrLength(dbstrndx);
    if(threadIdx.x == 0) dbstrdst = GetDbStrDst(dbstrndx);
    if(threadIdx.x == 0) qrylen = GetQueryLength(qryndx);
    if(threadIdx.x == 0) qrydst = GetQueryDst(qryndx);

    dbstrlen = __shfl_sync(0xffffffff/*mask*/, dbstrlen, 0/*srcLane*/);
    dbstrdst = __shfl_sync(0xffffffff/*mask*/, dbstrdst, 0/*srcLane*/);
    qrylen = __shfl_sync(0xffffffff/*mask*/, qrylen, 0/*srcLane*/);
    qrydst = __shfl_sync(0xffffffff/*mask*/, qrydst, 0/*srcLane*/);

    qrypos = diagnumb * lStep;
    rfnpos = 0;

    if(qrylen - lEdge <= qrypos) {
        rfnpos = ((qrypos - (qrylen - lEdge)) & (~lStepLog2)) + lStep;
        qrypos = 0;
    }

    //all threads exit if upon the out of bounds condition:
    if(qrylen - lEdge <= qrypos || dbstrlen - lEdge <= rfnpos) return;

    float prvsum = 0.0f, prvmin = 0.0f, maxsc = 0.0f;

    for(; qrypos < qrylen && rfnpos < dbstrlen; qrypos += lXdim, rfnpos += lXdim)
    {
        int qp = qrypos + threadIdx.x;
        int rp = rfnpos + threadIdx.x;
        float scores = 0.0f;
        float pxmins, locmax;

        //calculate positional scores:
        if(qp < qrylen && rp < dbstrlen) {
            const char qryRE = GetQueryRsd(qrydst + qp);
            const char rfnRE = GetDbStrRsd(dbstrdst + rp);
            scores = GetGonnetScore(qryRE, rfnRE);
        }

        //add the previous sum (tid 0 only), calculate prefix sums:
        if(threadIdx.x == 0) scores += prvsum;
        scores = mywarpincprefixsum(scores);
        pxmins = scores;
        //take min, calculate prefix mins of the prefix sums:
        if(threadIdx.x == 0) pxmins = myhdmin(pxmins, prvmin);
        pxmins = mywarpincprefixmin(pxmins);
        //update previous values for tid 0 only:
        prvsum = __shfl_sync(0xffffffff/*mask*/, scores, 31/*srcLane*/);
        prvmin = __shfl_sync(0xffffffff/*mask*/, pxmins, 31/*srcLane*/);
        // if(threadIdx.x == 0 || threadIdx.x == 31) {
        //     prvsum = __shfl_sync(0x80000001/*mask*/, scores, 31/*srcLane*/);
        //     prvmin = __shfl_sync(0x80000001/*mask*/, pxmins, 31/*srcLane*/);
        // }
        // __syncwarp();
        //calculate local alignment scores
        scores -= myhdmin(0.0f, pxmins);
        //find max score:
        locmax = mywarpreducemax(scores);
        locmax = __shfl_sync(0xffffffff/*mask*/, locmax, 0/*srcLane*/);
        maxsc = myhdmax(maxsc, locmax);
        //check if it is grater than the threshold:
        if(seqsimthrscore <= maxsc) break;
    }

    //NOTE: several blocks may write at the same time at the same location;
    //NOTE: safe as long as it's the same value;
    if(threadIdx.x == 0 && seqsimthrscore <= maxsc)
        //reset convergence flag:
        wrkmemaux[mloc0 + dbstrndx] = 0;
}
