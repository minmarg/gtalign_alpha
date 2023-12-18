/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libutil/CLOptions.h"
#include "libgenp/gproc/gproc.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/custages/fields.cuh"
#include "libmycu/custages/covariance.cuh"
#include "local_similarity02.cuh"

// -------------------------------------------------------------------------
// CalcLocalSimilarity2_frg2: calculate provisional local similarity during 
// extensive fragment-based search of optimal superpositions;
// NOTE: thread block is 1D and processes along reference structures;
// NOTE: CUSF_TBSP_LOCAL_SIMILARITY_XDIM should be 32: warp-level sync used;
// thrsimilarityperc, threshold percentage of local similarity score for a 
// fragment to be considered as one having the potential to improve superposition;
// depth, superposition depth for calculating query and reference positions;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db structure positions in the chunk;
// dbxpad, number of padded positions for memory alignment;
// maxnsteps, max number of steps (blockIdx.y) to perform for each reference structure;
// qryfragfct, fragment factor for query (to be multiplied by step dependent upon lengths);
// rfnfragfct, fragment factor for reference (to be multiplied by step dependent upon lengths);
// fragndx, fragment index determining the fragment size dependent upon lengths;
// NOTE: memory pointers should be aligned!
// dpscoremtx, input rounded global dp score matrix;
// wrkmemaux, auxiliary working memory;
// 
__global__
void CalcLocalSimilarity2_frg2(
    const float thrsimilarityperc,
    const int depth,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    const char* __restrict__ dpscoremtx,
    float* __restrict__ wrkmemaux)
{
    // blockIdx.x, reference serial number;
    // blockIdx.y, fragment factor;
    // blockIdx.z, query serial number;
    enum {
        lnYIts = 3,//number of iterations along y-axis
        lYdim = CUSF_TBSP_LOCAL_SIMILARITY_YDIM,
        lXdim = CUSF_TBSP_LOCAL_SIMILARITY_XDIM
    };
    // cache for dp scores:
    __shared__ uint dpsCache[lYdim][lXdim+1];
    //reference serial number
    const uint dbstrndx = blockIdx.x;
    const uint sfragfct = blockIdx.y;//fragment factor
    const uint sfragfct0 = sfragfct & (~1);
    const uint sfragfct1 = sfragfct | (1);
    const uint qryndx = blockIdx.z;//query serial number
    fragndx = (sfragfct & 1);
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;
    int qrypos, rfnpos;//starting query and reference position
    int fraglen;


    //NOTE: all threads exit for fragndx>0!
    if(fragndx) return;


    //read convergence first
    if(threadIdx.x == 0 && threadIdx.y == 0) {
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        dpsCache[0][lXdim] = (unsigned int)(wrkmemaux[mloc0 + dbstrndx]);
        if(dpsCache[0][lXdim] <= CONVERGED_FRAGREF_bitval) dpsCache[0][lXdim] = 0;
    }

    __syncthreads();

    if(dpsCache[0][lXdim]) {
        uint mlocf0 = ((qryndx * maxnsteps + sfragfct0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        uint mlocf1 = ((qryndx * maxnsteps + sfragfct1) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        //NOTE: global conv flag distributes across sfragfct;
        //NOTE: do not write at 0: avoid race conditions!
        if(threadIdx.x == 0 && threadIdx.y == 0 && sfragfct0)
            wrkmemaux[mlocf0 + dbstrndx] = (dpsCache[0][lXdim] | CONVERGED_SCOREDP_bitval);
        if(threadIdx.x == 0 && threadIdx.y == 1)
            wrkmemaux[mlocf1 + dbstrndx] = (dpsCache[0][lXdim] | CONVERGED_SCOREDP_bitval);
        //NOTE: safe exit for all threads in the block: dpsCache[0][lXdim] won't be overwritten!
        return;
    }


    //NOTE: the kernel distributes the convergence flag when no similarity is to be computed
    if(thrsimilarityperc <= 0.0f) return;


    //read query and reference attributes:
    if(threadIdx.x == 0 && threadIdx.y == 0) dpsCache[0][0] = GetDbStrLength(dbstrndx);
    if(threadIdx.x == 0 && threadIdx.y == 1) dpsCache[0][1] = GetDbStrDst(dbstrndx);

    if(threadIdx.x == 0 && threadIdx.y == 2) dpsCache[0][2] = GetQueryLength(qryndx);
    if(threadIdx.x == 0 && threadIdx.y == 3) dpsCache[0][3] = GetQueryDst(qryndx);

    __syncthreads();

    dbstrlen = dpsCache[0][0]; dbstrdst = dpsCache[0][1];
    qrylen = dpsCache[0][2]; qrydst = dpsCache[0][3];

    __syncthreads();


    //calculate start positions
    GetQryRfnPos_frg2(
        depth,
        qrypos, rfnpos, qrylen, dbstrlen, sfragfct, qryfragfct, rfnfragfct, fragndx
    );

    fraglen = GetNAlnPoss_frg(
        qrylen, dbstrlen, 0/*qrypos,unused*/, 0/*rfnpos,unused*/,
        qryfragfct/*unused*/, rfnfragfct/*unused*/, 0/*fragndx; first smaller*/);

    //if fragment is out of bounds, set the convergence flag anyway
    if(qrylen < qrypos + fraglen || dbstrlen < rfnpos + fraglen || fraglen < 1) {
        uint mlocf0 = ((qryndx * maxnsteps + sfragfct0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        uint mlocf1 = ((qryndx * maxnsteps + sfragfct1) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        //NOTE: global conv flag (checked above already) distributes across sfragfct;
        //NOTE: do not write at 0: avoid race conditions!
        if(threadIdx.x == 0 && threadIdx.y == 0 && sfragfct0)
            wrkmemaux[mlocf0 + dbstrndx] = CONVERGED_SCOREDP_bitval;
        if(threadIdx.x == 0 && threadIdx.y == 1)
            wrkmemaux[mlocf1 + dbstrndx] = CONVERGED_SCOREDP_bitval;
        return;
    }

    fraglen = GetNAlnPoss_frg(
        qrylen, dbstrlen, 0/*qrypos,unused*/, 0/*rfnpos,unused*/,
        qryfragfct/*unused*/, rfnfragfct/*unused*/,
        1/*fragndx; always use the larger*/);

    //convergence flags for the fragments of smaller and larger lengths:
    int convflag0 = 0;
    int convflag1 =
        (qrylen < qrypos + fraglen || dbstrlen < rfnpos + fraglen || fraglen < 1)
        ? CONVERGED_SCOREDP_bitval: 0;

    //actual length of fragment within which local scores are verified:
    // fraglen = lnYIts * lYdim;
    if(qrylen < qrypos + fraglen) fraglen = qrylen - qrypos;
    if(dbstrlen < rfnpos + fraglen) fraglen = dbstrlen - rfnpos;


    //read a block of dp matrix values (coalesced reads), get max, and
    //set the convergence flag if calculated local similarity is too low:
    const int dblen = ndbCposs + dbxpad;
    int i;

    dpsCache[threadIdx.y][threadIdx.x] = 0;

    for(qrypos += threadIdx.y, i = 0; 
        qrypos < qrylen && i < lnYIts;
        i++, qrypos += lYdim/*blockDim.y*/)
    {
        //NOTE: +3 for max difference of address up-alignment
        if(rfnpos + (threadIdx.x+1) * 4 + 3 < dbstrlen)
        {
            int rpos = (qrydst + qrypos) * dblen + dbstrdst + rfnpos + threadIdx.x * 4;
            //NOTE: make sure position is 4-bytes aligned as data is read in words!
            //NOTE: aligned bytes imply that the results may differ a bit upon a
            //NOTE: different configuration of #queries and references in a chunk!
            //NOTE: that's because rpos takes on a different value and dpsCache
            //NOTE: reads from different locations at the boundaries; this may cause a
            //NOTE: different value for max!
            rpos = ALIGN_UP(rpos, 4/*sizeof(int)*/);
            uint nval = *(const uint*)(dpscoremtx + rpos);//READ 4 bytes
            uint oval = dpsCache[threadIdx.y][threadIdx.x];
            dpsCache[threadIdx.y][threadIdx.x] = __vmaxu4(nval, oval);//get per-byte max
        }
    }

    {   //max in each warp
        uint value = dpsCache[threadIdx.y][threadIdx.x];
        value = __vmaxu4(value, __shfl_down_sync(0xffffffff, value, 16));
        value = __vmaxu4(value, __shfl_down_sync(0xffffffff, value, 8));
        value = __vmaxu4(value, __shfl_down_sync(0xffffffff, value, 4));
        value = __vmaxu4(value, __shfl_down_sync(0xffffffff, value, 2));
        value = __vmaxu4(value, __shfl_down_sync(0xffffffff, value, 1));
        if(threadIdx.x == 0) dpsCache[threadIdx.y][0] = value;
    }

    __syncthreads();

    if(threadIdx.y == 0) {
        //global max across warps
        uint value = dpsCache[threadIdx.x][0];
        value = __vmaxu4(value, __shfl_down_sync(0xffffffff, value, 16));
        value = __vmaxu4(value, __shfl_down_sync(0xffffffff, value, 8));
        value = __vmaxu4(value, __shfl_down_sync(0xffffffff, value, 4));
        value = __vmaxu4(value, __shfl_down_sync(0xffffffff, value, 2));
        value = __vmaxu4(value, __shfl_down_sync(0xffffffff, value, 1));
        if(threadIdx.x == 0) {
            //get max byte and save it
            uint max = myhdmax(value & 0xff, (value >> 24) & 0xff);
            max = myhdmax(max, (value >> 16) & 0xff);
            max = myhdmax(max, (value >> 8) & 0xff);
            // dpsCache[0][0] = max;
            //NOTE: consider ignoring check for lengths <9 (4+3+2, word+mem.aln.+margin).
            if((float)max < (float)fraglen * thrsimilarityperc) {
                convflag0 = convflag0 | CONVERGED_SCOREDP_bitval;
                convflag1 = convflag1 | CONVERGED_SCOREDP_bitval;
            }
            //write convergence flags
            uint mlocf0 = ((qryndx * maxnsteps + sfragfct0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
            uint mlocf1 = ((qryndx * maxnsteps + sfragfct1) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
            //NOTE: do not write at 0: avoid race conditions!
            if(sfragfct0) wrkmemaux[mlocf0 + dbstrndx] = convflag0;
            wrkmemaux[mlocf1 + dbstrndx] = convflag1;
        }
    }
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
