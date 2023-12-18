/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fields.cuh"
#include "libmycu/custages/scoring.cuh"
#include "libmycu/cudp/dpw_btck.cuh"
#include "libmycu/cudp/dpssw_btck.cuh"
#include "libmycu/cudp/dpw_btck_specscore.cuh"
#include "dpagc_ssw_btck_specscore.cuh"

// #define CUDPAGC_2DSPECAN_SS_BTCK_TESTPRINT 0 //3024//4899

// =========================================================================
// device functions for executing dynamic programming with backtracking 
// information;
// NOTE: Version for CUDP_2DSPECAN_DIM_D==CUDP_2DSPECAN_DIM_X!
// NOTE: See COMER2/COTHER source code for a general case!
// NOTE: USESS, template parameter, using secondary structure scoring additionally;
// blkdiagnum, block diagonal serial number;
// (starting at x=-CUDP_2DSPECAN_DIM);
// ndbCstrs, number of references in a chunk;
// ndbCposs, total number of db reference structure positions in a chunk;
// dbxpad, number of padded positions for memory alignment;
// ssweight, secondary structure score weight if ss scoring is in use;
// gapopencost, gap open cost;
// gapextcost, gap extension cost;

// DP processing layout:
// +---------===-+--------+--+-
// |  /  /  /  / |  /  /  |  |
// | /  /  /  /  | /  /  /| /|
// |/  /  /  /  /|/  /  / |/ |
// +---======----+---=====+--+-
// |  /  /  /  / |  /  /  |  |
// +=====--------+=====---+--+-
// (double line indicates current parallel processing)

// -------------------------------------------------------------------------
// ExecDPAGCSSwBtckSpecScores3264x: execute dynamic programming with affine
// gap cost model, secondary structure and backtracking information using 
// shared memory and 32(64)-fold unrolling along the diagonal of dimension 
// CUDP_2DSPECAN_DIM_D;
// NOTE: memory pointers should be aligned!
// specscores, address of spectral scores between a query and reference;
// tmpdpdiagbuffers, temporary buffers for last calculated diagonal scores;
// tmpdpbotbuffer, temporary buffers for last calculated bottom scores;
// maxscoordsbuf, coordinates (positions) of maximum alignment scores;
// btckdata, backtracking information data;
// 
template<bool USESS>
__global__
void ExecDPAGCSSwBtckSpecScores3264x(
    const uint blkdiagnum,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const float ssweight,
    const float gapopencost,
    const float gapextcost,
    const float* __restrict__ specscores,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
//     uint* __restrict__ maxscoordsbuf,
    char* __restrict__ btckdata)
{
    // blockIdx.x is the oblique block index in the current iteration of 
    // processing anti-diagonal blocks for all query-reference pairs in the chunk;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number;
    constexpr int DDIM = CUDP_2DSPECAN_DIM_D + 1;//inner dimension for diagonal buffers
    //cache of scores: use padding of +1 to eliminate bank conflicts when accessing the cache
    __shared__ float scoreCache[CUDP_2DSPECAN_DIM_D][CUDP_2DSPECAN_DIM_X+1];
    __shared__ float diag1Cache[nTDPAGCDiagScoreSubsections * DDIM];//cache for scores of the 1st diagonal
    __shared__ float diag2Cache[nTDPAGCDiagScoreSubsections * DDIM];//last (2nd) diagonal
    __shared__ float bottmCache[nTDPAGCDiagScoreSubsections * CUDP_2DSPECAN_DIM_X];//bottom scores
    char qrySS;//query secondary structure assignment; reference below:
    __shared__ char rfnSS[CUDP_2DCACHE_DIM_DpX+1];//+1 to avoid bank conflicts
    //NOTE: max scores and their coordinates are not recorded for semi-global alignment!
    //NOTE: comment out the variables:
    ///float maxscCache;//maximum scores of the last processed diagonal
    ///uint maxscCoords = 0;//coordinates of the maximum alignment score maxscCache
    //SECTION for backtracking information
    __shared__ char btckCache[CUDP_2DSPECAN_DIM_D][CUDP_2DSPECAN_DIM_X+1];
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse ccmCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(blockIdx.y, (int*)diag1Cache);
        GetQueryLenDst(blockIdx.z, (int*)diag1Cache + 2);
    }

#if (CUDP_2DSPECAN_DIM_D <= 32)
    __syncwarp();
#else
    __syncthreads();
#endif

    //NOTE: no bank conflict when threads from the same warp access the same address;
    dbstrlen = ((int*)diag1Cache)[0]; dbstrdst = ((int*)diag1Cache)[1];
    qrylen = ((int*)diag1Cache)[2]; qrydst = ((int*)diag1Cache)[3];

#if (CUDP_2DSPECAN_DIM_D <= 32)
    __syncwarp();
#else
    __syncthreads();
#endif


    //lastydiagnum, last block diagonal serial number along y axis:
    //each division separates a number of diagonals (nsepds);
    constexpr int nsepds = 2;//(float)CUDP_2DSPECAN_DIM_D/(float)CUDP_2DSPECAN_DIM_X + 1.0f;
    //the number of the last diagonal starting at x=-CUDP_2DSPECAN_DIM_D
    ///int nintdivs = (qrylen-1)>>CUDP_2DSPECAN_DIM_D_LOG2;//(qrylen-1)/CUDP_2DSPECAN_DIM_D;
    ///uint lastydiagnum = nsepds * nintdivs + 1 - 1;//-1 for zero-based indices;
    uint lastydiagnum = ((qrylen-1) >> CUDP_2DSPECAN_DIM_D_LOG2) * nsepds;


    // blockIdx.x is block serial number s within diagonal blkdiagnum;
    // (x,y) is the bottom-left corner (x,y) coordinates for structure blockIdx.y
    int x, y;
    if( blkdiagnum <= lastydiagnum) {
        //x=-!(d%2)w+2ws; y=dw/2+w-sw -1 (-1, zero-based indices); [when w==b]
        //(b, block's length; w, block's width)
        x = (2*blockIdx.x - (!(blkdiagnum & 1))) * CUDP_2DSPECAN_DIM_D;
        y = ((blkdiagnum>>1) + 1 - blockIdx.x) * CUDP_2DSPECAN_DIM_D - 1;
    } else {
        //x=-w+(d-d_l)w+2ws; y=dw/2+w-sw -1; [when w==b]
        x = (2*blockIdx.x + (blkdiagnum-lastydiagnum-1)) * CUDP_2DSPECAN_DIM_D;
        y = ((lastydiagnum>>1) + 1 - blockIdx.x) * CUDP_2DSPECAN_DIM_D - 1;
    }


    //number of iterations for this block to perform;
    int ilim = GetMaqxNoIterations(x, y, qrylen, dbstrlen, CUDP_2DSPECAN_DIM_X);

    if(y < 0 || qrylen <= (y+1 - CUDP_2DSPECAN_DIM_D) || 
       dbstrlen <= x /*+ CUDP_2DSPECAN_DIM_DpX */ ||
       ilim < 1)
        //block does not participate: out of bounds
        return;


    //READ DATA
    int qpos = y - threadIdx.x;//going upwards
    //x is now the position this thread will process
    x += threadIdx.x;

    if(USESS) {
        if(0 <= qpos && qpos < qrylen)
            DPLocCacheQrySS(qrySS, qpos + qrydst);
        else DPLocInitSS<0/*shift*/,pmvLOOP>(qrySS);
    }

    //db reference structure position corresponding to the oblique block's
    // bottom-left corner in the buffers dc_pm2dvfields_ (score matrix) 
    // plus the offset determined by thread id:
    int dbpos = x + dbstrdst;//going right
    int dblen = ndbCposs + dbxpad;
    //offset (w/o a factor) to the beginning of the data along the y axis 
    // wrt query blockIdx.z: 
    int yofff = dblen * blockIdx.z;

    if(USESS) {
        if(0 <= x && x < dbstrlen)
            DPLocCacheRfnSS<0/*shift*/>(rfnSS, dbpos);
        else DPLocInitSS<0/*shift*/,0/*VALUE*/>(rfnSS);

        if(0 <= (x+CUDP_2DCACHE_DIM_D) && (x+CUDP_2DCACHE_DIM_D) < dbstrlen)
            //NOTE: blockDim.x==CUDP_2DCACHE_DIM_D
            DPLocCacheRfnSS<CUDP_2DCACHE_DIM_D>(rfnSS, dbpos + CUDP_2DCACHE_DIM_D);
        else DPLocInitSS<CUDP_2DCACHE_DIM_D/*shift*/,0/*VALUE*/>(rfnSS);
    }

    //cache spectral scores
    #pragma unroll 4
    for(int i = 0; i < CUDP_2DCACHE_DIM_D; i++) {
        //going upwards
        scoreCache[i][threadIdx.x] = 0.0f;
        if(0 <= y-i && y-i < qrylen && 0 <= x+i && x+i < dbstrlen) {
            //starting position of line i of the oblq. diagonal block in the 
            //score matrix:
            int doffs = (qrydst + (y-i)) * dblen + i;
            scoreCache[i][threadIdx.x] = specscores[doffs + dbpos];
        }
    }


    //cache TWO DIAGONALS from the previous (along the x axis) oblique block;
    //the structure of tmpdpdiagbuffers is position-specific (1D, along the x axis)
    if(0 <= x-1 && x-1 < dbstrlen) {
        int doffs = nTDPDiagScoreSections * nTDPAGCDiagScoreSubsections * yofff;
        DPLocCacheBuffer<DDIM,0,nTDPAGCDiagScoreSubsections>(
                        diag1Cache, tmpdpdiagbuffers, dbpos-1, doffs, dblen);
        DPLocCacheBuffer<DDIM,1,nTDPAGCDiagScoreSubsections>(
                        diag2Cache, tmpdpdiagbuffers, dbpos-1, 
                          doffs + dpdssDiag2 * nTDPAGCDiagScoreSubsections * dblen, 
                          dblen);
        //NOTE: max scores and their coordinates are not recorded for 
        //NOTE: semi-global alignment!
        //cache the buffer of maximum scores here
        ///doffs += dpdssDiagM * nTDPAGCDiagScoreSubsections * dblen;
        ///maxscCache = tmpdpdiagbuffers[doffs + dbpos - 1];
        ///maxscCoords = maxscoordsbuf[yofff + dbpos - 1];
    }
    else {
        DPLocInitCache<DDIM,0,nTDPAGCDiagScoreSubsections>(diag1Cache);
        DPLocInitCache<DDIM,1/*shift*/,nTDPAGCDiagScoreSubsections>(diag2Cache);
        ///maxscCache = 0.0f;
    }

    //cache the bottom of the upper oblique blocks;
    //the structure of tmpdpbotbuffer is position-specific (1D, along x-axis)
    {
        int doffs = nTDPBottomScoreSections * nTDPAGCDiagScoreSubsections * yofff;
        if(CUDP_2DSPECAN_DIM_D <= y && 
           0 <= x+CUDP_2DSPECAN_DIM_D-1 && x+CUDP_2DSPECAN_DIM_D-1 < dbstrlen)
        {
            DPLocCacheBuffer<CUDP_2DSPECAN_DIM_X,0,nTDPAGCDiagScoreSubsections>( 
                bottmCache, tmpdpbotbuffer, dbpos+CUDP_2DSPECAN_DIM_D-1, doffs, dblen);
        }
        else {
            DPLocInitCache<CUDP_2DSPECAN_DIM_X,0,nTDPAGCDiagScoreSubsections>(bottmCache);
        }
    }


// #if (CUDP_2DSPECAN_DIM_D <= 32)
//     __syncwarp();
// #else
    //NOTE: surprisingly, __syncthreads here saves a lot of
    //NOTE: registers for later architectures
    __syncthreads();
// #endif


    float *pdiag1 = diag1Cache;
    float *pdiag2 = diag2Cache;

    //start calculations for this position with 32x/64x unrolling
    //NOTE: sync inside: do not branch;
    for(int i = 0; i < ilim/*CUDP_2DSPECAN_DIM_X*/; i++)
    {
        float val1, val2, max;
        int btck;

        if(threadIdx.x+1 == CUDP_2DSPECAN_DIM_D) {
            pdiag1[GetBufferNdx<DDIM>(dpagcdsssStateMM,threadIdx.x+1)] =
                bottmCache[GetBufferNdx<CUDP_2DSPECAN_DIM_X>(dpagcdsssStateMM,i)];
            pdiag1[GetBufferNdx<DDIM>(dpagcdsssStateMG,threadIdx.x+1)] =
                bottmCache[GetBufferNdx<CUDP_2DSPECAN_DIM_X>(dpagcdsssStateMG,i)];
            pdiag1[GetBufferNdx<DDIM>(dpagcdsssStateGM,threadIdx.x+1)] =
                bottmCache[GetBufferNdx<CUDP_2DSPECAN_DIM_X>(dpagcdsssStateGM,i)];
        }

        //MM state update (diagonal direction)
        val1 = pdiag2[GetBufferNdx<DDIM>(dpagcdsssStateMG,threadIdx.x+1)];
        val2 = pdiag2[GetBufferNdx<DDIM>(dpagcdsssStateGM,threadIdx.x+1)];
        val1 = myhdmax(val1, val2);
        val2 = pdiag2[GetBufferNdx<DDIM>(dpagcdsssStateMM,threadIdx.x+1)];
        val1 = myhdmax(val1, val2);
        val1 += scoreCache[threadIdx.x][i];
        if(USESS) val1 += (float)(qrySS == rfnSS[threadIdx.x+i]) * ssweight;
        max = val1;
        btck = dpbtckDIAG;
        //NOTE: max scores and their coordinates are not recorded for semi-global alignment
        ///dpmaxandcoords(maxscCache, val1, maxscCoords, x+i, y-threadIdx.x);

        //sync to update pdiag1; also no read for pdiag2 in this iteration
#if (CUDP_2DSPECAN_DIM_D <= 32)
        __syncwarp();
#else
        __syncthreads();
#endif
        pdiag2[GetBufferNdx<DDIM>(dpagcdsssStateMM,threadIdx.x)] = val1;//WRITE

        //GM state update (left direction)
        val1 = pdiag1[GetBufferNdx<DDIM>(dpagcdsssStateMM,threadIdx.x)] + gapopencost;
        val2 = pdiag1[GetBufferNdx<DDIM>(dpagcdsssStateGM,threadIdx.x)] + gapextcost;
        val1 = myhdmax(val1, val2);
        pdiag2[GetBufferNdx<DDIM>(dpagcdsssStateGM,threadIdx.x)] = val1;//WRITE
        myhdmaxassgn(max, val1, btck, (int)dpbtckLEFT);

        //MG state update (up direction)
        val1 = pdiag1[GetBufferNdx<DDIM>(dpagcdsssStateMM,threadIdx.x+1)] + gapopencost;
        val2 = pdiag1[GetBufferNdx<DDIM>(dpagcdsssStateMG,threadIdx.x+1)] + gapextcost;
        val1 = myhdmax/*fmaxf*/(val1, val2);
        pdiag2[GetBufferNdx<DDIM>(dpagcdsssStateMG,threadIdx.x)] = val1;//WRITE
        myhdmaxassgn(max, val1, btck, (int)dpbtckUP);

        btckCache[threadIdx.x][i] = btck;//WRITE

        if(threadIdx.x == 0) {
            //WRITE
            //this position is not used by other threads in the current iteration
            bottmCache[GetBufferNdx<CUDP_2DSPECAN_DIM_X>(dpagcdsssStateMM,i)] =
                pdiag2[GetBufferNdx<DDIM>(dpagcdsssStateMM,threadIdx.x)];
            bottmCache[GetBufferNdx<CUDP_2DSPECAN_DIM_X>(dpagcdsssStateMG,i)] =
                pdiag2[GetBufferNdx<DDIM>(dpagcdsssStateMG,threadIdx.x)];
            bottmCache[GetBufferNdx<CUDP_2DSPECAN_DIM_X>(dpagcdsssStateGM,i)] =
                pdiag2[GetBufferNdx<DDIM>(dpagcdsssStateGM,threadIdx.x)];
        }

#ifdef CUDPAGC_2DSPECAN_SS_BTCK_TESTPRINT
        if(blockIdx.y==CUDPAGC_2DSPECAN_SS_BTCK_TESTPRINT){
            printf(" d=%u(%u) s=%u i%02d/%u (t%02u): len= %d addr= %u SC= %.4f (yx: %d,%d) "
                    "MM= %.6f  MG= %.6f  GM= %.6f  BTCK= %d\n",// MAX= %.6f COORD= %x\n"
                    blkdiagnum,lastydiagnum,blockIdx.x,i,ilim,threadIdx.x,
                    dbstrlen,dbstrdst,val1, y-threadIdx.x,x+i,
                    pdiag2[GetBufferNdx<DDIM>(dpagcdsssStateMM,threadIdx.x)],
                    pdiag2[GetBufferNdx<DDIM>(dpagcdsssStateMG,threadIdx.x)],
                    pdiag2[GetBufferNdx<DDIM>(dpagcdsssStateGM,threadIdx.x)],
                    btck//maxscCache, maxscCoords,
            );
            for(size_t _k=0;_k<1000000000UL;_k++)clock();
            for(size_t _k=0;_k<1000000000UL;_k++)clock();
        }
#endif

        myhdswap(pdiag1, pdiag2);

        //sync for updates
// #if (CUDP_2DSPECAN_DIM_D <= 32)
//         __syncwarp();
// #else
        __syncthreads();
// #endif
    }


#ifdef CUDPAGC_2DSPECAN_SS_BTCK_TESTPRINT
    if(blockIdx.y==CUDPAGC_2DSPECAN_SS_BTCK_TESTPRINT)
        printf(" >>> d=%u(%u) s=%u (t%02u): len= %d wrt= %d xpos= %d\n", 
            blkdiagnum,lastydiagnum,blockIdx.x,threadIdx.x, dbstrlen,
            x+CUDP_2DSPECAN_DIM_X-1<dbstrlen, x+CUDP_2DSPECAN_DIM_X-1);
#endif


    //write the result for next-iteration blocks;
    //WRITE two diagonals;
    if(0 <= x+CUDP_2DSPECAN_DIM_X-1 && x+CUDP_2DSPECAN_DIM_X-1 < dbstrlen) {
        int doffs = nTDPDiagScoreSections * nTDPAGCDiagScoreSubsections * yofff;
        DPLocWriteBuffer<DDIM,0,nTDPAGCDiagScoreSubsections>(
                        pdiag1, tmpdpdiagbuffers, dbpos+CUDP_2DSPECAN_DIM_X-1, doffs,
                          dblen);
        DPLocWriteBuffer<DDIM,1,nTDPAGCDiagScoreSubsections>(
                        pdiag2, tmpdpdiagbuffers, dbpos+CUDP_2DSPECAN_DIM_X-1, 
                          doffs + dpdssDiag2 * nTDPAGCDiagScoreSubsections * dblen, 
                          dblen);
    }

    //NOTE: max scores and their coordinates are not recorded 
    //NOTE: for semi-global alignment; ignore this section
    ///int ndx = CUDP_2DSPECAN_DIM_X-1;
    ////top-left x coordinate of the oblique block:
    ///int xtl = x - threadIdx.x + blockDim.x - 1;
    ///if(xtl >= dbstrlen) ndx = 0;
    ///else if(xtl + ndx >= dbstrlen) ndx = dbstrlen - xtl - 1;

    //NOTE: max scores and their coordinates are not recorded 
    //NOTE: for semi-global alignment; ignore this section
    //WRITE the buffer of maximum scores
    ///if(0 <= x + ndx && x + ndx < dbstrlen) {
    ///    int doffs = nTDPDiagScoreSections * nTDPAGCDiagScoreSubsections * yofff;
    ///    doffs += dpdssDiagM * nTDPAGCDiagScoreSubsections * dblen;
    ///    if(CUDP_2DSPECAN_DIM_D <= y) {
    ///        if(tmpdpdiagbuffers[doffs + dbpos+ndx] < maxscCache) {
    ///            tmpdpdiagbuffers[doffs + dbpos+ndx] = maxscCache;
    ///            maxscoordsbuf[yofff + dbpos+ndx] = maxscCoords;
    ///        }
    ///    } else {
    ///        tmpdpdiagbuffers[doffs + dbpos+ndx] = maxscCache;
    ///        maxscoordsbuf[yofff + dbpos+ndx] = maxscCoords;
    ///    }
    ///}

    //WRITE the bottom of the diagonal block;
    {
        int doffs = nTDPBottomScoreSections * nTDPAGCDiagScoreSubsections * yofff;
        if(0 <= x && x < dbstrlen)
            DPLocWriteBuffer<CUDP_2DSPECAN_DIM_X,0,nTDPAGCDiagScoreSubsections>(
                bottmCache, tmpdpbotbuffer, dbpos, doffs, dblen);
    }

    //WRITE backtracking information
    for(int i = 0; i < CUDP_2DSPECAN_DIM_D; i++) {
        //going upwards
        if(0 <= y-i && y-i < qrylen && 0 <= x+i && x+i < dbstrlen
            //NOTE: as long as btckCache is computed for all block 
            //NOTE: data of a thread block, write all this data;
            //NOTE: alternatively, btckCache should initialize to 
            //NOTE: dpbtckSTOP for non-computed cells; this, however,
            //NOTE: does not provide considerable speed improvement;
        ) {
            char bv = btckCache[i][threadIdx.x];
            //starting position of line i of the oblq. diagonal block in the matrix:
            qpos = (qrydst + (y-i)) * dblen + i;
            btckdata[qpos + dbpos] = bv;
        }
    }
}

// =========================================================================
// Instantiations
//
#define INSTANTIATE_ExecDPAGCSSwBtckSpecScores3264x(tpUSESS) \
    template \
    __global__ void ExecDPAGCSSwBtckSpecScores3264x<tpUSESS>( \
        const uint blkdiagnum, \
        const uint ndbCstrs, const uint ndbCposs, const uint dbxpad, \
        const float ssweight, const float gapopencost, const float gapextcost, \
        const float* __restrict__ specscores, \
        float* __restrict__ tmpdpdiagbuffers, \
        float* __restrict__ tmpdpbotbuffer, \
        char* __restrict__ btckdata);

INSTANTIATE_ExecDPAGCSSwBtckSpecScores3264x(false);
INSTANTIATE_ExecDPAGCSSwBtckSpecScores3264x(true);

// -------------------------------------------------------------------------
