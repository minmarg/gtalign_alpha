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
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/covariance_refn.cuh"
#include "libmycu/custages/transform.cuh"
#include "libmycu/custages/scoring.cuh"
#include "libmycu/custages/fields.cuh"
#include "dpw_btck.cuh"
#include "dpssw_btck.cuh"

// #define CUDPSS_INIT_BTCK_TESTPRINT 0 //3024//4899

// =========================================================================

// NOTE: parameters are passed to the device via constant memory and are 
// limited to 4 KB
// 
// device functions for executing dynamic programming with backtracking 
// information using secondary structure information;
// NOTE: Version for CUDP_2DCACHE_DIM_DequalsX: CUDP_2DCACHE_DIM_D==CUDP_2DCACHE_DIM_X!
// NOTE: See COMER2/COTHER source code for a general case!
// USESEQSCORING, template parameter, flag of additionally using residue score matrix;
// blkdiagnum, block diagonal serial number;
// (starting at x=-CUDP_2DCACHE_DIM);
// ndbCstrs, number of references in a chunk;
// ndbCposs, total number of db reference structure positions in a chunk;
// dbxpad, number of padded positions for memory alignment;
// maxnsteps, max number of steps performed for each reference structure 
// during alignment refinement;
// weight4ss, weight for scoring secondary structure;
// weight4rr, weight for pairwise residue score;
// gapopencost, gap open cost;

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
// ExecDPSSwBtck3264x: execute dynamic programming with secondary 
// structure and backtracking information using shared memory and 
// 32(64)-fold unrolling along the diagonal of dimension CUDP_2DCACHE_DIM;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary buffers for last calculated diagonal scores;
// tmpdpbotbuffer, temporary buffers for last calculated bottom scores;
// maxscoordsbuf, coordinates (positions) of maximum alignment scores;
// btckdata, backtracking information data;
// 
template <bool USESEQSCORING>
__global__
void ExecDPSSwBtck3264x(
    const uint blkdiagnum,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const float weight4ss,
    const float weight4rr,
    const float gapopencost,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
    char* __restrict__ btckdata)
{
    // blockIdx.x is the oblique block index in the current iteration of 
    // processing anti-diagonal blocks for all query-reference pairs in the chunk;
    // uint dbstrndx = blockIdx.y;//reference serial number;
    // uint qryndx = blockIdx.z;//query serial number
    constexpr int DDIM = CUDP_2DCACHE_DIM_D + 1;//inner dimension for diagonal buffers
    //cache for scores, coordinates, and transformation matrix:
    __shared__ float diag1Cache[nTDPDiagScoreSubsections * DDIM];//cache for scores of the 1st diagonal
    __shared__ float diag2Cache[nTDPDiagScoreSubsections * DDIM];//last (2nd) diagonal
    __shared__ float bottmCache[nTDPDiagScoreSubsections * CUDP_2DCACHE_DIM_X];//bottom scores
    char qrySS, qryRE;//query secondary structure assignment and residue; reference below:
    __shared__ char rfnSS[CUDP_2DCACHE_DIM_DpX+1];//+1 to avoid bank conflicts
    __shared__ char rfnRE[CUDP_2DCACHE_DIM_DpX+1];//+1 to avoid bank conflicts
    //NOTE: max scores and their coordinates are not recorded for semi-global alignment!
    //NOTE: comment out the variables:
    ///float maxscCache;//maximum scores of the last processed diagonal
    ///uint maxscCoords = 0;//coordinates of the maximum alignment score maxscCache
    //SECTION for backtracking information
    __shared__ char btckCache[CUDP_2DCACHE_DIM_D][CUDP_2DCACHE_DIM_X+1];
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;


//     //check for SCORE convergence
//     if(threadIdx.x == 0) {
//         uint mloc = ((qryndx * maxnsteps + 0/*sfragfct*/) * nTAuxWorkingMemoryVars) * ndbCstrs;
//         diag1Cache[0] = wrkmemaux[mloc + tawmvConverged * ndbCstrs + dbstrndx];
//     }

// #if (CUDP_2DCACHE_DIM_D <= 32)
//     __syncwarp();
// #else
//     __syncthreads();
// #endif

//     if(((int)diag1Cache[0]) & (CONVERGED_LOWTMSC_bitval))
//         //the termination flag for this pair is set;
//         //all threads in the block exit;
//         return;


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse cache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(blockIdx.y, (int*)diag2Cache);
        GetQueryLenDst(blockIdx.z, (int*)diag2Cache + 2);
    }

#if (CUDP_2DCACHE_DIM_D <= 32)
    __syncwarp();
#else
    __syncthreads();
#endif

    //NOTE: no bank conflict when two threads from the same warp access the same address;
    dbstrlen = ((int*)diag2Cache)[0]; dbstrdst = ((int*)diag2Cache)[1];
    qrylen = ((int*)diag2Cache)[2]; qrydst = ((int*)diag2Cache)[3];

#if (CUDP_2DCACHE_DIM_D <= 32)
    __syncwarp();
#else
    __syncthreads();
#endif


    //lastydiagnum, last block diagonal serial number along y axis:
    //each division separates a number of diagonals (nsepds);
    constexpr int nsepds = 2;//(float)CUDP_2DCACHE_DIM_D/(float)CUDP_2DCACHE_DIM_X + 1.0f;
    //the number of the last diagonal starting at x=-CUDP_2DCACHE_DIM_D
    ///int nintdivs = (qrylen-1)>>CUDP_2DCACHE_DIM_D_LOG2;//(qrylen-1)/CUDP_2DCACHE_DIM_D;
    ///uint lastydiagnum = nsepds * nintdivs + 1 - 1;//-1 for zero-based indices;
    uint lastydiagnum = ((qrylen-1) >> CUDP_2DCACHE_DIM_D_LOG2) * nsepds;


    // blockIdx.x is block serial number s within diagonal blkdiagnum;
    // (x,y) is the bottom-left corner (x,y) coordinates for structure blockIdx.y
    int x, y;
    if( blkdiagnum <= lastydiagnum) {
        //x=-!(d%2)w+2ws; y=dw/2+w-sw -1 (-1, zero-based indices); [when w==b]
        //(b, block's length; w, block's width)
        x = (2*blockIdx.x - (!(blkdiagnum & 1))) * CUDP_2DCACHE_DIM_D;
        y = ((blkdiagnum>>1) + 1 - blockIdx.x) * CUDP_2DCACHE_DIM_D - 1;
    } else {
        //x=-w+(d-d_l)w+2ws; y=dw/2+w-sw -1; [when w==b]
        x = (2*blockIdx.x + (blkdiagnum-lastydiagnum-1)) * CUDP_2DCACHE_DIM_D;
        y = ((lastydiagnum>>1) + 1 - blockIdx.x) * CUDP_2DCACHE_DIM_D - 1;
    }


    //number of iterations for this block to perform;
    int ilim = GetMaqxNoIterations(x, y, qrylen, dbstrlen, CUDP_2DCACHE_DIM_X);

    if(y < 0 || qrylen <= (y+1 - CUDP_2DCACHE_DIM_D) || 
       dbstrlen <= x /*+ CUDP_2DCACHE_DIM_DpX */ ||
       ilim < 1)
        //block does not participate: out of bounds
        return;


    //READ secondary structure assignment
    int qpos = y - threadIdx.x;//going upwards
    //x is now the position this thread will process
    x += threadIdx.x;

    if(0 <= qpos && qpos < qrylen) {
        if(USESEQSCORING) DPLocCacheQryRsd(qryRE, qpos + qrydst);
        DPLocCacheQrySS(qrySS, qpos + qrydst);
    } else {
        if(USESEQSCORING) DPLocInitRsd<0/*shift*/,0/*value*/>(qryRE);
        DPLocInitSS<0/*shift*/,pmvLOOP>(qrySS);
    }

    //db reference structure position corresponding to the oblique block's
    // bottom-left corner in the buffers dc_pm2dvfields_ (score matrix) 
    // plus the offset determined by thread id:
    int dbpos = x + dbstrdst;//going right
    int dblen = ndbCposs + dbxpad;
    //offset (w/o a factor) to the beginning of the data along the y axis 
    // wrt query blockIdx.z: 
    int yofff = dblen * blockIdx.z;

    if(0 <= x && x < dbstrlen) {
        if(USESEQSCORING) DPLocCacheRfnRsd<0/*shift*/>(rfnRE, dbpos);
        DPLocCacheRfnSS<0/*shift*/>(rfnSS, dbpos);
    } else {
        if(USESEQSCORING) DPLocInitRsd<0/*shift*/,0/*VALUE*/>(rfnRE);
        DPLocInitSS<0/*shift*/,0/*VALUE*/>(rfnSS);
    }

    if(0 <= (x+CUDP_2DCACHE_DIM_D) && (x+CUDP_2DCACHE_DIM_D) < dbstrlen) {
        //NOTE: blockDim.x==CUDP_2DCACHE_DIM_D
        if(USESEQSCORING) DPLocCacheRfnSS<CUDP_2DCACHE_DIM_D>(rfnRE, dbpos + CUDP_2DCACHE_DIM_D);
        DPLocCacheRfnSS<CUDP_2DCACHE_DIM_D>(rfnSS, dbpos + CUDP_2DCACHE_DIM_D);
    } else {
        if(USESEQSCORING) DPLocInitRsd<CUDP_2DCACHE_DIM_D/*shift*/,0/*VALUE*/>(rfnRE);
        DPLocInitSS<CUDP_2DCACHE_DIM_D/*shift*/,0/*VALUE*/>(rfnSS);
    }


    //cache TWO DIAGONALS from the previous (along the x axis) oblique block;
    //the structure of tmpdpdiagbuffers is position-specific (1D, along the x axis)
    if(0 <= x-1 && x-1 < dbstrlen) {
        int doffs = nTDPDiagScoreSections * nTDPDiagScoreSubsections * yofff;
        DPLocCacheBuffer<DDIM>(diag1Cache, tmpdpdiagbuffers, dbpos-1, doffs, dblen);
        DPLocCacheBuffer<DDIM,1>(diag2Cache, tmpdpdiagbuffers, dbpos-1, 
                          doffs + dpdssDiag2 * nTDPDiagScoreSubsections * dblen, 
                          dblen);
        //NOTE: max scores and their coordinates are not recorded for 
        //NOTE: semi-global alignment!
        //cache the buffer of maximum scores here
        ///doffs += dpdssDiagM * nTDPDiagScoreSubsections * dblen;
        ///maxscCache = tmpdpdiagbuffers[doffs + dbpos - 1];
        ///maxscCoords = maxscoordsbuf[yofff + dbpos - 1];
    }
    else {
        DPLocInitCache<DDIM>(diag1Cache, DPSSDEFINITSCOREVAL);
        DPLocInitCache<DDIM,1/*shift*/>(diag2Cache, DPSSDEFINITSCOREVAL);
        ///maxscCache = 0.0f;
    }

    //cache the bottom of the upper oblique blocks;
    //the structure of tmpdpbotbuffer is position-specific (1D, along x-axis)
    {
        int doffs = nTDPBottomScoreSections * nTDPDiagScoreSubsections * yofff;
        if(CUDP_2DCACHE_DIM_D <= y && 
           0 <= x+CUDP_2DCACHE_DIM_D-1 && x+CUDP_2DCACHE_DIM_D-1 < dbstrlen)
        {
            DPLocCacheBuffer<CUDP_2DCACHE_DIM_X>( 
                bottmCache, tmpdpbotbuffer, dbpos+CUDP_2DCACHE_DIM_D-1, doffs, dblen);
        }
        else {
            DPLocInitCache<CUDP_2DCACHE_DIM_X>(bottmCache, DPSSDEFINITSCOREVAL);
        }
    }


// #if (CUDP_2DCACHE_DIM_D <= 32)
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
    for(int i = 0; i < ilim/*CUDP_2DCACHE_DIM_X*/; i++)
    {
        float val1, val2;
        int btck;

        if(threadIdx.x+1 == CUDP_2DCACHE_DIM_D) {
            pdiag1[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x+1)] =
                bottmCache[GetBufferNdx<CUDP_2DCACHE_DIM_X>(dpdsssStateMM,i)];
        }

        val1 = (float)(qrySS == rfnSS[threadIdx.x+i]) * weight4ss;
        if(USESEQSCORING) val1 += GetGonnetScore(qryRE, rfnRE[threadIdx.x+i]) * weight4rr;

        //NOTE: TRICK to implement a special case of DP with affine gap cost scheme:
        //NOTE: gap extension cost is 0;
        //NOTE: this version for positive and negative match scores:
        //NOTE: add and appropriately subtract max possible match score to/from max 
        //NOTE: value to indicate diagonal direction in alignment;

        //MM state update (diagonal direction)
        val2 = pdiag2[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x+1)];
        //val2 < large_val means diagonal direction, adjust appropriately:
        if(val2 < DPSSDEFINITSCOREVAL_cmp) val2 -= DPSSDEFINITSCOREVAL;
        val1 += val2;
        btck = dpbtckDIAG;
        //NOTE: max scores and their coordinates are not recorded for semi-global alignment
        ///dpmaxandcoords(maxscCache, val1, maxscCoords, x+i, y-threadIdx.x);

        //sync to update pdiag1; also no read for pdiag2 in this iteration
#if (CUDP_2DCACHE_DIM_D <= 32)
        __syncwarp();
#else
        __syncthreads();
#endif

        //IM state update (left direction)
        val2 = pdiag1[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x)];
        //val2 < large_val means diagonal direction, adjust appropriately and add cost:
        if(val2 < DPSSDEFINITSCOREVAL_cmp) val2 = gapopencost + val2 - DPSSDEFINITSCOREVAL;
        myhdmaxassgn(val1, val2, btck, (int)dpbtckLEFT);

        //MI state update (up direction)
        val2 = pdiag1[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x+1)];
        if(val2 < DPSSDEFINITSCOREVAL_cmp) val2 = gapopencost + val2 - DPSSDEFINITSCOREVAL;
        myhdmaxassgn(val1, val2, btck, (int)dpbtckUP);

        //WRITE: write max value
        pdiag2[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x)] =
            (btck != dpbtckDIAG)? val1: val1 + DPSSDEFINITSCOREVAL;

        //WRITE
        btckCache[threadIdx.x][i] = btck;

        if(threadIdx.x == 0) {
            //WRITE
            //this position is not used by other threads in the current iteration
            bottmCache[GetBufferNdx<CUDP_2DCACHE_DIM_X>(dpdsssStateMM,i)] =
                pdiag2[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x)];
        }

#ifdef CUDPSS_INIT_BTCK_TESTPRINT
        if(blockIdx.y==CUDPSS_INIT_BTCK_TESTPRINT){
            printf(" d=%u(%u) s=%u i%02d/%u (t%02u): len= %d addr= %u SC= %.4f (yx: %d,%d) "
                    "MM= %.6f  "// MAX= %.6f COORD= %x\n"// BTCK= %d\n"
                    "  >qSS= %d   dSS= %d\n",
                    blkdiagnum,lastydiagnum,blockIdx.x,i,ilim,threadIdx.x,
                    dbstrlen,dbstrdst,val1, y-threadIdx.x,x+i,
                    pdiag2[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x)],
                    //maxscCache, maxscCoords,// btck,
                    qrySS, rfnSS[threadIdx.x+i]
            );
            for(size_t _k=0;_k<1000000000UL;_k++)clock();
            for(size_t _k=0;_k<1000000000UL;_k++)clock();
        }
#endif

        myhdswap(pdiag1, pdiag2);

        //sync for updates
// #if (CUDP_2DCACHE_DIM_D <= 32)
//         __syncwarp();
// #else
        __syncthreads();
// #endif
    }


#ifdef CUDPSS_INIT_BTCK_TESTPRINT
    if(blockIdx.y==CUDPSS_INIT_BTCK_TESTPRINT)
        printf(" >>> d=%u(%u) s=%u (t%02u): len= %d wrt= %d xpos= %d\n", 
            blkdiagnum,lastydiagnum,blockIdx.x,threadIdx.x, dbstrlen,
            x+CUDP_2DCACHE_DIM_X-1<dbstrlen, x+CUDP_2DCACHE_DIM_X-1);
#endif


    //write the result for next-iteration blocks;
    //WRITE two diagonals;
    if(0 <= x+CUDP_2DCACHE_DIM_X-1 && x+CUDP_2DCACHE_DIM_X-1 < dbstrlen) {
        int doffs = nTDPDiagScoreSections * nTDPDiagScoreSubsections * yofff;
        DPLocWriteBuffer<DDIM>(pdiag1, tmpdpdiagbuffers, dbpos+CUDP_2DCACHE_DIM_X-1, doffs,
                          dblen);
        DPLocWriteBuffer<DDIM,1>(pdiag2, tmpdpdiagbuffers, dbpos+CUDP_2DCACHE_DIM_X-1, 
                          doffs + dpdssDiag2 * nTDPDiagScoreSubsections * dblen, 
                          dblen);
    }

    //NOTE: max scores and their coordinates are not recorded 
    //NOTE: for semi-global alignment; ignore this section
    ///int ndx = CUDP_2DCACHE_DIM_X-1;
    ////top-left x coordinate of the oblique block:
    ///int xtl = x - threadIdx.x + blockDim.x - 1;
    ///if(xtl >= dbstrlen) ndx = 0;
    ///else if(xtl + ndx >= dbstrlen) ndx = dbstrlen - xtl - 1;

    //NOTE: max scores and their coordinates are not recorded 
    //NOTE: for semi-global alignment; ignore this section
    //WRITE the buffer of maximum scores
    ///if(0 <= x + ndx && x + ndx < dbstrlen) {
    ///    int doffs = nTDPDiagScoreSections * nTDPDiagScoreSubsections * yofff;
    ///    doffs += dpdssDiagM * nTDPDiagScoreSubsections * dblen;
    ///    if(CUDP_2DCACHE_DIM_D <= y) {
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
        int doffs = nTDPBottomScoreSections * nTDPDiagScoreSubsections * yofff;
        if(0 <= x && x < dbstrlen)
            DPLocWriteBuffer<CUDP_2DCACHE_DIM_X>(
                bottmCache, tmpdpbotbuffer, dbpos, doffs, dblen);
    }

    //WRITE backtracking information
    #pragma unroll 8
    for(int i = 0; i < CUDP_2DCACHE_DIM_D; i++) {
        //going upwards
        if(0 <= y-i && y-i < qrylen && 0 <= x+i && x+i < dbstrlen) {
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
#define INSTANTIATE_ExecDPSSwBtck3264x(tpUSESEQSCORING) \
    template __global__ void ExecDPSSwBtck3264x<tpUSESEQSCORING>( \
        const uint blkdiagnum, \
        const uint ndbCstrs, const uint ndbCposs, const uint dbxpad, \
        const uint maxnsteps, \
        const float weight4ss, const float weight4rr, const float gapopencost, \
        const float* __restrict__ wrkmemaux, \
        float* __restrict__ tmpdpdiagbuffers, \
        float* __restrict__ tmpdpbotbuffer, \
        char* __restrict__ btckdata);

INSTANTIATE_ExecDPSSwBtck3264x(false);
INSTANTIATE_ExecDPSSwBtck3264x(true);

// -------------------------------------------------------------------------
