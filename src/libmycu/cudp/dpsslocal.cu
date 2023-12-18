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
#include "dpsslocal.cuh"

// #define CUDPSSLOCAL_TESTPRINT 0

// =========================================================================
// device functions for executing dynamic programming with backtracking 
// information using secondary structure information;
// NOTE: Version for CUDP_2DCACHE_DIM_DequalsX: CUDP_2DCACHE_DIM_D==CUDP_2DCACHE_DIM_X!
// NOTE: See COMER2/COTHER source code for a general case!
// blkdiagnum, block diagonal serial number;
// (starting at x=-CUDP_2DCACHE_DIM);
// ndbCstrs, number of references in a chunk;
// ndbCposs, total number of db reference structure positions in a chunk;
// dbxpad, number of padded positions for memory alignment;
// maxnsteps, max number of steps performed for each reference structure 
// during alignment refinement;
//

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
// ExecDPSSLocal3264x: execute dynamic programming for local alignment using
// secondary structure information with 32(64)-fold unrolling along the 
// diagonal of dimension CUDP_2DCACHE_DIM;
// this version fills in the dp matrix with local scores w/o backtracking
// information;
// NOTE: the modulo-2^8 dp matrix is written, hence it is unsuitable for
// local calculations spanning >256 (when scores are within [0,1])!
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary buffers for last calculated diagonal scores;
// tmpdpbotbuffer, temporary buffers for last calculated bottom scores;
// maxscoordsbuf, coordinates (positions) of maximum alignment scores;
// dpscoremtx, rounded dp score matrix;
// 
__global__
void ExecDPSSLocal3264x(
    const uint blkdiagnum,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const float gapcost,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
    char* __restrict__ dpscoremtx)
{
    // blockIdx.x is the oblique block index in the current iteration of 
    // processing anti-diagonal blocks for all query-reference pairs in the chunk;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number;
    const uint dbstrndx = blockIdx.y;
    const uint qryndx = blockIdx.z;
    constexpr int DDIM = CUDP_2DCACHE_DIM_D + 1;//inner dimension for diagonal buffers
    //cache for scores, coordinates, and transformation matrix:
    __shared__ float diag1Cache[nTDPDiagScoreSubsections * DDIM];//cache for scores of the 1st diagonal
    __shared__ float diag2Cache[nTDPDiagScoreSubsections * DDIM];//last (2nd) diagonal
    __shared__ float bottmCache[nTDPDiagScoreSubsections * CUDP_2DCACHE_DIM_X];//bottom scores
    char qrySS;//query secondary structure assignment; reference below:
    __shared__ char rfnSS[CUDP_2DCACHE_DIM_DpX+1];//+1 to avoid bank conflicts
    //NOTE: max scores and their coordinates are not recorded for semi-global alignment!
    //NOTE: comment out the variables:
    //SECTION for dp scores
    __shared__ char dpsCache[CUDP_2DCACHE_DIM_D][CUDP_2DCACHE_DIM_X+1];
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;


    //check convergence first
    if(threadIdx.x == 0) {
        //NOTE: reuse cache to read global convergence flag at sfragfct==0:
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        diag1Cache[0] = wrkmemaux[mloc0 + dbstrndx];
    }

#if (CUDP_2DCACHE_DIM_D <= 32)
    __syncwarp();
#else
    __syncthreads();
#endif

    if(((int)diag1Cache[0]) & (CONVERGED_SCOREDP_bitval|CONVERGED_NOTMPRG_bitval|CONVERGED_LOWTMSC_bitval))
        //all threads in the block exit upon appropriate type of convergence;
        return;


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse cache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(dbstrndx, (int*)diag2Cache);
        GetQueryLenDst(qryndx, (int*)diag2Cache + 2);
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
    // (x,y) is the bottom-left corner (x,y) coordinates for structure dbstrndx
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

    if(0 <= qpos && qpos < qrylen)
        DPLocCacheQrySS(qrySS, qpos + qrydst);
    else
        DPLocInitSS<0/*shift*/,pmvLOOP>(qrySS);

    //db reference structure position corresponding to the oblique block's
    // bottom-left corner in the buffers dc_pm2dvfields_ (score matrix) 
    // plus the offset determined by thread id:
    int dbpos = x + dbstrdst;//going right
    int dblen = ndbCposs + dbxpad;
    //offset (w/o a factor) to the beginning of the data along the y axis wrt query qryndx: 
    int yofff = dblen * qryndx;

    if(0 <= x && x < dbstrlen)
        DPLocCacheRfnSS<0/*shift*/>(rfnSS, dbpos);
    else
        DPLocInitSS<0/*shift*/,0/*VALUE*/>(rfnSS);

    if(0 <= (x+CUDP_2DCACHE_DIM_D) && (x+CUDP_2DCACHE_DIM_D) < dbstrlen)
        //NOTE: blockDim.x==CUDP_2DCACHE_DIM_D
        DPLocCacheRfnSS<CUDP_2DCACHE_DIM_D>(rfnSS, dbpos + CUDP_2DCACHE_DIM_D);
    else
        DPLocInitSS<CUDP_2DCACHE_DIM_D/*shift*/,0/*VALUE*/>(rfnSS);


    //cache TWO DIAGONALS from the previous (along the x axis) oblique block;
    //the structure of tmpdpdiagbuffers is position-specific (1D, along the x axis)
    if(0 <= x-1 && x-1 < dbstrlen) {
        int doffs = nTDPDiagScoreSections * nTDPDiagScoreSubsections * yofff;
        DPLocCacheBuffer<DDIM>(diag1Cache, tmpdpdiagbuffers, dbpos-1, doffs, dblen);
        DPLocCacheBuffer<DDIM,1>(diag2Cache, tmpdpdiagbuffers, dbpos-1, 
                          doffs + dpdssDiag2 * nTDPDiagScoreSubsections * dblen, 
                          dblen);
    }
    else {
        DPLocInitCache<DDIM>(diag1Cache);
        DPLocInitCache<DDIM,1/*shift*/>(diag2Cache);
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
            DPLocInitCache<CUDP_2DCACHE_DIM_X>(bottmCache);
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
        // int btck = dpbtckSTOP;

        if(threadIdx.x+1 == CUDP_2DCACHE_DIM_D) {
            pdiag1[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x+1)] =
                bottmCache[GetBufferNdx<CUDP_2DCACHE_DIM_X>(dpdsssStateMM,i)];
        }

        //NOTE: match score:
        val1 = (float)((qrySS == rfnSS[threadIdx.x+i]) * 2) - 1.0f;

        //MM state update (diagonal direction)
        val1 += pdiag2[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x+1)];
        if(val1 < 0.0f) val1 = 0.0f;
        // if(val1) btck = dpbtckDIAG;

        //sync to update pdiag1; also no read for pdiag2 in this iteration
#if (CUDP_2DCACHE_DIM_D <= 32)
        __syncwarp();
#else
        __syncthreads();
#endif

        //IM state update (left direction)
        val2 = pdiag1[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x)] + gapcost;
        if(val1 < val2) val1 = val2;
        // myhdmaxassgn(val1, val2, btck, (int)dpbtckLEFT);

        //MI state update (up direction)
        val2 = pdiag1[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x+1)] + gapcost;
        if(val1 < val2) val1 = val2;
        // myhdmaxassgn(val1, val2, btck, (int)dpbtckUP);

        //WRITE: write max value
        pdiag2[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x)] = val1;

        //WRITE
        dpsCache[threadIdx.x][i] = (char)(((unsigned int)val1) & 255);//val1%256

        if(threadIdx.x == 0) {
            //WRITE
            //this position is not used by other threads in the current iteration
            bottmCache[GetBufferNdx<CUDP_2DCACHE_DIM_X>(dpdsssStateMM,i)] =
                pdiag2[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x)];
        }

#ifdef CUDPSSLOCAL_TESTPRINT
        if(dbstrndx==CUDPSSLOCAL_TESTPRINT){
            printf(" d=%u(%u) s=%u i%02d/%u (t%02u): len= %d addr= %u SC= %.4f (%d) (yx: %d,%d) "
                    "MM= %.6f    >qSS= %d   dSS= %d\n",
                    blkdiagnum,lastydiagnum,blockIdx.x,i,ilim,threadIdx.x,
                    dbstrlen,dbstrdst,val1,(((unsigned int)val1) & 255), y-threadIdx.x,x+i,
                    pdiag2[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x)],
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


#ifdef CUDPSSLOCAL_TESTPRINT
    if(dbstrndx==CUDPSSLOCAL_TESTPRINT)
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

    //WRITE the bottom of the diagonal block;
    {
        int doffs = nTDPBottomScoreSections * nTDPDiagScoreSubsections * yofff;
        if(0 <= x && x < dbstrlen)
            DPLocWriteBuffer<CUDP_2DCACHE_DIM_X>(
                bottmCache, tmpdpbotbuffer, dbpos, doffs, dblen);
    }

    //WRITE dp scores
    #pragma unroll 8
    for(int i = 0; i < CUDP_2DCACHE_DIM_D; i++) {
        //going upwards
        if(0 <= y-i && y-i < qrylen && 0 <= x+i && x+i < dbstrlen) {
            char bv = dpsCache[i][threadIdx.x];
            //starting position of line i of the oblq. diagonal block in the matrix:
            qpos = (qrydst + (y-i)) * dblen + i;
            dpscoremtx[qpos + dbpos] = bv;
        }
    }
}

// =========================================================================
// -------------------------------------------------------------------------
