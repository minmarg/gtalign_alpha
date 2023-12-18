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
#include "libmycu/custages/transform.cuh"
#include "libmycu/custages/scoring.cuh"
#include "libmycu/custages/fields.cuh"
#include "libmycu/cudp/dpssw_btck.cuh"
#include "libmycu/cudp/dpw_btck.cuh"
#include "dpssw_tfm_btck.cuh"

// =========================================================================

// NOTE: parameters are passed to the device via constant memory and are
// limited to 4 KB
//
// device functions for executing dynamic programming with backtracking
// information;
// NOTE: Version for CUDP_2DCACHE_DIM_DequalsX: CUDP_2DCACHE_DIM_D==CUDP_2DCACHE_DIM_X!
// NOTE: See COMER2/COTHER source code for a general case!
// NOTE: ANCHORRGN, template parameter, anchor region is in use:
// +-------------+--------+--+-
// |     _|______|     |  |__|
// |____|_|      |    _|__|  |
// |    |        |   | |  |  |
// +-------------+--------+--+-
// |        | |  |  | |   |  |
// +-------------+--------+--+-
// NOTE: Regions outside the anchor are not explored,
// NOTE: decreasing computational complexity;
// NOTE: BANDED, template parameter, banded alignment;
// NOTE: GAP0, template parameter, gap open cost ignored (=0);
// blkdiagnum, block diagonal serial number;
// (starting at x=-CUDP_2DCACHE_DIM);
// ndbCstrs, number of references in a chunk;
// ndbCposs, total number of db reference structure positions in a chunk;
// dbxpad, number of padded positions for memory alignment;
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
// ExecDPTFMSSwBtck3264x: execute dynamic programming with secondary
// structure and backtracking information using shared memory and
// 32(64)-fold unrolling along the diagonal of dimension CUDP_2DCACHE_DIM;
// GLOBTFM, template parameter, flag of using the globally best
// transformation matrix;
// GAP0, template parameter, flag of gap open cost 0;
// USESS, template parameter, flag of using secondary structure scoring;
// blkdiagnum, block diagonal serial number;
// ndbCstrs, total number of references in the chunk;
// ndbCposs, total number of reference positions in the chunk;
// dbxpad, padding along the dimension of reference structures;
// maxnsteps, max number of steps performed for each reference structure
// during alignment refinement;
// stepnumber, step number corresponding to the slot to read transformation
// matrix from;
// NOTE: memory pointers should be aligned!
// tfmmem (wrkmemtmibest), transformation matrix address space;
// /* wrkmemaux */, working memory for global flags across kernels;
// tmpdpdiagbuffers, temporary buffers for last calculated diagonal scores;
// tmpdpbotbuffer, temporary buffers for last calculated bottom scores;
// maxscoordsbuf, coordinates (positions) of maximum alignment scores;
// btckdata, backtracking information data;
//
template<bool GLOBTFM, bool GAP0, bool USESS, int D02IND>
__global__
void ExecDPTFMSSwBtck3264x(
    const uint blkdiagnum,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const uint stepnumber,
    const float ssweight,
    const float gapopencost,
    const float* __restrict__ tfmmem,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
//     uint* __restrict__ maxscoordsbuf,
    char* __restrict__ btckdata)
{
    // blockIdx.x is the oblique block index in the current iteration of
    // processing anti-diagonal blocks for all query-reference pairs in the chunk;
    uint dbstrndx = blockIdx.y;//reference serial number;
    uint qryndx = blockIdx.z;//query serial number
    constexpr int DDIM = CUDP_2DCACHE_DIM_D + 1;//inner dimension for diagonal buffers
    //cache for scores, coordinates, and transformation matrix:
    __shared__ float diag1Cache[nTDPDiagScoreSubsections * DDIM];//cache for scores of the 1st diagonal
    __shared__ float diag2Cache[nTDPDiagScoreSubsections * DDIM];//last (2nd) diagonal
    __shared__ float bottmCache[nTDPDiagScoreSubsections * CUDP_2DCACHE_DIM_X];//bottom scores
    //__shared__ float qryCoords[pmv2DNoElems * CUDP_2DCACHE_DIM_D];
    float qry2DX, qry2DY, qry2DZ;
    __shared__ float rfnCoords[pmv2DNoElems * CUDP_2DCACHE_DIM_DpX];
    __shared__ float tfmCache[nTTranformMatrix];
    char qrySS;//query secondary structure assignment; reference below:
    __shared__ char rfnSS[CUDP_2DCACHE_DIM_DpX+1];//+1 to avoid bank conflicts
    //NOTE: max scores and their coordinates are not recorded for semi-global alignment!
    //NOTE: comment out the variables:
    ///float maxscCache;//maximum scores of the last processed diagonal
    ///uint maxscCoords = 0;//coordinates of the maximum alignment score maxscCache
    //SECTION for backtracking information
    __shared__ char btckCache[CUDP_2DCACHE_DIM_D][CUDP_2DCACHE_DIM_X+1];
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;


    //check SCORE convergence first
    if(threadIdx.x == 0) {
        uint mloc = ((qryndx * maxnsteps + 0/*sfragfct*/) * nTAuxWorkingMemoryVars) * ndbCstrs;
        rfnCoords[0] = wrkmemaux[mloc + tawmvConverged * ndbCstrs + dbstrndx];
    }

    __syncthreads();

    if(((int)(rfnCoords[0])) & (CONVERGED_LOWTMSC_bitval))
        //(the termination flag for this pair is set);
        //all threads in the block exit;
        return;


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse ccmCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(dbstrndx, (int*)tfmCache);
        GetQueryLenDst(qryndx, (int*)tfmCache + 2);
    }

#if (CUDP_2DCACHE_DIM_D <= 32)
    __syncwarp();
#else
    __syncthreads();
#endif

    //NOTE: no bank conflict when two threads from the same warp access the same address;
    dbstrlen = ((int*)tfmCache)[0]; dbstrdst = ((int*)tfmCache)[1];
    qrylen = ((int*)tfmCache)[2]; qrydst = ((int*)tfmCache)[3];

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
        //block does not participate: out of boundaries
        return;


    //READ DATA
    int qpos = y - threadIdx.x;//going upwards
    //x is now the position this thread will process
    x += threadIdx.x;

    //read query coordinates
    if(0 <= qpos && qpos < qrylen)
        DPLocCacheQryCoords(qry2DX, qry2DY, qry2DZ, qpos + qrydst);
        //DPLocCacheQryCoords(qryCoords, qpos + qrydst);
    else
        DPLocInitCoords<0/*shift*/,CUDP_DEFCOORD_QRY>(qry2DX, qry2DY, qry2DZ);
        //DPLocInitCoords<0/*shift*/,CUDP_DEFCOORD_QRY>(qryCoords);

    //read ss information
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
    //offset (w/o a factor) to the beginning of the data along the y axis wrt query qryndx:
    int yofff = dblen * qryndx;

    //read reference coordinates
    if(0 <= x && x < dbstrlen)
        DPLocCacheRfnCoords<0/*shift*/>(rfnCoords, dbpos);
    else DPLocInitCoords<0/*shift*/,CUDP_DEFCOORD_RFN>(rfnCoords);

    if(0 <= (x+CUDP_2DCACHE_DIM_D) && (x+CUDP_2DCACHE_DIM_D) < dbstrlen)
        //NOTE: blockDim.x==CUDP_2DCACHE_DIM_D
        DPLocCacheRfnCoords<CUDP_2DCACHE_DIM_D>(rfnCoords, dbpos + CUDP_2DCACHE_DIM_D);
    else DPLocInitCoords<CUDP_2DCACHE_DIM_D/*shift*/,CUDP_DEFCOORD_RFN>(rfnCoords);

    if(USESS) {//read ss information
        if(0 <= x && x < dbstrlen)
            DPLocCacheRfnSS<0/*shift*/>(rfnSS, dbpos);
        else DPLocInitSS<0/*shift*/,0/*VALUE*/>(rfnSS);

        if(0 <= (x+CUDP_2DCACHE_DIM_D) && (x+CUDP_2DCACHE_DIM_D) < dbstrlen)
            //NOTE: blockDim.x==CUDP_2DCACHE_DIM_D
            DPLocCacheRfnSS<CUDP_2DCACHE_DIM_D>(rfnSS, dbpos + CUDP_2DCACHE_DIM_D);
        else DPLocInitSS<CUDP_2DCACHE_DIM_D/*shift*/,0/*VALUE*/>(rfnSS);
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
        DPLocInitCache<DDIM>(diag1Cache);
        DPLocInitCache<DDIM,1/*shift*/>(diag2Cache);
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
            DPLocInitCache<CUDP_2DCACHE_DIM_X>(bottmCache);
        }
    }


    //READ TRANSFORMATION MATRIX for query-reference pair
    if(threadIdx.x < nTTranformMatrix) {
        uint mloc0;
        if(GLOBTFM) {
            //globally best transformation matrix for a pair:
            mloc0 = (qryndx * ndbCstrs + dbstrndx) * nTTranformMatrix;
        } else {
            //iteration-best transformation matrix written at position 0;
            //alternatively, transformation matrix can be written at position stepnumber:
            mloc0 = ((qryndx * maxnsteps + stepnumber) * ndbCstrs + dbstrndx) * nTTranformMatrix;
        }
        tfmCache[threadIdx.x] = tfmmem/* wrkmemtmibest */[mloc0 + threadIdx.x];
    }


// #if (CUDP_2DCACHE_DIM_D <= 32)
//     __syncwarp();
// #else
    //NOTE: surprisingly, __syncthreads here saves a lot of
    //NOTE: registers for later architectures
    __syncthreads();
// #endif


    //transform the query fragment read
    if(0 <= qpos && qpos < qrylen)
        transform_point(tfmCache, qry2DX, qry2DY, qry2DZ);
        //transform_point(tfmCache,
        //    qryCoords[GetCoordsNdx(pmv2DX,threadIdx.x)],
        //    qryCoords[GetCoordsNdx(pmv2DY,threadIdx.x)],
        //    qryCoords[GetCoordsNdx(pmv2DZ,threadIdx.x)]);


    float *pdiag1 = diag1Cache;
    float *pdiag2 = diag2Cache;
    float d02;
    if(D02IND == D02IND_SEARCH) d02 = GetD02(qrylen, dbstrlen);
    else if(D02IND == D02IND_DPSCAN) d02 = GetD02_dpscan(qrylen, dbstrlen);

    //start calculations for this position with 32x/64x unrolling
    //NOTE: sync inside: do not branch;
    for(int i = 0; i < ilim/*CUDP_2DCACHE_DIM_X*/; i++)
    {
        float val1 = 0.0f, val2;//, left, up;
        float rfn2DX = rfnCoords[GetCoordsNdx(pmv2DX, threadIdx.x + i)];
        int btck;

        if(threadIdx.x+1 == CUDP_2DCACHE_DIM_D) {
            pdiag1[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x+1)] =
                bottmCache[GetBufferNdx<CUDP_2DCACHE_DIM_X>(dpdsssStateMM,i)];
        }

        if(qry2DX < CUDP_DEFCOORD_QRY_cmp && CUDP_DEFCOORD_RFN_cmp < rfn2DX) {
            val1 = distance2(
                qry2DX, qry2DY, qry2DZ,
                rfn2DX,
                rfnCoords[GetCoordsNdx(pmv2DY, threadIdx.x + i)],
                rfnCoords[GetCoordsNdx(pmv2DZ, threadIdx.x + i)]
            );
            val1 = GetPairScore(d02, val1);//superposition score
        }
        if(USESS) val1 += (float)(qrySS == rfnSS[threadIdx.x+i]) * ssweight;
        // if(USESS) val1 += (float)(qrySS == rfnSS[threadIdx.x+i] ||
        //         (isloop(qrySS) && isloop(rfnSS[threadIdx.x+i]))) * ssweight;

        //NOTE: TRICK to implement a special case of DP with affine gap cost scheme:
        //NOTE: gap extension cost is 0;
        //NOTE: match scores are always non-negative; hence, an alignemnt score too;
        //NOTE: save NEGATED match scores to indicate diagonal direction in alignment;
        //NOTE: when gaps lead to negative scores, match scores will always be preferred;

        //MM state update (diagonal direction)
        val1 +=
            GAP0? pdiag2[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x+1)]:
            fabsf(pdiag2[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x+1)]);
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
        val2 = /*left = */pdiag1[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x)];
        if(!GAP0 && val2 < 0.0f) val2 = gapopencost - val2;
        myhdmaxassgn(val1, val2, btck, (int)dpbtckLEFT);

        //MI state update (up direction)
        val2 = /*up = */pdiag1[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x+1)];
        if(!GAP0 && val2 < 0.0f) val2 = gapopencost - val2;
        myhdmaxassgn(val1, val2, btck, (int)dpbtckUP);

        //WRITE: write max value
        pdiag2[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x)] =
            (GAP0 || btck != dpbtckDIAG)? val1: -val1;

        //WRITE
        btckCache[threadIdx.x][i] = btck;

        if(threadIdx.x == 0) {
            //WRITE
            //this position is not used by other threads in the current iteration
            bottmCache[GetBufferNdx<CUDP_2DCACHE_DIM_X>(dpdsssStateMM,i)] =
                pdiag2[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x)];
        }

        myhdswap(pdiag1, pdiag2);

        //sync for updates
// #if (CUDP_2DCACHE_DIM_D <= 32)
//         __syncwarp();
// #else
        __syncthreads();
// #endif
    }


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
    for(int i = 0; i < CUDP_2DCACHE_DIM_D; i++) {
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
#define INSTANTIATE_ExecDPTFMSSwBtck3264x(tpGLOBTFM,tpGAP0,tpUSESS,tpD02IND) \
    template \
    __global__ void ExecDPTFMSSwBtck3264x<tpGLOBTFM,tpGAP0,tpUSESS,tpD02IND>( \
        const uint blkdiagnum, \
        const uint ndbCstrs, const uint ndbCposs, const uint dbxpad, \
        const uint maxnsteps, const uint stepnumber, \
        const float ssweight, const float gapopencost, \
        const float* __restrict__ tfmmem, \
        const float* __restrict__ /* wrkmemaux */, \
        float* __restrict__ tmpdpdiagbuffers, \
        float* __restrict__ tmpdpbotbuffer, \
        char* __restrict__ btckdata);

INSTANTIATE_ExecDPTFMSSwBtck3264x(true,false,true,D02IND_DPSCAN);
INSTANTIATE_ExecDPTFMSSwBtck3264x(true,true,false,D02IND_SEARCH);

// -------------------------------------------------------------------------
