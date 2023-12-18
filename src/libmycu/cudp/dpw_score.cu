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
#include "libmycu/cudp/dpw_btck.cuh"
#include "dpw_score.cuh"

// #define CUDP_INIT_BTCK_TESTPRINT 0 //3024//4899

// =========================================================================

// NOTE: parameters are passed to the device via constant memory and are 
// limited to 4 KB
// 
// device functions for executing dynamic programming in linear space;
// NOTE: Version for CUDP_SWFT_2DCACHE_DIM_DequalsX: CUDP_SWFT_2DCACHE_DIM_D==CUDP_SWFT_2DCACHE_DIM_X!
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
// (starting at x=-CUDP_SWFT_2DCACHE_DIM);
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
// ExecDPScore3264x: kernel for executing dynamic programming to calculate 
// max score completely in linear space (without backtracking) shared 
// memory and 32(64)-fold unrolling along the diagonal of dimension 
// CUDP_SWFT_2DCACHE_DIM_D;
// maxnsteps, max number of steps performed for each reference structure 
// during alignment refinement;
// NOTE: memory pointers should be aligned!
// wrkmemtm, memory of iteration-specific transformation matrices;
// wrkmemaux, auxiliary working memory;
// tmpdpdiagbuffers, temporary buffers for last calculated diagonal scores;
// tmpdpbotbuffer, temporary buffers for last calculated bottom scores;
// maxscoordsbuf, coordinates (positions) of maximum alignment scores;
// 
template<bool ANCHORRGN, bool BANDED, bool GAP0, bool CHECKCONV>
__global__
void ExecDPScore3264x(
    const uint blkdiagnum,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const float gapopencost,
    const float* __restrict__ wrkmemtm,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer)
{
    // blockIdx.x is the oblique block index in the current iteration of 
    // processing anti-diagonal blocks for all query-reference pairs in the chunk;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number TIMES fragment factor;
    constexpr int DDIM = CUDP_SWFT_2DCACHE_DIM_D + 1;//inner dimension for diagonal buffers
    //cache for scores, coordinates, and transformation matrix:
    __shared__ float diag1Cache[nTDPDiagScoreSubsections * DDIM];//cache for scores of the 1st diagonal
    __shared__ float diag2Cache[nTDPDiagScoreSubsections * DDIM];//last (2nd) diagonal
    __shared__ float bottmCache[nTDPDiagScoreSubsections * CUDP_SWFT_2DCACHE_DIM_X];//bottom scores
    //__shared__ float qryCoords[pmv2DNoElems * CUDP_SWFT_2DCACHE_DIM_D];
    float qry2DX, qry2DY, qry2DZ;
    __shared__ float rfnCoords[pmv2DNoElems * CUDP_SWFT_2DCACHE_DIM_DpX];
    __shared__ float tfmCache[nTTranformMatrix];
    ///float maxscCache;//maximum scores of the last processed diagonal
    ///uint maxscCoords = 0;//coordinates of the maximum alignment score maxscCache
    ///cell coordinates (positions) of matched atom pairs:
    ///__shared__ uint maxscCoords[CUDP_SWFT_2DCACHE_DIM_DpX+1];
    const uint dbstrndx = blockIdx.y;
    const uint sfragfct = blockIdx.z / nqystrs;//fragment factor
    const uint qryndx = blockIdx.z - sfragfct * nqystrs;//query serial number
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;
    int qrypos = 0, rfnpos = 0;
    int sfragndx = 0, sfragpos = 0;
    int fraglen = 0;


    if(CHECKCONV) {
        if(threadIdx.x == 0) {
            //NOTE: reuse cache to read convergence flag at both 0 and sfragfct:
            uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
            rfnCoords[0] = wrkmemaux[mloc0 + dbstrndx];
        }

#if (CUDP_SWFT_2DCACHE_DIM_D <= 32)
        if(threadIdx.x == 0) {//same warp
#else
        if(threadIdx.x == 32) {//next warp
#endif
            uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
            rfnCoords[1] = wrkmemaux[mloc + dbstrndx];
        }

        __syncthreads();

        if((((int)(rfnCoords[0])) & (CONVERGED_LOWTMSC_bitval)) || (rfnCoords[1]))
            //all threads in the block exit upon any type of convergence;
            return;
    }


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse ccmCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(dbstrndx, (int*)tfmCache);
        GetQueryLenDst(qryndx, (int*)tfmCache + 2);
    }

#if (CUDP_SWFT_2DCACHE_DIM_D <= 32)
    __syncwarp();
#else
    __syncthreads();
#endif

    //NOTE: no bank conflict when two threads from the same warp access the same address;
    dbstrlen = ((int*)tfmCache)[0]; dbstrdst = ((int*)tfmCache)[1];
    qrylen = ((int*)tfmCache)[2]; qrydst = ((int*)tfmCache)[3];

#if (CUDP_SWFT_2DCACHE_DIM_D <= 32)
    __syncwarp();
#else
    __syncthreads();
#endif


    //lastydiagnum, last block diagonal serial number along y axis:
    //each division separates a number of diagonals (nsepds);
    constexpr int nsepds = 2;//(float)CUDP_SWFT_2DCACHE_DIM_D/(float)CUDP_SWFT_2DCACHE_DIM_X + 1.0f;
    //the number of the last diagonal starting at x=-CUDP_SWFT_2DCACHE_DIM_D
    ///int nintdivs = (qrylen-1)>>CUDP_SWFT_2DCACHE_DIM_D_LOG2;//(qrylen-1)/CUDP_SWFT_2DCACHE_DIM_D;
    ///uint lastydiagnum = nsepds * nintdivs + 1 - 1;//-1 for zero-based indices;
    uint lastydiagnum = ((qrylen-1) >> CUDP_SWFT_2DCACHE_DIM_D_LOG2) * nsepds;


    // blockIdx.x is block serial number s within diagonal blkdiagnum;
    // (x,y) is the bottom-left corner (x,y) coordinates for structure dbstrndx
    int x, y;
    if( blkdiagnum <= lastydiagnum) {
        //x=-!(d%2)w+2ws; y=dw/2+w-sw -1 (-1, zero-based indices); [when w==b]
        //(b, block's length; w, block's width)
        x = (2*blockIdx.x - (!(blkdiagnum & 1))) * CUDP_SWFT_2DCACHE_DIM_D;
        y = ((blkdiagnum>>1) + 1 - blockIdx.x) * CUDP_SWFT_2DCACHE_DIM_D - 1;
    } else {
        //x=-w+(d-d_l)w+2ws; y=dw/2+w-sw -1; [when w==b]
        x = (2*blockIdx.x + (blkdiagnum-lastydiagnum-1)) * CUDP_SWFT_2DCACHE_DIM_D;
        y = ((lastydiagnum>>1) + 1 - blockIdx.x) * CUDP_SWFT_2DCACHE_DIM_D - 1;
    }


    //number of iterations for this block to perform;
    int ilim = GetMaqxNoIterations(x, y, qrylen, dbstrlen, CUDP_SWFT_2DCACHE_DIM_X);

    if(y < 0 || qrylen <= (y+1 - CUDP_SWFT_2DCACHE_DIM_D) || 
       dbstrlen <= x /*+ CUDP_SWFT_2DCACHE_DIM_DpX */ ||
       ilim < 1)
        //block does not participate: out of boundaries
        return;


    if(ANCHORRGN) {
        if(threadIdx.x == tawmvQRYpos || threadIdx.x == tawmvRFNpos ||
        threadIdx.x == tawmvSubFragNdx || threadIdx.x == tawmvSubFragPos)
        {
            //NOTE: reuse cache to contain query and reference positions and other fields
            //structure-specific-formatted data: 4 uncoalesced reads
            uint mloc = ((qryndx * maxnsteps + /*0*/sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
            bottmCache[threadIdx.x] = wrkmemaux[mloc + threadIdx.x * ndbCstrs + dbstrndx];
        }

#if (CUDP_SWFT_2DCACHE_DIM_D <= 32)
        __syncwarp();
#else
        __syncthreads();
#endif

        qrypos = bottmCache[tawmvQRYpos]; rfnpos = bottmCache[tawmvRFNpos];
        sfragndx = bottmCache[tawmvSubFragNdx]; sfragpos = bottmCache[tawmvSubFragPos];

#if (CUDP_SWFT_2DCACHE_DIM_D <= 32)
        __syncwarp();
#else
        __syncthreads();
#endif

        fraglen = GetFragLength(qrylen, dbstrlen, qrypos, rfnpos, sfragndx);
        if(fraglen < 1)
            //fraglen was saved to be valid, but verify anyway
            return;

        qrypos += sfragpos; rfnpos += sfragpos;
    }

    if(CellXYInvalidLowerArea<ANCHORRGN,BANDED>(
            //oblique block's upper-right corner
            x + CUDP_SWFT_2DCACHE_DIM_DpX-2, y+1 - CUDP_SWFT_2DCACHE_DIM_D,
            qrylen, dbstrlen, qrypos, rfnpos, fraglen) ||
       CellXYInvalidUpperArea<ANCHORRGN,BANDED>(
            x, y,//oblique block's bottom-left corner
            qrylen, dbstrlen, qrypos, rfnpos, fraglen)
    )   //unexplored region: all threads exit
        return;


    //READ COORDINATES
    int qpos = y - threadIdx.x;//going upwards
    //x is now the position this thread will process
    x += threadIdx.x;

    if(0 <= qpos && qpos < qrylen &&
       !CellXYInvalidLowerArea<ANCHORRGN,BANDED>(
           x + CUDP_SWFT_2DCACHE_DIM_X-1, qpos,//oblique block's right edge
           qrylen, dbstrlen, qrypos, rfnpos, fraglen) &&
       !CellXYInvalidUpperArea<ANCHORRGN,BANDED>(
           x, qpos,//oblique block's left edge
           qrylen, dbstrlen, qrypos, rfnpos, fraglen)
    )
        DPLocCacheQryCoords(qry2DX, qry2DY, qry2DZ, qpos + qrydst);
        //DPLocCacheQryCoords(qryCoords, qpos + qrydst);
    else
        DPLocInitCoords<0/*shift*/,CUDP_DEFCOORD_QRY>(qry2DX, qry2DY, qry2DZ);
        //DPLocInitCoords<0/*shift*/,CUDP_DEFCOORD_QRY>(qryCoords);

    //db reference structure position corresponding to the oblique block's
    // bottom-left corner in the buffers dc_pm2dvfields_ (score matrix) 
    // plus the offset determined by thread id:
    int dbpos = x + dbstrdst;//going right
    int dblen = ndbCposs + dbxpad;
    //offset (w/o a factor) to the beginning of the data along the y axis 
    // wrt query qryndx: 
    //int yofff = dblen * qryndx;
    int yofff = (qryndx * maxnsteps + sfragfct) * dblen;

    if(0 <= x && x < dbstrlen &&
       !CellXYInvalidLowerArea<ANCHORRGN,BANDED>(
           x, qpos,//oblique block's left (bottom) edge
           qrylen, dbstrlen, qrypos, rfnpos, fraglen) &&
       !CellXYInvalidUpperArea<ANCHORRGN,BANDED>(
           x, y,//oblique block's bottom edge
           qrylen, dbstrlen, qrypos, rfnpos, fraglen)
    )
        DPLocCacheRfnCoords<0/*shift*/>(rfnCoords, dbpos);
    else
        DPLocInitCoords<0/*shift*/,CUDP_DEFCOORD_RFN>(rfnCoords);

    if(0 <= (x+CUDP_SWFT_2DCACHE_DIM_D) && (x+CUDP_SWFT_2DCACHE_DIM_D) < dbstrlen &&
       !CellXYInvalidLowerArea<ANCHORRGN,BANDED>(
           x + CUDP_SWFT_2DCACHE_DIM_D, y - CUDP_SWFT_2DCACHE_DIM_D+1,//oblique block's top edge
           qrylen, dbstrlen, qrypos, rfnpos, fraglen) &&
       !CellXYInvalidUpperArea<ANCHORRGN,BANDED>(
           x + CUDP_SWFT_2DCACHE_DIM_D, qpos,//oblique block's right+1 (top) edge
           qrylen, dbstrlen, qrypos, rfnpos, fraglen)
    )
        //NOTE: blockDim.x==CUDP_SWFT_2DCACHE_DIM_D
        DPLocCacheRfnCoords<CUDP_SWFT_2DCACHE_DIM_D>(rfnCoords, dbpos + CUDP_SWFT_2DCACHE_DIM_D);
    else
        DPLocInitCoords<CUDP_SWFT_2DCACHE_DIM_D/*shift*/,CUDP_DEFCOORD_RFN>(rfnCoords);


    //cache TWO DIAGONALS from the previous (along the x axis) oblique block;
    //the structure of tmpdpdiagbuffers is position-specific (1D, along the x axis)
    if(0 <= x-1 && x-1 < dbstrlen &&
       !CellXYInvalidLowerArea<ANCHORRGN,BANDED>(
           x-1, qpos,//oblique block's left-1 edge
           qrylen, dbstrlen, qrypos, rfnpos, fraglen) &&
       !CellXYInvalidUpperArea<ANCHORRGN,BANDED>(
           x-1, qpos,//oblique block's left-1 edge
           qrylen, dbstrlen, qrypos, rfnpos, fraglen)
    ) {
        int doffs = nTDPDiagScoreSections * nTDPDiagScoreSubsections * yofff;
        DPLocCacheBuffer<DDIM>(diag1Cache, tmpdpdiagbuffers, dbpos-1, doffs, dblen);
        DPLocCacheBuffer<DDIM,1>(diag2Cache, tmpdpdiagbuffers, dbpos-1, 
                          doffs + dpdssDiag2 * nTDPDiagScoreSubsections * dblen, 
                          dblen);
        //cache the buffer of maximum scores here
        ///doffs += dpdssDiagM * nTDPDiagScoreSubsections * dblen;
        ///maxscCache = tmpdpdiagbuffers[doffs + dbpos - 1];
        ///cell coordinates corresponding to the right edge of the left oblique data block:
        ///maxscCoords[threadIdx.x] = maxscoordsbuf[yofff + dbpos - 1];
    }
    else {
        DPLocInitCache<DDIM>(diag1Cache);
        DPLocInitCache<DDIM,1/*shift*/>(diag2Cache);
        ///maxscCache = 0.0f;
        ///maxscCoords[threadIdx.x] = DPCELLSTOP;//CombineCoords(...);//cell (-1,-1)
    }

    //cache the bottom of the upper oblique blocks;
    //the structure of tmpdpbotbuffer is position-specific (1D, along x-axis)
    {
        int doffs = nTDPBottomScoreSections * nTDPDiagScoreSubsections * yofff;
        if(CUDP_SWFT_2DCACHE_DIM_D <= y && 
           0 <= x+CUDP_SWFT_2DCACHE_DIM_D-1 && x+CUDP_SWFT_2DCACHE_DIM_D-1 < dbstrlen &&
           !CellXYInvalidLowerArea<ANCHORRGN,BANDED>(
                x + CUDP_SWFT_2DCACHE_DIM_D-1, y - CUDP_SWFT_2DCACHE_DIM_D,//oblique block's top-1 edge
                qrylen, dbstrlen, qrypos, rfnpos, fraglen) &&
           !CellXYInvalidUpperArea<ANCHORRGN,BANDED>(
                x + CUDP_SWFT_2DCACHE_DIM_D-1, y - CUDP_SWFT_2DCACHE_DIM_D,//oblique block's top-1 edge
                qrylen, dbstrlen, qrypos, rfnpos, fraglen)
        )
        {
            DPLocCacheBuffer<CUDP_SWFT_2DCACHE_DIM_X>( 
                bottmCache, tmpdpbotbuffer, dbpos+CUDP_SWFT_2DCACHE_DIM_D-1, doffs, dblen);
            ///cell coordinates corresponding to the bottom edge - 1 of the above oblique block:
            ///maxscCoords[threadIdx.x + CUDP_SWFT_2DCACHE_DIM_D] =
            ///    maxscoordsbuf[yofff + dbpos+CUDP_SWFT_2DCACHE_DIM_D-1];
        }
        else {
            DPLocInitCache<CUDP_SWFT_2DCACHE_DIM_X>(bottmCache);
            ///maxscCoords[threadIdx.x + CUDP_SWFT_2DCACHE_DIM_D] = DPCELLSTOP;//CombineCoords(...);//cell (-1,-1)
        }
    }


    //READ TRANSFORMATION MATRIX for query-reference pair
    if(threadIdx.x < nTTranformMatrix) {
        //iteration-best transformation matrix written at position 0;
        //iteration-specific transformation matrix written at position sfragfct;
        uint mloc = ((qryndx * maxnsteps + /*0*/sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix;
        tfmCache[threadIdx.x] = wrkmemtm[mloc + threadIdx.x];
    }


// #if (CUDP_SWFT_2DCACHE_DIM_D <= 32)
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
    float d02 = GetD02/*_dpscan*/(qrylen, dbstrlen);

    //start calculations for this position with 32x/64x unrolling
    //NOTE: sync inside: do not branch;
    for(int i = 0; i < ilim/*CUDP_SWFT_2DCACHE_DIM_X*/; i++)
    {
        float val1 = 0.0f, val2;//, left, up;
        float rfn2DX = rfnCoords[GetCoordsNdx(pmv2DX, threadIdx.x + i)];
        ///previous max score coordinates; sync below before its use;
        ///uint leftmaxsc = maxscCoords[threadIdx.x+i];
        int btck;

        if(threadIdx.x+1 == CUDP_SWFT_2DCACHE_DIM_D) {
            pdiag1[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x+1)] =
                bottmCache[GetBufferNdx<CUDP_SWFT_2DCACHE_DIM_X>(dpdsssStateMM,i)];
        }

        if(qry2DX < CUDP_DEFCOORD_QRY_cmp && CUDP_DEFCOORD_RFN_cmp < rfn2DX) {
            val1 = distance2(
                qry2DX, qry2DY, qry2DZ,
                rfn2DX,
                rfnCoords[GetCoordsNdx(pmv2DY, threadIdx.x + i)],
                rfnCoords[GetCoordsNdx(pmv2DZ, threadIdx.x + i)]
            );
            //val1 = distance2(
            //    qryCoords[GetCoordsNdx(pmv2DX, threadIdx.x)],
            //    qryCoords[GetCoordsNdx(pmv2DY, threadIdx.x)],
            //    qryCoords[GetCoordsNdx(pmv2DZ, threadIdx.x)],//
            //    rfnCoords[GetCoordsNdx(pmv2DX, threadIdx.x + i)],
            //    rfnCoords[GetCoordsNdx(pmv2DY, threadIdx.x + i)],
            //    rfnCoords[GetCoordsNdx(pmv2DZ, threadIdx.x + i)]
            //);
            val1 = GetPairScore(d02, val1);//score
        }

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
        ///dpmaxandcoords(maxscCache, val1, maxscCoords, x+i, y-threadIdx.x);
        ///if(maxscCache < val1) maxscCache = val1;

        //sync to update pdiag1; also no read for pdiag2 in this iteration
#if (CUDP_SWFT_2DCACHE_DIM_D <= 32)
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

        ///WRITE the wavefront of max match score coordinates;
        ///maxscCoords[0] is the cell from the left data block, hence +1:
        ///if(btck == dpbtckDIAG)
        ///    maxscCoords[threadIdx.x+i+1] = CombineCoords((uint)(x+i), (uint)(qpos));
        ///else if(btck == dpbtckLEFT) 
        ///    maxscCoords[threadIdx.x+i+1] = leftmaxsc;
        ///else if(btck == dpbtckUP);//maxscCoords[threadIdx.x+i+1] stays the same (same position along x-axis)

        if(threadIdx.x == 0) {
            //WRITE
            //this position is not used by other threads in the current iteration
            bottmCache[GetBufferNdx<CUDP_SWFT_2DCACHE_DIM_X>(dpdsssStateMM,i)] =
                pdiag2[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x)];
        }

#ifdef CUDP_INIT_BTCK_TESTPRINT
        if(dbstrndx==CUDP_INIT_BTCK_TESTPRINT){
            printf(" d=%u(%u) s=%u i%02d/%u (t%02u): len= %d addr= %u SC= %.4f (yx: %d,%d) "
                    "MM= %.6f  "//MAX= %.6f  "//COORD= %08x  "//BTCK= %d\n"
                    "  >qX= %.4f qY= %.4f qZ= %.4f   dX= %.4f dY= %.4f dZ= %.4f\n",
                    blkdiagnum,lastydiagnum,blockIdx.x,i,ilim,threadIdx.x,
                    dbstrlen,dbstrdst,val1, y-threadIdx.x,x+i,
                    pdiag2[GetBufferNdx<DDIM>(dpdsssStateMM,threadIdx.x)],
                    //maxscCache, //maxscCoords[threadIdx.x+i+1],// btck,
                    qry2DX, qry2DY, qry2DZ,
                    //qryCoords[GetCoordsNdx(pmv2DX, threadIdx.x)],
                    //qryCoords[GetCoordsNdx(pmv2DY, threadIdx.x)],
                    //qryCoords[GetCoordsNdx(pmv2DZ, threadIdx.x)],//
                    rfnCoords[GetCoordsNdx(pmv2DX, threadIdx.x+i)],
                    rfnCoords[GetCoordsNdx(pmv2DY, threadIdx.x+i)],
                    rfnCoords[GetCoordsNdx(pmv2DZ, threadIdx.x+i)]
            );
            for(size_t _k=0;_k<1000000000UL;_k++)clock();
            for(size_t _k=0;_k<1000000000UL;_k++)clock();
        }
#endif

        myhdswap(pdiag1, pdiag2);

        //sync for updates
// #if (CUDP_SWFT_2DCACHE_DIM_D <= 32)
//         __syncwarp();
// #else
        __syncthreads();
// #endif
    }


#ifdef CUDP_INIT_BTCK_TESTPRINT
    if(dbstrndx==CUDP_INIT_BTCK_TESTPRINT)
        printf(" >>> d=%u(%u) s=%u (t%02u): len= %d wrt= %d xpos= %d\n", 
        //printf(" >>> d=%u(%u) s=%u (t%02u): len= %d MAX= %.6f COORD= %08x (yx %u %u) wrt= %d xpos= %d\n", 
            blkdiagnum,lastydiagnum,blockIdx.x,threadIdx.x, dbstrlen,
            //maxscCache, //maxscCoords[threadIdx.x+CUDP_SWFT_2DCACHE_DIM_X-1],
            //GetCoordY(maxscCoords[threadIdx.x+CUDP_SWFT_2DCACHE_DIM_X-1]),
            //GetCoordX(maxscCoords[threadIdx.x+CUDP_SWFT_2DCACHE_DIM_X-1]),
            x+CUDP_SWFT_2DCACHE_DIM_X-1<dbstrlen, x+CUDP_SWFT_2DCACHE_DIM_X-1);
#endif


    //write the result for next-iteration blocks;
    //{{before writing the diagonals, adjust the position so that
    //the value of the last cell is always written at the last position
    //(dbstrlen-1) for the last block irrespective of its placement 
    //within the DP matrix; this ensures that the last cell
    //contains the max score; (ilim>0 by definition; beginning)
    int shft = CUDP_SWFT_2DCACHE_DIM_X-1;
    if(qrylen <= y+1 && ilim < CUDP_SWFT_2DCACHE_DIM_X) shft = ilim - 1;
    //}}

    //WRITE two diagonals;
    if(0 <= x + shft && x + shft < dbstrlen &&
       !CellXYInvalidLowerArea<ANCHORRGN,BANDED>(
           x + shft, qpos,//oblique block's right edge
           qrylen, dbstrlen, qrypos, rfnpos, fraglen) &&
       !CellXYInvalidUpperArea<ANCHORRGN,BANDED>(
           x + shft, qpos,//oblique block's right edge
           qrylen, dbstrlen, qrypos, rfnpos, fraglen)
    ) {
        int doffs = nTDPDiagScoreSections * nTDPDiagScoreSubsections * yofff;
        DPLocWriteBuffer<DDIM>(pdiag1, tmpdpdiagbuffers, dbpos + shft, doffs,
                          dblen);
        DPLocWriteBuffer<DDIM,1>(pdiag2, tmpdpdiagbuffers, dbpos + shft, 
                          doffs + dpdssDiag2 * nTDPDiagScoreSubsections * dblen, 
                          dblen);
    }

    //NOTE: max scores are not recorded for semi-global alignment;
    ///int ndx = CUDP_SWFT_2DCACHE_DIM_X-1;
    ////top-left x coordinate of the oblique block:
    ///int xtl = x - threadIdx.x + blockDim.x - 1;
    ///if(xtl >= dbstrlen) ndx = 0;
    ///else if(xtl + ndx >= dbstrlen) ndx = dbstrlen - xtl - 1;

    //NOTE: max scores are not recorded for semi-global alignment;
    //WRITE the buffer of maximum scores
    ///if(0 <= x + ndx && x + ndx < dbstrlen) {
    ///    int doffs = nTDPDiagScoreSections * nTDPDiagScoreSubsections * yofff;
    ///    doffs += dpdssDiagM * nTDPDiagScoreSubsections * dblen;
    ///    if(CUDP_SWFT_2DCACHE_DIM_D <= y) {
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
        if(0 <= x && x < dbstrlen &&
           !CellXYInvalidLowerArea<ANCHORRGN,BANDED>(
               x, y,//oblique block's bottom edge
               qrylen, dbstrlen, qrypos, rfnpos, fraglen) &&
           !CellXYInvalidUpperArea<ANCHORRGN,BANDED>(
               x, y,//oblique block's bottom edge
               qrylen, dbstrlen, qrypos, rfnpos, fraglen)
        )
            DPLocWriteBuffer<CUDP_SWFT_2DCACHE_DIM_X>(
                bottmCache, tmpdpbotbuffer, dbpos, doffs, dblen);
    }

    ///WRITE DP cell coordinates of maximum scores, which corresponds to 
    ///writing found coordinates of the bottom and right edge of data block
    ///if(0 <= x && x < dbstrlen)
    ///    //maxscCoords[threadIdx.x+1] is the left edge of the current data block:
    ///    maxscoordsbuf[yofff + dbpos] = maxscCoords[threadIdx.x+1];
    ///make sure maxscCoords last index falls within bounds
    ///if(threadIdx.x < CUDP_SWFT_2DCACHE_DIM_D-1 &&
    ///   0 <= x+CUDP_SWFT_2DCACHE_DIM_D && x+CUDP_SWFT_2DCACHE_DIM_D < dbstrlen)
    ///    //write the right edge of the data block: maxscCoords[threadIdx.x+1+CUDP_SWFT_2DCACHE_DIM_D]:
    ///    maxscoordsbuf[yofff + dbpos + CUDP_SWFT_2DCACHE_DIM_D] =
    ///        maxscCoords[threadIdx.x+1+CUDP_SWFT_2DCACHE_DIM_D];
}

// =========================================================================
// Instantiations
//
#define INSTANTIATE_ExecDPScore3264x(tpANCHORRGN,tpBANDED,tpGAP0,tpCHECKCONV) \
    template \
    __global__ void ExecDPScore3264x<tpANCHORRGN,tpBANDED,tpGAP0,tpCHECKCONV>( \
        const uint blkdiagnum, \
        const uint nqystrs, const uint ndbCstrs, const uint ndbCposs, \
        const uint dbxpad, \
        const uint maxnsteps, const float gapopencost, \
        const float* __restrict__ wrkmemtm, \
        const float* __restrict__ wrkmemaux, \
        float* __restrict__ tmpdpdiagbuffers, \
        float* __restrict__ tmpdpbotbuffer);

INSTANTIATE_ExecDPScore3264x(false,false,true,true);

// -------------------------------------------------------------------------
