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
#include "dpw_btck_complete.cuh"

// #define CUDP_COMPLETE_BTCK_TESTPRINT 0 //3024//4899

// =========================================================================

// NOTE: parameters are passed to the device via constant memory and are 
// limited to 4 KB
// 
// device functions for executing dynamic programming with backtracking 
// information;
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

// -------------------------------------------------------------------------
// ExecCompleteDPwBtck512x: execute complete dynamic programming with 
// backtracking information using shared memory and 512-fold unrolling;
// maxnsteps, max number of steps performed for each reference structure 
// during alignment refinement;
// stepnumber, step number corresponding to the slot to read transformatioon
// matrix from;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary buffers for last calculated diagonal scores;
// tmpdpbotbuffer, temporary buffers for last calculated bottom scores;
// maxscoordsbuf, coordinates (positions) of maximum alignment scores;
// btckdata, backtracking information data;
// 
template<bool ANCHORRGN, bool BANDED, bool GAP0, int D02IND>
__global__
void ExecCompleteDPwBtck512x(
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const uint stepnumber,
    const float gapopencost,
    const float* __restrict__ wrkmemtmibest,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpbotbuffer,
//     uint* __restrict__ maxscoordsbuf,
    char* __restrict__ btckdata)
{
    // blockIdx.x is the reference serial number;
    // blockIdx.y is the query serial number;
    const uint dbstrndx = blockIdx.x;
    const uint qryndx = blockIdx.y;
    enum {MINDDIM = CUDP_COMPLETE_2DCACHE_MINDIM_D,//minimum value for dimension
        LOG2MINDDIM = CUDP_COMPLETE_2DCACHE_MINDIM_D_LOG2};
    const int DDIM = blockDim.x;//inner dimension of oblique blocks
    const int DPAD = DDIM >> LOG2MINDDIM;//padding to avoid bank conflicts
    const int DDIMPAD = DDIM + DPAD;//dimension + padding
    const int DDIMPAD1 = DDIMPAD + 1;//1 extra for side edges (diagonals)
    const int dblen = ndbCposs + dbxpad;
    //offset (w/o a factor) to the beginning of the data along the y axis 
    // wrt query qryndx: 
    const int yofff = dblen * qryndx;
    const int doffs = nTDPBottomScoreSections * nTDPDiagScoreSubsections * yofff;
    //cache for scores, coordinates, and transformation matrix 
    //(use dynamically allocated SM since its size varies for smaller proteins):
    extern __shared__ float dataCache[];
    float* tfmCache = dataCache;//transformation matrix [nTTranformMatrix]
    float* diag1Cache = tfmCache + nTTranformMatrix;//diag-1 scores [nTDPDiagScoreSubsections * DDIMPAD1]
    float* diag2Cache = diag1Cache + nTDPDiagScoreSubsections * DDIMPAD1;//last (2nd) diag scores
    float* bottmCache = diag2Cache + nTDPDiagScoreSubsections * DDIMPAD1;//bottom scores [nTDPDiagScoreSubsections * DDIMPAD]
    float* rfnCoords = bottmCache + nTDPDiagScoreSubsections * DDIMPAD;//cached coordinates [pmv2DNoElems * DDIM * 2]
    //backtracking information of dimensions [DDIM][32+1] ([DDIM][MINDDIM+1])
    char* btckCache = (char*)(rfnCoords + pmv2DNoElems * DDIM * 2);
    float qry2DX, qry2DY, qry2DZ;//query coordiinates
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;
    int qrypos = 0, rfnpos = 0;
    int sfragndx = 0, sfragpos = 0;
    int fraglen = 0;


    //check SCORE convergence first
    if(threadIdx.x == 0) {
        uint mloc = ((qryndx * maxnsteps + 0/*sfragfct*/) * nTAuxWorkingMemoryVars) * ndbCstrs;
        rfnCoords[0] = wrkmemaux[mloc + tawmvConverged * ndbCstrs + dbstrndx];
    }

    __syncthreads();

    if(((int)rfnCoords[0]) & 
       (CONVERGED_SCOREDP_bitval | CONVERGED_NOTMPRG_bitval | CONVERGED_LOWTMSC_bitval))
        //DP or finding rotation matrix converged already; 
        //(or the termination flag for this pair is set);
        //all threads in the block exit;
        return;


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse ccmCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(dbstrndx, (int*)tfmCache);
        GetQueryLenDst(qryndx, (int*)tfmCache + 2);
    }

    __syncthreads();

    //NOTE: no bank conflict when two threads from the same warp access the same address;
    dbstrlen = ((int*)tfmCache)[0]; dbstrdst = ((int*)tfmCache)[1];
    qrylen = ((int*)tfmCache)[2]; qrydst = ((int*)tfmCache)[3];

    __syncthreads();


    if(ANCHORRGN) {
        if(threadIdx.x == tawmvQRYpos || threadIdx.x == tawmvRFNpos ||
        threadIdx.x == tawmvSubFragNdx || threadIdx.x == tawmvSubFragPos)
        {
            //NOTE: reuse cache to contain query and reference positions and other fields
            //structure-specific-formatted data: 4 uncoalesced reads
            uint mloc = ((qryndx * maxnsteps + 0/*sfragfct*/) * nTAuxWorkingMemoryVars) * ndbCstrs;
            bottmCache[threadIdx.x] = wrkmemaux[mloc + threadIdx.x * ndbCstrs + dbstrndx];
        }

        __syncthreads();

        qrypos = bottmCache[tawmvQRYpos]; rfnpos = bottmCache[tawmvRFNpos];
        sfragndx = bottmCache[tawmvSubFragNdx]; sfragpos = bottmCache[tawmvSubFragPos];

        __syncthreads();

        fraglen = GetFragLength(qrylen, dbstrlen, qrypos, rfnpos, sfragndx);
        if(fraglen < 1)
            //fraglen was saved to be valid, but verify anyway
            return;

        qrypos += sfragpos; rfnpos += sfragpos;
    }


    //READ TRANSFORMATION MATRIX for query-reference pair
    if(threadIdx.x < nTTranformMatrix) {
        //iteration-best transformation matrix written at position 0;
        //alternatively, transformation matrix can be written at position stepnumber:
        uint mloc0 = ((qryndx * maxnsteps + stepnumber/*0*/) * ndbCstrs + dbstrndx) * nTTranformMatrix;
        tfmCache[threadIdx.x] = wrkmemtmibest[mloc0 + threadIdx.x];
    }


    __syncthreads();


    float *pdiag1 = diag1Cache;
    float *pdiag2 = diag2Cache;
    float d02;
    if(D02IND == D02IND_SEARCH) d02 = GetD02(qrylen, dbstrlen);
    else if(D02IND == D02IND_DPSCAN) d02 = GetD02_dpscan(qrylen, dbstrlen);

    // (x,y) is the bottom-left corner (x,y) coordinates in DP matrix
    for(int y = DDIM - 1; y+1-DDIM < qrylen; y += DDIM)
    {
        if(ANCHORRGN && (y < qrypos || qrypos + fraglen <= y+1-DDIM)) continue;

        //thread's position:
        int qpos = y - threadIdx.x;//going upwards

        //read query coordinates
        DPLocInitCoords<0/*shift*/,CUDP_DEFCOORD_QRY>(qry2DX, qry2DY, qry2DZ);
        if(0 <= qpos && qpos < qrylen) {
            DPLocCacheQryCoords(qry2DX, qry2DY, qry2DZ, qpos + qrydst);
            //transform the query fragment read
            transform_point(tfmCache, qry2DX, qry2DY, qry2DZ);
        }

        DPLocInitCoords(DDIM/*shift*/,CUDP_DEFCOORD_RFN, rfnCoords);
        DPLocInitCache512x(DDIMPAD1, 0/*shift*/, pdiag1);
        DPLocInitCache512x(DDIMPAD1, 1/*shift*/, pdiag2);

        __syncthreads();


        for(int x = -DDIM; x < dbstrlen; x += DDIM)
        {
            if(ANCHORRGN && (x+2*DDIM-2 < rfnpos || rfnpos + fraglen <= x)) continue;

            //the position this thread will process
            int xx = x + threadIdx.x;
            //db reference structure position corresponding to the oblique block's
            // bottom-left corner plus the offset determined by thread id:
            int dbpos = xx + dbstrdst;//going right

            //copy reference coordinates cached previously
            DPLocAssignCoords(0/*shft_trg*/, DDIM/*shft_src*/, rfnCoords);

            //no sync as long as the shifts are the same for each thread;
            //required, however, for the sync of bottmCache
            __syncthreads();

            //read reference coordinates
            DPLocInitCoords(DDIM/*shift*/,CUDP_DEFCOORD_RFN, rfnCoords);
            if(0 <= (xx+DDIM) && (xx+DDIM) < dbstrlen)
                DPLocCacheRfnCoords(DDIM, rfnCoords, dbpos + DDIM);

            //cache the bottom of the upper oblique blocks
            DPLocInitCache512x(DDIMPAD, 0/*shift*/, bottmCache);
            if(DDIM <= y && 0 <= (xx+DDIM-1) && (xx+DDIM-1) < dbstrlen)
                DPLocCacheBuffer(DDIMPAD, 0/*shift*/,
                    bottmCache, tmpdpbotbuffer, dbpos+DDIM-1, doffs, dblen);

            __syncthreads();


            //calculations with 512x unrolling;
            //NOTE: sync inside: do not branch;
            for(int i = 0; i < DDIM && x+i < dbstrlen; i++)
            {
                float val1 = 0.0f, val2;
                float rfn2DX = DPLocGetCoord<pmv2DX>(threadIdx.x+i, rfnCoords);
                int btck;

                if(threadIdx.x+1 == DDIM)
                    DPLocSetCacheVal<LOG2MINDDIM>(DDIMPAD1, dpdsssStateMM, threadIdx.x+1, pdiag1, 
                        DPLocGetCacheVal<LOG2MINDDIM>(DDIMPAD, dpdsssStateMM, i, bottmCache));

                if(qry2DX < CUDP_DEFCOORD_QRY_cmp && CUDP_DEFCOORD_RFN_cmp < rfn2DX) {
                    val1 = distance2(
                        qry2DX, qry2DY, qry2DZ,
                        rfn2DX,
                        DPLocGetCoord<pmv2DY>(threadIdx.x+i, rfnCoords),
                        DPLocGetCoord<pmv2DZ>(threadIdx.x+i, rfnCoords)
                    );
                    val1 = GetPairScore(d02, val1);//score
                }

                //NOTE: TRICK to implement a special case of DP with affine gap cost scheme:
                //NOTE: gap extension cost is 0;
                //NOTE: match scores are always non-negative; hence, an alignemnt score too;
                //NOTE: save NEGATED match scores to indicate diagonal direction in alignment;
                //NOTE: when gaps lead to negative scores, match scores will always be preferred;

                //MM state update (diagonal direction)
                val1 += 
                    GAP0? DPLocGetCacheVal<LOG2MINDDIM>(DDIMPAD1, dpdsssStateMM, threadIdx.x+1, pdiag2):
                    fabsf(DPLocGetCacheVal<LOG2MINDDIM>(DDIMPAD1, dpdsssStateMM, threadIdx.x+1, pdiag2));
                btck = dpbtckDIAG;

                //sync to update pdiag1; also no read for pdiag2 in this iteration
                __syncthreads();

                //IM state update (left direction)
                val2 = DPLocGetCacheVal<LOG2MINDDIM>(DDIMPAD1, dpdsssStateMM, threadIdx.x, pdiag1);
                if(!GAP0 && val2 < 0.0f) val2 = gapopencost - val2;
                myhdmaxassgn(val1, val2, btck, (int)dpbtckLEFT);

                //MI state update (up direction)
                val2 = DPLocGetCacheVal<LOG2MINDDIM>(DDIMPAD1, dpdsssStateMM, threadIdx.x+1, pdiag1);
                if(!GAP0 && val2 < 0.0f) val2 = gapopencost - val2;
                myhdmaxassgn(val1, val2, btck, (int)dpbtckUP);

                //WRITE: write max value
                DPLocSetCacheVal<LOG2MINDDIM>(DDIMPAD1, dpdsssStateMM, threadIdx.x, pdiag2, 
                    (GAP0 || btck != dpbtckDIAG)? val1: -val1);

                //WRITE backtracking information to smem
                //TODO: make btck dpbtckSTOP outside the anchor region
                DPLocSetBtckCacheVal(MINDDIM+1, threadIdx.x, i&(MINDDIM-1), btckCache, btck);

                if(threadIdx.x == 0) {
                    //WRITE: position not used by other threads in the current iteration
                    DPLocSetCacheVal<LOG2MINDDIM>(DDIMPAD, dpdsssStateMM, i, bottmCache, 
                        DPLocGetCacheVal<LOG2MINDDIM>(DDIMPAD1, dpdsssStateMM, threadIdx.x, pdiag2));
                }

#ifdef CUDP_COMPLETE_BTCK_TESTPRINT
                if(dbstrndx==CUDP_COMPLETE_BTCK_TESTPRINT){
                    printf(" i%02d (t%02u): len= %d addr= %u SC= %.4f (yx: %d,%d) "
                        "MM= %.6f  "// MAX= %.6f COORD= %x\n"// BTCK= %d\n"
                        "  >qX= %.4f qY= %.4f qZ= %.4f   dX= %.4f dY= %.4f dZ= %.4f\n",
                        i,threadIdx.x,  dbstrlen,dbstrdst,val1, y-threadIdx.x,xx+i,
                        DPLocGetCacheVal<LOG2MINDDIM>(DDIMPAD1, dpdsssStateMM, threadIdx.x, pdiag2),
                        //maxscCache, maxscCoords,// btck,
                        qry2DX, qry2DY, qry2DZ,
                        DPLocGetCoord<pmv2DX>(threadIdx.x+i, rfnCoords),
                        DPLocGetCoord<pmv2DY>(threadIdx.x+i, rfnCoords),
                        DPLocGetCoord<pmv2DZ>(threadIdx.x+i, rfnCoords)
                    );
                    for(size_t _k=0;_k<1000000000UL;_k++)clock();
                    for(size_t _k=0;_k<1000000000UL;_k++)clock();
                }
#endif

                myhdswap(pdiag1, pdiag2);

                //sync for updates
                __syncthreads();

                //{{WRITE a block [DDIM][MINDDIM] of backtracking information to gmem
                //once synced
                if((((i+1)&(MINDDIM-1)) == 0) || (x+i+1 >= dbstrlen)) {
                    //NOTE: the following pragma prevents unrolling and 
                    //keeps #registers at a reasonable limit:
                    #pragma unroll 1
                    for(int p = 0; p < MINDDIM; p++) {
                        //going upwards
                        int ii = p * DPAD + (threadIdx.x >> LOG2MINDDIM);//pos in y dimension
                        int tt = threadIdx.x & (MINDDIM-1);//pos in x dimension
                        int ttt = tt + (i >> LOG2MINDDIM) * MINDDIM;
                        if(0 <= y-ii && y-ii < qrylen && 0 <= x+ttt+ii && x+ttt+ii < dbstrlen)
                        {
                            //starting position of line ii of the oblq. diagonal block in the matrix:
                            int qqpos = (qrydst + (y-ii)) * dblen + ii;
                            btckdata[qqpos + x+ttt+dbstrdst] = 
                                DPLocGetBtckCacheVal(MINDDIM+1, ii, tt, btckCache);
                        }
                    }
                }
                //}}btck WRT
            }//for(i)

            //WRITE the bottom edge of the oblique block processed;
            if(y+1 < qrylen && 0 <= (xx) && (xx) < dbstrlen)
                DPLocWriteBuffer(DDIMPAD, 0/*shift*/,
                    bottmCache, tmpdpbotbuffer, dbpos, doffs, dblen);

        }//for(x)
    }//for(y)
}

// =========================================================================
// Instantiations
//
#define INSTANTIATE_ExecCompleteDPwBtck512x(tpANCHORRGN, tpBANDED, tpGAP0, tpD02IND) \
    template \
    __global__ void ExecCompleteDPwBtck512x<tpANCHORRGN,tpBANDED,tpGAP0,tpD02IND>( \
        const uint ndbCstrs, const uint ndbCposs, const uint dbxpad, \
        const uint maxnsteps, const uint stepnumber, \
        const float gapopencost, \
        const float* __restrict__ wrkmemtmibest, \
        const float* __restrict__ wrkmemaux, \
        float* __restrict__ tmpdpbotbuffer, \
        char* __restrict__ btckdata);

INSTANTIATE_ExecCompleteDPwBtck512x(false,false,false,D02IND_SEARCH);
INSTANTIATE_ExecCompleteDPwBtck512x(false,false,true,D02IND_SEARCH);

// -------------------------------------------------------------------------
