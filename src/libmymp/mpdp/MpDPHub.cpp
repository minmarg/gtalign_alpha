/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include <omp.h>

#include "libutil/cnsts.h"
#include "libutil/alpha.h"
#include "libutil/macros.h"
#include "libutil/templates.h"
#include "libutil/CLOptions.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gproc/dputils.h"
#include "libgenp/gproc/btckcoords.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libgenp/gdats/PMBatchStrData.h"
#include "libmymp/mpproc/mpprocconf.h"
#include "libmymp/mputil/simdscan.h"
#include "libmymp/mplayout/MpGlobalMemory.h"
#include "libmymp/mpstages/scoringbase.h"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmymp/mpdp/mpdpbase.h"
#include "MpDPHub.h"



// -------------------------------------------------------------------------
// #define MPDP_INIT_BTCK_TESTPRINT 0
// -------------------------------------------------------------------------
// dynamic programming with backtracking information;
// NOTE: Version for MPDP_2DCACHE_DIM_DequalsX: MPDP_2DCACHE_DIM_D==MPDP_2DCACHE_DIM_X!
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
// ALTSCTMS, template flag indicating the use of alternative memory section of tms;
// (starting at x=-MPDP_2DCACHE_DIM);
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
// ExecDPwBtck128xKernel: execute dynamic programming with backtracking 
// information using shared memory and 128-fold unrolling 
// along the diagonal of dimension MPDP_2DCACHE_DIM;
// maxnsteps, max number of steps performed for each reference structure 
// during alignment refinement;
// stepnumber, step number corresponding to the slot to read transformation
// matrix from;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary buffers for last calculated diagonal scores;
// tmpdpbotbuffer, temporary buffers for last calculated bottom scores;
// maxscoordsbuf, coordinates (positions) of maximum alignment scores;
// btckdata, backtracking information data;
// 
template<bool ANCHOR, bool BANDED, bool GAP0, int D02IND, bool ALTSCTMS>
void MpDPHub::ExecDPwBtck128xKernel(
    const float gapopencost,
    const int stepnumber,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* const __RESTRICT__ wrkmemtmibest,
    const float* const __RESTRICT__ wrkmemaux,
    float* const __RESTRICT__ tmpdpdiagbuffers,
    float* const __RESTRICT__ tmpdpbotbuffer,
//     uint* const __RESTRICT__ maxscoordsbuf,
    char* const __RESTRICT__ btckdata)
{
    enum {
        DIMD = MPDP_2DCACHE_DIM_D,
        DIMD1 = DIMD + 1,
        DIMX = MPDP_2DCACHE_DIM_X,
        DIMDpX = MPDP_2DCACHE_DIM_DpX,
        DIMD_LOG2 = MPDP_2DCACHE_DIM_D_LOG2
    };

    MYMSG("MpDPHub::ExecDPwBtck128xKernel", 4);
    // static const std::string preamb = "MpDPHub::ExecDPwBtck128xKernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = CuMemoryBase::GetMinMemAlignment();

    //execution configuration for DP:
    //1D thread block processes 2D DP matrix oblique block of dimension 
    //MPDP_2DCACHE_DIM_D x MPDP_2DCACHE_DIM_X;
    //NOTE: using block diagonals, where blocks share a common point 
    //NOTE: (corner) with a neighbour in a diagonal;
    const int maxblkdiagelems = GetMaxBlockDiagonalElems(
            dbstr1len_, qystr1len_, MPDP_2DCACHE_DIM_D, MPDP_2DCACHE_DIM_X);
    const int nblocks_x = maxblkdiagelems;
    const int nblocks_y = ndbCstrs_;
    const int nblocks_z = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_z * (size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPDP_2DCACHE_CHSIZE);

    //number of regular DIAGONAL block diagonal series, each of given dimensions;
    //rect coords (x,y) are related to diagonal number d by d=x+y-1;
    //then, this number d is divided by the length of the block diagonal;
    //uint nblkdiags =
    // ((dbstr1len+qystr1len-1)+MPDP_2DCACHE_DIM_X-1)/MPDP_2DCACHE_DIM_X;
    //REVISION: due to the positioning of the first block, the first 
    // 1-position diagonal of the first diagonal block is out of bounds: remove -1
    int nblkdiags = (int)
        (((dbstr1len_ + qystr1len_) + MPDP_2DCACHE_DIM_X-1) / MPDP_2DCACHE_DIM_X);
    //----
    //NOTE: now use block DIAGONALS, where blocks share a COMMON POINT 
    //NOTE: (corner, instead of edge) with a neighbour in a diagonal;
    //the number of series of such block diagonals equals 
    // #regular block diagonals (above) + {(l-1)/w}, 
    // where l is query length (y coord.), w is block width (dimension), and
    // {} denotes floor rounding; {(l-1)/w} is #interior divisions;
    nblkdiags += (int)(qystr1len_ - 1) / MPDP_2DCACHE_DIM_D;

    float diag1[nTDPDiagScoreSubsections * DIMD1];//cache for scores of the 1st diagonal
    float diag2[nTDPDiagScoreSubsections * DIMD1];//last (2nd) diagonal
    float bottm[nTDPDiagScoreSubsections * DIMX];//bottom scores
    float rfnCoords[pmv2DNoElems * DIMDpX];
    float qryCoords[pmv2DNoElems * DIMD];
    float tfm[nTTranformMatrix];
    //NOTE: max scores and their coordinates are not recorded for semi-global alignment!
    ///float maxscCache;//maximum scores of the last processed diagonal
    ///uint maxscCoords = 0;//coordinates of the maximum alignment score maxscCache
    //SECTION for backtracking information
    char btck[DIMD][DIMX];

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        private(diag1,diag2,bottm, rfnCoords,qryCoords, tfm, btck) \
        shared(nblkdiags)
    {
        //iterate over olblique block anti-diagonals to perform DP;
        //nblkdiags, total (max) number of anti-diagonals:
        for(int dd = 0; dd < nblkdiags; dd++)
        {
            #pragma omp for collapse(3) schedule(dynamic, chunksize)
            for(int qi = 0; qi < nblocks_z; qi++)
                for(int ri = 0; ri < nblocks_y; ri++)
                    for(int bi = 0; bi < nblocks_x; bi++)
                    {//threads process oblique blocks on anti-diagonals of query-reference pairs
                        //check convergence:
                        int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                        int mloc = ((qi * maxnsteps_ + stepnumber) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                        tfm[0] = tfm[1] = wrkmemaux[mloc0 + ri];//reuse cache
                        if(stepnumber != 0) tfm[1] = wrkmemaux[mloc + ri];
                        if((((int)(tfm[0])) & (CONVERGED_LOWTMSC_bitval)) ||
                           (((int)(tfm[1])) & (CONVERGED_SCOREDP_bitval | CONVERGED_NOTMPRG_bitval | CONVERGED_LOWTMSC_bitval)))
                            continue;

                        const int qrylen = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                        const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                        const int dbstrlen = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                        const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);

                        //lastydiagnum, last block diagonal serial number along y axis:
                        //each division separates a number of diagonals (nsepds);
                        constexpr int nsepds = 2;//(float)DIMD/(float)DIMX + 1.0f;
                        //the number of the last diagonal starting at x=-DIMD
                        ///int nintdivs = (qrylen-1)>>DIMD_LOG2;//(qrylen-1)/DIMD;
                        ///uint lastydiagnum = nsepds * nintdivs + 1 - 1;//-1 for zero-based indices;
                        int lastydiagnum = ((qrylen-1) >> DIMD_LOG2) * nsepds;

                        // blockIdx.x is block serial number s within diagonal blkdiagnum;
                        // (x,y) is the bottom-left corner (x,y) coordinates for structure dbstrndx
                        int x, y;
                        if(dd <= lastydiagnum) {
                            //x=-!(d%2)w+2ws; y=dw/2+w-sw -1 (-1, zero-based indices); [when w==b]
                            //(b, block's length; w, block's width)
                            x = (2 * bi - (!(dd & 1))) * DIMD;
                            y = ((dd >> 1) + 1 - bi) * DIMD - 1;
                        } else {
                            //x=-w+(d-d_l)w+2ws; y=dw/2+w-sw -1; [when w==b]
                            x = (2 * bi + (dd - lastydiagnum - 1)) * DIMD;
                            y = ((lastydiagnum >> 1) + 1 - bi) * DIMD - 1;
                        }

                        //number of iterations for this block to perform;
                        int ilim = GetMaqxNoIterations(x, y, qrylen, dbstrlen, DIMX);

                        if(y < 0 || qrylen <= (y+1 - DIMD) || 
                           dbstrlen <= x /*+ DIMDpX */ ||
                           ilim < 1)
                            continue;//out of boundaries

                        //READ TRANSFORMATION MATRIX for query-reference pair
                        //iteration-best transformation matrix written at position 0;
                        mloc0 = ((qi * maxnsteps_ + 0) * ndbCstrs_ + ri) * nTTranformMatrix;
                        //alternatively, transformation matrix can be written at position stepnumber:
                        //NOTE: CUS1_TBSP_DPSCORE_TOP_N_REFINEMENTxMAX_CONFIGS for tms!
                        if(ALTSCTMS)
                            mloc0 = ((qi * CUS1_TBSP_DPSCORE_TOP_N_REFINEMENTxMAX_CONFIGS + stepnumber) * ndbCstrs_ + ri) *
                            nTTranformMatrix;
                        #pragma omp simd aligned(wrkmemtmibest:memalignment)
                        for(int f = 0; f < nTTranformMatrix; f++)
                            tfm[f] = wrkmemtmibest[mloc0 + f];

                        //READ COORDINATES
                        // int qpos = y - threadIdx.x;//going upwards
                        // //x is now the position this thread will process
                        // x += threadIdx.x;

                        ReadAndTransformQryCoords<DIMD,PMBSdatalignment>(
                            x, y, qrydst, qrylen,  querypmbeg, tfm,  qryCoords);

                        //db reference structure position corresponding to the oblique block's
                        //bottom-left corner in the buffers dc_pm2dvfields_ (score matrix) 
                        //plus the offset determined by thread id:
                        // int dbpos = x + dbstrdst;//going right
                        int dblen = ndbCposs_ + dbxpad_;
                        //offset (w/o a factor) to the beginning of the data along the y axis wrt query qi: 
                        int yofff = dblen * qi;

                        ReadRfnCoords<DIMDpX,PMBSdatalignment>(
                            x, y, dbstrdst, dbstrlen,  bdbCpmbeg,  rfnCoords);

                        //cache TWO DIAGONALS from the previous (along the x axis) oblique block;
                        //tmpdpdiagbuffers: (1D, along the x axis)

                        ReadTwoDiagonals<DIMD,DIMD1,memalignment>(
                            x, y, dbstrdst, dbstrlen, dblen, yofff,  tmpdpdiagbuffers,  diag1, diag2);

                        //cache the bottom of the upper oblique blocks;
                        ReadBottomEdge<DIMD,DIMX,memalignment>(
                            x, y, dbstrdst, dbstrlen, dblen, yofff,  tmpdpbotbuffer,  bottm);

                        float *pdiag1 = diag1;
                        float *pdiag2 = diag2;
                        float d02;
                        if(D02IND == D02IND_SEARCH) d02 = GetD02(qrylen, dbstrlen);
                        else if(D02IND == D02IND_DPSCAN) d02 = GetD02_dpscan(qrylen, dbstrlen);

                        //start calculations for this position with Nx unrolling
                        for(int i = 0; i < ilim/*DIMX*/; i++)
                        {
                            pdiag1[lfDgNdx<DIMD1>(dpdsssStateMM, DIMD)] =
                                bottm[lfDgNdx<DIMX>(dpdsssStateMM, i)];

                            #pragma omp simd
                            for(int pi = 0; pi < DIMD; pi++) {
                                float val1 = 0.0f, val2;//, left, up;
                                float qry2DX = qryCoords[lfNdx(pmv2DX, pi)];
                                float qry2DY = qryCoords[lfNdx(pmv2DY, pi)];
                                float qry2DZ = qryCoords[lfNdx(pmv2DZ, pi)];
                                float rfn2DX = rfnCoords[lfNdx(pmv2DX, pi + i)];
                                float rfn2DY = rfnCoords[lfNdx(pmv2DY, pi + i)];
                                float rfn2DZ = rfnCoords[lfNdx(pmv2DZ, pi + i)];
                                int bk;

                                if(qry2DX < CUDP_DEFCOORD_QRY_cmp && CUDP_DEFCOORD_RFN_cmp < rfn2DX) {
                                    val1 = distance2(qry2DX, qry2DY, qry2DZ,  rfn2DX, rfn2DY, rfn2DZ);
                                    val1 = GetPairScore(d02, val1);//score
                                }

                                //NOTE: TRICK to implement a special case of DP with affine gap cost scheme:
                                //NOTE: gap extension cost is 0;
                                //NOTE: match scores are always non-negative; hence, an alignemnt score too;
                                //NOTE: save NEGATED match scores to indicate diagonal direction in alignment;
                                //NOTE: when gaps lead to negative scores, match scores will always be preferred;

                                //MM state update (diagonal direction)
                                val1 += 
                                    GAP0? pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, pi+1)]:
                                    fabsf(pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, pi+1)]);
                                bk = dpbtckDIAG;
                                //NOTE: max scores and their coordinates are not recorded for semi-global alignment
                                ///dpmaxandcoords(maxscCache, val1, maxscCoords, x+pi+i, y-pi);

                                //////// SYNC ///////////

                                //IM state update (left direction)
                                val2 = /*left = */pdiag1[lfDgNdx<DIMD1>(dpdsssStateMM, pi)];
                                if(!GAP0 && val2 < 0.0f) val2 = gapopencost - val2;
                                mymaxassgn(val1, val2, bk, (int)dpbtckLEFT);

                                //MI state update (up direction)
                                val2 = /*up = */pdiag1[lfDgNdx<DIMD1>(dpdsssStateMM, pi+1)];
                                if(!GAP0 && val2 < 0.0f) val2 = gapopencost - val2;
                                mymaxassgn(val1, val2, bk, (int)dpbtckUP);

                                //WRITE: write max value
                                pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, pi)] =
                                    (GAP0 || bk != dpbtckDIAG)? val1: -val1;

                                //WRITE btck
                                btck[pi][i] = bk;

                            }//simd for(pi < DIMD)

                            bottm[lfDgNdx<DIMX>(dpdsssStateMM, i)] =
                                pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, 0)];

#ifdef MPDP_INIT_BTCK_TESTPRINT
                            if(ri==MPDP_INIT_BTCK_TESTPRINT)
                                for(int pi = 0; pi < DIMD; pi++)
                                printf(" d=%u(%u) s=%u i%02d/%u (t%02u): len= %d addr= %u SC= <> (yx: %d,%d) "
                                        "MM= %.6f  "// MAX= %.6f COORD= %x\n"// BTCK= %d\n"
                                        "  >qX= %.4f qY= %.4f qZ= %.4f   dX= %.4f dY= %.4f dZ= %.4f\n",
                                        dd, lastydiagnum, bi, i, ilim, pi,
                                        dbstrlen, dbstrdst,  y-pi, x+i,
                                        pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, pi)],
                                        //maxscCache, maxscCoords,// btck,
                                        qry2DX, qry2DY, qry2DZ,  rfn2DX, rfn2DY, rfn2DZ
                                );
#endif
                            myswap(pdiag1, pdiag2);
                        }//for(i < ilim)

#ifdef MPDP_INIT_BTCK_TESTPRINT
                        if(ri==CUDP_INIT_BTCK_TESTPRINT)
                            for(int pi = 0; pi < DIMD; pi++)
                            printf(" >>> d=%u(%u) s=%u (t%02u): len= %d wrt= %d xpos= %d\n", 
                                dd, lastydiagnum, bi, pi, dbstrlen,
                                x+DIMX-1+pi<dbstrlen, x+DIMX-1+pi);
#endif

                        //WRITE resulting TWO DIAGONALS for next-iteration blocks;
                        WriteTwoDiagonals<DIMD,DIMD1,memalignment>(
                            DIMX,
                            x, y, dbstrdst, dbstrlen, dblen, yofff,  tmpdpdiagbuffers,  diag1, diag2);

                        //WRITE the bottom edge of the oblique blocks;
                        WriteBottomEdge<DIMX,memalignment>(
                            x, y, dbstrdst, dbstrlen, dblen, yofff,  tmpdpbotbuffer,  bottm);

                        //WRITE backtracking information
                        WriteBtckInfo<DIMD,DIMX,memalignment>(
                            x, y, qrydst, qrylen, dbstrdst, dbstrlen,  dblen,  btckdata, btck);
                    }//omp for

            //use explicit barrier here
            #pragma omp barrier
        }//for(dd < nblkdiags)
    }//omp parallel
}

// =========================================================================
// Instantiations
// 
#define INSTANTIATE_MpDPHub_ExecDPwBtck128xKernel(tpANCHORRGN,tpBANDED,tpGAP0,tpD02IND,tpALTSCTMS) \
    template void MpDPHub::ExecDPwBtck128xKernel<tpANCHORRGN,tpBANDED,tpGAP0,tpD02IND,tpALTSCTMS>( \
        const float gapopencost, const int stepnumber, \
        const char* const * const __RESTRICT__ querypmbeg, \
        const char* const * const __RESTRICT__ bdbCpmbeg, \
        const float* const __RESTRICT__ wrkmemtmibest, \
        const float* const __RESTRICT__ wrkmemaux, \
        float* const __RESTRICT__ tmpdpdiagbuffers, \
        float* const __RESTRICT__ tmpdpbotbuffer, \
        char* const __RESTRICT__ btckdata);

INSTANTIATE_MpDPHub_ExecDPwBtck128xKernel(false,false,false,D02IND_SEARCH,false);
INSTANTIATE_MpDPHub_ExecDPwBtck128xKernel(false,false,true,D02IND_SEARCH,false);
INSTANTIATE_MpDPHub_ExecDPwBtck128xKernel(false,false,true,D02IND_SEARCH,true);
INSTANTIATE_MpDPHub_ExecDPwBtck128xKernel(false,false,true,D02IND_DPSCAN,false);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// #define MPDP_SCORE_TESTPRINT 0
// -------------------------------------------------------------------------
// ExecDPScore128xKernel: kernel for executing dynamic programming to 
// calculate max score in linear space (without backtracking) with N-fold 
// unrolling along the diagonal of dimension MPDP_SWFT_2DCACHE_DIM_D;
// NOTE: memory pointers should be aligned!
// wrkmemtm, memory of iteration-specific transformation matrices;
// wrkmemaux, auxiliary working memory;
// tmpdpdiagbuffers, temporary buffers for last calculated diagonal scores;
// tmpdpbotbuffer, temporary buffers for last calculated bottom scores;
// 
template<bool GAP0>
void MpDPHub::ExecDPScore128xKernel(
    const float gapopencost,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* const __RESTRICT__ wrkmemtm,
    const float* const __RESTRICT__ wrkmemaux,
    float* const __RESTRICT__ tmpdpdiagbuffers,
    float* const __RESTRICT__ tmpdpbotbuffer)
{
    enum {
        DIMD = MPDP_SWFT_2DCACHE_DIM_D,
        DIMD1 = DIMD + 1,
        DIMX = MPDP_SWFT_2DCACHE_DIM_X,
        DIMDpX = MPDP_SWFT_2DCACHE_DIM_DpX,
        DIMD_LOG2 = MPDP_SWFT_2DCACHE_DIM_D_LOG2,
        CHSIZE = MPDP_SWFT_2DCACHE_CHSIZE,
        NTOPTFMS = CUS1_TBSP_DPSCORE_TOP_N//number of best-performing tfms per section
    };

    MYMSG("MpDPHub::ExecDPScore128xKernel", 4);
    // static const std::string preamb = "MpDPHub::ExecDPwBtck128xKernel: ";
    static const int depth = CLOptions::GetC_DEPTH();
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = CuMemoryBase::GetMinMemAlignment();

    //execution configuration for DP:
    //1D thread block processes 2D DP matrix oblique block of dimension DIMD x DIMX;
    //NOTE: using block diagonals, where blocks share a common point 
    //NOTE: (corner) with a neighbour in a diagonal;
    const int maxblkdiagelems = GetMaxBlockDiagonalElems(
            dbstr1len_, qystr1len_, DIMD, DIMX);
    //configuration for swift DP: calculate optimal order-dependent scores:
    const uint nconfigsections = 
        (depth <= CLOptions::csdDeep)? 3: ((depth <= CLOptions::csdHigh)? 2: 1);
    const uint nconfigs = (NTOPTFMS) * nconfigsections;
    const int nblocks_x = maxblkdiagelems;
    const int nblocks_y = ndbCstrs_;
    const int nblocks_s = nconfigs;
    const int nblocks_z = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_z * (size_t)nblocks_s * (size_t)nblocks_y *
         (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)CHSIZE);

    int nblkdiags = (int)(((dbstr1len_ + qystr1len_) + DIMX - 1) / DIMX);
    nblkdiags += (int)(qystr1len_ - 1) / DIMD;

    float diag1[nTDPDiagScoreSubsections * DIMD1];//cache for scores of the 1st diagonal
    float diag2[nTDPDiagScoreSubsections * DIMD1];//last (2nd) diagonal
    float bottm[nTDPDiagScoreSubsections * DIMX];//bottom scores
    float rfnCoords[pmv2DNoElems * DIMDpX];
    float qryCoords[pmv2DNoElems * DIMD];
    float tfm[nTTranformMatrix];

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        private(diag1,diag2,bottm, rfnCoords,qryCoords, tfm) \
        shared(nblkdiags)
    {
        //iterate over olblique block anti-diagonals to perform DP;
        //nblkdiags, total (max) number of anti-diagonals:
        for(int dd = 0; dd < nblkdiags; dd++)
        {
            #pragma omp for collapse(4) schedule(dynamic, chunksize)
            for(int qi = 0; qi < nblocks_z; qi++)
                for(int si = 0; si < nblocks_s; si++)
                for(int ri = 0; ri < nblocks_y; ri++)
                    for(int bi = 0; bi < nblocks_x; bi++)
                    {//threads process oblique blocks on anti-diagonals of query-reference pairs
                        //check convergence:
                        int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                        int mloc = ((qi * maxnsteps_ + si) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                        tfm[0] = tfm[1] = wrkmemaux[mloc0 + ri];//reuse cache
                        if(si != 0) tfm[1] = wrkmemaux[mloc + ri];
                        if((((int)(tfm[0])) & (CONVERGED_LOWTMSC_bitval)) ||(tfm[1]))
                            continue;

                        const int qrylen = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                        const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                        const int dbstrlen = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                        const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);

                        //lastydiagnum, last block diagonal serial number along y axis:
                        //each division separates a number of diagonals (nsepds);
                        constexpr int nsepds = 2;//(float)DIMD/(float)DIMX + 1.0f;
                        //the number of the last diagonal starting at x=-DIMD
                        ///int nintdivs = (qrylen-1)>>DIMD_LOG2;//(qrylen-1)/DIMD;
                        ///uint lastydiagnum = nsepds * nintdivs + 1 - 1;//-1 for zero-based indices;
                        int lastydiagnum = ((qrylen-1) >> DIMD_LOG2) * nsepds;

                        // blockIdx.x is block serial number s within diagonal blkdiagnum;
                        // (x,y) is the bottom-left corner (x,y) coordinates for structure dbstrndx
                        int x, y;
                        if(dd <= lastydiagnum) {
                            //x=-!(d%2)w+2ws; y=dw/2+w-sw -1 (-1, zero-based indices); [when w==b]
                            //(b, block's length; w, block's width)
                            x = (2 * bi - (!(dd & 1))) * DIMD;
                            y = ((dd >> 1) + 1 - bi) * DIMD - 1;
                        } else {
                            //x=-w+(d-d_l)w+2ws; y=dw/2+w-sw -1; [when w==b]
                            x = (2 * bi + (dd - lastydiagnum - 1)) * DIMD;
                            y = ((lastydiagnum >> 1) + 1 - bi) * DIMD - 1;
                        }

                        //number of iterations for this block to perform;
                        int ilim = GetMaqxNoIterations(x, y, qrylen, dbstrlen, DIMX);

                        if(y < 0 || qrylen <= (y+1 - DIMD) || 
                           dbstrlen <= x /*+ DIMDpX */ ||
                           ilim < 1)
                            continue;//out of boundaries

                        //READ TRANSFORMATION MATRIX for query-reference pair;
                        mloc = ((qi * maxnsteps_ + si) * ndbCstrs_ + ri) * nTTranformMatrix;
                        #pragma omp simd aligned(wrkmemtm:memalignment)
                        for(int f = 0; f < nTTranformMatrix; f++) tfm[f] = wrkmemtm[mloc + f];

                        //READ COORDINATES
                        // int qpos = y - threadIdx.x;//going upwards
                        // //x is now the position this thread will process
                        // x += threadIdx.x;

                        ReadAndTransformQryCoords<DIMD,PMBSdatalignment>(
                            x, y, qrydst, qrylen,  querypmbeg, tfm,  qryCoords);

                        //db reference structure position corresponding to the oblique block's
                        //bottom-left corner in the buffers dc_pm2dvfields_ (score matrix) 
                        //plus the offset determined by thread id:
                        // int dbpos = x + dbstrdst;//going right
                        int dblen = ndbCposs_ + dbxpad_;
                        //offset (w/o a factor) to the beginning of the data along the y axis wrt query qi: 
                        // int yofff = dblen * qi;
                        int yofff = (qi * maxnsteps_ + si) * dblen;

                        ReadRfnCoords<DIMDpX,PMBSdatalignment>(
                            x, y, dbstrdst, dbstrlen,  bdbCpmbeg,  rfnCoords);

                        //cache TWO DIAGONALS from the previous (along the x axis) oblique block;
                        //tmpdpdiagbuffers: (1D, along the x axis)

                        ReadTwoDiagonals<DIMD,DIMD1,memalignment>(
                            x, y, dbstrdst, dbstrlen, dblen, yofff,  tmpdpdiagbuffers,  diag1, diag2);

                        //cache the bottom of the upper oblique blocks;
                        ReadBottomEdge<DIMD,DIMX,memalignment>(
                            x, y, dbstrdst, dbstrlen, dblen, yofff,  tmpdpbotbuffer,  bottm);

                        float *pdiag1 = diag1;
                        float *pdiag2 = diag2;
                        float d02 = GetD02/*_dpscan*/(qrylen, dbstrlen);

                        //start calculations for this position with Nx unrolling
                        for(int i = 0; i < ilim/*DIMX*/; i++)
                        {
                            pdiag1[lfDgNdx<DIMD1>(dpdsssStateMM, DIMD)] =
                                bottm[lfDgNdx<DIMX>(dpdsssStateMM, i)];

                            #pragma omp simd
                            for(int pi = 0; pi < DIMD; pi++) {
                                float val1 = 0.0f, val2;//, left, up;
                                float qry2DX = qryCoords[lfNdx(pmv2DX, pi)];
                                float qry2DY = qryCoords[lfNdx(pmv2DY, pi)];
                                float qry2DZ = qryCoords[lfNdx(pmv2DZ, pi)];
                                float rfn2DX = rfnCoords[lfNdx(pmv2DX, pi + i)];
                                float rfn2DY = rfnCoords[lfNdx(pmv2DY, pi + i)];
                                float rfn2DZ = rfnCoords[lfNdx(pmv2DZ, pi + i)];
                                int bk;

                                if(qry2DX < CUDP_DEFCOORD_QRY_cmp && CUDP_DEFCOORD_RFN_cmp < rfn2DX) {
                                    val1 = distance2(qry2DX, qry2DY, qry2DZ,  rfn2DX, rfn2DY, rfn2DZ);
                                    val1 = GetPairScore(d02, val1);//score
                                }

                                //NOTE: TRICK to implement a special case of DP with affine gap cost scheme:
                                //NOTE: gap extension cost is 0;
                                //NOTE: match scores are always non-negative; hence, an alignemnt score too;
                                //NOTE: save NEGATED match scores to indicate diagonal direction in alignment;
                                //NOTE: when gaps lead to negative scores, match scores will always be preferred;

                                //MM state update (diagonal direction)
                                val1 += 
                                    GAP0? pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, pi+1)]:
                                    fabsf(pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, pi+1)]);
                                bk = dpbtckDIAG;
                                //NOTE: max scores and their coordinates are not recorded for semi-global alignment
                                ///dpmaxandcoords(maxscCache, val1, maxscCoords, x+pi+i, y-pi);

                                //////// SYNC ///////////

                                //IM state update (left direction)
                                val2 = /*left = */pdiag1[lfDgNdx<DIMD1>(dpdsssStateMM, pi)];
                                if(!GAP0 && val2 < 0.0f) val2 = gapopencost - val2;
                                mymaxassgn(val1, val2, bk, (int)dpbtckLEFT);

                                //MI state update (up direction)
                                val2 = /*up = */pdiag1[lfDgNdx<DIMD1>(dpdsssStateMM, pi+1)];
                                if(!GAP0 && val2 < 0.0f) val2 = gapopencost - val2;
                                mymaxassgn(val1, val2, bk, (int)dpbtckUP);

                                //WRITE: write max value
                                pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, pi)] =
                                    (GAP0 || bk != dpbtckDIAG)? val1: -val1;

                            }//simd for(pi < DIMD)

                            bottm[lfDgNdx<DIMX>(dpdsssStateMM, i)] =
                                pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, 0)];

#ifdef MPDP_SCORE_TESTPRINT
                            if(ri==MPDP_SCORE_TESTPRINT)
                                for(int pi = 0; pi < DIMD; pi++)
                                printf(" d=%u(%u) s=%u i%02d/%u (t%02u): len= %d addr= %u SC= <> (yx: %d,%d) "
                                        "MM= %.6f  "// MAX= %.6f COORD= %x\n"// BTCK= %d\n"
                                        "  >qX= %.4f qY= %.4f qZ= %.4f   dX= %.4f dY= %.4f dZ= %.4f\n",
                                        dd, lastydiagnum, bi, i, ilim, pi,
                                        dbstrlen, dbstrdst,  y-pi, x+i,
                                        pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, pi)],
                                        //maxscCache, maxscCoords,// btck,
                                        qry2DX, qry2DY, qry2DZ,  rfn2DX, rfn2DY, rfn2DZ
                                );
#endif
                            myswap(pdiag1, pdiag2);
                        }//for(i < ilim)

#ifdef MPDP_SCORE_TESTPRINT
                        if(ri==MPDP_SCORE_TESTPRINT)
                            for(int pi = 0; pi < DIMD; pi++)
                            printf(" >>> d=%u(%u) s=%u (t%02u): len= %d wrt= %d xpos= %d\n", 
                                dd, lastydiagnum, bi, pi, dbstrlen,
                                x+DIMX-1+pi<dbstrlen, x+DIMX-1+pi);
#endif

                        //WRITE resulting TWO DIAGONALS for next-iteration blocks;
                        //{{before writing the diagonals, adjust the position so that
                        //the value of the last cell is always written at the last position
                        //(dbstrlen-1) for the last block irrespective of its placement 
                        //within the DP matrix; this ensures that the last cell
                        //contains the max score; (ilim>0 by definition; beginning)
                        int shift = DIMX - 1;
                        if(qrylen <= y+1 && ilim < DIMX) shift = ilim - 1;
                        //}}
                        WriteTwoDiagonals<DIMD,DIMD1,memalignment>(
                            shift,
                            x, y, dbstrdst, dbstrlen, dblen, yofff,  tmpdpdiagbuffers,  diag1, diag2);

                        //WRITE the bottom edge of the oblique blocks;
                        WriteBottomEdge<DIMX,memalignment>(
                            x, y, dbstrdst, dbstrlen, dblen, yofff,  tmpdpbotbuffer,  bottm);
                    }//omp for

            //use explicit barrier here
            #pragma omp barrier
        }//for(dd < nblkdiags)
    }//omp parallel
}

// =========================================================================
// Instantiations
// 
#define INSTANTIATE_MpDPHub_ExecDPScore128xKernel(tpGAP0) \
    template void MpDPHub::ExecDPScore128xKernel<tpGAP0>( \
        const float gapopencost, \
        const char* const * const __RESTRICT__ querypmbeg, \
        const char* const * const __RESTRICT__ bdbCpmbeg, \
        const float* const __RESTRICT__ wrkmemtm, \
        const float* const __RESTRICT__ wrkmemaux, \
        float* const __RESTRICT__ tmpdpdiagbuffers, \
        float* const __RESTRICT__ tmpdpbotbuffer);

INSTANTIATE_MpDPHub_ExecDPScore128xKernel(true);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// ExecDPTFMSSwBtck128xKernel: execute dynamic programming with secondary
// structure and backtracking information using shared memory and
// 128-fold unrolling along the diagonal of dimension MPDP_2DCACHE_DIM;
// GLOBTFM, template parameter, flag of using the globally best
// transformation matrix;
// GAP0, template parameter, flag of gap open cost 0;
// USESS, template parameter, flag of using secondary structure scoring;
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
void MpDPHub::ExecDPTFMSSwBtck128xKernel(
    const float gapopencost,
    const float ssweight,
    const int stepnumber,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* const __RESTRICT__ tfmmem,
    const float* const __RESTRICT__ wrkmemaux,
    float* const __RESTRICT__ tmpdpdiagbuffers,
    float* const __RESTRICT__ tmpdpbotbuffer,
//     uint* const __RESTRICT__ maxscoordsbuf,
    char* const __RESTRICT__ btckdata)
{
    enum {
        DIMD = MPDP_2DCACHE_DIM_D,
        DIMD1 = DIMD + 1,
        DIMX = MPDP_2DCACHE_DIM_X,
        DIMDpX = MPDP_2DCACHE_DIM_DpX,
        DIMD_LOG2 = MPDP_2DCACHE_DIM_D_LOG2
    };

    MYMSG("MpDPHub::ExecDPTFMSSwBtck128xKernel", 4);
    // static const std::string preamb = "MpDPHub::ExecDPTFMSSwBtck128xKernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = CuMemoryBase::GetMinMemAlignment();

    //execution configuration for DP:
    //1D thread block processes 2D DP matrix oblique block of dimension 
    //MPDP_2DCACHE_DIM_D x MPDP_2DCACHE_DIM_X;
    const int maxblkdiagelems = GetMaxBlockDiagonalElems(
            dbstr1len_, qystr1len_, MPDP_2DCACHE_DIM_D, MPDP_2DCACHE_DIM_X);
    const int nblocks_x = maxblkdiagelems;
    const int nblocks_y = ndbCstrs_;
    const int nblocks_z = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_z * (size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPDP_2DCACHE_CHSIZE);

    //number of regular DIAGONAL block diagonal series, each of given dimensions;
    //rect coords (x,y) are related to diagonal number d by d=x+y-1;
    int nblkdiags = (int)
        (((dbstr1len_ + qystr1len_) + MPDP_2DCACHE_DIM_X-1) / MPDP_2DCACHE_DIM_X);
    //NOTE: now use block DIAGONALS, where blocks share a COMMON POINT 
    //NOTE: (corner, instead of edge) with a neighbour in a diagonal;
    //the number of series of such block diagonals equals 
    // #regular block diagonals (above) + {(l-1)/w}, 
    // where l is query length (y coord.), w is block width (dimension), and
    // {} denotes floor rounding; {(l-1)/w} is #interior divisions;
    nblkdiags += (int)(qystr1len_ - 1) / MPDP_2DCACHE_DIM_D;

    float diag1[nTDPDiagScoreSubsections * DIMD1];//cache for scores of the 1st diagonal
    float diag2[nTDPDiagScoreSubsections * DIMD1];//last (2nd) diagonal
    float bottm[nTDPDiagScoreSubsections * DIMX];//bottom scores
    float rfnCoords[pmv2DNoElems * DIMDpX];
    float qryCoords[pmv2DNoElems * DIMD];
    float tfm[nTTranformMatrix];
    //NOTE: max scores and their coordinates are not recorded for semi-global alignment!
    ///float maxscCache;//maximum scores of the last processed diagonal
    ///uint maxscCoords = 0;//coordinates of the maximum alignment score maxscCache
    char rfnSS[DIMDpX];
    char qrySS[DIMD];
    //SECTION for backtracking information
    char btck[DIMD][DIMX];

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        private(diag1,diag2,bottm, rfnCoords,qryCoords, tfm, rfnSS,qrySS, btck) \
        shared(nblkdiags)
    {
        //iterate over olblique block anti-diagonals to perform DP;
        //nblkdiags, total (max) number of anti-diagonals:
        for(int dd = 0; dd < nblkdiags; dd++)
        {
            #pragma omp for collapse(3) schedule(dynamic, chunksize)
            for(int qi = 0; qi < nblocks_z; qi++)
                for(int ri = 0; ri < nblocks_y; ri++)
                    for(int bi = 0; bi < nblocks_x; bi++)
                    {//threads process oblique blocks on anti-diagonals of query-reference pairs
                        //check convergence:
                        int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                        tfm[0] = wrkmemaux[mloc0 + ri];//reuse cache
                        if(((int)(tfm[0])) & (CONVERGED_LOWTMSC_bitval))
                            continue;

                        const int qrylen = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                        const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                        const int dbstrlen = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                        const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);

                        //lastydiagnum, last block diagonal serial number along y axis:
                        //each division separates a number of diagonals (nsepds);
                        constexpr int nsepds = 2;//(float)DIMD/(float)DIMX + 1.0f;
                        //the number of the last diagonal starting at x=-DIMD
                        ///int nintdivs = (qrylen-1)>>DIMD_LOG2;//(qrylen-1)/DIMD;
                        ///uint lastydiagnum = nsepds * nintdivs + 1 - 1;//-1 for zero-based indices;
                        int lastydiagnum = ((qrylen-1) >> DIMD_LOG2) * nsepds;

                        // blockIdx.x is block serial number s within diagonal blkdiagnum;
                        // (x,y) is the bottom-left corner (x,y) coordinates for structure dbstrndx
                        int x, y;
                        if(dd <= lastydiagnum) {
                            //x=-!(d%2)w+2ws; y=dw/2+w-sw -1 (-1, zero-based indices); [when w==b]
                            //(b, block's length; w, block's width)
                            x = (2 * bi - (!(dd & 1))) * DIMD;
                            y = ((dd >> 1) + 1 - bi) * DIMD - 1;
                        } else {
                            //x=-w+(d-d_l)w+2ws; y=dw/2+w-sw -1; [when w==b]
                            x = (2 * bi + (dd - lastydiagnum - 1)) * DIMD;
                            y = ((lastydiagnum >> 1) + 1 - bi) * DIMD - 1;
                        }

                        //number of iterations for this block to perform;
                        int ilim = GetMaqxNoIterations(x, y, qrylen, dbstrlen, DIMX);

                        if(y < 0 || qrylen <= (y+1 - DIMD) || 
                           dbstrlen <= x /*+ DIMDpX */ ||
                           ilim < 1)
                            continue;//out of boundaries

                        //READ TRANSFORMATION MATRIX for query-reference pair
                        //iteration-best transformation matrix written at position 0;
                        //alternatively, transformation matrix can be written at position stepnumber:
                        mloc0 = ((qi * maxnsteps_ + stepnumber) * ndbCstrs_ + ri) * nTTranformMatrix;
                        //NOTE: globally best transformation matrix for a pair:
                        if(GLOBTFM) mloc0 = (qi * ndbCstrs_ + ri) * nTTranformMatrix;
                        #pragma omp simd aligned(tfmmem:memalignment)
                        for(int f = 0; f < nTTranformMatrix; f++)
                            tfm[f] = tfmmem[mloc0 + f];

                        //READ COORDINATES
                        // int qpos = y - threadIdx.x;//going upwards
                        // //x is now the position this thread will process
                        // x += threadIdx.x;

                        ReadAndTransformQryCoords<DIMD,PMBSdatalignment>(
                            x, y, qrydst, qrylen,  querypmbeg, tfm,  qryCoords);

                        //read ss information
                        if(USESS)
                            ReadQrySS<DIMD,PMBSdatalignment>(
                                x, y, qrydst, qrylen,  querypmbeg, qrySS);

                        //db reference structure position corresponding to the oblique block's
                        //bottom-left corner in the buffers dc_pm2dvfields_ (score matrix) 
                        //plus the offset determined by thread id:
                        // int dbpos = x + dbstrdst;//going right
                        int dblen = ndbCposs_ + dbxpad_;
                        //offset (w/o a factor) to the beginning of the data along the y axis wrt query qi: 
                        int yofff = dblen * qi;

                        ReadRfnCoords<DIMDpX,PMBSdatalignment>(
                            x, y, dbstrdst, dbstrlen,  bdbCpmbeg,  rfnCoords);

                        //read ss information
                        if(USESS)
                            ReadRfnSS<DIMDpX,PMBSdatalignment>(
                                x, y, dbstrdst, dbstrlen,  bdbCpmbeg, rfnSS);

                        //cache TWO DIAGONALS from the previous (along the x axis) oblique block;
                        //tmpdpdiagbuffers: (1D, along the x axis)

                        ReadTwoDiagonals<DIMD,DIMD1,memalignment>(
                            x, y, dbstrdst, dbstrlen, dblen, yofff,  tmpdpdiagbuffers,  diag1, diag2);

                        //cache the bottom of the upper oblique blocks;
                        ReadBottomEdge<DIMD,DIMX,memalignment>(
                            x, y, dbstrdst, dbstrlen, dblen, yofff,  tmpdpbotbuffer,  bottm);

                        float *pdiag1 = diag1;
                        float *pdiag2 = diag2;
                        float d02;
                        if(D02IND == D02IND_SEARCH) d02 = GetD02(qrylen, dbstrlen);
                        else if(D02IND == D02IND_DPSCAN) d02 = GetD02_dpscan(qrylen, dbstrlen);

                        //start calculations for this position with Nx unrolling
                        for(int i = 0; i < ilim/*DIMX*/; i++)
                        {
                            pdiag1[lfDgNdx<DIMD1>(dpdsssStateMM, DIMD)] =
                                bottm[lfDgNdx<DIMX>(dpdsssStateMM, i)];

                            #pragma omp simd
                            for(int pi = 0; pi < DIMD; pi++) {
                                float val1 = 0.0f, val2;//, left, up;
                                float qry2DX = qryCoords[lfNdx(pmv2DX, pi)];
                                float qry2DY = qryCoords[lfNdx(pmv2DY, pi)];
                                float qry2DZ = qryCoords[lfNdx(pmv2DZ, pi)];
                                float rfn2DX = rfnCoords[lfNdx(pmv2DX, pi + i)];
                                float rfn2DY = rfnCoords[lfNdx(pmv2DY, pi + i)];
                                float rfn2DZ = rfnCoords[lfNdx(pmv2DZ, pi + i)];
                                int bk;

                                if(qry2DX < CUDP_DEFCOORD_QRY_cmp && CUDP_DEFCOORD_RFN_cmp < rfn2DX) {
                                    val1 = distance2(qry2DX, qry2DY, qry2DZ,  rfn2DX, rfn2DY, rfn2DZ);
                                    val1 = GetPairScore(d02, val1);//score
                                }
                                if(USESS) val1 += (float)(qrySS[pi] == rfnSS[pi+i]) * ssweight;
                                // if(USESS) val1 += (float)(qrySS[pi] == rfnSS[pi+i] ||
                                //         (isloop(qrySS[pi]) && isloop(rfnSS[pi+i]))) * ssweight;

                                //NOTE: TRICK to implement a special case of DP with affine gap cost scheme:
                                //NOTE: gap extension cost is 0;
                                //NOTE: match scores are always non-negative; hence, an alignemnt score too;
                                //NOTE: save NEGATED match scores to indicate diagonal direction in alignment;
                                //NOTE: when gaps lead to negative scores, match scores will always be preferred;

                                //MM state update (diagonal direction)
                                val1 += 
                                    GAP0? pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, pi+1)]:
                                    fabsf(pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, pi+1)]);
                                bk = dpbtckDIAG;
                                //NOTE: max scores and their coordinates are not recorded for semi-global alignment
                                ///dpmaxandcoords(maxscCache, val1, maxscCoords, x+pi+i, y-pi);

                                //////// SYNC ///////////

                                //IM state update (left direction)
                                val2 = /*left = */pdiag1[lfDgNdx<DIMD1>(dpdsssStateMM, pi)];
                                if(!GAP0 && val2 < 0.0f) val2 = gapopencost - val2;
                                mymaxassgn(val1, val2, bk, (int)dpbtckLEFT);

                                //MI state update (up direction)
                                val2 = /*up = */pdiag1[lfDgNdx<DIMD1>(dpdsssStateMM, pi+1)];
                                if(!GAP0 && val2 < 0.0f) val2 = gapopencost - val2;
                                mymaxassgn(val1, val2, bk, (int)dpbtckUP);

                                //WRITE: write max value
                                pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, pi)] =
                                    (GAP0 || bk != dpbtckDIAG)? val1: -val1;

                                //WRITE btck
                                btck[pi][i] = bk;

                            }//simd for(pi < DIMD)

                            bottm[lfDgNdx<DIMX>(dpdsssStateMM, i)] =
                                pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, 0)];

                            myswap(pdiag1, pdiag2);
                        }//for(i < ilim)

                        //WRITE resulting TWO DIAGONALS for next-iteration blocks;
                        WriteTwoDiagonals<DIMD,DIMD1,memalignment>(
                            DIMX,
                            x, y, dbstrdst, dbstrlen, dblen, yofff,  tmpdpdiagbuffers,  diag1, diag2);

                        //WRITE the bottom edge of the oblique blocks;
                        WriteBottomEdge<DIMX,memalignment>(
                            x, y, dbstrdst, dbstrlen, dblen, yofff,  tmpdpbotbuffer,  bottm);

                        //WRITE backtracking information
                        WriteBtckInfo<DIMD,DIMX,memalignment>(
                            x, y, qrydst, qrylen, dbstrdst, dbstrlen,  dblen,  btckdata, btck);
                    }//omp for

            //use explicit barrier here
            #pragma omp barrier
        }//for(dd < nblkdiags)
    }//omp parallel
}

// =========================================================================
// Instantiations
// 
#define INSTANTIATE_MpDPHub_ExecDPTFMSSwBtck128xKernel(tpGLOBTFM,tpGAP0,tpUSESS,tpD02IND) \
    template void MpDPHub::ExecDPTFMSSwBtck128xKernel<tpGLOBTFM,tpGAP0,tpUSESS,tpD02IND>( \
        const float gapopencost, const float ssweight, const int stepnumber, \
        const char* const * const __RESTRICT__ querypmbeg, \
        const char* const * const __RESTRICT__ bdbCpmbeg, \
        const float* const __RESTRICT__ tfmmem, \
        const float* const __RESTRICT__ wrkmemaux, \
        float* const __RESTRICT__ tmpdpdiagbuffers, \
        float* const __RESTRICT__ tmpdpbotbuffer, \
        char* const __RESTRICT__ btckdata);

INSTANTIATE_MpDPHub_ExecDPTFMSSwBtck128xKernel(true,false,true,D02IND_DPSCAN);
INSTANTIATE_MpDPHub_ExecDPTFMSSwBtck128xKernel(true,true,false,D02IND_SEARCH);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// ExecDPSSwBtck128xKernel: execute dynamic programming with secondary
// structure and backtracking information using shared memory and
// 128-fold unrolling along the diagonal of dimension MPDP_2DCACHE_DIM;
// USESEQSCORING, template parameter, flag of additionally using residue score matrix;
// gapopencost, gap open cost;
// weight4ss, weight for scoring secondary structure;
// weight4rr, weight for pairwise residue score;
// NOTE: memory pointers should be aligned!
// wrkmemaux, working memory for global flags across kernels;
// tmpdpdiagbuffers, temporary buffers for last calculated diagonal scores;
// tmpdpbotbuffer, temporary buffers for last calculated bottom scores;
// btckdata, backtracking information data;
// 
template<bool USESEQSCORING>
void MpDPHub::ExecDPSSwBtck128xKernel(
    const float gapopencost,
    const float weight4ss,
    const float weight4rr,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* const __RESTRICT__ /*wrkmemaux*/,
    float* const __RESTRICT__ tmpdpdiagbuffers,
    float* const __RESTRICT__ tmpdpbotbuffer,
    char* const __RESTRICT__ btckdata)
{
    enum {
        DIMD = MPDP_2DCACHE_DIM_D,
        DIMD1 = DIMD + 1,
        DIMX = MPDP_2DCACHE_DIM_X,
        DIMDpX = MPDP_2DCACHE_DIM_DpX,
        DIMD_LOG2 = MPDP_2DCACHE_DIM_D_LOG2
    };

    MYMSG("MpDPHub::ExecDPSSwBtck128xKernel", 4);
    // static const std::string preamb = "MpDPHub::ExecDPSSwBtck128xKernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = CuMemoryBase::GetMinMemAlignment();

    //execution configuration for DP:
    //1D thread block processes 2D DP matrix oblique block of dimension 
    //MPDP_2DCACHE_DIM_D x MPDP_2DCACHE_DIM_X;
    const int maxblkdiagelems = GetMaxBlockDiagonalElems(
            dbstr1len_, qystr1len_, MPDP_2DCACHE_DIM_D, MPDP_2DCACHE_DIM_X);
    const int nblocks_x = maxblkdiagelems;
    const int nblocks_y = ndbCstrs_;
    const int nblocks_z = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_z * (size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPDP_2DCACHE_CHSIZE);

    //number of regular DIAGONAL block diagonal series, each of given dimensions;
    //rect coords (x,y) are related to diagonal number d by d=x+y-1;
    int nblkdiags = (int)
        (((dbstr1len_ + qystr1len_) + MPDP_2DCACHE_DIM_X-1) / MPDP_2DCACHE_DIM_X);
    nblkdiags += (int)(qystr1len_ - 1) / MPDP_2DCACHE_DIM_D;

    float diag1[nTDPDiagScoreSubsections * DIMD1];//cache for scores of the 1st diagonal
    float diag2[nTDPDiagScoreSubsections * DIMD1];//last (2nd) diagonal
    float bottm[nTDPDiagScoreSubsections * DIMX];//bottom scores
    char rfnSS[DIMDpX];
    char rfnRE[DIMDpX];
    char qrySS[DIMD];
    char qryRE[DIMD];
    char btck[DIMD][DIMX];//backtracking information

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        private(diag1,diag2,bottm, rfnSS,rfnRE,qrySS,qryRE, btck) \
        shared(GONNET_SCORES, nblkdiags)
    {
        //iterate over olblique block anti-diagonals to perform DP;
        //nblkdiags, total (max) number of anti-diagonals:
        for(int dd = 0; dd < nblkdiags; dd++)
        {
            #pragma omp for collapse(3) schedule(dynamic, chunksize)
            for(int qi = 0; qi < nblocks_z; qi++)
                for(int ri = 0; ri < nblocks_y; ri++)
                    for(int bi = 0; bi < nblocks_x; bi++)
                    {//threads process oblique blocks on anti-diagonals of query-reference pairs
                        //check convergence:
                        // int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                        // bottm[0] = wrkmemaux[mloc0 + ri];//reuse cache
                        // if(((int)(bottm[0])) & (CONVERGED_LOWTMSC_bitval))
                        //     continue;

                        const int qrylen = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                        const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                        const int dbstrlen = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                        const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);

                        //lastydiagnum, last block diagonal serial number along y axis:
                        //each division separates a number of diagonals (nsepds);
                        constexpr int nsepds = 2;//(float)DIMD/(float)DIMX + 1.0f;
                        //the number of the last diagonal starting at x=-DIMD
                        ///int nintdivs = (qrylen-1)>>DIMD_LOG2;//(qrylen-1)/DIMD;
                        ///uint lastydiagnum = nsepds * nintdivs + 1 - 1;//-1 for zero-based indices;
                        int lastydiagnum = ((qrylen-1) >> DIMD_LOG2) * nsepds;

                        // blockIdx.x is block serial number s within diagonal blkdiagnum;
                        // (x,y) is the bottom-left corner (x,y) coordinates for structure dbstrndx
                        int x, y;
                        if(dd <= lastydiagnum) {
                            //x=-!(d%2)w+2ws; y=dw/2+w-sw -1 (-1, zero-based indices); [when w==b]
                            //(b, block's length; w, block's width)
                            x = (2 * bi - (!(dd & 1))) * DIMD;
                            y = ((dd >> 1) + 1 - bi) * DIMD - 1;
                        } else {
                            //x=-w+(d-d_l)w+2ws; y=dw/2+w-sw -1; [when w==b]
                            x = (2 * bi + (dd - lastydiagnum - 1)) * DIMD;
                            y = ((lastydiagnum >> 1) + 1 - bi) * DIMD - 1;
                        }

                        //number of iterations for this block to perform;
                        int ilim = GetMaqxNoIterations(x, y, qrylen, dbstrlen, DIMX);

                        if(y < 0 || qrylen <= (y+1 - DIMD) || 
                           dbstrlen <= x /*+ DIMDpX */ ||
                           ilim < 1)
                            continue;//out of boundaries

                        //READ DATA
                        // int qpos = y - threadIdx.x;//going upwards
                        // //x is now the position this thread will process
                        // x += threadIdx.x;

                        //read ss information
                        if(USESEQSCORING)
                            ReadQryRE<DIMD,PMBSdatalignment>(
                                x, y, qrydst, qrylen,  querypmbeg, qryRE);
                        ReadQrySS<DIMD,PMBSdatalignment>(
                            x, y, qrydst, qrylen,  querypmbeg, qrySS);

                        //db reference structure position corresponding to the oblique block's
                        //bottom-left corner in the buffers dc_pm2dvfields_ (score matrix) 
                        //plus the offset determined by thread id:
                        // int dbpos = x + dbstrdst;//going right
                        int dblen = ndbCposs_ + dbxpad_;
                        //offset (w/o a factor) to the beginning of the data along the y axis wrt query qi: 
                        int yofff = dblen * qi;

                        //read ss information
                        if(USESEQSCORING)
                            ReadRfnRE<DIMDpX,PMBSdatalignment>(
                                x, y, dbstrdst, dbstrlen,  bdbCpmbeg, rfnRE);
                        ReadRfnSS<DIMDpX,PMBSdatalignment>(
                            x, y, dbstrdst, dbstrlen,  bdbCpmbeg, rfnSS);

                        //cache TWO DIAGONALS from the previous (along the x axis) oblique block;
                        //tmpdpdiagbuffers: (1D, along the x axis)

                        ReadTwoDiagonals<DIMD,DIMD1,memalignment>(
                            x, y, dbstrdst, dbstrlen, dblen, yofff,
                            tmpdpdiagbuffers,  diag1, diag2, DPSSDEFINITSCOREVAL);

                        //cache the bottom of the upper oblique blocks;
                        ReadBottomEdge<DIMD,DIMX,memalignment>(
                            x, y, dbstrdst, dbstrlen, dblen, yofff,
                            tmpdpbotbuffer, bottm, DPSSDEFINITSCOREVAL);

                        float *pdiag1 = diag1;
                        float *pdiag2 = diag2;

                        //start calculations for this position with Nx unrolling
                        for(int i = 0; i < ilim/*DIMX*/; i++)
                        {
                            pdiag1[lfDgNdx<DIMD1>(dpdsssStateMM, DIMD)] =
                                bottm[lfDgNdx<DIMX>(dpdsssStateMM, i)];

                            #pragma omp simd
                            for(int pi = 0; pi < DIMD; pi++) {
                                float val1, val2;//, left, up;
                                int bk;

                                val1 = (float)(qrySS[pi] == rfnSS[pi+i]) * weight4ss;
                                if(USESEQSCORING)
                                    val1 += GONNET_SCORES.get(qryRE[pi], rfnRE[pi+i]) * weight4rr;

                                //NOTE: TRICK to implement a special case of DP with affine gap cost scheme:
                                //NOTE: gap extension cost is 0;
                                //NOTE: this version for positive and negative match scores:
                                //NOTE: add and appropriately subtract max possible match score to/from max 
                                //NOTE: value to indicate diagonal direction in alignment;

                                //MM state update (diagonal direction)
                                val2 = pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, pi+1)];
                                //val2 < large_val means diagonal direction, adjust appropriately:
                                if(val2 < DPSSDEFINITSCOREVAL_cmp) val2 -= DPSSDEFINITSCOREVAL;
                                val1 += val2;
                                bk = dpbtckDIAG;

                                //////// SYNC ///////////

                                //IM state update (left direction)
                                val2 = pdiag1[lfDgNdx<DIMD1>(dpdsssStateMM, pi)];
                                //val2 < large_val means diagonal direction, adjust appropriately and add cost:
                                if(val2 < DPSSDEFINITSCOREVAL_cmp) val2 = gapopencost + val2 - DPSSDEFINITSCOREVAL;
                                mymaxassgn(val1, val2, bk, (int)dpbtckLEFT);

                                //MI state update (up direction)
                                val2 = pdiag1[lfDgNdx<DIMD1>(dpdsssStateMM, pi+1)];
                                if(val2 < DPSSDEFINITSCOREVAL_cmp) val2 = gapopencost + val2 - DPSSDEFINITSCOREVAL;
                                mymaxassgn(val1, val2, bk, (int)dpbtckUP);

                                //WRITE: write max value
                                pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, pi)] =
                                    (bk != dpbtckDIAG)? val1: val1 + DPSSDEFINITSCOREVAL;

                                //WRITE btck
                                btck[pi][i] = bk;

                            }//simd for(pi < DIMD)

                            bottm[lfDgNdx<DIMX>(dpdsssStateMM, i)] =
                                pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, 0)];

                            myswap(pdiag1, pdiag2);
                        }//for(i < ilim)

                        //WRITE resulting TWO DIAGONALS for next-iteration blocks;
                        WriteTwoDiagonals<DIMD,DIMD1,memalignment>(
                            DIMX,
                            x, y, dbstrdst, dbstrlen, dblen, yofff,  tmpdpdiagbuffers,  diag1, diag2);

                        //WRITE the bottom edge of the oblique blocks;
                        WriteBottomEdge<DIMX,memalignment>(
                            x, y, dbstrdst, dbstrlen, dblen, yofff,  tmpdpbotbuffer,  bottm);

                        //WRITE backtracking information
                        WriteBtckInfo<DIMD,DIMX,memalignment>(
                            x, y, qrydst, qrylen, dbstrdst, dbstrlen,  dblen,  btckdata, btck);
                    }//omp for

            //use explicit barrier here
            #pragma omp barrier
        }//for(dd < nblkdiags)
    }//omp parallel
}

// =========================================================================
// Instantiations
// 
#define INSTANTIATE_MpDPHub_ExecDPSSwBtck128xKernel(tpUSESEQSCORING) \
    template void MpDPHub::ExecDPSSwBtck128xKernel<tpUSESEQSCORING>( \
        const float gapopencost, const float weight4ss, const float weight4rr, \
        const char* const * const __RESTRICT__ querypmbeg, \
        const char* const * const __RESTRICT__ bdbCpmbeg, \
        const float* const __RESTRICT__ wrkmemaux, \
        float* const __RESTRICT__ tmpdpdiagbuffers, \
        float* const __RESTRICT__ tmpdpbotbuffer, \
        char* const __RESTRICT__ btckdata);

INSTANTIATE_MpDPHub_ExecDPSSwBtck128xKernel(false);
INSTANTIATE_MpDPHub_ExecDPSSwBtck128xKernel(true);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// #define MPDPSSLOCAL_TESTPRINT 0
// -------------------------------------------------------------------------
// ExecDPSSLocal128xKernel: execute dynamic programming for local alignment
// using secondary structure information with N-fold unrolling along the 
// diagonal of dimension MPDP_2DCACHE_DIM;
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
void MpDPHub::ExecDPSSLocal128xKernel(
    const float gapcost,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* const __RESTRICT__ wrkmemaux,
    float* const __RESTRICT__ tmpdpdiagbuffers,
    float* const __RESTRICT__ tmpdpbotbuffer,
    char* const __RESTRICT__ dpscoremtx)
{
    enum {
        DIMD = MPDP_2DCACHE_DIM_D,
        DIMD1 = DIMD + 1,
        DIMX = MPDP_2DCACHE_DIM_X,
        DIMDpX = MPDP_2DCACHE_DIM_DpX,
        DIMD_LOG2 = MPDP_2DCACHE_DIM_D_LOG2
    };

    MYMSG("MpDPHub::ExecDPSSLocal128xKernel", 4);
    // static const std::string preamb = "MpDPHub::ExecDPSSLocal128xKernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = CuMemoryBase::GetMinMemAlignment();

    //execution configuration for DP:
    //1D thread block processes 2D DP matrix oblique block of dimension 
    //MPDP_2DCACHE_DIM_D x MPDP_2DCACHE_DIM_X;
    const int maxblkdiagelems = GetMaxBlockDiagonalElems(
            dbstr1len_, qystr1len_, MPDP_2DCACHE_DIM_D, MPDP_2DCACHE_DIM_X);
    const int nblocks_x = maxblkdiagelems;
    const int nblocks_y = ndbCstrs_;
    const int nblocks_z = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_z * (size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPDP_2DCACHE_CHSIZE);

    //number of regular DIAGONAL block diagonal series, each of given dimensions;
    //rect coords (x,y) are related to diagonal number d by d=x+y-1;
    int nblkdiags = (int)
        (((dbstr1len_ + qystr1len_) + MPDP_2DCACHE_DIM_X-1) / MPDP_2DCACHE_DIM_X);
    nblkdiags += (int)(qystr1len_ - 1) / MPDP_2DCACHE_DIM_D;

    float diag1[nTDPDiagScoreSubsections * DIMD1];//cache for scores of the 1st diagonal
    float diag2[nTDPDiagScoreSubsections * DIMD1];//last (2nd) diagonal
    float bottm[nTDPDiagScoreSubsections * DIMX];//bottom scores
    char rfnSS[DIMDpX];
    char qrySS[DIMD];
    char dpsc[DIMD][DIMX];//dp scores

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        private(diag1,diag2,bottm, rfnSS,qrySS, dpsc) \
        shared(nblkdiags)
    {
        //iterate over olblique block anti-diagonals to perform DP;
        //nblkdiags, total (max) number of anti-diagonals:
        for(int dd = 0; dd < nblkdiags; dd++)
        {
            #pragma omp for collapse(3) schedule(dynamic, chunksize)
            for(int qi = 0; qi < nblocks_z; qi++)
                for(int ri = 0; ri < nblocks_y; ri++)
                    for(int bi = 0; bi < nblocks_x; bi++)
                    {//threads process oblique blocks on anti-diagonals of query-reference pairs
                        //check convergence:
                        int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                        diag1[0] = wrkmemaux[mloc0 + ri];//reuse cache
                        if(((int)(diag1[0])) &
                           (CONVERGED_SCOREDP_bitval|CONVERGED_NOTMPRG_bitval|CONVERGED_LOWTMSC_bitval))
                            continue;

                        const int qrylen = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                        const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                        const int dbstrlen = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                        const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);

                        //lastydiagnum, last block diagonal serial number along y axis:
                        //each division separates a number of diagonals (nsepds);
                        constexpr int nsepds = 2;//(float)DIMD/(float)DIMX + 1.0f;
                        //the number of the last diagonal starting at x=-DIMD
                        ///int nintdivs = (qrylen-1)>>DIMD_LOG2;//(qrylen-1)/DIMD;
                        ///uint lastydiagnum = nsepds * nintdivs + 1 - 1;//-1 for zero-based indices;
                        int lastydiagnum = ((qrylen-1) >> DIMD_LOG2) * nsepds;

                        // blockIdx.x is block serial number s within diagonal blkdiagnum;
                        // (x,y) is the bottom-left corner (x,y) coordinates for structure dbstrndx
                        int x, y;
                        if(dd <= lastydiagnum) {
                            //x=-!(d%2)w+2ws; y=dw/2+w-sw -1 (-1, zero-based indices); [when w==b]
                            //(b, block's length; w, block's width)
                            x = (2 * bi - (!(dd & 1))) * DIMD;
                            y = ((dd >> 1) + 1 - bi) * DIMD - 1;
                        } else {
                            //x=-w+(d-d_l)w+2ws; y=dw/2+w-sw -1; [when w==b]
                            x = (2 * bi + (dd - lastydiagnum - 1)) * DIMD;
                            y = ((lastydiagnum >> 1) + 1 - bi) * DIMD - 1;
                        }

                        //number of iterations for this block to perform;
                        int ilim = GetMaqxNoIterations(x, y, qrylen, dbstrlen, DIMX);

                        if(y < 0 || qrylen <= (y+1 - DIMD) || 
                           dbstrlen <= x /*+ DIMDpX */ ||
                           ilim < 1)
                            continue;//out of boundaries

                        //READ secondary structure assignment
                        // int qpos = y - threadIdx.x;//going upwards
                        // //x is now the position this thread will process
                        // x += threadIdx.x;

                        ReadQrySS<DIMD,PMBSdatalignment>(
                            x, y, qrydst, qrylen,  querypmbeg, qrySS);

                        //db reference structure position corresponding to the oblique block's
                        //bottom-left corner in the buffers dc_pm2dvfields_ (score matrix) 
                        //plus the offset determined by thread id:
                        // int dbpos = x + dbstrdst;//going right
                        int dblen = ndbCposs_ + dbxpad_;
                        //offset (w/o a factor) to the beginning of the data along the y axis wrt query qi: 
                        int yofff = dblen * qi;

                        ReadRfnSS<DIMDpX,PMBSdatalignment>(
                            x, y, dbstrdst, dbstrlen,  bdbCpmbeg, rfnSS);

                        //cache TWO DIAGONALS from the previous (along the x axis) oblique block;
                        //tmpdpdiagbuffers: (1D, along the x axis)

                        ReadTwoDiagonals<DIMD,DIMD1,memalignment>(
                            x, y, dbstrdst, dbstrlen, dblen, yofff,  tmpdpdiagbuffers,  diag1, diag2);

                        //cache the bottom of the upper oblique blocks;
                        ReadBottomEdge<DIMD,DIMX,memalignment>(
                            x, y, dbstrdst, dbstrlen, dblen, yofff,  tmpdpbotbuffer,  bottm);

                        float *pdiag1 = diag1;
                        float *pdiag2 = diag2;

                        //start calculations for this position with Nx unrolling
                        for(int i = 0; i < ilim/*DIMX*/; i++)
                        {
                            pdiag1[lfDgNdx<DIMD1>(dpdsssStateMM, DIMD)] =
                                bottm[lfDgNdx<DIMX>(dpdsssStateMM, i)];

                            #pragma omp simd
                            for(int pi = 0; pi < DIMD; pi++) {
                                float val1, val2;

                                //NOTE: match score:
                                val1 = (float)((qrySS[pi] == rfnSS[pi+i]) * 2) - 1.0f;

                                //MM state update (diagonal direction)
                                val1 += pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, pi+1)];
                                if(val1 < 0.0f) val1 = 0.0f;
                                // if(val1) bk = dpbtckDIAG;

                                //////// SYNC ///////////

                                //IM state update (left direction)
                                val2 = pdiag1[lfDgNdx<DIMD1>(dpdsssStateMM, pi)] + gapcost;
                                if(val1 < val2) val1 = val2;
                                // mymaxassgn(val1, val2, bk, (int)dpbtckLEFT);

                                //MI state update (up direction)
                                val2 = pdiag1[lfDgNdx<DIMD1>(dpdsssStateMM, pi+1)] + gapcost;
                                if(val1 < val2) val1 = val2;
                                // mymaxassgn(val1, val2, bk, (int)dpbtckUP);

                                //WRITE: write max value
                                pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, pi)] = val1;

                                //WRITE
                                dpsc[pi][i] = (char)(((unsigned int)val1) & 255);//val1%256

                            }//simd for(pi < DIMD)

                            bottm[lfDgNdx<DIMX>(dpdsssStateMM, i)] =
                                pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, 0)];

#ifdef MPDPSSLOCAL_TESTPRINT
                            if(ri==MPDPSSLOCAL_TESTPRINT)
                                for(int pi = 0; pi < DIMD; pi++)
                                printf(" d=%u(%u) s=%u i%02d/%u (t%02u): len= %d addr= %u SC= <> (yx: %d,%d) "
                                        "MM= %.6f    >qSS= %d   dSS= %d\n",
                                        dd, lastydiagnum, bi, i, ilim, pi,
                                        dbstrlen, dbstrdst,  y-pi, x+pi+i,
                                        pdiag2[lfDgNdx<DIMD1>(dpdsssStateMM, pi)], qrySS[pi], rfnSS[pi+i]
                                );
#endif
                            myswap(pdiag1, pdiag2);
                        }//for(i < ilim)

#ifdef MPDPSSLOCAL_TESTPRINT
                        if(ri==MPDPSSLOCAL_TESTPRINT)
                            for(int pi = 0; pi < DIMD; pi++)
                            printf(" >>> d=%u(%u) s=%u (t%02u): len= %d wrt= %d xpos= %d\n", 
                                dd, lastydiagnum, bi, pi, dbstrlen,
                                x+DIMX-1+pi<dbstrlen, x+DIMX-1+pi);
#endif

                        //WRITE resulting TWO DIAGONALS for next-iteration blocks;
                        WriteTwoDiagonals<DIMD,DIMD1,memalignment>(
                            DIMX,
                            x, y, dbstrdst, dbstrlen, dblen, yofff,  tmpdpdiagbuffers,  diag1, diag2);

                        //WRITE the bottom edge of the oblique blocks;
                        WriteBottomEdge<DIMX,memalignment>(
                            x, y, dbstrdst, dbstrlen, dblen, yofff,  tmpdpbotbuffer,  bottm);

                        //WRITE backtracking information
                        WriteBtckInfo<DIMD,DIMX,memalignment>(
                            x, y, qrydst, qrylen, dbstrdst, dbstrlen,  dblen,  dpscoremtx, dpsc);
                    }//omp for

            //use explicit barrier here
            #pragma omp barrier
        }//for(dd < nblkdiags)
    }//omp parallel
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// #define MPDP_MTCH_ALN_TESTPRINT 0
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// BtckToMatched128xKernel: backtrack to produce match (aligned) positions;
// NOTE: ANCHORRGN, template parameter, anchor region is in use:
// NOTE: Regions outside the anchor are not explored,
// NOTE: decreasing computational complexity;
// NOTE: BANDED, template parameter, banded alignment;
// stepnumber, step number which also corresponds to the superposition
// variant used;
// NOTE: memory pointers should be aligned!
// btckdata, backtracking information data;
// wrkmemaux, auxiliary working memory;
// tmpdpalnpossbuffer, destination of copied coordinates;
//
template<bool ANCHORRGN, bool BANDED>
void MpDPHub::BtckToMatched128xKernel(
    const uint stepnumber,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const char* const __RESTRICT__ btckdata,
    float* const __RESTRICT__ wrkmemaux,
    float* const __RESTRICT__ tmpdpalnpossbuffer)
{
    enum {DIMX = MPDP_MATCHED_DIM_X};
    enum {bmQRYNDX, bmRFNNDX, bmTotal};

    MYMSG("MpDPHub::BtckToMatched128xKernel", 4);
    // static const std::string preamb = "MpDPHub::BtckToMatched128xKernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = mycemin((size_t)PMBSdatalignment, CuMemoryBase::GetMinMemAlignment());

    //execution configuration for DP:
    //one thread processes multiple query-reference structure pairs; 
    const int nblocks_x = ndbCstrs_;
    const int nblocks_y = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPDP_MATCHED_CHSIZE);

    int poss[bmTotal][DIMX];//query-reference match positions

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        private(poss)
    {
        #pragma omp for collapse(2) schedule(dynamic, chunksize)
        for(int qi = 0; qi < nblocks_y; qi++)
            for(int ri = 0; ri < nblocks_x; ri++)
            {//threads backtrack for query-reference pairs
                //check convergence:
                int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                int mloc = ((qi * maxnsteps_ + stepnumber) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                poss[1][0] = poss[1][1] = wrkmemaux[mloc0 + ri];//reuse cache: float->int
                if(stepnumber != 0) poss[1][1] = wrkmemaux[mloc + ri];//float->int
                if((poss[1][0] & (CONVERGED_LOWTMSC_bitval)) ||
                   (poss[1][1] & (CONVERGED_SCOREDP_bitval | CONVERGED_NOTMPRG_bitval | CONVERGED_LOWTMSC_bitval)))
                {   //NOTE: set alignment length (#matched/aligned positions) at pos==0 to 0 to halt refinement:
                    mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvNAlnPoss) * ndbCstrs_;
                    wrkmemaux[mloc0 + ri] = 0.0f;//alnlen;
                    continue;
                }

                const int qrylen = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                const int dbstrlen = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);

                //terminal cell is the bottom-right-most cell
                int y = qrylen - 1, x = dbstrlen - 1;

#ifdef MPDP_MTCH_ALN_TESTPRINT
                if((MPDP_MTCH_ALN_TESTPRINT>=0)? ri==MPDP_MTCH_ALN_TESTPRINT: 1)
                printf(" MTCH: bid= %u tid= %u: len= %d addr= %u "
                    "qrypos= %d rfnpos= %d fraglen= %d (y= %d x= %d)\n",
                    ri, 0, dbstrlen, dbstrdst, qrypos, rfnpos, 0/*fraglen*/, y, x
                );
#endif
                int alnlen = 0;
                char btck = dpbtckDIAG;
                const int dblen = ndbCposs_ + dbxpad_;
                //offset to the beginning of the data along the y axis wrt query qi:
                //NOTE: write alignment at pos==0 for refinement to follow!
                const int yofff = (qi * maxnsteps_ + 0/*sfragfct*/) * dblen * nTDPAlignedPoss;
                //const int yofff = (qi * maxnsteps_ + stepnumber/*sfragfct*/) * dblen * nTDPAlignedPoss;

                //backtrack over the alignment
                while(btck != dpbtckSTOP) {
                    int ndx = 0;
                    //record matched positions
                    for(; ndx < DIMX;) {
                        if(x < 0 || y < 0) {
                            btck = dpbtckSTOP;
                            break;
                        }
                        int qpos = (qrydst + y) * dblen + dbstrdst + x;
                        btck = btckdata[qpos];//READ
                        if(btck == dpbtckSTOP)
                            break;
                        if(btck == dpbtckUP) {
                            y--; 
                            continue; 
                        }
                        else if(btck == dpbtckLEFT) { 
                            x--; 
                            continue; 
                        }
                        //(btck == dpbtckDIAG)
                        poss[bmQRYNDX][ndx] = y;
                        poss[bmRFNNDX][ndx] = x;
                        x--; y--; ndx++;
                    }

#ifdef MPDP_MTCH_ALN_TESTPRINT
                    if((MPDP_MTCH_ALN_TESTPRINT>=0)? ri==MPDP_MTCH_ALN_TESTPRINT: 1){
                        for(int ii=0; ii<ndx; ii++) printf(" %5d", poss[bmQRYNDX][ii]);
                        printf("\n");
                        for(int ii=0; ii<ndx; ii++) printf(" %5d", poss[bmRFNNDX][ii]);
                        printf("\n\n");
                    }
#endif
                    //write the coordinates of the matched positions to memory
                    #pragma omp simd aligned(querypmbeg,bdbCpmbeg,tmpdpalnpossbuffer:memalignment)
                    for(int pi = 0; pi < ndx; pi++) {
                        //READ coordinates
                        int qrypos = qrydst + poss[bmQRYNDX][pi];
                        int rfnpos = dbstrdst + poss[bmRFNNDX][pi];

                        float qx = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DX>(querypmbeg, qrypos);
                        float qy = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DY>(querypmbeg, qrypos);
                        float qz = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DZ>(querypmbeg, qrypos);

                        float rx = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DX>(bdbCpmbeg, rfnpos);
                        float ry = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DY>(bdbCpmbeg, rfnpos);
                        float rz = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DZ>(bdbCpmbeg, rfnpos);

                        //WRITE coordinates in the order reverse to alignment itself;
                        //take into account #positions written already (alnlen);
                        tmpdpalnpossbuffer[yofff + dbstrdst + alnlen  + dpapsQRYx * dblen + pi] = qx;
                        tmpdpalnpossbuffer[yofff + dbstrdst + alnlen  + dpapsQRYy * dblen + pi] = qy;
                        tmpdpalnpossbuffer[yofff + dbstrdst + alnlen  + dpapsQRYz * dblen + pi] = qz;

                        tmpdpalnpossbuffer[yofff + dbstrdst + alnlen  + dpapsRFNx * dblen + pi] = rx;
                        tmpdpalnpossbuffer[yofff + dbstrdst + alnlen  + dpapsRFNy * dblen + pi] = ry;
                        tmpdpalnpossbuffer[yofff + dbstrdst + alnlen  + dpapsRFNz * dblen + pi] = rz;
                    }

                    //#aligned positions have increased by ndx
                    alnlen += ndx;
                }//while(btck != dpbtckSTOP)

                //WRITE #matched (aligned) positions 
                //NOTE: write alignment length at pos==0 for refinement to follow
                mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvNAlnPoss) * ndbCstrs_;
                // mloc0 = ((qi * maxnsteps_ + stepnumber) * nTAuxWorkingMemoryVars + tawmvNAlnPoss) * ndbCstrs_;
                wrkmemaux[mloc0 + ri] = alnlen;

#ifdef MPDP_MTCH_ALN_TESTPRINT
                if((MPDP_MTCH_ALN_TESTPRINT>=0)? ri==MPDP_MTCH_ALN_TESTPRINT: 1)
                printf(" MTCH (pronr=%u): y= %d x= %d alnlen= %d qrylen= %d dbstrlen= %d\n\n"
                    ri, y, x, alnlen, qrylen,dbstrlen
                );
#endif
            }//omp for
    }//omp parallel
}

// =========================================================================
// Instantiations
// 
#define INSTANTIATE_MpDPHub_BtckToMatched128xKernel(tpANCHORRGN,tpBANDED) \
    template void MpDPHub::BtckToMatched128xKernel<tpANCHORRGN,tpBANDED>( \
        const uint stepnumber, \
        const char* const * const __RESTRICT__ querypmbeg, \
        const char* const * const __RESTRICT__ bdbCpmbeg, \
        const char* const __RESTRICT__ btckdata, \
        float* const __RESTRICT__ wrkmemaux, \
        float* const __RESTRICT__ tmpdpalnpossbuffer);

INSTANTIATE_MpDPHub_BtckToMatched128xKernel(false,false);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// #define MPDP_CONSTRAINED_MTCH_TESTPRINT 0
// -------------------------------------------------------------------------
// ConstrainedBtckToMatched128xKernel: backtrack the coordinates of matched 
// (aligned) positions within a given distance threshold to destination 
// location for final refinement;
// NOTE: memory pointers should be aligned!
// btckdata, backtracking information data;
// tfmmem, transformation matrix address space;
// wrkmemaux, auxiliary working memory;
// tmpdpalnpossbuffer, destination of the coordinates of matched positions;
//
void MpDPHub::ConstrainedBtckToMatched128xKernel(
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const char* const __RESTRICT__ btckdata,
    const float* const __RESTRICT__ tfmmem,
    float* const __RESTRICT__ wrkmemaux,
    float* const __RESTRICT__ tmpdpalnpossbuffer)
{
    enum {DIMX = MPDP_CONST_MATCH_DIM_X};
    enum {bmQRYNDX, bmRFNNDX, bmTotal};
    //cache for query and reference coordinates:
    //position included (flag), position accumulated index, distance2, total:
    enum {ccPOSINC = nTDPAlignedPoss, ccPOSNDX, ccPSDST2, ccTotal};

    MYMSG("MpDPHub::ConstrainedBtckToMatched128xKernel", 4);
    // static const std::string preamb = "MpDPHub::ConstrainedBtckToMatched128xKernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = mycemin((size_t)PMBSdatalignment, CuMemoryBase::GetMinMemAlignment());

    //execution configuration for DP:
    //one thread processes multiple query-reference structure pairs; 
    const int nblocks_x = ndbCstrs_;
    const int nblocks_y = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPDP_MATCHED_CHSIZE);

    int poss[bmTotal][DIMX];//query-reference match positions
    float crds[ccTotal][DIMX];//coordinates and indices
    float tfm[nTTranformMatrix];//transformation matrix
    float tmp[DIMX];//temporary array

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        private(poss,crds,tfm,tmp)
    {
        #pragma omp for collapse(2) schedule(dynamic, chunksize)
        for(int qi = 0; qi < nblocks_y; qi++)
            for(int ri = 0; ri < nblocks_x; ri++)
            {//threads backtrack for query-reference pairs
                //check convergence:
                int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                poss[1][0] = wrkmemaux[mloc0 + ri];//reuse cache: float->int
                if(poss[1][0] & (CONVERGED_LOWTMSC_bitval))
                    continue;

                const int qrylen = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                const int dbstrlen = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);

                //get the globally best transformation matrix for a pair:
                mloc0 = (qi * ndbCstrs_ + ri) * nTTranformMatrix;
                #pragma omp simd aligned(tfmmem:memalignment)
                for(int f = 0; f < nTTranformMatrix; f++)
                    tfm[f] = tfmmem[mloc0 + f];

                //terminal cell is the bottom-right-most cell
                int y = qrylen - 1, x = dbstrlen - 1;

                int alnlen = 0;
                char btck = dpbtckDIAG;
                const int dblen = ndbCposs_ + dbxpad_;
                //offset to the beginning of the data along the y axis wrt query qi:
                //NOTE: write alignment at pos==0 for refinement to follow!
                const int yofff = (qi * maxnsteps_ + 0/*sfragfct*/) * dblen * nTDPAlignedPoss;
                //offset for writing distance and positional data for producing final alignments:
                const int yoff1 = (qi * maxnsteps_ + 1/*sfragfct*/) * dblen * nTDPAlignedPoss;
                const float d82 = GetD82(qrylen, dbstrlen);

#ifdef MPDP_CONSTRAINED_MTCH_TESTPRINT
                if((MPDP_CONSTRAINED_MTCH_TESTPRINT>=0)? ri==MPDP_CONSTRAINED_MTCH_TESTPRINT: 1)
                printf(" CNTMTCH: bid= %u: qlen= %d qaddr= %u len= %d addr= %u (y= %d x= %d) d82= %.2f\n",
                    ri,qrylen,qrydst,dbstrlen,dbstrdst,y,x,d82
                );
#endif
                //backtrack over the alignment
                while(btck != dpbtckSTOP) {
                    int ndx = 0;
                    //record matched positions
                    for(; ndx < DIMX;) {
                        if(x < 0 || y < 0) {
                            btck = dpbtckSTOP;
                            break;
                        }
                        int qpos = (qrydst + y) * dblen + dbstrdst + x;
                        btck = btckdata[qpos];//READ
                        if(btck == dpbtckSTOP)
                            break;
                        if(btck == dpbtckUP) {
                            y--; 
                            continue; 
                        }
                        else if(btck == dpbtckLEFT) { 
                            x--; 
                            continue; 
                        }
                        //(btck == dpbtckDIAG)
                        poss[bmQRYNDX][ndx] = y;
                        poss[bmRFNNDX][ndx] = x;
                        x--; y--; ndx++;
                    }

#ifdef MPDP_CONSTRAINED_MTCH_TESTPRINT
                    if((MPDP_CONSTRAINED_MTCH_TESTPRINT>=0)? ri==MPDP_CONSTRAINED_MTCH_TESTPRINT: 1){
                        for(int ii=0; ii<ndx; ii++) printf(" %5d", poss[bmQRYNDX][ii]); printf("\n");
                        for(int ii=0; ii<ndx; ii++) printf(" %5d", poss[bmRFNNDX][ii]); printf("\n\n");
                    }
#endif
                    //READ the coordinates of matched positions
                    #pragma omp simd aligned(querypmbeg,bdbCpmbeg:memalignment)
                    for(int pi = 0; pi < ndx; pi++) {
                        int qrypos = qrydst + poss[bmQRYNDX][pi];
                        int rfnpos = dbstrdst + poss[bmRFNNDX][pi];

                        crds[dpapsQRYx][pi] = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DX>(querypmbeg, qrypos);
                        crds[dpapsQRYy][pi] = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DY>(querypmbeg, qrypos);
                        crds[dpapsQRYz][pi] = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DZ>(querypmbeg, qrypos);

                        crds[dpapsRFNx][pi] = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DX>(bdbCpmbeg, rfnpos);
                        crds[dpapsRFNy][pi] = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DY>(bdbCpmbeg, rfnpos);
                        crds[dpapsRFNz][pi] = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DZ>(bdbCpmbeg, rfnpos);
                    }

                    //calculate distances
                    #pragma omp simd
                    for(int pi = 0; pi < ndx; pi++) {
                        float dst2 = transform_and_distance2(tfm,
                            crds[dpapsQRYx][pi], crds[dpapsQRYy][pi], crds[dpapsQRYz][pi],
                            crds[dpapsRFNx][pi], crds[dpapsRFNy][pi], crds[dpapsRFNz][pi]);
                        crds[ccPOSINC][pi] = crds[ccPOSNDX][pi] = (dst2 <= d82);
                        crds[ccPSDST2][pi] = (dst2);
                    }

                    //calculate the inclusive prefix sum of inclusion flags;
                    //this also gives the accumulated indices of included aligned pairs
                    //TODO: change 1st agrument to the real size;
                    mysimdincprefixsum<DIMX>(DIMX, crds[ccPOSNDX], tmp);
                    #pragma omp simd
                    for(int pi = 0; pi < ndx; pi++) {
                        //posndx > 0 always where posinc == 1:
                        crds[ccPOSNDX][pi] -= 1.0f;
                    }

#ifdef MPDP_CONSTRAINED_MTCH_TESTPRINT
                    if((MPDP_CONSTRAINED_MTCH_TESTPRINT>=0)? ri==MPDP_CONSTRAINED_MTCH_TESTPRINT: 1){
                        for(int ii=0; ii<ndx; ii++) printf(" %5.0f", crds[ccPOSINC][ii]); printf("*\n");
                        for(int ii=0; ii<ndx; ii++) printf(" %5.0f", crds[ccPOSNDX][ii]); printf("*\n\n");
                    }
#endif
                    //write the coordinates of the matched positions to memory
                    #pragma omp simd aligned(tmpdpalnpossbuffer:memalignment)
                    for(int pi = 0; pi < ndx; pi++) {
                        const int tdndx = crds[ccPOSNDX][pi];
                        const int tdinc = crds[ccPOSINC][pi];
                        //WRITE coordinates in the order reverse to alignment itself;
                        //take into account #positions written already (alnlen);
                        //write only selected pairs
                        if(tdinc) {
                            tmpdpalnpossbuffer[yofff + dbstrdst + alnlen  + dpapsQRYx * dblen + tdndx] = crds[dpapsQRYx][pi];
                            tmpdpalnpossbuffer[yofff + dbstrdst + alnlen  + dpapsQRYy * dblen + tdndx] = crds[dpapsQRYy][pi];
                            tmpdpalnpossbuffer[yofff + dbstrdst + alnlen  + dpapsQRYz * dblen + tdndx] = crds[dpapsQRYz][pi];
                            tmpdpalnpossbuffer[yofff + dbstrdst + alnlen  + dpapsRFNx * dblen + tdndx] = crds[dpapsRFNx][pi];
                            tmpdpalnpossbuffer[yofff + dbstrdst + alnlen  + dpapsRFNy * dblen + tdndx] = crds[dpapsRFNy][pi];
                            tmpdpalnpossbuffer[yofff + dbstrdst + alnlen  + dpapsRFNz * dblen + tdndx] = crds[dpapsRFNz][pi];

                            tmpdpalnpossbuffer[yoff1 + dbstrdst + alnlen  + 0 * dblen + tdndx] = crds[ccPSDST2][pi];
                            tmpdpalnpossbuffer[yoff1 + dbstrdst + alnlen  + 1 * dblen + tdndx] = poss[bmQRYNDX][pi];
                            tmpdpalnpossbuffer[yoff1 + dbstrdst + alnlen  + 2 * dblen + tdndx] = poss[bmRFNNDX][pi];
                        }
                    }

#ifdef MPDP_CONSTRAINED_MTCH_TESTPRINT
int ps1 = yoff1 + dbstrdst + alnlen;
if((MPDP_CONSTRAINED_MTCH_TESTPRINT>=0)? ri==MPDP_CONSTRAINED_MTCH_TESTPRINT: 1){
    for(int ii=0; ii<ndx; ii++) printf(" %5.0f", crds[ccPOSINC][ii]); printf("+\n");
    for(int ii=0; ii<ndx; ii++) printf(" %5.2f", tmpdpalnpossbuffer[ps1+(int)crds[ccPOSNDX][ii]]); printf("+\n");
    for(int ii=0; ii<ndx; ii++) printf(" %5.0f", tmpdpalnpossbuffer[ps1+dblen+(int)crds[ccPOSNDX][ii]]); printf("+\n");
    for(int ii=0; ii<ndx; ii++) printf(" %5.0f", tmpdpalnpossbuffer[ps1+2*dblen+(int)crds[ccPOSNDX][ii]]); printf("+\n\n\n\n");
}
#endif
                    //#selected aligned positions is at (ndx-1) as a result of the prefix sum
                    if(0 < ndx) alnlen += (crds[ccPOSNDX][ndx-1] + 1.0f);

                }//while(btck != dpbtckSTOP)

                //WRITE #selected matched (aligned) positions 
                mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvNAlnPoss) * ndbCstrs_;
                wrkmemaux[mloc0 + ri] = alnlen;

#ifdef MPDP_CONSTRAINED_MTCH_TESTPRINT
                if((MPDP_CONSTRAINED_MTCH_TESTPRINT>=0)? ri==MPDP_CONSTRAINED_MTCH_TESTPRINT: 1)
                printf(" CNTMTCH (pronr=%u): y= %d x= %d alnlen= %d qrylen= %d dbstrlen= %d\n\n",
                    ri, y, x, alnlen, qrylen, dbstrlen
                );
#endif
            }//omp for
    }//omp parallel
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// #define MPDP_PRODUCTION_ALN_TESTPRINT 0
// -------------------------------------------------------------------------
// ProductionMatchToAlignment128xKernel: produce the final alignment with 
// accompanying information from the given match (aligned) positions; 
// using Nx unrolling;
// nodeletions, flag of not including deletion positions in alignments;
// d2equiv, squared distance threshold for structural equivalence; 
// NOTE: memory pointers should be aligned!
// tmpdpalnpossbuffer, coordinates of matched positions;
// wrkmemaux, auxiliary working memory;
// alndatamem, memory for full alignment information;
// alnsmem, memory for output full alignments;
// 
void MpDPHub::ProductionMatchToAlignment128xKernel(
    const bool nodeletions,
    const float d2equiv,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* const __RESTRICT__ tmpdpalnpossbuffer,
    const float* const __RESTRICT__ wrkmemaux,
    float* const __RESTRICT__ alndatamem,
    char* const __RESTRICT__ alnsmem)
{
    enum {DIMX = MPDP_PRODUCTION_ALN_DIM_X};
    enum {bmQRDST2, bmQRYNDX, bmRFNNDX, bmTotal};

    MYMSG("MpDPHub::ProductionMatchToAlignment128xKernel", 4);
    // static const std::string preamb = "MpDPHub::ProductionMatchToAlignment128xKernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = mycemin((size_t)PMBSdatalignment, CuMemoryBase::GetMinMemAlignment());

    //execution configuration for DP:
    //one thread processes multiple query-reference structure pairs; 
    const int nblocks_x = ndbCstrs_;
    const int nblocks_y = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPDP_PRODUCTION_ALN_CHSIZE);

    float poss[bmTotal][DIMX];//query-reference match positions
    float outSts[nTDP2OutputAlnDataPart1End];//alignment statistics
    int outAln[nTDP2OutputAlignmentSSS][DIMX];//alignment
    int tmp[DIMX];//temporary array

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        private(poss,outSts,outAln,tmp)
    {
        #pragma omp for collapse(2) schedule(dynamic, chunksize)
        for(int qi = 0; qi < nblocks_y; qi++)
            for(int ri = 0; ri < nblocks_x; ri++)
            {//threads backtrack for query-reference pairs
                //check convergence:
                int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                outAln[0][0] = wrkmemaux[mloc0 + ri];//reuse cache: float->int
                if(outAln[0][0] & (CONVERGED_LOWTMSC_bitval))
                    continue;

                const int qrylen = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                // const int dbstrlen = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);

                //NOTE: #matched positions tawmvNAlnPoss written at sfragfct==0:
                mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                const int mtchlen = wrkmemaux[mloc0 + tawmvNAlnPoss * ndbCstrs_ + ri];

                int falnlen = 0, ppaln = 0;
                //NOTE: assign big numbers to qp1 and rp1 (>possible lengths) to
                //correctly process the first aligned position:
                int qp0, rp0, qp1 = 999999, rp1 = 999999;

                //#identities, #structurally equivalent positions, #gaps:
                int idts = 0, psts = 0, gaps = 0;
                uint bcrds = 0, ecrds = 0;//beginning and end positions
                const int dblen = ndbCposs_ + dbxpad_;
                //offset to the beginning of the data along the y axis wrt query qi: 
                const int yoff1 = (qi * maxnsteps_ + 1/*sfragfct*/) * dblen * nTDPAlignedPoss;
                //offset for alignments
                const int alnofff = (qi * ndbCposs_ + ndbCstrs_ * (qrydst + qi)) * nTDP2OutputAlignmentSSS;
                //aln length for query qi across all references:
                const int dbalnlen = ndbCposs_ + ndbCstrs_ * (qrylen + 1);
                //end alignment position for query qi:
                //const int dbalnend = dbstrdst + ri * (qrylen + 1) + qrylen + dbstrlen;
                //start alignment position for query qi:
                const int dbalnbeg = dbstrdst + ri * (qrylen + 1);

                //unroll along data blocks:
                for(int relpos = 0; relpos < mtchlen; relpos += DIMX)
                {
                    //starting position in tmpdpalnpossbuffer for a pair:
                    //NOTE: alignment written in reverse order:
                    const int pos = yoff1 + dbstrdst + mtchlen-1 - (relpos);

                    //READ distances and matched positions
                    int piend = mymin(mtchlen - relpos, (int)DIMX);
                    #pragma omp simd collapse(2) aligned(tmpdpalnpossbuffer:memalignment)
                    for(int f = 0; f < bmTotal; f++)
                        for(int pi = 0; pi < piend; pi++) {
                            poss[f][pi] = tmpdpalnpossbuffer[pos - pi + f * dblen];
                        }

#ifdef MPDP_PRODUCTION_ALN_TESTPRINT
                    if((MPDP_PRODUCTION_ALN_TESTPRINT>=0)? ri==MPDP_PRODUCTION_ALN_TESTPRINT: 1){
                        for(int ii=0; ii<DIMX && relpos+ii<mtchlen; ii++) printf(" %5.1f", poss[bmQRDST2][ii]); printf("\n");
                        for(int ii=0; ii<DIMX && relpos+ii<mtchlen; ii++) printf(" %5.0f", poss[bmQRYNDX][ii]); printf("\n");
                        for(int ii=0; ii<DIMX && relpos+ii<mtchlen; ii++) printf(" %5.0f", poss[bmRFNNDX][ii]); printf("\n\n");
                    }
#endif
                    //produce alignment for the fragment of matched positions read 
                    for(int p = relpos-1, pp = -1; p < mtchlen && pp < DIMX; p++, pp++)
                    {
                        qp0 = qp1; rp0 = rp1;
                        qp1 = qp0; rp1 = rp0;

                        if(p+1 < mtchlen && pp+1 < DIMX) {
                            qp1 = poss[bmQRYNDX][pp+1];//query position
                            rp1 = poss[bmRFNNDX][pp+1];//reference position
                        }

                        if(0 <= pp) {
                            //update alignment statistics
                            float dst2 = poss[bmQRDST2][pp];//distance2
                            ecrds = CombineCoords(rp0+1,qp0+1);
                            if(p == 0) bcrds = ecrds;
                            if(dst2 < d2equiv) psts++;
                            //minus to indicate reading
                            outAln[dp2oaQuery][ppaln] = outAln[dp2oaQuerySSS][ppaln] = -qp0;
                            outAln[dp2oaTarget][ppaln] = outAln[dp2oaTargetSSS][ppaln] = -rp0;
                            outAln[dp2oaMiddle][ppaln] = (dst2 < d2equiv)?'+':' ';
                            ppaln++;
                        }

                        qp0++; rp0++;

                        while(1) {
                            while(1) {
                                for(; qp0 < qp1 && ppaln < DIMX; qp0++, ppaln++) {
                                    outAln[dp2oaQuery][ppaln] = outAln[dp2oaQuerySSS][ppaln] = -qp0;
                                    outAln[dp2oaTarget][ppaln] = '-';
                                    outAln[dp2oaTargetSSS][ppaln] = ' ';
                                    outAln[dp2oaMiddle][ppaln] = ' ';
                                    gaps++;
                                }
                                if(DIMX <= ppaln) break;
                                if(nodeletions) break;//NOTE: command-line option
                                for(; rp0 < rp1 && ppaln < DIMX; rp0++, ppaln++) {
                                    outAln[dp2oaTarget][ppaln] = outAln[dp2oaTargetSSS][ppaln] = -rp0;
                                    outAln[dp2oaQuery][ppaln] = '-';
                                    outAln[dp2oaQuerySSS][ppaln] = ' ';
                                    outAln[dp2oaMiddle][ppaln] = ' ';
                                    gaps++;
                                }
                                break;
                            }

                            if(ppaln < DIMX) break;

                            //the buffer has filled; write the alignment fragment to gmem
                            idts +=
                            WriteAlignmentFragment<DIMX,memalignment>(qrydst, dbstrdst,
                                alnofff, dbalnlen, dbalnbeg, falnlen/* written */,
                                ppaln/* towrite */, ppaln/* tocheck */,
                                querypmbeg, bdbCpmbeg, tmp, outAln, alnsmem);

#ifdef MPDP_PRODUCTION_ALN_TESTPRINT
                            if((MPDP_PRODUCTION_ALN_TESTPRINT>=0)? ri==MPDP_PRODUCTION_ALN_TESTPRINT: 1){
                                for(int ii=0; ii<ppaln; ii++) printf("%c", outAln[dp2oaQuerySSS][ii]); printf("\n");
                                for(int ii=0; ii<ppaln; ii++) printf("%c", outAln[dp2oaQuery][ii]); printf("\n");
                                for(int ii=0; ii<ppaln; ii++) printf("%c", outAln[dp2oaMiddle][ii]); printf("\n");
                                for(int ii=0; ii<ppaln; ii++) printf("%c", outAln[dp2oaTarget][ii]); printf("\n");
                                for(int ii=0; ii<ppaln; ii++) printf("%c", outAln[dp2oaTargetSSS][ii]); printf("\n\n");
                            }
#endif
                            falnlen += ppaln;
                            ppaln = 0;
                        }//while(1)
                    }////for(p < mtchlen)
                }//for(relpos < mtchlen)

                //write terminator 0
                if(ppaln < DIMX) {
                    //the condition ppaln < DIMX ensured by the above inner loop
                    for(int f = 0; f < nTDP2OutputAlignmentSSS; f++) outAln[f][ppaln] = 0;
                }

                //write the last alignment fragment to memory
                idts +=
                WriteAlignmentFragment<DIMX,memalignment>(qrydst, dbstrdst,
                    alnofff, dbalnlen, dbalnbeg, falnlen/* written */,
                    ppaln + (ppaln < DIMX)/* towrite */, ppaln/* tocheck */,
                    querypmbeg, bdbCpmbeg, tmp, outAln, alnsmem);

                falnlen += ppaln;

                //write allignment statistics
                //NOTE: unbelievable: clang incorrectly optimizes the commented out!
                // *(uint*)(outSts + dp2oadBegCoords) = bcrds;
                // *(uint*)(outSts + dp2oadEndCoords) = ecrds;
                outSts[dp2oadBegCoords] = *(float*)&bcrds;
                outSts[dp2oadEndCoords] = *(float*)&ecrds;
                outSts[dp2oadAlnLength] = falnlen;
                outSts[dp2oadPstvs] = psts;
                outSts[dp2oadIdnts] = idts;
                outSts[dp2oadNGaps] = gaps;

                mloc0 = (qi * ndbCstrs_ + ri) * nTDP2OutputAlnData;
                #pragma omp simd aligned(alndatamem:memalignment)
                for(int pi = nTDP2OutputAlnDataPart1Beg; pi < nTDP2OutputAlnDataPart1End; pi++)
                    alndatamem[mloc0 + pi] = outSts[pi];


#ifdef MPDP_PRODUCTION_ALN_TESTPRINT
                if((MPDP_PRODUCTION_ALN_TESTPRINT>=0)? ri==MPDP_PRODUCTION_ALN_TESTPRINT: 1){
                    for(int ii=0; ii<ppaln+1; ii++) printf("%c", outAln[dp2oaQuerySSS][ii]); printf("\n");
                    for(int ii=0; ii<ppaln+1; ii++) printf("%c", outAln[dp2oaQuery][ii]); printf("\n");
                    for(int ii=0; ii<ppaln+1; ii++) printf("%c", outAln[dp2oaMiddle][ii]); printf("\n");
                    for(int ii=0; ii<ppaln+1; ii++) printf("%c", outAln[dp2oaTarget][ii]); printf("\n");
                    for(int ii=0; ii<ppaln+1; ii++) printf("%c", outAln[dp2oaTargetSSS][ii]); printf("\n\n");
                }
#endif

#ifdef MPDP_PRODUCTION_ALN_TESTPRINT
                if((MPDP_PRODUCTION_ALN_TESTPRINT>=0)? ri==MPDP_PRODUCTION_ALN_TESTPRINT: 1)
                printf(" PRODALN (dbstr= %u): qrylen= %d dbstrlen= --  y0= %d x0= %d ye= %d xe= %d "
                    "falnlen= %d  psts= %d idts= %d gaps= %d\n\n", ri, qrylen,/* dbstrlen, */
                    GetCoordY(bcrds), GetCoordX(bcrds), GetCoordY(ecrds), GetCoordX(ecrds),
                    falnlen, psts, idts, gaps
                );
#endif
            }//omp for
    }//omp parallel
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
