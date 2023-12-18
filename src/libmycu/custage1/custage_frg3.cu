/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include <math.h>

#include <string>
#include <vector>
#include <map>

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libutil/CLOptions.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gproc/dputils.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cucom/cutimer.cuh"
#include "libmycu/cucom/cugraphs.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"

#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/scoring.cuh"
#include "libmycu/custages/covariance.cuh"
#include "libmycu/custages/covariance_plus.cuh"
#include "libmycu/custages/covariance_swift_scan.cuh"
#include "libmycu/custage1/custage1.cuh"

#include "libmycu/custgfrg/local_similarity02.cuh"
#include "libmycu/custgfrg/linear_scoring.cuh"
#include "libmycu/custgfrg/linear_scoring2.cuh"
#include "libmycu/custgfrg2/linear_scoring2_complete.cuh"

#include "libmycu/cudp/dpsslocal.cuh"
#include "libmycu/cudp/dpw_btck.cuh"
#include "libmycu/cudp/dpw_score.cuh"
#include "libmycu/cudp/btck2match.cuh"

#include "custage_frg3.cuh"

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// run_stagefrg3: search for superposition between multiple molecules 
// simultaneously by exhaustively matching their fragments for similarity;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagefrg3::run_stagefrg3(
    std::map<CGKey,MyCuGraph>& stgraphs,
    cudaStream_t streamproc,
    const int maxndpiters,
    const uint maxnsteps,
    const uint minfraglen,
    const float /*scorethld*/,
    const float prescore,
    uint nqystrs, uint ndbCstrs,
    uint nqyposs, uint ndbCposs,
    uint qystr1len, uint dbstr1len,
    uint qystrnlen, uint dbstrnlen,
    uint dbxpad,
    float* __restrict__ /*scores*/, 
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
    float* __restrict__ tmpdpalnpossbuffer,
    uint* __restrict__ maxscoordsbuf,
    char* __restrict__ btckdata,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemccd,
    float* __restrict__ wrkmemtmalt,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ tfmmem,
    uint* __restrict__ /*globvarsbuf*/)
{
    MYMSG("stagefrg3::run_stagefrg3", 4);

    stagefrg3_extensive_frg_swift(
        streamproc,
        maxnsteps,
        nqystrs, ndbCstrs,
        nqyposs, ndbCposs,
        qystr1len, dbstr1len,
        qystrnlen, dbstrnlen,
        dbxpad,
        tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, btckdata,
        wrkmem, wrkmemccd, 
        wrkmemtmalt/*out*/, wrkmemtm, wrkmemtmibest, 
        wrkmemaux, wrkmem2, tfmmem);

    //process top N best-performing tfms found from the extensive 
    //application of spatial index:
    stagefrg3_refinement_tfmaltconfig(
        stgraphs,
        streamproc,
        maxnsteps,
        minfraglen,
        nqystrs, ndbCstrs,
        nqyposs, ndbCposs,
        qystr1len, dbstr1len,
        qystrnlen, dbstrnlen,
        dbxpad,
        tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, btckdata,
        wrkmem, wrkmemccd,
        wrkmemtmalt/*in*/, wrkmemtm, wrkmemtmibest,
        wrkmemaux, wrkmem2, tfmmem/*out*/);

    //refine alignment boundaries identified in the previous 
    //substage by applying DP;
    //1. With a gap cost:
    stage1_dprefine<false/*GAP0*/,false/*PRESCREEN*/,true/*WRKMEMTM1*/>(
        stgraphs,
        streamproc,
        maxndpiters,
        prescore,
        maxnsteps, minfraglen,
        nqystrs, ndbCstrs,
        nqyposs, ndbCposs,
        qystr1len, dbstr1len,
        qystrnlen, dbstrnlen,
        dbxpad,
        tmpdpdiagbuffers,
        tmpdpbotbuffer,
        tmpdpalnpossbuffer,
        maxscoordsbuf,
        btckdata,
        wrkmem, wrkmemccd, wrkmemtm, wrkmemtmibest,
        wrkmemaux, wrkmem2, tfmmem/*out*/);

    //2. No gap cost:
    stage1_dprefine<true/*GAP0*/,false/*PRESCREEN*/>(
        stgraphs,
        streamproc,
        maxndpiters,
        prescore,
        maxnsteps, minfraglen,
        nqystrs, ndbCstrs,
        nqyposs, ndbCposs,
        qystr1len, dbstr1len,
        qystrnlen, dbstrnlen,
        dbxpad,
        tmpdpdiagbuffers,
        tmpdpbotbuffer,
        tmpdpalnpossbuffer,
        maxscoordsbuf,
        btckdata,
        wrkmem, wrkmemccd, wrkmemtm, wrkmemtmibest,
        wrkmemaux, wrkmem2, tfmmem/*out*/);
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stagefrg3_refinement_tfmaltconfig: refine all alternative best-performing 
// superpositions obtained through the extensive application of spatial 
// index;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
#define STAGE1_REFINEFRAG(tpCONDITIONAL,tpSECONDARYUPDATE) \
    stage1::stage1_refinefrag<tpCONDITIONAL,tpSECONDARYUPDATE>( \
        stgraphs, \
        stg1REFINE_INITIAL_DP/*fragments identified by DP*/, \
        FRAGREF_NMAXCONVIT/*#iterations until convergence*/, \
        streamproc, \
        maxnsteps, minfraglen, \
        nqystrs, ndbCstrs, nqyposs, ndbCposs, \
        qystr1len, dbstr1len, qystrnlen, dbstrnlen, dbxpad, \
        tmpdpdiagbuffers, tmpdpalnpossbuffer, \
        wrkmem, wrkmemccd, wrkmemtm, wrkmemtmibest, \
        wrkmemaux, wrkmem2, tfmmem);

void stagefrg3::stagefrg3_refinement_tfmaltconfig(
    std::map<CGKey,MyCuGraph>& stgraphs,
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint minfraglen,
    uint nqystrs, uint ndbCstrs,
    uint nqyposs, uint ndbCposs,
    uint qystr1len, uint dbstr1len,
    uint qystrnlen, uint dbstrnlen,
    uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
    float* __restrict__ tmpdpalnpossbuffer,
    char* __restrict__ btckdata,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemccd,
    float* __restrict__ wrkmemtmalt,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ tfmmem)
{
    MYMSG("stagefrg3::stagefrg3_refinement_tfmaltconfig", 4);
    static const std::string preamb = "stagefrg3::stagefrg3_refinement_tfmaltconfig: ";


    //execution configuration for DP:
    const uint maxblkdiagelems = GetMaxBlockDiagonalElems(
            dbstr1len, qystr1len, CUDP_2DCACHE_DIM_D, CUDP_2DCACHE_DIM_X);
    dim3 nthrds_dp(CUDP_2DCACHE_DIM_D,1,1);
    dim3 nblcks_dp(maxblkdiagelems,ndbCstrs,nqystrs);
    //number of regular DIAGONAL block diagonal series;
    uint nblkdiags = (uint)
        (((dbstr1len + qystr1len) + CUDP_2DCACHE_DIM_X-1) / CUDP_2DCACHE_DIM_X);
    nblkdiags += (uint)(qystr1len - 1) / CUDP_2DCACHE_DIM_D;

    //execution configuration for DP-matched positions:
    dim3 nthrds_mtch(CUDP_MATCHED_DIM_X,CUDP_MATCHED_DIM_Y,1);
    dim3 nblcks_mtch(ndbCstrs,nqystrs,1);


    //configurations to verfy alternatively:
    //(CUS1_TBSP_DPSCORE_TOP_N_REFINEMENT)
    const int nbranches = CLOptions::GetC_NBRANCHES();
    const int depth = CLOptions::GetC_DEPTH();
    const int nconfigsections = 
        (depth <= CLOptions::csdDeep)? 3: ((depth <= CLOptions::csdHigh)? 2: 1);
    const int nconfigs = nbranches * nconfigsections;

    //process top N best-performing tfms found from the extensive 
    //application of spatial index:
    for(int rci = 0; rci < nconfigs; rci++)
    {
        //launch blocks along block diagonals to perform DP;
        //nblkdiags, total number of diagonals:
        for(uint d = 0; d < nblkdiags; d++)
        {
            ExecDPwBtck3264x
                <false/*ANCHORRGN*/,false/*BANDED*/,true/*GAP0*/,D02IND_SEARCH,true/*ALTSCTMS*/>
                <<<nblcks_dp,nthrds_dp,0,streamproc>>>(
                    d, ndbCstrs, ndbCposs, dbxpad, maxnsteps, rci/*stepnumber*/,
                    0.0f/*gap open cost*/,
                    wrkmemtmalt/*in*/, wrkmemaux,
                    tmpdpdiagbuffers, tmpdpbotbuffer, btckdata);
            MYCUDACHECKLAST;
        }

        //process DP result;
        BtckToMatched32x<false/*ANCHORRGN*/,false/*BANDED*/>
            <<<nblcks_mtch,nthrds_mtch,0,streamproc>>>(
                ndbCstrs, ndbCposs, dbxpad, maxnsteps, rci/*stepnumber*/,
                btckdata, wrkmemaux, tmpdpalnpossbuffer);
        MYCUDACHECKLAST;


        if(rci < 1) {
            STAGE1_REFINEFRAG(false/*true/CONDITIONAL */, SECONDARYUPDATE_UNCONDITIONAL);
        } else {
            STAGE1_REFINEFRAG(false, SECONDARYUPDATE_CONDITIONAL);
        }


        if(depth <= CLOptions::csdHigh)
        {   //one additional iteration of full DP sweep
            for(uint d = 0; d < nblkdiags; d++)
            {
                ExecDPwBtck3264x
                    <false/*ANCHORRGN*/,false/*BANDED*/,true/*GAP0*/,D02IND_SEARCH,false/*ALTSCTMS*/>
                    <<<nblcks_dp,nthrds_dp,0,streamproc>>>(
                        d, ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber*/,
                        0.0f/*gap open cost*/,
                        wrkmemtmibest/*in*/, wrkmemaux,
                        tmpdpdiagbuffers, tmpdpbotbuffer, btckdata);
                MYCUDACHECKLAST;
            }

            //process DP result;
            BtckToMatched32x<false/*ANCHORRGN*/,false/*BANDED*/>
                <<<nblcks_mtch,nthrds_mtch,0,streamproc>>>(
                    ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber*/,
                    btckdata, wrkmemaux, tmpdpalnpossbuffer);
            MYCUDACHECKLAST;

            if(rci < 1) {
                STAGE1_REFINEFRAG(false/*CONDITIONAL*/, SECONDARYUPDATE_UNCONDITIONAL);
            } else {
                STAGE1_REFINEFRAG(false, SECONDARYUPDATE_CONDITIONAL);
            }
        }
    }//rci
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stagefrg3_extensive_frg_swift: calculate tmscores and find most favorable 
// initial superposition based on fragment matching of multiple queries and 
// references;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagefrg3::stagefrg3_extensive_frg_swift(
    cudaStream_t streamproc,
    const uint maxnsteps,
    uint nqystrs, uint ndbCstrs,
    uint /*nqyposs*/, uint ndbCposs,
    uint qystr1len, uint dbstr1len,
    uint /*qystrnlen*/, uint /*dbstrnlen*/,
    uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
    float* __restrict__ tmpdpalnpossbuffer,
    char* __restrict__ btckdata,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemccd,
    float* __restrict__ wrkmemtmalt,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ /*tfmmem*/)
{
    MYMSG("stagefrg3::stagefrg3_extensive_frg_swift", 4);
    static const std::string preamb = "stagefrg3::stagefrg3_extensive_frg_swift: ";

    const int simthreshold = CLOptions::GetC_TRIGGER();

    //#iterations on alignment pairs to perform excluding the
    //initial (tfm) and final (score) kernels: 1, 2, or 3:
    static const int napiterations = 2;
    static const bool scoreachiteration = false;
    static const bool dynamicorientation = true;
    static const float thrsimilarityperc = (float)simthreshold / 100.0f;
    static const float thrscorefactor = 1.0f;
    static const float locgapcost = -0.8f;

    enum{nfrags = 2};//number of fragments of different length used
    static const int frags[nfrags] = {20, 100};
    static const int fragndx = 1;//should be the longest!
    //int fraglen = GetNAlnPoss_frg(
    //        qystr1len, dbstr1len, 0/*qrypos,unsed*/, 0/*rfnpos,unused*/,
    //        0/*qryfragfct,unsed*/, 0/*rfnfragfct,unused*/, 0/*fraglen index*/);

    const int depth = CLOptions::GetC_DEPTH();
    const int fctdiv =
        (depth==CLOptions::csdShallow)? GetFragStepSize_frg_shallow_factor(): 1;
    const int minnsteps = 10 / fctdiv;
    int qrystepsz = GetFragStepSize_frg_shallow(qystr1len);
    int rfnstepsz = GetFragStepSize_frg_shallow(dbstr1len);
    if(depth == CLOptions::csdDeep) {
        qrystepsz = GetFragStepSize_frg_deep(qystr1len);
        rfnstepsz = GetFragStepSize_frg_deep(dbstr1len);
    } else if(depth == CLOptions::csdHigh) {
        qrystepsz = GetFragStepSize_frg_high(qystr1len);
        rfnstepsz = GetFragStepSize_frg_high(dbstr1len);
    } else if(depth == CLOptions::csdMedium) {
        qrystepsz = GetFragStepSize_frg_medium(qystr1len);
        rfnstepsz = GetFragStepSize_frg_medium(dbstr1len);
    }
    //set minimum #steps to 10 since length 150 leads to 150/15=10,
    //the largest among #steps for medium-sized structures:
    const int nstepsy = myhdmax(minnsteps, (int)qystr1len/qrystepsz + 1);
    const int nstepsx = myhdmax(minnsteps, (int)dbstr1len/rfnstepsz + 1);

//     if(maxnsteps < nstepsx)
//         //maxnsteps computed for the minimum of #query and reference positions
//         //(step of 40<45 used by CuDeviceMemory)
//         throw MYRUNTIME_ERROR(preamb + "Number of steps exceeds the predetermined one.");


    //configuration for DP SS: fill in DP matrices for local alignment
    //based on SS match; this will be used for screening for plausible
    //local similarities
    const uint maxblkdiagelems_ss = GetMaxBlockDiagonalElems(
            dbstr1len, qystr1len, CUDP_2DCACHE_DIM_D, CUDP_2DCACHE_DIM_X);
    dim3 nthrds_dp_ss(CUDP_2DCACHE_DIM_D,1,1);
    dim3 nblcks_dp_ss(maxblkdiagelems_ss,ndbCstrs,nqystrs);
    //number of regular DIAGONAL block diagonal series;
    uint nblkdiags_ss = (uint)
        (((dbstr1len + qystr1len) + CUDP_2DCACHE_DIM_X-1) / CUDP_2DCACHE_DIM_X);
    nblkdiags_ss += (uint)(qystr1len - 1) / CUDP_2DCACHE_DIM_D;

    if(0.0f < thrsimilarityperc) {
        //launch blocks along block diagonals to perform DP;
        //nblkdiags_ss, total number of diagonals:
        for(uint d = 0; d < nblkdiags_ss; d++)
        {
            ExecDPSSLocal3264x<<<nblcks_dp_ss,nthrds_dp_ss,0,streamproc>>>(
                d, ndbCstrs, ndbCposs, dbxpad, maxnsteps, locgapcost,
                wrkmemaux/*cnv*/, tmpdpdiagbuffers/*wrk*/, tmpdpbotbuffer/*wrk*/,
                btckdata/*out*/);
            MYCUDACHECKLAST;
        }//dpss
    }


    //NOTE: InitScores<INITOPT_BEST>... moved one level up to CuBatch.cu!!
    // //execution configuration for scores initialization:
    // //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    // dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    // dim3 nblcks_scinit(
    //     (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
    //     nqystrs, maxnsteps);

    // //initialize memory for best scores;
    // InitScores<INITOPT_BEST><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
    //     ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
    // MYCUDACHECKLAST;


//     //{{the code below unused:
//     //initialize memory for query and reference positions;
//     InitScores<INITOPT_QRYRFNPOS><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
//         ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
//     MYCUDACHECKLAST;
// 
//     //initialize memory for fragment specifications;
//     InitScores<INITOPT_FRAGSPECS><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
//         ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
//     MYCUDACHECKLAST;
// 
//     //reset convergence flag;
//     InitScores<INITOPT_CONVFLAG_FRAGREF>
//         <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
//         ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
//     MYCUDACHECKLAST;
//     //}}


    //execution configuration for finding the maximum among scores 
    //calculated for each fragment factor:
    //each block processes one query and CUS1_TBSP_SCORE_MAX_XDIM references:
    dim3 nthrds_scmax(CUS1_TBSP_SCORE_MAX_XDIM,CUS1_TBSP_SCORE_MAX_YDIM,1);
    dim3 nblcks_scmax(
        (ndbCstrs + CUS1_TBSP_SCORE_MAX_XDIM - 1)/CUS1_TBSP_SCORE_MAX_XDIM,
        nqystrs, 1);

    //step number (<=maxnsteps) for efficiently launching numerous processing 
    //kernels and calculating scores on the query-reference dimensions:
    int stepnumber = 0;
    int ysndxproc = 0, xsndxproc = 0;//processed indices

    //there are fragment variants; divide max allowed accommodation by 2:
    const uint maxnstepso2 = (maxnsteps >> 1);

    for(int ysndx = 0; ysndx < nstepsy; ysndx++)
    {
        //increase #step indices over query and reference structures;
        //maxnsteps is max allowed steps to be processed in parallel simultaneously
        for(int xsndx = 0; xsndx < nstepsx; xsndx++)
        {
            bool lastiteration = (nstepsy <= ysndx + 1) && (nstepsx <= xsndx + 1);

            stepnumber++;

            if(stepnumber < (int)maxnstepso2 && !lastiteration)
                continue;

            //wrkmemtmibest contains best tfms over all recurses
            stagefrg3_score_based_on_fragmatching3(
                napiterations,
                scoreachiteration,
                dynamicorientation,
                thrsimilarityperc,
                thrscorefactor,
                streamproc,
                frags[fragndx],
                ysndxproc, xsndxproc, fragndx,
                maxnsteps, (stepnumber<<1),
                qystr1len, dbstr1len,
                nqystrs, ndbCstrs, ndbCposs, dbxpad,
                tmpdpdiagbuffers, tmpdpalnpossbuffer, btckdata,
                wrkmem, wrkmemtmibest, wrkmemaux, wrkmem2, wrkmemccd/*for wrkmemtm*/
            );

            if(depth <= CLOptions::csdHigh) {
                //save top CUS1_TBSP_DPSCORE_TOP_N secondary linear-alignment-based superposition 
                //configurations for selecting the currently best order-dependent superposition
                SaveTopNScoresAndTMsAmongSecondaryBests<<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
                    (depth),
                    (ysndxproc == 0 && xsndxproc == 0),//firstit
                    (depth <= CLOptions::csdDeep),//twoconfs,
                    (xsndxproc),//rfnfragfctinit
                    ndbCstrs, maxnsteps, (stepnumber<<1)/*effnsteps*/,
                    wrkmemtmibest/*in*/, wrkmemtm/*out*/, wrkmemaux);
                MYCUDACHECKLAST;
            }

            stepnumber = 0;
            ysndxproc = ysndx;
            xsndxproc = ysndx * nstepsx + xsndx + 1;

        }//reference positions
    }//query positions

    //save top CUS1_TBSP_DPSCORE_TOP_N linear-alignment-based superposition 
    //configurations for selecting the currently best order-dependent superposition
    SaveTopNScoresAndTMsAmongBests<<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
            ndbCstrs,  maxnsteps,  wrkmemtmibest/*in*/, wrkmemtm/*out*/, wrkmemaux);
    MYCUDACHECKLAST;


    //configuration for swift DP: calculate optimal order-dependent scores:
    const uint nconfigsections = 
        (depth <= CLOptions::csdDeep)? 3: ((depth <= CLOptions::csdHigh)? 2: 1);
    const uint nconfigs = (CUS1_TBSP_DPSCORE_TOP_N) * nconfigsections;
    const uint maxblkdiagelems_swft = GetMaxBlockDiagonalElems(
            dbstr1len, qystr1len, CUDP_SWFT_2DCACHE_DIM_D, CUDP_SWFT_2DCACHE_DIM_X);
    dim3 nthrds_dp_swft(CUDP_SWFT_2DCACHE_DIM_D,1,1);
    dim3 nblcks_dp_swft(maxblkdiagelems_swft, ndbCstrs, nqystrs * nconfigs);
    uint nblkdiags_swft = (uint)
        (((dbstr1len + qystr1len) + CUDP_SWFT_2DCACHE_DIM_X-1) / CUDP_SWFT_2DCACHE_DIM_X);
    nblkdiags_swft += (uint)(qystr1len - 1) / CUDP_SWFT_2DCACHE_DIM_D;

    //launch blocks along block diagonals to perform DP;
    //nblkdiags_swft, total number of diagonals:
    for(uint d = 0; d < nblkdiags_swft; d++)
    {
        ExecDPScore3264x<false/*ANCHORRGN*/,false/*BANDED*/,true/*GAP0*/,true/*CHECKCONV*/>
            <<<nblcks_dp_swft,nthrds_dp_swft,0,streamproc>>>(
                d, nqystrs, ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0.0f/*gcost*/,
                wrkmemtm, wrkmemaux,
                tmpdpdiagbuffers, tmpdpbotbuffer);
        MYCUDACHECKLAST;
    }//swift_dp

    //(CUS1_TBSP_DPSCORE_TOP_N_REFINEMENT)
    const int nbranches = CLOptions::GetC_NBRANCHES();

    //execution configuration for finding the maximum among dp scores:
    //each block processes one query and CUS1_TBSP_DPSCORE_MAX_XDIM references:
    dim3 nthrds_dpscmax(CUS1_TBSP_DPSCORE_MAX_XDIM,CUS1_TBSP_DPSCORE_MAX_YDIM,1);
    dim3 nblcks_dpscmax(
        (ndbCstrs + CUS1_TBSP_DPSCORE_MAX_XDIM - 1)/CUS1_TBSP_DPSCORE_MAX_XDIM,
        nqystrs, nconfigsections);

    //sort the best of the order-dependent scores and save the corresponding tfms:
    SortBestDPscoresAndTMsAmongDPswifts
        <<<nblcks_dpscmax,nthrds_dpscmax,0,streamproc>>>(
            nbranches,  ndbCstrs, ndbCposs, dbxpad,  maxnsteps,
            tmpdpdiagbuffers, wrkmemtm/*in*/, wrkmemtmalt/*out*/, wrkmemaux);
    MYCUDACHECKLAST;
}





// -------------------------------------------------------------------------
// stagefrg3_score_based_on_fragmatching3: obtain alignments in a couple of 
// iterations for massive number of variants obtained by fragment matching 
// between query and reference structures; score them;
// maxnsteps, max #steps that can be executed in parallel for one 
// query-reference pair; it corresponds to different #variants for a pair;
// actualnsteps, actual #variants (steps);
//
void stagefrg3::stagefrg3_score_based_on_fragmatching3(
    const int napiterations,
    const bool scoreachiteration,
    const bool dynamicorientation,
    const float thrsimilarityperc,
    const float thrscorefactor,
    cudaStream_t streamproc,
    const int maxfraglen,
    const int qryfragfct, const int rfnfragfct, const int fragndx,
    const uint maxnsteps,
    const uint actualnsteps,
    const uint qystr1len, const uint dbstr1len,
    const uint nqystrs, const uint ndbCstrs,
    const uint ndbCposs, const uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpalnpossbuffer,
    const char* __restrict__ dpscoremtx,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemtm)
{
//     if(1)
        stagefrg3_score_based_on_fragmatching3_helper(
            napiterations,
            scoreachiteration,
            dynamicorientation,
            thrsimilarityperc,
            thrscorefactor,
            streamproc,
            maxfraglen,
            qryfragfct, rfnfragfct, fragndx,
            maxnsteps, actualnsteps,
            qystr1len, dbstr1len,
            nqystrs, ndbCstrs, ndbCposs, dbxpad,
            tmpdpdiagbuffers, tmpdpalnpossbuffer, dpscoremtx,
            wrkmem, wrkmemtmibest, wrkmemaux, wrkmem2, wrkmemtm
        );
//     else
//         stagefrg3_score_based_on_fragmatching3_helper2(
//             napiterations,
//             scoreachiteration,
//             streamproc,
//             maxfraglen,
//             qryfragfct, rfnfragfct, fragndx,
//             maxnsteps, actualnsteps,
//             qystr1len, dbstr1len,
//             nqystrs, ndbCstrs, ndbCposs, dbxpad,
//             tmpdpdiagbuffers, tmpdpalnpossbuffer,
//             wrkmem, wrkmemtmibest, wrkmemaux, wrkmem2, wrkmemtm
//         );
}

// -------------------------------------------------------------------------
// stagefrg3_score_based_on_fragmatching3: helper method to obtain and score
// alignments by a complete kernel for massive number of variants 
// obtained by fragment matching between query and reference structures;
//
void stagefrg3::stagefrg3_score_based_on_fragmatching3_helper2(
    const int /*napiterations*/,
    const bool /*scoreachiteration*/,
    cudaStream_t streamproc,
    const int /*maxfraglen*/,
    const int qryfragfct, const int rfnfragfct, const int /*fragndx*/,
    const uint maxnsteps,
    const uint actualnsteps,
    const uint qystr1len, const uint /*dbstr1len*/,
    const uint nqystrs, const uint ndbCstrs,
    const uint ndbCposs, const uint /*dbxpad*/,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ /*wrkmem*/,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ /*wrkmem2*/,
    float* __restrict__ /*wrkmemtm*/)
{
    const int depth = CLOptions::GetC_DEPTH();

    //dynamically determined stack size for the linscos kernel:
    int stacksize_linsco = 1;
    if(0 < qystr1len)
        stacksize_linsco = myhdmin((int)17, (int)ceilf(log2f(qystr1len)) + 1);

    //execution configuration for complete linear scoring:
    //block processes one subfragment of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_linsco(CUSF_TBSP_COMPLETE_INDEX_SCORE_XDIM,1,1);
    dim3 nblcks_linsco(ndbCstrs, actualnsteps, nqystrs);

    ScoreFragmentBasedSuperpositionsLinearly2
        <<<nblcks_linsco,nthrds_linsco,0,streamproc>>>(
            stacksize_linsco, (depth),
            ndbCstrs, ndbCposs, maxnsteps, qryfragfct, rfnfragfct,
            tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stagefrg3_score_based_on_fragmatching3: helper method to obtain 
// alignments in a couple of iterations for massive number of variants 
// obtained by fragment matching between query and reference structures;
//
void stagefrg3::stagefrg3_score_based_on_fragmatching3_helper(
    const int napiterations,
    const bool scoreachiteration,
    const bool dynamicorientation,
    const float thrsimilarityperc,
    const float thrscorefactor,
    cudaStream_t streamproc,
    const int maxfraglen,
    const int qryfragfct, const int rfnfragfct, const int fragndx,
    const uint maxnsteps,
    const uint actualnsteps,
    const uint qystr1len, const uint dbstr1len,
    const uint nqystrs, const uint ndbCstrs,
    const uint ndbCposs, const uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpalnpossbuffer,
    const char* __restrict__ dpscoremtx,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemtm)
{
    //max #aligned positions over the pairs in a chunk
    //int minlenmax = myhdmin(qystr1len, dbstr1len);
    //alignment length can be >min due to the linear algorithm:
    const int minlenmax = dbstr1len;
    const uint maxlenmax =
        //stack size depends on the length of the structure indexed:
        dynamicorientation? myhdmax(dbstr1len, qystr1len): qystr1len;


    //execution configuration for CCM initialization:
    //each block processes one query and CUS1_TBINITSP_CCDINIT_XFCT references:
    dim3 nthrds_init(CUS1_TBINITSP_CCDINIT_XDIM,1,1);
    dim3 nblcks_init(
        (ndbCstrs + CUS1_TBINITSP_CCDINIT_XFCT - 1)/CUS1_TBINITSP_CCDINIT_XFCT,
        nqystrs, actualnsteps);

    //execution configuration for reduction:
    //block processes CUS1_TBINITSP_CCMCALC_XDIMLGL positions of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_ccmtx_frg(CUS1_TBINITSP_CCMCALC_XDIM,1,1);
    dim3 nblcks_ccmtx_frg(
        (maxfraglen + CUS1_TBINITSP_CCMCALC_XDIMLGL - 1)/CUS1_TBINITSP_CCMCALC_XDIMLGL,
        ndbCstrs, nqystrs * actualnsteps);

    //execution configuration for reduction:
    //block processes CUS1_TBINITSP_CCMCALC_XDIMLGL positions of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_ccmtx(CUS1_TBINITSP_CCMCALC_XDIM,1,1);
    dim3 nblcks_ccmtx(
        (minlenmax + CUS1_TBINITSP_CCMCALC_XDIMLGL - 1)/CUS1_TBINITSP_CCMCALC_XDIMLGL,
        ndbCstrs, nqystrs * actualnsteps);

    //execution configuration for reformatting data:
    //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
    dim3 nthrds_copyto(CUS1_TBINITSP_CCMCOPY_N,myhdmax(16,(int)twmvEndOfCCDataExt),1);
    dim3 nblcks_copyto(
        (ndbCstrs + CUS1_TBINITSP_CCMCOPY_N - 1)/CUS1_TBINITSP_CCMCOPY_N,
        nqystrs, actualnsteps);

    //execution configuration for calculating transformation matrices:
    //each block processes one query and CUS1_TBSP_TFM_N references:
    dim3 nthrds_tfm(CUS1_TBSP_TFM_N,1,1);
    dim3 nblcks_tfm(
        (ndbCstrs + CUS1_TBSP_TFM_N - 1)/CUS1_TBSP_TFM_N,
        nqystrs, actualnsteps);

    //execution configuration for reformatting data:
    //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
    dim3 nthrds_copyfrom(CUS1_TBINITSP_CCMCOPY_N,myhdmax(16,(int)nTTranformMatrix),1);
    dim3 nblcks_copyfrom(
        (ndbCstrs + CUS1_TBINITSP_CCMCOPY_N - 1)/CUS1_TBINITSP_CCMCOPY_N,
        nqystrs, actualnsteps);

    //execution configuration for calculating provisional scores (reduction):
    //block processes one fragment-based configuration of a query-reference pair:
    // //NOTE: previous version for calculating scores from global dp matrices!
    // dim3 nthrds_locsim(CUSF_TBSP_LOCAL_SIMILARITY_XDIM,1,1);
    // dim3 nblcks_locsim(
    //     (ndbCstrs + CUSF_TBSP_LOCAL_SIMILARITY_XDIM - 1)/CUSF_TBSP_LOCAL_SIMILARITY_XDIM,
    //     actualnsteps, nqystrs);
    dim3 nthrds_locsim(CUSF_TBSP_LOCAL_SIMILARITY_XDIM,CUSF_TBSP_LOCAL_SIMILARITY_YDIM,1);
    dim3 nblcks_locsim(ndbCstrs, actualnsteps, nqystrs);

    //execution configuration for calculating provisional scores (reduction):
    //block processes one fragment-based configuration of a query-reference pair:
    dim3 nthrds_simeval(CUS1_TBSP_SCORE_FRG2_HALT_CHK_XDIM,1,1);
    dim3 nblcks_simeval(ndbCstrs, actualnsteps, nqystrs);


    //execution configuration for linearly finding best-matching coordinates at each position (no dp):
    //block processes CUSF_TBSP_INDEX_SCORE_XDIMLGL positions of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_linscos(CUSF_TBSP_INDEX_SCORE_XDIM,1,1);
    dim3 nblcks_linscos(
        (myhdmin((uint)CUSF_TBSP_INDEX_SCORE_POSLIMIT2, dbstr1len) +
            CUSF_TBSP_INDEX_SCORE_XDIMLGL - 1)/CUSF_TBSP_INDEX_SCORE_XDIMLGL,
        ndbCstrs, nqystrs * actualnsteps);

    //dynamically determined stack size for the linscos kernel:
    uint stacksize_linscos = 1;
    if(0 < maxlenmax)
        stacksize_linscos = myhdmin((uint)17, (uint)ceilf(log2f(maxlenmax)) + 1);

    //size of dynamically allocted smem for the linscos kernel:
    uint szdsmem_linscos = sizeof(float) * (
#if (CUSF_TBSP_INDEX_SCORE_XFCT > 1)
        nTTranformMatrix +
#endif
        CUSF_TBSP_INDEX_SCORE_XDIM * stacksize_linscos * nStks_);

    MYCUDACHECK(cudaFuncSetAttribute(
        PositionalCoordsFromIndexLinear2<0>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, szdsmem_linscos));

    MYCUDACHECK(cudaFuncSetAttribute(
        PositionalCoordsFromIndexLinear2<1>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, szdsmem_linscos));


    //execution configuration for linearly drawing alignment (calculated before):
    //block processes CUSF2_TBSP_INDEX_ALIGNMENT_XDIMLGL positions of one query-reference pair:
    dim3 nthrds_linalign(CUSF2_TBSP_INDEX_ALIGNMENT_XDIM,CUSF2_TBSP_INDEX_ALIGNMENT_YDIM,1);
    dim3 nblcks_linalign(
        (myhdmin((uint)CUSF_TBSP_INDEX_SCORE_POSLIMIT2, dbstr1len) +
            CUSF2_TBSP_INDEX_ALIGNMENT_XDIMLGL - 1)/CUSF2_TBSP_INDEX_ALIGNMENT_XDIMLGL,
        ndbCstrs, nqystrs * actualnsteps);


    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs, actualnsteps);

    //execution configuration for calculating scores (reduction):
    //block processes CUS1_TBSP_SCORE_XDIMLGL positions of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_scores(CUS1_TBSP_SCORE_XDIM,1,1);
    dim3 nblcks_scores(
        (minlenmax + CUS1_TBSP_SCORE_XDIMLGL - 1)/CUS1_TBSP_SCORE_XDIMLGL,
        ndbCstrs, nqystrs * actualnsteps);

    //execution configuration for saving best performing transformation matrices:
    //each block processes one query and CUS1_TBINITSP_TMSAVE_XFCT references:
    dim3 nthrds_savetm(CUS1_TBINITSP_TMSAVE_XDIM,1,1);
    dim3 nblcks_savetm(
        (ndbCstrs + CUS1_TBINITSP_TMSAVE_XFCT - 1)/CUS1_TBINITSP_TMSAVE_XFCT,
        nqystrs, actualnsteps);

    //execution configuration for minimum score reduction:
    //block processes all positions of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_findd2(CUS1_TBINITSP_FINDD02_ITRD_XDIM,1,1);
    dim3 nblcks_findd2(ndbCstrs, nqystrs, actualnsteps);


    //reset alignment lengths;
    InitScores<INITOPT_NALNPOSS|INITOPT_CONVFLAG_SCOREDP>
        <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    stagefrg3_fragmatching3_Initial1(
        dynamicorientation,
        thrsimilarityperc,
        thrscorefactor,
        streamproc,
        qryfragfct, rfnfragfct, fragndx,
        maxnsteps,
        nqystrs, ndbCstrs, ndbCposs, dbxpad,
        dpscoremtx, wrkmem, wrkmemaux, wrkmem2, wrkmemtm,
        nblcks_init, nthrds_init,
        nblcks_ccmtx_frg, nthrds_ccmtx_frg,
        nblcks_copyto, nthrds_copyto,
        nblcks_copyfrom, nthrds_copyfrom,
        nblcks_locsim, nthrds_locsim,
        nblcks_simeval, nthrds_simeval,
        nblcks_tfm, nthrds_tfm);

    for(int n = 0; n < napiterations; n++)
    {
        bool secstrmatchaln = (n+1 < napiterations);
        bool completealn = true;//(napiterations <= n+1);
        bool reversetfms = !scoreachiteration && (n+1 < napiterations);
        bool writeqrypss = !reversetfms;//write query positions

        stagefrg3_fragmatching3_alignment2(
            dynamicorientation,
            secstrmatchaln,
            completealn,
            writeqrypss,
            streamproc,
            qryfragfct, rfnfragfct, fragndx,
            maxnsteps,
            nqystrs, ndbCstrs, ndbCposs, dbxpad,
            tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux, wrkmemtm,
            nblcks_linscos, nthrds_linscos, szdsmem_linscos, stacksize_linscos,
            nblcks_linalign, nthrds_linalign);

        stagefrg3_fragmatching3_superposition3(
            dynamicorientation,
            reversetfms,
            streamproc,
            maxnsteps,
            nqystrs, ndbCstrs, ndbCposs, dbxpad,
            tmpdpalnpossbuffer,
            wrkmem, wrkmemaux, wrkmem2, wrkmemtm,
            nblcks_init, nthrds_init,
            nblcks_ccmtx, nthrds_ccmtx,
            nblcks_copyto, nthrds_copyto,
            nblcks_copyfrom, nthrds_copyfrom,
            nblcks_tfm, nthrds_tfm);

        if(reversetfms) continue;

        stagefrg3_fragmatching3_aln_scoring4(
            streamproc,
            maxnsteps,
            nqystrs, ndbCstrs, ndbCposs, dbxpad,
            tmpdpdiagbuffers, tmpdpalnpossbuffer,
            wrkmem, wrkmemaux, wrkmem2, wrkmemtm, wrkmemtmibest,
            nblcks_init, nthrds_init,
            nblcks_ccmtx, nthrds_ccmtx,
            nblcks_findd2, nthrds_findd2,
            nblcks_scinit, nthrds_scinit,
            nblcks_scores, nthrds_scores,
            nblcks_savetm, nthrds_savetm,
            nblcks_copyto, nthrds_copyto,
            nblcks_copyfrom, nthrds_copyfrom,
            nblcks_tfm, nthrds_tfm);

        if(scoreachiteration && (n+1 < napiterations)) {
            //revert tfms for indexed alignment:
            stagefrg3_fragmatching3_superposition3(
                dynamicorientation,
                true/*reversetfms*/,
                streamproc,
                maxnsteps,
                nqystrs, ndbCstrs, ndbCposs, dbxpad,
                tmpdpalnpossbuffer,
                wrkmem, wrkmemaux, wrkmem2, wrkmemtm,
                nblcks_init, nthrds_init,
                nblcks_ccmtx, nthrds_ccmtx,
                nblcks_copyto, nthrds_copyto,
                nblcks_copyfrom, nthrds_copyfrom,
                nblcks_tfm, nthrds_tfm);
        }
    }
}



// -------------------------------------------------------------------------
// stagefrg3_fragmatching3_Initial1: subiteration 1 of producing structure 
// alignments by fragment matching in several iterations;
//
inline
void stagefrg3::stagefrg3_fragmatching3_Initial1(
    const bool dynamicorientation,
    const float thrsimilarityperc,
    const float /* thrscorefactor */,
    cudaStream_t streamproc,
    const int qryfragfct, const int rfnfragfct, const int fragndx,
    const uint maxnsteps,
    const uint nqystrs, const uint ndbCstrs,
    const uint ndbCposs, const uint dbxpad,
    const char* __restrict__ dpscoremtx,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemtm,
    //
    const dim3& nblcks_init, const dim3& nthrds_init,
    const dim3& nblcks_ccmtx_frg, const dim3& nthrds_ccmtx_frg,
    const dim3& nblcks_copyto, const dim3& nthrds_copyto,
    const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
    const dim3& nblcks_locsim, const dim3& nthrds_locsim,
    const dim3& /* nblcks_simeval */, const dim3& /* nthrds_simeval */,
    const dim3& nblcks_tfm, const dim3& nthrds_tfm)
{
    const int depth = CLOptions::GetC_DEPTH();

    //NOTE: when thrsimilarityperc<=0, the convergence flag is distributed:
    CalcLocalSimilarity2_frg2<<<nblcks_locsim,nthrds_locsim,0,streamproc>>>(
        thrsimilarityperc,
        (depth), ndbCstrs, ndbCposs, dbxpad, maxnsteps,
        qryfragfct, rfnfragfct, fragndx,
        dpscoremtx, wrkmemaux);
    MYCUDACHECKLAST;

    //initialize memory for calculating cross covariance matrices
    InitCCData<CHCKCONV_CHECK><<<nblcks_init,nthrds_init,0,streamproc>>>(
        ndbCstrs, maxnsteps,  wrkmem, wrkmemaux);
    // InitCCData0_frg2<<<nblcks_init,nthrds_init,0,streamproc>>>(
    //     (depth),  ndbCstrs,  maxnsteps, qryfragfct, rfnfragfct, fragndx,  wrkmem);
    MYCUDACHECKLAST;

    //calculate cross-covariance matrices with unrolling
    CalcCCMatrices64_frg2<<<nblcks_ccmtx_frg,nthrds_ccmtx_frg,0,streamproc>>>(
        (depth),  nqystrs, ndbCstrs,  maxnsteps, qryfragfct, rfnfragfct, fragndx,
        wrkmemaux, wrkmem);
    MYCUDACHECKLAST;

    //copy CC data to section 2 of working memory to enable efficient 
    //structure-specific calculation; READNPOS_NOREAD, do not verify whether
    //#positions on which tfms are calculated has changed:
    CopyCCDataToWrkMem2_frg2<<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
        (depth),  ndbCstrs,  maxnsteps, qryfragfct, rfnfragfct, fragndx,
        wrkmemaux, wrkmem/*in*/, wrkmem2/*out*/);
    MYCUDACHECKLAST;

    if(dynamicorientation)
        CalcTfmMatrices_DynamicOrientation<<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
            ndbCstrs, maxnsteps, wrkmem2);
    else
        //NOTE: calculate transformation matrices reversed wrt query-reference structure pair
        CalcTfmMatrices<TFMTX_REVERSE_TRUE><<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
            ndbCstrs, maxnsteps, wrkmem2);
    MYCUDACHECKLAST;

    //copy CC data from section 2 of working memory back for 
    // efficient calculation
    CopyTfmMtsFromWrkMem2<<<nblcks_copyfrom,nthrds_copyfrom,0,streamproc>>>(
        ndbCstrs,  maxnsteps,  wrkmem2/*in*/, wrkmemtm/*out*/);
    MYCUDACHECKLAST;

    //NOTE: previous version of screening for promising initial superpositions by
    //NOTE: calculating provisional scores; that does not work as intended;
    // CalcScoresUnrl_frg2<<<nblcks_simeval,nthrds_simeval,0,streamproc>>>(
    //     thrscorefactor, dynamicorientation,
    //     (depth), ndbCstrs, maxnsteps, qryfragfct, rfnfragfct, fragndx,
    //     wrkmemtm/*in*/, wrkmemaux);
    // MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stagefrg3_fragmatching3_alignment2: subiteration 2 of producing structure 
// alignments by fragment matching in several iterations;
//
inline
void stagefrg3::stagefrg3_fragmatching3_alignment2(
    const bool dynamicorientation,
    const bool secstrmatchaln,
    const bool complete,
    const bool writeqrypss,
    cudaStream_t streamproc,
    const int qryfragfct, const int rfnfragfct, const int fragndx,
    const uint maxnsteps,
    const uint nqystrs, const uint ndbCstrs,
    const uint ndbCposs, const uint /*dbxpad*/,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmemtm,
    //
    const dim3& nblcks_linscos, const dim3& nthrds_linscos,
        const uint szdsmem_linscos, const uint stacksize_linscos,
    const dim3& nblcks_linalign, const dim3& nthrds_linalign)
{
    const int depth = CLOptions::GetC_DEPTH();

    if(dynamicorientation) {
        if(secstrmatchaln)
            ProduceAlignmentUsingDynamicIndex2<1/*SECSTRFILT*/>
                <<<nblcks_linscos,nthrds_linscos,szdsmem_linscos,streamproc>>>(
                    (int)stacksize_linscos, writeqrypss, (depth),
                    nqystrs, ndbCstrs, ndbCposs,  maxnsteps,  qryfragfct, rfnfragfct,
                    wrkmemtm, tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux);
        else
            ProduceAlignmentUsingDynamicIndex2<0/*SECSTRFILT*/>
                <<<nblcks_linscos,nthrds_linscos,szdsmem_linscos,streamproc>>>(
                    (int)stacksize_linscos, writeqrypss, (depth),
                    nqystrs, ndbCstrs, ndbCposs,  maxnsteps,  qryfragfct, rfnfragfct,
                    wrkmemtm, tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux);
        MYCUDACHECKLAST;
    }
    else if(complete) {
        if(secstrmatchaln)
            ProduceAlignmentUsingIndex2<1/*SECSTRFILT*/>
                <<<nblcks_linscos,nthrds_linscos,szdsmem_linscos,streamproc>>>(
                    (int)stacksize_linscos, writeqrypss, (depth),
                    nqystrs, ndbCstrs, ndbCposs,  maxnsteps,  qryfragfct, rfnfragfct,
                    wrkmemtm, tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux);
        else
            ProduceAlignmentUsingIndex2<0/*SECSTRFILT*/>
                <<<nblcks_linscos,nthrds_linscos,szdsmem_linscos,streamproc>>>(
                    (int)stacksize_linscos, writeqrypss, (depth),
                    nqystrs, ndbCstrs, ndbCposs,  maxnsteps,  qryfragfct, rfnfragfct,
                    wrkmemtm, tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux);
        MYCUDACHECKLAST;
    } else {
        //calculate positional best-matching atoms using index trees
        if(secstrmatchaln)
            PositionalCoordsFromIndexLinear2<1/*SECSTRFILT*/>
                <<<nblcks_linscos,nthrds_linscos,szdsmem_linscos,streamproc>>>(
                    (int)stacksize_linscos, (depth),
                    nqystrs, ndbCstrs, ndbCposs,  maxnsteps,
                    qryfragfct, rfnfragfct, fragndx,
                    wrkmemtm, wrkmemaux, tmpdpalnpossbuffer);
        else
            PositionalCoordsFromIndexLinear2<0/*SECSTRFILT*/>
                <<<nblcks_linscos,nthrds_linscos,szdsmem_linscos,streamproc>>>(
                    (int)stacksize_linscos, (depth),
                    nqystrs, ndbCstrs, ndbCposs,  maxnsteps,
                    qryfragfct, rfnfragfct, fragndx,
                    wrkmemtm, wrkmemaux, tmpdpalnpossbuffer);
        MYCUDACHECKLAST;

        //draw alignment based on best-matching atom pairs
        MakeAlignmentLinear2<<<nblcks_linalign,nthrds_linalign,0,streamproc>>>(
                complete, (depth), nqystrs, ndbCstrs, ndbCposs,  maxnsteps,
                qryfragfct, rfnfragfct, fragndx,
                tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux);
        MYCUDACHECKLAST;
    }
}

// -------------------------------------------------------------------------
// stagefrg3_fragmatching3_superposition3: subiteration 1 of obtaining 
// superpositions based on resulting alignments;
//
inline
void stagefrg3::stagefrg3_fragmatching3_superposition3(
    const bool dynamicorientation,
    const bool reversetfms,
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint nqystrs, const uint ndbCstrs,
    const uint ndbCposs, const uint dbxpad,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemtm,
    //
    const dim3& nblcks_init, const dim3& nthrds_init,
    const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
    const dim3& nblcks_copyto, const dim3& nthrds_copyto,
    const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
    const dim3& nblcks_tfm, const dim3& nthrds_tfm)
{
    //initialize memory for calculating cross covariance matrices
    InitCCData<CHCKCONV_CHECK><<<nblcks_init,nthrds_init,0,streamproc>>>(
        ndbCstrs, maxnsteps,  wrkmem, wrkmemaux);
    MYCUDACHECKLAST;

    //calculate cross-covariance matrices with unrolling
    CalcCCMatrices64_SWFTscan<<<nblcks_ccmtx,nthrds_ccmtx,0,streamproc>>>(
        nqystrs, ndbCstrs, ndbCposs, dbxpad,  maxnsteps,
        wrkmemaux, tmpdpalnpossbuffer, wrkmem);
    MYCUDACHECKLAST;

    //copy CC data to section 2 of working memory to enable efficient 
    //structure-specific calculation; READNPOS_NOREAD, do not verify whether
    //#positions on which tfms are calculated has changed:
    CopyCCDataToWrkMem2_SWFTscan<READNPOS_NOREAD>
        <<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
            ndbCstrs,  maxnsteps,  wrkmemaux, wrkmem/*in*/, wrkmem2/*out*/);
    MYCUDACHECKLAST;

    if(reversetfms) {
        if(dynamicorientation)
            CalcTfmMatrices_DynamicOrientation<<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
                    ndbCstrs, maxnsteps, wrkmem2);
        else
            CalcTfmMatrices<TFMTX_REVERSE_TRUE><<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
                    ndbCstrs, maxnsteps, wrkmem2);
    } else
        CalcTfmMatrices<<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
            ndbCstrs, maxnsteps, wrkmem2);
    MYCUDACHECKLAST;

    //copy CC data from section 2 of working memory back for 
    // efficient calculation
    CopyTfmMtsFromWrkMem2<<<nblcks_copyfrom,nthrds_copyfrom,0,streamproc>>>(
        ndbCstrs,  maxnsteps,  wrkmem2/*in*/, wrkmemtm/*out*/);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stagefrg3_fragmatching3_aln_scoring4: subiteration 4 of scoring 
// alignments;
//
inline
void stagefrg3::stagefrg3_fragmatching3_aln_scoring4(
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint nqystrs, const uint ndbCstrs,
    const uint ndbCposs, const uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ /*wrkmem2*/,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    //
    const dim3& /*nblcks_init*/, const dim3& /*nthrds_init*/,
    const dim3& /*nblcks_ccmtx*/, const dim3& /*nthrds_ccmtx*/,
    const dim3& /*nblcks_findd2*/, const dim3& /*nthrds_findd2*/,
    const dim3& nblcks_scinit, const dim3& nthrds_scinit,
    const dim3& nblcks_scores, const dim3& nthrds_scores,
    const dim3& nblcks_savetm, const dim3& nthrds_savetm,
    const dim3& /*nblcks_copyto*/, const dim3& /*nthrds_copyto*/,
    const dim3& /*nblcks_copyfrom*/, const dim3& /*nthrds_copyfrom*/,
    const dim3& /*nblcks_tfm*/, const dim3& /*nthrds_tfm*/)
{
    //initialize memory for current scores only;
    InitScores<INITOPT_CURRENT><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    //calculate scores and temporarily save distances
    CalcScoresUnrl_SWFTscanProgressive<SAVEPOS_NOSAVE/*SAVEPOS_SAVE*/,CHCKALNLEN_NOCHECK>
        <<<nblcks_scores,nthrds_scores,0,streamproc>>>(
            nqystrs, ndbCstrs, ndbCposs, dbxpad,  maxnsteps,
            tmpdpalnpossbuffer, wrkmemtm, wrkmem, wrkmemaux, tmpdpdiagbuffers);
    MYCUDACHECKLAST;

    //save scores and tfms
    SaveBestScoreAndTM<false/*WRITEFRAGINFO*/>
        <<<nblcks_savetm,nthrds_savetm,0,streamproc>>>(
            ndbCstrs,  maxnsteps, 0/*sfragstep(unused)*/,
            wrkmemtm, wrkmemtmibest, wrkmemaux);
    MYCUDACHECKLAST;
}



// -------------------------------------------------------------------------
