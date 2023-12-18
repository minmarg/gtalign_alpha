/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include <string>
#include <vector>

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libutil/CLOptions.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gproc/dputils.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"

#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/scoring.cuh"
#include "libmycu/custages/transform.cuh"
#include "libmycu/custages2/covariance_fin_dp_refn_complete.cuh"
#include "libmycu/custages2/covariance_production_dp_refn_complete.cuh"

#include "libmycu/cudp/dpssw_tfm_btck.cuh"
#include "libmycu/cudp/btck2match.cuh"
#include "libmycu/cudp/constrained_btck2match.cuh"
#include "libmycu/cudp/production_match2aln.cuh"

#include "libmycu/custage1/custage1.cuh"
#include "custage_fin.cuh"

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// run_stagefin: final stage for the refinement of the best alignment 
// obtained and output data production;;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagefin::run_stagefin(
    cudaStream_t streamproc,
    const float d2equiv,
    const float /* scorethld */,
    const uint maxnsteps,
    const uint minfraglen,
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
    float* __restrict__ /* wrkmem */,
    float* __restrict__ /* wrkmemccd */,
    float* __restrict__ /* wrkmemtm */,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ /* wrkmem2 */,
    float* __restrict__ alndatamem,
    float* __restrict__ tfmmem,
    char* __restrict__ alnsmem,
    uint* __restrict__ /*globvarsbuf*/)
{
    MYMSG("stagefin::run_stagefin", 4);
    static std::string preamb = "stagefin::run_stagefin: ";

    //produce alignment to refine superposition on:
    stagefin_align(
        false/* constrainedbtck */,
        streamproc, maxnsteps,
        nqystrs, ndbCstrs, nqyposs, ndbCposs,
        qystr1len, dbstr1len, qystrnlen, dbstrnlen, dbxpad,
        tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer,
        maxscoordsbuf, btckdata, wrkmemaux, tfmmem);

    //refine superposition at the finest scale
    stagefin_refine(
        streamproc, maxnsteps,  minfraglen,
        nqystrs, ndbCstrs, nqyposs, ndbCposs,
        qystr1len, dbstr1len, qystrnlen, dbstrnlen, dbxpad,
        tmpdpdiagbuffers, tmpdpalnpossbuffer,
        wrkmemtmibest, wrkmemaux, tfmmem);

    //produce final alignment of matched positions:
    stagefin_align(
        true/* constrainedbtck */,
        streamproc, maxnsteps,
        nqystrs, ndbCstrs, nqyposs, ndbCposs,
        qystr1len, dbstr1len, qystrnlen, dbstrnlen, dbxpad,
        tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer,
        maxscoordsbuf, btckdata, wrkmemaux, tfmmem);

    //produce full alignments for output: 
    stagefin_produce_output_alignment(
        d2equiv,
        streamproc, maxnsteps,
        nqystrs, ndbCstrs, nqyposs, ndbCposs,
        qystr1len, dbstr1len, qystrnlen, dbstrnlen, dbxpad,
        tmpdpalnpossbuffer, wrkmemaux, alndatamem, alnsmem);

    //refine using production thresholds and calculate final scores for output
    stagefin_produce_output_scores(
        streamproc, maxnsteps,  minfraglen,
        nqystrs, ndbCstrs, nqyposs, ndbCposs,
        qystr1len, dbstr1len, qystrnlen, dbstrnlen, dbxpad,
        tmpdpdiagbuffers, tmpdpalnpossbuffer, wrkmemtmibest, 
        wrkmemaux, alndatamem, tfmmem);

    //revert transformation matrices if needed
    stagefin_adjust_tfms(
        streamproc, nqystrs,  ndbCstrs, nqyposs, ndbCposs,
        wrkmemaux, tfmmem);
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stagefin_align: obtain alignment based on the best superposition obtain;
// constrainedbtck, flag of using constrained backtracking;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagefin::stagefin_align(
    const bool constrainedbtck,
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint nqystrs, const uint ndbCstrs,
    const uint /*nqyposs*/, const uint ndbCposs,
    const uint qystr1len, const uint dbstr1len,
    const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
    const uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
    float* __restrict__ tmpdpalnpossbuffer,
    uint* __restrict__ /*maxscoordsbuf*/,
    char* __restrict__ btckdata,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tfmmem)
{
    MYMSG("stagefin::stagefin_align", 5);
    static std::string preamb = "stagefin::stagefin_align: ";

    //execution configuration for DP:
    //1D thread block processes 2D DP matrix oblique block of dimension 
    //CUDP_2DCACHE_DIM_D x CUDP_2DCACHE_DIM_X;
    const uint maxblkdiagelems = GetMaxBlockDiagonalElems(
            dbstr1len, qystr1len, CUDP_2DCACHE_DIM_D, CUDP_2DCACHE_DIM_X);
    dim3 nthrds_dp(CUDP_2DCACHE_DIM_D,1,1);
    dim3 nblcks_dp(maxblkdiagelems,ndbCstrs,nqystrs);

    //number of regular DIAGONAL block diagonal series, each of given dimensions;
    //rect coords (x,y) are related to diagonal number d by d=x+y-1;
    uint nblkdiags = (uint)
        (((dbstr1len + qystr1len) + CUDP_2DCACHE_DIM_X-1) / CUDP_2DCACHE_DIM_X);
    //NOTE: now use block DIAGONALS, where blocks share a COMMON POINT 
    //NOTE: (corner, instead of edge) with a neighbour in a diagonal;
    //the number of series of such block diagonals equals 
    // #regular block diagonals (above) + {(l-1)/w}, 
    // where l is query length (y coord.), w is block width (dimension), and
    // {} denotes floor rounding; {(l-1)/w} is #interior divisions;
    nblkdiags += (uint)(qystr1len - 1) / CUDP_2DCACHE_DIM_D;

    //execution configuration for extracting matched positions
    //identified during DP:
    dim3 nthrds_mtch(CUDP_MATCHED_DIM_X,CUDP_MATCHED_DIM_Y,1);
    dim3 nblcks_mtch(ndbCstrs,nqystrs,1);

    dim3 nthrds_const_mtch(CUDP_CONST_MATCH_DIM_X,CUDP_CONST_MATCH_DIM_Y,1);
    dim3 nblcks_const_mtch(ndbCstrs,nqystrs,1);


    //launch blocks along block diagonals to perform DP;
    //nblkdiags, total number of diagonals:
    for(uint d = 0; d < nblkdiags; d++)
    {
        ExecDPTFMSSwBtck3264x<true/* GLOBTFM */,true/* GAP0 */,false/* USESS */,D02IND_SEARCH>
            <<<nblcks_dp,nthrds_dp,0,streamproc>>>(
                d, ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber(unused)*/,
                0.0f/* sswgt */, 0.0f/* gcost */,
                tfmmem, wrkmemaux, tmpdpdiagbuffers, tmpdpbotbuffer, btckdata);
        MYCUDACHECKLAST;
    }

    //produce alignment for superposition
    if(constrainedbtck)
        ConstrainedBtckToMatched32x
            <<<nblcks_const_mtch,nthrds_const_mtch,0,streamproc>>>(
                ndbCstrs, ndbCposs, dbxpad, maxnsteps,
                btckdata, tfmmem, wrkmemaux, tmpdpalnpossbuffer);
    else
        BtckToMatched32x<false/*ANCHORRGN*/,false/*BANDED*/>
            <<<nblcks_mtch,nthrds_mtch,0,streamproc>>>(
                ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber*/,
                btckdata, wrkmemaux, tmpdpalnpossbuffer);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stagefin_refine: refine for final superposition-best alignment; 
// complete version;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagefin::stagefin_refine(
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint minfraglen,
    const uint nqystrs, const uint ndbCstrs,
    const uint /* nqyposs */, const uint ndbCposs,
    const uint qystr1len, const uint dbstr1len,
    const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
    const uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tfmmem)
{
    MYMSG("stagefin::stagefin_refine", 5);
    static std::string preamb = "stagefin::stagefin_refine: ";

    // const float scorethld = CLOptions::GetO_S();
    const int symmetric = CLOptions::GetC_SYMMETRIC();
    const int refinement = CLOptions::GetC_REFINEMENT();
    const int depth = CLOptions::GetC_DEPTH();

    //minimum length among largest
    int minlenmax = myhdmin(qystr1len, dbstr1len);
    int maxalnmax = minlenmax;//maximum alignment length

    //maximum number of fragment subdivisions
    const int nmaxsubfrags = FRAGREF_NMAXSUBFRAGS;
    // sfragstep, step to traverse subfragments;
    constexpr int sfragstep = FRAGREF_SFRAGSTEP;

    uint nlocsteps = 0;
    nlocsteps = (uint)GetMaxNFragSteps(maxalnmax, sfragstep, minfraglen);
    nlocsteps *= nmaxsubfrags;//total number across all fragment lengths

    if(nlocsteps < 1 || maxnsteps < nlocsteps)
        throw MYRUNTIME_ERROR(preamb +
        "Invalid number of superposition tests: "+std::to_string(nlocsteps));

    //NOTE: minimum of the largest structures to compare is assumed >=3;
    //step for the SECOND phase to final (finer-scale) refinement;
    constexpr int sfragstepmini = FRAGREF_SFRAGSTEP_mini;
    //max #fragment position factors around an identified position
    //**********************************************************************
    //NOTE: multiply maxalnmax by 2 since sub-optimal (first-phase) alignment
    //NOTE: position can be identified, e.g., at the end of alignment!
    //**********************************************************************
    uint maxnfragfcts = myhdmin(2 * maxalnmax, CUSFN_TBSP_FIN_REFINEMENT_MAX_NPOSITIONS);
    maxnfragfcts = (maxnfragfcts + sfragstepmini-1) / sfragstepmini;
    uint nlocsteps2 = maxnfragfcts;//total number for ONE fragment length
    if(refinement == CLOptions::csrFullASearch) nlocsteps2 *= nmaxsubfrags;//total number across all fragment lengths

    if(nlocsteps2 < 1 || maxnsteps < nlocsteps2)
        throw MYRUNTIME_ERROR(preamb +
        "Invalid number of superposition tests: "+std::to_string(nlocsteps2));


    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs, myhdmax(nlocsteps, nlocsteps2)/* maxnsteps */);

    //initialize memory for best scores only;
    // if(0.0f < scorethld) {
        InitScores<INITOPT_BEST><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
        MYCUDACHECKLAST;
    // }


    //execution configuration for complete refinement:
    //block processes one subfragment of a certain length of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_arcmpl(CUS1_TBINITSP_COMPLETEREFINE_XDIM,1,1);
    dim3 nblcks_arcmpl(ndbCstrs, nlocsteps, nqystrs);

    //refine alignment boundaries to improve scores
    if(symmetric)
        FinalFragmentBasedDPAlignmentRefinementPhase1<false/* D0FINAL */,CHCKDST_CHECK,true/*TFM_DINV*/>
            <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                nmaxsubfrags, maxnsteps, sfragstep, maxalnmax,
                tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
    else
        FinalFragmentBasedDPAlignmentRefinementPhase1<false/* D0FINAL */,CHCKDST_CHECK,false/*TFM_DINV*/>
            <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                nmaxsubfrags, maxnsteps, sfragstep, maxalnmax,
                tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
    MYCUDACHECKLAST;

    //execution configuration for finding the maximum among scores 
    //calculated for each fragment factor:
    //each block processes one query and CUS1_TBSP_SCORE_MAX_XDIM references:
    dim3 nthrds_scmax(CUS1_TBSP_SCORE_MAX_XDIM,CUS1_TBSP_SCORE_MAX_YDIM,1);
    dim3 nblcks_scmax(
        (ndbCstrs + CUS1_TBSP_SCORE_MAX_XDIM - 1)/CUS1_TBSP_SCORE_MAX_XDIM,
        nqystrs, 1);

    SaveBestScoreAndTMAmongBests<true/*WRITEFRAGINFO*/,true/*GRANDUPDATE*/,true/*FORCEWRITEFRAGINFO*/>
        <<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
            ndbCstrs,  maxnsteps, nlocsteps,  wrkmemtmibest, tfmmem, wrkmemaux);
    MYCUDACHECKLAST;

    if(depth == CLOptions::csdShallow || refinement == CLOptions::csrCoarseSearch)
        return;


    //second phase to production (finer-scale) refinement;
    //execution configuration for complete refinement:
    //block processes one subfragment of a certain length of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    nthrds_arcmpl = dim3(CUS1_TBINITSP_COMPLETEREFINE_XDIM,1,1);
    nblcks_arcmpl = dim3(ndbCstrs, nlocsteps2, nqystrs);

    if(refinement == CLOptions::csrFullASearch) {
        if(symmetric)
            FinalFragmentBasedDPAlignmentRefinementPhase2_fullsearch<false/* D0FINAL */,CHCKDST_CHECK,true/*TFM_DINV*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnfragfcts, maxnsteps, sfragstepmini, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
        else
            FinalFragmentBasedDPAlignmentRefinementPhase2_fullsearch<false/* D0FINAL */,CHCKDST_CHECK,false/*TFM_DINV*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnfragfcts, maxnsteps, sfragstepmini, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
    }
    else {
        if(symmetric)
            FinalFragmentBasedDPAlignmentRefinementPhase2<false/* D0FINAL */,CHCKDST_CHECK,true/*TFM_DINV*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnfragfcts, maxnsteps, sfragstepmini, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
        else
            FinalFragmentBasedDPAlignmentRefinementPhase2<false/* D0FINAL */,CHCKDST_CHECK,false/*TFM_DINV*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnfragfcts, maxnsteps, sfragstepmini, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
    }
    MYCUDACHECKLAST;

    //repeat finding the maximum among scores calculated for each fragment factor:
    SaveBestScoreAndTMAmongBests<false/*WRITEFRAGINFO*/>
        <<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
            ndbCstrs,  maxnsteps, nlocsteps2,  wrkmemtmibest, tfmmem, wrkmemaux);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stagefin_produce_output: produce full alignment; 
// complete version;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagefin::stagefin_produce_output_alignment(
    const float d2equiv,
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint nqystrs, const uint ndbCstrs,
    const uint /* nqyposs */, const uint ndbCposs,
    const uint /* qystr1len */, const uint /* dbstr1len */,
    const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
    const uint dbxpad,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmemaux,
    float* __restrict__ alndatamem,
    char* __restrict__ alnsmem)
{
    MYMSG("stagefin::stagefin_produce_output_alignment", 5);
    static std::string preamb = "stagefin::stagefin_produce_output_alignment: ";
    static const bool nodeletions = CLOptions::GetO_NO_DELETIONS();

    //execution configuration for producing alignment from matched positions:
    dim3 nthrds_mtch2aln(CUDP_PRODUCTION_ALN_DIM_X,CUDP_PRODUCTION_ALN_DIM_Y,1);
    dim3 nblcks_mtch2aln(ndbCstrs,nqystrs,1);

    ProductionMatchToAlignment32x<<<nblcks_mtch2aln,nthrds_mtch2aln,0,streamproc>>>(
        nodeletions, d2equiv, /*nqystrs, nqyposs,*/ ndbCstrs, ndbCposs, dbxpad, maxnsteps,
        tmpdpalnpossbuffer, wrkmemaux, alndatamem, alnsmem);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stagefin_produce_output_scores: refine using production thresholds and 
// calculate final scores for output; complete version;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stagefin::stagefin_produce_output_scores(
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint minfraglen,
    const uint nqystrs, const uint ndbCstrs,
    const uint /* nqyposs */, const uint ndbCposs,
    const uint qystr1len, const uint dbstr1len,
    const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
    const uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ alndatamem,
    float* __restrict__ tfmmem)
{
    MYMSG("stagefin::stagefin_produce_output_scores", 5);
    static std::string preamb = "stagefin::stagefin_produce_output_scores: ";

    const int symmetric = CLOptions::GetC_SYMMETRIC();
    const int refinement = CLOptions::GetC_REFINEMENT();
    //minimum length among largest
    int minlenmax = myhdmin(qystr1len, dbstr1len);
    int maxalnmax = minlenmax;//maximum alignment length

    //maximum number of fragments subdivisions
    const uint nmaxsubfrags = FRAGREF_NMAXSUBFRAGS;
    // sfragstep, step to traverse subfragments;
    int sfragstep = FRAGREF_SFRAGSTEP;

    uint nlocsteps = 0;
    nlocsteps = (uint)GetMaxNFragSteps(maxalnmax, sfragstep, minfraglen);
    nlocsteps *= nmaxsubfrags;//total number across all fragment lengths

    if(nlocsteps < 1 || maxnsteps < nlocsteps)
        throw MYRUNTIME_ERROR(preamb +
        "Invalid number of superposition tests: "+std::to_string(nlocsteps));

    //NOTE: minimum of the largest structures to compare is assumed >=3;
    //step for the SECOND phase to production (finer-scale) refinement;
    constexpr int sfragstepmini = FRAGREF_SFRAGSTEP_mini;
    //max #fragment position factors around an identified position
    //**********************************************************************
    //NOTE: multiply maxalnmax by 2 since sub-optimal (first-phase) alignment
    //NOTE: position can be identified at the end of alignment!
    //**********************************************************************
    uint maxnfragfcts = myhdmin(2 * maxalnmax, CUSFN_TBSP_FIN_REFINEMENT_MAX_NPOSITIONS);
    maxnfragfcts = (maxnfragfcts + sfragstepmini-1) / sfragstepmini;
    uint nlocsteps2 = 1;
    if(refinement >= CLOptions::csrFullSearch) nlocsteps2 = maxnfragfcts * nmaxsubfrags;//total number across all fragment lengths
    if(refinement == CLOptions::csrOneSearch ||
       refinement == CLOptions::csrCoarseSearch) nlocsteps2 = maxnfragfcts;//total number for ONE fragment length
    if(refinement == CLOptions::csrLogSearch) nlocsteps2 = 1;//log search inline

    if(nlocsteps2 < 1 || maxnsteps < nlocsteps2)
        throw MYRUNTIME_ERROR(preamb +
        "Invalid number of superposition tests: "+std::to_string(nlocsteps2));

    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs, myhdmax(nlocsteps, nlocsteps2)/* maxnsteps */);

    //initialize memory for best scores only;
    InitScores<INITOPT_BEST><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;



    //execution configuration for complete refinement:
    //block processes one subfragment of a certain length of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_arcmpl(CUS1_TBINITSP_COMPLETEREFINE_XDIM,1,1);
    dim3 nblcks_arcmpl(ndbCstrs, nlocsteps, nqystrs);

    if(symmetric)
        ProductionFragmentBasedDPAlignmentRefinementPhase1<true/*TFM_DINV*/>
            <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                nmaxsubfrags, maxnsteps, sfragstep, maxalnmax,
                tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest,
                wrkmemaux, alndatamem);
    else
        ProductionFragmentBasedDPAlignmentRefinementPhase1<false/*TFM_DINV*/>
            <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                nmaxsubfrags, maxnsteps, sfragstep, maxalnmax,
                tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest,
                wrkmemaux, alndatamem);
    MYCUDACHECKLAST;

    //execution configuration for finding the maximum among scores 
    //calculated for each fragment factor:
    //each block processes one query and CUS1_TBSP_SCORE_MAX_XDIM references:
    dim3 nthrds_scmax(CUS1_TBSP_SCORE_MAX_XDIM,CUS1_TBSP_SCORE_MAX_YDIM,1);
    dim3 nblcks_scmax(
        (ndbCstrs + CUS1_TBSP_SCORE_MAX_XDIM - 1)/CUS1_TBSP_SCORE_MAX_XDIM,
        nqystrs, 1);

    ProductionSaveBestScoresAndTMAmongBests
        <true/*WRITEFRAGINFO*/, false/* CONDITIONAL */>
            <<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
                ndbCstrs,  maxnsteps, nlocsteps,
                wrkmemtmibest, wrkmemaux, alndatamem, tfmmem);
    MYCUDACHECKLAST;



    //second phase to production (finer-scale) refinement;
    //execution configuration for complete refinement:
    //block processes one subfragment of a certain length of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    nthrds_arcmpl = dim3(CUS1_TBINITSP_COMPLETEREFINE_XDIM,1,1);
    nblcks_arcmpl = dim3(ndbCstrs, nlocsteps2, nqystrs);

    if(refinement == CLOptions::csrLogSearch) {
        if(symmetric)
            ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearch<true/*TFM_DINV*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnfragfcts, maxnsteps, sfragstepmini, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux, alndatamem, tfmmem);
        else
            ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearch<false/*TFM_DINV*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnfragfcts, maxnsteps, sfragstepmini, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux, alndatamem, tfmmem);
        MYCUDACHECKLAST;
        return;
    }

    if(refinement == CLOptions::csrOneSearch || refinement == CLOptions::csrCoarseSearch) {
        if(symmetric)
            ProductionFragmentBasedDPAlignmentRefinementPhase2<true/*TFM_DINV*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnfragfcts, maxnsteps, sfragstepmini, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
        else
            ProductionFragmentBasedDPAlignmentRefinementPhase2<false/*TFM_DINV*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnfragfcts, maxnsteps, sfragstepmini, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
    }
    else if(refinement >= CLOptions::csrFullSearch) {
        if(symmetric)
            ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearch<true/*TFM_DINV*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnfragfcts, maxnsteps, sfragstepmini, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
        else
            ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearch<false/*TFM_DINV*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    /* nqystrs, */ ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnfragfcts, maxnsteps, sfragstepmini, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
    }
    MYCUDACHECKLAST;

    //repeat finding the maximum among scores calculated for each fragment factor:
    ProductionSaveBestScoresAndTMAmongBests
        <false/*WRITEFRAGINFO*/, true/* CONDITIONAL */>
            <<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
                ndbCstrs,  maxnsteps, nlocsteps2,
                wrkmemtmibest, wrkmemaux, alndatamem, tfmmem);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stagefin_adjust_tfms: revert transformation matrices if needed
//
void stagefin::stagefin_adjust_tfms(
    cudaStream_t streamproc,
    const uint nqystrs, const uint ndbCstrs,
    const uint /*nqyposs*/, const uint /*ndbCposs*/,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tfmmem)
{
    // static std::string preamb = "stagefin::stagefin_adjust_tfms: ";
    static const int referenced = CLOptions::GetO_REFERENCED();
    if(referenced == 0) return;

    MYMSG("stagefin::stagefin_adjust_tfms", 5);
 
    dim3 nthrds_tfminit(CUS1_TBINITSP_TFMINIT_XDIM,1,1);
    dim3 nblcks_tfminit(
        (ndbCstrs + CUS1_TBINITSP_TFMINIT_XFCT - 1)/CUS1_TBINITSP_TFMINIT_XFCT,
        nqystrs, 1);

    //revert transformation matrices:
    RevertTfmMatrices<<<nblcks_tfminit,nthrds_tfminit,0,streamproc>>>(
        ndbCstrs, wrkmemaux, tfmmem);
    MYCUDACHECKLAST;
 }
