/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

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

#include "libmycu/custages/fragment.cuh"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/covariance.cuh"
#include "libmycu/custages/covariance_plus.cuh"
#include "libmycu/custages/covariance_refn.cuh"
#include "libmycu/custages/covariance_dp_refn.cuh"
#include "libmycu/custages2/covariance_complete.cuh"
#include "libmycu/custages2/covariance_refn_complete.cuh"
#include "libmycu/custages2/covariance_dp_refn_complete.cuh"
#include "libmycu/custages/transform.cuh"
#include "libmycu/custages/scoring.cuh"
#include "libmycu/cudp/dpw_btck.cuh"
#include "libmycu/cudp/btck2match.cuh"
#include "libmycu/cudp2/dpw_btck_complete.cuh"
#include "custage1.cuh"

// -------------------------------------------------------------------------
// preinitialize1: initialize memory before starting calculations
//
void stage1::preinitialize1(
    cudaStream_t streamproc,
    const bool condition4filter1,
    const uint maxnsteps,
    const uint minfraglen,
    uint nqystrs, uint ndbCstrs,
    uint /*nqyposs*/, uint /*ndbCposs*/,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tfmmem,
    float* __restrict__ alndatamem)
{
    //execution configuration for tfm matrix initialization:
    //each block processes one query and CUS1_TBINITSP_TFMINIT_XFCT references:
    dim3 nthrds_tfminitibest(CUS1_TBINITSP_TFMINIT_XDIM,1,1);
    dim3 nblcks_tfminitibest(
        (ndbCstrs + CUS1_TBINITSP_TFMINIT_XFCT - 1)/CUS1_TBINITSP_TFMINIT_XFCT,
        nqystrs, maxnsteps);

    dim3 nthrds_tfminit(CUS1_TBINITSP_TFMINIT_XDIM,1,1);
    dim3 nblcks_tfminit(
        (ndbCstrs + CUS1_TBINITSP_TFMINIT_XFCT - 1)/CUS1_TBINITSP_TFMINIT_XFCT,
        nqystrs, 1);

    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs, maxnsteps);

    //initialize memory for scores (once in the stage);
    if(condition4filter1)
        InitScores<INITOPT_ALL|INITOPT_CONVFLAG_LOWTMSC_SET>
            <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
                ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
    else
        InitScores<INITOPT_ALL><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    //initialize memory for transformation matrices (once in the stage);
    InitTfmMatrices<<<nblcks_tfminitibest,nthrds_tfminitibest,0,streamproc>>>(
       ndbCstrs, maxnsteps, minfraglen, FRAGREF_SFRAGSTEP, false/*checkfragos*/,
       wrkmemtmibest);
    MYCUDACHECKLAST;

    //initialize memory for final transformation matrices;
    InitGTfmMatricesAndAData<<<nblcks_tfminit,nthrds_tfminit,0,streamproc>>>(
        ndbCstrs, tfmmem, alndatamem);
    MYCUDACHECKLAST;
 }

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// run_stage1: initial (stage-1) search for superposition between multiple 
// molecules simultaneoulsy and identify fragments for further refinement;
// tmalign correspondence: get_initial, detailed_search, DP_iter;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stage1::run_stage1(
    std::map<CGKey,MyCuGraph>& stgraphs,
    cudaStream_t streamproc,
    const int maxndpiters,
    const uint maxnsteps,
    const uint minfraglen,
    const float /* scorethld */,
    const float prescore,
    int stepinit,
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
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ tfmmem,
    uint* __restrict__ /* globvarsbuf */)
{
    //find best scoring aligned gapless fragment
    stage1_findfrag2(
        streamproc, stepinit,
        maxnsteps, minfraglen,
        nqystrs, ndbCstrs,
        nqyposs, ndbCposs,
        qystr1len, dbstr1len,
        qystrnlen, dbstrnlen,
        dbxpad,
        tmpdpdiagbuffers,
        wrkmem, wrkmemaux, wrkmem2, wrkmemtm);

    //refine fragment boundaries to improve scores
    stage1_refinefrag<false/* CONDITIONAL */>(
        stgraphs,
        stg1REFINE_INITIAL/*fragments identified initially*/,
        FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
        streamproc,
        maxnsteps, minfraglen,
        nqystrs, ndbCstrs,
        nqyposs, ndbCposs,
        qystr1len, dbstr1len,
        qystrnlen, dbstrnlen,
        dbxpad,
        tmpdpdiagbuffers,
        tmpdpalnpossbuffer,
        wrkmem, wrkmemccd, wrkmemtm, wrkmemtmibest,
        wrkmemaux, wrkmem2, tfmmem);

    //refine alignment boundaries identified in the previous 
    //substage by applying DP
    stage1_dprefine<true/*GAP0*/,true/*PRESCREEN*/>(
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
        wrkmemaux, wrkmem2, tfmmem);
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stage1_findfrag: search for superposition between multiple molecules 
// simultaneously and identify fragments for further refinement;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stage1::stage1_findfrag2(
    cudaStream_t streamproc,
    int stepinit,
    const uint maxnsteps,
    const uint /*minfraglen*/,
    uint nqystrs, uint ndbCstrs,
    uint /*nqyposs*/, uint ndbCposs,
    uint qystr1len, uint dbstr1len,
    uint qystrnlen, uint dbstrnlen,
    uint /*dbxpad*/,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ /*wrkmem*/,
    float* __restrict__ wrkmemaux,
    float* __restrict__ /*wrkmem2*/,
    float* __restrict__ /*wrkmemtm*/)
{
    const int symmetric = CLOptions::GetC_SYMMETRIC();
    //NOTE: minlen, minimum of the largest structures to compare, assumed >=3
    //minimum length among largest
    int minlenmax = myhdmin(qystr1len, dbstr1len);
    //minimum length among smallest
    int minlenmin = myhdmin(qystrnlen, dbstrnlen);
    int minalnmin = myhdmax(minlenmin >> 1, 5);
    int n1 = minalnmin - dbstr1len;
    int n2 = qystr1len - minalnmin;

    uint maxnsteps1 = 0;

    //launch kernels to process a bulk of maxnsteps fragments over all query-reference
    //pairs in the chunk:
    for(; n1 <= n2; n1 += stepinit * (int)maxnsteps)
    {
        //reduce #thread blocks to be launched if maxnsteps implies exceeding the limit
        int nlocsteps = 
            (n1 + stepinit * (int)maxnsteps <= n2)? maxnsteps: (n2-n1)/stepinit + 1;

        if(!maxnsteps1) maxnsteps1 = nlocsteps;

        //execution configuration for complete fragment identification:
        //block processes one subfragment of one query-reference pair:
        //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
        dim3 nthrds_ficmpl(CUS1_TBINITSP_COMPLETEREFINE_XDIM,1,1);
        dim3 nblcks_ficmpl(ndbCstrs, nlocsteps, nqystrs);

        if(symmetric)
            FindGaplessAlignedFragment<true/*TFM_DINV*/>
                <<<nblcks_ficmpl,nthrds_ficmpl,0,streamproc>>>(
                    ndbCstrs, ndbCposs, maxnsteps, n1/*arg1*/, stepinit/*arg2*/, 0/*arg3*/,
                    tmpdpdiagbuffers, wrkmemaux);
        else
            FindGaplessAlignedFragment<false/*TFM_DINV*/>
                <<<nblcks_ficmpl,nthrds_ficmpl,0,streamproc>>>(
                    ndbCstrs, ndbCposs, maxnsteps, n1/*arg1*/, stepinit/*arg2*/, 0/*arg3*/,
                    tmpdpdiagbuffers, wrkmemaux);
    }

    //execution configuration for finding the maximum among scores 
    //calculated for each fragment factor:
    //each block processes one query and CUS1_TBSP_SCORE_MAX_XDIM references:
    dim3 nthrds_scmax(CUS1_TBSP_SCORE_MAX_XDIM,CUS1_TBSP_SCORE_MAX_YDIM,1);
    dim3 nblcks_scmax(
        (ndbCstrs + CUS1_TBSP_SCORE_MAX_XDIM - 1)/CUS1_TBSP_SCORE_MAX_XDIM,
        nqystrs, 1);

    SaveBestScoreAmongBests<<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
        ndbCstrs,  maxnsteps, maxnsteps1,  wrkmemaux);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stage1_findfrag: search for superposition between multiple molecules 
// simultaneously and identify fragments for further refinement;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stage1::stage1_findfrag(
    cudaStream_t streamproc,
    int stepinit,
    const uint maxnsteps,
    const uint minfraglen,
    uint nqystrs, uint ndbCstrs,
    uint /* nqyposs */, uint ndbCposs,
    uint qystr1len, uint dbstr1len,
    uint qystrnlen, uint dbstrnlen,
    uint /* dbxpad */,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemtm)
{
    //NOTE: minlen, minimum of the largest structures to compare, assumed >=3
    //minimum length among largest
    int minlenmax = myhdmin(qystr1len, dbstr1len);
    //minimum length among smallest
    int minlenmin = myhdmin(qystrnlen, dbstrnlen);
    int minalnmin = myhdmax(minlenmin >> 1, 5);
    int n1 = minalnmin - dbstr1len;
    int n2 = qystr1len - minalnmin;



    //execution configuration for tfm matrix initialization:
    //each block processes one query and CUS1_TBINITSP_TFMINIT_XFCT references:
    dim3 nthrds_tfminit(CUS1_TBINITSP_TFMINIT_XDIM,1,1);
    dim3 nblcks_tfminit(
        (ndbCstrs + CUS1_TBINITSP_TFMINIT_XFCT - 1)/CUS1_TBINITSP_TFMINIT_XFCT,
        nqystrs,1);

    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs, maxnsteps);

//     //initialize memory for transformation matrices (once in the stage);
//     InitTfmMatrices<<<nblcks_tfminit,nthrds_tfminit,0,streamproc>>>(
//         ndbCstrs,
//         tfmmem);
//     MYCUDACHECKLAST;

    //initialize memory for scores (once in the stage);
    InitScores<INITOPT_ALL><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    //for execution configuration of finding max scores
    uint maxnsteps1 = 0;


    //launch kernels to process a bulk of maxnsteps fragments over all query-reference
    //pairs in the chunk:
    for(; n1 <= n2; n1 += stepinit * (int)maxnsteps)
    {
        //reduce #thread blocks to be launched if maxnsteps implies exceeding the limit
        int nlocsteps = 
            (n1 + stepinit * (int)maxnsteps <= n2)? maxnsteps: (n2-n1)/stepinit + 1;

        if(!maxnsteps1) maxnsteps1 = nlocsteps;


        //execution configuration for tfm matrix initialization:
        //each block processes one query and CUS1_TBINITSP_TFMINIT_XFCT references:
        dim3 nthrds_tfminitibest(CUS1_TBINITSP_TFMINIT_XDIM,1,1);
        dim3 nblcks_tfminitibest(
            (ndbCstrs + CUS1_TBINITSP_TFMINIT_XFCT - 1)/CUS1_TBINITSP_TFMINIT_XFCT,
            nqystrs, nlocsteps);

        //initialize memory for transformation matrices;
        InitTfmMatrices<<<nblcks_tfminitibest,nthrds_tfminitibest,0,streamproc>>>(
            ndbCstrs, maxnsteps, minfraglen, stepinit, false/*checkfragos*/,
            wrkmemtm);
        MYCUDACHECKLAST;

        //execution configuration for initialization:
        //each block processes one query and CUS1_TBINITSP_CCDINIT_XFCT references:
        dim3 nthrds_init(CUS1_TBINITSP_CCDINIT_XDIM,1,1);
        dim3 nblcks_init(
            (ndbCstrs + CUS1_TBINITSP_CCDINIT_XFCT - 1)/CUS1_TBINITSP_CCDINIT_XFCT,
            nqystrs, nlocsteps);

        //execution configuration for reduction:
        //block processes CUS1_TBINITSP_CCMCALC_XDIMLGL positions of one query-reference pair:
        //NOTE: ndbCstrs and nqystrs * maxnsteps cannot be greater than 65535:
        //TODO: ensured by JobDispatcher and CuDeviceMemory
        dim3 nthrds_ccmtx(CUS1_TBINITSP_CCMCALC_XDIM,1,1);
        dim3 nblcks_ccmtx(
            (minlenmax + CUS1_TBINITSP_CCMCALC_XDIMLGL - 1)/CUS1_TBINITSP_CCMCALC_XDIMLGL,
            ndbCstrs, nqystrs * nlocsteps);

        //execution configuration for reformatting data:
        //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
        dim3 nthrds_copyto(CUS1_TBINITSP_CCMCOPY_N,myhdmax(16,(int)twmvEndOfCCDataExt),1);
        dim3 nblcks_copyto(
            (ndbCstrs + CUS1_TBINITSP_CCMCOPY_N - 1)/CUS1_TBINITSP_CCMCOPY_N,
            nqystrs, nlocsteps);

        //execution configuration for calculating transformation matrices:
        //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
        dim3 nthrds_tfm(CUS1_TBSP_TFM_N,1,1);
        dim3 nblcks_tfm(
            (ndbCstrs + CUS1_TBSP_TFM_N - 1)/CUS1_TBSP_TFM_N,
            nqystrs, nlocsteps);

        //execution configuration for reformatting data:
        //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
        dim3 nthrds_copyfrom(CUS1_TBINITSP_CCMCOPY_N,myhdmax(16,(int)nTTranformMatrix),1);
        dim3 nblcks_copyfrom(
            (ndbCstrs + CUS1_TBINITSP_CCMCOPY_N - 1)/CUS1_TBINITSP_CCMCOPY_N,
            nqystrs, nlocsteps);

        //execution configuration for scores initialization:
        //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
        nthrds_scinit = dim3(CUS1_TBSP_SCORE_SET_XDIM,1,1);
        nblcks_scinit = dim3(
            (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
            nqystrs, nlocsteps);

        //execution configuration for calculating scores (reduction):
        //block processes CUS1_TBSP_SCORE_XDIMLGL positions of one query-reference pair:
        //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
        dim3 nthrds_scores(CUS1_TBSP_SCORE_XDIM,1,1);
        dim3 nblcks_scores(
            (minlenmax + CUS1_TBSP_SCORE_XDIMLGL - 1)/CUS1_TBSP_SCORE_XDIMLGL,
            ndbCstrs, nqystrs * nlocsteps);

        //execution configuration for minimum score reduction:
        //block processes all positions of one query-reference pair:
        //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
        dim3 nthrds_findd2(CUS1_TBINITSP_FINDD02_ITRD_XDIM,1,1);
        dim3 nblcks_findd2(ndbCstrs, nqystrs, nlocsteps);


        stage1_findfrag_subiter1(
            streamproc,
            n1, stepinit,
            maxnsteps, minfraglen,
            nqystrs, ndbCstrs,
            wrkmem, wrkmemaux, wrkmem2, wrkmemtm,
            nblcks_init, nthrds_init,
            nblcks_ccmtx, nthrds_ccmtx,
            nblcks_copyto, nthrds_copyto,
            nblcks_copyfrom, nthrds_copyfrom,
            nblcks_tfm, nthrds_tfm);

        stage1_findfrag_subiter2(
            streamproc,
            n1, stepinit,
            maxnsteps, minfraglen,
            nqystrs, ndbCstrs, ndbCposs,
            tmpdpdiagbuffers,
            wrkmem, wrkmemaux, wrkmem2, wrkmemtm,
            nblcks_init, nthrds_init,
            nblcks_ccmtx,nthrds_ccmtx,
            nblcks_findd2,nthrds_findd2,
            nblcks_scinit, nthrds_scinit,
            nblcks_scores, nthrds_scores,
            nblcks_copyto, nthrds_copyto,
            nblcks_copyfrom, nthrds_copyfrom,
            nblcks_tfm, nthrds_tfm);

        stage1_findfrag_subiter3(
            streamproc,
            n1, stepinit,
            maxnsteps, minfraglen,
            nqystrs, ndbCstrs, ndbCposs,
            tmpdpdiagbuffers,
            wrkmem, wrkmemaux, wrkmem2, wrkmemtm,
            nblcks_init, nthrds_init,
            nblcks_ccmtx, nthrds_ccmtx,
            nblcks_findd2, nthrds_findd2,
            nblcks_scinit, nthrds_scinit,
            nblcks_scores, nthrds_scores,
            nblcks_copyto, nthrds_copyto,
            nblcks_copyfrom, nthrds_copyfrom,
            nblcks_tfm, nthrds_tfm);
    }

    //execution configuration for finding the maximum among scores 
    //calculated for each fragment factor:
    //each block processes one query and CUS1_TBSP_SCORE_MAX_XDIM references:
    dim3 nthrds_scmax(CUS1_TBSP_SCORE_MAX_XDIM,CUS1_TBSP_SCORE_MAX_YDIM,1);
    dim3 nblcks_scmax(
        (ndbCstrs + CUS1_TBSP_SCORE_MAX_XDIM - 1)/CUS1_TBSP_SCORE_MAX_XDIM,
        nqystrs, 1);

    SaveBestScoreAmongBests<<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
        ndbCstrs,  maxnsteps, maxnsteps1,  wrkmemaux);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stage1_findfrag_subiter1: subiteration 1 of stage1 for finding best 
// scoring aligned fragments: calculate cross-covariances 
// and rotation matrices
//
inline
void stage1::stage1_findfrag_subiter1(
    cudaStream_t streamproc,
    int n1, int stepinit,
    const uint maxnsteps,
    const uint /*minfraglen*/,
    uint nqystrs, uint ndbCstrs,
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
    // (required for each iteration);
    //NOTE: initialization only for participating pairs may be not
    // advantageous: requires additional 2*CUS1_TBINITSP_CCDINIT_XFCT 
    // reads per block and arithmetics;
    InitCCData0<<<nblcks_init,nthrds_init,0,streamproc>>>(
        ndbCstrs,  maxnsteps, n1, stepinit,  wrkmem);
    MYCUDACHECKLAST;

    //calculate cross-covariance matrices with unrolling
    CalcCCMatrices64<<<nblcks_ccmtx,nthrds_ccmtx,0,streamproc>>>(
        nqystrs,  ndbCstrs,  maxnsteps, n1, stepinit,  wrkmem, wrkmemaux);
    MYCUDACHECKLAST;

    //copy CC data to section 2 of working memory to enable
    // efficient structure-specific calculation
    CopyCCDataToWrkMem2<READNPOS_NOREAD><<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
        ndbCstrs,  maxnsteps, n1, stepinit,  wrkmemaux, wrkmem/*in*/, wrkmem2/*out*/);
    MYCUDACHECKLAST;

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
// stage1_findfrag_subiter2: subiteration 2 of stage1 for finding best 
// scoring aligned fragments: unconditionally calculate 
// and save scores, calculate cross-covariances for alignment positions 
// within given distances, and optionally calculate rotation matrices
//
inline
void stage1::stage1_findfrag_subiter2(
    cudaStream_t streamproc,
    int n1, int stepinit,
    const uint maxnsteps,
    const uint minfraglen,
    uint nqystrs, uint ndbCstrs, uint ndbCposs,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemtm,
    //
    const dim3& nblcks_init, const dim3& nthrds_init,
    const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
    const dim3& nblcks_findd2, const dim3& nthrds_findd2,
    const dim3& nblcks_scinit, const dim3& nthrds_scinit,
    const dim3& nblcks_scores, const dim3& nthrds_scores,
    const dim3& nblcks_copyto, const dim3& nthrds_copyto,
    const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
    const dim3& nblcks_tfm, const dim3& nthrds_tfm)
{
    //initialize memory for current scores only;
    InitScores<INITOPT_CURRENT><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    //calculate scores and temporarily save distances
    CalcScoresUnrl<SAVEPOS_SAVE,CHCKALNLEN_NOCHECK>
        <<<nblcks_scores,nthrds_scores,0,streamproc>>>(
            nqystrs, ndbCstrs, ndbCposs,  maxnsteps, n1, stepinit,
            wrkmemtm, wrkmem, wrkmemaux, tmpdpdiagbuffers);
    MYCUDACHECKLAST;

    //save scores
    SaveBestScore<<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, n1, stepinit,  wrkmemaux);
    MYCUDACHECKLAST;

    //find required minimum distances for adjusting rotation
    FindD02ThresholdsCCM<READCNST_CALC>
        <<<nblcks_findd2,nthrds_findd2,0,streamproc>>>(
            ndbCstrs, ndbCposs,  maxnsteps, n1, stepinit,
            tmpdpdiagbuffers, wrkmem, wrkmemaux);
    MYCUDACHECKLAST;

    //initialize memory before calculating cross-covariance matrices
    InitCCData0<<<nblcks_init,nthrds_init,0,streamproc>>>(
        ndbCstrs,  maxnsteps, n1, stepinit,  wrkmem);
    MYCUDACHECKLAST;

    //calculate cross-covariance matrices with unrolling;
    //include only positions within given distances
    CalcCCMatrices64Extended<READCNST_CALC>
        <<<nblcks_ccmtx,nthrds_ccmtx,0,streamproc>>>(
            nqystrs, ndbCstrs, ndbCposs,  maxnsteps, n1, stepinit,
            tmpdpdiagbuffers, wrkmemaux, wrkmem);
    MYCUDACHECKLAST;

    //copy CC data to section 2 of working memory to enable
    // efficient structure-specific calculation
    CopyCCDataToWrkMem2<READNPOS_READ><<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
        ndbCstrs,  maxnsteps, n1, stepinit,  wrkmemaux, wrkmem/*in*/, wrkmem2/*out*/);
    MYCUDACHECKLAST;

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
// stage1_findfrag_subiter3: subiteration 3 of stage1 for finding best 
// scoring aligned fragments: unconditionally calculate 
// and save scores, calculate cross-covariances for alignment positions 
// within given distances, optionally calculate rotation matrices, and
// calculate and save scores only 
//
inline
void stage1::stage1_findfrag_subiter3(
    cudaStream_t streamproc,
    int n1, int stepinit,
    const uint maxnsteps,
    const uint minfraglen,
    uint nqystrs, uint ndbCstrs, uint ndbCposs,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemtm,
    //
    const dim3& nblcks_init, const dim3& nthrds_init,
    const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
    const dim3& nblcks_findd2, const dim3& nthrds_findd2,
    const dim3& nblcks_scinit, const dim3& nthrds_scinit,
    const dim3& nblcks_scores, const dim3& nthrds_scores,
    const dim3& nblcks_copyto, const dim3& nthrds_copyto,
    const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
    const dim3& nblcks_tfm, const dim3& nthrds_tfm)
{
    //initialize memory for current scores only;
    InitScores<INITOPT_CURRENT><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    //calculate scores and temporarily save distances
    CalcScoresUnrl<SAVEPOS_SAVE,CHCKALNLEN_CHECK>
        <<<nblcks_scores,nthrds_scores,0,streamproc>>>(
            nqystrs, ndbCstrs, ndbCposs,  maxnsteps, n1, stepinit,
            wrkmemtm, wrkmem, wrkmemaux, tmpdpdiagbuffers);
    MYCUDACHECKLAST;

    //save scores
    SaveBestScore<<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, n1, stepinit,  wrkmemaux);
    MYCUDACHECKLAST;

    //find required minimum distances for adjusting rotation
    FindD02ThresholdsCCM<READCNST_CALC2>
        <<<nblcks_findd2,nthrds_findd2,0,streamproc>>>(
            ndbCstrs, ndbCposs,  maxnsteps, n1, stepinit,
            tmpdpdiagbuffers, wrkmem, wrkmemaux);
    MYCUDACHECKLAST;

    //initialize memory before calculating cross-covariance matrices
    InitCCData0<<<nblcks_init,nthrds_init,0,streamproc>>>(
        ndbCstrs,  maxnsteps, n1, stepinit,  wrkmem);
    MYCUDACHECKLAST;

    //calculate cross-covariance matrices with unrolling;
    //include only positions within given distances
    CalcCCMatrices64Extended<READCNST_CALC2>
        <<<nblcks_ccmtx,nthrds_ccmtx,0,streamproc>>>(
            nqystrs, ndbCstrs, ndbCposs,  maxnsteps, n1, stepinit,
            tmpdpdiagbuffers, wrkmemaux, wrkmem);
    MYCUDACHECKLAST;

    //copy CC data to section 2 of working memory to enable
    // efficient structure-specific calculation
    CopyCCDataToWrkMem2<READNPOS_READ><<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
        ndbCstrs,  maxnsteps, n1, stepinit,  wrkmemaux, wrkmem/*in*/, wrkmem2/*out*/);
    MYCUDACHECKLAST;

    CalcTfmMatrices<<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
        ndbCstrs, maxnsteps, wrkmem2);
    MYCUDACHECKLAST;

    //copy CC data from section 2 of working memory back for 
    // efficient calculation
    CopyTfmMtsFromWrkMem2<<<nblcks_copyfrom,nthrds_copyfrom,0,streamproc>>>(
        ndbCstrs,  maxnsteps,  wrkmem2/*in*/, wrkmemtm/*out*/);
    MYCUDACHECKLAST;

    //final calculation of scores:
    //initialize memory for current scores only;
    InitScores<INITOPT_CURRENT><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    CalcScoresUnrl<SAVEPOS_NOSAVE,CHCKALNLEN_CHECK>
        <<<nblcks_scores,nthrds_scores,0,streamproc>>>(
            nqystrs, ndbCstrs, ndbCposs,  maxnsteps, n1, stepinit,
            wrkmemtm, wrkmem, wrkmemaux, tmpdpdiagbuffers);
    MYCUDACHECKLAST;

    SaveBestScore<<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, n1, stepinit,  wrkmemaux);
    MYCUDACHECKLAST;
}










// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stage1_fragscore: score fragments identified in the previous step;
// fragbydp, flag of whether fragments are identified by DP;
// nmaxconvit, maximum number of iterations for convergence check;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stage1::stage1_fragscore(
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint minfraglen,
    uint nqystrs, uint ndbCstrs,
    uint /* nqyposs */, uint ndbCposs,
    uint qystr1len, uint dbstr1len,
    uint /*qystrnlen*/, uint /*dbstrnlen*/,
    uint /*dbxpad*/,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemccd,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ tfmmem)
{
    static std::string preamb = "stage1::stage1_fragscore: ";
    //NOTE: minimum of the largest structures to compare is assumed >=3
    //minimum length among largest
    int minlenmax = myhdmin(qystr1len, dbstr1len);
    //minimum length among smallest
//     int minlenmin = myhdmin(qystrnlen, dbstrnlen);
//     int minalnmin = myhdmax(minlenmin >> 1, 5);
    //maximum alignment length can be minimum of the lengths
    int maxalnmax = minlenmax;
//     int n1 = minalnmin - dbstr1len;
//     int n2 = qystr1len - minalnmin;


    // sfragstep, step to traverse subfragments;
    const int sfragstep = FRAGREF_SFRAGSTEP;


    //execution configuration for tfm matrix initialization:
    //each block processes one query and CUS1_TBINITSP_TFMINIT_XFCT references:
    //dim3 nthrds_tfminitibest(CUS1_TBINITSP_TFMINIT_XDIM,1,1);
    //dim3 nblcks_tfminitibest(
    //    (ndbCstrs + CUS1_TBINITSP_TFMINIT_XFCT - 1)/CUS1_TBINITSP_TFMINIT_XFCT,
    //    nqystrs, maxnsteps);

    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs, maxnsteps);

    //NOTE: do not initialize here for keeping the immediate last 
    //NOTE: matrix in the case of convergence;
    //InitTfmMatrices<<<nblcks_tfminitibest,nthrds_tfminitibest,0,streamproc>>>(
    //   ndbCstrs, maxnsteps, minfraglen, sfragstep, false/*checkfragos*/,
    //   wrkmemtmibest);
    //MYCUDACHECKLAST;

    //initialize memory for best scores only;
    InitScores<INITOPT_BEST><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;


    //maximum alignment length in this iteration
    int maxfraglen = GetFragLength(maxalnmax, maxalnmax, 0, 0, 0);
    int nlocsteps = 1;//GetMaxNFragSteps(maxalnmax, sfragstep, maxfraglen);


    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    nthrds_scinit = dim3(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    nblcks_scinit = dim3(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs, nlocsteps);

    //execution configuration for initialization:
    //each block processes one query and CUS1_TBINITSP_CCDINIT_XFCT references:
    dim3 nthrds_init(CUS1_TBINITSP_CCDINIT_XDIM,1,1);
    dim3 nblcks_init(
        (ndbCstrs + CUS1_TBINITSP_CCDINIT_XFCT - 1)/CUS1_TBINITSP_CCDINIT_XFCT,
        nqystrs, nlocsteps);

    //execution configuration for reduction:
    //block processes CUS1_TBINITSP_CCMCALC_XDIMLGL positions of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_ccmtx(CUS1_TBINITSP_CCMCALC_XDIM,1,1);
    dim3 nblcks_ccmtx(
        (minlenmax + CUS1_TBINITSP_CCMCALC_XDIMLGL - 1)/CUS1_TBINITSP_CCMCALC_XDIMLGL,
        ndbCstrs, nqystrs * nlocsteps);

    //execution configuration for reformatting data:
    //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
    dim3 nthrds_copyto(CUS1_TBINITSP_CCMCOPY_N,myhdmax(16,(int)twmvEndOfCCDataExt),1);
    dim3 nblcks_copyto(
        (ndbCstrs + CUS1_TBINITSP_CCMCOPY_N - 1)/CUS1_TBINITSP_CCMCOPY_N,
        nqystrs, nlocsteps);

    //execution configuration for calculating transformation matrices:
    //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
    dim3 nthrds_tfm(CUS1_TBSP_TFM_N,1,1);
    dim3 nblcks_tfm(
        (ndbCstrs + CUS1_TBSP_TFM_N - 1)/CUS1_TBSP_TFM_N,
        nqystrs, nlocsteps);

    //execution configuration for reformatting data:
    //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
    dim3 nthrds_copyfrom(CUS1_TBINITSP_CCMCOPY_N,myhdmax(16,(int)nTTranformMatrix),1);
    dim3 nblcks_copyfrom(
        (ndbCstrs + CUS1_TBINITSP_CCMCOPY_N - 1)/CUS1_TBINITSP_CCMCOPY_N,
        nqystrs, nlocsteps);

    //execution configuration for calculating scores (reduction):
    //block processes CUS1_TBSP_SCORE_XDIMLGL positions of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_scores(CUS1_TBSP_SCORE_XDIM,1,1);
    dim3 nblcks_scores(
        (minlenmax + CUS1_TBSP_SCORE_XDIMLGL - 1)/CUS1_TBSP_SCORE_XDIMLGL,
        ndbCstrs, nqystrs * nlocsteps);

    //execution configuration for saving best performing transformation matrices:
    //each block processes one query and CUS1_TBINITSP_TMSAVE_XFCT references:
    dim3 nthrds_savetm(CUS1_TBINITSP_TMSAVE_XDIM,1,1);
    dim3 nblcks_savetm(
        (ndbCstrs + CUS1_TBINITSP_TMSAVE_XFCT - 1)/CUS1_TBINITSP_TMSAVE_XFCT,
        nqystrs, nlocsteps);

    //execution configuration for minimum score reduction:
    //block processes all positions of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_findd2(CUS1_TBINITSP_FINDD02_ITRD_XDIM,1,1);
    dim3 nblcks_findd2(ndbCstrs, nqystrs, nlocsteps);

    //execution configuration for convergence verification:
    //each block processes one query and CUS1_TBINITSP_CCDCONV_XFCT references:
    dim3 nthrds_conv(CUS1_TBINITSP_CCDCONV_XDIM,1,1);
    dim3 nblcks_conv(
        (ndbCstrs + CUS1_TBINITSP_CCDCONV_XFCT - 1)/CUS1_TBINITSP_CCDCONV_XFCT,
        nqystrs, nlocsteps);


    SetCurrentFragSpecs<<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs, maxnsteps, 0/*sfragndx*/, wrkmemaux);
    MYCUDACHECKLAST;///


    //reset convergence flag;
    InitScores<INITOPT_CONVFLAG_FRAGREF>
        <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;


    stage1_refinefrag_subiter1(
        streamproc,
        maxnsteps, minfraglen, sfragstep,
        nqystrs, ndbCstrs, ndbCposs,
        wrkmem, wrkmemaux, wrkmemccd,
        nblcks_init, nthrds_init,
        nblcks_ccmtx, nthrds_ccmtx,
        nblcks_conv, nthrds_conv);


    stage1_refinefrag_subiter2(
        true/*lastsubiter*/,
        streamproc,
        maxnsteps, minfraglen, sfragstep,
        nqystrs, ndbCstrs, ndbCposs,
        tmpdpdiagbuffers,
        wrkmem, wrkmemaux, wrkmem2,
        wrkmemccd, wrkmemtm, wrkmemtmibest, tfmmem,
        0/*cit*/,
        nblcks_init, nthrds_init,
        nblcks_ccmtx, nthrds_ccmtx,
        nblcks_findd2, nthrds_findd2,
        nblcks_scinit, nthrds_scinit,
        nblcks_scores, nthrds_scores,
        nblcks_savetm, nthrds_savetm,
        nblcks_init/*nblcks_saveccd*/, nthrds_init/*nthrds_saveccd*/,
        nblcks_conv, nthrds_conv,
        nblcks_copyto, nthrds_copyto,
        nblcks_copyfrom, nthrds_copyfrom,
        nblcks_tfm, nthrds_tfm);


    //execution configuration for finding the maximum among scores 
    //calculated for each fragment factor:
    //each block processes one query and CUS1_TBSP_SCORE_MAX_XDIM references:
    dim3 nthrds_scmax(CUS1_TBSP_SCORE_MAX_XDIM,CUS1_TBSP_SCORE_MAX_YDIM,1);
    dim3 nblcks_scmax(
        (ndbCstrs + CUS1_TBSP_SCORE_MAX_XDIM - 1)/CUS1_TBSP_SCORE_MAX_XDIM,
        nqystrs, 1);

    SaveBestScoreAndTMAmongBests<true/*WRITEFRAGINFO*/>
        <<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
            ndbCstrs,  maxnsteps, maxnsteps,  wrkmemtmibest, tfmmem, wrkmemaux);
    MYCUDACHECKLAST;
}










// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stage1_refinefrag: refine identified fragments to improve the scores of 
// superposition;
// CONDITIONAL, template parameter, flag of writing scores on ly if they're
// greater at the same location;
// SECONDARYUPDATE, indication of whether and how secondary update of best scores is done;
// fragbydp, flag of whether fragments are identified by DP;
// nmaxconvit, maximum number of iterations for convergence check;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
template<bool CONDITIONAL, int SECONDARYUPDATE>
void stage1::stage1_refinefrag(
    std::map<CGKey,MyCuGraph>& /* stgraphs */,
    const int fragbydp,
    const int nmaxconvit,
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint minfraglen,
    uint nqystrs, uint ndbCstrs,
    uint nqyposs, uint ndbCposs,
    uint qystr1len, uint dbstr1len,
    uint qystrnlen, uint dbstrnlen,
    uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ /* wrkmem */,
    float* __restrict__ /* wrkmemccd */,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ /* wrkmem2 */,
    float* __restrict__ tfmmem)
{
//     if(fragbydp == stg1REFINE_INITIAL)
//         // NOTE: this version is much slower due to an enormous number of 
//         // kernel launches and overwrites field tawmvSubFragNdxCurrent, what
//         // leads to inconsistencies in the final stage of alignment refinement!
//         stage1_refinefrag_helper(
//             stgraphs, fragbydp, nmaxconvit, streamproc,
//             maxnsteps, minfraglen, nqystrs, ndbCstrs,
//             nqyposs, ndbCposs, qystr1len, dbstr1len,
//             qystrnlen, dbstrnlen, dbxpad,
//             tmpdpdiagbuffers, tmpdpalnpossbuffer,
//             wrkmem, wrkmemccd, wrkmemtm, wrkmemtmibest,
//             wrkmemaux, wrkmem2, tfmmem);
//     else
        stage1_refinefrag_helper2<CONDITIONAL,SECONDARYUPDATE>(
            fragbydp, nmaxconvit, streamproc,
            maxnsteps, minfraglen, nqystrs, ndbCstrs,
            nqyposs, ndbCposs, qystr1len, dbstr1len,
            qystrnlen, dbstrnlen, dbxpad,
            tmpdpdiagbuffers, tmpdpalnpossbuffer,
            wrkmemtm, wrkmemtmibest, wrkmemaux, tfmmem);
}

// Instantiations
// 
#define INSTANTIATE_stage1__stage1_refinefrag(tpCONDITIONAL,SECONDARYUPDATE) \
    template void stage1::stage1_refinefrag<tpCONDITIONAL,SECONDARYUPDATE>( \
        std::map<CGKey,MyCuGraph>& stgraphs, \
        const int fragbydp, \
        const int nmaxconvit, \
        cudaStream_t streamproc, \
        const uint maxnsteps, \
        const uint minfraglen, \
        uint nqystrs, uint ndbCstrs, \
        uint nqyposs, uint ndbCposs, \
        uint qystr1len, uint dbstr1len, \
        uint qystrnlen, uint dbstrnlen, \
        uint dbxpad, \
        float* __restrict__ tmpdpdiagbuffers, \
        float* __restrict__ tmpdpalnpossbuffer, \
        float* __restrict__ wrkmem, \
        float* __restrict__ wrkmemccd, \
        float* __restrict__ wrkmemtm, \
        float* __restrict__ wrkmemtmibest, \
        float* __restrict__ wrkmemaux, \
        float* __restrict__ wrkmem2, \
        float* __restrict__ tfmmem);

INSTANTIATE_stage1__stage1_refinefrag(true,SECONDARYUPDATE_NOUPDATE);
INSTANTIATE_stage1__stage1_refinefrag(false,SECONDARYUPDATE_NOUPDATE);
INSTANTIATE_stage1__stage1_refinefrag(false,SECONDARYUPDATE_UNCONDITIONAL);
INSTANTIATE_stage1__stage1_refinefrag(false,SECONDARYUPDATE_CONDITIONAL);

// -------------------------------------------------------------------------
// stage1_refinefrag_helper2: refine identified fragments to improve the 
// superposition score; complete version;
// CONDITIONAL, template parameter, flag of writing scores on ly if they're
// greater at the same location;
// SECONDARYUPDATE, indication of whether and how secondary update of best scores is done;
// fragbydp, flag of whether fragments are identified by DP;
// nmaxconvit, maximum number of iterations for convergence check;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
template<bool CONDITIONAL, int SECONDARYUPDATE>
void stage1::stage1_refinefrag_helper2(
    const int fragbydp,
    const int nmaxconvit,
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint minfraglen,
    uint nqystrs, uint ndbCstrs,
    uint /* nqyposs */, uint ndbCposs,
    uint qystr1len, uint dbstr1len,
    uint /*qystrnlen*/, uint /*dbstrnlen*/,
    uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tfmmem)
{
    static std::string preamb = "stage1::stage1_refinefrag_helper2: ";
    const int symmetric = CLOptions::GetC_SYMMETRIC();
    //NOTE: minimum of the largest structures to compare is assumed >=3
    //minimum length among largest
    int minlenmax = myhdmin(qystr1len, dbstr1len);
//     //minimum length among smallest
//     int minlenmin = myhdmin(qystrnlen, dbstrnlen);
//     int minalnmin = myhdmax(minlenmin >> 1, 5);
    //maximum alignment length can be minimum of the lengths
    int maxalnmax = minlenmax;
//     int n1 = minalnmin - dbstr1len;
//     int n2 = qystr1len - minalnmin;

    //maximum number of subdivisions of identified fragments
    // (for all structures in the chunk)
    const int nmaxsubfrags = FRAGREF_NMAXSUBFRAGS;
    // sfragstep, step to traverse subfragments;
    const int sfragstep = FRAGREF_SFRAGSTEP;
    //NOTE: frag info is written only at the stage of final refinement
    //(before production); this saves a lot of simple writes to gmem!
    constexpr bool WRITEFRAGINFO = false;//true

    if(fragbydp == stg1REFINE_INITIAL) {
        //execution configuration for scores initialization:
        //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
        dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
        dim3 nblcks_scinit(
            (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
            nqystrs, maxnsteps);

        //initialize memory for best scores only;
        InitScores<INITOPT_BEST><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
        MYCUDACHECKLAST;
    }

//     //reset convergence flag; NOTE: unused; ensure reset
//     InitScores<INITOPT_CONVFLAG_FRAGREF>
//         <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
//         ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
//     MYCUDACHECKLAST;

    int nlocsteps = 0;
///     for(int sfragndx = 0; sfragndx < nmaxsubfrags; sfragndx++)
///     {   //maximum alignment length for this frag index
///         int maxfraglen = GetFragLength(maxalnmax, maxalnmax, 0, 0, sfragndx);
///         if(maxfraglen < 1) break;
///         nlocsteps += GetMaxNFragSteps(maxalnmax, sfragstep, maxfraglen);
///     }
    nlocsteps = GetMaxNFragSteps(maxalnmax, sfragstep, minfraglen);
    nlocsteps *= nmaxsubfrags;//total number across all fragment lengths

    if(nlocsteps < 1 || (int)maxnsteps < nlocsteps)
        throw MYRUNTIME_ERROR(preamb +
        "Invalid number of superposition tests: "+std::to_string(nlocsteps));



    //execution configuration for complete refinement:
    //block processes one subfragment of a certain length of one query-reference pair:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
    dim3 nthrds_arcmpl(CUS1_TBINITSP_COMPLETEREFINE_XDIM,1,1);
    dim3 nblcks_arcmpl(ndbCstrs, nlocsteps, nqystrs);

    if(fragbydp == stg1REFINE_INITIAL) {
        if(symmetric)
            FragmentBasedAlignmentRefinement<WRITEFRAGINFO,true/*TFM_DINV*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    nmaxconvit, ndbCstrs, ndbCposs,
                    nmaxsubfrags, maxnsteps, sfragstep, maxalnmax,
                    tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
        else
            FragmentBasedAlignmentRefinement<WRITEFRAGINFO,false/*TFM_DINV*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    nmaxconvit, ndbCstrs, ndbCposs,
                    nmaxsubfrags, maxnsteps, sfragstep, maxalnmax,
                    tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
    } else if(fragbydp == stg1REFINE_INITIAL_DP) {
        if(symmetric)
            FragmentBasedDPAlignmentRefinement<WRITEFRAGINFO,CONDITIONAL,true/*TFM_DINV*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    false/*readlocalconv*/,
                    nmaxconvit, ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnsteps, sfragstep, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
        else
            FragmentBasedDPAlignmentRefinement<WRITEFRAGINFO,CONDITIONAL,false/*TFM_DINV*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    false/*readlocalconv*/,
                    nmaxconvit, ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnsteps, sfragstep, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
    } else if(fragbydp == stg1REFINE_ITERATIVE_DP) {
        if(symmetric)
            FragmentBasedDPAlignmentRefinement<WRITEFRAGINFO,CONDITIONAL,true/*TFM_DINV*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    true/*readlocalconv*/,
                    nmaxconvit, ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnsteps, sfragstep, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
        else
            FragmentBasedDPAlignmentRefinement<WRITEFRAGINFO,CONDITIONAL,false/*TFM_DINV*/>
                <<<nblcks_arcmpl,nthrds_arcmpl,0,streamproc>>>(
                    true/*readlocalconv*/,
                    nmaxconvit, ndbCstrs, ndbCposs, dbxpad,
                    nmaxsubfrags, maxnsteps, sfragstep, maxalnmax,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemtmibest, wrkmemaux);
    } else
        throw MYRUNTIME_ERROR(preamb +
        "Unknown alignment refinement strategy: "+std::to_string(fragbydp));



    //execution configuration for finding the maximum among scores 
    //calculated for each fragment factor:
    //each block processes one query and CUS1_TBSP_SCORE_MAX_XDIM references:
    dim3 nthrds_scmax(CUS1_TBSP_SCORE_MAX_XDIM,CUS1_TBSP_SCORE_MAX_YDIM,1);
    dim3 nblcks_scmax(
        (ndbCstrs + CUS1_TBSP_SCORE_MAX_XDIM - 1)/CUS1_TBSP_SCORE_MAX_XDIM,
        nqystrs, 1);

    SaveBestScoreAndTMAmongBests
        <WRITEFRAGINFO,true/*GRANDUPDATE*/,false/*FORCEWRITEFRAGINFO*/,SECONDARYUPDATE>
        <<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
            ndbCstrs,  maxnsteps, nlocsteps,
            wrkmemtmibest, tfmmem, wrkmemaux,  wrkmemtm);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stage1_refinefrag_helper: refine identified fragments to improve the 
// scores of superposition;
// NOTE: this version is much slower due to an enormous number of 
// NOTE: kernel launches and overwrites field tawmvSubFragNdxCurrent, what
// NOTE: leads to inconsistencies in the final stage of alignment refinement!
// fragbydp, flag of whether fragments are identified by DP;
// nmaxconvit, maximum number of iterations for convergence check;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void stage1::stage1_refinefrag_helper(
    std::map<CGKey,MyCuGraph>& stgraphs,
    const int fragbydp,
    const int nmaxconvit,
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint minfraglen,
    uint nqystrs, uint ndbCstrs,
    uint /* nqyposs */, uint ndbCposs,
    uint qystr1len, uint dbstr1len,
    uint qystrnlen, uint dbstrnlen,
    uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemccd,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ tfmmem)
{
    static std::string preamb = "stage1::stage1_refinefrag_helper: ";
    //NOTE: minimum of the largest structures to compare is assumed >=3
    //minimum length among largest
    int minlenmax = myhdmin(qystr1len, dbstr1len);
    //minimum length among smallest
    int minlenmin = myhdmin(qystrnlen, dbstrnlen);
    int minalnmin = myhdmax(minlenmin >> 1, 5);
    //maximum alignment length can be minimum of the lengths
    int maxalnmax = minlenmax;
//     int n1 = minalnmin - dbstr1len;
//     int n2 = qystr1len - minalnmin;

    //maximum number of subdivisions of identified fragments
    // (for all structures in the chunk)
    const int nmaxsubfrags = FRAGREF_NMAXSUBFRAGS;
    // sfragstep, step to traverse subfragments;
    const int sfragstep = FRAGREF_SFRAGSTEP;


    //execution configuration for tfm matrix initialization:
    //each block processes one query and CUS1_TBINITSP_TFMINIT_XFCT references:
    dim3 nthrds_tfminitibest(CUS1_TBINITSP_TFMINIT_XDIM,1,1);
    dim3 nblcks_tfminitibest(
        (ndbCstrs + CUS1_TBINITSP_TFMINIT_XFCT - 1)/CUS1_TBINITSP_TFMINIT_XFCT,
        nqystrs, maxnsteps);

    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs, maxnsteps);

    //NOTE: do not initialize here for keeping the immediate last 
    //NOTE: matrix in the case of convergence;
    //InitTfmMatrices<<<nblcks_tfminitibest,nthrds_tfminitibest,0,streamproc>>>(
    //   ndbCstrs, maxnsteps, minfraglen, sfragstep, false/*checkfragos*/,
    //   wrkmemtmibest);
    //MYCUDACHECKLAST;

    //initialize memory for best scores only;
    InitScores<INITOPT_BEST><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;


    for(int sfragndx = 0; sfragndx < nmaxsubfrags; sfragndx++)
    {
        //maximum alignment length in this iteration
        int maxfraglen = GetFragLength(maxalnmax, maxalnmax, 0, 0, sfragndx);

        if(maxfraglen < 1) break;

        int nlocsteps = GetMaxNFragSteps(maxalnmax, sfragstep, maxfraglen);


        //key to uniquely determine Cuda graph
        CGKey cgkey(fragbydp, nmaxconvit, sfragndx);
        auto foundpair = stgraphs.find(cgkey);


        //execution configuration for scores initialization:
        //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
        nthrds_scinit = dim3(CUS1_TBSP_SCORE_SET_XDIM,1,1);
        nblcks_scinit = dim3(
            (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
            nqystrs, nlocsteps);

        //execution configuration for initialization:
        //each block processes one query and CUS1_TBINITSP_CCDINIT_XFCT references:
        dim3 nthrds_init(CUS1_TBINITSP_CCDINIT_XDIM,1,1);
        dim3 nblcks_init(
            (ndbCstrs + CUS1_TBINITSP_CCDINIT_XFCT - 1)/CUS1_TBINITSP_CCDINIT_XFCT,
            nqystrs, nlocsteps);

        //execution configuration for reduction:
        //block processes CUS1_TBINITSP_CCMCALC_XDIMLGL positions of one query-reference pair:
        //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
        dim3 nthrds_ccmtx(CUS1_TBINITSP_CCMCALC_XDIM,1,1);
        dim3 nblcks_ccmtx(
            (minlenmax + CUS1_TBINITSP_CCMCALC_XDIMLGL - 1)/CUS1_TBINITSP_CCMCALC_XDIMLGL,
            ndbCstrs, nqystrs * nlocsteps);

        //execution configuration for reformatting data:
        //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
        dim3 nthrds_copyto(CUS1_TBINITSP_CCMCOPY_N,myhdmax(16,(int)twmvEndOfCCDataExt),1);
        dim3 nblcks_copyto(
            (ndbCstrs + CUS1_TBINITSP_CCMCOPY_N - 1)/CUS1_TBINITSP_CCMCOPY_N,
            nqystrs, nlocsteps);

        //execution configuration for calculating transformation matrices:
        //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
        dim3 nthrds_tfm(CUS1_TBSP_TFM_N,1,1);
        dim3 nblcks_tfm(
            (ndbCstrs + CUS1_TBSP_TFM_N - 1)/CUS1_TBSP_TFM_N,
            nqystrs, nlocsteps);

        //execution configuration for reformatting data:
        //each block processes one query and CUS1_TBINITSP_CCMCOPY_N references:
        dim3 nthrds_copyfrom(CUS1_TBINITSP_CCMCOPY_N,myhdmax(16,(int)nTTranformMatrix),1);
        dim3 nblcks_copyfrom(
            (ndbCstrs + CUS1_TBINITSP_CCMCOPY_N - 1)/CUS1_TBINITSP_CCMCOPY_N,
            nqystrs, nlocsteps);

        //execution configuration for calculating scores (reduction):
        //block processes CUS1_TBSP_SCORE_XDIMLGL positions of one query-reference pair:
        //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
        dim3 nthrds_scores(CUS1_TBSP_SCORE_XDIM,1,1);
        dim3 nblcks_scores(
            (minlenmax + CUS1_TBSP_SCORE_XDIMLGL - 1)/CUS1_TBSP_SCORE_XDIMLGL,
            ndbCstrs, nqystrs * nlocsteps);

        //execution configuration for saving best performing transformation matrices:
        //each block processes one query and CUS1_TBINITSP_TMSAVE_XFCT references:
        dim3 nthrds_savetm(CUS1_TBINITSP_TMSAVE_XDIM,1,1);
        dim3 nblcks_savetm(
            (ndbCstrs + CUS1_TBINITSP_TMSAVE_XFCT - 1)/CUS1_TBINITSP_TMSAVE_XFCT,
            nqystrs, nlocsteps);

        //execution configuration for minimum score reduction:
        //block processes all positions of one query-reference pair:
        //NOTE: ndbCstrs and nqystrs cannot be greater than 65535: ensured by JobDispatcher
        dim3 nthrds_findd2(CUS1_TBINITSP_FINDD02_ITRD_XDIM,1,1);
        dim3 nblcks_findd2(ndbCstrs, nqystrs, nlocsteps);

        //execution configuration for convergence verification:
        //each block processes one query and CUS1_TBINITSP_CCDCONV_XFCT references:
        dim3 nthrds_conv(CUS1_TBINITSP_CCDCONV_XDIM,1,1);
        dim3 nblcks_conv(
            (ndbCstrs + CUS1_TBINITSP_CCDCONV_XFCT - 1)/CUS1_TBINITSP_CCDCONV_XFCT,
            nqystrs, nlocsteps);


        SetCurrentFragSpecs<<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCstrs, maxnsteps, sfragndx, wrkmemaux);
        MYCUDACHECKLAST;///


        //{{GRAPH section
        if(foundpair != stgraphs.end()) {
            foundpair->second.Launch(streamproc);
            continue;
        }
        //key not found
        stgraphs.insert(std::make_pair(cgkey, MyCuGraph()));
        foundpair = stgraphs.find(cgkey);
        if(foundpair == stgraphs.end())
            throw MYRUNTIME_ERROR(preamb + "CUDA Graph not found.");
        foundpair->second.BeginCapture(streamproc);
        //}}


        //reset convergence flag;
        InitScores<INITOPT_CONVFLAG_FRAGREF>
            <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
        MYCUDACHECKLAST;


        if(fragbydp == stg1REFINE_INITIAL_DP || fragbydp == stg1REFINE_ITERATIVE_DP)
            stage1_refinefrag_dp_subiter1(
                streamproc,
                maxnsteps, minfraglen, sfragstep,
                nqystrs, ndbCstrs, ndbCposs, dbxpad,
                wrkmem, wrkmemaux, wrkmemccd, tmpdpalnpossbuffer,
                nblcks_init, nthrds_init,
                nblcks_ccmtx, nthrds_ccmtx,
                nblcks_conv, nthrds_conv);
        else
            stage1_refinefrag_subiter1(
                streamproc,
                maxnsteps, minfraglen, sfragstep,
                nqystrs, ndbCstrs, ndbCposs,
                wrkmem, wrkmemaux, wrkmemccd,
                nblcks_init, nthrds_init,
                nblcks_ccmtx, nthrds_ccmtx,
                nblcks_conv, nthrds_conv);


        for(int cit = 0; cit < nmaxconvit + 2; cit++) {
            if(fragbydp == stg1REFINE_INITIAL_DP || fragbydp == stg1REFINE_ITERATIVE_DP)
                stage1_refinefrag_dp_subiter2(
                    fragbydp,
                    nmaxconvit + 1 <= cit,
                    streamproc,
                    maxnsteps, minfraglen, sfragstep,
                    nqystrs, ndbCstrs, ndbCposs, dbxpad,
                    tmpdpdiagbuffers,
                    tmpdpalnpossbuffer,
                    wrkmem, wrkmemaux, wrkmem2,
                    wrkmemccd, wrkmemtm, wrkmemtmibest, tfmmem,
                    cit,
                    nblcks_init, nthrds_init,
                    nblcks_ccmtx, nthrds_ccmtx,
                    nblcks_findd2, nthrds_findd2,
                    nblcks_scinit, nthrds_scinit,
                    nblcks_scores, nthrds_scores,
                    nblcks_savetm, nthrds_savetm,
                    nblcks_init/*nblcks_saveccd*/, nthrds_init/*nthrds_saveccd*/,
                    nblcks_conv, nthrds_conv,
                    nblcks_copyto, nthrds_copyto,
                    nblcks_copyfrom, nthrds_copyfrom,
                    nblcks_tfm, nthrds_tfm);
            else
                stage1_refinefrag_subiter2(
                    nmaxconvit + 1 <= cit,
                    streamproc,
                    maxnsteps, minfraglen, sfragstep,
                    nqystrs, ndbCstrs, ndbCposs,
                    tmpdpdiagbuffers,
                    wrkmem, wrkmemaux, wrkmem2,
                    wrkmemccd, wrkmemtm, wrkmemtmibest, tfmmem,
                    cit,
                    nblcks_init, nthrds_init,
                    nblcks_ccmtx, nthrds_ccmtx,
                    nblcks_findd2, nthrds_findd2,
                    nblcks_scinit, nthrds_scinit,
                    nblcks_scores, nthrds_scores,
                    nblcks_savetm, nthrds_savetm,
                    nblcks_init/*nblcks_saveccd*/, nthrds_init/*nthrds_saveccd*/,
                    nblcks_conv, nthrds_conv,
                    nblcks_copyto, nthrds_copyto,
                    nblcks_copyfrom, nthrds_copyfrom,
                    nblcks_tfm, nthrds_tfm);
        }//for(cit)


        //{{GRAPH section
        foundpair->second.EndCaptureInstantiate(streamproc);
        foundpair->second.Launch(streamproc);
        //}}

    }//for(sfragndx)


    //execution configuration for finding the maximum among scores 
    //calculated for each fragment factor:
    //each block processes one query and CUS1_TBSP_SCORE_MAX_XDIM references:
    dim3 nthrds_scmax(CUS1_TBSP_SCORE_MAX_XDIM,CUS1_TBSP_SCORE_MAX_YDIM,1);
    dim3 nblcks_scmax(
        (ndbCstrs + CUS1_TBSP_SCORE_MAX_XDIM - 1)/CUS1_TBSP_SCORE_MAX_XDIM,
        nqystrs, 1);

    if(fragbydp != stg1REFINE_ITERATIVE_DP)
        SaveBestScoreAndTMAmongBests<true/*WRITEFRAGINFO*/>
            <<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
                ndbCstrs,  maxnsteps, maxnsteps,  wrkmemtmibest, tfmmem, wrkmemaux);
    else
        SaveBestScoreAndTMAmongBests<false/*WRITEFRAGINFO*/>
            <<<nblcks_scmax,nthrds_scmax,0,streamproc>>>(
                ndbCstrs,  maxnsteps, maxnsteps,  wrkmemtmibest, tfmmem, wrkmemaux);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stage1_refinefrag_subiter1: subiteration 1 of stage1 for refining the 
// boundaries of initially identified fragments and saving transformation 
// matrices which imply best scores;
//
inline
void stage1::stage1_refinefrag_subiter1(
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint /*minfraglen*/,
    const int sfragstep,
    uint nqystrs, uint ndbCstrs, uint /* ndbCposs */,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmemccd,
    //
    const dim3& /* nblcks_init */, const dim3& /* nthrds_init */,
    const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
    const dim3& nblcks_conv, const dim3& nthrds_conv)
{
    //initialize memory for calculating cross covariance matrices
    // (required for each iteration);
//     InitCCData<CHCKCONV_NOCHECK><<<nblcks_init,nthrds_init,0,streamproc>>>(
//         ndbCstrs,  maxnsteps,  wrkmem, wrkmemaux);
    InitCopyCheckConvergence64Refined<CC64Action_InitCCData>
        <<<nblcks_conv,nthrds_conv,0,streamproc>>>(
            ndbCstrs,  maxnsteps, sfragstep,
            wrkmem, wrkmemccd, wrkmemaux);
    MYCUDACHECKLAST;

    //calculate cross-covariance matrices with unrolling
    CalcCCMatrices64Refined<<<nblcks_ccmtx,nthrds_ccmtx,0,streamproc>>>(
        nqystrs, ndbCstrs,  maxnsteps, sfragstep,  wrkmemaux, wrkmem);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stage1_refinefrag_subiter2: subiteration 2 of stage1 for refining best 
// scoring aligned fragments: unconditionally calculate 
// and save scores, calculate cross-covariances for alignment positions 
// within given distances, and optionally calculate rotation matrices
//
inline
void stage1::stage1_refinefrag_subiter2(
    bool lastsubiter,
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint minfraglen,
    const int sfragstep,
    uint nqystrs, uint ndbCstrs, uint ndbCposs,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemccd,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ /*tfmmem*/,
    //
    int cit,
    const dim3& /* nblcks_init */, const dim3& /* nthrds_init */,
    const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
    const dim3& nblcks_findd2, const dim3& nthrds_findd2,
    const dim3& nblcks_scinit, const dim3& nthrds_scinit,
    const dim3& nblcks_scores, const dim3& nthrds_scores,
    const dim3& nblcks_savetm, const dim3& nthrds_savetm,
    const dim3& /* nblcks_saveccd */, const dim3& /* nthrds_saveccd */,
    const dim3& nblcks_conv, const dim3& nthrds_conv,
    const dim3& nblcks_copyto, const dim3& nthrds_copyto,
    const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
    const dim3& nblcks_tfm, const dim3& nthrds_tfm)
{
    //copy CC data to section 2 of working memory to enable
    // efficient structure-specific calculation
    CopyCCDataToWrkMem2Refined<<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
        ndbCstrs,  maxnsteps, sfragstep,  wrkmemaux, wrkmem, wrkmem2);
    MYCUDACHECKLAST;

    CalcTfmMatrices<<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
        ndbCstrs, maxnsteps, wrkmem2);
    MYCUDACHECKLAST;

    //copy CC data from section 2 of working memory back for 
    // efficient calculation
    CopyTfmMtsFromWrkMem2<<<nblcks_copyfrom,nthrds_copyfrom,0,streamproc>>>(
        ndbCstrs,  maxnsteps,  wrkmem2/*in*/, wrkmemtm/*tfmmem:out*/);
    MYCUDACHECKLAST;

    //initialize memory for current scores only;
    InitScores<INITOPT_CURRENT><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    //calculate scores and temporarily save distances
    //NOTE: always check convergence (required for later stages) 
    //NOTE: irrespective of cit<1
    if(cit < 1) {
        CalcScoresUnrlRefined<SAVEPOS_SAVE,CHCKCONV_CHECK>
            <<<nblcks_scores,nthrds_scores,0,streamproc>>>(
                nqystrs, ndbCstrs, ndbCposs,  maxnsteps, sfragstep,
                wrkmemtm, wrkmemaux, tmpdpdiagbuffers);
    } else {
        CalcScoresUnrlRefined<SAVEPOS_SAVE,CHCKCONV_CHECK>
            <<<nblcks_scores,nthrds_scores,0,streamproc>>>(
                nqystrs, ndbCstrs, ndbCposs,  maxnsteps, sfragstep,
                wrkmemtm, wrkmemaux, tmpdpdiagbuffers);
    }
    MYCUDACHECKLAST;

    //save scores and transformation matrices
    SaveBestScoreAndTM<true/*WRITEFRAGINFO*/>
        <<<nblcks_savetm,nthrds_savetm,0,streamproc>>>(
            ndbCstrs,  maxnsteps, sfragstep,
            wrkmemtm, wrkmemtmibest, wrkmemaux);
    MYCUDACHECKLAST;

    if(lastsubiter) return;

    //find required minimum distances for adjusting rotation
    if(cit < 1)
        FindD02ThresholdsCCMRefined<READCNST_CALC>
            <<<nblcks_findd2,nthrds_findd2,0,streamproc>>>(
                ndbCstrs, ndbCposs,  maxnsteps, sfragstep,
                tmpdpdiagbuffers, wrkmemaux);
#if DO_FINDD02_DURING_REFINEFRAG == 1
    else 
        FindD02ThresholdsCCMRefined<READCNST_CALC2>
            <<<nblcks_findd2,nthrds_findd2,0,streamproc>>>(
                ndbCstrs, ndbCposs,  maxnsteps, sfragstep,
                tmpdpdiagbuffers, wrkmemaux);
#endif
    MYCUDACHECKLAST;

    //initialize memory before calculating cross-covariance matrices;
    //check whether a pair requires initialization
    //NOTE: always check convergence (required for later stages) 
    //NOTE: irrespective of cit<1
//     if(cit < 1)
//         InitCCData<CHCKCONV_CHECK><<<nblcks_init,nthrds_init,0,streamproc>>>(
//             ndbCstrs,  maxnsteps,  wrkmem, wrkmemaux);
//     else
//         InitCCData<CHCKCONV_CHECK><<<nblcks_init,nthrds_init,0,streamproc>>>(
//             ndbCstrs,  maxnsteps,  wrkmem, wrkmemaux);
    InitCopyCheckConvergence64Refined<CC64Action_InitCCData>
        <<<nblcks_conv,nthrds_conv,0,streamproc>>>(
            ndbCstrs,  maxnsteps, sfragstep,
            wrkmem, wrkmemccd, wrkmemaux);
    MYCUDACHECKLAST;

    //calculate cross-covariance matrices with unrolling;
    //include only positions within given distances
    if(cit < 1)
        CalcCCMatrices64RefinedExtended<READCNST_CALC>
            <<<nblcks_ccmtx,nthrds_ccmtx,0,streamproc>>>(
                nqystrs, ndbCstrs, ndbCposs,  maxnsteps, sfragstep,
                tmpdpdiagbuffers, wrkmemaux, wrkmem);
    else
        CalcCCMatrices64RefinedExtended<READCNST_CALC2>
            <<<nblcks_ccmtx,nthrds_ccmtx,0,streamproc>>>(
                nqystrs, ndbCstrs, ndbCposs,  maxnsteps, sfragstep,
                tmpdpdiagbuffers, wrkmemaux, wrkmem);
    MYCUDACHECKLAST;

    if(0 < cit)
        InitCopyCheckConvergence64Refined<CC64Action_Convergence_CopyCCData>
            <<<nblcks_conv,nthrds_conv,0,streamproc>>>(
                ndbCstrs,  maxnsteps, sfragstep,
                wrkmem, wrkmemccd, wrkmemaux);
    else
        //save last cross-covarinace data
//         CopyCCDataRefined<<<nblcks_saveccd,nthrds_saveccd,0,streamproc>>>(
//             ndbCstrs,  maxnsteps, sfragstep, wrkmem, wrkmemccd, wrkmemaux);
        InitCopyCheckConvergence64Refined<CC64Action_CopyCCData>
            <<<nblcks_conv,nthrds_conv,0,streamproc>>>(
                ndbCstrs,  maxnsteps, sfragstep,
                wrkmem, wrkmemccd, wrkmemaux);
    MYCUDACHECKLAST;
}





// -------------------------------------------------------------------------
// stage1_refinefrag_dp_subiter1: subiteration 1 of stage1 for refining the 
// boundaries of fragments identified by DP and saving transformation 
// matrices which imply best scores;
//
inline
void stage1::stage1_refinefrag_dp_subiter1(
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint /*minfraglen*/,
    const int sfragstep,
    uint nqystrs, uint ndbCstrs, uint ndbCposs,
    uint dbxpad,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmemccd,
    float* __restrict__ tmpdpalnpossbuffer,
    //
    const dim3& /* nblcks_init */, const dim3& /* nthrds_init */,
    const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
    const dim3& nblcks_conv, const dim3& nthrds_conv)
{
    //initialize memory for calculating cross covariance matrices
    // (required for each iteration);
//     InitCCData<CHCKCONV_CHECK><<<nblcks_init,nthrds_init,0,streamproc>>>(
//         ndbCstrs,  maxnsteps,  wrkmem, wrkmemaux);
    InitCopyCheckConvergence64_DPRefined<CC64Action_InitCCData>
        <<<nblcks_conv,nthrds_conv,0,streamproc>>>(
            ndbCstrs,  maxnsteps, sfragstep,
            wrkmem, wrkmemccd, wrkmemaux);
    MYCUDACHECKLAST;

    //calculate cross-covariance matrices with unrolling
    CalcCCMatrices64_DPRefined
        <<<nblcks_ccmtx,nthrds_ccmtx,0,streamproc>>>(
            nqystrs, ndbCstrs, ndbCposs, dbxpad,  maxnsteps, sfragstep,
            wrkmemaux, tmpdpalnpossbuffer, wrkmem);
    MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// stage1_refinefrag_dp_subiter2: subiteration 2 of stage1 for refining best 
// scoring fragments identified and aligned by DP: unconditionally calculate 
// and save scores, calculate cross-covariances for alignment positions 
// within given distances, and optionally calculate rotation matrices
//
inline
void stage1::stage1_refinefrag_dp_subiter2(
    int fragbydp,
    bool lastsubiter,
    cudaStream_t streamproc,
    const uint maxnsteps,
    const uint minfraglen,
    const int sfragstep,
    uint nqystrs, uint ndbCstrs, uint ndbCposs, uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ wrkmemccd,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ /*tfmmem*/,
    //
    int cit,
    const dim3& /* nblcks_init */, const dim3& /* nthrds_init */,
    const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
    const dim3& nblcks_findd2, const dim3& nthrds_findd2,
    const dim3& nblcks_scinit, const dim3& nthrds_scinit,
    const dim3& nblcks_scores, const dim3& nthrds_scores,
    const dim3& nblcks_savetm, const dim3& nthrds_savetm,
    const dim3& /* nblcks_saveccd */, const dim3& /* nthrds_saveccd */,
    const dim3& nblcks_conv, const dim3& nthrds_conv,
    const dim3& nblcks_copyto, const dim3& nthrds_copyto,
    const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
    const dim3& nblcks_tfm, const dim3& nthrds_tfm)
{
    //copy CC data to section 2 of working memory to enable
    // efficient structure-specific calculation
    CopyCCDataToWrkMem2_DPRefined
        <<<nblcks_copyto,nthrds_copyto,0,streamproc>>>(
            ndbCstrs,  maxnsteps, sfragstep,  wrkmemaux, wrkmem, wrkmem2);
    MYCUDACHECKLAST;

    CalcTfmMatrices<<<nblcks_tfm,nthrds_tfm,0,streamproc>>>(
        ndbCstrs, maxnsteps, wrkmem2);
    MYCUDACHECKLAST;

    //copy CC data from section 2 of working memory back for 
    // efficient calculation
    CopyTfmMtsFromWrkMem2<<<nblcks_copyfrom,nthrds_copyfrom,0,streamproc>>>(
        ndbCstrs,  maxnsteps,  wrkmem2/*in*/, wrkmemtm/*tfmmem:out*/);
    MYCUDACHECKLAST;

    //initialize memory for current scores only;
    InitScores<INITOPT_CURRENT><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;

    //calculate scores and temporarily save distances
    //NOTE: always check convergence (required for later stages) 
    //NOTE: irrespective of cit<1
    CalcScoresUnrl_DPRefined<SAVEPOS_SAVE,CHCKCONV_CHECK>
        <<<nblcks_scores,nthrds_scores,0,streamproc>>>(
            nqystrs, ndbCstrs, ndbCposs, dbxpad,  maxnsteps, sfragstep,
            tmpdpalnpossbuffer, wrkmemtm, wrkmemaux, tmpdpdiagbuffers);
    MYCUDACHECKLAST;

    //save scores and transformation matrices
    if(fragbydp == stg1REFINE_INITIAL_DP)
        SaveBestScoreAndTM<true/*WRITEFRAGINFO*/>
            <<<nblcks_savetm,nthrds_savetm,0,streamproc>>>(
                ndbCstrs,  maxnsteps, sfragstep,
                wrkmemtm, wrkmemtmibest, wrkmemaux);
    else
        SaveBestScoreAndTM<false/*WRITEFRAGINFO*/>
            <<<nblcks_savetm,nthrds_savetm,0,streamproc>>>(
                ndbCstrs,  maxnsteps, sfragstep,
                wrkmemtm, wrkmemtmibest, wrkmemaux);
    MYCUDACHECKLAST;

    if(lastsubiter) return;

    //find required minimum distances for adjusting rotation
    if(cit < 1)
        FindD02ThresholdsCCM_DPRefined<READCNST_CALC>
            <<<nblcks_findd2,nthrds_findd2,0,streamproc>>>(
                ndbCstrs, ndbCposs,  maxnsteps, sfragstep,
                tmpdpdiagbuffers, wrkmemaux);
#if DO_FINDD02_DURING_REFINEFRAG == 1
    else 
        FindD02ThresholdsCCM_DPRefined<READCNST_CALC2>
            <<<nblcks_findd2,nthrds_findd2,0,streamproc>>>(
                ndbCstrs, ndbCposs,  maxnsteps, sfragstep,
                tmpdpdiagbuffers, wrkmemaux);
#endif
    MYCUDACHECKLAST;

    //initialize memory before calculating cross-covariance matrices;
    //check whether a pair requires initialization
    //NOTE: always check convergence (required for later stages) 
    //NOTE: irrespective of cit<1
//     InitCCData<CHCKCONV_CHECK><<<nblcks_init,nthrds_init,0,streamproc>>>(
//         ndbCstrs,  maxnsteps,  wrkmem, wrkmemaux);
    InitCopyCheckConvergence64_DPRefined<CC64Action_InitCCData>
        <<<nblcks_conv,nthrds_conv,0,streamproc>>>(
            ndbCstrs,  maxnsteps, sfragstep,
            wrkmem, wrkmemccd, wrkmemaux);
    MYCUDACHECKLAST;

    //calculate cross-covariance matrices with unrolling;
    //include only positions within given distances
    if(cit < 1)
        CalcCCMatrices64_DPRefinedExtended<READCNST_CALC>
            <<<nblcks_ccmtx,nthrds_ccmtx,0,streamproc>>>(
                nqystrs, ndbCstrs, ndbCposs, dbxpad,  maxnsteps, sfragstep,
                tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux, wrkmem);
    else
        CalcCCMatrices64_DPRefinedExtended<READCNST_CALC2>
            <<<nblcks_ccmtx,nthrds_ccmtx,0,streamproc>>>(
                nqystrs, ndbCstrs, ndbCposs, dbxpad,  maxnsteps, sfragstep,
                tmpdpalnpossbuffer, tmpdpdiagbuffers, wrkmemaux, wrkmem);
    MYCUDACHECKLAST;

    if(0 < cit)
        InitCopyCheckConvergence64_DPRefined<CC64Action_Convergence_CopyCCData>
            <<<nblcks_conv,nthrds_conv,0,streamproc>>>(
                ndbCstrs,  maxnsteps, sfragstep,
                wrkmem, wrkmemccd, wrkmemaux);
    else
        //save last cross-covarinace data
//         CopyCCDataRefined<<<nblcks_saveccd,nthrds_saveccd,0,streamproc>>>(
//             ndbCstrs,  maxnsteps, sfragstep,  wrkmem, wrkmemccd, wrkmemaux);
        InitCopyCheckConvergence64_DPRefined<CC64Action_CopyCCData>
            <<<nblcks_conv,nthrds_conv,0,streamproc>>>(
                ndbCstrs,  maxnsteps, sfragstep,
                wrkmem, wrkmemccd, wrkmemaux);
    MYCUDACHECKLAST;
}









// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stage1_dprefine: refine ungapped alignment identified during fragment 
// boundary refinement in the previous substage by iteratively applying 
// (gapped) DP followed by the same ungapped alignment boundary refinement;
// GAP0, template parameter, flag of using gap cost 0;
// PRESCREEN, template parameter, whether to verify scores for pre-termination;
// WRKMEMTM1, use for the 1st iteration tfms saved in wrkmemtm;
// prescorethr, provisional TM-score threshold for prescreening;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
template<bool GAP0, bool PRESCREEN, bool WRKMEMTM1>
void stage1::stage1_dprefine(
    std::map<CGKey,MyCuGraph>& stgraphs,
    cudaStream_t streamproc,
    const int maxndpiters,
    const float prescorethr,
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
    uint* __restrict__ /* maxscoordsbuf */,
    char* __restrict__ btckdata,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemccd,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ tfmmem)
{
    //NOTE: minimum of the largest structures to compare is assumed >=3
    //minimum length among largest
    // int minlenmax = myhdmin(qystr1len, dbstr1len);
    //minimum length among smallest
    // int minlenmin = myhdmin(qystrnlen, dbstrnlen);
    // int minalnmin = myhdmax(minlenmin >> 1, 5);
//     //maximum alignment length can be minimum of the lengths
//     int maxalnmax = minlenmax;
//     int n1 = minalnmin - dbstr1len;
//     int n2 = qystr1len - minalnmin;

//     enum{ncosts = 2};
//     static const float gcosts[ncosts] = {-0.6f, 0.0f};
    enum{ncosts = 1};
    static const float gcosts[ncosts] = {GAP0? 0.0f: -0.6f};

    constexpr bool COMPLETEAPPROACH = false;//complete DP approach
    constexpr bool vANCHORRGN = false;//using anchor region
    constexpr bool vBANDED = false;//banded alignment

    //execution configuration for scores initialization:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs, maxnsteps);

    //execution configuration for operating on scores at a fragment factor position of 0:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit0(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit0(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs, 1);

    //execution configuration for extracting matched positions
    //identified during DP:
    dim3 nthrds_mtch(CUDP_MATCHED_DIM_X,CUDP_MATCHED_DIM_Y,1);
    dim3 nblcks_mtch(ndbCstrs,nqystrs,1);


    //initialize memory for best scores only;
    InitScores<INITOPT_BEST><<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        ndbCstrs,  maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
    MYCUDACHECKLAST;


    for(int gi = 0; gi < ncosts; gi++)
    {
        //reset the score convergence flag first;
        InitScores<INITOPT_CONVFLAG_FRAGREF|INITOPT_CONVFLAG_SCOREDP>
            <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
                ndbCstrs, maxnsteps, minfraglen, false/*checkfragos*/,  wrkmemaux);
        MYCUDACHECKLAST;

        for(int dpi = 0; dpi < maxndpiters; dpi++)
        {
            RunDP<COMPLETEAPPROACH,vANCHORRGN,vBANDED,GAP0>(
                streamproc, gcosts[gi], maxnsteps, nqystrs, ndbCstrs, 
                nqyposs, ndbCposs, qystr1len, dbstr1len, dbxpad,
                tmpdpdiagbuffers, tmpdpbotbuffer, btckdata,
                ((WRKMEMTM1 && gi < 1 && dpi < 1)? wrkmemtm: wrkmemtmibest),
                wrkmemaux);

            //proces the result of DP;
            //stepnumber==0: write aligned positions at slot 0:
            BtckToMatched32x<vANCHORRGN,vBANDED>
                <<<nblcks_mtch,nthrds_mtch,0,streamproc>>>(
                    ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber*/,
                    btckdata, wrkmemaux, tmpdpalnpossbuffer);
            MYCUDACHECKLAST;


            stage1_refinefrag<false/* CONDITIONAL */>(
                stgraphs,
                stg1REFINE_ITERATIVE_DP/*fragments identified by DP*/,
                FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                streamproc,
                maxnsteps, minfraglen,
                nqystrs, ndbCstrs,
                nqyposs, ndbCposs,
                qystr1len, dbstr1len,
                qystrnlen, dbstrnlen,
                dbxpad,
                tmpdpdiagbuffers,
                tmpdpalnpossbuffer,
                wrkmem, wrkmemccd, wrkmemtm, wrkmemtmibest,
                wrkmemaux, wrkmem2, tfmmem);


            if(0 < dpi)
                CheckScoreConvergence<<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
                    ndbCstrs, maxnsteps, wrkmemaux);
            MYCUDACHECKLAST;


            if(dpi+1 < maxndpiters)
                SaveLastScore0<<<nblcks_scinit0,nthrds_scinit0,0,streamproc>>>(
                    ndbCstrs, maxnsteps, wrkmemaux);
            MYCUDACHECKLAST;

            if(PRESCREEN && maxndpiters <= dpi+1 && 0.0f < prescorethr)
                SetLowScoreConvergenceFlag<<<nblcks_scinit0,nthrds_scinit0,0,streamproc>>>(
                    prescorethr, ndbCstrs, maxnsteps, wrkmemaux);
            MYCUDACHECKLAST;
        }
    }

    //reset the score convergence flag for the steps to follow not to halt
    InitScores<INITOPT_CONVFLAG_SCOREDP>
        <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            ndbCstrs, maxnsteps, minfraglen, false/*checkfragos*/, wrkmemaux);
    MYCUDACHECKLAST;
}

// Instantiations
// 
#define INSTANTIATE_stage1__stage1_dprefine(tpGAP0,tpPRESCREEN,tpWRKMEMTM1) \
    template void stage1::stage1_dprefine<tpGAP0,tpPRESCREEN,tpWRKMEMTM1>( \
        std::map<CGKey,MyCuGraph>& stgraphs, \
        cudaStream_t streamproc, \
        const int maxndpiters, \
        const float prescore, \
        const uint maxnsteps, \
        const uint minfraglen, \
        uint nqystrs, uint ndbCstrs, \
        uint nqyposs, uint ndbCposs, \
        uint qystr1len, uint dbstr1len, \
        uint qystrnlen, uint dbstrnlen, \
        uint dbxpad, \
        float* __restrict__ tmpdpdiagbuffers, \
        float* __restrict__ tmpdpbotbuffer, \
        float* __restrict__ tmpdpalnpossbuffer, \
        uint* __restrict__ maxscoordsbuf, \
        char* __restrict__ btckdata, \
        float* __restrict__ wrkmem, \
        float* __restrict__ wrkmemccd, \
        float* __restrict__ wrkmemtm, \
        float* __restrict__ wrkmemtmibest, \
        float* __restrict__ wrkmemaux, \
        float* __restrict__ wrkmem2, \
        float* __restrict__ tfmmem);

INSTANTIATE_stage1__stage1_dprefine(true,true,false);
INSTANTIATE_stage1__stage1_dprefine(true,false,false);
INSTANTIATE_stage1__stage1_dprefine(false,false,false);
INSTANTIATE_stage1__stage1_dprefine(false,false,true);

// -------------------------------------------------------------------------
// RunTDintensiveDP: run thread block-intensive DP;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
//
template<bool COMPLETEAPPROACH, bool ANCHORRGN, bool BANDED, bool GAP0>
void stage1::RunDP(
    cudaStream_t streamproc,
    const float gapcost,
    const uint maxnsteps,
    uint nqystrs, uint ndbCstrs,
    uint nqyposs, uint ndbCposs,
    uint qystr1len, uint dbstr1len,
    uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
    char* __restrict__ btckdata,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux)
{
    if(COMPLETEAPPROACH)
        RunCompleteDP<ANCHORRGN,BANDED,GAP0>(
            streamproc,
            gapcost, maxnsteps, nqystrs, ndbCstrs, 
            nqyposs, ndbCposs, qystr1len, dbstr1len, dbxpad,
            tmpdpbotbuffer, btckdata, wrkmemtmibest, wrkmemaux);
    else
        RunTDintensiveDP<ANCHORRGN,BANDED,GAP0>(
            streamproc,
            gapcost, maxnsteps, nqystrs, ndbCstrs, 
            nqyposs, ndbCposs, qystr1len, dbstr1len, dbxpad,
            tmpdpdiagbuffers, tmpdpbotbuffer, btckdata, wrkmemtmibest, wrkmemaux);
}

// -------------------------------------------------------------------------
// RunTDintensiveDP: run thread block-intensive DP;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
//
template<bool ANCHORRGN, bool BANDED, bool GAP0>
void stage1::RunTDintensiveDP(
    cudaStream_t streamproc,
    const float gapcost,
    const uint maxnsteps,
    uint nqystrs, uint ndbCstrs,
    uint /*nqyposs*/, uint ndbCposs,
    uint qystr1len, uint dbstr1len,
    uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ tmpdpbotbuffer,
    char* __restrict__ btckdata,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux)
{
    //execution configuration for DP:
    //1D thread block processes 2D DP matrix oblique block of dimension 
    //CUDP_2DCACHE_DIM_D x CUDP_2DCACHE_DIM_X;
    //NOTE: using block diagonals, where blocks share a common point 
    //NOTE: (corner) with a neighbour in a diagonal;
    const uint maxblkdiagelems = GetMaxBlockDiagonalElems(
            dbstr1len, qystr1len, CUDP_2DCACHE_DIM_D, CUDP_2DCACHE_DIM_X);
    dim3 nthrds_dp(CUDP_2DCACHE_DIM_D,1,1);
    dim3 nblcks_dp(maxblkdiagelems,ndbCstrs,nqystrs);

    //number of regular DIAGONAL block diagonal series, each of given dimensions;
    //rect coords (x,y) are related to diagonal number d by d=x+y-1;
    //then, this number d is divided by the length of the block diagonal;
    //uint nblkdiags =
    // ((dbstr1len+qystr1len-1)+CUDP_2DCACHE_DIM_X-1)/CUDP_2DCACHE_DIM_X;
    //REVISION: due to the positioning of the first block, the first 
    // 1-position diagonal of the first diagonal block is out of bounds: remove -1
    uint nblkdiags = (uint)
        (((dbstr1len + qystr1len) + CUDP_2DCACHE_DIM_X-1) / CUDP_2DCACHE_DIM_X);
    //----
    //NOTE: now use block DIAGONALS, where blocks share a COMMON POINT 
    //NOTE: (corner, instead of edge) with a neighbour in a diagonal;
    //the number of series of such block diagonals equals 
    // #regular block diagonals (above) + {(l-1)/w}, 
    // where l is query length (y coord.), w is block width (dimension), and
    // {} denotes floor rounding; {(l-1)/w} is #interior divisions;
    nblkdiags += (uint)(qystr1len - 1) / CUDP_2DCACHE_DIM_D;

    //launch blocks along block diagonals to perform DP;
    //nblkdiags, total number of diagonals:
    for(uint d = 0; d < nblkdiags; d++)
    {
        ExecDPwBtck3264x<ANCHORRGN,BANDED,GAP0,D02IND_SEARCH>
            <<<nblcks_dp,nthrds_dp,0,streamproc>>>(
                d, ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber*/,
                gapcost,
                wrkmemtmibest, wrkmemaux,
                tmpdpdiagbuffers, tmpdpbotbuffer, btckdata);
        MYCUDACHECKLAST;
    }
}

// -------------------------------------------------------------------------
// RunTDintensiveDP: run complete DP, where a thread block performs 
// complete DP for a structure;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
//
template<bool ANCHORRGN, bool BANDED, bool GAP0>
void stage1::RunCompleteDP(
    cudaStream_t streamproc,
    const float gapcost,
    const uint maxnsteps,
    uint nqystrs, uint ndbCstrs,
    uint /*nqyposs*/, uint ndbCposs,
    uint qystr1len, uint dbstr1len,
    uint dbxpad,
    float* __restrict__ tmpdpbotbuffer,
    char* __restrict__ btckdata,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux)
{
    //execution configuration for DP:
    //1D thread block processes 2D DP matrix oblique block of dimensions 
    //CUDP_COMPLETE_2DCACHE_DIM_D^2;
    const uint maxlen = myhdmax(dbstr1len, qystr1len);
    const uint mincdim = CUDP_COMPLETE_2DCACHE_MINDIM_D;
    uint cdim = CUDP_COMPLETE_2DCACHE_DIM_D;
    for(; mincdim < cdim && maxlen < cdim; cdim >>= 1);
    dim3 nthrds_cdp(cdim,1,1);
    dim3 nblcks_cdp(ndbCstrs,nqystrs,1);

    const uint DPAD = cdim >> CUDP_COMPLETE_2DCACHE_MINDIM_D_LOG2;//padding
    const uint DDIMPAD = cdim + DPAD;//dimension + padding
    const uint DDIMPAD1 = DDIMPAD + 1;//1 extra for side edges (diagonals)

    //size of dynamically allocted smem for the cdp kernel:
    const uint szdsmem_cdp =
        sizeof(float) * (
            nTTranformMatrix +//tfm
            nTDPDiagScoreSubsections * DDIMPAD1 * 2 +//2 side edges (diagonals)
            nTDPDiagScoreSubsections * DDIMPAD +//bottom edge
            pmv2DNoElems * cdim * 2) +//reference coordinates
        sizeof(char) * cdim * (mincdim+1);//btck

    ExecCompleteDPwBtck512x<ANCHORRGN,false,GAP0,D02IND_SEARCH>
        <<<nblcks_cdp,nthrds_cdp,szdsmem_cdp,streamproc>>>(
            ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber*/,
            gapcost,
            wrkmemtmibest, wrkmemaux,  tmpdpbotbuffer, btckdata);
    MYCUDACHECKLAST;
}
