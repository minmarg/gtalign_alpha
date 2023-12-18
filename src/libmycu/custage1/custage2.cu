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
#include "libmycu/cucom/cugraphs.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"

#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/scoring.cuh"
#include "libmycu/cudp/dpw_btck.cuh"
#include "libmycu/cudp/dpssw_btck.cuh"
#include "libmycu/cudp/dpssw_tfm_btck.cuh"
#include "libmycu/cudp/btck2match.cuh"
#include "libmycu/custage1/custage1.cuh"
#include "custage2.cuh"

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// run_stage2: stage-2 search for superposition between multiple 
// molecules simultaneoulsy and identify alignments by DP using secondary 
// structure information;
// GAP0, template parameter, flag of gap open cost 0;
// USESS, template parameter, flag of using secondary structure scoring;
// D02IND, template parameter, index of how the d0 distance threshold has to be computed;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
template<bool GAP0, bool USESS, int D02IND>
void stage2::run_stage2(
    std::map<CGKey,MyCuGraph>& stgraphs,
    cudaStream_t streamproc,
    const bool check_for_low_scores,
    const float scorethld,
    const float prescore,
    const int maxndpiters,
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
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemccd,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem2,
    float* __restrict__ tfmmem,
    uint* __restrict__ /*globvarsbuf*/)
{
    //draw alignment using ss information and best superposition:
    stage2_dpss_align<GAP0,USESS,D02IND>(
        streamproc,
        maxnsteps,
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
        wrkmemaux,
        tfmmem);

    //refine alignment boundaries to improve scores
    stage1::stage1_refinefrag<false/* CONDITIONAL */>(
        stgraphs,
        stg1REFINE_INITIAL_DP/*fragments identified by DP*/,
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
    //substage by applying DP;
    //1. With a gap cost:
    stage1_dprefine<false/*GAP0*/,false/*PRESCREEN*/>(
        stgraphs,
        streamproc,
        2/* maxndpiters */,
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

    //execution configuration for checking scores:
    //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
    dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
    dim3 nblcks_scinit(
        (ndbCstrs + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        nqystrs, 1);

    if(check_for_low_scores && 0.0f < scorethld) {
        SetLowScoreConvergenceFlag<<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
            scorethld, ndbCstrs, maxnsteps, wrkmemaux);
    }
}

// Instantiations
// 
#define INSTANTIATE_stage2__run_stage2(tpGAP0,tpUSESS,tpD02IND) \
    template void stage2::run_stage2<tpGAP0,tpUSESS,tpD02IND>( \
    std::map<CGKey,MyCuGraph>& stgraphs, \
    cudaStream_t streamproc, \
    const bool check_for_low_scores, \
    const float scorethld, \
    const float prescore, \
    const int maxndpiters, \
    const uint maxnsteps, \
    const uint minfraglen, \
    uint nqystrs, uint ndbCstrs, \
    uint nqyposs, uint ndbCposs, \
    uint qystr1len, uint dbstr1len, \
    uint qystrnlen, uint dbstrnlen, \
    uint dbxpad, \
    float* __restrict__ /*scores*/,  \
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
    float* __restrict__ tfmmem, \
    uint* __restrict__ /*globvarsbuf*/);

INSTANTIATE_stage2__run_stage2(false,true,D02IND_DPSCAN);
INSTANTIATE_stage2__run_stage2(true,false,D02IND_SEARCH);



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stage2_dpss_align: obtain alignment based on the secondary structure 
// information and best superposition so  far refine;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
template<bool GAP0, bool USESS, int D02IND>
void stage2::stage2_dpss_align(
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
    uint* __restrict__ /*maxscoordsbuf*/,
    char* __restrict__ btckdata,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tfmmem)
{
    constexpr float gcost = {GAP0? 0.0f: -1.0f};
    static const float sswgt = 0.5f;

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

    //execution configuration for extracting matched positions
    //identified during DP:
    dim3 nthrds_mtch(CUDP_MATCHED_DIM_X,CUDP_MATCHED_DIM_Y,1);
    dim3 nblcks_mtch(ndbCstrs,nqystrs,1);


    //launch blocks along block diagonals to perform DP;
    //nblkdiags, total number of diagonals:
    for(uint d = 0; d < nblkdiags; d++)
    {
        ExecDPTFMSSwBtck3264x<true/* GLOBTFM */,GAP0,USESS,D02IND>
            <<<nblcks_dp,nthrds_dp,0,streamproc>>>(
                d, ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber(unused)*/,
                sswgt, gcost,
                tfmmem, wrkmemaux, tmpdpdiagbuffers, tmpdpbotbuffer, btckdata);
        MYCUDACHECKLAST;
    }

    //process the result of DP
    BtckToMatched32x<false/*ANCHORRGN*/,false/*BANDED*/>
        <<<nblcks_mtch,nthrds_mtch,0,streamproc>>>(
            ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber*/,
            btckdata, wrkmemaux, tmpdpalnpossbuffer);
    MYCUDACHECKLAST;
}
