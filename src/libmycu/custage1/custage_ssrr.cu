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
#include "libmycu/cudp/btck2match.cuh"
#include "libmycu/custage1/custage1.cuh"
#include "custage_ssrr.cuh"

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// run_stage_ssrr: search for superposition between multiple molecules
// simultaneoulsy and identify alignments by DP using secondary 
// structure information and sequence similarity criteria;
// USESEQSCORING, template parameter, flag of using sequence similarity scoring;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
template<bool USESEQSCORING>
void stage_ssrr::run_stage_ssrr(
    std::map<CGKey,MyCuGraph>& stgraphs,
    cudaStream_t streamproc,
    const float /*scorethld*/,
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
    //alignment based on ss information and sequence similarity:
    stage_ssrr_align<USESEQSCORING>(
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
        wrkmemaux);

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
}

// Instantiations
// 
#define INSTANTIATE_stage_ssrr__run_stage_ssrr(USESEQSCORING) \
    template void stage_ssrr::run_stage_ssrr<USESEQSCORING>( \
    std::map<CGKey,MyCuGraph>& stgraphs, \
    cudaStream_t streamproc, \
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

INSTANTIATE_stage_ssrr__run_stage_ssrr(false);
INSTANTIATE_stage_ssrr__run_stage_ssrr(true);



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// stage_ssrr_align: get alignment based on secondary structure information
// and sequence similarity;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
template<bool USESEQSCORING>
void stage_ssrr::stage_ssrr_align(
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
    float* __restrict__ wrkmemaux)
{
    constexpr float gcost = -1.0f;
    static const float wgt4ss = 1.0f;//weight for scoring ss
    static const float wgt4rr = 0.2f;//weight for pairwise residue scoring

    //execution configuration for DP:
    //1D thread block processes 2D DP matrix oblique block of dimension 
    //CUDP_2DCACHE_DIM_D x CUDP_2DCACHE_DIM_X;
    //NOTE: using block diagonals, where blocks share a common point 
    //NOTE: (corner) with a neighbour in a diagonal;
    const uint maxblkdiagelems = GetMaxBlockDiagonalElems(
            dbstr1len, qystr1len, CUDP_2DCACHE_DIM_D, CUDP_2DCACHE_DIM_X);
    dim3 nthrds_dp(CUDP_2DCACHE_DIM_D,1,1);
    dim3 nblcks_dp(maxblkdiagelems,ndbCstrs,nqystrs);
    uint nblkdiags = (uint)
        (((dbstr1len + qystr1len) + CUDP_2DCACHE_DIM_X-1) / CUDP_2DCACHE_DIM_X);
    nblkdiags += (uint)(qystr1len - 1) / CUDP_2DCACHE_DIM_D;

    //execution configuration for extracting matched positions
    //identified during DP:
    dim3 nthrds_mtch(CUDP_MATCHED_DIM_X,CUDP_MATCHED_DIM_Y,1);
    dim3 nblcks_mtch(ndbCstrs,nqystrs,1);

    //launch blocks along block diagonals to perform DP;
    //nblkdiags, total number of diagonals:
    for(uint d = 0; d < nblkdiags; d++)
    {
        ExecDPSSwBtck3264x<USESEQSCORING>
            <<<nblcks_dp,nthrds_dp,0,streamproc>>>(
                d, ndbCstrs, ndbCposs, dbxpad, maxnsteps,
                wgt4ss, wgt4rr, gcost,
                wrkmemaux, tmpdpdiagbuffers, tmpdpbotbuffer, btckdata);
        MYCUDACHECKLAST;
    }

    //process the result of DP
    BtckToMatched32x<false/*ANCHORRGN*/,false/*BANDED*/>
        <<<nblcks_mtch,nthrds_mtch,0,streamproc>>>(
            ndbCstrs, ndbCposs, dbxpad, maxnsteps, 0/*stepnumber*/,
            btckdata, wrkmemaux, tmpdpalnpossbuffer);
    MYCUDACHECKLAST;
}
