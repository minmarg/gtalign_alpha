/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __MpStageSSRR_h__
#define __MpStageSSRR_h__

#include "libutil/mybase.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmymp/mpproc/mpprocconfbase.h"
#include "libmymp/mpstages/transformbase.h"
#include "libmymp/mpstages/scoringbase.h"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/fragment.cuh"
#include "libmymp/mpstage1/MpStageBase.h"
#include "libmymp/mpstage1/MpStage1.h"
#include "libmymp/mpdp/MpDPHub.h"
#include "libmycu/cucom/cudef.h"

// -------------------------------------------------------------------------
// class MpStageSSRR for implementing structure comparison by matching 
// secondary structure and sequence similarity
//
class MpStageSSRR: public MpStage1 {
public:
    MpStageSSRR(
        const int maxndpiters,
        const uint maxnsteps,
        const uint minfraglen,
        const float prescore,
        const int stepinit,
        char** querypmbeg, char** querypmend,
        char** bdbCpmbeg, char** bdbCpmend,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint qystrnlen, uint dbstrnlen,
        uint dbxpad,
        float* scores, 
        float* tmpdpdiagbuffers, float* tmpdpbotbuffer,
        float* tmpdpalnpossbuffer, uint* maxscoordsbuf, char* btckdata,
        float* wrkmem, float* wrkmemccd, float* wrkmemtm, float* wrkmemtmibest,
        float* wrkmemaux, float* wrkmem2, float* alndatamem, float* tfmmem,
        uint* globvarsbuf)
    :
        MpStage1(
            maxndpiters, maxnsteps, minfraglen, prescore, stepinit,
            querypmbeg, querypmend, bdbCpmbeg, bdbCpmend,
            nqystrs, ndbCstrs, nqyposs, ndbCposs,
            qystr1len, dbstr1len, qystrnlen, dbstrnlen, dbxpad,
            scores,
            tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer,
            maxscoordsbuf, btckdata,
            wrkmem, wrkmemccd, wrkmemtm, wrkmemtmibest,
            wrkmemaux, wrkmem2, alndatamem, tfmmem,
            globvarsbuf
        )
    {}

    virtual void Run() {}

    template<bool USESEQSCORING>
    void RunSpecialized(const int maxndpiters)
    {
        //draw alignment based on ss and sequence similarity:
        Align<USESEQSCORING>();
        //refine alignment boundaries to improve scores
        RefineFragDPKernelCaller(false/*readlocalconv*/, FRAGREF_NMAXCONVIT);
        //refine alignment boundaries identified by applying DP;
        //1. With a gap cost:
        DPRefine<false/*GAP0*/,false/*PRESCREEN*/,false/*WRKMEMTM1*/>(2/*maxndpiters_*/, prescore_);
        //2. No gap cost:
        DPRefine<true/*GAP0*/,false/*PRESCREEN*/,false/*WRKMEMTM1*/>(maxndpiters, prescore_);
    }

protected:
    template<bool USESEQSCORING>
    void Align();
};



// -------------------------------------------------------------------------
// INLINES ...
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// Align: get alignment based on secondary structure information and
// sequence similarity;
// USESEQSCORING, flag of using sequence similarity scoring;
//
template<bool USESEQSCORING>
inline
void MpStageSSRR::Align()
{
    MYMSG("MpStageSSRR::Align", 5);
    // static std::string preamb = "MpStageSSRR::Align: ";

    constexpr float gcost = -1.0f;
    static const float wgt4ss = 1.0f;//weight for scoring ss
    static const float wgt4rr = 0.2f;//weight for pairwise residue scoring

    MpDPHub dphub(
        maxnsteps_,
        querypmbeg_, querypmend_, bdbCpmbeg_, bdbCpmend_,
        nqystrs_, ndbCstrs_, nqyposs_, ndbCposs_,  qystr1len_, dbstr1len_, dbxpad_,
        tmpdpdiagbuffers_, tmpdpbotbuffer_, tmpdpalnpossbuffer_, maxscoordsbuf_, btckdata_,
        wrkmem_, wrkmemccd_,  wrkmemtm_,  wrkmemtmibest_,
        wrkmemaux_, wrkmem2_, alndatamem_, tfmmem_, globvarsbuf_
    );

    dphub.ExecDPSSwBtck128xKernel<USESEQSCORING>(
        gcost, wgt4ss, wgt4rr,
        querypmbeg_, bdbCpmbeg_,
        wrkmemaux_, tmpdpdiagbuffers_, tmpdpbotbuffer_, btckdata_);

    dphub.BtckToMatched128xKernel<false/*ANCHORRGN*/,false/*BANDED*/>(
        0/*stepnumber*/,//slot 0
        querypmbeg_, bdbCpmbeg_, btckdata_, wrkmemaux_, tmpdpalnpossbuffer_);
}


// -------------------------------------------------------------------------

#endif//__MpStageSSRR_h__
