/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __MpStage2_h__
#define __MpStage2_h__

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
// class MpStage2 for implementing structure comparison at stage 2
//
class MpStage2: public MpStage1 {
public:
    MpStage2(
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

    template<bool GAP0, bool USESS, int D02IND>
    void RunSpecialized(
        const int maxndpiters, const bool check_for_low_scores, const float scorethld)
    {
        //draw alignment using ss information and best superposition:
        Align<GAP0, USESS, D02IND>();
        //refine alignment boundaries to improve scores
        RefineFragDPKernelCaller(false/*readlocalconv*/, FRAGREF_NMAXCONVIT);
        //refine alignment boundaries identified by applying DP;
        //1. With a gap cost:
        DPRefine<false/*GAP0*/,false/*PRESCREEN*/,false/*WRKMEMTM1*/>(2/*maxndpiters_*/, prescore_);
        //2. No gap cost:
        DPRefine<true/*GAP0*/,false/*PRESCREEN*/,false/*WRKMEMTM1*/>(maxndpiters, prescore_);
        //execution for checking scores:
        if(check_for_low_scores && 0.0f < scorethld)
            SetLowScoreConvergenceFlagKernel(
                scorethld,  querypmbeg_, bdbCpmbeg_, wrkmemaux_);
    }

protected:
    template<bool GAP0, bool USESS, int D02IND>
    void Align();
};



// -------------------------------------------------------------------------
// INLINES ...
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// Align: find alignment based on the secondary structure information and
// best superposition so far;
// GAP0, template parameter, flag of gap open cost 0;
// USESS, template parameter, flag of using secondary structure scoring;
// D02IND, template parameter, index of how the d0 distance threshold has to be computed;
//
template<bool GAP0, bool USESS, int D02IND>
inline
void MpStage2::Align()
{
    MYMSG("MpStage2::Align", 5);
    // static std::string preamb = "MpStage2::Align: ";

    constexpr float gcost = {GAP0? 0.0f: -1.0f};
    static const float sswgt = 0.5f;

    MpDPHub dphub(
        maxnsteps_,
        querypmbeg_, querypmend_, bdbCpmbeg_, bdbCpmend_,
        nqystrs_, ndbCstrs_, nqyposs_, ndbCposs_,  qystr1len_, dbstr1len_, dbxpad_,
        tmpdpdiagbuffers_, tmpdpbotbuffer_, tmpdpalnpossbuffer_, maxscoordsbuf_, btckdata_,
        wrkmem_, wrkmemccd_,  wrkmemtm_,  wrkmemtmibest_,
        wrkmemaux_, wrkmem2_, alndatamem_, tfmmem_, globvarsbuf_
    );

    dphub.ExecDPTFMSSwBtck128xKernel<true/* GLOBTFM */, GAP0, USESS, D02IND>(
        gcost, sswgt, 0/*stepnumber(unused)*/,
        querypmbeg_, bdbCpmbeg_,
        tfmmem_, wrkmemaux_, tmpdpdiagbuffers_, tmpdpbotbuffer_, btckdata_);

    dphub.BtckToMatched128xKernel<false/*ANCHORRGN*/,false/*BANDED*/>(
        0/*stepnumber*/,//slot 0
        querypmbeg_, bdbCpmbeg_, btckdata_, wrkmemaux_, tmpdpalnpossbuffer_);
}


// -------------------------------------------------------------------------

#endif//__MpStage2_h__
