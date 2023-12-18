/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __MpStageFin_h__
#define __MpStageFin_h__

#include "libutil/mybase.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmymp/mpproc/mpprocconfbase.h"
#include "libmymp/mpstages/transformbase.h"
#include "libmymp/mpstages/scoringbase.h"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/fragment.cuh"
#include "libmymp/mpstage1/MpStageBase.h"
#include "libmymp/mpdp/MpDPHub.h"
#include "libmycu/cucom/cudef.h"

// -------------------------------------------------------------------------
// class MpStageFin for implementing final alignment refinement for output
//
class MpStageFin: public MpStageBase {
public:
    MpStageFin(
        const float d2equiv,
        const uint maxnsteps,
        const uint minfraglen,
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
        float* wrkmemaux, float* wrkmem2, float* alndatamem, float* tfmmem, char* alnsmem,
        uint* globvarsbuf)
    :
        MpStageBase(
            maxnsteps, minfraglen,
            querypmbeg, querypmend, bdbCpmbeg, bdbCpmend,
            nqystrs, ndbCstrs, nqyposs, ndbCposs,
            qystr1len, dbstr1len, qystrnlen, dbstrnlen, dbxpad,
            scores,
            tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer,
            maxscoordsbuf, btckdata,
            wrkmem, wrkmemccd, wrkmemtm, wrkmemtmibest,
            wrkmemaux, wrkmem2, alndatamem, tfmmem,
            globvarsbuf
        ),
        dphub_(
            maxnsteps_,
            querypmbeg_, querypmend_, bdbCpmbeg_, bdbCpmend_,
            nqystrs_, ndbCstrs_, nqyposs_, ndbCposs_,  qystr1len_, dbstr1len_, dbxpad_,
            tmpdpdiagbuffers_, tmpdpbotbuffer_, tmpdpalnpossbuffer_, maxscoordsbuf_, btckdata_,
            wrkmem_, wrkmemccd_,  wrkmemtm_,  wrkmemtmibest_,
            wrkmemaux_, wrkmem2_, alndatamem, tfmmem_, globvarsbuf_
        ),
        d2equiv_(d2equiv),
        alnsmem_(alnsmem)
    {}


    virtual void Run() {
        MYMSG("MpStageFin::Run", 4);
        static const bool nodeletions = CLOptions::GetO_NO_DELETIONS();
        static const int referenced = CLOptions::GetO_REFERENCED();
        //produce alignment to refine superposition on
        Align(false/* constrainedbtck */);
        //refine superposition at the finest scale
        Refine();
        //produce final alignment of matched positions
        Align(true/* constrainedbtck */);
        //produce full alignments for output
        dphub_.ProductionMatchToAlignment128xKernel(
            nodeletions, d2equiv_,
            querypmbeg_, bdbCpmbeg_,
            tmpdpalnpossbuffer_, wrkmemaux_, alndatamem_, alnsmem_);
        //refine using production thresholds and calculate final scores for output
        ProduceOutputScores();
        //revert transformation matrices if needed
        if(referenced) RevertTfmMatricesKernel(tfmmem_);
    }


protected:
    void Align(const bool constrainedbtck);
    void Refine();
    void ProduceOutputScores();


protected:
    template<bool D0FINAL, int CHCKDST, bool TFM_DINV>
    void FinalFragmentBasedDPAlignmentRefinementPhase1Kernel(
        const int nmaxconvit,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const float* const __RESTRICT__ tmpdpalnpossbuffer,
        float* const __RESTRICT__ tmpdpdiagbuffers,
        float* const __RESTRICT__ wrkmemtmibest,
        float* const __RESTRICT__ wrkmemaux,
        float* const __RESTRICT__ tfmmem);

    template<bool D0FINAL, int CHCKDST, bool TFM_DINV>
    void FinalFragmentBasedDPAlignmentRefinementPhase2Kernel(
        const int nmaxconvit,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const float* const __RESTRICT__ tmpdpalnpossbuffer,
        float* const __RESTRICT__ tmpdpdiagbuffers,
        float* const __RESTRICT__ wrkmemtmibest,
        float* const __RESTRICT__ wrkmemaux,
        float* const __RESTRICT__ tfmmem);

    template<bool D0FINAL, int CHCKDST, bool TFM_DINV>
    void FinalFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel(
        const int nmaxconvit,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const float* const __RESTRICT__ tmpdpalnpossbuffer,
        float* const __RESTRICT__ tmpdpdiagbuffers,
        float* const __RESTRICT__ wrkmemtmibest,
        float* const __RESTRICT__ wrkmemaux,
        float* const __RESTRICT__ tfmmem);


protected:
    template<bool TFM_DINV>
    void ProductionFragmentBasedDPAlignmentRefinementPhase1Kernel(
        const int nmaxconvit,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const float* const __RESTRICT__ tmpdpalnpossbuffer,
        float* const __RESTRICT__ tmpdpdiagbuffers,
        float* const __RESTRICT__ wrkmemtmibest,
        float* const __RESTRICT__ wrkmemaux,
        float* const __RESTRICT__ alndatamem,
        float* const __RESTRICT__ tfmmem);

    template<bool TFM_DINV>
    void ProductionFragmentBasedDPAlignmentRefinementPhase2Kernel(
        const int nmaxconvit,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const float* const __RESTRICT__ tmpdpalnpossbuffer,
        float* const __RESTRICT__ tmpdpdiagbuffers,
        float* const __RESTRICT__ wrkmemtmibest,
        float* const __RESTRICT__ wrkmemaux,
        float* const __RESTRICT__ alndatamem,
        float* const __RESTRICT__ tfmmem);

    template<bool TFM_DINV>
    void ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel(
        const int nmaxconvit,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const float* const __RESTRICT__ tmpdpalnpossbuffer,
        float* const __RESTRICT__ tmpdpdiagbuffers,
        float* const __RESTRICT__ wrkmemtmibest,
        float* const __RESTRICT__ wrkmemaux,
        float* const __RESTRICT__ alndatamem,
        float* const __RESTRICT__ tfmmem);

    template<bool TFM_DINV>
    void ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearchKernel(
        const int nmaxconvit,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const float* const __RESTRICT__ tmpdpalnpossbuffer,
        float* const __RESTRICT__ tmpdpdiagbuffers,
        float* const __RESTRICT__ wrkmemtmibest,
        float* const __RESTRICT__ wrkmemaux,
        float* const __RESTRICT__ alndatamem,
        float* const __RESTRICT__ tfmmem);


protected:
    void RevertTfmMatricesKernel(
        float* const __RESTRICT__ tfmmem);


protected:
    float CalcRMSD_Complete(
        float* __RESTRICT__ ccm, float* __RESTRICT__ rr);

    template<int SMIDIM, int XDIM, int DATALN>
    void CalcExtCCMatrices_DPRefined_Complete(
        const int qryndx,
        const int ndbCposs,
        const int dbxpad,
        const int maxnsteps,
        const int /*sfragfctxndx*/,
        const int dbstrdst,
        const int fraglen,
        int qrylen, int dbstrlen,
        int qrypos, int rfnpos,
        const float* const __RESTRICT__ tmpdpalnpossbuffer,
        float (* __RESTRICT__ ccm)[XDIM]);

    template<bool WRITEFRAGINFO, bool CONDITIONAL>
    void SaveBestQRScoresAndTM_Complete(
        const float best,
        const float gbest,
        const int qryndx,
        const int dbstrndx,
        const int ndbCstrs,
        const int maxnsteps,
        const int sfragfctxndx,
        const int sfragndx,
        const int sfragpos,
        const float* __RESTRICT__ tfm,
        float* __RESTRICT__ wrkmemtmibest,
        float* __RESTRICT__ wrkmemaux);

    void SaveBestQRScoresAndTM_Phase2_logsearch_Complete(
        float best,
        float gbest,
        const int qryndx,
        const int dbstrndx,
        const int ndbCstrs,
        const int qrylenorg,
        const int dbstrlenorg,
        const float* __RESTRICT__ tfm,
        float* const __RESTRICT__ tfmmem,
        float* const __RESTRICT__ alndatamem);


    template<int XDIM, int DATALN, bool WRITEFRAGINFO, bool CONDITIONAL>
    void ProductionSaveBestScoresAndTMAmongBests(
        const int qryndx,
        const int rfnblkndx,
        const int ndbCstrs,
        const int maxnsteps,
        const int effnsteps,
        float (* __RESTRICT__ ccm)[XDIM],
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const float* const __RESTRICT__ wrkmemtmibest,
        float* const __RESTRICT__ wrkmemaux,
        float* const __RESTRICT__ alndatamem,
        float* const __RESTRICT__ tfmmem);


protected:
    template<int XDIM, int DATALN>
    void UpdateExtCCMOneAlnPos_DPRefined(
        int pos, const int dblen,
        const float* const __RESTRICT__ tmpdpalnpossbuffer,
        float (* __RESTRICT__ ccm)[XDIM], int pi);


private:
    template<int nEFFDS, int XDIM, int DATALN, bool TFM_DINV>
    void ProductionRefinementPhase2InnerLoop(
        const int sfragfctxndx, const int qryndx, const int nmaxconvit,
        const int ndbCposs, const int dbxpad, const int maxnsteps,
        const int qrylenorg, const int dbstrlenorg, const int qrylen, const int dbstrlen,
        const int dbstrdst, const int qrypos, const int rfnpos, const int sfragpos, const int fraglen,
        const float d0, const float d02, const float d82, float& best,
        float (* __RESTRICT__ ccm)[XDIM],
        float* __RESTRICT__ ccmLast,
        float* __RESTRICT__ tfm,
        float* __RESTRICT__ tfmBest,
        const float* const __RESTRICT__ tmpdpalnpossbuffer,
        float* const __RESTRICT__ tmpdpdiagbuffers);


protected:
    MpDPHub dphub_;
    const float d2equiv_;
    char* const alnsmem_;
};



// -------------------------------------------------------------------------
// INLINES ...
// -------------------------------------------------------------------------
// Align: find alignment based on the best superposition obtained;
// constrainedbtck, flag of using constrained backtracking;
//
inline
void MpStageFin::Align(const bool constrainedbtck)
{
    MYMSG("MpStageFin::Align", 5);
    // static std::string preamb = "MpStageFin::Align: ";

    dphub_.ExecDPTFMSSwBtck128xKernel<true/* GLOBTFM */,true/* GAP0 */,false/* USESS */,D02IND_SEARCH>(
        0.0f/* gcost */, 0.0f/* sswgt */, 0/*stepnumber(unused)*/,
        querypmbeg_, bdbCpmbeg_,
        tfmmem_, wrkmemaux_, tmpdpdiagbuffers_, tmpdpbotbuffer_, btckdata_);

    //produce alignment for superposition
    if(constrainedbtck)
        dphub_.ConstrainedBtckToMatched128xKernel(
            querypmbeg_, bdbCpmbeg_, btckdata_, tfmmem_, wrkmemaux_, tmpdpalnpossbuffer_);
    else
        dphub_.BtckToMatched128xKernel<false/*ANCHORRGN*/,false/*BANDED*/>(
            0/*stepnumber*/,//slot 0
            querypmbeg_, bdbCpmbeg_, btckdata_, wrkmemaux_, tmpdpalnpossbuffer_);
}

// -------------------------------------------------------------------------
// Refine: refine for final superposition-best alignment; 
//
inline
void MpStageFin::Refine()
{
    MYMSG("MpStageFin::Refine", 5);
    // static std::string preamb = "MpStageFin::Refine: ";

    const int symmetric = CLOptions::GetC_SYMMETRIC();
    const int refinement = CLOptions::GetC_REFINEMENT();
    const int depth = CLOptions::GetC_DEPTH();

    //refine alignment boundaries to improve scores
    if(symmetric)
        FinalFragmentBasedDPAlignmentRefinementPhase1Kernel
            <false/* D0FINAL */,CHCKDST_CHECK,true/*TFM_DINV*/>(
                FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                querypmbeg_, bdbCpmbeg_,
                tmpdpalnpossbuffer_, tmpdpdiagbuffers_, wrkmemtmibest_, wrkmemaux_, tfmmem_);
    else
        FinalFragmentBasedDPAlignmentRefinementPhase1Kernel
            <false/* D0FINAL */,CHCKDST_CHECK,false/*TFM_DINV*/>(
                FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                querypmbeg_, bdbCpmbeg_,
                tmpdpalnpossbuffer_, tmpdpdiagbuffers_, wrkmemtmibest_, wrkmemaux_, tfmmem_);

    if(depth == CLOptions::csdShallow || refinement == CLOptions::csrCoarseSearch)
        return;

    if(refinement == CLOptions::csrFullASearch) {
        if(symmetric)
            FinalFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel
                <false/* D0FINAL */,CHCKDST_CHECK,true/*TFM_DINV*/>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    querypmbeg_, bdbCpmbeg_,
                    tmpdpalnpossbuffer_, tmpdpdiagbuffers_, wrkmemtmibest_, wrkmemaux_, tfmmem_);
        else
            FinalFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel
                <false/* D0FINAL */,CHCKDST_CHECK,false/*TFM_DINV*/>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    querypmbeg_, bdbCpmbeg_,
                    tmpdpalnpossbuffer_, tmpdpdiagbuffers_, wrkmemtmibest_, wrkmemaux_, tfmmem_);
    }
    else {
        if(symmetric)
            FinalFragmentBasedDPAlignmentRefinementPhase2Kernel
                <false/* D0FINAL */,CHCKDST_CHECK,true/*TFM_DINV*/>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    querypmbeg_, bdbCpmbeg_,
                    tmpdpalnpossbuffer_, tmpdpdiagbuffers_, wrkmemtmibest_, wrkmemaux_, tfmmem_);
        else
            FinalFragmentBasedDPAlignmentRefinementPhase2Kernel
                <false/* D0FINAL */,CHCKDST_CHECK,false/*TFM_DINV*/>(
                    FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                    querypmbeg_, bdbCpmbeg_,
                    tmpdpalnpossbuffer_, tmpdpdiagbuffers_, wrkmemtmibest_, wrkmemaux_, tfmmem_);
    }
}

// -------------------------------------------------------------------------
// ProduceOutputScores: refine using production thresholds and 
// calculate final scores for output; complete version;
//
inline
void MpStageFin::ProduceOutputScores()
{
    MYMSG("MpStageFin::ProduceOutputScores", 5);
    // static std::string preamb = "MpStageFin::ProduceOutputScores: ";

    const int symmetric = CLOptions::GetC_SYMMETRIC();
    const int refinement = CLOptions::GetC_REFINEMENT();

    //refine alignment boundaries to improve scores
    if(symmetric)
        ProductionFragmentBasedDPAlignmentRefinementPhase1Kernel<true/*TFM_DINV*/>(
            FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
            querypmbeg_, bdbCpmbeg_,
            tmpdpalnpossbuffer_, tmpdpdiagbuffers_, wrkmemtmibest_, wrkmemaux_,
            alndatamem_, tfmmem_);
    else
        ProductionFragmentBasedDPAlignmentRefinementPhase1Kernel<false/*TFM_DINV*/>(
            FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
            querypmbeg_, bdbCpmbeg_,
            tmpdpalnpossbuffer_, tmpdpdiagbuffers_, wrkmemtmibest_, wrkmemaux_,
            alndatamem_, tfmmem_);


    if(refinement == CLOptions::csrLogSearch) {
        if(symmetric)
            ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearchKernel<true/*TFM_DINV*/>(
                FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                querypmbeg_, bdbCpmbeg_,
                tmpdpalnpossbuffer_, tmpdpdiagbuffers_, wrkmemtmibest_, wrkmemaux_,
                alndatamem_, tfmmem_);
        else
            ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearchKernel<false/*TFM_DINV*/>(
                FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                querypmbeg_, bdbCpmbeg_,
                tmpdpalnpossbuffer_, tmpdpdiagbuffers_, wrkmemtmibest_, wrkmemaux_,
                alndatamem_, tfmmem_);
    }
    else if(refinement == CLOptions::csrOneSearch || refinement == CLOptions::csrCoarseSearch) {
        if(symmetric)
            ProductionFragmentBasedDPAlignmentRefinementPhase2Kernel<true/*TFM_DINV*/>(
                FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                querypmbeg_, bdbCpmbeg_,
                tmpdpalnpossbuffer_, tmpdpdiagbuffers_, wrkmemtmibest_, wrkmemaux_,
                alndatamem_, tfmmem_);
        else
            ProductionFragmentBasedDPAlignmentRefinementPhase2Kernel<false/*TFM_DINV*/>(
                FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                querypmbeg_, bdbCpmbeg_,
                tmpdpalnpossbuffer_, tmpdpdiagbuffers_, wrkmemtmibest_, wrkmemaux_,
                alndatamem_, tfmmem_);
    }
    else if(refinement >= CLOptions::csrFullSearch) {
        if(symmetric)
            ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel<true/*TFM_DINV*/>(
                FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                querypmbeg_, bdbCpmbeg_,
                tmpdpalnpossbuffer_, tmpdpdiagbuffers_, wrkmemtmibest_, wrkmemaux_,
                alndatamem_, tfmmem_);
        else
            ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel<false/*TFM_DINV*/>(
                FRAGREF_NMAXCONVIT/*#iterations until convergence*/,
                querypmbeg_, bdbCpmbeg_,
                tmpdpalnpossbuffer_, tmpdpdiagbuffers_, wrkmemtmibest_, wrkmemaux_,
                alndatamem_, tfmmem_);
    }
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// =========================================================================
// CalcRMSD_Complete: calculate RMSD;
// ccm, cache for the cross-covariance matrix and reuse;
// rr, temporary array [6] for a triangle of the R matrix;
// Based on the original Kabsch algorithm (see CalcTfmMatrices_Complete);
//
inline
float MpStageFin::CalcRMSD_Complete(
    float* __RESTRICT__ ccm, float* __RESTRICT__ rr)
{
    //#positions used to calculate cross-covarinaces:
    float nalnposs = ccm[twmvNalnposs];

    if(nalnposs <= 0.0f) return 0.0f;

    //calculate query center vector in advance
    #pragma omp simd
    for(int pi = twmvCVq_0; pi <= twmvCVq_2; pi++) ccm[pi] /= nalnposs;
    #pragma omp simd
    for(int pi = twmvCV2q_0; pi <= twmvCV2q_2; pi++) ccm[pi] /= nalnposs;

    CalcRmatrix(ccm);

    //calculate reference center vector now
    #pragma omp simd
    for(int pi = twmvCVr_0; pi <= twmvCVr_2; pi++) ccm[pi] /= nalnposs;
    #pragma omp simd
    for(int pi = twmvCV2r_0; pi <= twmvCV2r_2; pi++) ccm[pi] /= nalnposs;

    //NOTE: scale correlation matrix to enable rotation matrix 
    // calculation in single precision without overflow and underflow:
    //ScaleRmatrix(ccmCache);
    float scale = GetRScale(ccm);
    #pragma omp simd
    for(int pi = 0; pi < twmvEndOfCCMtx; pi++) ccm[pi] /= scale;

    //calculate determinant
    float det = CalcDet(ccm);

    //calculate the product transposed(R) * R
    CalcRTR(ccm, rr);

    float E_0;//E_0 in Kabsch's 1978 paper
    E_0 = //two variances
        ccm[twmvCV2q_0] - SQRD(ccm[twmvCVq_0]) +
        ccm[twmvCV2q_1] - SQRD(ccm[twmvCVq_1]) +
        ccm[twmvCV2q_2] - SQRD(ccm[twmvCVq_2]) +
        ccm[twmvCV2r_0] - SQRD(ccm[twmvCVr_0]) +
        ccm[twmvCV2r_1] - SQRD(ccm[twmvCVr_1]) +
        ccm[twmvCV2r_2] - SQRD(ccm[twmvCVr_2]);

    //Kabsch:
    //eigenvalues: form characteristic cubic x**3-3*spur*x**2+3*cof*x-det=0
    float spur = (rr[0] + rr[2] + rr[5]) * oneTHIRDf;
    float cof = (((((rr[2] * rr[5] - SQRD(rr[4])) + rr[0] * rr[5]) -
                SQRD(rr[3])) + rr[0] * rr[2]) -
                SQRD(rr[1])) * oneTHIRDf;

    bool abok = (spur > 0.0f);

    float e0, e1, e2;//polynomial roots (eigenvalues)
    e0 = e1 = e2 = spur;

    if(abok)
    {   //Kabsch:
        //reduce cubic to standard form y**3-3hy+2g=0 by putting x=y+spur;
        //Kabsch: solve cubic: roots are e[0],e[1],e[2] in decreasing order
        //Kabsch: handle special case of 3 identical roots
        SolveCubic(det, spur, cof, e0, e1, e2);
    }

    e0 = (e0 <= 0.0f)? 0.0f: sqrtf(e0);
    e1 = (e1 <= 0.0f)? 0.0f: sqrtf(e1);
    e2 = (e2 <= 0.0f)? 0.0f: (sqrtf(e2) * ((det < 0.0f)? -1.0f: 1.0f));

    //write rmsd to E_0:
    //NOTE: scale the eigenvalues to get values for the unscaled RTR;
    E_0 -= (2.0f * scale * (e0 + e1 + e2)) / nalnposs;
    E_0 = (E_0 <= 0.0f)? 0.0f: sqrtf(E_0);

    return E_0;
}

// =========================================================================



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// CalcExtCCMatrices_DPRefined_Complete: calculate cross-covariance matrix 
// between the query and reference structures for refinement, i.e. 
// delineation of suboptimal fragment boundaries;
// This complete version complements CalcCCMatrices_DPRefined_Complete by
// additionally calculating the sum of squares required for RMSD computation;
// NOTE: process alignment fragment (1D) along structure positions;
// qryndx, query serial number;
// ndbCposs, total number of reference positions in the chunk;
// dbxpad, #pad positions along the dimension of reference structures;
// maxnsteps, max number of steps to perform for each reference structure;
// sfragfctxndx, current fragment factor x fragment length index, which is <maxnsteps;
// dbstrdst, distances in positions to the beginnings of the reference structures;
// fraglen, fragment length;
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// NOTE: memory pointers should be aligned!
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// ccm, cache for the cross-covarinace matrix and related data;
// 
template<int SMIDIM, int XDIM, int DATALN>
inline
void MpStageFin::CalcExtCCMatrices_DPRefined_Complete(
    const int qryndx,
    const int ndbCposs,
    const int dbxpad,
    const int maxnsteps,
    const int /*sfragfctxndx*/,
    const int dbstrdst,
    const int fraglen,
    int qrylen, int dbstrlen,
    int qrypos, int rfnpos,
    const float* const __RESTRICT__ tmpdpalnpossbuffer,
    float (* __RESTRICT__ ccm)[XDIM])
{
    //initialize cache:
    #pragma omp simd collapse(2)
    for(int f = 0; f < SMIDIM; f++)
        for(int pi = 0; pi < XDIM; pi++) ccm[f][pi] = 0.0f;

    //qrylen == dbstrlen; reuse qrylen for original alignment length;
    //update positions and assign virtual query and reference lengths:
    UpdateLengths(dbstrlen/*qrylen*/, dbstrlen, qrypos, rfnpos, fraglen);

    const int dblen = ndbCposs + dbxpad;
    //offset to the beginning of the data along the y axis wrt query qryndx: 
    const int yofff = (qryndx * maxnsteps + 0/*sfragfctxndx*/) * dblen * nTDPAlignedPoss;

    //manually unroll along data blocks:
    for(int ai = 0; ai < fraglen; ai += XDIM)
    {
        //starting position in tmpdpalnpossbuffer for a pair:
        //NOTE: qrypos == rfnpos as well as lengths: use qrylen as the 
        //NOTE: original alignment length here;
        //NOTE: alignment written in reverse order:
        int piend = mymin(XDIM, fraglen - ai);
        #pragma omp simd
        for(int pi = 0; pi < piend; pi++)
            UpdateExtCCMOneAlnPos_DPRefined<XDIM,DATALN>(
                // yofff + dbstrdst + qrylen-1 - (rfnpos + ai + pi), dblen,
                //NOTE: to make address increment is 1:
                yofff + dbstrdst + qrylen-1 - (rfnpos + ai + piend-1) + pi, dblen,
                tmpdpalnpossbuffer,
                ccm, pi);
    }

    //sum reduction for each field
    for(int f = 0; f < SMIDIM; f++) {
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for(int pi = 0; pi < XDIM; pi++) sum += ccm[f][pi];
        //write sum back to ccm
        ccm[f][0] = sum;
    }

    //write nalnposs
    ccm[twmvNalnposs][0] = fraglen;
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// UpdateExtCCMOneAlnPos_DPRefined: extension to UpdateCCMOneAlnPos_DPRefined;
// update one position of the alignment obtained by DP, contributing to the
// cross-covariance matrix between the query and reference structures;
// coordinate squares are updated too;
// XDIM, template parameter: inner-most dimensions of the cache matrix;
// pos, position index to read alignment coordinates;
//
#if defined(OS_MS_WINDOWS)
#define OMPDECLARE_MpStageFin_UpdateExtCCMOneAlnPos_DPRefined
#else 
#pragma omp declare simd linear(pi,pos:1) \
  uniform(dblen, tmpdpalnpossbuffer,ccm) \
  aligned(tmpdpalnpossbuffer:DATALN) \
  notinbranch
#endif
template<int XDIM, int DATALN>
inline
void MpStageFin::UpdateExtCCMOneAlnPos_DPRefined(
    int pos, const int dblen,
    const float* const __RESTRICT__ tmpdpalnpossbuffer,
    float (* __RESTRICT__ ccm)[XDIM], int pi)
{
    float qx = tmpdpalnpossbuffer[pos + dpapsQRYx * dblen];
    float qy = tmpdpalnpossbuffer[pos + dpapsQRYy * dblen];
    float qz = tmpdpalnpossbuffer[pos + dpapsQRYz * dblen];

    float rx = tmpdpalnpossbuffer[pos + dpapsRFNx * dblen];
    float ry = tmpdpalnpossbuffer[pos + dpapsRFNy * dblen];
    float rz = tmpdpalnpossbuffer[pos + dpapsRFNz * dblen];

    UpdateCCMCacheHelper<XDIM>(qx, qy, qz,  rx, ry, rz,  ccm, pi);

    ccm[twmvCV2q_0][pi] += SQRD(qx);
    ccm[twmvCV2q_1][pi] += SQRD(qy);
    ccm[twmvCV2q_2][pi] += SQRD(qz);

    ccm[twmvCV2r_0][pi] += SQRD(rx);
    ccm[twmvCV2r_1][pi] += SQRD(ry);
    ccm[twmvCV2r_2][pi] += SQRD(rz);
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// SaveBestQRScoresAndTM_Complete: complete version of saving transformation
// and the best scores calculated for the query and reference structures;
// save fragment indices and starting positions too;
// WRITEFRAGINFO, template parameter, flag of writing fragment attributes;
// CONDITIONAL, template parameter, flag of writing the score if it's greater at the same location;
// best, best score calculated for the smaller length;
// gbest, best score calculated for the greater length;
// qryndx, query serial number;
// dbstrndx, reference serial number;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps to perform for each reference structure;
// sfragfctxndx, current fragment factor x fragment length index, which is <maxnsteps;
// sfragndx, index defining fragment length;
// sfragpos, starting position within fragment;
// NOTE: memory pointers should be aligned!
// tfm, cached transformation matrix;
// wrkmemtmibest, working memory for iteration-best transformation matrices;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// 
template<bool WRITEFRAGINFO, bool CONDITIONAL>
inline
void MpStageFin::SaveBestQRScoresAndTM_Complete(
    const float best,
    const float gbest,
    const int qryndx,
    const int dbstrndx,
    const int ndbCstrs,
    const int maxnsteps,
    const int sfragfctxndx,
    const int sfragndx,
    const int sfragpos,
    const float* __RESTRICT__ tfm,
    float* __RESTRICT__ wrkmemtmibest,
    float* __RESTRICT__ wrkmemaux)
{
    // if(best <= 0.0f) return;

    float currentbest = -1.0f;
#ifdef MPSTAGEBASE_CONDITIONAL_FRAGNDX
    int currentsfragndx = 99;
#endif

    int mloc = ((qryndx * maxnsteps + sfragfctxndx) * nTAuxWorkingMemoryVars) * ndbCstrs;

    if(CONDITIONAL) {
        currentbest = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx];
#ifdef MPSTAGEBASE_CONDITIONAL_FRAGNDX
        currentsfragndx = wrkmemaux[mloc + tawmvSubFragNdxCurrent * ndbCstrs + dbstrndx];
#endif
    }

    bool condition = (currentbest < best)
#ifdef MPSTAGEBASE_CONDITIONAL_FRAGNDX
    || (currentbest == best && sfragndx < currentsfragndx)
#endif
    ;

    if(condition) {
        wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx] = best;
        //reuse the tawmvBest0 slot, which has been used only in DP-based refinement
        wrkmemaux[mloc + tawmvBest0 * ndbCstrs + dbstrndx] = gbest;
        if(WRITEFRAGINFO) {
            wrkmemaux[mloc + tawmvSubFragNdxCurrent * ndbCstrs + dbstrndx] = sfragndx;
            wrkmemaux[mloc + tawmvSubFragPosCurrent * ndbCstrs + dbstrndx] = sfragpos;
        }
    }

    //save transformation matrix
    if(condition) {
        mloc = ((qryndx * maxnsteps + sfragfctxndx) * ndbCstrs + dbstrndx) * nTTranformMatrix;
        #pragma omp simd
        for(int f = 0; f < nTTranformMatrix; f++) wrkmemtmibest[mloc + f] = tfm[f];
    }
}

// -------------------------------------------------------------------------
// SaveBestQRScoresAndTM_Phase2_logsearch_Complete: complete version of 
// saving transformation and the best scores calculated for the query and 
// reference structures directly to the production output memory region;
// see also ProductionSaveBestScoresAndTMAmongBests;
// best, best score calculated for the smaller length;
// gbest, best score calculated for the greater length;
// qryndx, query serial number;
// dbstrndx, reference serial number;
// ndbCstrs, total number of reference structures in the chunk;
// qrylenorg, dbstrlenorg, query and reference lengths;
// NOTE: memory pointers should be aligned!
// tfm, cached transformation matrix;
// tfmmem, output memory for best transformation matrices;
// alndatamem, memory for full alignment information, including scores;
// 
inline
void MpStageFin::SaveBestQRScoresAndTM_Phase2_logsearch_Complete(
    float best,
    float gbest,
    const int qryndx,
    const int dbstrndx,
    const int ndbCstrs,
    const int qrylenorg,
    const int dbstrlenorg,
    const float* __RESTRICT__ tfm,
    float* const __RESTRICT__ tfmmem,
    float* const __RESTRICT__ alndatamem)
{
    //save best scores
    int mloc = (qryndx * ndbCstrs + dbstrndx) * nTDP2OutputAlnData;
    //make best represent the query score:
    if(dbstrlenorg < qrylenorg) myswap(best, gbest);
    //NOTE: d0Q and d0R thresholds are assumed to be saved previously;
    alndatamem[mloc + dp2oadScoreQ] = best / (float)qrylenorg;
    alndatamem[mloc + dp2oadScoreR] = gbest / (float)dbstrlenorg;

    //save transformation matrix
    int tfmloc = (qryndx * ndbCstrs + dbstrndx) * nTTranformMatrix;
    #pragma omp simd
    for(int f = 0; f < nTTranformMatrix; f++) tfmmem[tfmloc + f] = tfm[f];
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// ProductionSaveBestScoresAndTMAmongBests: save best scores and respective 
// transformation matrices by considering all partial best scores 
// calculated over all fragment factors; production version;
// WRITEFRAGINFO, template parameter, whether to save a fragment length 
// index and position for the best score;
// CONDITIONAL, template parameter, flag of whether the grand best score is
// compared with the current best before writing;
// qryndx, query index;
// rfnblkndx, reference block index;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps performed for each reference structure;
// effnsteps, effective (actual maximum) number of steps;
// NOTE: memory pointers should be aligned!
// ccm, for local cache;
// wrkmemtmibest, working memory for iteration-best transformation matrices;
// wrkmemaux, auxiliary working memory;
// alndatamem, memory for full alignment information, including scores;
// tfmmem, memory for transformation matrices;
// 
template<int XDIM, int DATALN, bool WRITEFRAGINFO, bool CONDITIONAL>
inline
void MpStageFin::ProductionSaveBestScoresAndTMAmongBests(
    const int qryndx,
    const int rfnblkndx,
    const int ndbCstrs,
    const int maxnsteps,
    const int effnsteps,
    float (* __RESTRICT__ ccm)[XDIM],
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* const __RESTRICT__ wrkmemtmibest,
    float* const __RESTRICT__ wrkmemaux,
    float* const __RESTRICT__ alndatamem,
    float* const __RESTRICT__ tfmmem)
{
    //NOTE: lnxN <= dp2oadScoreQ and nTDP2OutputAlnData <= #ccm assumed!
    enum {lnxSCR, lnxNDX, lnxLEN, lnxGRD, lnxN};

    const int istr0 = rfnblkndx * XDIM;
    const int istre = mymin(istr0 + XDIM, (int)ndbCstrs);

    const int qrylen = PMBatchStrData::GetLengthAt(querypmbeg, qryndx);

    //initialize cache for best scores and their indices:
    #pragma omp simd collapse(2)
    for(int f = 0; f < lnxN; f++)
        for(int ii = 0; ii < XDIM; ii++) ccm[f][ii] = 0.0f;
    #pragma omp simd collapse(2)
    for(int f = dp2oadScoreQ; f < nTDP2OutputAlnData; f++)
        for(int ii = 0; ii < XDIM; ii++) ccm[f][ii] = 0.0f;

    #pragma omp simd aligned(bdbCpmbeg:DATALN)
    for(int ri = istr0; ri < istre; ri++) {
        int ii = ri - istr0;
        ccm[lnxLEN][ii] = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
    }

    for(int si = 0; si < effnsteps; si++) {
        int mloc = ((qryndx * maxnsteps + si/*sfragfct*/) * nTAuxWorkingMemoryVars) * ndbCstrs;
        #pragma omp simd aligned(wrkmemaux:DATALN)
        for(int ri = istr0; ri < istre; ri++) {
            int ii = ri - istr0;
            float bscore = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + ri/*dbstrndx*/];
            //reuse cache for scores:
            if(ccm[lnxSCR][ii] < bscore) {
                ccm[lnxSCR][ii] = bscore;
                ccm[lnxNDX][ii] = si;
            }
        }
    }

    bool wrtgrand = 0;
    const float d0Q = GetD0fin(qrylen, qrylen);//threshold for query

    //ccm[0][...] contains maximums; write max values to slot 0
    #pragma omp simd aligned(wrkmemaux:DATALN)
    for(int ri = istr0; ri < istre; ri++) {
        int ii = ri - istr0;
        int si = ccm[lnxNDX][ii];
        float bscore = ccm[lnxSCR][ii];
        float grand = 0.0f;
        int mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        int mloc = ((qryndx * maxnsteps + si) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(CONDITIONAL)
            grand = wrkmemaux[mloc0 + tawmvGrandBest * ndbCstrs + ri/*dbstrndx*/];
        ccm[lnxGRD][ii] = wrtgrand = (grand < bscore);//reuse cache
        //coalesced WRITE for multiple references
        if(wrtgrand) {
            wrkmemaux[mloc0 + tawmvGrandBest * ndbCstrs + ri/*dbstrndx*/] = bscore;
            //score calculated for the longer structure:
            float gbest = wrkmemaux[mloc + tawmvBest0 * ndbCstrs + ri/*dbstrndx*/];
            const float d0R = GetD0fin(ccm[lnxLEN][ii], ccm[lnxLEN][ii]);//threshold for reference
            //make bscore (should not be used below associated clauses) represent the query score:
            if(ccm[lnxLEN][ii] < (float)qrylen) myswap(bscore, gbest);
            //write alignment information in cache:
            ccm[dp2oadScoreQ][ii] = bscore / (float)qrylen;
            ccm[dp2oadScoreR][ii] = gbest / ccm[lnxLEN][ii];
            ccm[dp2oadD0Q][ii] = d0Q;
            ccm[dp2oadD0R][ii] = d0R;
            if(WRITEFRAGINFO) {
                float frgndx = wrkmemaux[mloc + tawmvSubFragNdxCurrent * ndbCstrs + ri/*dbstrndx*/];
                float frgpos = wrkmemaux[mloc + tawmvSubFragPosCurrent * ndbCstrs + ri/*dbstrndx*/];
                wrkmemaux[mloc0 + tawmvSubFragNdx * ndbCstrs + ri/*dbstrndx*/] = frgndx;
                wrkmemaux[mloc0 + tawmvSubFragPos * ndbCstrs + ri/*dbstrndx*/] = frgpos;
            }
        }
    }

    for(int ri = istr0; ri < istre; ri++) {
        int ii = ri - istr0;
        int si = ccm[lnxNDX][ii];
        int mloc = ((qryndx * maxnsteps + si) * ndbCstrs + ri/*dbstrndx*/) * nTTranformMatrix;
        int aloc = (qryndx * ndbCstrs + ri/*dbstrndx*/) * nTDP2OutputAlnData;
        int tfmloc = (qryndx * ndbCstrs + ri/*dbstrndx*/) * nTTranformMatrix;
        wrtgrand = ccm[lnxGRD][ii];
        //READ and WRITE iteration-best transformation matrices with the currently grand best score
        if(wrtgrand) {
            //NOTE: Clang crashes when alignment is given for both sections!
            // #pragma omp simd aligned(wrkmemtmibest,tfmmem:DATALN)
            #pragma omp simd aligned(wrkmemtmibest:DATALN)
            for(int f = 0; f < nTTranformMatrix; f++)
                tfmmem[tfmloc + f] = wrkmemtmibest[mloc + f];//READ/WRITE

            #pragma omp simd aligned(alndatamem:DATALN)
            for(int f = dp2oadScoreQ; f < nTDP2OutputAlnData; f++)
                alndatamem[aloc + f] = ccm[f][ii];//WRITE
        }
    }
}

// -------------------------------------------------------------------------

#endif//__MpStageFin_h__
