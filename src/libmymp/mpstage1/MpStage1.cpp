/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include <omp.h>

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libutil/templates.h"
#include "libutil/CLOptions.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libgenp/gdats/PMBatchStrData.h"
#include "libgenp/gdats/PMBatchStrData.inl"
#include "libmymp/mpproc/mpprocconf.h"
#include "libmymp/mplayout/MpGlobalMemory.h"
#include "libmymp/mpstages/scoringbase.h"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmymp/mpstage1/MpStageBase.h"
#include "libmymp/mpdp/MpDPHub.h"
#include "MpStage1.h"



// -------------------------------------------------------------------------
// Preinitialize1Kernel: initialize memory before major computations
//
void MpStage1::Preinitialize1Kernel(
    const bool condition4filter1,
    float* const __RESTRICT__ wrkmemtmibest,
    float* const __RESTRICT__ tfmmem,
    float* const __RESTRICT__ alndatamem,
    float* const __RESTRICT__ wrkmemaux)
{
    enum{
        attXDIM = MPS1_TBSP_SCORE_SET_XDIM,
        tfmXDIM = MPS1_TBINITSP_TFMINIT_XFCT
    };

    MYMSG("MpStage1::Preinitialize1Kernel", 4);
    // static const std::string preamb = "MpStage1::Preinitialize1Kernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = CuMemoryBase::GetMinMemAlignment();
    //execution configuration for attribute and tfm matrix initialization:
    //process pairs of one query and tfmXDIM reference structures:
    const int nblocks_x_attr = (ndbCstrs_ + attXDIM - 1) / attXDIM;
    const int nblocks_x = (ndbCstrs_ + tfmXDIM - 1) / tfmXDIM;
    const int nblocks_y = nqystrs_;
    const int nblocks_z = maxnsteps_;

    //cache for write flags of structure information
    // int wrts[tfmXDIM+1];

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared)
    {
        #pragma omp for collapse(3)
        for(int si = 0; si < nblocks_z; si++)
            for(int qi = 0; qi < nblocks_y; qi++)
                for(int bi = 0; bi < nblocks_x; bi++)
                {//threads process a group of references
                    const int istr0 = bi * tfmXDIM;
                    const int istre = mymin(istr0 + tfmXDIM, (int)ndbCstrs_);
                    const int memloc = ((qi * maxnsteps_ + si) * ndbCstrs_) * nTTranformMatrix;
                    const int mibeg = memloc + istr0 * nTTranformMatrix;
                    const int miend = memloc + istre * nTTranformMatrix;

                    //initialize wrkmemtmibest_
                    #pragma omp simd aligned(wrkmemtmibest:memalignment)
                    for(int mi = mibeg; mi < miend; mi++) wrkmemtmibest[mi] = 0.0f;

                    for(int mi = mibeg; mi < miend; mi += nTTranformMatrix) {
                        wrkmemtmibest[mi + tfmmRot_0_0] = 1.0f;
                        wrkmemtmibest[mi + tfmmRot_1_1] = 1.0f;
                        wrkmemtmibest[mi + tfmmRot_2_2] = 1.0f;
                    }

                    if(si == 0) {
                        //initialize grand best transformation matrices 
                        const int tfmloc = (qi * ndbCstrs_) * nTTranformMatrix;
                        const int tibeg = tfmloc + istr0 * nTTranformMatrix;
                        const int tiend = tfmloc + istre * nTTranformMatrix;

                        #pragma omp simd aligned(tfmmem:memalignment)
                        for(int mi = tibeg; mi < tiend; mi++) tfmmem[mi] = 0.0f;

                        for(int mi = tibeg; mi < tiend; mi += nTTranformMatrix) {
                            tfmmem[mi + tfmmRot_0_0] = 1.0f;
                            tfmmem[mi + tfmmRot_1_1] = 1.0f;
                            tfmmem[mi + tfmmRot_2_2] = 1.0f;
                        }

                        //initialize alignment data for query and reference pairs
                        const int almloc = (qi * ndbCstrs_) * nTDP2OutputAlnData;
                        const int aibeg = almloc + istr0 * nTDP2OutputAlnData;
                        const int aiend = almloc + istre * nTDP2OutputAlnData;

                        #pragma omp simd aligned(alndatamem:memalignment)
                        for(int mi = aibeg; mi < aiend; mi++) alndatamem[mi] = 0.0f;

                        //assign large values to RMSDs
                        for(int mi = aibeg; mi < aiend; mi += nTDP2OutputAlnData)
                            alndatamem[mi + dp2oadRMSD] = 9999.9f;
                    }
                }
        //implicit barrier here

        #pragma omp for collapse(3)
        for(int si = 0; si < nblocks_z; si++)
            for(int qi = 0; qi < nblocks_y; qi++)
                for(int bi = 0; bi < nblocks_x_attr; bi++)
                {//threads process a group of references
                    const int istr0 = bi * attXDIM;
                    const int istre = mymin(istr0 + attXDIM, (int)ndbCstrs_);
                    const int memloc = ((qi * maxnsteps_ + si) * nTAuxWorkingMemoryVars) * ndbCstrs_;

                    for(int f = 0; f < nTAuxWorkingMemoryVars; f++) {
                        //initialize all fields of auxiliary memory section
                        const int mibeg = memloc + f * (int)ndbCstrs_ + istr0;
                        const int miend = memloc + f * (int)ndbCstrs_ + istre;
                        const float value = 
                            (f == tawmvConverged && condition4filter1)? CONVERGED_LOWTMSC_bitval: 0.0f;
                        #pragma omp simd aligned(wrkmemaux:memalignment)
                        for(int mi = mibeg; mi < miend; mi++) wrkmemaux[mi] = value;
                    }
                }
    }
}



// -------------------------------------------------------------------------
// VerifyAlignmentScoreKernel: calculate local sequence alignment score
// between the queries and reference structures and set the flag of low 
// score if it is below the threshold;
// seqsimthrscore, sequence similarity threshold score;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// NOTE: process ungapped fragments of aligned query-reference structures;
// 
void MpStage1::VerifyAlignmentScoreKernel(
    const float seqsimthrscore,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    float* const __RESTRICT__ wrkmemaux)
{
    enum {
        CHSIZE = MPFL_TBSP_SEQUENCE_SIMILARITY_CHSIZE,
        DIMD = MPFL_TBSP_SEQUENCE_SIMILARITY_XDIM,
        EDGE = MPFL_TBSP_SEQUENCE_SIMILARITY_EDGE,
        STEP = MPFL_TBSP_SEQUENCE_SIMILARITY_STEP,
        STEPLOG2 = MPFL_TBSP_SEQUENCE_SIMILARITY_STEPLOG2
    };

    MYMSG("MpStage1::VerifyAlignmentScoreKernel", 4);
    // static const std::string preamb = "MpStage1::VerifyAlignmentScoreKernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    // constexpr int memalignment = CuMemoryBase::GetMinMemAlignment();
    //execution configuration for calculating local ungapped sequence alignment:
    const int ndiagonals = (qystr1len_ + dbstr1len_ - 2 * EDGE) >> STEPLOG2;
    const int nblocks_x = ndiagonals;
    const int nblocks_y = ndbCstrs_;
    const int nblocks_z = nqystrs_;

    if(seqsimthrscore <= 0.0f) return;

    size_t chunksize_helper = 
        ((size_t)nblocks_z * (size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)CHSIZE);

    //working cache:
    char rfnRE[DIMD];
    char qryRE[DIMD];
    float scores[DIMD];
    float pxmins[DIMD];
    float tmp[DIMD];

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        private(rfnRE,qryRE, scores,pxmins,tmp)
    {
        #pragma omp for collapse(3) schedule(dynamic, chunksize)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int ri = 0; ri < nblocks_y; ri++)
                for(int di = 0; di < nblocks_x; di++)
                {//threads process diagonals of a query-reference pairs
                    //read convergence first
                    const int mloc0 =
                        ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                    const int conv = (int)(wrkmemaux[mloc0 + ri]);
                    //skip upon unset convergence
                    if(conv == 0) continue;

                    const int qrylen = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                    const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                    const int dbstrlen = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                    const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);

                    int qrypos = di * STEP;
                    int rfnpos = 0;

                    if(qrylen - EDGE <= qrypos) {
                        rfnpos = ((qrypos - (qrylen - EDGE)) & (~STEPLOG2)) + STEP;
                        qrypos = 0;
                    }

                    //skip upon the out of bounds condition:
                    if(qrylen - EDGE <= qrypos || dbstrlen - EDGE <= rfnpos) continue;

                    const bool abovethreshold = 
                        PMBatchStrData::CheckAlignmentScore<DIMD>(
                            seqsimthrscore,
                            qrydst, qrylen, dbstrdst, dbstrlen, qrypos, rfnpos,
                            querypmbeg, bdbCpmbeg, rfnRE, qryRE, scores, pxmins, tmp);

                    //NOTE: several threads may write at the same time at the same location;
                    //NOTE: safe as long as it's the same value;
                    //reset convergence flag:
                    if(abovethreshold) wrkmemaux[mloc0 + ri] = 0;
                }
        //implicit barrier here
    }//omp parallel
}



// -------------------------------------------------------------------------
// FindFragKernel: search for superposition between multiple molecules 
// simultaneously and identify fragments for further refinement;
// TFM_DINV, use doubly inverted transformation matrices under suitable conditions;
//
template<bool TFM_DINV>
void MpStage1::FindFragKernel(
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    float* const __RESTRICT__ tmpdpdiagbuffers,
    float* const __RESTRICT__ wrkmemaux)
{
    enum{
        //effective number of fields:
        nEFFDS = twmvEndOfCCDataExt,
        XDIM = MPS1_TBINITSP_COMPLETEREFINE_XDIM
    };

    MYMSG("MpStage1::FindFragKernel", 4);
    // static const std::string preamb = "MpStage1::FindFragKernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = mycemin((size_t)PMBSdatalignment, CuMemoryBase::GetMinMemAlignment());
    //NOTE: minlen, minimum of the largest structures to compare, assumed >=3
    //minimum length among largest
    // int minlenmax = myhdmin(qystr1len_, dbstr1len_);
    //minimum length among smallest
    int minlenmin = mymin(qystrnlen_, dbstrnlen_);
    int minalnmin = mymax(minlenmin >> 1, 5);
    const int n1 = minalnmin - dbstr1len_;
    const int n2 = qystr1len_ - minalnmin;

    //execution configuration: process and refine multiple
    //query-reference alignment variants:
    const int nblocks_x = ndbCstrs_;
    const int nblocks_x_best = (ndbCstrs_ + XDIM - 1) / XDIM;
    const int nblocks_y = (n2 - n1) / stepinit_ + 1;
    const int nblocks_z = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_z * (size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPS1_TBINITSP_COMPLETEREFINE_CHSIZE);

    //cache for the cross-covarinace matrix and related data: 
    float ccm[nEFFDS][XDIM];
    float tfm[nEFFDS];//nEFFDS>nTTranformMatrix

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        private(ccm, tfm)
    {
        #pragma omp for collapse(3) schedule(static, chunksize)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nblocks_y; si++)
                for(int ri = 0; ri < nblocks_x; ri++)
                {//threads process references
                    const int qrylen = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                    const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                    const int dbstrlen = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                    const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);

                    const int abspos = n1 + si * stepinit_;
                    const int qrypos = mymax(0,abspos);
                    const int rfnpos = mymax(-abspos,0);
                    const int tid = omp_get_thread_num();

                    //out-of-bounds check:
                    const int prvminalnlen = mymax(mymin(qrylen, dbstrlen) >> 1, 5);
                    if(qrylen <= qrypos || dbstrlen <= rfnpos) continue;
                    if(qrylen - qrypos < prvminalnlen || dbstrlen - rfnpos < prvminalnlen) continue;

                    const float nalnposs = mymin(qrylen-qrypos, dbstrlen-rfnpos);

                    //threshold calculated for the original lengths
                    const float d0 = GetD0(qrylen, dbstrlen);
                    const float d02 = SQRD(d0);
                    float dst32 = CP_LARGEDST;
                    float best = 0.0f;//best score obtained

                    for(;;)
                    {
                        CalcCCMatrices_Complete<nEFFDS,XDIM,memalignment>(
                            qrydst, dbstrdst, nalnposs,  qrypos, rfnpos,
                            querypmbeg, bdbCpmbeg, ccm);

                        if(ccm[twmvNalnposs][0] < 1.0f) break;

                        //copy sums to tfm:
                        #pragma omp simd
                        for(int f = 0; f < nEFFDS; f++) tfm[f] = ccm[f][0];

                        CalcTfmMatrices_Complete<TFM_DINV>(qrylen, dbstrlen, tfm);


                        CalcScoresUnrl_Complete<SAVEPOS_SAVE,XDIM,memalignment>(
                            READCNST_CALC,
                            qi/*qryndx*/, ndbCposs_, maxnsteps_, tid/*sfragfct*/, qrydst, dbstrdst,
                            nalnposs, qrypos, rfnpos, d0, d02,
                            querypmbeg, bdbCpmbeg, tmpdpdiagbuffers,  tfm, ccm[0]/*scv*/, ccm/*dstv*/);

                        //distance threshold for at least three aligned pairs:
                        dst32 = ccm[0][1];
                        if(best < ccm[0][0]) best = ccm[0][0];//score written at [0,0]

                        CalcCCMatricesExtended_Complete<nEFFDS,XDIM,memalignment>(
                            qi/*qryndx*/, ndbCposs_, maxnsteps_, tid/*sfragfct*/, qrydst, dbstrdst,
                            nalnposs, qrypos, rfnpos, dst32,
                            querypmbeg, bdbCpmbeg, tmpdpdiagbuffers, ccm);

                        if(ccm[twmvNalnposs][0] < 1.0f) break;//exit the loop
                        if(ccm[twmvNalnposs][0] == nalnposs) break;//exit the loop

                        //copy sums to tfm:
                        #pragma omp simd
                        for(int f = 0; f < nEFFDS; f++) tfm[f] = ccm[f][0];

                        CalcTfmMatrices_Complete<TFM_DINV>(qrylen, dbstrlen, tfm);


                        CalcScoresUnrl_Complete<SAVEPOS_SAVE,XDIM,memalignment>(
                            READCNST_CALC2,
                            qi/*qryndx*/, ndbCposs_, maxnsteps_, tid/*sfragfct*/, qrydst, dbstrdst,
                            nalnposs, qrypos, rfnpos, d0, d02,
                            querypmbeg, bdbCpmbeg, tmpdpdiagbuffers,  tfm, ccm[0]/*scv*/, ccm/*dstv*/);

                        //distance threshold for at least three aligned pairs:
                        dst32 = ccm[0][1];
                        if(best < ccm[0][0]) best = ccm[0][0];//score written at [0,0]

                        CalcCCMatricesExtended_Complete<nEFFDS,XDIM,memalignment>(
                            qi/*qryndx*/, ndbCposs_, maxnsteps_, tid/*sfragfct*/, qrydst, dbstrdst,
                            nalnposs, qrypos, rfnpos, dst32,
                            querypmbeg, bdbCpmbeg, tmpdpdiagbuffers, ccm);

                        if(ccm[twmvNalnposs][0] < 1.0f) break;//exit the loop
                        if(ccm[twmvNalnposs][0] == nalnposs) break;//exit the loop

                        //copy sums to tfm:
                        #pragma omp simd
                        for(int f = 0; f < nEFFDS; f++) tfm[f] = ccm[f][0];

                        CalcTfmMatrices_Complete<TFM_DINV>(qrylen, dbstrlen, tfm);

                        CalcScoresUnrl_Complete<SAVEPOS_NOSAVE,XDIM,memalignment>(
                            READCNST_CALC2,
                            qi/*qryndx*/, ndbCposs_, maxnsteps_, tid/*sfragfct*/, qrydst, dbstrdst,
                            nalnposs, qrypos, rfnpos, d0, d02,
                            querypmbeg, bdbCpmbeg, tmpdpdiagbuffers,  tfm, ccm[0]/*scv*/, ccm/*dstv*/);

                        if(best < ccm[0][0]) best = ccm[0][0];//score written at [0,0]
                        break;
                    }//for(;;)

                    SaveBestScoreAndPositions_Complete(
                        best,  qi/*qryndx*/, ri/*dbstrndx*/, ndbCstrs_, 
                        maxnsteps_, tid/*sfragfct*/, qrypos, rfnpos, wrkmemaux);
                }
        //implicit barrier here

        //find the max score among fragment variants
        #pragma omp for collapse(2)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int bi = 0; bi < nblocks_x_best; bi++)
            {//threads process blocks of references
                SaveBestScoreAmongBests<XDIM,memalignment>(
                    qi/*qryndx*/, bi/*rfnblkndx*/, ndbCstrs_,
                    maxnsteps_, nthreads/*effnsteps*/, ccm, wrkmemaux);
            }
    }
}

// =========================================================================
// Instantiations
// 
#define INSTANTIATE_MpStage1_FindFragKernel(tpTFM_DINV) \
    template void MpStage1::FindFragKernel<tpTFM_DINV>( \
        const char* const * const __RESTRICT__ querypmbeg, \
        const char* const * const __RESTRICT__ bdbCpmbeg, \
        float* const tmpdpdiagbuffers, \
        float* const wrkmemaux);

INSTANTIATE_MpStage1_FindFragKernel(false);
INSTANTIATE_MpStage1_FindFragKernel(true);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// DPRefine: refine ungapped alignment identified during fragment 
// boundary refinement in the previous substage by iteratively applying 
// (gapped) DP followed by the same ungapped alignment boundary refinement;
// GAP0, template parameter, flag of using gap cost 0;
// PRESCREEN, template parameter, whether to verify scores for pre-termination;
// WRKMEMTM1, use for the 1st iteration tfms saved in wrkmemtm;
// prescorethr, provisional TM-score threshold for prescreening;
//
template<bool GAP0, bool PRESCREEN, bool WRKMEMTM1>
void MpStage1::DPRefine(
    const int maxndpiters,
    const float prescorethr)
{
    enum{ncosts = 1};
    static const float gcosts[ncosts] = {GAP0? 0.0f: -0.6f};

    constexpr bool vANCHORRGN = false;//using anchor region
    constexpr bool vBANDED = false;//banded alignment

    MpDPHub dphub(
        maxnsteps_,
        querypmbeg_, querypmend_, bdbCpmbeg_, bdbCpmend_,
        nqystrs_, ndbCstrs_, nqyposs_, ndbCposs_,  qystr1len_, dbstr1len_, dbxpad_,
        tmpdpdiagbuffers_, tmpdpbotbuffer_, tmpdpalnpossbuffer_, maxscoordsbuf_, btckdata_,
        wrkmem_, wrkmemccd_,  wrkmemtm_,  wrkmemtmibest_,
        wrkmemaux_, wrkmem2_, alndatamem_, tfmmem_, globvarsbuf_
    );

    for(int gi = 0; gi < ncosts; gi++)
    {
        for(int dpi = 0; dpi < maxndpiters; dpi++)
        {
            dphub.ExecDPwBtck128xKernel<vANCHORRGN,vBANDED,GAP0,D02IND_SEARCH>(
                gcosts[gi], 0/*stepnumber*/,
                querypmbeg_, bdbCpmbeg_,
                ((WRKMEMTM1 && gi < 1 && dpi < 1)? wrkmemtm_: wrkmemtmibest_),
                wrkmemaux_,  tmpdpdiagbuffers_, tmpdpbotbuffer_, btckdata_);

            dphub.BtckToMatched128xKernel<vANCHORRGN,vBANDED>(
                0/*stepnumber*/,//slot 0
                querypmbeg_, bdbCpmbeg_, btckdata_, wrkmemaux_, tmpdpalnpossbuffer_);

            RefineFragDPKernelCaller(true/*readlocalconv*/, FRAGREF_NMAXCONVIT);

            if(0 < dpi) CheckScoreConvergenceKernel(wrkmemaux_);

            if(dpi+1 < maxndpiters) SaveLastScore0Kernel(wrkmemaux_);

            if(PRESCREEN && maxndpiters <= dpi+1 && 0.0f < prescorethr)
                SetLowScoreConvergenceFlagKernel(prescorethr, querypmbeg_, bdbCpmbeg_, wrkmemaux_);
        }

        //reset the score convergence flag for the steps to follow not to halt
        InitScoresKernel<INITOPT_CONVFLAG_SCOREDP>(wrkmemaux_);
    }
}

// =========================================================================
// Instantiations
// 
#define INSTANTIATE_MpStage1_DPRefine(tpGAP0,tpPRESCREEN,tpWRKMEMTM1) \
    template void MpStage1::DPRefine<tpGAP0,tpPRESCREEN,tpWRKMEMTM1>( \
        const int maxndpiters, const float prescorethr);

INSTANTIATE_MpStage1_DPRefine(true,true,false);
INSTANTIATE_MpStage1_DPRefine(true,false,false);
INSTANTIATE_MpStage1_DPRefine(false,false,false);
INSTANTIATE_MpStage1_DPRefine(false,false,true);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// RefineFragInitKernel: refine identified fragments to improve the superposition score;
// TFM_DINV, use doubly inverted transformation matrices under suitable conditions;
// nmaxconvit, maximum number of superposition iterations;
//
template<bool TFM_DINV>
void MpStage1::RefineFragInitKernel(
    const int nmaxconvit,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    float* const __RESTRICT__ tmpdpdiagbuffers,
    float* const __RESTRICT__ wrkmemtm,
    float* const __RESTRICT__ wrkmemtmibest,
    float* const __RESTRICT__ wrkmemaux,
    float* const __RESTRICT__ tfmmem)
{
    enum{
        //effective number of fields:
        nEFFDS = twmvEndOfCCDataExt,
        XDIM = MPS1_TBINITSP_COMPLETEREFINE_XDIM
    };

    MYMSG("MpStage1::RefineFragInitKernel", 4);
    static const std::string preamb = "MpStage1::RefineFragInitKernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = CuMemoryBase::GetMinMemAlignment();
    //NOTE: minlen, minimum of the largest structures to compare, assumed >=3
    //minimum length among largest
    int minlenmax = myhdmin(qystr1len_, dbstr1len_);
    //minimum length among smallest
    // int minlenmin = mymin(qystrnlen_, dbstrnlen_);
    // int minalnmin = mymax(minlenmin >> 1, 5);
    int maxalnmax = minlenmax;
    //maximum number of subdivisions of identified fragments
    // (for all structures in the chunk)
    const int nmaxsubfrags = FRAGREF_NMAXSUBFRAGS;
    // sfragstep, step to traverse subfragments;
    const int sfragstep = FRAGREF_SFRAGSTEP;

    int nlocsteps = 0;
    nlocsteps = GetMaxNFragSteps(maxalnmax, sfragstep, minfraglen_);
    nlocsteps *= nmaxsubfrags;//total number across all fragment lengths

    if(nlocsteps < 1 || (int)maxnsteps_ < nlocsteps)
        throw MYRUNTIME_ERROR(preamb +
        "Invalid number of superposition tests: "+std::to_string(nlocsteps));

    //execution configuration: process and refine multiple
    //query-reference alignment variants:
    const int nblocks_x = ndbCstrs_;
    const int nblocks_x_best = (ndbCstrs_ + XDIM - 1) / XDIM;
    const int nblocks_y = nlocsteps;
    const int nblocks_z = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_z * (size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPS1_TBINITSP_COMPLETEREFINE_CHSIZE);

    size_t chunksizeinit_helper = 
        ((size_t)nblocks_z * (size_t)nthreads * (size_t)nblocks_x_best + (size_t)nthreads - 1) / nthreads;
    const int chunksizeinit = (int)mymin(chunksizeinit_helper, (size_t)MPS1_TBINITSP_COMPLETEREFINE_CHSIZE);

    //cache for the cross-covarinace matrix and related data: 
    float ccm[nEFFDS][XDIM];
    float tfm[nEFFDS];//nEFFDS>nTTranformMatrix
    float ccmLast[nEFFDS];
    float tfmBest[nEFFDS];

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        private(ccm, tfm, ccmLast, tfmBest)
    {
        //initialize best scores
        #pragma omp for collapse(3) schedule(dynamic, chunksizeinit)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nthreads; si++)
                for(int bi = 0; bi < nblocks_x_best; bi++)
                {//threads process blocks of references
                    const int istr0 = bi * XDIM;
                    const int istre = mymin(istr0 + XDIM, (int)ndbCstrs_);
                    const int mloc = ((qi * maxnsteps_ + si) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    #pragma omp simd aligned(wrkmemaux:memalignment)
                    for(int ri = istr0; ri < istre; ri++)
                        wrkmemaux[mloc + tawmvBestScore * ndbCstrs_ + ri/*dbstrndx*/] = 0.0f;
                }
        //implicit barrier here

        #pragma omp for collapse(3) schedule(static, chunksize)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nblocks_y; si++)
                for(int ri = 0; ri < nblocks_x; ri++)
                {//threads process references
                    const int sfragfct = si / nmaxsubfrags;//fragment factor
                    const int sfragndx = si - sfragfct * nmaxsubfrags;//fragment length index

                    const int qrylen = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                    const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                    const int dbstrlen = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                    const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);

                    const int mloc = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    const int qrypos = wrkmemaux[mloc + tawmvQRYpos * ndbCstrs_ + ri];
                    const int rfnpos = wrkmemaux[mloc + tawmvRFNpos * ndbCstrs_ + ri];
                    const int sfragpos = sfragfct * sfragstep;
                    const int tid = omp_get_thread_num();

                    //out-of-bounds check:
                    if(qrylen <= qrypos || dbstrlen <= rfnpos) continue;
                    const int fraglen = GetFragLength(qrylen, dbstrlen, qrypos, rfnpos, sfragndx);
                    if(fraglen < 1) continue;
                    if(qrylen + sfragstep <= qrypos + sfragpos + fraglen ||
                       dbstrlen + sfragstep <= rfnpos + sfragpos + fraglen)
                        continue;

                    //threshold calculated for the original lengths
                    const float d0 = GetD0(qrylen, dbstrlen);
                    const float d02 = SQRD(d0);
                    const float d82 = GetD82(qrylen, dbstrlen);
                    float dst32 = CP_LARGEDST;
                    float best = 0.0f;//best score obtained

                    CalcCCMatricesRefined_Complete<nEFFDS,XDIM,memalignment>(
                        qrydst, dbstrdst, fraglen,
                        qrylen, dbstrlen,  qrypos + sfragpos, rfnpos + sfragpos,
                        querypmbeg, bdbCpmbeg, ccm);

                    for(int cit = 0; cit < nmaxconvit + 2; cit++)
                    {
                        if(0 < cit) {
                            CalcCCMatricesRefinedExtended_Complete<nEFFDS,XDIM,memalignment>(
                                (cit < 2)? READCNST_CALC: READCNST_CALC2,
                                qi/*qryndx*/, ndbCposs_, maxnsteps_, tid/*sfragfctxndx*/, qrydst, dbstrdst,
                                qrylen, dbstrlen, qrypos, rfnpos,  d0, dst32,
                                querypmbeg, bdbCpmbeg, tmpdpdiagbuffers, ccm);

                            CheckConvergenceRefined_Complete<nEFFDS,XDIM>(ccm, ccmLast);
                            if(ccmLast[0]) break;//converged
                        }

                        if(ccm[twmvNalnposs][0] < 1.0f) break;

                        //copy sums to tfm and ccmLast:
                        #pragma omp simd
                        for(int f = 0; f < nEFFDS; f++) tfm[f] = ccmLast[f] = ccm[f][0];

                        CalcTfmMatrices_Complete<TFM_DINV>(qrylen, dbstrlen, tfm);

                        CalcScoresUnrlRefined_Complete<XDIM,memalignment>(
                            (cit < 1)? READCNST_CALC: READCNST_CALC2,
                            qi/*qryndx*/, ndbCposs_, maxnsteps_, tid/*sfragfctxndx*/, qrydst, dbstrdst,
                            qrylen, dbstrlen, qrypos, rfnpos,  d0, d02, d82,
                            querypmbeg, bdbCpmbeg, tmpdpdiagbuffers,  tfm, ccm[0]/*scv*/, ccm/*dstv*/);

                        //distance threshold for at least three aligned pairs:
                        dst32 = ccm[0][1];
                        if(best < ccm[0][0]) {
                            best = ccm[0][0];//score written at [0,0]
                            #pragma omp simd
                            for(int f = 0; f < nTTranformMatrix; f++) tfmBest[f] = tfm[f];
                        }
                    }//for(;cit;)

                    //NOTE: CONDITIONAL==true because effnsteps(==nthreads)<<maxnsteps
                    SaveBestScoreAndTM_Complete<false/*WRITEFRAGINFO*/,true/*CONDITIONAL*/>(
                        best,  qi/*qryndx*/, ri/*dbstrndx*/, ndbCstrs_, 
                        maxnsteps_, tid/*sfragfctxndx*/, sfragndx, sfragpos,
                        tfmBest, wrkmemtmibest, wrkmemaux);
                }
        //implicit barrier here

        //find the max score among fragment variants
        #pragma omp for collapse(2)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int bi = 0; bi < nblocks_x_best; bi++)
            {//threads process blocks of references
                SaveBestScoreAndTMAmongBests
                    <XDIM,memalignment,
                     false/*WRITEFRAGINFO*/,
                     true/*GRANDUPDATE*/,
                     false/*FORCEWRITEFRAGINFO*/,
                     SECONDARYUPDATE_NOUPDATE>(
                        qi/*qryndx*/, bi/*rfnblkndx*/, ndbCstrs_,
                        maxnsteps_, nthreads/*effnsteps*/,
                        ccm, wrkmemtmibest, tfmmem, wrkmemaux,  wrkmemtm);
            }
    }
}

// =========================================================================
// Instantiations
// 
#define INSTANTIATE_MpStage1_RefineFragInitKernel(tpTFM_DINV) \
    template void MpStage1::RefineFragInitKernel<tpTFM_DINV>( \
        const int nmaxconvit, \
        const char* const * const __RESTRICT__ querypmbeg, \
        const char* const * const __RESTRICT__ bdbCpmbeg, \
        float* const __RESTRICT__ tmpdpdiagbuffers, \
        float* const __RESTRICT__ wrkmemtm, \
        float* const __RESTRICT__ wrkmemtmibest, \
        float* const __RESTRICT__ wrkmemaux, \
        float* const __RESTRICT__ tfmmem);

INSTANTIATE_MpStage1_RefineFragInitKernel(false);
INSTANTIATE_MpStage1_RefineFragInitKernel(true);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// RefineFragDPKernel: refine alignment and its boundaries 
// within the single kernel's actions to obtain favorable superposition;
// TFM_DINV, use doubly inverted transformation matrices under suitable conditions;
// // CONDITIONAL, template parameter, flag of writing the score if it's 
// // greater at the same location;
// SECONDARYUPDATE, indication of whether and how secondary update of best scores is done;
// readlocalconv, flag of reading local convergence flag;
// nmaxconvit, maximum number of superposition iterations;
//
template<bool TFM_DINV, int SECONDARYUPDATE>
void MpStage1::RefineFragDPKernel(
    const bool readlocalconv,
    const int nmaxconvit,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* const __RESTRICT__ tmpdpalnpossbuffer,
    float* const __RESTRICT__ tmpdpdiagbuffers,
    float* const __RESTRICT__ wrkmemtm,
    float* const __RESTRICT__ wrkmemtmibest,
    float* const __RESTRICT__ wrkmemaux,
    float* const __RESTRICT__ tfmmem)
{
    enum{
        //effective number of fields:
        nEFFDS = twmvEndOfCCDataExt,
        XDIM = MPS1_TBINITSP_COMPLETEREFINE_XDIM
    };

    MYMSG("MpStage1::RefineFragDPKernel", 4);
    static const std::string preamb = "MpStage1::RefineFragDPKernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = CuMemoryBase::GetMinMemAlignment();
    //NOTE: minlen, minimum of the largest structures to compare, assumed >=3
    //minimum length among largest
    int minlenmax = myhdmin(qystr1len_, dbstr1len_);
    //minimum length among smallest
    // int minlenmin = mymin(qystrnlen_, dbstrnlen_);
    // int minalnmin = mymax(minlenmin >> 1, 5);
    int maxalnmax = minlenmax;
    //maximum number of subdivisions of identified fragments
    // (for all structures in the chunk)
    const int nmaxsubfrags = FRAGREF_NMAXSUBFRAGS;
    // sfragstep, step to traverse subfragments;
    const int sfragstep = FRAGREF_SFRAGSTEP;

    int nlocsteps = 0;
    nlocsteps = GetMaxNFragSteps(maxalnmax, sfragstep, minfraglen_);
    nlocsteps *= nmaxsubfrags;//total number across all fragment lengths

    if(nlocsteps < 1 || (int)maxnsteps_ < nlocsteps)
        throw MYRUNTIME_ERROR(preamb +
        "Invalid number of superposition tests: "+std::to_string(nlocsteps));

    //execution configuration: process and refine multiple
    //query-reference alignment variants:
    const int nblocks_x = ndbCstrs_;
    const int nblocks_x_best = (ndbCstrs_ + XDIM - 1) / XDIM;
    const int nblocks_y = nlocsteps;
    const int nblocks_z = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_z * (size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPS1_TBINITSP_COMPLETEREFINE_CHSIZE);

    size_t chunksizeinit_helper = 
        ((size_t)nblocks_z * (size_t)nthreads * (size_t)nblocks_x_best + (size_t)nthreads - 1) / nthreads;
    const int chunksizeinit = (int)mymin(chunksizeinit_helper, (size_t)MPS1_TBINITSP_COMPLETEREFINE_CHSIZE);

    //cache for the cross-covarinace matrix and related data: 
    float ccm[nEFFDS][XDIM];
    float tfm[nEFFDS];//nEFFDS>nTTranformMatrix
    float ccmLast[nEFFDS];
    float tfmBest[nEFFDS];

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        private(ccm, tfm, ccmLast, tfmBest)
    {
        //initialize best scores
        #pragma omp for collapse(3) schedule(dynamic, chunksizeinit)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nthreads; si++)
                for(int bi = 0; bi < nblocks_x_best; bi++)
                {//threads process blocks of references
                    const int istr0 = bi * XDIM;
                    const int istre = mymin(istr0 + XDIM, (int)ndbCstrs_);
                    const int mloc = ((qi * maxnsteps_ + si) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    #pragma omp simd aligned(wrkmemaux:memalignment)
                    for(int ri = istr0; ri < istre; ri++)
                        wrkmemaux[mloc + tawmvBestScore * ndbCstrs_ + ri/*dbstrndx*/] = 0.0f;
                }
        //implicit barrier here

        #pragma omp for collapse(3) schedule(static, chunksize)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nblocks_y; si++)
                for(int ri = 0; ri < nblocks_x; ri++)
                {//threads process references
                    //check convergence:
                    int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                    int mloc = ((qi * maxnsteps_ + si) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                    tfm[6] = tfm[7] = wrkmemaux[mloc0 + ri];//reuse cache
                    if(readlocalconv && si != 0) tfm[7] = wrkmemaux[mloc + ri];
                    if((((int)(tfm[6])) & (CONVERGED_LOWTMSC_bitval)) || tfm[7])
                        //(NOTE:any type of convergence applies locally and CONVERGED_LOWTMSC_bitval globally);
                        continue;

                    const int sfragfct = si / nmaxsubfrags;//fragment factor
                    const int sfragndx = si - sfragfct * nmaxsubfrags;//fragment length index

                    const int qrylenorg = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                    // const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                    const int dbstrlenorg = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                    const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);

                    enum {qrypos = 0, rfnpos = 0};
                    const int sfragpos = sfragfct * sfragstep;
                    const int tid = omp_get_thread_num();
                    int qrylen, dbstrlen;

                    //NOTE: #matched positions tawmvNAlnPoss written at sfragfct==0:
                    mloc = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    qrylen = dbstrlen = wrkmemaux[mloc + tawmvNAlnPoss * ndbCstrs_ + ri];

                    //out-of-bounds check:
                    const int fraglen = GetFragLength(qrylen, dbstrlen, qrypos, rfnpos, sfragndx);
                    if(fraglen < 1) continue;
                    if(qrylen + sfragstep <= qrypos + sfragpos + fraglen ||
                       dbstrlen + sfragstep <= rfnpos + sfragpos + fraglen)
                        continue;

                    //threshold calculated for the original lengths
                    const float d0 = GetD0(qrylenorg, dbstrlenorg);
                    const float d02 = SQRD(d0);
                    const float d82 = GetD82(qrylenorg, dbstrlenorg);
                    float dst32 = CP_LARGEDST;
                    float best = 0.0f;//best score obtained

                    CalcCCMatrices_DPRefined_Complete<nEFFDS,XDIM,memalignment>(
                        qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_, tid/*sfragfctxndx*/, dbstrdst, fraglen,
                        qrylen, dbstrlen,  qrypos + sfragpos, rfnpos + sfragpos,
                        tmpdpalnpossbuffer, ccm);

                    for(int cit = 0; cit < nmaxconvit + 2; cit++)
                    {
                        if(0 < cit) {
                            CalcCCMatrices_DPRefinedExtended_Complete<nEFFDS,XDIM,memalignment>(
                                (cit < 2)? READCNST_CALC: READCNST_CALC2,
                                qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_, tid/*sfragfctxndx*/, dbstrdst,
                                qrylen, dbstrlen, qrypos, rfnpos,  d0, dst32,
                                tmpdpdiagbuffers, tmpdpalnpossbuffer, ccm);

                            CheckConvergenceRefined_Complete<nEFFDS,XDIM>(ccm, ccmLast);
                            if(ccmLast[0]) break;//converged
                        }

                        if(ccm[twmvNalnposs][0] < 1.0f) break;

                        //copy sums to tfm and ccmLast:
                        #pragma omp simd
                        for(int f = 0; f < nEFFDS; f++) tfm[f] = ccmLast[f] = ccm[f][0];

                        CalcTfmMatrices_Complete<TFM_DINV>(qrylenorg, dbstrlenorg, tfm);

                        CalcScoresUnrl_DPRefined_Complete<XDIM,memalignment,CHCKDST_CHECK>(
                            (cit < 1)? READCNST_CALC: READCNST_CALC2,
                            qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_, tid/*sfragfctxndx*/, dbstrdst,
                            qrylen, dbstrlen, qrypos, rfnpos,  d0, d02, d82,
                            tmpdpdiagbuffers, tmpdpalnpossbuffer,  tfm, ccm[0]/*scv*/, ccm/*dstv*/);

                        //distance threshold for at least three aligned pairs:
                        dst32 = ccm[0][1];
                        if(best < ccm[0][0]) {
                            best = ccm[0][0];//score written at [0,0]
                            #pragma omp simd
                            for(int f = 0; f < nTTranformMatrix; f++) tfmBest[f] = tfm[f];
                        }
                    }//for(;cit;)

                    //NOTE: CONDITIONAL==true because effnsteps(==nthreads)<<maxnsteps
                    SaveBestScoreAndTM_Complete<false/*WRITEFRAGINFO*/,true/*CONDITIONAL*/>(
                        best,  qi/*qryndx*/, ri/*dbstrndx*/, ndbCstrs_, 
                        maxnsteps_, tid/*sfragfctxndx*/, sfragndx, sfragpos,
                        tfmBest, wrkmemtmibest, wrkmemaux);
                }
        //implicit barrier here

        //find the max score among fragment variants
        #pragma omp for collapse(2)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int bi = 0; bi < nblocks_x_best; bi++)
            {//threads process blocks of references
                SaveBestScoreAndTMAmongBests
                    <XDIM,memalignment,
                     false/*WRITEFRAGINFO*/,
                     true/*GRANDUPDATE*/,
                     false/*FORCEWRITEFRAGINFO*/,
                     SECONDARYUPDATE>(
                        qi/*qryndx*/, bi/*rfnblkndx*/, ndbCstrs_,
                        maxnsteps_, nthreads/*effnsteps*/,
                        ccm, wrkmemtmibest, tfmmem, wrkmemaux,  wrkmemtm);
            }
    }
}

// =========================================================================
// Instantiations
// 
#define INSTANTIATE_MpStage1_RefineFragDPKernel(tpTFM_DINV,tpSECONDARYUPDATE) \
    template void MpStage1::RefineFragDPKernel<tpTFM_DINV,tpSECONDARYUPDATE>( \
        const bool readlocalconv, const int nmaxconvit, \
        const char* const * const __RESTRICT__ querypmbeg, \
        const char* const * const __RESTRICT__ bdbCpmbeg, \
        const float* const __RESTRICT__ tmpdpalnpossbuffer, \
        float* const __RESTRICT__ tmpdpdiagbuffers, \
        float* const __RESTRICT__ wrkmemtm, \
        float* const __RESTRICT__ wrkmemtmibest, \
        float* const __RESTRICT__ wrkmemaux, \
        float* const __RESTRICT__ tfmmem);

INSTANTIATE_MpStage1_RefineFragDPKernel(false,SECONDARYUPDATE_NOUPDATE);
INSTANTIATE_MpStage1_RefineFragDPKernel(true,SECONDARYUPDATE_NOUPDATE);
INSTANTIATE_MpStage1_RefineFragDPKernel(false,SECONDARYUPDATE_UNCONDITIONAL);
INSTANTIATE_MpStage1_RefineFragDPKernel(true,SECONDARYUPDATE_UNCONDITIONAL);
INSTANTIATE_MpStage1_RefineFragDPKernel(false,SECONDARYUPDATE_CONDITIONAL);
INSTANTIATE_MpStage1_RefineFragDPKernel(true,SECONDARYUPDATE_CONDITIONAL);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// InitScoresKernel: initialize scores and associated fields;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
template<int INITOPT>
void MpStage1::InitScoresKernel(
    float* const __RESTRICT__ wrkmemaux)
{
    enum{XDIM = MPS1_TBSP_SCORE_SET_XDIM};

    MYMSG("MpStage1::InitScoresKernel", 4);
    static const std::string preamb = "MpStage1::InitScoresKernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = CuMemoryBase::GetMinMemAlignment();

    //execution configuration for query-reference flags across fragment factors:
    const int nblocks_x = (ndbCstrs_ + XDIM - 1) / XDIM;
    const int nblocks_y = maxnsteps_;//nthreads
    const int nblocks_z = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_z * (size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPS1_TBSP_SCORE_SET_CHSIZE);

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared)
    {
        //initialize best scores
        #pragma omp for collapse(3) schedule(dynamic, chunksize)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nblocks_y; si++)
                for(int bi = 0; bi < nblocks_x; bi++)
                {//threads process blocks of references
                    const int istr0 = bi * XDIM;
                    const int istre = mymin(istr0 + XDIM, (int)ndbCstrs_);
                    const int mloc = ((qi * maxnsteps_ + si) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    if(INITOPT & INITOPT_ALL) {
                        for(int f = 0; f < nTAuxWorkingMemoryVars; f++) {
                            #pragma omp simd aligned(wrkmemaux:memalignment)
                            for(int ri = istr0; ri < istre; ri++)
                                wrkmemaux[mloc + f * ndbCstrs_ + ri] = 0.0f;
                        }
                    }
                    if(INITOPT & INITOPT_BEST) {
                        #pragma omp simd aligned(wrkmemaux:memalignment)
                        for(int ri = istr0; ri < istre; ri++)
                            wrkmemaux[mloc + tawmvBestScore * ndbCstrs_ + ri] = 0.0f;
                    }
                    if(INITOPT & INITOPT_CURRENT) {
                        #pragma omp simd aligned(wrkmemaux:memalignment)
                        for(int ri = istr0; ri < istre; ri++)
                            wrkmemaux[mloc + tawmvScore * ndbCstrs_ + ri] = 0.0f;
                    }
                    if(INITOPT & INITOPT_NALNPOSS) {
                        #pragma omp simd aligned(wrkmemaux:memalignment)
                        for(int ri = istr0; ri < istre; ri++)
                            wrkmemaux[mloc + tawmvNAlnPoss * ndbCstrs_ + ri] = 0.0f;
                    }
                    if(INITOPT & INITOPT_CONVFLAG_ALL) {
                        #pragma omp simd aligned(wrkmemaux:memalignment)
                        for(int ri = istr0; ri < istre; ri++)
                            wrkmemaux[mloc + tawmvConverged * ndbCstrs_ + ri] = 0.0f;
                    }
                    if(INITOPT &
                        (INITOPT_CONVFLAG_FRAGREF | INITOPT_CONVFLAG_SCOREDP |
                        INITOPT_CONVFLAG_NOTMPRG | INITOPT_CONVFLAG_LOWTMSC |
                        INITOPT_CONVFLAG_LOWTMSC_SET))
                    {
                        #pragma omp simd aligned(wrkmemaux:memalignment)
                        for(int ri = istr0; ri < istre; ri++) {
                            int memloc = mloc + tawmvConverged * ndbCstrs_ + ri;
                            int convflag = wrkmemaux[memloc];//float->int
                            if(INITOPT & INITOPT_CONVFLAG_FRAGREF)
                                if(convflag & CONVERGED_FRAGREF_bitval)
                                    convflag = convflag & (~CONVERGED_FRAGREF_bitval);
                            if(INITOPT & INITOPT_CONVFLAG_SCOREDP)
                                if(convflag & CONVERGED_SCOREDP_bitval)
                                    convflag = convflag & (~CONVERGED_SCOREDP_bitval);
                            if(INITOPT & INITOPT_CONVFLAG_NOTMPRG)
                                if(convflag & CONVERGED_NOTMPRG_bitval)
                                    convflag = convflag & (~CONVERGED_NOTMPRG_bitval);
                            if(INITOPT & INITOPT_CONVFLAG_LOWTMSC)
                                if(convflag & CONVERGED_LOWTMSC_bitval)
                                    convflag = convflag & (~CONVERGED_LOWTMSC_bitval);
                            if(INITOPT & INITOPT_CONVFLAG_LOWTMSC_SET)
                                convflag = convflag | (CONVERGED_LOWTMSC_bitval);
                            wrkmemaux[memloc] = (float)convflag;//int->float
                        }
                    }
                }//omp for
    }//omp parallel
}

// =========================================================================
// Instantiations
// 
#define INSTANTIATE_MpStage1_InitScoresKernel(tpINITOPT) \
    template void MpStage1::InitScoresKernel<tpINITOPT>( \
        float* const __RESTRICT__ wrkmemaux);

INSTANTIATE_MpStage1_InitScoresKernel(INITOPT_ALL);
INSTANTIATE_MpStage1_InitScoresKernel(INITOPT_BEST);
INSTANTIATE_MpStage1_InitScoresKernel(INITOPT_CURRENT);
INSTANTIATE_MpStage1_InitScoresKernel(INITOPT_NALNPOSS);
INSTANTIATE_MpStage1_InitScoresKernel(INITOPT_CONVFLAG_ALL);
INSTANTIATE_MpStage1_InitScoresKernel(INITOPT_CONVFLAG_SCOREDP);
// INSTANTIATE_MpStage1_InitScoresKernel(INITOPT_CONVFLAG_NOTMPRG);
// INSTANTIATE_MpStage1_InitScoresKernel(INITOPT_CONVFLAG_LOWTMSC);
INSTANTIATE_MpStage1_InitScoresKernel(INITOPT_CONVFLAG_LOWTMSC_SET|INITOPT_ALL);
INSTANTIATE_MpStage1_InitScoresKernel(INITOPT_NALNPOSS|INITOPT_CONVFLAG_SCOREDP);
INSTANTIATE_MpStage1_InitScoresKernel(INITOPT_BEST|INITOPT_CONVFLAG_ALL);

// -------------------------------------------------------------------------
// CheckScoreConvergenceKernel: check whether the score of the last two 
// procedures converged, i.e., the difference is small;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
void MpStage1::CheckScoreConvergenceKernel(
    float* const __RESTRICT__ wrkmemaux)
{
    enum{XDIM = MPS1_TBSP_SCORE_SET_XDIM};

    MYMSG("MpStage1::CheckScoreConvergenceKernel", 4);
    static const std::string preamb = "MpStage1::CheckScoreConvergenceKernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = CuMemoryBase::GetMinMemAlignment();

    //execution configuration for query-reference flags across fragment factors:
    const int nblocks_x = (ndbCstrs_ + XDIM - 1) / XDIM;
    const int nblocks_y = maxnsteps_;//nthreads
    const int nblocks_z = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_z * (size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPS1_TBSP_SCORE_SET_CHSIZE);

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared)
    {
        //initialize best scores
        #pragma omp for collapse(3) schedule(dynamic, chunksize)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nblocks_y; si++)
                for(int bi = 0; bi < nblocks_x; bi++)
                {//threads process blocks of references
                    const int istr0 = bi * XDIM;
                    const int istre = mymin(istr0 + XDIM, (int)ndbCstrs_);
                    const int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    const int mloc = ((qi * maxnsteps_ + si) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    #pragma omp simd aligned(wrkmemaux:memalignment)
                    for(int ri = istr0; ri < istre; ri++) {
                        int converged = wrkmemaux[mloc + tawmvConverged * ndbCstrs_ + ri];//float->int
                        //best scores are recorded at a fragment factor position of 0
                        float prevbest0 = wrkmemaux[mloc0 + tawmvBest0 * ndbCstrs_ + ri];
                        float best0 = wrkmemaux[mloc0 + tawmvBestScore * ndbCstrs_ + ri];
                        bool condition = 
                            !(converged & (CONVERGED_SCOREDP_bitval | CONVERGED_LOWTMSC_bitval)) &&
                            (fabsf(best0-prevbest0) < SCORE_CONVEPSILON);
                        if(condition)
                            wrkmemaux[mloc + tawmvConverged * ndbCstrs_ + ri] =
                                (float)(converged | CONVERGED_SCOREDP_bitval);
                    }
                }//omp for
    }//omp parallel
}

// -------------------------------------------------------------------------
// SaveLastScore0Kernel: save the last best score at the position 
// corresponding to fragment factor 0;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
void MpStage1::SaveLastScore0Kernel(
    float* const __RESTRICT__ wrkmemaux)
{
    enum{XDIM = MPS1_TBSP_SCORE_SET_XDIM};

    MYMSG("MpStage1::SaveLastScore0Kernel", 4);
    static const std::string preamb = "MpStage1::SaveLastScore0Kernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = CuMemoryBase::GetMinMemAlignment();

    //execution configuration for query-reference flags across fragment factors:
    const int nblocks_x = (ndbCstrs_ + XDIM - 1) / XDIM;
    const int nblocks_y = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPS1_TBSP_SCORE_SET_CHSIZE);

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared)
    {
        //initialize best scores
        #pragma omp for collapse(2) schedule(dynamic, chunksize)
        for(int qi = 0; qi < nblocks_y; qi++)
            for(int bi = 0; bi < nblocks_x; bi++)
            {//threads process blocks of references
                const int istr0 = bi * XDIM;
                const int istre = mymin(istr0 + XDIM, (int)ndbCstrs_);
                const int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                #pragma omp simd aligned(wrkmemaux:memalignment)
                for(int ri = istr0; ri < istre; ri++) {
                    int converged = wrkmemaux[mloc0 + tawmvConverged * ndbCstrs_ + ri];//float->int
                    //score convergence applies:
                    bool condition = 
                        !(converged & (CONVERGED_SCOREDP_bitval | CONVERGED_LOWTMSC_bitval));
                    if(condition)
                        wrkmemaux[mloc0 + tawmvBest0 * ndbCstrs_ + ri] =
                            wrkmemaux[mloc0 + tawmvBestScore * ndbCstrs_ + ri];
                }
            }//omp for
    }//omp parallel
}

// -------------------------------------------------------------------------
// SetLowScoreConvergenceFlagKernel: set the appropriate convergence flag for 
// the pairs for which the score is below the threshold;
// scorethld, score threshold;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
void MpStage1::SetLowScoreConvergenceFlagKernel(
    const float scorethld,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    float* const __RESTRICT__ wrkmemaux)
{
    enum{XDIM = MPS1_TBSP_SCORE_SET_XDIM};

    MYMSG("MpStage1::SetLowScoreConvergenceFlagKernel", 4);
    static const std::string preamb = "MpStage1::SetLowScoreConvergenceFlagKernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = mycemin((size_t)PMBSdatalignment, CuMemoryBase::GetMinMemAlignment());

    //execution configuration for query-reference flags across fragment factors:
    const int nblocks_x = (ndbCstrs_ + XDIM - 1) / XDIM;
    const int nblocks_y = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPS1_TBSP_SCORE_SET_CHSIZE);

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared)
    {
        //initialize best scores
        #pragma omp for collapse(2) schedule(dynamic, chunksize)
        for(int qi = 0; qi < nblocks_y; qi++)
            for(int bi = 0; bi < nblocks_x; bi++)
            {//threads process blocks of references
                const int istr0 = bi * XDIM;
                const int istre = mymin(istr0 + XDIM, (int)ndbCstrs_);
                const int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                const int qrylen = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                #pragma omp simd aligned(bdbCpmbeg,wrkmemaux:memalignment)
                for(int ri = istr0; ri < istre; ri++) {
                    int dbstrlen = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                    //grand scores fragment factor position 0:
                    float grand = wrkmemaux[mloc0 + tawmvGrandBest * ndbCstrs_ + ri];
                    int convflag = wrkmemaux[mloc0 + tawmvConverged * ndbCstrs_ + ri];//float->int
                    //check for low scores:
                    bool condition = (grand < scorethld * (float)mymin(qrylen, dbstrlen));
                    if(condition)
                        wrkmemaux[mloc0 + tawmvConverged * ndbCstrs_ + ri] =
                            (float)(convflag | CONVERGED_LOWTMSC_bitval);
                }
            }//omp for
    }//omp parallel
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
