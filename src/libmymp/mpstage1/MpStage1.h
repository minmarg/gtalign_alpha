/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __MpStage1_h__
#define __MpStage1_h__

#include "libutil/mybase.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmymp/mpproc/mpprocconfbase.h"
#include "libmymp/mpstages/transformbase.h"
#include "libmymp/mpstages/scoringbase.h"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/fragment.cuh"
#include "libmymp/mpstage1/MpStageBase.h"
#include "libmycu/cucom/cudef.h"

// -------------------------------------------------------------------------
// class MpStage1 for implementing structure comparison at stage 1
//
class MpStage1: public MpStageBase {
    // constants defining the type of refinement: initial call, initial call on 
    // aligned positions, and iterative refinement involving DP:
    enum {
        mpstg1REFINE_INITIAL,
        mpstg1REFINE_INITIAL_DP,
        mpstg1REFINE_ITERATIVE_DP,
    };

public:
    MpStage1(
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
        maxndpiters_(maxndpiters),
        prescore_(prescore),
        stepinit_(stepinit)
    {}

    void Preinitialize1(const bool condition4filter1) {
        Preinitialize1Kernel(
            condition4filter1,
            wrkmemtmibest_, tfmmem_, alndatamem_, wrkmemaux_);
    }

    void VerifyAlignmentScore(const float seqsimthrscore) {
        VerifyAlignmentScoreKernel(
            seqsimthrscore, querypmbeg_, bdbCpmbeg_, wrkmemaux_);
    }

    virtual void Run() {
        FindFragKernelCaller();
        RefineFragInitKernelCaller(FRAGREF_NMAXCONVIT);
        DPRefine<true/*GAP0*/,true/*PRESCREEN*/,false/*WRKMEMTM1*/>(maxndpiters_, prescore_);
    }

protected:
    template<bool GAP0, bool PRESCREEN, bool WRKMEMTM1>
    void DPRefine(const int maxndpiters, const float prescorethr);

    void RefineFragInitKernelCaller(const int nmaxconvit) {
        if(CLOptions::GetC_SYMMETRIC())
            RefineFragInitKernel<true/*TFM_DINV*/>(
                nmaxconvit,
                querypmbeg_, bdbCpmbeg_, tmpdpdiagbuffers_,
                wrkmemtm_, wrkmemtmibest_, wrkmemaux_, tfmmem_);
        else
            RefineFragInitKernel<false/*TFM_DINV*/>(
                nmaxconvit,
                querypmbeg_, bdbCpmbeg_, tmpdpdiagbuffers_,
                wrkmemtm_, wrkmemtmibest_, wrkmemaux_, tfmmem_);
    }

    template<int SECONDARYUPDATE = SECONDARYUPDATE_NOUPDATE>
    void RefineFragDPKernelCaller(const bool readlocalconv, const int nmaxconvit) {
        if(CLOptions::GetC_SYMMETRIC())
            RefineFragDPKernel<true/*TFM_DINV*/,SECONDARYUPDATE>(
                readlocalconv, nmaxconvit,
                querypmbeg_, bdbCpmbeg_, tmpdpalnpossbuffer_, tmpdpdiagbuffers_,
                wrkmemtm_, wrkmemtmibest_, wrkmemaux_, tfmmem_);
        else
            RefineFragDPKernel<false/*TFM_DINV*/,SECONDARYUPDATE>(
                readlocalconv, nmaxconvit,
                querypmbeg_, bdbCpmbeg_, tmpdpalnpossbuffer_, tmpdpdiagbuffers_,
                wrkmemtm_, wrkmemtmibest_, wrkmemaux_, tfmmem_);
    }


    template<bool TFM_DINV>
    void RefineFragInitKernel(
        const int nmaxconvit,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        float* const __RESTRICT__ tmpdpdiagbuffers,
        float* const __RESTRICT__ wrkmemtm,
        float* const __RESTRICT__ wrkmemtmibest,
        float* const __RESTRICT__ wrkmemaux,
        float* const __RESTRICT__ tfmmem);

    template<bool TFM_DINV, int SECONDARYUPDATE>
    void RefineFragDPKernel(
        const bool readlocalconv,
        const int nmaxconvit,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const float* const __RESTRICT__ tmpdpalnpossbuffer,
        float* const __RESTRICT__ tmpdpdiagbuffers,
        float* const __RESTRICT__ wrkmemtm,
        float* const __RESTRICT__ wrkmemtmibest,
        float* const __RESTRICT__ wrkmemaux,
        float* const __RESTRICT__ tfmmem);

    template<int INITOPT>
    void InitScoresKernel(
        float* const __RESTRICT__ wrkmemaux);

    void CheckScoreConvergenceKernel(
        float* const __RESTRICT__ wrkmemaux);

    void SaveLastScore0Kernel(
        float* const __RESTRICT__ wrkmemaux);

    void SetLowScoreConvergenceFlagKernel(
        const float scorethld,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        float* const __RESTRICT__ wrkmemaux);

private:
    void Preinitialize1Kernel(
        const bool condition4filter1,
        float* const __RESTRICT__ wrkmemtmibest,
        float* const __RESTRICT__ tfmmem,
        float* const __RESTRICT__ alndatamem,
        float* const __RESTRICT__ wrkmemaux);

    void VerifyAlignmentScoreKernel(
        const float seqsimthrscore,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        float* const __RESTRICT__ wrkmemaux);

    void FindFragKernelCaller() {
        if(CLOptions::GetC_SYMMETRIC())
            FindFragKernel<true/*TFM_DINV*/>(
                querypmbeg_, bdbCpmbeg_, tmpdpdiagbuffers_, wrkmemaux_);
        else
            FindFragKernel<false/*TFM_DINV*/>(
                querypmbeg_, bdbCpmbeg_, tmpdpdiagbuffers_, wrkmemaux_);
    }

    template<bool TFM_DINV>
    void FindFragKernel(
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        float* const tmpdpdiagbuffers, float* const wrkmemaux);

protected:
    // {{---------------------------------------------
    template<int nEFFDS, int XDIM, int DATALN>
    void CalcCCMatrices_Complete(
        const int qrydst,
        const int dbstrdst,
        const int nalnposs,
        int qrypos, int rfnpos,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        float (* __RESTRICT__ ccm)[XDIM]);

    template<int nEFFDS, int XDIM, int DATALN>
    void CalcCCMatricesRefined_Complete(
        const int qrydst,
        const int dbstrdst,
        const int fraglen,
        int qrylen, int dbstrlen,
        int qrypos, int rfnpos,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        float (* __RESTRICT__ ccm)[XDIM]);
    // }}---------------------------------------------


    // {{---------------------------------------------
    template<int SAVEPOS, int XDIM, int DATALN>
    void CalcScoresUnrl_Complete(
        const int READCNST,
        const int qryndx,
        const int ndbCposs,
        const int maxnsteps,
        const int sfragfct,
        const int qrydst,
        const int dbstrdst,
        const int maxnalnposs,
        const int qrypos, const int rfnpos,
        const float d0, const float d02,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        float* __RESTRICT__ tmpdpdiagbuffers,
        const float* __RESTRICT__ tfm,
        float* __RESTRICT__ scv,
        float (* __RESTRICT__ dstv)[XDIM]);

    template<int XDIM, int DATALN>
    void CalcScoresUnrlRefined_Complete(
        const int READCNST,
        const int qryndx,
        const int ndbCposs,
        const int maxnsteps,
        const int sfragfctxndx,
        const int qrydst,
        const int dbstrdst,
        const int qrylen, const int dbstrlen,
        const int qrypos, const int rfnpos,
        const float d0, const float d02, const float d82,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        float* __RESTRICT__ tmpdpdiagbuffers,
        const float* __RESTRICT__ tfm,
        float* __RESTRICT__ scv,
        float (* __RESTRICT__ dstv)[XDIM]);
    // }}---------------------------------------------


    // {{---------------------------------------------
    template<int nEFFDS, int XDIM, int DATALN>
    void CalcCCMatricesExtended_Complete(
        const int qryndx,
        const int ndbCposs,
        const int maxnsteps,
        const int sfragfct,
        const int qrydst,
        const int dbstrdst,
        const int nalnposs,
        const int qrypos, const int rfnpos,
        const float dst32,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const float* __RESTRICT__ tmpdpdiagbuffers,
        float (* __RESTRICT__ ccm)[XDIM]);

    template<int nEFFDS, int XDIM, int DATALN>
    void CalcCCMatricesRefinedExtended_Complete(
        const int READCNST,
        const int qryndx,
        const int ndbCposs,
        const int maxnsteps,
        const int sfragfctxndx,
        const int qrydst,
        const int dbstrdst,
        const int qrylen, const int dbstrlen,
        const int qrypos, const int rfnpos,
        const float d0, const float dst32,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const float* __RESTRICT__ tmpdpdiagbuffers,
        float (* __RESTRICT__ ccm)[XDIM]);
    // }}---------------------------------------------


    // {{---------------------------------------------
    void SaveBestScoreAndPositions_Complete(
        float best,
        const int qryndx,
        const int dbstrndx,
        const int ndbCstrs,
        const int maxnsteps,
        const int sfragfct,
        const int qrypos, const int rfnpos,
        float* __RESTRICT__ wrkmemaux);
    // }}---------------------------------------------


    // {{---------------------------------------------
    template<int XDIM, int DATALN>
    void SaveBestScoreAmongBests(
        const int qryndx,
        const int rfnblkndx,
        const int ndbCstrs,
        const int maxnsteps,
        const int effnsteps,
        float (* __RESTRICT__ ccm)[XDIM],
        float* __RESTRICT__ wrkmemaux);
    // }}---------------------------------------------


    template<int XDIM, int DATALN>
    void UpdateCCMCache(
        int qrypos, int rfnpos,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        float (* __RESTRICT__ ccm)[XDIM], int pi);

    template<int XDIM, int DATALN>
    void UpdateCCMOneAlnPosExtended(
        float d02s,
        int qrypos, int rfnpos, int scrpos,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const float* __RESTRICT__ tmpdpdiagbuffers,
        float (* __RESTRICT__ ccm)[XDIM], int pi);

    template<int SAVEPOS, int CHCKDST, int DATALN>
    float UpdateOneAlnPosScore(
        float d02, float d82,
        int qrypos, int rfnpos, int scrpos,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const float* __RESTRICT__ tfm,
        float* __RESTRICT__ scv,
        float* __RESTRICT__ tmpdpdiagbuffers, int pi);

protected:
    const int maxndpiters_;
    const float prescore_;
    const int stepinit_;
};



// -------------------------------------------------------------------------
// INLINES ...
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// CalcCCMatrices_Complete: calculate cross-covariance matrix 
// between the query and reference structures for a given fragment;
// qrydst, distances in positions to the beginnings of the query structures;
// dbstrdst, distances in positions to the beginnings of the reference structures;
// nalnposs, #aligned positions (fragment length);
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// ccm, cache for the cross-covarinace matrix and related data;
// 
template<int nEFFDS, int XDIM, int DATALN>
inline
void MpStage1::CalcCCMatrices_Complete(
    const int qrydst,
    const int dbstrdst,
    const int nalnposs,
    int qrypos, int rfnpos,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    float (* __RESTRICT__ ccm)[XDIM])
{
    //initialize cache:
    for(int f = 0; f < nEFFDS; f++) {
        #pragma omp simd
        for(int pi = 0; pi < XDIM; pi++) ccm[f][pi] = 0.0f;
    }

    //manually unroll along data blocks:
    for(int ai = 0; ai < nalnposs; ai += XDIM)
    {
        int piend = mymin(XDIM, nalnposs - ai);
        #pragma omp simd
        for(int pi = 0; pi < piend; pi++)
            UpdateCCMCache<XDIM,DATALN>(
                qrydst + qrypos + ai + pi,
                dbstrdst + rfnpos + ai + pi,
                querypmbeg, bdbCpmbeg,
                ccm, pi);
    }

    //sum reduction for each field
    for(int f = 0; f < twmvEndOfCCData; f++) {
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for(int pi = 0; pi < XDIM; pi++) sum += ccm[f][pi];
        //write sum back to ccm
        ccm[f][0] = sum;
    }

    //write nalnposs
    ccm[twmvNalnposs][0] = nalnposs;
}

// -------------------------------------------------------------------------
// CalcCCMatricesRefined_Complete: calculate cross-covariance matrix 
// between the query and reference structures for refinement, i.e. 
// delineation of suboptimal fragment boundaries;
// qrydst, distances in positions to the beginnings of the query structures;
// dbstrdst, distances in positions to the beginnings of the reference structures;
// fraglen, fragment length;
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// ccm, cache for the cross-covarinace matrix and related data;
// 
template<int nEFFDS, int XDIM, int DATALN>
inline
void MpStage1::CalcCCMatricesRefined_Complete(
    const int qrydst,
    const int dbstrdst,
    const int fraglen,
    int qrylen, int dbstrlen,
    int qrypos, int rfnpos,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    float (* __RESTRICT__ ccm)[XDIM])
{
    //initialize cache:
    for(int f = 0; f < nEFFDS; f++) {
        #pragma omp simd
        for(int pi = 0; pi < XDIM; pi++) ccm[f][pi] = 0.0f;
    }

    //update positions and assign virtual query and reference lengths:
    UpdateLengths(qrylen, dbstrlen, qrypos, rfnpos, fraglen);

    //manually unroll along data blocks:
    for(int ai = 0; ai < fraglen; ai += XDIM)
    {
        int piend = mymin(XDIM, fraglen - ai);
        #pragma omp simd
        for(int pi = 0; pi < piend; pi++)
            UpdateCCMCache<XDIM,DATALN>(
                qrydst + qrypos + ai + pi,
                dbstrdst + rfnpos + ai + pi,
                querypmbeg, bdbCpmbeg,
                ccm, pi);
    }

    //sum reduction for each field
    for(int f = 0; f < twmvEndOfCCData; f++) {
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
// UpdateCCMCache: update cross-covariance cache data given query and 
// reference coordinates, respectively;
//
#if defined(OS_MS_WINDOWS)
#define OMPDECLARE_MpStage1_UpdateCCMCache
#else 
#pragma omp declare simd linear(pi,qrypos,rfnpos:1) \
  uniform(querypmbeg,bdbCpmbeg,ccm) \
  aligned(querypmbeg,bdbCpmbeg:DATALN) \
  notinbranch
#endif
template<int XDIM, int DATALN>
inline
void MpStage1::UpdateCCMCache(
    int qrypos, int rfnpos,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    float (* __RESTRICT__ ccm)[XDIM], int pi)
{
    float qx = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DX>(querypmbeg, qrypos);
    float qy = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DY>(querypmbeg, qrypos);
    float qz = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DZ>(querypmbeg, qrypos);

    float rx = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DX>(bdbCpmbeg, rfnpos);
    float ry = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DY>(bdbCpmbeg, rfnpos);
    float rz = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DZ>(bdbCpmbeg, rfnpos);

    UpdateCCMCacheHelper<XDIM>(qx, qy, qz,  rx, ry, rz,  ccm, pi);
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// CalcScoresUnrl_Complete: calculate/reduce UNNORMALIZED scores for 
// obtained superpositions; complete version for fragment identification; 
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// SAVEPOS, template parameter to request saving distances;
// qryndx, query serial number;
// ndbCposs, total number of reference positions in the chunk;
// maxnsteps, max number of steps to perform for each reference structure;
// sfragfct, current fragment factor;
// qrydst, distances in positions to the beginnings of the query structures;
// dbstrdst, distances in positions to the beginnings of the reference structures;
// maxnalnposs, #originally aligned positions (original fragment length);
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// d0, d02, distance thresholds;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary diagonal buffers reused here for saving positional scores/distances;
// tfmCache, cached transformation matrix;
// scvCache, cache for scores;
//
template<int SAVEPOS, int XDIM, int DATALN>
inline
void MpStage1::CalcScoresUnrl_Complete(
    const int READCNST,
    const int qryndx,
    const int ndbCposs,
    const int maxnsteps,
    const int sfragfct,
    const int qrydst,
    const int dbstrdst,
    const int maxnalnposs,
    const int qrypos, const int rfnpos,
    const float d0, const float d02,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    float* __RESTRICT__ tmpdpdiagbuffers,
    const float* __RESTRICT__ tfm,
    float* __RESTRICT__ scv,
    float (* __RESTRICT__ dstv)[XDIM])
{
    //initialize cache:
    #pragma omp simd
    for(int pi = 0; pi < XDIM; pi++) scv[pi] = 0.0f;

    //initialize cache of distances (dstv indices are 1-based):
    for(int f = 1; f <= 3; f++) {
        #pragma omp simd
        for(int pi = 0; pi < XDIM; pi++) dstv[f][pi] = CP_LARGEDST;
    }

    const int mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs;

    //manually unroll along data blocks:
    for(int ai = 0; ai < maxnalnposs; ai += XDIM)
    {
        int piend = mymin(XDIM, maxnalnposs - ai);
        #pragma omp simd
        for(int pi = 0; pi < piend; pi++) {
            //calculated distance is written to to gmem: SAVEPOS_SAVE
            float dst = 
            UpdateOneAlnPosScore<SAVEPOS,CHCKDST_NOCHECK,DATALN>(
                d02, d02,
                qrydst + qrypos + ai + pi,//query position
                dbstrdst + rfnpos + ai + pi,//reference position
                mloc + dbstrdst + ai + pi,//position for scores
                querypmbeg, bdbCpmbeg,//query, reference data
                tfm, scv, tmpdpdiagbuffers, pi);//tfm, scores, dsts written to gmem
            //store three min distance values
            if(SAVEPOS == SAVEPOS_SAVE)
                StoreMinDst<XDIM>(dst, dstv, pi);
        }
    }

    //sum reduction for scores
    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for(int pi = 0; pi < XDIM; pi++) sum += scv[pi];
    //write sum back to scv
    scv[0] = sum;

    float min1 = CP_LARGEDST, min2 = CP_LARGEDST, min3 = CP_LARGEDST;
    // int fm1 = 9, fm2 = 9, pim1 = XDIM, pim2 = XDIM;

    if(SAVEPOS == SAVEPOS_SAVE) {
        //3-fold min reduction for distances
        #pragma omp simd reduction(min:min1) collapse(2)
        for(int f = 1; f <= 3; f++)
            for(int pi = 0; pi < XDIM; pi++)
                if(dstv[f][pi] < min1) min1 = dstv[f][pi];
                // if(dstv[f][pi] < min1) {
                //     min1 = dstv[f][pi];
                //     fm1 = f; pim1 = pi;
                // }

        #pragma omp simd reduction(min:min2) collapse(2)
        for(int f = 1; f <= 3; f++)
            for(int pi = 0; pi < XDIM; pi++)
                if(min1 < dstv[f][pi] && dstv[f][pi] < min2) min2 = dstv[f][pi];
                // if((f != fm1 || pi != pim1) && dstv[f][pi] < min2) {
                //     min2 = dstv[f][pi];
                //     fm2 = f; pim2 = pi;
                // }

        #pragma omp simd reduction(min:min3) collapse(2)
        for(int f = 1; f <= 3; f++)
            for(int pi = 0; pi < XDIM; pi++)
                if(min2 < dstv[f][pi] && dstv[f][pi] < min3) min3 = dstv[f][pi];
                // if((f != fm1 || pi != pim1) && (f != fm2 || pi != pim2) && dstv[f][pi] < min3)
                //     min3 = dstv[f][pi];

        //write the minimum score that ensures at least 3 aligned positions:
        float d02s = GetD02s(d0);
        if(READCNST == READCNST_CALC2) d02s += D02s_PROC_INC;

        if(CP_LARGEDST_cmp < min3 || min3 < d02s || maxnalnposs <= 3)
            //max number of alignment positions (maxnalnposs) <3;
            //use the dfault threshold
            min3 = d02s;
        else {//round the 3rd minimum score to the next multiple of 0.5:
            //obtained from d02s + k*0.5 >= min3
            min3 = d02s + ceilf((min3 - d02s) * 2.0f) * 0.5f;
        }

        //write distance min3 to scv[1] as scv[0] is reserved for the score:
        scv[1] = min3;
    }
}

// -------------------------------------------------------------------------
// CalcScoresUnrlRefined_Complete: calculate/reduce UNNORMALIZED scores for 
// obtained superpositions; complete version for fragments refinement; 
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// qryndx, query serial number;
// ndbCposs, total number of reference positions in the chunk;
// maxnsteps, max number of steps to perform for each reference structure;
// sfragfctxndx, current fragment factor x fragment length index, which is <maxnsteps;
// qrydst, distances in positions to the beginnings of the query structures;
// dbstrdst, distances in positions to the beginnings of the reference structures;
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// d0, d02, d82, distance thresholds;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary diagonal buffers reused here for saving positional scores/distances;
// tfm, cached transformation matrix;
// scv, cache for scores;
//
template<int XDIM, int DATALN>
inline
void MpStage1::CalcScoresUnrlRefined_Complete(
    const int READCNST,
    const int qryndx,
    const int ndbCposs,
    const int maxnsteps,
    const int sfragfctxndx,
    const int qrydst,
    const int dbstrdst,
    const int qrylen, const int dbstrlen,
    const int qrypos, const int rfnpos,
    const float d0, const float d02, const float d82,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    float* __RESTRICT__ tmpdpdiagbuffers,
    const float* __RESTRICT__ tfm,
    float* __RESTRICT__ scv,
    float (* __RESTRICT__ dstv)[XDIM])
{
    //initialize cache:
    #pragma omp simd
    for(int pi = 0; pi < XDIM; pi++) scv[pi] = 0.0f;

    //initialize cache of distances (dstv indices are 1-based):
    for(int f = 1; f <= 3; f++) {
        #pragma omp simd
        for(int pi = 0; pi < XDIM; pi++) dstv[f][pi] = CP_LARGEDST;
    }

    const int maxnalnposs = mymin(qrylen - qrypos, dbstrlen - rfnpos);
    const int mloc = (qryndx * maxnsteps + sfragfctxndx) * ndbCposs;

    //manually unroll along data blocks:
    for(int ai = 0; ai < maxnalnposs; ai += XDIM)
    {
        int piend = mymin(XDIM, maxnalnposs - ai);
        #pragma omp simd
        for(int pi = 0; pi < piend; pi++) {
            //calculated distance is written to to gmem: SAVEPOS_SAVE
            float dst = 
            UpdateOneAlnPosScore<SAVEPOS_SAVE,CHCKDST_CHECK,DATALN>(
                d02, d82,
                qrydst + qrypos + ai + pi,//query position
                dbstrdst + rfnpos + ai + pi,//reference position
                mloc + dbstrdst + ai + pi,//position for scores
                querypmbeg, bdbCpmbeg,//query, reference data
                tfm, scv, tmpdpdiagbuffers, pi);//tfm, scores, dsts written to gmem
            //store three min distance values
#if (DO_FINDD02_DURING_REFINEFRAG == 0)
            if(READCNST == READCNST_CALC)
#endif
                StoreMinDst<XDIM>(dst, dstv, pi);
        }
    }

    //sum reduction for scores
    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for(int pi = 0; pi < XDIM; pi++) sum += scv[pi];
    //write sum back to scv
    scv[0] = sum;

    float min1 = CP_LARGEDST, min2 = CP_LARGEDST, min3 = CP_LARGEDST;
    int fm1 = 9, fm2 = 9, pim1 = XDIM, pim2 = XDIM;

#if (DO_FINDD02_DURING_REFINEFRAG == 0)
    if(READCNST == READCNST_CALC)
#endif
    {   //3-fold min reduction for distances
        #pragma omp simd reduction(min:min1) collapse(2)
        for(int f = 1; f <= 3; f++)
            for(int pi = 0; pi < XDIM; pi++)
                if(dstv[f][pi] < min1) {
                    min1 = dstv[f][pi];
                    fm1 = f; pim1 = pi;
                }

        #pragma omp simd reduction(min:min2) collapse(2)
        for(int f = 1; f <= 3; f++)
            for(int pi = 0; pi < XDIM; pi++)
                if((f != fm1 || pi != pim1) && dstv[f][pi] < min2) {
                    min2 = dstv[f][pi];
                    fm2 = f; pim2 = pi;
                }

        #pragma omp simd reduction(min:min3) collapse(2)
        for(int f = 1; f <= 3; f++)
            for(int pi = 0; pi < XDIM; pi++)
                if((f != fm1 || pi != pim1) && (f != fm2 || pi != pim2) && dstv[f][pi] < min3)
                    min3 = dstv[f][pi];

        //write the minimum score that ensures at least 3 aligned positions:
        float d0s = GetD0s(d0) + ((READCNST == READCNST_CALC2)? 1.0f: -1.0f);
        float d02s = SQRD(d0s);

        if(CP_LARGEDST_cmp < min3 || min3 < d02s || maxnalnposs <= 3)
            //max number of alignment positions (GetGplAlnLength) <3;
            //use the dfault threshold
            min3 = d02s;
        else {//round the 3rd minimum score according to the below:
            //obtained from (d0s + k*0.5)^2 >= min3 (squared distance)
            min3 = d0s + ceilf((sqrtf(min3) - d0s) * 2.0f) * 0.5f;
            min3 = SQRD(min3);
        }

        //write min3 to scv[1] as scv[0] is reserved for the score:
        scv[1] = min3;
    }
}

// -------------------------------------------------------------------------
// UpdateOneAlnPosScore: update score unconditionally for one alignment 
// position;
// SAVEPOS, template parameter to request saving positional scores;
// CHCKDST, template parameter to request accumulating scores within the 
// given threshold distance only;
// d02, d0 squared used for calculating score;
// d82, distance threshold for reducing scores;
// qrypos, starting query position;
// rfnpos, starting reference position;
// scrpos, position index to write the score obtained at the alignment 
// position;
// tfm, address of the transformation matrix;
// scv, address of the vector of scores;
// tmpdpdiagbuffers, global memory address for saving positional scores;
//
#if defined(OS_MS_WINDOWS)
#define OMPDECLARE_MpStage1_UpdateOneAlnPosScore
#else 
#pragma omp declare simd linear(pi,qrypos,rfnpos,scrpos:1) \
  uniform(d02,d82, querypmbeg,bdbCpmbeg,tfm,scv,tmpdpdiagbuffers) \
  aligned(querypmbeg,bdbCpmbeg,tmpdpdiagbuffers:DATALN) \
  notinbranch
#endif
template<int SAVEPOS, int CHCKDST, int DATALN>
inline
float MpStage1::UpdateOneAlnPosScore(
    float d02, float d82,
    int qrypos, int rfnpos, int scrpos,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* __RESTRICT__ tfm,
    float* __RESTRICT__ scv,
    float* __RESTRICT__ tmpdpdiagbuffers, int pi)
{
    float qx = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DX>(querypmbeg, qrypos);
    float qy = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DY>(querypmbeg, qrypos);
    float qz = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DZ>(querypmbeg, qrypos);

    float rx = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DX>(bdbCpmbeg, rfnpos);
    float ry = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DY>(bdbCpmbeg, rfnpos);
    float rz = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DZ>(bdbCpmbeg, rfnpos);

    float dst = transform_and_distance2(tfm, qx, qy, qz,  rx, ry, rz);

    constexpr int reduce = (CHCKDST == CHCKDST_CHECK)? 0: 1;

    if(reduce || dst <= d82)
        //calculate score
        scv[pi] += GetPairScore(d02, dst);

    if(SAVEPOS == SAVEPOS_SAVE)
        tmpdpdiagbuffers[scrpos] = dst;

    return dst;
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// CalcCCMatricesExtended_Complete: calculate cross-covariance 
// matrix between the query and reference structures based on aligned 
// positions within given distance;
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// qryndx, query serial number;
// ndbCposs, total number of reference positions in the chunk;
// maxnsteps, max number of steps to perform for each reference structure;
// sfragfct, current fragment factor <maxnsteps;
// qrydst, distances in positions to the beginnings of the query structures;
// dbstrdst, distances in positions to the beginnings of the reference structures;
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// dst32, squared distance threshold at which at least three aligned pairs are observed;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary diagonal buffers reused for reading positional scores;
// ccmCache, cache for the cross-covariance matrix and related data;
// 
template<int nEFFDS, int XDIM, int DATALN>
inline
void MpStage1::CalcCCMatricesExtended_Complete(
    const int qryndx,
    const int ndbCposs,
    const int maxnsteps,
    const int sfragfct,
    const int qrydst,
    const int dbstrdst,
    const int nalnposs,
    const int qrypos, const int rfnpos,
    const float dst32,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* __RESTRICT__ tmpdpdiagbuffers,
    float (* __RESTRICT__ ccm)[XDIM])
{
    //initialize cache:
    for(int f = 0; f < nEFFDS; f++) {
        #pragma omp simd
        for(int pi = 0; pi < XDIM; pi++) ccm[f][pi] = 0.0f;
    }

    const float d02s = dst32;

    const int mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs;

    //manually unroll along data blocks:
    for(int ai = 0; ai < nalnposs; ai += XDIM)
    {
        int piend = mymin(XDIM, nalnposs - ai);
        #pragma omp simd
        for(int pi = 0; pi < piend; pi++)
            UpdateCCMOneAlnPosExtended<XDIM,DATALN>(
                d02s,
                qrydst + qrypos + ai + pi,//query position
                dbstrdst + rfnpos + ai + pi,//reference position
                mloc + dbstrdst + ai + pi,//position for scores
                querypmbeg, bdbCpmbeg,//query, reference data
                tmpdpdiagbuffers,//scores/dsts
                ccm, pi);//reduction output
    }

    //sum reduction for each field
    for(int f = 0; f < nEFFDS; f++) {
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for(int pi = 0; pi < XDIM; pi++) sum += ccm[f][pi];
        //write sum back to ccm
        ccm[f][0] = sum;
    }
}

// -------------------------------------------------------------------------
// CalcCCMatricesRefinedExtended_Complete: calculate cross-covariance 
// matrix between the query and reference structures based on aligned 
// positions within given distance;
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// READCNST, flag indicating the stage of this local processing;
// qryndx, query serial number;
// ndbCposs, total number of reference positions in the chunk;
// maxnsteps, max number of steps to perform for each reference structure;
// sfragfctxndx, current fragment factor x fragment length index, which is <maxnsteps;
// qrydst, distances in positions to the beginnings of the query structures;
// dbstrdst, distances in positions to the beginnings of the reference structures;
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// d0, distance threshold;
// dst32, squared distance threshold at which at least three aligned pairs are observed;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary diagonal buffers reused for reading positional scores;
// ccm, cache for the cross-covariance matrix and related data;
// 
template<int nEFFDS, int XDIM, int DATALN>
inline
void MpStage1::CalcCCMatricesRefinedExtended_Complete(
    const int READCNST,
    const int qryndx,
    const int ndbCposs,
    const int maxnsteps,
    const int sfragfctxndx,
    const int qrydst,
    const int dbstrdst,
    const int qrylen, const int dbstrlen,
    const int qrypos, const int rfnpos,
    const float d0, const float dst32,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* __RESTRICT__ tmpdpdiagbuffers,
    float (* __RESTRICT__ ccm)[XDIM])
{
    //initialize cache:
    for(int f = 0; f < nEFFDS; f++) {
        #pragma omp simd
        for(int pi = 0; pi < XDIM; pi++) ccm[f][pi] = 0.0f;
    }

    float d02s = dst32;

#if (DO_FINDD02_DURING_REFINEFRAG == 0)
    if(READCNST != READCNST_CALC) {
        float d0s = GetD0s(d0) + 1.0f;
        d02s = SQRD(d0s);
    }
#endif

    const int nalnposs = mymin(qrylen - qrypos, dbstrlen - rfnpos);
    const int mloc = (qryndx * maxnsteps + sfragfctxndx) * ndbCposs;

    //manually unroll along data blocks:
    for(int ai = 0; ai < nalnposs; ai += XDIM)
    {
        int piend = mymin(XDIM, nalnposs - ai);
        #pragma omp simd
        for(int pi = 0; pi < piend; pi++)
            UpdateCCMOneAlnPosExtended<XDIM,DATALN>(
                d02s,
                qrydst + qrypos + ai + pi,//query position
                dbstrdst + rfnpos + ai + pi,//reference position
                mloc + dbstrdst + ai + pi,//position for scores
                querypmbeg, bdbCpmbeg,//query, reference data
                tmpdpdiagbuffers,//scores/dsts
                ccm, pi);//reduction output
    }

    //sum reduction for each field
    for(int f = 0; f < nEFFDS; f++) {
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for(int pi = 0; pi < XDIM; pi++) sum += ccm[f][pi];
        //write sum back to ccm
        ccm[f][0] = sum;
    }
}

// -------------------------------------------------------------------------
// UpdateCCMOneAlnPosExtended: update one position contributing to the 
// cross-covariance matrix between the query and reference structures 
// only if transformed query is within the given distance from reference;
// XDIM, template parameter: inner-most dimensions of the cache matrix;
// d02s, d0 squared used for the inclusion of pairs in the alignment;
// scrpos, position index to read the score obtained at the alignment 
// position;
//
#if defined(OS_MS_WINDOWS)
#define OMPDECLARE_MpStage1_UpdateCCMOneAlnPosExtended
#else 
#pragma omp declare simd linear(pi,qrypos,rfnpos,scrpos:1) \
  uniform(d02s, querypmbeg,bdbCpmbeg,tmpdpdiagbuffers,ccm) \
  aligned(querypmbeg,bdbCpmbeg,tmpdpdiagbuffers:DATALN) \
  notinbranch
#endif
template<int XDIM, int DATALN>
inline
void MpStage1::UpdateCCMOneAlnPosExtended(
    float d02s,
    int qrypos, int rfnpos, int scrpos,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* __RESTRICT__ tmpdpdiagbuffers,
    float (* __RESTRICT__ ccm)[XDIM], int pi)
{
    float dst = tmpdpdiagbuffers[scrpos];

    //distant positions do not contribute to cross-covariance:
    if(d02s < dst) return;

    float qx = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DX>(querypmbeg, qrypos);
    float qy = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DY>(querypmbeg, qrypos);
    float qz = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DZ>(querypmbeg, qrypos);

    float rx = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DX>(bdbCpmbeg, rfnpos);
    float ry = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DY>(bdbCpmbeg, rfnpos);
    float rz = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DZ>(bdbCpmbeg, rfnpos);

    UpdateCCMCacheHelper<XDIM>(qx, qy, qz,  rx, ry, rz,  ccm, pi);

    //update the number of positions
    ccm[twmvNalnposs][pi] += 1.0f;
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// SaveBestScoreAndPositions_Complete: complete version of saving the best 
// score along with query and reference positions;
// best, best score obtained by the thread block;
// qryndx, query serial number;
// dbstrndx, reference serial number;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps to perform for each reference structure;
// sfragfct, current fragment factor <maxnsteps;
// qrypos, rfnpos, starting query and reference positions;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory (includes the section of scores);
// 
inline
void MpStage1::SaveBestScoreAndPositions_Complete(
    float best,
    const int qryndx,
    const int dbstrndx,
    const int ndbCstrs,
    const int maxnsteps,
    const int sfragfct,
    const int qrypos, const int rfnpos,
    float* __RESTRICT__ wrkmemaux)
{
    if(best <= 0.0f) return;

    //save best score
    const int mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
    float currentbest = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx];
    if(currentbest < best) {
        wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx] = best;
        wrkmemaux[mloc + tawmvQRYpos * ndbCstrs + dbstrndx] = qrypos;
        wrkmemaux[mloc + tawmvRFNpos * ndbCstrs + dbstrndx] = rfnpos;
    }
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// SaveBestScoreAmongBests: save best score along with query and reference 
// positions by considering all partial best scores calculated over all 
// fragment factors; write it to the location of fragment factor 0;
// qryndx, query index;
// rfnblkndx, reference block index;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// wrkmemaux, auxiliary working memory;
// 
template<int XDIM, int DATALN>
inline
void MpStage1::SaveBestScoreAmongBests(
    const int qryndx,
    const int rfnblkndx,
    const int ndbCstrs,
    const int maxnsteps,
    const int effnsteps,
    float (* __RESTRICT__ ccm)[XDIM],
    float* __RESTRICT__ wrkmemaux)
{
    const int istr0 = rfnblkndx * XDIM;
    const int istre = mymin(istr0 + XDIM, (int)ndbCstrs);

    //initialize cache for best scores and their indices:
    #pragma omp simd collapse(2)
    for(int f = 0; f < 2; f++)
        for(int ii = 0; ii < XDIM; ii++)
            ccm[f][ii] = 0.0f;

    for(int si = 0; si < effnsteps; si++) {
        int mloc = ((qryndx * maxnsteps + si/*sfragfct*/) * nTAuxWorkingMemoryVars) * ndbCstrs;
        #pragma omp simd aligned(wrkmemaux:DATALN)
        for(int ri = istr0; ri < istre; ri++) {
            int ii = ri - istr0;
            float bscore = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + ri/*dbstrndx*/];
            //reuse cache for scores:
            if(ccm[0][ii] < bscore) {
                ccm[0][ii] = bscore;
                ccm[1][ii] = si;
            }
        }
    }

    //ccm[0][...] contains maximums; write max values to slot 0
    #pragma omp simd aligned(wrkmemaux:DATALN)
    for(int ri = istr0; ri < istre; ri++) {
        int ii = ri - istr0;
        int si = ccm[1][ii];
        float bscore = ccm[0][ii];
        int mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        int mloc = ((qryndx * maxnsteps + si) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(si != 0) {
            float qrypos = wrkmemaux[mloc + tawmvQRYpos * ndbCstrs + ri/*dbstrndx*/];
            float rfnpos = wrkmemaux[mloc + tawmvRFNpos * ndbCstrs + ri/*dbstrndx*/];
            //coalesced WRITE for multiple references
            wrkmemaux[mloc0 + tawmvBestScore * ndbCstrs + ri] = bscore;
            wrkmemaux[mloc0 + tawmvQRYpos * ndbCstrs + ri] = qrypos;
            wrkmemaux[mloc0 + tawmvRFNpos * ndbCstrs + ri] = rfnpos;
        }
        wrkmemaux[mloc0 + tawmvInitialBest * ndbCstrs + ri] = bscore;
    }
}

// -------------------------------------------------------------------------

#endif//__MpStage1_h__
