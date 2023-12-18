/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __MpStageBase_h__
#define __MpStageBase_h__

#include "libutil/mybase.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmymp/mpproc/mpprocconfbase.h"
#include "libmymp/mpstages/transformbase.h"
#include "libmymp/mpstages/scoringbase.h"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/fragment.cuh"
#include "libmycu/cucom/cudef.h"

//NOTE: conditional fragment index reduces the effect of fluctuating
//NOTE: superposition scores which result when multiple fragment-based 
//NOTE: semi-optimized superpositions produce same scores!
// #define MPSTAGEBASE_CONDITIONAL_FRAGNDX

// -------------------------------------------------------------------------
// base class MpStageBase for the stages of structure comparison
//
class MpStageBase {
public:
    MpStageBase(
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
        float* wrkmemaux, float* wrkmem2, float* alndatamem, float* tfmmem,
        uint* globvarsbuf)
    :
        maxnsteps_(maxnsteps), minfraglen_(minfraglen),
        querypmbeg_(querypmbeg), querypmend_(querypmend),
        bdbCpmbeg_(bdbCpmbeg), bdbCpmend_(bdbCpmend),
        nqystrs_(nqystrs), ndbCstrs_(ndbCstrs),
        nqyposs_(nqyposs), ndbCposs_(ndbCposs),
        qystr1len_(qystr1len), dbstr1len_(dbstr1len),
        qystrnlen_(qystrnlen), dbstrnlen_(dbstrnlen),
        dbxpad_(dbxpad),
        scores_(scores),
        tmpdpdiagbuffers_(tmpdpdiagbuffers), tmpdpbotbuffer_(tmpdpbotbuffer),
        tmpdpalnpossbuffer_(tmpdpalnpossbuffer),
        maxscoordsbuf_(maxscoordsbuf), btckdata_(btckdata),
        wrkmem_(wrkmem), wrkmemccd_(wrkmemccd), wrkmemtm_(wrkmemtm), wrkmemtmibest_(wrkmemtmibest),
        wrkmemaux_(wrkmemaux), wrkmem2_(wrkmem2), alndatamem_(alndatamem), tfmmem_(tfmmem),
        globvarsbuf_(globvarsbuf)
    {}

    virtual void Run() = 0;

protected:
    template<int nEFFDS, int XDIM, int DATALN>
    void CalcCCMatrices_DPRefined_Complete(
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


    template<int nEFFDS, int XDIM, int DATALN>
    void CalcCCMatrices_DPRefinedExtended_Complete(
        const int READCNST,
        const int qryndx,
        const int ndbCposs,
        const int dbxpad,
        const int maxnsteps,
        const int sfragfctxndx,
        const int dbstrdst,
        const int qrylen, const int dbstrlen,
        const int qrypos, const int rfnpos,
        const float d0, const float dst32,
        const float* const __RESTRICT__ tmpdpdiagbuffers,
        const float* const __RESTRICT__ tmpdpalnpossbuffer,
        float (* __RESTRICT__ ccm)[XDIM]);


    template<int XDIM, int DATALN, int CHCKDST>
    void CalcScoresUnrl_DPRefined_Complete(
        const int READCNST,
        const int qryndx,
        const int ndbCposs,
        const int dbxpad,
        const int maxnsteps,
        const int sfragfctxndx,
        const int dbstrdst,
        const int qrylen, const int dbstrlen,
        const int qrypos, const int rfnpos,
        const float d0, const float d02, const float d82,
        float* const __RESTRICT__ tmpdpdiagbuffers,
        const float* const __RESTRICT__ tmpdpalnpossbuffer,
        const float* __RESTRICT__ tfm,
        float* __RESTRICT__ scv,
        float (* __RESTRICT__ dstv)[XDIM]);


    template<bool WRITEFRAGINFO, bool CONDITIONAL>
    void SaveBestScoreAndTM_Complete(
        const float best,
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


    template<
        int XDIM, int DATALN,
        bool WRITEFRAGINFO,
        bool GRANDUPDATE,
        bool FORCEWRITEFRAGINFO,
        int SECONDARYUPDATE>
    void SaveBestScoreAndTMAmongBests(
        const int qryndx,
        const int rfnblkndx,
        const int ndbCstrs,
        const int maxnsteps,
        const int effnsteps,
        float (* __RESTRICT__ ccm)[XDIM],
        float* __RESTRICT__ wrkmemtmibest,
        float* __RESTRICT__ tfmmem,
        float* __RESTRICT__ wrkmemaux,
        float* __RESTRICT__ wrkmemtmibest2nd);

protected:
    template<int nEFFDS, int XDIM>
    void CheckConvergenceRefined_Complete(
        const float (* __RESTRICT__ ccm)[XDIM],
        float* __RESTRICT__ ccmLast);


    template<int XDIM, int DATALN>
    void UpdateCCMOneAlnPos_DPRefined(
        int pos, const int dblen,
        const float* const __RESTRICT__ tmpdpalnpossbuffer,
        float (* __RESTRICT__ ccm)[XDIM], int pi);

    template<int XDIM, int DATALN>
    void UpdateCCMOneAlnPos_DPExtended(
        float d02s,
        int pos, const int dblen, int scrpos,
        const float* const __RESTRICT__ tmpdpalnpossbuffer,
        const float* const __RESTRICT__ tmpdpdiagbuffers,
        float (* __RESTRICT__ ccm)[XDIM], int pi);

    template<int XDIM>
    void UpdateCCMCacheHelper(
        const float qx, const float qy, const float qz,
        const float rx, const float ry, const float rz,
        float (* __RESTRICT__ ccm)[XDIM], int pi);


    template<int SAVEPOS, int CHCKDST, int DATALN>
    float UpdateOneAlnPosScore_DPRefined(
        float d02, float d82,
        int pos, int dblen, int scrpos,
        const float* const __RESTRICT__ tmpdpalnpossbuffer,
        const float* __RESTRICT__ tfm,
        float* __RESTRICT__ scv,
        float* const __RESTRICT__ tmpdpdiagbuffers, int pi);

    template<int XDIM>
    void StoreMinDst(
        float dst, float (* __RESTRICT__ dstv)[XDIM], int pi);


    template<bool DOUBLY_INVERTED = false>
    void CalcTfmMatrices_Complete(
        int qrylen, int dbstrlen, float* ccm);

    void CalcTfmMatrices_DynamicOrientation_Complete(
        int qrylen, int dbstrlen, float* ccm);

    void CalcTfmMatricesHelper_Complete(float* ccm, float nalnposs);

protected:
    const uint maxnsteps_;
    const uint minfraglen_;
    char* const * const querypmbeg_, * const * const querypmend_;
    char* const * const bdbCpmbeg_, * const *const bdbCpmend_;
    const uint nqystrs_, ndbCstrs_;
    const uint nqyposs_, ndbCposs_;
    const uint qystr1len_, dbstr1len_;
    const uint qystrnlen_, dbstrnlen_;
    const uint dbxpad_;
    float* const scores_;
    float* const tmpdpdiagbuffers_, *const tmpdpbotbuffer_, *const tmpdpalnpossbuffer_;
    uint* const maxscoordsbuf_;
    char* const btckdata_;
    float* const wrkmem_, *const wrkmemccd_, *const wrkmemtm_, *const wrkmemtmibest_;
    float* const wrkmemaux_, *const wrkmem2_, *const alndatamem_, *const tfmmem_;
    uint* const globvarsbuf_;
};



// -------------------------------------------------------------------------
// INLINES ...
// -------------------------------------------------------------------------
// CheckConvergenceRefined_Complete: check whether calculating rotation 
// matrices converged by verifying the absolute difference of two latest 
// cross-covariance matrix data between the query and reference structures;
//
template<int nEFFDS, int XDIM>
inline
void MpStageBase::CheckConvergenceRefined_Complete(
    const float (* __RESTRICT__ ccm)[XDIM],
    float* __RESTRICT__ ccmLast)
{
    //effective number of fields (16):
    enum {neffds = twmvEndOfCCDataExt};
    int fldval = 0;

    #pragma omp simd reduction(+:fldval)
    for(int f = 0; f < nEFFDS; f++) {
        float dat1 = ccm[f][0];
        float dat2 = ccmLast[f];
        //convergence criterion for all fields: |a-b| / min{|a|,|b|} < epsilon
        if(fabsf(dat1-dat2) < mymin(fabsf(dat1), fabsf(dat2)) * RM_CONVEPSILON)
            fldval++;
    }

    //write the convergence flag back to ccmLast:
    ccmLast[0] = 0.0f;
    if(nEFFDS <= fldval) ccmLast[0] = 1.0f;
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// CalcCCMatrices_DPRefined_Complete: calculate cross-covariance matrix 
// between the query and reference structures for refinement, i.e. 
// delineation of suboptimal fragment boundaries;
// refinement of fragment boundaries obtained as a result of the application of DP;
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
template<int nEFFDS, int XDIM, int DATALN>
inline
void MpStageBase::CalcCCMatrices_DPRefined_Complete(
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
    for(int f = 0; f < nEFFDS; f++) {
        #pragma omp simd
        for(int pi = 0; pi < XDIM; pi++) ccm[f][pi] = 0.0f;
    }

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
            UpdateCCMOneAlnPos_DPRefined<XDIM,DATALN>(
                // yofff + dbstrdst + qrylen-1 - (rfnpos + ai + pi), dblen,
                //NOTE: to make address increment is 1:
                yofff + dbstrdst + qrylen-1 - (rfnpos + ai + piend-1) + pi, dblen,
                tmpdpalnpossbuffer,
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
// CalcCCMatrices_DPRefinedExtended_Complete: calculate cross-covariance 
// matrix between the query and reference structures based on aligned 
// positions within given distance;
// refinement of fragment boundaries and superposition obtained as a result of DP application;
// READCNST, flag indicating the stage of this local processing;
// qryndx, query serial number;
// ndbCposs, total number of reference positions in the chunk;
// dbxpad, #pad positions along the dimension of reference structures;
// maxnsteps, max number of steps to perform for each reference structure;
// sfragfctxndx, current fragment factor x fragment length index, which is <maxnsteps;
// dbstrdst, distances in positions to the beginnings of the reference structures;
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// d0, distance threshold;
// dst32, squared distance threshold at which at least three aligned pairs are observed;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary diagonal buffers reused for reading positional scores;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// ccm, cache for the cross-covariance matrix and related data;
// 
template<int nEFFDS, int XDIM, int DATALN>
inline
void MpStageBase::CalcCCMatrices_DPRefinedExtended_Complete(
    const int READCNST,
    const int qryndx,
    const int ndbCposs,
    const int dbxpad,
    const int maxnsteps,
    const int sfragfctxndx,
    const int dbstrdst,
    const int qrylen, const int dbstrlen,
    const int qrypos, const int rfnpos,
    const float d0, const float dst32,
    const float* const __RESTRICT__ tmpdpdiagbuffers,
    const float* const __RESTRICT__ tmpdpalnpossbuffer,
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

    const int dblen = ndbCposs + dbxpad;
    //offset to the beginning of the data along the y axis wrt query qryndx: 
    const int yofff = (qryndx * maxnsteps + 0/*sfragfctxndx*/) * dblen * nTDPAlignedPoss;
    const int mloc = (qryndx * maxnsteps + sfragfctxndx) * ndbCposs;
    const int nalnposs = mymin(qrylen - qrypos, dbstrlen - rfnpos);

    //manually unroll along data blocks:
    for(int ai = 0; ai < nalnposs; ai += XDIM)
    {
        //starting position in tmpdpalnpossbuffer for a pair:
        //NOTE: aligned coordinates in tmpdpalnpossbuffer are in the reverse order!
        //NOTE: qrypos == rfnpos as well as qrylen == dbstrlen here
        int piend = mymin(XDIM, nalnposs - ai);
        #pragma omp simd
        for(int pi = 0; pi < piend; pi++)
            UpdateCCMOneAlnPos_DPExtended<XDIM,DATALN>(
                d02s,//tmpdpalnpossbuffer position next
                // yofff + dbstrdst + dbstrlen-1 - (rfnpos + ai + pi), dblen,
                //NOTE: make position increment is 1
                yofff + dbstrdst + dbstrlen-1 - (rfnpos + ai + piend-1) + pi, dblen,
                mloc + dbstrdst + ai + pi,//position for scores
                tmpdpalnpossbuffer,//coordinates
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
// UpdateCCMOneAlnPos_DPRefined: update one position of the alignment 
// obtained by DP, contributing to the cross-covariance matrix between the 
// query and reference structures;
// XDIM, template parameter: inner-most dimensions of the cache matrix;
// pos, position index to read alignment coordinates;
//
#if defined(OS_MS_WINDOWS)
#define OMPDECLARE_MpStageBase_UpdateCCMOneAlnPos_DPRefined
#else 
#pragma omp declare simd linear(pi,pos:1) \
  uniform(dblen, tmpdpalnpossbuffer,ccm) \
  aligned(tmpdpalnpossbuffer:DATALN) \
  notinbranch
#endif
template<int XDIM, int DATALN>
inline
void MpStageBase::UpdateCCMOneAlnPos_DPRefined(
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
}

// -------------------------------------------------------------------------
// UpdateCCMOneAlnPos_DPExtended: update one position contributing to the 
// cross-covariance matrix between the query and reference structures 
// only if transformed query is within the given distance from reference;
// XDIM, template parameter: inner-most dimensions of the cache matrix;
// d02s, d0 squared used for the inclusion of pairs in the alignment;
// pos, position in alignment buffer tmpdpalnpossbuffer;
// dblen, step (db length) by which coordinates of different dimension 
// written in tmpdpalnpossbuffer;
// scrpos, position index to read the score obtained at the alignment 
// position;
//
#if defined(OS_MS_WINDOWS)
#define OMPDECLARE_MpStageBase_UpdateCCMOneAlnPos_DPExtended
#else 
#pragma omp declare simd linear(pi,pos,scrpos:1) \
  uniform(d02s,dblen, tmpdpalnpossbuffer,tmpdpdiagbuffers,ccm) \
  aligned(tmpdpalnpossbuffer,tmpdpdiagbuffers:DATALN) \
  notinbranch
#endif
template<int XDIM, int DATALN>
inline
void MpStageBase::UpdateCCMOneAlnPos_DPExtended(
    float d02s,
    int pos, const int dblen, int scrpos,
    const float* const __RESTRICT__ tmpdpalnpossbuffer,
    const float* const __RESTRICT__ tmpdpdiagbuffers,
    float (* __RESTRICT__ ccm)[XDIM], int pi)
{
    float dst = tmpdpdiagbuffers[scrpos];

    //distant positions do not contribute to cross-covariance:
    if(d02s < dst) return;

    float qx = tmpdpalnpossbuffer[pos + dpapsQRYx * dblen];
    float qy = tmpdpalnpossbuffer[pos + dpapsQRYy * dblen];
    float qz = tmpdpalnpossbuffer[pos + dpapsQRYz * dblen];

    float rx = tmpdpalnpossbuffer[pos + dpapsRFNx * dblen];
    float ry = tmpdpalnpossbuffer[pos + dpapsRFNy * dblen];
    float rz = tmpdpalnpossbuffer[pos + dpapsRFNz * dblen];

    // UpdateCCMCacheExtended<SMIDIM>(ccmCache,  qx, qy, qz,  rx, ry, rz);
    UpdateCCMCacheHelper<XDIM>(qx, qy, qz,  rx, ry, rz,  ccm, pi);

    //update the number of positions
    ccm[twmvNalnposs][pi] += 1.0f;
}

// -------------------------------------------------------------------------
// UpdateCCMCache: update cross-covariance cache data given query and 
// reference coordinates, respectively;
//
#if defined(OS_MS_WINDOWS)
#define OMPDECLARE_MpStageBase_UpdateCCMCacheHelper
#else 
#pragma omp declare simd linear(pi:1) uniform(ccm) notinbranch
#endif
template<int XDIM>
inline
void MpStageBase::UpdateCCMCacheHelper(
    const float qx, const float qy, const float qz,
    const float rx, const float ry, const float rz,
    float (* __RESTRICT__ ccm)[XDIM], int pi)
{
    ccm[twmvCCM_0_0][pi] += qx * rx;
    ccm[twmvCCM_0_1][pi] += qx * ry;
    ccm[twmvCCM_0_2][pi] += qx * rz;

    ccm[twmvCCM_1_0][pi] += qy * rx;
    ccm[twmvCCM_1_1][pi] += qy * ry;
    ccm[twmvCCM_1_2][pi] += qy * rz;

    ccm[twmvCCM_2_0][pi] += qz * rx;
    ccm[twmvCCM_2_1][pi] += qz * ry;
    ccm[twmvCCM_2_2][pi] += qz * rz;

    ccm[twmvCVq_0][pi] += qx;
    ccm[twmvCVq_1][pi] += qy;
    ccm[twmvCVq_2][pi] += qz;

    ccm[twmvCVr_0][pi] += rx;
    ccm[twmvCVr_1][pi] += ry;
    ccm[twmvCVr_2][pi] += rz;
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// CalcScoresUnrl_DPRefined_Complete: calculate/reduce UNNORMALIZED scores for 
// obtained superpositions; complete version for the refinement of fragments 
// obtained by DP; 
// CHCKDST, template parameter to request accumulating scores within the 
// given threshold distance only;
// qryndx, query serial number;
// ndbCposs, total number of reference positions in the chunk;
// dbxpad, #pad positions along the dimension of reference structures;
// maxnsteps, max number of steps to perform for each reference structure;
// sfragfctxndx, current fragment factor x fragment length index, which is <maxnsteps;
// dbstrdst, distances in positions to the beginnings of the reference structures;
// qrylen, dbstrlen, query and reference lengths;
// qrylenorg, dbstrlenorg, original query and reference lengths;
// qrypos, rfnpos, starting query and reference positions;
// d0, d02, d82, distance thresholds;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary diagonal buffers reused here for saving positional scores/distances;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// tfm, cached transformation matrix;
// scv, cache for scores;
//
template<int XDIM, int DATALN, int CHCKDST>
inline
void MpStageBase::CalcScoresUnrl_DPRefined_Complete(
    const int READCNST,
    const int qryndx,
    const int ndbCposs,
    const int dbxpad,
    const int maxnsteps,
    const int sfragfctxndx,
    const int dbstrdst,
    const int qrylen, const int dbstrlen,
    const int qrypos, const int rfnpos,
    const float d0, const float d02, const float d82,
    float* const __RESTRICT__ tmpdpdiagbuffers,
    const float* const __RESTRICT__ tmpdpalnpossbuffer,
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

    const int dblen = ndbCposs + dbxpad;
    //offset to the beginning of the data along the y axis wrt query qryndx: 
    const int yofff = (qryndx * maxnsteps + 0/*sfragfctxndx*/) * dblen * nTDPAlignedPoss;
    const int mloc = (qryndx * maxnsteps + sfragfctxndx) * ndbCposs;
    const int maxnalnposs = mymin(qrylen - qrypos, dbstrlen - rfnpos);

    //manually unroll along data blocks:
    for(int ai = 0; ai < maxnalnposs; ai += XDIM)
    {
        //NOTE: aligned coordinates in tmpdpalnpossbuffer are in the reverse order!
        //NOTE: qrypos == rfnpos as well as qrylen == dbstrlen here
        int piend = mymin(XDIM, maxnalnposs - ai);
        #pragma omp simd
        for(int pi = 0; pi < piend; pi++) {
            //calculated distance is written to to gmem: SAVEPOS_SAVE
            float dst = 
            UpdateOneAlnPosScore_DPRefined<SAVEPOS_SAVE,CHCKDST,DATALN>(
                d02, d82,
                // yofff + dbstrdst + dbstrlen-1 - (rfnpos + ai + pi), dblen,
                //NOTE: make position increment is 1
                yofff + dbstrdst + dbstrlen-1 - (rfnpos + ai + piend-1) + pi, dblen,
                mloc + dbstrdst + ai + pi,//position for scores
                tmpdpalnpossbuffer,//coordinates
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
// UpdateOneAlnPosScore_DPRefined: update score unconditionally for one 
// position of the alignment obtained by DP;
// SAVEPOS, template parameter to request saving positional scores;
// CHCKDST, template parameter to request accumulating scores within the 
// given threshold distance only;
// d02, d0 squared used for calculating score;
// d82, distance threshold for reducing scores;
// pos, position in alignment buffer tmpdpalnpossbuffer;
// dblen, step (db length) by which coordinates of different dimension 
// written in tmpdpalnpossbuffer;
// scrpos, position index to write the score/dst obtained at the alignment 
// position;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// tfm, address of the transformation matrix;
// scv, address of the vector of scores;
// tmpdpdiagbuffers, global memory address for saving positional scores;
//
#if defined(OS_MS_WINDOWS)
#define OMPDECLARE_MpStageBase_UpdateOneAlnPosScore_DPRefined
#else 
#pragma omp declare simd linear(pi,pos,scrpos:1) \
  uniform(d02,d82,dblen, tmpdpalnpossbuffer,tfm,scv,tmpdpdiagbuffers) \
  aligned(tmpdpalnpossbuffer,tmpdpdiagbuffers:DATALN) \
  notinbranch
#endif
template<int SAVEPOS, int CHCKDST, int DATALN>
inline
float MpStageBase::UpdateOneAlnPosScore_DPRefined(
    float d02, float d82,
    int pos, int dblen, int scrpos,
    const float* const __RESTRICT__ tmpdpalnpossbuffer,
    const float* __RESTRICT__ tfm,
    float* __RESTRICT__ scv,
    float* const __RESTRICT__ tmpdpdiagbuffers, int pi)
{
    float qx = tmpdpalnpossbuffer[pos + dpapsQRYx * dblen];
    float qy = tmpdpalnpossbuffer[pos + dpapsQRYy * dblen];
    float qz = tmpdpalnpossbuffer[pos + dpapsQRYz * dblen];

    float rx = tmpdpalnpossbuffer[pos + dpapsRFNx * dblen];
    float ry = tmpdpalnpossbuffer[pos + dpapsRFNy * dblen];
    float rz = tmpdpalnpossbuffer[pos + dpapsRFNz * dblen];

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
// StoreMinDst: store minimum distances in three cache buffers
// NOTE: dstv indices are 1-based
//
#if defined(OS_MS_WINDOWS)
#define OMPDECLARE_MpStageBase_StoreMinDst
#else 
#pragma omp declare simd linear(pi:1) \
    uniform(dstv) \
    notinbranch
#endif
template<int XDIM>
inline
void MpStageBase::StoreMinDst(
    float dst, float (* __RESTRICT__ dstv)[XDIM], int pi)
{
    if(dst < dstv[1][pi]) {
        dstv[3][pi] = dstv[2][pi];
        dstv[2][pi] = dstv[1][pi];
        dstv[1][pi] = dst;
    } else if(dst < dstv[2][pi]) {
        dstv[3][pi] = dstv[2][pi];
        dstv[2][pi] = dst;
    } else if(dst < dstv[3][pi])
        dstv[3][pi] = dst;
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// CalcTfmMatricesHelper_Complete: thread block calculates a tranformation 
// matrix;
// ccm, cache for the cross-covariance matrix and reuse;
// NOTE: Based on the original Kabsch algorithm:
/*
c**** CALCULATES A BEST ROTATION & TRANSLATION BETWEEN TWO VECTOR SETS
c**** SUCH THAT U*X+T IS THE CLOSEST APPROXIMATION TO Y.
c**** THE CALCULATED BEST SUPERPOSITION MAY NOT BE UNIQUE AS INDICATED
c**** BY A RESULT VALUE IER=-1. HOWEVER IT IS GARANTIED THAT WITHIN
c**** NUMERICAL TOLERANCES NO OTHER SUPERPOSITION EXISTS GIVING A
c**** SMALLER VALUE FOR RMS.
c**** THIS VERSION OF THE ALGORITHM IS OPTIMIZED FOR THREE-DIMENSIONAL
c**** REAL VECTOR SPACE.
c**** USE OF THIS ROUTINE IS RESTRICTED TO NON-PROFIT ACADEMIC
c**** APPLICATIONS.
c**** PLEASE REPORT ERRORS TO
c**** PROGRAMMER:  W.KABSCH   MAX-PLANCK-INSTITUTE FOR MEDICAL RESEARCH
c        JAHNSTRASSE 29, 6900 HEIDELBERG, FRG.
c**** REFERENCES:  W.KABSCH   ACTA CRYST.(1978).A34,827-828
c           W.KABSCH ACTA CRYST.(1976).A32,922-923
c
c  W    - W(M) IS WEIGHT FOR ATOM PAIR  # M           (GIVEN)
c  X    - X(I,M) ARE COORDINATES OF ATOM # M IN SET X       (GIVEN)
c  Y    - Y(I,M) ARE COORDINATES OF ATOM # M IN SET Y       (GIVEN)
c  N    - N IS number OF ATOM PAIRS             (GIVEN)
c  MODE  - 0:CALCULATE RMS ONLY              (GIVEN)
c      1:CALCULATE RMS,U,T   (TAKES LONGER)
c  RMS   - SUM OF W*(UX+T-Y)**2 OVER ALL ATOM PAIRS        (RESULT)
c  U    - U(I,J) IS   ROTATION  MATRIX FOR BEST SUPERPOSITION  (RESULT)
c  T    - T(I)   IS TRANSLATION VECTOR FOR BEST SUPERPOSITION  (RESULT)
c  IER   - 0: A UNIQUE OPTIMAL SUPERPOSITION HAS BEEN DETERMINED(RESULT)
c     -1: SUPERPOSITION IS NOT UNIQUE BUT OPTIMAL
c     -2: NO RESULT OBTAINED BECAUSE OF NEGATIVE WEIGHTS W
c      OR ALL WEIGHTS EQUAL TO ZERO.
c
c-----------------------------------------------------------------------
*/
inline
void MpStageBase::CalcTfmMatricesHelper_Complete(float* ccm, float nalnposs)
{
    float aCache[twmvEndOfCCMtx];
    float rr[6];

    //initialize matrix a to an identity matrix;
    //NOTE: only valid when indices start from 0
    #pragma omp simd
    for(int mi = 0; mi < twmvEndOfCCMtx; mi++) aCache[mi] = 0.0f;
    aCache[twmvCCM_0_0] = aCache[twmvCCM_1_1] = aCache[twmvCCM_2_2] = 1.0f;

    //calculate query center vector in advance
    #pragma omp simd
    for(int mi = twmvCVq_0; mi <= twmvCVq_2; mi++) ccm[mi] /= nalnposs;

    CalcRmatrix(ccm);

    //calculate reference center vector now
    #pragma omp simd
    for(int mi = twmvCVr_0; mi <= twmvCVr_2; mi++) ccm[mi] /= nalnposs;


    //NOTE: scale correlation matrix to enable rotation matrix 
    //NOTE: calculation in single precision without overflow and underflow:
    //ScaleRmatrix(ccmCache);
    float scale = GetRScale(ccm);
    #pragma omp simd
    for(int mi = 0; mi < twmvEndOfCCMtx; mi++) ccm[mi] /= scale;


    //calculate determinant
    float det = CalcDet(ccm);

    //calculate the product transposed(R) * R
    CalcRTR(ccm, rr);

    //Kabsch:
    //eigenvalues: form characteristic cubic x**3-3*spur*x**2+3*cof*x-det=0
    float spur = (rr[0] + rr[2] + rr[5]) * oneTHIRDf;
    float cof = (((((rr[2] * rr[5] - SQRD(rr[4])) + rr[0] * rr[5]) -
                SQRD(rr[3])) + rr[0] * rr[2]) -
                SQRD(rr[1])) * oneTHIRDf;

    bool abok = (spur > 0.0f);

    if(abok)
    {   //Kabsch:
        //reduce cubic to standard form y**3-3hy+2g=0 by putting x=y+spur

        //Kabsch: solve cubic: roots are e[0],e[1],e[2] in decreasing order
        //Kabsch: handle special case of 3 identical roots
        float e0, e1, e2;
        if(SolveCubic(det, spur, cof, e0, e1, e2))
        {
            //Kabsch: eigenvectors
            //almost always this branch gets executed
            CalcPartialA_Reg<0>(e0, rr, aCache);
            CalcPartialA_Reg<2>(e2, rr, aCache);
            abok = CalcCompleteA(e0, e1, e2, aCache);
        }
        if(abok) {
            //Kabsch: rotation matrix
            abok = CalcRotMtx(aCache, ccm);
        }
    }

    if(!abok) {//RotMtxToIdentity(ccm);
        #pragma omp simd
        for(int mi = 0; mi < twmvEndOfCCMtx; mi++) ccm[mi] = 0.0f;
        ccm[twmvCCM_0_0] = ccm[twmvCCM_1_1] = ccm[twmvCCM_2_2] = 1.0f;
    }

    //Kabsch: translation vector
    //NOTE: scaling translation vector would be needed if the data 
    //NOTE: vectors were scaled previously so that transformation is 
    //NOTE: applied in the original coordinate space
    CalcTrlVec(ccm);
}

// CalcTfmMatrices_Complete: calculate a tranformation matrix;
// DOUBLY_INVERTED, change places of query and reference sums so that a
// tranformation matrix is calculated wrt the query;
// then, revert back the transformation matrix to obtain it wrt the reference
// again; NOTE: for numerical stability (index) and symmetric results;
//
template<bool DOUBLY_INVERTED>
inline
void MpStageBase::CalcTfmMatrices_Complete(
    int qrylen, int dbstrlen, float* ccm)
{
    //#positions used to calculate cross-covarinaces:
    float nalnposs = ccm[twmvNalnposs];

    if(nalnposs <= 0.0f) return;

    if(DOUBLY_INVERTED && (qrylen < dbstrlen)) {
        TransposeRotMtx(ccm);
        myswap(ccm[twmvCVq_0], ccm[twmvCVr_0]);
        myswap(ccm[twmvCVq_1], ccm[twmvCVr_1]);
        myswap(ccm[twmvCVq_2], ccm[twmvCVr_2]);
    }

    CalcTfmMatricesHelper_Complete(ccm, nalnposs);

    if(DOUBLY_INVERTED && (qrylen < dbstrlen)) {
        InvertRotMtx(ccm);
        InvertTrlVec(ccm);
    }
}

// CalcTfmMatrices_DynamicOrientation_Complete: calculate tranformation 
// matrices wrt query or reference structure, whichever is longer;
inline
void MpStageBase::CalcTfmMatrices_DynamicOrientation_Complete(
    int qrylen, int dbstrlen, float* ccm)
{
    //#positions used to calculate cross-covarinaces:
    float nalnposs = ccm[twmvNalnposs];

    if(nalnposs <= 0.0f) return;

    if(!(qrylen < dbstrlen)) {
        TransposeRotMtx(ccm);
        myswap(ccm[twmvCVq_0], ccm[twmvCVr_0]);
        myswap(ccm[twmvCVq_1], ccm[twmvCVr_1]);
        myswap(ccm[twmvCVq_2], ccm[twmvCVr_2]);
    }

    CalcTfmMatricesHelper_Complete(ccm, nalnposs);
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// SaveBestScoreAndTM_Complete: complete version of saving the best 
// score along with transformation;
// save fragment indices and starting positions too;
// WRITEFRAGINFO, template parameter, flag of writing fragment attributes;
// CONDITIONAL, template parameter, flag of writing the score if it's greater at the same location;
// best, best score so far;
// qryndx, query serial number;
// dbstrndx, reference serial number;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps to perform for each reference structure;
// sfragfctxndx, current fragment factor x fragment length index, which is <maxnsteps;
// sfragndx, index defining fragment length;
// sfragpos, starting position within fragment;
// NOTE: memory pointers should be aligned!
// tfmCache, cached transformation matrix;
// wrkmemtmibest, working memory for iteration-best transformation matrices;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// 
template<bool WRITEFRAGINFO, bool CONDITIONAL>
inline
void MpStageBase::SaveBestScoreAndTM_Complete(
    const float best,
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
    if(best <= 0.0f) return;

    float currentbest = 0.0f;
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
// -------------------------------------------------------------------------
// SaveBestScoreAndTMAmongBests: save best scores and respective 
// transformation matrices by considering all partial best scores 
// calculated over all fragment factors; write the information to the 
// location of fragment factor 0;
// WRITEFRAGINFO, write fragment information if the best score is obtained;
// GRANDUPDATE, update the grand best score if the best score is obtained;
// FORCEWRITEFRAGINFO, force writing frag info for the best score obtained
// among the bests;
// SECONDARYUPDATE, indication of whether and how secondary update of best scores is done;
// qryndx, query index;
// rfnblkndx, reference block index;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps performed for each reference structure;
// effnsteps, effective (actual maximum) number of steps;
// NOTE: memory pointers should be aligned!
// ccm, for local cache;
// wrkmemtmibest, working memory for iteration-best transformation matrices;
// tfmmem, memory for transformation matrices;
// wrkmemaux, auxiliary working memory;
// wrkmemtmibest2nd, secondary working memory for iteration-best transformation 
// matrices (only slot 0 is used but indexing involves maxnsteps);
// 
template<
    int XDIM, int DATALN,
    bool WRITEFRAGINFO,
    bool GRANDUPDATE,
    bool FORCEWRITEFRAGINFO,
    int SECONDARYUPDATE>
inline
void MpStageBase::SaveBestScoreAndTMAmongBests(
    const int qryndx,
    const int rfnblkndx,
    const int ndbCstrs,
    const int maxnsteps,
    const int effnsteps,
    float (* __RESTRICT__ ccm)[XDIM],
    float* __RESTRICT__ wrkmemtmibest,
    float* __RESTRICT__ tfmmem,
    float* __RESTRICT__ wrkmemaux,
    float* __RESTRICT__ wrkmemtmibest2nd)
{
    enum {lnxSCR, lnxNDX, lnx2ND, lnxGRD, lnxN};

    const int istr0 = rfnblkndx * XDIM;
    const int istre = mymin(istr0 + XDIM, (int)ndbCstrs);

    //initialize cache for best scores and their indices:
    #pragma omp simd collapse(2)
    for(int f = 0; f < lnxN/*2*/; f++)
        for(int ii = 0; ii < XDIM; ii++)
            ccm[f][ii] = 0.0f;

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
    bool wrt2ndry = (SECONDARYUPDATE == SECONDARYUPDATE_UNCONDITIONAL);

    //ccm[0][...] contains maximums; write max values to slot 0
    //NOTE: aligned crashes when not using native instructions!
    #pragma omp simd //aligned(wrkmemaux,wrkmemtmibest2nd:DATALN)
    for(int ri = istr0; ri < istre; ri++) {
        int ii = ri - istr0;
        int si = ccm[lnxNDX][ii];
        float bscore = ccm[lnxSCR][ii];
        int mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        int mloc = ((qryndx * maxnsteps + si) * nTAuxWorkingMemoryVars) * ndbCstrs;
        ccm[lnxGRD][ii] = 0.0f;
        ccm[lnx2ND][ii] = wrt2ndry;
        //coalesced WRITE for multiple references
        if(si != 0)
            wrkmemaux[mloc0 + tawmvBestScore * ndbCstrs + ri] = bscore;
        {
            int mloc20 = ((qryndx * maxnsteps + 0) * ndbCstrs + ndbCstrs) * nTTranformMatrix;
            if(SECONDARYUPDATE == SECONDARYUPDATE_CONDITIONAL) {
                //NOTE: 2nd'ry scores written immediately following tfms
                float bscore2nd = wrkmemtmibest2nd[mloc20 + ri/*dbstrndx*/];
                ccm[lnx2ND][ii] = wrt2ndry = (bscore2nd < bscore);//reuse cache
            }
            if(wrt2ndry) wrkmemtmibest2nd[mloc20 + ri/*dbstrndx*/] = bscore;
        }
        if(GRANDUPDATE) {//update the grand best
            float grand = wrkmemaux[mloc0 + tawmvGrandBest * ndbCstrs + ri/*dbstrndx*/];
            ccm[lnxGRD][ii] = wrtgrand = (grand < bscore);//reuse cache
        }
        //coalesced WRITE for multiple references
        if(wrtgrand)
            wrkmemaux[mloc0 + tawmvGrandBest * ndbCstrs + ri/*dbstrndx*/] = bscore;
        if(WRITEFRAGINFO && (FORCEWRITEFRAGINFO || wrtgrand)) {
            float frgndx = wrkmemaux[mloc + tawmvSubFragNdxCurrent * ndbCstrs + ri/*dbstrndx*/];
            float frgpos = wrkmemaux[mloc + tawmvSubFragPosCurrent * ndbCstrs + ri/*dbstrndx*/];
            wrkmemaux[mloc0 + tawmvSubFragNdx * ndbCstrs + ri/*dbstrndx*/] = frgndx;
            wrkmemaux[mloc0 + tawmvSubFragPos * ndbCstrs + ri/*dbstrndx*/] = frgpos;
        }
    }

    for(int ri = istr0; ri < istre; ri++) {
        int ii = ri - istr0;
        int si = ccm[lnxNDX][ii];
        int mloc0 = ((qryndx * maxnsteps + 0) * ndbCstrs + ri/*dbstrndx*/) * nTTranformMatrix;
        int mloc = ((qryndx * maxnsteps + si) * ndbCstrs + ri/*dbstrndx*/) * nTTranformMatrix;
        int tfmloc = (qryndx * ndbCstrs + ri/*dbstrndx*/) * nTTranformMatrix;
        wrtgrand = ccm[lnxGRD][ii];
        wrt2ndry = ccm[lnx2ND][ii];
        //READ and WRITE iteration-best transformation matrices;
        //NOTE: aligned crashes when not using native instructions!
        #pragma omp simd //aligned(wrkmemtmibest,wrkmemtmibest2nd,tfmmem:DATALN)
        for(int f = 0; f < nTTranformMatrix; f++) {
            float value = 0.0f;
            if(si != 0 || wrtgrand || wrt2ndry) value = wrkmemtmibest[mloc + f];//READ
            if(si != 0) wrkmemtmibest[mloc0 + f] = value;//WRITE
            if(wrt2ndry) wrkmemtmibest2nd[mloc0 + f] = value;//WRITE
            if(wrtgrand) tfmmem[tfmloc + f] = value;//WRITE
        }
    }
}

// -------------------------------------------------------------------------

#endif//__MpStageBase_h__
