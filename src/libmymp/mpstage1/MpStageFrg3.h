/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __MpStageFrg3_h__
#define __MpStageFrg3_h__

#include <algorithm>

#include "libutil/mybase.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libgenp/gdats/PMBatchStrData.h"
#include "libgenp/gdats/PMBatchStrDataIndex.h"
#include "libmymp/mpproc/mpprocconfbase.h"
#include "libmymp/mpstages/transformbase.h"
#include "libmymp/mpstages/scoringbase.h"
#include "libmymp/mpstages/covariancebase.h"
#include "libmymp/mpstages/linearscoringbase.h"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/fragment.cuh"
#include "libmymp/mpstage1/MpStageBase.h"
#include "libmymp/mpstage1/MpStage1.h"
#include "libmymp/mpdp/MpDPHub.h"
#include "libmycu/cucom/cudef.h"

// -------------------------------------------------------------------------
// class MpStageFrg3 for implementing structure comparison at stage 2
//
class MpStageFrg3: public MpStage1 {
public:
    MpStageFrg3(
        const int maxndpiters,
        const uint maxnsteps,
        const uint minfraglen,
        const float prescore,
        const int stepinit,
        char** querypmbeg, char** querypmend,
        char** bdbCpmbeg, char** bdbCpmend,
        char** queryndxpmbeg, char** queryndxpmend,
        char** bdbCndxpmbeg, char** bdbCndxpmend,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint qystrnlen, uint dbstrnlen,
        uint dbxpad,
        float* scores, 
        float* tmpdpdiagbuffers, float* tmpdpbotbuffer,
        float* tmpdpalnpossbuffer, uint* maxscoordsbuf, char* btckdata,
        float* wrkmem, float* wrkmemccd,
        float* wrkmemorytmalt, float* wrkmemtm, float* wrkmemtmibest,
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
        ),
        dphub_(
            maxnsteps_,
            querypmbeg_, querypmend_, bdbCpmbeg_, bdbCpmend_,
            nqystrs_, ndbCstrs_, nqyposs_, ndbCposs_,  qystr1len_, dbstr1len_, dbxpad_,
            tmpdpdiagbuffers_, tmpdpbotbuffer_, tmpdpalnpossbuffer_, maxscoordsbuf_, btckdata_,
            wrkmem_, wrkmemccd_,  wrkmemtm_,  wrkmemtmibest_,
            wrkmemaux_, wrkmem2_, alndatamem, tfmmem_, globvarsbuf_
        ),
        wrkmemtmalt_(wrkmemorytmalt),
        queryndxpmbeg_(queryndxpmbeg), queryndxpmend_(queryndxpmend),
        bdbCndxpmbeg_(bdbCndxpmbeg), bdbCndxpmend_(bdbCndxpmend)
    {}

    virtual void Run() {
        const int simthreshold = CLOptions::GetC_TRIGGER();
        static const float thrsimilarityperc = (float)simthreshold / 100.0f;
        static const float locgapcost = -0.8f;
        //fill in DP matrix with local similarity scores:
        if(0.0f < thrsimilarityperc)
            dphub_.ExecDPSSLocal128xKernel(
                locgapcost,
                querypmbeg_, bdbCpmbeg_,
                wrkmemaux_/*cnv*/, tmpdpdiagbuffers_/*wrk*/, tmpdpbotbuffer_/*wrk*/,
                btckdata_/*out*/);
        //find a number of favorable superpositions using a linear algorithm
        ScoreBasedOnFragmatching3Kernel(
            thrsimilarityperc,
            querypmbeg_, bdbCpmbeg_, queryndxpmbeg_, bdbCndxpmbeg_,
            wrkmemtmibest_, wrkmemtm_, wrkmemaux_, btckdata_/*dpscoremtx_*/);
        //calculate order-dependent scores for the configs of each section:
        dphub_.ExecDPScore128xKernel<true/*GAP0*/>(
            0.0f/*gapopencost*/,
            querypmbeg_, bdbCpmbeg_, wrkmemtm_/*in*/,
            wrkmemaux_/*cnv*/, tmpdpdiagbuffers_/*wrk*/, tmpdpbotbuffer_/*wrk*/);
        //sort best order-dependent scores and save the corresponding tfms:
        SortAmongDPswiftsKernel(
            bdbCpmbeg_,
            tmpdpdiagbuffers_/*in*/, wrkmemtm_/*in*/, wrkmemtmalt_/*out*/, wrkmemaux_);
        //process top N tfms from the extensive application of spatial index:
        Refine_tfmaltconfig();
        //refine alignment and its boundaries using DP;
        //1. With a gap cost:
        DPRefine<false/*GAP0*/,false/*PRESCREEN*/,true/*WRKMEMTM1*/>(maxndpiters_, prescore_);
        //2. No gap cost:
        DPRefine<true/*GAP0*/,false/*PRESCREEN*/,false/*WRKMEMTM1*/>(maxndpiters_, prescore_);
    }


protected:
    void ScoreBasedOnFragmatching3Kernel(
        const float thrsimilarityperc,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const char* const * const __RESTRICT__ queryndxpmbeg,
        const char* const * const __RESTRICT__ bdbCndxpmbeg,
        float* const __RESTRICT__ wrkmemtmibest,
        float* const __RESTRICT__ wrkmemtm,
        float* const __RESTRICT__ wrkmemaux,
        const char* const __RESTRICT__ dpscoremtx);

    void SortAmongDPswiftsKernel(
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const float* const __RESTRICT__ tmpdpdiagbuffers,
        const float* const __RESTRICT__ wrkmemtm,
        float* const __RESTRICT__ wrkmemtmtarget,
        float* const __RESTRICT__ wrkmemaux);

    void Refine_tfmaltconfig();


protected:
    template<int nFRGS, int DPSCDIMY, int DPSCDIMX, int DATALN>
    void CalcLocalSimilarity2_frg2(
        const float thrsimilarityperc,
        const int ndbCposs,
        const int dbxpad,
        const int qrydst, const int dbstrdst,
        const int qrylen, const int dbstrlen,
        const int qrypos, const int rfnpos,
        const char* const __RESTRICT__ dpscoremtx,
        unsigned char dpsc[DPSCDIMX],
        int convflags[nFRGS]);


    template<int SCORDIMX, int DATALN>
    void ProduceAlignmentUsingDynamicIndex2(
        const bool secstrmatchaln,
        float* const __RESTRICT__ stack,
        const int stacksize,
        const int qrydst, const int dbstrdst,
        int qrylen, int dbstrlen,
        int qrypos, int rfnpos, int fraglen,
        const bool WRTNDX,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const char* const * const __RESTRICT__ queryndxpmbeg,
        const char* const * const __RESTRICT__ bdbCndxpmbeg,
        float* const __RESTRICT__ tfm,
        float* const __RESTRICT__ coords,
        char* const __RESTRICT__ ssas);

    template<int SECSTRFILT, int SCORDIMX, int DATALN>
    void ProduceAlignmentUsingIndex2Reference(
        float* const __RESTRICT__ stack,
        const int stacksize,
        const int qrydst, const int dbstrdst,
        int qrylen, const int dbstrlen,
        int qrypos, const int /* rfnpos */, int fraglen,
        const bool WRTNDX,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const char* const * const __RESTRICT__ queryndxpmbeg,
        const char* const * const __RESTRICT__ bdbCndxpmbeg,
        float* const __RESTRICT__ tfm,
        float* const __RESTRICT__ coords,
        char* const __RESTRICT__ ssas);

    template<int SECSTRFILT, int SCORDIMX, int DATALN>
    void ProduceAlignmentUsingIndex2Query(
        float* const __RESTRICT__ stack,
        const int stacksize,
        const int qrydst, const int dbstrdst,
        const int qrylen, int dbstrlen,
        const int /* qrypos */, int rfnpos, int fraglen,
        const bool WRTNDX,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const char* const * const __RESTRICT__ queryndxpmbeg,
        const char* const * const __RESTRICT__ bdbCndxpmbeg,
        float* const __RESTRICT__ tfm,
        float* const __RESTRICT__ coords,
        char* const __RESTRICT__ ssas);


    template<int nEFFDS, int XDIM, int SCORDIMX>
    void CalcCCMatrices_SWFTscan_Complete(
        const int nalnposs,
        const float* const __RESTRICT__ coords,
        float (* __RESTRICT__ ccm)[XDIM]);

    template<int nEFFDS, int XDIM, int SCORDIMX>
    void CalcScoresUnrl_SWFTscanProgressive_Complete(
        const float d02,
        const int nalnposs,
        const float* __RESTRICT__ tfm,
        float* const __RESTRICT__ coords,
        float (* __RESTRICT__ ccm)[XDIM]);


    template<int N2SCTS, int NTOPTFMS, int SECTION2, int DATALN>
    void Save2ndryScoreAndTM_Complete(
        const float best,
        const int qryndx,
        const int dbstrndx,
        const int ndbCstrs,
        const int maxnsteps,
        const int nthreads,
        const int tid,
        const float* __RESTRICT__ tfm,
        float* __RESTRICT__ wrkmemtmibest);

    template<int NTOPTFMS, int XDIM, int DATALN>
    void SaveTopNScoresAndTMsAmongBests(
        const int qryndx,
        const int rfnblkndx,
        const int ndbCstrs,
        const int maxnsteps,
        const int effnsteps,
        float scoN[NTOPTFMS][XDIM],
        int ndxN[NTOPTFMS][XDIM],
        float* __RESTRICT__ wrkmemtmibest,
        float* __RESTRICT__ wrkmemtm,
        float* __RESTRICT__ wrkmemaux);

    template<int N2SCTS, int NTOPTFMS, int SECTION2, int XDIM, int DATALN>
    void SaveTopNScoresAndTMsAmongSecondaryBests(
        const int qryndx,
        const int rfnblkndx,
        const int ndbCstrs,
        const int maxnsteps,
        const int effnsteps,
        const int nthreads,
        float scoN[NTOPTFMS][XDIM],
        int ndxN[NTOPTFMS][XDIM],
        float* __RESTRICT__ wrkmemtmibest,
        float* __RESTRICT__ wrkmemtm,
        float* __RESTRICT__ wrkmemaux);


    template<int N2SCTS, int NTOPTFMS, int NMAXREFN, int XDIM, int DATALN>
    void SortBestDPscoresAndTMsAmongDPswifts(
        const int SECTION, 
        const int nbranches,
        const int qryndx,
        const int rfnblkndx,
        const int ndbCstrs,
        const int ndbCposs,
        const int dbxpad,
        const int maxnsteps,
        float scoN[XDIM][NTOPTFMS],
        int ndxN[XDIM][NTOPTFMS],
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const float* const __RESTRICT__ tmpdpdiagbuffers,
        const float* const __RESTRICT__ wrkmemtm,
        float* const __RESTRICT__ wrkmemtmtarget,
        float* const __RESTRICT__ wrkmemaux);


protected:
    template<int SECSTRFILT>
    void NNByIndexQuery(
        int STACKSIZE,
        int& nestndx,
        float& qxn, float& qyn, float& qzn, 
        float rx, float ry, float rz, char rss,
        int qrydst, int root, int dimndx,
        const char* const * const __RESTRICT__ querypmbeg,
        const char* const * const __RESTRICT__ /* bdbCpmbeg */,
        const char* const * const __RESTRICT__ queryndxpmbeg,
        const char* const * const __RESTRICT__ /* bdbCndxpmbeg */,
        float* const __RESTRICT__ stack);

    template<int SECSTRFILT>
    void NNByIndexReference(
        int STACKSIZE,
        int& nestndx,
        float& rxn, float& ryn, float& rzn, 
        float qx, float qy, float qz, char qss,
        int dbstrdst, int root, int dimndx,
        const char* const * const __RESTRICT__ /* querypmbeg */,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const char* const * const __RESTRICT__ /* queryndxpmbeg */,
        const char* const * const __RESTRICT__ bdbCndxpmbeg,
        float* const __RESTRICT__ stack);

protected:
    MpDPHub dphub_;
    float* const wrkmemtmalt_;
    char* const * const queryndxpmbeg_, * const * const queryndxpmend_;
    char* const * const bdbCndxpmbeg_, * const *const bdbCndxpmend_;
};



// -------------------------------------------------------------------------
// INLINES ...
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// CalcLocalSimilarity2_frg2: calculate provisional local similarity during 
// extensive fragment-based search of optimal superpositions;
// NOTE: contrary to the GPU version, no conv flag distribution applies!
// thrsimilarityperc, threshold percentage of local similarity score for a 
// fragment to be considered as one having the potential to improve superposition;
// NOTE: `converged' if provisional score is less than the given threshold;
// ndbCposs, total number of db structure positions in the chunk;
// dbxpad, number of padded positions for memory alignment;
// NOTE: memory pointers should be aligned!
// dpscoremtx, input rounded global dp score matrix;
// dpsc, locally used cache for dp scores;
// convflags, convergence flags for fragment indices 0 and 1;
// 
template<int nFRGS, int DPSCDIMY, int DPSCDIMX, int DATALN>
inline
void MpStageFrg3::CalcLocalSimilarity2_frg2(
    const float thrsimilarityperc,
    const int ndbCposs,
    const int dbxpad,
    const int qrydst, const int dbstrdst,
    const int qrylen, const int dbstrlen,
    const int qrypos, const int rfnpos,
    const char* const __RESTRICT__ dpscoremtx,
    unsigned char dpsc[DPSCDIMX],
    int convflags[nFRGS])
{
    convflags[0] = convflags[1] = CONVERGED_SCOREDP_bitval;

    int fraglen = GetNAlnPoss_frg(
        qrylen, dbstrlen, 0/*qrypos,unused*/, 0/*rfnpos,unused*/,
        0/*qryfragfct,unused*/, 0/*rfnfragfct,unused*/, 0/*fragndx; first smaller*/);

    //false if fragment is out of bounds
    if(qrylen < qrypos + fraglen || dbstrlen < rfnpos + fraglen || fraglen < 1)
        return;

    fraglen = GetNAlnPoss_frg(
        qrylen, dbstrlen, 0/*qrypos,unused*/, 0/*rfnpos,unused*/,
        0/*qryfragfct,unused*/, 0/*rfnfragfct,unused*/,
        1/*fragndx; always use the larger*/);

    //convergence flag for the shorter and longer fragments:
    convflags[0] = 0;
    convflags[1] =
        (qrylen < qrypos + fraglen || dbstrlen < rfnpos + fraglen || fraglen < 1)
        ? CONVERGED_SCOREDP_bitval: 0;

    if(thrsimilarityperc <= 0.0f) return;

    //actual length of fragment within which local scores are verified:
    // fraglen = lnYIts * lYdim;
    if(qrylen < qrypos + fraglen) fraglen = qrylen - qrypos;
    if(dbstrlen < rfnpos + fraglen) fraglen = dbstrlen - rfnpos;

    //read a block of dp matrix values (coalesced reads), get max, and
    //set the convergence flag if calculated local similarity is too low:
    const int dblen = ndbCposs + dbxpad;
    //NOTE: make calculations as close as possible to the GPU version:
    //NOTE: end pos. from: (rfnpos + (threadIdx.x+1) * 4 + 3 < dbstrlen);
    //NOTE: +3 for max difference of address up-alignment;
    const int pidelta = mymin(DPSCDIMX, (dbstrlen - 3 - rfnpos) & (~3));
    int ii = 0;

    #pragma omp simd
    for(int pi = 0; pi < DPSCDIMX; pi++) dpsc[pi] = 0;

    for(; qrypos + ii < qrylen && ii < DPSCDIMY; ii++) {
        int rpos = (qrydst + qrypos + ii) * dblen + dbstrdst + rfnpos;
        //NOTE: make sure position is 4-bytes aligned as in the GPU version!
        //NOTE: aligned bytes imply that the results may differ a bit upon a
        //NOTE: different configuration of #queries and references in a chunk!
        //NOTE: that's because rpos takes on a different value and dpsCache
        //NOTE: reads from different locations at the boundaries; this may cause a
        //NOTE: different value for max!
        const int pibeg = ALIGN_UP(rpos, 4/*sizeof(int)*/);
        const int piend = pibeg + pidelta;
        unsigned char max1 = 0;
        #pragma omp simd aligned(dpscoremtx:DATALN) reduction(max:max1)
        for(int pi = pibeg; pi < piend; pi++)
            max1 = mymax(max1, (unsigned char)(dpscoremtx[pi]));
        dpsc[ii] = max1;
    }

    //global max score byte
    unsigned char maxg = 0;
    #pragma omp simd reduction(max:maxg)
    for(int pi = 0; pi < DPSCDIMX; pi++) maxg = mymax(maxg, dpsc[pi]);

    //NOTE: consider ignoring check for lengths <9 (4+3+2, word+mem.aln.+margin).
    if((float)maxg < (float)fraglen * thrsimilarityperc) {
        convflags[0] = convflags[0] | CONVERGED_SCOREDP_bitval;
        convflags[1] = convflags[1] | CONVERGED_SCOREDP_bitval;
    }
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// ProduceAlignmentUsingDynamicIndex2: find coordinates of nearest 
// reference atoms at each query/reference (determined dynamically) position,
// using index; superpositions based on fragments;
// see ProduceAlignmentUsingIndex2Reference/ProduceAlignmentUsingIndex2Query
// for parameter description;
// 
template<int SCORDIMX, int DATALN>
inline
void MpStageFrg3::ProduceAlignmentUsingDynamicIndex2(
    const bool secstrmatchaln,
    float* const __RESTRICT__ stack,
    const int stacksize,
    const int qrydst, const int dbstrdst,
    int qrylen, int dbstrlen,
    int qrypos, int rfnpos, int fraglen,
    const bool WRTNDX,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const char* const * const __RESTRICT__ queryndxpmbeg,
    const char* const * const __RESTRICT__ bdbCndxpmbeg,
    float* const __RESTRICT__ tfm,
    float* const __RESTRICT__ coords,
    char* const __RESTRICT__ ssas)
{
    if(qrylen < dbstrlen) {
        if(secstrmatchaln)
            ProduceAlignmentUsingIndex2Reference<1/*SECSTRFILT*/,SCORDIMX,DATALN>(
                stack, stacksize, qrydst, dbstrdst,
                qrylen, dbstrlen, qrypos, rfnpos, fraglen, WRTNDX,
                querypmbeg, bdbCpmbeg, queryndxpmbeg, bdbCndxpmbeg,
                tfm, coords, ssas);
        else
            ProduceAlignmentUsingIndex2Reference<0/*SECSTRFILT*/,SCORDIMX,DATALN>(
                stack, stacksize, qrydst, dbstrdst,
                qrylen, dbstrlen, qrypos, rfnpos, fraglen, WRTNDX,
                querypmbeg, bdbCpmbeg, queryndxpmbeg, bdbCndxpmbeg,
                tfm, coords, ssas);
    } else {
        if(secstrmatchaln)
            ProduceAlignmentUsingIndex2Query<1/*SECSTRFILT*/,SCORDIMX,DATALN>(
                stack, stacksize, qrydst, dbstrdst,
                qrylen, dbstrlen, qrypos, rfnpos, fraglen, WRTNDX,
                querypmbeg, bdbCpmbeg, queryndxpmbeg, bdbCndxpmbeg,
                tfm, coords, ssas);
        else
            ProduceAlignmentUsingIndex2Query<0/*SECSTRFILT*/,SCORDIMX,DATALN>(
                stack, stacksize, qrydst, dbstrdst,
                qrylen, dbstrlen, qrypos, rfnpos, fraglen, WRTNDX,
                querypmbeg, bdbCpmbeg, queryndxpmbeg, bdbCndxpmbeg,
                tfm, coords, ssas);
    }
}

// -------------------------------------------------------------------------
// ProduceAlignmentUsingIndex2Reference: find coordinates of nearest 
// reference atoms at each query position using index; the result 
// follows from superpositions based on fragments;
// write the coordinates of neighbors for each position processed;
// NOTE: 1D block processes reference fragment along structure positions;
// SECSTRFILT, flag of whether the secondary structure match is required for 
// building an alignment;
// stack, stack for index;
// stacksize, dynamically determined stack size;
// qrydst, dbstrdst, distances to the beginnings of query and reference structures;
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, query and reference fragment starting positions;
// fraglen, fragment length;
// WRTNDX, flag of writing query indices participating in an alignment;
// NOTE: memory pointers should be aligned!
// tfm, transformation matrix;
// coords, cache for coordinates to be saved for further processing;
// ssas, cache for secondary structure assignments;
// 
template<int SECSTRFILT, int SCORDIMX, int DATALN>
inline
void MpStageFrg3::ProduceAlignmentUsingIndex2Reference(
    float* const __RESTRICT__ stack,
    const int stacksize,
    const int qrydst, const int dbstrdst,
    int qrylen, const int dbstrlen,
    int qrypos, const int /* rfnpos */, int fraglen,
    const bool WRTNDX,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const char* const * const __RESTRICT__ queryndxpmbeg,
    const char* const * const __RESTRICT__ bdbCndxpmbeg,
    float* const __RESTRICT__ tfm,
    float* const __RESTRICT__ coords,
    char* const __RESTRICT__ ssas)
{
    fraglen = mymin(dbstrlen, SCORDIMX);
    qrypos = mymax(0, qrypos - (fraglen>>1));
    qrylen = mymin(qrylen, qrypos + fraglen);
    qrypos = mymax(0, qrylen - fraglen);

    int nalnposs = 0;
    //#matched (aligned) positions (including those masked):
    tfm[twmvNalnposs] = nalnposs = (qrylen - qrypos);

    #pragma omp simd aligned(querypmbeg:DATALN)
    for(int pi = 0; pi < nalnposs; pi++) {
        int dpos = qrydst + qrypos + pi;
        float qx, qy, qz, qxt, qyt, qzt;
        qx = qxt = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DX>(querypmbeg, dpos);
        qy = qyt = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DY>(querypmbeg, dpos);
        qz = qzt = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DZ>(querypmbeg, dpos);
        if(SECSTRFILT == 1) ssas[pi] = PMBatchStrData::GetFieldAt<char,pmv2Dss>(querypmbeg, dpos);
        //cache query coordinates:
        coords[SCORDIMX * dpapsQRYx + pi] = qx;
        coords[SCORDIMX * dpapsQRYy + pi] = qy;
        coords[SCORDIMX * dpapsQRYz + pi] = qz;
        //transform atom:
        transform_point(tfm, qxt, qyt, qzt);
        //save transformation for reuse below:
        coords[SCORDIMX * dpapsRFNx + pi] = qxt;
        coords[SCORDIMX * dpapsRFNy + pi] = qyt;
        coords[SCORDIMX * dpapsRFNz + pi] = qzt;
    }

    for(int pi = 0; pi < nalnposs; pi++) {
        char qss;
        float rx = 0.0f, ry = 0.0f, rz = 0.0f;
        //these below are transformed query coordinates!
        float qx = coords[SCORDIMX * dpapsRFNx + pi];
        float qy = coords[SCORDIMX * dpapsRFNy + pi];
        float qz = coords[SCORDIMX * dpapsRFNz + pi];
        if(SECSTRFILT == 1) qss = ssas[pi];
        //reference index of the position nearest to a query atom:
        int bestrnx = -1;

        //nearest neighbour using the index tree:
        NNByIndexReference<SECSTRFILT>(
            stacksize,
            bestrnx,//returned
            rx, ry, rz,//returned
            qx, qy, qz, qss,
            dbstrdst, (dbstrlen >> 1)/*root*/, 0/*dimndx*/,
            querypmbeg, bdbCpmbeg, queryndxpmbeg, bdbCndxpmbeg, stack);

        //mask aligned position for no contribution to the alignment:
        //TODO: bestqnx<0 since no difference in values is examined
        if(bestrnx <= 0) {rx = ry = rz = SCNTS_COORD_MASK;}

        //save reference coordinates:
        coords[SCORDIMX * dpapsRFNx + pi] = rx;
        coords[SCORDIMX * dpapsRFNy + pi] = ry;
        coords[SCORDIMX * dpapsRFNz + pi] = rz;

        //WRITE reference position;
        //TODO: 0<=bestrnx since no difference in values is examined
        if(WRTNDX && 0 < bestrnx) coords[SCORDIMX * nTDPAlignedPoss + pi] = bestrnx;
    }
}

// -------------------------------------------------------------------------
// ProduceAlignmentUsingIndex2Query: find coordinates of nearest 
// query atoms at each reference position using index; the result 
// follows from superpositions based on fragments;
// write the coordinates of neighbors for each position processed;
// NOTE: 1D block processes reference fragment along structure positions;
// SECSTRFILT, flag of whether the secondary structure match is required for 
// building an alignment;
// stack, stack for index;
// stacksize, dynamically determined stack size;
// qrydst, dbstrdst, distances to the beginnings of query and reference structures;
// qrylen, dbstrlen, query and reference lengths;
// qrypos, rfnpos, query and reference fragment starting positions;
// fraglen, fragment length;
// WRTNDX, flag of writing query indices participating in an alignment;
// NOTE: memory pointers should be aligned!
// tfm, transformation matrix;
// coords, cache for coordinates to be saved for further processing;
// ssas, cache for secondary structure assignments;
// 
template<int SECSTRFILT, int SCORDIMX, int DATALN>
inline
void MpStageFrg3::ProduceAlignmentUsingIndex2Query(
    float* const __RESTRICT__ stack,
    const int stacksize,
    const int qrydst, const int dbstrdst,
    const int qrylen, int dbstrlen,
    const int /* qrypos */, int rfnpos, int fraglen,
    const bool WRTNDX,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const char* const * const __RESTRICT__ queryndxpmbeg,
    const char* const * const __RESTRICT__ bdbCndxpmbeg,
    float* const __RESTRICT__ tfm,
    float* const __RESTRICT__ coords,
    char* const __RESTRICT__ ssas)
{
    fraglen = mymin(qrylen, SCORDIMX);
    //qrypos = mymax(0, qrypos - (fraglen>>1));
    rfnpos = mymax(0, rfnpos - (fraglen>>1));
    dbstrlen = mymin(dbstrlen, rfnpos + fraglen);
    rfnpos = mymax(0, dbstrlen - fraglen);

    int nalnposs = 0;
    //#matched (aligned) positions (including those masked):
    tfm[twmvNalnposs] = nalnposs = (dbstrlen - rfnpos);

    #pragma omp simd aligned(querypmbeg:DATALN)
    for(int pi = 0; pi < nalnposs; pi++) {
        int dpos = dbstrdst + rfnpos + pi;
        float rx, ry, rz, rxt, ryt, rzt;
        rx = rxt = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DX>(bdbCpmbeg, dpos);
        ry = ryt = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DY>(bdbCpmbeg, dpos);
        rz = rzt = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DZ>(bdbCpmbeg, dpos);
        if(SECSTRFILT == 1) ssas[pi] = PMBatchStrData::GetFieldAt<char,pmv2Dss>(bdbCpmbeg, dpos);
        //cache query coordinates:
        coords[SCORDIMX * dpapsRFNx + pi] = rx;
        coords[SCORDIMX * dpapsRFNy + pi] = ry;
        coords[SCORDIMX * dpapsRFNz + pi] = rz;
        //transform atom:
        transform_point(tfm, rxt, ryt, rzt);
        //save transformation for reuse below:
        coords[SCORDIMX * dpapsQRYx + pi] = rxt;
        coords[SCORDIMX * dpapsQRYy + pi] = ryt;
        coords[SCORDIMX * dpapsQRYz + pi] = rzt;
    }

    for(int pi = 0; pi < nalnposs; pi++) {
        char rss;
        float qx = 0.0f, qy = 0.0f, qz = 0.0f;
        //these below are transformed reference coordinates!
        float rx = coords[SCORDIMX * dpapsQRYx + pi];
        float ry = coords[SCORDIMX * dpapsQRYy + pi];
        float rz = coords[SCORDIMX * dpapsQRYz + pi];
        if(SECSTRFILT == 1) rss = ssas[pi];
        //query index of the position nearest to a reference atom:
        int bestqnx = -1;

        //nearest neighbour using the index tree:
        NNByIndexQuery<SECSTRFILT>(
            stacksize,
            bestqnx,//returned
            qx, qy, qz,//returned
            rx, ry, rz, rss,
            qrydst, (qrylen >> 1)/*root*/, 0/*dimndx*/,
            querypmbeg, bdbCpmbeg, queryndxpmbeg, bdbCndxpmbeg, stack);

        //mask aligned position for no contribution to the alignment:
        //TODO: bestqnx<0 since no difference in values is examined
        if(bestqnx <= 0) {qx = qy = qz = SCNTS_COORD_MASK;}

        //save query coordinates:
        coords[SCORDIMX * dpapsQRYx + pi] = qx;
        coords[SCORDIMX * dpapsQRYy + pi] = qy;
        coords[SCORDIMX * dpapsQRYz + pi] = qz;

        //WRITE query position;
        //TODO: 0<=bestqnx since no difference in values is examined
        if(WRTNDX && 0 < bestqnx) coords[SCORDIMX * nTDPAlignedPoss + pi] = bestqnx;
    }
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// CalcCCMatrices_SWFTscan_Complete: calculate cross-covariance matrix 
// between the query and reference structures given an alignment between them;
// Version for alignments obtained as a result of linearity application;
// nalnposs, #matched (aligned) positions (alignment length);
// coords, saved coordinates;
// ccm, cache for the cross-covariance matrix and related data;
// 
template<int nEFFDS, int XDIM, int SCORDIMX>
inline
void MpStageFrg3::CalcCCMatrices_SWFTscan_Complete(
    const int nalnposs,
    const float* const __RESTRICT__ coords,
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
        {
            float qx = coords[SCORDIMX * dpapsQRYx + (ai + pi)];
            float qy = coords[SCORDIMX * dpapsQRYy + (ai + pi)];
            float qz = coords[SCORDIMX * dpapsQRYz + (ai + pi)];

            float rx = coords[SCORDIMX * dpapsRFNx + (ai + pi)];
            float ry = coords[SCORDIMX * dpapsRFNy + (ai + pi)];
            float rz = coords[SCORDIMX * dpapsRFNz + (ai + pi)];

            //compare only the first coordinates:
            if(qx < SCNTS_COORD_MASK_cmp && rx < SCNTS_COORD_MASK_cmp) {
                UpdateCCMCacheHelper<XDIM>(qx, qy, qz,  rx, ry, rz,  ccm, pi);
                //update the number of positions
                ccm[twmvNalnposs][pi] += 1.0f;
            }
        }
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
// -------------------------------------------------------------------------
// CalcScoresUnrl_SWFTscanProgressive_Complete: calculate/reduce scores for
// obtained superpositions progressively so that order-dependent alignment
// preserves; results from exhaustively applying a linear algorithm; 
// nalnposs, #matched (aligned) positions (alignment length);
// tfm, transformation matrix;
// coords, saved coordinates;
// ccm, cache for the cross-covariance matrix and related data;
// 
template<int nEFFDS, int XDIM, int SCORDIMX>
inline
void MpStageFrg3::CalcScoresUnrl_SWFTscanProgressive_Complete(
    const float d02,
    const int nalnposs,
    const float* __RESTRICT__ tfm,
    float* const __RESTRICT__ coords,
    float (* __RESTRICT__ ccm)[XDIM])
{
    enum {//NOTE: lxNinpout<=NFCRDS assumed!
        lxOUTMAX,//maximum score at a position (output)
        lxOUTNDX,//original index at a position (output)
        lxINPSCO,//original score (input)
        lxINPNDX,//original index (input)
        lxNinpout//total number of fields
    };

    //manually unroll along data blocks:
    for(int ai = 0; ai < nalnposs; ai += XDIM)
    {
        int piend = mymin(XDIM, nalnposs - ai);
        #pragma omp simd
        for(int pi = 0; pi < piend; pi++)
        {
            float qx = coords[SCORDIMX * dpapsQRYx + (ai + pi)];
            float qy = coords[SCORDIMX * dpapsQRYy + (ai + pi)];
            float qz = coords[SCORDIMX * dpapsQRYz + (ai + pi)];

            float rx = coords[SCORDIMX * dpapsRFNx + (ai + pi)];
            float ry = coords[SCORDIMX * dpapsRFNy + (ai + pi)];
            float rz = coords[SCORDIMX * dpapsRFNz + (ai + pi)];

            float tqnx = coords[SCORDIMX * nTDPAlignedPoss + (ai + pi)];

            float qnx = -1.0f;
            float sco = 0.0f;
            float dst;

            //compare only the first coordinates:
            if(qx < SCNTS_COORD_MASK_cmp && rx < SCNTS_COORD_MASK_cmp) {
                dst = transform_and_distance2(tfm, qx, qy, qz,  rx, ry, rz);
                sco = GetPairScore(d02, dst);
                qnx = tqnx;
            }

            //reuse ccm cache for packing required data:
            ccm[0][pi] = sco;
            ccm[1][pi] = qnx;
        }

        //overwrite portion of used data with new:
        #pragma omp simd
        for(int pi = 0; pi < piend; pi++) {
            coords[SCORDIMX * lxINPSCO + (ai + pi)] = ccm[0][pi];//score
            coords[SCORDIMX * lxINPNDX + (ai + pi)] = ccm[1][pi];//index
        }
    }

    //initialize cache:
    #pragma omp simd
    for(int pi = 0; pi < SCORDIMX; pi++) {
        coords[SCORDIMX * lxOUTMAX + pi] = 0.0f;
        coords[SCORDIMX * lxOUTNDX + pi] = -1.0f;
    }

    //process query/reference positions pregressively to find max score
    for(int pi = 0; pi < nalnposs; pi++) {
        float sco = coords[SCORDIMX * lxINPSCO + pi];
        float qnx = coords[SCORDIMX * lxINPNDX + pi];

        if(sco <= 0.0f || qnx < 0.0f) continue;

        //find max score up to position qnx:
        float max1 = 0.0f;
        #pragma omp simd reduction(max:max1)
        for(int ni = 0; ni < SCORDIMX; ni++)
            if(coords[SCORDIMX * lxOUTNDX + ni] < qnx)
                max1 = mymax(max1, coords[SCORDIMX * lxOUTMAX + ni]);

        //save max score to cache.
        //extremely simple hash function for the cache index:
        int c = (int)(qnx) & (SCORDIMX-1);
        float stqnx = coords[SCORDIMX * lxOUTNDX + c];//stored query position
        float stsco = coords[SCORDIMX * lxOUTMAX + c];//stored score
        float newsco = max1 + sco;//new score
        bool half2nd = (pi > (nalnposs>>1));
        //heuristics: under hash collision, update position and 
        //score wrt to which reference (query) half is under process:
        if(stqnx < 0.0f ||
          (stqnx == qnx && stsco < newsco) ||
          ((half2nd && stqnx < qnx) || (!half2nd && qnx < stqnx))) {
            coords[SCORDIMX * lxOUTNDX + c] = qnx;
            coords[SCORDIMX * lxOUTMAX + c] = newsco;
        }
    }

    //find max score over all query (reference) positions:
    float max1 = 0.0f;
    #pragma omp simd reduction(max:max1)
    for(int ni = 0; ni < SCORDIMX; ni++)
        max1 = mymax(max1, coords[SCORDIMX * lxOUTMAX + ni]);

    //write the max score to ccm:
    ccm[0][0] = max1;
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// Save2ndryScoreAndTM_Complete: complete version of saving secondary best 
// scores along with transformation matrices;
// N2SCTS, total number of secondary sections;
// NTOPTFMS, total number of best tfms within a section;
// SECTION2, secondary section index: 0 or 1;
// DATALN, data alignment;
// best, secondary best score so far;
// qryndx, query serial number;
// dbstrndx, reference serial number;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps to perform for each reference structure;
// nthreads, total number of threads running;
// sid, current fragment factor x fragment length index, corresponding to t.id;
// NOTE: memory pointers should be aligned!
// tfm, transformation matrix;
// wrkmemtmibest, working memory for iteration-best transformation matrices;
// 
template<int N2SCTS, int NTOPTFMS, int SECTION2, int DATALN>
inline
void MpStageFrg3::Save2ndryScoreAndTM_Complete(
    const float best,
    const int qryndx,
    const int dbstrndx,
    const int ndbCstrs,
    const int maxnsteps,
    const int nthreads,
    const int sid,
    const float* __RESTRICT__ tfm,
    float* __RESTRICT__ wrkmemtmibest)
{
    if(best <= 0.0f) return;

    float currentbest;
    const int nmaxscores = mymax((int)NTOPTFMS, nthreads);

    int mloc = ((qryndx * maxnsteps + nmaxscores * N2SCTS) * ndbCstrs) * nTTranformMatrix;

    currentbest = wrkmemtmibest[mloc + (nmaxscores * SECTION2 + sid) * ndbCstrs + dbstrndx];

    bool condition = (currentbest < best);

    if(condition)
        wrkmemtmibest[mloc + (nmaxscores * SECTION2 + sid) * ndbCstrs + dbstrndx] = best;

    //save transformation matrix
    if(condition) {
        mloc = ((qryndx * maxnsteps + (nmaxscores * (SECTION2+1) + sid)) * ndbCstrs + dbstrndx) * nTTranformMatrix;
        #pragma omp simd aligned(wrkmemtmibest:DATALN)
        for(int f = 0; f < nTTranformMatrix; f++) wrkmemtmibest[mloc + f] = tfm[f];
    }
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// SaveTopNScoresAndTMsAmongBests: save top N scores and respective 
// transformation matrices by considering all partial best scores 
// calculated over all fragment factors; write the information to the first
// N locations of fragment factors;
// NTOPTFMS, N scores and tfms;
// qryndx, query index;
// rfnblkndx, reference block index;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps performed for each reference structure;
// effnsteps, effective (actual maximum) number of steps;
// NOTE: memory pointers should be aligned!
// ccm, for local cache;
// wrkmemtmibest, working memory for iteration-best transformation matrices;
// wrkmemtm, working memory for selected transformation matrices;
// wrkmemaux, auxiliary working memory;
// 
template<int NTOPTFMS, int XDIM, int DATALN>
inline
void MpStageFrg3::SaveTopNScoresAndTMsAmongBests(
    const int qryndx,
    const int rfnblkndx,
    const int ndbCstrs,
    const int maxnsteps,
    const int effnsteps,
    float scoN[NTOPTFMS][XDIM],
    int ndxN[NTOPTFMS][XDIM],
    float* __RESTRICT__ wrkmemtmibest,
    float* __RESTRICT__ wrkmemtm,
    float* __RESTRICT__ wrkmemaux)
{
    const int istr0 = rfnblkndx * XDIM;
    const int istre = mymin(istr0 + XDIM, (int)ndbCstrs);

    for(int si = 0; si < NTOPTFMS && si < maxnsteps; si++) {
        int mloc = ((qryndx * maxnsteps + si/*sfragfct*/) * nTAuxWorkingMemoryVars) * ndbCstrs;
        #pragma omp simd aligned(wrkmemaux:DATALN)
        for(int ri = istr0; ri < istre; ri++) {
            int ii = ri - istr0;
            float bscore = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + ri/*dbstrndx*/];
            //cache scores:
            scoN[si][ii] = bscore;
            ndxN[si][ii] = si;
        }
    }

    //alternative/heuristics to partial sorting/nth_element:
    for(int si = NTOPTFMS; si < effnsteps; si++) {
        int sic = si & (NTOPTFMS - 1);//NTOPTFMS is a power of two
        int mloc = ((qryndx * maxnsteps + si/*sfragfct*/) * nTAuxWorkingMemoryVars) * ndbCstrs;
        #pragma omp simd aligned(wrkmemaux:DATALN)
        for(int ri = istr0; ri < istre; ri++) {
            int ii = ri - istr0;
            float bscore = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + ri/*dbstrndx*/];
            if(scoN[sic][ii] < bscore) {
                scoN[sic][ii] = bscore;
                ndxN[sic][ii] = si;
            }
        }
    }

    //scoN[.][...] contains maximums; write max values to the first slots
    for(int sic = 0; sic < NTOPTFMS && sic < maxnsteps; sic++) {
        #pragma omp simd aligned(wrkmemaux:DATALN)
        for(int ri = istr0; ri < istre; ri++) {
            int ii = ri - istr0;
            int si = ndxN[sic][ii];
            float bscore = scoN[sic][ii];
            int mloc = ((qryndx * maxnsteps + sic) * nTAuxWorkingMemoryVars) * ndbCstrs;
            //convergence
            int convflag = 0;
            if(sic == 0)
                convflag = wrkmemaux[mloc + tawmvConverged * ndbCstrs + ri];
            if(0.0f < bscore) convflag = convflag & (~CONVERGED_SCOREDP_bitval);//reset
            else convflag = convflag | CONVERGED_SCOREDP_bitval;//set
            //adjust global/local convergence
            wrkmemaux[mloc + tawmvConverged * ndbCstrs + ri] = convflag;
            //coalesced WRITE for multiple references
            if(sic != si)
                wrkmemaux[mloc + tawmvBestScore * ndbCstrs + ri] = bscore;
        }
    }

    //write best-performing tfms
    for(int sic = 0; sic < NTOPTFMS && sic < maxnsteps; sic++) {
        for(int ri = istr0; ri < istre; ri++) {
            int ii = ri - istr0;
            int si = ndxN[sic][ii];
            float bscore = scoN[sic][ii];
            if(bscore <= 0.0f) continue;
            int mlocs = ((qryndx * maxnsteps + si) * ndbCstrs + ri/*dbstrndx*/) * nTTranformMatrix;
            int mloct = ((qryndx * maxnsteps + sic) * ndbCstrs + ri/*dbstrndx*/) * nTTranformMatrix;
            //READ and WRITE best transformation matrices
            #pragma omp simd aligned(wrkmemtmibest,wrkmemtm:DATALN)
            for(int f = 0; f < nTTranformMatrix; f++)
                wrkmemtm[mloct + f] = wrkmemtmibest[mlocs + f];
        }
    }
}

// -------------------------------------------------------------------------
// SaveTopNScoresAndTMsAmongSecondaryBests: save secondary top N scores and 
// respective transformation matrices by considering all partial best scores 
// calculated over all fragment factors; write the information to the first
// N locations of fragment factors;
// N2SCTS, total number of secondary sections;
// NTOPTFMS, N scores and tfms;
// SECTION2, secondary section index: 0 or 1;
// qryndx, query index;
// rfnblkndx, reference block index;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps performed for each reference structure;
// effnsteps, effective (actual maximum) number of steps;
// NOTE: memory pointers should be aligned!
// ccm, for local cache;
// wrkmemtmibest, working memory for iteration-best transformation matrices;
// wrkmemtm, working memory for selected transformation matrices;
// wrkmemaux, auxiliary working memory;
// 
template<int N2SCTS, int NTOPTFMS, int SECTION2, int XDIM, int DATALN>
inline
void MpStageFrg3::SaveTopNScoresAndTMsAmongSecondaryBests(
    const int qryndx,
    const int rfnblkndx,
    const int ndbCstrs,
    const int maxnsteps,
    const int effnsteps,
    const int nthreads,
    float scoN[NTOPTFMS][XDIM],
    int ndxN[NTOPTFMS][XDIM],
    float* __RESTRICT__ wrkmemtmibest,
    float* __RESTRICT__ wrkmemtm,
    float* __RESTRICT__ wrkmemaux)
{
    const int istr0 = rfnblkndx * XDIM;
    const int istre = mymin(istr0 + XDIM, (int)ndbCstrs);
    const int nmaxscores = mymax((int)NTOPTFMS, nthreads);

    for(int si = 0; si < NTOPTFMS && si < maxnsteps; si++) {
        int mloc = ((qryndx * maxnsteps + nmaxscores * N2SCTS) * ndbCstrs) * nTTranformMatrix;
        #pragma omp simd aligned(wrkmemtmibest:DATALN)
        for(int ri = istr0; ri < istre; ri++) {
            int ii = ri - istr0;
            float bscore = wrkmemtmibest[mloc + (nmaxscores * SECTION2 + si) * ndbCstrs + ri];
            //cache scores:
            scoN[si][ii] = bscore;
            ndxN[si][ii] = si;
        }
    }

    //alternative/heuristics to partial sorting/nth_element:
    for(int si = NTOPTFMS; si < effnsteps; si++) {
        int sic = si & (NTOPTFMS - 1);//NTOPTFMS is a power of two
        int mloc = ((qryndx * maxnsteps + nmaxscores * N2SCTS) * ndbCstrs) * nTTranformMatrix;
        #pragma omp simd aligned(wrkmemtmibest:DATALN)
        for(int ri = istr0; ri < istre; ri++) {
            int ii = ri - istr0;
            float bscore = wrkmemtmibest[mloc + (nmaxscores * SECTION2 + si) * ndbCstrs + ri];
            if(scoN[sic][ii] < bscore) {
                scoN[sic][ii] = bscore;
                ndxN[sic][ii] = si;
            }
        }
    }

    //scoN[.][...] contains maximums; write max values to the first slots
    for(int sic = 0; sic < NTOPTFMS && sic < maxnsteps; sic++) {
        #pragma omp simd aligned(wrkmemaux:DATALN)
        for(int ri = istr0; ri < istre; ri++) {
            int ii = ri - istr0;
            // int si = ndxN[sic][ii];
            float bscore = scoN[sic][ii];
            int mloc = ((qryndx * maxnsteps + (NTOPTFMS * (SECTION2+1) + sic)) * nTAuxWorkingMemoryVars) * ndbCstrs;
            //convergence
            int convflag = 0;
            if(bscore <= 0.0f) convflag = CONVERGED_SCOREDP_bitval;//set
            //adjust local convergence
            wrkmemaux[mloc + tawmvConverged * ndbCstrs + ri] = convflag;
            //coalesced WRITE for multiple references (can be omitted)
            //wrkmemaux[mloc + tawmvBestScore * ndbCstrs + ri] = bscore;
        }
    }

    //write best-performing tfms
    for(int sic = 0; sic < NTOPTFMS && sic < maxnsteps; sic++) {
        for(int ri = istr0; ri < istre; ri++) {
            int ii = ri - istr0;
            int si = ndxN[sic][ii];
            float bscore = scoN[sic][ii];
            if(bscore <= 0.0f) continue;
            int mlocs = ((qryndx * maxnsteps + (nmaxscores * (SECTION2+1) + si)) * ndbCstrs + ri) * nTTranformMatrix;
            int mloct = ((qryndx * maxnsteps + (NTOPTFMS * (SECTION2+1) + sic)) * ndbCstrs + ri) * nTTranformMatrix;
            //READ and WRITE best transformation matrices
            #pragma omp simd aligned(wrkmemtmibest,wrkmemtm:DATALN)
            for(int f = 0; f < nTTranformMatrix; f++)
                wrkmemtm[mloct + f] = wrkmemtmibest[mlocs + f];
        }
    }
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// SortBestDPscoresAndTMsAmongDPswifts: sort best DP scores and then save 
// them along with respective transformation matrices by considering all 
// partial DP swift scores calculated over all fragment factors; write the 
// information to the first fragment factor locations;
// N2SCTS, total number of sections;
// NTOPTFMS, N scores and tfms;
// NMAXREFN, max number of branches for refinement (CUS1_TBSP_DPSCORE_TOP_N_REFINEMENT);
// SECTION, section index: 0, 1, or 2 (0..N2SCTS-1);
// nbranches, #final superposition-stage branches for further exploration
// (CUS1_TBSP_DPSCORE_TOP_N_REFINEMENT for max);
// qryndx, query index;
// rfnblkndx, reference block index;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db reference structure positions in a chunk;
// dbxpad, number of padded positions for memory alignment;
// maxnsteps, max number of steps performed for each reference structure;
// NOTE: memory pointers should be aligned!
// scoN, ndxN, working arrays: scores and indices;
// tmpdpdiagbuffers, memory section of DP scores;
// wrkmemtm, input working memory of calculated transformation matrices;
// wrkmemtmtarget, working memory for iteration-best (target) transformation matrices;
// wrkmemaux, auxiliary working memory;
//
template<int N2SCTS, int NTOPTFMS, int NMAXREFN, int XDIM, int DATALN>
inline
void MpStageFrg3::SortBestDPscoresAndTMsAmongDPswifts(
    const int SECTION, 
    const int nbranches,
    const int qryndx,
    const int rfnblkndx,
    const int ndbCstrs,
    const int ndbCposs,
    const int dbxpad,
    const int maxnsteps,
    float scoN[XDIM][NTOPTFMS],
    int ndxN[XDIM][NTOPTFMS],
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* const __RESTRICT__ tmpdpdiagbuffers,
    const float* const __RESTRICT__ wrkmemtm,
    float* const __RESTRICT__ wrkmemtmtarget,
    float* const __RESTRICT__ wrkmemaux)
{
    enum {lNTOPRFNS = N2SCTS * NMAXREFN};

    const int istr0 = rfnblkndx * XDIM;
    const int istre = mymin(istr0 + XDIM, (int)ndbCstrs);
    const int dblen = ndbCposs + dbxpad;

    for(int sih = 0; sih < NTOPTFMS; sih++) {
        int si = SECTION * NTOPTFMS + sih;
        if(maxnsteps <= si) break;
        int yofff = (qryndx * maxnsteps + si) * dblen;
        int doffs = nTDPDiagScoreSections * nTDPDiagScoreSubsections * yofff;
        int mloc = ((qryndx * maxnsteps + si) * nTAuxWorkingMemoryVars) * ndbCstrs;
        #pragma omp simd aligned(bdbCpmbeg,wrkmemaux,tmpdpdiagbuffers:DATALN)
        for(int ri = istr0; ri < istre; ri++) {
            int ii = ri - istr0;
            float dpscore = 0.0f;
            const int dbstrlen = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
            const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);
            int convflag = wrkmemaux[mloc + tawmvConverged * ndbCstrs + ri];
            if(convflag == 0)
                dpscore = tmpdpdiagbuffers[doffs + dbstrdst + dbstrlen - 1];
            //cache scores:
            scoN[ii][sih] = dpscore;
            ndxN[ii][sih] = sih;
        }
    }

    //sort scores and accompanying indices in descending order:
    //NOTE: assert nbranches <= NTOPTFMS!
    for(int ri = istr0; ri < istre; ri++) {
        int ii = ri - istr0;
        std::nth_element(ndxN[ii], ndxN[ii] + nbranches, ndxN[ii] + NTOPTFMS,
            [scoN,ii](int i, int j) {return scoN[ii][i] > scoN[ii][j];}
        );
    }

    //scoN[ndxN[.]][...] contains maximums; update convergence in the first slots
    for(int sih = 0; sih < NTOPTFMS; sih++) {
        int si = SECTION * NTOPTFMS + sih;
        if(maxnsteps <= si) break;
        int mloc = ((qryndx * maxnsteps + si) * nTAuxWorkingMemoryVars) * ndbCstrs;
        #pragma omp simd aligned(wrkmemaux:DATALN)
        for(int ri = istr0; ri < istre; ri++) {
            int ii = ri - istr0;
            float dpscore = scoN[ii][ ndxN[ii][sih] ];
            //convergence
            int convflag = 0;
            if(dpscore <= 0.0f) convflag = CONVERGED_SCOREDP_bitval;//set
            //adjust local convergence
            wrkmemaux[mloc + tawmvConverged * ndbCstrs + ri] = convflag;
            //coalesced WRITE for multiple references (for test)
            //wrkmemaux[mloc + tawmvScore * ndbCstrs + ri] = dpscore;
        }
    }

    //write best-performing tfms
    //READ and WRITE iteration-best transformation matrices;
    //rearrange nbranches best performing tfms at the first slots (si indices)
    for(int sih = 0; sih < nbranches; sih++) {
        int si = SECTION * NTOPTFMS + sih;
        if(maxnsteps <= si) break;
        for(int ri = istr0; ri < istre; ri++) {
            int ii = ri - istr0;
            int ssih = ndxN[ii][sih];
            int ssi = SECTION * NTOPTFMS + ssih;
            float bscore = scoN[ii][ssih];
            if(bscore <= 0.0f) continue;
            int mlocs = ((qryndx * maxnsteps + ssi) * ndbCstrs + ri) * nTTranformMatrix;
            //NOTE: lNTOPRFNS for target tms!
            int mloct = ((qryndx * lNTOPRFNS + (SECTION * nbranches + sih)) * ndbCstrs + ri) * nTTranformMatrix;
            //READ and WRITE best transformation matrices
            #pragma omp simd aligned(wrkmemtmtarget,wrkmemtm:DATALN)
            for(int f = 0; f < nTTranformMatrix; f++)
                wrkmemtmtarget[mloct + f] = wrkmemtm[mlocs + f];
        }
    }
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// NNByIndexQuery: find query node nearest to the atom with the given coordinates;
// SECSTRFILT, template parameter, find nearest neighbour with matching ss;
// STACKSIZE, dynamically determined stack size;
// nestndx, index of the nearest node;
// qxn, qyn, qzn, coordinates of the nearest atom;
// rx, ry, rz, coordinates of the atom searched for;
// rss, reference secondary structure at the position under process;
// qrydst, beginning address of a query structure;
// root, root node index;
// dimndx, starting dimension for searching in the index;
// stack, stack for traversing the index tree iteratively;
//
template<int SECSTRFILT>
inline
void MpStageFrg3::NNByIndexQuery(
    int STACKSIZE,
    int& nestndx,
    float& qxn, float& qyn, float& qzn, 
    float rx, float ry, float rz, char rss,
    int qrydst, int root, int dimndx,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ /* bdbCpmbeg */,
    const char* const * const __RESTRICT__ queryndxpmbeg,
    const char* const * const __RESTRICT__ /* bdbCndxpmbeg */,
    float* const __RESTRICT__ stack)
{
    int stackptr = 0;//stack pointer
    int nvisited = 0;//number of visited nodes
    float nestdst = 9.9e6f;//squared best distance

    //while stack is nonempty or the current node is not a terminator
    while(0 < stackptr || 0 <= root)
    {
        if(0 <= root) {
            nvisited++;
            if(CUSF_TBSP_INDEX_SCORE_MAXDEPTH <= nvisited)
                break;
            int qryorgndx;
            char qss;
            if(SECSTRFILT == 1) {
                //map the index in the index tree to the original structure position index
                qryorgndx = PMBatchStrDataIndex::GetFieldAt<int,pmv2DNdxOrgndx>(queryndxpmbeg, qrydst + root);
                qss = PMBatchStrData::GetFieldAt<char,pmv2Dss>(querypmbeg, qrydst + qryorgndx);
            }
            //READ coordinates:
            float qx = PMBatchStrDataIndex::GetFieldAt<float,pmv2DNdxCoords+pmv2DX>(queryndxpmbeg, qrydst + root);
            float qy = PMBatchStrDataIndex::GetFieldAt<float,pmv2DNdxCoords+pmv2DY>(queryndxpmbeg, qrydst + root);
            float qz = PMBatchStrDataIndex::GetFieldAt<float,pmv2DNdxCoords+pmv2DZ>(queryndxpmbeg, qrydst + root);
            float dst2 = distance2(qx, qy, qz,  rx, ry, rz);
            if((nestndx < 0 || dst2 < nestdst) 
                //&& ((SECSTRFILT == 1)? !helix_strnd(rss, qss): 1)
                && ((SECSTRFILT == 1)? (rss == qss): 1)
            ) {
                nestdst = dst2;
                qxn = qx; qyn = qy; qzn = qz;
                if(SECSTRFILT == 1) nestndx = qryorgndx;
                else nestndx = root;
            }
            if(nestdst == 0.0f)
                break;
            float diffc = (dimndx == 0)? (qx - rx): ((dimndx == 1)? (qy - ry): (qz - rz));
            if(pmv2DZ < ++dimndx) dimndx = 0;
            if(stackptr < STACKSIZE) {
                stack[nStks_ * stackptr + stkNdx_Dim_] = NNSTK_COMBINE_NDX_DIM(root, dimndx);
                stack[nStks_ * stackptr + stkDiff_] = diffc;
                stackptr++;
            }
            root = (0.0f < diffc)//READ
                ? PMBatchStrDataIndex::GetFieldAt<int,pmv2DNdxLeft>(queryndxpmbeg, qrydst + root)
                : PMBatchStrDataIndex::GetFieldAt<int,pmv2DNdxRight>(queryndxpmbeg, qrydst + root);
        }
        else {
            bool cond = false;
            float diffc = 0.0f;
            for(; 0 < stackptr && !cond;) {
                stackptr--;
                int comb = stack[nStks_ * stackptr + stkNdx_Dim_];
                diffc = stack[nStks_ * stackptr + stkDiff_];
                root = NNSTK_GET_NDX_FROM_COMB(comb);
                dimndx = NNSTK_GET_DIM_FROM_COMB(comb);
                cond = (diffc * diffc < nestdst);
            }
            if(!cond) break;
            root = (0.0f < diffc)//READ
                ? PMBatchStrDataIndex::GetFieldAt<int,pmv2DNdxRight>(queryndxpmbeg, qrydst + root)
                : PMBatchStrDataIndex::GetFieldAt<int,pmv2DNdxLeft>(queryndxpmbeg, qrydst + root);
        }
    }

    if(SECSTRFILT == 1) {
        if(64.0f < nestdst) nestndx = -1;
    } else {
        //map the index in the index tree to the original structure position index
        if(0 <= nestndx)
            nestndx = PMBatchStrDataIndex::GetFieldAt<int,pmv2DNdxOrgndx>(queryndxpmbeg, qrydst + nestndx);
    }
}

// -------------------------------------------------------------------------
// NNByIndexReference version for reference
//
template<int SECSTRFILT>
inline
void MpStageFrg3::NNByIndexReference(
    int STACKSIZE,
    int& nestndx,
    float& rxn, float& ryn, float& rzn, 
    float qx, float qy, float qz, char qss,
    int dbstrdst, int root, int dimndx,
    const char* const * const __RESTRICT__ /* querypmbeg */,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const char* const * const __RESTRICT__ /* queryndxpmbeg */,
    const char* const * const __RESTRICT__ bdbCndxpmbeg,
    float* const __RESTRICT__ stack)
{
    int stackptr = 0;//stack pointer
    int nvisited = 0;//number of visited nodes
    float nestdst = 9.9e6f;//squared best distance

    //while stack is nonempty or the current node is not a terminator
    while(0 < stackptr || 0 <= root)
    {
        if(0 <= root) {
            nvisited++;
            if(CUSF_TBSP_INDEX_SCORE_MAXDEPTH <= nvisited)
                break;
            int rfnorgndx;
            char rss;
            if(SECSTRFILT == 1) {
                //map the index in the index tree to the original structure position index
                rfnorgndx = PMBatchStrDataIndex::GetFieldAt<int,pmv2DNdxOrgndx>(bdbCndxpmbeg, dbstrdst + root);
                rss = PMBatchStrData::GetFieldAt<char,pmv2Dss>(bdbCpmbeg, dbstrdst + rfnorgndx);
            }
            //READ coordinates:
            float rx = PMBatchStrDataIndex::GetFieldAt<float,pmv2DNdxCoords+pmv2DX>(bdbCndxpmbeg, dbstrdst + root);
            float ry = PMBatchStrDataIndex::GetFieldAt<float,pmv2DNdxCoords+pmv2DY>(bdbCndxpmbeg, dbstrdst + root);
            float rz = PMBatchStrDataIndex::GetFieldAt<float,pmv2DNdxCoords+pmv2DZ>(bdbCndxpmbeg, dbstrdst + root);
            float dst2 = distance2(qx, qy, qz,  rx, ry, rz);
            if((nestndx < 0 || dst2 < nestdst) 
                //&& ((SECSTRFILT == 1)? !helix_strnd(rss, qss): 1)
                && ((SECSTRFILT == 1)? (rss == qss): 1)
            ) {
                nestdst = dst2;
                rxn = rx; ryn = ry; rzn = rz;
                if(SECSTRFILT == 1) nestndx = rfnorgndx;
                else nestndx = root;
            }
            if(nestdst == 0.0f)
                break;
            float diffc = (dimndx == 0)? (rx - qx): ((dimndx == 1)? (ry - qy): (rz - qz));
            if(pmv2DZ < ++dimndx) dimndx = 0;
            if(stackptr < STACKSIZE) {
                stack[nStks_ * stackptr + stkNdx_Dim_] = NNSTK_COMBINE_NDX_DIM(root, dimndx);
                stack[nStks_ * stackptr + stkDiff_] = diffc;
                stackptr++;
            }
            root = (0.0f < diffc)//READ
                ? PMBatchStrDataIndex::GetFieldAt<int,pmv2DNdxLeft>(bdbCndxpmbeg, dbstrdst + root)
                : PMBatchStrDataIndex::GetFieldAt<int,pmv2DNdxRight>(bdbCndxpmbeg, dbstrdst + root);
        }
        else {
            bool cond = false;
            float diffc = 0.0f;
            for(; 0 < stackptr && !cond;) {
                stackptr--;
                int comb = stack[nStks_ * stackptr + stkNdx_Dim_];
                diffc = stack[nStks_ * stackptr + stkDiff_];
                root = NNSTK_GET_NDX_FROM_COMB(comb);
                dimndx = NNSTK_GET_DIM_FROM_COMB(comb);
                cond = (diffc * diffc < nestdst);
            }
            if(!cond) break;
            root = (0.0f < diffc)//READ
                ? PMBatchStrDataIndex::GetFieldAt<int,pmv2DNdxRight>(bdbCndxpmbeg, dbstrdst + root)
                : PMBatchStrDataIndex::GetFieldAt<int,pmv2DNdxLeft>(bdbCndxpmbeg, dbstrdst + root);
        }
    }

    if(SECSTRFILT == 1) {
        if(64.0f < nestdst) nestndx = -1;
    } else {
        //map the index in the index tree to the original structure position index
        if(0 <= nestndx)
            nestndx = PMBatchStrDataIndex::GetFieldAt<int,pmv2DNdxOrgndx>(bdbCndxpmbeg, dbstrdst + nestndx);
    }
}

// -------------------------------------------------------------------------

#endif//__MpStageFrg3_h__
