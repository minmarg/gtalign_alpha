/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/covariance.cuh"
#include "libmycu/custages/covariance_plus.cuh"
#include "libmycu/custages/covariance_refn.cuh"
#include "libmycu/custages/covariance_dp_refn.cuh"
#include "libmycu/custages2/covariance_complete.cuh"
#include "libmycu/custages2/covariance_refn_complete.cuh"
#include "libmycu/custages2/covariance_dp_refn_complete.cuh"
#include "covariance_production_dp_refn_complete.cuh"

// =========================================================================
// ProductionFragmentBasedDPAlignmentRefinementPhase1: phase 1 to perform 
// production-version tuning of the the best superposition obtained 
// within the single kernel's actions; write the final superposition scores;
// TFM_DINV, use doubly inverted transformation matrices under suitable conditions;
// nmaxconvit, maximum number of superposition iterations;
// nqystrs, total number of query structures in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of reference positions in the chunk;
// dbxpad, #pad positions along the dimension of reference structures;
// nmaxsubfrags, total number of fragment lengths to consider;
// maxnsteps, total number of steps that should be performed for each reference structure;
// sfragstep, step size to traverse subfragments;
// maxalnmax, maximum alignment length across all query-reference pairs;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// tmpdpdiagbuffers, temporary diagonal buffers for positional scores;
// wrkmemtmibest, working memory for best-performing transformation matrices;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// 
template<bool TFM_DINV>
__global__ 
void ProductionFragmentBasedDPAlignmentRefinementPhase1(
    const int nmaxconvit,
//     const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint nmaxsubfrags,
    const uint maxnsteps,
    const int sfragstep,
    const int /*maxalnmax*/,
    const float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ alndatamem)
{
    uint dbstrndx = blockIdx.x;//reference serial number
    uint sfragfctxndx = blockIdx.y;//fragment factor x fragment length index
    uint sfragfct = sfragfctxndx / nmaxsubfrags;//fragment factor
    uint sfragndx = sfragfctxndx - sfragfct * nmaxsubfrags;//fragment length index
    uint qryndx = blockIdx.z;//query serial number
    //cache for the cross-covarinace matrix and related data: 
    //no bank conflicts as long as inner-most dim is odd
    enum {neffds = twmvEndOfCCDataExt,//effective number of fields
        smidim = twmvEndOfCCDataExtPlus};
    __shared__ float ccmCache[
        smidim * CUS1_TBINITSP_COMPLETEREFINE_XDIM + twmvEndOfCCDataExt * 2 + nTTranformMatrix];
//     __shared__ float ccmLast[twmvEndOfCCDataExt];
//     __shared__ float tfmCache[twmvEndOfCCDataExt];//twmvEndOfCCDataExt>nTTranformMatrix
    float* ccmLast = ccmCache + smidim * CUS1_TBINITSP_COMPLETEREFINE_XDIM;
    float* tfmCache = ccmLast + twmvEndOfCCDataExt;//twmvEndOfCCDataExt>nTTranformMatrix
    float* tfmBest = tfmCache + twmvEndOfCCDataExt;//of size nTTranformMatrix

    int qrylenorg, dbstrlenorg;//original query and reference lengths
    int qrylen, dbstrlen;//pseudo query and reference length, #matched positions
    //distances in positions to the beginnings of the query and reference structures:
    uint /*qrydst, */dbstrdst;
    enum {qrypos = 0, rfnpos = 0};
    int sfragpos, fraglen;//fragment position and length


    if(threadIdx.x == 0) {
        uint mloc = ((qryndx * maxnsteps + 0/*sfragfctxndx*/) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        ccmCache[6] = wrkmemaux[mloc + dbstrndx];
    }

    __syncthreads();

    if(((int)(ccmCache[6])) & (CONVERGED_LOWTMSC_bitval))
        //(the termination flag for this pair is set);
        //all threads in the block exit;
        return;

    //NOTE: no sync as long ccmCache cell for convergence is not overwritten;


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse ccmCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(dbstrndx, (int*)ccmCache);
        //GetQueryLenDst(qryndx, (int*)ccmCache + 2);
        if(threadIdx.x == 0) ((int*)ccmCache)[2] = GetQueryLength(qryndx);
    }

    //NOTE: use a different warp for structure-specific-formatted data;
#if (CUS1_TBINITSP_COMPLETEREFINE_XDIM >= 64)
    if(threadIdx.x == tawmvNAlnPoss + 32) {
#else
    if(threadIdx.x == tawmvNAlnPoss) {
#endif
        //NOTE: reuse ccmCache to read values written at sfragfct==0:
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        ccmCache[threadIdx.x] = wrkmemaux[mloc0 + threadIdx.x * ndbCstrs + dbstrndx];
    }

    __syncthreads();


    dbstrdst = ((int*)ccmCache)[1];
    //qrydst = ((int*)ccmCache)[3];
    qrylen = dbstrlen = ccmCache[tawmvNAlnPoss];
    sfragpos = sfragfct * sfragstep;
    dbstrlenorg = ((int*)ccmCache)[0];
    qrylenorg = ((int*)ccmCache)[2];

    __syncthreads();


    fraglen = GetFragLength(qrylen, dbstrlen, qrypos, rfnpos, sfragndx);
    if(fraglen < 1) 
        //all threads in the block exit
        return;

    if(qrylen + sfragstep <= qrypos + sfragpos + fraglen ||
       dbstrlen + sfragstep <= rfnpos + sfragpos + fraglen)
        //all threads in the block exit
        return;


    //threshold calculated for the original lengths
    const float d0 = GetD0fin(qrylenorg, dbstrlenorg);
    const float d02 = SQRD(d0);
    const float d82 = GetD82(qrylenorg, dbstrlenorg);
    float dst32 = CP_LARGEDST;
    float best = -1.0f;//best score obtained


    //calculate rmsd when the fragment being processed represents full alignment
    if(sfragfctxndx == 0) {//also implies sfragndx == 0
        CalcExtCCMatrices64_DPRefined_Complete<smidim,neffds>(
            qryndx,  ndbCposs, dbxpad,  maxnsteps, sfragfctxndx, dbstrdst, fraglen,
            qrylen, dbstrlen,  qrypos + sfragpos, rfnpos + sfragpos,
            tmpdpalnpossbuffer, ccmCache);
        //synced above; save the original ccm data:
        CopyCCMDataToTFM_Complete(ccmCache, ccmLast);
        __syncwarp();
        float rmsd = CalcRMSD_Complete(ccmCache);
        //write rmsd to gmem
        uint mloc = (qryndx * ndbCstrs + dbstrndx) * nTDP2OutputAlnData;
        if(threadIdx.x == 0) alndatamem[mloc + dp2oadRMSD] = rmsd;//WRITE
        //restore the original ccm data:
        CopyCCMDataToTFM_Complete(ccmLast, ccmCache);
        __syncwarp();
    }
    else
        CalcCCMatrices64_DPRefined_Complete<smidim,neffds>(
            qryndx,  ndbCposs, dbxpad,  maxnsteps, sfragfctxndx, dbstrdst, fraglen,
            qrylen, dbstrlen,  qrypos + sfragpos, rfnpos + sfragpos,
            tmpdpalnpossbuffer, ccmCache);


    for(int cit = 0; cit < nmaxconvit + 2; cit++)
    {
        if(0 < cit) {
            CalcCCMatrices64_DPRefinedExtended_Complete<smidim,neffds>(
                (cit < 2)? READCNST_CALC: READCNST_CALC2,
                qryndx, ndbCposs, dbxpad, maxnsteps, sfragfctxndx, dbstrdst,
                qrylen, dbstrlen, qrypos, rfnpos,  d0, dst32,
                tmpdpdiagbuffers, tmpdpalnpossbuffer, ccmCache);

            CheckConvergence64Refined_Complete(ccmCache, ccmLast);
            if(ccmLast[0]) break;//converged
            __syncthreads();//prevent overwriting ccmLast[0]
        }

        //NOTE: synced above and below before ccmCache gets updated;
        if(ccmCache[twmvNalnposs] < 1.0f) break;

        SaveCCMData_Complete(ccmCache, tfmCache, ccmLast);
        //NOTE: tfmCache updated by the first warp; 
        //NOTE: CalcTfmMatrices_Complete uses only the first warp;
        //NOTE: ccmLast not used until the first syncthreads below;
        __syncwarp();

        CalcTfmMatrices_Complete<TFM_DINV>(tfmCache, qrylenorg, dbstrlenorg);
        //all threads synced and see the tfm

        CalcScoresUnrl_DPRefined_Complete<CHCKDST_NOCHECK>(
            (cit < 1)? READCNST_CALC: READCNST_CALC2,
            qryndx, ndbCposs, dbxpad, maxnsteps, sfragfctxndx, dbstrdst,
            qrylen, dbstrlen, qrypos, rfnpos,  d0, d02, d82,
            tmpdpdiagbuffers, tmpdpalnpossbuffer, tfmCache, ccmCache+1);

        //distance threshold for at least three aligned pairs:
        dst32 = ccmCache[2];

        //NOTE: no sync inside:
        SaveLocalBestScoreAndTM(best, ccmCache[1]/*score*/, tfmCache, tfmBest);

        //sync all threads to see dst32 (and prevent overwriting the cache):
        __syncthreads();
    }

    //calculate the score for the larger structure of the two:
    //threshold calculated for the greater length
    const int greaterlen = myhdmax(qrylenorg, dbstrlenorg);
    const float g0 = GetD0fin(greaterlen, greaterlen);
    const float g02 = SQRD(g0);
    float gbest = best;//score calculated for the other structure

    if(qrylenorg != dbstrlenorg) {
        CalcScoresUnrl_DPRefined_Complete<CHCKDST_NOCHECK>(
            READCNST_CALC2,
            qryndx, ndbCposs, dbxpad, maxnsteps, sfragfctxndx, dbstrdst,
            qrylen, dbstrlen, qrypos, rfnpos,  g0, g02, d82,
            tmpdpdiagbuffers, tmpdpalnpossbuffer, tfmBest, ccmCache+1);
        gbest = ccmCache[1];//score
    }

    //NOTE: synced either after the last cit or convergence check:
    SaveBestQRScoresAndTM_Complete<true/* WRITEFRAGINFO */,false/* CONDITIONAL */>(
        best, gbest,  qryndx, dbstrndx, ndbCstrs, 
        maxnsteps, sfragfctxndx, sfragndx, sfragpos,
        tfmBest, wrkmemtmibest, wrkmemaux);
}

// -------------------------------------------------------------------------
// Instantiations
//
#define INSTANTIATE_ProductionFragmentBasedDPAlignmentRefinementPhase1(tpTFM_DINV) \
    template __global__ void ProductionFragmentBasedDPAlignmentRefinementPhase1<tpTFM_DINV>( \
        const int nmaxconvit,const uint ndbCstrs,const uint ndbCposs,const uint dbxpad, \
        const uint nmaxsubfrags,const uint maxnsteps,const int sfragstep,const int /*maxalnmax*/, \
        const float* __restrict__ tmpdpalnpossbuffer, \
        float* __restrict__ tmpdpdiagbuffers, \
        float* __restrict__ wrkmemtmibest, \
        float* __restrict__ wrkmemaux, \
        float* __restrict__ alndatamem);

INSTANTIATE_ProductionFragmentBasedDPAlignmentRefinementPhase1(false);
INSTANTIATE_ProductionFragmentBasedDPAlignmentRefinementPhase1(true);

// -------------------------------------------------------------------------



// =========================================================================
// ProductionRefinementPhase2InnerLoop: inner loop for 
// ProductionFragmentBasedDPAlignmentRefinementPhase2 variants;
// sfragfctxndx, fragment factor x fragment length index;
// qryndx, query serial number;
// nmaxconvit, maximum number of superposition iterations;
// ndbCposs, total number of reference positions in the chunk;
// dbxpad, #pad positions along the dimension of reference structures;
// maxnsteps, total number of steps that should be performed for each reference structure;
// qrylenorg, dbstrlenorg, original query and reference lengths;
// qrylen, dbstrlen, pseudo query and reference length, #matched positions;
// dbstrdst, distance in positions to the beginnings of the reference structures
// qrypos, rfnpos, query and reference starting positions in alignment (0);
// sfragpos, fraglen, fragment position and length;
// d0, d02, d82, distance thresholds;
// best, best score so far;
// ccmCache, ccmLast, tfmCache, tfmBest, smem working cache;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// tmpdpdiagbuffers, temporary diagonal buffers for positional scores;
//
template<bool TFM_DINV, int SMIDIM, int NEFFDS>
__device__ __forceinline__
void ProductionRefinementPhase2InnerLoop(
    const uint sfragfctxndx,
    const uint qryndx,
    const int nmaxconvit,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const int qrylenorg, const int dbstrlenorg,
    const int qrylen, const int dbstrlen,
    const uint dbstrdst,
    const int qrypos, const int rfnpos,
    const int sfragpos, const int fraglen,
    const float d0, const float d02, const float d82,
    float& best,
    //NOTE: remove __restrict__ here: latest compilers' memory optimizations
    float* /* __restrict__ */ ccmCache,
    float* /* __restrict__ */ ccmLast,
    float* /* __restrict__ */ tfmCache,
    float* /* __restrict__ */ tfmBest,
    const float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers)
{
    float dst32 = CP_LARGEDST;

    CalcCCMatrices64_DPRefined_Complete<SMIDIM,NEFFDS>(
        qryndx,  ndbCposs, dbxpad,  maxnsteps, sfragfctxndx, dbstrdst, fraglen,
        qrylen, dbstrlen,  qrypos + sfragpos, rfnpos + sfragpos,
        tmpdpalnpossbuffer, ccmCache);

    for(int cit = 0; cit < nmaxconvit + 2; cit++)
    {
        if(0 < cit) {
            CalcCCMatrices64_DPRefinedExtended_Complete<SMIDIM,NEFFDS>(
                (cit < 2)? READCNST_CALC: READCNST_CALC2,
                qryndx, ndbCposs, dbxpad, maxnsteps, sfragfctxndx, dbstrdst,
                qrylen, dbstrlen, qrypos, rfnpos,  d0, dst32,
                tmpdpdiagbuffers, tmpdpalnpossbuffer, ccmCache);

            CheckConvergence64Refined_Complete(ccmCache, ccmLast);
            if(ccmLast[0]) break;//converged
            __syncthreads();//prevent overwriting ccmLast[0]
        }

        //NOTE: synced above and below before ccmCache gets updated;
        if(ccmCache[twmvNalnposs] < 1.0f) break;

        SaveCCMData_Complete(ccmCache, tfmCache, ccmLast);
        //NOTE: tfmCache updated by the first warp; 
        //NOTE: CalcTfmMatrices_Complete uses only the first warp;
        //NOTE: ccmLast not used until the first syncthreads below;
        __syncwarp();

        CalcTfmMatrices_Complete<TFM_DINV>(tfmCache, qrylenorg, dbstrlenorg);
        //all threads synced and see the tfm

        CalcScoresUnrl_DPRefined_Complete<CHCKDST_NOCHECK>(
            (cit < 1)? READCNST_CALC: READCNST_CALC2,
            qryndx, ndbCposs, dbxpad, maxnsteps, sfragfctxndx, dbstrdst,
            qrylen, dbstrlen, qrypos, rfnpos,  d0, d02, d82,
            tmpdpdiagbuffers, tmpdpalnpossbuffer, tfmCache, ccmCache+1);

        //distance threshold for at least three aligned pairs:
        dst32 = ccmCache[2];

        //NOTE: no sync inside:
        SaveLocalBestScoreAndTM(best, ccmCache[1]/*score*/, tfmCache, tfmBest);

        //sync all threads to see dst32 (and prevent overwriting the cache):
        __syncthreads();
    }
}

// =========================================================================
// ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearch: phase 2 to
// perform production-version tuning of the the best superposition obtained 
// within the single kernel's actions; write the final superposition scores;
// this version performs full search of maxnfragfcts positions from the 
// identified one in phase 1 for each fragment length;
// TFM_DINV, use doubly inverted transformation matrices under suitable conditions;
// nmaxconvit, maximum number of superposition iterations;
// nqystrs, total number of query structures in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of reference positions in the chunk;
// dbxpad, #pad positions along the dimension of reference structures;
// nmaxsubfrags, total number of fragment lengths to consider;
// maxnfragfcts, max number of fragment position factors around an identified position;
// maxnsteps, total number of steps that should be performed for each reference structure;
// sfragstep, step size to traverse subfragments;
// maxalnmax, maximum alignment length across all query-reference pairs;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// tmpdpdiagbuffers, temporary diagonal buffers for positional scores;
// wrkmemtmibest, working memory for best-performing transformation matrices;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// 
template<bool TFM_DINV>
__global__ 
void ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearch(
    const int nmaxconvit,
//     const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint nmaxsubfrags,
    const uint maxnfragfcts,
    const uint maxnsteps,
    const int sfragstep,
    const int /*maxalnmax*/,
    const float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux)
{
    uint dbstrndx = blockIdx.x;//reference serial number
    const uint sfragfctxndx = blockIdx.y;//fragment factor x fragment length index
    const uint sfragfct = sfragfctxndx / nmaxsubfrags;//fragment factor
    uint sfragndx = sfragfctxndx - sfragfct * nmaxsubfrags;//fragment length index
    uint qryndx = blockIdx.z;//query serial number
    //cache for the cross-covarinace matrix and related data: 
    //no bank conflicts as long as inner-most dim is odd
    enum {neffds = twmvEndOfCCDataExt,//effective number of fields
        smidim = neffds + 1};
    __shared__ float ccmCache[
        smidim * CUS1_TBINITSP_COMPLETEREFINE_XDIM + twmvEndOfCCDataExt * 2 + nTTranformMatrix];
//     __shared__ float ccmLast[twmvEndOfCCDataExt];
//     __shared__ float tfmCache[twmvEndOfCCDataExt];//twmvEndOfCCDataExt>nTTranformMatrix
    float* ccmLast = ccmCache + smidim * CUS1_TBINITSP_COMPLETEREFINE_XDIM;
    float* tfmCache = ccmLast + twmvEndOfCCDataExt;//twmvEndOfCCDataExt>nTTranformMatrix
    float* tfmBest = tfmCache + twmvEndOfCCDataExt;//of size nTTranformMatrix

    int qrylenorg, dbstrlenorg;//original query and reference lengths
    int qrylen, dbstrlen;//pseudo query and reference length, #matched positions
    //distances in positions to the beginnings of the query and reference structures:
    uint /*qrydst, */dbstrdst;
    enum {qrypos = 0, rfnpos = 0};
    int sfragpos, fraglen;//fragment position and length


    if(threadIdx.x == 0) {
        uint mloc = ((qryndx * maxnsteps + 0/*sfragfctxndx*/) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        ccmCache[6] = wrkmemaux[mloc + dbstrndx];
    }

    __syncthreads();

    if(((int)(ccmCache[6])) & (CONVERGED_LOWTMSC_bitval))
        //(the termination flag for this pair is set);
        //all threads in the block exit;
        return;

    //NOTE: no sync as long ccmCache cell for convergence is not overwritten;


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse ccmCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(dbstrndx, (int*)ccmCache);
        //GetQueryLenDst(qryndx, (int*)ccmCache + 2);
        if(threadIdx.x == 0) ((int*)ccmCache)[2] = GetQueryLength(qryndx);
    }

    //NOTE: use a different warp for structure-specific-formatted data;
#if (CUS1_TBINITSP_COMPLETEREFINE_XDIM >= 64)
    if(threadIdx.x == tawmvNAlnPoss + 32 ||
       threadIdx.x == tawmvSubFragPos + 32/* || threadIdx.x == tawmvSubFragNdx + 32 */) {
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        ccmCache[threadIdx.x-32] = wrkmemaux[mloc0 + (threadIdx.x-32) * ndbCstrs + dbstrndx];
#else
    if(threadIdx.x == tawmvNAlnPoss || threadIdx.x == tawmvSubFragPos/* || threadIdx.x == tawmvSubFragNdx */) {
        //NOTE: reuse ccmCache to read values written at sfragfct==0:
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        ccmCache[threadIdx.x] = wrkmemaux[mloc0 + threadIdx.x * ndbCstrs + dbstrndx];
#endif
    }

    __syncthreads();


    dbstrdst = ((int*)ccmCache)[1];
    //qrydst = ((int*)ccmCache)[3];
    qrylen = dbstrlen = ccmCache[tawmvNAlnPoss];
    sfragpos = (int)(ccmCache[tawmvSubFragPos]) + ((int)sfragfct - (int)(maxnfragfcts>>1)) * sfragstep;
    // sfragndx = (int)(ccmCache[tawmvSubFragNdx]);
    dbstrlenorg = ((int*)ccmCache)[0];
    qrylenorg = ((int*)ccmCache)[2];

    __syncthreads();


    fraglen = GetFragLength(qrylen, dbstrlen, qrypos, rfnpos, sfragndx);
    if(fraglen < 1) 
        //all threads in the block exit
        return;

    if(sfragpos < 0 ||
       qrylen + sfragstep <= qrypos + sfragpos + fraglen ||
       dbstrlen + sfragstep <= rfnpos + sfragpos + fraglen)
        //all threads in the block exit
        return;


    //threshold calculated for the original lengths
    const float d0 = GetD0fin(qrylenorg, dbstrlenorg);
    const float d02 = SQRD(d0);
    const float d82 = GetD82(qrylenorg, dbstrlenorg);
    float best = -1.0f;//best score obtained


    ProductionRefinementPhase2InnerLoop<TFM_DINV, smidim, neffds>(
        sfragfctxndx, qryndx,
        nmaxconvit, ndbCposs, dbxpad, maxnsteps,
        qrylenorg, dbstrlenorg, qrylen, dbstrlen, dbstrdst,
        qrypos, rfnpos, sfragpos, fraglen,
        d0, d02, d82,   best/**/,
        ccmCache, ccmLast, tfmCache, tfmBest,
        tmpdpalnpossbuffer, tmpdpdiagbuffers);


    //calculate the score for the larger structure of the two:
    //threshold calculated for the greater length
    const int greaterlen = myhdmax(qrylenorg, dbstrlenorg);
    const float g0 = GetD0fin(greaterlen, greaterlen);
    const float g02 = SQRD(g0);
    float gbest = best;//score calculated for the other structure

    if(qrylenorg != dbstrlenorg) {
        CalcScoresUnrl_DPRefined_Complete<CHCKDST_NOCHECK>(
            READCNST_CALC2,
            qryndx, ndbCposs, dbxpad, maxnsteps, sfragfctxndx, dbstrdst,
            qrylen, dbstrlen, qrypos, rfnpos,  g0, g02, d82,
            tmpdpdiagbuffers, tmpdpalnpossbuffer, tfmBest, ccmCache+1);
        gbest = ccmCache[1];//score
    }

    //NOTE: synced either after the last cit or convergence check:
    SaveBestQRScoresAndTM_Complete<false/* WRITEFRAGINFO */,false/* CONDITIONAL */>(
        best, gbest,  qryndx, dbstrndx, ndbCstrs, 
        maxnsteps, sfragfctxndx, sfragndx, sfragpos,
        tfmBest, wrkmemtmibest, wrkmemaux);
}

// -------------------------------------------------------------------------
// Instantiations
//
#define INSTANTIATE_ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearch(tpTFM_DINV) \
    template __global__ void ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearch<tpTFM_DINV>( \
        const int nmaxconvit,const uint ndbCstrs,const uint ndbCposs,const uint dbxpad, \
        const uint nmaxsubfrags,const uint maxnfragfcts,const uint maxnsteps,const int sfragstep, \
        const int /*maxalnmax*/, \
        const float* __restrict__ tmpdpalnpossbuffer, \
        float* __restrict__ tmpdpdiagbuffers, \
        float* __restrict__ wrkmemtmibest, \
        float* __restrict__ wrkmemaux);

INSTANTIATE_ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearch(false);
INSTANTIATE_ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearch(true);

// =========================================================================
// ProductionFragmentBasedDPAlignmentRefinementPhase2: phase 2 to perform 
// production-version tuning of the the best superposition obtained 
// within the single kernel's actions; write the final superposition scores;
// TFM_DINV, use doubly inverted transformation matrices under suitable conditions;
// nmaxconvit, maximum number of superposition iterations;
// nqystrs, total number of query structures in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of reference positions in the chunk;
// dbxpad, #pad positions along the dimension of reference structures;
// nmaxsubfrags, total number of fragment lengths to consider;
// maxnfragfcts, max number of fragment position factors around an identified position;
// maxnsteps, total number of steps that should be performed for each reference structure;
// sfragstep, step size to traverse subfragments;
// maxalnmax, maximum alignment length across all query-reference pairs;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// tmpdpdiagbuffers, temporary diagonal buffers for positional scores;
// wrkmemtmibest, working memory for best-performing transformation matrices;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// 
template<bool TFM_DINV>
__global__ 
void ProductionFragmentBasedDPAlignmentRefinementPhase2(
    const int nmaxconvit,
//     const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint nmaxsubfrags,
    const uint maxnfragfcts,
    const uint maxnsteps,
    const int sfragstep,
    const int /*maxalnmax*/,
    const float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux)
{
    uint dbstrndx = blockIdx.x;//reference serial number
    const uint sfragfctxndx = blockIdx.y;//fragment factor x fragment length index
    const uint sfragfct = sfragfctxndx;//fragment factor
    uint sfragndx;//fragment length index to be read
    uint qryndx = blockIdx.z;//query serial number
    //cache for the cross-covarinace matrix and related data: 
    //no bank conflicts as long as inner-most dim is odd
    enum {neffds = twmvEndOfCCDataExt,//effective number of fields
        smidim = neffds + 1};
    __shared__ float ccmCache[
        smidim * CUS1_TBINITSP_COMPLETEREFINE_XDIM + twmvEndOfCCDataExt * 2 + nTTranformMatrix];
//     __shared__ float ccmLast[twmvEndOfCCDataExt];
//     __shared__ float tfmCache[twmvEndOfCCDataExt];//twmvEndOfCCDataExt>nTTranformMatrix
    float* ccmLast = ccmCache + smidim * CUS1_TBINITSP_COMPLETEREFINE_XDIM;
    float* tfmCache = ccmLast + twmvEndOfCCDataExt;//twmvEndOfCCDataExt>nTTranformMatrix
    float* tfmBest = tfmCache + twmvEndOfCCDataExt;//of size nTTranformMatrix

    int qrylenorg, dbstrlenorg;//original query and reference lengths
    int qrylen, dbstrlen;//pseudo query and reference length, #matched positions
    //distances in positions to the beginnings of the query and reference structures:
    uint /*qrydst, */dbstrdst;
    enum {qrypos = 0, rfnpos = 0};
    int sfragpos, fraglen;//fragment position and length


    if(threadIdx.x == 0) {
        uint mloc = ((qryndx * maxnsteps + 0/*sfragfctxndx*/) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        ccmCache[6] = wrkmemaux[mloc + dbstrndx];
    }

    __syncthreads();

    if(((int)(ccmCache[6])) & (CONVERGED_LOWTMSC_bitval))
        //(the termination flag for this pair is set);
        //all threads in the block exit;
        return;

    //NOTE: no sync as long ccmCache cell for convergence is not overwritten;


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse ccmCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(dbstrndx, (int*)ccmCache);
        //GetQueryLenDst(qryndx, (int*)ccmCache + 2);
        if(threadIdx.x == 0) ((int*)ccmCache)[2] = GetQueryLength(qryndx);
    }

    //NOTE: use a different warp for structure-specific-formatted data;
#if (CUS1_TBINITSP_COMPLETEREFINE_XDIM >= 64)
    if(threadIdx.x == tawmvNAlnPoss + 32 ||
       threadIdx.x == tawmvSubFragPos + 32 || threadIdx.x == tawmvSubFragNdx + 32) {
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        ccmCache[threadIdx.x-32] = wrkmemaux[mloc0 + (threadIdx.x-32) * ndbCstrs + dbstrndx];
#else
    if(threadIdx.x == tawmvNAlnPoss || threadIdx.x == tawmvSubFragPos || threadIdx.x == tawmvSubFragNdx) {
        //NOTE: reuse ccmCache to read values written at sfragfct==0:
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        ccmCache[threadIdx.x] = wrkmemaux[mloc0 + threadIdx.x * ndbCstrs + dbstrndx];
#endif
    }

    __syncthreads();


    dbstrdst = ((int*)ccmCache)[1];
    //qrydst = ((int*)ccmCache)[3];
    qrylen = dbstrlen = ccmCache[tawmvNAlnPoss];
    sfragpos = (int)(ccmCache[tawmvSubFragPos]) + ((int)sfragfct - (int)(maxnfragfcts>>1)) * sfragstep;
    sfragndx = (int)(ccmCache[tawmvSubFragNdx]);
    if(sfragndx == 0) sfragndx++;
    dbstrlenorg = ((int*)ccmCache)[0];
    qrylenorg = ((int*)ccmCache)[2];

    __syncthreads();


    fraglen = GetFragLength(qrylen, dbstrlen, qrypos, rfnpos, sfragndx);
    if(fraglen < 1) 
        //all threads in the block exit
        return;

    if(sfragpos < 0 ||
       qrylen + sfragstep <= qrypos + sfragpos + fraglen ||
       dbstrlen + sfragstep <= rfnpos + sfragpos + fraglen)
        //all threads in the block exit
        return;


    //threshold calculated for the original lengths
    const float d0 = GetD0fin(qrylenorg, dbstrlenorg);
    const float d02 = SQRD(d0);
    const float d82 = GetD82(qrylenorg, dbstrlenorg);
    float best = -1.0f;//best score obtained


    ProductionRefinementPhase2InnerLoop<TFM_DINV, smidim, neffds>(
        sfragfctxndx, qryndx,
        nmaxconvit, ndbCposs, dbxpad, maxnsteps,
        qrylenorg, dbstrlenorg, qrylen, dbstrlen, dbstrdst,
        qrypos, rfnpos, sfragpos, fraglen,
        d0, d02, d82,   best/**/,
        ccmCache, ccmLast, tfmCache, tfmBest,
        tmpdpalnpossbuffer, tmpdpdiagbuffers);


    //calculate the score for the larger structure of the two:
    //threshold calculated for the greater length
    const int greaterlen = myhdmax(qrylenorg, dbstrlenorg);
    const float g0 = GetD0fin(greaterlen, greaterlen);
    const float g02 = SQRD(g0);
    float gbest = best;//score calculated for the other structure

    if(qrylenorg != dbstrlenorg) {
        CalcScoresUnrl_DPRefined_Complete<CHCKDST_NOCHECK>(
            READCNST_CALC2,
            qryndx, ndbCposs, dbxpad, maxnsteps, sfragfctxndx, dbstrdst,
            qrylen, dbstrlen, qrypos, rfnpos,  g0, g02, d82,
            tmpdpdiagbuffers, tmpdpalnpossbuffer, tfmBest, ccmCache+1);
        gbest = ccmCache[1];//score
    }

    //NOTE: synced either after the last cit or convergence check:
    SaveBestQRScoresAndTM_Complete<false/* WRITEFRAGINFO */,false/* CONDITIONAL */>(
        best, gbest,  qryndx, dbstrndx, ndbCstrs, 
        maxnsteps, sfragfctxndx, sfragndx, sfragpos,
        tfmBest, wrkmemtmibest, wrkmemaux);
}

// -------------------------------------------------------------------------
// Instantiations
//
#define INSTANTIATE_ProductionFragmentBasedDPAlignmentRefinementPhase2(tpTFM_DINV) \
    template __global__ void ProductionFragmentBasedDPAlignmentRefinementPhase2<tpTFM_DINV>( \
        const int nmaxconvit,const uint ndbCstrs,const uint ndbCposs,const uint dbxpad, \
        const uint nmaxsubfrags,const uint maxnfragfcts,const uint maxnsteps,const int sfragstep, \
        const int /*maxalnmax*/, \
        const float* __restrict__ tmpdpalnpossbuffer, \
        float* __restrict__ tmpdpdiagbuffers, \
        float* __restrict__ wrkmemtmibest, \
        float* __restrict__ wrkmemaux);

INSTANTIATE_ProductionFragmentBasedDPAlignmentRefinementPhase2(false);
INSTANTIATE_ProductionFragmentBasedDPAlignmentRefinementPhase2(true);

// =========================================================================

// -------------------------------------------------------------------------
// ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearch: phase 2 to
// perform production-version tuning of the the best superposition obtained 
// within the single kernel's actions; write the final superposition scores;
// NOTE: This version performs a log number of superposition evaluations;
// TFM_DINV, use doubly inverted transformation matrices under suitable conditions;
// nmaxconvit, maximum number of superposition iterations;
// nqystrs, total number of query structures in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of reference positions in the chunk;
// dbxpad, #pad positions along the dimension of reference structures;
// nmaxsubfrags, total number of fragment lengths to consider;
// maxnfragfcts, max number of fragment position factors around an identified position;
// maxnsteps, total number of steps that should be performed for each reference structure;
// sfragstep, step size to traverse subfragments;
// maxalnmax, maximum alignment length across all query-reference pairs;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// tmpdpdiagbuffers, temporary diagonal buffers for positional scores;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// alndatamem, memory for full alignment information, including scores;
// tfmmem, memory for transformation matrices;
// 
template<bool TFM_DINV>
__global__ 
void ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearch(
    const int nmaxconvit,
//     const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint nmaxsubfrags,
    const uint /*maxnfragfcts*/,
    const uint maxnsteps,
    const int /*sfragstep*/,
    const int /*maxalnmax*/,
    const float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux,
    float* __restrict__ alndatamem,
    float* __restrict__ tfmmem)
{
    uint dbstrndx = blockIdx.x;//reference serial number
    const uint sfragfctxndx = blockIdx.y;//fragment factor x fragment length index
    // const uint sfragfct = sfragfctxndx;//fragment factor
    // uint sfragndx;//fragment length index
    uint qryndx = blockIdx.z;//query serial number
    //cache for the cross-covarinace matrix and related data: 
    //no bank conflicts as long as inner-most dim is odd
    enum {neffds = twmvEndOfCCDataExt,//effective number of fields
        smidim = neffds + 1};
    __shared__ float ccmCache[
        smidim * CUS1_TBINITSP_COMPLETEREFINE_XDIM + twmvEndOfCCDataExt * 2 + nTTranformMatrix];
//     __shared__ float ccmLast[twmvEndOfCCDataExt];
//     __shared__ float tfmCache[twmvEndOfCCDataExt];//twmvEndOfCCDataExt>nTTranformMatrix
    float* ccmLast = ccmCache + smidim * CUS1_TBINITSP_COMPLETEREFINE_XDIM;
    float* tfmCache = ccmLast + twmvEndOfCCDataExt;//twmvEndOfCCDataExt>nTTranformMatrix
    float* tfmBest = tfmCache + twmvEndOfCCDataExt;//of size nTTranformMatrix

    int qrylenorg, dbstrlenorg;//original query and reference lengths
    int qrylen, dbstrlen;//pseudo query and reference length, #matched positions
    //distances in positions to the beginnings of the query and reference structures:
    uint /*qrydst, */dbstrdst;
    enum {qrypos = 0, rfnpos = 0, tmpcslot = tawmvEndOfCCMDat};


    if(threadIdx.x == 0) {
        uint mloc = ((qryndx * maxnsteps + 0/*sfragfctxndx*/) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        ccmCache[6] = wrkmemaux[mloc + dbstrndx];
    }

    __syncthreads();

    if(((int)(ccmCache[6])) & (CONVERGED_LOWTMSC_bitval))
        //(the termination flag for this pair is set);
        //all threads in the block exit;
        return;

    //NOTE: no sync as long ccmCache cell for convergence is not overwritten;


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse ccmCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(dbstrndx, (int*)ccmCache + tmpcslot);
        //GetQueryLenDst(qryndx, (int*)ccmCache + tmpcslot + 2);
        if(threadIdx.x == 0) ((int*)ccmCache)[tmpcslot+2] = GetQueryLength(qryndx);
    }

    //NOTE: use a different warp for structure-specific-formatted data;
#if (CUS1_TBINITSP_COMPLETEREFINE_XDIM >= 64)
    if(threadIdx.x == tawmvNAlnPoss + 32 || threadIdx.x == tawmvGrandBest + 32 ||
       threadIdx.x == tawmvSubFragPos + 32/* || threadIdx.x == tawmvSubFragNdx + 32 */) {
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        ccmCache[threadIdx.x-32] = wrkmemaux[mloc0 + (threadIdx.x-32) * ndbCstrs + dbstrndx];
#else
    if(threadIdx.x == tawmvNAlnPoss || threadIdx.x == tawmvGrandBest ||
       threadIdx.x == tawmvSubFragPos/* || threadIdx.x == tawmvSubFragNdx */) {
        //NOTE: reuse ccmCache to read values written at sfragfct==0:
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        ccmCache[threadIdx.x] = wrkmemaux[mloc0 + threadIdx.x * ndbCstrs + dbstrndx];
#endif
    }

    __syncthreads();


    dbstrdst = ((int*)ccmCache)[tmpcslot+1];
    //qrydst = ((int*)ccmCache)[tmpcslot+3];
    qrylen = dbstrlen = ccmCache[tawmvNAlnPoss];
    // sfragndx = (int)(ccmCache[tawmvSubFragNdx]);
    int sfragposorg = (int)(ccmCache[tawmvSubFragPos]);
    dbstrlenorg = ((int*)ccmCache)[tmpcslot+0];
    qrylenorg = ((int*)ccmCache)[tmpcslot+2];

    float bestorg = ccmCache[tawmvGrandBest];//best score obtained
    float best = bestorg;

    __syncthreads();

    //threshold calculated for the original lengths
    const float d0 = GetD0fin(qrylenorg, dbstrlenorg);
    const float d02 = SQRD(d0);
    const float d82 = GetD82(qrylenorg, dbstrlenorg);


    for(int sfragstepabs = 32; sfragstepabs >= 1; sfragstepabs >>= 1) {
        for(int sgn = 1; sgn >= -1; sgn -= 2)
        {
            int sfragstep = sgn * sfragstepabs;
            int sfragpos = sfragposorg + sfragstep;
            for(int sfragndx = 1; sfragndx < nmaxsubfrags; sfragndx++)
            {
                int fraglen = GetFragLength(qrylen, dbstrlen, qrypos, rfnpos, sfragndx);

                if(fraglen < 1) continue;
                if(sfragpos < 0 ||
                   qrylen + myhdmax(0, sfragstep) <= qrypos + sfragpos + fraglen ||
                   dbstrlen + myhdmax(0, sfragstep) <= rfnpos + sfragpos + fraglen)
                    continue;

                float bestprev = best;

                ProductionRefinementPhase2InnerLoop<TFM_DINV, smidim, neffds>(
                    sfragfctxndx, qryndx,
                    nmaxconvit, ndbCposs, dbxpad, maxnsteps,
                    qrylenorg, dbstrlenorg, qrylen, dbstrlen, dbstrdst,
                    qrypos, rfnpos, sfragpos, fraglen,
                    d0, d02, d82,   best/**/,
                    ccmCache, ccmLast, tfmCache, tfmBest,
                    tmpdpalnpossbuffer, tmpdpdiagbuffers);

                if(bestprev < best) sfragposorg = sfragpos;
            }
        }
    }


    if(best <= bestorg) return;//all exit

    //calculate the score for the larger structure of the two:
    //threshold calculated for the greater length
    const int greaterlen = myhdmax(qrylenorg, dbstrlenorg);
    const float g0 = GetD0fin(greaterlen, greaterlen);
    const float g02 = SQRD(g0);
    float gbest = best;//score calculated for the other structure

    if(qrylenorg != dbstrlenorg) {
        CalcScoresUnrl_DPRefined_Complete<CHCKDST_NOCHECK>(
            READCNST_CALC2,
            qryndx, ndbCposs, dbxpad, maxnsteps, sfragfctxndx, dbstrdst,
            qrylen, dbstrlen, qrypos, rfnpos,  g0, g02, d82,
            tmpdpdiagbuffers, tmpdpalnpossbuffer, tfmBest, ccmCache+1);
        gbest = ccmCache[1];//score
    }

    //NOTE: synced either after the last cit or convergence check;
    //NOTE: write directly to production output memory:
    SaveBestQRScoresAndTM_Phase2_logsearch_Complete(
        best, gbest,  qryndx, dbstrndx, ndbCstrs,  qrylenorg, dbstrlenorg,
        tfmBest, tfmmem, alndatamem);
}

// -------------------------------------------------------------------------
// Instantiations
//
#define INSTANTIATE_ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearch(tpTFM_DINV) \
    template __global__ void ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearch<tpTFM_DINV>( \
        const int nmaxconvit,const uint ndbCstrs,const uint ndbCposs,const uint dbxpad, \
        const uint nmaxsubfrags,const uint maxnfragfcts,const uint maxnsteps,const int sfragstep, \
        const int /*maxalnmax*/, \
        const float* __restrict__ tmpdpalnpossbuffer, \
        float* __restrict__ tmpdpdiagbuffers, \
        float* __restrict__ wrkmemaux, \
        float* __restrict__ alndatamem, \
        float* __restrict__ tfmmem);

INSTANTIATE_ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearch(false);
INSTANTIATE_ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearch(true);

// =========================================================================
