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
#include "covariance_dp_refn_complete.cuh"

// =========================================================================
// FragmentBasedDPAlignmentRefinement: refine alignment and its boundaries 
// within the single kernel's actions to obtain favorable superposition;
// WRITEFRAGINFO, template parameter, flag of whether refined fragment 
// boundaries should be saved;
// CONDITIONAL, template parameter, flag of writing the score if it's 
// greater at the same location;
// TFM_DINV, use doubly inverted transformation matrices under suitable conditions;
// readlocalconv, flag of reading local convergence flag;
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
template<bool WRITEFRAGINFO, bool CONDITIONAL, bool TFM_DINV>
__global__ 
void FragmentBasedDPAlignmentRefinement(
    const bool readlocalconv,
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
    float* __restrict__ wrkmemaux)
{
    uint dbstrndx = blockIdx.x;//reference serial number
    uint sfragfctxndx = blockIdx.y;//fragment factor x fragment length index
    uint sfragfct = sfragfctxndx / nmaxsubfrags;//fragment factor
    uint sfragndx = sfragfctxndx - sfragfct * nmaxsubfrags;//fragment length index
    uint qryndx = blockIdx.z;//query serial number
    //cache for the cross-covarinace matrix and related data: 
    //no bank conflicts as long as inner-most dim is odd
    enum {neffds = twmvEndOfCCDataExt,//effective number of fields
        smidim = neffds+1};
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
    int sfragpos, fraglen;


///     //get sfragfct and sfragndx given sfragfctxndx
///     GetSubfragFctAndNdx(sfragfct, sfragndx,
///         sfragfctxndx, nmaxsubfrags, sfragstep, maxalnmax);
/// 
///     //all threads exits: out of bounds
///     if(nmaxsubfrags <= sfragndx) return;


    if(threadIdx.x == 0) {
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        uint mloc = ((qryndx * maxnsteps + sfragfctxndx) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        ccmCache[6] = ccmCache[7] = wrkmemaux[mloc0 + dbstrndx];
        if(readlocalconv && sfragfctxndx != 0) ccmCache[7] = wrkmemaux[mloc + dbstrndx];
    }

    __syncthreads();

    if((((int)(ccmCache[6])) & (CONVERGED_LOWTMSC_bitval)) || ccmCache[7])
        //(NOTE:any type of convergence applies locally and CONVERGED_LOWTMSC_bitval globally);
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
        //NOTE: reuse ccmCache to read #matched positions;
        //NOTE: tawmvNAlnPoss written at sfragfct==0:
        uint mloc = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        ccmCache[tawmvNAlnPoss] = wrkmemaux[mloc + tawmvNAlnPoss * ndbCstrs + dbstrndx];
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
    const float d0 = GetD0(qrylenorg, dbstrlenorg);
    const float d02 = SQRD(d0);
    const float d82 = GetD82(qrylenorg, dbstrlenorg);
    float dst32 = CP_LARGEDST;
    float best = 0.0f;//best score obtained

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

        CalcScoresUnrl_DPRefined_Complete(
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

    //NOTE: synced either after the last cit or convergence check:
    SaveBestScoreAndTM_Complete<WRITEFRAGINFO,CONDITIONAL>(
        best,  qryndx, dbstrndx, ndbCstrs, 
        maxnsteps, sfragfctxndx, sfragndx, sfragpos,
        tfmBest, wrkmemtmibest, wrkmemaux);
}

// -------------------------------------------------------------------------
// Instantiations
// 
#define INSTANTIATE_FragmentBasedDPAlignmentRefinement(tpWRITEFRAGINFO,tpCONDITIONAL,tpTFM_DINV) \
    template __global__ void FragmentBasedDPAlignmentRefinement<tpWRITEFRAGINFO,tpCONDITIONAL,tpTFM_DINV>( \
        const bool readlocalconv, \
        const int nmaxconvit, /*const uint nqystrs,*/ \
        const uint ndbCstrs, const uint ndbCposs, const uint dbxpad, \
        const uint nmaxsubfrags, const uint maxnsteps, \
        const int sfragstep, const int maxalnmax, \
        const float* __restrict__ tmpdpalnpossbuffer, \
        float* __restrict__ tmpdpdiagbuffers, \
        float* __restrict__ wrkmemtmibest, \
        float* __restrict__ wrkmemaux);

INSTANTIATE_FragmentBasedDPAlignmentRefinement(false/* true */,true,false);
INSTANTIATE_FragmentBasedDPAlignmentRefinement(false/* true */,false,false);

INSTANTIATE_FragmentBasedDPAlignmentRefinement(false/* true */,true,true);
INSTANTIATE_FragmentBasedDPAlignmentRefinement(false/* true */,false,true);

// =========================================================================
