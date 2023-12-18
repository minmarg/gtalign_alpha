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
#include "libmycu/cucom/mysort.cuh"
#include "libmycu/cumath/cumath.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/custages/fields.cuh"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/covariance.cuh"
#include "libmycu/custages/covariance_refn.cuh"
#include "libmycu/custages/transform.cuh"
#include "scoring.cuh"

// -------------------------------------------------------------------------
// SetCurrentFragSpecs: set the specifications of the current fragment 
// under process;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// sfragndx, index defining fragment length;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
__global__ void SetCurrentFragSpecs(
    const uint ndbCstrs,
    const uint maxnsteps,
    const int sfragndx,
    float* __restrict__ wrkmemaux)
{
    //index of the reference structure:
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number
    uint sfragfct = blockIdx.z;//fragment factor

    if(ndbCstrs <= dbstrndx)
        //no sync below: exit
        return;

    uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;

    wrkmemaux[mloc + tawmvSubFragNdxCurrent * ndbCstrs + dbstrndx] = sfragndx;
}

// -------------------------------------------------------------------------
// SetLowScoreConvergenceFlag: set the appropriate convergence flag for 
// the pairs for which the score is below the threshold;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
__global__ void SetLowScoreConvergenceFlag(
    const float scorethld,
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux)
{
    //index of the reference structure:
    //blockDim.x == CUS1_TBSP_SCORE_SET_XDIM
    const uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qryndx = blockIdx.y;//query serial number
    const uint sfragfct = 0;//fragment factor

    if(ndbCstrs <= dbstrndx)
        //no sync below: exit
        return;

    uint mloc0 = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;

    //grand scores fragment factor position 0:
    float grand = wrkmemaux[mloc0 + tawmvGrandBest * ndbCstrs + dbstrndx];

    int qrylen = GetQueryLength(qryndx);
    int dbstrlen = GetDbStrLength(dbstrndx);

    //check for low scores:
    if(grand < scorethld * (float)myhdmin(qrylen, dbstrlen)) {
        int convflag = wrkmemaux[mloc0 + tawmvConverged * ndbCstrs + dbstrndx];//float->int
        wrkmemaux[mloc0 + tawmvConverged * ndbCstrs + dbstrndx] =
            (float)(convflag | CONVERGED_LOWTMSC_bitval);
    }
}

// -------------------------------------------------------------------------
// InitScores: initialize best and current scores to 0;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// minfraglen, minimum fragment length for which maxnsteps is calculated;
// checkfragos, check whether calculated fragment position is within boundaries;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
template<int INITOPT>
__global__ void InitScores(
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint /* minfraglen */,
    const bool checkfragos,
    float* __restrict__ wrkmemaux)
{
    //index of the reference structure:
    //blockDim.x == CUS1_TBSP_SCORE_SET_XDIM
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number
    uint sfragfct = blockIdx.z;//fragment factor


//     //{{TODO: to be removed as two reads per pair is more expensive than one write
//     if(checkfragos) {
//         uint sfragpos = sfragfct * FRAGREF_SFRAGSTEP;//fragment position
//         int dbstrlen = 0;
//         __shared__ int qrylen;
// 
//         qrylen = 0;
// 
//         if(dbstrndx < ndbCstrs) {
//             dbstrlen = GetDbStrLength(dbstrndx);
//             if(threadIdx.x == 0) qrylen = GetQueryLength(qryndx);
//         }
// 
//         __syncthreads();
// 
//         if(dbstrndx < ndbCstrs) {
//             uint maxalnlen = myhdmin(dbstrlen, qrylen);
//             dbstrlen = FragPosWithinAlnBoundaries(maxalnlen, FRAGREF_SFRAGSTEP, sfragpos, minfraglen);
//         }
// 
//         //__syncthreads();//no sync as each thread accesses own data
// 
//         if(dbstrlen == 0) return;//no sync below: exit
//     }
//     //}}TODO


    if(ndbCstrs <= dbstrndx)
        //no sync below: exit
        return;

    uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;

    if(INITOPT & INITOPT_ALL) {
        #pragma unroll
        for(int f = 0; f < nTAuxWorkingMemoryVars; f++)
            wrkmemaux[mloc + f * ndbCstrs + dbstrndx] = 0.0f;
    }

    //NOTE: one read and write are more efficient than two reads plus additional 
    // calculation (for bounds) and these memory instructions (done when checking 
    // whether a pair is out of bounds)
    if(INITOPT & INITOPT_BEST)
        wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx] = 0.0f;

    if(INITOPT & INITOPT_CURRENT) 
        wrkmemaux[mloc + tawmvScore * ndbCstrs + dbstrndx] = 0.0f;


    if(INITOPT & INITOPT_QRYRFNPOS) {
        wrkmemaux[mloc + tawmvQRYpos * ndbCstrs + dbstrndx] = 0.0f;
        wrkmemaux[mloc + tawmvRFNpos * ndbCstrs + dbstrndx] = 0.0f;
    }


    if(INITOPT & INITOPT_FRAGSPECS) {
        wrkmemaux[mloc + tawmvSubFragNdxCurrent * ndbCstrs + dbstrndx] = 0.0f;
        wrkmemaux[mloc + tawmvSubFragPosCurrent * ndbCstrs + dbstrndx] = 0.0f;
    }


    if(INITOPT & INITOPT_NALNPOSS) 
        wrkmemaux[mloc + tawmvNAlnPoss * ndbCstrs + dbstrndx] = 0.0f;


    if(INITOPT & INITOPT_CONVFLAG_ALL) 
        wrkmemaux[mloc + tawmvConverged * ndbCstrs + dbstrndx] = 0.0f;


    if(INITOPT &
        (INITOPT_CONVFLAG_FRAGREF | INITOPT_CONVFLAG_SCOREDP |
        INITOPT_CONVFLAG_NOTMPRG | INITOPT_CONVFLAG_LOWTMSC |
        INITOPT_CONVFLAG_LOWTMSC_SET))
    {
        mloc += tawmvConverged * ndbCstrs + dbstrndx;
        int convflag = wrkmemaux[mloc];//float->int

        //NOTE: do not set any value if the process for this pair is to terminate
        if(convflag & CONVERGED_LOWTMSC_bitval) return;

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

        wrkmemaux[mloc] = (float)convflag;//int->float
    }
}

// Instantiations:
// 
#define INSTANTIATE_InitScores(tpINITOPT) \
    template __global__ void InitScores<tpINITOPT>( \
        const uint ndbCstrs, \
        const uint maxnsteps, const uint minfraglen, \
        const bool checkfragos, \
        float* __restrict__ wrkmemaux);

INSTANTIATE_InitScores(INITOPT_ALL);
INSTANTIATE_InitScores(INITOPT_BEST);
INSTANTIATE_InitScores(INITOPT_CURRENT);
INSTANTIATE_InitScores(INITOPT_QRYRFNPOS);
INSTANTIATE_InitScores(INITOPT_FRAGSPECS);
INSTANTIATE_InitScores(INITOPT_NALNPOSS);
INSTANTIATE_InitScores(INITOPT_CONVFLAG_ALL);
INSTANTIATE_InitScores(INITOPT_CONVFLAG_FRAGREF);
INSTANTIATE_InitScores(INITOPT_CONVFLAG_SCOREDP);
INSTANTIATE_InitScores(INITOPT_CONVFLAG_NOTMPRG);
INSTANTIATE_InitScores(INITOPT_CONVFLAG_LOWTMSC);
INSTANTIATE_InitScores(INITOPT_CONVFLAG_LOWTMSC_SET|INITOPT_ALL);
INSTANTIATE_InitScores(INITOPT_CONVFLAG_FRAGREF|INITOPT_CONVFLAG_SCOREDP);
INSTANTIATE_InitScores(INITOPT_NALNPOSS|INITOPT_CONVFLAG_SCOREDP);
INSTANTIATE_InitScores(INITOPT_BEST|INITOPT_CONVFLAG_ALL);

// -------------------------------------------------------------------------
// SaveLastScore: save the last best score at the position corresponding to 
// fragment factor 0;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
__global__ void SaveLastScore0(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux)
{
    //index of the structure to start with (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number

    if(ndbCstrs <= dbstrndx)
        //no sync below: exit
        return;

    uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;

    int converged = wrkmemaux[mloc0 + tawmvConverged * ndbCstrs + dbstrndx];//float->int

    //score convergence applies:
    if(converged & (CONVERGED_SCOREDP_bitval | CONVERGED_LOWTMSC_bitval)) return;

    wrkmemaux[mloc0 + tawmvBest0 * ndbCstrs + dbstrndx] =
        wrkmemaux[mloc0 + tawmvBestScore * ndbCstrs + dbstrndx];
}

// -------------------------------------------------------------------------
// SaveBestScore: save best score along with query and reference 
// positions for which this score is observed;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// n1, starting position that determines positions in query and reference;
// step, step size in positions used to traverse query and reference 
// ungapped alignments;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
__global__ void SaveBestScore(
    const uint ndbCstrs,
    const uint maxnsteps,
    int n1, int step,
    float* __restrict__ wrkmemaux)
{
    //index of the first structure to start with (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number
    uint sfragfct = blockIdx.z;//fragment factor
    n1 += sfragfct * step;
    int qrypos = myhdmax(0,n1);//starting query position
    int rfnpos = myhdmax(-n1,0);//starting reference position

    //no sync below, threads process independently: exit
    if(ndbCstrs <= dbstrndx) return;

    uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;

    float bestscore = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx];
    float score = wrkmemaux[mloc + tawmvScore * ndbCstrs + dbstrndx];

    //NOTE: two reads are more efficient than two structure length reads plus 
    // additional calculation (for bounds) (done when checking whether a 
    // pair is out of bounds);
    // the if clause will not be true for an out-of-bounds case
    if(bestscore < score) {
        wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx] = score;
        wrkmemaux[mloc + tawmvQRYpos * ndbCstrs + dbstrndx] = qrypos;
        wrkmemaux[mloc + tawmvRFNpos * ndbCstrs + dbstrndx] = rfnpos;
    }
}

// -------------------------------------------------------------------------
// SaveBestScoreAmongBests: save best score along with query and reference 
// positions by considering all partial best scores calculated over all 
// fragment factors; write it to the location of fragment factor 0;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// effnsteps, effective (actual maximum) number of steps (blockIdx.z);
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
__global__ void SaveBestScoreAmongBests(
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint effnsteps,
    float* __restrict__ wrkmemaux)
{
    //index of the structure (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number
    __shared__ float scvCache[CUS1_TBSP_SCORE_MAX_YDIM][CUS1_TBSP_SCORE_MAX_XDIM+1];
    __shared__ uint ndxCache[CUS1_TBSP_SCORE_MAX_YDIM][CUS1_TBSP_SCORE_MAX_XDIM+1];

    scvCache[threadIdx.y][threadIdx.x] = 0.0f;
    ndxCache[threadIdx.y][threadIdx.x] = 0;

    //no sync; threads do not access other cells below

    for(uint sfragfct = threadIdx.y; sfragfct < effnsteps; sfragfct += blockDim.y) {
        float bscore = 0.0f;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(dbstrndx < ndbCstrs)//READ, coalesced for multiple references
            bscore = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx];
        if(scvCache[threadIdx.y][threadIdx.x] < bscore) {
            scvCache[threadIdx.y][threadIdx.x] = bscore;
            ndxCache[threadIdx.y][threadIdx.x] = sfragfct;
        }
        //no sync, every thread works in its own space
    }

    __syncthreads();

    //reduce/unroll for max best score over the fragment factors:
    for(int ydim = (CUS1_TBSP_SCORE_MAX_YDIM>>1); ydim >= 1; ydim >>= 1) {
        if(threadIdx.y < ydim &&
            scvCache[threadIdx.y][threadIdx.x] <
            scvCache[threadIdx.y+ydim][threadIdx.x])
        {
            scvCache[threadIdx.y][threadIdx.x] = scvCache[threadIdx.y+ydim][threadIdx.x];
            ndxCache[threadIdx.y][threadIdx.x] = ndxCache[threadIdx.y+ydim][threadIdx.x];
        }

        __syncthreads();
    }

    //scvCache[0][...] now contains maximum
    if(threadIdx.y == 0) {
        uint sfragfct = ndxCache[0][threadIdx.x];
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        float bscore = scvCache[0][threadIdx.x];
        if(sfragfct != 0 && dbstrndx < ndbCstrs) {
            float qrypos = wrkmemaux[mloc + tawmvQRYpos * ndbCstrs + dbstrndx];
            float rfnpos = wrkmemaux[mloc + tawmvRFNpos * ndbCstrs + dbstrndx];
            //coalesced WRITE for multiple references
            wrkmemaux[mloc0 + tawmvBestScore * ndbCstrs + dbstrndx] = bscore;
            wrkmemaux[mloc0 + tawmvQRYpos * ndbCstrs + dbstrndx] = qrypos;
            wrkmemaux[mloc0 + tawmvRFNpos * ndbCstrs + dbstrndx] = rfnpos;
        }
        if(dbstrndx < ndbCstrs)
            wrkmemaux[mloc0 + tawmvInitialBest * ndbCstrs + dbstrndx] = bscore;
    }
}

// -------------------------------------------------------------------------
// CheckScoreConvergence: check whether the score of the last two 
// procedures converged, i.e., the difference is small;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
__global__ void CheckScoreConvergence(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux)
{
    //index of the structure to start with (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number
    int sfragfct = blockIdx.z;//fragment factor

    if(ndbCstrs <= dbstrndx)
        //no sync below, threads process independently: exit
        return;

    uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
    uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;

    int converged = wrkmemaux[mloc + tawmvConverged * ndbCstrs + dbstrndx];//float->int

    //score convergence applies:
    if(converged & (CONVERGED_SCOREDP_bitval | CONVERGED_LOWTMSC_bitval)) return;

    //best scores are recorded at a fragment factor position of 0
    float prevbest0 = wrkmemaux[mloc0 + tawmvBest0 * ndbCstrs + dbstrndx];
    float best0 = wrkmemaux[mloc0 + tawmvBestScore * ndbCstrs + dbstrndx];

    //check score convergence; populate convergence flag over all fragment factors
    if(fabsf(best0-prevbest0) < SCORE_CONVEPSILON)
        wrkmemaux[mloc + tawmvConverged * ndbCstrs + dbstrndx] =
            (float)(converged | CONVERGED_SCOREDP_bitval);
}

#if 0
// -------------------------------------------------------------------------
// CheckScoreProgression: check whether the difference between the maximum 
// score and the score of the last procedure is large enough; if not, set 
// the appropriate convergence flag;
// ndbCstrs, total number of reference structures in the chunk;
// maxscorefct, factor for the maximum score;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// 
__global__ void CheckScoreProgression(
    uint ndbCstrs,
    float maxscorefct,
    float* __restrict__ wrkmemaux)
{
    //index of the structure to start with (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number

    if(ndbCstrs <= dbstrndx)
        //no sync below, threads process independently: exit
        return;

    int converged = wrkmemaux[
        (qryndx * nTAuxWorkingMemoryVars + tawmvConverged) *
        ndbCstrs + dbstrndx];//float->int

    //score convergence applies:
    if(converged & (CONVERGED_NOTMPRG_bitval | CONVERGED_LOWTMSC_bitval)) return;

    float grand = wrkmemaux[
        (qryndx * nTAuxWorkingMemoryVars + tawmvGrandBest) *
        ndbCstrs + dbstrndx];
    float best = wrkmemaux[
        (qryndx * nTAuxWorkingMemoryVars + tawmvBestScore) *
        ndbCstrs + dbstrndx];

    //check score progression
    if(best <= grand * maxscorefct)
        wrkmemaux[
            (qryndx * nTAuxWorkingMemoryVars + tawmvConverged) *
            ndbCstrs + dbstrndx] =
                (float)(converged | CONVERGED_NOTMPRG_bitval);
}
#endif

// -------------------------------------------------------------------------
// SaveBestScoreAndTM: save best scores along with transformation matrices;
// save fragment indices and starting positions too;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// sfragstep, step size to traverse subfragments;
// sfragndx, index defining fragment length;
// sfragpos, starting position within fragment;
// NOTE: memory pointers should be aligned!
// wrkmemtm, working memory for transformation matrices;
// wrkmemtmibest, working memory for iteration-best transformation matrices;
// tfmmem, memory for transformation matrices;
// wrkmemaux, auxiliary working memory;
// NOTE: unroll by a factor of CUS1_TBINITSP_TMSAVE_XFCT: this number of 
// structures processed by a thread block
// 
template<bool WRITEFRAGINFO>
__global__ void SaveBestScoreAndTM(
    const uint ndbCstrs,
    const uint maxnsteps,
    const int sfragstep,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux)
{
    //index of the first structure to start with (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * CUS1_TBINITSP_TMSAVE_XFCT;
    uint qryndx = blockIdx.y;//query serial number
    uint sfragfct = blockIdx.z;//fragment factor
    uint ndx = 0;//relative reference index < CUS1_TBINITSP_TMSAVE_XFCT
    __shared__ float scvCache[CUS1_TBINITSP_TMSAVE_XFCT];


    if(threadIdx.x < CUS1_TBINITSP_TMSAVE_XFCT)
        scvCache[threadIdx.x] = 0.0f;

    uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;

    if(threadIdx.x < CUS1_TBINITSP_TMSAVE_XFCT && dbstrndx + threadIdx.x < ndbCstrs) {
        scvCache[threadIdx.x] = wrkmemaux[mloc + tawmvScore * ndbCstrs + dbstrndx + threadIdx.x];
        float best = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx + threadIdx.x];
        if(scvCache[threadIdx.x] <= best) scvCache[threadIdx.x] = 0.0f;
    }

    __syncthreads();

    #pragma unroll
    for(int i = 1; i < CUS1_TBINITSP_TMSAVE_XFCT; i++)
        if(i * nTTranformMatrix <= threadIdx.x) ndx = i;

    //save scores first
    if(threadIdx.x < CUS1_TBINITSP_TMSAVE_XFCT && scvCache[threadIdx.x]) {
        wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx + threadIdx.x] = scvCache[threadIdx.x];

        if(WRITEFRAGINFO) {
            //NOTE: using this function with WRITEFRAGINFO==true will not be correct and may 
            //lead to inconsistent results in the final stage of the best alignment refinement!
            // wrkmemaux[mloc + tawmvSubFragNdxCurrent * ndbCstrs + dbstrndx + threadIdx.x] =
            //     wrkmemaux[mloc + tawmvSubFragNdxCurrent * ndbCstrs + dbstrndx + threadIdx.x];

            wrkmemaux[mloc + tawmvSubFragPosCurrent * ndbCstrs + dbstrndx + threadIdx.x] =
                sfragfct * sfragstep;
        }
    }

    //save transformation matrices next
    if(threadIdx.x < nTTranformMatrix * (ndx+1) && scvCache[ndx]) {
        mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        float value = wrkmemtm[mloc];
        wrkmemtmibest[mloc] = value;
    }
}

// -------------------------------------------------------------------------
//Instantiations:
//
#define INSTANTIATE_SaveBestScoreAndTM(tpWRITEFRAGINFO) \
    template __global__ void SaveBestScoreAndTM<tpWRITEFRAGINFO>( \
        const uint ndbCstrs, const uint maxnsteps, const int sfragstep, \
        const float* __restrict__ wrkmemtm, \
        float* __restrict__ wrkmemtmibest, \
        float* __restrict__ wrkmemaux);

INSTANTIATE_SaveBestScoreAndTM(false);
INSTANTIATE_SaveBestScoreAndTM(true);

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
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps performed for each reference structure;
// effnsteps, effective (actual maximum) number of steps;
// NOTE: memory pointers should be aligned!
// wrkmemtmibest, working memory for iteration-best transformation matrices;
// tfmmem, memory for transformation matrices;
// wrkmemaux, auxiliary working memory;
// wrkmemtmibest2nd, secondary working memory for iteration-best transformation 
// matrices (only slot 0 is used but indexing involves maxnsteps);
// 
template<
    bool WRITEFRAGINFO,
    bool GRANDUPDATE,
    bool FORCEWRITEFRAGINFO,
    int SECONDARYUPDATE>
__global__
void SaveBestScoreAndTMAmongBests(
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint effnsteps,
    float* __restrict__ wrkmemtmibest,
    float* __restrict__ tfmmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmemtmibest2nd)
{
    //index of the structure (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number
    __shared__ float scvCache[CUS1_TBSP_SCORE_MAX_YDIM][CUS1_TBSP_SCORE_MAX_XDIM+1];
    __shared__ uint ndxCache[CUS1_TBSP_SCORE_MAX_YDIM][CUS1_TBSP_SCORE_MAX_XDIM+1];

    scvCache[threadIdx.y][threadIdx.x] = 0.0f;
    ndxCache[threadIdx.y][threadIdx.x] = 0;

    //no sync; threads do not access other cells below

    for(uint sfragfct = threadIdx.y; sfragfct < /*maxnsteps*/effnsteps; sfragfct += blockDim.y) {
        float bscore = 0.0f;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(dbstrndx < ndbCstrs)//READ, coalesced for multiple references
            bscore = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx];
        if(scvCache[threadIdx.y][threadIdx.x] < bscore) {
            scvCache[threadIdx.y][threadIdx.x] = bscore;
            ndxCache[threadIdx.y][threadIdx.x] = sfragfct;
        }
        //no sync, every thread works in its own space
    }

    __syncthreads();

    //reduce/unroll for max best score over the fragment factors:
    for(int ydim = (CUS1_TBSP_SCORE_MAX_YDIM>>1); ydim >= 1; ydim >>= 1) {
        if(threadIdx.y < ydim &&
            scvCache[threadIdx.y][threadIdx.x] <
            scvCache[threadIdx.y+ydim][threadIdx.x])
        {
            scvCache[threadIdx.y][threadIdx.x] = scvCache[threadIdx.y+ydim][threadIdx.x];
            ndxCache[threadIdx.y][threadIdx.x] = ndxCache[threadIdx.y+ydim][threadIdx.x];
        }

        __syncthreads();
    }

    //scvCache[0][...] now contains maximum
    uint sfragfct = ndxCache[0][threadIdx.x];
    bool wrtgrand = 0;
    bool wrt2ndry = (SECONDARYUPDATE == SECONDARYUPDATE_UNCONDITIONAL);

    //write scores first
    if(threadIdx.y == 0) {
        ndxCache[1][threadIdx.x] = 0;
        ndxCache[2][threadIdx.x] = wrt2ndry;
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(sfragfct != 0 && dbstrndx < ndbCstrs) {
            float bscore = scvCache[0][threadIdx.x];
            //coalesced WRITE for multiple references
            wrkmemaux[mloc0 + tawmvBestScore * ndbCstrs + dbstrndx] = bscore;
        }
        //update the grand best scores
        if(dbstrndx < ndbCstrs) {
            float bscore2nd;
            float bscore = scvCache[0][threadIdx.x];
            float grand = wrkmemaux[mloc0 + tawmvGrandBest * ndbCstrs + dbstrndx];
            {
                uint mloc20 = ((qryndx * maxnsteps + 0) * ndbCstrs + ndbCstrs) * nTTranformMatrix;
                if(SECONDARYUPDATE == SECONDARYUPDATE_CONDITIONAL) {
                    //NOTE: 2nd'ry scores written immediately following tfms
                    bscore2nd = wrkmemtmibest2nd[mloc20 + dbstrndx];
                    ndxCache[2][threadIdx.x] = wrt2ndry = (bscore2nd < bscore);//reuse cache
                }
                if(wrt2ndry) wrkmemtmibest2nd[mloc20 + dbstrndx] = bscore;
            }
            if(GRANDUPDATE)
                ndxCache[1][threadIdx.x] = wrtgrand = (grand < bscore);//reuse cache
            //coalesced WRITE for multiple references
            if(wrtgrand)
                wrkmemaux[mloc0 + tawmvGrandBest * ndbCstrs + dbstrndx] = bscore;
            if(WRITEFRAGINFO && (FORCEWRITEFRAGINFO || wrtgrand)) {
                float frgndx = wrkmemaux[mloc + tawmvSubFragNdxCurrent * ndbCstrs + dbstrndx];
                float frgpos = wrkmemaux[mloc + tawmvSubFragPosCurrent * ndbCstrs + dbstrndx];
                wrkmemaux[mloc0 + tawmvSubFragNdx * ndbCstrs + dbstrndx] = frgndx;
                wrkmemaux[mloc0 + tawmvSubFragPos * ndbCstrs + dbstrndx] = frgpos;
            }
        }
    }

    __syncthreads();

    //NOTE: change indexing so that threadIdx.y refers to a different reference
    sfragfct = ndxCache[0][threadIdx.y];
    wrtgrand = ndxCache[1][threadIdx.y];
    wrt2ndry = ndxCache[2][threadIdx.y];

    __syncthreads();

    //NOTE: change reference structure indexing: threadIdx.x -> threadIdx.y
    dbstrndx = blockIdx.x * blockDim.x + threadIdx.y;

    //READ and WRITE iteration-best transformation matrices
    if(threadIdx.x < nTTranformMatrix && dbstrndx < ndbCstrs) {
        uint mloc0 = ((qryndx * maxnsteps + 0) * ndbCstrs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        float value = 0.0f;
        if(sfragfct != 0 || wrtgrand || wrt2ndry) value = wrkmemtmibest[mloc];//READ from gmem
        if(sfragfct != 0) wrkmemtmibest[mloc0] = value;//WRITE to gmem
        //save this transformation matrix for tfmmem below
        scvCache[threadIdx.y][threadIdx.x] = value;
    }

    __syncthreads();

    //WRITE the transformation matrix with the currently grand best score
    if(wrtgrand && threadIdx.x < nTTranformMatrix && dbstrndx < ndbCstrs) {
        uint tfmloc = (qryndx * ndbCstrs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        tfmmem[tfmloc] = scvCache[threadIdx.y][threadIdx.x];
    }

    //WRITE tfm corresponding to the best-performing over all wrkmemtmibest of multiple passes
    if(wrt2ndry && threadIdx.x < nTTranformMatrix && dbstrndx < ndbCstrs) {
        uint mloc0 = ((qryndx * maxnsteps + 0) * ndbCstrs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        wrkmemtmibest2nd[mloc0] = scvCache[threadIdx.y][threadIdx.x];
    }
}

// -------------------------------------------------------------------------
//Instantiations:
//
#define INSTANTIATE_SaveBestScoreAndTMAmongBests( \
    tpWRITEFRAGINFO,tpGRANDUPDATE,tpFORCEWRITEFRAGINFO,tpSECONDARYUPDATE) \
    template __global__ void SaveBestScoreAndTMAmongBests \
        <tpWRITEFRAGINFO,tpGRANDUPDATE,tpFORCEWRITEFRAGINFO,tpSECONDARYUPDATE>( \
            const uint ndbCstrs, const uint maxnsteps, const uint effnsteps, \
            float* __restrict__ wrkmemtmibest, \
            float* __restrict__ tfmmem, \
            float* __restrict__ wrkmemaux, \
            float* __restrict__ wrkmemtmibest2nd);

INSTANTIATE_SaveBestScoreAndTMAmongBests(false,true,false,SECONDARYUPDATE_NOUPDATE);
INSTANTIATE_SaveBestScoreAndTMAmongBests(false,true,false,SECONDARYUPDATE_UNCONDITIONAL);
INSTANTIATE_SaveBestScoreAndTMAmongBests(false,true,false,SECONDARYUPDATE_CONDITIONAL);
INSTANTIATE_SaveBestScoreAndTMAmongBests(true,true,false,SECONDARYUPDATE_NOUPDATE);
INSTANTIATE_SaveBestScoreAndTMAmongBests(true,true,true,SECONDARYUPDATE_NOUPDATE);

// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// ProductionSaveBestScoresAndTMAmongBests: save best scores and respective 
// transformation matrices by considering all partial best scores 
// calculated over all fragment factors; production version;
// WRITEFRAGINFO, template parameter, whether to save a fragment length 
// index and position for the best score;
// CONDITIONAL, template parameter, flag of whether the grand best score is
// compared with the current best before writing;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps performed for each reference structure;
// effnsteps, effective (actual maximum) number of steps;
// NOTE: memory pointers should be aligned!
// wrkmemtmibest, working memory for iteration-best transformation matrices;
// wrkmemaux, auxiliary working memory;
// alndatamem, memory for full alignment information, including scores;
// tfmmem, memory for transformation matrices;
// 
template<bool WRITEFRAGINFO, bool CONDITIONAL>
__global__
void ProductionSaveBestScoresAndTMAmongBests(
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint effnsteps,
    const float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemaux,
    float* __restrict__ alndatamem,
    float* __restrict__ tfmmem)
{
    //index of the structure (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number
    enum {
        //array index for query
        QRYX = CUS1_TBSP_SCORE_MAX_XDIM,
        //#alignment data entries to write
        nADT = (nTDP2OutputAlnData - dp2oadScoreQ)
    };
    __shared__ float scvCache[CUS1_TBSP_SCORE_MAX_YDIM][CUS1_TBSP_SCORE_MAX_XDIM+1];
    __shared__ uint ndxCache[CUS1_TBSP_SCORE_MAX_YDIM][CUS1_TBSP_SCORE_MAX_XDIM+1];
    __shared__ float adtCache[CUS1_TBSP_SCORE_MAX_XDIM][nADT];
    __shared__ int lenCache[CUS1_TBSP_SCORE_MAX_XDIM+1];

    scvCache[threadIdx.y][threadIdx.x] = 0.0f;
    ndxCache[threadIdx.y][threadIdx.x] = 0;

    if(threadIdx.y < nADT && dbstrndx < ndbCstrs)
        adtCache[threadIdx.x][threadIdx.y] = 0.0f;

    if(threadIdx.y == 0 && dbstrndx < ndbCstrs)
        lenCache[threadIdx.x] = GetDbStrLength(dbstrndx);

    if(threadIdx.y == 1 && threadIdx.x == 0)
        lenCache[QRYX] = GetQueryLength(qryndx);

    //no sync; threads do not access other cells below

    for(uint sfragfct = threadIdx.y; sfragfct < /* maxnsteps */effnsteps; sfragfct += blockDim.y) {
        float bscore = 0.0f;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(dbstrndx < ndbCstrs)//READ, coalesced for multiple references
            bscore = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx];
        if(scvCache[threadIdx.y][threadIdx.x] < bscore) {
            scvCache[threadIdx.y][threadIdx.x] = bscore;
            ndxCache[threadIdx.y][threadIdx.x] = sfragfct;
        }
        //no sync, every thread works in its own space
    }

    __syncthreads();

    //reduce/unroll for max best score over the fragment factors:
    for(int ydim = (CUS1_TBSP_SCORE_MAX_YDIM>>1); ydim >= 1; ydim >>= 1) {
        if(threadIdx.y < ydim &&
            scvCache[threadIdx.y][threadIdx.x] <
            scvCache[threadIdx.y+ydim][threadIdx.x])
        {
            scvCache[threadIdx.y][threadIdx.x] = scvCache[threadIdx.y+ydim][threadIdx.x];
            ndxCache[threadIdx.y][threadIdx.x] = ndxCache[threadIdx.y+ydim][threadIdx.x];
        }

        __syncthreads();
    }

    //scvCache[0][...] now contains maximum
    uint sfragfct = ndxCache[0][threadIdx.x];
    bool wrtgrand = 0;

    //write scores first
    if(threadIdx.y == 0) {
        ndxCache[1][threadIdx.x] = 0;
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        //update the grand best scores
        if(dbstrndx < ndbCstrs) {
            float bscore = scvCache[0][threadIdx.x];
            float grand = 0.0f;
            if(CONDITIONAL)
                grand = wrkmemaux[mloc0 + tawmvGrandBest * ndbCstrs + dbstrndx];
            ndxCache[1][threadIdx.x] = wrtgrand = (grand < bscore);//reuse cache
            //coalesced WRITE for multiple references
            if(wrtgrand) {
                wrkmemaux[mloc0 + tawmvGrandBest * ndbCstrs + dbstrndx] = bscore;
                //score calculated for the longer structure:
                float gbest = wrkmemaux[mloc + tawmvBest0 * ndbCstrs + dbstrndx];
                const float d0Q = GetD0fin(lenCache[QRYX], lenCache[QRYX]);//threshold for query
                const float d0R = GetD0fin(lenCache[threadIdx.x], lenCache[threadIdx.x]);//for reference
                //make bscore (should not be used below associated clauses) represent the query score:
                if(lenCache[threadIdx.x] < lenCache[QRYX]) myhdswap(bscore, gbest);
                //write alignment information in cache:
                adtCache[threadIdx.x][(dp2oadScoreQ-dp2oadScoreQ)] = __fdividef(bscore, lenCache[QRYX]);
                adtCache[threadIdx.x][(dp2oadScoreR-dp2oadScoreQ)] = __fdividef(gbest, lenCache[threadIdx.x]);
                adtCache[threadIdx.x][(dp2oadD0Q-dp2oadScoreQ)] = d0Q;
                adtCache[threadIdx.x][(dp2oadD0R-dp2oadScoreQ)] = d0R;
                if(WRITEFRAGINFO) {
                    float frgndx = wrkmemaux[mloc + tawmvSubFragNdxCurrent * ndbCstrs + dbstrndx];
                    float frgpos = wrkmemaux[mloc + tawmvSubFragPosCurrent * ndbCstrs + dbstrndx];
                    wrkmemaux[mloc0 + tawmvSubFragNdx * ndbCstrs + dbstrndx] = frgndx;
                    wrkmemaux[mloc0 + tawmvSubFragPos * ndbCstrs + dbstrndx] = frgpos;
                }
            }
        }
    }

    __syncthreads();

    //NOTE: change indexing so that threadIdx.y refers to a different reference
    sfragfct = ndxCache[0][threadIdx.y];
    wrtgrand = ndxCache[1][threadIdx.y];

    __syncthreads();

    //NOTE: change reference structure indexing: threadIdx.x -> threadIdx.y
    dbstrndx = blockIdx.x * blockDim.x + threadIdx.y;

    //READ iteration-best transformation matrices
    if(threadIdx.x < nTTranformMatrix && dbstrndx < ndbCstrs) {
        //uint mloc0 = ((qryndx * maxnsteps + 0) * ndbCstrs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        float value = 0.0f;
        if(wrtgrand) value = wrkmemtmibest[mloc];//READ from gmem
        //save this transformation matrix for tfmmem below
        scvCache[threadIdx.y][threadIdx.x] = value;
    }

    __syncthreads();

    //WRITE the transformation matrix with the currently grand best score
    if(wrtgrand && threadIdx.x < nTTranformMatrix && dbstrndx < ndbCstrs) {
        uint tfmloc = (qryndx * ndbCstrs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        tfmmem[tfmloc] = scvCache[threadIdx.y][threadIdx.x];
    }

    //WRITE the transformation matrix with the currently grand best score
    if(wrtgrand && threadIdx.x < nADT && dbstrndx < ndbCstrs) {
        uint mloc = (qryndx * ndbCstrs + dbstrndx) * nTDP2OutputAlnData;
        alndatamem[mloc + dp2oadScoreQ + threadIdx.x] = adtCache[threadIdx.y][threadIdx.x];
    }
}

// -------------------------------------------------------------------------
//Instantiations:
//
#define INSTANTIATE_ProductionSaveBestScoresAndTMAmongBests(tpWRITEFRAGINFO,tpCONDITIONAL) \
    template __global__ void ProductionSaveBestScoresAndTMAmongBests<tpWRITEFRAGINFO,tpCONDITIONAL>( \
        const uint ndbCstrs, const uint maxnsteps, const uint effnsteps, \
        const float* __restrict__ wrkmemtmibest, \
        float* __restrict__ wrkmemaux, \
        float* __restrict__ alndatamem, \
        float* __restrict__ tfmmem);

INSTANTIATE_ProductionSaveBestScoresAndTMAmongBests(true,false);
INSTANTIATE_ProductionSaveBestScoresAndTMAmongBests(false,true);

// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// SaveTopNScoresAndTMsAmongSecondaryBests: save secondary top N scores and 
// respective transformation matrices by considering all partial best scores 
// calculated over all fragment factors; write the information to the first
// N locations of fragment factors;
// depth, superposition depth for calculating query and reference position factors;
// firstit, flag of the first iteration;
// twoconfs, process two configurations of secondary bests scores (with varying pace);
// rfnfragfctinit, initial fragment factor for reference to calculate query and
// reference positions;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps performed for each reference structure;
// effnsteps, effective (actual maximum) number of steps;
// NOTE: memory pointers should be aligned!
// wrkmemtmibest, working memory for iteration-best transformation matrices;
// wrkmemtm, working memory for selected transformation matrices;
// wrkmemaux, auxiliary working memory;
// 
// __launch_bounds__(1024,1)//for tests
__global__
void SaveTopNScoresAndTMsAmongSecondaryBests(
    const int depth,
    const bool firstit,
    const bool twoconfs,
    const int rfnfragfctinit,
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint effnsteps,
    const float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemaux)
{
    enum {
        //length threshold for triggering the analysis of secondary best scores:
        lLENTHR = 150,
        lYDIM = CUS1_TBSP_SCORE_MAX_YDIM,
        lXDIM = CUS1_TBSP_SCORE_MAX_XDIM,
        lQRYNDX = lXDIM,
        lTOPN = CUS1_TBSP_DPSCORE_TOP_N,
        lMAXS = CUS1_TBSP_DPSCORE_TOP_N_MAX_CONFIGS,
        lTOPNxMAXS = lTOPN * lMAXS,
        lTOPN2 = lTOPN * 2
    };
    //index of the structure (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number
    __shared__ float scvCache2[lYDIM][lXDIM+1];
    __shared__ float scvCache3[lYDIM][lXDIM+1];
    __shared__ int ndxCache2[lYDIM][lXDIM+1];
    __shared__ int ndxCache3[lYDIM][lXDIM+1];
    __shared__ int dbstrlen[lXDIM+1];

    //coalesced reads of lengths of multiple reference structures:
    if(threadIdx.y == 1 && threadIdx.x == 0) dbstrlen[lQRYNDX] = GetQueryLength(qryndx);
    if(threadIdx.y == 0 && dbstrndx < ndbCstrs) {
        dbstrlen[threadIdx.x] = GetDbStrLength(dbstrndx);
    }

    scvCache2[threadIdx.x][threadIdx.y] = 0.0f;
    ndxCache2[threadIdx.x][threadIdx.y] = -1;
    // if(threadIdx.y == 0) ndxCache2[threadIdx.x][lYDIM] = 0;//counter
    if(twoconfs) {
        scvCache3[threadIdx.x][threadIdx.y] = 0.0f;
        ndxCache3[threadIdx.x][threadIdx.y] = -1;
        // if(threadIdx.y == 0) ndxCache3[threadIdx.x][lYDIM] = 0;//counter
    }

    //sync for lengths;
    __syncthreads();
 
    if(!firstit) {
        //read previously saved scores (NOTE) in wrkmemtm memory, which is assumed to be large enough!
        uint mloc = ((qryndx * maxnsteps + lTOPNxMAXS) * ndbCstrs) * nTTranformMatrix;
        if(dbstrndx < ndbCstrs && (lLENTHR < dbstrlen[threadIdx.x] || lLENTHR < dbstrlen[lQRYNDX]))
        {
            scvCache2[threadIdx.x][threadIdx.y] =
                wrkmemtm[mloc + threadIdx.y * ndbCstrs + dbstrndx];
            if(twoconfs)
                scvCache3[threadIdx.x][threadIdx.y] =
                    wrkmemtm[mloc + (lTOPN + threadIdx.y) * ndbCstrs + dbstrndx];
        }
    }

    //no sync; threads do not access other cells below

    for(uint sfragfct = threadIdx.y; sfragfct < effnsteps; sfragfct += blockDim.y)
    {
        if(dbstrlen[threadIdx.x] <= lLENTHR && dbstrlen[lQRYNDX] <= lLENTHR)
            continue;//no sync below in this block

        float bscore = 0.0f;
        int qryfragfct, rfnfragfct;

        if(dbstrndx < ndbCstrs)
            GetQryRfnFct_frg2(
                depth, &qryfragfct, &rfnfragfct,
                dbstrlen[lQRYNDX], dbstrlen[threadIdx.x],
                sfragfct, rfnfragfctinit);

        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;

        bool sec2met =
            (dbstrndx < ndbCstrs) &&
            ((dbstrlen[lQRYNDX] <= lLENTHR) || ((qryfragfct & 1) == 0)) && 
            ((dbstrlen[threadIdx.x] <= lLENTHR) || ((rfnfragfct & 1) == 0));

        bool sec3met =
            twoconfs && (dbstrndx < ndbCstrs) &&
            ((dbstrlen[lQRYNDX] <= lLENTHR) || (myfastmod3(qryfragfct) == 0)) && 
            ((dbstrlen[threadIdx.x] <= lLENTHR) || (myfastmod3(rfnfragfct) == 0));

        if(dbstrndx < ndbCstrs && (sec2met || sec3met))//coalesced READ
            bscore = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx];

        if(sec2met) {
            // int locndx = atomicAdd(&ndxCache2[threadIdx.x][lYDIM], 1);
            // ndxCache2[threadIdx.x][lYDIM] = ndxCache2[threadIdx.x][lYDIM] & (lYDIM-1);
            int locndx = threadIdx.y;
            if(scvCache2[threadIdx.x][locndx] < bscore) {
                scvCache2[threadIdx.x][locndx] = bscore;
                ndxCache2[threadIdx.x][locndx] = sfragfct;
            }
        }

        if(sec3met) {
            // int locndx = atomicAdd(&ndxCache3[threadIdx.x][lYDIM], 1);
            // ndxCache3[threadIdx.x][lYDIM] = ndxCache3[threadIdx.x][lYDIM] & (lYDIM-1);
            int locndx = threadIdx.y;
            if(scvCache3[threadIdx.x][locndx] < bscore) {
                scvCache3[threadIdx.x][locndx] = bscore;
                ndxCache3[threadIdx.x][locndx] = sfragfct;
            }
        }

        //no sync, every thread works in its own space
    }

    __syncthreads();

    //NOTE: do not sort.
    // //sort scores and accompanying indices:
    // BatcherSortYDIMparallel<lYDIM,false/*descending*/>(
    //     lYDIM, scvCache2[threadIdx.x], ndxCache2[threadIdx.x]);
    // //no sync: sync'ed in BatcherSortYDIMparallel

    // if(twoconfs) {
    //     //synced inside
    //     BatcherSortYDIMparallel<lYDIM,false/*descending*/>(
    //         lYDIM, scvCache3[threadIdx.x], ndxCache3[threadIdx.x]);
    // }

    //scvCache[...][0..lTOPN-1]now contain top N scores
    //write scores first
    if(threadIdx.y < lTOPN && threadIdx.y < maxnsteps) {
        uint mloc = ((qryndx * maxnsteps + lTOPNxMAXS) * ndbCstrs) * nTTranformMatrix;
        if(dbstrndx < ndbCstrs) {
            //int sfragfct2 = ndxCache2[threadIdx.x][threadIdx.y];//can be <0
            wrkmemtm[mloc + threadIdx.y * ndbCstrs + dbstrndx] =
                scvCache2[threadIdx.x][threadIdx.y];
            if(twoconfs) {
                //int sfragfct3 = ndxCache3[threadIdx.x][threadIdx.y];//can be <0
                wrkmemtm[mloc + (lTOPN + threadIdx.y) * ndbCstrs + dbstrndx] =
                    scvCache3[threadIdx.x][threadIdx.y];
            }
        }
    }

    //NOTE: no sync as long as caches are not overwritten from the last sync:
    //__syncthreads();

    //NOTE: change reference structure indexing: threadIdx.x -> threadIdx.y
    dbstrndx = blockIdx.x * blockDim.x + threadIdx.y;

    constexpr int nmtxs = lXDIM / nTTranformMatrix;
    int ndx = 0;//relative reference index
    for(int i = 1; i < nmtxs; i++)
        if(i * nTTranformMatrix <= threadIdx.x) ndx = i;

    //READ and WRITE iteration-best transformation matrices;
    //rearrange lTOPN best performing mtxs at the first slots (sfragfct indices)
    for(int sx = 0; sx < lTOPN; sx += nmtxs) {
        if(threadIdx.x < nTTranformMatrix * nmtxs && 
           threadIdx.x < nTTranformMatrix * (ndx+1) && dbstrndx < ndbCstrs) {
            uint tid = threadIdx.x - nTTranformMatrix * ndx;
            uint sxx = sx + ndx;
            if(sxx < lTOPN && sxx < maxnsteps) {
                //NOTE: indexing changed so that threadIdx.y refers to a different reference
                int sfragfct = ndxCache2[threadIdx.y][sxx];
                uint mlocs = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix + tid;
                uint mloct = ((qryndx * maxnsteps + (lTOPN + sxx)) * ndbCstrs + dbstrndx) * nTTranformMatrix + tid;
                if(0 <= sfragfct)
                    wrkmemtm[mloct] = wrkmemtmibest[mlocs];//GMEM READ/WRITE
                else if(firstit) {
                    //initialize otherwise; GMEM READ/WRITE
                    wrkmemtm[mloct] = (tid==tfmmRot_0_0 || tid==tfmmRot_1_1 || tid==tfmmRot_2_2)? 1.0f: 0.0f;
                }
                if(twoconfs) {
                    sfragfct = ndxCache3[threadIdx.y][sxx];
                    mlocs = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix + tid;
                    mloct = ((qryndx * maxnsteps + (lTOPN2 + sxx)) * ndbCstrs + dbstrndx) * nTTranformMatrix + tid;
                    if(0 <= sfragfct)
                        wrkmemtm[mloct] = wrkmemtmibest[mlocs];//GMEM READ/WRITE
                    else if(firstit) {
                        //initialize otherwise; GMEM READ/WRITE
                        wrkmemtm[mloct] = (tid==tfmmRot_0_0 || tid==tfmmRot_1_1 || tid==tfmmRot_2_2)? 1.0f: 0.0f;
                    }
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// SaveTopNScoresAndTMsAmongBests: save top N scores and respective 
// transformation matrices by considering all partial best scores 
// calculated over all fragment factors; write the information to the first
// N locations of fragment factors;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps performed for each reference structure;
// effnsteps, effective (actual maximum) number of steps;
// NOTE: memory pointers should be aligned!
// wrkmemtmibest, working memory for iteration-best transformation matrices;
// wrkmemtm, working memory for selected transformation matrices;
// wrkmemaux, auxiliary working memory;
// 
__global__
void SaveTopNScoresAndTMsAmongBests(
    const uint ndbCstrs,
    const uint maxnsteps,
//     const uint effnsteps,
    const float* __restrict__ wrkmemtmibest,
    float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemaux)
{
    //index of the structure (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number
    __shared__ float scvCache[CUS1_TBSP_SCORE_MAX_YDIM][CUS1_TBSP_SCORE_MAX_XDIM+1];
    __shared__ uint ndxCache[CUS1_TBSP_SCORE_MAX_YDIM][CUS1_TBSP_SCORE_MAX_XDIM+1];

    scvCache[threadIdx.x][threadIdx.y] = 0.0f;
    ndxCache[threadIdx.x][threadIdx.y] = 0;

    //no sync; threads do not access other cells below

    for(uint sfragfct = threadIdx.y; sfragfct < maxnsteps/*effnsteps*/; sfragfct += blockDim.y) {
        float bscore = 0.0f;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(dbstrndx < ndbCstrs)//READ, coalesced for multiple references
            bscore = wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx];
        if(scvCache[threadIdx.x][threadIdx.y] < bscore) {
            scvCache[threadIdx.x][threadIdx.y] = bscore;
            ndxCache[threadIdx.x][threadIdx.y] = sfragfct;
        }
        //no sync, every thread works in its own space
    }

    __syncthreads();

    //sort scores and accompanying indices:
    BatcherSortYDIMparallel<CUS1_TBSP_SCORE_MAX_YDIM,false/*descending*/>(
        CUS1_TBSP_SCORE_MAX_YDIM, scvCache[threadIdx.x], ndxCache[threadIdx.x]);
    //no sync: sync'ed in BatcherSortYDIMparallel

    //scvCache[...][0..CUS1_TBSP_DPSCORE_TOP_N-1]now contain top N scores
    uint sfragfct = ndxCache[threadIdx.x][threadIdx.y];

    //write scores first
    if(threadIdx.y < CUS1_TBSP_DPSCORE_TOP_N && threadIdx.y < maxnsteps) {
        uint mloc = ((qryndx * maxnsteps + threadIdx.y) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(dbstrndx < ndbCstrs) {
            float bscore = scvCache[threadIdx.x][threadIdx.y];
            int convflag = 0;
            if(threadIdx.y == 0)
                convflag = wrkmemaux[mloc + tawmvConverged * ndbCstrs + dbstrndx];
            if(0.0f < bscore) convflag = convflag & (~CONVERGED_SCOREDP_bitval);//reset
            else convflag = convflag | CONVERGED_SCOREDP_bitval;//set
            //adjust global/local convergence
            wrkmemaux[mloc + tawmvConverged * ndbCstrs + dbstrndx] = convflag;
            //coalesced WRITE for multiple references
            if(sfragfct != threadIdx.y)
                wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx] = bscore;
        }
    }

    //NOTE: no sync as long as caches are not overwritten from the last sync:
    //__syncthreads();

    //NOTE: change reference structure indexing: threadIdx.x -> threadIdx.y
    dbstrndx = blockIdx.x * blockDim.x + threadIdx.y;

    constexpr int nmtxs = CUS1_TBSP_SCORE_MAX_XDIM / nTTranformMatrix;
    int ndx = 0;//relative reference index
    for(int i = 1; i < nmtxs; i++)
        if(i * nTTranformMatrix <= threadIdx.x) ndx = i;

    //READ and WRITE iteration-best transformation matrices;
    //rearrange CUS1_TBSP_DPSCORE_TOP_N best performing mtxs at the first slots (sfragfct indices)
    for(int sx = 0; sx < CUS1_TBSP_DPSCORE_TOP_N; sx += nmtxs) {
        if(threadIdx.x < nTTranformMatrix * nmtxs && 
           threadIdx.x < nTTranformMatrix * (ndx+1) && dbstrndx < ndbCstrs) {
            uint tid = threadIdx.x - nTTranformMatrix * ndx;
            uint sxx = sx + ndx;
            if(sxx < CUS1_TBSP_DPSCORE_TOP_N && sxx < maxnsteps) {
                //NOTE: indexing changed so that threadIdx.y refers to a different reference
                sfragfct = ndxCache[threadIdx.y][sxx];
                float bscore = scvCache[threadIdx.y][sxx];
                if(0.0f < bscore) {
                    uint mlocs = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix + tid;
                    uint mloct = ((qryndx * maxnsteps + sxx) * ndbCstrs + dbstrndx) * nTTranformMatrix + tid;
                    wrkmemtm[mloct] = wrkmemtmibest[mlocs];//GMEM READ/WRITE
                }
            }
        }
    }
}

// -------------------------------------------------------------------------



#if 0
// -------------------------------------------------------------------------
// SaveBestDPscoreAndTMAmongDPswifts: save best DP scores and respective 
// transformation matrices by considering all partial DP swift scores 
// calculated over all fragment factors; write the information to the 
// location of fragment factor 0;
// WRITEFRAGINFO, template parameter, write fragment specifications too;
// READSCORE, read previously written DP swift score;
// STEPx5, template parameter, multiply the step by 5 when calculating 
// query and reference positions;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db reference structure positions in a chunk;
// dbxpad, number of padded positions for memory alignment;
// maxnsteps, max number of steps performed for each reference structure;
// effnsteps, effective (actual maximum) number of steps;
// qryfragfct, fragment factor for query (to be multiplied by step dependent 
// upon lengths);
// rfnfragfct, fragment factor for reference;
// fragndx, fragment index determining the fragment length;
// NOTE: memory pointers should be aligned!
// wrkmemtmtarget, working memory for iteration-best (target) transformation matrices;
// tfmmem, memory for transformation matrices;
// wrkmemaux, auxiliary working memory;
// 
// template<bool WRITEFRAGINFO, bool READSCORE, bool STEPx5>
__global__
void SaveBestDPscoreAndTMAmongDPswifts(
    bool WRITEFRAGINFO, bool READSCORE,bool STEPx5,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const uint effnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    const float* __restrict__ tmpdpdiagbuffers,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmtarget,
    float* __restrict__ wrkmemaux)
{
    //index of the structure (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    uint qryndx = blockIdx.y;//query serial number
    __shared__ float scvCache[CUS1_TBSP_SCORE_MAX_YDIM][CUS1_TBSP_SCORE_MAX_XDIM+1];
    __shared__ uint ndxCache[CUS1_TBSP_SCORE_MAX_YDIM][CUS1_TBSP_SCORE_MAX_XDIM+1];
    //distances in positions to the beginnings of the reference structures:
    __shared__ uint dbstrdst[CUS1_TBSP_SCORE_MAX_XDIM];
    //lengths of references; the last index is for the query
    __shared__ int dbstrlen[CUS1_TBSP_SCORE_MAX_XDIM+1];
    int qrylen;//query length

    scvCache[threadIdx.y][threadIdx.x] = 0.0f;
    ndxCache[threadIdx.y][threadIdx.x] = 0;

    //coalesced reads of lengths an addresses of multiple reference structures:
    if(dbstrndx < ndbCstrs) {
        if(threadIdx.y == 0) dbstrlen[threadIdx.x] = GetDbStrLength(dbstrndx);
        if(threadIdx.y == 1) dbstrdst[threadIdx.x] = GetDbStrDst(dbstrndx);
    }

    if(threadIdx.y == 2 && threadIdx.x == 0)
        dbstrlen[CUS1_TBSP_SCORE_MAX_XDIM] = GetQueryLength(qryndx);

    __syncthreads();

    qrylen = dbstrlen[CUS1_TBSP_SCORE_MAX_XDIM];

    //no sync; dbstrlen cache will not be overwritten below!

    for(uint sfragfct = threadIdx.y; sfragfct < effnsteps; sfragfct += blockDim.y) {
        float dpscore = 0.0f;
        int dblen = ndbCposs + dbxpad;
        int yofff = (qryndx * maxnsteps + sfragfct) * dblen;
        int doffs = nTDPDiagScoreSections * nTDPDiagScoreSubsections * yofff;

        if(dbstrndx < ndbCstrs)
            //READ; uncoalesced for multiple references, but rare kernel call and 
            //compact thread block packaging counterbalance this inefficiency;
            //NOTE: last score is considered: assumed no banded alignment;
            dpscore = tmpdpdiagbuffers[doffs + dbstrdst[threadIdx.x] + dbstrlen[threadIdx.x]-1];

        if(scvCache[threadIdx.y][threadIdx.x] < dpscore) {
            scvCache[threadIdx.y][threadIdx.x] = dpscore;
            ndxCache[threadIdx.y][threadIdx.x] = sfragfct;
        }
        //no sync, every thread works in its own space
    }

    __syncthreads();

    //reduce/unroll for max best score over the fragment factors:
    for(int ydim = (CUS1_TBSP_SCORE_MAX_YDIM>>1); ydim >= 1; ydim >>= 1) {
        if(threadIdx.y < ydim &&
            scvCache[threadIdx.y][threadIdx.x] <
            scvCache[threadIdx.y+ydim][threadIdx.x])
        {
            scvCache[threadIdx.y][threadIdx.x] = scvCache[threadIdx.y+ydim][threadIdx.x];
            ndxCache[threadIdx.y][threadIdx.x] = ndxCache[threadIdx.y+ydim][threadIdx.x];
        }

        __syncthreads();
    }

    //scvCache[0][...] now contains maximum
    uint sfragfct = ndxCache[0][threadIdx.x];
    bool wrtscore = 0;

    //write scores first
    if(threadIdx.y == 0) {
        ndxCache[1][threadIdx.x] = 0;
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        //uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(dbstrndx < ndbCstrs) {
            float prevdpscore = -1.0f;
            float dpscore = scvCache[0][threadIdx.x];
            //coalesced READ/WRITE for multiple references
            if(READSCORE) prevdpscore = wrkmemaux[mloc0 + tawmvScore * ndbCstrs + dbstrndx];
            ndxCache[1][threadIdx.x] = wrtscore = (prevdpscore < dpscore);//reuse cache
            if(wrtscore) {
                wrkmemaux[mloc0 + tawmvScore * ndbCstrs + dbstrndx] = dpscore;
                if(WRITEFRAGINFO) {
                    int qrypos, rfnpos;
                    GetQryRfnPos_varP(STEPx5, qrypos, rfnpos,
                        qrylen, dbstrlen[threadIdx.x], sfragfct, qryfragfct, rfnfragfct, fragndx);
                    float frgndx = fragndx;//frag length to be determined by GetNAlnPoss_var
                    float frgpos = 0.0f;//fragpos is 0 for this particular calculation
                    wrkmemaux[mloc0 + tawmvQRYpos * ndbCstrs + dbstrndx] = qrypos;
                    wrkmemaux[mloc0 + tawmvRFNpos * ndbCstrs + dbstrndx] = rfnpos;
                    wrkmemaux[mloc0 + tawmvSubFragNdx * ndbCstrs + dbstrndx] = frgndx;
                    wrkmemaux[mloc0 + tawmvSubFragPos * ndbCstrs + dbstrndx] = frgpos;
                }
            }
        }
    }

    __syncthreads();

    //NOTE: change indexing so that threadIdx.y refers to a different reference
    sfragfct = ndxCache[0][threadIdx.y];
    wrtscore = ndxCache[1][threadIdx.y];

    __syncthreads();

    //NOTE: change reference structure indexing: threadIdx.x -> threadIdx.y
    dbstrndx = blockIdx.x * blockDim.x + threadIdx.y;

    //READ and WRITE iteration-best transformation matrices
    if(threadIdx.x < nTTranformMatrix && dbstrndx < ndbCstrs) {
        uint mloc0 = ((qryndx * maxnsteps + 0) * ndbCstrs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix + threadIdx.x;
        if(wrtscore) wrkmemtmtarget[mloc0] = wrkmemtm[mloc];//READ/WRITE to gmem
    }
}

// -------------------------------------------------------------------------
//Instantiations:
//
// #define INSTANTIATE_SaveBestDPscoreAndTMAmongDPswifts(tpWRITEFRAGINFO,tpREADSCORE,tpSTEPx5) 
//     template __global__ void SaveBestDPscoreAndTMAmongDPswifts<tpWRITEFRAGINFO,tpREADSCORE,tpSTEPx5>( 
//         const uint ndbCstrs, const uint ndbCposs, const uint dbxpad, 
//         const uint maxnsteps, const uint effnsteps, 
//         int qryfragfct, int rfnfragfct, int fragndx, 
//         const float* __restrict__ tmpdpdiagbuffers, 
//         const float* __restrict__ wrkmemtm, 
//         float* __restrict__ wrkmemtmtarget, 
//         float* __restrict__ wrkmemaux);
// 
// INSTANTIATE_SaveBestDPscoreAndTMAmongDPswifts(false,false,false);
// INSTANTIATE_SaveBestDPscoreAndTMAmongDPswifts(false,false,true);
// INSTANTIATE_SaveBestDPscoreAndTMAmongDPswifts(false,true,false);
// INSTANTIATE_SaveBestDPscoreAndTMAmongDPswifts(false,true,true);

// -------------------------------------------------------------------------
#endif



// -------------------------------------------------------------------------
// SortBestDPscoresAndTMsAmongDPswifts: sort best DP scores and then save 
// them along with respective transformation matrices by considering all 
// partial DP swift scores calculated over all fragment factors; write the 
// information to the first fragment factor locations;
// nbranches, #final superposition-stage branches for further exploration
// (CUS1_TBSP_DPSCORE_TOP_N_REFINEMENT);
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db reference structure positions in a chunk;
// dbxpad, number of padded positions for memory alignment;
// maxnsteps, max number of steps performed for each reference structure;
// effnsteps, effective (actual maximum) number of steps;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, memory section of DP scores;
// wrkmemtm, input working memory of calculated transformation matrices;
// wrkmemtmtarget, working memory for iteration-best (target) transformation matrices;
// wrkmemaux, auxiliary working memory;
// 
__global__
void SortBestDPscoresAndTMsAmongDPswifts(
    const uint nbranches,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const float* __restrict__ tmpdpdiagbuffers,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ wrkmemtmtarget,
    float* __restrict__ wrkmemaux)
{
    enum {
        lYDIM = CUS1_TBSP_SCORE_MAX_YDIM,
        lXDIM = CUS1_TBSP_SCORE_MAX_XDIM,
        lTOPN = CUS1_TBSP_DPSCORE_TOP_N,
        lREFN = CUS1_TBSP_DPSCORE_TOP_N_REFINEMENT,
        lMAXS = CUS1_TBSP_DPSCORE_TOP_N_MAX_CONFIGS,
        lREFNxMAXS = lREFN * lMAXS
    };
    //index of the structure (blockIdx.x, refn. serial number):
    uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qryndx = blockIdx.y;//query serial number
    const uint secndx = blockIdx.z;//score section index
    __shared__ float scvCache[lYDIM][lXDIM+1];
    __shared__ uint ndxCache[lYDIM][lXDIM+1];
    //distances in positions to the beginnings of the reference structures:
    __shared__ uint dbstrdst[lXDIM+1];
    //lengths of references; the last index is for the query
    __shared__ int dbstrlen[lXDIM+1];

    scvCache[threadIdx.x][threadIdx.y] = 0.0f;
    ndxCache[threadIdx.x][threadIdx.y] = 0;

    //coalesced reads of lengths an addresses of multiple reference structures:
    if(dbstrndx < ndbCstrs) {
        if(threadIdx.y == 0) dbstrlen[threadIdx.x] = GetDbStrLength(dbstrndx);
        if(threadIdx.y == 1) dbstrdst[threadIdx.x] = GetDbStrDst(dbstrndx);
    }

    __syncthreads();

    for(uint sfragfct = secndx * lTOPN + threadIdx.y; 
        sfragfct < (secndx + 1) * lTOPN && sfragfct < maxnsteps; 
        sfragfct += blockDim.y)
    {
        float dpscore = 0.0f;
        int dblen = ndbCposs + dbxpad;
        int yofff = (qryndx * maxnsteps + sfragfct) * dblen;
        int doffs = nTDPDiagScoreSections * nTDPDiagScoreSubsections * yofff;

        int convflag = CONVERGED_SCOREDP_bitval;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(dbstrndx < ndbCstrs)
            convflag = wrkmemaux[mloc + tawmvConverged * ndbCstrs + dbstrndx];

        if(convflag == 0/* dbstrndx < ndbCstrs */)
            //READ; uncoalesced for multiple references, but rare kernel call and 
            //compact thread block packaging counterbalance this inefficiency;
            //NOTE: last score is considered: assumed no banded alignment;
            dpscore = tmpdpdiagbuffers[doffs + dbstrdst[threadIdx.x] + dbstrlen[threadIdx.x]-1];

        if(scvCache[threadIdx.x][threadIdx.y] < dpscore) {
            scvCache[threadIdx.x][threadIdx.y] = dpscore;
            ndxCache[threadIdx.x][threadIdx.y] = sfragfct;
        }
        //no sync, every thread works in its own space
    }

    __syncthreads();

    //sort scores and accompanying indices:
    BatcherSortYDIMparallel<lTOPN,false/*descending*/>(
        lTOPN, scvCache[threadIdx.x], ndxCache[threadIdx.x]);
    //no sync: sync'ed in BatcherSortYDIMparallel

    //scvCache[...][0..lTOPN-1] now contain s
    //uint sfragfct = ndxCache[threadIdx.x][threadIdx.y];

    //write scores first (NOTE: scores required only for testing):
    if(threadIdx.y < lTOPN && (secndx * lTOPN + threadIdx.y) < maxnsteps) {
        uint mloc = ((qryndx * maxnsteps + (secndx * lTOPN + threadIdx.y)) * nTAuxWorkingMemoryVars) * ndbCstrs;
        if(dbstrndx < ndbCstrs) {
            float dpscore = scvCache[threadIdx.x][threadIdx.y];
            //NOTE: convergence flags have to be set/reset again after new sorting.
            int convflag = 0;
            if(threadIdx.y == 0)
                convflag = wrkmemaux[mloc + tawmvConverged * ndbCstrs + dbstrndx];
            if(0.0f < dpscore) convflag = convflag & (~CONVERGED_SCOREDP_bitval);//reset
            else convflag = convflag | CONVERGED_SCOREDP_bitval;//set
            //adjust global/local convergence
            wrkmemaux[mloc + tawmvConverged * ndbCstrs + dbstrndx] = convflag;
            //coalesced WRITE for multiple references (for test)
            // wrkmemaux[mloc + tawmvScore * ndbCstrs + dbstrndx] = dpscore;
            // reset score/may not be reinitialized during refinement (bugfix: 231110/v0.13):
            wrkmemaux[mloc + tawmvBestScore * ndbCstrs + dbstrndx] = 0.0f;
        }
    }

    //NOTE: no sync as long as caches are not overwritten from the last sync:
    //__syncthreads();

    //NOTE: change reference structure indexing: threadIdx.x -> threadIdx.y
    dbstrndx = blockIdx.x * blockDim.x + threadIdx.y;

    constexpr int nmtxs = lXDIM / nTTranformMatrix;
    int ndx = 0;//relative reference index
    for(int i = 1; i < nmtxs; i++)
        if(i * nTTranformMatrix <= threadIdx.x) ndx = i;

    //READ and WRITE iteration-best transformation matrices;
    //rearrange lREFN best performing mtxs at the first slots (sfragfct indices)
    for(int sx = 0; sx < (int)nbranches; sx += nmtxs) {
        if(threadIdx.x < nTTranformMatrix * nmtxs && 
           threadIdx.x < nTTranformMatrix * (ndx+1) && dbstrndx < ndbCstrs) {
            uint tid = threadIdx.x - nTTranformMatrix * ndx;
            uint sxx = sx + ndx;
            if(sxx < nbranches && sxx < maxnsteps) {
                //NOTE: indexing changed so that threadIdx.y refers to a different reference
                uint sfragfct = ndxCache[threadIdx.y][sxx];
                float dpscore = scvCache[threadIdx.y][sxx];
                if(0.0f < dpscore) {
                    uint mlocs = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix + tid;
                    //NOTE: lREFN for target tms!
                    uint mloct = ((qryndx * lREFNxMAXS + (secndx * nbranches + sxx)) * ndbCstrs + dbstrndx) *
                            nTTranformMatrix + tid;
                    wrkmemtmtarget[mloct] = wrkmemtm[mlocs];//GMEM READ/WRITE
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
