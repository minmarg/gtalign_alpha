/***************************************************************************
 *   Copyright (C) 2021-2024 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cucom/cutemplates.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/fields.cuh"
#include "production_2tmscore.cuh"

// -------------------------------------------------------------------------
// Production2TMscores: calculate secondary TM-scores, 2TM-scores, and write
// them to memory;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of reference positions in the chunk;
// dbxpad, #pad positions along the dimension of reference structures;
// maxnsteps, total number of steps performed for each reference structure;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// tfmmem, memory for transformation matrices;
// alndatamem, memory for full alignment information, including scores;
// 
__global__ 
void Production2TMscores(
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ tfmmem,
    float* __restrict__ alndatamem)
{
    uint dbstrndx = blockIdx.x;//reference serial number
    uint qryndx = blockIdx.y;//query serial number
    //cache for scores: add 1 to avoid bank conflicts:
    __shared__ float tfmCache[nTTranformMatrix + 1 + CUDP_PRODUCTION_2TMSCORE_DIM_X];
    float* scvCache = tfmCache + nTTranformMatrix + 1;

    int qrylenorg, dbstrlenorg;//original query and reference lengths
    int qrylen, dbstrlen;//pseudo query and reference length, #matched positions
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;
    enum {qrypos = 0, rfnpos = 0, tmpcslot = tawmvEndOfCCMDat/*<16-28*/};


    if(threadIdx.x == 0) {
        uint mloc = ((qryndx * maxnsteps + 0/*sfragfctxndx*/) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        scvCache[6] = wrkmemaux[mloc + dbstrndx];
    }

    __syncthreads();

    if(((int)(scvCache[6])) & (CONVERGED_LOWTMSC_bitval))
        //(the termination flag for this pair is set);
        //all threads in the block exit;
        return;

    //NOTE: no sync as long cache cell for convergence is not overwritten;


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse cache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(dbstrndx, (int*)scvCache + tmpcslot);
        GetQueryLenDst(qryndx, (int*)scvCache + tmpcslot + 2);
    }

    //NOTE: use a different warp for structure-specific-formatted data;
#if (CUDP_PRODUCTION_2TMSCORE_DIM_X >= 64)
    if(threadIdx.x == tawmvNAlnPoss + 32) {
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        scvCache[threadIdx.x-32] = wrkmemaux[mloc0 + (threadIdx.x-32) * ndbCstrs + dbstrndx];
    }
#else
    if(threadIdx.x == tawmvNAlnPoss) {
        //NOTE: reuse cache to read values written at sfragfct==0:
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        scvCache[threadIdx.x] = wrkmemaux[mloc0 + threadIdx.x * ndbCstrs + dbstrndx];
    }
#endif

#if (CUDP_PRODUCTION_2TMSCORE_DIM_X >= 128)
    if(64 <= threadIdx.x && threadIdx.x < nTTranformMatrix + 64) {
        //globally best transformation matrix for a pair:
        uint mloc0 = (qryndx * ndbCstrs + dbstrndx) * nTTranformMatrix;
        tfmCache[threadIdx.x-64] = tfmmem[mloc0 + threadIdx.x-64];
    }
#elif (CUDP_PRODUCTION_2TMSCORE_DIM_X >= 64)
    if(32 <= threadIdx.x && threadIdx.x < nTTranformMatrix + 32) {
        //globally best transformation matrix for a pair:
        uint mloc0 = (qryndx * ndbCstrs + dbstrndx) * nTTranformMatrix;
        tfmCache[threadIdx.x-32] = tfmmem[mloc0 + threadIdx.x-32];
    }
#else
    if(threadIdx.x < nTTranformMatrix) {
        //globally best transformation matrix for a pair:
        uint mloc0 = (qryndx * ndbCstrs + dbstrndx) * nTTranformMatrix;
        tfmCache[threadIdx.x] = tfmmem[mloc0 + threadIdx.x];
    }
#endif

    __syncthreads();


    dbstrdst = ((int*)scvCache)[tmpcslot+1];
    qrydst = ((int*)scvCache)[tmpcslot+3];
    qrylen = dbstrlen = scvCache[tawmvNAlnPoss];
    dbstrlenorg = ((int*)scvCache)[tmpcslot+0];
    qrylenorg = ((int*)scvCache)[tmpcslot+2];

    __syncthreads();

    //threshold calculated for the original lengths
    const float d0 = GetD0fin(qrylenorg, dbstrlenorg);
    const float d02 = SQRD(d0);


    Calc2TMscoresUnrl_Complete(
        qryndx, ndbCposs, dbxpad, maxnsteps,
        qrydst, dbstrdst, qrylen, dbstrlen, qrypos, rfnpos, d02,
        tmpdpalnpossbuffer, tfmCache, scvCache);

    const float best = scvCache[0];//score: synced inside the above function;

    //sync for scvCache[0] not to be overwritten by other warps:
    __syncthreads();


    //calculate the score for the larger structure of the two:
    //threshold calculated for the greater length
    const int greaterlen = myhdmax(qrylenorg, dbstrlenorg);
    const float g0 = GetD0fin(greaterlen, greaterlen);
    const float g02 = SQRD(g0);
    float gbest = best;//score calculated for the other structure

    if(qrylenorg != dbstrlenorg) {
        Calc2TMscoresUnrl_Complete(
            qryndx, ndbCposs, dbxpad, maxnsteps,
            qrydst, dbstrdst, qrylen, dbstrlen, qrypos, rfnpos, g02,
            tmpdpalnpossbuffer, tfmCache, scvCache);
        gbest = scvCache[0];
    }


    //NOTE: scvCache not overwritten any more;
    //NOTE: write directly to production output memory:
    SaveBestQR2TMscores_Complete(
        best, gbest, qryndx, dbstrndx, ndbCstrs, qrylenorg, dbstrlenorg,
        alndatamem);
}

// -------------------------------------------------------------------------
// =========================================================================
