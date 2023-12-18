/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __production_match2aln_cuh__
#define __production_match2aln_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gproc/btckcoords.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fields.cuh"

// =========================================================================
// ProductionMatchToAlignment32x: produce the final alignment with 
// accompanying information from the given match (aligned) positions; 
// use 32x unrolling;
__global__
void ProductionMatchToAlignment32x(
    const bool nodeletions,
    const float d2equiv,
    // const uint nqystrs,
    // const uint nqyposs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ alndatamem,
    char* __restrict__ alnsmem
);

// =========================================================================
// WriteAlignmentFragment: write alignment fragment to gmem
// written, alignment length written already;
// lentowrite, length of alignment fragment to write;
// lentocheck, alignment fragment length to check for modification;
// outAlnCache, alignment fragment cache;
// alnsmem, global memory of alignments;
// 
template<int blockDim_x1>
__device__ __forceinline__
int WriteAlignmentFragment(
    const uint qrydst, const uint dbstrdst,
    const int alnofff, const int dbalnlen, const int dbalnbeg,
    const int written, const int lentowrite, const int lentocheck,
    int* __restrict__ outAlnCache,
    char* __restrict__ alnsmem)
{
    int wpos, idnts = 0;

    if(threadIdx.x < lentocheck) {
        if(threadIdx.y == 0 && (wpos = outAlnCache[dp2oaQuery * blockDim_x1 + threadIdx.x]) <= 0)
            outAlnCache[dp2oaQuery * blockDim_x1 + threadIdx.x] = GetQueryRsd(qrydst - wpos);
        if(threadIdx.y == 1 && (wpos = outAlnCache[dp2oaQuerySSS * blockDim_x1 + threadIdx.x]) <= 0)
            outAlnCache[dp2oaQuerySSS * blockDim_x1 + threadIdx.x] = GetQuerySS(qrydst - wpos);
        if(threadIdx.y == 2 && (wpos = outAlnCache[dp2oaTarget * blockDim_x1 + threadIdx.x]) <= 0)
            outAlnCache[dp2oaTarget * blockDim_x1 + threadIdx.x] = GetDbStrRsd(dbstrdst - wpos);
        if(threadIdx.y == 0 && (wpos = outAlnCache[dp2oaTargetSSS * blockDim_x1 + threadIdx.x]) <= 0)
            outAlnCache[dp2oaTargetSSS * blockDim_x1 + threadIdx.x] = GetDbStrSS(dbstrdst - wpos);
    }

    __syncthreads();

    if(threadIdx.x < lentocheck) {
        if(threadIdx.y == 0) {
            int mc = outAlnCache[dp2oaMiddle * blockDim_x1 + threadIdx.x];
            int r1 = outAlnCache[dp2oaQuery * blockDim_x1 + threadIdx.x];
            int r2 = outAlnCache[dp2oaTarget * blockDim_x1 + threadIdx.x];
            idnts = (/* r1 &&  */r1 != '-' && r1 == r2);
            if(idnts)
                outAlnCache[dp2oaMiddle * blockDim_x1 + threadIdx.x] =
                    (mc == '+')? r1: (r1|32)/*lower*/;
        }
    }

    __syncthreads();

    if(threadIdx.x < lentowrite) {//WRITE:
        for(int i = threadIdx.y; i < nTDP2OutputAlignmentSSS; i += CUDP_PRODUCTION_ALN_DIM_Y)
            alnsmem[alnofff + dbalnbeg + written + dbalnlen * i] =
                (char)(outAlnCache[i * blockDim_x1 + threadIdx.x]);
    }

    //NOTE: sum in a warp: warp-sync along axis y==0 drastically expands #registers:
    /* if(threadIdx.y == 0) */ idnts = mywarpreducesum(idnts);

    return idnts;
}

// =========================================================================
// -------------------------------------------------------------------------

#endif//__production_match2aln_cuh__
