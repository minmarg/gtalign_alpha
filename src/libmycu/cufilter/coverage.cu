/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fields.cuh"
#include "coverage.cuh"

// -------------------------------------------------------------------------
// CheckMaxCoverage: calculate maximum coverage between the queries and 
// reference structures and set the skip flag (convergence) if it is 
// below the threshold; the function actually resets convergence for the 
// pairs within the threshold;
// covthreshold, coverage threshold;
// ntotqstrs, total #queries processed and being processed so far;
// // nqystrs, total number of query structures in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps to perform for each reference structure;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// NOTE: thread block is 1D and processes query-reference structure pairs;
// 
__global__ void CheckMaxCoverage(
    const float covthreshold,
    const int ntotqstrs,
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux)
{
    //index of the reference structure:
    const uint dbstrndx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint qryndx = blockIdx.y;//query serial number
    const uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;

    if(ndbCstrs <= dbstrndx) return;//no sync below: exit

    const float qrylen = GetQueryLength(qryndx);
    const float dbstrlen = GetDbStrLength(dbstrndx);
    const int dbstrglobndx = GetDbStrField<INTYPE,pps2DType>(dbstrndx);

    //NOTE: several blocks may write at the same time at the same location;
    //NOTE: safe as long as it's the same value;
    if(dbstrglobndx < ntotqstrs &&
       myhdmax(qrylen, dbstrlen) * covthreshold <= myhdmin(qrylen, dbstrlen))
        //reset convergence flag:
        wrkmemaux[mloc0 + dbstrndx] = 0;
}
