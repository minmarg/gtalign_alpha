/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __linear_scoring2_cuh__
#define __linear_scoring2_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fragment.cuh"
#include "libmycu/custages/fields.cuh"


// -------------------------------------------------------------------------
// ProduceAlignmentUsingIndex2: find coordinates of nearest query 
// atoms at each reference position for following processing, using 
// index; the result follows from superpositions based on fragments;
// write the coordinates of neighbors for each position processed;
template<int SECSTRFILT>
__global__
void ProduceAlignmentUsingIndex2(
    const int stacksize,
    const bool WRTNDX,
    const int depth,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int qryfragfct,
    const int rfnfragfct,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux
);

// ProduceAlignmentUsingDynamicIndex2: same as ProduceAlignmentUsingIndex2 
// except that the alignments are produced using dynamically selected index
template<int SECSTRFILT>
__global__
void ProduceAlignmentUsingDynamicIndex2(
    const int stacksize,
    const bool WRTNDX,
    const int depth,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int qryfragfct,
    const int rfnfragfct,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux
);

// -------------------------------------------------------------------------
// PositionalCoordsFromIndexLinear2: find coordinates of nearest query 
// atoms at each reference position for following processing, using 
// index; the result follows from superpositions based on fragments;
template<int SECSTRFILT = 0>
__global__
void PositionalCoordsFromIndexLinear2(
    const int stacksize,
    const int depth,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int qryfragfct,
    const int rfnfragfct,
    int fragndx,
    const float* __restrict__ wrkmemtm,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpalnpossbuffer
);

// MakeAlignmentLinear2: make alignment from indices identified previously 
// by using the index tree; 
__global__
void MakeAlignmentLinear2(
    const bool complete,
    const int depth,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int qryfragfct,
    const int rfnfragfct,
    int fragndx,
    float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux
);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------

#endif//__linear_scoring2_cuh__
