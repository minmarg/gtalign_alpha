/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __custage2_h__
#define __custage2_h__

#include <map>

#include "libutil/macros.h"
#include "libmycu/cucom/cugraphs.cuh"
#include "libmycu/custage1/custage1.cuh"

// -------------------------------------------------------------------------
// class stage2 for implementing structure comparison at stage 2
//
class stage2: public stage1 {
public:
    template<bool GAP0, bool USESS, int D02IND>
    static void run_stage2(
        std::map<CGKey,MyCuGraph>& stgraphs,
        cudaStream_t streamproc,
        const bool check_for_low_scores,
        const float scorethld,
        const float prescore,
        const int maxndpiters,
        const uint maxnsteps,
        const uint minfraglen,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint qystrnlen, uint dbstrnlen,
        uint dbxpad,
        float* __restrict__ scores, 
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpbotbuffer,
        float* __restrict__ tmpdpalnpossbuffer,
        uint* __restrict__ maxscoordsbuf,
        char* __restrict__ btckdata,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemccd,
        float* __restrict__ wrkmemtm,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem,
        uint* __restrict__ globvarsbuf
    );

protected:
    template<bool GAP0, bool USESS, int D02IND>
    static void stage2_dpss_align(
        cudaStream_t streamproc,
        const uint maxnsteps,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint qystrnlen, uint dbstrnlen,
        uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpbotbuffer,
        float* __restrict__ tmpdpalnpossbuffer,
        uint* __restrict__ maxscoordsbuf,
        char* __restrict__ btckdata,
        float* __restrict__ wrkmemaux,
        float* __restrict__ tfmmem
    );

private:
};

// -------------------------------------------------------------------------

#endif//__custage2_h__
