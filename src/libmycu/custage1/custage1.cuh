/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __custage1_h__
#define __custage1_h__

#include <map>

#include "libutil/macros.h"
#include "libmycu/cucom/cugraphs.cuh"
#include "libmycu/custages/scoring.cuh"

// -------------------------------------------------------------------------
// constants defining the type of refinement: initial call, initial call on 
// aligned positions, and iterative refinement involving DP:
enum {
    stg1REFINE_INITIAL,
    stg1REFINE_INITIAL_DP,
    stg1REFINE_ITERATIVE_DP,
};

// -------------------------------------------------------------------------
// class stage1 for implementing structure comparison at stage 1
//
class stage1 {
public:
    static void preinitialize1(
        cudaStream_t streamproc,
        const bool condition4filter1,
        const uint maxnsteps,
        const uint minfraglen,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ tfmmem,
        float* __restrict__ alndatamem
    );

    static void run_stage1(
        std::map<CGKey,MyCuGraph>& stgraphs,
        cudaStream_t streamproc,
        const int maxndpiters,
        const uint maxnsteps,
        const uint minfraglen,
        const float scorethld,
        const float prescore,
        int stepinit,
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

    //{{ --- DP ---
    template<bool COMPLETEAPPROACH, bool ANCHORRGN, bool BANDED, bool GAP0>
    static void RunDP(
        cudaStream_t streamproc,
        const float gapcost,
        const uint maxnsteps,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpbotbuffer,
        char* __restrict__ btckdata,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux);

    template<bool ANCHORRGN, bool BANDED, bool GAP0>
    static void RunTDintensiveDP(
        cudaStream_t streamproc,
        const float gapcost,
        const uint maxnsteps,
        uint nqystrs, uint ndbCstrs,
        uint /*nqyposs*/, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpbotbuffer,
        char* __restrict__ btckdata,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux);

    template<bool ANCHORRGN, bool BANDED, bool GAP0>
    static void RunCompleteDP(
        cudaStream_t streamproc,
        const float gapcost,
        const uint maxnsteps,
        uint nqystrs, uint ndbCstrs,
        uint /*nqyposs*/, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint dbxpad,
        float* __restrict__ tmpdpbotbuffer,
        char* __restrict__ btckdata,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux);
    //}}

protected:
    static void stage1_findfrag2(
        cudaStream_t streamproc,
        int stepinit,
        const uint maxnsteps,
        const uint minfraglen,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint qystrnlen, uint dbstrnlen,
        uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ wrkmemtm
    );

    static void stage1_findfrag(
        cudaStream_t streamproc,
        int stepinit,
        const uint maxnsteps,
        const uint minfraglen,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint qystrnlen, uint dbstrnlen,
        uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ wrkmemtm
    );

    static void stage1_fragscore(
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint minfraglen,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint qystrnlen, uint dbstrnlen,
        uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemccd,
        float* __restrict__ wrkmemtm,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem
    );

    template<
        bool CONDITIONAL,
        int SECONDARYUPDATE = SECONDARYUPDATE_NOUPDATE>
    static void stage1_refinefrag(
        std::map<CGKey,MyCuGraph>& stgraphs,
        const int fragbydp,
        const int nmaxconvit,
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint minfraglen,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint qystrnlen, uint dbstrnlen,
        uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpalnpossbuffer,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemccd,
        float* __restrict__ wrkmemtm,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem
    );

public:
    template<
        bool GAP0,
        bool PRESCREEN = false,
        bool WRKMEMTM1 = false>
    static void stage1_dprefine(
        std::map<CGKey,MyCuGraph>& stgraphs,
        cudaStream_t streamproc,
        const int maxndpiters,
        const float prescore,
        const uint maxnsteps,
        const uint minfraglen,
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
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemccd,
        float* __restrict__ wrkmemtm,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem
    );

private:
    static void stage1_refinefrag_helper(
        std::map<CGKey,MyCuGraph>& stgraphs,
        const int fragbydp,
        const int nmaxconvit,
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint minfraglen,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint qystrnlen, uint dbstrnlen,
        uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpalnpossbuffer,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemccd,
        float* __restrict__ wrkmemtm,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem
    );

    template<bool CONDITIONAL, int SECONDARYUPDATE>
    static void stage1_refinefrag_helper2(
        const int fragbydp,
        const int nmaxconvit,
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint minfraglen,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint qystrnlen, uint dbstrnlen,
        uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpalnpossbuffer,
        float* __restrict__ wrkmemtm,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ tfmmem
    );

    static void stage1_findfrag_subiter1(
        cudaStream_t streamproc,
        int n1, int stepinit,
        const uint maxnsteps,
        const uint minfraglen,
        uint nqystrs, uint ndbCstrs,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem,
        //
        const dim3& nblcks_init, const dim3& nthrds_init,
        const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
        const dim3& nblcks_copyto, const dim3& nthrds_copyto,
        const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
        const dim3& nblcks_tfm, const dim3& nthrds_tfm
    );

    static void stage1_findfrag_subiter2(
        cudaStream_t streamproc,
        int n1, int stepinit,
        const uint maxnsteps,
        const uint minfraglen,
        uint nqystrs, uint ndbCstrs, uint ndbCposs,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem,
        //
        const dim3& nblcks_init, const dim3& nthrds_init,
        const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
        const dim3& nblcks_findd2, const dim3& nthrds_findd2,
        const dim3& nblcks_scinit, const dim3& nthrds_scinit,
        const dim3& nblcks_scores, const dim3& nthrds_scores,
        const dim3& nblcks_copyto, const dim3& nthrds_copyto,
        const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
        const dim3& nblcks_tfm, const dim3& nthrds_tfm
    );

    static void stage1_findfrag_subiter3(
        cudaStream_t streamproc,
        int n1, int stepinit,
        const uint maxnsteps,
        const uint minfraglen,
        uint nqystrs, uint ndbCstrs, uint ndbCposs,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ tfmmem,
        //
        const dim3& nblcks_init, const dim3& nthrds_init,
        const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
        const dim3& nblcks_findd2, const dim3& nthrds_findd2,
        const dim3& nblcks_scinit, const dim3& nthrds_scinit,
        const dim3& nblcks_scores, const dim3& nthrds_scores,
        const dim3& nblcks_copyto, const dim3& nthrds_copyto,
        const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
        const dim3& nblcks_tfm, const dim3& nthrds_tfm
    );

    //----------------------------------------------------------------------

    static void stage1_refinefrag_subiter1(
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint minfraglen,
        const int sfragstep,
        uint nqystrs, uint ndbCstrs, uint ndbCposs,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmemccd,
        //
        const dim3& nblcks_init, const dim3& nthrds_init,
        const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
        const dim3& nblcks_conv, const dim3& nthrds_conv
    );

    static void stage1_refinefrag_dp_subiter1(
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint minfraglen,
        const int sfragstep,
        uint nqystrs, uint ndbCstrs, uint ndbCposs,
        uint dbxpad,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmemccd,
        float* __restrict__ tmpdpalnpossbuffer,
        //
        const dim3& nblcks_init, const dim3& nthrds_init,
        const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
        const dim3& nblcks_conv, const dim3& nthrds_conv
    );

    static void stage1_refinefrag_subiter2(
        bool lastsubiter,
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint minfraglen,
        const int sfragstep,
        uint nqystrs, uint ndbCstrs, uint ndbCposs,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ wrkmemccd,
        float* __restrict__ wrkmemtm,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ tfmmem,
        //
        int cit,
        const dim3& nblcks_init, const dim3& nthrds_init,
        const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
        const dim3& nblcks_findd2, const dim3& nthrds_findd2,
        const dim3& nblcks_scinit, const dim3& nthrds_scinit,
        const dim3& nblcks_scores, const dim3& nthrds_scores,
        const dim3& nblcks_savetm, const dim3& nthrds_savetm,
        const dim3& nblcks_saveccd, const dim3& nthrds_saveccd,
        const dim3& nblcks_conv, const dim3& nthrds_conv,
        const dim3& nblcks_copyto, const dim3& nthrds_copyto,
        const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
        const dim3& nblcks_tfm, const dim3& nthrds_tfm
    );

    static void stage1_refinefrag_dp_subiter2(
        int fragbydp,
        bool lastsubiter,
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint minfraglen,
        const int sfragstep,
        uint nqystrs, uint ndbCstrs, uint ndbCposs, uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpalnpossbuffer,
        float* __restrict__ wrkmem,
        float* __restrict__ wrkmemaux,
        float* __restrict__ wrkmem2,
        float* __restrict__ wrkmemccd,
        float* __restrict__ wrkmemtm,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ tfmmem,
        //
        int cit,
        const dim3& nblcks_init, const dim3& nthrds_init,
        const dim3& nblcks_ccmtx, const dim3& nthrds_ccmtx,
        const dim3& nblcks_findd2, const dim3& nthrds_findd2,
        const dim3& nblcks_scinit, const dim3& nthrds_scinit,
        const dim3& nblcks_scores, const dim3& nthrds_scores,
        const dim3& nblcks_savetm, const dim3& nthrds_savetm,
        const dim3& nblcks_saveccd, const dim3& nthrds_saveccd,
        const dim3& nblcks_conv, const dim3& nthrds_conv,
        const dim3& nblcks_copyto, const dim3& nthrds_copyto,
        const dim3& nblcks_copyfrom, const dim3& nthrds_copyfrom,
        const dim3& nblcks_tfm, const dim3& nthrds_tfm
    );
};

// -------------------------------------------------------------------------

#endif//__custage1_h__
