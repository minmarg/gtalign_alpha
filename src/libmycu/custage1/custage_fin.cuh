/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __custage_fin_h__
#define __custage_fin_h__

#include "libutil/macros.h"
#include "libmycu/custage1/custage1.cuh"

// -------------------------------------------------------------------------
// class stagefin for implementing final alignment refinement for output
//
class stagefin: public stage1 {
public:
    static void run_stagefin(
        cudaStream_t streamproc,
        const float d2equiv,
        const float scorethld,
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
        float* __restrict__ alndatamem,
        float* __restrict__ tfmmem,
        char* __restrict__ alnsmem,
        uint* __restrict__ globvarsbuf
    );

protected:
    static void stagefin_align(
        const bool constrainedbtck,
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint nqystrs, const uint ndbCstrs,
        const uint /*nqyposs*/, const uint ndbCposs,
        const uint qystr1len, const uint dbstr1len,
        const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
        const uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpbotbuffer,
        float* __restrict__ tmpdpalnpossbuffer,
        uint* __restrict__ /*maxscoordsbuf*/,
        char* __restrict__ btckdata,
        float* __restrict__ wrkmemaux,
        float* __restrict__ tfmmem
    );

    static void stagefin_refine(
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint minfraglen,
        const uint nqystrs, const uint ndbCstrs,
        const uint /* nqyposs */, const uint ndbCposs,
        const uint qystr1len, const uint dbstr1len,
        const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
        const uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpalnpossbuffer,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ tfmmem
    );

    static void stagefin_produce_output_alignment(
        const float d2equiv,
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint nqystrs, const uint ndbCstrs,
        const uint /* nqyposs */, const uint ndbCposs,
        const uint /* qystr1len */, const uint /* dbstr1len */,
        const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
        const uint dbxpad,
        float* __restrict__ tmpdpalnpossbuffer,
        float* __restrict__ wrkmemaux,
        float* __restrict__ alndatamem,
        char* __restrict__ alnsmem
    );

    static void stagefin_produce_output_scores(
        cudaStream_t streamproc,
        const uint maxnsteps,
        const uint minfraglen,
        const uint nqystrs, const uint ndbCstrs,
        const uint /* nqyposs */, const uint ndbCposs,
        const uint qystr1len, const uint dbstr1len,
        const uint /*qystrnlen*/, const uint /*dbstrnlen*/,
        const uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers,
        float* __restrict__ tmpdpalnpossbuffer,
        float* __restrict__ wrkmemtmibest,
        float* __restrict__ wrkmemaux,
        float* __restrict__ alndatamem,
        float* __restrict__ tfmmem
    );

    static void stagefin_adjust_tfms(
        cudaStream_t streamproc,
        const uint nqystrs, const uint ndbCstrs,
        const uint /*nqyposs*/, const uint /*ndbCposs*/,
        float* __restrict__ wrkmemaux,
        float* __restrict__ tfmmem
    );

private:
};

// -------------------------------------------------------------------------

#endif//__custage_fin_h__
