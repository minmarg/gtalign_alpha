/***************************************************************************
 *   Copyright (C) 2021-2026 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __cusecstr_cuh__
#define __cusecstr_cuh__

#include <map>

#include "libutil/macros.h"
#include "libmycu/cucom/cugraphs.cuh"

// -------------------------------------------------------------------------
// class cusecstr for calculating secondary structures
//
class cusecstr {
public:
    static void calc_secstr(
        cudaStream_t streamproc,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint qystrnlen, uint dbstrnlen,
        uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers
    );

protected:
    static void calc_secstr_protein(
        cudaStream_t streamproc,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint qystrnlen, uint dbstrnlen,
        uint dbxpad
    );

    template<int STRUCTS>
    static void calc_secstr_na(
        cudaStream_t streamproc,
        uint nstrs, uint nposs, uint str1len,
        uint dbxpad,
        float* __restrict__ tmpdpdiagbuffers
    );
};

// -------------------------------------------------------------------------

#endif//__cusecstr_cuh__
