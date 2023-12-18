/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __MpSecStr_h__
#define __MpSecStr_h__

#include "libutil/macros.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmymp/mpstages/transformbase.h"

// -------------------------------------------------------------------------
// class MpSecStr for calculating secondary structures
//
class MpSecStr {
    //indices for distances along str. positions: two, three, four residues apart
    enum {css2RESdst, css3RESdst, css4RESdst, cssTotal};

public:
    MpSecStr(
        char** querypmbeg, char** querypmend,
        char** bdbCpmbeg, char** bdbCpmend,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        uint qystr1len, uint dbstr1len,
        uint qystrnlen, uint dbstrnlen,
        uint dbxpad)
    :
        querypmbeg_(querypmbeg), querypmend_(querypmend),
        bdbCpmbeg_(bdbCpmbeg), bdbCpmend_(bdbCpmend),
        nqystrs_(nqystrs), ndbCstrs_(ndbCstrs),
        nqyposs_(nqyposs), ndbCposs_(ndbCposs),
        qystr1len_(qystr1len), dbstr1len_(dbstr1len),
        qystrnlen_(qystrnlen), dbstrnlen_(dbstrnlen),
        dbxpad_(dbxpad)
    {}

    void Run() {
        ssk_kernel_helper(querypmbeg_, querypmend_, nqystrs_, qystr1len_);
        ssk_kernel_helper(bdbCpmbeg_, bdbCpmend_, ndbCstrs_, dbstr1len_);
    }

private:
    void ssk_kernel_helper(
        char* const * const pmbeg_, char* const * const pmend_,
        int nstrs, int str1len);

    template<int S>
    float GetDistance(
        const float (*crds)[pmv2DNoElems], int pi);

    char AassignSecStr(
        const float (*dsts)[cssTotal], int pi);

private:
    char* const * const querypmbeg_, * const * const querypmend_;
    char* const * const bdbCpmbeg_, * const *const bdbCpmend_;
    const uint nqystrs_, ndbCstrs_;
    const uint nqyposs_, ndbCposs_;
    const uint qystr1len_, dbstr1len_;
    const uint qystrnlen_, dbstrnlen_;
    const uint dbxpad_;
};

// -------------------------------------------------------------------------
// INLINES ...
// -------------------------------------------------------------------------
// SSKGetDistance: calculate distances between two residues in the same 
// sequence;
// crds, cached coordinates;
// pi, destination position;
// S, step of several residues apart;
//
#if defined(OS_MS_WINDOWS)
#define OMPDECLARE_MpSecStr_GetDistance
#else 
#pragma omp declare simd linear(pi:1) uniform(crds) notinbranch
#endif
template<int S>
inline float MpSecStr::GetDistance(
    const float (*crds)[pmv2DNoElems], int pi)
{
    return sqrtf(distance2(
        crds[pi][pmv2DX], crds[pi][pmv2DY], crds[pi][pmv2DZ],
        crds[pi + S][pmv2DX], crds[pi + S][pmv2DY], crds[pi + S][pmv2DZ]
    ));
}

// -------------------------------------------------------------------------
// AassignSecStr: assign secondary structure for given position once 
// the required distances have been calculated;
// dsts, cache of calculated distances;
// pi, position to calculate secondary structure for;
// NOTE: pi corresponds to the structure position, hence
// NOTE: dsts position is nMRG0 greater;
// NOTE: pi assumed to be valid;
//
#if defined(OS_MS_WINDOWS)
#define OMPDECLARE_MpSecStr_AassignSecStr
#else 
#pragma omp declare simd linear(pi:1) uniform(dsts) notinbranch
#endif
inline char MpSecStr::AassignSecStr(
    const float (*dsts)[cssTotal], int pi)
{
    //NOTE: calculating for position 3 of 5 consecutive position
    float d13 = dsts[pi][css2RESdst];
    float d14 = dsts[pi][css3RESdst];
    float d15 = dsts[pi][css4RESdst];
    float d24 = dsts[pi+1][css2RESdst];
    float d25 = dsts[pi+1][css3RESdst];
    float d35 = dsts[pi+2][css2RESdst];

    float error = 2.1f;

    if(fabsf(d15-6.37f) < error && fabsf(d14-5.18f) < error &&
       fabsf(d25-5.18f) < error && fabsf(d13-5.45f) < error &&
       fabsf(d24-5.45f) < error && fabsf(d35-5.45f) < error)
        return pmvHELIX;//helix

    error = 1.42f;

    if(fabsf(d15-13.0f) < error && fabsf(d14-10.4f) < error &&
       fabsf(d25-10.4f) < error && fabsf(d13-6.1f) < error &&
       fabsf(d24-6.1f) < error && fabsf(d35-6.1f) < error)
        return pmvSTRND;//strand

    if(d15 < 8.0f) return pmvTURN;//turn

    return pmvLOOP;//loop/coil
}

#endif//__MpSecStr_h__
