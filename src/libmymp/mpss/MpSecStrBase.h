/***************************************************************************
 *   Copyright (C) 2021-2026 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __MpSecStrBase_h__
#define __MpSecStrBase_h__

#include "libgenp/gdats/PM2DVectorFields.h"

// =========================================================================
enum{
        lTMPBUFFNDX_POSIT = 0,//pairig position with least deviation
        lTMPBUFFNDX_DEVIA = 1,//minimum deviation
        lTMPBUFFNDX_MUTEX = 2//positional mutex
};
// -------------------------------------------------------------------------
// SSKNAGetPairingCondition: get the condition for pairing bases;
// rsdy, rsdx, two nucleic acid bases;
//
#if defined(__CUDA_ARCH__)
__host__ __device__ __forceinline__
#else
inline
#endif
bool SSKNAGetPairingCondition(const char rsdy, const char rsdx)
{
    return
        ((rsdy == 'T' || rsdy == 'U') && (rsdx == 'A')) ||
        ((rsdy == 'A') && (rsdx == 'T' || rsdx == 'U')) ||
        ((rsdy == 'G') && (rsdx == 'C' || rsdx == 'U')) ||
        ((rsdy == 'C' || rsdy == 'U') && (rsdx == 'G'));
}

// -------------------------------------------------------------------------
// SSKNAGetDstDeviation: get distance deviation from the statistical 
// average of distances between paired nucleic acid atoms of given type;
// atype, nucleic acid atom type;
// dst, observed distance between atoms of given type;
//
#if defined(__CUDA_ARCH__)
__host__ __device__ __forceinline__
#else
inline
#endif
float SSKNAGetDstDeviation(const int atype, float dst)
{
    float dev;
    const float devl = 9999.9f;
    if(atype == gtnaatC3p) {dev = fabsf(dst - gtnaatC3p_LUB_AVG); return (gtnaatC3p_LUB_DLT < dev)? devl: dev;}
    if(atype == gtnaatC4p) {dev = fabsf(dst - gtnaatC4p_LUB_AVG); return (gtnaatC4p_LUB_DLT < dev)? devl: dev;}
    if(atype == gtnaatC5p) {dev = fabsf(dst - gtnaatC5p_LUB_AVG); return (gtnaatC5p_LUB_DLT < dev)? devl: dev;}
    if(atype == gtnaatO3p) {dev = fabsf(dst - gtnaatO3p_LUB_AVG); return (gtnaatO3p_LUB_DLT < dev)? devl: dev;}
    if(atype == gtnaatO5p) {dev = fabsf(dst - gtnaatO5p_LUB_AVG); return (gtnaatO5p_LUB_DLT < dev)? devl: dev;}
    if(atype == gtnaatP) {dev = fabsf(dst - gtnaatP_LUB_AVG); return (gtnaatP_LUB_DLT < dev)? devl: dev;}
    return devl;
}

#endif//__MpSecStrBase_h__
