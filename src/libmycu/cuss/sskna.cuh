/***************************************************************************
 *   Copyright (C) 2021-2025 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __sskna_cuh__
#define __sskna_cuh__

#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/custages/transform.cuh"
#include "libmycu/custages/fields.cuh"
#include "sskbase.cuh"

// =========================================================================
// CalcNASecStrs: calculate secondary structures for all query OR reference 
// nucleic acid structures in the chunk;
template<int STRUCTS>
__global__ void CalcSecStrs_NASS(
    const uint ndbCposs,
    float* __restrict__ tmpdpdiagbuffers
);

// Initialize_NASS: initialize temporary memory buffer for calculating
// nucleic acid secondary structures
__global__ void Initialize_NASS(
    const uint ndbCposs,
    float* __restrict__ tmpdpdiagbuffers
);

// CalcDistances_NASS: calculate relevant pairwise distances between
// residues for all query OR reference nucleic acid structures in the chunk;
template<int STRUCTS>
__global__ void CalcDistances_NASS(
    const int atomtype,
    const uint ndbCposs,
    float* __restrict__ tmpdpdiagbuffers
);

// CalcDistances_NASS_CC7: calculate relevant pairwise distances between
// residues for all query OR reference nucleic acid structures in the chunk;
// NOTE: version for compute capability starting with No. 7;
template<int STRUCTS>
__global__ void CalcDistances_NASS_CC7(
    const int atomtype,
    const uint ndbCposs,
    float* __restrict__ tmpdpdiagbuffers
);

// =========================================================================
// =========================================================================

#endif//__sskna_cuh__
