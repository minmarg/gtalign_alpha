/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __reformatter_cuh__
#define __reformatter_cuh__

// MakeDbCandidateList: make list of reference structure (database)
// candidates proceeding to more detailed stages of superposition search and
// refinement;
__global__ void MakeDbCandidateList(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint maxnsteps,
    const float* __restrict__ wrkmemaux,
    uint* __restrict__ globvarsbuf
);

// ReformatStructureDataPartStore: reformat a reference database chunk to
// include candidates proceeding to stages of more detailed superposition
// search and refinement; this part corresponds to storing data to
// secondary (temporary) location first;
__global__ void ReformatStructureDataPartStore(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint maxndbCposs,
    const uint maxnsteps,
    // const uint ndbCstrs2,
    // const uint ndbCposs2,
    // const uint dbstr1len2,
    const uint* __restrict__ globvarsbuf,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ tfmmem,
    float* __restrict__ tmpdpdiagbuffers
);

// ReformatStructureDataPartLoad: reformat a reference database chunk to
// include candidates proceeding to stages of more detailed superposition
// search and refinement; this part corresponds to data load from secondary
// (temporary) location;
__global__ void ReformatStructureDataPartLoad(
    const uint nqystrs,
    const uint maxndbCposs,
    const uint maxnsteps,
    const uint ndbCstrs2,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tfmmem
);

#endif//__reformatter_cuh__
