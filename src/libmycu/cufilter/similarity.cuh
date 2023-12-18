/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __similarity_cuh__
#define __similarity_cuh__

// VerifyAlignmentScore: calculate local sequence alignment score between the
// queries and reference structures and set the flag of low score if it is 
// below the threshold;
__global__ void VerifyAlignmentScore(
    const float seqsimthrscore,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux
);

#endif//__similarity_cuh__
