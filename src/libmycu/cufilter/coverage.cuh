/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __coverage_cuh__
#define __coverage_cuh__

// CheckMaxCoverage: calculate maximum coverage between the queries and 
// reference structures and set the skip flag (convergence) if it is 
// below the threshold; the function actually resets convergence for the 
// pairs within the threshold;
__global__ void CheckMaxCoverage(
    const float covthreshold,
    const int ntotqstrs,
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux
);

#endif//__coverage_cuh__
