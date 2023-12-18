/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __local_similarity02_h__
#define __local_similarity02_h__

// CalcLocalSimilarity2_frg2: calculate provisional local similarity during 
// extensive fragment-based search of optimal superpositions;
__global__ void CalcLocalSimilarity2_frg2(
    const float thrsimilarityperc,
    const int depth,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    const char* __restrict__ dpscoremtx,
    float* __restrict__ wrkmemaux
);

#endif//__local_similarity02_h__
