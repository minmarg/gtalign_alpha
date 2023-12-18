/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __linearscoringbase_h__
#define __linearscoringbase_h__

// constants for searching in the index:
// stack structure: positional index and dimension (2 bits) combined, and 
// difference in dimension:
enum {stkNdx_Dim_, stkDiff_, nStks_};
// forward combination of index and dimension:
#define NNSTK_COMBINE_NDX_DIM(ndx, dim) ((dim << 16)|(ndx & 0xffff))
// extract index and dimension from combination:
#define NNSTK_GET_NDX_FROM_COMB(comb) (comb & 0xffff)
#define NNSTK_GET_DIM_FROM_COMB(comb) (comb >> 16)

#endif//__linearscoringbase_h__
