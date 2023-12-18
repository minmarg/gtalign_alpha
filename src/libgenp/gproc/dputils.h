/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __dputils_h__
#define __dputils_h__

#include "libutil/macros.h"
#include <stddef.h>

// -------------------------------------------------------------------------
// GetMaxBlockDiagonalElems: get maximum number of elements (oblique data 
// blocks) a block diagonal can contain, where blocks in a diagonal share a 
// point (top-right and/or bottom-left);
// It equals the minimum of [l/w] and [(x+w)/(b+w)], where x is dbstrlen, 
// l is the query length, w is blockwidth (the length of an oblique edge),
// b is the block length; [], ceil rounding;
// dbstrlen, #db reference structure positions in the chunk (x coord.);
// querylen, query length (y coord.);
// blockwidth, oblique block's width (e.g., 32);
// blocklen, block's length;
inline
unsigned int GetMaxBlockDiagonalElems(
    size_t dbstrlen, size_t querylen, size_t blockwidth, size_t blocklen)
{
    const size_t blocksalongquery = (querylen + blockwidth - 1) / blockwidth;
    const size_t bpw = blockwidth + blocklen;
    size_t maxblkdiagelems = ((dbstrlen + bpw) + bpw - 1) / bpw;
    return (unsigned int)PCMIN(blocksalongquery, maxblkdiagelems);
}

#endif//__dputils_h__
