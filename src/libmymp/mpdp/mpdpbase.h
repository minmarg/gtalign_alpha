/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __mpdpbase_h__
#define __mpdpbase_h__

#include "libmycu/cucom/cudef.h"

// =========================================================================
// scoring by secondary structure and sequence similarity:
// default initial score value: 2^20; this value is also used to 
// determine match state; calculated so that it is two times 
// greater than max score * max length:
#define DPSSDEFINITSCOREVAL (-1048576.0f)
// value used in comparison statements: 2^20/2
#define DPSSDEFINITSCOREVAL_cmp (-524288.0f)

// =========================================================================
// default coordinate value for query
// NOTE: use integer values: a macro will convert them to FP numbers
#define CUDP_DEFCOORD_QRY 99999
#define CUDP_DEFCOORD_QRY_cmp 99990.0f
// default coordinate value for reference
#define CUDP_DEFCOORD_RFN -99999
#define CUDP_DEFCOORD_RFN_cmp -99990.0f

// indicator values for calculating constant d02:
// D02IND_SEARCH, value for general search;
// D02IND_DPSCAN, value for scan by DP;
#define D02IND_SEARCH 0
#define D02IND_DPSCAN 1

// =========================================================================
// -------------------------------------------------------------------------
// GetMaqxNoIterations: get the maximum number of iterations for the 
// block to perform;
// TODO: constrain ilim even more with respect to areas of non-exploration:
// lower and upper areas implied the anchor region and alignment bandwidth;
// 
__DINLINE__
int GetMaqxNoIterations(
    int x, int y,
    int qrylen, int dbstrlen,
    int blklen)
{
    x = dbstrlen - x;
    if(x <= 0) return 0;
    y -= qrylen-1;//NOTE
    if(0 < y) x -= y;
    if(x <= 0) return 0;
    if(blklen <= x) return blklen;
    return x;
}

// =========================================================================

#endif//__mpdpbase_h__
