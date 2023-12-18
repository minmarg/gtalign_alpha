/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __stagecnsts_h__
#define __stagecnsts_h__

#include <math.h>
#include "libutil/macros.h"
#include "libmycu/cucom/cudef.h"
#include "libmycu/cucom/cutemplates.h"

// -------------------------------------------------------------------------
// large distance
#define CP_LARGEDST 999999.9f
// large distance for comparison
#define CP_LARGEDST_cmp 399999.9f

// template constants for kernel FindD02ThresholdsCCM:
// calculate constants:
#define READCNST_CALC 0
// calculate constants in the next pass:
#define READCNST_CALC2 1

// increment for d02s when the condition of # aln. positions is not met
// #define D02s_CHCK_INC 0.5f
// increment for d02s when proceeding
#define D02s_PROC_INC 1.0f

// number of iterations for increasing distance threshold to ensure
// sufficient number of atom pairs (>=3) for the calculation of rotation
// matrices; two phases:
// #define N_ITER_DST_INCREASE 15
// #define N_ITER_DST_INCREASE2 0

#define SCNTS_COORD_MASK 99999.9f
#define SCNTS_COORD_MASK_cmp 99990.0f

// precision of convergence for all fields in kernel InitCopyCheckConvergence64Refined
#define RM_CONVEPSILON 1.e-5f

// -------------------------------------------------------------------------
// GetLnorm: return score-normalizing constant, which is the minimum of the
// query and reference lengths;
//
__DINLINE__
float GetLnorm(int qrylen, int dbstrlen)
{
//     int minlen = myhdmin(qrylen, dbstrlen);//NOTE: assumed >=3
    return myhdmin(qrylen, dbstrlen);//NOTE: assumed >=3
}


// -------------------------------------------------------------------------
// GetD8: calculate the distance threshold used for scoring fragments
// once the rotation matrix has been calculated;
//
__DINLINE__
float GetD8(int qrylen, int dbstrlen)
{
    float lnorm = GetLnorm(qrylen, dbstrlen);
    return 1.5f * d8powf(lnorm, 0.3f) + 3.5f;
}

// GetD82: calculate distance threshold d8 squared
//
__DINLINE__
float GetD82(int qrylen, int dbstrlen)
{
    float d8 = GetD8(qrylen, dbstrlen);
    return SQRD(d8);
}


// -------------------------------------------------------------------------
// GetD0: calculate the distance constant d0 based on the query and
// reference lengths
//
// value for final alignment refinement
__DINLINE__
float GetD0fin(int qrylen, int dbstrlen)
{
    float lnorm = GetLnorm(qrylen, dbstrlen);
    float d0 = 0.5f;

    if(lnorm > 21.f)
        //d0 = 1.24f * powf(lnorm - 15.f, 1.f/3.f) - 1.8f;
        d0 = 1.24f * cbrtf(lnorm - 15.f) - 1.8f;
    if(d0 < 0.5f) d0 = 0.5f;
    return d0;
}

// GetD02fin: d0 squared for final alignment refinement
__DINLINE__
float GetD02fin(int qrylen, int dbstrlen)
{
    float d0 = GetD0fin(qrylen, dbstrlen);
    return SQRD(d0);
}

__DINLINE__
float GetD0(int qrylen, int dbstrlen)
{
    float lnorm = GetLnorm(qrylen, dbstrlen);
    float d0 = 0.168f;

    if(lnorm > 19.f)
        //d0 = 1.24f * powf(lnorm - 15.f, 1.f/3.f) - 1.8f;
        d0 = 1.24f * cbrtf(lnorm - 15.f) - 1.8f;
    d0 += 0.8f;
    return d0;
}

// GetD02: calculate d0 squared
__DINLINE__
float GetD02(int qrylen, int dbstrlen)
{
    float d0 = GetD0(qrylen, dbstrlen);
    return SQRD(d0);
}

// GetD02_dpscan: calculate d0 squared tuned for the scan by DP
//
__DINLINE__
float GetD02_dpscan(int qrylen, int dbstrlen)
{
    float d0 = GetD0(qrylen, dbstrlen) + 1.5f;
    return SQRD(d0);
}

// -------------------------------------------------------------------------
// GetD0s: calculate the distance constant that is used for finding optimal
// set of positions on which the rotation matrix is calculated (for search)
//
__DINLINE__
float GetD0s(float d0)
{
    float d0s = d0;
    d0s = myhdmin(d0s, 8.f);
    d0s = myhdmax(d0s, 4.5f);
    return d0s;
}

// GetD02s: calculate squared GetD0s
//
__DINLINE__
float GetD02s(float d0)
{
    float d0s = GetD0s(d0);
    return SQRD(d0s);
}

#endif//__stagecnsts_h__
