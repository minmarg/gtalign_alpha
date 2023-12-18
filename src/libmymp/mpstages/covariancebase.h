/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __covariancebase_h__
#define __covariancebase_h__

#include "libutil/cnsts.h"
#include "libutil/CLOptions.h"
#include "libmycu/cucom/cudef.h"
#include "libmycu/cucom/cutemplates.h"

// -------------------------------------------------------------------------
// GetFragStepSize_frg_deep: get the the step size which corresponds to 
// the depth of superposition exploration based on fragments dependent upon 
// structure lengths; version of more extensive parallelization;
// NOTE: deep depth;
// length, structure length;
//
__HDINLINE__
int GetFragStepSize_frg_deep(int length)
{
    if(length > 150) return 15;
    int leno3 = myhdmax(1, (int)((float)length * oneTHIRDf));
    if(leno3 < 15) return leno3;
    return 15;
}
// NOTE: high depth;
__HDINLINE__
int GetFragStepSize_frg_high(int length)
{
    if(length > 150) return 23;
    int leno3 = myhdmax(1, (int)((float)length * oneTHIRDf));
    if(leno3 < 15) return leno3;
    return 15;
}
// NOTE: medium depth;
__HDINLINE__
int GetFragStepSize_frg_medium(int length)
{
    if(length > 250) return 45;
    if(length > 200) return 35;
    if(length > 150) return 25;
    int leno3 = myhdmax(1, (int)((float)length * oneTHIRDf));
    if(leno3 < 15) return leno3;
    return 15;
}
// NOTE: shallow depth;
__HDINLINE__
constexpr int GetFragStepSize_frg_shallow_factor() {return 2;}
//
__HDINLINE__
int GetFragStepSize_frg_shallow(int length)
{
    return
        GetFragStepSize_frg_medium(length) *
        GetFragStepSize_frg_shallow_factor();
}

// -------------------------------------------------------------------------
// GetQryRfnStepsize2: return the stepsize for query and reference when 
// calculating transformation matrices from variable-length fragments;
//
__HDINLINE__
void GetQryRfnStepsize2(
    const int depth,
    int qrylen, int dbstrlen,
    int* qrystepsz, int* rfnstepsz)
{
    *qrystepsz = GetFragStepSize_frg_shallow(qrylen);
    *rfnstepsz = GetFragStepSize_frg_shallow(dbstrlen);
    if(depth == CLOptions::csdDeep) {
        *qrystepsz = GetFragStepSize_frg_deep(qrylen);
        *rfnstepsz = GetFragStepSize_frg_deep(dbstrlen);
    } else if(depth == CLOptions::csdHigh) {
        *qrystepsz = GetFragStepSize_frg_high(qrylen);
        *rfnstepsz = GetFragStepSize_frg_high(dbstrlen);
    } else if(depth == CLOptions::csdMedium) {
        *qrystepsz = GetFragStepSize_frg_medium(qrylen);
        *rfnstepsz = GetFragStepSize_frg_medium(dbstrlen);
    }
}

// -------------------------------------------------------------------------
// GetNAlnPoss_frg: return the maximum number of alignment positions 
// (fragment size) given the lengths and start positions of the query and 
// reference structures for length-dependent fragments;
// version of more extensive parallelization;
// arg3 is fragndx, fragment index determining the fragment size dependent 
// upon lengths;
__HDINLINE__
int GetNAlnPoss_frg(
    int qrylen, int dbstrlen,
    int /*qrypos*/, int /*rfnpos*/,
    int /*arg1*/, int /*arg2*/, int arg3)
{
    int minlen = myhdmin(qrylen, dbstrlen);

    if(arg3 == 0) {
        int leno3 = myhdmax(1, (int)((float)minlen * oneTHIRDf));
        return (leno3 < 20)? leno3: 20;
    }

    // arg3 == 1
    int leno2 = (minlen >> 1);
    return (leno2 < 100)? leno2: 100;
}

// -------------------------------------------------------------------------

#endif//__covariancebase_h__
