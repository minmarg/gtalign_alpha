/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __covariance_h__
#define __covariance_h__

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libutil/CLOptions.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmymp/mpstages/covariancebase.h"
#include "libmycu/cucom/cutemplates.h"
#include "libmycu/custages/fields.cuh"


// do not check convergence of finding optimal rotation matrix and data processing:
#define CHCKCONV_NOCHECK 0
// check convergence:
#define CHCKCONV_CHECK 1

// InitCCData0: device code for initializing cross-covariance data 
// between the query and reference structures when unconditionally 
// calculating superpositions for fragments (initial phase);
__global__ void InitCCData0(
    const uint ndbCstrs,
    const uint maxnsteps,
    int n1, int step,
    float* __restrict__ wrkmem
);

#if 0
// InitCCData0_var: device code for initializing cross-covariance data 
// between the query and reference structures when unconditionally 
// calculating superpositions for VARIABLE-LENGTH fragments;
template<bool STEPx5>
__global__ void InitCCData0_var(
    const uint ndbCstrs,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    float* __restrict__ wrkmem
);

// InitCCData0_frg: device code for initializing cross-covariance data 
// between the query and reference structures when unconditionally 
// calculating superpositions for VARIABLE-LENGTH fragments;
// version of more extensive parallelization;
__global__ void InitCCData0_frg(
    const bool stepx5,
    const uint ndbCstrs,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    float* __restrict__ wrkmem
);
// InitCCData0_frgbest: initialize cross-covariance data for best 
// fragment identified before;
__global__ void InitCCData0_frgbest(
    const uint ndbCstrs,
    const uint maxnsteps,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem
);
#endif
// InitCCData0_frg2: device code for initializing cross-covariance data 
// between the query and reference structures when unconditionally 
// calculating superpositions for variable-length fragments;
// version of more extensive parallelization;
__global__ void InitCCData0_frg2(
    const int depth,
    const uint ndbCstrs,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    float* __restrict__ wrkmem
);


// memory initialization
template<int CHCKCONV>
__global__ void InitCCData(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux
);


// CalcCCMatrices: calculate cross-covariance matrix between the query and 
// reference structures;
__global__ void CalcCCMatrices64(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint maxnsteps,
    int n1, int step,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem
);

#if 0
// CalcCCMatrices64_var: calculate cross-covariance matrix between the 
// query and reference structures over a fragment whose length depends on 
// structure lengths;
template<bool STEPx5>
__global__ void CalcCCMatrices64_var(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    float* __restrict__ wrkmem
);

// CalcCCMatrices64_frg: calculate cross-covariance matrix between the 
// query and reference structures over a fragment whose length depends on 
// structure lengths;
// version of more extensive parallelization;
__global__ void CalcCCMatrices64_frg(
    const bool stepx5,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    float* __restrict__ wrkmem
);
// CalcCCMatrices64_frgbest: calculate cross-covariance matrix between the 
// query and reference structures for best-identified fragment;
__global__ void CalcCCMatrices64_frgbest(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint maxnsteps,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem
);
#endif
// CalcCCMatrices64_frg2: calculate cross-covariance matrix between the 
// query and reference structures over a fragment whose length depends on 
// structure lengths; version of more extensive parallelization;
__global__ void CalcCCMatrices64_frg2(
    const int depth,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem
);


#define READNPOS_NOREAD 0
#define READNPOS_READ 1
// CopyCCDataToWrkMem2: copy cross-covariance matrix between the query and 
// reference structures to section 2 to enable efficient Kabsch 
// algorithm application for multiple structures simultaneously;
template<int READNPOS>
__global__ void CopyCCDataToWrkMem2(
    const uint ndbCstrs,
    const uint maxnsteps,
    int n1, int step,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmem2
);

#if 0
// CopyCCDataToWrkMem2_var: this version differs from the previous one in 
// determining the out-of-bounds condition when the fragment length and 
// position depends on structure lengths;
template<int READNPOS, bool STEPx5>
__global__ void CopyCCDataToWrkMem2_var(
    const uint ndbCstrs,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmem2
);

// CopyCCDataToWrkMem2_frg: same as CopyCCDataToWrkMem2_var with more 
// extensive parallelization;
__global__ void CopyCCDataToWrkMem2_frg(
    const bool stepx5,
    const uint ndbCstrs,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmem2
);
// CopyCCDataToWrkMem2_frgbest: copy cross-covariance matrix for 
// best-identified fragment to section 2 to enable efficient Kabsch 
// algorithm application for multiple structures simultaneously;
// 
__global__ void CopyCCDataToWrkMem2_frgbest(
    const uint ndbCstrs,
    const uint maxnsteps,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmem2
);
#endif
// CopyCCDataToWrkMem2_frg2:
// CopyCCDataToWrkMem2_frg2: includes determining the out-of-bounds 
// condition when the fragment length and position depends on structure 
// lengths; more extensive parallelization;
__global__ void CopyCCDataToWrkMem2_frg2(
    const int depth,
    const uint ndbCstrs,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmem2
);


// formatting transformation matrices in original format
__global__ void CopyTfmMtsFromWrkMem2(
    const uint ndbCstrs,
    const uint maxnsteps,
    const float* __restrict__ wrkmem2,
    float* __restrict__ tfmmem
);





// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// GetQryRfnPos: return the query and reference positions for the initial 
// stage of calculating transformation matrices from fragments at least 
// half of the minimum length of two structures being compared;
// arg1 is n1, starting position that determines positions in query and 
// reference;
// arg2 is step, step size in positions used to traverse query and reference 
// ungapped alignments;
__device__ __forceinline__
void GetQryRfnPos(
    const int /*depth*/,
    int& qrypos, int& rfnpos,
    int /*qrylen*/, int /*dbstrlen*/,
    uint sfragfct, int arg1, int arg2, int /*arg3*/)
{
    arg1/*n1*/ += sfragfct * arg2/*step*/;
    qrypos = myhdmax(0,arg1/*n1*/);
    rfnpos = myhdmax(-arg1/*n1*/,0);
}

// -------------------------------------------------------------------------
// PositionsOutofBounds: return true if the positions given for query and 
// reference imply out-of-bounds condition
__device__ __forceinline__
bool PositionsOutofBounds(
    int& qrylen, int& dbstrlen,
    int qrypos, int rfnpos,
    int /*arg1*/, int /*arg2*/, int /*arg3*/)
{
    int minlen = myhdmin(qrylen, dbstrlen);
    minlen = myhdmax(minlen >> 1, 5);

    if(qrylen - qrypos < minlen ||
       dbstrlen - rfnpos < minlen)
        return true;

    return false;
}

// -------------------------------------------------------------------------
// GetNAlnPoss: return the maximum number of alignment positions given the 
// lengths and start positions of the query and reference structures;
__device__ __forceinline__
int GetNAlnPoss(
    int qrylen, int dbstrlen,
    int qrypos, int rfnpos,
    int /*arg1*/, int /*arg2*/, int /*arg3*/)
{
    return myhdmin(qrylen-qrypos, dbstrlen-rfnpos);
}



#if 0
// -------------------------------------------------------------------------
// GetFragStepSize_var: get the the step size which used to move fragments 
// dependent upon structure lengths;
// length, structure length;
__device__ __forceinline__
int GetFragStepSize_var(int length)
{
    if(length > 250) return 45;
    if(length > 200) return 35;
    if(length > 150) return 25;
    int leno3 = myhdmax(1, (int)((float)length * oneTHIRDf));
    if(leno3 < 15) return leno3;
    return 15;
}

// -------------------------------------------------------------------------
// GetQryRfnPos_var: return the query and reference positions when 
// calculating transformation matrices from variable-length fragments of 
// two structures being compared;
// arg1 is qryfragfct, fragment factor for query (to be multiplied by step 
// dependent upon lengths);
// arg2 is rfnfragfct, fragment factor for reference (to be multiplied by step 
// dependent upon lengths after summation);
__device__ __forceinline__
void GetQryRfnPos_var(
    int& qrypos, int& rfnpos,
    int qrylen, int dbstrlen,
    uint sfragfct, int arg1, int arg2, int /*arg3*/)
{
    qrypos = arg1 * GetFragStepSize_var(qrylen);
    rfnpos = (arg2 + sfragfct) * GetFragStepSize_var(dbstrlen);
}

// GetQryRfnPos_var5: positions obtained by GetQryRfnPos_var multiplied by 5;
__device__ __forceinline__
void GetQryRfnPos_var5(
    int& qrypos, int& rfnpos,
    int qrylen, int dbstrlen,
    uint sfragfct, int arg1, int arg2, int arg3)
{
    GetQryRfnPos_var(qrypos, rfnpos,  qrylen, dbstrlen, sfragfct, arg1, arg2, arg3);
    qrypos *= 5;
    rfnpos *= 5;
}

// GetQryRfnPos_varT: template version of GetQryRfnPos_var:
template<bool STEPx5>
__device__ __forceinline__
void GetQryRfnPos_varT(
    int& qrypos, int& rfnpos,
    int qrylen, int dbstrlen,
    uint sfragfct, int arg1, int arg2, int arg3)
{
    if(STEPx5) GetQryRfnPos_var5(qrypos, rfnpos, qrylen, dbstrlen, sfragfct, arg1, arg2, arg3);
    else GetQryRfnPos_var(qrypos, rfnpos, qrylen, dbstrlen, sfragfct, arg1, arg2, arg3);
}

// GetQryRfnPos_varP: parameterized version of GetQryRfnPos_var:
__device__ __forceinline__
void GetQryRfnPos_varP(
    bool STEPx5,
    int& qrypos, int& rfnpos,
    int qrylen, int dbstrlen,
    uint sfragfct, int arg1, int arg2, int arg3)
{
    if(STEPx5) GetQryRfnPos_var5(qrypos, rfnpos, qrylen, dbstrlen, sfragfct, arg1, arg2, arg3);
    else GetQryRfnPos_var(qrypos, rfnpos, qrylen, dbstrlen, sfragfct, arg1, arg2, arg3);
}


// -------------------------------------------------------------------------
// GetNAlnPoss_var: return the maximum number of alignment positions 
// (fragment size) given the lengths and start positions of the query and 
// reference structures for length-dependent fragments;
// arg3 is fragndx, fragment index determining the fragment size dependent 
// upon lengths;
__device__ __forceinline__
int GetNAlnPoss_var(
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
// PositionsOutofBounds_var: return true if the positions given for
// query and reference imply the out-of-bounds condition when the alignment
// length is dependent upon structure lengths;
// arg2 is fragndx, fragment index determining the fragment size dependent 
// upon lengths;
__device__ __forceinline__
bool PositionsOutofBounds_var(
    int& qrylen, int& dbstrlen,
    int qrypos, int rfnpos,
    int arg1, int arg2, int arg3)
{
    int fraglen = GetNAlnPoss_var(qrylen, dbstrlen, qrypos, rfnpos, arg1, arg2, arg3);

    if(qrylen - qrypos < fraglen ||
       dbstrlen - rfnpos < fraglen)
        return true;

    qrylen = qrypos + fraglen;
    dbstrlen = rfnpos + fraglen;

    return false;
}
#endif



#if 0
// -------------------------------------------------------------------------
// GetQryRfnPos_frg: return the query and reference positions when 
// calculating transformation matrices from variable-length fragments of 
// two structures being compared;
// version of more extensive parallelization;
// arg1 is qryfragfct, fragment factor for query (to be multiplied by step 
// dependent upon lengths);
// arg2 is rfnfragfct, fragment factor for reference (to be multiplied by step 
// dependent upon lengths after summation);
__device__ __forceinline__
void GetQryRfnPos_frg(
    int& qrypos, int& rfnpos,
    int qrylen, int dbstrlen,
    uint sfragfct, int arg1, int arg2, int arg3)
{
    qrypos = rfnpos = 0;
    int fraglen = GetNAlnPoss_frg(
            qrylen, dbstrlen, 0/*qrypos,unsed*/, 0/*rfnpos,unused*/,
            arg1, arg2, arg3);
    //number of fragments along the reference structure
    //(step size is 1):
    int nrfnfrgs = dbstrlen - fraglen;
    if(nrfnfrgs < 1) return;
    //fragment factor for query:
    int qryfragfct = (arg2 + sfragfct) / nrfnfrgs;
    int rfnfragfct = (arg2 + sfragfct) - qryfragfct * nrfnfrgs;
    qrypos = (arg1 + qryfragfct) * GetFragStepSize_frg(qrylen);
    rfnpos = (rfnfragfct);//* 1;
}

// GetQryRfnPos_frg5: reference positions obtained by GetQryRfnPos_frg 
// multiplied by a step size factor of 5;
__device__ __forceinline__
void GetQryRfnPos_frg5(
    int& qrypos, int& rfnpos,
    int qrylen, int dbstrlen,
    uint sfragfct, int arg1, int arg2, int arg3)
{
    constexpr int szstep = 5;
    qrypos = rfnpos = 0;
    int fraglen = GetNAlnPoss_frg(
            qrylen, dbstrlen, 0/*qrypos,unsed*/, 0/*rfnpos,unused*/,
            arg1, arg2, arg3);
    //number of fragments along the reference structure
    //(step size is 5):
    int nrfnfrgs = dbstrlen - fraglen;
    if(nrfnfrgs < 1) return;
    nrfnfrgs = (nrfnfrgs + szstep - 1) / szstep;
    //fragment factor for query:
    int qryfragfct = (arg2 + sfragfct) / nrfnfrgs;
    int rfnfragfct = (arg2 + sfragfct) - qryfragfct * nrfnfrgs;
    qrypos = (arg1 + qryfragfct) * GetFragStepSize_frg(qrylen);
    rfnpos = (rfnfragfct) * szstep;
}
#endif

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// GetQryRfnPos_frg2: return the query and reference positions when 
// calculating transformation matrices from variable-length fragments of 
// two structures being compared; version of more extensive parallelization;
// arg1 is qryfragfct, fragment factor for query (to be multiplied by step 
// dependent upon lengths);
// arg2 is rfnfragfct, fragment factor for reference (to be multiplied by step 
// dependent upon lengths after summation);
__device__ __forceinline__
void GetQryRfnPos_frg2(
    const int depth,
    int& qrypos, int& rfnpos,
    int qrylen, int dbstrlen,
    uint sfragfct, int /*arg1*/, int arg2, int /*arg3*/)
{
    int qrystepsz = GetFragStepSize_frg_shallow(qrylen);
    int rfnstepsz = GetFragStepSize_frg_shallow(dbstrlen);
    if(depth == CLOptions::csdDeep) {
        qrystepsz = GetFragStepSize_frg_deep(qrylen);
        rfnstepsz = GetFragStepSize_frg_deep(dbstrlen);
    } else if(depth == CLOptions::csdHigh) {
        qrystepsz = GetFragStepSize_frg_high(qrylen);
        rfnstepsz = GetFragStepSize_frg_high(dbstrlen);
    } else if(depth == CLOptions::csdMedium) {
        qrystepsz = GetFragStepSize_frg_medium(qrylen);
        rfnstepsz = GetFragStepSize_frg_medium(dbstrlen);
    }
    //number of fragments along the reference structure:
    int nrfnfrgs = myhdmax(1, dbstrlen / rfnstepsz);
    //fragment factor for query:
    int qryfragfct = (arg2 + (sfragfct>>1)) / nrfnfrgs;
    int rfnfragfct = (arg2 + (sfragfct>>1)) - qryfragfct * nrfnfrgs;
    // NOTE: previous version with different qryfragfct and rfnfragfct journaling:
    // qrypos = (arg1 + qryfragfct) * qrystepsz;
    qrypos = (qryfragfct) * qrystepsz;
    rfnpos = (rfnfragfct) * rfnstepsz;
}

// -------------------------------------------------------------------------
// GetQryRfnFct_frg2: return the query and reference position factorss given
// their lengths and general index sfragfct;
// arg2, starting fragment factor rfnfragfct for reference;
__device__ __forceinline__
void GetQryRfnFct_frg2(
    const int depth,
    int* qryfragfct, int* rfnfragfct,
    int qrylen, int dbstrlen,
    uint sfragfct, int arg2)
{
    int qrystepsz = GetFragStepSize_frg_shallow(qrylen);
    int rfnstepsz = GetFragStepSize_frg_shallow(dbstrlen);
    if(depth == CLOptions::csdDeep) {
        qrystepsz = GetFragStepSize_frg_deep(qrylen);
        rfnstepsz = GetFragStepSize_frg_deep(dbstrlen);
    } else if(depth == CLOptions::csdHigh) {
        qrystepsz = GetFragStepSize_frg_high(qrylen);
        rfnstepsz = GetFragStepSize_frg_high(dbstrlen);
    } else if(depth == CLOptions::csdMedium) {
        qrystepsz = GetFragStepSize_frg_medium(qrylen);
        rfnstepsz = GetFragStepSize_frg_medium(dbstrlen);
    }
    //number of fragments along the reference structure:
    int nrfnfrgs = myhdmax(1, (int)__fdividef(dbstrlen, rfnstepsz));
    //fragment factor for query:
    *qryfragfct = __fdividef((arg2 + (sfragfct>>1)), nrfnfrgs);
    *rfnfragfct = (arg2 + (sfragfct>>1)) - (*qryfragfct) * nrfnfrgs;
}

// -------------------------------------------------------------------------
// PositionsOutofBounds_frg: return true if the positions given for
// query and reference imply the out-of-bounds condition when the alignment
// length is dependent upon structure lengths;
// version of more extensive parallelization;
// arg2 is fragndx, fragment index determining the fragment size dependent 
// upon lengths;
__device__ __forceinline__
bool PositionsOutofBounds_frg(
    int& qrylen, int& dbstrlen,
    int qrypos, int rfnpos,
    int arg1, int arg2, int arg3)
{
    int fraglen = GetNAlnPoss_frg(
            qrylen, dbstrlen, qrypos, rfnpos, arg1, arg2, arg3);

    if(qrylen - qrypos < fraglen ||
       dbstrlen - rfnpos < fraglen)
        return true;

    qrylen = qrypos + fraglen;
    dbstrlen = rfnpos + fraglen;

    return false;
}



#if 0
// -------------------------------------------------------------------------
// GetNAlnPoss_frgbets: return the maximum number of alignment positions 
// (fragment size) for best fragment identified previously;
__device__ __forceinline__
int GetNAlnPoss_frgbest(
    int /*qrylen*/, int /*dbstrlen*/,
    int /*qrypos*/, int /*rfnpos*/,
    int /*arg1*/, int /*arg2*/, int arg3)
{
    return arg3;
}
// GetQryRfnPos_frgbest: return the query and reference positions for best 
// fragment identified before;
__device__ __forceinline__
void GetQryRfnPos_frgbest(
    int& qrypos, int& rfnpos,
    int /*qrylen*/, int /*dbstrlen*/,
    uint /*sfragfct*/, int arg1, int arg2, int /*arg3*/)
{
    qrypos = arg1;
    rfnpos = arg2;
}
// PositionsOutofBounds_frgbest: return true if the positions given for
// query and reference imply the out-of-bounds condition;
__device__ __forceinline__
bool PositionsOutofBounds_frgbest(
    int& qrylen, int& dbstrlen,
    int qrypos, int rfnpos,
    int arg1, int arg2, int arg3)
{
    int fraglen = arg3;

    if(qrylen - qrypos < fraglen ||
       dbstrlen - rfnpos < fraglen)
        return true;

    qrylen = qrypos + fraglen;
    dbstrlen = rfnpos + fraglen;

    return false;
}
#endif



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// UpdateCCMCache: update cross-covariance cache data given query and 
// reference coordinates, respectively;
//
template<int SMIDIM = twmvEndOfCCData>
__device__ __forceinline__
void UpdateCCMCache(
    FPTYPE* __restrict__ ccmCache,
    float qx, float qy, float qz,
    float rx, float ry, float rz)
{
    ccmCache[threadIdx.x * SMIDIM + twmvCCM_0_0] += qx * rx;
    ccmCache[threadIdx.x * SMIDIM + twmvCCM_0_1] += qx * ry;
    ccmCache[threadIdx.x * SMIDIM + twmvCCM_0_2] += qx * rz;

    ccmCache[threadIdx.x * SMIDIM + twmvCCM_1_0] += qy * rx;
    ccmCache[threadIdx.x * SMIDIM + twmvCCM_1_1] += qy * ry;
    ccmCache[threadIdx.x * SMIDIM + twmvCCM_1_2] += qy * rz;

    ccmCache[threadIdx.x * SMIDIM + twmvCCM_2_0] += qz * rx;
    ccmCache[threadIdx.x * SMIDIM + twmvCCM_2_1] += qz * ry;
    ccmCache[threadIdx.x * SMIDIM + twmvCCM_2_2] += qz * rz;

    ccmCache[threadIdx.x * SMIDIM + twmvCVq_0] += qx;
    ccmCache[threadIdx.x * SMIDIM + twmvCVq_1] += qy;
    ccmCache[threadIdx.x * SMIDIM + twmvCVq_2] += qz;

    ccmCache[threadIdx.x * SMIDIM + twmvCVr_0] += rx;
    ccmCache[threadIdx.x * SMIDIM + twmvCVr_1] += ry;
    ccmCache[threadIdx.x * SMIDIM + twmvCVr_2] += rz;
}

// UpdateExtCCMCache: extension to UpdateCCMCache, where coordinate 
// squares are also updated;
//
template<int SMIDIM = twmvEndOfCCData>
__device__ __forceinline__
void UpdateExtCCMCache(
    float* __restrict__ ccmCache,
    float qx, float qy, float qz,
    float rx, float ry, float rz)
{
    UpdateCCMCache<SMIDIM>(ccmCache,  qx, qy, qz,  rx, ry, rz);

    ccmCache[threadIdx.x * SMIDIM + twmvCV2q_0] += SQRD(qx);
    ccmCache[threadIdx.x * SMIDIM + twmvCV2q_1] += SQRD(qy);
    ccmCache[threadIdx.x * SMIDIM + twmvCV2q_2] += SQRD(qz);

    ccmCache[threadIdx.x * SMIDIM + twmvCV2r_0] += SQRD(rx);
    ccmCache[threadIdx.x * SMIDIM + twmvCV2r_1] += SQRD(ry);
    ccmCache[threadIdx.x * SMIDIM + twmvCV2r_2] += SQRD(rz);
}

// -------------------------------------------------------------------------
// UpdateCCMOneAlnPos: update one position contributing to the 
// cross-covariance matrix between the query and reference structures
//
template<int SMIDIM = twmvEndOfCCData>
__device__ __forceinline__
void UpdateCCMOneAlnPos(
    int qrypos,
    int rfnpos,
    FPTYPE* __restrict__ ccmCache)
{
    float qx = GetQueryCoord<pmv2DX>(qrypos);
    float qy = GetQueryCoord<pmv2DY>(qrypos);
    float qz = GetQueryCoord<pmv2DZ>(qrypos);

    float rx = GetDbStrCoord<pmv2DX>(rfnpos);
    float ry = GetDbStrCoord<pmv2DY>(rfnpos);
    float rz = GetDbStrCoord<pmv2DZ>(rfnpos);

    UpdateCCMCache<SMIDIM>(ccmCache,  qx, qy, qz,  rx, ry, rz);
}

#endif//__covariance_h__
