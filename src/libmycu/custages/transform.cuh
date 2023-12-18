/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __transform_h__
#define __transform_h__

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libmymp/mpstages/transformbase.h"
#include "libmycu/cucom/cutemplates.h"

// memory initialization for final transformation matrices and data
__global__ void InitGTfmMatricesAndAData(
    const uint ndbCstrs,
    float* __restrict__ tfmmem,
    float* __restrict__ alndatamem
);

// memory initialization for transformation matrices
__global__ void InitTfmMatrices(
    const uint ndbCstrs,
    const uint maxnsteps,
    const uint minfraglen,
    const int sfragstep,
    const bool checkfragos,
    float* __restrict__ wrkmemtmibest
);

// RevertTfmMatrices: revert transformation matrices:
__global__ void RevertTfmMatrices(
    const uint ndbCstrs,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ tfmmem
);


// CalcTfmMatrices_DynamicOrientation: calculate tranformation matrices wrt
// query or reference structure, whichever is longer;
__global__ void CalcTfmMatrices_DynamicOrientation(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmem2
);

#define TFMTX_REVERSE_FALSE false
#define TFMTX_REVERSE_TRUE true

// apply the Kabsch algorithm to pairs of structures
template<bool REVERSE = TFMTX_REVERSE_FALSE, bool TFM_DINV = false>
__global__
void CalcTfmMatrices(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmem2
);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// CalcPartialA: calculate part of the matrix A while estimating 
// eigenvectors
//
template<int col_l, int ODIM = CUS1_TBSP_TFM_N>
__device__ __forceinline__
void CalcPartialA(
    float d,
    const float* __restrict__ rr,
    float* __restrict__ a)
{
    __shared__ float ssCache[ODIM][6];
    float* ss = ssCache[threadIdx.x];
    ss[0] = (d - rr[2]) * (d - rr[5]) - rr[4] * rr[4];
    ss[1] = (d - rr[5]) * rr[1] + rr[3] * rr[4];
    ss[2] = (d - rr[0]) * (d - rr[5]) - rr[3] * rr[3];
    ss[3] = (d - rr[2]) * rr[3] + rr[1] * rr[4];
    ss[4] = (d - rr[0]) * rr[4] + rr[1] * rr[3];
    ss[5] = (d - rr[0]) * (d - rr[2]) - rr[1] * rr[1];

    if(fabsf(ss[0]) <= TFMEPSILON) ss[0] = 0.0f;
    if(fabsf(ss[1]) <= TFMEPSILON) ss[1] = 0.0f;
    if(fabsf(ss[2]) <= TFMEPSILON) ss[2] = 0.0f;
    if(fabsf(ss[3]) <= TFMEPSILON) ss[3] = 0.0f;
    if(fabsf(ss[4]) <= TFMEPSILON) ss[4] = 0.0f;
    if(fabsf(ss[5]) <= TFMEPSILON) ss[5] = 0.0f;

    //change indexing
    int j = 3;
    if(fabsf(ss[0]) >= fabsf(ss[2])) {
        j = 0;
        if(fabsf(ss[0]) < fabsf(ss[5])) j = 3;
    }
    else if(fabsf(ss[2]) >= fabsf(ss[5])) j = 1;

    d = 0.0f;

    a[col_l]/*[0][l]*/ = ss[j]; d += ss[j] * ss[j];
    a[col_l+3]/*[1][l]*/ = ss[j+1]; d += ss[j+1] * ss[j+1];
    j = myhdmin(j+1+2,5);
    a[col_l+6]/*[2][l]*/ = ss[j]; d += ss[j] * ss[j];

    d = (d > TFMEPSILON)? rsqrtf(d): 0.0f;

    a[col_l]/*[0][l]*/ *= d;
    a[col_l+3]/*[1][l]*/ *= d;
    a[col_l+6]/*[2][l]*/ *= d;
}

// -------------------------------------------------------------------------
// CalcTfmMatricesHelper: helper to calculate tranformation matrices;
// NOTE: thread block is 1D and calculates multiple matrices simulatneously;
// REVERSE, template parameter, change places of query and reference sums 
// so that tranformation matrices are calculated wrt queries;
// REVERT_BACK, template parameter, revert back transformation matrices to 
// get them calculated wrt references again; NOTE: to achieve numerical 
// stability (index) and symmetric results for the same query-reference pair;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// NOTE: memory pointers should be aligned!
// wrkmem2, working memory, including the section of CC data;
// NOTE: keep #registers <64 when CUS1_TBSP_TFM_N == 32!
// NOTE: Based on the original Kabsch algorithm:
/*
c**** CALCULATES A BEST ROTATION & TRANSLATION BETWEEN TWO VECTOR SETS
c**** SUCH THAT U*X+T IS THE CLOSEST APPROXIMATION TO Y.
c**** THE CALCULATED BEST SUPERPOSITION MAY NOT BE UNIQUE AS INDICATED
c**** BY A RESULT VALUE IER=-1. HOWEVER IT IS GARANTIED THAT WITHIN
c**** NUMERICAL TOLERANCES NO OTHER SUPERPOSITION EXISTS GIVING A
c**** SMALLER VALUE FOR RMS.
c**** THIS VERSION OF THE ALGORITHM IS OPTIMIZED FOR THREE-DIMENSIONAL
c**** REAL VECTOR SPACE.
c**** USE OF THIS ROUTINE IS RESTRICTED TO NON-PROFIT ACADEMIC
c**** APPLICATIONS.
c**** PLEASE REPORT ERRORS TO
c**** PROGRAMMER:  W.KABSCH   MAX-PLANCK-INSTITUTE FOR MEDICAL RESEARCH
c        JAHNSTRASSE 29, 6900 HEIDELBERG, FRG.
c**** REFERENCES:  W.KABSCH   ACTA CRYST.(1978).A34,827-828
c           W.KABSCH ACTA CRYST.(1976).A32,922-923
c
c  W    - W(M) IS WEIGHT FOR ATOM PAIR  # M           (GIVEN)
c  X    - X(I,M) ARE COORDINATES OF ATOM # M IN SET X       (GIVEN)
c  Y    - Y(I,M) ARE COORDINATES OF ATOM # M IN SET Y       (GIVEN)
c  N    - N IS number OF ATOM PAIRS             (GIVEN)
c  MODE  - 0:CALCULATE RMS ONLY              (GIVEN)
c      1:CALCULATE RMS,U,T   (TAKES LONGER)
c  RMS   - SUM OF W*(UX+T-Y)**2 OVER ALL ATOM PAIRS        (RESULT)
c  U    - U(I,J) IS   ROTATION  MATRIX FOR BEST SUPERPOSITION  (RESULT)
c  T    - T(I)   IS TRANSLATION VECTOR FOR BEST SUPERPOSITION  (RESULT)
c  IER   - 0: A UNIQUE OPTIMAL SUPERPOSITION HAS BEEN DETERMINED(RESULT)
c     -1: SUPERPOSITION IS NOT UNIQUE BUT OPTIMAL
c     -2: NO RESULT OBTAINED BECAUSE OF NEGATIVE WEIGHTS W
c      OR ALL WEIGHTS EQUAL TO ZERO.
c
c-----------------------------------------------------------------------
*/
template<bool REVERSE, bool REVERT_BACK = false>
__device__ __forceinline__
void CalcTfmMatricesHelper(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmem2)
{
    // blockIdx.x is the reference serial number;
    // blockIdx.y is the query serial number (index in the chunk);
    // blockIdx.z is the fragment factor;

    //cache for cross-covarinace matrices and reuse: 
    //bank conflicts resolved as long as twmvEndOfCCData is odd
    __shared__ float ccmCache[CUS1_TBSP_TFM_N][twmvEndOfCCData];
    __shared__ float aCache[CUS1_TBSP_TFM_N][twmvEndOfCCMtx];
    __shared__ float rrCache[CUS1_TBSP_TFM_N][6];
    float* rr = rrCache[threadIdx.x];

    //blockIdx.x * CUS1_TBSP_TFM_N is the index of the first 
    //structure to start with in the thread block (blockIdx.x, refn. number;
    //(unrolling by a factor of CUS1_TBSP_TFM_N):
    int absndx = blockIdx.x * CUS1_TBSP_TFM_N + threadIdx.x;
    int qryndx = blockIdx.y;//query index in the chunk
    int sfragfct = blockIdx.z;//fragment factor

    if(ndbCstrs <= absndx)
        //no sync below: each thread runs independently: exit
        return;

    uint mloc = ((qryndx * maxnsteps + sfragfct) * nTWorkingMemoryVars) * ndbCstrs;
    //read the number of positions used to calculate cross-covarinaces:
    float nalnposs = wrkmem2[mloc + twmvNalnposs * ndbCstrs + absndx];

    if(nalnposs <= 0.0f)
        //no sync below: each thread runs independently: exit
        return;

    //read: coalesced in the block
    if(REVERSE == false) {
        #pragma unroll
        for(int i = 0; i < twmvEndOfCCData; i++)
            ccmCache[threadIdx.x][i] = wrkmem2[mloc + i * ndbCstrs + absndx];
    } else {
        //change places: query <-> reference
        ccmCache[threadIdx.x][twmvCCM_0_0] = wrkmem2[mloc + twmvCCM_0_0 * ndbCstrs + absndx];
        ccmCache[threadIdx.x][twmvCCM_0_1] = wrkmem2[mloc + twmvCCM_1_0 * ndbCstrs + absndx];
        ccmCache[threadIdx.x][twmvCCM_0_2] = wrkmem2[mloc + twmvCCM_2_0 * ndbCstrs + absndx];
        ccmCache[threadIdx.x][twmvCCM_1_0] = wrkmem2[mloc + twmvCCM_0_1 * ndbCstrs + absndx];
        ccmCache[threadIdx.x][twmvCCM_1_1] = wrkmem2[mloc + twmvCCM_1_1 * ndbCstrs + absndx];
        ccmCache[threadIdx.x][twmvCCM_1_2] = wrkmem2[mloc + twmvCCM_2_1 * ndbCstrs + absndx];
        ccmCache[threadIdx.x][twmvCCM_2_0] = wrkmem2[mloc + twmvCCM_0_2 * ndbCstrs + absndx];
        ccmCache[threadIdx.x][twmvCCM_2_1] = wrkmem2[mloc + twmvCCM_1_2 * ndbCstrs + absndx];
        ccmCache[threadIdx.x][twmvCCM_2_2] = wrkmem2[mloc + twmvCCM_2_2 * ndbCstrs + absndx];
        ccmCache[threadIdx.x][twmvCVr_0] = wrkmem2[mloc + twmvCVq_0 * ndbCstrs + absndx];
        ccmCache[threadIdx.x][twmvCVr_1] = wrkmem2[mloc + twmvCVq_1 * ndbCstrs + absndx];
        ccmCache[threadIdx.x][twmvCVr_2] = wrkmem2[mloc + twmvCVq_2 * ndbCstrs + absndx];
        ccmCache[threadIdx.x][twmvCVq_0] = wrkmem2[mloc + twmvCVr_0 * ndbCstrs + absndx];
        ccmCache[threadIdx.x][twmvCVq_1] = wrkmem2[mloc + twmvCVr_1 * ndbCstrs + absndx];
        ccmCache[threadIdx.x][twmvCVq_2] = wrkmem2[mloc + twmvCVr_2 * ndbCstrs + absndx];
    }

    //initialize matrix a
    //NOTE: only valid when indices start from 0
    RotMtxToIdentity(aCache[threadIdx.x]);

//     //NOTE: scale all dimensions of cross-covariance by 1/n^2 and vectors by 1/n to
//     // enable rotation matrix calculation in single-precision without overflow
//     #pragma unroll
//     for(int ii = 0; ii < twmvEndOfCCMtx; ii++)
//         ccmCache[threadIdx.x][ii] = fdividef(ccmCache[threadIdx.x][ii], SQRD(nalnposs));
//     #pragma unroll
//     for(int ii = twmvEndOfCCMtx; ii < twmvEndOfCCData; ii++)
//         ccmCache[threadIdx.x][ii] = fdividef(ccmCache[threadIdx.x][ii], nalnposs);


    //calculate query center vector in advance
    #pragma unroll
    for(int i = twmvCVq_0; i <= twmvCVq_2; i++)
        ccmCache[threadIdx.x][i] = fdividef(ccmCache[threadIdx.x][i], nalnposs);

    CalcRmatrix(ccmCache[threadIdx.x]);

    //calculate reference center vector now
    #pragma unroll
    for(int i = twmvCVr_0; i <= twmvCVr_2; i++)
        ccmCache[threadIdx.x][i] = fdividef(ccmCache[threadIdx.x][i], nalnposs);


    //NOTE: scale correlation matrix to enable rotation matrix 
    // calculation in single-precision without overflow and underflow
    ScaleRmatrix(ccmCache[threadIdx.x]);


    //calculate determinant
    float det = CalcDet(ccmCache[threadIdx.x]);

    //calculate the product transposed(R) * R
    CalcRTR(ccmCache[threadIdx.x], rr);

    //Kabsch:
    //eigenvalues: form characteristic cubic x**3-3*spur*x**2+3*cof*x-det=0
    float spur = (rr[0] + rr[2] + rr[5]) * oneTHIRDf;
    float cof = (((((rr[2] * rr[5] - SQRD(rr[4])) + rr[0] * rr[5]) -
                SQRD(rr[3])) + rr[0] * rr[2]) -
                SQRD(rr[1])) * oneTHIRDf;

    bool abok = (spur > 0.0f);

    if(abok)
    {   //Kabsch:
        //reduce cubic to standard form y**3-3hy+2g=0 by putting x=y+spur

        //Kabsch: solve cubic: roots are e[0],e[1],e[2] in decreasing order
        //Kabsch: handle special case of 3 identical roots
        float e0, e1, e2;
        if(SolveCubic(det, spur, cof, e0, e1, e2))
        {
            //Kabsch: eigenvectors
            //almost always this branch gets executed
            CalcPartialA_Reg<0>(e0, rr, aCache[threadIdx.x]);
            CalcPartialA_Reg<2>(e2, rr, aCache[threadIdx.x]);
            abok = CalcCompleteA(e0, e1, e2, aCache[threadIdx.x]);
        }
        if(abok) {
            //Kabsch: rotation matrix
            abok = CalcRotMtx(aCache[threadIdx.x], ccmCache[threadIdx.x]);
        }
    }

    if(!abok) RotMtxToIdentity(ccmCache[threadIdx.x]);

    //Kabsch: translation vector
    //NOTE: scaling translation vector would be needed if the data 
    // vectors were scaled previously so that transformation is 
    // applied in the original coordinate space
    CalcTrlVec(ccmCache[threadIdx.x]);

    if(REVERT_BACK) {
        InvertRotMtx(ccmCache[threadIdx.x]);
        InvertTrlVec(ccmCache[threadIdx.x]);
    }

    //write calculated transformation matrices back to the same 
    // area of memory: (coalesced in the block);
    // no sync: thread writes its own data
    #pragma unroll
    for(int i = 0; i < twmvEndOfTFMData; i++)
        wrkmem2[mloc + i * ndbCstrs + absndx] = ccmCache[threadIdx.x][i];
}

#endif//__transform_h__
