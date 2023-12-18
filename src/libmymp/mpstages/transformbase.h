/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __transformbase_h__
#define __transformbase_h__

#include <math.h>
#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libmycu/cucom/cudef.h"
#include "libmycu/cucom/cutemplates.h"

#define TFMEPSILON 1.e-8f
#define TFMTOL 0.01f

// -------------------------------------------------------------------------
// norm2: norm squared of the vector given by coordinates x, y, z;
//
__DINLINE__
float norm2(float x, float y, float z)
{
    return (x*x + y*y + z*z);
}

// -------------------------------------------------------------------------
// norm: calculate the norm of the vector given by coordinates x, y, z;
//
__DINLINE__
float norm(float x, float y, float z)
{
    return sqrtf(norm2(x, y, z));
}

// -------------------------------------------------------------------------
// distance2: calculate squared distance between two points;
//
__DINLINE__
float distance2(
    float x0, float x1, float x2,
    float y0, float y1, float y2)
{
    return myhdsqrdv(x0-y0) + myhdsqrdv(x1-y1) + myhdsqrdv(x2-y2);
}

// -------------------------------------------------------------------------
// translate_point: translate point in-place;
// tfm, transformation matrix;
// x0, x1, x2: x, y, z, coordinates of the point;
//
__DINLINE__
void translate_point(
    const float *__RESTRICT__ tfm,
    float& x0, float& x1, float& x2)
{
    x0 += tfm[tfmmTrl_0];
    x1 += tfm[tfmmTrl_1];
    x2 += tfm[tfmmTrl_2];
}
// -------------------------------------------------------------------------
// rotate_point: rotate point in-place;
// tfm, transformation matrix;
// x0, x1, x2: x, y, z, coordinates of the point;
//
__DINLINE__
void rotate_point(
    const float *__RESTRICT__ tfm,
    float& x0, float& x1, float& x2)
{
    float x = x0, y = x1, z = x2;
    x0 = tfm[tfmmRot_0_0] * x + tfm[tfmmRot_0_1] * y + tfm[tfmmRot_0_2] * z;
    x1 = tfm[tfmmRot_1_0] * x + tfm[tfmmRot_1_1] * y + tfm[tfmmRot_1_2] * z;
    x2 = tfm[tfmmRot_2_0] * x + tfm[tfmmRot_2_1] * y + tfm[tfmmRot_2_2] * z;
}

// -------------------------------------------------------------------------
// transform_point: transform point in-place;
// tfm, transformation matrix;
// x0, x1, x2: x, y, z, coordinates of the point;
//
__DINLINE__
void transform_point(
    const float *__RESTRICT__ tfm,
    float& x0, float& x1, float& x2)
{
    float x = x0, y = x1, z = x2;
    x0 = tfm[tfmmRot_0_0] * x + tfm[tfmmRot_0_1] * y + tfm[tfmmRot_0_2] * z + tfm[tfmmTrl_0];
    x1 = tfm[tfmmRot_1_0] * x + tfm[tfmmRot_1_1] * y + tfm[tfmmRot_1_2] * z + tfm[tfmmTrl_1];
    x2 = tfm[tfmmRot_2_0] * x + tfm[tfmmRot_2_1] * y + tfm[tfmmRot_2_2] * z + tfm[tfmmTrl_2];
}

// -------------------------------------------------------------------------
// transform_and_distance2: transform point in-place and calculate squared 
// distance;
// tfm, transformation matrix;
// x0, x1, x2: x, y, z, coordinates of the point to transform;
// y0, y1, y2: x, y, z, coordinates of the other point;
//
__DINLINE__
float transform_and_distance2(
    const float *__RESTRICT__ tfm,
    float x0, float x1, float x2,
    float y0, float y1, float y2)
{
    transform_point(tfm, x0, x1, x2);
    return distance2(x0, x1, x2,  y0, y1, y2);
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// TransposeRotMtx: matrix transpose;
//
__DINLINE__
void TransposeRotMtx(float *__RESTRICT__ u)
{
    myhdswap(u[twmvCCM_0_1], u[twmvCCM_1_0]);
    myhdswap(u[twmvCCM_0_2], u[twmvCCM_2_0]);
    myhdswap(u[twmvCCM_1_2], u[twmvCCM_2_1]);
}

// InvertRotMtx: invert rotation matrix u in the cache; result is transpose;
__DINLINE__
void InvertRotMtx(float *__RESTRICT__ u)
{
    TransposeRotMtx(u);
}

// -------------------------------------------------------------------------
// InvertTrlVec: calculate reverted translation vector t (which is in ut);
// the resulting t overwrites the query center vector in the cache;
//
__DINLINE__
void InvertTrlVec(float *__RESTRICT__ ut)
{
    rotate_point(ut, ut[twmvCVq_0], ut[twmvCVq_1], ut[twmvCVq_2]);
    ut[twmvCVq_0] = -ut[twmvCVq_0];
    ut[twmvCVq_1] = -ut[twmvCVq_1];
    ut[twmvCVq_2] = -ut[twmvCVq_2];
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// RotMtxToIdentity: assign rotation matrix u in the cache to the identity 
// matrix;
//
__DINLINE__
void RotMtxToIdentity(float *__RESTRICT__ u)
{
    u[twmvCCM_0_0] = u[twmvCCM_1_1] = u[twmvCCM_2_2] = 1.0f;

    u[twmvCCM_0_1] = u[twmvCCM_0_2] = 0.0f;
    u[twmvCCM_1_0] = u[twmvCCM_1_2] = 0.0f;
    u[twmvCCM_2_0] = u[twmvCCM_2_1] = 0.0f;
}

// -------------------------------------------------------------------------
// CalcHmatrix: calculate the H matrix inline (multiplied by n; 
// cross-covariance transpose - n x mean product; Kabsch correlation matrix r)
//
__DINLINE__
void CalcRmatrix(float* __RESTRICT__ ccmCache)
{
    //read values that will be updated before reading them next;
    float ccm01 = ccmCache[twmvCCM_0_1];
    float ccm02 = ccmCache[twmvCCM_0_2];
    float ccm12 = ccmCache[twmvCCM_1_2];

    ccmCache[twmvCCM_0_0] = ccmCache[twmvCCM_0_0] - ccmCache[twmvCVq_0] * ccmCache[twmvCVr_0];
    ccmCache[twmvCCM_0_1] = ccmCache[twmvCCM_1_0] - ccmCache[twmvCVq_1] * ccmCache[twmvCVr_0];
    ccmCache[twmvCCM_0_2] = ccmCache[twmvCCM_2_0] - ccmCache[twmvCVq_2] * ccmCache[twmvCVr_0];

    ccmCache[twmvCCM_1_0] = ccm01 - ccmCache[twmvCVq_0] * ccmCache[twmvCVr_1];
    ccmCache[twmvCCM_1_1] = ccmCache[twmvCCM_1_1] - ccmCache[twmvCVq_1] * ccmCache[twmvCVr_1];
    ccmCache[twmvCCM_1_2] = ccmCache[twmvCCM_2_1] - ccmCache[twmvCVq_2] * ccmCache[twmvCVr_1];

    ccmCache[twmvCCM_2_0] = ccm02 - ccmCache[twmvCVq_0] * ccmCache[twmvCVr_2];
    ccmCache[twmvCCM_2_1] = ccm12 - ccmCache[twmvCVq_1] * ccmCache[twmvCVr_2];
    ccmCache[twmvCCM_2_2] = ccmCache[twmvCCM_2_2] - ccmCache[twmvCVq_2] * ccmCache[twmvCVr_2];
}

// -------------------------------------------------------------------------
// GetRScale: calculate the scale for correlation matrix r
//
__DINLINE__
float GetRScale(const float *__RESTRICT__ r)
{
    return (
        fabsf(r[twmvCCM_0_0]) + fabsf(r[twmvCCM_0_1]) + fabsf(r[twmvCCM_0_2]) +
        fabsf(r[twmvCCM_1_0]) + fabsf(r[twmvCCM_1_1]) + fabsf(r[twmvCCM_1_2]) +
        fabsf(r[twmvCCM_2_0]) + fabsf(r[twmvCCM_2_1]) + fabsf(r[twmvCCM_2_2])) *
        oneNINTHf;//oneSIXTHf;
}

// -------------------------------------------------------------------------
// ScaleRmatrix: scale correlation matrix r
//
__DINLINE__
void ScaleRmatrix(float *__RESTRICT__ r)
{
    float scale = GetRScale(r);

    r[twmvCCM_0_0] = tfmfdividef(r[twmvCCM_0_0], scale);
    r[twmvCCM_0_1] = tfmfdividef(r[twmvCCM_0_1], scale);
    r[twmvCCM_0_2] = tfmfdividef(r[twmvCCM_0_2], scale);

    r[twmvCCM_1_0] = tfmfdividef(r[twmvCCM_1_0], scale);
    r[twmvCCM_1_1] = tfmfdividef(r[twmvCCM_1_1], scale);
    r[twmvCCM_1_2] = tfmfdividef(r[twmvCCM_1_2], scale);

    r[twmvCCM_2_0] = tfmfdividef(r[twmvCCM_2_0], scale);
    r[twmvCCM_2_1] = tfmfdividef(r[twmvCCM_2_1], scale);
    r[twmvCCM_2_2] = tfmfdividef(r[twmvCCM_2_2], scale);
}

// -------------------------------------------------------------------------
// CalcDet: calculate the determinant of R
//
__DINLINE__
float CalcDet(const float* __RESTRICT__ ccmCache)
{
    return 
        ccmCache[twmvCCM_0_0] * (
            ccmCache[twmvCCM_1_1] * ccmCache[twmvCCM_2_2] -
            ccmCache[twmvCCM_1_2] * ccmCache[twmvCCM_2_1])
        -
        ccmCache[twmvCCM_0_1] * (
            ccmCache[twmvCCM_1_0] * ccmCache[twmvCCM_2_2] -
            ccmCache[twmvCCM_1_2] * ccmCache[twmvCCM_2_0])
        +
        ccmCache[twmvCCM_0_2] * (
            ccmCache[twmvCCM_1_0] * ccmCache[twmvCCM_2_1] -
            ccmCache[twmvCCM_1_1] * ccmCache[twmvCCM_2_0])
    ;
}

// -------------------------------------------------------------------------
// CalcRTR: calculate the product of R transposed and R, and write the 
// result of the upper triangle rr
//
__DINLINE__
void CalcRTR(
    const float* __RESTRICT__ ccmCache,
    float* __RESTRICT__ rr)
{
    rr[0] = ccmCache[twmvCCM_0_0] * ccmCache[twmvCCM_0_0] +
        ccmCache[twmvCCM_1_0] * ccmCache[twmvCCM_1_0] +
        ccmCache[twmvCCM_2_0] * ccmCache[twmvCCM_2_0];

    rr[1] = ccmCache[twmvCCM_0_0] * ccmCache[twmvCCM_0_1] +
        ccmCache[twmvCCM_1_0] * ccmCache[twmvCCM_1_1] +
        ccmCache[twmvCCM_2_0] * ccmCache[twmvCCM_2_1];

    rr[2] = ccmCache[twmvCCM_0_1] * ccmCache[twmvCCM_0_1] +
        ccmCache[twmvCCM_1_1] * ccmCache[twmvCCM_1_1] +
        ccmCache[twmvCCM_2_1] * ccmCache[twmvCCM_2_1];

    rr[3] = ccmCache[twmvCCM_0_0] * ccmCache[twmvCCM_0_2] +
        ccmCache[twmvCCM_1_0] * ccmCache[twmvCCM_1_2] +
        ccmCache[twmvCCM_2_0] * ccmCache[twmvCCM_2_2];

    rr[4] = ccmCache[twmvCCM_0_1] * ccmCache[twmvCCM_0_2] +
        ccmCache[twmvCCM_1_1] * ccmCache[twmvCCM_1_2] +
        ccmCache[twmvCCM_2_1] * ccmCache[twmvCCM_2_2];

    rr[5] = ccmCache[twmvCCM_0_2] * ccmCache[twmvCCM_0_2] +
        ccmCache[twmvCCM_1_2] * ccmCache[twmvCCM_1_2] +
        ccmCache[twmvCCM_2_2] * ccmCache[twmvCCM_2_2];
}

// -------------------------------------------------------------------------
// SolveCubic: Kabsch: solve cubic: roots are e[0],e[1],e[2] in decreasing 
// order;
// Kabsch: handle special case of 3 identical roots
//
__DINLINE__
bool SolveCubic(
    float det, float spur, float cof,
    float& e0, float& e1, float& e2)
{
    float h = spur * spur - cof;
    if(h <= 0.0f) return false;

    float g = (spur * cof - det * det) * 0.5f - spur * h;
    float sqrth = sqrtf(h);
    float cth, sth;
//     float d = h * h * h - g * g;//sqrth * h
//     if(d < 0.0f) d = 0.0f;
//     d = atan2f(sqrtf(d), -g) * oneTHIRDf;
    float d = atan2f(sqrtf(myhdmax(h*h*h - g*g, 0.0f)), -g) * oneTHIRDf;
    //NOTE: sincosf uses lots of registers;
    // compiler starts using stack frame;
    // __sincosf approximation requires much less registers:
    // argument d is within -pi/2 and pi/2 ([-pi;pi] according to the 
    // documentation): __sincosf error is 2^-21.19
    tfmsincosf(d, sth, cth);
    cth = sqrth * cth;
    sth = sqrth * SQRTf3 * sth;
    e0 = (spur + cth) + cth;
    e1 = (spur - cth) + sth;
    e2 = (spur - cth) - sth;
    return true;
}

// -------------------------------------------------------------------------
// CalcPartialA_Reg: calculate part of the matrix A while estimating 
// eigenvectors; using registers for local variables;
//
template<int col_l>
__DINLINE__
void CalcPartialA_Reg(
    float d,
    const float* __RESTRICT__ rr,
    float* __RESTRICT__ a)
{
    float ss0 = (d - rr[2]) * (d - rr[5]) - rr[4] * rr[4];
    float ss1 = (d - rr[5]) * rr[1] + rr[3] * rr[4];
    float ss2 = (d - rr[0]) * (d - rr[5]) - rr[3] * rr[3];
    float ss3 = (d - rr[2]) * rr[3] + rr[1] * rr[4];
    float ss4 = (d - rr[0]) * rr[4] + rr[1] * rr[3];
    float ss5 = (d - rr[0]) * (d - rr[2]) - rr[1] * rr[1];

    if(fabsf(ss0) <= TFMEPSILON) ss0 = 0.0f;
    if(fabsf(ss1) <= TFMEPSILON) ss1 = 0.0f;
    if(fabsf(ss2) <= TFMEPSILON) ss2 = 0.0f;
    if(fabsf(ss3) <= TFMEPSILON) ss3 = 0.0f;
    if(fabsf(ss4) <= TFMEPSILON) ss4 = 0.0f;
    if(fabsf(ss5) <= TFMEPSILON) ss5 = 0.0f;

    if(fabsf(ss0) >= fabsf(ss2)) {
        if(fabsf(ss0) < fabsf(ss5)) {
            a[col_l] = ss3; a[col_l+3] = ss4; a[col_l+6] = ss5;
        } else {
            a[col_l] = ss0; a[col_l+3] = ss1; a[col_l+6] = ss3;
        }
    }
    else if(fabsf(ss2) >= fabsf(ss5)) {
        a[col_l] = ss1; a[col_l+3] = ss2; a[col_l+6] = ss4;
    } else {
        a[col_l] = ss3; a[col_l+3] = ss4; a[col_l+6] = ss5;
    }

    d = SQRD(a[col_l]) + SQRD(a[col_l+3]) + SQRD(a[col_l+6]);

    d = (d > TFMEPSILON)? tfmrsqrtf(d): 0.0f;

    a[col_l]/*[0][l]*/ *= d;
    a[col_l+3]/*[1][l]*/ *= d;
    a[col_l+6]/*[2][l]*/ *= d;
}

// -------------------------------------------------------------------------
// CalcCompleteA: complete calculating matrix A while estimating 
// eigenvectors;
// return a flag whether the a calculation succeeded;
//
__DINLINE__
bool CalcCompleteA(
    float e0, float e1, float e2,
    float* __RESTRICT__ a)
{
    float d = a[0]/*[0][0]*/ * a[2]/*[0][2]*/ +
        a[3]/*[1][0]*/ * a[5]/*[1][2]*/ + a[6]/*[2][0]*/ * a[8]/*[2][2]*/;

    int m1 = 0, m = 2;

    if(e0 - e1 > e1 - e2) {
        m1 = 2;
        m = 0;
    }

    float p = 0.0f;

    a[m1]/*[0][m1]*/ -= d * a[m]/*[0][m]*/; p += a[m1] * a[m1];
    a[m1+3]/*[1][m1]*/ -= d * a[m+3]/*[1][m]*/; p += a[m1+3] * a[m1+3];
    a[m1+6]/*[2][m1]*/ -= d * a[m+6]/*[2][m]*/; p += a[m1+6] * a[m1+6];

    if(p > TFMTOL) {
        p = tfmrsqrtf(p);
        a[m1]/*[0][m1]*/ *= p;
        a[m1+3]/*[1][m1]*/ *= p;
        a[m1+6]/*[2][m1]*/ *= p;
    }
    else {
        p = 1.0f;

        //j, k, and l are row indices of a
        int j = 6, k = 0, l = 3;

        if(fabsf(a[m]/*[0][m]*/) <= p) { p = fabsf(a[m]); j = 0; k = 3; l = 6; }
        if(fabsf(a[m+3]/*[1][m]*/) <= p) { p = fabsf(a[m+3]); j = 3; k = 6; l = 0; }
        if(fabsf(a[m+6]/*[2][m]*/) <= p) { p = fabsf(a[m+6]); j = 6; k = 0; l = 3; }

        p = sqrtf(a[k+m]/*[k][m]*/ * a[k+m] + a[l+m]/*[l][m]*/ * a[l+m]);

        if(p <= TFMTOL)
            return false;//failed

        a[j+m1]/*[j][m1]*/ = 0.0f;
        a[k+m1]/*[k][m1]*/ = tfmfdividef(-a[l+m]/*[l][m]*/, p);
        a[l+m1]/*[l][m1]*/ = tfmfdividef(a[k+m]/*[k][m]*/, p);
    }

    a[1]/*[0][1]*/ = a[5]/*[1][2]*/ * a[6]/*[2][0]*/ - a[3]/*[1][0]*/ * a[8]/*[2][2]*/;
    a[4]/*[1][1]*/ = a[8]/*[2][2]*/ * a[0]/*[0][0]*/ - a[6]/*[2][0]*/ * a[2]/*[0][2]*/;
    a[7]/*[2][1]*/ = a[2]/*[0][2]*/ * a[3]/*[1][0]*/ - a[0]/*[0][0]*/ * a[5]/*[1][2]*/;

    return true;
}

// -------------------------------------------------------------------------
// CalcRotMtx: calculate rotation matrix after calculating unit vectors 
// (matrix) b; b and rotation matrix u later are all written inline in the 
// same cache used to keep correlation matrix r;
// return a flag whether the u (rot. matrix) calculation succeeded;
//
__DINLINE__
bool CalcRotMtx(
    const float* __RESTRICT__ a,
    float *__RESTRICT__ bu)
{
    float d = 0.0f;
    float r0 = bu[twmvCCM_0_0];
    float r1 = bu[twmvCCM_1_0];
    float r2 = bu[twmvCCM_2_0];

    //l==0
    bu[twmvCCM_0_0]/*b[0][l]*/ = r0 * a[0]/*[0][l]*/ +
        bu[twmvCCM_0_1]/*r[0][1]*/ * a[3]/*[1][l]*/ +
        bu[twmvCCM_0_2]/*r[0][2]*/ * a[6]/*[2][l]*/;
    d += SQRD(bu[twmvCCM_0_0]);

    bu[twmvCCM_1_0]/*b[1][l]*/ = r1 * a[0]/*[0][l]*/ +
        bu[twmvCCM_1_1]/*r[1][1]*/ * a[3]/*[1][l]*/ +
        bu[twmvCCM_1_2]/*r[1][2]*/ * a[6]/*[2][l]*/;
    d += SQRD(bu[twmvCCM_1_0]);

    bu[twmvCCM_2_0]/*b[2][l]*/ = r2 * a[0]/*[0][l]*/ +
        bu[twmvCCM_2_1]/*r[2][1]*/ * a[3]/*[1][l]*/ +
        bu[twmvCCM_2_2]/*r[2][2]*/ * a[6]/*[2][l]*/;
    d += SQRD(bu[twmvCCM_2_0]);

    d = (d > TFMEPSILON)? tfmrsqrtf(d): 0.0f;
    bu[twmvCCM_0_0] *= d;
    bu[twmvCCM_1_0] *= d;
    bu[twmvCCM_2_0] *= d;

    //l==1
    d = 0.0f;
    bu[twmvCCM_0_1]/*b[0][l]*/ = r0 * a[1]/*[0][l]*/ +
        bu[twmvCCM_0_1]/*r[0][1]*/ * a[4]/*[1][l]*/ +
        bu[twmvCCM_0_2]/*r[0][2]*/ * a[7]/*[2][l]*/;
    d += SQRD(bu[twmvCCM_0_1]);

    bu[twmvCCM_1_1]/*b[1][l]*/ = r1 * a[1]/*[0][l]*/ +
        bu[twmvCCM_1_1]/*r[1][1]*/ * a[4]/*[1][l]*/ +
        bu[twmvCCM_1_2]/*r[1][2]*/ * a[7]/*[2][l]*/;
    d += SQRD(bu[twmvCCM_1_1]);

    bu[twmvCCM_2_1]/*b[2][l]*/ = r2 * a[1]/*[0][l]*/ +
        bu[twmvCCM_2_1]/*r[2][1]*/ * a[4]/*[1][l]*/ +
        bu[twmvCCM_2_2]/*r[2][2]*/ * a[7]/*[2][l]*/;
    d += SQRD(bu[twmvCCM_2_1]);

    d = (d > TFMEPSILON)? tfmrsqrtf(d): 0.0f;
    bu[twmvCCM_0_1] *= d;
    bu[twmvCCM_1_1] *= d;
    bu[twmvCCM_2_1] *= d;


    d = bu[twmvCCM_0_0] * bu[twmvCCM_0_1] +
        bu[twmvCCM_1_0] * bu[twmvCCM_1_1] +
        bu[twmvCCM_2_0] * bu[twmvCCM_2_1];

    float p = 0.0f;

    bu[twmvCCM_0_1] -= d * bu[twmvCCM_0_0]; p += SQRD(bu[twmvCCM_0_1]);
    bu[twmvCCM_1_1] -= d * bu[twmvCCM_1_0]; p += SQRD(bu[twmvCCM_1_1]);
    bu[twmvCCM_2_1] -= d * bu[twmvCCM_2_0]; p += SQRD(bu[twmvCCM_2_1]);

    if(p > TFMTOL) {
        p = tfmrsqrtf(p);
        bu[twmvCCM_0_1] *= p;
        bu[twmvCCM_1_1] *= p;
        bu[twmvCCM_2_1] *= p;
    }
    else {
        p = 1.0f;

        //j, k, and l are row indices of b
        int j = twmvCCM_2_0, k = twmvCCM_0_0, l = twmvCCM_1_0;

        if(fabsf(bu[twmvCCM_0_0]) <= p) {
            p = fabsf(bu[twmvCCM_0_0]);
            j = twmvCCM_0_0; k = twmvCCM_1_0; l = twmvCCM_2_0;
        }
        if(fabsf(bu[twmvCCM_1_0]) <= p) {
            p = fabsf(bu[twmvCCM_1_0]);
            j = twmvCCM_1_0; k = twmvCCM_2_0; l = twmvCCM_0_0;
        }
        if(fabsf(bu[twmvCCM_2_0]) <= p) {
            p = fabsf(bu[twmvCCM_2_0]);
            j = twmvCCM_2_0; k = twmvCCM_0_0; l = twmvCCM_1_0;
        }

        p = sqrtf(SQRD(bu[k]) + SQRD(bu[l]));

        if(p <= TFMTOL)
            return false;//failed

        bu[j+1] = 0.0f;//2nd column
        bu[k+1] = tfmfdividef(-bu[l], p);
        bu[l+1] = tfmfdividef(bu[k], p);
    }

    bu[twmvCCM_0_2] = 
        bu[twmvCCM_1_0] * bu[twmvCCM_2_1] -
        bu[twmvCCM_1_1] * bu[twmvCCM_2_0];

    bu[twmvCCM_1_2] = 
        bu[twmvCCM_2_0] * bu[twmvCCM_0_1] -
        bu[twmvCCM_2_1] * bu[twmvCCM_0_0];

    bu[twmvCCM_2_2] = 
        bu[twmvCCM_0_0] * bu[twmvCCM_1_1] -
        bu[twmvCCM_0_1] * bu[twmvCCM_1_0];

    //rotation matrix:
    r0 = bu[twmvCCM_0_0]; r1 = bu[twmvCCM_0_1]; r2 = bu[twmvCCM_0_2];

    bu[twmvCCM_0_0] = r0 * a[0]/*[0][0]*/ + r1 * a[1]/*[0][1]*/ + r2 * a[2]/*[0][2]*/;
    bu[twmvCCM_0_1] = r0 * a[3]/*[1][0]*/ + r1 * a[4]/*[1][1]*/ + r2 * a[5]/*[1][2]*/;
    bu[twmvCCM_0_2] = r0 * a[6]/*[2][0]*/ + r1 * a[7]/*[2][1]*/ + r2 * a[8]/*[2][2]*/;

    r0 = bu[twmvCCM_1_0]; r1 = bu[twmvCCM_1_1]; r2 = bu[twmvCCM_1_2];

    bu[twmvCCM_1_0] = r0 * a[0]/*[0][0]*/ + r1 * a[1]/*[0][1]*/ + r2 * a[2]/*[0][2]*/;
    bu[twmvCCM_1_1] = r0 * a[3]/*[1][0]*/ + r1 * a[4]/*[1][1]*/ + r2 * a[5]/*[1][2]*/;
    bu[twmvCCM_1_2] = r0 * a[6]/*[2][0]*/ + r1 * a[7]/*[2][1]*/ + r2 * a[8]/*[2][2]*/;

    r0 = bu[twmvCCM_2_0]; r1 = bu[twmvCCM_2_1]; r2 = bu[twmvCCM_2_2];

    bu[twmvCCM_2_0] = r0 * a[0]/*[0][0]*/ + r1 * a[1]/*[0][1]*/ + r2 * a[2]/*[0][2]*/;
    bu[twmvCCM_2_1] = r0 * a[3]/*[1][0]*/ + r1 * a[4]/*[1][1]*/ + r2 * a[5]/*[1][2]*/;
    bu[twmvCCM_2_2] = r0 * a[6]/*[2][0]*/ + r1 * a[7]/*[2][1]*/ + r2 * a[8]/*[2][2]*/;

    return true;
}

// -------------------------------------------------------------------------
// CalcTrlVec: calculate translation vector t; rotation matrix u has been 
// already written inline in the same cache used to keep correlation 
// matrix r; 
// center vectors have not been overwritten and are used to calculate t;
// the resulting t overwrites the query center vector in the cache;
//
__DINLINE__
void CalcTrlVec(float *__RESTRICT__ ut/*, float scale*/)
{
    float x0 = ut[twmvCVq_0];
    float x1 = ut[twmvCVq_1];
    float x2 = ut[twmvCVq_2];

    ut[twmvCVq_0] = /*scale * */(ut[twmvCVr_0]/*yc[i]*/ -
            ut[twmvCCM_0_0]/*u[i][0]*/ * x0 -
            ut[twmvCCM_0_1]/*u[i][1]*/ * x1 -
            ut[twmvCCM_0_2]/*u[i][2]*/ * x2);

    ut[twmvCVq_1] = /*scale * */(ut[twmvCVr_1]/*yc[i]*/ -
            ut[twmvCCM_1_0]/*u[i][0]*/ * x0 -
            ut[twmvCCM_1_1]/*u[i][1]*/ * x1 -
            ut[twmvCCM_1_2]/*u[i][2]*/ * x2);

    ut[twmvCVq_2] = /*scale * */(ut[twmvCVr_2]/*yc[i]*/ -
            ut[twmvCCM_2_0]/*u[i][0]*/ * x0 -
            ut[twmvCCM_2_1]/*u[i][1]*/ * x1 -
            ut[twmvCCM_2_2]/*u[i][2]*/ * x2);
}

#endif//__transformbase_h__
