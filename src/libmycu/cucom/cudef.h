/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __cudef_h__
#define __cudef_h__

#ifdef GPUINUSE
#   include <cuda.h>
#   include <cuda_runtime_api.h>
#   define __HDINLINE__ __host__ __device__ __forceinline__
#   define __DINLINE__ __device__ __forceinline__
#   define __RESTRICT__ __restrict__
#   define d8powf __powf
#   define tfmsincosf(d, sth, cth) __sincosf(d, &sth, &cth)
#   define tfmrsqrtf(d) rsqrtf(d)
#   define tfmfdividef(a, p) fdividef(a, p)
#   define scrfdividef(a, p) __fdividef(a, p)
    // CUDA L2 Cache line size
#   define CUL2CLINESIZE 128
#else
#   define __HDINLINE__ inline
#   define __DINLINE__ inline
#   define __RESTRICT__ __restrict
#   define d8powf powf
#   define tfmsincosf(d, sth, cth) sth = sinf(d); cth = cosf(d);
#   define tfmrsqrtf(d) 1.0f/sqrtf(d)
#   define tfmfdividef(a, p) ((a) / (p))
#   define scrfdividef(a, p) ((a) / (p))
#   define CUL2CLINESIZE 128
#endif

#endif//__cudef_h__
