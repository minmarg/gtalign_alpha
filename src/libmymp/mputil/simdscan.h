/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __simdscan_h__
#define __simdscan_h__

#include "libutil/templates.h"

// -------------------------------------------------------------------------
// // mysimdincprefixsum: inplace inclusive prefix sum using simd instructions;
// // LEVEL, initially is DIM;
// // DIM, power of 2 necessarily;
// // data, data of size DIM;
// // tmp, temporary array of dimension DIM;
// template <int LEVEL, int DIM, typename T>
// inline
// void mysimdincprefixsum(T* const data, T* const tmp)
// {
//     mysimdincprefixsum<(LEVEL>>1),DIM>(data, tmp);
//     constexpr int pibeg = LEVEL;
//     #pragma omp simd
//     for(int pi = pibeg; pi < DIM; pi++)
//         tmp[pi] = data[pi - pibeg];
//     #pragma omp simd
//     for(int pi = pibeg; pi < DIM; pi++)
//         data[pi] += tmp[pi];
// }

// template <> inline void mysimdincprefixsum<0,0>(float* const, float* const) {}
// template <> inline void mysimdincprefixsum<0,1>(float* const, float* const) {}
// template <> inline void mysimdincprefixsum<0,2>(float* const, float* const) {}
// template <> inline void mysimdincprefixsum<0,4>(float* const, float* const) {}
// template <> inline void mysimdincprefixsum<0,8>(float* const, float* const) {}
// template <> inline void mysimdincprefixsum<0,16>(float* const, float* const) {}
// template <> inline void mysimdincprefixsum<0,32>(float* const, float* const) {}
// template <> inline void mysimdincprefixsum<0,64>(float* const, float* const) {}
// template <> inline void mysimdincprefixsum<0,128>(float* const, float* const) {}
// template <> inline void mysimdincprefixsum<0,256>(float* const, float* const) {}
// template <> inline void mysimdincprefixsum<0,512>(float* const, float* const) {}
// template <> inline void mysimdincprefixsum<0,1024>(float* const, float* const) {}

// template <> inline void mysimdincprefixsum<0,0>(int* const, int* const) {}
// template <> inline void mysimdincprefixsum<0,1>(int* const, int* const) {}
// template <> inline void mysimdincprefixsum<0,2>(int* const, int* const) {}
// template <> inline void mysimdincprefixsum<0,4>(int* const, int* const) {}
// template <> inline void mysimdincprefixsum<0,8>(int* const, int* const) {}
// template <> inline void mysimdincprefixsum<0,16>(int* const, int* const) {}
// template <> inline void mysimdincprefixsum<0,32>(int* const, int* const) {}
// template <> inline void mysimdincprefixsum<0,64>(int* const, int* const) {}
// template <> inline void mysimdincprefixsum<0,128>(int* const, int* const) {}
// template <> inline void mysimdincprefixsum<0,256>(int* const, int* const) {}
// template <> inline void mysimdincprefixsum<0,512>(int* const, int* const) {}
// template <> inline void mysimdincprefixsum<0,1024>(int* const, int* const) {}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// mysimdincprefixsum: inplace inclusive prefix sum using simd instructions;
// LEVEL, power of 2 greater/equal (>=) than DIM;
// DIM, data cardinality;
// data, data of size DIM;
// tmp, temporary array of dimension DIM;
template <int LEVEL, typename T>
inline
void mysimdincprefixsum(const int DIM, T* const data, T* const tmp)
{
    mysimdincprefixsum<(LEVEL>>1)>(DIM, data, tmp);
    constexpr int pibeg = LEVEL;
    #pragma omp simd
    for(int pi = pibeg; pi < DIM; pi++)
        tmp[pi] = data[pi - pibeg];
    #pragma omp simd
    for(int pi = pibeg; pi < DIM; pi++)
        data[pi] += tmp[pi];
}

template <> inline void mysimdincprefixsum<0>(const int, float* const, float* const) {}
template <> inline void mysimdincprefixsum<0>(const int, int* const, int* const) {}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
#define PFX_MIN 0 //template argument value for MIN
#define PFX_MAX 1 //template argument value for MAX
// -------------------------------------------------------------------------
// mysimdincprefixminmax: inplace inclusive prefix min/max using simd instructions;
// LEVEL, power of 2 greater/equal (>=) than DIM;
// DIM, data cardinality;
// data, prefix min/max data of size DIM;
// tmp, temporary array of dimension DIM;
template <int LEVEL, int MINMAX, typename T>
inline
void mysimdincprefixminmax(const int DIM, T* const data, T* const tmp)
{
    mysimdincprefixminmax<(LEVEL>>1),MINMAX>(DIM, data, tmp);
    constexpr int pibeg = LEVEL;
    #pragma omp simd
    for(int pi = pibeg; pi < DIM; pi++)
        tmp[pi] = data[pi - pibeg];
    #pragma omp simd
    for(int pi = pibeg; pi < DIM; pi++) {
        if(MINMAX == PFX_MIN) data[pi] = mymin(data[pi], tmp[pi]);
        else data[pi] = mymax(data[pi], tmp[pi]);
    }
}

template <> inline void mysimdincprefixminmax<0,PFX_MIN>(const int, float* const, float* const) {}
template <> inline void mysimdincprefixminmax<0,PFX_MAX>(const int, float* const, float* const) {}

// -------------------------------------------------------------------------
// mysimdincprefixmaxndx: inplace inclusive prefix max with index tracking
// using simd instructions;
// LEVEL, power of 2 greater/equal (>=) than DIM;
// DIM, data cardinality;
// data, prefix max data of size DIM;
// index, index array of size DIM;
// tmp, temporary array of dimension DIM;
// tmpndx, temporary index array of dimension DIM;
// NOTE: index should be pre-intiliazed with sequential values;
template <int LEVEL, typename T>
inline
void mysimdincprefixmaxndx(const int DIM, T* const data, int* const index, T* const tmp, int* const tmpndx)
{
    mysimdincprefixmaxndx<(LEVEL>>1)>(DIM, data, index, tmp, tmpndx);
    constexpr int pibeg = LEVEL;
    #pragma omp simd
    for(int pi = pibeg; pi < DIM; pi++) {
        tmp[pi] = data[pi - pibeg];
        tmpndx[pi] = index[pi - pibeg];
    }
    #pragma omp simd
    for(int pi = pibeg; pi < DIM; pi++)
        //NOTE <= to properly force index passing:
        if(data[pi] <= tmp[pi]) {data[pi] = tmp[pi]; index[pi] = tmpndx[pi];}
        // mymaxassgn(data[pi], tmp[pi], index[pi], tmpndx[pi]);
}

template <> inline void mysimdincprefixmaxndx<0>(const int, float* const, int* const, float* const, int* const) {}

// -------------------------------------------------------------------------

#endif//__simdscan_h__
