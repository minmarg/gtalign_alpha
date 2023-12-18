/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __mysort_cuh__
#define __mysort_cuh__

#include "libmycu/cucom/cucommon.h"

// =========================================================================

// -------------------------------------------------------------------------
// BatcherSortYDIMparallel: sort data in-place using Batcher's method 
// (based on Knuth's The Art of Computer Programming vol. 3, Algorithm M, p. 111);
// NOTE: version parallelized along thread y-dimension (O([logn]^2) time complexity)!
// NMAX, template parameter, max #elements the vectors can contain; (power of two);
// ASCENDING, template parameter, flag of sorting the data in ascending order;
// TPRM, template parameter, typename for the primary array;
// TSEC, template parameter, typename for the secondary array;
// n, number of values to sort;
// ioprm, primary input/output data; its values are sorted/compared (keys);
// iosec, secondary input/output vector; its values are rearranged consistently with ioprm;
template<int NMAX, bool ASCENDING, typename TPRM, typename TSEC>
__device__ __forceinline__
void BatcherSortYDIMparallel(
    const int n, TPRM* __restrict__ ioprm, TSEC* __restrict__ iosec)
{
    int tmax1 = 32 - __clz(NMAX) - 1;//ceilf(log2f(.)) - 1
    int tm1 = 32 - __clz(n) - 1;//ceilf(log2f(n)) - 1

    if(tmax1 < 0) tmax1 = 0;
    if(tm1 < 0) tm1 = 0;

    for(int p = (1 << tm1), pmax = (1 << tmax1); pmax > 0;)
    {
        for(int qmax = (1 << tmax1), dmax = pmax,
            q = (1 << tm1), d = p, r = 0;//q=2^(t-1)
                dmax > 0;//same as do{}while(q>p)
            )
        {
            for(int i = threadIdx.y; i < NMAX - dmax; i += blockDim.y) {
                //NOTE: dmax==d always ensures a sufficient number of i-iterations
                if(pmax <= p && dmax == d && i < n - d && (i & p) == r) {
                    if(ASCENDING? (ioprm[i+dmax] < ioprm[i]): (ioprm[i] < ioprm[i+dmax]))
                    {
                        //exchange values of primary and secondary arrays:
                        myhdswap(ioprm[i], ioprm[i+dmax]);
                        myhdswap(iosec[i], iosec[i+dmax]);
                    }
                }
            }
            __syncthreads();

            if(dmax == d) { d = q - p; q >>= 1; }
            dmax = qmax - pmax; qmax >>= 1;
            r = p;
        }

        if(pmax <= p) p >>= 1;
        pmax >>= 1;
    }
}

// =========================================================================

#endif//__mysort_cuh__
