/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __myatomic_cuh__
#define __myatomic_cuh__

// -------------------------------------------------------------------------
// atomicMinFloat: atomic min function for float; based on CUDA programming 
// guide (B14.Atomic Functions) and Bonsai github
// 
__device__ __forceinline__
float atomicMinFloat(float* address, float val)
{
    int* address_as_int = (int*)address;
    int old = __float_as_int(*address);

    while(val < __int_as_float(old))
    {
        int assumed = old;
        //NOTE: use integer comparison to avoid hang in case of NaN (since NaN != NaN)
        if((old = atomicCAS(address_as_int, assumed, __float_as_int(val))) == assumed)
            break;
    }

    return __int_as_float(old);
}

#endif//__myatomic_cuh__
