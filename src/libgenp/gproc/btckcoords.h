/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __btckcoords_h__
#define __btckcoords_h__

// invalid DP cell coordinates (positions; as a result of CombineCoords);
// or Stop marker:
#define INVALIDDPCELLCOORDS 0xffffffff
#define DPCELLSTOP 0xffffffff

// -------------------------------------------------------------------------
// CombineCoords: combine (x,y) coordinates (structure pairwise positions) 
// into one integer value;
// NOTE: the arguments x and y are supposed to contain 16-bit values!
#ifdef GPUINUSE
__host__ __device__ __forceinline__
#else
inline
#endif
unsigned int CombineCoords(unsigned int x, unsigned int y)
{
    return (x << 16) | y;//(y & 0xffff);
}

// GetCoordX and GetCoordY extract x and y coordinates from the 
// combined value
#ifdef GPUINUSE
__host__ __device__ __forceinline__
#else
inline
#endif
unsigned int GetCoordX(unsigned int xy)
{
    return (xy >> 16) & 0xffff;
}

#ifdef GPUINUSE
__host__ __device__ __forceinline__
#else
inline
#endif
unsigned int GetCoordY(unsigned int xy)
{
    return xy & 0xffff;
}

#endif//__btckcoords_h__
