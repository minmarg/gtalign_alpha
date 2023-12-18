/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __mymemory_h__
#define __mymemory_h__

#include <stdio.h>
#include <cstdlib>
#include "platform.h"

#ifdef OS_MS_WINDOWS
//memory allocation
inline void* my_aligned_alloc(size_t alignment, size_t size)
{
    //NOTE: arguments order
    return _aligned_malloc(size, alignment);
}
//memory deallocation
inline void my_aligned_free(void* memptr)
{
    return _aligned_free(memptr);
}
#else
//memory allocation
inline void* my_aligned_alloc(size_t alignment, size_t size)
{
    return aligned_alloc(alignment, size);
}
//memory deallocation
inline void my_aligned_free(void* memptr)
{
    return free(memptr);
}
#endif

#endif//__mymemory_h__
