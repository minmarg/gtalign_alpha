/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/mybase.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstdlib>

#include "libmympbase/mplayout/CuMemoryBase.h"
#include "MpGlobalMemory.h"

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// constructor
//
MpGlobalMemory::MpGlobalMemory(
    size_t deviceallocsize,
    int nareas)
:
    CuMemoryBase(deviceallocsize, nareas),
    devname_("n/a"),
    g_heap_(NULL)
{
    MYMSG("MpGlobalMemory::MpGlobalMemory", 4);
    Initialize();
}

// -------------------------------------------------------------------------
// destructor
//
MpGlobalMemory::~MpGlobalMemory()
{
    MYMSG("MpGlobalMemory::~MpGlobalMemory", 4);
}





// =========================================================================
// AllocateHeap: allocate device memory
inline
void MpGlobalMemory::AllocateHeap()
{
    MYMSG("MpGlobalMemory::AllocateHeap", 6);
    g_heap_ = (char*)my_aligned_alloc(GetMemAlignment(), GetAllocSize());
    if(g_heap_ == NULL)
        throw MYRUNTIME_ERROR(
        "MpGlobalMemory::AllocateHeap: Not enough memory.");
}

// -------------------------------------------------------------------------
// FreeDevicePtr: free device pointer
inline
void MpGlobalMemory::DeallocateHeap()
{
    MYMSG("MpGlobalMemory::DeallocateHeap", 6);
    if(g_heap_) {
        my_aligned_free(g_heap_);
        g_heap_ = NULL;
    }
}
