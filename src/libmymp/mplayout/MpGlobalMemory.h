/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __MpGlobalMemory_h__
#define __MpGlobalMemory_h__

#include "libutil/mybase.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "libutil/CLOptions.h"
#include "libmympbase/mplayout/CuMemoryBase.h"


////////////////////////////////////////////////////////////////////////////
// CLASS MpGlobalMemory
// global memory arrangement for structure search and alignment computation
//
class MpGlobalMemory: public CuMemoryBase
{
public:
    MpGlobalMemory(
        size_t deviceallocsize,
        int nareas );

    virtual ~MpGlobalMemory();

    const std::string& GetDeviceName() const { return devname_;}

    virtual char* GetHeap() const {return g_heap_;}

    virtual size_t GetMemAlignment() const {
        // return CuMemoryBase::GetMemAlignment();
        size_t cszalnment = 512UL;
        return cszalnment;
    }

protected:

    virtual void AllocateHeap();
    virtual void DeallocateHeap();

private:
    std::string devname_;
    char* g_heap_;//global heap containing all data written, generated, and read
};

// -------------------------------------------------------------------------
// INLINES ...
//

#endif//__MpGlobalMemory_h__
