/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __DRDataDeleter_h__
#define __DRDataDeleter_h__

#include "libutil/mybase.h"

#include <memory>

#if defined(GPUINUSE) && 0
#   include <cuda_runtime_api.h>
#endif

#include "libutil/CLOptions.h"

// -------------------------------------------------------------------------
//
struct DRDataDeleter {
    void operator()(void* p) const {
        if(p)
            std::free(p);
    };
};

struct DRHostDataDeleter {
    void operator()(void* p) const {
        if(p) {
#if defined(GPUINUSE) && 0
            if(CLOptions::GetIO_UNPINNED() == 0)
                cudaFreeHost(p);
            else
#endif
                // std::free(p);
                my_aligned_free(p);
        }
    };
};

// -------------------------------------------------------------------------

#endif//__DRDataDeleter_h__
