/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __CuDeviceMemory_h__
#define __CuDeviceMemory_h__

#include "libutil/mybase.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "libutil/CLOptions.h"
#include "libmympbase/mplayout/CuMemoryBase.h"
#include "libmycu/cuproc/Devices.h"

////////////////////////////////////////////////////////////////////////////
// CLASS CuDeviceMemory
// Device memory arrangement for structure search and alignment computation
//
class CuDeviceMemory: public CuMemoryBase
{
public:
    CuDeviceMemory(
        DeviceProperties dprop, 
        size_t deviceallocsize,
        int nareas );

    virtual ~CuDeviceMemory();

    void CacheCompleteData();

    const DeviceProperties& GetDeviceProp() const { return deviceProp_;}
    const std::string& GetDeviceName() const { return deviceProp_.name_;}

    virtual char* GetHeap() const {return d_heap_;}

    virtual size_t GetMemAlignment() const {
        size_t cszalnment = 256UL;
        cszalnment = PCMAX(cszalnment, deviceProp_.textureAlignment_);
        return cszalnment;
    }

    void TransferQueryPMDataAndIndex(
        char** queryndxpmbeg, char** queryndxpmend,
        char** querypmbeg, char** querypmend)
    {
        TransferQueryPMDataToDevice(querypmbeg, querypmend);
        TransferQueryPMIndexToDevice(queryndxpmbeg, queryndxpmend);
    }

    void TransferCPMDataAndIndex(
        char** bdbCndxpmbeg, char** bdbCndxpmend,
        char** bdbCpmbeg, char** bdbCpmend)
    {
        TransferCPMDataToDevice(bdbCpmbeg, bdbCpmend);
        TransferCPMIndexToDevice(bdbCndxpmbeg, bdbCndxpmend);
    }

    void TransferQueryPMDataToDevice(char** querypmbeg, char** querypmend);
    void TransferQueryPMIndexToDevice(char** queryndxpmbeg, char** queryndxpmend);
    void TransferCPMDataToDevice(char** bdbCpmbeg, char** bdbCpmend);
    void TransferCPMIndexToDevice(char** bdbCndxpmbeg, char** bdbCndxpmend);

protected:

    virtual void AllocateHeap();
    virtual void DeallocateHeap() {FreeDevicePtr(d_heap_);}

    template <size_t NFIELDS>
    void CopyCPMDataToDevice(
        char** bdbCpmbeg,
        char** bdbCpmend,
        char* dev_pckdpm,
        const size_t szmaxsize,
        const size_t cmbegndx,
        const char* sinfo = ""
    );

    void FreeDevicePtr(char*& d_ptr);

private:
    //{{device data
    DeviceProperties deviceProp_;//the properties of device deviceno_
    char* d_heap_;//global heap containing all data written, generated, and read
    //}}
};

// -------------------------------------------------------------------------
// INLINES ...
//

#endif//__CuDeviceMemory_h__
