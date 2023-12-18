/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __Devices_h__
#define __Devices_h__

#include <stdio.h>

#include <string>
#include <memory>
#include <vector>

#include "libutil/mybase.h"
#include "tsafety/TSCounterVar.h"

#define MAXNDEVS 32

class Devices;

extern Devices DEVPROPs;

// -------------------------------------------------------------------------
//
struct DeviceProperties {
    enum {
        DEVMEMORYRESERVE = 256 * ONEM
    };
    DeviceProperties():
        shdcnt_(new TSCounterVar),
        devid_(-1),
        ccmajor_(0),ccminor_(0),
        totmem_(0),reqmem_(0),
        textureAlignment_(0),maxTexture1DLinear_(0),
        deviceOverlap_(0),asyncEngineCount_(0),
        computeMode_(0)
    {
        memset(maxGridSize_, 0, 3 * sizeof(int));
    };
    DeviceProperties( const DeviceProperties& dprop ):
        shdcnt_(dprop.shdcnt_),
        devid_(dprop.devid_),
        ccmajor_(dprop.ccmajor_),
        ccminor_(dprop.ccminor_),
        totmem_(dprop.totmem_),
        reqmem_(dprop.reqmem_),
        textureAlignment_(dprop.textureAlignment_),
        maxTexture1DLinear_(dprop.maxTexture1DLinear_),
        deviceOverlap_(dprop.deviceOverlap_),
        asyncEngineCount_(dprop.asyncEngineCount_),
        computeMode_(dprop.computeMode_),
        name_(dprop.name_)
    {
        memcpy(maxGridSize_, dprop.maxGridSize_, 3 * sizeof(int));
    };
    bool DevidValid() const {return devid_ < 0? false: true;}
    int GridMaxXdim() const {return maxGridSize_[0];}
    int GridMaxYdim() const {return maxGridSize_[1];}
    //shared device counter for synchronization between worker threads 
    // communicating with the same device:
    std::shared_ptr<TSCounterVar> shdcnt_;
    int devid_;//device id
    int ccmajor_;//major compute capability
    int ccminor_;//minor compute capability
    size_t totmem_;//total global memory
    size_t reqmem_;//requested amount of global memory
    size_t textureAlignment_;//alignment size for textures
    int maxTexture1DLinear_;//1D texture size
    int maxGridSize_[3];//maximumm grid size in each dimension
    int deviceOverlap_;//concurrent memory transfer and kernel execution
    int asyncEngineCount_;//number of asynchronous memory transfer engines
    int computeMode_;//enum:
    //cudaComputeModeDefault = 0
    //cudaComputeModeExclusive = 1
    //cudaComputeModeProhibited = 2
    //cudaComputeModeExclusiveProcess = 3
    std::string name_;
};

// _________________________________________________________________________
// Class Devices
//
// Implementation of queries for device properties
//
class Devices
{
public:
    Devices( int maxdvs = MAXNDEVS ): 
        maxdevs_(maxdvs),
        maxmem_(-1L)
    {};
    ~Devices() {};

    void SetMaxMemoryAmount( ssize_t mem_in_mb ) { maxmem_ = mem_in_mb * ONEM; };

    void PrintDevices( FILE* );
    void PrettyPrintUsedDevices();

    void RegisterDevices();

    int GetNDevices() const;
    const DeviceProperties* GetDevicePropertiesAt(size_t n) const;

    int GetDevIdWithMinRequestedMem() const;

protected:
    void ReadDevices();
    void SortDevices();
    bool RegisterDeviceProperties( int devid, ssize_t maxmem, bool checkduplicates );
    void PruneRegisteredDevices( int ndevs );

private:
    int maxdevs_;//maximum number of devices
    ssize_t maxmem_;//maximum memory amount for all devices
    std::vector<DeviceProperties> devs_;//list of devices
};

////////////////////////////////////////////////////////////////////////////
// INLINES
//
// -------------------------------------------------------------------------
//
inline
int Devices::GetNDevices() const
{
    MYMSG( "Devices::GetNDevices", 7 );
    return (int)devs_.size();
}

// -------------------------------------------------------------------------
//
inline
const DeviceProperties* Devices::GetDevicePropertiesAt(size_t n) const
{
    MYMSG( "Devices::GetDevicePropertiesAt", 6 );
    if( devs_.size() <= n || devs_.data() == NULL )
        return NULL;
    return devs_.data() + n;
}

#endif//__Devices_h__
