/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <algorithm>

#include <cuda_runtime_api.h>

#include "libutil/mybase.h"
#include "libutil/CLOptions.h"
#include "libmycu/cucom/cucommon.h"

#include "Devices.h"

Devices DEVPROPs;

// _________________________________________________________________________
// Class DeviceProperties
//
void Devices::PrintDevices( FILE* fp )
{
    MYMSG( "Devices::PrintDevices", 6 );

    int deviceCount = 0;
    MYCUDACHECK( cudaGetDeviceCount( &deviceCount ));

    if( deviceCount < 1 ) {
        fprintf( fp, "%sThere are no available CUDA-enabled device(s).%s", NL,NL);
        return;
    }

    fprintf( fp, "%s%d CUDA-capable device(s) detected:%s%s", NL, deviceCount, NL,NL);

    for( int dev = 0; dev < deviceCount; dev++ ) {
        MYCUDACHECK( cudaSetDevice( dev ));
        cudaDeviceProp deviceProp;
        size_t freemem, totalmem;
        MYCUDACHECK( cudaGetDeviceProperties( &deviceProp, dev ));
        MYCUDACHECK( cudaMemGetInfo ( &freemem, &totalmem ));
        fprintf( fp, "Device id %d: \"%s\"%s (free: %.1fGB total: %.1fGB)%s", 
            dev, deviceProp.name,
            (deviceProp.computeMode == cudaComputeModeProhibited)? " [Prohibited!]": "",
            (float)freemem/(float)ONEG, (float)totalmem/(float)ONEG,
            NL);
    }

    fprintf( fp, "%s",NL);
}

// -------------------------------------------------------------------------
// RegisterDevices: register devices scheduled for use
//
void Devices::RegisterDevices()
{
    MYMSG( "Devices::RegisterDevices", 6 );

    SetMaxMemoryAmount( CLOptions::GetDEV_MEM());

    std::string devN = CLOptions::GetDEV_N();
    size_t pos;
    char* p = NULL;
    long int did, ndevs = -1;

    //parse command line option here to get the list of devices
    if( !devN.empty() && devN[0] == ',') {
        devs_.clear();
        devN = devN.substr(1);
        //device ids are listed
        for( pos = devN.find(','); ; pos = devN.find(',')) {
            std::string strval = devN.substr( 0, pos );
            errno = 0;
            did = strtol( strval.c_str(), &p, 10 );
            if( errno || *p || strval.empty())
                throw MYRUNTIME_ERROR("Invalid ID specified by option dev-N.");
            if( did < 0 || 1000000 < did )
                throw MYRUNTIME_ERROR("Invalid ID specified by command-line option dev-N.");
            RegisterDeviceProperties( did, maxmem_, true/*checkduplicates*/);
            if( pos == std::string::npos )
                break;
            devN = devN.substr(pos+1);
        }
    }
    else {
        //certain number of devices is given
        errno = 0;
        ndevs = strtol( devN.c_str(), &p, 10 );
        if(errno || *p)
            throw MYRUNTIME_ERROR("Invalid number of devices given: "
            "Begin option dev-N with ',' if a list of IDs is intended.");
        if(ndevs <= 0 || 100 < ndevs)
            throw MYRUNTIME_ERROR("Invalid number of devices given by option dev-N (max 100).");
        // if(1 < ndevs)
        //     throw MYRUNTIME_ERROR("Multiple GPUs option (dev-N>1) not implemented yet.");
        ReadDevices();
        SortDevices();
    }

    PruneRegisteredDevices( ndevs );

    PrettyPrintUsedDevices();

    //NOTE:reset errno as it may have been previously set by CUDA calls!
    errno = 0;
}

// -------------------------------------------------------------------------
// ReadDevices: read all CUDA-capable devices available on the system
//
void Devices::ReadDevices()
{
    MYMSG( "Devices::ReadDevices", 6 );

    int deviceCount = 0;
    MYCUDACHECK( cudaGetDeviceCount( &deviceCount ));

    if( deviceCount < 1 )
        return;

    devs_.clear();

    for( int dev = 0; dev < deviceCount; dev++ ) {
        RegisterDeviceProperties( dev, maxmem_, false/*checkduplicates*/);
    }
}

// -------------------------------------------------------------------------
// GetDevIdWithMinRequestedMem: get id of a device with minimum requested 
// memory
//
int Devices::GetDevIdWithMinRequestedMem() const
{
    MYMSG( "Devices::GetDevIdWithMinRequestedMem", 7 );
    int did = -1, i = 0;
    size_t minreqmem = (size_t)-1;
    for( auto devit = devs_.begin(); devit != devs_.end(); devit++, i++ ) {
        if( devit->reqmem_ < minreqmem ) {
            minreqmem = devit->reqmem_;
            did = i;
        }
    }
    return did;
}

// -------------------------------------------------------------------------
// SortDevices: sort CUDA-capable devices that have been saved in the list
//
void Devices::SortDevices()
{
    MYMSG( "Devices::SortDevices", 6 );
    std::sort(devs_.begin(), devs_.end(),
        [](const DeviceProperties& dp1, const DeviceProperties& dp2) {
            //[dp2]-[dp1], to sort in descending order
            return 
                (dp2.ccmajor_ == dp1.ccmajor_)
                ?   ((dp2.ccminor_ == dp1.ccminor_)
                    ?  ((dp2.reqmem_>>20) < (dp1.reqmem_>>20)) //NOTE:prefer reqmem_ to totmem_
                    :  (dp2.ccminor_ < dp1.ccminor_))
                :   (dp2.ccmajor_ < dp1.ccmajor_)
            ;
        });
}

// -------------------------------------------------------------------------
// RegisterDeviceProperties: register the properties of device with id devid
//
bool Devices::RegisterDeviceProperties( int devid, ssize_t maxmem, bool checkduplicates )
{
    MYMSG( "Devices::RegisterDeviceProperties", 6 );

    if( checkduplicates ) {
        //if checking for duplicate ids is on, 
        // ignore devices that have been already registered (same id)
        auto dit = std::find_if(devs_.begin(), devs_.end(),
            [devid](const DeviceProperties& dp){
                return dp.devid_ == devid;
            });
        if(dit != devs_.end())
            return false;//device with devid found
    }

    cudaDeviceProp deviceProp;
    size_t freemem, totalmem;
    //if false, the device cannot be queried; a possible reason is occupied memory
    if(!MYCUDACHECKPASS( cudaSetDevice( devid ))) return false;
    if(!MYCUDACHECKPASS( cudaGetDeviceProperties( &deviceProp, devid ))) return false;
    if(!MYCUDACHECKPASS( cudaMemGetInfo ( &freemem, &totalmem ))) return false;
    //
    if( deviceProp.computeMode == cudaComputeModeProhibited )
        return false;
    //
    DeviceProperties dprop;//to be registered
    dprop.devid_ = devid;
    dprop.ccmajor_ = deviceProp.major;
    dprop.ccminor_ = deviceProp.minor;
    dprop.totmem_ = deviceProp.totalGlobalMem;
    //NOTE: better memory size determination:
    if( freemem <= DeviceProperties::DEVMEMORYRESERVE )
        throw MYRUNTIME_ERROR("Devices::GetDeviceProperties: Not enough memory.");
    dprop.reqmem_ = freemem - DeviceProperties::DEVMEMORYRESERVE;
    if( 0 < maxmem ) {
        if( maxmem <= (ssize_t)dprop.reqmem_ )
            dprop.reqmem_ = maxmem;
        else {
            char msgbuf[KBYTE];
            sprintf(msgbuf,"Device %d: free memory < requested amount: %zu(MB) < %zd(MB).",
                dprop.devid_, dprop.reqmem_>>20, maxmem>>20);
            warning(msgbuf);
        }
    }
//     //{{NOTE: previous
//     dprop.reqmem_ = dprop.totmem_;
//     if( 0 < maxmem && maxmem < (ssize_t)dprop.totmem_ )
//         dprop.reqmem_ = maxmem;
//     if( dprop.totmem_>>20 >= 16000UL ) {
//         if( dprop.reqmem_ + DeviceProperties::DEVMINMEMORYRESERVE_16G > dprop.totmem_)
//             dprop.reqmem_ = dprop.totmem_ - DeviceProperties::DEVMINMEMORYRESERVE_16G;
//     }
//     else {
//         if( dprop.reqmem_ + DeviceProperties::DEVMINMEMORYRESERVE > dprop.totmem_)
//             dprop.reqmem_ = (dprop.totmem_ <= DeviceProperties::DEVMINMEMORYRESERVE)?
//                 dprop.totmem_: dprop.totmem_ - DeviceProperties::DEVMINMEMORYRESERVE;
//     }
//     //}}
    dprop.textureAlignment_ = deviceProp.textureAlignment;
    dprop.maxTexture1DLinear_ = deviceProp.maxTexture1DLinear;
    dprop.maxGridSize_[0] = deviceProp.maxGridSize[0];
    dprop.maxGridSize_[1] = deviceProp.maxGridSize[1];
    dprop.maxGridSize_[2] = deviceProp.maxGridSize[2];
    dprop.deviceOverlap_ = deviceProp.deviceOverlap;
    dprop.asyncEngineCount_ = deviceProp.asyncEngineCount;
    dprop.computeMode_ = deviceProp.computeMode;
    dprop.name_ = deviceProp.name;
    //
    MYCUDACHECK( cudaDeviceReset());
    //
    devs_.push_back(dprop);
    return true;
}

// -------------------------------------------------------------------------
// PruneRegisteredDevices: prune registered devices so that their number 
// does not exceed the allowed number
//
void Devices::PruneRegisteredDevices( int ndevs )
{
    MYMSG( "Devices::PruneRegisteredDevices", 6 );

    if( ndevs < 0 )
        ndevs = maxdevs_;

    if((int)devs_.size() <= ndevs )
        return;

    devs_.erase(devs_.begin() + ndevs, devs_.end());
}

// -------------------------------------------------------------------------
// PrintSavedDevicesTest: print devices that are currently in the list
//
void Devices::PrettyPrintUsedDevices()
{
    MYMSGBEGl(1)
        char msgbuf[KBYTE];
        if( devs_.size())
            MYMSG("Devices scheduled for use:",1);
        for( auto dit = devs_.begin(); dit != devs_.end(); dit++ ) {
            sprintf( msgbuf, 
                "Device id %d: Compute_capability %d.%d Tot_Mem %zu(MB) Used_Mem %zu(MB) "
                " TextAln %zu Text1D %d(MB) Ovlp %d Engs %d Mode %d", 
                dit->devid_, dit->ccmajor_, dit->ccminor_, dit->totmem_>>20, dit->reqmem_>>20,
                dit->textureAlignment_, dit->maxTexture1DLinear_>>20, dit->deviceOverlap_,
                dit->asyncEngineCount_, dit->computeMode_ );
            MYMSG(msgbuf,1);
        }
    MYMSGENDl
}
