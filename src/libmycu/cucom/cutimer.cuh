/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __cutimer_cuh__
#define __cutimer_cuh__

#include "libutil/mybase.h"
#include "libmycu/cucom/cucommon.h"

////////////////////////////////////////////////////////////////////////////
// CLASS Cuda Timer for measuring kernel performance
//
class MyCuTimer {
public:
    MyCuTimer(): start_(NULL), stop_(NULL), elapsed_(0.0f) {}

    //destroy graph instance
    ~MyCuTimer() {Destroy();}

    //start measuring performance
    void Start(cudaStream_t stream) {
        if(start_ == NULL) CreateEvent(start_);
        if(stop_ == NULL) CreateEvent(stop_);
        MYCUDACHECK(cudaEventRecord(start_, stream));
        MYCUDACHECKLAST;
    }

    //stop the timer; measure the elapsed time
    void Stop(cudaStream_t stream) {
        if(start_ == NULL || stop_ == NULL) 
            throw MYRUNTIME_ERROR("MyCuTimer: Stop: Null event(s).");
        MYCUDACHECK(cudaEventRecord(stop_, stream));
        MYCUDACHECKLAST;
        MYCUDACHECK(cudaEventSynchronize(stop_));
        MYCUDACHECKLAST;
        MYCUDACHECK(cudaEventElapsedTime(&elapsed_, start_, stop_));
        MYCUDACHECKLAST;
    }

    float GetElapsedTime() const {return elapsed_;}

protected:
    //create event
    void CreateEvent(cudaEvent_t& ev) {
        DestroyEvent(ev);
        MYCUDACHECK(cudaEventCreate(&ev));
        MYCUDACHECKLAST;
    }

    //destroy event
    void DestroyEvent(cudaEvent_t& ev) {
        if(ev) {
            MYCUDACHECKPASS(cudaEventDestroy(ev));
            ev = NULL;
        }
    }

    //destroy the events
    void Destroy() {
        DestroyEvent(start_);
        DestroyEvent(stop_);
        elapsed_ = 0.0f;
    }

private:
    cudaEvent_t start_, stop_;
    float elapsed_;//time elapsed in ms
};

#endif//__cutimer_cuh__
