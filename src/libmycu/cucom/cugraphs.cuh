/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __cugraphs_cuh__
#define __cugraphs_cuh__

#include "libutil/mybase.h"
#include "libmycu/cucom/cucommon.h"

// -------------------------------------------------------------------------
// struct CGKey: key for associating it with a MyCuGraph graph object
//
struct CGKey {
    CGKey(): fragbydp_(0), nmaxconvit_(0), sfragndx_(0) {};
    CGKey(int fbydp, int maxncit, int sfragndx)
    : fragbydp_(fbydp), nmaxconvit_(maxncit), sfragndx_(sfragndx) {};
    int fragbydp_;//DP-btained fragment
    int nmaxconvit_;//max #iterations until convergence
    int sfragndx_;//index for subfragment length
};

inline
bool operator<(const CGKey& left, const CGKey& right)
{
    return 
        (left.fragbydp_ != right.fragbydp_)
        ? left.fragbydp_ < right.fragbydp_
        : ((left.nmaxconvit_ != right.nmaxconvit_)
            ? left.nmaxconvit_ < right.nmaxconvit_
            : left.sfragndx_ < right.sfragndx_
          )
    ;
}

////////////////////////////////////////////////////////////////////////////
// CLASS Cuda Graph for creating, instantiating, and using Cuda graphs
//
class MyCuGraph {
public:
    MyCuGraph(): instance_(NULL) {}

    //destroy graph instance
    ~MyCuGraph() {Destroy();}

    //whether a graph is captured
    bool IsCaptured() {return instance_ != NULL;}

    //destroy before instantiating a new graph
    void Destroy() {
        if(instance_) {
            MYCUDACHECKPASS(cudaGraphExecDestroy(instance_));
            instance_ = NULL;
        }
    }

    //start capturing kernels issued in the stream
    void BeginCapture(cudaStream_t mystream) {
#if (CUDART_VERSION <= 10000)
        MYCUDACHECK(cudaStreamBeginCapture(mystream));
#elif (CUDART_VERSION > 10000)
        MYCUDACHECK(cudaStreamBeginCapture(mystream,cudaStreamCaptureModeGlobal));
#else
#   error CUDART_VERSION Undefined.
#endif
        MYCUDACHECKLAST;
    }

    //end capture and instantiate a graph
    void EndCaptureInstantiate(cudaStream_t mystream) {
        char errbuf[BUF_MAX];
        cudaGraph_t graph;
        MYCUDACHECK(cudaStreamEndCapture(mystream, &graph));
        MYCUDACHECKLAST;
        Destroy();
        MYCUDACHECK(cudaGraphInstantiate(&instance_, graph, NULL, errbuf, BUF_MAX-1));
        MYCUDACHECKLAST;
        MYCUDACHECKPASS(cudaGraphDestroy(graph));
    }

    //launch instantiated graph
    void Launch(cudaStream_t mystream) {
        if(instance_ == NULL)
            throw MYRUNTIME_ERROR("MyCuGraph: Launch: Null CUDA graph.");
        MYCUDACHECK(cudaGraphLaunch(instance_, mystream));
        MYCUDACHECKLAST;
    }

private:
    cudaGraphExec_t instance_;
};

#endif//__cugraphs_cuh__
