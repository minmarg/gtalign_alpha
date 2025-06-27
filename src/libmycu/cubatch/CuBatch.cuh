/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __CuBatch_h__
#define __CuBatch_h__

#include "libutil/mybase.h"

#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <memory>
#include <mutex>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "tsafety/TSCounterVar.h"

#include "libutil/CLOptions.h"
#include "libgenp/gdats/PMBatchStrData.h"
#include "libgenp/goutp/TdClustWriter.h"
#include "libgenp/goutp/TdAlnWriter.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/culayout/CuDeviceMemory.cuh"
// #include "libmycu/cusco/CuBatchSM.cuh"
// #include "libmycu/cudp/CuBatchDP.cuh"
#include "libgenp/goutp/TdFinalizer.h"

////////////////////////////////////////////////////////////////////////////
// CLASS CuBatch
// Batch computation of structure alignment using the CUDA architecture
//
class CuBatch
{
    enum {
        NBATCHSTRDATA = 2
    };

public:
    CuBatch(
        size_t ringsize,
        CuDeviceMemory* dmem, int dareano,
        TdAlnWriter* writer, TdClustWriter* clustwriter,
        size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs
    );

    virtual ~CuBatch();

    void TransferQueryPMDataToDevice(
        char** queryndxpmbeg, char** queryndxpmend,
        char** querypmbeg, char** querypmend)
    {
        dmem_->TransferQueryPMDataAndIndex(
            queryndxpmbeg, queryndxpmend, querypmbeg, querypmend);
    }

    void TransferCPMDataToDevice(
        char** bdbCndxpmbeg, char** bdbCndxpmend,
        char** bdbCpmbeg, char** bdbCpmend)
    {
        dmem_->TransferCPMDataAndIndex(
            bdbCndxpmbeg, bdbCndxpmend, bdbCpmbeg, bdbCpmend);
    }

    void TransferCPMDataToDevice(char** bdbCpmbeg, char** bdbCpmend) {
        dmem_->TransferCPMDataToDevice(bdbCpmbeg, bdbCpmend);
    }

    void ProcessBlock(
        float scorethld,
        int qrysernrbeg,
        const char** querydesc,
        char** querypmbeg,
        char** querypmend,
        const char** bdbCdesc,
        char** bdbCpmbeg,
        char** bdbCpmend,
        TSCounterVar* qrscnt,
        TSCounterVar* cnt
    );

    void CheckFinalizer() {
        int rspcode = cbpfin_->GetResponse();
        if(rspcode == CUBPTHREAD_MSG_ERROR)
            throw MYRUNTIME_ERROR("CuBatch: Finalizer terminated with errors.");
    }

    void WaitForIdleChilds() {
        if(cbpfin_)
            std::lock_guard<std::mutex> lck(cbpfin_->GetPrivateMutex());
    }

    size_t GetCurrentMaxDbPos() const { return dmem_->GetCurrentMaxDbPos(); }
    size_t GetCurrentMaxNDbStrs() const { return dmem_->GetCurrentMaxNDbStrs(); }
    unsigned int GetCurrentDbxPadding() const { return curdbxpad_; }

    size_t GetCurrentMaxDbPosPass2() const { return dmem_->GetCurrentMaxDbPosPass2(); }
    size_t GetCurrentMaxNDbStrsPass2() const { return dmem_->GetCurrentMaxNDbStrsPass2(); }
    unsigned int GetCurrentDbxPaddingPass2() const { return dbxpadphase2_; }

protected:
    void SetCurrentDbxPadding( unsigned int value ) { curdbxpad_ = value; }

    void SetCurrentDbxPaddingPass2( unsigned int value ) { dbxpadphase2_ = value; }

    size_t GetHeapSectionOffset(int sec) const {return dmem_->GetHeapSectionOffset(devareano_,sec);}
    char* GetHeapSectionAddress(int sec) const {
        return dmem_->GetHeap()+dmem_->GetHeapSectionOffset(devareano_,sec);
    }

    const std::string& GetDeviceName() const {return dmem_->GetDeviceName();}

    //Filter out flagged references
    void FilteroutReferences(
        cudaStream_t streamproc,
        const char** bdbCdesc,
        char** bdbCpmbeg,
        char** bdbCpmend,
        const size_t nqyposs, const size_t nqystrs,
        const size_t ndbCposs, const size_t ndbCstrs,
        const size_t dbstr1len, const size_t dbstrnlen,
        const size_t dbxpad, const size_t maxnsteps,
        size_t& ndbCposs2, size_t& ndbCstrs2,
        size_t& dbstr1len2, size_t& dbstrnlen2,
        size_t& dbxpad2,
        float* tmpdpdiagbuffers,
        float* tfmmemory,
        float* auxwrkmemory,
        unsigned int* globvarsbuf);

    //{{ ===== results processing =====
    void TransferResultsFromDevice(
        double tdrtn,
        float scorethld,
        int qrysernrbeg,
        size_t nqyposs, size_t nqystrs,
        size_t ndbCposs, size_t ndbCstrs,
        const char** querydesc,
        char** querypmbeg,
        char** querypmend,
        const char** bdbCdesc,
        char** bdbCpmbeg,
        char** bdbCpmend,
        TSCounterVar* qrscnt,
        TSCounterVar* cnt,
        size_t szaligns2,
        unsigned int* passedstats
    );

    void TransferResultsForClustering(
        int qrysernrbeg,
        size_t nqyposs, size_t nqystrs,
        const char** querydesc,
        char** querypmbeg,
        char** querypmend,
        const char** bdbCdesc,
        char** bdbCpmbeg,
        char** bdbCpmend,
        TSCounterVar* qrscnt,
        TSCounterVar* cnt,
        unsigned int* passedstats
    );

    void CheckHostResultsSync();

    void HostFreeResults()
    {
        MYMSG("CuBatch::HostFreeResults",4);
        if(h_results_) {
            if(lockedresmem_) {
                MYCUDACHECK(cudaFreeHost(h_results_));
                MYCUDACHECKLAST;
            }
            else
                free(h_results_);
        }
        h_results_ = NULL;
        lockedresmem_ = false;
    }

    void HostAllocResults(size_t szresults)
    {
        HostFreeResults();
        lockedresmem_ = true;
        if(cudaSuccess !=
           cudaHostAlloc((void**)&h_results_, szresults, cudaHostAllocDefault))
        {
            cudaGetLastError();
            h_results_ = NULL;
            lockedresmem_ = false;
            h_results_ = (char*)malloc(szresults);
            if(h_results_ == NULL)
                throw MYRUNTIME_ERROR(
                "CuBatch::HostAllocResults: Not enough memory.");
        }

        sz_mem_results_ = szresults;

        MYMSGBEGl(4)
            char msgbuf[BUF_MAX];
            sprintf(msgbuf, "CuBatch::HostAllocResults: sz %zu lckd %d",
                szresults, lockedresmem_);
            MYMSG( msgbuf,4 );
        MYMSGENDl
    }
    //}}


private:
    const size_t ringsize_;//#workers assigned to GPUs
    CuDeviceMemory* dmem_;//device memory configuration
    const int devareano_;//device area number
    const size_t chunkdatasize_;
    const size_t chunkdatalen_;
    const size_t chunknstrs_;
    int ndxbatchstrdata_;//current index for bdbCstruct_
    PMBatchStrData bdbCstruct_[NBATCHSTRDATA];//copy of read data when using multiple GPUs and filtering
    std::unique_ptr<unsigned int[]> filterdata_;//passed structures (globals)
    //statistics of structures passed to the next processing stages (globals):
    std::unique_ptr<unsigned int[]> passedstatscntrd_;
//     std::unique_ptr<CuBatchSM> cbsm_;//batch score matrix
//     std::unique_ptr<CuBatchDP> cbdp_;//batch dynamic programming object
    std::unique_ptr<TdFinalizer> cbpfin_;//results finalizer
    TdClustWriter* clustwriter_;//clusterer
    float scorethld_;//score threshold for pass-2 alignments

    unsigned int curdbxpad_;//current padding in positions along the x (db) structure positions
    unsigned int dbxpadphase2_;//padding in positions along the x (db) axis in phase 2
    //{{host allocated pointers for results
    cudaStream_t streamcopyres_;//stream for copying results
    char* h_results_;//results received from device
    bool lockedresmem_;//page-locked memory for results; also specifies the stream is initialized
    size_t sz_mem_results_;//size of allocated memory for results
    size_t limit_beg_results_;//boundary where the results sections begin
    //}}
};

// -------------------------------------------------------------------------
// INLINES ...
//

#endif//__CuBatch_h__
