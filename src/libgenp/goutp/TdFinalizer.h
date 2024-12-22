/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __TdFinalizer_h__
#define __TdFinalizer_h__

#include "libutil/mybase.h"

#include <stdio.h>

#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>

#ifdef GPUINUSE
#   include <cuda.h>
#   include <cuda_runtime_api.h>
#   include "libmycu/cuproc/Devices.h"
#endif

#include "tsafety/TSCounterVar.h"
#include "libgenp/gproc/gproc.h"

#include "libgenp/gdats/PM2DVectorFields.h"
// #include "libgenp/gdats/PMBatchStrData.h"
#include "libgenp/goutp/TdAlnWriter.h"
#include "libmycu/cuproc/cuprocconf.h"

#define CUBPTHREAD_MSG_UNSET -1
#define CUBPTHREAD_MSG_ERROR -2

// _________________________________________________________________________
// Class TdFinalizer
//
// thread class for finalizing results calculated on a device
//
class TdFinalizer
{
public:
    enum TCUBPThreadMsg {
        cubpthreadmsgFinalize,
        cubpthreadmsgTerminate
    };
    enum TCUBPThreadResponse {
        cubptrespmsgFinalizing,
        cubptrespmsgTerminating
    };

public:
#ifdef GPUINUSE
    TdFinalizer(
        cudaStream_t& strcopyres,
        DeviceProperties dprop, 
        TdAlnWriter*
    );
#endif

    TdFinalizer(
        TdAlnWriter*
    );

    ~TdFinalizer();

    std::mutex& GetPrivateMutex() {return mx_dataccess_;}

    //{{NOTE: messaging functions accessed from outside!
    void Notify(int msg) {
        {//mutex must be unlocked before notifying
            std::unique_lock<std::mutex> lck(mx_dataccess_);
            if( req_msg_ != CUBPTHREAD_MSG_ERROR)
                req_msg_ = msg;
        }
        cv_msg_.notify_one();
    }
    void waitForDataAccess() {
        std::unique_lock<std::mutex> lck(mx_dataccess_);
        cv_msg_.wait(lck,
            [this] {return
                rsp_msg_ == cubptrespmsgTerminating || 
                rsp_msg_ == CUBPTHREAD_MSG_ERROR ||
               (rsp_msg_ == CUBPTHREAD_MSG_UNSET && req_msg_ == CUBPTHREAD_MSG_UNSET);
            }
        );
    }
    int GetResponse() const {
        std::lock_guard<std::mutex> lck(mx_dataccess_);
        return rsp_msg_;
    }
    void ResetResponse() {
        std::lock_guard<std::mutex> lck(mx_dataccess_);
        if( rsp_msg_!= CUBPTHREAD_MSG_ERROR )
            rsp_msg_ = CUBPTHREAD_MSG_UNSET;
    }
    //}}


    void SetCuBPBDbdata(
        double tdrtn,
        float scorethld,
        int qrysernrbeg,
        const std::string& devname,
        size_t nqyposs, size_t nqystrs,
        size_t ndbCposs, size_t ndbCstrs,
        const char** querydesc, char** querypmbeg, char** querypmend, 
        const char** bdbCdesc, char** bdbCpmbeg, char** bdbCpmend,
        TSCounterVar* qrscnt,
        TSCounterVar* cnt,
        unsigned int* passedstats,
        const char* h_results,
        size_t sz_alndata,
        size_t sz_tfmmatrices,
        size_t sz_alns,
        unsigned int dbalnlen2)
    {
        std::unique_lock<std::mutex> lck(mx_dataccess_);
        cv_msg_.wait(lck,
            [this]{
                return (req_msg_ == CUBPTHREAD_MSG_UNSET ||
                        req_msg_ == CUBPTHREAD_MSG_ERROR ||
                        rsp_msg_ == CUBPTHREAD_MSG_ERROR);
            }
        );
        if(req_msg_ == CUBPTHREAD_MSG_ERROR || rsp_msg_ == CUBPTHREAD_MSG_ERROR)
            return;
        cubp_set_duration_ = tdrtn;
        cubp_set_scorethld_ = scorethld;
        cubp_set_qrysernrbeg_ = qrysernrbeg;
        cubp_set_devname_ = devname;
        cubp_set_nqyposs_ = (int)nqyposs;
        cubp_set_nqystrs_ = nqystrs;
        cubp_set_ndbCposs_ = ndbCposs;
        cubp_set_ndbCstrs_ = ndbCstrs;
        cubp_set_querydesc_ = querydesc;
        if(querypmbeg && querypmend) {
            memcpy(cubp_set_querypmbeg_, querypmbeg, pmv2DTotFlds * sizeof(void*));
            memcpy(cubp_set_querypmend_, querypmend, pmv2DTotFlds * sizeof(void*));
        }
        else {
            memset(cubp_set_querypmbeg_, 0, pmv2DTotFlds * sizeof(void*));
            memset(cubp_set_querypmend_, 0, pmv2DTotFlds * sizeof(void*));
        }
        cubp_set_bdbCdesc_ = bdbCdesc;
        if(bdbCpmbeg && bdbCpmend) {
            memcpy(cubp_set_bdbCpmbeg_, bdbCpmbeg, pmv2DTotFlds * sizeof(void*));
            memcpy(cubp_set_bdbCpmend_, bdbCpmend, pmv2DTotFlds * sizeof(void*));
        }
        else {
            memset(cubp_set_bdbCpmbeg_, 0, pmv2DTotFlds * sizeof(void*));
            memset(cubp_set_bdbCpmend_, 0, pmv2DTotFlds * sizeof(void*));
        }
        cubp_set_nposits_.clear();
        cubp_set_nstrs_.clear();
        for(size_t i = 0; i < nqystrs; i++) {
            size_t loff = nDevGlobVariables * i;
            cubp_set_nposits_.push_back(passedstats[loff + dgvNPosits]);
            cubp_set_nstrs_.push_back((int)passedstats[loff + dgvNPassedStrs]);
        }
        cubp_set_qrscnt_ = qrscnt;
        cubp_set_cnt_ = cnt;
        cubp_set_h_results_ = h_results;
        cubp_set_sz_alndata_ = sz_alndata;
        cubp_set_sz_tfmmatrices_ = sz_tfmmatrices;
        cubp_set_sz_alns_ = sz_alns;
        cubp_set_sz_dbalnlen2_ = dbalnlen2;
    }


protected:
    void Execute(void* args);

    void SetResponseError() {
        std::lock_guard<std::mutex> lck(mx_dataccess_);
        rsp_msg_ = CUBPTHREAD_MSG_ERROR;
    }

    void FinalizeQueries();
    void SortCompressedResults();
    void PassResultsToWriter();
    void PrintCompressedResults() const;


    //{{formating methods for PLAIN format
    void CompressResultsPlain();
    void GetSizeOfCompressedResultsPlain(
        size_t* szannot, size_t* szalns, size_t* szalnswodesc) const;

    void MakeAnnotationPlain(char*& outptr,
        const int strndx, const unsigned int orgstrndx, const char* desc,
        const unsigned int alnlen, const int dbstrlen) const;

    void FormatScoresPlain(char*& outptr,
        int strndx, unsigned int orgstrndx, unsigned int alnlen,
        float score, int qrystrlen, int dbstrlen);

    void FormatAlignmentPlain(char*& outptr,
        int strndx, unsigned int orgstrndx, unsigned int dbstr2dst,
        int alnlen, int qrystrlen, int dbstrlen, const int width);

    void FormatFooterPlain(char*& outptr, int strndx);
    //}}
    //{{formating methods for JSON format
    void CompressResultsJSON();
    void GetSizeOfCompressedResultsJSON(
        size_t* szannot, size_t* szalns, size_t* szalnswodesc) const;

    void MakeAnnotationJSON(char*& outptr,
        const int strndx, const char* desc,
        const unsigned int alnlen, const int dbstrlen) const;

    void FormatScoresJSON(char*& outptr, int strndx, unsigned int alnlen);

    void FormatAlignmentJSON(char*& outptr,
        int strndx, unsigned int orgstrndx, unsigned int dbstr2dst,
        int alnlen, int qrystrlen, int dbstrlen);

    void FormatFooterJSON(char*& outptr, int strndx);
    //}}

    
    const char* GetBegOfAlns() {
        return cubp_set_h_results_ + cubp_set_sz_alndata_ + cubp_set_sz_tfmmatrices_;
    }

    const char* GetAlnSectionAt(const char* ptr, const int sctndx) const {
        return ptr + sctndx * dbalnlen_;
    }

    template<typename T>
    T GetOutputAlnDataField(int strndx, int field) const
        {
            return 
                *(T*)((float*)cubp_set_h_results_ + 
                    nTDP2OutputAlnData * (cumnstrs_ + strndx) + field);
        }

    float GetOutputTfmMtxField(int strndx, int field) const
        {
            return 
                *((float*)(cubp_set_h_results_ + cubp_set_sz_alndata_) + 
                    nTTranformMatrix * (cumnstrs_ + strndx) + field);
        }

    float* GetOutputTfmMtxAddress(int strndx) const
        {
            return 
                (float*)(cubp_set_h_results_ + cubp_set_sz_alndata_) + 
                    nTTranformMatrix * (cumnstrs_ + strndx);
        }

    template<typename T>
    T GetStructureField(
        char* const pmbeg[pmv2DTotFlds],
        unsigned int ndx, unsigned int field) const
        {
            return ((T*)(pmbeg[field]))[ndx];
        }

    template<typename T>
    T GetQueryField(unsigned int orgqryndx, unsigned int field) const;
    template<typename T>
    T GetQueryFieldPos(unsigned int field, unsigned int pos) const;
    template<typename T>
    T GetDbStructureField(unsigned int orgstrndx, unsigned int field) const;
    template<typename T>
    T GetDbStructureFieldPos(unsigned int orgstrndx, unsigned int field, unsigned int pos) const;

    void GetDbStructureDesc(const char*& desc, unsigned int orgstrndx) const;


    void ReserveVectors(int capacity) {
        srtindxs_.reset(new std::vector<int>);
        scores_.reset(new std::vector<float>);
        alnptrs_.reset(new std::vector<char*>);
        annotptrs_.reset(new std::vector<char*>);
        if(capacity > 0) {
            if(srtindxs_) srtindxs_->reserve(capacity);
            if(scores_) scores_->reserve(capacity);
            if(alnptrs_) alnptrs_->reserve(capacity);
            if(annotptrs_) annotptrs_->reserve(capacity);
        }
    }

private:
    //thread section
    std::thread* tobj_;//thread object
private:
    //{{messaging
    std::condition_variable cv_msg_;//condition variable for messaging
    mutable std::mutex mx_dataccess_;//mutex for accessing class data
    int req_msg_;//request message issued for thread
    int rsp_msg_;//private response message
    //}}
    //
#ifdef GPUINUSE
    //properties of device the thread is associated with:
    cudaStream_t& strcopyres_;
    DeviceProperties dprop_;
#endif
    //results writer:
    TdAlnWriter* alnwriter_;
    //{{data arguments: 
    // cubp-set data/addresses:
    int qrysernr_;//serial number of the query under processing
    std::string qrydesc_;//description of query qrysernr_
    int qrystrlen_;//length of query qrysernr_
    int qrynstrs_;//number of target db structures for query qrysernr_
    unsigned int qrynposits_;//total number of target db structure positions for query qrysernr_
    int cumnstrs_;//cumulative number of structures over all queries up to qrysernr_
    int offsetalns_;//offset to the start of the alignments for query qrysernr_
    int dbalnlen_;//alignment length for query qrysernr_ across all references
    double cubp_set_duration_;//time duration
    float cubp_set_scorethld_;//tm-score threshold
    int cubp_set_qrysernrbeg_;//serial number of the first query in the chunk
    int cubp_set_nqyposs_;//total length of queries in the chunk
    size_t cubp_set_nqystrs_;
    size_t cubp_set_ndbCposs_;
    size_t cubp_set_ndbCstrs_;
    std::string cubp_set_devname_;//device name
    const char** cubp_set_querydesc_;
    char* cubp_set_querypmbeg_[pmv2DTotFlds];
    char* cubp_set_querypmend_[pmv2DTotFlds];
    const char** cubp_set_bdbCdesc_;
    char* cubp_set_bdbCpmbeg_[pmv2DTotFlds];
    char* cubp_set_bdbCpmend_[pmv2DTotFlds];
    TSCounterVar* cubp_set_qrscnt_;//counter for queries
    TSCounterVar* cubp_set_cnt_;//counter for references
    std::vector<unsigned int> cubp_set_nposits_;//total number of positions in the transfered results for each query
    std::vector<int> cubp_set_nstrs_;//number of structures in the transfered results for each query
    const char* cubp_set_h_results_;//cubp-set host-side results
    size_t cubp_set_sz_alndata_;//size of alignment data of results
    size_t cubp_set_sz_tfmmatrices_;//size of transformation matrices of results
    size_t cubp_set_sz_alns_;//size of alignments of results
    int cubp_set_sz_dbalnlen2_;
    //}}
    //{{formatted results for one particular query:
    std::unique_ptr<char,WritersDataDestroyer> annotations_;
    std::unique_ptr<char,WritersDataDestroyer> alignments_;
    std::unique_ptr<std::vector<int>> srtindxs_;//index vector of sorted scores
    std::unique_ptr<std::vector<float>> scores_;//vector of scores
    std::unique_ptr<std::vector<char*>> alnptrs_;//vector of alignments
    std::unique_ptr<std::vector<char*>> annotptrs_;//vector of annotations
    //}}
};

// -------------------------------------------------------------------------
// GetQueryField: get a field of the given query structure;
// NOTE: the pointer of the query structure under process is accessed!
template<typename T>
inline
T TdFinalizer::GetQueryField(
    unsigned int orgqryndx, unsigned int field) const
{
    return GetStructureField<T>(cubp_set_querypmbeg_, orgqryndx, field);
}
// GetQueryFieldPos: get a field of the query structure at the given 
// position;
// NOTE: the pointer of the query structure under process is accessed!
template<typename T>
inline
T TdFinalizer::GetQueryFieldPos(
    unsigned int field, unsigned int pos) const
{
    return GetStructureField<T>(cubp_set_querypmbeg_, pos, field);
}
// GetDbStructureField: get a field of the given Db structure;
// orgstrndx, index of the structure over all pm data structures;
template<typename T>
inline
T TdFinalizer::GetDbStructureField(
    unsigned int orgstrndx, unsigned int field) const
{
    return GetStructureField<T>(cubp_set_bdbCpmbeg_, orgstrndx, field);
}
// GetDbStructureFieldPos: get a field of the given Db structure at the 
// given position;
// orgstrndx, index of the structure over all pm data structures;
template<typename T>
inline
T TdFinalizer::GetDbStructureFieldPos(
    unsigned int /*orgstrndx*/, unsigned int field, unsigned int pos) const
{
    return GetStructureField<T>(cubp_set_bdbCpmbeg_, pos, field);
}

// GetDbStructureDesc: get the db structure description;
// orgstrndx, index of the structure over all pm data structures;
inline
void TdFinalizer::GetDbStructureDesc(
    const char*& desc, unsigned int orgstrndx) const
{
    desc = cubp_set_bdbCdesc_[orgstrndx];
}

#endif//__TdFinalizer_h__
