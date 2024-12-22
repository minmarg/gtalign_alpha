/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/mybase.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <functional>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <thread>

#ifdef GPUINUSE
#   include <cuda.h>
#   include <cuda_runtime_api.h>
#   include "libmycu/cucom/cucommon.h"
#   include "libmycu/cuproc/Devices.h"
static cudaStream_t streamdummy;
#endif

#include "libutil/CLOptions.h"
#include "tsafety/TSCounterVar.h"

#include "libgenp/gdats/PM2DVectorFields.h"
// #include "libgenp/gdats/PMBatchStrData.h"
#include "libgenp/goutp/TdAlnWriter.h"

#include "libmycu/cuproc/cuprocconf.h"
#include "TdFinalizer.h"

// _________________________________________________________________________
// class statics
//

// _________________________________________________________________________
// Class TdFinalizer
//
// Constructors
//
#ifdef GPUINUSE
TdFinalizer::TdFinalizer(
    cudaStream_t& strcopyres,
    DeviceProperties dprop, 
    TdAlnWriter* writer)
:   tobj_(NULL),
    req_msg_(CUBPTHREAD_MSG_UNSET),
    rsp_msg_(CUBPTHREAD_MSG_UNSET),
    strcopyres_(strcopyres),
    dprop_(dprop),
    alnwriter_(writer),
    //data arguments:
    qrysernr_(-1),
    // qrydesc_(NULL),
    qrystrlen_(0),
    qrynstrs_(0),
    qrynposits_(0),
    cumnstrs_(0),
    offsetalns_(0),
    dbalnlen_(0),
    cubp_set_duration_(0.0),
    cubp_set_scorethld_(0.0f),
    cubp_set_qrysernrbeg_(-1),
    cubp_set_nqyposs_(0),
    cubp_set_nqystrs_(0UL),
    cubp_set_ndbCposs_(0UL),
    cubp_set_ndbCstrs_(0UL),
    cubp_set_querydesc_(NULL),
    cubp_set_bdbCdesc_(NULL),
    cubp_set_qrscnt_(NULL),
    cubp_set_cnt_(NULL),
    cubp_set_h_results_(NULL),
    cubp_set_sz_alndata_(0UL),
    cubp_set_sz_alns_(0UL),
    cubp_set_sz_dbalnlen2_(0),
    annotations_(nullptr),
    alignments_(nullptr)
{
    MYMSG("TdFinalizer::TdFinalizer", 3);
    memset(cubp_set_querypmbeg_, 0, pmv2DTotFlds * sizeof(void*));
    memset(cubp_set_querypmend_, 0, pmv2DTotFlds * sizeof(void*));
    memset(cubp_set_bdbCpmbeg_, 0, pmv2DTotFlds * sizeof(void*));
    memset(cubp_set_bdbCpmend_, 0, pmv2DTotFlds * sizeof(void*));
    tobj_ = new std::thread(&TdFinalizer::Execute, this, (void*)NULL);
}
#endif

TdFinalizer::TdFinalizer(
    TdAlnWriter* writer)
:   tobj_(NULL),
    req_msg_(CUBPTHREAD_MSG_UNSET),
    rsp_msg_(CUBPTHREAD_MSG_UNSET),
#ifdef GPUINUSE
    strcopyres_(streamdummy),
#endif
    alnwriter_(writer),
    //data arguments:
    qrysernr_(-1),
    // qrydesc_(NULL),
    qrystrlen_(0),
    qrynstrs_(0),
    qrynposits_(0),
    cumnstrs_(0),
    offsetalns_(0),
    dbalnlen_(0),
    cubp_set_duration_(0.0),
    cubp_set_scorethld_(0.0f),
    cubp_set_qrysernrbeg_(-1),
    cubp_set_nqyposs_(0),
    cubp_set_nqystrs_(0UL),
    cubp_set_ndbCposs_(0UL),
    cubp_set_ndbCstrs_(0UL),
    cubp_set_querydesc_(NULL),
    cubp_set_bdbCdesc_(NULL),
    cubp_set_qrscnt_(NULL),
    cubp_set_cnt_(NULL),
    cubp_set_h_results_(NULL),
    cubp_set_sz_alndata_(0UL),
    cubp_set_sz_alns_(0UL),
    cubp_set_sz_dbalnlen2_(0),
    annotations_(nullptr),
    alignments_(nullptr)
{
    MYMSG("TdFinalizer::TdFinalizer", 3);
    memset(cubp_set_querypmbeg_, 0, pmv2DTotFlds * sizeof(void*));
    memset(cubp_set_querypmend_, 0, pmv2DTotFlds * sizeof(void*));
    memset(cubp_set_bdbCpmbeg_, 0, pmv2DTotFlds * sizeof(void*));
    memset(cubp_set_bdbCpmend_, 0, pmv2DTotFlds * sizeof(void*));
    tobj_ = new std::thread(&TdFinalizer::Execute, this, (void*)NULL);
}

// Destructor
//
TdFinalizer::~TdFinalizer()
{
    MYMSG("TdFinalizer::~TdFinalizer", 3);
    if( tobj_ ) {
        tobj_->join();
        delete tobj_;
        tobj_ = NULL;
    }
}

// -------------------------------------------------------------------------
// Execute: thread's starting point for execution
//
void TdFinalizer::Execute(void*)
{
    MYMSG("TdFinalizer::Execute", 3);
    myruntime_error mre;
    char msgbuf[BUF_MAX];

    try {
//         const int outfmt = CLOptions::GetB_FMT();

#ifdef GPUINUSE
        if(dprop_.DevidValid()) {
            MYCUDACHECK(cudaSetDevice(dprop_.devid_));
            MYCUDACHECKLAST;
        }
#endif

        cubp_set_devname_.reserve(256);
        cubp_set_nposits_.reserve(MAX_QUERY_STRUCTURES_PER_CHUNK);
        cubp_set_nstrs_.reserve(MAX_QUERY_STRUCTURES_PER_CHUNK);

        while(1) {
            //wait for a message
            std::unique_lock<std::mutex> lck_msg(mx_dataccess_);

            cv_msg_.wait(lck_msg,
                [this]{
                    return (
                        (0 <= req_msg_ && req_msg_ <= cubpthreadmsgTerminate) || 
                        req_msg_ == CUBPTHREAD_MSG_ERROR
                    );
                }
            );

            MYMSGBEGl(3)
                sprintf(msgbuf, "TdFinalizer::Execute: Msg %d", req_msg_);
                MYMSG(msgbuf, 3);
            MYMSGENDl

            //thread owns the lock after the wait;
            //read message req_msg_
            int reqmsg = req_msg_;

            //unset the message to avoid live cycle when starting over the loop
            req_msg_ = CUBPTHREAD_MSG_UNSET;

            //set response msg to error upon exception
            rsp_msg_ = CUBPTHREAD_MSG_ERROR;
            int rspmsg = rsp_msg_;

            switch(reqmsg) {
                case cubpthreadmsgFinalize:
#ifdef GPUINUSE
                        if(dprop_.DevidValid()) {
                            //data addresses have been written already;
                            //make sure data transfer has finished
                            MYCUDACHECK(cudaStreamSynchronize(strcopyres_));
                            MYCUDACHECKLAST;
                        }
#endif
                        ;;
                        FinalizeQueries();
                        ;;
                        //parent does not wait for a response nor requires data to read;
                        //unset response code
                        rspmsg = CUBPTHREAD_MSG_UNSET;//cubptrespmsgFinalizing;
                        break;
                case cubpthreadmsgTerminate:
                        rspmsg = cubptrespmsgTerminating;
                        break;
                default:
                        //rspmsg = CUBPTHREAD_MSG_UNSET;
                        break;
            };

            MYMSGBEGl(3)
                sprintf(msgbuf, "TdFinalizer::Execute: Msg %d Rsp %d", reqmsg, rspmsg);
                MYMSG(msgbuf, 3);
            MYMSGENDl

            //save response code
            rsp_msg_ = rspmsg;

            //unlock the mutex and notify the parent
            lck_msg.unlock();
            cv_msg_.notify_one();

            if(reqmsg < 0 || reqmsg == cubpthreadmsgTerminate)
                //terminate execution
                break;
        }

    } catch(myruntime_error const& ex) {
        mre = ex;
    } catch(myexception const& ex) {
        mre = ex;
    } catch(...) {
        mre = MYRUNTIME_ERROR("Unknown exception caught.");
    }

    if(mre.isset()) {
        error(mre.pretty_format().c_str());
        SetResponseError();
        cv_msg_.notify_one();
        return;
    }
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// FinalizeQueries: finalize all queries in the chunk: iterate over the 
// queries, format results for each query and pass them to the writer;
// NOTE: all operations performed under lock
//
void TdFinalizer::FinalizeQueries()
{
    MYMSG("TdFinalizer::FinalizeQueries", 3);
    static const std::string preamb = "TdFinalizer::FinalizeQueries: ";
    static const int outfmt = CLOptions::GetO_OUTFMT();

    int i = 0;
    int nqystrs = cubp_set_nqystrs_;
    char errbuf[BUF_MAX];

    qrydesc_.clear();
    qrynstrs_ = 0;
    qrynposits_ = 0;
    cumnstrs_ = 0;
    offsetalns_ = 0;
    dbalnlen_ = 0;

    for(i = 0, qrysernr_ = cubp_set_qrysernrbeg_;
        i < nqystrs; qrysernr_++, i++) 
    {
        qrystrlen_ = GetQueryField<INTYPE>(i, pps2DLen);
        //size of all alignments produced for the current query
        dbalnlen_ = cubp_set_nposits_[i] + cubp_set_nstrs_[i] * (qrystrlen_+1);

        if(qrystrlen_ < 1) {
            sprintf(errbuf, "TdFinalizer::FinalizeQueries: "
                "Invalid length of query no. %d (%d): %d", 
                qrysernr_, i, qrystrlen_);
            throw MYRUNTIME_ERROR(errbuf);
        }

        if(cubp_set_nstrs_[i] < 0 || (int)cubp_set_nposits_[i] < cubp_set_nstrs_[i]) {
            sprintf(errbuf, "TdFinalizer::FinalizeQueries: Invalid #ref. structures "
                "and/or their total positions for query no. %d: %d, %u",
                qrysernr_, cubp_set_nstrs_[i], cubp_set_nposits_[i]);
            throw MYRUNTIME_ERROR(errbuf);
        }

        // qrydesc_ = cubp_set_querydesc_? cubp_set_querydesc_[i]: NULL;
        if(cubp_set_querydesc_) qrydesc_ = cubp_set_querydesc_[i]; else qrydesc_.clear();
        qrynstrs_ = cubp_set_nstrs_[i];
        qrynposits_ = cubp_set_nposits_[i];

        if(outfmt == CLOptions::oofJSON)
            CompressResultsJSON();
        else
            CompressResultsPlain();

        //results have been processed and the buffers are no longer needed;
        //decrement the counter of the data chunk processed by the agent
        if(i+1 == nqystrs) {
            if(cubp_set_qrscnt_) cubp_set_qrscnt_->dec();
            if(cubp_set_cnt_) cubp_set_cnt_->dec();
        }

        SortCompressedResults();
        PassResultsToWriter();

        //increment by the size of all alignments produced for the current query
        offsetalns_ += dbalnlen_ * nTDP2OutputAlignmentSSS;
        cumnstrs_ += cubp_set_nstrs_[i];
    }
}

// -------------------------------------------------------------------------
// PassResultsToWriter: transfer the addresses of sorted results to the 
// alignment writer;
// NOTE: all operations performed under lock
//
void TdFinalizer::PassResultsToWriter()
{
    MYMSG("TdFinalizer::PassResultsToWriter", 4);
    static const std::string preamb = "TdFinalizer::PassResultsToWriter: ";

    if(alnwriter_) {
        alnwriter_->PushPartOfResults(
            qrysernr_,
            qrystrlen_,
            qrydesc_,
            cubp_set_devname_,
            cubp_set_nqystrs_,
            cubp_set_duration_,
            cubp_set_scorethld_,
            cubp_set_ndbCposs_,
            cubp_set_ndbCstrs_,
            std::move(annotations_),
            std::move(alignments_),
            std::move(srtindxs_),
            std::move(scores_),
            std::move(alnptrs_),
            std::move(annotptrs_)
        );
    }
}

// -------------------------------------------------------------------------
// SortCompressedResults: sort formatted results;
// NOTE: all operations performed under lock
//
void TdFinalizer::SortCompressedResults()
{
    MYMSG("TdFinalizer::SortCompressedResults", 4);
    static const std::string preamb = "TdFinalizer::SortCompressedResults: ";

    if( !srtindxs_ || !scores_ || !alnptrs_ || !annotptrs_ )
        throw MYRUNTIME_ERROR(preamb + "Null compressed results.");

    if( srtindxs_->size() != scores_->size() ||
        scores_->size() != alnptrs_->size() ||
        scores_->size() != annotptrs_->size())
        throw MYRUNTIME_ERROR(preamb + "Inconsistent result sizes.");

    std::sort(srtindxs_->begin(), srtindxs_->end(),
        [this](size_t n1, size_t n2) {
            return (*scores_)[n1] > (*scores_)[n2];
        }
    );
}



// -------------------------------------------------------------------------
// PrintCompressedResults: print formatted alignments
//
void TdFinalizer::PrintCompressedResults() const
{
    MYMSG("TdFinalizer::PrintCompressedResults", 4);
    static const std::string preamb = "TdFinalizer::PrintCompressedResults: ";

    if( !srtindxs_ || !scores_ || !alnptrs_ || !annotptrs_ )
        throw MYRUNTIME_ERROR(preamb + "Null compressed results.");

    if( srtindxs_->size() != scores_->size() ||
        scores_->size() != alnptrs_->size() ||
        scores_->size() != annotptrs_->size())
        throw MYRUNTIME_ERROR(preamb + "Inconsistent result sizes.");

    for(size_t i = 0; i < srtindxs_->size(); i++ ) {
        fprintf(stdout,"%s",(*annotptrs_)[(*srtindxs_)[i]]);
    }

    fprintf(stdout,"%s",NL);

    for(size_t i = 0; i < srtindxs_->size(); i++ ) {
        fprintf(stdout,"%f%s",(*scores_)[(*srtindxs_)[i]],NL);
        fprintf(stdout,"%s",(*alnptrs_)[(*srtindxs_)[i]]);
    }
}
