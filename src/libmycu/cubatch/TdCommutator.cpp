/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/mybase.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>

#include "tsafety/TSCounterVar.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libgenp/gdats/PMBatchStrData.h"
#include "libgenp/goutp/TdClustWriter.h"
#include "libgenp/goutp/TdAlnWriter.h"
#include "libmycu/cuproc/Devices.h"
#include "libmycu/culayout/CuDeviceMemory.cuh"
#include "libmycu/cubatch/CuBatch.cuh"
#include "TdCommutator.h"

// _________________________________________________________________________
// Class TdCommutator
//
// Constructor
//
TdCommutator::TdCommutator(
    size_t ringsize,
    int tid, CuDeviceMemory* dmem, int areano,
    TdAlnWriter* writer, TdClustWriter* clustwriter,
    size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs)
:   
    ringsize_(ringsize),
    mytid_(tid),
    myareano_(areano),
    tobj_(NULL),
    req_msg_(THREAD_MSG_UNSET),
    msg_addressee_(THREAD_MSG_ADDRESSEE_NONE),
    rsp_msg_(THREAD_MSG_UNSET),
    //
    dmem_(dmem),
    alnwriter_(writer),
    clustwriter_(clustwriter),
    chunkdatasize_(chunkdatasize),
    chunkdatalen_(chunkdatalen),
    chunknstrs_(chunknstrs),
    //query-specific arguments:
    nqrsposs_(0UL),
    mstr_set_nqrsposs_(0UL),
    mstr_set_scorethld_(0.0f),
    //data arguments:
    mstr_set_lastchunk_(false),
    mstr_set_chunkno_(-1),
    mstr_set_qrysernrbeg_(-1),
    mstr_set_newsetqrs_(false),
    mstr_set_querydesc_(NULL),
    mstr_set_bdbCdesc_(NULL),
    mstr_set_qrscnt_(NULL),
    mstr_set_cnt_(NULL),
    scorethld_(0.0f),
    lastchunk_(false),
    chunkno_(-1),
    qrysernrbeg_(-1),
    newsetqrs_(false),
    querydesc_(NULL),
    bdbCdesc_(NULL),
    qrscnt_(NULL),
    cnt_(NULL)
{
    MYMSG("TdCommutator::TdCommutator", 3);

    if(dmem_ == NULL)
        throw MYRUNTIME_ERROR(
        "TdCommutator::TdCommutator: Null device memory object.");

    memset( mstr_set_queryndxpmbeg_, 0, pmv2DTotIndexFlds * sizeof(void*));
    memset( mstr_set_queryndxpmend_, 0, pmv2DTotIndexFlds * sizeof(void*));
    memset( mstr_set_querypmbeg_, 0, pmv2DTotFlds * sizeof(void*));
    memset( mstr_set_querypmend_, 0, pmv2DTotFlds * sizeof(void*));
    memset( mstr_set_bdbCndxpmbeg_, 0, pmv2DTotIndexFlds * sizeof(void*));
    memset( mstr_set_bdbCndxpmend_, 0, pmv2DTotIndexFlds * sizeof(void*));
    memset( mstr_set_bdbCpmbeg_, 0, pmv2DTotFlds * sizeof(void*));
    memset( mstr_set_bdbCpmend_, 0, pmv2DTotFlds * sizeof(void*));
    memset( queryndxpmbeg_, 0, pmv2DTotIndexFlds * sizeof(void*));
    memset( queryndxpmend_, 0, pmv2DTotIndexFlds * sizeof(void*));
    memset( querypmbeg_, 0, pmv2DTotFlds * sizeof(void*));
    memset( querypmend_, 0, pmv2DTotFlds * sizeof(void*));
    memset( bdbCndxpmbeg_, 0, pmv2DTotIndexFlds * sizeof(void*));
    memset( bdbCndxpmend_, 0, pmv2DTotIndexFlds * sizeof(void*));
    memset( bdbCpmbeg_, 0, pmv2DTotFlds * sizeof(void*));
    memset( bdbCpmend_, 0, pmv2DTotFlds * sizeof(void*));

    tobj_ = new std::thread( &TdCommutator::Execute, this, (void*)NULL );
}

// Destructor
//
TdCommutator::~TdCommutator()
{
    MYMSG("TdCommutator::~TdCommutator", 3);
    if(tobj_) {
        tobj_->join();
        delete tobj_;
        tobj_ = NULL;
    }
}

// -------------------------------------------------------------------------
// Execute: thread's starting point and execution process
//
void TdCommutator::Execute( void* )
{
    MYMSG( "TdCommutator::Execute", 3 );
    myruntime_error mre;
    char msgbuf[BUF_MAX];

    try {
        CuBatch cbpc( 
            ringsize_,
            dmem_, myareano_, alnwriter_, clustwriter_,
            chunkdatasize_, chunkdatalen_, chunknstrs_
        );

        while(1) {
            //wait until the master sends a message
            std::unique_lock<std::mutex> lck_msg(mx_rsp_msg_);

            cv_rsp_msg_.wait(lck_msg,
                [this]{
                    return msg_addressee_ == mytid_
                            &&
                        ((0 <= req_msg_ && req_msg_ <= tthreadmsgTerminate) || 
                        req_msg_ == THREAD_MSG_ERROR);
                }
            );

            MYMSGBEGl(3)
                sprintf(msgbuf, "TdCommutator::Execute: Msg %d Adr %d",
                        req_msg_, msg_addressee_);
                MYMSG(msgbuf, 3);
            MYMSGENDl

            //thread owns the lock after the wait;
            //read message req_msg_
            int reqmsg = req_msg_;

            //unset the message to avoid live cycle when starting over the loop
            req_msg_ = THREAD_MSG_UNSET;
            msg_addressee_ = THREAD_MSG_ADDRESSEE_NONE;

            //set response msg to error in advance if an exception throws
            rsp_msg_ = THREAD_MSG_ERROR;
            int rspmsg = rsp_msg_;

            //immediately read the master-set data
            switch(reqmsg) {
                case tthreadmsgGetDataChunkSize:
                        //master has set the query length and the score threshold;
                        GetArgsOnMsgGetDataChunkSize(cbpc);
                        break;
                case tthreadmsgProcessNewData:
                        //master has written data addresses
                        CopyDataOnMsgProcessNewData(cbpc);
                        break;
                case tthreadmsgProbe: break;
                case tthreadmsgTerminate: break;
                default: break;
            };

            switch(reqmsg) {
                case tthreadmsgGetDataChunkSize:
                        //master has set the query length and the score threshold;
                        CalculateMaxDbDataChunkSize(cbpc);
                        rspmsg = ttrespmsgChunkSizeReady;
                        break;
                case tthreadmsgProcessNewData:
                        //master has written data addresses
                        ProcessBlock(cbpc);
                        //master does not wait for a response nor requires data to read;
                        //unset response code
                        rspmsg = THREAD_MSG_UNSET;//ttrespmsgInProgress;
                        break;
                case tthreadmsgProbe:
//                         cbpc.WaitForIdleChilds();
                        rspmsg = ttrespmsgProbed;
                        break;
                case tthreadmsgTerminate:
                        rspmsg = ttrespmsgTerminating;
                        break;
                default:
                        rspmsg = THREAD_MSG_UNSET;
                        break;
            };

            MYMSGBEGl(3)
                sprintf(msgbuf, "TdCommutator::Execute: Msg %d Rsp %d",reqmsg, rspmsg);
                MYMSG(msgbuf, 3);
            MYMSGENDl

            //save response code
            rsp_msg_ = rspmsg;

            //send a message back to the master:
            //unlock the mutex to avoid to block the awaiting master and 
            // notify it
            lck_msg.unlock();
            cv_rsp_msg_.notify_one();

            if(reqmsg < 0 || reqmsg == tthreadmsgTerminate)
                //terminate execution
                break;
        }

    } catch( myruntime_error const& ex ) {
        mre = ex;
    } catch( myexception const& ex ) {
        mre = ex;
    } catch( ... ) {
        mre = MYRUNTIME_ERROR("Unknown exception caught.");
    }

    if(mre.isset()) {
        error(mre.pretty_format().c_str());
        {//notify the master
            std::lock_guard<std::mutex> lck_msg(mx_rsp_msg_);
            rsp_msg_ = THREAD_MSG_ERROR;
        }
        cv_rsp_msg_.notify_one();
        ResetMasterData();
        return;
    }
}

// -------------------------------------------------------------------------
// CalculateMaxDbDataChunkSize: calculate maximum database data chunk size 
// given query length previously set by the master thread
//
void TdCommutator::CalculateMaxDbDataChunkSize(CuBatch& /*cbpc*/)
{
    throw MYRUNTIME_ERROR(
    "TdCommutator::CalculateMaxDbDataChunkSize: Should not be called!");
    //safely write member variables as long as the master waiting for them is blocked
//     size_t chunkdatasize = cbpc.CalcMaxDbDataChunkSize( nqyposs_ );
//     size_t chunkdatalen = cbpc.GetCurrentMaxDbPos();
//     size_t chunknpros = cbpc.GetCurrentMaxNDbPros();
//     SetChunkDataAttributes( chunkdatasize, chunkdatalen, chunknpros);
}

// -------------------------------------------------------------------------
// ProcessBlock: batch processing of a block of the matrix 
// formed by the total number of queries and reference structures
//
void TdCommutator::ProcessBlock(CuBatch& cbpc)
{
    bool bdbCinf = PMBatchStrData::ContainsData(bdbCpmbeg_, bdbCpmend_);
    //NOTE: lock so that other threads (if any) assigned to the same 
    // device wait for the transfer to complete
    {   std::lock_guard<std::mutex> lck(dmem_->GetDeviceProp().shdcnt_->get_mutex());
        int cntval = dmem_->GetDeviceProp().shdcnt_->get_under_lock();
        if(cntval != chunkno_ || newsetqrs_) {
            // if(chunkno_ < cntval)
            //     dmem_->GetDeviceProp().shdcnt_->reset_under_lock();
            // dmem_->GetDeviceProp().shdcnt_->inc_under_lock();
            dmem_->GetDeviceProp().shdcnt_->set_under_lock(chunkno_);
            if(bdbCinf &&
               bdbCndxpmbeg_[0] && bdbCndxpmend_[0] &&
               bdbCpmbeg_[0] && bdbCpmend_[0])
                cbpc.TransferCPMDataToDevice(
                    bdbCndxpmbeg_, bdbCndxpmend_, bdbCpmbeg_, bdbCpmend_);
        }
        //TODO: each agent on the same device transfers query data;
        //TODO: make only the first agent to transfer query data;
        //TODO: additional shared counter (shdcnt) will be required for this;
        if(newsetqrs_ &&
           queryndxpmbeg_[0] && queryndxpmend_[0] &&
           querypmbeg_[0] && querypmend_[0])
            cbpc.TransferQueryPMDataToDevice(
                queryndxpmbeg_, queryndxpmend_, querypmbeg_, querypmend_);
    }

    char msgbuf[BUF_MAX];
    int nqueries = PMBatchStrData::GetNoStructs(querypmbeg_, querypmend_);
    int qrysernrend = qrysernrbeg_ + nqueries - 1;

    if(nqueries < 1) {
        //sprintf(msgbuf, "Invalid number of queries to be processed: %d", nqueries);
        //warning(msgbuf);
        //NOTE: queries can be empty: decrease the counters
        if(qrscnt_) qrscnt_->dec();
        if(cnt_) cnt_->dec();
        return;
    }

    MYMSGBEGl(3)
        sprintf(msgbuf, "Processing QUERY nos. %d - %d", qrysernrbeg_, qrysernrend);
        MYMSG(msgbuf,3);
    MYMSGENDl

    //increase # chunks being processed only if this is not the last chunk!
    //this ensures the processing of all chunks and informs the writer to begin 
    // writing once the last chunk has finished
    if(!lastchunk_) {
        qrscnt_->inc();//each query is referenced as many times as there are chunks
        if(alnwriter_) alnwriter_->IncreaseQueryNParts(qrysernrbeg_, qrysernrend);
        if(clustwriter_) clustwriter_->IncreaseQueryNParts(qrysernrbeg_, qrysernrend);
    }

    if(clustwriter_ && !bdbCinf)
        clustwriter_->PushPartOfResults(
            qrysernrbeg_, 0/*nqyposs (unused)*/, nqueries,
            querydesc_, querypmbeg_, querypmend_,  bdbCdesc_, bdbCpmbeg_, bdbCpmend_,
            qrscnt_, cnt_, NULL/*passedstats*/, NULL/*h_results_*/, 0/*szresults*/
        );
    else
        cbpc.ProcessBlock(
            scorethld_,
            qrysernrbeg_,
            querydesc_,
            querypmbeg_[0]? querypmbeg_: NULL, querypmend_[0]? querypmend_: NULL,
            bdbCdesc_,
            bdbCpmbeg_[0]? bdbCpmbeg_: NULL, bdbCpmend_[0]? bdbCpmend_: NULL,
            qrscnt_, cnt_
        );

    MYMSGBEGl(2)
        if(lastchunk_) {
            sprintf(msgbuf, "Worker %d: %d queries processed", mytid_, qrysernrbeg_ + nqueries);
            MYMSG(msgbuf,2);
        }
    MYMSGENDl
}
