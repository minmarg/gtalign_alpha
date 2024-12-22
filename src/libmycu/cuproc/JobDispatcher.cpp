/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/mybase.h"

#include <stdlib.h>
#include <string.h>

#include <string>
#include <memory>
#include <mutex>
#include <functional>
#include <condition_variable>
#include <thread>
#include <vector>
#include <tuple>

#include "libutil/CLOptions.h"
#include "tsafety/TSCounterVar.h"

#include "libgenp/gdats/InputFilelist.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libgenp/gdats/PMBatchStrData.h"
#include "libgenp/gdats/TdDataReader.h"
#include "libmycu/culayout/CuDeviceMemory.cuh"
#include "libmycu/cubatch/TdCommutator.h"
#include "libgenp/goutp/TdClustWriter.h"
#include "libgenp/goutp/TdAlnWriter.h"
#include "libmycu/cuproc/Devices.h"

#include "JobDispatcher.h"

// _________________________________________________________________________
// Class JobDispatcher
//
// Constructors
//
JobDispatcher::JobDispatcher(
    const std::vector<std::string>& inputlist,
    const std::vector<std::string>& dnamelist,
    const std::vector<std::string>& sfxlst, 
    const char* output,
    const char* cachedir)
:
    output_(output),
    cachedir_(cachedir),
    inputlist_(inputlist),
    dnamelist_(dnamelist),
    sfxlst_(sfxlst),
    writer_(NULL),
    clustwriter_(NULL),
    qrsreader_(NULL)
{
    queries_.reset(new InputFilelist(inputlist, sfxlst));
    references_.reset(new InputFilelist(dnamelist, sfxlst, false/*clustering*/, false/*construct*/));

    if(!queries_ || !references_)
        throw MYRUNTIME_ERROR("JobDispatcher: Failed to construct structure lists.");

    readers_.reserve(16);
}

JobDispatcher::JobDispatcher(
    const std::vector<std::string>& clustlist,
    const std::vector<std::string>& sfxlst, 
    const char* output,
    const char* cachedir)
:
    output_(output),
    cachedir_(cachedir),
    clustlist_(clustlist),
    sfxlst_(sfxlst),
    writer_(NULL),
    clustwriter_(NULL),
    qrsreader_(NULL)
{
    queries_.reset(new InputFilelist(clustlist, sfxlst, true/*clustering*/));

    if(!queries_)
        throw MYRUNTIME_ERROR("JobDispatcher: Failed to construct structure lists.");

    readers_.reserve(16);
}

// Destructor
//
JobDispatcher::~JobDispatcher()
{
    for(int tid = 0; tid < (int)hostworkers_.size(); tid++ ) {
        if(hostworkers_[tid]) {
            delete hostworkers_[tid];
            hostworkers_[tid] = NULL;
        }
    }
    for(int d = 0; d < (int)memdevs_.size(); d++ ) {
        if( memdevs_[d]) {
            delete memdevs_[d];
            memdevs_[d] = NULL;
        }
    }
    if(clustwriter_) {
        clustwriter_->Notify(TdClustWriter::clstwrthreadmsgTerminate);
        delete clustwriter_;
        clustwriter_ = NULL;
    }
    if(writer_) {
        writer_->Notify(TdAlnWriter::wrtthreadmsgTerminate);
        delete writer_;
        writer_ = NULL;
    }
    if(qrsreader_) {
        qrsreader_->Notify(TdDataReader::tdrmsgTerminate);
        delete qrsreader_;
        qrsreader_ = NULL;
    }
    std::for_each(readers_.begin(), readers_.end(),
        [](std::unique_ptr<TdDataReader>& p) {
            if(p) p->Notify(TdDataReader::tdrmsgTerminate);
        }
    );
}



// =========================================================================
// CreateReader: create the thread for reading reference data from files
//
void JobDispatcher::CreateReader( 
    int maxstrlen, bool mapped, int ndatbufs, int nagents,
    size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs)
{
    MYMSG("JobDispatcher::CreateReader", 3);
    const bool indexed = true;
    const bool clustering = clustlist_.size();
    InputFilelist* references = clustering? queries_.get(): references_.get();
    size_t cputhsread = CLOptions::GetCPU_THREADS_READING();
    size_t nthreads = cputhsread;
    int cacheflag = TdDataReader::tdrcheNotSet;

    if(!clustering && cachedir_)
        cacheflag = TdDataReader::GetDbsCacheFlag(
                nagents, cachedir_, GetDNamelist(),
                chunkdatasize, chunkdatalen, chunknstrs);

    if(cacheflag != TdDataReader::tdrcheReadCached) {
        references->ConstructFileList();
        nthreads = mymin(cputhsread, references->GetStrFilelist().size());
    }

    for(size_t n = 0; n < nthreads; n++) {
        std::unique_ptr<TdDataReader> tdr(
            new TdDataReader(
                cachedir_,
                clustering? GetClustList(): GetDNamelist(),
                references->GetStrFilelist(),
                references->GetPntFilelist(),
                references->GetStrFilePositionlist(),
                references->GetStrFilesizelist(),
                references->GetStrParenttypelist(),
                references->GetStrFiletypelist(),
                references->GetFilendxlist(),
                references->GetGlobalIds(),
                n/*ndxstartwith*/,
                nthreads/*ndxstep*/,
                maxstrlen, mapped, indexed, ndatbufs, nagents,
                clustering)
        );
        if(!tdr)
            throw MYRUNTIME_ERROR(
            "JobDispatcher::CreateReader: Failed to create reader thread(s).");
        readers_.push_back(std::move(tdr));
    }
}

// -------------------------------------------------------------------------
// CreateQrsReader: create the thread for reading query data from files
//
void JobDispatcher::CreateQrsReader( 
    int maxstrlen,
    bool mapped,
    int ndatbufs,
    int nagents)
{
    MYMSG("JobDispatcher::CreateQrsReader", 3);
    const bool indexed = true;
    const bool clustering = clustlist_.size();
    const bool clustmaster = clustering? true: false;
    qrsreader_ = new TdDataReader(
            "",//cachedir_,//no caching for queries
            inputlist_,
            queries_->GetStrFilelist(),
            queries_->GetPntFilelist(),
            queries_->GetStrFilePositionlist(),
            queries_->GetStrFilesizelist(),
            queries_->GetStrParenttypelist(),
            queries_->GetStrFiletypelist(),
            queries_->GetFilendxlist(),
            queries_->GetGlobalIds(),
            0/*ndxstartwith*/,
            1/*ndxstep*/,
            maxstrlen, mapped, indexed, ndatbufs, nagents,
            clustering, clustmaster);
    if(qrsreader_ == NULL)
        throw MYRUNTIME_ERROR(
        "JobDispatcher::CreateQrsReader: Failed to create the reader for queries.");
}

// -------------------------------------------------------------------------
// GetDataFromReader: request data from the reader and wait for data to
// be ready;
// return false if there are no data to be read;
//
bool JobDispatcher::GetDataFromReader(
    TdDataReader* reader,
    char**& bdbCdescs, char**& bdbCpmbeg, char**& bdbCpmend,
    TSCounterVar*& tscnt,
    bool* lastchunk)
{
    MYMSG("JobDispatcher::GetDataFromReader", 3);

    if(reader == NULL)
        throw MYRUNTIME_ERROR(
        "JobDispatcher::GetDataFromReader: Null reader object.");

    reader->Notify(TdDataReader::tdrmsgGetData);

    int rsp = reader->Wait(TdDataReader::tdrrespmsgDataReady, TdDataReader::tdrrespmsgNoData);

    if(rsp == TREADER_MSG_ERROR)
        throw MYRUNTIME_ERROR(
        "JobDispatcher::GetDataFromReader: A reader terminated with errors.");

    if(rsp != TdDataReader::tdrrespmsgDataReady && rsp != TdDataReader::tdrrespmsgNoData) {
        throw MYRUNTIME_ERROR(
        "JobDispatcher::GetDataFromReader: Invalid response from a reader.");
    }

    *lastchunk = true;
    rsp = (rsp != TdDataReader::tdrrespmsgNoData);

    reader->GetbdbCdata(bdbCdescs, bdbCpmbeg, bdbCpmend,  tscnt, lastchunk);

    return rsp;
}

// GetDataFromReader: request data from the reader and wait for data to
// be ready; version to get index too;
// return false if there are no data to be read;
//
bool JobDispatcher::GetDataFromReader(
    TdDataReader* reader,
    char**& bdbCdescs, char**& bdbCpmbeg, char**& bdbCpmend,
    char**& bdbCNdxpmbeg, char**& bdbCNdxpmend,
    TSCounterVar*& tscnt,
    bool* lastchunk)
{
    MYMSG("JobDispatcher::GetDataFromReader", 3);

    if(reader == NULL)
        throw MYRUNTIME_ERROR(
        "JobDispatcher::GetDataFromReader: Null reader object.");

    reader->Notify(TdDataReader::tdrmsgGetData);

    int rsp = reader->Wait(TdDataReader::tdrrespmsgDataReady, TdDataReader::tdrrespmsgNoData);

    if(rsp == TREADER_MSG_ERROR)
        throw MYRUNTIME_ERROR(
        "JobDispatcher::GetDataFromReader: A reader terminated with errors.");

    if(rsp != TdDataReader::tdrrespmsgDataReady && rsp != TdDataReader::tdrrespmsgNoData) {
        throw MYRUNTIME_ERROR(
        "JobDispatcher::GetDataFromReader: Invalid response from a reader.");
    }

    *lastchunk = true;
    rsp = (rsp != TdDataReader::tdrrespmsgNoData);

    if(rsp)
        reader->GetbdbCdata(
            bdbCdescs, bdbCpmbeg, bdbCpmend, bdbCNdxpmbeg, bdbCNdxpmend,  tscnt, lastchunk);

    return rsp;
}

// -------------------------------------------------------------------------
// GetReferenceData: request data from the readers and wait for data to
// be ready; version to get index too;
// return false if there are no data to be read;
//
bool JobDispatcher::GetReferenceData(
    char**& bdbCdescs, char**& bdbCpmbeg, char**& bdbCpmend,
    char**& bdbCNdxpmbeg, char**& bdbCNdxpmend,
    TSCounterVar*& tscnt, bool* lastchunk, bool rewind,
    size_t ntotqstrs, int queryblocks,
    char* const * const * const querypmbegs,
    char* const * const * const querypmends)
{
    MYMSG("JobDispatcher::GetReferenceData", 3);
    static const std::string preamb = "JobDispatcher::GetReferenceData ";
    int ret = 0;//no data
    static size_t nit = 0;

    //message-broadcast functional
    std::function<void()> lfGetDataBcst = 
        [this,ntotqstrs,queryblocks,querypmbegs,querypmends]() {
        std::for_each(readers_.begin(), readers_.end(),
            [ntotqstrs,queryblocks,querypmbegs,querypmends]
                (std::unique_ptr<TdDataReader>& p)
            {
                if(p)
                    p->Notify(
                        TdDataReader::tdrmsgGetData,
                        ntotqstrs,
                        queryblocks, querypmbegs, querypmends);
            }
        );
    };

    if(rewind) {
        nit = 0;
        std::for_each(readers_.begin(), readers_.end(),
            [](std::unique_ptr<TdDataReader>& p) {if(p) p->Rewind();}
        );
    }

    //all round has passed: broadcast a message
    if(nit == 0) lfGetDataBcst();

    if(rewind) {
        nit = (size_t)(-1);
        return false;//no wait for data
    }

    if(nit == (size_t)(-1)) nit = 0;

    for(size_t npr = nit; nit < readers_.size() && !ret;)
    {
        if(readers_[nit]) {
            TdDataReader* trd = readers_[nit].get();
            ret = trd->Wait(TdDataReader::tdrrespmsgDataReady, TdDataReader::tdrrespmsgNoData);
            if(ret == TREADER_MSG_ERROR)
                throw MYRUNTIME_ERROR(preamb + "Reader terminated with errors.");
            if(ret != TdDataReader::tdrrespmsgDataReady && ret != TdDataReader::tdrrespmsgNoData)
                throw MYRUNTIME_ERROR(preamb + "Invalid response from Reader.");
            ret = (ret != TdDataReader::tdrrespmsgNoData);
            trd->GetbdbCdata(
                bdbCdescs, bdbCpmbeg, bdbCpmend, bdbCNdxpmbeg, bdbCNdxpmend,  tscnt, lastchunk);
        }
        if(readers_.size() <= (++nit)) nit = 0;
        if(nit == npr/* || nit == 0*/) break;//cycled over; comment revised
        //revision: all round has passed: broadcast a message
        if(nit == 0 && !ret) lfGetDataBcst();
    }

    return (ret);
}



// =========================================================================
// CreateAlnWriter: create a thread for writing results to files
//
void JobDispatcher::CreateAlnWriter( 
    const char* outdirname,
    const std::vector<std::string>& dnamelist)
{
    MYMSG("JobDispatcher::CreateAlnWriter", 3);
    writer_ = new TdAlnWriter(outdirname, dnamelist);
    if( writer_ == NULL )
        throw MYRUNTIME_ERROR(
        "JobDispatcher::CreateAlnWriter: Failed to create the writer thread.");
}

// -------------------------------------------------------------------------
// NotifyAlnWriter: notify the writer of the complete results for a query
//
void JobDispatcher::NotifyAlnWriter()
{
    MYMSG("JobDispatcher::NotifyAlnWriter", 3);
    if(writer_) {
        int rspcode = writer_->GetResponse();
        if(rspcode == WRITERTHREAD_MSG_ERROR || 
           rspcode == TdAlnWriter::wrttrespmsgTerminating)
            throw MYRUNTIME_ERROR(
            "JobDispatcher::NotifyAlnWriter: Results writer terminated with errors.");
        writer_->Notify(TdAlnWriter::wrtthreadmsgWrite);
    }
}

// -------------------------------------------------------------------------
// WaitForAlnWriterToFinish: notify the writer about the process end and 
// wait for it to finish writings
//
void JobDispatcher::WaitForAlnWriterToFinish(bool error)
{
    char msgbuf[BUF_MAX];

    MYMSGBEGl(3)
        sprintf(msgbuf, "JobDispatcher::WaitForAlnWriterToFinish: Snd Msg %d",
            TdAlnWriter::wrtthreadmsgTerminate);
        MYMSG(msgbuf,3);
    MYMSGENDl

    if(writer_ == NULL) {
        MYMSGBEGl(3)
            MYMSG("JobDispatcher::WaitForAlnWriterToFinish: Null Writer.",3);
        MYMSGENDl
        return;
    }

    int rsp = WRITERTHREAD_MSG_UNSET;

    if(!error)
        rsp = writer_->WaitDone();

    if(rsp == WRITERTHREAD_MSG_ERROR) {
        MYMSG("JobDispatcher::WaitForAlnWriterToFinish: "
            "Writer terminated with ERRORS.", 1);
        return;
    }

    writer_->Notify(TdAlnWriter::wrtthreadmsgTerminate);
    //NOTE: do not wait, as the writer might have finished
//     int rsp = writer_->Wait(AlnWriter::wrttrespmsgTerminating);
//     if(rsp != AlnWriter::wrttrespmsgTerminating) {
//         throw MYRUNTIME_ERROR(
//         "JobDispatcher::WaitForAlnWriterToFinish: Invalid response from the writer.");
//     }
//     MYMSGBEGl(3)
//         sprintf(msgbuf,"JobDispatcher::WaitForAlnWriterToFinish: Rcv Msg %d",rsp);
//         MYMSG(msgbuf,3);
//     MYMSGENDl
}



// =========================================================================
// CreateClustWriter: create a thread for clustering and writing
//
void JobDispatcher::CreateClustWriter( 
    const char* outdirname,
    const std::vector<std::string>& clustlist,
    const std::vector<std::string>& devnames,
    const int nmaxchunkqueries,
    const int nagents)
{
    MYMSG("JobDispatcher::CreateClustWriter", 3);
    clustwriter_ = new TdClustWriter(outdirname, clustlist, devnames, nmaxchunkqueries, nagents);
    if(clustwriter_ == NULL)
        throw MYRUNTIME_ERROR(
        "JobDispatcher::CreateClustWriter: Failed to create the writer thread.");
}

// -------------------------------------------------------------------------
// NotifyClustWriter: notify the clusterer of the complete results
//
void JobDispatcher::NotifyClustWriter()
{
    MYMSG("JobDispatcher::NotifyClustWriter", 3);
    if(clustwriter_) {
        int rspcode = clustwriter_->GetResponse();
        if(rspcode == CLSTWRTTHREAD_MSG_ERROR || 
           rspcode == TdClustWriter::clstwrtrespmsgTerminating)
            throw MYRUNTIME_ERROR(
            "JobDispatcher::NotifyClustWriter: Writer terminated with errors.");
        clustwriter_->Notify(TdClustWriter::clstwrthreadmsgWrite);
    }
}

// -------------------------------------------------------------------------
// WaitForClustWriterToFinish: notify the clusterer of the process end and 
// wait for it to finish
//
void JobDispatcher::WaitForClustWriterToFinish(bool error)
{
    char msgbuf[BUF_MAX];

    MYMSGBEGl(3)
        sprintf(msgbuf, "JobDispatcher::WaitForClustWriterToFinish: Snd Msg %d",
            TdClustWriter::clstwrthreadmsgTerminate);
        MYMSG(msgbuf,3);
    MYMSGENDl

    if(clustwriter_ == NULL) {
        MYMSGBEGl(3)
            MYMSG("JobDispatcher::WaitForClustWriterToFinish: Null Writer.",3);
        MYMSGENDl
        return;
    }

    int rsp = CLSTWRTTHREAD_MSG_UNSET;

    if(!error) rsp = clustwriter_->WaitDone();

    if(rsp == CLSTWRTTHREAD_MSG_ERROR) {
        MYMSG("JobDispatcher::WaitForClustWriterToFinish: Writer terminated with ERRORS.", 1);
        return;
    }

    //clstwrthreadmsgWrite also terminates execution once done:
    if(error) clustwriter_->Notify(TdClustWriter::clstwrthreadmsgTerminate);
    else clustwriter_->Notify(TdClustWriter::clstwrthreadmsgWrite);
    //NOTE: do not wait as it might have finished
}



// =========================================================================
// CreateDevMemoryConfigs: create memory configurations for all devices
//
void JobDispatcher::CreateDevMemoryConfigs(size_t nareasperdevice)
{
    MYMSG("JobDispatcher::CreateDevMemoryConfigs", 3);
    static const std::string preamb = "JobDispatcher::CreateDevMemoryConfigs: ";

    for(int tid = 0; tid < DEVPROPs.GetNDevices(); tid++) {
        const DeviceProperties* dprop = DEVPROPs.GetDevicePropertiesAt(tid);

        if(dprop == NULL)
            continue;

        CuDeviceMemory* dmem = new CuDeviceMemory( 
            *dprop,
            dprop->reqmem_,
            (int)nareasperdevice
        );

        if(dmem == NULL) {
            warning((preamb + 
                "Not enough memory to create all device memory configurations." 
                "Skipping the rest of devices starting with " + 
                std::to_string(tid)).c_str());
            break;
        }

        memdevs_.push_back(dmem);
        dmem->CacheCompleteData();

    }
}



// =========================================================================
// CreateWorkerThreads: create worker threads on the host side
//
void JobDispatcher::CreateWorkerThreads(
    size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs)
{
    MYMSG("JobDispatcher::CreateWorkerThreads", 3);
    static const std::string preamb = "JobDispatcher::CreateWorkerThreads: ";

    int tid = 0;

    for(int dmc = 0; dmc < (int)memdevs_.size(); dmc++) {

        if(memdevs_[dmc] == NULL)
            throw MYRUNTIME_ERROR(preamb + "Null device memory object.");

        for(int ano = 0; ano < memdevs_[dmc]->GetNAreas(); ano++, tid++) {

            TdCommutator* t = new TdCommutator(
                memdevs_.size(),
                tid, memdevs_[dmc], ano, GetAlnWriter(), GetClustWriter(),
                chunkdatasize, chunkdatalen, chunknstrs
            );

            if(t == NULL)
                throw MYRUNTIME_ERROR(preamb + 
                "Not enough memory to create all required worker threads.");

            hostworkers_.push_back(t);
        }
    }
}

// -------------------------------------------------------------------------
// SubmitWorkerJob: submit a job for the given worker
//
void JobDispatcher::SubmitWorkerJob(
    int tid, int chunkno, bool lastchunk,
    bool newsetqrs,
    int qrysernrbeg,
    float scorethld,
    char** queryndxpmbeg, char** queryndxpmend,
    const char** querydesc, char** querypmbeg, char** querypmend,
    const char** bdbCdesc, char** bdbCpmbeg, char** bdbCpmend,
    char** bdbCndxpmbeg, char** bdbCndxpmend,
    TSCounterVar* qrstscnt, TSCounterVar* tscnt)
{
    char msgbuf[BUF_MAX];

    MYMSGBEGl(3)
        sprintf(msgbuf, "JobDispatcher::SubmitWorkerJob: Snd Msg %d Adr %d [no wait]",
            TdCommutator::tthreadmsgProcessNewData, tid);
        MYMSG(msgbuf,3);
    MYMSGENDl

    TdCommutator* tdc = hostworkers_[tid];

    if(tdc == NULL)
        throw MYRUNTIME_ERROR(
        "JobDispatcher::SubmitWorkerJob: Null worker thread.");

    tdc->SetMstrQueryBDbdata(
        chunkno,
        lastchunk,
        newsetqrs,
        qrysernrbeg,
        scorethld,
        queryndxpmbeg, queryndxpmend,
        querydesc, querypmbeg, querypmend,
        bdbCdesc, bdbCpmbeg, bdbCpmend,
        bdbCndxpmbeg, bdbCndxpmend,
        qrstscnt, tscnt);

    tdc->Notify(TdCommutator::tthreadmsgProcessNewData, tid);
}

// -------------------------------------------------------------------------
// WaitForWorker: wait until the worker can accept new portion of data
//
void JobDispatcher::WaitForWorker(int tid)
{
    char msgbuf[BUF_MAX];

    MYMSGBEGl(3)
        sprintf(msgbuf, "JobDispatcher::WaitForWorker: Wrk %d",tid);
        MYMSG(msgbuf,3);
    MYMSGENDl

    TdCommutator* tdc = hostworkers_[tid];

    if(tdc == NULL) {
        sprintf(msgbuf,"JobDispatcher::WaitForWorker: "
            "Null Worker %d. Job not submitted",tid);
        warning(msgbuf);
        return;
    }

    //wait until data can be accessed and get the last response if available
    int rsp = tdc->waitForDataAccess();

    if(rsp == THREAD_MSG_ERROR) {
        sprintf(msgbuf, "JobDispatcher::WaitForWorker: "
            "Worker %d terminated with errors", tid);
        throw MYRUNTIME_ERROR(msgbuf);
    }

    //wait until the worker has finished processing
    tdc->IsIdle(NULL/*mstr_set_data_empty*/, true/*wait*/);
}

// -------------------------------------------------------------------------
// ProbeWorker: check for whether the worker is in idle state
//
void JobDispatcher::ProbeWorker(int tid)
{
    char msgbuf[BUF_MAX];
    MYMSGBEGl(3)
        sprintf(msgbuf, "JobDispatcher::ProbeWorker: Snd Msg %d Adr %d",
            TdCommutator::tthreadmsgProbe, tid);
        MYMSG(msgbuf,3);
    MYMSGENDl

    TdCommutator* tdc = hostworkers_[tid];

    if(tdc == NULL) {
        MYMSGBEGl(1)
            sprintf(msgbuf,"JobDispatcher::ProbeWorker: Null Worker %d",tid);
            MYMSG(msgbuf, 1);
        MYMSGENDl
        return;
    }

    //first, make sure the worker is idle and there are no pending jobs
    bool idle = false, data_empty = false;

    for(; !idle || !data_empty;) 
        idle = tdc->IsIdle(&data_empty, true/*wait*/);

    //then, send a probe message
    tdc->Notify(TdCommutator::tthreadmsgProbe, tid);

    int rsp = tdc->Wait(TdCommutator::ttrespmsgProbed);

    if(rsp != TdCommutator::ttrespmsgProbed) {
        throw MYRUNTIME_ERROR(
        "JobDispatcher::ProbeWorker: Invalid response from a worker thread.");
    }

    MYMSGBEGl(3)
        sprintf(msgbuf, "JobDispatcher::ProbeWorker: Rcv Msg %d Adr %d",
            rsp, tid);
        MYMSG(msgbuf, 3);
    MYMSGENDl
}

// -------------------------------------------------------------------------
// WaitForAllWorkersToFinish: wait until all workers become idle;
//
inline
void JobDispatcher::WaitForAllWorkersToFinish()
{
    for(int tid = 0; tid < (int)hostworkers_.size(); tid++)
        ProbeWorker(tid);
}

// -------------------------------------------------------------------------
// TerminateWorker: terminate a worker
//
void JobDispatcher::TerminateWorker(int tid)
{
    char msgbuf[BUF_MAX];

    MYMSGBEGl(3)
        sprintf(msgbuf, "JobDispatcher::TerminateWorker: Snd Msg %d Adr %d [no wait]",
                TdCommutator::tthreadmsgTerminate, tid);
        MYMSG(msgbuf,3);
    MYMSGENDl

    TdCommutator* tdc = hostworkers_[tid];

    if(tdc == NULL) {
        MYMSGBEGl(3)
            sprintf(msgbuf, "JobDispatcher::TerminateWorker: Null Worker %d", tid);
            MYMSG(msgbuf, 3);
        MYMSGENDl
        return;
    }

    tdc->Notify(TdCommutator::tthreadmsgTerminate, tid);
    //do not wait for a response
}

// -------------------------------------------------------------------------
// TerminateAllWorkers: terminate all workers
//
void JobDispatcher::TerminateAllWorkers()
{
    for(int tid = 0; tid < (int)hostworkers_.size(); tid++)
        TerminateWorker(tid);
}



// =========================================================================
// GetAvailableWorker: identify a worker ready to accept new data for 
// processing
//
void JobDispatcher::GetAvailableWorker(int* tid)
{
    MYMSG("JobDispatcher::GetAvailableWorker", 7);

    bool alltermed = true;//all workers terminated

    if(tid == NULL)
        return;

    *tid = JDSP_WORKER_BUSY;

    //first, try to find a waiting worker;
    //if not that, try to find a worker with an empty data slot
    for(int ptid = 0; ptid < (int)hostworkers_.size(); ptid++ ) {
        //iterate over all worker to check availability
        TdCommutator* tdc = hostworkers_[ptid];

        if( tdc == NULL )
            continue;

        alltermed = false;

        bool idle, data_empty;
        idle = tdc->IsIdle(&data_empty);

        if(data_empty && (idle || *tid < 0)) {
            *tid = ptid;
            if(idle)
                //the worker is ready to accept new data and perform computation
                break;
        }
    }

    if(alltermed)
        *tid = JDSP_WORKER_NONE;

    MYMSGBEGl(5)
        char msgbuf[BUF_MAX];
        sprintf(msgbuf,"JobDispatcher::GetAvailableWorker: Wrk %d",*tid);
        MYMSG(msgbuf,5);
    MYMSGENDl
}

// -------------------------------------------------------------------------
// WaitForAvailableWorker: wait until a worker becomes ready to process data
// or accept new data for processing
inline
void JobDispatcher::WaitForAvailableWorker(int* tid)
{
    MYMSG("JobDispatcher::WaitForAvailableWorker", 7);

    if(tid == NULL)
        return;

    for(*tid = JDSP_WORKER_BUSY; *tid == JDSP_WORKER_BUSY; 
        GetAvailableWorker(tid));

    MYMSGBEGl(3)
        char msgbuf[BUF_MAX];
        sprintf(msgbuf, "JobDispatcher::WaitForAvailableWorker: Wrk %d", *tid);
        MYMSG(msgbuf, 3);
    MYMSGENDl
}

// -------------------------------------------------------------------------
// GetNextWorker: get the next worker irrespective of its busy or idle 
// status;
// NOTE: tid, the address of the current busy worker should be initialized
inline
void JobDispatcher::GetNextWorker(int* tid)
{
    MYMSG("JobDispatcher::GetNextWorker", 7);
    if(tid == NULL)
        return;

    int initid = (0 <= *tid && *tid < (int)hostworkers_.size())? *tid: 0;

    *tid = JDSP_WORKER_NONE;

    for(int ptid = initid+1; ; ptid++ ) {

        if((int)hostworkers_.size() <= ptid)
            ptid = 0;

        if((int)hostworkers_.size() <= ptid)
            break;

        TdCommutator* tdc = hostworkers_[ptid];

        if(tdc == NULL) {
            if(ptid == initid)
                break;
            else
                continue;
        }

        MYMSGBEGl(3)
            char msgbuf[BUF_MAX];
            sprintf(msgbuf, "JobDispatcher::GetNextWorker: Wrk %d", ptid);
            MYMSG(msgbuf, 3);
        MYMSGENDl

        *tid = ptid;
        break;
    }
}

// =========================================================================
// =========================================================================



// =========================================================================
// Run: starting point for structure search and alignment
//
void JobDispatcher::Run()
{
    MYMSG( "JobDispatcher::Run", 3 );
    static const std::string preamb = "JobDispatcher::Run: ";

    myruntime_error mre;
    char msgbuf[BUF_MAX];
    size_t chunkdatasize = 0UL;
    size_t chunkdatalen = 0UL;//maximum total length of db structure positions that can be processed at once
    size_t chunknstrs = 0UL;//maximum number of db structures allowed to be processed at once
    int totqrsposs = CLOptions::GetDEV_QRES_PER_CHUNK();

    const float scorethld = CLOptions::GetO_S();
    std::unique_ptr<char**[]> querydescs;//query descriptions
    std::unique_ptr<char**[]> querypmbegs;//encapsulated query structure data
    std::unique_ptr<char**[]> querypmends;//encapsulated query structure data
    std::unique_ptr<char**[]> queryndxpmbegs;//query indexed data
    std::unique_ptr<char**[]> queryndxpmends;//query indexed data
    std::vector<int> qrysernrbegs;//query beginning serial numbers for each block
    std::unique_ptr<TSCounterVar*[]> qrstscnt;//counter of how many agents read the data
    // TSCounterVar* qrstscnt;//counter of how many agents read the data
    bool qrslastchunk = false;//indicator of the last chunk for the queries
    bool newsetqrs = true;

    size_t qrschunkdatalen = totqrsposs;
    size_t qrschunkdatasize = PMBatchStrData::GetPMDataSizeUB(qrschunkdatalen);
    size_t qrschunknstrs = (size_t)CLOptions::GetDEV_QRS_PER_CHUNK();

    char** bdbCdescs;//structure descriptions
    char** bdbCpmbeg, **bdbCpmend;//encapsulated structure data
    char** bdbCndxpmbeg, **bdbCndxpmend;//encapsulated db index
    TSCounterVar* tscnt;//counter of how many agents read the data
    bool lastchunk = false;//indicator of the last chunk
    char** bdbCdescsNext;//structure descriptions
    char** bdbCpmbegNext, **bdbCpmendNext;//encapsulated structure data
    char** bdbCndxpmbegNext, **bdbCndxpmendNext;//encapsulated db index
    TSCounterVar* tscntNext;//counter of how many agents read the data
    bool lastchunkNext = false;//indicator of the last chunk
    int tid = JDSP_WORKER_BUSY;//worker id
    int didmm = -1;//id of a device with the minimum requested memory

    if(DEVPROPs.GetNDevices() < 1) {
        warning("There is no available device to run the program on.");
        message("Please use a version to run on CPU.");
        return;
    }

    try {
        //INITIALIZATION...
        if((didmm = DEVPROPs.GetDevIdWithMinRequestedMem()) < 0)
            throw MYRUNTIME_ERROR(preamb + "Failed to obtain required device id.");

        //create the results writing thread
        CreateAlnWriter(output_, GetDNamelist());

        //create memory configuration for each device
        CreateDevMemoryConfigs(1/*nareasperdevice*/);

        if(memdevs_.size() < 1)
            throw MYRUNTIME_ERROR(preamb + 
                "Failed to create device memory configuration.");

        //initialize the memory sections of all devices
        // according to the given maximum query length
        for(int dmc = 0; dmc < (int)memdevs_.size(); dmc++ ) {
            if(memdevs_[dmc] == NULL)
                continue;

            size_t szchdata = memdevs_[dmc]->CalcMaxDbDataChunkSize(totqrsposs);

            //get globally valid data chunk size
            if(didmm == dmc) {
                chunkdatasize = szchdata;
                chunkdatalen = memdevs_[dmc]->GetCurrentMaxDbPos();
                chunknstrs = memdevs_[dmc]->GetCurrentMaxNDbStrs();
                //make sure max # structures does not exceed technical specifications
                chunknstrs = PCMIN(chunknstrs,
                    (size_t)memdevs_[dmc]->GetDeviceProp().GridMaxYdim());
            }
        }

        //create worker threads
        CreateWorkerThreads(chunkdatasize, chunkdatalen, chunknstrs);

        if(hostworkers_.size() < 1)
            throw MYRUNTIME_ERROR(preamb + "Failed to create worker threads.");

        CreateQrsReader( 
            -1,//query length unlimited (limited by max total length)
            CLOptions::GetIO_FILEMAP(),
            2 * hostworkers_.size(),//CLOptions::GetIO_NBUFFERS(),
            //nagents: +1 for late processing; a counter for each worker data buffer
            1/* hostworkers_.size() */);

        CreateReader( 
            CLOptions::GetDEV_MAXRLEN(),
            CLOptions::GetIO_FILEMAP(),
            CLOptions::GetIO_NBUFFERS(),
            hostworkers_.size(),//nagents: not updated
            chunkdatasize, chunkdatalen, chunknstrs);

        MYMSGBEGl(1)
            char strbuf[BUF_MAX];
            sprintf(strbuf, "Processing in chunks of: "
                "size %zu length %zu #structures %zu",
                chunkdatasize, chunkdatalen, chunknstrs);
            MYMSG(strbuf, 1);
        MYMSGENDl

        if(chunkdatasize < 1 || chunkdatalen < 1 || chunknstrs < 1)
            throw MYRUNTIME_ERROR(preamb + "Invalid calculated data chunk size.");

        SetChunkDataAttributesReaders(chunkdatasize, chunkdatalen, chunknstrs);

        querydescs.reset(new char**[hostworkers_.size()]);
        querypmbegs.reset(new char**[hostworkers_.size()]);
        querypmends.reset(new char**[hostworkers_.size()]);
        queryndxpmbegs.reset(new char**[hostworkers_.size()]);
        queryndxpmends.reset(new char**[hostworkers_.size()]);
        qrysernrbegs.resize(hostworkers_.size(), 0);
        qrstscnt.reset(new TSCounterVar*[hostworkers_.size()]);

        if(!(queryndxpmbegs) || !(queryndxpmends) ||
           !(querydescs) || !(querypmbegs) || !(querypmends) || !(qrstscnt))
            throw MYRUNTIME_ERROR(preamb + "Not enough memory.");

        size_t ntotqstrs = 0;
        std::vector<std::tuple<size_t,int,int>> valnwparams;
        valnwparams.reserve(hostworkers_.size());


        //QUERY BLOCKS..
        for(int qblk = 0;;)
        {
            newsetqrs = true;

            GetQrsReader()->SetChunkDataAttributes(
                qrschunkdatasize, qrschunkdatalen, qrschunknstrs);

            memset(querydescs.get(), 0, hostworkers_.size() * sizeof(void*));
            memset(querypmbegs.get(), 0, hostworkers_.size() * sizeof(void*));
            memset(querypmends.get(), 0, hostworkers_.size() * sizeof(void*));
            memset(queryndxpmbegs.get(), 0, hostworkers_.size() * sizeof(void*));
            memset(queryndxpmends.get(), 0, hostworkers_.size() * sizeof(void*));
            memset(qrstscnt.get(), 0, hostworkers_.size() * sizeof(void*));

            valnwparams.clear();

            int wrk = 0;
            for(; wrk < (int)hostworkers_.size(); wrk++) {
                qrysernrbegs[wrk] = ntotqstrs;

                if(!GetDataFromReader(
                    GetQrsReader(),
                    querydescs[wrk], querypmbegs[wrk], querypmends[wrk],
                    queryndxpmbegs[wrk], queryndxpmends[wrk],
                    qrstscnt[wrk], &qrslastchunk))
                    //leave if no data
                    break;

                size_t nqstrs = PMBatchStrData::GetNoStructs(querypmbegs[wrk], querypmends[wrk]);
                size_t nqposs = PMBatchStrData::GetNoPosits(querypmbegs[wrk], querypmends[wrk]);

                MYMSGBEGl(1)
                    sprintf(msgbuf,"%s[*******] Query BLOCK No. %d: "
                            "%zu positions (%zu strs.) assigned to worker %d",
                            NL, qblk, nqposs, nqstrs, wrk);
                    MYMSG(msgbuf,1);
                MYMSGENDl

                ntotqstrs += (int)nqstrs;
                qblk++;

                //for every new query block, make sure the writer's counters are 
                //incremented in advance for each NON-EMPTY chunk of references!
                if(nqstrs)
                    valnwparams.push_back(
                        std::make_tuple(nqstrs, qrysernrbegs[wrk], qrysernrbegs[wrk]+nqstrs-1));
            }

            if(wrk < 1) break;


            bool datapresent = GetReferenceData(
                    bdbCdescs, bdbCpmbeg, bdbCpmend,
                    bdbCndxpmbeg, bdbCndxpmend,
                    tscnt, &lastchunk, false/*no rewind*/,
                    ntotqstrs,
                    0/*wrk*/, NULL/*querypmbegs.get()*/, NULL/*querypmends.get()*/);

            bool datapresentNext = datapresent;

            //CHUNKS..
            for(int chkno = 0; tid != JDSP_WORKER_NONE;
                chkno++,
                datapresent = datapresentNext,
                bdbCdescs = bdbCdescsNext,
                bdbCpmbeg = bdbCpmbegNext,
                bdbCpmend = bdbCpmendNext,
                bdbCndxpmbeg = bdbCndxpmbegNext,
                bdbCndxpmend = bdbCndxpmendNext,
                tscnt = tscntNext)
            {
                datapresentNext = GetReferenceData(
                    bdbCdescsNext, bdbCpmbegNext, bdbCpmendNext,
                    bdbCndxpmbegNext, bdbCndxpmendNext,
                    tscntNext, &lastchunkNext, false/*no rewind*/,
                    ntotqstrs,
                    0/*wrk*/, NULL/*querypmbegs.get()*/, NULL/*querypmends.get()*/);

                lastchunk = false;
                if(!datapresentNext) lastchunk = true;

                size_t nCstrs = PMBatchStrData::GetNoStructs(bdbCpmbeg, bdbCpmend);
                size_t nposits = PMBatchStrData::GetNoPosits(bdbCpmbeg, bdbCpmend);

                //if the chunk isn't empty, increment the writer's counters for the query blocks!
                if(datapresent && chkno == 0)
                    for(auto&& ps: valnwparams)
                        GetAlnWriter()->IncreaseQueryNParts(std::get<1>(ps),std::get<2>(ps));

                //reset the counter for queries if no chunk with data:
                if(!datapresent && chkno == 0)
                    for(int w = 0; w < (int)hostworkers_.size(); w++)
                        qrstscnt[w]->reset();

                //if no data, finish with the chunk:
                if(!datapresent) break;

                //NOTE: wait for all workers to finish; new chunk implies new data!
                for(tid = 0; tid < (int)hostworkers_.size(); tid++)
                    WaitForWorker(tid);

                MYMSGBEGl(1)
                    sprintf(msgbuf,"%s[=======] Processing database CHUNK No. %d: "
                            "%zu ref. positions (%zu strs.)",
                            NL, chkno, nposits, nCstrs);
                    MYMSG(msgbuf,1);
                MYMSGENDl

                //QUERIES passed to WORKERS...
                for(tid = 0; tid < (int)hostworkers_.size(); tid++)
                {
                    ProcessPart(
                        tid, chkno+1, lastchunk,
                        newsetqrs,
                        qrysernrbegs[tid],
                        scorethld,
                        queryndxpmbegs[tid], queryndxpmends[tid],
                        (const char**)querydescs[tid], querypmbegs[tid], querypmends[tid],
                        (const char**)bdbCdescs, bdbCpmbeg, bdbCpmend,
                        bdbCndxpmbeg, bdbCndxpmend,
                        qrstscnt[tid], tscnt);
                }

                newsetqrs = false;
            }//for(int chkno...)//CHUNKS

            //start from the beginning of the reference data
            GetReferenceData(
                bdbCdescsNext, bdbCpmbegNext, bdbCpmendNext,
                bdbCndxpmbegNext, bdbCndxpmendNext,
                tscntNext, &lastchunkNext, true/*do rewind*/,
                ntotqstrs,
                0/*wrk*/, NULL/*querypmbegs.get()*/, NULL/*querypmends.get()*/);

        }//for(int qblk...)//query blocks

        //wait for the workers to finish
        for(tid = 0; tid < (int)hostworkers_.size(); tid++)
            WaitForWorker(tid);

        WaitForAllWorkersToFinish();

    } catch(myexception const& ex) {
        mre = ex;
    } catch(...) {
        mre = myruntime_error("Exception caught.");
    }

    //workers' children may still have job; wait for the writer to finish first
    WaitForAlnWriterToFinish(mre.isset());

    TerminateAllWorkers();

    if(mre.isset())
        throw mre;

    MYMSG("Done.",1);
}

// -------------------------------------------------------------------------
// ProcessPart: process part of data, multiple reference structures for 
// multiple queries;
// tid, worker id;
// chunkno, chunk serial number;
// lastchunk, indicator whether the chunk is last;
// newsetqrs, indicator whether the set of queries is a new set;
// qrysernrbeg, serial number of the first query in the chunk;
// scorethld, tmscore threshold;
// querydesc, descriptions for each query in the data chunk;
// querypmbeg, beginnings of the query structure data fields in the chunk;
// querypmend, ends of the query structure data fields in the chunk;
// bdbCdesc, descriptions for each db structure in the data chunk;
// bdbCpmbeg, beginnings of the db structure data fields in the chunk;
// bdbCpmend, ends of the db structure data fields in the chunk;
// tscnt, thread-safe counter of processing agents;
//
void JobDispatcher::ProcessPart(
    int tid, int chunkno, bool lastchunk,
    bool newsetqrs,
    int qrysernrbeg,
    float scorethld,
    char** queryndxpmbeg, char** queryndxpmend,
    const char** querydesc, char** querypmbeg, char** querypmend,
    const char** bdbCdesc, char** bdbCpmbeg, char** bdbCpmend,
    char** bdbCndxpmbeg, char** bdbCndxpmend,
    TSCounterVar* qrstscnt, TSCounterVar* tscnt)
{
    MYMSG("JobDispatcher::ProcessPart", 4);
    static const std::string preamb = "JobDispatcher::ProcessPart: ";

#ifdef __DEBUG__
    if(!querydesc || !querypmbeg || !querypmend || !queryndxpmbeg || !queryndxpmend)
        throw MYRUNTIME_ERROR(preamb + "Null query addresses.");
    if(!bdbCdesc || !bdbCpmbeg || !bdbCpmend )
        throw MYRUNTIME_ERROR(preamb + "Null arguments.");
    if(!qrstscnt || !tscnt)
        throw MYRUNTIME_ERROR(preamb + "Null counters.");
#endif

    //bool queryinf = PMBatchStrData::ContainsData(querypmbeg, querypmend);
    bool bdbCinf = PMBatchStrData::ContainsData(bdbCpmbeg, bdbCpmend);

    //queries can be empty (NULL) and still a worker has decrease the 
    // associated counter
    if(writer_ && (/*!queryinf || */!bdbCinf))
        throw MYRUNTIME_ERROR(preamb + "Structure data is empty.");

    SubmitWorkerJob(
        tid, chunkno, lastchunk,
        newsetqrs,
        qrysernrbeg,
        scorethld,
        queryndxpmbeg, queryndxpmend,
        querydesc, querypmbeg, querypmend,
        bdbCdesc, bdbCpmbeg, bdbCpmend,
        bdbCndxpmbeg, bdbCndxpmend,
        qrstscnt, tscnt);
}

// =========================================================================
// RunClust: run structure clustering
//
void JobDispatcher::RunClust()
{
    MYMSG( "JobDispatcher::RunClust", 3 );
    static const std::string preamb = "JobDispatcher::RunClust: ";

    myruntime_error mre;
    char msgbuf[BUF_MAX];
    size_t chunkdatasize = 0UL;
    size_t chunkdatalen = 0UL;//maximum total length of db structure positions that can be processed at once
    size_t chunknstrs = 0UL;//maximum number of db structures allowed to be processed at once
    int totqrsposs = CLOptions::GetDEV_QRES_PER_CHUNK();

    const float scorethld = CLOptions::GetO_S();
    std::vector<std::string> devnames;//list of device names
    std::unique_ptr<char**[]> querydescs;//query descriptions
    std::unique_ptr<char**[]> querypmbegs;//encapsulated query structure data
    std::unique_ptr<char**[]> querypmends;//encapsulated query structure data
    std::unique_ptr<char**[]> queryndxpmbegs;//query indexed data
    std::unique_ptr<char**[]> queryndxpmends;//query indexed data
    std::vector<int> qrysernrbegs;//query beginning serial numbers for each block
    std::unique_ptr<TSCounterVar*[]> qrstscnt;//counter of how many agents read the data
    // TSCounterVar* qrstscnt;//counter of how many agents read the data
    bool qrslastchunk = false;//indicator of the last chunk for the queries
    bool newsetqrs = true;

    size_t qrschunkdatalen = totqrsposs;
    size_t qrschunkdatasize = PMBatchStrData::GetPMDataSizeUB(qrschunkdatalen);
    size_t qrschunknstrs = (size_t)CLOptions::GetDEV_QRS_PER_CHUNK();

    char** bdbCdescs;//structure descriptions
    char** bdbCpmbeg, **bdbCpmend;//encapsulated structure data
    char** bdbCndxpmbeg, **bdbCndxpmend;//encapsulated db index
    TSCounterVar* tscnt;//counter of how many agents read the data
    bool lastchunk = false;//indicator of the last chunk
    char** bdbCdescsNext;//structure descriptions
    char** bdbCpmbegNext, **bdbCpmendNext;//encapsulated structure data
    char** bdbCndxpmbegNext, **bdbCndxpmendNext;//encapsulated db index
    TSCounterVar* tscntNext;//counter of how many agents read the data
    bool lastchunkNext = false;//indicator of the last chunk
    int tid = JDSP_WORKER_BUSY;//worker id
    int didmm = -1;//id of a device with the minimum requested memory
    int nagents = 0;//number of workers

    if(DEVPROPs.GetNDevices() < 1) {
        warning("There is no available device to run the program on.");
        message("Please use a version to run on CPU.");
        return;
    }

    try {
        //INITIALIZATION...
        if((didmm = DEVPROPs.GetDevIdWithMinRequestedMem()) < 0)
            throw MYRUNTIME_ERROR(preamb + "Failed to obtain required device id.");

        for(int di = 0; di < DEVPROPs.GetNDevices(); di++)
            if(DEVPROPs.GetDevicePropertiesAt(di))
                devnames.push_back(DEVPROPs.GetDevicePropertiesAt(di)->name_);

        //create memory configuration for each device
        CreateDevMemoryConfigs(1/*nareasperdevice*/);

        if(memdevs_.size() < 1)
            throw MYRUNTIME_ERROR(preamb + 
                "Failed to create device memory configuration.");

        //initialize the memory sections of all devices
        // according to the given maximum query length
        for(int dmc = 0; dmc < (int)memdevs_.size(); dmc++ ) {
            if(memdevs_[dmc] == NULL)
                continue;

            size_t szchdata = memdevs_[dmc]->CalcMaxDbDataChunkSize(totqrsposs);
            nagents += memdevs_[dmc]->GetNAreas();

            //get globally valid data chunk size
            if(didmm == dmc) {
                chunkdatasize = szchdata;
                chunkdatalen = memdevs_[dmc]->GetCurrentMaxDbPos();
                chunknstrs = memdevs_[dmc]->GetCurrentMaxNDbStrs();
                //make sure max # structures does not exceed technical specifications
                chunknstrs = PCMIN(chunknstrs,
                    (size_t)memdevs_[dmc]->GetDeviceProp().GridMaxYdim());
            }
        }

        //create the results writing thread
        CreateClustWriter(output_, GetClustList(), devnames, qrschunknstrs, nagents);

        //create worker threads
        CreateWorkerThreads(chunkdatasize, chunkdatalen, chunknstrs);

        if(hostworkers_.size() < 1)
            throw MYRUNTIME_ERROR(preamb + "Failed to create worker threads.");

        if((int)hostworkers_.size() != nagents)
            throw MYRUNTIME_ERROR(preamb + "Inconsistent number of workers.");

        CreateQrsReader( 
            -1,//query length unlimited (limited by max total length)
            CLOptions::GetIO_FILEMAP(),
            2 * hostworkers_.size(),//CLOptions::GetIO_NBUFFERS(),
            //nagents: +1 for late processing; a counter for each worker data buffer
            1/* hostworkers_.size() */);

        CreateReader( 
            CLOptions::GetDEV_MAXRLEN(),
            CLOptions::GetIO_FILEMAP(),
            CLOptions::GetIO_NBUFFERS(),
            hostworkers_.size(),//nagents: not updated
            chunkdatasize, chunkdatalen, chunknstrs);

        MYMSGBEGl(1)
            char strbuf[BUF_MAX];
            sprintf(strbuf, "Processing in chunks of: "
                "size %zu length %zu #structures %zu",
                chunkdatasize, chunkdatalen, chunknstrs);
            MYMSG(strbuf, 1);
        MYMSGENDl

        if(chunkdatasize < 1 || chunkdatalen < 1 || chunknstrs < 1)
            throw MYRUNTIME_ERROR(preamb + "Invalid calculated data chunk size.");

        SetChunkDataAttributesReaders(chunkdatasize, chunkdatalen, chunknstrs);

        querydescs.reset(new char**[hostworkers_.size()]);
        querypmbegs.reset(new char**[hostworkers_.size()]);
        querypmends.reset(new char**[hostworkers_.size()]);
        queryndxpmbegs.reset(new char**[hostworkers_.size()]);
        queryndxpmends.reset(new char**[hostworkers_.size()]);
        qrysernrbegs.resize(hostworkers_.size(), 0);
        qrstscnt.reset(new TSCounterVar*[hostworkers_.size()]);

        if(!(queryndxpmbegs) || !(queryndxpmends) ||
           !(querydescs) || !(querypmbegs) || !(querypmends) || !(qrstscnt))
            throw MYRUNTIME_ERROR(preamb + "Not enough memory.");

        size_t ntotqstrs = 0;
        std::vector<std::tuple<size_t,int,int>> valnwparams;
        valnwparams.reserve(hostworkers_.size());


        //QUERY BLOCKS..
        for(int qblk = 0;;)
        {
            newsetqrs = true;

            GetQrsReader()->SetChunkDataAttributes(
                qrschunkdatasize, qrschunkdatalen, qrschunknstrs);

            memset(querydescs.get(), 0, hostworkers_.size() * sizeof(void*));
            memset(querypmbegs.get(), 0, hostworkers_.size() * sizeof(void*));
            memset(querypmends.get(), 0, hostworkers_.size() * sizeof(void*));
            memset(queryndxpmbegs.get(), 0, hostworkers_.size() * sizeof(void*));
            memset(queryndxpmends.get(), 0, hostworkers_.size() * sizeof(void*));
            memset(qrstscnt.get(), 0, hostworkers_.size() * sizeof(void*));

            valnwparams.clear();

            int wrk = 0;
            for(; wrk < (int)hostworkers_.size(); wrk++) {
                qrysernrbegs[wrk] = ntotqstrs;

                if(!GetDataFromReader(
                    GetQrsReader(),
                    querydescs[wrk], querypmbegs[wrk], querypmends[wrk],
                    queryndxpmbegs[wrk], queryndxpmends[wrk],
                    qrstscnt[wrk], &qrslastchunk))
                    //leave if no data
                    break;

                size_t nqstrs = PMBatchStrData::GetNoStructs(querypmbegs[wrk], querypmends[wrk]);
                size_t nqposs = PMBatchStrData::GetNoPosits(querypmbegs[wrk], querypmends[wrk]);

                MYMSGBEGl(1)
                    sprintf(msgbuf,"%s[*******] Query BLOCK No. %d: "
                            "%zu strs. (%zu pos.) (worker %d) / %zu processed",
                            NL, qblk, nqstrs, nqposs, wrk, ntotqstrs);
                    MYMSG(msgbuf,1);
                MYMSGENDl

                ntotqstrs += (int)nqstrs;
                qblk++;

                //for every new query block, make sure the writer's counters are 
                //incremented in advance for each NON-EMPTY chunk of references!
                if(nqstrs)
                    valnwparams.push_back(
                        std::make_tuple(nqstrs, qrysernrbegs[wrk], qrysernrbegs[wrk]+nqstrs-1));
            }

            if(wrk < 1) break;


            bool datapresent = GetReferenceData(
                    bdbCdescs, bdbCpmbeg, bdbCpmend,
                    bdbCndxpmbeg, bdbCndxpmend,
                    tscnt, &lastchunk, false/*no rewind*/,
                    ntotqstrs,
                    0/*wrk*/, NULL/*querypmbegs.get()*/, NULL/*querypmends.get()*/);

            bool datapresentNext = datapresent;

            //CHUNKS..
            for(int chkno = 0; tid != JDSP_WORKER_NONE;
                chkno++,
                datapresent = datapresentNext,
                bdbCdescs = bdbCdescsNext,
                bdbCpmbeg = bdbCpmbegNext,
                bdbCpmend = bdbCpmendNext,
                bdbCndxpmbeg = bdbCndxpmbegNext,
                bdbCndxpmend = bdbCndxpmendNext,
                tscnt = tscntNext)
            {
                datapresentNext = GetReferenceData(
                    bdbCdescsNext, bdbCpmbegNext, bdbCpmendNext,
                    bdbCndxpmbegNext, bdbCndxpmendNext,
                    tscntNext, &lastchunkNext, false/*no rewind*/,
                    ntotqstrs,
                    0/*wrk*/, NULL/*querypmbegs.get()*/, NULL/*querypmends.get()*/);

                lastchunk = false;
                if(!datapresentNext) lastchunk = true;

                size_t nCstrs = PMBatchStrData::GetNoStructs(bdbCpmbeg, bdbCpmend);
                size_t nposits = PMBatchStrData::GetNoPosits(bdbCpmbeg, bdbCpmend);

                //NOTE: wait for all workers to finish; new chunk implies new data!
                //NOTE: inserted before parts update!
                for(tid = 0; tid < (int)hostworkers_.size(); tid++)
                    WaitForWorker(tid);

                //NOTE: wait for idle clusterer too: its buffers are block-allocated;
                GetClustWriter()->Wait();

                //if the chunk isn't empty, increment the writer's counters for the query blocks!
                if(datapresent && chkno == 0)
                    for(size_t vi = 0; vi < valnwparams.size(); vi++)
                        GetClustWriter()->IncreaseQueryNParts(
                            vi/*agent*/, std::get<1>(valnwparams[vi]),std::get<2>(valnwparams[vi]));

                //reset the counter for queries if no chunk with data:
                if(!datapresent && chkno == 0)
                    for(int w = 0; w < (int)hostworkers_.size(); w++)
                        qrstscnt[w]->reset();

                //if no data, finish with the chunk:
                if(!datapresent) break;

                // //NOTE: wait for all workers to finish; new chunk implies new data!
                // for(tid = 0; tid < (int)hostworkers_.size(); tid++)
                //     WaitForWorker(tid);

                MYMSGBEGl(2)
                    sprintf(msgbuf,"%s[=======] Processing database CHUNK No. %d: "
                            "%zu ref. positions (%zu strs.)",
                            NL, chkno, nposits, nCstrs);
                    MYMSG(msgbuf,2);
                MYMSGENDl

                //QUERIES passed to WORKERS...
                for(tid = 0; tid < (int)hostworkers_.size(); tid++)
                {
                    ProcessPart(
                        tid, chkno+1, lastchunk,
                        newsetqrs,
                        qrysernrbegs[tid],
                        scorethld,
                        queryndxpmbegs[tid], queryndxpmends[tid],
                        (const char**)querydescs[tid], querypmbegs[tid], querypmends[tid],
                        (const char**)bdbCdescs, bdbCpmbeg, bdbCpmend,
                        bdbCndxpmbeg, bdbCndxpmend,
                        qrstscnt[tid], tscnt);
                }

                newsetqrs = false;
            }//for(int chkno...)//CHUNKS

            //start from the beginning of the reference data
            GetReferenceData(
                bdbCdescsNext, bdbCpmbegNext, bdbCpmendNext,
                bdbCndxpmbegNext, bdbCndxpmendNext,
                tscntNext, &lastchunkNext, true/*do rewind*/,
                ntotqstrs,
                0/*wrk*/, NULL/*querypmbegs.get()*/, NULL/*querypmends.get()*/);

        }//for(int qblk...)//query blocks

        //wait for the workers to finish
        for(tid = 0; tid < (int)hostworkers_.size(); tid++)
            WaitForWorker(tid);

        WaitForAllWorkersToFinish();

    } catch(myexception const& ex) {
        mre = ex;
    } catch(...) {
        mre = myruntime_error("Exception caught.");
    }

    //workers' children may still have job; wait for the writer to finish first
    WaitForClustWriterToFinish(mre.isset());

    TerminateAllWorkers();

    if(mre.isset())
        throw mre;

    MYMSG("Done.",1);
}
