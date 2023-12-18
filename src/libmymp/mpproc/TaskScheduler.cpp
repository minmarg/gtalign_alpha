/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/mybase.h"

#include <stdlib.h>
#include <string.h>

#include <string>
#include <thread>
#include <vector>
#include <tuple>
#include <functional>

#include "libutil/CLOptions.h"
#include "tsafety/TSCounterVar.h"

#include "libgenp/gdats/InputFilelist.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libgenp/gdats/PMBatchStrData.h"
#include "libgenp/gdats/TdDataReader.h"
#include "libgenp/goutp/TdAlnWriter.h"
#include "libmymp/mplayout/MpGlobalMemory.h"
#include "libmymp/mpbatch/MpBatch.h"
#include "TaskScheduler.h"

// _________________________________________________________________________
// Class TaskScheduler
//
// Constructor
//
TaskScheduler::TaskScheduler(
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
    qrsreader_(NULL)
{
    queries_.reset(new InputFilelist(inputlist, sfxlst));
    references_.reset(new InputFilelist(dnamelist, sfxlst));

    if(!queries_ || !references_)
        throw MYRUNTIME_ERROR("TaskScheduler: Failed to construct structure lists.");

    readers_.reserve(16);
}

// Destructor
//
TaskScheduler::~TaskScheduler()
{
    for(int d = 0; d < (int)memcfgs_.size(); d++) {
        if(memcfgs_[d]) {
            delete memcfgs_[d];
            memcfgs_[d] = NULL;
        }
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
// CreateReader: create threads for reading reference data from files
//
void TaskScheduler::CreateReader( 
    int maxstrlen,
    bool mapped,
    int ndatbufs,
    int nagents)
{
    MYMSG("TaskScheduler::CreateReader", 3);
    size_t cputhsread = CLOptions::GetCPU_THREADS_READING();
    size_t nthreads = mymin(cputhsread, references_->GetStrFilelist().size());
    const bool indexed = true;
    for(size_t n = 0; n < nthreads; n++) {
        std::unique_ptr<TdDataReader> tdr(
            new TdDataReader(
                cachedir_,
                GetDNamelist(),
                references_->GetStrFilelist(),
                references_->GetPntFilelist(),
                references_->GetStrFilePositionlist(),
                references_->GetStrFilesizelist(),
                references_->GetStrParenttypelist(),
                references_->GetStrFiletypelist(),
                references_->GetFilendxlist(),
                references_->GetGlobalIds(),
                n/*ndxstartwith*/,
                nthreads/*ndxstep*/,
                maxstrlen, mapped, indexed, ndatbufs, nagents)
        );
        if(!tdr)
            throw MYRUNTIME_ERROR(
            "TaskScheduler::CreateReader: Failed to create reader thread(s).");
        readers_.push_back(std::move(tdr));
    }
}

// -------------------------------------------------------------------------
// CreateQrsReader: create a thread for reading query data from files
//
void TaskScheduler::CreateQrsReader( 
    int maxstrlen,
    bool mapped,
    int ndatbufs,
    int nagents)
{
    MYMSG("TaskScheduler::CreateQrsReader", 3);
    const bool indexed = true;
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
            maxstrlen, mapped, indexed, ndatbufs, nagents);
    if(qrsreader_ == NULL)
        throw MYRUNTIME_ERROR(
        "TaskScheduler::CreateQrsReader: Failed to create the reader for queries.");
}

// -------------------------------------------------------------------------
// GetDataFromReader: request data from a reader and wait for data to be ready;
// return false if there are no data to be read;
//
bool TaskScheduler::GetDataFromReader(
    TdDataReader* reader,
    char**& bdbCdescs, char**& bdbCpmbeg, char**& bdbCpmend,
    TSCounterVar*& tscnt,
    bool* lastchunk)
{
    MYMSG("TaskScheduler::GetDataFromReader", 3);

    if(reader == NULL)
        throw MYRUNTIME_ERROR(
        "TaskScheduler::GetDataFromReader: Null reader object.");

    reader->Notify(TdDataReader::tdrmsgGetData);

    int rsp = reader->Wait(TdDataReader::tdrrespmsgDataReady, TdDataReader::tdrrespmsgNoData);

    if(rsp == TREADER_MSG_ERROR)
        throw MYRUNTIME_ERROR(
        "TaskScheduler::GetDataFromReader: A reader terminated with errors.");

    if(rsp != TdDataReader::tdrrespmsgDataReady && rsp != TdDataReader::tdrrespmsgNoData) {
        throw MYRUNTIME_ERROR(
        "TaskScheduler::GetDataFromReader: Invalid response from a reader.");
    }

    *lastchunk = true;
    rsp = (rsp != TdDataReader::tdrrespmsgNoData);

    reader->GetbdbCdata(bdbCdescs, bdbCpmbeg, bdbCpmend,  tscnt, lastchunk);

    return rsp;
}

// GetDataFromReader: request data from a reader and wait for data to be ready;
// version to get index too;
// return false if there are no data to be read;
//
bool TaskScheduler::GetDataFromReader(
    TdDataReader* reader,
    char**& bdbCdescs, char**& bdbCpmbeg, char**& bdbCpmend,
    char**& bdbCNdxpmbeg, char**& bdbCNdxpmend,
    TSCounterVar*& tscnt,
    bool* lastchunk)
{
    MYMSG("TaskScheduler::GetDataFromReader", 3);

    if(reader == NULL)
        throw MYRUNTIME_ERROR(
        "TaskScheduler::GetDataFromReader: Null reader object.");

    reader->Notify(TdDataReader::tdrmsgGetData);

    int rsp = reader->Wait(TdDataReader::tdrrespmsgDataReady, TdDataReader::tdrrespmsgNoData);

    if(rsp == TREADER_MSG_ERROR)
        throw MYRUNTIME_ERROR(
        "TaskScheduler::GetDataFromReader: A reader terminated with errors.");

    if(rsp != TdDataReader::tdrrespmsgDataReady && rsp != TdDataReader::tdrrespmsgNoData) {
        throw MYRUNTIME_ERROR(
        "TaskScheduler::GetDataFromReader: Invalid response from a reader.");
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
bool TaskScheduler::GetReferenceData(
    char**& bdbCdescs, char**& bdbCpmbeg, char**& bdbCpmend,
    char**& bdbCNdxpmbeg, char**& bdbCNdxpmend,
    TSCounterVar*& tscnt, bool* lastchunk, bool rewind,
    size_t ntotqstrs, int queryblocks,
    char* const * const * const querypmbegs,
    char* const * const * const querypmends)
{
    MYMSG("TaskScheduler::GetReferenceData", 3);
    static const std::string preamb = "TaskScheduler::GetReferenceData ";
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
void TaskScheduler::CreateAlnWriter( 
    const char* outdirname,
    const std::vector<std::string>& dnamelist)
{
    MYMSG("TaskScheduler::CreateAlnWriter", 3);
    writer_ = new TdAlnWriter(outdirname, dnamelist);
    if( writer_ == NULL )
        throw MYRUNTIME_ERROR(
        "TaskScheduler::CreateAlnWriter: Failed to create the writer thread.");
}

// -------------------------------------------------------------------------
// NotifyAlnWriter: notify the writer of the complete results for a query
//
void TaskScheduler::NotifyAlnWriter()
{
    MYMSG("TaskScheduler::NotifyAlnWriter", 3);
    if(writer_) {
        int rspcode = writer_->GetResponse();
        if(rspcode == WRITERTHREAD_MSG_ERROR || 
           rspcode == TdAlnWriter::wrttrespmsgTerminating)
            throw MYRUNTIME_ERROR(
            "TaskScheduler::NotifyAlnWriter: Results writer terminated with errors.");
        writer_->Notify(TdAlnWriter::wrtthreadmsgWrite);
    }
}

// -------------------------------------------------------------------------
// WaitForAlnWriterToFinish: notify the writer about the process end and 
// wait for it to finish writings
//
void TaskScheduler::WaitForAlnWriterToFinish(bool error)
{
    char msgbuf[BUF_MAX];

    MYMSGBEGl(3)
        sprintf(msgbuf, "TaskScheduler::WaitForAlnWriterToFinish: Snd Msg %d",
            TdAlnWriter::wrtthreadmsgTerminate);
        MYMSG(msgbuf,3);
    MYMSGENDl

    if(writer_ == NULL) {
        MYMSGBEGl(3)
            MYMSG("TaskScheduler::WaitForAlnWriterToFinish: Null Writer.",3);
        MYMSGENDl
        return;
    }

    int rsp = WRITERTHREAD_MSG_UNSET;

    if(!error)
        rsp = writer_->WaitDone();

    if(rsp == WRITERTHREAD_MSG_ERROR) {
        MYMSG("TaskScheduler::WaitForAlnWriterToFinish: "
            "Writer terminated with ERRORS.", 1);
        return;
    }

    writer_->Notify(TdAlnWriter::wrtthreadmsgTerminate);
    //NOTE: do not wait, as the writer might have finished
//     int rsp = writer_->Wait(AlnWriter::wrttrespmsgTerminating);
//     if(rsp != AlnWriter::wrttrespmsgTerminating) {
//         throw MYRUNTIME_ERROR(
//         "TaskScheduler::WaitForAlnWriterToFinish: Invalid response from the writer.");
//     }
//     MYMSGBEGl(3)
//         sprintf(msgbuf,"TaskScheduler::WaitForAlnWriterToFinish: Rcv Msg %d",rsp);
//         MYMSG(msgbuf,3);
//     MYMSGENDl
}



// =========================================================================
// CreateMemoryConfigs: create global memory configurations
//
void TaskScheduler::CreateMemoryConfigs(size_t nareasperdevice)
{
    MYMSG("TaskScheduler::CreateMemoryConfigs", 3);
    static const std::string preamb = "TaskScheduler::CreateMemoryConfigs: ";

    size_t maxmem = 16 * ONEG;

    if(0 < CLOptions::GetDEV_MEM())
        maxmem = (size_t)CLOptions::GetDEV_MEM() * (size_t)ONEM;

    MpGlobalMemory* gmem = new MpGlobalMemory( 
        maxmem,
        (int)nareasperdevice
    );

    if(gmem == NULL)
        throw MYRUNTIME_ERROR(
        preamb + "Failed to create global memory configuration.");

    memcfgs_.push_back(gmem);
}



// =========================================================================
// Run: starting point for structure search and alignment
//
void TaskScheduler::Run()
{
    MYMSG("TaskScheduler::Run", 3);
    static const std::string preamb = "TaskScheduler::Run: ";

    myruntime_error mre;
    char msgbuf[BUF_MAX];
    size_t chunkdatasize = 0UL;
    size_t chunkdatalen = 0UL;//maximum total length of db structure positions that can be processed at once
    size_t chunknstrs = 0UL;//maximum number of db structures allowed to be processed at once
    int totqrsposs = CLOptions::GetDEV_QRES_PER_CHUNK();

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

    const size_t nagents = 1;//#data-processing agents

    try {
        //create the results writing thread
        CreateAlnWriter(output_, GetDNamelist());

        //create global memory configuration
        CreateMemoryConfigs(1/*nareasperdevice*/);

        if(memcfgs_.size() < 1)
            throw MYRUNTIME_ERROR(preamb + 
            "Failed to create global memory configuration.");

        //initialize the memory sections of all devices
        // according to the given maximum query length
        for(int gmc = 0; gmc < (int)memcfgs_.size(); gmc++ ) {
            if(memcfgs_[gmc] == NULL)
                continue;

            size_t szchdata = memcfgs_[gmc]->CalcMaxDbDataChunkSize(totqrsposs);

            chunkdatasize = szchdata;
            chunkdatalen = memcfgs_[gmc]->GetCurrentMaxDbPos();
            chunknstrs = memcfgs_[gmc]->GetCurrentMaxNDbStrs();

            break;
        }

        MpBatch cbpc(memcfgs_[0], 0/*areano*/, GetAlnWriter());

        CreateQrsReader( 
            -1,//query length unlimited (limited by max total length)
            CLOptions::GetIO_FILEMAP(),
            2 * nagents,//CLOptions::GetIO_NBUFFERS(),
            //nagents: +1 for late processing; a counter for each worker data buffer
            1/* hostworkers_.size() */);

        CreateReader( 
            CLOptions::GetDEV_MAXRLEN(),
            CLOptions::GetIO_FILEMAP(),
            CLOptions::GetIO_NBUFFERS(),
            nagents);

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

        querydescs.reset(new char**[nagents]);
        querypmbegs.reset(new char**[nagents]);
        querypmends.reset(new char**[nagents]);
        queryndxpmbegs.reset(new char**[nagents]);
        queryndxpmends.reset(new char**[nagents]);
        qrysernrbegs.resize(nagents, 0);
        qrstscnt.reset(new TSCounterVar*[nagents]);

        if(!(queryndxpmbegs) || !(queryndxpmends) ||
           !(querydescs) || !(querypmbegs) || !(querypmends) || !(qrstscnt))
            throw MYRUNTIME_ERROR(preamb + "Not enough memory.");

        size_t ntotqstrs = 0;
        std::vector<std::tuple<size_t,int,int>> valnwparams;
        valnwparams.reserve(nagents);


        //QUERY BLOCKS..
        for(int qblk = 0;;)
        {
            newsetqrs = true;

            GetQrsReader()->SetChunkDataAttributes(
                qrschunkdatasize, qrschunkdatalen, qrschunknstrs);

            memset(querydescs.get(), 0, nagents * sizeof(void*));
            memset(querypmbegs.get(), 0, nagents * sizeof(void*));
            memset(querypmends.get(), 0, nagents * sizeof(void*));
            memset(queryndxpmbegs.get(), 0, nagents * sizeof(void*));
            memset(queryndxpmends.get(), 0, nagents * sizeof(void*));
            memset(qrstscnt.get(), 0, nagents * sizeof(void*));

            valnwparams.clear();

            size_t wrk = 0;
            for(; wrk < nagents; wrk++) {
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
                            "%zu positions (%zu strs.) assigned to worker %zu",
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
            for(int chkno = 0;;
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
                    for(size_t w = 0; w < nagents; w++)
                        qrstscnt[w]->reset();

                //if no data, finish with the chunk:
                if(!datapresent) break;

                //NOTE: all workers must have finished; new chunk implies new data!

                MYMSGBEGl(1)
                    sprintf(msgbuf,"%s[=======] Processing database CHUNK No. %d: "
                            "%zu ref. positions (%zu strs.)",
                            NL, chkno, nposits, nCstrs);
                    MYMSG(msgbuf,1);
                MYMSGENDl

                //QUERIES passed to WORKERS...
                for(size_t w = 0; w < nagents; w++)
                {
                    ProcessPart(
                        cbpc,
                        w, chkno+1, lastchunk,
                        newsetqrs,
                        qrysernrbegs[w],
                        queryndxpmbegs[w], queryndxpmends[w],
                        (const char**)querydescs[w], querypmbegs[w], querypmends[w],
                        (const char**)bdbCdescs, bdbCpmbeg, bdbCpmend,
                        bdbCndxpmbeg, bdbCndxpmend,
                        qrstscnt[w], tscnt);
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

        //NOTE: wait for the workers to finish.

    } catch(myexception const& ex) {
        mre = ex;
    } catch(...) {
        mre = myruntime_error("Exception caught.");
    }

    //children may still have job; wait for the writer to finish
    WaitForAlnWriterToFinish(mre.isset());

    if(mre.isset())
        throw mre;

    MYMSG("Done.",1);
}

// -------------------------------------------------------------------------
// ProcessPart: process part of data, multiple reference structures for 
// multiple queries;
// cbpc, multi-processing batch object;
// workerid, worker id;
// chunkno, chunk serial number;
// lastchunk, indicator whether the chunk is last;
// newsetqrs, indicator whether the set of queries is a new set;
// qrysernrbeg, serial number of the first query in the chunk;
// querydesc, descriptions for each query in the data chunk;
// querypmbeg, beginnings of the query structure data fields in the chunk;
// querypmend, ends of the query structure data fields in the chunk;
// bdbCdesc, descriptions for each db structure in the data chunk;
// bdbCpmbeg, beginnings of the db structure data fields in the chunk;
// bdbCpmend, ends of the db structure data fields in the chunk;
// tscnt, thread-safe counter of processing agents;
//
void TaskScheduler::ProcessPart(
    MpBatch& cbpc,
    size_t /*workerid*/, int /*chunkno*/, bool lastchunk,
    bool /*newsetqrs*/,
    int qrysernrbeg,
    char** queryndxpmbeg, char** queryndxpmend,
    const char** querydesc, char** querypmbeg, char** querypmend,
    const char** bdbCdesc, char** bdbCpmbeg, char** bdbCpmend,
    char** bdbCndxpmbeg, char** bdbCndxpmend,
    TSCounterVar* qrstscnt, TSCounterVar* tscnt)
{
    MYMSG("TaskScheduler::ProcessPart", 4);
    static const std::string preamb = "TaskScheduler::ProcessPart: ";

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

    //queries can be empty (NULL) for workers to decrease the associated counter
    if(/*!queryinf || */!bdbCinf)
        throw MYRUNTIME_ERROR(preamb + "Structure data is empty.");

    char msgbuf[BUF_MAX];
    int nqueries = PMBatchStrData::GetNoStructs(querypmbeg, querypmend);
    int qrysernrend = qrysernrbeg + nqueries - 1;

    if(nqueries < 1) {
        //sprintf(msgbuf, "Invalid number of queries to be processed: %d", nqueries);
        //warning(msgbuf);
        //NOTE: queries can be empty: decrease the counters
        if(qrstscnt) qrstscnt->dec();
        if(tscnt) tscnt->dec();
        return;
    }

    MYMSGBEGl(3)
        sprintf(msgbuf, "Processing QUERY nos. %d - %d", qrysernrbeg, qrysernrend);
        MYMSG(msgbuf,3);
    MYMSGENDl

    //increase # chunks being processed only if this is not the last chunk!
    //this ensures the processing of all chunks and informs the writer to begin 
    // writing once the last chunk has finished
    if(!lastchunk) {
        qrstscnt->inc();//each query is referenced as many times as there are chunks
        writer_->IncreaseQueryNParts(qrysernrbeg, qrysernrend);
    }

    cbpc.ProcessBlock(
        qrysernrbeg,
        queryndxpmbeg, queryndxpmend,
        querydesc,
        querypmbeg[0]? querypmbeg: NULL, querypmend[0]? querypmend: NULL,
        bdbCdesc,
        bdbCpmbeg[0]? bdbCpmbeg: NULL, bdbCpmend[0]? bdbCpmend: NULL,
        bdbCndxpmbeg, bdbCndxpmend,
        qrstscnt, tscnt
    );
}
