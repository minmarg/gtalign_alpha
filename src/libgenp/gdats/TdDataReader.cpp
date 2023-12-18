/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/mybase.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <memory>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <algorithm>
#include <fstream>
#include <cstdio>
#include <thread>

#include "PM2DVectorFields.h"
#include "PMBatchStrData.h"
#include "PMBatchStrDataIndex.h"
#include "FlexDataRead.h"
#include "TdDataReader.h"

const char* TdDataReader::signaturefile_ = "__dbsign__";
const char* TdDataReader::packetfilebasename_ = "__packet__";

// _________________________________________________________________________
// Class TdDataReader
//
// Constructor
// 
// dnamelist, file list;
// sfxlst, suffix list;
// mapped, flag of whether the database is mapped;
// ndatbufs, number of buffers used to cache data;
// nagents, number of agents processing each chunk of data read;
//
TdDataReader::TdDataReader(
    const char* cachedir,
    const std::vector<std::string>& inputlist,
    const std::vector<std::string>& strfilelist,
    const std::vector<std::string>& pntfilelist,
    const std::vector<size_t>& strfilepositionlist,
    const std::vector<size_t>& strfilesizelist,
    const std::vector<int>& strparenttypelist,
    const std::vector<int>& strfiletypelist,
    const std::vector<size_t>& filendxlist,
    std::vector<std::vector<int>>& globalids,
    const size_t ndxstartwith,
    const size_t ndxstep,
    int maxstrlen,
    bool mapped,
    bool indexed,
    int ndatbufs,
    int nagents,
    const bool clustering,
    const bool clustmaster)
:
    tobj_(NULL),
    fgetnextdata(&TdDataReader::GetNextData),
    freaddatachunk(&TdDataReader::ReadDataChunk),
    cacheflag_(tdrcheNotSet),
    req_msg_(TREADER_MSG_UNSET),
    rsp_msg_(TREADER_MSG_UNSET),
    //
    chunkdatasize_(0UL),
    chunkdatalen_(0UL),
    chunknstrs_(0UL),
    //
    dbobj_(nullptr),
    cachedir_(cachedir),
    inputlist_(inputlist),
    strfilelist_(strfilelist),
    pntfilelist_(pntfilelist),
    strfilepositionlist_(strfilepositionlist),
    strfilesizelist_(strfilesizelist),
    strparenttypelist_(strparenttypelist),
    strfiletypelist_(strfiletypelist),
    filendxlist_(filendxlist),
    globalids_(globalids),
    ndxstartwith_(ndxstartwith),
    ndxstep_(ndxstep),
    ndxpacketfile_(ndxstartwith),
    ndxlastpacketfile_(ndxstartwith),
    maxstrlen_(maxstrlen),
    mapped_(mapped),
    indexed_(indexed),
    //data:
//     bdbClengths_(nullptr),
//     bdbClengths_from_(0UL),
//     bdbClengths_to_(0UL),
//     bdbCdesc_end_addrs_(nullptr),
    bdbCstruct_(ndatbufs),
    bdbCindex_(ndatbufs),
    pbdbCstruct_ndx_(-1),
//     bdbCdata_from_(0UL),
//     bdbCdata_to_(0UL),
//     bdbCdata_poss_from_(0UL),
//     bdbCdata_poss_to_(0UL),
//     addrdescproced_(0),
    eodclustcached_(false),
    endofdata_(false),
    recycled_(false),
    rewind_(false),
    bdbCNdxpmbeg_(NULL),
    bdbCNdxpmend_(NULL),
    bdbCptrdescs_(NULL),
    bdbCpmbeg_(NULL),
    bdbCpmend_(NULL),
//     szpm2dvfields_(NULL),
    ccnt_(NULL),
    lastchunk_(true),
    nagents_(nagents),
    //
    clustering_(clustering),
    clustmaster_(clustmaster),
    //
    ntotqstrs_(0),
    queryblocks_(0),
    querypmbegs_(NULL),
    querypmends_(NULL)
{
    MYMSG("TdDataReader::TdDataReader", 3);
    tobj_ = new std::thread(&TdDataReader::Execute, this, (void*)NULL);
}

// Destructor
//
TdDataReader::~TdDataReader()
{
    MYMSG("TdDataReader::~TdDataReader", 3);
    if(tobj_) {
        std::for_each(bdbCstruct_.begin(), bdbCstruct_.end(),
            [](PMBatchStrData& d){d.cnt_.reset();}
        );
        Notify(tdrmsgTerminate);
        tobj_->join();
        delete tobj_;
        tobj_ = NULL;
    }
}

// -------------------------------------------------------------------------
// UpdateCacheFlag: update cache flag once
//
void TdDataReader::UpdateCacheFlag()
{
    MYMSG("TdDataReader::UpdateCacheFlag", 3);
    static const std::string preamb = "TdDataReader::UpdateCacheFlag: ";

    if(tdrcheNotSet < cacheflag_) return;

    if(!cachedir_ || strlen(cachedir_) < 1) {
        cacheflag_ = tdrcheNoCaching;
        return;
    }

    std::string signfullfile = std::string(cachedir_) + DIRSEPSTR + signaturefile_;
    std::string dbname;
    std::for_each(inputlist_.begin(), inputlist_.end(),
        [&dbname](const std::string& s){dbname += s;}
    );
    size_t dbnamelen = dbname.size();
    size_t now = (size_t)time(NULL);

    if(clustering_) {
        fgetnextdata = &TdDataReader::GetNextDataClustCache;
        freaddatachunk = &TdDataReader::ReadDataChunkClustCache;
        if(file_exists(signfullfile.c_str())) std::remove(signfullfile.c_str());
        cacheflag_ = tdrcheCacheAndRead;
        return;
    }

    while(file_exists(signfullfile.c_str())) {
        std::ifstream fp(signfullfile.c_str(), std::ios::binary);
        if(!fp) break;
        int nagents;
        size_t chunkdatasize = 0, chunkdatalen = 0, chunknstrs = 0;
        size_t dbnamelentmp = 0, nowt = 0;
        fp.read(reinterpret_cast<char*>(&nowt), sizeof(nowt));
        fp.read(reinterpret_cast<char*>(&nagents), sizeof(nagents));
        fp.read(reinterpret_cast<char*>(&chunkdatasize), sizeof(chunkdatasize));
        fp.read(reinterpret_cast<char*>(&chunkdatalen), sizeof(chunkdatalen));
        fp.read(reinterpret_cast<char*>(&chunknstrs), sizeof(chunknstrs));
        fp.read(reinterpret_cast<char*>(&dbnamelentmp), sizeof(dbnamelentmp));
        if(now < nowt || (now - nowt) < 60 || nagents_ != nagents ||
           chunkdatasize_ != chunkdatasize || chunkdatalen_ != chunkdatalen ||
           chunknstrs_ != chunknstrs || dbnamelen != dbnamelentmp)
            break;
        std::string dbnametmp; dbnametmp.resize(dbnamelentmp);
        fp.read(const_cast<char*>(dbnametmp.data()), dbnamelentmp);
        if(!(dbname == dbnametmp)) break;
        cacheflag_ = tdrcheReadCached;
        return;
    }

    cacheflag_ = tdrcheCacheAndRead;

    if(0 < ndxstartwith_) return;

    //only one thread writes a signature:
    std::ofstream fp(signfullfile.c_str(), std::ios::binary);
    if(fp.bad() || fp.fail())
        throw MYRUNTIME_ERROR(
            preamb + "Failed to open file for writing db signature (option -c).");
    fp.write(reinterpret_cast<const char*>(&now), sizeof(now));
    fp.write(reinterpret_cast<const char*>(&nagents_), sizeof(nagents_));
    fp.write(reinterpret_cast<const char*>(&chunkdatasize_), sizeof(chunkdatasize_));
    fp.write(reinterpret_cast<const char*>(&chunkdatalen_), sizeof(chunkdatalen_));
    fp.write(reinterpret_cast<const char*>(&chunknstrs_), sizeof(chunknstrs_));
    fp.write(reinterpret_cast<const char*>(&dbnamelen), sizeof(dbnamelen));
    fp.write(dbname.data(), dbnamelen);
    if(fp.bad()) throw MYRUNTIME_ERROR(preamb + "Db signature write failed (option -c).");
}

// -------------------------------------------------------------------------
// Execute: thread's starting point for execution
//
void TdDataReader::Execute( void* )
{
    MYMSG("TdDataReader::Execute", 3);
    static const std::string preamb = "TdDataReader::Execute: ";
    myruntime_error mre;


    //get-data functional
    std::function<int(
        size_t,size_t,size_t,
        const size_t, const int,
        char* const * const * const,
        char* const * const * const)> lfGetData = 
    [this](
        size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs,
        const size_t ntotqstrs,
        const int queryblocks,
        char* const * const * const querypmbegs,
        char* const * const * const querypmends)
    {
        if(!GetData(
           chunkdatasize, chunkdatalen, chunknstrs,
           ntotqstrs, queryblocks, querypmbegs, querypmends))
        {
            return tdrrespmsgNoData;
        }
        return tdrrespmsgDataReady;
    };

    //rewind to the beginning of the data and reset related paramteres;
    //NOTE: under lock
    std::function<void()> lfRewind = [this]() {
        // dbobj_->Open();
        rewind_ = false;
        endofdata_ = false;
        recycled_ = false;//
        // pbdbCstruct_ndx_ = -1;
        if((int)bdbCstruct_.size() <= ++pbdbCstruct_ndx_)//
            pbdbCstruct_ndx_ = 0;
    };


    try {
        dbobj_.reset(
            new FlexDataRead(
                strfilelist_,
                pntfilelist_,
                strfilepositionlist_,
                strfilesizelist_,
                strparenttypelist_,
                strfiletypelist_,
                filendxlist_,
                globalids_,
                ndxstartwith_, ndxstep_,
                maxstrlen_,
                clustering_,
                clustmaster_));

        if(!dbobj_)
            throw MYRUNTIME_ERROR(preamb + "Not enough memory.");

        dbobj_->Open();

        while(1) {
            //wait until the master bradcasts a message
            std::unique_lock<std::mutex> lck_msg(mx_dataccess_);

            cv_msg_.wait(lck_msg,
                [this]{return 
                    ((0 <= req_msg_ && req_msg_ <= tdrmsgTerminate) || 
                    req_msg_ == TREADER_MSG_ERROR
                    );}
            );

            MYMSGBEGl(3)
                char msgbuf[BUF_MAX];
                sprintf(msgbuf, "%sMsg %d", preamb.c_str(), req_msg_);
                MYMSG(msgbuf, 3);
            MYMSGENDl

            //thread owns the lock after the wait;
            //read message req_msg_
            int reqmsg = req_msg_;

            //unset the message to avoid live cycle when starting over the loop
            req_msg_ = TREADER_MSG_UNSET;

            //set response msg to error upon exception occurrence
            rsp_msg_ = TREADER_MSG_ERROR;
            int rspmsg = rsp_msg_;

            size_t chunkdatasize = chunkdatasize_;
            size_t chunkdatalen = chunkdatalen_;
            size_t chunknstrs = chunknstrs_;
            const size_t ntotqstrs = ntotqstrs_;
            const int queryblocks = queryblocks_;
            char* const * const * const querypmbegs = querypmbegs_;
            char* const * const * const querypmends = querypmends_;

            UpdateCacheFlag();

            //immediately read the master data and take action
            switch(reqmsg) {
                case tdrmsgGetSize:
                        rspmsg = tdrrespmsgSize;
                        break;
                case tdrmsgGetData:
                        ;;
                        if(rewind_) lfRewind();
                        rspmsg = lfGetData(
                            chunkdatasize, chunkdatalen, chunknstrs,
                            ntotqstrs, queryblocks, querypmbegs, querypmends);
                        ;;
                        break;
                case tdrmsgTerminate:
                        rspmsg = tdrrespmsgTerminating;
                        break;
                default:
                        rspmsg = TREADER_MSG_UNSET;
                        break;
            };

            MYMSGBEGl(3)
                char msgbuf[BUF_MAX];
                sprintf(msgbuf, "%sMsg %d Rsp %d", preamb.c_str(), reqmsg, rspmsg);
                MYMSG(msgbuf, 3);
            MYMSGENDl

            //save response code and proceed
            rsp_msg_ = rspmsg;

            //send a message back to the master:
            //unlock the mutex to avoid blocking the awaiting master and 
            // notify the master using the cv
            lck_msg.unlock();
            cv_msg_.notify_one();

            if( reqmsg < 0 || reqmsg == tdrmsgTerminate)
                //terminate execution
                break;

            //if not end of file, read the next portion of data in advance;
            //NOTE: the lock has been released
            if(reqmsg == tdrmsgGetData)
                (this->*fgetnextdata)(
                    chunkdatasize, chunkdatalen, chunknstrs,
                    ntotqstrs, queryblocks, querypmbegs, querypmends);
        }

    } catch( myruntime_error const& ex ) {
        mre = ex;
    } catch( myexception const& ex ) {
        mre = ex;
    } catch( ... ) {
        mre = MYRUNTIME_ERROR("Unknown exception caught.");
    }

    if( mre.isset())
        error( mre.pretty_format().c_str());

    if(dbobj_)
        dbobj_->Close();//the thread closes db

    if( mre.isset()) {
        {//notify the master
            std::lock_guard<std::mutex> lck_msg(mx_dataccess_);
            rsp_msg_ = TREADER_MSG_ERROR;
        }
        cv_msg_.notify_one();
        return;
    }
    {//set the exit flag
        std::lock_guard<std::mutex> lck_msg(mx_dataccess_);
        rsp_msg_ = TREADER_MSG_EXIT;
    }
    cv_msg_.notify_one();
}



// =========================================================================
// GetData: get portion of data given maximum data chunk size;
// chunkdatasize, data chunk size;
// chunkdatalen, limit for the total length of structures;
// chunknstrs, limit for the number of structures;
// return whether data has been read;
// NOTE: data accessed under lock!
// 
bool TdDataReader::GetData(
    size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs,
    const size_t ntotqstrs, const int queryblocks,
    char* const * const * const querypmbegs,
    char* const * const * const querypmends)
{
    MYMSG("TdDataReader::GetData", 3);
    bool eod = endofdata_;//flag of the end of data

    if( pbdbCstruct_ndx_ < 0 ) {
        //starting filling in the first buffer for the first time
        pbdbCstruct_ndx_ = 0;
        eod = 
            (this->*freaddatachunk)(
                &bdbCindex_[pbdbCstruct_ndx_],
                &bdbCstruct_[pbdbCstruct_ndx_], NULL, 
                    chunkdatasize, chunkdatalen, chunknstrs,
                    ntotqstrs, queryblocks, querypmbegs, querypmends);
    }

    bdbCNdxpmbeg_ = bdbCindex_[pbdbCstruct_ndx_].bdbCpmbeg_;
    bdbCNdxpmend_ = bdbCindex_[pbdbCstruct_ndx_].bdbCpmend_;

    bdbCptrdescs_ = bdbCstruct_[pbdbCstruct_ndx_].bdbCptrdescs_.get();
    bdbCpmbeg_ = bdbCstruct_[pbdbCstruct_ndx_].bdbCpmbeg_;
    bdbCpmend_ = bdbCstruct_[pbdbCstruct_ndx_].bdbCpmend_;
//     szpm2dvfields_ = bdbCstruct_[pbdbCstruct_ndx_].szpm2dvfields_;
    ccnt_ = &bdbCstruct_[pbdbCstruct_ndx_].cnt_;
    bool datain = bdbCstruct_[pbdbCstruct_ndx_].ContainsData();
    lastchunk_ = eod;
    //NOTE: every chunk to be processed: true for clustering!
    return (recycled_)? false: (clustering_? true: datain);
}

// -------------------------------------------------------------------------
// GetNextData: get next portion of data given maximum data chunk size;
// chunkdatasize, data chunk size;
// chunkdatalen, limit for the total length of structures;
// chunknstrs, limit for the number of structures;
// NOTE: lock released!
// 
void TdDataReader::GetNextData(
    size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs,
    const size_t ntotqstrs, const int queryblocks,
    char* const * const * const querypmbegs,
    char* const * const * const querypmends)
{
    MYMSG("TdDataReader::GetNextData", 3);
    static const std::string preamb = "TdDataReader::GetNextData: ";

    if( pbdbCstruct_ndx_ < 0 )
        throw MYRUNTIME_ERROR(preamb + "Memory access error.");

    if(recycled_) return;

    const PMBatchStrData* pbdbC_prev = NULL;
    int prevndx = pbdbCstruct_ndx_;
    int nextndx = prevndx + 1;
    if((int)bdbCstruct_.size() <= nextndx)
        nextndx = 0;

    if(endofdata_) {//
        dbobj_->Open();//
        recycled_ = true;//
        ndxpacketfile_ = ndxstartwith_;//reset
        if(cacheflag_ == tdrcheCacheAndRead) cacheflag_ = tdrcheReadCached;
    }
    else {
        pbdbCstruct_ndx_ = nextndx;
        pbdbC_prev = &bdbCstruct_[prevndx];
    }

    endofdata_ = 
        (this->*freaddatachunk)(
            &bdbCindex_[nextndx], &bdbCstruct_[nextndx], pbdbC_prev, 
                chunkdatasize, chunkdatalen, chunknstrs,
                ntotqstrs, queryblocks, querypmbegs, querypmends);

    MYMSGBEGl(3)
        char strbuf[BUF_MAX];
        sprintf(strbuf,"%seof %d recycled %d", preamb.c_str(), endofdata_, recycled_);
        MYMSG(strbuf,3);
    MYMSGENDl
}

// -------------------------------------------------------------------------
// ReadDataChunk: read a data chunk from file(s);
// pbdbCNdx, batch object of indexed data (if required)
// pbdbC, batch object to contain new data;
// pbdbC_prev, batch object used previously for reading data, possibly 
// containing one overhead structure data to be copied to pbdbC;
// chunkdatasize, data chunk size;
// chunkdatalen, limit for the total length of structures;
// chunknstrs, limit for the number of structures;
// return true if there are no data to read;
// 
bool TdDataReader::ReadDataChunk( 
    PMBatchStrDataIndex* pbdbCNdx,
    PMBatchStrData* pbdbC, const PMBatchStrData* pbdbC_prev,
    size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs,
    const size_t /* ntotqstrs */, const int queryblocks,
    char* const * const * const querypmbegs,
    char* const * const * const querypmends)
{
    MYMSG("TdDataReader::ReadDataChunk", 3);
    static const std::string preamb = "TdDataReader::ReadDataChunk: ";

#ifdef __DEBUG__
    if(!dbobj_)
        throw MYRUNTIME_ERROR(preamb + "NULL data read object.");

    if(pbdbCNdx == NULL || pbdbC == NULL)
        throw MYRUNTIME_ERROR(preamb + "Memory access error.");
#endif

    MYMSG("TdDataReader::ReadDataChunk: Waiting until release of data...", 5 );
    //NOTE: wait for all data-using agents to finish their work
    if(pbdbC->ContainsData()) pbdbC->cnt_.wait0();
    MYMSG("TdDataReader::ReadDataChunk: Wait finished", 5 );

    if( chunkdatasize < 1 || chunkdatalen < 1 || chunknstrs < 1 )
        throw MYRUNTIME_ERROR(preamb + "Invalid requested data sizes.");

    //the following resets pointers too:
    pbdbC->AllocateSpace(chunkdatasize, chunkdatalen, chunknstrs);

    //first, copy data from the overhead, if any, of the previous object
    if(pbdbC_prev && cacheflag_ < tdrcheReadCached) {
        //this code executed under released lock when the master 
        // potentially accesses (reads) pbdbC_prev data simultaneously;
        //NOTE: pbdbC_prev will always contain original data as 
        //NOTE: filtering on GPU(s) works on copies
        PMBatchStrData::TPMBSDFinRetCode rcode =
            pbdbC_prev->CopyOvhdTo(*pbdbC);
        if(rcode != PMBatchStrData::pmbsdfEmpty && 
           rcode != PMBatchStrData::pmbsdfWritten)
            //NOTE: this should never happen!
            warning((std::string("Structure unexpectedly ignored: ") +
                pbdbC_prev->GetOvhdStrDescription()).c_str());
    }

    std::string packetfullfilename = 
        std::string(cachedir_) + DIRSEPSTR + 
        packetfilebasename_ + std::to_string((int)ndxpacketfile_);

    //read data next
    bool eod;
    if(cacheflag_ < tdrcheReadCached) {
        eod = dbobj_->ReadData(*pbdbC,  queryblocks, querypmbegs, querypmends);
        if(cacheflag_ == tdrcheCacheAndRead)
            pbdbC->Serialize(packetfullfilename.c_str(), nagents_, (size_t)eod);
    } else//tdrcheReadCached
        eod = (bool)pbdbC->Deserialize(packetfullfilename.c_str(), nagents_);

    ndxpacketfile_ += ndxstep_;

    //sort just-read data for efficient processing
    pbdbC->Sort();

    //NOTE: set the number of agents expected to access the data read
    if(pbdbC->ContainsData()) pbdbC->cnt_.set(nagents_);//UPDATE!

    if(indexed_ && pbdbC->ContainsData())
        pbdbCNdx->MakeIndex(*pbdbC);

    return eod;
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// GetNextDataClust: get next data portion for clustering with caching;
// chunkdatasize, data chunk size;
// chunkdatalen, limit for the total length of structures;
// chunknstrs, limit for the number of structures;
// NOTE: lock released!
// 
void TdDataReader::GetNextDataClustCache(
    size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs,
    const size_t ntotqstrs, const int queryblocks,
    char* const * const * const querypmbegs,
    char* const * const * const querypmends)
{
    MYMSG("TdDataReader::GetNextDataClustCache", 3);
    static const std::string preamb = "TdDataReader::GetNextDataClustCache: ";

    if( pbdbCstruct_ndx_ < 0 )
        throw MYRUNTIME_ERROR(preamb + "Memory access error.");

    if(recycled_) return;

    const PMBatchStrData* pbdbC_prev = NULL;
    int prevndx = pbdbCstruct_ndx_;
    int nextndx = prevndx + 1;
    if((int)bdbCstruct_.size() <= nextndx)
        nextndx = 0;

    if(endofdata_) {//
        recycled_ = true;//
        if(!(eodclustcached_ && bdbC4CC.FieldTypeValid() && !bdbC4CC.GetNoPositsOvhd()))
            ndxpacketfile_ = ndxstartwith_;//reset
    }
    else {
        pbdbCstruct_ndx_ = nextndx;
        pbdbC_prev = &bdbCstruct_[prevndx];
    }

    endofdata_ = 
        (this->*freaddatachunk)(
            &bdbCindex_[nextndx], &bdbCstruct_[nextndx], pbdbC_prev, 
                chunkdatasize, chunkdatalen, chunknstrs,
                ntotqstrs, queryblocks, querypmbegs, querypmends);

    MYMSGBEGl(3)
        char strbuf[BUF_MAX];
        sprintf(strbuf,"%seof %d recycled %d", preamb.c_str(), endofdata_, recycled_);
        MYMSG(strbuf,3);
    MYMSGENDl
}

// -------------------------------------------------------------------------
// ReadDataChunkClust: read a data chunk from file(s); clustering with 
// caching version;
// pbdbCNdx, batch object of indexed data (if required)
// pbdbC, batch object to contain new data;
// pbdbC_prev, batch object used previously for reading data, possibly 
// containing one overhead structure data to be copied to pbdbC;
// chunkdatasize, data chunk size;
// chunkdatalen, limit for the total length of structures;
// chunknstrs, limit for the number of structures;
// return true if there are no data to read;
// 
bool TdDataReader::ReadDataChunkClustCache( 
    PMBatchStrDataIndex* pbdbCNdx,
    PMBatchStrData* pbdbC, const PMBatchStrData* /*pbdbC_prev*/,
    size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs,
    const size_t ntotqstrs, const int queryblocks,
    char* const * const * const querypmbegs,
    char* const * const * const querypmends)
{
    MYMSG("TdDataReader::ReadDataChunkClustCache", 3);
    static const std::string preamb = "TdDataReader::ReadDataChunkClustCache: ";

    MYMSG("TdDataReader::ReadDataChunkClustCache: Waiting until release of data...", 5 );
    //NOTE: wait for all data-using agents to finish their work
    if(pbdbC->ContainsData()) pbdbC->cnt_.wait0();
    MYMSG("TdDataReader::ReadDataChunkClustCache: Wait finished", 5 );

    if( chunkdatasize < 1 || chunkdatalen < 1 || chunknstrs < 1 )
        throw MYRUNTIME_ERROR(preamb + "Invalid requested data sizes.");

    //the following resets pointers too:
    pbdbC->AllocateSpace(chunkdatasize, chunkdatalen, chunknstrs);

    std::string packetfullfilename = 
        std::string(cachedir_) + DIRSEPSTR + 
        packetfilebasename_ + std::to_string((int)ndxpacketfile_);

    //read data next
    bool eod = false;
    if(ndxpacketfile_ < ndxlastpacketfile_) {
        pbdbC->Deserialize(packetfullfilename.c_str(), nagents_);
        ndxpacketfile_ += ndxstep_;
    } else {
        eod = true;
        //read if no data or all structures have been processed:
        //NOTE: it can be that ovhd contains the last reference;
        //NOTE: it's still unimportant as it'll match the last query;
        const bool readdata = bdbC4CC.FieldTypeValid();
        if(readdata) {
            if(bdbC4CC.ContainsData()) {
                bdbC4CC.Serialize(packetfullfilename.c_str(), nagents_, 1/*always set eod*/);
                ndxpacketfile_ += ndxstep_;
                ndxlastpacketfile_ += ndxstep_;
                //copy the overhead, if any, of the object;
                PMBatchStrData::TPMBSDFinRetCode rcode = bdbC4CC.CopyOvhdTo(*pbdbC);
                if(rcode != PMBatchStrData::pmbsdfEmpty && rcode != PMBatchStrData::pmbsdfWritten)
                    //NOTE: this should never happen!
                    warning((std::string("Structure unexpectedly ignored: ") +
                        bdbC4CC.GetOvhdStrDescription()).c_str());
            }
            eodclustcached_ = dbobj_->ReadData(*pbdbC,  queryblocks, querypmbegs, querypmends);
            bdbC4CC.AllocateSpace(chunkdatasize, chunkdatalen, chunknstrs);
            bdbC4CC.CopyFrom(*pbdbC);
        }
        //adjust the Type field
        //adjust the Type field
        int nstructswrt = (int)bdbC4CC.GetNoStructsWritten();
        for(int i = 0; i < nstructswrt; i++) {
            int type = bdbC4CC.GetFieldAt<INTYPE,pps2DType>(i);
            if(type == INT_MAX) {
                int glbndx = dbobj_->GetGlobalId(bdbC4CC.GetFileNdxAt(i), bdbC4CC.GetStructNdxAt(i));
                if(glbndx < (int)ntotqstrs) bdbC4CC.SetFieldAt<INTYPE,pps2DType>(i, glbndx);
            }
        }
        pbdbC->CopyFrom(bdbC4CC);
    }

    //sort just-read data for efficient processing
    pbdbC->Sort();

    //NOTE: set the number of agents expected to access the data read
    if(pbdbC->ContainsData()) pbdbC->cnt_.set(nagents_);//UPDATE!

    if(indexed_ && pbdbC->ContainsData())
        pbdbCNdx->MakeIndex(*pbdbC);

    return eod;
}
