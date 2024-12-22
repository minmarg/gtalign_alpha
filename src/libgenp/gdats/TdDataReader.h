/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __TdDataReader_h__
#define __TdDataReader_h__

#include "libutil/mybase.h"

#include <stdio.h>
#include <string.h>

#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>

#include "tsafety/TSCounterVar.h"
#include "PM2DVectorFields.h"
#include "PMBatchStrData.h"
#include "PMBatchStrDataIndex.h"
#include "FlexDataRead.h"

#define TREADER_MSG_UNSET -1
#define TREADER_MSG_ERROR -2
#define TREADER_MSG_EXIT -3

class TdDataReader;

using TGetNextData = void (TdDataReader::*)(
        size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs,
        const size_t ntotqstrs,
        const int queryblocks,
        char* const * const * const querypmbegs,
        char* const * const * const querypmends);

using TReadDataChunk = bool (TdDataReader::*)(
        PMBatchStrDataIndex*,
        PMBatchStrData*, const PMBatchStrData*, 
        size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs,
        const size_t ntotqstrs,
        const int queryblocks,
        char* const * const * const querypmbegs,
        char* const * const * const querypmends);

// _________________________________________________________________________
// Class TdDataReader
//
// thread class responsible for reading profile data
//
class TdDataReader
{
public:
    enum {
        tdrcheNotSet,
        tdrcheNoCaching,
        tdrcheCacheAndRead,
        tdrcheReadCached
    };
    enum TDRMsg {
        tdrmsgGetSize,
        tdrmsgGetData,
        tdrmsgTerminate
    };
    enum TDRResponseMsg {
        tdrrespmsgSize,
        tdrrespmsgDataReady,
        tdrrespmsgNoData,
        tdrrespmsgTerminating
    };

public:
    TdDataReader(
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
        const bool clustering = false,
        const bool clustmaster = false
    );
    TdDataReader();
    ~TdDataReader();

    static int GetDbsCacheFlag(
        int nagents, 
        const char* cachedir,
        const std::vector<std::string>& inputlist,
        size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs);

    // std::mutex& GetPrivateMutex() {return mx_dataccess_;}

    //{{NOTE: messaging functions accessed from outside!
    void Notify(
        int msg,
        size_t ntotqstrs = 0,
        int queryblocks = 0,
        char* const * const * const querypmbegs = NULL,
        char* const * const * const querypmends = NULL)
    {
        {//mutex must be unlocked before notifying
            std::lock_guard<std::mutex> lck(mx_dataccess_);
            req_msg_ = msg;
            ntotqstrs_ = ntotqstrs;
            queryblocks_ = queryblocks;
            querypmbegs_ = querypmbegs;
            querypmends_ = querypmends;
        }
        cv_msg_.notify_one();
    }
    int Wait(int rsp1, int rsp2 = TREADER_MSG_ERROR) {
        //wait until a response arrives
        std::unique_lock<std::mutex> lck_msg(mx_dataccess_);
        cv_msg_.wait(lck_msg,
            [this,rsp1,rsp2]{
                return (rsp_msg_ == rsp1 || rsp_msg_ == rsp2 || 
                        rsp_msg_ == TREADER_MSG_ERROR ||
                        rsp_msg_ == TREADER_MSG_EXIT);
            }
        );
        //lock is back; unset the response
        int rspmsg = rsp_msg_;
        //NOTE: master may change rsp after the reader set it repeatedly!
        //NOTE: may lead to dead lock when chunks delivered non-continuously!
        if( rsp_msg_!= TREADER_MSG_ERROR )
            rsp_msg_ = TREADER_MSG_UNSET;
        return rspmsg;
    }
    void Rewind() {
        std::lock_guard<std::mutex> lck(mx_dataccess_);
        rewind_ = true;
    }
    int GetResponseAsync() const {
        //get a response if available
        std::unique_lock<std::mutex> lck(mx_dataccess_, std::defer_lock);
        int rsp = TREADER_MSG_UNSET;
        if(lck.try_lock()) rsp = rsp_msg_;
        return rsp;
    }
    int GetResponse() const {
        std::lock_guard<std::mutex> lck(mx_dataccess_);
        return rsp_msg_;
    }
    void ResetResponse() {
        std::lock_guard<std::mutex> lck(mx_dataccess_);
        if( rsp_msg_!= TREADER_MSG_ERROR )
            rsp_msg_ = TREADER_MSG_UNSET;
    }
    //}}

    void GetbdbCdata(
        char**& bdbCdescs,
        char**& bdbCpmbeg, char**& bdbCpmend,
        char**& bdbCNdxpmbeg, char**& bdbCNdxpmend,
        TSCounterVar*& tscnt,
        bool* lastchunk )
    {
        std::lock_guard<std::mutex> lck(mx_dataccess_);
        bdbCNdxpmbeg = bdbCNdxpmbeg_;
        bdbCNdxpmend = bdbCNdxpmend_;
        bdbCdescs = bdbCptrdescs_;
        bdbCpmbeg = bdbCpmbeg_;
        bdbCpmend = bdbCpmend_;
        tscnt = ccnt_;
        *lastchunk = lastchunk_;
    }

    void GetbdbCdata(
        char**& bdbCdescs,
        char**& bdbCpmbeg, char**& bdbCpmend,
//         size_t*& szpm2dvfields,
        TSCounterVar*& tscnt,
        bool* lastchunk )
    {
        std::lock_guard<std::mutex> lck(mx_dataccess_);
        bdbCdescs = bdbCptrdescs_;
        bdbCpmbeg = bdbCpmbeg_;
        bdbCpmend = bdbCpmend_;
//         szpm2dvfields = szpm2dvfields_;
        tscnt = ccnt_;
        *lastchunk = lastchunk_;

//         if( bdbCpmbeg && bdbCpmend ) {
//             memcpy( bdbCpmbeg, bdbCpmbeg_, pmv2DTotFlds * sizeof(void*));
//             memcpy( bdbCpmend, bdbCpmend_, pmv2DTotFlds * sizeof(void*));
//         }
//         if( szpm2dvfields )
//             memcpy( szpm2dvfields, szpm2dvfields_, pmv2DTotFlds * sizeof(size_t));
    }

    void SetChunkDataAttributes( 
        size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs)
    {
        std::lock_guard<std::mutex> lck(mx_dataccess_);
        chunkdatasize_ = chunkdatasize;
        chunkdatalen_ = chunkdatalen;
        chunknstrs_ = chunknstrs;
    }

protected:
    void Execute(void* args);

    void UpdateCacheFlag();

    bool GetData(
        size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs,
        const size_t ntotqstrs,
        const int queryblocks,
        char* const * const * const querypmbegs,
        char* const * const * const querypmends);
    void GetNextData(
        size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs,
        const size_t ntotqstrs,
        const int queryblocks,
        char* const * const * const querypmbegs,
        char* const * const * const querypmends);
    void GetNextDataClustCache(
        size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs,
        const size_t ntotqstrs,
        const int queryblocks,
        char* const * const * const querypmbegs,
        char* const * const * const querypmends);
    bool ReadDataChunk(
        PMBatchStrDataIndex*,
        PMBatchStrData*, const PMBatchStrData*, 
        size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs,
        const size_t ntotqstrs,
        const int queryblocks,
        char* const * const * const querypmbegs,
        char* const * const * const querypmends);
    bool ReadDataChunkClustCache(
        PMBatchStrDataIndex*,
        PMBatchStrData*, const PMBatchStrData*, 
        size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs,
        const size_t ntotqstrs,
        const int queryblocks,
        char* const * const * const querypmbegs,
        char* const * const * const querypmends);

private:
    //thread section
    std::thread* tobj_;//thread object
private:
    TGetNextData fgetnextdata;
    TReadDataChunk freaddatachunk;
    //{{caching
    int cacheflag_;
    //}}
    //{{messaging
    std::condition_variable cv_msg_;//condition variable for messaging
    mutable std::mutex mx_dataccess_;//mutex for accessing data
    int req_msg_;//request message issued for thread
    int rsp_msg_;//private response message
    //}}
    //{{variables that determine the size of data chunks being read
    size_t chunkdatasize_;
    size_t chunkdatalen_;
    size_t chunknstrs_;
    //}}
    //{{db name and configuration:
    std::unique_ptr<FlexDataRead> dbobj_;//ptr to database(s)
    const char* cachedir_;//cache directory
    const std::vector<std::string>& inputlist_;//input list
    const std::vector<std::string>& strfilelist_;//structure file list
    const std::vector<std::string>& pntfilelist_;//parent file list
    const std::vector<size_t>& strfilepositionlist_;//list of structure file positions within an archive
    const std::vector<size_t>& strfilesizelist_;//list of structure file sizes
    const std::vector<int>& strparenttypelist_;//parent file type list of structure files (e.g, tar)
    const std::vector<int>& strfiletypelist_;//list of structure file types
    const std::vector<size_t>& filendxlist_;//list of the indices of files sorted by filesize
    std::vector<std::vector<int>>& globalids_;//global ids for structures across all files
    const size_t ndxstartwith_;//file index to start with
    const size_t ndxstep_;//file index step
    size_t ndxpacketfile_;//index of a packet file for caching
    size_t ndxlastpacketfile_;//index of the last packet file analyzed for clustering with caching
    const int maxstrlen_;//max structure length
    bool mapped_;//database mapped
    bool indexed_;//database to be indexed
    //}}
    //{{data:
//     std::unique_ptr<int,DRDataDeleter> bdbClengths_;//profile lengths read
//     size_t bdbClengths_from_, bdbClengths_to_;//number of lengths read (in the number of profiles)
//     std::unique_ptr<size_t,DRDataDeleter> bdbCdesc_end_addrs_;//description end addresses read (numbers as above)
    //
    PMBatchStrData bdbC4CC;//farthest data read for clustering with caching
    std::vector<PMBatchStrData> bdbCstruct_;//read data
    std::vector<PMBatchStrDataIndex> bdbCindex_;//indexed data
    int pbdbCstruct_ndx_;//index of the data block to be read 
    //
//     size_t bdbCdata_from_, bdbCdata_to_;//profile data read in the number of profiles
//     size_t bdbCdata_poss_from_, bdbCdata_poss_to_;//profile data read in the number of positions
//     size_t addrdescproced_;//address of the last description processed
    bool eodclustcached_;//flag of final EOD for clustering with caching
    bool endofdata_;//flag to indicate that no data read on the last call
    bool recycled_;//starting over the full cycle of data and the first data chunk has been read again
    bool rewind_;//flag set by the master and this thread for rewinding database
    //}}
    //{{addresses of indexed data to return
    char** bdbCNdxpmbeg_;//addresses of the field beginnings
    char** bdbCNdxpmend_;//addresses of the field endings
    //}}
    //{{addresses of read data to return
    char** bdbCptrdescs_;//structure descriptions
    char** bdbCpmbeg_;//addresses of the beginnings of the fields
    char** bdbCpmend_;//addresses of the endings of the fields
//     size_t* szpm2dvfields_;//beginnings in bytes (sizes) of the fields
    TSCounterVar* ccnt_;//counter associated with data to be return
    bool lastchunk_;//flag of whether the last chunk has been read
    int nagents_;//number of agents accessing the data
    //
    const bool clustering_;//flag indicating clustering
    const bool clustmaster_;//when clustering is on, the master will create global ids
    //}}
    size_t ntotqstrs_;
    //{{pairwise sequence similarity verification:
    int queryblocks_;
    char* const * const * querypmbegs_;
    char* const * const * querypmends_;
    //}}
    static const char* signaturefile_;
    static const char* packetfilebasename_;
};

#endif//__TdDataReader_h__
