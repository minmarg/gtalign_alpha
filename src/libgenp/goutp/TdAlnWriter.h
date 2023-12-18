/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __TdAlnWriter_h__
#define __TdAlnWriter_h__

#include "libutil/mybase.h"

#include <stdio.h>
#include <cmath>

#include <string>
#include <memory>
#include <utility>
#include <functional>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>

#include "libutil/CLOptions.h"

#define WRITERTHREAD_MSG_UNSET -1
#define WRITERTHREAD_MSG_ERROR -2

// -------------------------------------------------------------------------

struct WritersDataDestroyer {
    void operator()(char* p) const {
        std::free(p);
    };
};

// _________________________________________________________________________
// Class TdAlnWriter
//
// alignment writer thread
//
class TdAlnWriter
{
    enum {
        szTmpBuffer = KBYTE,
        szWriterBuffer = TIMES4(KBYTE)
    };

public:
    enum TWriterThreadMsg {
        wrtthreadmsgWrite,
        wrtthreadmsgTerminate
    };
    enum TWriterThreadResponse {
        wrttrespmsgWriting,
        wrttrespmsgTerminating
    };

public:
    TdAlnWriter( 
        const char* outdirname,
        const std::vector<std::string>& rfilelist
    );

    ~TdAlnWriter();

    std::mutex& GetPrivateMutex() {return mx_dataccess_;}

    //{{NOTE: messaging functions accessed from outside!
    void Notify(int msg) {
        {//mutex must be unlocked before notifying
            std::lock_guard<std::mutex> lck(mx_dataccess_);
            req_msg_ = msg;
        }
        cv_msg_.notify_all();
    }
    int Wait(int rsp) {
        //wait until the response is received
        std::unique_lock<std::mutex> lck_msg(mx_dataccess_);
        cv_msg_.wait(lck_msg,
            [this,rsp]{
                return (rsp_msg_ == rsp ||
                        rsp_msg_ == WRITERTHREAD_MSG_ERROR);
            }
        );
        //lock is back; unset the response
        int rspmsg = rsp_msg_;
        if(rsp_msg_ != WRITERTHREAD_MSG_ERROR)
            rsp_msg_ = WRITERTHREAD_MSG_UNSET;
        return rspmsg;
    }
    int WaitDone() {
        for(size_t i = 0; i < parts_qrs_.size();)
        {
            //wait until all queries have been processed
            std::unique_lock<std::mutex> lck_msg(mx_dataccess_);
            cv_msg_.wait(lck_msg,
                [this,i]{
                    return (parts_qrs_[i] <= 0 ||
                        rsp_msg_ == WRITERTHREAD_MSG_ERROR);
                }
            );
            if(rsp_msg_ == WRITERTHREAD_MSG_ERROR)
                return rsp_msg_;
            if(req_msg_ != WRITERTHREAD_MSG_UNSET)
                continue;
            i++;
        }
        return rsp_msg_;
    }
    void IncreaseQueryNParts(int qrysernrfrom, int qrysernrto) {
        std::lock_guard<std::mutex> lck(mx_dataccess_);
        if(qrysernrto < qrysernrfrom)
            return;
        if((int)parts_qrs_.size() <= qrysernrto) {
            //let the grand master decide on triggering the write by
            //initializing #parts to 1
            parts_qrs_.resize(qrysernrto+1, 0);
            ResizeVectors(qrysernrto+1);
        }
        for(int qsn = qrysernrfrom; qsn <= qrysernrto; qsn++)
            parts_qrs_[qsn]++;
    }
//     void DereaseNPartsAndTrigger(int qrysernr) {
//         std::unique_lock<std::mutex> lck(mx_dataccess_);
//         cv_msg_.wait(lck,
//             [this]{return req_msg_ == WRITERTHREAD_MSG_UNSET;}
//         );
//         if( --parts_qrs_[qrysernr] <= 0 ) {
//             //this is the last part for the given query:
//             //trigger write to a file
//             qrysernr_ = qrysernr;
//             req_msg_ = wrtthreadmsgWrite;
//             lck.unlock();
//             cv_msg_.notify_all();
//         }
//     }
    int GetResponse() const {
        std::lock_guard<std::mutex> lck(mx_dataccess_);
        return rsp_msg_;
    }
//     void ResetResponse() {
//         std::lock_guard<std::mutex> lck(mx_dataccess_);
//         if( rsp_msg_!= WRITERTHREAD_MSG_ERROR )
//             rsp_msg_ = WRITERTHREAD_MSG_UNSET;
//     }
    //}}

    void PushPartOfResults( 
        int qrysernr,
        int nqyposs,
        const std::string& qrydesc,
        const std::string& devanme,
        const size_t nqystrs,
        const double duration,
        const float tmsthld,
        const size_t ndbCposs,
        const size_t ndbCstrs,
        std::unique_ptr<char,WritersDataDestroyer> annotations,
        std::unique_ptr<char,WritersDataDestroyer> alignments,
        std::unique_ptr<std::vector<int>> srtindxs,
        std::unique_ptr<std::vector<float>> tmscores,
        std::unique_ptr<std::vector<char*>> alnptrs,
        std::unique_ptr<std::vector<char*>> annotptrs)
    {
        std::unique_lock<std::mutex> lck(mx_dataccess_);
        //{{NOTE: [inserted]
        cv_msg_.wait(lck,
            [this]{
                return req_msg_ == WRITERTHREAD_MSG_UNSET ||
                    rsp_msg_ == WRITERTHREAD_MSG_ERROR ||
                    rsp_msg_ == wrttrespmsgTerminating;
            }
        );
        //}}
        if(rsp_msg_ == WRITERTHREAD_MSG_ERROR ||
           rsp_msg_ == wrttrespmsgTerminating)
            return;
        //
        if((int)parts_qrs_.size() <= qrysernr || qrysernr < 0)
            throw MYRUNTIME_ERROR(
            "TdAlnWriter::PushPartOfResults: Invalid query serial number.");
        vec_duration_[qrysernr] += duration;
        vec_nposschd_[qrysernr] += ndbCposs;
        vec_nentries_[qrysernr] += ndbCstrs;
        vec_nqystrs_[qrysernr] = nqystrs;
        vec_nqyposs_[qrysernr] = nqyposs;
        vec_qrydesc_[qrysernr] = qrydesc;
        vec_devname_[qrysernr] = devanme;
        vec_tmsthld_[qrysernr] = tmsthld;
        //
        vec_annotations_[qrysernr].push_back(std::move(annotations));
        vec_alignments_[qrysernr].push_back(std::move(alignments));
        vec_srtindxs_[qrysernr].push_back(std::move(srtindxs));
        vec_tmscores_[qrysernr].push_back(std::move(tmscores));
        vec_alnptrs_[qrysernr].push_back(std::move(alnptrs));
        vec_annotptrs_[qrysernr].push_back(std::move(annotptrs));
        //{{NOTE: [commented out] the following statement must be the last
        //lck.unlock();
        //DereaseNPartsAndTrigger(qrysernr);
        //}}
        if( --parts_qrs_[qrysernr] <= 0 ) {
            //this is the last part for the given query:
            //trigger write to a file
            qrysernr_ = qrysernr;
            req_msg_ = wrtthreadmsgWrite;
            lck.unlock();
            cv_msg_.notify_all();
        }
    }


public:
    static int WritePrognamePlain( char*& outptr, int maxsize, const int width );
    static void WriteCommandLinePlain(FILE* fp,
        char* const buffer, const int szbuffer, char*& outptr, int& offset);
    static void WriteSearchInformationPlain(FILE* fp,
        char* const buffer, const int szbuffer, char*& outptr, int& offset,
        char* tmpbuf, int sztmpbuf, 
        const std::vector<std::string>& rfilelist,
        const size_t npossearched, const size_t nentries,
        const float tmsthrld, const int indent, const bool found,
        const bool clustering = false);

    static void BufferData( 
        FILE* fp, 
        char* const buffer, const int szbuffer, char*& outptr, int& offset, 
        const char* data, int szdata );
    static void WriteToFile( FILE* fp, char* data, int szdata );

protected:
    void Execute( void* args );

    void SetResponseError() {
        std::lock_guard<std::mutex> lck(mx_dataccess_);
        rsp_msg_ = WRITERTHREAD_MSG_ERROR;
    }

    void MergeResults();
    void WriteResults();
    void GetOutputFilename( 
        std::string& outfilename,
        const char* outdirname,
        const std::string& qrydesc,
        const int qrynr);
    int WriteProgname( char*& outptr, int maxsize, const int width );
    int WriteQueryDescription(char*& outptr, int maxsize,
        const int qrylen, const char* desc, const int width );
    int WriteSummary(char*& outptr,
        const int qrylen, const size_t npossearched, const size_t nentries,
        const int nqystrs, const double duration, const std::string&);

    void WriteResultsPlain();
    int WriteQueryDescriptionPlain(char*& outptr, int maxsize,
        const int qrylen, const std::string& desc, const int width );
    int WriteSummaryPlain(char*& outptr,
        const int qrylen, const size_t npossearched, const size_t nentries,
        const int nqystrs, const double duration, const std::string&);

    void WriteResultsJSON();
    int WritePrognameJSON( char*& outptr, int maxsize, const int width );
    int WriteQueryDescriptionJSON(char*& outptr, int maxsize,
        const int qrylen, const std::string& desc, const int width );
    int WriteSearchInformationJSON(char*& outptr, int maxsize, 
        const std::vector<std::string>& rfilelist,
        const size_t npossearched, const size_t nentries,
        const float tmsthrld, const int indent, const int annotlen, const bool found );
    int WriteSummaryJSON(char*& outptr,
        const int qrylen, const size_t npossearched, const size_t nentries);

    int GetTotalNumberOfRecords() const
    {
        if((int)parts_qrs_.size() <= qrysernr_ || qrysernr_ < 0 )
            throw MYRUNTIME_ERROR(
            "TdAlnWriter::GetTotalNumberOfRecords: Invalid query serial number.");
        int ntot = 0;
        for(size_t i = 0; i < vec_srtindxs_[qrysernr_].size(); i++) {
            if(vec_srtindxs_[qrysernr_][i])
                ntot += (int)vec_srtindxs_[qrysernr_][i]->size();
        }
        return ntot;
    }

    void InitializeVectors();

    void ResizeVectors(int newsize) {
        vec_duration_.resize(newsize, 0.0);
        vec_nposschd_.resize(newsize, 0);
        vec_nentries_.resize(newsize, 0);
        vec_nqystrs_.resize(newsize);
        vec_nqyposs_.resize(newsize);
        vec_qrydesc_.resize(newsize);
        vec_devname_.resize(newsize);
        vec_tmsthld_.resize(newsize);
        vec_annotations_.resize(newsize);
        vec_alignments_.resize(newsize);
        vec_srtindxs_.resize(newsize);
        vec_tmscores_.resize(newsize);
        vec_alnptrs_.resize(newsize);
        vec_annotptrs_.resize(newsize);
    }

    void ReleaseAllocations() {
        if((int)parts_qrs_.size() <= qrysernr_ || qrysernr_ < 0 )
            throw MYRUNTIME_ERROR(
            "TdAlnWriter::ReleaseAllocations: Invalid query serial number.");
        vec_annotations_[qrysernr_].clear();
        vec_alignments_[qrysernr_].clear();
        vec_srtindxs_[qrysernr_].clear();
        vec_tmscores_[qrysernr_].clear();
        vec_alnptrs_[qrysernr_].clear();
        vec_annotptrs_[qrysernr_].clear();
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
    //{{variables global to all queries:
    const char* mstr_set_outdirname_;//output directory name
    std::vector<std::string> mstr_set_rfilelist_;//filelist of reference (target) structures
    std::vector<double> vec_duration_;//accumulated duration for a query batch
    std::vector<size_t> vec_nposschd_;//total target db size in positions
    std::vector<size_t> vec_nentries_;//number of database entries
    //}}
    //{{query and summary data:
    std::vector<int> vec_nqystrs_;//batch size: #queries
    std::vector<int> vec_nqyposs_;//query lengths
    std::vector<std::string> vec_qrydesc_;//query descriptions
    std::vector<std::string> vec_devname_;//device names
    std::vector<float> vec_tmsthld_;//TM-score thresholds
    //}}
    //{{vectors of formatted results for QUERIES:
    std::vector<std::vector< std::unique_ptr<char,WritersDataDestroyer> >> vec_annotations_;
    std::vector<std::vector< std::unique_ptr<char,WritersDataDestroyer> >> vec_alignments_;
    std::vector<std::vector< std::unique_ptr<std::vector<int>> >> vec_srtindxs_;//index vectors of sorted TM-scores
    std::vector<std::vector< std::unique_ptr<std::vector<float>> >> vec_tmscores_;//2D vector of TM-scores for queries
    std::vector<std::vector< std::unique_ptr<std::vector<char*>> >> vec_alnptrs_;//2D vector of alignments for queries
    std::vector<std::vector< std::unique_ptr<std::vector<char*>> >> vec_annotptrs_;//2D vector of annotations for queries
    //}}
    //{{sorted indices over all parts (vectors) of results for a query:
    std::vector<int> allsrtindxs_;//indices along all vectors
    std::vector<int> allsrtvecs_;//corresponding vector indices (part numbers)
    std::vector<int> finalsrtindxs_;//globally (over all parts) sorted indices 
    std::vector<int> finalsrtindxs_dup_;//duplicate of globally sorted indices (for efficient memory management)
    std::vector<int>* p_finalindxs_;//pointer to the final vector of sorted indices
    //}}
    //buffer for writing to file:
    char buffer_[szWriterBuffer];
    //vector of the number of parts to be processed for each query serial number:
    std::vector<int> parts_qrs_;
    int qrysernr_;//query serial number
};

////////////////////////////////////////////////////////////////////////////
// TdAlnWriter INLINES
//
inline
void TdAlnWriter::WriteResults()
{
//     const int outfmt = CLOptions::GetB_FMT();

//     if(outfmt==CLOptions::ofJSON) {
//         WriteResultsJSON();
//         return;
//     }

    WriteResultsPlain();
    return;
}

// -------------------------------------------------------------------------
//
inline
int TdAlnWriter::WriteProgname( char*& outptr, int maxsize, const int width )
{
//     const int outfmt = CLOptions::GetB_FMT();

//     if(outfmt==CLOptions::ofJSON)
//         return WritePrognameJSON(outptr, maxsize, width);

    return WritePrognamePlain(outptr, maxsize, width);
}

// -------------------------------------------------------------------------
//
inline
int TdAlnWriter::WriteQueryDescription(
    char*& outptr, int maxsize,
    const int qrylen, const char* desc, const int width)
{
//     const int outfmt = CLOptions::GetB_FMT();

//     if(outfmt==CLOptions::ofJSON)
//         return WriteQueryDescriptionJSON(outptr, maxsize, qrylen, desc, width);

    return 
        WriteQueryDescriptionPlain(
            outptr, maxsize, qrylen, desc, width);
}

// -------------------------------------------------------------------------
//
inline
int TdAlnWriter::WriteSummary(
    char*& outptr,
    const int qrylen, const size_t npossearched, const size_t nentries,
    const int nqystrs, const double duration, const std::string& devname)
{
//     const int outfmt = CLOptions::GetB_FMT();

//     if(outfmt==CLOptions::ofJSON)
//         return 
//             WriteSummaryJSON(
//                 outptr, qrylen, npossearched, nentries);

    return 
        WriteSummaryPlain(
            outptr, qrylen, npossearched, nentries, nqystrs, duration, devname);
}

#endif//__TdAlnWriter_h__
