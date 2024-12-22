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
#include <memory>
#include <functional>
#include <algorithm>
#include <numeric>
#include <mutex>
#include <condition_variable>
#include <thread>

#include "libutil/CLOptions.h"
#include "TdAlnWriter.h"

//100 queries for initial reservation of vectors:
#define INITVECRESERVATION 128

// _________________________________________________________________________
// Class TdAlnWriter
//
// Constructor
//
TdAlnWriter::TdAlnWriter( 
    const char* outdirname,
    const std::vector<std::string>& rfilelist)
:   tobj_(NULL),
    req_msg_(WRITERTHREAD_MSG_UNSET),
    rsp_msg_(WRITERTHREAD_MSG_UNSET),
    //
    mstr_set_outdirname_(outdirname),
    //
    p_finalindxs_(NULL),
    //
    qrysernr_(-1)
{
    MYMSG("TdAlnWriter::TdAlnWriter", 3);

    vec_duration_.reserve(INITVECRESERVATION);
    vec_nposschd_.reserve(INITVECRESERVATION);
    vec_nentries_.reserve(INITVECRESERVATION);

    vec_nqystrs_.reserve(INITVECRESERVATION);
    vec_nqyposs_.reserve(INITVECRESERVATION);
    vec_qrydesc_.reserve(INITVECRESERVATION);
    vec_devname_.reserve(INITVECRESERVATION);
    vec_tmsthld_.reserve(INITVECRESERVATION);
    vec_annotations_.reserve(INITVECRESERVATION);
    vec_alignments_.reserve(INITVECRESERVATION);
    vec_srtindxs_.reserve(INITVECRESERVATION);
    vec_tmscores_.reserve(INITVECRESERVATION);
    vec_alnptrs_.reserve(INITVECRESERVATION);
    vec_annotptrs_.reserve(INITVECRESERVATION);

    parts_qrs_.reserve(INITVECRESERVATION);

    for(const std::string& dbn: rfilelist) {
        //std::string::npos+1==0
        std::string::size_type pos = 0;//dbn.rfind(DIRSEP) + 1;
        mstr_set_rfilelist_.push_back(dbn.substr(pos));
    }

    tobj_ = new std::thread(&TdAlnWriter::Execute, this, (void*)NULL);
}

// Destructor
//
TdAlnWriter::~TdAlnWriter()
{
    MYMSG("TdAlnWriter::~TdAlnWriter", 3);
    if( tobj_ ) {
        tobj_->join();
        delete tobj_;
        tobj_ = NULL;
    }
}

// -------------------------------------------------------------------------
// Execute: thread's starting point for execution
//
void TdAlnWriter::Execute( void* )
{
    MYMSG("TdAlnWriter::Execute", 3);
    myruntime_error mre;
    char msgbuf[BUF_MAX];

    try {
        while(1) {
            //wait for a message
            std::unique_lock<std::mutex> lck_msg(mx_dataccess_);

            cv_msg_.wait(lck_msg,
                [this]{
                    return (
                        (0 <= req_msg_ && req_msg_ <= wrtthreadmsgTerminate) || 
                        req_msg_ == WRITERTHREAD_MSG_ERROR
                    );
                }
            );

            MYMSGBEGl(3)
                sprintf(msgbuf, "TdAlnWriter::Execute: Msg %d", req_msg_);
                MYMSG(msgbuf, 3);
            MYMSGENDl

            //thread owns the lock after the wait;
            //read message req_msg_
            int reqmsg = req_msg_;

            //unset the message to avoid live cycle when starting over the loop
            req_msg_ = WRITERTHREAD_MSG_UNSET;

            //set response msg to error upon exception
            rsp_msg_ = WRITERTHREAD_MSG_ERROR;
            int rspmsg = rsp_msg_;

            switch(reqmsg) {
                case wrtthreadmsgWrite:
                        ;;
                        if((int)parts_qrs_.size() <= qrysernr_ || qrysernr_ < 0 )
                            throw MYRUNTIME_ERROR(
                            "TdAlnWriter::Execute: Invalid query serial number.");
                        MergeResults();
                        WriteResults();
                        ReleaseAllocations();
                        qrysernr_ = -1;
                        ;;
                        //parent does not wait for a response nor requires data to read;
                        //unset response code
                        rspmsg = WRITERTHREAD_MSG_UNSET;
                        break;
                case wrtthreadmsgTerminate:
                        rspmsg = wrttrespmsgTerminating;
                        break;
                default:
                        rspmsg = WRITERTHREAD_MSG_UNSET;
                        break;
            };

            MYMSGBEGl(3)
                sprintf(msgbuf, "TdAlnWriter::Execute: Msg %d Rsp %d", reqmsg, rspmsg);
                MYMSG(msgbuf, 3);
            MYMSGENDl

            //save response code
            rsp_msg_ = rspmsg;

            //unlock the mutex and notify awaiting threads
            lck_msg.unlock();
            cv_msg_.notify_all();

            if(reqmsg < 0 || reqmsg == wrtthreadmsgTerminate)
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

    if( mre.isset()) {
        error(mre.pretty_format().c_str());
        SetResponseError();
        cv_msg_.notify_all();
        return;
    }
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// MergeResults: merge parts of results
//
void TdAlnWriter::MergeResults()
{
    MYMSG("TdAlnWriter::MergeResults", 4);
    static const std::string preamb = "TdAlnWriter::MergeResults: ";

    InitializeVectors();

    if( vec_srtindxs_[qrysernr_].size() != vec_annotations_[qrysernr_].size() ||
        vec_srtindxs_[qrysernr_].size() != vec_alignments_[qrysernr_].size() ||
        vec_srtindxs_[qrysernr_].size() != vec_tmscores_[qrysernr_].size() ||
        vec_srtindxs_[qrysernr_].size() != vec_alnptrs_[qrysernr_].size() ||
        vec_srtindxs_[qrysernr_].size() != vec_annotptrs_[qrysernr_].size())
        throw MYRUNTIME_ERROR(preamb + "Inconsistent result sizes.");

    //partially merged and new merged vectors:
    std::vector<int>* p_prtmerged = &finalsrtindxs_;
    std::vector<int>* p_newmerged = &finalsrtindxs_dup_;
    p_finalindxs_ = NULL;

    //number of entries merged already
    size_t nmerged = 0UL;

    for(size_t i = 0; i < vec_srtindxs_[qrysernr_].size(); i++) {
        if(!vec_srtindxs_[qrysernr_][i] || !vec_tmscores_[qrysernr_][i])
            continue;
        if(vec_srtindxs_[qrysernr_][i]->size() < 1)
            continue;
        std::merge(
            p_prtmerged->begin(), p_prtmerged->begin() + nmerged,
            p_prtmerged->begin() + nmerged, 
                    p_prtmerged->begin() + nmerged + vec_srtindxs_[qrysernr_][i]->size(),
            p_newmerged->begin(),
            [this](size_t n1, size_t n2) {
                return //sort in reverse order
                    ( *vec_tmscores_[qrysernr_][allsrtvecs_[n1]] )[allsrtindxs_[n1]] > 
                    ( *vec_tmscores_[qrysernr_][allsrtvecs_[n2]] )[allsrtindxs_[n2]];
            }
        );
        //save the pointer pointing to the results
        p_finalindxs_ = p_newmerged;
        //swap pointers:
        p_newmerged = p_prtmerged;
        p_prtmerged = p_finalindxs_;
        //the number of merged entries has increased:
        nmerged += vec_srtindxs_[qrysernr_][i]->size();
    }
}

// -------------------------------------------------------------------------
// InitializeVectors: initialize index vectors
inline
void TdAlnWriter::InitializeVectors()
{
    MYMSG("TdAlnWriter::InitializeVectors", 5);
    int nrecords = GetTotalNumberOfRecords();

    allsrtindxs_.reserve(nrecords);
    allsrtvecs_.reserve(nrecords);
    finalsrtindxs_.reserve(nrecords);
    finalsrtindxs_dup_.reserve(nrecords);

    allsrtindxs_.clear();
    allsrtvecs_.clear();
    finalsrtindxs_.clear();
    finalsrtindxs_dup_.clear();

    for(size_t i = 0; i < vec_srtindxs_[qrysernr_].size(); i++) {
        if(!vec_srtindxs_[qrysernr_][i] || vec_srtindxs_[qrysernr_][i]->size() < 1)
            continue;
        allsrtindxs_.insert(allsrtindxs_.end(),
            vec_srtindxs_[qrysernr_][i]->begin(), vec_srtindxs_[qrysernr_][i]->end());
        allsrtvecs_.insert(allsrtvecs_.end(), vec_srtindxs_[qrysernr_][i]->size(), (int)i);
        //sequentially enumerate all records:
        //std::vector<int>::iterator finalprevend = finalsrtindxs_.end();
        int szfinal = (int)finalsrtindxs_.size();
        finalsrtindxs_.resize(szfinal + vec_srtindxs_[qrysernr_][i]->size());
        //std::iota(finalprevend, finalsrtindxs_.end(), szfinal);
        std::iota(finalsrtindxs_.begin() + szfinal, finalsrtindxs_.end(), szfinal);
    }

    finalsrtindxs_dup_ = finalsrtindxs_;
}



// =========================================================================
// BufferData: buffer data and write the buffer contents to file when the 
// buffer is full;
// fp, file pointer;
// buffer, buffer to store data in;
// szbuffer, size of the buffer;
// outptr, varying address of the pointer pointing to a location in the 
//  buffer;
// offset, outptr offset from the beginning of the buffer (fill size);
// data, data to store in the buffer;
// szdata, size of the data;
//
void TdAlnWriter::BufferData( 
    FILE* fp, 
    char* const buffer, const int szbuffer, char*& outptr, int& offset, 
    const char* data, int szdata )
{
    MYMSG("TdAlnWriter::BufferData", 9);
    while(szdata > 0)
    {
        if(szbuffer <= offset + szdata) {
            int left = szbuffer - offset;
            if(left >= 0) {
                if(left)
                    strncpy(outptr, data, left);
                WriteToFile(fp, buffer, szbuffer);//WRITE TO FILE
                data += left;
                szdata -= left;
                outptr = buffer;
                offset = 0;
            }
        }

        if(offset + szdata <= szbuffer && szdata > 0) {
            strncpy(outptr, data, szdata);
            outptr += szdata;
            offset += szdata;
            szdata = 0;
        }
    }
}

// -------------------------------------------------------------------------
// WriteToFile: write data to file
//
void TdAlnWriter::WriteToFile(FILE* fp, char* data, int szdata)
{
    if(fwrite(data, sizeof(char), szdata, fp) != (size_t)szdata)
        throw MYRUNTIME_ERROR(
        "TdAlnWriter::WriteToFile: write to file failed.");
}

// -------------------------------------------------------------------------
// GetOutputFilename: make filename for the output file of alignments;
// outfilename, filename to be constructed;
// outdirname, output directory name given;
// qrydesc, query description;
// qrynr, query serial number;
//
void TdAlnWriter::GetOutputFilename( 
    std::string& outfilename,
    const char* outdirname,
    const std::string& qrydesc,
    const int qrynr)
{
    std::string::size_type pos;
    static const int outfmt = CLOptions::GetO_OUTFMT();
    const char* ext = (outfmt == CLOptions::oofJSON)? "json": "out";
    char tail[20];

    sprintf(tail, "__%d.%s", qrynr, ext);

    if(outdirname)
        outfilename = outdirname;

    if(!outfilename.empty() && outfilename[outfilename.size()-1] != DIRSEP)
        outfilename += DIRSEP;

    std::string qn = my_basename(qrydesc.c_str());
    if((pos = qn.rfind(" Chn:")) != std::string::npos)
        qn = qn.substr(0,pos);
    if((pos = qn.rfind(" (M:")) != std::string::npos)
        qn = qn.substr(0,pos);
    if((pos = qn.rfind('.')) != std::string::npos && qn.size()-pos <= 4 )
        qn = qn.substr(0,pos);
    outfilename += qn + tail;
}
