/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/mybase.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>

#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include <numeric>
#include <mutex>
#include <condition_variable>
#include <thread>

#include "libutil/CLOptions.h"
#include "TdAlnWriter.h"
#include "TdClustWriter.h"

//100 queries for initial reservation of vectors:
#define CLUSTVECRESERVATION (256 * KBYTE)
//initial reservation for #clusters:
#define NCLUSTERSRESERVATION (32 * KBYTE)
//initial reservation for #clusters:
#define NSUBCLUSTERSRESERVATION (4)

// _________________________________________________________________________
// Class TdClustWriter
//
// Constructor
//
TdClustWriter::TdClustWriter( 
    const char* outdirname,
    const std::vector<std::string>& rfilelist,
    const std::vector<std::string>& rdevnames,
    const int nmaxchunkqueries,
    const int nagents)
:
    tobj_(NULL),
    req_msg_(CLSTWRTTHREAD_MSG_UNSET),
    rsp_msg_(CLSTWRTTHREAD_MSG_UNSET),
    mstr_set_outdirname_(outdirname),
    mstr_set_nmaxchunkqueries_(nmaxchunkqueries),
    mstr_set_nagents_(nagents),
    querytoprocessnext_(0),
    ntotqueries_(0),
    ntotresidues_(0),
    boutputseqs_(CLOptions::GetB_CLS_OUT_SEQUENCES()),
    tmpfp_(nullptr, [](FILE* p){if(p) fclose(p);})
{
    MYMSG("TdClustWriter::TdClustWriter", 3);

    if(mstr_set_nmaxchunkqueries_ < 1 || mstr_set_nagents_ < 1)
        throw MYRUNTIME_ERROR(
        "TdClustWriter::TdClustWriter: Invalid max #queries per chunk.");

    //#running entries:
    const int nrunentries = mstr_set_nmaxchunkqueries_ * mstr_set_nagents_;

    vec_querydesc_.resize(nrunentries);
    vec_queryseqn_.resize(nrunentries);
    vec_querydsts_.resize(nrunentries);
    for(size_t i = 0; i < vec_querydsts_.size(); i++)
        vec_querydsts_[i].reserve(CLUSTVECRESERVATION);
    vec_pi_.reserve(CLUSTVECRESERVATION);
    vec_lambda_.reserve(CLUSTVECRESERVATION);
    vec_ids_.reserve(CLUSTVECRESERVATION);
    if(boutputseqs_) {
        vec_filepos_.reserve(CLUSTVECRESERVATION);
        vec_filewrt_.reserve(CLUSTVECRESERVATION);
    }

    parts_qrs_.reserve(CLUSTVECRESERVATION);

    cls_inlayer_.reserve(NCLUSTERSRESERVATION);
    cls_clusters_.reserve(NCLUSTERSRESERVATION);
    cls_tolayer_.reserve(CLUSTVECRESERVATION);
    cls_toclusters_.reserve(NCLUSTERSRESERVATION);

    for(const std::string& dbn: rfilelist) {
        //std::string::npos+1==0
        std::string::size_type pos = 0;//dbn.rfind(DIRSEP) + 1;
        mstr_set_rfilelist_.push_back(dbn.substr(pos));
    }

    for(const std::string& dn: rdevnames)
        mstr_set_rdevnames_.push_back(dn);

    tobj_ = new std::thread(&TdClustWriter::Execute, this, (void*)NULL);
}

// Destructor
//
TdClustWriter::~TdClustWriter()
{
    MYMSG("TdClustWriter::~TdClustWriter", 3);
    if( tobj_ ) {
        tobj_->join();
        delete tobj_;
        tobj_ = NULL;
    }
}

// -------------------------------------------------------------------------
// Execute: thread's starting point for execution
//
void TdClustWriter::Execute( void* )
{
    MYMSG("TdClustWriter::Execute", 3);
    myruntime_error mre;
    char msgbuf[BUF_MAX];
    std::string tmpfilename;

    try {
        if(boutputseqs_) {
            if(mstr_set_outdirname_) tmpfilename = mstr_set_outdirname_;
            if(!tmpfilename.empty() && tmpfilename.back() != DIRSEP)
                tmpfilename += DIRSEP;
            tmpfilename += "__gtalignclust.tmp__";
            //use mode b for OS_MS_WINDOWS to not use translation
            tmpfp_.reset(fopen(tmpfilename.c_str(), "w+b"));
            if(!tmpfp_)
                throw MYRUNTIME_ERROR("TdClustWriter::Execute: "
                "Failed to open temporary file for writing: " + tmpfilename);
        }

        while(1) {
            //wait for a message
            std::unique_lock<std::mutex> lck_msg(mx_dataccess_);

            cv_msg_.wait(lck_msg,
                [this]{
                    return (
                        (0 <= req_msg_ && req_msg_ <= clstwrthreadmsgTerminate) || 
                        req_msg_ == CLSTWRTTHREAD_MSG_ERROR
                    );
                }
            );

            MYMSGBEGl(3)
                sprintf(msgbuf, "TdClustWriter::Execute: Msg %d", req_msg_);
                MYMSG(msgbuf, 3);
            MYMSGENDl

            //thread owns the lock after the wait;
            //read message req_msg_
            int reqmsg = req_msg_;

            //unset the message to avoid live cycle when starting over the loop
            req_msg_ = CLSTWRTTHREAD_MSG_UNSET;

            //set response msg to error upon exception
            rsp_msg_ = CLSTWRTTHREAD_MSG_ERROR;
            int rspmsg = rsp_msg_;

            switch(reqmsg) {
                case clstwrthreadmsgNewData:
                        OnNewData();
                        rspmsg = CLSTWRTTHREAD_MSG_UNSET;
                        break;
                case clstwrthreadmsgWrite:
                        ;;
                        MakeClusters();
                        ArrangeClusters();
                        WriteResults();
                        ;;
                        //unset response code
                        rspmsg = CLSTWRTTHREAD_MSG_UNSET;
                        break;
                case clstwrthreadmsgTerminate:
                        rspmsg = clstwrtrespmsgTerminating;
                        break;
                default:
                        rspmsg = CLSTWRTTHREAD_MSG_UNSET;
                        break;
            };

            MYMSGBEGl(3)
                sprintf(msgbuf, "TdClustWriter::Execute: Msg %d Rsp %d", reqmsg, rspmsg);
                MYMSG(msgbuf, 3);
            MYMSGENDl

            //save response code
            rsp_msg_ = rspmsg;

            //unlock the mutex and notify awaiting threads
            lck_msg.unlock();
            cv_msg_.notify_all();

            if(reqmsg < 0 ||
               reqmsg == clstwrthreadmsgWrite || reqmsg == clstwrthreadmsgTerminate)
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
    }

    if(file_exists(tmpfilename.c_str())) {
        tmpfp_.reset(nullptr);
        std::remove(tmpfilename.c_str());
    }
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// OnNewData: process portion of reference data for a block of queries
//
void TdClustWriter::OnNewData()
{
    MYMSG("TdClustWriter::OnNewData", 4);
    static const std::string preamb = "TdClustWriter::OnNewData: ";
    static const int clsalgorithm = CLOptions::GetB_CLS_ALGORITHM();

    // if((int)parts_qrs_.size() <= querytoprocessnext_)
    //     throw MYRUNTIME_ERROR(preamb + "Invalid query index to process.");

    for(; querytoprocessnext_ < (int)parts_qrs_.size() &&
          parts_qrs_[querytoprocessnext_] == 0; querytoprocessnext_++)
    {
        const int vecndx = GetValidatedVecIndexForQrynum(querytoprocessnext_);

        if(clsalgorithm == CLOptions::bcCompleteLinkage) 
            CLINK(querytoprocessnext_, vecndx);
        else /* if(clsalgorithm == CLOptions::bcSingleLinkage) */
            SLINK(querytoprocessnext_, vecndx);

        vec_ids_[querytoprocessnext_] = GetIdFromDesc(vec_querydesc_[vecndx]);

        if(!boutputseqs_) continue;

        vec_filepos_[querytoprocessnext_] = ftell(tmpfp_.get());

        int nwritten =
            fprintf(tmpfp_.get(), ">%s%s%s%s",
                vec_querydesc_[vecndx].c_str(), NL, vec_queryseqn_[vecndx].c_str(), NL);
        if(nwritten <= 0)
            throw MYRUNTIME_ERROR(preamb + "Failed to write a sequence to tmp file.");

        vec_filewrt_[querytoprocessnext_] = nwritten;
    }
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// MakeClusters: make clusters once all iterations of SLINK have finished
//
void TdClustWriter::MakeClusters()
{
    MYMSG("TdClustWriter::MakeClusters", 4);
    static const std::string preamb = "TdClustWriter::MakeClusters: ";

    if(vec_pi_.size() != vec_lambda_.size())
        throw MYRUNTIME_ERROR(preamb + "Inconsistent number of query indices.");

    static const int sortby = CLOptions::GetO_SORT();
    static const float clsthld = CLOptions::GetB_CLS_THRESHOLD();

    switch(sortby) {
        case CLOptions::osTMscoreGreater:
        case CLOptions::osTMscoreReference:
        case CLOptions::osTMscoreQuery:
        case CLOptions::osTMscoreHarmonic:
            const_cast<float&>(clsthld) = mymax(0.0f, 1.0f - clsthld);
    };

    //check the sizes of pointer data structures
    std::function<void()> lfCheckSizes = [this]() {
        if(cls_toclusters_.size() != cls_inlayer_.size())
            throw MYRUNTIME_ERROR(preamb + "Inconsistent cluster indices.");
    };

    //create a subcluster and cluster for the given query indices
    std::function<void(int, int)> lfNewCluster = [this](int q, int qq) {
        //forward pointers:
        int ndxlyr = cls_inlayer_.size();
        int ndxclt = cls_clusters_.size();
        std::vector<int> subclst, clst;
        clst.reserve(NSUBCLUSTERSRESERVATION);
        subclst.reserve(NSUBCLUSTERSRESERVATION);
        subclst.push_back(q);
        if(0 <= qq) subclst.push_back(qq);
        cls_inlayer_.push_back(subclst);
        clst.push_back(ndxlyr);
        cls_clusters_.push_back(clst);
        //backward pointers:
        cls_tolayer_[q] = ndxlyr;
        if(0 <= qq) cls_tolayer_[qq] = ndxlyr;
        cls_toclusters_.resize(ndxlyr + 1, -1);
        cls_toclusters_[ndxlyr] = ndxclt;
    };

    cls_tolayer_.resize(vec_pi_.size(), -1);

    for(size_t q = 0; q < vec_pi_.size(); q++) {
        if((q & 1023) == 0) lfCheckSizes();
        int qq = vec_pi_[q];
        int ndxlyr1 = cls_tolayer_[q];
        int ndxlyr2 = cls_tolayer_[qq];
        float dst = vec_lambda_[q];
        if(q+1 < vec_pi_.size() && qq <= (int)q)
            throw MYRUNTIME_ERROR(preamb + "Invalid query indices.");
        if(dst <= clsthld) {
            if(ndxlyr1 < 0 && ndxlyr2 < 0) {
                lfNewCluster(q, qq);
            }
            else if(0 <= ndxlyr1 && 0 <= ndxlyr2) {
                if((int)cls_toclusters_.size() <= ndxlyr1 || (int)cls_toclusters_.size() <= ndxlyr2)
                    throw MYRUNTIME_ERROR(preamb + "Invalid subcluster index.");
                int ndxclt1 = cls_toclusters_[ndxlyr1];
                int ndxclt2 = cls_toclusters_[ndxlyr2];
                if((int)cls_clusters_.size() <= ndxclt1 || (int)cls_clusters_.size() <= ndxclt2)
                    throw MYRUNTIME_ERROR(preamb + "Invalid cluster index.");
                //merge clusters; forward pointers:
                if(cls_clusters_[ndxclt1].size() < cls_clusters_[ndxclt2].size()) myswap(ndxclt1, ndxclt2);
                cls_clusters_[ndxclt1].insert(
                    cls_clusters_[ndxclt1].end(), cls_clusters_[ndxclt2].cbegin(), cls_clusters_[ndxclt2].cend());
                //backward pointers:
                std::for_each(cls_clusters_[ndxclt2].cbegin(), cls_clusters_[ndxclt2].cend(),
                        [this,ndxclt1](const int& ndx) {cls_toclusters_[ndx] = ndxclt1;});
                cls_clusters_[ndxclt2].clear();
            }
            else if(ndxlyr1 < 0) {
                cls_inlayer_[ndxlyr2].push_back(q);//forward pointers: insert
                cls_tolayer_[q] = ndxlyr2;//insert backward pointers
            }
            else if(ndxlyr2 < 0) {
                cls_inlayer_[ndxlyr1].push_back(qq);//forward pointers: insert
                cls_tolayer_[qq] = ndxlyr1;//insert backward pointers
            }
        } else {//above threshold
            if(ndxlyr1 < 0) lfNewCluster(q, -1);
            if((int)q != qq && ndxlyr2 < 0) lfNewCluster(qq, -1);
        }
    }

    lfCheckSizes();
}

// -------------------------------------------------------------------------
// ArrangeClusters: arrange clusters: set representatives and sort by size
//
void TdClustWriter::ArrangeClusters()
{
    MYMSG("TdClustWriter::ArrangeClusters", 4);
    static const std::string preamb = "TdClustWriter::ArrangeClusters: ";

    //only forward pointer structures needed; reuse backward pointer structures;
    //reuse vec_pi_ for cluster indices too;
    vec_pi_.clear();
    cls_tolayer_.clear();
    cls_toclusters_.clear();

    for(size_t ci = 0; ci < cls_clusters_.size(); ci++) {
        int szclst = 0;//cluster size
        int ndxqryrep = -1;//representative query index
        float mindst = CLUST_TOOBIGDISTANCE;

        for(size_t sci = 0; sci < cls_clusters_[ci].size(); sci++)
        {
            int ndxsub = cls_clusters_[ci][sci];
            if((int)cls_inlayer_.size() <= ndxsub || ndxsub < 0)
                throw MYRUNTIME_ERROR(preamb + "Invalid subcluster index.");

            for(size_t sqi = 0; sqi < cls_inlayer_[ndxsub].size(); sqi++) {
                int ndxqry = cls_inlayer_[ndxsub][sqi];
                if((int)vec_lambda_.size() <= ndxqry || ndxqry < 0)
                    throw MYRUNTIME_ERROR(preamb + "Invalid query index.");
                //ensure valid indices in the case of singletons
                if(vec_lambda_[ndxqry] < mindst || ndxqryrep < 0) {
                    mindst = vec_lambda_[ndxqry];
                    ndxqryrep = ndxqry;
                }
                szclst++;
            }
        }

        if(0 < szclst) {
            if(ndxqryrep < 0)
                throw MYRUNTIME_ERROR(preamb + "Invalid representative.");
            cls_tolayer_.push_back(ndxqryrep);
            cls_toclusters_.push_back(szclst);
            vec_pi_.push_back(ci);//reuse vec_pi_!
        }
    }
}



// =========================================================================
// WriteResultsPlain: write clusters to file(s)
//
void TdClustWriter::WriteResultsPlain()
{
    MYMSG("TdClustWriter::WriteResultsPlain", 4);
    static const std::string preamb = "TdClustWriter::WriteResultsPlain: ";
    const unsigned int dscwidth = MAX_DESCRIPTION_LENGTH;//no wrap (pathname)

    char tmpinfo[szWriterBuffer];
    char* pb = buffer_, *ptmp = tmpinfo;
    int size = 0;//#characters written

    std::string filenamebase, filename;
    std::unique_ptr<FILE,void(*)(FILE*)> fp(
        nullptr,
        [](FILE* p) {if(p) fclose(p);}
    );

    if(mstr_set_outdirname_) filenamebase = mstr_set_outdirname_;
    if(!filenamebase.empty() && filenamebase.back() != DIRSEP) filenamebase += DIRSEP;
    filenamebase += "gtalignclusters";
    filename = filenamebase + ".lst";

    //use mode b for OS_MS_WINDOWS to not use translation
    fp.reset(fopen(filename.c_str(), "wb"));

    if(!(fp))
        throw MYRUNTIME_ERROR(preamb + "Failed to open file for writing: " + filename);


    //write info
    size += TdAlnWriter::WritePrognamePlain(pb, szWriterBuffer, dscwidth);

    TdAlnWriter::WriteCommandLinePlain(fp.get(),
        buffer_, szWriterBuffer, pb, size);

    TdAlnWriter::WriteSearchInformationPlain(fp.get(),
        buffer_, szWriterBuffer, pb, size,
        ptmp, szWriterBuffer/*maxsize*/,
        mstr_set_rfilelist_,
        ntotresidues_, ntotqueries_,
        0.0f/*tmsthld*/, 0/*indent*/, true/*hitsfound*/,true/*clustering*/);

    WriteSummaryPlain(fp.get(),
        buffer_, szWriterBuffer, pb, size,
        ptmp, szWriterBuffer/*maxsize*/,
        mstr_set_rdevnames_);


    //sort clusters by size
    std::vector<int> tmpndxs;
    tmpndxs.resize(vec_pi_.size());
    //indices from 0 to #structures
    std::iota(tmpndxs.begin(), tmpndxs.end(), 0);
    std::sort(tmpndxs.begin(), tmpndxs.end(),
        [this](size_t n1, size_t n2) {
            //cls_toclusters_ contain sizes now; sort in descending order
            return (cls_toclusters_[n1] == cls_toclusters_[n2])
                ?   (vec_ids_[cls_tolayer_[n1]].compare(vec_ids_[cls_tolayer_[n2]]) < 0)
                :   (cls_toclusters_[n1] > cls_toclusters_[n2]);
        });


    //function for writing a string
    std::function<void(FILE*, const char*, bool)> lfBufferString =
        [this,&pb,&size,ptmp](FILE* fp, const char* s, bool addspace) {
            int wrttn = sprintf(ptmp,"%s%s", s, (addspace? " ":""));
            TdAlnWriter::BufferData(fp,
                buffer_, szWriterBuffer,  pb, size,  ptmp, wrttn);
        };

    //function for writing a sequence to file
    std::function<void(FILE*, FILE*, int)> lfWriteSequence =
        [this,ptmp](FILE* fpfa, FILE* fptmp, int ndxqry) {
            if(fseek(fptmp, vec_filepos_[ndxqry], SEEK_SET) < 0)
                throw MYRUNTIME_ERROR(preamb + "Failed to fseek to a tmp file position.");
            int wrtsize = vec_filewrt_[ndxqry];
            for(int sz = 0; sz < wrtsize; sz += szWriterBuffer) {
                int sztoread = mymin((int)szWriterBuffer, wrtsize - sz);
                int szdata = fread(ptmp, sizeof(char), sztoread, fptmp);
                if(szdata != sztoread)
                    throw MYRUNTIME_ERROR(preamb + "Failed to read from the tmp file.");
                TdAlnWriter::WriteToFile(fpfa, ptmp, szdata);
            }
        };


    //write clusters and sequences if required
    for(size_t ti = 0; ti < tmpndxs.size(); ti++) {
        // cls_tolayer_ holds representative query index (ndxqryrep);
        // cls_toclusters_ holds cluster sizes (szclst);
        // vec_pi_ holds cluster indices (ci);
        int ndxt = tmpndxs[ti];//sorted index
        int ci = vec_pi_[ndxt];//original (forward) cluster index
        int ndxqryrep = cls_tolayer_[ndxt];//representative query
        // int szclst = cls_toclusters_[ndxt];

        if((int)cls_clusters_.size() <= ci || ci < 0)
            throw MYRUNTIME_ERROR(preamb + "Invalid cluster index.");

        if((int)vec_ids_.size() <= ndxqryrep || ndxqryrep < 0)
            throw MYRUNTIME_ERROR(preamb + "Invalid representative query.");

        lfBufferString(fp.get(), vec_ids_[ndxqryrep].c_str(), true/*space*/);

        std::string fafilename = filenamebase + "_" + std::to_string(ti) + ".fa";
        std::unique_ptr<FILE,void(*)(FILE*)> fpfa(
            nullptr, [](FILE* p) {if(p) fclose(p);}
        );

        if(boutputseqs_) {
            fpfa.reset(fopen(fafilename.c_str(), "wb"));
            if(!(fpfa))
                throw MYRUNTIME_ERROR(preamb + "Failed to open file for writing: " + fafilename);
            lfWriteSequence(fpfa.get(), tmpfp_.get(), ndxqryrep);
        }

        for(size_t sci = 0; sci < cls_clusters_[ci].size(); sci++)
        {
            int ndxsub = cls_clusters_[ci][sci];
            if((int)cls_inlayer_.size() <= ndxsub || ndxsub < 0)
                throw MYRUNTIME_ERROR(preamb + "Invalid subcluster index.");

            for(size_t sqi = 0; sqi < cls_inlayer_[ndxsub].size(); sqi++)
            {
                int ndxqry = cls_inlayer_[ndxsub][sqi];
                if((int)vec_ids_.size() <= ndxqry || ndxqry < 0)
                    throw MYRUNTIME_ERROR(preamb + "Invalid query index.");
                if(ndxqry == ndxqryrep) continue;
                lfBufferString(fp.get(), vec_ids_[ndxqry].c_str(), true/*space*/);
                if(boutputseqs_) lfWriteSequence(fpfa.get(), tmpfp_.get(), ndxqry);
            }
        }

        lfBufferString(fp.get(), NL, false/*space*/);
    }

    //flush to file
    if(size > 0) TdAlnWriter::WriteToFile(fp.get(), buffer_, size);
}

// -------------------------------------------------------------------------
// WriteSummaryPlain: write summary: devices and time;
// fp, file pointer;
// buffer, buffer to store data in;
// szbuffer, size of the buffer;
// outptr, varying address of the pointer pointing to a location in the buffer;
// offset, outptr offset from the beginning of the buffer (fill size);
// tmpbuf, address of a temporary buffer for writing;
// sztmpbuf, size of tmpbuf;
// rdevnames, list of device names;
//
void TdClustWriter::WriteSummaryPlain( 
    FILE* fp,
    char* const buffer, const int szbuffer, char*& outptr, int& offset,
    char* tmpbuf, int /* sztmpbuf */,
    const std::vector<std::string>& rdevnames)
{
    static const size_t maxszname = MAX_FILENAME_LENGTH_TOSHOW;
    static const int sznl = (int)strlen(NL);
    int written;
    // const char* dots = "..." NL NL;
    // const int szdots = (int)strlen(dots);

    std::chrono::high_resolution_clock::time_point tnow = 
    std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elpsd = tnow - gtSTART;

    written = sprintf(tmpbuf,"%s%s Devices:%s",NL,NL,NL);
    TdAlnWriter::BufferData(fp,
        buffer, szbuffer, outptr, offset,
        tmpbuf, written);

    //print filenames
    for(const std::string& name: rdevnames) {
        size_t szname = name.size();
        if(maxszname < szname)
            szname = maxszname;
        strncpy(tmpbuf, name.substr(name.size()-szname).c_str(), szname);
        //termination at the beginning
        if(szname < name.size() && 3 < szname)
            for(int i=0; i<3; i++) tmpbuf[i] = '.';
        TdAlnWriter::BufferData(fp,
            buffer, szbuffer, outptr, offset,
            tmpbuf, szname);
        TdAlnWriter::BufferData(fp,
            buffer, szbuffer, outptr, offset,
            NL, sznl);
    }

    written = sprintf(tmpbuf,"%sTime elapsed from process initiation: %.6f sec%s", NL, elpsd.count(), NL);
    TdAlnWriter::BufferData(fp,
        buffer, szbuffer, outptr, offset,
        tmpbuf, written);

    written = sprintf(tmpbuf,
        "%s============================================================================%s%s",
        NL,NL,NL);
    TdAlnWriter::BufferData(fp,
        buffer, szbuffer, outptr, offset,
        tmpbuf, written);
}



// =========================================================================
// GetIdFromDesc: generate id from a given description
//
std::string TdClustWriter::GetIdFromDesc(const std::string& desc) const
{
    static const char* pdbstr = "pdb";
    static const char* chnstr = " Chn:";
    static const char* modstr = " (M:";
    static const size_t lenpdbstr = strlen(pdbstr);
    static const int lenchnstr = strlen(chnstr);
    static const int lenmodstr = strlen(modstr);
    enum {maxchnlen = 5, maxchnlen1};
    char chn[maxchnlen1] = {0}, mod[maxchnlen1] = {0};
    int i;
    std::string::size_type pos, tmp;
    std::string qid = my_basename(desc.c_str());

    if(lenpdbstr < qid.size() && qid.compare(0, lenpdbstr, pdbstr, lenpdbstr) == 0)
        qid = qid.substr(lenpdbstr);

    if((pos = qid.rfind(modstr)) != std::string::npos) {
        mod[0] = '_';
        for(tmp = pos + lenmodstr, i = 1;
            i < maxchnlen && tmp < qid.size() && qid[tmp] != ')' && qid[tmp] != ' ';
            tmp++, i++)
            mod[i] = qid[tmp];
        mod[i] = 0;
        qid = qid.substr(0,pos);
    }

    if((pos = qid.rfind(chnstr)) != std::string::npos) {
        chn[0] = '_';
        for(tmp = pos + lenchnstr, i = 1;
            i < maxchnlen && tmp < qid.size() && qid[tmp] != ' ';
            tmp++, i++)
            chn[i] = qid[tmp];
        chn[i] = 0;
        qid = qid.substr(0,pos);
    }

    if((pos = qid.find('.')) != std::string::npos && 4 <= pos)
        qid = qid.substr(0,pos);
    else if((pos = qid.rfind('.')) != std::string::npos && qid.size()-pos <= 4)
        qid = qid.substr(0,pos);

    if(chn[0] && chn[1] != 0) qid += chn;
    if(mod[0] && mod[1] != 0 && mod[1] != '0') qid += mod;

    if(szMaxIdLength < qid.size()) {
        qid = qid.substr(qid.size() - szMaxIdLength);
        qid[0] = qid[1] = qid[2] = '.';
    }

    return qid;
}
