/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __TdClustWriter_h__
#define __TdClustWriter_h__

#include "libutil/mybase.h"

#include <stdio.h>
#include <cmath>

#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>

#include "libutil/CLOptions.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PMBatchStrData.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#define CLUST_TOOBIGDISTANCE 999.9f
#define CLUST_RNDTMSCORE 0.17f

#define CLSTWRTTHREAD_MSG_UNSET -1
#define CLSTWRTTHREAD_MSG_ERROR -2

#if defined(OS_MS_WINDOWS) || defined (__clang__)
#define OMPDECLARE_GetOutputAlnDataField
#else 
#define OMPDECLARE_GetOutputAlnDataField \
    _Pragma("omp declare simd linear(strndx) uniform(h_results) ")
#endif

// _________________________________________________________________________
// Class TdClustWriter
//
// clusters writer thread
//
class TdClustWriter
{
    enum {
        szMaxIdLength = 32,
        szTmpBuffer = KBYTE,
        szWriterBuffer = TIMES4(KBYTE)
    };

public:
    enum TClustWriterThreadMsg {
        clstwrthreadmsgNewData,
        clstwrthreadmsgWrite,
        clstwrthreadmsgTerminate
    };
    enum TClustWriterThreadResponse {
        clstwrtrespmsgNewDataOn,
        clstwrtrespmsgWriting,
        clstwrtrespmsgTerminating
    };

public:
    TdClustWriter( 
        const char* outdirname,
        const std::vector<std::string>& rfilelist,
        const std::vector<std::string>& rdevnames,
        const int nmaxchunkqueries,
        const int nagents
    );

    ~TdClustWriter();

    //{{NOTE: messaging functions accessed from outside!
    void Notify(int msg) {
        {//mutex must be unlocked before notifying
            std::lock_guard<std::mutex> lck(mx_dataccess_);
            req_msg_ = msg;
        }
        cv_msg_.notify_all();
    }
    int Wait(/* int rsp */) {
        //wait until the response is received
        std::unique_lock<std::mutex> lck_msg(mx_dataccess_);
        cv_msg_.wait(lck_msg,
            [this/* ,rsp */]{
                return((req_msg_ == CLSTWRTTHREAD_MSG_UNSET &&
                        rsp_msg_ == CLSTWRTTHREAD_MSG_UNSET) ||
                        rsp_msg_ == CLSTWRTTHREAD_MSG_ERROR ||
                        rsp_msg_ == clstwrtrespmsgTerminating);
            }
        );
        //lock is back; reset the response
        int rspmsg = rsp_msg_;
        // if(rsp_msg_ != CLSTWRTTHREAD_MSG_ERROR)
        //     rsp_msg_ = CLSTWRTTHREAD_MSG_UNSET;
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
                        rsp_msg_ == CLSTWRTTHREAD_MSG_ERROR);
                }
            );
            if(rsp_msg_ == CLSTWRTTHREAD_MSG_ERROR)
                return rsp_msg_;
            if(req_msg_ != CLSTWRTTHREAD_MSG_UNSET)
                continue;
            i++;
        }
        return rsp_msg_;
    }
    void IncreaseQueryNParts(int agent, int qrysernrfrom, int qrysernrto) {
        std::lock_guard<std::mutex> lck(mx_dataccess_);
        if(qrysernrto < qrysernrfrom)
            return;
        if(mstr_set_nmaxchunkqueries_ < qrysernrto - qrysernrfrom + 1)
            throw MYRUNTIME_ERROR(
            "TdClustWriter::IncreaseQueryNParts: Invalid #queries per chunk.");
        if((int)parts_qrs_.size() <= qrysernrto) {
            //let the grand master decide on triggering the write by
            //initializing #parts to 1
            parts_qrs_.resize(qrysernrto+1, 0);
            ResizeVectors(agent, qrysernrto+1);
        }
        for(int qsn = qrysernrfrom; qsn <= qrysernrto; qsn++)
            parts_qrs_[qsn]++;
    }
    void IncreaseQueryNParts(int qrysernrfrom, int qrysernrto) {
        std::lock_guard<std::mutex> lck(mx_dataccess_);
        if(qrysernrto < qrysernrfrom)
            return;
        if((int)parts_qrs_.size() <= qrysernrto)
            throw MYRUNTIME_ERROR(
            "TdClustWriter::IncreaseQueryNParts: Invalid query global identifier.");
        for(int qsn = qrysernrfrom; qsn <= qrysernrto; qsn++)
            parts_qrs_[qsn]++;
    }
    int GetResponse() const {
        std::lock_guard<std::mutex> lck(mx_dataccess_);
        return rsp_msg_;
    }
    //}}

    //{{results submission part
    void PushPartOfResults( 
        int qrysernrbeg,
        int /* nqyposs */, int nqystrs,
        const char** querydesc, char** querypmbeg, char** querypmend, 
        const char** /* bdbCdesc */, char** bdbCpmbeg, char** bdbCpmend,
        TSCounterVar* qrscnt,
        TSCounterVar* cnt,
        unsigned int* passedstats,
        const char* h_results,
        size_t /* sz_alndata */)
    {
        std::unique_lock<std::mutex> lck(mx_dataccess_);
        cv_msg_.wait(lck,
            [this]{
                return req_msg_ == CLSTWRTTHREAD_MSG_UNSET ||
                    rsp_msg_ == CLSTWRTTHREAD_MSG_ERROR ||
                    rsp_msg_ == clstwrtrespmsgTerminating;
            }
        );

        static const std::string preamb = "TdClustWriter::PushPartOfResults: ";
        std::function<void()> lfError = [this,qrscnt,cnt]() {
                if(qrscnt) qrscnt->dec();
                if(cnt) cnt->dec();
                rsp_msg_ = CLSTWRTTHREAD_MSG_ERROR;
        };

        if(rsp_msg_ == CLSTWRTTHREAD_MSG_ERROR ||
           rsp_msg_ == clstwrtrespmsgTerminating) {
            lfError();
            return;
        }

        if(!querypmbeg || !querypmend || !querypmbeg[0] || !querypmend[0] ||
           !bdbCpmbeg || !bdbCpmend || !bdbCpmbeg[0] || !bdbCpmend[0]) {
            lfError();
            throw MYRUNTIME_ERROR(preamb + "Null data.");
        }

        if(nqystrs <= 0 || mstr_set_nmaxchunkqueries_ < nqystrs) {
            lfError();
            throw MYRUNTIME_ERROR(preamb + "Invalid #queries in a chunk.");
        }

        static const int sortby = CLOptions::GetO_SORT();
        static const float covthld = CLOptions::GetB_CLS_COVERAGE();
        static const bool covonesided = CLOptions::GetB_CLS_ONE_SIDED_COVERAGE();

        // //NOTE: sorted queries in block can appear in different order
        // const int qryglobndx = PMBatchStrData::GetFieldAt<INTYPE,pps2DType>(querypmbeg, 0);
        // if(qryglobndx != qrysernrbeg) {
        //     lfError();
        //     throw MYRUNTIME_ERROR(preamb + "Invalid query global identifier.");
        // }

        const int nrfns = PMBatchStrData::GetNoStructs(bdbCpmbeg, bdbCpmend);
        int ntotstrs = 0;

        for(int i = 0; i < nqystrs; i++) {
            //qsn==qryglobndx:
            const int qsn = PMBatchStrData::GetFieldAt<INTYPE,pps2DType>(querypmbeg, i);
            if(qsn < qrysernrbeg || qrysernrbeg + nqystrs <= qsn) {
                lfError();
                throw MYRUNTIME_ERROR(preamb + "Invalid query global identifier.");
            }
            const int vecndx = GetValidatedVecIndexForQrynum(qsn);
            const int lenqry = PMBatchStrData::GetLengthAt(querypmbeg, i);
            parts_qrs_[qsn]--;
            if(parts_qrs_[qsn] <= 0) {
                unsigned int addr = PMBatchStrData::GetAddressAt(querypmbeg, i);
                const char* rsds = querypmbeg[pmv2Drsd] + addr;
                vec_querydesc_[vecndx] = querydesc[i];
                if(boutputseqs_) vec_queryseqn_[vecndx] = std::string(rsds, lenqry);
                ntotresidues_ += lenqry;
                ntotqueries_++;
            }
            int soff = nDevGlobVariables * i;
            int nstrs = passedstats? passedstats[soff + dgvNPassedStrs]: 0;
            if(nstrs != nrfns) {
                lfError();
                throw MYRUNTIME_ERROR(preamb + "Inconsistent number of references in a chunk.");
            }
            // #pragma omp simd
            for(int rn = 0; rn < nstrs; rn++) {
                const int globndx = PMBatchStrData::GetFieldAt<INTYPE,pps2DType>(bdbCpmbeg, rn);
                if(globndx < 0 || qrysernrbeg + nqystrs <= globndx) continue;
                const int lenrfn = PMBatchStrData::GetLengthAt(bdbCpmbeg, rn);
                if(globndx == qsn && lenrfn != lenqry) {
                    lfError();
                    throw MYRUNTIME_ERROR(preamb + "Inconsistent structure ids.");
                }
                //distance:
                const int alnlen = (int)GetOutputAlnDataField<float,dp2oadAlnLength>(h_results, ntotstrs + rn);
                const int ngaps = (int)GetOutputAlnDataField<float,dp2oadNGaps>(h_results, ntotstrs + rn);
                const float nalnres = alnlen - ngaps;//#aligned residues
                float tmscoreq = GetOutputAlnDataField<float,dp2oadScoreQ>(h_results, ntotstrs + rn);
                float tmscorer = GetOutputAlnDataField<float,dp2oadScoreR>(h_results, ntotstrs + rn);
                float rmsd = GetOutputAlnDataField<float,dp2oadRMSD>(h_results, ntotstrs + rn);
                float tmscoregrt = mymax(tmscoreq, tmscorer);
                float tmscorehmn = (0.0f < tmscoregrt)? (2.f * tmscoreq * tmscorer) / (tmscoreq + tmscorer): 0.0f;
                float distance = mymax(0.0f, 1.0f - tmscoregrt);
                if(sortby == CLOptions::osTMscoreReference) distance = mymax(0.0f, 1.0f - tmscorer);
                if(sortby == CLOptions::osTMscoreQuery) distance = mymax(0.0f, 1.0f - tmscoreq);
                if(sortby == CLOptions::osTMscoreHarmonic) distance = mymax(0.0f, 1.0f - tmscorehmn);
                if(sortby == CLOptions::osRMSD) distance = rmsd;
                if(covonesided
                   ? (nalnres < covthld * (float)lenrfn && nalnres < covthld * (float)lenqry)
                   : (nalnres < covthld * (float)lenrfn || nalnres < covthld * (float)lenqry))
                    distance = GetBigDistance();
                //set distance:
                vec_querydsts_[vecndx][globndx] = distance;
            }
            ntotstrs += nstrs;
        }

        if(qrscnt) qrscnt->dec();
        if(cnt) cnt->dec();
        req_msg_ = clstwrthreadmsgNewData;
        //unlock the mutex before notifying
        lck.unlock();
        cv_msg_.notify_all();
    }
    //}}

protected:
    void Execute( void* args );

    void SetResponseError() {
        std::lock_guard<std::mutex> lck(mx_dataccess_);
        rsp_msg_ = CLSTWRTTHREAD_MSG_ERROR;
    }

    OMPDECLARE_GetOutputAlnDataField
    template<typename T, int field>
    T GetOutputAlnDataField(const char* h_results, int strndx) const
    {
        return *(T*)((float*)h_results + nTDP2OutputAlnData * strndx + field);
    }

    int GetValidatedVecIndexForQrynum(size_t qrynum) const
    {
        if(parts_qrs_.size() <= qrynum)
            throw MYRUNTIME_ERROR(
            "TdClustWriter::GetValidatedVecIndexForQrynum: Invalid query serial number.");
        int vecndx =
            mstr_set_nagents_ * mstr_set_nmaxchunkqueries_ - (int)(parts_qrs_.size() - qrynum);
        if(vecndx < 0)
            throw MYRUNTIME_ERROR(
            "TdClustWriter::GetValidatedVecIndexForQrynum: Query serial number exceeds limits.");
        return vecndx;
    }

    float GetBigDistance() const {
        static const int sortby = CLOptions::GetO_SORT();
        if(sortby == CLOptions::osRMSD) return 100.0f;
        return 1.0f;//(1.0f - CLUST_RNDTMSCORE);
    }

    void ResizeVectors(int agent, int newsize)
    {
        static const float largedst = GetBigDistance();
        const int nbeg = agent * mstr_set_nmaxchunkqueries_;
        const int nend = nbeg + mstr_set_nmaxchunkqueries_;
        if(agent < 0 || mstr_set_nagents_ <= agent || (int)vec_querydsts_.size() < nend)
            throw MYRUNTIME_ERROR("TdClustWriter::ResizeVectors: Invalid agent number.");
        vec_pi_.resize(newsize);
        vec_lambda_.resize(newsize);
        vec_ids_.resize(newsize);
        if(boutputseqs_) {
            vec_filepos_.resize(newsize);
            vec_filewrt_.resize(newsize);
        }
        for(int i = nbeg; i < nend; i++) {
            vec_querydsts_[i].resize(newsize);
            //reinitialize every time anew
            std::fill(vec_querydsts_[i].begin(), vec_querydsts_[i].end(), largedst);
        }
    }

    std::string GetIdFromDesc(const std::string& desc) const;

    void OnNewData();
    void SLINK(int ndxnewquery, int vecndx);
    void CLINK(int ndxnewquery, int vecndx);

    void MakeClusters();
    void ArrangeClusters();
    void WriteResults();

    void WriteResultsPlain();
    void WriteSummaryPlain( 
        FILE* fp,
        char* const buffer, const int szbuffer, char*& outptr, int& offset,
        char* tmpbuf, int sztmpbuf,
        const std::vector<std::string>& rdevnames);

private:
    //thread section
    std::thread* tobj_;//thread object
private:
    //{{messaging
    std::condition_variable cv_msg_;//condition variable for messaging
    mutable std::mutex mx_dataccess_;//mutex for accessing class data
    int req_msg_;//request message issued for thread
    int rsp_msg_;//private response message
    const char* mstr_set_outdirname_;//output directory name
    std::vector<std::string> mstr_set_rfilelist_;//filelist of reference (target) structures
    std::vector<std::string> mstr_set_rdevnames_;//list of device names
    const int mstr_set_nmaxchunkqueries_;//#max queries per chunk
    const int mstr_set_nagents_;//#operating agents
    //{{query and summary data:
    std::vector<std::string> vec_querydesc_;//query descriptions
    std::vector<std::string> vec_queryseqn_;//query sequences
    std::vector<std::vector<float>> vec_querydsts_;//running distances for the queries under process
    std::vector<int> vec_pi_;//indices in point representationn of clusters
    std::vector<float> vec_lambda_;//distances in point representationn of clusters
    std::vector<std::string> vec_ids_;//(query) ids clustered
    std::vector<long> vec_filepos_;//file positions for query sequences
    std::vector<int> vec_filewrt_;//written number of bytes of query sequences
    int querytoprocessnext_;//index of the query to process next
    size_t ntotqueries_;//total number of queries
    size_t ntotresidues_;//total number of residues
    //}}
    //{{clustering data:
    //forward pointers:
    //subclusters; inner-most vector contains query indices
    std::vector<std::vector<int>> cls_inlayer_;
    //actual clusters; inner-most vector contains subcluster indices
    std::vector<std::vector<int>> cls_clusters_;
    //backward pointers:
    //indices of subclusters (multiple can belong to the same cluster) for each query index;
    std::vector<int> cls_tolayer_;
    //indices of clusters for each subcluster index;
    std::vector<int> cls_toclusters_;
    //}}
    //buffer for writing to file:
    char buffer_[szWriterBuffer];
    //vector of the number of parts to be processed for each query:
    std::vector<int> parts_qrs_;
    //flag of producing sequences for each cluster
    const int boutputseqs_;
    //temporary file for sequences
    std::unique_ptr<FILE,void(*)(FILE*)> tmpfp_;
};

////////////////////////////////////////////////////////////////////////////
// TdClustWriter INLINES
//
// SLINK: Progressive SLINK (single linkage) algorithm for up to ndxnewquery
// data points (Sibson, 1973);
// NOTE: necessary allocations assumed to be done before;
//
inline
void TdClustWriter::SLINK(int ndxnewquery, int vecndx)
{
    std::vector<float>& vec_mu = vec_querydsts_[vecndx];
    //step 1: initialization
    vec_pi_[ndxnewquery] = ndxnewquery;
    vec_lambda_[ndxnewquery] = CLUST_TOOBIGDISTANCE;
    //step 2: distances in vec_mu/vec_querydsts_[.]
    //step 3: main loop/actions
    for(int i = 0; i < ndxnewquery; i++) {
        if(vec_lambda_[i] >= vec_mu[i]) {
            vec_mu[vec_pi_[i]] = mymin(vec_mu[vec_pi_[i]], vec_lambda_[i]);
            vec_lambda_[i] = vec_mu[i];
            vec_pi_[i] = ndxnewquery;
        }
        if(vec_lambda_[i] < vec_mu[i])
            vec_mu[vec_pi_[i]] = mymin(vec_mu[vec_pi_[i]], vec_mu[i]);
    }
    //step 4: lambdas check against one another
    for(int i = 0; i < ndxnewquery; i++) {
        if(vec_lambda_[i] >= vec_lambda_[vec_pi_[i]])
            vec_pi_[i] = ndxnewquery;
    }
}

// -------------------------------------------------------------------------
// CLINK: Progressive CLINK (complete linkage) algorithm for up to
// ndxnewquery data points (Defays, 1977);
// NOTE: necessary allocations assumed to be done before;
//
inline
void TdClustWriter::CLINK(int ndxnewquery, int vecndx)
{
    std::vector<float>& vec_mu = vec_querydsts_[vecndx];
    //step 1: initialization
    vec_pi_[ndxnewquery] = ndxnewquery;
    vec_lambda_[ndxnewquery] = CLUST_TOOBIGDISTANCE;
    //step 2: distances in vec_mu/vec_querydsts_[.]
    //step 3: main loop 1
    for(int i = 0; i < ndxnewquery; i++) {
        if(vec_lambda_[i] < vec_mu[i]) {
            vec_mu[vec_pi_[i]] = mymax(vec_mu[vec_pi_[i]], vec_mu[i]);
            vec_mu[i] = CLUST_TOOBIGDISTANCE;
        }
    }
    //step 4: set a = n
    int a = ndxnewquery - 1;
    //step 5: main loop 2
    for(int i = 0; i < ndxnewquery; i++) {
        int ri = ndxnewquery - 1 - i;
        if(vec_lambda_[ri] >= vec_mu[vec_pi_[ri]]) {
            if(vec_mu[ri] < vec_mu[a]) a = ri;
        }
        if(vec_lambda_[ri] < vec_mu[vec_pi_[ri]])
            vec_mu[ri] = CLUST_TOOBIGDISTANCE;
    }
    //step 6: set b = pi[a]; c = lambda[a]; pi[a] = n+1; lambda[a] = mu[a]
    int b = a, c = a;
    if(0 <= a) {
        b = vec_pi_[a]; vec_pi_[a] = ndxnewquery;
        c = vec_lambda_[a]; vec_lambda_[a] = vec_mu[a];
    }
    //step 7: point representation adjustment
    if(a < ndxnewquery - 1) {
        while(b < ndxnewquery - 1) {
            int d = vec_pi_[b]; vec_pi_[b] = ndxnewquery;
            int e = vec_lambda_[b]; vec_lambda_[b] = c;
            b = d; c = e;
        }
        if(b == ndxnewquery - 1) {
            vec_pi_[b] = ndxnewquery;
            vec_lambda_[b] = c;
        }
    }
    //step 8: final loop along with lambdas check
    for(int i = 0; i < ndxnewquery; i++) {
       if(vec_pi_[vec_pi_[i]] == ndxnewquery)
            if(vec_lambda_[i] >= vec_lambda_[vec_pi_[i]])
                vec_pi_[i] = ndxnewquery;
    }
}

// -------------------------------------------------------------------------
//
inline
void TdClustWriter::WriteResults()
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

#endif//__TdClustWriter_h__
