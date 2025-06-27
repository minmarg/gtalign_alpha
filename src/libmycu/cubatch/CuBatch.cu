/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/mybase.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <mutex>
#include <cmath>

#include "libutil/mptimer.h"
#include "libutil/CLOptions.h"
#include "tsafety/TSCounterVar.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libgenp/gdats/PMBatchStrData.h"
#include "libgenp/goutp/TdAlnWriter.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/cugraphs.cuh"
#include "libmycu/cuproc/Devices.h"
#include "libmycu/culayout/CuDeviceMemory.cuh"
#include "libmycu/cufilter/coverage.cuh"
#include "libmycu/cufilter/similarity.cuh"
#include "libmycu/cufilter/reformatter.cuh"
#include "libmycu/cuss/cusecstr.cuh"
#include "libmycu/cudp/dpw_btck.cuh"
//#include "libmycu/cusa/cuspecsim.cuh"
//#include "libmycu/cusa/cuspecsim2.cuh"
//#include "libmycu/cusa/cuspecsim3.cuh"
//#include "libmycu/cusa/cuspecsim32.cuh"
//#include "libmycu/cusa/cuspecsim7.cuh"
#include "libmycu/custages/fragment.cuh"
#include "libmycu/custage1/custage1.cuh"
#include "libmycu/custage1/custage2.cuh"
#include "libmycu/custage1/custage_ssrr.cuh"
#include "libmycu/custage1/custage_frg3.cuh"
#include "libmycu/custage1/custage_fin.cuh"
#include "libgenp/goutp/TdFinalizer.h"
#include "CuBatch.cuh"

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// constructor
//
CuBatch::CuBatch(
    size_t ringsize,
    CuDeviceMemory* dmem, int dareano,
    TdAlnWriter* writer, TdClustWriter* clustwriter,
    size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs)
:
    ringsize_(ringsize),
    dmem_(dmem),
    devareano_(dareano),
    chunkdatasize_(chunkdatasize),
    chunkdatalen_(chunkdatalen),
    chunknstrs_(chunknstrs),
    ndxbatchstrdata_(0),
    filterdata_(nullptr),
    passedstatscntrd_(nullptr),
//     cbsm_(nullptr),
//     cbdp_(nullptr),
    cbpfin_(nullptr),
    clustwriter_(clustwriter),
    scorethld_(0.0f),
    curdbxpad_(0),
    dbxpadphase2_(0),
    h_results_(NULL),
    lockedresmem_(false),
    sz_mem_results_(0UL),
    limit_beg_results_(0UL)
{
    MYMSG("CuBatch::CuBatch", 4);

    if(dmem_ == NULL)
        throw MYRUNTIME_ERROR("CuBatch::CuBatch: Null device memory object.");

    if(devareano_ < 0 || dmem_->GetNAreas() <= devareano_)
        throw MYRUNTIME_ERROR("CuBatch::CuBatch: Invalid device memory area number.");

    if(dmem_->GetMemAlignment() < CUL2CLINESIZE * sizeof(float)) {
        char errbuf[BUF_MAX];
        sprintf(errbuf,
            "Memory alignment size (%zu) < defined CUL2CLINESIZE (%d) * %zu. "
            "Decrease CUL2CLINESIZE to a multiple of 32 and recompile.",
            dmem_->GetMemAlignment(), CUL2CLINESIZE, sizeof(float));
        throw MYRUNTIME_ERROR(errbuf);
    }

    static const float seqsimthrscore = CLOptions::GetP_PRE_SIMILARITY();
    static const float prescore = CLOptions::GetP_PRE_SCORE();
    const int maxnqrsperchunk = CLOptions::GetDEV_QRS_PER_CHUNK();
    const size_t szstats = maxnqrsperchunk * nDevGlobVariables;
    const size_t nfilterdata = GetCurrentMaxNDbStrs() * nTFilterData;
    //filtering condition for clustering:
    bool condition4filter0 = 
        clustwriter_;// clustwriter_ && (clstonesided == 0) && (clstcoverage > 0.001f);

    //if filtering in use, all GPUs work on their own copy
    if(/*1 < ringsize_ &&*/ (0.0f < seqsimthrscore || 0.0f < prescore || condition4filter0)) {
        //NOTE: allocate NBATCHSTRDATA buffers to have one more available when
        //NOTE: new data arrives; synchronization with Finalizer makes 2 enough;
        for(int i = 0; i < NBATCHSTRDATA; i++)
            bdbCstruct_[i].AllocateSpace(chunkdatasize_, chunkdatalen_, chunknstrs_);
    }

    filterdata_.reset(new unsigned int[nfilterdata]);
    passedstatscntrd_.reset(new unsigned int[szstats]);

    if(!filterdata_ || !passedstatscntrd_)
        throw MYRUNTIME_ERROR("CuBatch::CuBatch: Not enough memory.");

//     cbsm_.reset(new CuBatchSM());
//     if(!cbsm_)
//         throw MYRUNTIME_ERROR("CuBatch::CuBatch: Not enough memory.");

//     cbdp_.reset(new CuBatchDP());
//     if(!cbdp_)
//         throw MYRUNTIME_ERROR("CuBatch::CuBatch: Not enough memory.");

    MYCUDACHECK(cudaSetDevice(dmem_->GetDeviceProp().devid_));
    MYCUDACHECKLAST;

    MYCUDACHECK(cudaStreamCreate(&streamcopyres_));
    MYCUDACHECKLAST;

    cbpfin_.reset(new TdFinalizer(
        streamcopyres_, 
        dmem_->GetDeviceProp(),
        writer
    ));

    if(!cbpfin_)
        throw MYRUNTIME_ERROR("CuBatch::CuBatch: Not enough memory.");

    size_t szalloc = 
        //size allocated for results: allocate max size initially
        GetHeapSectionOffset(CuDeviceMemory::ddsEndOfDP2Alns) -
        GetHeapSectionOffset(CuDeviceMemory::ddsBegOfDP2AlnData);

    HostAllocResults(szalloc);
}

// -------------------------------------------------------------------------
// destructor
//
CuBatch::~CuBatch()
{
    MYMSG("CuBatch::~CuBatch", 4);

    if(cbpfin_) {
        cbpfin_->Notify(TdFinalizer::cubpthreadmsgTerminate);
        cbpfin_.reset();
    }

    MYCUDACHECK(cudaStreamDestroy(streamcopyres_));
    MYCUDACHECKLAST;

    //free memory after the finalizer has finished (deleted)
    HostFreeResults();

    //device can still be in use by Devices
//     MYCUDACHECK(cudaDeviceReset());
//     MYCUDACHECKLAST;
}

// -------------------------------------------------------------------------
// CheckHostResultsSync: verify whether, and perform if needed, 
// (results) transfer synchronization is required
inline
void CuBatch::CheckHostResultsSync()
{
//     size_t szresults = //size allocated for results
//         sz_heapsections_[ddsEndOfDP2Alns] - 
//         sz_heapsections_[ddsBegOfDP2AlnData];

    //synchronize only if the current end of the DP section
    // overlaps with the beginning of the space allocated previously for 
    // results (OR the required size has increased --- moved to finalization)
    if(limit_beg_results_ < GetHeapSectionOffset(CuDeviceMemory::ddsEndOfDPBackTckData)
        /*|| sz_mem_results_ < szresults*/) 
    {
        //make sure no results data transfer is ongoing:
        MYCUDACHECK(cudaStreamSynchronize(streamcopyres_));
        MYCUDACHECKLAST;
    }

    limit_beg_results_ = GetHeapSectionOffset(CuDeviceMemory::ddsBegOfDP2AlnData);
}





// =========================================================================
// -------------------------------------------------------------------------
// FilteroutReferences: Filter out flagged references;
// bdbCdesc, descriptions of reference structures;
// bdbCpmbeg, beginning addresses of structures read from the database; 
// bdbCpmend, terminal addresses of structures read from the database;
//
void CuBatch::FilteroutReferences(
    cudaStream_t streamproc,
    const char** bdbCdesc,
    char** bdbCpmbeg,
    char** bdbCpmend,
    const size_t /* nqyposs */,
    const size_t nqystrs,
    const size_t /* ndbCposs */,
    const size_t ndbCstrs,
    const size_t /* dbstr1len */,
    const size_t /* dbstrnlen */,
    const size_t /* dbxpad */,
    const size_t maxnsteps,
    size_t& ndbCposs2,
    size_t& ndbCstrs2,
    size_t& dbstr1len2,
    size_t& dbstrnlen2,
    size_t& dbxpad2,
    float* tmpdpdiagbuffers,
    float* tfmmemory,
    float* auxwrkmemory,
    unsigned int* globvarsbuf)
{
    MYMSG("CuBatch::FilteroutReferences", 4);
    static const std::string preamb = "CuBatch::FilteroutReferences: ";
    static const int msglev = (clustwriter_)? 2: 1;
    char msgbuf[KBYTE];

    //--- FILTER ACTIONS -----------------------------------------------
    //execution configuration for filtering reference structures and making a
    //list of progressing further entries (one block):
    dim3 nthrds_filter(CUFL_MAKECANDIDATELIST_XDIM,CUFL_MAKECANDIDATELIST_YDIM,1);
    dim3 nblcks_filter(1, 1, 1);

    //NOTE: similarity prescreening works across all queries in the chunk!
    //NOTE: that means that having a diff #queries in the chunk may result
    //NOTE: in slightly different results because a greater #queries implies
    //NOTE: a greater #selected references, which on average increases
    //NOTE: the probability for some of them to score above the threshold!
    MakeDbCandidateList<<<nblcks_filter,nthrds_filter,0,streamproc>>>(
        nqystrs, ndbCstrs, maxnsteps,  auxwrkmemory, globvarsbuf);
    MYCUDACHECKLAST;

    const size_t szfilterdata = ndbCstrs * nTFilterData * sizeof(int);

    MYCUDACHECK(cudaStreamSynchronize(streamproc));
    MYCUDACHECKLAST;

    // //make sure results have been copied before overwriting them:
    // MYCUDACHECK(cudaStreamSynchronize(streamcopyres_));
    // MYCUDACHECKLAST;

    MYCUDACHECK(cudaMemcpy(
        filterdata_.get(),
        GetHeapSectionAddress(CuDeviceMemory::ddsBegOfGlobVars),
        szfilterdata,
        cudaMemcpyDeviceToHost));
    MYCUDACHECKLAST;

    //NOTE: filter structures on the host side!
    PMBatchStrData::FilterStructs(
        bdbCdesc, bdbCpmbeg, bdbCpmend, filterdata_.get());

    //new statistics for the structures passed to the next round of analysis
    ndbCposs2 = PMBatchStrData::GetNoPosits(bdbCpmbeg, bdbCpmend);
    ndbCstrs2 = PMBatchStrData::GetNoStructs(bdbCpmbeg, bdbCpmend);
    //NOTE: 1st structure is still the largest after filtering
    if(ndbCstrs2) {
        dbstr1len2 = PMBatchStrData::GetLengthAt(bdbCpmbeg, 0);
        dbstrnlen2 = PMBatchStrData::GetLengthAt(bdbCpmbeg, ndbCstrs2-1);
    }

    //new padding along the x axis
    dbxpad2 = ALIGN_UP(ndbCposs2, CUL2CLINESIZE) - ndbCposs2;

    if(GetCurrentMaxDbPos() < ndbCposs2 + dbxpad2) {
        sprintf(msgbuf, "Invalid number of positions of selected db structures: "
            "%zu < %zu+%zu.", GetCurrentMaxDbPos(), ndbCposs2, dbxpad2);
        throw MYRUNTIME_ERROR(preamb + msgbuf);
    }

    MYMSGBEGl(msglev)
        sprintf(msgbuf,"%6c%zu structure(s) proceed(s) (%zu pos.)",' ',ndbCstrs2,ndbCposs2);
        MYMSG(msgbuf,1);
    MYMSGENDl
    MYMSGBEGl(msglev+1)
        sprintf(msgbuf,"%12c (max_length= %zu)",' ',dbstr1len2);
        MYMSG(msgbuf,2);
    MYMSGENDl

    //execution configuration for copying selected reference structures
    //from the original locations:
    dim3 nthrds_fdcopyfrom(CUFL_STORECANDIDATEDATA_XDIM,1,1);
    dim3 nblcks_fdcopyfrom(
        (dbstr1len2 + CUFL_STORECANDIDATEDATA_XDIM - 1)/CUFL_STORECANDIDATEDATA_XDIM,
        ndbCstrs, 1);
    //execution configuration for copying selected reference structures
    //from the original locations:
    dim3 nthrds_fdcopyback(CUFL_LOADCANDIDATEDATA_XDIM,1,1);
    dim3 nblcks_fdcopyback(
        (dbstr1len2 + CUFL_LOADCANDIDATEDATA_XDIM - 1)/CUFL_LOADCANDIDATEDATA_XDIM,
        ndbCstrs2/*!!*/, 1);

    if(ndbCstrs2 && ndbCposs2) {
        ReformatStructureDataPartStore<<<nblcks_fdcopyfrom,nthrds_fdcopyfrom,0,streamproc>>>(
            nqystrs, ndbCstrs, GetCurrentMaxDbPos(),  maxnsteps,
            globvarsbuf, auxwrkmemory, tfmmemory,  tmpdpdiagbuffers);
        MYCUDACHECKLAST;
        ReformatStructureDataPartLoad<<<nblcks_fdcopyback,nthrds_fdcopyback,0,streamproc>>>(
            nqystrs, GetCurrentMaxDbPos(),  maxnsteps, ndbCstrs2,
            tmpdpdiagbuffers,  auxwrkmemory, tfmmemory);
        MYCUDACHECKLAST;
    }
    //--- END OF FILTER ACTIONS ----------------------------------------
}

// =========================================================================
// -------------------------------------------------------------------------
// ProcessBlock: implement structure alignment algorithm for part of query 
// and reference database structures on device;
// scorethld, TM-score threshold;
// querydesc, descriptions of query structures;
// qrysernrbeg, serial number of the beginning query in the chunk;
// querypmbeg, relative beginning addresses of queries;
// querypmend, relative terminal addresses of queries;
// bdbCdesc, descriptions of reference structures;
// bdbCpmbeg, beginning addresses of structures read from the database; 
// bdbCpmend, terminal addresses of structures read from the database;
//
void CuBatch::ProcessBlock(
    float scorethld,
    int qrysernrbeg,
    const char** querydesc,
    char** querypmbeg,
    char** querypmend,
    const char** bdbCdesc,
    char** bdbCpmbeg,
    char** bdbCpmend,
    TSCounterVar* qrscnt,
    TSCounterVar* cnt)
{
    MYMSG("CuBatch::ProcessBlock", 4);
    static const std::string preamb = "CuBatch::ProcessBlock: ";
    static const int clstonesided = CLOptions::GetB_CLS_ONE_SIDED_COVERAGE();
    static const float clstcoverage = CLOptions::GetB_CLS_COVERAGE();
    static const float seqsimthrscore = CLOptions::GetP_PRE_SIMILARITY();
    static const float prescore = CLOptions::GetP_PRE_SCORE();
    const int depth = CLOptions::GetC_DEPTH();
    const int maxnqrsperchunk = CLOptions::GetDEV_QRS_PER_CHUNK();
    const int addsearchbyss = CLOptions::GetC_ADDSEARCHBYSS();
    const int nodetailedsearch = CLOptions::GetC_NODETAILEDSEARCH();
    const int maxndpiters = CLOptions::GetC_CONVERGENCE();//#convergence tests
    const float d2equiv = SQRD(EQUIVALENCE_DISTANCE);
    myruntime_error mre;
    char msgbuf[KBYTE];

    const int stepinit = (depth == CLOptions::csdShallow)? 5: 1;

    if(sizeof(int) != sizeof(float) || sizeof(uint) != sizeof(float))
        throw MYRUNTIME_ERROR(
            preamb + "Calculation impossible: sizeof(int) != sizeof(float).");

// #ifdef __DEBUG__
    if(!querypmbeg || !querypmend)
        throw MYRUNTIME_ERROR( preamb + "Null query addresses.");
    if(!bdbCpmbeg || !bdbCpmend)
        throw MYRUNTIME_ERROR( preamb + "Null addresses of reference structures.");
// #endif

    const size_t nqyposs = PMBatchStrData::GetNoPosits(querypmbeg, querypmend);
    const size_t nqystrs = PMBatchStrData::GetNoStructs(querypmbeg, querypmend);
    size_t ndbCposs = PMBatchStrData::GetNoPosits(bdbCpmbeg, bdbCpmend);
    //total number of reference structures in the chunk:
    size_t ndbCstrs = PMBatchStrData::GetNoStructs(bdbCpmbeg, bdbCpmend);

    if(nqystrs < 1 || ndbCstrs < 1)
        throw MYRUNTIME_ERROR(preamb + "No query and/or reference structures in the chunk.");

    //NOTE: structures are sorted by length: 1st is the largest
    size_t qystr1len = PMBatchStrData::GetLengthAt(querypmbeg, 0);//length of the largest query structure
    size_t dbstr1len = PMBatchStrData::GetLengthAt(bdbCpmbeg, 0);//length of the largest reference structure
    //NOTE: last is the smallest
    size_t qystrnlen = PMBatchStrData::GetLengthAt(querypmbeg, nqystrs-1);//smallest length among queries
    size_t dbstrnlen = PMBatchStrData::GetLengthAt(bdbCpmbeg, ndbCstrs-1);//smallest length among references

    size_t ndbxposs = ndbCposs;

    if((size_t)maxnqrsperchunk < nqystrs)
        throw MYRUNTIME_ERROR(preamb + "Too many query structures in the chunk.");


    //padding for queries (along the y axis; when processed independently)
    // const size_t qsypad = ALIGN_UP(nqyposs, CUL2CLINESIZE) - nqyposs;
    //padding along the x axis
    size_t dbxpad = ALIGN_UP(ndbxposs, CUL2CLINESIZE) - ndbxposs;
    SetCurrentDbxPadding((unsigned int)dbxpad);

    size_t szsspace = nqyposs * ndbxposs;//search space
    if(nqystrs < 1)
        throw MYRUNTIME_ERROR(preamb + "Invalid number of query structures in the chunk.");
    if(ndbCstrs < 1)
        throw MYRUNTIME_ERROR(preamb + "Invalid number of db structures in the chunk.");
    if(szsspace < 1)
        throw MYRUNTIME_ERROR(preamb + "Invalid search space size.");
    if(GetCurrentMaxDbPos() < ndbCposs + dbxpad ) {
        sprintf(msgbuf, "Invalid number of positions of db structures in the chunk: "
            "%zu < %zu+%zu.", GetCurrentMaxDbPos(), ndbCposs, dbxpad);
        throw MYRUNTIME_ERROR(preamb + msgbuf);
    }

    cudaStream_t streamproc;
    MYCUDACHECK(cudaStreamCreate(&streamproc));

    std::map<CGKey,MyCuGraph> stgraphs;

    //max number of steps to calculate superposition for a maximum-length alignment 
    const size_t maxnsteps = CuDeviceMemory::GetMaxNFragSteps();
    const size_t maxalnmax = CuDeviceMemory::GetMaxAlnLength();
    const size_t minfraglen = GetMinFragLengthForAln(maxalnmax);

    float* scores = (float*)GetHeapSectionAddress(CuDeviceMemory::ddsBegOfMtxScores);
    float* tmpdpdiagbuffers = (float*)GetHeapSectionAddress(CuDeviceMemory::ddsBegOfDPDiagScores);
    float* tmpdpbotbuffer = (float*)GetHeapSectionAddress(CuDeviceMemory::ddsBegOfDPBottomScores);
    float* tmpdpalnpossbuffer = (float*)GetHeapSectionAddress(CuDeviceMemory::ddsBegOfDPAlignedPoss);
    unsigned int* maxcoordsbuf = (unsigned int*)GetHeapSectionAddress(CuDeviceMemory::ddsBegOfDPMaxCoords);
    char* btckdata = (char*)GetHeapSectionAddress(CuDeviceMemory::ddsBegOfDPBackTckData);
    float* wrkmemory = (float*)GetHeapSectionAddress(CuDeviceMemory::ddsBegOfWrkMemory);
    float* wrkmemoryccd = (float*)GetHeapSectionAddress(CuDeviceMemory::ddsBegOfWrkMemoryCCD);
    float* wrkmemorytmalt = (float*)GetHeapSectionAddress(CuDeviceMemory::ddsBegOfWrkMemoryTMalt);
    float* wrkmemorytm = (float*)GetHeapSectionAddress(CuDeviceMemory::ddsBegOfWrkMemoryTM);
    float* wrkmemorytmibest = (float*)GetHeapSectionAddress(CuDeviceMemory::ddsBegOfWrkMemoryTMibest);
    float* auxwrkmemory = (float*)GetHeapSectionAddress(CuDeviceMemory::ddsBegOfAuxWrkMemory);
    float* wrkmemory2 = (float*)GetHeapSectionAddress(CuDeviceMemory::ddsBegOfWrkMemory2);
    float* alndatamemory = (float*)GetHeapSectionAddress(CuDeviceMemory::ddsBegOfDP2AlnData);
    float* tfmmemory = (float*)GetHeapSectionAddress(CuDeviceMemory::ddsBegOfTfmMatrices);
    char* alnsmemory = (char*)GetHeapSectionAddress(CuDeviceMemory::ddsBegOfDP2Alns);
    unsigned int* globvarsbuf = (unsigned int*)GetHeapSectionAddress(CuDeviceMemory::ddsBegOfGlobVars);


    try {
        MyMpTimer<> t; t.Start();

        //filtering condition for clustering:
        bool condition4filter0 = 
            //NOTE: && abandoned to check global index of references:
            clustwriter_;
            // clustwriter_ && (clstonesided == 0) && (clstcoverage > 0.001f);

        bool condition4filter1 =
            (0.0f < seqsimthrscore &&
            CUFL_TBSP_SEQUENCE_SIMILARITY_EDGE < qystr1len &&
            CUFL_TBSP_SEQUENCE_SIMILARITY_EDGE < dbstr1len);

        //initialize memory before starting calculations
        stage1::preinitialize1(
            streamproc, (condition4filter1 || condition4filter0),
            maxnsteps, minfraglen, nqystrs, ndbCstrs, nqyposs, ndbCposs,
            wrkmemorytmibest, auxwrkmemory, tfmmemory, alndatamemory
        );

        //NOTE: for (#GPUs>1 with) filtering, make a copy for all (NOTE!) GPUs
        if(/*1 < ringsize_ &&*/ (0.0f < seqsimthrscore || 0.0f < prescore || condition4filter0))
        {
            bdbCstruct_[ndxbatchstrdata_].Copy(bdbCdesc, bdbCpmbeg, bdbCpmend);
            bdbCdesc = (const char**)(bdbCstruct_[ndxbatchstrdata_].bdbCptrdescs_.get());
            bdbCpmbeg = bdbCstruct_[ndxbatchstrdata_].bdbCpmbeg_;
            bdbCpmend = bdbCstruct_[ndxbatchstrdata_].bdbCpmend_;
            if(NBATCHSTRDATA <= ++ndxbatchstrdata_) ndxbatchstrdata_ = 0;
        }


        //--- FILTER0 ------------------------------------------------------
        if(condition4filter0) {
            //execution configuration for calculating maximum coverage:
            dim3 nthrds_covchk(CUFL_TBSP_SEQUENCE_COVERAGE_XDIM,1,1);
            dim3 nblcks_covchk(
                (ndbCstrs + CUFL_TBSP_SEQUENCE_COVERAGE_XDIM - 1)/CUFL_TBSP_SEQUENCE_COVERAGE_XDIM,
                nqystrs, 1);

            //NOTE: filtered out references will be marked; the next kernel 
            //filtering by ss similarity will check references for convergence
            //before taking action;
            CheckMaxCoverage<<<nblcks_covchk,nthrds_covchk,0,streamproc>>>(
                clstcoverage, qrysernrbeg + (int)nqystrs, ndbCstrs, maxnsteps, auxwrkmemory
            );
        }
        //--- EBD FILTER0 --------------------------------------------------


        //--- FILTER -------------------------------------------------------
        size_t ndbCposs1 = ndbCposs;
        size_t ndbCstrs1 = ndbCstrs;
        size_t dbstr1len1 = dbstr1len;
        size_t dbstrnlen1 = dbstrnlen;
        size_t dbxpad1 = dbxpad;

        if(condition4filter1) {
            //execution configuration for calculating local ungapped sequence alignment:
            const uint ndiagonals = 
                (qystr1len + dbstr1len - 2 * CUFL_TBSP_SEQUENCE_SIMILARITY_EDGE) >>
                CUFL_TBSP_SEQUENCE_SIMILARITY_STEPLOG2;
            dim3 nthrds_seqaln(CUFL_TBSP_SEQUENCE_SIMILARITY_XDIM,1,1);
            dim3 nblcks_seqaln(ndbCstrs, ndiagonals, nqystrs);

            VerifyAlignmentScore<<<nblcks_seqaln,nthrds_seqaln,0,streamproc>>>(
                seqsimthrscore, nqystrs, ndbCstrs, maxnsteps, auxwrkmemory
            );
        }

        if(condition4filter0 || condition4filter1)
            FilteroutReferences(
                streamproc, bdbCdesc, bdbCpmbeg, bdbCpmend,
                nqyposs, nqystrs, ndbCposs, ndbCstrs, dbstr1len, dbstrnlen, dbxpad,
                maxnsteps, ndbCposs1, ndbCstrs1, dbstr1len1, dbstrnlen1, dbxpad1,
                tmpdpdiagbuffers, tfmmemory, auxwrkmemory, globvarsbuf);
        //--- EBD FILTER ---------------------------------------------------


        //calculate secondary structures for the structures in the chunk
        if(ndbCstrs1 && ndbCposs1)
            cusecstr::calc_secstr(
                streamproc,
                nqystrs, ndbCstrs1, nqyposs, ndbCposs1,
                qystr1len, dbstr1len1, qystrnlen, dbstrnlen1, dbxpad1
            );

        //run the first stage for finding transformation matrices
        if(ndbCstrs1 && ndbCposs1)
            stage1::run_stage1(
                stgraphs, streamproc, 2/*maxndpiters*/,
                maxnsteps, minfraglen, scorethld, prescore, stepinit,
                nqystrs, ndbCstrs1, nqyposs, ndbCposs1,
                qystr1len, dbstr1len1, qystrnlen, dbstrnlen1, dbxpad1,
                scores, tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, maxcoordsbuf, btckdata,
                wrkmemory, wrkmemoryccd, wrkmemorytm, wrkmemorytmibest,
                auxwrkmemory, wrkmemory2, tfmmemory,
                globvarsbuf
            );


        //--- FILTER -------------------------------------------------------
        size_t ndbCposs2 = ndbCposs1;
        size_t ndbCstrs2 = ndbCstrs1;
        size_t dbstr1len2 = dbstr1len1;
        size_t dbstrnlen2 = dbstrnlen1;
        size_t dbxpad2 = dbxpad1;

        if(0.0f < prescore && ndbCstrs1 && ndbCposs1)
            FilteroutReferences(
                streamproc, bdbCdesc, bdbCpmbeg, bdbCpmend,
                nqyposs, nqystrs, ndbCposs1, ndbCstrs1, dbstr1len1, dbstrnlen1, dbxpad1,
                maxnsteps, ndbCposs2, ndbCstrs2, dbstr1len2, dbstrnlen2, dbxpad2,
                tmpdpdiagbuffers, tfmmemory, auxwrkmemory, globvarsbuf);
        //--- EBD FILTER ---------------------------------------------------


        //reinitialize passedstatscntrd_
        for(size_t i = 0; i < nqystrs; i++) {
            size_t loff = nDevGlobVariables * i;
            passedstatscntrd_[loff + dgvNPassedStrs] = ndbCstrs2;
            passedstatscntrd_[loff + dgvNPosits] = ndbCposs2;
        }


        //execution configuration for scores initialization:
        //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
        dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
        dim3 nblcks_scinit(
            (ndbCstrs2 + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
            nqystrs, maxnsteps);

        //reset best scores and convergence flags;
        if(addsearchbyss && ndbCstrs2 && ndbCposs2) {
            //convergence is checked during refinement;
            InitScores<INITOPT_BEST|INITOPT_CONVFLAG_ALL>
                <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
                    ndbCstrs2,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,
                    auxwrkmemory);
            MYCUDACHECKLAST;

            //run superposition search by a combination of secondary structure and
            //sequence similarity
            stage_ssrr::run_stage_ssrr<true/*USESEQSCORING*/>(
                stgraphs, streamproc, scorethld, prescore, 2/*maxndpiters*/,
                maxnsteps, minfraglen, nqystrs, ndbCstrs2, nqyposs, ndbCposs2,
                qystr1len, dbstr1len2, qystrnlen, dbstrnlen2, dbxpad2,
                scores, tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, maxcoordsbuf, btckdata,
                wrkmemory, wrkmemoryccd, wrkmemorytm, wrkmemorytmibest,
                auxwrkmemory, wrkmemory2, tfmmemory,
                globvarsbuf
            );
        }

        //reset best scores and convergence flags;
        if(ndbCstrs2 && ndbCposs2)
            InitScores<INITOPT_BEST|INITOPT_CONVFLAG_ALL>
                <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
                    ndbCstrs2,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,
                    auxwrkmemory);
        MYCUDACHECKLAST;


        //run the stage for finding optimal superpositions from spectral similarities
        if(nodetailedsearch == 0 && ndbCstrs2 && ndbCposs2) {
            stagefrg3::run_stagefrg3(
                stgraphs, streamproc, 2/*maxndpiters*/,
                maxnsteps, minfraglen, scorethld, prescore, nqystrs, ndbCstrs2, nqyposs, ndbCposs2,
                qystr1len, dbstr1len2, qystrnlen, dbstrnlen2, dbxpad2,
                scores, tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, maxcoordsbuf, btckdata,
                wrkmemory, wrkmemoryccd, wrkmemorytmalt, wrkmemorytm, wrkmemorytmibest,
                auxwrkmemory, wrkmemory2, tfmmemory,
                globvarsbuf
            );
        }

        //run the stage using the best superposition so far
        if(ndbCstrs2 && ndbCposs2) {
            stage2::run_stage2<false/* GAP0 */, true/* USESS */, D02IND_DPSCAN>(
                stgraphs, streamproc, false/*check_for_low_scores*/, scorethld, prescore, 2/*maxndpiters*/,
                maxnsteps, minfraglen, nqystrs, ndbCstrs2, nqyposs, ndbCposs2,
                qystr1len, dbstr1len2, qystrnlen, dbstrnlen2, dbxpad2,
                scores, tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, maxcoordsbuf, btckdata,
                wrkmemory, wrkmemoryccd, wrkmemorytm, wrkmemorytmibest,
                auxwrkmemory, wrkmemory2, tfmmemory,
                globvarsbuf
            );
        }

        //run DP refinement using the best superposition and final parameters
        if(ndbCstrs2 && ndbCposs2) {
            stage2::run_stage2<true/* GAP0 */, false/* USESS */, D02IND_SEARCH>(
                stgraphs, streamproc, true/*check_for_low_scores*/, scorethld, prescore, maxndpiters,
                maxnsteps, minfraglen, nqystrs, ndbCstrs2, nqyposs, ndbCposs2,
                qystr1len, dbstr1len2, qystrnlen, dbstrnlen2, dbxpad2,
                scores, tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, maxcoordsbuf, btckdata,
                wrkmemory, wrkmemoryccd, wrkmemorytm, wrkmemorytmibest,
                auxwrkmemory, wrkmemory2, tfmmemory,
                globvarsbuf
            );
        }

        //run the final stage for producing optimal alignments
        if(ndbCstrs2 && ndbCposs2) {
            stagefin::run_stagefin(
                streamproc, d2equiv, scorethld,
                maxnsteps, minfraglen, nqystrs, ndbCstrs2, nqyposs, ndbCposs2,
                qystr1len, dbstr1len2, qystrnlen, dbstrnlen2, dbxpad2,
                scores, tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, maxcoordsbuf, btckdata,
                wrkmemory, wrkmemoryccd, wrkmemorytm, wrkmemorytmibest,
                auxwrkmemory, wrkmemory2, alndatamemory, tfmmemory, alnsmemory,
                globvarsbuf
            );
        }

        // //run the stage for finding optimal superpositions from spectral similarities
        // if(ndbCstrs2 && ndbCposs2) {
        //     specsim/*32*/::run_stage_on_specsim/*32*/(
        //         stgraphs, streamproc,
        //         maxnsteps, minfraglen, scorethld, nqystrs, ndbCstrs2, nqyposs, ndbCposs2,
        //         qystr1len, dbstr1len2, qystrnlen, dbstrnlen2, qsypad, dbxpad2,
        //         scores, tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, maxcoordsbuf, btckdata,
        //         wrkmemory, wrkmemoryccd, wrkmemorytm, wrkmemorytmibest,
        //         auxwrkmemory, wrkmemory2, tfmmtxmemory,
        //         globvarsbuf
        //     );
        // }

        //total size for alignments, including padding
        size_t szaligns2 = dmem_->GetSizeOfDP2Alns(nqystrs, nqyposs, ndbCposs2, ndbCstrs2);

        MYCUDACHECK(cudaStreamSynchronize(streamproc));
        MYCUDACHECKLAST;

        t.Stop();

        if(clustwriter_)
            TransferResultsForClustering(
                qrysernrbeg,  nqyposs, nqystrs,
                querydesc, querypmbeg, querypmend,
                bdbCdesc, bdbCpmbeg, bdbCpmend,
                qrscnt, cnt,  passedstatscntrd_.get());
        else
            TransferResultsFromDevice(
                t.GetElapsedTime(),
                scorethld,
                qrysernrbeg,
                nqyposs, nqystrs,
                //NOTE: original statistics for: ndbCposs, ndbCstrs!
                ndbCposs, ndbCstrs,
                querydesc, querypmbeg, querypmend,
                bdbCdesc, bdbCpmbeg, bdbCpmend,
                qrscnt, cnt,  szaligns2,  passedstatscntrd_.get());

    } catch(myruntime_error const& ex) {
        mre = ex;
    } catch(myexception const& ex) {
        mre = ex;
    } catch(...) {
        mre = MYRUNTIME_ERROR("Exception caught.");
    }

    MYCUDACHECK(cudaStreamDestroy(streamproc));
    MYCUDACHECKLAST;

    if( mre.isset())
        throw mre;
}





// =========================================================================
// TransferResultsFromDevice: transfer calculated results from device and 
// provide Finalizer with them;
// tdrtn, time duration of computations;
// scorethld, TM-score threshold;
// qrysernrbeg, serial number of the first query in the chunk;
// nqyposs, total length of queries in the chunk;
// nqystrs, number of queries in the chunk;
// ndbCposs, total length of reference structures in the chunk;
// ndbCstrs, number of reference structures in the chunk;
// querydesc, query descriptions;
// querypmbeg, beginning addresses of queries;
// querypmend, terminal addresses of queries;
// bdbCdesc, descriptions of reference structures;
// bdbCpmbeg, beginning addresses of reference structures in the chunk; 
// bdbCpmend, terminal addresses of reference structures in the chunk;
// qrscnt, counter associated with how many agents access query data;
// cnt, counter associated with how many agents access (iterations 
// performed on) the chunk data;
// szaligns2, total alignment size over all query-reference pairs; if 0,
// none of the queries in the chunk have significant matches; 
// passedstats, data structure of max total number of passed db reference 
// structure positions (maxnposits) for each query in the chunk, and
// max number of passed db reference structures (maxnstrs) for each 
// query in the chunk;
//
void CuBatch::TransferResultsFromDevice(
    double tdrtn,
    float scorethld,
    int qrysernrbeg,
    size_t nqyposs, size_t nqystrs,
    size_t ndbCposs, size_t ndbCstrs,
    const char** querydesc,
    char** querypmbeg,
    char** querypmend,
    const char** bdbCdesc,
    char** bdbCpmbeg,
    char** bdbCpmend,
    TSCounterVar* qrscnt,
    TSCounterVar* cnt,
    size_t szaligns2,
    unsigned int* passedstats)
{
    MYMSG("CuBatch::TransferResultsFromDevice", 3);
    static const std::string preamb = "CuBatch::TransferResultsFromDevice: ";

    size_t szresults = 
        //size allocated for results: transform. matrices + aln. data:
        GetHeapSectionOffset(CuDeviceMemory::ddsEndOfTfmMatrices) -
        GetHeapSectionOffset(CuDeviceMemory::ddsBegOfDP2AlnData) +
        //plus actual alignment data size, as opposed to allocated
        szaligns2;

    //make sure of no pending transfer in this stream
    MYCUDACHECK(cudaStreamSynchronize(streamcopyres_));
    MYCUDACHECKLAST;

#ifdef __DEBUG__
//     if( !h_results_)
//         throw MYRUNTIME_ERROR( preamb + "Null results address.");
    if(!(cbpfin_))
        throw MYRUNTIME_ERROR(preamb + "Null finalizer object.");
#endif

    CheckFinalizer();

    //allocate memory if the required size has increased
    if(sz_mem_results_ < szresults) {
        char errbuf[BUF_MAX];
        sprintf(errbuf, "Size of results exceeds allocation: %zu < %zu",
            sz_mem_results_, szresults);
        throw MYRUNTIME_ERROR(preamb + errbuf);
    }

    if(szaligns2)
    {   //lock this section as the data is about to change
        std::lock_guard<std::mutex> lck(cbpfin_->GetPrivateMutex());

        if(lockedresmem_) {
            MYCUDACHECK(cudaMemcpyAsync(
                h_results_,
                GetHeapSectionAddress(CuDeviceMemory::ddsBegOfDP2AlnData),
                szresults,
                cudaMemcpyDeviceToHost,
                streamcopyres_));
            MYCUDACHECKLAST;
        }
        else {
            MYCUDACHECK(cudaMemcpy(
                h_results_,
                GetHeapSectionAddress(CuDeviceMemory::ddsBegOfDP2AlnData),
                szresults,
                cudaMemcpyDeviceToHost));
            MYCUDACHECKLAST;
        }
    }

    cbpfin_->SetCuBPBDbdata(
        tdrtn,
        scorethld,
        qrysernrbeg,
        GetDeviceName(),
        nqyposs, nqystrs,
        ndbCposs, ndbCstrs,
        querydesc, querypmbeg, querypmend, 
        bdbCdesc, bdbCpmbeg, bdbCpmend,
        qrscnt, cnt,
        passedstats,
        h_results_,
        GetHeapSectionOffset(CuDeviceMemory::ddsEndOfDP2AlnData)-
            GetHeapSectionOffset(CuDeviceMemory::ddsBegOfDP2AlnData),
        GetHeapSectionOffset(CuDeviceMemory::ddsEndOfTfmMatrices)-
            GetHeapSectionOffset(CuDeviceMemory::ddsBegOfTfmMatrices),
        szaligns2,
        szaligns2
    );

    cbpfin_->Notify(TdFinalizer::cubpthreadmsgFinalize);
}

// =========================================================================
// TransferResultsForClustering: transfer calculated results for clustering;
// qrysernrbeg, serial number of the first query in the chunk;
// nqyposs, total length of queries in the chunk;
// nqystrs, number of queries in the chunk;
// querydesc, query descriptions;
// querypmbeg, beginning addresses of queries;
// querypmend, terminal addresses of queries;
// bdbCdesc, descriptions of reference structures;
// bdbCpmbeg, beginning addresses of reference structures in the chunk; 
// bdbCpmend, terminal addresses of reference structures in the chunk;
// qrscnt, counter associated with how many agents access query data;
// cnt, counter associated with how many agents access (iterations 
// performed on) the chunk data;
// passedstats, data structure of max total number of passed db reference 
// structure positions (maxnposits) for each query in the chunk, and
// max number of passed db reference structures (maxnstrs) for each 
// query in the chunk;
//
void CuBatch::TransferResultsForClustering(
    int qrysernrbeg,
    size_t nqyposs, size_t nqystrs,
    const char** querydesc,
    char** querypmbeg,
    char** querypmend,
    const char** bdbCdesc,
    char** bdbCpmbeg,
    char** bdbCpmend,
    TSCounterVar* qrscnt,
    TSCounterVar* cnt,
    unsigned int* passedstats)
{
    MYMSG("CuBatch::TransferResultsForClustering", 3);
    static const std::string preamb = "CuBatch::TransferResultsForClustering: ";

    if(!clustwriter_)
        throw MYRUNTIME_ERROR(preamb + "Null clustering object.");

    size_t szresults = 
        //size allocated for results: aln. data:
        GetHeapSectionOffset(CuDeviceMemory::ddsEndOfDP2AlnData) -
        GetHeapSectionOffset(CuDeviceMemory::ddsBegOfDP2AlnData);

    //make sure of no pending transfer in this stream
    MYCUDACHECK(cudaStreamSynchronize(streamcopyres_));
    MYCUDACHECKLAST;

    if(clustwriter_->GetResponse() == CLSTWRTTHREAD_MSG_ERROR)
        throw MYRUNTIME_ERROR(preamb + "Clusterer terminated with errors.");

    {
        //NOTE: data processed instantly: copy synchronously
        // if(lockedresmem_)
        //     MYCUDACHECK(cudaMemcpyAsync(
        //         h_results_,
        //         GetHeapSectionAddress(CuDeviceMemory::ddsBegOfDP2AlnData),
        //         szresults,
        //         cudaMemcpyDeviceToHost,
        //         streamcopyres_));
        // else
            MYCUDACHECK(cudaMemcpy(
                h_results_,
                GetHeapSectionAddress(CuDeviceMemory::ddsBegOfDP2AlnData),
                szresults,
                cudaMemcpyDeviceToHost));
        MYCUDACHECKLAST;
    }

    clustwriter_->PushPartOfResults( 
        qrysernrbeg,
        nqyposs, nqystrs,
        querydesc, querypmbeg, querypmend, 
        bdbCdesc, bdbCpmbeg, bdbCpmend,
        qrscnt, cnt,
        passedstats,
        h_results_,
        szresults
    );
}
