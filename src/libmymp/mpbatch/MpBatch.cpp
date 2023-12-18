/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/mybase.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "libutil/mptimer.h"
#include "libutil/CLOptions.h"
#include "tsafety/TSCounterVar.h"
#include "libgenp/gdats/PMBatchStrData.h"
#include "libgenp/goutp/TdFinalizer.h"
#include "libgenp/goutp/TdAlnWriter.h"
#include "libmymp/mplayout/MpGlobalMemory.h"

#include "libmycu/cucom/cudef.h"
#include "libmycu/custages/fragment.cuh"
#include "libmymp/mpss/MpSecStr.h"
#include "libmymp/mpdp/mpdpbase.h"
#include "libmymp/mpfilter/MpReform.h"
#include "libmymp/mpstage1/MpStage1.h"
#include "libmymp/mpstage1/MpStage2.h"
#include "libmymp/mpstage1/MpStageSSRR.h"
#include "libmymp/mpstage1/MpStageFrg3.h"
#include "libmymp/mpstage1/MpStageFin.h"
#include "MpBatch.h"

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// constructor
//
MpBatch::MpBatch(
    MpGlobalMemory* gmem, int dareano, TdAlnWriter* writer)
:
    gmem_(gmem),
    devareano_(dareano),
    filterdata_(nullptr),
    passedstatscntrd_(nullptr),
    cbpfin_(nullptr)
{
    MYMSG("MpBatch::MpBatch", 4);

    if(gmem_ == NULL)
        throw MYRUNTIME_ERROR("MpBatch::MpBatch: Null global memory object.");

    if(devareano_ < 0 || gmem_->GetNAreas() <= devareano_)
        throw MYRUNTIME_ERROR("MpBatch::MpBatch: Invalid global memory area number.");

    if(gmem_->GetMemAlignment() < CUL2CLINESIZE * sizeof(float)) {
        char errbuf[BUF_MAX];
        sprintf(errbuf,
            "Memory alignment size (%zu) < defined CUL2CLINESIZE (%d) * %zu. "
            "Decrease CUL2CLINESIZE to a multiple of 32 and recompile.",
            gmem_->GetMemAlignment(), CUL2CLINESIZE, sizeof(float));
        throw MYRUNTIME_ERROR(errbuf);
    }

    const int maxnqrsperchunk = CLOptions::GetDEV_QRS_PER_CHUNK();
    const size_t szstats = maxnqrsperchunk * nDevGlobVariables;
    const size_t nfilterdata = GetCurrentMaxNDbStrs() * nTFilterData;

    filterdata_.reset(new unsigned int[nfilterdata]);
    passedstatscntrd_.reset(new unsigned int[szstats]);

    if(!filterdata_ || !passedstatscntrd_)
        throw MYRUNTIME_ERROR("MpBatch::MpBatch: Not enough memory.");

    cbpfin_.reset(new TdFinalizer(writer));

    if(!cbpfin_)
        throw MYRUNTIME_ERROR("MpBatch::MpBatch: Not enough memory.");
}

// -------------------------------------------------------------------------
// destructor
//
MpBatch::~MpBatch()
{
    MYMSG("MpBatch::~MpBatch", 4);

    if(cbpfin_) {
        cbpfin_->Notify(TdFinalizer::cubpthreadmsgTerminate);
        cbpfin_.reset();
    }
}





// =========================================================================
// -------------------------------------------------------------------------
// FilteroutReferences: Filter out flagged references;
// bdbCdesc, descriptions of reference structures;
// bdbCpmbeg, beginning addresses of structures read from the database; 
// bdbCpmend, terminal addresses of structures read from the database;
//
void MpBatch::FilteroutReferences(
    char** queryndxpmbeg,
    char** queryndxpmend,
    char** querypmbeg,
    char** querypmend,
    const char** bdbCdesc,
    char** bdbCpmbeg,
    char** bdbCpmend,
    char** bdbCndxpmbeg,
    char** bdbCndxpmend,
    const size_t nqyposs,
    const size_t nqystrs,
    const size_t ndbCposs,
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
    MYMSG("MpBatch::FilteroutReferences", 4);
    static const std::string preamb = "MpBatch::FilteroutReferences: ";
    char msgbuf[KBYTE];

    //--- FILTER ACTIONS -----------------------------------------------
    //NOTE: similarity prescreening works across all queries in the chunk!
    //NOTE: that means that having a diff #queries in the chunk may result
    //NOTE: in slightly different results because a greater #queries implies
    //NOTE: a greater #selected references, which on average increases
    //NOTE: the probability for some of them to score above the threshold!
    MpReform refm(
        maxnsteps,
        querypmbeg, querypmend, bdbCpmbeg, bdbCpmend,
        queryndxpmbeg, queryndxpmend, bdbCndxpmbeg, bdbCndxpmend,
        nqystrs, ndbCstrs, nqyposs, ndbCposs,
        tmpdpdiagbuffers,
        auxwrkmemory, tfmmemory, globvarsbuf, filterdata_.get());

    refm.MakeDbCandidateList();

    // const size_t szfilterdata = ndbCstrs * nTFilterData * sizeof(int);

    //NOTE: filter structures on the host side!
    //NOTE: indices must be filtered first: PMBatchStrData used there!
    PMBatchStrDataIndex::FilterStructs(
        bdbCndxpmbeg, bdbCndxpmend,
        bdbCpmbeg, bdbCpmend, filterdata_.get());
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

    MYMSGBEGl(1)
        sprintf(msgbuf,"%6c%zu structure(s) proceed(s) (%zu pos.)",' ',ndbCstrs2,ndbCposs2);
        MYMSG(msgbuf,1);
    MYMSGENDl
    MYMSGBEGl(2)
        sprintf(msgbuf,"%12c (max_length= %zu)",' ',dbstr1len2);
        MYMSG(msgbuf,2);
    MYMSGENDl

    if(ndbCstrs2 && ndbCposs2)
        refm.SelectAndReformat(ndbCstrs2, GetCurrentMaxDbPos());

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
void MpBatch::ProcessBlock(
    int qrysernrbeg,
    char** queryndxpmbeg,
    char** queryndxpmend,
    const char** querydesc,
    char** querypmbeg,
    char** querypmend,
    const char** bdbCdesc,
    char** bdbCpmbeg,
    char** bdbCpmend,
    char** bdbCndxpmbeg,
    char** bdbCndxpmend,
    TSCounterVar* qrscnt,
    TSCounterVar* cnt)
{
    MYMSG("MpBatch::ProcessBlock", 4);
    static const std::string preamb = "MpBatch::ProcessBlock: ";
    static const float scorethld = CLOptions::GetO_S();
    static const float seqsimthrscore = CLOptions::GetP_PRE_SIMILARITY();
    const float prescore = CLOptions::GetP_PRE_SCORE();
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
    if(!queryndxpmbeg || !queryndxpmend || !querypmbeg || !querypmend)
        throw MYRUNTIME_ERROR(preamb + "Null query addresses.");
    if(!bdbCndxpmbeg || !bdbCndxpmend || !bdbCpmbeg || !bdbCpmend)
        throw MYRUNTIME_ERROR(preamb + "Null addresses of reference structures.");
// #endif

    //NOTE: wait for the finalizer to finish processing the previous data;
    //NOTE: heap section data will change!
    cbpfin_->waitForDataAccess();

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
    const size_t dbxpad = ALIGN_UP(ndbxposs, CUL2CLINESIZE) - ndbxposs;

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


    //max number of steps to calculate superposition for a maximum-length alignment 
    const size_t maxnsteps = MpGlobalMemory::GetMaxNFragSteps();
    const size_t maxalnmax = MpGlobalMemory::GetMaxAlnLength();
    const size_t minfraglen = GetMinFragLengthForAln(maxalnmax);

    float* scores = (float*)GetHeapSectionAddress(MpGlobalMemory::ddsBegOfMtxScores);
    float* tmpdpdiagbuffers = (float*)GetHeapSectionAddress(MpGlobalMemory::ddsBegOfDPDiagScores);
    float* tmpdpbotbuffer = (float*)GetHeapSectionAddress(MpGlobalMemory::ddsBegOfDPBottomScores);
    float* tmpdpalnpossbuffer = (float*)GetHeapSectionAddress(MpGlobalMemory::ddsBegOfDPAlignedPoss);
    unsigned int* maxcoordsbuf = (unsigned int*)GetHeapSectionAddress(MpGlobalMemory::ddsBegOfDPMaxCoords);
    char* btckdata = (char*)GetHeapSectionAddress(MpGlobalMemory::ddsBegOfDPBackTckData);
    float* wrkmemory = (float*)GetHeapSectionAddress(MpGlobalMemory::ddsBegOfWrkMemory);
    float* wrkmemoryccd = (float*)GetHeapSectionAddress(MpGlobalMemory::ddsBegOfWrkMemoryCCD);
    float* wrkmemorytmalt = (float*)GetHeapSectionAddress(MpGlobalMemory::ddsBegOfWrkMemoryTMalt);
    float* wrkmemorytm = (float*)GetHeapSectionAddress(MpGlobalMemory::ddsBegOfWrkMemoryTM);
    float* wrkmemorytmibest = (float*)GetHeapSectionAddress(MpGlobalMemory::ddsBegOfWrkMemoryTMibest);
    float* auxwrkmemory = (float*)GetHeapSectionAddress(MpGlobalMemory::ddsBegOfAuxWrkMemory);
    float* wrkmemory2 = (float*)GetHeapSectionAddress(MpGlobalMemory::ddsBegOfWrkMemory2);
    float* alndatamemory = (float*)GetHeapSectionAddress(MpGlobalMemory::ddsBegOfDP2AlnData);
    float* tfmmemory = (float*)GetHeapSectionAddress(MpGlobalMemory::ddsBegOfTfmMatrices);
    char* alnsmemory = (char*)GetHeapSectionAddress(MpGlobalMemory::ddsBegOfDP2Alns);
    unsigned int* globvarsbuf = (unsigned int*)GetHeapSectionAddress(MpGlobalMemory::ddsBegOfGlobVars);


    try {
        MyMpTimer<> t; t.Start();

        bool condition4filter1 =
            (0.0f < seqsimthrscore &&
            MPFL_TBSP_SEQUENCE_SIMILARITY_EDGE < qystr1len &&
            MPFL_TBSP_SEQUENCE_SIMILARITY_EDGE < dbstr1len);

        //first stage object:
        MpStage1 stg1(
            2/*maxndpiters*/,
            maxnsteps, minfraglen, prescore, stepinit,
            querypmbeg, querypmend, bdbCpmbeg, bdbCpmend,
            nqystrs, ndbCstrs, nqyposs, ndbCposs,
            qystr1len, dbstr1len, qystrnlen, dbstrnlen, dbxpad,
            scores, tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, maxcoordsbuf, btckdata,
            wrkmemory, wrkmemoryccd, wrkmemorytm, wrkmemorytmibest,
            auxwrkmemory, wrkmemory2, alndatamemory, tfmmemory,
            globvarsbuf
        );

        //initialize memory before computations
        stg1.Preinitialize1(condition4filter1);


        //--- FILTER -------------------------------------------------------
        size_t ndbCposs1 = ndbCposs;
        size_t ndbCstrs1 = ndbCstrs;
        size_t dbstr1len1 = dbstr1len;
        size_t dbstrnlen1 = dbstrnlen;
        size_t dbxpad1 = dbxpad;

        if(condition4filter1) {
            stg1.VerifyAlignmentScore(seqsimthrscore);

            FilteroutReferences(
                queryndxpmbeg, queryndxpmend, querypmbeg, querypmend,
                bdbCdesc, bdbCpmbeg, bdbCpmend, bdbCndxpmbeg, bdbCndxpmend,
                nqyposs, nqystrs, ndbCposs, ndbCstrs, dbstr1len, dbstrnlen, dbxpad,
                maxnsteps, ndbCposs1, ndbCstrs1, dbstr1len1, dbstrnlen1, dbxpad1,
                tmpdpdiagbuffers, tfmmemory, auxwrkmemory, globvarsbuf);
        }
        //--- EBD FILTER ---------------------------------------------------


        //calculate secondary structures for the structures in the chunk
        if(ndbCstrs1 && ndbCposs1)
            MpSecStr(
                querypmbeg, querypmend, bdbCpmbeg, bdbCpmend,
                nqystrs, ndbCstrs1, nqyposs, ndbCposs1,
                qystr1len, dbstr1len1, qystrnlen, dbstrnlen1, dbxpad1
            ).Run();

        //run the first stage for finding transformation matrices
        if(ndbCstrs1 && ndbCposs1)
            MpStage1(
                2/*maxndpiters*/,
                maxnsteps, minfraglen, prescore, stepinit,
                querypmbeg, querypmend, bdbCpmbeg, bdbCpmend,
                nqystrs, ndbCstrs1, nqyposs, ndbCposs1,
                qystr1len, dbstr1len1, qystrnlen, dbstrnlen1, dbxpad1,
                scores, tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, maxcoordsbuf, btckdata,
                wrkmemory, wrkmemoryccd, wrkmemorytm, wrkmemorytmibest,
                auxwrkmemory, wrkmemory2, alndatamemory, tfmmemory,
                globvarsbuf
            ).Run();


        //--- FILTER -------------------------------------------------------
        size_t ndbCposs2 = ndbCposs1;
        size_t ndbCstrs2 = ndbCstrs1;
        size_t dbstr1len2 = dbstr1len1;
        size_t dbstrnlen2 = dbstrnlen1;
        size_t dbxpad2 = dbxpad1;

        if(0.0f < prescore && ndbCstrs1 && ndbCposs1)
            FilteroutReferences(
                queryndxpmbeg, queryndxpmend, querypmbeg, querypmend,
                bdbCdesc, bdbCpmbeg, bdbCpmend, bdbCndxpmbeg, bdbCndxpmend,
                nqyposs, nqystrs, ndbCposs1, ndbCstrs1, dbstr1len1, dbstrnlen1, dbxpad1,
                maxnsteps, ndbCposs2, ndbCstrs2, dbstr1len2, dbstrnlen2, dbxpad2,
                tmpdpdiagbuffers, tfmmemory, auxwrkmemory, globvarsbuf);
        //--- END FILTER ---------------------------------------------------


        //reinitialize passedstatscntrd_
        for(size_t i = 0; i < nqystrs; i++) {
            size_t loff = nDevGlobVariables * i;
            passedstatscntrd_[loff + dgvNPassedStrs] = ndbCstrs2;
            passedstatscntrd_[loff + dgvNPosits] = ndbCposs2;
        }



        // //execution configuration for scores initialization:
        // //each block processes one query and CUS1_TBSP_SCORE_SET_XDIM references:
        // dim3 nthrds_scinit(CUS1_TBSP_SCORE_SET_XDIM,1,1);
        // dim3 nblcks_scinit(
        //     (ndbCstrs2 + CUS1_TBSP_SCORE_SET_XDIM - 1)/CUS1_TBSP_SCORE_SET_XDIM,
        //     nqystrs, maxnsteps);

        MpStageSSRR stgSSRR(
            2/*maxndpiters*/,
            maxnsteps, minfraglen, prescore, stepinit,
            querypmbeg, querypmend, bdbCpmbeg, bdbCpmend,
            nqystrs, ndbCstrs2, nqyposs, ndbCposs2,
            qystr1len, dbstr1len2, qystrnlen, dbstrnlen2, dbxpad2,
            scores, tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, maxcoordsbuf, btckdata,
            wrkmemory, wrkmemoryccd, wrkmemorytm, wrkmemorytmibest,
            auxwrkmemory, wrkmemory2, alndatamemory, tfmmemory,
            globvarsbuf
        );

        //run superposition search by a combination of secondary structure and
        //sequence similarity
        if(addsearchbyss && ndbCstrs2 && ndbCposs2)
            stgSSRR.RunSpecialized<true/*USESEQSCORING*/>(2/*maxndpiters*/);

        // //reset best scores and convergence flags;
        // InitScores<INITOPT_BEST|INITOPT_CONVFLAG_ALL>
        //     <<<nblcks_scinit,nthrds_scinit,0,streamproc>>>(
        //         ndbCstrs2,  maxnsteps, 0/*minfraglen(unused)*/, false/*checkfragos*/,
        //         auxwrkmemory);
        // MYCUDACHECKLAST;



        //run the stage for finding optimal superpositions from spectral similarities
        if(nodetailedsearch == 0 && ndbCstrs2 && ndbCposs2)
            MpStageFrg3(
                2/*maxndpiters*/,
                maxnsteps, minfraglen, prescore, stepinit,
                querypmbeg, querypmend, bdbCpmbeg, bdbCpmend,
                queryndxpmbeg, queryndxpmend, bdbCndxpmbeg, bdbCndxpmend,
                nqystrs, ndbCstrs2, nqyposs, ndbCposs2,
                qystr1len, dbstr1len2, qystrnlen, dbstrnlen2, dbxpad2,
                scores, tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, maxcoordsbuf, btckdata,
                wrkmemory, wrkmemoryccd, wrkmemorytmalt, wrkmemorytm, wrkmemorytmibest,
                auxwrkmemory, wrkmemory2, alndatamemory, tfmmemory,
                globvarsbuf
            ).Run();

        //run the stage using the best superposition so far
        MpStage2 stg2(
            2/*maxndpiters*/,
            maxnsteps, minfraglen, prescore, stepinit,
            querypmbeg, querypmend, bdbCpmbeg, bdbCpmend,
            nqystrs, ndbCstrs2, nqyposs, ndbCposs2,
            qystr1len, dbstr1len2, qystrnlen, dbstrnlen2, dbxpad2,
            scores, tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, maxcoordsbuf, btckdata,
            wrkmemory, wrkmemoryccd, wrkmemorytm, wrkmemorytmibest,
            auxwrkmemory, wrkmemory2, alndatamemory, tfmmemory,
            globvarsbuf
        );

        //run DP refinement using SS information and best superposition so far
        if(ndbCstrs2 && ndbCposs2)
            stg2.RunSpecialized<false/* GAP0 */, true/* USESS */, D02IND_DPSCAN>(
                2/*maxndpiters*/, false/*check_for_low_scores*/, scorethld);

        //run DP refinement using the best superposition and final parameters
        if(ndbCstrs2 && ndbCposs2)
            stg2.RunSpecialized<true/* GAP0 */, false/* USESS */, D02IND_SEARCH>(
                maxndpiters, true/*check_for_low_scores*/, scorethld);

        //run the final stage for producing optimal alignments
        if(ndbCstrs2 && ndbCposs2)
            MpStageFin(
                d2equiv, maxnsteps, minfraglen,
                querypmbeg, querypmend, bdbCpmbeg, bdbCpmend,
                nqystrs, ndbCstrs2, nqyposs, ndbCposs2,
                qystr1len, dbstr1len2, qystrnlen, dbstrnlen2, dbxpad2,
                scores, tmpdpdiagbuffers, tmpdpbotbuffer, tmpdpalnpossbuffer, maxcoordsbuf, btckdata,
                wrkmemory, wrkmemoryccd, wrkmemorytm, wrkmemorytmibest,
                auxwrkmemory, wrkmemory2, alndatamemory, tfmmemory, alnsmemory,
                globvarsbuf
            ).Run();

        //total size for alignments, including padding
        size_t szaligns2 = gmem_->GetSizeOfDP2Alns(nqystrs, nqyposs, ndbCposs2, ndbCstrs2);

        t.Stop();

        TriggerFinalization(
            t.GetElapsedTime(),
            qrysernrbeg,
            nqyposs, nqystrs,
            //NOTE: original statistics for: ndbCposs, ndbCstrs!
            ndbCposs, ndbCstrs,
            querydesc, querypmbeg, querypmend,
            bdbCdesc, bdbCpmbeg, bdbCpmend,
            qrscnt, cnt,
            szaligns2,
            passedstatscntrd_.get()
        );

    } catch(myruntime_error const& ex) {
        mre = ex;
    } catch(myexception const& ex) {
        mre = ex;
    } catch(...) {
        mre = MYRUNTIME_ERROR("Exception caught.");
    }

    if( mre.isset())
        throw mre;
}





// =========================================================================
// TriggerFinalization: trigger finalization once results have been obtained
// by notifying Finalizer;
// tdrtn, time duration of computations;
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
void MpBatch::TriggerFinalization(
    double tdrtn,
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
    MYMSG("MpBatch::TriggerFinalization", 3);
    static const std::string preamb = "MpBatch::TriggerFinalization: ";
    static const float scorethld = CLOptions::GetO_S();

    const char* h_results = GetHeapSectionAddress(MpGlobalMemory::ddsBegOfDP2AlnData);

    // size_t szresults = 
    //     //size allocated for results: transform. matrices + aln. data:
    //     GetHeapSectionOffset(MpGlobalMemory::ddsEndOfTfmMatrices) -
    //     GetHeapSectionOffset(MpGlobalMemory::ddsBegOfDP2AlnData) +
    //     //plus actual alignment data size, as opposed to allocated
    //     szaligns2;

#ifdef __DEBUG__
    if(!(cbpfin_))
        throw MYRUNTIME_ERROR(preamb + "Null finalizer object.");
#endif

    int rspcode = cbpfin_->GetResponse();
    if(rspcode == CUBPTHREAD_MSG_ERROR)
        throw MYRUNTIME_ERROR(preamb + "Finalizer terminated with errors.");

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
        h_results,
        GetHeapSectionOffset(MpGlobalMemory::ddsEndOfDP2AlnData)-
            GetHeapSectionOffset(MpGlobalMemory::ddsBegOfDP2AlnData),
        GetHeapSectionOffset(MpGlobalMemory::ddsEndOfTfmMatrices)-
            GetHeapSectionOffset(MpGlobalMemory::ddsBegOfTfmMatrices),
        szaligns2,
        szaligns2
    );

    cbpfin_->Notify(TdFinalizer::cubpthreadmsgFinalize);
}
