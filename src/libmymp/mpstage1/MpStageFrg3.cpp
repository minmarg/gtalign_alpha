/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include <omp.h>

#include <math.h>
#include <string>

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libutil/templates.h"
#include "libutil/CLOptions.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libgenp/gdats/PMBatchStrData.h"
#include "libmymp/mpproc/mpprocconf.h"
#include "libmymp/mplayout/MpGlobalMemory.h"
#include "libmymp/mpstages/scoringbase.h"
#include "libmymp/mpstages/covariancebase.h"
#include "libmymp/mpstages/linearscoringbase.h"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmymp/mpstage1/MpStageBase.h"
#include "libmymp/mpstage1/MpStage1.h"
#include "libmymp/mpdp/MpDPHub.h"
#include "MpStageFrg3.h"

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// ExtensiveFrgSwift/ScoreBasedOnFragmatching3Kernel: calculate tmscores and
// find most favorable initial superposition based on fragment matching of
// multiple queries and references;
//
void MpStageFrg3::ScoreBasedOnFragmatching3Kernel(
    const float thrsimilarityperc,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const char* const * const __RESTRICT__ queryndxpmbeg,
    const char* const * const __RESTRICT__ bdbCndxpmbeg,
    float* const __RESTRICT__ wrkmemtmibest,
    float* const __RESTRICT__ wrkmemtm,
    float* const __RESTRICT__ wrkmemaux,
    const char* const __RESTRICT__ dpscoremtx)
{
    enum{
        nFRGS = 2,//number of fragments of different length used
        LENTHR = 150,//length threshold
        nEFFDS = twmvEndOfCCDataExt,//effective number of fields
        COVDIMX = MPS1_TBINITSP_COMPLETEREFINE_XDIM,
        DPSCYIts = 3,//number of iterations along y-axis
        DPSCDIMX = MPSF_TBSP_LOCAL_SIMILARITY_XDIM,
        //NOTE: ensure DPSCDIMY <= DPSCDIMX!!
        DPSCDIMY = MPSF_TBSP_LOCAL_SIMILARITY_YDIM * DPSCYIts,
        SCORDIMX = CUSF_TBSP_INDEX_SCORE_POSLIMIT2,//MPSF_TBSP_INDEX_SCORE_XDIM,
        SZSTCK = 17 * nStks_,//max size for stack
        NFCRDS = nTDPAlignedPoss + 1,//number of coordinates fields
        N2SCTS = CUS1_TBSP_DPSCORE_TOP_N_MAX_CONFIGS,//number of secondary sections
        NTOPTFMS = CUS1_TBSP_DPSCORE_TOP_N,//number of best-performing tfms
        MAXDIMX = MPS1_TBSP_SCORE_MAX_XDIM//MPSF_TBSP_COMMON_INDEX_SCORE_XDIM
    };

    MYMSG("MpStageFrg3::ScoreBasedOnFragmatching3Kernel", 4);
    // static const std::string preamb = "MpStageFrg3::ExtensiveFrgSwift: ";
    static const int depth = CLOptions::GetC_DEPTH();
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = CuMemoryBase::GetMinMemAlignment();

    //#iterations on alignment pairs to perform excluding the
    //initial (tfm) and final (score) kernels: 1, 2, or 3:
    static const int napiterations = 2;
    static const bool scoreachiteration = false;
    // static const bool dynamicorientation = true;
    // static const float thrscorefactor = 1.0f;

    // static const int frags[nFRGS] = {20, 100};
    // static const int fragndx = 1;//should be the longest!
    //int fraglen = GetNAlnPoss_frg(
    //        qystr1len, dbstr1len, 0/*qrypos,unsed*/, 0/*rfnpos,unused*/,
    //        0/*qryfragfct,unsed*/, 0/*rfnfragfct,unused*/, 0/*fraglen index*/);

    const int fctdiv =
        (depth==CLOptions::csdShallow)? GetFragStepSize_frg_shallow_factor(): 1;
    const int minnsteps = 10 / fctdiv;
    int qrystepsz, rfnstepsz;
    GetQryRfnStepsize2(depth, qystr1len_, dbstr1len_, &qrystepsz, &rfnstepsz);

    //set minimum #steps to 10 since length 150 leads to 150/15=10,
    //the largest among #steps for medium-sized structures:
    const int nstepsy = mymax(minnsteps, (int)qystr1len_/qrystepsz + 1);
    const int nstepsx = mymax(minnsteps, (int)dbstr1len_/rfnstepsz + 1);

    const int nblocks_x = ndbCstrs_;
    const int nblocks_x_best = (ndbCstrs_ + COVDIMX - 1) / COVDIMX;
    const int nblocks_x_max = (ndbCstrs_ + MAXDIMX - 1) / MAXDIMX;
    const int nblocks_z = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_z * (size_t)nstepsy * (size_t)nstepsx * //nFRGS *
         (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize =
        (int)mymin(chunksize_helper, (size_t)mymin((int)ndbCstrs_, 128/*MPSF_TBSP_COMMON_INDEX_SCORE_CHSIZE*/));

    size_t chunksizeinit_helper = 
        ((size_t)nblocks_z * (size_t)nthreads * (size_t)nblocks_x_best + (size_t)nthreads - 1) / nthreads;
    const int chunksizeinit = (int)mymin(chunksizeinit_helper, (size_t)MPSF_TBSP_COMMON_INDEX_SCORE_CHSIZE);

    chunksizeinit_helper = 
        ((size_t)nblocks_z * (size_t)nblocks_x_max + (size_t)nthreads - 1) / nthreads;
    const int chunksize2 = (int)mymin(chunksizeinit_helper, (size_t)MPSF_TBSP_COMMON_INDEX_SCORE_CHSIZE);

    //step number (<=maxnsteps) for efficiently launching numerous processing 
    //kernels and calculating scores on the query-reference dimensions:
    // int stepnumber = 0;
    // int ysndxproc = 0, xsndxproc = 0;//processed indices

    //there are fragment variants; divide max allowed accommodation by 2:
    // const int maxnstepso2 = (maxnsteps_ >> 1);

    //alignment length can be >min due to the linear algorithm;
    //stack size depends on the length of the structure indexed:
    const int maxlenmax =
        /* dynamicorientation?  */mymax(dbstr1len_, qystr1len_)/* : qystr1len_ */;
    //dynamically determined stack size:
    int stacksize = 1;
    if(0 < maxlenmax) stacksize = mymin((int)17, (int)ceilf(log2f(maxlenmax)) + 1);

    int convflags[nFRGS];
    unsigned char dpsc[DPSCDIMX];//dp score cache
    float ccm[nEFFDS][COVDIMX];//cross-covarinace matrix and related
    float tfm[nEFFDS];//nEFFDS>nTTranformMatrix
    float stack[SZSTCK];//stack for recursion
    float coords[NFCRDS * SCORDIMX];
    char ssas[SCORDIMX];//secondary structure assignments
    float scoN[NTOPTFMS][MAXDIMX];//top N scores for finding max
    int ndxN[NTOPTFMS][MAXDIMX];//indices of top N best-performing configs

    //minimum #sections of scores for a query-reference pair:
    const int silimit = mymax(nthreads, (int)NTOPTFMS);

    //NOTE: constants, members, and pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        shared(stacksize) \
        private(convflags, dpsc, ccm,tfm, stack,coords,ssas, scoN,ndxN)
    {
        //initialize best scores and flags
        #pragma omp for collapse(3) schedule(dynamic, chunksizeinit)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < silimit; si++)
                for(int bi = 0; bi < nblocks_x_best; bi++)
                {//threads process blocks of references
                    const int istr0 = bi * COVDIMX;
                    const int istre = mymin(istr0 + COVDIMX, (int)ndbCstrs_);
                    const int mloc = ((qi * maxnsteps_ + si) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    const int sloc = ((qi * maxnsteps_ + silimit *N2SCTS) * ndbCstrs_) * nTTranformMatrix;
                    #pragma omp simd aligned(wrkmemaux,wrkmemtmibest:memalignment)
                    for(int ri = istr0; ri < istre; ri++) {
                        wrkmemaux[mloc + tawmvBestScore * ndbCstrs_ + ri] = 0.0f;
                        wrkmemaux[mloc + tawmvConverged * ndbCstrs_ + ri] = 0.0f;
                        wrkmemaux[mloc + tawmvNAlnPoss * ndbCstrs_ + ri] = 0.0f;
                        //secondary scores:
                        wrkmemtmibest[sloc + (silimit * 0 + si) * ndbCstrs_ + ri] = 0.0f;
                        wrkmemtmibest[sloc + (silimit * 1 + si) * ndbCstrs_ + ri] = 0.0f;
                    }
                }
        //implicit barrier here

        //initialize a section index for scores and tfms:
        int sfragfctxndx = -1;

        //ysndx and xsndx, step indices over query and reference structures;
        //maxnsteps is max allowed steps to be processed in parallel simultaneously
        #pragma omp for collapse(4) schedule(static, chunksize)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int ysndx = 0; ysndx < nstepsy; ysndx++)
                for(int xsndx = 0; xsndx < nstepsx; xsndx++)
                    for(int ri = 0; ri < nblocks_x; ri++)
                    {
                        //check convergence:
                        int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                        if(((int)(wrkmemaux[mloc0 + ri])) > (CONVERGED_FRAGREF_bitval))
                            continue;
                        //NOTE: differently from the GPU version, no flag value distributes across sfragfct;
                        //NOTE: that's because of different processing layout (complete within the scope of a pair).

                        const int qrylen = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                        const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                        const int dbstrlen = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                        const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);
                        int qrystep, rfnstep;
                        GetQryRfnStepsize2(depth, qrylen, dbstrlen, &qrystep, &rfnstep);

                        const int qrypos = (ysndx) * qrystep;
                        const int rfnpos = (xsndx) * rfnstep;
                        const int tid = omp_get_thread_num();

                        const bool sec2met =
                            (depth <= CLOptions::csdHigh) &&
                            ((qrylen <= LENTHR) || ((ysndx/*qryfragfct*/ & 1) == 0)) && 
                            ((dbstrlen <= LENTHR) || ((xsndx/*rfnfragfct*/ & 1) == 0));

                        const bool sec3met = //assume compiler's optimization on mod 3; see myfastmod3
                            (depth <= CLOptions::csdDeep) &&
                            ((qrylen <= LENTHR) || ((ysndx % 3) == 0)) && 
                            ((dbstrlen <= LENTHR) || ((xsndx % 3) == 0));

                        //threshold calculated for the original lengths
                        float d02 = GetD02(qrylen, dbstrlen);

                        CalcLocalSimilarity2_frg2<nFRGS,DPSCDIMY,DPSCDIMX,memalignment>(
                            thrsimilarityperc, ndbCposs_, dbxpad_,
                            qrydst, dbstrdst,  qrylen, dbstrlen,  qrypos, rfnpos,
                            dpscoremtx, dpsc, convflags);

                        for(int fi = 0; fi < nFRGS; fi++)
                        {
                            if(convflags[fi]) continue;

                            const int fraglen = GetNAlnPoss_frg(
                                qrylen, dbstrlen, 0/*qrypos,unused*/, 0/*rfnpos,unused*/,
                                ysndx/*unused*/, xsndx/*unused*/, fi/*fragndx*/);

                            CalcCCMatrices_Complete<nEFFDS,COVDIMX,memalignment>(
                                qrydst, dbstrdst, fraglen,  qrypos, rfnpos,
                                querypmbeg, bdbCpmbeg, ccm);

                            if(ccm[twmvNalnposs][0] < 1.0f) continue;

                            //copy sums to tfm:
                            #pragma omp simd
                            for(int f = 0; f < nEFFDS; f++) tfm[f] = ccm[f][0];

                            //dynamicorientation = true always; stable and symmetric:
                            CalcTfmMatrices_DynamicOrientation_Complete(qrylen, dbstrlen, tfm);

                            for(int n = 0; n < napiterations; n++)
                            {
                                bool secstrmatchaln = (n+1 < napiterations);
                                // bool completealn = true;//(napiterations <= n+1);
                                bool reversetfms = !scoreachiteration && (n+1 < napiterations);
                                bool writeqrypss = !reversetfms;//write query positions

                                ProduceAlignmentUsingDynamicIndex2<SCORDIMX,PMBSdatalignment>(
                                    secstrmatchaln, stack, stacksize,  qrydst, dbstrdst,
                                    qrylen, dbstrlen, qrypos, rfnpos, fraglen, writeqrypss,
                                    querypmbeg, bdbCpmbeg, queryndxpmbeg, bdbCndxpmbeg,
                                    tfm, coords, ssas);

                                //NOTE: alignment length written in tfm[twmvNalnposs];
                                CalcCCMatrices_SWFTscan_Complete<nEFFDS,COVDIMX,SCORDIMX>(
                                    tfm[twmvNalnposs]/*nalnposs*/, coords, ccm);

                                if(ccm[twmvNalnposs][0] < 1.0f) continue;

                                //copy sums to tfm:
                                #pragma omp simd
                                for(int f = 0; f < nEFFDS; f++) tfm[f] = ccm[f][0];

                                if(reversetfms)
                                    CalcTfmMatrices_DynamicOrientation_Complete(qrylen, dbstrlen, tfm);
                                else
                                    CalcTfmMatrices_Complete(qrylen, dbstrlen, tfm);

                                if(reversetfms) continue;

                                CalcScoresUnrl_SWFTscanProgressive_Complete<nEFFDS,COVDIMX,SCORDIMX>(
                                    d02, ccm[twmvNalnposs][0]/*nalnposs*/, tfm, coords, ccm);

                                //{{
                                if(NTOPTFMS <= sfragfctxndx || sfragfctxndx < 0) sfragfctxndx = tid;

                                //NOTE: ccm[0][0] is a score written at [0,0]; CONDITIONAL==true
                                SaveBestScoreAndTM_Complete<false/*WRITEFRAGINFO*/,true/*CONDITIONAL*/>(
                                    ccm[0][0]/*best*/,  qi/*qryndx*/, ri/*dbstrndx*/, ndbCstrs_, 
                                    maxnsteps_, sfragfctxndx, 0/*sfragndx*/, 0/*sfragpos*/,
                                    tfm, wrkmemtmibest, wrkmemaux);

                                //update secondary scores and relative tfms:
                                if(sec2met)
                                    Save2ndryScoreAndTM_Complete<N2SCTS,NTOPTFMS,0/*SECTION2*/,memalignment>(
                                        ccm[0][0]/*best*/, qi/*qryndx*/, ri/*dbstrndx*/, ndbCstrs_,
                                        maxnsteps_, nthreads, sfragfctxndx, tfm, wrkmemtmibest);
                                if(sec3met)
                                    Save2ndryScoreAndTM_Complete<N2SCTS,NTOPTFMS,1/*SECTION2*/,memalignment>(
                                        ccm[0][0]/*best*/, qi/*qryndx*/, ri/*dbstrndx*/, ndbCstrs_,
                                        maxnsteps_, nthreads, sfragfctxndx, tfm, wrkmemtmibest);

                                //section index grows up to NTOPTFMS:
                                sfragfctxndx += nthreads;
                                //}}

                                //TODO: running this branch would require ccmLast<-ccm before!
                                //NOTE: this does not change the result!
                                if(scoreachiteration && (n+1 < napiterations))
                                    CalcTfmMatrices_Complete(qrylen, dbstrlen, tfm);
                            }
                        }
                    }//omp for
        //implicit barrier here

        //find the max score among fragment variants
        #pragma omp for collapse(2) schedule(dynamic, chunksize2)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int bi = 0; bi < nblocks_x_max; bi++)
            {//threads process blocks of references
                SaveTopNScoresAndTMsAmongBests<NTOPTFMS,MAXDIMX,memalignment>(
                    qi/*qryndx*/, bi/*rfnblkndx*/, ndbCstrs_,
                    maxnsteps_, nthreads/*effnsteps*/,
                    scoN, ndxN, wrkmemtmibest, wrkmemtm, wrkmemaux);
                //select best secondary scores and relative tfms:
                if(depth <= CLOptions::csdHigh)
                    SaveTopNScoresAndTMsAmongSecondaryBests
                        <N2SCTS,NTOPTFMS,0/*SECTION2*/,MAXDIMX,memalignment>(
                            qi/*qryndx*/, bi/*rfnblkndx*/, ndbCstrs_,
                            maxnsteps_, nthreads/*effnsteps*/, nthreads,
                            scoN, ndxN, wrkmemtmibest, wrkmemtm, wrkmemaux);
                if(depth <= CLOptions::csdDeep)
                    SaveTopNScoresAndTMsAmongSecondaryBests
                        <N2SCTS,NTOPTFMS,1/*SECTION2*/,MAXDIMX,memalignment>(
                            qi/*qryndx*/, bi/*rfnblkndx*/, ndbCstrs_,
                            maxnsteps_, nthreads/*effnsteps*/, nthreads,
                            scoN, ndxN, wrkmemtmibest, wrkmemtm, wrkmemaux);
            }
    }//omp parallel
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// SortAmongDPswiftsKernel: sort best DP scores along with respective 
// transformation matrices by considering all partial DP swift scores 
// calculated for all query-reference pairs in rhe chunk;
//
void MpStageFrg3::SortAmongDPswiftsKernel(
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* const __RESTRICT__ tmpdpdiagbuffers,
    const float* const __RESTRICT__ wrkmemtm,
    float* const __RESTRICT__ wrkmemtmtarget,
    float* const __RESTRICT__ wrkmemaux)
{
    enum{
        N2SCTS = CUS1_TBSP_DPSCORE_TOP_N_MAX_CONFIGS,//number of secondary sections
        NTOPTFMS = CUS1_TBSP_DPSCORE_TOP_N,//number of best-performing tfms
        NMAXREFN = CUS1_TBSP_DPSCORE_TOP_N_REFINEMENT,//max #branches for refinement
        MAXDIMX = MPS1_TBSP_SCORE_MAX_XDIM//MPSF_TBSP_COMMON_INDEX_SCORE_XDIM
    };

    MYMSG("MpStageFrg3::SortAmongDPswiftsKernel", 4);
    // static const std::string preamb = "MpStageFrg3::SortAmongDPswiftsKernel: ";
    static const int depth = CLOptions::GetC_DEPTH();
    static const int nbranches = CLOptions::GetC_NBRANCHES();
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = mycemin((size_t)PMBSdatalignment, CuMemoryBase::GetMinMemAlignment());

    //configuration for swift DP: optimal order-dependent score sections:
    const int nconfigsections = 
        (depth <= CLOptions::csdDeep)? 3: ((depth <= CLOptions::csdHigh)? 2: 1);
    const int nblocks_x_max = (ndbCstrs_ + MAXDIMX - 1) / MAXDIMX;
    const int nblocks_s = nconfigsections;
    const int nblocks_z = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_z * (size_t)nblocks_s * (size_t)nblocks_x_max + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPSF_TBSP_COMMON_INDEX_SCORE_CHSIZE);

    float scoN[MAXDIMX][NTOPTFMS];//top N scores for finding max
    int ndxN[MAXDIMX][NTOPTFMS];//indices of top N best-performing configs

    //NOTE: constants, members, and pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        private(scoN,ndxN)
    {
        //find the max score among DP swifts
        #pragma omp for collapse(3) schedule(dynamic, chunksize)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int ss = 0; ss < nblocks_s; ss++)
                for(int bi = 0; bi < nblocks_x_max; bi++)
                {//threads process blocks of references
                    SortBestDPscoresAndTMsAmongDPswifts<N2SCTS,NTOPTFMS,NMAXREFN,MAXDIMX,memalignment>(
                        ss/*SECTION*/, nbranches, qi/*qryndx*/, bi/*rfnblkndx*/,
                        ndbCstrs_, ndbCposs_, dbxpad_, maxnsteps_,
                        scoN, ndxN, bdbCpmbeg, tmpdpdiagbuffers, wrkmemtm, wrkmemtmtarget, wrkmemaux);
                }
    }//omp parallel
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// Refine_tfmaltconfig: refine all alternative best-performing 
// superpositions obtained through the extensive application of spatial 
// index;
//
void MpStageFrg3::Refine_tfmaltconfig()
{
    MYMSG("MpStageFrg3::Refine_tfmaltconfig", 4);
    // static const std::string preamb = "MpStageFrg3::Refine_tfmaltconfig: ";

    //configurations to verify alternatively:
    //(CUS1_TBSP_DPSCORE_TOP_N_REFINEMENT)
    static const int nbranches = CLOptions::GetC_NBRANCHES();
    static const int depth = CLOptions::GetC_DEPTH();
    const int nconfigsections = 
        (depth <= CLOptions::csdDeep)? 3: ((depth <= CLOptions::csdHigh)? 2: 1);
    const int nconfigs = nbranches * nconfigsections;

    constexpr bool vANCHORRGN = false;//using anchor region
    constexpr bool vBANDED = false;//banded alignment
    constexpr bool GAP0 = true;

    //top N tfms from the extensive application of spatial index:
    for(int rci = 0; rci < nconfigs; rci++)
    {
        dphub_.ExecDPwBtck128xKernel<vANCHORRGN,vBANDED,GAP0,D02IND_SEARCH,true/*ALTSCTMS*/>(
            0.0f/*gap open cost*/, rci/*stepnumber*/,
            querypmbeg_, bdbCpmbeg_,  wrkmemtmalt_/*in*/,
            wrkmemaux_,  tmpdpdiagbuffers_, tmpdpbotbuffer_, btckdata_);

        dphub_.BtckToMatched128xKernel<vANCHORRGN,vBANDED>(
            rci/*stepnumber*/,
            querypmbeg_, bdbCpmbeg_, btckdata_, wrkmemaux_, tmpdpalnpossbuffer_);

        if(rci < 1)
            RefineFragDPKernelCaller<SECONDARYUPDATE_UNCONDITIONAL>(false/*readlocalconv*/, FRAGREF_NMAXCONVIT);
        else RefineFragDPKernelCaller<SECONDARYUPDATE_CONDITIONAL>(false/*readlocalconv*/, FRAGREF_NMAXCONVIT);

        if(depth <= CLOptions::csdHigh)
        {   //one additional iteration of full DP sweep
            dphub_.ExecDPwBtck128xKernel<vANCHORRGN,vBANDED,GAP0,D02IND_SEARCH,false/*ALTSCTMS*/>(
                0.0f/*gap open cost*/, 0/*stepnumber*/,
                querypmbeg_, bdbCpmbeg_,  wrkmemtmibest_/*in*/,
                wrkmemaux_,  tmpdpdiagbuffers_, tmpdpbotbuffer_, btckdata_);

            dphub_.BtckToMatched128xKernel<vANCHORRGN,vBANDED>(
                0/*stepnumber*/,
                querypmbeg_, bdbCpmbeg_, btckdata_, wrkmemaux_, tmpdpalnpossbuffer_);

            if(rci < 1)
                RefineFragDPKernelCaller<SECONDARYUPDATE_UNCONDITIONAL>(false/*readlocalconv*/, FRAGREF_NMAXCONVIT);
            else RefineFragDPKernelCaller<SECONDARYUPDATE_CONDITIONAL>(false/*readlocalconv*/, FRAGREF_NMAXCONVIT);
        }
    }//rci
}
