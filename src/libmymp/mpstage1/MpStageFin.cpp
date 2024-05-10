/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include <omp.h>

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
#include "libmycu/custages/stagecnsts.cuh"
#include "libmymp/mpstage1/MpStageBase.h"
#include "libmymp/mpdp/MpDPHub.h"
#include "MpStageFin.h"

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
//NOTE: use static scheduling as opposed to dynamic to ensure deterministic nature!
//NOTE: that's because same scores may result from fragment-based optimization!
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// FinalFragmentBasedDPAlignmentRefinementPhase1Kernel: perform final alignment 
// refinement based on the best superposition obtained in the course of 
// complete superposition search within the single kernel's actions;
// D0FINAL, template flag of using the final threshold value for D0;
// CHCKDST, template parameter to request accumulating scores within the 
// given threshold distance only;
// TFM_DINV, use doubly inverted transformation matrices under suitable conditions;
// nmaxconvit, maximum number of superposition iterations;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// tmpdpdiagbuffers, temporary diagonal buffers for positional scores/dsts;
// wrkmemtmibest, working memory for best-performing transformation matrices;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// 
template<bool D0FINAL, int CHCKDST, bool TFM_DINV>
void MpStageFin::FinalFragmentBasedDPAlignmentRefinementPhase1Kernel(
    const int nmaxconvit,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* const __RESTRICT__ tmpdpalnpossbuffer,
    float* const __RESTRICT__ tmpdpdiagbuffers,
    float* const __RESTRICT__ wrkmemtmibest,
    float* const __RESTRICT__ wrkmemaux,
    float* const __RESTRICT__ tfmmem)
{
    enum{
        //effective number of fields:
        nEFFDS = twmvEndOfCCDataExt,
        XDIM = MPS1_TBINITSP_COMPLETEREFINE_XDIM
    };

    MYMSG("MpStageFin::FinalFragmentBasedDPAlignmentRefinementPhase1Kernel", 4);
    static const std::string preamb = "MpStageFin::FinalFragmentBasedDPAlignmentRefinementPhase1Kernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = CuMemoryBase::GetMinMemAlignment();
    //minimum length among largest
    int minlenmax = myhdmin(qystr1len_, dbstr1len_);
    int maxalnmax = minlenmax;
    //maximum number of subdivisions
    const int nmaxsubfrags = FRAGREF_NMAXSUBFRAGS;
    // sfragstep, step to traverse subfragments;
    constexpr int sfragstep = FRAGREF_SFRAGSTEP;

    int nlocsteps = 0;
    nlocsteps = GetMaxNFragSteps(maxalnmax, sfragstep, minfraglen_);
    nlocsteps *= nmaxsubfrags;//total number across all fragment lengths

    if(nlocsteps < 1 || (int)maxnsteps_ < nlocsteps)
        throw MYRUNTIME_ERROR(preamb +
        "Invalid number of superposition tests: "+std::to_string(nlocsteps));

    //execution configuration: process and refine multiple
    //query-reference alignment variants:
    const int nblocks_x = ndbCstrs_;
    const int nblocks_x_best = (ndbCstrs_ + XDIM - 1) / XDIM;
    const int nblocks_y = nlocsteps;
    const int nblocks_z = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_z * (size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPS1_TBINITSP_COMPLETEREFINE_CHSIZE);

    size_t chunksizeinit_helper = 
        ((size_t)nblocks_z * (size_t)nthreads * (size_t)nblocks_x_best + (size_t)nthreads - 1) / nthreads;
    const int chunksizeinit = (int)mymin(chunksizeinit_helper, (size_t)MPS1_TBINITSP_COMPLETEREFINE_CHSIZE);

    //cache for the cross-covarinace matrix and related data: 
    float ccm[nEFFDS][XDIM];
    float tfm[nEFFDS];//nEFFDS>nTTranformMatrix
    float ccmLast[nEFFDS];
    float tfmBest[nEFFDS];

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        private(ccm, tfm, ccmLast, tfmBest)
    {
        //initialize best scores
        #pragma omp for collapse(3) schedule(dynamic, chunksizeinit)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nthreads; si++)
                for(int bi = 0; bi < nblocks_x_best; bi++)
                {//threads process blocks of references
                    const int istr0 = bi * XDIM;
                    const int istre = mymin(istr0 + XDIM, (int)ndbCstrs_);
                    const int mloc = ((qi * maxnsteps_ + si) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    #pragma omp simd aligned(wrkmemaux:memalignment)
                    for(int ri = istr0; ri < istre; ri++)
                        wrkmemaux[mloc + tawmvBestScore * ndbCstrs_ + ri/*dbstrndx*/] = 0.0f;
                }
        //implicit barrier here

        #pragma omp for collapse(3) schedule(static, chunksize)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nblocks_y; si++)
                for(int ri = 0; ri < nblocks_x; ri++)
                {//threads process references
                    //check convergence:
                    int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                    tfm[6] = wrkmemaux[mloc0 + ri];//reuse cache
                    if(((int)(tfm[6])) & (CONVERGED_LOWTMSC_bitval))
                        continue;

                    const int sfragfct = si / nmaxsubfrags;//fragment factor
                    const int sfragndx = si - sfragfct * nmaxsubfrags;//fragment length index

                    const int qrylenorg = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                    // const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                    const int dbstrlenorg = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                    const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);

                    enum {qrypos = 0, rfnpos = 0};
                    const int sfragpos = sfragfct * sfragstep;
                    const int tid = omp_get_thread_num();
                    int qrylen, dbstrlen;

                    //NOTE: #matched positions tawmvNAlnPoss written at sfragfct==0:
                    mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    qrylen = dbstrlen = wrkmemaux[mloc0 + tawmvNAlnPoss * ndbCstrs_ + ri];

                    //out-of-bounds check:
                    const int fraglen = GetFragLength(qrylen, dbstrlen, qrypos, rfnpos, sfragndx);
                    if(fraglen < 1) continue;
                    if(qrylen + sfragstep <= qrypos + sfragpos + fraglen ||
                       dbstrlen + sfragstep <= rfnpos + sfragpos + fraglen)
                        continue;

                    //threshold calculated for the original lengths
                    const float d0 = D0FINAL? GetD0fin(qrylenorg, dbstrlenorg): GetD0(qrylenorg, dbstrlenorg);
                    const float d02 = SQRD(d0);
                    const float d82 = GetD82(qrylenorg, dbstrlenorg);
                    float dst32 = CP_LARGEDST;
                    float best = -1.0f;//best score obtained

                    CalcCCMatrices_DPRefined_Complete<nEFFDS,XDIM,memalignment>(
                        qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_, tid/*sfragfctxndx*/, dbstrdst, fraglen,
                        qrylen, dbstrlen,  qrypos + sfragpos, rfnpos + sfragpos,
                        tmpdpalnpossbuffer, ccm);

                    for(int cit = 0; cit < nmaxconvit + 2; cit++)
                    {
                        if(0 < cit) {
                            CalcCCMatrices_DPRefinedExtended_Complete<nEFFDS,XDIM,memalignment>(
                                (cit < 2)? READCNST_CALC: READCNST_CALC2,
                                qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_, tid/*sfragfctxndx*/, dbstrdst,
                                qrylen, dbstrlen, qrypos, rfnpos,  d0, dst32,
                                tmpdpdiagbuffers, tmpdpalnpossbuffer, ccm);

                            CheckConvergenceRefined_Complete<nEFFDS,XDIM>(ccm, ccmLast);
                            if(ccmLast[0]) break;//converged
                        }

                        if(ccm[twmvNalnposs][0] < 1.0f) break;

                        //copy sums to tfm and ccmLast:
                        #pragma omp simd
                        for(int f = 0; f < nEFFDS; f++) tfm[f] = ccmLast[f] = ccm[f][0];

                        CalcTfmMatrices_Complete<TFM_DINV>(qrylenorg, dbstrlenorg, tfm);

                        CalcScoresUnrl_DPRefined_Complete<XDIM,memalignment,CHCKDST>(
                            (cit < 1)? READCNST_CALC: READCNST_CALC2,
                            qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_, tid/*sfragfctxndx*/, dbstrdst,
                            qrylen, dbstrlen, qrypos, rfnpos,  d0, d02, d82,
                            tmpdpdiagbuffers, tmpdpalnpossbuffer,  tfm, ccm[0]/*scv*/, ccm/*dstv*/);

                        //distance threshold for at least three aligned pairs:
                        dst32 = ccm[0][1];
                        if(best < ccm[0][0]) {
                            best = ccm[0][0];//score written at [0,0]
                            #pragma omp simd
                            for(int f = 0; f < nTTranformMatrix; f++) tfmBest[f] = tfm[f];
                        }
                    }//for(;cit;)

                    //NOTE: CONDITIONAL==true because effnsteps(==nthreads)<<maxnsteps
                    SaveBestScoreAndTM_Complete<true/*WRITEFRAGINFO*/,true/*CONDITIONAL*/>(
                        best,  qi/*qryndx*/, ri/*dbstrndx*/, ndbCstrs_, 
                        maxnsteps_, tid/*sfragfctxndx*/, sfragndx, sfragpos,
                        tfmBest, wrkmemtmibest, wrkmemaux);
                }//omp for
        //implicit barrier here

        //find the max score among fragment variants
        #pragma omp for collapse(2)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int bi = 0; bi < nblocks_x_best; bi++)
            {//threads process blocks of references
                SaveBestScoreAndTMAmongBests
                    <XDIM,memalignment,
                     true/*WRITEFRAGINFO*/,
                     true/*GRANDUPDATE*/,
                     true/*FORCEWRITEFRAGINFO*/,
                     SECONDARYUPDATE_NOUPDATE>(
                        qi/*qryndx*/, bi/*rfnblkndx*/, ndbCstrs_,
                        maxnsteps_, nthreads/*effnsteps*/,
                        ccm, wrkmemtmibest, tfmmem, wrkmemaux, NULL/*wrkmemtm*/);
            }
    }
}

// =========================================================================
// Instantiations
// 
#define INSTANTIATE_MpStageFin_FinalFragmentBasedDPAlignmentRefinementPhase1Kernel(tpD0FINAL,tpCHCKDST,tpTFM_DINV) \
    template void MpStageFin::FinalFragmentBasedDPAlignmentRefinementPhase1Kernel<tpD0FINAL,tpCHCKDST,tpTFM_DINV>( \
        const int nmaxconvit, \
        const char* const * const __RESTRICT__ querypmbeg, \
        const char* const * const __RESTRICT__ bdbCpmbeg, \
        const float* const __RESTRICT__ tmpdpalnpossbuffer, \
        float* const __RESTRICT__ tmpdpdiagbuffers, \
        float* const __RESTRICT__ wrkmemtmibest, \
        float* const __RESTRICT__ wrkmemaux, \
        float* const __RESTRICT__ tfmmem);

INSTANTIATE_MpStageFin_FinalFragmentBasedDPAlignmentRefinementPhase1Kernel(false,CHCKDST_CHECK,false);
INSTANTIATE_MpStageFin_FinalFragmentBasedDPAlignmentRefinementPhase1Kernel(false,CHCKDST_CHECK,true);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// FinalFragmentBasedDPAlignmentRefinementPhase2Kernel: phase 2 to perform 
// finer-scale refinement of the the best superposition obtained 
// within the single kernel's actions;
// D0FINAL, template flag of using the final threshold value for D0;
// CHCKDST, template parameter to request accumulating scores within the 
// given threshold distance only;
// TFM_DINV, use doubly inverted transformation matrices under suitable conditions;
// nmaxconvit, maximum number of superposition iterations;
// maxnfragfcts, max number of fragment position factors around an identified position;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// tmpdpdiagbuffers, temporary diagonal buffers for positional scores/dsts;
// wrkmemtmibest, working memory for best-performing transformation matrices;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// tfmmem, memory region for transformation matrices;
// 
template<bool D0FINAL, int CHCKDST, bool TFM_DINV>
void MpStageFin::FinalFragmentBasedDPAlignmentRefinementPhase2Kernel(
    const int nmaxconvit,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* const __RESTRICT__ tmpdpalnpossbuffer,
    float* const __RESTRICT__ tmpdpdiagbuffers,
    float* const __RESTRICT__ wrkmemtmibest,
    float* const __RESTRICT__ wrkmemaux,
    float* const __RESTRICT__ tfmmem)
{
    enum{
        //effective number of fields:
        nEFFDS = twmvEndOfCCDataExt,
        XDIM = MPS1_TBINITSP_COMPLETEREFINE_XDIM
    };

    MYMSG("MpStageFin::FinalFragmentBasedDPAlignmentRefinementPhase2Kernel", 4);
    static const std::string preamb = "MpStageFin::FinalFragmentBasedDPAlignmentRefinementPhase2Kernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    // static const int refinement = CLOptions::GetC_REFINEMENT();
    constexpr int memalignment = CuMemoryBase::GetMinMemAlignment();
    //minimum length among largest
    int minlenmax = myhdmin(qystr1len_, dbstr1len_);
    int maxalnmax = minlenmax;
    //maximum number of subdivisions
    // const int nmaxsubfrags = FRAGREF_NMAXSUBFRAGS;
    //NOTE: step for the SECOND phase to final (finer-scale) refinement;
    constexpr int sfragstep = FRAGREF_SFRAGSTEP_mini;

    //max #fragment position factors around an identified position
    //**********************************************************************
    //NOTE: multiply maxalnmax by 2 since sub-optimal (first-phase) alignment
    //NOTE: position can be identified, e.g., at the end of alignment!
    //**********************************************************************
    int maxnfragfcts = myhdmin(2 * maxalnmax, CUSFN_TBSP_FIN_REFINEMENT_MAX_NPOSITIONS);
    maxnfragfcts = (maxnfragfcts + sfragstep-1) / sfragstep;
    int nlocsteps2 = maxnfragfcts;//total number for ONE fragment length
    // if(refinement == CLOptions::csrFullASearch) nlocsteps2 *= nmaxsubfrags;//total number across all fragment lengths

    if(nlocsteps2 < 1 || (int)maxnsteps_ < nlocsteps2)
        throw MYRUNTIME_ERROR(preamb +
        "Invalid number of superposition tests: "+std::to_string(nlocsteps2));

    //execution configuration: process and refine multiple
    //query-reference alignment variants:
    const int nblocks_x = ndbCstrs_;
    const int nblocks_x_best = (ndbCstrs_ + XDIM - 1) / XDIM;
    const int nblocks_y = nlocsteps2;
    const int nblocks_z = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_z * (size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPS1_TBINITSP_COMPLETEREFINE_CHSIZE);

    size_t chunksizeinit_helper = 
        ((size_t)nblocks_z * (size_t)nthreads * (size_t)nblocks_x_best + (size_t)nthreads - 1) / nthreads;
    const int chunksizeinit = (int)mymin(chunksizeinit_helper, (size_t)MPS1_TBINITSP_COMPLETEREFINE_CHSIZE);

    //cache for the cross-covarinace matrix and related data: 
    float ccm[nEFFDS][XDIM];
    float tfm[nEFFDS];//nEFFDS>nTTranformMatrix
    float ccmLast[nEFFDS];
    float tfmBest[nEFFDS];

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        shared(maxnfragfcts) \
        private(ccm, tfm, ccmLast, tfmBest)
    {
        //initialize best scores
        #pragma omp for collapse(3) schedule(dynamic, chunksizeinit)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nthreads; si++)
                for(int bi = 0; bi < nblocks_x_best; bi++)
                {//threads process blocks of references
                    const int istr0 = bi * XDIM;
                    const int istre = mymin(istr0 + XDIM, (int)ndbCstrs_);
                    const int mloc = ((qi * maxnsteps_ + si) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    #pragma omp simd aligned(wrkmemaux:memalignment)
                    for(int ri = istr0; ri < istre; ri++)
                        wrkmemaux[mloc + tawmvBestScore * ndbCstrs_ + ri/*dbstrndx*/] = 0.0f;
                }
        //implicit barrier here

        #pragma omp for collapse(3) schedule(static, chunksize)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nblocks_y; si++)
                for(int ri = 0; ri < nblocks_x; ri++)
                {//threads process references
                    //check convergence:
                    int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                    tfm[6] = wrkmemaux[mloc0 + ri];//reuse cache
                    if(((int)(tfm[6])) & (CONVERGED_LOWTMSC_bitval))
                        continue;

                    const int sfragfct = si;//fragment factor
                    int sfragndx, sfragpos;//fragment length index and position

                    const int qrylenorg = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                    // const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                    const int dbstrlenorg = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                    const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);

                    enum {qrypos = 0, rfnpos = 0};
                    const int tid = omp_get_thread_num();
                    int qrylen, dbstrlen;

                    //NOTE: #matched positions tawmvNAlnPoss written at sfragfct==0:
                    mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    qrylen = dbstrlen = wrkmemaux[mloc0 + tawmvNAlnPoss * ndbCstrs_ + ri];
                    sfragndx = wrkmemaux[mloc0 + tawmvSubFragNdx * ndbCstrs_ + ri];
                    sfragpos = wrkmemaux[mloc0 + tawmvSubFragPos * ndbCstrs_ + ri];

                    if(sfragndx == 0) sfragndx++;
                    sfragpos += (sfragfct - (maxnfragfcts>>1)) * sfragstep;

                    //out-of-bounds check:
                    const int fraglen = GetFragLength(qrylen, dbstrlen, qrypos, rfnpos, sfragndx);
                    if(fraglen < 1) continue;
                    if(sfragpos < 0 ||
                       qrylen + sfragstep <= qrypos + sfragpos + fraglen ||
                       dbstrlen + sfragstep <= rfnpos + sfragpos + fraglen)
                        continue;

                    //threshold calculated for the original lengths
                    const float d0 = D0FINAL? GetD0fin(qrylenorg, dbstrlenorg): GetD0(qrylenorg, dbstrlenorg);
                    const float d02 = SQRD(d0);
                    const float d82 = GetD82(qrylenorg, dbstrlenorg);
                    float dst32 = CP_LARGEDST;
                    float best = -1.0f;//best score obtained

                    CalcCCMatrices_DPRefined_Complete<nEFFDS,XDIM,memalignment>(
                        qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_, tid/*sfragfctxndx*/, dbstrdst, fraglen,
                        qrylen, dbstrlen,  qrypos + sfragpos, rfnpos + sfragpos,
                        tmpdpalnpossbuffer, ccm);

                    for(int cit = 0; cit < nmaxconvit + 2; cit++)
                    {
                        if(0 < cit) {
                            CalcCCMatrices_DPRefinedExtended_Complete<nEFFDS,XDIM,memalignment>(
                                (cit < 2)? READCNST_CALC: READCNST_CALC2,
                                qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_, tid/*sfragfctxndx*/, dbstrdst,
                                qrylen, dbstrlen, qrypos, rfnpos,  d0, dst32,
                                tmpdpdiagbuffers, tmpdpalnpossbuffer, ccm);

                            CheckConvergenceRefined_Complete<nEFFDS,XDIM>(ccm, ccmLast);
                            if(ccmLast[0]) break;//converged
                        }

                        if(ccm[twmvNalnposs][0] < 1.0f) break;

                        //copy sums to tfm and ccmLast:
                        #pragma omp simd
                        for(int f = 0; f < nEFFDS; f++) tfm[f] = ccmLast[f] = ccm[f][0];

                        CalcTfmMatrices_Complete<TFM_DINV>(qrylenorg, dbstrlenorg, tfm);

                        CalcScoresUnrl_DPRefined_Complete<XDIM,memalignment,CHCKDST>(
                            (cit < 1)? READCNST_CALC: READCNST_CALC2,
                            qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_, tid/*sfragfctxndx*/, dbstrdst,
                            qrylen, dbstrlen, qrypos, rfnpos,  d0, d02, d82,
                            tmpdpdiagbuffers, tmpdpalnpossbuffer,  tfm, ccm[0]/*scv*/, ccm/*dstv*/);

                        //distance threshold for at least three aligned pairs:
                        dst32 = ccm[0][1];
                        if(best < ccm[0][0]) {
                            best = ccm[0][0];//score written at [0,0]
                            #pragma omp simd
                            for(int f = 0; f < nTTranformMatrix; f++) tfmBest[f] = tfm[f];
                        }
                    }//for(;cit;)

                    //NOTE: CONDITIONAL==true because effnsteps(==nthreads)<<maxnsteps
                    SaveBestScoreAndTM_Complete<false/*WRITEFRAGINFO*/,true/*CONDITIONAL*/>(
                        best,  qi/*qryndx*/, ri/*dbstrndx*/, ndbCstrs_, 
                        maxnsteps_, tid/*sfragfctxndx*/, sfragndx, sfragpos,
                        tfmBest, wrkmemtmibest, wrkmemaux);
                }//omp for
        //implicit barrier here

        //find the max score among fragment variants
        #pragma omp for collapse(2)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int bi = 0; bi < nblocks_x_best; bi++)
            {//threads process blocks of references
                SaveBestScoreAndTMAmongBests
                    <XDIM,memalignment,
                     false/*WRITEFRAGINFO*/,
                     true/*GRANDUPDATE*/,
                     false/*FORCEWRITEFRAGINFO*/,
                     SECONDARYUPDATE_NOUPDATE>(
                        qi/*qryndx*/, bi/*rfnblkndx*/, ndbCstrs_,
                        maxnsteps_, nthreads/*effnsteps*/,
                        ccm, wrkmemtmibest, tfmmem, wrkmemaux, NULL/*wrkmemtm*/);
            }
    }
}

// =========================================================================
// Instantiations
// 
#define INSTANTIATE_MpStageFin_FinalFragmentBasedDPAlignmentRefinementPhase2Kernel(tpD0FINAL,tpCHCKDST,tpTFM_DINV) \
    template void MpStageFin::FinalFragmentBasedDPAlignmentRefinementPhase2Kernel<tpD0FINAL,tpCHCKDST,tpTFM_DINV>( \
        const int nmaxconvit, \
        const char* const * const __RESTRICT__ querypmbeg, \
        const char* const * const __RESTRICT__ bdbCpmbeg, \
        const float* const __RESTRICT__ tmpdpalnpossbuffer, \
        float* const __RESTRICT__ tmpdpdiagbuffers, \
        float* const __RESTRICT__ wrkmemtmibest, \
        float* const __RESTRICT__ wrkmemaux, \
        float* const __RESTRICT__ tfmmem);

INSTANTIATE_MpStageFin_FinalFragmentBasedDPAlignmentRefinementPhase2Kernel(false,CHCKDST_CHECK,false);
INSTANTIATE_MpStageFin_FinalFragmentBasedDPAlignmentRefinementPhase2Kernel(false,CHCKDST_CHECK,true);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// FinalFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel: phase 2 to
// perform finer-scale refinement of the the best superposition obtained 
// within the single kernel's actions;
// this version performs full search of maxnfragfcts positions from the 
// identified one in phase 1 for each fragment length;
// D0FINAL, template flag of using the final threshold value for D0;
// CHCKDST, template parameter to request accumulating scores within the 
// given threshold distance only;
// TFM_DINV, use doubly inverted transformation matrices under suitable conditions;
// nmaxconvit, maximum number of superposition iterations;
// maxnfragfcts, max number of fragment position factors around an identified position;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// tmpdpdiagbuffers, temporary diagonal buffers for positional scores/dsts;
// wrkmemtmibest, working memory for best-performing transformation matrices;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// tfmmem, memory region for transformation matrices;
// 
template<bool D0FINAL, int CHCKDST, bool TFM_DINV>
void MpStageFin::FinalFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel(
    const int nmaxconvit,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* const __RESTRICT__ tmpdpalnpossbuffer,
    float* const __RESTRICT__ tmpdpdiagbuffers,
    float* const __RESTRICT__ wrkmemtmibest,
    float* const __RESTRICT__ wrkmemaux,
    float* const __RESTRICT__ tfmmem)
{
    enum{
        //effective number of fields:
        nEFFDS = twmvEndOfCCDataExt,
        XDIM = MPS1_TBINITSP_COMPLETEREFINE_XDIM
    };

    MYMSG("MpStageFin::FinalFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel", 4);
    static const std::string preamb = "MpStageFin::FinalFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    // static const int refinement = CLOptions::GetC_REFINEMENT();
    constexpr int memalignment = CuMemoryBase::GetMinMemAlignment();
    //minimum length among largest
    int minlenmax = myhdmin(qystr1len_, dbstr1len_);
    int maxalnmax = minlenmax;
    //maximum number of subdivisions
    const int nmaxsubfrags = FRAGREF_NMAXSUBFRAGS;
    //NOTE: step for the SECOND phase to final (finer-scale) refinement;
    constexpr int sfragstep = FRAGREF_SFRAGSTEP_mini;

    //max #fragment position factors around an identified position
    //**********************************************************************
    //NOTE: multiply maxalnmax by 2 since sub-optimal (first-phase) alignment
    //NOTE: position can be identified, e.g., at the end of alignment!
    //**********************************************************************
    int maxnfragfcts = myhdmin(2 * maxalnmax, CUSFN_TBSP_FIN_REFINEMENT_MAX_NPOSITIONS);
    maxnfragfcts = (maxnfragfcts + sfragstep-1) / sfragstep;
    int nlocsteps2 = maxnfragfcts;//total number for ONE fragment length
    // if(refinement == CLOptions::csrFullASearch)
    nlocsteps2 *= nmaxsubfrags;//total number across all fragment lengths

    if(nlocsteps2 < 1 || (int)maxnsteps_ < nlocsteps2)
        throw MYRUNTIME_ERROR(preamb +
        "Invalid number of superposition tests: "+std::to_string(nlocsteps2));

    //execution configuration: process and refine multiple
    //query-reference alignment variants:
    const int nblocks_x = ndbCstrs_;
    const int nblocks_x_best = (ndbCstrs_ + XDIM - 1) / XDIM;
    const int nblocks_y = nlocsteps2;
    const int nblocks_z = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_z * (size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPS1_TBINITSP_COMPLETEREFINE_CHSIZE);

    size_t chunksizeinit_helper = 
        ((size_t)nblocks_z * (size_t)nthreads * (size_t)nblocks_x_best + (size_t)nthreads - 1) / nthreads;
    const int chunksizeinit = (int)mymin(chunksizeinit_helper, (size_t)MPS1_TBINITSP_COMPLETEREFINE_CHSIZE);

    //cache for the cross-covarinace matrix and related data: 
    float ccm[nEFFDS][XDIM];
    float tfm[nEFFDS];//nEFFDS>nTTranformMatrix
    float ccmLast[nEFFDS];
    float tfmBest[nEFFDS];

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        shared(maxnfragfcts) \
        private(ccm, tfm, ccmLast, tfmBest)
    {
        //initialize best scores
        #pragma omp for collapse(3) schedule(dynamic, chunksizeinit)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nthreads; si++)
                for(int bi = 0; bi < nblocks_x_best; bi++)
                {//threads process blocks of references
                    const int istr0 = bi * XDIM;
                    const int istre = mymin(istr0 + XDIM, (int)ndbCstrs_);
                    const int mloc = ((qi * maxnsteps_ + si) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    #pragma omp simd aligned(wrkmemaux:memalignment)
                    for(int ri = istr0; ri < istre; ri++)
                        wrkmemaux[mloc + tawmvBestScore * ndbCstrs_ + ri/*dbstrndx*/] = 0.0f;
                }
        //implicit barrier here

        #pragma omp for collapse(3) schedule(static, chunksize)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nblocks_y; si++)
                for(int ri = 0; ri < nblocks_x; ri++)
                {//threads process references
                    //check convergence:
                    int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                    tfm[6] = wrkmemaux[mloc0 + ri];//reuse cache
                    if(((int)(tfm[6])) & (CONVERGED_LOWTMSC_bitval))
                        continue;

                    const int sfragfct = si / nmaxsubfrags;//fragment factor
                    const int sfragndx = si - sfragfct * nmaxsubfrags;//fragment length index
                    int sfragpos;//fragment position

                    const int qrylenorg = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                    // const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                    const int dbstrlenorg = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                    const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);

                    enum {qrypos = 0, rfnpos = 0};
                    const int tid = omp_get_thread_num();
                    int qrylen, dbstrlen;

                    //NOTE: #matched positions tawmvNAlnPoss written at sfragfct==0:
                    mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    qrylen = dbstrlen = wrkmemaux[mloc0 + tawmvNAlnPoss * ndbCstrs_ + ri];
                    sfragpos = wrkmemaux[mloc0 + tawmvSubFragPos * ndbCstrs_ + ri];
                    sfragpos += (sfragfct - (maxnfragfcts>>1)) * sfragstep;

                    //out-of-bounds check:
                    const int fraglen = GetFragLength(qrylen, dbstrlen, qrypos, rfnpos, sfragndx);
                    if(fraglen < 1) continue;
                    if(sfragpos < 0 ||
                       qrylen + sfragstep <= qrypos + sfragpos + fraglen ||
                       dbstrlen + sfragstep <= rfnpos + sfragpos + fraglen)
                        continue;

                    //threshold calculated for the original lengths
                    const float d0 = D0FINAL? GetD0fin(qrylenorg, dbstrlenorg): GetD0(qrylenorg, dbstrlenorg);
                    const float d02 = SQRD(d0);
                    const float d82 = GetD82(qrylenorg, dbstrlenorg);
                    float dst32 = CP_LARGEDST;
                    float best = -1.0f;//best score obtained

                    CalcCCMatrices_DPRefined_Complete<nEFFDS,XDIM,memalignment>(
                        qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_, tid/*sfragfctxndx*/, dbstrdst, fraglen,
                        qrylen, dbstrlen,  qrypos + sfragpos, rfnpos + sfragpos,
                        tmpdpalnpossbuffer, ccm);

                    for(int cit = 0; cit < nmaxconvit + 2; cit++)
                    {
                        if(0 < cit) {
                            CalcCCMatrices_DPRefinedExtended_Complete<nEFFDS,XDIM,memalignment>(
                                (cit < 2)? READCNST_CALC: READCNST_CALC2,
                                qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_, tid/*sfragfctxndx*/, dbstrdst,
                                qrylen, dbstrlen, qrypos, rfnpos,  d0, dst32,
                                tmpdpdiagbuffers, tmpdpalnpossbuffer, ccm);

                            CheckConvergenceRefined_Complete<nEFFDS,XDIM>(ccm, ccmLast);
                            if(ccmLast[0]) break;//converged
                        }

                        if(ccm[twmvNalnposs][0] < 1.0f) break;

                        //copy sums to tfm and ccmLast:
                        #pragma omp simd
                        for(int f = 0; f < nEFFDS; f++) tfm[f] = ccmLast[f] = ccm[f][0];

                        CalcTfmMatrices_Complete<TFM_DINV>(qrylenorg, dbstrlenorg, tfm);

                        CalcScoresUnrl_DPRefined_Complete<XDIM,memalignment,CHCKDST>(
                            (cit < 1)? READCNST_CALC: READCNST_CALC2,
                            qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_, tid/*sfragfctxndx*/, dbstrdst,
                            qrylen, dbstrlen, qrypos, rfnpos,  d0, d02, d82,
                            tmpdpdiagbuffers, tmpdpalnpossbuffer,  tfm, ccm[0]/*scv*/, ccm/*dstv*/);

                        //distance threshold for at least three aligned pairs:
                        dst32 = ccm[0][1];
                        if(best < ccm[0][0]) {
                            best = ccm[0][0];//score written at [0,0]
                            #pragma omp simd
                            for(int f = 0; f < nTTranformMatrix; f++) tfmBest[f] = tfm[f];
                        }
                    }//for(;cit;)

                    //NOTE: CONDITIONAL==true because effnsteps(==nthreads)<<maxnsteps
                    SaveBestScoreAndTM_Complete<false/*WRITEFRAGINFO*/,true/*CONDITIONAL*/>(
                        best,  qi/*qryndx*/, ri/*dbstrndx*/, ndbCstrs_, 
                        maxnsteps_, tid/*sfragfctxndx*/, sfragndx, sfragpos,
                        tfmBest, wrkmemtmibest, wrkmemaux);
                }//omp for
        //implicit barrier here

        //find the max score among fragment variants
        #pragma omp for collapse(2)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int bi = 0; bi < nblocks_x_best; bi++)
            {//threads process blocks of references
                SaveBestScoreAndTMAmongBests
                    <XDIM,memalignment,
                     false/*WRITEFRAGINFO*/,
                     true/*GRANDUPDATE*/,
                     false/*FORCEWRITEFRAGINFO*/,
                     SECONDARYUPDATE_NOUPDATE>(
                        qi/*qryndx*/, bi/*rfnblkndx*/, ndbCstrs_,
                        maxnsteps_, nthreads/*effnsteps*/,
                        ccm, wrkmemtmibest, tfmmem, wrkmemaux, NULL/*wrkmemtm*/);
            }
    }
}

// =========================================================================
// Instantiations
// 
#define INSTANTIATE_MpStageFin_FinalFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel(tpD0FINAL,tpCHCKDST,tpTFM_DINV) \
    template void MpStageFin::FinalFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel<tpD0FINAL,tpCHCKDST,tpTFM_DINV>( \
        const int nmaxconvit, \
        const char* const * const __RESTRICT__ querypmbeg, \
        const char* const * const __RESTRICT__ bdbCpmbeg, \
        const float* const __RESTRICT__ tmpdpalnpossbuffer, \
        float* const __RESTRICT__ tmpdpdiagbuffers, \
        float* const __RESTRICT__ wrkmemtmibest, \
        float* const __RESTRICT__ wrkmemaux, \
        float* const __RESTRICT__ tfmmem);

INSTANTIATE_MpStageFin_FinalFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel(false,CHCKDST_CHECK,false);
INSTANTIATE_MpStageFin_FinalFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel(false,CHCKDST_CHECK,true);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// =========================================================================
// =========================================================================
// -------------------------------------------------------------------------
// ProductionFragmentBasedDPAlignmentRefinementPhase1Kernel: phase 1 to perform 
// production-version tuning of the the best superposition obtained 
// within the single kernel's actions; write the final superposition scores;
// TFM_DINV, use doubly inverted transformation matrices under suitable conditions;
// nmaxconvit, maximum number of superposition iterations;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// tmpdpdiagbuffers, temporary diagonal buffers for positional scores/dsts;
// wrkmemtmibest, working memory for best-performing transformation matrices;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// alndatamem, memory for full alignment information;
// 
template<bool TFM_DINV>
void MpStageFin::ProductionFragmentBasedDPAlignmentRefinementPhase1Kernel(
    const int nmaxconvit,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* const __RESTRICT__ tmpdpalnpossbuffer,
    float* const __RESTRICT__ tmpdpdiagbuffers,
    float* const __RESTRICT__ wrkmemtmibest,
    float* const __RESTRICT__ wrkmemaux,
    float* const __RESTRICT__ alndatamem,
    float* const __RESTRICT__ tfmmem)
{
    enum{
        //effective number of fields:
        nEFFDS = twmvEndOfCCDataExt,
        SMIDIM = twmvEndOfCCDataExtPlus,
        XDIM = MPS1_TBINITSP_COMPLETEREFINE_XDIM
    };

    MYMSG("MpStageFin::ProductionFragmentBasedDPAlignmentRefinementPhase1Kernel", 4);
    static const std::string preamb = "MpStageFin::ProductionFragmentBasedDPAlignmentRefinementPhase1Kernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = mycemin((size_t)PMBSdatalignment, CuMemoryBase::GetMinMemAlignment());
    //minimum length among largest
    int minlenmax = myhdmin(qystr1len_, dbstr1len_);
    int maxalnmax = minlenmax;
    //maximum number of subdivisions
    const int nmaxsubfrags = FRAGREF_NMAXSUBFRAGS;
    // sfragstep, step to traverse subfragments;
    constexpr int sfragstep = FRAGREF_SFRAGSTEP;

    int nlocsteps = 0;
    nlocsteps = GetMaxNFragSteps(maxalnmax, sfragstep, minfraglen_);
    nlocsteps *= nmaxsubfrags;//total number across all fragment lengths

    if(nlocsteps < 1 || (int)maxnsteps_ < nlocsteps)
        throw MYRUNTIME_ERROR(preamb +
        "Invalid number of superposition tests: "+std::to_string(nlocsteps));

    //execution configuration: process and refine multiple
    //query-reference alignment variants:
    const int nblocks_x = ndbCstrs_;
    const int nblocks_x_best = (ndbCstrs_ + XDIM - 1) / XDIM;
    const int nblocks_y = nlocsteps;
    const int nblocks_z = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_z * (size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPS1_TBINITSP_COMPLETEREFINE_CHSIZE);

    size_t chunksizeinit_helper = 
        ((size_t)nblocks_z * (size_t)nthreads * (size_t)nblocks_x_best + (size_t)nthreads - 1) / nthreads;
    const int chunksizeinit = (int)mymin(chunksizeinit_helper, (size_t)MPS1_TBINITSP_COMPLETEREFINE_CHSIZE);

    //cache for the cross-covarinace matrix and related data: 
    float ccm[SMIDIM][XDIM];
    float tfm[nEFFDS];//nEFFDS>nTTranformMatrix
    float ccmLast[SMIDIM];
    float tfmBest[nEFFDS];
    float tmp[6];

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        private(ccm, tfm, ccmLast, tfmBest, tmp)
    {
        //initialize best scores
        #pragma omp for collapse(3) schedule(dynamic, chunksizeinit)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nthreads; si++)
                for(int bi = 0; bi < nblocks_x_best; bi++)
                {//threads process blocks of references
                    const int istr0 = bi * XDIM;
                    const int istre = mymin(istr0 + XDIM, (int)ndbCstrs_);
                    const int mloc = ((qi * maxnsteps_ + si) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    #pragma omp simd aligned(wrkmemaux:memalignment)
                    for(int ri = istr0; ri < istre; ri++)
                        wrkmemaux[mloc + tawmvBestScore * ndbCstrs_ + ri/*dbstrndx*/] = 0.0f;
                }
        //implicit barrier here

        #pragma omp for collapse(3) schedule(static, chunksize)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nblocks_y; si++)
                for(int ri = 0; ri < nblocks_x; ri++)
                {//threads process references
                    //check convergence:
                    int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                    tfm[6] = wrkmemaux[mloc0 + ri];//reuse cache
                    if(((int)(tfm[6])) & (CONVERGED_LOWTMSC_bitval))
                        continue;

                    const int sfragfct = si / nmaxsubfrags;//fragment factor
                    const int sfragndx = si - sfragfct * nmaxsubfrags;//fragment length index

                    const int qrylenorg = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                    // const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                    const int dbstrlenorg = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                    const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);

                    enum {qrypos = 0, rfnpos = 0};
                    const int sfragpos = sfragfct * sfragstep;
                    const int tid = omp_get_thread_num();
                    int qrylen, dbstrlen;

                    //NOTE: #matched positions tawmvNAlnPoss written at sfragfct==0:
                    mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    qrylen = dbstrlen = wrkmemaux[mloc0 + tawmvNAlnPoss * ndbCstrs_ + ri];

                    //out-of-bounds check:
                    const int fraglen = GetFragLength(qrylen, dbstrlen, qrypos, rfnpos, sfragndx);
                    if(fraglen < 1) continue;
                    if(qrylen + sfragstep <= qrypos + sfragpos + fraglen ||
                       dbstrlen + sfragstep <= rfnpos + sfragpos + fraglen)
                        continue;

                    //threshold calculated for the original lengths
                    const float d0 = GetD0fin(qrylenorg, dbstrlenorg);
                    const float d02 = SQRD(d0);
                    const float d82 = GetD82(qrylenorg, dbstrlenorg);
                    float dst32 = CP_LARGEDST;
                    float best = -1.0f;//best score obtained

                    //calculate rmsd when the fragment being processed represents full alignment
                    if(si == 0) {//also implies sfragndx == 0
                        CalcExtCCMatrices_DPRefined_Complete<SMIDIM,XDIM,memalignment>(
                            qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_, tid/*sfragfctxndx*/, dbstrdst, fraglen,
                            qrylen, dbstrlen,  qrypos + sfragpos, rfnpos + sfragpos,
                            tmpdpalnpossbuffer, ccm);
                        //ccm leaves unchanged for further processing
                        #pragma omp simd
                        for(int f = 0; f < SMIDIM; f++) ccmLast[f] = ccm[f][0];
                        float rmsd = CalcRMSD_Complete(ccmLast, tmp);
                        //write rmsd to memory
                        mloc0 = (qi * ndbCstrs_ + ri) * nTDP2OutputAlnData;
                        alndatamem[mloc0 + dp2oadRMSD] = rmsd;//WRITE
                    }
                    else
                        CalcCCMatrices_DPRefined_Complete<nEFFDS,XDIM,memalignment>(
                            qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_, tid/*sfragfctxndx*/, dbstrdst, fraglen,
                            qrylen, dbstrlen,  qrypos + sfragpos, rfnpos + sfragpos,
                            tmpdpalnpossbuffer, ccm);

                    for(int cit = 0; cit < nmaxconvit + 2; cit++)
                    {
                        if(0 < cit) {
                            CalcCCMatrices_DPRefinedExtended_Complete<nEFFDS,XDIM,memalignment>(
                                (cit < 2)? READCNST_CALC: READCNST_CALC2,
                                qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_, tid/*sfragfctxndx*/, dbstrdst,
                                qrylen, dbstrlen, qrypos, rfnpos,  d0, dst32,
                                tmpdpdiagbuffers, tmpdpalnpossbuffer, ccm);

                            CheckConvergenceRefined_Complete<nEFFDS,XDIM>(ccm, ccmLast);
                            if(ccmLast[0]) break;//converged
                        }

                        if(ccm[twmvNalnposs][0] < 1.0f) break;

                        //copy sums to tfm and ccmLast:
                        #pragma omp simd
                        for(int f = 0; f < nEFFDS; f++) tfm[f] = ccmLast[f] = ccm[f][0];

                        CalcTfmMatrices_Complete<TFM_DINV>(qrylenorg, dbstrlenorg, tfm);

                        CalcScoresUnrl_DPRefined_Complete<XDIM,memalignment,CHCKDST_NOCHECK>(
                            (cit < 1)? READCNST_CALC: READCNST_CALC2,
                            qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_, tid/*sfragfctxndx*/, dbstrdst,
                            qrylen, dbstrlen, qrypos, rfnpos,  d0, d02, d82,
                            tmpdpdiagbuffers, tmpdpalnpossbuffer,  tfm, ccm[0]/*scv*/, ccm/*dstv*/);

                        //distance threshold for at least three aligned pairs:
                        dst32 = ccm[0][1];
                        if(best < ccm[0][0]) {
                            best = ccm[0][0];//score written at [0,0]
                            #pragma omp simd
                            for(int f = 0; f < nTTranformMatrix; f++) tfmBest[f] = tfm[f];
                        }
                    }//for(;cit;)

                    //calculate the score for the larger structure of the two:
                    //threshold calculated for the greater length
                    const int greaterlen = mymax(qrylenorg, dbstrlenorg);
                    const float g0 = GetD0fin(greaterlen, greaterlen);
                    const float g02 = SQRD(g0);
                    float gbest = best;//score calculated for the other structure

                    if(qrylenorg != dbstrlenorg) {
                        CalcScoresUnrl_DPRefined_Complete<XDIM,memalignment,CHCKDST_NOCHECK>(
                            READCNST_CALC2,
                            qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_, tid/*sfragfctxndx*/, dbstrdst,
                            qrylen, dbstrlen, qrypos, rfnpos,  g0, g02, d82,
                            tmpdpdiagbuffers, tmpdpalnpossbuffer,  tfmBest, ccm[0]/*scv*/, ccm/*dstv*/);
                        gbest = ccm[0][0];//score
                    }

                    //NOTE: CONDITIONAL==true because effnsteps(==nthreads)<<maxnsteps
                    SaveBestQRScoresAndTM_Complete<true/*WRITEFRAGINFO*/,true/*CONDITIONAL*/>(
                        best, gbest,  qi/*qryndx*/, ri/*dbstrndx*/, ndbCstrs_, 
                        maxnsteps_, tid/*sfragfctxndx*/, sfragndx, sfragpos,
                        tfmBest, wrkmemtmibest, wrkmemaux);
                }//omp for
        //implicit barrier here

        //find the max score among fragment variants
        #pragma omp for collapse(2)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int bi = 0; bi < nblocks_x_best; bi++)
            {//threads process blocks of references
                ProductionSaveBestScoresAndTMAmongBests
                    <XDIM,memalignment,true/*WRITEFRAGINFO*/,false/*CONDITIONAL*/>(
                        qi/*qryndx*/, bi/*rfnblkndx*/, ndbCstrs_,
                        maxnsteps_, nthreads/*effnsteps*/,
                        ccm, querypmbeg, bdbCpmbeg,
                        wrkmemtmibest, wrkmemaux, alndatamem, tfmmem);
            }
    }
}

// =========================================================================
// Instantiations
// 
#define INSTANTIATE_MpStageFin_ProductionFragmentBasedDPAlignmentRefinementPhase1Kernel(tpTFM_DINV) \
    template void MpStageFin::ProductionFragmentBasedDPAlignmentRefinementPhase1Kernel<tpTFM_DINV>( \
        const int nmaxconvit, \
        const char* const * const __RESTRICT__ querypmbeg, \
        const char* const * const __RESTRICT__ bdbCpmbeg, \
        const float* const __RESTRICT__ tmpdpalnpossbuffer, \
        float* const __RESTRICT__ tmpdpdiagbuffers, \
        float* const __RESTRICT__ wrkmemtmibest, \
        float* const __RESTRICT__ wrkmemaux, \
        float* const __RESTRICT__ alndatamem, \
        float* const __RESTRICT__ tfmmem);

INSTANTIATE_MpStageFin_ProductionFragmentBasedDPAlignmentRefinementPhase1Kernel(false);
INSTANTIATE_MpStageFin_ProductionFragmentBasedDPAlignmentRefinementPhase1Kernel(true);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// =========================================================================
// ProductionRefinementPhase2InnerLoop: inner loop for 
// ProductionFragmentBasedDPAlignmentRefinementPhase2 variants;
// sfragfctxndx, fragment factor x fragment length index;
// qryndx, query serial number;
// nmaxconvit, maximum number of superposition iterations;
// ndbCposs, total number of reference positions in the chunk;
// dbxpad, #pad positions along the dimension of reference structures;
// maxnsteps, total number of steps that should be performed for each reference structure;
// qrylenorg, dbstrlenorg, original query and reference lengths;
// qrylen, dbstrlen, pseudo query and reference length, #matched positions;
// dbstrdst, distance in positions to the beginnings of the reference structures
// qrypos, rfnpos, query and reference starting positions in alignment (0);
// sfragpos, fraglen, fragment position and length;
// d0, d02, d82, distance thresholds;
// best, best score so far;
// ccm, ccmLast, tfm, tfmBest, working cache;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// tmpdpdiagbuffers, temporary diagonal buffers for positional scores/dsts;
//
template<int nEFFDS, int XDIM, int DATALN, bool TFM_DINV>
inline
void MpStageFin::ProductionRefinementPhase2InnerLoop(
    const int sfragfctxndx,
    const int qryndx,
    const int nmaxconvit,
    const int ndbCposs,
    const int dbxpad,
    const int maxnsteps,
    const int qrylenorg, const int dbstrlenorg,
    const int qrylen, const int dbstrlen,
    const int dbstrdst,
    const int qrypos, const int rfnpos,
    const int sfragpos, const int fraglen,
    const float d0, const float d02, const float d82,
    float& best,
    float (* __RESTRICT__ ccm)[XDIM],
    float* __RESTRICT__ ccmLast,
    float* __RESTRICT__ tfm,
    float* __RESTRICT__ tfmBest,
    const float* const __RESTRICT__ tmpdpalnpossbuffer,
    float* const __RESTRICT__ tmpdpdiagbuffers)
{
    float dst32 = CP_LARGEDST;

    CalcCCMatrices_DPRefined_Complete<nEFFDS,XDIM,DATALN>(
        qryndx, ndbCposs, dbxpad, maxnsteps, sfragfctxndx, dbstrdst, fraglen,
        qrylen, dbstrlen,  qrypos + sfragpos, rfnpos + sfragpos,
        tmpdpalnpossbuffer, ccm);

    for(int cit = 0; cit < nmaxconvit + 2; cit++)
    {
        if(0 < cit) {
            CalcCCMatrices_DPRefinedExtended_Complete<nEFFDS,XDIM,DATALN>(
                (cit < 2)? READCNST_CALC: READCNST_CALC2,
                qryndx, ndbCposs, dbxpad, maxnsteps, sfragfctxndx, dbstrdst,
                qrylen, dbstrlen, qrypos, rfnpos,  d0, dst32,
                tmpdpdiagbuffers, tmpdpalnpossbuffer, ccm);

            CheckConvergenceRefined_Complete<nEFFDS,XDIM>(ccm, ccmLast);
            if(ccmLast[0]) break;//converged
        }

        if(ccm[twmvNalnposs][0] < 1.0f) break;

        //copy sums to tfm and ccmLast:
        #pragma omp simd
        for(int f = 0; f < nEFFDS; f++) tfm[f] = ccmLast[f] = ccm[f][0];

        CalcTfmMatrices_Complete<TFM_DINV>(qrylenorg, dbstrlenorg, tfm);

        CalcScoresUnrl_DPRefined_Complete<XDIM,DATALN,CHCKDST_NOCHECK>(
            (cit < 1)? READCNST_CALC: READCNST_CALC2,
            qryndx, ndbCposs, dbxpad, maxnsteps, sfragfctxndx, dbstrdst,
            qrylen, dbstrlen, qrypos, rfnpos,  d0, d02, d82,
            tmpdpdiagbuffers, tmpdpalnpossbuffer,  tfm, ccm[0]/*scv*/, ccm/*dstv*/);

        //distance threshold for at least three aligned pairs:
        dst32 = ccm[0][1];
        if(best < ccm[0][0]) {
            best = ccm[0][0];//score written at [0,0]
            #pragma omp simd
            for(int f = 0; f < nTTranformMatrix; f++) tfmBest[f] = tfm[f];
        }
    }
}

// =========================================================================



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel: phase 2 to
// perform production-version tuning of the the best superposition obtained 
// within the single kernel's actions; write the final superposition scores;
// this version performs full search of maxnfragfcts positions from the 
// identified one in phase 1 for each fragment length;
// TFM_DINV, use doubly inverted transformation matrices under suitable conditions;
// nmaxconvit, maximum number of superposition iterations;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// tmpdpdiagbuffers, temporary diagonal buffers for positional scores/dsts;
// wrkmemtmibest, working memory for best-performing transformation matrices;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// alndatamem, memory for full alignment information;
// tfmmem, memory for production transformation matrices;
// 
template<bool TFM_DINV>
void MpStageFin::ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel(
    const int nmaxconvit,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* const __RESTRICT__ tmpdpalnpossbuffer,
    float* const __RESTRICT__ tmpdpdiagbuffers,
    float* const __RESTRICT__ wrkmemtmibest,
    float* const __RESTRICT__ wrkmemaux,
    float* const __RESTRICT__ alndatamem,
    float* const __RESTRICT__ tfmmem)
{
    enum{
        //effective number of fields:
        nEFFDS = twmvEndOfCCDataExt,
        XDIM = MPS1_TBINITSP_COMPLETEREFINE_XDIM
    };

    MYMSG("MpStageFin::ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel", 4);
    static const std::string preamb =
    "MpStageFin::ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = mycemin((size_t)PMBSdatalignment, CuMemoryBase::GetMinMemAlignment());
    //minimum length among largest
    int minlenmax = myhdmin(qystr1len_, dbstr1len_);
    int maxalnmax = minlenmax;
    //maximum number of subdivisions
    const int nmaxsubfrags = FRAGREF_NMAXSUBFRAGS;
    //NOTE: step for the SECOND phase to final (finer-scale) refinement;
    constexpr int sfragstep = FRAGREF_SFRAGSTEP_mini;

    //max #fragment position factors around an identified position
    //**********************************************************************
    //NOTE: multiply maxalnmax by 2 since sub-optimal (first-phase) alignment
    //NOTE: position can be identified at the end of alignment!
    //**********************************************************************
    int maxnfragfcts = myhdmin(2 * maxalnmax, CUSFN_TBSP_FIN_REFINEMENT_MAX_NPOSITIONS);
    maxnfragfcts = (maxnfragfcts + sfragstep-1) / sfragstep;
    int nlocsteps2 = maxnfragfcts * nmaxsubfrags;//total number across all fragment lengths

    if(nlocsteps2 < 1 || (int)maxnsteps_ < nlocsteps2)
        throw MYRUNTIME_ERROR(preamb +
        "Invalid number of superposition tests: "+std::to_string(nlocsteps2));

    //execution configuration: process and refine multiple
    //query-reference alignment variants:
    const int nblocks_x = ndbCstrs_;
    const int nblocks_x_best = (ndbCstrs_ + XDIM - 1) / XDIM;
    const int nblocks_y = nlocsteps2;
    const int nblocks_z = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_z * (size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPS1_TBINITSP_COMPLETEREFINE_CHSIZE);

    size_t chunksizeinit_helper = 
        ((size_t)nblocks_z * (size_t)nthreads * (size_t)nblocks_x_best + (size_t)nthreads - 1) / nthreads;
    const int chunksizeinit = (int)mymin(chunksizeinit_helper, (size_t)MPS1_TBINITSP_COMPLETEREFINE_CHSIZE);

    //cache for the cross-covarinace matrix and related data: 
    float ccm[nEFFDS][XDIM];
    float tfm[nEFFDS];//nEFFDS>nTTranformMatrix
    float ccmLast[nEFFDS];
    float tfmBest[nEFFDS];

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        shared(maxnfragfcts) \
        private(ccm, tfm, ccmLast, tfmBest)
    {
        //initialize best scores
        #pragma omp for collapse(3) schedule(dynamic, chunksizeinit)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nthreads; si++)
                for(int bi = 0; bi < nblocks_x_best; bi++)
                {//threads process blocks of references
                    const int istr0 = bi * XDIM;
                    const int istre = mymin(istr0 + XDIM, (int)ndbCstrs_);
                    const int mloc = ((qi * maxnsteps_ + si) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    #pragma omp simd aligned(wrkmemaux:memalignment)
                    for(int ri = istr0; ri < istre; ri++)
                        wrkmemaux[mloc + tawmvBestScore * ndbCstrs_ + ri/*dbstrndx*/] = 0.0f;
                }
        //implicit barrier here

        #pragma omp for collapse(3) schedule(static, chunksize)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nblocks_y; si++)
                for(int ri = 0; ri < nblocks_x; ri++)
                {//threads process references
                    //check convergence:
                    int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                    tfm[6] = wrkmemaux[mloc0 + ri];//reuse cache
                    if(((int)(tfm[6])) & (CONVERGED_LOWTMSC_bitval))
                        continue;

                    const int sfragfct = si / nmaxsubfrags;//fragment factor
                    const int sfragndx = si - sfragfct * nmaxsubfrags;//fragment length index

                    const int qrylenorg = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                    // const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                    const int dbstrlenorg = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                    const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);

                    enum {qrypos = 0, rfnpos = 0};
                    const int tid = omp_get_thread_num();
                    int qrylen, dbstrlen, sfragpos;

                    //NOTE: #matched positions tawmvNAlnPoss written at sfragfct==0:
                    mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    qrylen = dbstrlen = wrkmemaux[mloc0 + tawmvNAlnPoss * ndbCstrs_ + ri];
                    sfragpos = wrkmemaux[mloc0 + tawmvSubFragPos * ndbCstrs_ + ri];
                    sfragpos += (sfragfct - (maxnfragfcts>>1)) * sfragstep;

                    //out-of-bounds check:
                    const int fraglen = GetFragLength(qrylen, dbstrlen, qrypos, rfnpos, sfragndx);
                    if(fraglen < 1) continue;
                    if(qrylen + sfragstep <= qrypos + sfragpos + fraglen ||
                       dbstrlen + sfragstep <= rfnpos + sfragpos + fraglen)
                        continue;

                    //threshold calculated for the original lengths
                    const float d0 = GetD0fin(qrylenorg, dbstrlenorg);
                    const float d02 = SQRD(d0);
                    const float d82 = GetD82(qrylenorg, dbstrlenorg);
                    float best = -1.0f;//best score obtained

                    ProductionRefinementPhase2InnerLoop<nEFFDS,XDIM,memalignment,TFM_DINV>(
                        tid/*sfragfctxndx*/, qi/*qryndx*/,
                        nmaxconvit, ndbCposs_, dbxpad_, maxnsteps_,
                        qrylenorg, dbstrlenorg, qrylen, dbstrlen, dbstrdst,
                        qrypos, rfnpos, sfragpos, fraglen,
                        d0, d02, d82,  best/**/,
                        ccm, ccmLast, tfm, tfmBest,
                        tmpdpalnpossbuffer, tmpdpdiagbuffers);

                    //calculate the score for the larger structure of the two:
                    //threshold calculated for the greater length
                    const int greaterlen = mymax(qrylenorg, dbstrlenorg);
                    const float g0 = GetD0fin(greaterlen, greaterlen);
                    const float g02 = SQRD(g0);
                    float gbest = best;//score calculated for the other structure

                    if(qrylenorg != dbstrlenorg) {
                        CalcScoresUnrl_DPRefined_Complete<XDIM,memalignment,CHCKDST_NOCHECK>(
                            READCNST_CALC2,
                            qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_, tid/*sfragfctxndx*/, dbstrdst,
                            qrylen, dbstrlen, qrypos, rfnpos,  g0, g02, d82,
                            tmpdpdiagbuffers, tmpdpalnpossbuffer,  tfmBest, ccm[0]/*scv*/, ccm/*dstv*/);
                        gbest = ccm[0][0];//score
                    }

                    //NOTE: CONDITIONAL==true because effnsteps(==nthreads)<<maxnsteps
                    SaveBestQRScoresAndTM_Complete<false/*WRITEFRAGINFO*/,true/*CONDITIONAL*/>(
                        best, gbest,  qi/*qryndx*/, ri/*dbstrndx*/, ndbCstrs_, 
                        maxnsteps_, tid/*sfragfctxndx*/, sfragndx, sfragpos,
                        tfmBest, wrkmemtmibest, wrkmemaux);
                }//omp for
        //implicit barrier here

        //find the max score among fragment variants
        #pragma omp for collapse(2)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int bi = 0; bi < nblocks_x_best; bi++)
            {//threads process blocks of references
                ProductionSaveBestScoresAndTMAmongBests
                    <XDIM,memalignment,false/*WRITEFRAGINFO*/,true/*CONDITIONAL*/>(
                        qi/*qryndx*/, bi/*rfnblkndx*/, ndbCstrs_,
                        maxnsteps_, nthreads/*effnsteps*/,
                        ccm, querypmbeg, bdbCpmbeg,
                        wrkmemtmibest, wrkmemaux, alndatamem, tfmmem);
            }
    }
}

// =========================================================================
// Instantiations
// 
#define INSTANTIATE_MpStageFin_ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel(tpTFM_DINV) \
    template void MpStageFin::ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel<tpTFM_DINV>( \
        const int nmaxconvit, \
        const char* const * const __RESTRICT__ querypmbeg, \
        const char* const * const __RESTRICT__ bdbCpmbeg, \
        const float* const __RESTRICT__ tmpdpalnpossbuffer, \
        float* const __RESTRICT__ tmpdpdiagbuffers, \
        float* const __RESTRICT__ wrkmemtmibest, \
        float* const __RESTRICT__ wrkmemaux, \
        float* const __RESTRICT__ alndatamem, \
        float* const __RESTRICT__ tfmmem);

INSTANTIATE_MpStageFin_ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel(false);
INSTANTIATE_MpStageFin_ProductionFragmentBasedDPAlignmentRefinementPhase2_fullsearchKernel(true);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// ProductionFragmentBasedDPAlignmentRefinementPhase2Kernel: phase 2 to perform 
// production-version tuning of the the best superposition obtained 
// within the single kernel's actions; write the final superposition scores;
// TFM_DINV, use doubly inverted transformation matrices under suitable conditions;
// nmaxconvit, maximum number of superposition iterations;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// tmpdpdiagbuffers, temporary diagonal buffers for positional scores/dsts;
// wrkmemtmibest, working memory for best-performing transformation matrices;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// alndatamem, memory for full alignment information;
// tfmmem, memory for production transformation matrices;
// 
template<bool TFM_DINV>
void MpStageFin::ProductionFragmentBasedDPAlignmentRefinementPhase2Kernel(
    const int nmaxconvit,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* const __RESTRICT__ tmpdpalnpossbuffer,
    float* const __RESTRICT__ tmpdpdiagbuffers,
    float* const __RESTRICT__ wrkmemtmibest,
    float* const __RESTRICT__ wrkmemaux,
    float* const __RESTRICT__ alndatamem,
    float* const __RESTRICT__ tfmmem)
{
    enum{
        //effective number of fields:
        nEFFDS = twmvEndOfCCDataExt,
        XDIM = MPS1_TBINITSP_COMPLETEREFINE_XDIM
    };

    MYMSG("MpStageFin::ProductionFragmentBasedDPAlignmentRefinementPhase2Kernel", 4);
    static const std::string preamb =
    "MpStageFin::ProductionFragmentBasedDPAlignmentRefinementPhase2Kernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = mycemin((size_t)PMBSdatalignment, CuMemoryBase::GetMinMemAlignment());
    //minimum length among largest
    int minlenmax = mymin(qystr1len_, dbstr1len_);
    int maxalnmax = minlenmax;
    //maximum number of subdivisions
    // const int nmaxsubfrags = FRAGREF_NMAXSUBFRAGS;
    //NOTE: step for the SECOND phase to final (finer-scale) refinement;
    constexpr int sfragstep = FRAGREF_SFRAGSTEP_mini;

    //max #fragment position factors around an identified position
    //**********************************************************************
    //NOTE: multiply maxalnmax by 2 since sub-optimal (first-phase) alignment
    //NOTE: position can be identified at the end of alignment!
    //**********************************************************************
    int maxnfragfcts = mymin(2 * maxalnmax, CUSFN_TBSP_FIN_REFINEMENT_MAX_NPOSITIONS);
    maxnfragfcts = (maxnfragfcts + sfragstep-1) / sfragstep;
    int nlocsteps2 = maxnfragfcts;//total number for ONE fragment length

    if(nlocsteps2 < 1 || (int)maxnsteps_ < nlocsteps2)
        throw MYRUNTIME_ERROR(preamb +
        "Invalid number of superposition tests: "+std::to_string(nlocsteps2));

    //execution configuration: process and refine multiple
    //query-reference alignment variants:
    const int nblocks_x = ndbCstrs_;
    const int nblocks_x_best = (ndbCstrs_ + XDIM - 1) / XDIM;
    const int nblocks_y = nlocsteps2;
    const int nblocks_z = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_z * (size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPS1_TBINITSP_COMPLETEREFINE_CHSIZE);

    size_t chunksizeinit_helper = 
        ((size_t)nblocks_z * (size_t)nthreads * (size_t)nblocks_x_best + (size_t)nthreads - 1) / nthreads;
    const int chunksizeinit = (int)mymin(chunksizeinit_helper, (size_t)MPS1_TBINITSP_COMPLETEREFINE_CHSIZE);

    //cache for the cross-covarinace matrix and related data: 
    float ccm[nEFFDS][XDIM];
    float tfm[nEFFDS];//nEFFDS>nTTranformMatrix
    float ccmLast[nEFFDS];
    float tfmBest[nEFFDS];

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        shared(maxnfragfcts) \
        private(ccm, tfm, ccmLast, tfmBest)
    {
        //initialize best scores
        #pragma omp for collapse(3) schedule(dynamic, chunksizeinit)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nthreads; si++)
                for(int bi = 0; bi < nblocks_x_best; bi++)
                {//threads process blocks of references
                    const int istr0 = bi * XDIM;
                    const int istre = mymin(istr0 + XDIM, (int)ndbCstrs_);
                    const int mloc = ((qi * maxnsteps_ + si) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    #pragma omp simd aligned(wrkmemaux:memalignment)
                    for(int ri = istr0; ri < istre; ri++)
                        wrkmemaux[mloc + tawmvBestScore * ndbCstrs_ + ri/*dbstrndx*/] = 0.0f;
                }
        //implicit barrier here

        #pragma omp for collapse(3) schedule(static, chunksize)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nblocks_y; si++)
                for(int ri = 0; ri < nblocks_x; ri++)
                {//threads process references
                    //check convergence:
                    int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                    tfm[6] = wrkmemaux[mloc0 + ri];//reuse cache
                    if(((int)(tfm[6])) & (CONVERGED_LOWTMSC_bitval))
                        continue;

                    const int sfragfct = si;//fragment factor
                    int sfragndx, sfragpos;//fragment length index and position

                    const int qrylenorg = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                    // const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                    const int dbstrlenorg = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                    const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);

                    enum {qrypos = 0, rfnpos = 0};
                    const int tid = omp_get_thread_num();
                    int qrylen, dbstrlen;

                    //NOTE: #matched positions tawmvNAlnPoss written at sfragfct==0:
                    mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    qrylen = dbstrlen = wrkmemaux[mloc0 + tawmvNAlnPoss * ndbCstrs_ + ri];
                    sfragndx = wrkmemaux[mloc0 + tawmvSubFragNdx * ndbCstrs_ + ri];
                    sfragpos = wrkmemaux[mloc0 + tawmvSubFragPos * ndbCstrs_ + ri];
                    sfragpos += (sfragfct - (maxnfragfcts>>1)) * sfragstep;
                    if(sfragndx == 0) sfragndx++;

                    //out-of-bounds check:
                    const int fraglen = GetFragLength(qrylen, dbstrlen, qrypos, rfnpos, sfragndx);
                    if(fraglen < 1) continue;
                    if(sfragpos < 0 ||
                       qrylen + sfragstep <= qrypos + sfragpos + fraglen ||
                       dbstrlen + sfragstep <= rfnpos + sfragpos + fraglen)
                        continue;

                    //threshold calculated for the original lengths
                    const float d0 = GetD0fin(qrylenorg, dbstrlenorg);
                    const float d02 = SQRD(d0);
                    const float d82 = GetD82(qrylenorg, dbstrlenorg);
                    float best = -1.0f;//best score obtained

                    ProductionRefinementPhase2InnerLoop<nEFFDS,XDIM,memalignment,TFM_DINV>(
                        tid/*sfragfctxndx*/, qi/*qryndx*/,
                        nmaxconvit, ndbCposs_, dbxpad_, maxnsteps_,
                        qrylenorg, dbstrlenorg, qrylen, dbstrlen, dbstrdst,
                        qrypos, rfnpos, sfragpos, fraglen,
                        d0, d02, d82,  best/**/,
                        ccm, ccmLast, tfm, tfmBest,
                        tmpdpalnpossbuffer, tmpdpdiagbuffers);

                    //calculate the score for the larger structure of the two:
                    //threshold calculated for the greater length
                    const int greaterlen = mymax(qrylenorg, dbstrlenorg);
                    const float g0 = GetD0fin(greaterlen, greaterlen);
                    const float g02 = SQRD(g0);
                    float gbest = best;//score calculated for the other structure

                    if(qrylenorg != dbstrlenorg) {
                        CalcScoresUnrl_DPRefined_Complete<XDIM,memalignment,CHCKDST_NOCHECK>(
                            READCNST_CALC2,
                            qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_, tid/*sfragfctxndx*/, dbstrdst,
                            qrylen, dbstrlen, qrypos, rfnpos,  g0, g02, d82,
                            tmpdpdiagbuffers, tmpdpalnpossbuffer,  tfmBest, ccm[0]/*scv*/, ccm/*dstv*/);
                        gbest = ccm[0][0];//score
                    }

                    //NOTE: CONDITIONAL==true because effnsteps(==nthreads)<<maxnsteps
                    SaveBestQRScoresAndTM_Complete<false/*WRITEFRAGINFO*/,true/*CONDITIONAL*/>(
                        best, gbest,  qi/*qryndx*/, ri/*dbstrndx*/, ndbCstrs_, 
                        maxnsteps_, tid/*sfragfctxndx*/, sfragndx, sfragpos,
                        tfmBest, wrkmemtmibest, wrkmemaux);
                }//omp for
        //implicit barrier here

        //find the max score among fragment variants
        #pragma omp for collapse(2)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int bi = 0; bi < nblocks_x_best; bi++)
            {//threads process blocks of references
                ProductionSaveBestScoresAndTMAmongBests
                    <XDIM,memalignment,false/*WRITEFRAGINFO*/,true/*CONDITIONAL*/>(
                        qi/*qryndx*/, bi/*rfnblkndx*/, ndbCstrs_,
                        maxnsteps_, nthreads/*effnsteps*/,
                        ccm, querypmbeg, bdbCpmbeg,
                        wrkmemtmibest, wrkmemaux, alndatamem, tfmmem);
            }
    }
}

// =========================================================================
// Instantiations
// 
#define INSTANTIATE_MpStageFin_ProductionFragmentBasedDPAlignmentRefinementPhase2Kernel(tpTFM_DINV) \
    template void MpStageFin::ProductionFragmentBasedDPAlignmentRefinementPhase2Kernel<tpTFM_DINV>( \
        const int nmaxconvit, \
        const char* const * const __RESTRICT__ querypmbeg, \
        const char* const * const __RESTRICT__ bdbCpmbeg, \
        const float* const __RESTRICT__ tmpdpalnpossbuffer, \
        float* const __RESTRICT__ tmpdpdiagbuffers, \
        float* const __RESTRICT__ wrkmemtmibest, \
        float* const __RESTRICT__ wrkmemaux, \
        float* const __RESTRICT__ alndatamem, \
        float* const __RESTRICT__ tfmmem);

INSTANTIATE_MpStageFin_ProductionFragmentBasedDPAlignmentRefinementPhase2Kernel(false);
INSTANTIATE_MpStageFin_ProductionFragmentBasedDPAlignmentRefinementPhase2Kernel(true);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearchKernel: phase 2 to
// perform production-version tuning of the the best superposition obtained 
// within the single kernel's actions; write the final superposition scores;
// NOTE: This version performs a log number of superposition evaluations;
// TFM_DINV, use doubly inverted transformation matrices under suitable conditions;
// nmaxconvit, maximum number of superposition iterations;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// tmpdpdiagbuffers, temporary diagonal buffers for positional scores/dsts;
// wrkmemtmibest, working memory for best-performing transformation matrices;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// alndatamem, memory for full alignment information, including scores;
// tfmmem, memory for production transformation matrices;
// 
template<bool TFM_DINV>
void MpStageFin::ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearchKernel(
    const int nmaxconvit,
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* const __RESTRICT__ tmpdpalnpossbuffer,
    float* const __RESTRICT__ tmpdpdiagbuffers,
    float* const __RESTRICT__ /* wrkmemtmibest */,
    float* const __RESTRICT__ wrkmemaux,
    float* const __RESTRICT__ alndatamem,
    float* const __RESTRICT__ tfmmem)
{
    enum{
        //effective number of fields:
        nEFFDS = twmvEndOfCCDataExt,
        XDIM = MPS1_TBINITSP_COMPLETEREFINE_XDIM
    };

    MYMSG("MpStageFin::ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearchKernel", 4);
    // static const std::string preamb =
    // "MpStageFin::ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearchKernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = mycemin((size_t)PMBSdatalignment, CuMemoryBase::GetMinMemAlignment());
    //minimum length among largest
    // int minlenmax = myhdmin(qystr1len_, dbstr1len_);
    // int maxalnmax = minlenmax;
    //maximum number of subdivisions
    const int nmaxsubfrags = FRAGREF_NMAXSUBFRAGS;
    const int nlocsteps2 = 1;

    //execution configuration: process and refine multiple
    //query-reference alignment variants:
    const int nblocks_x = ndbCstrs_;
    const int nblocks_x_best = (ndbCstrs_ + XDIM - 1) / XDIM;
    const int nblocks_y = nlocsteps2;
    const int nblocks_z = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_z * (size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPS1_TBINITSP_COMPLETEREFINE_CHSIZE);

    size_t chunksizeinit_helper = 
        ((size_t)nblocks_z * (size_t)nthreads * (size_t)nblocks_x_best + (size_t)nthreads - 1) / nthreads;
    const int chunksizeinit = (int)mymin(chunksizeinit_helper, (size_t)MPS1_TBINITSP_COMPLETEREFINE_CHSIZE);

    //cache for the cross-covarinace matrix and related data: 
    float ccm[nEFFDS][XDIM];
    float tfm[nEFFDS];//nEFFDS>nTTranformMatrix
    float ccmLast[nEFFDS];
    float tfmBest[nEFFDS];

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        private(ccm, tfm, ccmLast, tfmBest)
    {
        //initialize best scores
        #pragma omp for collapse(3) schedule(dynamic, chunksizeinit)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nthreads; si++)
                for(int bi = 0; bi < nblocks_x_best; bi++)
                {//threads process blocks of references
                    const int istr0 = bi * XDIM;
                    const int istre = mymin(istr0 + XDIM, (int)ndbCstrs_);
                    const int mloc = ((qi * maxnsteps_ + si) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    #pragma omp simd aligned(wrkmemaux:memalignment)
                    for(int ri = istr0; ri < istre; ri++)
                        wrkmemaux[mloc + tawmvBestScore * ndbCstrs_ + ri/*dbstrndx*/] = 0.0f;
                }
        //implicit barrier here

        #pragma omp for collapse(3) schedule(static, chunksize)
        for(int qi = 0; qi < nblocks_z; qi++)
            for(int si = 0; si < nblocks_y; si++)
                for(int ri = 0; ri < nblocks_x; ri++)
                {//threads process references
                    //check convergence:
                    int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                    tfm[6] = wrkmemaux[mloc0 + ri];//reuse cache
                    if(((int)(tfm[6])) & (CONVERGED_LOWTMSC_bitval))
                        continue;

                    const int qrylenorg = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                    // const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                    const int dbstrlenorg = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                    const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);

                    enum {qrypos = 0, rfnpos = 0};
                    const int tid = omp_get_thread_num();
                    int qrylen, dbstrlen, sfragposorg;
                    float bestorg;//best score obtained

                    //NOTE: #matched positions tawmvNAlnPoss written at sfragfct==0:
                    mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                    qrylen = dbstrlen = wrkmemaux[mloc0 + tawmvNAlnPoss * ndbCstrs_ + ri];
                    sfragposorg = wrkmemaux[mloc0 + tawmvSubFragPos * ndbCstrs_ + ri];
                    bestorg = wrkmemaux[mloc0 + tawmvGrandBest * ndbCstrs_ + ri];

                    //threshold calculated for the original lengths
                    const float d0 = GetD0fin(qrylenorg, dbstrlenorg);
                    const float d02 = SQRD(d0);
                    const float d82 = GetD82(qrylenorg, dbstrlenorg);
                    float best = bestorg;//best score obtained

                    for(int sfragstepabs = 32; sfragstepabs >= 1; sfragstepabs >>= 1) {
                        for(int sgn = 1; sgn >= -1; sgn -= 2)
                        {
                            int sfragstep = sgn * sfragstepabs;
                            int sfragpos = sfragposorg + sfragstep;
                            for(int sfragndx = 1; sfragndx < nmaxsubfrags; sfragndx++)
                            {
                                int fraglen = GetFragLength(qrylen, dbstrlen, qrypos, rfnpos, sfragndx);

                                if(fraglen < 1) continue;
                                if(sfragpos < 0 ||
                                   qrylen + mymax(0, sfragstep) <= qrypos + sfragpos + fraglen ||
                                   dbstrlen + mymax(0, sfragstep) <= rfnpos + sfragpos + fraglen)
                                    continue;

                                float bestprev = best;

                                ProductionRefinementPhase2InnerLoop<nEFFDS,XDIM,memalignment,TFM_DINV>(
                                    tid/*sfragfctxndx*/, qi/*qryndx*/,
                                    nmaxconvit, ndbCposs_, dbxpad_, maxnsteps_,
                                    qrylenorg, dbstrlenorg, qrylen, dbstrlen, dbstrdst,
                                    qrypos, rfnpos, sfragpos, fraglen,
                                    d0, d02, d82,  best/**/,
                                    ccm, ccmLast, tfm, tfmBest,
                                    tmpdpalnpossbuffer, tmpdpdiagbuffers);

                                if(bestprev < best) sfragposorg = sfragpos;
                            }
                        }
                    }

                    if(best <= bestorg) continue;

                    //calculate the score for the larger structure of the two:
                    //threshold calculated for the greater length
                    const int greaterlen = mymax(qrylenorg, dbstrlenorg);
                    const float g0 = GetD0fin(greaterlen, greaterlen);
                    const float g02 = SQRD(g0);
                    float gbest = best;//score calculated for the other structure

                    if(qrylenorg != dbstrlenorg) {
                        CalcScoresUnrl_DPRefined_Complete<XDIM,memalignment,CHCKDST_NOCHECK>(
                            READCNST_CALC2,
                            qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_, tid/*sfragfctxndx*/, dbstrdst,
                            qrylen, dbstrlen, qrypos, rfnpos,  g0, g02, d82,
                            tmpdpdiagbuffers, tmpdpalnpossbuffer,  tfmBest, ccm[0]/*scv*/, ccm/*dstv*/);
                        gbest = ccm[0][0];//score
                    }

                    //NOTE: write directly to production output memory:
                    SaveBestQRScoresAndTM_Phase2_logsearch_Complete(
                        best, gbest,  qi/*qryndx*/, ri/*dbstrndx*/, ndbCstrs_,
                        qrylenorg, dbstrlenorg,
                        tfmBest, tfmmem, alndatamem);
                }//omp for
    }//omp parallel
}

// =========================================================================
// Instantiations
// 
#define INSTANTIATE_MpStageFin_ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearchKernel(tpTFM_DINV) \
    template void MpStageFin::ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearchKernel<tpTFM_DINV>( \
        const int nmaxconvit, \
        const char* const * const __RESTRICT__ querypmbeg, \
        const char* const * const __RESTRICT__ bdbCpmbeg, \
        const float* const __RESTRICT__ tmpdpalnpossbuffer, \
        float* const __RESTRICT__ tmpdpdiagbuffers, \
        float* const __RESTRICT__ wrkmemtmibest, \
        float* const __RESTRICT__ wrkmemaux, \
        float* const __RESTRICT__ alndatamem, \
        float* const __RESTRICT__ tfmmem);

INSTANTIATE_MpStageFin_ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearchKernel(false);
INSTANTIATE_MpStageFin_ProductionFragmentBasedDPAlignmentRefinementPhase2_logsearchKernel(true);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// Production2TMscoresKernel: calculate secondary TM-scores, 2TM-scores, and 
// write them to memory;
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// wrkmemaux, auxiliary working memory (includes the section of scores);
// tfmmem, memory of transformation matrices;
// alndatamem, memory for full alignment information, including scores;
// 
void MpStageFin::Production2TMscoresKernel(
    const char* const * const __RESTRICT__ querypmbeg,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* const __RESTRICT__ tmpdpalnpossbuffer,
    const float* const __RESTRICT__ wrkmemaux,
    const float* const __RESTRICT__ tfmmem,
    float* const __RESTRICT__ alndatamem)
{
    enum{
        XDIM = MPDP_PRODUCTION_2TMSCORE_XDIM
    };

    MYMSG("MpStageFin::Production2TMscoresKernel", 4);
    static const std::string preamb = "MpStageFin::Production2TMscoresKernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = mycemin((size_t)PMBSdatalignment, CuMemoryBase::GetMinMemAlignment());

    //execution configuration for calculating 2tmscores:
    const int nblocks_x = ndbCstrs_;
    const int nblocks_y = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPDP_PRODUCTION_2TMSCORE_CHSIZE);

    //cache for for scores and tfm: 
    float scv[XDIM];
    float tfm[nTTranformMatrix];

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        private(scv, tfm)
    {
        #pragma omp for collapse(2) schedule(static, chunksize)
        for(int qi = 0; qi < nblocks_y; qi++)
            for(int ri = 0; ri < nblocks_x; ri++)
            {//threads process references
                //check convergence:
                int mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs_;
                tfm[6] = wrkmemaux[mloc0 + ri];//reuse cache
                if(((int)(tfm[6])) & (CONVERGED_LOWTMSC_bitval))
                    continue;

                const int qrylenorg = PMBatchStrData::GetLengthAt(querypmbeg, qi);
                const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);
                const int dbstrlenorg = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
                const int dbstrdst = PMBatchStrData::GetAddressAt(bdbCpmbeg, ri);

                enum {qrypos = 0, rfnpos = 0};
                int qrylen, dbstrlen;

                //NOTE: #matched positions tawmvNAlnPoss written at sfragfct==0:
                mloc0 = ((qi * maxnsteps_ + 0) * nTAuxWorkingMemoryVars) * ndbCstrs_;
                qrylen = dbstrlen = wrkmemaux[mloc0 + tawmvNAlnPoss * ndbCstrs_ + ri];

                //READ globally best transformation matrix for a pair:
                mloc0 = (qi * ndbCstrs_ + ri) * nTTranformMatrix;
                #pragma omp simd aligned(tfmmem:memalignment)
                for(int f = 0; f < nTTranformMatrix; f++)
                    tfm[f] = tfmmem[mloc0 + f];

                //threshold calculated for the original lengths
                const float d0 = GetD0fin(qrylenorg, dbstrlenorg);
                const float d02 = SQRD(d0);

                Calc2TMscoresUnrl_Complete<XDIM,memalignment>(
                    qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_,
                    qrydst, dbstrdst, qrylen, dbstrlen, qrypos, rfnpos, d02,
                    querypmbeg, bdbCpmbeg, tmpdpalnpossbuffer, tfm, scv);

                const float best = scv[0];//score


                //calculate the score for the larger structure of the two:
                //threshold calculated for the greater length
                const int greaterlen = mymax(qrylenorg, dbstrlenorg);
                const float g0 = GetD0fin(greaterlen, greaterlen);
                const float g02 = SQRD(g0);
                float gbest = best;//score calculated for the other structure

                if(qrylenorg != dbstrlenorg) {
                    Calc2TMscoresUnrl_Complete<XDIM,memalignment>(
                        qi/*qryndx*/, ndbCposs_, dbxpad_, maxnsteps_,
                        qrydst, dbstrdst, qrylen, dbstrlen, qrypos, rfnpos, g02,
                        querypmbeg, bdbCpmbeg, tmpdpalnpossbuffer, tfm, scv);
                    gbest = scv[0];//score
                }


                //NOTE: write directly to production output memory:
                SaveBestQR2TMscores_Complete(
                    best, gbest,  qi/*qryndx*/, ri/*dbstrndx*/, ndbCstrs_, 
                    qrylenorg, dbstrlenorg, alndatamem);
            }//omp for
        //implicit barrier here
    }
}
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// RevertTfmMatricesKernel: revert transformation matrices;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// tfmmem, memory section for transformation matrices;
// NOTE: unrolling by a factor of N;
// 
void MpStageFin::RevertTfmMatricesKernel(
    float* const __RESTRICT__ tfmmem)
{
    enum{
        nFDS = nTTranformMatrix,
        XFCT = MPS1_TBINITSP_TFMINIT_XFCT
    };

    MYMSG("MpStageFin::RevertTfmMatricesKernel", 4);
    // static const std::string preamb = "MpStageFin::RevertTfmMatricesKernel: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = CuMemoryBase::GetMinMemAlignment();

    //configuration: process multiple query-reference transformation matrices:
    const int nblocks_x = (ndbCstrs_ + XFCT - 1) / XFCT;
    const int nblocks_y = nqystrs_;

    size_t chunksize_helper = 
        ((size_t)nblocks_y * (size_t)nblocks_x + (size_t)nthreads - 1) / nthreads;
    const int chunksize = (int)mymin(chunksize_helper, (size_t)MPS1_TBINITSP_TFMINIT_CHSIZE);

    //cache for transformation matrices: 
    float tfms[XFCT * nFDS];

    //NOTE: constant member pointers are shared by definition;
    #pragma omp parallel num_threads(nthreads) default(shared) \
        private(tfms)
    {
        #pragma omp for collapse(2) schedule(dynamic, chunksize)
        for(int qi = 0; qi < nblocks_y; qi++)
            for(int bi = 0; bi < nblocks_x; bi++)
            {//threads process blocks of references
                const int istr0 = bi * XFCT;
                const int istre = mymin(istr0 + XFCT, (int)ndbCstrs_);
                //beginning address:
                const int mloc = (qi * ndbCstrs_ + istr0) * nTTranformMatrix;
                //zero-based indices:
                const int ii0 = 0;
                const int iie = istre - istr0;
                //READ tfms:
                #pragma omp simd collapse(2) aligned(tfmmem:memalignment)
                for(int ii = ii0; ii < iie; ii++)
                    for(int f = 0; f < nFDS; f++)
                        tfms[ii * nFDS + f] = tfmmem[mloc + ii * nFDS + f];
                //revert:
                #pragma omp simd
                for(int ii = ii0; ii < iie; ii++) {
                    InvertRotMtx(&tfms[ii * nFDS]);
                    InvertTrlVec(&tfms[ii * nFDS]);
                }
                //WRITE tfms back:
                #pragma omp simd collapse(2) aligned(tfmmem:memalignment)
                for(int ii = ii0; ii < iie; ii++)
                    for(int f = 0; f < nFDS; f++)
                        tfmmem[mloc + ii * nFDS + f] = tfms[ii * nFDS + f];
            }
    }//omp parallel
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
