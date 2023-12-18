/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __MpReform_h__
#define __MpReform_h__

#include <math.h>
#include "libutil/mybase.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libgenp/gdats/PMBatchStrData.h"
#include "libgenp/gdats/PMBatchStrDataIndex.h"
#include "libmymp/mpproc/mpprocconf.h"
#include "libmymp/mputil/simdscan.h"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/fragment.cuh"
#include "libmycu/cucom/cudef.h"

// -------------------------------------------------------------------------
// class MpReform for reformatting data of structure pairs that 
// proceed to the next stage
//
class MpReform {
public:
    MpReform(
        const uint maxnsteps,
        char** querypmbeg, char** querypmend,
        char** bdbCpmbeg, char** bdbCpmend,
        char** queryndxpmbeg, char** queryndxpmend,
        char** bdbCndxpmbeg, char** bdbCndxpmend,
        uint nqystrs, uint ndbCstrs,
        uint nqyposs, uint ndbCposs,
        float* tmpdpdiagbuffers,
        float* wrkmemaux, float* tfmmem, uint* globvarsbuf, uint* filterdata)
    :
        maxnsteps_(maxnsteps),
        querypmbeg_(querypmbeg), querypmend_(querypmend),
        bdbCpmbeg_(bdbCpmbeg), bdbCpmend_(bdbCpmend),
        queryndxpmbeg_(queryndxpmbeg), queryndxpmend_(queryndxpmend),
        bdbCndxpmbeg_(bdbCndxpmbeg), bdbCndxpmend_(bdbCndxpmend),
        nqystrs_(nqystrs), ndbCstrs_(ndbCstrs),
        nqyposs_(nqyposs), ndbCposs_(ndbCposs),
        tmpdpdiagbuffers_(tmpdpdiagbuffers),
        wrkmemaux_(wrkmemaux), tfmmem_(tfmmem), globvarsbuf_(globvarsbuf),
        filterdata_(filterdata)
    {}

    void MakeDbCandidateList() {
        constexpr int memalignment = 
            mycemin((size_t)PMBSdatalignment, CuMemoryBase::GetMinMemAlignment());
        MakeDbCandidateListHelper<memalignment>(
            nqystrs_, ndbCstrs_, maxnsteps_,
            querypmbeg_, bdbCpmbeg_, wrkmemaux_, filterdata_);
    }

    void SelectAndReformat(const int ndbCstrs2, const int maxndbCposs) {
        SelectAndReformatKernel(
            ndbCstrs2, maxndbCposs,
            filterdata_, tfmmem_, wrkmemaux_, tmpdpdiagbuffers_);
    }


protected:
    template<int DATALN>
    void MakeDbCandidateListHelper(
        const int nqystrs, const int ndbCstrs, const int maxnsteps,
        const char* const * const __RESTRICT__ /* querypmbeg */,
        const char* const * const __RESTRICT__ bdbCpmbeg,
        const float* const __RESTRICT__ wrkmemaux,
        uint* const __RESTRICT__ filterdata);

    void SelectAndReformatKernel(
        const int ndbCstrs2,
        const int maxndbCposs,
        const uint* const __RESTRICT__ filterdata,
        float* const __RESTRICT__ tfmmem,
        float* const __RESTRICT__ wrkmemaux,
        float* const __RESTRICT__ tmpdpdiagbuffers);


protected:
    const uint maxnsteps_;
    char* const * const querypmbeg_, * const * const querypmend_;
    char* const * const bdbCpmbeg_, * const *const bdbCpmend_;
    char* const * const queryndxpmbeg_, * const * const queryndxpmend_;
    char* const * const bdbCndxpmbeg_, * const *const bdbCndxpmend_;
    const uint nqystrs_, ndbCstrs_;
    const uint nqyposs_, ndbCposs_;
    float* const tmpdpdiagbuffers_;
    float* const wrkmemaux_, *const tfmmem_;
    uint* const globvarsbuf_, *const filterdata_;
};



// -------------------------------------------------------------------------
// INLINES ...
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// MakeDbCandidateListHelper: make list of reference structure (database)
// candidates proceeding to stages of more detailed superposition search and
// refinement;
// nqystrs, total number of query structures in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps to perform for each reference structure;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// filterdata, memory of new indices and addresses of passing references;
// NOTE: processes the reference structures over all queries for flags;
//
template<int DATALN>
inline
void MpReform::MakeDbCandidateListHelper(
    const int nqystrs, const int ndbCstrs, const int maxnsteps,
    const char* const * const __RESTRICT__ /* querypmbeg */,
    const char* const * const __RESTRICT__ bdbCpmbeg,
    const float* const __RESTRICT__ wrkmemaux,
    uint* const __RESTRICT__ filterdata)
{
    enum {
        PREPXS = 0,//index for the previous prefix sum and padding
        pad = 1,//padding
        lXFLG = fdNewReferenceIndex,//index for reference structure convergence flags/new indices
        lXLEN = fdNewReferenceAddress,//index for reference structure convergence lengths/new addresses
        lYDIM = nTFilterData,//MPFL_MAKECANDIDATELIST_YDIM,
        lXDIM = MPFL_MAKECANDIDATELIST_XDIM,
        lXDIM1 = MPFL_MAKECANDIDATELIST_XDIM + pad
    };

    //indices and lengths of selected structures
    int strd[lYDIM][lXDIM1];
    int tmp[lXDIM];

    strd[lXFLG][PREPXS] = 0;
    strd[lXLEN][PREPXS] = 0;

    for(int ri0 = 0; ri0 < ndbCstrs; ri0 += lXDIM)
    {
        //update the prefix sums originated from processing the last data block:
        if(ri0) {
            strd[lXFLG][PREPXS] = strd[lXFLG][lXDIM1 - 1];
            strd[lXLEN][PREPXS] = strd[lXLEN][lXDIM1 - 1];
        }

        // uint dbstrndx = dbstrndx0 + threadIdx.x;//reference index
        // int value = 0;//convflag for lXFLG and dbstrlen for lXLEN

        const int riend = mymin(ndbCstrs, ri0 + lXDIM);

        #pragma omp simd
        for(int ri = ri0; ri < riend; ri++) {
            int ii = ri - ri0 + pad;
            strd[lXFLG][ii] = 0;
        }

        //get convergence flags (over all queries)
        for(int qi/*qryndx*/ = 0; qi < nqystrs; qi++) {
            int mloc0 = ((qi * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
            #pragma omp simd aligned(wrkmemaux:DATALN)
            for(int ri = ri0; ri < riend; ri++) {
                int ii = ri - ri0 + pad;
                int lconv = wrkmemaux[mloc0 + tawmvConverged * ndbCstrs + ri];//float->int
                strd[lXFLG][ii] += ((lconv & CONVERGED_LOWTMSC_bitval) != 0);
            }
        }

        //selected structures have no convergence flags set for all queries;
        //get reference lengths too;
        #pragma omp simd aligned(bdbCpmbeg:DATALN)
        for(int ri = ri0; ri < riend; ri++) {
            int ii = ri - ri0 + pad;
            int value = strd[lXFLG][ii];
            strd[lXFLG][ii] = (value < nqystrs);
            strd[lXLEN][ii] = PMBatchStrData::GetLengthAt(bdbCpmbeg, ri);
        }

        //set reference lengths to 0 where convflag is set
        #pragma omp simd
        for(int ri = ri0; ri < riend; ri++) {
            int ii = ri - ri0 + pad;
            if(strd[lXFLG][ii] == 0)
                strd[lXLEN][ii] = 0;
        }

        //calculate inclusive (!) prefix sums for both flags, which then give indices,
        //and lengths for addresses:
        //TODO: change 1st agrument to the real size;
        mysimdincprefixsum<lXDIM>(lXDIM, &strd[lXFLG][1], tmp);
        mysimdincprefixsum<lXDIM>(lXDIM, &strd[lXLEN][1], tmp);

        //correct the prefix sums by adding the previously obtained values:
        #pragma omp simd
        for(int ri = ri0; ri < riend; ri++) {
            int ii = ri - ri0 + pad;
            strd[lXFLG][ii] += strd[lXFLG][PREPXS];
            strd[lXLEN][ii] += strd[lXLEN][PREPXS];
        }

        //write to output:
        #pragma omp simd
        for(int ri = ri0; ri < riend; ri++) {
            int ii = ri - ri0 + pad;
            int mloc = lXFLG * ndbCstrs + ri;
            int valueprev = strd[lXFLG][ii - 1];
            int value = strd[lXFLG][ii];
            //set to 0 for filtered-out structures:
            if(value == valueprev) value = 0;
            filterdata[mloc] = value;//WRITE indices
            //same for addresses:
            mloc = lXLEN * ndbCstrs + ri;
            valueprev = strd[lXLEN][ii - 1];
            value = strd[lXLEN][ii];
            if(value == valueprev) valueprev = 0;
            filterdata[mloc] = valueprev;//WRITE adjusted addresses
        }
    }
}

// -------------------------------------------------------------------------

#endif//__MpReform_h__
