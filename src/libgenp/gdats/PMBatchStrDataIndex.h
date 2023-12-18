/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __PMBatchStrDataIndexIndex_h__
#define __PMBatchStrDataIndexIndex_h__

#include "libutil/mybase.h"

#include <string.h>

#include <vector>
#include <memory>

#include "libutil/CLOptions.h"
#include "PM2DVectorFields.h"
#include "DRDataDeleter.h"
#include "PMBatchStrData.h"

// -------------------------------------------------------------------------
// PMBatchStrDataIndex: Complete indexed batch structure data for parallel 
// processing
//
class PMBatchStrDataIndex {
public:
    PMBatchStrDataIndex()
    :   tmpbdbCdata_(nullptr),
        bdbCdata_(nullptr),
        szbdbCdata_(0)
    {
        memset(bdbCpmbeg_, 0, pmv2DTotIndexFlds * sizeof(void*));
        memset(bdbCpmend_, 0, pmv2DTotIndexFlds * sizeof(void*));
    }

    ~PMBatchStrDataIndex() {}

    // *** METHODS ***
    //index coordinates present in PMBatchStrData object
    void MakeIndex(const PMBatchStrData&);

    size_t GetPMDataSize();
    static size_t GetPMDataSize(size_t totallen);

    bool ContainsData() const {//whether data is present
        return bdbCpmbeg_[pmv2DNdxCoords] < bdbCpmend_[pmv2DNdxCoords] &&
            bdbCpmbeg_[pmv2DNdxLeft] < bdbCpmend_[pmv2DNdxLeft] &&
            bdbCpmbeg_[pmv2DNdxRight] < bdbCpmend_[pmv2DNdxRight];
    }

    //change/pack structure pointers to include structures indexed in filterdata
    static void FilterStructs(
        char** bdbndxpmbeg, char** bdbndxpmend,
        char* const* bdbpmbeg, char* const* bdbpmend,
        const unsigned int* filterdata);

    size_t GetNoPositsWritten() const {//#positions written in bdbCdata_
        return (size_t)(bdbCpmend_[pmv2DNdxCoords]-bdbCpmbeg_[pmv2DNdxCoords]) / SZFPTYPE;
    }

    //field of the structure at the given position/index (NOTE:pointers assumed valid):
    template<typename T, int F>
    static T GetFieldAt(const char* const * const bdbpmbeg, int ndx) {
        return ((T*)(bdbpmbeg[F]))[ndx];
    }

    void Search(const PMBatchStrData&, FPTYPE crds[]) const;

    void Print() const;

private:
    constexpr static size_t ndims_ = 3;
    enum {ptrLeft_, ptrRight_, nPtrs_};

    void AllocateSpace(size_t);

    int ConstructKdtree(
        FPTYPE* crds[], INTYPE* ptrs[],
        int begin, int end, int dimndx);

    void NNRecursive(
        int address, int root, FPTYPE crds[ndims_], int dimndx,
        int& nvisited, int& nestndx, float& nestdst) const;

    void NNIterative(
        int address, int root, FPTYPE crds[ndims_], int dimndx,
        int& nvisited, int& nestndx, float& nestdst) const;

    void NNNaive(
        int address, int len, FPTYPE crds[ndims_],
        int& nvisited, int& nestndx, float& nestdst) const;

    //tree branch (left/right) at a given position (node assumed to be valid):
    INTYPE GetOrgndxAt(int address, int node) const {
        return ((INTYPE*)(bdbCpmbeg_[pmv2DNdxOrgndx]))[address + node];
    }

    //tree branch (left/right) at a given position (node assumed to be valid):
    INTYPE GetBranchAt(int address, int node, int branch) const {
        return ((INTYPE*)
            (bdbCpmbeg_[(branch == ptrLeft_)? pmv2DNdxLeft: pmv2DNdxRight]))
                [address + node];
    }

    //coordinate value at a given position (node assumed to be valid):
    FPTYPE GetCoordinateAt(int address, int node, int dim) const {
        return ((FPTYPE*)(bdbCpmbeg_[pmv2DNdxCoords+dim]))[address + node];
    }

    //squared distance to the point (node assumed to be valid):
    float GetDistance2(int address, int node, FPTYPE crds[]) const {
        return
            SQRD(GetCoordinateAt(address, node, 0) - crds[0]) +
            SQRD(GetCoordinateAt(address, node, 1) - crds[1]) +
            SQRD(GetCoordinateAt(address, node, 2) - crds[2]);
    }

private:
    template<typename T, int field>
    void index_helper_psfields_assign(size_t nposs);

    template <typename T, int field>
    void index_helper_psfields_adjend(size_t nposs);

private:
    std::unique_ptr<char,DRDataDeleter> tmpbdbCdata_;//temporary buffer for indexed data
    std::vector<INTYPE> tmpndxs_;//temporary index vector

public:
    // *** MEMBER VARIABLES ***
    std::unique_ptr<char,DRHostDataDeleter> bdbCdata_;
    char* bdbCpmbeg_[pmv2DTotIndexFlds];//addresses of the beginnings of the fields in bdbCdata_
    char* bdbCpmend_[pmv2DTotIndexFlds];//end addresses of the fields in bdbCdata_

    size_t szbdbCdata_;//size allocated for bdbCdata_
};

// =========================================================================
// INLINES
// -------------------------------------------------------------------------
// GetPMDataSize: get the size of the indexed structure data
inline
size_t PMBatchStrDataIndex::GetPMDataSize()
{
    MYMSG("PMBatchStrDataIndex::GetPMDataSize",6);
    size_t size = 0;
    for(int n = 0; n < pmv2DTotIndexFlds; n++)
        size += (size_t)(bdbCpmend_[n]-bdbCpmbeg_[n]);
    return size;
}

// -------------------------------------------------------------------------
// GetPMDataSize: get the size of complete indexed structure model data;
// totallen, the total number of positions (residues/atoms);
inline
size_t PMBatchStrDataIndex::GetPMDataSize(size_t totallen)
{
    MYMSG("PMBatchStrDataIndex::GetPMDataSize [static]",6);
    size_t size = 0;
    for(int n = 0; n < pmv2DTotIndexFlds; n++)
        size += TPM2DIndexFieldSize::szvfs_[n] * totallen;
    return size;
}

#endif//__PMBatchStrDataIndexIndex_h__
