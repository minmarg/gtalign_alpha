/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/mybase.h"

#include <stdio.h>
#include <assert.h>

#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>

#if defined(GPUINUSE) && 0
#   include <cuda_runtime_api.h>
#   include "libmycu/cucom/cucommon.h"
#endif

#include "libutil/CLOptions.h"
#include "libgenp/gproc/gproc.h"
#include "PM2DVectorFields.h"
#include "PMBatchStrData.h"
#include "PMBatchStrDataIndex.h"

// -------------------------------------------------------------------------
// template helpers: assignemnt of the block of position-specific field 
// using temporary buffer
template <typename T, int field>
void PMBatchStrDataIndex::index_helper_psfields_assign(size_t nposs)
{
    const size_t szT = TPM2DIndexFieldSize::szvfs_[field];

    for(size_t n = 0; n < nposs; n++) {
        int ndx = tmpndxs_[n];
        ((T*)(tmpbdbCdata_.get()))[n] = ((T*)(bdbCpmend_[field]))[ndx];
    }

    //copy the whole block of data from temporary buffer to the original location:
    memcpy(bdbCpmend_[field], tmpbdbCdata_.get(), nposs * szT);
}

// index_helper_psfields_adjend: adjust the end pointer for a field
template <typename T, int field>
void PMBatchStrDataIndex::index_helper_psfields_adjend(size_t nposs)
{
    const size_t szT = TPM2DIndexFieldSize::szvfs_[field];
    bdbCpmend_[field] += nposs * szT;
}

// -------------------------------------------------------------------------
// MakeIndex: index all structures present in the object strs;
// 
void PMBatchStrDataIndex::MakeIndex(const PMBatchStrData& strs)
{
    static const std::string preamb = "PMBatchStrData::MakeIndex: ";
    size_t nstts = strs.GetNoStructsWritten();
    size_t nposs = strs.GetNoPositsWritten();

    if(nposs <= nstts)
        throw MYRUNTIME_ERROR2(
        preamb + "Inconsistent source data.", CRITICAL);

    AllocateSpace(nposs);

    const FPTYPE* srccrds[ndims_] = {
        (FPTYPE*)(strs.bdbCpmbeg_[pmv2DCoords]),
        (FPTYPE*)(strs.bdbCpmbeg_[pmv2DCoords+1]),
        (FPTYPE*)(strs.bdbCpmbeg_[pmv2DCoords+2])
    };

    FPTYPE* dstcrds[ndims_] = {
        (FPTYPE*)(bdbCpmend_[pmv2DNdxCoords]),
        (FPTYPE*)(bdbCpmend_[pmv2DNdxCoords+1]),
        (FPTYPE*)(bdbCpmend_[pmv2DNdxCoords+2])
    };

    INTYPE* dstptrs[nPtrs_] = {
        (INTYPE*)(bdbCpmend_[pmv2DNdxLeft]),
        (INTYPE*)(bdbCpmend_[pmv2DNdxRight]),
    };

    INTYPE* dstndxs = (INTYPE*)(bdbCpmend_[pmv2DNdxOrgndx]);

    size_t nposprocessed = 0;

    for(size_t n = 0; n < nstts; n++)
    {
        int len = (int)PMBatchStrData::GetLengthAt(strs.bdbCpmbeg_, n);

        if(len < 1) continue;

        nposprocessed += len;

        if(nposs < nposprocessed)
            throw MYRUNTIME_ERROR2(
            preamb + "Invalid source data.", CRITICAL);

        tmpndxs_.resize(len);
        //indices from 0 to #positions
        std::iota(tmpndxs_.begin(), tmpndxs_.end(), 0);

        memcpy(dstcrds[0], srccrds[0], TPM2DIndexFieldSize::szvfs_[pmv2DNdxCoords] * len);
        memcpy(dstcrds[1], srccrds[1], TPM2DIndexFieldSize::szvfs_[pmv2DNdxCoords+1] * len);
        memcpy(dstcrds[2], srccrds[2], TPM2DIndexFieldSize::szvfs_[pmv2DNdxCoords+2] * len);

        ConstructKdtree(dstcrds, dstptrs,  0/*begin*/, len/*end*/, 0/*dimndx*/);

        //dstptrs contains pointers to the tree branches;
        //tmpndxs_ contains valid indices in the constructed index (tree);
        //rearrange coordinates to match the indices in tmpndxs_:
        index_helper_psfields_assign<FPTYPE,pmv2DNdxCoords>(len);
        index_helper_psfields_assign<FPTYPE,pmv2DNdxCoords+1>(len);
        index_helper_psfields_assign<FPTYPE,pmv2DNdxCoords+2>(len);
        std::copy(tmpndxs_.begin(), tmpndxs_.end(), dstndxs);

        srccrds[0] += len; srccrds[1] += len; srccrds[2] += len;
        dstcrds[0] += len; dstcrds[1] += len; dstcrds[2] += len;
        dstptrs[ptrLeft_] += len; dstptrs[ptrRight_] += len;
        dstndxs += len;

        //adjust end pointers of the fields
        index_helper_psfields_adjend<FPTYPE,pmv2DNdxCoords>(len);
        index_helper_psfields_adjend<FPTYPE,pmv2DNdxCoords+1>(len);
        index_helper_psfields_adjend<FPTYPE,pmv2DNdxCoords+2>(len);
        index_helper_psfields_adjend<INTYPE,pmv2DNdxLeft>(len);
        index_helper_psfields_adjend<INTYPE,pmv2DNdxRight>(len);
        index_helper_psfields_adjend<INTYPE,pmv2DNdxOrgndx>(len);
    }
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// Local type declarations:
// local position-specific field assigner
struct TL_ps_field_assigner {
    char** l_bdbpmbeg_, **l_bdbpmend_;
    TL_ps_field_assigner(char** bdbpmbeg, char** bdbpmend)
    : l_bdbpmbeg_(bdbpmbeg), l_bdbpmend_(bdbpmend) {}
    template<typename T, int field>
    void assign(unsigned int newaddr, unsigned int address, unsigned int length) {
        if(newaddr < address)
            std::copy(
                ((T*)(l_bdbpmbeg_[field])) + address,
                ((T*)(l_bdbpmbeg_[field])) + address + length,
                ((T*)(l_bdbpmbeg_[field])) + newaddr);
        l_bdbpmend_[field] = (char*)(((T*)(l_bdbpmbeg_[field])) + newaddr + length);
    }
};

// FilterStructs: change/pack structure pointers to include structures
// indexed in filterdata;
// NOTE: must be called before the same procedure is applied to PMBatchStrData!
//
void PMBatchStrDataIndex::FilterStructs(
    char** bdbndxpmbeg, char** bdbndxpmend,
    char* const* bdbpmbeg, char* const* bdbpmend,
    const unsigned int* filterdata)
{
    MYMSG("PMBatchStrDataIndex::FilterStructs",4);
    static const std::string preamb = "PMBatchStrDataIndex::FilterStructs: ";

    const unsigned int ndbstrs = PMBatchStrData::GetNoStructs(bdbpmbeg, bdbpmend);
    TL_ps_field_assigner ps_assigner(bdbndxpmbeg, bdbndxpmend);

    //no data when all structures are to be filtered out:
    for(int fld = 0; fld < pmv2DTotIndexFlds; fld++)
        bdbndxpmend[fld] = bdbndxpmbeg[fld];

    for(unsigned int n = 0; n < ndbstrs; n++) {
        unsigned int length = PMBatchStrData::GetLengthAt(bdbpmbeg, n);
        unsigned int address = PMBatchStrData::GetAddressAt(bdbpmbeg, n);
        //NOTE: data (indices and addresses) are values calculated by inclusive prefix sum!
        // unsigned int newndx = filterdata[ndbstrs * fdNewReferenceIndex + n];
        unsigned int newaddr = filterdata[ndbstrs * fdNewReferenceAddress + n];
        // if(newndx == 0) continue;
        // //adjust the index appropriately (NOTE: newaddr is already valid):
        // newndx--;
        //copy the entire structure to a different location in the same address space;
        //position-specific fields:
        ps_assigner.assign<FPTYPE,pmv2DNdxCoords+0>(newaddr, address, length);
        ps_assigner.assign<FPTYPE,pmv2DNdxCoords+1>(newaddr, address, length);
        ps_assigner.assign<FPTYPE,pmv2DNdxCoords+2>(newaddr, address, length);
        ps_assigner.assign<INTYPE,pmv2DNdxLeft>(newaddr, address, length);
        ps_assigner.assign<INTYPE,pmv2DNdxRight>(newaddr, address, length);
        ps_assigner.assign<INTYPE,pmv2DNdxOrgndx>(newaddr, address, length);
    }
}

// -------------------------------------------------------------------------
// AllocateSpace: allocate space for indexed structure data;
// totlen, total length to allocate for indexed data;
// 
void PMBatchStrDataIndex::AllocateSpace(size_t totlen)
{
    MYMSG("PMBatchStrDataIndex::AllocateSpace",4);
    static const std::string preamb = "PMBatchStrDataIndex::AllocateSpace: ";
    //max size of data given max total number of positions (residues)
    size_t szdata = PMBatchStrDataIndex::GetPMDataSize(totlen);

    size_t requestedsize = 
        szdata + //allowed size of data
        PMBSdatalignment * pmv2DTotIndexFlds;//for data alignment

    if(szbdbCdata_ < requestedsize)
    {
        char strbuf[BUF_MAX];
        sprintf(strbuf, "%sNew allocation for structure data, %zuB",
                preamb.c_str(), requestedsize);
        MYMSG(strbuf, 3);
        szbdbCdata_ = requestedsize;

#if defined(GPUINUSE) && 0
        if(CLOptions::GetIO_UNPINNED() == 0) {
            char* h_mpinned;
            MYCUDACHECK(cudaMallocHost((void**)&h_mpinned, szbdbCdata_) );
            MYCUDACHECKLAST;
            bdbCdata_.reset(h_mpinned);
        }
        else
#endif
            bdbCdata_.reset((char*)my_aligned_alloc(PMBSdatalignment, szbdbCdata_));

        //allocate space for tmp data too
        tmpbdbCdata_.reset((char*)std::malloc(totlen * sizeof(FPTYPE)));

        if(bdbCdata_.get() == NULL || tmpbdbCdata_.get() == NULL)
            throw MYRUNTIME_ERROR2(
            preamb + "Memory allocation failed.", CRITICAL);

        //reserve space for temporary vector of position indices (once)
        tmpndxs_.reserve(totlen+1);
    }

    size_t szdat = 0, szval = 0;
    char* pbdbCdata = bdbCdata_.get();

    //initialize pointers:
    for(int n = 0; n < pmv2DTotIndexFlds; n++)
    {
        bdbCpmbeg_[n] = bdbCpmend_[n] = pbdbCdata;
        szdat = TPM2DIndexFieldSize::szvfs_[n] * totlen;
        szval = ALIGN_UP(szdat, PMBSdatalignment);
        pbdbCdata += szval;
    }

    if(bdbCdata_.get() + 
       szdata + 
       PMBSdatalignment * pmv2DTotIndexFlds < pbdbCdata)
        throw MYRUNTIME_ERROR2(
        preamb + "Invalid calculated allocation size.",
        CRITICAL);
}

// -------------------------------------------------------------------------
// ConstructKdtree: construct k-d-tree for one particular structure;
// crds, 3D coordinates of a structure;
// ptrs, arrays of the pointers to the left and right in the k-d-tree;
// begin, current index of the beginning position of a structure;
// end, current index of the end position (exclusive) of a structure;
// dimndx, current dimension index;
// 
int PMBatchStrDataIndex::ConstructKdtree(
    FPTYPE* crds[ndims_], INTYPE* ptrs[nPtrs_],
    int begin, int end, int dimndx)
{
    if(end <= begin)
        return -1;
    int n = begin + ((end - begin) >> 1);
    //note tmpndxs_ contains original indices before the first call
    auto pndxs = tmpndxs_.begin();
    std::nth_element(
        pndxs + begin, pndxs + n, pndxs + end,
        [=](INTYPE n1, INTYPE n2) {
            //partially sort by coordinate in current dimension dimndx
            return crds[dimndx][n1] < crds[dimndx][n2];
        });
    dimndx = (dimndx + 1) % ndims_;
    ptrs[ptrLeft_][n] = ConstructKdtree(crds, ptrs, begin, n, dimndx);
    ptrs[ptrRight_][n] = ConstructKdtree(crds, ptrs, n + 1, end, dimndx);
    return n;
}

// -------------------------------------------------------------------------
// NNRecursive: find nearest node to the point with the given coordinates 
// crds[];
// address, beginning address of a structure;
// root, root node;
// nvisited, number of visited nodes;
// nestndx, index of the nearest node;
// nestdst, distance to the nearest node;
// 
void PMBatchStrDataIndex::NNRecursive(
    int address, int root, FPTYPE crds[ndims_], int dimndx,
    int& nvisited, int& nestndx, float& nestdst) const
{
    if(root < 0)
        return;
    nvisited++;
    float dst2 = GetDistance2(address, root, crds);
    if(nestndx < 0 || dst2 < nestdst) {
        nestdst = dst2;
        //nestndx = root;
        nestndx = GetOrgndxAt(address, root);
    }
    if(nestdst == 0.0f)
        return;
    float diffc = GetCoordinateAt(address, root, dimndx) - crds[dimndx];
    dimndx = (dimndx + 1) % ndims_;
    NNRecursive(
        address,
        (0 < diffc)? GetBranchAt(address, root, ptrLeft_): GetBranchAt(address, root, ptrRight_),
        crds, dimndx, nvisited, nestndx, nestdst);
    if(nestdst <= diffc * diffc)
        return;//the other branch cannot hold a better candidate
    NNRecursive(
        address,
        (0 < diffc)? GetBranchAt(address, root, ptrRight_): GetBranchAt(address, root, ptrLeft_),
        crds, dimndx, nvisited, nestndx, nestdst);
}

// -------------------------------------------------------------------------
// NNIterative: find nearest node to the point with the given coordinates 
// crds[] iteratively;
// address, beginning address of a structure;
// root, root node;
// nvisited, number of visited nodes;
// nestndx, index of the nearest node;
// nestdst, distance to the nearest node;
// 
void PMBatchStrDataIndex::NNIterative(
    int address, int root, FPTYPE crds[ndims_], int dimndx,
    int& nvisited, int& nestndx, float& nestdst) const
{
    constexpr size_t stacksize = 17;//32;
    std::vector<std::tuple<int,int,float>> snodesim;//stack-of-nodes simulator
    snodesim.reserve(stacksize);

    bool limitreached = false;

    // if the current node is null and the stack is also empty, we are done
    while(!snodesim.empty() || 0 <= root)
    {
        if(!limitreached) {
            limitreached = (snodesim.size() >= stacksize);
            if(limitreached)
                warning("PMBatchStrDataIndex::NNIterative: Stack size limit reached.");
        }

        if(0 <= root) {
            nvisited++;
            float dst2 = GetDistance2(address, root, crds);
            if(nestndx < 0 || dst2 < nestdst) {
                nestdst = dst2;
                //nestndx = root;
                nestndx = GetOrgndxAt(address, root);
            }
            if(nestdst == 0.0f)
                return;
            float diffc = GetCoordinateAt(address, root, dimndx) - crds[dimndx];
            int branch = (0 < diffc)? ptrLeft_: ptrRight_;
            dimndx = (dimndx + 1) % ndims_;
            if(!limitreached)
                snodesim.push_back(std::make_tuple(root,dimndx,diffc));
            root = GetBranchAt(address, root, branch);
        }
        else {
            bool cond = false;
            float diffc = 0.0f;
            for(; !snodesim.empty() && !cond;) {
                auto tuple = snodesim.back();
                root = std::get<0>(tuple);
                dimndx = std::get<1>(tuple);
                diffc = std::get<2>(tuple);
                snodesim.pop_back();
                cond = (diffc * diffc < nestdst);
            }
            if(!cond) break;
            int branch = (0 < diffc)? ptrRight_: ptrLeft_;
            root = GetBranchAt(address, root, branch);
        }
    }
}

// -------------------------------------------------------------------------
// NNNaive: naive version of finding nearest node to the point with the 
// given coordinates crds[];
// address, beginning address of a structure;
// len, length of a structure;
// nvisited, number of visited nodes;
// nestndx, index of the nearest node;
// nestdst, distance to the nearest node;
// 
void PMBatchStrDataIndex::NNNaive(
    int address, int len, FPTYPE crds[ndims_],
    int& nvisited, int& nestndx, float& nestdst) const
{
    for(int n = 0; n < len; n++) {
        float dst2 = GetDistance2(address, n, crds);
        nvisited++;
        if(nestndx < 0 || dst2 < nestdst) {
            nestdst = dst2;
            //nestndx = n;
            nestndx = GetOrgndxAt(address, n);
        }
        if(nestdst == 0.0f)
            return;
    }
}

// -------------------------------------------------------------------------
// Search: search the point with the coordinates crds where the object strs 
// serves as a container for meta-data (length, total number of positions)
// 
void PMBatchStrDataIndex::Search(const PMBatchStrData& strs, FPTYPE crds[ndims_]) const
{
    static const std::string preamb = "PMBatchStrDataIndex::Search: ";
    char msgbuf[KBYTE];
    size_t nstts = strs.GetNoStructsWritten();
    size_t nposs = strs.GetNoPositsWritten();

    //local function to print info in the buffer
    std::function<void(int,const char*,int,int,float)> lfPrintInfo = 
        [this,&msgbuf](int address, const char* method, int nvisited, int nestndx, float nestdst) {
            if(0 <= nestndx)
                sprintf(msgbuf,
                    "%s: nvisited= %d nestndx= %d nestdst= %.3f  (%.3f, %.3f, %.3f)", 
                    method, nvisited, nestndx, /*sqrtf*/(nestdst),
                    GetCoordinateAt(address, nestndx, 0),
                    GetCoordinateAt(address, nestndx, 1),
                    GetCoordinateAt(address, nestndx, 2)
                );
            else
                sprintf(msgbuf,
                    "%s: nvisited= %d nestndx= %d nestdst= %.3f  (Not found/error)",
                    method, nvisited, nestndx, nestdst);
    };

    size_t nposprocessed = 0;
    int address = 0;

    for(size_t n = 0; n < nstts; n++)
    {
        int len = (int)PMBatchStrData::GetLengthAt(strs.bdbCpmbeg_, n);

        if(len < 1) continue;

        nposprocessed += len;

        if(nposs < nposprocessed)
            throw MYRUNTIME_ERROR2(
            preamb + "Invalid source data.", CRITICAL);

        int root = len >> 1;

        sprintf(msgbuf, "Search tests for structure of length %d:", len);
        message(msgbuf, true, 0);

        int nvisited = 0;
        int nestndx = -1;
        float nestdst = 999.9f;

        NNRecursive(address, root, crds, 0/*dimndx*/, nvisited, nestndx, nestdst);
        lfPrintInfo(address, "NNRecursive", nvisited, nestndx, nestdst);
        message(msgbuf, true, 0);

        nvisited = 0;
        nestndx = -1;
        nestdst = 999.9f;

        NNIterative(address, root, crds, 0/*dimndx*/, nvisited, nestndx, nestdst);
        lfPrintInfo(address, "NNIterative", nvisited, nestndx, nestdst);
        message(msgbuf, true, 0);

        nvisited = 0;
        nestndx = -1;
        nestdst = 999.9f;

        NNNaive(address, len, crds, nvisited, nestndx, nestdst);
        lfPrintInfo(address, "NNNaive", nvisited, nestndx, nestdst);
        message(msgbuf, true, 0);

        address += len;
    }
}

// -------------------------------------------------------------------------
// Print: print data in the buffers to stdout for testing
// 
void PMBatchStrDataIndex::Print() const
{
    size_t nposs = GetNoPositsWritten();

    fprintf(stdout, "#total index positions: %zu\n", nposs);

    for(size_t i = 0; i < nposs; i++) {
        float coord[pmv2DNoElems] = {
            ((FPTYPE*)(bdbCpmbeg_[pmv2DNdxCoords+0]))[i],
            ((FPTYPE*)(bdbCpmbeg_[pmv2DNdxCoords+1]))[i],
            ((FPTYPE*)(bdbCpmbeg_[pmv2DNdxCoords+2]))[i]
        };
        int left = ((INTYPE*)(bdbCpmbeg_[pmv2DNdxLeft]))[i];
        int right = ((INTYPE*)(bdbCpmbeg_[pmv2DNdxRight]))[i];
        int orgndx = ((INTYPE*)(bdbCpmbeg_[pmv2DNdxOrgndx]))[i];
        fprintf(stdout, "  %8.3f %8.3f %8.3f  l:%d r:%d  org:%d\n", 
            coord[0], coord[1], coord[2], left, right, orgndx);
    }

    fprintf(stdout, "\n\n");
}
