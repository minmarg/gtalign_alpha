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

#include <fstream>

#if defined(GPUINUSE) && 0
#   include <cuda_runtime_api.h>
#   include "libmycu/cucom/cucommon.h"
#endif

#include "libutil/CLOptions.h"
#include "libgenp/gproc/gproc.h"
#include "PMBatchStrDataIndex.h"
#include "PMBatchStrData.h"
#include "PMBatchStrData.inl"

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// template helpers: assignemnt of the block of structure-specific field 
// using temporary buffer
template <typename T, int field>
void PMBatchStrData::sort_helper_ssfields_assign(size_t nstts)
{
    const size_t szT = TPM2DVectorFieldSize::szvfs_[field];

    for(size_t n = 0; n < nstts; n++) {
        int ndx = tmpndxs_[n];
        ((T*)(tmpbdbCdata_.get()))[n] = ((T*)(bdbCpmbeg_[field]))[ndx];
    }

    if((size_t)(bdbCpmend_[field]-bdbCpmbeg_[field]) != nstts * szT)
        throw MYRUNTIME_ERROR2(
        "sort_helper_ssfields_assign: Invalid size of copied data.",
        CRITICAL);

    //copy the whole block of data from temporary buffer to the original location:
    memcpy(bdbCpmbeg_[field], tmpbdbCdata_.get(), nstts * szT);
}

// specific instantiation for field pps2DDist
template<>
void PMBatchStrData::sort_helper_ssfields_assign<LNTYPE,pps2DDist>(size_t nstts)
{
    const size_t szT = TPM2DVectorFieldSize::szvfs_[pps2DDist];
    LNTYPE dstval = 0;

    for(size_t n = 0; n < nstts; n++) {
        //NOTE: bdbCpmbeg_ assumed to contain updated data
        INTYPE lenatndx = ((INTYPE*)(bdbCpmbeg_[pps2DLen]))[n];
        ((LNTYPE*)(tmpbdbCdata_.get()))[n] = dstval;
        dstval += lenatndx;
    }

    if((size_t)(bdbCpmend_[pps2DDist]-bdbCpmbeg_[pps2DDist]) != nstts * szT)
        throw MYRUNTIME_ERROR2(
        "sort_helper_ssfields_assign: Invalid size of copied data.",
        CRITICAL);

    //copy the whole block of data from temporary buffer to the original location:
    memcpy(bdbCpmbeg_[pps2DDist], tmpbdbCdata_.get(), nstts * szT);
}

// -------------------------------------------------------------------------
// assignemnt of the block of position-specific field 
// using temporary buffer
template <int field>
void PMBatchStrData::sort_helper_psfields_assign(size_t nstts)
{
    const size_t szfld = TPM2DVectorFieldSize::szvfs_[field];
    char* ptmpdat = tmpbdbCdata_.get();

    for(size_t n = 0; n < nstts; n++) {
        int ndx = tmpndxs_[n];
        LNTYPE dstatndx = ((LNTYPE*)(bdbCpmbeg_[pps2DDist]))[ndx];
        INTYPE lenatndx = ((INTYPE*)(bdbCpmbeg_[pps2DLen]))[ndx];
        size_t szdstatndx = dstatndx * szfld;
        size_t szfatndx = lenatndx * szfld;
        memcpy(ptmpdat, bdbCpmbeg_[field] + szdstatndx, szfatndx);
        ptmpdat += szfatndx;
    }

    size_t szcopied = (size_t)(ptmpdat - tmpbdbCdata_.get());

    if((size_t)(bdbCpmend_[field]-bdbCpmbeg_[field]) != szcopied)
        throw MYRUNTIME_ERROR2(
        "sort_helper_psfields_assign: Invalid size of copied data.",
        CRITICAL);

    //copy the whole block of data from temporary buffer to the original location:
    memcpy(bdbCpmbeg_[field], tmpbdbCdata_.get(), szcopied);
}

// -------------------------------------------------------------------------
// Sort: sort data by length in-place; NOTE: used when all chunk data has 
// been compiled (after finalization)
//
void PMBatchStrData::Sort()
{
    if(!tmpbdbCdata_)
        throw MYRUNTIME_ERROR2(
        "Sort: Memory access error.", CRITICAL);

    size_t nstts = GetNoStructsWritten();

    if(nbdbCptrdescs_ < nstts)
        throw MYRUNTIME_ERROR2(
        "Sort: Memory access error.", CRITICAL);

    tmpndxs_.resize(nstts);
    //indices from 0 to #structures
    std::iota(tmpndxs_.begin(), tmpndxs_.end(), 0);

    std::sort(tmpndxs_.begin(), tmpndxs_.end(),
        [this](size_t n1, size_t n2) {
            //sort by length in descending order
            return 
                ((INTYPE*)(bdbCpmbeg_[pps2DLen]))[n1] > 
                ((INTYPE*)(bdbCpmbeg_[pps2DLen]))[n2];
        });

    //write sorted data in the temporary buffer;
    //NOTE: start with position-specific fields since non-overwritten 
    // structure-specific fields are required here:
    sort_helper_psfields_assign<pmv2DCoords>(nstts);
    sort_helper_psfields_assign<pmv2DCoords+1>(nstts);
    sort_helper_psfields_assign<pmv2DCoords+2>(nstts);
    sort_helper_psfields_assign<pmv2D_Ins_Ch_Ord>(nstts);
    sort_helper_psfields_assign<pmv2DResNumber>(nstts);
    sort_helper_psfields_assign<pmv2Drsd>(nstts);
    //NOTE: SS calculated and assigned during the processing
    //sort_helper_psfields_assign<pmv2Dss>(nstts);

    //structure-specific fields in the end:
    sort_helper_ssfields_assign<INTYPE,pps2DLen>(nstts);
    sort_helper_ssfields_assign<INTYPE,pps2DType>(nstts);
    //NOTE: update field pps2DDist after pps2DLen has been updated!
    sort_helper_ssfields_assign<LNTYPE,pps2DDist>(nstts);

    //finally, rearrange pointers to structure descriptions
    char** tmptrdescs = reinterpret_cast<char**>(tmpbdbCdata_.get());

    for(size_t n = 0; n < nstts; n++) {
        int ndx = tmpndxs_[n];
        tmptrdescs[n] = bdbCptrdescs_[ndx];
    }
    //copy pointers back to the original location
    for(size_t n = 0; n < nstts; n++)
        bdbCptrdescs_[n] = tmptrdescs[n];
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// Local type declarations:
// local structure-specific field assigner
struct TL_ss_field_assigner {
    char** l_bdbpmbeg_, **l_bdbpmend_;
    TL_ss_field_assigner(char** bdbpmbeg, char** bdbpmend)
    : l_bdbpmbeg_(bdbpmbeg), l_bdbpmend_(bdbpmend) {}
    template<typename T, int field>
    void assign(unsigned int newndx, unsigned int n) {
        if(newndx < n)
            ((T*)(l_bdbpmbeg_[field]))[newndx] = ((T*)(l_bdbpmbeg_[field]))[n];
        l_bdbpmend_[field] = (char*)(((T*)(l_bdbpmbeg_[field])) + newndx + 1);
    }
};

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
// indexed in filterdata
//
void PMBatchStrData::FilterStructs(
    const char** bdbdesc, char** bdbpmbeg, char** bdbpmend,
    const unsigned int* filterdata)
{
    MYMSG("PMBatchStrData::FilterStructs",4);
    static const std::string preamb = "PMBatchStrData::FilterStructs: ";

    const unsigned int ndbstrs = PMBatchStrData::GetNoStructs(bdbpmbeg, bdbpmend);
    TL_ss_field_assigner ss_assigner(bdbpmbeg, bdbpmend);
    TL_ps_field_assigner ps_assigner(bdbpmbeg, bdbpmend);

    //no data when all structures are to be filtered out:
    for(int fld = 0; fld < pmv2DTotFlds; fld++)
        bdbpmend[fld] = bdbpmbeg[fld];

    for(unsigned int n = 0; n < ndbstrs; n++) {
        unsigned int length = PMBatchStrData::GetLengthAt(bdbpmbeg, n);
        unsigned int address = PMBatchStrData::GetAddressAt(bdbpmbeg, n);
        //NOTE: data (indices and addresses) are values calculated by inclusive prefix sum!
        unsigned int newndx = filterdata[ndbstrs * fdNewReferenceIndex + n];
        unsigned int newaddr = filterdata[ndbstrs * fdNewReferenceAddress + n];
        if(newndx == 0) continue;
        //adjust the index appropriately (NOTE: newaddr is already valid):
        newndx--;
        //copy the entire structure to a different location in the same address space;
        //structure-specific fields:
        ss_assigner.assign<INTYPE,pps2DLen>(newndx, n);
        ss_assigner.assign<INTYPE,pps2DType>(newndx, n);
        ((LNTYPE*)(bdbpmbeg[pps2DDist]))[newndx] = newaddr;
        bdbpmend[pps2DDist] = (char*)(((LNTYPE*)(bdbpmbeg[pps2DDist])) + newndx + 1);
        //position-specific fields:
        ps_assigner.assign<FPTYPE,pmv2DCoords>(newaddr, address, length);
        ps_assigner.assign<FPTYPE,pmv2DCoords+1>(newaddr, address, length);
        ps_assigner.assign<FPTYPE,pmv2DCoords+2>(newaddr, address, length);
        ps_assigner.assign<LNTYPE,pmv2D_Ins_Ch_Ord>(newaddr, address, length);
        ps_assigner.assign<INTYPE,pmv2DResNumber>(newaddr, address, length);
        ps_assigner.assign<CHTYPE,pmv2Drsd>(newaddr, address, length);
        ps_assigner.assign<CHTYPE,pmv2Dss>(newaddr, address, length);
        //descriptions:
        bdbdesc[newndx] = bdbdesc[n];
    }
    //NOTE: see TODO in the header for ovhd!
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// Copy: copy essential data from source arrays
// 
void PMBatchStrData::Copy(
    const char** bdbdesc,
    char* const * const bdbpmbeg, char* const * const bdbpmend)
{
    MYMSG("PMBatchStrData::Copy",4);
    static const std::string preamb = "PMBatchStrData::Copy: ";
    const size_t ndbstrs = PMBatchStrData::GetNoStructs(bdbpmbeg, bdbpmend);
    const size_t nposits = PMBatchStrData::GetNoPosits(bdbpmbeg, bdbpmend);
    if(maxnstrs_ < ndbstrs || nbdbCptrdescs_ < ndbstrs)
        throw MYRUNTIME_ERROR2(preamb + "Too many structures.", CRITICAL);
    if(maxdatalen_ < nposits)
        throw MYRUNTIME_ERROR2(preamb + "Too many total positions.", CRITICAL);

    //{{actual copy
    int n = 0;
    for(; n < pps2DStrFlds; n++) {
        // std::copy(bdbpmbeg[n], bdbpmbeg[n] + (bdbpmend[n] - bdbpmbeg[n]), bdbCpmbeg_[n]);
        memcpy(bdbCpmbeg_[n], bdbpmbeg[n], (bdbpmend[n] - bdbpmbeg[n]));
        bdbCpmend_[n] = bdbCpmbeg_[n] + (bdbpmend[n] - bdbpmbeg[n]);
        bdbCpmendovhd_[n] = bdbCpmend_[n];
    }
    for(; n < pmv2DTotFlds; n++) {
        // std::copy(bdbpmbeg[n], bdbpmbeg[n] + (bdbpmend[n] - bdbpmbeg[n]), bdbCpmbeg_[n]);
        memcpy(bdbCpmbeg_[n], bdbpmbeg[n], (bdbpmend[n] - bdbpmbeg[n]));
        bdbCpmend_[n] = bdbCpmbeg_[n] + (bdbpmend[n] - bdbpmbeg[n]);
        bdbCpmendovhd_[n] = bdbCpmend_[n];
    }
    //copy descriptions; assume valid pointers
    char** ptrdescs = bdbCptrdescs_.get();
    for(size_t di = 0; di < ndbstrs; di++) {
        ptrdescs[di] = bdbCdescs_.get() + di * PMBSdatDEFDESCLEN;
        strcpy(ptrdescs[di], bdbdesc[di]);
    }
    //}}
}

// -------------------------------------------------------------------------
// CopyFrom: copy essential data, including bdbCpmendovhd_, from source to
// this object
// 
void PMBatchStrData::CopyFrom(const PMBatchStrData& bsd)
{
    MYMSG("PMBatchStrData::CopyFrom",4);
    static const std::string preamb = "PMBatchStrData::CopyFrom: ";
    const size_t ndbstrs = bsd.GetNoStructsWritten();
    const size_t nposits = bsd.GetNoPositsWritten();
    const size_t npositsovhd = bsd.GetNoPositsOvhd();
    if(maxnstrs_ < ndbstrs || nbdbCptrdescs_ < ndbstrs)
        throw MYRUNTIME_ERROR2(preamb + "Too many structures.", CRITICAL);
    if(maxdatalen_ < nposits)
        throw MYRUNTIME_ERROR2(preamb + "Too many total positions.", CRITICAL);

    //{{actual copy
    int n = 0;
    for(; n < pps2DStrFlds; n++) {
        std::copy(bsd.bdbCpmbeg_[n], bsd.bdbCpmend_[n], bdbCpmbeg_[n]);
        bdbCpmend_[n] = bdbCpmbeg_[n] + (bsd.bdbCpmend_[n] - bsd.bdbCpmbeg_[n]);
        memcpy(bdbCpmend_[n], bsd.bdbCpmend_[n], TPM2DVectorFieldSize::szvfs_[n]);
        bdbCpmendovhd_[n] = bdbCpmend_[n] + (bsd.bdbCpmendovhd_[n] - bsd.bdbCpmend_[n]);
    }
    for(; n < pmv2DTotFlds; n++) {
        std::copy(bsd.bdbCpmbeg_[n], bsd.bdbCpmend_[n], bdbCpmbeg_[n]);
        bdbCpmend_[n] = bdbCpmbeg_[n] + (bsd.bdbCpmend_[n] - bsd.bdbCpmbeg_[n]);
        memcpy(bdbCpmend_[n], bsd.bdbCpmend_[n], TPM2DVectorFieldSize::szvfs_[n] * npositsovhd);
        bdbCpmendovhd_[n] = bdbCpmend_[n] + (bsd.bdbCpmendovhd_[n] - bsd.bdbCpmend_[n]);
    }
    //copy descriptions; assume valid pointers
    char** ptrdescs = bdbCptrdescs_.get();
    char** bsdptrdescs = bsd.bdbCptrdescs_.get();
    for(size_t di = 0; di < ndbstrs; di++) {
        ptrdescs[di] = bdbCdescs_.get() + di * PMBSdatDEFDESCLEN;
        strcpy(ptrdescs[di], bsdptrdescs[di]);
    }
    //ovhd:
    strncpy(ptrdescs[ndbstrs], bsdptrdescs[ndbstrs], PMBSdatDEFDESCLEN);
    //}}
    //file indices for clustering:
    tmpclustfilendxs_ = bsd.tmpclustfilendxs_;
    tmpcluststrndxs_ = bsd.tmpcluststrndxs_;
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// AllocateSpaceForData: allocate space for structure data plus 
// space for the end addresses of structure descriptions
// 
void PMBatchStrData::AllocateSpaceForData()
{
    MYMSG("PMBatchStrData::AllocateSpaceForData",4);
    static const std::string preamb = "PMBatchStrData::AllocateSpaceForData: ";
    //max size of data given max total number of positions (residues)
    size_t szdata = PMBatchStrData::GetPMDataSizeUB(maxdatalen_);
    size_t szovhd = PMBatchStrData::GetPMDataSizeUB(PMBSmaxonestructurelength);

    size_t sizefortmpdata = 
        PCMAX(maxdatasize_, szdata) + //allowed size of data
        PMBSdatalignment * pmv2DTotFlds + //for data alignment
        (maxnstrs_+1) * sizeof(size_t);//description end addresses

    size_t requestedsize = 
        PCMAX(maxdatasize_, szdata) + //allowed size of data
        szovhd + //extra space for one additional structure
        PMBSdatalignment * pmv2DTotFlds + //for data alignment
        (maxnstrs_+1) * sizeof(size_t);//description end addresses

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
            MYCUDACHECK( cudaMallocHost((void**)&h_mpinned, szbdbCdata_) );
            MYCUDACHECKLAST;
            bdbCdata_.reset(h_mpinned);
        }
        else
#endif
            bdbCdata_.reset((char*)my_aligned_alloc(PMBSdatalignment, szbdbCdata_));

        //allocate space for tmp data too
        tmpbdbCdata_.reset((char*)std::malloc(sizefortmpdata));

        if(bdbCdata_.get() == NULL || tmpbdbCdata_.get() == NULL)
            throw MYRUNTIME_ERROR2(
            preamb + "Memory allocation failed.", CRITICAL);

        //reserve space for temporary vector of lengths (once)
        tmpndxs_.reserve(maxnstrs_+1);
        //index vectors for clustering
        tmpclustfilendxs_.reserve(maxnstrs_+1);
        tmpcluststrndxs_.reserve(maxnstrs_+1);
    }

    tmpclustfilendxs_.clear();
    tmpcluststrndxs_.clear();

    size_t szdat = 0, szval = 0;
    char* pbdbCdata = bdbCdata_.get();

    //initialize pointers:
    for(int n = 0; n < pmv2DTotFlds; n++)
    {
        bdbCpmbeg_[n] = bdbCpmend_[n] = bdbCpmendovhd_[n] = pbdbCdata;
        szdat = TPM2DVectorFieldSize::szvfs_[n] * 
                    (maxdatalen_ + PMBSmaxonestructurelength);
        szval = ALIGN_UP(szdat, PMBSdatalignment);
        pbdbCdata += szval;
//         szpm2dvf_[n] = szpm2dvfovhd_[n] = 0;//no data written
    }

    if(bdbCdata_.get() + 
       PCMAX(maxdatasize_, szdata) + szovhd + 
       PMBSdatalignment * pmv2DTotFlds < pbdbCdata)
        throw MYRUNTIME_ERROR2(
        preamb + "Invalid calculated allocation size.",
        CRITICAL);
}

// -------------------------------------------------------------------------
// AllocateSpaceForDescriptions: allocate space for structure descriptions
// 
bool PMBatchStrData::AllocateSpaceForDescriptions()
{
    assert(PMBSdatDEFDESCLEN > 4);

    //NOTE: +1 for one additional structure left to be included in the 
    // next round of batch data:
    size_t requestedsize = (maxnstrs_+1) * PMBSdatDEFDESCLEN;
    if( requestedsize <= szbdbCdescs_ )
        return false;
    char strbuf[BUF_MAX];
    sprintf(strbuf, "PMBatchStrData::AllocateSpaceForDescriptions: "
        "New allocation for descriptions, %zuB", requestedsize);
    MYMSG( strbuf, 3 );
    szbdbCdescs_ = requestedsize;
    bdbCdescs_.reset((char*)std::malloc(szbdbCdescs_));
    if( bdbCdescs_.get() == NULL )
        throw MYRUNTIME_ERROR2(
        "PMBatchStrData::AllocateSpaceForDescriptions: Not enough memory.",
        CRITICAL);
    return true;
}

// -------------------------------------------------------------------------
// AllocateSpaceForDescPtrs: allocate space for pointers to the 
// descriptions of each structure
// 
void PMBatchStrData::AllocateSpaceForDescPtrs()
{
    static const std::string preamb = "PMBatchStrData::AllocateSpaceForDescPtrs: ";
    //NOTE: +1 for one additional structure left to be icluded in the 
    // next round of batch data:
    size_t requestedno = maxnstrs_+1;

    if(nbdbCptrdescs_ < requestedno)
    {
        char strbuf[BUF_MAX];
        sprintf(strbuf, "%sNew number of description pointers, %zu", 
                preamb.c_str(), requestedno);
        MYMSG( strbuf, 3 );
        nbdbCptrdescs_ = requestedno;
        bdbCptrdescs_.reset(new char*[nbdbCptrdescs_+1]);//NOTE +1
        if(bdbCptrdescs_.get() == NULL)
            throw MYRUNTIME_ERROR2(preamb + "Not enough memory.",
            CRITICAL);
    }

    if(bdbCdescs_.get() == NULL)
        throw MYRUNTIME_ERROR2(preamb + "Space for descriptions not allocated.",
        CRITICAL);

    //initialize pointers in advance and once:
    //NOTE: bdbCdescs_ assumed to have been initialized already
    char** ptrdescs = bdbCptrdescs_.get();
    int n = 0;

    for(; n < (int)requestedno; n++)
        ptrdescs[n] = bdbCdescs_.get() + n * PMBSdatDEFDESCLEN;

    ptrdescs[n] = NULL;
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// Serialize: serialize packed structures;
// eod, end-of-data flag
// 
void PMBatchStrData::Serialize(
    const std::string& fname,
    // const PMBatchStrDataIndex& index,
    const size_t nagents,
    const size_t eod) const
{
    MYMSG("PMBatchStrData::Serialize", 3);
    static const std::string preamb = "PMBatchStrData::Serialize: ";

    std::ofstream fp(fname.c_str(), std::ios::binary|std::ios::out);

    if(fp.bad() || fp.fail())
        throw MYRUNTIME_ERROR(
            preamb + "Failed to open file for writing: " + fname);

    //write a variable to file:
    std::function<void(size_t)> lfWriteVar = [&fp,&fname](size_t var) {
        size_t bytes = sizeof(var);
        fp.write(reinterpret_cast<const char*>(&var), bytes);
        if(fp.bad()) throw MYRUNTIME_ERROR(preamb + "Write failed: " + fname);
    };

    lfWriteVar(eod);
    lfWriteVar(nagents);
    lfWriteVar(maxdatasize_);
    lfWriteVar(maxdatalen_);
    lfWriteVar(maxnstrs_);

    lfWriteVar(szbdbCdata_);
    lfWriteVar(szbdbCdescs_);
    lfWriteVar(nbdbCptrdescs_);

    fp.write(bdbCdata_.get(), szbdbCdata_);
    if(fp.bad()) throw MYRUNTIME_ERROR(preamb + "Write failed: " + fname);

    for(int n = 0; n < pmv2DTotFlds; n++) {
        lfWriteVar((size_t)(bdbCpmbeg_[n] - bdbCdata_.get()));
        lfWriteVar((size_t)(bdbCpmend_[n] - bdbCdata_.get()));
    }

    fp.write(bdbCdescs_.get(), szbdbCdescs_);
    if(fp.bad()) throw MYRUNTIME_ERROR(preamb + "Write failed: " + fname);
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// Deserialize: deserialize packed structures;
// return end-of-data flag;
// NOTE: this method should be called after a call to AllocateSpace!
// NOTE: should be called before Sort since bdbCptrdescs_ aren't rearranged!
// 
size_t PMBatchStrData::Deserialize(
    const std::string& fname,
    // PMBatchStrDataIndex& index,
    const size_t nagents)
{
    MYMSG("PMBatchStrData::Deserialize", 3);
    static const std::string preamb = "PMBatchStrData::Deserialize: ";
    size_t tmpvar1, tmpvar2, eod;

    std::ifstream fp(fname.c_str(), std::ios::binary);

    if(!fp)
        throw MYRUNTIME_ERROR(preamb + "Failed to open file: " + fname);

    if(fp.rdbuf())
        fp.rdbuf()->pubsetbuf(tmpreadbuff_, PMBatchStrData_TMPBUFFSIZE);

    //write a variable to file:
    std::function<void(size_t*)> lfReadVar = [&fp,&fname](size_t* var) {
        size_t bytes = sizeof(*var);
        fp.read(reinterpret_cast<char*>(var), bytes);
        if(fp.bad() || (size_t)fp.gcount() != bytes)
            throw MYRUNTIME_ERROR(preamb + "Ill-formed file; Read failed: " + fname);
    };

    lfReadVar(&eod);
    if(eod != 0 && eod != 1)
        throw MYRUNTIME_ERROR(preamb + "Invalid EOD flag; File: " + fname);

    lfReadVar(&tmpvar1);
    if(tmpvar1 != nagents)
        throw MYRUNTIME_ERROR(preamb + "Invalid #agents; File: " + fname);

    lfReadVar(&tmpvar1);
    if(tmpvar1 != maxdatasize_)
        throw MYRUNTIME_ERROR(preamb + "Invalid data size; File: " + fname);

    lfReadVar(&tmpvar1);
    if(tmpvar1 != maxdatalen_)
        throw MYRUNTIME_ERROR(preamb + "Invalid total length; File: " + fname);

    lfReadVar(&tmpvar1);
    if(tmpvar1 != maxnstrs_)
        throw MYRUNTIME_ERROR(preamb + "Invalid #structures; File: " + fname);

    lfReadVar(&tmpvar1);
    if(tmpvar1 != szbdbCdata_)
        throw MYRUNTIME_ERROR(preamb + "Invalid size; File: " + fname);

    lfReadVar(&tmpvar1);
    if(tmpvar1 != szbdbCdescs_)
        throw MYRUNTIME_ERROR(preamb + "Invalid size of descriptions; File: " + fname);

    lfReadVar(&tmpvar1);
    if(tmpvar1 != nbdbCptrdescs_)
        throw MYRUNTIME_ERROR(preamb + "Invalid #descriptions; File: " + fname);

    fp.read(bdbCdata_.get(), szbdbCdata_);
    if(fp.bad() || (size_t)fp.gcount() != szbdbCdata_)
        throw MYRUNTIME_ERROR(preamb + "Read failed: " + fname);

    for(int n = 0; n < pmv2DTotFlds; n++) {
        lfReadVar(&tmpvar1); lfReadVar(&tmpvar2);
        if(szbdbCdata_ <= tmpvar1 || szbdbCdata_ <= tmpvar2)
            throw MYRUNTIME_ERROR(preamb + "Corrupted file: " + fname);
        bdbCpmbeg_[n] = bdbCdata_.get() + tmpvar1;
        bdbCpmend_[n] = bdbCpmendovhd_[n] = bdbCdata_.get() + tmpvar2;
    }

    fp.read(bdbCdescs_.get(), szbdbCdescs_);
    if(fp.bad() || (size_t)fp.gcount() != szbdbCdescs_)
        throw MYRUNTIME_ERROR(preamb + "Descriptions read failed: " + fname);

    return eod;
}



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// Print: print data in the buffers to stdout for testing
// 
void PMBatchStrData::Print() const
{
    size_t nstts = GetNoStructsWritten();
    size_t nposs = GetNoPositsWritten();
    size_t npossrun = 0;

    fprintf(stdout, "#structures: %zu  #total positions: %zu\n", nstts, nposs);

    for(size_t n = 0; n < nstts; n++) {
        int slen = ((INTYPE*)(bdbCpmbeg_[pps2DLen]))[n];
        int stype = ((INTYPE*)(bdbCpmbeg_[pps2DType]))[n];
        unsigned int dbeg = ((LNTYPE*)(bdbCpmbeg_[pps2DDist]))[n];
        char** ptrdescs = bdbCptrdescs_.get();
        if(!ptrdescs)
            throw MYRUNTIME_ERROR(
                "PMBatchStrData::Print: Invalid description address.");
        npossrun += slen;
        fprintf(stdout, "\n\n STRUCT: \"%s\" LEN: %d TYPE: %d ADDR: %u\n", 
            ptrdescs[n], slen, stype, dbeg);
        for(int i = 0; i < slen; i++) {
            float coord[pmv2DNoElems] = {
                ((FPTYPE*)(bdbCpmbeg_[pmv2DCoords+0]))[i+dbeg],
                ((FPTYPE*)(bdbCpmbeg_[pmv2DCoords+1]))[i+dbeg],
                ((FPTYPE*)(bdbCpmbeg_[pmv2DCoords+2]))[i+dbeg]
            };
            unsigned int inschord = ((LNTYPE*)(bdbCpmbeg_[pmv2D_Ins_Ch_Ord]))[i+dbeg];
            char inscode = PM2D_GET_INSCODE(inschord);
            char chain = PM2D_GET_CHID(inschord);
            char chord = PM2D_GET_CHORD(inschord);
            int resnum = ((INTYPE*)(bdbCpmbeg_[pmv2DResNumber]))[i+dbeg];
            char rsdcode = ((CHTYPE*)(bdbCpmbeg_[pmv2Drsd]))[i+dbeg];
            fprintf(stdout, "  %c %c %c(%d) %5d  %8.3f %8.3f %8.3f  %.8x\n", 
                rsdcode, inscode, chain, chord, resnum,
                coord[0], coord[1], coord[2],  inschord);
        }
    }

    fprintf(stdout, "\n#total positions: %zu  #calculated positions: %zu\n",
        nposs, npossrun);
}

// -------------------------------------------------------------------------
// Print: print data in the buffers to stdout for testing
// 
void PMBatchStrData::Print(char* const * const bdbpmbeg, char* const * const bdbpmend)
{
    size_t nstts = GetNoStructs(bdbpmbeg, bdbpmend);
    size_t nposs = GetNoPosits(bdbpmbeg, bdbpmend);
    size_t npossrun = 0;

    fprintf(stdout, "#structures: %zu  #total positions: %zu\n", nstts, nposs);

    for(size_t n = 0; n < nstts; n++) {
        int slen = ((INTYPE*)(bdbpmbeg[pps2DLen]))[n];
        int stype = ((INTYPE*)(bdbpmbeg[pps2DType]))[n];
        unsigned int dbeg = ((LNTYPE*)(bdbpmbeg[pps2DDist]))[n];
        npossrun += slen;
        fprintf(stdout, "\n\n STRUCT LEN: %d TYPE: %d ADDR: %u\n", 
            slen, stype, dbeg);
        for(int i = 0; i < slen; i++) {
            float coord[pmv2DNoElems] = {
                ((FPTYPE*)(bdbpmbeg[pmv2DCoords+0]))[i+dbeg],
                ((FPTYPE*)(bdbpmbeg[pmv2DCoords+1]))[i+dbeg],
                ((FPTYPE*)(bdbpmbeg[pmv2DCoords+2]))[i+dbeg]
            };
            unsigned int inschord = ((LNTYPE*)(bdbpmbeg[pmv2D_Ins_Ch_Ord]))[i+dbeg];
            char inscode = PM2D_GET_INSCODE(inschord);
            char chain = PM2D_GET_CHID(inschord);
            char chord = PM2D_GET_CHORD(inschord);
            int resnum = ((INTYPE*)(bdbpmbeg[pmv2DResNumber]))[i+dbeg];
            char rsdcode = ((CHTYPE*)(bdbpmbeg[pmv2Drsd]))[i+dbeg];
            char ssacode = ((CHTYPE*)(bdbpmbeg[pmv2Dss]))[i+dbeg];
            fprintf(stdout, "  %c %c %c(%d) %5d  %8.3f %8.3f %8.3f  %.8x  '%c'\n", 
                rsdcode, inscode, chain, chord, resnum,
                coord[0], coord[1], coord[2],  inschord,  ssacode);
        }
    }

    fprintf(stdout, "\n#total positions: %zu  #calculated positions: %zu\n",
        nposs, npossrun);

    fflush(stdout);
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// SequenceSimilarityOvhd: return true if the sequence similarity between the
// structure in Ovhd and any of the queries in given blocks is above the threshold;
// queryblocks, #query blocks in data (outer-most dimension);
// querypmbegs, beginning addresses of the query fields;
// querypmends, end addresses of the query fields;
// 
bool PMBatchStrData::SequenceSimilarityOvhd(
    const int queryblocks,
    char* const * const * const querypmbegs,
    char* const * const * const querypmends) const
{
    enum {
        DIMD = 128,
        EDGE = 20//unprocessed tail length 
    };

    MYMSG("PMBatchStrData::SequenceSimilarityOvhd", 5);
    // static const std::string preamb = "PMBatchStrData::SequenceSimilarityOvhd: ";
    static const float seqsimthrscore = CLOptions::GetP_PRE_SIMILARITY();
    //sequence similarity signal: true if Ovhd is similar to any of the queries:
    bool signal = false;

    const int dbstrlen = GetNoPositsOvhd();//#positions (length) in Ovhd
    const int dbstrdst = 0;//distance to data (Ovhd)
    constexpr int step = 2;//use a step of 2 for speed!
    char rfnRE[DIMD];
    char qryRE[DIMD];
    float scores[DIMD];
    float pxmins[DIMD];
    float tmp[DIMD];

    if(seqsimthrscore <= 0.0f || queryblocks < 1 || !querypmbegs || !querypmends) return true;
    // if(dbstrlen <= EDGE) return true;///NOTE: length constraint!

    for(int bi = 0; bi < queryblocks && !signal; bi++) {
        int nqstrs = PMBatchStrData::GetNoStructs(querypmbegs[bi], querypmends[bi]);
        for(int qi = 0; qi < nqstrs && !signal; qi++) {
            char* const * const querypmbeg = querypmbegs[bi];
            const int qrylen = PMBatchStrData::GetLengthAt(querypmbeg, qi);
            const int qrydst = PMBatchStrData::GetAddressAt(querypmbeg, qi);

            // if(qrylen <= EDGE) return true;///NOTE: length constraint!

            int ddend = mymax(0, qrylen - EDGE);
            for(int dd = 0; dd < ddend && !signal; dd += step) {
                const int qrybegpos = dd;
                const int rfnbegpos = 0;
                signal = CheckAlignmentScore<DIMD>(
                    seqsimthrscore, qrydst, qrylen, dbstrdst, dbstrlen,
                    qrybegpos, rfnbegpos, querypmbeg, bdbCpmend_,
                    rfnRE, qryRE, scores, pxmins, tmp);
            }
            ddend = mymax(0, dbstrlen - EDGE);
            for(int dd = step; dd < ddend && !signal; dd += step) {
                const int qrybegpos = 0;
                const int rfnbegpos = dd;
                signal = CheckAlignmentScore<DIMD>(
                    seqsimthrscore, qrydst, qrylen, dbstrdst, dbstrlen,
                    qrybegpos, rfnbegpos, querypmbeg, bdbCpmend_,
                    rfnRE, qryRE, scores, pxmins, tmp);
            }
        }
    }

    return signal;
}
