/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __PMBatchStrData_h__
#define __PMBatchStrData_h__

#include "libutil/mybase.h"

#include <string.h>

#include <vector>
#include <memory>
#include <algorithm>

#include "tsafety/TSCounterVar.h"
#include "libutil/CLOptions.h"
#include "PM2DVectorFields.h"
#include "DRDataDeleter.h"
#include "gdconst.h"

class PMBatchStrDataIndex;

// -------------------------------------------------------------------------
//
enum PMBatchStrDataCnsts {
    PMBSdatalignment = CACHECLINESIZE,
    PMBSdatDEFDESCLEN = DEFAULT_DESCRIPTION_LENGTH,//default description length (>4)
    //max structure length, i.e. allowed number of support atoms:
    //NOTE: encoding of structure positions is currently limited to one word (2 bytes):
    //NOTE: assign max length to 2^16-2 when recording DP cell coordinates: value of 
    //NOTE: 2^16-1 is left for the stop marker;
    PMBSmaxonestructurelength = 65535 //100 * ONEK
};

enum PMBatchStrDataMolType {
    PMBSMTProtein = 0,//Protein
    PMBSMTRNA = 1//RNA
};

// -------------------------------------------------------------------------
// PMBatchStrData: Complete batch structure data for parallel processing
//
class PMBatchStrData {
public:
    enum TPMBSDFinRetCode {
        pmbsdfEmpty=1,//no structure data (0 for compatibility)
        pmbsdfShort,//structure too short
        pmbsdfAbandoned,//structure is to large to be kept
        pmbsdfLowSimilarity,//low sequence similarity of structures
        pmbsdfLimits,//structure in the temporary buffers due to space limits
        pmbsdfWritten//structure has been written in the buffers
    };
    enum {
        PMBatchStrData_TMPBUFFSIZE = 4096
    };
public:
    PMBatchStrData()
    :   tmpbdbCdata_(nullptr),
        bdbCdata_(nullptr),
        bdbCdescs_(nullptr),
        bdbCptrdescs_(nullptr),
        szbdbCdata_(0),
        szbdbCdescs_(0),
        nbdbCptrdescs_(0),
        maxdatasize_(0),
        maxdatalen_(0),
        maxnstrs_(0)
    {
        memset( bdbCpmbeg_, 0, pmv2DTotFlds * sizeof(void*));
        memset( bdbCpmend_, 0, pmv2DTotFlds * sizeof(void*));
        memset( bdbCpmendovhd_, 0, pmv2DTotFlds * sizeof(void*));
//         memset( szpm2dvf_, 0, pmv2DTotFlds * sizeof(size_t));
//         memset( szpm2dvfovhd_, 0, pmv2DTotFlds * sizeof(size_t));
    }

    ~PMBatchStrData() {}

    // *** METHODS ***
    void AllocateSpace(size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs);

    //serialize/deserialize
    void Serialize(const std::string&, const size_t nagents, const size_t eod) const;
    size_t Deserialize(const std::string&, const size_t nagents);

    void Fallback() {FallbackOvhdPtrs();}

    //sort by length; used when all chunk data has been compiled
    void Sort();

    void Copy(const char** bdbdesc, char* const * const bdbpmbeg, char* const * const bdbpmend);
    void CopyFrom(const PMBatchStrData& bsd);

    size_t GetPMDataSize();
    static size_t GetPMDataSize1(size_t structlen);
    static size_t GetPMDataSizeUB(size_t totallen);

    static bool ContainsData(char* const * const bdbpmbeg, char* const * const bdbpmend) {
        return 
            bdbpmbeg[pps2DLen] < bdbpmend[pps2DLen] &&
            bdbpmbeg[pps2DDist] < bdbpmend[pps2DDist] &&
            bdbpmbeg[pps2DType] < bdbpmend[pps2DType] &&
            bdbpmbeg[pmv2DCoords] < bdbpmend[pmv2DCoords];
    }

    bool ContainsData() const {//whether data is present
        return bdbCpmbeg_[pmv2DCoords] < bdbCpmend_[pmv2DCoords];
    }

    bool ContainsDataLast() const {//whether data has been written for the last structure
        return bdbCpmend_[pmv2DCoords] < bdbCpmendovhd_[pmv2DCoords];
    }

    //change/pack structure pointers to include structures indexed in filterdata
    static void FilterStructs(
        const char** bdbdesc, char** bdbpmbeg, char** bdbpmend,
        const unsigned int* filterdata);

    //length of the structure at the given position/index (NOTE:pointers assumed valid):
    static size_t GetLengthAt(const char* const * const bdbpmbeg, int ndx) {
        return ((INTYPE*)(bdbpmbeg[pps2DLen]))[ndx];
    }

    //address of the structure at the given position/index (NOTE:pointers assumed valid):
    static size_t GetAddressAt(const char* const * const bdbpmbeg, int ndx) {
        return ((LNTYPE*)(bdbpmbeg[pps2DDist]))[ndx];
    }

    //field of the structure at the given position/index (NOTE:pointers assumed valid):
    template<typename T, int F>
    static T GetFieldAt(const char* const * const bdbpmbeg, int ndx) {
        return ((T*)(bdbpmbeg[F]))[ndx];
    }

    template<typename T, int F>
    T GetFieldAt(int ndx) const {return ((T*)(bdbCpmbeg_[F]))[ndx];}

    //set a structure field at the given position/index (NOTE:pointers assumed valid):
    template<typename T, int F>
    static void SetFieldAt(char* const * const bdbpmbeg, int ndx, T value) {
        ((T*)(bdbpmbeg[F]))[ndx] = value;
    }

    template<typename T, int F>
    void SetFieldAt(int ndx, T value) {((T*)(bdbCpmbeg_[F]))[ndx] = value;}

    //#structures written in the buffers (NOTE:pointers assumed valid):
    static size_t GetNoStructs(char* const * const bdbpmbeg, char* const * const bdbpmend) {
        return (size_t)(bdbpmend[pps2DLen]-bdbpmbeg[pps2DLen]) / SZINTYPE;
    }

    size_t GetNoStructsWritten() const {//#structures written in bdbCdata_
        return (size_t)(bdbCpmend_[pps2DLen]-bdbCpmbeg_[pps2DLen]) / SZINTYPE;
    }

    //#total positions written in the buffers (NOTE:pointers assumed valid):
    static size_t GetNoPosits(char* const * const bdbpmbeg, char* const * const bdbpmend) {
        return (size_t)(bdbpmend[pmv2Drsd]-bdbpmbeg[pmv2Drsd]) / SZCHTYPE;
    }

    size_t GetNoPositsWritten() const {//#positions written in bdbCdata_
        return (size_t)(bdbCpmend_[pmv2Drsd]-bdbCpmbeg_[pmv2Drsd]) / SZCHTYPE;
    }

    size_t GetNoPositsOvhd() const {//#positions in the structure being compiled
        return (size_t)(bdbCpmendovhd_[pmv2Drsd]-bdbCpmend_[pmv2Drsd]) / SZCHTYPE;
    }

    //{{these methods are for filling by residue one structure at a time
    bool AddOneResidue(
        int maxstrlen,
        CHTYPE rsdcode, INTYPE resnum, char inscode, char chain, char chord, 
        FPTYPE coords[pmv2DNoElems]);

    TPMBSDFinRetCode FinalizeCrntStructure(//finalize the structure being compiled
        size_t filendx, int strndx,
        int moltype, const std::string& description, 
        const std::string& strchain, const std::string& strmodel, 
        bool usechaininfo, bool usemodelinfo,
        const int queryblocks,
        char* const * const * const querypmbegs,
        char* const * const * const querypmends);
    //}}

    bool FieldTypeValid() const;//is field Type valid across all structures written?

    size_t GetFileNdxAt(int ndx) const {return tmpclustfilendxs_[ndx];}
    int GetStructNdxAt(int ndx) const {return tmpcluststrndxs_[ndx];}

    //get the description of the structure in the overhead buffer
    const char* GetOvhdStrDescription() const;

    //copy overhead to another batch object:
    TPMBSDFinRetCode CopyOvhdTo(PMBatchStrData&) const;

    //return true if the sequence similairty between the structure in Ovhd and 
    //any of the queries in given blocks is above the threshold
    bool SequenceSimilarityOvhd(
        const int queryblocks,
        char* const * const * const querypmbegs,
        char* const * const * const querypmends) const;

    template<int DIMD>
    static bool CheckAlignmentScore(
        const float seqsimthrscore,
        const int qrydst, const int qrylen,
        const int dbstrdst, const int dbstrlen,
        const int qrybegpos, const int rfnbegpos,
        const char* const * const __restrict querypmbeg,
        const char* const * const __restrict bdbCpmbeg,
        char rfnRE[DIMD], char qryRE[DIMD],
        float scores[DIMD], float pxmins[DIMD],
        float tmp[DIMD]);

    void Print() const;
    static void Print(char* const * const bdbpmbeg, char* const * const bdbpmend);

private:
    void SetMaxDataLimits(
        size_t maxdatasize, size_t maxdatalen, size_t maxnstrs)
    {
        maxdatasize_ = maxdatasize;
        maxdatalen_ = maxdatalen;
        maxnstrs_ = maxnstrs;
    }

    void AllocateSpaceForData();
    bool AllocateSpaceForDescriptions();
    void AllocateSpaceForDescPtrs();

    void FallbackOvhdPtrs() {//fall back overhead pointers
        for(int f = 0; f < pmv2DTotFlds; f++)
            bdbCpmendovhd_[f] = bdbCpmend_[f];
    }

    std::string FormatDescription(
        std::string description,
        const std::string& strchain, const std::string& strmodel, 
        bool usechaininfo, bool usemodelinfo);

private:
    template<typename T, int field>
    void sort_helper_ssfields_assign(size_t nstts);
    template<int field>
    void sort_helper_psfields_assign(size_t nstts);

private:
    std::unique_ptr<char,DRDataDeleter> tmpbdbCdata_;//temporary buffer for sorted data
    std::vector<INTYPE> tmpndxs_;//temporary index vector for structure lengths
    char tmpreadbuff_[PMBatchStrData_TMPBUFFSIZE];//tmp buffer for data read

public:
    std::vector<size_t> tmpclustfilendxs_;//temporary file index vector for clustering
    std::vector<int> tmpcluststrndxs_;//temporary structure-within-file index vector for clustering

public:
    // *** MEMBER VARIABLES ***
    std::unique_ptr<char,DRHostDataDeleter> bdbCdata_;
    char* bdbCpmbeg_[pmv2DTotFlds];//addresses of the beginnings of the fields in bdbCdata_
    char* bdbCpmend_[pmv2DTotFlds];//end addresses of the fields in bdbCdata_
    //end addresses of the fields in bdbCdata_
    // including one additional structure if any that does not fit into the memory limits:
    char* bdbCpmendovhd_[pmv2DTotFlds];
    //TODO: introduce bdbCpmendovhdbeg_ instead of using bdbCpmend_ for CPU version (ovhd copy)!

//     //sizes in bytes of the fields of complete structures written in bdbCdata_:
//     size_t szpm2dvf_[pmv2DTotFlds];
//     //sizes in bytes of the fields written in bdbCdata_, 
//     // including one additional structure if any that does not fit into the memory limits:
//     size_t szpm2dvfovhd_[pmv2DTotFlds];

    std::unique_ptr<char,DRDataDeleter> bdbCdescs_;//descriptions
    std::unique_ptr<char*[]> bdbCptrdescs_;//pointers to structure descriptions in bdbCdescs_

    size_t szbdbCdata_;//size allocated for bdbCdata_
    size_t szbdbCdescs_;//size allocated for bdbCdescs_
    size_t nbdbCptrdescs_;//number of slots allocated for bdbCptrdescs_

//     size_t szcrntsize_;//size currently used by bdbCdata_
//     size_t szcrntlen_;//total #positions (length) currently occupied by bdbCdata_
//     size_t szcrntstrs_;//#structures currently in bdbCdata_

    size_t maxdatasize_;//max size of data for bdbCdata_ to contain
    size_t maxdatalen_;//max total #positions (length) over all structures
    size_t maxnstrs_;//max #structures whose data bdbCdata_ can contain

    TSCounterVar cnt_;//thread-safe counter of how many agents access the data
};

// =========================================================================
// INLINES
// 
// AllocateSpace: allocate space for structure data and descriptions given the 
// limits of data chunk size, total number of positions (residues), and the 
// number of structures
//
inline
void PMBatchStrData::AllocateSpace(
    size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs)
{
    //the following comes first:
    SetMaxDataLimits(chunkdatasize, chunkdatalen, chunknstrs);

    //allocate space for structure data plus space for the end addresses of 
    // structure descriptions once the limits have been set
    AllocateSpaceForData();

    AllocateSpaceForDescriptions();
    AllocateSpaceForDescPtrs();
}

// -------------------------------------------------------------------------
// GetPMDataSize: get the size of the structure model data written
inline
size_t PMBatchStrData::GetPMDataSize()
{
    MYMSG("PMBatchStrData::GetPMDataSize",6);
    size_t size = 0;
    for(int n = 0; n < pmv2DTotFlds; n++)
        size += (size_t)(bdbCpmend_[n]-bdbCpmbeg_[n]);
    return size;
}

// -------------------------------------------------------------------------
// GetPMDataSize1: get the size of complete structure model data of one 
// structure 
inline
size_t PMBatchStrData::GetPMDataSize1(size_t structlen)
{
    MYMSG("PMBatchStrData::GetPMDataSize1",6);
    size_t size = 0;
    if(structlen < 1) return size;
    int n = 0;
    for(; n < pps2DStrFlds; n++)
        size += TPM2DVectorFieldSize::szvfs_[n];
    for(; n < pmv2DTotFlds; n++)
        size += TPM2DVectorFieldSize::szvfs_[n] * structlen;
    return size;
}

// -------------------------------------------------------------------------
// GetPMDataSizeUB: get the max size of complete structure model data when 
// the total number of positions (residues) is totallen
inline
size_t PMBatchStrData::GetPMDataSizeUB(size_t totallen)
{
    MYMSG("PMBatchStrData::GetPMDataSizeUB",6);
    size_t size = 0;
    for(int n = 0; n < pmv2DTotFlds; n++)
        //[pps2DLen]: when #structs=totallen
        //[pps2DDist]: when #structs=totallen
        size += TPM2DVectorFieldSize::szvfs_[n] * totallen;
    return size;
}

// -------------------------------------------------------------------------
// AddOneResidue: save data for one residue in the batch data object; 
// adjust accordingly pointers and running sizes;
// maxstrlen, max structure length (-1, use default max);
// rsdcode, one-letter residue code;
// resnum, residue serial number as appears in the structure file;
// inscode, residue insertion code;
// chain, chain id of the structure;
// chord, chain serial number (order) in the structure file;
// coords, residue coordinates
inline
bool PMBatchStrData::AddOneResidue(
    int maxstrlen,
    CHTYPE rsdcode, INTYPE resnum, char inscode, char chain, char chord, 
    FPTYPE coords[pmv2DNoElems])
{
    //NOTE: structure-specific fields pps2DLen, pps2DType, and pps2DDist 
    // have to be updated on finalizing the data of the current structure

    //number of positions in this structure (not finalized) so far
    size_t nposits = GetNoPositsOvhd();

    if((0 < maxstrlen && (size_t)maxstrlen < nposits)/*max length violation*/||
       PMBSmaxonestructurelength < nposits/*buffer size exceeded*/||
       maxdatalen_ < nposits/*allocation too small*/)
    {
        //fallback, abandon the structure
        FallbackOvhdPtrs();
        return false;
    }

    //number of structures written so far:
    // INTYPE nstructs = GetNoStructsWritten();
    int n, f;

    for(n = 0, f = pmv2DCoords; n < pmv2DNoElems; n++, f++) {
        *(FPTYPE*)(bdbCpmendovhd_[f]) = coords[n];
        bdbCpmendovhd_[f] += TPM2DVectorFieldSize::szvfs_[f];
    }

    *(LNTYPE*)(bdbCpmendovhd_[pmv2D_Ins_Ch_Ord]) = 
        (LNTYPE)PM2D_MAKEINT_Ins_Ch_Ord(
            (unsigned int)inscode, (unsigned int)chain, (unsigned int)chord);
    bdbCpmendovhd_[pmv2D_Ins_Ch_Ord] += TPM2DVectorFieldSize::szvfs_[pmv2D_Ins_Ch_Ord];

    *(INTYPE*)(bdbCpmendovhd_[pmv2DResNumber]) = resnum;
    bdbCpmendovhd_[pmv2DResNumber] += TPM2DVectorFieldSize::szvfs_[pmv2DResNumber];

    *(CHTYPE*)(bdbCpmendovhd_[pmv2Drsd]) = rsdcode;
    bdbCpmendovhd_[pmv2Drsd] += TPM2DVectorFieldSize::szvfs_[pmv2Drsd];

    //NOTE: secondary structure assignments skipped; they will be 
    // calculated and filled in by an accelerator
    //*(CHTYPE*)(bdbCpmendovhd_[pmv2Dss]) = ...;
    bdbCpmendovhd_[pmv2Dss] += TPM2DVectorFieldSize::szvfs_[pmv2Dss];

    return true;
}

// -------------------------------------------------------------------------
// FinalizeCrntStructure: finalize the structure being compiled by 
// moving the data from the overhead buffer to the end of the data in the 
// chunk; moving data actually corresponds to changing the values of 
// pointers;
// return false (pmbsdfAbandoned) if the structure is too large to be 
// kept by the buffers associated with the chunk of data;
// filendx, file index for clustering;
// strndx, structure index within a file for clustering;
inline
PMBatchStrData::TPMBSDFinRetCode 
PMBatchStrData::FinalizeCrntStructure(
    size_t filendx, int strndx,
    int moltype, const std::string& description, 
    const std::string& strchain, const std::string& strmodel, 
    bool usechaininfo, bool usemodelinfo,
    const int /* queryblocks */,
    char* const * const * const /* querypmbegs */,
    char* const * const * const /* querypmends */)
{
    MYMSG("PMBatchStrData::FinalizeCrntStructure",6);
    static const std::string preamb = "PMBatchStrData::FinalizeCrntStructure: ";
    static const size_t devminrlength = CLOptions::GetDEV_MINRLEN();
    //number of positions in this structure (not finalized)
    size_t nposits = GetNoPositsOvhd();//#positions to be written
    size_t npositswrt = GetNoPositsWritten();//#positions written
    size_t szstruct = PMBatchStrData::GetPMDataSize1(nposits);//size of data to be written
    size_t szstructswrt = GetPMDataSize();//size of data written
    INTYPE nstructswrt = GetNoStructsWritten();//number of structures written
    static constexpr size_t szalign = PMBSdatalignment * pmv2DTotFlds;//max size for data alignment

    if(nposits < 1) {
        FallbackOvhdPtrs();
        return pmbsdfEmpty;
    }
    if(nposits < devminrlength) {
        FallbackOvhdPtrs();
        return pmbsdfShort;
    }

    if(maxdatasize_ < szstruct+szalign || 
       maxdatalen_ < nposits)//allocation too small
    {
        //fallback, abandon the structure
        FallbackOvhdPtrs();
        return pmbsdfAbandoned;
    }

    // //verify mutual sequence similarity
    // //NOTE: verified at the computation level.
    // if(0 < queryblocks && querypmbegs && querypmends &&
    //    !SequenceSimilarityOvhd(queryblocks, querypmbegs, querypmends))
    // {
    //     //fallback, abandon the structure
    //     FallbackOvhdPtrs();
    //     return pmbsdfLowSimilarity;
    // }


    //{{ fill in accompanying data before returning with 
    // pmbsdfLimits (to be moved later) or pmbsdfWritten
    char** ptrdescs = bdbCptrdescs_.get();//assume a valid pointer

    std::string desc = 
        FormatDescription(description, strchain, strmodel, 
            usechaininfo, usemodelinfo);

    memcpy(ptrdescs[nstructswrt], desc.c_str(), desc.size());
    ptrdescs[nstructswrt][desc.size()] = 0;
    //}}


    //fill in structure-specific fields; use pps2DType as a type and ID simultaneously;
    *(INTYPE*)(bdbCpmendovhd_[pps2DLen]) = (INTYPE)nposits;
    *(INTYPE*)(bdbCpmendovhd_[pps2DType]) = (INTYPE)(moltype);
    *(LNTYPE*)(bdbCpmendovhd_[pps2DDist]) = (LNTYPE)npositswrt;

    tmpclustfilendxs_.push_back(filendx);
    tmpcluststrndxs_.push_back(strndx);

    if(maxdatasize_ < szstructswrt + szstruct + szalign || 
       maxdatalen_ < npositswrt + nposits || 
       maxnstrs_ < (size_t)nstructswrt + 1)
    {
        //this structure cannot be contained currently, leave it for the next round;
        //structure will be the 1st, distance = 0
        *(LNTYPE*)(bdbCpmendovhd_[pps2DDist]) = 0;
        return pmbsdfLimits;
    }

    if((size_t)INT_MAX <= npositswrt + nposits)
        throw MYRUNTIME_ERROR2( 
        preamb + "Overflow detected. Data chunk size must be reduced.", 
        CRITICAL);

    bdbCpmendovhd_[pps2DLen] += TPM2DVectorFieldSize::szvfs_[pps2DLen];
    bdbCpmendovhd_[pps2DType] += TPM2DVectorFieldSize::szvfs_[pps2DType];
    bdbCpmendovhd_[pps2DDist] += TPM2DVectorFieldSize::szvfs_[pps2DDist];

    //adjust pointers to the end of overhead data section
    for(int f = 0; f < pmv2DTotFlds; f++)
        bdbCpmend_[f] = bdbCpmendovhd_[f];

    return pmbsdfWritten;
}

// -------------------------------------------------------------------------
// FormatDescription: format structure description;
// return `move' string
inline
std::string PMBatchStrData::FormatDescription(
    std::string description,
    const std::string& strchain, const std::string& strmodel, 
    bool usechaininfo, bool usemodelinfo)
{
    if(usechaininfo && strchain.size() && strchain[0])
        description = description + " Chn:" + strchain;//chain
    if(usemodelinfo && strmodel.size() && strmodel[0])
        description = description + " (M:" + strmodel + ")";//model
    std::string desc = 
        (PMBSdatDEFDESCLEN <= description.size()) //4: "..." + 0
        ?   "..." + description.substr(description.size()-PMBSdatDEFDESCLEN+4)
        :   description;
    return desc;
}

// -------------------------------------------------------------------------
// FieldTypeValid: verify whether the Type field is valid across all 
// written structures
// 
inline
bool PMBatchStrData::FieldTypeValid() const
{
    int nstructswrt = (int)GetNoStructsWritten();
    for(int i = 0; i < nstructswrt; i++)
        if(GetFieldAt<INTYPE,pps2DType>(i) == INT_MAX)
            return false;
    return true;
}

// -------------------------------------------------------------------------
// CopyOvhdTo: move data (of one structure) from the overhead buffers to 
// another batch object;
// bsd, batch object to move data to;
// return false (pmbsdfAbandoned) if the structure is too large to be 
// kept (should not happen without a bug);
inline
PMBatchStrData::TPMBSDFinRetCode
PMBatchStrData::CopyOvhdTo(PMBatchStrData& bsd) const
{
    MYMSG("PMBatchStrData::CopyOvhdTo",6);
    static const std::string preamb = "PMBatchStrData::CopyOvhdTo: ";
    //number of positions in this structure (not finalized)
    size_t nposits = GetNoPositsOvhd();//#positions to be written

    if(nposits < 1)
        return pmbsdfEmpty;

    size_t npositswrtbsd = bsd.GetNoPositsWritten();//#positions written
    size_t szstruct = PMBatchStrData::GetPMDataSize1(nposits);//size of data to be written
    size_t szstructswrtbsd = PMBatchStrData::GetPMDataSize1(npositswrtbsd);//size of data written
    INTYPE nstructswrtbsd = bsd.GetNoStructsWritten();//number of structures written
    static constexpr size_t szalign = PMBSdatalignment * pmv2DTotFlds;//max size for data alignment

    if(bsd.maxdatasize_ < szstruct+szalign || 
       bsd.maxdatalen_ < nposits)//allocation too small
    {
        //NOTE: this should not happen!
        return pmbsdfAbandoned;
    }

    if(bsd.maxdatasize_ < szstructswrtbsd + szstruct + szalign || 
       bsd.maxdatalen_ < npositswrtbsd + nposits || 
       bsd.maxnstrs_ < (size_t)nstructswrtbsd + 1)
    {
        //NOTE: should not happen!
        return pmbsdfLimits;
    }

    if((size_t)INT_MAX <= npositswrtbsd + nposits)
        //NOTE: should not happen!
        throw MYRUNTIME_ERROR2( 
        preamb + "Overflow detected. Data chunk size must be reduced.", 
        CRITICAL);

    //{{actual copy
    int n = 0;
    for(; n < pps2DStrFlds; n++) {
        memcpy(bsd.bdbCpmend_[n], bdbCpmend_[n], TPM2DVectorFieldSize::szvfs_[n]);
        bsd.bdbCpmend_[n] += TPM2DVectorFieldSize::szvfs_[n];
        bsd.bdbCpmendovhd_[n] = bsd.bdbCpmend_[n];
    }
    for(; n < pmv2DTotFlds; n++) {
        memcpy(bsd.bdbCpmend_[n], bdbCpmend_[n], TPM2DVectorFieldSize::szvfs_[n] * nposits);
        bsd.bdbCpmend_[n] += TPM2DVectorFieldSize::szvfs_[n] * nposits;
        bsd.bdbCpmendovhd_[n] = bsd.bdbCpmend_[n];
    }
    //copy description
    char* const * ptrdescs = bdbCptrdescs_.get();//assume valid pointers
    char** bsdptrdescs = bsd.bdbCptrdescs_.get();
    INTYPE nstructswrt = GetNoStructsWritten();//number of structures written
    strcpy(bsdptrdescs[nstructswrtbsd], ptrdescs[nstructswrt]);
    //}}

    if(tmpclustfilendxs_.size() != tmpcluststrndxs_.size() ||
       tmpclustfilendxs_.size() != GetNoStructsWritten() + 1)
        throw MYRUNTIME_ERROR2( 
        preamb + "Inconsistent file index size.", CRITICAL);

    if(tmpclustfilendxs_.size()) bsd.tmpclustfilendxs_.push_back(tmpclustfilendxs_.back());
    if(tmpcluststrndxs_.size()) bsd.tmpcluststrndxs_.push_back(tmpcluststrndxs_.back());

    return pmbsdfWritten;
}

// -------------------------------------------------------------------------
// GetOvhdStrDescription: get the description of the structure in the 
// overhead buffer
inline
const char* PMBatchStrData::GetOvhdStrDescription() const
{
    size_t nposits = GetNoPositsOvhd();//#positions to be written

    if(nposits < 1) return "";

    INTYPE nstructswrt = GetNoStructsWritten();
    char* const * ptrdescs = bdbCptrdescs_.get();

    return ptrdescs? ptrdescs[nstructswrt]: "";
}

// -------------------------------------------------------------------------

#endif//__PMBatchStrData_h__
