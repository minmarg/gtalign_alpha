/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/mybase.h"

#include <omp.h>

#include <sys/types.h>
#include <sys/stat.h>

#ifdef OS_MS_WINDOWS
#else
#include <unistd.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <assert.h>

#include <cctype>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <thread>

#include "extzlib/zlib.h"
#include "libutil/mydirent.h"
#include "libutil/alpha.h"

#include "PMBatchStrData.h"
#include "InputFilelist.h"
#include "FlexDataRead.h"

// -------------------------------------------------------------------------
// Constructor
//
FlexDataRead::FlexDataRead(
    const std::vector<std::string>& strfilelist,
    const std::vector<std::string>& pntfilelist,
    const std::vector<size_t>& strfilepositionlist,
    const std::vector<size_t>& strfilesizelist,
    const std::vector<int>& strparenttypelist,
    const std::vector<int>& strfiletypelist,
    const std::vector<size_t>& filendxlist,
    std::vector<std::vector<int>>& globalids,
    const size_t ndxstartwith,
    const size_t ndxstep,
    int maxstrlen,
    const bool clustering,
    const bool clustmaster)
:   
    strfilelist_(strfilelist),
    pntfilelist_(pntfilelist),
    strfilepositionlist_(strfilepositionlist),
    strfilesizelist_(strfilesizelist),
    strparenttypelist_(strparenttypelist),
    strfiletypelist_(strfiletypelist),
    filendxlist_(filendxlist),
    globalids_(globalids),
    ndxstartwith_(ndxstartwith),
    ndxstep_(ndxstep),
    //
    currentfilendx_(ndxstartwith),
    maxstrlen_(maxstrlen),
    mapped_(false),
    //
    clustering_(clustering),
    clustmaster_(clustmaster),
    idgenerator_(0),
    //
    strmodel_(4, 0),//allocate space once for model number string
    strchain_(5, 0),//allocate space once for chain string
    strresnum_(5, 0),//allocate space once for resnum
    strcoord_(9, 0)//allocate space once for coordinate
{
    MYMSG("FlexDataRead::FlexDataRead",4);
    if(clustering_)
        structcounter_.resize(filendxlist_.size(), 0);
    Initializer();
}

// -------------------------------------------------------------------------
// Initializer: initialize data for object construction
//
void FlexDataRead::Initializer()
{
    //fieldndxs_[nciffields] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    memset(fieldndxs_, 0xff, nciffields * sizeof(fieldndxs_[mysetflag]));

    InvalidateFileDescriptor();

    //{{page data
    ssize_t pgs = ObtainPageSize();

    if(pgs <= 0)
        throw MYRUNTIME_ERROR(
        "FlexDataRead::FlexDataRead: Unable to determine system page size.");
    //}}

    //{{zip stream variable for allocating inflate state
    z_stream_.zalloc = Z_NULL;
    z_stream_.zfree = Z_NULL;
    z_stream_.opaque = Z_NULL;
    z_stream_.avail_in = 0;
    z_stream_.next_in = Z_NULL;
    if(inflateInit2(&z_stream_, (MAX_WBITS+16)) != Z_OK)
        throw MYRUNTIME_ERROR(
        "FlexDataRead::FlexDataRead: Failed to initialize zlib stream.");
    //}}

    //{{zip buffer
    current_zipdata_.pagesize_ = pgs;
    current_zipdata_.data_ = NULL;
    current_zipdata_.data_ = (char*)malloc(pgs);
    if(!current_zipdata_.data_)
        throw MYRUNTIME_ERROR("FlexDataRead::FlexDataRead: Not enough memory.");
    //}}

    //{{current buffer
    current_strdata_.pagesize_ = pgs;
    current_strdata_.data_ = NULL;
    current_strdata_.data_ = (char*)malloc(pgs);
    if(!current_strdata_.data_)
        throw MYRUNTIME_ERROR("FlexDataRead::FlexDataRead: Not enough memory.");
    //}}

    //{{profile buffer
    profile_buffer_.data_ = NULL;
    profile_buffer_.pagesize_ = 0;
    size_t szbufpg =
        PCMAX((size_t)InputFilelist::MAXCOMPLETESTRUCTSIZE, (size_t)(pgs * 2));
    profile_buffer_.data_ = (char*)malloc(szbufpg);
    if(!profile_buffer_.data_)
        throw MYRUNTIME_ERROR( "FlexDataRead::FlexDataRead: Not enough memory." );
    profile_buffer_.pagesize_ = szbufpg;
    //}}

    ResetDbMetaData();
}

// -------------------------------------------------------------------------
// Destructor
//
FlexDataRead::~FlexDataRead()
{
    MYMSG("FlexDataRead::~FlexDataRead",4);
    Destroy();
}

// -------------------------------------------------------------------------
// Destroy: destroy allocated resources and close files
//
void FlexDataRead::Destroy()
{
    MYMSG("FlexDataRead::Destroy",4);
    Close();

    inflateEnd(&z_stream_);

    if(current_zipdata_.data_) {
        free(current_zipdata_.data_);
        current_zipdata_.data_ = NULL;
        current_zipdata_.pagesize_ = 0;
        ResetZipPageData();
    }
    if(current_strdata_.data_) {
        free(current_strdata_.data_);
        current_strdata_.data_ = NULL;
        current_strdata_.pagesize_ = 0;
        ResetCurrentPageData();
    }
    if(profile_buffer_.data_) {
        free(profile_buffer_.data_);
        profile_buffer_.data_ = NULL;
        profile_buffer_.pagesize_ = 0;
        ResetProfileBufferData();
    }
}

// -------------------------------------------------------------------------
// Open: make a file list for processing individual files
//
void FlexDataRead::Open()
{
    MYMSG("FlexDataRead::Open",4);
    //reset counters:
    Close();
    currentfilendx_ = ndxstartwith_;
    SetDbMetaData();
}

// -------------------------------------------------------------------------
// Close: close a file or database
//
void FlexDataRead::Close()
{
    MYMSG("FlexDataRead::Close",8);

    // if(GetMapped())
    //     UnmapFile();

    strmodel_.clear();//[0] = 0;
    CloseFile();

    ResetZipPageData();
    ResetCurrentPageData();
    ResetProfileBufferData();
    ResetDbMetaData();
}

// -------------------------------------------------------------------------
// ReadData: read structure data from file(s)
//
bool FlexDataRead::ReadData(
    PMBatchStrData& bsd,
    const int queryblocks,
    char* const * const * const querypmbegs,
    char* const * const * const querypmends)
{
    MYMSG("FlexDataRead::ReadData",4);
    bool readon = true;//continue reading

    while(!EndOfData() && readon) {
        if(Eof()) {
            //next file...
            Close();
            currentfilendx_ += ndxstep_;
            SetDbMetaData();
        }
        if(!ValidFileDescriptor()) {
            OpenFile(db_parentname_.c_str(), db_position_);
            // if(GetMapped())
            //     MapFile();
        }
        try {
            switch(db_filetype_) {
                case InputFilelist::FDRFlZip: 
                case InputFilelist::FDRFlPDBZip:
                case InputFilelist::FDRFlStructure:
                case InputFilelist::FDRFlStructurePDB:
                    readon = ReadDataPDB(bsd,  queryblocks, querypmbegs, querypmends);
                    break;
                case InputFilelist::FDRFlPDBxmmCIFZip:
                case InputFilelist::FDRFlStructurePDBxmmCIF:
                    readon = ReadDataCIF(bsd,  queryblocks, querypmbegs, querypmends);
                    break;
                default:
                    throw MYRUNTIME_ERROR(
                    "FlexDataRead::ReadData: Unrecognized file type: " +
                    db_filename_);
                    //break;
            };
        } catch( myexception const& ex ) {
            if(ex.eclass() != NOCLASS)
                throw ex;
            SetEof();
            if(!clustering_ || clustmaster_)
                warning((std::string(ex.what()) + " File ignored: " + db_entrydesc_).c_str());
        }
    }

    return readon? EndOfData(): false;
}





// -------------------------------------------------------------------------
// ReadDataPDB: read structure data from a PDB file;
// return false if bsd data buffers are full and no additional data can be 
// stored
//
bool FlexDataRead::ReadDataPDB(
    PMBatchStrData& bsd,
    const int queryblocks,
    char* const * const * const querypmbegs,
    char* const * const * const querypmends)
{
    MYMSG("FlexDataRead::ReadDataPDB",6);
    static const std::string preamb = "FlexDataRead::ReadDataPDB: ";
    //constexpr size_t maxlnwidth = 128;
    constexpr size_t terlnwidth = 26;//linewidth(-1) for the TER record
    constexpr size_t sfflnwidth = 54;//sufficient linewidth
    //sufficient linewidth for a model number (14, still format errors occur):
    constexpr size_t sffmodlnwidth = 12;
    static const int teropt = CLOptions::GetI_TER();
    static const int splitopt = CLOptions::GetI_SPLIT();
    static const int hetatm = CLOptions::GetI_HETATM();
    static const std::string satom = CLOptions::GetI_ATOM_PROT();
    const bool usechid = !(teropt == CLOptions::istEOF &&
        (splitopt == CLOptions::issaNoSplit || splitopt == CLOptions::issaByMODEL));
    const bool usemodn = !(teropt == CLOptions::istEOF &&
        (splitopt == CLOptions::issaNoSplit));

    //moved to constructor
    //static std::string strresnum(5, 0);//allocate space once for resnum
    //static std::string strcoord(9, 0);//allocate space once for coordinate

    assert(satom.size()==4);

    enum {szmod = 4};
    char strmodel[szmod] = {0};//model serial number string
    char strmodelprev[szmod+1] = {0};//previous model serial number
    int moltype = 0;//type: protein or RNA 
    char chord = 0;//chain serial number (order)
    char prevchid = 0, prevchidx = 0;//previous chain IDs (1st used also as a flag)
    char chid, insc;//chain ID and res. insertion code
    CHTYPE resd;//one-letter residue code
    INTYPE resnum;//residue serial number 
    FPTYPE coords[pmv2DNoElems];//residue coordinates


    //print warning only without setting eof
    std::function<void(int)> lfWarning = [this,&prevchidx](int warn) {
        if(clustering_ && !clustmaster_) return;
        switch(warn) {
            case PMBatchStrData::pmbsdfShort:
                warning((std::string("Chain ") + std::string(1,prevchidx) + 
                " too short and ignored; File: " + db_entrydesc_).c_str());
                break;
        }
    };

    //set Eof and reset chain and molecular serial numbers; print warning
    std::function<void(int)> lfSetEof = [this](int warn) {
        SetEof();
        if(clustering_ && !clustmaster_) return;
        switch(warn) {
            case PMBatchStrData::pmbsdfEmpty:
                warning(("No atoms selected; File ignored: " + 
                db_entrydesc_).c_str());
                break;
            case PMBatchStrData::pmbsdfAbandoned:
                warning(("Structure too big and ignored; File: " + 
                db_entrydesc_).c_str());
                break;
        }
    };

    //increment the structure counter associated with the current file
    std::function<void()> lfIncStr = [this]() {
        if(!clustering_) return;
        if(structcounter_.size() <= currentfilendx_)
            throw MYRUNTIME_ERROR2(
            preamb + "Invalid structure counter.", CRITICAL);
        structcounter_[currentfilendx_]++;
        if(!clustmaster_) return;
        globalids_[currentfilendx_].push_back(idgenerator_++);
        if((int)globalids_[currentfilendx_].size() != structcounter_[currentfilendx_])
            throw MYRUNTIME_ERROR2(
            preamb + "Inconsistent structure counter.", CRITICAL);
    };

    //get the global structure ID for the current structure of the current file
    std::function<int(int)> lfGetGlobID = [this](int moltype) {
        if(!clustering_) return moltype;
        if(clustmaster_) return idgenerator_;//(0 <= moltype)? idgenerator_: -idgenerator_;
        if(structcounter_.size() <= currentfilendx_)
            throw MYRUNTIME_ERROR2(
            preamb + "Invalid structure counter (2).", CRITICAL);
        if((int)globalids_[currentfilendx_].size() <= structcounter_[currentfilendx_])
            //master hasn't yet processed all structures of the current file
            return INT_MAX;
        return globalids_[currentfilendx_][ structcounter_[currentfilendx_] ];
    };


    while(!Eof()) {
        if(profile_buffer_.eof())
            NextDataPage();

        if(profile_buffer_.datlen_ < profile_buffer_.curpos_) {
            bsd.Fallback();
            lfSetEof(false);
            throw MYRUNTIME_ERROR2(
            preamb + "Invalid file buffer's position.", 
            CRITICAL);
        }

        if(profile_buffer_.datlen_ - profile_buffer_.curpos_ < maxlnwidth)
            NextDataPage();

        const char* pbeg = profile_buffer_.data_ + profile_buffer_.curpos_;
        const char* pend = profile_buffer_.data_ + profile_buffer_.datlen_;
        const char* p = pbeg;

        //locate the first occurrence of NL in the buffer;
        //NOTE: located on the next iteration through .incposnl
        //for(; p < pend && *p != '\n' && *p != '\r'; p++);
        //for(; p < pend && (*p == '\n' || *p == '\r'); p++);

        size_t datwidth = (size_t)(pend-p);

        if(datwidth < 3) {
            profile_buffer_.incposnl((size_t)(p-pbeg));
            continue;
        }

        if(bsd.ContainsDataLast()) {
            //at least one position has been saved
            if((teropt >= CLOptions::istEND && memcmp(p,"END",3) == 0) ||
               (teropt >= CLOptions::istTER_ENDorChain && memcmp(p,"TER",3) == 0))
            {   //make eof to close file
                lfSetEof(false);
                break;
            }
        }

        if(splitopt > CLOptions::issaNoSplit && memcmp(p,"END",3) == 0) {
            //chain id can be the same as before, assign to 0
            prevchid = 0;
            profile_buffer_.incposnl((size_t)(p-pbeg) + 3);
            continue;
        }

        if(memcmp(p,"MODEL ",6) == 0 && datwidth > sffmodlnwidth) {
            char* sp = strmodel;
            for(int i = 10; i < 14 && p[i] != '\n' && p[i] != '\r'; i++)
                if(p[i] != ' ' && p[i] != '\t') *sp++ = p[i];
            strmodel_.assign(strmodel, sp - strmodel);
            profile_buffer_.incposnl((size_t)(p-pbeg) + sffmodlnwidth);
            continue;//update model serial number
        }

        if(datwidth < sfflnwidth || (p[16]!=' ' && p[16]!='A')) {
            //invalid data amount or alternate location indicator (considered)
            profile_buffer_.incposnl((size_t)(p-pbeg));
            continue;
        }

        if(memcmp(p,"ATOM  ",6) != 0 && (!hetatm || memcmp(p,"HETATM",6) != 0)) {
            profile_buffer_.incposnl((size_t)(p-pbeg) + 
                ((memcmp(p,"TER",3) == 0)? terlnwidth: sfflnwidth));
            continue;//this is not an atom section
        }

        if(memcmp(p+12,satom.c_str(),4) != 0) {
            profile_buffer_.incposnl((size_t)(p-pbeg) + sfflnwidth);
            continue;//not the atom of interest
        }

        chid = p[21];//chain id
        insc = p[26];//insertion code
        resd = ResName2Code(p+17);

        try {
            resnum = std::stoi(strresnum_.assign(p+22, 4));//residue serial number 
            coords[0] = std::stof(strcoord_.assign(p+30, 8));//residue coordinates
            coords[1] = std::stof(strcoord_.assign(p+38, 8));//residue coordinates
            coords[2] = std::stof(strcoord_.assign(p+46, 8));//residue coordinates
        } catch(...) {
            bsd.Fallback();
            lfSetEof(false);
            throw MYRUNTIME_ERROR(preamb + "Failed to read residue data from PDB file.");
        }

        if(prevchid != chid) {
            bool cond1 = (teropt >= CLOptions::istENDorChain);//istENDorChain||istTER_ENDorChain
            bool cond2 = (splitopt == CLOptions::issaByChain || 
                (prevchid == 0 && splitopt == CLOptions::issaByMODEL));
            if(bsd.ContainsDataLast() && (cond1 || cond2))
            {
                chord = 0;//reset chain number on each structure finalization
                if(cond1) lfSetEof(false);
                switch(bsd.FinalizeCrntStructure(
                        currentfilendx_, clustering_? structcounter_[currentfilendx_]: 0,
                        lfGetGlobID(moltype), db_entrydesc_, 
                        std::string(1,prevchidx), std::string(strmodelprev), 
                        usechid, usemodn,  queryblocks, querypmbegs, querypmends))
                {
                    case PMBatchStrData::pmbsdfEmpty:
                        lfSetEof(PMBatchStrData::pmbsdfEmpty); return true;
                    case PMBatchStrData::pmbsdfShort:
                        lfWarning(PMBatchStrData::pmbsdfShort); break;
                    case PMBatchStrData::pmbsdfAbandoned:
                        lfSetEof(PMBatchStrData::pmbsdfAbandoned); return true;
                    case PMBatchStrData::pmbsdfLowSimilarity: return true;
                    //structure does not fit into the allocation limits;
                    //profile_buffer_.curpos_ is left as is to read the same line again later
                    case PMBatchStrData::pmbsdfLimits: lfIncStr(); return false;
                    case PMBatchStrData::pmbsdfWritten: lfIncStr(); break;
                }
                if(cond1) return true;
                // if(cond1) {
                //     lfSetEof(false);
                //     return true;
                // }
                moltype = 0;//reset mol. type having finished finalization 
            }
            #pragma omp simd
            for(int i = 0; i < szmod; i++) strmodelprev[i] = strmodel_[i];
            strmodelprev[strmodel_.size()] = 0;
            prevchid = prevchidx = chid;
            chord++;
        }

        if(p[17]==' ' && (p[18]==' '||p[18]=='D')) moltype++;
        else moltype--;

        if(!bsd.AddOneResidue(maxstrlen_, resd, resnum, insc, chid, chord, coords)) {
            lfSetEof(PMBatchStrData::pmbsdfAbandoned);
            return true;
        }

        profile_buffer_.incposnl((size_t)(p-pbeg) + sfflnwidth);
    }//while(!Eof())

    switch(bsd.FinalizeCrntStructure(
            currentfilendx_, clustering_? structcounter_[currentfilendx_]: 0,
            lfGetGlobID(moltype), db_entrydesc_,
            std::string(1,prevchidx), std::string(strmodelprev),
            usechid, usemodn,  queryblocks, querypmbegs, querypmends))
    {
        case PMBatchStrData::pmbsdfEmpty: 
            lfSetEof(PMBatchStrData::pmbsdfEmpty); return true;
        case PMBatchStrData::pmbsdfShort:
            lfWarning(PMBatchStrData::pmbsdfShort); break;
        case PMBatchStrData::pmbsdfAbandoned: 
            lfSetEof(PMBatchStrData::pmbsdfAbandoned); return true;
        case PMBatchStrData::pmbsdfLowSimilarity: return true;
        case PMBatchStrData::pmbsdfLimits: lfIncStr(); return false;
        case PMBatchStrData::pmbsdfWritten: lfIncStr(); break;
    }

    return true;
}





// -------------------------------------------------------------------------
// ReadDataCIF: read structure data from an mmCIF file;
// return false if bsd data buffers are full and no additional data can be 
// stored
//
bool FlexDataRead::ReadDataCIF(
    PMBatchStrData& bsd,
    const int queryblocks,
    char* const * const * const querypmbegs,
    char* const * const * const querypmends)
{
    MYMSG("FlexDataRead::ReadDataCIF",6);
    static const std::string preamb = "FlexDataRead::ReadDataCIF: ";
    //constexpr size_t maxlnwidth = 128;
    constexpr size_t sfflnwidth = 54;//sufficient linewidth
    static const int teropt = CLOptions::GetI_TER();
    static const int splitopt = CLOptions::GetI_SPLIT();
    static const int hetatm = CLOptions::GetI_HETATM();
    static const std::string satom = CLOptions::GetI_ATOM_PROT_trimmed();
    const bool usechid = !(teropt == CLOptions::istEOF &&
        (splitopt == CLOptions::issaNoSplit || splitopt == CLOptions::issaByMODEL));
    const bool usemodn = !(teropt == CLOptions::istEOF &&
        (splitopt == CLOptions::issaNoSplit));
    //moved to constructor:
    //static std::string strmodel(4, 0);//allocate space once for model number string
    //static std::string strchain(5, 0);//allocate space once for chain string
    //static std::string strresnum(5, 0);//allocate space once for resnum
    //static std::string strcoord(9, 0);//allocate space once for coordinate

    assert(satom.size()<=4);

    //statics moved to constructor:
    //static int fieldndxs[nciffields] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    //static const char* ptks[maxlnwidth];//array of pointers to tokens in a line
    //static const char* ptkends[maxlnwidth];//array of pointers to the ends of tokens in a line
    std::string strmodelprev(4, 0);//previous model number string
    std::string strchainprev(5, 0);//previous chain ID string
    int loop = -1;//section indicator and field index value
    int moltype = 0;//molecular type: protein or RNA
    char chord = 0;//chain serial number (order)
    char chid, insc;//chain ID and res. insertion code
    CHTYPE resd;//one-letter residue code
    INTYPE resnum;//residue serial number 
    FPTYPE coords[pmv2DNoElems];//residue coordinates


    //set field index and check for completeness
    std::function<void(const char*, size_t, int)> 
    lfSetFieldNdx = [this](const char* p, size_t dw, int ndx) {
        //TODO: lengths hard coded for speed
        if(dw >= 9 && memcmp(p,"group_PDB",9) == 0) fieldndxs_[group_PDB] = ndx;
        else if(dw >= 13 && memcmp(p,"label_atom_id",13) == 0) fieldndxs_[label_atom_id] = ndx;
        else if(dw >= 12 && memcmp(p,"label_alt_id",12) == 0) fieldndxs_[label_alt_id] = ndx;
        else if(dw >= 13 && memcmp(p,"label_comp_id",13) == 0) fieldndxs_[label_comp_id] = ndx;
        else if(dw >= 13 && memcmp(p,"label_asym_id",13) == 0) fieldndxs_[label_asym_id] = ndx;
        else if(dw >= 12 && memcmp(p,"label_seq_id",12) == 0) fieldndxs_[label_seq_id] = ndx;
        else if(dw >= 17 && memcmp(p,"pdbx_PDB_ins_code",17) == 0) fieldndxs_[pdbx_PDB_ins_code] = ndx;
        else if(dw >= 7 && memcmp(p,"Cartn_x",7) == 0 && std::isspace(p[7])) fieldndxs_[Cartn_x] = ndx;
        else if(dw >= 7 && memcmp(p,"Cartn_y",7) == 0 && std::isspace(p[7])) fieldndxs_[Cartn_y] = ndx;
        else if(dw >= 7 && memcmp(p,"Cartn_z",7) == 0 && std::isspace(p[7])) fieldndxs_[Cartn_z] = ndx;
        else if(dw >= 11 && memcmp(p,"auth_seq_id",11) == 0) fieldndxs_[auth_seq_id] = ndx;
        else if(dw >= 12 && memcmp(p,"auth_asym_id",12) == 0) fieldndxs_[auth_asym_id] = ndx;
        else if(dw >= 18 && memcmp(p,"pdbx_PDB_model_num",18) == 0) fieldndxs_[pdbx_PDB_model_num] = ndx;
        //check for completeness
        if(0 <= fieldndxs_[group_PDB] && 
           0 <= fieldndxs_[label_atom_id] && 0 <= fieldndxs_[label_comp_id] && 
            (0 <= fieldndxs_[auth_asym_id] || 0 <= fieldndxs_[label_asym_id]) && 
            (0 <= fieldndxs_[auth_seq_id] || 0 <= fieldndxs_[label_seq_id]) &&
           0 <= fieldndxs_[Cartn_x] && 0 <= fieldndxs_[Cartn_y] && 0 <= fieldndxs_[Cartn_z])
            fieldndxs_[mysetflag] = 1;
    };

    //check atom type; return true if it matches satom
    std::function<bool(const char*)> lfCheckAtom = [](const char* pt) {
        if(*pt == '"') pt++;
        for(size_t i = 0; i < satom.size(); i++)
            if(satom[i] != *pt++)
                return false;
        if(*pt!='"' && *pt!=' ' && *pt!='\t' && *pt!='\n' && *pt!='\r')
            return false;
        return true;
    };

    //print warning only without setting eof
    std::function<void(int)> lfWarning = [this,&strchainprev](int warn) {
        if(clustering_ && !clustmaster_) return;
        switch(warn) {
            case PMBatchStrData::pmbsdfShort:
                warning((std::string("Chain ") + strchainprev + 
                " too short and ignored; File: " + db_entrydesc_).c_str());
                break;
        }
    };

    //set Eof and reset chain and molecular serial numbers; print warning
    std::function<void(int)> lfSetEof = [this](int warn) {
        memset(fieldndxs_, 0xff, nciffields * sizeof(fieldndxs_[mysetflag]));
        SetEof();
        if(clustering_ && !clustmaster_) return;
        switch(warn) {
            case PMBatchStrData::pmbsdfEmpty:
                warning(("No atoms selected; File ignored: " + 
                db_entrydesc_).c_str());
                break;
            case PMBatchStrData::pmbsdfAbandoned:
                warning(("Structure too big; File ignored: " + 
                db_entrydesc_).c_str());
                break;
        }
    };

    //increment the structure counter associated with the current file
    std::function<void()> lfIncStr = [this]() {
        if(!clustering_) return;
        if(structcounter_.size() <= currentfilendx_)
            throw MYRUNTIME_ERROR2(
            preamb + "Invalid structure counter.", CRITICAL);
        structcounter_[currentfilendx_]++;
        if(!clustmaster_) return;
        globalids_[currentfilendx_].push_back(idgenerator_++);
        if((int)globalids_[currentfilendx_].size() != structcounter_[currentfilendx_])
            throw MYRUNTIME_ERROR2(
            preamb + "Inconsistent structure counter.", CRITICAL);
    };

    //get the global structure ID for the current structure of the current file
    std::function<int(int)> lfGetGlobID = [this](int moltype) {
        if(!clustering_) return moltype;
        if(clustmaster_) return idgenerator_;//(0 <= moltype)? idgenerator_: -idgenerator_;
        if(structcounter_.size() <= currentfilendx_)
            throw MYRUNTIME_ERROR2(
            preamb + "Invalid structure counter (2).", CRITICAL);
        if((int)globalids_[currentfilendx_].size() <= structcounter_[currentfilendx_])
            //master hasn't yet processed all structures of the current file
            return INT_MAX;
        return globalids_[currentfilendx_][ structcounter_[currentfilendx_] ];
    };


    while(!Eof()) {
        if(profile_buffer_.eof())
            NextDataPage();

        if(profile_buffer_.datlen_ < profile_buffer_.curpos_) {
            bsd.Fallback();
            lfSetEof(false);
            throw MYRUNTIME_ERROR2(
            preamb + "Invalid file buffer's position.", 
            CRITICAL);
        }

        if(profile_buffer_.datlen_ - profile_buffer_.curpos_ < maxlnwidth)
            NextDataPage();

        const char* pbeg = profile_buffer_.data_ + profile_buffer_.curpos_;
        const char* pend = profile_buffer_.data_ + profile_buffer_.datlen_;
        const char* p = pbeg;

        //locate the first occurrence of NL in the buffer
        //NOTE: located on the next iteration through .incposnl
        //for(; p < pend && *p != '\n' && *p != '\r'; p++);
        //for(; p < pend && (*p == '\n' || *p == '\r'); p++);

        size_t datwidth = (size_t)(pend-p);

        if(datwidth < 5 || *p == '#') {
            memset(fieldndxs_, 0xff, nciffields * sizeof(fieldndxs_[mysetflag]));
            profile_buffer_.incposnl((size_t)(p-pbeg));
            continue;
        }

        if(memcmp(p,"loop_",5) == 0) {
            memset(fieldndxs_, 0xff, nciffields * sizeof(fieldndxs_[mysetflag]));
            loop = 0;
            profile_buffer_.incposnl((size_t)(p-pbeg) + 5);
            continue;
        }

        if(0 <= loop && (datwidth < 11 || memcmp(p,"_atom_site.",11) != 0))
            loop = -1;

        if(0 <= loop) {
            //set a field index and increment the loop index:
            lfSetFieldNdx(p+11, datwidth-11, loop++);
            profile_buffer_.incposnl((size_t)(p-pbeg) + 11);
            continue;
        }

        if(/*loop < 0 && */fieldndxs_[mysetflag] <= 0) {
            //neither loop nor atom section
            profile_buffer_.incposnl((size_t)(p-pbeg));
            continue;
        }

        int ptksndxs = 0;//# ptks indices
        //NOTE: p no longer points to the beginning of the line
        for(const char* pmax = p + maxlnwidth; p < pend && p < pmax && *p!='\n' && *p!='\r';) {
            for(; p < pend && p < pmax && (*p==' '||*p=='\t'); p++);
            ptks_[ptksndxs] = p;
            for(; p < pend && p < pmax && *p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'; p++);
            ptkends_[ptksndxs++] = p;
        }

        if(ptksndxs < 1 || datwidth < sfflnwidth) {
            //memset(fieldndxs, 0xff, nciffields * sizeof(fieldndxs[mysetflag]));
            profile_buffer_.incposnl((size_t)(p-pbeg));
            continue;
        }

        if(ptksndxs <= fieldndxs_[group_PDB] ||
           (memcmp(ptks_[fieldndxs_[group_PDB]],"ATOM",4) != 0 && 
                (!hetatm || memcmp(ptks_[fieldndxs_[group_PDB]],"HETATM",6) != 0)))
        {
            profile_buffer_.incposnl((size_t)(p-pbeg));
            continue;//this is not an atom section
        }

        if(0 <= fieldndxs_[label_alt_id] && (
            ptksndxs <= fieldndxs_[label_alt_id] || (
             *ptks_[fieldndxs_[label_alt_id]] != '.' && 
             *ptks_[fieldndxs_[label_alt_id]] != 'A'))) 
        {
            profile_buffer_.incposnl((size_t)(p-pbeg));
            continue;//invalid alternate location indicator (considered)
        }

        if(ptksndxs <= fieldndxs_[label_atom_id] ||
           lfCheckAtom(ptks_[fieldndxs_[label_atom_id]]) == false) 
        {
            profile_buffer_.incposnl((size_t)(p-pbeg));
            continue;//not the atom of interest
        }

        if(ptksndxs <= fieldndxs_[Cartn_x] ||
           ptksndxs <= fieldndxs_[Cartn_y] ||
           ptksndxs <= fieldndxs_[Cartn_z] ||
           ptksndxs <= fieldndxs_[label_seq_id] ||
           ptksndxs <= fieldndxs_[label_asym_id] ||
           ptksndxs <= fieldndxs_[label_comp_id]) 
        {
            profile_buffer_.incposnl((size_t)(p-pbeg));
            continue;
        }

        int flndx = label_asym_id;
        if(0 <= fieldndxs_[auth_asym_id] && fieldndxs_[auth_asym_id] < ptksndxs)
            flndx = auth_asym_id;

        size_t sztk = (size_t)(ptkends_[fieldndxs_[flndx]]-ptks_[fieldndxs_[flndx]]);
        if(sztk < 1) {
            bsd.Fallback();
            lfSetEof(false);
            throw MYRUNTIME_ERROR(preamb + "Invalid PDBx/mmCIF file format.");
        }

        strchain_.assign(ptks_[fieldndxs_[flndx]], sztk);
        chid = strchain_[0];//chain id
        if(chid == '.') chid = ' ';

        insc = ' ';
        if(0 <= fieldndxs_[pdbx_PDB_ins_code] && fieldndxs_[pdbx_PDB_ins_code] < ptksndxs)
            insc = *ptks_[fieldndxs_[pdbx_PDB_ins_code]];//insertion code
        if(insc == '?') insc = ' ';

        resd = ResName2Code(ptks_[fieldndxs_[label_comp_id]]);

        try {
            flndx = label_seq_id;
            if(0 <= fieldndxs_[auth_seq_id] && fieldndxs_[auth_seq_id] < ptksndxs)
                flndx = auth_seq_id;
            resnum = std::stoi(strresnum_.assign(ptks_[fieldndxs_[flndx]],
                (size_t)(ptkends_[fieldndxs_[flndx]]-ptks_[fieldndxs_[flndx]])));//residue serial number 

            coords[0] = std::stof(strcoord_.assign(ptks_[fieldndxs_[Cartn_x]],
                (size_t)(ptkends_[fieldndxs_[Cartn_x]]-ptks_[fieldndxs_[Cartn_x]])));

            coords[1] = std::stof(strcoord_.assign(ptks_[fieldndxs_[Cartn_y]],
                (size_t)(ptkends_[fieldndxs_[Cartn_y]]-ptks_[fieldndxs_[Cartn_y]])));

            coords[2] = std::stof(strcoord_.assign(ptks_[fieldndxs_[Cartn_z]],
                (size_t)(ptkends_[fieldndxs_[Cartn_z]]-ptks_[fieldndxs_[Cartn_z]])));
        } catch(...) {
            bsd.Fallback();
            lfSetEof(false);
            throw MYRUNTIME_ERROR(preamb +
                "Failed to read residue data from PDBx/mmCIF file.");
        }

        bool seteof = false;
        bool finalize = false;

        if(0 <= fieldndxs_[pdbx_PDB_model_num] && fieldndxs_[pdbx_PDB_model_num] < ptksndxs) {
            strmodel_.assign(ptks_[fieldndxs_[pdbx_PDB_model_num]], 
                (size_t)(ptkends_[fieldndxs_[pdbx_PDB_model_num]]-ptks_[fieldndxs_[pdbx_PDB_model_num]]));

            if(strmodel_ != strmodelprev) {
                if(teropt >= CLOptions::istEND) {
                    finalize = true;
                    seteof = true;
                }
                if(splitopt > CLOptions::issaNoSplit)
                    finalize = true;
            }
        }

        if(strchain_ != strchainprev) {
            if(teropt >= CLOptions::istENDorChain) {
                finalize = true;
                seteof = true;
            }
            if(splitopt >= CLOptions::issaByChain)
                finalize = true;
            chord++;
        }

        if(finalize && bsd.ContainsDataLast()) {
            chord = 1;//reset chain number on each structure finalization
            if(seteof) lfSetEof(false);
            switch(bsd.FinalizeCrntStructure(
                    currentfilendx_, clustering_? structcounter_[currentfilendx_]: 0,
                    lfGetGlobID(moltype), db_entrydesc_, 
                    strchainprev, strmodelprev, usechid, usemodn,
                    queryblocks, querypmbegs, querypmends)) 
            {
                case PMBatchStrData::pmbsdfEmpty: 
                    lfSetEof(PMBatchStrData::pmbsdfEmpty); return true;
                case PMBatchStrData::pmbsdfShort: 
                    lfWarning(PMBatchStrData::pmbsdfShort); break;
                case PMBatchStrData::pmbsdfAbandoned: 
                    lfSetEof(PMBatchStrData::pmbsdfAbandoned); return true;
                case PMBatchStrData::pmbsdfLowSimilarity: return true;
                //structure does not fit into the allocation limits;
                //profile_buffer_.curpos_ is left as is to read the same line again later
                case PMBatchStrData::pmbsdfLimits: lfIncStr(); return false;
                case PMBatchStrData::pmbsdfWritten: lfIncStr(); break;
            }
            if(seteof) return true;
            // if(seteof) {
            //     lfSetEof(false);
            //     return true;
            // }
            moltype = 0;//reset mol. type having finished finalization 
        }

        strmodelprev = strmodel_;
        strchainprev = strchain_;

        sztk = (size_t)(ptkends_[fieldndxs_[label_comp_id]]-ptks_[fieldndxs_[label_comp_id]]);
        if(sztk < 2 || (sztk==2 && *ptks_[fieldndxs_[label_comp_id]]=='D')) moltype++;
        else moltype--;

        if(!bsd.AddOneResidue(maxstrlen_, resd, resnum, insc, chid, chord, coords)) {
            lfSetEof(PMBatchStrData::pmbsdfAbandoned);
            return true;
        }

        profile_buffer_.incposnl((size_t)(p-pbeg));
    }//while(!Eof())

    switch(bsd.FinalizeCrntStructure(
            currentfilendx_, clustering_? structcounter_[currentfilendx_]: 0,
            lfGetGlobID(moltype), db_entrydesc_, 
            strchainprev, strmodelprev, usechid, usemodn,
            queryblocks, querypmbegs, querypmends))
    {
        case PMBatchStrData::pmbsdfEmpty: 
            lfSetEof(PMBatchStrData::pmbsdfEmpty); return true;
        case PMBatchStrData::pmbsdfShort: 
            lfWarning(PMBatchStrData::pmbsdfShort); break;
        case PMBatchStrData::pmbsdfAbandoned: 
            lfSetEof(PMBatchStrData::pmbsdfAbandoned); return true;
        case PMBatchStrData::pmbsdfLowSimilarity: return true;
        case PMBatchStrData::pmbsdfLimits: lfIncStr(); return false;
        case PMBatchStrData::pmbsdfWritten: lfIncStr(); break;
    }

    memset(fieldndxs_, 0xff, nciffields * sizeof(fieldndxs_[mysetflag]));
    return true;
}












// =========================================================================
// NextDataPageZip: read and cache the next page from the zip file;
//
void FlexDataRead::NextDataPageZip()
{
    MYMSG("FlexDataRead::NextDataPageZip",8);
    if(InflatePage(current_zipdata_, current_strdata_))
        MoveDataFromPageToProfileBuffer();
}

// -------------------------------------------------------------------------
// NextDataPageDirect: read and cache the next page from the file;
//
void FlexDataRead::NextDataPageDirect()
{
    MYMSG("FlexDataRead::NextDataPageDirect",8);
    if(ReadPage(current_strdata_))
        MoveDataFromPageToProfileBuffer();
}

// -------------------------------------------------------------------------
// ReadPage: read at most one page from the file;
// character stream chstr is modified on exit and
// NOTE: the data buffer field of chstr is assumed to be preallocated to the
// page size
//
bool FlexDataRead::ReadPage(TCharStream& chstr)
{
    MYMSG("FlexDataRead::ReadPage",8);
    static const std::string preamb = "FlexDataRead::ReadPage: ";

    if(!ValidFileDescriptor())
        throw MYRUNTIME_ERROR2(preamb + "Invalid file descriptor.", CRITICAL);

    if( !chstr.data_ || chstr.pagesize_ < 1 )
        throw MYRUNTIME_ERROR2(preamb + "Invalid argument.", CRITICAL);

    if( db_filesize_ <= chstr.pageoff_)
        return false;

    size_t nbytes = db_filesize_ - chstr.pageoff_;

    if( chstr.pagesize_ < nbytes )
        nbytes = chstr.pagesize_;

    ssize_t bytesread;

#ifdef OS_MS_WINDOWS
    DWORD bytesreadhlp = 0;
    if( !ReadFile(
        db_fp_,
        chstr.data_,
        (DWORD)nbytes,//nNumberOfBytesToRead (DWORD)
        &bytesreadhlp,//lpNumberOfBytesRead (LPDWORD)
        NULL)//lpOverlapped
      )
        throw MYRUNTIME_ERROR2(preamb + 
        "Read from file failed:" + db_filename_, CRITICAL);

    bytesread = bytesreadhlp;
#else
    bytesread = read(db_fp_, chstr.data_, nbytes );
#endif

    if(bytesread < 0)
        throw MYRUNTIME_ERROR2(preamb +
        "Failed to read form file: " + db_filename_, CRITICAL);

    chstr.datlen_ = bytesread;
    chstr.curpos_ = 0;
    if( 0 < bytesread ) {
        chstr.pagenr_++;
        chstr.pageoff_ += bytesread;
    }

    return true;
}

// -------------------------------------------------------------------------
// InflatePage: (read if necessary and) inflate at most one page from
// gzipped data;
// character streams are modified on exit;
// NOTE: the data buffer field is assumed to be preallocated to the page size;
//
bool FlexDataRead::InflatePage(TCharStream& z_chstr, TCharStream& out_chstr)
{
    MYMSG("FlexDataRead::InflatePage",8);
    static const std::string preamb = "FlexDataRead::InflatePage: ";

    if(!out_chstr.data_ || out_chstr.pagesize_ < 1)
        throw MYRUNTIME_ERROR2(preamb + "Invalid argument.", CRITICAL);

    //NOTE: do not use offset function here, as it tells buffered position,
    //NOTE: including portion of actually unprocessed/not received data:
    if(db_filesize_ <= out_chstr.pageoff_)
       return false;

    if(z_stream_.avail_in < 1) {
        ReadPage(z_chstr);
        //make sure to inflate what belongs to compressed data:
        z_stream_.avail_in = mymin(z_chstr.datlen_, (db_filesize_ - out_chstr.pageoff_));
        z_stream_.next_in = (Bytef*)(z_chstr.data_);
    }

    ssize_t bytesconsumed = z_stream_.avail_in;
    ssize_t bytesinflated = out_chstr.pagesize_;

    z_stream_.avail_out = out_chstr.pagesize_;
    z_stream_.next_out = (Bytef*)(out_chstr.data_);

    int ret = inflate(&z_stream_, Z_NO_FLUSH);
    switch(ret) {
        case Z_MEM_ERROR:
        case Z_DATA_ERROR:
        case Z_STREAM_ERROR:
        case Z_NEED_DICT:
            throw MYRUNTIME_ERROR2(preamb + 
            "Failed to read compressed data from file: " + db_entrydesc_, CRITICAL);
    }

    bytesconsumed -= (ssize_t)z_stream_.avail_in;
    bytesinflated -= (ssize_t)z_stream_.avail_out;

    if(bytesconsumed < 0 || bytesinflated < 0)
        throw MYRUNTIME_ERROR2(preamb + 
        "Unexpected error upon decompressing data from file: " + db_entrydesc_, CRITICAL);

    out_chstr.datlen_ = bytesinflated;
    out_chstr.curpos_ = 0;
    if(0 < bytesinflated) out_chstr.pagenr_++;
    out_chstr.pageoff_ += (size_t)bytesconsumed;

    if(z_stream_.avail_in || out_chstr.pagesize_ <= (size_t)bytesinflated ||
       db_filesize_ <= out_chstr.pageoff_)
        return true;


    //ROUND 2: read compressed data again if there is space for inflation:
    // if(z_stream_.avail_in < 1 && bytesinflated < out_chstr.pagesize_ &&
    //    out_chstr.pageoff_ < db_filesize_)

    ReadPage(z_chstr);
    z_stream_.avail_in = mymin(z_chstr.datlen_, (db_filesize_ - out_chstr.pageoff_));
    z_stream_.next_in = (Bytef*)(z_chstr.data_);

    bytesconsumed = z_stream_.avail_in;

    z_stream_.next_out = (Bytef*)(out_chstr.data_ + bytesinflated);
    bytesinflated = (out_chstr.pagesize_ - (size_t)bytesinflated);
    z_stream_.avail_out = bytesinflated;

    ret = inflate(&z_stream_, Z_NO_FLUSH);
    switch(ret) {
        case Z_MEM_ERROR:
        case Z_DATA_ERROR:
        case Z_STREAM_ERROR:
        case Z_NEED_DICT:
            throw MYRUNTIME_ERROR2(preamb + 
            "Failed to read compressed data from file: " + db_entrydesc_, CRITICAL);
    }

    bytesconsumed -= (ssize_t)z_stream_.avail_in;
    bytesinflated -= (ssize_t)z_stream_.avail_out;

    if(bytesconsumed < 0 || bytesinflated < 0)
        throw MYRUNTIME_ERROR2(preamb + 
        "Unexpected error upon decompressing data from file: " + db_entrydesc_, CRITICAL);

    out_chstr.datlen_ += bytesinflated;
    out_chstr.pageoff_ += (size_t)bytesconsumed;

    return true;
}
