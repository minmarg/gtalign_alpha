/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __FlexDataRead_h__
#define __FlexDataRead_h__

#ifdef OS_MS_WINDOWS
#   include <Windows.h>
#else
#   include <sys/types.h>
#   include <sys/stat.h>
#   include <sys/mman.h>
#   include <unistd.h>
#   include <fcntl.h>
#endif

#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <thread>

#include <fstream>

#include "extzlib/zlib.h"
#include "libutil/mybase.h"
#include "libutil/CLOptions.h"
#include "InputFilelist.h"
#include "PMBatchStrData.h"

// -------------------------------------------------------------------------
// fields of interest for mmCIF format
enum {
    mysetflag,//whether the field indices have been set
    group_PDB,
    label_atom_id,
    label_alt_id,
    label_comp_id,
    label_asym_id,
    label_seq_id,
    pdbx_PDB_ins_code,
    Cartn_x,
    Cartn_y,
    Cartn_z,
    auth_seq_id,
    auth_asym_id,
    pdbx_PDB_model_num,
    nciffields
};
// -------------------------------------------------------------------------
// _________________________________________________________________________
// Class FlexDataRead
//
// Provides interface to reading files of different formats
//
class FlexDataRead
{
    enum {
        CUDBREADER_TMPBUFFSIZE = 4096
    };

    static constexpr size_t maxlnwidth = 128;//max line width

public:
    FlexDataRead(
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
            const bool clustering = false,
            const bool clustmaster = false);

    virtual ~FlexDataRead();

    virtual void Destroy();

    virtual void Open();//open database
    virtual void Close();//close file and reset associated data

    bool GetMapped() const {return mapped_;}

    bool EndOfData() {
        bool eod = (filendxlist_.size() <= currentfilendx_)
            ?   true
            :   ((filendxlist_.size() <= currentfilendx_ + ndxstep_)
                ?   Eof()
                :   false
                );
        if(!clustering_ || clustmaster_ || eod) return eod;
        if(CLOptions::GetB_CACHE_ON()) return eod;
        return ((int)globalids_[currentfilendx_].size() < structcounter_[currentfilendx_]);
    }

    bool Eof() const { return EofDirect(); }

    int GetGlobalId(size_t filendx, int strndx) const {
        if(globalids_.size() <= filendx) return INT_MAX;
        if((int)globalids_[filendx].size() <= strndx) return INT_MAX;
        return globalids_[filendx][strndx];
    }

    size_t GetCurrentFileSize() const {return db_filesize_;}

    bool ReadData(
        PMBatchStrData&,
        const int queryblocks,
        char* const * const * const querypmbegs,
        char* const * const * const querypmends);

protected:
    void Execute(void* args);
    void Initializer();

    bool FileTypeIsZip() const {
        return
            InputFilelist::FDRFlZip <= db_filetype_ &&
            db_filetype_ <= InputFilelist::FDRFlPDBxmmCIFZip;
    }

    void NextDataPage() {
        if(FileTypeIsZip()) NextDataPageZip();
        else NextDataPageDirect();
    }
    void NextDataPageDirect();
    void NextDataPageZip();


    bool ReadDataPDB(
        PMBatchStrData&,
        const int queryblocks,
        char* const * const * const querypmbegs,
        char* const * const * const querypmends);
    bool ReadDataCIF(
        PMBatchStrData&,
        const int queryblocks,
        char* const * const * const querypmbegs,
        char* const * const * const querypmends);


    void MoveDataFromPageToProfileBuffer();

    void ResetZipPageData() {
        if(/* z_stream_.avail_in &&  */z_stream_.next_in) {
            //z_stream_ was not reset before
            if(inflateReset2(&z_stream_, (MAX_WBITS+16)) != Z_OK)
                throw MYRUNTIME_ERROR(
                "FlexDataRead::ResetZipPageData: Failed to reset z_stream.");
        }
        z_stream_.avail_in = 0;
        z_stream_.next_in = Z_NULL;
        current_zipdata_.datlen_ = 0;
        current_zipdata_.curpos_ = 0;
        current_zipdata_.pagenr_ = 0;
        current_zipdata_.pageoff_ = 0;
    }
    void ResetCurrentPageData() {
        current_strdata_.datlen_ = 0;
        current_strdata_.curpos_ = 0;
        current_strdata_.pagenr_ = 0;
        current_strdata_.pageoff_ = 0;
    }
    void ResetProfileBufferData() {
        profile_buffer_.datlen_ = 0;
        profile_buffer_.curpos_ = 0;
        profile_buffer_.pagenr_ = 0;
        profile_buffer_.pageoff_ = 0;
    }

    void MoveBlockOfData( 
        char* stream, size_t szstream, 
        size_t dstpos, size_t srcpos, size_t srclen );


    bool EofDirect() const {
        return
            db_filesize_ <= current_strdata_.pageoff_ &&
            profile_buffer_.datlen_ <= profile_buffer_.curpos_;
    }

    void SetEof() {
        profile_buffer_.curpos_ = profile_buffer_.datlen_;
        current_strdata_.pageoff_ = db_filesize_;
    }


    void ResetDbMetaData() {
        db_position_ = 0;
        db_filesize_ = 0;
        db_filetype_ = InputFilelist::FDRFlN;
        db_parenttype_ = InputFilelist::FDRFlN;
        db_filename_.clear();
        db_parentname_.clear();
        db_entrydesc_.clear();
    }

    void SetDbMetaData() {
        ResetDbMetaData();
        if(filendxlist_.size() <= currentfilendx_)
            return;
        size_t ndx = filendxlist_[currentfilendx_];
        db_position_ = strfilepositionlist_[ndx];
        db_filesize_ = strfilesizelist_[ndx];
        db_parenttype_ = strparenttypelist_[ndx];
        db_filetype_ = strfiletypelist_[ndx];
        db_filename_ = strfilelist_[ndx];
        db_parentname_ = pntfilelist_[ndx];
        db_entrydesc_ =
            (db_parenttype_ == InputFilelist::FDRFlTar)
            ? db_parentname_ + ":" + db_filename_
            : db_filename_;
        if(clustering_ && !clustmaster_)
            std::fill(structcounter_.begin(), structcounter_.end(), 0);
    }

    size_t GetPageSize() const {return current_strdata_.pagesize_;}
    static ssize_t ObtainPageSize();

    TCharStream* GetProfileBuffer() {return &profile_buffer_;}
    void ResetProfileBufferPosition() {profile_buffer_.curpos_ = 0;}

    bool ValidFileDescriptor();
    void InvalidateFileDescriptor();

    void OpenFile(const char*, size_t);
    void CloseFile();

    void MapFile();
    void UnmapFile();

    bool ReadPage(TCharStream&);
    bool InflatePage(TCharStream& z_chstr, TCharStream& out_chstr);

private:
    const std::vector<std::string>& strfilelist_;//structure file list
    const std::vector<std::string>& pntfilelist_;//parent file list
    const std::vector<size_t>& strfilepositionlist_;//list of structure file positions within an archive
    const std::vector<size_t>& strfilesizelist_;//list of structure file sizes
    const std::vector<int>& strparenttypelist_;//parent file type list of structure files (e.g, tar)
    const std::vector<int>& strfiletypelist_;//list of structure file types
    const std::vector<size_t>& filendxlist_;//list of the indices of files sorted by filesize
    std::vector<std::vector<int>>& globalids_;//global ids for structures across all files
    std::vector<int> structcounter_;//structure-per-file counter when clustering is on

    const size_t ndxstartwith_;//file index to start with
    const size_t ndxstep_;//file index step

    size_t currentfilendx_;//current index of a file in processing

    const int maxstrlen_;//maximum structure length (-1, unlimited)
    const bool mapped_;//whether a database is memory-mapped

    const bool clustering_;//flag indicating clustering
    const bool clustmaster_;//when clustering is on, the master will create global ids
    int idgenerator_;//master's id generator for clustering

    size_t db_position_;//archive position (if an archive) of the current file
    size_t db_filesize_;//size of the current file in bytes
    int db_parenttype_;//type of the parent of the current file
    int db_filetype_;//type of the current file
    std::string db_filename_;//name of the current file
    std::string db_parentname_;//parent name of the current file
    std::string db_entrydesc_;//entry (current file) description

#ifdef OS_MS_WINDOWS
    HANDLE db_fp_;//database/regular file descriptor
#else
    int db_fp_;//database/regular file descriptor
#endif
    gzFile gzfp_;//file descriptor for zip files

    z_stream z_stream_;//zlib stream for compressed files
    TCharStream current_zipdata_;//current zip data in stream
    TCharStream current_strdata_;//current stream data
    TCharStream profile_buffer_;//stream for processing profile data
    char tmpbuff_[CUDBREADER_TMPBUFFSIZE];//buffer for temporary data

    //working object-specific variables
    std::string strmodel_;
    std::string strchain_;
    std::string strresnum_;
    std::string strcoord_;
    int fieldndxs_[nciffields];
    const char* ptks_[maxlnwidth];//array of pointers to tokens in a line
    const char* ptkends_[maxlnwidth];//array of pointers to the ends of tokens in a line
};


// /////////////////////////////////////////////////////////////////////////
// INLINES
//
// -------------------------------------------------------------------------
// MoveDataFromPageToProfileBuffer: move data from the current page to the
// file buffer and accordingly adjust the data within the structures
inline
void FlexDataRead::MoveDataFromPageToProfileBuffer()
{
    MYMSG("FlexDataRead::MoveDataFromPageToProfileBuffer",7);

    size_t leftinstr = (current_strdata_.curpos_ <= current_strdata_.datlen_)?
        current_strdata_.datlen_ - current_strdata_.curpos_: 0;

    if( leftinstr < 1 )
        return;

    if( !profile_buffer_.data_ ||
        profile_buffer_.pagesize_ < profile_buffer_.datlen_)
        throw MYRUNTIME_ERROR(
        "FlexDataRead::MoveDataFromPageToProfileBuffer: Invalid file buffer data.");

    if( profile_buffer_.pagesize_ < profile_buffer_.datlen_ + leftinstr)
    {
        if( profile_buffer_.datlen_ < profile_buffer_.curpos_)
            throw MYRUNTIME_ERROR(
            "FlexDataRead::MoveDataFromPageToProfileBuffer: "
            "Invalid file buffer's position.");

        MoveBlockOfData(
            profile_buffer_.data_, profile_buffer_.pagesize_, 
            0, profile_buffer_.curpos_, 
            profile_buffer_.datlen_ - profile_buffer_.curpos_ );

        profile_buffer_.datlen_ -= profile_buffer_.curpos_;
        profile_buffer_.curpos_ = 0;

        if( profile_buffer_.pagesize_ < profile_buffer_.datlen_ + leftinstr)
            throw MYRUNTIME_ERROR(
            "FlexDataRead::MoveDataFromPageToProfileBuffer: Too small file buffer size.");
    }

    if( !current_strdata_.data_)
        throw MYRUNTIME_ERROR(
        "FlexDataRead::MoveDataFromPageToProfileBuffer: Null page data.");

    memcpy(profile_buffer_.data_ + profile_buffer_.datlen_,
            current_strdata_.data_ + current_strdata_.curpos_, leftinstr);

    profile_buffer_.datlen_ += leftinstr;
    current_strdata_.datlen_ = 0;
    current_strdata_.curpos_ = 0;
}


// -------------------------------------------------------------------------
// MoveBlockOfData: move a block of data at position srcpos of the given 
// stream to the destination position;
// stream and szstream, stream and its size;
// dstpos, destination position in the stream;
// srcpos, source position in the stream;
// srclen, size of a block to move;
inline
void FlexDataRead::MoveBlockOfData( 
    char* stream, size_t szstream, 
    size_t dstpos, size_t srcpos, size_t srclen )
{
    MYMSG("FlexDataRead::MoveBlockOfData",6);

    if( !stream || szstream < 1 )
        return;

    if( dstpos == srcpos || srclen < 1)
        return;

    if( szstream < srclen || szstream < dstpos + srclen )
        throw MYRUNTIME_ERROR("FlexDataRead::MoveBlockOfData: Invalid parameters.");

    if( (dstpos < srcpos) 
        ?   dstpos + srclen < srcpos
        :   srcpos + srclen < dstpos )
    {
        memcpy(stream + dstpos, stream + srcpos, srclen);
        return;
    }

    for( size_t szmvd = 0; szmvd < srclen; ) {
        size_t szchnk = PCMIN((size_t)CUDBREADER_TMPBUFFSIZE, srclen - szmvd);
        size_t offset =
            (srcpos < dstpos )
            ?   srclen - szmvd - szchnk
            :   szmvd;
        memcpy(tmpbuff_, stream + srcpos + offset, szchnk);
        memcpy(stream + dstpos + offset, tmpbuff_, szchnk);
        szmvd += szchnk;
    }
}



// -------------------------------------------------------------------------
// GetPageSize: get the system page size
//
inline
ssize_t FlexDataRead::ObtainPageSize()
{
#ifdef OS_MS_WINDOWS
    SYSTEM_INFO systeminfo;
    GetSystemInfo(&systeminfo);
    return static_cast<ssize_t>(systeminfo.dwAllocationGranularity);
#else
    return sysconf(_SC_PAGESIZE);
#endif
}

// -------------------------------------------------------------------------
// OpenFile: open one of the files for reading
inline
void FlexDataRead::OpenFile(const char* filename, size_t position)
{
    if(filename == NULL)
        throw MYRUNTIME_ERROR("FlexDataRead::OpenFile: Null filename.");

    if(ValidFileDescriptor())
        throw MYRUNTIME_ERROR((std::string(
            "FlexDataRead::OpenFile: File has been already opened: ") +
            filename).c_str());

#ifdef OS_MS_WINDOWS
        db_fp_ = CreateFileA(
            filename,
            GENERIC_READ,
            FILE_SHARE_READ,
            NULL,//security descriptor: child processes cannot inherit the handle
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            NULL);
#else
        db_fp_ = open(filename, O_RDONLY);
#endif

    if(!ValidFileDescriptor())
        throw MYRUNTIME_ERROR((std::string(
            "FlexDataRead::OpenFile: Failed to open file: ") +
            filename).c_str());

    if(position < 1)
        return;

#ifdef OS_MS_WINDOWS
    LARGE_INTEGER offset;
    offset.QuadPart = position;
    offset.LowPart = SetFilePointer(
        db_fp_,
        offset.LowPart,//lDistanceToMove (LONG), lower double word
        &offset.HighPart,//lpDistanceToMoveHigh (PLONG), upper double word
        FILE_BEGIN
    );
    //NOTE: GetLastError() is thread-safe
    if(offset.LowPart == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR) {
        offset.QuadPart = -1;
        throw MYRUNTIME_ERROR(
        "FlexDataRead::OpenFile: Failed to set file position.");
    }
#else
    if(lseek(db_fp_, position, SEEK_SET) == (off_t)-1)
        throw MYRUNTIME_ERROR(
        "FlexDataRead::OpenFile: Failed to set file position.");
#endif
}

// CloseFile: close the file opened
inline
void FlexDataRead::CloseFile()
{
    if(ValidFileDescriptor())
    {
#ifdef OS_MS_WINDOWS
        CloseHandle(db_fp_);
#else
        close(db_fp_);
#endif
        InvalidateFileDescriptor();
    }
}

// -------------------------------------------------------------------------
// MapFile: memory-map the current file for reading
inline
void FlexDataRead::MapFile()
{
    if( !ValidFileDescriptor())
        throw MYRUNTIME_ERROR(
        "FlexDataRead::MapFile: Invalid file descriptor.");

    size_t mapsize = GetCurrentFileSize();

    if( mapsize < 1 )
        throw MYRUNTIME_ERROR(
        "FlexDataRead::MapFile: Invalid file size.");

#ifdef OS_MS_WINDOWS
    profile_buffer_.hMapFile_ = CreateFileMapping(
        db_fp_,
        NULL,//security descriptor: cannot be inherited
        PAGE_READONLY,
        //zeros indicate that the size of file is used for...
        // the maximum size of the file mapping object:
        ((uint64_t)mapsize)>>32/*0*/,//dwMaximumSizeHigh: upper double word
        ((uint64_t)mapsize)&0xffffffff/*0*/,//dwMaximumSizeLow: lower double word
        NULL);//no name

    if( profile_buffer_.hMapFile_ == NULL)
        throw MYRUNTIME_ERROR(
        "FlexDataRead::MapFile: Failed to create a file mapping object.");

    profile_buffer_.data_ = static_cast<char*>(
        MapViewOfFile(
            profile_buffer_.hMapFile_,
            FILE_MAP_READ,
            0,//dwFileOffsetHigh
            0,//dwFileOffsetLow
            0)//dwNumberOfBytesToMap: 
        //mapping extends from the specified offset to the end of the file mapping
    );

    if(profile_buffer_.data_ == NULL)
        throw MYRUNTIME_ERROR(
        "FlexDataRead::MapFile: Failed to map a view of the file mapping object.");
#else
    profile_buffer_.data_ = (char*)
        mmap(NULL, mapsize, PROT_READ, MAP_SHARED, db_fp_, 0);

    if(profile_buffer_.data_ == MAP_FAILED)
        throw MYRUNTIME_ERROR(
        "FlexDataRead::MapFile: Failed to memory-map an opened file of structures.");
#endif

    profile_buffer_.datlen_ = mapsize;
    profile_buffer_.curpos_ = 0;
    profile_buffer_.pagenr_ = 0;
    profile_buffer_.pagesize_ = mapsize;
    profile_buffer_.pageoff_ = 0;
}

// UnmapFile: unmap the opened file
inline
void FlexDataRead::UnmapFile()
{
#ifdef OS_MS_WINDOWS
    if( profile_buffer_.data_ )
        UnmapViewOfFile(profile_buffer_.data_);

    if( profile_buffer_.hMapFile_ )
        CloseHandle(profile_buffer_.hMapFile_);

    profile_buffer_.hMapFile_ = NULL;
#else
    if( profile_buffer_.data_ )
        munmap(profile_buffer_.data_, profile_buffer_.pagesize_);
#endif

    profile_buffer_.data_ = NULL;
    profile_buffer_.datlen_ = 0;
    profile_buffer_.curpos_ = 0;
    profile_buffer_.pagenr_ = 0;
    profile_buffer_.pagesize_ = 0;
    profile_buffer_.pageoff_ = 0;
}

// -------------------------------------------------------------------------
// ValidFileDescriptor: get a flag of whether the file descriptor is valid
inline
bool FlexDataRead::ValidFileDescriptor()
{
    return 
        db_filetype_ < InputFilelist::FDRFlN && 
        db_parenttype_ < InputFilelist::FDRFlN && 
        // (FileTypeIsZip()
        // ?   gzfp_ != NULL   :
#ifdef OS_MS_WINDOWS
            (db_fp_ != INVALID_HANDLE_VALUE)
#else
            (db_fp_ >= 0)
#endif
        ;// );
}
// InvalidateFileDescriptor: invalidate the given file descriptor
inline
void FlexDataRead::InvalidateFileDescriptor()
{
    gzfp_ = NULL;
#ifdef OS_MS_WINDOWS
    db_fp_ = INVALID_HANDLE_VALUE;
#else
    db_fp_ = -1;
#endif
}

#endif//__FlexDataRead_h__
