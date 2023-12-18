/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/mybase.h"

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

#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "libutil/mydirent.h"
#include "InputFilelist.h"

// -------------------------------------------------------------------------
//maximum depth of subdirectories considered
static constexpr int MAXDIRDEPTH_ = 3;

const std::string InputFilelist::knownexts_[] = {
    ".tar",//FDRKETar,//.tar file
    ".gz",//FDRKEGz,//.gz file
    ".pdb.gz",//FDRKEPDBGz,//.pdb.gz file
    ".ent.gz",//FDRKEENTGz,//.ent.gz file
    ".cif.gz",//FDRKECIFGz,//.cif.gz file
    ".pdb",//FDRKEPDB,//.pdb file
    ".ent",//FDRKEENT,//.ent file
    ".cif"//FDRKECIF,//.cif file
};

// -------------------------------------------------------------------------
// Constructor
//
InputFilelist::InputFilelist(
    const std::vector<std::string>& dnamelst, 
    const std::vector<std::string>& sfxlst,
    const bool clustering)
:
    dnamelst_(dnamelst),
    sfxlst_(sfxlst),
    clustering_(clustering)
{
    MYMSG("InputFilelist::InputFilelist",4);

    if(!dnamelst_.size())
        throw MYRUNTIME_ERROR("InputFilelist::InputFilelist: No structure files.");

    const size_t initnentries = 300 * KBYTE;

    try {
        strfilelist_.reserve(initnentries);
        pntfilelist_.reserve(initnentries);
        strfilepositionlist_.reserve(initnentries);
        strfilesizelist_.reserve(initnentries);
        strparenttypelist_.reserve(initnentries);
        strfiletypelist_.reserve(initnentries);
        filendxlist_.reserve(initnentries);
        if(clustering_) globalids_.reserve(initnentries);
    } catch(...) {
        throw MYRUNTIME_ERROR("InputFilelist::InputFilelist: Not enough memory.");
    }

    ConstructFileList();
}

// -------------------------------------------------------------------------
// Destructor
//
InputFilelist::~InputFilelist()
{
    MYMSG("InputFilelist::~InputFilelist",4);
}

// -------------------------------------------------------------------------
// AddFilesFromTAR: add files from a tar archive to a list of structure files
//
void InputFilelist::AddFilesFromTAR(const std::string& entryname)
{
    enum {
        tarHeaderSize = 512,//header size in bytes
        tarRecordSize = tarHeaderSize,//record size in bytes
        tarhaddrName = 0,//0-terminated name field address in a header
        tarhfwdtName = 100,//name field width
        tarhaddrSize = 124,//size field address
        tarhfwdtSize = 12,//size field width
        tarhaddrChksum = 148,//checksum field address
        tarhfwdtChksum = 8,//checksum field width
        tarhaddrTypeflag = 156,//typeflag field address
        tarhfwdtTypeflag = 1,//typeflag field width
        tarhfvalTypeflagReg = '0'//typeflag field value for regular file
    };

    //octal to decimal number
    std::function<size_t(const char*, int)> lfOct2Dec =
        [](const char* hp, int w) {
            size_t res = 0;
            for(; !('0' <= *hp && *hp <= '7') && 0 < w; hp++, w--);
            for(; ('0' <= *hp && *hp <= '7') && 0 < w; hp++, w--)
                res = 8 * res + *hp - '0';
            return res;
        };

    std::ifstream tarfile(entryname.c_str(), std::ios::binary);

    if(!tarfile) {
        warning(("Failed to open tar archive; Ignored: " + entryname).c_str());
        return;
    }

    if(tarfile.rdbuf())
        tarfile.rdbuf()->pubsetbuf(tmpbuff_, CUDBREADER_TMPBUFFSIZE);

    char header[tarHeaderSize];
    //expected sum of the checksum field bytes
    const int expsumchksumfld = tarhfwdtChksum * 32;
    size_t positiontar = 0;
    int n;

    for(; !tarfile.eof();)
    {
        tarfile.read(header, tarHeaderSize);
        std::streamsize szread = tarfile.gcount();

        if(szread < tarHeaderSize) {
            warning(("Ill-formed tar archive: " + entryname).c_str());
            return;
        }

        int chksum = (int)lfOct2Dec(header + tarhaddrChksum, tarhfwdtChksum);
        int sumchksumfld = 0;//sum of the checksum field bytes
        int sum = 0;//sum of the full header bytes

        #pragma omp simd reduction(+:sumchksumfld)
        for(n = 0; n < tarhfwdtChksum; n++)
            sumchksumfld += reinterpret_cast<unsigned char*>(header)[tarhaddrChksum+n];

        #pragma omp simd reduction(+:sum)
        for(n = 0; n < tarHeaderSize; n++)
            sum += reinterpret_cast<unsigned char*>(header)[n];

        if(sum == 0)
            //two records consisting entirely of zero bytes designate the archive end;
            break;

        if((sum - sumchksumfld + expsumchksumfld) != chksum) {
            warning(("Corrupted tar archive; Ignored: " + entryname).c_str());
            return;
        }

        int filetypetar = (int)header[tarhaddrTypeflag];
        size_t filesize = lfOct2Dec(header + tarhaddrSize, tarhfwdtSize);
        //file size in tar is a multiple of tarRecordSize
        size_t filesizetar = ((filesize + tarRecordSize - 1) / tarRecordSize) * tarRecordSize;
        size_t position = positiontar + tarHeaderSize;
        positiontar += tarHeaderSize + filesizetar;

        //set the position indicator in tar to the next file before verifying data
        tarfile.seekg(filesizetar, std::ios_base::cur);
        if(tarfile.fail()) {
            warning(("Failed to move to the next file in tar archive: " + entryname).c_str());
            return;
        }

        if(filetypetar != tarhfvalTypeflagReg && filetypetar != 0/*compatibility*/)
            continue;

        if(filesize < 10)
            continue;

        //NOTE: overwrite the 1st byte of the next field, mode (unused here),
        //NOTE: in the header:
        header[tarhfwdtName] = 0;
        std::string filename(header);

        if(sfxlst_.size() && !SuffixFound(filename))
            continue;

        if(MAXCOMPLETESTRUCTSIZE < filesize) {
            warning(("Too large file in tar archive ignored: " +
                entryname + ":" + filename).c_str());
            continue;
        }

        int filetype = GetFileTypeFromFilename(filename);
        int parenttype = FDRFlTar;

        AddEntry(filename, entryname, position, filesize, parenttype, filetype);
    }//for(;!eof;)
}

// -------------------------------------------------------------------------
// ConstructFileList: construct a file list to be processed; sort the 
// files by size
//
void InputFilelist::ConstructFileList()
{
    static const int nofilesort = CLOptions::GetNOFILESORT();
    //
    strfilelist_.clear();
    pntfilelist_.clear();
    strfilepositionlist_.clear();
    strfilesizelist_.clear();
    strfiletypelist_.clear();
    filendxlist_.clear();
    //
    for(const std::string& entry: dnamelst_)
        ProcessEntry<MAXDIRDEPTH_>(entry);
    //
    if(nofilesort == 0)
        std::sort(filendxlist_.begin(), filendxlist_.end(),
            [this](size_t n1, size_t n2) {
                //place files from a tar archive at the beginning (for favorable access pattern);
                //sort by file size in descending order else
                return 
                    // (strparenttypelist_[n1] == FDRKETar) ||
                    (strfilesizelist_[n1] > strfilesizelist_[n2]);
            });
}

// -------------------------------------------------------------------------
// ProcessEntry: recursive processing of the entry;
// entryname, full pathname;
// rlevel (LEVEL for template version), recursion level 
//  corresponding to the depth of subdirectories;
// maxlevel (MAXLEVEL for template version), maximum recursion level
//
template<int LEVEL>
void InputFilelist::ProcessEntry(const std::string& entryname)
{
    if(file_exists(entryname.c_str())) {
        AddFile(entryname, LEVEL < MAXDIRDEPTH_);
        return;
    }

//     //for runtime version of recursion
//     if(rlevel <= 0)
//         return;

    if(!directory_exists(entryname.c_str())) {
        warning(("Unable to determine the type of entry; Ignored: "+entryname).c_str());
        return;
    }

    //process this directory
    dirent* entry = NULL;
    std::unique_ptr<DIR,void(*)(DIR*)> direct(
        opendir(entryname.c_str()),
        [](DIR* dp) {if(dp) closedir(dp);}
    );
    std::string dirname = entryname + DIRSEP;
    std::string fullname;

    if(!(direct)) {
        warning(("Unable to open directory; Ignored: "+entryname).c_str());
        return;
    }

    while((entry = readdir(direct.get())))
    {
        if(entry->d_name[0] == '.')
            //ignore all directory entries starting with '.'
            continue;

        fullname = dirname + entry->d_name;

        ProcessEntry<LEVEL-1>(fullname);
    }
    //closedir(direct);//managed by unique_ptr
}

// -------------------------------------------------------------------------
template<>
void InputFilelist::ProcessEntry<0>(
    const std::string& entryname)
{
    if(file_exists(entryname.c_str())) {
        AddFile(entryname, 0 < MAXDIRDEPTH_);
        return;
    }
}
