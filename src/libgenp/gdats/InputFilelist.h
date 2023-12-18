/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __InputFilelist_h__
#define __InputFilelist_h__

#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>
#include <algorithm>
#include <functional>

#include <fstream>

#include "libutil/mybase.h"
#include "libutil/CLOptions.h"

// -------------------------------------------------------------------------
// _________________________________________________________________________
// Class InputFilelist
//
// Controller for conducting reading of data
//
class InputFilelist
{
public:
    enum {
        DEFNUMCHAINS = 16,//default number of chains per structure file
        CUDBREADER_TMPBUFFSIZE = 4096,
        MAXCOMPLETESTRUCTSIZE = 128 * ONEM
    };
    //extensions
    enum TKnownExts{
            FDRKETar,//.tar file
            FDRKEGz,//.gz file
            FDRKEPDBGz,//.pdb.gz file
            FDRKEENTGz,//.ent.gz file
            FDRKECIFGz,//.cif.gz file
            FDRKEPDB,//.pdb file
            FDRKEENT,//.ent file
            FDRKECIF,//.cif file
            FDRKEN
    };
    //data file type...
    enum TDataFile{
            FDRFlTar,//file is a tar archive
            FDRFlZip,//file is a compressed structure file
            FDRFlPDBZip,//file is a compressed structure PDB file
            FDRFlPDBxmmCIFZip,//file is a compressed structure PDBx/mmCIF file
            FDRFlStructure,//file is a structure file either in PDB or PDBx/mmCIF format
            FDRFlStructurePDB,//file is a structure file in PDB format
            FDRFlStructurePDBxmmCIF,//file is a structure file in PDBx/mmCIF format
            FDRFlN
    };

    InputFilelist(const std::vector<std::string>& dnamelst,
                const std::vector<std::string>& sfxlst,
                const bool clustering = false);
    ~InputFilelist();

    void ConstructFileList();
    int GetFileTypeFromFilename(const std::string&);

    const std::vector<std::string>& GetStrFilelist() const {return strfilelist_;}
    const std::vector<std::string>& GetPntFilelist() const {return pntfilelist_;}
    const std::vector<size_t>& GetStrFilePositionlist() const {return strfilepositionlist_;}
    const std::vector<size_t>& GetStrFilesizelist() const {return strfilesizelist_;}
    const std::vector<int>& GetStrParenttypelist() const {return strparenttypelist_;}
    const std::vector<int>& GetStrFiletypelist() const {return strfiletypelist_;}
    const std::vector<size_t>& GetFilendxlist() const {return filendxlist_;}

    const std::vector<std::vector<int>>& GetGlobalIds() const {return globalids_;}
    std::vector<std::vector<int>>& GetGlobalIds() {return globalids_;}

protected:
    template<int LEVEL> void ProcessEntry(const std::string&);
    void AddFile(const std::string&, bool);
    void AddFilesFromTAR(const std::string&);

    bool SuffixFound(const std::string& entryname) const
    {
        //verify whether the file suffix is among the specified ones
        auto sit = std::find_if(sfxlst_.begin(), sfxlst_.end(),
            [&entryname](const std::string& sfx){
                return//file has this extension/suffix if true:
                    sfx.size() <= entryname.size() && 
                    entryname.compare(entryname.size()-sfx.size(), 
                        sfx.size(), sfx) == 0;
            });
        if(sit == sfxlst_.end())
            //file does not have a valid extension
            return false;
        return true;
    }

    void AddEntry(
        const std::string& entryname, const std::string& parentname,
        size_t position, size_t filesize, int parenttype, int filetype)
    {
        strfilelist_.push_back(entryname);
        pntfilelist_.push_back(parentname);
        strfilepositionlist_.push_back(position);
        strfilesizelist_.push_back(filesize);
        strparenttypelist_.push_back(parenttype);
        strfiletypelist_.push_back(filetype);
        filendxlist_.push_back(filendxlist_.size());
        if(clustering_) {
            globalids_.push_back(std::vector<int>());
            if(!globalids_.empty()) globalids_.back().reserve(DEFNUMCHAINS);
        }
    }

private:
    const std::vector<std::string>& dnamelst_;//input
    const std::vector<std::string>& sfxlst_;//input
    const bool clustering_;//file list for clustering

    std::vector<std::string> strfilelist_;//structure file list
    std::vector<std::string> pntfilelist_;//parent file list
    std::vector<size_t> strfilepositionlist_;//list of structure file positions within an archive
    std::vector<size_t> strfilesizelist_;//list of structure file sizes
    std::vector<int> strparenttypelist_;//parent file type list of structure files (e.g, tar)
    std::vector<int> strfiletypelist_;//list of structure file types
    std::vector<size_t> filendxlist_;//list of the indices of files sorted by filesize

    std::vector<std::vector<int>> globalids_;//global ids for structures across all files

    char tmpbuff_[CUDBREADER_TMPBUFFSIZE];//buffer for temporary data

public:
    static const std::string knownexts_[FDRKEN];//known extension
};


// /////////////////////////////////////////////////////////////////////////
// INLINES
//
// -------------------------------------------------------------------------
// GetFileTypeFromFilename: get file type from the filename
//
inline
int InputFilelist::GetFileTypeFromFilename(const std::string& filename)
{
    //verify file extension for match
    std::function<bool(int)> lfExtMatched = [&filename](int extcode) {
        if(knownexts_[extcode].size() < filename.size() && 
           filename.compare(filename.size()-knownexts_[extcode].size(),
                knownexts_[extcode].size(), knownexts_[extcode]) == 0)
            return true;
        return false;
    };

    if(lfExtMatched(FDRKEGz)) {
        if(CLOptions::GetI_INFMT() == CLOptions::iifPDB) return FDRFlPDBZip;
        if(CLOptions::GetI_INFMT() == CLOptions::iifmmCIF) return FDRFlPDBxmmCIFZip;
        if(lfExtMatched(FDRKEPDBGz) || lfExtMatched(FDRKEENTGz)) return FDRFlPDBZip;
        if(lfExtMatched(FDRKECIFGz)) return FDRFlPDBxmmCIFZip;
        return FDRFlZip;
    }
    if(lfExtMatched(FDRKETar)) return FDRFlTar;
    if(CLOptions::GetI_INFMT() == CLOptions::iifPDB) return FDRFlStructurePDB;
    if(CLOptions::GetI_INFMT() == CLOptions::iifmmCIF) return FDRFlStructurePDBxmmCIF;
    if(lfExtMatched(FDRKEPDB)) return FDRFlStructurePDB;
    if(lfExtMatched(FDRKEENT)) return FDRFlStructurePDB;
    if(lfExtMatched(FDRKECIF)) return FDRFlStructurePDBxmmCIF;
    //if no extension is recognized, assume it is a structure file whose 
    //type is to be determined
    return FDRFlStructure;
}

// -------------------------------------------------------------------------
// AddFile: add file to a list of structure files
//
inline
void InputFilelist::AddFile(const std::string& entryname, bool sfxcheck)
{
    if(sfxcheck && sfxlst_.size())
        if(!SuffixFound(entryname))
            return;

    size_t filesize = 0;

    if(file_size(entryname.c_str(), &filesize) != 0) {
        warning(("Unable to get file information; Ignored: " + entryname).c_str());
        return;
    }

    if(filesize < 10)
        return;

    int filetype = GetFileTypeFromFilename(entryname);
    int parenttype = filetype;
    size_t position = 0;

    if(filetype == FDRFlTar) {
        AddFilesFromTAR(entryname);
        return;
    }

    AddEntry(entryname, entryname/*parentname*/,
        position, filesize, parenttype, filetype);
}

#endif//__InputFilelist_h__
