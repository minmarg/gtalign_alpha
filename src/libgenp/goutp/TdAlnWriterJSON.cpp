/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/mybase.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>
#include <memory>

#include "libutil/CLOptions.h"
#include "libutil/format.h"
#include "TdAlnWriter.h"

// -------------------------------------------------------------------------
// WriteResultsJSON: write merged results to file in JSON format
//
void TdAlnWriter::WriteResultsJSON()
{
    MYMSG("TdAlnWriter::WriteResultsJSON", 4);
    static const std::string preamb = "TdAlnWriter::WriteResultsJSON: ";
    static const unsigned int indent = OUTPUTINDENT;
    static const unsigned int annotlen = ANNOTATIONLEN;
    const unsigned int dsclen = DEFAULT_DESCRIPTION_LENGTH;
    const unsigned int dscwidth = MAX_DESCRIPTION_LENGTH;
    const size_t nhits = CLOptions::GetO_NHITS();
    const size_t nalns = CLOptions::GetO_NALNS();
    std::string filename;
    std::unique_ptr<FILE,void(*)(FILE*)> fp(
        nullptr,
        [](FILE* p) {if(p) fclose(p);}
    );
    char srchinfo[szWriterBuffer];
    char* pb = buffer_, *ptmp = srchinfo;
    int written, left;
    int size = 0;//#characters written already in the buffer

    if((int)parts_qrs_.size() <= qrysernr_ || qrysernr_ < 0 )
        throw MYRUNTIME_ERROR(preamb + "Invalid query serial number.");

    GetOutputFilename(filename,
        mstr_set_outdirname_, vec_qrydesc_[qrysernr_], qrysernr_);

    //use mode b for OS_MS_WINDOWS to not use translation
    fp.reset(fopen(filename.c_str(), "wb"));

    if(!(fp))
        throw MYRUNTIME_ERROR(
        preamb + "Failed to open file for writing: " + filename);

    left = szWriterBuffer;
    size += WritePrognameJSON(pb, left/*maxsize*/, dscwidth);

    left = szWriterBuffer - size;
    left = PCMIN(left, (int)dsclen);

    size += 
        WriteQueryDescriptionJSON(
            pb, left/*maxsize*/,
            vec_nqyposs_[qrysernr_], vec_qrydesc_[qrysernr_], dscwidth);

    bool hitsfound = p_finalindxs_ != NULL && p_finalindxs_->size()>0;

    written = 
        WriteSearchInformationJSON(
            ptmp, szWriterBuffer/*maxsize*/,
            mstr_set_rfilelist_,
            vec_nposschd_[qrysernr_], vec_nentries_[qrysernr_],
            vec_tmsthld_[qrysernr_], indent, annotlen, hitsfound);

    BufferData(fp.get(),
        buffer_, szWriterBuffer, pb, size,
        srchinfo, written);

    written = sprintf(srchinfo,"  \"search_summary\": [%s",NL);
    BufferData(fp.get(),
        buffer_, szWriterBuffer, pb, size,
        srchinfo, written);

    const char* recsep = "," NL;//record separator
    const char* finsep = NL;//separator following the last record
    const int lenrecsep = (int)strlen(recsep);
    const int lenfinsep = (int)strlen(finsep);

    if(hitsfound)
    {
        //first, buffer annotations
        for(size_t i = 0; i < p_finalindxs_->size() && i < nhits; i++) {
            const char* annot = 
                (*vec_annotptrs_[qrysernr_][allsrtvecs_[ (*p_finalindxs_)[i] ]])
                                            [allsrtindxs_[ (*p_finalindxs_)[i] ]];
            int annotlen = (int)strlen(annot);
            bool last = !(i+1 < p_finalindxs_->size() && i+1 < nhits);
            BufferData(fp.get(),
                buffer_, szWriterBuffer, pb, size,
                annot, annotlen);
            BufferData(fp.get(),
                buffer_, szWriterBuffer, pb, size,
                last? finsep: recsep, last? lenfinsep: lenrecsep);
        }
    }

    written = sprintf(srchinfo,"  ],%s  \"search_results\": [%s",NL,NL);
    BufferData(fp.get(),
        buffer_, szWriterBuffer, pb, size,
        srchinfo, written);

    if(hitsfound)
    {
        //buffer alignments now
        for(size_t i = 0; i < p_finalindxs_->size() && i < nalns; i++) {
            const char* aln = 
                (*vec_alnptrs_[qrysernr_][allsrtvecs_[ (*p_finalindxs_)[i] ]])
                                        [allsrtindxs_[ (*p_finalindxs_)[i] ]];
            int alnlen = (int)strlen(aln);
            bool last = !(i+1 < p_finalindxs_->size() && i+1 < nhits);
            BufferData(fp.get(),
                buffer_, szWriterBuffer, pb, size,
                aln, alnlen);
            BufferData(fp.get(),
                buffer_, szWriterBuffer, pb, size,
                last? finsep: recsep, last? lenfinsep: lenrecsep);
        }
    }

    written = sprintf(srchinfo,"  ],%s",NL);
    BufferData(fp.get(),
        buffer_, szWriterBuffer, pb, size,
        srchinfo, written);

    ptmp = srchinfo;

    written = 
        WriteSummaryJSON(ptmp,
            vec_nqyposs_[qrysernr_],
            vec_nposschd_[qrysernr_], vec_nentries_[qrysernr_]);

    BufferData(fp.get(),
        buffer_, szWriterBuffer, pb, size,
        srchinfo, written);

    //flush the buffer to file
    if(size > 0)
        WriteToFile(fp.get(), buffer_, size);
}

// -------------------------------------------------------------------------
// WritePrognameJSON: write the program name in JSON format;
// outptr, address of the buffer to write;
// maxsize, maximum allowed number of bytes to write;
// return the number of bytes written;
//
int TdAlnWriter::WritePrognameJSON(char*& outptr, int maxsize, const int /*width*/)
{
    //max allowed length for the section of references
    const int maxlenreferences = KBYTE;
    //max space for the `references' text itself:
    const int maxlenrefinf = maxlenreferences - 60;
    int written;
    int size = 0;

    written = sprintf(outptr,
            "{\"structure_search\": {%s"
            "  \"program\": \"%s\",%s"
            "  \"version\": \"%s\",%s",
            NL,PROGNAME? PROGNAME: "",
            NL,PROGVERSION? PROGVERSION: "",NL);
    outptr += written;
    size += written;

    //subtract the maximum reserved space for references
    maxsize -= size + maxlenreferences;

    if(maxsize < 1)
        return size;

    written = sprintf(outptr,"  \"references\": [%s",NL);
    outptr += written;
    size += written;

    int totlenref = 0;
    bool bwrt = true;

    for(int i = 0; !i || (bwrt && PROGREFERENCES[i]); i++)
    {
        const char* strref = PROGREFERENCES[i];
        totlenref += strref? (int)strlen(strref): 0;
        bwrt = totlenref < maxlenrefinf;
        written = sprintf( outptr,
            "    \"%s\"%s%s",
            (bwrt && strref)? strref: "",
            (bwrt && strref && PROGREFERENCES[i+1])? ",":"", NL);
        outptr += written;
        size += written;
        if(!PROGREFERENCES[i])
            break;
    }

    written = sprintf(outptr,"  ],%s",NL);
    outptr += written;
    size += written;

    return size;
}

// -------------------------------------------------------------------------
// WriteQueryDescriptionJSON: write query description in JSON format;
// outptr, address of the buffer to write;
// maxsize, maximum allowed number of bytes to write;
// qrylen, query length;
// desc, query description;
// width, width to wrap the query description;
// return the number of bytes written;
//
int TdAlnWriter::WriteQueryDescriptionJSON( 
    char*& outptr,
    int maxsize,
    const int qrylen,
    const std::string& desc,
    const int /*width*/)
{
    int written;
    int size = 0;

    if(maxsize < 60)
        return size;

    written = sprintf(outptr,
            "  \"query\": {%s"
            "    \"length\": %d,%s",
            NL,qrylen,NL);
    outptr += written;
    size += written;

    //subtract BUF_MAX for accounting for the two fields preceding description
    maxsize -= size + BUF_MAX;

    if( maxsize < 1 )
        return size;

    char* p = outptr;
    int outpos = 0;

    written = sprintf(outptr,"    \"entity\": \"");
    outptr += written;
    outpos += written;
    FormatDescriptionJSON(outptr, desc.c_str(), maxsize, outpos);
    written = sprintf(outptr,"\"%s  },%s",NL,NL);
    outptr += written;
    outpos += written;

    written = (int)(outptr - p);
    size += written;

    return size;
}

// -------------------------------------------------------------------------
// WriteSearchInformationJSON: write search information in JSON format;
// outptr, address of the buffer to write;
// maxsize, maximum allowed number of bytes to write;
// rfilelist, list of filenames/dirnames searched;
// npossearched, total number of positions searched;
// nentries, total number of structures searched;
// tmsthld, TM-score threshold;
// found, whether any profiles have been found;
// return the number of bytes written;
//
int TdAlnWriter::WriteSearchInformationJSON( 
    char*& outptr,
    int maxsize,
    const std::vector<std::string>& rfilelist,
    const size_t npossearched,
    const size_t nentries,
    const float tmsthld,
    const int /*indent*/,
    const int /*annotlen*/,
    const bool found )
{
    static const size_t maxszname = MAX_FILENAME_LENGTH_TOSHOW;
    static const int envlpines = 7;//number of lines wrapping filenames
    static const int headlines = 4;//number of lines for information
    static const int maxfieldlen = 40;//JSON max field length
    static const int maxlinelen = 90;//maximum length of other lines
    int written, size = 0;
    const char* dots = "      \"...\"" NL;
    //const int szdots = (int)strlen(dots);

    std::function<bool(int)> lfn = 
        [&maxsize](int szdelta) {
            maxsize -= szdelta;
            return(maxsize < 1);
        };

    if(lfn(maxfieldlen*envlpines))
        //7, minimum database section lines (including 1 name or dots)
        return size;

    written = sprintf(outptr,
            "  \"database\": {%s"
            "    \"entries\": [%s",NL,NL);
    outptr += written;
    size += written;

    //print filenames
    //for(const std::string& fname: rfilelist) {
    for(size_t i = 0; i < rfilelist.size(); i++) {
        int outpos = 0;
        size_t szname = rfilelist[i].size();
        if(maxszname < szname)
            szname = maxszname;
        if(lfn((int)(szname+15))) {
            written = sprintf(outptr,"%s",dots);
            outptr += written;
            size += written;
            break;
        }
        written = sprintf(outptr,"      \"");
        outptr += written;
        size += written;
        FormatDescriptionJSON(outptr, rfilelist[i].c_str(), szname, outpos);
        size += outpos;
        written = (i+1 < rfilelist.size())? 
            sprintf(outptr,"\",%s",NL): sprintf(outptr,"\"%s",NL);
        outptr += written;
        size += written;
    }

    written = sprintf(outptr,
            "    ],%s"
            "    \"number_of_structures\": %zu,%s"
            "    \"number_of_positions\": %zu%s"
            "  },%s",
            NL,nentries,NL,npossearched,NL,NL);
    outptr += written;
    size += written;

    if(lfn(maxlinelen*headlines))
        return size;

    if(found) {
        written = sprintf(outptr,
            "  \"message\": \"Structures found above the TM-score threshold:\",%s",NL);
        outptr += written;
        size += written;
    }
    else {
        written = sprintf(outptr,
            "  \"message\": \"No structures found above a TM-score threshold of %g\",%s",
            tmsthld,NL);
        outptr += written;
        size += written;
    }

    return size;
}

// -------------------------------------------------------------------------
// WriteSummaryJSON: write summary statistics in JSON format;
// outptr, address of the output buffer;
// qrylen, query length;
// npossearched, total positions searched;
// nentries, total number of structures searched;
// return the number of bytes written;
//
int TdAlnWriter::WriteSummaryJSON( 
    char*& outptr,
    const int qrylen,
    const size_t npossearched,
    const size_t nentries)
{
    int written;
    int size = 0;

    written = sprintf(outptr,
            "  \"search_statistics\": {%s"
            "    \"query_length\": %d,%s"
            "    \"database_size\": %zu,%s"
            "    \"search_space\": %zu%s"
            "  }%s}}%s",
            NL,qrylen,NL,npossearched,NL,
            npossearched * nentries,NL,NL,NL);
    outptr += written;
    size += written;

    return size;
}
