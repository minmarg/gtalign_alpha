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

#include "libutil/templates.h"
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
    // static const bool nodeletions = CLOptions::GetO_NO_DELETIONS();
    // static const unsigned int annotlen = ANNOTATIONLEN;
    const unsigned int dscwidth = MAX_DESCRIPTION_LENGTH;
    const unsigned int dsclen = DEFAULT_DESCRIPTION_LENGTH;
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
    left = mymin(left, (int)dsclen);

    WriteCommandLineJSON(fp.get(),
        buffer_, szWriterBuffer, pb, size);

    written = 
        WriteQueryDescriptionJSON(
            ptmp, szWriterBuffer/*maxsize*/,
            vec_nqyposs_[qrysernr_], vec_qrydesc_[qrysernr_], dscwidth);

    BufferData(fp.get(),
        buffer_, szWriterBuffer, pb, size,
        srchinfo, written);

    bool hitsfound = p_finalindxs_ != NULL && p_finalindxs_->size()>0;

    WriteSearchInformationJSON(fp.get(),
        buffer_, szWriterBuffer, pb, size,
        ptmp, szWriterBuffer/*maxsize*/,
        mstr_set_rfilelist_,
        vec_nposschd_[qrysernr_], vec_nentries_[qrysernr_],
        vec_tmsthld_[qrysernr_], hitsfound);

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
            bool last = !(i+1 < p_finalindxs_->size() && i+1 < nalns);
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
            vec_nqyposs_[qrysernr_], vec_nposschd_[qrysernr_],
            vec_nentries_[qrysernr_], vec_nqystrs_[qrysernr_],
            vec_duration_[qrysernr_], vec_devname_[qrysernr_]);

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
            "{\"gtalign_search\": {%s"
            "  \"program\": \"%s\",%s"
            "  \"version\": \"%s\",%s"
            "  \"search_date\": \"",
            NL,PROGNAME? PROGNAME: "",
            NL,PROGVERSION? PROGVERSION: "",NL);
    outptr += written;
    size += written;
    written = getdtime(outptr, mymin(40, maxsize - size));
    outptr += written;
    size += written;
    written = sprintf(outptr,"\",%s",NL);
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
// WriteCommandLineJSON: write the command line in JSON format;
// fp, file pointer;
// buffer, buffer to store data in;
// szbuffer, size of the buffer;
// outptr, varying address of the pointer pointing to a location in the buffer;
// offset, outptr offset from the beginning of the buffer (fill size);
//
void TdAlnWriter::WriteCommandLineJSON(
    FILE* fp,
    char* const buffer, const int szbuffer, char*& outptr, int& offset)
{
    if(!__PARGC__ || !__PARGV__ || !*__PARGV__) return;

    static const char* cmdline = "  \"command_line\": \"";
    static const int lencmdline = (int)strlen(cmdline);
    static const int sznl = (int)strlen(NL);
    std::string::size_type n;
    std::string tmpstr;
    tmpstr.reserve(KBYTE);

    BufferData(fp,
        buffer, szbuffer, outptr, offset,
        cmdline, lencmdline);

    for(int i = 0; i < *__PARGC__; i++) {
        if((*__PARGV__)[i] == NULL) continue;
        tmpstr = (*__PARGV__)[i];
        for(n = 0; (n = tmpstr.find('\\', n)) != std::string::npos; n += 2) tmpstr.insert(n, 1, '\\');
        for(n = 0; (n = tmpstr.find('/', n)) != std::string::npos; n += 2) tmpstr.insert(n, 1, '\\');
        for(n = 0; (n = tmpstr.find('"', n)) != std::string::npos; n += 2) tmpstr.insert(n, 1, '\\');
        BufferData(fp,
            buffer, szbuffer, outptr, offset,
            tmpstr.c_str(), tmpstr.size());
        if(i+1 < *__PARGC__)
            BufferData(fp,
                buffer, szbuffer, outptr, offset,
                " ", 1);
    }

    BufferData(fp,
        buffer, szbuffer, outptr, offset,
        "\"," NL, 2 + sznl);
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
    int size = 0, outpos = 0;

    if(maxsize < 128)
        return size;

    written = sprintf(outptr,
            "  \"query\": {%s"
            "    \"length\": %d,%s"
            "    \"description\": \"",
            NL,qrylen,NL);
    outptr += written;
    size += written;

    maxsize -= size + 16/*ending*/;

    FormatDescriptionJSON(outptr, desc.c_str(), desc.size(), maxsize, outpos);
    size += outpos;
    written = sprintf(outptr,"\"%s  },%s",NL,NL);
    outptr += written;
    size += written;

    return size;
}

// -------------------------------------------------------------------------
// WriteSearchInformationJSON: write search information in JSON format;
// fp, file pointer;
// buffer, buffer to store data in;
// szbuffer, size of the buffer;
// outptr, varying address of the pointer pointing to a location in the buffer;
// offset, outptr offset from the beginning of the buffer (fill size);
// tmpbuf, address of a temporary buffer for writing;
// sztmpbuf, size of tmpbuf;
// rfilelist, list of filenames/dirnames searched;
// npossearched, total number of positions searched;
// nentries, total number of structures searched;
// tmsthld, TM-score threshold;
// found, whether any structures have been found;
//
void TdAlnWriter::WriteSearchInformationJSON( 
    FILE* fp,
    char* const buffer, const int szbuffer, char*& outptr, int& offset,
    char* tmpbuf, int /* sztmpbuf */,
    const std::vector<std::string>& rfilelist,
    const size_t npossearched,
    const size_t nentries,
    const float tmsthld,
    const bool found)
{
    static const int nalns = CLOptions::GetO_NALNS();
    static const bool nodeletions = CLOptions::GetO_NO_DELETIONS();
    static const float seqsimthrscore = CLOptions::GetP_PRE_SIMILARITY();
    static const float prescore = CLOptions::GetP_PRE_SCORE();
    static const bool prescron = seqsimthrscore || prescore;
    static const size_t maxszname = MAX_FILENAME_LENGTH_TOSHOW;
    int written;

    written = sprintf(tmpbuf,
            "  \"database\": {%s"
            "    \"entries\": [%s",NL,NL);
    BufferData(fp,
        buffer, szbuffer, outptr, offset,
        tmpbuf, written);

    //print filenames
    //for(const std::string& fname: rfilelist) {
    for(size_t i = 0; i < rfilelist.size(); i++) {
        int outpos = 0;
        int szname = (int)rfilelist[i].size();
        written = sprintf(tmpbuf,"      \"");
        BufferData(fp,
            buffer, szbuffer, outptr, offset,
            tmpbuf, written);
        char* p = tmpbuf;
        FormatDescriptionJSON(p, rfilelist[i].c_str(), szname, maxszname, outpos);
        BufferData(fp,
            buffer, szbuffer, outptr, offset,
            tmpbuf, outpos);
        written = (i+1 < rfilelist.size())? 
            sprintf(tmpbuf,"\",%s",NL): sprintf(tmpbuf,"\"%s",NL);
        BufferData(fp,
            buffer, szbuffer, outptr, offset,
            tmpbuf, written);
    }

    written = sprintf(tmpbuf,
            "    ],%s"
            "    \"number_of_structures\": %zu,%s"
            "    \"number_of_positions\": %zu%s"
            "  },%s",
            NL,nentries,NL,npossearched,NL,NL);
    BufferData(fp,
        buffer, szbuffer, outptr, offset,
        tmpbuf, written);

    if(found) {
        if(nodeletions && nalns)
            written = sprintf(tmpbuf,
                "  \"message\": \"NOTE: "
                "ALIGNMENTS with DELETION POSITIONS (gaps in query) REMOVED\",%s",NL);
        else
            written = sprintf(tmpbuf,
                "  \"message\": \"\",%s",NL);
    } else
        written = sprintf(tmpbuf,
            "  \"message\": \"No structures found above a TM-score threshold of %g.%s\",%s",
            tmsthld,(prescron? " (Pre-screening on!)":""),NL);
    BufferData(fp,
        buffer, szbuffer, outptr, offset,
        tmpbuf, written);
}

// -------------------------------------------------------------------------
// WriteSummaryJSON: write summary statistics in JSON format;
// outptr, address of the output buffer;
// qrylen, query length;
// npossearched, total positions searched;
// nentries, total number of structures searched;
// duration, accumulated duration for a query batch;
// devname, device name;
// return the number of bytes written;
//
int TdAlnWriter::WriteSummaryJSON( 
    char*& outptr,
    const int qrylen,
    const size_t npossearched,
    const size_t /* nentries */,
    const int nqystrs,
    const double duration,
    std::string devname)
{
    int written;
    int size = 0;
    std::string::size_type n;
    std::chrono::high_resolution_clock::time_point tnow = 
    std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elpsd = tnow - gtSTART;

    for(n = 0; (n = devname.find('\\', n)) != std::string::npos; n += 2) devname.insert(n, 1, '\\');
    for(n = 0; (n = devname.find('/', n)) != std::string::npos; n += 2) devname.insert(n, 1, '\\');
    for(n = 0; (n = devname.find('"', n)) != std::string::npos; n += 2) devname.insert(n, 1, '\\');

    written = sprintf(outptr,
            "  \"search_statistics\": {%s"
            "    \"query_length\": %d,%s"
            "    \"total_database_length\": %zu,%s"
            "    \"search_space\": %zu,%s"
            "    \"time_elapsed_from_process_initiation\": %.6f,%s"
            "    \"query_batch_execution_time\": %.6f,%s"
            "    \"query_batch_size\": %d,%s"
            "    \"device\": \"%s\"%s"
            "  }%s}}%s",
            NL,qrylen,NL,npossearched,NL,qrylen * npossearched,NL,
            elpsd.count(),NL,duration,NL,nqystrs,NL,devname.c_str(),NL,NL,NL);
    outptr += written;
    size += written;

    return size;
}
