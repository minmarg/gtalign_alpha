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
#include "libgenp/gproc/gproc.h"
#include "TdAlnWriter.h"

// -------------------------------------------------------------------------
// WriteResultsPlain: write merged results to file
//
void TdAlnWriter::WriteResultsPlain()
{
    MYMSG("TdAlnWriter::WriteResultsPlain", 4);
    static const std::string preamb = "TdAlnWriter::WriteResultsPlain: ";
    static const bool nodeletions = CLOptions::GetO_NO_DELETIONS();
    static const unsigned int indent = OUTPUTINDENT;
    // static const unsigned int annotlen = ANNOTATIONLEN;
    const unsigned int dscwidth = MAX_DESCRIPTION_LENGTH;//no wrap (pathname)
    const unsigned int dsclen = (dscwidth>>1);//DEFAULT_DESCRIPTION_LENGTH;
    const unsigned int txtwidth = ANNOTATIONLEN;
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
    size += WritePrognamePlain(pb, left/*maxsize*/, dscwidth);

    left = szWriterBuffer - size;
    left = PCMIN(left, (int)dsclen);

    WriteCommandLinePlain(fp.get(),
        buffer_, szWriterBuffer, pb, size);

    written = 
        WriteQueryDescriptionPlain(
            ptmp, szWriterBuffer/*maxsize*/,
            vec_nqyposs_[qrysernr_], vec_qrydesc_[qrysernr_], txtwidth);

    BufferData(fp.get(),
        buffer_, szWriterBuffer, pb, size,
        srchinfo, written);

    bool hitsfound = p_finalindxs_ != NULL && p_finalindxs_->size()>0;

    WriteSearchInformationPlain(fp.get(),
        buffer_, szWriterBuffer, pb, size,
        ptmp, szWriterBuffer/*maxsize*/,
        mstr_set_rfilelist_,
        vec_nposschd_[qrysernr_], vec_nentries_[qrysernr_],
        vec_tmsthld_[qrysernr_], indent, hitsfound);

    if(hitsfound)
    {
        //first, buffer annotations
        for(size_t i = 0; i < p_finalindxs_->size() && i < nhits; i++) {
            char* annot = 
                (*vec_annotptrs_[qrysernr_][allsrtvecs_[ (*p_finalindxs_)[i] ]])
                                            [allsrtindxs_[ (*p_finalindxs_)[i] ]];
            if(annot) {
                //fill in the serial number here:
                int wrtloc = sprintf(annot,"%6d",(int)(i+1));
                annot[wrtloc] = ' ';
            }
            int annotlen = (int)strlen(annot);
            BufferData(fp.get(),
                buffer_, szWriterBuffer, pb, size,
                annot, annotlen);
        }

        if(nodeletions && nalns)
            written =
                sprintf(srchinfo,"%s%s%sNOTE: "
                "ALIGNMENTS with DELETION POSITIONS (gaps in query) REMOVED:%s%s",
                NL,NL,NL,NL,NL);
        else
            written = sprintf(srchinfo,"%s%s%s",NL,NL,NL);
        BufferData(fp.get(),
            buffer_, szWriterBuffer, pb, size,
            srchinfo, written);

        //buffer alignments next
        for( size_t i = 0; i < p_finalindxs_->size() && i < nalns; i++ ) {
            char* aln = 
                (*vec_alnptrs_[qrysernr_][allsrtvecs_[ (*p_finalindxs_)[i] ]])
                                        [allsrtindxs_[ (*p_finalindxs_)[i] ]];
            if(aln) {
                //fill in the serial number here:
                int wrtloc = sprintf(aln,"%-d.",(int)(i+1));
                aln[wrtloc] = ' ';
            }
            int alnlen = (int)strlen(aln);
            BufferData(fp.get(),
                buffer_, szWriterBuffer, pb, size,
                aln, alnlen);
        }
    }

    ptmp = srchinfo;

    written = 
        WriteSummaryPlain(ptmp,
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
// WritePrognamePlain: write the program name;
// outptr, address of the buffer to write;
// maxsize, maximum allowed number of bytes to write;
// width, text width used for wrapping;
// return the number of bytes written;
//
int TdAlnWriter::WritePrognamePlain(char*& outptr, int maxsize, const int width)
{
    static const int sznl = (int)strlen(NL) * 3 + 1;
    int written;
    int size = 0;

    if(PROGNAME) {
        written = sprintf(outptr, "%s", PROGNAME);
        outptr += written;
        size += written;
    }
    if(PROGVERSION) {
        written = sprintf(outptr, " %s", PROGVERSION);
        outptr += written;
        size += written;
    }
    if(PROGNAME || PROGVERSION) {
        written = sprintf(outptr,"%28c",' ');
        outptr += written;
        size += written;
    }
    if(size < maxsize) {
        written = getdtime(outptr, PCMIN(40, maxsize - size));
        outptr += written;
        size += written;
    }

    written = sprintf(outptr, "%s%s", NL,NL);
    outptr += written;
    size += written;

    maxsize -= size + sznl;

    if(maxsize < 1)
        return size;

    char* p = outptr;
    int outpos = 0;
    int linepos = 0;
    char addsep = 0;

    for(int i = 0; PROGREFERENCES[i] && outpos < maxsize; i++) {
        FormatDescription(
            outptr, PROGREFERENCES[i],
            maxsize, 0/*indent*/, width, outpos, linepos, addsep);
        outpos += PutNL(outptr);
        linepos = 0;
    }

    written = sprintf(outptr, "%s", NL);
    outptr += written;

    written = (int)(outptr - p);
    size += written;

    return size;
}

// -------------------------------------------------------------------------
// WriteCommandLinePlain: write the command line in plain format;
// fp, file pointer;
// buffer, buffer to store data in;
// szbuffer, size of the buffer;
// outptr, varying address of the pointer pointing to a location in the buffer;
// offset, outptr offset from the beginning of the buffer (fill size);
//
void TdAlnWriter::WriteCommandLinePlain(
    FILE* fp,
    char* const buffer, const int szbuffer, char*& outptr, int& offset)
{
    if(!__PARGC__ || !__PARGV__ || !*__PARGV__) return;

    static const char* cmdline = NL " Command line:" NL;
    static const int lencmdline = (int)strlen(cmdline);
    static const int sznl = (int)strlen(NL);

    BufferData(fp,
        buffer, szbuffer, outptr, offset,
        cmdline, lencmdline);

    for(int n = 0; n < *__PARGC__; n++) {
        if((*__PARGV__)[n] == NULL) continue;
        BufferData(fp,
            buffer, szbuffer, outptr, offset,
            (*__PARGV__)[n], strlen((*__PARGV__)[n]));
        if(n+1 < *__PARGC__)
            BufferData(fp,
                buffer, szbuffer, outptr, offset,
                " ", 1);
    }

    BufferData(fp,
        buffer, szbuffer, outptr, offset,
        NL, sznl);
}

// -------------------------------------------------------------------------
// WriteQueryDescriptionPlain: write query description;
// outptr, address of the buffer to write;
// maxsize, maximum allowed number of bytes to write;
// printname, include the name in the output text;
// qrylen, query length;
// desc, query description;
// width, width to wrap the query description;
// return the number of bytes written;
//
int TdAlnWriter::WriteQueryDescriptionPlain( 
    char*& outptr,
    int maxsize,
    const int qrylen,
    const std::string& desc,
    const int width )
{
    int written;
    int size = 0;

    if(maxsize < 40)
        return size;

    written = sprintf(outptr, "%s Query (%d residues):%s", NL, qrylen, NL);
    outptr += written;
    size += written;

    maxsize -= size;

    if(maxsize < 1)
        return size;

    char* p = outptr;
    int outpos = 0;
    int linepos = 0;
    char addsep = 0;

    FormatDescription( 
        outptr, desc.c_str(),
        maxsize, 0/*indent*/, width, outpos, linepos, addsep);

    written = (int)(outptr - p);
    size += written;

    return size;
}

// -------------------------------------------------------------------------
// WriteSearchInformationPlain: write search information;
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
// indent, indentation length;
// annotlen, annotation length;
// found, whether any structures have been found;
// return the number of bytes written;
//
void TdAlnWriter::WriteSearchInformationPlain( 
    FILE* fp,
    char* const buffer, const int szbuffer, char*& outptr, int& offset,
    char* tmpbuf, int /* sztmpbuf */,
    const std::vector<std::string>& rfilelist,
    const size_t npossearched,
    const size_t nentries,
    const float tmsthld,
    const int /* indent */,
    const bool found,
    const bool clustering)
{
    static const float seqsimthrscore = CLOptions::GetP_PRE_SIMILARITY();
    static const float prescore = CLOptions::GetP_PRE_SCORE();
    static const bool prescron = seqsimthrscore || prescore;
    static const size_t maxszname = MAX_FILENAME_LENGTH_TOSHOW;
    static const int sznl = (int)strlen(NL);
    int written;
    // const char* dots = "..." NL NL;
    // const int szdots = (int)strlen(dots);

    written = sprintf(tmpbuf,"%s%s %s:%s",NL,NL,(clustering? "Clustered": "Searched"),NL);
    BufferData(fp,
        buffer, szbuffer, outptr, offset,
        tmpbuf, written);

    //print filenames
    for(const std::string& fname: rfilelist) {
        size_t szname = fname.size();
        if(maxszname < szname)
            szname = maxszname;
        strncpy(tmpbuf, fname.substr(fname.size()-szname).c_str(), szname);
        //termination at the beginning
        if(szname < fname.size() && 3 < szname)
            for(int i=0; i<3; i++) tmpbuf[i] = '.';
        BufferData(fp,
            buffer, szbuffer, outptr, offset,
            tmpbuf, szname);
        BufferData(fp,
            buffer, szbuffer, outptr, offset,
            NL, sznl);
    }

    written = sprintf(tmpbuf,"%13c%zu structure(s)%s",' ',nentries,NL);
    BufferData(fp,
        buffer, szbuffer, outptr, offset,
        tmpbuf, written);

    written = sprintf(tmpbuf,"%13c%zu total residues%s%s",' ',npossearched,NL,NL);
    BufferData(fp,
        buffer, szbuffer, outptr, offset,
        tmpbuf, written);

    if(clustering) return;

    if(!found) {
        written = sprintf(tmpbuf,
            " No structures found above a TM-score threshold of %g.%s%s%s%s",
            tmsthld,prescron? " (Pre-screening on!)":"",NL,NL,NL);
        BufferData(fp,
            buffer, szbuffer, outptr, offset,
            tmpbuf, written);
        return;
    }

    written = sprintf(tmpbuf," Legend:%s"
        "TM-score (Refn./Query), Reference/Query length-normalized TM-score%s"
        "2TM-score, secondary TM-score excluding unmatched helices%s"
        "d0 (Refn./Query), Normalizing inter-residue distance d0 for Reference/Query%s"
        "RMSD, Root-mean-square deviation (A); Chn, Chain; (M), Model%s"
        "+, pairs of aligned residues within a distance of %.0f A%s"
        "Secondary structure: h, Helix; e, Strand; t, Turn%s%s%s",
        NL,NL,NL,NL,NL,EQUIVALENCE_DISTANCE,NL,NL,NL,NL);
    BufferData(fp,
        buffer, szbuffer, outptr, offset,
        tmpbuf, written);

    written = sprintf(tmpbuf,"%61s|%5s|%30s|%s",
        "Query_length-normalized_TM-score","RMSD","Reference_alignment_boundaries",NL);
    BufferData(fp,
        buffer, szbuffer, outptr, offset,
        tmpbuf, written);
    written = sprintf(tmpbuf,"%54s|%30s|%s",
        "Reference_length-normalized_TM-score","Query_alignment_boundaries",NL);
    BufferData(fp,
        buffer, szbuffer, outptr, offset,
        tmpbuf, written);
    written = sprintf(tmpbuf,"%6s|%40s|%25s|%29s|%s%s",
        "No.","Reference_description","#Aligned_residues","Reference_length",NL,NL);
    BufferData(fp,
        buffer, szbuffer, outptr, offset,
        tmpbuf, written);
}

// -------------------------------------------------------------------------
// WriteSummaryPlain: write summary statistics;
// outptr, address of the output buffer;
// qrylen, query length;
// npossearched, total positions searched;
// nentries, total number of structures searched;
// duration, accumulated duration for a query batch;
// devname, device name;
// return the number of bytes written;
//
int TdAlnWriter::WriteSummaryPlain( 
    char*& outptr,
    const int qrylen,
    const size_t npossearched,
    const size_t /* nentries */,
    const int nqystrs,
    const double duration,
    const std::string& devname)
{
    int written;
    int size = 0;
    std::chrono::high_resolution_clock::time_point tnow = 
    std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elpsd = tnow - gtSTART;

    written = sprintf(outptr,"Query length: %d%s", qrylen, NL);
    outptr += written;
    size += written;
    written = sprintf(outptr,"Total length of reference structures: %zu%s", npossearched, NL);
    outptr += written;
    size += written;
    written = sprintf(outptr,"Search space: %zu%s", qrylen * npossearched, NL);
    outptr += written;
    size += written;
    written = sprintf(outptr,"Time elapsed from process initiation: %.6f sec%s", elpsd.count(), NL);
    outptr += written;
    size += written;
    written = sprintf(outptr,"Query batch execution time: %.6f sec%s", duration, NL);
    outptr += written;
    size += written;
    written = sprintf(outptr,"Query batch size: %d%s", nqystrs, NL);
    outptr += written;
    size += written;
    written = sprintf(outptr,"Device: \"%s\"%s%s", devname.c_str(), NL,NL);
    outptr += written;
    size += written;

    return size;
}
