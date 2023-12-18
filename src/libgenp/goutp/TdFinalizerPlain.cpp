/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/mybase.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cmath>
#include <utility>
#include <algorithm>

#include "libutil/CLOptions.h"
#include "libutil/format.h"

// #include "libgenp/gdats/PM2DVectorFields.h"
// #include "libgenp/gdats/PMBatchStrData.h"
// #include "libgenp/goutp/TdAlnWriter.h"
#include "libgenp/gproc/btckcoords.h"
#include "libgenp/gproc/gproc.h"

// #include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "TdFinalizer.h"

// -------------------------------------------------------------------------
// CompressResultsPlain: process results obtained; compress them for passing 
// them to the writing thread; use plain format
// NOTE: all operations performed under lock
//
void TdFinalizer::CompressResultsPlain()
{
    MYMSG("TdFinalizer::CompressResultsPlain", 4);
    static const unsigned int indent = OUTPUTINDENT;
    // static const unsigned int annotlen = ANNOTATIONLEN;
    const unsigned int dsclen = DEFAULT_DESCRIPTION_LENGTH;
    const unsigned int dscwidth = MAX_DESCRIPTION_LENGTH;//no wrap (pathname)
    const unsigned int alnwidth = CLOptions::GetO_WRAP();
    const unsigned int width = indent < dscwidth? dscwidth: 1;
    size_t szannot = 0UL;
    size_t szalns = 0UL;
    size_t szalnswodesc = 0UL;
    const char* desc;//structure description
    int written, sernr = 0;//initialize serial number here

    if(cubp_set_qrysernrbeg_ < 0 || qrysernr_ < cubp_set_qrysernrbeg_ || 
       cubp_set_qrysernrbeg_ + (int)cubp_set_nqystrs_ <= qrysernr_)
        throw MYRUNTIME_ERROR(
        "TdFinalizer::CompressResultsPlain: Invalid query indices.");

    GetSizeOfCompressedResultsPlain(&szannot, &szalns, &szalnswodesc);

    annotations_.reset();
    alignments_.reset();

    ReserveVectors(qrynstrs_);

    if(szalns < szannot || 
       szalnswodesc > 4 * 
            (cubp_set_sz_alndata_ + cubp_set_sz_tfmmatrices_ + cubp_set_sz_alns_))
        throw MYRUNTIME_ERROR(
        "TdFinalizer::CompressResultsPlain: "
        "Size of formatted results is unusually large.");

    if(szannot < 1 || szalns < 1)
        return;

    annotations_.reset((char*)std::malloc(szannot));
    alignments_.reset((char*)std::malloc(szalns));

    if(!annotations_ || !alignments_)
        throw MYRUNTIME_ERROR(
        "TdFinalizer::CompressResultsPlain: Not enough memory.");

    if(!srtindxs_ || !scores_ || !alnptrs_ || !annotptrs_)
        throw MYRUNTIME_ERROR(
        "TdFinalizer::CompressResultsPlain: Not enough memory.");

    char* annptr = annotations_.get();
    char* outptr = alignments_.get();

    const int sortby = CLOptions::GetO_SORT();

    for(int strndx = 0; strndx < qrynstrs_; strndx++)
    {
        float tmscoreq = GetOutputAlnDataField<float>(strndx, dp2oadScoreQ);
        float tmscorer = GetOutputAlnDataField<float>(strndx, dp2oadScoreR);
        float rmsd = GetOutputAlnDataField<float>(strndx, dp2oadRMSD);
        float tmscoregrt = PCMAX(tmscoreq, tmscorer);
        float tmscore = tmscoregrt;
        if(sortby == CLOptions::osTMscoreReference) tmscore = tmscorer;
        if(sortby == CLOptions::osTMscoreQuery) tmscore = tmscoreq;
        if(sortby == CLOptions::osRMSD) tmscore = -rmsd;

        if(tmscoregrt < cubp_set_scorethld_)
            continue;

        unsigned int alnlen = (unsigned int)GetOutputAlnDataField<float>(strndx, dp2oadAlnLength);
        //NOTE: alignment length can be 0 due to pre-screening
        if(alnlen < 1)
            continue;

//         //get the index over all (two) db pm structures
//         unsigned int orgstrndx = GetOutputAlnDataField<unsigned int>(strndx, dp2oadOrgStrNo);
//         //distance form the beginning in phase 2 (structures passed through phase 1)
//         unsigned int dbstr2dst = GetOutputAlnDataField<unsigned int>(strndx, dp2oadStrNewDst);
//         int dbstrlen = GetDbStructureField<INTYPE>(orgstrndx, pps2DLen);
        unsigned int dbstr2dst = GetDbStructureField<unsigned int>(strndx, pps2DDist);
        int dbstrlen = GetDbStructureField<INTYPE>(strndx, pps2DLen);
        if(qrynposits_ < dbstr2dst + (unsigned int)dbstrlen)
            //the structure has not been processed due to GPU memory restrictions
            continue;

        //save the addresses of the annotations and alignment records
        srtindxs_->push_back(sernr++);
        scores_->push_back(tmscore);
        annotptrs_->push_back(annptr);
        alnptrs_->push_back(outptr);

//         //get the name and description
//         GetDbStructureDesc(desc, orgstrndx);
        GetDbStructureDesc(desc, strndx);
        //make an annotation
        MakeAnnotationPlain(annptr,
            strndx, strndx, desc, alnlen, dbstrlen);
        *annptr++ = 0;//end of record

        //reserve space for serial number:
        written = sprintf(outptr,"%13c%s",' ',NL);
        outptr += written;
        //put the description...
        int outpos = 0;
        int linepos = 0;
        char addsep = '>';
        FormatDescription(
            outptr, desc,
            dsclen, indent, width, outpos, linepos, addsep);
        PutNL(outptr);

//         //compress the alignment and relative information...
//         FormatScoresPlain(outptr,
//             strndx, orgstrndx, alnlen, tmscore, qrystrlen_, dbstrlen);
//         FormatAlignmentPlain(outptr,
//             strndx, orgstrndx, dbstr2dst, alnlen, 
//             qrystrlen_, dbstrlen, alnwidth);
        FormatScoresPlain(outptr,
            strndx, strndx, alnlen, tmscore, qrystrlen_, dbstrlen);
        FormatAlignmentPlain(outptr,
            strndx, strndx, dbstr2dst, alnlen, 
            qrystrlen_, dbstrlen, alnwidth);
        FormatFooterPlain(outptr, strndx);

        written = sprintf( outptr,"%s%s",NL,NL);
        outptr += written;
        *outptr++ = 0;//end of record
    }
}

// -------------------------------------------------------------------------
// MakeAnnotationPlain: format structure description;
// NOTE: space is assumed to be pre-allocated;
// outptr, pointer to the output buffer;
// desc, structure description;
// maxoutlen, maximum length of output description;
// width, width to wrap the structure description;
// tmscoreq, tmscorer, TM-scores normalized by the query and reference lengths;
inline
void TdFinalizer::MakeAnnotationPlain( 
    char*& outptr,
    const int strndx,
    const unsigned int /*orgstrndx*/,
    const char* desc,
    const unsigned int alnlen,
    const int dbstrlen) const
{
    int written;
    float rmsd = GetOutputAlnDataField<float>(strndx, dp2oadRMSD);
    float tmscoreq = GetOutputAlnDataField<float>(strndx, dp2oadScoreQ);
    float tmscorer = GetOutputAlnDataField<float>(strndx, dp2oadScoreR);
    // int psts = (int)GetOutputAlnDataField<float>(strndx, dp2oadPstvs);
    // int idts = (int)GetOutputAlnDataField<float>(strndx, dp2oadIdnts);
    int gaps = (int)GetOutputAlnDataField<float>(strndx, dp2oadNGaps);
    //alignment beginning coordinates:
    unsigned int alnbegcoords = GetOutputAlnDataField<unsigned int>(strndx, dp2oadBegCoords);
    int qrybeg = GetCoordY(alnbegcoords);
    int trgbeg = GetCoordX(alnbegcoords);
    //alignment end coordinates:
    unsigned int alnendcoords = GetOutputAlnDataField<unsigned int>(strndx, dp2oadEndCoords);
    int qryend = GetCoordY(alnendcoords);
    int trgend = GetCoordX(alnendcoords);

    //reserve for serial number
    written = sprintf(outptr,"%6c",' ');
    outptr += written;

    int desclen = strlen(desc);
    if(desclen <= ANNOTATION_DESCLEN)
        written = sprintf(outptr," %-" TOSTR(ANNOTATION_DESCLEN) "s",desc);
    else {
        written = sprintf(outptr," ...");
        outptr += written;
        written = sprintf(outptr,"%-" TOSTR(ANNOTATION_DESCLEN_3l) "s",
            desc+desclen-ANNOTATION_DESCLEN_3l);
    }
    outptr += written;

    written = sprintf(outptr," %6.4f %6.4f %5.2f",tmscorer,tmscoreq,rmsd);
    outptr += written;

    written = sprintf(outptr," %5d %5d-%-5d %5d-%-5d %5d%s",
        (int)alnlen-gaps, qrybeg,qryend,trgbeg,trgend, dbstrlen,NL);
    outptr += written;
}

// -------------------------------------------------------------------------
// outptr, pointer to the output buffer;
// strndx, structure index in the results list;
// orgstrndx, structure index over all pm data structures;
// alnlen, alignment length;
// score, score (TM-score/RMSD);
// querylen, query length;
// dbstrlen, db structure length;
inline
void TdFinalizer::FormatScoresPlain(
    char*& outptr,
    int strndx,
    unsigned int /*orgstrndx*/,
    unsigned int alnlen,
    float /* score */,
    int querylen,
    int dbstrlen)
{
    int written;
    float rmsd = GetOutputAlnDataField<float>(strndx, dp2oadRMSD);
    float tmscoreq = GetOutputAlnDataField<float>(strndx, dp2oadScoreQ);
    float tmscorer = GetOutputAlnDataField<float>(strndx, dp2oadScoreR);
    float d0q = GetOutputAlnDataField<float>(strndx, dp2oadD0Q);
    float d0r = GetOutputAlnDataField<float>(strndx, dp2oadD0R);
    int psts = (int)GetOutputAlnDataField<float>(strndx, dp2oadPstvs);
    int idts = (int)GetOutputAlnDataField<float>(strndx, dp2oadIdnts);
    int gaps = (int)GetOutputAlnDataField<float>(strndx, dp2oadNGaps);

    written = 
    sprintf(outptr,"  Length: Refn. = %d, Query = %d%s%s", dbstrlen, querylen, NL,NL);
    outptr += written;
    written = 
    sprintf(outptr," TM-score (Refn./Query) = %.5f / %.5f, "
        "d0 (Refn./Query) = %.2f / %.2f,  RMSD = %.2f A",
        tmscorer, tmscoreq, d0r, d0q, rmsd);
    outptr += written;
    written = sprintf(outptr,"%s",NL);
    outptr += written;
    if(idts > 0) {
        written = 
            sprintf(outptr," Identities = %d/%d (%d%%)",idts,alnlen,idts*100/alnlen);
        outptr += written;
    }
    if(psts > 0) {
        if(idts) {
            written = sprintf( outptr,",");
            outptr += written;
        }
        written = 
            sprintf(outptr," Matched = %d/%d (%d%%)",psts,alnlen,psts*100/alnlen);
        outptr += written;
    }
    if(gaps > 0) {
        if(idts || psts) {
            written = sprintf( outptr,",");
            outptr += written;
        }
        written = sprintf( outptr," Gaps = %d/%d (%d%%)",gaps,alnlen,gaps*100/alnlen);
        outptr += written;
    }
    written = sprintf( outptr,"%s%s",NL,NL);
    outptr += written;
}

// -------------------------------------------------------------------------
// outptr, pointer to the output buffer;
// strndx, structure index in the results list;
// orgstrndx, structure index over all pm data structures;
// dbstr2dst, distance from the beginning in phase 2;
// alnlen, alignment length;
// querylen, query length;
// dbstrlen, db structure length;
// width, alignment output width;
// printsss, print SS assignment information;
inline
void TdFinalizer::FormatAlignmentPlain(
    char*& outptr,
    int strndx,
    unsigned int /*orgstrndx*/,
    unsigned int dbstr2dst,
    int alnlen,
    int querylen,
    int /* dbstrlen */,
    const int width)
{
    static const bool nodeletions = CLOptions::GetO_NO_DELETIONS();
    //alignment beginning coordinates:
    unsigned int alnbegcoords = GetOutputAlnDataField<unsigned int>(strndx, dp2oadBegCoords);
    unsigned int qrybeg = GetCoordY(alnbegcoords);
    unsigned int trgbeg = GetCoordX(alnbegcoords);

    unsigned int alnendcoords = GetOutputAlnDataField<unsigned int>(strndx, dp2oadEndCoords);
    // int qryend = GetCoordY(alnendcoords);
    int trgend = GetCoordX(alnendcoords);

    //alignment beginning position:
    int alnbeg = 
        offsetalns_ + //offset to the alignments produced for the current query (qrysernr_)
        dbstr2dst + strndx * (querylen + 1);//alignment position for strndx within the query section

    //beginning of the alignment:
    const char* palnbeg = GetBegOfAlns() + alnbeg;
    const char* p;

    int written;
    int nbytes;
    int fgaps;

    for(int f = 0; f < alnlen; f += width, palnbeg += width)
    {
        nbytes = alnlen - f;
        if( width < nbytes )
            nbytes = width;
        if(1) {
            p = GetAlnSectionAt(palnbeg, dp2oaQuerySSS);
            written = sprintf(outptr,"%-13s","struct");
            outptr += written;
            strncpy(outptr, p, nbytes);
            outptr += nbytes;
            PutNL(outptr);
        }
        {   p = GetAlnSectionAt(palnbeg, dp2oaQuery);
            written = sprintf(outptr,"Query: %5u ", qrybeg);
            outptr += written;
            strncpy(outptr, p, nbytes);
            outptr += nbytes;
            fgaps = (int)(std::count(p, p+nbytes, '-'));
            qrybeg += nbytes - fgaps;
            written = sprintf(outptr," %-5d%s", nbytes<=fgaps? qrybeg: qrybeg-1,NL);
            outptr += written;
        }
        {   p = GetAlnSectionAt(palnbeg, dp2oaMiddle);
            written = sprintf(outptr,"%13c",' ');
            outptr += written;
            strncpy(outptr, p, nbytes);
            outptr += nbytes;
            PutNL(outptr);
        }
        {   p = GetAlnSectionAt(palnbeg, dp2oaTarget);
            if(nodeletions && f)
                written = sprintf(outptr,"Refn.: %5s ", "...");
            else
                written = sprintf(outptr,"Refn.: %5u ", trgbeg);
            outptr += written;
            strncpy(outptr, p, nbytes);
            outptr += nbytes;
            fgaps = (int)(std::count(p, p+nbytes, '-'));
            trgbeg += nbytes - fgaps;
            if(nodeletions) {
                if(alnlen <= f + width)
                    written = sprintf(outptr," %-5d%s", trgend, NL);
                else written = sprintf(outptr," %-5s%s", "...", NL);
            } else
                written = sprintf(outptr," %-5d%s", nbytes<=fgaps? trgbeg: trgbeg-1,NL);
            outptr += written;
        }
        if(1) {
            p = GetAlnSectionAt(palnbeg, dp2oaTargetSSS);
            written = sprintf(outptr,"%-13s","struct");
            outptr += written;
            strncpy(outptr, p, nbytes);
            outptr += nbytes;
            PutNL(outptr);
        }
        PutNL(outptr);
    }
}

// -------------------------------------------------------------------------
// outptr, pointer to the output buffer;
// strndx, structure index in the results list;
inline
void TdFinalizer::FormatFooterPlain(
    char*& outptr,
    int strndx)
{
    static const int referenced = CLOptions::GetO_REFERENCED();
    //address of the relevant transformation matrix data:
    float* ptfmmtx = GetOutputTfmMtxAddress(strndx);
    int written;

    written =
        sprintf(outptr," Rotation [3,3] and translation [3,1] for %s:%s",
            (referenced? "Reference":"Query"), NL);
    outptr += written;

    written = sprintf(outptr,"  %10.6f %10.6f %10.6f    %14.6f%s",
        ptfmmtx[tfmmRot_0_0],ptfmmtx[tfmmRot_0_1],ptfmmtx[tfmmRot_0_2],ptfmmtx[tfmmTrl_0],NL);
    outptr += written;
    written = sprintf(outptr,"  %10.6f %10.6f %10.6f    %14.6f%s",
        ptfmmtx[tfmmRot_1_0],ptfmmtx[tfmmRot_1_1],ptfmmtx[tfmmRot_1_2],ptfmmtx[tfmmTrl_1],NL);
    outptr += written;
    written = sprintf(outptr,"  %10.6f %10.6f %10.6f    %14.6f%s%s",
        ptfmmtx[tfmmRot_2_0],ptfmmtx[tfmmRot_2_1],ptfmmtx[tfmmRot_2_2],ptfmmtx[tfmmTrl_2],NL,NL);
    outptr += written;
}



// -------------------------------------------------------------------------
// GetSizeOfCompressedResultsPlain: get total size required for annotations 
// and complete alignments; using plain format;
// szannot, size of annotations;
// szalns, size of complete alignments (with descriptions);
// szalnswodesc, size of alignments without descriptions;
//
inline
void TdFinalizer::GetSizeOfCompressedResultsPlain(
    size_t* szannot, size_t* szalns, size_t* szalnswodesc) const
{
    MYMSG("TdFinalizer::GetSizeOfCompressedResultsPlain", 5);
    static const unsigned int sznl = (int)strlen(NL);
    static const unsigned int indent = OUTPUTINDENT;
//     static const unsigned int annotlen = ANNOTATIONLEN;
    const unsigned int dsclen = DEFAULT_DESCRIPTION_LENGTH;
    const unsigned int dscwidth = MAX_DESCRIPTION_LENGTH;//no wrap (pathname)
    const unsigned int alnwidth = CLOptions::GetO_WRAP();
    const unsigned int width = indent < dscwidth? dscwidth-indent: 1;
    static const unsigned int headlines = 3;//number of lines for scores, etc.
    static const unsigned int footlines = 4;//number of lines for transformation matrix
    static const unsigned int maxlinelen = 170;//maximum length of lines other than alignment lines
    static const unsigned int maxsernlen = 13;//max length for serial number
    int alnsize;//alignment size
    const char* desc;//structure description

    //bool qrysssinuse = true;

    if(!cubp_set_querypmbeg_[0] || !cubp_set_querypmend_[0] || !cubp_set_querydesc_)
        throw MYRUNTIME_ERROR(
        "TdFinalizer::GetSizeOfCompressedResultsPlain: Null query data.");

    if(cubp_set_bdbCpmbeg_[0] && cubp_set_bdbCpmend_[0] && !cubp_set_bdbCdesc_)
        throw MYRUNTIME_ERROR(
        "TdFinalizer::GetSizeOfCompressedResultsPlain: Null structure descriptions.");

    *szannot = 0UL;
    *szalns = 0UL;
    *szalnswodesc = 0UL;

    //on the first receive of results, they can be empty if 
    // there are no hits found
    if(!cubp_set_h_results_)
        return;

    for(int strndx = 0; strndx < qrynstrs_; strndx++)
    {
        float tmscoreq = GetOutputAlnDataField<float>(strndx, dp2oadScoreQ);
        float tmscorer = GetOutputAlnDataField<float>(strndx, dp2oadScoreR);
        float tmscore = PCMAX(tmscoreq, tmscorer);
        if( tmscore < cubp_set_scorethld_)
            continue;

        unsigned int alnlen = (unsigned int)GetOutputAlnDataField<float>(strndx, dp2oadAlnLength);
        //NOTE: alignment length can be 0 due to pre-screening
        if( alnlen < 1 )
            continue;

        // //get the index over all (two) db pm data structures:
        // unsigned int orgstrndx = GetOutputAlnDataField<unsigned int>(strndx, dp2oadOrgStrNo);
        // //distance form the beginning in phase 2 (structures passed through phase 1)
        // unsigned int dbstr2dst = GetOutputAlnDataField<unsigned int>(strndx, dp2oadStrNewDst);
        // int dbstrlen = GetDbStructureField<INTYPE>(orgstrndx, pps2DLen);
        unsigned int dbstr2dst = GetDbStructureField<unsigned int>(strndx, pps2DDist);
        int dbstrlen = GetDbStructureField<INTYPE>(strndx, pps2DLen);
        if(qrynposits_ < dbstr2dst + (unsigned int)dbstrlen)
            //the structure has not been processed due to memory restrictions
            continue;

// fprintf(stderr,"dp2oadOrgStrNo= %d\ndp2oadStrNewDst= %d\n"
// "dp2oadBegCoords= %x (%d %d)\ndp2oadEndCoords= %x (%d %d)\n"
// "dp2oadAlnLength= %d\ndp2oadPstvs= %d dp2oadIdnts= %d dp2oadNGaps= %d\n"
// "dp2oadRMSD= %.4f dp2oadScoreQ= %.6f dp2oadScoreR= %.6f "
// "dp2oadD0Q= %.2f dp2oadD0R= %.2f\n",
// GetOutputAlnDataField<int>(strndx,dp2oadOrgStrNo),
// GetOutputAlnDataField<int>(strndx,dp2oadStrNewDst),
// GetOutputAlnDataField<int>(strndx,dp2oadBegCoords),
// GetCoordY(GetOutputAlnDataField<int>(strndx,dp2oadBegCoords)),
// GetCoordX(GetOutputAlnDataField<int>(strndx,dp2oadBegCoords)),
// GetOutputAlnDataField<int>(strndx,dp2oadEndCoords),
// GetCoordY(GetOutputAlnDataField<int>(strndx,dp2oadEndCoords)),
// GetCoordX(GetOutputAlnDataField<int>(strndx,dp2oadEndCoords)),
// GetOutputAlnDataField<int>(strndx,dp2oadAlnLength),
// GetOutputAlnDataField<int>(strndx,dp2oadPstvs),
// GetOutputAlnDataField<int>(strndx,dp2oadIdnts),
// GetOutputAlnDataField<int>(strndx,dp2oadNGaps),
// GetOutputAlnDataField<float>(strndx,dp2oadRMSD),
// GetOutputAlnDataField<float>(strndx,dp2oadScoreQ),
// GetOutputAlnDataField<float>(strndx,dp2oadScoreR),
// GetOutputAlnDataField<float>(strndx,dp2oadD0Q),
// GetOutputAlnDataField<float>(strndx,dp2oadD0R)
// );

        unsigned int varwidth = alnlen < alnwidth? alnlen: alnwidth;
        //calculate the size of the alignment section...
        int alnfrags = (alnlen + alnwidth - 1)/alnwidth;
        int alnlines = nTDP2OutputAlignmentSSS;

        alnsize = maxsernlen + 4 * sznl;//alignment separator, including serial number
        alnsize += headlines * (maxlinelen+sznl) + 2 * sznl;
        alnsize += alnfrags * alnlines * (varwidth + 2*indent + sznl) + (alnfrags+1) * sznl;
        alnsize += footlines * (maxlinelen+sznl) + 2 * sznl;
        *szalnswodesc += alnsize;

        //calculate the size for description...
        // GetDbStructureDesc(desc, orgstrndx);
        GetDbStructureDesc(desc, strndx);

// /**TEST*/fprintf(stderr,"desc= %s\n strndx= %u cubp_set_scorethld_= %f tmscore= %f "
// "cubp_set_nqyposs_= %d dbstrlen= %d alnlen= %u alnsize= %d alnfrags= %d\n",
// desc,strndx,cubp_set_scorethld_,tmscore,cubp_set_nqyposs_,dbstrlen,alnlen,alnsize,alnfrags);

        alnlen = (unsigned int)(strlen(desc) + 2);
        alnlen = PCMIN(alnlen, dsclen);
        varwidth = alnlen < dscwidth? alnlen: dscwidth;
        alnfrags = (alnlen + width - 1)/width;
        alnsize += alnfrags * (varwidth + sznl) + sznl;

        *szannot += maxlinelen + sznl;
        *szalns += alnsize;
    }

    MYMSGBEGl(5)
        char msgbuf[KBYTE];
        sprintf(msgbuf,"%sszannot %zu szalns %zu (w/o desc. %zu)",
            "TdFinalizer::GetSizeOfCompressedResultsPlain: ",
            *szannot,*szalns,*szalnswodesc);
        MYMSG(msgbuf, 5);
    MYMSGENDl
}
