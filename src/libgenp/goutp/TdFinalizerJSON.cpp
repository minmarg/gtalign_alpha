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
// CompressResultsJSON: process results obtained; compress them for passing 
// them to the writing thread; use JSON format
// NOTE: all operations performed under lock
//
void TdFinalizer::CompressResultsJSON()
{
    MYMSG( "TdFinalizer::CompressResultsJSON", 4 );
    // static const unsigned int annotlen = ANNOTATIONLEN;
    const unsigned int dsclen = DEFAULT_DESCRIPTION_LENGTH;
    size_t szannot = 0UL;
    size_t szalns = 0UL;
    size_t szalnswodesc = 0UL;
    const char* desc;//structure description
    int written, sernr = 0;//initialize serial number here

    if(cubp_set_qrysernrbeg_ < 0 || qrysernr_ < cubp_set_qrysernrbeg_ || 
       cubp_set_qrysernrbeg_ + (int)cubp_set_nqystrs_ <= qrysernr_)
        throw MYRUNTIME_ERROR(
        "TdFinalizer::CompressResultsJSON: Invalid query indices.");

    GetSizeOfCompressedResultsJSON(&szannot, &szalns, &szalnswodesc);

    annotations_.reset();
    alignments_.reset();

    ReserveVectors(qrynstrs_);

    if(szalns < szannot || 
       szalnswodesc > 10 * 
            (cubp_set_sz_alndata_ + cubp_set_sz_tfmmatrices_ + cubp_set_sz_alns_))
        throw MYRUNTIME_ERROR(
        "TdFinalizer::CompressResultsJSON: "
        "Size of formatted results is unusually large.");

    if(szannot < 1 || szalns < 1)
        return;

    annotations_.reset((char*)std::malloc(szannot));
    alignments_.reset((char*)std::malloc(szalns));

    if(!annotations_ || !alignments_)
        throw MYRUNTIME_ERROR(
        "TdFinalizer::CompressResultsJSON: Not enough memory.");

    if(!srtindxs_ || !scores_ || !alnptrs_ || !annotptrs_)
        throw MYRUNTIME_ERROR(
        "TdFinalizer::CompressResultsJSON: Not enough memory.");

    char* annptr = annotations_.get();
    char* outptr = alignments_.get();

    static const int bsectmscore = CLOptions::GetO_2TM_SCORE();
    static const int sortby = CLOptions::GetO_SORT();

    for(int strndx = 0; strndx < qrynstrs_; strndx++)
    {
        float tmscoreq = GetOutputAlnDataField<float>(strndx, dp2oadScoreQ);
        float tmscorer = GetOutputAlnDataField<float>(strndx, dp2oadScoreR);
        float rmsd = GetOutputAlnDataField<float>(strndx, dp2oadRMSD);
        float tmscoregrt = PCMAX(tmscoreq, tmscorer);
        float tmscorehmn = (0.0f < tmscoregrt)? (2.f * tmscoreq * tmscorer) / (tmscoreq + tmscorer): 0.0f;
        float tmscore = tmscoregrt;
        if(sortby == CLOptions::osTMscoreReference) tmscore = tmscorer;
        if(sortby == CLOptions::osTMscoreQuery) tmscore = tmscoreq;
        if(sortby == CLOptions::osTMscoreHarmonic) tmscore = tmscorehmn;
        if(sortby == CLOptions::osRMSD) tmscore = -rmsd;
        if(bsectmscore && sortby > CLOptions::osRMSD) {
            float sectmscoreq = GetOutputAlnDataField<float>(strndx, dp2oad2ScoreQ);
            float sectmscorer = GetOutputAlnDataField<float>(strndx, dp2oad2ScoreR);
            tmscore = PCMAX(sectmscoreq, sectmscorer);
            float sectmscorehmn = (0.0f < tmscore)? (2.f * sectmscoreq * sectmscorer) / (sectmscoreq + sectmscorer): 0.0f;
            if(sortby == CLOptions::os2TMscoreReference) tmscore = sectmscorer;
            if(sortby == CLOptions::os2TMscoreQuery) tmscore = sectmscoreq;
            if(sortby == CLOptions::os2TMscoreHarmonic) tmscore = sectmscorehmn;
        }

        if(tmscoregrt < cubp_set_scorethld_)
            continue;

        unsigned int alnlen = (unsigned int)GetOutputAlnDataField<float>(strndx, dp2oadAlnLength);
        //NOTE: alignment length can be 0 due to pre-screening
        if(alnlen < 1)
            continue;

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

        //get the name and description
        GetDbStructureDesc(desc, strndx);
        //make an annotation
        MakeAnnotationJSON(annptr, strndx, desc, alnlen, dbstrlen);
        *annptr++ = 0;//end of record

        //compress the alignment and relative information...
        written = sprintf(outptr,
                "    {\"hit_record\": {%s"
                "      \"reference_description\": \"",NL);
        outptr += written;

        //put the description...
        int outpos = 0;
        FormatDescriptionJSON(outptr, desc, strlen(desc), dsclen, outpos);
        written = sprintf(outptr,"\",%s",NL);
        outptr += written;
        written = sprintf(outptr,
                "      \"query_length\": %d,%s"
                "      \"reference_length\": %d,%s"
                "      \"alignment\": {%s",
                qrystrlen_, NL, dbstrlen, NL,NL);
        outptr += written;

        FormatScoresJSON(outptr, strndx, alnlen);
        FormatAlignmentJSON(outptr,
            strndx, strndx, dbstr2dst, alnlen,
            qrystrlen_, dbstrlen);
        written = sprintf(outptr,"      },%s",NL);
        outptr += written;

        FormatFooterJSON(outptr, strndx);

        written = sprintf(outptr,"    }}");//,%s",NL);
        outptr += written;
        *outptr++ = 0;//end of record
    }
}

// -------------------------------------------------------------------------
// MakeAnnotationJSON: format structure description;
// NOTE: space is assumed to be pre-allocated;
// outptr, pointer to the output buffer;
// strndx, reference structure serial number;
// desc, structure description;
// alnlen, alignment length;
// dbstrlen, reference structure length;
// tmscore, TM-score;
inline
void TdFinalizer::MakeAnnotationJSON( 
    char*& outptr,
    const int strndx,
    const char* desc,
    const unsigned int alnlen,
    const int dbstrlen) const
{
    int written, outpos = 0;
    const unsigned int anndesclen = ANNOTATION_DESCLEN;
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
    int desclen = strlen(desc);

    written = sprintf(outptr,
                "    {\"summary_entry\": {%s"
                "      \"description\": \"",NL);
    outptr += written;

    FormatDescriptionJSON(outptr, desc, desclen, anndesclen, outpos);

    written = sprintf(outptr,"\",%s",NL);
    outptr += written;

    written = sprintf(outptr,
                "      \"tmscore_refn\": %.4f,%s"
                "      \"tmscore_query\": %.4f,%s"
                "      \"rmsd\": %.2f,%s"
                "      \"n_aligned\": %d,%s"
                "      \"query_from\": %d,%s"
                "      \"query_to\": %d,%s"
                "      \"refn_from\": %d,%s"
                "      \"refn_to\": %d,%s"
                "      \"refn_length\": %d%s"
                "    }}",
        tmscorer,NL,tmscoreq,NL,rmsd,NL,
        (int)alnlen-gaps,NL,qrybeg,NL,qryend,NL,trgbeg,NL,trgend,NL,dbstrlen,NL);
    outptr += written;
}

// -------------------------------------------------------------------------
// outptr, pointer to the output buffer;
// strndx, structure index in the results list;
// alnlen, alignment length;
inline
void TdFinalizer::FormatScoresJSON(
    char*& outptr,
    int strndx,
    unsigned int alnlen)
{
    static const int bsectmscore = CLOptions::GetO_2TM_SCORE();
    int written;
    float rmsd = GetOutputAlnDataField<float>(strndx, dp2oadRMSD);
    float tmscoreq = GetOutputAlnDataField<float>(strndx, dp2oadScoreQ);
    float tmscorer = GetOutputAlnDataField<float>(strndx, dp2oadScoreR);
    float sectmscoreq = GetOutputAlnDataField<float>(strndx, dp2oad2ScoreQ);
    float sectmscorer = GetOutputAlnDataField<float>(strndx, dp2oad2ScoreR);
    float d0q = GetOutputAlnDataField<float>(strndx, dp2oadD0Q);
    float d0r = GetOutputAlnDataField<float>(strndx, dp2oadD0R);
    int psts = (int)GetOutputAlnDataField<float>(strndx, dp2oadPstvs);
    int idts = (int)GetOutputAlnDataField<float>(strndx, dp2oadIdnts);
    int gaps = (int)GetOutputAlnDataField<float>(strndx, dp2oadNGaps);

    static const char* na = "NA";
    char tm2qbuf[BUF_MAX/2], tm2rbuf[BUF_MAX/2];
    const char* tm2q = na, *tm2r = na;

    if(bsectmscore) {
        sprintf(tm2qbuf,"%.5g",sectmscoreq); tm2q = tm2qbuf;
        sprintf(tm2rbuf,"%.5g",sectmscorer); tm2r = tm2rbuf;
    }

    written = sprintf(outptr,
            "        \"tmscore_refn\": %.5f,%s"
            "        \"tmscore_query\": %.5f,%s"
            "        \"d0_refn\": %.2f,%s"
            "        \"d0_query\": %.2f,%s"
            "        \"rmsd\": %.2f,%s"
            "        \"2tmscore_refn\": \"%s\",%s"
            "        \"2tmscore_query\": \"%s\",%s"
            "        \"n_identities\": %d,%s"
            "        \"n_matched\": %d,%s"
            "        \"n_gaps\": %d,%s"
            "        \"n_aligned\": %d,%s",
            tmscorer,NL,tmscoreq,NL,d0r,NL,d0q,NL,rmsd,NL,
            tm2r,NL,tm2q,NL,
            idts,NL,psts,NL,gaps,NL,alnlen,NL);
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
inline
void TdFinalizer::FormatAlignmentJSON(
    char*& outptr,
    int strndx,
    unsigned int /*orgstrndx*/,
    unsigned int dbstr2dst,
    int alnlen,
    int querylen,
    int /* dbstrlen */)
{
    //alignment beginning coordinates:
    unsigned int alnbegcoords = GetOutputAlnDataField<unsigned int>(strndx, dp2oadBegCoords);
    unsigned int qrybeg = GetCoordY(alnbegcoords);
    unsigned int trgbeg = GetCoordX(alnbegcoords);

    //alignment beginning position:
    int alnbeg = 
        offsetalns_ + //offset to the alignments produced for the current query (qrysernr_)
        dbstr2dst + strndx * (querylen + 1);//alignment position for strndx within the query section

    //beginning of the alignment:
    const char* palnbeg = GetBegOfAlns() + alnbeg;
    const char* pquerysss = GetAlnSectionAt(palnbeg, dp2oaQuerySSS);
    const char* pqueryaln = GetAlnSectionAt(palnbeg, dp2oaQuery);
    const char* prefnaln = GetAlnSectionAt(palnbeg, dp2oaTarget);
    const char* prefnsss = GetAlnSectionAt(palnbeg, dp2oaTargetSSS);
    const char* pmiddle = GetAlnSectionAt(palnbeg, dp2oaMiddle);

    int written;
    int nbytes = alnlen;
    int ngapsqry = (int)(std::count(pqueryaln, pqueryaln + nbytes, '-'));//#gaps in query
    int ngapsrfn = (int)(std::count(prefnaln, prefnaln + nbytes, '-'));//#gaps in reference

    unsigned int qryend = qrybeg + nbytes - ngapsqry;
    unsigned int rfnend = trgbeg + nbytes - ngapsrfn;

    written = sprintf(outptr,
            "        \"query_from\": %u,%s"
            "        \"query_to\": %u,%s"
            "        \"refn_from\": %u,%s"
            "        \"refn_to\": %u,%s",
            qrybeg,NL,nbytes<=ngapsqry? qryend: qryend-1,NL,
            trgbeg,NL,nbytes<=ngapsrfn? rfnend: rfnend-1,NL);
    outptr += written;

    written = sprintf(outptr,"        \"query_secstr\": \"");
    outptr += written;
    strncpy(outptr, pquerysss, nbytes);
    outptr += nbytes;
    written = sprintf( outptr,"\",%s        \"refrn_secstr\": \"",NL);
    outptr += written;
    strncpy(outptr, prefnsss, nbytes);
    outptr += nbytes;
    written = sprintf( outptr,"\",%s        \"query_aln\": \"",NL);
    outptr += written;
    strncpy(outptr, pqueryaln, nbytes);
    outptr += nbytes;
    written = sprintf( outptr,"\",%s        \"refrn_aln\": \"",NL);
    outptr += written;
    strncpy(outptr, prefnaln, nbytes);
    outptr += nbytes;
    written = sprintf( outptr,"\",%s        \"middle\": \"",NL);
    outptr += written;
    strncpy(outptr, pmiddle, nbytes);
    outptr += nbytes;
    written = sprintf( outptr,"\"%s",NL);
    outptr += written;
}

// -------------------------------------------------------------------------
// outptr, pointer to the output buffer;
// strndx, structure index in the results list;
inline
void TdFinalizer::FormatFooterJSON(
    char*& outptr,
    int strndx)
{
    static const int referenced = CLOptions::GetO_REFERENCED();
    //address of the relevant transformation matrix data:
    float* ptfmmtx = GetOutputTfmMtxAddress(strndx);
    int written;

    written = sprintf(outptr,
            "      \"tfm_referenced\": %d,%s",referenced,NL);
    outptr += written;
    written = sprintf(outptr,
            "      \"rotation_matrix_rowmajor\": ["
            "%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f],%s",
        ptfmmtx[tfmmRot_0_0],ptfmmtx[tfmmRot_0_1],ptfmmtx[tfmmRot_0_2],
        ptfmmtx[tfmmRot_1_0],ptfmmtx[tfmmRot_1_1],ptfmmtx[tfmmRot_1_2],
        ptfmmtx[tfmmRot_2_0],ptfmmtx[tfmmRot_2_1],ptfmmtx[tfmmRot_2_2],
        NL);
    outptr += written;
    written = sprintf(outptr,
            "      \"translation_vector\": [%.6f, %.6f, %.6f]%s",
        ptfmmtx[tfmmTrl_0],ptfmmtx[tfmmTrl_1],ptfmmtx[tfmmTrl_2],NL);
    outptr += written;
}



// -------------------------------------------------------------------------
// GetSizeOfCompressedResultsJSON: get total size required for annotations 
// and complete alignments; using JSON format;
// szannot, size of annotations;
// szalns, size of complete alignments (with descriptions);
// szalnswodesc, size of alignments without descriptions;
//
inline
void TdFinalizer::GetSizeOfCompressedResultsJSON(
    size_t* szannot, size_t* szalns, size_t* szalnswodesc) const
{
    MYMSG("TdFinalizer::GetSizeOfCompressedResultsJSON", 5);
    static const unsigned int sznl = (int)strlen(NL);
    static const unsigned int maxfieldlen = 40;//JSON max field length
    // static const unsigned int annotlen = ANNOTATIONLEN;
    static const unsigned int annotheadlines = 9;//number of lines for statistics in annotation
    const unsigned int dsclen = DEFAULT_DESCRIPTION_LENGTH;
    const unsigned int anndesclen = ANNOTATION_DESCLEN * 2;
    static const unsigned int openlines = 4;//number of lines for opening a hit record
    static const unsigned int headlines = 17;//number of lines for scores, etc.
    static const unsigned int footlines = 2;//nTTranformMatrix;//number of lines for transformation matrix entries
    static const unsigned int closlines = 2;//number of lines for closing a hit record
    static const unsigned int maxlinelen = 90;//maximum length of lines other than alignment lines
    static const unsigned int maxopenlinelen = 50;//maximum length of `openlines'
    static const unsigned int maxcloslinelen = 20;//maximum length of `closlines'
    int alnsize;//alignment size
    const char* desc;//structure description

    //bool qrysssinuse = true;

    if(!cubp_set_querypmbeg_[0] || !cubp_set_querypmend_[0] || !cubp_set_querydesc_)
        throw MYRUNTIME_ERROR(
        "TdFinalizer::GetSizeOfCompressedResultsJSON: Null query data.");

    if(cubp_set_bdbCpmbeg_[0] && cubp_set_bdbCpmend_[0] && !cubp_set_bdbCdesc_)
        throw MYRUNTIME_ERROR(
        "TdFinalizer::GetSizeOfCompressedResultsJSON: Null structure descriptions.");

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
        if(tmscore < cubp_set_scorethld_)
            continue;

        unsigned int alnlen = (unsigned int)GetOutputAlnDataField<float>(strndx, dp2oadAlnLength);
        //NOTE: alignment length can be 0 due to pre-screening
        if(alnlen < 1)
            continue;

        unsigned int dbstr2dst = GetDbStructureField<unsigned int>(strndx, pps2DDist);
        int dbstrlen = GetDbStructureField<INTYPE>(strndx, pps2DLen);
        if(qrynposits_ < dbstr2dst + (unsigned int)dbstrlen)
            //the structure has not been processed due to memory restrictions
            continue;

        unsigned int varwidth = alnlen;
        //calculate the size of the alignment section...
        int alnlines = nTDP2OutputAlignmentSSS;

        alnsize = openlines * (maxopenlinelen+sznl);//opening
        alnsize += headlines * (maxlinelen+sznl);
        alnsize += alnlines * (varwidth + maxfieldlen + sznl + 2) + 2 * (maxfieldlen + sznl + 2);//+2 (quotes)
        alnsize += footlines * (3 * maxlinelen + sznl) + 2 * (maxfieldlen + sznl + 2);//+2 (quotes);
        alnsize += closlines * (maxcloslinelen + sznl);
        *szalnswodesc += alnsize;

        //calculate the size for description...
        GetDbStructureDesc(desc, strndx);

        alnlen = (unsigned int)(strlen(desc) + 2);
        alnlen = PCMIN(alnlen, dsclen);
        varwidth = alnlen;
        alnsize += (varwidth + maxfieldlen + sznl + 2);

        *szannot += maxfieldlen + sznl/*open*/ + maxcloslinelen + sznl/*close*/ + 
                anndesclen + maxfieldlen + sznl + 2/*annotation*/ + 
                annotheadlines * (maxopenlinelen + sznl)/*statistics*/;
        *szalns += alnsize;
    }

    MYMSGBEGl(5)
        char msgbuf[KBYTE];
        sprintf(msgbuf,"%sszannot %zu szalns %zu (w/o desc. %zu)",
            "TdFinalizer::GetSizeOfCompressedResultsJSON: ",
            *szannot,*szalns,*szalnswodesc);
        MYMSG(msgbuf, 5);
    MYMSGENDl
}
