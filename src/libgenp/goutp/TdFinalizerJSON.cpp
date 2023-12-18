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
    static const unsigned int annotlen = ANNOTATIONLEN;
    const unsigned int dsclen = DEFAULT_DESCRIPTION_LENGTH;
    size_t szannot = 0UL;
    size_t szalns = 0UL;
    size_t szalnswodesc = 0UL;
    const char* desc;//structure description
    int written, sernr = 0;//initialize serial number here

    GetSizeOfCompressedResultsJSON(&szannot, &szalns, &szalnswodesc);

    annotations_.reset();
    alignments_.reset();

    ReserveVectors(qrynstrs_);

    if(szalns < szannot || 
       szalnswodesc > 4 * 
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

    for(int strndx = 0; strndx < qrynstrs_; strndx++)
    {
        float tmscore = GetOutputAlnDataField<float>(strndx, dp2oadScoreQ);
        if( tmscore < cubp_set_scorethld_)
            continue;

        unsigned int alnlen = GetOutputAlnDataField<unsigned int>(strndx, dp2oadAlnLength);
        if(alnlen < 1)
            continue;

        //get the index over all (two) db pm structures
        unsigned int orgstrndx = GetOutputAlnDataField<unsigned int>(strndx, dp2oadOrgStrNo);
        //distance form the beginning in phase 2 (structures passed through phase 1)
        unsigned int dbstr2dst = GetOutputAlnDataField<unsigned int>(strndx, dp2oadStrNewDst);
        int dbstrlen = GetDbStructureField<INTYPE>(orgstrndx, pps2DLen);
        if(qrynposits_ < dbstr2dst + (unsigned int)dbstrlen)
            //the structure has not been processed due to GPU memory restrictions
            continue;

        //save the addresses of the annotations and alignment records
        srtindxs_->push_back(sernr++);
        scores_->push_back(tmscore);
        annotptrs_->push_back(annptr);
        alnptrs_->push_back(outptr);

        //get the name and description
        GetDbStructureDesc(desc, orgstrndx);
        //make an annotation
        MakeAnnotationJSON(annptr, desc, annotlen, tmscore);
        *annptr++ = 0;//end of record

        //compress the alignment and relative information...
        written = sprintf(outptr,
                "    {\"pair_record\": {%s"
                "      \"reference_description\": \"",NL);
        outptr += written;

        //put the description...
        int outpos = 0;
        FormatDescriptionJSON(outptr, desc, dsclen, outpos);
        written = sprintf(outptr,"\",%s",NL);
        outptr += written;
        written = sprintf(outptr,
                "      \"query_length\": %d,%s"
                "      \"reference_length\": %d,%s"
                "      \"alignment\": {%s",
                qrystrlen_, NL, dbstrlen, NL,NL);
        outptr += written;

        FormatScoresJSON(outptr, strndx, alnlen, tmscore);
        FormatAlignmentJSON(outptr,
            strndx, orgstrndx, dbstr2dst, alnlen,
            qrystrlen_, dbstrlen);
        written = sprintf(outptr,"      }%s",NL);
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
// desc, structure description;
// maxoutlen, maximum length of output description;
// tmscore, TM-score;
inline
void TdFinalizer::MakeAnnotationJSON( 
    char*& outptr,
    const char* desc,
    const int maxoutlen,
    const float tmscore) const
{
    int outpos = 0;
    int written;

    written = sprintf(outptr,
                "    {\"summary_entry\": {%s"
                "      \"description\": \"",NL);
    outptr += written;

    FormatDescriptionJSON(outptr, desc, maxoutlen, outpos);

    written = sprintf(outptr,"\",%s",NL);
    outptr += written;

    written = sprintf(outptr,
                "      \"tmscore\": %.4f%s"
                "    }}",//,%s",
            tmscore,NL/*,NL*/);
    outptr += written;
}

// -------------------------------------------------------------------------
// outptr, pointer to the output buffer;
// strndx, structure index in the results list;
// alnlen, alignment length;
// tmscore, TM-score;
inline
void TdFinalizer::FormatScoresJSON(
    char*& outptr,
    int strndx,
    unsigned int alnlen,
    float tmscore)
{
    int written;
    const float d0arg = CLOptions::GetC_D0();
    const int normlenarg = CLOptions::GetC_U();
    const int athrldarg = CLOptions::GetC_A();
    float rmsd = GetOutputAlnDataField<float>(strndx, dp2oadRMSD);
    float tmscoreb = GetOutputAlnDataField<float>(strndx, dp2oadScoreQ);
    int psts = (int)GetOutputAlnDataField<float>(strndx, dp2oadPstvs);
    int idts = (int)GetOutputAlnDataField<float>(strndx, dp2oadIdnts);
    int gaps = (int)GetOutputAlnDataField<float>(strndx, dp2oadNGaps);

    static const char* na = "NA";
    char tmabbuf[BUF_MAX/2], tmlbuf[BUF_MAX/2], tmd0buf[BUF_MAX/2];
    const char* tmab = na, *tml =  na, *tmd0 =  na;

    if(athrldarg == CLOptions::csnAvgLength) {
        float tmscoreab = GetOutputAlnDataField<float>(strndx, dp2oadScoreR);
        sprintf(tmabbuf,"%.5g",tmscoreab);
        tmab = tmabbuf;
    }

    if(normlenarg) {
        float tmscorel = GetOutputAlnDataField<float>(strndx, dp2oadD0Q);
        sprintf(tmlbuf,"%.5g",tmscorel);
        tml = tmlbuf;
    }

    if(d0arg) {
        float tmscored0 = GetOutputAlnDataField<float>(strndx, dp2oadD0R);
        sprintf(tmd0buf,"%.5g",tmscored0);
        tmd0 = tmd0buf;
    }

    written = sprintf(outptr,
            "        \"tmscore\": %.5f,%s"
            "        \"tmscore_query\": %.5f,%s"
            "        \"RMSD\": %.2f,%s"
            "        \"tmscore_avglen\": \"%s\",%s"
            "        \"tmscore_length\": \"%s\",%s"
            "        \"tmscore_d0\": \"%s\",%s",
            tmscore,NL,tmscoreb,NL,rmsd,NL,
            tmab,NL,tml,NL,tmd0,NL);
    outptr += written;
    written = sprintf(outptr,
            "        \"n_identities\": %d,%s"
            "        \"n_matched\": %d,%s"
            "        \"n_gaps\": %d,%s"
            "        \"aln_length\": %d,%s",
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
    int dbstrlen)
{
    //alignment beginning coordinates:
    unsigned int alnbegcoords = GetOutputAlnDataField<unsigned int>(strndx, dp2oadBegCoords);
    unsigned int qrybeg = GetCoordY(alnbegcoords)+1;
    unsigned int trgbeg = GetCoordX(alnbegcoords)+1;
    //offset to the end of the db structure in phase 2:
    int dbpos2off = dbstr2dst + dbstrlen-1;
    //alignment beginning position:
    int alnbeg = 
        offsetalns_ + //offset to the alignments produced for the current query (qrysernr_)
        dbpos2off + (strndx+1) * querylen - alnlen;//TODO: consider adding 1 to qlen, +1

    //beginning of the alignment:
    const char* palnbeg = GetBegOfAlns() + alnbeg;
    const char* pqueryaln = GetAlnSectionAt(palnbeg, dp2oaQuery);
    const char* ptargetaln = GetAlnSectionAt(palnbeg, dp2oaTarget);
    const char* p;

    int written;
    int nbytes = alnlen;
    int ngapsqry = (int)(std::count(pqueryaln, pqueryaln+nbytes, '-'));//#gaps in query
    int ngapstrg = (int)(std::count(ptargetaln, ptargetaln+nbytes, '-'));//#gaps in target

    unsigned int qryend = qrybeg + nbytes - ngapsqry;
    unsigned int trgend = trgbeg + nbytes - ngapstrg;

    written = sprintf(outptr,
            "        \"query_from\": %u,%s"
            "        \"query_to\": %u,%s"
            "        \"refrn_from\": %u,%s"
            "        \"refrn_to\": %u,%s",
            qrybeg,NL,nbytes<=ngapsqry? qryend: qryend-1,NL,
            trgbeg,NL,nbytes<=ngapstrg? trgend: trgend-1,NL);
    outptr += written;

    written = sprintf(outptr,
            "        \"query_secstr_available\": %d,%s"
            "        \"refrn_secstr_available\": %d,%s"
            "        \"query_secstr\": \"",
            1,NL,1,NL);
    outptr += written;

    if(1) {
        p = GetAlnSectionAt(palnbeg, dp2oaQuerySSS);
        strncpy(outptr, p, nbytes);
        outptr += nbytes;
    }
    written = sprintf( outptr,"\",%s        \"refrn_secstr\": \"",NL);
    outptr += written;
    if(1) {
        p = GetAlnSectionAt(palnbeg, dp2oaTargetSSS);
        strncpy(outptr, p, nbytes);
        outptr += nbytes;
    }
    written = sprintf( outptr,"\",%s        \"query_aln\": \"",NL);
    outptr += written;
    strncpy(outptr, pqueryaln, nbytes);
    outptr += nbytes;
    written = sprintf( outptr,"\",%s        \"refrn_aln\": \"",NL);
    outptr += written;
    strncpy(outptr, ptargetaln, nbytes);
    outptr += nbytes;
    written = sprintf( outptr,"\",%s        \"middle_ln\": \"",NL);
    outptr += written;
    p = GetAlnSectionAt(palnbeg, dp2oaMiddle);
    strncpy(outptr, p, nbytes);
    outptr += nbytes;
    written = sprintf( outptr,"\",%s",NL);
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
    //address of the relevant transformation matrix data:
    float* ptfmmtx = GetOutputTfmMtxAddress(strndx);
    int written;

    written = sprintf(outptr,
            "      \"transformation_matrix\": {%s"
            "        \"transl_vector_0\": %.6f,%s"
            "        \"transl_vector_1\": %.6f,%s"
            "        \"transl_vector_2\": %.6f,%s"
            "        \"rot_matrix_0_0\": %.6f,%s"
            "        \"rot_matrix_0_1\": %.6f,%s"
            "        \"rot_matrix_0_2\": %.6f,%s"
            "        \"rot_matrix_1_0\": %.6f,%s"
            "        \"rot_matrix_1_1\": %.6f,%s"
            "        \"rot_matrix_1_2\": %.6f,%s"
            "        \"rot_matrix_2_0\": %.6f,%s"
            "        \"rot_matrix_2_1\": %.6f,%s"
            "        \"rot_matrix_2_2\": %.6f%s"
            "      }%s",
        NL,
        ptfmmtx[tfmmTrl_0],NL,ptfmmtx[tfmmTrl_1],NL,ptfmmtx[tfmmTrl_2],NL,
        ptfmmtx[tfmmRot_0_0],NL,ptfmmtx[tfmmRot_0_1],NL,ptfmmtx[tfmmRot_0_2],NL,
        ptfmmtx[tfmmRot_1_0],NL,ptfmmtx[tfmmRot_1_1],NL,ptfmmtx[tfmmRot_1_2],NL,
        ptfmmtx[tfmmRot_2_0],NL,ptfmmtx[tfmmRot_2_1],NL,ptfmmtx[tfmmRot_2_2],NL,
        NL);
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
    static const unsigned int annotlen = ANNOTATIONLEN;
    static const unsigned int annotheadlines = 1;//number of lines for score
    const unsigned int dsclen = DEFAULT_DESCRIPTION_LENGTH;
    static const unsigned int openlines = 4;//number of lines for opening a hit record
    static const unsigned int headlines = 16;//number of lines for scores, etc.
    static const unsigned int footlines = nTTranformMatrix;//number of lines for transformation matrix entries
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
        float tmscore = GetOutputAlnDataField<float>(strndx, dp2oadScoreQ);
        if(tmscore < cubp_set_scorethld_)
            continue;

        unsigned int alnlen = GetOutputAlnDataField<unsigned int>(strndx, dp2oadAlnLength);
        if( alnlen < 1 )
            continue;

        //get the index over all (two) db pm data structures:
        unsigned int orgstrndx = GetOutputAlnDataField<unsigned int>(strndx, dp2oadOrgStrNo);
        //distance form the beginning in phase 2 (structures passed through phase 1)
        unsigned int dbstr2dst = GetOutputAlnDataField<unsigned int>(strndx, dp2oadStrNewDst);
        int dbstrlen = GetDbStructureField<INTYPE>(orgstrndx, pps2DLen);
        if(qrynposits_ < dbstr2dst + (unsigned int)dbstrlen)
            //the structure has not been processed due to memory restrictions
            continue;

        unsigned int varwidth = alnlen;
        //calculate the size of the alignment section...
        int alnlines = nTDP2OutputAlignment;

        if(1) {
            //unsigned int dbstrdst = GetDbStructureField<LNTYPE>(orgstrndx, pps2DDist);
            bool trgsssinuse = true;
            if(trgsssinuse)
                alnlines = nTDP2OutputAlignmentSSS;
        }

        alnsize = openlines * (maxopenlinelen+sznl);//opening
        alnsize += headlines * (maxlinelen+sznl);
        alnsize += alnlines * (varwidth + maxfieldlen + sznl + 2) + 2*(maxfieldlen + sznl + 2);//+2 (quotes)
        alnsize += footlines * (maxlinelen+sznl) + 2*(maxfieldlen + sznl + 2);//+2 (quotes);
        alnsize += closlines * (maxcloslinelen+sznl);
        *szalnswodesc += alnsize;

        //calculate the size for description...
        GetDbStructureDesc(desc, orgstrndx);

        alnlen = (unsigned int)(strlen(desc) + 2);
        alnlen = PCMIN(alnlen, dsclen);
        varwidth = alnlen;
        alnsize += (varwidth + maxfieldlen + sznl + 2);

        *szannot += maxfieldlen + sznl/*open*/ + maxcloslinelen + sznl/*close*/ + 
                annotlen + maxfieldlen + sznl + 2/*annotation*/ + 
                annotheadlines * (maxlinelen+sznl)/*score*/;
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
