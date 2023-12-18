/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __PMBatchStrData_inl__
#define __PMBatchStrData_inl__

#include "libutil/mybase.h"

#include <omp.h>

#include <string>

#include "libmymp/mputil/simdscan.h"
#include "libutil/alpha.h"
#include "PM2DVectorFields.h"
#include "PMBatchStrData.h"

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// CheckAlignmentScore: calculate local ungapped alignment score and return
// true if the sequence similarity between the structure in bdbCpmbeg (Ovhd)
// and query querypmbeg is above the threshold seqsimthrscore;
// seqsimthrscore, sequence similarity threshold;
// qrydst, distance to the beginning of the query structure data;
// qrylen, query length;
// dbstrdst, distance to the beginning of the reference structure data;
// dbstrlen, reference length;
// qrybegpos, beginning position for query;
// rfnbegpos, beginning position for reference;
// querypmbeg, query field addresses;
// bdbCpmbeg, reference field addresses;
// 
template<int DIMD>
inline
bool PMBatchStrData::CheckAlignmentScore(
    const float seqsimthrscore,
    const int qrydst,
    const int qrylen,
    const int dbstrdst,
    const int dbstrlen,
    const int qrybegpos,
    const int rfnbegpos,
    const char* const * const __restrict querypmbeg,
    const char* const * const __restrict bdbCpmbeg,
    char rfnRE[DIMD], char qryRE[DIMD],
    float scores[DIMD], float pxmins[DIMD],
    float tmp[DIMD])
{
    MYMSG("PMBatchStrData::CheckAlignmentScore", 5);
    // static const std::string preamb = "PMBatchStrData::CheckAlignmentScore: ";
    //sequence similarity signal: true if Ovhd is similar to any of the queries:
    bool signal = false;

    const int qp = qrybegpos;
    const int rp = rfnbegpos;
    const int alnlen1 = mymin(qrylen - qp, dbstrlen);
    const int alnlen2 = mymin(dbstrlen - rp, qrylen);
    const int alnlen = mymin(alnlen1, alnlen2);
    float prvsum = 0.0f, prvmin = 0.0f, maxsc = 0.0f;

    for(int ai = 0; ai < alnlen && !signal; ai += DIMD)
    {
        //NOTE: piend > 0 by definition;
        const int piend = mymin(DIMD, alnlen - ai);
        #pragma omp simd aligned(querypmbeg,bdbCpmbeg:PMBSdatalignment)
        for(int pi = 0; pi < piend; pi++) {
            qryRE[pi] = PMBatchStrData::GetFieldAt<char,pmv2Drsd>(querypmbeg, qrydst + qp + ai + pi);
            rfnRE[pi] = PMBatchStrData::GetFieldAt<char,pmv2Drsd>(bdbCpmbeg, dbstrdst + rp + ai + pi);
        }
        //calculate positional scores:
        #pragma omp simd
        for(int pi = 0; pi < piend; pi++)
            scores[pi] = GONNET_SCORES.get(qryRE[pi], rfnRE[pi]);
        //add the previous sum, calculate prefix sums and copy them to pxmins:
        scores[0] += prvsum;
        mysimdincprefixsum<DIMD>(piend, scores, tmp);
        #pragma omp simd
        for(int pi = 0; pi < piend; pi++) pxmins[pi] = scores[pi];
        //take min, calculate prefix mins of the prefix sums:
        pxmins[0] = mymin(pxmins[0], prvmin);
        mysimdincprefixminmax<DIMD,PFX_MIN>(piend, pxmins, tmp);
        //update previous values:
        prvsum = scores[piend - 1];
        prvmin = pxmins[piend - 1];
        //calculate local alignment scores
        #pragma omp simd
        for(int pi = 0; pi < piend; pi++) scores[pi] -= mymin(0.0f, pxmins[pi]);
        //find max score:
        #pragma omp simd reduction(max:maxsc)
        for(int pi = 0; pi < piend; pi++) maxsc = mymax(maxsc, scores[pi]);
        if(seqsimthrscore <= maxsc) signal = true;
    }

    return signal;
}

// -------------------------------------------------------------------------

#endif//__PMBatchStrData_inl__