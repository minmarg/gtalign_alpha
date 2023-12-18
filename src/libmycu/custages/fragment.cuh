/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __fragment_cuh__
#define __fragment_cuh__

#include <math.h>
#include "libmycu/cucom/cudef.h"
#include "libmycu/cucom/cutemplates.h"
#include "libmycu/cuproc/cuprocconf.h"

// -------------------------------------------------------------------------
// FragPosWithinAlnBoundaries: return true if position sfragpos of an
// alignment of length maxalnlen is not out of bounds;
// fraglen, fragment length starting at position sfragpos;
__HDINLINE__
int FragPosWithinAlnBoundaries(int maxalnlen, int sfragstep, int sfragpos, int fraglen)
{
    return (sfragpos + fraglen < maxalnlen + sfragstep);
}

// -------------------------------------------------------------------------
// GetMaxNFragSteps: get the max number of steps for an alignment of size 
// maxalnmax using a fragment of length minfraglen and a step of sfragstep; 
// calculated so that the following holds:
// sfragstep * nmaxsteps + minfraglen < maxalnlen + sfragstep
__HDINLINE__
int GetMaxNFragSteps(int maxalnlen, int sfragstep, int fraglen)
{
    //step starts with 0, hence:
    return (int)ceilf((float)(maxalnlen+sfragstep-fraglen)/sfragstep);
}

// -------------------------------------------------------------------------
// GetGplAlnLength: get the maximum gapless alignment length given the lengths 
// and positions of query and reference structures;
__HDINLINE__
int GetGplAlnLength(
    int qrylen, int dbstrlen,
    int qrypos, int rfnpos)
{
    return myhdmin(qrylen-qrypos, dbstrlen-rfnpos);
}

// -------------------------------------------------------------------------
// GetFragLength: return the length of fragment under process, given 
// lengths and positions of query and reference structures; <1 stops 
// further processing
__HDINLINE__
int GetFragLength(
    int qrylen, int dbstrlen,
    int qrypos, int rfnpos,
    int sfragndx)
{
    int alnlen = GetGplAlnLength(qrylen, dbstrlen, qrypos, rfnpos);

    if(alnlen < 4 && sfragndx == 0) return alnlen;

    alnlen >>= sfragndx;

    //manage the case of (alnlen >> (sfragndx-1))==5
    //TODO: consider removing this and check the accuracy
    if(alnlen == 2 && (0 < sfragndx) && ((alnlen >> (sfragndx-1)) & 1)) alnlen = 4;
    if(alnlen < 3) return 0;
    if(alnlen < 4) alnlen = 4;
    if(sfragndx+1 == FRAGREF_NMAXSUBFRAGS) alnlen = 4;

    return alnlen;
}

// -------------------------------------------------------------------------
// GetMinFragLengthForAln: return the minimum length of fragment, given max 
// alignment length maxalnlen;
inline
int GetMinFragLengthForAln(int maxalnlen)
{
    if(maxalnlen < 4) return maxalnlen;
    int fraglen = 0;

    for(int sfragndx = FRAGREF_NMAXSUBFRAGS-1; sfragndx >= 0; sfragndx--)
        if((fraglen = GetFragLength(maxalnlen, maxalnlen, 0, 0, sfragndx)) > 0)
            break;

    return fraglen;
}

// -------------------------------------------------------------------------
// UpdateLengths: update the (virtual) lengths and positions of query and 
// reference structures; 
// NOTE: assumptions: fraglen <= qrylen && fraglen <= dbstrlen
__DINLINE__
void UpdateLengths(
    int& qrylen, int& dbstrlen,
    int& qrypos, int& rfnpos,
    int fraglen)
{
    int qryend = qrypos + fraglen;
    int rfnend = rfnpos + fraglen;

    int delta = myhdmax(qryend - qrylen, rfnend - dbstrlen);

    if(0 < delta) {
        qrypos -= delta; rfnpos -= delta;
    } else {
        qrylen = qryend; dbstrlen = rfnend;
    }
}

// -------------------------------------------------------------------------

#endif//__fragment_cuh__
