/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __scoringbase_h__
#define __scoringbase_h__

#include "libmycu/cucom/cudef.h"


// template constants for kernel InitScores:
// initialize all data:
#define INITOPT_ALL 1
// initialize best scores only:
#define INITOPT_BEST 2
// initialize current scores only:
#define INITOPT_CURRENT 4
// reset query and reference positions:
#define INITOPT_QRYRFNPOS 8
// reset fragment specifications:
#define INITOPT_FRAGSPECS 16
// reset the number of aligned positions:
#define INITOPT_NALNPOSS 32
// reset convergence flag only:
#define INITOPT_CONVFLAG_ALL 64
#define INITOPT_CONVFLAG_FRAGREF 128
#define INITOPT_CONVFLAG_SCOREDP 256
#define INITOPT_CONVFLAG_NOTMPRG 512
#define INITOPT_CONVFLAG_LOWTMSC 1024
#define INITOPT_CONVFLAG_LOWTMSC_SET 2048


// precision of convergence for scores in 
// kernel CheckScoreConvergence
#define SCORE_CONVEPSILON 1.e-6f

// do not check convergence of finding optimal rotation matrix:
// #define CHCKCONV_NOCHECK 0
// check convergence:
// #define CHCKCONV_CHECK 1

// do not save psitional scores:
#define SAVEPOS_NOSAVE 0
// save psitional scores:
#define SAVEPOS_SAVE 1

// reduce all scores unconditionally
#define CHCKDST_NOCHECK 0
// accumulate scores within given distance threshold
#define CHCKDST_CHECK 1

// template arguments for secondary update of scores
#define SECONDARYUPDATE_NOUPDATE 0
#define SECONDARYUPDATE_UNCONDITIONAL 1
#define SECONDARYUPDATE_CONDITIONAL 2

// -------------------------------------------------------------------------
// GetPairScore: given squared normalizing distance d02, calculate score 
// for a pair of atoms the squared distance between which is dst
__DINLINE__
float GetPairScore(float d02, float dst)
{
    return scrfdividef(d02, d02 + dst);
}

#endif//__scoringbase_h__
