/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __alpha_h__
#define __alpha_h__

#include "platform.h"

#if defined (GPUINUSE) || defined(OS_MS_WINDOWS)
#define ALPHA_OMPDECLARE
#else 
#define ALPHA_OMPDECLARE _Pragma("omp declare simd notinbranch ")
#endif

// cardinality of the English alphabet
#define NEA 26

//{{SS states
enum SSSTATES {
    SS_C,
    SS_E,
    SS_H,
    SS_NSTATES
};
extern const char* gSSAlphabet;//{"CEH"}
extern const char* gSSlcAlphabet;//{" eh"}
//}}

//NOTE: map functions for 3-letter residue names!
char ResName2Code(const char*);
const char* ResCode2Name(char);

//for testing
void testmyresnamehash();

// GONNET residue scores
struct _GONNET_SCORES_
{
    _GONNET_SCORES_();
    float get(char res1, char res2);
    float data_[NEA * NEA];
};

extern _GONNET_SCORES_  GONNET_SCORES;

// -------------------------------------------------------------------------
// _GONNET_SCORES_::get: get a pairwise residue score obtained from the
// Gonnet frequencies
//
ALPHA_OMPDECLARE
inline
float _GONNET_SCORES_::get(char res1, char res2)
{
    if('A' <= res1 && res1 <= 'Z' && 'A' <= res2 && res2 <= 'Z')
        return data_[(res1 - 'A') * NEA + (res2 - 'A')];
    return 0.0f;
}

// -------------------------------------------------------------------------

#endif//__alpha_h__
