/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

// #include <math.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include <string>

#include "mybase.h"
#include "CLOptions.h"

using namespace CLOptions;

// =========================================================================
// MACROs
//
#define CLDEFINEASSIGNMENT( NAME, TYPE, DEFVALUE, COND, OPTNAME ) \
  TYPE CLOptions::val##NAME##_ = DEFVALUE; \
  void CLOptions::AssignCLOpt##NAME( TYPE newvalue ) { \
    TYPE value = newvalue; \
    if(!( COND )) \
        throw MYRUNTIME_ERROR(("Invalid command-line option " TOSTR(OPTNAME))); \
    val##NAME##_ = value; \
  }

// -------------------------------------------------------------------------
// command-line options for the main program
//
CLDEFINEASSIGNMENT( B_CACHE_ON, int, 0, value==0 || value==1, -c);
CLDEFINEASSIGNMENT( B_CLS_THRESHOLD, float, 0.5f, value>=0.0f && value<100.0f, --cls-threshold);
CLDEFINEASSIGNMENT( B_CLS_COVERAGE, float, 0.7f, value>0.0f && value<=1.0f, --cls-coverage);
CLDEFINEASSIGNMENT( B_CLS_ONE_SIDED_COVERAGE, int, 0, value==0 || value==1, --cls-one-sided-coverage);
CLDEFINEASSIGNMENT( B_CLS_OUT_SEQUENCES, int, 0, value==0 || value==1, --cls-out-sequences);
CLDEFINEASSIGNMENT( B_CLS_ALGORITHM, int, 0, value>=0 && value<bcnBClustering, --cls-algorithm);
CLDEFINEASSIGNMENT( O_S, float, 0.5f, value>=0.0f && value<1.0f, -s);
CLDEFINEASSIGNMENT( O_2TM_SCORE, int, 0, value==0 || value==1, --2tm-score);
CLDEFINEASSIGNMENT( O_SORT, int, 0, value>=0 && value<osnOSortingTotal, --sort);
CLDEFINEASSIGNMENT( O_NHITS, int, 2000, value>0 && value<1000000, --nhits);
CLDEFINEASSIGNMENT( O_NALNS, int, 2000, value>0 && value<1000000, --nalns);
CLDEFINEASSIGNMENT( O_WRAP, int, 80, value>=40 && value<99999999, --wrap);
CLDEFINEASSIGNMENT( O_NO_DELETIONS, int, 0, value==0 || value==1, --no-deletions);
CLDEFINEASSIGNMENT( O_REFERENCED, int, 0, value==0 || value==1, --referenced);
CLDEFINEASSIGNMENT( O_OUTFMT, int, 1, value>=0 && value<oofnOOutputFormat, --outfmt);
CLDEFINEASSIGNMENT( I_INFMT, int, 0, value>=0 && value<iifnIInputFormat, --infmt);
CLDEFINEASSIGNMENT( I_ATOM_PROT, std::string, " CA ", 1, --atom);
CLDEFINEASSIGNMENT( I_ATOM_RNA, std::string, " C3'", 1, --atom);
CLDEFINEASSIGNMENT( I_ATOM_PROT_trimmed, std::string, "CA", 1, nooption);
CLDEFINEASSIGNMENT( I_ATOM_RNA_trimmed, std::string, "C3'", 1, nooption);
CLDEFINEASSIGNMENT( I_HETATM, int, 0, value==0 || value==1, --hetatm);
CLDEFINEASSIGNMENT( I_TER, int, 3, value>=0 && value<istnIStructTerminator, --ter);
CLDEFINEASSIGNMENT( I_SPLIT, int, 0, value>=0 && value<issanIStructSplitApproach, --split);
CLDEFINEASSIGNMENT( I_SUPERP, int, 0, value>=0 && value<iaanIAlnAlgorithm, --superp);
CLDEFINEASSIGNMENT( P_PRE_SIMILARITY, float, 0.0f, value>=0.0f, --pre-pre-similarity);
CLDEFINEASSIGNMENT( P_PRE_SCORE, float, 0.3f, value>=0.0f && value<1.0f, --pre-score);
CLDEFINEASSIGNMENT( C_Il, std::string, "", 1, -i);
CLDEFINEASSIGNMENT( C_Iu, std::string, "", 1, -I);
CLDEFINEASSIGNMENT( C_D0, float, 0.0f, value>0.0f && value<50.0f, --d0);
CLDEFINEASSIGNMENT( C_U, int, 0, value>0, -u);
CLDEFINEASSIGNMENT( C_A, int, 0, value>=0 && value<csnnCScoreNormalization, -a);
CLDEFINEASSIGNMENT( C_SYMMETRIC, int, 0, value==0 || value==1, --symmetric);
CLDEFINEASSIGNMENT( C_REFINEMENT, int, csrOneSearch, value>=0 && value<csrnCSuperpRefinement, --refinement);
CLDEFINEASSIGNMENT( C_DEPTH, int, csdMedium, value>=0 && value<csdnCSuperpDepth, --depth);
CLDEFINEASSIGNMENT( C_TRIGGER, int, 50, value>=0 && value<=100, --trigger);
CLDEFINEASSIGNMENT( C_NBRANCHES, int, 5, value>=3 && value<=16, --nbranches);
CLDEFINEASSIGNMENT( C_ADDSEARCHBYSS, int, 0, value==0 || value==1, --add-search-by-ss);
CLDEFINEASSIGNMENT( C_NODETAILEDSEARCH, int, 0, value==0 || value==1, --no-detailed-search);
CLDEFINEASSIGNMENT( C_CONVERGENCE, int, 18, value>=1 && value<=30, --convergence);
CLDEFINEASSIGNMENT( C_SPEED, int, 8, value>=0 && value<=13, --speed);
CLDEFINEASSIGNMENT( C_CP, int, 0, value==0 || value==1, --cp);
CLDEFINEASSIGNMENT( C_MIRROR, int, 0, value==0 || value==1, --mirror);
//
CLDEFINEASSIGNMENT( CPU_THREADS_READING, int, 10, value>=1 && value<=64, --cpu-threads-reading)
CLDEFINEASSIGNMENT( CPU_THREADS, int, 1, value>=1 && value<=1024, --cpu-threads)
CLDEFINEASSIGNMENT( DEV_QRS_PER_CHUNK, int, 2, value>=1 && value<=100, --dev-queries-per-chunk)
CLDEFINEASSIGNMENT( DEV_QRES_PER_CHUNK, int, 4000, value>=100 && value<=50000, --dev-queries-total-length-per-chunk)
CLDEFINEASSIGNMENT( DEV_MAXRLEN, int, 4000, value>=100 && value<=65535, --dev-max-length)
CLDEFINEASSIGNMENT( DEV_MINRLEN, int, 20, value>=3 && value<=32767, --dev-min-length)
CLDEFINEASSIGNMENT( NOFILESORT, int, 0, value==0 || value==1, --no-file-sort);
//
CLDEFINEASSIGNMENT( DEV_N, std::string, "1", 1, --dev-N)
CLDEFINEASSIGNMENT( DEV_MEM, int, -1, value>=100 && value<=1000000, --dev-mem)
CLDEFINEASSIGNMENT( DEV_EXPCT_DBPROLEN, int, 50, value>=20 && value<=200, --dev-expected-length);
CLDEFINEASSIGNMENT( DEV_PASS2MEMP, int, 10, value>=1 && value<=100, --dev-pass2memp)
CLDEFINEASSIGNMENT( IO_NBUFFERS, int, 3, value>=2 && value<=6, --io-nbuffers);
CLDEFINEASSIGNMENT( IO_FILEMAP, int, 0, value==0 || value==1, --io-filemap);
CLDEFINEASSIGNMENT( IO_UNPINNED, int, 0, value==0 || value==1, --io-unpinned);

// command-line options for accompanying programs
//
