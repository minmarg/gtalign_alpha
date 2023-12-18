/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __CLOptions__
#define __CLOptions__

#include <string>
#include "preproc.h"

// the OPER argument in the following macro represents operator along with 
// operand, e.g., `*0.01'.
#define DECLAREOPTION( NAME, TYPE, TYPECAST, OPER ) \
  extern TYPE val##NAME##_; \
  static inline TYPECAST Get##NAME() { return (TYPECAST)val##NAME##_ OPER; } \
  ; // void Read##NAME();

#define CLDECLAREOPTION( NAME, TYPE, TYPECAST, OPER ) \
    DECLAREOPTION( NAME, TYPE, TYPECAST, OPER ); \
    void AssignCLOpt##NAME( TYPE );

#define CLOPTASSIGN( NAME, VALUE ) \
    CLOptions::AssignCLOpt##NAME( VALUE );

//command-line options
namespace CLOptions {
enum TBClustering {
    bcCompleteLinkage,
    bcSingleLinkage,
    bcnBClustering
};
enum TOSorting {
    osTMscoreGreater,
    osTMscoreReference,
    osTMscoreQuery,
    osRMSD,
    osnOSorting
};
enum TOOutputFormat {
    oofFull,
    oofAlignment,
    oofTabular,
    oofnOOutputFormat
};
enum TIInputFormat {
    iifAuto,
    iifPDB,
    iifmmCIF,
    iifnIInputFormat
};
enum TIStructTerminator {
    istEOF,
    istEND,
    istENDorChain,
    istTER_ENDorChain,
    istnIStructTerminator
};
enum TIStructSplitApproach {
    issaNoSplit,
    issaByMODEL,
    issaByChain,
    issanIStructSplitApproach
};
enum TIAlnAlgorithm {
    iaaSeqIndependent,
    iaaSeqDependentByRes,
    iaaSeqDependentByResAndChain,
    iaaSeqDependentByResAndChainOrder,
    iaanIAlnAlgorithm
};
enum TCScoreNormalization {
    csnRefLength,
    csnAvgLength,
    csnShorterLength,
    csnLongerLength,
    csnnCScoreNormalization
};
enum TCSuperpRefinement {
    csrLogSearch,
    csrCoarseSearch,
    csrOneSearch,
    csrFullSearch,
    csrFullASearch,
    csrnCSuperpRefinement
};
enum TCSuperpDepth {
    csdDeep,
    csdHigh,
    csdMedium,
    csdShallow,
    csdnCSuperpDepth
};
//program options;
//B_, O_, I_, C_, DEV_ for 
// basic, output, interpretation, computation, and device options
CLDECLAREOPTION( B_CACHE_ON, int, int, );
CLDECLAREOPTION( B_CLS_THRESHOLD, float, float, );
CLDECLAREOPTION( B_CLS_COVERAGE, float, float, );
CLDECLAREOPTION( B_CLS_ONE_SIDED_COVERAGE, int, int, );
CLDECLAREOPTION( B_CLS_OUT_SEQUENCES, int, int, );
CLDECLAREOPTION( B_CLS_ALGORITHM, int, int, );
CLDECLAREOPTION( O_S, float, float, );
CLDECLAREOPTION( O_SORT, int, int, );
CLDECLAREOPTION( O_NHITS, int, int, );
CLDECLAREOPTION( O_NALNS, int, int, );
CLDECLAREOPTION( O_WRAP, int, int, );
CLDECLAREOPTION( O_NO_DELETIONS, int, int, );
CLDECLAREOPTION( O_REFERENCED, int, int, );
CLDECLAREOPTION( O_OUTFMT, int, int, );
CLDECLAREOPTION( I_INFMT, int, int, );
CLDECLAREOPTION( I_ATOM_PROT, std::string, std::string, );
CLDECLAREOPTION( I_ATOM_RNA, std::string, std::string, );
CLDECLAREOPTION( I_ATOM_PROT_trimmed, std::string, std::string, );
CLDECLAREOPTION( I_ATOM_RNA_trimmed, std::string, std::string, );
CLDECLAREOPTION( I_HETATM, int, int, );
CLDECLAREOPTION( I_TER, int, int, );
CLDECLAREOPTION( I_SPLIT, int, int, );
CLDECLAREOPTION( I_SUPERP, int, int, );
CLDECLAREOPTION( P_PRE_SIMILARITY, float, float, );
CLDECLAREOPTION( P_PRE_SCORE, float, float, );
CLDECLAREOPTION( C_Il, std::string, std::string, );
CLDECLAREOPTION( C_Iu, std::string, std::string, );
CLDECLAREOPTION( C_D0, float, float, );
CLDECLAREOPTION( C_U, int, int, );
CLDECLAREOPTION( C_A, int, int, );
CLDECLAREOPTION( C_SYMMETRIC, int, int, );
CLDECLAREOPTION( C_REFINEMENT, int, int, );
CLDECLAREOPTION( C_DEPTH, int, int, );
CLDECLAREOPTION( C_TRIGGER, int, int, );
CLDECLAREOPTION( C_NBRANCHES, int, int, );
CLDECLAREOPTION( C_ADDSEARCHBYSS, int, int, );
CLDECLAREOPTION( C_NODETAILEDSEARCH, int, int, );
CLDECLAREOPTION( C_CONVERGENCE, int, int, );
CLDECLAREOPTION( C_SPEED, int, int, );
CLDECLAREOPTION( C_CP, int, int, );
CLDECLAREOPTION( C_MIRROR, int, int, );
//
CLDECLAREOPTION( CPU_THREADS_READING, int, int, );
CLDECLAREOPTION( CPU_THREADS, int, int, );
CLDECLAREOPTION( DEV_QRS_PER_CHUNK, int, int, );
CLDECLAREOPTION( DEV_QRES_PER_CHUNK, int, int, );
CLDECLAREOPTION( DEV_MAXRLEN, int, int, );
CLDECLAREOPTION( DEV_MINRLEN, int, int, );
CLDECLAREOPTION( NOFILESORT, int, int, );
//
CLDECLAREOPTION( DEV_N, std::string, std::string, );
CLDECLAREOPTION( DEV_MEM, int, int, );
CLDECLAREOPTION( DEV_EXPCT_DBPROLEN, int, int, );
CLDECLAREOPTION( DEV_PASS2MEMP, int, float, *0.01f );
CLDECLAREOPTION( IO_NBUFFERS, int, int, );
CLDECLAREOPTION( IO_FILEMAP, int, int, );
CLDECLAREOPTION( IO_UNPINNED, int, int, );

//accompanying programs' options
}//namespace CLOptions

#endif//__CLOptions__
