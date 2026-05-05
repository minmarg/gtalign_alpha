/***************************************************************************
 *   Copyright (C) 2021-2026 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __CLOptions__
#define __CLOptions__

#include <string>
#include "preproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"

//possible values for gap cost used to estimate local similarity
#define LOCALN_GAP_COST_0 (-0.8f)
#define LOCALN_GAP_COST_1 (-1.2f)
#define LOCALN_GAP_COST_2 (-3.0f)
#define LOCALN_GAP_COST_D LOCALN_GAP_COST_1 //default

//possible seed rule values for initializing superposition configurations
#define SEED_RULE_0 0
#define SEED_RULE_1 64
#define SEED_RULE_2 128
#define SEED_RULE_D SEED_RULE_1 //default
//seed rule for nucleic acids upon the local alignment rule:
#define SEED_RULE_NA 128

//min/max/default window size used to analyze candidate superpositions
#define MIN_WINDOW_SIZE 256
#define MAX_WINDOW_SIZE 512
#define DEF_WINDOW_SIZE MIN_WINDOW_SIZE

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

inline int GetNAAtomType(std::string snaatype)
{
    if(snaatype == "C3'") return gtnaatC3p;
    if(snaatype == "C4'") return gtnaatC4p;
    if(snaatype == "C5'") return gtnaatC5p;
    if(snaatype == "O3'") return gtnaatO3p;
    if(snaatype == "O5'") return gtnaatO5p;
    if(snaatype == "P") return gtnaatP;
    return gtnaatOther;
}

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
    osTMscoreHarmonic,
    osRMSD,
    osnOSorting,
    os2TMscoreGreater = osnOSorting,
    os2TMscoreReference,
    os2TMscoreQuery,
    os2TMscoreHarmonic,
    osnOSortingTotal
};
enum TOOutputFormat {
    oofPlain,
    oofJSON,
    oofnOOutputFormat
};
enum TIInputFormat {
    iifAuto,
    iifPDB,
    iifmmCIF,
    iifnIInputFormat
};
enum TIStructType {
    imtProtein,
    imtNucleicAcid,
    imtDetermined,
    imtnIStructType
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
    csdnCSuperpDepth,
    csdDepthDefault = csdMedium
};
enum TCSuperpGapCost {
    csgcGapCost0,
    csgcGapCost1,
    csgcGapCost2,
    csgcnCSuperpGapCost,
    csgcGapCostDefault = csgcGapCost1
};
enum TCSuperpDefaults {
    cstTriggerDefault = 50,
    csnNbranchesDefault = 5,
    csnNDPsDefault = 2
};
enum TCSuperpSeedRule {
    cssrContinuousFragments2,
    cssrLocalAlignment64,
    cssrLocalAlignment128,
    cssrnCSuperpSeedRule,
    cssrSuperpSeedRuleDefault = cssrLocalAlignment64
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
CLDECLAREOPTION( O_2TM_SCORE, int, int, );
CLDECLAREOPTION( O_SORT, int, int, );
CLDECLAREOPTION( O_NHITS, int, int, );
CLDECLAREOPTION( O_NALNS, int, int, );
CLDECLAREOPTION( O_WRAP, int, int, );
CLDECLAREOPTION( O_NO_DELETIONS, int, int, );
CLDECLAREOPTION( O_REFERENCED, int, int, );
CLDECLAREOPTION( O_OUTFMT, int, int, );
CLDECLAREOPTION( I_INFMT, int, int, );
CLDECLAREOPTION( I_AATOM, std::string, std::string, );
CLDECLAREOPTION( I_NATOM, std::string, std::string, );
CLDECLAREOPTION( I_AATOM_trimmed, std::string, std::string, );
CLDECLAREOPTION( I_NATOM_trimmed, std::string, std::string, );
CLDECLAREOPTION( I_NATOM_type, int, int, );
CLDECLAREOPTION( I_HETATM, int, int, );
CLDECLAREOPTION( I_MOL, int, int, );
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
CLDECLAREOPTION( C_GAPCOST, int, int, );
CLDECLAREOPTION( C_TRIGGER, int, int, );
CLDECLAREOPTION( C_SEEDRULE, int, int, );
CLDECLAREOPTION( C_WINDOW, int, int, );
CLDECLAREOPTION( C_NBRANCHES, int, int, );
CLDECLAREOPTION( C_ADDSEARCHBYSS, int, int, );
CLDECLAREOPTION( C_NODETAILEDSEARCH, int, int, );
CLDECLAREOPTION( C_CONVERGENCE, int, int, );
CLDECLAREOPTION( C_SPEED, int, int, );
CLDECLAREOPTION( C_CP, int, int, );
CLDECLAREOPTION( C_MIRROR, int, int, );
//Hidden (if any):
CLDECLAREOPTION( H_N_SPATIAL_ITERATIONS, int, int, );
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

//accompanying program options
inline float GetC_GapCost()
{
    switch(GetC_GAPCOST()) {
        case csgcGapCost0: return LOCALN_GAP_COST_0;
        case csgcGapCost1: return LOCALN_GAP_COST_1;
        case csgcGapCost2: return LOCALN_GAP_COST_2;
    }
    return LOCALN_GAP_COST_D;
}
inline int GetC_SeedRuleValue()
{
    switch(GetC_SEEDRULE()) {
        case cssrContinuousFragments2: return SEED_RULE_0;
        case cssrLocalAlignment64: return SEED_RULE_1;
        case cssrLocalAlignment128: return SEED_RULE_2;
    }
    return SEED_RULE_D;
}

}//namespace CLOptions

#endif//__CLOptions__
