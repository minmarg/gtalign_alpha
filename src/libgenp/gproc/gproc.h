/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __gproc_h__
#define __gproc_h__

// Distance of equivalence for aligned residues
#define EQUIVALENCE_DISTANCE (5.0f)

// === Transformation matrix section =======================================

// indices for accessing transformation matrix
enum TTranformMatrix {
    tfmmRot_0_0,//rotation matrix...
    tfmmRot_0_1,
    tfmmRot_0_2,
    tfmmRot_1_0,//2nd row
    tfmmRot_1_1,
    tfmmRot_1_2,
    tfmmRot_2_0,//3rd row
    tfmmRot_2_1,
    tfmmRot_2_2,
    tfmmTrl_0,//translation vector...
    tfmmTrl_1,
    tfmmTrl_2,
    nTTranformMatrix
};

// === End of transformation matrix section ================================

// === Auxiliary working memory section ====================================

// NOTE: convergence values expressed in powers of 2 SO THAT the SUM and 
// NOTE: OR of different values is distinct:
// bit value reserved for convergence adjusting transformation on the same 
// fragment:
#define CONVERGED_FRAGREF_bitval 1
#define CONVERGED_FRAGREF INT2FLOAT(CONVERGED_FRAGREF_bitval)
// bit value reserved for score convergence during DP application:
#define CONVERGED_SCOREDP_bitval 2
#define CONVERGED_SCOREDP INT2FLOAT(CONVERGED_SCOREDP_bitval)
// no tmscore progress: bit value instructs to skip the immediate stage:
#define CONVERGED_NOTMPRG_bitval 4
#define CONVERGED_NOTMPRG INT2FLOAT(CONVERGED_NOTMPRG_bitval)
// bit value reserved for termination due to expected low tmscore:
#define CONVERGED_LOWTMSC_bitval 8
#define CONVERGED_LOWTMSC INT2FLOAT(CONVERGED_LOWTMSC_bitval)

// indices for accessing working memory;
// all variables/fields are placed along structures and used for 
// structure-specific processing; i.e., each field represents an 
// array continuous along structures;
// NOTE: keep #entries < twmvEndOfCCData
enum TAuxWorkingMemoryVars {
    tawmvGrandBest,//total best score
    tawmvBestScore,//latest best score
    tawmvQRYpos,//start alignment position for query
    tawmvRFNpos,//start alignment position for reference
    tawmvLastD02s,//last d0 squared used for the inclusion of pairs in the alignment (search)
    tawmvScore,//partial score (computed for a fragment of alignment)
    tawmvBest0,//previous best score saved
    tawmvConverged,//flag of convergence of finding an optimal rotation matrix
    tawmvInitialBest,//unrefined initial best score
    tawmvSubFragNdx,//index defining fragment length: for determining the anchor in DP matrices
    tawmvSubFragPos,//starting position within fragment; for the anchor in DP matrices
    tawmvSubFragNdxCurrent,//current value of index defining fragment length
    tawmvSubFragPosCurrent,//current starting position within fragment
    tawmvNAlnPoss,//number (length) of aligned pairs obtained during DP refinement
    tawmvEndOfCCMDat,//end of the section of data associated with CC matrix calculation
    nTAuxWorkingMemoryVars=tawmvEndOfCCMDat
};

// === End of auxiliary working memory section =============================

// === Working memory section ==============================================

// indices for accessing working memory;
// all variables are placed sequentially for the same structure and 
// used for position-specific processing
enum TWorkingMemoryVars {
    twmvCCM_0_0,//cross-covariance matrix (rotation matrix on output)...
    twmvCCM_0_1,
    twmvCCM_0_2,
    twmvCCM_1_0,//2nd row
    twmvCCM_1_1,
    twmvCCM_1_2,
    twmvCCM_2_0,//3rd row
    twmvCCM_2_1,
    twmvCCM_2_2,
    twmvEndOfCCMtx,//end of matrix
    twmvCVq_0=twmvEndOfCCMtx,//center vector x (query; translation vector on output)...
    twmvCVq_1,
    twmvCVq_2,
    twmvEndOfTFMData,//end of transformation matrix data on output
    twmvCVr_0=twmvEndOfTFMData,//center vector y (reference)...
    twmvCVr_1,
    twmvCVr_2,
    twmvEndOfCCData,
    twmvNalnposs=twmvEndOfCCData,//#alignment positions giving rise to CCM data
    twmvEndOfCCDataExt,
    nTWorkingMemoryVars=twmvEndOfCCDataExt,
    //the below data do not occupy additional gmem and are stored in cache:
    twmvCV2q_0=nTWorkingMemoryVars,//vector of squares for x (query)...
    twmvCV2q_1,
    twmvCV2q_2,
    twmvCV2r_0,//vector of squares y (reference)...
    twmvCV2r_1,
    twmvCV2r_2,
    twmvEndOfCCDataExtPlus
};

// === End of working memory section =======================================

// === Working memory section 2 ============================================

// indices for accessing working memory;
// each variable written sequentially for each structure in the chunk and 
// used for structure-specific processing;
// NOTE: one-to-one correspondence between this and TWorkingMemoryVars 
// structure is NECESSARY!
enum TWorkingMemory2Vars {
    twm2CCM_0_0,//cross-covariance matrix...
    twm2CCM_0_1,
    twm2CCM_0_2,
    twm2CCM_1_0,//2nd row
    twm2CCM_1_1,
    twm2CCM_1_2,
    twm2CCM_2_0,//3rd row
    twm2CCM_2_1,
    twm2CCM_2_2,
    twm2CVq_0,//center vector x (query)...
    twm2CVq_1,
    twm2CVq_2,
    twm2CVr_0,//center vector y (reference)...
    twm2CVr_1,
    twm2CVr_2,
    twm2EndOfCCData,
    twm2Nalnposs=twm2EndOfCCData,//#alignment positions giving rise to CCM data
    twm2EndOfCCDataExt,
    nTWorkingMemory2Vars=twm2EndOfCCDataExt
};

// === End of working memory section 2 =====================================



// --- enums ---------------------------------------------------------------

enum TDPDiagScoreSections {
    //DP scores
    dpdssDiag1,//buffer of the first diagonal
    dpdssDiag2,//buffer of the second diagonal
    ///NOTE: not in use:
    ///dpdssDiagM,//buffer of the maximum scores found up to the second diagonal
    nTDPDiagScoreSections
};

enum TDPBottomScoreSections {
    //DP scores
    dpbssBottm,//buffer of the bottom DP scores
    nTDPBottomScoreSections
};

enum TDPDiagScoreSubsections {
    dpdsssStateMM,//match-to-match state
    nTDPDiagScoreSubsections
};

//subsections for affine gap cost model
enum TDPAGCDiagScoreSubsections {
    dpagcdsssStateMM,//match-to-match state
    dpagcdsssStateMG,//emission-to-gap
    dpagcdsssStateGM,//gap-to-emission
    nTDPAGCDiagScoreSubsections
};

//backtracking direction constants
enum TDPBtckDirection {
    dpbtckSTOP = 0,
    dpbtckLEFT = 1,//left, gap in query
    dpbtckUP = 2,//up, gap in db structure
    dpbtckDIAG = 3,//diagonal, match between query and db target
};

//sections for positions temporarily aligned by DP;
//positions extend over db reference structure axis:
// i.e., db reference structure positions
enum TDPAlignedPoss {
    dpapsQRYx,//query x-coordinate
    dpapsQRYy,//query y-coordinate
    dpapsQRYz,//query z-coordinate
    dpapsRFNx,//reference x-coordinate
    dpapsRFNy,//reference y-coordinate
    dpapsRFNz,//reference z-coordinate
    nTDPAlignedPoss
};

// === End of DP section ===================================================



// === Alignment statistics section ========================================

// alignment statistics calculated to be transferred to host
enum TDP2OutputAlnData {
    dp2oadOrgStrNo,//original structure serial number
    dp2oadStrNewDst,//structure distance in the new buffer (after a series of iterations)
    nTDP2OutputAlnDataPart1Beg,
    dp2oadBegCoords = nTDP2OutputAlnDataPart1Beg,//alignment beginning coordinates
    dp2oadEndCoords,//alignment end coordinates
    dp2oadAlnLength,//alignment length
    dp2oadPstvs,//number of matched positions in the alignment
    dp2oadIdnts,//number of identical residues in the alignment
    dp2oadNGaps,//number of gaps in the alignment
    nTDP2OutputAlnDataPart1End,
    dp2oadRMSD = nTDP2OutputAlnDataPart1End,//RMSD
    dp2oadScoreQ,//TM-score normalized by the query length
    dp2oadScoreR,//TM-score normalized by the reference length
    dp2oadD0Q,//TM-score normalized by the specified length
    dp2oadD0R,//TM-score obtained using the specified d0 in normalization
    nTDP2OutputAlnDataPart2,
    dp2oad2ScoreQ = nTDP2OutputAlnDataPart2,//2TM-score normalized by the query length
    dp2oad2ScoreR,//2TM-score normalized by the reference length
    nTDP2OutputAlnData
};

// subsections for the alignment itself
enum TDP2OutputAlignment {
    dp2oaQuery,//query alignment sequence
    dp2oaMiddle,//middle (information) sequence of the alignment 
    dp2oaTarget,//reference alignment sequence
    nTDP2OutputAlignment,
    dp2oaQuerySSS=nTDP2OutputAlignment,//query sec. str. sequence
    dp2oaTargetSSS,//reference SS sequence
    nTDP2OutputAlignmentSSS
};

// === End of alignment statistics section =================================



// === Global variable section =============================================

enum TFilterData {
    fdNewReferenceIndex,//new reference structure index (inclusive /prefix sum/)
    fdNewReferenceAddress,///new reference address (inclusive /prefix sum/)
    nTFilterData
};

enum TDevGlobVariables {
    dgvNPassedStrs,//number of structures passed to the next stage
    dgvNPosits,//total number of positions (length) of passed structures
    dgvMaxStrLen,//maximum structure length over passed structure
    dgvMutex,//mutex for registering structures passing to the next stage
    nDevGlobVariables
};

// === End of Global variable section ======================================

#endif//__gproc_h__
