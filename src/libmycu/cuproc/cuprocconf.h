/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __cuprocconf_h__
#define __cuprocconf_h__

#include "libmymp/mpproc/mpprocconfbase.h"

// === Secondary structure section =========================================

// x-dimension for calculating secondary structure
// NOTE: unrolling factor is 1: single relatively non-intense calculation
#define CUSS_CALCSTR_XDIM 128

// === End of secondary structure section ==================================



// === Filter section ======================================================
// x-dimension for calculating maximum coverage:
#define CUFL_TBSP_SEQUENCE_COVERAGE_XDIM 128

// x-dimension for calculating sequence similarity, unprocessed tail 
// length for query and reference sequences, and diagonal step:
//(NOTE: XDIM should be 32 as to the kernel requirement.)
#define CUFL_TBSP_SEQUENCE_SIMILARITY_XDIM 32
#define CUFL_TBSP_SEQUENCE_SIMILARITY_EDGE 20
#define CUFL_TBSP_SEQUENCE_SIMILARITY_STEP 2
#define CUFL_TBSP_SEQUENCE_SIMILARITY_STEPLOG2 1

// x,y-dimensions for making list of database candidates progressing to the
// next stages:
#define CUFL_MAKECANDIDATELIST_XDIM 512
#define CUFL_MAKECANDIDATELIST_YDIM 2

// x-dimension for copying selected database candidates from and to the
// original location, respectively:
#define CUFL_STORECANDIDATEDATA_XDIM 32

#define CUFL_LOADCANDIDATEDATA_XDIM 32

// === End of filter section ===============================================



// === Scores section ======================================================

#define CUBSM_TYPE  float
#define CUBSM_Q(CNST) (CNST ## . ## f)

// STAGE 1

// x-dimension of thread block for calculating complete alignment refinement, 
// used as a solution to relatively large kernel submission latency:
#define CUS1_TBINITSP_COMPLETEREFINE_XDIM 32 //64 //128
// thread block x-dimension x for calculating secondary TMscores;
#define CUDP_PRODUCTION_2TMSCORE_DIM_X 128


// x-dimension of thread block for initializing transformation matrices:
#define CUS1_TBINITSP_TFMINIT_XDIM 64
// unrolling factor (#structures) for initializing transformation matrices:
#define CUS1_TBINITSP_TFMINIT_XFCT 5

// x-dimension of thread block for initializing cross-covariance data:
#define CUS1_TBINITSP_CCDINIT_XDIM 64
// unrolling factor (#structures) for initializing cross-covariance data:
#define CUS1_TBINITSP_CCDINIT_XFCT 4

// x-dimension of thread block for calculating cross-covariance matrices:
// NOTE: do not use larger than 64: occupancy may be dramatically reduced
#define CUS1_TBINITSP_CCMCALC_XDIM 64
// unrolling factor for calculating cross-covariance matrices: 2, 4, 8
#define CUS1_TBINITSP_CCMCALC_XFCT 64 //16 //4
// logical x dimension for calculating cross-covariance matrices
#define CUS1_TBINITSP_CCMCALC_XDIMLGL (CUS1_TBINITSP_CCMCALC_XDIM * CUS1_TBINITSP_CCMCALC_XFCT)

// x-dimension of thread block for calculating cross-covariance matrices 
// iteratively:
#define CUS1_TBINITSP_CCMCALC_ITRD_XDIM 1024

// x-dimension (>=64) of thread block for calculating distance thresholds for the 
// inclusion of positions the refined rotaion matrices will be based on;
// NOTE: dimension of 1024 greatly reduces #active thread blocks and leads to 
// NOTE: several times LOWER kernel performance on average;
// NOTE: set to nearly average protein size:
#define CUS1_TBINITSP_FINDD02_ITRD_XDIM 256
// // perform the calculation of distance thresholds during fragment boundary 
// // refinment in iterations applied to convergence
// #define DO_FINDD02_DURING_REFINEFRAG 0
// //maximum number of subdivisions of identified fragments during refinement
// #define FRAGREF_NMAXSUBFRAGS 6
// //step size with which fragments are traversed during refinement
// #define FRAGREF_SFRAGSTEP 40
// #define FRAGREF_SFRAGSTEP_mini 1
// //maximum number of iterations until convergence
// #define FRAGREF_NMAXCONVIT 20
// #define FRAGREF_NMAXCONVIT_mini 4

// number of structures to be copied by thread block (shared memory dimension)
#define CUS1_TBINITSP_CCMCOPY_N 64

// unrolling factor for computing transformation matrices:
// number of query-reference structure pairs processed in 
// parallel by one block
#define CUS1_TBSP_TFM_N 32

// x-dimension of thread block for calculating scores (>=64):
#define CUS1_TBSP_SCORE_XDIM 128 //64
// unrolling factor for calculating scores: 2, 4, 8
#define CUS1_TBSP_SCORE_XFCT 64 //16 //4
// logical x dimension for calculating scores
#define CUS1_TBSP_SCORE_XDIMLGL (CUS1_TBSP_SCORE_XDIM * CUS1_TBSP_SCORE_XFCT)
// thread block x-dimension for calculating scores when checking for halt:
#define CUS1_TBSP_SCORE_FRG2_HALT_CHK_XDIM 32 //64

// x-dimension of thread block for setting calculated score 
// (corresponds to #structures):
#define CUS1_TBSP_SCORE_SET_XDIM 512 //256

// x-dimension of thread block for checking convergence:
#define CUS1_TBINITSP_CCDCONV_XDIM 256 //64
// unrolling factor (#structures) for checking convergence:
#define CUS1_TBINITSP_CCDCONV_XFCT 16 //4

// x-dimension of thread block for saving best-performing transformation matrices:
#define CUS1_TBINITSP_TMSAVE_XDIM 256 //128
// unrolling factor for saving best-performing transformation matrices:
#define CUS1_TBINITSP_TMSAVE_XFCT 20 //10

// thread block dimensions for finding the maximum among calculated scores 
// over all fragment factors (x-dim corresponds to #structures):
#define CUS1_TBSP_SCORE_MAX_XDIM 32
#define CUS1_TBSP_SCORE_MAX_YDIM 32

// thread block dimensions for finding the maximum among calculated DP scores 
// over all fragment factors (x-dim corresponds to #structures):
#define CUS1_TBSP_DPSCORE_MAX_XDIM 32
#define CUS1_TBSP_DPSCORE_MAX_YDIM 32
// // consideration of top N DP scores calculated over all fragment factors for 
// // further processing (should be <= CUS1_TBSP_DPSCORE_MAX_YDIM (32, warpsize) 
// // and expressed as a power of 2):
// #define CUS1_TBSP_DPSCORE_TOP_N 32 //16
// // #max sections over which top N DP scores are calculated; hence, total max 
// // number of DP scores to calculate is 
// // CUS1_TBSP_DPSCORE_TOP_N * CUS1_TBSP_DPSCORE_TOP_N_MAX_CONFIGS
// #define CUS1_TBSP_DPSCORE_TOP_N_MAX_CONFIGS 3
// // MAX number of best-scoring TFM configurations selected from top N DP scores for 
// // further refinement 
// // (obviously, CUS1_TBSP_DPSCORE_TOP_N_REFINEMENT<=CUS1_TBSP_DPSCORE_TOP_N)
// #define CUS1_TBSP_DPSCORE_TOP_N_REFINEMENT 16
// // total MAX number of confihurations to verify alternatively
// #define CUS1_TBSP_DPSCORE_TOP_N_REFINEMENTxMAX_CONFIGS 
//     ((CUS1_TBSP_DPSCORE_TOP_N_MAX_CONFIGS) * (CUS1_TBSP_DPSCORE_TOP_N_REFINEMENT))

// // maximum number of iterations for applying DP to the refinement of alignment 
// // identified during fragment boundary refinement
// #define DPREFINE_MAXNITS 18 //30
// #define DPREFINE_MAXNITS_FAST 2


// STAGE FRG (index-based scoring from fragment superpositions)

// x-dimension of thread block for completely calculating fragment-based 
// superposition scores using spatial index:
// NOTE: currently should be 32
#define CUSF_TBSP_COMPLETE_INDEX_SCORE_XDIM 32


// x-dimension of thread block for calculating scores (>=64):
#define CUSF_TBSP_INDEX_SCORE_XDIM 256 //512
// unrolling factor for calculating scores: 2, 4, 8
#define CUSF_TBSP_INDEX_SCORE_XFCT 1
// logical x dimension for calculating scores
#define CUSF_TBSP_INDEX_SCORE_XDIMLGL (CUSF_TBSP_INDEX_SCORE_XDIM * CUSF_TBSP_INDEX_SCORE_XFCT)
// // max number of positions away from the beginning of a fragment on both sides to consider
// #define CUSF_TBSP_INDEX_SCORE_POSLIMIT 256 /*!*/ //512 //256
// #define CUSF_TBSP_INDEX_SCORE_POSLIMIT2 (2 * CUSF_TBSP_INDEX_SCORE_POSLIMIT)
// // max index depth in searching for the nearest neighbour atom
// #define CUSF_TBSP_INDEX_SCORE_MAXDEPTH 64
// // use secondary structure filtering when searching for the nearest neighbour atom
// #define CUSF_TBSP_INDEX_SCORE_SECSTRFILT 0

// x-dimension of thread block for calculating provisional local similarity
// (should be 32 for warp-scale sync is used in the kernel; previous version):
#define CUSF_TBSP_LOCAL_SIMILARITY_XDIM 32
#define CUSF_TBSP_LOCAL_SIMILARITY_YDIM 32
// x-dimension of thread block for reducing scores obtained from index (>=64):
#define CUSF_TBSP_INDEX_SCORE_REDUCE_XDIM 128 //64
// unrolling factor for reducing scores: 2, 4, 8
#define CUSF_TBSP_INDEX_SCORE_REDUCE_XFCT 64 //16 //4
// logical x dimension for calculating scores
#define CUSF_TBSP_INDEX_SCORE_REDUCE_XDIMLGL (CUSF_TBSP_INDEX_SCORE_REDUCE_XDIM * CUSF_TBSP_INDEX_SCORE_REDUCE_XFCT)

// x-dimension of thread block for saving the config of best-performing 
// transformation matrices:
#define CUSF_TBSP_INDEX_SCORE_SAVECFG_XDIM 256 //128

// thread block dimensions for finding the maximum among calculated scores 
// over all fragment factors (x-dim corresponds to #structures):
#define CUSF_TBSP_INDEX_SCORE_MAX_XDIM 32
#define CUSF_TBSP_INDEX_SCORE_MAX_YDIM 32

// dimension x of thread block for identifying matched positions (coordinates);
#define CUSF2_TBSP_INDEX_ALIGNMENT_XDIM 128 //64
// dimension y of thread block for matched positions (must be 6);
#define CUSF2_TBSP_INDEX_ALIGNMENT_YDIM 6
// unrolling factor for matched positions: 2, 4, 8
#define CUSF2_TBSP_INDEX_ALIGNMENT_XFCT 64 //16 //4
// logical x dimension for matched positions
#define CUSF2_TBSP_INDEX_ALIGNMENT_XDIMLGL (CUSF2_TBSP_INDEX_ALIGNMENT_XDIM * CUSF2_TBSP_INDEX_ALIGNMENT_XFCT)

// cache size for query positions when calculating scores 
// progressively from alignment (unordered in position) obtained by 
// applying a linear alignment algorithm
#define CUSF2_TBSP_INDEX_SCORE_QNX_CACHE_SIZE 512


// FINAL STAGE (final alignment refinement)

// // max number of positions around an identified position to consider in the
// // final refinement stage
// #define CUSFN_TBSP_FIN_REFINEMENT_MAX_NPOSITIONS 128 //512



// === End of scores section ===============================================



// === SPECTRAL ANALYSIS section ===========================================

// max number of distances analyzed wrt an atom under consideration in 
// algorithm 32 (this number does not include cardinalities written first)
#define CUSA32_SPECAN_MAX_NDSTS 32
// max number of secondary structure states wrt which different distance 
// distributions are calculated in algorithm 32
#define CUSA32_SPECAN_MAX_NSSSTATES 3
enum {
    //section for distances: each section consists of #distances, distances,
    //the norm of distances, and positions:
    C3MAXNDSTS32 = CUSA32_SPECAN_MAX_NDSTS,
    //#values including 2 #values for positive and negative ranges:
    C3MAXNVALS32 = 1 + 1 + C3MAXNDSTS32,
    C3MAXNCMPL32 = C3MAXNVALS32 + C3MAXNDSTS32,//complete #values
    //-------
    //positions (seq separations) start address for gmem:
    C3POSSTART32 = C3MAXNVALS32 * CUSA32_SPECAN_MAX_NSSSTATES,
};
// min and max sequence separation thresholds used in algorithm 32:
#define CUSA32_SPECAN_MIN_SEQSEP 6
#define CUSA32_SPECAN_MAX_SEQSEP 127

// execution configuration for sorting distances in spatial domain: 
// x,y dimensions of thread block:
#define CUSA32_SPECAN_SRTSD_XDIM 32
#define CUSA32_SPECAN_SRTSD_YDIM 16
// sort distances separately for each ss section; if 0, thread blocks 
// perform sorting in all sections
#define CUSA32_SPECAN_SRTSD_BYSECTION 0

// flag of centering PRF scores 
#define CUSA32_SPECAN_PRFSCORES_CENTERED 1

// execution configuration for spectral scores calculations: 
// x,y dimensions of thread block:
#define CUSA32_SPECAN_PRFSCORES_XDIM 32
#define CUSA32_SPECAN_PRFSCORES_YDIM 32
// unrolling factor (for #positions) along one (x) of the dimensions of the
// 2D global score matrix:
//NOTE: should not be greater than 4 because of high shared memory demand
#define CUSA32_SPECAN_PRFSCORES_XFCT 1
// score per aligned pair of distance values to be added to a 
// pairwise score for preventing from HSPs from short stretches of values:
#define CUSA32_SPECAN_PRFSCORES_DELTA 0.1f


// max number of distances analyzed wrt an atom under consideration in 
// algorithm 3 (this number does not include its own value written first)
#define CUSA3_SPECAN_MAX_NDSTS 19
// max number of secondary structure states wrt which different distance 
// distributions are calculated in algorithm 3
#define CUSA3_SPECAN_MAX_NSSSTATES 3
enum {
    //section for distances: each section consists of #distances, distances,
    //the norm of distances, and positions:
    C3MAXNDSTS = CUSA3_SPECAN_MAX_NDSTS,
    C3MAXNVALS = 1 + C3MAXNDSTS + 1,//#values without the norm
    C3MAXNCMPL = C3MAXNVALS + C3MAXNDSTS,//complete #values
    //-------
    //positions start address for gmem:
    C3POSSTART = C3MAXNVALS * CUSA3_SPECAN_MAX_NSSSTATES,
};

// execution configuration for sorting distances in spatial domain: 
// x,y dimensions of thread block:
#define CUSA3_SPECAN_SRTSD_XDIM 32
#define CUSA3_SPECAN_SRTSD_YDIM 16
// sort distances separately for each ss section; if 0, thread blocks 
// perform sorting in all sections
#define CUSA3_SPECAN_SRTSD_BYSECTION 0

// flag of centering PRF scores 
#define CUSA3_SPECAN_PRFSCORES_CENTERED 0

// execution configuration for spectral scores calculations: 
// x,y dimensions of thread block:
#define CUSA3_SPECAN_PRFSCORES_XDIM 32
#define CUSA3_SPECAN_PRFSCORES_YDIM 32
// unrolling factor (for #positions) along one (x) of the dimensions of the
// 2D global score matrix:
//NOTE: should not be greater than 4 because of high shared memory demand
#define CUSA3_SPECAN_PRFSCORES_XFCT 1


// max number of distances (atom coordinates) analyzed wrt an atom under 
// consideration (this number does not include its own value written first) 
#define CUSA2_SPECAN_MAX_NDSTS 20
// threshold (in A squared) for calculating spatial distance distribution scores
#define CUSA2_DP_SCORES_SPT_THRLD 4.0f //2*2


// execution configuration for spatial domain calculations: 
// x,y dimensions of thread block:
#define CUSA_SPECAN_SPATIAL_XDIM 32
#define CUSA_SPECAN_SPATIAL_YDIM CUSA_SPECAN_SPATIAL_XDIM
// unrolling factor (for #positions) along one (y) of the dimensions of the
// 2D distance map:
#define CUSA_SPECAN_SPATIAL_XFCT 4
// forward and backward neighbors (>=1) to analyze for axis rearrangement:
#define CUSA_SPECAN_SPATIAL_FWD_NGBRS 6
#define CUSA_SPECAN_SPATIAL_BWD_NGBRS 6

// execution configuration for frequency domain calculations: 
// x,y dimensions of thread block:
#define CUSA_SPECAN_FREQD_XDIM 32
#define CUSA_SPECAN_FREQD_YDIM 32

// execution configuration for spectral scores calculations: 
// x,y dimensions of thread block:
#define CUSA_SPECAN_PSDSCORES_XDIM 32
#define CUSA_SPECAN_PSDSCORES_YDIM 32
// unrolling factor (for #positions) along one (x) of the dimensions of the
// 2D global score matrix:
//NOTE: should not be greater than 4 because of high shared memory demand
#define CUSA_SPECAN_PSDSCORES_XFCT 2
#define CUSA2_SPECAN_DDSCORES_XFCT 2
// flag for centering PSD scores 
#define CUSA_SPECAN_PSDSCORES_CENTERED 1

//{{ SPECTRAL ANALYSIS:
// number of sampling points (>=4!!) for the theta and phi
// angles in a spherical coordinate system (NOTE: power of 2 for both);
#define SPECAN_THETA_NSMPL_POINTS 8
#define SPECAN_PHI_NSMPL_POINTS 16 //8
#define SPECAN_LOG2_THETA_NSMPL_POINTS 3
#define SPECAN_LOG2_PHI_NSMPL_POINTS 4 //3
// total number of sampling points over two dimensions
#define SPECAN_TOT_NSMPL_POINTS (SPECAN_THETA_NSMPL_POINTS * SPECAN_PHI_NSMPL_POINTS)
#define SPECAN_LOG2_TOT_NSMPL_POINTS (SPECAN_LOG2_THETA_NSMPL_POINTS + SPECAN_LOG2_PHI_NSMPL_POINTS)
#define SPECAN_PSD_NVALS (SPECAN_THETA_NSMPL_POINTS * ((SPECAN_PHI_NSMPL_POINTS>>1) + 1))

// max number of secondary structure states wrt which different distance 
// distributions are calculated in algorithm 7
#define CUSA7_SPECAN_MAX_NSSSTATES 3
// calculations can be performed in frequency domain (=1) or, 
// alternatively, in spatial domain (=0)
#define CUSA7_SPECAN_FRQDOMAIN_CALCULUS 1
// flag of centering PSD scores 
#define CUSA7_SPECAN_PSDSCORES_CENTERED 1
// unrolling factor (for #positions) for the 2D global score matrix:
//NOTE: should not be greater than 4 because of high shared memory demand
#define CUSA7_SPECAN_PSDSCORES_XFCT 1

// radii of atom context shells within which neighbors' power spectrum is analyzed;
// minimum sequence separation at which atoms are considered for SPECAN;
enum {
    nTSAContextShells = 5,
    nTSAMinSeqSeparation = 6,
    nTSAMaxSeqSeparation = 63,//127,//(NOTE:>=2!!)
    //sequence separation dimension used as the inner dimension of the
    //transform in frequency domain;
    //NOTE: calculated as 2 * nTSAMaxSeqSeparation + 1 and rounded up to the
    //NOTE: nearest power of two!
    nTSASEQSEPDIM = 128,//256,
    nTSALOG2SEQSEPDIM = 7,//8,
#if (CUSA7_SPECAN_FRQDOMAIN_CALCULUS == 1)
    //dimension for power spectral density:
    nTSASEQSEPPSDDIM = ((nTSASEQSEPDIM>>1) + 1)
#else
    nTSASEQSEPPSDDIM = nTSASEQSEPDIM
#endif
};
struct TSAContextShellRadius {
    constexpr static float minradius_ = 2.0f;
    constexpr static float radii_[nTSAContextShells] = {
        6.0f, 8.0f, 10.0f, 12.0f, 15.0f
    };
};
// weights with which the spectrum of each context shell contributes to the score
struct TSAContextShellWeights {
    constexpr static float weights_[nTSAContextShells] = {
        0.29f, 0.33f, 0.24f, 0.12f, 0.02f//Poisson weights: lambda==8
//         0.2f, 0.4f, 0.3f, 0.07f, 0.03f
    };
};
//}}

// === End of spectral analysis section ====================================



// === DP section ==========================================================

#define CUBDP_TYPE  float
#define CUBDP_Q(CNST) (CNST ## . ## f)

// Dimensions of 2D cache of shared memory for performing dynamic programming;
// it also represents 2D dimensions for the CUDA thread block

// DP on spectral scores:
// always use configuration of CUDP_2DSPECAN_DIM_D==CUDP_2DSPECAN_DIM_X
// diagonal size (32, warp size)
#define CUDP_2DSPECAN_DIM_D 32
#define CUDP_2DSPECAN_DIM_D_LOG2 5
// size along x dimension (cannot be greater than CUDP_2DSPECAN_DIM_D)
#define CUDP_2DSPECAN_DIM_X CUDP_2DSPECAN_DIM_D
// CUDP_2DSPECAN_DIM_D + CUDP_2DSPECAN_DIM_X
#define CUDP_2DSPECAN_DIM_DpX (CUDP_2DSPECAN_DIM_D * 2)

// dimension for the approach of complete DP over structures
#define CUDP_COMPLETE_2DCACHE_DIM_D 512
// //minimum value for dimension
#define CUDP_COMPLETE_2DCACHE_MINDIM_D 32
#define CUDP_COMPLETE_2DCACHE_MINDIM_D_LOG2 5

// #define CUDP_2DCACHE_DIM_DequalsX
// always use configuration of CUDP_2DCACHE_DIM_D==CUDP_2DCACHE_DIM_X
// diagonal size (32, warp size)
#define CUDP_2DCACHE_DIM_D CUDP_2DCACHE_DIM_D_BASE //32
#define CUDP_2DCACHE_DIM_D_LOG2 CUDP_2DCACHE_DIM_D_LOG2_BASE //5
// size along x dimension (cannot be greater than CUDP_2DCACHE_DIM_D)
#define CUDP_2DCACHE_DIM_X CUDP_2DCACHE_DIM_D
// CUDP_2DCACHE_DIM_D + CUDP_2DCACHE_DIM_X
#define CUDP_2DCACHE_DIM_DpX (CUDP_2DCACHE_DIM_D * 2)
// // half of the bandwidth for banded alignment when it is in use:
// #define CUDP_BANDWIDTH_HALF 1024

// Configuration for executing swift DP;
// #define CUDP_SWFT_2DCACHE_DIM_DequalsX
// always use configuration of CUDP_SWFT_2DCACHE_DIM_D==CUDP_SWFT_2DCACHE_DIM_X
#define CUDP_SWFT_2DCACHE_DIM_D 128 //64 //32
#define CUDP_SWFT_2DCACHE_DIM_D_LOG2 7 //6 //5
// size along x dimension (cannot be greater than CUDP_SWFT_2DCACHE_DIM_D)
#define CUDP_SWFT_2DCACHE_DIM_X CUDP_SWFT_2DCACHE_DIM_D
// CUDP_SWFT_2DCACHE_DIM_D + CUDP_SWFT_2DCACHE_DIM_X
#define CUDP_SWFT_2DCACHE_DIM_DpX (CUDP_SWFT_2DCACHE_DIM_D * 2)

// dimension x of thread block for identifying matched positions (coordinates);
#define CUDP_MATCHED_DIM_X 32
// dimension y thread block for matched positions;
// NOTE: use nTDPAlignedPoss (=6) or 3:
#define CUDP_MATCHED_DIM_Y 6

// same as CUDP_MATCHED_DIM_XY but for constrained backtracking:
#define CUDP_CONST_MATCH_DIM_X 32
// NOTE: use nTDPAlignedPoss (=6) or 3:
#define CUDP_CONST_MATCH_DIM_Y 6

// dimensions x and y of thread block for producing final alignments;
#define CUDP_PRODUCTION_ALN_DIM_X 32
#define CUDP_PRODUCTION_ALN_DIM_Y 3

// === End of DP section ===================================================

#endif//__cuprocconf_h__
