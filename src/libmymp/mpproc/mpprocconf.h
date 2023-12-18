/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __mpprocconf_h__
#define __mpprocconf_h__

#include "libmymp/mpproc/mpprocconfbase.h"

// === Secondary structure section =========================================

// x-dimension for calculating secondary structure
// NOTE: unrolling factor is 1: single relatively non-intense calculation
#define MPSS_CALCSTR_XDIM 128

// === End of secondary structure section ==================================



// === Filter section ======================================================
// x-dimension for calculating sequence similarity, unprocessed tail 
// length for query and reference sequences, and diagonal step:
#define MPFL_TBSP_SEQUENCE_SIMILARITY_XDIM 128
#define MPFL_TBSP_SEQUENCE_SIMILARITY_EDGE 20
#define MPFL_TBSP_SEQUENCE_SIMILARITY_STEP 2
#define MPFL_TBSP_SEQUENCE_SIMILARITY_STEPLOG2 1
#define MPFL_TBSP_SEQUENCE_SIMILARITY_CHSIZE 32

// x,y-dimensions for making list of database candidates progressing to the
// next stages:
#define MPFL_MAKECANDIDATELIST_XDIM 128
#define MPFL_MAKECANDIDATELIST_YDIM 2

// x-dimension for copying selected database candidates from and to the
// original location, respectively:
#define MPFL_STORECANDIDATEDATA_XDIM 32
#define MPFL_STORECANDIDATEDATA_CHSIZE 32

#define MPFL_LOADCANDIDATEDATA_XDIM 32
#define MPFL_LOADCANDIDATEDATA_CHSIZE 32

// === End of filter section ===============================================



// === Scores section ======================================================
// STAGE 1

// x-dimension of thread block for calculating complete alignment refinement:
#define MPS1_TBINITSP_COMPLETEREFINE_XDIM 128
// chunk size for dynamic scheduling policy (applied to reference structures)
#define MPS1_TBINITSP_COMPLETEREFINE_CHSIZE 32


// // x-dimension of thread block for initializing transformation matrices:
// #define MPS1_TBINITSP_TFMINIT_XDIM 128
// unrolling factor (#structures) for initializing transformation matrices:
#define MPS1_TBINITSP_TFMINIT_XFCT 10
#define MPS1_TBINITSP_TFMINIT_CHSIZE 32

// x-dimension of thread block for initializing cross-covariance data:
#define MPS1_TBINITSP_CCDINIT_XDIM 64
// unrolling factor (#structures) for initializing cross-covariance data:
#define MPS1_TBINITSP_CCDINIT_XFCT 4

// x-dimension of thread block for calculating cross-covariance matrices:
// NOTE: do not use larger than 64: occupancy may be dramatically reduced
#define MPS1_TBINITSP_CCMCALC_XDIM 64
// unrolling factor for calculating cross-covariance matrices: 2, 4, 8
#define MPS1_TBINITSP_CCMCALC_XFCT 64 //16 //4
// logical x dimension for calculating cross-covariance matrices
#define MPS1_TBINITSP_CCMCALC_XDIMLGL (MPS1_TBINITSP_CCMCALC_XDIM * MPS1_TBINITSP_CCMCALC_XFCT)

// x-dimension of thread block for calculating cross-covariance matrices 
// iteratively:
#define MPS1_TBINITSP_CCMCALC_ITRD_XDIM 1024

// x-dimension (>=64) of thread block for calculating distance thresholds for the 
// inclusion of positions the refined rotaion matrices will be based on;
// NOTE: dimension of 1024 greatly reduces #active thread blocks and leads to 
// NOTE: several times LOWER kernel performance on average;
// NOTE: set to nearly average protein size:
#define MPS1_TBINITSP_FINDD02_ITRD_XDIM 256

// number of structures to be copied by thread block (shared memory dimension)
#define MPS1_TBINITSP_CCMCOPY_N 64

// unrolling factor for computing transformation matrices:
// number of query-reference structure pairs processed in 
// parallel by one block
#define MPS1_TBSP_TFM_N 32

// x-dimension of thread block for calculating scores (>=64):
#define MPS1_TBSP_SCORE_XDIM 128 //64
// unrolling factor for calculating scores: 2, 4, 8
#define MPS1_TBSP_SCORE_XFCT 64 //16 //4
// logical x dimension for calculating scores
#define MPS1_TBSP_SCORE_XDIMLGL (MPS1_TBSP_SCORE_XDIM * MPS1_TBSP_SCORE_XFCT)
// thread block x-dimension for calculating scores when checking for halt:
#define MPS1_TBSP_SCORE_FRG2_HALT_CHK_XDIM 32 //64

// x-dimension of thread block for setting calculated score 
// (corresponds to #structures):
#define MPS1_TBSP_SCORE_SET_XDIM 128
#define MPS1_TBSP_SCORE_SET_CHSIZE 32

// x-dimension of thread block for checking convergence:
#define MPS1_TBINITSP_CCDCONV_XDIM 256 //64
// unrolling factor (#structures) for checking convergence:
#define MPS1_TBINITSP_CCDCONV_XFCT 16 //4

// x-dimension of thread block for saving best-performing transformation matrices:
#define MPS1_TBINITSP_TMSAVE_XDIM 256 //128
// unrolling factor for saving best-performing transformation matrices:
#define MPS1_TBINITSP_TMSAVE_XFCT 20 //10

// thread block dimensions for finding the maximum among calculated scores 
// over all fragment factors (x-dim corresponds to #structures):
#define MPS1_TBSP_SCORE_MAX_XDIM 32
// #define MPS1_TBSP_SCORE_MAX_YDIM 32

// thread block dimensions for finding the maximum among calculated DP scores 
// over all fragment factors (x-dim corresponds to #structures):
#define MPS1_TBSP_DPSCORE_MAX_XDIM 32
#define MPS1_TBSP_DPSCORE_MAX_YDIM 32


// STAGE FRG (index-based scoring from fragment superpositions)

// x-dimension for common tasks within the scope of index-based scoring
#define MPSF_TBSP_COMMON_INDEX_SCORE_XDIM 128
#define MPSF_TBSP_COMMON_INDEX_SCORE_CHSIZE 32

// x-dimension of thread block for completely calculating fragment-based 
// superposition scores using spatial index:
// NOTE: currently should be 32
#define MPSF_TBSP_COMPLETE_INDEX_SCORE_XDIM 32


// x-dimension of thread block for calculating scores (>=64):
#define MPSF_TBSP_INDEX_SCORE_XDIM 256 //512
// unrolling factor for calculating scores: 2, 4, 8
#define MPSF_TBSP_INDEX_SCORE_XFCT 1
// logical x dimension for calculating scores
#define MPSF_TBSP_INDEX_SCORE_XDIMLGL (MPSF_TBSP_INDEX_SCORE_XDIM * MPSF_TBSP_INDEX_SCORE_XFCT)

// x-dimension of cache for calculating provisional local similarity:
#define MPSF_TBSP_LOCAL_SIMILARITY_XDIM 128 //(32x4)
#define MPSF_TBSP_LOCAL_SIMILARITY_YDIM 32
// x-dimension of thread block for reducing scores obtained from index (>=64):
#define MPSF_TBSP_INDEX_SCORE_REDUCE_XDIM 128 //64
// unrolling factor for reducing scores: 2, 4, 8
#define MPSF_TBSP_INDEX_SCORE_REDUCE_XFCT 64 //16 //4
// logical x dimension for calculating scores
#define MPSF_TBSP_INDEX_SCORE_REDUCE_XDIMLGL (MPSF_TBSP_INDEX_SCORE_REDUCE_XDIM * MPSF_TBSP_INDEX_SCORE_REDUCE_XFCT)

// x-dimension of thread block for saving the config of best-performing 
// transformation matrices:
#define MPSF_TBSP_INDEX_SCORE_SAVECFG_XDIM 256 //128

// thread block dimensions for finding the maximum among calculated scores 
// over all fragment factors (x-dim corresponds to #structures):
#define MPSF_TBSP_INDEX_SCORE_MAX_XDIM 32
#define MPSF_TBSP_INDEX_SCORE_MAX_YDIM 32

// dimension x of thread block for identifying matched positions (coordinates);
#define MPSF2_TBSP_INDEX_ALIGNMENT_XDIM 128 //64
// dimension y of thread block for matched positions (must be 6);
#define MPSF2_TBSP_INDEX_ALIGNMENT_YDIM 6
// unrolling factor for matched positions: 2, 4, 8
#define MPSF2_TBSP_INDEX_ALIGNMENT_XFCT 64 //16 //4
// logical x dimension for matched positions
#define MPSF2_TBSP_INDEX_ALIGNMENT_XDIMLGL (MPSF2_TBSP_INDEX_ALIGNMENT_XDIM * MPSF2_TBSP_INDEX_ALIGNMENT_XFCT)

// cache size for query positions when calculating scores 
// progressively from alignment (unordered in position) obtained by 
// applying a linear alignment algorithm
#define MPSF2_TBSP_INDEX_SCORE_QNX_CACHE_SIZE 512


// FINAL STAGE (final alignment refinement)

// === End of scores section ===============================================



// === DP section ==========================================================

#define MPBDP_TYPE  float
#define MPBDP_Q(CNST) (CNST ## . ## f)

// Dimensions of 2D cache of shared memory for performing dynamic programming;

// DP on spectral scores:
// always use configuration of MPDP_2DSPECAN_DIM_D==MPDP_2DSPECAN_DIM_X
// diagonal size (32, warp size)
#define MPDP_2DSPECAN_DIM_D 32
#define MPDP_2DSPECAN_DIM_D_LOG2 5
// size along x dimension (cannot be greater than MPDP_2DSPECAN_DIM_D)
#define MPDP_2DSPECAN_DIM_X MPDP_2DSPECAN_DIM_D
// MPDP_2DSPECAN_DIM_D + MPDP_2DSPECAN_DIM_X
#define MPDP_2DSPECAN_DIM_DpX (MPDP_2DSPECAN_DIM_D * 2)

// dimension for the approach of complete DP over structures
#define MPDP_COMPLETE_2DCACHE_DIM_D 512
// //minimum value for dimension
#define MPDP_COMPLETE_2DCACHE_MINDIM_D 32
#define MPDP_COMPLETE_2DCACHE_MINDIM_D_LOG2 5

// #define MPDP_2DCACHE_DIM_DequalsX
// always use configuration of MPDP_2DCACHE_DIM_D==MPDP_2DCACHE_DIM_X
// diagonal size (32, warp size)
#define MPDP_2DCACHE_DIM_D CUDP_2DCACHE_DIM_D_BASE //32
#define MPDP_2DCACHE_DIM_D_LOG2 CUDP_2DCACHE_DIM_D_LOG2_BASE //5
// size along x dimension (cannot be greater than MPDP_2DCACHE_DIM_D)
#define MPDP_2DCACHE_DIM_X MPDP_2DCACHE_DIM_D
// MPDP_2DCACHE_DIM_D + MPDP_2DCACHE_DIM_X
#define MPDP_2DCACHE_DIM_DpX (MPDP_2DCACHE_DIM_D * 2)
// chunk size for dynamic scheduling policy
#define MPDP_2DCACHE_CHSIZE 8 //32


// Configuration for executing swift DP;
// #define MPDP_SWFT_2DCACHE_DIM_DequalsX
// always use configuration of MPDP_SWFT_2DCACHE_DIM_D==MPDP_SWFT_2DCACHE_DIM_X
#define MPDP_SWFT_2DCACHE_DIM_D 128 //64 //32
#define MPDP_SWFT_2DCACHE_DIM_D_LOG2 7 //6 //5
// size along x dimension (cannot be greater than MPDP_SWFT_2DCACHE_DIM_D)
#define MPDP_SWFT_2DCACHE_DIM_X MPDP_SWFT_2DCACHE_DIM_D
// MPDP_SWFT_2DCACHE_DIM_D + MPDP_SWFT_2DCACHE_DIM_X
#define MPDP_SWFT_2DCACHE_DIM_DpX (MPDP_SWFT_2DCACHE_DIM_D * 2)
#define MPDP_SWFT_2DCACHE_CHSIZE 8

// dimension x of thread block for identifying matched positions (coordinates);
#define MPDP_MATCHED_DIM_X 128
// chunk size for dynamic scheduling policy
#define MPDP_MATCHED_CHSIZE 32

// same as MPDP_MATCHED_DIM_XY but for constrained backtracking:
#define MPDP_CONST_MATCH_DIM_X 128

// dimensions x and y of thread block for producing final alignments;
#define MPDP_PRODUCTION_ALN_DIM_X 128
// chunk size for dynamic scheduling policy
#define MPDP_PRODUCTION_ALN_CHSIZE 32

// === End of DP section ===================================================

#endif//__mpprocconf_h__
