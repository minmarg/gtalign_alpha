/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __mpprocconfbase_h__
#define __mpprocconfbase_h__

#include "libmycu/cucom/cudef.h"

// === General section =====================================================

#define MAX_QUERY_STRUCTURES_PER_CHUNK 100

// === End of general section ==============================================




// === Scores section ======================================================

// STAGE 1

// perform the calculation of distance thresholds during fragment boundary 
// refinment in iterations applied to convergence
#define DO_FINDD02_DURING_REFINEFRAG 0
//maximum number of subdivisions of identified fragments during refinement
#define FRAGREF_NMAXSUBFRAGS 6
//step size with which fragments are traversed during refinement
#define FRAGREF_SFRAGSTEP 40
#define FRAGREF_SFRAGSTEP_mini 1
//maximum number of iterations until convergence
#define FRAGREF_NMAXCONVIT 20
#define FRAGREF_NMAXCONVIT_mini 4

// consideration of top N DP scores calculated over all fragment factors for 
// further processing (should be <= CUS1_TBSP_DPSCORE_MAX_YDIM (32, warpsize) 
// and expressed as a power of 2):
#define CUS1_TBSP_DPSCORE_TOP_N 32 //16
// #max sections over which top N DP scores are calculated; hence, total max 
// number of DP scores to calculate is 
// CUS1_TBSP_DPSCORE_TOP_N * CUS1_TBSP_DPSCORE_TOP_N_MAX_CONFIGS
#define CUS1_TBSP_DPSCORE_TOP_N_MAX_CONFIGS 3
// MAX number of best-scoring TFM configurations selected from top N DP scores for 
// further refinement 
// (obviously, CUS1_TBSP_DPSCORE_TOP_N_REFINEMENT<=CUS1_TBSP_DPSCORE_TOP_N)
#define CUS1_TBSP_DPSCORE_TOP_N_REFINEMENT 16
// total MAX number of confihurations to verify alternatively
#define CUS1_TBSP_DPSCORE_TOP_N_REFINEMENTxMAX_CONFIGS \
    ((CUS1_TBSP_DPSCORE_TOP_N_MAX_CONFIGS) * (CUS1_TBSP_DPSCORE_TOP_N_REFINEMENT))

// maximum number of iterations for applying DP to the refinement of alignment 
// identified during fragment boundary refinement
#define DPREFINE_MAXNITS 18 //30
#define DPREFINE_MAXNITS_FAST 2


// STAGE FRG (index-based scoring from fragment superpositions)

// max number of positions away from the beginning of a fragment on both sides to consider
#define CUSF_TBSP_INDEX_SCORE_POSLIMIT 256 /*!*/ //512 //256
#define CUSF_TBSP_INDEX_SCORE_POSLIMIT2 (2 * CUSF_TBSP_INDEX_SCORE_POSLIMIT)
// max index depth in searching for the nearest neighbour atom
#define CUSF_TBSP_INDEX_SCORE_MAXDEPTH 64
// use secondary structure filtering when searching for the nearest neighbour atom
#define CUSF_TBSP_INDEX_SCORE_SECSTRFILT 0


// FINAL STAGE (final alignment refinement)

// max number of positions around an identified position to consider in the
// final refinement stage
#define CUSFN_TBSP_FIN_REFINEMENT_MAX_NPOSITIONS 128 //512

// === End of scores section ===============================================



// === DP section ==========================================================

// half of the bandwidth for banded alignment when it is in use:
#define CUDP_BANDWIDTH_HALF 1024

// #define CUDP_2DCACHE_DIM_DequalsX
// always use configuration of CUDP_2DCACHE_DIM_D==CUDP_2DCACHE_DIM_X
// diagonal size (32, warp size)
#ifdef GPUINUSE
#   define CUDP_2DCACHE_DIM_D_BASE 32
#   define CUDP_2DCACHE_DIM_D_LOG2_BASE 5
#else
#   define CUDP_2DCACHE_DIM_D_BASE 32 //64 //128
#   define CUDP_2DCACHE_DIM_D_LOG2_BASE 5 //6 //7
#endif

// === End of DP section ===================================================

#endif//__mpprocconfbase_h__
