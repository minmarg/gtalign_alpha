/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/custages/fields.cuh"
#include "libmycu/custages/covariance.cuh"
#include "local_similarity0.cuh"

// -------------------------------------------------------------------------
// CalcLocalSimilarity_frg2: calculate provisional local similarity during 
// extensive fragment-based search of optimal superpositions;
// NOTE: thread block is 1D and processes along reference structures;
// NOTE: CUSF_TBSP_LOCAL_SIMILARITY_XDIM should be 32: warp-level sync used;
// thrsimilarityperc, threshold percentage of local similarity score for a 
// fragment to be considered as one having the potential to improve superposition;
// depth, superposition depth for calculating query and reference positions;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db structure positions in the chunk;
// dbxpad, number of padded positions for memory alignment;
// maxnsteps, max number of steps (blockIdx.y) to perform for each reference structure;
// qryfragfct, fragment factor for query (to be multiplied by step dependent upon lengths);
// rfnfragfct, fragment factor for reference (to be multiplied by step dependent upon lengths);
// fragndx, fragment index determining the fragment size dependent upon lengths;
// NOTE: memory pointers should be aligned!
// dpscoremtx, input rounded global dp score matrix;
// wrkmemaux, auxiliary working memory;
// 
__global__
void CalcLocalSimilarity_frg2(
    const float thrsimilarityperc,
    const int depth,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    int qryfragfct, int rfnfragfct, int fragndx,
    const char* __restrict__ dpscoremtx,
    float* __restrict__ wrkmemaux)
{
    // blockIdx.x is the reference serial number;
    // blockIdx.y is a fragment factor;
    // blockIdx.z is the query serial number;
    //cache for convergence flags: 
    // __shared__ float cnvCache[CUSF_TBSP_LOCAL_SIMILARITY_XDIM];
    int cnvCache, globcnv;
    //reference serial number
    const uint dbstrndx = blockIdx.x * CUSF_TBSP_LOCAL_SIMILARITY_XDIM + threadIdx.x;
    const uint sfragfct = blockIdx.y;//fragment factor
    const uint qryndx = blockIdx.z;//query serial number
    fragndx = (sfragfct & 1);
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;
    int qrypos, rfnpos;//starting query and reference position
    int fraglen;

    globcnv = cnvCache = CONVERGED_SCOREDP_bitval;

    //read convergence first
    if(dbstrndx < ndbCstrs) {
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        globcnv = wrkmemaux[mloc0 + dbstrndx];
        if(globcnv <= CONVERGED_FRAGREF_bitval) globcnv = 0;
        cnvCache = globcnv;
        if(cnvCache) cnvCache = cnvCache | CONVERGED_SCOREDP_bitval;
    }

    if(dbstrndx < ndbCstrs && sfragfct != 0 && !globcnv) {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        cnvCache = wrkmemaux[mloc + dbstrndx];
    }


    //read query and reference attributes:
    if(dbstrndx < ndbCstrs && !globcnv) {
        dbstrlen = GetDbStrLength(dbstrndx);
        dbstrdst = GetDbStrDst(dbstrndx);
    }

    if(threadIdx.x == 0) {
        qrylen = GetQueryLength(qryndx);
        qrydst = GetQueryDst(qryndx);
    }

    qrylen = __shfl_sync(0xffffffff, qrylen, 0/*srcLane*/);
    qrydst = __shfl_sync(0xffffffff, qrydst, 0/*srcLane*/);


    //calculate start positions
    if(dbstrndx < ndbCstrs && !globcnv) {
        GetQryRfnPos_frg2(
            depth,
            qrypos, rfnpos, qrylen, dbstrlen, sfragfct, qryfragfct, rfnfragfct, fragndx
        );

        fraglen = GetNAlnPoss_frg(
            qrylen, dbstrlen, 0/*qrypos,unused*/, 0/*rfnpos,unused*/,
            qryfragfct/*unused*/, rfnfragfct/*unused*/, 0/*fragndx; first smaller*/);

        //if fragment is out of bounds, set the convergence flag anyway
        if(qrylen < qrypos + fraglen || dbstrlen < rfnpos + fraglen || fraglen < 1)
            cnvCache = cnvCache | CONVERGED_SCOREDP_bitval;
        else {
            fraglen = GetNAlnPoss_frg(
                qrylen, dbstrlen, 0/*qrypos,unused*/, 0/*rfnpos,unused*/,
                qryfragfct/*unused*/, rfnfragfct/*unused*/,
                1/*fragndx; always use the larger*/);

            if(qrylen < qrypos + fraglen) fraglen = qrylen - qrypos;
            if(dbstrlen < rfnpos + fraglen) fraglen = dbstrlen - rfnpos;

            if(qrylen < qrypos + fraglen || dbstrlen < rfnpos + fraglen || fraglen < 1)
                cnvCache = cnvCache | CONVERGED_SCOREDP_bitval;
        }
    }


    const int dblen = ndbCposs + dbxpad;

    //read dp matrix values at the appropriate positions (uncoalesced reads) and
    //set the convergence flag if calculated local similarity is too low:
    if(dbstrndx < ndbCstrs && !globcnv && !cnvCache) {
        int sposu = (qrydst + qrypos) * dblen + dbstrdst + rfnpos + fraglen-1;//upper cell pos.
        int sposl = (qrydst + qrypos + fraglen-1) * dblen + dbstrdst + rfnpos;//left cell pos.
        int spose = (qrydst + qrypos + fraglen-1) * dblen + dbstrdst + rfnpos + fraglen-1;
        int dpuval = (int)((unsigned char)(dpscoremtx[sposu]));//READ upper value
        int dplval = (int)((unsigned char)(dpscoremtx[sposl]));//READ left value
        int dpeval = (int)((unsigned char)(dpscoremtx[spose]));//READ bottom right
        float similarityu = (float)((dpuval <= dpeval)? (dpeval - dpuval): (256 - dpuval + dpeval));
        float similarityl = (float)((dplval <= dpeval)? (dpeval - dplval): (256 - dplval + dpeval));
        if(myhdmin(similarityu, similarityl) < (float)fraglen * thrsimilarityperc)
            cnvCache = cnvCache | CONVERGED_SCOREDP_bitval;
    }


    //write back the convergence flag for each query-reference pair processed by thread block
    if(dbstrndx < ndbCstrs) {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        //NOTE: if !globcnv, its value distributes across sfragfct;
        wrkmemaux[mloc + dbstrndx] = cnvCache;
    }
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
