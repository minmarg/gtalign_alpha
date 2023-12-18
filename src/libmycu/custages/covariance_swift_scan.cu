/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

// All kernels in this module designed to process covariance and related 
// data following the identification of matched positions

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/covariance.cuh"
#include "libmycu/custages/covariance_plus.cuh"
#include "libmycu/custages/covariance_refn.cuh"
#include "libmycu/custages/covariance_dp_refn.cuh"
#include "covariance_swift_scan.cuh"

#define COVARIANCE_SWIFT_SCAN_CHECK_CONVERGENCE 1

// -------------------------------------------------------------------------
// CopyCCDataToWrkMem2_SWFTscan: copy cross-covariance matrix between the 
// query and reference structures to section 2 to enable efficient Kabsch 
// algorithm application for multiple structures simultaneously; 
// Version for alignments obtained as a result of linearity application;
// NOTE: thread block is 2D and copies structures' data: from:
// NOTE: | struct i          | struct i+1        | ...
// NOTE: | field1,dield2,... | field1,dield2,... | ...
// NOTE: to 
// NOTE: | struct i | struct i+1 | ... | struct i | ... 
// NOTE: | field1   | field1     | ... | field2   | ...
// should be read;
// READNPOS, template parameter for checking whether nalnposs has changed;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// wrkmem, working memory, including the section of CC data (saved as 
// whole for each structure) to copy;
// wrkmem2, working memory, including the section of CC data to be written by 
// field;
// 
template<int READNPOS>
__global__ 
void CopyCCDataToWrkMem2_SWFTscan(
    const uint ndbCstrs,
    const uint maxnsteps,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmem2)
{
    enum{ndxproc = twmvEndOfCCDataExt};//index of processing flag in cache
    //cache for cross-covariance matrices and related data: 
    //bank conflicts resolved as long as innermost dim is odd
    __shared__ float ccmCache[CUS1_TBINITSP_CCMCOPY_N][twmvEndOfCCDataExt+1];
    //index of the first structure to start with (blockIdx.x, refn. serial number):
    int dbstrndx = blockIdx.x * CUS1_TBINITSP_CCMCOPY_N;
    int qryndx = blockIdx.y;//query index in the chunk
    int sfragfct = blockIdx.z;//fragment factor
    int absndx = dbstrndx + threadIdx.x;
    //int nalnposs = 0;

#if (COVARIANCE_SWIFT_SCAN_CHECK_CONVERGENCE != 1)
    ccmCache[threadIdx.x][tawmvConverged] = 0.0f;
#endif

    if(absndx < ndbCstrs && (threadIdx.y == tawmvNAlnPoss
#if (COVARIANCE_SWIFT_SCAN_CHECK_CONVERGENCE == 1)
        || threadIdx.y == tawmvConverged
#endif
        ))
    {
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + threadIdx.y) * ndbCstrs;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + threadIdx.y) * ndbCstrs;
        ccmCache[threadIdx.x][threadIdx.y] = wrkmemaux[mloc + absndx/*dbstrndx*/];
        if(threadIdx.y == tawmvConverged) {
            float convflag = ccmCache[threadIdx.x][threadIdx.y];
            if(sfragfct != 0 ) convflag = wrkmemaux[mloc0 + absndx];
            ccmCache[threadIdx.x][threadIdx.y] = //any convergence applies locally
                (((int)(convflag)) & (CONVERGED_LOWTMSC_bitval)) ||
                (ccmCache[threadIdx.x][threadIdx.y]);
        }
    }

    __syncthreads();

    //write to smem #alignment positions
    if(absndx < ndbCstrs && threadIdx.y == 0)
    {
        ccmCache[threadIdx.x][ndxproc] = ccmCache[threadIdx.x][tawmvNAlnPoss];
        //any type of convergence applies
        if(ccmCache[threadIdx.x][tawmvConverged]) {
            //assign 0 #aligned positions so that no memory and 
            //computing operations are executed
            ccmCache[threadIdx.x][twmvNalnposs] = 0.0f;
            ccmCache[threadIdx.x][ndxproc] = 0.0f;
        }
    }

    __syncthreads();

    //cache data: iterative coalesced read
    for(int reldbndx = threadIdx.y; reldbndx < CUS1_TBINITSP_CCMCOPY_N; reldbndx += blockDim.y) {
        int absndxloc = dbstrndx + reldbndx;
        if(absndxloc < ndbCstrs && 
           threadIdx.x < twmvEndOfCCDataExt && 
           ccmCache[reldbndx][ndxproc])
        {
            //read only if not converged and not out of bounds
            uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + absndxloc) * nTWorkingMemoryVars;
            ccmCache[reldbndx][threadIdx.x] = wrkmem[mloc + threadIdx.x];
            if(READNPOS==READNPOS_READ &&
               threadIdx.x == twmvNalnposs && 
               ccmCache[reldbndx][twmvNalnposs] == ccmCache[reldbndx][ndxproc]) {
                //NOTE: if nalnposs equals maximum possible for given qrypos and rfnpos,
                //assign it to 0 so that the Kabsch algorithm is not applied to this 
                //particular query-reference pair:
                ccmCache[reldbndx][twmvNalnposs] = 0.0f;
                ccmCache[reldbndx][ndxproc] = 0.0f;
            }
        }
    }

    __syncthreads();

    //write data to gmem; coalesced write;
    //first write nalnposs 
    if(absndx < ndbCstrs && threadIdx.y == twmvNalnposs) {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTWorkingMemoryVars + threadIdx.y) * ndbCstrs;
        wrkmem2[mloc + absndx] = ccmCache[threadIdx.x][threadIdx.y];
    }

    if(absndx < ndbCstrs && threadIdx.y < twmvEndOfCCData &&
       ccmCache[threadIdx.x][ndxproc]) {
        //write only if nalnposs >0;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTWorkingMemoryVars + threadIdx.y) * ndbCstrs;
        wrkmem2[mloc + absndx] = ccmCache[threadIdx.x][threadIdx.y];
    }
}

// =========================================================================
// Instantiations:
// 
#define INSTANTIATE_CopyCCDataToWrkMem2_SWFTscan(tpREADNPOS) \
    template __global__ void CopyCCDataToWrkMem2_SWFTscan<tpREADNPOS>( \
        const uint ndbCstrs, const uint maxnsteps, \
        const float* __restrict__ wrkmemaux, \
        const float* __restrict__ wrkmem, \
        float* __restrict__ wrkmem2);

INSTANTIATE_CopyCCDataToWrkMem2_SWFTscan(READNPOS_NOREAD);
INSTANTIATE_CopyCCDataToWrkMem2_SWFTscan(READNPOS_READ);

// -------------------------------------------------------------------------
// CalcCCMatrices64_SWFTscan: calculate cross-covariance matrix between the 
// query and reference structures given an alignment between them;
// Version for alignments obtained as a result of linearity application;
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// NOTE: Version for CUS1_TBINITSP_CCMCALC_XDIM==64!
// nqystrs, total number of queries;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of reference positions in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// tmpdpalnpossbuffer, global address of aligned positions;
// wrkmem, working memory, including the section of CC data;
// 
__global__ 
void CalcCCMatrices64_SWFTscan(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint /*dbxpad*/,
    const uint maxnsteps,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmem)
{
    // blockIdx.x is the block index of positions for query-reference pair;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number TIMES fragment factor;
    //cache for the cross-covarinace matrix and related data: 
    //effective number of fields, and their total number to resolve bank conflicts:
    enum{neffds = twmvEndOfCCDataExt, smidim = neffds + 1};
    __shared__ float ccmCache[smidim * CUS1_TBINITSP_CCMCALC_XDIM];
    //relative position index:
    const uint ndx0 = blockIdx.x * blockDim.x * CUS1_TBINITSP_CCMCALC_XFCT;
    const uint ndx = ndx0 + threadIdx.x;
    const uint sfragfct = blockIdx.z / nqystrs;//fragment factor
    const uint qryndx = blockIdx.z - sfragfct * nqystrs;//query serial number
    const uint dbstrndx = blockIdx.y;
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    uint /*qrydst, */dbstrdst;
    const int qrypos = 0, rfnpos = 0;


#if (COVARIANCE_SWIFT_SCAN_CHECK_CONVERGENCE == 1)
    if(threadIdx.x == 0) {
        //NOTE: reuse ccmCache to read convergence flag at both 0 and sfragfct:
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        ccmCache[6] = wrkmemaux[mloc0 + dbstrndx];
    }
    if(threadIdx.x == 32) {//next warp
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        ccmCache[7] = wrkmemaux[mloc + dbstrndx];
    }

    __syncthreads();

    if((((int)(ccmCache[6])) & (CONVERGED_LOWTMSC_bitval)) || ccmCache[7])
        //(NOTE:any type of convergence applies locally);
        //all threads in the block exit;
        return;

    //NOTE: no sync as long as ccmCache cells for convergence not overwritten;
#endif


    //reuse ccmCache
    if(threadIdx.x == 0) {
        ((int*)ccmCache)[1] = GetDbStrDst(dbstrndx);
        //((int*)ccmCache)[3] = GetQueryDst(qryndx);
    }
    if(threadIdx.x == tawmvNAlnPoss + 32) {
        //NOTE: reuse ccmCache to read #matched positions (tawmvNAlnPoss) written at sfragfct;
        //NOTE: use different warp; structure-specific-formatted data;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        ccmCache[threadIdx.x] = wrkmemaux[mloc + (threadIdx.x-32) * ndbCstrs + dbstrndx];
    }

    __syncthreads();


    //NOTE: no bank conflict when two threads from the same warp access the same address;
    //blockDim.x includes several warps
    //TODO: integers in [0;16777216] can be exactly represented by float:
    //TODO: consider updating memory limits calculation or using int cache!
    dbstrdst = ((int*)ccmCache)[1];
    //qrydst = ((int*)ccmCache)[3];
    qrylen = dbstrlen = ccmCache[tawmvNAlnPoss+32];

    __syncthreads();


    if(qrylen <= qrypos + ndx0 || dbstrlen <= rfnpos + ndx0)
        //all threads in the block exit if thread 0 is out of bounds
        return;


    //qrylen == dbstrlen; reuse qrylen for original alignment length;

    //initialize cache:
    //(initialization in parts is more efficient wrt #registers)
    InitCCMCacheExtended<smidim,0,neffds>(ccmCache);


    const int dblen = ndbCposs/* + dbxpad*/;
    //offset to the beginning of the data along the y axis 
    // wrt query qryndx and maxnsteps: 
    const int yofff = (qryndx * maxnsteps + sfragfct) * dblen * nTDPAlignedPoss;

    #pragma unroll
    for(int i = 0; i < CUS1_TBINITSP_CCMCALC_XFCT; i++) {
        //manually unroll along data blocks by a factor of CUS1_TBINITSP_CCMCALC_XFCT
        if(!(/*qrypos + ndx + i * blockDim.x < qrylen &&*/
             rfnpos + ndx + i * blockDim.x < dbstrlen))
            break;
        //starting position in tmpdpalnpossbuffer for a pair:
        //NOTE: qrypos == rfnpos as well as lengths;
        int pos = yofff + dbstrdst + (rfnpos + ndx + i * blockDim.x);
        UpdateCCMOneAlnPos_SWFTRefined<smidim>(//no sync;
            pos, dblen,
            tmpdpalnpossbuffer,
            ccmCache
        );
    }

    //sync now:
    __syncthreads();

    //unroll by a factor 2
    if(threadIdx.x < (CUS1_TBINITSP_CCMCALC_XDIM>>1)) {
        #pragma unroll
        for(int i = 0; i < neffds; i++)
            ccmCache[threadIdx.x * smidim + i] +=
                ccmCache[(threadIdx.x + (CUS1_TBINITSP_CCMCALC_XDIM>>1)) * smidim + i];
    }

    __syncthreads();

    //unroll warp
    if(threadIdx.x < 32) {
        #pragma unroll
        for(int i = 0; i < neffds; i++) {
            float sum = ccmCache[threadIdx.x * smidim + i];
            sum = mywarpreducesum(sum);
            //write to the first data slot of SMEM
            if(threadIdx.x == 0) ccmCache[i] = sum;
        }
    }

    //in case of twmvEndOfCCData gets larger than warpSize
    __syncthreads();

    uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTWorkingMemoryVars;

    //add the result and write to global memory
    if(threadIdx.x < neffds)
        atomicAdd(&wrkmem[mloc + threadIdx.x], ccmCache[threadIdx.x]);
}

// -------------------------------------------------------------------------
// FindD02ThresholdsCCM_SWFTscan: efficiently find distance thresholds 
// for the inclusion of aligned positions for CCM and rotation matrix 
// calculations during the exhaustive search of alignment variants;
// NOTE: thread block is 1D and processes alignment along structure
// positions;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db structure positions in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary diagonal buffers filled with positional 
// scores;
// wrkmem, working memory, including the section of CC data;
// wrkmemaux, auxiliary working memory;
// 
template<int READCNST>
__global__
void FindD02ThresholdsCCM_SWFTscan(
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const float* __restrict__ tmpdpdiagbuffers,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux)
{
    // blockIdx.x is the reference serial number;
    // blockIdx.y is the query serial number;
    // blockIdx.z is the fragment factor;
    //cache for minimum scores: 
    //no bank conflicts as long as inner-most dim is odd
    enum{smidim = 3};//top three min scores
    __shared__ float ccmCache[smidim * CUS1_TBINITSP_FINDD02_ITRD_XDIM];
    const uint dbstrndx = blockIdx.x;
    const uint qryndx = blockIdx.y;//query serial number
    const uint sfragfct = blockIdx.z;//fragment factor
    int qrylenorg, dbstrlenorg;//original query and reference lengths
    int qrylen, dbstrlen;//pseudo query and reference length, #matched positions
    //distances in positions to the beginnings of the query and reference structures:
    uint /*qrydst, */dbstrdst;
    enum{qrypos = 0, rfnpos = 0};


#if (COVARIANCE_SWIFT_SCAN_CHECK_CONVERGENCE == 1)
    if(threadIdx.x == 0) {
        //NOTE: reuse ccmCache to read convergence flag at sfragfct==0:
        uint mloc = ((qryndx * maxnsteps + 0/*sfragfct*/) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        ccmCache[6] = wrkmemaux[mloc + dbstrndx];
    }

    __syncthreads();

    if(ccmCache[6])
        //(NOTE:any type of convergence applies);
        //all threads in the block exit;
        return;

    //NOTE: no sync as long as ccmCache cell for convergence is not overwritten;
#endif


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse ccmCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(dbstrndx, (int*)ccmCache);
        //GetQueryLenDst(qryndx, (int*)ccmCache + 2);
        if(threadIdx.x == 0) ((int*)ccmCache)[2] = GetQueryLength(qryndx);
    }
    if(threadIdx.x == tawmvNAlnPoss + 32) {
        //NOTE: reuse ccmCache to read #matched positions (tawmvNAlnPoss) written at sfragfct;
        //NOTE: use different warp; structure-specific-formatted data;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        ccmCache[threadIdx.x] = wrkmemaux[mloc + (threadIdx.x-32) * ndbCstrs + dbstrndx];
    }

    __syncthreads();

    //NOTE: no bank conflict when two threads from the same warp access the same address;
    dbstrlenorg = ((int*)ccmCache)[0]; dbstrdst = ((int*)ccmCache)[1];
    qrylenorg = ((int*)ccmCache)[2]; //qrydst = ((int*)ccmCache)[3];
    qrylen = dbstrlen = ccmCache[tawmvNAlnPoss+32];


    __syncthreads();


    if(READCNST == READCNST_CALC2) {
        if(threadIdx.x == 0) {
            //NOTE: reuse ccmCache[0] to contain twmvLastD02s. ccmCache[1] twmvNalnposs
            uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTWorkingMemoryVars;
            ccmCache[1] = wrkmem[mloc + twmvNalnposs];
        }

        __syncthreads();

        int nalnposs = ccmCache[1];
        if(nalnposs == dbstrlen)
            //dbstrlen is #originally aligned positions;
            //all threads in the block exit;
            return;

        //cache will be overwritten below, sync
        __syncthreads();
    }


    //calculate the threshold over the original fragment
    //initialize cache
    #pragma unroll
    for(int i = 0; i < smidim; i++)
        ccmCache[threadIdx.x * smidim + i] = CP_LARGEDST;

    for(int rpos = threadIdx.x; qrypos + rpos < qrylen && rfnpos + rpos < dbstrlen;
        rpos += blockDim.x)
    {
        //manually unroll along alignment;
        //x2 to account for scores and query positions:
        int mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs * 2;
        GetMinScoreOneAlnPos<smidim>(//no sync;
            mloc + dbstrdst + rpos,//position for scores
            tmpdpdiagbuffers,
            ccmCache
        );
    }

    //sync now:
    __syncthreads();

    //unroll until reaching warpSize; 
    for(int xdim = (CUS1_TBINITSP_FINDD02_ITRD_XDIM>>1); xdim >= 32; xdim >>= 1) {
        int tslot = threadIdx.x * smidim;
        //ccmCache will contain 3x32 (or length-size) (possibly equal) minimum scores 
        if(threadIdx.x < xdim &&
           qrypos + threadIdx.x + xdim < qrylen &&
           rfnpos + threadIdx.x + xdim < dbstrlen)
            StoreMinDstSrc(ccmCache + tslot, ccmCache + tslot + xdim * smidim);

        __syncthreads();
    }

    //unroll warp
    if(threadIdx.x < 32/*warpSize*/) {
        for(int xdim = (32>>1); xdim >= 1; xdim >>= 1) {
            int tslot = threadIdx.x * smidim;
            if(threadIdx.x < xdim)
                StoreMinDstSrc(ccmCache + tslot, ccmCache + tslot + xdim * smidim);
            __syncwarp();
        }
    }

    //write to gmem the minimum score that ensures at least 3 aligned positions:
    if(threadIdx.x == 2) {
        float d0 = GetD0(qrylenorg, dbstrlenorg);
        float d02s = GetD02s(d0);
        if(READCNST == READCNST_CALC2) d02s += D02s_PROC_INC;

        float min3 = ccmCache[threadIdx.x];

        //TODO: move the clause (maxnalnposs <= 3) along with the write to gmem up
        if(CP_LARGEDST_cmp < min3 || min3 < d02s ||
           GetGplAlnLength(qrylen, dbstrlen, qrypos, rfnpos) <= 3)
            //max number of alignment positions (maxnalnposs) <3;
            //use the dfault threshold
            min3 = d02s;
        else {//round the 3rd minimum score to the next multiple of 0.5:
            //obtained from d02s + k*0.5 >= min3
            min3 = d02s + ceilf((min3 - d02s) * 2.0f) * 0.5f;
            //d0 = floorf(min3);
            //d02s = min3 - d0;
            //if(d02s) min3 = d0 + ((d02s <= 0.5f)? 0.5f: 1.0f);
        }

        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvLastD02s) * ndbCstrs;
        wrkmemaux[mloc + dbstrndx] = min3;
    }
}

// =========================================================================
// Instantiations
//
#define INSTANTIATE_FindD02ThresholdsCCM_SWFTscan(tpREADCNST) \
    template \
    __global__ void FindD02ThresholdsCCM_SWFTscan<tpREADCNST>( \
        const uint ndbCstrs, const uint ndbCposs, const uint maxnsteps, \
        const float* __restrict__ tmpdpdiagbuffers, \
        const float* __restrict__ wrkmem, \
        float* __restrict__ wrkmemaux);

INSTANTIATE_FindD02ThresholdsCCM_SWFTscan(READCNST_CALC);
INSTANTIATE_FindD02ThresholdsCCM_SWFTscan(READCNST_CALC2);

// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// CalcCCMatrices64_SWFTscanExtended: calculate cross-covariance matrix 
// between the query and reference structures based on aligned positions 
// within given distance;
// Version for alignments obtained through the exhaustive search over 
// alignment variants;
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// NOTE: Version for CUS1_TBINITSP_CCMCALC_XDIM==64!
// nqystrs, total number of queries;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db structure positions in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// NOTE: memory pointers should be aligned!
// tmpdpalnpossbuffer, global address of the buffers of aligned positions;
// tmpdpdiagbuffers, temporary diagonal buffers filled with positional 
// scores;
// wrkmemaux, auxiliary working memory;
// wrkmem, working memory, including the section of CC data;
// 
template<int READCNST>
__global__
void CalcCCMatrices64_SWFTscanExtended(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint /*dbxpad*/,
    const uint maxnsteps,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ tmpdpdiagbuffers,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ wrkmem)
{
    // blockIdx.x is the block index of positions for query-reference pair;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number TIMES fragment factor;
    //cache for the cross-covarinace matrix and related data: 
    //no bank conflicts as long as inner-most dim is odd;
    //effective number of fields:
    enum{neffds = twmvEndOfCCDataExt, smidim = neffds + 1};
    __shared__ float ccmCache[smidim * CUS1_TBINITSP_CCMCALC_XDIM];
    //relative position index:
    const uint ndx0 = blockIdx.x * blockDim.x * CUS1_TBINITSP_CCMCALC_XFCT;
    const uint ndx = ndx0 + threadIdx.x;
    const uint sfragfct = blockIdx.z / nqystrs;//fragment factor
    const uint qryndx = blockIdx.z - sfragfct * nqystrs;//query serial number
    const uint dbstrndx = blockIdx.y;
    int qrylen, dbstrlen;//pseudo query and reference length, #matched positions
    //distances in positions to the beginnings of the query and reference structures:
    uint /*qrydst, */dbstrdst;
    enum{qrypos = 0, rfnpos = 0};


#if (COVARIANCE_SWIFT_SCAN_CHECK_CONVERGENCE == 1)
    if(threadIdx.x == 0) {
        //NOTE: reuse ccmCache to read convergence flag at sfragfct==0:
        uint mloc = ((qryndx * maxnsteps + 0/*sfragfct*/) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        ccmCache[6] = wrkmemaux[mloc + dbstrndx];
    }

    __syncthreads();

    if(ccmCache[6])
        //(NOTE:any type of convergence applies);
        //all threads in the block exit;
        return;

    //NOTE: no sync as long as ccmCache cell for convergence is not overwritten;
#endif


    //reuse ccmCache
    if(threadIdx.x == 0) {
        ((int*)ccmCache)[1] = GetDbStrDst(dbstrndx);
        //((int*)ccmCache)[3] = GetQueryDst(qryndx);
    }
    if(threadIdx.x == tawmvNAlnPoss + 32) {
        //NOTE: reuse ccmCache to read #matched positions (tawmvNAlnPoss) written at sfragfct;
        //NOTE: use different warp; structure-specific-formatted data;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        ccmCache[threadIdx.x] = wrkmemaux[mloc + (threadIdx.x-32) * ndbCstrs + dbstrndx];
    }

    __syncthreads();


    //NOTE: no bank conflict when two threads from the same warp access the same address;
    //blockDim.x includes several warps
    dbstrdst = ((int*)ccmCache)[1];
    //qrydst = ((int*)ccmCache)[3];
    qrylen = dbstrlen = ccmCache[tawmvNAlnPoss+32];

    __syncthreads();


    if(qrylen <= qrypos + ndx0 || dbstrlen <= rfnpos + ndx0)
        //all threads in the block exit if thread 0 is out of bounds
        return;


    InitCCMCacheExtended<smidim,6,neffds>(ccmCache);

    if(threadIdx.x == 0) {
        //NOTE: reuse ccmCache[0] to contain twmvLastD02s, ccmCache[1] twmvNalnposs
        //structure-specific-formatted data
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvLastD02s) * ndbCstrs;
        ccmCache[0] = wrkmemaux[mloc + dbstrndx];

        if(READCNST == READCNST_CALC2) {
            mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTWorkingMemoryVars;
            ccmCache[1] = wrkmem[mloc + twmvNalnposs];
        }
    }

    __syncthreads();

    float d02s = ccmCache[0];

    if(READCNST == READCNST_CALC2) {
        int nalnposs = ccmCache[1];
        if(nalnposs == dbstrlen)
            //dbstrlen is #originally aligned positions;
            //all threads in the block exit;
            return;
    }

    //cache will be overwritten below, sync
    __syncthreads();


    //cache initialization divided into two parts for a more efficient use of registers
    InitCCMCacheExtended<smidim,0,6>(ccmCache);

    const int dblen = ndbCposs/* + dbxpad*/;
    //offset to the beginning of the data along the y axis 
    // wrt query qryndx and maxnsteps: 
    const int yofff = (qryndx * maxnsteps + sfragfct) * dblen * nTDPAlignedPoss;

    for(int i = 0; i < CUS1_TBINITSP_CCMCALC_XFCT; i++) {
        //manually unroll along data blocks by a factor of CUS1_TBINITSP_CCMCALC_XFCT;
        //x2 to account for scores and query positions:
        int mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs * 2;
        int pos0 = ndx + i * blockDim.x;//position index starting from 0
        if(!(qrypos + pos0 < qrylen && rfnpos + pos0 < dbstrlen))
            break;
        //starting position in tmpdpalnpossbuffer for a pair:
        //NOTE: qrypos == rfnpos as well as qrylen == dbstrlen here
        int dppos = yofff + dbstrdst + (rfnpos + pos0);
        UpdateCCMOneAlnPos_SWFTExtended<smidim>(//no sync;
            d02s,
            dppos, dblen,
            mloc + dbstrdst + pos0,//position for scores
            tmpdpalnpossbuffer,//coordinates
            tmpdpdiagbuffers,//scores
            ccmCache//reduction output
        );
    }

    //sync now:
    __syncthreads();

    //unroll by a factor of 2
    if(threadIdx.x < (CUS1_TBINITSP_CCMCALC_XDIM>>1)) {
        #pragma unroll
        for(int i = 0; i < neffds; i++)
            ccmCache[threadIdx.x * smidim + i] +=
                ccmCache[(threadIdx.x + (CUS1_TBINITSP_CCMCALC_XDIM>>1)) * smidim + i];
    }

    __syncthreads();

    //unroll warp
    if(threadIdx.x < 32) {
        #pragma unroll
        for(int i = 0; i < neffds; i++) {
            float sum = ccmCache[threadIdx.x * smidim + i];
            sum = mywarpreducesum(sum);
            //write to the first data slot of SMEM
            if(threadIdx.x == 0) ccmCache[i] = sum;
        }
    }

    //in case of neffds gets larger than warpSize
    __syncthreads();

    //add the result and write to global memory
    if(threadIdx.x < neffds) {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTWorkingMemoryVars;
        atomicAdd(&wrkmem[mloc + threadIdx.x], ccmCache[threadIdx.x]);
    }
}

// =========================================================================
// Instantiations
//
#define INSTANTIATE_CalcCCMatrices64_SWFTscanExtended(tpREADCNST) \
    template \
    __global__ void CalcCCMatrices64_SWFTscanExtended<tpREADCNST>( \
        const uint nqystrs, const uint ndbCstrs, const uint ndbCposs, const uint dbxpad, \
        const uint maxnsteps, \
        const float* __restrict__ tmpdpalnpossbuffer, \
        const float* __restrict__ tmpdpdiagbuffers, \
        const float* __restrict__ wrkmemaux, \
        float* __restrict__ wrkmem);

INSTANTIATE_CalcCCMatrices64_SWFTscanExtended(READCNST_CALC);
INSTANTIATE_CalcCCMatrices64_SWFTscanExtended(READCNST_CALC2);

// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// CalcScoresUnrl_SWFTscan: calculate/reduce scores for obtained 
// superpositions; version for alignments obtained by exhaustively applying 
// a linear algorithm; 
// NOTE: save partial sums;
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions;
// NOTE: Universal version for any CUS1_TBSP_SCORE_XDIM multiple of 32;
// SAVEPOS, template parameter to request saving positional scores;
// CHCKALNLEN, template parameter for checking whether alignment length has 
// changed;
// nqystrs, total number of queries;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db structure positions in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// NOTE: memory pointers should be aligned!
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// wrkmemtm, working memory of transformation matrices;
// wrkmem, working memory for cross-covariance data;
// wrkmemaux, auxiliary working memory;
// tmpdpdiagbuffers, temporary diagonal buffers reused here for saving 
// positional scores;
// NOTE: keep #registers <= 32
// 
template<int SAVEPOS, int CHCKALNLEN>
__global__
void CalcScoresUnrl_SWFTscan(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint /*dbxpad*/,
    const uint maxnsteps,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ wrkmemtm,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers)
{
    // blockIdx.x is the block index of positions for query-reference pair;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number TIMES fragment factor;
    //no bank conflicts as long as inner-most dim is odd
    enum{pad = 1};//padding
    //cache for scores and transformation matrix: 
    __shared__ float scvCache[pad + CUS1_TBSP_SCORE_XDIM + nTTranformMatrix];
    //pointer to transformation matrix;
    float* tfmCache = scvCache + pad + CUS1_TBSP_SCORE_XDIM;
    //relative position index:
    const uint ndx0 = blockIdx.x * blockDim.x * CUS1_TBSP_SCORE_XFCT;
    const uint ndx = ndx0 + threadIdx.x;
    const uint sfragfct = blockIdx.z / nqystrs;//fragment factor
    const uint qryndx = blockIdx.z - sfragfct * nqystrs;//query serial number
    const uint dbstrndx = blockIdx.y;
    int qrylenorg, dbstrlenorg;//original query and reference lengths
    int qrylen, dbstrlen;//pseudo query and reference length, #matched positions
    //distances in positions to the beginnings of the query and reference structures:
    uint /*qrydst, */dbstrdst;
    enum{qrypos = 0, rfnpos = 0};


#if (COVARIANCE_SWIFT_SCAN_CHECK_CONVERGENCE == 1)
    if(threadIdx.x == 0) {
        //NOTE: reuse ccmCache to read convergence flag at sfragfct==0:
        uint mloc = ((qryndx * maxnsteps + 0/*sfragfct*/) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        scvCache[6] = wrkmemaux[mloc + dbstrndx];
    }

    __syncthreads();

    if(scvCache[6])
        //(NOTE:any type of convergence applies);
        //all threads in the block exit;
        return;

    //NOTE: no sync as long as scvCache cell for convergence is not overwritten;
#endif


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse scvCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(dbstrndx, (int*)scvCache);
        GetQueryLenDst(qryndx, (int*)scvCache + 2);
    }
    if(threadIdx.x == tawmvNAlnPoss + 32) {
        //NOTE: reuse ccmCache to read #matched positions (tawmvNAlnPoss) written at sfragfct;
        //NOTE: use different warp; structure-specific-formatted data;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        scvCache[threadIdx.x] = wrkmemaux[mloc + (threadIdx.x-32) * ndbCstrs + dbstrndx];
    }

    __syncthreads();

    //NOTE: no bank conflict when two threads from the same warp access the same address;
    //blockDim.x includes several warps
    dbstrlenorg = ((int*)scvCache)[0]; dbstrdst = ((int*)scvCache)[1];
    qrylenorg = ((int*)scvCache)[2]; //qrydst = ((int*)scvCache)[3];
    qrylen = dbstrlen = scvCache[tawmvNAlnPoss+32];

    __syncthreads();


    if(qrylen <= qrypos + ndx0 || dbstrlen <= rfnpos + ndx0)
        //all threads in the block exit if thread 0 is out of bounds
        return;


    if(CHCKALNLEN == CHCKALNLEN_CHECK) {
        if(threadIdx.x == 0) {
            //NOTE: reuse scvCache[0] for twmvNalnposs
            uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTWorkingMemoryVars;
            scvCache[0] = wrkmem[mloc + twmvNalnposs];
        }

        __syncthreads();

        int nalnposs = scvCache[0];
        if(nalnposs == dbstrlen)
            //dbstrlen is #originally aligned positions;
            //score has been calculated before; 
            //all threads in the block exit;
            return;
        //no sync as scvCache[0(pad-1)] is not used below
    }


    //threshold calculated for the original lengths
    float d02 = GetD02(qrylenorg, dbstrlenorg);

    //initialize cache
    scvCache[pad + threadIdx.x] = 0.0f;

    //read transformation matrix for query-reference pair
    if(threadIdx.x < nTTranformMatrix) {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix;
        tfmCache[threadIdx.x] = wrkmemtm[mloc + threadIdx.x];
    }

    __syncthreads();


    const int dblen = ndbCposs/* + dbxpad*/;
    //offset to the beginning of the data along the y axis 
    // wrt query qryndx and maxnsteps: 
    const int yofff = (qryndx * maxnsteps + sfragfct) * dblen * nTDPAlignedPoss;

    for(int i = 0; i < CUS1_TBSP_SCORE_XFCT; i++) {
        //manually unroll along data blocks by a factor of CUS1_TBSP_SCORE_XFCT;
        //x2 to account for scores and query positions:
        int mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs * 2;
        int pos0 = ndx + i * blockDim.x;//position index starting from 0
        if(!(qrypos + pos0 < qrylen && rfnpos + pos0 < dbstrlen))
            break;
        //starting position in tmpdpalnpossbuffer for a pair:
        //NOTE: qrypos == rfnpos as well as qrylen == dbstrlen here
        int dppos = yofff + dbstrdst + (rfnpos + pos0);
        UpdateOneAlnPosScore_SWFTRefined<SAVEPOS,CHCKDST_NOCHECK>(//no sync;
            d02, d02,
            dppos, dblen,
            mloc + dbstrdst + pos0,//position for scores
            tmpdpalnpossbuffer,//coordinates
            tfmCache,//tfm. mtx.
            scvCache + pad,//score cache
            tmpdpdiagbuffers//scores written to gmem
        );
    }

    //sync now:
    __syncthreads();

    //unroll until reaching warpSize 
    #pragma unroll
    for(int xdim = (CUS1_TBSP_SCORE_XDIM>>1); xdim >= 32; xdim >>= 1) {
        if(threadIdx.x < xdim)
            scvCache[pad + threadIdx.x] +=
                scvCache[pad + threadIdx.x + xdim];

        __syncthreads();
    }

    //unroll warp
    if(threadIdx.x < 32/*warpSize*/) {
        float sum = scvCache[pad + threadIdx.x];
        sum = mywarpreducesum(sum);
        //write to the first data slot of SMEM
        if(threadIdx.x == 0) scvCache[0] = sum;
    }

    //add the score and write to global memory
    if(threadIdx.x == 0) {
        //structure-specific-formatted data; scvCache[0] is the reduced score
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvScore) * ndbCstrs;
        atomicAdd(&wrkmemaux[mloc + dbstrndx], scvCache[0]);
    }
}

// -------------------------------------------------------------------------
// Instantiations
// 
#define INSTANTIATE_CalcScoresUnrl_SWFTscan(tpSAVEPOS,tpCHCKALNLEN) \
    template __global__ void CalcScoresUnrl_SWFTscan<tpSAVEPOS,tpCHCKALNLEN>( \
        const uint nqystrs, const uint ndbCstrs, const uint ndbCposs, const uint dbxpad, \
        const uint maxnsteps, \
        const float* __restrict__ tmpdpalnpossbuffer, \
        const float* __restrict__ wrkmemtm, \
        const float* __restrict__ wrkmem, \
        float* __restrict__ wrkmemaux, \
        float* __restrict__ tmpdpdiagbuffers);

INSTANTIATE_CalcScoresUnrl_SWFTscan(SAVEPOS_SAVE,CHCKALNLEN_NOCHECK);
INSTANTIATE_CalcScoresUnrl_SWFTscan(SAVEPOS_SAVE,CHCKALNLEN_CHECK);
INSTANTIATE_CalcScoresUnrl_SWFTscan(SAVEPOS_NOSAVE,CHCKALNLEN_CHECK);

// -------------------------------------------------------------------------





// -------------------------------------------------------------------------
// CalcScoresUnrl_SWFTscanProgressive: calculate/reduce scores for obtained 
// superpositions progressively so that alignment increasing in 
// positions is ensured; version for alignments obtained by exhaustively 
// applying a linear algorithm; 
// NOTE: save partial sums;
// NOTE: thread block is 1D and processes alignment fragment along structure
// positions; preffered as long as possible;
// NOTE: Universal version for any CUS1_TBSP_SCORE_XDIM multiple of 32;
// SAVEPOS, template parameter to request saving positional scores;
// CHCKALNLEN, template parameter for checking whether alignment length has 
// changed;
// nqystrs, total number of queries;
// ndbCstrs, total number of reference structures in the chunk;
// ndbCposs, total number of db structure positions in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// NOTE: memory pointers should be aligned!
// tmpdpalnpossbuffer, coordinates of matched positions obtained by DP;
// wrkmemtm, working memory of transformation matrices;
// wrkmem, working memory for cross-covariance data;
// wrkmemaux, auxiliary working memory;
// tmpdpdiagbuffers, temporary diagonal buffers reused here for saving 
// positional scores;
// NOTE: keep #registers <= 32
// 
template<int SAVEPOS, int CHCKALNLEN>
__global__
void CalcScoresUnrl_SWFTscanProgressive(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint /*dbxpad*/,
    const uint maxnsteps,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ wrkmemtm,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpdiagbuffers)
{
    // blockIdx.x is the block index of positions for query-reference pair;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number TIMES fragment factor;
    //no bank conflicts as long as inner-most dim is odd
    enum{   pad = 1,//padding
            xfct = CUS1_TBSP_SCORE_XFCT,//unrolling factor in the x dimension
            xdim = CUS1_TBSP_SCORE_XDIM,//x-dimension of the thread block
            szqnxch = CUSF2_TBSP_INDEX_SCORE_QNX_CACHE_SIZE,//cache size
            nwrpsdim = (xdim>>5),//XDIM/warpsize(2^5)
    };
    __shared__ float tmpSMbuf[nwrpsdim];
    //cache for query positions: 
    __shared__ float qnxCache[pad + szqnxch];
    //cache for maximum scores over all query positions: 
    __shared__ float maxCache[pad + szqnxch];
    //working cache for query positions: 
    __shared__ float qnxWrkch[pad + xdim];
    //cache for scores and transformation matrix: 
    __shared__ float scvCache[pad + xdim + nTTranformMatrix];
    //pointer to transformation matrix;
    float* tfmCache = scvCache + pad + xdim;
    //relative position index:
    const uint ndx0 = blockIdx.x * blockDim.x * xfct;
    const uint ndx = ndx0 + threadIdx.x;
    const uint sfragfct = blockIdx.z / nqystrs;//fragment factor
    const uint qryndx = blockIdx.z - sfragfct * nqystrs;//query serial number
    const uint dbstrndx = blockIdx.y;
    int qrylenorg, dbstrlenorg;//original query and reference lengths
    int qrylen, dbstrlen;//pseudo query and reference length, #matched positions
    //distances in positions to the beginnings of the query and reference structures:
    uint /*qrydst, */dbstrdst;
    enum{qrypos = 0, rfnpos = 0};


#if (COVARIANCE_SWIFT_SCAN_CHECK_CONVERGENCE == 1)
    if(threadIdx.x == 0) {
        //NOTE: reuse scvCache to read convergence flag at both 0 and sfragfct:
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        scvCache[6] = wrkmemaux[mloc0 + dbstrndx];
    }
    if(threadIdx.x == 32) {//next warp
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        scvCache[7] = wrkmemaux[mloc + dbstrndx];
    }

    __syncthreads();

    if((((int)(scvCache[6])) & (CONVERGED_LOWTMSC_bitval)) || scvCache[7])
        //(NOTE:any type of convergence applies locally);
        //all threads in the block exit;
        return;

    //NOTE: no sync as long as scvCache cells for convergence not overwritten;
#endif


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse scvCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(dbstrndx, (int*)scvCache);
        GetQueryLenDst(qryndx, (int*)scvCache + 2);
    }
    if(threadIdx.x == tawmvNAlnPoss + 32) {
        //NOTE: reuse ccmCache to read #matched positions (tawmvNAlnPoss) written at sfragfct;
        //NOTE: use different warp; structure-specific-formatted data;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        scvCache[threadIdx.x] = wrkmemaux[mloc + (threadIdx.x-32) * ndbCstrs + dbstrndx];
    }

    __syncthreads();

    //NOTE: no bank conflict when two threads from the same warp access the same address;
    //blockDim.x includes several warps
    dbstrlenorg = ((int*)scvCache)[0]; dbstrdst = ((int*)scvCache)[1];
    qrylenorg = ((int*)scvCache)[2]; //qrydst = ((int*)scvCache)[3];
    qrylen = dbstrlen = scvCache[tawmvNAlnPoss+32];

    __syncthreads();


    if(qrylen <= qrypos + ndx0 || dbstrlen <= rfnpos + ndx0)
        //all threads in the block exit if thread 0 is out of bounds
        return;


    if(CHCKALNLEN == CHCKALNLEN_CHECK) {
        if(threadIdx.x == 0) {
            //NOTE: reuse scvCache[0] for twmvNalnposs
            uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTWorkingMemoryVars;
            scvCache[0] = wrkmem[mloc + twmvNalnposs];
        }

        __syncthreads();

        int nalnposs = scvCache[0];
        if(nalnposs == dbstrlen)
            //dbstrlen is #originally aligned positions;
            //score has been calculated before; 
            //all threads in the block exit;
            return;
        //no sync as scvCache[0(pad-1)] is not used below
    }


    //threshold calculated for the original lengths
    float d02 = GetD02(qrylenorg, dbstrlenorg);

    //read transformation matrix for query-reference pair
    if(threadIdx.x < nTTranformMatrix) {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + dbstrndx) * nTTranformMatrix;
        tfmCache[threadIdx.x] = wrkmemtm[mloc + threadIdx.x];
    }

    //initialize cache:
    for(int i = threadIdx.x; i < szqnxch; i += xdim) {
        qnxCache[pad+i] = -1.0f;
        maxCache[pad+i] = 0.0f;
    }


    __syncthreads();


    const int dblen = ndbCposs/* + dbxpad*/;
    //offset to the beginning of the data along the y axis 
    // wrt query qryndx and maxnsteps: 
    const int yofff = (qryndx * maxnsteps + sfragfct) * dblen * nTDPAlignedPoss;

    for(int i = 0; i < xfct; i++) {
        //manually unroll along data blocks by a factor of xfct;
        //x2 to account for scores and query positions:
        int mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs * 2;
        int pos0 = ndx + i * blockDim.x;//position index starting from 0
        int pos00 = ndx0 + i * blockDim.x;//thread-0 position index
        //starting position in tmpdpalnpossbuffer for a pair:
        //NOTE: qrypos == rfnpos as well as qrylen == dbstrlen here
        int dppos = yofff + dbstrdst + (rfnpos + pos0);

        if(dbstrlen <= rfnpos + pos00) break;

        if(rfnpos + pos0 < dbstrlen) {
            CacheAlnPosScore_SWFTProgressive<SAVEPOS>(
                d02,  dppos, dblen,
                mloc + dbstrdst + pos0,//position for scores
                tmpdpalnpossbuffer,//coordinates
                tfmCache,//tfm. mtx.
                scvCache + pad,//score cache
                qnxWrkch + pad,//query position cache
                tmpdpdiagbuffers//scores written to gmem
            );
        }

        __syncthreads();

        //process xdim reference positions to pregressively find max score
        for(int p = 0; p < xdim && rfnpos + pos00 + p < dbstrlen; p++) {
            float sco = scvCache[pad+p];
            float qnx = qnxWrkch[pad+p];

            if(sco <= 0.0f || qnx < 0.0f) continue;

            //find max score up to position qnx:
            float max =
                FindMax_SWFTProgressive<pad,xdim,szqnxch,nwrpsdim,false/*uncnd*/>(
                    qnx, qnxCache, maxCache, tmpSMbuf);

            //save max score to cache:
            if(threadIdx.x == 0) {
                //extremely simple hash function for the cache index:
                int c = (int)(qnx) & (szqnxch-1);
                float stqnx = qnxCache[pad+c];//stored query position
                float stsco = maxCache[pad+c];//stored score
                float newsco = max + sco;//new score
                bool half2nd = (rfnpos + pos00 + p > (dbstrlen>>1));
                //heuristics: under hash collision, update position and 
                //score wrt to which reference half is under process:
                if(stqnx < 0.0f ||
                  (stqnx == qnx && stsco < newsco) ||
                  ((half2nd && stqnx < qnx) || (!half2nd && qnx < stqnx))) {
                    qnxCache[pad+c] = qnx;
                    maxCache[pad+c] = newsco;
                }
            }

            __syncthreads();
        }
    }

    //find max score over all query positions:
    float max =
        FindMax_SWFTProgressive<pad,xdim,szqnxch,nwrpsdim,true/*uncnd*/>(
            0.0f, qnxCache, maxCache, tmpSMbuf);

    //add the score and write to global memory;
    //NOTE: heuristics: scores are calculated for data blocks of 
    //NOTE: size xdim*xfct independently;
    if(threadIdx.x == 0) {
        //structure-specific-formatted data;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + tawmvScore) * ndbCstrs;
        atomicAdd(&wrkmemaux[mloc + dbstrndx], max);
    }
}

// -------------------------------------------------------------------------
// Instantiations
// 
#define INSTANTIATE_CalcScoresUnrl_SWFTscanProgressive(tpSAVEPOS,tpCHCKALNLEN) \
    template __global__ void CalcScoresUnrl_SWFTscanProgressive<tpSAVEPOS,tpCHCKALNLEN>( \
        const uint nqystrs, const uint ndbCstrs, const uint ndbCposs, const uint dbxpad, \
        const uint maxnsteps, \
        const float* __restrict__ tmpdpalnpossbuffer, \
        const float* __restrict__ wrkmemtm, \
        const float* __restrict__ wrkmem, \
        float* __restrict__ wrkmemaux, \
        float* __restrict__ tmpdpdiagbuffers);

INSTANTIATE_CalcScoresUnrl_SWFTscanProgressive(SAVEPOS_SAVE,CHCKALNLEN_NOCHECK);
INSTANTIATE_CalcScoresUnrl_SWFTscanProgressive(SAVEPOS_SAVE,CHCKALNLEN_CHECK);
INSTANTIATE_CalcScoresUnrl_SWFTscanProgressive(SAVEPOS_NOSAVE,CHCKALNLEN_CHECK);
INSTANTIATE_CalcScoresUnrl_SWFTscanProgressive(SAVEPOS_NOSAVE,CHCKALNLEN_NOCHECK);

// -------------------------------------------------------------------------
