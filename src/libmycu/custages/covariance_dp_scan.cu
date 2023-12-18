/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

// All kernels in this module designed to process covariance and related 
// data following the application of DP and obtaining of matched positions

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
#include "covariance_dp_scan.cuh"

// -------------------------------------------------------------------------
// CopyCCDataToWrkMem2_DPscan: copy cross-covariance matrix between the 
// query and reference structures to section 2 to enable efficient Kabsch 
// algorithm application for multiple structures simultaneously; 
// Version for alignments obtained as a result of the application of DP;
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
void CopyCCDataToWrkMem2_DPscan(
    const uint ndbCstrs,
    const uint maxnsteps,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ wrkmem,
    float* __restrict__ wrkmem2)
{
    constexpr int ndxproc = twmvEndOfCCDataExt;//index of processing flag in cache
    //cache for cross-covariance matrices and related data: 
    //bank conflicts resolved as long as innermost dim is odd
    __shared__ float ccmCache[CUS1_TBINITSP_CCMCOPY_N][twmvEndOfCCDataExt+1];
    //index of the first structure to start with (blockIdx.x, refn. serial number):
    int dbstrndx = blockIdx.x * CUS1_TBINITSP_CCMCOPY_N;
    int qryndx = blockIdx.y;//query index in the chunk
    int sfragfct = blockIdx.z;//fragment factor
    int absndx = dbstrndx + threadIdx.x;
    //int nalnposs = 0;

    if(absndx < ndbCstrs && 
       (threadIdx.y == tawmvConverged || threadIdx.y == tawmvNAlnPoss))
    {
        uint mloc = //tawmvConverged, convergence flag for a pair at sfragfct==0:
            (threadIdx.y == tawmvConverged)
            ? ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + threadIdx.y) * ndbCstrs
            : ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars + threadIdx.y) * ndbCstrs;
        ccmCache[threadIdx.x][threadIdx.y] = wrkmemaux[mloc + absndx/*dbstrndx*/];
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
#define INSTANTIATE_CopyCCDataToWrkMem2_DPscan(tpREADNPOS) \
    template __global__ void CopyCCDataToWrkMem2_DPscan<tpREADNPOS>( \
        const uint ndbCstrs, const uint maxnsteps, \
        const float* __restrict__ wrkmemaux, \
        const float* __restrict__ wrkmem, \
        float* __restrict__ wrkmem2);

INSTANTIATE_CopyCCDataToWrkMem2_DPscan(READNPOS_NOREAD);
INSTANTIATE_CopyCCDataToWrkMem2_DPscan(READNPOS_READ);

// -------------------------------------------------------------------------
// CalcCCMatrices64_DPscan: calculate cross-covariance matrix between the 
// query and reference structures given an alignment between them;
// Version for alignments obtained as a result of the application of DP;
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
void CalcCCMatrices64_DPscan(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ tmpdpalnpossbuffer,
    float* __restrict__ wrkmem)
{
    // blockIdx.x is the block index of positions for query-reference pair;
    // blockIdx.y is the reference serial number;
    // blockIdx.z is the query serial number TIMES fragment factor;
    //cache for the cross-covarinace matrix and related data: 
    //no bank conflicts as long as twmvEndOfCCData is odd
    __shared__ float ccmCache[twmvEndOfCCData * CUS1_TBINITSP_CCMCALC_XDIM];
    //relative position index:
    const uint ndx0 = blockIdx.x * blockDim.x * CUS1_TBINITSP_CCMCALC_XFCT;
    const uint ndx = ndx0 + threadIdx.x;
    uint sfragfct = blockIdx.z / nqystrs;//fragment factor
    uint qryndx = blockIdx.z - sfragfct * nqystrs;//query serial number
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    uint /*qrydst, */dbstrdst;
    const int qrypos = 0, rfnpos = 0;


    if(threadIdx.x == 0) {
        //NOTE: reuse ccmCache to read convergence flag at sfragfct==0:
        uint mloc = ((qryndx * maxnsteps + 0/*sfragfct*/) * nTAuxWorkingMemoryVars) * ndbCstrs;
        ccmCache[6] = wrkmemaux[mloc + tawmvConverged * ndbCstrs + blockIdx.y/*dbstrndx*/];
    }

    __syncthreads();

    if(ccmCache[6])
        //(NOTE:any type of convergence applies);
        //all threads in the block exit;
        return;

    //NOTE: no sync as long as ccmCache cell for convergence is not overwritten;


    //reuse ccmCache
    if(threadIdx.x == 0) {
        ((int*)ccmCache)[1] = GetDbStrDst(blockIdx.y);
        //((int*)ccmCache)[3] = GetQueryDst(qryndx);
    }
    if(threadIdx.x == tawmvNAlnPoss + 32) {
        //NOTE: reuse ccmCache to read #matched positions (tawmvNAlnPoss) written at sfragfct;
        //NOTE: use different warp; structure-specific-formatted data;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        ccmCache[threadIdx.x] = wrkmemaux[mloc + (threadIdx.x-32) * ndbCstrs + blockIdx.y/*dbstrndx*/];
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
    InitCCMCacheExtended<twmvEndOfCCData,0,twmvEndOfCCData>(ccmCache);


    const int dblen = ndbCposs + dbxpad;
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
        //NOTE: qrypos == rfnpos as well as lengths: use qrylen as the 
        //NOTE: original alignment length here;
        //NOTE: alignment written in reverse order:
        int pos = yofff + dbstrdst + qrylen-1 - (rfnpos + ndx + i * blockDim.x);
        UpdateCCMOneAlnPos_DPRefined(//no sync;
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
        for(int i = 0; i < twmvEndOfCCData; i++)
            ccmCache[threadIdx.x * twmvEndOfCCData +i] +=
                ccmCache[(threadIdx.x + (CUS1_TBINITSP_CCMCALC_XDIM>>1)) * twmvEndOfCCData +i];
    }

    __syncthreads();

    //unroll warp
    if(threadIdx.x < 32) {
        #pragma unroll
        for(int i = 0; i < twmvEndOfCCData; i++) {
            float sum = ccmCache[threadIdx.x * twmvEndOfCCData + i];
            sum = mywarpreducesum(sum);
            //write to the first data slot of SMEM
            if(threadIdx.x == 0) ccmCache[i] = sum;
        }
    }

    //in case of twmvEndOfCCData gets larger than warpSize
    __syncthreads();

    uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + blockIdx.y/*dbstrndx*/) * nTWorkingMemoryVars;

    //only one block and its one thread writes nalnposs
    if(blockIdx.x == 0 && threadIdx.x == 0)
        wrkmem[mloc + twmvNalnposs] = dbstrlen;

    //add the result and write to global memory
    if(threadIdx.x < twmvEndOfCCData)
        atomicAdd(&wrkmem[mloc + threadIdx.x], ccmCache[threadIdx.x]);
}

// -------------------------------------------------------------------------
// FindD02ThresholdsCCM_DPscan: efficiently find distance thresholds 
// for the inclusion of aligned positions for CCM and rotation matrix 
// calculations during exhaustive application of DP;
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
void FindD02ThresholdsCCM_DPscan(
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
    constexpr int smidim = 3;//top three min scores
    __shared__ float ccmCache[smidim * CUS1_TBINITSP_FINDD02_ITRD_XDIM];
    uint qryndx = blockIdx.y;//query serial number
    uint sfragfct = blockIdx.z;//fragment factor
    int qrylenorg, dbstrlenorg;//original query and reference lengths
    int qrylen, dbstrlen;//pseudo query and reference length, #matched positions
    //distances in positions to the beginnings of the query and reference structures:
    uint /*qrydst, */dbstrdst;
    constexpr int qrypos = 0;
    constexpr int rfnpos = 0;


    if(threadIdx.x == 0) {
        //NOTE: reuse ccmCache to read convergence flag at sfragfct==0:
        uint mloc = ((qryndx * maxnsteps + 0/*sfragfct*/) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        ccmCache[6] = wrkmemaux[mloc + blockIdx.x/*dbstrndx*/];
    }

    __syncthreads();

    if(ccmCache[6])
        //(NOTE:any type of convergence applies);
        //all threads in the block exit;
        return;

    //NOTE: no sync as long as ccmCache cell for convergence is not overwritten;


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse ccmCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(blockIdx.x, (int*)ccmCache);
        //GetQueryLenDst(qryndx, (int*)ccmCache + 2);
        if(threadIdx.x == 0) ((int*)ccmCache)[2] = GetQueryLength(qryndx);
    }
    if(threadIdx.x == tawmvNAlnPoss + 32) {
        //NOTE: reuse ccmCache to read #matched positions (tawmvNAlnPoss) written at sfragfct;
        //NOTE: use different warp; structure-specific-formatted data;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        ccmCache[threadIdx.x] = wrkmemaux[mloc + (threadIdx.x-32) * ndbCstrs + blockIdx.x/*dbstrndx*/];
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
            uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + blockIdx.x/*dbstrndx*/) * nTWorkingMemoryVars;
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
        //manually unroll along alignment
        int mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs;
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
        wrkmemaux[mloc + blockIdx.x/*dbstrndx*/] = min3;
    }
}

// =========================================================================
// Instantiations
//
#define INSTANTIATE_FindD02ThresholdsCCM_DPscan(tpREADCNST) \
    template \
    __global__ void FindD02ThresholdsCCM_DPscan<tpREADCNST>( \
        const uint ndbCstrs, const uint ndbCposs, const uint maxnsteps, \
        const float* __restrict__ tmpdpdiagbuffers, \
        const float* __restrict__ wrkmem, \
        float* __restrict__ wrkmemaux);

INSTANTIATE_FindD02ThresholdsCCM_DPscan(READCNST_CALC);
INSTANTIATE_FindD02ThresholdsCCM_DPscan(READCNST_CALC2);

// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// CalcCCMatrices64_DPscanExtended: calculate cross-covariance matrix 
// between the query and reference structures based on aligned positions 
// within given distance;
// Version for alignments obtained as a result of the exhaustive application of DP;
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
void CalcCCMatrices64_DPscanExtended(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
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
    //no bank conflicts as long as inner-most dim is odd
    constexpr int neffds = twmvEndOfCCDataExt;//effective number of fields
    constexpr int smidim = neffds+1;
    __shared__ float ccmCache[smidim * CUS1_TBINITSP_CCMCALC_XDIM];
    //relative position index:
    const uint ndx0 = blockIdx.x * blockDim.x * CUS1_TBINITSP_CCMCALC_XFCT;
    const uint ndx = ndx0 + threadIdx.x;
    uint sfragfct = blockIdx.z / nqystrs;//fragment factor
    uint qryndx = blockIdx.z - sfragfct * nqystrs;//query serial number
    int qrylen, dbstrlen;//pseudo query and reference length, #matched positions
    //distances in positions to the beginnings of the query and reference structures:
    uint /*qrydst, */dbstrdst;
    constexpr int qrypos = 0;
    constexpr int rfnpos = 0;


    if(threadIdx.x == 0) {
        //NOTE: reuse ccmCache to read convergence flag at sfragfct==0:
        uint mloc = ((qryndx * maxnsteps + 0/*sfragfct*/) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        ccmCache[6] = wrkmemaux[mloc + blockIdx.y/*dbstrndx*/];
    }

    __syncthreads();

    if(ccmCache[6])
        //(NOTE:any type of convergence applies);
        //all threads in the block exit;
        return;

    //NOTE: no sync as long as ccmCache cell for convergence is not overwritten;


    //reuse ccmCache
    if(threadIdx.x == 0) {
        ((int*)ccmCache)[1] = GetDbStrDst(blockIdx.y);
        //((int*)ccmCache)[3] = GetQueryDst(qryndx);
    }
    if(threadIdx.x == tawmvNAlnPoss + 32) {
        //NOTE: reuse ccmCache to read #matched positions (tawmvNAlnPoss) written at sfragfct;
        //NOTE: use different warp; structure-specific-formatted data;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        ccmCache[threadIdx.x] = wrkmemaux[mloc + (threadIdx.x-32) * ndbCstrs + blockIdx.y/*dbstrndx*/];
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
        ccmCache[0] = wrkmemaux[mloc + blockIdx.y/*dbstrndx*/];

        if(READCNST == READCNST_CALC2) {
            mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + blockIdx.y/*dbstrndx*/) * nTWorkingMemoryVars;
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

    const int dblen = ndbCposs + dbxpad;
    //offset to the beginning of the data along the y axis 
    // wrt query qryndx and maxnsteps: 
    const int yofff = (qryndx * maxnsteps + sfragfct) * dblen * nTDPAlignedPoss;

    for(int i = 0; i < CUS1_TBINITSP_CCMCALC_XFCT; i++) {
        //manually unroll along data blocks by a factor of CUS1_TBINITSP_CCMCALC_XFCT
        int mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs;
        int pos0 = ndx + i * blockDim.x;//position index starting from 0
        if(!(qrypos + pos0 < qrylen && rfnpos + pos0 < dbstrlen))
            break;
        //starting position in tmpdpalnpossbuffer for a pair:
        //NOTE: aligned coordinates in tmpdpalnpossbuffer are in the reverse order!
        //NOTE: qrypos == rfnpos as well as qrylen == dbstrlen here
        int dppos = yofff + dbstrdst + dbstrlen-1 - (rfnpos + pos0);
        UpdateCCMOneAlnPos_DPExtended<smidim>(//no sync;
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

    //in case of neffds gets larger than warpSize
    __syncthreads();

    //add the result and write to global memory
    if(threadIdx.x < neffds) {
        uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + blockIdx.y/*dbstrndx*/) * nTWorkingMemoryVars;
        atomicAdd(&wrkmem[mloc + threadIdx.x], ccmCache[threadIdx.x]);
    }
}

// =========================================================================
// Instantiations
//
#define INSTANTIATE_CalcCCMatrices64_DPscanExtended(tpREADCNST) \
    template \
    __global__ void CalcCCMatrices64_DPscanExtended<tpREADCNST>( \
        const uint nqystrs, const uint ndbCstrs, const uint ndbCposs, const uint dbxpad, \
        const uint maxnsteps, \
        const float* __restrict__ tmpdpalnpossbuffer, \
        const float* __restrict__ tmpdpdiagbuffers, \
        const float* __restrict__ wrkmemaux, \
        float* __restrict__ wrkmem);

INSTANTIATE_CalcCCMatrices64_DPscanExtended(READCNST_CALC);
INSTANTIATE_CalcCCMatrices64_DPscanExtended(READCNST_CALC2);

// -------------------------------------------------------------------------



// -------------------------------------------------------------------------
// CalcScoresUnrl_DPscan: calculate/reduce scores for obtained 
// superpositions; version for alignments obtained by exhaustively applying 
// DP; 
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
void CalcScoresUnrl_DPscan(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
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
    constexpr int pad = 1;//padding
    //cache for scores and transformation matrix: 
    __shared__ float scvCache[pad + CUS1_TBSP_SCORE_XDIM + nTTranformMatrix];
    //pointer to transformation matrix;
    float* tfmCache = scvCache + pad + CUS1_TBSP_SCORE_XDIM;
    //relative position index:
    const uint ndx0 = blockIdx.x * blockDim.x * CUS1_TBSP_SCORE_XFCT;
    const uint ndx = ndx0 + threadIdx.x;
    uint sfragfct = blockIdx.z / nqystrs;//fragment factor
    uint qryndx = blockIdx.z - sfragfct * nqystrs;//query serial number
    int qrylenorg, dbstrlenorg;//original query and reference lengths
    int qrylen, dbstrlen;//pseudo query and reference length, #matched positions
    //distances in positions to the beginnings of the query and reference structures:
    uint /*qrydst, */dbstrdst;
    constexpr int qrypos = 0;
    constexpr int rfnpos = 0;


    if(threadIdx.x == 0) {
        //NOTE: reuse ccmCache to read convergence flag at sfragfct==0:
        uint mloc = ((qryndx * maxnsteps + 0/*sfragfct*/) * nTAuxWorkingMemoryVars + tawmvConverged) * ndbCstrs;
        scvCache[6] = wrkmemaux[mloc + blockIdx.y/*dbstrndx*/];
    }

    __syncthreads();

    if(scvCache[6])
        //(NOTE:any type of convergence applies);
        //all threads in the block exit;
        return;

    //NOTE: no sync as long as scvCache cell for convergence is not overwritten;


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse scvCache
    if(threadIdx.x < 2) {
        GetDbStrLenDst(blockIdx.y, (int*)scvCache);
        GetQueryLenDst(qryndx, (int*)scvCache + 2);
    }
    if(threadIdx.x == tawmvNAlnPoss + 32) {
        //NOTE: reuse ccmCache to read #matched positions (tawmvNAlnPoss) written at sfragfct;
        //NOTE: use different warp; structure-specific-formatted data;
        uint mloc = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
        scvCache[threadIdx.x] = wrkmemaux[mloc + (threadIdx.x-32) * ndbCstrs + blockIdx.y/*dbstrndx*/];
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
            uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + blockIdx.y/*dbstrndx*/) * nTWorkingMemoryVars;
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
        uint mloc = ((qryndx * maxnsteps + sfragfct) * ndbCstrs + blockIdx.y/*dbstrndx*/) * nTTranformMatrix;
        tfmCache[threadIdx.x] = wrkmemtm[mloc + threadIdx.x];
    }

    __syncthreads();


    const int dblen = ndbCposs + dbxpad;
    //offset to the beginning of the data along the y axis 
    // wrt query qryndx and maxnsteps: 
    const int yofff = (qryndx * maxnsteps + sfragfct) * dblen * nTDPAlignedPoss;

    #pragma unroll
    for(int i = 0; i < CUS1_TBSP_SCORE_XFCT; i++) {
        //manually unroll along data blocks by a factor of CUS1_TBSP_SCORE_XFCT
        int mloc = (qryndx * maxnsteps + sfragfct) * ndbCposs;
        int pos0 = ndx + i * blockDim.x;//position index starting from 0
        if(!(qrypos + pos0 < qrylen && rfnpos + pos0 < dbstrlen))
            break;
        //starting position in tmpdpalnpossbuffer for a pair:
        //NOTE: aligned coordinates in tmpdpalnpossbuffer are in the reverse order!
        //NOTE: qrypos == rfnpos as well as qrylen == dbstrlen here
        int dppos = yofff + dbstrdst + dbstrlen-1 - (rfnpos + pos0);
        UpdateOneAlnPosScore_DPRefined<SAVEPOS,CHCKDST_NOCHECK>(//no sync;
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
        atomicAdd(&wrkmemaux[mloc + blockIdx.y/*dbstrndx*/], scvCache[0]);
    }
}

// -------------------------------------------------------------------------
// Instantiations
// 
#define INSTANTIATE_CalcScoresUnrl_DPscan(tpSAVEPOS,tpCHCKALNLEN) \
    template __global__ void CalcScoresUnrl_DPscan<tpSAVEPOS,tpCHCKALNLEN>( \
        const uint nqystrs, const uint ndbCstrs, const uint ndbCposs, const uint dbxpad, \
        const uint maxnsteps, \
        const float* __restrict__ tmpdpalnpossbuffer, \
        const float* __restrict__ wrkmemtm, \
        const float* __restrict__ wrkmem, \
        float* __restrict__ wrkmemaux, \
        float* __restrict__ tmpdpdiagbuffers);

INSTANTIATE_CalcScoresUnrl_DPscan(SAVEPOS_SAVE,CHCKALNLEN_NOCHECK);
INSTANTIATE_CalcScoresUnrl_DPscan(SAVEPOS_SAVE,CHCKALNLEN_CHECK);
INSTANTIATE_CalcScoresUnrl_DPscan(SAVEPOS_NOSAVE,CHCKALNLEN_CHECK);

// -------------------------------------------------------------------------
