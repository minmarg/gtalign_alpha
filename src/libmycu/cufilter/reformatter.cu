/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fields.cuh"
#include "reformatter.cuh"

// -------------------------------------------------------------------------
// MakeDbCandidateList: make list of reference structure (database)
// candidates proceeding to stages of more detailed superposition search and
// refinement;
// nqystrs, total number of query structures in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// maxnsteps, max number of steps (blockIdx.z) to perform for each 
// reference structure;
// NOTE: memory pointers should be aligned!
// wrkmemaux, auxiliary working memory;
// globvarsbuf, memory of new indices and addresses of passing references;
// NOTE: thread block is 2D (y-dim=2 for indices and addresses) and
// NOTE: processes the reference structures over all queries for flags;
// 
__global__ void MakeDbCandidateList(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint maxnsteps,
    const float* __restrict__ wrkmemaux,
    uint* __restrict__ globvarsbuf)
{
    enum {
        //index for the previous prefix sum and padding
        PREPXS = 0,
        pad = 1,
        lXFLG = fdNewReferenceIndex,//index for reference structure convergence flags/new indices
        lXLEN = fdNewReferenceAddress,//index for reference structure convergence lengths/new addresses
        lYDIM = CUFL_MAKECANDIDATELIST_YDIM,
        lXDIM = CUFL_MAKECANDIDATELIST_XDIM
    };
    constexpr uint sfragfct = 0;//fragment factor
    __shared__ int strdata[lYDIM][lXDIM + pad];

    if(threadIdx.x == 0)
        strdata[threadIdx.y][PREPXS] = 0;

    for(uint dbstrndx0 = 0; dbstrndx0 < ndbCstrs; dbstrndx0 += blockDim.x)
    {
        //update the prefix sums originated from processing the last data block:
        if(threadIdx.x == 0 && dbstrndx0)
            strdata[threadIdx.y][PREPXS] = strdata[threadIdx.y][pad + lXDIM - 1];

        //NOTE: strdata is overwritten below, sync:
        __syncthreads();

        uint dbstrndx = dbstrndx0 + threadIdx.x;//reference index
        int value = 0;//convflag for lXFLG and dbstrlen for lXLEN

        //get convergence flags (over all queries)
        if(threadIdx.y == lXFLG && dbstrndx < ndbCstrs) {
            for(uint qryndx = 0; qryndx < nqystrs; qryndx++) {
                uint mloc0 = ((qryndx * maxnsteps + sfragfct) * nTAuxWorkingMemoryVars) * ndbCstrs;
                int lconv = wrkmemaux[mloc0 + tawmvConverged * ndbCstrs + dbstrndx];//float->int
                value += ((lconv & CONVERGED_LOWTMSC_bitval) != 0);
            }
        }
        if(threadIdx.y == lXFLG)
            //progressing structures have no convergence flags set for all queries
            strdata[threadIdx.y][pad + threadIdx.x] = value = (value < nqystrs);

        //get reference lengths
        if(threadIdx.y == lXLEN && dbstrndx < ndbCstrs)
            strdata[threadIdx.y][pad + threadIdx.x] = value = GetDbStrLength(dbstrndx);

        __syncthreads();

        //set reference lengths to 0 where convflag is set
        if(threadIdx.y == lXLEN && strdata[lXFLG][pad + threadIdx.x] == 0)
            strdata[threadIdx.y][pad + threadIdx.x] = value = 0;

        __syncthreads();

        //calculate inclusive (!) prefix sums for both flags, which then give indices,
        //and lengths for addresses:
        for(uint xdim = 1; xdim < blockDim.x; xdim <<= 1) {
            if(xdim <= threadIdx.x)
                value += strdata[threadIdx.y][pad + threadIdx.x - xdim];
            __syncthreads();
            if(xdim <= threadIdx.x)
                strdata[threadIdx.y][pad + threadIdx.x] = value;
            __syncthreads();
        }

        //correct the prefix sums by adding the previously obtained values:
        strdata[threadIdx.y][pad + threadIdx.x] += strdata[threadIdx.y][PREPXS];
        __syncthreads();

        //write to GMEM:
        if(dbstrndx < ndbCstrs) {
            uint mloc = threadIdx.y * ndbCstrs + dbstrndx;
            int valueprev = strdata[threadIdx.y][pad + threadIdx.x - 1];
            value = strdata[threadIdx.y][pad + threadIdx.x];
            //set to 0 for filtered-out structures:
            if(value == valueprev) value = valueprev = 0;
            //write adjusted addresses to gmem:
            if(threadIdx.y == lXLEN) value = valueprev;
            globvarsbuf[mloc] = value;
        }
        __syncthreads();
    }
}



// -------------------------------------------------------------------------
// ReformatStructureDataPartStore: reformat a reference database chunk to
// include candidates proceeding to stages of more detailed superposition
// search and refinement; this part corresponds to storing data to
// secondary (temporary) location first;
// nqystrs, total number of queries in the chunk;
// ndbCstrs, total number of reference structures in the chunk;
// maxndbCposs, max number of db structure positions in the chunk;
// maxnsteps, max number of steps allocated for each reference structure;
// ndbCstrs2, total number of selected reference structures;
// ndbCposs2, total number of positions of selected reference structures;
// dbstr1len2, length of the largest reference structure selected;
// it is used also for address adjustment;
// NOTE: memory pointers should be aligned!
// globvarsbuf, memory of new indices and addresses of selected references;
// wrkmemaux, auxiliary working memory;
// tfmmem, memory of transformation matrices;
// tmpdpdiagbuffers, temporary memory large enough (!) to contain all data to
// be copied from structure data, wrkmemaux, and tfmmem;
// NOTE: thread block is 1D and processes a fragment of each reference structure;
// NOTE: thread block's x-dimension assumed to be 32!
// 
__global__ void ReformatStructureDataPartStore(
    const uint nqystrs,
    const uint ndbCstrs,
    const uint maxndbCposs,
    const uint maxnsteps,
    // const uint ndbCstrs2,
    // const uint ndbCposs2,
    // const uint dbstr1len2,
    const uint* __restrict__ globvarsbuf,
    const float* __restrict__ wrkmemaux,
    const float* __restrict__ tfmmem,
    float* __restrict__ tmpdpdiagbuffers)
{
    enum {
        lXNDX = fdNewReferenceIndex,//index for reference structure convergence flags/new indices
        lXADD = fdNewReferenceAddress,//index for reference structure convergence lengths/new addresses
        lXDIM = CUFL_STORECANDIDATEDATA_XDIM
    };
    // blockIdx.x is the block index of positions for one reference;
    // blockIdx.y is the reference serial number;
    // const uint dbstrfrg = blockIdx.x;
    const uint dbstrndx = blockIdx.y;
    //relative position index:
    const uint pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint newndx, newaddr;
    INTYPE length;
    LNTYPE dbstrdst;//address;

    if(threadIdx.x == 0) {
        newndx = globvarsbuf[lXNDX * ndbCstrs + dbstrndx];
        newaddr = globvarsbuf[lXADD * ndbCstrs + dbstrndx];
    }

    newndx = __shfl_sync(0xffffffff, newndx, 0/*srcLane*/);
    newaddr = __shfl_sync(0xffffffff, newaddr, 0/*srcLane*/);

    if(newndx == 0) return;//all threads exit

    //adjust index appropriately (NOTE: newaddr is already valid):
    newndx--;

    if(threadIdx.x == 0) {
        length = GetDbStrField<INTYPE,pps2DLen>(dbstrndx);
        dbstrdst = GetDbStrField<LNTYPE,pps2DDist>(dbstrndx);
    }

    if(blockIdx.x == 0 && threadIdx.x == 0) {
        //read originals:
        INTYPE type = GetDbStrField<INTYPE,pps2DType>(dbstrndx);
        //write structure-specific fields to new gmem location:
        ((INTYPE*)(tmpdpdiagbuffers + pps2DLen * maxndbCposs))[newndx] = length;
        ((INTYPE*)(tmpdpdiagbuffers + pps2DType * maxndbCposs))[newndx] = type;
        ((LNTYPE*)(tmpdpdiagbuffers + pps2DDist * maxndbCposs))[newndx] = newaddr;
    }

    length = __shfl_sync(0xffffffff, length, 0/*srcLane*/);
    dbstrdst = __shfl_sync(0xffffffff, dbstrdst, 0/*srcLane*/);

    //read and write position-specific fields:
    if(pos < length) {
        FPTYPE coord0 = GetDbStrField<FPTYPE,pmv2DCoords+0>(dbstrdst + pos);
        FPTYPE coord1 = GetDbStrField<FPTYPE,pmv2DCoords+1>(dbstrdst + pos);
        FPTYPE coord2 = GetDbStrField<FPTYPE,pmv2DCoords+2>(dbstrdst + pos);
        LNTYPE icho = GetDbStrField<LNTYPE,pmv2D_Ins_Ch_Ord>(dbstrdst + pos);
        INTYPE rnum = GetDbStrField<INTYPE,pmv2DResNumber>(dbstrdst + pos);
        CHTYPE rsd = GetDbStrField<CHTYPE,pmv2Drsd>(dbstrdst + pos);
        CHTYPE ssa = GetDbStrField<CHTYPE,pmv2Dss>(dbstrdst + pos);
        ((FPTYPE*)(tmpdpdiagbuffers + (pmv2DCoords+0) * maxndbCposs))[newaddr + pos] = coord0;
        ((FPTYPE*)(tmpdpdiagbuffers + (pmv2DCoords+1) * maxndbCposs))[newaddr + pos] = coord1;
        ((FPTYPE*)(tmpdpdiagbuffers + (pmv2DCoords+2) * maxndbCposs))[newaddr + pos] = coord2;
        ((LNTYPE*)(tmpdpdiagbuffers + pmv2D_Ins_Ch_Ord * maxndbCposs))[newaddr + pos] = icho;
        ((INTYPE*)(tmpdpdiagbuffers + pmv2DResNumber * maxndbCposs))[newaddr + pos] = rnum;
        ((CHTYPE*)(tmpdpdiagbuffers + pmv2Drsd * maxndbCposs))[newaddr + pos] = rsd;
        ((CHTYPE*)(tmpdpdiagbuffers + pmv2Dss * maxndbCposs))[newaddr + pos] = ssa;
    }

    //read and write position-specific fields of 3D index:
    if(pos < length) {
        uint offset = pmv2DTotFlds * maxndbCposs;
        FPTYPE ndxcrd0 = GetIndxdDbStrField<FPTYPE,pmv2DNdxCoords+0>(dbstrdst + pos);
        FPTYPE ndxcrd1 = GetIndxdDbStrField<FPTYPE,pmv2DNdxCoords+1>(dbstrdst + pos);
        FPTYPE ndxcrd2 = GetIndxdDbStrField<FPTYPE,pmv2DNdxCoords+2>(dbstrdst + pos);
        INTYPE left = GetIndxdDbStrField<INTYPE,pmv2DNdxLeft>(dbstrdst + pos);
        INTYPE right = GetIndxdDbStrField<INTYPE,pmv2DNdxRight>(dbstrdst + pos);
        INTYPE ndxorg = GetIndxdDbStrField<INTYPE,pmv2DNdxOrgndx>(dbstrdst + pos);
        ((FPTYPE*)(tmpdpdiagbuffers + offset + (pmv2DNdxCoords+0) * maxndbCposs))[newaddr + pos] = ndxcrd0;
        ((FPTYPE*)(tmpdpdiagbuffers + offset + (pmv2DNdxCoords+1) * maxndbCposs))[newaddr + pos] = ndxcrd1;
        ((FPTYPE*)(tmpdpdiagbuffers + offset + (pmv2DNdxCoords+2) * maxndbCposs))[newaddr + pos] = ndxcrd2;
        ((INTYPE*)(tmpdpdiagbuffers + offset + pmv2DNdxLeft * maxndbCposs))[newaddr + pos] = left;
        ((INTYPE*)(tmpdpdiagbuffers + offset + pmv2DNdxRight * maxndbCposs))[newaddr + pos] = right;
        ((INTYPE*)(tmpdpdiagbuffers + offset + pmv2DNdxOrgndx * maxndbCposs))[newaddr + pos] = ndxorg;
    }

    //read and write query-reference statistics obtained so far:
    for(uint qryndx = blockIdx.x; qryndx < nqystrs; qryndx++)
    {
        const uint offset = (pmv2DTotFlds + pmv2DTotIndexFlds + qryndx) * maxndbCposs;
        const uint f = threadIdx.x;
        if(f < nTAuxWorkingMemoryVars) {
            //NOTE: uncoalesced read, coalesced write:
            uint mloc = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + f) * ndbCstrs;
            //NOTE: nTAuxWorkingMemoryVars * max(ndbCstrs) < max(ndbCposs) by definition (memory layout)
            uint tloc = offset + nTAuxWorkingMemoryVars * newndx;
            tmpdpdiagbuffers[tloc + f] = wrkmemaux[mloc + dbstrndx];
        }
        //only the last block loops until all query sections are processed:
        if(blockIdx.x + 1 < gridDim.x) break;
    }

    //read and write query-reference transformation matrices obtained so far:
    for(uint qryndx = blockIdx.x; qryndx < nqystrs; qryndx++)
    {
        const uint offset = (pmv2DTotFlds + pmv2DTotIndexFlds + nqystrs + qryndx) * maxndbCposs;
        const uint f = threadIdx.x;
        if(f < nTTranformMatrix) {
            //NOTE: coalesced read, coalesced write:
            uint mloc = (qryndx * ndbCstrs + dbstrndx) * nTTranformMatrix;
            //NOTE: nTTranformMatrixv * max(ndbCstrs) < max(ndbCposs) by definition (memory layout)
            uint tloc = offset + nTTranformMatrix * newndx;
            tmpdpdiagbuffers[tloc + f] = tfmmem[mloc + f];
        }
        //only the last block loops until all query sections are processed:
        if(blockIdx.x + 1 < gridDim.x) break;
    }
}



// -------------------------------------------------------------------------
// ReformatStructureDataPartLoad: reformat a reference database chunk to
// include candidates proceeding to stages of more detailed superposition
// search and refinement; this part corresponds to data load from secondary
// (temporary) location;
// nqystrs, total number of queries in the chunk;
// maxndbCposs, max number of db structure positions the chunk can accommodate;
// maxnsteps, max number of steps allocated for each reference structure;
// ndbCstrs2, total number of selected reference structures;
// NOTE: memory pointers should be aligned!
// tmpdpdiagbuffers, temporary memory containing all data to be copied back to
// structure data, wrkmemaux, and tfmmem;
// wrkmemaux, auxiliary working memory;
// tfmmem, memory of transformation matrices;
// NOTE: thread block is 1D and processes a fragment of each reference structure;
// NOTE: thread block's x-dimension assumed to be 32!
// 
__global__ void ReformatStructureDataPartLoad(
    const uint nqystrs,
    const uint maxndbCposs,
    const uint maxnsteps,
    const uint ndbCstrs2,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tfmmem)
{
    enum {lXDIM = CUFL_LOADCANDIDATEDATA_XDIM};
    // blockIdx.x is the block index of positions for one reference;
    // blockIdx.y is the reference serial number;
    // const uint dbstrfrg = blockIdx.x;
    const uint dbstrndx = blockIdx.y;//corresponds to newndx of part store
    //relative position index:
    const uint pos = blockIdx.x * blockDim.x + threadIdx.x;
    LNTYPE dbstrdst;//address; corresponds to newaddr of part store
    INTYPE length;

    if(threadIdx.x == 0) {
        //read structure-specific fields from a stored location:
        length = ((INTYPE*)(tmpdpdiagbuffers + pps2DLen * maxndbCposs))[dbstrndx];
        dbstrdst = ((LNTYPE*)(tmpdpdiagbuffers + pps2DDist * maxndbCposs))[dbstrndx];
        if(blockIdx.x == 0) {
            //only the first block writes structure-specific fields:
            INTYPE type = ((INTYPE*)(tmpdpdiagbuffers + pps2DType * maxndbCposs))[dbstrndx];
            SetDbStrField<INTYPE,pps2DLen>(dbstrndx, length);
            SetDbStrField<LNTYPE,pps2DDist>(dbstrndx, dbstrdst);
            SetDbStrField<INTYPE,pps2DType>(dbstrndx, type);
        }
    }

    length = __shfl_sync(0xffffffff, length, 0/*srcLane*/);
    dbstrdst = __shfl_sync(0xffffffff, dbstrdst, 0/*srcLane*/);

    //read and write position-specific fields:
    if(pos < length) {
        FPTYPE coord0 = ((FPTYPE*)(tmpdpdiagbuffers + (pmv2DCoords+0) * maxndbCposs))[dbstrdst + pos];
        FPTYPE coord1 = ((FPTYPE*)(tmpdpdiagbuffers + (pmv2DCoords+1) * maxndbCposs))[dbstrdst + pos];
        FPTYPE coord2 = ((FPTYPE*)(tmpdpdiagbuffers + (pmv2DCoords+2) * maxndbCposs))[dbstrdst + pos];
        LNTYPE icho = ((LNTYPE*)(tmpdpdiagbuffers + pmv2D_Ins_Ch_Ord * maxndbCposs))[dbstrdst + pos];
        INTYPE rnum = ((INTYPE*)(tmpdpdiagbuffers + pmv2DResNumber * maxndbCposs))[dbstrdst + pos];
        CHTYPE rsd = ((CHTYPE*)(tmpdpdiagbuffers + pmv2Drsd * maxndbCposs))[dbstrdst + pos];
        CHTYPE ssa = ((CHTYPE*)(tmpdpdiagbuffers + pmv2Dss * maxndbCposs))[dbstrdst + pos];
        SetDbStrField<FPTYPE,pmv2DCoords+0>(dbstrdst + pos, coord0);
        SetDbStrField<FPTYPE,pmv2DCoords+1>(dbstrdst + pos, coord1);
        SetDbStrField<FPTYPE,pmv2DCoords+2>(dbstrdst + pos, coord2);
        SetDbStrField<LNTYPE,pmv2D_Ins_Ch_Ord>(dbstrdst + pos, icho);
        SetDbStrField<INTYPE,pmv2DResNumber>(dbstrdst + pos, rnum);
        SetDbStrField<CHTYPE,pmv2Drsd>(dbstrdst + pos, rsd);
        SetDbStrField<CHTYPE,pmv2Dss>(dbstrdst + pos, ssa);
    }

    //read and write position-specific fields of 3D index:
    if(pos < length) {
        uint offset = pmv2DTotFlds * maxndbCposs;
        FPTYPE ndxcrd0 = ((FPTYPE*)(tmpdpdiagbuffers + offset + (pmv2DNdxCoords+0) * maxndbCposs))[dbstrdst + pos];
        FPTYPE ndxcrd1 = ((FPTYPE*)(tmpdpdiagbuffers + offset + (pmv2DNdxCoords+1) * maxndbCposs))[dbstrdst + pos];
        FPTYPE ndxcrd2 = ((FPTYPE*)(tmpdpdiagbuffers + offset + (pmv2DNdxCoords+2) * maxndbCposs))[dbstrdst + pos];
        INTYPE left = ((INTYPE*)(tmpdpdiagbuffers + offset + pmv2DNdxLeft * maxndbCposs))[dbstrdst + pos];
        INTYPE right = ((INTYPE*)(tmpdpdiagbuffers + offset + pmv2DNdxRight * maxndbCposs))[dbstrdst + pos];
        INTYPE ndxorg = ((INTYPE*)(tmpdpdiagbuffers + offset + pmv2DNdxOrgndx * maxndbCposs))[dbstrdst + pos];
        SetIndxdDbStrField<FPTYPE,pmv2DNdxCoords+0>(dbstrdst + pos, ndxcrd0);
        SetIndxdDbStrField<FPTYPE,pmv2DNdxCoords+1>(dbstrdst + pos, ndxcrd1);
        SetIndxdDbStrField<FPTYPE,pmv2DNdxCoords+2>(dbstrdst + pos, ndxcrd2);
        SetIndxdDbStrField<INTYPE,pmv2DNdxLeft>(dbstrdst + pos, left);
        SetIndxdDbStrField<INTYPE,pmv2DNdxRight>(dbstrdst + pos, right);
        SetIndxdDbStrField<INTYPE,pmv2DNdxOrgndx>(dbstrdst + pos, ndxorg);
    }

    //read and write query-reference statistics obtained so far:
    for(uint qryndx = blockIdx.x; qryndx < nqystrs; qryndx++)
    {
        const uint offset = (pmv2DTotFlds + pmv2DTotIndexFlds + qryndx) * maxndbCposs;
        const uint f = threadIdx.x;
        if(f < nTAuxWorkingMemoryVars) {
            //NOTE: coalesced read, uncoalesced write:
            uint mloc = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars + f) * ndbCstrs2;
            uint tloc = offset + nTAuxWorkingMemoryVars * dbstrndx;
            wrkmemaux[mloc + dbstrndx] = tmpdpdiagbuffers[tloc + f];
            //reset convergence flags:
            if(f == tawmvConverged) wrkmemaux[mloc + dbstrndx] = 0.0f;
        }
        //only the last block loops until all query sections are processed:
        if(blockIdx.x + 1 < gridDim.x) break;
    }

    //read and write query-reference transformation matrices obtained so far:
    for(uint qryndx = blockIdx.x; qryndx < nqystrs; qryndx++)
    {
        const uint offset = (pmv2DTotFlds + pmv2DTotIndexFlds + nqystrs + qryndx) * maxndbCposs;
        const uint f = threadIdx.x;
        if(f < nTTranformMatrix) {
            //NOTE: coalesced read, coalesced write:
            uint mloc = (qryndx * ndbCstrs2 + dbstrndx) * nTTranformMatrix;
            uint tloc = offset + nTTranformMatrix * dbstrndx;
            tfmmem[mloc + f] = tmpdpdiagbuffers[tloc + f];
        }
        //only the last block loops until all query sections are processed:
        if(blockIdx.x + 1 < gridDim.x) break;
    }
}
