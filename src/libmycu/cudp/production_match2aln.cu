/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gproc/btckcoords.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"
#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/custages/fields.cuh"
#include "production_match2aln.cuh"

// #define CUDP_PRODUCTION_ALN_TESTPRINT 0

// =========================================================================
// ProductionMatchToAlignment32x: produce the final alignment with 
// accompanying information from the given match (aligned) positions; 
// use 32x unrolling;
// nodeletions, flag of not including deletion positions in alignments;
// d2equiv, squared distance threshold for structural equivalence; 
// nqystrs, number of queries in a chunk;
// nqyposs, total number of query structure positions in a chunk;
// ndbCstrs, number of references in a chunk;
// ndbCposs, total number of db reference structure positions in a chunk;
// dbxpad, number of padded positions for memory alignment;
// maxnsteps, max number of steps performed for each reference structure 
// during alignment refinement;
// NOTE: memory pointers should be aligned!
// tmpdpalnpossbuffer, coordinates of matched positions;
// wrkmemaux, auxiliary working memory;
// alndatamem, memory for full alignment information;
// alnsmem, memory for output full alignments;
// 
__global__
void ProductionMatchToAlignment32x(
    const bool nodeletions,
    const float d2equiv,
    // const uint nqystrs,
    // const uint nqyposs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const float* __restrict__ tmpdpalnpossbuffer,
    const float* __restrict__ wrkmemaux,
    float* __restrict__ alndatamem,
    char* __restrict__ alnsmem)
{
    // blockDim.x defines the number of positions read and written in 
    // parallel within a block;
    // blockDim.y, #informationn sections read and written in parallel 
    // within a block;
    const uint dbstrndx = blockIdx.x;//reference serial number
    const uint qryndx = blockIdx.y;//query serial number
    enum {
        blockDim_x = CUDP_PRODUCTION_ALN_DIM_X,//block's x-dimension
        blockDim_x1 = CUDP_PRODUCTION_ALN_DIM_X+1
    };
    enum {bmQRDST2, bmQRYNDX, bmRFNNDX, bmTotal};
    __shared__ float SMEM[nTDP2OutputAlnDataPart1End + (bmTotal + nTDP2OutputAlignmentSSS) * blockDim_x1];
    //cache for alignment statistics [nTDP2OutputAlnDataPart1End]:
    float* outStsCache = SMEM;
    //cache of distances and query and reference match positions [bmTotal][CUDP_PRODUCTION_ALN_DIM_X+1]:
    float* posCache = outStsCache + nTDP2OutputAlnDataPart1End;
    //cache for alignment [nTDP2OutputAlignmentSSS][CUDP_PRODUCTION_ALN_DIM_X+1]:
    int* outAlnCache = (int*)(posCache + bmTotal * blockDim_x1);
    int qrylen/* , dbstrlen */;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;
    int mtchlen = 0;


    //check convergence
    if(threadIdx.x == 0 && threadIdx.y == 0) {
        //check for coonvergence at sfragfct==0:
        uint mloc = ((qryndx * maxnsteps + 0/*sfragfct*/) * nTAuxWorkingMemoryVars) * ndbCstrs;
        outAlnCache[31] = wrkmemaux[mloc + tawmvConverged * ndbCstrs + dbstrndx];
    }

    __syncthreads();

    if(outAlnCache[31] & (CONVERGED_LOWTMSC_bitval))
        //(the termination flag for this pair is set);
        //all threads in the block exit;
        return;


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse cache
    if(threadIdx.x < 2 && threadIdx.y == 0) {
        //GetDbStrLenDst(dbstrndx, &outAlnCache[0]);
        if(threadIdx.x ==  0) outAlnCache[1] = GetDbStrDst(dbstrndx);
        GetQueryLenDst(qryndx, &outAlnCache[2]);
    }

    if(threadIdx.x == tawmvNAlnPoss && threadIdx.y == 1) {
        //NOTE: reuse cache to read #matched positions;
        //NOTE: tawmvNAlnPoss written at sfragfct==0:
        uint mloc0 = ((qryndx * maxnsteps + 0) * nTAuxWorkingMemoryVars) * ndbCstrs;
        outAlnCache[tawmvNAlnPoss] = wrkmemaux[mloc0 + tawmvNAlnPoss * ndbCstrs + dbstrndx];
    }

    __syncthreads();

    //NOTE: no bank conflict when two threads from the same warp access the same address;
    /* dbstrlen = outAlnCache[0]; */ dbstrdst = outAlnCache[1];
    qrylen = outAlnCache[2]; qrydst = outAlnCache[3];
    mtchlen = outAlnCache[tawmvNAlnPoss];

    __syncthreads();

    int falnlen = 0, ppaln = 0;
    //NOTE: assign big numbers to qp1 and rp1 (>possible lengths) to
    //correctly process the first aligned position:
    int qp0, rp0, qp1 = 999999, rp1 = 999999;

    //#identities, #structurally equivalent positions, #gaps:
    int idts = 0, psts = 0, gaps = 0;
    uint bcrds = 0, ecrds = 0;//beginning and end positions
    const int dblen = ndbCposs + dbxpad;
    //offset to the beginning of the data along the y axis wrt query qryndx: 
    const int yoff1 = (qryndx * maxnsteps + 1/*sfragfct*/) * dblen * nTDPAlignedPoss;
    //offset for alignments
    const int alnofff = (qryndx * ndbCposs + ndbCstrs * (qrydst + qryndx)) * nTDP2OutputAlignmentSSS;
    //aln length for query qryndx across all references:
    const int dbalnlen = ndbCposs + ndbCstrs * (qrylen + 1);
    //end alignment position for query qryndx, processed by threaIdx.x:
    //const int dbalnend = dbstrdst + dbstrndx * (qrylen + 1) + qrylen + dbstrlen - (int)threadIdx.x;
    //start alignment position for query qryndx, processed by threaIdx.x:
    const int dbalnbeg = dbstrdst + dbstrndx * (qrylen + 1) + threadIdx.x;


    //manually unroll along data blocks:
    for(int relpos = threadIdx.x; relpos < mtchlen + (int)threadIdx.x; relpos += blockDim_x)
    {
        //starting position in tmpdpalnpossbuffer for a pair:
        //NOTE: alignment written in reverse order:
        int pos = yoff1 + dbstrdst + mtchlen-1 - (relpos);

        //READ distances and matched positions
        if(relpos < mtchlen)
            posCache[threadIdx.y * blockDim_x1 + threadIdx.x] = tmpdpalnpossbuffer[pos + threadIdx.y * dblen];
        __syncthreads();

#ifdef CUDP_PRODUCTION_ALN_TESTPRINT
        if(threadIdx.x == 0 && threadIdx.y == 0) {
            if((CUDP_PRODUCTION_ALN_TESTPRINT>=0)? dbstrndx==CUDP_PRODUCTION_ALN_TESTPRINT: 1){
                for(int ii=0; ii<blockDim_x && relpos+ii<mtchlen; ii++) printf(" %5.1f", posCache[bmQRDST2*blockDim_x1+ii]); printf("\n");
                for(int ii=0; ii<blockDim_x && relpos+ii<mtchlen; ii++) printf(" %5.0f", posCache[bmQRYNDX*blockDim_x1+ii]); printf("\n");
                for(int ii=0; ii<blockDim_x && relpos+ii<mtchlen; ii++) printf(" %5.0f", posCache[bmRFNNDX*blockDim_x1+ii]); printf("\n\n");
            }
        }
#endif
        //produce alignment for the fragment of matched positions read 
        for(int p = relpos-1, pp = -1; p < mtchlen + (int)threadIdx.x && pp < blockDim_x; p++, pp++)
        {
            if(threadIdx.x == 0 && threadIdx.y == 0) {
                qp0 = qp1; rp0 = rp1;
                qp1 = qp0; rp1 = rp0;
                if(p+1 < mtchlen + threadIdx.x && pp+1 < blockDim_x) {
                    qp1 = posCache[bmQRYNDX * blockDim_x1 + pp+1];//query position
                    rp1 = posCache[bmRFNNDX * blockDim_x1 + pp+1];//reference position
                }
                if(0 <= pp) {
                    //update alignment statistics
                    float dst2 = posCache[bmQRDST2 * blockDim_x1 + pp];//distance2
                    ecrds = CombineCoords(rp0+1,qp0+1);
                    if(p == 0) bcrds = ecrds;
                    if(dst2 < d2equiv) psts++;
                    //minus to indicate reading
                    outAlnCache[dp2oaQuery * blockDim_x1 + ppaln] =
                    outAlnCache[dp2oaQuerySSS * blockDim_x1 + ppaln] = -qp0;
                    outAlnCache[dp2oaTarget * blockDim_x1 + ppaln] =
                    outAlnCache[dp2oaTargetSSS * blockDim_x1 + ppaln] = -rp0;
                    outAlnCache[dp2oaMiddle * blockDim_x1 + ppaln] = (dst2 < d2equiv)?'+':' ';
                    ppaln++;
                }
                qp0++; rp0++;
            }
            //TODO: make clear why this sync is required here!
            __syncthreads();
            while(1) {
                while(threadIdx.x == 0 && threadIdx.y == 0) {
                    for(; qp0 < qp1 && ppaln < blockDim_x; qp0++, ppaln++) {
                        outAlnCache[dp2oaQuery * blockDim_x1 + ppaln] =
                        outAlnCache[dp2oaQuerySSS * blockDim_x1 + ppaln] = -qp0;
                        outAlnCache[dp2oaTarget * blockDim_x1 + ppaln] = '-';
                        outAlnCache[dp2oaTargetSSS * blockDim_x1 + ppaln] = ' ';
                        outAlnCache[dp2oaMiddle * blockDim_x1 + ppaln] = ' ';
                        gaps++;
                    }
                    outAlnCache[0 * blockDim_x1 + blockDim_x] = ppaln;
                    if(blockDim_x <= ppaln) break;
                    if(nodeletions) break;//NOTE: command-line option
                    for(; rp0 < rp1 && ppaln < blockDim_x; rp0++, ppaln++) {
                        outAlnCache[dp2oaTarget * blockDim_x1 + ppaln] =
                        outAlnCache[dp2oaTargetSSS * blockDim_x1 + ppaln] = -rp0;
                        outAlnCache[dp2oaQuery * blockDim_x1 + ppaln] = '-';
                        outAlnCache[dp2oaQuerySSS * blockDim_x1 + ppaln] = ' ';
                        outAlnCache[dp2oaMiddle * blockDim_x1 + ppaln] = ' ';
                        gaps++;
                    }
                    outAlnCache[0 * blockDim_x1 + blockDim_x] = ppaln;
                    break;
                }

                __syncthreads();
                ppaln = outAlnCache[0 * blockDim_x1 + blockDim_x];
                if(ppaln < blockDim_x) break;

                //the buffer has filled; write the alignment fragment to gmem
                idts +=
                WriteAlignmentFragment<blockDim_x1>(qrydst, dbstrdst,
                    alnofff, dbalnlen, dbalnbeg, falnlen/* written */,
                    ppaln/* towrite */, ppaln/* tocheck */,
                    outAlnCache, alnsmem);

                //for outAlnCache is not overwritten before write to gmem is complete:
                __syncthreads();

#ifdef CUDP_PRODUCTION_ALN_TESTPRINT
                if(threadIdx.x == 0 && threadIdx.y == 0) {
                    if((CUDP_PRODUCTION_ALN_TESTPRINT>=0)? dbstrndx==CUDP_PRODUCTION_ALN_TESTPRINT: 1){
                        for(int ii=0; ii<ppaln; ii++) printf("%c", outAlnCache[dp2oaQuerySSS*blockDim_x1+ii]); printf("\n");
                        for(int ii=0; ii<ppaln; ii++) printf("%c", outAlnCache[dp2oaQuery*blockDim_x1+ii]); printf("\n");
                        for(int ii=0; ii<ppaln; ii++) printf("%c", outAlnCache[dp2oaMiddle*blockDim_x1+ii]); printf("\n");
                        for(int ii=0; ii<ppaln; ii++) printf("%c", outAlnCache[dp2oaTarget*blockDim_x1+ii]); printf("\n");
                        for(int ii=0; ii<ppaln; ii++) printf("%c", outAlnCache[dp2oaTargetSSS*blockDim_x1+ii]); printf("\n\n");
                    }
                }
#endif
                falnlen += ppaln;
                ppaln = 0;
            }//while(1)
        }//for(p = relpos...)
    }//for(relpos = threadIdx.x...)

    //write terminator 0
    if(threadIdx.x == 0 && ppaln < blockDim_x) {
        //the condition ppaln < blockDim_x ensured by the above inner loop
        for(int i = threadIdx.y; i < nTDP2OutputAlignmentSSS; i += CUDP_PRODUCTION_ALN_DIM_Y)
            outAlnCache[i * blockDim_x1 + ppaln] = 0;
    }

    __syncthreads();

    //write the last alignment fragment to gmem
    idts +=
    WriteAlignmentFragment<blockDim_x1>(qrydst, dbstrdst,
        alnofff, dbalnlen, dbalnbeg, falnlen/* written */,
        ppaln + (ppaln < blockDim_x)/* towrite */, ppaln/* tocheck */,
        outAlnCache, alnsmem);

    falnlen += ppaln;

    //write allignment statistics
    if(threadIdx.y == 0) {
        if(threadIdx.x == 0) {
            *(uint*)(outStsCache+dp2oadBegCoords) = bcrds;
            *(uint*)(outStsCache+dp2oadEndCoords) = ecrds;
            outStsCache[dp2oadAlnLength] = falnlen;
            outStsCache[dp2oadPstvs] = psts;
            outStsCache[dp2oadIdnts] = idts;
            outStsCache[dp2oadNGaps] = gaps;
        }

        __syncwarp();

        if(nTDP2OutputAlnDataPart1Beg <= threadIdx.x && threadIdx.x < nTDP2OutputAlnDataPart1End) {
            uint mloc = (qryndx * ndbCstrs + dbstrndx) * nTDP2OutputAlnData;
            alndatamem[mloc + threadIdx.x] = outStsCache[threadIdx.x];
        }
    }

#ifdef CUDP_PRODUCTION_ALN_TESTPRINT
    if(threadIdx.x == 0 && threadIdx.y == 0) {
        if((CUDP_PRODUCTION_ALN_TESTPRINT>=0)? dbstrndx==CUDP_PRODUCTION_ALN_TESTPRINT: 1){
            for(int ii=0; ii<ppaln+1; ii++) printf("%c", outAlnCache[dp2oaQuerySSS*blockDim_x1+ii]); printf("\n");
            for(int ii=0; ii<ppaln+1; ii++) printf("%c", outAlnCache[dp2oaQuery*blockDim_x1+ii]); printf("\n");
            for(int ii=0; ii<ppaln+1; ii++) printf("%c", outAlnCache[dp2oaMiddle*blockDim_x1+ii]); printf("\n");
            for(int ii=0; ii<ppaln+1; ii++) printf("%c", outAlnCache[dp2oaTarget*blockDim_x1+ii]); printf("\n");
            for(int ii=0; ii<ppaln+1; ii++) printf("%c", outAlnCache[dp2oaTargetSSS*blockDim_x1+ii]); printf("\n\n");
        }
    }
#endif

#ifdef CUDP_PRODUCTION_ALN_TESTPRINT
    if(threadIdx.x == 0 && threadIdx.y == 0) {
        if((CUDP_PRODUCTION_ALN_TESTPRINT>=0)? dbstrndx==CUDP_PRODUCTION_ALN_TESTPRINT: 1)
            printf(" PRODALN (dbstr= %u): qrylen= %d dbstrlen= --  y0= %d x0= %d ye= %d xe= %d "
                "falnlen= %d  psts= %d idts= %d gaps= %d\n\n", dbstrndx,qrylen,/* dbstrlen, */
                GetCoordY(bcrds),GetCoordX(bcrds),GetCoordY(ecrds),GetCoordX(ecrds),
                falnlen,psts,idts,gaps
            );
    }
#endif
}

// =========================================================================
// -------------------------------------------------------------------------
