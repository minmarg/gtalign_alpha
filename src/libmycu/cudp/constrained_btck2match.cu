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
#include "libmycu/custages/transform.cuh"
#include "libmycu/custages/covariance_refn.cuh"
#include "libmycu/custages/fields.cuh"
#include "libmycu/cudp/dpw_btck.cuh"
#include "libmycu/cudp/btck2match.cuh"
#include "constrained_btck2match.cuh"

// #define CUDP_CONSTRAINED_MTCH_TESTPRINT 0

// =========================================================================
// ConstrainedBtckToMatched32x: copy the coordinates of matched (aligned) 
// positions within a given distance threshold to destination location for 
// final refinement; use 32x unrolling;
// nqystrs, number of queries in a chunk;
// nqyposs, total number of query structure positions in a chunk;
// ndbCstrs, number of references in a chunk;
// ndbCposs, total number of db reference structure positions in a chunk;
// dbxpad, number of padded positions for memory alignment;
// maxnsteps, max number of steps performed for each reference structure 
// during alignment refinement;
// NOTE: memory pointers should be aligned!
// btckdata, backtracking information data;
// tfmmem, transformation matrix address space;
// wrkmemaux, auxiliary working memory;
// tmpdpalnpossbuffer, destination of the coordinates of matched positions;
// 
__global__
void ConstrainedBtckToMatched32x(
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const char* __restrict__ btckdata,
    const float* __restrict__ tfmmem,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpalnpossbuffer)
{
    // blockDim.x defines the number of matched positions read and written in 
    // parallel within a block;
    // blockDim.y (3 or 6) number of coordinates read and written in parallel 
    // within a block;
    const uint dbstrndx = blockIdx.x;//reference serial number
    const uint qryndx = blockIdx.y;//query serial number
    //cache of the query and reference match positions:
    enum {bmQRYNDX, bmRFNNDX, bmTotal};
    __shared__ int posCache[bmTotal][CUDP_CONST_MATCH_DIM_X+1];
    //cache of the query and reference coordinates:
    //position included (flag), position accumulated index, total:
    enum {ccPOSINC = nTDPAlignedPoss, ccPOSNDX, ccTotal};
    __shared__ float crdCache[ccTotal][CUDP_CONST_MATCH_DIM_X+1];
    __shared__ float tfmCache[nTTranformMatrix];//transformation matrix
    int qrylen, dbstrlen;//query and reference length
    //distances in positions to the beginnings of the query and reference structures:
    uint qrydst, dbstrdst;


    //check convergence
    if(threadIdx.x == 0 && threadIdx.y == 0) {
        //check for coonvergence at sfragfct==0:
        uint mloc = ((qryndx * maxnsteps + 0/*sfragfct*/) * nTAuxWorkingMemoryVars) * ndbCstrs;
        posCache[1][0] = wrkmemaux[mloc + tawmvConverged * ndbCstrs + dbstrndx];//float->int
    }

    __syncthreads();

    if(posCache[1][0] & (CONVERGED_LOWTMSC_bitval))
        //(the termination flag for this pair is set);
        //all threads in the block exit;
        return;


    //NOTE: pps2DLen and pps2DDist assumed to be adjacent: see PM2DVectorFields.h!
    //reuse cache
    if(threadIdx.x < 2 && threadIdx.y == 0) {
        GetDbStrLenDst(dbstrndx, &posCache[0][0]);
        GetQueryLenDst(qryndx, &posCache[0][2]);
    }

    __syncthreads();

    //NOTE: no bank conflict when two threads from the same warp access the same address;
    dbstrlen = posCache[0][0]; dbstrdst = posCache[0][1];
    qrylen = posCache[0][2]; qrydst = posCache[0][3];

    if(threadIdx.x < nTTranformMatrix && threadIdx.y == 0) {
        //globally best transformation matrix for a pair:
        uint mloc0 = (qryndx * ndbCstrs + dbstrndx) * nTTranformMatrix;
        tfmCache[threadIdx.x] = tfmmem[mloc0 + threadIdx.x];
    }

    __syncthreads();

    int x, y;

    GetTerminalCellXY<false/* ANCHORRGN */,false/* BANDED */>(
        x, y, qrylen, dbstrlen, 0/* qrypos */, 0/* rfnpos */, 0/* fraglen(all unused) */);


    int alnlen = 0;
    char btck = dpbtckDIAG;
    const int dblen = ndbCposs + dbxpad;
    //offset to the beginning of the data along the y axis wrt query qryndx: 
    const int yofff = (qryndx * maxnsteps + 0/*sfragfct*/) * dblen * nTDPAlignedPoss;
    //offset for writing distance and positional data for producing final alignments:
    const int yoff1 = (qryndx * maxnsteps + 1/*sfragfct*/) * dblen * nTDPAlignedPoss;
    const float d82 = GetD82(qrylen, dbstrlen);


#ifdef CUDP_CONSTRAINED_MTCH_TESTPRINT
    if(threadIdx.x == 0 && threadIdx.y == 0) {
        if((CUDP_CONSTRAINED_MTCH_TESTPRINT>=0)? dbstrndx==CUDP_CONSTRAINED_MTCH_TESTPRINT: 1)
            printf(" CNTMTCH: bid= %u: qlen= %d qaddr= %u len= %d addr= %u (y= %d x= %d) d82= %.2f\n",
                dbstrndx,qrylen,qrydst,dbstrlen,dbstrdst,y,x,d82
            );
    }
#endif


    //backtrace over the alignment
    while(btck != dpbtckSTOP)
    {
        int ndx = 0;
        if(threadIdx.x == 0 && threadIdx.y == 0) {
            //thread 0 records matched positions
            for(; ndx < CUDP_CONST_MATCH_DIM_X;) {
                if(x < 0 || y < 0) {
                    btck = dpbtckSTOP;
                    break;
                }
                int qpos = (qrydst + y) * dblen + dbstrdst + x;
                btck = btckdata[qpos];//READ
                if(btck == dpbtckSTOP)
                    break;
                if(btck == dpbtckUP) {
                    y--; 
                    continue; 
                }
                else if(btck == dpbtckLEFT) { 
                    x--; 
                    continue; 
                }
                //(btck == dpbtckDIAG)
                posCache[bmQRYNDX][ndx] = y;
                posCache[bmRFNNDX][ndx] = x;
                x--; y--; ndx++;
            }
            //save ndx and btck values in cache:
            posCache[bmQRYNDX][CUDP_CONST_MATCH_DIM_X] = ndx;
            posCache[bmRFNNDX][CUDP_CONST_MATCH_DIM_X] = btck;
        }

        __syncthreads();

        ndx = posCache[bmQRYNDX][CUDP_CONST_MATCH_DIM_X];
        btck = posCache[bmRFNNDX][CUDP_CONST_MATCH_DIM_X];

#ifdef CUDP_CONSTRAINED_MTCH_TESTPRINT
        if(threadIdx.x == 0 && threadIdx.y == 0) {
            if((CUDP_CONSTRAINED_MTCH_TESTPRINT>=0)? dbstrndx==CUDP_CONSTRAINED_MTCH_TESTPRINT: 1){
                for(int ii=0; ii<ndx; ii++) printf(" %5d", posCache[bmQRYNDX][ii]); printf("\n");
                for(int ii=0; ii<ndx; ii++) printf(" %5d", posCache[bmRFNNDX][ii]); printf("\n\n");
            }
        }
#endif

        //READ the coordinates of the matched positions
        if(threadIdx.x < ndx) {
            //READ coordinates
            int pos = qrydst + posCache[bmQRYNDX][threadIdx.x];
            int crd = threadIdx.y;
            //field section (query, reference):
            int fldsec = ndx_qrs_dc_pm2dvfields_;
#if (CUDP_CONST_MATCH_DIM_Y == 3)
            crdCache[threadIdx.y][threadIdx.x] = GetQueryCoord(crd, pos);
            pos = dbstrdst + posCache[bmRFNNDX][threadIdx.x];
            crdCache[threadIdx.y+pmv2DNoElems][threadIdx.x] = GetDbStrCoord(crd, pos);
#elif (CUDP_CONST_MATCH_DIM_Y == 6)
            if(pmv2DNoElems <= threadIdx.y) {
                pos = dbstrdst + posCache[bmRFNNDX][threadIdx.x];
                crd -= pmv2DNoElems;
                fldsec = ndx_dbs_dc_pm2dvfields_;
            }
            crdCache[threadIdx.y][threadIdx.x] = GetQryRfnCoord(fldsec, crd, pos);
#else
#error "INVALID EXECUTION CONFIGURATION: Assign CUDP_CONST_MATCH_DIM_Y to 3 or 6."
#endif
        }

        __syncthreads();

        //inclusionn flag, accumulated index, and distance2
        float posinc = 0.0f, posndx = 0.0f, dst2 = 999999.9f;

        //calculate distances
        if(threadIdx.x < ndx && threadIdx.y == 0) {
            dst2 = transform_and_distance2(tfmCache,
                crdCache[dpapsQRYx][threadIdx.x], crdCache[dpapsQRYy][threadIdx.x], crdCache[dpapsQRYz][threadIdx.x],
                crdCache[dpapsRFNx][threadIdx.x], crdCache[dpapsRFNy][threadIdx.x], crdCache[dpapsRFNz][threadIdx.x]);
            crdCache[ccPOSINC][threadIdx.x] = posinc = (dst2 <= d82);
        }

        if(threadIdx.y == 0) {
            //warp-sync to calculate the inclusive prefix sum of inclusion flags;
            //this also gives the accumulated indices of included aligned pairs
            posndx = mywarpincprefixsum(posinc);
            //posndx > 0 always where posinc == 1:
            crdCache[ccPOSNDX][threadIdx.x] = posndx - 1.0f;
        }

        __syncthreads();

#ifdef CUDP_CONSTRAINED_MTCH_TESTPRINT
        if(threadIdx.x == 0 && threadIdx.y == 0) {
            if((CUDP_CONSTRAINED_MTCH_TESTPRINT>=0)? dbstrndx==CUDP_CONSTRAINED_MTCH_TESTPRINT: 1){
                for(int ii=0; ii<ndx; ii++) printf(" %5.0f", crdCache[ccPOSINC][ii]); printf("*\n");
                for(int ii=0; ii<ndx; ii++) printf(" %5.0f", crdCache[ccPOSNDX][ii]); printf("*\n\n");
            }
        }
#endif

        if(threadIdx.x < ndx) {
            //WRITE coordinates in the order reverse to alignment itself;
            //take into account #positions written already (alnlen);
            //threadIdx.y * dblen represents a particular coordinates section in nTDPAlignedPoss:
            int pos = yofff + dbstrdst + alnlen  + threadIdx.y * dblen;
            int ps1 = yoff1 + dbstrdst + alnlen  + threadIdx.y * dblen;
            int tidndx = crdCache[ccPOSNDX][threadIdx.x];
            int tidinc = crdCache[ccPOSINC][threadIdx.x];
            //write only selected pairs
            if(tidinc)
                tmpdpalnpossbuffer[pos+tidndx] = crdCache[threadIdx.y][threadIdx.x];
#if (CUDP_CONST_MATCH_DIM_Y == 3)
            pos += pmv2DNoElems * dblen;
            tmpdpalnpossbuffer[pos+tidndx] = crdCache[threadIdx.y+pmv2DNoElems][threadIdx.x];
#endif
            //write distances (calculated by tid.y==0) and positions of selected pairs
            if(tidinc && threadIdx.y == 0)
                tmpdpalnpossbuffer[ps1+tidndx] = dst2;
            if(tidinc && 0 < threadIdx.y && threadIdx.y <= bmTotal)
                tmpdpalnpossbuffer[ps1+tidndx] = posCache[threadIdx.y-1][threadIdx.x];
        }

        //no need for sync as long as crdCache is not modified until the next sync
        //__syncthreads();

#ifdef CUDP_CONSTRAINED_MTCH_TESTPRINT
        __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0) {
            int ps1 = yoff1 + dbstrdst + alnlen  + threadIdx.y * dblen;
            if((CUDP_CONSTRAINED_MTCH_TESTPRINT>=0)? dbstrndx==CUDP_CONSTRAINED_MTCH_TESTPRINT: 1){
                for(int ii=0; ii<ndx; ii++) printf(" %5.0f", crdCache[ccPOSINC][ii]); printf("+\n");
                for(int ii=0; ii<ndx; ii++) printf(" %5.2f", tmpdpalnpossbuffer[ps1+(int)crdCache[ccPOSNDX][ii]]); printf("+\n");
                for(int ii=0; ii<ndx; ii++) printf(" %5.0f", tmpdpalnpossbuffer[ps1+dblen+(int)crdCache[ccPOSNDX][ii]]); printf("+\n");
                for(int ii=0; ii<ndx; ii++) printf(" %5.0f", tmpdpalnpossbuffer[ps1+2*dblen+(int)crdCache[ccPOSNDX][ii]]); printf("+\n\n\n\n");
            }
        }
#endif

        //#selected aligned positions is at (ndx-1) as a result of the prefix sum
        if(0 < ndx) alnlen += (crdCache[ccPOSNDX][ndx-1] + 1.0f);
    }//while(btck)


    //WRITE #selected matched (aligned) positions 
    if(threadIdx.x == 0 && threadIdx.y == 0) {
        uint mloc0 = ((qryndx * maxnsteps + 0/*sfragfct*/) * nTAuxWorkingMemoryVars) * ndbCstrs;
        wrkmemaux[mloc0 + tawmvNAlnPoss * ndbCstrs + dbstrndx] = alnlen;
    }


#ifdef CUDP_CONSTRAINED_MTCH_TESTPRINT
    if(threadIdx.x == 0 && threadIdx.y == 0) {
        if((CUDP_CONSTRAINED_MTCH_TESTPRINT>=0)? dbstrndx==CUDP_CONSTRAINED_MTCH_TESTPRINT: 1)
            printf(" CNTMTCH (pronr=%u): y= %d x= %d alnlen= %d qrylen= %d dbstrlen= %d\n\n",
                dbstrndx,y,x,alnlen,qrylen,dbstrlen
            );
    }
#endif
}

// =========================================================================
// -------------------------------------------------------------------------
