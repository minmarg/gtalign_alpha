/***************************************************************************
 *   Copyright (C) 2021-2026 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include <omp.h>

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libutil/templates.h"
#include "libutil/CLOptions.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libgenp/gdats/PMBatchStrData.h"
#include "libmymp/mplayout/MpGlobalMemory.h"
#include "libmymp/mpproc/mpprocconf.h"
#include "MpSecStr.h"

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// RunHelper: calculate secondary structure for a batch of 3D structures;
// pmbeg_, pmend_, structure data beginning and end addresses;
// nstrs, total number of structures in the chunk;
// str1len, length of the largest structure;
//
void MpSecStr::ssk_kernel_helper(
    char* const * const pmbeg, char* const * const /*pmend*/,
    int nstrs, int str1len)
{
    enum{   nMRG0 = 2,//margin from one end
            nMRG2 = 4,//total margin: two from both ends
            XDIM = MPSS_CALCSTR_XDIM,
            nMAXPOS = XDIM + nMRG2//#max positions in the cache
    };

    MYMSG("MpSecStr::ssk_kernel_helper", 4);
    // static const std::string preamb = "MpSecStr::ssk_kernel_helper: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    //execution configurations for secondary structure calculation:
    //process MPSS_CALCSTR_XDIM positions along the structures in parallel:
    const int nblocks_x = (str1len + XDIM - 1) / XDIM;
    const int nblocks_y = nstrs;

    //cache for coordinates: 
    float crds[nMAXPOS][pmv2DNoElems];
    //cache for distances along str. positions: two, three, four residues apart
    float dsts[nMAXPOS][cssTotal];

    // #pragma omp threadprivate(crds, dsts)

    //NOTE: pmbeg, pmend are shared by definition: constant addresses
    #pragma omp parallel for num_threads(nthreads) default(shared) \
        private(crds, dsts) \
        shared(/*pmbeg, pmend,*/ nstrs) \
        collapse(2)
    for(int si = 0; si < nblocks_y; si++)
        for(int bi = 0; bi < nblocks_x; bi++)
        {//threads process blocks
            const int strlen = PMBatchStrData::GetLengthAt(pmbeg, si);
            const int strdst = PMBatchStrData::GetAddressAt(pmbeg, si);
            const int typex = PMBatchStrData::GetFieldAt<INTYPE,pmv2D_Ins_Ch_Ord>(pmbeg, strdst);
            const int type = (GetMoleculeType(typex) == gtmtProtein);
            const int ndx0 = bi * XDIM;
            const int strdstndx0 = strdst + ndx0;
            //if thread is not out of bounds
            if(type && ndx0 < strlen) {
                // const int blkbeg = bi? mymax(0, nMRG0 - ndx0) - nMRG0: 0;
                int blkbeg = bi? -nMRG0: 0;
                int blkend = (ndx0 + XDIM + nMRG0 < strlen)? (XDIM + nMRG0): (strlen - ndx0);

                #pragma omp simd aligned(pmbeg:PMBSdatalignment)
                for(int pi = blkbeg; pi < blkend; pi++)
                {//simd for positions
                    crds[pi + nMRG0][pmv2DX] = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DX>(pmbeg, strdstndx0 + pi);
                    crds[pi + nMRG0][pmv2DY] = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DY>(pmbeg, strdstndx0 + pi);
                    crds[pi + nMRG0][pmv2DZ] = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DZ>(pmbeg, strdstndx0 + pi);
                }

                //cache and calculate linear distances
                blkbeg = bi? 0: nMRG0;
                blkend = (ndx0 + XDIM + 2 < strlen)? (XDIM + 2): (strlen - ndx0);
                #pragma omp simd
                for(int pi = blkbeg; pi < blkend; pi++)
                    dsts[pi][css2RESdst] = GetDistance<2>(crds, pi);

                blkend = (ndx0 + XDIM + 3 < strlen)? (XDIM + 1): (strlen - ndx0);
                #pragma omp simd
                for(int pi = blkbeg; pi < blkend; pi++)
                    dsts[pi][css3RESdst] = GetDistance<3>(crds, pi);

                blkend = (ndx0 + XDIM + 4 < strlen)? (XDIM): (strlen - ndx0);
                #pragma omp simd
                for(int pi = blkbeg; pi < blkend; pi++)
                    dsts[pi][css4RESdst] = GetDistance<4>(crds, pi);

                //assign secondary structure:
                blkbeg = bi? 0: nMRG0;
                blkend = (ndx0 + XDIM + nMRG0 < strlen)? (XDIM): mymax(0, strlen - ndx0 - nMRG0);
                int lseend = (ndx0 + XDIM + nMRG0 < strlen)? (XDIM): mymax(0, strlen - ndx0);

                #pragma omp simd aligned(pmbeg:PMBSdatalignment)
                for(int pi = 0; pi < nMRG0; pi++)
                    PMBatchStrData::SetFieldAt<CHTYPE,pmv2Dss>(pmbeg, strdstndx0 + pi, pmvLOOP);

                #pragma omp simd aligned(pmbeg:PMBSdatalignment)
                for(int pi = blkend; pi < lseend; pi++)
                    PMBatchStrData::SetFieldAt<CHTYPE,pmv2Dss>(pmbeg, strdstndx0 + pi, pmvLOOP);

                #pragma omp simd aligned(pmbeg:PMBSdatalignment)
                for(int pi = blkbeg; pi < blkend; pi++)
                    PMBatchStrData::SetFieldAt<CHTYPE,pmv2Dss>(pmbeg, strdstndx0 + pi, AassignSecStr(dsts, pi));
            }
        }
}



// -------------------------------------------------------------------------
// nass_calcsecstrs_kernel_helper: calculate secondary structures for all 
// query OR reference nucleic acid structures in the chunk;
// tmpdpdiagbuffers, temporary buffer for distances and positions;
// nstrs, total number of structures in the chunk;
// nposs, total number of (query or reference) positions in the chunk;
// 
void MpSecStr::nass_calcsecstrs_kernel_helper(
    const float *const __RESTRICT__ tmpdpdiagbuffers,
    char *const *const pmbeg, char *const *const /*pmend*/,
    const int nstrs, const int nposs)
{
    enum{
        lXDIM = MPSS_NACALCSTR_XDIM
    };

    MYMSG("MpSecStr::nass_calcsecstrs_kernel_helper", 4);
    // static const std::string preamb = "MpSecStr::nass_calcsecstrs_kernel_helper: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = CuMemoryBase::GetMinMemAlignment();
    //execution configuration for secondary structure assignment:
    const int nblocks_x = nstrs;

    int flgs[lXDIM];
    int poss[lXDIM];

    //NOTE: constant scalars and pointers, pmbeg, pmend, shared by definition:
    #pragma omp parallel for num_threads(nthreads) default(shared) \
        private(flgs,poss)
    for(int si = 0; si < nblocks_x; si++)
    {//threads process blocks
        const int strlen = PMBatchStrData::GetLengthAt(pmbeg, si);
        const int strdst = PMBatchStrData::GetAddressAt(pmbeg, si);
        const int typex = PMBatchStrData::GetFieldAt<INTYPE,pmv2D_Ins_Ch_Ord>(pmbeg, strdst);
        const int type = (GetMoleculeType(typex) == gtmtNA);

        if(type) {
            #pragma omp simd aligned(pmbeg:PMBSdatalignment)
            for(int xi = 0; xi < strlen; xi++)
                PMBatchStrData::SetFieldAt<CHTYPE,pmv2Dss>(pmbeg, strdst + xi, pmnasUNPAIRED);

            for(int col0 = 0; col0 < strlen; col0 += lXDIM) {
                const int cole = mymin((int)lXDIM, strlen - col0);

                #pragma omp simd aligned(tmpdpdiagbuffers:memalignment)
                for(int xi = 0; xi < cole; xi++) {
                    int paired = tmpdpdiagbuffers[strdst + (col0 + xi) + nposs * lTMPBUFFNDX_POSIT];//int<-float
                    flgs[xi] = (0 <= paired && paired < strlen && (col0 + xi) < paired);
                    poss[xi] = paired;
                }

                #pragma omp simd aligned(tmpdpdiagbuffers:memalignment)
                for(int xi = 0; xi < cole; xi++) {
                    if(flgs[xi]) {
                        //verifying reciprocal pairing:
                        int paired_rcp = tmpdpdiagbuffers[strdst + poss[xi] + nposs * lTMPBUFFNDX_POSIT];
                        //if pairing is mutually consistent:
                        flgs[xi] = (paired_rcp == (col0 + xi));
                    }
                }

                #pragma omp simd aligned(pmbeg:PMBSdatalignment)
                for(int xi = 0; xi < cole; xi++) {
                    if(flgs[xi]) {
                        PMBatchStrData::SetFieldAt<CHTYPE,pmv2Dss>(pmbeg, strdst + (col0 + xi), pmnasOPEN);
                        PMBatchStrData::SetFieldAt<CHTYPE,pmv2Dss>(pmbeg, strdst + poss[xi], pmnasCLOSE);
                    }
                }
            }

            // #pragma omp simd aligned(pmbeg:PMBSdatalignment) aligned(tmpdpdiagbuffers:memalignment)
            // for(int xi = 0; xi < strlen; xi++) {
            //     //position paired with:
            //     int paired = tmpdpdiagbuffers[strdst + xi + nposs * lTMPBUFFNDX_POSIT];//int<-float
            //     if(0 <= paired && paired < strlen && xi < paired) {
            //         //verifying reciprocal pairing:
            //         int paired_rcp = tmpdpdiagbuffers[strdst + paired + nposs * lTMPBUFFNDX_POSIT];
            //         //if pairing is mutually consistent:
            //         if(paired_rcp == xi) {
            //             PMBatchStrData::SetFieldAt<CHTYPE,pmv2Dss>(pmbeg, strdst + xi, pmnasOPEN);
            //             PMBatchStrData::SetFieldAt<CHTYPE,pmv2Dss>(pmbeg, strdst + paired, pmnasCLOSE);
            //         }
            //     }
            // }
        }
    }
}

// -------------------------------------------------------------------------
// nass_initialize_kernel_helper: initialize temporary memory buffer for 
// calculating nucleic acid secondary structures;
// tmpdpdiagbuffers, temporary buffer for distances and positions;
// nposs, total number of (query or reference) positions in the chunk;
//
void MpSecStr::nass_initialize_kernel_helper(
    float *const __RESTRICT__ tmpdpdiagbuffers,
    const int nposs)
{
    enum{
        lXDIM = MPSS_NASSINIT_XDIM
    };

    MYMSG("MpSecStr::nass_initialize_kernel_helper", 4);
    // static const std::string preamb = "MpSecStr::nass_initialize_kernel_helper: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    constexpr int memalignment = CuMemoryBase::GetMinMemAlignment();
    //execution configuration:
    const int nblocks_x = (nposs + lXDIM - 1) / lXDIM;

    //NOTE: constants shared by definition;
    #pragma omp parallel for num_threads(nthreads) default(shared) //shared(nposs)
    for(int bi = 0; bi < nblocks_x; bi++)
    {//threads process blocks
        const int ip0 = bi * lXDIM;
        const int ipe = mymin(ip0 + lXDIM, nposs);
        #pragma omp simd aligned(tmpdpdiagbuffers:memalignment)
        for(int mi = ip0; mi < ipe; mi++) {
            tmpdpdiagbuffers[mi + nposs * lTMPBUFFNDX_POSIT] = -1.0f;
            tmpdpdiagbuffers[mi + nposs * lTMPBUFFNDX_DEVIA] = 9999.9f;
            tmpdpdiagbuffers[mi + nposs * lTMPBUFFNDX_MUTEX] = 0.0f;
        }
    }
}

// -------------------------------------------------------------------------
// nass_calcdistances_kernel_helper: calculate relevant pairwise distances 
// between residues for all query OR reference nucleic acid structures in the 
// chunk;
// tmpdpdiagbuffers, temporary buffer for distances and positions;
// nposs, total number of reference positions in the chunk;
// atomtype, nucleic acid atom type processed;
// 
void MpSecStr::nass_calcdistances_kernel_helper(
    float *const __RESTRICT__ tmpdpdiagbuffers,
    const char *const *const pmbeg, const char *const *const /*pmend*/,
    const int nstrs, const int nposs, const int str1len, const int atomtype)
{
    enum{
        lXDIM = MPSS_NARESDST_XDIM,
        lYDIM = MPSS_NARESDST_YDIM
    };

    MYMSG("MpSecStr::nass_calcdistances_kernel_helper", 4);
    // static const std::string preamb = "MpSecStr::nass_calcdistances_kernel_helper: ";
    static const int nthreads = CLOptions::GetCPU_THREADS();
    //execution configurations for secondary structure calculation:
    //process MPSS_CALCSTR_XDIM positions along the structures in parallel:
    const int nblocks_y = (str1len + lYDIM - 1) / lYDIM;
    const int nblocks_z = nstrs;

    //cache for coordinates: 
    float crdsy[lYDIM][pmv2DNoElems];
    float crdsx[lXDIM][pmv2DNoElems];
    float tmpdx[lXDIM];//temporary buffer
    //cache for distances along structure positions:
    float dstsy[lXDIM];
    int possy[lXDIM];
    //cache for nucleotides:
    char rsdsy[lYDIM];
    char rsdsx[lXDIM];

    // #pragma omp threadprivate(crds, dsts)

    //NOTE: constant scalars and pointers, pmbeg, pmend, shared by definition:
    #pragma omp parallel for num_threads(nthreads) default(shared) \
        private(crdsy,crdsx, tmpdx, dstsy,possy, rsdsy,rsdsx) \
        collapse(2)
    for(int si = 0; si < nblocks_z; si++)
        for(int bi = 0; bi < nblocks_y; bi++)
        {//threads process blocks
            const int strlen = PMBatchStrData::GetLengthAt(pmbeg, si);
            const int strdst = PMBatchStrData::GetAddressAt(pmbeg, si);
            const int typex = PMBatchStrData::GetFieldAt<INTYPE,pmv2D_Ins_Ch_Ord>(pmbeg, strdst);
            const int type = (GetMoleculeType(typex) == gtmtNA);
            const int ndx0 = bi * lYDIM;
            const int ndxe = mymin((int)lYDIM, strlen - ndx0);

            if(type) {
                #pragma omp simd aligned(pmbeg:PMBSdatalignment)
                for(int yi = 0; yi < ndxe; yi++) {
                    crdsy[yi][pmv2DX] = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DX>(pmbeg, strdst + ndx0 + yi);
                    crdsy[yi][pmv2DY] = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DY>(pmbeg, strdst + ndx0 + yi);
                    crdsy[yi][pmv2DZ] = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DZ>(pmbeg, strdst + ndx0 + yi);
                    rsdsy[yi] = PMBatchStrData::GetFieldAt<CHTYPE,pmv2Drsd>(pmbeg, strdst + ndx0 + yi);
                }
            }

            for(int yi = 0; type && yi < ndxe; yi++)
            {
                #pragma omp simd
                for(int xi = 0; xi < lXDIM; xi++) {
                    dstsy[xi] = 9999.9f;
                    possy[xi] = -1;
                }

                for(int col0 = 0; col0 < strlen; col0 += lXDIM) {
                    const int cole = mymin((int)lXDIM, strlen - col0);

                    #pragma omp simd aligned(pmbeg:PMBSdatalignment)
                    for(int xi = 0; xi < cole; xi++) {
                        crdsx[xi][pmv2DX] = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DX>(pmbeg, strdst + col0 + xi);
                        crdsx[xi][pmv2DY] = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DY>(pmbeg, strdst + col0 + xi);
                        crdsx[xi][pmv2DZ] = PMBatchStrData::GetFieldAt<float,pmv2DCoords+pmv2DZ>(pmbeg, strdst + col0 + xi);
                        rsdsx[xi] = PMBatchStrData::GetFieldAt<CHTYPE,pmv2Drsd>(pmbeg, strdst + col0 + xi);
                        tmpdx[xi] = 9999.9f;
                    }

                    #pragma omp simd
                    for(int xi = 0; xi < cole; xi++) {
                        bool cond = GetNAPairingCondition(rsdsy[yi], &rsdsx[xi]);
                        if(cond && 2 < abs((col0 + xi) - (ndx0 + yi))) {
                            float dst = GetDistance_yx(crdsy, yi,  crdsx, xi);
                            tmpdx[xi] = dst;
                        }
                    }

                    #pragma omp simd
                    for(int xi = 0; xi < cole; xi++) {
                        float dev = GetNADstDeviation(atomtype, &tmpdx[xi]);
                        tmpdx[xi] = dev;
                    }

                    #pragma omp simd
                    for(int xi = 0; xi < cole; xi++) {
                        float dev = tmpdx[xi];
                        if(dev < dstsy[xi]) { dstsy[xi] = dev; possy[xi] = col0 + xi; }
                    }
                }

                float devmin = 9999.9f;
                int colmin = 0;

                #pragma omp simd reduction(min:devmin)
                for(int xi = 0; xi < lXDIM; xi++) {
                    float dev = dstsy[xi];
                    if(dev < devmin) { devmin = dev; colmin = possy[xi]; }
                }

                if(devmin < gtnaat_LUB_AVG_ERROR) {
                    tmpdpdiagbuffers[strdst + (ndx0 + yi) + nposs * lTMPBUFFNDX_POSIT] = colmin;
                    tmpdpdiagbuffers[strdst + (ndx0 + yi) + nposs * lTMPBUFFNDX_DEVIA] = devmin;
                }
            }
        }
}
