/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
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
            const int ndx0 = bi * XDIM;
            const int strdstndx0 = strdst + ndx0;
            //if thread is not out of bounds
            if(ndx0 < strlen) {
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
