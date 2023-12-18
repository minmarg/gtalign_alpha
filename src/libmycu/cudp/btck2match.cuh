/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __btck2match_cuh__
#define __btck2match_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gproc/btckcoords.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"

// =========================================================================
// BtckToMatched32x: kernel for copying with 32x unrolling the 
// coordinates of matched (aligned) positions to destination location 
template<bool ANCHORRGN, bool BANDED>
__global__
void BtckToMatched32x(
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint dbxpad,
    const uint maxnsteps,
    const uint stepnumber,
    const char* __restrict__ btckdata,
    float* __restrict__ wrkmemaux,
    float* __restrict__ tmpdpalnpossbuffer
);


// =========================================================================
// -------------------------------------------------------------------------
// GetTerminalCellXY: return the reference and query poitions of the 
// terminal cell (cell (x,y); y, query; x, reference pos.) wrt anchor 
// region and alignment bandwidth if they are used;
// ANCHORRGN, template parameter for considering the boundaries 
// implied by the anchor region;
// BANDED, template parameter to indicate banded alignment;
// x, position along the reference (db) direction;
// y, position along the query direction;
// 
template<bool ANCHORRGN, bool BANDED>
__device__ __forceinline__ 
void GetTerminalCellXY(
    int& x, int& y,
    int qrylen, int dbstrlen,
    int qrypos, int rfnpos, int /* fraglen */)
{
    //terminal cell is the bottom-right-most cell
    y = qrylen - 1; x = dbstrlen - 1;

    if(ANCHORRGN) {
        //as long as fraglen>0, traversing through the 
        //anchor region will always reach the bottom-right corner
    }

    if(BANDED) {
        int delta = rfnpos - qrypos;
        //equation of the lower line: x-delta=y-b;
        //equation of the upper line: x-delta=y+b;
        //b, bandwidth/2; delta=x0-y0, where (x0,y0) is the middle point;
        //x coordinate of the lower line at y=qrylen-1:
        int xylatqr = qrylen-1 - CUDP_BANDWIDTH_HALF + delta;
        //x coordinate of the upper line at y=qrylen-1:
        int xyuatqr = qrylen-1 + CUDP_BANDWIDTH_HALF + delta;
        if(xylatqr < x && x < xyuatqr) return;
        if(xyuatqr <= x) {x = myhdmax(0,xyuatqr-1); return;}
        //y coordinate of the lower line at x=dbstrlen-1:
        xylatqr = dbstrlen-1 + CUDP_BANDWIDTH_HALF - delta;
        //y coordinate of the upper line at x=dbstrlen-1:
        //xyuatqr = dbstrlen-1 - CUDP_BANDWIDTH_HALF - delta;
        if(xylatqr <= y) {y = myhdmax(0,xylatqr-1); return;}
    }
}

// =========================================================================
// -------------------------------------------------------------------------

#endif//__btck2match_cuh__
