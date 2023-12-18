/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/cnsts.h"
#include "libutil/macros.h"
#include "libutil/CLOptions.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"

#include "libmycu/cucom/cucommon.h"
#include "libmycu/cucom/warpscan.cuh"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/culayout/cuconstant.cuh"

#include "libmycu/custages/stagecnsts.cuh"
#include "libmycu/cuss/ssk.cuh"
#include "cusecstr.cuh"

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// calc_secstr: calculate secondary structure for the query and reference 
// 3D structures;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void cusecstr::calc_secstr(
    cudaStream_t streamproc, 
    uint nqystrs, uint ndbCstrs, 
    uint /*nqyposs*/, uint /*ndbCposs*/, 
    uint qystr1len, uint dbstr1len, 
    uint /*qystrnlen*/, uint /*dbstrnlen*/, 
    uint /*dbxpad*/)
{
    //execution configurations for secondary structure calculation:
    //block processes CUSS_CALCSTR_XDIM positions along one structure:
    //NOTE: ndbCstrs and nqystrs cannot be greater than 65535 (ensured by JobDispatcher);
    //configuration for queries
    dim3 nthrds_qryssk(CUSS_CALCSTR_XDIM,1,1);
    dim3 nblcks_qryssk(
        (qystr1len + CUSS_CALCSTR_XDIM - 1)/CUSS_CALCSTR_XDIM,
        nqystrs,1);

    //configuration for references
    dim3 nthrds_rfnssk(CUSS_CALCSTR_XDIM,1,1);
    dim3 nblcks_rfnssk(
        (dbstr1len + CUSS_CALCSTR_XDIM - 1)/CUSS_CALCSTR_XDIM,
        ndbCstrs,1);

    CalcSecStrs<SSK_STRUCTS_QRIES>
        <<<nblcks_qryssk,nthrds_qryssk,0,streamproc>>>();
    MYCUDACHECKLAST;

    CalcSecStrs<SSK_STRUCTS_REFNS>
        <<<nblcks_rfnssk,nthrds_rfnssk,0,streamproc>>>();
    MYCUDACHECKLAST;
}
