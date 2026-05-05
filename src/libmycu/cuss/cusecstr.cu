/***************************************************************************
 *   Copyright (C) 2021-2026 Mindaugas Margelevicius                       *
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
#include "libmycu/cuss/sskna.cuh"
#include "libmycu/cuss/ssk.cuh"
#include "cusecstr.cuh"

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
    uint nqyposs, uint ndbCposs, 
    uint qystr1len, uint dbstr1len, 
    uint qystrnlen, uint dbstrnlen, 
    uint dbxpad,
    float* __restrict__ tmpdpdiagbuffers)
{
    calc_secstr_protein(streamproc,
        nqystrs, ndbCstrs,  nqyposs, ndbCposs, 
        qystr1len, dbstr1len,  qystrnlen, dbstrnlen, dbxpad);

    calc_secstr_na<SSK_STRUCTS_QRIES>(streamproc,
        nqystrs, nqyposs, qystr1len, dbxpad,  tmpdpdiagbuffers);

    calc_secstr_na<SSK_STRUCTS_REFNS>(streamproc,
        ndbCstrs, ndbCposs, dbstr1len, dbxpad,  tmpdpdiagbuffers);
}

// -------------------------------------------------------------------------
// calc_secstr_protein: calculate secondary structure for protein query and
// reference 3D structures;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
void cusecstr::calc_secstr_protein(
    cudaStream_t streamproc, 
    uint nqystrs, uint ndbCstrs, 
    uint /*nqyposs*/, uint /* ndbCposs */, 
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

// -------------------------------------------------------------------------
// calc_secstr_na: calculate secondary structure for nucleic acid query and
// reference 3D structures;
// qystr1len, length of the largest query;
// dbstr1len, length of the largest reference;
// qystrnlen, length of the smallest query;
// dbstrnlen, length of the smallest reference;
//
template<int STRUCTS>
void cusecstr::calc_secstr_na(
    cudaStream_t streamproc,
    uint nstrs, uint nposs, uint str1len,
    uint /* dbxpad */,
    float* __restrict__ tmpdpdiagbuffers)
{
    static const int atomtype = CLOptions::GetI_NATOM_type();

    //configuration for initialization
    dim3 nthrds_initsskna(CUSS_NASSINIT_XDIM,CUSS_NASSINIT_YDIM,1);
    dim3 nblcks_initsskna(
        (nposs + CUSS_NASSINIT_XDIM - 1)/CUSS_NASSINIT_XDIM, 1,1);

    Initialize_NASS<<<nblcks_initsskna,nthrds_initsskna,0,streamproc>>>
        (nposs, tmpdpdiagbuffers);
    MYCUDACHECKLAST;

    // //configuration for calculating inter-residue distances
    // dim3 nthrds_dstsskna(CUSS_NARESDST_XYDIM,CUSS_NARESDST_XYDIM,1);
    // dim3 nblcks_dstsskna(
    //     (str1len + (2 * CUSS_NARESDST_XYDIM) - 1) / (2 * CUSS_NARESDST_XYDIM),
    //     (str1len + CUSS_NARESDST_XYDIM - 1) / CUSS_NARESDST_XYDIM,
    //     nstrs);

    // CalcDistances_NASS_CC7<STRUCTS>
    //     <<<nblcks_dstsskna,nthrds_dstsskna,0,streamproc>>>
    //         (atomtype, nposs, tmpdpdiagbuffers);
    // MYCUDACHECKLAST;

    //configuration for calculating inter-residue distances
    dim3 nthrds_dstsskna(CUSS_NARESDST_XYDIM,CUSS_NARESDST_XYDIM,1);
    dim3 nblcks_dstsskna(1, (str1len + CUSS_NARESDST_XYDIM - 1) / CUSS_NARESDST_XYDIM, nstrs);

    CalcDistances_NASS<STRUCTS>
        <<<nblcks_dstsskna,nthrds_dstsskna,0,streamproc>>>
            (atomtype, nposs, tmpdpdiagbuffers);
    MYCUDACHECKLAST;

    //configuration for secondary structure assignment
    dim3 nthrds_asgnsskna(CUSS_NACALCSTR_XDIM,1,1);
    dim3 nblcks_asgnsskna(nstrs, 1,1);

    CalcSecStrs_NASS<STRUCTS>
        <<<nblcks_asgnsskna,nthrds_asgnsskna,0,streamproc>>>
            (nposs, tmpdpdiagbuffers);
    MYCUDACHECKLAST;

}

// =========================================================================
// Instantiations
//
#define INSTANTIATE_cusecstr_calc_secstr_na(tpSTRUCTS) \
    template void cusecstr::calc_secstr_na<tpSTRUCTS>( \
        cudaStream_t streamproc, \
        uint nstrs, uint nposs, uint str1len, uint dbxpad, \
        float* __restrict__ tmpdpdiagbuffers);

INSTANTIATE_cusecstr_calc_secstr_na(SSK_STRUCTS_QRIES);
INSTANTIATE_cusecstr_calc_secstr_na(SSK_STRUCTS_REFNS);

// -------------------------------------------------------------------------
