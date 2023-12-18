/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __linear_scoring_cuh__
#define __linear_scoring_cuh__

#include "libutil/macros.h"
#include "libgenp/gproc/gproc.h"
#include "libgenp/gdats/PM2DVectorFields.h"
#include "libmymp/mpstages/linearscoringbase.h"
#include "libmycu/cucom/cucommon.h"
#include "libmycu/cuproc/cuprocconf.h"
#include "libmycu/custages/fragment.cuh"
#include "libmycu/custages/fields.cuh"


// PositionalScoresFromIndexLinear: calculate scores at each reference 
// position for following reduction, using index; scores follow from 
// superpositions based on fragments;
template<int SECSTRFILT = 0>
__global__
void PositionalScoresFromIndexLinear(
    const int stacksize,
    const bool stepx5,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int qryfragfct,
    const int rfnfragfct,
    const int fragndx,
    const float* __restrict__ wrkmemtm,
    float* __restrict__ tmpdpdiagbuffers
);

// ReduceScoresLinear: reduce positional scores obtained previously; 
__global__
void ReduceScoresLinear(
    const bool stepx5,
    const uint nqystrs,
    const uint ndbCstrs,
    const uint ndbCposs,
    const uint maxnsteps,
    const int qryfragfct,
    const int rfnfragfct,
    const int fragndx,
    const float* __restrict__ tmpdpdiagbuffers,
    float* __restrict__ wrkmemaux
);

// SaveBestScoreAndConfigLinear: save best scores and associated 
// configuration of query and reference positions along with fragment length;
__global__ void SaveBestScoreAndConfigLinear(
    const bool stepx5,
    const uint ndbCstrs,
    const uint maxnsteps,
    const int qryfragfct,
    const int rfnfragfct,
    const int fragndx,
    float* __restrict__ wrkmemaux
);

// SaveBestScoreAndConfigAmongBestsLinear: save best scores and respective 
// fragment configuration for transformation matrices by considering all 
// partial best scores calculated over all fragment factors;
__global__ void SaveBestScoreAndConfigAmongBestsLinear(
    const uint ndbCstrs,
    const uint maxnsteps,
    float* __restrict__ wrkmemaux
);

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// NNByIndex: find nearest node to the atom with the given coordinates 
// nestndx, index of the nearest node;
// SECSTRFILT, template parameter, find nearest neighbour with matching ss;
// nestdst, squared distance to the nearest node;
// rx,ry,rz iteratively;
// rss, reference secondary structure at the position under process;
// qrydst, beginning address of a query structure;
// root, root node index;
// dimndx, starting dimension for searching in the index;
// stack, stack for traversing the index tree iteratively;
// 
template<int SECSTRFILT>
__device__ __forceinline__
void NNByIndex(
    int STACKSIZE,
    int& nestndx, float& nestdst,
    float rx, float ry, float rz,
    char rss,
    int qrydst, int root, int dimndx,
    float* __restrict__ stack)
{
    int stackptr = 0;//stack pointer
    int nvisited = 0;//number of visited nodes

    //while stack is nonempty or the current node is not a terminator
    while(0 < stackptr || 0 <= root)
    {
        if(0 <= root) {
            nvisited++;
            if(CUSF_TBSP_INDEX_SCORE_MAXDEPTH <= nvisited)
                break;
            int qryorgndx;
            char qss;
            if(SECSTRFILT == 1) {
                //map the index in the index tree to the original structure position index
                qryorgndx = GetIndxdQueryOrgndx(qrydst + root);
                qss = GetQuerySS(qrydst + qryorgndx);
            }
            //READ coordinates:
            float qx = GetIndxdQueryCoord<pmv2DX>(qrydst + root);
            float qy = GetIndxdQueryCoord<pmv2DY>(qrydst + root);
            float qz = GetIndxdQueryCoord<pmv2DZ>(qrydst + root);
            float dst2 = distance2(qx, qy, qz,  rx, ry, rz);
            if((nestndx < 0 || dst2 < nestdst) 
                //&& ((SECSTRFILT == 1)? !helix_strnd(rss, qss): 1)
                && ((SECSTRFILT == 1)? (rss == qss): 1)
            ) {
                nestdst = dst2;
                if(SECSTRFILT == 1) nestndx = qryorgndx;
                else nestndx = root;
            }
            if(nestdst == 0.0f)
                break;
            float diffc = (dimndx == 0)? (qx - rx): ((dimndx == 1)? (qy - ry): (qz - rz));
            if(pmv2DZ < ++dimndx) dimndx = 0;
            if(stackptr < STACKSIZE) {
                stack[nStks_ * stackptr + stkNdx_Dim_] = NNSTK_COMBINE_NDX_DIM(root, dimndx);
                stack[nStks_ * stackptr + stkDiff_] = diffc;
                stackptr++;
            }
            root = (0.0f < diffc)//READ
                ? GetIndxdQueryBranchLeft(qrydst + root)
                : GetIndxdQueryBranchRight(qrydst + root);
        }
        else {
            bool cond = false;
            float diffc = 0.0f;
            for(; 0 < stackptr && !cond;) {
                stackptr--;
                int comb = stack[nStks_ * stackptr + stkNdx_Dim_];
                diffc = stack[nStks_ * stackptr + stkDiff_];
                root = NNSTK_GET_NDX_FROM_COMB(comb);
                dimndx = NNSTK_GET_DIM_FROM_COMB(comb);
                cond = (diffc * diffc < nestdst);
            }
            if(!cond) break;
            root = (0.0f < diffc)//READ
                ? GetIndxdQueryBranchRight(qrydst + root)
                : GetIndxdQueryBranchLeft(qrydst + root);
        }
    }

    if(SECSTRFILT == 1) {
        if(64.0f < nestdst) nestndx = -1;
    } else {
        //map the index in the index tree to the original structure position index
        if(0 <= nestndx) nestndx = GetIndxdQueryOrgndx(qrydst + nestndx);
    }
}

// NNByIndex version with different arguments
//
template<int SECSTRFILT>
__device__ __forceinline__
void NNByIndex(
    int STACKSIZE,
    int& nestndx,
    float& qxn, float& qyn, float& qzn, 
    float rx, float ry, float rz, char rss,
    int qrydst, int root, int dimndx,
    float* __restrict__ stack)
{
    int stackptr = 0;//stack pointer
    int nvisited = 0;//number of visited nodes
    float nestdst = 9.9e6f;//squared best distance

    //while stack is nonempty or the current node is not a terminator
    while(0 < stackptr || 0 <= root)
    {
        if(0 <= root) {
            nvisited++;
            if(CUSF_TBSP_INDEX_SCORE_MAXDEPTH <= nvisited)
                break;
            int qryorgndx;
            char qss;
            if(SECSTRFILT == 1) {
                //map the index in the index tree to the original structure position index
                qryorgndx = GetIndxdQueryOrgndx(qrydst + root);
                qss = GetQuerySS(qrydst + qryorgndx);
            }
            //READ coordinates:
            float qx = GetIndxdQueryCoord<pmv2DX>(qrydst + root);
            float qy = GetIndxdQueryCoord<pmv2DY>(qrydst + root);
            float qz = GetIndxdQueryCoord<pmv2DZ>(qrydst + root);
            float dst2 = distance2(qx, qy, qz,  rx, ry, rz);
            if((nestndx < 0 || dst2 < nestdst) 
                //&& ((SECSTRFILT == 1)? !helix_strnd(rss, qss): 1)
                && ((SECSTRFILT == 1)? (rss == qss): 1)
            ) {
                nestdst = dst2;
                qxn = qx; qyn = qy; qzn = qz;
                if(SECSTRFILT == 1) nestndx = qryorgndx;
                else nestndx = root;
            }
            if(nestdst == 0.0f)
                break;
            float diffc = (dimndx == 0)? (qx - rx): ((dimndx == 1)? (qy - ry): (qz - rz));
            if(pmv2DZ < ++dimndx) dimndx = 0;
            if(stackptr < STACKSIZE) {
                stack[nStks_ * stackptr + stkNdx_Dim_] = NNSTK_COMBINE_NDX_DIM(root, dimndx);
                stack[nStks_ * stackptr + stkDiff_] = diffc;
                stackptr++;
            }
            root = (0.0f < diffc)//READ
                ? GetIndxdQueryBranchLeft(qrydst + root)
                : GetIndxdQueryBranchRight(qrydst + root);
        }
        else {
            bool cond = false;
            float diffc = 0.0f;
            for(; 0 < stackptr && !cond;) {
                stackptr--;
                int comb = stack[nStks_ * stackptr + stkNdx_Dim_];
                diffc = stack[nStks_ * stackptr + stkDiff_];
                root = NNSTK_GET_NDX_FROM_COMB(comb);
                dimndx = NNSTK_GET_DIM_FROM_COMB(comb);
                cond = (diffc * diffc < nestdst);
            }
            if(!cond) break;
            root = (0.0f < diffc)//READ
                ? GetIndxdQueryBranchRight(qrydst + root)
                : GetIndxdQueryBranchLeft(qrydst + root);
        }
    }

    if(SECSTRFILT == 1) {
        if(64.0f < nestdst) nestndx = -1;
    } else {
        //map the index in the index tree to the original structure position index
        if(0 <= nestndx) nestndx = GetIndxdQueryOrgndx(qrydst + nestndx);
    }
}

// NNByIndexReference version for reference
//
template<int SECSTRFILT>
__device__ __forceinline__
void NNByIndexReference(
    int STACKSIZE,
    int& nestndx,
    float& rxn, float& ryn, float& rzn, 
    float qx, float qy, float qz, char qss,
    int dbstrdst, int root, int dimndx,
    float* __restrict__ stack)
{
    int stackptr = 0;//stack pointer
    int nvisited = 0;//number of visited nodes
    float nestdst = 9.9e6f;//squared best distance

    //while stack is nonempty or the current node is not a terminator
    while(0 < stackptr || 0 <= root)
    {
        if(0 <= root) {
            nvisited++;
            if(CUSF_TBSP_INDEX_SCORE_MAXDEPTH <= nvisited)
                break;
            int rfnorgndx;
            char rss;
            if(SECSTRFILT == 1) {
                //map the index in the index tree to the original structure position index
                rfnorgndx = GetIndxdDbStrOrgndx(dbstrdst + root);
                rss = GetDbStrSS(dbstrdst + rfnorgndx);
            }
            //READ coordinates:
            float rx = GetIndxdDbStrCoord<pmv2DX>(dbstrdst + root);
            float ry = GetIndxdDbStrCoord<pmv2DY>(dbstrdst + root);
            float rz = GetIndxdDbStrCoord<pmv2DZ>(dbstrdst + root);
            float dst2 = distance2(qx, qy, qz,  rx, ry, rz);
            if((nestndx < 0 || dst2 < nestdst) 
                //&& ((SECSTRFILT == 1)? !helix_strnd(rss, qss): 1)
                && ((SECSTRFILT == 1)? (rss == qss): 1)
            ) {
                nestdst = dst2;
                rxn = rx; ryn = ry; rzn = rz;
                if(SECSTRFILT == 1) nestndx = rfnorgndx;
                else nestndx = root;
            }
            if(nestdst == 0.0f)
                break;
            float diffc = (dimndx == 0)? (rx - qx): ((dimndx == 1)? (ry - qy): (rz - qz));
            if(pmv2DZ < ++dimndx) dimndx = 0;
            if(stackptr < STACKSIZE) {
                stack[nStks_ * stackptr + stkNdx_Dim_] = NNSTK_COMBINE_NDX_DIM(root, dimndx);
                stack[nStks_ * stackptr + stkDiff_] = diffc;
                stackptr++;
            }
            root = (0.0f < diffc)//READ
                ? GetIndxdDbStrBranchLeft(dbstrdst + root)
                : GetIndxdDbStrBranchRight(dbstrdst + root);
        }
        else {
            bool cond = false;
            float diffc = 0.0f;
            for(; 0 < stackptr && !cond;) {
                stackptr--;
                int comb = stack[nStks_ * stackptr + stkNdx_Dim_];
                diffc = stack[nStks_ * stackptr + stkDiff_];
                root = NNSTK_GET_NDX_FROM_COMB(comb);
                dimndx = NNSTK_GET_DIM_FROM_COMB(comb);
                cond = (diffc * diffc < nestdst);
            }
            if(!cond) break;
            root = (0.0f < diffc)//READ
                ? GetIndxdDbStrBranchRight(dbstrdst + root)
                : GetIndxdDbStrBranchLeft(dbstrdst + root);
        }
    }

    if(SECSTRFILT == 1) {
        if(64.0f < nestdst) nestndx = -1;
    } else {
        //map the index in the index tree to the original structure position index
        if(0 <= nestndx) nestndx = GetIndxdDbStrOrgndx(dbstrdst + nestndx);
    }
}

// -------------------------------------------------------------------------
// NNNaiveTest: naive version of finding nearest node to the atom with the 
// given coordinates for testing;
// nestndx, index of the nearest node;
// nestdst, squared distance to the nearest node;
// rx,ry,rz, coordinates;
// qrydst, beginning address of a query structure;
// qrylen, query length;
// 
__device__ __forceinline__
void NNNaiveTest(
    int& nestndx, float& nestdst,
    float rx, float ry, float rz,
    int qrydst, int qrylen)
{
    for(int n = 0; n < qrylen; n++) {
        //READ coordinates:
        float qx = GetQueryCoord<pmv2DX>(qrydst + n);
        float qy = GetQueryCoord<pmv2DY>(qrydst + n);
        float qz = GetQueryCoord<pmv2DZ>(qrydst + n);
        float dst2 = distance2(qx, qy, qz,  rx, ry, rz);
        if(nestndx < 0 || dst2 < nestdst) {
            nestdst = dst2;
            nestndx = n;
        }
        if(nestdst == 0.0f)
            return;
    }
}

// -------------------------------------------------------------------------

#endif//__linear_scoring_cuh__
