/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __PM2DVectorFields_h__
#define __PM2DVectorFields_h__

// #include "libutil/mybase.h"

// for the case when FPTYPE==INTYPE
// #define FPTYPEisINTYPE
// #define LNTYPEisINTYPE
// #define CHTYPEisINTYPE

//NOTE: for alignment and proper data type conversion by CUDA in a device,
// make all types of the same size

#define FPTYPE      float
#define SZFPTYPE    (sizeof(FPTYPE))
#define FP0 0.0f

#define LNTYPE      unsigned int
#define SZLNTYPE    (sizeof(LNTYPE))

#define INTYPE      int //short
#define SZINTYPE    (sizeof(INTYPE))

#ifdef CHTYPEisINTYPE
#   define CHTYPE   int
#else
#   define CHTYPE   char
#endif
#define SZCHTYPE    (sizeof(CHTYPE))

//make integer variable by combining 1-byte residue insertion code, 
// chain id and serial number (order):
#define PM2D_MAKEINT_Ins_Ch_Ord(Ins, Ch, Ord) \
    ( (((Ins)&0xff)<<16) | (((Ch)&0xff)<<8) | ((Ord)&0xff) )
//decompose:
#define PM2D_GET_INSCODE(Ins_Ch_Ord)  ( ((Ins_Ch_Ord)>>16)&0xff )
#define PM2D_GET_CHID(Ins_Ch_Ord)  ( ((Ins_Ch_Ord)>>8)&0xff )
#define PM2D_GET_CHORD(Ins_Ch_Ord)  ( (Ins_Ch_Ord)&0xff )

//secondary structure assignment macros
#define pmvLOOP ' '
#define pmvSTRND 'e'
#define pmvHELIX 'h'
#define pmvTURN 't'

#if defined(__CUDA_ARCH__)
__host__ __device__ __forceinline__
#else
inline
#endif
bool isloop(const char sss) {return (sss == pmvTURN) || (sss == pmvLOOP);}

#if defined(__CUDA_ARCH__)
__host__ __device__ __forceinline__
#else
inline
#endif
bool helix_strnd(const char ss1, const char ss2) {
    return (ss1 == pmvHELIX && ss2 == pmvSTRND) ||
        (ss2 == pmvHELIX && ss1 == pmvSTRND);
}

//constants
enum TPSVectorConsts {
        pmv2DX,
        pmv2DY,
        pmv2DZ,
        pmv2DNoElems,//number of coordinates (dimensions)
};

//NOTE: data alignment is NOT required, as a separate buffer is 
// allocated for each field

enum TPM2DVectorFields {
//fields and their indices of the 2D vector, representing structure-specific data for a 
//given structure:
//NOTE: pps2DLen and pps2DDist have to be adjacent for efficient reading in the device!
        pps2DLen = 0,//structure length; [SHORT INT]
        pps2DDist = pps2DLen + 1,//total number of positions up to this structure; [UINT]
        pps2DType = pps2DDist + 1,//structure type (protein, RNA); [SHORT INT]
//{{Fields and their indices of 2D vector representing one position of 
//structure for scoring;
//NOTE: structure address of type unsigned short is reasonable as long as the 
// max number of structures is unlikely to be processed in parallel simultaneously (on both CPU and GPU)
        pmv2DCoords = pps2DType + 1,//coordinates; pmv2DNoElems fields in total
        //NOTE: composite field of INSERTION CODE, CHAIN ID, and CHAIN ORDER, 
        // all wrapped in 3 of 4 bytes of an int variable:
        pmv2D_Ins_Ch_Ord = pmv2DCoords + pmv2DNoElems,//; [INT]
        pmv2DResNumber = pmv2D_Ins_Ch_Ord + 1,//residue serial number [SHORT INT]
        pmv2Drsd = pmv2DResNumber + 1,//residue letter; [CHAR]
        pmv2Dss = pmv2Drsd + 1,//secondary structure (SS); [CHAR]; [*OPTIONAL*]
//}}
        pmv2DTotFlds//total number of fields
};
enum {
        pps2DStrFlds = pps2DType + 1,//total number of structure-specific fields
};

struct TPM2DVectorFieldSize {
    constexpr static size_t szvfs_[pmv2DTotFlds] = {
        SZINTYPE,//pps2DLen
        SZLNTYPE,//pps2DDist
        SZINTYPE,//pps2DType
        SZFPTYPE,//pmv2DCoords+0
        SZFPTYPE,//pmv2DCoords+1
        SZFPTYPE,//pmv2DCoords+2
        SZLNTYPE,//pmv2D_Ins_Ch_Ord
        SZINTYPE,//pmv2DResNumber
        SZCHTYPE,//pmv2Drsd
        SZCHTYPE//pmv2Dss
    };
};

enum TPM2DIndexFields {
//{{Fields and their indices of 2D vector representing one position of indexed structure;
        pmv2DNdxCoords = 0,//coordinates; pmv2DNoElems fields in total
        pmv2DNdxLeft = pmv2DNdxCoords + pmv2DNoElems,//pointer (index) to left node in the tree [SHORT INT]
        pmv2DNdxRight = pmv2DNdxLeft + 1,//pointer (index) to right node in the tree [SHORT INT]
        pmv2DNdxOrgndx = pmv2DNdxRight + 1,//original position (node) index [SHORT INT]
//         pmv2DNdxCutDim = pmv2DNdxOrgndx + 1,//cutting dimension [CHAR]
//}}
        pmv2DTotIndexFlds//total number of fields
};

struct TPM2DIndexFieldSize {
    constexpr static size_t szvfs_[pmv2DTotIndexFlds] = {
        SZFPTYPE,//pmv2DNdxCoords+0
        SZFPTYPE,//pmv2DNdxCoords+1
        SZFPTYPE,//pmv2DNdxCoords+2
        SZINTYPE,//pmv2DNdxLeft
        SZINTYPE,//pmv2DNdxRight
        SZINTYPE//pmv2DNdxOrgndx
//         ,SZCHTYPE//pmv2DNdxCutDim
    };
};

#endif//__PM2DVectorFields_h__
