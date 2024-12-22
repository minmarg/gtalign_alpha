/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/mybase.h"

#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "libutil/CLOptions.h"

#include "libmycu/cucom/cudef.h"
#include "CuMemoryBase.h"

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// constructor
//
CuMemoryBase::CuMemoryBase(
    size_t deviceallocsize,
    int nareas)
:
    deviceallocsize_(deviceallocsize),
    nareas_(nareas),
    curmaxdbpos_(0UL),
    curmaxndbstrs_(0UL),
    curmaxdbposspass2_(0UL),
    curmaxdbstrspass2_(0UL),
    sz_heapsections_(NULL)
{
    MYMSG("CuMemoryBase::CuMemoryBase", 4);

    if(nareas_ < 1)
        throw MYRUNTIME_ERROR(
        "CuMemoryBase::CuMemoryBase: Invalid number of memory areas.");

    sz_heapsections_ = (size_t(*)[nDevDataSections])
        malloc(nareas_ * nDevDataSections * sizeof(size_t));

    if(sz_heapsections_ == NULL )       
        throw MYRUNTIME_ERROR(
        "CuMemoryBase::CuMemoryBase: Not enough memory.");

//     for( int a = 0; a < nareas_; a++ )
//         for( int i = 0; i < nDevDataSections; i++ )
//             sz_heapsections_[a][i] = 0UL;
    memset(sz_heapsections_, 0, nareas_ * nDevDataSections * sizeof(size_t));
}

// -------------------------------------------------------------------------
// obligate initialization
//
void CuMemoryBase::Initialize()
{
    if(deviceallocsize_)
    {
        const size_t cszalnment = GetMemAlignment();

        AllocateHeap();

        size_t heapaligned = ALIGN_UP((size_t)GetHeap(), cszalnment);
        for(int a = 0; a < nareas_; a++) {
            sz_heapsections_[a][ddsEndOfPadding] = heapaligned - (size_t)GetHeap();
            //NOTE: the following's ensured by enum when no constant data is present
            //sz_heapsections_[a][ddsEndOfConstantData] = sz_heapsections_[a][ddsEndOfPadding];
        }
    }
}

// -------------------------------------------------------------------------
// destructor
//
CuMemoryBase::~CuMemoryBase()
{
    MYMSG("CuMemoryBase::~CuMemoryBase", 4);

    if(sz_heapsections_) {
        free(sz_heapsections_);
        sz_heapsections_ = NULL;
    }

    DeallocateHeap();
}





// =========================================================================
// CalcMaxDbDataChunkSize: divide device memory to sections given total 
// maximum length of queries; the boundaries of device memory sections are 
// calculated for each memory area
//
size_t CuMemoryBase::CalcMaxDbDataChunkSize(size_t totqrsposs)
{
    MYMSG("CuMemoryBase::CalcMaxDbDataChunkSize", 4);
    const std::string preamb = "CuMemoryBase::CalcMaxDbDataChunkSize: ";
    char msgbuf[BUF_MAX];

    if(deviceallocsize_ <= sz_heapsections_[0][ddsEndOfConstantData]) {
        sprintf(msgbuf, "Insufficient amount of allocated device memory: %zu.", deviceallocsize_);
        throw MYRUNTIME_ERROR(preamb + msgbuf);
    }

    if(nareas_ < 1)
        throw MYRUNTIME_ERROR(preamb + "Invalid number of memory areas.");

    const size_t gapbtwas = 16UL*ONEM;//gap between areas
    const size_t residualsize = deviceallocsize_ - sz_heapsections_[0][ddsEndOfConstantData];
    const size_t areasize = residualsize / nareas_ - (nareas_-1) * gapbtwas;

    size_t chunkdatasize = CalcMaxDbDataChunkSizeHelper(totqrsposs, areasize);

    MsgAddressTable(0, preamb, 3);

    //the first memory area has been configured;
    //derive memory section sizes for the other memory areas
    for(int a = 1; a < nareas_; a++)
    {
        sz_heapsections_[a][ddsBegOfQrsChunk] = sz_heapsections_[0][ddsBegOfQrsChunk];
        sz_heapsections_[a][ddsEndOfQrsChunk] = sz_heapsections_[0][ddsEndOfQrsChunk];

        sz_heapsections_[a][ddsBegOfDbsChunk] = sz_heapsections_[0][ddsBegOfDbsChunk];
        sz_heapsections_[a][ddsEndOfDbsChunk] = sz_heapsections_[0][ddsEndOfDbsChunk];

        sz_heapsections_[a][ddsBegOfQrsIndex] = sz_heapsections_[0][ddsBegOfQrsIndex];
        sz_heapsections_[a][ddsEndOfQrsIndex] = sz_heapsections_[0][ddsEndOfQrsIndex];

        sz_heapsections_[a][ddsBegOfDbsIndex] = sz_heapsections_[0][ddsBegOfDbsIndex];
        sz_heapsections_[a][ddsEndOfDbsIndex] = sz_heapsections_[0][ddsEndOfDbsIndex];

        sz_heapsections_[a][ddsBegOfMtxScores] = sz_heapsections_[a-1][nDevDataSections-1]  +  gapbtwas;
        //sz_heapsections_[a][ddsBegOfMtxScores] = ALIGN_UP(sz_heapsections_[a][ddsBegOfMtxScores],4096);

        for(int s = ddsEndOfMtxScores; s < nDevDataSections; s++)
        {
            sz_heapsections_[a][s] = sz_heapsections_[a][ddsBegOfMtxScores] + 
                (sz_heapsections_[0][s]-sz_heapsections_[0][ddsBegOfMtxScores]);

            if(deviceallocsize_ < sz_heapsections_[a][s]) {
                sprintf(msgbuf, "Size of section %d of area %d exceeds "
                    "device memory allocations: %zu > %zu.",
                    s, a, sz_heapsections_[a][s], deviceallocsize_);
                throw MYRUNTIME_ERROR(preamb + msgbuf);
            }
        }

        MsgAddressTable(a, preamb, 3);

        //TODO: once memory configuration has changed, notify 
        // processing children to check host memory synchronization necessity
        //CheckHostResultsSync();
    }

    return chunkdatasize;
}

// -------------------------------------------------------------------------
// CalcMaxDbDataChunkSizeHelper: recalculate and return a maximum allowed 
// number of positions and memory size for target db structure data 
// so that the implied search space does not exceed allocated memory 
// (representing the maximum allowed limit);
// NOTE: results are saved for the first memory area;
// 
size_t CuMemoryBase::CalcMaxDbDataChunkSizeHelper(
    size_t totqrsposs, size_t residualsize)
{
    MYMSG("CuMemoryBase::CalcMaxDbDataChunkSizeHelper", 4);
    static const std::string preamb = "CuMemoryBase::CalcMaxDbDataChunkSizeHelper: ";
    const size_t cszalnment = GetMemAlignment();

    if(nareas_ < 1 || sz_heapsections_ == NULL)
        throw MYRUNTIME_ERROR(preamb + "Memory access error.");
    if(totqrsposs < 1)
        throw MYRUNTIME_ERROR(preamb + "Invalid number of query positions.");

    size_t tmpmaxndbstrs, maxndbstrs = 0UL;//maximum number of target db structures
    size_t tmpmaxdbstrspass2, maxdbstrspass2 = 0UL;//maximum number of significant db hits
    size_t tmpmaxdbposspass2, maxdbposspass2 = 0UL;//maximum #positions of significant db structures

    size_t tmpmaxdbposs, maxdbposs = 0UL;//maximum #db structure positions
    size_t tmpmaxsizedbposs, maxsizedbposs = 0UL;//maximum size (in bytes) for db structure positions
    size_t tmpmaxsizeqrsposs, maxsizeqrsposs = 0UL;//maximum size for query db structure positions
    size_t tmpmaxsizeqrsindex, maxsizeqrsindex = 0UL;//maximum size for query db index
    size_t tmpmaxsizedbsindex, maxsizedbsindex = 0UL;//maximum size for target db index
    size_t tmpsztfmmtcs, sztfmmtcs = 0UL;//maximum size for transformation matrices
    size_t tmpszsmatrix, szsmatrix = 0UL;//maximum size for scores
    size_t tmpszdpdiag, szdpdiag = 0UL;//maximum size for DP diagonal score buffers
    size_t tmpszdpbottom, szdpbottom = 0UL;//maximum size for DP bottom score buffers
    size_t tmpszdpalnposs, szdpalnposs = 0UL;//maximum size for DP-aligned positions buffers
    size_t tmpszdpmaxcoords, szdpmaxcoords = 0UL;//maximum size for coordinates of max alignment scores
    size_t tmpszdpbtckdat, szdpbtckdat = 0UL;//maximum size for DP backtracking data

    size_t tmpszwrkmem, szwrkmem = 0UL;//maximum size for working memory
    size_t tmpszwrkmemccd, szwrkmemccd = 0UL;//maximum size for working memory of additional CCD
    size_t tmpszwrkmemtmalt, szwrkmemtmalt = 0UL;//maximum size for working memory of alternative TMs
    size_t tmpszwrkmemtm, szwrkmemtm = 0UL;//maximum size for working memory of additional TM data
    size_t tmpszwrkmemtmibest, szwrkmemtmibest = 0UL;//maximum size for memory of iteration-best TM
    size_t tmpszauxwrkmem, szauxwrkmem = 0UL;//maximum size for auxiliary working memory
    size_t tmpszwrkmem2, szwrkmem2 = 0UL;//maximum size for the second division of working memory
    size_t tmpszdp2alndata, szdp2alndata = 0UL;//maximum size for alignment statistics
    size_t tmpszdp2alns, szdp2alns = 0UL;//maximum size for alignments themselves
    size_t tmpszglbvars, szglbvars = 0UL;//maximum size for global variables

    size_t tmpszovlpos, szovlpos = 0UL;//overall size of data excluding maxsizedbposs
    size_t tmpdiff;

    //NOTE: align #positions for 32-bit memory alignment
    totqrsposs = ALIGN_UP(totqrsposs, 4);

    //search space divided by #db structure positions;
    //note: assuming matrices of type float;
    //NOTE: score matrices is not used as they are computed on-the-fly,
    //NOTE: but backtracking matrices are in use:
    const size_t qsspace = totqrsposs * sizeof(char);//sizeof(float);
    //maximum number of binary search iterations here
    const int maxit = 16;//10;//NOTE: 10 may be insufficient when dev-mem is large!
    const size_t permdiff = ONEM;//permissible difference
    //maximum number of allowed target Db structure positions not causing overflow
    const size_t locMAXALLWDPOSS = GetMaxAllowedNumberDbPositions(totqrsposs);
    //percentage of residualsize corresponding to the max value for maxsizedbposs:
    // to maximize the number of target db structure positions
    const float maxperc = 0.3f;
    size_t maxresidual = (size_t)(residualsize * maxperc);
    maxresidual = ALIGN_UP(maxresidual, cszalnment);

    //initial guess:
    tmpmaxdbposs = residualsize / qsspace;
    tmpmaxdbposs = ALIGN_DOWN(tmpmaxdbposs, CUL2CLINESIZE);
    tmpmaxndbstrs = tmpmaxdbposs / CLOptions::GetDEV_EXPCT_DBPROLEN();

    MYMSGBEGl(3)
        char msgbuf[BUF_MAX];
        sprintf(msgbuf,"%sMax allowed # db chunk positions: %zu",
            preamb.c_str(),locMAXALLWDPOSS);
        MYMSG(msgbuf, 3);
    MYMSGENDl

    GetTotalMemoryReqs( 
        totqrsposs, tmpmaxdbposs, maxresidual,
        //query db positions
        &tmpmaxsizeqrsindex,
        &tmpmaxsizeqrsposs,
        //target db positions
        &tmpmaxsizedbsindex,
        &tmpmaxsizedbposs,
        //transformations
        &tmpsztfmmtcs,
        //scores
        &tmpszsmatrix,
        //for DP phase
        &tmpszdpdiag, &tmpszdpbottom, &tmpszdpalnposs,
        &tmpszdpmaxcoords, &tmpszdpbtckdat,
        &tmpmaxdbposspass2, &tmpmaxdbstrspass2,
        &tmpszwrkmem, &tmpszwrkmemccd,
        &tmpszwrkmemtmalt, &tmpszwrkmemtm, &tmpszwrkmemtmibest,
        &tmpszauxwrkmem, &tmpszwrkmem2,
        &tmpszdp2alndata, &tmpszdp2alns,
        //global variables
        &tmpszglbvars,
        //overall size
        &tmpszovlpos
    );

    tmpdiff = tmpmaxdbposs;

    //binary search for finding maximum number of target db structure positions
    for(int i = 0; i < maxit; i++) {
        MYMSGBEGl(5)
            char msgbuf[KBYTE];
            sprintf(msgbuf, "%s%sTotal size adjustment for db structures: "
                        "it %d: %zu; maxdbpos= %zu; szdpdiag= %zu, szdpbottom= %zu, "
                        "szdpalnposs= %zu, szdpmaxcoords= %zu, szdpbtckdat= %zu; szovlpos= %zu; "
                        "leftoversize= %zu", preamb.c_str(), NL, i,
                    tmpmaxsizedbposs, tmpmaxdbposs, tmpszdpdiag, tmpszdpbottom, tmpszdpalnposs,
                    tmpszdpmaxcoords, tmpszdpbtckdat, tmpszovlpos, residualsize);
            MYMSG(msgbuf, 5);
        MYMSGENDl
        tmpdiff /= 2;
        tmpdiff = ALIGN_DOWN(tmpdiff, CUL2CLINESIZE);
        if(tmpdiff == 0)
            break;
        if(tmpszovlpos < residualsize && 
           tmpmaxdbposs < locMAXALLWDPOSS &&
           tmpmaxsizedbposs <= maxresidual) 
        {
            //save this configuration
            maxdbposs = tmpmaxdbposs; 
            maxndbstrs = tmpmaxndbstrs;
            maxsizedbposs = tmpmaxsizedbposs;
            maxsizedbsindex = tmpmaxsizedbsindex;
            maxsizeqrsposs = tmpmaxsizeqrsposs;
            maxsizeqrsindex = tmpmaxsizeqrsindex;
            sztfmmtcs = tmpsztfmmtcs;
            szsmatrix = tmpszsmatrix;
            szdpdiag = tmpszdpdiag; 
            szdpbottom = tmpszdpbottom;
            szdpalnposs = tmpszdpalnposs;
            szdpmaxcoords = tmpszdpmaxcoords; 
            szdpbtckdat = tmpszdpbtckdat;
            maxdbposspass2 = tmpmaxdbposspass2; 
            maxdbstrspass2 = tmpmaxdbstrspass2; 
            szwrkmem = tmpszwrkmem; 
            szwrkmemccd = tmpszwrkmemccd; 
            szwrkmemtmalt = tmpszwrkmemtmalt; 
            szwrkmemtm = tmpszwrkmemtm; 
            szwrkmemtmibest = tmpszwrkmemtmibest; 
            szauxwrkmem = tmpszauxwrkmem; 
            szwrkmem2 = tmpszwrkmem2; 
            szdp2alndata = tmpszdp2alndata; 
            szdp2alns = tmpszdp2alns;
            szglbvars = tmpszglbvars;
            szovlpos = tmpszovlpos;

            tmpmaxdbposs += tmpdiff;
        }
        else
            tmpmaxdbposs -= tmpdiff;

        tmpmaxndbstrs = tmpmaxdbposs / CLOptions::GetDEV_EXPCT_DBPROLEN();

        if(maxdbposs && 
           GetAbsDiff(tmpszovlpos, residualsize) < permdiff)
            break;

        GetTotalMemoryReqs( 
            totqrsposs, tmpmaxdbposs, maxresidual,
            //query db positions
            &tmpmaxsizeqrsindex,
            &tmpmaxsizeqrsposs,
            //target db positions
            &tmpmaxsizedbsindex,
            &tmpmaxsizedbposs,
            //transformations
            &tmpsztfmmtcs,
            //scores
            &tmpszsmatrix,
            //for DP phase
            &tmpszdpdiag, &tmpszdpbottom, &tmpszdpalnposs,
            &tmpszdpmaxcoords, &tmpszdpbtckdat,
            &tmpmaxdbposspass2, &tmpmaxdbstrspass2,
            &tmpszwrkmem, &tmpszwrkmemccd,
            &tmpszwrkmemtmalt, &tmpszwrkmemtm, &tmpszwrkmemtmibest,
            &tmpszauxwrkmem, &tmpszwrkmem2,
            &tmpszdp2alndata, &tmpszdp2alns,
            //global variables
            &tmpszglbvars,
            //overall size
            &tmpszovlpos
        );
    }

    if( maxdbposs < 1 || maxsizedbposs < 1 || 
        szovlpos < 1 || residualsize <= szovlpos) {
        char msgbuf[KBYTE];
        sprintf(msgbuf, "Insufficient amount of residual device memory: %zu%s"
                "(maxszdbpos= %zu, szdpdiag= %zu, szdpbottom= %zu, "
                "szdpalnposs= %zu, szdpmaxcoords= %zu, szdpbtckdat= %zu; ndbpos= %zu; "
                "szovlpos= %zu).",
                residualsize, NL, maxsizedbposs, szdpdiag, szdpbottom, szdpalnposs,
                szdpmaxcoords, szdpbtckdat, maxdbposs, szovlpos );
        throw MYRUNTIME_ERROR(preamb + msgbuf);
    }

    MYMSGBEGl(3)
        char msgbuf[TIMES2(KBYTE)];
        sprintf(msgbuf, "%s%sTotal size for db structures "
                "(max %.1f of residual): %zu; maxdbpos= %zu [maxndbstrs= %zu]%s"
                " (dev alloc= %zu; dev residual= %zu; totqrslen= %zu%s "
                "szdpdiag= %zu, szdpbottom= %zu, szdpalnposs= %zu, "
                "szdpmaxcoords= %zu, szdpbtckdat= %zu; szovlpos= %zu)", 
                preamb.c_str(), NL,
                maxperc, maxsizedbposs, maxdbposs, maxndbstrs, NL,
                deviceallocsize_, residualsize, totqrsposs, NL,
                szdpdiag, szdpbottom, szdpalnposs,
                szdpmaxcoords, szdpbtckdat, szovlpos);
        MYMSG(msgbuf, 3);
    MYMSGENDl

    SetCurrentMaxDbPos(maxdbposs);
    SetCurrentMaxNDbStrs(maxndbstrs);

    SetCurrentMaxDbPosPass2(maxdbposspass2);
    SetCurrentMaxNDbStrsPass2(maxdbstrspass2);

    //the following is ensured by enum:
    //sz_heapsections_[0][ddsBegOfQrsChunk] = sz_heapsections_[0][ddsEndOfConstantData];
    sz_heapsections_[0][ddsEndOfQrsChunk] = sz_heapsections_[0][ddsBegOfQrsChunk] + maxsizeqrsposs;

    //ensured by enum:
    //sz_heapsections_[0][ddsBegOfDbsChunk] = sz_heapsections_[0][ddsEndOfQrsChunk];
    sz_heapsections_[0][ddsEndOfDbsChunk] = sz_heapsections_[0][ddsBegOfDbsChunk] + maxsizedbposs;

    //ensured by enum:
    //sz_heapsections_[0][ddsBegOfQrsIndex] = sz_heapsections_[0][ddsEndOfDbsChunk];
    sz_heapsections_[0][ddsEndOfQrsIndex] = sz_heapsections_[0][ddsBegOfQrsIndex] + maxsizeqrsindex;

    //ensured by enum:
    //sz_heapsections_[0][ddsBegOfDbsIndex] = sz_heapsections_[0][ddsEndOfQrsIndex];
    sz_heapsections_[0][ddsEndOfDbsIndex] = sz_heapsections_[0][ddsBegOfDbsIndex] + maxsizedbsindex;

    sz_heapsections_[0][ddsBegOfMtxScores] = sz_heapsections_[0][ddsEndOfDbsIndex];
    sz_heapsections_[0][ddsEndOfMtxScores] = sz_heapsections_[0][ddsBegOfMtxScores] + szsmatrix;

    //DP-phase sections
    //ensured by enum:
    //sz_heapsections_[0][ddsBegOfDPDiagScores] = sz_heapsections_[0][ddsEndOfMtxScores];
    sz_heapsections_[0][ddsEndOfDPDiagScores] = sz_heapsections_[0][ddsBegOfDPDiagScores] + szdpdiag;

    //ensured by enum:
    //sz_heapsections_[0][ddsBegOfDPBottomScores] = sz_heapsections_[0][ddsEndOfDPDiagScores];
    sz_heapsections_[0][ddsEndOfDPBottomScores] = sz_heapsections_[0][ddsBegOfDPBottomScores] + szdpbottom;

    //ensured by enum:
    //sz_heapsections_[0][ddsBegOfDPAlignedPoss] = sz_heapsections_[0][ddsEndOfDPBottomScores];
    sz_heapsections_[0][ddsEndOfDPAlignedPoss] = sz_heapsections_[0][ddsBegOfDPAlignedPoss] + szdpalnposs;

    //ensured by enum:
    //sz_heapsections_[0][ddsBegOfDPMaxCoords] = sz_heapsections_[0][ddsEndOfDPAlignedPoss];
    sz_heapsections_[0][ddsEndOfDPMaxCoords] = sz_heapsections_[0][ddsBegOfDPMaxCoords] + szdpmaxcoords;

    //ensured by enum:
    //sz_heapsections_[0][ddsBegOfDPBackTckData] = sz_heapsections_[0][ddsEndOfDPMaxCoords];
    sz_heapsections_[0][ddsEndOfDPBackTckData] = sz_heapsections_[0][ddsBegOfDPBackTckData] + szdpbtckdat;

    //working memory section:
    //ensured by enum:
    //sz_heapsections_[0][ddsBegOfWrkMemory] = sz_heapsections_[0][ddsEndOfDPBackTckData];
    sz_heapsections_[0][ddsEndOfWrkMemory] = sz_heapsections_[0][ddsBegOfWrkMemory] + szwrkmem;

    //ensured by enum:
    //sz_heapsections_[0][ddsBegOfWrkMemoryCCD] = sz_heapsections_[0][ddsEndOfWrkMemory];
    sz_heapsections_[0][ddsEndOfWrkMemoryCCD] = sz_heapsections_[0][ddsBegOfWrkMemoryCCD] + szwrkmemccd;

    //ensured by enum:
    //sz_heapsections_[0][ddsBegOfWrkMemoryTMalt] = sz_heapsections_[0][ddsEndOfWrkMemoryCCD];
    sz_heapsections_[0][ddsEndOfWrkMemoryTMalt] = sz_heapsections_[0][ddsBegOfWrkMemoryTMalt] + szwrkmemtmalt;

    //ensured by enum:
    //sz_heapsections_[0][ddsBegOfWrkMemoryTM] = sz_heapsections_[0][ddsEndOfWrkMemoryTMalt];
    sz_heapsections_[0][ddsEndOfWrkMemoryTM] = sz_heapsections_[0][ddsBegOfWrkMemoryTM] + szwrkmemtm;

    //ensured by enum:
    //sz_heapsections_[0][ddsBegOfWrkMemoryTMibest] = sz_heapsections_[0][ddsEndOfWrkMemoryTM];
    sz_heapsections_[0][ddsEndOfWrkMemoryTMibest] = sz_heapsections_[0][ddsBegOfWrkMemoryTMibest] + szwrkmemtmibest;

    //ensured by enum:
    //sz_heapsections_[0][ddsBegOfAuxWrkMemory] = sz_heapsections_[0][ddsEndOfWrkMemoryTMibest];
    sz_heapsections_[0][ddsEndOfAuxWrkMemory] = sz_heapsections_[0][ddsBegOfAuxWrkMemory] + szauxwrkmem;

    //ensured by enum:
    //sz_heapsections_[0][ddsBegOfWrkMemory2] = sz_heapsections_[0][ddsEndOfAuxWrkMemory];
    sz_heapsections_[0][ddsEndOfWrkMemory2] = sz_heapsections_[0][ddsBegOfWrkMemory2] + szwrkmem2;

    //alignment statistics and alignment data sections:
    //ensured by enum:
    //sz_heapsections_[0][ddsBegOfDP2AlnData] = sz_heapsections_[0][ddsEndOfWrkMemory2];
    sz_heapsections_[0][ddsEndOfDP2AlnData] = sz_heapsections_[0][ddsBegOfDP2AlnData] + szdp2alndata;

    //ensured by enum:
    //sz_heapsections_[0][ddsBegOfTfmMatrices] = sz_heapsections_[0][ddsEndOfDP2AlnData];
    sz_heapsections_[0][ddsEndOfTfmMatrices] = sz_heapsections_[0][ddsBegOfTfmMatrices] + sztfmmtcs;

    //ensured by enum:
    //sz_heapsections_[0][ddsBegOfDP2Alns] = sz_heapsections_[0][ddsEndOfTfmMatrices];
    sz_heapsections_[0][ddsEndOfDP2Alns] = sz_heapsections_[0][ddsBegOfDP2Alns] + szdp2alns;

    //variables:
    //ensured by enum:
    //sz_heapsections_[0][ddsBegOfGlobVars] = sz_heapsections_[0][ddsEndOfDP2Alns];
    sz_heapsections_[0][ddsEndOfGlobVars] = sz_heapsections_[0][ddsBegOfGlobVars] + szglbvars;

    return maxsizedbposs;
}
