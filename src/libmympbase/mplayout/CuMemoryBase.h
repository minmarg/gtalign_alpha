/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __CuMemoryBase_h__
#define __CuMemoryBase_h__

#include "libutil/mybase.h"
#include "libutil/CLOptions.h"
#include "libgenp/gdats/PMBatchStrData.h"
#include "libgenp/gdats/PMBatchStrDataIndex.h"
#include "libgenp/gproc/gproc.h"
#include "libmymp/mpproc/mpprocconfbase.h"
#include "libmycu/custages/fragment.cuh"

////////////////////////////////////////////////////////////////////////////
// CLASS CuMemoryBase
// Basic memory arrangement for structure search and alignment computation
//
class CuMemoryBase
{
public:
    //relevant positions of sections
    enum TDevDataSections {
        /*E*/ddsEndOfPadding,//end of padding to align data
        /*E*/ddsEndOfConstantData = ddsEndOfPadding,//no constant data
        //
            /*b*/ddsBegOfQrsChunk = ddsEndOfConstantData,
        /*E*/ddsEndOfQrsChunk,//end of data chunk of query database(s) in the allocated block of memory
            /*b*/ddsBegOfDbsChunk = ddsEndOfQrsChunk,
        /*E*/ddsEndOfDbsChunk,//end of data chunk of target database(s) in the allocated block of memory
            /*b*/ddsBegOfQrsIndex = ddsEndOfDbsChunk,
        /*E*/ddsEndOfQrsIndex,//end of index of query database(s) in the allocated block of memory
            /*b*/ddsBegOfDbsIndex = ddsEndOfQrsIndex,
        /*E*/ddsEndOfDbsIndex,//end of index of target database(s) in the allocated block of memory
        //
            /*b*/ddsBegOfMtxScores,//beginning of the section of scores
        /*E*/ddsEndOfMtxScores,//end of the section of calculated (matrix) scores
        //
        // --- DP division ---
            /*b*/ddsBegOfDPDiagScores = ddsEndOfMtxScores,
        /*E*/ddsEndOfDPDiagScores,//end of the section of temporary diagonal scores for DP matrices
            /*b*/ddsBegOfDPBottomScores = ddsEndOfDPDiagScores,
        /*E*/ddsEndOfDPBottomScores,//end of the section of bottom-line scores for DP matrices
            /*b*/ddsBegOfDPAlignedPoss = ddsEndOfDPBottomScores,
        /*E*/ddsEndOfDPAlignedPoss,//end of the section of aligned positions and respective coordinates
            /*b*/ddsBegOfDPMaxCoords = ddsEndOfDPAlignedPoss,
        /*E*/ddsEndOfDPMaxCoords,//end of the section of the (mtx-)coordinates of maximum alignment scores
            /*b*/ddsBegOfDPBackTckData = ddsEndOfDPMaxCoords,
        /*E*/ddsEndOfDPBackTckData,//end of the section of DP backtracking data
        //
            /*b*/ddsBegOfWrkMemory = ddsEndOfDPBackTckData,
        /*E*/ddsEndOfWrkMemory,//end of the section of working memory [structure-specific]
            /*b*/ddsBegOfWrkMemoryCCD = ddsEndOfWrkMemory,
        /*E*/ddsEndOfWrkMemoryCCD,//end of the section of working memory [structure-specific]
            /*b*/ddsBegOfWrkMemoryTMalt = ddsEndOfWrkMemoryCCD,
        /*E*/ddsEndOfWrkMemoryTMalt,//end of the section of alternative TM configurations [structure-specific]
            /*b*/ddsBegOfWrkMemoryTM = ddsEndOfWrkMemoryTMalt,
        /*E*/ddsEndOfWrkMemoryTM,//end of the section of working memory [structure-specific]
            /*b*/ddsBegOfWrkMemoryTMibest = ddsEndOfWrkMemoryTM,
        /*E*/ddsEndOfWrkMemoryTMibest,//end of the section iteration-best TM [structure-specific]
            /*b*/ddsBegOfAuxWrkMemory = ddsEndOfWrkMemoryTMibest,
        /*E*/ddsEndOfAuxWrkMemory,//end of the section of auxiliary working memory [structure-specific]
            /*b*/ddsBegOfWrkMemory2 = ddsEndOfAuxWrkMemory,
        /*E*/ddsEndOfWrkMemory2,//end of the section of working memory [structure-specific]
        //
            /*b*/ddsBegOfDP2AlnData = ddsEndOfWrkMemory2,
        /*E*/ddsEndOfDP2AlnData,//end of the alignment data section [structure-specific, output for host]
            /*b*/ddsBegOfTfmMatrices = ddsEndOfDP2AlnData,
        /*E*/ddsEndOfTfmMatrices,//end of the section of calculated transformation matrices [str.-specific, output for host]
            /*b*/ddsBegOfDP2Alns = ddsEndOfTfmMatrices,
        /*E*/ddsEndOfDP2Alns,//end of the section of drawn alignments [structure-specific, output for host]
        //
        // --- variables ---
            /*b*/ddsBegOfGlobVars = ddsEndOfDP2Alns,
        /*E*/ddsEndOfGlobVars,//end of the section of global variables
        //
        nDevDataSections
    };


    CuMemoryBase(size_t deviceallocsize, int nareas);

    virtual ~CuMemoryBase();

    static size_t GetMaxQueriesPerChunk() {return (size_t)CLOptions::GetDEV_QRS_PER_CHUNK();}
    static size_t GetMaxAlnLength();
    static size_t GetMaxNFragSteps();

    size_t GetAllocSize() const {return deviceallocsize_;}
    int GetNAreas() const {return nareas_;}

    virtual char* GetHeap() const {return NULL;}

    size_t CalcMaxDbDataChunkSize(size_t totqrsposs);

    size_t GetCurrentMaxDbPos() const { return curmaxdbpos_; }
    size_t GetCurrentMaxNDbStrs() const { return curmaxndbstrs_; }

    size_t GetCurrentMaxDbPosPass2() const { return curmaxdbposspass2_; }
    size_t GetCurrentMaxNDbStrsPass2() const { return curmaxdbstrspass2_; }

    size_t GetHeapSectionOffset(int ano, int s) const;

    //minimum value for memory alignment
    static constexpr size_t GetMinMemAlignment() {
        size_t cszalnment = 256UL;
        return cszalnment;
    }

    virtual size_t GetMemAlignment() const {
        return GetMinMemAlignment();
    }

protected:

    virtual void AllocateHeap() {}
    virtual void DeallocateHeap() {}

    void Initialize();

    size_t CalcMaxDbDataChunkSizeHelper(size_t totqrsposs, size_t residualsize);

    void SetCurrentMaxDbPos(size_t value) {curmaxdbpos_ = value;}
    void SetCurrentMaxNDbStrs(size_t value) {curmaxndbstrs_ = value;}

    void SetCurrentMaxDbPosPass2(size_t value) {curmaxdbposspass2_ = value;}
    void SetCurrentMaxNDbStrsPass2(size_t value) {curmaxdbstrspass2_ = value;}


    size_t GetSizeOfTfmMatrices(size_t ndbstrs) const;
    size_t GetSizeOfPowerSpectrum(size_t maxposs) const;
    size_t GetSizeOfDPDiagScores(size_t maxdbposs) const;
    size_t GetSizeOfDPBottomScores(size_t maxdbposs) const;
    size_t GetSizeOfDPAlignedPoss(size_t maxdbposs) const;
    size_t GetSizeOfDPMaxCoords(size_t maxdbposs) const;
    size_t GetSizeOfDPBackTckData(size_t totqrsposs, size_t maxdbposs) const;
    size_t GetSizeOfMtxScores(size_t totqrsposs, size_t maxdbposs) const;
    size_t GetSizeOfWrkMemory(size_t ndbstrs) const;
    size_t GetSizeOfWrkMemoryCCD(size_t ndbstrs) const;
    size_t GetSizeOfWrkMemoryTMalt(size_t ndbstrs) const;
    size_t GetSizeOfWrkMemoryTM(size_t ndbstrs) const;
    size_t GetSizeOfWrkMemoryTMibest(size_t ndbstrs) const;
    size_t GetSizeOfAuxWrkMemory(size_t ndbstrs) const;
    size_t GetSizeOfWrkMemory2(size_t ndbstrs) const;
public:
    size_t GetSizeOfTfmMatrices(size_t nqystrs, size_t ndbstrs) const;
    size_t GetSizeOfDP2AlnData(size_t nqystrs, size_t ndbstrs) const;
    size_t GetSizeOfDP2Alns(
        size_t nqystrs, size_t totqrsposs, size_t ndbposs, size_t ndbstrs, bool sssinuse = true) const;
protected:
    size_t GetSizeOfDP2AlnData(size_t ndbstrs) const;
    size_t GetSizeOfDP2Alns(
        size_t totqrsposs, size_t ndbposs, size_t ndbstrs, bool sssinuse = true) const;
    size_t GetSizeOfGlobVariables(size_t ndbstrs) const;

    size_t GetMaxAllowedNumberDbPositions(size_t totqrsposs, bool sssinuse = true) const;

    void GetTotalMemoryReqs(
        size_t totqrsposs, size_t maxdbposs, size_t maxsizefordbstrs,
        //{{query db positions
        size_t* maxsizeqrsindex,
        size_t* maxsizeqrsposs,
        //}}{{target db positions
        size_t* maxsizedbsindex,
        size_t* maxsizedbposs,
        //}}{{transformations
        size_t* sztfmmtcs,
        //}}{{scores
        size_t* szsmatrix,
        //}}{{division
        size_t* szdpdiag, size_t* szdpbottom, size_t* szdpalnposs,
        size_t* szdpmaxcoords, size_t* szdpbtckdat,
        size_t* maxdbposspass2, size_t* maxdbstrspass2,
        size_t* szwrkmem, size_t* szwrkmemccd,
        size_t* szwrkmemtmalt, size_t* szwrkmemtm, size_t* szwrkmemtmibest,
        size_t* szauxwrkmem, size_t* szwrkmem2, 
        size_t* szdp2alndata, size_t* szdp2alns,
        //}}{{variables
        size_t* szglbvars,
        //}}{{overall size
        size_t* szovlpos
        //}}
    ) const;

    void MsgAddressTable(int areano, const std::string preamb, const int level) const;

private:
    size_t totqrslength_;//(total) length of queries to adjust memory configuration to
    size_t deviceallocsize_;//allocated size for device (max limit of memory used for device)
    int nareas_;//number of distinct computation areas within the limits of device memory
    size_t curmaxdbpos_;//current maximum number of db structure positions that can be used for calculations
    size_t curmaxndbstrs_;//current maximum number of db structures that can be processed
    size_t curmaxdbposspass2_;//maximum number of positions of significant db structures
    size_t curmaxdbstrspass2_;//maximum number of significant db structures
protected:
    //{{device data
    //device pointers to the beginnings of sections in the heap for each area:
    size_t (*sz_heapsections_)[nDevDataSections];
    //}}
};

// -------------------------------------------------------------------------
// INLINES ...
//
inline
size_t CuMemoryBase::GetHeapSectionOffset(int ano, int s) const
{
#ifdef __DEBUG__
    if( ano < 0 || ano >= nareas_ || s < 0 || s >= nDevDataSections)
        throw MYRUNTIME_ERROR(
        "CuMemoryBase::GetHeapSectionOffset: Memory access error.");
#endif
    return sz_heapsections_[ano][s];
}

// -------------------------------------------------------------------------
// MsgAddressTable: print as message the address table of the sections;
// ano, area number
inline
void CuMemoryBase::MsgAddressTable(int ano, const std::string preamb, const int level) const
{
    MYMSGBEGl(level)
        char msgbuf[BUF_MAX];
        sprintf(msgbuf, "Address table: %p [area no. %d]", GetHeap(), ano );
        MYMSGnonl((preamb + msgbuf).c_str(),level);
        sprintf(msgbuf, "%7c +%zu ddsEndOfPadding",' ',sz_heapsections_[ano][ddsEndOfPadding]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        sprintf(msgbuf, "%7c +%zu ddsEndOfQrsChunk (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfQrsChunk],
            sz_heapsections_[ano][ddsEndOfQrsChunk]-sz_heapsections_[ano][ddsBegOfQrsChunk]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        sprintf(msgbuf, "%7c +%zu ddsEndOfDbsChunk (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfDbsChunk],
            sz_heapsections_[ano][ddsEndOfDbsChunk]-sz_heapsections_[ano][ddsBegOfDbsChunk]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        sprintf(msgbuf, "%7c +%zu ddsEndOfQrsIndex (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfQrsIndex],
            sz_heapsections_[ano][ddsEndOfQrsIndex]-sz_heapsections_[ano][ddsBegOfQrsIndex]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        sprintf(msgbuf, "%7c +%zu ddsEndOfDbsIndex (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfDbsIndex],
            sz_heapsections_[ano][ddsEndOfDbsIndex]-sz_heapsections_[ano][ddsBegOfDbsIndex]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        sprintf(msgbuf, "%7c +%zu ddsBegOfMtxScores",' ',sz_heapsections_[ano][ddsBegOfMtxScores]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);
        sprintf(msgbuf, "%7c +%zu ddsEndOfMtxScores (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfMtxScores],
            sz_heapsections_[ano][ddsEndOfMtxScores]-sz_heapsections_[ano][ddsBegOfMtxScores]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        MYMSGnonl((preamb + "------- DP division").c_str(),level);
        sprintf(msgbuf, "%7c +%zu ddsEndOfDPDiagScores (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfDPDiagScores],
            sz_heapsections_[ano][ddsEndOfDPDiagScores]-sz_heapsections_[ano][ddsBegOfDPDiagScores]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        sprintf(msgbuf, "%7c +%zu ddsEndOfDPBottomScores (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfDPBottomScores],
            sz_heapsections_[ano][ddsEndOfDPBottomScores]-sz_heapsections_[ano][ddsBegOfDPBottomScores]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        sprintf(msgbuf, "%7c +%zu ddsEndOfDPAlignedPoss (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfDPAlignedPoss],
            sz_heapsections_[ano][ddsEndOfDPAlignedPoss]-sz_heapsections_[ano][ddsBegOfDPAlignedPoss]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        sprintf(msgbuf, "%7c +%zu ddsEndOfDPMaxCoords (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfDPMaxCoords],
            sz_heapsections_[ano][ddsEndOfDPMaxCoords]-sz_heapsections_[ano][ddsBegOfDPMaxCoords]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        sprintf(msgbuf, "%7c +%zu ddsEndOfDPBackTckData (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfDPBackTckData],
            sz_heapsections_[ano][ddsEndOfDPBackTckData]-sz_heapsections_[ano][ddsBegOfDPBackTckData]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        MYMSGnonl((preamb + "------- Working memory division").c_str(),level);
        sprintf(msgbuf, "%7c +%zu ddsEndOfWrkMemory (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfWrkMemory],
            sz_heapsections_[ano][ddsEndOfWrkMemory]-sz_heapsections_[ano][ddsBegOfWrkMemory]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        sprintf(msgbuf, "%7c +%zu ddsEndOfWrkMemoryCCD (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfWrkMemoryCCD],
            sz_heapsections_[ano][ddsEndOfWrkMemoryCCD]-sz_heapsections_[ano][ddsBegOfWrkMemoryCCD]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        sprintf(msgbuf, "%7c +%zu ddsEndOfWrkMemoryTMalt (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfWrkMemoryTMalt],
            sz_heapsections_[ano][ddsEndOfWrkMemoryTMalt]-sz_heapsections_[ano][ddsBegOfWrkMemoryTMalt]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        sprintf(msgbuf, "%7c +%zu ddsEndOfWrkMemoryTM (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfWrkMemoryTM],
            sz_heapsections_[ano][ddsEndOfWrkMemoryTM]-sz_heapsections_[ano][ddsBegOfWrkMemoryTM]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        sprintf(msgbuf, "%7c +%zu ddsEndOfWrkMemoryTMibest (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfWrkMemoryTMibest],
            sz_heapsections_[ano][ddsEndOfWrkMemoryTMibest]-sz_heapsections_[ano][ddsBegOfWrkMemoryTMibest]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        sprintf(msgbuf, "%7c +%zu ddsEndOfAuxWrkMemory (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfAuxWrkMemory],
            sz_heapsections_[ano][ddsEndOfAuxWrkMemory]-sz_heapsections_[ano][ddsBegOfAuxWrkMemory]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        sprintf(msgbuf, "%7c +%zu ddsEndOfWrkMemory2 (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfWrkMemory2],
            sz_heapsections_[ano][ddsEndOfWrkMemory2]-sz_heapsections_[ano][ddsBegOfWrkMemory2]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        MYMSGnonl((preamb + "------- Alignment data division").c_str(),level);
        sprintf(msgbuf, "%7c +%zu ddsEndOfDP2AlnData (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfDP2AlnData],
            sz_heapsections_[ano][ddsEndOfDP2AlnData]-sz_heapsections_[ano][ddsBegOfDP2AlnData]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        sprintf(msgbuf, "%7c +%zu ddsEndOfTfmMatrices (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfTfmMatrices],
            sz_heapsections_[ano][ddsEndOfTfmMatrices]-sz_heapsections_[ano][ddsBegOfTfmMatrices]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        sprintf(msgbuf, "%7c +%zu ddsEndOfDP2Alns (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfDP2Alns],
            sz_heapsections_[ano][ddsEndOfDP2Alns]-sz_heapsections_[ano][ddsBegOfDP2Alns]);
        MYMSGnonl((preamb + msgbuf).c_str(),level);

        MYMSGnonl((preamb + "------- Globals").c_str(),level);
        sprintf(msgbuf, "%7c +%zu ddsEndOfGlobVars (sz %zu)",' ',
            sz_heapsections_[ano][ddsEndOfGlobVars],
            sz_heapsections_[ano][ddsEndOfGlobVars]-sz_heapsections_[ano][ddsBegOfGlobVars]);
        MYMSG((preamb + msgbuf).c_str(),level);
    MYMSGENDl

    if(deviceallocsize_ < sz_heapsections_[ano][nDevDataSections-1])
        throw MYRUNTIME_ERROR(preamb + "Out of range of allocated memory.");
}



// -------------------------------------------------------------------------
// GetMaxAlnLength: get the max alignment length
inline
size_t CuMemoryBase::GetMaxAlnLength()
{
    MYMSG("CuMemoryBase::GetMaxAlnLength", 8);
    //max total length of queries, which is used here as max query length
    static const size_t maxqres = CLOptions::GetDEV_QRES_PER_CHUNK();
    //max reference length
    static const size_t maxrlen = CLOptions::GetDEV_MAXRLEN();
    static const size_t maxalnmax = mymin(maxqres, maxrlen);
    return maxalnmax;
}

// -------------------------------------------------------------------------
// GetMaxNFragSteps: get max number of steps to calculate superposition for 
// an alignment of maximum length given the smallest fragment length and 
// step size
inline
size_t CuMemoryBase::GetMaxNFragSteps()
{
    MYMSG("CuMemoryBase::GetMaxNFragSteps", 8);
    static const size_t maxalnmax = GetMaxAlnLength();
    constexpr size_t sfragstep = FRAGREF_SFRAGSTEP;
    int minfraglen = GetMinFragLengthForAln(maxalnmax);
    if(minfraglen < 1)
        throw MYRUNTIME_ERROR("Invalid minimum fragment length obtained: " +
            std::to_string(minfraglen));
    //max number of steps for an alignment of size maxalnmax using a 
    //fragment of length minfraglen and a step of sfragstep; calculated so 
    //that the following holds:
    //sfragstep * nmaxsteps + minfraglen < maxalnmax + sfragstep
    //step starts with 0, hence:
    size_t maxnsteps = ::GetMaxNFragSteps(maxalnmax, sfragstep, minfraglen);
    maxnsteps *= FRAGREF_NMAXSUBFRAGS;//total number across all fragment lengths
    //consider max #fragment factors in the final stage of alignment refinement:
    size_t maxnfragfcts = mymin((size_t)CUSFN_TBSP_FIN_REFINEMENT_MAX_NPOSITIONS, maxalnmax);
    maxnfragfcts *= FRAGREF_NMAXSUBFRAGS;//total number across all fragment lengths
    maxnsteps = mymax(maxnfragfcts, maxnsteps);
#if !defined(GPUINUSE)
    static const size_t nthreads = CLOptions::GetCPU_THREADS();
    //nthreads * 3 + 6 is required for memory region of working tfms
    maxnsteps = mymax(nthreads * 3 + 6, maxnsteps);
#endif
    //set at least >=1024 for the maximum number of additional parallelization factor
    maxnsteps = mymax((size_t)1024, maxnsteps);
    return maxnsteps;
}



// -------------------------------------------------------------------------
// GetSizeOfTfmMatrices: get the size of transformation matrices over all 
// query-target structure pairs
inline
size_t CuMemoryBase::GetSizeOfTfmMatrices(size_t nqystrs, size_t ndbstrs) const
{
    MYMSG("CuMemoryBase::GetSizeOfTfmMatrices", 8);
    const size_t cszalnment = GetMemAlignment();
    size_t sztfmmtcs = 
            //calculated for each query-target structure pair:
            nqystrs *
            (ndbstrs * nTTranformMatrix) * 
            sizeof(float);
    sztfmmtcs = ALIGN_UP(sztfmmtcs, cszalnment);
    return sztfmmtcs;
}

// -------------------------------------------------------------------------
// GetSizeOfTfmMatrices: get the size of transformation matrices
inline
size_t CuMemoryBase::GetSizeOfTfmMatrices(size_t ndbstrs) const
{
    // MYMSG("CuMemoryBase::GetSizeOfTfmMatrices", 9);
    const size_t maxnqrsperchunk = GetMaxQueriesPerChunk();
    return GetSizeOfTfmMatrices(maxnqrsperchunk, ndbstrs);
}

// -------------------------------------------------------------------------
// GetSizeOfPowerSpectrum: get the size of the buffers required to store the 
// whole spectrum at each query or reference position;
// maxposs, max number of positions over all queries/reference 
// structures in the chunk;
//
inline
size_t CuMemoryBase::GetSizeOfPowerSpectrum(size_t maxposs) const
{
    MYMSG("CuMemoryBase::GetSizeOfPowerSpectrum", 8);
    const size_t cszalnment = GetMemAlignment();
    const size_t padding = 
        ((CUL2CLINESIZE < CUDP_2DCACHE_DIM_D_BASE)? CUDP_2DCACHE_DIM_D_BASE: CUL2CLINESIZE) * 2;
    size_t ndatpnts = CUSA2_SPECAN_MAX_NDSTS * pmv2DNoElems + 1;
    size_t ndatpnts3 = C3MAXNCMPL * CUSA3_SPECAN_MAX_NSSSTATES;
    size_t ndatpnts32 = C3MAXNCMPL32 * CUSA32_SPECAN_MAX_NSSSTATES;
    size_t ndatpnts7 = nTSASEQSEPDIM * CUSA7_SPECAN_MAX_NSSSTATES;
    ndatpnts = mymax(ndatpnts, (size_t)SPECAN_TOT_NSMPL_POINTS);
    ndatpnts = mymax(ndatpnts, ndatpnts3);
    ndatpnts = mymax(ndatpnts, ndatpnts32);
    ndatpnts = mymax(ndatpnts, ndatpnts7);
    size_t szpspec = 
            (maxposs + padding) *
            ( ndatpnts
            ) * sizeof(float);
    szpspec = ALIGN_UP(szpspec, cszalnment);
    return szpspec;
}

// -------------------------------------------------------------------------
// GetSizeOfDPDiagScores: get the size of the buffers of diagonal scores 
// used for dynamic programming
//
inline
size_t CuMemoryBase::GetSizeOfDPDiagScores(size_t maxdbposs) const
{
    MYMSG("CuMemoryBase::GetSizeOfDPDiagScores", 8);
    const size_t cszalnment = GetMemAlignment();
    const size_t maxnqrsperchunk = GetMaxQueriesPerChunk();
    const size_t maxnsteps = GetMaxNFragSteps();
    const size_t padding = 
        ((CUL2CLINESIZE < CUDP_2DCACHE_DIM_D_BASE)? CUDP_2DCACHE_DIM_D_BASE: CUL2CLINESIZE) * 2;
    //tmpdpdiagbuffers will also be used as a tmp buffer for copying selected candidates;
    //make sure it'll contain enough space (maxnsteps >> #copied_data_sections by definition);
    //NOTE: ndbstrs is calculated so that ndbstrs = maxdbposs / expected_length (>=20);
    //NOTE: hence, maxdbposs > ndbstrs * nTAuxWorkingMemoryVars (nTTranformMatrix)!
    //NOTE: two maxes below are not necessary!
    // // maxdbposs = mymax(maxdbposs, ndbstrs * nTAuxWorkingMemoryVars);
    // // maxdbposs = mymax(maxdbposs, ndbstrs * nTTranformMatrix);
    //NOTE: nTDPAGCDiagScoreSubsections when spectral score and affine gap costs are in use
    constexpr size_t nsubsections = nTDPDiagScoreSubsections;//nTDPAGCDiagScoreSubsections
    //NOTE: two diagonal buffers with all appropriate states are required for 
    // DP execution and one for tracking max score;
    //NOTE: the worst case scenario is all structures of unit length (or length 3); 
    //NOTE: add diagonal width to both ends of the db positions;
    //NOTE: use this size of memory for buffers (2nd line): 
    // a series of block diagonals are launched on the device for each query;
    size_t szdpdiag = 
            //diagonal buffers are required for each query structure:
            maxnqrsperchunk *
            maxnsteps *
            (maxdbposs + padding) *
            ( nTDPDiagScoreSections * nsubsections
            ) * sizeof(float);
    szdpdiag = ALIGN_UP(szdpdiag, cszalnment);
    return szdpdiag;
}

// -------------------------------------------------------------------------
// GetSizeOfDPBottomScores: get the size of the buffer of bottom scores 
// used for dynamic programming
//
inline
size_t CuMemoryBase::GetSizeOfDPBottomScores(size_t maxdbposs) const
{
    MYMSG("CuMemoryBase::GetSizeOfDPBottomScores", 8);
    const size_t cszalnment = GetMemAlignment();
    const size_t maxnqrsperchunk = GetMaxQueriesPerChunk();
    const size_t maxnsteps = GetMaxNFragSteps();
    //NOTE: should be carefully revised when using spectral scores and alignment:
    const_cast<size_t&>(maxnsteps) = mymin((size_t)200,maxnsteps);
    const size_t padding = 
        ((CUL2CLINESIZE < CUDP_2DCACHE_DIM_D_BASE)? CUDP_2DCACHE_DIM_D_BASE: CUL2CLINESIZE) * 2;
    //NOTE: nTDPAGCDiagScoreSubsections when spectral score and affine gap costs are in use
    constexpr size_t nsubsections = nTDPDiagScoreSubsections;//nTDPAGCDiagScoreSubsections
    size_t szdpbottom = 
            //buffers are required for each query structure:
            maxnqrsperchunk *
            maxnsteps *
            (maxdbposs + padding) * 
            ( nsubsections/*dpbssBottm*/ 
            ) * sizeof(float);
    szdpbottom = ALIGN_UP(szdpbottom, cszalnment);
    return szdpbottom;
}

// -------------------------------------------------------------------------
// GetSizeOfDPAlignedPoss: get the size of the buffer of positions, along 
// with respective coordinates, aligned using dynamic programming
//
inline
size_t CuMemoryBase::GetSizeOfDPAlignedPoss(size_t maxdbposs) const
{
    MYMSG("CuMemoryBase::GetSizeOfDPAlignedPoss", 8);
    const size_t cszalnment = GetMemAlignment();
    const size_t maxnqrsperchunk = GetMaxQueriesPerChunk();
    const size_t maxnsteps = GetMaxNFragSteps();
    const size_t padding = 
        ((CUL2CLINESIZE < CUDP_2DCACHE_DIM_D_BASE)? CUDP_2DCACHE_DIM_D_BASE: CUL2CLINESIZE) * 2;
    size_t szdpalnposs = 
            //buffers are required for each query structure:
            maxnqrsperchunk *
            maxnsteps *
            (maxdbposs + padding) * 
            ( nTDPAlignedPoss
            ) * sizeof(float);
    szdpalnposs = ALIGN_UP(szdpalnposs, cszalnment);
    return szdpalnposs;
}

// -------------------------------------------------------------------------
// GetSizeOfDPMaxCoords: get the size of the buffer of the coordinates of 
// maximum alignment scores calculated by dynamic programming
//
inline
size_t CuMemoryBase::GetSizeOfDPMaxCoords(size_t maxdbposs) const
{
    MYMSG("CuMemoryBase::GetSizeOfDPMaxCoords", 8);
    const size_t cszalnment = GetMemAlignment();
    const size_t maxnqrsperchunk = GetMaxQueriesPerChunk();
    const size_t padding = 
        ((CUL2CLINESIZE < CUDP_2DCACHE_DIM_D_BASE)? CUDP_2DCACHE_DIM_D_BASE: CUL2CLINESIZE) * 2;
    size_t szdpdiagcoords = 
            //buffers are required for each query structure:
            maxnqrsperchunk *
            (maxdbposs + padding) * 
            sizeof(uint);
    szdpdiagcoords = ALIGN_UP(szdpdiagcoords, cszalnment);
    return szdpdiagcoords;
}

// -------------------------------------------------------------------------
// GetSizeOfDPBackTckData: get the size of the buffer of backtracking 
// information obtained from dynamic programming
inline
size_t CuMemoryBase::GetSizeOfDPBackTckData(size_t totqrsposs, size_t maxdbposs) const
{
    MYMSG("CuMemoryBase::GetSizeOfDPBackTckData", 8);
    const size_t cszalnment = GetMemAlignment();
    const size_t padding = 
        ((CUL2CLINESIZE < CUDP_2DCACHE_DIM_D_BASE)? CUDP_2DCACHE_DIM_D_BASE: CUL2CLINESIZE) * 2;
    size_t szdpbtckdat = 
            ((maxdbposs/**/ + padding/**/) * totqrsposs) * 
            sizeof(char);
    szdpbtckdat = ALIGN_UP(szdpbtckdat, cszalnment);
    return szdpbtckdat;
}

// -------------------------------------------------------------------------
// GetSizeOfMtxScores: get the size of spectrum scores in matrix
inline
size_t CuMemoryBase::GetSizeOfMtxScores(size_t totqrsposs, size_t maxdbposs) const
{
    MYMSG("CuMemoryBase::GetSizeOfMtxScores", 8);
    const size_t cszalnment = GetMemAlignment();
    const size_t padding = 
        ((CUL2CLINESIZE < CUDP_2DCACHE_DIM_D_BASE)? CUDP_2DCACHE_DIM_D_BASE: CUL2CLINESIZE) * 2;
    size_t szmtxscores = 
            ((maxdbposs/**/ + padding/**/) * totqrsposs) * 
            sizeof(float);
    szmtxscores = ALIGN_UP(szmtxscores, cszalnment);
    return szmtxscores;
}


// -------------------------------------------------------------------------
// GetSizeOfWrkMemory: get the size of the section of working memory
inline
size_t CuMemoryBase::GetSizeOfWrkMemory(size_t ndbstrs) const
{
    MYMSG("CuMemoryBase::GetSizeOfWrkMemory", 8);
    const size_t cszalnment = GetMemAlignment();
    const size_t maxnqrsperchunk = GetMaxQueriesPerChunk();
    const size_t maxnsteps = GetMaxNFragSteps();
    size_t szwrkmem = 
            //calculated for each query-target structure pair:
            maxnqrsperchunk *
            (ndbstrs * maxnsteps * nTWorkingMemoryVars) *
            sizeof(float);
    szwrkmem = ALIGN_UP(szwrkmem, cszalnment);
    return szwrkmem;
}

// -------------------------------------------------------------------------
// GetSizeOfWrkMemoryCCD: get the size of the section of working memory 
// assigned to additional cross-covariance data
inline
size_t CuMemoryBase::GetSizeOfWrkMemoryCCD(size_t ndbstrs) const
{
    MYMSG("CuMemoryBase::GetSizeOfWrkMemoryCCD", 8);
    const size_t cszalnment = GetMemAlignment();
    const size_t maxnqrsperchunk = GetMaxQueriesPerChunk();
    const size_t maxnsteps = GetMaxNFragSteps();
    size_t szwrkmemccd = 
            //calculated for each query-target structure pair:
            maxnqrsperchunk *
            (ndbstrs * maxnsteps * nTWorkingMemoryVars) *
            sizeof(float);
    szwrkmemccd = ALIGN_UP(szwrkmemccd, cszalnment);
    return szwrkmemccd;
}

// -------------------------------------------------------------------------
// GetSizeOfWrkMemoryTMalt: get the size of the section of working memory 
// assigned to a small number of alternative transformation matrices to be 
// processed by refinement initialized with them 
inline
size_t CuMemoryBase::GetSizeOfWrkMemoryTMalt(size_t ndbstrs) const
{
    MYMSG("CuMemoryBase::GetSizeOfWrkMemoryTMalt", 8);
    const size_t cszalnment = GetMemAlignment();
    const size_t maxnqrsperchunk = GetMaxQueriesPerChunk();
    //max #alternative consfigurations:
    const size_t maxnalts = CUS1_TBSP_DPSCORE_TOP_N_REFINEMENTxMAX_CONFIGS;
    size_t szwrkmemtmalt = 
            //calculated for each query-target structure pair:
            maxnqrsperchunk *
            (ndbstrs * maxnalts * nTTranformMatrix) *
            sizeof(float);
    szwrkmemtmalt = ALIGN_UP(szwrkmemtmalt, cszalnment);
    return szwrkmemtmalt;
}

// -------------------------------------------------------------------------
// GetSizeOfWrkMemoryTM: get the size of the section of working memory 
// assigned to additional transformation matrix data
inline
size_t CuMemoryBase::GetSizeOfWrkMemoryTM(size_t ndbstrs) const
{
    MYMSG("CuMemoryBase::GetSizeOfWrkMemoryTM", 8);
    const size_t cszalnment = GetMemAlignment();
    const size_t maxnqrsperchunk = GetMaxQueriesPerChunk();
    const size_t maxnsteps = GetMaxNFragSteps();
    size_t szwrkmemtm = 
            //calculated for each query-target structure pair:
            maxnqrsperchunk *
            (ndbstrs * maxnsteps * nTTranformMatrix) *
            sizeof(float);
    szwrkmemtm = ALIGN_UP(szwrkmemtm, cszalnment);
    return szwrkmemtm;
}

// -------------------------------------------------------------------------
// GetSizeOfWrkMemoryTMibest: get the size of the section of working 
// memory for iteration-best transformation matrix data
inline
size_t CuMemoryBase::GetSizeOfWrkMemoryTMibest(size_t ndbstrs) const
{
    MYMSG("CuMemoryBase::GetSizeOfWrkMemoryTMibest", 8);
    const size_t cszalnment = GetMemAlignment();
    const size_t maxnqrsperchunk = GetMaxQueriesPerChunk();
    const size_t maxnsteps = GetMaxNFragSteps();
    size_t szwrkmemtmibest = 
            //calculated for each query-target structure pair:
            maxnqrsperchunk *
            (ndbstrs * maxnsteps * nTTranformMatrix) *
            sizeof(float);
    szwrkmemtmibest = ALIGN_UP(szwrkmemtmibest, cszalnment);
    return szwrkmemtmibest;
}

// -------------------------------------------------------------------------
// GetSizeOfAuxWrkMemory: get the size of the section of auxiliary working 
// memory
inline
size_t CuMemoryBase::GetSizeOfAuxWrkMemory(size_t ndbstrs) const
{
    MYMSG("CuMemoryBase::GetSizeOfAuxWrkMemory", 8);
    const size_t cszalnment = GetMemAlignment();
    const size_t maxnqrsperchunk = GetMaxQueriesPerChunk();
    const size_t maxnsteps = GetMaxNFragSteps();
    size_t szauxwrkmem = 
            //calculated for each query-target structure pair:
            maxnqrsperchunk *
            (ndbstrs * maxnsteps * nTAuxWorkingMemoryVars) *
            sizeof(float);
    szauxwrkmem = ALIGN_UP(szauxwrkmem, cszalnment);
    return szauxwrkmem;
}

// -------------------------------------------------------------------------
// GetSizeOfWrkMemory2: get the size of the second section of working memory
inline
size_t CuMemoryBase::GetSizeOfWrkMemory2(size_t ndbstrs) const
{
    MYMSG("CuMemoryBase::GetSizeOfWrkMemory2", 8);
    const size_t cszalnment = GetMemAlignment();
    const size_t maxnqrsperchunk = GetMaxQueriesPerChunk();
    const size_t maxnsteps = GetMaxNFragSteps();
    size_t szwrkmem2 = 
            //calculated for each query-target structure pair:
            maxnqrsperchunk *
            (ndbstrs * maxnsteps * nTWorkingMemory2Vars) *
            sizeof(float);
    szwrkmem2 = ALIGN_UP(szwrkmem2, cszalnment);
    return szwrkmem2;
}


// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// GetSizeOfDP2AlnData: get the size of statistics of obtained alignments
inline
size_t CuMemoryBase::GetSizeOfDP2AlnData(size_t nqystrs, size_t ndbstrs) const
{
    MYMSG("CuMemoryBase::GetSizeOfDP2AlnData", 8);
    const size_t cszalnment = GetMemAlignment();
    size_t szdp2alndat = 
            //calculated for each query-target structure pair:
            nqystrs *
            (ndbstrs * nTDP2OutputAlnData) * 
            sizeof(float);
    szdp2alndat = ALIGN_UP(szdp2alndat, cszalnment);
    return szdp2alndat;
}

// -------------------------------------------------------------------------
// GetSizeOfDP2AlnData: get the size of statistics of obtained alignments
inline
size_t CuMemoryBase::GetSizeOfDP2AlnData(size_t ndbstrs) const
{
    // MYMSG("CuMemoryBase::GetSizeOfDP2AlnData", 9);
    const size_t maxnqrsperchunk = GetMaxQueriesPerChunk();
    return GetSizeOfDP2AlnData(maxnqrsperchunk, ndbstrs);
}

// -------------------------------------------------------------------------
// GetSizeOfDP2Alns: get the size of the buffer for alignments themselves 
inline
size_t CuMemoryBase::GetSizeOfDP2Alns(
    size_t nqystrs, size_t totqrsposs, size_t ndbposs, size_t ndbstrs, bool sssinuse) const
{
    MYMSG("CuMemoryBase::GetSizeOfDP2Alns", 8);
    const size_t cszalnment = GetMemAlignment();
    //maximum alignment length: l_query + l_target
    size_t szdp2alns = 
            //configuration for multiple query-target pairs over all 
            //queries and targets in the chunk:
            ((ndbposs * nqystrs + 
                ndbstrs * (totqrsposs + nqystrs)) *
             (sssinuse? nTDP2OutputAlignmentSSS: nTDP2OutputAlignment)
            ) * sizeof(char);
    szdp2alns = ALIGN_UP(szdp2alns, cszalnment);
    return szdp2alns;
}

// -------------------------------------------------------------------------
// GetSizeOfDP2Alns: get the size of the buffer for alignments themselves 
inline
size_t CuMemoryBase::GetSizeOfDP2Alns(
    size_t totqrsposs, size_t ndbposs, size_t ndbstrs, bool sssinuse) const
{
    // MYMSG("CuMemoryBase::GetSizeOfDP2Alns", 9);
    const size_t maxnqrsperchunk = GetMaxQueriesPerChunk();
    return GetSizeOfDP2Alns(maxnqrsperchunk, totqrsposs, ndbposs, ndbstrs, sssinuse);
}


// -------------------------------------------------------------------------
// GetSizeOfGlobVariables: get the size of the global variables section
inline
size_t CuMemoryBase::GetSizeOfGlobVariables(size_t ndbstrs) const
{
    MYMSG( "CuMemoryBase::GetSizeOfGlobVariables", 8 );
    const size_t cszalnment = GetMemAlignment();
    size_t szglbvars = 
        ndbstrs * nTFilterData * sizeof(int);
    szglbvars = ALIGN_UP(szglbvars, cszalnment);
    return szglbvars;
}



// =========================================================================
// GetMaxAllowedNumberDbPositions: get the maximum allowed number of Db 
// positions given total length of queries to avoid overflow
inline
size_t CuMemoryBase::GetMaxAllowedNumberDbPositions( 
    size_t totqrsposs, bool sssinuse ) const
{
    MYMSG("CuMemoryBase::GetMaxAllowedNumberDbPositions", 7);

    const size_t maxnqrsperchunk = GetMaxQueriesPerChunk();
    const size_t maxnsteps = GetMaxNFragSteps();
    const size_t padding = 
        ((CUL2CLINESIZE < CUDP_2DCACHE_DIM_D_BASE)? CUDP_2DCACHE_DIM_D_BASE: CUL2CLINESIZE) * 2;

    //dependent maximum length derived from backtracking/score matrix size requirements:
    size_t deplen1 = 
        (size_t)INT_MAX / totqrsposs - padding;

    //maximum length dependent on and derived from the number of alignments:
    // (ndbstr*expct_dbprolen=ndbposs)
    //NOTE: multiple queries do have effect on #db positions
    size_t deplen2 = (size_t)(
        (float)INT_MAX / (sssinuse? nTDP2OutputAlignmentSSS: nTDP2OutputAlignment) /
        ((float)/*1.0f*/maxnqrsperchunk +
            (float)(totqrsposs + maxnqrsperchunk + 1/*rounding*/) /
            (float)CLOptions::GetDEV_EXPCT_DBPROLEN()
        )
    );

    //independent maximum length derived from working memory requirements:
    //(evidently large)
    size_t indeplen0 = 
       (size_t)INT_MAX * CLOptions::GetDEV_EXPCT_DBPROLEN() /
           (/*1*/maxnqrsperchunk * maxnsteps * nTWorkingMemoryVars);

    //independent maximum length derived from alignment statistics size requirements:
    //(evidently large)
    //size_t indeplen1 = 
    //    (size_t)INT_MAX * CLOptions::GetDEV_EXPCT_DBPROLEN() /
    //        (/*1*/maxnqrsperchunk * nTDP2OutputAlnData);

    //independent maximum length derived from the number of transformation matrices:
    //(evidently large)
    //size_t indeplen2 = 
    //   (size_t)INT_MAX * CLOptions::GetDEV_EXPCT_DBPROLEN() /
    //       (/*1*/maxnqrsperchunk * maxnsteps * nTTranformMatrix);

    //NOTE: nTDPAGCDiagScoreSubsections when spectral score and affine gap costs are in use
    constexpr size_t nsubsections = nTDPDiagScoreSubsections;//nTDPAGCDiagScoreSubsections
    //independent maximum length derived from #DP diagonal scores:
    size_t indeplen3 = 
        (size_t)INT_MAX /
            (maxnqrsperchunk * maxnsteps *
                (nTDPDiagScoreSections * nsubsections)
            )
        - padding;

    //independent maximum length derived from #DP-aligned positions and coordinates:
    size_t indeplen4 = 
        (size_t)INT_MAX / (maxnqrsperchunk * maxnsteps * (nTDPAlignedPoss))
        - padding;

    //independent maximum length derived from #spectrum scores:
    size_t ndatpnts = CUSA2_SPECAN_MAX_NDSTS * pmv2DNoElems + 1;
    size_t ndatpnts3 = C3MAXNCMPL * CUSA3_SPECAN_MAX_NSSSTATES;
    size_t ndatpnts32 = C3MAXNCMPL32 * CUSA32_SPECAN_MAX_NSSSTATES;
    size_t ndatpnts7 = nTSASEQSEPDIM * CUSA7_SPECAN_MAX_NSSSTATES;
    ndatpnts = mymax(ndatpnts, (size_t)SPECAN_TOT_NSMPL_POINTS);
    ndatpnts = mymax(ndatpnts, ndatpnts3);
    ndatpnts = mymax(ndatpnts, ndatpnts32);
    ndatpnts = mymax(ndatpnts, ndatpnts7);
    size_t indeplen5 =
        (size_t)INT_MAX / (ndatpnts)
        - padding;

    //take minimum:
    size_t maxallwdposs = PCMIN(deplen1, deplen2);
    maxallwdposs = PCMIN(maxallwdposs, indeplen0);
    //maxallwdposs = PCMIN(maxallwdposs, indeplen1);
    //maxallwdposs = PCMIN(maxallwdposs, indeplen2);
    maxallwdposs = PCMIN(maxallwdposs, indeplen3);
    maxallwdposs = PCMIN(maxallwdposs, indeplen4);
    maxallwdposs = PCMIN(maxallwdposs, indeplen5);
    return maxallwdposs;
}



// overall memory requirements ---------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// GetTotalMemoryReqs: get total memeory requirements given the total 
// number of query and db structure positions;
// maxdbposspass2 and maxdbstrspass2 represent quantities of target Db 
// structures that pass the significance threshold;
// NOTE: addresses are assumed to be valid and they are not verified
inline
void CuMemoryBase::GetTotalMemoryReqs( 
    size_t totqrsposs, size_t maxdbposs, size_t /*maxsizefordbstrs*/,
    //{{query db positions
    size_t* maxsizeqrsindex,
    size_t* maxsizeqrsposs,
    //}}{{target db positions
    size_t* maxsizedbsindex,
    size_t* maxsizedbposs,
    //}}{{transformations
    size_t* sztfmmtcs,
    //}}{{scores
    size_t* szsmatrix,
    //}}{{DP division
    size_t* szdpdiag, size_t* szdpbottom, size_t* szdpalnposs,
    size_t* szdpmaxcoords, size_t* szdpbtckdat,
    size_t* maxdbposspass2, size_t* maxdbstrspass2,
    size_t* szwrkmem, size_t* szwrkmemccd,
    size_t* szwrkmemtmalt, size_t* szwrkmemtm, size_t* szwrkmemtmibest,
    size_t* szauxwrkmem, size_t* szwrkmem2, 
    size_t* szdp2alndata, size_t* szdp2alns,
    //}}{{variables
    size_t* szglbvars,
    //}}{{overall size
    size_t* szovlpos
    //}}
    ) const
{
    MYMSG("CuMemoryBase::GetTotalMemoryReqs", 7);
    const size_t cszalnment = GetMemAlignment();
    const float pass2memperc = CLOptions::GetDEV_PASS2MEMP();
    size_t maxdbstrs = maxdbposs / CLOptions::GetDEV_EXPCT_DBPROLEN();
    //{{memory for structure data of queries
    *maxsizeqrsindex = PMBatchStrDataIndex::GetPMDataSize(totqrsposs);
    *maxsizeqrsindex += pmv2DTotIndexFlds * cszalnment;//all fields aligned
    *maxsizeqrsposs = PMBatchStrData::GetPMDataSizeUB(totqrsposs);
    *maxsizeqrsposs += pmv2DTotFlds * cszalnment;//all fields aligned
    //}}{{memory for db structure data itself
    *maxsizedbsindex = PMBatchStrDataIndex::GetPMDataSize(maxdbposs);
    *maxsizedbsindex += pmv2DTotIndexFlds * cszalnment;//all fields aligned
    *maxsizedbposs = PMBatchStrData::GetPMDataSizeUB(maxdbposs);
    //*maxsizedbposs = SLC_MIN(*maxsizedbposs, maxsizefordbstrs);//NOTE:maxdbposs unchanged
    *maxsizedbposs += pmv2DTotFlds * cszalnment;//all fields aligned
    //}}{{memory for transformation matrices
    *sztfmmtcs = GetSizeOfTfmMatrices(maxdbstrs);
    //}}{{memory for scores
    //NOTE: scores are calculated on-the-fly during DP; 
    //NOTE: still they're needed for spectrum analysis:
    *szsmatrix = 0;///GetSizeOfMtxScores(totqrsposs, maxdbposs);//NOTE: unused when 0
    //}}{{memory requirements for DP
    *szdpdiag = GetSizeOfDPDiagScores(maxdbposs);
    //NOTE: take max between diagonal and spectrum scores used independently:
    *szdpdiag = mymax(*szdpdiag, GetSizeOfPowerSpectrum(maxdbposs));
    *szdpbottom = GetSizeOfDPBottomScores(maxdbposs);
    //NOTE: take max between bottom and (query) spectrum scores used independently:
    *szdpbottom = mymax(*szdpbottom, GetSizeOfPowerSpectrum(totqrsposs));
    *szdpalnposs = GetSizeOfDPAlignedPoss(maxdbposs);
    *szdpmaxcoords = GetSizeOfDPMaxCoords(maxdbposs);
    *szdpbtckdat = GetSizeOfDPBackTckData(totqrsposs, maxdbposs);
    //}}{{alignment data memory requirements
    *maxdbposspass2 = (size_t)(maxdbposs * pass2memperc);
    *maxdbposspass2 = ALIGN_DOWN(*maxdbposspass2, CUL2CLINESIZE);
    *maxdbstrspass2 = *maxdbposspass2 / CLOptions::GetDEV_EXPCT_DBPROLEN();
    *szwrkmem = GetSizeOfWrkMemory(maxdbstrs);
    *szwrkmemccd = GetSizeOfWrkMemoryCCD(maxdbstrs);
    *szwrkmemtmalt = GetSizeOfWrkMemoryTMalt(maxdbstrs);
    *szwrkmemtm = GetSizeOfWrkMemoryTM(maxdbstrs);
    *szwrkmemtmibest = GetSizeOfWrkMemoryTMibest(maxdbstrs);
    *szauxwrkmem = GetSizeOfAuxWrkMemory(maxdbstrs);
    *szwrkmem2 = GetSizeOfWrkMemory2(maxdbstrs);
    //*szdp2alndata = GetSizeOfDP2AlnData(*maxdbstrspass2);
    //*szdp2alns = GetSizeOfDP2Alns(totqrsposs, *maxdbposspass2, *maxdbstrspass2);
    *szdp2alndata = GetSizeOfDP2AlnData(maxdbstrs);//use org. size
    *szdp2alns = GetSizeOfDP2Alns(totqrsposs, maxdbposs, maxdbstrs);//use org. size
    *szglbvars = GetSizeOfGlobVariables(maxdbstrs);
    //}}
    *szovlpos = 
        *maxsizeqrsposs + *maxsizeqrsindex +
        *maxsizedbposs + *maxsizedbsindex +
        *sztfmmtcs +
        *szsmatrix +
        *szdpdiag + *szdpbottom + *szdpalnposs + *szdpmaxcoords + *szdpbtckdat +
        *szwrkmem + *szwrkmemccd + *szwrkmemtmalt + *szwrkmemtm + *szwrkmemtmibest +
        *szauxwrkmem + *szwrkmem2 + *szdp2alndata + *szdp2alns +
        *szglbvars;
}

#endif//__CuMemoryBase_h__
