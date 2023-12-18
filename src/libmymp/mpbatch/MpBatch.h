/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __MpBatch_h__
#define __MpBatch_h__

#include "libutil/mybase.h"

#include <stdio.h>
#include <stdlib.h>

#include <memory>
#include <mutex>

#include "tsafety/TSCounterVar.h"
#include "libgenp/goutp/TdAlnWriter.h"
#include "libgenp/goutp/TdFinalizer.h"
#include "libmymp/mplayout/MpGlobalMemory.h"

////////////////////////////////////////////////////////////////////////////
// CLASS MpBatch
// multi-processing Batch computation of structure alignment
//
class MpBatch
{
public:
    MpBatch(MpGlobalMemory*, int dareano, TdAlnWriter*);

    ~MpBatch();

    void ProcessBlock(
        int qrysernrbeg,
        char** queryndxpmbeg,
        char** queryndxpmend,
        const char** querydesc,
        char** querypmbeg,
        char** querypmend,
        const char** bdbCdesc,
        char** bdbCpmbeg,
        char** bdbCpmend,
        char** bdbCndxpmbeg,
        char** bdbCndxpmend,
        TSCounterVar* qrscnt,
        TSCounterVar* cnt
    );

    // void WaitForIdleChilds() {
    //     if(cbpfin_)
    //         std::lock_guard<std::mutex> lck(cbpfin_->GetPrivateMutex());
    // }

    size_t GetCurrentMaxDbPos() const { return gmem_->GetCurrentMaxDbPos(); }
    size_t GetCurrentMaxNDbStrs() const { return gmem_->GetCurrentMaxNDbStrs(); }

    size_t GetCurrentMaxDbPosPass2() const { return gmem_->GetCurrentMaxDbPosPass2(); }
    size_t GetCurrentMaxNDbStrsPass2() const { return gmem_->GetCurrentMaxNDbStrsPass2(); }

protected:

    size_t GetHeapSectionOffset(int sec) const {return gmem_->GetHeapSectionOffset(devareano_,sec);}
    char* GetHeapSectionAddress(int sec) const {
        return gmem_->GetHeap() + gmem_->GetHeapSectionOffset(devareano_,sec);
    }

    const std::string& GetDeviceName() const {return gmem_->GetDeviceName();}

    //Filter out flagged references
    void FilteroutReferences(
        char** queryndxpmbeg, char** queryndxpmend,
        char** querypmbeg, char** querypmend,
        const char** bdbCdesc, char** bdbCpmbeg, char** bdbCpmend,
        char** bdbCndxpmbeg, char** bdbCndxpmend,
        const size_t nqyposs, const size_t nqystrs,
        const size_t ndbCposs, const size_t ndbCstrs,
        const size_t dbstr1len, const size_t dbstrnlen,
        const size_t dbxpad, const size_t maxnsteps,
        size_t& ndbCposs2, size_t& ndbCstrs2,
        size_t& dbstr1len2, size_t& dbstrnlen2,
        size_t& dbxpad2,
        float* tmpdpdiagbuffers,
        float* tfmmemory,
        float* auxwrkmemory,
        unsigned int* globvarsbuf);

    //{{ results processing
    void TriggerFinalization(
        double tdrtn,
        int qrysernrbeg,
        size_t nqyposs, size_t nqystrs,
        size_t ndbCposs, size_t ndbCstrs,
        const char** querydesc,
        char** querypmbeg,
        char** querypmend,
        const char** bdbCdesc,
        char** bdbCpmbeg,
        char** bdbCpmend,
        TSCounterVar* qrscnt,
        TSCounterVar* cnt,
        size_t szaligns2,
        unsigned int* passedstats
    );
    //}}

private:
    MpGlobalMemory* gmem_;//global memory configuration
    const int devareano_;//memory area number
    std::unique_ptr<unsigned int[]> filterdata_;//passed structures (globals)
    //statistics of structures passed to the next processing stages (globals):
    std::unique_ptr<unsigned int[]> passedstatscntrd_;
    std::unique_ptr<TdFinalizer> cbpfin_;//results finalizer
};

// -------------------------------------------------------------------------
// INLINES ...
//

#endif//__MpBatch_h__
