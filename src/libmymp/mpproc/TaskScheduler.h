/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __TaskScheduler_h__
#define __TaskScheduler_h__

#include "libutil/mybase.h"

#include <stdio.h>

#include <memory>
#include <vector>

#include "tsafety/TSCounterVar.h"

#include "libgenp/gdats/InputFilelist.h"
#include "libgenp/gdats/PMBatchStrData.h"
#include "libgenp/gdats/TdDataReader.h"
#include "libgenp/goutp/TdAlnWriter.h"
#include "libmymp/mplayout/MpGlobalMemory.h"
#include "libmymp/mpbatch/MpBatch.h"

// _________________________________________________________________________
// Class TaskScheduler
//
// multi-processing task scheduler 
//
class TaskScheduler
{
public:
    TaskScheduler(
            const std::vector<std::string>& inputlist,
            const std::vector<std::string>& dnamelist,
            const std::vector<std::string>& sfxlst, 
            const char* output,
            const char* cachedir
    );

    ~TaskScheduler();

    void Run();

    const std::vector<std::string>& GetInputList() const { return inputlist_; }
    const std::vector<std::string>& GetDNamelist() const { return dnamelist_; }
    const char* GetOutput() const { return output_; }

protected:
    void CreateReader( 
        int maxstrlen,
        bool mapped,
        int ndatbufs,
        int nagents
    );

    void CreateQrsReader(
        int maxstrlen,
        bool mapped,
        int ndatbufs,
        int nagents
    );

    bool GetDataFromReader(
        TdDataReader* reader,
        char**& bdbCdesc, char**& bdbCpmbeg, char**& bdbCpmend,
        char**& bdbCNdxpmbeg, char**& bdbCNdxpmend,
        TSCounterVar*& tscnt,
        bool* lastchunk);

    bool GetDataFromReader(
        TdDataReader* reader,
        char**& bdbCdesc, char**& bdbCpmbeg, char**& bdbCpmend,
        TSCounterVar*& tscnt,
        bool* lastchunk);

    bool GetReferenceData(
        char**& bdbCdescs, char**& bdbCpmbeg, char**& bdbCpmend,
        char**& bdbCNdxpmbeg, char**& bdbCNdxpmend,
        TSCounterVar*& tscnt, bool* lastchunk, bool rewind,
        size_t ntotqstrs, int queryblocks,
        char* const * const * const querypmbegs,
        char* const * const * const querypmends);

    void CreateAlnWriter( 
        const char* outdirname,
        const std::vector<std::string>& dnamelist
    );

    void NotifyAlnWriter();

    void WaitForAlnWriterToFinish(bool error);

    void CreateMemoryConfigs(size_t nareasperdevice);

    void ProcessPart(
        MpBatch&,
        size_t workerid, int chunkno, bool lastchunk,
        bool newsetqrs,
        int qrysernrbeg,
        char** queryndxpmbeg, char** queryndxpmend,
        const char** querydesc, char** querypmbeg, char** querypmend,
        const char** bdbCdesc, char** bdbCpmbeg, char** bdbCpmend,
        char** bdbCndxpmbeg, char** bdbCndxpmend,
        TSCounterVar* qrstscnt, TSCounterVar* tscnt);

    const TdAlnWriter* GetAlnWriter() const {return writer_;}
    TdAlnWriter* GetAlnWriter() {return writer_;}

    const TdDataReader* GetQrsReader() const {return qrsreader_;}
    TdDataReader* GetQrsReader() {return qrsreader_;}

    //set chunk attributes for each reference data reader
    void SetChunkDataAttributesReaders(
        size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs)
    {
        std::for_each(readers_.begin(), readers_.end(),
            [chunkdatasize, chunkdatalen, chunknstrs](std::unique_ptr<TdDataReader>& p) {
                if(p) p->SetChunkDataAttributes(
                        chunkdatasize, chunkdatalen, chunknstrs);
            }
        );
    }

private:
    const char* output_;//pattern for output file (null=standard output)
    const char* cachedir_;//directory for cached data
    std::vector<std::string> inputlist_;//input files/databases
    std::vector<std::string> dnamelist_;//reference structure files/databases
    std::vector<std::string> sfxlst_;//suffix list 
    std::vector<MpGlobalMemory*> memcfgs_;//global memory configurations

    std::unique_ptr<InputFilelist> queries_;
    std::unique_ptr<InputFilelist> references_;

    TdAlnWriter*            writer_;//alignment results writer
    TdDataReader*           qrsreader_;//reader of query structure files and databases
    std::vector<std::unique_ptr<TdDataReader>> readers_;//structure files and database readers
};


////////////////////////////////////////////////////////////////////////////
// INLINES
//

#endif//__TaskScheduler_h__
