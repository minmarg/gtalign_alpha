/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __JobDispatcher_h__
#define __JobDispatcher_h__

#include "libutil/mybase.h"

#include <stdio.h>

#include <memory>
#include <vector>

#include "libutil/CLOptions.h"
#include "tsafety/TSCounterVar.h"

#include "libgenp/gdats/InputFilelist.h"
#include "libgenp/gdats/PMBatchStrData.h"
#include "libgenp/gdats/TdDataReader.h"
// #include "libmycu/culayout/CuDeviceMemory.cuh"
#include "libmycu/cubatch/TdCommutator.h"
#include "libgenp/goutp/TdClustWriter.h"
#include "libgenp/goutp/TdAlnWriter.h"
// #include "libmycu/cuproc/Devices.h"

#define JDSP_WORKER_BUSY -1
#define JDSP_WORKER_NONE -2

class JobDispatcher;

template<typename T>
using TWriteDataForWorker1 = 
        void (JobDispatcher::*)(
            int req, int addr,
            TdCommutator*, 
            const std::vector<T>&,
            std::unique_ptr<PMBatchStrData>);
template<typename T>
using TGetDataFromWorker1 = 
        void (JobDispatcher::*)(
            int req, int rsp, int addr,
            TdCommutator*,
            std::vector<T>&);

// _________________________________________________________________________
// Class JobDispatcher
//
// Implementation of distributing jobs over CPU and GPU threads
//
class JobDispatcher
{
public:
    JobDispatcher(
            const std::vector<std::string>& inputlist,
            const std::vector<std::string>& dnamelist,
            const std::vector<std::string>& sfxlst, 
            const char* output,
            const char* cachedir
    );

    JobDispatcher(
            const std::vector<std::string>& clustlist,
            const std::vector<std::string>& sfxlst, 
            const char* output,
            const char* cachedir
    );

    ~JobDispatcher();

    void Run();
    void RunClust();

    const std::vector<std::string>& GetInputList() const { return inputlist_; }
    const std::vector<std::string>& GetDNamelist() const { return dnamelist_; }
    const std::vector<std::string>& GetClustList() const { return clustlist_; }
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

    void CreateClustWriter( 
        const char* outdirname,
        const std::vector<std::string>& clustlist,
        const std::vector<std::string>& devnames,
        const int nmaxchunkqueries,
        const int nagents
    );

    void NotifyAlnWriter();
    void WaitForAlnWriterToFinish(bool error);

    void NotifyClustWriter();
    void WaitForClustWriterToFinish(bool error);


    void CreateDevMemoryConfigs(size_t nareasperdevice);


    void CreateWorkerThreads(
        size_t chunkdatasize, size_t chunkdatalen, size_t chunknstrs);

    void SubmitWorkerJob(
        int tid, int chunkno, bool lastchunk,
        bool newsetqrs,
        int qrysernrbeg,
        float scorethld,
        char** queryndxpmbeg, char** queryndxpmend,
        const char** querydesc, char** querypmbeg, char** querypmend,
        const char** bdbCdesc, char** bdbCpmbeg, char** bdbCpmend,
        char** bdbCndxpmbeg, char** bdbCndxpmend,
        TSCounterVar* qrstscnt, TSCounterVar* tscnt);

    void WaitForWorker(int tid);

    void ProbeWorker(int tid);

    void TerminateWorker(int tid);

    void TerminateAllWorkers();

    void WaitForAllWorkersToFinish();

    void GetAvailableWorker(int* tid);
    void WaitForAvailableWorker(int* tid);
    void GetNextWorker(int* tid);



    void ProcessPart(
        int tid, int chunkno, bool lastchunk,
        bool newsetqrs,
        int qrysernrbeg,
        float scorethld,
        char** queryndxpmbeg, char** queryndxpmend,
        const char** querydesc, char** querypmbeg, char** querypmend,
        const char** bdbCdesc, char** bdbCpmbeg, char** bdbCpmend,
        char** bdbCndxpmbeg, char** bdbCndxpmend,
        TSCounterVar* qrstscnt, TSCounterVar* tscnt);



    const TdAlnWriter* GetAlnWriter() const {return writer_;}
    TdAlnWriter* GetAlnWriter() {return writer_;}

    const TdClustWriter* GetClustWriter() const {return clustwriter_;}
    TdClustWriter* GetClustWriter() {return clustwriter_;}

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
    std::vector<std::string> clustlist_;//input files/databases for clustering
    std::vector<std::string> sfxlst_;//suffix list 
    std::vector<CuDeviceMemory*> memdevs_;//memory configurations for devices
    std::vector<TdCommutator*> hostworkers_;//host worker threads

    std::unique_ptr<InputFilelist> queries_;
    std::unique_ptr<InputFilelist> references_;

    TdAlnWriter* writer_;//alignment results writer
    TdClustWriter* clustwriter_;//clustering results writer
    TdDataReader* qrsreader_;//reader of query structure files and databases
    std::vector<std::unique_ptr<TdDataReader>> readers_;//structure files and database readers
};


////////////////////////////////////////////////////////////////////////////
// INLINES
//

#endif//__JobDispatcher_h__
