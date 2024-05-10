/***************************************************************************
 *   Copyright (C) 2021-2023 Mindaugas Margelevicius                       *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#include "libutil/mybase.h"
// #include "libutil/alpha.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <vector>
#include <thread>
#include <algorithm>

#include "incconf/localconfig.h"
#include "libutil/mygetopt.h"
#include "libutil/CLOptions.h"
#include "libgenp/gdats/PMBatchStrData.h"
#include "libgenp/gdats/FlexDataRead.h"

#ifdef GPUINUSE
#   include "libmycu/cuproc/Devices.h"
#   include "libmycu/cuproc/JobDispatcher.h"
#else
#   include "libmymp/mpproc/TaskScheduler.h"
#endif

#include "gtalign.h"

#   include "libmymp/mputil/simdscan.h"

// =========================================================================
// declarations:
//
void SplitString(std::string argstring, std::string option, std::vector<std::string>&);

inline int mystring2int(int& retval, const std::string& strval, const char* errstr)
{
    char* p;
    retval = strtol( strval.c_str(), &p, 10 );
    if( errno || *p ) {
        error( errstr );
        return EXIT_FAILURE;
    }
    return 0;
}

inline int mystring2float(float& retval, const std::string& strval, const char* errstr)
{
    char* p;
    retval = strtof( strval.c_str(), &p);
    if( errno || *p ) {
        error( errstr );
        return EXIT_FAILURE;
    }
    return 0;
}

// =========================================================================

int main( int argc, char *argv[] )
{
    int c;
    float f;
    char* p;
//     char            strbuf[BUF_MAX];
    //names of input file and database
    std::string     myoptarg;
    //
    std::vector<std::string> qrydblst;//list of query databases/directories
    std::vector<std::string> refdblst;//list of reference structure databases/directories
    std::vector<std::string> clsdblst;//list of databases/directories for clustering
    std::vector<std::string> a2adblst;//list of databases/directories for all-against-all alignment
    std::vector<std::string> sfxlst;//list of suffices to search for structure files in directories
    //
    std::string     qrydb;//db (directory) of queries
    std::string     refdb;//reference db (directory)
    std::string     clsdb;//db to cluster
    std::string     a2adb;//all-against-all db
    std::string     sfxs;//suffices
    std::string     outdir;//output directory
    std::string     cachedir;//cache directory
    std::string     clsthld;//threshold score for clustering
    std::string     clscoverage;//coverage threshold for clustering
    bool            cls_onesided_coverage = 0;//flag of one-sided coverage
    bool            cls_output_seqs = 0;//flag for output of sequences
    std::string     clsalgorithm;//clustering algorithm
    //
    std::string     thrscore;//threshold score
    bool            sectmscore = 0;//include 2TM-score in calculations
    std::string     sortby;//key code for sorting results
    std::string     nhits;//#hits to show
    std::string     nalns;//#alignments to output
    std::string     wrap;//wrap width
    bool            no_deletions = 0;//remove deletion positions
    bool            referenced = 0;//reference tfms
    std::string     outfmt;//output format
    //
    std::string     inpfmt;//input format
    std::string     atom;//atom name
    bool            hetatm = 0;//consider HETATM records
    std::string     term;//structure terminator
    std::string     split;//structure split approach
    std::string     superp;//superposition/alignment algorithm
    //
    std::string     preseqsim;//sequence similarity threshold score for pre-screening
    std::string     prescore;//threshold provisional score for pre-screening
    //
    std::string     initalnfile;//file of initial alignment
    std::string     permalnfile;//file of permanent alignment
    std::string     d0str;//scale distance d0
    std::string     normlength;//normalization length
    std::string     normcode;//TM-score normalization code
    bool            symmetric = 0;//flag of symmetric alignments
    std::string     refinement;//refinement detail
    std::string     depth;//superposition search depth
    std::string     trigger;//provisional similarity percentage
    std::string     nbranches;//# final superposition branches to explore
    bool            addsearchbyss = 0;//flag for adding search by ss
    bool            nodetailedsearch = 0;//flag for no detailed search
    std::string     convergence;//#convergence tests
    std::string     speed;//spped flag
    bool            cperm = 0;//flag of circular permutation
    bool            mirror = 0;//query mirror image
    //
    std::string     verbose;
    std::string     cputhreadsreading;//#cpu threads for reading
    std::string     cputhreads;//#cpu threads for computation
    std::string     devmaxnqrs;//max #queries in the chunk
    std::string     devmaxqres;//max total length of queries
    std::string     devmaxrlen;//max reference length
    std::string     devminrlen;//min reference length
    bool            nofilesort = 0;//flag of no file sorting
    std::string     devmaxN;
    std::string     devmaxmem;
    std::string     expct_prolen;
    std::string     devpass2memp;
    std::string     nbuffers;//number of input buffers for read
    bool            filemap = 0;//use file mapping
    bool            unpinned = 0;//use unpinned CPU memory
    bool            listdevices = 0;//whether to list available devices
    int             verblev = 0;//suppress warnings

    SetArguments( &argc, &argv );
    SetProgramName( argv[0], version );

    if( argc <= 1 ) {
        fprintf( stdout, "%s", usage(argv[0],instructs,version,verdate).c_str());
// float f=0.0f;
// ((char*)&f)[3] = ((char*)&f)[3] | 0x41;
// printf(" %g %.6f %08x\n",f,f,*(reinterpret_cast<int*>(&f)));
// constexpr int DIM = 64;
// float data[DIM+1]={0},data1[DIM+1]={0},data2[DIM]={0},tmp[DIM],max1=0.0f,min1=0.0f;
// int ndx[DIM], tmpndx[DIM];
// for(int ii=0;ii<DIM;ii++) ndx[ii] = ii;
// std::srand(33);
// for(int ii=0/*1*/;ii<DIM;ii++) data[ii] = (int)((float)std::rand()/(float)RAND_MAX*10.f-5.0f);
// printf("Data: ");for(int ii=0;ii<DIM;ii++) printf(" %3.0f",data1[ii]=data[ii]);printf("\n");
// for(int ii=0;ii<DIM;ii++) printf(" %.0f",data[ndx[ii]]);printf("\n");
// for(size_t ii=0;ii<1000000000UL;ii++){std::nth_element(data,data+16,data+DIM);}
// for(size_t ii=0;ii<1000000000UL;ii++){std::nth_element(ndx,ndx+16,ndx+DIM,[data](int i, int j){return data[i]>data[j];});}
// StaticSort<DIM> staticSort;for(size_t ii=0;ii<1000000000UL;ii++) staticSort(data);
// for(int ii=0;ii<DIM;ii++) printf(" %.0f",data[ndx[ii]]);printf("\n");
// MYMSG("Done",0);
// for(int dd=0;dd<DIM;dd+=4) {
//     mysimdincprefixsum<4>(4,data+dd,tmp);data[dd+4]+=data[dd+4-1];
//     #pragma omp simd
//     for(int ii=0;ii<4;ii++) data1[dd+ii] = data[dd+ii];
//     data1[dd]=mymin(data1[dd],min1);
//     mysimdincprefixminmax<4,PFX_MIN>(4,data1+dd,tmp);
//     min1=data1[dd+4-1];
//     #pragma omp simd
//     for(int ii=0;ii<4;ii++) data[dd+ii] -= mymin(0.0f, data1[dd+ii]);
//     #pragma omp simd reduction(max:max1)
//     for(int ii=0;ii<4;ii++) max1 = mymax(max1, data[dd+ii]);
//     printf(" max: %.0f\n\n", max1);
// }
// // mysimdincprefixsum<DIM>(DIM,data,tmp);
// MYMSG("Done",0);
// printf("Sums: ");for(int ii=0;ii<DIM;ii++) printf(" %3.0f",/* data1[ii]= */data[ii]);printf("\n\n");
// MYMSG("Done",0);
// // mysimdincprefixminmax<DIM,PFX_MIN>(DIM,data1,tmp);
// MYMSG("Done",0);
// printf("Mins: ");for(int ii=0;ii<DIM;ii++) printf(" %3.0f",data1[ii]);printf("\n\n");
// MYMSG("Done",0);
// #pragma omp simd
// for(int ii=0;ii<DIM;ii++) data[ii] -= mymin(0.0f, data1[ii]);
// MYMSG("Done",0);
// printf("Runs: ");for(int ii=0;ii<DIM;ii++) printf(" %3.0f",data[ii]);printf("\n\n");
// #pragma omp simd reduction(max:max1)
// for(int ii=0;ii<DIM;ii++) max1 = mymax(max1, data[ii]);
// printf("Max: %.0f\n\n", max1);
// return 0;
// for(int i = 0; i < 15; i++) printf(" = %d\n",1<<i);
/*
size_t c=1000000;
std::srand(std::time(0));
std::vector<float> v; v.reserve(c);
for(size_t i=1;i<c;i++) v.push_back((float)std::rand()/(float)RAND_MAX*10.f);
float sum1=0.f,sum2=0.f,sum3=0.f;
for(size_t i=0;i<v.size();i++) sum1+=v[i]*v[i];
std::random_shuffle(v.begin(),v.end());
for(size_t i=0;i<v.size();i++) sum2+=v[i]*v[i];
std::random_shuffle(v.begin(),v.end());
for(size_t i=0;i<v.size();i++) sum3+=v[i]*v[i];
printf(" %f %f %f   %d %d %d\n",sum1,sum2,sum3, sum1==sum2,sum1==sum3,sum2==sum3);
*/
        return EXIT_SUCCESS;
    }

    enum {
        gtaOptionsStartWithValue = 99,
        gtaOpt_qrs, gtaOpt_rfs, gtaOpt_a2a, gtaOpt_sfx, gtaOpt_o, gtaOpt_c,
        //
        gtaOpt_cls, gtaOpt_cls_threshold, gtaOpt_cls_coverage,
        gtaOpt_cls_one_sided_coverage, gtaOpt_cls_out_sequences,
        gtaOpt_cls_algorithm,
        //
        gtaOpt_s, gtaOpt_2tm_score, gtaOpt_sort, gtaOpt_nhits, gtaOpt_nalns,
        gtaOpt_wrap, gtaOpt_no_deletions, gtaOpt_referenced, gtaOpt_outfmt,
        //
        gtaOpt_infmt, gtaOpt_atom, gtaOpt_hetatm, gtaOpt_ter, gtaOpt_split,
        gtaOpt_superp,
        //
        gtaOpt_pre_similarity, gtaOpt_pre_score,
        //
        gtaOpt_i, gtaOpt_I, gtaOpt_d0, gtaOpt_u, gtaOpt_a,
        gtaOpt_symmetric, gtaOpt_refinement,
        gtaOpt_depth, gtaOpt_trigger, gtaOpt_nbranches,
        gtaOpt_add_search_by_ss, gtaOpt_no_detailed_search,
        gtaOpt_convergence, gtaOpt_speed,
        gtaOpt_cp, gtaOpt_mirror,
        //
        gtaOpt_cpu_threads_reading, gtaOpt_cpu_threads,
        gtaOpt_dev_queries_per_chunk,
        gtaOpt_dev_queries_total_length_per_chunk,
        gtaOpt_dev_max_length, gtaOpt_dev_min_length, gtaOpt_no_file_sort,
        //
        gtaOpt_dev_N, gtaOpt_dev_mem, gtaOpt_dev_expected_length, gtaOpt_dev_pass2memp,
        gtaOpt_io_nbuffers, gtaOpt_io_filemap, gtaOpt_io_unpinned,
        //
        gtaOpt_dev_list, gtaOpt_v, gtaOpt_h
    };

    static struct myoption long_options[] = {
        {"qrs", my_required_argument, gtaOpt_qrs},
        {"rfs", my_required_argument, gtaOpt_rfs},
        {"a2a", my_required_argument, gtaOpt_a2a},
        {"sfx", my_required_argument, gtaOpt_sfx},
        {"o", my_required_argument, gtaOpt_o},
        {"c", my_required_argument, gtaOpt_c},
        //
        {"cls", my_required_argument, gtaOpt_cls},
        {"cls-threshold", my_required_argument, gtaOpt_cls_threshold},
        {"cls-coverage", my_required_argument, gtaOpt_cls_coverage},
        {"cls-one-sided-coverage", my_no_argument, gtaOpt_cls_one_sided_coverage},
        {"cls-out-sequences", my_no_argument, gtaOpt_cls_out_sequences},
        {"cls-algorithm", my_required_argument, gtaOpt_cls_algorithm},
        //
        {"s", my_required_argument, gtaOpt_s},
        {"2tm-score", my_no_argument, gtaOpt_2tm_score},
        {"sort", my_required_argument, gtaOpt_sort},
        {"nhits", my_required_argument, gtaOpt_nhits},
        {"nalns", my_required_argument, gtaOpt_nalns},
        {"wrap", my_required_argument, gtaOpt_wrap},
        {"no-deletions", my_no_argument, gtaOpt_no_deletions},
        {"referenced", my_no_argument, gtaOpt_referenced},
        {"outfmt", my_required_argument, gtaOpt_outfmt},
        //
        {"infmt", my_required_argument, gtaOpt_infmt},
        {"atom", my_required_argument, gtaOpt_atom},
        {"hetatm", my_no_argument, gtaOpt_hetatm},
        {"ter", my_required_argument, gtaOpt_ter},
        {"split", my_required_argument, gtaOpt_split},
        {"superp", my_required_argument, gtaOpt_superp},
        //
        {"pre-similarity", my_required_argument, gtaOpt_pre_similarity},
        {"pre-score", my_required_argument, gtaOpt_pre_score},
        //
        {"i", my_required_argument, gtaOpt_i},
        {"I", my_required_argument, gtaOpt_I},
        {"d0", my_required_argument, gtaOpt_d0},
        {"u", my_required_argument, gtaOpt_u},
        {"a", my_required_argument, gtaOpt_a},
        {"symmetric", my_no_argument, gtaOpt_symmetric},
        {"refinement", my_required_argument, gtaOpt_refinement},
        {"depth", my_required_argument, gtaOpt_depth},
        {"trigger", my_required_argument, gtaOpt_trigger},
        {"nbranches", my_required_argument, gtaOpt_nbranches},
        {"add-search-by-ss", my_no_argument, gtaOpt_add_search_by_ss},
        {"no-detailed-search", my_no_argument, gtaOpt_no_detailed_search},
        {"convergence", my_required_argument, gtaOpt_convergence},
        {"speed", my_required_argument, gtaOpt_speed},
        {"cp", my_no_argument, gtaOpt_cp},
        {"mirror", my_no_argument, gtaOpt_mirror},
        //
        {"cpu-threads-reading", my_required_argument, gtaOpt_cpu_threads_reading},
        {"cpu-threads", my_required_argument, gtaOpt_cpu_threads},
        {"dev-queries-per-chunk", my_required_argument, gtaOpt_dev_queries_per_chunk},
        {"dev-queries-total-length-per-chunk", my_required_argument, gtaOpt_dev_queries_total_length_per_chunk},
        {"dev-max-length", my_required_argument, gtaOpt_dev_max_length},
        {"dev-min-length", my_required_argument, gtaOpt_dev_min_length},
        {"no-file-sort", my_no_argument, gtaOpt_no_file_sort},
        //
        {"dev-N", my_required_argument, gtaOpt_dev_N},
        {"dev-mem", my_required_argument, gtaOpt_dev_mem},
        {"dev-expected-length", my_required_argument, gtaOpt_dev_expected_length},
        {"dev-pass2memp", my_required_argument, gtaOpt_dev_pass2memp},
        {"io-nbuffers", my_required_argument, gtaOpt_io_nbuffers},
        {"io-filemap", my_no_argument, gtaOpt_io_filemap},
        {"io-unpinned", my_no_argument, gtaOpt_io_unpinned},
        //
        {"dev-list", my_no_argument, gtaOpt_dev_list},
        {"v", my_optional_argument, gtaOpt_v},
        {"h", my_no_argument, gtaOpt_h},
        { NULL, my_n_targflags, 0 }
    };

    try {
        try {
            MyGetopt mygetopt( long_options, (const char**)argv, argc );
            while(( c = mygetopt.GetNextOption( &myoptarg )) >= 0 ) {
                switch( c ) {
                    case ':':   fprintf( stdout, "Argument missing. Please try option -h for help.%s", NL );
                                return EXIT_FAILURE;
                    case '?':   fprintf( stdout, "Unrecognized option. Please try option -h for help.%s", NL );
                                return EXIT_FAILURE;
                    case '!':   fprintf( stdout, "Ill-formed option. Please try option -h for help.%s", NL );
                                return EXIT_FAILURE;
                    case gtaOpt_h: fprintf( stdout, "%s", usage(argv[0],instructs,version,verdate).c_str());
                                return EXIT_SUCCESS;
                    //
                    case gtaOpt_qrs:    qrydb = myoptarg; break;
                    case gtaOpt_rfs:    refdb = myoptarg; break;
                    case gtaOpt_a2a:    a2adb = myoptarg; break;
                    case gtaOpt_sfx:    sfxs = myoptarg; break;
                    case gtaOpt_o:      outdir = myoptarg; break;
                    case gtaOpt_c:      cachedir = myoptarg; break;
                    //
                    case gtaOpt_cls:    clsdb = myoptarg; break;
                    case gtaOpt_cls_threshold:  clsthld = myoptarg; break;
                    case gtaOpt_cls_coverage:   clscoverage = myoptarg; break;
                    case gtaOpt_cls_one_sided_coverage: cls_onesided_coverage = 1; break;
                    case gtaOpt_cls_out_sequences: cls_output_seqs = 1; break;
                    case gtaOpt_cls_algorithm:  clsalgorithm = myoptarg; break;
                    //
                    case gtaOpt_s:      thrscore = myoptarg; break;
                    case gtaOpt_2tm_score: sectmscore = 1; break;
                    case gtaOpt_sort:   sortby = myoptarg; break;
                    case gtaOpt_nhits:  nhits = myoptarg; break;
                    case gtaOpt_nalns:  nalns = myoptarg; break;
                    case gtaOpt_wrap:   wrap = myoptarg; break;
                    case gtaOpt_no_deletions: no_deletions = 1; break;
                    case gtaOpt_referenced: referenced = 1; break;
                    case gtaOpt_outfmt: outfmt = myoptarg; break;
                    //
                    case gtaOpt_infmt:  inpfmt = myoptarg; break;
                    case gtaOpt_atom:   atom = myoptarg; break;
                    case gtaOpt_hetatm: hetatm = 1; break;
                    case gtaOpt_ter:    term = myoptarg; break;
                    case gtaOpt_split:  split = myoptarg; break;
                    case gtaOpt_superp: superp = myoptarg; break;
                    //
                    case gtaOpt_pre_similarity:  preseqsim = myoptarg; break;
                    case gtaOpt_pre_score:  prescore = myoptarg; break;
                    //
                    case gtaOpt_i:      initalnfile = myoptarg; break;
                    case gtaOpt_I:      permalnfile = myoptarg; break;
                    case gtaOpt_d0:     d0str = myoptarg; break;
                    case gtaOpt_u:      normlength = myoptarg; break;
                    case gtaOpt_a:      normcode = myoptarg; break;
                    case gtaOpt_symmetric:  symmetric = 1; break;
                    case gtaOpt_refinement: refinement = myoptarg; break;
                    case gtaOpt_depth:      depth = myoptarg; break;
                    case gtaOpt_trigger:    trigger = myoptarg; break;
                    case gtaOpt_nbranches:  nbranches = myoptarg; break;
                    case gtaOpt_add_search_by_ss: addsearchbyss = 1; break;
                    case gtaOpt_no_detailed_search: nodetailedsearch = 1; break;
                    case gtaOpt_convergence: convergence = myoptarg; break;
                    case gtaOpt_speed:      speed = myoptarg; break;
                    case gtaOpt_cp:     cperm = 1; break;
                    case gtaOpt_mirror: mirror = 1; break;
                    //
                    case gtaOpt_cpu_threads_reading:    cputhreadsreading = myoptarg; break;
                    case gtaOpt_cpu_threads:            cputhreads = myoptarg; break;
                    case gtaOpt_dev_queries_per_chunk:  devmaxnqrs = myoptarg; break;
                    case gtaOpt_dev_queries_total_length_per_chunk: devmaxqres = myoptarg; break;
                    case gtaOpt_dev_max_length:         devmaxrlen = myoptarg; break;
                    case gtaOpt_dev_min_length:         devminrlen = myoptarg; break;
                    case gtaOpt_no_file_sort:           nofilesort = 1; break;
                    //
                    case gtaOpt_dev_N:          devmaxN = myoptarg; break;
                    case gtaOpt_dev_mem:        devmaxmem = myoptarg; break;
                    case gtaOpt_dev_expected_length: expct_prolen = myoptarg; break;
                    case gtaOpt_dev_pass2memp:  devpass2memp = myoptarg; break;
                    case gtaOpt_io_nbuffers:    nbuffers = myoptarg; break;
                    case gtaOpt_io_filemap:     filemap = 1; break;
                    case gtaOpt_io_unpinned:    unpinned = 1; break;
                    //
                    case gtaOpt_dev_list:   listdevices = 1; break;
                    case gtaOpt_v:          verblev = 1; verbose = myoptarg; break;
                    default:    break;
                }
            }
        } catch( myexception const& ex ) {
            error( dynamic_cast<myruntime_error const&>(ex).pretty_format().c_str());
            return EXIT_FAILURE;
        }
    } catch ( ... ) {
        error("Unknown exception caught.");
        return EXIT_FAILURE;
    }


    if( !verbose.empty()) {
        verblev = strtol( verbose.c_str(), &p, 10 );
        if( errno || *p || verblev < 0 ) {
            error( "Invalid verbose mode argument." );
            return EXIT_FAILURE;
        }
    }

    SetVerboseMode( verblev );


    //{{determine #cpu threads
    TRY
        if(!cputhreadsreading.empty()) {
            if(mystring2int(c, cputhreadsreading, "Invalid argument of option --cpu-threads-reading."))
                return EXIT_FAILURE;
            CLOPTASSIGN(CPU_THREADS_READING, c);
        }

        if(!cputhreads.empty()) {
            if(mystring2int(c, cputhreads, "Invalid argument of option --cpu-threads."))
                return EXIT_FAILURE;
            CLOPTASSIGN(CPU_THREADS, c);
        } else {//cputhreads.empty()
            int nreadthreads = CLOptions::GetCPU_THREADS_READING();
            int ncores = std::thread::hardware_concurrency();
            CLOPTASSIGN(CPU_THREADS, mymax(1, ncores - nreadthreads));
        }
    CATCH_ERROR_RETURN(;);
    //}}


    //{{list available devices and exit
    if(listdevices) {
        TRY
#ifdef GPUINUSE
            DEVPROPs.PrintDevices(stdout);
#else
            fprintf(stdout, "%sNOTE: %s compiled without GPU support.%s%s",NL,PROGNAME,NL,NL);
#endif
            fprintf(stdout, "%sDefault number of CPU threads (option --cpu-threads): %d%s%s",
                NL,CLOptions::GetCPU_THREADS(),NL,NL);
        CATCH_ERROR_RETURN(;);
        return EXIT_SUCCESS;
    }
    //}}


    //{{ COMMAND-LINE OPTIONS
    TRY
        if(clsdb.empty()) {
            if(qrydb.empty()) {
                error("Query structure or database/directory should be specified.");
                return EXIT_FAILURE;
            }
            if(refdb.empty()) {
                error("Reference structure or database/directory should be specified.");
                return EXIT_FAILURE;
            }
            //
            if(!qrydb.empty()) SplitString(qrydb, "--qrs", qrydblst);
            if(!refdb.empty()) SplitString(refdb, "--rfs", refdblst);
        }
        else if(!(qrydb.empty() && refdb.empty())) {
            error("Clustering cannot be configured with "
            "additional input query or reference structures.");
            return EXIT_FAILURE;
        }
        else {
#ifdef GPUINUSE
            SplitString(clsdb, "--cls", clsdblst);
#else
            error("The clustering option is available only in the GPU version.");
            return EXIT_FAILURE;
#endif
        }

        if(!sfxs.empty())
            SplitString(sfxs, "--sfx", sfxlst);

        if( outdir.empty()) {
            error( "Output directory is not specified." );
            return EXIT_FAILURE;
        }

        if( directory_exists(outdir.c_str()) == false && 
            mymkdir(outdir.c_str()) < 0 )
        {
            error(("Failed to create directory: " + outdir).c_str());
            return EXIT_FAILURE;
        }

        if(!cachedir.empty() &&
            directory_exists(cachedir.c_str()) == false &&
            mymkdir(cachedir.c_str()) < 0)
        {
            error(("Failed to create cache directory: " + cachedir).c_str());
            return EXIT_FAILURE;
        }

        CLOPTASSIGN(B_CACHE_ON, !cachedir.empty());
        CLOPTASSIGN(B_CLS_ONE_SIDED_COVERAGE, cls_onesided_coverage);
        CLOPTASSIGN(B_CLS_OUT_SEQUENCES, cls_output_seqs);


        print_dtime(1);


        if( !thrscore.empty()) {
            if( mystring2float(f, thrscore, "Invalid argument of option -s."))
                return EXIT_FAILURE;
            CLOPTASSIGN(O_S, f);
        }

        CLOPTASSIGN(O_2TM_SCORE, sectmscore);

        if(!clsdblst.empty()) {
            if(!thrscore.empty()) warning("Option -s ignored for clustering.");
            CLOPTASSIGN(O_S, 0.0f);
            if(CLOptions::GetO_2TM_SCORE()) {
                error("Option --2tm-score cannot be set for clustering.");
                return EXIT_FAILURE;
            }
        }

        if( !sortby.empty()) {
            if( mystring2int(c, sortby, "Invalid argument of option --sort."))
                return EXIT_FAILURE;
            if(CLOptions::GetO_2TM_SCORE() == 0 && CLOptions::osnOSorting <= c) {
                error("Set option --2tm-score to sort results by 2TM-score.");
                return EXIT_FAILURE;
            }
            CLOPTASSIGN(O_SORT, c);
        }

        if( !clsthld.empty()) {
            if( mystring2float(f, clsthld, "Invalid argument of option --cls-threshold."))
                return EXIT_FAILURE;
            const int sortby = CLOptions::GetO_SORT();
            if(1.0f < f && sortby != CLOptions::osRMSD) {
                error("Argument of option --cls-threshold cannot be >1.0 for "
                    "TM-score thresholds (--sort).");
                return EXIT_FAILURE;
            }
            CLOPTASSIGN(B_CLS_THRESHOLD, f);
        }

        if( !clscoverage.empty()) {
            if( mystring2float(f, clscoverage, "Invalid argument of option --cls-coverage."))
                return EXIT_FAILURE;
            CLOPTASSIGN(B_CLS_COVERAGE, f);
        }

        if( !clsalgorithm.empty()) {
            if( mystring2int(c,clsalgorithm, "Invalid argument of option --cls-algorithm."))
                return EXIT_FAILURE;
            CLOPTASSIGN(B_CLS_ALGORITHM, c);
        }


        if( !nhits.empty()) {
            if( mystring2int(c, nhits, "Invalid argument of option --nhits."))
                return EXIT_FAILURE;
            CLOPTASSIGN(O_NHITS, c);
        }

        if( !nalns.empty()) {
            if( mystring2int(c, nalns, "Invalid argument of option --nalns."))
                return EXIT_FAILURE;
            CLOPTASSIGN(O_NALNS, c);
        }

        if(!wrap.empty()) {
            if(mystring2int(c, wrap, "Invalid argument of option --wrap."))
                return EXIT_FAILURE;
            CLOPTASSIGN(O_WRAP, c);
        }

        CLOPTASSIGN(O_NO_DELETIONS, no_deletions);
        CLOPTASSIGN(O_REFERENCED, referenced);

        if( !outfmt.empty()) {
            if( mystring2int(c, outfmt, "Invalid argument of option --outfmt."))
                return EXIT_FAILURE;
            CLOPTASSIGN(O_OUTFMT, c);
        }


        if( !inpfmt.empty()) {
            if( mystring2int(c, inpfmt, "Invalid argument of option --infmt."))
                return EXIT_FAILURE;
            CLOPTASSIGN(I_INFMT, c);
        }

        if( !atom.empty()) {
            if( atom.size() != 4 ) {
                error( "Option --atom is to be exactly a 4-character string." );
                return EXIT_FAILURE;
            }
            CLOPTASSIGN(I_ATOM_PROT, atom);
            CLOPTASSIGN(I_ATOM_RNA, atom);

            //trim leading and trailing whitespaces
            size_t atpos = atom.find_first_not_of(" \t");
            atom = (atpos == std::string::npos)? "": atom.substr(atpos);
            atpos = atom.find_last_not_of(" \t");
            atom = (atpos == std::string::npos)? "": atom.substr(0, atpos+1);

            if(atom.empty()) {
                error( "Invalid argument of option --atom." );
                return EXIT_FAILURE;
            }

            CLOPTASSIGN(I_ATOM_PROT_trimmed, atom);
            CLOPTASSIGN(I_ATOM_RNA_trimmed, atom);
        }

        CLOPTASSIGN(I_HETATM, hetatm);

        if( !term.empty()) {
            if( mystring2int(c, term, "Invalid argument of option --ter."))
                return EXIT_FAILURE;
            CLOPTASSIGN(I_TER, c);
        }

        //NOTE: should follow term
        if( !split.empty()) {
            if( mystring2int(c, split, "Invalid argument of option --split."))
                return EXIT_FAILURE;
            if(c == CLOptions::issaByMODEL && 
               CLOptions::GetI_TER() != CLOptions::istEOF) {
                error(std::string("Option --split=" + 
                    std::to_string(CLOptions::issaByMODEL) + 
                    " is only compatible with option --ter=0.").c_str());
                return EXIT_FAILURE;
            }
            if(c == CLOptions::issaByChain && (
               CLOptions::GetI_TER() >= CLOptions::istENDorChain)) {
                error(std::string("Option --split=" + 
                    std::to_string(CLOptions::issaByChain) + 
                    " is only compatible with option --ter=0|1.").c_str());
                return EXIT_FAILURE;
            }
            CLOPTASSIGN(I_SPLIT, c);
        }

        if( !superp.empty()) {
            if( mystring2int(c, split, "Invalid argument of option --superp."))
                return EXIT_FAILURE;
            if(c >= CLOptions::iaaSeqDependentByRes && (
               !initalnfile.empty() || !permalnfile.empty())) {
                error(std::string("Option --superp >= " + 
                    std::to_string(CLOptions::iaaSeqDependentByRes) + 
                    " is incompatible with options -i and -I.").c_str());
                return EXIT_FAILURE;
            }
            if(c >= CLOptions::iaaSeqDependentByResAndChain && (
               CLOptions::GetI_TER() >= CLOptions::istENDorChain)) {
                error(std::string("Option --superp >= " + 
                    std::to_string(CLOptions::iaaSeqDependentByResAndChain) + 
                    " is only compatible with option --ter=0|1.").c_str());
                return EXIT_FAILURE;
            }
            CLOPTASSIGN(I_SUPERP, c);
        }


        if( !preseqsim.empty()) {
            if( mystring2float(f, preseqsim, "Invalid argument of option --pre-seq-sim."))
                return EXIT_FAILURE;
            CLOPTASSIGN(P_PRE_SIMILARITY, f);
        }

        if( !prescore.empty()) {
            if( mystring2float(f, prescore, "Invalid argument of option --pre-score."))
                return EXIT_FAILURE;
            CLOPTASSIGN(P_PRE_SCORE, f);
        }


        if( !initalnfile.empty() && !permalnfile.empty()) {
            error("One of the options -i and -I can only be specified.");
            return EXIT_FAILURE;
        }

        if( !initalnfile.empty())
            CLOPTASSIGN(C_Il, initalnfile);

        if( !permalnfile.empty())
            CLOPTASSIGN(C_Iu, permalnfile);

        if( !d0str.empty()) {
            if( mystring2float(f, d0str, "Invalid argument of option --d0."))
                return EXIT_FAILURE;
            CLOPTASSIGN(C_D0, f);
        }

        if( !normlength.empty()) {
            if( mystring2int(c, normlength, "Invalid argument of option -u."))
                return EXIT_FAILURE;
            CLOPTASSIGN(C_U, c);
        }

        if( !normcode.empty()) {
            if( mystring2int(c, normcode, "Invalid argument of option -a."))
                return EXIT_FAILURE;
            CLOPTASSIGN(C_A, c);
        }

        if( !refinement.empty()) {
            if( mystring2int(c, refinement, "Invalid argument of option --refinement."))
                return EXIT_FAILURE;
            CLOPTASSIGN(C_REFINEMENT, c+1);
        }

        if( !depth.empty()) {
            if( mystring2int(c, depth, "Invalid argument of option --depth."))
                return EXIT_FAILURE;
            CLOPTASSIGN(C_DEPTH, c);
        }

        if( !trigger.empty()) {
            if( mystring2int(c, trigger, "Invalid argument of option --trigger."))
                return EXIT_FAILURE;
            CLOPTASSIGN(C_TRIGGER, c);
        }

        if( !nbranches.empty()) {
            if( mystring2int(c, nbranches, "Invalid argument of option --nbranches."))
                return EXIT_FAILURE;
            CLOPTASSIGN(C_NBRANCHES, c);
        }

        if( !convergence.empty()) {
            if( mystring2int(c, convergence, "Invalid argument of option --convergence."))
                return EXIT_FAILURE;
            CLOPTASSIGN(C_CONVERGENCE, c);
        }

        if( cperm && (!initalnfile.empty() || !permalnfile.empty())) {
            error("Option --cp is incompatible with options -i and -I.");
            return EXIT_FAILURE;
        }

        CLOPTASSIGN(C_SYMMETRIC, symmetric);
        CLOPTASSIGN(C_ADDSEARCHBYSS, addsearchbyss);
        CLOPTASSIGN(C_NODETAILEDSEARCH, nodetailedsearch);

        if( !speed.empty()) {
            if( mystring2int(c, speed, "Invalid argument of option --speed."))
                return EXIT_FAILURE;
            CLOPTASSIGN(C_SPEED, c);
            const int coarseref = CLOptions::csrCoarseSearch;
            int valrefinement = CLOptions::GetC_REFINEMENT();
            int valdepth = CLOptions::GetC_DEPTH();
            int valtrigger = CLOptions::GetC_TRIGGER();
            int valnbranches = CLOptions::GetC_NBRANCHES();
            int valconvergence = CLOptions::GetC_CONVERGENCE();
            // nodetailedsearch = 0;
            switch(c) {
                case  0: valdepth = 0; valtrigger = 0; valnbranches = 16; addsearchbyss = 1; break;
                case  1: valdepth = 0; valtrigger = 0; break;
                case  2: valdepth = 0; valtrigger = 20; break;
                case  3: valdepth = 0; valtrigger = 50; break;
                case  4: valdepth = 1; valtrigger = 0; break;
                case  5: valdepth = 1; valtrigger = 20; break;
                case  6: valdepth = 1; valtrigger = 50; break;
                case  7: valdepth = 2; valtrigger = 0; break;
                case  8: valdepth = 2; valtrigger = 20; break;
                case  9: valdepth = 2; valtrigger = 50; break;
                case 10: valdepth = 3; valtrigger = 0; valrefinement = coarseref; valconvergence = 2; break;
                case 11: valdepth = 3; valtrigger = 20; valrefinement = coarseref; valconvergence = 2; break;
                case 12: valdepth = 3; valtrigger = 50; valrefinement = coarseref; valconvergence = 2; break;
                case 13: nodetailedsearch = 1; valrefinement = coarseref; valconvergence = 2; break;
            };
            CLOPTASSIGN(C_REFINEMENT, valrefinement);
            CLOPTASSIGN(C_DEPTH, valdepth);
            CLOPTASSIGN(C_TRIGGER, valtrigger);
            CLOPTASSIGN(C_NBRANCHES, valnbranches);
            CLOPTASSIGN(C_CONVERGENCE, valconvergence);
            CLOPTASSIGN(C_ADDSEARCHBYSS, addsearchbyss);
            CLOPTASSIGN(C_NODETAILEDSEARCH, nodetailedsearch);
        }

        CLOPTASSIGN(C_CP, cperm);
        CLOPTASSIGN(C_MIRROR, mirror);


        if(!devmaxnqrs.empty()) {
            if(mystring2int(c, devmaxnqrs, "Invalid argument of option --dev-queries-per-chunk."))
                return EXIT_FAILURE;
            CLOPTASSIGN(DEV_QRS_PER_CHUNK, c);
        }

        if(!devmaxqres.empty()) {
            if(mystring2int(c, devmaxqres, "Invalid argument of option --dev-queries-total-length-per-chunk."))
                return EXIT_FAILURE;
            CLOPTASSIGN(DEV_QRES_PER_CHUNK, c);
        }

        if(!devmaxrlen.empty()) {
            if(mystring2int(c, devmaxrlen, "Invalid argument of option --dev-max-length."))
                return EXIT_FAILURE;
            CLOPTASSIGN(DEV_MAXRLEN, c);
        }

        if(!devminrlen.empty()) {
            if(mystring2int(c, devminrlen, "Invalid argument of option --dev-min-length."))
                return EXIT_FAILURE;
            int valrmaxlen = CLOptions::GetDEV_MAXRLEN();
            if(valrmaxlen < c) {
                error("Invalid option values: --dev-min-length > --dev-max-length.");
                return EXIT_FAILURE;
            }
            CLOPTASSIGN(DEV_MINRLEN, c);
        }

        if(!clsdblst.empty()) {
            int maxqrslen = CLOptions::GetDEV_QRES_PER_CHUNK();
            int maxrfnlen = CLOptions::GetDEV_MAXRLEN();
            if(maxqrslen != maxrfnlen) {
                error("--dev-queries-total-length-per-chunk and --dev-max-length should be equal for clustering.");
                return EXIT_FAILURE;
            }
        }

        CLOPTASSIGN(NOFILESORT, nofilesort);


        if( !devmaxN.empty())
            CLOPTASSIGN(DEV_N, devmaxN);

        if( !devmaxmem.empty()) {
            if( mystring2int(c, devmaxmem, "Invalid argument of option --dev-mem."))
                return EXIT_FAILURE;
            CLOPTASSIGN(DEV_MEM, c);
        }

        if( !expct_prolen.empty()) {
            if( mystring2int(c, expct_prolen, "Invalid argument of option --dev-expected-length."))
                return EXIT_FAILURE;
            CLOPTASSIGN(DEV_EXPCT_DBPROLEN, c);
        }

        if( !devpass2memp.empty()) {
            if( mystring2int(c, devpass2memp, "Invalid argument of option --dev-pass2memp."))
                return EXIT_FAILURE;
            CLOPTASSIGN(DEV_PASS2MEMP, c);
        }

        if( !nbuffers.empty()) {
            if( mystring2int(c, nbuffers, "Invalid argument of option --io-nbuffers."))
                return EXIT_FAILURE;
            CLOPTASSIGN(IO_NBUFFERS, c);
        }

        CLOPTASSIGN(IO_FILEMAP, filemap);
        CLOPTASSIGN(IO_UNPINNED, unpinned);

#ifdef GPUINUSE
        DEVPROPs.RegisterDevices();
#endif

    CATCH_ERROR_RETURN(;);
    //}}


    int ret = EXIT_SUCCESS;

    TRY
#ifdef GPUINUSE
        if(clsdblst.empty()) {
            JobDispatcher jdisp(qrydblst, refdblst, sfxlst, outdir.c_str(), cachedir.c_str());
            jdisp.Run();
        } else {
            JobDispatcher jdisp(clsdblst, sfxlst, outdir.c_str(), cachedir.c_str());
            jdisp.RunClust();
        }
#else
        TaskScheduler ts(qrydblst, refdblst, sfxlst, outdir.c_str(), cachedir.c_str());
        ts.Run();
#endif
    CATCH_ERROR_RETURN(;);

    if( ret == EXIT_SUCCESS )
        checkforwarnings();

    return ret;
}

// -------------------------------------------------------------------------
// SplitString: form a list of values separated by commas and written in 
// one string;
// argstring, input string of values separated by commas;
// option, option name;
// vlist, vector of split values
//
void SplitString(std::string argstring, std::string option, std::vector<std::string>& vlist)
{
    MYMSG( "Main::SplitString", 5 );
    std::string::size_type pos;
    vlist.clear();
    for( pos = argstring.find(','); !argstring.empty(); pos = argstring.find(',')) {
        std::string dbname = argstring.substr(0, pos);
        if( dbname.empty())
            throw MYRUNTIME_ERROR(
            "Invalid value (two commas?) specified by option " + option);
        vlist.push_back(std::move(dbname));
        if( pos == std::string::npos )
            break;
        argstring = argstring.substr(pos+1);
    }
    //remove duplicates
    std::sort(vlist.begin(), vlist.end());
    auto last = std::unique(vlist.begin(), vlist.end());
    vlist.erase(last, vlist.end());
}

