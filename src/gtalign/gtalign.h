/***************************************************************************
 *   Copyright (C) 2021-2023 by Mindaugas Margelevicius                    *
 *   Institute of Biotechnology, Vilnius University                        *
 ***************************************************************************/

#ifndef __gtalign_h__
#define __gtalign_h__

static const char*  version = "0.14.00";
static const char*  verdate = "";

static const char*  instructs = "\n\
<> []\n\
\n\
GTalign, HPC protein structure alignment, superposition and search tool.\n\
(C)2021-2023 Mindaugas Margelevicius, Institute of Biotechnology, Vilnius University\n\
\n\
\n\
Usage (one of the two):\n\
<> --qrs=(<structs>|<dirs>|<archs>) --rfs=(<structs>|<dirs>|<archs>) -o <out_dir> [<options>]\n\
<> --cls=(<structs>|<dirs>|<archs>) -o <out_dir> [<options>]   *GPU version*\n\
\n\
Basic options:\n\
--qrs=(<structs>,<dirs>,<archs>)\n\
                            Comma-separated list of structure files (PDB,\n\
                            PDBx/mmCIF, and gzip), tar archives (of structure\n\
                            files gzipped or not) and/or directories of\n\
                            query structures. If a directory is specified,\n\
                            subdirectories up to 3 levels deep will be\n\
                            searched for structures.\n\
--rfs=(<structs>,<dirs>,<archs>)\n\
                            Comma-separated list of structure files (PDB,\n\
                            PDBx/mmCIF, and gzip), tar archives and/or\n\
                            directories of reference structures (to align\n\
                            queries with). For directories, subdirectories\n\
                            up to 3 levels deep will be searched.\n\
                            RECOMMENDED: -c <dir> when --speed > 9.\n\
--sfx=<file_extension_list> Comma-separated list of extensions of structures\n\
                            to be searched for in the directories/archives\n\
                            specified by --qrs and --rfs (or --cls).\n\
                            By default, all extensions are valid.\n\
-o <output_directory>       Directory of output files for each query or\n\
                            cluster.\n\
-c <cache_directory>        Directory for cached data, which can provide a\n\
                            considerable speedup for multiple queries or\n\
                            clustering and can be reused later (for same\n\
                            --rfs or --cls). By default, not used.\n\
\n\
Clustering options:\n\
--cls=(<structs>,<dirs>,<archs>)\n\
                            Comma-separated list of structure files (PDB,\n\
                            PDBx/mmCIF, and gzip), tar archives (of files\n\
                            gzipped and not) and directories (see --qrs) of\n\
                            structures to be clustered.\n\
                            NOTE: The clustering criterion defined by --sort.\n\
                            RECOMMENDED: --speed=13 for large datasets.\n\
                            RECOMMENDED: -c <dir> when --speed > 9.\n\
--cls-threshold=<threshold> TM-score (equal or greater) or RMSD (equal or\n\
                            less) threshold for a pair to be considered\n\
                            part of the same cluster.\n\
                        Default=0.5\n\
--cls-coverage=<fraction>   Length coverage threshold (0,1].\n\
                        Default=0.7\n\
--cls-one-sided-coverage    Apply coverage threshold to one pair member.\n\
--cls-out-sequences         Output each cluster's sequences in FASTA format.\n\
--cls-algorithm=<code>      0: Complete-linkage clustering;\n\
                            1: Single-linkage clustering.\n\
                        Default=0\n\
\n\
Output control options (for search usage except --sort):\n\
-s <TMscore_threshold>      Report results down to this TM-score limit [0,1).\n\
                            0 implies all results are valid for report.\n\
                            NOTE: Also check the pre-screening options below.\n\
                        Default=0.5\n\
--sort=<code>               0: Sort results by the greater TM-score of the two;\n\
                            1: Sort by reference length-normalized TM-score;\n\
                            2: Sort by query length-normalized TM-score;\n\
                            3: Sort by RMSD.\n\
                        Default=0\n\
--nhits=<count>             Number of highest-scoring structures to list in\n\
                            the results for each query.\n\
                        Default=2000\n\
--nalns=<count>             Number of highest-scoring structure alignments\n\
                            and superpositions to output for each query.\n\
                        Default=2000\n\
--wrap=<width>              Wrap produced alignments to this width [40,).\n\
                        Default=80\n\
--no-deletions              Remove deletion positions (gaps in query) from\n\
                            produced alignments.\n\
--referenced                Produce transformation matrices for reference\n\
                            structures instead of query(-ies).\n\
\n\
Interpretation options:\n\
--infmt=<code>              Format of input structures:\n\
                            0: PDB or PDBx/mmCIF, detected automatically;\n\
                            1: PDB;\n\
                            2: PDBx/mmCIF.\n\
                        Default=0\n\
--atom=<atom_name>          4-character atom name (with spaces) to represent a\n\
                            residue (base); e.g., \" C3'\" (RNAs).\n\
                        Default=\" CA \" (proteins)\n\
--hetatm                    Consider and align both ATOM and HETATM residues.\n\
--ter=<code>                The end of a structure (chain) in a file is\n\
                            designated by\n\
                            0: end of file;\n\
                            1: ENDMDL or END;\n\
                            2: ENDMDL, END, or different chain ID;\n\
                            3: TER, ENDMDL, END or different chain ID.\n\
                        Default=3\n\
--split=<code>              Use the following interpretation of a PDB\n\
                            structure file:\n\
                            0: the entire structure is one single chain;\n\
                            1: each MODEL is a separate chain (--ter=0);\n\
                            2: each chain is a seperate chain (--ter=0|1).\n\
                        Default=0\n\
\n\
Similarity pre-screening options:\n\
--pre-similarity=<similarity_threshold>\n\
                            Minimum pairwise sequence similarity score [0,)\n\
                            for conducting structure comparison. Values >=10\n\
                            start taking considerable effect on speed.\n\
                            0, all pairs are subject to further processing.\n\
                        Default=0.0\n\
--pre-score=<TMscore_threshold>\n\
                            Minimum provisional TM-score [0,1) for structure\n\
                            pairs to proceed to further stages.\n\
                            0, all pairs are subject to further processing.\n\
                        Default=0.3\n\
\n\
Per-pair computation options:\n\
--symmetric                 Always produce symmetric alignments for the same\n\
                            query-reference (reference-query) pair.\n\
--refinement=<code>         Superposition refinement detail level [0,3].\n\
                            Difference between 3 and 1 in TM-score is ~0.001.\n\
                        Default=1\n\
--depth=<code>              Superposition search depth:\n\
                            0: deep; 1: high; 2: medium; 3: shallow.\n\
                        Default=2\n\
--trigger=<percentage>      Threshold for estimated fragment similarity in\n\
                            percent [0,100] to trigger superposition\n\
                            analysis for a certain configuration\n\
                            (0, unconditional analysis).\n\
                        Default=50\n\
--nbranches=<number>        Number [3,16] of independent top-performing\n\
                            branches identified during superposition search to\n\
                            explore in more detail.\n\
                        Default=5\n\
--add-search-by-ss          Include superposition search by a combination of\n\
                            secondary structure and sequence similarity, which\n\
                            helps optimization for some pairs.\n\
--no-detailed-search        Skip detailed (most computationally demanding)\n\
                            superposition search. Options --depth, --trigger,\n\
                            and --nbranches then have no effect.\n\
--convergence=<number>      Number of final convergence tests [1,30].\n\
                        Default=18\n\
--speed=<code>              Speed up the GTalign alignment algorithm at the\n\
                            expense of optimality (larger values => faster;\n\
                            NOTE: the pre-screening options are not affected;\n\
                            NOTE: settings override specified options):\n\
                             0: --depth=0 --trigger=0 --nbranches=16 --add-search-by-ss\n\
                             1: --depth=0 --trigger=0\n\
                             2: --depth=0 --trigger=20\n\
                             3: --depth=0 --trigger=50\n\
                             4: --depth=1 --trigger=0\n\
                             5: --depth=1 --trigger=20\n\
                             6: --depth=1 --trigger=50\n\
                             7: --depth=2 --trigger=0\n\
                             8: --depth=2 --trigger=20\n\
                             9: --depth=2 --trigger=50\n\
                            10: --depth=3 --trigger=0 --refinement=0 --convergence=2\n\
                            11: --depth=3 --trigger=20 --refinement=0 --convergence=2\n\
                            12: --depth=3 --trigger=50 --refinement=0 --convergence=2\n\
                            13: --no-detailed-search --refinement=0 --convergence=2\n\
                        Default=9\n\
\n\
HPC options:\n\
--cpu-threads-reading=<count>\n\
                            Number of CPU threads [1,64] for reading\n\
                            reference data. NOTE that computation on GPU can\n\
                            be faster than reading data by 1 CPU thread.\n\
                        Default=10\n\
--cpu-threads=<count>       Number of CPU threads [1,1024] for parallel\n\
                            computation when compiled without support for\n\
                            GPUs.\n\
                            NOTE: Default number is shown using --dev-list.\n\
                        Default=[MAX(1, #cpu_cores - <cpu-threads-reading>)]\n\
--dev-queries-per-chunk=<count>\n\
                            Maximum number [1,100] of queries processed\n\
                            as one chunk in parallel. Large values can lead\n\
                            to better performance for numerous queries but\n\
                            require more memory. Use smaller values for\n\
                            scarce device memory and/or big queries.\n\
                        Default=2\n\
--dev-queries-total-length-per-chunk=<length>\n\
                            Maximum total length [100,50000] of\n\
                            queries processed as one chunk in parallel.\n\
                            Queries of length larger than the specified\n\
                            length will be skipped. Use large values if\n\
                            required and memory limits permit since they\n\
                            greatly reduce #structure pairs processed in\n\
                            parallel.\n\
                        Default=4000\n\
--dev-max-length=<length>   Maximum length [100,65535] for reference\n\
                            structures. References of length larger than this\n\
                            specified value will be skipped.\n\
                            NOTE: Large values greatly reduce #structure pairs\n\
                            processed in parallel.\n\
                        Default=4000\n\
--dev-min-length=<length>   Minimum length [3,32767] for reference structures.\n\
                            References shorter than this specified value will\n\
                            be skipped.\n\
                        Default=20\n\
--no-file-sort              Do not sort files by size. Data locality can be\n\
                            beneficial when reading files lasts longer than\n\
                            computation.\n\
\n\
Device options:\n\
--dev-N=(<number>|,<id_list>)\n\
                            Maximum number of GPUs to use. This can be\n\
                            specified by a number or given by a comma-separated\n\
                            list of GPU identifiers, which should start with a\n\
                            comma. In the latter case, work is distributed in\n\
                            the specified order. Otherwise, more powerful GPUs\n\
                            are selected first.\n\
                            NOTE: The first symbol preceding a list is a comma.\n\
                            NOTE: The option has no effect for the version\n\
                            compiled without support for GPUs.\n\
                        Default=1 (most powerful GPU)\n\
--dev-mem=<megabytes>       Maximum amount of GPU memory (MB) that can be used.\n\
                            All memory is used if a GPU has less than the\n\
                            specified amount of memory.\n\
                        Default=[all memory of GPU(s)] (with support for GPUs)\n\
                        Default=16384 (without support for GPUs)\n\
--dev-expected-length=<length>\n\
                            Expected length of database proteins. Its values\n\
                            are restricted to the interval [20,200].\n\
                            NOTE: Increasing it reduces memory requirements,\n\
                            but mispredictions may cost additional computation\n\
                            time.\n\
                        Default=50\n\
--io-nbuffers=<count>       Number of buffers [2,6] used to cache data read\n\
                            from file. Values greater than 1 lead to increased\n\
                            performance at the expense of increased memory\n\
                            consumption.\n\
                        Default=3\n\
--io-unpinned               Do not use pinned (page-locked) CPU memory.\n\
                            Pinned CPU memory provides better performance, but\n\
                            reduces system resources. If RAM memory is scarce\n\
                            (<2GB), using pinned memory may reduce overall\n\
                            system performance.\n\
                            By default, pinned memory is used.\n\
\n\
Other options:\n\
--dev-list                  List all GPUs compatible and available on the\n\
                            system, print a default number for option\n\
                            --cpu-threads (for the CPU version), and exit.\n\
-v [<level_number>]         Verbose mode.\n\
-h                          This text.\n\
\n\
\n\
Examples:\n\
<> -v --qrs=str1.cif.gz --rfs=my_huge_structure_database.tar -o my_output_directory\n\
<> -v --qrs=struct1.pdb --rfs=struct2.pdb,struct3.pdb,struct4.pdb -o my_output_directory\n\
<> -v --qrs=struct1.pdb,my_struct_directory --rfs=my_ref_directory -o my_output_directory\n\
<> -v --qrs=str1.pdb.gz,str2.cif.gz --rfs=archive.tar,my_ref_dir -s 0 -o mydir\n\
<> -v --cls=my_huge_structure_database.tar -o my_output_directory\n\
\n\
";

// <> --a2a=(<structs>|<dirs>|<DBs>) -o <out_dir> [<options>]\n
// --a2a=(<structs>,<dirs>,<DBs>) Directory or database (constructed by maketmdb)\n
//                             of structures to be aligned all-against-all. This is\n
//                             more efficient than using --qrs and --rfs since the\n
//                             same pair of structures will be compared once.\n
// --outfmt=<code>             Output format:\n
//                             0: full output with structural superpositions;\n
//                               View superposed aligned regions in RasMol or PyMOL:\n
//                                 rasmol -script <filename>_TM_sup\n
//                                 pymol -d @<filename>_TM_sup.pml\n
//                               View superposed C-alpha traces of all regions:\n
//                                 rasmol -script <filename>_TM_sup_all\n
//                                 pymol -d @<filename>_TM_sup_all.pml\n
//                               View full-atom superposition of aligned regions:\n
//                                 rasmol -script <filename>_TM_sup_atm\n
//                                 pymol -d @<filename>_TM_sup_atm.pml\n
//                               View full-atom superposition of all regions:\n
//                                 rasmol -script <filename>_TM_sup_all_atm\n
//                                 pymol -d @<filename>_TM_sup_all_atm.pml\n
//                               View full-atom superposition with ligands:\n
//                                 rasmol -script <filename>_TM_sup_all_atm_lig\n
//                                 pymol -d @<filename>_TM_sup_all_atm_lig.pml\n
//                             1: alignments and rotation matrices only;\n
//                             2: compact tabular output format.\n
//                         Default=1\n
// --superp=<code>             Alignment algorithm:\n
//                             0: sequence-independent alignment (GTalign);\n
//                             1: sequence-dependent superposition (TMscore),\n
//                                i.e., alignment by residue index;\n
//                             2: sequence-dependent superposition (TMscore -c);\n
//                                alignment by residue index and chain ID;\n
//                                (--ter=0|1);\n
//                             3: sequence-dependent superposition (TMscore -c);\n
//                                alignment by residue index and chain order;\n
//                                (--ter=0|1).\n
//                         Default=0\n
// -i <filename>               Align a pair of structures starting with an\n
//                             alignment specified in this file in FASTA format.\n
//                             NOTE: Valid for one pair given by --qrs and --rfs.\n
// -I <filename>               Transform (rotate and translate) the query structure\n
//                             given its alignment with the reference structure in\n
//                             this file in FASTA format.\n
//                             NOTE: Valid for one pair given by --qrs and --rfs.\n
// --d0=<Angstroms>            Additionally, use this value (<50) of the normalizing\n
//                             inter-residue distance d0 in the TM-score formula.\n
// -u <normalization_length>   Normalize TM-score also by this length.\n
//                             NOTE: TM-score > 1 if the specified value < the\n
//                             minimum length of the structures being aligned.\n
// -a <code>                   When thresholding on TM-score (-s), normalize it by\n
//                             0: length of the reference (second) structure;\n
//                             1: average length of the structures being aligned;\n
//                             2: length of the shorter structure;\n
//                             3: length of the longer structure.\n
//                         Default=0\n
// --cp                        Alignment with circular permutation.\n
// --mirror                    Align the mirror image of the query structure(s).\n
// 
//                             NOTE: For a small number of queries, using a moderate\n
//                             amount of memory (~4GB) is more efficient.\n
// 
// --dev-pass2memp=<percentage> GPU memory proportion dedicated to recalculation of\n
//                             hits that passed significance threshold.\n
//                             (Expected proportion of significant hits.)\n
//                         Default=10\n
// --io-filemap                Map files into memory.\n
//                             In general, the performance with or without file\n
//                             mapping is similar. In some systems, however,\n
//                             mapping can lead to increased computation time.\n

#endif//__gtalign_h__
