```

gtalign 0.10.00 (compiled with GPU support)

GTalign, HPC protein structure alignment, superposition and search method.
(C)2021-2023 Mindaugas Margelevicius, Institute of Biotechnology, Vilnius University


Usage (one of the two):
gtalign --qrs=(<structs>|<dirs>|<archs>) --rfs=(<structs>|<dirs>|<archs>) -o <out_dir> [<options>]
gtalign --cls=(<structs>|<dirs>|<archs>) -o <out_dir> [<options>]   *GPU version*

Basic options:
--qrs=(<structs>,<dirs>,<archs>)
                            Comma-separated list of structure files (PDB,
                            PDBx/mmCIF, and gzip), tar archives (of structure
                            files gzipped or not) and/or directories of
                            query structures. If a directory is specified,
                            subdirectories up to 3 levels deep will be
                            searched for structures.
--rfs=(<structs>,<dirs>,<archs>)
                            Comma-separated list of structure files (PDB,
                            PDBx/mmCIF, and gzip), tar archives and/or
                            directories of reference structures (to align
                            queries with). If a directory is specified,
                            subdirectories up to 3 levels deep will be
                            searched.
--sfx=<file_extension_list> Comma-separated list of extensions of structures
                            to be searched for in the directories/archives
                            specified by --qrs and --rfs (or --cls).
                            By default, all extensions are valid.
-o <output_directory>       Directory of output files for each query or
                            cluster.

Clustering options:
--cls=(<structs>,<dirs>,<archs>)
                            Comma-separated list of structure files (PDB,
                            PDBx/mmCIF, and gzip), tar archives (of files
                            gzipped and not) and directories (see --qrs) of
                            structures to be clustered.
                            NOTE: The clustering criterion defined by --sort.
                            RECOMMENDED: --speed=13 for large datasets.
--cls-threshold=<threshold> TM-score (equal or greater) or RMSD (equal or
                            less) threshold for a pair to be considered
                            part of the same cluster.
                        Default=0.5
--cls-coverage=<fraction>   Length coverage threshold (0,1].
                        Default=0.7
--cls-one-sided-coverage    Apply coverage threshold to one pair member.
--cls-out-sequences         Output each cluster's sequences in FASTA format.

Output control options (for search usage except --sort):
-s <TMscore_threshold>      Report results down to this TM-score limit [0,1).
                            0 implies all results are valid for report.
                            NOTE: Also check the pre-screening options below.
                        Default=0.5
--sort=<code>               0: Sort results by the greater TM-score of the two;
                            1: Sort by reference length-normalized TM-score;
                            2: Sort by query length-normalized TM-score;
                            3: Sort by RMSD.
                        Default=0
--nhits=<count>             Number of highest-scoring structures to list in
                            the results for each query.
                        Default=2000
--nalns=<count>             Number of highest-scoring structure alignments
                            and superpositions to output for each query.
                        Default=2000
--wrap=<width>              Wrap produced alignments to this width [40,).
                        Default=80
--no-deletions              Remove deletion positions (gaps in query) from
                            produced alignments.
--referenced                Produce transformation matrices for reference
                            structures instead of query(-ies).

Interpretation options:
--infmt=<code>              Format of input structures:
                            0: PDB or PDBx/mmCIF, detected automatically;
                            1: PDB;
                            2: PDBx/mmCIF.
                        Default=0
--atom=<atom_name>          4-character atom name (with spaces) to represent a
                            residue (base); e.g., " C3'" (RNAs).
                        Default=" CA " (proteins)
--hetatm                    Consider and align both ATOM and HETATM residues.
--ter=<code>                The end of a structure (chain) in a file is
                            designated by
                            0: end of file;
                            1: ENDMDL or END;
                            2: ENDMDL, END, or different chain ID;
                            3: TER, ENDMDL, END or different chain ID.
                        Default=3
--split=<code>              Use the following interpretation of a PDB
                            structure file:
                            0: the entire structure is one single chain;
                            1: each MODEL is a separate chain (--ter=0);
                            2: each chain is a seperate chain (--ter=0|1).
                        Default=0

Similarity pre-screening options:
--pre-similarity=<similarity_threshold>
                            Minimum pairwise sequence similarity score [0,)
                            for conducting structure comparison. Values >=10
                            start taking considerable effect on speed.
                            0, all pairs are subject to further processing.
                        Default=0.0
--pre-score=<TMscore_threshold>
                            Minimum provisional TM-score [0,1) for structure
                            pairs to proceed to further stages.
                            0, all pairs are subject to further processing.
                        Default=0.3

Per-pair computation options:
--symmetric                 Always produce symmetric alignments for the same
                            query-reference (reference-query) pair.
--refinement=<code>         Superposition refinement detail level [0,3].
                            Difference between 3 and 1 in TM-score is ~0.001.
                        Default=1
--depth=<code>              Superposition search depth:
                            0: deep; 1: high; 2: medium; 3: shallow.
                        Default=2
--trigger=<percentage>      Threshold for estimated fragment similarity in
                            percent [0,100] to trigger superposition
                            analysis for a certain configuration
                            (0, unconditional analysis).
                        Default=50
--nbranches=<number>        Number [3,16] of independent top-performing
                            branches identified during superposition search to
                            explore in more detail.
                        Default=5
--add-search-by-ss          Include superposition search by a combination of
                            secondary structure and sequence similarity, which
                            helps optimization for some pairs.
--no-detailed-search        Skip detailed (most computationally demanding)
                            superposition search. Options --depth, --trigger,
                            and --nbranches then have no effect.
--convergence=<number>      Number of final convergence tests [1,30].
                        Default=18
--speed=<code>              Speed up the GTalign alignment algorithm at the
                            expense of optimality (larger values => faster;
                            NOTE: the pre-screening options are not affected;
                            NOTE: settings override specified options):
                             0: --depth=0 --trigger=0 --nbranches=16 --add-search-by-ss
                             1: --depth=0 --trigger=0
                             2: --depth=0 --trigger=20
                             3: --depth=0 --trigger=50
                             4: --depth=1 --trigger=0
                             5: --depth=1 --trigger=20
                             6: --depth=1 --trigger=50
                             7: --depth=2 --trigger=0
                             8: --depth=2 --trigger=20
                             9: --depth=2 --trigger=50
                            10: --depth=3 --trigger=0 --refinement=0 --convergence=2
                            11: --depth=3 --trigger=20 --refinement=0 --convergence=2
                            12: --depth=3 --trigger=50 --refinement=0 --convergence=2
                            13: --no-detailed-search --refinement=0 --convergence=2
                        Default=9

HPC options:
--cpu-threads-reading=<count>
                            Number of CPU threads [1,64] for reading
                            reference data. NOTE that computation on GPU can
                            be faster than reading data by 1 CPU thread.
                        Default=10
--cpu-threads=<count>       Number of CPU threads [1,1024] for parallel
                            computation when compiled without support for
                            GPUs.
                            NOTE: Default number is shown using --dev-list.
                        Default=[MAX(1, #cpu_cores - <cpu-threads-reading>)]
--dev-queries-per-chunk=<count>
                            Maximum number [1,100] of queries processed
                            as one chunk in parallel. Large values can lead
                            to better performance for numerous queries but
                            require more memory. Use smaller values for
                            scarce device memory and/or big queries.
                        Default=2
--dev-queries-total-length-per-chunk=<length>
                            Maximum total length [100,50000] of
                            queries processed as one chunk in parallel.
                            Queries of length larger than the specified
                            length will be skipped. Use large values if
                            required and memory limits permit since they
                            greatly reduce #structure pairs processed in
                            parallel.
                        Default=4000
--dev-max-length=<length>   Maximum length [100,65535] for reference
                            structures. References of length larger than this
                            specified value will be skipped.
                            NOTE: Large values greatly reduce #structure pairs
                            processed in parallel.
                        Default=4000
--dev-min-length=<length>   Minimum length [3,32767] for reference structures.
                            References shorter than this specified value will
                            be skipped.
                        Default=20
--no-file-sort              Do not sort files by size. Data locality can be
                            beneficial when reading files lasts longer than
                            computation.

Device options:
--dev-N=(<number>|,<id_list>)
                            Maximum number of GPUs to use. This can be
                            specified by a number or given by a comma-separated
                            list of GPU identifiers, which should start with a
                            comma. In the latter case, work is distributed in
                            the specified order. Otherwise, more powerful GPUs
                            are selected first.
                            NOTE: The first symbol preceding a list is a comma.
                            NOTE: The option has no effect for the version
                            compiled without support for GPUs.
                        Default=1 (most powerful GPU)
--dev-mem=<megabytes>       Maximum amount of GPU memory (MB) that can be used.
                            All memory is used if a GPU has less than the
                            specified amount of memory.
                        Default=[all memory of GPU(s)] (with support for GPUs)
                        Default=16384 (without support for GPUs)
--dev-expected-length=<length>
                            Expected length of database proteins. Its values
                            are restricted to the interval [20,200].
                            NOTE: Increasing it reduces memory requirements,
                            but mispredictions may cost additional computation
                            time.
                        Default=50
--io-nbuffers=<count>       Number of buffers [2,6] used to cache data read
                            from file. Values greater than 1 lead to increased
                            performance at the expense of increased memory
                            consumption.
                        Default=3
--io-unpinned               Do not use pinned (page-locked) CPU memory.
                            Pinned CPU memory provides better performance, but
                            reduces system resources. If RAM memory is scarce
                            (<2GB), using pinned memory may reduce overall
                            system performance.
                            By default, pinned memory is used.

Other options:
--dev-list                  List all GPUs compatible and available on the
                            system, print a default number for option
                            --cpu-threads (for the CPU version), and exit.
-v [<level_number>]         Verbose mode.
-h                          This text.


Examples:
gtalign -v --qrs=str1.cif.gz --rfs=my_huge_structure_database.tar -o my_output_directory
gtalign -v --qrs=struct1.pdb --rfs=struct2.pdb,struct3.pdb,struct4.pdb -o my_output_directory
gtalign -v --qrs=struct1.pdb,my_struct_directory --rfs=my_ref_directory -o my_output_directory
gtalign -v --qrs=str1.pdb.gz,str2.cif.gz --rfs=archive.tar,my_ref_dir -s 0 -o mydir
gtalign -v --cls=my_huge_structure_database.tar -o my_output_directory

```
