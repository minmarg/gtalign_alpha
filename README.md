[![Release](https://img.shields.io/github/v/release/minmarg/gtalign_alpha)](https://github.com/minmarg/gtalign_alpha/releases)
[![DOI](https://zenodo.org/badge/590487846.svg)](https://zenodo.org/doi/10.5281/zenodo.10433419)
![Header image](imgs/gtalign_header.jpg)

# GTalign (alpha release)

GTalign, a novel high-performance (HPC) protein structure alignment, 
superposition and search method (with flexible structure clustering ability)

## Features

  *  CPU/multiprocessing version
  *  Graphics processing unit (GPU) version
  *  Configurable GPU memory
  *  Utilization of multiple GPUs
  *  Tested on NVIDIA Pascal (GeForce GTX 1080MQ), Turing (GeForce RTX 2080Ti, GTX 1650), 
  Volta (V100), Ampere (A100), and Ada Lovelace (GeForce RTX 4090) GPU architectures
  *  Same executable for different architectures
  *  `>`1000x faster on a single GPU (Volta) than TM-align
  *  Fast prescreening for similarities in both **sequence** and **structure** space (further n-fold speedup for database searches)
  *  Alignment of complexes up to 65,535 residues long; 7a4i and 7a4j (37,860 residues each) complexes alignment is ~900,000x faster (Volta) than TM-align
  *  Running on Ampere is 2x faster than on Volta
  *  Running on Ada Lovelace is ~1.5x faster than on Ampere
  *  More sensitive and accurate compared to TM-align when using deep superposition search
  *  Correct TM-scores are guaranteed for produced superpositions
  *  Correct RMSDs are guaranteed for produced alignments
  *  Many options for speed-accuracy tradeoff
  *  Support for PDB, PDBx/mmCIF, and gzip (thanks to [zlib](https://github.com/madler/zlib))
  formats
  *  Reading (un)compressed structures from TAR archives 
  *  Directories for search up to 3 levels deep can be specified
  *  Flexible structure **clustering** ability (GPU only)
  *  Cross-platform/portable code
  *  Opportunities for improvements in speed and accuracy (superposition optimality)

## A note on the CPU/multiprocessing version

  GTalign is optimized to run on GPUs. Its CPU/multiprocessing version is 
  based on the algorithms developed for GPUs, but the implementation details 
  differ. The CPU/multiprocessing version produces similar results, but 
  superpositions and alignments produced for some structure pairs may be 
  different. TM-scores and RMSDs are correct within numerical error.

  The CPU/multiprocessing version using 20 threads is 10-20x slower than the 
  GPU version running on a V100. The difference also depends on the options used:
  It increases with decreasing `--speed` value.

## Available Platforms

  The GTalign source code should compile and run on Linux, MS Windows, and macOS. 
  GTalign was tested on and the binaries are provided for the following platforms:

  *  Linux x64
  *  Windows 10/11 x64

  Tested compilers include GCC versions 7.5.0, 8.3.0, and 11.4.0; 
  LLVM/Clang version 10.0.0; and native MSVC compilers.

## System requirements (GPU version)

  *  CUDA-enabled GPU(s) with compute capability >=3.5 (released in 2012)
  *  NVIDIA driver version >=418.87 (>=425.25 for Win64) and CUDA version >=10.1

## System requirements (CPU/multiprocessing version)

  *  GLIBC version >=2.16 (Linux)

## Installation of pre-compiled binaries

  Download or clone the repository:

  `git clone https://github.com/minmarg/gtalign_alpha.git`

  On Linux, run the shell scripts, for the GPU and CPU versions, respectively, and 
  follow the instructions:

  `Linux_installer_GPU/GTalign-linux64-installer-GPU.sh`

  `Linux_installer_mp/GTalign-linux64-installer-mp.sh`

  On MS Windows 10/11, run the GPU-version installer:

  `MS_Windows10_installer_GPU/GTalign-win64-installer.msi`

## Installing Conda packages on Linux and macOS

  To install the multiprocessing version (CPU) of GTalign on Linux and macOS, run:

  `conda install minmarg::gtalign_mp`

  For the GPU version of GTalign, which is available as a Conda 
  package on Linux, use:

  `conda install minmarg::gtalign_gpu`

## Installation from source code

### Installation on Linux and macOS

#### Software requirements

  To build and install the GTalign software from the source code
  on Linux or macOS (CPU version), these tools are required to be installed:

  *  CMake version 3.10 or greater

  *  GNU Make version 3.82 or greater

  *  GNU GCC compiler version (7.5) or greater, or LLVM clang compiler
     version 10 or greater (or another C++ compiler that supports C++14)

  *  [the NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-downloads) version 10.0 or greater
     (required for GPU version only)

#### Installation

  Run the shell script for the GPU (Linux) and CPU versions, respectively, 
  using GCC or LLVM/Clang compilers (takes several minutes to compile):

  `BUILD_and_INSTALL__GPU__unix.sh`

  `BUILD_and_INSTALL__GPU__unix__clang.sh`

  `BUILD_and_INSTALL__mp__unix.sh`

  `BUILD_and_INSTALL__mp__unix__clang.sh`

### Installation on MS Windows

#### Software requirements

   To build and install the GTalign software from the source code
   on MS Windows, these tools are required to be installed:

  *  CMake version 3.10 or greater (free software)

  *  Visual C++ compiler, e.g., Visual Studio Community (free for open 
     source projects; GTalign is an open source project)

  *  [the NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-downloads) version 10.0 or greater 
     (free software) (required for GPU version only)

#### Installation

  Run the command (batch) file for the GPU and CPU versions, respectively:

  `BUILD_and_INSTALL__GPU__win64.cmd`

  `BUILD_and_INSTALL__mp__win64.cmd`

## Getting started

  Type `gtalign` for a description of the [options](out/gtalign_options.md). 

  Query structures and/or directories with queries are specified with the option `--qrs`.
  Reference structures (to align queries with) and/or their directories to be 
  searched are specified with the option `--rfs`.

  Note that GTalign reads `.tar` archives of compressed and uncompressed structures,
  meaning that big structure databases such as AlphaFold2 and ESM archived structural
  models are ready for use once downloaded.

  Here are some examples:

`gtalign -v --qrs=str1.cif.gz --rfs=my_huge_structure_database.tar -o my_output_directory`

`gtalign -v --qrs=struct1.pdb --rfs=struct2.pdb,struct3.pdb,struct4.pdb -o my_output_directory`

`gtalign -v --qrs=struct1.pdb,my_struct_directory --rfs=my_ref_directory -o my_output_directory`

`gtalign -v --qrs=str1.pdb.gz,str2.cif.gz --rfs=str3.cif.gz,str4.ent,my_ref_dir -s 0 -o mydir`

  Queries and references are processed in chunks.
  The maximum total length of queries in one chunk is controlled with the option 
  `--dev-queries-total-length-per-chunk`. 
  The maximum length of a reference structure can be specified with the option 
  `--dev-max-length`.
  Larger structures will be skipped during a search.
  A good practice is to keep `--dev-max-length` reasonably large (e.g., <10000; unless your 
  set of references are all larger) so that many structure pairs are processed in parallel.

  For comparing protein complexes, it usually suffices to set `--ter=0`.
  The options `--ter=0 --split=2` are used to consider all chains present in structure 
  files when executing the program.

## Alignment sorting

  GTalign offers the `--sort` option to arrange alignment based on various criteria.
  Users can choose to sort alignments by TM-score, RMSD (root-mean-squared deviation), or the 
  secondary TM-score, 2TM-score, which is calculated over the alignment while excluding 
  unmatched helices.
  Consequently, the 2TM-score penalizes topological inconsistencies more than the TM-score.

  Additionally, the `--sort` option allows for sorting by the harmonic mean of the 
  TM-scores or 2TM-scores.
  The harmonic mean is particularly effective in reducing the significance of structural 
  alignments for pairs with large length differences.
  Therefore, sorting by the harmonic mean may prove beneficial when seeking and analyzing 
  evolutionarily related or structurally similar proteins with length ratios not exceeding 
  several times.

## Clustering

  The GPU version of GTalign allows for clustering (by complete or single linkage) of large 
  protein structure datasets.
  This option is as highly configurable as the search. A simplest command line example is:

`gtalign -v --cls=my_huge_structure_database.tar -o my_output_directory`

  which instructs GTalign to cluster structures archived in `my_huge_structure_database.tar`
  with default parameters.
  The superimposed members of a cluster can then be obtained by running `gtalign` with 
  the first member as query and all others as references and using options 
  `--pre-score=0 -s 0 --referenced`, which produces transformation matrices for the reference 
  structures to be superimposed on the query.

  The clustering options, which can be used in combination with other options to make
  clustering flexible, can be found in the complete list of [options](out/gtalign_options.md).

## GTalign demo notebooks on Google Colab

The GTalign demo notebooks, [GTalign_demo](GTalign_demo.ipynb) and 
[GTalign_demo_search](GTalign_demo_search.ipynb), for Google Colab are available. 
The [first](GTalign_demo.ipynb) notebook showcases structure alignment for two large 
protein complexes -- virus nucleocapsid variants 7a4i and 7a4j -- and runs on Google 
Colab with a Tesla T4 GPU (finishes in a minute).
The [second](GTalign_demo_search.ipynb) demonstrates the alignment of all against all 
queries of the PDB20 dataset, completing in half a minute.

## Citation

If you use, reference, or benefit from the GTalign software or data, please cite:

Margelevicius, M. GTalign: spatial index-driven protein structure alignment, 
superposition, and search. Nat Commun 15, 7305 (2024). 
https://doi.org/10.1038/s41467-024-51669-z

```bibtex
@article{Margelevicius_s41467-024-51669-z,
  author = {Margelevi{\v{c}}ius, Mindaugas},
  title = {GTalign: spatial index-driven protein structure alignment, superposition, and search},
  journal = {Nature Communications},
  year = {2024},
  month = {Aug},
  day = {24},
  volume = {15},
  number = {1},
  pages = {7305},
  issn = {2041-1723},
  doi = {10.1038/s41467-024-51669-z},
  url = {https://doi.org/10.1038/s41467-024-51669-z}
}
```

## Contacts

Bug reports, comments, suggestions are welcome.
If you have other questions, please contact Mindaugas Margelevicius at
[mindaugas.margelevicius@bti.vu.lt](mailto:mindaugas.margelevicius@bti.vu.lt).

## License

Copyright 2023 Mindaugas Margelevicius, Institute of Biotechnology, Vilnius University

[Licensed](LICENSE.md) under the Apache License, Version 2.0 (the "License"); you may not 
use this file except in compliance with the License. You may obtain a copy of the 
License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the 
License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, 
either express or implied. 
See the License for the specific language governing permissions and limitations under the 
License.

## Funding

This project received funding from the Research Council of Lithuania (LMTLT; grant S-MIP-23-104).

