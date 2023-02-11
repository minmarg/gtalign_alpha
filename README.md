[![Release](https://img.shields.io/github/v/release/minmarg/gtalign_alpha)](https://github.com/minmarg/gtalign_alpha/releases)
![Header image](imgs/gtalign_header.jpg)
*3BV8\_A and 1D8J\_A (colored; left and right, respectively) superimposed on 
7CZ9\_B (grey). Green, GTalign superposition; red, TM-align superposition. 
The GTalign output file for this example is available 
[Here](out/2021-09-18_00000039_1__0.md).*

# GTalign (alpha release)

GTalign, a novel high-performance computing (HPC) protein structure alignment, 
superposition and search method.

## Features

  *  Graphics processing unit (GPU) version
  *  Configurable GPU memory
  *  Tested on NVIDIA Pascal (GeForce GTX 1080MQ), Turing (GeForce RTX 2080Ti), 
  Volta (V100), and Ampere (A100) GPU architectures
  *  Same executable for different architectures
  *  ~1000x faster on a single GPU (Volta) than TM-align (tested with 186 queries and 
  18,861 database 
  entries)
  *  Alignment of complexes up to 65,535 residues long; the alignment of 37,860 
  residue-long complexes (7A4I and 7A4J) is ~900,000x faster (Volta) than TM-align
  *  Running on Ampere is 2x faster than on Volta
  *  Similar accuracy (optimality) to TM-align
  *  Correct TM-scores are guaranteed for produced superpositions
  *  Correct RMSDs are guaranteed for produced alignments
  *  Support for PDB, PDBx/mmCIF, and gzip (thanks to [zlib](https://github.com/madler/zlib))
  formats
  *  Directories for search can be specified; subdirectories up to 3 levels deep are 
  then searched

## Features to come

  *  Further improvements in speed and accuracy (superposition optimality)
  *  Fast prescreening for similarities in structure and sequence space, leading to a 
  further 10-100-fold speedup for database searches
  *  Reading multiple compressed structures from gzip archives 
  *  Utilization of multiple GPUs
  *  CPU version
  *  Cross-platform support

  The source code will be published later.

## Available Platforms

  The GTalign executable is provided for the following platform:

  *  Linux x64

## System requirements

  *  CUDA-enabled GPU(s) with compute capability >=3.5 (released in 2012)
  *  NVIDIA driver version >=418.87 and CUDA version >=10.1

## Installation

  Download the executable `bin/gtalign`.

## Getting started

  Type `bin/gtalign` for a description of the options. 
  Query structures and/or directories with queries are specified with the option `--qrs`.
  Reference structures (to align queries with) and/or their directories to be 
  searched are specified with the option `--rfs`.

  Here are some examples:

`bin/gtalign --qrs=struct1.pdb --rfs=struct2.pdb,struct3.pdb,struct4.pdb -o my_output_directory`

`bin/gtalign --qrs=struct1.pdb,my_struct_directory --rfs=my_ref_directory -o my_output_directory`

`bin/gtalign --qrs=str1.pdb.gz,str2.cif.gz --rfs=str3.cif.gz,str4.ent,my_ref_dir -s 0 -o mydir`

  Queries and references are processed in chunks.
  The maximum total length of queries in one chunk is controlled with the option 
  `--dev-queries-total-length-per-chunk`. 
  The maximum length of a reference structure can be specified with the option 
  `--dev-max-length`.
  Larger structures will be skipped during a search.
  A good practice is to keep `--dev-max-length` reasonably large (e.g., <10000; unless your 
  set of references are all larger) so that many structure pairs are processed in parallel.

  For comparing protein complexes, it usually suffices to set `--ter=0`.

## Citation

If you use this software, please cite:

```bibtex
@software{Margelevicius_GTalign_alpha_2023,
  author = {Margelevicius, Mindaugas},
  month = {1},
  title = {{GTalign, HPC protein structure alignment, superposition and search (alpha release)}},
  url = {https://github.com/minmarg/gtalign_alpha},
  version = {0.1.0},
  year = {2023}
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

