#!/bin/bash

echo

MYHOMEDEF=/${HOME:1}/local/gtalign_mp_clang

read -ep "Enter GTalign install path: " -i "${MYHOMEDEF}" MYHOME

echo
echo Install path: $MYHOME
echo

if [ ! -d build_mp_clang ]; then mkdir build_mp_clang || exit 1; fi
cd build_mp_clang || exit 1


CC=clang-10 CXX=clang++-10 \
cmake -DGPUINUSE=0 -DFASTMATH=1 -DCMAKE_INSTALL_PREFIX=${MYHOME} \
    ../src/  ||  (cd ..; exit 1)

cmake --build . --config Release --target install  ||  (cd ..; exit 1)

cd ..

exit 0

