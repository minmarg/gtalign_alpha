#!/bin/bash

echo

MYHOMEDEF=/${HOME:1}/local/gtalign_mp

read -ep "Enter GTalign install path: " -i "${MYHOMEDEF}" MYHOME

echo
echo Install path: $MYHOME
echo

if [ ! -d buildmp ]; then mkdir buildmp || exit 1; fi
cd buildmp || exit 1

cmake -DGPUINUSE=0 -DFASTMATH=1 -DCMAKE_INSTALL_PREFIX=${MYHOME} -DCMAKE_VERBOSE_MAKEFILE=ON \
    ../src/  ||  (cd ..; exit 1)

cmake --build . --config Release --target install  ||  (cd ..; exit 1)

cd ..

exit 0

