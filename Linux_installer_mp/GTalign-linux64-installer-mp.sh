#!/bin/bash

echo

MYHOMEDEF=${HOME}/local/gtalign_mp

read -ep "Enter GTalign install path: " -i "${MYHOMEDEF}" MYHOME

echo
echo Install path: ${MYHOME}
echo

srcdir="$(dirname $0)"

[[ -d "${srcdir}/bin" ]] || (echo "ERROR: Source directories not found!" && exit 1)

[ -f "${srcdir}/bin/gtalign" ] || (echo "ERROR: Incomplete software package: main executable missing!" && exit 1)

mkdir -p "${MYHOME}" || (echo "ERROR: Failed to create destination directory!" && exit 1)

[ -d "${MYHOME}/bin" ] || (mkdir -p "${MYHOME}/bin" || exit 1)

cp -R "${srcdir}"/bin/* "${MYHOME}/bin/" || (echo "ERROR: Failed to install the package!" && exit 1)

echo Installation complete.

exit 0

