#!/bin/bash

echo

MYHOMEDEF=${HOME}/local/gtalign

read -ep "Enter GTalign install path: " -i "${MYHOMEDEF}" MYHOME

echo
echo Install path: ${MYHOME}
echo

srcdir="$(dirname $0)"

[[ -d "${srcdir}/bin" ]] || (echo "ERROR: Source directories not found!" && exit 1)

[ -f "${srcdir}/bin/gtalign" ] || (echo "ERROR: Incomplete software package: main executable missing!" && exit 1)

mkdir -p "${MYHOME}" || (echo "ERROR: Failed to create destination directory!" && exit 1)

[ -d "${MYHOME}/bin" ] || (mkdir -p "${MYHOME}/bin" || exit 1)

cmd="${srcdir}/bin/gtalign --dev-list >/dev/null 2>&1"
eval ${cmd} || cat <<EOF

WARNING: The gtalign executable will not run on the system!
WARNING: Please make sure a GPU and appropriate NVIDIA and 
WARNING: CUDA drivers are installed. 
WARNING: If they are installed, please compile and install
WARNING: the software from the source code by typing:
WARNING: BUILD_and_INSTALL__GPU__unix.sh (gcc compiler) or
WARNING: BUILD_and_INSTALL__GPU__unix__clang.sh (clang compiler).

EOF

cp -R "${srcdir}"/bin/* "${MYHOME}/bin/" || (echo "ERROR: Failed to install the package!" && exit 1)

echo Installation complete.

exit 0

