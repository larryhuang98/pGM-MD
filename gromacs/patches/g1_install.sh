#!/bin/bash
# G1: install src/gromacs/pgm (pGM electrostatics module) into a GROMACS source tree, then apply g1_hooks.py.
#   g1_install.sh GROMACS_SRC_ROOT
set -e
H=$(cd "$(dirname "$0")" && pwd)
S=$1
mkdir -p $S/src/gromacs/pgm
cp $H/../src/pgm/pgm.h $H/../src/pgm/pgm.cpp $H/../src/pgm/CMakeLists.txt $S/src/gromacs/pgm/
python3 $H/apply.py $H/g1_hooks.py $S
