#!/bin/bash
# G1 + G2-GVDW: install src/gromacs/pgm (pGM electrostatics + GVDW) into a GROMACS source tree and apply the stages.
#   g1_install.sh GROMACS_SRC_ROOT
set -e
H=$(cd "$(dirname "$0")" && pwd)
S=$1
mkdir -p $S/src/gromacs/pgm
cp $H/../src/pgm/pgm.h $H/../src/pgm/pgm.cpp $H/../src/pgm/CMakeLists.txt $S/src/gromacs/pgm/
python3 $H/apply.py $H/g1_hooks.py $S
python3 $H/apply.py $H/g1b_checks.py $S
# GVDW (stage g2): the shared pair kernel next to the module, then the GROMACS changes
cp $H/../../core/gvdw_kernel.h $S/src/gromacs/pgm/
python3 $H/apply.py $H/g2_gvdw.py $S
