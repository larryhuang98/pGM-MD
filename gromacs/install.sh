#!/bin/bash
# Apply the pGM patch to a GROMACS 2026.4 source tree (git checkout of tag v2026.4 or the release tarball).
#   gromacs/install.sh GROMACS_SRC
# Stages (patches/): G0a topology data ([ pgm_atoms ], [ pgm_dipoles ]); G0b coulombtype = pGM, the mdp
# options and the LJ-only nbnxm kernels; G1 the pGM module src/gromacs/pgm (with core/pgm_math.h,
# core/pgm_constants.h) and its hooks in do_force; G1b input checks; G2 Gaussian van der Waals; G4 domain
# decomposition; G3 the GPU engine core/cuda/pgm_cuda.cu (CUDA builds, GMX_PGM_GPU=1 at run time).
# Every edit is an exact string replacement that must match once (patches/apply.py), so the script stops
# without a partial edit of a file if the tree is not GROMACS 2026.4, and running it twice changes nothing.
set -e
H=$(cd "$(dirname "$0")" && pwd)
S=$(cd "${1:?usage: install.sh GROMACS_SRC}" && pwd)
grep -q "project(Gromacs VERSION 2026" "$S/CMakeLists.txt" 2>/dev/null \
    || { echo "$S does not look like a GROMACS 2026 source tree (CMakeLists.txt)"; exit 1; }
P=$H/patches
python3 $P/apply.py $P/g0a_interactions.py "$S"
python3 $P/apply.py $P/g0b_coulombtype.py "$S"
bash $P/g0b_elecnone.sh "$S"
bash $P/g1_install.sh "$S"
echo "pGM patch applied to $S"
