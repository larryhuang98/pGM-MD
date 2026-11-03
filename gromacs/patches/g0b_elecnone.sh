#!/bin/bash
# G0b: build the LJ-only ("ElecNone") nbnxm CPU kernels always, not only with an external FMM library.
# They are guarded by GMX_USE_EXT_FMM; introduce GMX_NBNXM_ELEC_NONE (always 1) for the CPU kernels.
#   g0b_elecnone.sh GROMACS_SRC_ROOT        (idempotent)
set -e
S=$1
cd $S
grep -q "GMX_NBNXM_ELEC_NONE" src/config.h.cmakein || sed -i 's|^#cmakedefine01 GMX_USE_EXT_FMM$|#cmakedefine01 GMX_USE_EXT_FMM\n/* pGM: LJ-only nbnxm CPU kernels (electrostatics elsewhere: FMM or src/gromacs/pgm) are always built */\n#define GMX_NBNXM_ELEC_NONE 1|' src/config.h.cmakein
for f in src/gromacs/nbnxm/kernels_reference/kernel_ref_1x1.cpp src/gromacs/nbnxm/kernels_reference/kernel_ref_1x1.h \
         src/gromacs/nbnxm/kernels_reference/kernel_ref_4x4.cpp src/gromacs/nbnxm/kernels_reference/kernel_ref_4x4.h \
         src/gromacs/nbnxm/kernels_reference/kernel_ref_outer.h src/gromacs/nbnxm/simd_kernel.h \
         src/gromacs/nbnxm/kernels_simd_2xmm/kernels.h src/gromacs/nbnxm/kernels_simd_4xm/kernels.h \
         src/gromacs/nbnxm/kernel_file_generator/make_verlet_simd_kernel_files.py; do
    sed -i 's/GMX_USE_EXT_FMM/GMX_NBNXM_ELEC_NONE/g' $f
done
for f in src/gromacs/nbnxm/kernels_simd_2xmm/CMakeLists.txt src/gromacs/nbnxm/kernels_simd_4xm/CMakeLists.txt; do
    sed -i 's/^    if (GMX_USE_EXT_FMM)$/    if (TRUE) # pGM: LJ-only kernels always (was GMX_USE_EXT_FMM)/' $f
done
grep -c "GMX_NBNXM_ELEC_NONE" src/config.h.cmakein src/gromacs/nbnxm/simd_kernel.h
grep -n "if (TRUE)" src/gromacs/nbnxm/kernels_simd_*/CMakeLists.txt
