# Tools used by the test scripts; source this file.  Every variable can be set in the environment instead.
#   PGM_DATA      test inputs (tests/data)
#   PY            Python with NumPy, SciPy and ParmEd (topology conversion, comparisons)
#   OMM_PY        Python with OpenMM and the plugin's openmmpgm / pgm_prmtop modules
#   GMX_DOUBLE    gmx of a double-precision GROMACS build with the pGM patch (CPU tests)
#   GMX_MIXED     gmx of a mixed-precision CUDA GROMACS build with the pGM patch
#   AMBER_BIN     directory with pmemd.pgm and pmemd.pgm.cuda_DPFP / _SPFP (AMBER with the pGM patch)
#   PGM_GVDW_DATA the data of the GVDW paper (topology/rayl_512_v2.prmtop, inputs/lj/inpcrd.restrt), GVDW tests only
PGM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${PGM_DATA:=$PGM_ROOT/tests/data}"
: "${PY:=python3}"
: "${OMM_PY:=$PY}"
: "${GMX_DOUBLE:=gmx_d}"
: "${GMX_MIXED:=gmx}"
: "${AMBER_BIN:=${AMBERHOME:+$AMBERHOME/bin}}"
gmx_bin() { if [ "$1" = double ]; then echo "$GMX_DOUBLE"; else echo "$GMX_MIXED"; fi; }
