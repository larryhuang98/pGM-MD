#!/bin/bash
# Tests of the OpenMM plugin (see tests/env.sh for OMM_PY).
#   tests/openmm_tests.sh [PLATFORM ...]      (default: Reference CPU; add CUDA and/or OpenCL on a GPU machine)
# API and contract tests; the TailCorrectionForce; GVDWForce unit tests; PGMForce against the pGM-JAX
# reference energies, forces and dipoles of 512 pGM waters (tests/data/ref_w512_*_novdw.npz, levels: charges q,
# + permanent dipoles p, + induced dipoles i); each GPU platform against the CPU platform in every precision;
# GVDW against pmemd-pgm (needs PGM_GVDW_DATA); a short NVE run.
H=$(cd "$(dirname "$0")" && pwd)
. "$H/env.sh"
PLATS=("$@"); [ ${#PLATS[@]} -gt 0 ] || PLATS=(Reference CPU)
W=$PGM_DATA/water512
echo "== test_pgm_api (${PLATS[*]})"
$OMM_PY -u $H/test_pgm_api.py $W/pgm3p25_512.prmtop $W/pgm3p25_512.rst7 --platforms "${PLATS[@]}" 2>&1 | grep -v -i warning | tail -3
echo "== test_tail"
$OMM_PY -u $H/test_tail.py "${PLATS[@]}" 2>&1 | grep -E "FAIL|PASS|failures"
echo "== test_gvdw_api"
$OMM_PY -u $H/test_gvdw_api.py --platforms "${PLATS[@]}" 2>&1 | grep -E "FAIL|PASS|failures"
for L in qpi q qp qi; do
    echo "== OpenMM (CPU, double) vs pGM-JAX, levels $L"
    $OMM_PY $H/compare_openmm.py $W/pgm3p25_512.prmtop $PGM_DATA/ref_w512_${L}_novdw.npz --elec $L --no-lj --ke 138.935458 2>&1 \
        | grep -v -i warning | grep -v "^vdw" | tail -4
done
for P in "${PLATS[@]}"; do
    case $P in CUDA|OpenCL|HIP) ;; *) continue;; esac
    for prec in double mixed single; do
        echo "== $P $prec vs CPU"
        $OMM_PY -u $H/compare_platforms.py $W/pgm3p25_512.prmtop $PGM_DATA/ref_w512_qpi_novdw.npz --platform $P --precision $prec 2>&1 \
            | grep -v -i warning | tail -8
    done
done
if [ -n "$PGM_GVDW_DATA" ]; then
    echo "== GVDWForce vs pmemd-pgm (${PLATS[*]})"
    $OMM_PY -u $H/compare_gvdw.py $PGM_GVDW_DATA/topology/rayl_512_v2.prmtop $PGM_DATA/gvdw_box.txt --platforms "${PLATS[@]}" 2>&1 \
        | grep -v -i warning | tail -8
fi
P=${PLATS[-1]}
echo "== NVE water512, $P, 400 steps"
$OMM_PY -u $H/md_nve.py $W/pgm3p25_512.prmtop $W/pgm3p25_512.rst7 --platform $P --precision mixed --steps 400 --every 100 2>&1 \
    | grep -v -i warning | tail -3
