#!/bin/bash
# Tests of the GROMACS pGM patch (see tests/env.sh for GMX_DOUBLE, GMX_MIXED, PY, OMM_PY).
#   tests/gromacs_tests.sh [double|cuda ...]     (default: double; in an empty work directory)
# water512 single point against core/ (run_dump) and the pGM-JAX reference; GVDW single points against NumPy
# and pmemd-pgm (needs PGM_GVDW_DATA); GVDW with 1-4 pairs against OpenMM (double; needs the AMBER test
# topology qm7b_7203, see gmx/gvdw_14.sh); domain decomposition (double); NVE of water512.
H=$(cd "$(dirname "$0")" && pwd)
. "$H/env.sh"
VS=("$@"); [ ${#VS[@]} -gt 0 ] || VS=(double)
WD=$(pwd)
[ -x core_tests/run_dump ] || bash $H/core_tests.sh > /dev/null
for V in "${VS[@]}"; do
    echo "== GROMACS $V: water512 pGM single point vs core and pGM-JAX"
    mkdir -p $WD/w512_$V && cd $WD/w512_$V && cp $PGM_DATA/gmx_w512/* . && bash $H/gmx/sp_w512.sh $V 8 > sp.out 2>&1 || tail -5 sp.out
    $WD/core_tests/run_dump dump_$V.txt 2>&1 | grep -E "^total|^forces|^virial max"
    $PY $H/compare_gmx.py dump_$V.txt $PGM_DATA/ref_w512_qpi_novdw.npz 2>&1 | tail -4
    if [ -n "$PGM_GVDW_DATA" ]; then
        echo "== GROMACS $V: GVDW single points (NumPy, pmemd-pgm), overrides + tail"
        mkdir -p $WD/gmx_gvdw_$V && cd $WD/gmx_gvdw_$V && rm -f slater.top gauss.top
        bash $H/gmx/gvdw_sp.sh $V > sp.out 2>&1 || tail -5 sp.out
        for f in slater gauss; do $PY $H/compare_gmx_gvdw.py dump_$f.txt $PGM_DATA/gvdw_box.txt $f 2>&1 | tail -3; done
        bash $H/gmx/gvdw_extra.sh $V > extra.out 2>&1 || tail -5 extra.out
        $PY $H/compare_gmx_gvdw.py dump_extra.txt $PGM_DATA/gvdw_box.txt slater --override 1 2 300.0 0.002 1.5 --tail 2>&1 | tail -3
    fi
    if [ $V = double ]; then
        if [ -f "${QM7B_DIR:-$AMBERHOME/test/nmode}/qm7b_7203.prmtop" ]; then
            echo "== GROMACS $V: GVDW with 1-4 pairs (qm7b_7203) vs OpenMM"
            mkdir -p $WD/qm7b_$V && cd $WD/qm7b_$V && bash $H/gmx/gvdw_14.sh $V > run.out 2>&1 \
                && $OMM_PY $H/gmx/gvdw_14_compare.py . 2>&1 | grep -E "FAIL|PASS|failures"
        fi
        echo "== GROMACS $V: domain decomposition (4096 waters, 1-8 ranks)"
        mkdir -p $WD/gmx_dd && cd $WD/gmx_dd && bash $H/gmx/dd_sp.sh $V 2>&1 | grep -E "ok  |FAIL|NVE"
    fi
    echo "== GROMACS $V: water512 NVE"
    cd $WD/w512_$V && bash $H/gmx/nve_w512.sh $V 8 2>&1 | grep -E "Total Energy|Performance" | head -2
done
