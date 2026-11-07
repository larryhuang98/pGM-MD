#!/bin/bash
# Cross-implementation check: AMBER (pmemd.pgm CPU as the reference; pmemd.pgm.cuda DPFP and SPFP), the OpenMM
# plugin and GROMACS, single points of the same coordinates of two systems: the DMP liquid of the AMBER tests
# (amber/test/dmp, 10 A cutoff, beta 0.30/A, 48^3, order 5) and 512 pGM waters (tests/data/water512, 9 A,
# 0.35/A, 36^3, order 6).  Prints the potential-energy difference and the relative RMS force deviation of
# each implementation from the AMBER CPU code (the manuscript's cross-implementation table).
#   run_xplat.sh [cpu|all]        (in an empty work directory; all: also the GPU paths; see tests/env.sh)
set -e
H=$(cd "$(dirname "$0")" && pwd)
. "$H/../env.sh"
MODE=${1:-cpu}
[ -n "$AMBER_BIN" ] || { echo "set AMBER_BIN (or AMBERHOME) to the AMBER installation with pmemd.pgm"; exit 1; }
for s in dmp w512; do
    mkdir -p $s && cd $s
    if [ $s = dmp ]; then
        gunzip -c $PGM_ROOT/amber/test/dmp/prmtop.gz > dmp.prmtop; cp $PGM_ROOT/amber/test/dmp/restrt0 dmp.rst7
        top=dmp.prmtop; crd=dmp.rst7; cut=10; beta=0.30; grid=48; order=5
    else
        cp $PGM_DATA/water512/pgm3p25_512.prmtop $PGM_DATA/water512/pgm3p25_512.rst7 .
        top=pgm3p25_512.prmtop; crd=pgm3p25_512.rst7; cut=9; beta=0.35; grid=36; order=6
    fi
    echo "== $s"
    $AMBER_BIN/pmemd.pgm -O -i $H/${s}_sp.in -o sp0.out -p $top -c $crd -frc sp0.frc -x sp0.nc -r sp0.rst -inf sp0.inf > sp0.log 2>&1
    if [ $MODE = all ]; then
        for p in DPFP SPFP; do
            $AMBER_BIN/pmemd.pgm.cuda_$p -AllowSmallBox -O -i $H/${s}_sp.in -o sp_$p.out -p $top -c $crd -frc sp_$p.frc \
                -x sp_$p.nc -r sp_$p.rst -inf sp_$p.inf > sp_$p.log 2>&1
            $PY $H/amber_vs_amber.py sp0.out sp0.frc sp_$p.out sp_$p.frc "AMBER CUDA $p"
        done
    fi
    A="--cut $cut --beta $beta --grid $grid --order $order --no-lrc"
    $OMM_PY $H/omm_vs_amber.py $top $crd sp0.out sp0.frc $A --platform CPU --precision double 2>&1 | grep -v -i warn
    if [ $MODE = all ]; then
        for p in double mixed; do
            $OMM_PY $H/omm_vs_amber.py $top $crd sp0.out sp0.frc $A --platform CUDA --precision $p 2>&1 | grep -v -i warn
        done
    fi
    rc=$(echo "$cut / 10" | bc -l); bn=$(echo "$beta * 10" | bc -l)
    mkdir -p gmx_double && (cd gmx_double && CUDA_VISIBLE_DEVICES= bash $H/gmx_sp.sh $GMX_DOUBLE ../$top ../$crd g $rc $grid $order $bn \
        && $PY $H/gmx_vs_amber.py $GMX_DOUBLE gsp ../sp0)
    if [ $MODE = all ]; then
        mkdir -p gmx_gpu && (cd gmx_gpu && GMX_PGM_GPU=1 bash $H/gmx_sp.sh $GMX_MIXED ../$top ../$crd g $rc $grid $order $bn \
            && $PY $H/gmx_vs_amber.py $GMX_MIXED gsp ../sp0)
    fi
    cd ..
done
