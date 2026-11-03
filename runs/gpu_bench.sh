#!/bin/bash
# CUDA: the double-precision check after the fixed-point grid change, then OpenMM vs pmemd.pgm.cuda speed
# on the same 4096-water system and settings (short runs).
export CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=$HOME/project/ext/nvrtc13.0/lib/libnvrtc.so.13
PY=$HOME/miniconda3/envs/pgm-omm/bin/python
cd $HOME/project/pgm-native/runs
W=$HOME/project/pGM-JAX/examples/water512
echo "== compare_platforms CUDA double vs CPU"
$PY -u ../tests/compare_platforms.py $W/pgm3p25_512.prmtop ref_w512_qpi_novdw.npz --platform CUDA --precision double 2>&1 | grep -E "^(nocut|pme)"
B=$HOME/project/pgm-native/runs/gpubench
S=/home8/larry/pgm-exp/gpubench
cd $B
cp $S/w4096.prmtop $S/w4096.rst7 $S/mdin_s4k_040_9_56 .
sed -e "s/nstlim=500/nstlim=2000/; s/ntpr=100/ntpr=500/" mdin_s4k_040_9_56 > mdin_bench
echo "== pmemd.pgm.cuda_SPFP, w4096, 2000 steps (cut 9 A, ew_coeff 0.40, 56^3, order 6, tol 1e-4, dt 1 fs, NVE)"
/home8/larry/ambers/pgm-larry/build/src/pmemd/src/pmemd.pgm.cuda_SPFP -O -i mdin_bench -o out_pmemd -p w4096.prmtop -c w4096.rst7 -r rst_pmemd -inf inf_pmemd > log_pmemd 2>&1
grep -E "ns/day|Etot" out_pmemd | tail -4
echo "== OpenMM CUDA mixed, same settings, 2000 steps"
$PY -u $HOME/project/pgm-native/tests/md_nve.py w4096.prmtop w4096.rst7 --platform CUDA --precision mixed --steps 2000 --every 500 --tol 1e-4 --beta 4.0 --grid 56 56 56 --cutoff 0.9 --lrc 2>&1 | grep -v -i warning | tail -3
echo GPU_BENCH_DONE
