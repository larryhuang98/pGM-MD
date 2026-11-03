#!/bin/bash
# Every validation of pgm-native in one run (rayl8 login node); one summary block per check, details in
# the work directories under runs/.  Needs: the GROMACS builds install-double / install-cuda, the OpenMM
# plugin in the pgm-omm env, runs/gmx_w512 (sp.mdp with rlist 1.0, w512.g96, w512.top), runs/gvdw_box.txt,
# runs/ref_w512_*_novdw.npz (tests/ref_pgmjax.py).
#   tests/validate_all.sh > runs/validate.out 2>&1
R=$HOME/project/pgm-native/runs
T=$HOME/project/pgm-native/tests
P=$HOME/miniconda3/envs/SE3nv/bin/python
export PATH=/home8/apps/gcc/11.3.0/bin:$PATH LD_LIBRARY_PATH=/home8/apps/gcc/11.3.0/lib64:$LD_LIBRARY_PATH
export OCL_ICD_VENDORS=$HOME/project/ext/pocl-env/etc/OpenCL/vendors OCL_LIB=$HOME/project/ext/pocl-env/lib CUDA_VISIBLE_DEVICES=
echo "== core"
cd $HOME/project/pgm-native/core/tests
g++ -O2 -std=c++17 -pthread -I.. ../pgm_core.cpp test_core.cpp -o test_core && g++ -O2 -std=c++17 -pthread -I.. ../pgm_core.cpp run_dump.cpp -o run_dump && g++ -O2 -std=c++17 -I.. test_gvdw.cpp -o test_gvdw
./test_core 2>&1 | tail -1; ./test_gvdw 2>&1 | tail -1
for V in double cuda; do
  echo "== GROMACS $V: water512 pGM single point vs pgm_core and pGM-JAX"
  cd $R/gmx_w512 && bash $T/gmx/sp_w512.sh $V 8 > sp_$V.out 2>&1
  $R/../core/tests/run_dump dump_$V.txt 2>&1 | grep -E "^total|^forces|^virial max"
  $P $T/compare_gmx.py dump_$V.txt $R/ref_w512_qpi_novdw.npz 2>&1 | tail -4
  echo "== GROMACS $V: GVDW single points (NumPy, pmemd), overrides + tail"
  mkdir -p $R/gmx_gvdw_$V && cd $R/gmx_gvdw_$V && rm -f slater.top gauss.top
  bash $T/gmx/gvdw_sp.sh $V > sp.out 2>&1 || tail -5 sp.out
  for f in slater gauss; do $P $T/compare_gmx_gvdw.py dump_$f.txt $R/gvdw_box.txt $f 2>&1 | tail -3; done
  bash $T/gmx/gvdw_extra.sh $V > extra.out 2>&1 || tail -5 extra.out
  $P $T/compare_gmx_gvdw.py dump_extra.txt $R/gvdw_box.txt slater --override 1 2 300.0 0.002 1.5 --tail 2>&1 | tail -3
  if [ $V = double ]; then   # the bonded terms of a mixed build are single precision: double only
    echo "== GROMACS $V: GVDW with 1-4 pairs (qm7b) vs OpenMM"
    mkdir -p $R/qm7b_$V && cd $R/qm7b_$V && bash $T/gmx/gvdw_14.sh $V > run.out 2>&1 \
      && $R/glibc228.sh python $T/gmx/gvdw_14_compare.py . 2>&1 | grep -E "FAIL|PASS|failures"
  fi
done
echo "== GROMACS cuda (mixed): water512 NVE 4 ps"
cd $R/gmx_w512 && MDRUN_EXTRA="-pin on" bash $T/gmx/nve_w512.sh cuda 8 2>&1 | grep -E "Total Energy|Performance" | head -2
echo "== OpenMM"
cd $R && bash ./test_pgm_api.sh 2>&1 | grep -E "FAIL|PASS|failures"
./glibc228.sh python -u $T/test_gvdw_api.py 2>&1 | grep -E "FAIL|PASS|failures"
W=$HOME/project/pGM-JAX/examples/water512
for L in qpi q qp qi; do
  echo "-- OpenMM vs pGM-JAX, levels $L"
  ./glibc228.sh python $T/compare_openmm.py $W/pgm3p25_512.prmtop ref_w512_${L}_novdw.npz --elec $L --no-lj --ke 138.935458 2>&1 \
    | grep -v -i warning | grep -v "^vdw" | tail -4
done
echo ALL_DONE
