#!/bin/bash
# OpenMM pGM/GVDW tests on a GPU node (CUDA platform), short runs only.
# NVRTC 13.4 of the env makes PTX newer than the node driver (CUDA 13.3) accepts: preload NVRTC 13.0.
export CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=$HOME/project/ext/nvrtc13.0/lib/libnvrtc.so.13
E=$HOME/miniconda3/envs/pgm-omm
PY=$E/bin/python
cd $HOME/project/pgm-native/runs
W=$HOME/project/pGM-JAX/examples/water512
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "== test_pgm_api (Reference, CUDA)"
$PY -u ../tests/test_pgm_api.py $W/pgm3p25_512.prmtop $W/pgm3p25_512.rst7 --platforms Reference CUDA 2>&1 | grep -v -i warning
echo "== test_tail (TailCorrectionForce)"
$PY -u ../tests/test_tail.py Reference CPU CUDA 2>&1 | grep -E "FAIL|PASS|failures"
echo "== test_gvdw_api (Reference, CUDA)"
$PY -u ../tests/test_gvdw_api.py --platforms Reference CUDA 2>&1 | grep -E "FAIL|PASS|failures"
for prec in double mixed single; do
  echo "== compare_platforms CUDA $prec vs CPU"
  $PY -u ../tests/compare_platforms.py $W/pgm3p25_512.prmtop ref_w512_qpi_novdw.npz --platform CUDA --precision $prec 2>&1 | grep -v -i warning | tail -8
done
GD=$HOME/pgm-gvdw-data
echo "== compare_gvdw (Reference, CUDA)"
$PY -u ../tests/compare_gvdw.py $GD/topology/rayl_512_v2.prmtop gvdw_box.txt --platforms Reference CUDA 2>&1 | grep -v -i warning | tail -8
echo "== NVE water512, CUDA mixed, 2000 steps"
$PY -u ../tests/md_nve.py $W/pgm3p25_512.prmtop $W/pgm3p25_512.rst7 --platform CUDA --precision mixed --steps 2000 --every 200 2>&1 | grep -v -i warning | tail -6
echo GPU_TESTS_DONE
