#!/bin/bash
# OpenMM NPT water512 (CUDA mixed): the pgm_cuda engine against the Common kernels (PGM_OPENMM_COMMON=1).
export CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=$HOME/project/ext/nvrtc13.0/lib/libnvrtc.so.13
PY=$HOME/miniconda3/envs/pgm-omm/bin/python
W=$HOME/project/pGM-JAX/examples/water512
cd $HOME/project/pgm-native/runs
for seed in 1 2; do
  for mode in engine common; do
    if [ $mode = common ]; then export PGM_OPENMM_COMMON=1; else unset PGM_OPENMM_COMMON; fi
    echo "== $mode seed $seed"
    $PY -u ../tests/md_npt.py $W/pgm3p25_512.prmtop $W/pgm3p25_512.rst7 --platform CUDA --precision mixed \
      --ps ${PS:-300} --equil-ps 50 --every 1000 --dt-fs 2 --beta 4.0 --grid 48 48 48 --tol 1e-5 --seed $seed 2>&1 \
      | grep -v -i warning > omm_npt_${mode}_$seed.log
    tail -2 omm_npt_${mode}_$seed.log
  done
done
echo NPT_DONE
