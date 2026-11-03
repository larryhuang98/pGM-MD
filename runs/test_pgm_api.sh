#!/bin/bash
export OCL_ICD_VENDORS=$HOME/project/ext/pocl-env/etc/OpenCL/vendors OCL_LIB=$HOME/project/ext/pocl-env/lib CUDA_VISIBLE_DEVICES=
cd $HOME/project/pgm-native/runs
W=$HOME/project/pGM-JAX/examples/water512
./glibc228.sh python -u ../tests/test_pgm_api.py $W/pgm3p25_512.prmtop $W/pgm3p25_512.rst7 "$@" 2>&1 | grep -v -i "warning"
