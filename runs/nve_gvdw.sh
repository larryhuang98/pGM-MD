#!/bin/bash
# NVE of the GVDW manuscript box (pGM + GVDW Slater) on OpenCL (pocl, mixed), login node via glibc228.sh
export OCL_ICD_VENDORS=$HOME/project/ext/pocl-env/etc/OpenCL/vendors OCL_LIB=$HOME/project/ext/pocl-env/lib CUDA_VISIBLE_DEVICES=
cd $HOME/project/pgm-native/runs
D=$HOME/pgm-gvdw-data
./glibc228.sh python ../tests/md_nve.py $D/topology/rayl_512_v2.prmtop $D/inputs/lj/inpcrd.restrt --vdw gvdw --gvdw ${1:-slater} --cutoff 0.8 --beta 4.0 --grid 48 48 48 --platform OpenCL --precision mixed --steps ${2:-400} --every 20 2>&1 | grep -v -i warning
