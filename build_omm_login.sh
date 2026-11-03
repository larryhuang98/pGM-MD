#!/bin/bash
# Build and install the OpenMM pGM plugin on the login node with gcc 11 (running it needs a compute node:
# libOpenMM needs glibc 2.27).
set -e
E=$HOME/miniconda3/envs/pgm-omm
export PATH=/home8/apps/gcc/11.3.0/bin:$E/bin:/home3/apps/cmake-3.31.6/bin:/home3/apps/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/home8/apps/gcc/11.3.0/lib64:$LD_LIBRARY_PATH
export CC=/home8/apps/gcc/11.3.0/bin/gcc CXX=/home8/apps/gcc/11.3.0/bin/g++
B=$HOME/project/pgm-native/openmm/build-gcc11
mkdir -p $B && cd $B
cmake .. -DOPENMM_DIR=$E -DCMAKE_INSTALL_PREFIX=$E -DPYTHON_EXECUTABLE=$E/bin/python -DSWIG_EXECUTABLE=$E/bin/swig -DPGM_BUILD_COMMON=ON "$@" > cmake.log 2>&1 || { tail -30 cmake.log; exit 1; }
make -j8 > make.log 2>&1 || { grep -E "error|Error" -A3 make.log | head -60; exit 1; }
make install > install.log 2>&1 || { tail -20 install.log; exit 1; }
make PythonInstall > python.log 2>&1 || { grep -E "error|Error" -A3 python.log | head -40; exit 1; }
echo BUILD_OK
