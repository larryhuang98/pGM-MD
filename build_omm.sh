#!/bin/bash
# Build and install the OpenMM pGM plugin into the pgm-omm env (run on a compute node).
set -e
E=$HOME/miniconda3/envs/pgm-omm
export PATH=$E/bin:/home3/apps/cmake-3.31.6/bin:/home3/apps/cuda-12.8/bin:$PATH
B=$HOME/project/pgm-native/openmm/build
mkdir -p $B && cd $B
cmake .. -DOPENMM_DIR=$E -DCMAKE_INSTALL_PREFIX=$E -DPYTHON_EXECUTABLE=$E/bin/python -DSWIG_EXECUTABLE=$E/bin/swig "$@" > cmake.log 2>&1 || { tail -30 cmake.log; exit 1; }
make -j6 > make.log 2>&1 || { grep -E "error|Error" -A3 make.log | head -60; exit 1; }
make install > install.log 2>&1 || { tail -20 install.log; exit 1; }
make PythonInstall > python.log 2>&1 || { grep -E "error|Error" -A3 python.log | head -40; exit 1; }
echo BUILD_OK
