#!/bin/bash
# Build and install the OpenMM pGM plugin.
#   openmm/build.sh [extra cmake arguments]
# Environment (defaults in parentheses):
#   OPENMM_DIR   OpenMM installation, e.g. a conda environment with openmm ($CONDA_PREFIX)
#   PGM_PREFIX   where to install the plugin libraries ($OPENMM_DIR, so that OpenMM finds the platform plugins)
#   PYTHON       Python of that OpenMM ($OPENMM_DIR/bin/python); SWIG ($OPENMM_DIR/bin/swig, or swig in PATH)
#   BUILD_DIR    build directory (openmm/build)
# CUDA is found through nvcc in PATH (or CUDAToolkit_ROOT); set CMAKE_CUDA_ARCHITECTURES with -D... if the
# default (80;89;120) does not cover the GPU.
set -e
H=$(cd "$(dirname "$0")" && pwd)
: "${OPENMM_DIR:=${CONDA_PREFIX:?set OPENMM_DIR to the OpenMM installation}}"
: "${PGM_PREFIX:=$OPENMM_DIR}" "${PYTHON:=$OPENMM_DIR/bin/python}" "${BUILD_DIR:=$H/build}"
SWIG=${SWIG:-$([ -x "$OPENMM_DIR/bin/swig" ] && echo "$OPENMM_DIR/bin/swig" || command -v swig)}
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
cmake "$H" -DOPENMM_DIR="$OPENMM_DIR" -DCMAKE_INSTALL_PREFIX="$PGM_PREFIX" -DPYTHON_EXECUTABLE="$PYTHON" \
      -DSWIG_EXECUTABLE="$SWIG" "$@" > cmake.log 2>&1 || { tail -30 cmake.log; exit 1; }
make -j"${NPROC:-8}" > make.log 2>&1 || { grep -E "error|Error" -A3 make.log | head -60; exit 1; }
make install > install.log 2>&1 || { tail -20 install.log; exit 1; }
make PythonInstall > python.log 2>&1 || { grep -E "error|Error" -A3 python.log | head -40; exit 1; }
echo "OpenMM pGM plugin installed in $PGM_PREFIX (Python modules openmmpgm, pgm_prmtop for $PYTHON)"
