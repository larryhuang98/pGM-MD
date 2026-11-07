#!/bin/bash
# Configure, build and install a patched GROMACS (after install.sh).
#   gromacs/build.sh GROMACS_SRC PREFIX [mixed|double] [extra cmake arguments]
# mixed (default): single precision with CUDA (the pGM GPU engine; nvcc in PATH or -DCUDAToolkit_ROOT=...,
#   -DCMAKE_CUDA_ARCHITECTURES=... for the GPU); double: double precision, CPU only (gmx_d).
# Thread-MPI, GROMACS's own FFTW (downloaded by CMake unless -DGMX_BUILD_OWN_FFTW_URL=... points to a local copy).
set -e
S=$(cd "${1:?usage: build.sh GROMACS_SRC PREFIX [mixed|double]}" && pwd)
PREFIX=${2:?install prefix}
PREC=${3:-mixed}; shift 3 2>/dev/null || shift $#
B=$S/build-pgm-$PREC
case $PREC in
    mixed)  OPTS="-DGMX_GPU=CUDA -DGMX_DOUBLE=OFF" ;;
    double) OPTS="-DGMX_GPU=OFF -DGMX_DOUBLE=ON" ;;
    *) echo "precision must be mixed or double"; exit 1 ;;
esac
mkdir -p "$B" && cd "$B"
cmake "$S" -DCMAKE_BUILD_TYPE=Release $OPTS -DGMX_MPI=OFF -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=OFF \
      -DGMX_EXTERNAL_BLAS=OFF -DGMX_EXTERNAL_LAPACK=OFF -DCMAKE_INSTALL_PREFIX="$PREFIX" "$@" > cmake.log 2>&1 \
    || { tail -40 cmake.log; exit 1; }
make -j"${NPROC:-8}" > make.log 2>&1 || { grep -E "error" -A4 make.log | head -60; exit 1; }
make install > install.log 2>&1
echo "GROMACS ($PREC) with pGM installed in $PREFIX"
