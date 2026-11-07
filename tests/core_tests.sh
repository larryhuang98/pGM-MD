#!/bin/bash
# Unit tests of the shared C++ reference implementation (core/): pGM energies, forces and induction against
# finite differences and internal identities, and the GVDW pair kernel.
#   tests/core_tests.sh            (builds in ./core_tests; CXX, default g++ with C++17)
set -e
H=$(cd "$(dirname "$0")" && pwd)
C=$H/../core
mkdir -p core_tests && cd core_tests
${CXX:-g++} -O2 -std=c++17 -pthread -I$C $C/pgm_core.cpp $C/tests/test_core.cpp -o test_core
${CXX:-g++} -O2 -std=c++17 -pthread -I$C $C/pgm_core.cpp $C/tests/run_dump.cpp -o run_dump
${CXX:-g++} -O2 -std=c++17 -I$C $C/tests/test_gvdw.cpp -o test_gvdw
./test_core 2>&1 | tail -1
./test_gvdw 2>&1 | tail -1
