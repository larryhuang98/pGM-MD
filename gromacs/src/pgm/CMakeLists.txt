# pGM electrostatics (coulombtype = pGM), see pgm-native/docs/gromacs_design.md
gmx_add_libgromacs_sources(
    pgm.cpp
    )
# The GPU engine (pgm-native core/cuda, copied here by g1_install.sh; GMX_PGM_GPU=1 at run time) is its
# own static library: its CG vectors, energies and strain are double precision, which libgromacs's
# CUDA flags (-Xptxas -warn-double-usage -Werror) forbid.  libgromacs links it (g3_gpu.py).
if (GMX_GPU_CUDA)
    add_library(pgm_cuda_engine STATIC pgm_cuda.cu)
    set_target_properties(pgm_cuda_engine PROPERTIES POSITION_INDEPENDENT_CODE ON
                          CUDA_ARCHITECTURES "${GMX_CUDA_ARCHITECTURES}")
    target_compile_options(pgm_cuda_engine PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>")
    target_link_libraries(pgm_cuda_engine PRIVATE CUDA::cufft)
endif()
