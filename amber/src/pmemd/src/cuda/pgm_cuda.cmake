# pGM CUDA libraries for AMBER24 PMEMD (included at the end of src/pmemd/src/cuda/CMakeLists.txt)
set(PGM_CUDA_SOURCES kSelf_PGM.cu kPMEInterpolation_PGM.cu kCalculatePMENonbondEnergy_PGM.cu
    kLocalToGlobal_PGM.cu kSCF_PGM.cu kChainRule_PGM.cu kNLRemap_PGM.cu kPrecondCache_PGM.cu kCPULS_PGM.cu)
set(PGM_CUDA_CXX_SOURCES pgm_gpu.cpp)
set_property(SOURCE ${PGM_CUDA_CXX_SOURCES} PROPERTY COMPILE_FLAGS "${OPT_CXXFLAGS_SPC}")
set_property(SOURCE ${PGM_CUDA_CXX_SOURCES} PROPERTY COMPILE_DEFINITIONS ${PMEMD_C_DEFINITIONS})
set_property(SOURCE ${PGM_CUDA_CXX_SOURCES} PROPERTY INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS})

foreach(PRECISION ${PMEMD_CUDA_PRECISIONS})
    cuda_add_library(pmemd_pgm_cuda_${PRECISION} STATIC ${PMEMD_CUDA_CUDA_SOURCES} ${PMEMD_CUDA_GTI_SOURCES} ${PGM_CUDA_SOURCES}
        ${PMEMD_CUDA_FORTRAN_SOURCES} ${PMEMD_CUDA_CXX_SOURCES} ${PGM_CUDA_CXX_SOURCES}
        OPTIONS ${PMEMD_NVCC_FLAGS} -Duse_${PRECISION} -DPGM)
    # build after the standard library to limit peak nvcc memory
    add_dependencies(pmemd_pgm_cuda_${PRECISION} pmemd_cuda_${PRECISION})
    target_compile_definitions(pmemd_pgm_cuda_${PRECISION} PRIVATE -Duse_${PRECISION} -DPGM)
    target_link_libraries(pmemd_pgm_cuda_${PRECISION} curand cublas cusolver)
    if(VKFFT)
        target_link_libraries(pmemd_pgm_cuda_${PRECISION} cuda cudadevrt nvrtc)
    endif()
    if(NCCL)
        target_link_libraries(pmemd_pgm_cuda_${PRECISION} nccl)
    endif()
endforeach()
