# pGM build module for AMBER24 PMEMD (included at the end of src/pmemd/src/CMakeLists.txt)
# Builds pmemd.pgm (CPU), and pmemd.pgm.cuda_{SPFP,DPFP} when CUDA is enabled.
# Standard PMEMD targets are not modified.

set(PGM_FORTRAN_SOURCES pGM_direct.F90 pGM_induced.F90
    pGM_interface_data_mod.F90 pGM_interface.F90
    pGM_mdin_dat.F90 pGM_mdin.F90
    pGM_multipoles.F90 pGM_readparm.F90
    pGM_self.F90)
set(PGM_CUDA_FORTRAN_SOURCES pGM_gpu_interface.F90 pGM_gpu_setup.F90)
set_property(SOURCE ${PGM_FORTRAN_SOURCES} ${PGM_CUDA_FORTRAN_SOURCES} PROPERTY COMPILE_FLAGS "${PMEMD_FFLAGS_SPC}")

set(PGM_DEFINITIONS ${PMEMD_DEFINITIONS} PGM)
set(PGM_FORTRAN_WRAPPER "${CMAKE_CURRENT_SOURCE_DIR}/pGM/fortran_with_cpp.sh")
execute_process(COMMAND chmod a+x ${PGM_FORTRAN_WRAPPER})

# ---- CPU ----
add_executable(pmemd.pgm ${PMEMD_C_SOURCES} ${PMEMD_CXX_SOURCES} ${PMEMD_FORTRAN_SOURCES} ${PGM_FORTRAN_SOURCES} ${PMEMD_NFE_SOURCES} ${PLUMED_SOURCE})
config_module_dirs(pmemd.pgm ${PMEMD_MOD_DIR}/pgm ${AMBER_COMMON_MOD_DIR} ${NETCDF_FORTRAN_MOD_DIR})
set_target_properties(pmemd.pgm PROPERTIES Fortran_COMPILER_LAUNCHER ${PGM_FORTRAN_WRAPPER})
target_link_libraries(pmemd.pgm emil amber_common netcdff netlib boost_headers ${KMMD_LIB} PMEMD::xray_cpu)
target_compile_definitions(pmemd.pgm PRIVATE ${PGM_DEFINITIONS})
set_property(TARGET pmemd.pgm PROPERTY LINKER_LANGUAGE Fortran)
if(PLUMED_RUNTIME_LINK)
    target_link_libraries(pmemd.pgm dl)
    set_target_properties(pmemd.pgm PROPERTIES ENABLE_EXPORTS TRUE)
elseif(plumed_ENABLED)
    target_link_libraries(pmemd.pgm plumed::plumed)
endif()
install(TARGETS pmemd.pgm DESTINATION ${BINDIR} COMPONENT pmemd_pgm)

if(MPI)
    make_mpi_version(pmemd.pgm pmemd.pgm.MPI LANGUAGES Fortran)
    config_module_dirs(pmemd.pgm.MPI ${PMEMD_MOD_DIR}/pgm_mpi ${AMBER_COMMON_MOD_DIR} ${NETCDF_FORTRAN_MOD_DIR})
    set_target_properties(pmemd.pgm.MPI PROPERTIES Fortran_COMPILER_LAUNCHER ${PGM_FORTRAN_WRAPPER})
    target_compile_definitions(pmemd.pgm.MPI PRIVATE ${PMEMD_MPI_DEFINITIONS})
    target_compile_options(pmemd.pgm.MPI PRIVATE ${PMEMD_MPI_FLAGS})
    install(TARGETS pmemd.pgm.MPI DESTINATION ${BINDIR} COMPONENT pmemd_pgm_MPI)
endif()

# ---- CUDA ----
if(CUDA)
    add_library(pmemd_pgm_obj_cuda OBJECT ${PMEMD_C_SOURCES} ${PMEMD_CXX_SOURCES} ${PMEMD_FORTRAN_SOURCES} ${PMEMD_FORTRAN_GPU_SOURCES}
                ${PGM_FORTRAN_SOURCES} ${PGM_CUDA_FORTRAN_SOURCES} ${PMEMD_NFE_SOURCES} ${PLUMED_SOURCE})
    config_module_dirs(pmemd_pgm_obj_cuda ${PMEMD_MOD_DIR}/pgm_cuda ${AMBER_COMMON_MOD_DIR} ${NETCDF_FORTRAN_MOD_DIR})
    add_dependencies(pmemd_pgm_obj_cuda amber_common netcdff)
    target_compile_definitions(pmemd_pgm_obj_cuda PRIVATE ${PGM_DEFINITIONS} ${PMEMD_CUDA_DEFINES})
    target_include_directories(pmemd_pgm_obj_cuda PRIVATE $<TARGET_PROPERTY:boost_headers,INTERFACE_INCLUDE_DIRECTORIES>)
    set_target_properties(pmemd_pgm_obj_cuda PROPERTIES Fortran_COMPILER_LAUNCHER ${PGM_FORTRAN_WRAPPER})
    if(${CMAKE_VERSION} VERSION_LESS 3.12)
        target_include_directories(pmemd_pgm_obj_cuda PUBLIC $<TARGET_PROPERTY:PMEMD::xray_gpu,INTERFACE_INCLUDE_DIRECTORIES>)
        target_compile_definitions(pmemd_pgm_obj_cuda PUBLIC $<TARGET_PROPERTY:PMEMD::xray_gpu,INTERFACE_COMPILE_DEFINITIONS>)
        add_dependencies(pmemd_pgm_obj_cuda PMEMD::xray_gpu)
    else()
        target_link_libraries(pmemd_pgm_obj_cuda PUBLIC PMEMD::xray_gpu)
    endif()

    foreach(PRECISION ${PMEMD_CUDA_PRECISIONS})
        set(PGM_EXE_NAME pmemd.pgm.cuda_${PRECISION})
        add_executable(${PGM_EXE_NAME} $<TARGET_OBJECTS:pmemd_pgm_obj_cuda>)
        set_property(TARGET ${PGM_EXE_NAME} PROPERTY LINKER_LANGUAGE Fortran)
        target_link_libraries(${PGM_EXE_NAME} pmemd_pgm_cuda_${PRECISION} emil amber_common netcdff netlib boost_headers ${KMMD_LIB} PMEMD::xray_gpu)
        cuda_add_cufft_to_target(${PGM_EXE_NAME})
        if(PLUMED_RUNTIME_LINK)
            target_link_libraries(${PGM_EXE_NAME} dl)
            set_target_properties(${PGM_EXE_NAME} PROPERTIES ENABLE_EXPORTS TRUE)
        elseif(plumed_ENABLED)
            target_link_libraries(${PGM_EXE_NAME} plumed)
        endif()
        install(TARGETS ${PGM_EXE_NAME} DESTINATION ${BINDIR} COMPONENT pmemd_pgm_CUDA)
    endforeach()
    install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
        \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${BINDIR}/pmemd.pgm.cuda_${PMEMD_DEFAULT_PRECISION}
        \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${BINDIR}/pmemd.pgm.cuda)" COMPONENT pmemd_pgm_CUDA)
endif()
