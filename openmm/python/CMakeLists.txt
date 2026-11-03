set(WRAP_FILE PGMPluginWrapper.cpp)
set(MODULE_NAME openmmpgm)

add_custom_command(
    OUTPUT "${WRAP_FILE}"
    COMMAND "${SWIG_EXECUTABLE}"
        -python -c++
        -o "${WRAP_FILE}"
        "-I${OPENMM_DIR}/include"
        "-I${CMAKE_SOURCE_DIR}/openmmapi/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/openmmpgm.i"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/openmmpgm.i" "${CMAKE_SOURCE_DIR}/openmmapi/include/PGMForce.h"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

add_custom_target(PythonInstall DEPENDS "${WRAP_FILE}" ${SHARED_PGM_TARGET})
set(PGM_HEADER_DIR "${CMAKE_SOURCE_DIR}/openmmapi/include")
set(PGM_LIBRARY_DIR "${CMAKE_BINARY_DIR}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pgm_prmtop.py ${CMAKE_CURRENT_BINARY_DIR}/pgm_prmtop.py COPYONLY)
add_custom_command(TARGET PythonInstall
    COMMAND "${PYTHON_EXECUTABLE}" -m pip install --no-deps --no-build-isolation .
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
