#ifndef OPENMM_WINDOWSEXPORTPGM_H_
#define OPENMM_WINDOWSEXPORTPGM_H_

#ifdef _MSC_VER
    #if defined(OPENMM_PGM_BUILDING_SHARED_LIBRARY)
        #define OPENMM_EXPORT_PGM __declspec(dllexport)
    #elif defined(OPENMM_PGM_BUILDING_STATIC_LIBRARY) || defined(OPENMM_PGM_USE_STATIC_LIBRARIES)
        #define OPENMM_EXPORT_PGM
    #else
        #define OPENMM_EXPORT_PGM __declspec(dllimport)
    #endif
#else
    #define OPENMM_EXPORT_PGM
#endif

#endif // OPENMM_WINDOWSEXPORTPGM_H_
