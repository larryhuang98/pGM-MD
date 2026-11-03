"""G1: build src/gromacs/pgm, own it in t_forcerec (fr->pgm, created for coulombtype = pGM), call it in
do_force next to the long-range nonbondeds (PME mesh) on slow-force steps.  Its forces go to the
ForceWithVirial output with the analytical virial."""

EDITS = [
    ("src/gromacs/CMakeLists.txt",
     "add_subdirectory(ewald)\n",
     "add_subdirectory(ewald)\nadd_subdirectory(pgm)\n"),
    # ---------------------------------------------------------------- t_forcerec
    ("src/gromacs/mdtypes/forcerec.h",
     "class WholeMoleculeTransform;\n",
     "class WholeMoleculeTransform;\nclass PgmElectrostatics;\n"),
    ("src/gromacs/mdtypes/forcerec.h",
     "    std::unique_ptr<DispersionCorrection> dispersionCorrection;\n",
     "    std::unique_ptr<DispersionCorrection> dispersionCorrection;\n\n"
     "    /* pGM electrostatics (coulombtype = pGM), nullptr otherwise */\n"
     "    std::unique_ptr<gmx::PgmElectrostatics> pgm;\n"),
    ("src/gromacs/mdlib/forcerec.cpp",
     '#include "gromacs/nbnxm/nbnxm.h"\n',
     '#include "gromacs/nbnxm/nbnxm.h"\n#include "gromacs/pgm/pgm.h"\n'),
    ("src/gromacs/mdlib/forcerec.cpp",
     "        forcerec->dispersionCorrection->print(mdlog);\n    }\n",
     "        forcerec->dispersionCorrection->print(mdlog);\n    }\n\n"
     "    if (usingPgm(inputrec.coulombtype))\n    {\n"
     "        if (simulationWork.useMts)\n        {\n"
     '            gmx_fatal(FARGS, "coulombtype = pGM does not support multiple time stepping yet");\n'
     "        }\n"
     "        if (simulationWork.haveSeparatePmeRank)\n        {\n"
     '            gmx_fatal(FARGS, "coulombtype = pGM computes its own PME on the PP ranks: use mdrun -npme 0");\n'
     "        }\n"
     "        if (simulationWork.useGpuNonbonded)\n        {\n"
     '            gmx_fatal(FARGS, "coulombtype = pGM needs the nonbonded interactions on the CPU (mdrun -nb cpu)");\n'
     "        }\n"
     "        forcerec->pgm = std::make_unique<gmx::PgmElectrostatics>(inputrec, fplog);\n    }\n"),
    # ---------------------------------------------------------------- do_force
    ("src/gromacs/mdlib/sim_util.cpp",
     '#include "gromacs/nbnxm/nbnxm.h"\n',
     '#include "gromacs/nbnxm/nbnxm.h"\n#include "gromacs/pgm/pgm.h"\n'),
    ("src/gromacs/mdlib/sim_util.cpp",
     "                                       stepWork,\n                                       ddBalanceRegionHandler);\n    }\n",
     "                                       stepWork,\n                                       ddBalanceRegionHandler);\n"
     "        if (fr->pgm)\n        {\n"
     "            // pGM electrostatics: direct space over the nbnxm CPU lists, PME of charges and dipoles\n"
     "            fr->pgm->calculate(x.unpaddedConstArrayRef(),\n"
     "                               box,\n"
     "                               mdatoms->homenr,\n"
     "                               mdatoms->chargeA,\n"
     "                               *fr->nbv,\n"
     "                               top->idef,\n"
     "                               fr->shift_vec,\n"
     "                               haveDDAtomOrdering(*cr) ? ArrayRef<const int>(cr->dd->globalAtomIndices)\n"
     "                                                       : ArrayRef<const int>(),\n"
     "                               &forceOutMtsLevel1->forceWithVirial(),\n"
     "                               enerd,\n"
     "                               stepWork);\n"
     "        }\n    }\n"),
    # ---------------------------------------------------------------- tables (DispCorr builds them)
    ("src/gromacs/tables/forcetable.cpp",
     "        case CoulombInteractionType::Cut: tabsel[etiCOUL] = etabCOUL; break;\n",
     "        case CoulombInteractionType::Cut:\n"
     "        case CoulombInteractionType::Pgm: // only the dispersion correction uses these tables\n"
     "            tabsel[etiCOUL] = etabCOUL;\n"
     "            break;\n"),
]
