"""G1b: input checks and output for coulombtype = pGM.

* grompp: every atom needs a [ pgm_atoms ] entry; 1-4 pairs must not carry electrostatics (fudgeQQ = 0,
  no pair type 2 / pairs_nb), since the pGM sum over all pairs already contains them.
* mdrun: fatal error for GPU nonbonded (the pGM pair kernels run on the CPU lists until phase G3).
* energy output: no "pGM Cov. Dip." term (a data list without energy).
"""

EDITS = [
    ("src/gromacs/gmxpreprocess/grompp.cpp",
     "        checkExclusionDistances(sys, *ir, state.x, state.box, logger, &wi);\n    }\n",
     "        checkExclusionDistances(sys, *ir, state.x, state.box, logger, &wi);\n    }\n\n"
     "    if (usingPgm(ir->coulombtype))\n    {\n"
     "        if (gmx_mtop_ftype_count(sys, InteractionFunction::PgmAtom) != sys.natoms)\n        {\n"
     "            wi.addError(gmx::formatString(\n"
     '                    "coulombtype = pGM: %d of %d atoms have a [ pgm_atoms ] entry; every atom needs one",\n'
     "                    gmx_mtop_ftype_count(sys, InteractionFunction::PgmAtom),\n"
     "                    sys.natoms));\n        }\n"
     "        if ((fudgeQQ != 0 && gmx_mtop_ftype_count(sys, InteractionFunction::LennardJones14) > 0)\n"
     "            || gmx_mtop_ftype_count(sys, InteractionFunction::LennardJonesCoulomb14Q) > 0\n"
     "            || gmx_mtop_ftype_count(sys, InteractionFunction::LennardJonesCoulombNonBondedPairs) > 0)\n"
     "        {\n"
     "            wi.addError(\n"
     '                    "coulombtype = pGM: 1-4 pairs must carry Lennard-Jones only (fudgeQQ = 0, no pair "\n'
     '                    "type 2 or [ pairs_nb ]): the pGM electrostatics already include all pairs");\n'
     "        }\n    }\n"),
    ("src/gromacs/mdlib/forcerec.cpp",
     "        forcerec->pgm = std::make_unique<gmx::PgmElectrostatics>(inputrec, fplog);\n",
     "        if (simulationWork.useGpuNonbonded)\n        {\n"
     '            gmx_fatal(FARGS, "coulombtype = pGM needs the nonbonded interactions on the CPU (mdrun -nb cpu)");\n'
     "        }\n"
     "        forcerec->pgm = std::make_unique<gmx::PgmElectrostatics>(inputrec, fplog);\n"),
    ("src/gromacs/mdlib/energyoutput.cpp",
     "    bEner_[InteractionFunction::LennardJonesCoulombNonBondedPairs] = false;\n",
     "    bEner_[InteractionFunction::LennardJonesCoulombNonBondedPairs] = false;\n"
     "    bEner_[InteractionFunction::PgmCovalentDipole] = false; // pGM: data, the energy is elsewhere\n"),
]
