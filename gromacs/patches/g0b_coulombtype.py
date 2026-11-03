"""G0b: coulombtype = pGM, the pGM solver options in t_inputrec / mdp / tpr, LJ-only nbnxm kernels.

With coulombtype = pGM the nbnxm kernels compute Lennard-Jones only (CoulombKernelType::None, the FMM
kernels, now built always: g0b_elecnone.sh), GROMACS's own PME of charges is off (usingPme() is false),
and the electrostatics come from src/gromacs/pgm (phase G1).  PME grid, order, ewald-rtol and rcoulomb
are the usual options.
"""

EDITS = [
    # ---------------------------------------------------------------- the enum
    ("api/legacy/include/gromacs/mdtypes/md_enums.h",
     "    RFZero,\n    Fmm,\n    Count,\n    Default = Cut\n",
     "    RFZero,\n    Fmm,\n    Pgm, //!< pGM: Gaussian charges, covalent and induced Gaussian dipoles (src/gromacs/pgm)\n"
     "    Count,\n    Default = Cut\n"),
    ("api/legacy/include/gromacs/mdtypes/md_enums.h",
     "    return (usingPmeOrEwald(cit) || cit == CoulombInteractionType::Poisson\n"
     "            || cit == CoulombInteractionType::Fmm);\n}\n",
     "    return (usingPmeOrEwald(cit) || cit == CoulombInteractionType::Poisson\n"
     "            || cit == CoulombInteractionType::Fmm || cit == CoulombInteractionType::Pgm);\n}\n\n"
     "//! Returns whether we use pGM electrostatics (own real space and PME of charges and dipoles)\n"
     "static inline bool usingPgm(const CoulombInteractionType& cit)\n{\n"
     "    return cit == CoulombInteractionType::Pgm;\n}\n"),
    ("src/gromacs/mdtypes/md_enums.cpp",
     '        "Reaction-Field-zero",\n        "FMM"\n',
     '        "Reaction-Field-zero",\n        "FMM",\n        "pGM"\n'),
    # ---------------------------------------------------------------- inputrec fields
    ("api/legacy/include/gromacs/mdtypes/inputrec.h",
     "    //! Epsilon for PME dipole correction\n    real epsilon_surface = 0;\n",
     "    //! Epsilon for PME dipole correction\n    real epsilon_surface = 0;\n"
     "    //! pGM: induced-dipole convergence max|alpha r| / mean|alpha E_perm|\n    real pgm_dipole_tol = 1e-5;\n"
     "    //! pGM: maximum number of conjugate-gradient iterations\n    int pgm_max_iter = 100;\n"
     "    //! pGM: initial guess of the dipoles: 0 alpha E_perm, 1 quadratic, 2 cubic extrapolation (mu4)\n    int pgm_predictor = 2;\n"
     "    //! pGM: final step mu += peek alpha r (pmemd scf_sor_coefficient; 0 disables)\n    real pgm_peek = 0.65;\n"),
    ("src/gromacs/gmxpreprocess/readir.cpp",
     '    ir->epsilon_surface        = get_ereal(&inp, "epsilon-surface", 0.0, wi);\n',
     '    ir->epsilon_surface        = get_ereal(&inp, "epsilon-surface", 0.0, wi);\n'
     '    printStringNoNewline(&inp, "pGM (coulombtype = pGM): induced-dipole solver");\n'
     '    ir->pgm_dipole_tol = get_ereal(&inp, "pgm-dipole-tol", 1e-5, wi);\n'
     '    ir->pgm_max_iter   = get_eint(&inp, "pgm-max-iter", 100, wi);\n'
     '    ir->pgm_predictor  = get_eint(&inp, "pgm-predictor", 2, wi);\n'
     '    ir->pgm_peek       = get_ereal(&inp, "pgm-peek", 0.65, wi);\n'),
    ("src/gromacs/fileio/tpxio.cpp",
     "    serializer->doEnumAsInt(&ir->ewald_geometry);\n    serializer->doReal(&ir->epsilon_surface);\n",
     "    serializer->doEnumAsInt(&ir->ewald_geometry);\n    serializer->doReal(&ir->epsilon_surface);\n"
     "    if (file_version >= tpxv_PgmInteractions)\n    {\n"
     "        serializer->doReal(&ir->pgm_dipole_tol);\n        serializer->doInt(&ir->pgm_max_iter);\n"
     "        serializer->doInt(&ir->pgm_predictor);\n        serializer->doReal(&ir->pgm_peek);\n    }\n"),
    ("src/gromacs/mdtypes/inputrec.cpp",
     '        PR("epsilon-surface", ir->epsilon_surface);\n',
     '        PR("epsilon-surface", ir->epsilon_surface);\n'
     '        PR("pgm-dipole-tol", ir->pgm_dipole_tol);\n        PI("pgm-max-iter", ir->pgm_max_iter);\n'
     '        PI("pgm-predictor", ir->pgm_predictor);\n        PR("pgm-peek", ir->pgm_peek);\n'),
    ("src/gromacs/mdtypes/inputrec.cpp",
     '    cmp_real(fp, "inputrec->epsilon_surface", -1, ir1->epsilon_surface, ir2->epsilon_surface, ftol, abstol);\n',
     '    cmp_real(fp, "inputrec->epsilon_surface", -1, ir1->epsilon_surface, ir2->epsilon_surface, ftol, abstol);\n'
     '    cmp_real(fp, "inputrec->pgm_dipole_tol", -1, ir1->pgm_dipole_tol, ir2->pgm_dipole_tol, ftol, abstol);\n'
     '    cmp_int(fp, "inputrec->pgm_max_iter", -1, ir1->pgm_max_iter, ir2->pgm_max_iter);\n'
     '    cmp_int(fp, "inputrec->pgm_predictor", -1, ir1->pgm_predictor, ir2->pgm_predictor);\n'
     '    cmp_real(fp, "inputrec->pgm_peek", -1, ir1->pgm_peek, ir2->pgm_peek, ftol, abstol);\n'),
    # ---------------------------------------------------------------- grompp checks
    ("src/gromacs/gmxpreprocess/readir.cpp",
     "              || ir->coulombtype == CoulombInteractionType::Fmm))\n        {\n            wi->addError(\n"
     '                    "With Verlet lists only cut-off, reaction-field, PME, Ewald, and FMM"\n',
     "              || ir->coulombtype == CoulombInteractionType::Fmm || usingPgm(ir->coulombtype)))\n        {\n"
     "            wi->addError(\n"
     '                    "With Verlet lists only cut-off, reaction-field, PME, Ewald, FMM and pGM"\n'),
    ("src/gromacs/gmxpreprocess/readir.cpp",
     "    if (usingPme(ir->coulombtype) || usingLJPme(ir->vdwtype))\n    {\n"
     "        // TODO: Move these checks into the ewald module with the options class\n",
     "    if (usingPgm(ir->coulombtype))\n    {\n"
     '        sprintf(err_buf, "coulombtype = pGM needs periodic boundary conditions in x, y and z");\n'
     "        CHECK(ir->pbcType != PbcType::Xyz);\n"
     '        sprintf(err_buf, "coulombtype = pGM does not support free-energy perturbation yet");\n'
     "        CHECK(ir->efep != FreeEnergyPerturbationType::No);\n"
     '        sprintf(err_buf, "coulombtype = pGM: pgm-predictor must be 0, 1 or 2");\n'
     "        CHECK(ir->pgm_predictor < 0 || ir->pgm_predictor > 2);\n"
     '        sprintf(err_buf, "coulombtype = pGM: pgm-dipole-tol must be positive");\n'
     "        CHECK(ir->pgm_dipole_tol <= 0);\n"
     '        sprintf(err_buf, "coulombtype = pGM needs 4 <= pme-order <= 12");\n'
     "        CHECK(ir->pme_order < 4 || ir->pme_order > 12);\n"
     '        sprintf(err_buf, "coulombtype = pGM does not support epsilon-surface");\n'
     "        CHECK(ir->epsilon_surface != 0);\n    }\n\n"
     "    if (usingPme(ir->coulombtype) || usingLJPme(ir->vdwtype))\n    {\n"
     "        // TODO: Move these checks into the ewald module with the options class\n"),
    ("src/gromacs/gmxpreprocess/grompp.cpp",
     "    if (usingFullElectrostatics(ir->coulombtype) || usingLJPme(ir->vdwtype))\n    {\n"
     "        // We may have exclusion forces beyond the cut-off distance.\n",
     "    if ((usingFullElectrostatics(ir->coulombtype) && !usingPgm(ir->coulombtype)) || usingLJPme(ir->vdwtype))\n    {\n"
     "        // We may have exclusion forces beyond the cut-off distance (pGM electrostatics have no exclusions).\n"),
    # ---------------------------------------------------------------- mdrun: kernels and buffer
    ("src/gromacs/nbnxm/kerneldispatch.cpp",
     "    else if (coulombInteractionType == CoulombInteractionType::Fmm)\n    {\n",
     "    else if (coulombInteractionType == CoulombInteractionType::Pgm)\n    {\n"
     "        // LJ only: the pGM electrostatics are computed by src/gromacs/pgm over the same pair lists\n"
     "        return CoulombKernelType::None;\n    }\n"
     "    else if (coulombInteractionType == CoulombInteractionType::Fmm)\n    {\n"),
    ("src/gromacs/mdlib/forcerec.cpp",
     "        case CoulombInteractionType::Fmm:\n"
     '            GMX_RELEASE_ASSERT(false, "FMM is not yet supported by mdrun");\n            break;\n',
     "        case CoulombInteractionType::Fmm:\n"
     '            GMX_RELEASE_ASSERT(false, "FMM is not yet supported by mdrun");\n            break;\n\n'
     "        case CoulombInteractionType::Pgm:\n"
     "            forcerec->nbkernel_elec_interaction = NbkernelElecType::None;\n            break;\n"),
    ("src/gromacs/mdlib/calc_verletbuf.cpp",
     "    else if (usingPme(ir.coulombtype) || ir.coulombtype == CoulombInteractionType::Ewald)\n    {\n",
     "    else if (usingPme(ir.coulombtype) || ir.coulombtype == CoulombInteractionType::Ewald\n"
     "             || usingPgm(ir.coulombtype))\n    {\n"
     "        // pGM: the point-charge Ewald estimate (the Gaussian widths only soften the kernel)\n"),
]
