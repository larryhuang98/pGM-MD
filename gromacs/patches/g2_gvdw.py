"""G2-GVDW: the Gaussian-density van der Waals (pmemd-pgm igvdw = 1) in GROMACS, computed by src/gromacs/pgm.

Topology (per molecule type):

    [ pgm_gvdw ]          ; atom  funct  sqrtA(sqrt(kJ/mol))  sqrtC6(sqrt(kJ/mol nm^6))  b
    [ pgm_gvdw_pairs ]    ; ai  aj  funct  A(kJ/mol)  C6(kJ/mol nm^6)  B     (replaces the pair's rule)

stored as interaction functions PgmGvdwAtom (1 atom, 3 parameters) and PgmGvdwPair (2 atoms, 3 parameters,
energy term "GVDW pairs").  The Gaussian radius is the atom's [ pgm_atoms ] radius.  Pairs within rvdw
interact through A_ij = a_i a_j, C6_ij = c_i c_j, B_ij = (b_i + b_j)/2 unless excluded (the topology's
exclusions, as for LJ) or overridden by [ pgm_gvdw_pairs ] (evaluated at any distance).  mdp:
pgm-gvdw-rep-form = 1 (Slater A exp(-B y), default) or 0 (Gaussian A exp(-B y^2)), pmemd's gvdw_rep_form.
DispCorr: the C6 tail -2 pi (sum c)^2/(3 V rvdw^3) (Amber vdwmeth = 1) is added to GROMACS's LJ correction.
"""

EDITS = [
    # ---------------------------------------------------------------- interaction functions
    ("api/legacy/include/gromacs/topology/ifunc.h",
     "    PgmCovalentDipole, //!< pGM covalent dipole p_i += c unit(r_j - r_i)\n",
     "    PgmCovalentDipole, //!< pGM covalent dipole p_i += c unit(r_j - r_i)\n"
     "    PgmGvdwAtom,       //!< GVDW atom parameters (sqrt A, sqrt C6, b)\n"
     "    PgmGvdwPair,       //!< GVDW pair parameters (A, C6, B) replacing the combination rule\n"),
    ("src/gromacs/topology/ifunc.cpp",
     '    def_bonded("PGM_DIPOLE", "pGM Cov. Dip.", 2, 1, 0),\n',
     '    def_bonded("PGM_DIPOLE", "pGM Cov. Dip.", 2, 1, 0),\n'
     '    def_bonded("PGM_GVDW", "GVDW atom", 1, 3, 0),\n'
     '    def_bonded("PGM_GVDW_PAIR", "GVDW pairs", 2, 3, 0),\n'),
    ("api/legacy/include/gromacs/topology/idef.h",
     "    struct\n    {\n        real strength;\n    } pgmDipole;\n",
     "    struct\n    {\n        real strength;\n    } pgmDipole;\n"
     "    struct\n    {\n        real sqrtA, sqrtC6, b;\n    } pgmGvdwAtom;\n"
     "    struct\n    {\n        real A, C6, B;\n    } pgmGvdwPair;\n"),
    ("src/gromacs/topology/idef.cpp",
     "        case InteractionFunction::PgmCovalentDipole:\n"
     '            writer->writeLineFormatted("strength=%15.8e", iparams.pgmDipole.strength);\n'
     "            break;\n",
     "        case InteractionFunction::PgmCovalentDipole:\n"
     '            writer->writeLineFormatted("strength=%15.8e", iparams.pgmDipole.strength);\n'
     "            break;\n"
     "        case InteractionFunction::PgmGvdwAtom:\n"
     '            writer->writeLineFormatted("sqrtA=%15.8e sqrtC6=%15.8e b=%15.8e", iparams.pgmGvdwAtom.sqrtA,\n'
     "                                       iparams.pgmGvdwAtom.sqrtC6, iparams.pgmGvdwAtom.b);\n"
     "            break;\n"
     "        case InteractionFunction::PgmGvdwPair:\n"
     '            writer->writeLineFormatted("A=%15.8e C6=%15.8e B=%15.8e", iparams.pgmGvdwPair.A,\n'
     "                                       iparams.pgmGvdwPair.C6, iparams.pgmGvdwPair.B);\n"
     "            break;\n"),
    ("src/gromacs/gmxpreprocess/convparm.cpp",
     "        case InteractionFunction::PgmCovalentDipole: newparam->pgmDipole.strength = old[0]; break;\n",
     "        case InteractionFunction::PgmCovalentDipole: newparam->pgmDipole.strength = old[0]; break;\n"
     "        case InteractionFunction::PgmGvdwAtom:\n"
     "            newparam->pgmGvdwAtom.sqrtA  = old[0];\n"
     "            newparam->pgmGvdwAtom.sqrtC6 = old[1];\n"
     "            newparam->pgmGvdwAtom.b      = old[2];\n"
     "            break;\n"
     "        case InteractionFunction::PgmGvdwPair:\n"
     "            newparam->pgmGvdwPair.A  = old[0];\n"
     "            newparam->pgmGvdwPair.C6 = old[1];\n"
     "            newparam->pgmGvdwPair.B  = old[2];\n"
     "            break;\n"),
    # ---------------------------------------------------------------- tpr
    ("src/gromacs/fileio/tpxio.cpp",
     "        case InteractionFunction::PgmCovalentDipole: serializer->doReal(&iparams->pgmDipole.strength); break;\n",
     "        case InteractionFunction::PgmCovalentDipole: serializer->doReal(&iparams->pgmDipole.strength); break;\n"
     "        case InteractionFunction::PgmGvdwAtom:\n"
     "            serializer->doReal(&iparams->pgmGvdwAtom.sqrtA);\n"
     "            serializer->doReal(&iparams->pgmGvdwAtom.sqrtC6);\n"
     "            serializer->doReal(&iparams->pgmGvdwAtom.b);\n"
     "            break;\n"
     "        case InteractionFunction::PgmGvdwPair:\n"
     "            serializer->doReal(&iparams->pgmGvdwPair.A);\n"
     "            serializer->doReal(&iparams->pgmGvdwPair.C6);\n"
     "            serializer->doReal(&iparams->pgmGvdwPair.B);\n"
     "            break;\n"),
    ("src/gromacs/fileio/tpxio.cpp",
     "    tpxv_PgmInteractions,       /**< pGM: PgmAtom / PgmCovalentDipole types, coulombtype pGM and its options */\n",
     "    tpxv_PgmInteractions,       /**< pGM: PgmAtom / PgmCovalentDipole types, coulombtype pGM and its options */\n"
     "    tpxv_PgmGvdw,               /**< pGM: PgmGvdwAtom / PgmGvdwPair types, pgm-gvdw-rep-form */\n"),
    ("src/gromacs/fileio/tpxio.cpp",
     "    { tpxv_PgmInteractions, InteractionFunction::PgmCovalentDipole },\n",
     "    { tpxv_PgmInteractions, InteractionFunction::PgmCovalentDipole },\n"
     "    { tpxv_PgmGvdw, InteractionFunction::PgmGvdwAtom },\n"
     "    { tpxv_PgmGvdw, InteractionFunction::PgmGvdwPair },\n"),
    ("src/gromacs/fileio/tpxio.cpp",
     "    AddPgmIFuncTypes,  //! ftupd changed to include the pGM types.\n",
     "    AddPgmIFuncTypes,  //! ftupd changed to include the pGM types.\n"
     "    AddPgmGvdwTypes,   //! ftupd changed to include the GVDW types.\n"),
    ("src/gromacs/fileio/tpxio.cpp",
     "        serializer->doInt(&ir->pgm_predictor);\n        serializer->doReal(&ir->pgm_peek);\n    }\n",
     "        serializer->doInt(&ir->pgm_predictor);\n        serializer->doReal(&ir->pgm_peek);\n    }\n"
     "    if (file_version >= tpxv_PgmGvdw)\n    {\n        serializer->doInt(&ir->pgm_gvdw_rep_form);\n    }\n"),
    # ---------------------------------------------------------------- listed forces: not computed there
    ("src/gromacs/listed_forces/bonded.cpp",
     "    BondedInteractions{ unimplemented, -1 }, // InteractionFunction::PgmCovalentDipole (src/gromacs/pgm)\n",
     "    BondedInteractions{ unimplemented, -1 }, // InteractionFunction::PgmCovalentDipole (src/gromacs/pgm)\n"
     "    BondedInteractions{ unimplemented, -1 }, // InteractionFunction::PgmGvdwAtom (src/gromacs/pgm)\n"
     "    BondedInteractions{ unimplemented, -1 }, // InteractionFunction::PgmGvdwPair (src/gromacs/pgm)\n"),
    ("src/gromacs/listed_forces/utilities.h",
     "                || ftype == InteractionFunction::PgmAtom || ftype == InteractionFunction::PgmCovalentDipole);\n",
     "                || ftype == InteractionFunction::PgmAtom || ftype == InteractionFunction::PgmCovalentDipole\n"
     "                || ftype == InteractionFunction::PgmGvdwAtom || ftype == InteractionFunction::PgmGvdwPair);\n"),
    ("api/nblib/listed_forces/conversionscommon.h",
     "                   NotInNblibButInGMX, //    InteractionFunction::PgmCovalentDipole,\n",
     "                   NotInNblibButInGMX, //    InteractionFunction::PgmCovalentDipole,\n"
     "                   NotInNblibButInGMX, //    InteractionFunction::PgmGvdwAtom,\n"
     "                   NotInNblibButInGMX, //    InteractionFunction::PgmGvdwPair,\n"),
    # ---------------------------------------------------------------- topology directives
    ("src/gromacs/gmxpreprocess/topdirs.h",
     "    d_pgm_atoms,\n    d_pgm_dipoles,\n",
     "    d_pgm_atoms,\n    d_pgm_dipoles,\n    d_pgm_gvdw,\n    d_pgm_gvdw_pairs,\n"),
    ("src/gromacs/gmxpreprocess/topdirs.cpp",
     '        "pgm_atoms",\n        "pgm_dipoles",\n',
     '        "pgm_atoms",\n        "pgm_dipoles",\n        "pgm_gvdw",\n        "pgm_gvdw_pairs",\n'),
    ("src/gromacs/gmxpreprocess/topdirs.cpp",
     "        case Directive::d_pgm_dipoles: return InteractionFunction::PgmCovalentDipole;\n",
     "        case Directive::d_pgm_dipoles: return InteractionFunction::PgmCovalentDipole;\n"
     "        case Directive::d_pgm_gvdw: return InteractionFunction::PgmGvdwAtom;\n"
     "        case Directive::d_pgm_gvdw_pairs: return InteractionFunction::PgmGvdwPair;\n"),
    ("src/gromacs/gmxpreprocess/topdirs.cpp",
     "        set_nec(&(necessary[Directive::d_pgm_dipoles]), Directive::d_atoms, Directive::d_none);\n",
     "        set_nec(&(necessary[Directive::d_pgm_dipoles]), Directive::d_atoms, Directive::d_none);\n"
     "        set_nec(&(necessary[Directive::d_pgm_gvdw]), Directive::d_atoms, Directive::d_none);\n"
     "        set_nec(&(necessary[Directive::d_pgm_gvdw_pairs]), Directive::d_atoms, Directive::d_none);\n"),
    ("src/gromacs/gmxpreprocess/topio.cpp",
     "                        case Directive::d_pgm_atoms:\n                        case Directive::d_pgm_dipoles:\n",
     "                        case Directive::d_pgm_atoms:\n                        case Directive::d_pgm_dipoles:\n"
     "                        case Directive::d_pgm_gvdw:\n                        case Directive::d_pgm_gvdw_pairs:\n"),
    # ---------------------------------------------------------------- mdp option
    ("api/legacy/include/gromacs/mdtypes/inputrec.h",
     "    real pgm_peek = 0.65;\n",
     "    real pgm_peek = 0.65;\n"
     "    //! GVDW repulsion: 0 Gaussian A exp(-B y^2), 1 Slater A exp(-B y) (pmemd-pgm gvdw_rep_form)\n"
     "    int pgm_gvdw_rep_form = 1;\n"),
    ("src/gromacs/gmxpreprocess/readir.cpp",
     '    ir->pgm_peek       = get_ereal(&inp, "pgm-peek", 0.65, wi);\n',
     '    ir->pgm_peek       = get_ereal(&inp, "pgm-peek", 0.65, wi);\n'
     '    ir->pgm_gvdw_rep_form = get_eint(&inp, "pgm-gvdw-rep-form", 1, wi);\n'),
    ("src/gromacs/gmxpreprocess/readir.cpp",
     "        CHECK(ir->pgm_predictor < 0 || ir->pgm_predictor > 2);\n",
     "        CHECK(ir->pgm_predictor < 0 || ir->pgm_predictor > 2);\n"
     '        sprintf(err_buf, "coulombtype = pGM: pgm-gvdw-rep-form must be 0 (Gaussian) or 1 (Slater)");\n'
     "        CHECK(ir->pgm_gvdw_rep_form < 0 || ir->pgm_gvdw_rep_form > 1);\n"),
    ("src/gromacs/mdtypes/inputrec.cpp",
     '        PR("pgm-peek", ir->pgm_peek);\n',
     '        PR("pgm-peek", ir->pgm_peek);\n        PI("pgm-gvdw-rep-form", ir->pgm_gvdw_rep_form);\n'),
    ("src/gromacs/mdtypes/inputrec.cpp",
     '    cmp_real(fp, "inputrec->pgm_peek", -1, ir1->pgm_peek, ir2->pgm_peek, ftol, abstol);\n',
     '    cmp_real(fp, "inputrec->pgm_peek", -1, ir1->pgm_peek, ir2->pgm_peek, ftol, abstol);\n'
     '    cmp_int(fp, "inputrec->pgm_gvdw_rep_form", -1, ir1->pgm_gvdw_rep_form, ir2->pgm_gvdw_rep_form);\n'),
    # ---------------------------------------------------------------- grompp: GVDW needs the pGM radii
    ("src/gromacs/gmxpreprocess/grompp.cpp",
     "    if (usingPgm(ir->coulombtype))\n    {\n"
     "        if (gmx_mtop_ftype_count(sys, InteractionFunction::PgmAtom) != sys.natoms)\n",
     "    if (!usingPgm(ir->coulombtype)\n"
     "        && (gmx_mtop_ftype_count(sys, InteractionFunction::PgmGvdwAtom) > 0\n"
     "            || gmx_mtop_ftype_count(sys, InteractionFunction::PgmGvdwPair) > 0))\n    {\n"
     '        wi.addError("[ pgm_gvdw ] and [ pgm_gvdw_pairs ] need coulombtype = pGM (the GVDW radii are the pGM radii)");\n'
     "    }\n"
     "    if (usingPgm(ir->coulombtype))\n    {\n"
     "        if (gmx_mtop_ftype_count(sys, InteractionFunction::PgmAtom) != sys.natoms)\n"),
    # ---------------------------------------------------------------- energy output
    ("src/gromacs/mdlib/energyoutput.cpp",
     "    bEner_[InteractionFunction::PgmCovalentDipole] = false; // pGM: data, the energy is elsewhere\n",
     "    bEner_[InteractionFunction::PgmCovalentDipole] = false; // pGM: data, the energy is elsewhere\n"
     "    bEner_[InteractionFunction::PgmGvdwAtom]       = false; // GVDW pairs are in LJ (SR)\n"),
    # ---------------------------------------------------------------- dispersion correction: add, not assign
    ("src/gromacs/mdlib/sim_util.cpp",
     "            enerd->term[InteractionFunction::DispersionCorrection] = correction.energy;\n",
     "            // += : the pGM module may have added the GVDW tail already (terms are reset every step)\n"
     "            enerd->term[InteractionFunction::DispersionCorrection] += correction.energy;\n"),
    ("src/gromacs/mdlib/sim_util.cpp",
     "            enerd->term[InteractionFunction::PressureDispersionCorrection] = correction.pressure;\n",
     "            enerd->term[InteractionFunction::PressureDispersionCorrection] += correction.pressure;\n"),
    # ---------------------------------------------------------------- the exclusions for the module
    ("src/gromacs/mdlib/sim_util.cpp",
     "                               top->idef,\n                               fr->shift_vec,\n",
     "                               top->idef,\n                               top->excls,\n                               fr->shift_vec,\n"),
]
