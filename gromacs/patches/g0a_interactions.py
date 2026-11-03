"""G0a: the pGM topology data: interaction functions PgmAtom (1 atom: radius, polarizability) and
PgmCovalentDipole (2 atoms: strength), directives [ pgm_atoms ] and [ pgm_dipoles ], tpr serialization.

Both are pure data lists for the pGM electrostatics (src/gromacs/pgm): IF_BOND so that domain
decomposition distributes them like bonded interactions, not chemical bonds (no exclusions), skipped by
the listed-force kernels.  PgmAtom's energy slot ("pGM Polariz.") carries the polarization energy.
"""

EDITS = [
    # ---------------------------------------------------------------- interaction functions
    ("api/legacy/include/gromacs/topology/ifunc.h",
     "    AnharmonicPolarization,\n",
     "    AnharmonicPolarization,\n"
     "    PgmAtom,           //!< pGM atom parameters (radius, polarizability); energy: pGM polarization\n"
     "    PgmCovalentDipole, //!< pGM covalent dipole p_i += c unit(r_j - r_i)\n"),
    ("src/gromacs/topology/ifunc.cpp",
     '    def_bondnb("ANHARM_POL", "Anharm. Pol.", 2, 3, 0),\n',
     '    def_bondnb("ANHARM_POL", "Anharm. Pol.", 2, 3, 0),\n'
     '    def_bonded("PGM_ATOM", "pGM Polariz.", 1, 2, 0),\n'
     '    def_bonded("PGM_DIPOLE", "pGM Cov. Dip.", 2, 1, 0),\n'),
    ("api/legacy/include/gromacs/topology/idef.h",
     "        real alpha, drcut, khyp;\n    } anharm_polarize;\n",
     "        real alpha, drcut, khyp;\n    } anharm_polarize;\n"
     "    struct\n    {\n        real radius, alpha;\n    } pgmAtom;\n"
     "    struct\n    {\n        real strength;\n    } pgmDipole;\n"),
    ("src/gromacs/topology/idef.cpp",
     "        case InteractionFunction::AnharmonicPolarization:\n"
     '            writer->writeLineFormatted("alpha=%15.8e drcut=%15.8e khyp=%15.8e",\n',
     "        case InteractionFunction::PgmAtom:\n"
     '            writer->writeLineFormatted("radius=%15.8e alpha=%15.8e", iparams.pgmAtom.radius, iparams.pgmAtom.alpha);\n'
     "            break;\n"
     "        case InteractionFunction::PgmCovalentDipole:\n"
     '            writer->writeLineFormatted("strength=%15.8e", iparams.pgmDipole.strength);\n'
     "            break;\n"
     "        case InteractionFunction::AnharmonicPolarization:\n"
     '            writer->writeLineFormatted("alpha=%15.8e drcut=%15.8e khyp=%15.8e",\n'),
    ("src/gromacs/gmxpreprocess/convparm.cpp",
     "        case InteractionFunction::Polarization: newparam->polarize.alpha = old[0]; break;\n",
     "        case InteractionFunction::Polarization: newparam->polarize.alpha = old[0]; break;\n"
     "        case InteractionFunction::PgmAtom:\n"
     "            newparam->pgmAtom.radius = old[0];\n"
     "            newparam->pgmAtom.alpha  = old[1];\n"
     "            break;\n"
     "        case InteractionFunction::PgmCovalentDipole: newparam->pgmDipole.strength = old[0]; break;\n"),
    # ---------------------------------------------------------------- tpr
    ("src/gromacs/fileio/tpxio.cpp",
     "        case InteractionFunction::Polarization: serializer->doReal(&iparams->polarize.alpha); break;\n",
     "        case InteractionFunction::Polarization: serializer->doReal(&iparams->polarize.alpha); break;\n"
     "        case InteractionFunction::PgmAtom:\n"
     "            serializer->doReal(&iparams->pgmAtom.radius);\n"
     "            serializer->doReal(&iparams->pgmAtom.alpha);\n"
     "            break;\n"
     "        case InteractionFunction::PgmCovalentDipole: serializer->doReal(&iparams->pgmDipole.strength); break;\n"),
    ("src/gromacs/fileio/tpxio.cpp",
     "    tpxv_AwhHistogramTolerance, /**< Add AWH histogram tolerance */\n",
     "    tpxv_AwhHistogramTolerance, /**< Add AWH histogram tolerance */\n"
     "    tpxv_PgmInteractions,       /**< pGM: PgmAtom / PgmCovalentDipole types, coulombtype pGM and its options */\n"),
    ("src/gromacs/fileio/tpxio.cpp",
     "    { tpxv_Pre96Version76, InteractionFunction::AnharmonicPolarization },\n",
     "    { tpxv_Pre96Version76, InteractionFunction::AnharmonicPolarization },\n"
     "    { tpxv_PgmInteractions, InteractionFunction::PgmAtom },\n"
     "    { tpxv_PgmInteractions, InteractionFunction::PgmCovalentDipole },\n"),
    ("src/gromacs/fileio/tpxio.cpp",
     "    AddNNPotIFuncType, //! ftupd changed to include NNPotIFuncType (bumped retrospectively for commit 152bde36).\n",
     "    AddNNPotIFuncType, //! ftupd changed to include NNPotIFuncType (bumped retrospectively for commit 152bde36).\n"
     "    AddPgmIFuncTypes,  //! ftupd changed to include the pGM types.\n"),
    # ---------------------------------------------------------------- listed forces: not computed there
    ("src/gromacs/listed_forces/bonded.cpp",
     "    BondedInteractions{ anharm_polarize<flavor>, eNR_ANHARM_POL }, // InteractionFunction::AnharmonicPolarization\n",
     "    BondedInteractions{ anharm_polarize<flavor>, eNR_ANHARM_POL }, // InteractionFunction::AnharmonicPolarization\n"
     "    BondedInteractions{ unimplemented, -1 }, // InteractionFunction::PgmAtom (src/gromacs/pgm)\n"
     "    BondedInteractions{ unimplemented, -1 }, // InteractionFunction::PgmCovalentDipole (src/gromacs/pgm)\n"),
    ("src/gromacs/listed_forces/utilities.h",
     "                || ftype == InteractionFunction::FlatBottomedPositionRestraints);\n",
     "                || ftype == InteractionFunction::FlatBottomedPositionRestraints\n"
     "                || ftype == InteractionFunction::PgmAtom || ftype == InteractionFunction::PgmCovalentDipole);\n"),
    ("api/nblib/listed_forces/conversionscommon.h",
     "                   NotInNblibButInGMX, //    InteractionFunction::AnharmonicPolarization,\n",
     "                   NotInNblibButInGMX, //    InteractionFunction::AnharmonicPolarization,\n"
     "                   NotInNblibButInGMX, //    InteractionFunction::PgmAtom,\n"
     "                   NotInNblibButInGMX, //    InteractionFunction::PgmCovalentDipole,\n"),
    # ---------------------------------------------------------------- topology directives
    ("src/gromacs/gmxpreprocess/topdirs.h",
     "    d_thole_polarization,\n",
     "    d_thole_polarization,\n    d_pgm_atoms,\n    d_pgm_dipoles,\n"),
    ("src/gromacs/gmxpreprocess/topdirs.cpp",
     '        "thole_polarization",\n',
     '        "thole_polarization",\n        "pgm_atoms",\n        "pgm_dipoles",\n'),
    ("src/gromacs/gmxpreprocess/topdirs.cpp",
     "        case Directive::d_thole_polarization: return InteractionFunction::TholePolarization;\n",
     "        case Directive::d_thole_polarization: return InteractionFunction::TholePolarization;\n"
     "        case Directive::d_pgm_atoms: return InteractionFunction::PgmAtom;\n"
     "        case Directive::d_pgm_dipoles: return InteractionFunction::PgmCovalentDipole;\n"),
    ("src/gromacs/gmxpreprocess/topdirs.cpp",
     "        set_nec(&(necessary[Directive::d_thole_polarization]), Directive::d_atoms, Directive::d_none);\n",
     "        set_nec(&(necessary[Directive::d_thole_polarization]), Directive::d_atoms, Directive::d_none);\n"
     "        set_nec(&(necessary[Directive::d_pgm_atoms]), Directive::d_atoms, Directive::d_none);\n"
     "        set_nec(&(necessary[Directive::d_pgm_dipoles]), Directive::d_atoms, Directive::d_none);\n"),
    ("src/gromacs/gmxpreprocess/topio.cpp",
     "                        case Directive::d_thole_polarization:\n                            GMX_RELEASE_ASSERT(\n",
     "                        case Directive::d_thole_polarization:\n"
     "                        case Directive::d_pgm_atoms:\n"
     "                        case Directive::d_pgm_dipoles:\n"
     "                            GMX_RELEASE_ASSERT(\n"),
]
