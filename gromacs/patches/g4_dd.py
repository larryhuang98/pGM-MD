"""G4: pGM with domain decomposition (several PP ranks, thread-MPI or MPI).

The module gets the DD (gmx_domdec_t, nullptr on one rank) and does its own halo communication of
dipoles, fields and forces with the DD communication setup; separate PME ranks and MTS stay
unsupported (fatal errors in g1_hooks.py's forcerec block).  Apply after g1_hooks.py and g2_gvdw.py."""

EDITS = [
    # ---------------------------------------------------------------- do_force: pass the DD
    ("src/gromacs/mdlib/sim_util.cpp",
     "                               haveDDAtomOrdering(*cr) ? ArrayRef<const int>(cr->dd->globalAtomIndices)\n"
     "                                                       : ArrayRef<const int>(),\n"
     "                               &forceOutMtsLevel1->forceWithVirial(),\n",
     "                               haveDDAtomOrdering(*cr) ? ArrayRef<const int>(cr->dd->globalAtomIndices)\n"
     "                                                       : ArrayRef<const int>(),\n"
     "                               simulationWork.havePpDomainDecomposition ? cr->dd : nullptr,\n"
     "                               &forceOutMtsLevel1->forceWithVirial(),\n"),
]
