"""G3: the pGM GPU engine (core/cuda/pgm_cuda.cu, a static library built in src/gromacs/pgm) linked
into libgromacs in CUDA builds; selected at run time with GMX_PGM_GPU=1 (one rank, -nb cpu).  do_force
starts the GPU part (PgmElectrostatics::launch, on a worker thread) before the CPU nonbonded kernels, and
the usual hook (calculate) finishes it, so the GPU runs during the CPU short-range work."""

EDITS = [
    ("src/gromacs/CMakeLists.txt",
     "    target_compile_options(libgromacs PRIVATE \"$<$<COMPILE_LANGUAGE:CUDA>:${GMX_CUDA_FLAGS}>\")\n",
     "    target_compile_options(libgromacs PRIVATE \"$<$<COMPILE_LANGUAGE:CUDA>:${GMX_CUDA_FLAGS}>\")\n"
     "    if (TARGET pgm_cuda_engine)\n"
     "        target_link_libraries(libgromacs PRIVATE pgm_cuda_engine) # pGM on the GPU (src/gromacs/pgm)\n"
     "    endif()\n"),
    ("src/gromacs/mdlib/sim_util.cpp",
     "     */\n\n    const bool useOrEmulateGpuNb = simulationWork.useGpuNonbonded || fr->nbv->emulateGpu();\n",
     "     */\n\n"
     "    if (fr->pgm && stepWork.computeSlowForces)\n"
     "    {\n"
     "        // pGM on the GPU (GMX_PGM_GPU): start it now, the hook after the long-range part finishes it\n"
     "        fr->pgm->launch(x.unpaddedConstArrayRef(),\n"
     "                        box,\n"
     "                        mdatoms->homenr,\n"
     "                        mdatoms->chargeA,\n"
     "                        top->idef,\n"
     "                        top->excls,\n"
     "                        haveDDAtomOrdering(*cr) ? ArrayRef<const int>(cr->dd->globalAtomIndices)\n"
     "                                                : ArrayRef<const int>(),\n"
     "                        simulationWork.havePpDomainDecomposition ? cr->dd : nullptr,\n"
     "                        stepWork);\n"
     "    }\n\n"
     "    const bool useOrEmulateGpuNb = simulationWork.useGpuNonbonded || fr->nbv->emulateGpu();\n"),
    # trees patched before the launch call took the exclusions (GVDW on the GPU)
    ("src/gromacs/mdlib/sim_util.cpp",
     "                        top->idef,\n"
     "                        haveDDAtomOrdering(*cr) ? ArrayRef<const int>(cr->dd->globalAtomIndices)\n"
     "                                                : ArrayRef<const int>(),\n"
     "                        simulationWork.havePpDomainDecomposition ? cr->dd : nullptr,\n"
     "                        stepWork);\n",
     "                        top->idef,\n"
     "                        top->excls,\n"
     "                        haveDDAtomOrdering(*cr) ? ArrayRef<const int>(cr->dd->globalAtomIndices)\n"
     "                                                : ArrayRef<const int>(),\n"
     "                        simulationWork.havePpDomainDecomposition ? cr->dd : nullptr,\n"
     "                        stepWork);\n"),
]
