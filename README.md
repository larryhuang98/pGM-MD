# pGM-MD: polarizable Gaussian multipole electrostatics for AMBER, OpenMM and GROMACS

This repository holds the implementations of the polarizable Gaussian multipole (pGM) model for three
molecular dynamics programs. pGM uses Gaussian charges, covalent permanent dipoles and induced Gaussian
dipoles. Every pair interacts, with no exclusions or scaled 1-4 terms, and periodic systems use particle
mesh Ewald (PME) for charges and dipoles. Gaussian van der Waals (GVDW) is also available as an
alternative to Lennard-Jones.

The three programs share one set of conventions, so a pGM parameter set (an AMBER parameter-topology file)
gives the same energies and forces in each of them. `tests/xplat` checks this.

| Directory | Contents | License |
|---|---|---|
| [`amber/`](amber/README.md) | Patch for AMBER24 PMEMD: `pmemd.pgm` (CPU, MPI) and `pmemd.pgm.cuda` (NVIDIA GPUs, SPFP and DPFP) | applies to the licensed AMBER source; see `amber/LICENSE` |
| [`openmm/`](openmm/README.md) | OpenMM 8 plugin: `PGMForce` and `GVDWForce` on the Reference, CPU, OpenCL and CUDA platforms; Python reader for pGM prmtop files (`pgm_prmtop`) | MIT |
| [`gromacs/`](gromacs/README.md) | Patch for GROMACS 2026.4: `coulombtype = pGM`, GVDW, domain decomposition, pGM on the GPU | LGPL-2.1 (as GROMACS) |
| `core/` | Shared by OpenMM and GROMACS: C++ reference implementation (`pgm_core.cpp`) and CUDA engine (`cuda/pgm_cuda.cu`) | MIT |
| `tests/` | Unit tests, per-program test suites and the cross-implementation comparison; small inputs in `tests/data` | MIT |
| `docs/` | Model specification (`spec.md`) and design notes for the OpenMM and GROMACS implementations | MIT |

## Conventions shared by the implementations

* **Coulomb constant.** All three programs use 332.05382 kcal Å mol⁻¹ e⁻², the value used by the AMBER pGM
  code (138.931318 kJ nm mol⁻¹ e⁻²).
  * OpenMM: this is the default of `pgm_prmtop.create_system`.
  * GROMACS: its constant is fixed at compile time, so it is matched through the relative permittivity,
    `epsilon-r = 1.0000297942640506`. `gromacs/tools/prmtop_to_gmx.py` prints this value.
* **1-4 pairs.** The 1-4 electrostatics are part of the pGM sums. The 1-4 van der Waals is unscaled: full
  Lennard-Jones, or full GVDW when GVDW is used (AMBER `igvdw = 1`).
* **GVDW parameters.** The repulsion amplitude A and the dispersion coefficient C6 are read per atom type
  from the prmtop's `LENNARD_JONES_ACOEF` and `LENNARD_JONES_BCOEF`. These tables must follow the
  geometric combination rule.
* **Induction solver.** The convergence criterion is the largest α-scaled residual relative to the mean
  α-scaled permanent field (the tolerance τ). The initial guess is a cubic extrapolation of earlier dipoles.
  * AMBER CUDA tests convergence after every three conjugate-gradient iterations.
  * OpenMM and GROMACS test after every iteration.
  * At the same nominal τ, AMBER therefore stops at a slightly tighter accuracy.
* **Remaining difference.** The AMBER reciprocal-space influence function contains an additional factor.
  It changes the electrostatic energy by a few 10⁻⁷ of its value, depending on the grid spacing and the
  interpolation order.

## Building

Each program is built from its own directory; the sub-READMEs give the details.

```bash
# AMBER24 (source tree with the official updates applied)
amber/install.sh --amber-src /path/to/amber24_src --build --build-dir /path/to/amber24_src/build

# OpenMM plugin (OpenMM 8 from conda-forge; CUDA toolkit for the CUDA platform)
OPENMM_DIR=$CONDA_PREFIX openmm/build.sh

# GROMACS 2026.4
git clone --branch v2026.4 https://gitlab.com/gromacs/gromacs.git gromacs-2026.4
gromacs/install.sh gromacs-2026.4
gromacs/build.sh gromacs-2026.4 /opt/gromacs-pgm mixed     # CUDA, single precision (gmx)
gromacs/build.sh gromacs-2026.4 /opt/gromacs-pgm double    # CPU, double precision (gmx_d)
```

## Tests

`tests/env.sh` lists the tools that the test scripts use. Each one can be set in the environment: the
Python environments (`PY`, `OMM_PY`), the GROMACS binaries (`GMX_DOUBLE`, `GMX_MIXED`) and the AMBER
binaries (`AMBER_BIN`).

```bash
tests/core_tests.sh                       # C++ reference implementation
tests/openmm_tests.sh Reference CPU CUDA  # OpenMM plugin
tests/gromacs_tests.sh double cuda        # GROMACS patch
amber/test/run_tests.sh $AMBERHOME/bin    # AMBER patch
tests/xplat/run_xplat.sh all              # AMBER vs OpenMM vs GROMACS
```

`tests/xplat/run_xplat.sh` compares single points of the same coordinates for two systems (output of a full
run: `tests/xplat/reference_output.txt`):

* a liquid of 2,6-dimethylphenol (DMP), 8550 atoms;
* 512 pGM waters.

The reference is the AMBER CPU code, run in double precision with τ = 10⁻¹⁰. Typical deviations are:

| Implementation | ΔE, DMP (kcal/mol) | ΔE, 512 waters (kcal/mol) | Force deviation (relative RMS) |
|---|---|---|---|
| AMBER CUDA, DPFP | 0.0000 | 0.0000 | 10⁻⁷ |
| OpenMM CPU and CUDA, double precision; GROMACS CPU, double precision | +0.022 | −0.004 | 1–3 × 10⁻⁶ |
| AMBER CUDA SPFP; OpenMM CUDA mixed; GROMACS CUDA mixed | ≤ 0.01 | ≤ 0.05 | 10⁻⁵ |

The double-precision differences come from the reciprocal-space difference described above.

## Recommended settings (AMBER CUDA)

* **Thermostatted NVT and NPT simulations.** Use the SPFP build with a 2 fs time step and τ = 10⁻⁴.
* **NVE simulations that need tight energy conservation.** Use the DPFP build with a 1 fs time step.

## Citation

If you use this code, please cite the paper that describes the GPU implementation and the
cross-implementation comparison (Huang, Z. et al., in preparation; the reference will be added here).

The pGM formulation:

* H. Wei, R. Qi, J. Wang, P. Cieplak, Y. Duan, R. Luo, *J. Chem. Phys.* **153**, 114116 (2020),
  doi:10.1063/5.0019560
* H. Wei, P. Cieplak, Y. Duan, R. Luo, *J. Chem. Phys.* **156**, 114114 (2022), doi:10.1063/5.0082548

Methods and implementation of pGM in molecular dynamics:

* Z. Huang, S. Zhao, P. Cieplak, Y. Duan, R. Luo, H. Wei, Optimal scheme to achieve energy conservation in
  induced dipole models, *J. Chem. Theory Comput.* **19**, 5047–5057 (2023)
* Z. Huang, Y. Wu, Y. Duan, R. Luo, Performance tuning of polarizable Gaussian multipole model in molecular
  dynamics simulations, *J. Chem. Theory Comput.* **21**, 847–858 (2025), doi:10.1021/acs.jctc.4c01368
* Z. Huang, X. Wu, R. Luo, Isotropic periodic sum for polarizable Gaussian multipole model,
  *J. Chem. Theory Comput.* **21**, 4040 (2025)
* Z. Huang, R. Luo, Y. Duan, A density-consistent, finite van der Waals potential from Gaussian atomic
  densities, *J. Phys. Chem. Lett.* (2026)
