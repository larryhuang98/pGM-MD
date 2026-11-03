# OpenMM pGM plugin

A native OpenMM force, `PGMForce`, for pGM electrostatics (docs/spec.md): Gaussian charges, covalent
permanent dipoles and Gaussian induced dipoles, every pair interacting (no exclusions), smooth PME for
periodic systems.  Van der Waals stays in OpenMM's own forces (with the topology's exclusions).

| Platform | Implementation |
|---|---|
| Reference, CPU | `core/pgm_core.cpp` (double precision, O(N^2) pair search: for tests and small systems) |
| CUDA, OpenCL | Common kernels `openmm/platforms/common/src/kernels/pgm*.cc` (tile pair kernels on OpenMM's neighbour list, PME of charges + dipoles, CG solver on the device) |

## Build (rayl8 cluster)

```bash
# on a compute node (the login node's glibc is too old for the conda OpenMM)
~/project/pgm-native/build_omm.sh -DPGM_BUILD_COMMON=ON
```

Installs `libOpenMMPGM.so`, the platform plugins `libOpenMMPGM{Reference,CUDA,OpenCL}.so` and the
Python modules `openmmpgm` (the force) and `pgm_prmtop` (a pmemd-pgm prmtop reader) into the
`pgm-omm` conda environment (`~/miniconda3/envs/pgm-omm`, OpenMM 8.6.1).

## Use

```python
import openmm as mm, openmm.app as app, openmm.unit as u
import pgm_prmtop

system, top, pgm = pgm_prmtop.create_system("water.prmtop", cutoff=0.9, ewald_beta=3.5, grid=(36, 36, 36),
                                            dipole_tol=1e-5, predictor="mu4", peek=0.65)
crd = app.AmberInpcrdFile("water.rst7")
sim = app.Simulation(top, system, mm.LangevinMiddleIntegrator(298*u.kelvin, 1/u.picosecond, 1*u.femtosecond),
                     mm.Platform.getPlatformByName("CUDA"), {"Precision": "mixed"})
sim.context.setPositions(crd.positions); sim.context.setPeriodicBoxVectors(*crd.boxVectors)
sim.step(1000)
mu = pgm.getInducedDipoles(sim.context)          # (N, 3) e nm
```

`create_system` builds, from the prmtop: bonded terms, masses and constraints (AmberPrmtopFile);
`PGMForce` from the `POL_GAUSS_*` sections; the Lennard-Jones as AmberPrmtopFile's NonbondedForce with
all charges zero (keeps the prmtop's exclusions and 1-4 LJ), cut at the cutoff, no switch; and Amber's
vdwmeth=1 tail correction (r^-6 term only, as pGM-JAX) as a CustomVolumeForce.  Force groups: pGM 0,
LJ 1.

`PGMForce` by hand:

```python
f = openmmpgm.PGMForce()
i = f.addParticle(charge, radius_nm, polarizability_nm3)      # alpha = 0: not polarizable
f.addCovalentDipole(i, j, strength_e_nm)                       # p_i += c unit(r_j - r_i)
f.setNonbondedMethod(f.PME); f.setCutoffDistance(0.9); f.setPMEParameters(3.5, 36, 36, 36); f.setPMEOrder(6)
f.setPolarization(f.Mutual)            # or f.NoPolarization (levels q / qp)
f.setInducedDipoleTolerance(1e-5)      # max|alpha r| / mean|alpha E_perm|, componentwise (pmemd's criterion)
f.setPredictor(f.Mu4); f.setPeekCoefficient(0.65)
f.setCoulombConstant(332.05382*0.4184) # optional: pmemd-pgm's constant (default ONE_4PI_EPS0)
```

A grid given explicitly is used as given; with alpha = 0 the Ewald coefficient and grid follow the Ewald
error tolerance (NonbondedForce's rule).

## Validation (tests/)

512 pGM3P-25 waters, truncated octahedron, cutoff 0.9 nm, beta 3.5 /nm, grid 36^3, order 6, one
configuration, against pGM-JAX (`ref_pgmjax.py` + `compare_openmm.py`; Coulomb constant set to pGM-JAX's
138.935458):

| level | elec. energy, rel. diff. | forces max diff (kJ/mol/nm) | induced dipoles max diff (e nm) | CG iterations |
|---|---|---|---|---|
| qpi | < 1e-12 | 2e-10 | 1e-15 | 17 / 17 |
| q | 5e-15 | 3e-10 | – | – |
| qp | 1e-14 | 2e-10 | – | – |
| qi | 7e-15 | 2e-10 | 1e-15 | 17 / 17 |

LJ + tail correction agree with pGM-JAX (to OpenMM CPU's single-precision LJ, 3e-4 kJ/mol of 2800).
Common kernels against the Reference kernel (`compare_platforms.py`), mixed precision: energies 1-2e-7
relative, forces 1e-5 relative (rms), induced dipoles 2e-5 relative, same CG iteration count; no-cutoff
cases in double precision to 1e-15.  NVE (`md_nve.py`, mixed, 1 fs, SETTLE, mu4, tol 1e-5, peek 0.65):
total energy fluctuation 0.3 kJ/mol in 2.1e6, drift -0.7 kJ/mol/ps, 4 CG iterations per step.
`core/tests/test_core.cpp`: forces and the strain derivative against finite differences.

## Known issues

* OpenMM's own Reference-platform NonbondedForce gives a Lennard-Jones energy 0.36 kJ/mol off in the
  truncated-octahedron box (CPU platform and brute force agree): compare on the CPU platform.
* pocl (OpenCL on the CPU, used for testing without a GPU: `runs/ocl_env.sh`) gives wrong double-precision
  FFTs, for OpenMM's own PME too; mixed and single precision are fine.
* Not implemented yet: quadrupoles, charge flux, GVDW / DE van der Waals (CustomNonbondedForce planned),
  virtual sites, external fields; the strain derivative is computed by pgm_core but not exposed.
