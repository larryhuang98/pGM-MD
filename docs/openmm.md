# OpenMM pGM plugin

A native OpenMM force, `PGMForce`, for pGM electrostatics (docs/spec.md): Gaussian charges, covalent
permanent dipoles and Gaussian induced dipoles, every pair interacting (no exclusions), smooth PME for
periodic systems; and `GVDWForce`, the Gaussian-density van der Waals of Huang, Luo and Duan
(pmemd-pgm `igvdw = 1`), as a native force like AMOEBA's `AmoebaVdwForce`.  Lennard-Jones van der Waals
stays in OpenMM's NonbondedForce (with the topology's exclusions).

| Platform | Implementation |
|---|---|
| Reference, CPU | `core/pgm_core.cpp` (double precision, O(N^2) pair search: for tests and small systems) |
| CUDA, OpenCL | Common kernels `openmm/platforms/common/src/kernels/pgm*.cc` (tile pair kernels on OpenMM's neighbour list, PME of charges + dipoles, CG solver on the device) |
| GVDW, Reference / CPU | `ReferenceGVDWKernels.cpp`: double precision, pairs of the atoms that carry GVDW, `core/gvdw_kernel.h` |
| GVDW, CUDA / OpenCL | `kernels/gvdw.cc`: a pair interaction in OpenMM's nonbonded tile kernel (NonbondedUtilities, with exclusions); dispersion correction on the host |

## Build (rayl8 cluster)

```bash
~/project/pgm-native/build_omm_login.sh          # login node, gcc 11 (or build_omm.sh on a compute node)
```

The conda OpenMM needs glibc 2.27; on the login node (glibc 2.17) run it through conda's glibc 2.28
sysroot: `runs/glibc228.sh python script.py` (Reference, CPU and pocl OpenCL; no GPU there).

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

### GVDW

```python
system, top, pgm = pgm_prmtop.create_system("box.prmtop", cutoff=0.8, ewald_beta=4.0, grid=(48, 48, 48),
                                            vdw="gvdw", gvdw="slater", lj_lrc=False)
```

replaces the Lennard-Jones by `GVDWForce` (force group 1): the pGM radii; pmemd-pgm's global
`gvdw_arep`, `gvdw_c6`, `b_rep_scale` and `gvdw_rep_form` on every atom with a nonzero LJ epsilon
(`gvdw="slater"` / `"gauss"`: the two water models of the GVDW manuscript, `pgm_prmtop.PGM3P_GVDW`;
or a dict `{"rep", "arep", "c6", "b"}`, or `{"rep", "types": {amber_type: (sqrtA, sqrtC6, b)}}` per atom
type); the prmtop's exclusions; scaled 1-4 Lennard-Jones pairs stay as NonbondedForce exceptions;
`lj_lrc` = Amber's vdwmeth = 1 for the C6 term.  By hand:

```python
g = openmmpgm.GVDWForce()
g.addParticle(radius_nm, sqrtA, sqrtC6, b)     # A_ij = a_i a_j, C6_ij = c_i c_j, B_ij = (b_i + b_j)/2
g.addExclusion(i, j)
g.setRepulsionForm(g.Slater)                   # A exp(-B y) (or g.Gaussian: A exp(-B y^2)), y = b_ij r
g.setNonbondedMethod(g.CutoffPeriodic); g.setCutoffDistance(0.8); g.setUseDispersionCorrection(True)
```

U_ij = A_ij exp(-B_ij y^(1|2)) - C6_ij F(y)/r^6 with b_ij = 1/sqrt(2 (R_i^2 + R_j^2)), the damping
F(y) = A(y)^2 + g(y)^2/2 finite at r = 0 (`GVDWForce.h`, `core/gvdw_kernel.h`).

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

GVDW (`tests/compare_gvdw.py`, the GVDW manuscript's 512-water box `rayl_512_v2.prmtop` +
`inputs/lj/inpcrd.restrt`, 10 A, no tail; `core/tests/test_gvdw.cpp` for the kernel):

| | Slater | Gaussian |
|---|---|---|
| Reference energy vs pmemd-pgm VDWAALS (kcal/mol) | 784.728185 vs 784.7282 | 868.369864 vs 868.3699 |
| forces vs finite differences (max, kJ/mol/nm; max force 790) | 6e-6 | 4e-6 |
| CPU vs Reference | identical | identical |
| OpenCL double vs Reference: energy, forces rms rel. | 1e-14, 3e-12 | 1e-15, 3e-12 |
| OpenCL mixed vs Reference | 3e-6, 3e-5 | 3e-6, 3e-5 |

Dispersion correction = -2 pi (sum c)^2/(3 V rc^3) to all digits.  NVE with pGM + GVDW (Slater), OpenCL
mixed (pocl), 1 fs, 0.8 nm, beta 4, 48^3, tol 1e-5, mu4, peek 0.65: total energy fluctuation 0.5 kJ/mol over
0.4 ps, 4 CG iterations per step (`runs/nve_gvdw.sh`).

## Known issues

* OpenMM's own Reference-platform NonbondedForce gives a Lennard-Jones energy 0.36 kJ/mol off in the
  truncated-octahedron box (CPU platform and brute force agree): compare on the CPU platform.
* pocl (OpenCL on the CPU, used for testing without a GPU: `runs/ocl_env.sh`) gives wrong double-precision
  FFTs, for OpenMM's own PME too; mixed and single precision are fine.
* Not implemented yet: quadrupoles, charge flux, DE van der Waals, GVDW in GROMACS, virtual sites, external fields; the strain derivative is computed by pgm_core but not exposed.
