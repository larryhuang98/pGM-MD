# pGM in GROMACS 2026.4: design and plan

Source: `~/project/gromacs-pgm` (git clone of v2026.4, branch `pgm`); build `build_gmx.sh` /
`rebuild.sh` (CUDA 12.8, sm_89 + sm_120, AVX2_256, own FFTW, internal BLAS/LAPACK).
The model is docs/spec.md; the references are pGM-JAX, the OpenMM plugin and `core/pgm_core.cpp`.

## What GROMACS assumes, and what pGM needs instead

| GROMACS | pGM |
|---|---|
| point charges in nbnxm kernels (`q`, `xq.w`, `epsfac` folded into q_i) | Gaussian charges, permanent + induced dipoles, kernel `G_n = B_n[erf(a r)/r] - B_n[erf(beta r)/r]` |
| one exclusion mask for Coulomb and LJ (topology exclusions + diagonal + FEP masking) | electrostatics: no exclusions; LJ: the topology's exclusions |
| Ewald exclusion correction (-erf(beta r)/r) for excluded pairs, self term `q^2 V(0)/2` in the kernels | Gaussian self terms: `-(beta/sqrt(pi)) q^2 - (2 beta^3/(3 sqrt(pi))) |d|^2`, no exclusion correction |
| PME of charges (theta, dtheta), one spread/solve/gather per grid | PME of charges + dipoles (needs d2theta for the dipole forces), several solves per step (CG) |
| one force evaluation per step (shells: `relax_shell_flexcon` calls do_force again) | permanent field, then 3-20 CG matrix-vector products (pair pass + PME of dipoles), then forces |

## Architecture

* **LJ** stays in the nbnxm kernels with `ElecType::None` (`CoulombKernelType::None`, LJ only, exclusions
  applied).  These kernels exist for FMM but are compiled only with `GMX_USE_EXT_FMM`; build them always.
* **pGM electrostatics** is a new module `src/gromacs/pgm/` called from `do_force` after the nbnxm LJ
  kernels (`do_nb_verlet`, sim_util.cpp ~2111-2156) and before the force reduction
  (`atomdata_add_nbat_f_to_f`, ~2157):
  1. permanent dipoles from the covalent list (local atoms, PBC as listed interactions; forces with shift
     forces);
  2. permanent field: pair pass over the nbnxm cluster lists (electrostatics ignore the masks except the
     diagonal `cj == ci_sh && j <= i`; padding atoms have zero parameters) + PME of q and p;
  3. predictor (mu4) + Jacobi PCG, each iteration a dipole-field pair pass + PME of the search direction;
     convergence max|alpha r| / mean|alpha E_perm| (one reduction);
  4. forces at fixed dipoles: pair pass (forces + total field), PME force gather with second
     derivatives, covalent chain rule (G1: all forces into `ForceWithVirial` with the analytical virial
     Xi = S^T/2; a SIMD kernel may put the pair forces into the shift-force buffer instead);
  5. energies: `CoulombSR` (direct), `CoulombReciprocalSpace` (PME + self + background), `Polarization`
     (|mu|^2/(2 alpha)); virial: shift forces for the pair part, PME mesh virial + the dipole term
     `sum_i E_rec,i (x) d_i`, background, covalent terms (all checked in pgm_core by finite differences).
* **PME**: extend `ewald/` with a pGM path: splines once per step (theta, dtheta, d2theta), then
  `spread(q, d) -> FFT -> solveCoulombYZX -> FFT -> gather(field)` per solve and a final
  `gather(field, field gradient)` for forces; reuse `PmeAndFftGrids`, `gmx_parallel_3dfft`, the
  thread-overlap reduction.  `computeAllSplineCoefficients` must be on (atoms with q = 0 and d != 0).
  PME load balancing (`gmx_pme_reinit`) and separate PME ranks: off for pGM at first
  (`disablePmeRanks`, the QMMM precedent).
* **Atom data**: per-atom `rg = 2 R^2`, `alpha`, dipoles and field accumulators next to `nbnxn_atomdata_t::Params`
  (filled like `nbnxn_atomdata_set_charges`, padded with zeros); `HasCharge` atom-info flag also for
  q = 0 atoms with a dipole or polarizability (or `includeAllPairs`), so their clusters stay in the list.
* **Domain decomposition** (later): halo exchange of the search direction / dipoles and reduction of the
  fields per CG iteration with `dd_move_x` / `dd_move_f`-style calls on rvec arrays.
* **GPU** (later): pGM kernels on the GPU pairlist after `k_calc_nb` in the same stream (the GPU FEP kernel
  `gpu_launch_free_energy_kernel` is the model), per-atom pGM buffers in `NBAtomDataGpu`, GPU PME with
  dipoles, CG vectors on the device.

## Input

* Gaussian charges are the ordinary atom charges (`t_atom.q`), so net charge, `mu_tot` and analysis tools
  work.
* New topology directives (per molecule type):
  ```
  [ pgm_atoms ]          ; atom  funct  radius(nm)  polarizability(nm^3)
  [ pgm_dipoles ]        ; ai  aj  funct  strength(e nm)      p_ai += c unit(r_aj - r_ai)
  ```
  stored as two new interaction functions, `PgmAtom` (1 atom, 2 parameters) and `PgmCovalentDipole`
  (2 atoms, 1 parameter, not a chemical bond), with listed-force kernels "unimplemented" (the NNPot
  pattern: ifunc.h/.cpp, idef.h, convparm.cpp, tpxio do_iparams + ftupd + TpxGeneration, bonded.cpp).
* mdp: `coulombtype = pGM` (new `CoulombInteractionType`), PME grid from `fourierspacing` / `fourier-nx`,
  `pme-order`, `ewald-rtol`, `rcoulomb`; solver options `pgm-dipole-tol`, `pgm-max-iter`,
  `pgm-predictor` (none / mu3 / mu4), `pgm-peek`.
* 1-4 pairs: LJ only (`fudgeQQ = 0`: pGM's all-pairs sum already contains the 1-4 electrostatics);
  exclusions as usual (they act on LJ only).
* Converter `tools/prmtop_to_gmx.py`: pmemd-pgm prmtop -> .top/.gro (ParmEd for the standard parts,
  POL_GAUSS sections -> the two directives, rigid water as settles).

## Phases

| | content | check |
|---|---|---|
| G0 | directives, interaction functions, tpr, mdp options, `coulombtype = pGM` accepted by grompp; LJ-only nbnxm kernels without FMM | grompp + gmx dump; LJ energy of water512 = pGM-JAX's vdW |
| G1 | mdrun CPU reference: pGM electrostatics over the nbnxm lists (scalar kernel), GROMACS PME extended to dipoles, CG, forces, energies, virial; single rank, OpenMP | water512 energies/forces/dipoles = OpenMM Reference / pGM-JAX; NVE drift; NPT density with C-rescale |
| G2 | SIMD (4xM) pGM kernels, threaded PME paths | same numbers, speed |
| G3 | CUDA: pGM pair kernels on the GPU list, GPU PME with dipoles, device CG | same numbers, speed vs pmemd.pgm.cuda |
| G4 | domain decomposition, separate PME ranks | multi-rank = single-rank |

## Status

### Patches and build

```
python3 gromacs/patches/apply.py gromacs/patches/g0a_interactions.py $SRC   # topology data, tpr
python3 gromacs/patches/apply.py gromacs/patches/g0b_coulombtype.py $SRC    # coulombtype = pGM, mdp options
bash    gromacs/patches/g0b_elecnone.sh $SRC                                # LJ-only nbnxm kernels always
bash    gromacs/patches/g1_install.sh $SRC                                  # src/gromacs/pgm + hooks (g1_hooks.py)
```

All stages are idempotent string replacements that fail loudly if upstream code moved.  Builds on
rayl8: `build_gmx.sh cuda` (mixed precision, CUDA, AVX2_256) and
`build_gmx.sh double -DGMX_DOUBLE=ON -DGMX_GPU=OFF` (`gmx_d`), incremental `rebuild.sh cuda|double`.

### G1 (done): CPU reference in mdrun

`src/gromacs/pgm/pgm.{h,cpp}` (`gmx::PgmElectrostatics`, owned by `t_forcerec::pgm`, created for
`coulombtype = pGM`) ports `core/pgm_core.cpp` to GROMACS's data structures:

* direct space: the local nbnxm CPU cluster lists (`pairlistSets().pairlistSet(Local).cpuLists()`,
  `getLocalAtomOrder()`), exclusion masks ignored, overlapping (diagonal) cluster pairs counted with
  nbnxm index j > i, pairs beyond rcoulomb skipped; kernels in double;
* PME of charges and dipoles with the GROMACS FFT (`gmx_parallel_3dfft`, one rank; the complex grid is
  y-major / x-minor as in `pme_solve`, whatever `complex_order` says), splines theta, dtheta, d2theta of
  order 4-12, influence function with OpenMM's B-spline moduli;
* Jacobi PCG with the mu3/mu4 predictor (history in the global atom order, so it survives the
  repartitioning that also happens on a single rank), peek step;
* outputs: forces into `ForceWithVirial` with the analytical virial Xi = S^T/2 (pairs, covalent chain
  rule, reciprocal including the dipole term, background); energies: Coulomb (SR) = direct, Coul. recip.
  = reciprocal + self + background, "pGM Polariz." = sum |mu|^2/(2 alpha);
* OpenMP: the i-cluster entries of all lists dealt out dynamically into per-thread pair vectors, per-thread
  field/force buffers, atoms (splines, gather), per-thread spread grids, the threaded `gmx_parallel_3dfft`,
  convolution; `-nb cpu`, one rank, no MTS (fatal errors otherwise).
* `GMX_PGM_DUMP=file`: the first force call's settings, terms, virial, per-atom positions, parameters,
  forces, dipoles and the covalent list, in the global order; `core/tests/run_dump` recomputes it with
  pgm_core, `tests/compare_gmx.py` compares it with a pGM-JAX reference.

Validation, water512 (pGM3P-25, 1536 atoms; beta 3.5/nm, grid 36^3, order 6, rc 0.9 nm, dipole tol 1e-10,
no predictor / peek; `tests/gmx/sp_w512.sh`, `runs/gmx_w512/rl10`: rlist 1.0 nm):

| | electrostatic energy | forces max abs diff (kJ/mol/nm; RMS force 1499) | dipoles max diff (e nm) | virial max diff (kJ/mol; max 5.6e4) |
|---|---|---|---|---|
| gmx_d vs pgm_core (same input) | 2e-15 relative | 2e-11 | 1e-16 | 6e-10 |
| gmx_d vs pGM-JAX (float64) | 3e-11 relative | 1e-4 (g96 positions, 1e-9 nm) | 7e-10 | |
| gmx (mixed) vs pgm_core | 2e-11 relative | 9e-4 | 4e-9 | 2e-3 |

CG iterations 17 in all three.  With `rlist = rcoulomb` and no Verlet buffer (`runs/gmx_w512/sp.mdp`) a
few pairs just inside the cut-off are missing from the lists (float bounding boxes): 1e-9 relative in
the energy, 0.09 kJ/mol/nm in the forces; any buffer removes it.

Dynamics (mixed precision, 8 OpenMP threads, `tests/gmx/nve_w512.sh`, `npt_w512.sh`):

* NVE, 1 fs, 4 ps, rigid water (SETTLE), plain LJ cut-off, Verlet buffer tolerance 1e-4, dipole tol
  1e-5, mu4 predictor, peek 0.65: total energy fluctuation 0.36 kJ/mol, drift -0.07 kJ/mol over 4 ps;
  4.0 CG iterations per step.
* NPT 298 K / 1 bar (`tests/gmx/npt.mdp`: 0.9 nm, beta 4/nm, 48^3 order 6, DispCorr EnerPres = the LJ
  tail, tol 1e-5, SETTLE, 2 fs; v-rescale, C-rescale), 7.4 ns/day, 6.0 CG iterations per step.  The box
  (`examples/water512/pgm3p25_512.prmtop`) is pGM-JAX's README box: pGM3P-25 electrostatics on TIP3P's
  geometry and LJ, whose reference density is 1.018 g/cm^3 (pGM-JAX, 14.9 ns, same box and settings,
  docs/dielectric.md); pmemd's 1.0097 is the paper's geometry and LJ, another model.  GROMACS, 750 ps,
  averages from 100 ps: density 1.0192 +- 0.0007 g/cm^3, pressure 27 +- 16 bar, T 298.2 K
  (`runs/gmx_w512/npt_ext.out`).
* GROMACS-side input checks: finite epsilon-r, no energy groups (one CoulombSR pair), every atom in
  `[ pgm_atoms ]`, no 1-4 electrostatics, `-nb cpu`, one rank, no MTS; polarizabilities <= 1e-9 nm^3
  count as zero.
* Speed of this reference (water512, 36^3 grid, 8 OpenMP threads, double build): 4.4 ns/day (19 ms per
  step at ~4 CG iterations: pair setup 4.7 ms, 6 field pair passes 2.2 ms, 7 PME solves 9.6 ms, final
  forces 1.4 ms; `GMX_PGM_TIMING=1` prints this breakdown).  The pair setup evaluates two erf kernels
  per pair in double; the PME is a plain per-thread-grid spread / FFT / gather.  Speed is G2's job.

### GVDW (done)

The Gaussian-density van der Waals of pmemd-pgm (`igvdw = 1`; OpenMM's `GVDWForce`), computed in
`src/gromacs/pgm` with the shared pair function `core/gvdw_kernel.h` (copied next to the module by
`g1_install.sh`; patch stage `g2_gvdw.py`):

* topology: `[ pgm_gvdw ]` (atom, funct 1, sqrtA [sqrt(kJ/mol)], sqrtC6 [sqrt(kJ/mol nm^6)], b) and
  `[ pgm_gvdw_pairs ]` (ai, aj, funct 1, A, C6, B: replaces the combination rule for that pair, at any
  distance, minimum image), interaction functions `PgmGvdwAtom` / `PgmGvdwPair`, tpr version
  `tpxv_PgmGvdw`; the radius is the atom's `[ pgm_atoms ]` radius, so GVDW needs coulombtype = pGM;
* mdp: `pgm-gvdw-rep-form` = 1 (Slater A exp(-B y), default) or 0 (Gaussian A exp(-B y^2)); `rvdw` is
  the GVDW cutoff (plain cut, as pmemd); `DispCorr` != no adds the C6 tail -2 pi (sum c)^2/(3 V rvdw^3)
  (Amber vdwmeth = 1) to GROMACS's LJ correction, with its virial and "Pres. DC" for EnerPres
  (sim_util's DispCorr now adds instead of assigning);
* pairs: the same nbnxm pair loop as the pGM direct sum, pairs within rvdw of atoms with a or c != 0,
  the topology's exclusions applied (as for LJ: `gmx_localtop_t::excls`), overridden pairs skipped;
  forces and virial into ForceWithVirial; energies: LJ (SR) (combination rule), "GVDW pairs"
  (overrides), Disper. corr. (tail);
* converter: `prmtop_to_gmx.py --gvdw slater|gauss [--gvdw-params A_kcal C6_kcalA6 b]` writes pmemd's
  global parameters as `[ pgm_gvdw ]` on every atom with LJ epsilon > 0 and zeroes the atom types'
  epsilon (molecules with 1-4 pairs not yet: their LJ would come from the zeroed types).

Validation (`tests/gmx/gvdw_sp.sh`, `gvdw_extra.sh`, `tests/compare_gmx_gvdw.py`; the GVDW manuscript's
512-water box, rvdw 1.0 nm, against an independent NumPy sum and pmemd-pgm):

| | Slater | Gaussian |
|---|---|---|
| gmx_d energy vs NumPy | 3e-11 relative | 6e-11 |
| gmx_d vs pmemd-pgm VDWAALS (kcal/mol) | -2e-5 | -4e-5 |
| gmx_d forces / virial max diff (kJ/mol/nm, kJ/mol) | 1e-8 / 2e-7 | 2e-8 / 4e-7 |
| gmx (mixed) energy vs NumPy | 9e-8 (float parameters in the tpr) | 4e-7 |
| gmx (mixed) forces max diff | 5e-5 | 9e-5 |

Overrides (an O-H pair of every water, A 300, C6 0.002, B 1.5) to 1e-15, the tail -88.19541348 kJ/mol =
formula = OpenMM's, Pres. DC = E/V.  NVE (mixed, pGM + GVDW Slater, 1 fs, 4 ps, tol 1e-5, mu4, peek 0.65):
total energy fluctuation 0.33 kJ/mol, drift -0.21 kJ/mol over 4 ps; 4.0 CG iterations; 4.6 ns/day.
Direct-space shifts are now built in double from the box (mixed builds: forces 25x closer to double).

### G2 plan (next)

Measured costs of the G1 reference (above) set the order:

1. **PME** (half the step): spread/gather of order 6 over 216 grid points per atom, 7 solves per step.
   Use GROMACS's `PmeAndFftGrids` thread-overlap decomposition (spread into thread-local subgrids with
   halo reduction, as `spread_on_grid`) instead of full per-thread grids; mixed precision on the grid;
   reuse the splines across the CG solves (already done) and gather field and field gradient in one pass.
2. **Pair setup**: evaluate G1, G2 (and G0, G3 for the final pass) in SIMD over the 4xM cluster pairs
   (`gmx::SimdReal`, erfc via `pmeForceCorrection`-style polynomials for beta and the exact series /
   recursion for the Gaussian a_ij), store them per cluster pair in the list order (float in mixed
   precision), so each CG pass is a streaming SIMD loop: load x_i, d_i (4 atoms), x_j, d_j (M atoms),
   G1, G2 tile, accumulate E_i in registers, E_j with the nbnxm j-force reduction pattern.
3. **CG vectors** in nbnxm atom order (no gathers through `getLocalAtomOrder` per pass).

Check: the same G1 validation (gmx_d against pgm_core to 1e-12, mixed against double), then speed against
pmemd.pgm (CPU) and OpenMM (CPU / pocl).
