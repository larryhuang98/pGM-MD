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
* **GPU** (G3, done): the whole pGM electrostatics and the combination-rule GVDW in a separate engine,
  `core/cuda/pgm_cuda` (own pair list, PME of charges and dipoles with cuFFT, CG on the device, one CUDA
  graph per step), started by `do_force` before the CPU short-range kernels on a worker thread and
  collected at the pGM hook; LJ stays on the CPU (`-nb cpu`).  See G3 below.

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
| G3 | CUDA: the pGM electrostatics on the GPU (own pair list, PME with dipoles, device CG), overlapped with the CPU short-range work | same numbers, speed vs pmemd.pgm.cuda |
| G4 | domain decomposition, separate PME ranks | multi-rank = single-rank |

## Status

### Patches and build

```
python3 gromacs/patches/apply.py gromacs/patches/g0a_interactions.py $SRC   # topology data, tpr
python3 gromacs/patches/apply.py gromacs/patches/g0b_coulombtype.py $SRC    # coulombtype = pGM, mdp options
bash    gromacs/patches/g0b_elecnone.sh $SRC                                # LJ-only nbnxm kernels always
bash    gromacs/patches/g1_install.sh $SRC                                  # src/gromacs/pgm + hooks (g1_hooks.py,
                                                                            # g1b_checks.py, g2_gvdw.py, g4_dd.py,
                                                                            # g3_gpu.py; core/cuda in CUDA builds)
```

All stages are idempotent string replacements that fail loudly if upstream code moved.  Builds on
rayl8: `build_gmx.sh cuda` (mixed precision, CUDA, AVX2_256) and
`build_gmx.sh double -DGMX_DOUBLE=ON -DGMX_GPU=OFF` (`gmx_d`), incremental `rebuild.sh cuda|double`.

### G1 (done): CPU reference in mdrun

`src/gromacs/pgm/pgm.{h,cpp}` (`gmx::PgmElectrostatics`, owned by `t_forcerec::pgm`, created for
`coulombtype = pGM`) ports `core/pgm_core.cpp` to GROMACS's data structures:

* direct space: the local nbnxm CPU cluster lists (`pairlistSets().pairlistSet(Local).cpuLists()`,
  `getLocalAtomOrder()`), exclusion masks ignored, overlapping (diagonal) cluster pairs counted with
  nbnxm index j > i, pairs beyond rcoulomb skipped; kernels in double (G2 below for how);
* PME of charges and dipoles with the GROMACS FFT (`gmx_parallel_3dfft`, one rank; the complex grid is
  y-major / x-minor as in `pme_solve`, whatever `complex_order` says), splines theta, dtheta, d2theta of
  order 4-12, influence function with OpenMM's B-spline moduli;
* Jacobi PCG with the mu3/mu4 predictor (history in the global atom order, so it survives the
  repartitioning that also happens on a single rank), peek step;
* outputs: forces into `ForceWithVirial` with the analytical virial Xi = S^T/2 (pairs, covalent chain
  rule, reciprocal including the dipole term, background); energies: Coulomb (SR) = direct, Coul. recip.
  = reciprocal + self + background, "pGM Polariz." = sum |mu|^2/(2 alpha);
* OpenMP: see G2; domain decomposition: see G4; `-nb cpu`, no separate PME ranks (`-npme 0`), no MTS
  (fatal errors otherwise).
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
* Speed of the first G1 version (water512, 36^3 grid, 8 OpenMP threads, double build): 4.4 ns/day (19 ms
  per step at ~4 CG iterations: pair setup 4.7 ms, 6 field pair passes 2.2 ms, 7 PME solves 9.6 ms,
  final forces 1.4 ms; `GMX_PGM_TIMING=1` prints this breakdown).  G2 below.

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
  global parameters (full double precision) as `[ pgm_gvdw ]` on every atom with LJ epsilon > 0, writes
  the 1-4 Lennard-Jones parameters of every `[ pairs ]` line explicitly (from gen-pairs/fudgeLJ/comb-rule
  or `[ pairtypes ]`) and then zeroes the epsilon of the atom types and `[ nonbond_params ]`: the 1-4 LJ
  stays as it was, GVDW acts on the non-excluded pairs, as in pmemd-pgm.

Validation (`tests/gmx/gvdw_sp.sh`, `gvdw_extra.sh`, `tests/compare_gmx_gvdw.py`; the GVDW manuscript's
512-water box, rvdw 1.0 nm, against an independent NumPy sum and pmemd-pgm):

| | Slater | Gaussian |
|---|---|---|
| gmx_d energy vs NumPy | 3e-11 relative | 6e-11 |
| gmx_d vs pmemd-pgm VDWAALS (kcal/mol) | -2e-5 | -4e-5 |
| gmx_d forces / virial max diff (kJ/mol/nm, kJ/mol) | 1e-8 / 2e-7 | 2e-8 / 4e-7 |
| gmx (mixed) energy vs NumPy | 9e-8 (float parameters in the tpr) | 4e-7 |
| gmx (mixed) forces max diff | 5e-5 | 9e-5 |

Molecules with 1-4 pairs (`tests/gmx/gvdw_14.sh` + `gvdw_14_compare.py`: the Amber pGM test molecule
qm7b_7203, 20 atoms, 46 1-4 pairs, alone in a 4 nm box, rvdw 1.2 nm, converted with LJ, Slater and Gauss
GVDW, against OpenMM's `pgm_prmtop` NoCutoff on the same coordinates): GVDW = LJ (SR) to 1e-13 kJ/mol,
LJ-14 identical in all three conversions and = OpenMM's 1-4 LJ to 2e-6 (ParmEd's 8-digit parameters), bonds,
angles, dihedrals to 1e-6, Coulomb-14 = 0.

Overrides (an O-H pair of every water, A 300, C6 0.002, B 1.5) to 1e-15, the tail -88.19541348 kJ/mol =
formula = OpenMM's, Pres. DC = E/V.  NVE (mixed, pGM + GVDW Slater, 1 fs, 4 ps, tol 1e-5, mu4, peek 0.65):
total energy fluctuation 0.33 kJ/mol, drift -0.21 kJ/mol over 4 ps; 4.0 CG iterations; 4.6 ns/day.
Direct-space shifts are now built in double from the box (mixed builds: forces 25x closer to double).

### G2 (done): CPU speed

Water512, 8 OpenMP threads, `-pin on` (the node's logical CPUs are interleaved over 4 NUMA nodes;
unpinned threads cost ~30%), `GMX_PGM_TIMING=1`:

| | G1 | G2 |
|---|---|---|
| NVE 1 fs, 36^3, double (`nve_w512.sh double`) | 4.4 ns/day | 11.1 ns/day |
| NVE 1 fs, 36^3, mixed | | 11.0 ns/day |
| NPT 2 fs, 48^3, mixed (`npt_w512.sh`) | 7.4 ns/day | 15.4 ns/day |
| ms per force call (double): pair setup / field passes / PME / final forces | 4.7 / 2.2 / 9.6 / 1.4 | 1.5 / 1.5 / 3.0 / 0.9 |

What changed:

1. **Pair candidates cached per pair search.**  At `doNeighborSearch` steps every atom pair of the nbnxm
   cluster pairs (not only those within the atomic rlist: the cluster bounding boxes are part of the Verlet
   buffer, filtering them out raised the mixed NVE drift from -0.07 to 1.0 kJ/mol / 4 ps; with dynamic
   pruning the *outer* list `ciOuter`/`cjOuter`, since `ci`/`cj` are re-pruned every few steps) is stored once,
   in segments of one i atom and one shift, with its GVDW eligibility (exclusions and overrides checked
   there, not at every step).  Each step compacts the pairs within rc into structure-of-arrays segments.
2. **SIMD kernels.**  G_0..G_3 in the erfc form Bc(beta) - Bc(a) (`pgmcore::ewaldPairKernels`, now also
   pgm_core's and the OpenMM kernels' form) with GROMACS's double-precision SIMD `erfc`/`exp`
   (AVX2: 4 pairs), the few pairs with a r < 1 (bonded neighbours) again from the series.
3. **Pair passes** run over the i-segments: atom i's field and force in registers, j scattered.
4. **PME**: grid indices precomputed per atom (no modulo in the inner loops); the z dimension padded by
   order - 1 so an atom's z points are contiguous (z splines stored in that order); owner-computes
   spreading over tiles of (x, y) columns at least `order` wide (each atom listed in its <= 4 tiles; no
   per-thread grids, no reduction, no colouring), the padding folded into the FFT grid; gathers with
   contiguous z.

Accuracy unchanged: gmx_d vs pgm_core 1e-15 relative energy, 4e-11 forces, 3e-16 dipoles; mixed 3e-11 /
9e-4; GVDW to 1e-16 (NumPy), pmemd -2e-5 / -4e-5 kcal/mol; NVE mixed 4 ps, three runs (the trajectory
depends on the threads' summation order): total-energy fluctuation 0.37-0.52 kJ/mol, drift -0.3 to +0.6
kJ/mol, within what the Verlet buffer tolerance (1e-4 kJ/mol/ps per atom: 0.6 kJ/mol) allows.

### G4 (done): domain decomposition

Several PP ranks (thread-MPI or MPI, `-npme 0`), `gromacs/patches/g4_dd.py` passes the `gmx_domdec_t`:

* local atoms = home atoms + the halo of the DD zones (coordinates and charges communicated by GROMACS);
  the candidates come from the local *and* non-local nbnxm lists, so every pair within rc is on exactly
  one rank, with halo atoms as i or j;
* the module's own halo communication with the DD setup (`comm->cd[].ind[]`, as `dd_move_x` /
  `dd_move_f`, double precision, own MPI tag): radii and GVDW parameters (one-atom interactions live on
  the owner) and the dipoles before every field pass are broadcast to the halo; the fields of every pass,
  the permanent dipoles (a covalent dipole may be assigned to a rank where its atom is halo), the forces
  and the GVDW forces are reduced to the owners; the total field is broadcast once more for the covalent
  chain rule;
* CG on the home atoms with global dot products and maxima; the predictor history and the "same
  positions" test in the global atom order, summed over the ranks, so atoms can migrate;
* PME: each rank spreads its home atoms, the grid is summed over the ranks, every rank does the FFT (fine
  for one node; a decomposed FFT / PME ranks is the next step for many nodes); k-space energy and strain,
  the background term and the GVDW tail are counted on the main rank, everything per atom on the owner;
* GVDW overrides: the override list of all ranks by global index (an overridden pair's interaction and
  its place in the pair lists can be on different ranks);
* not supported: screw PBC, the direct halo exchange (`GMX_FILLERS_IN_LOCAL_STATE`), `GMX_PGM_DUMP` with
  DD (use one rank).

Validation (`tests/gmx/dd_sp.sh` + `dd_compare.py`; water512 replicated 2x2x2 = 4096 waters, grid 72^3,
dipole tol 1e-10): energies (every term incl. virial and pressure) and forces against one rank:

| DD grid | pulses | max energy-term rel. diff | max force diff (kJ/mol/nm; max 3488) |
|---|---|---|---|
| 2x1x1, 4x1x1, 4x2x1 | 1 | 6e-12 | 4e-10 |
| 8x1x1 (0.56 nm cells) | 3 | 5e-12 | 1e-10 |
| 2x2x2 | 1 | 5e-12 | 4e-10 |
| 3x2x1 | 1 | 9e-12 | 4e-10 |
| GVDW + overrides + tail, 4 ranks | 1 | 7e-11 (Pres-XY) | 1e-10 |

The potential agrees to 1e-15 relative; the largest relative differences are in small off-diagonal
pressure components.  NVE (4096 waters, 2 ps, 1 fs, tol 1e-5, mu4, double): total-energy fluctuation
1.11 kJ/mol and drift 0.8 kJ/mol on 1 rank, 1.24 and 1.7 kJ/mol on 4 ranks (the Verlet buffer tolerance
allows 2.5 kJ/mol here).  Started from MD coordinates the trajectories of 1 and n ranks differ from the
first step by ~4e-3 kJ/mol: the waters (update groups) stick out of the box by up to 0.08 nm and the
nbnxm lists then miss a few pairs just inside rc at the search, different ones for different
decompositions (`GMX_PGM_CHECK_PAIRS=1`: 8 of 1855136 pairs on 1 rank, 3 on 4 ranks); pGM uses
GROMACS's pair set, as GROMACS's own kernels do.

Speed (4096 waters, NVE 1 fs, 72^3 grid, mixed build, pinned, ns/day): 1 rank x 8 threads 1.18,
x 16 threads 1.17 (OpenMP saturated); 2 x 4: 0.97, 4 x 2: 0.68, 8 x 1: 0.44, 4 x 4: 0.76.  Correct but
not yet faster: per force call (4 x 2) pair setup 19 ms and field passes 24 ms are as on one rank (the
same 8 cores), the PME takes 75 instead of 20 ms: every rank sums the whole grid (3.2 MB, 7 times per
step) and does the whole FFT with its few threads.  A decomposed PME (below) is what DD needs to pay off.

`GMX_PGM_CHECK_PAIRS=1` counts, every step, the pairs within rc on all ranks against all pairs of the
global coordinates (O(N^2), for testing) and lists missing ones on one rank.

### G3 (done): the GPU

`GMX_PGM_GPU=1` with a CUDA build, one rank, `-nb cpu`: the pGM electrostatics and the combination-rule
GVDW run in `core/cuda/pgm_cuda.{h,cu}` (`pgmcuda::Engine`, also usable outside GROMACS), compiled as a
static library next to the module (libgromacs's CUDA flags forbid double precision; `g3_gpu.py` links
it).  GVDW overrides and the dispersion tail, LJ, bonded terms and everything else stay on the CPU path.

* **Pairs**: the engine's own list instead of the nbnxm lists (whose transfer cost more than the whole
  GPU step): a cell list on the GPU (cells rl/2 wide, atoms sorted by cell with a stable radix sort),
  per atom the full list of the atoms within rl = max(rc, rvdw) + skin (0.15 nm, less in small boxes;
  minimum image), built with one block per cell and neighbour slab (the neighbour atoms read once per
  cell through shared memory; deterministic order), rebuilt when an atom has moved by more than skin/2
  or the box changed (`GMX_PGM_GPU_SKIN`).  GROMACS's reordering at its pair search only renames the
  list (rows moved, partners renamed on the GPU).  The displacement check is the first kernel of the
  step's graph: the host predicts from the last step (its displacement + twice its largest move) whether
  to rebuild before the step, and rebuilds and repeats the evaluation should the check fail after all
  (the history has a fifth slot, so a repeat reads what the first run read; 0 repeats in the benchmark).
  Each step one pass compacts the pairs within the radius (warp ballot) with their kernels G1, G2,
  together with the first field pass; the final pass computes all four itself.  One warp per atom over
  its full list: no atomics.
* **GVDW** (`Engine::setGvdw`): per-atom sqrt A, sqrt C6, b, the excluded and the overridden pairs as
  per-atom skip lists, marked once per list build in the partner index; the pair (core/gvdw_kernel.h in
  float) in the final pass.
* **Precision** as OpenMM's mixed mode: pair kernels, splines and grids in float, PME grid and covalent
  forces in 64-bit fixed point (2^32), the CG vectors, dipoles, energies and the strain in double.
* **PME**: splines once per step (one thread per atom and dimension, the order a template parameter);
  spread one warp per atom over the p^3 points with the spline values in lanes (shuffles); cuFFT R2C/C2R;
  gather one thread per atom with the splines in registers (the instantiation with the second
  derivatives is the faster one on sm_120, so every gather uses it).
* **CG**: Jacobi PCG over many blocks with atomic reductions, the scalars and the loop condition in the
  last block of a reduction kernel; the mu3/mu4 predictor history on the device in the global order;
  peek.  When the guess comes from the history, the permanent field and the guess's field are one pair
  pass, one spread into two grids and one batched FFT pair.
* **One CUDA graph per step** (CUDA >= 12.4; `PGM_CUDA_NO_GRAPH=1` runs the same kernels with a host
  loop): pairs, the first fields, the CG loop as a conditional while node, peek, history, final pass,
  reductions, copies to pinned host memory; the box in constant memory, so the graph is captured again
  only when buffers grow.  The strain derivative only on virial steps.
* **Overlap**: `PgmElectrostatics::launch` (a second hook in `do_force`, before the CPU nonbonded
  kernels) starts the evaluation on a worker thread; `calculate` at the usual hook waits for it and adds
  forces, virial and energies.  Both hand-overs spin before they block (a sleeping thread's wake-up
  cost up to 0.1 ms).  The positions go from GROMACS's float array (registered with CUDA once) straight
  into the graph (`Engine::computeF`), the forces are read from the engine's pinned buffer.

Validation (`tests/gmx/gpu_check.sh`, GPU against the CPU path of the same mixed build; relative
differences, float output):

| single point | LJ (SR) / GVDW | Coulomb (SR) | Coul. recip. | pGM Polariz. | potential | pressure | max force diff (max \|F\|) |
|---|---|---|---|---|---|---|---|
| water512 | 0 | -2.9e-7 | -7e-8 | 1.3e-6 | -1.2e-7 | 0.03 bar | 7.8e-3 (3488) |
| 4096 waters (benchmark settings) | 0 | -6.3e-7 | 0 | 9e-7 | -1.1e-7 | 0.06 bar | 1.5e-2 (1106) |
| GVDW, overrides + tail | 1.5e-6 | -1.9e-7 | 0 | 1.3e-6 | 0 | 0.002 bar | 1.1e-2 (3637) |
| GVDW, 4096 waters | 1.2e-6 | -2.2e-7 | 0 | 1.4e-6 | -1.2e-7 | 0.009 bar | 2.4e-2 (3637) |

`core/cuda/tests/test_cuda` checks the engine against pgm_core in double on a dump (total energy 7e-8
relative, dipoles 3e-6 relative, strain 8e-7), its own list against given candidates and after
displacements that rebuild it, GVDW against a double brute-force sum (1.5e-6), and the mu4 predictor
over moved steps (1.3e-7 against pgm_core at every step), the list after a reordering and the float
positions (identical results).  NVE, 10 ps, 1 fs: 4096 waters total-energy fluctuation 2.4 kJ/mol and
drift -8.1 kJ/mol on the GPU, 1.4 and -5.0 kJ/mol on the CPU path; with GVDW 0.6 and +0.3 kJ/mol.  NPT water512 (C-rescale, 150 ps, the same seed): density 1020.6 +- 2.1
kg/m^3 and potential -2.11509e6 kJ/mol on the GPU, 1022.9 +- 1.9 and -2.11509e6 on the CPU path (three
GPU runs of the code's stages: 1017-1021).

Speed, 4096 waters (12288 atoms), pmemd's benchmark settings (rc 0.9 nm, beta 4/nm, 56^3, order 6, tol
1e-4, mu4, peek 0.65, 1 fs, NVE), RTX PRO 6000 Blackwell, 8 CPU threads, ns/day:

| | ns/day |
|---|---|
| GROMACS, pGM on the GPU, LJ, nstlist 10 | 124 |
| the same, nstlist 40 (GROMACS's Verlet buffer for LJ) | 155 |
| GROMACS, pGM + GVDW on the GPU, nstlist 10 | 117 |
| pmemd.pgm.cuda (SPFP) | 91.7 |
| OpenMM plugin, CUDA mixed | 77.2 |
| GROMACS, CPU path (8 threads), LJ | 4.3 |

Per step at nstlist 40 (0.52 ms): the GPU part 0.43 ms of kernels (2 CG iterations; per step the
first pair pass with the pair setup 64 us, the other field passes 22 us each, the final pass 44 us, FFTs
52 us, spreads 56 us, gathers 35 us) plus 0.02 ms of host work, launch 0.03 ms, the CPU's own nonbonded
work (0.26 ms) hidden behind the GPU, GROMACS's update and search the rest.  Tried and kept off: the kernels computed
in every pass instead of stored (`PGM_CUDA_KERNELS_ON_THE_FLY`, slower), G0/G3 stored too
(`PGM_CUDA_STORE_G03`, the same speed).

### Next

1. **Decomposed PME for DD**: spread into the local grid region only, send the parts to the owners of
   FFT slabs (`gmx_parallel_3dfft` with the PP communicator), distributed FFT and convolution, return
   the potential to the ranks whose atoms need it; instead of the whole-grid sum + redundant FFT.
2. **GPU with DD**: one engine per rank on its home + halo atoms (the module's halo communication), or
   one GPU engine for the whole system on the main rank.
   One rank: the kernels are now the step (FFT, spread and pair passes about equal); LJ on the GPU would
   only help when the CPU's share stops being hidden (fewer CPU cores).
3. CPU, if needed: the remaining field passes (~0.2 ms each) as 4xM cluster SIMD tiles; the FFT
   (0.9 ms per step at 36^3, 14 transforms) with fewer threads or a batched 3D FFT.

Check: the same validations (`tests/validate_all.sh`), then speed against pmemd.pgm (CPU / GPU) and
OpenMM.
