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
  4. forces at fixed dipoles: pair pass (forces + total field, i-forces into `fshift[shift]` so the nbnxm
     virial machinery stays correct), PME force gather with second derivatives, covalent chain rule;
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
