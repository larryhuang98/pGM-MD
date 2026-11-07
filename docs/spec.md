# pGM model specification for the native OpenMM and GROMACS implementations

This is the model the native engines must reproduce. It is read off the two existing codes:

* **pGM-JAX** (master ab9c213): `pgm_jax/md/forcefield.py` (MD energy, solver,
  forces), `md/kernels.py` (radial kernels), `md/pme.py` (smooth PME of charges + dipoles),
  `densities.py`, `vdw.py`, `de.py`, `md/topology.py` (vdW exclusions), `protein/pmemd.py`
  (prmtop sections and the pmemd differences, checked term by term).
* **pmemd-pgm** (branch pgm-cuda-larry-merge-master, 83ede07):
  `src/pmemd/src/pGM_*.F90` and `src/pmemd/src/cuda/*_PGM.*`.

pGM-JAX reproduces pmemd-pgm to its print precision (1e-4 kcal/mol, forces 2e-5 kcal/mol/A) once the
two differences of Sec. 8 are accounted for, so pGM-JAX is the executable reference for every test.

Units here: nm, e, e nm, nm^3, kJ/mol (OpenMM's). Amber files are in A, A^3, e A, kcal/mol.

## 1. What is different from a standard fixed-charge / AMOEBA force field

| | standard (Amber/CHARMM, NonbondedForce) | AMOEBA | **pGM** |
|---|---|---|---|
| charges | point | point | **Gaussian**, width from the atomic radius |
| permanent dipoles | none | local frames (z-then-x, bisector...) and torques | **covalent basis vectors**: p_i = sum_j c_ij u_ij along bonds, no frames, no torques |
| quadrupoles | none | yes | none in MD (pGM-JAX has them for gas phase only) |
| polarization | none | Thole-damped induced point dipoles | **Gaussian induced dipoles**, damping = the same erf kernel |
| electrostatic exclusions | 1-2, 1-3 excluded, 1-4 scaled | polarization groups, 4 scale sets | **none: every pair interacts** (incl. 1-2, 1-3, 1-4, and an atom's induced dipole with its bonded partners) |
| 1-4 electrostatics | scaled (SCEE) | scaled | **none separate** (already in the all-pairs sum) |
| vdW exclusions | 1-2, 1-3, 1-4 scaled | AMOEBA buffered 14-7 | **yes**: as the prmtop says (Sec. 7) — only the vdW has exclusions |

## 2. Parameters and where they come from

Per atom i: Gaussian charge q_i, radius R_i, polarizability alpha_i; per covalent dipole m = (i, j):
strength c_m (atom i may have several, one per partner j). pmemd-pgm prmtop sections (pmemd finds
them by name, `pGM_MPOLE_readparm`):

| section | content | unit in file |
|---|---|---|
| `POL_GAUSS_FORCEFIELD` | 1 | – |
| `POL_GAUSS_COVALENT_POINTERS_LIST` | number of covalent dipoles of each atom | – |
| `POL_GAUSS_COVALENT_ATOMS_LIST` | partners j (1-based), atom after atom | – |
| `POL_GAUSS_COVALENT_DIPOLES_LIST` | strengths c (p_i += c unit(r_j - r_i)) | e A |
| `POL_GAUSS_MONOPOLES_LIST` | q | e |
| `POL_GAUSS_RADII_LIST` | R | A |
| `POL_GAUSS_POLARIZABILITY_LIST` | alpha | A^3 |
| `CHARGE` | 18.2223 q, **not used** by pmemd-pgm (analysis tools only) | |

Example, pGM3P-25 water (`pGM-JAX/examples/water512/pgm3p25_512.prmtop`): q = -2.0406 / 1.0203 e,
R = 0.60515 / 0.53623 A, alpha = 1.11765 / 0.32963 A^3; O has two covalent dipoles (to H1, H2,
c = -0.19120 e A) and each H one (to O, c = 0.085877 e A).

## 3. Permanent dipoles (covalent basis)

    u_m = (r_j - r_i) / |r_j - r_i|   (minimum image)
    p_i = sum_{m = (i, j)} c_m u_m

(pmemd `pGM_MPOLE_local_to_global`, pGM-JAX `PGMForceField.perm_dipoles`). There are no local
frames: p follows the geometry smoothly and the force of p is the chain rule of Sec. 6.

## 4. Electrostatic energy

Total dipole d_i = p_i + mu_i (mu: induced). With KE the Coulomb constant (Sec. 8),

    E_elec = KE [ U_dir + U_rec + U_self + U_bg + sum_i |mu_i|^2 / (2 alpha_i) ]

**Gaussian pair kernel.** Gaussian density rho_i ~ exp(-r^2 / (2 R_i^2)); the pair exponent is

    a_ij = 1 / sqrt(2 (R_i^2 + R_j^2))      (pmemd: sqrt(1/(r_gauss_i + r_gauss_j)), r_gauss = 2 R^2)

and two unit Gaussian charges interact through phi(r) = erf(a_ij r)/r. Radial functions of a
kernel f(r): B_0 = f, B_{n+1} = -(1/r) dB_n/dr. For erf(a r)/r, closed form
B_n = [(2n-1) B_{n-1} - (2a^2)^(n-1) * 2a exp(-a^2 r^2)/sqrt(pi)] / r^2, and for s = (a r)^2 < 1 the
series B_n = (2 a^(2n+1)/sqrt(pi)) 2^n sum_{m=0}^{14} (-s)^m / (m! (2m+2n+1)) (the closed form loses
one digit per order at a r ~ 0.7 in float32; md/kernels.py `erf_kernels`).

**Direct space** (pairs i < j with r < r_c, *all* pairs, periodic minimum image; x = r_i - r_j):

    G_n = B_n[erf(a_ij r)/r] - B_n[erf(beta r)/r]          (= B_n[erfc(beta r)/r] - B_n[erfc(a_ij r)/r])
    U_dir = sum q_i q_j G0 + (q_i d_j.x - q_j d_i.x) G1 + (d_i.d_j) G1 - (d_i.x)(d_j.x) G2

Numerically (`core/pgm_math.h` `ewaldPairKernels`, used by pgm_core, the GROMACS module and the
OpenMM kernels): for a_ij r < 1 the difference of the two erf series; for a_ij r >= 1 the erfc form
Bc(beta) - Bc(a_ij), Bc_n = [(2n-1) Bc_{n-1} + (2a^2)^(n-1) 2a exp(-a^2 r^2)/sqrt(pi)] / r^2 (all terms
positive), which avoids the cancellation of the 1/r^(2n+1) parts at long range (the erf form loses
~5 digits of G_0 at r_c).

(beta: Ewald coefficient; pairs beyond r_c are assumed to have erfc(beta r) and erfc(a_ij r)
negligible, which holds since a_ij > beta in practice: water a_OH = 8.7 /nm, beta ~ 3.5 /nm.  The
OpenMM plugin checks it: erfc(a_min r_c), a_min = 1/(2 R_max), must not exceed the Ewald error
tolerance.)

**Reciprocal space:** standard smooth PME of point charges and point dipoles with the same beta
(Essmann 1995; dipoles as Sagui 2004): a dipole is spread as d . grad of its charge spline.
pGM-JAX's PME follows OpenMM's reference PME (B-spline recursion, Euler exponential-spline moduli);
the grid-index convention differs between implementations by a constant shift of the grid (OpenMM
reference: points floor(w)..floor(w)+p-1; pgm_core and the GROMACS module: M_p(fr + j) on point
floor(w) - j; the GPU kernels: AMOEBA's floor(w) - p + 1 ...), which leaves energies and forces
unchanged. The Gaussian widths enter only the direct
sum.

**Self and background:**

    U_self = -(beta/sqrt(pi)) sum q_i^2 - (2 beta^3/(3 sqrt(pi))) sum |d_i|^2
    U_bg   = -pi Q^2 / (2 V beta^2),   Q = sum q_i

No intra-atom term: an atom's permanent and induced dipoles do not interact with each other (the
polarization energy |mu|^2/(2 alpha) takes that role).

**Field** (needed by the solver and the forces): E_i = -dU/dd_i, in e/nm^2 (multiply by KE for
kJ/mol/(e nm)). Direct part from j: E_i += q_j x G1 - d_j G1 + (d_j.x) x G2; self part
E_i += (4 beta^3/(3 sqrt(pi))) d_i (pmemd `pGM_SELF_dipole_field`: dip_field -= factor*dip with its
sign convention for the gradient); reciprocal part from the PME potential gradient.

## 5. Induced dipoles

mu minimizes E_elec (a quadratic in mu): for every polarizable atom

    mu_i / alpha_i - E_i(q, p + mu) = 0,     i.e.  (alpha^-1 - T) mu = E^perm

with E^perm the field of the charges and the permanent dipoles (direct + PME + self) and T the
dipole-dipole field tensor (direct + PME + self). Atoms with alpha = 0 (pmemd: alpha <= 1e-6 A^3)
are not polarizable: mu_i = 0 and their |mu|^2/(2 alpha) is 0 (every engine applies the threshold
1e-9 nm^3, core/pgm_constants.h).

Solver (as pmemd-pgm GPU and pGM-JAX, `Induction`):

* preconditioned conjugate gradients on the symmetric positive definite system above, Jacobi
  preconditioner z = alpha r (optionally a few inner iterations on the short-range tensor);
* convergence: max_i |alpha_i r_i| / mean_i |alpha_i E^perm_i| <= tol (pmemd `dipole_scf_tol`,
  default 2e-4 in pmemd's mdin, pGM-JAX default 1e-5), then one "peek" step mu += omega alpha r
  (omega = scf_sor_coefficient = 0.65);
* initial guess: extrapolation of the converged dipoles of the previous steps, "mu4":
  4 mu_1 - 6 mu_2 + 4 mu_3 - mu_4 (pmemd GPU PGM_GPU_PRED=mu4; 3-4 CG iterations at tol 1e-4 in
  water at 1 fs), "mu3" quadratic, "ls" pmemd CPU's least-squares extrapolation, "none" alpha E^perm.

The energy is reported at the solved mu with the variational expression of Sec. 4 (so it is
second-order accurate in the solver error).

## 6. Forces and virial

E is variational in mu, so F = -dE/dR at fixed mu:

* pair terms: analytic gradient of the U_dir pair energy with d = p + mu held fixed
  (grad_x G_n = -G_{n+1} x, so G3 is needed);
* PME: the usual spline-derivative forces of charges and dipoles;
* covalent dipoles: with g_i = dE/dd_i = -KE E_i (total field at i, including the field of the
  induced dipoles), each covalent dipole m = (i, j) with v = r_j - r_i adds
  dE/dr_j += c_m (g_i - (g_i.u_m) u_m) / |v|,  dE/dr_i -= the same;
* virial / pressure: OpenMM's MC barostat needs only energies (no virial); GROMACS needs the
  virial: molecular-free atomic virial of the pair and PME terms plus the covalent-dipole chain term
  (pGM-JAX computes the strain derivative by autodiff; use it as the reference).

## 7. Van der Waals

Options (pGM-JAX `Terms.vdw`; pmemd igvdw / idegauss / ibuckvdw):

* **LJ** (default, pGM3P-25): Amber form from LENNARD_JONES_ACOEF/BCOEF, cutoff r_c, isotropic
  long-range correction (Amber vdwmeth = 1, pGM-JAX `lj_lrc`).
* **DE** (DEGAUSS double exponential): u = eps/(a-b) [b e^{a(1-r/rm)} - a e^{b(1-r/rm)}], a = 18.17,
  b = 3.65 by default, eps and rm from the LJ combination (or prmtop DOUBLE_EXPONENTIAL_* sections).
* **GVDW** (Gaussian-density vdW, pmemd igvdw = 1): y = a_ij r with the pGM a_ij of Sec. 4,
  U = A exp(-B y^2) ("gauss") or A exp(-B y) ("slater") - C6 F(y)/r^6,
  F(y) = [erf y - (2y/sqrt(pi))(1 + 2y^2/3) e^{-y^2}]^2 + (1/2)[(4y^3/(3 sqrt(pi))) e^{-y^2}]^2,
  series below y = 0.6; tail correction -2 pi (sum c_i)^2/(3 V r_c^3). pmemd uses one global
  (A, C6, B) for LJ-bearing atoms; pGM-JAX allows per-type values (A_ij = a_i a_j, C6_ij = c_i c_j,
  B_ij = (b_i + b_j)/2).
* (pmemd also has Buckingham; not in pGM-JAX.)

**vdW exclusions** (the only exclusions in pGM): EXCLUDED_ATOMS_LIST; rigid molecules (water, ions):
every intramolecular pair excluded; flexible molecules: pairs fewer than lj_min_sep bonds apart.
1-4 pairs: SCNB = 1 hard-coded in pmemd-pgm; a different 1-4 LJ scale is carried in the CHARMM
LENNARD_JONES_14_ACOEF/BCOEF tables (pGM-JAX `write_pgm_prmtop`). 1-4 electrostatics: none.

vdW is a plain pair potential with exclusions, so in OpenMM it is a CustomNonbondedForce (+ a
CustomBondForce for 1-4 pairs), compiled to GPU code by OpenMM; in GROMACS LJ is the standard
kernel and DE/GVDW are tabulated user potentials. The native work is the electrostatics.

## 8. Constants and known differences between the codes

* Coulomb constant: pGM-JAX 138.935458, OpenMM ONE_4PI_EPS0 = 138.935456 (both CODATA, 1e-8
  apart), pmemd-pgm Tinker's 332.05382 kcal A/mol = 138.935318 kJ nm/mol (2.98e-5 lower). The
  native force takes KE as a parameter (default OpenMM's) so parity runs can use pmemd's.
* pmemd's PME influence function has an extra factor (factor_lambda, pme_recip_dat.F90):
  2-5e-7 of E_elec at 0.08 nm spacing, order 6; nothing at 0.04 nm, order 8. pGM-JAX = OpenMM here.

## 9. Out of scope for the first version

Quadrupoles, charge flux, IPS long range, external field, extended-Lagrangian dipoles (iEL),
virtual sites, alchemical scaling. Each is optional in pGM-JAX; add them after the core is validated.

## 10. Validation targets

1. Water512 (pGM3P-25 + LJ, `examples/water512`), 10 NVE steps at 1 fs, double precision:
   E_pot(step 10) = -2114884.26 kJ/mol (pGM-JAX README). Single-point energy, per-term energies,
   forces (max/RMS), induced dipoles against pGM-JAX `PGMEngine.compute` at the same coordinates.
2. Finite-difference force checks (tight tol), all four levels q / qp / qi / qpi.
3. A flexible molecule with exclusions and 1-4 pairs (protein test of protein/pmemd.py), against
   pGM-JAX and pmemd.pgm.cuda.
4. GVDW water (`scripts/validation/validate_gvdw.py`: pmemd VDWAALS 784.7282 slater / 868.3699 gauss
   kcal/mol).
5. NVE energy drift, NPT density of water (1.0183 +- 0.0010 g/cm^3 native MC barostat).
