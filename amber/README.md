# pGM for AMBER PMEMD

Polarizable Gaussian multipole (pGM) molecular dynamics for AMBER PMEMD on CPUs
(`pmemd.pgm`, `pmemd.pgm.MPI`) and NVIDIA GPUs (`pmemd.pgm.cuda`, SPFP and DPFP).
The patch adds these executables to an existing AMBER source tree; the standard
`pmemd` and `pmemd.cuda` executables are built exactly as before.

Supported AMBER version: **AMBER24** (PMEMD 24) with the official updates applied
(`./update_amber --update`; tested with updates 1-3). The unpatched AMBER24 release tarball is not supported:
the installer detects it and changes nothing.
GPU builds: CUDA 11.x-12.9; with CUDA 12.8 or later the build also targets
Blackwell GPUs (sm_100, sm_120, e.g. RTX 5090).

## Install

```bash
git clone https://github.com/XXXX/pgm-amber.git
cd pgm-amber
./install.sh --amber-src /path/to/pmemd24_src            # patch only
./install.sh --amber-src /path/to/pmemd24_src \
             --build --build-dir /path/to/pmemd24_src/build   # patch and build
```

`install.sh` checks the AMBER version and that the patch applies before it changes
anything, saves every file it modifies in `<amber-src>/.pgm_backup`, and can undo
the installation with `./install.sh --uninstall --amber-src DIR`.

## Input

pGM is switched on with `ipgm=1` in `&cntrl`; its settings are in `&pol_gauss`.
The defaults of the GPU code are the optimized induction solver described in the
paper (cubic extrapolation with the combined field, diagonally preconditioned CG,
convergence test after every three iterations, concurrent direct/reciprocal space).

| `&pol_gauss` variable | default | meaning |
|---|---|---|
| `dipole_scf_tol` | 2e-4 | induction tolerance τ (thermostatted GPU runs: 1e-4 with a 2 fs time step; NVE: 1 fs and 1e-4 or tighter) |
| `scf_solv_opt` | 1 (GPU), 2 (CPU) | 1 diagonal CG; 2, 3 local-preconditioned CG (3 adds Jacobi under-relaxation on the CPU); on the GPU, 2 and 3 use separate permanent and induced fields |
| `scf_gpu_pred` | 1 | GPU predictor: 0 quadratic, 1 cubic + combined field, 2 MOE |
| `scf_gpu_check` | 3 | CG iterations between convergence tests (GPU) |
| `scf_gpu_overlap` | 1 | concurrent direct- and reciprocal-space evaluation (GPU) |
| `pgm_power` | 0 | N > 0: print the induction contribution to the energy drift (eq 12 of the paper) every N steps; use DPFP |
| `igvdw` | 0 | van der Waals model: 0 Lennard-Jones 12-6, 1 Gaussian VDW |
| `gvdw_rep_form` | 0 | GVDW repulsion: 0 Gaussian, 1 Slater |
| `b_rep_scale` | 1.0 | GVDW repulsion range factor |

With `igvdw=1` the topology `LENNARD_JONES_ACOEF` and `LENNARD_JONES_BCOEF` arrays hold the
GVDW repulsion amplitude A and dispersion coefficient C6 for each pair of atom types.
1-4 pairs use the same GVDW form (unscaled, as for Lennard-Jones with pGM).

GPU limitations: single GPU; PME interpolation order 5 or 6; no IPS.

## Tests

```bash
test/run_tests.sh $AMBERHOME/bin
```

runs 20-step NVE trajectories of a 2,6-dimethylphenol liquid with Lennard-Jones and with Gaussian
van der Waals (including 1-4 pairs) for each pGM executable that is installed (`pmemd.pgm`,
`pmemd.pgm.cuda_DPFP`, `pmemd.pgm.cuda_SPFP`), and on the GPU also with the local-preconditioned
solver (`scf_solv_opt=3`), and compares the total and potential energies with
`test/reference.txt` (tolerance 0.002 kcal/mol, 0.05 for SPFP).

## Citation

Huang, Z.; Lee, T.-S.; Duan, Y.; Luo, R. GPU-Accelerated Polarizable Gaussian
Multipole Molecular Dynamics in AMBER. *J. Chem. Theory Comput.* (submitted).
