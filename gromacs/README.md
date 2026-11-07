# GROMACS pGM patch

This patch adds pGM electrostatics to GROMACS 2026.4 as a new Coulomb interaction type,
`coulombtype = pGM`. It includes:

* Gaussian van der Waals (GVDW);
* domain decomposition on the CPU;
* a GPU path that runs the pGM terms on one GPU while the Lennard-Jones and bonded terms run on the CPU.

## Install and build

```bash
git clone --branch v2026.4 https://gitlab.com/gromacs/gromacs.git gromacs-2026.4
gromacs/install.sh gromacs-2026.4                        # patch the source tree (idempotent)
gromacs/build.sh gromacs-2026.4 $HOME/opt/gmx-pgm mixed   # CUDA build, single precision: bin/gmx
gromacs/build.sh gromacs-2026.4 $HOME/opt/gmx-pgm double  # CPU build, double precision: bin/gmx_d
```

`install.sh` copies the module `src/pgm` and the shared headers and CUDA engine from `../core` into
`src/gromacs/pgm`. It then applies the patch stages in `patches/`. Each stage is a list of exact string
replacements, and each replacement must match exactly once. If the tree is not GROMACS 2026.4, the script
stops before it changes a file.

## Topologies

`tools/prmtop_to_gmx.py` converts an AMBER pGM parameter-topology file and restart into a GROMACS topology
and coordinates. The topology carries the pGM parameters in two new directives, `[ pgm_atoms ]` (Gaussian
radius, polarizability) and `[ pgm_dipoles ]` (covalent dipoles). The script needs ParmEd.

```bash
python gromacs/tools/prmtop_to_gmx.py system.prmtop system.rst7 system            # Lennard-Jones
python gromacs/tools/prmtop_to_gmx.py system.prmtop system.rst7 system --gvdw-topology slater 1.0   # GVDW from the prmtop
```

The script prints the `epsilon-r` that reproduces the Coulomb constant of the AMBER pGM code. Put this value
in the mdp file so that the energies match those of AMBER and OpenMM.

## mdp options

| option | meaning |
|---|---|
| `coulombtype = pGM` | pGM electrostatics. `rcoulomb`, `fourier-n*`, `pme-order` and `ewald-rtol` set the real-space cutoff and the PME. |
| `pgm-dipole-tol` | induction tolerance τ |
| `pgm-max-iter` | maximum number of conjugate-gradient iterations (default 100) |
| `pgm-predictor` | initial dipoles: 0 α E_perm, 1 quadratic, 2 cubic extrapolation from four previous steps (default) |
| `pgm-peek` | under-relaxed correction after convergence (default 0.65, as in AMBER; 0 off) |
| `pgm-gvdw-rep-form` | GVDW repulsion: 1 Slater, 0 Gaussian |

To run the pGM terms on the GPU, set `GMX_PGM_GPU=1` and run `gmx mdrun -nb cpu -ntmpi 1` with a CUDA build.
`docs/gromacs_design.md` has the design notes, and `tests/gromacs_tests.sh` runs the tests.
