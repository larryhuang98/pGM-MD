# OpenMM pGM plugin

This plugin adds two native OpenMM forces:

* `PGMForce`: pGM electrostatics, with Gaussian charges, covalent permanent dipoles and induced Gaussian
  dipoles. Periodic systems use PME of charges and dipoles.
* `GVDWForce`: Gaussian van der Waals, with Slater or Gaussian repulsion.

Both forces run on the Reference, CPU, OpenCL and CUDA platforms. The CUDA platform in mixed precision
uses the pGM engine in `../core/cuda`, the same engine that GROMACS uses. The Python module `pgm_prmtop`
builds an OpenMM `System` from an AMBER pGM parameter-topology file. It follows the conventions of the
AMBER code: the Coulomb constant 332.05382 kcal Å mol⁻¹ e⁻², unscaled 1-4 van der Waals, and GVDW
parameters per atom type.

## Build

```bash
conda create -n pgm -c conda-forge openmm=8.6 cmake swig numpy scipy parmed
conda activate pgm
OPENMM_DIR=$CONDA_PREFIX openmm/build.sh
```

The CUDA platform needs `nvcc` in `PATH` at build time. Pass `-DCMAKE_CUDA_ARCHITECTURES=...` to
`build.sh` for GPUs that the default (80;89;120) does not cover. `-DPGM_BUILD_COMMON=OFF` builds only the
Reference and CPU platforms.

## Use

```python
import openmm as mm, openmm.app as app, openmm.unit as u
import pgm_prmtop

system, top, pgm = pgm_prmtop.create_system("water.prmtop", cutoff=0.9, ewald_beta=3.5, grid=(36, 36, 36),
                                            dipole_tol=1e-4, predictor="mu4", peek=0.65)
crd = app.AmberInpcrdFile("water.rst7")
sim = app.Simulation(top, system, mm.LangevinMiddleIntegrator(298*u.kelvin, 1/u.picosecond, 2*u.femtosecond),
                     mm.Platform.getPlatformByName("CUDA"), {"Precision": "mixed"})
sim.context.setPositions(crd.positions); sim.context.setPeriodicBoxVectors(*crd.boxVectors)
sim.step(1000)
```

`docs/openmm.md` describes the platforms, the solver options and the validation in more detail.
`tests/openmm_tests.sh` runs the tests.
