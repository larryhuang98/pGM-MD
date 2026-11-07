# Tests

| Script | What it checks | Needs |
|---|---|---|
| `core_tests.sh` | the C++ reference implementation (`core/`): energies, forces, induction, GVDW kernel | C++17 compiler |
| `openmm_tests.sh [PLATFORM ...]` | the OpenMM plugin: API, `TailCorrectionForce`, `GVDWForce`, agreement with the pGM-JAX reference (`data/ref_w512_*`), GPU platforms against CPU, GVDW against pmemd-pgm, NVE | `OMM_PY` |
| `gromacs_tests.sh [double\|cuda ...]` | the GROMACS patch: single point against `core/` and pGM-JAX, GVDW, GVDW 1-4 pairs against OpenMM, domain decomposition (1-8 ranks), NVE | `GMX_DOUBLE`, `GMX_MIXED`, `PY`, `OMM_PY` |
| `xplat/run_xplat.sh [cpu\|all]` | AMBER (CPU reference, CUDA DPFP/SPFP), OpenMM and GROMACS single points of the same coordinates (DMP liquid, 512 waters) | all of the above, `AMBER_BIN` |
| `../amber/test/run_tests.sh BIN` | the AMBER patch: NVE trajectories against reference energies | AMBER with the patch |

The variables are described in `env.sh`. Optional inputs:

* `PGM_GVDW_DATA`: the 512-water box of the GVDW paper, for the GVDW comparisons with pmemd-pgm.
* `QM7B_DIR`: `qm7b_7203.prmtop`/`.inpcrd` of the AMBER test suite (`$AMBERHOME/test/nmode`), for the
  GVDW 1-4 test.

Tests that need missing inputs are skipped.

Notes:

* `test_pgm_api.py` and `test_gvdw_api.py` keep several OpenMM Contexts alive at once. On a GPU in
  exclusive-process compute mode, their CUDA cases therefore fail with "No compatible CUDA device is
  available". Run them on a GPU in the default compute mode. The other CUDA tests need only one Context.
* `data/ref_w512_*_novdw.npz` were computed with pGM-JAX (`ref_pgmjax.py`; electrostatics only, levels q,
  qp, qi, qpi).
