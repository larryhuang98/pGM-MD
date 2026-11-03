"""Compare the OpenMM pGM plugin with a pGM-JAX reference (ref_pgmjax.py output).

    python compare_openmm.py PRMTOP REF.npz [--platform Reference] [--elec qpi] [--cutoff 0.9] ...

Prints the energy terms of both codes, the force deviation and the induced-dipole deviation.  Run
with the pgm-omm environment.
"""

import argparse

import numpy as np
import openmm as mm
from openmm import unit

import pgm_prmtop

ap = argparse.ArgumentParser()
ap.add_argument("prmtop")
ap.add_argument("ref")
ap.add_argument("--platform", default="CPU")
ap.add_argument("--precision", default="double")
ap.add_argument("--elec", default="qpi")
ap.add_argument("--cutoff", type=float, default=0.9)
ap.add_argument("--beta", type=float, default=3.5)
ap.add_argument("--grid", type=int, nargs=3, default=[36, 36, 36])
ap.add_argument("--order", type=int, default=6)
ap.add_argument("--tol", type=float, default=1e-10)
ap.add_argument("--no-lrc", action="store_true")
ap.add_argument("--no-lj", action="store_true")
ap.add_argument("--ke", type=float, default=None, help="Coulomb constant (pGM-JAX: 138.935458)")
a = ap.parse_args()

ref = np.load(a.ref)
H = ref["H"]
assert abs(H[0, 1]) + abs(H[0, 2]) + abs(H[1, 2]) < 1e-12, "box not in OpenMM's reduced form"
system, top, pgm = pgm_prmtop.create_system(
    a.prmtop, cutoff=a.cutoff, ewald_beta=a.beta, grid=tuple(a.grid), order=a.order, dipole_tol=a.tol,
    max_iter=1000, peek=0.0, predictor="none", polarization=a.elec in ("qi", "qpi"), lj_lrc=not a.no_lrc, coulomb=a.ke, lj=not a.no_lj,
)  # fmt: skip
if a.elec in ("q", "qi"):
    for k in range(pgm.getNumCovalentDipoles()):
        i, j, c = pgm.getCovalentDipoleParameters(k)
        pgm.setCovalentDipoleParameters(k, i, j, 0.0)
system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*H[k]) for k in range(3)])
plat = mm.Platform.getPlatformByName(a.platform)
props = {"Precision": a.precision} if a.platform in ("CUDA", "OpenCL") else {}
ctx = mm.Context(system, mm.VerletIntegrator(0.001), plat, props)
ctx.setPeriodicBoxVectors(*[mm.Vec3(*H[k]) for k in range(3)])
ctx.setPositions(ref["pos"])


def group(g):
    s = ctx.getState(getEnergy=True, getForces=True, groups={g})
    return s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), s.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )


e_pgm, f_pgm = group(0)
e_lj, f_lj = group(1) if not a.no_lj else (0.0, 0.0)
mu = pgm.getInducedDipoles(ctx)
it = pgm.getInducedDipoleIterations(ctx)
F = f_pgm + f_lj
Fr = ref["forces"]
print(f"platform {a.platform} {a.precision}")
print(f"elec   openmm {e_pgm:.6f}   pgm-jax {float(ref['term_elec']):.6f}   diff {e_pgm - float(ref['term_elec']):.3e}")
print(f"vdw    openmm {e_lj:.6f}   pgm-jax {float(ref['term_vdw']):.6f}   diff {e_lj - float(ref['term_vdw']):.3e}")
print(f"total  openmm {e_pgm + e_lj:.6f}   pgm-jax {float(ref['energy']):.6f}   rel {(e_pgm + e_lj) / float(ref['energy']) - 1:.3e}")
dF = F - Fr
print(f"forces: max|dF| {np.abs(dF).max():.3e}  rms dF {np.sqrt((dF**2).mean()):.3e}  rms F {np.sqrt((Fr**2).mean()):.3e}  kJ/mol/nm")
dmu = mu - ref["mu"]
print(f"mu: max|dmu| {np.abs(dmu).max():.3e}  max|mu| {np.abs(ref['mu']).max():.3e} e nm   iterations {it} (pgm-jax {int(ref['iterations'])})")
