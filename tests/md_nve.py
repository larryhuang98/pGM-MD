"""NVE molecular dynamics with PGMForce + LJ or GVDW: energy conservation and timing.

    python md_nve.py PRMTOP RST7 [--platform OpenCL] [--precision mixed] [--steps 200] [--dt-fs 1.0]

Rigid water (SETTLE), velocities at 298 K, Verlet integrator, mu4 predictor and peek 0.65 (the pmemd-pgm
GPU defaults), induced-dipole tolerance --tol.  Prints potential, kinetic and total energy every
--every steps, CG iterations, the drift of the total energy and ns/day.
"""

import argparse
import time

import numpy as np
import openmm as mm
from openmm import app, unit

import openmmpgm
import pgm_prmtop

ap = argparse.ArgumentParser()
ap.add_argument("prmtop")
ap.add_argument("rst7")
ap.add_argument("--platform", default="OpenCL")
ap.add_argument("--precision", default="mixed")
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--every", type=int, default=20)
ap.add_argument("--dt-fs", type=float, default=1.0)
ap.add_argument("--tol", type=float, default=1e-5)
ap.add_argument("--beta", type=float, default=3.5)
ap.add_argument("--grid", type=int, nargs=3, default=[36, 36, 36])
ap.add_argument("--predictor", default="mu4")
ap.add_argument("--peek", type=float, default=0.65)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--cutoff", type=float, default=0.9)
ap.add_argument("--vdw", default="lj", help="lj | gvdw | none")
ap.add_argument("--gvdw", default="slater", help="pgm_prmtop.PGM3P_GVDW key (with --vdw gvdw)")
ap.add_argument("--lrc", action="store_true", help="vdW dispersion correction")
ap.add_argument("--nopol", action="store_true", help="no induced dipoles (fixed charges + permanent dipoles)")
ap.add_argument("--replicate", type=int, nargs=3, default=None, help="the box nx ny nz times (tests/replicate.py)")
a = ap.parse_args()

system, top, pgm = pgm_prmtop.create_system(
    a.prmtop, cutoff=a.cutoff, ewald_beta=a.beta, grid=tuple(a.grid), dipole_tol=a.tol, peek=a.peek, predictor=a.predictor,
    vdw=a.vdw, gvdw=a.gvdw if a.vdw == "gvdw" else None, lj_lrc=a.lrc, polarization=not a.nopol,
)  # fmt: skip
crd = app.AmberInpcrdFile(a.rst7)
positions = crd.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
boxv = [mm.Vec3(*[float(x) for x in v.value_in_unit(unit.nanometer)]) for v in crd.boxVectors]
if a.replicate:
    import replicate
    system, positions, B = replicate.replicate(system, positions, [[v.x, v.y, v.z] for v in boxv], a.replicate)
    boxv = [mm.Vec3(*row) for row in B]
    pgm = [openmmpgm.PGMForce.cast(f) for f in system.getForces() if openmmpgm.PGMForce.isinstance(f)][0]
system.setDefaultPeriodicBoxVectors(*boxv)
plat = mm.Platform.getPlatformByName(a.platform)
props = {"Precision": a.precision} if a.platform in ("CUDA", "OpenCL", "HIP") else {}
integ = mm.VerletIntegrator(a.dt_fs * unit.femtosecond)
integ.setConstraintTolerance(1e-8)
ctx = mm.Context(system, integ, plat, props)
ctx.setPositions(positions)
ctx.setPeriodicBoxVectors(*boxv)
ctx.applyConstraints(1e-8)
ctx.setVelocitiesToTemperature(298 * unit.kelvin, a.seed)
ctx.applyVelocityConstraints(1e-8)
print(f"# {system.getNumParticles()} atoms; vdw {a.vdw}{' ' + a.gvdw if a.vdw == 'gvdw' else ''}, cutoff {a.cutoff}; {a.platform} {a.precision}, dt {a.dt_fs} fs, tol {a.tol}, predictor {a.predictor}, peek {a.peek}")
print("# step    epot            ekin          etot            iters")
rows = []
t0 = time.time()
for step in range(0, a.steps + 1, a.every):
    if step > 0:
        integ.step(a.every)
    s = ctx.getState(getEnergy=True)
    ep = s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    ek = s.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    rows.append((step, ep, ek, ep + ek))
    print(f"{step:6d} {ep:16.4f} {ek:12.4f} {ep + ek:16.4f} {pgm.getInducedDipoleIterations(ctx):4d}", flush=True)
wall = time.time() - t0
r = np.array(rows)
slope = np.polyfit(r[:, 0] * a.dt_fs * 1e-3, r[:, 3], 1)[0]   # kJ/mol per ps
ndof = 3 * system.getNumParticles() - system.getNumConstraints() - 3
print(f"# total energy: drift {slope:.4f} kJ/mol/ps ({slope / ndof * 1e3:.4f} J/mol/ps per dof), "
      f"rms fluctuation {r[:, 3].std():.4f} kJ/mol, ekin {r[:, 2].mean():.1f}")
print(f"# {a.steps} steps in {wall:.1f} s: {a.steps * a.dt_fs * 1e-6 / wall * 86400:.2f} ns/day (incl. energy reports)")
