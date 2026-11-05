"""NPT molecular dynamics with PGMForce + LJ or GVDW: density and timing.

    python md_npt.py PRMTOP RST7 [--platform CUDA] [--precision mixed] [--ps 100] [--equil-ps 20]

Rigid water (SETTLE), Langevin middle integrator (1/ps), MonteCarloBarostat (1 bar, every 25 steps),
mu4 predictor and peek 0.65.  Prints the density every --every steps and its mean and standard error
(blocks of 10 ps) after --equil-ps.
"""

import argparse
import time

import numpy as np
import openmm as mm
from openmm import app, unit

import pgm_prmtop

ap = argparse.ArgumentParser()
ap.add_argument("prmtop")
ap.add_argument("rst7")
ap.add_argument("--platform", default="CUDA")
ap.add_argument("--precision", default="mixed")
ap.add_argument("--ps", type=float, default=100)
ap.add_argument("--equil-ps", type=float, default=20)
ap.add_argument("--every", type=int, default=500)
ap.add_argument("--dt-fs", type=float, default=1.0)
ap.add_argument("--temperature", type=float, default=298.0)
ap.add_argument("--tol", type=float, default=1e-5)
ap.add_argument("--beta", type=float, default=3.5)
ap.add_argument("--grid", type=int, nargs=3, default=[36, 36, 36])
ap.add_argument("--cutoff", type=float, default=0.9)
ap.add_argument("--vdw", default="lj")
ap.add_argument("--gvdw", default="slater")
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()

system, top, pgm = pgm_prmtop.create_system(
    a.prmtop, cutoff=a.cutoff, ewald_beta=a.beta, grid=tuple(a.grid), dipole_tol=a.tol, peek=0.65, predictor="mu4",
    vdw=a.vdw, gvdw=a.gvdw if a.vdw == "gvdw" else None, lj_lrc=True,
)  # fmt: skip
crd = app.AmberInpcrdFile(a.rst7)
system.setDefaultPeriodicBoxVectors(*crd.boxVectors)
T = a.temperature * unit.kelvin
system.addForce(mm.MonteCarloBarostat(1 * unit.bar, T, 25))
mass = sum(system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles()))
plat = mm.Platform.getPlatformByName(a.platform)
props = {"Precision": a.precision} if a.platform in ("CUDA", "OpenCL", "HIP") else {}
integ = mm.LangevinMiddleIntegrator(T, 1 / unit.picosecond, a.dt_fs * unit.femtosecond)
integ.setConstraintTolerance(1e-8)
integ.setRandomNumberSeed(a.seed)
ctx = mm.Context(system, integ, plat, props)
ctx.setPositions(crd.positions)
ctx.setPeriodicBoxVectors(*crd.boxVectors)
ctx.applyConstraints(1e-8)
ctx.setVelocitiesToTemperature(T, a.seed)
nsteps = int(round(a.ps * 1000 / a.dt_fs))
print(f"# NPT {a.platform} {a.precision}, {a.temperature} K, 1 bar, dt {a.dt_fs} fs, {a.ps} ps, tol {a.tol}")
print("# step     t/ps    density/(kg/m3)  epot             iters")
rows = []
t0 = time.time()
for step in range(0, nsteps + 1, a.every):
    if step > 0:
        integ.step(a.every)
    s = ctx.getState(getEnergy=True)
    v = s.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)
    rho = mass * 1.66053906660e-27 / (v * 1e-27)
    ep = s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    t = step * a.dt_fs * 1e-3
    rows.append((t, rho, ep))
    print(f"{step:7d} {t:8.2f} {rho:12.3f} {ep:16.4f} {pgm.getInducedDipoleIterations(ctx):4d}", flush=True)
wall = time.time() - t0
r = np.array(rows)
r = r[r[:, 0] >= a.equil_ps]
nb = max(1, int((r[-1, 0] - r[0, 0]) // 10))
blocks = [b[:, 1].mean() for b in np.array_split(r, nb)]
err = np.std(blocks, ddof=1) / np.sqrt(nb) if nb > 1 else float("nan")
print(f"# density after {a.equil_ps} ps: {r[:, 1].mean():.2f} +- {err:.2f} kg/m3 ({nb} blocks), fluctuation {r[:, 1].std():.2f}")
print(f"# {nsteps} steps in {wall:.1f} s: {nsteps * a.dt_fs * 1e-6 / wall * 86400:.2f} ns/day")
