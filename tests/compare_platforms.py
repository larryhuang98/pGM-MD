"""Compare PGMForce on two OpenMM platforms (default: CPU = Reference kernel vs OpenCL/CUDA).

    python compare_platforms.py PRMTOP REF.npz [--platform OpenCL] [--precision double] [--nmol 40]

Runs six cases: no cutoff (first nmol molecules, no box) and PME (the whole box), each with the
levels q, qp and qpi; prints energy differences, force deviations and induced-dipole deviations.
"""

import argparse
import faulthandler

import numpy as np
import openmm as mm
from openmm import unit

import pgm_prmtop

faulthandler.enable()
ap = argparse.ArgumentParser()
ap.add_argument("prmtop")
ap.add_argument("ref")
ap.add_argument("--platform", default="OpenCL")
ap.add_argument("--precision", default="double")
ap.add_argument("--ref-platform", default="CPU")
ap.add_argument("--nmol", type=int, default=40)
ap.add_argument("--cutoff", type=float, default=0.9)
ap.add_argument("--cases", default="nocut-q,nocut-qp,nocut-qpi,pme-q,pme-qp,pme-qpi")
a = ap.parse_args()

d = np.load(a.ref)
H, pos = d["H"], d["pos"]


def build(case):
    method, level = case.split("-")
    periodic = method == "pme"
    system, top, pgm = pgm_prmtop.create_system(
        a.prmtop, periodic=periodic, cutoff=a.cutoff, ewald_beta=3.5, grid=(36, 36, 36), polarization=level == "qpi", lj=False,
        peek=0.0, predictor="none", dipole_tol=1e-8, max_iter=500,
    )  # fmt: skip
    if level == "q":
        for k in range(pgm.getNumCovalentDipoles()):
            i, j, c = pgm.getCovalentDipoleParameters(k)
            pgm.setCovalentDipoleParameters(k, i, j, 0.0)
    x = pos
    if not periodic:
        # keep the first nmol molecules (3 atoms each): a new system with those atoms only
        n = 3 * a.nmol
        sub = mm.System()
        new = type(pgm)()
        new.setNonbondedMethod(new.NoCutoff)
        new.setPolarization(pgm.getPolarization())
        new.setInducedDipoleTolerance(1e-8)
        new.setMaxInducedDipoleIterations(500)
        new.setPeekCoefficient(0.0)
        new.setPredictor(new.NoPredictor)
        for i in range(n):
            sub.addParticle(system.getParticleMass(i))
            new.addParticle(*pgm.getParticleParameters(i))
        for k in range(pgm.getNumCovalentDipoles()):
            i, j, c = pgm.getCovalentDipoleParameters(k)
            if i < n and j < n:
                new.addCovalentDipole(i, j, c)
        sub.addForce(new)
        system, pgm, x = sub, new, pos[:n]
    else:
        system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*H[k]) for k in range(3)])
    return system, pgm, x


def evaluate(system, pgm, x, platform, precision):
    plat = mm.Platform.getPlatformByName(platform)
    props = {"Precision": precision} if platform in ("CUDA", "OpenCL", "HIP") else {}
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), plat, props)
    if system.usesPeriodicBoundaryConditions():
        ctx.setPeriodicBoxVectors(*[mm.Vec3(*H[k]) for k in range(3)])
    ctx.setPositions(x)
    s = ctx.getState(getEnergy=True, getForces=True)
    e = s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f = s.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    mu = pgm.getInducedDipoles(ctx)
    p = pgm.getPermanentDipoles(ctx)
    it = pgm.getInducedDipoleIterations(ctx)
    return e, np.asarray(f), mu, p, it


for case in a.cases.split(","):
    system, pgm, x = build(case)
    e0, f0, mu0, p0, it0 = evaluate(system, pgm, x, a.ref_platform, "double")
    e1, f1, mu1, p1, it1 = evaluate(system, pgm, x, a.platform, a.precision)
    df = np.abs(f1 - f0).max()
    print(f"{case:10s} E {e0:.6f} vs {e1:.6f} rel {e1 / e0 - 1:+.2e} | max|dF| {df:.2e} (rms F {np.sqrt((f0**2).mean()):.1e})"
          f" | max|dp| {np.abs(p1 - p0).max():.1e} max|dmu| {np.abs(mu1 - mu0).max():.1e} | it {it0}/{it1}", flush=True)
    slope = float(np.sum(f0 * f1) / np.sum(f0 * f0))
    resid = f1 - slope * f0
    print(f"    force slope {slope:.6f}, residual rms {np.sqrt((resid**2).mean()):.3e}; atoms 0-2 ref {f0[:3].round(2).tolist()} new {f1[:3].round(2).tolist()}", flush=True)
