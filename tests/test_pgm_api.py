"""API and contract tests of PGMForce (and GVDWForce at r = 0) on every available platform.

    python test_pgm_api.py PRMTOP RST7 [--platforms Reference CPU OpenCL]

PRMTOP/RST7: a periodic pmemd-pgm water box (examples/water512).  OpenCL runs in mixed precision (pocl's
double-precision FFT is broken).  Covers: setter validation; the checks
at Context creation (PME grid >= order, Gaussian width vs cutoff); finiteness at r = 0 (two sites on top
of each other, PGMForce and GVDWForce); getInducedDipoles before the first evaluation;
updateParametersInContext with atom reordering (cutoff, OpenCL/CUDA) against a fresh Context; the
same-step predictor rule (a second evaluation at the same step does not change the next step).
Exits with the number of failures.
"""
import argparse
import math
import sys

import numpy as np
import openmm as mm
from openmm import app, unit

import openmmpgm
import pgm_prmtop

ap = argparse.ArgumentParser()
ap.add_argument("prmtop")
ap.add_argument("rst7")
ap.add_argument("--platforms", nargs="+", default=["Reference", "CPU", "OpenCL"])
a = ap.parse_args()
names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
platforms = [p for p in a.platforms if p in names]
fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fails.append(name)


def raises(f):
    try:
        f()
        return False
    except Exception:
        return True


def context(system, platform, pos, box=None, precision="double"):
    props = {"Precision": precision} if platform in ("OpenCL", "CUDA", "HIP") else {}
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(platform), props)
    if box is not None:
        ctx.setPeriodicBoxVectors(*box)
    ctx.setPositions(pos)
    return ctx


def energy(ctx, forces=False):
    st = ctx.getState(getEnergy=True, getForces=forces)
    e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    if forces:
        return e, np.asarray(st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    return e


# ------------------------------------------------------------------ setters
f = openmmpgm.PGMForce()
check("setters reject illegal values", all(raises(g) for g in (
    lambda: f.setCutoffDistance(0), lambda: f.setPMEParameters(3.5, 0, 10, 10), lambda: f.setPMEParameters(-1, 10, 10, 10),
    lambda: f.setEwaldErrorTolerance(0), lambda: f.setInducedDipoleTolerance(0), lambda: f.setMaxInducedDipoleIterations(-1),
    lambda: f.setPeekCoefficient(-0.1), lambda: f.setCoulombConstant(0), lambda: f.setPMEOrder(3),
    lambda: f.addParticle(0.0, 0.0, 0.0), lambda: f.addParticle(0.0, 0.07, -1.0))))

# ------------------------------------------------------------------ water box systems
crd = app.AmberInpcrdFile(a.rst7)
box = crd.boxVectors
pos = np.asarray(crd.positions.value_in_unit(unit.nanometer))


def water_system(**kw):
    s, _, pgm = pgm_prmtop.create_system(a.prmtop, cutoff=0.9, ewald_beta=3.5, grid=(36, 36, 36), lj=False,
                                         dipole_tol=1e-8, **kw)
    s.setDefaultPeriodicBoxVectors(*box)
    return s


# checks at Context creation
s = water_system()
pgm = [openmmpgm.PGMForce.cast(s.getForce(k)) for k in range(s.getNumForces()) if openmmpgm.PGMForce.isinstance(s.getForce(k))][0]
pgm.setPMEParameters(3.5, 4, 36, 36)
check("PME grid smaller than the order rejected", raises(lambda: context(s, "Reference", pos, box)))
s = water_system()
pgm = [openmmpgm.PGMForce.cast(s.getForce(k)) for k in range(s.getNumForces()) if openmmpgm.PGMForce.isinstance(s.getForce(k))][0]
q, R, al = pgm.getParticleParameters(0)
pgm.setParticleParameters(0, q, 0.3, al)
check("too wide Gaussian for the cutoff rejected", raises(lambda: context(s, "Reference", pos, box)))

# getInducedDipoles before the first evaluation, the same on every platform
mus = {}
for p in platforms:
    s = water_system()
    pgm = [openmmpgm.PGMForce.cast(s.getForce(k)) for k in range(s.getNumForces()) if openmmpgm.PGMForce.isinstance(s.getForce(k))][0]
    ctx = context(s, p, pos, box, "mixed" if p == "OpenCL" else "double")
    mus[p] = pgm.getInducedDipoles(ctx)
    del ctx
ref = mus.get("Reference")
for p, mu in mus.items():
    check(f"getInducedDipoles before evaluation, {p}", np.abs(mu).max() > 1e-3 and (ref is None or np.abs(mu - ref).max() < 1e-5),
          f"max|mu| {np.abs(mu).max():.3e}")

# updateParametersInContext with atom reordering (the cutoff makes OpenCL/CUDA reorder molecules)
for p in platforms:
    s = water_system()
    pgm = [openmmpgm.PGMForce.cast(s.getForce(k)) for k in range(s.getNumForces()) if openmmpgm.PGMForce.isinstance(s.getForce(k))][0]
    prec = "mixed" if p == "OpenCL" else "double"   # pocl's double-precision FFT is broken
    ctx = context(s, p, pos, box, prec)
    integ = ctx.getIntegrator()
    ctx.setVelocitiesToTemperature(300, 1)
    integ.step(30)                      # reordering happens on the GPU platforms
    x = np.asarray(ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    for i in (30, 31, 32):              # one water: different charges -> no longer identical to the others
        qq, RR, aa = pgm.getParticleParameters(i)
        pgm.setParticleParameters(i, qq * 1.1, RR, aa * 0.9)
    pgm.updateParametersInContext(ctx)
    ctx.setPositions(x)
    e1 = energy(ctx)
    integ.step(30)                      # more reorders must keep the changed water's parameters with it
    x2 = np.asarray(ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    e2 = energy(ctx)
    fresh = context(mm.XmlSerializer.clone(s), p, x2, box, prec)
    e3 = energy(fresh)
    fresh0 = context(mm.XmlSerializer.clone(s), p, x, box, prec)
    e0 = energy(fresh0)
    del ctx, fresh, fresh0
    check(f"updateParametersInContext keeps parameters with their atoms, {p}",
          abs(e1 - e0) < 1e-6 * abs(e0) and abs(e2 - e3) < 1e-6 * abs(e3), f"dE {e1 - e0:.2e} {e2 - e3:.2e}")

# same-step evaluations do not change the predictor history
for p in platforms:
    traj = []
    for extra in (False, True):
        s = water_system(predictor="mu4")
        pgm = [openmmpgm.PGMForce.cast(s.getForce(k)) for k in range(s.getNumForces()) if openmmpgm.PGMForce.isinstance(s.getForce(k))][0]
        pgm.setInducedDipoleTolerance(1e-4)
        ctx = context(s, p, pos, box, "mixed" if p == "OpenCL" else "double")
        ctx.setVelocitiesToTemperature(300, 3)
        its = []
        for step in range(8):
            ctx.getIntegrator().step(1)
            if extra:   # an evaluation at the same step with displaced positions (a barostat trial)
                st = ctx.getState(getPositions=True)
                ctx.setPositions(st.getPositions(asNumpy=True) * 1.001)
                energy(ctx)
                ctx.setPositions(st.getPositions())
            its.append(pgm.getInducedDipoleIterations(ctx))
        traj.append(energy(ctx))
        del ctx
    check(f"same-step evaluations leave the predictor alone, {p}", abs(traj[0] - traj[1]) < 1e-6 * abs(traj[0]),
          f"{traj[0]:.6f} {traj[1]:.6f}")

# r = 0: two sites on top of each other (PGMForce and GVDWForce, no cutoff)
for p in platforms:
    s = mm.System()
    for _ in range(3):
        s.addParticle(1.0)
    g = openmmpgm.PGMForce()
    for qq, RR, aa in ((0.4, 0.07, 0.001), (-0.3, 0.05, 0.0005), (-0.1, 0.06, 0.0)):
        g.addParticle(qq, RR, aa)
    g.addCovalentDipole(0, 2, 0.002)
    s.addForce(g)
    v = openmmpgm.GVDWForce()
    for RR in (0.07, 0.05, 0.06):
        v.addParticle(RR, 20.0, 0.04, 2.0)
    v.setRepulsionForm(v.Gaussian)
    s.addForce(v)
    x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.15, 0.02, -0.03]])
    e, frc = energy(context(s, p, x, None, "mixed" if p == "OpenCL" else "double"), True)
    check(f"finite at r = 0, {p}", math.isfinite(e) and np.isfinite(frc).all(), f"E {e:.6f}")

print(f"{len(fails)} failures" if fails else "ALL PASSED")
sys.exit(len(fails))
