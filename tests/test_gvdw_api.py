"""Unit tests of GVDWForce on small random systems, against an independent brute-force Python sum.

    python test_gvdw_api.py [--platforms Reference CPU OpenCL]

Covers: both repulsion forms; NoCutoff, CutoffNonPeriodic, CutoffPeriodic (triclinic box); exceptions
(excluded and with own parameters, with and without periodic exceptions); the dispersion correction;
finite differences; updateParametersInContext; XML serialization; createExceptionsFromBonds; input
validation.  Exits with the number of failures.
"""
import argparse
import math
import sys

import numpy as np
import openmm as mm
from openmm import unit

import openmmpgm

ap = argparse.ArgumentParser()
ap.add_argument("--platforms", nargs="+", default=["Reference", "CPU", "OpenCL"])
a = ap.parse_args()
G = openmmpgm.GVDWForce
fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fails.append(name)


# ------------------------------------------------------------------ independent reference
C0 = 8 / (9 * math.pi)


def G_closed(y):
    e = math.exp(-y * y)
    A = math.erf(y) - 2 / math.sqrt(math.pi) * y * (1 + 2 * y * y / 3) * e
    g = 4 / (3 * math.sqrt(math.pi)) * y**3 * e
    return (A * A + 0.5 * g * g) / y**6


def G_of(y):
    # below 0.6: Richardson-free check through the closed form at 0.6 and a short Taylor series
    if y < 0.6:
        c = [1.0, -2.0, 58 / 25, -188 / 105, 2222 / 2205, -1532 / 3465, 64244 / 405405, -162056 / 3378375,
             1587734 / 126351225, -211108 / 72747675, 11902564 / 19860115275, -611992 / 5465775315,
             755028172 / 39529267903125, -213335384 / 71152682225625, 15291334888 / 35078272337233125]
        return C0 * sum(ck * y ** (2 * k) for k, ck in enumerate(c))
    return G_closed(y)


def pair_energy(r, beta, A, C6, B, rep):
    y = beta * r
    erep = A * math.exp(-B * y * y) if rep == G.Gaussian else A * math.exp(-B * y)
    return erep - C6 * beta**6 * G_of(y)


def min_image(d, box):
    if box is None:
        return d
    d = d - box[2] * round(d[2] / box[2][2])
    d = d - box[1] * round(d[1] / box[1][1])
    return d - box[0] * round(d[0] / box[0][0])


def brute_energy(par, exc, x, box, method, rc, rep, exc_periodic, lrc):
    """par: (n, 4) radius, a, c, b; exc: {(i, j): (A, C6, B)}."""
    n, e = len(x), 0.0
    for i in range(n):
        for j in range(i + 1, n):
            key = (i, j)
            beta = 1 / math.sqrt(2 * (par[i, 0] ** 2 + par[j, 0] ** 2))
            if key in exc:
                A, C6, B = exc[key]
                if A == 0 and C6 == 0:
                    continue
                d = x[i] - x[j]
                if exc_periodic and method == G.CutoffPeriodic:
                    d = min_image(d, box)
                e += pair_energy(np.linalg.norm(d), beta, A, C6, B, rep)
                continue
            d = min_image(x[i] - x[j], box if method == G.CutoffPeriodic else None)
            r = np.linalg.norm(d)
            if method != G.NoCutoff and r >= rc:
                continue
            e += pair_energy(r, beta, par[i, 1] * par[j, 1], par[i, 2] * par[j, 2], 0.5 * (par[i, 3] + par[j, 3]), rep)
    if lrc and method == G.CutoffPeriodic:
        e += -2 * math.pi * par[:, 2].sum() ** 2 / (3 * box[0][0] * box[1][1] * box[2][2] * rc**3)
    return e


# ------------------------------------------------------------------ test system
rng = np.random.default_rng(7)
n = 48
box = np.array([[2.6, 0, 0], [0.4, 2.5, 0], [-0.3, 0.6, 2.4]])   # reduced triclinic
x = rng.random((n, 3)) @ box
par = np.column_stack([rng.uniform(0.05, 0.09, n), rng.uniform(10, 40, n), rng.uniform(0.02, 0.06, n),
                       rng.uniform(0.5, 4.5, n)])
par[::7, 1:3] = 0.0                       # some atoms without GVDW
exc = {}
for k in range(0, n - 1, 5):
    exc[(k, k + 1)] = (0.0, 0.0, 0.0)     # excluded
for k in range(2, n - 3, 9):
    exc[(k, k + 3)] = (150.0, 0.0011, 2.0)   # own parameters
rc = 1.0


def make_force(method, rep, exc_periodic=False, lrc=False):
    f = G()
    for p in par:
        f.addParticle(*map(float, p))
    for (i, j), v in exc.items():
        f.addException(i, j, *v)
    f.setNonbondedMethod(method)
    f.setCutoffDistance(rc)
    f.setRepulsionForm(rep)
    f.setExceptionsUsePeriodicBoundaryConditions(exc_periodic)
    f.setUseDispersionCorrection(lrc)
    return f


def make_context(force, platform, precision="double"):
    s = mm.System()
    for _ in range(n):
        s.addParticle(1.0)
    s.setDefaultPeriodicBoxVectors(*[mm.Vec3(*v) for v in box])
    s.addForce(force)
    props = {"Precision": precision} if platform in ("OpenCL", "CUDA", "HIP") else {}
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(platform), props)
    ctx.setPositions(x)
    return ctx


def energy_forces(ctx, pos=None):
    if pos is not None:
        ctx.setPositions(pos)
    st = ctx.getState(getEnergy=True, getForces=True)
    return (st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            np.asarray(st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)))


platforms = [p for p in a.platforms if p in [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]]
cases = [(G.NoCutoff, False, False), (G.CutoffNonPeriodic, False, False), (G.CutoffPeriodic, False, True),
         (G.CutoffPeriodic, True, False)]
for rep in (G.Slater, G.Gaussian):
    for method, exc_periodic, lrc in cases:
        tag = f"rep {rep} method {method} excPBC {int(exc_periodic)} lrc {int(lrc)}"
        e_ref = brute_energy(par, exc, x, box, method, rc, rep, exc_periodic, lrc)
        ctx = make_context(make_force(method, rep, exc_periodic, lrc), "Reference")
        e0, f0 = energy_forces(ctx)
        check(f"Reference energy = brute force, {tag}", abs(e0 - e_ref) <= 1e-9 * max(1, abs(e_ref)), f"{e0:.10f} {e_ref:.10f}")
        # finite differences on three atoms
        h, worst = 1e-6, 0.0
        for i in (1, 17, 30):
            for k in range(3):
                xp, xm = x.copy(), x.copy()
                xp[i, k] += h
                xm[i, k] -= h
                fd = -(energy_forces(ctx, xp)[0] - energy_forces(ctx, xm)[0]) / (2 * h)
                worst = max(worst, abs(fd - f0[i, k]) / max(1.0, abs(f0[i, k])))
        ctx.setPositions(x)
        check(f"Reference forces = finite differences, {tag}", worst < 1e-5, f"{worst:.1e}")
        for p in platforms:
            if p == "Reference":
                continue
            for prec in (["double", "mixed"] if p != "CPU" else ["-"]):
                try:
                    e1, f1 = energy_forces(make_context(make_force(method, rep, exc_periodic, lrc), p, prec))
                except Exception as ex:
                    check(f"{p} {prec} runs, {tag}", False, str(ex).splitlines()[0][:100])
                    continue
                tol = 1e-12 if prec in ("double", "-") else 1e-5
                de = abs(e1 - e0) / max(1, abs(e0))
                df = np.abs(f1 - f0).max() / max(1, np.abs(f0).max())
                check(f"{p} {prec} = Reference, {tag}", de < 10 * tol and df < 10 * tol * 10, f"dE {de:.1e} dF {df:.1e}")

# ------------------------------------------------------------------ updateParametersInContext
for p in platforms:
    f = make_force(G.CutoffPeriodic, G.Slater, False, True)
    ctx = make_context(f, p, "double")
    for i in range(0, n, 3):
        R, aa, c, b = f.getParticleParameters(i)
        f.setParticleParameters(i, R * 1.1, aa * 0.9, c * 1.2, b + 0.3)
    k = [kk for kk in range(f.getNumExceptions()) if f.getExceptionParameters(kk)[2] != 0][0]
    i, j, A, C6, B = f.getExceptionParameters(k)
    f.setExceptionParameters(k, i, j, A * 2, C6 * 0.5, B + 1)
    f.updateParametersInContext(ctx)
    e_upd = energy_forces(ctx)[0]
    e_new = energy_forces(make_context(mm.XmlSerializer.clone(f), p, "double"))[0]   # a System owns its forces
    check(f"updateParametersInContext, {p}", abs(e_upd - e_new) <= 1e-10 * max(1, abs(e_new)), f"{e_upd:.10f} {e_new:.10f}")
    if True:   # the same contract on every platform
        f.setExceptionParameters(k, i, j, 0.0, 0.0, 0.0)
        try:
            f.updateParametersInContext(ctx)
            check(f"updateParametersInContext rejects a change of the excluded set, {p}", False)
        except Exception:
            check(f"updateParametersInContext rejects a change of the excluded set, {p}", True)

# ------------------------------------------------------------------ serialization
f = make_force(G.CutoffPeriodic, G.Gaussian, True, True)
f.setForceGroup(3)
f.setName("my gvdw")
xml = mm.XmlSerializer.serialize(f)
g0 = mm.XmlSerializer.deserialize(xml)   # owned by Python; cast() gives a view of it
ok = openmmpgm.GVDWForce.isinstance(g0)
if ok:
    g = openmmpgm.GVDWForce.cast(g0)
    ok = (g.getNumParticles() == n and g.getNumExceptions() == len(exc) and g.getForceGroup() == 3
          and g.getName() == "my gvdw" and g.getRepulsionForm() == G.Gaussian and g.getUseDispersionCorrection()
          and g.getExceptionsUsePeriodicBoundaryConditions()
          and all(np.allclose(g.getParticleParameters(i), f.getParticleParameters(i), rtol=1e-15) for i in range(n)))
    e1 = energy_forces(make_context(mm.XmlSerializer.clone(f), "Reference"))[0]
    e2 = energy_forces(make_context(g0, "Reference"))[0]
    ok = ok and e1 == e2
check("GVDWForce XML round trip", ok)
pg = openmmpgm.PGMForce()
pg.addParticle(-0.5, 0.07, 0.001)
pg.addParticle(0.5, 0.05, 0.0)
pg.addCovalentDipole(0, 1, 0.003)
pg.setNonbondedMethod(pg.PME)
pg.setPMEParameters(3.5, 24, 25, 27)
pg.setPredictor(pg.Mu3)
pg.setCoulombConstant(138.935458)
h = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(pg))
ok = openmmpgm.PGMForce.isinstance(h)
if ok:
    h = openmmpgm.PGMForce.cast(h)
    ok = (h.getNumParticles() == 2 and h.getNumCovalentDipoles() == 1 and list(h.getPMEParameters()) == [3.5, 24, 25, 27]
          and h.getPredictor() == pg.Mu3 and h.getCoulombConstant() == 138.935458
          and list(h.getParticleParameters(0)) == list(pg.getParticleParameters(0)))
check("PGMForce XML round trip", ok)

# ------------------------------------------------------------------ createExceptionsFromBonds
f = G()
for _ in range(6):
    f.addParticle(0.07, 4.0, 0.05, 2.0)
f.addException(0, 5, 1.0, 2.0, 3.0)
f.createExceptionsFromBonds([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 0.5, 0.25)
got = {tuple(sorted(f.getExceptionParameters(k)[:2])): f.getExceptionParameters(k)[2:] for k in range(f.getNumExceptions())}
want12_13 = {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 2), (1, 3), (2, 4), (3, 5)}
want14 = {(0, 3), (1, 4), (2, 5)}
ok = set(got) == want12_13 | want14 | {(0, 5)}
ok = ok and all(got[k] == [0.0, 0.0, 0.0] for k in want12_13)
ok = ok and all(np.allclose(got[k], [0.5 * 16.0, 0.25 * 0.0025, 2.0]) for k in want14) and got[(0, 5)] == [1.0, 2.0, 3.0]
check("createExceptionsFromBonds", ok, str(len(got)))

# ------------------------------------------------------------------ validation
for bad in (lambda: G().addParticle(0.0, 1, 1, 1), lambda: G().addParticle(0.07, -1, 1, 1),
            lambda: G().setCutoffDistance(0.0)):
    try:
        bad()
        check("input validation", False)
    except Exception:
        check("input validation", True)
f = G()
f.addParticle(0.07, 1, 1, 1)
f.addParticle(0.07, 1, 1, 1)
f.addException(0, 1, 0, 0, 0)
try:
    f.addException(1, 0, 1, 1, 1)
    check("duplicate exception rejected", False)
except Exception:
    check("duplicate exception rejected", True)

print(f"{len(fails)} failures" if fails else "ALL PASSED")
sys.exit(len(fails))
