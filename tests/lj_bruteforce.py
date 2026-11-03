"""Brute-force LJ energy of a water box (O-O only) three ways, to check minimum-image conventions.

sequential: OpenMM / pGM-JAX reduction (c, then b, then a); nearest: the shortest image (search n in
-2..2); every: all images within rc.  Also the OpenMM NonbondedForce (Reference) on the same O atoms.
"""
import sys
import numpy as np
import openmm as mm
import pgm_prmtop as P

top, ref = sys.argv[1], sys.argv[2]
rc = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
s = P.read_sections(top)
A = s["LENNARD_JONES_ACOEF"][0] * 4.184e-12
B = s["LENNARD_JONES_BCOEF"][0] * 4.184e-6
d = np.load(ref)
x, H = d["pos"], d["H"]
O = x[s["ATOM_TYPE_INDEX"] == 1]
rng = (-2, -1, 0, 1, 2)
shifts = np.array([[i, j, k] for i in rng for j in rng for k in rng]) @ H
lj = lambda r: np.sum(A / r**12 - B / r**6)
e = {"sequential": 0.0, "nearest": 0.0, "every": 0.0}
npair = {k: 0 for k in e}
nbad = 0
for a in range(len(O)):
    dx = O[a] - O[a + 1:]
    y = dx.copy()
    for c in (2, 1, 0):
        y -= np.round(y[:, c] / H[c, c])[:, None] * H[c]
    r = np.linalg.norm(y, axis=1)
    e["sequential"] += lj(r[r < rc]); npair["sequential"] += (r < rc).sum()
    rr = np.linalg.norm(dx[:, None, :] + shifts[None], axis=2)
    rmin = rr.min(axis=1)
    e["nearest"] += lj(rmin[rmin < rc]); npair["nearest"] += (rmin < rc).sum()
    e["every"] += lj(rr[rr < rc]); npair["every"] += (rr < rc).sum()
    nbad += np.sum((rmin < rc) & (r > rmin + 1e-9))
for k in e:
    print(f"{k:10s} {e[k]:.6f} kJ/mol  {npair[k]} pairs")
print("pairs whose sequential image is not the nearest one (nearest < rc):", nbad)
# OpenMM NonbondedForce on the O atoms only
sysm = mm.System()
sysm.setDefaultPeriodicBoxVectors(*[mm.Vec3(*H[k]) for k in range(3)])
nb = mm.NonbondedForce()
nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
nb.setCutoffDistance(rc)
nb.setUseDispersionCorrection(False)
sig, eps = (A / B) ** (1 / 6), B * B / (4 * A)
for _ in O:
    sysm.addParticle(16.0)
    nb.addParticle(0.0, sig, eps)
sysm.addForce(nb)
for plat in ("Reference", "CPU"):
    ctx = mm.Context(sysm, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(plat))
    ctx.setPositions(O)
    print(plat, "NonbondedForce", ctx.getState(getEnergy=True).getPotentialEnergy())
# the same with every atom wrapped into the reduced box first
Ow = O.copy()
for c in (2, 1, 0):
    Ow -= np.floor(Ow[:, c] / H[c, c])[:, None] * H[c]
for plat in ("Reference", "CPU"):
    ctx = mm.Context(sysm, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(plat))
    ctx.setPositions(Ow)
    print(plat, "NonbondedForce, wrapped positions", ctx.getState(getEnergy=True).getPotentialEnergy())
