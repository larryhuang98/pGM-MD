"""Smoke test of an OpenMM platform: a plain NonbondedForce (PME) water box, then PGMForce alone."""
import sys
import faulthandler
faulthandler.enable()
import numpy as np
import openmm as mm
import pgm_prmtop

plat_name = sys.argv[1] if len(sys.argv) > 1 else "OpenCL"
top, ref = sys.argv[2], sys.argv[3]
prec = sys.argv[4] if len(sys.argv) > 4 else "double"
d = np.load(ref)
H = d["H"]
plat = mm.Platform.getPlatformByName(plat_name)
props = {"Precision": prec}
# 1. OpenMM's own NonbondedForce with charges
sysm = mm.System()
sysm.setDefaultPeriodicBoxVectors(*[mm.Vec3(*H[k]) for k in range(3)])
nb = mm.NonbondedForce()
nb.setNonbondedMethod(mm.NonbondedForce.PME)
nb.setCutoffDistance(0.9)
for i in range(len(d["pos"])):
    sysm.addParticle(1.0)
    nb.addParticle(0.4 if i % 3 else -0.8, 0.3, 0.1)
sysm.addForce(nb)
ctx = mm.Context(sysm, mm.VerletIntegrator(0.001), plat, props)
ctx.setPositions(d["pos"])
print("NonbondedForce ok:", ctx.getState(getEnergy=True).getPotentialEnergy(), flush=True)
del ctx
# 2. PGMForce only, no polarization
system, _, pgm = pgm_prmtop.create_system(top, ewald_beta=3.5, grid=(36, 36, 36), polarization=False, lj=False, peek=0.0, predictor="none")
system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*H[k]) for k in range(3)])
print("creating PGM context", flush=True)
ctx = mm.Context(system, mm.VerletIntegrator(0.001), plat, props)
ctx.setPositions(d["pos"])
print("PGM qp energy:", ctx.getState(getEnergy=True).getPotentialEnergy(), flush=True)
