"""Tiny PME checks of PGMForce: two charges (and a dipole) in a cubic or triclinic box, two platforms."""
import sys
import numpy as np
import openmm as mm
import openmmpgm

plat2 = sys.argv[1] if len(sys.argv) > 1 else "OpenCL"
PREC = sys.argv[2] if len(sys.argv) > 2 else "double"


def run(box, pos, q, cov, plat, cutoff=0.05, grid=(20, 20, 20)):
    s = mm.System()
    s.setDefaultPeriodicBoxVectors(*[mm.Vec3(*b) for b in box])
    f = openmmpgm.PGMForce()
    for qi in q:
        s.addParticle(1.0)
        f.addParticle(qi, 0.07, 0.0)
    for i, j, c in cov:
        f.addCovalentDipole(i, j, c)
    f.setNonbondedMethod(f.PME)
    f.setCutoffDistance(cutoff)
    f.setPMEParameters(3.5, *grid)
    f.setPolarization(f.NoPolarization)
    s.addForce(f)
    props = {"Precision": PREC} if plat in ("OpenCL", "CUDA") else {}
    c = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(plat), props)
    c.setPositions(pos)
    st = c.getState(getEnergy=True, getForces=True)
    return st.getPotentialEnergy()._value, np.asarray(st.getForces(asNumpy=True)._value)


cubic = np.diag([2.0, 2.0, 2.0])
tric = np.array([[2.0, 0, 0], [-0.6, 1.9, 0], [-0.6, -0.9, 1.7]])
pos = np.array([[0.31, 0.42, 0.53], [1.11, 0.97, 1.23], [0.5, 1.4, 0.2]])
for name, box in (("cubic", cubic), ("triclinic", tric)):
    for label, q, cov in (("charges", [1.0, -1.0, 0.0], []), ("dipole", [0.0, 0.0, 0.0], [(2, 0, 0.05)]),
                          ("charge+dipole", [0.7, -0.4, -0.3], [(2, 0, 0.05), (0, 1, -0.03)])):
        e0, f0 = run(box, pos, q, cov, "CPU")
        e1, f1 = run(box, pos, q, cov, plat2)
        print(f"{name:9s} {label:14s} E {e0:.8f} {e1:.8f}  max|dF| {np.abs(f1 - f0).max():.2e}  F0[0] {f0[0].round(4)} F1[0] {f1[0].round(4)}", flush=True)

# OpenMM's own NonbondedForce PME on the same platforms (is the platform's FFT sane?)
for name, box in (("cubic", cubic), ("triclinic", tric)):
    out = []
    for plat in ("CPU", plat2):
        s = mm.System()
        s.setDefaultPeriodicBoxVectors(*[mm.Vec3(*b) for b in box])
        nb = mm.NonbondedForce()
        nb.setNonbondedMethod(nb.PME)
        nb.setCutoffDistance(0.5)
        nb.setPMEParameters(3.5, 20, 20, 20)
        for qi in (1.0, -1.0, 0.0):
            s.addParticle(1.0)
            nb.addParticle(qi, 0.3, 0.0)
        s.addForce(nb)
        props = {"Precision": PREC} if plat in ("OpenCL", "CUDA") else {}
        c = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(plat), props)
        c.setPositions(pos)
        out.append(c.getState(getEnergy=True).getPotentialEnergy()._value)
    print(f"NonbondedForce PME {name}: CPU {out[0]:.8f}  {plat2} {out[1]:.8f}", flush=True)
