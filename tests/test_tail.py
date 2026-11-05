"""TailCorrectionForce: E = c/V on every platform given, its force group, XML round trip, no forces."""
import sys
import numpy as np
import openmm as mm
from openmm import unit
import openmmpgm

plats = sys.argv[1:] or ["Reference", "CUDA"]
bad = 0
c = -1234.5
s = mm.System()
for i in range(3):
    s.addParticle(1.0)
box = [mm.Vec3(2.0, 0, 0), mm.Vec3(0.3, 2.1, 0), mm.Vec3(-0.2, 0.4, 2.2)]
s.setDefaultPeriodicBoxVectors(*box)
f = openmmpgm.TailCorrectionForce(c)
f.setForceGroup(3)
s.addForce(f)
xml = mm.XmlSerializer.serialize(s)
s2 = mm.XmlSerializer.deserialize(xml)
g = openmmpgm.TailCorrectionForce.cast(s2.getForce(0))
ok = g.getCoefficient() == c and g.getForceGroup() == 3
print("ok  " if ok else "FAIL", "XML round trip", g.getCoefficient(), g.getForceGroup()); bad += not ok
V = 2.0 * 2.1 * 2.2
for p in plats:
    ctx = mm.Context(s2, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(p))
    ctx.setPositions(np.random.rand(3, 3))
    e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    e0 = ctx.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    F = ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    ok = abs(e - c / V) < 1e-9 * abs(c / V) and e0 == 0 and np.abs(F).max() == 0
    print("ok  " if ok else "FAIL", p, "E", e, "expected", c / V, "group 0", e0); bad += not ok
print("ALL PASSED" if not bad else f"{bad} failures")
sys.exit(bad)
