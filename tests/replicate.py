"""Replicate an OpenMM System (molecules in a periodic box, any reduced triclinic one) nx x ny x nz times along
its box vectors a, b, c (as gmx genconf -nbox), for benchmarks.

    system2, positions2, box2 = replicate(system, positions, box, (2, 2, 2))

Forces handled: HarmonicBond/Angle, PeriodicTorsion, NonbondedForce (particles, exceptions), CMMotionRemover,
PGMForce (particles, covalent dipoles), GVDWForce (particles, exceptions), TailCorrectionForce (coefficient
times the number of copies squared: it sums over all pairs); constraints.  Positions in nm (N x 3).
"""
import numpy as np
import openmm as mm
from openmm import unit

import openmmpgm


def replicate(system, positions, box, reps):
    n = system.getNumParticles()
    box = np.asarray(box, dtype=float)
    pos = np.asarray(positions, dtype=float)
    shifts = [i * box[0] + j * box[1] + k * box[2] for i in range(reps[0]) for j in range(reps[1]) for k in range(reps[2])]
    m = len(shifts)
    new = mm.System()
    for t in range(m):
        for i in range(n):
            new.addParticle(system.getParticleMass(i))
    for t in range(m):
        for c in range(system.getNumConstraints()):
            i, j, d = system.getConstraintParameters(c)
            new.addConstraint(i + t * n, j + t * n, d)
    nb_box = box * np.array(reps, dtype=float)[:, None]
    new.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) * unit.nanometer for row in nb_box])
    for f in system.getForces():
        name = type(f).__name__
        if isinstance(f, mm.HarmonicBondForce):
            g = mm.HarmonicBondForce()
            for t in range(m):
                for b in range(f.getNumBonds()):
                    i, j, r, k = f.getBondParameters(b)
                    g.addBond(i + t * n, j + t * n, r, k)
        elif isinstance(f, mm.HarmonicAngleForce):
            g = mm.HarmonicAngleForce()
            for t in range(m):
                for b in range(f.getNumAngles()):
                    i, j, l, a, k = f.getAngleParameters(b)
                    g.addAngle(i + t * n, j + t * n, l + t * n, a, k)
        elif isinstance(f, mm.PeriodicTorsionForce):
            g = mm.PeriodicTorsionForce()
            for t in range(m):
                for b in range(f.getNumTorsions()):
                    i, j, l, p, per, ph, k = f.getTorsionParameters(b)
                    g.addTorsion(i + t * n, j + t * n, l + t * n, p + t * n, per, ph, k)
        elif isinstance(f, mm.NonbondedForce):
            g = mm.NonbondedForce()
            g.setNonbondedMethod(f.getNonbondedMethod())
            g.setCutoffDistance(f.getCutoffDistance())
            g.setUseDispersionCorrection(f.getUseDispersionCorrection())
            g.setUseSwitchingFunction(f.getUseSwitchingFunction())
            g.setReactionFieldDielectric(f.getReactionFieldDielectric())
            for t in range(m):
                for i in range(n):
                    g.addParticle(*f.getParticleParameters(i))
            for t in range(m):
                for e in range(f.getNumExceptions()):
                    i, j, q, s, eps = f.getExceptionParameters(e)
                    g.addException(i + t * n, j + t * n, q, s, eps)
        elif isinstance(f, mm.CMMotionRemover):
            g = mm.CMMotionRemover(f.getFrequency())
        elif openmmpgm.PGMForce.isinstance(f):
            f = openmmpgm.PGMForce.cast(f)
            g = openmmpgm.PGMForce()
            for t in range(m):
                for i in range(n):
                    g.addParticle(*f.getParticleParameters(i))
            for t in range(m):
                for c in range(f.getNumCovalentDipoles()):
                    i, j, s = f.getCovalentDipoleParameters(c)
                    g.addCovalentDipole(i + t * n, j + t * n, s)
            g.setNonbondedMethod(f.getNonbondedMethod())
            g.setCutoffDistance(f.getCutoffDistance())
            for meth in ["Polarization", "InducedDipoleTolerance", "MaxInducedDipoleIterations", "PeekCoefficient",
                         "Predictor", "CoulombConstant", "EwaldErrorTolerance", "PMEOrder"]:
                if hasattr(f, "get" + meth):
                    getattr(g, "set" + meth)(getattr(f, "get" + meth)())
            a, nx, ny, nz = f.getPMEParameters()
            g.setPMEParameters(a, nx, ny, nz)
        elif openmmpgm.GVDWForce.isinstance(f):
            f = openmmpgm.GVDWForce.cast(f)
            g = openmmpgm.GVDWForce()
            g.setRepulsionForm(f.getRepulsionForm())
            g.setNonbondedMethod(f.getNonbondedMethod())
            g.setCutoffDistance(f.getCutoffDistance())
            g.setUseDispersionCorrection(f.getUseDispersionCorrection())
            g.setExceptionsUsePeriodicBoundaryConditions(f.getExceptionsUsePeriodicBoundaryConditions())
            for t in range(m):
                for i in range(n):
                    g.addParticle(*f.getParticleParameters(i))
            for t in range(m):
                for e in range(f.getNumExceptions()):
                    i, j, A, C6, B = f.getExceptionParameters(e)
                    g.addException(i + t * n, j + t * n, A, C6, B)
        elif openmmpgm.TailCorrectionForce.isinstance(f):
            f = openmmpgm.TailCorrectionForce.cast(f)
            g = openmmpgm.TailCorrectionForce(f.getCoefficient() * m * m)
        else:
            raise ValueError("replicate: unsupported force " + name)
        g.setForceGroup(f.getForceGroup())
        new.addForce(g)
    newpos = np.concatenate([pos + s for s in shifts])
    return new, newpos, nb_box
