"""Compare the GROMACS single points of gvdw_14.sh with OpenMM (pgm_prmtop, NoCutoff, same coordinates).
    python gvdw_14_compare.py WORKDIR        (Python with OpenMM and the plugin)
Checks, for the LJ, Slater-GVDW and Gauss-GVDW conversions of qm7b_7203: LJ (SR) + LJ-14 against OpenMM's
NonbondedForce (LJ); with GVDW, LJ (SR) (which then includes the 1-4 pairs, nrexcl = 2) against OpenMM's
GVDWForce (1-4 pairs as exceptions with the full GVDW) and no LJ-14 (pmemd-pgm igvdw = 1: the 1-4 pairs get
the full GVDW and no Lennard-Jones); bonds/angles/dihedrals against OpenMM's bonded forces.  Exits with the
number of failures.
"""
import os
import sys

import numpy as np
import openmm as mm
from openmm import unit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "openmm", "python"))
import openmmpgm  # noqa: E402
import pgm_prmtop  # noqa: E402

wd = sys.argv[1]


def gmx_energies(xvg):
    """The first frame of a `gmx energy -dp` xvg: {term: value}."""
    names, vals = [], None
    for ln in open(xvg):
        if ln.startswith("@ s") and " legend " in ln:
            names.append(ln.split('"')[1])
        elif ln.strip() and ln[0] not in "#@" and vals is None:
            vals = [float(v) for v in ln.split()[1:]]
    return dict(zip(names, vals))


def gro_positions(path):
    lines = open(path).read().splitlines()
    n = int(lines[1])
    return np.array([[float(ln[20 + 8 * c:28 + 8 * c]) for c in range(3)] for ln in lines[2:2 + n]])


fails = []


def check(name, a, b, tol):
    ok = abs(a - b) <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {name:34s} GROMACS {a:16.8f}  OpenMM {b:16.8f}  diff {a - b:.2e}")
    if not ok:
        fails.append(name)


prmtop = os.path.join(wd, "qm7b_7203.prmtop")
for m in ("lj", "slater", "gauss"):
    eg = gmx_energies(os.path.join(wd, f"e_{m}.xvg"))
    x = gro_positions(os.path.join(wd, m + "_box.gro"))
    kw = dict(vdw="lj") if m == "lj" else dict(vdw="gvdw", gvdw=m)
    s, _, _ = pgm_prmtop.create_system(prmtop, periodic=False, rigid_water=False, **kw)
    groups = {}
    for k, f in enumerate(s.getForces()):
        f.setForceGroup(k)
        name = "GVDWForce" if openmmpgm.GVDWForce.isinstance(f) else type(f).__name__
        groups.setdefault(name, []).append(k)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(x)

    def e(names):
        mask = 0
        for nm in names:
            for k in groups.get(nm, []):
                mask |= 1 << k
        if not mask:
            return 0.0
        return ctx.getState(getEnergy=True, groups=mask).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    print(f"== {m}: OpenMM forces {sorted(groups)}")
    # ParmEd writes the bonded and LJ parameters with 7-8 digits: 1e-5 kJ/mol for those terms
    dih = sum(v for nm, v in eg.items() if "Dih" in nm)
    check(f"{m}: bonds", eg.get("Bond", 0.0), e(["HarmonicBondForce"]), 1e-5)
    check(f"{m}: angles", eg.get("Angle", 0.0), e(["HarmonicAngleForce"]), 1e-5)
    check(f"{m}: dihedrals", dih, e(["PeriodicTorsionForce"]), 1e-5)
    if m == "lj":
        check(f"{m}: LJ (SR) + LJ-14", eg["LJ (SR)"] + eg["LJ-14"], e(["NonbondedForce"]), 1e-5)
    else:   # GVDW, 1-4 pairs included on both sides: same coordinates and parameters, so to round-off
        check(f"{m}: GVDW = LJ (SR)", eg["LJ (SR)"], e(["GVDWForce"]), 1e-10 * abs(eg["LJ (SR)"]))
        check(f"{m}: no LJ-14", eg.get("LJ-14", 0.0), 0.0, 1e-12)
        check(f"{m}: no OpenMM Lennard-Jones", e(["NonbondedForce"]), 0.0, 1e-12)
    check(f"{m}: Coulomb-14 is zero", eg.get("Coulomb-14", 0.0), 0.0, 1e-12)
    del ctx
print(f"{len(fails)} failures" if fails else "ALL PASSED")
sys.exit(len(fails))
