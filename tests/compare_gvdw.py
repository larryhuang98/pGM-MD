"""Check GVDWForce: pmemd-pgm's VDWAALS, finite differences, platforms against Reference.

    python compare_gvdw.py system.prmtop box.txt [--cutoff 1.0] [--platforms Reference CPU OpenCL]

box.txt from tests/gvdw_input.py (positions, reduced box; nm).  The reference energies are pmemd-pgm's
VDWAALS of the GVDW manuscript's 512-water box (rayl_512_v2.prmtop + inputs/lj/inpcrd.restrt, 10 A,
vdwmeth = 0; pGM-JAX data/validation/validate_gvdw.json), with pmemd-pgm's global parameters
(pgm_prmtop.PGM3P_GVDW) on the atoms that carry LJ.
"""
import argparse
import sys

import numpy as np
import openmm as mm
from openmm import unit

import openmmpgm
import pgm_prmtop

PMEMD_KCAL = {"slater": 784.7282, "gauss": 868.3699}

ap = argparse.ArgumentParser()
ap.add_argument("prmtop")
ap.add_argument("box")
ap.add_argument("--cutoff", type=float, default=1.0)
ap.add_argument("--platforms", nargs="+", default=["Reference", "CPU", "OpenCL"])
ap.add_argument("--precision", default="mixed")
a = ap.parse_args()

lines = open(a.box).read().split("\n")
n = int(lines[0])
H = np.array([[float(v) for v in lines[1 + k].split()] for k in range(3)])
pos = np.array([[float(v) for v in lines[4 + i].split()[:3]] for i in range(n)])


def gvdw_system(rep, lrc=False):
    system, _, _ = pgm_prmtop.create_system(a.prmtop, cutoff=a.cutoff, vdw="gvdw", gvdw=rep, lj_lrc=lrc,
                                            ewald_tol=1e-4)
    for k in reversed(range(system.getNumForces())):
        if openmmpgm.PGMForce.isinstance(system.getForce(k)):
            system.removeForce(k)
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*v) for v in H])
    for k in range(system.getNumForces()):
        f = system.getForce(k)
        if openmmpgm.GVDWForce.isinstance(f):
            f.setForceGroup(1)
        else:
            f.setForceGroup(0)
    return system


def evaluate(system, platform, x=pos):
    props = {"Precision": a.precision} if platform in ("OpenCL", "CUDA") else {}
    integ = mm.VerletIntegrator(0.001)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName(platform), props)
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*v) for v in H])
    ctx.setPositions(x)
    st = ctx.getState(getEnergy=True, getForces=True, groups={1})
    e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f = st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    del ctx
    return e, np.asarray(f)


bad = 0
for rep in ("slater", "gauss"):
    system = gvdw_system(rep)
    e_ref, f_ref = evaluate(system, "Reference")
    print(f"== {rep}: Reference E = {e_ref / 4.184:.6f} kcal/mol, pmemd-pgm {PMEMD_KCAL[rep]:.4f}, "
          f"diff {e_ref / 4.184 - PMEMD_KCAL[rep]:.2e}")
    if abs(e_ref / 4.184 - PMEMD_KCAL[rep]) > 1e-3:
        bad += 1
    # finite differences on a few oxygens (central, h = 1e-5 nm)
    h, worst = 1e-5, 0.0
    for i in (0, 3, 300, 1533):
        for k in range(3):
            xp, xm = pos.copy(), pos.copy()
            xp[i, k] += h
            xm[i, k] -= h
            fd = -(evaluate(system, "Reference", xp)[0] - evaluate(system, "Reference", xm)[0]) / (2 * h)
            worst = max(worst, abs(fd - f_ref[i, k]))
    print(f"   finite differences: max |F_fd - F| = {worst:.2e} kJ/mol/nm (max |F| {np.abs(f_ref).max():.1f})")
    if worst > 1e-3:
        bad += 1
    for p in a.platforms:
        if p == "Reference":
            continue
        try:
            e, f = evaluate(system, p)
        except Exception as ex:  # platform not available
            print(f"   {p}: skipped ({str(ex).splitlines()[0][:80]})")
            continue
        df = f - f_ref
        print(f"   {p:9s} E rel diff {(e - e_ref) / abs(e_ref):.2e};  forces max diff {np.abs(df).max():.2e}, "
              f"rms diff / rms force {np.sqrt((df**2).mean() / (f_ref**2).mean()):.2e}")
        tol = 1e-9 if p == "CPU" else 1e-5
        if abs(e - e_ref) > tol * abs(e_ref) * 10 or np.sqrt((df**2).mean() / (f_ref**2).mean()) > tol * 10:
            bad += 1
    # dispersion correction
    e_lrc, _ = evaluate(gvdw_system(rep, lrc=True), "Reference")
    vol = H[0, 0] * H[1, 1] * H[2, 2]
    nlj = sum(1 for i in range(n) if lines[4 + i].split()[4] == "1")
    g = pgm_prmtop.PGM3P_GVDW[rep]
    sum_c = nlj * pgm_prmtop.gvdw_from_pmemd(g["arep"], g["c6"], g["b"])[1]
    expect = -2 * np.pi * sum_c**2 / (3 * vol * a.cutoff**3)
    print(f"   dispersion correction: {e_lrc - e_ref:.8f} kJ/mol, expected {expect:.8f} (V = {vol:.4f} nm^3)")
    if abs(e_lrc - e_ref - expect) > 1e-8 * abs(expect) + 1e-9:
        bad += 1
print("PASSED" if bad == 0 else f"FAILED ({bad})")
sys.exit(bad)
