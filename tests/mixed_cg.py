"""Induced-dipole solve in mixed precision against the double-precision Reference platform.

    python mixed_cg.py PRMTOP RST7 [--platform OpenCL] [--tols 1e-5 1e-6 1e-7 1e-8]

For each tolerance: CG iterations, max |mu - mu_ref| / max |mu_ref| and the energy difference, with
mu_ref the Reference solution at tolerance 1e-10.  Shows down to which tolerance the mixed-precision
solve still converges (the operator itself is applied in single precision).
"""
import argparse

import numpy as np
import openmm as mm
from openmm import app, unit

import openmmpgm
import pgm_prmtop

ap = argparse.ArgumentParser()
ap.add_argument("prmtop")
ap.add_argument("rst7")
ap.add_argument("--platform", default="OpenCL")
ap.add_argument("--precision", default="mixed")
ap.add_argument("--tols", nargs="+", type=float, default=[1e-5, 1e-6, 1e-7, 1e-8])
ap.add_argument("--by-site", type=int, default=0, help="also by site of an N-site molecule")
ap.add_argument("--nocutoff", action="store_true", help="no PME (separates the direct-space error)")
a = ap.parse_args()
crd = app.AmberInpcrdFile(a.rst7)
pos = crd.positions


def run(platform, tol, prec=None):
    kw = dict(periodic=False) if a.nocutoff else dict(cutoff=0.9, ewald_beta=3.5, grid=(36, 36, 36))
    s, _, pgm = pgm_prmtop.create_system(a.prmtop, lj=False, dipole_tol=tol, predictor="none", max_iter=200, **kw)
    s.setDefaultPeriodicBoxVectors(*crd.boxVectors)
    pgm = [openmmpgm.PGMForce.cast(s.getForce(k)) for k in range(s.getNumForces())
           if openmmpgm.PGMForce.isinstance(s.getForce(k))][0]
    props = {"Precision": prec} if prec else {}
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(platform), props)
    ctx.setPositions(pos)
    e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    mu = np.asarray(pgm.getInducedDipoles(ctx))
    it = pgm.getInducedDipoleIterations(ctx)
    del ctx
    return e, mu, it


e0, mu0, it0 = run("Reference", 1e-10)
print(f"Reference tol 1e-10: {it0} iterations, E {e0:.8f} kJ/mol")
print(f"{'tol':>8s} {'iter':>5s} {'max|dmu|/max|mu|':>17s} {'rms|dmu|/rms|mu|':>17s} {'dE kJ/mol':>12s}")
for tol in a.tols:
    e, mu, it = run(a.platform, tol, a.precision)
    d = mu - mu0
    print(f"{tol:8.0e} {it:5d} {np.abs(d).max() / np.abs(mu0).max():17.3e} "
          f"{np.sqrt((d**2).mean() / (mu0**2).mean()):17.3e} {e - e0:12.3e}")
    if a.by_site:
        for k in range(a.by_site):
            dk, mk = d[k::a.by_site], mu0[k::a.by_site]
            print(f"         site {k}: rms|dmu| {np.sqrt((dk**2).mean()):.3e}  rms|mu| {np.sqrt((mk**2).mean()):.3e}")
