"""OpenMM pGM plugin against a pmemd.pgm single point (energy and forces) of the same prmtop and restart.

    python omm_vs_amber.py PRMTOP RST AMBER_OUT AMBER_FRC --cut A --beta 1/A --grid n --order k
                           [--platform CPU] [--precision double] [--no-lrc] [--vdw lj|gvdw]

Converged induction (tol 1e-10, no predictor), all bonds flexible, pmemd-pgm's Coulomb constant
(332.05382 kcal/mol A/e^2, the default of pgm_prmtop).  AMBER_FRC is the NetCDF force file of pmemd (ntwf=1).
"""
import argparse, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "openmm", "python"))
import openmm as mm, openmm.app as app
from openmm import unit
from scipy.io import netcdf_file
import pgm_prmtop

ap = argparse.ArgumentParser()
for k in ["prmtop", "rst", "out", "frc"]:
    ap.add_argument(k)
ap.add_argument("--cut", type=float, required=True); ap.add_argument("--beta", type=float, required=True)
ap.add_argument("--grid", type=int, required=True); ap.add_argument("--order", type=int, default=6)
ap.add_argument("--platform", default="CPU"); ap.add_argument("--precision", default="double")
ap.add_argument("--vdw", default="lj"); ap.add_argument("--no-lrc", action="store_true")
a = ap.parse_args()
KJ = 4.184
system, top, pgm = pgm_prmtop.create_system(a.prmtop, cutoff=a.cut / 10, ewald_beta=a.beta * 10, grid=(a.grid,) * 3,
    order=a.order, dipole_tol=1e-10, max_iter=1000, peek=0.0, predictor="none", rigid_water=False, constraints=None,
    vdw=a.vdw, lj_lrc=not a.no_lrc)
crd = app.AmberInpcrdFile(a.rst)
props = {"Precision": a.precision} if a.platform in ("CUDA", "OpenCL", "HIP") else {}
ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(a.platform), props)
ctx.setPeriodicBoxVectors(*crd.getBoxVectors()); ctx.setPositions(crd.getPositions())
st = ctx.getState(getEnergy=True, getForces=True)
E = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / KJ
F = st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer) / KJ / 10   # kcal/mol/A
txt = open(a.out).read(); blk = txt[txt.index("NSTEP =        0"):]
ep = float(re.search(r"EPtot\s+=\s*(\S+)", blk).group(1))
Fa = np.array(netcdf_file(a.frc, "r", mmap=False).variables["forces"][:], float)[0]
d = F - Fa; rms = np.sqrt((Fa ** 2).mean())
print(f"OpenMM {a.platform} {a.precision}: EPtot amber {ep:.4f} openmm {E:.4f} diff {E - ep:+.4f} rel {(E - ep) / abs(ep):.1e}; "
      f"forces rel rms {np.sqrt((d ** 2).mean()) / rms:.1e} max {np.abs(d).max():.1e} kcal/mol/A")
