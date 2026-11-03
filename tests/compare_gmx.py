"""Compare the GROMACS pGM dump (GMX_PGM_DUMP, first force call) with a pGM-JAX reference (ref_pgmjax.py).

    python compare_gmx.py DUMP REF.npz [--ke 138.935458]

The reference must be electrostatics only (--no-vdw reference: its forces are the pGM forces).
Energies and forces are rescaled from the GROMACS Coulomb constant (read from the dump's log line or
--ke-gmx) to the reference's --ke.  Dipoles do not depend on the Coulomb constant.
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("ref")
ap.add_argument("--ke", type=float, default=138.935458, help="Coulomb constant of the reference")
ap.add_argument("--ke-gmx", type=float, default=None, help="GROMACS Coulomb constant (default: c_one4PiEps0)")
a = ap.parse_args()

if a.ke_gmx is None:
    # GROMACS units.h (2026): c_one4PiEps0 = 1/(4 pi eps0) with eps0 from the 2019 SI constants
    c_e, c_na, c_eps0_si = 1.602176634e-19, 6.02214076e23, 8.8541878128e-12
    a.ke_gmx = c_e**2 * c_na / (4 * np.pi * c_eps0_si) * 1e9 / 1e3
s = a.ke / a.ke_gmx

lines = [ln for ln in open(a.dump) if not ln.startswith("#")]
n = int(lines[0].split()[0])
t = lines[1].split()
terms = np.array([float(v) for v in t[:5]]) * s
it, res = int(t[5]), float(t[6])
vir = np.array([[float(v) for v in ln.split()] for ln in lines[5:8]]) * s
dat = np.array([[float(v) for v in ln.split()] for ln in lines[8:8 + n]])
F, mu, perm = dat[:, 6:9] * s, dat[:, 9:12], dat[:, 12:15]

r = np.load(a.ref)
E = terms.sum()
print(f"Coulomb constants: reference {a.ke}, GROMACS {a.ke_gmx:.12g} (scale {s:.15f})")
print("terms (direct recip self bg pol):", " ".join(f"{v:.10f}" for v in terms))
print(f"elec  gmx {E:.10f}  ref {float(r['term_elec']):.10f}  diff {E - float(r['term_elec']):.3e}  "
      f"rel {(E - float(r['term_elec'])) / abs(float(r['term_elec'])):.3e}")
dF = F - r["forces"]
print(f"forces max|diff| {np.abs(dF).max():.3e}  rms diff {np.sqrt((dF**2).mean()):.3e}  "
      f"rms ref {np.sqrt((r['forces']**2).mean()):.3e}")
dm = mu - r["mu"]
print(f"mu max|diff| {np.abs(dm).max():.3e} e nm  (max|mu| {np.abs(r['mu']).max():.3e});  "
      f"CG iterations gmx {it} ref {int(r['iterations'])}  residual {res:.2e}")
print("virial (Xi) [kJ/mol]:\n", np.array2string(vir, precision=6))
print(f"virial asymmetry max {np.abs(vir - vir.T).max():.3e}")
