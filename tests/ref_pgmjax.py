"""Reference values from pGM-JAX for a pmemd-pgm prmtop + rst7 (one configuration).

    python ref_pgmjax.py PRMTOP RST7 OUT.npz [--elec qpi] [--cutoff 0.9] [--beta 3.5] [--grid 36 36 36]

Runs pgm_jax.interfaces.engine.PGMEngine (double precision, tight induction, no predictor, no peek)
and saves positions, box, energy terms, forces, induced and permanent dipoles.  Run with the pgmjax
environment (JAX_PLATFORMS=cpu is fine).
"""

import argparse
import json

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from pgm_jax.interfaces.engine import PGMEngine
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.io import read_coordinates_nm

ap = argparse.ArgumentParser()
ap.add_argument("prmtop")
ap.add_argument("rst7")
ap.add_argument("out")
ap.add_argument("--elec", default="qpi")
ap.add_argument("--vdw", default="lj")
ap.add_argument("--cutoff", type=float, default=0.9)
ap.add_argument("--beta", type=float, default=3.5)
ap.add_argument("--grid", type=int, nargs=3, default=[36, 36, 36])
ap.add_argument("--order", type=int, default=6)
ap.add_argument("--tol", type=float, default=1e-10)
ap.add_argument("--no-lrc", action="store_true")
a = ap.parse_args()

st = MDSettings().replace(
    precision="double", elec=a.elec, vdw=a.vdw, cutoff=a.cutoff, ewald_beta=a.beta, pme_grid=tuple(a.grid),
    pme_order=a.order, dipole_tol=a.tol, max_iter=1000, predictor="none", fused=False, peek=0.0,
    lj_lrc=not a.no_lrc,
)  # fmt: skip
pos, _, H = read_coordinates_nm(a.rst7)
eng = PGMEngine.from_amber(a.prmtop, a.rst7, settings=st)
print(eng.describe())
res = eng.compute(pos, H)
print("energy", repr(res.energy), type(res.energy), np.asarray(res.forces).dtype)
out = {
    "pos": np.asarray(pos),
    "H": np.asarray(H),
    "energy": res.energy,
    "forces": np.asarray(res.forces),
    "iterations": res.iterations,
    "mu": np.asarray(res.induced_dipoles) if a.elec in ("qi", "qpi") else np.zeros_like(pos),
}
for k, v in res.terms.items():
    out["term_" + k] = float(v)
try:
    comp = np.asarray(res.dipole_components)
    out["dipole_components"] = comp
except Exception as err:  # noqa: BLE001
    print("no dipole components:", err)
np.savez(a.out, **out)
print(json.dumps({k: (float(v) if np.ndim(v) == 0 else list(np.shape(v))) for k, v in out.items()}, indent=1))
