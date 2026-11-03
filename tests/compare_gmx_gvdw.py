"""Compare the GVDW part of a GROMACS pGM dump with an independent NumPy sum and pmemd-pgm.

    python compare_gmx_gvdw.py DUMP box.txt slater|gauss [--rc 1.0]

box.txt (tests/gvdw_input.py) marks the GVDW atoms (LJ epsilon > 0, pmemd-pgm's rule); the parameters are
pmemd-pgm's global ones of the GVDW manuscript.  Checks energy (and against pmemd-pgm's VDWAALS), forces
and the virial Xi = -1/2 sum_pairs x_ij (x) F_ij.
"""
import argparse
import math

import numpy as np
from scipy.special import erf

MODELS = {"slater": (87500.0, 594.825035, 4.52, 1, 784.7282), "gauss": (422.0, 594.825035, 0.9453, 0, 868.3699)}
ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("box")
ap.add_argument("model", choices=MODELS)
ap.add_argument("--rc", type=float, default=1.0)
ap.add_argument("--override", nargs=5, type=float, metavar=("AI", "AJ", "A", "C6", "B"),
                help="a [ pgm_gvdw_pairs ] entry of every molecule (1-based atoms in a molecule of --natmol atoms)")
ap.add_argument("--natmol", type=int, default=3)
ap.add_argument("--tail", action="store_true", help="DispCorr = EnerPres was on")
a = ap.parse_args()
arep, c6, b, form, pmemd = MODELS[a.model]
A, C6 = arep * 4.184, c6 * 4.184e-6

lines = [ln for ln in open(a.dump) if not ln.startswith("#")]
n = int(lines[0].split()[0])
box = np.array([[float(v) for v in ln.split()] for ln in lines[2:5]])
dat = np.array([[float(v) for v in ln.split()] for ln in lines[8:8 + n]])
x, R = dat[:, 0:3], dat[:, 4]
ncov = int(lines[0].split()[1])
g = lines[8 + n + ncov:]
eg = [float(v) for v in g[0].split()]
vir_g = np.array([[float(v) for v in ln.split()] for ln in g[1:4]])
F_g = np.array([[float(v) for v in ln.split()] for ln in g[4:4 + n]])
active = np.array([ln.split()[4] == "1" for ln in open(a.box).read().split("\n")[4:4 + n]])

idx = np.where(active)[0]
i, j = np.triu_indices(len(idx), 1)
i, j = idx[i], idx[j]
d = x[i] - x[j]
for c in (2, 1, 0):
    d -= np.outer(np.round(d[:, c] / box[c, c]), box[c])
r = np.linalg.norm(d, axis=1)
m = r < a.rc
i, j, d, r = i[m], j[m], d[m], r[m]
beta = 1 / np.sqrt(2 * (R[i] ** 2 + R[j] ** 2))
y = beta * r
e = np.exp(-y * y)
Ay = erf(y) - 2 / math.sqrt(math.pi) * y * (1 + 2 * y * y / 3) * e
gy = 4 / (3 * math.sqrt(math.pi)) * y**3 * e
F = Ay * Ay + 0.5 * gy * gy
Ap = 8 / (3 * math.sqrt(math.pi)) * y**4 * e
gp = 4 / math.sqrt(math.pi) * y * y * (1 - 2 * y * y / 3) * e
Fp = 2 * Ay * Ap + gy * gp
erep = A * (np.exp(-b * y * y) if form == 0 else np.exp(-b * y))
derep = erep * (-2 * b * y * beta if form == 0 else -b * beta)          # dU_rep/dr
U = erep - C6 * F / r**6
dU = derep - C6 * (Fp * beta / r**6 - 6 * F / r**7)                        # dU/dr  (all y > 0.6 here)
assert y.min() > 0.6, y.min()
fi = -(dU / r)[:, None] * d                                               # force on i
Fn = np.zeros((n, 3))
np.add.at(Fn, i, fi)
np.add.at(Fn, j, -fi)
vir = -0.5 * np.einsum("pa,pb->ab", d, fi)
E = U.sum()
eo, Fo, viro = 0.0, np.zeros((n, 3)), np.zeros((3, 3))
if a.override:
    ai, aj, Ao, C6o, Bo = a.override
    oi = np.arange(0, n, a.natmol) + int(ai) - 1
    oj = np.arange(0, n, a.natmol) + int(aj) - 1
    do = x[oi] - x[oj]
    for c in (2, 1, 0):
        do -= np.outer(np.round(do[:, c] / box[c, c]), box[c])
    ro = np.linalg.norm(do, axis=1)
    bo = 1 / np.sqrt(2 * (R[oi] ** 2 + R[oj] ** 2))
    yo = bo * ro
    eeo = np.exp(-yo * yo)
    Ao_y = erf(yo) - 2 / math.sqrt(math.pi) * yo * (1 + 2 * yo * yo / 3) * eeo
    go = 4 / (3 * math.sqrt(math.pi)) * yo**3 * eeo
    Fy = Ao_y * Ao_y + 0.5 * go * go
    Apo = 8 / (3 * math.sqrt(math.pi)) * yo**4 * eeo
    gpo = 4 / math.sqrt(math.pi) * yo * yo * (1 - 2 * yo * yo / 3) * eeo
    Fpo = 2 * Ao_y * Apo + go * gpo
    ero = Ao * (np.exp(-Bo * yo * yo) if form == 0 else np.exp(-Bo * yo))
    dro = ero * (-2 * Bo * yo * bo if form == 0 else -Bo * bo)
    assert yo.min() > 0.6, yo.min()
    eo = (ero - C6o * Fy / ro**6).sum()
    dUo = dro - C6o * (Fpo * bo / ro**6 - 6 * Fy / ro**7)
    fo = -(dUo / ro)[:, None] * do
    np.add.at(Fo, oi, fo)
    np.add.at(Fo, oj, -fo)
    viro = -0.5 * np.einsum("pa,pb->ab", do, fo)
    print(f"overrides: gmx {eg[1]:.10f} numpy {eo:.10f} kJ/mol  rel {(eg[1] - eo) / eo:.2e}")
et = 0.0
if a.tail:
    V = abs(np.linalg.det(box))
    et = -2 * math.pi * (active.sum() * math.sqrt(C6)) ** 2 / (3 * V * a.rc**3)
    print(f"tail: gmx {eg[2]:.10f} formula {et:.10f} kJ/mol")
Fn = Fn + Fo
vir = vir + viro - 0.5 * et * np.eye(3) * (1 if a.tail else 0)
print(f"GVDW {a.model}: gmx {eg[0]:.8f} numpy {E:.8f} kJ/mol  rel {(eg[0] - E) / E:.2e};  "
      f"pmemd-pgm {pmemd * 4.184:.4f} (diff {eg[0] / 4.184 - pmemd:.2e} kcal/mol)")
print(f"forces max|diff| {np.abs(F_g - Fn).max():.3e} kJ/mol/nm (max |F| {np.abs(Fn).max():.1f})")
print(f"virial max|diff| {np.abs(vir_g - vir).max():.3e} kJ/mol (max |Xi| {np.abs(vir).max():.1f}); overrides {eg[1]}, tail {eg[2]}")
