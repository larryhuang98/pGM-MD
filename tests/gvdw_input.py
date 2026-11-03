"""Write the GVDW test input of core/tests/test_gvdw.cpp from a prmtop and an Amber restart.

    python gvdw_input.py system.prmtop restart out.txt

restart: NetCDF (.restrt / .ncrst) or ASCII rst7.  Output (nm): line 1 "n"; 3 lines of reduced box vectors
(OpenMM's convention); n lines "x y z radius has_lj" (has_lj: LJ epsilon > 0, pmemd-pgm's GVDW atoms);
then the excluded pairs "i j" (0-based) of the prmtop.
"""
import math
import re
import sys

import numpy as np



def read_sections(path):
    """Return the %FLAG sections of a prmtop: numbers as numpy arrays, strings as lists."""
    out, name, fmt, buf = {}, None, "", []

    def flush():
        if name is None:
            return
        m = re.match(r"\((\d*)([aAiIeEfF])(\d+)", fmt)
        w, kind = int(m.group(3)), m.group(2).lower()
        vals = [ln[k:k + w].strip() for ln in buf for k in range(0, len(ln), w) if ln[k:k + w].strip()]
        out[name] = vals if kind == "a" else np.array([float(v) for v in vals])

    for raw in open(path):
        line = raw.rstrip("\n")
        if line.startswith("%FLAG"):
            flush()
            name, fmt, buf = line.split()[1], "", []
        elif line.startswith("%FORMAT"):
            fmt = line[line.index("("):]
        elif not line.startswith("%"):
            buf.append(line)
    flush()
    return out


def read_restart(path):
    try:
        from scipy.io import netcdf_file

        f = netcdf_file(path, "r", mmap=False)
        xyz = np.array(f.variables["coordinates"][:], float)
        lengths = np.array(f.variables["cell_lengths"][:], float)
        angles = np.array(f.variables["cell_angles"][:], float)
        f.close()
        return xyz, lengths, angles
    except Exception:
        lines = open(path).read().splitlines()
        n = int(lines[1].split()[0])
        vals = [float(line[k:k + 12]) for line in lines[2:] for k in range(0, len(line.rstrip()), 12)]
        xyz = np.array(vals[:3 * n]).reshape(n, 3)
        box = vals[-6:]
        return xyz, np.array(box[:3]), np.array(box[3:])


def reduced_box(lengths, angles):
    """Box vectors as rows (OpenMM computePeriodicBoxVectors, then reduced form)."""
    a, b, c = lengths
    al, be, ga = np.radians(angles)
    A = np.array([a, 0, 0])
    B = np.array([b * math.cos(ga), b * math.sin(ga), 0])
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    C = np.array([cx, cy, math.sqrt(c * c - cx * cx - cy * cy)])
    C = C - B * round(C[1] / B[1])
    C = C - A * round(C[0] / A[0])
    B = B - A * round(B[0] / A[0])
    return np.array([A, B, C])


prmtop, rst, out = sys.argv[1:4]
s = read_sections(prmtop)
xyz, lengths, angles = read_restart(rst)
H = reduced_box(lengths, angles) * 0.1
x = xyz * 0.1
R = np.asarray(s["POL_GAUSS_RADII_LIST"], float) * 0.1
ntypes = int(s["POINTERS"][1])
types = s["ATOM_TYPE_INDEX"].astype(int) - 1
nbi = s["NONBONDED_PARM_INDEX"].astype(int)
B6 = np.asarray(s["LENNARD_JONES_BCOEF"], float)
has_lj = np.array([B6[nbi[t * ntypes + t] - 1] > 0 for t in types])
nexcl = s["NUMBER_EXCLUDED_ATOMS"].astype(int)
excl = s["EXCLUDED_ATOMS_LIST"].astype(int)
pairs, k = [], 0
for i, m in enumerate(nexcl):
    for j in excl[k:k + m]:
        if j > 0:
            pairs.append((i, j - 1))
    k += m
with open(out, "w") as f:
    f.write(f"{len(x)}\n")
    for v in H:
        f.write(" ".join(f"{c:.17g}" for c in v) + "\n")
    for i in range(len(x)):
        f.write(f"{x[i, 0]:.17g} {x[i, 1]:.17g} {x[i, 2]:.17g} {R[i]:.17g} {int(has_lj[i])}\n")
    for i, j in pairs:
        f.write(f"{i} {j}\n")
print(f"wrote {out}: {len(x)} atoms, {int(has_lj.sum())} with LJ, {len(pairs)} exclusions")
