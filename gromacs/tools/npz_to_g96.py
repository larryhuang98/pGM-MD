"""Write the positions and box of a reference .npz (ref_pgmjax.py output) as GROMACS .g96, full precision.

    python npz_to_g96.py REF.npz template.gro out.g96

Atom and residue names come from the .gro (same atom order).  Box: the triclinic g96 BOX line
(v1x v2y v3z v1y v1z v2x v2z v3x v3y).
"""
import sys

import numpy as np

ref, gro, out = sys.argv[1:4]
d = np.load(ref)
x, H = d["pos"], d["H"]
lines = open(gro).read().splitlines()
n = int(lines[1])
assert n == len(x), (n, len(x))
with open(out, "w") as f:
    f.write("TITLE\npGM reference configuration\nEND\nPOSITION\n")
    for i in range(n):
        ln = lines[2 + i]
        resnr, resname, atom = int(ln[0:5]), ln[5:10].strip(), ln[10:15].strip()
        f.write(f"{resnr:5d} {resname:5s} {atom:5s}{i + 1:7d}{x[i, 0]:15.9f}{x[i, 1]:15.9f}{x[i, 2]:15.9f}\n")
    f.write("END\nBOX\n")
    b = [H[0, 0], H[1, 1], H[2, 2], H[0, 1], H[0, 2], H[1, 0], H[1, 2], H[2, 0], H[2, 1]]
    f.write("".join(f"{v:15.9f}" for v in b) + "\nEND\n")
print(f"wrote {out}: {n} atoms")
