#!/usr/bin/env python3
"""Write a GVDW test topology: every nonzero LENNARD_JONES_ACOEF/BCOEF entry is replaced by the
GVDW repulsion amplitude A and dispersion coefficient C6 (zero entries stay zero).

This is a consistency test only (CPU against GPU); it is not a physical GVDW model of the molecule.

usage: make_gvdw_prmtop.py in.prmtop out.prmtop A C6
"""
import sys

src, dst, A, C6 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
lines = open(src).read().split("\n")
out, i = [], 0
while i < len(lines):
    line = lines[i]
    out.append(line)
    if line.startswith("%FLAG LENNARD_JONES_ACOEF") or line.startswith("%FLAG LENNARD_JONES_BCOEF"):
        value = A if "ACOEF" in line else C6
        out.append(lines[i + 1])  # %FORMAT(5E16.8)
        i += 2
        vals = []
        while not lines[i].startswith("%"):
            vals += lines[i].split()
            i += 1
        new = ["%16.8E" % (value if float(v) != 0.0 else 0.0) for v in vals]
        for k in range(0, len(new), 5):
            out.append("".join(new[k:k + 5]))
        continue
    i += 1
open(dst, "w").write("\n".join(out))
