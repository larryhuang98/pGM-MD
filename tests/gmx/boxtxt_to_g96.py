"""Write a GROMACS .g96 from a box text file of tests/gvdw_input.py (nm, reduced box) and a .gro template.

    python boxtxt_to_g96.py box.txt template.gro out.g96
"""
import sys

box_txt, gro, out = sys.argv[1:4]
lines = open(box_txt).read().split("\n")
n = int(lines[0])
H = [[float(v) for v in lines[1 + k].split()] for k in range(3)]
x = [[float(v) for v in lines[4 + i].split()[:3]] for i in range(n)]
g = open(gro).read().splitlines()
assert int(g[1]) == n, (g[1], n)
with open(out, "w") as f:
    f.write("TITLE\nGVDW test box\nEND\nPOSITION\n")
    for i in range(n):
        ln = g[2 + i]
        f.write(f"{int(ln[0:5]):5d} {ln[5:10].strip():5s} {ln[10:15].strip():5s}{i + 1:7d}"
                f"{x[i][0]:15.9f}{x[i][1]:15.9f}{x[i][2]:15.9f}\n")
    f.write("END\nBOX\n")
    b = [H[0][0], H[1][1], H[2][2], H[0][1], H[0][2], H[1][0], H[1][2], H[2][0], H[2][1]]
    f.write("".join(f"{v:15.9f}" for v in b) + "\nEND\n")
print(f"wrote {out}")
