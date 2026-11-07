"""Two pmemd.pgm single points (e.g. CPU and CUDA): potential energy and forces.
    python amber_vs_amber.py REF.out REF.frc TEST.out TEST.frc [label]"""
import re, sys
import numpy as np
from scipy.io import netcdf_file


def ep(f):
    t = open(f).read(); b = t[t.index("NSTEP =        0"):]
    return float(re.search(r"EPtot\s+=\s*(\S+)", b).group(1))


def frc(f):
    return np.array(netcdf_file(f, "r", mmap=False).variables["forces"][:], float)[0]


e0, e1, f0, f1 = ep(sys.argv[1]), ep(sys.argv[3]), frc(sys.argv[2]), frc(sys.argv[4])
d = f1 - f0; r = np.sqrt((f0 ** 2).mean())
lab = sys.argv[5] if len(sys.argv) > 5 else "AMBER"
print(f"{lab}: EPtot ref {e0:.4f} test {e1:.4f} diff {e1 - e0:+.4f} rel {(e1 - e0) / abs(e0):.1e}; "
      f"forces rel rms {np.sqrt((d ** 2).mean()) / r:.1e} max {np.abs(d).max():.1e} kcal/mol/A")
