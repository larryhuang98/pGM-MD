"""GROMACS single point (run by gmx_sp.sh) against pmemd.pgm: potential energy (gmx energy -dp) and forces (trr).

    python gmx_vs_amber.py GMX DEFFNM AMBER_PREFIX       (in the GROMACS work directory; AMBER_PREFIX.out/.frc)
"""
import os, re, subprocess, sys
import numpy as np
from scipy.io import netcdf_file

G, deffnm, ref = sys.argv[1:4]
subprocess.run(f"printf 'Potential\\n\\n' | {G} energy -f {deffnm}.edr -dp -o pot.xvg", shell=True, capture_output=True)
pot = float([l for l in open("pot.xvg") if l[0] not in "#@"][0].split()[1]) / 4.184
t = subprocess.run(f"{G} dump -f {deffnm}.trr", shell=True, capture_output=True, text=True).stdout
fb = t[t.index("f ("):]
F = np.array([[float(v) for v in re.findall(r"[-+]?\d\.\d+e[-+]\d+", ln)] for ln in fb.splitlines()[1:] if "f[" in ln]) / 4.184 / 10
txt = open(ref + ".out").read(); b = txt[txt.index("NSTEP =        0"):]
ep = float(re.search(r"EPtot\s+=\s*(\S+)", b).group(1))
Fa = np.array(netcdf_file(ref + ".frc", "r", mmap=False).variables["forces"][:], float)[0]
d = F - Fa; rms = np.sqrt((Fa ** 2).mean())
print(f"GROMACS {os.path.basename(G)}: EPtot amber {ep:.4f} gmx {pot:.4f} diff {pot - ep:+.4f} rel {(pot - ep) / abs(ep):.1e}; "
      f"forces rel rms {np.sqrt((d ** 2).mean()) / rms:.1e} max {np.abs(d).max():.1e} kcal/mol/A")
