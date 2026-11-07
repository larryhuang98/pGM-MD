"""Finite-difference check of the GROMACS pGM path: E(x0 - h d) - E(x0 + h d) = 2 h F(x0).d for d = F/|F| and a
random unit direction (all atoms move), single points with tol 1e-10.
    python fd.py GMX MODE H        (MODE cpu | gpu)"""
import os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
exec(open(os.path.join(HERE, "dd_compare.py")).read().split("ref = sys.argv[1]")[0].split("args = sys.argv[1:]")[0])
G, mode, h = sys.argv[1], sys.argv[2], float(sys.argv[3])
R = os.path.join(HERE, "..", "data", "gmx_w4096")
env = dict(os.environ, GMX_PGM_QUIET="1")
if mode == "gpu": env["GMX_PGM_GPU"] = "1"
else: env.pop("GMX_PGM_GPU", None)
lines = open(R + "/w4096.gro").read().split("\n")
n = int(lines[1]); atoms = lines[2:2+n]; boxv = [float(v) for v in lines[2+n].split()]
x0 = np.array([[float(a[20:28]), float(a[28:36]), float(a[36:44])] for a in atoms])
def write_g96(fn, x):
    with open(fn, "w") as f:
        f.write("TITLE\nfd\nEND\nPOSITION\n")
        for i, a in enumerate(atoms):
            f.write("%5d %-5s %-5s%7d%15.9f%15.9f%15.9f\n" % (int(a[0:5]), a[5:10].strip(), a[10:15].strip(), i+1, *x[i]))
        f.write("END\nBOX\n")
        b = boxv + [0.0]*(9-len(boxv))
        f.write("".join("%15.9f" % v for v in b) + "\nEND\n")
mdp = open(R + "/bench.mdp").read().split("\n")
keep = [l for l in mdp if not l.split("=")[0].strip() in ("vdw-modifier","nsteps","pgm-dipole-tol","gen-vel","continuation","nstcalcenergy","nstenergy","nstfout")]
open("sp.mdp","w").write("\n".join(keep) + "\nnsteps = 0\npgm-dipole-tol = 1e-10\npgm-max-iter = 500\ngen-vel = no\ncontinuation = yes\nnstcalcenergy = 1\nnstenergy = 1\nnstfout = 1\nvdw-modifier = Potential-shift\n")
def sp(tag, x):
    write_g96(tag + ".g96", x)
    subprocess.run([G, "grompp", "-f", "sp.mdp", "-c", tag + ".g96", "-p", R + "/w4096.top", "-o", tag + ".tpr", "-maxwarn", "3"], env=env, capture_output=True, check=True)
    subprocess.run([G, "mdrun", "-s", tag + ".tpr", "-deffnm", tag, "-nb", "cpu", "-ntmpi", "1", "-ntomp", "8"], env=env, capture_output=True, check=True)
    out = subprocess.run([G, "energy", "-f", tag + ".edr", "-dp", "-o", tag + "_e.xvg"], input="Potential\n\n", env=env, capture_output=True, text=True).stdout
    e = [float(l.split()[1]) for l in open(tag + "_e.xvg") if l.strip() and l[0] not in "#@"][0]
    return e
e0 = sp("x0", x0)
_, _, F = read_trr_first("x0.trr")
rng = np.random.default_rng(7)
for name, d in (("force direction", F), ("random direction", rng.normal(size=x0.shape))):
    d = d / np.linalg.norm(d)
    ep, em = sp("xp", x0 + h*d), sp("xm", x0 - h*d)
    fd = (em - ep) / (2*h); an = float(np.sum(F*d))
    print(f"{mode} {name}: E0 {e0:.6f}  (E- - E+)/2h {fd:.6f}  F.d {an:.6f}  rel diff {(fd-an)/abs(an):+.2e}  (|F| {np.linalg.norm(F):.1f}, h {h})", flush=True)
