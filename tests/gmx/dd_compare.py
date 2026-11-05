"""Compare GROMACS single points run on several PP ranks (domain decomposition) with the one-rank run.

    python dd_compare.py [--gpu] REF_PREFIX PREFIX [PREFIX ...]

Each PREFIX has PREFIX.trr (first frame, double precision, with forces: nstfout = 1) and PREFIX_e.xvg
(`gmx energy -dp`).  Prints the largest energy-term difference and the largest force difference against
REF_PREFIX; exits with the number of runs beyond 1e-9 relative (energies) or 1e-6 kJ/mol/nm (forces).
--gpu (the GPU engine against the CPU path, both single precision): energy terms 5e-6 relative (to max(|E|,
1)), the pressure and virial tensors (and the other pressure-like terms) 1 bar / 2 kJ/mol absolute (they
are small differences of terms of 1e4 bar / kJ/mol: the off-diagonals are near zero), forces 5e-5 of the
largest force.
"""
import struct
import sys

import numpy as np


def read_trr_first(path):
    """box (3x3), x, f of the first frame of a .trr (XDR, single or double precision)."""
    b = open(path, "rb").read()
    o = 0

    def ints(n):
        nonlocal o
        v = struct.unpack(">" + "i" * n, b[o:o + 4 * n])
        o += 4 * n
        return v

    magic, slen, l = ints(3)
    assert magic == 1993, "not a trr file"
    o += (l + 3) // 4 * 4
    (ir, e, box, vir, pres, top, sym, xs, vs, fs, natoms, step, nre) = ints(13)
    w = box // 9 if box else (xs or fs) // (3 * natoms)
    fmt = "d" if w == 8 else "f"
    o += 2 * w  # t, lambda

    def reals(nbytes):
        nonlocal o
        n = nbytes // w
        v = np.array(struct.unpack(">" + fmt * n, b[o:o + nbytes]))
        o += nbytes
        return v

    B = reals(box).reshape(3, 3) if box else None
    reals(vir)
    reals(pres)
    X = reals(xs).reshape(natoms, 3) if xs else None
    reals(vs)
    F = reals(fs).reshape(natoms, 3) if fs else None
    return B, X, F


def energies(xvg):
    names, vals = [], None
    for ln in open(xvg):
        if ln.startswith("@ s") and " legend " in ln:
            names.append(ln.split('"')[1])
        elif ln.strip() and ln[0] not in "#@" and vals is None:
            vals = [float(v) for v in ln.split()[1:]]
    return dict(zip(names, vals))


args = sys.argv[1:]
gpu = "--gpu" in args
args = [x for x in args if x != "--gpu"]


def pressure_like(k):
    return k.startswith("Pres") or k == "Pressure" or "SurfTen" in k


def scaled(k, d, e0):
    """the difference d over its tolerance (ok: <= 1)"""
    if not gpu:
        return d / max(abs(e0), 1.0) / 1e-9
    if pressure_like(k):
        return d / 1.0
    if k.startswith("Vir"):
        return d / 2.0
    return d / max(abs(e0), 1.0) / 5e-6


ref = args[0]
_, _, F0 = read_trr_first(ref + ".trr")
E0 = energies(ref + "_e.xvg")
bad = 0
for p in args[1:]:
    _, _, F = read_trr_first(p + ".trr")
    E = energies(p + "_e.xvg")
    worst = max((scaled(k, abs(E[k] - E0[k]), E0[k]), k) for k in E0 if k in E)
    df = np.abs(F - F0).max()
    ftol = 5e-5 * np.abs(F0).max() if gpu else 1e-6
    ok = worst[0] <= 1 and df < ftol
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {p}: max energy-term diff / tolerance {worst[0]:.2e} ({worst[1]}), "
          f"max force diff {df:.2e} kJ/mol/nm (tolerance {ftol:.1e}, max |F| {np.abs(F0).max():.0f}); "
          f"potential {E.get('Potential', float('nan')):.10f} vs {E0.get('Potential', float('nan')):.10f}")
sys.exit(bad)
