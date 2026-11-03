"""Convert a pmemd-pgm prmtop (+ rst7) to a GROMACS topology and coordinates for coulombtype = pGM.

    python prmtop_to_gmx.py system.prmtop system.rst7 out [--gvdw slater|gauss]   # writes out.top, out.gro

ParmEd writes the standard parts (atoms, LJ types, bonded terms, exclusions via nrexcl, 1-4 pairs,
settles for rigid water).  Then, for every [ moleculetype ], this script:

* sets the atom charges to the pGM Gaussian charges (POL_GAUSS_MONOPOLES_LIST);
* adds [ pgm_atoms ] (radius nm, polarizability nm^3) and [ pgm_dipoles ] (strength e nm), in the
  molecule's own atom numbering, from its first instance in the prmtop;
* sets fudgeQQ = 0 in [ defaults ]: 1-4 pairs carry LJ only, since the pGM sum over all pairs already
  contains the 1-4 electrostatics (pmemd-pgm drops the 1-4 electrostatics too);
* with --gvdw: replaces the Lennard-Jones by GVDW (pmemd-pgm igvdw = 1): pmemd's global gvdw_arep,
  gvdw_c6, b_rep_scale (the GVDW manuscript's water models, or --gvdw-params A_kcal C6_kcalA6 b) as
  [ pgm_gvdw ] on every atom with an LJ epsilon > 0, every atom type's epsilon set to 0; set
  pgm-gvdw-rep-form in the mdp (1 Slater, 0 Gaussian).  Molecules with 1-4 pairs are not converted yet
  (their LJ would come from the zeroed atom types).

Needs ParmEd (e.g. the SE3nv environment on rayl8).
"""

import math
import re
import sys

import numpy as np
import parmed


def pol_sections(path):
    """Return the POL_GAUSS_* lists of a prmtop (numbers)."""
    out, name, fmt, buf = {}, None, None, []

    def flush():
        if name and name.startswith("POL_GAUSS"):
            w = int(re.match(r"\((\d*)[aAiIeEfF](\d+)", fmt).group(2))
            vals = [ln[k:k + w] for ln in buf for k in range(0, len(ln), w) if ln[k:k + w].strip()]
            out[name] = np.array([float(v) for v in vals])

    for raw in open(path):
        line = raw.rstrip("\n")
        if line.startswith("%FLAG"):
            flush()
            name, fmt, buf = line.split()[1], None, []
        elif line.startswith("%FORMAT"):
            fmt = line[line.index("("):]
        elif line.startswith("%COMMENT") or line.startswith("%VERSION"):
            continue
        elif name:
            buf.append(line)
    flush()
    return out


# pmemd-pgm GVDW water models of the GVDW manuscript (O-O only): gvdw_arep [kcal/mol], gvdw_c6
# [kcal/mol A^6], b_rep_scale, gvdw_rep_form
PGM3P_GVDW = {"slater": (87500.0, 594.825035, 4.52, 1), "gauss": (422.0, 594.825035, 0.9453, 0)}


def main(prmtop, rst7, outbase, gvdw=None):
    s = parmed.load_file(prmtop, xyz=rst7)
    pol = pol_sections(prmtop)
    q = pol["POL_GAUSS_MONOPOLES_LIST"]
    radius = pol["POL_GAUSS_RADII_LIST"] * 0.1                 # A -> nm
    alpha = pol["POL_GAUSS_POLARIZABILITY_LIST"] * 1e-3        # A^3 -> nm^3
    nptr = pol["POL_GAUSS_COVALENT_POINTERS_LIST"].astype(int)
    partners = pol["POL_GAUSS_COVALENT_ATOMS_LIST"].astype(int) - 1
    strength = pol["POL_GAUSS_COVALENT_DIPOLES_LIST"] * 0.1     # e A -> e nm
    start = np.concatenate([[0], np.cumsum(nptr)])
    for i, a in enumerate(s.atoms):
        a.charge = float(q[i])
    gmx = parmed.gromacs.GromacsTopologyFile.from_structure(s)
    gmx.defaults.fudgeQQ = 0.0
    gmx.write(outbase + ".top")
    s.save(outbase + ".gro", overwrite=True)

    text = open(outbase + ".top").read()
    blocks = re.split(r"(?m)^(?=\[ moleculetype \])", text)
    out = [blocks[0]]
    instances = _instance_offsets(text)
    for blk in blocks[1:]:
        name = re.search(r"\[ moleculetype \][^\n]*\n(?:\s*;[^\n]*\n)*\s*(\S+)", blk).group(1)
        if name not in instances:
            out.append(blk)
            continue
        g0, n = instances[name]
        lines = ["", "[ pgm_atoms ]", ";   atom funct   radius(nm)       alpha(nm^3)"]
        for k in range(n):
            lines.append(f"{k + 1:7d} {1:5d} {radius[g0 + k]:16.8e} {alpha[g0 + k]:16.8e}")
        if gvdw is not None:
            if "[ pairs ]" in blk:
                raise NotImplementedError(f"molecule {name}: 1-4 pairs with GVDW are not converted yet")
            arep, c6, b = gvdw[:3]
            sa, sc = math.sqrt(arep * 4.184), math.sqrt(c6 * 4.184 * 1e-6)
            lines += ["", "[ pgm_gvdw ]", ";   atom funct  sqrtA(sqrt(kJ/mol))  sqrtC6(sqrt(kJ/mol nm^6))  b"]
            for k in range(n):
                if s.atoms[g0 + k].epsilon > 0:
                    lines.append(f"{k + 1:7d} {1:5d} {sa:16.10e} {sc:16.10e} {b:12.6f}")
        lines += ["", "[ pgm_dipoles ]", ";     ai     aj funct  strength(e nm)  p_ai += c unit(r_aj - r_ai)"]
        for k in range(n):
            for m in range(start[g0 + k], start[g0 + k + 1]):
                j = partners[m] - g0
                if not 0 <= j < n:
                    raise ValueError(f"covalent dipole {g0 + k}->{partners[m]} leaves molecule {name}")
                lines.append(f"{k + 1:8d} {j + 1:6d} {1:5d} {strength[m]:16.8e}")
        tail = re.search(r"(?m)^\[ (system|molecules) \]", blk)
        ins = "\n".join(lines) + "\n\n"
        out.append(blk[:tail.start()] + ins + blk[tail.start():] if tail else blk.rstrip("\n") + "\n" + ins)
    text = "".join(out)
    if gvdw is not None:
        text = _zero_atomtype_epsilon(text)
    open(outbase + ".top", "w").write(text)
    print(f"wrote {outbase}.top, {outbase}.gro: {len(s.atoms)} atoms, {len(instances)} molecule types, "
          f"{len(partners)} covalent dipoles")


def _zero_atomtype_epsilon(text):
    """Set the epsilon (last column) of every [ atomtypes ] line to 0: GVDW replaces the Lennard-Jones."""
    out, inside = [], False
    for ln in text.splitlines(keepends=True):
        t = ln.strip()
        if t.startswith("["):
            inside = t.replace(" ", "") == "[atomtypes]"
        elif inside and t and not t.startswith(";") and not t.startswith("#"):
            f = ln.split(";")[0].split()
            f[-1] = "0"
            ln = " ".join(f) + "\n"
        out.append(ln)
    return "".join(out)


def _sections(block):
    """Return {section name: [data lines]} of a topology text block (comments and blanks dropped)."""
    out, cur = {}, None
    for ln in block.splitlines():
        t = ln.split(";")[0].strip()
        m = re.match(r"\[\s*(\w+)\s*\]", t)
        if m:
            cur = m.group(1)
            out.setdefault(cur, [])
        elif t and cur and not t.startswith("#"):
            out[cur].append(t)
    return out


def _instance_offsets(text):
    """Return {molecule type: (global index of its first atom, atoms per molecule)} from a .top text."""
    natoms = {}
    for blk in re.split(r"(?m)^(?=\[ moleculetype \])", text)[1:]:
        sec = _sections(blk)
        natoms[sec["moleculetype"][0].split()[0]] = len(sec.get("atoms", []))
    res, g = {}, 0
    for ln in _sections(text)["molecules"]:
        name, count = ln.split()[0], int(ln.split()[1])
        res.setdefault(name, (g, natoms[name]))
        g += natoms[name] * count
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prmtop")
    ap.add_argument("rst7")
    ap.add_argument("out")
    ap.add_argument("--gvdw", choices=sorted(PGM3P_GVDW), help="replace LJ by GVDW with this model")
    ap.add_argument("--gvdw-params", nargs=3, type=float, metavar=("AREP_KCAL", "C6_KCAL_A6", "B"),
                    help="pmemd-pgm gvdw_arep, gvdw_c6, b_rep_scale (with --gvdw for the form)")
    a = ap.parse_args()
    g = None
    if a.gvdw:
        g = PGM3P_GVDW[a.gvdw]
        if a.gvdw_params:
            g = (*a.gvdw_params, g[3])
        print(f"GVDW {a.gvdw}: set pgm-gvdw-rep-form = {g[3]} in the mdp")
    main(a.prmtop, a.rst7, a.out, g)
