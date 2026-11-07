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
  pgm-gvdw-rep-form in the mdp (1 Slater, 0 Gaussian); or with --gvdw-topology FORM B: A and C6 per atom
  type from the prmtop's ACOEF / BCOEF, as pmemd-pgm igvdw = 1 reads them.  The 1-4 pairs get the full
  GVDW, as in pmemd-pgm: nrexcl = 2 and no [ pairs ], so only 1-2 and 1-3 pairs are excluded.

Needs ParmEd.
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


# GROMACS's pGM uses ke = c_one4PiEps0 / epsilon-r.  pmemd-pgm's Coulomb constant, 332.05382 kcal/mol A/e^2, is
# the one shared by the pGM implementations: epsilon-r = c_one4PiEps0 / (332.05382 * 0.4184) reproduces it.
_C_ONE4PIEPS0 = 1.602176634e-19**2 * 6.02214076e23 / (4 * math.pi * 8.8541878128e-12) * 1e9 / 1e3
EPSILON_R_PMEMD = _C_ONE4PIEPS0 / (332.05382 * 4.184 / 10.0)


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
    per_atom = gvdw_from_topology(s) if gvdw is not None and gvdw[0] is None else None
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
            arep, c6, b = gvdw[:3]
            lines += ["", "[ pgm_gvdw ]", ";   atom funct  sqrtA(sqrt(kJ/mol))  sqrtC6(sqrt(kJ/mol nm^6))  b"]
            for k in range(n):
                if arep is None:            # per atom type from the prmtop's ACOEF/BCOEF (pmemd-pgm igvdw = 1)
                    sa, sc = per_atom[g0 + k]
                    if sa == 0.0 and sc == 0.0:
                        continue
                elif s.atoms[g0 + k].epsilon > 0:
                    sa, sc = math.sqrt(arep * 4.184), math.sqrt(c6 * 4.184 * 1e-6)
                else:
                    continue
                lines.append(f"{k + 1:7d} {1:5d} {sa:.16e} {sc:.16e} {b!r}")
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
        text = _gvdw_14(text)
        text = _zero_atomtype_epsilon(text)
    open(outbase + ".top", "w").write(text)
    print(f"Coulomb constant of pmemd-pgm (332.05382 kcal/mol A/e^2): set epsilon-r = {EPSILON_R_PMEMD!r} in the mdp")
    print(f"wrote {outbase}.top, {outbase}.gro: {len(s.atoms)} atoms, {len(instances)} molecule types, "
          f"{len(partners)} covalent dipoles")


def gvdw_from_topology(s):
    """Per-atom (sqrtA [kJ/mol]^1/2, sqrtC6 [kJ/mol nm^6]^1/2) from a GVDW prmtop, whose LENNARD_JONES_ACOEF /
    BCOEF hold the GVDW A [kcal/mol] and C6 [kcal/mol A^6] per type pair (pmemd-pgm igvdw = 1); the tables
    must follow the geometric rule."""
    d = s.parm_data
    ntypes = int(d["POINTERS"][1])
    nbi = [int(v) for v in d["NONBONDED_PARM_INDEX"]]
    A, C = d["LENNARD_JONES_ACOEF"], d["LENNARD_JONES_BCOEF"]

    def tab(t, a, b):
        idx = nbi[a * ntypes + b]
        if idx <= 0:
            raise ValueError("GVDW topology: 10-12 entries are not supported")
        return float(t[idx - 1])

    a = [math.sqrt(max(tab(A, t, t), 0.0)) for t in range(ntypes)]
    c = [math.sqrt(max(tab(C, t, t), 0.0)) for t in range(ntypes)]
    for x in range(ntypes):
        for y in range(ntypes):
            for t, v, name in ((A, a, "ACOEF"), (C, c, "BCOEF")):
                want, got = v[x] * v[y], tab(t, x, y)
                if abs(got - want) > 1e-9 * max(abs(got), abs(want), 1e-12):
                    raise ValueError(f"GVDW topology: {name} of types {x + 1},{y + 1} is not geometric")
    types = [int(v) - 1 for v in d["ATOM_TYPE_INDEX"]]
    return [(a[t] * math.sqrt(4.184), c[t] * math.sqrt(4.184e-6)) for t in types]


def _gvdw_14(text):
    """GVDW for the 1-4 pairs, as pmemd-pgm with igvdw = 1 (full GVDW for the nb14 pairs, no Lennard-Jones):
    nrexcl = 2 in every [ moleculetype ], so that only 1-2 and 1-3 pairs are excluded and the 1-4 pairs enter
    the nonbonded GVDW, and no [ pairs ] (they would add a 1-4 Lennard-Jones)."""
    if re.search(r"(?m)^\[ exclusions \]", text):
        raise NotImplementedError("explicit [ exclusions ] with GVDW 1-4 pairs")
    out = []
    for blk in re.split(r"(?m)^(?=\[ moleculetype \])", text):
        if not blk.startswith("[ moleculetype ]"):
            out.append(blk)
            continue
        lines, inside, seen = [], None, False
        for ln in blk.splitlines(keepends=True):
            t = ln.split(";")[0].strip()
            if t.startswith("["):
                inside = t.replace(" ", "")
                if inside == "[pairs]":
                    continue
            elif inside == "[pairs]":
                continue
            elif inside == "[moleculetype]" and t and not seen:
                f = t.split()
                ln = f"{f[0]:<17s} 2\n"
                seen = True
            lines.append(ln)
        out.append("".join(lines))
    return "".join(out)


def _explicit_pair_parameters(text):
    """Give every [ pairs ] line without parameters its 1-4 Lennard-Jones parameters explicitly.

    The pair parameters GROMACS would generate from the atom types (gen-pairs, fudgeLJ, comb-rule) or take
    from [ pairtypes ] are written on the line, so that zeroing the atom-type epsilon for GVDW leaves the
    1-4 Lennard-Jones untouched (pmemd-pgm with igvdw = 1 keeps the scaled 1-4 LJ; GVDW acts on the
    non-excluded pairs only).
    """
    sec = _sections(text)
    d = sec["defaults"][0].split()
    comb, genpairs, fudge = int(d[1]), d[2].lower() == "yes", float(d[3])
    types = {}
    for ln in sec.get("atomtypes", []):
        f = ln.split()
        types[f[0]] = (float(f[-2]), float(f[-1]))
    pairtypes = {}
    for ln in sec.get("pairtypes", []):
        f = ln.split()
        if int(f[2]) != 1:
            raise NotImplementedError(f"[ pairtypes ] funct {f[2]}")
        pairtypes[(f[0], f[1])] = pairtypes[(f[1], f[0])] = (float(f[3]), float(f[4]))

    def combine(ti, tj):
        if (ti, tj) in pairtypes:
            return pairtypes[(ti, tj)]
        if not genpairs:
            raise ValueError(f"no [ pairtypes ] entry for {ti} {tj} and gen-pairs = no")
        (si, ei), (sj, ej) = types[ti], types[tj]
        if comb == 2:                                   # sigma, epsilon; Lorentz-Berthelot
            return 0.5 * (si + sj), fudge * math.sqrt(ei * ej)
        if comb == 1:                                   # c6, c12: both geometric and scaled
            return fudge * math.sqrt(si * sj), fudge * math.sqrt(ei * ej)
        return math.sqrt(si * sj), fudge * math.sqrt(ei * ej)   # comb 3: sigma geometric

    out = []
    for blk in re.split(r"(?m)^(?=\[ moleculetype \])", text):
        if not blk.startswith("[ moleculetype ]"):
            out.append(blk)
            continue
        atype = {int(ln.split()[0]): ln.split()[1] for ln in _sections(blk).get("atoms", [])}
        lines, inside = [], False
        for ln in blk.splitlines(keepends=True):
            t = ln.split(";")[0].strip()
            if t.startswith("["):
                inside = t.replace(" ", "") == "[pairs]"
            elif inside and t and not t.startswith("#"):
                f = t.split()
                if int(f[2]) != 1:
                    raise NotImplementedError(f"[ pairs ] funct {f[2]}")
                if len(f) == 3:
                    p, q = combine(atype[int(f[0])], atype[int(f[1])])
                    ln = f"{int(f[0]):7d} {int(f[1]):6d} {1:5d} {p:16.10e} {q:16.10e}\n"
            lines.append(ln)
        out.append("".join(lines))
    return "".join(out)


def _zero_atomtype_epsilon(text):
    """Set the epsilon (last column) of every [ atomtypes ] and [ nonbond_params ] line to 0: GVDW
    replaces the Lennard-Jones of the non-excluded pairs ([ pairs ] keep theirs, see above)."""
    out, inside = [], False
    for ln in text.splitlines(keepends=True):
        t = ln.strip()
        if t.startswith("["):
            inside = t.replace(" ", "") in ("[atomtypes]", "[nonbond_params]")
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
    ap.add_argument("--gvdw-topology", nargs=2, metavar=("FORM", "B"),
                    help="GVDW with A and C6 per atom type from the prmtop's ACOEF/BCOEF (pmemd-pgm igvdw = 1); "
                         "FORM gauss|slater (gvdw_rep_form 0|1), B = b_rep_scale")
    a = ap.parse_args()
    g = None
    if a.gvdw_topology:
        form = {"gauss": 0, "slater": 1}[a.gvdw_topology[0]]
        g = (None, None, float(a.gvdw_topology[1]), form)
        print(f"GVDW from the topology: set pgm-gvdw-rep-form = {form} in the mdp")
    elif a.gvdw:
        g = PGM3P_GVDW[a.gvdw]
        if a.gvdw_params:
            g = (*a.gvdw_params, g[3])
        print(f"GVDW {a.gvdw}: set pgm-gvdw-rep-form = {g[3]} in the mdp")
    main(a.prmtop, a.rst7, a.out, g)
