"""Convert a pmemd-pgm prmtop (+ rst7) to a GROMACS topology and coordinates for coulombtype = pGM.

    python prmtop_to_gmx.py system.prmtop system.rst7 out      # writes out.top, out.gro

ParmEd writes the standard parts (atoms, LJ types, bonded terms, exclusions via nrexcl, 1-4 pairs,
settles for rigid water).  Then, for every [ moleculetype ], this script:

* sets the atom charges to the pGM Gaussian charges (POL_GAUSS_MONOPOLES_LIST);
* adds [ pgm_atoms ] (radius nm, polarizability nm^3) and [ pgm_dipoles ] (strength e nm), in the
  molecule's own atom numbering, from its first instance in the prmtop;
* sets fudgeQQ = 0 in [ defaults ]: 1-4 pairs carry LJ only, since the pGM sum over all pairs already
  contains the 1-4 electrostatics (pmemd-pgm drops the 1-4 electrostatics too).

Needs ParmEd (e.g. the SE3nv environment on rayl8).
"""

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


def main(prmtop, rst7, outbase):
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
    open(outbase + ".top", "w").write("".join(out))
    print(f"wrote {outbase}.top, {outbase}.gro: {len(s.atoms)} atoms, {len(instances)} molecule types, "
          f"{len(partners)} covalent dipoles")


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
    main(*sys.argv[1:4])
