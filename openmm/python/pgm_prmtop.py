"""Build an OpenMM System for a pmemd-pgm prmtop: PGMForce + Lennard-Jones (+ Amber's tail correction).

    import pgm_prmtop
    system, top, pgm = pgm_prmtop.create_system("water.prmtop", cutoff=0.9, ewald_beta=3.5, grid=(36, 36, 36))

What it builds (docs/spec.md):

* bonded terms, masses, constraints and the topology from openmm.app.AmberPrmtopFile;
* PGMForce from the POL_GAUSS_* sections (Gaussian charges, radii, polarizabilities, covalent dipoles;
  Amber units converted: A -> nm, A^3 -> nm^3, e A -> e nm).  All pairs interact: no exclusions;
* Lennard-Jones: the NonbondedForce of AmberPrmtopFile with every charge set to zero (it keeps the LJ
  tables, the exclusions and the 1-4 LJ pairs of the prmtop), cut at the cutoff without a switch and
  without OpenMM's dispersion correction;
* the LJ long-range correction of Amber's vdwmeth = 1 (pGM-JAX lj_lrc): -2 pi/(3 V rc^3) sum_ij B_ij over
  all ordered atom pairs, as a TailCorrectionForce (only the r^-6 term, unlike OpenMM's correction);
* or, with vdw="gvdw", GVDWForce (the Gaussian-density van der Waals of pmemd-pgm igvdw = 1) in place
  of the Lennard-Jones: the pGM radii, GVDW parameters from `gvdw` (pmemd-pgm's global
  gvdw_arep / gvdw_c6 / b_rep_scale on the atoms that carry LJ, as pmemd-pgm, or per atom type), the
  prmtop's exclusions (GVDW exceptions with A = C6 = 0); the 1-4 pairs keep their scaled Lennard-Jones
  (NonbondedForce exceptions);
  dispersion correction of the C6 term with lj_lrc.

Units: OpenMM's (nm, kJ/mol).
"""

from __future__ import annotations

import math
import re

import numpy as np
import openmm as mm
from openmm import app, unit

import openmmpgm

KCAL = 4.184

# pmemd-pgm GVDW water models of the GVDW manuscript (Huang, Luo, Duan; O-O only): &pol_gauss values
PGM3P_GVDW = {
    "slater": {"rep": "slater", "arep": 87500.0, "c6": 594.825035, "b": 4.52},
    "gauss": {"rep": "gauss", "arep": 422.0, "c6": 594.825035, "b": 0.9453},
}


def gvdw_from_pmemd(arep_kcal: float, c6_kcal_A6: float, b: float) -> tuple[float, float, float]:
    """Return per-atom (sqrt A [sqrt(kJ/mol)], sqrt C6 [sqrt(kJ/mol nm^6)], b) for pmemd-pgm's GVDW pair
    parameters gvdw_arep [kcal/mol], gvdw_c6 [kcal/mol A^6], b_rep_scale."""
    return math.sqrt(arep_kcal * KCAL), math.sqrt(c6_kcal_A6 * KCAL * 1e-6), float(b)


def read_sections(path: str, names: tuple[str, ...] | None = None) -> dict:
    """Return the %FLAG sections of a prmtop as numpy arrays (numbers) or lists of strings."""
    out, name, fmt, buf = {}, None, None, []

    def flush():
        if name is None or (names is not None and name not in names):
            return
        m = re.match(r"\((\d+)([aAiIeEfF])(\d+)", fmt or "")
        text = "".join(buf)
        if m is None:
            out[name] = buf
            return
        count, kind, width = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        vals = []
        for line in buf:
            for k in range(0, len(line), width):
                tok = line[k:k + width]
                if tok.strip():
                    vals.append(tok.strip())
        if kind == "a":
            out[name] = vals
        elif kind == "i":
            out[name] = np.array([int(v) for v in vals], int)
        else:
            out[name] = np.array([float(v) for v in vals], float)

    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("%FLAG"):
                flush()
                name, fmt, buf = line.split()[1], None, []
            elif line.startswith("%FORMAT"):
                fmt = line[line.index("(") :]
            elif line.startswith("%COMMENT") or line.startswith("%VERSION"):
                continue
            elif name is not None:
                buf.append(line)
        flush()
    return out


def pgm_parameters(path: str) -> dict:
    """Return the pGM parameters of a pmemd-pgm prmtop in OpenMM units.

    Keys: q (e), radius (nm), alpha (nm^3), cov_atom, cov_partner (0-based), cov_strength (e nm).
    """
    s = read_sections(path)
    if "POL_GAUSS_MONOPOLES_LIST" not in s:
        raise ValueError(f"{path}: not a pGM prmtop (no POL_GAUSS_MONOPOLES_LIST)")
    nptr = s["POL_GAUSS_COVALENT_POINTERS_LIST"].astype(int)
    atoms = s["POL_GAUSS_COVALENT_ATOMS_LIST"].astype(int) - 1
    cov_atom = np.repeat(np.arange(len(nptr)), nptr)
    return {
        "q": np.asarray(s["POL_GAUSS_MONOPOLES_LIST"], float),
        "radius": np.asarray(s["POL_GAUSS_RADII_LIST"], float) * 0.1,
        "alpha": np.asarray(s["POL_GAUSS_POLARIZABILITY_LIST"], float) * 1e-3,
        "cov_atom": cov_atom,
        "cov_partner": atoms,
        "cov_strength": np.asarray(s["POL_GAUSS_COVALENT_DIPOLES_LIST"], float) * 0.1,
        "sections": s,
    }


def lj_tail_constant(s: dict) -> float:
    """Return sum_ij B_ij over all ordered atom pairs [kJ/mol nm^6] (B: LENNARD_JONES_BCOEF)."""
    ntypes = int(s["POINTERS"][1])
    types = s["ATOM_TYPE_INDEX"].astype(int) - 1
    nbi = s["NONBONDED_PARM_INDEX"].astype(int)
    B = np.asarray(s["LENNARD_JONES_BCOEF"], float)
    counts = np.bincount(types, minlength=ntypes).astype(float)
    total = 0.0
    for a in range(ntypes):
        for b in range(ntypes):
            idx = nbi[a * ntypes + b]
            if idx > 0:
                total += counts[a] * counts[b] * B[idx - 1]
    return total * KCAL * 1e-6  # kcal/mol A^6 -> kJ/mol nm^6


def create_system(
    prmtop: str,
    cutoff: float = 0.9,
    ewald_beta: float = 0.0,
    grid: tuple[int, int, int] | None = None,
    order: int = 6,
    ewald_tol: float = 5e-4,
    periodic: bool = True,
    polarization: bool = True,
    lj: bool = True,
    lj_lrc: bool = True,
    vdw: str = "lj",
    gvdw: dict | str | None = None,
    rigid_water: bool = True,
    constraints=None,
    coulomb: float | None = None,
    dipole_tol: float = 1e-5,
    max_iter: int = 100,
    peek: float = 0.65,
    predictor: str = "mu4",
    pgm_group: int = 0,
    lj_group: int = 1,
) -> tuple[mm.System, app.Topology, openmmpgm.PGMForce]:
    """Return (system, topology, pgm_force) for a pmemd-pgm prmtop.

    cutoff [nm] applies to the direct-space electrostatics and the LJ; ewald_beta [1/nm] and grid set
    the PME (ewald_beta = 0: chosen from ewald_tol as NonbondedForce does).  predictor: "none", "mu3",
    "mu4".  coulomb: the Coulomb constant (None: OpenMM's; pmemd-pgm's is 332.05382 * 0.4184).

    vdw: "lj" (the prmtop's Lennard-Jones), "gvdw" or "none" (lj=False is "none").  gvdw: a key of
    PGM3P_GVDW, a pmemd-style dict {"rep": "slater"|"gauss", "arep": kcal/mol, "c6": kcal/mol A^6,
    "b": b_rep_scale} applied to every atom with a nonzero LJ epsilon (pmemd-pgm's rule), or
    {"rep": ..., "types": {amber_atom_type: (sqrtA, sqrtC6, b)}} in OpenMM units.
    """
    if not lj:
        vdw = "none"
    if vdw not in ("lj", "gvdw", "none"):
        raise ValueError(f"vdw must be lj, gvdw or none, not {vdw!r}")
    amber = app.AmberPrmtopFile(prmtop)
    method = app.CutoffPeriodic if periodic else app.NoCutoff
    kw = {"nonbondedCutoff": cutoff * unit.nanometer} if periodic else {}
    system = amber.createSystem(nonbondedMethod=method, rigidWater=rigid_water, constraints=constraints, **kw)
    par = pgm_parameters(prmtop)
    n = system.getNumParticles()
    if len(par["q"]) != n:
        raise ValueError("POL_GAUSS sections and the prmtop disagree on the number of atoms")

    # Lennard-Jones: AmberPrmtopFile's NonbondedForce without charges
    for f in list(system.getForces()):
        if isinstance(f, mm.NonbondedForce):
            nb = f
    for i in range(n):
        _, sig, eps = nb.getParticleParameters(i)
        nb.setParticleParameters(i, 0.0, sig, eps)
    for k in range(nb.getNumExceptions()):
        i, j, _, sig, eps = nb.getExceptionParameters(k)
        nb.setExceptionParameters(k, i, j, 0.0, sig, eps)
    nb.setUseDispersionCorrection(False)
    nb.setUseSwitchingFunction(False)
    nb.setReactionFieldDielectric(1.0)
    nb.setForceGroup(lj_group)
    if vdw == "none":
        system.removeForce([k for k, f in enumerate(system.getForces()) if isinstance(f, mm.NonbondedForce)][0])
    elif vdw == "lj" and periodic and lj_lrc:
        c = 2.0 * math.pi * lj_tail_constant(par["sections"]) / (3.0 * cutoff**3)
        tail = openmmpgm.TailCorrectionForce(-float(c))   # E = -c/V (CustomVolumeForce: host copies every step)
        tail.setForceGroup(lj_group)
        system.addForce(tail)
    elif vdw == "gvdw":
        add_gvdw(system, nb, par, gvdw, cutoff, periodic, lj_lrc, lj_group)

    # pGM electrostatics
    pgm = openmmpgm.PGMForce()
    for i in range(n):
        pgm.addParticle(float(par["q"][i]), float(par["radius"][i]), float(par["alpha"][i]))
    for a, b, c in zip(par["cov_atom"], par["cov_partner"], par["cov_strength"]):
        pgm.addCovalentDipole(int(a), int(b), float(c))
    if periodic:
        pgm.setNonbondedMethod(openmmpgm.PGMForce.PME)
        pgm.setCutoffDistance(cutoff)
        pgm.setEwaldErrorTolerance(ewald_tol)
        if ewald_beta > 0:
            if grid is None:
                raise ValueError("give the PME grid together with ewald_beta")
            pgm.setPMEParameters(ewald_beta, *grid)
        pgm.setPMEOrder(order)
    else:
        pgm.setNonbondedMethod(openmmpgm.PGMForce.NoCutoff)
    pgm.setPolarization(openmmpgm.PGMForce.Mutual if polarization else openmmpgm.PGMForce.NoPolarization)
    pgm.setInducedDipoleTolerance(dipole_tol)
    pgm.setMaxInducedDipoleIterations(max_iter)
    pgm.setPeekCoefficient(peek)
    pgm.setPredictor({"none": 0, "mu3": 1, "mu4": 2}[predictor])
    if coulomb is not None:
        pgm.setCoulombConstant(coulomb)
    pgm.setForceGroup(pgm_group)
    system.addForce(pgm)
    return system, amber.topology, pgm


def add_gvdw(system, nb, par, gvdw, cutoff, periodic, lrc, group):
    """Replace the Lennard-Jones of `nb` (a charge-free NonbondedForce of `system`) by a GVDWForce."""
    if gvdw is None:
        raise ValueError("vdw='gvdw' needs the gvdw parameters (e.g. gvdw='slater')")
    if isinstance(gvdw, str):
        gvdw = PGM3P_GVDW[gvdw]
    n = system.getNumParticles()
    eps = np.array([nb.getParticleParameters(i)[2].value_in_unit(unit.kilojoule_per_mole) for i in range(n)])
    if "types" in gvdw:
        types = par["sections"]["AMBER_ATOM_TYPE"]
        prm = [gvdw["types"].get(t, (0.0, 0.0, 1.0)) for t in types]
    else:
        one = gvdw_from_pmemd(gvdw["arep"], gvdw["c6"], gvdw["b"])
        prm = [one if e > 0 else (0.0, 0.0, 1.0) for e in eps]
    f = openmmpgm.GVDWForce()
    f.setRepulsionForm({"gauss": openmmpgm.GVDWForce.Gaussian, "slater": openmmpgm.GVDWForce.Slater}[gvdw["rep"]])
    for i in range(n):
        f.addParticle(float(par["radius"][i]), *(float(v) for v in prm[i]))
    # every exception of the prmtop (1-2, 1-3, 1-4) is excluded from GVDW; the 1-4 pairs keep the prmtop's
    # scaled Lennard-Jones below (pmemd-pgm evaluates GVDW only in the nonbonded pair list)
    for k in range(nb.getNumExceptions()):
        i, j, _, _, _ = nb.getExceptionParameters(k)
        f.addException(i, j, 0.0, 0.0, 0.0)
    if periodic:
        f.setNonbondedMethod(openmmpgm.GVDWForce.CutoffPeriodic)
        f.setCutoffDistance(cutoff)
        f.setUseDispersionCorrection(lrc)
    else:
        f.setNonbondedMethod(openmmpgm.GVDWForce.NoCutoff)
    f.setForceGroup(group)
    system.addForce(f)
    # the Lennard-Jones goes; the scaled 1-4 Lennard-Jones (exceptions with epsilon != 0) stays
    for i in range(n):
        q, sig, _ = nb.getParticleParameters(i)
        nb.setParticleParameters(i, q, sig, 0.0)
    if not any(nb.getExceptionParameters(k)[4].value_in_unit(unit.kilojoule_per_mole) != 0
               for k in range(nb.getNumExceptions())):
        system.removeForce([k for k, g in enumerate(system.getForces()) if isinstance(g, mm.NonbondedForce)][0])
    return f
