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
  all ordered atom pairs, as a CustomVolumeForce (only the r^-6 term, unlike OpenMM's correction).

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
    """
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
    if not lj:
        system.removeForce([k for k, f in enumerate(system.getForces()) if isinstance(f, mm.NonbondedForce)][0])
    elif periodic and lj_lrc:
        c = 2.0 * math.pi * lj_tail_constant(par["sections"]) / (3.0 * cutoff**3)
        tail = mm.CustomVolumeForce(f"-{float(c):.17g}/v")
        tail.setForceGroup(lj_group)
        system.addForce(tail)

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
