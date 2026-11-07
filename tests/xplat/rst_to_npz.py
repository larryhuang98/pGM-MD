"""AMBER restart to an npz with positions and box vectors (nm), for gromacs/tools/npz_to_g96.py."""
import sys, numpy as np, openmm.app as app
from openmm import unit
c = app.AmberInpcrdFile(sys.argv[1])
x = np.array(c.getPositions().value_in_unit(unit.nanometer)); H = np.array([v.value_in_unit(unit.nanometer) for v in c.getBoxVectors()])
np.savez(sys.argv[2], pos=x, H=H); print(H)
