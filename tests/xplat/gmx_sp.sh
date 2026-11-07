#!/bin/bash
# GROMACS single point of a pmemd-pgm prmtop/restart (converged induction, flexible bonds, no dispersion
# correction, pmemd-pgm's Coulomb constant through epsilon-r), for gmx_vs_amber.py.
#   gmx_sp.sh GMX PRMTOP RST NAME RC_nm GRID ORDER BETA_nm      (in an empty work directory)
# GMX_PGM_GPU=1 in the environment runs the pGM terms on the GPU (mixed-precision CUDA build).
set -e
H=$(cd "$(dirname "$0")" && pwd)
. "$H/../env.sh"
G=$1; shift
T=$PGM_ROOT/gromacs/tools
$PY $T/prmtop_to_gmx.py $1 $2 $3 > conv.log 2>&1 || { tail conv.log; exit 1; }
$OMM_PY $H/rst_to_npz.py $2 rst.npz > /dev/null && $PY $T/npz_to_g96.py rst.npz $3.gro $3.g96 > /dev/null
RTOL=$($PY -c "import math; print(math.erfc($7 * $4))")
EPSR=$($PY -c "import sys; sys.path.insert(0, '$T'); import prmtop_to_gmx as m; print(repr(m.EPSILON_R_PMEMD))")
cat > sp.mdp <<M
integrator = md
nsteps = 0
dt = 0.000001
cutoff-scheme = Verlet
verlet-buffer-tolerance = -1
rlist = $4
nstlist = 10
pbc = xyz
coulombtype = pGM
rcoulomb = $4
vdwtype = cut-off
vdw-modifier = None
rvdw = $4
DispCorr = no
fourier-nx = $5
fourier-ny = $5
fourier-nz = $5
pme-order = $6
ewald-rtol = $RTOL
pgm-dipole-tol = 1e-10
pgm-max-iter = 1000
pgm-predictor = 0
pgm-peek = 0
epsilon-r = $EPSR
constraints = none
tcoupl = no
pcoupl = no
nstcalcenergy = 1
nstenergy = 1
nstlog = 1
nstfout = 1
continuation = yes
M
$G grompp -f sp.mdp -c $3.g96 -p $3.top -o sp.tpr -maxwarn 5 > grompp.log 2>&1 || { tail -30 grompp.log; exit 1; }
$G mdrun -s sp.tpr -deffnm gsp -nb cpu -ntmpi 1 -ntomp ${NT:-8} > mdrun.log 2>&1 || { tail -30 mdrun.log; exit 1; }
