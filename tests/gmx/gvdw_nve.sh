#!/bin/bash
# NVE of the GVDW manuscript's box with pGM + GVDW (Slater) in GROMACS: energy conservation.
# Run after gvdw_sp.sh in the same directory.   gvdw_nve.sh [cuda|double] [nsteps]
set -e
V=${1:-cuda}
G=$HOME/project/gromacs-pgm/install-$V/bin/gmx
[ "$V" = double ] && G=${G}_d
export LD_LIBRARY_PATH=/home8/apps/gcc/11.3.0/lib64:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=
sed -e "s/^nsteps = 0/nsteps = ${2:-4000}/" -e "s/^verlet-buffer-tolerance = -1/verlet-buffer-tolerance = 0.0001/" \
    -e "/^rlist/d" -e "s/^pgm-dipole-tol = 1e-6/pgm-dipole-tol = 1e-5/" -e "s/^pgm-predictor = 0/pgm-predictor = 2/" \
    -e "s/^continuation = yes/continuation = no\ngen-vel = yes\ngen-temp = 298\ngen-seed = 11\ndt = 0.001\nnstenergy = 10\nnstcalcenergy = 10\nnstlog = 1000/" \
    -e "/^nstcalcenergy = 1$/d" -e "/^nstenergy = 1$/d" slater.mdp > nve.mdp
$G grompp -f nve.mdp -c slater.g96 -p slater.top -o nve.tpr -maxwarn 2 > grompp_nve.log 2>&1 || { tail -20 grompp_nve.log; exit 1; }
$G mdrun -s nve.tpr -deffnm nve -nb cpu -ntmpi 1 -ntomp 8 > mdrun_nve.log 2>&1 || { tail -20 mdrun_nve.log; exit 1; }
echo -e "Potential\nTotal-Energy\nTemperature\n" | $G energy -f nve.edr -o nve.xvg > energy_nve.log 2>&1
grep -E "^(Potential|Total Energy|Temperature)" energy_nve.log
grep -E "Performance|pGM:" nve.log mdrun_nve.log
