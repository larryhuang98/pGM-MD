#!/bin/bash
# water512 NPT with GROMACS coulombtype = pGM; density, potential energy and pressure averages.
#   npt_w512.sh [cuda|double] [ntomp] [skip_ps]     (run in runs/gmx_w512)
set -e
V=${1:-cuda}
D=$(cd "$(dirname "$0")" && pwd)
G=$HOME/project/gromacs-pgm/install-$V/bin/gmx
[ "$V" = double ] && G=${G}_d
export LD_LIBRARY_PATH=/home8/apps/gcc/11.3.0/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=
N=npt_$V
$G grompp -f $D/npt.mdp -c w512.g96 -p w512.top -o $N.tpr -po mdout_$N.mdp -maxwarn 2 > grompp_$N.log 2>&1 \
  || { tail -30 grompp_$N.log; exit 1; }
$G mdrun -s $N.tpr -deffnm $N -nb cpu -ntmpi 1 -ntomp ${2:-8} > mdrun_$N.log 2>&1 || { tail -30 mdrun_$N.log; exit 1; }
echo -e "Density\nPotential\nPressure\nTemperature\nVolume\n" | $G energy -f $N.edr -b ${3:-30} -o $N.xvg > energy_$N.log 2>&1
grep -E "^(Density|Potential|Pressure|Temperature|Volume)" energy_$N.log
grep -E "Performance|pGM:" $N.log mdrun_$N.log
