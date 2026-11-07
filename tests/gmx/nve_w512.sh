#!/bin/bash
# water512 NVE with GROMACS coulombtype = pGM; prints drift and fluctuation of the conserved energy.
# MDRUN_EXTRA: more mdrun options, e.g. "-pin on" (unpinned OpenMP threads wander across NUMA nodes: ~30% slower)
#   nve_w512.sh [cuda|double] [ntomp] [mdp]     (run in a copy of tests/data/gmx_w512: sp.mdp, w512.g96, w512.top)
set -e
V=${1:-cuda}
D=$(cd "$(dirname "$0")" && pwd)
. "$(cd "$(dirname "$0")" && pwd)/../env.sh"
G=$(gmx_bin $V)
export CUDA_VISIBLE_DEVICES=
M=${3:-$D/nve.mdp}
N=nve_$V
$G grompp -f $M -c w512.g96 -p w512.top -o $N.tpr -po mdout_$N.mdp -maxwarn 2 > grompp_$N.log 2>&1 \
  || { tail -30 grompp_$N.log; exit 1; }
$G mdrun -s $N.tpr -deffnm $N -nb cpu -ntmpi 1 -ntomp ${2:-8} $MDRUN_EXTRA > mdrun_$N.log 2>&1 || { tail -30 mdrun_$N.log; exit 1; }
echo -e "Potential\nTotal-Energy\nConserved-En.\nTemperature\n" | $G energy -f $N.edr -o $N.xvg > energy_$N.log 2>&1
grep -E "^(Potential|Total Energy|Conserved En.|Temperature)" energy_$N.log
grep -E "Performance|pGM:" $N.log mdrun_$N.log
