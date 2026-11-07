#!/bin/bash
# Single point of water512 with GROMACS coulombtype = pGM; dump of the pGM terms, forces, dipoles, virial.
#   sp_w512.sh [cuda|double] [ntomp]       (run in a copy of tests/data/gmx_w512: sp.mdp, w512.g96, w512.top)
set -e
V=${1:-cuda}
. "$(cd "$(dirname "$0")" && pwd)/../env.sh"
G=$(gmx_bin $V)
export CUDA_VISIBLE_DEVICES=
$G grompp -f sp.mdp -c w512.g96 -p w512.top -o sp_$V.tpr -po mdout_$V.mdp -maxwarn 2 > grompp_$V.log 2>&1 \
  || { tail -30 grompp_$V.log; exit 1; }
GMX_PGM_DUMP=dump_$V.txt $G mdrun -s sp_$V.tpr -deffnm sp_$V -nb cpu -ntmpi 1 -ntomp ${2:-8} > mdrun_$V.log 2>&1 \
  || { tail -30 mdrun_$V.log; exit 1; }
grep -A12 "Energies (kJ/mol)" sp_$V.log | head -14
grep "pGM:" mdrun_$V.log || true
