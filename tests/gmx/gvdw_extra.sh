#!/bin/bash
# GVDW overrides ([ pgm_gvdw_pairs ]: an O-H pair of every water, excluded otherwise) and the dispersion
# tail (DispCorr = EnerPres), Slater form; run after gvdw_sp.sh in the same directory.
#   gvdw_extra.sh [cuda|double]
set -e
V=${1:-double}
D=$(cd "$(dirname "$0")" && pwd)
G=$HOME/project/gromacs-pgm/install-$V/bin/gmx
[ "$V" = double ] && G=${G}_d
export LD_LIBRARY_PATH=/home8/apps/gcc/11.3.0/lib64:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES= GMX_PGM_QUIET=1
python3 - <<PY
s = open("slater.top").read()
i = s.index("[ system ]")
s = s[:i] + "[ pgm_gvdw_pairs ]\n; ai aj funct  A(kJ/mol)  C6(kJ/mol nm^6)  B\n    1  2  1  300.0  0.002  1.5\n\n" + s[i:]
open("extra.top", "w").write(s)
PY
sed "s/^DispCorr = no/DispCorr = EnerPres/" slater.mdp > extra.mdp
$G grompp -f extra.mdp -c slater.g96 -p extra.top -o extra.tpr -maxwarn 2 > grompp_extra.log 2>&1 || { tail -20 grompp_extra.log; exit 1; }
GMX_PGM_DUMP=dump_extra.txt $G mdrun -s extra.tpr -deffnm extra -nb cpu -ntmpi 1 -ntomp 8 > mdrun_extra.log 2>&1 || { tail -20 mdrun_extra.log; exit 1; }
grep -A6 "Energies (kJ/mol)" extra.log | head -7
