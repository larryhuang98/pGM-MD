#!/bin/bash
# GVDW with 1-4 pairs: a pGM organic molecule (Amber test qm7b_7203, 20 atoms, dihedrals) converted with
# the prmtop's LJ and with --gvdw slater/gauss, alone in a 4 nm box (rvdw 1.2 nm covers every
# intramolecular pair).  The GROMACS energies are compared with OpenMM (pgm_prmtop, NoCutoff) by
# gvdw_14_compare.py: LJ (SR) + LJ-14, the bonded terms, and with GVDW the 1-4 pairs in the GVDW (no LJ-14).
#   gvdw_14.sh [double|cuda]      (run in an empty work directory)
set -e
V=${1:-double}
D=$(cd "$(dirname "$0")" && pwd)
. "$(cd "$(dirname "$0")" && pwd)/../env.sh"
G=$(gmx_bin $V)
export CUDA_VISIBLE_DEVICES= GMX_PGM_QUIET=1
P=$PY
SRC=${QM7B_DIR:-$AMBERHOME/test/nmode}   # qm7b_7203.prmtop/.inpcrd of the AMBER test suite
cp $SRC/qm7b_7203.prmtop $SRC/qm7b_7203.inpcrd .
for m in lj slater gauss; do
    opt=$([ $m = lj ] && echo "" || echo "--gvdw $m")
    $P $D/../../gromacs/tools/prmtop_to_gmx.py qm7b_7203.prmtop qm7b_7203.inpcrd $m $opt > conv_$m.log 2>&1 \
        || { cat conv_$m.log; exit 1; }
    $G editconf -f $m.gro -o ${m}_box.gro -box 4 4 4 -c > /dev/null 2>&1
    rf=$([ $m = gauss ] && echo 0 || echo 1)
    cat > $m.mdp <<MDP
integrator = md
nsteps = 0
cutoff-scheme = Verlet
verlet-buffer-tolerance = -1
rlist = 1.3
nstlist = 10
pbc = xyz
coulombtype = pGM
rcoulomb = 1.2
vdwtype = cut-off
vdw-modifier = None
rvdw = 1.2
DispCorr = no
fourier-nx = 40
fourier-ny = 40
fourier-nz = 40
pme-order = 6
ewald-rtol = 1e-6
pgm-dipole-tol = 1e-8
pgm-predictor = 0
pgm-gvdw-rep-form = $rf
nstcalcenergy = 1
nstenergy = 1
constraints = none
MDP
    $G grompp -f $m.mdp -c ${m}_box.gro -p $m.top -o $m.tpr -maxwarn 2 > grompp_$m.log 2>&1 || { tail -20 grompp_$m.log; exit 1; }
    $G mdrun -s $m.tpr -deffnm $m -nb cpu -ntmpi 1 -ntomp 4 > mdrun_$m.log 2>&1 || { tail -20 mdrun_$m.log; exit 1; }
    printf "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n\n" | $G energy -f $m.edr -dp -o e_$m.xvg > /dev/null 2>&1
done
grep -c "^ *[0-9]" <(sed -n '/\[ pairs \]/,/\[ angles \]/p' lj.top) | sed 's/^/1-4 pairs (LJ conversion): /'
