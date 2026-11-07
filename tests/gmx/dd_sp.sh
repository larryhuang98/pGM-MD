#!/bin/bash
# pGM with domain decomposition: a 4096-water box (water512 replicated 2x2x2, grid 72^3) single point on
# 1, 2, 4, 8 thread-MPI ranks; energies (all terms, virial, pressure) and forces against the one-rank run
# (dd_compare.py), GVDW with overrides and tail on 1 and 4 ranks, then a 2 ps NVE on 1 and 4 ranks.
#   dd_sp.sh [double|cuda]        (run in an empty work directory)
set -e
V=${1:-double}
D=$(cd "$(dirname "$0")" && pwd)
. "$(cd "$(dirname "$0")" && pwd)/../env.sh"
G=$(gmx_bin $V)
export CUDA_VISIBLE_DEVICES= GMX_PGM_QUIET=1
P=$PY
W=$PGM_DATA/gmx_w512
$G genconf -f $W/w512.g96 -nbox 2 2 2 -o w4096.g96 > genconf.log 2>&1
sed "s/^\(WAT *\)512/\14096/" $W/w512.top > w4096.top
sed -e "s/^fourier-n\(.\) .*/fourier-n\1 = 72/" $W/sp.mdp > sp.mdp
echo "nstfout = 1" >> sp.mdp
$G grompp -f sp.mdp -c w4096.g96 -p w4096.top -o sp.tpr -maxwarn 2 > grompp.log 2>&1 || { tail -20 grompp.log; exit 1; }
for nr in 1 2 4 8; do
    $G mdrun -s sp.tpr -deffnm sp$nr -nb cpu -ntmpi $nr -ntomp 2 -npme 0 > mdrun_sp$nr.log 2>&1 || { tail -20 mdrun_sp$nr.log; exit 1; }
    n=$($G dump -e sp$nr.edr 2>/dev/null | sed -n '/energy components/,/^$/p' | grep -c "(")
    seq 1 $n | $G energy -f sp$nr.edr -dp -o sp${nr}_e.xvg > energy_sp$nr.log 2>&1
    echo "$nr ranks: $(grep 'Domain decomposition grid' sp$nr.log)"
done
# more decompositions: several pulses (8 x 1 x 1: 0.56 nm cells, 3 pulses), 3D, odd
for spec in "8 1 1:p811" "2 2 2:p222" "3 2 1:p321"; do
    dd=${spec%%:*}; nm=${spec##*:}; nr=$(( $(echo $dd | tr " " "*") ))
    $G mdrun -s sp.tpr -deffnm $nm -nb cpu -ntmpi $nr -ntomp 1 -npme 0 -dd $dd > mdrun_$nm.log 2>&1 || { tail -20 mdrun_$nm.log; exit 1; }
    n=$($G dump -e $nm.edr 2>/dev/null | sed -n '/energy components/,/^$/p' | grep -c "(")
    seq 1 $n | $G energy -f $nm.edr -dp -o ${nm}_e.xvg > energy_$nm.log 2>&1
    echo "-dd $dd: $(grep 'number of communication pulses' $nm.log)"
done
$P $D/dd_compare.py sp1 sp2 sp4 sp8 p811 p222 p321 || true

# GVDW (Slater, overrides of an O-H pair per water, dispersion tail) on the replicated GVDW box
GV=${GVDW_DIR:-../gmx_gvdw_$V}   # the GVDW single points of gvdw_sp.sh
if [ -f $GV/extra.top ]; then
    $G genconf -f $GV/slater.g96 -nbox 2 2 2 -o g4096.g96 > genconf_g.log 2>&1
    awk '/\[ molecules \]/{m=1} m && $2 ~ /^[0-9]+$/ {$2 = $2 * 8} {print}' $GV/extra.top > g4096.top
    sed -e "s/^fourier-n\(.\) .*/fourier-n\1 = 80/" $GV/extra.mdp > g.mdp
    echo "nstfout = 1" >> g.mdp
    $G grompp -f g.mdp -c g4096.g96 -p g4096.top -o g.tpr -maxwarn 2 > grompp_g.log 2>&1 || { tail -20 grompp_g.log; exit 1; }
    for nr in 1 4; do
        $G mdrun -s g.tpr -deffnm g$nr -nb cpu -ntmpi $nr -ntomp 2 -npme 0 > mdrun_g$nr.log 2>&1 || { tail -20 mdrun_g$nr.log; exit 1; }
        n=$($G dump -e g$nr.edr 2>/dev/null | sed -n '/energy components/,/^$/p' | grep -c "(")
        seq 1 $n | $G energy -f g$nr.edr -dp -o g${nr}_e.xvg > energy_g$nr.log 2>&1
    done
    echo "GVDW + overrides + tail:"
    $P $D/dd_compare.py g1 g4 || true
fi

# NVE, 1 and 4 ranks: the same energy conservation.  (Not the same trajectory: started from MD
# coordinates, the waters (update groups) stick out of the box by up to 0.08 nm and the nbnxm lists of
# 1 and 4 ranks then miss different few pairs just inside rc at the search, GMX_PGM_CHECK_PAIRS=1 shows
# them; the pair set is GROMACS's, as for its own kernels.  GROMACS's initial constraining also differs.)
sed -e "s/^fourier-n\(.\) .*/fourier-n\1 = 72/" -e "s/^nsteps .*/nsteps = 1/" $D/nve.mdp > vel.mdp
$G grompp -f vel.mdp -c w4096.g96 -p w4096.top -o vel.tpr -maxwarn 2 > grompp_vel.log 2>&1 || { tail -20 grompp_vel.log; exit 1; }
$G mdrun -s vel.tpr -c vel.g96 -e vel.edr -g vel.log -o vel.trr -cpo vel.cpt -nb cpu -ntmpi 1 -ntomp 8 > mdrun_vel.log 2>&1  # 300 K, constrained
sed -e "s/^fourier-n\(.\) .*/fourier-n\1 = 72/" -e "s/^nsteps .*/nsteps = 2000/" \
    -e "s/^continuation .*/continuation = yes/" -e "s/^gen-vel .*/gen-vel = no/" $D/nve.mdp > nve.mdp
$G grompp -f nve.mdp -c vel.g96 -p w4096.top -o nve.tpr -maxwarn 2 > grompp_nve.log 2>&1 || { tail -20 grompp_nve.log; exit 1; }
for nr in 1 4; do
    $G mdrun -s nve.tpr -deffnm nve$nr -nb cpu -ntmpi $nr -ntomp 2 -npme 0 > mdrun_nve$nr.log 2>&1 || { tail -20 mdrun_nve$nr.log; exit 1; }
    echo -e "Total-Energy\nConserved-En.\n" | $G energy -f nve$nr.edr -dp -o nve${nr}_e.xvg > energy_nve$nr.log 2>&1
    echo "NVE $nr ranks: $(grep -E '^Total Energy' energy_nve$nr.log)  $(grep Performance nve$nr.log)"
done
