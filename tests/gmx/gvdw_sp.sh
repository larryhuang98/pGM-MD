#!/bin/bash
# GVDW single points of the GVDW manuscript's 512-water box with GROMACS (pGM + GVDW, rvdw 1.0 nm, no tail),
# both repulsion forms; the dump (GMX_PGM_DUMP) has the GVDW energy, virial and forces.
#   gvdw_sp.sh [cuda|double]      (run in an empty work directory; needs runs/gvdw_box.txt)
set -e
V=${1:-double}
D=$(cd "$(dirname "$0")" && pwd)
R=$D/../../runs
G=$HOME/project/gromacs-pgm/install-$V/bin/gmx
[ "$V" = double ] && G=${G}_d
export LD_LIBRARY_PATH=/home8/apps/gcc/11.3.0/lib64:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES= GMX_PGM_QUIET=1
P=$HOME/miniconda3/envs/SE3nv/bin/python
GD=$HOME/pgm-gvdw-data
for form in slater gauss; do
    [ -f $form.top ] || $P $D/../../gromacs/tools/prmtop_to_gmx.py $GD/topology/rayl_512_v2.prmtop $GD/inputs/lj/inpcrd.restrt $form --gvdw $form > conv_$form.log 2>&1
    $P $D/boxtxt_to_g96.py $R/gvdw_box.txt $form.gro $form.g96 > /dev/null
    rf=$([ $form = slater ] && echo 1 || echo 0)
    cat > $form.mdp <<MDP
integrator = md
nsteps = 0
cutoff-scheme = Verlet
verlet-buffer-tolerance = -1
rlist = 1.1
nstlist = 10
pbc = xyz
coulombtype = pGM
rcoulomb = 1.0
vdwtype = cut-off
vdw-modifier = None
rvdw = 1.0
DispCorr = no
fourier-nx = 40
fourier-ny = 40
fourier-nz = 40
pme-order = 6
ewald-rtol = 1e-6
pgm-dipole-tol = 1e-6
pgm-predictor = 0
pgm-gvdw-rep-form = $rf
nstcalcenergy = 1
nstenergy = 1
continuation = yes
MDP
    $G grompp -f $form.mdp -c $form.g96 -p $form.top -o $form.tpr -maxwarn 2 > grompp_$form.log 2>&1 || { tail -20 grompp_$form.log; exit 1; }
    GMX_PGM_DUMP=dump_$form.txt $G mdrun -s $form.tpr -deffnm $form -nb cpu -ntmpi 1 -ntomp 8 > mdrun_$form.log 2>&1 || { tail -20 mdrun_$form.log; exit 1; }
    echo "== $form"; grep -A4 "Energies (kJ/mol)" $form.log | head -5
done
