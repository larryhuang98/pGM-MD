#!/bin/bash
# GROMACS pGM on the GPU (GMX_PGM_GPU=1, G3) against the CPU path of the same mixed-precision build, on a
# GPU node: single points (energies, virial, forces) of water512, 4096 waters (the pmemd benchmark
# settings) and the GVDW box with overrides and tail; NVE energy conservation of the 4096 waters (GPU and
# CPU; with GVDW on the GPU); a water512 NPT on the GPU (C-rescale: the virial and box changes); the
# benchmark speed with LJ and with GVDW.
#   gpu_check.sh            (in an empty work directory; needs runs/gmx_w512, runs/gmx_gpu4k, runs/gmx_gvdw_cuda)
set -e
D=$(cd "$(dirname "$0")" && pwd)
R=$D/../../runs
G=$HOME/project/gromacs-pgm/install-cuda/bin/gmx
P=$HOME/miniconda3/envs/SE3nv/bin/python
export LD_LIBRARY_PATH=/home3/apps/cuda-12.8/lib64:/home8/apps/gcc/11.3.0/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} GMX_PGM_QUIET=1
NT=${NT:-8}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

run() {   # run MODE NAME TPR [mdrun options]: MODE cpu or gpu
    local mode=$1 name=$2 tpr=$3
    shift 3
    if [ $mode = gpu ]; then export GMX_PGM_GPU=1; else unset GMX_PGM_GPU; fi
    $G mdrun -s $tpr -deffnm $name -nb cpu -ntmpi 1 -ntomp $NT -pin on "$@" > mdrun_$name.log 2>&1 \
        || { tail -20 mdrun_$name.log; exit 1; }
}
energies() {   # all terms, full precision
    local n=$($G dump -e $1.edr 2>/dev/null | sed -n '/energy components/,/^$/p' | grep -c "(")
    seq 1 $n | $G energy -f $1.edr -dp -o $1_e.xvg > /dev/null 2>&1
}
sp() {   # sp NAME MDP CONF TOP
    (sed -e "/^nstfout/d" -e "s/^nsteps .*/nsteps = 0/" $2; echo "nstfout = 1") > $1.mdp
    $G grompp -f $1.mdp -c $3 -p $4 -o $1.tpr -maxwarn 2 > grompp_$1.log 2>&1 || { tail grompp_$1.log; exit 1; }
    for mode in cpu gpu; do
        run $mode $1_$mode $1.tpr
        energies $1_$mode
    done
    echo "== single point $1: GPU vs CPU"
    $P $D/dd_compare.py --gpu $1_cpu $1_gpu || true
}
sp w512 $R/gmx_w512/sp.mdp $R/gmx_w512/w512.g96 $R/gmx_w512/w512.top
sp w4096 $R/gmx_gpu4k/bench.mdp $R/gmx_gpu4k/w4096.gro $R/gmx_gpu4k/w4096.top
GV=$R/gmx_gvdw_cuda
sp gvdw $GV/extra.mdp $GV/slater.g96 $GV/extra.top
# GVDW, 4096 waters (the GVDW box replicated, benchmark settings), overrides + tail
$G genconf -f $GV/slater.g96 -nbox 2 2 2 -o g4096.g96 > /dev/null 2>&1
for t in slater extra; do
    awk '/\[ molecules \]/{m=1} m && $2 ~ /^[0-9]+$/ {$2 = $2 * 8} {print}' $GV/$t.top > g4096_$t.top
done
(cat $R/gmx_gpu4k/bench.mdp; echo "pgm-gvdw-rep-form = 1") > gbench.mdp
sp gvdw4096 gbench.mdp g4096.g96 g4096_extra.top

echo "== NVE, 4096 waters, 10 ps (1 fs)"
sed -e "s/^nsteps .*/nsteps = 10000/" -e "s/^nstcalcenergy .*/nstcalcenergy = 10/" \
    -e "s/^nstenergy .*/nstenergy = 10/" $R/gmx_gpu4k/bench.mdp > nve.mdp
$G grompp -f nve.mdp -c $R/gmx_gpu4k/w4096.gro -p $R/gmx_gpu4k/w4096.top -o nve.tpr -maxwarn 2 > grompp_nve.log 2>&1
for mode in gpu cpu; do
    run $mode nve_$mode nve.tpr
    echo -e "Total-Energy\nConserved-En.\n" | $G energy -f nve_$mode.edr > energy_nve_$mode.log 2>&1
    echo "$mode: $(grep '^Total Energy' energy_nve_$mode.log)  $(grep Performance nve_$mode.log)"
done

echo "== NVE, 4096 waters with GVDW (Slater), 10 ps, GPU"
sed -e "s/^nsteps .*/nsteps = 10000/" -e "s/^nstcalcenergy .*/nstcalcenergy = 10/" \
    -e "s/^nstenergy .*/nstenergy = 10/" gbench.mdp > gnve.mdp
$G grompp -f gnve.mdp -c g4096.g96 -p g4096_slater.top -o gnve.tpr -maxwarn 2 > grompp_gnve.log 2>&1
run gpu gnve_gpu gnve.tpr
echo -e "Total-Energy\nConserved-En.\n" | $G energy -f gnve_gpu.edr > energy_gnve_gpu.log 2>&1
echo "gpu: $(grep '^Total Energy' energy_gnve_gpu.log)  $(grep Performance gnve_gpu.log)"

echo "== NPT water512 on the GPU, 150 ps (C-rescale, v-rescale; the CPU run: runs/gmx_w512/energy_npt_cuda.log)"
$G grompp -f $D/npt.mdp -c $R/gmx_w512/w512.g96 -p $R/gmx_w512/w512.top -o npt.tpr -maxwarn 2 > grompp_npt.log 2>&1
run gpu npt_gpu npt.tpr
echo -e "Density\nPotential\nPressure\nTemperature\n" | $G energy -f npt_gpu.edr -b 30 > energy_npt_gpu.log 2>&1
grep -E "^(Density|Potential|Pressure|Temperature)" energy_npt_gpu.log
grep Performance npt_gpu.log

echo "== benchmark, 4096 waters, 3000 steps (pmemd.pgm.cuda 91.7 ns/day): LJ, GVDW (Slater)"
sed -e "s/^nsteps .*/nsteps = 3000/" $R/gmx_gpu4k/bench.mdp > bench.mdp
$G grompp -f bench.mdp -c $R/gmx_gpu4k/w4096.gro -p $R/gmx_gpu4k/w4096.top -o bench.tpr -maxwarn 2 > grompp_bench.log 2>&1
sed -e "s/^nsteps .*/nsteps = 3000/" gbench.mdp > gbench3k.mdp
$G grompp -f gbench3k.mdp -c g4096.g96 -p g4096_slater.top -o gbench.tpr -maxwarn 2 > grompp_gbench.log 2>&1
for b in bench gbench; do
    GMX_PGM_TIMING=1 run gpu ${b}_gpu $b.tpr
    echo "$b: $(grep Performance ${b}_gpu.log)"
    grep "pGM GPU ms" mdrun_${b}_gpu.log | tail -1
done
echo GPU_CHECK_DONE
