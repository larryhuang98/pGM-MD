#!/usr/bin/env bash
# =============================================================================
# pGM for AMBER24: regression tests
#
# Runs 20-step NVE trajectories of a 2,6-dimethylphenol liquid (pGM, Lennard-Jones van der Waals)
# and of the same box with Gaussian van der Waals (igvdw=1, including 1-4 pairs), with every pGM
# executable that is found, and compares Etot and EPtot at steps 1, 10 and 20 with reference.txt.
# The GPU executables also run the LIPCG solver (scf_solv_opt=3) on the Lennard-Jones box.
#
# Usage: test/run_tests.sh [BIN_DIR]      (default: $AMBERHOME/bin)
#   Executables tested when present: pmemd.pgm, pmemd.pgm.cuda_DPFP, pmemd.pgm.cuda_SPFP
#   Tolerances (kcal/mol): 0.002 for pmemd.pgm and DPFP, 0.05 for SPFP.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${1:-${AMBERHOME:-}/bin}"
[ -d "$BIN" ] || { echo "run_tests: no bin directory (give BIN_DIR or set AMBERHOME)"; exit 2; }
command -v python3 >/dev/null || { echo "run_tests: python3 is required"; exit 2; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/pgm_tests.XXXXXX")"
gunzip -c "$HERE/dmp/prmtop.gz" > "$WORK/prmtop"
python3 "$HERE/make_gvdw_prmtop.py" "$WORK/prmtop" "$WORK/prmtop_gvdw" 124.880270 79.315764
cp "$HERE/dmp/restrt0" "$WORK/"

nfail=0; nrun=0
check() {   # check OUTFILE SYSTEM TOL LABEL
  python3 - "$1" "$2" "$3" "$HERE/reference.txt" "$4" <<'EOF'
import re, sys
out, system, tol, ref_file, label = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
ref = {}
for line in open(ref_file):
    if line.startswith("#") or not line.strip():
        continue
    s, step, e, p = line.split()
    if s == system:
        ref[int(step)] = (float(e), float(p))
got, step = {}, None
try:
    for line in open(out):
        if "A V E R" in line:
            break
        m = re.match(r"\s*NSTEP =\s*(\d+)", line)
        if m:
            step = int(m.group(1))
        m = re.match(r"\s*Etot\s*=\s*(\S+).*EPtot\s*=\s*(\S+)", line)
        if m and step in ref:
            got[step] = (float(m.group(1)), float(m.group(2)))
except OSError:
    pass
if len(got) != len(ref):
    print(f"FAIL  {label}: run did not finish (see {out})")
    sys.exit(1)
dmax = max(max(abs(got[s][0] - ref[s][0]), abs(got[s][1] - ref[s][1])) for s in ref)
print(f"{'PASS' if dmax <= tol else 'FAIL'}  {label}: max |dE| = {dmax:.4f} kcal/mol (tolerance {tol})")
sys.exit(0 if dmax <= tol else 1)
EOF
}

for exe in pmemd.pgm pmemd.pgm.cuda_DPFP pmemd.pgm.cuda_SPFP; do
  [ -x "$BIN/$exe" ] || { echo "skip  $exe (not in $BIN)"; continue; }
  case $exe in
    pmemd.pgm)           tol=0.002; sfx=cpu ;;
    pmemd.pgm.cuda_DPFP) tol=0.002; sfx=gpu ;;
    pmemd.pgm.cuda_SPFP) tol=0.05;  sfx=gpu ;;
  esac
  systems="lj gvdw"; [ $sfx = gpu ] && systems="lj gvdw lipcg"
  for system in $systems; do
    case $system in
      lj)    top=prmtop;      inp="$HERE/dmp/mdin_$sfx";       ref=lj ;;
      gvdw)  top=prmtop_gvdw; inp="$HERE/dmp/mdin_gvdw_$sfx";  ref=gvdw ;;
      lipcg) top=prmtop;      inp="$HERE/dmp/mdin_lipcg_gpu";  ref=lj ;;
    esac
    out="$WORK/${exe}_$system.out"
    (cd "$WORK" && "$BIN/$exe" -O -i "$inp" -o "$out" -p $top -c restrt0 -r "$WORK/${exe}_$system.rst" \
        -inf "$WORK/${exe}_$system.inf" > "$WORK/${exe}_$system.log" 2>&1)
    nrun=$((nrun + 1))
    check "$out" $ref $tol "$exe $system" || nfail=$((nfail + 1))
  done
done

echo "pGM tests: $nrun run, $nfail failed (outputs in $WORK)"
[ $nrun -gt 0 ] || exit 2
[ $nfail -eq 0 ]
