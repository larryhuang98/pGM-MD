#!/usr/bin/env bash
# =============================================================================
# pGM for AMBER PMEMD: installer
#
# Adds the polarizable Gaussian multipole (pGM) model to an AMBER PMEMD source
# tree and builds pmemd.pgm (CPU) and pmemd.pgm.cuda (GPU).  The standard
# pmemd / pmemd.cuda executables are not changed.
#
# Usage:
#   ./install.sh [--amber-src DIR] [--build-dir DIR] [--build] [--jobs N]
#   ./install.sh --uninstall [--amber-src DIR]
#
#   --amber-src DIR  AMBER source tree (the directory that contains src/pmemd).
#                    Default: $AMBER_SRC, or $AMBERHOME if it is a source tree.
#   --build-dir DIR  existing CMake build directory of that tree; with --build,
#                    the pGM targets are built and installed from it.
#   --build          build and install pmemd.pgm and pmemd.pgm.cuda after patching
#   --dry-run        check that the patch applies, change nothing
#   --uninstall      restore the files saved by a previous installation
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PGM_VERSION="$(cat "$HERE/VERSION" 2>/dev/null || echo dev)"
AMBER_SRC="${AMBER_SRC:-}"
BUILD_DIR=""
DO_BUILD=0
DRY=0
UNINSTALL=0
JOBS="${JOBS:-8}"

die()  { echo "pGM install: ERROR: $*" >&2; exit 1; }
info() { echo "pGM install: $*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --amber-src) AMBER_SRC="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --build)     DO_BUILD=1; shift ;;
    --jobs)      JOBS="$2"; shift 2 ;;
    --dry-run)   DRY=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help)   sed -n '2,22p' "$0"; exit 0 ;;
    *) die "unknown option $1" ;;
  esac
done

# --- locate the AMBER source tree --------------------------------------------
if [ -z "$AMBER_SRC" ] && [ -n "${AMBERHOME:-}" ] && [ -f "$AMBERHOME/src/pmemd/src/pmemd.F90" ]; then
  AMBER_SRC="$AMBERHOME"
fi
[ -n "$AMBER_SRC" ] || die "AMBER source tree not found; use --amber-src DIR (the directory that contains src/pmemd)"
AMBER_SRC="$(cd "$AMBER_SRC" && pwd)"
PM="$AMBER_SRC/src/pmemd/src"
[ -f "$PM/pmemd.F90" ] || die "$AMBER_SRC does not contain src/pmemd/src/pmemd.F90 (an installed AMBER without sources cannot be patched)"

BACKUP="$AMBER_SRC/.pgm_backup"

# --- uninstall ----------------------------------------------------------------
if [ $UNINSTALL -eq 1 ]; then
  [ -d "$BACKUP" ] || die "no backup found in $BACKUP"
  info "restoring AMBER files from $BACKUP"
  (cd "$BACKUP/orig" && find . -type f -print0 | while IFS= read -r -d '' f; do cp -p "$f" "$AMBER_SRC/$f"; done)
  if [ -f "$BACKUP/added.txt" ]; then
    while read -r f; do rm -f "$AMBER_SRC/$f"; rmdir "$AMBER_SRC/$(dirname "$f")" 2>/dev/null || true; done < "$BACKUP/added.txt"
  fi
  rm -rf "$BACKUP"
  info "done; rebuild AMBER to remove the pGM executables from the build"
  exit 0
fi

# --- identify the AMBER version (AMBER24 only) ----------------------------------
if [ -f "$BACKUP/installed" ]; then
  die "pGM $(cat "$BACKUP/installed") is already installed in $AMBER_SRC; run with --uninstall first"
elif [ -f "$PM/pGM_interface.F90" ]; then
  die "this AMBER tree already contains pGM (newer than AMBER24); only AMBER24 is supported"
elif grep -q "Amber 24 PMEMD" "$PM"/*.F90 2>/dev/null; then
  HOOKS=amber24
else
  die "unsupported AMBER version in $AMBER_SRC (only AMBER24 is supported)"
fi
HOOKDIR="$HERE/hooks/$HOOKS"
info "AMBER source: $AMBER_SRC   (detected: $HOOKS)   pGM patch: $PGM_VERSION"

# --- check that the hooks apply -----------------------------------------------
command -v patch >/dev/null || die "the 'patch' program is required"
( cd "$AMBER_SRC" && patch -p1 --dry-run --silent < "$HOOKDIR/hooks.patch" ) \
  || die "the $HOOKS hooks do not apply cleanly to this tree; nothing was changed. The patch needs AMBER24 with the official updates applied (run ./update_amber --update in the AMBER source tree first); locally modified AMBER sources are not supported"
info "hooks apply cleanly"
[ $DRY -eq 1 ] && { info "dry run: nothing changed"; exit 0; }

# --- back up files that will change, then install ------------------------------
mkdir -p "$BACKUP/orig"
: > "$BACKUP/added.txt"
grep '^+++ ' "$HOOKDIR/hooks.patch" | sed -e 's#^+++ b/##' -e 's#\t.*##' | while read -r f; do
  mkdir -p "$BACKUP/orig/$(dirname "$f")"; cp -p "$AMBER_SRC/$f" "$BACKUP/orig/$f"
done
(cd "$HERE" && find src \( -type f -o -type l \)) | while read -r f; do
  if [ -f "$AMBER_SRC/$f" ]; then
    mkdir -p "$BACKUP/orig/$(dirname "$f")"; cp -p "$AMBER_SRC/$f" "$BACKUP/orig/$f"
  else
    echo "$f" >> "$BACKUP/added.txt"
  fi
  mkdir -p "$AMBER_SRC/$(dirname "$f")"; rm -f "$AMBER_SRC/$f"; cp -P "$HERE/$f" "$AMBER_SRC/$f"   # -P keeps the pGM/*.i.F90 links
done
( cd "$AMBER_SRC" && patch -p1 --silent < "$HOOKDIR/hooks.patch" )
chmod a+x "$PM/pGM/fortran_with_cpp.sh"
echo "$PGM_VERSION $HOOKS $(date +%F)" > "$BACKUP/installed"
info "pGM sources installed; originals saved in $BACKUP"

# --- build ---------------------------------------------------------------------
if [ $DO_BUILD -eq 1 ]; then
  [ -n "$BUILD_DIR" ] || die "--build needs --build-dir (your existing AMBER CMake build directory)"
  [ -f "$BUILD_DIR/CMakeCache.txt" ] || die "$BUILD_DIR is not a configured CMake build directory"
  cd "$BUILD_DIR"
  cmake . > pgm_cmake.log 2>&1 || die "cmake reconfiguration failed; see $BUILD_DIR/pgm_cmake.log"
  TARGETS="pmemd.pgm"
  if grep -q "^CUDA:BOOL=\(TRUE\|ON\)" CMakeCache.txt; then
    TARGETS="$TARGETS pmemd.pgm.cuda_SPFP pmemd.pgm.cuda_DPFP"
  fi
  for t in $TARGETS; do
    info "building $t"
    make -j"$JOBS" "$t" > "pgm_build_$t.log" 2>&1 || die "build of $t failed; see $BUILD_DIR/pgm_build_$t.log"
  done
  cmake --install . --component pmemd_pgm > /dev/null 2>&1 || true
  cmake --install . --component pmemd_pgm_CUDA > /dev/null 2>&1 || true
  info "built: $TARGETS"
else
  info "next: in your AMBER build directory run 'cmake .' and 'make install', or rerun with --build --build-dir DIR"
fi
info "test with: $HERE/test/run_tests.sh"
