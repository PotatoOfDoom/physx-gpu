#!/bin/bash
# Shared mutation-sweep harness (P9.39b lineage: guards G1-G5 + the mtime stamp).
# Callers set SH (target source), SPV (fixture shader output), TEST (suite name),
# optionally TMO (seconds; FIXES the per-run bound, as MUTATION_SWEEP_TIMEOUT does),
# then source this file and call run_mutant NAME OLD NEW; finish with sweep_finish.
#
#  G1 restore from a SNAPSHOT, never `git checkout` (it deletes uncommitted work);
#  G2 a mutant whose build failed or whose .spv did not change is NOT-RUN, not a survivor;
#  G3 every fixture run is wrapped in `timeout` -- a hang is a kill, not a wait;
#  G4 the snapshot must equal git HEAD before the run starts (a killed harness leaves a mutant);
#  G5 an anchor miss ABORTS -- the source is not what the script thinks it is.
#  G7 (P9.97) the CLEAN fixture runs once, at source time, before any mutant: it must
#     be green and must finish inside the baseline bound, and its time sets the
#     per-run bound (tools/sweep_timeout.sh).  Up to P9.96 there was no baseline here
#     at all, so a TMO below the clean run's time read EVERY mutant as "killed (HANG)",
#     and a caller that forgot TMO read every mutant as "killed" (set -u fails inside
#     the $( ) and the empty output is not green) -- false kills, both.
set -u
# The repo root is derived from this file's own location (P9.97: the hermetic
# gpuslang.sweepRunners test sources a COPY of this file inside a mirror tree);
# SWEEP_REPO overrides it.
SWEEP_REPO=${SWEEP_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
BUILD=$SWEEP_REPO/build-gpuslang
cd "$SWEEP_REPO"
if ! git diff --quiet -- "$SH"; then
  echo "REFUSING TO RUN: $SH differs from git HEAD (a previous harness was probably killed"
  echo "before its restore line ran).  Restore it and re-run."
  exit 2
fi
SNAP=$(mktemp); cp "$SH" "$SNAP"
BASE=$(mktemp); cp "$SPV" "$BASE"
abort () { echo "ABORT: $1"; cp "$SNAP" "$SH"; rm -f "$SNAP" "$BASE"; exit 3; }
export PXG_SHADER_DIR=$BUILD/shaders
. "$SWEEP_REPO/physx/source/gpuslang/tools/sweep_timeout.sh"
[ -n "${TMO:-}" ] && MUTATION_SWEEP_TIMEOUT=$TMO
sweep_init MUTATION_SWEEP 300 || { rm -f "$SNAP" "$BASE"; exit 2; }
# A baseline CRASH (rc >= 128 that is not the bound expiring -- sweep_run_bounded already
# maps a SIGKILL-at-bound 137 to 124) is retried, as run_mutation_sweep.sh does: llvmpipe
# faults on its own (P9.29), and one platform crash must not refuse the whole sweep.  A
# hang is not retried -- re-running it only costs another bound.
: "${MUTATION_SWEEP_FAULT_RETRIES:=3}"
_btries=0
while :; do
  _btries=$((_btries+1))
  _t0=$(date +%s)
  _bout=$( cd "$BUILD" && sweep_run_bounded ./gpuslang-tests "$TEST" 2>&1 ); _brc=$?
  _bsecs=$(( $(date +%s) - _t0 ))
  [ $_brc -ge 128 ] && [ $_brc -ne 124 ] && [ $_btries -le $MUTATION_SWEEP_FAULT_RETRIES ] || break
done
if [ $_brc -eq 124 ]; then
  echo "REFUSING TO RUN: the CLEAN $TEST exceeded ${SWEEP_TMO_KNOB}=${SWEEP_TMO}s -- every mutant would read as a hang"
  rm -f "$SNAP" "$BASE"; exit 3
fi
# Green = "N test(s) ran" with no NON-ZERO failure count, and not "0 test(s) ran".  (A
# bare grep for "0 failed, 0 check failure" also matches "10 failed, 0 check
# failure(s)" -- the false green run_mutation_sweep.sh's all_green was written against.)
sc_green() { printf '%s' "$1" | grep -q "test(s) ran" && ! printf '%s' "$1" | grep -Eq ", [1-9][0-9]* failed" \
             && ! printf '%s' "$1" | grep -q "^0 test(s) ran"; }
if ! sc_green "$_bout"; then
  echo "REFUSING TO RUN: the CLEAN $TEST is not green (rc=$_brc, $_btries run(s)) or matched no test:"
  printf '%s\n' "$_bout" | tail -5
  rm -f "$SNAP" "$BASE"; exit 3
fi
sweep_set_bound "$_bsecs"
echo "baseline green in ${SWEEP_BASE_SECS}s; every run below is bounded by ${SWEEP_TMO}s ($SWEEP_TMO_HOW)"
pass=0; fail=0; notrun=0; survivors=(); skipped=()
run_mutant () {
  local name="$1" old="$2" new="$3"
  cp "$SNAP" "$SH"
  python3 - "$SH" "$old" "$new" <<'PY' || abort "anchor for $name not found exactly once -- the source is not what this script expects"
import sys
p,o,n=sys.argv[1],sys.argv[2],sys.argv[3]
s=open(p).read()
if s.count(o)!=1: sys.exit(1)
open(p,"w").write(s.replace(o,n,1))
PY
  touch -d "+1 second" "$SH"
  if ! ( cd "$BUILD" && cmake --build . -j"$(nproc)" --target gpuslang-shaders >/dev/null 2>&1 ); then
    echo "NOT-RUN   $name (build failed)"; notrun=$((notrun+1)); skipped+=("$name (build)"); return
  fi
  if cmp -s "$SPV" "$BASE"; then
    # G6 (P9.48a): a "no change" verdict is retried with the .spv DELETED first.
    # Twice now a mutant that demonstrably changes behaviour was reported inert
    # because make skipped the rebuild -- the source stamp is a second in the
    # future and the previous mutant's build landed inside that second.  Both
    # were hand-verified as killed afterwards, which is exactly the kind of
    # silent under-reporting a sweep must not do.
    rm -f "$SPV"
    if ! ( cd "$BUILD" && cmake --build . -j"$(nproc)" --target gpuslang-shaders >/dev/null 2>&1 ); then
      echo "NOT-RUN   $name (build failed on retry)"; notrun=$((notrun+1)); skipped+=("$name (build)"); return
    fi
    if cmp -s "$SPV" "$BASE"; then
      echo "NOT-RUN   $name (.spv unchanged after a forced rebuild)"
      notrun=$((notrun+1)); skipped+=("$name (inert build)"); return
    fi
  fi
  local out
  out=$( cd "$BUILD" && sweep_run_bounded ./gpuslang-tests "$TEST" 2>&1 )
  local rc=$?
  if [ $rc -eq 124 ]; then
    echo "killed    $name (HANG > ${SWEEP_TMO}s; baseline ${SWEEP_BASE_SECS}s)"; fail=$((fail+1)); return
  fi
  if [ $rc -ge 128 ]; then
    echo "killed    $name (CRASH rc=$rc)"; fail=$((fail+1)); return
  fi
  if sc_green "$out"; then
    echo "SURVIVED  $name"; survivors+=("$name"); pass=$((pass+1))
  else
    echo "killed    $name"; fail=$((fail+1))
  fi
}
sweep_finish () {
  cp "$SNAP" "$SH"; rm -f "$SNAP" "$BASE"
  touch -d "+1 second" "$SH"
  ( cd "$BUILD" && cmake --build . -j"$(nproc)" --target gpuslang-shaders >/dev/null 2>&1 )
  echo
  echo "=== $TEST: killed $fail, survived $pass, not-run $notrun"
  for s in "${survivors[@]:-}"; do [ -n "$s" ] && echo "    SURVIVOR: $s"; done
  for s in "${skipped[@]:-}"; do [ -n "$s" ] && echo "    NOT-RUN:  $s"; done
}
