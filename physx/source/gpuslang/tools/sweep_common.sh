#!/bin/bash
# Shared mutation-sweep harness (P9.39b lineage: guards G1-G5 + the mtime stamp).
# Callers set SH (target source), SPV (fixture shader output), TEST (suite name),
# TMO (seconds), then source this file and call run_mutant NAME OLD NEW; finish
# with sweep_finish.
#
#  G1 restore from a SNAPSHOT, never `git checkout` (it deletes uncommitted work);
#  G2 a mutant whose build failed or whose .spv did not change is NOT-RUN, not a survivor;
#  G3 every fixture run is wrapped in `timeout` -- a hang is a kill, not a wait;
#  G4 the snapshot must equal git HEAD before the run starts (a killed harness leaves a mutant);
#  G5 an anchor miss ABORTS -- the source is not what the script thinks it is.
set -u
BUILD=/home/user/physx-gpu/build-gpuslang
cd /home/user/physx-gpu
if ! git diff --quiet -- "$SH"; then
  echo "REFUSING TO RUN: $SH differs from git HEAD (a previous harness was probably killed"
  echo "before its restore line ran).  Restore it and re-run."
  exit 2
fi
SNAP=$(mktemp); cp "$SH" "$SNAP"
BASE=$(mktemp); cp "$SPV" "$BASE"
abort () { echo "ABORT: $1"; cp "$SNAP" "$SH"; rm -f "$SNAP" "$BASE"; exit 3; }
export PXG_SHADER_DIR=$BUILD/shaders
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
  out=$( cd "$BUILD" && timeout "$TMO" ./gpuslang-tests "$TEST" 2>&1 )
  local rc=$?
  if [ $rc -eq 124 ]; then
    echo "killed    $name (HANG)"; fail=$((fail+1)); return
  fi
  if [ $rc -ge 128 ]; then
    echo "killed    $name (CRASH rc=$rc)"; fail=$((fail+1)); return
  fi
  if printf '%s' "$out" | grep -q "0 failed, 0 check failure"; then
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
