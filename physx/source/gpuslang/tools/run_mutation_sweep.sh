#!/bin/sh
# run_mutation_sweep.sh -- the canonical forced-fresh mutation sweep.
#
# WHY THIS EXISTS.  Every tranche sweeps its new shader by hand, and the hand-written
# runners keep making the same mistake: they restore the SOURCE after each mutant but
# never rebuild the SPIR-V, so the last mutant's .spv is left on disk against a clean
# source.  The next test run then measures a mutant and reports it as a regression in
# committed code.
#
#   * P8.83 hit it via SIGPIPE (`| head` killed the sweep mid-mutant) and wrote the
#     rule down;
#   * P9.9 hit it TWICE anyway -- once misdiagnosed as a stale test binary -- because
#     the rule lived in a note and not in a tool.  The signature was 146 deterministic
#     mismatches from `r26-system-from-gridY`, whose .spv was still on disk.
#
# So this runner guarantees the invariant instead of relying on discipline:
#   1. it snapshots the pristine source once;
#   2. it restores source AND rebuilds the .spv after every mutant;
#   3. a `trap` does the same on interrupt, so ^C cannot leave a mutant behind;
#   4. it proves the BASELINE is green before reporting anything, and again at the
#      end -- a sweep whose baseline is red is measuring nothing.
#
# Usage:
#   sh run_mutation_sweep.sh <shader-stem> <test-suite> <mutants-file>
#
# <mutants-file> is one mutant per line, three TAB-separated fields:
#   name <TAB> old-text <TAB> new-text
# Use \n in old/new for newlines.  Anchors must be UNIQUE in the source; a non-unique
# or missing anchor is reported as ANCHOR-FAIL and counted as untested, never as a kill.
#
# Reported per mutant: KILLED (test failed), KILLED-FAULT (test crashed -- also a kill,
# counted separately), SURVIVED, NOCOMPILE (not a kill), ANCHOR-FAIL (not a kill).

set -u

if [ $# -ne 3 ]; then
    echo "usage: sh run_mutation_sweep.sh <shader-stem> <test-suite> <mutants-file>" >&2
    exit 2
fi

STEM="$1"; SUITE="$2"; MUTANTS="$3"
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
SRC="$ROOT/physx/source/gpuslang/shaders/$STEM.slang"
BUILD="$ROOT/build-gpuslang"
SPV="$BUILD/shaders/$STEM.spv"
BAK="$(mktemp)"

[ -f "$SRC" ] || { echo "no such shader: $SRC" >&2; exit 2; }
[ -d "$BUILD" ] || { echo "no build tree: $BUILD" >&2; exit 2; }
[ -f "$MUTANTS" ] || { echo "no mutants file: $MUTANTS" >&2; exit 2; }

cp "$SRC" "$BAK"

rebuild() {
    rm -f "$SPV"
    ( cd "$ROOT" && cmake --build build-gpuslang -j"$(nproc)" --target gpuslang-shaders ) >/dev/null 2>&1
}
restore() { cp "$BAK" "$SRC"; rebuild; }

# The invariant, enforced rather than remembered: however this script exits, the tree
# is left with the pristine source AND a .spv built from it.
trap 'echo; echo "!! interrupted -- restoring pristine source and .spv"; restore; rm -f "$BAK"; exit 130' INT TERM

run_suite() { ( cd "$BUILD" && PXG_SHADER_DIR="$BUILD/shaders" ./gpuslang-tests "$SUITE" 2>&1 ); }

baseline_ok() {
    out=$(run_suite)
    echo "$out" | grep -q "0 failed" && echo "$out" | grep -q "test(s) ran"
}

restore
if ! baseline_ok; then
    echo "!! BASELINE IS RED before the sweep -- refusing to measure mutants against it"
    run_suite | tail -5
    rm -f "$BAK"; exit 1
fi
echo "baseline green; sweeping $STEM against suite '$SUITE'"

killed=0; faulted=0; survived=0; nocompile=0; anchorfail=0

while IFS='	' read -r name old new; do
    [ -z "${name:-}" ] && continue
    case "$name" in \#*) continue ;; esac

    cp "$BAK" "$SRC"
    if ! python3 - "$SRC" "$old" "$new" <<'PY'
import sys
p, o, n = sys.argv[1], sys.argv[2].replace('\\n', '\n'), sys.argv[3].replace('\\n', '\n')
s = open(p).read()
c = s.count(o)
if c != 1:
    sys.stderr.write("anchor count %d\n" % c); sys.exit(1)
open(p, 'w').write(s.replace(o, n))
PY
    then
        echo "ANCHOR-FAIL   $name"; anchorfail=$((anchorfail+1)); continue
    fi

    rebuild
    if [ ! -f "$SPV" ]; then
        echo "NOCOMPILE     $name"; nocompile=$((nocompile+1)); continue
    fi

    out=$(run_suite); rc=$?
    if [ $rc -ge 128 ] || ! echo "$out" | grep -q "test(s) ran"; then
        echo "KILLED-FAULT  $name   (crashed: rc=$rc)"; faulted=$((faulted+1))
    elif echo "$out" | grep -q "0 failed"; then
        echo "SURVIVED      $name"; survived=$((survived+1))
    else
        echo "KILLED        $name"; killed=$((killed+1))
    fi
done < "$MUTANTS"

restore
total=$((killed+faulted+survived+nocompile+anchorfail))
echo "=== sweep: $total mutants -- $killed killed, $faulted killed-by-fault, $survived SURVIVED, $nocompile nocompile, $anchorfail anchor-fail ==="

if ! baseline_ok; then
    echo "!! BASELINE IS RED after the restore -- the tree is NOT clean, do not trust the run"
    rm -f "$BAK"; exit 1
fi
echo "baseline green again; source and .spv restored"
rm -f "$BAK"

# A survivor is not a result until it is diagnosed.  The caller must prove each one
# equivalent or fix the fixture; this script deliberately exits non-zero so a survivor
# cannot be scrolled past.
[ "$survived" -eq 0 ]
