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
# A SECOND way a sweep stops measuring, added in P9.10: a mutant can HANG rather than
# fail or crash.  `t3-child-shift-12` on the particle mesh midphase makes a BV32 child
# offset point back at an ancestor, so the traversal becomes cyclic and the test spins
# forever -- it sat there 37 minutes before anyone looked.  A hang is a kill (the
# mutant plainly changed behaviour) but it has to be CLASSIFIED, not allowed to wedge
# the run, so every suite invocation is now bounded by MUTATION_SWEEP_TIMEOUT seconds
# and a mutant that reaches it is reported as KILLED-HANG.
#
# So this runner guarantees the invariant instead of relying on discipline:
#   1. it snapshots the pristine source once;
#   2. it restores source AND rebuilds the .spv after every mutant;
#   3. a `trap` does the same on interrupt, so ^C cannot leave a mutant behind;
#   4. it proves the BASELINE is green before reporting anything, and again at the
#      end -- a sweep whose baseline is red is measuring nothing.
#
# Usage:
#   sh run_mutation_sweep.sh <shader-stem> <test-suite> <mutants-file> [source-stem]
#
# <source-stem> defaults to <shader-stem> and exists for SHARED MODULES: a file
# like pxgSlangTex3d.slang has no entry point and therefore no .spv of its own,
# so the thing to mutate and the thing to check for a successful rebuild are two
# different files.  Pass the module as <source-stem> and one of its importers as
# <shader-stem>.  This was added in P9.12a; before it, sweeping a shared module
# meant duplicating its code into the importer, which is exactly the duplication
# this port exists to avoid.
#
# <mutants-file> is one mutant per line, three TAB-separated fields:
#   name <TAB> old-text <TAB> new-text
# Use \n in old/new for newlines.  Anchors must be UNIQUE in the source; a non-unique
# or missing anchor is reported as ANCHOR-FAIL and counted as untested, never as a kill.
#
# Reported per mutant: KILLED (test failed), KILLED-FAULT (test crashed), KILLED-HANG
# (test exceeded MUTATION_SWEEP_TIMEOUT, default 300s) -- all three are kills, counted
# separately -- plus SURVIVED, NOCOMPILE (not a kill) and ANCHOR-FAIL (not a kill).
#
# The BASELINE run is bounded by the same timeout: if the clean suite cannot finish
# inside it, every mutant would be reported as a hang, so that is refused up front.

set -u

if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "usage: sh run_mutation_sweep.sh <shader-stem> <test-suite> <mutants-file> [source-stem]" >&2
    exit 2
fi

STEM="$1"; SUITE="$2"; MUTANTS="$3"; SRCSTEM="${4:-$1}"
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
SRC="$ROOT/physx/source/gpuslang/shaders/$SRCSTEM.slang"
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

# Seconds any single suite run may take.  Generous by default -- it is a backstop
# against a mutant that never terminates, not a performance budget.
: "${MUTATION_SWEEP_TIMEOUT:=300}"
TIMEOUT_RC=124   # what GNU coreutils `timeout` exits with when it fires

run_suite() {
    ( cd "$BUILD" && PXG_SHADER_DIR="$BUILD/shaders" \
        timeout -k 10 "$MUTATION_SWEEP_TIMEOUT" ./gpuslang-tests "$SUITE" 2>&1 )
}

# A run that executed NOTHING.  The binary prints "0 test(s) ran, 0 failed" for a
# suite filter that matches no test -- which satisfies both a "0 failed" grep and a
# "test(s) ran" grep.  P9.13f hit exactly that: the suite was passed as
# "gpuslang.<name>" (the CTEST name) where the binary wants the bare PXG_TEST name,
# and the sweep reported a green baseline and 28 SURVIVED without ever running the
# shader.  A sweep that measures nothing must fail loudly, not report survivors.
ran_nothing() { echo "$1" | grep -q "^0 test(s) ran"; }

baseline_ok() {
    out=$(run_suite); brc=$?
    if [ $brc -eq $TIMEOUT_RC ]; then
        echo "!! the CLEAN suite exceeded MUTATION_SWEEP_TIMEOUT=${MUTATION_SWEEP_TIMEOUT}s;" >&2
        echo "   raise it or every mutant will be reported as a hang" >&2
        return 1
    fi
    if ran_nothing "$out"; then
        echo "!! the suite filter '$SUITE' matched NO test -- the binary takes the bare" >&2
        echo "   PXG_TEST name, not the ctest name (no 'gpuslang.' prefix)." >&2
        return 1
    fi
    echo "$out" | grep -q "0 failed" && echo "$out" | grep -q "test(s) ran"
}

restore
if ! baseline_ok; then
    echo "!! BASELINE IS RED before the sweep -- refusing to measure mutants against it"
    run_suite | tail -5
    rm -f "$BAK"; exit 1
fi
echo "baseline green; sweeping $SRCSTEM (via $STEM.spv) against suite '$SUITE'"

killed=0; faulted=0; hung=0; survived=0; nocompile=0; anchorfail=0

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
    if [ $rc -eq $TIMEOUT_RC ]; then
        echo "KILLED-HANG   $name   (exceeded ${MUTATION_SWEEP_TIMEOUT}s)"; hung=$((hung+1))
    elif [ $rc -ge 128 ] || ! echo "$out" | grep -q "test(s) ran"; then
        echo "KILLED-FAULT  $name   (crashed: rc=$rc)"; faulted=$((faulted+1))
    elif ran_nothing "$out"; then
        # Not a survivor: the suite stopped matching, so nothing was measured.
        echo "!! '$name' left the suite matching NO test -- aborting rather than" >&2
        echo "   counting an unmeasured mutant as a survivor" >&2
        restore; rm -f "$BAK"; exit 1
    elif echo "$out" | grep -q "0 failed"; then
        echo "SURVIVED      $name"; survived=$((survived+1))
    else
        echo "KILLED        $name"; killed=$((killed+1))
    fi
done < "$MUTANTS"

restore
total=$((killed+faulted+hung+survived+nocompile+anchorfail))
echo "=== sweep: $total mutants -- $killed killed, $faulted killed-by-fault, $hung killed-by-hang, $survived SURVIVED, $nocompile nocompile, $anchorfail anchor-fail ==="

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
