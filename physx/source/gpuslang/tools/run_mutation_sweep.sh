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
# (test exceeded the per-run bound, see below) -- all three are kills, counted
# separately -- plus SURVIVED, NOCOMPILE (not a kill) and ANCHOR-FAIL (not a kill).
#
# MUTATION_SWEEP_SPV_ONLY=1 (added in P9.83b-8): rebuild ONLY <shader-stem>.spv per
# mutant, through the shader target's own per-output make rule, instead of the whole
# gpuslang-shaders target.  WHY: every shader depends on every pxgSlang*.slang module
# (the SHADER_COMMON_DEPS glob), so a mutant in a shared module rebuilt all ~460 shaders
# -- 3-4 minutes per mutant, a 44-mutant sweep past two hours, and one container restart
# in that window left a mutant in the tree (P9.83b-8 (2/n)).  The single rule takes
# about half a second and produced a bit-identical .spv when measured.  It is SOUND only
# when the suite under test loads no other importer of the mutated module -- true for a
# fixture suite over its own .spv -- because the other importers are never rebuilt
# against the mutant.  restore() still does the FULL build, so the tree is left
# consistent either way.  Unix Makefiles only; with another generator the full build is
# used and a note says so.
#
# THE PER-RUN BOUND (P9.97; policy and reasons in sweep_timeout.sh).  Unless
# MUTATION_SWEEP_TIMEOUT is set, the baseline is TIMED (bounded only by
# MUTATION_SWEEP_BASELINE_TIMEOUT, default 3600 s) and every later run is bounded by
#     max(MUTATION_SWEEP_TIMEOUT_FLOOR (300), MUTATION_SWEEP_TIMEOUT_FACTOR (3) x baseline)
# -- 300 s, as before, for any suite up to 100 s; 1305 s for the 435 s triTriCollisionBda
# suite that the old fixed 300 refused.  MUTATION_SWEEP_TIMEOUT=<s> still fixes the bound
# for every run, the baseline included, and a clean suite that does not fit is refused
# up front (every mutant would otherwise read as a hang); one that uses more than half
# of it is warned about.  The bound in force is printed after the baseline.

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
. "$ROOT/physx/source/gpuslang/tools/sweep_timeout.sh"
sweep_init MUTATION_SWEEP 300 || { rm -f "$BAK"; exit 2; }

cp "$SRC" "$BAK"

rebuild_full() {
    rm -f "$SPV"
    ( cd "$ROOT" && cmake --build build-gpuslang -j"$(nproc)" --target gpuslang-shaders ) >/dev/null 2>&1
}
SHADER_MAKEFILE="$BUILD/CMakeFiles/gpuslang-shaders.dir/build.make"
if [ "${MUTATION_SWEEP_SPV_ONLY:-0}" = 1 ] && [ ! -f "$SHADER_MAKEFILE" ]; then
    echo "note: MUTATION_SWEEP_SPV_ONLY ignored -- no $SHADER_MAKEFILE (not Unix Makefiles?)" >&2
    MUTATION_SWEEP_SPV_ONLY=0
fi
rebuild() {
    if [ "${MUTATION_SWEEP_SPV_ONLY:-0}" = 1 ]; then
        rm -f "$SPV"
        ( cd "$BUILD" && make -s -f "$SHADER_MAKEFILE" "shaders/$STEM.spv" ) >/dev/null 2>&1
    else
        rebuild_full
    fi
}
restore() { cp "$BAK" "$SRC"; rebuild_full; }

# The invariant, enforced rather than remembered: however this script exits, the tree
# is left with the pristine source AND a .spv built from it.
trap 'echo; echo "!! interrupted -- restoring pristine source and .spv"; restore; rm -f "$BAK"; exit 130' INT TERM

# Seconds any single suite run may take: SWEEP_TMO, owned by sweep_timeout.sh -- the
# baseline bound until the baseline is green, the derived (or fixed) bound after it.
# A backstop against a mutant that never terminates, not a performance budget.
TIMEOUT_RC=124   # what sweep_run_bounded returns whenever the bound expired

# <test-suite> may be a COMMA-SEPARATED list of bare PXG_TEST filters (P9.86c): a shader
# driven by several fixtures is measured against all of them, and a mutant dies if any
# one fails.  The rc is the worst of the runs (a timeout or crash wins over a failure).
# A TIMEOUT IS STICKY (P9.97): a later filter's crash must not overwrite an earlier
# filter's hang -- the crash rc would send the mutant round the fault retries, and every
# retry re-runs the hang for one more whole bound, only to report KILLED-FAULT.
run_suite() {
    worst=0
    for s in $(echo "$SUITE" | tr ',' ' '); do
        ( cd "$BUILD" && export PXG_SHADER_DIR="$BUILD/shaders" \
            && sweep_run_bounded ./gpuslang-tests "$s" 2>&1 )
        r=$?
        if [ $r -eq $TIMEOUT_RC ]; then worst=$TIMEOUT_RC
        elif [ $r -ge 128 ] && [ $worst -ne $TIMEOUT_RC ]; then worst=$r
        elif [ $r -ne 0 ] && [ $worst -eq 0 ]; then worst=$r; fi
    done
    return $worst
}
# "N test(s) ran, M failed": green only if no run reports a NON-ZERO M.  (The old
# `grep "0 failed"` also matched "10 failed" -- a false survivor, never a false kill.)
all_green() { echo "$1" | grep -q "test(s) ran" && ! echo "$1" | grep -Eq ", [1-9][0-9]* failed"; }

# A CRASH THAT DOES NOT REPRODUCE IS NOT EVIDENCE.  Added in P9.29.
#
# This platform faults on its own: llvmpipe sometimes places a JIT code object
# with its preceding page unmapped and the code reads into it (P9.28a has the
# core dumps -- fault at object_base - 0x1000, rip at object_base + 0x767, twice,
# to the byte).  The rate depends on how many shaders a suite dispatches; on
# femClothSim it is around one run in ten.
#
# A sweep that reads one crash as KILLED-FAULT therefore credits itself with
# roughly three kills per thirty mutants that it did not earn -- and those false
# kills land on whichever mutants happen to run when the allocator misbehaves,
# so they are invisible in the summary and unrepeatable between runs.  That is
# worse than a missed kill: it reports discrimination the fixtures do not have.
#
# So a crash is retried.  If ANY attempt completes, that attempt is the result --
# a mutant that really breaks the shader breaks it every time, so a run that
# finishes is proof the crash was the platform.  Only an all-attempts-crash is
# reported as KILLED-FAULT, and the count is printed so the tolerance is visible
# rather than silent.
: "${MUTATION_SWEEP_FAULT_RETRIES:=3}"

# Sets SUITE_OUT / SUITE_RC / SUITE_TRIES / SUITE_SECS (wall seconds of the attempt
# that is the result -- for the baseline, the number the bound is derived from).  A
# timeout is NOT retried: a hang is a behaviour change, and re-running it only costs
# another bound.
run_suite_timed() {
    _t0=$(date +%s); SUITE_OUT=$(run_suite); SUITE_RC=$?; SUITE_SECS=$(( $(date +%s) - _t0 ))
}
run_suite_tolerant() {
    SUITE_TRIES=1
    run_suite_timed
    while [ $SUITE_RC -ge 128 ] && [ $SUITE_RC -ne $TIMEOUT_RC ] \
          && [ $SUITE_TRIES -le $MUTATION_SWEEP_FAULT_RETRIES ]; do
        SUITE_TRIES=$((SUITE_TRIES+1))
        run_suite_timed
    done
}

# A run that executed NOTHING.  The binary prints "0 test(s) ran, 0 failed" for a
# suite filter that matches no test -- which satisfies both a "0 failed" grep and a
# "test(s) ran" grep.  P9.13f hit exactly that: the suite was passed as
# "gpuslang.<name>" (the CTEST name) where the binary wants the bare PXG_TEST name,
# and the sweep reported a green baseline and 28 SURVIVED without ever running the
# shader.  A sweep that measures nothing must fail loudly, not report survivors.
ran_nothing() { echo "$1" | grep -q "^0 test(s) ran"; }

baseline_ok() {
    run_suite_tolerant; out="$SUITE_OUT"; brc=$SUITE_RC
    if [ $brc -ge 128 ] && [ $brc -ne $TIMEOUT_RC ]; then
        echo "!! the CLEAN suite crashed $SUITE_TRIES time(s) in a row (rc=$brc)." >&2
        echo "   That is past what the platform fault explains -- fix it before" >&2
        echo "   measuring mutants against it." >&2
        return 1
    fi
    if [ $brc -eq $TIMEOUT_RC ]; then
        echo "!! the CLEAN suite exceeded ${SWEEP_TMO_KNOB}=${SWEEP_TMO}s${SWEEP_BASE_SECS:+ (the first baseline took ${SWEEP_BASE_SECS}s)};" >&2
        # The advice is for the FIRST baseline only: after it the bound is the derived
        # (or fixed) one, which is no knob of its own, and the closing baseline prints
        # its own message.
        if [ -z "$SWEEP_BASE_SECS" ]; then
            if [ -n "$SWEEP_FIXED" ]; then
                echo "   raise it (or unset it to derive the bound) -- every mutant would be reported as a hang" >&2
            else
                echo "   raise it -- it bounds the clean run itself, not the mutants" >&2
            fi
        fi
        return 1
    fi
    if ran_nothing "$out"; then
        echo "!! the suite filter '$SUITE' matched NO test -- the binary takes the bare" >&2
        echo "   PXG_TEST name, not the ctest name (no 'gpuslang.' prefix)." >&2
        return 1
    fi
    all_green "$out"
}

restore
if ! baseline_ok; then
    echo "!! BASELINE IS NOT GREEN before the sweep -- refusing to measure mutants against it"
    # The run just made, not a second one: up to P9.96 this line re-ran the suite, and
    # after a timeout that cost one more whole bound to print a truncated tail.
    printf '%s\n' "$SUITE_OUT" | tail -5
    rm -f "$BAK"; exit 1
fi
sweep_set_bound "$SUITE_SECS"
echo "baseline green in ${SWEEP_BASE_SECS}s; every run below is bounded by ${SWEEP_TMO}s ($SWEEP_TMO_HOW)"
echo "baseline green; sweeping $SRCSTEM (via $STEM.spv) against suite '$SUITE'"

killed=0; faulted=0; hung=0; survived=0; nocompile=0; anchorfail=0; retried=0

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

    run_suite_tolerant; out="$SUITE_OUT"; rc=$SUITE_RC
    [ "$SUITE_TRIES" -gt 1 ] && retried=$((retried+1))
    if [ $rc -eq $TIMEOUT_RC ]; then
        echo "KILLED-HANG   $name   (exceeded ${SWEEP_TMO}s; baseline ${SWEEP_BASE_SECS}s)"; hung=$((hung+1))
    elif [ $rc -ge 128 ] || ! echo "$out" | grep -q "test(s) ran"; then
        echo "KILLED-FAULT  $name   (crashed $SUITE_TRIES/$((MUTATION_SWEEP_FAULT_RETRIES+1)) times: rc=$rc)"
        faulted=$((faulted+1))
    elif ran_nothing "$out"; then
        # Not a survivor: the suite stopped matching, so nothing was measured.
        echo "!! '$name' left the suite matching NO test -- aborting rather than" >&2
        echo "   counting an unmeasured mutant as a survivor" >&2
        restore; rm -f "$BAK"; exit 1
    elif all_green "$out"; then
        echo "SURVIVED      $name"; survived=$((survived+1))
    else
        echo "KILLED        $name"; killed=$((killed+1))
    fi
done < "$MUTANTS"

restore
total=$((killed+faulted+hung+survived+nocompile+anchorfail))
echo "=== sweep: $total mutants -- $killed killed, $faulted killed-by-fault, $hung killed-by-hang, $survived SURVIVED, $nocompile nocompile, $anchorfail anchor-fail ==="
[ "$retried" -gt 0 ] && echo "    ($retried mutant(s) hit a platform fault and were re-run; see the note at run_suite_tolerant)"

if ! baseline_ok; then
    if [ "$SUITE_RC" -eq $TIMEOUT_RC ]; then
        echo "!! the RESTORED suite did not finish inside ${SWEEP_TMO}s -- slower machine or unclean tree;"
        echo "   run the suite by hand before trusting this sweep"
    else
        echo "!! BASELINE IS RED after the restore -- the tree is NOT clean, do not trust the run"
    fi
    rm -f "$BAK"; exit 1
fi
echo "baseline green again in ${SUITE_SECS}s; source and .spv restored"
rm -f "$BAK"

# A survivor is not a result until it is diagnosed.  The caller must prove each one
# equivalent or fix the fixture; this script deliberately exits non-zero so a survivor
# cannot be scrolled past.
[ "$survived" -eq 0 ]
