#!/bin/sh
# run_scene_mutation_sweep.sh -- the forced-fresh mutation sweep for an ENGINE
# kernel whose only observer is a SCENE.
#
# WHY A SECOND RUNNER.  run_mutation_sweep.sh drives `gpuslang-tests <name>`, and
# that is right for every Phase-1 fixture.  But a growing set of shipped shaders
# have NO fixture of their own by design -- cvxCorePlaneNpBda, cvxCoreCvxNpBda,
# cvxCoreTriNpBda, sphereHfNpBda, sphereTriNpBda -- because they are BDA WIRING
# around layers that are each fixture-verified somewhere else.  Their wiring is
# exactly the part no fixture sees, and up to P9.70c it was swept by hand or not
# at all.  "Not at all" is the one that matters: a tranche that ships wiring with
# no discrimination evidence is asserting the wiring works, not showing it.
#
# It keeps every invariant run_mutation_sweep.sh enforces, for the same reasons
# recorded in that file's header:
#   1. the pristine source is snapshotted once;
#   2. source AND .spv are restored after every mutant -- P9.9 was bitten twice by
#      a sweep that left a mutant's .spv on disk;
#   3. a trap does the same on interrupt;
#   4. the BASELINE is proved green before anything is reported, and a mutant that
#      hangs is classified rather than left to wedge the run.
#
# WHAT COUNTS AS A KILL.  gpu-sdk-step prints one verdict line per scene.  A
# MISSING verdict is its own outcome and never a pass -- that is P9.3a's finding,
# where an unknown scene name printed neither marker and an ad-hoc grep read the
# silence as green.
#
# Usage:
#   sh run_scene_mutation_sweep.sh <shader-stem> <scene> <mutants-file> [source-stem]
#
# <source-stem> defaults to <shader-stem>, for shared includes and modules that
# have no .spv of their own (see run_mutation_sweep.sh's note on P9.12a).
#
# <mutants-file> is one mutant per line, three TAB-separated fields:
#   name <TAB> old-text <TAB> new-text
# Use \n for newlines.  A non-unique or missing anchor is ANCHOR-FAIL and counts as
# untested, never as a kill.
#
# Environment: SCENE_SWEEP_TIMEOUT (default 900s -- a scene is minutes, not
# seconds, so this is far above the fixture runner's 300), plus any PXG_* the
# scene itself takes, which are passed through.

set -u

if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "usage: sh run_scene_mutation_sweep.sh <shader-stem> <scene> <mutants-file> [source-stem]" >&2
    exit 2
fi

STEM="$1"; SCENE="$2"; MUTANTS="$3"; SRCSTEM="${4:-$1}"
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
SRC="$ROOT/physx/source/gpuslang/shaders/$SRCSTEM.slang"
BUILD="$ROOT/build-gpuslang"
SDK="$ROOT/build-sdk"
SPV="$BUILD/shaders/$STEM.spv"
BAK="$(mktemp)"

[ -f "$SRC" ] || { echo "no such shader: $SRC" >&2; exit 2; }
[ -d "$BUILD" ] || { echo "no build tree: $BUILD" >&2; exit 2; }
[ -x "$SDK/gpu-sdk-step" ] || { echo "no gpu-sdk-step in $SDK" >&2; exit 2; }
[ -f "$MUTANTS" ] || { echo "no mutants file: $MUTANTS" >&2; exit 2; }

cp "$SRC" "$BAK"

rebuild() {
    rm -f "$SPV"
    ( cd "$ROOT" && cmake --build build-gpuslang -j"$(nproc)" --target gpuslang-shaders ) >/dev/null 2>&1
}
restore() { cp "$BAK" "$SRC"; rebuild; }

trap 'echo; echo "!! interrupted -- restoring pristine source and .spv"; restore; rm -f "$BAK"; exit 130' INT TERM

: "${SCENE_SWEEP_TIMEOUT:=900}"
TIMEOUT_RC=124

run_scene() {
    ( cd "$SDK" && PXG_SHADER_DIR="$BUILD/shaders" PXG_SCENE="$SCENE" PXG_GPU_STEP=1 \
        timeout -k 10 "$SCENE_SWEEP_TIMEOUT" ./gpu-sdk-step 2>&1 )
}

# The three verdict shapes, in the order they must be tested.  "no verdict" is
# NOT a pass (P9.3a).
scene_verdict() {   # stdin: the run's output -> echoes green|red|silent
    if echo "$1" | grep -q "^\[  OK  \] gpu-sdk-step: scene"; then echo green
    elif echo "$1" | grep -q "^\[ FAIL \]"; then echo red
    else echo silent
    fi
}

# The platform faults on its own (P9.28a: llvmpipe places a JIT code object with
# its preceding page unmapped).  A crash that does not reproduce is not evidence,
# so a crash is retried and only an all-attempts-crash is reported as a fault.
: "${SCENE_SWEEP_FAULT_RETRIES:=2}"

run_scene_tolerant() {
    SCENE_TRIES=1
    SCENE_OUT=$(run_scene); SCENE_RC=$?
    while [ $SCENE_RC -ge 128 ] && [ $SCENE_RC -ne $TIMEOUT_RC ] \
          && [ $SCENE_TRIES -le $SCENE_SWEEP_FAULT_RETRIES ]; do
        SCENE_TRIES=$((SCENE_TRIES+1))
        SCENE_OUT=$(run_scene); SCENE_RC=$?
    done
}

restore
run_scene_tolerant
if [ $SCENE_RC -eq $TIMEOUT_RC ]; then
    echo "!! the CLEAN scene exceeded SCENE_SWEEP_TIMEOUT=${SCENE_SWEEP_TIMEOUT}s;" >&2
    echo "   raise it or every mutant will be reported as a hang" >&2
    rm -f "$BAK"; exit 1
fi
if [ "$(scene_verdict "$SCENE_OUT")" != "green" ]; then
    echo "!! BASELINE IS RED (or silent) before the sweep -- refusing to measure mutants against it"
    echo "$SCENE_OUT" | grep -viE "Could not find CUDA module" | tail -8
    rm -f "$BAK"; exit 1
fi
echo "baseline green; sweeping $SRCSTEM (via $STEM.spv) against scene '$SCENE'"

killed=0; faulted=0; hung=0; silent=0; survived=0; nocompile=0; anchorfail=0

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
    sys.stderr.write("anchor count %d\n" % c)
    sys.exit(1)
open(p, 'w').write(s.replace(o, n))
PY
    then
        printf '  %-40s ANCHOR-FAIL\n' "$name"
        anchorfail=$((anchorfail+1))
        continue
    fi

    rebuild
    if [ ! -f "$SPV" ]; then
        printf '  %-40s NOCOMPILE\n' "$name"
        nocompile=$((nocompile+1))
        continue
    fi

    run_scene_tolerant
    if [ $SCENE_RC -eq $TIMEOUT_RC ]; then
        printf '  %-40s KILLED-HANG\n' "$name"
        hung=$((hung+1))
    elif [ $SCENE_RC -ge 128 ]; then
        printf '  %-40s KILLED-FAULT (%d tries)\n' "$name" "$SCENE_TRIES"
        faulted=$((faulted+1))
    else
        case "$(scene_verdict "$SCENE_OUT")" in
            green)  printf '  %-40s SURVIVED\n' "$name"; survived=$((survived+1)) ;;
            red)    printf '  %-40s KILLED\n' "$name"; killed=$((killed+1)) ;;
            silent) printf '  %-40s KILLED-SILENT (no verdict printed)\n' "$name"; silent=$((silent+1)) ;;
        esac
    fi
done < "$MUTANTS"

restore
run_scene_tolerant
if [ "$(scene_verdict "$SCENE_OUT")" != "green" ]; then
    echo "!! the RESTORED tree is not green -- the sweep left something behind" >&2
    rm -f "$BAK"; exit 1
fi

total_killed=$((killed+faulted+hung+silent))
echo
echo "killed $total_killed (fail $killed, fault $faulted, hang $hung, silent $silent)" \
     "| survived $survived | nocompile $nocompile | anchor-fail $anchorfail"
rm -f "$BAK"
[ $survived -eq 0 ]
