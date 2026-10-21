#!/bin/bash
# ============================================================================
# run_suite.sh — the per-tranche verification run, with the evidence preserved.
#
# WHY THIS EXISTS.  The tranche recipe ends with `ctest -j4` in BOTH memory modes.
# Five times across the port that run has failed EXACTLY ONCE, always in the first
# parallel run after a forced-fresh .spv rebuild, and passed on every run after.
# Each time the failure was un-diagnosable for the same mundane reason: ctest
# overwrites Testing/Temporary/LastTest.log on the next invocation, and the next
# invocation was always the re-run that confirmed it green.  The evidence was
# destroyed by the act of checking.
#
# Ten deliberate reproduction cycles did NOT fire it (see RUNTIME_STATUS, P8.13),
# so it cannot be chased by bisection.  This script instead makes the next natural
# occurrence self-documenting: on any failure it archives the log, the ctest
# output and the environment under a timestamped directory before anything can
# overwrite them, and prints the failing test names.
#
# USAGE:   tools/run_suite.sh [build-dir]      (default: build-gpuslang)
# ARCHIVE: <build-dir>/suite-failures/<UTC timestamp>-<mode>/
# ============================================================================
set -u

BUILD_DIR="${1:-build-gpuslang}"
if [ ! -f "$BUILD_DIR/CTestTestfile.cmake" ]; then
    echo "run_suite.sh: '$BUILD_DIR' is not a configured ctest build directory" >&2
    exit 2
fi

ARCHIVE_ROOT="$BUILD_DIR/suite-failures"
FAILURES=0

# PXG_SHADER_DIR, or two tests fail for a reason that has nothing to do with the
# tranche (P9.49).
#
# `gpuslang.helperUniformity` and its siblings read the COMPILED .spv files, and
# they find them through PXG_SHADER_DIR.  ctest does not inherit an export the
# caller forgot, so the whole suite hinges on whether whoever typed the command
# had exported it in that shell -- which is invisible in the log afterwards.  It
# has now cost two investigations in one session: a red suite that was really a
# missing variable, twice, both times mid-tranche where a red suite is the one
# signal that must be trustworthy.
#
# A runner that only works when its caller remembers something is not a runner,
# so it is set here, derived from the build directory that was actually passed.
# An explicit value from the environment still wins -- someone testing a
# different shader set must be able to say so.
if [ -z "${PXG_SHADER_DIR:-}" ]; then
    PXG_SHADER_DIR="$(cd "$BUILD_DIR" && pwd)/shaders"
fi
if [ ! -d "$PXG_SHADER_DIR" ]; then
    echo "run_suite.sh: PXG_SHADER_DIR='$PXG_SHADER_DIR' is not a directory;" >&2
    echo "  the .spv-reading checks would fail for that reason alone." >&2
    exit 2
fi
export PXG_SHADER_DIR
echo "run_suite.sh: PXG_SHADER_DIR=$PXG_SHADER_DIR"

run_mode() {
    local mode_name="$1"; shift
    echo "=== $mode_name ==="

    local out
    out="$(mktemp)"
    # The env arguments are per-invocation.  The device-local run REMOVES an inherited
    # PXG_DEVICE_V2 (env -u), so a caller that exported =0 still gets a real default
    # run, and it exercises the default exactly as a user gets it (unset).  The
    # host-visible run sets PXG_DEVICE_V2=0 explicitly: since the default flip, "0" is
    # the only value that selects it (include/PxgSlangEnv.h: unset/""/"1" device-local).
    ( cd "$BUILD_DIR" && env "$@" ctest -j4 ) > "$out" 2>&1
    local rc=$?

    tail -3 "$out"

    if [ $rc -ne 0 ]; then
        FAILURES=$((FAILURES + 1))
        local stamp
        stamp="$(date -u +%Y%m%dT%H%M%SZ)"
        local dir="$ARCHIVE_ROOT/$stamp-$mode_name"
        mkdir -p "$dir"
        # LastTest.log FIRST: it is the one the next ctest invocation destroys.
        cp "$BUILD_DIR/Testing/Temporary/LastTest.log" "$dir/LastTest.log" 2>/dev/null
        cp "$out" "$dir/ctest-stdout.txt"
        {
            echo "mode: $mode_name"
            echo "env:  $*"
            echo "date: $(date -u)"
            echo "host: $(uname -a)"
            echo "load: $(cat /proc/loadavg 2>/dev/null)"
            echo "disk: $(df -h . | tail -1)"
            echo "mem:  $(free -m 2>/dev/null | sed -n 2p)"
        } > "$dir/context.txt"
        echo "run_suite.sh: FAILURE ARCHIVED -> $dir"
        echo "  failing tests:"
        grep -E "^[[:space:]]*[0-9]+ - " "$out" | sed 's/^/    /' | head -20
    fi

    rm -f "$out"
}

run_mode device-local -u PXG_DEVICE_V2
run_mode host-visible PXG_DEVICE_V2=0

if [ $FAILURES -ne 0 ]; then
    echo "run_suite.sh: $FAILURES of 2 modes failed; evidence under $ARCHIVE_ROOT"
    exit 1
fi
echo "run_suite.sh: both memory modes green"
