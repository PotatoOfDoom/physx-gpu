#!/bin/sh
# run_validation.sh -- P9.98.  Runs the gpuslang suite and/or gpu-sdk-step scenes under
# the Khronos validation layer with the project's configuration, and FAILS on any
# validation message tools/validation_waivers.txt does not cover.
#
# Usage:
#   sh run_validation.sh [--best-practices] [--out DIR] <what> [<what> ...]
#     suite[:0|:1]       full ctest (-j4) in build-gpuslang, memory mode 0 / 1 (default both)
#     subset[:0|:1]      gpuslang-tests on a small, fast set of suites (default both modes)
#     scene:<label>|<env...>   one gpu-sdk-step row in build-sdk, e.g.
#                        'scene:box|PXG_SCENE=box'  'scene:sb|PXG_SCENE=softbody PXG_SB_FLOOR=mesh'
#                        (prefix the label with v1/ for PXG_DEVICE_V2=1: 'scene:v1/box|PXG_SCENE=box')
#   --best-practices     use validation/vk_layer_settings_bp.txt (perf advice; F5/F6 waived,
#                        and the debug_utils special-use warning PXG_VK_DEBUG_NAMES causes)
#   --out DIR            where the logs go (default: a fresh mktemp -d, removed on a PASS)
#
# The layer: PXG_VVL_DIR = the root of an extracted vulkan-validationlayers package
# (it holds usr/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json and
# usr/lib/<triplet>/libVkLayer_khronos_validation.so), e.g.
#     apt-get download vulkan-validationlayers && dpkg-deb -x vulkan-validationlayers_*.deb vvl
# Without PXG_VVL_DIR, VK_LAYER_PATH and VK_ADD_LAYER_PATH (':'-separated) and the system
# layer directories are searched.  No layer -> exit 77 ("skipped", the code ctest's
# SKIP_RETURN_CODE maps), never a silent pass.  A PXG_VVL_DIR that is set but holds no
# layer is a usage error (exit 2), not a skip: the caller asked for a layer.  The P9.98
# baseline was measured with VVL 1.3.275.
#
# Memory modes: mode 0 runs with PXG_DEVICE_V2 UNSET (an inherited value is removed), mode
# 1 with PXG_DEVICE_V2=1.  "Set to 0" is not mode 0: the engine bridge
# (bridge/PhysXGpuVulkan.cpp) takes ANY value as v2, as tools/run_suite.sh notes.
#
# What it sets: VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation, VK_LAYER_SETTINGS_PATH to
# the project config (core + sync + thread safety, no message cap), PXG_VK_DEBUG_NAMES=1 so
# every message names its entry point, PXG_VK_ALLOC_STATS=1 so every process reports its
# peak live VkDeviceMemory count (the F5 waiver's bound), and PXG_SHADER_DIR (derived as
# run_scenes.sh does).  It first PROVES the layer loads (VK_LOADER_DEBUG=layer on one
# fixture), because a layer that silently fails to load reports nothing -- which would
# read as a pass.
#
# What it REMOVES (check round 1): every inherited VK_LAYER_* and VK_KHRONOS_VALIDATION_*
# variable except the loader's search paths VK_LAYER_PATH / VK_ADD_LAYER_PATH.  The layer
# reads its settings from the environment too, and an environment setting OVERRIDES
# vk_layer_settings.txt: measured with VVL 1.3.275, an inherited
# VK_LAYER_MESSAGE_ID_FILTER (or VK_KHRONOS_VALIDATION_MESSAGE_ID_FILTER) silenced the
# messages and the run still printed PASS.  The removed names are listed.
#
# Besides the layer's messages, the triage (tools/vvl_triage.py) FAILS on the port's own
# signals: a device / shim refusal (a pipeline, launch, transfer or allocation that never
# happened -- outside the fixtures that provoke them on purpose), a push range widened for
# an entry validation_waivers.txt does not list, and a process whose peak allocation count
# exceeds the device's maxMemoryAllocationCount.
#
# Exit: 0 clean (waived messages only), 1 unwaived messages or port signals, 3 a run itself
# failed (test failure / scene FAIL / crash), 77 no layer, 2 usage (including a PXG_VVL_DIR
# without a layer, and a malformed waivers file).

set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
BUILD_T="${PXG_BUILD_TESTS:-$ROOT/build-gpuslang}"
BUILD_S="${PXG_BUILD_SDK:-$ROOT/build-sdk}"
CFG="$HERE/validation/vk_layer_settings.txt"
OUT=""
WHAT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --best-practices) CFG="$HERE/validation/vk_layer_settings_bp.txt" ;;
        --out) shift; OUT="$1" ;;
        -h|--help) sed -n '2,55p' "$0"; exit 0 ;;
        *) WHAT="$WHAT
$1" ;;
    esac
    shift
done
[ -n "$WHAT" ] || { echo "run_validation.sh: nothing to run (see --help)" >&2; exit 2; }

# `suite` runs ctest, whose gpuslang.validationSubset would start this script again
# under the same layer; the subset's suites are already part of the suite run, so the
# inner run is skipped (77) instead of measuring them twice.
if [ -n "${PXG_VVL_RUNNING:-}" ]; then
    echo "run_validation.sh: SKIP -- already inside a run_validation.sh run" >&2
    exit 77
fi
export PXG_VVL_RUNNING=1

# ---- the layer ----
LAYER_JSON=""
if [ -n "${PXG_VVL_DIR:-}" ]; then
    LAYER_JSON="$(find "$PXG_VVL_DIR" -name VkLayer_khronos_validation.json 2>/dev/null | head -1)"
    LAYER_LIB="$(find "$PXG_VVL_DIR" -name libVkLayer_khronos_validation.so 2>/dev/null | head -1)"
    if [ -z "$LAYER_JSON" ]; then
        echo "run_validation.sh: PXG_VVL_DIR=$PXG_VVL_DIR holds no VkLayer_khronos_validation.json" >&2
        exit 2
    fi
    export VK_LAYER_PATH="$(dirname "$LAYER_JSON")"
    [ -n "$LAYER_LIB" ] && export LD_LIBRARY_PATH="$(dirname "$LAYER_LIB")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    # VK_LAYER_PATH / VK_ADD_LAYER_PATH are ':'-separated lists, as the loader reads them.
    old_ifs="$IFS"; IFS=:
    for d in ${VK_LAYER_PATH:-} ${VK_ADD_LAYER_PATH:-} /usr/share/vulkan/explicit_layer.d \
             /usr/local/share/vulkan/explicit_layer.d /etc/vulkan/explicit_layer.d; do
        if [ -n "$d" ] && [ -f "$d/VkLayer_khronos_validation.json" ]; then
            LAYER_JSON="$d/VkLayer_khronos_validation.json"; break
        fi
    done
    IFS="$old_ifs"
fi
if [ -z "$LAYER_JSON" ]; then
    echo "run_validation.sh: SKIP -- no VK_LAYER_KHRONOS_validation (set PXG_VVL_DIR)" >&2
    exit 77
fi

# ---- only the checked-in configuration may configure the layer ----
scrubbed=""
for v in $(env | sed -n 's/^\(VK_LAYER_[A-Za-z0-9_]*\)=.*/\1/p; s/^\(VK_KHRONOS_VALIDATION_[A-Za-z0-9_]*\)=.*/\1/p'); do
    case "$v" in VK_LAYER_PATH|VK_ADD_LAYER_PATH) continue ;; esac
    unset "$v"
    scrubbed="$scrubbed $v"
done
[ -n "$scrubbed" ] && echo "run_validation.sh: removed inherited layer settings (they override the config):$scrubbed"

if [ -z "${PXG_SHADER_DIR:-}" ]; then
    PXG_SHADER_DIR="$BUILD_T/shaders"
fi
[ -d "$PXG_SHADER_DIR" ] || { echo "run_validation.sh: no shader dir $PXG_SHADER_DIR" >&2; exit 2; }
export PXG_SHADER_DIR
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
export VK_LAYER_SETTINGS_PATH="$CFG"
export PXG_VK_DEBUG_NAMES=1
export PXG_VK_ALLOC_STATS=1
OWN_OUT=0
[ -n "$OUT" ] || { OUT="$(mktemp -d "${TMPDIR:-/tmp}/pxg-vvl.XXXXXX")"; OWN_OUT=1; }
mkdir -p "$OUT"
echo "run_validation.sh: layer $LAYER_JSON"
echo "run_validation.sh: config $CFG"
echo "run_validation.sh: logs   $OUT"

# ---- prove the layer loads ----
VK_LOADER_DEBUG=layer "$BUILD_T/gpuslang-tests" pipelineRefusal_reflect > "$OUT/preflight.log" 2>&1
if ! grep -q 'Insert instance layer "VK_LAYER_KHRONOS_validation"' "$OUT/preflight.log"; then
    echo "run_validation.sh: the validation layer did NOT load (see $OUT/preflight.log)" >&2
    exit 3
fi

SUBSET="pipelineRefusal scan p6Indirect solverTGS sdfFind sdfParticle psSdfMidphase psDiffuseContactGen shimBindless"
# Run "$@" in memory mode $1: 0 = PXG_DEVICE_V2 removed from the environment, 1 = set to 1.
in_mode() {
    if [ "$1" = 1 ]; then shift; env PXG_DEVICE_V2=1 "$@"; else shift; env -u PXG_DEVICE_V2 "$@"; fi
}
runfail=0
logs=""
while IFS= read -r w; do
    [ -n "$w" ] || continue
    case "$w" in
        suite|suite:*)
            modes="0 1"; [ "$w" = "suite:0" ] && modes=0; [ "$w" = "suite:1" ] && modes=1
            for m in $modes; do
                log="$OUT/suite_v$m.log"
                echo "run_validation.sh: ctest mode $m -> $log"
                ( cd "$BUILD_T" && in_mode $m ctest -j4 -V --timeout 3600 ) > "$log" 2>&1 || runfail=1
                grep -E '^[0-9]+% tests passed|tests failed out of' "$log" | tail -1
                logs="$logs $log"
            done ;;
        subset|subset:*)
            modes="0 1"; [ "$w" = "subset:0" ] && modes=0; [ "$w" = "subset:1" ] && modes=1
            for m in $modes; do
                for s in $SUBSET; do
                    log="$OUT/subset_v${m}_$s.log"
                    in_mode $m "$BUILD_T/gpuslang-tests" "$s" > "$log" 2>&1 || { runfail=1; echo "  FAILED: $s (mode $m)"; }
                    logs="$logs $log"
                done
            done
            echo "run_validation.sh: subset ($SUBSET) ran in modes $modes" ;;
        scene:*)
            row="${w#scene:}"; lab="${row%%|*}"; envs="${row#*|}"; m=0
            case "$lab" in v1/*) m=1; lab="${lab#v1/}" ;; esac
            log="$OUT/scene_v${m}_$lab.log"
            # shellcheck disable=SC2086
            ( cd "$BUILD_S" && in_mode $m $envs PXG_GPU_STEP=1 ./gpu-sdk-step ) > "$log" 2>&1
            rc=$?
            v="$(grep -E '^\[ *(OK|FAIL) *\] gpu-sdk-step' "$log" | tail -1)"
            echo "run_validation.sh: scene $lab (mode $m) rc=$rc ${v:-<NO VERDICT>}" | cut -c1-160
            case "$v" in "[  OK  ]"*) [ $rc -eq 0 ] || runfail=1 ;; *) runfail=1 ;; esac
            logs="$logs $log" ;;
        *) echo "run_validation.sh: unknown target '$w'" >&2; exit 2 ;;
    esac
done <<EOF2
$WHAT
EOF2

# ---- triage: every message waived, or the run fails ----
# shellcheck disable=SC2086
python3 "$HERE/vvl_triage.py" --shader-dir "$PXG_SHADER_DIR" $logs
trc=$?
[ $trc -eq 2 ] && exit 2
if [ $runfail -ne 0 ]; then
    echo "run_validation.sh: FAIL -- a test or scene itself failed (logs in $OUT)"
    exit 3
fi
if [ $trc -eq 0 ]; then
    echo "run_validation.sh: PASS -- no unwaived validation message"
    # A log directory this run made for itself is removed on a pass (ctest's
    # gpuslang.validationSubset would otherwise leave one per run); --out keeps it.
    [ $OWN_OUT -eq 1 ] && rm -rf "$OUT"
    exit 0
fi
echo "run_validation.sh: FAIL -- unwaived validation messages (logs in $OUT)"
exit 1
