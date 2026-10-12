#!/bin/sh
# test_vvl_triage.sh -- self-test of tools/vvl_triage.py against the checked-in
# tools/validation_waivers.txt (P9.98 check round 1).  Registered as ctest
# gpuslang.vvlTriage; no GPU, no layer, about 2 s.
#
# Each row writes a tiny log and checks the exit code: 0 (everything waived), 1 (an
# unwaived message or port signal), 2 (input error).  The rows pin what round 1 added:
# a device / shim / bridge refusal FAILS outside the fixtures that provoke it on purpose
# (it used to pass silently -- the checker's box run under a lowered shared-memory limit
# refused solveWholeIslandTGS 365 times and printed PASS), a widened push range fails
# unless its entry is listed, and the F5 waivers hold only under their allocation bound.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
T="$(mktemp -d "${TMPDIR:-/tmp}/pxg-tri.XXXXXX")"
trap 'rm -rf "$T"' EXIT
SD="$T/shaders"
mkdir -p "$SD"
fails=0
row() {   # row <expected rc> <description> <log file name> <log text (printf format)>
    want="$1"; what="$2"; name="$3"; shift 3
    mkdir -p "$T/l"; rm -f "$T/l/"*
    printf "$@" > "$T/l/$name"
    python3 "$HERE/vvl_triage.py" --shader-dir "$SD" --summary-only "$T/l/$name" > "$T/out" 2>&1
    got=$?
    if [ "$got" = "$want" ]; then echo "ok    rc $got  $what"; else
        echo "FAIL  rc $got (want $want)  $what"; sed 's/^/        /' "$T/out"; fails=$((fails+1)); fi
}
REF='[PxgVulkan] createPipeline(solveWholeIslandTGS): Workgroup (groupshared) storage 31744 B exceeds the device'"'"'s maxComputeSharedMemorySize 16384 B; refused\n'
BP='UNASSIGNED-BestPractices-vkAllocateMemory-small-allocation(WARN / PERF): msgNum: 1 - Validation Performance Warning: [ UNASSIGNED-BestPractices-vkAllocateMemory-small-allocation ] Object 0: handle = 0x1, type = VK_OBJECT_TYPE_DEVICE; | MessageID = 0x1 | small\n'

row 0 "a clean scene log"                                        scene_v0_box.log 'step 1\n[  OK  ] gpu-sdk-step\n'
row 1 "a pipeline refusal in a scene"                            scene_v0_box.log "$REF"
row 1 "a [pxg-refused] launch in a scene"                        scene_v0_box.log '[pxg-refused] sized solveWholeIslandTGS: no pipeline\n'
row 1 "a refused dispatch (grid) in a scene"                     scene_v0_box.log '[PxgVulkan] dispatch: grid (70000, 1, 1) exceeds the device'"'"'s maxComputeWorkGroupCount[0] 65535 (pipeline 3); nothing recorded\n'
row 1 "a refused allocation in a scene"                          scene_v0_box.log '[PxgVulkan] createBuffer(64 B): vkAllocateMemory failed with VkResult -2; returning 0\n'
row 1 "a degraded init step in a scene"                          scene_v0_box.log '[PxgVulkan] init: vkCreatePipelineCache failed with VkResult -1; continuing without a pipeline cache\n'
row 1 "an out-of-range shim transfer in a scene"                 scene_v0_box.log '[PxgShim] memcpyHtoD: 8 B at offset 396 runs past the 400-byte allocation; InvalidValue\n'
row 0 "refusals in the pipelineRefusal fixture (subset log)"     subset_v0_pipelineRefusal.log "$REF"
row 0 "refusals in the shimBindless fixture (subset log)"        subset_v1_shimBindless.log "$REF"
row 1 "a refusal in any other suite (subset log)"                subset_v0_scan.log "$REF"
row 0 "ctest -V: a refusal attributed to pipelineRefusal"        suite_v0.log "      Start  7: gpuslang.pipelineRefusal\n7: $REF"
row 1 "ctest -V: a refusal attributed to scan"                   suite_v0.log "      Start  7: gpuslang.pipelineRefusal\n      Start  8: gpuslang.scan\n7: ok\n8: $REF"
row 0 "a widened push range on a listed engine entry"            scene_v1_particle.log '[PxgVulkan] createPipeline(ps_primitivesDiffuseCollisionLaunch): push block 176 B wider than caller 104 B; zero-filled\n'
row 1 "a widened push range on an unlisted entry"                scene_v1_particle.log '[PxgVulkan] createPipeline(sdfCollVertex): push block 64 B wider than caller 48 B; zero-filled\n'
row 1 "a widened range is not waived by a refusal fixture"       subset_v0_pipelineRefusal.log '[PxgVulkan] createPipeline(sdfCollVertex): push block 64 B wider than caller 48 B; zero-filled\n'
row 0 "an allocation peak under maxMemoryAllocationCount"        scene_v0_box.log '[pxg-alloc] peak 53 live VkDeviceMemory allocation(s) (maxMemoryAllocationCount 4096)\n'
row 1 "an allocation peak over maxMemoryAllocationCount"         scene_v0_box.log '[pxg-alloc] peak 5000 live VkDeviceMemory allocation(s) (maxMemoryAllocationCount 4096)\n'
row 0 "F5 small-allocation under its alloc-peak bound"           scene_v0_box.log "$BP[pxg-alloc] peak 53 live VkDeviceMemory allocation(s) (maxMemoryAllocationCount 4294967295)\n"
row 1 "F5 small-allocation over its alloc-peak bound"            scene_v0_box.log "$BP[pxg-alloc] peak 3000 live VkDeviceMemory allocation(s) (maxMemoryAllocationCount 4294967295)\n"
row 1 "F5 small-allocation in a log with no allocation report"   scene_v0_box.log "$BP"

# An input error is exit 2, not a pass: a scope with no name.
printf 'PXG-refusal  suite:  always\n' > "$T/bad_waivers.txt"
printf "$REF" > "$T/scene.log"
python3 "$HERE/vvl_triage.py" --shader-dir "$SD" --waivers "$T/bad_waivers.txt" "$T/scene.log" > /dev/null 2>&1
got=$?
if [ "$got" = 2 ]; then echo "ok    rc 2  a waiver with an empty suite: scope"; else
    echo "FAIL  rc $got (want 2)  a waiver with an empty suite: scope"; fails=$((fails+1)); fi

echo "test_vvl_triage.sh: $fails failure(s)"
[ "$fails" -eq 0 ]
