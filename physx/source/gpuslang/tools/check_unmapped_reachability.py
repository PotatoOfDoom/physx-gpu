#!/usr/bin/env python3
# Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.
#
# check_unmapped_reachability.py -- the THIRTEENTH static checker.
#
# NOT EVERY UNMAPPED KERNEL IS REMAINING WORK.
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS
# ---------------------------------------------------------------------------
# "403 of 501 mapped, 98 to go" is the number this port has quoted at itself for a
# long time, and it is misleading.  P9.65 classified all 98 and found that most of
# them CANNOT BE REACHED IN THIS BUILD AT ALL -- not "are not yet exercised by a
# scene", but cannot be constructed:
#
#   * The isosurface extractors, the SDF builder, the anisotropy and smoothed-position
#     generators and the particle neighbourhood provider are application-facing
#     features.  They are created only through `PxGetPhysicsGpu()`, whose factory
#     lives in physxgpu/src/PxgPhysXGpu.cpp.
#   * That file is NOT COMPILED into the gpuslang build -- the sdktest recipe globs
#     gpucommon/gpubroadphase/gpunarrowphase/gpusimulationcontroller/gpusolver/
#     gpuarticulation but never physxgpu -- and the Vulkan bridge's replacement
#     `PxGpuCreatePhysicsGpu()` returns NULL.
#   * So no application, and no scene, can construct any of those owners here.
#
# Porting those kernels would raise PXG_SLANG_KERNEL_MAPPED without raising what the
# port can DO, and nothing in the build could observe a wrong answer from any of them.
# That is worth knowing before choosing a tranche, and worth keeping true afterwards.
#
# ---------------------------------------------------------------------------
# WHAT IS CHECKED
# ---------------------------------------------------------------------------
# Every unmapped manifest row is classified into exactly one bucket:
#
#   ORPHAN      no `PxgKernelIds::<enum>` reference anywhere outside gpuslang, i.e.
#               nothing launches it even in the stock CUDA build.  Cross-checked
#               against check_disabled_kernels.py's ORPHANED table, which must
#               already contain it -- a new orphan that is not recorded there is an
#               error, because that is the checker that keeps orphans honest.
#
#   STANDALONE  launched only from files belonging to an owner in STANDALONE_OWNERS
#               below, i.e. unreachable in this build.  The claim is not taken on
#               trust: see VERIFYING THE STANDALONE CLAIM.
#
#   ENGINE      anything else -- launched from the simulation pipeline.  THIS is the
#               remaining work, and it is a RATCHET (engine_reachable_unmapped.txt):
#               the list may shrink as kernels are ported, never grow silently.
#
# ---------------------------------------------------------------------------
# VERIFYING THE STANDALONE CLAIM
# ---------------------------------------------------------------------------
# A hardcoded "this is unreachable" list would rot the first time someone wired one
# of these features into the engine, and would do so silently -- exactly the failure
# this file exists to prevent.  So each of the three legs is re-derived every run:
#
#   1. every owner class in STANDALONE_OWNERS is `PX_NEW`'d ONLY inside the app
#      factory physxgpu/src/PxgPhysXGpu.cpp;
#   2. that factory's entry point is stubbed out in the bridge -- gpuslang's
#      `PxGpuCreatePhysicsGpu()` returns NULL;
#   3. physxgpu/src is not in the sdktest source glob.
#
# If any leg breaks, the bucket is no longer sound and the check fails loudly rather
# than continuing to report a comfortable number.
#
# ---------------------------------------------------------------------------
# WHAT THIS DELIBERATELY CANNOT DO
# ---------------------------------------------------------------------------
#  * It says nothing about the stock NVIDIA CUDA build, where PxgPhysXGpu.cpp IS
#    compiled and an application CAN create these objects.  STANDALONE means
#    "unreachable in THIS build", which is the build every measurement here uses.
#  * ENGINE means "the pipeline launches it", not "a scene reaches it".  A kernel can
#    be launched only under a feature no scene enables.  That distinction belongs to
#    check_disabled_kernels.py, which already tracks it.
#  * Launch-site detection is a `PxgKernelIds::<enum>` text match over .cpp outside
#    gpuslang.  A launch built by string or macro would be missed, and the kernel
#    would be misfiled as an ORPHAN -- so ORPHAN is cross-checked against the guard
#    list rather than trusted on its own.
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GS = os.path.dirname(HERE)                       # .../source/gpuslang
SRC = os.path.dirname(GS)                        # .../source
RATCHET = os.path.join(HERE, "engine_reachable_unmapped.txt")

# owner .cpp -> the classes whose construction gates it.  Both halves are checked.
STANDALONE_OWNERS = {
    "PxgIsosurfaceExtraction.cpp": ["PxgDenseGridIsosurfaceExtractor",
                                    "PxgSparseGridIsosurfaceExtractor"],
    "PxgSDFBuilder.cpp":           ["PxgSDFBuilder"],
    "PxgSparseGridStandalone.cpp": ["PxgParticleNeighborhoodProvider"],
    "PxgSmoothing.cpp":            ["PxgSmoothedPositionGenerator"],
    "PxgAnisotropy.cpp":           ["PxgAnisotropyGenerator"],
}

FACTORY = os.path.join(SRC, "physxgpu/src/PxgPhysXGpu.cpp")
BRIDGE = os.path.join(GS, "bridge/PhysXGpuVulkan.cpp")
SDKTEST_CMAKE = os.path.join(GS, "sdktest/CMakeLists.txt")


def kernel_enums():
    """CUDA kernel name -> PxgKernelIds enum member."""
    out = {}
    path = os.path.join(SRC, "gpucommon/include/PxgKernelNames.h")
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r'\s*KERNEL_DEF\(\s*(\w+)\s*,\s*"([^"]+)"\s*\)', line)
        if m:
            out[m.group(2)] = m.group(1)
    return out


def unmapped_rows():
    path = os.path.join(GS, "include/PxgSlangKernelManifest.h")
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r'\s*\{\s*"([^"]+)"\s*,\s*nullptr\s*,\s*nullptr\s*\}', line)
        if m:
            out.append(m.group(1))
    return out


def host_sources():
    """Every .cpp outside gpuslang -- the engine and the app factory."""
    out = []
    for dirpath, _d, files in os.walk(SRC):
        if os.sep + "gpuslang" in dirpath:
            continue
        for fn in files:
            if fn.endswith(".cpp"):
                out.append(os.path.join(dirpath, fn))
    return out


def guard_orphans():
    """The ORPHANED table in check_disabled_kernels.py, as a set of kernel names."""
    path = os.path.join(HERE, "check_disabled_kernels.py")
    text = open(path, encoding="utf-8", errors="replace").read()
    # anchor on the assignment, not the first mention -- "ORPHANED" also appears in
    # the file's prose header, and starting there lands on a bracket in a comment
    m = re.search(r"^ORPHANED\s*=\s*\[", text, re.M)
    if m is None:
        return None
    depth, end = 0, len(text)
    start = m.end() - 1
    for j in range(start, len(text)):
        if text[j] == "[":
            depth += 1
        elif text[j] == "]":
            depth -= 1
            if depth == 0:
                end = j
                break
    return set(re.findall(r"'([A-Za-z_][A-Za-z_0-9]*)'", text[start:end]))


def verify_standalone_basis(texts):
    """Re-derive the three legs that make STANDALONE sound. Returns a list of errors."""
    errs = []

    # leg 1 -- every owner class is constructed only in the app factory
    for owner, classes in sorted(STANDALONE_OWNERS.items()):
        for cls in classes:
            sites = sorted(p for p, t in texts.items() if ("PX_NEW(%s)" % cls) in t)
            outside = [p for p in sites if os.path.abspath(p) != os.path.abspath(FACTORY)]
            if not sites:
                errs.append("%s: no PX_NEW(%s) anywhere -- the owner list is stale"
                            % (owner, cls))
            elif outside:
                errs.append("%s: %s is constructed OUTSIDE the app factory (%s); it may "
                            "now be reachable from the engine" % (owner, cls, ", ".join(outside)))

    # leg 2 -- the bridge stubs the factory entry point to NULL
    if not os.path.exists(BRIDGE):
        errs.append("bridge/PhysXGpuVulkan.cpp not found")
    else:
        b = open(BRIDGE, encoding="utf-8", errors="replace").read()
        m = re.search(r"PxGpuCreatePhysicsGpu\s*\([^)]*\)\s*\{([^}]*)\}", b)
        if m is None:
            errs.append("bridge: PxGpuCreatePhysicsGpu() not found")
        elif "NULL" not in m.group(1) and "nullptr" not in m.group(1):
            errs.append("bridge: PxGpuCreatePhysicsGpu() no longer returns NULL -- the "
                        "app-facing GPU feature factory may now be live")

    # leg 3 -- physxgpu/src is not compiled into the sdk test binary
    if not os.path.exists(SDKTEST_CMAKE):
        errs.append("sdktest/CMakeLists.txt not found")
    else:
        c = open(SDKTEST_CMAKE, encoding="utf-8", errors="replace").read()
        m = re.search(r"file\(GLOB_RECURSE\s+GPUHOST_SRC(.*?)\)", c, re.S)
        if m is None:
            errs.append("sdktest: GPUHOST_SRC glob not found")
        elif "physxgpu/src" in m.group(1):
            errs.append("sdktest: physxgpu/src is now in the GPUHOST_SRC glob -- the app "
                        "factory is compiled and STANDALONE no longer holds")
    return errs


def load_ratchet():
    out = set()
    if os.path.exists(RATCHET):
        for line in open(RATCHET, encoding="utf-8"):
            line = line.split("#")[0].strip()
            if line:
                out.add(line)
    return out


def main():
    enums = kernel_enums()
    rows = unmapped_rows()
    paths = host_sources()
    texts = {p: open(p, encoding="utf-8", errors="replace").read() for p in paths}

    basis_errs = verify_standalone_basis(texts)

    buckets = {"ORPHAN": [], "STANDALONE": [], "ENGINE": []}
    launchers = {}
    for name in rows:
        enum = enums.get(name)
        if enum is None:
            buckets["ENGINE"].append(name)   # not in the registry: treat as work, loudly
            launchers[name] = {"<no PxgKernelIds enum>"}
            continue
        pat = re.compile(r"PxgKernelIds::" + re.escape(enum) + r"\b")
        files = {os.path.basename(p) for p, t in texts.items() if pat.search(t)}
        launchers[name] = files
        if not files:
            buckets["ORPHAN"].append(name)
        elif files <= set(STANDALONE_OWNERS):
            buckets["STANDALONE"].append(name)
        else:
            buckets["ENGINE"].append(name)

    print("unmapped-reachability over %d unmapped manifest rows" % len(rows))
    print("  ORPHAN     (nothing launches it, even upstream): %d" % len(buckets["ORPHAN"]))
    print("  STANDALONE (app-facing feature, not constructible here): %d" % len(buckets["STANDALONE"]))
    print("  ENGINE     (launched from the simulation pipeline): %d" % len(buckets["ENGINE"]))

    errs = list(basis_errs)

    guard = guard_orphans()
    if guard is None:
        errs.append("could not read the ORPHANED table from check_disabled_kernels.py")
    else:
        loose = sorted(set(buckets["ORPHAN"]) - guard)
        if loose:
            errs.append("%d orphan(s) not recorded in check_disabled_kernels.py's ORPHANED "
                        "table: %s" % (len(loose), ", ".join(loose)))

    known = load_ratchet()
    engine = set(buckets["ENGINE"])
    fresh = sorted(engine - known)
    stale = sorted(known - engine)

    if fresh:
        print("\n%d ENGINE-REACHABLE unmapped kernel(s) not in the ratchet:\n" % len(fresh))
        for n in fresh:
            print("  %-46s launched from %s" % (n, ", ".join(sorted(launchers[n])) or "?"))
        print("\nThese are remaining work on a live path.  Port them, or add them to")
        print("engine_reachable_unmapped.txt with the reason.")
    if stale:
        print("\n%d STALE entry(ies) in engine_reachable_unmapped.txt:\n" % len(stale))
        for n in stale:
            print("  %s -- no longer an unmapped engine kernel; remove the line" % n)
        print("\nA ratchet that keeps finished kernels on its list stops ratcheting.")

    if errs:
        print("\nTHE STANDALONE CLASSIFICATION IS NO LONGER SOUND:\n")
        for e in errs:
            print("  " + e)
        print("\nRe-establish the basis, or reclassify -- do not silently keep the number.")

    if fresh or stale or errs:
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
