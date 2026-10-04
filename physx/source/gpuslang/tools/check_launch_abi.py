#!/usr/bin/env python3
# check_launch_abi.py -- P9.48d.  The EIGHTH static checker.
#
# Every kernel the manifest maps to a compiled Slang entry must ALSO have a row
# in the launch-ABI table (bridge/PxgVulkanKernelAbi.cpp).  Without one the
# bridge computes a push-constant size of 0, the shim's exact blob check rejects
# every launch, and the kernel NEVER RUNS on the engine path.
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS
# ---------------------------------------------------------------------------
# Because it already happened, to four kernels at once, and nothing said so.
#
# P9.48a mapped the three particle-side consumers of the FEM contact chain and
# P9.48b the softbody-particle attachment.  Every fixture passed, the manifest
# test passed, the suite passed in both memory modes, the scene sweep passed.
# The observing scene did not: a particle cloud dropped on a softbody fell
# straight through it.  The launch trace said the chain was running -- 361
# launches of the new consumer over 40 steps -- and it was wrong, because the
# trace printed one line per launch ATTEMPT and did not distinguish OK from
# REFUSED at a glance.  All 361 were refusals.
#
# The gap was one step of wiring: PxgVulkanKernelAbi.cpp is the single source of
# truth for a kernel's CUDA argument sizes, pushConstantSizeFor() derives the
# blob size from it, and an unlisted kernel gets 0.  Mapping a kernel in the
# manifest does not make it run; this table is what makes it run.
#
# The failure mode is the worst kind this port has: silent, total, and
# indistinguishable from "the kernel ran and found nothing".
#
# ---------------------------------------------------------------------------
# THE RULE, AND WHY IT IS A RATCHET
# ---------------------------------------------------------------------------
# The honest state of the tree is that 54 mapped kernels have no ABI row.  They
# are not all defects: most are on paths the engine never drives in this port
# (the aggregate broadphase, the direct-API getters and setters, the threshold
# stream), and writing a justification for each would be inventing certainty
# this checker does not have.
#
# So the rule is a RATCHET rather than a clean bill:
#
#   * the set of unwired kernels may SHRINK, never grow;
#   * a kernel newly mapped without its ABI row fails the check by name;
#   * and when one is wired, its name comes off this list, which is the only
#     way the number goes down.
#
# A ratchet is worth more here than an empty allowlist would be: the defect this
# exists for is a NEW kernel arriving unwired, and that is exactly what it
# catches on the day it happens.

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
MANIFEST = os.path.join(ROOT, "physx", "source", "gpuslang", "include",
                        "PxgSlangKernelManifest.h")
ABI = os.path.join(ROOT, "physx", "source", "gpuslang", "bridge",
                   "PxgVulkanKernelAbi.cpp")

# Kernels that are mapped to a Slang entry but carry no launch-ABI row.  Every
# name here is a kernel that WOULD be refused if the engine launched it.  The
# list may shrink; a name that is not on it and has no ABI row fails the check.
KNOWN_UNWIRED = {
    # (P9.79 wired the three direct-API getters PXG_SCENE=directapi launches and
    # updateBodiesLaunchDirectAPI; P9.93 wired the rest of the rigid direct API --
    # the five setters and the two acceleration getters -- behind
    # PXG_SCENE=directapiwrite, and updateBodyExternalVelocitiesLaunch, reached only
    # by PxRigidDynamic::addForce / addTorque & co., behind PXG_SCENE=extforce.)
    # (P9.91 wired the threshold / force-change stream -- all twelve, plus the two
    # radix kernels that were not mapped at all -- behind PXG_SCENE=forcethreshold.)
    # solver / sort odds and ends.  P9.79 wired radixSortCopyBits2 and
    # radixSortCopy2: PxgGpuNarrowphaseCore::computeRigidsToShapes() is their only
    # caller and it runs only under the direct-GPU API, so PXG_SCENE=directapi is
    # what finally launched them (and found them refused).
    # (P9.95 wired applyTGSSubstepGravity behind PXG_SCENE=substepgravity,
    # eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS.)
    # (P9.92 wired the aggregate broadphase -- all fourteen kernels of aggregate.cu --
    # behind PXG_SCENE=aggregate.)
    #
    # The thirteen below are UNREACHABLE in this build (P9.95 scope): every one of
    # them is launched only by an object that the PhysicsGpu factory constructs
    # (physxgpu/src/PxgPhysXGpu.cpp), and this build does not compile that factory --
    # PxgPhysXGpu.cpp is not in the sdktest glob, PxGetPhysicsGpu() stays NULL without
    # libcuda, and the bridge's PxGpuCreatePhysicsGpu() returns NULL.  They are also
    # binding-only ports, so an ABI row alone would not make them run.
    #
    # Generic scan / radix utilities: PxGpuScan and PxGpuRadixSort<PxU32>
    # (gpusimulationcontroller/src/PxgAlgorithms.cpp), owned only by the factory's
    # SDF builder, particle neighborhood provider and isosurface extractors.
    # reorderKernel's launcher (reorderLaunch, PxgAlgorithms.cpp:95) has no caller
    # even upstream.  interleaveBuffers is launched only by PxgArrayConverter
    # (PxgArrayConverter.cpp:57), factory-only (PxgPhysXGpu.cpp:458).
    "scanPerBlockKernel", "scanPerBlockKernel4x4", "addBlockSumsKernel",
    "addBlockSumsKernel4x4", "radixFourBitCountPerBlockKernel",
    "radixFourBitReorderKernel", "reorderKernel", "interleaveBuffers",
    # Deformable skinning / normal generation: launched by PxgDeformableSkinning
    # (PxgDeformableSkinning.cpp:59-87), not PxgArrayConverter; its only
    # construction site is the factory (PxgPhysXGpu.cpp:472).
    "interpolateSkinnedClothVertices", "interpolateSkinnedSoftBodyVertices",
    "normalVectorsAreaWeighted", "normalizeNormals", "zeroNormals",
}


def mapped_kernels(text):
    """cudaName of every manifest row that names a .spv."""
    return re.findall(r'\{\s*"([A-Za-z0-9_]+)"\s*,\s*"[^"]+\.spv"', text)


def abi_kernels(text):
    """cudaName of every launch-ABI row."""
    return set(re.findall(r'\{\s*"([A-Za-z0-9_]+)"\s*,\s*k[A-Za-z0-9_]+\s*,', text))


def selftest():
    """Negative control.  A checker that matched the manifest's UNMAPPED rows, or
    that could not see an ABI row at all, would report an empty problem set and
    read exactly like a clean tree."""
    ok = True
    fake_manifest = '''
    { "aMapped", "someShader.spv", "aMapped" },
    { "bUnmapped", nullptr, nullptr },
    { "cMapped", "other.spv", "cMapped" },
    '''
    fake_abi = '''
    { "aMapped", kSomeRow, uint32_t(sizeof(kSomeRow)) },
    { "zOther", kOtherRow, uint32_t(sizeof(kOtherRow)) },
    '''
    m = mapped_kernels(fake_manifest)
    if m != ["aMapped", "cMapped"]:
        print("SELFTEST FAIL: the mapped-row scan found %r" % (m,)); ok = False
    a = abi_kernels(fake_abi)
    if a != {"aMapped", "zOther"}:
        print("SELFTEST FAIL: the ABI-row scan found %r" % (sorted(a),)); ok = False
    missing = [k for k in m if k not in a]
    if missing != ["cMapped"]:
        print("SELFTEST FAIL: the missing set is %r, expected ['cMapped']" % (missing,))
        ok = False
    if "bUnmapped" in m:
        print("SELFTEST FAIL: an UNMAPPED manifest row was counted as mapped -- the "
              "check would then demand an ABI row for a kernel that has no shader")
        ok = False
    return ok


def main():
    if not selftest():
        print("the checker's own control failed; its verdicts mean nothing")
        return 1
    for path in (MANIFEST, ABI):
        if not os.path.isfile(path):
            print("cannot read %s" % path)
            return 1

    mapped = mapped_kernels(open(MANIFEST).read())
    have = abi_kernels(open(ABI).read())
    missing = [k for k in mapped if k not in have]

    new = [k for k in missing if k not in KNOWN_UNWIRED]
    fixed = sorted(KNOWN_UNWIRED - set(missing))

    if new:
        print("A MAPPED KERNEL HAS NO LAUNCH-ABI ROW, SO THE ENGINE CANNOT RUN IT.")
        print("pushConstantSizeFor() (bridge/PhysXGpuVulkan.cpp:178) derives the push-")
        print("constant size from bridge/PxgVulkanKernelAbi.cpp and yields 0 for an")
        print("unlisted kernel; the shim's blob check is EXACT (PxgCudaShim.cpp:232), so")
        print("every launch is refused and the kernel silently never runs.  A fixture")
        print("cannot see this -- it dispatches the shader directly -- and neither can")
        print("the manifest test.  Add the argument-size row:")
        print()
        for k in new:
            print("  %s" % k)
        return 1

    if fixed:
        print("These kernels now HAVE a launch-ABI row and must come off KNOWN_UNWIRED:")
        for k in fixed:
            print("  %s" % k)
        print()
        print("The list is a ratchet -- it may shrink, but only by being edited when it")
        print("does, or it stops meaning anything.")
        return 1

    print("launchAbi: %d mapped kernels, %d with a launch-ABI row, %d known unwired "
          "(the ratchet: it may shrink, never grow)"
          % (len(mapped), len(mapped) - len(missing), len(missing)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
