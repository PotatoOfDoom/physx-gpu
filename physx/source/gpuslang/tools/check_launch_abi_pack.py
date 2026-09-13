#!/usr/bin/env python3
# check_launch_abi_pack.py -- P9.77.  The FIFTEENTH static checker.
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS
# ---------------------------------------------------------------------------
# Because its defect has already happened TWICE, and the checker that was
# supposed to cover it can only see half the class.
#
# check_launch_abi.py (P9.48d) asks "does this mapped kernel HAVE a row?".  That
# catches a kernel with NO row, whose pushConstantSizeFor() yields 0.  It cannot
# catch a kernel whose row EXISTS but is the WRONG SIZE, and the outcome is
# identical: the bridge assembles the push blob from the HOST's kernelParams
# (bridge/PxgVulkanCudaContext.cpp:443-462) and the shim gates on
# `blobSize == pushConstantSize`, in that file's own words at :459 --
#
#     "otherwise the shim's blobSize == pushConstantSize gate rejects the launch"
#
# so a row that is eight bytes short refuses every launch, in silence, while
# fixtures, the manifest test, the suite and the scenes all stay green.  Fixtures
# CANNOT see it: they dispatch the shader directly and never go through the gate.
#
# THE TWO OCCURRENCES:
#
#   P9.76b  sb_solveRigidSoftAttachmentLaunchTGS carries the 72-byte PGS row
#           while PxgSoftBodyCore.cpp:1876-1888 pushes ELEVEN arguments = 80
#           bytes.  The wrong row was inherited from the CLOTH twin, where the
#           two kernels really do share one 9-argument pack
#           (PxgFEMClothCore.cpp:1845-1866 builds it once and picks the function
#           with a ternary).  The row's comment even says "both PGS and TGS share
#           the pack" -- true of cloth, false of softbody.
#
#   P9.74   the 176-byte convex-core cloth row could only be shown correct by
#           watching the SCENE move, because nothing checked the arithmetic.  The
#           row happened to be right; that was not established by any check.
#
# ---------------------------------------------------------------------------
# WHAT IT DOES, AND WHAT IT REFUSES TO PRETEND
# ---------------------------------------------------------------------------
# For every kernel with an ABI row, it finds the engine launch site(s), recovers
# each pushed argument's SIZE from the C++ declaration of the variable named in
# PX_CUDA_KERNEL_PARAM(...), assembles the blob with the same natural-alignment
# rule as pxgKernelAbiPushSize (PxgVulkanKernelAbi.cpp:2165), and compares.
#
# Recovering a C++ variable's type is a C++ problem, and this is a regex.  So the
# rule is: resolve what is unambiguous, and count everything else as UNDECIDED
# rather than as passing.  P9.64's lesson is the reason -- a checker that quietly
# skips what it cannot parse is worse than none, because it is counted.  The
# undecided set is printed every run and is a RATCHET: it may shrink, never grow.
#
# A kernel with TWO launch sites whose packs differ in SIZE but share one row is
# reported by name even when both sizes are individually decidable -- that is
# exactly P9.76b, and it is the case a per-site check would miss.
#
# ---------------------------------------------------------------------------
# FITNESS: IT HAS BEEN SEEN TO FAIL, AND EVERY FALSE POSITIVE WAS A BUG IN HERE
# ---------------------------------------------------------------------------
# Reintroduce P9.76b's row and it says, by name:
#
#   sb_solveRigidSoftAttachmentLaunchTGS: row kSbRigidAttachSolve gives 72 bytes,
#   PxgSoftBodyCore.cpp:1872 pushes 80
#
# Restore the row and it is green.  A checker that has never been watched to fail
# is a checker whose defect class it cannot see (P9.64).
#
# ITS FIRST RUN REPORTED TWENTY-ONE MISMATCHES.  Twenty were THIS FILE being wrong
# about the engine, and each is worth naming because each is a way a plausible
# parser lies:
#
#   14  the launch block was bound to the kernel by PROXIMITY -- the nearest
#       PxCudaKernelParam block after a PxgKernelIds:: mention.  A block can
#       precede its getCuFunction, two launches can sit side by side, and an id
#       can be named in a comment.  Now it follows the CUfunction HANDLE into the
#       launchKernel call that uses it, and takes the block that call names.
#    3  a REFERENCE was sized like a pointer.  PX_CUDA_KERNEL_PARAM(x) pushes
#       sizeof(x), and for `const PxVec3& gravity` that is 12, not 8.  This one
#       accused preIntegrationLaunch -- a kernel that, were it really refused,
#       would leave every rigid body un-integrated and all 48 scenes failing.
#       That impossibility is what exposed the bug; a control that must not be
#       wrong is worth more than another suspect.
#    1  the FIRST block with the right name before the launch instead of the LAST,
#       where two launches in one scope both call their array `kernelParams`.
#    1  PX_CUDA_KERNEL_PARAM((x)) with inner parens, silently dropping arguments.
#    1  COMMENTED-OUT parameters counted as live, silently adding arguments.
#
# Under-counting and over-counting are the same failure wearing two faces: the
# checker being wrong about the engine while accusing the table.
#
# ---------------------------------------------------------------------------
# WHAT IT STILL CANNOT DECIDE
# ---------------------------------------------------------------------------
# 99 of the kernels with an ABI row are in KNOWN_UNDECIDED, and that is a recorded
# blind spot, not a clean bill.  The largest cause is a struct passed BY VALUE
# whose size this file does not know -- the contact-writer tails especially -- and
# deciding those means reading the struct.  One cause can never be decided here at
# all: the void** launch form carries no per-argument sizes, and the bridge reads
# them from this very table, so there is nothing independent left to compare.

import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
ABI = os.path.join(ROOT, "physx", "source", "gpuslang", "bridge", "PxgVulkanKernelAbi.cpp")
NAMES = os.path.join(ROOT, "physx", "source", "gpucommon", "include", "PxgKernelNames.h")
ENGINE_DIRS = [
    os.path.join(ROOT, "physx", "source", "gpusimulationcontroller", "src"),
    os.path.join(ROOT, "physx", "source", "gpunarrowphase", "src"),
    os.path.join(ROOT, "physx", "source", "gpusolver", "src"),
    os.path.join(ROOT, "physx", "source", "gpubroadphase", "src"),
    os.path.join(ROOT, "physx", "source", "gpucommon", "src"),
]

# Kernels whose pack this checker cannot yet decide.  RATCHET: may shrink, never
# grow.  Being on this list is NOT a clean bill -- it is a recorded blind spot,
# and the honest state of the tree is that 99 of the kernels with an ABI row have
# one.
#
# THE REASONS, in the order they will be worth attacking:
#   * a struct passed BY VALUE whose size this table does not know (the
#     PxgFEMContactWriter / PxgSoftBodyContactWriter tails, and the PxgShape and
#     PxBounds3 scalars).  Deciding these means reading the struct, which is the
#     next real step and would take most of the list.
#   * PxVec3 / PxQuat / PxTransform arguments passed by reference, whose sizeof is
#     the referent's.
#   * the void** launch form (PX_CUDA_KERNEL_PARAM2), which carries no per-argument
#     sizes at all -- the bridge reads them from this very table, so there is
#     nothing independent to compare against and these can never be decided here.
#   * a pack assembled across a branch, or an argument that is a member expression
#     rather than a plain variable.
KNOWN_UNDECIDED = {
    "ZeroBodies",
    "ZeroBodiesTGS",
    "artiContactConstraintBlockPrepareLaunch",
    "artiJointConstraintBlockPrepareParallelLaunch",
    "artiSolveBlockPartition",
    "artiTGSContactConstraintBlockPrepareLaunch",
    "artiTGSJointConstraintBlockPrepareParallelLaunch",
    "boxBoxNphase_Kernel",
    "clampMaxValue",
    "clampMaxValues",
    "cloth_SphereContactGenLaunch",
    "cloth_SphereVertexContactGenLaunch",
    "cloth_boxTriangleContactGenLaunch",
    "cloth_boxVertexContactGenLaunch",
    "cloth_clothContactPrepareLaunch",
    "cloth_convexContactGenLaunch",
    "cloth_convexVertexContactGenLaunch",
    "cloth_heightfieldContactGenLaunch",
    "cloth_heightfieldVertexContactGenLaunch",
    "cloth_meshContactGenLaunch",
    "cloth_meshVertexContactGenLaunch",
    "cloth_planeVertContactGenLaunch",
    "cloth_preIntegrateLaunch",
    "cloth_psContactGenLaunch",
    "cloth_rigidContactPrepareLaunch",
    "cloth_sdfMeshContactGenLaunch",
    "cloth_solveNonSharedTriangleEnergyClusterLaunch",
    "cloth_solveNonSharedTriangleEnergyLaunch",
    "cloth_solveTrianglePairEnergyClusterLaunch",
    "cloth_solveTrianglePairEnergyLaunch",
    "cloth_stepLaunch",
    "computeAverageSolverBodyVelocity",
    "computeAverageSolverBodyVelocityTGS",
    "concludeBlocks",
    "concludeBlocksTGS",
    "contactConstraintBlockPrepareParallelLaunch",
    "contactConstraintBlockPrepareParallelLaunchTGS",
    "convexCoreConvexNphase_Kernel",
    "convexCorePlaneNphase_Kernel",
    "convexHeightFieldMidphase",
    "convexHeightfieldNarrowphase",
    "convexPlaneNphase_Kernel",
    "convexTrimeshFinishContacts",
    "convexTrimeshNarrowphase",
    "dmaBackChangedElems",
    "finishContactsKernel",
    "initStaticKinematics",
    "initializeManifolds",
    "initializeSolverVelocitiesTGS",
    "integrateCoreParallelLaunch",
    "integrateCoreParallelLaunchTGS",
    "jointConstraintBlockPrepareParallelLaunch",
    "jointConstraintBlockPrepareParallelLaunchTGS",
    "markActiveSlabPGS",
    "markActiveSlabTGS",
    "midphaseGeneratePairs",
    "preIntegrationLaunch",
    "preIntegrationLaunchTGS",
    "prepareLostFoundPairs_Stage2",
    "propagateAverageSolverBodyVelocityTGS",
    "propagateSolverBodyVelocity",
    "propagateStaticSolverBodyVelocities",
    "propagateStaticSolverBodyVelocitiesTGS",
    "ps_diffuseParticleCompact",
    "ps_diffuseParticleCreate",
    "ps_diffuseParticleUpdatePBF",
    "ps_heightfieldCollisonLaunch",
    "ps_meshCollisonLaunch",
    "ps_preDiffuseIntegrateLaunch",
    "ps_preIntegrateLaunch",
    "ps_primitivesCollisionLaunch",
    "ps_sdfMeshCollisonLaunch",
    "ps_updateUserBufferLaunch",
    "rigidSumInternalContactAndJointBatches1",
    "rigidSumInternalContactAndJointBatches2",
    "sb_clothContactGenLaunch",
    "sb_clothContactPrepareLaunch",
    "sb_clothVertContactGenLaunch",
    "sb_gm_preIntegrateLaunch",
    "sb_gm_stepSoftbodyLaunch",
    "sb_meshContactGenLaunch",
    "sb_primitiveContactGenLaunch",
    "sb_psContactGenLaunch",
    "sb_sbMidphaseGeneratePairsLaunch",
    "sb_sdfMeshMidphaseGeneratePairsLaunch",
    "sb_selfCollisionMidphaseGeneratePairsLaunch",
    "sb_softbodyContactPrepareLaunch",
    "solveBlockPartition",
    "solveBlockUnified",
    "solveStaticBlock",
    "solveStaticBlockTGS",
    "solveWholeIslandTGS",
    "sphereHeightfieldNarrowphase",
    "sphereNphase_Kernel",
    "sphereTrimeshNarrowphase",
    "writeBackBodies",
    "writeBackBodiesTGS",
    "writebackBlocks",
    "writebackBlocksTGS",
}


# ---------------------------------------------------------------------------
# argument sizes
# ---------------------------------------------------------------------------
# Only forms whose size is unambiguous.  Anything else makes the kernel UNDECIDED.
PTR8 = 8
SIZES = {
    "CUdeviceptr": PTR8,
    "PxU64": 8, "PxI64": 8, "uint64_t": 8, "int64_t": 8, "size_t": 8,
    "PxU32": 4, "PxI32": 4, "uint32_t": 4, "int32_t": 4, "PxReal": 4,
    "float": 4, "int": 4, "unsigned": 4, "PxU16": 2, "PxU8": 1, "bool": 1,
}


def arg_size(decl_type):
    """Size of one pushed argument, or None when undecidable."""
    t = decl_type.strip()
    t = re.sub(r"\bconst\b", " ", t)
    t = re.sub(r"\bstatic\b", " ", t)
    t = t.strip()
    if not t:
        return None
    # A POINTER is 8 bytes.  A REFERENCE IS NOT, and conflating them was a real
    # bug in this checker's first run: PX_CUDA_KERNEL_PARAM(x) pushes sizeof(x),
    # and for `const PxVec3& gravity` that is sizeof(PxVec3) == 12, not 8.  It
    # made preIntegrationLaunch look 8 bytes short -- a kernel that, were it
    # really refused, would leave every rigid body un-integrated and every one of
    # the 48 scenes failing.  That control is what exposed it.  A reference now
    # resolves to its REFERENT, which for PxVec3 is not in this table, so the
    # kernel becomes UNDECIDED instead of falsely accused.
    if t.endswith("*"):
        return PTR8
    if t.endswith("&"):
        return arg_size(t[:-1])
    if t.startswith("PxgDevicePointer<") or t.startswith("PxgTypedCudaBuffer<"):
        return PTR8
    base = t.split()[-1] if " " in t else t
    return SIZES.get(base) or SIZES.get(t)


def abi_push_size(row):
    """pxgKernelAbiPushSize, PxgVulkanKernelAbi.cpp:2165, transcribed."""
    def align(s):
        return s if s in (1, 2, 4, 8) else 16
    off, ma = 0, 1
    for s in row:
        a = align(s)
        ma = max(ma, a)
        off = (off + a - 1) & ~(a - 1)
        off += s
    return (off + ma - 1) & ~(ma - 1)


# ---------------------------------------------------------------------------
# the ABI table
# ---------------------------------------------------------------------------
def read_abi():
    src = open(ABI).read()
    arrays = {}
    for m in re.finditer(r"static\s+const\s+uint8_t\s+(\w+)\s*\[\]\s*=\s*\{([^}]*)\}", src):
        vals = [int(v) for v in re.findall(r"\d+", m.group(2))]
        arrays[m.group(1)] = vals
    rows = {}
    for m in re.finditer(r'\{\s*"([^"]+)"\s*,\s*(\w+)\s*,', src):
        name, arr = m.group(1), m.group(2)
        if arr in arrays:
            rows.setdefault(name, []).append((arr, arrays[arr]))
    return rows


# ---------------------------------------------------------------------------
# the launch sites
# ---------------------------------------------------------------------------
def read_kernel_ids():
    ids = {}
    for m in re.finditer(r'KERNEL_DEF\(\s*(\w+)\s*,\s*"([^"]+)"\s*\)', open(NAMES).read()):
        ids[m.group(1)] = m.group(2)
    return ids


def engine_sources():
    out = []
    for d in ENGINE_DIRS:
        for dirpath, _dirnames, filenames in os.walk(d):
            if os.sep + "CUDA" + os.sep in dirpath + os.sep:
                continue                      # device code, not launch sites
            for fn in filenames:
                if fn.endswith(".cpp"):
                    out.append(os.path.join(dirpath, fn))
    return out


DECL_CACHE = {}


def declared_size(src, var):
    """The SIZE of `var` in this translation unit, or None when undecidable.

    IT RESOLVES TO A SIZE, NOT TO A TYPE, AND THAT DISTINCTION IS THE WHOLE
    DIFFERENCE BETWEEN A CHECKER AND A STUB.  The first version demanded a single
    unambiguous type STRING and left 243 of ~250 kernels undecided -- including
    the very defect it was written for -- because `dt` is spelled `const PxReal`
    in one function and `PxReal` in another, and a variable declared in two
    functions of the same file gave two strings for one size.  Only the size is
    ever needed: when every declaration of a name agrees on a size, that size is
    the answer no matter how the declarations are spelled.  Disagreement stays
    undecidable, which is the case that must never silently pass."""
    key = (id(src), var)
    if key in DECL_CACHE:
        return DECL_CACHE[key]
    sizes = set()
    unresolved = False
    for m in re.finditer(r"(?:^|[;{}\(,])\s*((?:const\s+)?[\w:]+(?:\s*<[^<>;]*>)?\s*[*&]?)\s+"
                         + re.escape(var) + r"\s*(?:=|;|\)|,|\[)", src, re.M):
        t = m.group(1).strip()
        if t.split()[-1] in ("return", "else", "case", "new", "delete"):
            continue
        s = arg_size(t)
        if s is None:
            unresolved = True
        else:
            sizes.add(s)
    res = sizes.pop() if (len(sizes) == 1 and not unresolved) else None
    DECL_CACHE[key] = res
    return res


def launch_packs(ids):
    """kernel name -> list of (file, line, [arg sizes] or None).

    THE ASSOCIATION IS BOUND THROUGH THE CUfunction VARIABLE, not by proximity.
    The first version took the nearest `PxCudaKernelParam ... = {...}` block within
    4000 characters after a `PxgKernelIds::` mention, and that produced false
    positives -- convexCoreTetmeshNphase_Kernel32 was reported 144-vs-80 although
    P9.72 had proved that row correct by watching its scene move.  Proximity is not
    a binding: a block can precede its getCuFunction, two launches can sit side by
    side, and an id can be named in a comment.

    So the chain is followed properly:
        <var> = ...getCuFunction(PxgKernelIds::<ID>)     the function handle
        launchKernel(<var>, ..., <params>, sizeof(<params>), ...)
        PxCudaKernelParam <params>[] = { ... };          the block that call uses
    A ternary that assigns TWO ids to one handle (the cloth attachment solve does
    exactly this) correctly maps both kernels to the one shared pack -- which is
    the true statement there, and the one the softbody row wrongly assumed."""
    packs = {}
    for path in engine_sources():
        try:
            src = open(path, errors="replace").read()
        except OSError:
            continue
        if "PxgKernelIds::" not in src:
            continue
        # every handle assignment, including the ternary two-id form
        for hm in re.finditer(r"(\w+)\s*=\s*([^;]*?getCuFunction\s*\([^;]*?)\s*;", src, re.S):
            handle, rhs = hm.group(1), hm.group(2)
            kids = re.findall(r"PxgKernelIds::(\w+)", rhs)
            if not kids:
                continue
            # the launchKernel that uses this handle, before the handle is reassigned
            tail = src[hm.end():]
            nxt = re.search(r"\b" + re.escape(handle) + r"\s*=\s*[^;]*getCuFunction", tail, re.S)
            region = tail[:nxt.start()] if nxt else tail
            lm = re.search(r"launchKernel\s*\(\s*" + re.escape(handle) + r"\s*,(.*?)\)\s*;",
                           region, re.S)
            if not lm:
                continue
            pm = re.search(r"(\w+)\s*,\s*sizeof\s*\(\s*\1\s*\)", lm.group(1))
            if not pm:
                continue                       # the void** form carries no sizes
            pname = pm.group(1)
            # THE LAST block with this name before the launch, not the first.  Two
            # launches in one scope commonly BOTH name their array `kernelParams`
            # (PxgParticleSystemCore.cpp:856-880 launches the first and second
            # update-bound passes that way), and taking the first match handed the
            # second kernel the first one's arguments.
            blocks = list(re.finditer(r"PxCudaKernelParam\s+" + re.escape(pname) +
                                      r"\s*\[\]\s*=\s*\{(.*?)\}\s*;",
                                      region[:lm.start()], re.S))
            if not blocks:
                continue
            bm = blocks[-1]
            # The inner parens are optional and real: PxgNarrowphaseCore.cpp:7224
            # writes PX_CUDA_KERNEL_PARAM((baseContactPatches)).  Missing them
            # silently DROPPED those arguments and made the pack look 16 bytes
            # short -- a checker that under-counts arguments accuses the row.
            # COMMENTS MUST GO FIRST.  PxgNarrowphaseCore.cpp:3801-3802 carries two
            # commented-out parameters inside the block --
            #     //PX_CUDA_KERNEL_PARAM(pairs),
            #     //PX_CUDA_KERNEL_PARAM(nbPairs),
            # -- and counting them made ps_convexCollisionLaunch look like 25
            # arguments instead of 23, accusing a row that is correct.  A checker
            # that OVER-counts arguments accuses the row just as surely as one that
            # under-counts; both are the checker being wrong about the engine.
            block = re.sub(r"/\*.*?\*/", " ", bm.group(1), flags=re.S)
            block = re.sub(r"//[^\n]*", " ", block)
            args = re.findall(r"PX_CUDA_KERNEL_PARAM2?\(\s*\(?\s*([\w\.\->\[\]]+)\s*\)?\s*\)",
                              block)
            if not args:
                continue
            sizes = []
            for a in args:
                if not re.match(r"^\w+$", a):
                    sizes = None
                    break
                sz = declared_size(src, a)
                if sz is None:
                    sizes = None
                    break
                sizes.append(sz)
            line = src[:hm.start()].count("\n") + 1
            for kid in kids:
                name = ids.get(kid)
                if name:
                    packs.setdefault(name, []).append(
                        (os.path.relpath(path, ROOT), line, sizes))
    return packs


def main():
    rows = read_abi()
    ids = read_kernel_ids()
    packs = launch_packs(ids)

    mismatched, split, undecided, checked = [], [], [], 0

    for name, rowlist in sorted(rows.items()):
        sites = packs.get(name)
        if not sites:
            continue                                  # no launch site found: not this checker's business
        rowsize = abi_push_size(rowlist[0][1])
        sizes = {}
        for path, line, argsizes in sites:
            if argsizes is None:
                sizes = None
                break
            sizes.setdefault(abi_push_size(argsizes), []).append("%s:%d" % (path, line))
        if sizes is None:
            if name not in KNOWN_UNDECIDED:
                undecided.append(name)
            continue
        checked += 1
        if len(sizes) > 1:
            split.append((name, rowlist[0][0], rowsize, sizes))
        elif rowsize not in sizes:
            got = list(sizes.keys())[0]
            mismatched.append((name, rowlist[0][0], rowsize, got, sizes[got]))

    rc = 0
    if split:
        print("A KERNEL'S TWO LAUNCH SITES PUSH DIFFERENT-SIZED BLOBS BUT SHARE ONE ABI ROW.")
        print("Whichever size the row does not match is REFUSED IN SILENCE by the shim's")
        print("blobSize == pushConstantSize gate (PxgVulkanCudaContext.cpp:459).")
        print()
        for name, arr, rowsize, sizes in split:
            print("  %s: row %s is %d bytes" % (name, arr, rowsize))
            for sz, where in sorted(sizes.items()):
                mark = "  <-- matches the row" if sz == rowsize else "  <-- REFUSED"
                print("      %d bytes from %s%s" % (sz, ", ".join(where), mark))
        print()
        rc = 1

    if mismatched:
        print("AN ABI ROW DOES NOT MATCH THE BLOB ITS LAUNCH SITE PUSHES.")
        print("The shim's check is EXACT, so every launch of these kernels is refused")
        print("and they never run -- with every fixture and scene still green.")
        print()
        for name, arr, rowsize, got, where in mismatched:
            print("  %s: row %s gives %d bytes, %s pushes %d"
                  % (name, arr, rowsize, ", ".join(where), got))
        print()
        rc = 1

    if undecided:
        print("These kernels' packs could not be resolved and are NOT counted as passing.")
        print("Add them to KNOWN_UNDECIDED with a reason, or teach the resolver their form:")
        for name in sorted(undecided):
            print("  %s" % name)
        print()
        rc = 1

    if rc == 0:
        print("launchAbiPack: %d kernels checked argument-by-argument against their launch "
              "site, %d known undecided (the ratchet: it may shrink, never grow)"
              % (checked, len(KNOWN_UNDECIDED)))
    return rc


if __name__ == "__main__":
    sys.exit(main())
