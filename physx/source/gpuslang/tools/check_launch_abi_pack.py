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
# *** NOT YET WIRED INTO THE BUILD.  IT IS NOT TRUSTWORTHY YET. ***
# ---------------------------------------------------------------------------
# It CATCHES its target: sb_solveRigidSoftAttachmentLaunchTGS is reported as
# "row kSbRigidAttachSolve gives 72 bytes, PxgSoftBodyCore.cpp:1873 pushes 80",
# which is exactly the hand derivation in P9.76b.
#
# It also reports about twenty other kernels, and AT LEAST ONE OF THOSE IS A
# FALSE POSITIVE: convexCoreTetmeshNphase_Kernel32 is flagged as 144-vs-80, and
# that row is independently known correct -- P9.72 proved it the only way
# available at the time, by watching the scene move from -73.8069 to a rest.  A
# row cannot be both refused on every launch and responsible for a scene that
# runs.
#
# THE CAUSE IS THE ASSOCIATION, not the arithmetic.  launch_packs() takes the
# nearest `PxCudaKernelParam ... = { ... }` block within 4000 characters AFTER a
# `PxgKernelIds::X` mention.  That is wrong wherever the block precedes the
# getCuFunction call, wherever two launches sit close together, or wherever the
# id is named in a comment.  Until the block is bound to its launch by SCOPE --
# same enclosing braces, and the same CUfunction variable that reaches
# launchKernel -- every report has to be checked by hand, which is precisely the
# work a checker exists to remove.
#
# A checker with a false-positive rate is not a weaker checker; it is a checker
# that gets ignored, and then its true positives are ignored with it.  So this
# file is committed as the work in progress it is, and stays out of
# tests/CMakeLists.txt until the association is exact and the twenty reports are
# each either confirmed or gone.  P9.64's rule cuts both ways: a checker that
# cannot see its defect is worthless, and one that sees defects that are not
# there is worse.

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
# grow.  Each entry must say WHY, so that "undecided" never quietly becomes
# "fine".  Populated on first run against the tree.
KNOWN_UNDECIDED = set()


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
    if t.endswith("*") or t.endswith("&"):          # any pointer / reference
        return PTR8
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
    """kernel name -> list of (file, line, [arg sizes] or None)."""
    packs = {}
    for path in engine_sources():
        try:
            src = open(path, errors="replace").read()
        except OSError:
            continue
        if "PxgKernelIds::" not in src:
            continue
        for m in re.finditer(r"PxgKernelIds::(\w+)", src):
            kid = m.group(1)
            name = ids.get(kid)
            if not name:
                continue
            # the kernelParams block that follows this getCuFunction, within reach
            tail = src[m.end():m.end() + 4000]
            b = re.search(r"PxCudaKernelParam\s+\w+\s*\[\]\s*=\s*\{(.*?)\}\s*;", tail, re.S)
            if not b:
                continue
            args = re.findall(r"PX_CUDA_KERNEL_PARAM2?\(\s*([\w\.\->\[\]]+)\s*\)", b.group(1))
            if not args:
                continue
            sizes = []
            for a in args:
                if not re.match(r"^\w+$", a):     # a.b, p->q, v[i] -- not resolvable here
                    sizes = None
                    break
                s = declared_size(src, a)
                if s is None:
                    sizes = None
                    break
                sizes.append(s)
            line = src[:m.start()].count("\n") + 1
            packs.setdefault(name, []).append((os.path.relpath(path, ROOT), line, sizes))
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
