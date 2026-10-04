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
# HOW A NAME IS RESOLVED (three tiers, P9.83a-6)
# ---------------------------------------------------------------------------
# Until P9.83a-6 a name was resolved over the FLAT TEXT of the whole .cpp, and
# that threw away information C++ already has.  Two functions of one file that
# declare the same name with different types made the name undecidable at EVERY
# launch site in that file, including the ones where only one of the two
# declarations is in scope.  The resolver now asks, in order:
#
#   1. THE ENCLOSING FUNCTION (parameters included).  If the name is declared
#      there, that is the answer -- a single size, or undecidable if the function
#      itself declares it twice with different sizes.  It is deliberately not
#      widened on a disagreement: more candidates cannot resolve an ambiguity.
#   2. THE TRANSLATION UNIT, as before, for file-scope names.
#   3. THE SIBLING HEADERS, and only when neither of the first two found the name
#      AT ALL.
#
# TIER 3 IS THE ONE THAT WAS MEASURED TO BE A NET LOSS IN P9.82 AND IS NOT NOW.
# Widening the FLAT scan to the headers gained one kernel and lost SIXTEEN, because
# a header contributes a second declaration of a name the translation unit also
# declares and the resolver then correctly refuses.  Function scope removes that
# cause: a name declared in the function never reaches tier 3 at all.  Measured
# again after the change -- tier 1 alone: +4 decided, 0 lost; tier 3 on top of it:
# +60 decided, 0 lost.
#
# ---------------------------------------------------------------------------
# WHAT IT STILL CANNOT DECIDE
# ---------------------------------------------------------------------------
# 149 -> 213 kernels now decide, and 37 (was 101) are in KNOWN_UNDECIDED.  That is
# still a recorded blind spot, not a clean bill.  The largest remaining cause is a
# struct passed BY VALUE whose size this file does not know -- the contact-writer
# tails especially -- and deciding those means reading the struct.  One cause can
# never be decided here at all: the void** launch form carries no per-argument
# sizes, and the bridge reads them from this very table, so there is nothing
# independent left to compare.
#
# ALL 64 NEWLY DECIDED KERNELS AGREE WITH THEIR ROWS.  That is worth stating: the
# widening was not a way to make the number look better, it was a test of 64 rows
# that had never been compared to anything, and every one of them passed --
# including the whole PGS/TGS solver and constraint-prep family, boxBoxNphase,
# convexTrimeshNarrowphase, sphereTrimeshNarrowphase and trimeshPlaneNarrowphase.

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
# and the honest state of the tree is that 28 of the kernels with an ABI row have
# one (101 before P9.83a-6 taught the resolver function scope; 37 before P9.93
# made three more kernels VISIBLE; 40 before P9.95 taught SIZES PxVec3; 29 before
# P9.95 taught the declaration regex `**` -- see the notes at the end of the set).
#
# THE REASONS, in the order they will be worth attacking:
#   * a struct passed BY VALUE whose size this table does not know (the
#     PxgFEMContactWriter / PxgSoftBodyContactWriter tails, and the PxgShape and
#     PxBounds3 scalars).  Deciding these means reading the struct, which is the
#     next real step and would take most of the list.
#
#     P9.81 MEASURED WHAT THAT STEP ACTUALLY NEEDS, because the two obvious fixes
#     are not it and the next reader should not spend the afternoon on them.  For
#     sb_heightfieldContactGenLaunch twelve of its thirteen pushed arguments
#     resolve; only `writer` is None.  Adding "PxgFEMContactWriter": 80 to SIZES
#     does not decide it, and neither does teaching the declaration regex the
#     constructor form `Type var(args);` -- both were tried and measured.  The
#     reason is that declared_size is FILE-scoped by design (see its docstring),
#     and PxgNarrowphaseCore.cpp declares the name `writer` TWICE with different
#     types: `PxgFEMContactWriter writer(softBodyCore);` at :1024 and
#     `PxgParticleContactWriter& writer` at :3100.  Two types for one name in one
#     translation unit is exactly the disagreement the resolver must refuse.  So
#     the real step is FUNCTION scope, not a bigger type table -- and doing it
#     without that scoping would make the checker confidently wrong, which is the
#     one outcome worse than undecided.
#   * PxQuat / PxTransform arguments passed by reference, whose sizeof is the
#     referent's.  (PxVec3 is decided since P9.95: SIZES knows it is 12, by value
#     and as the referent of a reference alike.)
#   * the void** launch form (PX_CUDA_KERNEL_PARAM2), which carries no per-argument
#     sizes at all -- the bridge reads them from this very table, so there is
#     nothing independent to compare against and these can never be decided here.
#   * a pack assembled across a branch, or an argument that is a member expression
#     rather than a plain variable.
KNOWN_UNDECIDED = {
    "cloth_SphereContactGenLaunch",
    "cloth_SphereVertexContactGenLaunch",
    "cloth_boxTriangleContactGenLaunch",
    "cloth_boxVertexContactGenLaunch",
    "cloth_convexContactGenLaunch",
    "cloth_convexVertexContactGenLaunch",
    "cloth_heightfieldContactGenLaunch",
    "cloth_heightfieldVertexContactGenLaunch",
    "cloth_meshContactGenLaunch",
    "cloth_meshVertexContactGenLaunch",
    "cloth_planeVertContactGenLaunch",
    "cloth_psContactGenLaunch",
    "cloth_sdfMeshContactGenLaunch",
    "ps_heightfieldCollisonLaunch",
    "ps_meshCollisonLaunch",
    "ps_primitivesCollisionLaunch",
    "ps_sdfMeshCollisonLaunch",
    "sb_clothContactGenLaunch",
    "sb_clothVertContactGenLaunch",
    "sb_heightfieldContactGenLaunch",
    "sb_meshContactGenLaunch",
    "sb_primitiveContactGenLaunch",
    "sb_psContactGenLaunch",
    "sb_sbMidphaseGeneratePairsLaunch",
    "sb_sdfMeshMidphaseGeneratePairsLaunch",
    "sb_selfCollisionMidphaseGeneratePairsLaunch",
    # P9.93 -- NOT a regression: these three were INVISIBLE before, skipped as "no
    # launch site found" because their PxCudaKernelParam block precedes the
    # getCuFunction handle.  P9.93 searches from the function head (and blanks
    # comments and `#if 0`), which made them visible; each falls into a category
    # above, and each pack was checked BY HAND against its row:
    #   convexCoreTetmeshNphase_Kernel32   PxgNarrowphaseCore.cpp:1050 -- `writer`,
    #     the PxgFEMContactWriter by value (9 pointers + PxU32 = 80); pack
    #     {4,8,8,8,8,8,8,8,80} == kCvxCoreTet.
    #   convexCoreClothmeshNphase_Kernel32 PxgNarrowphaseCore.cpp:1189 -- the same
    #     writer; pack {8 x10,4,4,8,80} == kCvxCoreCloth.
    #   ps_stepParticlesLaunch             PxgParticleSystemCore.cpp:2460 --
    #     `mGravity`, a PxVec3 CLASS MEMBER (PxgParticleSystemCore.h:292) passed by
    #     value; pack {8,8,4,4,1,1,1,12} == kPsStepParticles.
    #     (P9.95: decided now -- "PxVec3": 12 in SIZES -- and off this list.)
    "convexCoreTetmeshNphase_Kernel32",
    "convexCoreClothmeshNphase_Kernel32",
    # P9.95 taught SIZES "PxVec3": 12 when wiring applyTGSSubstepGravity, whose pack
    # pushes `gravity`, the `const PxVec3&` PARAMETER of solveContactMultiBlockParallel
    # (PxgTGSCudaSolverCore.cpp:1254, :1362-1367) -> {8,8,12} == kTwoPtrVec3.  That
    # also decided ELEVEN kernels that were on this list, every one of them in
    # AGREEMENT with its row: cloth_preIntegrateLaunch, cloth_stepLaunch,
    # preIntegrationLaunch, preIntegrationLaunchTGS, ps_diffuseParticleCompact,
    # ps_diffuseParticleUpdatePBF, ps_preDiffuseIntegrateLaunch,
    # ps_preIntegrateLaunch, ps_stepParticlesLaunch, sb_gm_preIntegrateLaunch and
    # sb_gm_stepSoftbodyLaunch.  Checked 269 -> 281 (those eleven + the new row),
    # KNOWN_UNDECIDED 40 -> 29.
    #
    # P9.95 (sb_copyOrApplySoftBodyDataDEPRECATED) taught the resolver three things:
    # never to decide on a partial argument list, a call-expression argument sized by its
    # declared return type (method_return_size), and a by-value PxgCudaBuffer object as
    # 32 bytes.  It decides the new row -- {32,4,8,4,8,8,4,4} = 80 -- once it can also read
    # `PxU8** data`, and that double-pointer declaration decided ONE more kernel that was
    # on this list, in agreement with its row:
    #   prepareLostFoundPairs_Stage2  PxgNarrowphaseCore.cpp:342 -- `lostFoundCms` and
    #     `cmArray` are PxsContactManager** locals; pack {8,8,8,8,8,8,8,4,8} == kLostFound2.
    # The full per-kernel decision table was diffed before and after: nothing else moved.
    # Checked 282 -> 284, KNOWN_UNDECIDED 29 -> 28.
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
    # P9.95: three floats, align 4 -- unambiguous by value and as a referent
    "PxVec3": 12,
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
    # resolves to its REFERENT, so the kernel is sized by sizeof(PxVec3) == 12
    # (in SIZES since P9.95; before that it was UNDECIDED instead of falsely
    # accused) -- and preIntegrationLaunch now decides, in agreement with its row.
    if t.endswith("*"):
        return PTR8
    if t.endswith("&"):
        return arg_size(t[:-1])
    # PxgDevicePointer<T> is a one-pointer wrapper: 8.  A PxgCudaBuffer / PxgTypedCudaBuffer<T>
    # is NOT -- it is the 32-byte host OBJECT (mPtr @0, mHeapMemoryAllocator* @8, mSize @16,
    # mStatGroup @24 + pad; PxgCudaBuffer.h:103-107), and PX_CUDA_KERNEL_PARAM(buffer) pushes
    # all 32 bytes.  P9.95: PxgSoftBodyCore.cpp:3134 does exactly that for
    # sb_copyOrApplySoftBodyDataDEPRECATED (core->getSoftBodyBuffer(), a reference to one);
    # this branch used to size it 8, which would have blessed a 56-byte signature row that
    # the shim refuses on every launch.
    if t.startswith("PxgDevicePointer<"):
        return PTR8
    if t.startswith("PxgTypedCudaBuffer<") or t == "PxgCudaBuffer":
        return 32
    base = t.split()[-1] if " " in t else t
    return SIZES.get(base) or SIZES.get(t)


def _param_align(size):
    """paramAlign (PxgVulkanCudaContext.cpp:114) / abiAlign (PxgVulkanKernelAbi.cpp
    :2206): the largest power of two that DIVIDES the size, capped at 16.

    Transcribed as a modulo ladder, which it is.  P9.78b's first draft wrote it as
    `size if size in (1,2,4,8) else 16`, which is right for every power-of-two size
    and wrong for a 12-byte PxVec3 -- the by-value `gravity` argument.  That gave
    16 where the bridge gives 4 and shifted every argument after it, mis-sizing 16
    of the 358 rows and inventing two "unread arguments" that were nothing of the
    kind.  The same wrong copy shipped in check_launch_abi_pack.py and is corrected
    there too."""
    if size == 0:
        return 1
    for a in (16, 8, 4, 2):
        if size % a == 0:
            return a
    return 1


def abi_push_size(row):
    """pxgKernelAbiPushSize, PxgVulkanKernelAbi.cpp:2231, transcribed."""
    off, ma = 0, 1
    for s in row:
        a = _param_align(s)
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
SPAN_CACHE = {}


def _mask_if0(text):
    """`text` with every `#if 0` ... `#endif` region (up to a same-depth `#else` /
    `#elif`, which is live again) blanked to spaces, newlines kept, so offsets and
    line numbers still index the original.  P9.93: PxgPBDParticleSystemCore.cpp:
    757-777 keeps a dead ps_applyDeltaLaunch site under `#if 0` whose pack --
    (particleSystemsd, id, mRelaxationFactor, dt) -- is not the kernel's signature
    at all; once launch sites were searched from the function head it was read as a
    second, live site of the kernel."""
    out = list(text)
    lines = text.split("\n")
    pos, dead, depth = 0, False, 0
    for ln in lines:
        m = re.match(r"\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)", ln)
        kind = m.group(1) if m else None
        if not dead:
            if kind == "if" and re.match(r"\s*0\s*(?://.*|/\*.*)?$", m.group(2)):
                dead, depth = True, 0
        else:
            if kind in ("if", "ifdef", "ifndef"):
                depth += 1
            elif kind == "endif":
                if depth == 0:
                    dead = False
                else:
                    depth -= 1
            elif kind in ("else", "elif") and depth == 0:
                dead = False
        if dead:
            for k in range(pos, pos + len(ln)):
                out[k] = " "
        pos += len(ln) + 1
    return "".join(out)


def _masked(src):
    """`src` with comments and string/char literals replaced by spaces of the same
    length, so brace counting sees only real braces.  Length is preserved because
    every offset computed on this copy indexes the ORIGINAL."""
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                out[k] = " "
            i = j
        elif c in "\"'":
            q, j = c, i + 1
            while j < n and src[j] != q:
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
            for k in range(i, j):
                out[k] = " "
            i = j
        else:
            i += 1
    return "".join(out)


def function_span(src, pos, cache_key):
    """(start, end) of the OUTERMOST brace block containing `pos` -- i.e. the body of
    the function the launch site sits in -- or None when `pos` is at file scope.

    WHY FUNCTION SCOPE (P9.83a-6, and predicted by the note below since P9.82).
    declared_size used to resolve a name over the flat text of the whole .cpp.  Two
    functions of one file that declare the same name with different types then made
    the name undecidable EVERYWHERE, even at a launch site where only one of the two
    declarations is in scope.  That is not a limitation of the data -- C++ already
    says which declaration wins -- it is the resolver throwing the answer away.

    The span is the outermost block because a launch site inside an `if` or a `for`
    still sees the enclosing function's locals; taking the innermost block would
    resolve nothing.  Parameters are picked up too: `start` is the body's opening
    brace, and the parameter list sits just before it, so the search text begins at
    the previous `)`'s matching `(`."""
    key = (cache_key, pos)
    if key in SPAN_CACHE:
        return SPAN_CACHE[key]
    mask = _masked(src)
    # walk back from pos, tracking depth, to the brace that opens the block at depth 1
    depth, i, opens = 0, pos, []
    while i > 0:
        i -= 1
        c = mask[i]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth == 0:
                opens.append(i)
            else:
                depth -= 1
    if not opens:
        SPAN_CACHE[key] = None
        return None
    start = opens[-1]                      # the OUTERMOST open brace still containing pos
    # its matching close
    depth, j = 0, start
    n = len(mask)
    while j < n:
        if mask[j] == "{":
            depth += 1
        elif mask[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    end = min(j + 1, n)
    # widen backwards over the parameter list so a parameter counts as a declaration
    k = src.rfind(")", 0, start)
    if k > 0:
        depth2, t = 0, k
        while t >= 0:
            if mask[t] == ")":
                depth2 += 1
            elif mask[t] == "(":
                depth2 -= 1
                if depth2 == 0:
                    break
            t -= 1
        if t >= 0 and start - t < 4000:
            start = t
    res = (start, end)
    SPAN_CACHE[key] = res
    return res


# WIDENING THE DECLARATION SEARCH TO THE SIBLING HEADERS WAS TRIED AND MEASURED
# TO BE A NET LOSS -- AND P9.83a-6 FIXED THE CAUSE AND PUT IT BACK AS TIER 3.
# The paragraph below is the P9.82 measurement, kept because it is the reason the
# widening is a THIRD tier rather than a wider flat scan.
#
# P9.82 needed it for trimeshPlaneNarrowphase: eighteen of its twenty pushed
# arguments resolve and the two that do not are `mContactStream` and
# `mPatchStream`, CUdeviceptr MEMBERS declared in PxgNarrowphaseCore.h, which this
# resolver never reads.  Appending each .cpp's sibling include/*.h does resolve
# them -- and turns SIXTEEN previously decided kernels UNDECIDED, because the
# headers contribute a second declaration of some name that disagrees with the
# one in the translation unit and declared_size then correctly refuses.
#
# One kernel gained, sixteen lost.  The real fix is the same one P9.81 measured for
# the by-value writers: resolve a name in the ENCLOSING FUNCTION (or class) rather
# than over a flat text, at which point both causes go away together.
#
# THEY DID.  P9.83a-6 implemented function scope and re-measured: the sixteen
# losses do not happen, because a name the translation unit declares never reaches
# the header tier.  trimeshPlaneNarrowphase -- the kernel that motivated the
# attempt -- is among the 60 the header tier now decides.


HEADER_CACHE = {}


def sibling_headers(path):
    """The .h files of the same module as `path` (src/foo.cpp -> include/*.h and
    src/*.h).  Returns [] when `path` is None -- the resolver is then being called
    without a translation unit and has nothing to widen to."""
    if not path:
        return []
    if path in HEADER_CACHE:
        return HEADER_CACHE[path]
    d = os.path.dirname(os.path.abspath(path))
    roots = [d]
    if os.path.basename(d) == "src":
        roots.append(os.path.join(os.path.dirname(d), "include"))
    out = []
    for r in roots:
        if not os.path.isdir(r):
            continue
        for fn in sorted(os.listdir(r)):
            if fn.endswith(".h"):
                out.append(os.path.join(r, fn))
    HEADER_CACHE[path] = out
    return out


def declared_size(src, var, cache_key=None, pos=None):
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
    # *** THE KEY IS THE CALLER'S, NOT id(src). ***  This cache was keyed on
    # `id(src)`, which is only unique while that string OBJECT is alive: the caller
    # reads one file per iteration into the same local, so a freed text can hand its
    # address to the next file's and a cached answer then belongs to the wrong
    # translation unit.  P9.82 hit it for real -- mContactStream resolves to 8 when
    # declared_size is called directly and came back None inside the sweep -- which
    # is a wrong answer, not merely a missing one.  The caller passes its path.
    span = function_span(src, pos, cache_key) if pos is not None else None
    key = (cache_key if cache_key is not None else id(src), var, span)
    if key in DECL_CACHE:
        return DECL_CACHE[key]

    def scan(text):
        """(sizes seen, whether any declaration had an unresolvable type)"""
        # P9.93: a qualifier between the type and the name -- `const void*
        # PX_RESTRICT data` (PxgSimulationCore::setRigidDynamicData) -- hid the
        # declaration, the flat-file fallback then met two unrelated `data`s of
        # unresolvable type, and all five direct-API setters came out UNDECIDED.
        # Dropping the macro changes no size and can only resolve more.
        text = re.sub(r"\bPX_RESTRICT\b", " ", text)
        sizes, unresolved = set(), False
        # P9.95: `\*{0,2}&?`, not `[*&]?` -- a double pointer (`PxU8** data`, the host
        # parameter of copyOrApplySoftBodyDataDEPRECATED) or a reference to a pointer was
        # not a declaration to this regex at all, so the name fell through to the flat file
        # scan and met unrelated `data`s there.  NOT `[*&]{0,2}`: that was tried and reads
        # `(anyArticulationConstraints && c == 0` (PxgTGSCudaSolverCore.cpp:1193) as a
        # declaration of `c` with type `anyArticulationConstraints&&`, which made
        # solveBlockUnified undecidable.  A logical AND is not a reference to a reference.
        for m in re.finditer(r"(?:^|[;{}\(,])\s*((?:const\s+)?[\w:]+(?:\s*<[^<>;]*>)?\s*\*{0,2}&?)\s+"
                             + re.escape(var) + r"\s*(?:=|;|\)|,|\[)", text, re.M):
        # NOTE the leading-delimiter class: it is what keeps `foo(bar, baz)` from
        # reading `bar` as a declaration of type `foo(`.
            t = m.group(1).strip()
            if t.split()[-1] in ("return", "else", "case", "new", "delete"):
                continue
            sz = arg_size(t)
            if sz is None:
                unresolved = True
            else:
                sizes.add(sz)
        return sizes, unresolved

    res = None
    if span is not None:
        sizes, unresolved = scan(src[span[0]:span[1]])
        if sizes or unresolved:
            # THE NAME IS DECLARED IN THIS FUNCTION, so this is the answer -- right or
            # undecidable.  It is deliberately NOT widened to file scope on a
            # disagreement: a name declared twice in one function body really is
            # ambiguous to this resolver, and widening would only add more candidates.
            res = sizes.pop() if (len(sizes) == 1 and not unresolved) else None
            DECL_CACHE[key] = res
            return res
        # not declared in the function at all -> a member or a file-scope name,
        # so fall through to the flat scan, which is what this resolver always did.
    sizes, unresolved = scan(src)
    if sizes or unresolved:
        res = sizes.pop() if (len(sizes) == 1 and not unresolved) else None
        DECL_CACHE[key] = res
        return res

    # TIER 3: THE SIBLING HEADERS, and only where the first two tiers found NOTHING.
    #
    # P9.82 tried widening the flat scan to the sibling include/*.h and measured a
    # NET LOSS -- one kernel gained, SIXTEEN lost -- because a header contributes a
    # second declaration of a name that the translation unit also declares, and
    # declared_size then correctly refuses.  The note that stood here predicted that
    # function scope would remove that cause, and it does: a name declared in the
    # function never reaches this point, so the header can no longer contradict it.
    #
    # The tier is therefore strictly monotone by construction.  It runs ONLY when
    # neither the function nor the translation unit declares the name at all, which
    # is exactly the CUdeviceptr-member case (mContactStream, mPatchStream) the
    # earlier attempt was after.  Measured: it may gain, it cannot lose.
    hsizes, hunresolved = set(), False
    for h in sibling_headers(cache_key):
        try:
            htext = open(h, errors="replace").read()
        except OSError:
            continue
        hs, hu = scan(htext)
        hsizes |= hs
        hunresolved = hunresolved or hu
    res = hsizes.pop() if (len(hsizes) == 1 and not hunresolved) else None
    DECL_CACHE[key] = res
    return res


# ---------------------------------------------------------------------------
# call-expression arguments (P9.95)
# ---------------------------------------------------------------------------
# PX_CUDA_KERNEL_PARAM(core->getSoftBodyBuffer()) (PxgSoftBodyCore.cpp:3134) pushes
# sizeof(<the call's return type>).  It is the only call-expression argument in gpu*/ (a
# grep of every PX_CUDA_KERNEL_PARAM found no other), and until P9.95 the argument regex
# could not match it: findall DROPPED it in silence, so the pack was read as seven
# arguments.  Now the argument regex accepts a trailing `()`, and `obj->method()` /
# `method()` is sized by the RETURN TYPE the method is declared with in gpu*/include.  The
# object's type is not checked: the method is looked up by NAME across those headers, so
# two declarations of one name with different sizes leave the argument undecidable --
# never a guess.
HEADER_DIRS = sorted(
    os.path.join(ROOT, "physx", "source", d, "include")
    for d in os.listdir(os.path.join(ROOT, "physx", "source"))
    if d.startswith("gpu") and os.path.isdir(os.path.join(ROOT, "physx", "source", d, "include")))
METHOD_CACHE = {}


def method_return_size(method):
    """sizeof the value `method()` returns, from its declarations in gpu*/include, or None."""
    if method in METHOD_CACHE:
        return METHOD_CACHE[method]
    sizes, unresolved = set(), False
    for d in HEADER_DIRS:
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".h"):
                continue
            try:
                text = _masked(open(os.path.join(d, fn), errors="replace").read())
            except OSError:
                continue
            for m in re.finditer(r"(?:^|[;{}\s])((?:const\s+)?[\w:]+(?:\s*<[^<>;]*>)?\s*\*{0,2}&?)\s+"
                                 + re.escape(method) + r"\s*\(\s*\)", text, re.M):
                t = m.group(1).strip()
                if t.split()[-1] in ("return", "else", "case", "new", "delete"):
                    continue
                sz = arg_size(t)
                if sz is None:
                    unresolved = True
                else:
                    sizes.add(sz)
    res = sizes.pop() if (len(sizes) == 1 and not unresolved) else None
    METHOD_CACHE[method] = res
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
        # P9.93: the chain is followed over the MASKED text (comments and literals
        # blanked, offsets preserved).  Commented-out launch code is common in these
        # files -- PxgParticleSystemCore.cpp:1470-1500 keeps a whole old
        # ps_averageVertsLaunch site under `//` -- and once the block search below
        # started at the function head it began reading those dead sites as live
        # ones.  Only real code launches anything.
        code = _mask_if0(_masked(src))
        # every handle assignment, including the ternary two-id form
        for hm in re.finditer(r"(\w+)\s*=\s*([^;]*?getCuFunction\s*\([^;]*?)\s*;", code, re.S):
            handle, rhs = hm.group(1), hm.group(2)
            kids = re.findall(r"PxgKernelIds::(\w+)", rhs)
            if not kids:
                continue
            # the launchKernel that uses this handle, before the handle is reassigned
            tail = code[hm.end():]
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
            #
            # P9.93: and the search starts at the ENCLOSING FUNCTION, not at the
            # handle.  PxgSimulationCore::getRigidDynamicData declares every case's
            # block BEFORE its `CUfunction kernelFunction = ...getCuFunction(...)`,
            # and so do updateBodies and others; searching only after the handle
            # found no block there, and those kernels were silently skipped as "no
            # launch site found" -- five wired getters and updateBodiesLaunch among
            # them.  C++ scoping makes the last block of that name before the launch
            # inside the function the one the launch uses.
            launchAbs = hm.end() + lm.start()
            span = function_span(src, hm.start(), path)
            searchFrom = span[0] if span else 0
            blocks = list(re.finditer(r"PxCudaKernelParam\s+" + re.escape(pname) +
                                      r"\s*\[\]\s*=\s*\{(.*?)\}\s*;",
                                      code[searchFrom:launchAbs], re.S))
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
            # P9.95: an optional trailing `()` -- a call expression (see method_return_size).
            args = re.findall(r"PX_CUDA_KERNEL_PARAM2?\(\s*\(?\s*([\w\.\->\[\]]+(?:\(\))?)\s*\)?\s*\)",
                              block)
            if not args:
                continue
            # P9.95: NEVER DECIDE ON A PARTIAL ARGUMENT LIST.  Every argument form the regex
            # above does not know is dropped by findall without a word, and the pack then
            # looks short -- that is how core->getSoftBodyBuffer() vanished.  If the block
            # has more PX_CUDA_KERNEL_PARAM's than the regex captured, the pack is UNDECIDED.
            if len(re.findall(r"PX_CUDA_KERNEL_PARAM2?\(", block)) != len(args):
                sizes = None
                args = []
            else:
                sizes = []
            for a in args:
                cm = re.match(r"^(?:\w+->|\w+\.)?(\w+)\(\)$", a)
                if cm:
                    sz = method_return_size(cm.group(1))
                elif re.match(r"^\w+$", a):
                    sz = declared_size(src, a, path, hm.start())
                else:
                    sz = None
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
