#!/usr/bin/env python3
# check_launch_abi_pack.py -- P9.77; PER ARGUMENT since P9.97b.  The FIFTEENTH static
# checker.
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS
# ---------------------------------------------------------------------------
# Every mapped kernel's launch-ABI row (bridge/PxgVulkanKernelAbi.cpp) is the bridge's
# only description of the kernel's CUDA argument list, and nothing else in the build
# compares it with anything.  Its defect has happened twice:
#
#   P9.76b  sb_solveRigidSoftAttachmentLaunchTGS carried the 72-byte PGS row while
#           PxgSoftBodyCore.cpp pushes ELEVEN arguments = 80 bytes.  The shim's exact
#           blobSize == pushConstantSize gate refused every launch, in silence, with the
#           fixtures, the suite and every scene green -- a fixture dispatches the shader
#           with its own blob and never passes the gate.
#   P9.74   the 176-byte convex-core cloth row could only be shown correct by watching
#           the SCENE move, because nothing checked the arithmetic.
#
# ---------------------------------------------------------------------------
# WHAT THE TWO LAUNCH FORMS MAKE RUNTIME-RELEVANT (P9.97b)
# ---------------------------------------------------------------------------
# * SIZED form (PxCudaKernelParam {ptr, size}; PxgVulkanCudaContext.cpp sized
#   launchKernel).  The blob is laid out from the HOST's sizes.  The row is consulted
#   through its TOTAL (the shim gate) and through hostPtrMask (translateHostPtrs indexes
#   the HOST offsets by ROW index and requires row.argSizes[i] == 8).
# * void** form (PX_CUDA_KERNEL_PARAM2; broadphase, AABB manager, articulation).  Layout
#   AND copy length come from the ROW: row.argSizes[i] bytes are copied from host
#   kernelParams[i].  The gate compares row with row, so it can never fire; a wrong entry
#   silently reads too few or too many bytes of a host variable.
# * Every row is also check_push_block.py's reference for the shader's push block.
#
# So the invariant chain is  CUDA signature == row == host pack  (this file)  and
# row == shader push block (check_push_block.py's join).
#
# P9.99: the signature parser also keeps the CUDA NAMES (param_name, signature_names,
# struct_field_names, HOST_ARG_FIELD_NAMES) for check_push_block.py's name leg; its entry
# point is arg_field_names.  They are side tables only: every decision of this file is
# unchanged (PXG_ABI_PACK_DUMP byte-identical before and after).  Names that cannot be
# decided are never guessed: arg_field_names returns None when two declarations with the
# same sizes disagree on the names, and '?' for a parameter or struct field it cannot name.
#
# ---------------------------------------------------------------------------
# WHAT IT CHECKS (P9.97b; before it, TOTALS ONLY)
# ---------------------------------------------------------------------------
# Until P9.97b this file compared abi_push_size(row) with abi_push_size(host) and
# printed "N kernels checked argument-by-argument".  It was not.  P9.97 measured seven
# blind spots against nine row mutants, and the two existing ABI checkers together let
# five through, two of them runtime-fatal:
#
#   B1  totals only: kTwoPtrVec3 {8,8,12} -> {8,8,8,4} passed (32 == 32)
#   B2  100 of 415 rows skipped without a word ("no launch site found"), 95 of them the
#       void** kernels whose row is the ONLY layout authority; gpuarticulation/src was not
#       even read
#   B3  argCount ignored: `uint32_t(sizeof(kTwoPtr))` on the applyTGSSubstepGravity row
#       makes the shim refuse every launch, and both checkers stayed green
#   B4  hostPtrMask never read: `1u << 15` on kCvxTriCore refuses every launch of four
#       narrowphase kernels, and both checkers stayed green
#   B5  "function scope" was the OUTERMOST brace -- `namespace physx {` in six files
#   B6  `static T x` and `T x(args);` were not declarations; the name fell through to
#       file scope and bound to ANOTHER function's declaration (ps_diffuseParticle*:
#       `count` from :246 instead of `static PxU32 count` at :1184 -- right by luck)
#   B7  only the first launchKernel per handle was read (14 sites never looked at)
#
# It now enforces, every run:
#
#   I1 ROW INTEGRITY  (read_abi, shared with check_push_block.py)
#      * the row models array[:argCount], which is what the bridge uses;
#      * `sizeof(X)` names the row's own array; a literal count equals the array length;
#      * every row's array was parsed -- an unparsed one FAILS, it is never skipped;
#      * entries are decimal literals and none is 0; no cudaName appears twice (the
#        bridge's pxgFindKernelAbi takes the FIRST; a dict would keep the LAST).
#   I2 HOST == ROW, PER ARGUMENT, PER LAUNCH SITE, in both forms
#      SEQ-DIFF (sized, equal total -- runtime-invisible, only this sees it), DIFF.
#   I3 TAIL-EXTRA only by NAMED waiver, any form, never a form-based allowance.
#   I4 HOST-POINTER MASK, size only: bit < argCount, row[i] == 8, and argument i is 8
#      bytes at every decided site.  NOT "resolves to a pointer declaration":
#      dmaArticulationResidual passes `CUdeviceptr ptr` (PxgTGSCudaSolverCore.cpp,
#      PxgCudaSolverCore.cpp) and that clause would fail the pristine tree.
#   I5 ROW == CUDA __global__ SIGNATURE (the cheap leg, ~1 s; P9.97's review measured
#      that it alone kills seven of the nine row mutants with no host resolver at all).
#      For a kernel launched in the void** form anywhere: exactly.  For a sized-only
#      kernel: exactly, or by name in KNOWN_ROW_FOLLOWS_HOST, whose rows deliberately
#      follow the HOST pack (every site sized and EQ).  Unresolved signatures FAIL.
#   I6 RATCHETS, each may shrink and never grow, and a stale entry FAILS:
#      KNOWN_UNDECIDED, KNOWN_NO_SITE, KNOWN_TAIL_EXTRA, KNOWN_ROW_FOLLOWS_HOST,
#      KNOWN_SIG_UNDECIDED.
#
# A SELFTEST runs first, every time (or alone with --selftest-only): Part A applies
# deliberately wrong rows to an in-memory scratch copy of the real ABI file and demands
# the named verdict for the named kernel; Part B feeds literal C++ snippets to the real
# resolver, each built so that a wrong binding gives a different size.  A selftest
# anchor that is missing is SELFTEST BROKEN: re-anchor it, never delete the case.
#
# PXG_ABI_PACK_DUMP=<file> writes the per-kernel decision table as JSON (a scratch path,
# never the tree) -- the before/after diff every resolver change must be shown with.
# --abi <file> checks a scratch copy of the ABI file instead of the tree's.
#
# ---------------------------------------------------------------------------
# THE RESOLVER, AND WHAT IT REFUSES TO PRETEND
# ---------------------------------------------------------------------------
# Recovering a C++ variable's type is a C++ problem, and this is a set of regexes.  So:
# resolve what is unambiguous, count everything else as UNDECIDED rather than as
# passing (P9.64: a checker that quietly skips what it cannot parse is worse than none,
# because it is counted).  A name is resolved at the PARAMS-BLOCK position:
#
#   1L LEXICAL: the declarations VISIBLE there -- in the enclosing function body (its
#      parameters included; function_body_span skips namespace, class and control
#      blocks) and not inside a block that has already closed.  The INNERMOST scope
#      wins (C++ shadowing); several declarations in that one scope must agree on a
#      size.  `static T x`, `T x(args)` and `T x{...}` are declarations.
#   2  THE TRANSLATION UNIT, flat, only when NO declaration is visible.
#   3  THE SIBLING HEADERS, only when neither found the name at all.
#
# Then a size: SIZES, a pointer 8, a reference its referent, `X::Enum` 4 (PhysX enums
# have no fixed underlying type; 4 on every target this builds for), PxgDevicePointer 8,
# a PxgCudaBuffer OBJECT 32, and a by-value struct read from its header by struct_layout
# -- which REFUSES on a base class, PX_ALIGN_PREFIX, a union, a bitfield, `#if` in the
# body, an array of non-literal size or any member it cannot size.  Struct sizing is
# only sound WITH the lexical tier: P9.97 measured it without, and `writer` bound to the
# 72-byte PxgParticleContactWriter& of another function gave 23 false accusations.
#
# FITNESS.  Every false accusation this file has ever made was a bug in this file (P9.77
# had twenty of twenty-one; P9.97 had twenty-three); every one is a way a plausible
# parser lies -- proximity binding, a reference sized as a pointer, the first block
# instead of the last, inner parens, commented-out parameters, a call expression dropped
# by findall, a declaration in a closed block.  The selftest pins the ones that matter.

import bisect
import collections
import json
import os
import re
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)          # sibling import, the check_block_coverage.py precedent

ROOT = os.path.abspath(os.path.join(TOOLS, "..", "..", "..", ".."))
SRC = os.path.join(ROOT, "physx", "source")
ABI = os.path.join(SRC, "gpuslang", "bridge", "PxgVulkanKernelAbi.cpp")
NAMES = os.path.join(SRC, "gpucommon", "include", "PxgKernelNames.h")
ENGINE_DIRS = [
    os.path.join(SRC, "gpusimulationcontroller", "src"),
    os.path.join(SRC, "gpunarrowphase", "src"),
    os.path.join(SRC, "gpusolver", "src"),
    os.path.join(SRC, "gpubroadphase", "src"),
    os.path.join(SRC, "gpucommon", "src"),
    os.path.join(SRC, "gpuarticulation", "src"),   # B2: fifty void** kernels launch here
]
HEADER_DIRS = sorted(
    os.path.join(SRC, d, "include")
    for d in os.listdir(SRC)
    if d.startswith("gpu") and os.path.isdir(os.path.join(SRC, d, "include")))
# Member and struct lookups also read the public headers and lowleveldynamics/shared
# (ArticulationConstraintProcessingConfig, whose VelLimit::Enum member mDoVelLimit is
# pushed by value).
MEMBER_HEADER_DIRS = HEADER_DIRS + [os.path.join(ROOT, "physx", "include"),
                                    os.path.join(SRC, "lowleveldynamics", "shared")]

# ---------------------------------------------------------------------------
# RATCHETS.  Each may SHRINK, never grow, and an entry that no longer applies FAILS.
# ---------------------------------------------------------------------------
# Kernels whose host pack this checker cannot decide.  Was 28 at P9.95 (101 before
# P9.83a-6); P9.97b's lexical tier, struct layout, void** forms, member arguments and
# `X::Enum` decide every one of them, each in AGREEMENT with its row.
KNOWN_UNDECIDED = set()

# Rows with no launch site this parser can follow -- formerly a SILENT skip of 100.  Each
# is checked against its CUDA signature by the signature leg, and each pack was checked
# by hand:
KNOWN_NO_SITE = {
    # id taken from the kernelIDList[] table, PxgSimulationCore.cpp (getD6JointData);
    # pack {8,8,4,8,4,8,4} == kD6JointGet
    "getD6JointForces": "id via the kernelIDList[] table (PxgSimulationCore.cpp)",
    "getD6JointTorques": "id via the kernelIDList[] table (PxgSimulationCore.cpp)",
    # id passed as an argument into removeLostPairsGpu (PxgNarrowphaseCore.cpp); {8,1}
    "removeContactManagers_Stage5": "id passed into removeLostPairsGpu (PxgNarrowphaseCore.cpp)",
    "removeContactManagers_Stage5_CvxTri": "id passed into removeLostPairsGpu (PxgNarrowphaseCore.cpp)",
    # handle conditionally re-assigned to UPDATE_BODIES_DIRECT_API (PxgSimulationCore.cpp);
    # {8} == kOnePtr
    "updateBodiesLaunch": "handle conditionally re-assigned inside an if (PxgSimulationCore.cpp)",
    # 5.12: ids taken from the kernelIds[2] table, PxgSimulationCore::ovdSnapshotRigidDynamicForces;
    # PX_CUDA_KERNEL_PARAM pack (CUdeviceptr x3, PxU32) {8,8,8,4} == kRigidGet
    "getRigidDynamicForce": "id via the kernelIds[2] table (PxgSimulationCore.cpp ovdSnapshotRigidDynamicForces)",
    "getRigidDynamicTorque": "id via the kernelIds[2] table (PxgSimulationCore.cpp ovdSnapshotRigidDynamicForces)",
    # 5.12: launched through the variadic PxgKernelLauncher::launchKernelThreadsAutoPackParameters
    # (PxgKernelLauncher.h), whose PxCudaKernelParam pack is the argument types in order:
    #   (CUdeviceptr x3, PxU32 nbBodies, float oneOverDt) {8,8,8,4,4} == kRigidComputeAccel
    #   (CUdeviceptr x2, PxU32 nbToCopy)                  {8,8,4}     == kRigidCopyPrevVel
    "computeRigidBodyAccelerations": "variadic launchKernelThreadsAutoPackParameters (PxgSimulationCore.cpp)",
    "copyRigidBodyVelocitiesToPrevious": "variadic launchKernelThreadsAutoPackParameters (PxgSimulationCore.cpp)",
}

# Host packs LONGER than the row whose extra arguments sit in the blob's tail padding
# (push(host) == push(row)).  Allowed only by name, in any form.
KNOWN_TAIL_EXTRA = {
    "propagateStaticSolverBodyVelocities":
        "row = CUDA signature (solverMultiBlock.cu, 5 args); the host reuses solveStaticBlock's "
        "6-argument pack (PxgCudaSolverCore.cpp), so the stray doFriction byte @28 lies in the "
        "32-byte blob's tail pad.  Upstream candidate: documented, never reported.",
}

# Sized-only kernels whose row deliberately follows the HOST pack rather than the CUDA
# signature (PxgVulkanKernelAbi.cpp's header: "verbatim except these").  Each needs every
# launch site sized and EQ; a void** kernel may never be here (its row IS the copy length).
KNOWN_ROW_FOLLOWS_HOST = {
    "propagateStaticSolverBodyVelocitiesTGS":
        "the host pushes solveStaticBlockTGS's 7-argument pack; signature has 5",
    "ps_updateUserBufferLaunch":
        "the host pushes a trailing PxU32 the 2-pointer signature lacks",
    "removeContactManagers_Stage1": "one 2-argument pack fired at all five stages; signature has 1",
    "removeContactManagers_Stage2": "one 2-argument pack fired at all five stages; signature has 1",
    "removeContactManagers_Stage3": "one 2-argument pack fired at all five stages; signature has 1",
    "removeContactManagers_Stage4": "one 2-argument pack fired at all five stages; signature has 1",
}

# Kernels whose CUDA signature this file cannot size.  Empty: ps_solveShapes' `#if
# ENABLE_PLASTIC_DEFORMATION` inside its parameter list is decided by _mask_pp.
KNOWN_SIG_UNDECIDED = set()

# Field layouts for HOST-SHAPED by-value arguments (row != signature), which the join in
# check_push_block.py flattens, keyed (kernel, argument index) -> [(offset, size, kind)].
# Empty since 5.12: its only entry was sb_copyOrApplySoftBodyDataDEPRECATED's argument 0,
# the 32-byte PxgTypedCudaBuffer<PxgSoftBody> OBJECT the host pushed by value (P9.95), and
# 5.12 removed that kernel.  check_push_block.py's selftest keeps the mechanism awake with
# a synthetic entry of the same shape.
HOST_ARG_FIELDS = {}
# P9.99: the same fields' NAMES (and types) for check_push_block.py's name leg, parallel to
# HOST_ARG_FIELDS entry by entry.
HOST_ARG_FIELD_NAMES = {}

# ---------------------------------------------------------------------------
# sizes
# ---------------------------------------------------------------------------
PTR8 = 8
SIZES = {
    "CUdeviceptr": PTR8,
    "PxU64": 8, "PxI64": 8, "uint64_t": 8, "int64_t": 8, "size_t": 8,
    "PxU32": 4, "PxI32": 4, "uint32_t": 4, "int32_t": 4, "PxReal": 4, "PxF32": 4,
    "float": 4, "int": 4, "unsigned": 4, "PxU16": 2, "PxU8": 1, "bool": 1,
    # P9.95: three floats, align 4 -- unambiguous by value and as a referent
    "PxVec3": 12,
}
FLOAT_TYPES = ("float", "PxReal", "PxF32")
# Vector types: (size, alignment, component kind, component count).  The CUDA built-ins
# have no header here; the PhysX classes are templates struct_layout cannot read.
VECTORS = {
    "float4": (16, 16, "f", 4), "float2": (8, 8, "f", 2),
    "uint4": (16, 16, "i", 4), "int4": (16, 16, "i", 4),
    "uint2": (8, 8, "i", 2), "int2": (8, 8, "i", 2),
    "PxVec3": (12, 4, "f", 3), "PxVec4": (16, 4, "f", 4), "PxQuat": (16, 4, "f", 4),
}


def _norm_type(t):
    t = re.sub(r"\b(const|static|volatile|mutable|PX_RESTRICT|__restrict__)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def arg_size(decl_type):
    """sizeof one pushed HOST argument of declared type `decl_type`, or None."""
    t = _norm_type(decl_type)
    if not t:
        return None
    # A POINTER is 8 bytes.  A REFERENCE IS NOT: PX_CUDA_KERNEL_PARAM(x) pushes sizeof(x),
    # and for `const PxVec3& gravity` that is 12 (P9.77's preIntegrationLaunch control).
    if t.endswith("*"):
        return PTR8
    if t.endswith("&"):
        return arg_size(t[:-1])
    if t.startswith("PxgDevicePointer<"):
        return PTR8
    # The 32-byte host OBJECT (mPtr, allocator*, mSize, statGroup + pad), never 8 (P9.95).
    if t.startswith("PxgTypedCudaBuffer<") or t == "PxgCudaBuffer":
        return 32
    base = t.split()[-1] if " " in t else t
    sz = SIZES.get(base) or SIZES.get(t)
    if sz:
        return sz
    # PhysX enums carry no fixed underlying type; 4 on gcc/clang/nvcc.  Only the
    # `X::Enum` spelling, never a guess about an arbitrary name.
    if re.search(r"::Enum$", base):
        return 4
    sl = struct_layout(base)
    return sl[0] if sl else None


def _param_align(size):
    """paramAlign (PxgVulkanCudaContext.cpp) / abiAlign (PxgVulkanKernelAbi.cpp): the
    largest power of two that DIVIDES the size, capped at 16 -- a modulo ladder (P9.78b's
    `size if size in (1,2,4,8) else 16` gave a 12-byte PxVec3 alignment 16)."""
    if size == 0:
        return 1
    for a in (16, 8, 4, 2):
        if size % a == 0:
            return a
    return 1


def arg_layout(sizes):
    """([(offset, size)] per argument, total) exactly as the bridge lays a blob out."""
    out, off, ma = [], 0, 1
    for s in sizes:
        a = _param_align(s)
        ma = max(ma, a)
        off = (off + a - 1) & ~(a - 1)
        out.append((off, s))
        off += s
    return out, (off + ma - 1) & ~(ma - 1)


def abi_push_size(row):
    """pxgKernelAbiPushSize (PxgVulkanKernelAbi.cpp), transcribed."""
    return arg_layout(row)[1]


# ---------------------------------------------------------------------------
# text masking
# ---------------------------------------------------------------------------
def _masked(src):
    """`src` with comments and string/char literals replaced by spaces of the same
    length (newlines kept), so every offset still indexes the ORIGINAL."""
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
                if out[k] != "\n":
                    out[k] = " "
            i = j
        elif c in "\"'":
            q, j = c, i + 1
            while j < n and src[j] != q and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
            for k in range(i, j):
                out[k] = " "
            i = j
        else:
            i += 1
    return "".join(out)


def _strip_comments(src):
    """Comments blanked, STRING LITERALS KEPT (the ABI rows' names are literals)."""
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
                if out[k] != "\n":
                    out[k] = " "
            i = j
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"' and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            i = j + 1
        else:
            i += 1
    return "".join(out)


def _mask_if0(text):
    """`text` with every `#if 0` ... `#endif` region (up to a same-depth `#else`/`#elif`)
    blanked.  Kept for callers outside this file; this file uses _mask_pp."""
    return _mask_pp(text, evaluate_names=False)


_PP = re.compile(r"\s*#\s*(if|ifdef|ifndef|elif|else|endif|define)\b(.*)")


def _mask_pp(text, evaluate_names=True):
    """`text` with the DEAD preprocessor branches blanked (offsets and newlines kept), and
    the conditional directive lines themselves blanked too.

    Decided: `#if <int>` and -- with evaluate_names -- `#if NAME` / `#elif NAME` where
    NAME was `#define`d to an integer earlier in the same live text.  That is what the
    real files need: `#define USE_NEW_LAUNCH_FUNCTION 1` + `#if USE_NEW_LAUNCH_FUNCTION`
    picks the void** EPILOG (PxgArticulationCore.cpp, PxgCudaBroadPhaseSap.cpp,
    PxgAABBManager.cpp), and `#define ENABLE_PLASTIC_DEFORMATION 0` drops two parameters
    from ps_solveShapes' signature (particlesystem.cu).  Anything else (#ifdef, defined(),
    an expression) is UNDECIDED and keeps BOTH branches live, as before.

    P9.93's reason for masking at all: PxgPBDParticleSystemCore.cpp keeps a dead
    ps_applyDeltaLaunch site under `#if 0` whose pack is not the kernel's signature."""
    defines = {}
    out = list(text)
    pos = 0
    # frames: [parent_live, live, decided, taken]
    stack = []

    def live_now():
        return stack[-1][1] if stack else True

    def value(expr):
        e = re.sub(r"//.*$", "", expr).strip()
        e = re.sub(r"/\*.*?\*/", "", e).strip()
        m = re.match(r"^\(?\s*(\d+)\s*\)?$", e)
        if m:
            return int(m.group(1)) != 0
        if evaluate_names:
            m = re.match(r"^\(?\s*(\w+)\s*\)?$", e)
            if m and m.group(1) in defines:
                return defines[m.group(1)] != 0
        return None

    for ln in text.split("\n"):
        m = _PP.match(ln)
        kind = m.group(1) if m else None
        blank_line = False
        if kind == "if" or kind in ("ifdef", "ifndef"):
            parent = live_now()
            v = value(m.group(2)) if kind == "if" else None
            if v is None:
                stack.append([parent, parent, False, False])
            else:
                stack.append([parent, parent and v, True, v])
            blank_line = True
        elif kind == "elif" and stack:
            f = stack[-1]
            v = value(m.group(2))
            if f[2] and f[3]:
                f[1] = False
            elif f[2] and v is not None:
                f[1] = f[0] and v
                f[3] = v
            else:
                f[1], f[2] = f[0], False
            blank_line = True
        elif kind == "else" and stack:
            f = stack[-1]
            f[1] = f[0] and not f[3] if f[2] else f[0]
            blank_line = True
        elif kind == "endif" and stack:
            stack.pop()
            blank_line = True
        elif kind == "define" and live_now():
            dm = re.match(r"\s*(\w+)\s+(\d+)\s*(?://.*)?$", m.group(2))
            if dm:
                defines[dm.group(1)] = int(dm.group(2))
        if blank_line or not live_now():
            for k in range(pos, pos + len(ln)):
                out[k] = " "
        pos += len(ln) + 1
    return "".join(out)


class Source(object):
    """One C++ file: the text, its masked copy (comments, literals and dead preprocessor
    branches blanked, offsets preserved) and a brace table, computed ONCE."""

    def __init__(self, path, src):
        self.path = path
        self.src = src
        mask = _mask_pp(_masked(src))
        # PX_RESTRICT changes no size; blanking it lets `const void* PX_RESTRICT data`
        # (P9.93) be read as a declaration.
        self.mask = re.sub(r"\bPX_RESTRICT\b", lambda m: " " * len(m.group(0)), mask)
        opens, closes, parents, stack = [], [], [], []
        for i, c in enumerate(self.mask):
            if c == "{":
                parents.append(stack[-1] if stack else -1)
                stack.append(len(opens))
                opens.append(i)
                closes.append(len(self.mask))
            elif c == "}" and stack:
                closes[stack.pop()] = i
        self.opens, self.closes, self.parents = opens, closes, parents

    def line(self, pos):
        return self.src.count("\n", 0, pos) + 1

    def block_at(self, pos):
        """Index of the innermost brace block strictly containing `pos`, or -1."""
        k = bisect.bisect_left(self.opens, pos) - 1
        while k >= 0 and self.closes[k] < pos:
            k = self.parents[k]
        return k

    def function_body(self, pos):
        """(param_open, body_open, body_close) of the innermost FUNCTION BODY containing
        `pos`, or None.  B5: not the outermost brace (a namespace in six files), and not a
        class, namespace, extern "C", lambda or control-statement block."""
        k = self.block_at(pos)
        m = self.mask
        while k >= 0:
            o = self.opens[k]
            j = o - 1
            while True:
                while j >= 0 and m[j].isspace():
                    j -= 1
                w = re.search(r"(\w+)$", m[max(0, j - 40):j + 1])
                if w and w.group(1) in ("const", "override", "PX_OVERRIDE", "noexcept", "final"):
                    j -= len(w.group(1))
                    continue
                break
            if j >= 0 and m[j] == ")":
                depth, t = 0, j
                while t >= 0:
                    if m[t] == ")":
                        depth += 1
                    elif m[t] == "(":
                        depth -= 1
                        if depth == 0:
                            break
                    t -= 1
                u = t - 1
                while u >= 0 and m[u].isspace():
                    u -= 1
                w = re.search(r"(\w+)$", m[max(0, u - 40):u + 1])
                if t >= 0 and w and w.group(1) not in ("if", "for", "while", "switch", "catch",
                                                       "return", "sizeof"):
                    return (t, o, self.closes[k])
            k = self.parents[k]
        return None


_SOURCES = {}


def load_source(path):
    if path not in _SOURCES:
        try:
            src = open(path, errors="replace").read()
        except OSError:
            src = ""
        _SOURCES[path] = Source(path, src)
    return _SOURCES[path]


# ---------------------------------------------------------------------------
# struct layout (by-value aggregates)
# ---------------------------------------------------------------------------
_HDR_TEXTS = None


def _header_texts():
    global _HDR_TEXTS
    if _HDR_TEXTS is None:
        _HDR_TEXTS = []
        for d in MEMBER_HEADER_DIRS:
            for dp, _dn, fns in os.walk(d):
                for fn in sorted(fns):
                    if fn.endswith(".h"):
                        try:
                            t = _masked(open(os.path.join(dp, fn), errors="replace").read())
                        except OSError:
                            continue
                        _HDR_TEXTS.append(re.sub(r"\bPX_RESTRICT\b", " ", t))
    return _HDR_TEXTS


_STRUCT_CACHE = {}
_STRUCT_NAMES = {}      # P9.99: name -> [(dotted field name, type)] parallel to struct_fields' fields
_ALIGN_OVERRIDE = {"CUdeviceptr": 8, "size_t": 8}


def _member_names(t, nsub):
    """[(suffix, type)] for the `nsub` leaf fields _member_shape gives a member of type `t`:
    '' for a scalar or pointer, '.x'.. for a vector, '.<field>' for a nested struct."""
    t = _norm_type(t or "")
    if (t.split() or [""])[-1] in VECTORS:
        return [("." + "xyzw"[i], t) for i in range(nsub)]
    if t.endswith("*") or nsub == 1:
        return [("", t)]
    sub = struct_field_names(t.split()[-1])
    if sub and len(sub) == nsub:
        return [("." + n, ty) for (n, ty) in sub]
    return [(".?", None)] * nsub


def struct_field_names(name):
    """[(dotted field name, type)] parallel to struct_fields(name)[2], or None (P9.99)."""
    if struct_fields(name) is None:
        return None
    return _STRUCT_NAMES.get(name)


def _member_shape(t):
    """(size, align, [(off, size, kind)]) of a struct member type, or None."""
    t = _norm_type(t)
    if t.endswith("*"):
        return 8, 8, [(0, 8, "p")]
    if t in VECTORS:
        sz, al, kind, cnt = VECTORS[t]
        csz = sz // cnt if t != "PxVec3" else 4
        return sz, al, [(i * csz, csz, kind) for i in range(cnt)]
    base = t.split()[-1]
    if base in SIZES:
        sz = SIZES[base]
        return sz, _ALIGN_OVERRIDE.get(base, sz), [(0, sz, "f" if base in FLOAT_TYPES else "i")]
    if re.search(r"::Enum$", base):
        return 4, 4, [(0, 4, "i")]
    sub = struct_fields(base)
    return sub


def struct_fields(name):
    """(size, align, [(offset, size, kind)]) of `struct name` from the headers, or None.

    REFUSES (None) on: a base class, PX_ALIGN_PREFIX, a union, a bitfield, `#if` in the
    body, an array of non-literal size, any member it cannot size, or two definitions
    that disagree.  Only depth-0 data members count; member functions are dropped."""
    if name in _STRUCT_CACHE:
        return _STRUCT_CACHE[name]
    _STRUCT_CACHE[name] = None              # recursion guard
    found = set()
    names_of = {}                           # P9.99: layout -> {field-name tuples}
    for text in _header_texts():
        for m in re.finditer(r"\bstruct\s+(PX_ALIGN_PREFIX\s*\(\s*\d+\s*\)\s*)?" + re.escape(name) +
                             r"\s*(:[^{;]*)?\{", text):
            if m.group(1) or m.group(2):
                found.add(None)
                continue
            i = m.end()
            depth, j = 1, i
            while j < len(text) and depth:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            body = text[i:j - 1]
            tail = text[j:j + 40]
            if (re.search(r"^\s*#\s*(if|ifdef|ifndef)", body, re.M) or re.search(r"\bunion\b", body)
                    or re.match(r"\s*PX_ALIGN_SUFFIX", tail)):
                found.add(None)
                continue
            flat, d = [], 0
            for ch in body:
                if ch == "{":
                    d += 1
                    if d == 1:
                        flat.append(";")
                    continue
                if ch == "}":
                    d -= 1
                    continue
                if d == 0:
                    flat.append(ch)
            off, ma, fields, fnames, ok = 0, 1, [], [], True
            for decl in "".join(flat).split(";"):
                decl = re.sub(r"^\s*(public|private|protected)\s*:", "", decl.strip()).strip()
                if not decl or "(" in decl or re.match(
                        r"(friend|typedef|using|static|enum|template|struct|class)\b", decl):
                    continue
                decl = _norm_type(decl)
                if ":" in decl.replace("::", ""):
                    ok = False                       # bitfield
                    break
                mm = re.match(r"^((?:[\w:]+(?:\s*<[^<>]*>)?)\s*\**)\s*(\w+(?:\s*,\s*\**\s*\w+)*)"
                              r"\s*(\[\s*(\w+)\s*\])?$", decl)
                if not mm:
                    ok = False
                    break
                shape = _member_shape(mm.group(1))
                if shape is None:
                    ok = False
                    break
                sz, al, sub = shape
                subn = _member_names(mm.group(1), len(sub))
                n = 1
                if mm.group(3):
                    if not mm.group(4).isdigit():
                        ok = False
                        break
                    n = int(mm.group(4))
                for vn in mm.group(2).split(","):
                    vn = vn.strip().lstrip("*").strip()
                    off = (off + al - 1) & ~(al - 1)
                    for e in range(n):
                        fields.extend((off + e * sz + so, ss, sk) for (so, ss, sk) in sub)
                        fnames.extend((vn + ("[%d]" % e if n > 1 else "") + sn, ty) for (sn, ty) in subn)
                    off += sz * n
                    ma = max(ma, al)
            if not ok:
                found.add(None)
                continue
            found.add(((off + ma - 1) & ~(ma - 1), ma, tuple(fields)))
            names_of.setdefault(((off + ma - 1) & ~(ma - 1), ma, tuple(fields)), set()).add(tuple(fnames))
    res = found.pop() if len(found) == 1 and None not in found else None
    if res is not None:
        nm = names_of.get(res, set())
        # two definitions with one layout but different names: the layout stands (as before
        # P9.99), the NAMES are undecided -- never a guess
        _STRUCT_NAMES[name] = list(next(iter(nm))) if len(nm) == 1 else None
        res = (res[0], res[1], list(res[2]))
    _STRUCT_CACHE[name] = res
    return res


def struct_layout(name):
    """(size, align) of a by-value struct, or None (see struct_fields)."""
    sf = struct_fields(name)
    return (sf[0], sf[1]) if sf else None


# ---------------------------------------------------------------------------
# the ABI table (shared with check_push_block.py)
# ---------------------------------------------------------------------------
Row = collections.namedtuple("Row", "name arr sizes count mask line")


def _eval_mask(expr, masks):
    e = expr.strip()
    if e in masks:
        return masks[e]
    total = 0
    for part in e.split("|"):
        p = part.strip().strip("()").strip()
        m = re.fullmatch(r"(\d+)[uU]?", p)
        if m:
            total |= int(m.group(1))
            continue
        m = re.fullmatch(r"1[uU]?\s*<<\s*(\d+)", p)
        if m:
            total |= 1 << int(m.group(1))
            continue
        if p in masks:
            total |= masks[p]
            continue
        raise ValueError("hostPtrMask `%s` is not a literal, `1u << n` or a named mask" % expr.strip())
    return total


def read_abi(text=None):
    """(rows, bad).  rows: cudaName -> Row, in table order, modelling array[:argCount]
    exactly as the bridge uses it.  bad: [(cudaName or array, category, message)] with
    category ROW-INTEGRITY or MASK-BAD -- every one a failure (I1, I4's row half)."""
    src = text if text is not None else open(ABI).read()
    code = _strip_comments(src)
    bad = []
    arrays = {}
    for m in re.finditer(r"static\s+const\s+uint8_t\s+(\w+)\s*\[\s*\]\s*=\s*\{([^}]*)\}\s*;", code):
        vals = [v.strip() for v in m.group(2).split(",")]
        if vals and vals[-1] == "":
            vals.pop()
        if not all(re.fullmatch(r"[0-9]+", v) for v in vals):
            bad.append((m.group(1), "ROW-INTEGRITY",
                        "array %s has a non-decimal entry {%s}" % (m.group(1), m.group(2).strip())))
            continue
        arrays[m.group(1)] = [int(v) for v in vals]
        if 0 in arrays[m.group(1)]:
            bad.append((m.group(1), "ROW-INTEGRITY", "array %s has a zero entry" % m.group(1)))
    for m in re.finditer(r"\buint8_t\s+(\w+)\s*\[", code):
        if m.group(1) not in arrays and not any(b[0] == m.group(1) for b in bad):
            bad.append((m.group(1), "ROW-INTEGRITY",
                        "array %s is not of the form `static const uint8_t %s[] = { ... };` -- its "
                        "entries cannot be read, so it is FAILED, never skipped" % (m.group(1), m.group(1))))
    masks = {}
    for m in re.finditer(r"static\s+const\s+uint32_t\s+(\w+)\s*=\s*([^;]+);", code):
        try:
            masks[m.group(1)] = _eval_mask(m.group(2), masks)
        except ValueError as e:
            bad.append((m.group(1), "MASK-BAD", str(e)))
    rows = collections.OrderedDict()
    km = re.search(r"\bkRows\s*\[\s*\]\s*=\s*\{", code)
    if not km:
        bad.append(("kRows", "ROW-INTEGRITY", "no `kRows[] = {` table found"))
        return rows, bad
    depth, j = 1, km.end()
    while j < len(code) and depth:
        if code[j] == "{":
            depth += 1
        elif code[j] == "}":
            depth -= 1
        j += 1
    body_start = km.end()
    for em in re.finditer(r"\{([^{}]*)\}", code[body_start:j - 1]):
        fields = [f.strip() for f in em.group(1).split(",")]
        if fields and fields[-1] == "":
            fields.pop()
        line = src.count("\n", 0, body_start + em.start()) + 1
        nm = re.fullmatch(r'"([^"]+)"', fields[0]) if fields else None
        if not nm or len(fields) not in (3, 4):
            bad.append(("kRows:%d" % line, "ROW-INTEGRITY", "unparsed row entry {%s}" % em.group(1).strip()))
            continue
        name, arr, cnt = nm.group(1), fields[1], fields[2]
        if name in rows:
            bad.append((name, "ROW-INTEGRITY",
                        "duplicate row (line %d); the bridge's pxgFindKernelAbi takes the FIRST "
                        "(line %d), so this one is dead" % (line, rows[name].line)))
            continue
        if arr not in arrays:
            bad.append((name, "ROW-INTEGRITY", "row names array %s, which was not parsed" % arr))
            continue
        n = None
        sm = re.fullmatch(r"(?:uint32_t\s*\(\s*)?sizeof\s*\(\s*(\w+)\s*\)\s*\)?", cnt)
        lm = re.fullmatch(r"(\d+)[uU]?", cnt)
        if sm:
            if sm.group(1) != arr:
                bad.append((name, "ROW-INTEGRITY",
                            "argCount sizeof(%s) counts ANOTHER array than the row's %s" % (sm.group(1), arr)))
            if sm.group(1) in arrays:
                n = len(arrays[sm.group(1)])
            else:
                bad.append((name, "ROW-INTEGRITY", "argCount sizeof(%s): no such array" % sm.group(1)))
        elif lm:
            n = int(lm.group(1))
            if n != len(arrays[arr]):
                bad.append((name, "ROW-INTEGRITY",
                            "literal argCount %d != the %d entries of %s" % (n, len(arrays[arr]), arr)))
        else:
            bad.append((name, "ROW-INTEGRITY", "argCount `%s` is neither sizeof(array) nor a literal" % cnt))
        mask = 0
        if len(fields) == 4:
            try:
                mask = _eval_mask(fields[3], masks)
            except ValueError as e:
                bad.append((name, "MASK-BAD", str(e)))
        sizes = arrays[arr][:n] if n is not None else list(arrays[arr])
        if n is not None and n > len(arrays[arr]):
            bad.append((name, "ROW-INTEGRITY", "argCount %d reads past the %d entries of %s"
                        % (n, len(arrays[arr]), arr)))
        rows[name] = Row(name, arr, sizes, n, mask, line)
        # I4, the row half: every masked bit names an 8-byte argument the bridge reads.
        b = 0
        while mask >> b:
            if mask >> b & 1:
                if b >= len(sizes):
                    bad.append((name, "MASK-BAD", "hostPtrMask bit %d >= argCount %d" % (b, len(sizes))))
                elif sizes[b] != 8:
                    bad.append((name, "MASK-BAD",
                                "hostPtrMask bit %d marks a %d-byte argument; translateHostPtrs "
                                "refuses every launch unless it is 8" % (b, sizes[b])))
            b += 1
    return rows, bad


# ---------------------------------------------------------------------------
# the CUDA signature leg (I5)
# ---------------------------------------------------------------------------
def _split_params(pl):
    out, d, cur = [], 0, ""
    for c in pl:
        if c in "<([":
            d += 1
        elif c in ">)]":
            d -= 1
        if c == "," and d == 0:
            out.append(cur)
            cur = ""
        else:
            cur += c
    if cur.strip():
        out.append(cur)
    return [re.sub(r"\s+", " ", x).strip() for x in out]


def param_shape(p):
    """(size, type, fields) of one CUDA parameter declaration, or (None, text, None).
    fields: [(offset, size, kind)] for a by-value aggregate, None for a scalar/pointer."""
    p = re.sub(r"=.*$", "", p).strip()                    # default argument
    p = _norm_type(p)
    if p in ("void", ""):
        return None, "void", None
    m = re.match(r"^(.*?)(\w+)\s*(\[\s*\w*\s*\])?$", p)
    if not m or not m.group(1).strip():
        return None, p, None
    t = m.group(1).strip()
    if m.group(3) or "*" in t:
        return 8, t + ("[]" if m.group(3) else ""), None
    if t.endswith("&"):
        t = t[:-1].strip()
    if t.startswith("PxgDevicePointer"):
        return 8, t, None
    base = t.split()[-1]
    if base == "uint":
        return 4, t, None
    if base in VECTORS or (base in SIZES and base not in VECTORS):
        shape = _member_shape(base)
        fields = shape[2] if base in VECTORS else None
        return shape[0], t, fields
    if re.search(r"::Enum$", base):
        return 4, t, None
    sf = struct_fields(base)
    if sf:
        return sf[0], t, sf[2]
    return None, t, None


def param_name(p):
    """The declared NAME of one CUDA parameter, or None (P9.99) -- param_shape's own regex,
    so the two can never disagree on where the name is."""
    p = re.sub(r"=.*$", "", p).strip()
    p = _norm_type(p)
    if p in ("void", ""):
        return None
    m = re.match(r"^(.*?)(\w+)\s*(\[\s*\w*\s*\])?$", p)
    if not m or not m.group(1).strip():
        return None
    return m.group(2)


_SIGS = None
_SIG_NAMES = {}         # P9.99: (relpath, line) -> [parameter names], filled by signatures_in


def cuda_signatures():
    """kernel -> [(relpath, line, [(size, type, fields)])] over every gpu*/ .cu/.cuh."""
    global _SIGS
    if _SIGS is not None:
        return _SIGS
    import check_grid_stride as cgs            # kernel_decls; sibling import
    out = collections.defaultdict(list)
    for d in sorted(os.listdir(SRC)):
        if not d.startswith("gpu") or d == "gpuslang":
            continue
        for dp, _dn, fns in os.walk(os.path.join(SRC, d)):
            for fn in sorted(fns):
                if not fn.endswith((".cu", ".cuh")):
                    continue
                path = os.path.join(dp, fn)
                out_k = signatures_in(open(path, errors="replace").read(), os.path.relpath(path, ROOT), cgs)
                for k, v in out_k.items():
                    out[k].extend(v)
    _SIGS = dict(out)
    return _SIGS


def signatures_in(src, rel, cgs=None):
    if cgs is None:
        import check_grid_stride as cgs
    t = _mask_pp(_masked(src))
    out = collections.defaultdict(list)
    for name, off in cgs.kernel_decls(t):
        j, depth, k = off - 1, 0, off - 1
        while k >= 0:
            if t[k] == ")":
                depth += 1
            elif t[k] == "(":
                depth -= 1
                if depth == 0:
                    break
            k -= 1
        plist = t[k + 1:j]
        line = t.count("\n", 0, off) + 1
        out[name].append((rel, line, [param_shape(p) for p in _split_params(plist)]))
        _SIG_NAMES[(rel, line)] = [param_name(p) for p in _split_params(plist)]
    return out


def signature_names(name, sigs=None):
    """The parameter names of the declaration signature_of decided on, or None.  Every
    declaration with the same sizes must carry the same names (a forward declaration that
    renames a parameter is an UNDECIDED name join, never a pick) -- P9.99."""
    sigs = cuda_signatures() if sigs is None else sigs
    st, sizes, info = signature_of(name, sigs)
    if st != "OK":
        return None
    seen = set()
    for rel, line, params in sigs.get(name, ()):
        if [s for s, _t, _f in params] == sizes:
            nm = _SIG_NAMES.get((rel, line))
            if nm is None:
                return None
            seen.add(tuple(nm))
    return list(seen.pop()) if len(seen) == 1 else None


def signature_of(name, sigs=None):
    """('EQ-able', sizes, params) | ('NO-DECL'|'UNRES'|'MULTI', detail, None)."""
    sigs = cuda_signatures() if sigs is None else sigs
    ds = sigs.get(name)
    if not ds:
        return "NO-DECL", None, None
    tups, unres = {}, []
    for rel, line, params in ds:
        if any(s is None for s, _t, _f in params):
            unres.append("%s:%d %s" % (rel, line, [t for s, t, _f in params if s is None]))
            continue
        tups.setdefault(tuple(s for s, _t, _f in params), (rel, line, params))
    if not tups:
        return "UNRES", "; ".join(unres), None
    if len(tups) > 1:
        return "MULTI", sorted("%s:%d %s" % (v[0], v[1], list(k)) for k, v in tups.items()), None
    (sizes, (rel, line, params)), = tups.items()
    return "OK", list(sizes), (rel, line, params)


def arg_fields(name, row_sizes, sigs=None):
    """For check_push_block.py's join: per row argument, a list of (offset-in-argument,
    size, kind) fields for a by-value aggregate, None for a scalar/pointer, or the string
    'UNDECIDED' when the argument is an aggregate this file cannot lay out."""
    st, sizes, info = signature_of(name, sigs)
    params = info[2] if st == "OK" else None
    out = []
    for i, s in enumerate(row_sizes):
        if (name, i) in HOST_ARG_FIELDS:
            out.append(HOST_ARG_FIELDS[(name, i)])
            continue
        if params is not None and i < len(params) and params[i][0] == s:
            if params[i][2] is not None:
                out.append(params[i][2])
                continue
            out.append(None)
            continue
        out.append(None if s in (1, 2, 4, 8) else "UNDECIDED")
    return out


def arg_field_names(name, row_sizes, sigs=None):
    """P9.99, for check_push_block.py's NAME leg: per row argument, [(dotted CUDA name, CUDA
    type)] -- one entry for a scalar/pointer, one per field (the order of arg_fields) for a
    by-value aggregate; (None, None) where the row does not follow the signature.

    None -- the whole kernel's name join UNDECIDED, never a pass -- when the signature is
    decided but its NAMES are not (two declarations with the same sizes and different
    parameter names).  A parameter the row follows whose name cannot be read is named '?',
    as is every field of a by-value struct whose names are undecided (`writer.?`); the
    name leg treats a reached member over a '?' as UNDECIDED too."""
    st, _sizes, info = signature_of(name, sigs)
    params = info[2] if st == "OK" else None
    pnames = signature_names(name, sigs) if params is not None else None
    if params is not None and pnames is None:
        return None
    out = []
    for i, (s, shape) in enumerate(zip(row_sizes, arg_fields(name, row_sizes, sigs))):
        if (name, i) in HOST_ARG_FIELD_NAMES:
            out.append(list(HOST_ARG_FIELD_NAMES[(name, i)]))
            continue
        follows = pnames is not None and i < len(params) and params[i][0] == s
        pn, pt = (pnames[i] or "?", params[i][1]) if follows else (None, None)
        if shape is None or shape == "UNDECIDED":
            out.append([(pn, pt)])
            continue
        sub = _member_names(pt, len(shape)) if pn else [("", None)] * len(shape)
        out.append([((pn + sn) if pn else None, ty) for (sn, ty) in sub])
    return out


# ---------------------------------------------------------------------------
# the host leg: launch sites
# ---------------------------------------------------------------------------
def read_kernel_ids(text=None):
    text = open(NAMES).read() if text is None else text
    return {m.group(1): m.group(2) for m in re.finditer(r'KERNEL_DEF\(\s*(\w+)\s*,\s*"([^"]+)"\s*\)', text)}


def engine_sources():
    out = []
    for d in ENGINE_DIRS:
        for dirpath, _dirnames, filenames in os.walk(d):
            if os.sep + "CUDA" + os.sep in dirpath + os.sep:
                continue                      # device code, not launch sites
            for fn in sorted(filenames):
                if fn.endswith(".cpp"):
                    out.append(os.path.join(dirpath, fn))
    return sorted(out)


def sibling_headers(path):
    """src/foo.cpp -> the module's include/*.h and src/*.h (tier 3)."""
    if not path or not os.path.isfile(path):
        return []
    d = os.path.dirname(os.path.abspath(path))
    roots = [d]
    if os.path.basename(d) == "src":
        roots.append(os.path.join(os.path.dirname(d), "include"))
    out = []
    for r in roots:
        if os.path.isdir(r):
            out += [os.path.join(r, fn) for fn in sorted(os.listdir(r)) if fn.endswith(".h")]
    return out


# `static`, the constructor form `T x(args)` and brace-init `T x{...}` are declarations
# (B6).  The leading-delimiter class keeps `foo(bar, baz)` from reading `bar` as a
# declaration of type `foo(`; `\*{0,2}&?` admits `PxU8** data` without reading
# `(a && c == 0` as a reference to a reference (P9.95).
DECL = (r"(?:^|[;{}\(,])\s*((?:static\s+)?(?:const\s+)?[\w:]+(?:\s*<[^<>;]*>)?\s*\*{0,2}&?)\s+(%s)"
        r"\s*(?:=|;|\)|,|\[|\(|\{)")
_NOT_TYPES = ("return", "else", "case", "new", "delete", "sizeof", "throw", "goto")


def _decl_types(text, var):
    """(position, type) of every declaration of `var` in `text`.  The pattern is anchored
    on each OCCURRENCE of the name and tried over the few lines before it (from a line
    start, so `^` means what it says), not slid over the whole file: same matches, and
    two adjacent declarations can no longer hide each other by consuming the delimiter
    the second one needs."""
    pat = re.compile(DECL % re.escape(var), re.M)
    for vm in re.finditer(r"\b%s\b" % re.escape(var), text):
        p = vm.start()
        lo = text.rfind("\n", 0, max(0, p - 300)) + 1
        for m in pat.finditer(text, lo, p + len(var) + 64):
            if m.start(2) != p:
                continue
            t = m.group(1).strip()
            if t.split()[-1] in _NOT_TYPES or not re.match(r"^(?:static\s+)?(?:const\s+)?[A-Za-z_]", t):
                continue
            yield m.start(1), t


def _sizes_of(types):
    sizes, unresolved = set(), False
    for t in types:
        sz = arg_size(t)
        if sz is None:
            unresolved = True
        else:
            sizes.add(sz)
    return sizes.pop() if (len(sizes) == 1 and not unresolved) else None, bool(sizes or unresolved)


_FLAT_CACHE = {}


def declared_size(sf, var, use_pos):
    """sizeof `var` as declared at `use_pos` of Source `sf` (tiers 1L, 2, 3), or None."""
    fb = sf.function_body(use_pos)
    if fb is not None:
        p0, body_open, body_close = fb
        vis = []
        for dpos, t in _decl_types(sf.mask[p0:use_pos], var):
            dpos += p0
            if dpos < body_open:
                scope = body_open                  # a parameter: the body's own scope
            else:
                k = sf.block_at(dpos)
                if k < 0 or sf.closes[k] < use_pos:
                    continue                       # its block closed before the use
                scope = sf.opens[k]
            vis.append((scope, t))
        if vis:
            inner = max(s for s, _t in vis)
            return _sizes_of([t for s, t in vis if s == inner])[0]
    key = (sf.path, var)
    if key not in _FLAT_CACHE:
        res, found = _sizes_of([t for _p, t in _decl_types(sf.mask, var)])
        if not found:
            types = []
            for h in sibling_headers(sf.path):
                types += [t for _p, t in _decl_types(load_source(h).mask, var)]
            res = _sizes_of(types)[0]
        _FLAT_CACHE[key] = res
    return _FLAT_CACHE[key]


_METHOD_CACHE = {}


def method_return_size(method):
    """sizeof the value `method()` returns, from its declarations in gpu*/include (P9.95,
    core->getSoftBodyBuffer()), or None on any disagreement."""
    if method in _METHOD_CACHE:
        return _METHOD_CACHE[method]
    types = []
    for d in HEADER_DIRS:
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".h"):
                text = load_source(os.path.join(d, fn)).mask
                for m in re.finditer(r"(?:^|[;{}\s])((?:const\s+)?[\w:]+(?:\s*<[^<>;]*>)?\s*\*{0,2}&?)\s+"
                                     + re.escape(method) + r"\s*\(\s*\)", text, re.M):
                    t = m.group(1).strip()
                    if t.split()[-1] not in _NOT_TYPES:
                        types.append(t)
    _METHOD_CACHE[method] = _sizes_of(types)[0]
    return _METHOD_CACHE[method]


_MEMBER_CACHE = {}


def member_size(member):
    """sizeof a data member named `member`, looked up BY NAME over every header dir; all
    declarations must agree (`mDesc.gpuBroadPhaseNbBitsShiftX` -> 1, `mDoVelLimit` -> 4)."""
    if member in _MEMBER_CACHE:
        return _MEMBER_CACHE[member]
    types = []
    pat = re.compile(r"(?:^|[;{}])\s*((?:const\s+)?[\w:]+(?:\s*<[^<>;]*>)?\s*\*{0,2}&?)\s+"
                     + re.escape(member) + r"\s*(?:;|=|\[)", re.M)
    for text in _header_texts():
        if member not in text:
            continue
        for m in pat.finditer(text):
            t = m.group(1).strip()
            if t.split()[-1] not in _NOT_TYPES:
                types.append(t)
    _MEMBER_CACHE[member] = _sizes_of(types)[0]
    return _MEMBER_CACHE[member]


ARG_RE = r"(?:PX_CUDA_KERNEL_PARAM2?|CUDA_KERNEL_PARAM)\(\s*\(?\s*([\w\.\->\[\]]+(?:\(\))?)\s*\)?\s*\)"
ARG_OPEN = r"(?:PX_CUDA_KERNEL_PARAM2?|CUDA_KERNEL_PARAM)\("
BLOCK_RE = r"(?:PxCudaKernelParam|KERNEL_PARAM_TYPE|void\s*\*)\s+%s\s*\[\s*\]\s*=\s*\{(.*?)\}\s*;"

Site = collections.namedtuple("Site", "path line form sizes args why")


def epilog_param(code):
    """The params-array name the LIVE `#define EPILOG` passes in the void** form (the
    identifier before PX_FL, skipping 0/NULL), or None when the EPILOG is sized."""
    m = re.search(r"#\s*define\s+EPILOG\s+([^\n]*)", code)
    if not m:
        return None
    toks = [t.strip() for t in m.group(1).split(",")]
    if any("sizeof" in t for t in toks) or "PX_FL" not in toks:
        return None
    i = toks.index("PX_FL") - 1
    while i >= 0 and toks[i] in ("0", "NULL", "nullptr"):
        i -= 1
    return toks[i] if i >= 0 and re.match(r"^\w+$", toks[i]) else None


def _size_arg(sf, a, pos):
    cm = re.match(r"^(?:\w+->|\w+\.)?(\w+)\(\)$", a)
    if cm:
        return method_return_size(cm.group(1))
    if re.match(r"^\w+$", a):
        return declared_size(sf, a, pos)
    mm = re.match(r"^\w+(?:\.|->)(\w+)$", a)
    if mm:
        return member_size(mm.group(1))
    return None


def _pack(sf, start, launch_pos, pname):
    """(sizes|None, args, why) from the LAST params block named `pname` in
    mask[start:launch_pos], resolved at the BLOCK's position (P9.97: resolving at the
    handle bound cloth_boxTriangleContactGenLaunch's `writer`, declared after its
    getCuFunction, to another function's 72-byte writer)."""
    blocks = list(re.finditer(BLOCK_RE % re.escape(pname), sf.mask[start:launch_pos], re.S))
    if not blocks:
        return None, None, "no-block"
    bm = blocks[-1]
    block = bm.group(1)
    block_pos = start + bm.start()
    args = re.findall(ARG_RE, block)
    if not args:
        return None, [], "no-args"
    # NEVER DECIDE ON A PARTIAL ARGUMENT LIST (P9.95): findall drops what it cannot match.
    if len(re.findall(ARG_OPEN, block)) != len(args):
        return None, args, "partial-arglist"
    sizes = []
    for a in args:
        sz = _size_arg(sf, a, block_pos)
        if sz is None:
            return None, args, "unresolved:" + a
        sizes.append(sz)
    return sizes, args, ""


def packs_from_source(sf, ids, rel=None):
    """kernel -> [Site] for one Source.  The association is BOUND THROUGH THE HANDLE:
        <h> = ...getCuFunction(PxgKernelIds::<ID>)       (a ternary may name two ids)
        launchKernel(<h>, ..., <p>, sizeof(<p>), ...)    sized
        launchKernel(<h>, ..., EPILOG)                   void** (the live #define EPILOG)
        launchKernel(<h>, ..., <p>, 0|NULL, PX_FL)       void**
        _launch<...>(PROLOG, PxgKernelIds::<ID>, ..., EPILOG)  void**, no handle
    and EVERY launch of the handle up to its reassignment or the end of its function is
    read (B7), not only the first."""
    rel = rel or os.path.relpath(sf.path, ROOT)
    code = sf.mask
    out = collections.defaultdict(list)
    if "PxgKernelIds" not in code:
        return out
    ep = epilog_param(code)
    for hm in re.finditer(r"(\w+)\s*=\s*([^;]*?getCuFunction\s*\([^;]*?)\s*;", code, re.S):
        handle, rhs = hm.group(1), hm.group(2)
        kids = re.findall(r"PxgKernelIds::(\w+)", rhs)
        if not kids:
            continue
        fb = sf.function_body(hm.start())
        end = fb[2] if fb else len(code)
        nxt = re.search(r"\b" + re.escape(handle) + r"\s*=\s*[^;]*getCuFunction", code[hm.end():end], re.S)
        region_end = hm.end() + nxt.start() if nxt else end
        start = fb[0] if fb else 0
        for lm in re.finditer(r"launchKernel\s*\(\s*" + re.escape(handle) + r"\s*,(.*?)\)\s*;",
                              code[hm.end():region_end], re.S):
            largs = lm.group(1)
            pm = re.search(r"(\w+)\s*,\s*sizeof\s*\(\s*\1\s*\)", largs)
            if pm:
                form, pname = "sized", pm.group(1)
            elif ep and re.search(r"\bEPILOG\b", largs):
                form, pname = "void**", ep
            else:
                vm = re.search(r",\s*(\w+)\s*,\s*(?:0|NULL|nullptr)\s*,\s*PX_FL\s*$", largs.strip())
                if not vm:
                    continue
                form, pname = "void**", vm.group(1)
            launch_pos = hm.end() + lm.start()
            sizes, args, why = _pack(sf, start, launch_pos, pname)
            if why == "no-block":
                continue
            for kid in kids:
                if kid in ids:
                    out[ids[kid]].append(Site(rel, sf.line(launch_pos), form, sizes, args, why))
    for lm in re.finditer(r"_launch\s*<[^>]*>\s*\(\s*PROLOG\s*,\s*PxgKernelIds::(\w+)\s*,(.*?)\)\s*;",
                          code, re.S):
        kid, largs = lm.group(1), lm.group(2)
        if not (ep and re.search(r"\bEPILOG\b", largs)):
            continue
        fb = sf.function_body(lm.start())
        sizes, args, why = _pack(sf, fb[0] if fb else 0, lm.start(), ep)
        if why == "no-block":
            continue
        if kid in ids:
            out[ids[kid]].append(Site(rel, sf.line(lm.start()), "void**", sizes, args, why))
    return out


def launch_packs(ids):
    packs = collections.defaultdict(list)
    for path in engine_sources():
        sf = load_source(path)
        for k, v in packs_from_source(sf, ids).items():
            packs[k].extend(v)
    return dict(packs)


# ---------------------------------------------------------------------------
# the verdicts
# ---------------------------------------------------------------------------
def _describe(row, host):
    rl, _ = arg_layout(row)
    hl, _ = arg_layout(host)
    for i in range(max(len(rl), len(hl))):
        r = rl[i] if i < len(rl) else None
        h = hl[i] if i < len(hl) else None
        if r != h:
            return "first difference at arg %d: host %s, row %s" % (
                i, "%d @%d" % (h[1], h[0]) if h else "-", "%d @%d" % (r[1], r[0]) if r else "-")
    return ""


def evaluate(rows, bad, packs, sigs, no_site=None, undecided=None, tail=None, follows=None,
             sig_undecided=None):
    """(findings, stats, table).  findings: [(category, kernel, message)], every one a
    failure.  The SAME code path for the tree and for the selftest's scratch rows."""
    no_site = KNOWN_NO_SITE if no_site is None else no_site
    undecided = KNOWN_UNDECIDED if undecided is None else undecided
    tail = KNOWN_TAIL_EXTRA if tail is None else tail
    follows = KNOWN_ROW_FOLLOWS_HOST if follows is None else follows
    sig_undecided = KNOWN_SIG_UNDECIDED if sig_undecided is None else sig_undecided
    findings = [(cat, name, msg) for (name, cat, msg) in bad]
    st = collections.Counter()
    table = {}
    tail_seen, nosite_seen, und_seen, follows_ok = set(), set(), set(), set()
    for name, row in rows.items():
        sites = packs.get(name, [])
        forms = sorted(set(s.form for s in sites))
        entry = {"arr": row.arr, "row": row.sizes, "forms": forms,
                 "sites": [[s.path, s.line, s.form, s.sizes, s.args, s.why] for s in sites]}
        table[name] = entry
        # ---- host leg ----
        if not sites:
            entry["host"] = "NO-SITE"
            nosite_seen.add(name)
            st["no-site"] += 1
            if name not in no_site:
                findings.append(("NO-SITE", name, "row %s has no launch site this checker can follow; "
                                 "add it to KNOWN_NO_SITE with the hand-checked pack, or teach the "
                                 "resolver its form" % row.arr))
        elif any(s.sizes is None for s in sites):
            entry["host"] = "UNDECIDED"
            und_seen.add(name)
            st["undecided"] += 1
            if name not in undecided:
                why = sorted(set("%s:%d %s" % (s.path, s.line, s.why) for s in sites if s.sizes is None))
                findings.append(("UNDECIDED", name, "pack not resolved (%s)" % "; ".join(why)))
        else:
            verdicts = []
            for s in sites:
                if s.sizes == row.sizes:
                    verdicts.append("EQ")
                elif len(s.sizes) > len(row.sizes) and s.sizes[:len(row.sizes)] == row.sizes:
                    if abi_push_size(s.sizes) == abi_push_size(row.sizes) and name in tail:
                        verdicts.append("TAIL-WAIVED")
                    else:
                        verdicts.append("TAIL-EXTRA")
                        findings.append(("TAIL-EXTRA", name,
                                         "row %s %s, %s:%d (%s) pushes %s -- extra trailing arguments%s"
                                         % (row.arr, row.sizes, s.path, s.line, s.form, s.sizes,
                                            "" if abi_push_size(s.sizes) == abi_push_size(row.sizes)
                                            else " that OVERFLOW the row's blob (%d > %d)"
                                            % (abi_push_size(s.sizes), abi_push_size(row.sizes)))))
                elif s.form == "sized" and abi_push_size(s.sizes) == abi_push_size(row.sizes):
                    verdicts.append("SEQ-DIFF")
                    findings.append(("SEQ-DIFF", name,
                                     "row %s %s vs %s:%d pushes %s -- equal totals (%d), so the shim's "
                                     "gate passes and only this check sees it; %s"
                                     % (row.arr, row.sizes, s.path, s.line, s.sizes,
                                        abi_push_size(row.sizes), _describe(row.sizes, s.sizes))))
                else:
                    verdicts.append("DIFF")
                    findings.append(("DIFF", name,
                                     "row %s %s (%d bytes) vs %s:%d (%s) pushes %s (%d bytes); %s"
                                     % (row.arr, row.sizes, abi_push_size(row.sizes), s.path, s.line,
                                        s.form, s.sizes, abi_push_size(s.sizes),
                                        _describe(row.sizes, s.sizes))))
            if "TAIL-WAIVED" in verdicts:
                tail_seen.add(name)
            entry["host"] = "+".join(sorted(set(verdicts)))
            if set(verdicts) <= {"EQ", "TAIL-WAIVED"}:
                st["host-eq"] += 1
                st["host-" + "/".join(forms)] += 1
                if "TAIL-WAIVED" in verdicts:
                    st["host-tail-waived"] += 1
                if all(v == "EQ" for v in verdicts) and forms == ["sized"]:
                    follows_ok.add(name)
            # I4, the host half: a masked argument is 8 bytes at every decided site.
            b = 0
            while row.mask >> b:
                if row.mask >> b & 1:
                    for s in sites:
                        if b >= len(s.sizes) or s.sizes[b] != 8:
                            findings.append(("MASK-BAD", name, "hostPtrMask bit %d: %s:%d pushes %s there"
                                             % (b, s.path, s.line,
                                                "nothing" if b >= len(s.sizes) else "%d bytes" % s.sizes[b])))
                b += 1
        # ---- signature leg ----
        sst, ssizes, sinfo = signature_of(name, sigs)
        entry["sig"] = ssizes if sst == "OK" else sst
        if sst != "OK":
            st["sig-undecided"] += 1
            if name not in sig_undecided:
                findings.append(("SIG-" + sst, name, "CUDA signature: %s" % (ssizes,)))
        elif ssizes == row.sizes:
            st["sig-eq"] += 1
            if name in follows:
                findings.append(("STALE", name, "KNOWN_ROW_FOLLOWS_HOST, but the row now equals its "
                                 "CUDA signature %s:%d; remove the entry" % (sinfo[0], sinfo[1])))
        elif name in follows and "void**" not in forms and name in follows_ok:
            st["sig-follows-host"] += 1
        else:
            why = ""
            if name in follows and "void**" in forms:
                why = " -- and a void** kernel may NEVER follow the host: its row is the copy length"
            elif name in follows:
                why = " -- KNOWN_ROW_FOLLOWS_HOST needs every site sized and EQ"
            findings.append(("SIG-DIFF", name, "row %s %s != CUDA signature %s at %s:%d%s; %s"
                             % (row.arr, row.sizes, ssizes, sinfo[0], sinfo[1], why,
                                _describe(ssizes, row.sizes))))
    # ---- stale ratchet entries ----
    for k in sorted(set(no_site) - nosite_seen):
        findings.append(("STALE", k, "KNOWN_NO_SITE entry: the kernel now has a decided launch site "
                         "or no row; remove it"))
    for k in sorted(set(undecided) - und_seen):
        findings.append(("STALE", k, "KNOWN_UNDECIDED entry: now decided (or gone); remove it"))
    for k in sorted(set(tail) - tail_seen):
        findings.append(("STALE", k, "KNOWN_TAIL_EXTRA waiver no longer applies; remove it"))
    for k in sorted(set(follows) - set(rows)):
        findings.append(("STALE", k, "KNOWN_ROW_FOLLOWS_HOST entry has no row"))
    for k in sorted(set(sig_undecided) - set(n for n in rows if signature_of(n, sigs)[0] != "OK")):
        findings.append(("STALE", k, "KNOWN_SIG_UNDECIDED entry: the signature now resolves"))
    st["rows"] = len(rows)
    st["masked"] = sum(1 for r in rows.values() if r.mask)
    st["sites"] = sum(len(packs.get(n, [])) for n in rows)
    return findings, st, table


# ---------------------------------------------------------------------------
# SELFTEST
# ---------------------------------------------------------------------------
# Part A: deliberately wrong rows on a scratch copy of the real ABI text (P9.97's
# mutants M1-M9 plus the I1 cases), each with the verdict it MUST produce.
SELFTEST_ROWS = [
    ("M1 vec3 split (sized)", "static const uint8_t kTwoPtrVec3[] = { 8, 8, 12 };",
     "static const uint8_t kTwoPtrVec3[] = { 8, 8, 8, 4 };",
     [("SEQ-DIFF", "applyTGSSubstepGravity"), ("SIG-DIFF", "applyTGSSubstepGravity")]),
    ("M2 sapbox swap (void**)", "static const uint8_t kSapBoxInit[] = { 8, 4, 1 };",
     "static const uint8_t kSapBoxInit[] = { 8, 1, 4 };",
     [("DIFF", "initializeSapBox1DLaunch"), ("DIFF", "updateBodiesLaunch1T"),
      ("DIFF", "updateBodiesLaunch_Part2"), ("DIFF", "updateBodiesLaunch_Part3"),
      ("SIG-DIFF", "initializeSapBox1DLaunch")]),
    ("M3 radix merge (mixed)", "static const uint8_t kRadixDesc[] = { 8, 4, 4 };",
     "static const uint8_t kRadixDesc[] = { 8, 8 };",
     [("DIFF", "radixSortMultiBlockLaunchWithCount")]),
    ("M4 getset merge (void**)", "static const uint8_t kGetSet6[] = { 8, 8, 8, 4, 4, 4 };",
     "static const uint8_t kGetSet6[] = { 8, 8, 8, 4, 8 };",
     [("DIFF", "getArtiDofStates"), ("SIG-DIFF", "getArtiDofStates")]),
    ("M5 propstatic merge (sized)", "static const uint8_t kPropagateStaticPGS[] = { 8, 8, 4, 4, 4 };",
     "static const uint8_t kPropagateStaticPGS[] = { 8, 8, 8, 4 };",
     [("SEQ-DIFF", "propagateStaticSolverBodyVelocities"),
      ("STALE", "propagateStaticSolverBodyVelocities")]),
    ("M6 literal count, wrong array", '{ "markUpdatedPairsLaunch", kOnePtr, 1u },',
     '{ "markUpdatedPairsLaunch", kTwoPtr, 1u },',
     [("ROW-INTEGRITY", "markUpdatedPairsLaunch")]),
    ("M7 sizeof of the wrong array",
     '{ "applyTGSSubstepGravity", kTwoPtrVec3, uint32_t(sizeof(kTwoPtrVec3)) },',
     '{ "applyTGSSubstepGravity", kTwoPtrVec3, uint32_t(sizeof(kTwoPtr)) },',
     [("ROW-INTEGRITY", "applyTGSSubstepGravity"), ("TAIL-EXTRA", "applyTGSSubstepGravity")]),
    ("M8 tail waiver goes stale", "static const uint8_t kPropagateStaticPGS[] = { 8, 8, 4, 4, 4 };",
     "static const uint8_t kPropagateStaticPGS[] = { 8, 8, 4, 4, 4, 1 };",
     [("STALE", "propagateStaticSolverBodyVelocities"),
      ("SIG-DIFF", "propagateStaticSolverBodyVelocities")]),
    ("M9 hostPtrMask on a u32", "static const uint32_t kCvxTriCoreHostPtrMask = 1u << 18;",
     "static const uint32_t kCvxTriCoreHostPtrMask = 1u << 15;",
     [("MASK-BAD", "convexTrimeshNarrowphase")]),
    ("I3 a waived tail that OVERFLOWS the row", "static const uint8_t kPropagateStaticPGS[] = { 8, 8, 4, 4, 4 };",
     "static const uint8_t kPropagateStaticPGS[] = { 8, 8, 4, 4 };",
     [("TAIL-EXTRA", "propagateStaticSolverBodyVelocities")]),
    ("I1 unparsed array (kX[3])", "static const uint8_t kOnePtr[] = { 8 };",
     "static const uint8_t kOnePtr[1] = { 8 };",
     [("ROW-INTEGRITY", "kOnePtr"), ("ROW-INTEGRITY", "markUpdatedPairsLaunch")]),
    ("I1 hex entry", "static const uint8_t kSapBoxInit[] = { 8, 4, 1 };",
     "static const uint8_t kSapBoxInit[] = { 0x8, 4, 1 };",
     [("ROW-INTEGRITY", "kSapBoxInit")]),
    ("I1 zero entry", "static const uint8_t kSapBoxInit[] = { 8, 4, 1 };",
     "static const uint8_t kSapBoxInit[] = { 8, 4, 0 };",
     [("ROW-INTEGRITY", "kSapBoxInit")]),
    ("I1 duplicate cudaName", '{ "markUpdatedPairsLaunch", kOnePtr, 1u },',
     '{ "markUpdatedPairsLaunch", kOnePtr, 1u },\n    { "markUpdatedPairsLaunch", kTwoPtr, 2u },',
     [("ROW-INTEGRITY", "markUpdatedPairsLaunch")]),
]

# Part B: literal C++ snippets through the REAL resolver.  Each is built so that the
# WRONG binding gives a DIFFERENT answer.
SELFTEST_IDS = {"K": "kernelK", "K2": "kernelK2"}
SELFTEST_SOURCES = [
    ("B6 static declaration", """
namespace physx {
void A::f() { const PxU64 count = 2; use(count); }
void A::g()
{
    static PxU32 count = 0;
    CUfunction h = w->getCuFunction(PxgKernelIds::K);
    PxCudaKernelParam kp[] = { PX_CUDA_KERNEL_PARAM(count) };
    ctx->launchKernel(h, 1, 1, 1, 1, 1, 1, 0, s, kp, sizeof(kp), 0, PX_FL);
}
}
""", {"kernelK": [("sized", [4])]}),
    ("B5 namespace is not a function", """
namespace physx {
void A::f()
{
    PxU32 n = 3;
    CUfunction h = w->getCuFunction(PxgKernelIds::K);
    PxCudaKernelParam kp[] = { PX_CUDA_KERNEL_PARAM(n) };
    ctx->launchKernel(h, 1, 1, 1, 1, 1, 1, 0, s, kp, sizeof(kp), 0, PX_FL);
}
void A::g() { PxU64 n = 4; use(n); }
}
""", {"kernelK": [("sized", [4])]}),
    ("B6 constructor form and struct layout", """
namespace physx {
void A::g(PxgParticleContactWriter& writer) { use(writer); }
void A::f()
{
    PxgFEMContactWriter writer(core);
    CUfunction h = w->getCuFunction(PxgKernelIds::K);
    PxCudaKernelParam kp[] = { PX_CUDA_KERNEL_PARAM(writer) };
    ctx->launchKernel(h, 1, 1, 1, 1, 1, 1, 0, s, kp, sizeof(kp), 0, PX_FL);
}
}
""", {"kernelK": [("sized", [80])]}),
    ("declaration after the handle", """
namespace physx {
void A::g(PxgParticleContactWriter& writer) { use(writer); }
void A::f()
{
    CUfunction h = w->getCuFunction(PxgKernelIds::K);
    PxgFEMContactWriter writer(core);
    PxCudaKernelParam kp[] = { PX_CUDA_KERNEL_PARAM(writer) };
    ctx->launchKernel(h, 1, 1, 1, 1, 1, 1, 0, s, kp, sizeof(kp), 0, PX_FL);
}
}
""", {"kernelK": [("sized", [80])]}),
    ("a declaration in a closed block", """
void A::f()
{
    { PxU64 x = 1; use(x); }
    PxU32 x = 2;
    CUfunction h = w->getCuFunction(PxgKernelIds::K);
    PxCudaKernelParam kp[] = { PX_CUDA_KERNEL_PARAM(x) };
    ctx->launchKernel(h, 1, 1, 1, 1, 1, 1, 0, s, kp, sizeof(kp), 0, PX_FL);
}
""", {"kernelK": [("sized", [4])]}),
    ("void** EPILOG under #if USE_NEW_LAUNCH_FUNCTION", """
#define USE_NEW_LAUNCH_FUNCTION 1
#if USE_NEW_LAUNCH_FUNCTION
    #define KERNEL_PARAM_TYPE void*
    #define CUDA_KERNEL_PARAM PX_CUDA_KERNEL_PARAM2
    #define EPILOG mStream, kernelParams, PX_FL
#else
    #define KERNEL_PARAM_TYPE PxCudaKernelParam
    #define CUDA_KERNEL_PARAM PX_CUDA_KERNEL_PARAM
    #define EPILOG mStream, kernelParams, sizeof(kernelParams), PX_FL
#endif
void A::f(const CUdeviceptr desc, PxU32 numHandles, bool isNew)
{
    KERNEL_PARAM_TYPE kernelParams[] = { CUDA_KERNEL_PARAM(desc), CUDA_KERNEL_PARAM(numHandles), CUDA_KERNEL_PARAM(isNew) };
    _launch<0>(PROLOG, PxgKernelIds::K, 1, 1, 1, 0, EPILOG);
}
""", {"kernelK": [("void**", [8, 4, 1])]}),
    ("B7 the second launch of one handle", """
void A::f()
{
    PxU32 a = 1; PxU64 b = 2;
    CUfunction h = w->getCuFunction(PxgKernelIds::K);
    PxCudaKernelParam p1[] = { PX_CUDA_KERNEL_PARAM(a) };
    ctx->launchKernel(h, 1, 1, 1, 1, 1, 1, 0, s, p1, sizeof(p1), 0, PX_FL);
    PxCudaKernelParam p2[] = { PX_CUDA_KERNEL_PARAM(b), PX_CUDA_KERNEL_PARAM(a) };
    ctx->launchKernel(h, 1, 1, 1, 1, 1, 1, 0, s, p2, sizeof(p2), 0, PX_FL);
}
""", {"kernelK": [("sized", [4]), ("sized", [8, 4])]}),
    ("member argument", """
#define EPILOG mStream, kernelParams, PX_FL
void A::f()
{
    void* kernelParams[] = { PX_CUDA_KERNEL_PARAM2(mDesc.gpuBroadPhaseNbBitsShiftX) };
    _launch<0>(PROLOG, PxgKernelIds::K, 1, 1, 1, 0, EPILOG);
}
""", {"kernelK": [("void**", [1])]}),
    ("#if 0 site is dead, X::Enum is 4", """
void A::f(PxArticulationGPUAPIReadType::Enum t)
{
    CUfunction h = w->getCuFunction(PxgKernelIds::K);
#if 0
    PxCudaKernelParam kp[] = { PX_CUDA_KERNEL_PARAM(junk) };
    ctx->launchKernel(h, 1, 1, 1, 1, 1, 1, 0, s, kp, sizeof(kp), 0, PX_FL);
#endif
    PxCudaKernelParam kq[] = { PX_CUDA_KERNEL_PARAM(t) };
    ctx->launchKernel(h, 1, 1, 1, 1, 1, 1, 0, s, kq, sizeof(kq), 0, PX_FL);
}
""", {"kernelK": [("sized", [4])]}),
]
SELFTEST_SIGNATURES = [
    ("signature sizes, struct and #if NAME in a parameter list", """
#define ENABLE_X 0
extern "C" __global__ void kernelK(const float* PX_RESTRICT a, const PxU32 n,
    PxgFEMContactWriter writer, const PxVec3 g,
#if ENABLE_X
    PxReal dropped,
#endif
    PxgSoftBody* PX_RESTRICT sb)
{
}
""", "kernelK", [8, 4, 80, 12, 8]),
]


def selftest(packs, sigs, abi_text):
    """[failure messages]; empty when every case produced its literal expectation."""
    fails = []
    rows0, bad0 = read_abi(abi_text)
    base = set((c, k) for (c, k, _m) in evaluate(rows0, bad0, packs, sigs)[0])
    for label, old, new, expect in SELFTEST_ROWS:
        n = abi_text.count(old)
        if n != 1:
            fails.append("SELFTEST BROKEN: %s -- anchor found %d times; re-anchor it, never delete it"
                         % (label, n))
            continue
        rows, bad = read_abi(abi_text.replace(old, new))
        got = set((c, k) for (c, k, _m) in evaluate(rows, bad, packs, sigs)[0]) - base
        for e in expect:
            if e not in got:
                fails.append("SELFTEST: %s must give %s at %s; got %s" % (label, e[0], e[1], sorted(got)[:6]))
    # A void** kernel may never follow the host (its row IS the copy length).  Scratch
    # state: the row AND its host pack both say {8,1,4} -- host leg EQ -- while the CUDA
    # signature says {8,4,1}; naming the kernel in KNOWN_ROW_FOLLOWS_HOST must not waive it.
    old = "static const uint8_t kSapBoxInit[] = { 8, 4, 1 };"
    if abi_text.count(old) != 1:
        fails.append("SELFTEST BROKEN: void**-follows-host case -- anchor not unique")
    else:
        rows, bad = read_abi(abi_text.replace(old, "static const uint8_t kSapBoxInit[] = { 8, 1, 4 };"))
        fake = dict(packs)
        for k in ("initializeSapBox1DLaunch", "updateBodiesLaunch1T", "updateBodiesLaunch_Part2",
                  "updateBodiesLaunch_Part3"):
            fake[k] = [s._replace(sizes=[8, 1, 4]) for s in packs.get(k, [])]
        follows = dict(KNOWN_ROW_FOLLOWS_HOST)
        follows["initializeSapBox1DLaunch"] = "selftest"
        got = set((c, k) for (c, k, _m) in evaluate(rows, bad, fake, sigs, follows=follows)[0])
        if ("SIG-DIFF", "initializeSapBox1DLaunch") not in got:
            fails.append("SELFTEST: a void** kernel named in KNOWN_ROW_FOLLOWS_HOST must still give "
                         "SIG-DIFF; got %s" % sorted(got)[:6])
        if ("DIFF", "initializeSapBox1DLaunch") in got:
            fails.append("SELFTEST BROKEN: the void**-follows-host scratch pack is not EQ to its row")
    # I4 has TWO halves and each must fire on its own: the row half in read_abi (bit <
    # argCount, row[i] == 8) and the host half in evaluate (8 bytes at every site).
    old = "static const uint32_t kCvxTriCoreHostPtrMask = 1u << 18;"
    if abi_text.count(old) == 1:
        rows, bad = read_abi(abi_text.replace(old, "static const uint32_t kCvxTriCoreHostPtrMask = 1u << 15;"))
        if not any(c == "MASK-BAD" and k == "convexTrimeshNarrowphase" for (k, c, _m) in bad):
            fails.append("SELFTEST: read_abi must report MASK-BAD for a mask bit on a 4-byte row entry")
        got = set((c, k) for (c, k, _m) in evaluate(rows, [b for b in bad if b[1] != "MASK-BAD"],
                                                    packs, sigs)[0])
        if ("MASK-BAD", "convexTrimeshNarrowphase") not in got:
            fails.append("SELFTEST: evaluate must report MASK-BAD for a mask bit on a 4-byte HOST argument")
    else:
        fails.append("SELFTEST BROKEN: I4 halves -- anchor not unique")
    # struct_fields: the three by-value writers are laid out, a base class is REFUSED.
    for nm, want in (("PxgFEMContactWriter", (80, 8, [8 * i for i in range(10)])),
                     ("PxgSoftBodyContactWriter", (56, 8, [8 * i for i in range(7)])),
                     ("PxgParticleContactWriter", (72, 8, [8 * i for i in range(9)]))):
        sf = struct_fields(nm)
        if not sf or (sf[0], sf[1], [f[0] for f in sf[2]]) != want:
            fails.append("SELFTEST: struct_fields(%s) gave %s, expected size/align/offsets %s" % (nm, sf, want))
    if struct_fields("PxgRadixSortBlockDesc") is not None:
        fails.append("SELFTEST: struct_fields must REFUSE a struct with a base class (PxgRadixSortBlockDesc)")
    # Part B: the resolver on literal snippets
    for label, src, expect in SELFTEST_SOURCES:
        sf = Source("<selftest:%s>" % label, src)
        got = packs_from_source(sf, SELFTEST_IDS, rel="<selftest>")
        have = {k: sorted((s.form, s.sizes) for s in v) for k, v in got.items()}
        want = {k: sorted(v) for k, v in expect.items()}
        if have != want:
            fails.append("SELFTEST: resolver case '%s' gave %s, expected %s" % (label, have, want))
    for label, src, kernel, expect in SELFTEST_SIGNATURES:
        got = signatures_in(src, "<selftest>")
        st, sizes, _i = signature_of(kernel, got)
        if st != "OK" or sizes != expect:
            fails.append("SELFTEST: signature case '%s' gave %s %s, expected %s" % (label, st, sizes, expect))
    return fails


# ---------------------------------------------------------------------------
def main(argv):
    abi_path = ABI
    if "--abi" in argv:
        abi_path = argv[argv.index("--abi") + 1]
    abi_text = open(abi_path).read()
    ids = read_kernel_ids()
    sigs = cuda_signatures()
    packs = launch_packs(ids)

    fails = selftest(packs, sigs, open(ABI).read())
    if fails:
        print("THE SELFTEST FAILED -- this checker cannot see what it claims to see:")
        for f in fails:
            print("  " + f)
        print()
    if "--selftest-only" in argv:
        print("launchAbiPack selftest: %s (%d scratch-row cases + void**-follows-host + both I4 halves "
              "+ 4 struct-layout pins, %d resolver cases, %d signature cases)"
              % ("FAILED" if fails else "ok", len(SELFTEST_ROWS), len(SELFTEST_SOURCES),
                 len(SELFTEST_SIGNATURES)))
        return 1 if fails else 0

    rows, bad = read_abi(abi_text)
    findings, st, table = evaluate(rows, bad, packs, sigs)

    dump = os.environ.get("PXG_ABI_PACK_DUMP")
    if dump:
        with open(dump, "w") as f:
            json.dump(table, f, indent=1, sort_keys=True)

    order = ["ROW-INTEGRITY", "MASK-BAD", "SEQ-DIFF", "DIFF", "TAIL-EXTRA", "SIG-DIFF", "SIG-NO-DECL",
             "SIG-UNRES", "SIG-MULTI", "NO-SITE", "UNDECIDED", "STALE"]
    heads = {
        "ROW-INTEGRITY": "I1: A ROW DOES NOT SAY WHAT THE BRIDGE READS (argCount, array, entries, duplicates):",
        "MASK-BAD": "I4: A hostPtrMask BIT IS NOT AN 8-BYTE ARGUMENT (translateHostPtrs refuses every launch):",
        "SEQ-DIFF": "I2: SAME TOTAL, DIFFERENT ARGUMENTS -- the shim's gate passes; the row describes another kernel:",
        "DIFF": "I2: A ROW DOES NOT MATCH THE PACK ITS LAUNCH SITE PUSHES:",
        "TAIL-EXTRA": "I3: A HOST PACK HAS EXTRA TRAILING ARGUMENTS AND NO NAMED WAIVER:",
        "SIG-DIFF": "I5: A ROW DOES NOT MATCH ITS CUDA __global__ SIGNATURE:",
        "SIG-NO-DECL": "I5: NO CUDA __global__ DECLARATION FOUND FOR A ROW:",
        "SIG-UNRES": "I5: A CUDA SIGNATURE THIS FILE CANNOT SIZE (teach it, or KNOWN_SIG_UNDECIDED):",
        "SIG-MULTI": "I5: DECLARATIONS OF ONE KERNEL DISAGREE:",
        "NO-SITE": "A ROW WITH NO FOLLOWABLE LAUNCH SITE (no longer a silent skip):",
        "UNDECIDED": "A PACK THAT COULD NOT BE RESOLVED IS NOT COUNTED AS PASSING:",
        "STALE": "A RATCHET ENTRY THAT NO LONGER APPLIES (ratchets shrink; remove it):",
    }
    by = collections.defaultdict(list)
    for c, k, m in findings:
        by[c].append((k, m))
    for c in order + sorted(set(by) - set(order)):
        if by.get(c):
            print(heads.get(c, c + ":"))
            for k, m in sorted(by[c]):
                print("  %s: %s" % (k, m))
            print()

    print("launchAbiPack: %d rows (%d masked); HOST leg: %d PER ARGUMENT over %d sites "
          "(%d sized, %d void**, %d mixed; %d by tail waiver), %d undecided, %d no-site; "
          "SIGNATURE leg: %d equal, %d host-shaped by name, %d unresolved; selftest %s "
          "(ratchets: %d/%d/%d/%d/%d, may shrink, never grow)"
          % (st["rows"], st["masked"], st["host-eq"], st["sites"], st["host-sized"], st["host-void**"],
             st["host-sized/void**"], st["host-tail-waived"], st["undecided"], st["no-site"],
             st["sig-eq"], st["sig-follows-host"], st["sig-undecided"], "FAILED" if fails else "ok",
             len(KNOWN_UNDECIDED), len(KNOWN_NO_SITE), len(KNOWN_TAIL_EXTRA),
             len(KNOWN_ROW_FOLLOWS_HOST), len(KNOWN_SIG_UNDECIDED)))
    if findings or fails:
        print("check_launch_abi_pack: %d finding(s)%s" % (len(findings), ", SELFTEST FAILED" if fails else ""))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
