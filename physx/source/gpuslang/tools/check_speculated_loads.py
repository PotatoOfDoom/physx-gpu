#!/usr/bin/env python3
# ===========================================================================
# check_speculated_loads.py -- find BDA reads whose only protection is a
# PER-INVOCATION guard, which this platform compiles away (P9.23).
#
# WHY THIS EXISTS
#
# P9.22 traced a crash that had been an open task since P8.13 -- filed as
# "sporadic failure under ctest -j" -- to a read that looked perfectly guarded:
#
#     uint countAt(Ptr<uint> counts, uint workIndex, uint nbBodies)
#     {
#         if (workIndex >= nbBodies) return 0u;
#         return counts[workIndex];
#     }
#
# llvmpipe packs eight shader invocations into one AVX register, so a condition
# that differs between them CANNOT be a branch -- it becomes a mask.  The
# generated code was read off the faulting instruction:
#
#     cmp   %r8d,%r9d                       ; index vs bound
#     sbb   %r10d,%r10d                     ; -> predicate
#     kmovd %r10d,%k5
#     vbroadcastss (%rax,%r9,4),%ymm2       ; THE LOAD, UNPREDICATED
#     vblendvps %ymm15,%ymm2,%ymm9,%ymm2    ; the mask, applied AFTER
#
# The guard controls the VALUE, not the ADDRESS.  For a BDA pointer the address
# is a raw process address, so the excluded lanes read wherever the arithmetic
# lands.  Whether that faults is decided by the process layout: with ASLR on it
# faulted 9 times in 200 runs, with `setarch -R` 0 times in 120.
#
# Nothing else in the verification chain can see this.  The fixtures cannot (the
# value is discarded, so every output matches); the mutation sweep cannot (same
# reason); spirv-val cannot (the SPIR-V is valid -- the speculation happens in
# the driver's LLVM back end); and robustBufferAccess would not help even if it
# were enabled, because PhysicalStorageBuffer has no robustness in Vulkan.
#
# WHAT IS REPORTED, and why the split is about DISTANCE
#
# Every site below has the same defect.  What differs -- and what decides whether
# it can ever fault rather than merely read rubbish the select discards -- is HOW
# FAR past the end the speculated address can reach.  So the tiers are about
# distance, and each is a claim the checker can actually establish from the
# function it is looking at:
#
#   LOW-loop    the index is the induction variable of a loop whose condition is
#               the bound: `for (i = tid; i < n; i += stride)`.  The overshoot is
#               at most one stride -- BUT ONLY IF THE LOOP STARTS INSIDE ITS OWN
#               BOUND.  See the P9.49 note below: that assumption was unstated
#               here for four tranches, and it was false for a whole family.
#
#   LOW-launch  the index is assigned IN THIS FUNCTION from a thread-id system
#               value.  The launch is sized to the data rounded up to a
#               workgroup, so the overshoot is less than one workgroup --
#               FOR A ONE-ITEM-PER-THREAD KERNEL ONLY.  P9.97a: a warp-packed
#               kernel (bpw = 32/numThreadsPerBody items per warp, the TGS
#               init/average/propagate family and applyTGSSubstepGravity) is
#               launched with a grid sized for 32 threads PER ITEM, so its
#               bodyId reaches 8*G*bpw, i.e. up to ~32x the item count at
#               bpw 32 -- about 31T float4 past a 2T array in
#               initializeSolverVelocitiesTGS.  Those three kernels therefore
#               clamp their load indices since P9.97a (bodyIdL = min(bodyId,
#               bound - 1u)) and are no longer reported.
#
#   LOW-const   the guard's bound is a compile-time constant, so the overshoot is
#               at most that constant.
#
#   HIGH        the index is a PARAMETER of a non-entry function, or comes from
#               memory.  Nothing in this function relates it to the buffer's
#               size -- that relation lives in a caller -- so nothing here bounds
#               the distance.  This is exactly P9.22's site: countAt takes
#               workIndex as a parameter, and the caller drives it to 32768
#               against nbBodies 20000, i.e. ~51 KB past the end.
#
# The LOW tiers are NOT "safe".  They are undefined behaviour with a bounded
# reach, which on this platform means they land in mapped memory and are
# invisible.  They are separated so the HIGH list stays something a person can
# act on -- a list of 349 that cannot be worked through is not a gate.
#
# P9.49 -- WHERE THE LOW-loop TIER WAS WRONG, AND WHAT IT COST
#
# "Overshoot <= one stride" reads like a fact about the loop.  It is not: it is a
# fact about the loop TOGETHER WITH an assumption nobody wrote down, that the
# first index is already inside the bound.  For `for (i = tid; ...)` that holds --
# tid is block-local.  For the midphase family it does not:
#
#     const uint globalWarpIdx = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub;   // [0, 2048)
#     for (uint vertexIdx = globalWarpIdx; vertexIdx < numVerts; vertexIdx += SM_NB_WARPS)
#
# The grid is PERSISTENT -- 1024 blocks, fixed by the kernel, the same whatever
# the data is -- so a warp whose number exceeds numVerts starts past the end and
# the loop's first test already fails.  The reach is (start - bound) + k*stride,
# and with a 2048-warp grid over 1100 vertices that is thousands of elements, not
# one stride.
#
# What made it reachable rather than merely undefined is the BARRIER at the top of
# those bodies: this platform has to reach it even for a warp with no work, so it
# runs the body with an all-inactive mask, and the read -- uniform across the
# (empty) set of active lanes -- is issued as a scalar load at "the first active
# lane, or lane 0 if there is none".
#
# It was measured, not argued.  From the core: `vmovss (%rcx,%rax,1)` with
# rcx = 3584*16, i.e. index 3584 = 1536 + 2048 against an 1100-element buffer --
# 39 KB past the end, faulting about one run in twenty.  Two separate sites in one
# test binary (femClothMeshVertMidphaseBda and femClothPsMidphaseBda), and both
# had sat in this tier, counted and dismissed, since P9.23.
#
# The lesson is narrower than "the tier was wrong": a tier that encodes a DISTANCE
# argument has to state the premise the distance rests on, or it silently launders
# an assumption into a clean bill.  persistent_grid_loop() now tests that premise
# instead of assuming it, and a loop that fails it is HIGH.
#
# A read is NOT reported when its index is CLAMPED -- assigned from min/clamp or
# from a ternary with a constant fallback.  That is what the P9.22 fix does, and
# a checker that re-flags the fix teaches people to ignore it.
#
# WHAT THIS CHECK DOES NOT SEE, written down because a list without its gaps
# gets read as completeness:
#
#   * ADDRESS ARITHMETIC.  `Ptr<uint>(base + i * STRIDE)[0]` has index 0 and
#     falls straight through; the overrun lives in `base`.  P9.23 clamped SIX
#     loops in femClothRigidContactBda / softBodyRigidContactBda where this check
#     could only see FOUR sites, for exactly this reason.
#     PARTLY CLOSED at P9.50: analyse_helper_calls() catches the case where the
#     arithmetic is wrapped in a function -- `byteAt(base, i)`, whose body is
#     `Ptr<uint>((base + i) & ~3)[0]` -- but ONLY inside a persistent-grid loop.
#     Four such reads existed when P9.49 was investigated, in the same family that
#     was faulting, and every one of them was found by an ad-hoc grep rather than
#     by this file.
#   * the INLINE form `Ptr<T>(addr)[i]` -- only named Ptr<> values are tracked.
#   * DESCRIPTOR-BOUND buffers.  `RWStructuredBuffer<T> b; ... b[i]` is not
#     scanned at all, and that limit is DELIBERATE AND MEASURED, not an oversight:
#     tests/testRobustProbe.cpp reads such a buffer 1 GiB past its allocation and
#     gets 0x00000000 back, with the process intact, so those accesses are bounds-
#     checked here even though robustBufferAccess is not enabled.  The measurement
#     exists because the log had asserted BOTH answers -- twice that robustness is
#     off, once that descriptor-bound SSBOs are "robust bounds-checked" on the
#     strength of a spirv-dis run that only showed they are not BDA.  55 findings
#     had been refuted partly on that claim.  Re-run the probe on any driver
#     change: PXG_PROBE_ROBUST=1 ./gpuslang-tests robustProbe.
#   * `else` branches: the condition of an else-block is not inverted and tested.
#   * ESCAPE guards are matched FLOW-INSENSITIVELY: an `if (...) return;` earlier
#     in the same function counts even where it does not dominate the read.  That
#     is deliberate -- P9.22's site had exactly that shape, and being
#     flow-sensitive would have missed it -- but it can over-report.
#
# USAGE
#   check_speculated_loads.py            check; exit 1 on any un-waived HIGH
#   check_speculated_loads.py -v         also list the LOW tier
#   check_speculated_loads.py --selftest-only
# ===========================================================================

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from check_barrier_uniformity import (  # noqa: E402
    strip_comments, header_before, braceless_guard, line_index, taint,
    norm_ids, Shader, CTRL, LOOPS, FOR_PARTS,
)

SHADERS = os.path.normpath(os.path.join(HERE, '..', 'shaders'))
WAIVERS = os.path.join(HERE, 'speculated_load_waivers.txt')

# A BDA pointer value: either a declared local/parameter, or the inline form.
PTR_DECL = re.compile(r'\bPtr<[^<>]*(?:<[^<>]*>)?[^<>]*>\s+([A-Za-z_]\w*)')
IDENT = re.compile(r'(?<![.\w])([A-Za-z_]\w*)')

# Names that are index arithmetic, not variables.
NOT_A_VAR = {
    'u', 'f', 'uint', 'int', 'float', 'uint2', 'uint3', 'uint4', 'min', 'max',
    'clamp', 'abs', 'asuint', 'asint', 'asfloat', 'uint64_t', 'true', 'false',
}
CMP_OPS = r'(?:<=|<|>=|>|==|!=)'


def index_roots(expr):
    return [v for v in IDENT.findall(expr) if v not in NOT_A_VAR]


def bounds(kind, cond, var):
    """Does `cond` bound `var` FROM ABOVE at the read?

    The direction matters and the first draft of this checker ignored it, which
    is worth writing down because the failure mode was silent.  `bodyVelocities[
    outputBody + totalBodyCount + bodyOffset]` under `if (bodyId < totalBodyCount)`
    matched on `totalBodyCount` -- the BOUND appearing in the index expression --
    and was reported as if the guard constrained the index.  Four of the first
    twenty-two findings were that same mistake.

    So the side and the operator are both read, and combined with the guard kind:
      escape `if (i >= n) return;`  -> after it, i < n         (bounded)
      escape `if (n < i) return;`   -> after it, i <= n        (bounded)
      body   `if (i < n) { ... }`   -> inside, i < n           (bounded)
      body   `if (n > i) { ... }`   -> inside, i < n           (bounded)
    and the four mirror images bound i from BELOW, which is no protection at all.
    """
    v = re.escape(var)
    for m in re.finditer(r'(?<![.\w])' + v + r'\s*(<=|<|>=|>)', cond):
        op = m.group(1)
        if (op in ('>=', '>')) == (kind == 'escape'):
            return True
    for m in re.finditer(r'(<=|<|>=|>)\s*' + v + r'(?![\w])', cond):
        op = m.group(1)
        if (op in ('<=', '<')) == (kind == 'escape'):
            return True

    # A SENTINEL guard.  `if (primitiveIndex != 0xffffffffu) { indices[primitiveIndex] }`
    # is a per-lane guard on a BDA read exactly like a range check, and it is WORSE:
    # the excluded lanes do not merely run past the end, they carry 0xffffffff, so
    # the speculated address is base + 0xffffffff * stride -- SIXTY-FOUR GIGABYTES
    # out for a uint4 array.  Whether that faults depends on the process layout,
    # which is why it shows up as a sporadic crash under concurrency and not as a
    # deterministic one.
    #
    # This was missed for a simple reason: the loop above asks whether the guard
    # BOUNDS the index, and `!=` against a constant does not bound anything, so
    # every such site was skipped rather than tiered.  Found in P9.28 by tracing an
    # actual reproducible SIGSEGV back to femClothMeshVertMidphaseBda, then
    # grepping the corpus and finding ELEVEN shaders with the same shape.
    #
    # Deliberately narrow: only a comparison against a large constant sentinel
    # counts.  `!= 0` is the null-pointer idiom this port uses everywhere and is
    # not this bug.
    for m in re.finditer(r'(?<![.\w])' + v + r'\s*(==|!=)\s*(0x[fF]{6,8}u?|4294967295u?)', cond):
        if (m.group(1) == '==') == (kind == 'escape'):
            return True
    for m in re.finditer(r'(0x[fF]{6,8}u?|4294967295u?)\s*(==|!=)\s*' + v + r'(?![\w])', cond):
        if (m.group(2) == '==') == (kind == 'escape'):
            return True
    return False


CLAMPED = re.compile(r'\b(?:min|clamp)\s*\(')

# `]` [member/index chain] `=`  -- a plain store.  Compound assignment is NOT a
# store for this purpose: it reads first.
STORE_TAIL = re.compile(r'\s*(?:\s*(?:\.\w+|\[[^\]]{0,40}\]))*\s*=(?!=)')


def clamped_vars(body):
    """Variables whose value cannot leave a valid range by construction:
    assigned from min()/clamp(), or from a ternary with a constant fallback
    (`cond ? i : 0u`) -- which is exactly the shape of the P9.22 fix."""
    out = set()
    for m in re.finditer(r'(?:(?<=[;{}()])|^)\s*(?:const\s+)?(?:[A-Za-z_]\w*\s+)?'
                         r'([A-Za-z_]\w*)\s*(?<![=!<>+\-*/&|^])=(?!=)\s*([^;]{1,200});', body):
        var, rhs = m.group(1), m.group(2)
        if CLAMPED.search(rhs):
            out.add(var)
        elif re.search(r'\?[^:?]{1,60}:\s*(?:0u|0|0\.0f|0\.f)\s*$', rhs.strip()):
            out.add(var)
    return out


def enclosing_guards(src, pos, start):
    """(kind, condition) for every control header whose body encloses `pos`,
    innermost first, stopping at the function's own opening brace `start`."""
    out = []
    depth, i = 0, pos
    while i > start:
        i -= 1
        c = src[i]
        if c == '}':
            depth += 1
        elif c == '{':
            if depth:
                depth -= 1
                continue
            kw, header = header_before(src, i)
            if kw in CTRL:
                out.append(('loop' if kw in LOOPS else kw, header))
    return out


def escape_guards(src, pos, start):
    """`if (...) return/continue/break;` guards that precede `pos` inside the
    same function.  Flow-insensitive on purpose: an escape earlier in the body
    is the shape the P9.22 site had, and missing it would miss the finding."""
    out = []
    for m in re.finditer(r'\bif\s*\(', src[start:pos]):
        a = start + m.end() - 1
        depth, k = 0, a
        while k < pos:
            if src[k] == '(':
                depth += 1
            elif src[k] == ')':
                depth -= 1
                if depth == 0:
                    break
            k += 1
        if k >= pos:
            continue
        tail = src[k + 1:k + 60]
        if re.match(r'\s*(return|continue|break)\b', tail):
            out.append(('escape', ' '.join(src[a + 1:k].split())))
    return out


def loop_induction(header):
    """(var, condition) of a `for` header, or (None, None)."""
    m = FOR_PARTS.match(header)
    if not m:
        return None, None
    return m.group(1), m.group(3)


# A start that grows with the LAUNCH rather than with the block.  These are the
# names this corpus builds a grid-wide index from.
GRIDWIDE_START = re.compile(r'\bglobalWarpIdx\b|\bglobalThreadIdx\b|\bgid\b'
                            r'|\bgroupId\b|\bdtid\b|\bblockIdx\b|\bdispatchThreadId\b')
# A stride that is sized to the DATA: the engine computed the grid from the count.
DATA_SIZED_STRIDE = re.compile(r'WorkgroupCount|GroupCount|gridDim|gridStride|numWarps')


# A function that takes a uint64_t base and dereferences a Ptr<> is a BDA READ
# WEARING A CALL.  The scan below cannot see one: there is no `namedPtr[expr]`
# anywhere at the call site, the index reaches the pointer as an argument, and
# inside the helper the read is an INLINE Ptr<uint>(addr)[0] with a constant
# index.  All three of those are on this file's own list of what it does not see.
BDA_HELPER_PARAM = re.compile(r'\buint64_t\b')


def bda_reading_helpers(src):
    """Names of functions in this shader that read through a BDA pointer built
    from a uint64_t parameter.

    `byteAt` is the one this corpus actually uses:

        uint byteAt(uint64_t base, uint idx)
        {
            const uint64_t addr = base + uint64_t(idx);
            const uint word = Ptr<uint>(addr & ~uint64_t(3))[0];

    P9.49 clamped four call sites of it inside persistent-grid loops.  Every one
    was a genuine out-of-range read of the same family that was faulting, and not
    one of them was reported here -- they were found by an ad-hoc grep, which is
    not a thing to rely on twice.
    """
    out = set()
    for m in re.finditer(r'\b(?:uint|int|float|uint2|uint3|uint4|float2|float3'
                         r'|float4|bool|uint64_t)\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*\{',
                         src):
        name, params = m.group(1), m.group(2)
        if not BDA_HELPER_PARAM.search(params):
            continue
        i = src.find('{', m.end() - 1)
        d, end = 0, len(src)
        for j in range(i, len(src)):
            if src[j] == '{':
                d += 1
            elif src[j] == '}':
                d -= 1
                if d == 0:
                    end = j
                    break
        if 'Ptr<' in src[i:end]:
            out.add(name)
    return out


def const_uint(src, name):
    m = re.search(r'static\s+const\s+uint\s+%s\s*=\s*([^;]+);' % re.escape(name), src)
    return m.group(1).strip() if m else None


def persistent_grid_loop(src, header):
    """Does this loop's FIRST index already sit past its own bound?

    P9.49.  LOW-loop's whole claim is "overshoot <= one stride", and that claim
    silently assumes the loop starts INSIDE the bound.  Two things together break
    it, and both have to be read off the header:

      * the start is a GRID-WIDE id -- it grows with the launch, not with the
        workgroup, so it is not bounded by a block size;
      * the stride is a COMPILE-TIME CONSTANT -- a persistent grid, whose warp
        count the kernel fixes and the data never enters.

    Then the first index of a warp lies somewhere in [0, gridWarps) while the
    bound is a data count, the two are unrelated, and the reach is
    (start - bound) + k*stride.  Measured on the midphase family: index 3584
    against a 1100-element buffer, 39 KB past the end, faulting about one run in
    twenty.  A start of `lane` or `tid` cannot exceed the bound by more than a
    block, and a stride built from WorkgroupCount() means the launch was sized to
    the data; neither is reported.
    """
    m = FOR_PARTS.match(header)
    if not m:
        return None
    init, step = m.group(2).strip(), m.group(4).strip()
    if not GRIDWIDE_START.search(init):
        return None
    # group(4) is the whole increment clause -- `i += STRIDE`, not `STRIDE`.
    # Reading it as the stride made every site fall through the constant test
    # below and kept the tier silently unchanged, which is the exact failure this
    # split exists to end.
    sm = re.match(r'^\s*[A-Za-z_]\w*\s*\+=\s*(.+)$', step)
    if not sm:
        return None            # ++ / -- : the stride is 1, never a persistent grid
    step = sm.group(1).strip()
    expanded = step
    for _ in range(3):
        nxt = expanded
        for nm in set(re.findall(r'\b[A-Za-z_]\w*\b', expanded)):
            val = const_uint(src, nm)
            if val:
                nxt = re.sub(r'\b%s\b' % re.escape(nm), '(%s)' % val, nxt)
        if nxt == expanded:
            break
        expanded = nxt
    if DATA_SIZED_STRIDE.search(expanded):
        return None
    if not re.fullmatch(r'[\s\d\(\)\*\+u]+', expanded):
        return None       # cannot show it is a constant -- do not claim it is
    return init, step


# The system values a launch geometry is derived from.  A variable assigned from
# one of these IN THIS FUNCTION is bounded by the launch: the dispatch is the
# data size rounded up to a workgroup, so the index cannot exceed the bound by
# more than one workgroup.
# The names actually bound to an SV_ semantic in this corpus, MEASURED not guessed:
#   dtid 327, groupThreadId 200, groupId 177, gtid 154, gid 147,
#   dispatchThreadId 65, tid 14, groupID 5, gdim 1.
# The pattern used to cover dtid/gtid/tid/gid plus a `thread` stem that needs a word
# boundary before it -- so it matched neither `groupId` nor `globalThreadIdx`, and an
# index built by hand as `groupThreadId.x + groupId.x * 64u` looked unrooted while the
# guard on it looked UNIFORM.  Both mistakes point the same way: such a site was
# skipped entirely rather than tiered.  Adding the names changes no verdict today
# (the corpus still reports zero un-waived HIGH), so this is a latent gap closed
# before it bites, not a fix with a finding behind it -- said plainly because a
# checker's credibility is exactly its record of not overstating.
SV_SEED = re.compile(r'(?i)\b(?:SV_)?(?:dispatch|group)?thread(?:id|idx|index)?\b|'
                     r'\b(?:dtid|gtid|tid|gid|gdim|groupID|groupId|groupIdx|'
                     r'groupThreadId|dispatchThreadId|globalThreadIdx|globalThreadIndex|'
                     r'globalIdx|globalIndex|groupIndex)\b')
PARAM_SPLIT = re.compile(r',(?![^<]*>)')


def params_of(header):
    """Parameter names of a function header `name(a, b, c)`."""
    i = header.find('(')
    if i < 0:
        return set()
    inner = header[i + 1:header.rfind(')')]
    out = set()
    for part in PARAM_SPLIT.split(inner):
        m = re.search(r'([A-Za-z_]\w*)\s*$', part.strip())
        if m:
            out.add(m.group(1))
    return out


def launch_rooted(body, var, depth=0):
    """Is EVERY assignment of `var` in this body rooted in a thread id?

    EVERY, not any -- and that distinction is the whole point.  The first version
    asked "any", and it let this through:

        uint primitiveIndex = 0xffffffffu;          // <- a SENTINEL
        if (lane < nbPrimitives)
            primitiveIndex = lane + primStartIndex; // <- launch-rooted
        if (primitiveIndex != 0xffffffffu)
            const uint4 mTri = meshIndices[primitiveIndex];

    One assignment IS a thread id, so the site was tiered LOW-launch with the
    reason "overshoot < one workgroup".  The excluded lanes do not carry a thread
    id at all: they carry 0xffffffff, and llvmpipe's mask issues the load for them
    anyway, at base + 0xffffffff * 16 -- SIXTY-FOUR GIGABYTES out.  Whether that
    faults depends on the process layout, which is why P9.28 found it as a
    sporadic SIGSEGV under concurrency rather than as a deterministic one.

    Requiring every assignment is what an "overshoot" claim actually needs: it is
    a statement about the WORST value the index can take, and a single well-behaved
    assignment says nothing about that.
    """
    if depth > 6:
        return False
    seen = False
    for m in re.finditer(r'(?:(?<=[;{}()])|^)\s*(?:const\s+)?(?:[A-Za-z_][\w<>:]*\s+)?' +
                         re.escape(var) + r'\s*(?<![=!<>+\-*/&|^])=(?!=)\s*([^;]{1,200});', body):
        rhs = m.group(1)
        seen = True
        if SV_SEED.search(rhs):
            continue
        if any(r != var and launch_rooted(body, r, depth + 1) for r in index_roots(rhs)):
            continue
        return False        # an assignment nothing bounds -- e.g. a sentinel
    # `for (var = tid; ...)`
    for m in re.finditer(r'\bfor\s*\([^;]*\b' + re.escape(var) + r'\s*=\s*([^;]{1,120});', body):
        if SV_SEED.search(m.group(1)):
            seen = True
    return seen


CONSTBOUND = re.compile(r'(?:<=|<|>=|>|==|!=)\s*(\d+u?|[A-Z][A-Z0-9_]{2,})\s*\)?\s*$')


class Finding(object):
    def __init__(self, shader, line, func, ptr, expr, kind, cond, tier, why):
        self.shader, self.line, self.func = shader, line, func
        self.ptr, self.expr = ptr, expr
        self.kind, self.cond, self.tier, self.why = kind, cond, tier, why

    def key(self):
        return '%s|%s|%s[%s]' % (self.shader, self.func, self.ptr, self.expr)

    def __str__(self):
        return ('%-34s %-28s %s[%s]\n        guarded by %s: %s\n        %s'
                % ('%s:%d' % (self.shader, self.line), self.func, self.ptr,
                   self.expr, self.kind, self.cond, self.why))


def _brace_range(src, pos):
    i = src.find('{', pos)
    if i < 0:
        return None, None
    d = 0
    for j in range(i, len(src)):
        if src[j] == '{':
            d += 1
        elif src[j] == '}':
            d -= 1
            if d == 0:
                return i, j
    return None, None


def analyse_helper_calls(sh):
    """P9.50: BDA reads that arrive as a CALL, inside a persistent-grid loop.

    Deliberately narrow.  The same call anywhere else in the corpus is not
    reported, because outside a persistent-grid loop the distance argument that
    makes LOW-loop acceptable still holds, and a rule that fired everywhere would
    bury the four-a-decade findings that matter under hundreds that do not.
    """
    src = sh.src
    readers = bda_reading_helpers(src)
    if not readers:
        return []
    findings = []
    for m in re.finditer(r'\bfor\s*\([^;]*;[^;]*;[^)]*\)', src):
        header = ' '.join(m.group(0).split())
        var, _cond = loop_induction(header)
        if not var:
            continue
        pg = persistent_grid_loop(src, header)
        if not pg:
            continue
        i, j = _brace_range(src, m.end())
        if i is None:
            continue
        body = src[i:j]
        safe = clamped_vars(body)
        v = re.escape(var)
        for c in re.finditer(r'\b([A-Za-z_]\w*)\s*\(([^()]*)\)', body):
            fname, args = c.group(1), c.group(2)
            if fname not in readers:
                continue
            hit = None
            for arg in args.split(','):
                a = arg.strip()
                if a == var or (re.search(r'(?<![.\w])' + v + r'(?![\w])', a)
                                and not CLAMPED.search(a)):
                    if a in safe:
                        continue
                    hit = a
                    break
            if hit is None:
                continue
            pos = i + c.start()
            enclosing = '?'
            for fn, occ in sh.funcs.items():
                for (fs, fe) in occ:
                    if fs <= pos < fe:
                        enclosing = fn
            findings.append(Finding(
                sh.name, sh.lineof(pos), enclosing,
                fname, hit, 'loop', header, 'HIGH',
                "P9.50: %s() reads through a BDA pointer built from its uint64_t "
                "argument, so this call IS an indexed load -- one with no `[` at the "
                "call site, which is why the scan above cannot see it.  It sits in a "
                "persistent-grid loop (start %s, stride %s), so the index can be past "
                "the bound on the very first pass.  Clamp the ARGUMENT."
                % (fname, pg[0], pg[1])))
    return findings


def analyse_shader(sh):
    src = sh.src
    findings = analyse_helper_calls(sh)
    for name, occ in sh.funcs.items():
        for (start, end) in occ:
            body = src[start:end]
            lane, warp = taint(body, frozenset())
            perInv = lane | warp
            safe = clamped_vars(body)
            ptrs = set(PTR_DECL.findall(body))
            # a Ptr parameter is declared in the header, not the body
            _, header = header_before(src, start)
            ptrs |= set(PTR_DECL.findall(header))
            if not ptrs:
                continue
            for ptr in ptrs:
                for m in re.finditer(re.escape(ptr) + r'\s*\[\s*([^\]\[]{1,80})\s*\]', body):
                    expr = m.group(1).strip()
                    # A STORE is masked, never speculated -- the hazard is loads.
                    #
                    # The member chain has to be skipped first, and leaving it out
                    # cost this checker thirteen of its first eighteen findings:
                    # `dbg[globalWarpIndex].rotOut0 = rot0;` is a store, but the
                    # `=` is not adjacent to the `]`.  A COMPOUND assignment stays
                    # a finding -- `p[i].x += v` reads before it writes.
                    if STORE_TAIL.match(body[m.end():m.end() + 96]):
                        continue
                    roots = index_roots(expr)
                    if not roots or (set(roots) & safe) or CLAMPED.search(expr):
                        continue
                    pos = start + m.start()
                    guards = enclosing_guards(src, pos, start) + \
                        escape_guards(src, pos, start)
                    hit = None
                    for (kind, cond) in guards:
                        for r in roots:
                            if not bounds(kind, cond, r):
                                continue
                            if not (norm_ids(cond) & perInv):
                                continue    # uniform guard: stays a real branch
                            hit = (kind, cond, r)
                            break
                        if hit:
                            break
                    if not hit:
                        continue
                    kind, cond, root = hit
                    tier, why = 'HIGH', None
                    if kind == 'loop' and loop_induction(cond)[0] == root:
                        pg = persistent_grid_loop(src, cond)
                        if pg:
                            tier = 'HIGH'
                            why = ("P9.49: this loop STARTS at a grid-wide id (%s) and "
                                   "strides by a compile-time constant (%s), so its FIRST "
                                   "index can already be past the bound -- the overshoot is "
                                   "(start - bound) + k*stride, not one stride.  Clamp the "
                                   "ADDRESS." % (pg[0], pg[1]))
                        else:
                            tier, why = 'LOW-loop', ("the index is this loop's own "
                                                     "induction variable: overshoot <= one "
                                                     "stride (the loop starts inside its "
                                                     "own bound -- see persistent_grid_loop)")
                    elif launch_rooted(body, root):
                        tier, why = 'LOW-launch', ('the index is assigned here from a '
                                                   'thread id: overshoot < one workgroup')
                    elif CONSTBOUND.search(cond):
                        tier, why = 'LOW-const', ('the bound is a compile-time constant: '
                                                  'overshoot <= %s'
                                                  % CONSTBOUND.search(cond).group(1))
                    else:
                        # The index may be bounded by an ENCLOSING LOOP rather
                        # than by the guard.  `for (ind = 0; ind < 3u; ++ind)`
                        # with `if (ind < dofCount)` inside caps the speculated
                        # address at three elements whatever dofCount is -- the
                        # guard is per-lane, the loop's trip count is not.
                        for (gk, gc) in enclosing_guards(src, pos, start):
                            if gk != 'loop':
                                continue
                            iv, ivcond = loop_induction(gc)
                            if iv == root and ivcond and CONSTBOUND.search(ivcond + ')'):
                                tier = 'LOW-const'
                                why = ('the index is bounded by an enclosing loop over a '
                                       'compile-time constant (%s): overshoot <= %s'
                                       % (' '.join(ivcond.split()),
                                          CONSTBOUND.search(ivcond + ')').group(1)))
                                break
                    if tier == 'HIGH' and why is None:
                        # `why` is already set for a P9.49 persistent-grid loop, and
                        # that reason is the specific one -- do not flatten it back
                        # into the generic "nothing bounds the distance".
                        params = params_of(header)
                        why = ('the index is a PARAMETER (%s) -- the relation to the '
                               'buffer size lives in a caller, so nothing here bounds '
                               'the distance' % root) if root in params else \
                              ('nothing in this function relates the index to the '
                               'buffer size, so nothing bounds the distance')
                    findings.append(Finding(sh.name, sh.lineof(pos), name, ptr,
                                            expr, kind, cond, tier, why))
    return findings


def load_corpus(mutate=None):
    shaders = []
    for fn in sorted(os.listdir(SHADERS)):
        if not fn.endswith('.slang'):
            continue
        path = os.path.join(SHADERS, fn)
        text = None
        if mutate and mutate[0] == fn:
            text = open(path).read().replace(mutate[1], mutate[2])
            if text == open(path).read():
                return None
        shaders.append(Shader(path, text))
    return shaders


def load_waivers():
    out = {}
    if not os.path.exists(WAIVERS):
        return out
    for raw in open(WAIVERS):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 2:
            out[parts[0]] = parts[-1]
    return out


# The mutation is the P9.22 bug, put back: the clamped read becomes the branch
# spelling that crashed.  A checker that is silent on fixed code proves nothing
# until it is shown loud on the broken code it exists for.
SELFTEST_FILE = 'npRigidSumBatches.slang'
SELFTEST_FROM = """    if (nbBodies == 0u)
        return 0u;
    const bool inRange = workIndex < nbBodies;
    const uint safeIndex = inRange ? workIndex : 0u;
    const uint value = counts[safeIndex];
    return inRange ? value : 0u;"""
SELFTEST_TO = """    if (workIndex >= nbBodies)
        return 0u;
    return counts[workIndex];"""


def selftest_persistent_grid():
    """Negative control for the P9.49 split.

    Four cases, and each one is a way the split could be wrong rather than merely
    incomplete.  The first two say it fires where it must; the last two say it
    stays quiet where the distance argument really does hold -- because a split
    that promoted every grid-stride loop would move 61 findings into the gate,
    read as thorough, and mean nothing.
    """
    src = ('static const uint STRIDE = 1024u * 2u;\n'
           'static const uint GRID = 512u;\n'
           'static const uint BLK = 256u;\n')
    cases = [
        # (header, expected-to-fire, what the case is for)
        ('for (uint v = globalWarpIdx; v < numVerts; v += STRIDE)', True,
         'the measured shape: grid-wide start, constant stride'),
        ('for (uint i = globalThreadIdx; i < numPairs; i += GRID * BLK)', True,
         'the same, with the stride written as a product of two constants'),
        ('for (uint i = tid; i < n; i += STRIDE)', False,
         'a BLOCK-LOCAL start cannot exceed the bound by more than a block'),
        ('for (uint i = idx; i < gNumShapes; i += BLK * WorkgroupCount().x)', False,
         'a stride built from WorkgroupCount() means the launch was sized to the data'),
    ]
    ok = True
    for header, want, why in cases:
        got = persistent_grid_loop(src, header) is not None
        if got != want:
            print('SELFTEST FAILED (P9.49): %r -> %s, expected %s\n  case: %s'
                  % (header, got, want, why))
            ok = False
    # A stride this checker cannot resolve to a constant must NOT be claimed as
    # one: an unresolvable name is unknown, not proven fixed.
    if persistent_grid_loop(src, 'for (uint i = dtid.x; i < n; i += mysteryStride)'):
        print('SELFTEST FAILED (P9.49): an UNRESOLVED stride was treated as a '
              'compile-time constant.  The tier would then claim a persistent grid '
              'it never established.')
        ok = False
    return ok


def selftest_helper_calls():
    """Negative control for the P9.50 rule.

    The rule exists because four real out-of-range reads wore a call and none of
    them was reported; so the control re-creates that exact shape and requires a
    finding.  The three quiet cases matter as much: a helper that takes no
    uint64_t is not a BDA read, a clamped argument is the FIX, and the same call
    outside a persistent-grid loop is the LOW-loop tier, where the distance
    argument still holds.  A rule that fired on those would put the corpus's whole
    byteAt population into the gate and mean nothing.
    """
    prelude = '''
    static const uint STRIDE = 1024u * 2u;
    uint byteAt(uint64_t base, uint idx)
    {
        const uint64_t addr = base + uint64_t(idx);
        const uint word = Ptr<uint>(addr & ~uint64_t(3))[0];
        return (word >> 8u) & 0xffu;
    }
    uint plainAdd(uint a, uint b) { return a + b; }
    '''
    cases = [
        ('void k() { for (uint v = globalWarpIdx; v < n; v += STRIDE) {'
         ' if (byteAt(hint, v) == 0u) continue; } }', 1,
         'the measured shape: a BDA-reading helper called with the raw index'),
        ('void k() { for (uint v = globalWarpIdx; v < n; v += STRIDE) {'
         ' const uint s = min(v, max(n, 1u) - 1u);'
         ' if (byteAt(hint, s) == 0u) continue; } }', 0,
         'the P9.49 fix itself must not be reported back as the defect'),
        ('void k() { for (uint v = globalWarpIdx; v < n; v += STRIDE) {'
         ' if (byteAt(hint, min(v, n - 1u)) == 0u) continue; } }', 0,
         'the clamp written inline at the call site counts too'),
        ('void k() { for (uint v = globalWarpIdx; v < n; v += STRIDE) {'
         ' const uint q = plainAdd(v, 1u); } }', 0,
         'a helper that takes no uint64_t reads no BDA pointer'),
        ('void k() { for (uint v = tid; v < n; v += STRIDE) {'
         ' if (byteAt(hint, v) == 0u) continue; } }', 0,
         'a block-local start keeps the LOW-loop distance argument intact'),
        ('void k() { for (uint v = globalWarpIdx; v < n; v += BLK * WorkgroupCount().x) {'
         ' if (byteAt(hint, v) == 0u) continue; } }', 0,
         'a data-sized stride means the launch was sized to the data'),
    ]
    ok = True
    for body, want, why in cases:
        sh = Shader(os.path.join(SHADERS, '_selftest.slang'), text=prelude + body)
        got = len(analyse_helper_calls(sh))
        if got != want:
            print('SELFTEST FAILED (P9.50): %d finding(s), expected %d\n  case: %s'
                  % (got, want, why))
            ok = False
    return ok


def selftest():
    if not selftest_persistent_grid():
        return 2
    if not selftest_helper_calls():
        return 2
    clean = load_corpus()
    base = {f.key() for f in sum((analyse_shader(s) for s in clean), [])
            if f.tier == 'HIGH'}
    broken = load_corpus((SELFTEST_FILE, SELFTEST_FROM, SELFTEST_TO))
    if broken is None:
        print('SELFTEST BROKEN: the anchor is no longer in %s.' % SELFTEST_FILE)
        print('  Re-anchor it on the current spelling of countAt; do NOT delete '
              'the selftest.')
        return 2
    after = {f.key() for f in sum((analyse_shader(s) for s in broken), [])
             if f.tier == 'HIGH'}
    gained = after - base
    if not any('countAt' in g for g in gained):
        print('SELFTEST FAILED: putting the P9.22 bug back did NOT produce a HIGH '
              'finding in countAt.  The check cannot see the defect it exists for.')
        return 2
    print('selftest: re-injecting the P9.22 read makes countAt a HIGH finding.  '
          'The check can still see it.')
    return 0


def main():
    verbose = '-v' in sys.argv or '--verbose' in sys.argv
    if '--selftest-only' in sys.argv:
        return selftest()
    rc = selftest()
    if rc:
        return rc

    shaders = load_corpus()
    findings = sum((analyse_shader(s) for s in shaders), [])
    waivers = load_waivers()
    high = [f for f in findings if f.tier == 'HIGH']
    low = [f for f in findings if f.tier != 'HIGH']
    live = [f for f in high if f.key() not in waivers]
    waived = [f for f in high if f.key() in waivers]

    print('\nspeculated-load check over %d shaders' % len(shaders))
    print('  HIGH  %3d  (%d waived)' % (len(high), len(waived)))
    for t in ('LOW-loop', 'LOW-launch', 'LOW-const'):
        n = len([f for f in findings if f.tier == t])
        print('  %-11s %3d' % (t, n))

    if verbose and low:
        print('\n--- LOW ---')
        for f in low:
            print('  %s' % f)
    if waived:
        print('\n--- WAIVED ---')
        for f in waived:
            print('  %s\n        WAIVED: %s' % (f, waivers[f.key()]))
    if live:
        print('\n--- HIGH, un-waived ---')
        for f in live:
            print('  %s' % f)
        print('\n%d un-waived HIGH finding(s).  Each is a BDA read whose guard the\n'
              'platform turns into a mask, so the out-of-range address is still\n'
              'formed and dereferenced.  Fix by CLAMPING the index and applying the\n'
              'guard to the value (see countAt in npRigidSumBatches.slang), or add a\n'
              'waiver line saying why the index cannot leave the allocation.'
              % len(live))
        return 1
    print('\nOK -- no un-waived HIGH findings.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
