#!/usr/bin/env python3
# ===========================================================================
# check_barrier_uniformity.py -- static audit for non-uniform OpControlBarrier
#
# WHY THIS EXISTS (P9.15a, in one paragraph)
#
# The CUDA this port follows aggregates with __ballot_sync / warpScanExclusive.
# Those need only WARP-uniform control flow, and CUDA kernels are written to
# supply exactly that -- a loop bound rounded up to 32, for instance, gives
# every lane of a warp the same trip count, which is all the intrinsic asks
# for.  We have no warp intrinsics.  We emulate them with `groupshared` plus
# GroupMemoryBarrierWithGroupSync(), and that compiles to OpControlBarrier with
# Workgroup execution scope, which SPIR-V and Vulkan require to be reached in
# control flow uniform across the WHOLE BLOCK.
#
# So the emulation MOVES THE SYNCHRONISATION DOMAIN from warp to block, and a
# transcription that is correct in CUDA can be undefined here.  In a 512-thread
# block with `(numVerts + 31) & ~31` as the bound and numVerts == 3, threads
# 0..31 run one iteration and threads 32..511 run none -- and the barrier in
# the loop body is then reached by 32 of 512 threads.  That is undefined.  It
# is also invisible: no output comparison can see it (the emitted set is
# identical -- the per-lane guard alone decides who writes), spirv-val cannot
# see it (the property is dynamic), and llvmpipe does not enforce it.
#
# Which is the whole reason for a static checker.  Nothing else in this
# repository's verification chain can find this class of bug.
#
# WHAT IT CHECKS
#
# Three ways a barrier can end up in non-uniform control flow, each a separate
# pass, because the first pass alone misses the other two entirely:
#
#   1. ENCLOSING SCOPE   -- the barrier sits inside an `if`/`for`/`while` whose
#                           condition is not block-uniform.
#   2. ESCAPE BEFORE     -- a `return`/`continue`/`break` under a non-uniform
#                           guard precedes the barrier, so some threads leave
#                           before reaching it.  `if (tid >= n) return;` is the
#                           single most common shape in this corpus and it is
#                           usually written WITHOUT braces, so a plain brace
#                           walk does not see it.
#   3. DIVERGENT CALL    -- the barrier is inside a called function and the CALL
#                           SITE is non-uniform.  Needs a corpus-wide call graph
#                           (a barrier five frames down still counts).
#
# SEVERITY DEPENDS ON THE BLOCK SIZE, and this is the point most easily missed:
#
#   * block == 32  -- the block IS one warp, so warp-uniform is block-uniform.
#     Only a LANE-varying condition is a violation here.
#   * block  > 32  -- warp-uniform is NOT enough.  A merely warp-uniform
#     condition is a violation.
#
# A condition is classified by taint: LANE if it depends on a thread/lane index,
# WARP if it depends on a warp index within the block.  Group ids (SV_GroupID,
# blockIdx) are uniform by definition and are never tainted.  Taint runs
# PER FUNCTION, not per file -- a file-wide pass reports `i` as thread-varying
# in every loop of the file merely because one other loop in the same file
# writes `for (uint i = tid; ...)`, which buries the real findings in noise.
#
# WAIVERS
#
# Most flags are proven-uniform values the checker cannot see through: a
# ballot mask (every lane holds the same bits), a warp reduction's result, a
# value loaded with a group-uniform index.  Each such site is listed in
# barrier_uniformity_waivers.txt with a one-line proof.  A waiver is a claim
# somebody checked, not a way to silence the tool -- and an unwaived finding
# fails the check, so a NEW divergent barrier cannot land unnoticed.
#
# USAGE
#
#   check_barrier_uniformity.py                  check; exit 1 on a violation
#   check_barrier_uniformity.py -v               also list the waived sites
#   check_barrier_uniformity.py --selftest-only  just the negative control
#   check_barrier_uniformity.py --chain f.slang:name
#                                                why `name` is judged divergent,
#                                                assignment by assignment
#
# Every run begins with the negative control: it puts the original P9.15a bug
# back into the corpus in memory and fails if it can no longer see it.  A green
# run therefore means the checker is awake, not merely quiet.  ALL EIGHT defects
# the first audit turned up were in the checker itself, not in the shaders, and
# three of those were SILENT -- they reported "clean" for the region they could
# not see rather than reporting nothing at all.  That is the failure mode a
# static checker has to be built against, and the reason the calibration is
# wired in rather than done once by hand.
#
# KNOWN LIMITS, stated so nobody over-trusts a green run:
#   * the taint is FLOW-INSENSITIVE -- a value made uniform by a later write
#     (the warp-max broadcast idiom) still reads as divergent, and needs a
#     waiver rather than being seen through;
#   * a warp collective's result is taken to be warp-uniform wholesale, with
#     shuffles excluded by name;
#   * a module's block size is known only through its call sites.
# ===========================================================================

import bisect
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHADERS = os.path.normpath(os.path.join(HERE, '..', 'shaders'))
WAIVERS = os.path.join(HERE, 'barrier_uniformity_waivers.txt')

BARRIER = 'GroupMemoryBarrierWithGroupSync'
CTRL = {'for', 'while', 'if', 'else', 'switch', 'do'}
LOOPS = {'for', 'while', 'do'}

# Anything derived from these varies BETWEEN LANES of one warp.  A barrier under
# such a condition is undefined at every block size, 32 included.
LANE_SEED = {
    'threadIdx', 'groupThreadId', 'gtid', 'localThreadIdx', 'threadIndexInWarp',
    'threadIdxInWarp', 'laneId', 'laneIndex', 'lane', 'idxInWarp', 'tid',
    'globalThreadIdx', 'dispatchThreadId', 'groupIndex',
    'threadIdx.x', 'groupThreadId.x', 'gtid.x', 'dispatchThreadId.x',
}
# Anything derived from these is uniform WITHIN a warp but varies between the
# warps of a block.  Harmless at block == 32, a violation above it.
WARP_SEED = {
    'warpId', 'warpIdx', 'warpIndex', 'warpIdxInBlock', 'wid', 'sub',
    'workIndex', 'globalWarpIndex', 'globalWarpIdx',
    # y/z of the group-thread id index the WARPS of a block, never its lanes:
    # the corpus spells a block as [numthreads(WARP, N_WARPS, 1)].
    'threadIdx.y', 'threadIdx.z', 'groupThreadId.y', 'groupThreadId.z',
    'gtid.y', 'gtid.z',
}

# Identifiers keep their swizzle: `gtid.x` is a LANE index while `gtid.y` is a
# WARP index, and collapsing both to `gtid` turns every warp index in the corpus
# into a false lane-divergence.
IDENT = re.compile(
    r'(?<![.\w])[A-Za-z_][A-Za-z_0-9]*(?:\s*\.\s*[xyzw]{1,4}(?![\w.]))?')


def norm_ids(expr):
    return {''.join(m.group(0).split()) for m in IDENT.finditer(expr)}


CALL = re.compile(r'\b([A-Za-z_][A-Za-z_0-9]*)\s*\(')
# A `for` split into its three clauses.  The trip count -- not the condition --
# is what decides whether the body's barriers are reached uniformly.
FOR_PARTS = re.compile(
    r'^for\s*\(\s*(?:const\s+)?(?:[A-Za-z_][A-Za-z_0-9<>,\s\*]*?\s+)?'
    r'([A-Za-z_][A-Za-z_0-9]*)\s*(?<![=!<>+\-*/&|^])=(?!=)\s*([^;]*);([^;]*);(.*)\)$')
# A warp-collective's result is warp-uniform BY CONSTRUCTION: in this port every
# one of them is groupshared + barriers, so it is already in the barrier-bearing
# set.  Shuffles are the exception -- a butterfly or a shuffle from a varying
# source lane returns something different in every lane.
NOT_UNIFORMISING = re.compile(r'(?i)(shfl|shuffle|exchange|scatter|gather)')
# The delimiter must NOT be consumed: `finditer` would then swallow the `;` that
# separates two assignments and silently skip every second one.
ASSIGN = re.compile(
    r'(?:(?<=[;{}()])|^)\s*(?:const\s+)?(?:static\s+)?'
    r'(?:[A-Za-z_][A-Za-z_0-9<>,\s\*]*?\s+)?'
    r'([A-Za-z_][A-Za-z_0-9]*)\s*(?<![=!<>+\-*/&|^])=(?!=)\s*([^;]*);')
FOR_INIT = re.compile(
    r'\bfor\s*\(\s*(?:const\s+)?(?:[A-Za-z_][A-Za-z_0-9<>,\s\*]*?\s+)?'
    r'([A-Za-z_][A-Za-z_0-9]*)\s*(?<![=!<>+\-*/&|^])=(?!=)\s*([^;]*);')
NUMTHREADS = re.compile(r'numthreads\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)')

# Builtins and intrinsics: never user functions, so never call-graph nodes.
BUILTIN = set('''
if for while switch return sizeof do else static_assert
uint int float bool half double void uint64_t int64_t uint16_t
float2 float3 float4 uint2 uint3 uint4 int2 int3 int4 bool2 bool3 bool4
float2x2 float3x3 float4x4 Ptr StructuredBuffer RWStructuredBuffer
min max abs dot cross normalize length sqrt rsqrt clamp lerp select mul
countbits firstbithigh firstbitlow reversebits asuint asfloat asint asdouble
saturate sign floor ceil round frac trunc fmod step smoothstep ldexp
exp exp2 log log2 pow sin cos tan sinh cosh tanh asin acos atan atan2
all any isnan isinf transpose determinant reflect refract distance
InterlockedAdd InterlockedOr InterlockedAnd InterlockedXor InterlockedMin
InterlockedMax InterlockedExchange InterlockedCompareExchange
GroupMemoryBarrierWithGroupSync GroupMemoryBarrier AllMemoryBarrier
AllMemoryBarrierWithGroupSync DeviceMemoryBarrier DeviceMemoryBarrierWithGroupSync
'''.split())


def strip_comments(src):
    """Blank out comments while preserving every byte offset and line break."""
    out = []
    i, n = 0, len(src)
    in_line = in_block = False
    while i < n:
        c = src[i]
        if in_line:
            if c == '\n':
                in_line = False
                out.append(c)
            else:
                out.append(' ')
            i += 1
        elif in_block:
            if src.startswith('*/', i):
                in_block = False
                out.append('  ')
                i += 2
            else:
                out.append('\n' if c == '\n' else ' ')
                i += 1
        elif src.startswith('//', i):
            in_line = True
            out.append('  ')
            i += 2
        elif src.startswith('/*', i):
            in_block = True
            out.append('  ')
            i += 2
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def header_before(src, brace):
    """The control-flow header owning the `{` at `brace`.

    Scanning back to the previous `;` is WRONG for `for (a; b; c)` -- it yields
    `c)` and the loop bound is lost.  Match the parenthesis instead.
    """
    j = brace - 1
    while j >= 0 and src[j] in ' \t\r\n':
        j -= 1
    if j >= 0 and src[j] == ')':
        depth, k = 0, j
        while k >= 0:
            if src[k] == ')':
                depth += 1
            elif src[k] == '(':
                depth -= 1
                if depth == 0:
                    break
            k -= 1
        m = k - 1
        while m >= 0 and src[m] in ' \t\r\n':
            m -= 1
        end = m + 1
        while m >= 0 and (src[m].isalnum() or src[m] == '_'):
            m -= 1
        return src[m + 1:end], ' '.join(src[m + 1:j + 1].split())
    end = j + 1
    s = j
    while s >= 0 and (src[s].isalnum() or src[s] == '_'):
        s -= 1
    return src[s + 1:end], src[s + 1:end]


def braceless_guard(src, pos):
    """The `if (...)` directly governing a braceless statement at `pos`."""
    j = pos - 1
    while j >= 0 and src[j] in ' \t\r\n':
        j -= 1
    if j < 0 or src[j] != ')':
        return None
    depth, k = 0, j
    while k >= 0:
        if src[k] == ')':
            depth += 1
        elif src[k] == '(':
            depth -= 1
            if depth == 0:
                break
        k -= 1
    m = k - 1
    while m >= 0 and src[m] in ' \t\r\n':
        m -= 1
    end = m + 1
    while m >= 0 and (src[m].isalnum() or src[m] == '_'):
        m -= 1
    kw = src[m + 1:end]
    if kw not in ('if', 'while', 'for'):
        return None
    return kw, ' '.join(src[m + 1:j + 1].split())


def line_index(src):
    lines = src.split('\n')
    offs = [0]
    for l in lines:
        offs.append(offs[-1] + len(l) + 1)

    def lineof(pos):
        lo, hi = 0, len(lines) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offs[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1
    return lineof


def taint(text, collectives):
    """Fixpoint taint over one function body.  Returns (lane_set, warp_set).

    `collectives` are the warp-collective function names: a value they return is
    the same in every lane of the warp, so it is demoted from LANE to WARP no
    matter how lane-varying its arguments were.  That is not a convenience --
    without it every ballot mask in the corpus reads as lane-varying and the
    real findings drown.
    """
    lane, warp = set(LANE_SEED), set(WARP_SEED)
    pairs = [(m.group(1), m.group(2)) for m in ASSIGN.finditer(text)]
    pairs += [(m.group(1), m.group(2)) for m in FOR_INIT.finditer(text)]
    uniformising = set()
    for var, rhs in pairs:
        for m in CALL.finditer(rhs):
            if m.group(1) in collectives and not NOT_UNIFORMISING.search(m.group(1)):
                uniformising.add(var)
                break
    for _ in range(12):
        changed = False
        for var, rhs in pairs:
            ids = norm_ids(rhs)
            if ids & lane and var not in uniformising:
                if var not in lane:
                    lane.add(var)
                    changed = True
            elif (ids & lane or ids & warp) and var not in warp and var not in lane:
                warp.add(var)
                changed = True
        if not changed:
            break
    return lane, warp


def classify(header, lane, warp, collectives=frozenset()):
    ids = norm_ids(strip_collectives(header, collectives))
    if ids & lane:
        return 'LANE', sorted(ids & lane)
    if ids & warp:
        return 'WARP', sorted(ids & warp)
    return None


CONSTDEF = re.compile(
    r'(?:^|\n)\s*(?:static\s+)?(?:const\s+)?u?int\s+([A-Za-z_][A-Za-z_0-9]*)'
    r'\s*=\s*(\d+)\s*u?\s*;')


_CONST_TABLES = {}


def _const_table(text):
    """name -> value for every `static const uint X = <literal>;` in `text`,
    plus one level of `A * B` products (how every block size here is spelled).

    Built ONCE per text.  Re-scanning the file for each lookup made this the
    single most expensive thing the checker did, by an order of magnitude.
    """
    key = hash(text)
    table = _CONST_TABLES.get(key)
    if table is not None:
        return table
    table = {}
    for m in CONSTDEF.finditer(text):
        table[m.group(1)] = int(m.group(2))
    for m in re.finditer(
            r'(?:^|\n)\s*(?:static\s+)?(?:const\s+)?u?int\s+'
            r'([A-Za-z_][A-Za-z_0-9]*)\s*=\s*([A-Za-z_0-9]+)\s*\*\s*'
            r'([A-Za-z_0-9]+)\s*;', text):
        a, b = m.group(2).rstrip('u'), m.group(3).rstrip('u')
        av = int(a) if a.isdigit() else table.get(a)
        bv = int(b) if b.isdigit() else table.get(b)
        if av is not None and bv is not None:
            table.setdefault(m.group(1), av * bv)
    _CONST_TABLES[key] = table
    return table


def const_value(name, *texts):
    """Numeric value of a compile-time constant like `SPHERE_BLOCK`.

    Without this the checker cannot tell that `i += SPHERE_BLOCK` and
    `(n + 31) & ~31` are the same rounding -- one is spelled as an identifier
    and the other as a literal, and the fix pattern uses both spellings.
    """
    if name.isdigit():
        return int(name)
    for t in texts:
        v = _const_table(t).get(name)
        if v is not None:
            return v
    return None


def strip_collectives(expr, collectives):
    """Blank out `collective(...)` call expressions.

    `if (pcReduceOr(tI, out) == 0u)` mentions a lane index, but only as an
    ARGUMENT to a warp reduction -- the value the branch tests is the same in
    every lane.  Judging the raw text would call this lane-varying.
    """
    out, i, n = [], 0, len(expr)
    while i < n:
        m = CALL.match(expr, i)
        prev = expr[i - 1] if i else ' '
        if m and not (prev.isalnum() or prev == '_' or prev == '.') \
                and m.group(1) in collectives \
                and not NOT_UNIFORMISING.search(m.group(1)):
            j = expr.index('(', i)
            depth = 0
            while j < n:
                if expr[j] == '(':
                    depth += 1
                elif expr[j] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            out.append(' 0 ')
            i = j + 1
        else:
            out.append(expr[i])
            i += 1
    return ''.join(out)


def rounded_to(bound_var, stride, text, whole):
    """True if `bound_var` is defined as a value rounded UP to a multiple of
    `stride` -- `(n + (S - 1)) & ~(S - 1)`, or the `(n + 31) & ~31` spelling
    when the stride literally is 32.

    This is the P9.15a fix pattern, and recognising it is the whole point: a
    grid-stride loop starting at the thread index has a uniform trip count
    exactly when its bound is a multiple of the stride.
    """
    want = const_value(stride.strip(), text, whole)
    for m in re.finditer(
            r'\b%s\s*(?<![=!<>+\-*/&|^])=(?!=)\s*([^;]*);' % re.escape(bound_var), text):
        rhs = ' '.join(m.group(1).split())
        if '&' not in rhs or '+' not in rhs:
            continue
        toks = {t.split('.')[0] for t in norm_ids(rhs)}
        if stride.strip() in toks:
            return True
        if want is None:
            continue
        # `(n + (S - 1)) & ~(S - 1)` with S spelled as a literal, or as another
        # constant that resolves to the same number.
        nums = {int(x) for x in re.findall(r'\d+', rhs)}
        nums |= {v for v in (const_value(t, text, whole) for t in toks)
                 if v is not None}
        if want in nums or (want - 1) in nums:
            return True
    return False


def classify_for(header, lane, warp, text, collectives=frozenset(), whole=''):
    """Classify a `for` by its TRIP COUNT, not by its condition.

    A grid-stride loop's condition mentions the loop variable and therefore
    always looks thread-varying; what matters is whether every thread runs the
    same number of iterations.  It does when init, bound and stride are all
    uniform -- or when the bound is rounded up to the stride, which makes the
    thread-indexed start irrelevant.
    """
    m = FOR_PARTS.match(header)
    if not m:
        return classify(header, lane, warp)
    var, init, cond, step = m.group(1), m.group(2), m.group(3), m.group(4)
    # Ignore the loop variable itself everywhere: it is re-bound by this header,
    # so a taint it picked up from an earlier loop in the same function is stale.
    def cls(expr):
        ids = norm_ids(strip_collectives(expr, collectives)) - {var}
        if ids & lane:
            return 'LANE', sorted(ids & lane)
        if ids & warp:
            return 'WARP', sorted(ids & warp)
        return None
    bound = re.sub(r'^[^<>=!]*[<>=!]+', '', cond).strip()
    stride = re.sub(r'^.*?(?:\+=|-=|=)', '', step).strip() or step.strip()
    worst = None
    for expr in (init, bound, stride):
        c = cls(expr)
        if c and c[0] == 'LANE':
            worst = c
            break
        if c and c[0] == 'WARP':
            worst = c
    if worst is None:
        return None
    # init thread-indexed + bound rounded up to the stride == uniform trip count
    if cls(init) and not cls(bound) and not cls(stride):
        if rounded_to(bound.split()[-1] if bound else '', stride, text, whole) or \
           rounded_to(bound, stride, text, whole):
            return None
    return worst


class Shader(object):
    def __init__(self, path, text=None):
        raw = open(path).read() if text is None else text
        self.name = os.path.basename(path)
        self.src = strip_comments(raw)
        self.lineof = line_index(self.src)
        self.funcs = {}          # name -> (start_brace, end_brace)
        self._sorted = None
        self._starts = None
        self.entries = []        # (line, block_size, name)
        self._scan_functions()
        self._scan_entries(raw)

    def _scan_functions(self):
        src = self.src
        i, n = 0, len(src)
        while i < n:
            if src[i] != '{':
                i += 1
                continue
            _, header = header_before(src, i)
            m = re.match(r'^([A-Za-z_][A-Za-z_0-9]*)\s*\(', header)
            depth, j = 0, i
            while j < n:
                if src[j] == '{':
                    depth += 1
                elif src[j] == '}':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if m and m.group(1) not in CTRL:
                self.funcs.setdefault(m.group(1), []).append((i, j))
            i = j + 1

    def _scan_entries(self, raw):
        lines = raw.split('\n')
        for i, l in enumerate(lines):
            m = NUMTHREADS.search(l)
            if not m:
                continue
            block = int(m.group(1)) * int(m.group(2)) * int(m.group(3))
            name = None
            for j in range(i, min(i + 5, len(lines))):
                mm = re.search(r'\bvoid\s+([A-Za-z_][A-Za-z_0-9]*)', lines[j])
                if mm:
                    name = mm.group(1)
                    break
            self.entries.append((i + 1, block, name))

    def block_at(self, line):
        """Block size of the entry point a line belongs to; None inside a module."""
        best = None
        for (el, block, _) in self.entries:
            if el <= line:
                best = block
        return best

    def func_at(self, pos):
        """(name, start, end) of the innermost function containing pos.

        Bisect, not a linear sweep: this is called once per barrier, per escape
        and per call site, and a linear scan over every function in the file
        made the whole check ten times slower than the analysis it serves.
        """
        if self._sorted is None:
            self._sorted = sorted(
                ((a, b, name) for name, occ in self.funcs.items() for (a, b) in occ))
            self._starts = [r[0] for r in self._sorted]
        k = bisect.bisect_right(self._starts, pos) - 1
        while k >= 0:
            a, b, name = self._sorted[k]
            if pos <= b:
                return (name, a, b)
            k -= 1
        return None


def walk(shader):
    """Events (kind, pos, line, brace-stack, braceless-guard) in source order."""
    src = shader.src
    events, stack = [], []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '{':
            kw, h = header_before(src, i)
            stack.append((kw, h, shader.lineof(i), i))
            i += 1
        elif c == '}':
            if stack:
                stack.pop()
            i += 1
        elif src.startswith(BARRIER, i):
            events.append(('BARRIER', i, shader.lineof(i), list(stack), None))
            i += len(BARRIER)
        else:
            m = re.match(r'(return|continue|break)\b', src[i:i + 9])
            if m and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == '_')):
                events.append((m.group(1).upper(), i, shader.lineof(i),
                               list(stack), braceless_guard(src, i)))
                i += len(m.group(1))
            else:
                i += 1
    return events


_CALLGRAPH_CACHE = {}


def barrier_bearing_functions(shaders):
    key = id(shaders)
    if key in _CALLGRAPH_CACHE:
        return _CALLGRAPH_CACHE[key]
    out = _barrier_bearing_functions(shaders)
    _CALLGRAPH_CACHE[key] = out
    return out


def _barrier_bearing_functions(shaders):
    """Function names that reach a barrier directly or through any call."""
    reaches = set()
    bodies = {}
    for sh in shaders.values():
        for name, occ in sh.funcs.items():
            for (a, b) in occ:
                bodies.setdefault(name, []).append(sh.src[a:b])
    for name, texts in bodies.items():
        if any(BARRIER in t for t in texts):
            reaches.add(name)
    for _ in range(16):
        changed = False
        for name, texts in bodies.items():
            if name in reaches:
                continue
            called = set()
            for t in texts:
                called |= {m.group(1) for m in CALL.finditer(t)}
            if (called - BUILTIN) & reaches:
                reaches.add(name)
                changed = True
        if not changed:
            break
    return reaches


def analyse(shaders):
    findings = []
    reaches = barrier_bearing_functions(shaders)

    for sh in shaders.values():
        if BARRIER not in sh.src:
            # pass 3 still applies: this file may CALL a barrier-bearing function.
            if not any(n in reaches for n in
                       {m.group(1) for m in CALL.finditer(sh.src)}):
                continue

        events = walk(sh)
        # per-function taint, memoised
        taints = {}

        def taint_at(pos):
            f = sh.func_at(pos)
            if f is None:
                return set(LANE_SEED), set(WARP_SEED), sh.src
            if f[1] not in taints:
                taints[f[1]] = taint(sh.src[f[1]:f[2]], reaches) + (sh.src[f[1]:f[2]],)
            return taints[f[1]]

        def verdict(kw, header, pos):
            lane, warp, text = taint_at(pos)
            if kw == 'for':
                return classify_for(header, lane, warp, text, reaches, sh.src)
            return classify(header, lane, warp, reaches)

        # ---- pass 1: enclosing scope -------------------------------------
        for (kind, pos, line, stack, _) in events:
            if kind != 'BARRIER':
                continue
            for (kw, h, hl, _hp) in [c for c in stack if c[0] in CTRL]:
                cls = verdict(kw, h, pos)
                if cls:
                    cls, why = cls
                    findings.append(dict(why=why,
                        pass_='scope', file=sh.name, block=sh.block_at(line),
                        line=line, cls=cls, kw=kw, guard=h, guard_line=hl))
                    break

        # ---- pass 2: escape before a barrier -----------------------------
        for (kind, pos, line, stack, _) in events:
            if kind != 'BARRIER' or not stack:
                continue
            fn_brace = stack[0][3]
            for (k2, pos2, line2, stack2, bg) in events:
                if k2 == 'BARRIER' or pos2 >= pos:
                    continue
                if not stack2 or stack2[0][3] != fn_brace:
                    continue
                braced = [c for c in stack2 if c[0] in CTRL]
                if bg:
                    guard = (bg[0], bg[1], line2)
                elif braced:
                    guard = (braced[-1][0], braced[-1][1], braced[-1][2])
                else:
                    continue
                if k2 in ('CONTINUE', 'BREAK'):
                    loops = [c for c in stack2 if c[0] in LOOPS]
                    if not loops:
                        continue
                    # the escaped loop must ENCLOSE the barrier, or the escape
                    # cannot split the threads that reach it
                    if not any(c[3] == loops[-1][3] for c in stack):
                        continue
                elif not bg and braced and any(c[3] == braced[-1][3] for c in stack):
                    continue    # barrier lives under the same guard: no split
                cls = verdict(guard[0], guard[1], pos)
                if cls:
                    cls, why = cls
                    findings.append(dict(why=why,
                        pass_='escape', file=sh.name, block=sh.block_at(line),
                        line=line, cls=cls, kw='%s/%s' % (k2.lower(), guard[0]),
                        guard=guard[1], guard_line=guard[2]))

        # ---- pass 3: divergent call site ---------------------------------
        stack = []
        i, n = 0, len(sh.src)
        while i < n:
            c = sh.src[i]
            if c == '{':
                kw, h = header_before(sh.src, i)
                stack.append((kw, h, sh.lineof(i), i))
                i += 1
            elif c == '}':
                if stack:
                    stack.pop()
                i += 1
            else:
                m = CALL.match(sh.src, i)
                prev = sh.src[i - 1] if i else ' '
                if m and not (prev.isalnum() or prev == '_' or prev == '.'):
                    name = m.group(1)
                    if name in reaches and name not in BUILTIN:
                        conds = [c2 for c2 in stack if c2[0] in CTRL]
                        if conds:
                            for (kw, h, hl, _hp) in conds:
                                cls = verdict(kw, h, i)
                                if cls:
                                    cls, why = cls
                                    findings.append(dict(why=why,
                                        pass_='call', file=sh.name,
                                        block=sh.block_at(sh.lineof(i)),
                                        line=sh.lineof(i), cls=cls,
                                        kw='call ' + name, guard=h, guard_line=hl))
                                    break
                    i += len(name)
                else:
                    i += 1
    return findings


def is_violation(f):
    """LANE breaks the rule at any block size; WARP only above one warp.

    A module (block None) is judged as LANE-only here: its callers' block sizes
    are checked by pass 3 at the call site, which is where the divergence is.
    """
    if f['cls'] == 'LANE':
        return True
    return f['block'] is not None and f['block'] > 32


def load_waivers():
    waived = {}
    if not os.path.exists(WAIVERS):
        return waived
    for raw in open(WAIVERS):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split('|')
        if len(parts) < 4:
            continue
        key = (parts[0].strip(), parts[1].strip(), ' '.join(parts[2].split()))
        waived[key] = parts[3].strip()
    return waived


def key_of(f):
    return (f['file'], f['pass_'], ' '.join(f['guard'].split()))


def load_corpus(mutate=None):
    shaders = {}
    for name in sorted(os.listdir(SHADERS)):
        if not name.endswith('.slang'):
            continue
        path = os.path.join(SHADERS, name)
        text = None
        if mutate:
            text = mutate(name, open(path).read())
        shaders[name] = Shader(path, text)
    return shaders


# The mutation is the ACTUAL P9.15a bug, spelled the way the CUDA spells it: a
# loop bound rounded to the warp inside a 512-thread block.
SELFTEST_FILE = 'femClothContactGenBda.slang'
SELFTEST_FROM = '(numVerts + (BOX_BLOCK - 1u)) & (~(BOX_BLOCK - 1u))'
SELFTEST_TO = '(numVerts + 31u) & (~31u)'


def selftest():
    """Put the P9.15a bug back and demand the checker sees it.

    A checker that is silent on the fixed corpus proves nothing until it is
    also shown to be loud on the broken one -- otherwise a later refactor can
    quietly blind it and every run still says OK.
    """
    def mutate(name, text):
        if name != SELFTEST_FILE:
            return text
        if SELFTEST_FROM not in text:
            print('SELFTEST BROKEN: the anchor is gone from %s.' % SELFTEST_FILE)
            print('  Re-anchor it on whatever now spells the block rounding in')
            print('  cloth_boxVertexContactGenLaunch; do NOT delete the selftest.')
            sys.exit(2)
        return text.replace(SELFTEST_FROM, SELFTEST_TO)

    shaders = load_corpus(mutate)
    waived = load_waivers()
    hits = [f for f in analyse(shaders)
            if is_violation(f) and key_of(f) not in waived
            and f['file'] == SELFTEST_FILE and f['block'] > 32]
    if not hits:
        print('SELFTEST FAILED: warp-rounding a bound in a %s-thread block was'
              % 512)
        print('  not reported.  The checker is blind to the bug it exists for.')
        return 2
    print('selftest: warp-rounded bound in a 512-thread block -> %d site(s) '
          'reported.  The checker can still see it.' % len(hits))
    return 0


def chain(spec):
    """`--chain file.slang:name` -- why the checker thinks `name` diverges.

    A verdict you cannot trace is a verdict you can only wave away.  This walks
    the assignments backwards from the reported identifier and prints, for each
    step, which of its inputs carried the taint.
    """
    fname, _, target = spec.partition(':')
    shaders = load_corpus()
    if fname not in shaders:
        print('no such shader: %s' % fname)
        return 2
    sh = shaders[fname]
    reaches = barrier_bearing_functions(shaders)
    for func, occ in sorted(sh.funcs.items()):
        for (a, b) in occ:
            text = sh.src[a:b]
            if not re.search(r'\b%s\b' % re.escape(target), text):
                continue
            lane, warp = taint(text, reaches)
            if target not in lane and target not in warp:
                continue
            print('--- %s :: %s   %s is %s' %
                  (fname, func, target, 'LANE' if target in lane else 'WARP'))
            pairs = [(m.group(1), ' '.join(m.group(2).split()))
                     for m in ASSIGN.finditer(text)]
            pairs += [(m.group(1), ' '.join(m.group(2).split()))
                      for m in FOR_INIT.finditer(text)]
            seen, stack = set(), [target]
            while stack:
                t = stack.pop()
                if t in seen:
                    continue
                seen.add(t)
                for (v, rhs) in pairs:
                    if v != t:
                        continue
                    hits = sorted((norm_ids(rhs) & lane) | (norm_ids(rhs) & warp))
                    print('    %-26s = %s' % (v, rhs[:100]))
                    if hits:
                        print('    %-26s   ^ tainted by: %s' % ('', ', '.join(hits)))
                    stack.extend(x.split('.')[0] for x in hits)
            print()
    return 0


def main():
    verbose = '-v' in sys.argv or '--verbose' in sys.argv
    for i, a in enumerate(sys.argv):
        if a == '--chain' and i + 1 < len(sys.argv):
            return chain(sys.argv[i + 1])
    if '--selftest-only' in sys.argv:
        return selftest()

    rc = selftest()
    if rc:
        return rc
    shaders = load_corpus()

    findings = analyse(shaders)
    waived = load_waivers()

    violations, excused, benign = [], [], []
    for f in findings:
        if not is_violation(f):
            benign.append(f)
        elif key_of(f) in waived:
            f['waiver'] = waived[key_of(f)]
            excused.append(f)
        else:
            violations.append(f)

    def show(f):
        # Name the identifier that forced the verdict.  A finding whose cause is
        # not visible cannot be judged, only guessed at.
        return ('%-34s %-7s blk=%-4s L%-5d [%s] %-20s %-62s <- %s'
                % (f['file'], f['pass_'], f['block'], f['line'], f['cls'],
                   f['kw'], f['guard'][:62], ','.join(f.get('why', []))))

    reaches = barrier_bearing_functions(shaders)
    entries = [(sh, b, n) for sh in shaders.values() for (_, b, n) in sh.entries]
    at_risk = [e for e in entries if e[1] > 32 and e[2] in reaches]
    print('barrier-uniformity check over %d shaders' % len(shaders))
    print('  entry points: %d   with block > 32: %d   of those reaching a barrier: %d'
          % (len(entries), sum(1 for e in entries if e[1] > 32), len(at_risk)))
    print('  (only in those %d can a merely warp-uniform condition break the rule)'
          % len(at_risk))
    print('  sites examined: %d   waived: %d   below the severity bar: %d'
          % (len(findings), len(excused), len(benign)))

    if verbose:
        for f in excused:
            print('  WAIVED  ' + show(f) + '\n            -> ' + f['waiver'])

    if violations:
        print('\n%d NON-UNIFORM BARRIER SITE(S):\n' % len(violations))
        for f in violations:
            print('  ' + show(f))
        print('\nEach is either a real bug -- round the loop bound to the BLOCK,')
        print('not the warp -- or a proven-uniform value the checker cannot see')
        print('through, in which case add it to')
        print('  %s' % os.path.relpath(WAIVERS, os.getcwd()))
        print('with the one-line proof.  A waiver is a claim somebody checked.')
        return 1

    print('\nOK -- every barrier is reached under block-uniform control flow.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
