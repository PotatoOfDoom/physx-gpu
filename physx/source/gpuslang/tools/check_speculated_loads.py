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
#               at most one stride.
#
#   LOW-launch  the index is assigned IN THIS FUNCTION from a thread-id system
#               value.  The launch is sized to the data rounded up to a
#               workgroup, so the overshoot is less than one workgroup.
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
#   * the INLINE form `Ptr<T>(addr)[i]` -- only named Ptr<> values are tracked.
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
    """Is `var` assigned, in this body, from something rooted in a thread id?"""
    if depth > 6:
        return False
    for m in re.finditer(r'(?:(?<=[;{}()])|^)\s*(?:const\s+)?(?:[A-Za-z_][\w<>:]*\s+)?' +
                         re.escape(var) + r'\s*(?<![=!<>+\-*/&|^])=(?!=)\s*([^;]{1,200});', body):
        rhs = m.group(1)
        if SV_SEED.search(rhs):
            return True
        for r in index_roots(rhs):
            if r != var and launch_rooted(body, r, depth + 1):
                return True
    # `for (var = tid; ...)`
    for m in re.finditer(r'\bfor\s*\([^;]*\b' + re.escape(var) + r'\s*=\s*([^;]{1,120});', body):
        if SV_SEED.search(m.group(1)):
            return True
    return False


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


def analyse_shader(sh):
    src = sh.src
    findings = []
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
                        tier, why = 'LOW-loop', ("the index is this loop's own "
                                                 "induction variable: overshoot <= one stride")
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
                    if tier == 'HIGH':
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


def selftest():
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
