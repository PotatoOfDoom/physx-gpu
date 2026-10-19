#!/usr/bin/env python3
# ===========================================================================
# check_cuda_twin_kernels.py -- keep "these two engine kernels are the same
# kernel" true (P9.20).
#
# WHY THIS EXISTS
#
# Some kernels in the vendored CUDA are written twice.  P9.20 found the first
# pair this port actually depends on: cloth_boxTriangleContactGenLaunch and
# cloth_convexContactGenLaunch are the same kernel, and their cores
# clothBoxTriangleCollision / clothConvexCollision differ only in a `const`
# qualifier, a redundant re-declaration of one pointer, and whether PxLoad3
# writes through named temporaries.  Their argument packs are identical, and
# both launch sites pass the same filter buffer despite naming the parameter
# differently.
#
# So the Slang port writes that body ONCE and gives it two entry points.  That
# is only correct while the CUDA bodies stay equivalent -- and nothing else in
# the build would notice if NVIDIA changed one of them.  A signature change
# would surface (the ABI rows would stop matching); a BODY change would not.
#
# This checker is that missing signal.  It re-derives the equivalence from the
# vendored CUDA on every run and fails when it stops holding, naming the pair
# and printing the surviving difference.
#
# WHAT "EQUIVALENT" MEANS HERE, spelled out rather than left to a ratio:
#   * comments removed;
#   * whitespace normalised;
#   * the renames listed per pair applied (a rename is a claim that two names
#     denote the same thing, so each one is written down, not inferred);
#   * a line that only RE-DECLARES a name already bound to the same expression
#     earlier in the body is dropped -- the box copy declares `triVertInds`
#     twice from `cloth.mTriangleVertexIndices`;
#   * `T x = PxLoad3(e); ... const PxVec3 v = x;` is folded to
#     `const PxVec3 v = PxLoad3(e);`;
#   * (PhysX 5.12) a declaration whose right-hand side is a plain FIELD READ
#     (`T* x = obj.mField;`, no call, no index, no arithmetic) is sunk to just
#     before the first line that uses `x`.  5.12's clothConvexCollision binds
#     `triVertInds = cloth.mTriangleVertexIndices_ownedEdges` right after
#     `cloth`, the box twin forty lines later; reading a pointer field of a
#     const reference has no effect to order, so where it is spelled is not part
#     of the kernel.
#
# Every one of those rules exists because a real, checked-by-hand difference
# needed it.  A rule that is not needed by some pair does not belong here: the
# looser the normalisation, the less the check means.
#
# USAGE
#   check_cuda_twin_kernels.py          check; exit 1 when a pair diverges
#   check_cuda_twin_kernels.py -v       also print the normalised bodies' sizes
# ===========================================================================

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..'))

# (file, function A, function B, {rename in A -> canonical}, why the pair matters)
TWINS = [
    (
        'physx/source/gpunarrowphase/src/CUDA/femClothPrimitives.cu',
        'clothBoxTriangleCollision', 'clothConvexCollision',
        {'vertIndices': 'triVertInds'},
        'P9.20 gives femClothContactGenBda.slang ONE body '
        '(clothConvexTriCgBody) two entry points, cloth_boxTriangleContactGenLaunch '
        'and cloth_convexContactGenLaunch, on the strength of this equivalence.',
    ),
    (
        'physx/source/gpunarrowphase/src/CUDA/femClothPrimitives.cu',
        'cloth_boxTriangleContactGenLaunch', 'cloth_convexContactGenLaunch',
        {'clothAttachmentPairs': 'clothRigidPairs',
         'nbClothAttachmentPairs': 'numClothRigidPairs',
         'clothBoxTriangleCollision': 'clothConvexCollision'},
        'The two wrappers around the pair above; same 15-argument pack, and both '
        'launch sites pass simulationCore->getRigidClothFilters().',
    ),
]


def strip_comments(src):
    out = []
    i, n = 0, len(src)
    in_line = in_block = False
    while i < n:
        c = src[i]
        if in_line:
            if c == '\n':
                in_line = False
                out.append(c)
            i += 1
        elif in_block:
            if src.startswith('*/', i):
                in_block = False
                i += 2
            else:
                if c == '\n':
                    out.append(c)
                i += 1
        elif src.startswith('//', i):
            in_line = True
            i += 2
        elif src.startswith('/*', i):
            in_block = True
            i += 2
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def function_body(src, name):
    """The brace-matched body of `name`, as normalised lines."""
    m = re.search(r'\b' + re.escape(name) + r'\s*\(', src)
    if not m:
        return None
    open_brace = src.index('{', src.index(')', m.end() - 1))
    depth, k = 0, open_brace
    while k < len(src):
        if src[k] == '{':
            depth += 1
        elif src[k] == '}':
            depth -= 1
            if depth == 0:
                break
        k += 1
    body = strip_comments(src[open_brace + 1:k])
    return [' '.join(l.split()) for l in body.split('\n') if l.strip()]


DECL = re.compile(r'^(?:const\s+)?[A-Za-z_][\w:<>\*&\s]*?\**\s*'
                  r'([A-Za-z_]\w*)\s*=\s*(.+);$')
LOAD3 = re.compile(r'^(?:const\s+)?PxVec3\s+([A-Za-z_]\w*)\s*=\s*PxLoad3\((.+)\);$')
ALIAS = re.compile(r'^(?:const\s+)?PxVec3\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*);$')


FIELD_DECL = re.compile(r'^(?:const\s+)?[A-Za-z_][\w:<>\s]*?\**\s*'
                        r'([A-Za-z_]\w*)\s*=\s*[A-Za-z_]\w*(?:\.|->)[A-Za-z_]\w*;$')


def sink_field_decls(lines):
    """Move each pure field-read declaration to just before its first use."""
    out = list(lines)
    i = 0
    while i < len(out):
        m = FIELD_DECL.match(out[i])
        if not m:
            i += 1
            continue
        name = m.group(1)
        use = re.compile(r'\b' + re.escape(name) + r'\b')
        j = next((k for k in range(i + 1, len(out)) if use.search(out[k])), None)
        if j is None or j == i + 1:
            i += 1
            continue
        decl = out.pop(i)
        out.insert(j - 1, decl)
        # the line now at i is new; examine it next
    return out


def canonicalise(lines, renames):
    """Apply the documented normalisation rules, in order."""
    out = []
    for l in lines:
        for a, b in renames.items():
            l = re.sub(r'\b' + re.escape(a) + r'\b', b, l)
        l = re.sub(r'\bconst\s+', '', l)
        out.append(l)

    # drop a re-declaration that binds a name already bound to the same rhs
    seen = {}
    kept = []
    for l in out:
        m = DECL.match(l)
        if m and seen.get(m.group(1)) == m.group(2):
            continue
        if m:
            seen[m.group(1)] = m.group(2)
        kept.append(l)

    # fold `PxVec3 t = PxLoad3(e);` + `PxVec3 v = t;` into `PxVec3 v = PxLoad3(e);`
    loads = {}
    folded = []
    for l in kept:
        m = LOAD3.match(l)
        if m:
            loads[m.group(1)] = m.group(2)
            folded.append(('LOAD', m.group(1), m.group(2), l))
            continue
        a = ALIAS.match(l)
        if a and a.group(2) in loads:
            folded.append(('FOLD', a.group(1), loads[a.group(2)], l))
            continue
        folded.append(('PLAIN', None, None, l))
    used = {f[2] for f in folded if f[0] == 'FOLD'}
    final = []
    for kind, name, expr, l in folded:
        if kind == 'LOAD' and expr in used:
            continue                       # the temporary is folded away
        if kind == 'FOLD':
            final.append('PxVec3 %s = PxLoad3(%s);' % (name, expr))
            continue
        final.append(l)
    return sink_field_decls(final)


def selftest():
    """Break one twin and demand the check notices.

    The normalisation rules here are deliberately generous -- they fold away a
    rename, a redundant declaration and a temporary.  Generous normalisation is
    exactly how a checker stops discriminating, so the calibration is wired in
    rather than done once by hand: a single changed OPERAND in one body must be
    enough to make the pair diverge.
    """
    rel, a_name, b_name, renames, _why = TWINS[0]
    src = open(os.path.join(ROOT, rel)).read()
    a_raw = function_body(src, a_name)
    if not a_raw:
        print('SELFTEST BROKEN: %s is gone from %s.' % (a_name, rel))
        return 2
    # A one-token change that no rule above can absorb.
    mutated = [l.replace('triIndices.z', 'triIndices.y') for l in a_raw]
    if mutated == a_raw:
        print('SELFTEST BROKEN: the anchor `triIndices.z` is no longer in %s.'
              % a_name)
        print('  Re-anchor it on some other operand of that body; do NOT delete '
              'the selftest.')
        return 2
    if canonicalise(mutated, renames) == canonicalise(function_body(src, b_name), {}):
        print('SELFTEST FAILED: a changed operand did not make the pair diverge -- '
              'the normalisation has become too generous to discriminate.')
        return 2
    print('selftest: one changed operand makes the pair diverge.  The check can '
          'still tell them apart.')
    return 0


def main():
    verbose = '-v' in sys.argv or '--verbose' in sys.argv
    if '--selftest-only' in sys.argv:
        return selftest()
    rc = selftest()
    if rc:
        return rc
    failures = 0
    print('CUDA twin-kernel equivalence check over %d pair(s)' % len(TWINS))
    for (rel, a_name, b_name, renames, why) in TWINS:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print('  MISSING  %s -- cannot check %s / %s' % (rel, a_name, b_name))
            failures += 1
            continue
        src = open(path).read()
        a_raw, b_raw = function_body(src, a_name), function_body(src, b_name)
        if a_raw is None or b_raw is None:
            print('  GONE     %s / %s -- one of the pair is no longer in %s.'
                  % (a_name, b_name, rel))
            print('           %s' % why)
            print('           Re-anchor this entry or remove it; do NOT delete the '
                  'check while the shared body still exists.')
            failures += 1
            continue
        a = canonicalise(a_raw, renames)
        b = canonicalise(b_raw, {})
        if a == b:
            if verbose:
                print('  OK       %-36s == %-32s (%d lines, %d raw/%d raw)'
                      % (a_name, b_name, len(a), len(a_raw), len(b_raw)))
            else:
                print('  OK       %s == %s' % (a_name, b_name))
            continue
        failures += 1
        print('  DIVERGED %s != %s' % (a_name, b_name))
        print('           %s' % why)
        import difflib
        shown = 0
        for line in difflib.unified_diff(a, b, lineterm='', n=1,
                                         fromfile=a_name, tofile=b_name):
            print('           ' + line[:110])
            shown += 1
            if shown > 24:
                print('           ...')
                break
    if failures:
        print('\n%d pair(s) no longer equivalent.  The Slang side shares ONE body '
              'between them, so this is not cosmetic: split the shared body, give '
              'each engine name its own, and port the difference into whichever '
              'one it belongs to.' % failures)
        return 1
    print('\nOK -- every declared twin is still one kernel written twice.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
