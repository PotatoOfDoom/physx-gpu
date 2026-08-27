#!/usr/bin/env python3
# ===========================================================================
# check_disabled_kernels.py -- keep "this engine kernel emits nothing, and that
# is upstream's doing" true (P9.26).
#
# WHY THIS EXISTS
#
# cloth_meshVertexContactGenLaunch (femClothPrimitives.cu:2910) computes a
# closest point, a normal, a penetration and a barycentric for every (cloth
# vertex, mesh triangle) pair its midphase produced -- and then throws all of it
# away.  The write is gated on `hasContacts`, and the ONLY assignment to
# `hasContacts` in the whole kernel is commented out:
#
#     bool hasContacts = false;                        // :2940
#     ...
#             contact = make_float4(...);
#             //hasContacts = true;                    // :3056  <-- COMMENTED
#         }
#     }
#     PxgOutputIndex indexer(__ballot_sync(FULL_MASK, hasContacts),
#                            writer.totalContactCount);
#     if (hasContacts) { ... writeRigidVsDeformableContact32(...) }
#
# PxgOutputIndex (:79) only performs its atomicAdd when the popcount is
# non-zero, so with an always-empty ballot the contact COUNTER does not move
# either.  The kernel's entire observable output is one atomicMax.
#
# The Slang port (femClothMeshVertContactGenBda.slang) therefore implements the
# observable behaviour -- an empty walk plus that atomicMax -- and deliberately
# does NOT transcribe the two hundred lines of closest-point geometry, because
# no fixture in this port could tell a correct transcription from a wrong one:
# the only observable is "no contacts", which every implementation of dead code
# satisfies equally.
#
# That decision is only correct while the line stays commented.  Nothing else in
# this build would notice if it stopped being: the SIGNATURE is checked (the ABI
# row), the launch site is checked (it still runs every step), the fixture is
# checked -- and all three stay green while the kernel starts emitting contacts
# the port does not produce.  This checker is that missing signal.  It re-derives
# the disabled state from the vendored CUDA on every run and fails the moment it
# changes, which is the same device P9.20 used for the twin kernels: a fact about
# someone else's source that this port depends on gets a check, not a comment.
#
# WHAT IS ACTUALLY CHECKED, spelled out rather than left to a grep:
#   * the kernel is still there;
#   * its flag is still declared `= false`;
#   * NO LIVE assignment to the flag exists anywhere in the body -- not just no
#     `= true`, any assignment;
#   * at least one COMMENTED-OUT assignment to the flag exists -- so the check
#     fails just as loudly if upstream DELETES the dead line as if it enables
#     it, because a deleted line means the kernel was rewritten and the port's
#     reading of it is stale either way;
#   * the write is still gated on the flag (`if (flag)`), and the ballot still
#     reads it -- if either stops being true the "emits nothing" conclusion no
#     longer follows from the flag alone;
#   * PxgOutputIndex's atomicAdd is still guarded by a non-zero popcount, which
#     is what makes the CONTACT COUNTER stay put and not merely the contacts.
#
# THE LIVE CONTROLS
#
# A checker that reports "disabled" must be able to report "live", or its green
# run means nothing.  So the same analysis is run over kernels in the SAME FILE
# that are live -- cloth_meshContactGenLaunch, the triangle sibling of the very
# kernel above -- and a run fails if any of them is read as disabled.  That is a
# control drawn from real source rather than a constructed one.
#
# The selftest goes the other way and does both directions on the real bodies:
# uncomment the dead line and the disabled kernel must read as LIVE; comment out
# the live one and it must read as DISABLED.  A comment-stripper that got either
# direction wrong would otherwise pass quietly.
#
# NOTES ARE NOT GATES
#
# cloth_heightfieldVertexContactGenLaunch (:2581) is disabled the same way, by
# the same commented line, and is launched live too -- upstream candidate 34's
# sibling.  This port does not cover it, so nothing here depends on it and it is
# reported as a NOTE, not a gate: a check that fails for a change that breaks
# nothing teaches people to ignore checks.  Move it to the gated table on the
# day a Slang port of it lands.
#
# USAGE
#   check_disabled_kernels.py           check; exit 1 when a claim stops holding
#   check_disabled_kernels.py -v        also print the evidence lines
# ===========================================================================

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, HERE)
from check_cuda_twin_kernels import strip_comments   # noqa: E402

CU = 'physx/source/gpunarrowphase/src/CUDA/femClothPrimitives.cu'

# Kernels this port implements AS EMITTING NOTHING.  A change here breaks the
# port, so these fail the build.
GATED = [
    {
        'rel': CU,
        'kernel': 'cloth_meshVertexContactGenLaunch',
        'flag': 'hasContacts',
        'depends':
            'femClothMeshVertContactGenBda.slang implements this kernel as an '
            'empty walk plus one atomicMax, and deliberately omits the '
            'closest-point geometry as unverifiable dead code.',
    },
]

# Same defect, nothing in this port depends on it yet.  Reported, not gated.
NOTED = [
    {
        'rel': CU,
        'kernel': 'cloth_heightfieldVertexContactGenLaunch',
        'flag': 'hasContacts',
        'depends':
            'Not ported.  Recorded because it is the same upstream defect; '
            'gate it when a Slang port of it lands.',
    },
]

# Kernels that ARE live, analysed by the same code so that a green run proves
# this checker can still tell the two states apart.
LIVE_CONTROLS = [
    {'rel': CU, 'kernel': 'cloth_meshContactGenLaunch', 'flag': 'hasContact'},
    {'rel': CU, 'kernel': 'cloth_heightfieldContactGenLaunch', 'flag': 'hasContact'},
]


# ---------------------------------------------------------------------------
def function_text(src, name):
    """The raw, brace-matched body of `name` -- comments intact."""
    for m in re.finditer(r'\b' + re.escape(name) + r'\s*\(', src):
        try:
            open_brace = src.index('{', src.index(')', m.end() - 1))
        except ValueError:
            continue
        # a declaration ends in ';' before any '{'
        between = src[m.end():open_brace]
        if ';' in between:
            continue
        depth, k = 0, open_brace
        while k < len(src):
            if src[k] == '{':
                depth += 1
            elif src[k] == '}':
                depth -= 1
                if depth == 0:
                    return src[open_brace + 1:k]
            k += 1
    return None


def assignments(body, flag):
    """Every assignment to `flag`, as (line_no_within_body, text).

    An assignment is `flag =` (or a compound assignment), but not `flag ==`.
    A declaration with an initialiser counts -- the caller decides what to do
    with `= false`.

    Note there is deliberately NO lookbehind before the name: the commented-out
    line reads `//hasContacts = true;`, so anything that excludes a preceding
    `/` makes this checker blind to exactly the line it is here to find.  `\\b`
    already keeps it from matching inside a longer identifier.
    """
    pat = re.compile(r'\b' + re.escape(flag) + r'\s*(?:[+\-*/&|^]?=)(?!=)')
    hits = []
    for i, line in enumerate(body.split('\n')):
        if pat.search(line):
            hits.append((i, line.strip()))
    return hits


def classify(src, kernel, flag):
    """Read the kernel and decide DISABLED / LIVE / (a reason it is neither)."""
    raw = function_text(src, kernel)
    if raw is None:
        return None, 'the kernel is not in the file', {}

    live = strip_comments(raw)
    live_assigns = assignments(live, flag)
    all_assigns = assignments(raw, flag)

    # the declaration is the live assignment whose rhs is `false`
    decl = [(i, l) for (i, l) in live_assigns
            if re.search(r'\b' + re.escape(flag) + r'\s*=\s*false\s*;', l)]
    other_live = [(i, l) for (i, l) in live_assigns if (i, l) not in decl]

    # an assignment that is present in the raw body but NOT in the live one is
    # commented out.  Compare on text, since line numbers survive stripping.
    live_texts = {i for (i, _) in live_assigns}
    commented = [(i, l) for (i, l) in all_assigns if i not in live_texts]

    ev = {
        'decl': decl,
        'live': other_live,
        'commented': commented,
        'gate': re.search(r'\bif\s*\(\s*' + re.escape(flag) + r'\s*\)', live)
                is not None,
        'ballot': re.search(r'__ballot_sync\s*\([^)]*\b' + re.escape(flag) +
                            r'\b', live) is not None
                  or re.search(r'globalScanExclusiveSingleWarp\s*\(\s*' +
                               re.escape(flag) + r'\b', live) is not None,
        'lines': len(raw.split('\n')),
    }
    if not decl:
        return None, ('`%s` is no longer declared `= false` in %s'
                      % (flag, kernel)), ev
    return ('LIVE' if other_live else 'DISABLED'), None, ev


def output_index_still_gated(src, verbose):
    """PxgOutputIndex's atomicAdd must still be guarded by a non-zero popcount.

    This is what makes the contact COUNTER stay put with an always-false flag,
    and it is a separate fact in a separate place from the flag itself -- the
    port's `if (gid.x == 0u && lane == 0u) InterlockedMax(...)` is the kernel's
    ONLY write precisely because of it.
    """
    # the FIRST `PxgOutputIndex(` in the file is the constructor; `struct
    # PxgOutputIndex` carries no paren and the `PxgOutputIndex indexer(...)`
    # uses come later
    ctor = function_text(src, 'PxgOutputIndex')
    if ctor is None:
        print('  BROKEN   PxgOutputIndex has no constructor body any more.')
        return False
    ctor = strip_comments(ctor)
    popc = re.search(r'(\w+)\s*=\s*__popc\s*\(', ctor)
    if not popc:
        print('  BROKEN   PxgOutputIndex no longer takes a popcount.')
        return False
    name = popc.group(1)

    # Every atomicAdd must sit inside the region an `if` on that popcount
    # controls.  Done on character spans rather than by counting braces per
    # line, because the guard here is a BRACELESS-looking one-liner as far as a
    # line walk is concerned -- the `{` is on the following line, so a depth
    # comparison closes the guard before its body starts.
    guards = []
    for m in re.finditer(r'\bif\s*\(', ctor):
        depth, k = 0, m.end() - 1
        while k < len(ctor):
            if ctor[k] == '(':
                depth += 1
            elif ctor[k] == ')':
                depth -= 1
                if depth == 0:
                    break
            k += 1
        cond = ctor[m.end():k]
        if not re.search(r'\b' + re.escape(name) + r'\s*(?:>\s*0|!=\s*0)\b', cond):
            continue
        j = k + 1
        while j < len(ctor) and ctor[j].isspace():
            j += 1
        if j < len(ctor) and ctor[j] == '{':
            depth, e = 0, j
            while e < len(ctor):
                if ctor[e] == '{':
                    depth += 1
                elif ctor[e] == '}':
                    depth -= 1
                    if depth == 0:
                        break
                e += 1
            guards.append((j, e))
        else:
            e = ctor.find(';', j)
            guards.append((j, len(ctor) if e < 0 else e))

    ok, seen = True, 0
    for m in re.finditer(r'\batomicAdd\s*\(', ctor):
        seen += 1
        if not any(a <= m.start() <= b for (a, b) in guards):
            ok = False
            line = ctor[:m.start()].count('\n')
            print('  UNGATED  PxgOutputIndex performs an atomicAdd that is '
                  'no longer behind `%s > 0`:' % name)
            print('           %s' % ' '.join(ctor.split('\n')[line].split())[:100])
    if seen == 0:
        print('  BROKEN   PxgOutputIndex no longer performs an atomicAdd -- '
              'the counter argument in the port needs re-reading.')
        return False
    if ok and verbose:
        print('  OK       PxgOutputIndex: %d atomicAdd behind `%s > 0`'
              % (seen, name))
    elif ok:
        print('  OK       PxgOutputIndex still gates its atomicAdd on a '
              'non-zero popcount')
    return ok


def report(entry, src, verbose, gated):
    kernel, flag = entry['kernel'], entry['flag']
    state, why, ev = classify(src, kernel, flag)
    tag = '  ' if gated else '  (note) '
    if state is None:
        print('%sBROKEN   %s -- %s' % (tag, kernel, why))
        print('%s         %s' % (tag, entry['depends']))
        return False
    if state == 'LIVE':
        print('%sRE-ENABLED %s now assigns `%s` on a LIVE line:'
              % (tag, kernel, flag))
        for (_, l) in ev['live']:
            print('%s           %s' % (tag, l[:100]))
        print('%s         %s' % (tag, entry['depends']))
        return False
    if not ev['commented']:
        print('%sREWRITTEN %s has no commented-out assignment to `%s` any more.'
              % (tag, kernel, flag))
        print('%s         The kernel still emits nothing, but it is no longer '
              'the kernel this port read: re-read it before trusting the port.'
              % tag)
        print('%s         %s' % (tag, entry['depends']))
        return False
    if not ev['gate'] or not ev['ballot']:
        print('%sRESHAPED %s: gate=%s ballot=%s -- "emits nothing" no longer '
              'follows from `%s` alone.'
              % (tag, kernel, ev['gate'], ev['ballot'], flag))
        print('%s         %s' % (tag, entry['depends']))
        return False
    if verbose:
        print('%sOK       %-42s disabled (%d lines, %d commented assignment(s))'
              % (tag, kernel, ev['lines'], len(ev['commented'])))
        for (_, l) in ev['commented']:
            print('%s           %s' % (tag, l[:100]))
    else:
        print('%sOK       %s still emits nothing' % (tag, kernel))
    return True


# ---------------------------------------------------------------------------
def selftest():
    """Both directions, on the real bodies.

    The whole check rests on telling a commented line from a live one, so that
    distinction is calibrated rather than assumed: uncommenting the dead line
    must flip the disabled kernel to LIVE, and commenting out the live one must
    flip the live control to DISABLED.  A stripper that lost either direction
    would otherwise pass quietly for as long as upstream left the file alone.
    """
    path = os.path.join(ROOT, CU)
    src = open(path).read()

    entry = GATED[0]
    raw = function_text(src, entry['kernel'])
    if raw is None:
        print('SELFTEST BROKEN: %s is gone.' % entry['kernel'])
        return 2
    revived = re.sub(r'//\s*(' + re.escape(entry['flag']) + r'\s*=\s*true\s*;)',
                     r'\1', raw)
    if revived == raw:
        # The anchor is gone, which is itself a finding -- and the main check
        # above has already reported it in the words that fit whichever way it
        # went (RE-ENABLED or REWRITTEN).  Saying "SELFTEST BROKEN" here would
        # bury an accurate diagnosis under a misleading one.
        print('selftest: cannot calibrate -- there is no commented `%s = true;` '
              'in %s any more.  See the finding above; it says which way the '
              'source moved.' % (entry['flag'], entry['kernel']))
        return 2
    if assignments(strip_comments(revived), entry['flag']) == \
       assignments(strip_comments(raw), entry['flag']):
        print('SELFTEST FAILED: uncommenting the dead line did not make it '
              'visible -- this checker cannot see a re-enabled kernel, which '
              'is the only thing it exists to see.')
        return 2

    ctrl = LIVE_CONTROLS[0]
    craw = function_text(src, ctrl['kernel'])
    if craw is None:
        print('selftest: cannot calibrate -- the live control %s is gone.  See '
              'the finding above.' % ctrl['kernel'])
        return 2
    killed = re.sub(r'^(\s*)(' + re.escape(ctrl['flag']) + r'\s*=\s*[^;]*;)',
                    r'\1//\2', craw, flags=re.M)
    if assignments(strip_comments(killed), ctrl['flag']) == \
       assignments(strip_comments(craw), ctrl['flag']):
        print('SELFTEST FAILED: commenting out the live assignment in %s did '
              'not hide it -- the checker cannot see a disabled kernel either.'
              % ctrl['kernel'])
        return 2

    print('selftest: uncommenting the dead line makes %s read LIVE, and '
          'commenting out %s\'s makes it read DISABLED.  Both directions work.'
          % (entry['kernel'], ctrl['kernel']))
    return 0


def main():
    verbose = '-v' in sys.argv or '--verbose' in sys.argv
    if '--selftest-only' in sys.argv:
        return selftest()

    # The selftest runs AFTER the checks here, unlike its siblings in this
    # directory.  It calibrates itself by mutating the very lines the checks
    # read, so when the source moves it cannot calibrate -- and running it first
    # would replace an accurate finding ("upstream re-enabled the line") with a
    # misleading one ("the anchor is missing").  Both still gate the build.
    path = os.path.join(ROOT, CU)
    if not os.path.exists(path):
        print('MISSING %s -- nothing can be checked.' % CU)
        return 1
    src = open(path).read()

    print('disabled-kernel check over %d gated claim(s), %d note(s), '
          '%d live control(s)' % (len(GATED), len(NOTED), len(LIVE_CONTROLS)))
    failures = 0

    for entry in GATED:
        if not report(entry, src, verbose, True):
            failures += 1
    for entry in NOTED:
        report(entry, src, verbose, False)

    for ctrl in LIVE_CONTROLS:
        state, why, ev = classify(src, ctrl['kernel'], ctrl['flag'])
        if state == 'LIVE':
            if verbose:
                print('  OK       %-42s live (%d lines, %d assignment(s))'
                      % (ctrl['kernel'], ev['lines'], len(ev['live'])))
            else:
                print('  OK       %s still writes contacts' % ctrl['kernel'])
            continue
        failures += 1
        print('  CONTROL  %s reads as %s%s -- this checker can no longer tell '
              'a live kernel from a disabled one, so its OK lines above mean '
              'nothing.' % (ctrl['kernel'], state, ' (%s)' % why if why else ''))

    if not output_index_still_gated(src, verbose):
        failures += 1

    if selftest():
        failures += 1

    if failures:
        print('\n%d claim(s) about the vendored CUDA no longer hold.  The port '
              'reads these kernels as emitting nothing and implements them that '
              'way; if that changed, port the geometry -- and give it a fixture '
              'that can now actually observe a contact.' % failures)
        return 1
    print('\nOK -- every kernel this port implements as silent is still silent '
          'upstream, and the checker can still tell silent from speaking.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
