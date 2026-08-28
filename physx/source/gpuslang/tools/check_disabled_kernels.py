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
# A kernel disabled the same way that this port does NOT cover belongs in NOTED
# rather than GATED: a check that fails for a change breaking nothing teaches
# people to ignore checks.  cloth_heightfieldVertexContactGenLaunch (:2581) sat
# there from P9.26 with the instruction "move it to the gated table on the day a
# Slang port of it lands".  P9.27b is that day, so it has moved, and NOTED is
# empty -- kept for the next one rather than deleted.
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
    {
        'rel': CU,
        'kernel': 'cloth_heightfieldVertexContactGenLaunch',
        'flag': 'hasContacts',
        'depends':
            'femClothHfContactGenBda.slang implements this kernel the same way, for '
            'the same reason.  It was a NOTE here from P9.26, with the instruction '
            '"gate it when a Slang port of it lands" -- P9.27b is that landing.',
    },
]

# Same defect, but NOT covered by this port: reported, never gated.  Empty since
# P9.27b moved its one entry into GATED.  See "NOTES ARE NOT GATES" above.
NOTED = []

# ---------------------------------------------------------------------------
# A THIRD KIND OF DISABLEMENT, found in P9.29: the HOST LAUNCH SITE.
#
# cloth_SphereVertexContactGenLaunch is a complete, live device function -- its
# `hasContacts` equivalent is assigned on a live line, so everything above reads
# it as LIVE and says so correctly.  It still never runs, because the host wraps
# its launch in `if (0)`:
#
#     if (1)  // contact gen                      PxgNarrowphaseCore.cpp:5735
#     { ... getCuFunction(PxgKernelIds::CLOTH_SPHERE_CG) ... }
#
#     if (0)  // cloth vertex vs. sphere contact gen.          :5774
#     { ... getCuFunction(PxgKernelIds::CLOTH_VERT_SPHERE_CG) ... }
#
# No amount of reading the .cu finds that, which is exactly why it needs its own
# table: the two mechanisms live in different files and neither checker sees the
# other's.  What the port depends on here is weaker than for GATED -- the vertex
# kernel IS ported, and its eCAPSULE branch is deliberately incomplete -- so the
# claim being kept true is "nothing in the engine can reach that gap".
#
# The check is deliberately EXHAUSTIVE rather than per-entry: it brace-matches
# every `if (0)` in the host file and demands that the set of kernels launched
# inside them is exactly this table.  A new one appearing anywhere fails the
# build and gets read, which is how this one was found in the first place.
CPP = 'physx/source/gpunarrowphase/src/PxgNarrowphaseCore.cpp'

HOST_GATED = [
    {
        'rel': CPP,
        'enum': 'CLOTH_VERT_SPHERE_CG',
        'kernel': 'cloth_SphereVertexContactGenLaunch',
        'depends':
            'femClothContactGenBda.slang ports this kernel with its eCAPSULE branch '
            'DEFERRED (a sphere-only body).  That gap is unreachable only while the '
            'host launch stays inside `if (0)`; the day it does not, the capsule '
            'branch has to be written and given a fixture.',
    },
]

# The host-side LIVE control: the sibling launched from `if (1)` in the same
# function, three lines up.  If the scanner ever reports THIS one as gated it is
# matching text rather than structure, and its findings mean nothing.
HOST_LIVE_CONTROLS = ['CLOTH_SPHERE_CG']

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


def host_gated_launches(cpp_src):
    """Every kernel launched inside an `if (0)` block, brace-matched.

    Returns (gated_enums, n_zero_blocks).  Comments are stripped first, so a
    commented-out `if (0)` cannot manufacture a finding; the block is then
    matched on braces rather than indentation, because several of these blocks
    nest further `if`s.
    """
    live = strip_comments(cpp_src)
    gated, blocks = set(), 0
    for m in re.finditer(r'\bif\s*\(\s*0\s*\)', live):
        blocks += 1
        j = m.end()
        while j < len(live) and live[j].isspace():
            j += 1
        if j >= len(live) or live[j] != '{':
            continue                     # a braceless `if (0) stmt;` -- no launch
        depth, k = 0, j
        while k < len(live):
            if live[k] == '{':
                depth += 1
            elif live[k] == '}':
                depth -= 1
                if depth == 0:
                    break
            k += 1
        body = live[j:k]
        for f in re.finditer(r'getCuFunction\s*\(\s*PxgKernelIds::(\w+)\s*\)', body):
            gated.add(f.group(1))
    return gated, blocks


def check_host_gates(verbose):
    """The `if (0)` launch sites, checked as a SET rather than one by one."""
    path = os.path.join(ROOT, CPP)
    if not os.path.exists(path):
        print('  MISSING  %s -- the host launch gates cannot be checked.' % CPP)
        return False
    cpp_src = open(path).read()
    gated, blocks = host_gated_launches(cpp_src)
    expected = {e['enum'] for e in HOST_GATED}

    ok = True
    for e in HOST_GATED:
        if e['enum'] not in gated:
            ok = False
            print('  RE-ENABLED %s (%s) is no longer launched from an `if (0)` '
                  'block.' % (e['enum'], e['kernel']))
            print('           %s' % e['depends'])
    for extra in sorted(gated - expected):
        ok = False
        print('  NEW-GATE %s is launched inside an `if (0)` block and is not in '
              'HOST_GATED.  Read it: a kernel the engine never runs is a kernel '
              'whose port cannot be validated by any scene.' % extra)

    # the control, from the same scan rather than a separate one
    for live_enum in HOST_LIVE_CONTROLS:
        if live_enum in gated:
            ok = False
            print('  CONTROL  %s reads as host-gated, and it is launched from '
                  '`if (1)` three lines from the gated one.  This scanner is '
                  'matching text, not structure.' % live_enum)
        elif not re.search(r'PxgKernelIds::' + re.escape(live_enum) + r'\b', cpp_src):
            ok = False
            print('  CONTROL  %s is not launched anywhere in %s any more, so it '
                  'cannot serve as the live control.' % (live_enum, CPP))

    if ok:
        if verbose:
            print('  OK       host gates: %d `if (0)` block(s), %d of them launch a '
                  'kernel: %s (control %s is live)'
                  % (blocks, len(gated), ', '.join(sorted(gated)) or '-none-',
                     ', '.join(HOST_LIVE_CONTROLS)))
        else:
            print('  OK       %d `if (0)` host block(s), %d launching: exactly the '
                  'expected set' % (blocks, len(gated)))
    return ok


def host_selftest():
    """Calibrate the host-gate scanner on the REAL file, in three directions.

    Same doctrine as the selftest below: a check that REASONS about a mutant can
    be wrong, one that RUNS it cannot.  All three mutations are applied to the
    source STRING -- the file on disk is never touched.

      1. re-enable the gated launch  -> it must stop being reported;
      2. gate the live control       -> it must start being reported;
      3. comment the `if (0)` out    -> it must NOT be reported, which is what
                                        separates reading structure from
                                        grepping for two characters.
    """
    path = os.path.join(ROOT, CPP)
    if not os.path.exists(path):
        return 0                       # check_host_gates already said so
    src = open(path).read()
    gate_re = re.compile(r'if\s*\(\s*0\s*\)(\s*//\s*cloth vertex vs\. sphere)')
    if not gate_re.search(src):
        print('host selftest: cannot calibrate -- the `if (0)` before the cloth '
              'vertex-sphere launch is gone.  See the finding above.')
        return 2

    revived = gate_re.sub(r'if (1)\1', src)
    if 'CLOTH_VERT_SPHERE_CG' in host_gated_launches(revived)[0]:
        print('HOST SELFTEST FAILED: turning the `if (0)` into `if (1)` did not '
              'clear the finding -- this scanner cannot see a re-enabled launch, '
              'which is the only thing it exists to see.')
        return 2

    live_re = re.compile(r'if\s*\(\s*1\s*\)(\s*//\s*contact gen)')
    if not live_re.search(src):
        print('host selftest: cannot calibrate -- the `if (1)` before the live '
              'control launch is gone.')
        return 2
    killed = live_re.sub(r'if (0)\1', src)
    if 'CLOTH_SPHERE_CG' not in host_gated_launches(killed)[0]:
        print('HOST SELFTEST FAILED: gating the live control did not produce a '
              'finding -- the scanner cannot see a newly disabled launch either.')
        return 2

    hidden = gate_re.sub(r'/* if (0) */ if (1)\1', src)
    if 'CLOTH_VERT_SPHERE_CG' in host_gated_launches(hidden)[0]:
        print('HOST SELFTEST FAILED: a COMMENTED-OUT `if (0)` still produced a '
              'finding -- the scanner is grepping for two characters rather than '
              'reading the code that runs.')
        return 2

    print('host selftest: re-enabling the gate clears it, gating the control '
          'raises it, and a commented-out `if (0)` raises nothing.  All three '
          'directions work.')
    return 0


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
          '%d live control(s), %d host gate(s)'
          % (len(GATED), len(NOTED), len(LIVE_CONTROLS), len(HOST_GATED)))
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

    if not check_host_gates(verbose):
        failures += 1

    if selftest():
        failures += 1
    if host_selftest():
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
