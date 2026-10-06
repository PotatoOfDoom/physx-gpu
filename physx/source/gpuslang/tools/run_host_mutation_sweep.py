#!/usr/bin/env python3
# run_host_mutation_sweep.py -- the forced-fresh mutation sweep for HOST C++ code.
#
# WHY THIS EXISTS.  tools/run_mutation_sweep.sh mutates SHADERS: it rebuilds the .spv
# and reruns a suite.  P9.84 fixed a host-side defect (PxgCudaShim::pipelineFor never
# built a bindless pipeline) and there was no way to sweep it, so the discipline that
# tool enforces is repeated here for a C++ source file:
#
#   1. the pristine source is snapshotted once and restored in a `finally`, so an
#      exception or ^C cannot leave a mutant behind;
#   2. the BASELINE must be green before anything is reported, and is rerun after
#      the restore -- a sweep whose baseline is red is measuring nothing;
#   3. the test binary is REBUILT after every mutant and after the restore, so no
#      run ever measures a stale binary;
#   4. every anchor must occur EXACTLY once, else ANCHOR-FAIL (untested, never a kill);
#   5. every suite run is bounded; a timeout is KILLED-HANG, a crash KILLED-FAULT.
#
# Usage (from anywhere; paths are resolved against the repo layout):
#   python3 run_host_mutation_sweep.py <source-file> <ctest-regex> <mutants-file>
#
# <source-file>  path relative to physx/source/gpuslang, e.g. src/PxgCudaShim.cpp
# <ctest-regex>  passed to `ctest -R`, e.g. 'gpuslang\.(shimBindless|p3Shim)$'
# <mutants-file> the SAME format as run_mutation_sweep.sh: one mutant per line,
#                name <TAB> old-text <TAB> new-text, with \n for newlines.
#
# Environment: PXG_BUILD_DIR (default <repo>/build-gpuslang), PXG_BUILD_TARGET
# (default gpuslang-tests).  PXG_SHADER_DIR defaults to <build>/shaders.
#
# The per-test bound (P9.97) follows tools/sweep_timeout.sh, with the same knobs:
# MUTATION_SWEEP_TIMEOUT fixes it; unset, the baseline ctest run is timed (bounded by
# MUTATION_SWEEP_BASELINE_TIMEOUT, default 3600) and every later `ctest --timeout` is
# max(MUTATION_SWEEP_TIMEOUT_FLOOR (300), MUTATION_SWEEP_TIMEOUT_FACTOR (3) x baseline).
# The baseline is the WHOLE matched set, so the per-test bound is never below what the
# slowest single test needs.
#
# ctest's own --timeout bounds each test.  The OUTER wall-clock bound around the whole
# ctest call is only a backstop for a ctest that itself does not return: N x the bound
# + MUTATION_SWEEP_OUTER_SLACK (default 60 s), N = the number of tests the regex
# matches (`ctest -N`), so every test may use its whole bound.  (A first P9.97 draft
# used 2 x bound + 60, which in FIXED mode is below what N slow-but-green tests need:
# measured, 5 x 30 s tests under MUTATION_SWEEP_TIMEOUT=31 were refused as a timed-out
# baseline.)  An outer expiry is reported as CTEST-WEDGED, not as a per-test hang.  On
# expiry the whole process group is killed -- killing only the shell left the test
# binary running (P9.97 measured it).
#
# A regex that matches NO test is refused before anything is built (`ctest -R` with no
# match exits 0: a green baseline and every mutant SURVIVED), and --no-tests=error is
# passed as well.  A baseline that is not green is reported and NOT re-run: the closing
# rebuild + re-run happens only when a mutant was written.

import os
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GPUSLANG = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(GPUSLANG, '..', '..', '..'))


def derive_timeout(baseline_s, factor, floor):
    """max(floor, factor x baseline) -- the table in tools/test_sweep_timeout.sh."""
    return max(int(floor), int(factor) * int(baseline_s))


def knob(name, default):
    """Whole seconds > 0, no leading 0 -- the same rule as sweep_uint in sweep_timeout.sh."""
    v = os.environ.get(name, '')
    if v == '':
        return default
    if not v.isdigit() or v.startswith('0'):
        raise SystemExit('%s=%r is not a whole number of seconds > 0' % (name, v))
    return int(v)


def run_bounded(cmd, cwd, env, timeout):
    """Run a shell command in its OWN process group; (rc, output).  On expiry the whole
    group is SIGKILLed and rc is 'TIMEOUT' (with whatever output had arrived)."""
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, _ = p.communicate()
        return 'TIMEOUT', out or ''


def count_tests(regex, cwd, env):
    """How many ctest tests the regex matches (`ctest -N`), or None if ctest -N failed."""
    rc, out = run_bounded("ctest -N -R '%s'" % regex, cwd, env, 300)
    m = re.search(r'Total Tests: (\d+)', out or '')
    return int(m.group(1)) if rc == 0 and m else None


def main():
    if len(sys.argv) != 4:
        print('usage: python3 run_host_mutation_sweep.py <source-file> <ctest-regex> <mutants-file>',
              file=sys.stderr)
        return 2
    src = os.path.join(GPUSLANG, sys.argv[1])
    regex = sys.argv[2]
    build = os.environ.get('PXG_BUILD_DIR', os.path.join(REPO, 'build-gpuslang'))
    target = os.environ.get('PXG_BUILD_TARGET', 'gpuslang-tests')
    fixed = knob('MUTATION_SWEEP_TIMEOUT', None)
    factor = knob('MUTATION_SWEEP_TIMEOUT_FACTOR', 3)
    floor = knob('MUTATION_SWEEP_TIMEOUT_FLOOR', 300)
    if factor < 2:
        raise SystemExit('MUTATION_SWEEP_TIMEOUT_FACTOR=%d leaves no headroom (need >= 2)' % factor)
    tmo = fixed if fixed is not None else knob('MUTATION_SWEEP_BASELINE_TIMEOUT', 3600)
    slack = knob('MUTATION_SWEEP_OUTER_SLACK', 60)
    env = dict(os.environ)
    env.setdefault('PXG_SHADER_DIR', os.path.join(build, 'shaders'))

    mutants = []
    with open(sys.argv[3]) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                print('malformed mutant line (need 3 TAB-separated fields): %r' % line, file=sys.stderr)
                return 2
            mutants.append((parts[0], parts[1].replace('\\n', '\n'), parts[2].replace('\\n', '\n')))

    def sh(cmd, timeout):
        return run_bounded(cmd, build, env, timeout)

    def rebuild():
        return sh('cmake --build . -j4 --target %s' % target, 3600)

    n_tests = count_tests(regex, build, env)
    if not n_tests:
        print('the regex %r matched NO test (ctest -N in %s) -- refusing: every mutant would '
              'read as SURVIVED.  Check the regex, or configure/build first (an unconfigured '
              'or unbuilt tree lists no tests)' % (regex, build))
        return 1

    def run():
        t0 = time.monotonic()
        rc, out = sh("ctest -R '%s' --timeout %d --no-tests=error" % (regex, tmo), n_tests * tmo + slack)
        return rc, out, time.monotonic() - t0

    pristine = open(src).read()
    counts = {}
    mutant_written = False
    try:
        rc, out = rebuild()
        if rc != 0:
            print('BASELINE BUILD FAILED\n' + out[-3000:])
            return 1
        rc, out, secs = run()
        if rc == 'TIMEOUT':
            print('BASELINE: ctest itself did not return within %ds (%d test(s) x %ds + %ds) -- '
                  'a wedged ctest, not a slow test; nothing measured' % (n_tests * tmo + slack, n_tests, tmo, slack))
            return 1
        if '***Timeout' in out:
            print('BASELINE TIMED OUT at %ds -- raise %s\n%s' % (tmo, 'MUTATION_SWEEP_TIMEOUT' if fixed
                  else 'MUTATION_SWEEP_BASELINE_TIMEOUT', out[-3000:]))
            return 1
        if rc != 0:
            print('BASELINE RED -- nothing below would mean anything\n' + out[-3000:])
            return 1
        base = int(secs + 0.999)
        if fixed is None:
            tmo = derive_timeout(base, factor, floor)
            how = 'max(floor %ds, %d x baseline %ds)' % (floor, factor, base)
        else:
            how = 'fixed by MUTATION_SWEEP_TIMEOUT'
            if 2 * base > tmo:
                print('!! warning: the clean run took %ds, more than half of MUTATION_SWEEP_TIMEOUT=%ds' % (base, tmo))
        print('baseline green in %ds; every ctest test below is bounded by %ds (%s); %d test(s), outer bound %ds'
              % (base, tmo, how, n_tests, n_tests * tmo + slack), flush=True)
        print('baseline green; sweeping %s against %s' % (sys.argv[1], regex), flush=True)
        for name, old, new in mutants:
            n = pristine.count(old)
            if n != 1:
                res = 'ANCHOR-FAIL'
                print('%-13s %s (anchor occurs %d times)' % (res, name, n), flush=True)
                counts[res] = counts.get(res, 0) + 1
                continue
            mutant_written = True
            with open(src, 'w') as f:
                f.write(pristine.replace(old, new))
            rc, out = rebuild()
            why = ''
            if rc != 0:
                res = 'NOCOMPILE'
            else:
                rc, out, _ = run()
                if rc == 0:
                    res = 'SURVIVED'
                elif rc == 'TIMEOUT':
                    res = 'KILLED-HANG'
                    why = '   (CTEST-WEDGED: ctest itself did not return within %ds)' % (n_tests * tmo + slack)
                elif '***Timeout' in out:
                    res = 'KILLED-HANG'
                    why = '   (a test exceeded %ds)' % tmo
                elif 'SEGFAULT' in out or '***Exception' in out or 'Child aborted' in out:
                    res = 'KILLED-FAULT'
                else:
                    res = 'KILLED'
            counts[res] = counts.get(res, 0) + 1
            print('%-13s %s%s' % (res, name, why), flush=True)
        summary = ', '.join('%d %s' % (v, k) for k, v in sorted(counts.items()))
        print('=== host sweep: %d mutants -- %s ===' % (len(mutants), summary), flush=True)
    finally:
        with open(src, 'w') as f:
            f.write(pristine)
        # Rebuild and re-run only if a mutant was ever written: after a refused baseline
        # the tree never changed, and a re-run would only cost the refusal a second time
        # (P9.97 measured 2 x 122 s for one timed-out baseline).
        if mutant_written:
            rc1, _ = rebuild()
            rc2, _, _ = run()
            print('source restored; rebuild rc=%s; baseline rerun %s'
                  % (rc1, 'green' if rc2 == 0 else 'RED (rc=%s)' % rc2), flush=True)
        else:
            print('source unchanged (no mutant was written); nothing to rebuild or re-run', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
