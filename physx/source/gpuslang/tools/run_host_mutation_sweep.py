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
# Environment: MUTATION_SWEEP_TIMEOUT (seconds per suite run, default 300),
# PXG_BUILD_DIR (default <repo>/build-gpuslang), PXG_BUILD_TARGET (default
# gpuslang-tests).  PXG_SHADER_DIR defaults to <build>/shaders.

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GPUSLANG = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(GPUSLANG, '..', '..', '..'))


def main():
    if len(sys.argv) != 4:
        print('usage: python3 run_host_mutation_sweep.py <source-file> <ctest-regex> <mutants-file>',
              file=sys.stderr)
        return 2
    src = os.path.join(GPUSLANG, sys.argv[1])
    regex = sys.argv[2]
    build = os.environ.get('PXG_BUILD_DIR', os.path.join(REPO, 'build-gpuslang'))
    target = os.environ.get('PXG_BUILD_TARGET', 'gpuslang-tests')
    tmo = int(os.environ.get('MUTATION_SWEEP_TIMEOUT', '300'))
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
        try:
            r = subprocess.run(cmd, shell=True, cwd=build, env=env, capture_output=True, text=True,
                               timeout=timeout)
            return r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return 'TIMEOUT', ''

    def rebuild():
        return sh('cmake --build . -j4 --target %s' % target, 3600)

    def run():
        return sh("ctest -R '%s' --timeout %d" % (regex, tmo), tmo * 4 + 60)

    pristine = open(src).read()
    counts = {}
    try:
        rc, out = rebuild()
        if rc != 0:
            print('BASELINE BUILD FAILED\n' + out[-3000:])
            return 1
        rc, out = run()
        if rc != 0:
            print('BASELINE RED -- nothing below would mean anything\n' + out[-3000:])
            return 1
        print('baseline green; sweeping %s against %s' % (sys.argv[1], regex), flush=True)
        for name, old, new in mutants:
            n = pristine.count(old)
            if n != 1:
                res = 'ANCHOR-FAIL'
                print('%-13s %s (anchor occurs %d times)' % (res, name, n), flush=True)
                counts[res] = counts.get(res, 0) + 1
                continue
            with open(src, 'w') as f:
                f.write(pristine.replace(old, new))
            rc, out = rebuild()
            if rc != 0:
                res = 'NOCOMPILE'
            else:
                rc, out = run()
                if rc == 0:
                    res = 'SURVIVED'
                elif rc == 'TIMEOUT' or '***Timeout' in out:
                    res = 'KILLED-HANG'
                elif 'SEGFAULT' in out or '***Exception' in out or 'Child aborted' in out:
                    res = 'KILLED-FAULT'
                else:
                    res = 'KILLED'
            counts[res] = counts.get(res, 0) + 1
            print('%-13s %s' % (res, name), flush=True)
        summary = ', '.join('%d %s' % (v, k) for k, v in sorted(counts.items()))
        print('=== host sweep: %d mutants -- %s ===' % (len(mutants), summary), flush=True)
    finally:
        with open(src, 'w') as f:
            f.write(pristine)
        rc1, _ = rebuild()
        rc2, _ = run()
        print('source restored; rebuild rc=%s; baseline rerun %s' % (rc1, 'green' if rc2 == 0 else 'RED (rc=%s)' % rc2),
              flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
