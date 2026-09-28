#!/usr/bin/env python3
"""foldscan_all.py -- the P9.86f sign-of-zero fold lint over every engine-reached shader.

    python3 tools/foldscan_all.py <build-gpuslang dir> <out dir> [-j N]

WHY. The source-level P9.86 audit missed six engine divergences: the literal zero was not
visible in the source.  It came from an inlined constant argument (hasJf = false -> 0 - x), a
const local (absFrom - to), a vector constructor lane (p + float3(dx, 0, 0)), or it was a plain
float min/max whose CUDA twin is a PxMin/PxMax select.  llvmpipe folds fsub(+0, x) -> -x and
fadd(x, +0) -> x even under SignedZeroInfNanPreserve, and Slang min/max are GLSL FMin/FMax.

WHAT.
  1. Every .spv the kernel manifest maps is recompiled with the build's own slangc command
     plus -g2 (NonSemantic DebugLine gives source file:line) into <out>/spv.
  2. spv_foldscan.py reports, in the functions reachable from the manifest entry points:
       F2 OpFSub whose first operand may be a literal +0 (scalar or lane),
       F3 OpFAdd with an operand that may be a literal +0,
       F4 GLSL FMin/FMax/FClamp/NMin/NMax on floats,
     through a conservative may-be-literal-zero dataflow (stores, lanes, call-site arguments,
     returns).  It re-finds all six bugs above on the pre-P9.86c tree (43cadd4).
  3. Hits whose zero comes only from the project's own select remedy
     ((x == 0) ? 0 : ...) are dropped.
  4. src_f1scan.py adds F1 (slangc's source-level 0 + x / zero-seeded straight-line add),
     which leaves no trace in SPIR-V, restricted to the reached source ranges.
Output: <out>/sites.json (unique source sites with kinds, zero origins and code text),
<out>/f1.jsonl, <out>/reached_ranges.json.  A site is a CANDIDATE, not a bug: it must be
judged against the CUDA twin (see RUNTIME_STATUS P9.86f for the classification rules).
"""
import collections, json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))           # physx/source/gpuslang
SHADERS = os.path.join(ROOT, 'shaders')
MANIFEST = os.path.join(ROOT, 'include', 'PxgSlangKernelManifest.h')
REMEDY = re.compile(r'==\s*0(\.0*)?f?\s*\)?\s*\?\s*(0(\.0*)?f?|float\d?\(0)')


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    build, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    jobs = int(sys.argv[sys.argv.index('-j') + 1]) if '-j' in sys.argv else os.cpu_count() or 4
    os.makedirs(os.path.join(out, 'spv'), exist_ok=True)

    man = open(MANIFEST).read()
    eng = collections.defaultdict(set)
    for _, spv, entry in re.findall(r'\{\s*"(\w+)",\s*"(\w+)\.spv",\s*"(\w+)"\s*\}', man):
        eng[spv].add(entry)
    bm = open(os.path.join(build, 'CMakeFiles', 'gpuslang-shaders.dir', 'build.make')).read()
    cmds = {}
    for m in re.finditer(r'(\S*slangc \S+/(\w+)\.slang [^\n&]*?-o \S+/(\w+)\.spv)', bm):
        cmds[m.group(3)] = m.group(1)

    def compile_one(spv):
        c = cmds.get(spv)
        if not c:
            return spv, 'no slangc command in build.make'
        c = c.replace(' -O0 ', ' -O0 -g2 ')
        c = re.sub(r' -o \S+', ' -o ' + os.path.join(out, 'spv', spv + '.spv'), c)
        r = subprocess.run(c, shell=True, capture_output=True, text=True)
        return spv, None if r.returncode == 0 else r.stderr[-400:]

    with ThreadPoolExecutor(jobs) as ex:
        for spv, err in ex.map(compile_one, sorted(eng)):
            if err:
                print('compile failed:', spv, err, file=sys.stderr)

    def scan_one(spv):
        p = os.path.join(out, 'spv', spv + '.spv')
        if not os.path.exists(p):
            return []
        r = subprocess.run([sys.executable, os.path.join(HERE, 'spv_foldscan.py'), p, ','.join(sorted(eng[spv]))],
                           capture_output=True, text=True)
        return [json.loads(l) for l in r.stdout.splitlines() if l.strip()]

    raw = []
    with ThreadPoolExecutor(jobs) as ex:
        for rows in ex.map(scan_one, sorted(eng)):
            raw.extend(rows)

    src_cache = {}

    def src_line(f, ln):
        if f not in src_cache:
            p = os.path.join(SHADERS, f)
            src_cache[f] = open(p).read().split('\n') if os.path.exists(p) else []
        L = src_cache[f]
        return L[ln - 1] if 0 < ln <= len(L) else ''

    ranges = collections.defaultdict(set)
    sites = collections.defaultdict(lambda: {'kinds': set(), 'via': set(), 'orig': set(), 'funcs': set()})
    for h in raw:
        if 'range' in h:
            ranges[h['range']].add((h['lo'], h['hi']))
            continue
        if 'kind' not in h:
            continue
        orig = h.get('orig', [])
        if not h['kind'].startswith('F4') and orig and all(
                REMEDY.search(src_line(o.rsplit(':', 1)[0], int(o.rsplit(':', 1)[1]))) for o in orig):
            continue
        s = sites[(h['file'], h['line'])]
        s['kinds'].add(h['kind'])
        s['via'] |= set(h['via'])
        s['orig'] |= set(orig)
        s['funcs'].add(h['func'])

    rr = {f: sorted(v) for f, v in ranges.items()}
    json.dump(rr, open(os.path.join(out, 'reached_ranges.json'), 'w'))
    out_sites = [{'file': f, 'line': l, 'code': src_line(f, l).strip(), 'kinds': sorted(s['kinds']),
                  'zeroVia': sorted(s['via']), 'zeroOrigin': sorted(s['orig'])[:8], 'funcs': sorted(s['funcs'])[:4]}
                 for (f, l), s in sorted(sites.items())]
    json.dump(out_sites, open(os.path.join(out, 'sites.json'), 'w'), indent=0)

    r = subprocess.run([sys.executable, os.path.join(HERE, 'src_f1scan.py'), os.path.join(out, 'reached_ranges.json'),
                        SHADERS + '/'], capture_output=True, text=True)
    open(os.path.join(out, 'f1.jsonl'), 'w').write(r.stdout)
    f1 = [json.loads(l) for l in r.stdout.splitlines() if l.strip()]
    kinds = collections.Counter(k for s in out_sites for k in s['kinds'])
    print('engine spv: %d; unique candidate sites: %d in %d files; %s' % (
        len(eng), len(out_sites), len({s['file'] for s in out_sites}), dict(kinds)))
    print('F1 source candidates: %s' % dict(collections.Counter((x['kind'], x['remedied']) for x in f1)))

    # compare with the reviewed baseline: every site must have a verdict (keyed by file + code text)
    base_path = os.path.join(HERE, 'foldscan_reviewed.json')
    if os.path.exists(base_path):
        norm = lambda c: ' '.join(c.split())
        base = {(b['file'], norm(b['code'])): b for b in json.load(open(base_path))}
        new = [s for s in out_sites if (s['file'], norm(s['code'])) not in base]
        open_fix = [s for s in out_sites if base.get((s['file'], norm(s['code'])), {}).get('verdict') == 'MATTERS-FIX']
        json.dump(new, open(os.path.join(out, 'new_sites.json'), 'w'), indent=0)
        print('reviewed baseline: %d entries; UNREVIEWED sites: %d; still-open MATTERS-FIX: %d' % (
            len(base), len(new), len(open_fix)))
        for s in new + open_fix:
            print('  %s:%d %s  %s' % (s['file'], s['line'], ','.join(s['kinds']), s['code'][:100]))
        if new or open_fix:
            sys.exit(1)


if __name__ == '__main__':
    main()
