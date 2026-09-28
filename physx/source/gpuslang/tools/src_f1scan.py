#!/usr/bin/env python3
"""P9.86f: source-level F1 candidates (slangc folds 0 + x / zero-seeded straight-line adds).

For every engine-reached source range (reached_ranges.json from foldscan.py), find
float-like variables assigned a literal zero, whose FIRST later `+=` / `v = v + ...`
update is not inside a loop opened after the zero assignment (straight-line, possibly
under an `if`). Also report literal `0.0f + x` / `x + 0.0f` expressions.
Output: JSON lines {file, zeroLine, updateLine, var, kind, zeroCode, updateCode, remedied}.
"""
import json, re, sys, os

SH = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shaders') + '/'
R = json.load(open(sys.argv[1]))

ZNUM = r'(?:0(?:\.0*)?f?|\.0+f?)'
ZVEC = r'(?:float[234]\s*\(\s*' + ZNUM + r'(?:\s*,\s*' + ZNUM + r')*\s*\))'
ZSV = r'(?:asmSVec\s*\(\s*' + ZVEC + r'\s*,\s*' + ZVEC + r'\s*\)|AI_ZERO|AB_SV_ZERO|abSvZero\s*\(\s*\)|abMat33Zero\s*\(\s*\))'
ZERO = r'(?:' + ZNUM + '|' + ZVEC + '|' + ZSV + ')'
TYPES = r'(?:float[234]?|half[234]?|AbSpatialVector|AbMat33|float3x3)'
DECL = re.compile(r'\b' + TYPES + r'\s+(\w+)\s*=\s*' + ZERO + r'\s*;')
ASSIGN = re.compile(r'(?<![\w.\]])(\w+)\s*=\s*' + ZERO + r'\s*;')
LIT = re.compile(r'(?<![\w.])' + ZNUM + r'\s*\+(?!\+)|\+\s*' + ZNUM + r'(?![\w.(])')
LOOP = re.compile(r'\b(for|while|do)\b')


def strip_comments(text):
    out, i, n = [], 0, len(text)
    while i < n:
        if text.startswith('//', i):
            j = text.find('\n', i)
            j = n if j < 0 else j
            i = j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            j = n if j < 0 else j + 2
            out.append('\n' * text.count('\n', i, j))
            i = j
        else:
            out.append(text[i])
            i += 1
    return ''.join(out)


def in_ranges(ranges, ln):
    return any(lo <= ln <= hi for lo, hi in ranges)


results = []
for f, ranges in sorted(R.items()):
    path = SH + f
    if not os.path.exists(path):
        continue
    raw = open(path).read()
    rawlines = raw.split('\n')
    lines = strip_comments(raw).split('\n')
    # block structure: for each line, the stack of open blocks (start line, is_loop)
    stack, pending_loop = [], False
    blocks_at = []
    for i, L in enumerate(lines):
        if LOOP.search(L):
            pending_loop = True
        for ch in L:
            if ch == '{':
                stack.append((i + 1, pending_loop))
                pending_loop = False
            elif ch == '}':
                if stack:
                    stack.pop()
        # a braceless loop body on the next line counts as a loop too
        blocks_at.append(list(stack))
    for i, L in enumerate(lines):
        ln = i + 1
        if not in_ranges(ranges, ln):
            continue
        for m in list(DECL.finditer(L)) + list(ASSIGN.finditer(L)):
            var = m.group(1)
            if var in ('return',):
                continue
            upd = re.compile(r'(?<![\w.])' + re.escape(var) + r'(?:\s*\[[^\]]*\]|\.\w+)?\s*(\+=)|(?<![\w.])' + re.escape(var)
                             + r'\s*=\s*' + re.escape(var) + r'\s*\+')
            reassign = re.compile(r'(?<![\w.])' + re.escape(var) + r'\s*=(?!=)')
            depth0 = len(blocks_at[i])
            for j in range(i + 1, min(len(lines), i + 400)):
                Lj = lines[j]
                # stop when the declaring scope closes
                if len(blocks_at[j]) < depth0 and '}' in Lj:
                    break
                if upd.search(Lj):
                    opened = [b for b in blocks_at[j] if b[0] > ln and b[1]]
                    loopish = bool(opened) or bool(LOOP.search(Lj))
                    rem = any(re.search(r'\b' + re.escape(var) + r'\b.*==\s*0(\.0*)?f?\s*\)?\s*\?\s*0', lines[k])
                              or re.search(r'[Pp]lusZero\w*\s*\(\s*' + re.escape(var) + r'\b', lines[k])
                              for k in range(j, min(len(lines), j + 12)))
                    results.append({'file': f, 'zeroLine': ln, 'updateLine': j + 1, 'var': var,
                                    'kind': 'loop' if loopish else 'straight',
                                    'zeroCode': rawlines[i].strip()[:160], 'updateCode': rawlines[j].strip()[:160],
                                    'remedied': rem})
                    break
                if reassign.search(Lj) and not upd.search(Lj):
                    break  # overwritten before any add
        for m in LIT.finditer(L):
            results.append({'file': f, 'zeroLine': ln, 'updateLine': ln, 'var': '', 'kind': 'literal-add',
                            'zeroCode': rawlines[i].strip()[:160], 'updateCode': '', 'remedied': False})

for r in results:
    print(json.dumps(r))
