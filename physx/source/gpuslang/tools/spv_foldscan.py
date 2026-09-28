#!/usr/bin/env python3
"""P9.86f: find sign-of-zero fold hazards in compiled (-O0 -g2) SPIR-V.

For every function reachable from the given entry points, report
  F2  OpFSub(a, b) where a may be a LITERAL +0 (scalar or a lane)
  F3  OpFAdd(a, b) where a or b may be a literal +0 (scalar or a lane)
  F4  GLSL.std.450 FMin/FMax/FClamp/NMin/NMax on a float type
"may be a literal +0" is a flow-insensitive fixpoint over: constants, OpConstantNull,
composite construct/extract/insert/shuffle, copy/select/phi, loads of Function/Private
variables (union of everything stored under the same access-chain key), function
parameters (union over call-site arguments), call results (union of the callee's
OpReturnValue), and stores through pointer parameters into the caller's variable.
Each hit is attributed to the source file:line of the preceding DebugLine and tagged
with how the zero arrives (const / lane / var / param / ret).
Usage: spv_foldscan.py <spv> <entry,entry,...>   -> JSON lines on stdout
"""
import struct, sys, json, collections

OP_NAME, OP_STRING, OP_EXTIMPORT, OP_EXTINST, OP_ENTRY = 5, 7, 11, 12, 15
OP_TBOOL, OP_TINT, OP_TFLOAT, OP_TVEC, OP_TARR, OP_TRARR, OP_TSTRUCT, OP_TPTR = 20, 21, 22, 23, 28, 29, 30, 32
OP_CTRUE, OP_CFALSE, OP_CONST, OP_CCOMP, OP_CNULL = 41, 42, 43, 44, 46
OP_FUNC, OP_FPARAM, OP_FEND, OP_FCALL = 54, 55, 56, 57
OP_VAR, OP_LOAD, OP_STORE, OP_AC, OP_IBAC, OP_PTRAC = 59, 61, 62, 65, 66, 67
OP_SHUF, OP_CCONS, OP_CEXT, OP_CINS, OP_COPY = 79, 80, 81, 82, 83
OP_BITCAST, OP_FADD, OP_FSUB = 124, 129, 131
OP_SELECT, OP_PHI, OP_RETV = 169, 245, 254
GLSL_MINMAX = {37: 'FMin', 40: 'FMax', 43: 'FClamp', 79: 'NMin', 80: 'NMax', 81: 'NClamp'}
DBG_SOURCE, DBG_LINE, DBG_FUNCTION = 35, 103, 20

ALL = frozenset(['*'])  # "some lane / field may be zero" for aggregates


def lit(words):
    return b''.join(struct.pack('<I', x) for x in words).split(b'\0')[0].decode(errors='replace')


def main(spvpath, entries):
    d = open(spvpath, "rb").read()
    w = struct.unpack('<%dI' % (len(d) // 4), d)
    insts = []
    i = 5
    while i < len(w):
        op = w[i] & 0xffff
        n = w[i] >> 16
        insts.append((op, w[i + 1:i + n]))
        i += n

    names, strings, ext, types, consts = {}, {}, {}, {}, {}
    typeof = {}
    entry_ids = []
    dbg_src = {}
    funcs = []  # list of (fid, [insts])
    cur = None
    for op, a in insts:
        if op == OP_NAME:
            names[a[0]] = lit(a[1:])
        elif op == OP_STRING:
            strings[a[0]] = lit(a[1:])
        elif op == OP_EXTIMPORT:
            ext[a[0]] = lit(a[1:])
        elif op == OP_ENTRY:
            entry_ids.append((a[1], lit(a[2:])))
        elif op in (OP_TBOOL, OP_TINT, OP_TFLOAT):
            types[a[0]] = ('scalar', op)
        elif op == OP_TVEC:
            types[a[0]] = ('vec', a[1], a[2])
        elif op in (OP_TARR, OP_TRARR, OP_TSTRUCT):
            types[a[0]] = ('agg',)
        elif op == OP_TPTR:
            types[a[0]] = ('ptr', a[1], a[2])
        elif op in (OP_CONST, OP_CCOMP, OP_CNULL, OP_CTRUE, OP_CFALSE):
            typeof[a[1]] = a[0]
            consts[a[1]] = (op, a[2:])
        elif op == OP_FUNC:
            cur = [a[1], []]
            typeof[a[1]] = a[0]
            funcs.append(cur)
        elif op == OP_FEND:
            cur = None
        elif cur is not None:
            cur[1].append((op, a))
        if op == OP_EXTINST and cur is None and ext.get(a[2], '').startswith('NonSemantic.Shader.DebugInfo'):
            if a[3] == DBG_SOURCE:
                dbg_src[a[1]] = strings.get(a[4], '?')
    fdict = {f[0]: f[1] for f in funcs}

    def is_float(t):
        tt = types.get(t)
        if not tt:
            return False
        if tt[0] == 'scalar':
            return tt[1] == OP_TFLOAT
        if tt[0] == 'vec':
            return is_float(tt[1])
        return False

    def nlanes(t):
        tt = types.get(t)
        return tt[2] if tt and tt[0] == 'vec' else 1

    def const_int(cid):
        c = consts.get(cid)
        return c[1][0] if c and c[0] == OP_CONST else None

    # ---- initial zero sets from constants ----
    Z = {}  # id -> frozenset of lanes ({0} for a scalar) or ALL for aggregates
    SRC = collections.defaultdict(set)  # id -> tags how the zero arrives

    def const_zero(cid):
        c = consts.get(cid)
        if c is None:
            return frozenset()
        op, ops = c
        t = typeof.get(cid)
        tt = types.get(t, ('?',))
        if op == OP_CNULL:
            return frozenset(range(nlanes(t))) if tt[0] in ('vec', 'scalar') else ALL
        if op == OP_CONST:
            return frozenset([0]) if tt[0] == 'scalar' and tt[1] == OP_TFLOAT and ops[0] == 0 else frozenset()
        if op == OP_CCOMP:
            if tt[0] == 'vec':
                return frozenset(k for k, e in enumerate(ops) if const_zero(e))
            return ALL if any(const_zero(e) for e in ops) else frozenset()
        return frozenset()

    for cid in consts:
        z = const_zero(cid)
        if z:
            Z[cid] = z
            SRC[cid].add('const' if typeof.get(cid) and types.get(typeof[cid], ('?',))[0] == 'scalar' else 'lane')

    # ---- pointer keys: (base var, tuple of const indices or '?') ----
    ptrkey = {}
    params = {}  # fid -> [param ids]
    for fid, body in fdict.items():
        params[fid] = [a[1] for op, a in body if op == OP_FPARAM]
        for op, a in body:
            if op in (OP_FPARAM,):
                typeof[a[1]] = a[0]
                ptrkey[a[1]] = (a[1], ())
            elif op == OP_VAR:
                typeof[a[1]] = a[0]
                ptrkey[a[1]] = (a[1], ())
    for op, a in insts:
        if op == OP_VAR:
            typeof[a[1]] = a[0]
            ptrkey.setdefault(a[1], (a[1], ()))
    changed = True
    while changed:
        changed = False
        for fid, body in fdict.items():
            for op, a in body:
                if op in (OP_AC, OP_IBAC, OP_PTRAC) and a[2] in ptrkey and a[1] not in ptrkey:
                    base, path = ptrkey[a[2]]
                    idx = a[3:] if op != OP_PTRAC else a[4:]
                    path2 = path + tuple(const_int(x) if const_int(x) is not None else '?' for x in idx)
                    ptrkey[a[1]] = (base, path2)
                    typeof[a[1]] = a[0]
                    changed = True
                elif op == OP_COPY and a[2] in ptrkey and a[1] not in ptrkey:
                    ptrkey[a[1]] = ptrkey[a[2]]
                    changed = True

    stores = collections.defaultdict(lambda: collections.defaultdict(frozenset))  # base -> path -> zset
    alias = collections.defaultdict(set)  # param-pointer base -> caller bases (pointer args)
    calls = collections.defaultdict(list)  # callee -> [(caller fid, result id, args)]
    rets = collections.defaultdict(set)
    for fid, body in fdict.items():
        for op, a in body:
            if op == OP_FCALL:
                calls[a[2]].append((fid, a[1], a[3:]))
                typeof[a[1]] = a[0]
            elif op == OP_RETV:
                rets[fid].add(a[0])
            elif len(a) >= 2 and op not in (OP_STORE, OP_EXTINST) and a[0] in types:
                typeof.setdefault(a[1], a[0])
    for callee, sites in calls.items():
        for (_, _, args) in sites:
            for p, arg in zip(params.get(callee, []), args):
                if arg in ptrkey:
                    alias[p].add(ptrkey[arg])

    def lanes_of_path(zset_by_path, path):
        """zero lanes seen by a load of `path` from a variable's store map"""
        out = set()
        for spath, zs in zset_by_path.items():
            if not zs:
                continue
            if spath == path:
                out |= zs
            elif len(spath) > len(path) and (spath[:len(path)] == path or '?' in spath[:len(path)] or '?' in path):
                # a store into a sub-element of what we load
                rest = spath[len(path):]
                if len(rest) == 1 and isinstance(rest[0], int):
                    out.add(rest[0])
                else:
                    out |= {'*'}
            elif len(spath) < len(path) and (path[:len(spath)] == spath or '?' in path or '?' in spath):
                # a store of a larger aggregate that contains what we load
                if '*' in zs:
                    out |= {'*'}
                elif len(path) - len(spath) == 1 and isinstance(path[-1], int) and path[-1] in zs:
                    out.add(0)
        return frozenset(out)

    def zget(x):
        return Z.get(x, frozenset())

    def zset(x, v, tags):
        v = frozenset(v)
        if not v:
            return False
        old = Z.get(x, frozenset())
        new = old | v
        ch = False
        if new != old:
            Z[x] = new
            ch = True
        oldt = len(SRC[x])
        SRC[x] |= tags
        return ch or len(SRC[x]) != oldt

    ORIG = collections.defaultdict(set)  # id or ('var', base) -> {(file, line)} of the zero's origin

    def const_loc(cid, loc):
        # a literal zero is "born" where it is used
        return {loc} if cid in consts else set()

    it = 0
    changed = True
    while changed and it < 60:
        it += 1
        changed = False
        for fid, body in fdict.items():
            loc = ('?', 0)
            for op, a in body:
                if op == OP_EXTINST and ext.get(a[2], '').startswith('NonSemantic.Shader.DebugInfo'):
                    if a[3] == DBG_LINE:
                        loc = (dbg_src.get(a[4], '?').split('/')[-1], const_int(a[5]) or 0)
                    continue
                if op == OP_STORE:
                    p, v = a[0], a[1]
                    if p in ptrkey and zget(v):
                        base, path = ptrkey[p]
                        targets = [(base, path)] + [(b2, p2 + path) for (b2, p2) in alias.get(base, ())]
                        for (b, pth) in targets:
                            old = stores[b][pth]
                            nz = old | zget(v)
                            if nz != old:
                                stores[b][pth] = nz
                                changed = True
                            if not SRC[('var', b)] >= SRC[v]:
                                SRC[('var', b)] |= SRC[v]
                            o = (ORIG[v] or set()) | const_loc(v, loc)
                            if not ORIG[('var', b)] >= o:
                                ORIG[('var', b)] |= o
                                changed = True
                elif op == OP_LOAD:
                    r, p = a[1], a[2]
                    if p in ptrkey:
                        base, path = ptrkey[p]
                        zl = lanes_of_path(stores[base], path)
                        tags = {'var'} | {t for t in SRC[('var', base)] if t in ('param', 'ret')}
                        if base in params.get(fid, []) and not stores[base]:
                            pass
                        changed |= zset(r, zl, tags)
                        if zl and not ORIG[r] >= ORIG[('var', base)]:
                            ORIG[r] |= ORIG[('var', base)]
                            changed = True
                elif op == OP_FPARAM:
                    pass
                elif op == OP_CCONS:
                    r = a[1]
                    lanes, k = set(), 0
                    tags = set()
                    for c in a[2:]:
                        if zget(c):
                            ORIG[r] |= ORIG[c] | const_loc(c, loc)
                        n = nlanes(typeof.get(c, 0)) if types.get(typeof.get(c, 0), ('?',))[0] in ('vec', 'scalar') else 1
                        zc = zget(c)
                        for j in range(n):
                            if j in zc or '*' in zc:
                                lanes.add(k + j)
                        if zc:
                            tags |= SRC[c] | {'lane'}
                        k += n
                    tt = types.get(a[0], ('?',))
                    if tt[0] != 'vec' and lanes:
                        lanes = {'*'}
                    changed |= zset(r, lanes, tags)
                elif op == OP_CEXT:
                    r, c, idx = a[1], a[2], a[3:]
                    zc = zget(c)
                    if zc and ('*' in zc or (len(idx) == 1 and idx[0] in zc)):
                        ORIG[r] |= ORIG[c]
                        tt = types.get(a[0], ('?',))
                        changed |= zset(r, {0} if tt[0] == 'scalar' else {'*'} if tt[0] == 'agg' else set(range(nlanes(a[0]))), SRC[c])
                elif op == OP_CINS:
                    r, obj, comp, idx = a[1], a[2], a[3], a[4:]
                    zl = set(zget(comp) - ({idx[0]} if len(idx) == 1 else set()))
                    if zget(obj) and len(idx) == 1:
                        zl.add(idx[0])
                    elif zget(obj):
                        zl.add('*')
                    changed |= zset(r, zl, SRC[comp] | SRC[obj])
                elif op == OP_SHUF:
                    r, v1, v2, comps = a[1], a[2], a[3], a[4:]
                    n1 = nlanes(typeof.get(v1, 0))
                    zl = set()
                    for k, c in enumerate(comps):
                        if c == 0xFFFFFFFF:
                            continue
                        src, j = (v1, c) if c < n1 else (v2, c - n1)
                        if j in zget(src) or '*' in zget(src):
                            zl.add(k)
                    changed |= zset(r, zl, SRC[v1] | SRC[v2])
                elif op in (OP_COPY,):
                    changed |= zset(a[1], zget(a[2]), SRC[a[2]])
                    ORIG[a[1]] |= ORIG[a[2]]
                elif op == OP_SELECT:
                    changed |= zset(a[1], zget(a[3]) | zget(a[4]), SRC[a[3]] | SRC[a[4]])
                elif op == OP_PHI:
                    vs = a[2::2]
                    changed |= zset(a[1], frozenset().union(*[zget(v) for v in vs]), set().union(*[SRC[v] for v in vs]))
                elif op == OP_FCALL:
                    r, callee, args = a[1], a[2], a[3:]
                    for p, arg in zip(params.get(callee, []), args):
                        if arg not in ptrkey:
                            changed |= zset(p, zget(arg), SRC[arg] | {'param'})
                            if zget(arg):
                                o = ORIG[arg] | const_loc(arg, loc)
                                if not ORIG[p] >= o:
                                    ORIG[p] |= o
                                    changed = True
                    rz = frozenset().union(*[zget(x) for x in rets.get(callee, ())]) if rets.get(callee) else frozenset()
                    changed |= zset(r, rz, set().union(*[SRC[x] for x in rets.get(callee, ())]) | {'ret'} if rz else set())
                    if rz:
                        ORIG[r] |= set().union(*[ORIG[x] for x in rets.get(callee, ())])

    # ---- reachability from the requested entry points ----
    want = set(entries)
    roots = [fid for fid, nm in entry_ids if nm in want]
    missing = want - {nm for _, nm in entry_ids}
    callg = collections.defaultdict(set)
    for fid, body in fdict.items():
        for op, a in body:
            if op == OP_FCALL:
                callg[fid].add(a[2])
    reach, stack = set(), list(roots)
    while stack:
        f = stack.pop()
        if f in reach:
            continue
        reach.add(f)
        stack.extend(callg[f])

    def dbgname(fid):
        return names.get(fid, '%%%d' % fid)

    hits = []
    for fid in reach:
        body = fdict[fid]
        loc = ('?', 0)
        for op, a in body:
            if op == OP_EXTINST and ext.get(a[2], '').startswith('NonSemantic.Shader.DebugInfo'):
                if a[3] == DBG_LINE:
                    src = dbg_src.get(a[4], '?')
                    ln = const_int(a[5]) or 0
                    loc = (src, ln)
                continue
            kind = None
            if op in (OP_FSUB, OP_FADD) and a[2] in consts and a[3] in consts:
                continue  # constant (op) constant: folded exactly
            if op == OP_FSUB and is_float(a[0]) and zget(a[2]):
                kind, zs, operand = 'F2-zero-minus', zget(a[2]), a[2]
            elif op == OP_FADD and is_float(a[0]) and (zget(a[2]) or zget(a[3])):
                operand = a[2] if zget(a[2]) else a[3]
                kind, zs = 'F3-zero-add', zget(operand)
            elif op == OP_EXTINST and ext.get(a[2], '') == 'GLSL.std.450' and a[3] in GLSL_MINMAX and is_float(a[0]):
                kind, zs, operand = 'F4-' + GLSL_MINMAX[a[3]], frozenset(), None
            if kind:
                tags = sorted(SRC[operand]) if operand is not None else []
                orig = sorted('%s:%d' % o for o in (ORIG[operand] | const_loc(operand, (loc[0].split('/')[-1], loc[1])))) if operand is not None else []
                hits.append({'orig': orig, 'file': loc[0].split('/')[-1], 'line': loc[1], 'kind': kind,
                             'lanes': sorted(str(x) for x in zs), 'via': tags, 'func': dbgname(fid)})
    # reached source ranges: (file, min line, max line) per reached function
    ranges = []
    for fid in reach:
        per = {}
        for op, a in fdict[fid]:
            if op == OP_EXTINST and ext.get(a[2], '').startswith('NonSemantic.Shader.DebugInfo') and a[3] == DBG_LINE:
                f = dbg_src.get(a[4], '?').split('/')[-1]
                ln = const_int(a[5]) or 0
                if ln:
                    lo, hi = per.get(f, (ln, ln))
                    per[f] = (min(lo, ln), max(hi, ln))
        for f, (lo, hi) in per.items():
            ranges.append({'range': f, 'lo': lo, 'hi': hi, 'func': names.get(fid, '%%%d' % fid)})
    for r in ranges:
        print(json.dumps(r))
    out = {"spv": spvpath.split('/')[-1], 'roots': len(roots), 'missing': sorted(missing), 'reach': len(reach), 'iters': it}
    print(json.dumps(out))
    for h in hits:
        print(json.dumps(h))


if __name__ == '__main__':
    main(sys.argv[1], [e for e in sys.argv[2].split(',') if e])
