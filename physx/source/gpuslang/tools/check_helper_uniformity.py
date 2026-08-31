#!/usr/bin/env python3
"""check_helper_uniformity.py -- P9.37.

Gate: a curated set of replicated engine helpers must compile to exactly ONE
arithmetic DAG across the whole corpus.  Clusters same-named functions in the built
SPIR-V by the expression graph they actually compute.

Source-token clustering (what P9.37 started with) splits on naming, statement
decomposition and literal suffixes -- every shape it flagged by hand turned out to
be a false split.  The compiler has already inlined single-use temporaries by the
time it emits SPIR-V, so comparing the emitted arithmetic answers the real question
-- "do these copies compute the same thing, in the same order?" -- without any of
that noise.

For each function we emit a canonical string: instructions in order, opcodes by
number, operands renumbered by first appearance within the function, and constants
resolved to their literal bit patterns (so 2.0f and 0.5f cannot collide).  Two
functions with the same string perform the same operations on the same operands in
the same order.  A DIFFERENT string is not automatically a defect -- inlining
context and surrounding code can legitimately differ -- but identical strings are
proof of identical arithmetic, which is what an audit for association errors needs.
"""

import collections
import hashlib
import os
import struct
import sys

OP_NAME = 5
OP_TYPE_VOID = 19
OP_TYPE_BOOL = 20
OP_TYPE_INT = 21
OP_TYPE_FLOAT = 22
OP_TYPE_VECTOR = 23
OP_CONSTANT = 43
OP_CONSTANT_COMPOSITE = 44
OP_CONSTANT_NULL = 46
OP_CONSTANT_TRUE = 41
OP_CONSTANT_FALSE = 42
OP_FUNCTION = 54
OP_FUNCTION_PARAMETER = 55
OP_FUNCTION_END = 56
OP_LABEL = 248

# Instructions that carry arithmetic meaning.  Loads/stores/labels/branches are
# excluded: they encode how the compiler happened to schedule things, not what is
# computed.
ARITH = {
    65: "ConvertFToU", 66: "ConvertFToS", 79: "VectorShuffle", 80: "CompositeConstruct",
    81: "CompositeExtract", 82: "CompositeInsert",
    127: "Negate", 128: "IAdd", 129: "FAdd", 130: "ISub", 131: "FSub",
    132: "IMul", 133: "FMul", 134: "UDiv", 135: "SDiv", 136: "FDiv",
    137: "UMod", 139: "FRem", 141: "FMod",
    142: "VectorTimesScalar", 148: "Dot",
    169: "FOrdEqual", 180: "FOrdLessThan", 184: "FOrdGreaterThan",
    186: "FOrdLessThanEqual", 190: "FOrdGreaterThanEqual",
    12: "ExtInst",
}


def parse(path):
    data = open(path, "rb").read()
    if len(data) < 20 or struct.unpack("<I", data[:4])[0] != 0x07230203:
        return None, None
    words = list(struct.unpack("<%dI" % (len(data) // 4), data))
    names, consts, fns = {}, {}, {}
    cur, body, params = None, None, []
    i = 5
    while i < len(words):
        wc = words[i] >> 16
        op = words[i] & 0xFFFF
        if wc == 0:
            break
        if op == OP_NAME:
            raw = b"".join(struct.pack("<I", w) for w in words[i + 2:i + wc])
            names[words[i + 1]] = raw.split(b"\0")[0].decode("utf-8", "replace")
        elif op == OP_CONSTANT and wc >= 4:
            consts[words[i + 2]] = words[i + 3]          # raw bit pattern
        elif op == OP_CONSTANT_NULL and wc >= 3:
            # zero is emitted as OpConstantNull by some modules and OpConstant by
            # others; without this the SAME source hashed differently per module.
            consts[words[i + 2]] = 0
        elif op == OP_CONSTANT_FALSE and wc >= 3:
            consts[words[i + 2]] = 0
        elif op == OP_CONSTANT_TRUE and wc >= 3:
            consts[words[i + 2]] = 1
        elif op == OP_FUNCTION:
            cur, body, params = words[i + 2], [], []
        elif op == OP_FUNCTION_PARAMETER and cur is not None:
            params.append(words[i + 2])
        elif op == OP_FUNCTION_END:
            if cur is not None:
                fns[cur] = (body, params)
            cur, body, params = None, None, []
        elif cur is not None and op in ARITH:
            body.append((op, words[i + 1] if wc > 1 else 0, list(words[i + 3:i + wc])))
        i += wc
    return names, (fns, consts)


def canon(entry, consts):
    """Order-INSENSITIVE canonical form: hash each value's expression subtree.

    An earlier version concatenated instructions in emission order, which split
    functions that differ only in how the compiler SCHEDULED two independent
    multiplies -- a false positive, since each multiply rounds its own operands
    regardless of when it is emitted.  Hashing the DAG rooted at each value removes
    scheduling entirely: two functions computing the same expressions from the same
    inputs produce the same set of subtree hashes however the instructions are laid
    out.  Leaves are canonical too -- constants by bit pattern, parameters by
    POSITION in the signature rather than by first appearance.
    """
    body, params = entry
    defs = {rid: (op, operands) for op, rid, operands in body if rid}
    leaf = {}
    for n, pid in enumerate(params):
        leaf[pid] = "p%d" % n
    for cid, bits in consts.items():
        leaf[cid] = "c%08x" % bits

    memo = {}
    stack_guard = set()

    def h(vid):
        if vid in leaf:
            return leaf[vid]
        if vid in memo:
            return memo[vid]
        if vid not in defs or vid in stack_guard:
            return "x"                      # external or cyclic (phi): opaque leaf
        stack_guard.add(vid)
        op, operands = defs[vid]
        if ARITH[op] == "ExtInst":
            # operands are [set-id, instruction-number, args...].  The set id is this
            # module's OpExtInstImport result and differs between modules for the same
            # call, so it is dropped; the instruction number is a LITERAL (32 =
            # InverseSqrt) and must be kept verbatim rather than resolved as an id --
            # doing the latter made the same source hash differently per module,
            # because the integer happened to name a constant in one module only.
            r = "ExtInst#%d(%s)" % (operands[1], ",".join(h(o) for o in operands[2:]))
        else:
            r = "%s(%s)" % (ARITH[op], ",".join(h(o) for o in operands))
        stack_guard.discard(vid)
        r = hashlib.md5(r.encode()).hexdigest()[:12]
        memo[vid] = r
        return r

    return " ".join(sorted(h(rid) for _op, rid, _o in body if rid))


# ---------------------------------------------------------------------------
# THE GATE
# ---------------------------------------------------------------------------
# Each entry is an engine helper this corpus writes out by hand many times over.
# Copies are the risk: P9.36 found one engine block ported FOUR times with one copy
# wrong, and P9.37 found `quatRotateInv` written 63 times with ONE copy carrying an
# algebraically-equivalent but bitwise-different form -- handed to the softbody
# plane path, whose result gates contact existence and is written out as the
# penetration the solver consumes.  Nothing in the suite noticed: the fixture's
# exact-arithmetic geometry makes the two forms agree.
#
# The value is the count as much as the uniformity.  If a name's copies drop below
# the recorded minimum, either a call site was rewritten or the name changed, and
# either way this table has stopped describing the corpus.
REQUIRE_UNIFORM = {
    "quatRotate": 69, "quatRotateInv": 62, "quatRotQ": 13, "quatRotInvQ": 10,
    "closestPtPointTriangle2": 19, "closestPtTetrahedron": 7, "closestPtLineLine": 4,
    "closestPtPointTriangle": 5, "closestPtPointBaryTriangle": 2,
    "extractRotation2D": 5, "getNormalized3": 4,
}

# Names that legitimately carry MORE than one shape, each with the reason.  These
# are deliberately NOT gated -- listing them here is what stops someone adding them
# to REQUIRE_UNIFORM later and "fixing" a difference that is correct.
KNOWN_MULTI = {
    "getNormalized":
        "three accepted spellings: v * rsqrt(m), v * (1.0f/sqrt(m)) -- P9.33a measured "
        "SPIR-V InverseSqrt against 1.0f/sqrtf over 258442 inputs and they agree bit for "
        "bit on this platform -- and npCvxTriWorldNormal's explicit dotX(v,v) instead of "
        "the dot intrinsic, which is what magnitudeSquared() literally is.",
    "getNormalizedQ": "same three spellings as getNormalized.",
    "computeBarycentric":
        "with and without the eps acceptance predicate.  Several call sites discard the "
        "engine's bool (fcStaticTriCloth.slang:214 says so), and those copies return the "
        "float4 alone; the barycentric arithmetic itself is identical.",
    "computeTetBarycentric":
        "as computeBarycentric, plus out-parameter versus returned float4.",
    "closestPointTetW": "warm-start variants carry an extra cached-simplex argument.",
    "closestPtEdgeEdge": "two callers with different degenerate-case contracts.",
    "closestPtPointEdge": "same.",
    "distancePointTriangleSquared":
        "the particle and convex families differ in their prologues; all groups carry "
        "the same barycentric tail, which is the part P9.36 corrected.",
    "byteAt": "different signatures entirely -- a BDA address versus an in-register word array.",
    "mat33FromQuat":
        "identical arithmetic; npSbExtractRotAPD returns three out-parameters where "
        "pxgSlangCommon returns a PxgMat33.  Worth consolidating, not a defect.",
    "pxgShflSyncU": "npWarpNative is the native-subgroup path against the emulation path (P2.1).",
    "pxgBallotSync": "as pxgShflSyncU.",
    "pxgShflXorSyncU": "as pxgShflSyncU.",
    "pxgShflUpSyncU": "as pxgShflSyncU.",
}


def selftest():
    """Negative control for the two properties this gate rests on.

    The screen would be worthless in either direction: one that split on emission
    order would flag every shader (an earlier version of this file did exactly that,
    and reported closestPtPointTriangle2 as two shapes when the only difference was
    where the compiler placed one independent FMul); one that ignored association
    would pass the defect it exists to catch.  Both are checked here rather than
    assumed.
    """
    FADD, FMUL = 129, 133
    p0, p1, p2 = 90, 91, 92          # parameters
    consts = {}

    def entry(body):
        return (body, [p0, p1, p2])

    # (a + b) + c   -- two spellings, differing ONLY in the order the two
    # independent instructions are emitted.  Must hash the SAME.
    ab = [(FMUL, 10, [p0, p1]), (FMUL, 11, [p0, p2]), (FADD, 12, [10, 11])]
    ba = [(FMUL, 11, [p0, p2]), (FMUL, 10, [p0, p1]), (FADD, 12, [10, 11])]
    if canon(entry(ab), consts) != canon(entry(ba), consts):
        print("SELFTEST FAIL: instruction ORDER changed the canonical form; the gate "
              "would flag scheduling differences as arithmetic ones")
        return False

    # (a*b)*c  versus  a*(b*c) -- the same operators in a different ASSOCIATION.
    # Must hash DIFFERENTLY: that is the defect class this gate is for.
    left = [(FMUL, 10, [p0, p1]), (FMUL, 11, [10, p2])]
    right = [(FMUL, 10, [p1, p2]), (FMUL, 11, [p0, 10])]
    if canon(entry(left), consts) == canon(entry(right), consts):
        print("SELFTEST FAIL: two different ASSOCIATIONS hashed the same; the gate "
              "cannot see the thing it exists to catch")
        return False

    return True


def main():
    if not selftest():
        print("the gate's own control failed; its verdicts mean nothing")
        return 1
    spv = os.environ.get("PXG_SHADER_DIR")
    if not spv or not os.path.isdir(spv):
        print("set PXG_SHADER_DIR")
        return 1
    targets = set(sys.argv[1:])
    byname = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in sorted(os.listdir(spv)):
        if not f.endswith(".spv"):
            continue
        names, parsed = parse(os.path.join(spv, f))
        if parsed is None:
            continue
        fns, consts = parsed
        for fid, entry in fns.items():
            nm = names.get(fid)
            if not nm or not entry[0]:
                continue
            if targets and not any(t in nm for t in targets):
                continue
            h = hashlib.md5(canon(entry, consts).encode()).hexdigest()[:8]
            byname[nm][h].append(f[:-4])

    bad = []
    for nm, want in sorted(REQUIRE_UNIFORM.items()):
        shapes = byname.get(nm)
        if not shapes:
            bad.append("%s: not found in any shader (expected %d copies)" % (nm, want))
            continue
        total = sum(len(v) for v in shapes.values())
        if len(shapes) > 1:
            detail = "; ".join(
                "%s x%d (%s)" % (h, len(v), ", ".join(sorted(v)[:4]))
                for h, v in sorted(shapes.items(), key=lambda kv: -len(kv[1])))
            bad.append("%s: %d copies compile to %d DIFFERENT arithmetic DAGs -- %s"
                       % (nm, total, len(shapes), detail))
        elif total < want:
            bad.append("%s: only %d copies, expected at least %d -- the table is stale"
                       % (nm, total, want))

    if bad:
        print("REPLICATED ENGINE HELPERS DISAGREE.")
        print("Each name below is written out by hand many times in this corpus and every")
        print("copy must compute the SAME arithmetic.  A copy that differs is either a")
        print("transcription slip or a deliberate variant -- if deliberate, move the name to")
        print("KNOWN_MULTI with the reason, do not delete it from the gate.")
        print()
        for b in bad:
            print("  " + b)
        return 1

    n = sum(sum(len(v) for v in byname[k].values()) for k in REQUIRE_UNIFORM if k in byname)
    print("helperUniformity: %d copies of %d gated helpers, one arithmetic DAG each "
          "(%d names known-multi)" % (n, len(REQUIRE_UNIFORM), len(KNOWN_MULTI)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
