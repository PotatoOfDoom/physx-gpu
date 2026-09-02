#!/usr/bin/env python3
# check_component_divide.py -- P9.45.  The SIXTH static checker.
#
# Flags every place in the built SPIR-V where the COMPONENTS OF ONE VECTOR are
# each divided by the SAME scalar, written out one component at a time.
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS, given that P9.36 already added gpuslang.vectorDivide
# ---------------------------------------------------------------------------
# P9.36's rule is: flag an OpFDiv whose RESULT TYPE is a vector and whose
# divisor is a splat of one scalar.  That is the signature of `v / m` written
# as a vector expression, and it swept about sixty sites out of the corpus.
#
# It cannot see the same defect written componentwise:
#
#     v = float3(v.x / m, v.y / m, v.z / m);       // three SCALAR OpFDivs
#     right.x = right.x / m; right.y = right.y / m; right.z = right.z / m;
#
# Those lower to scalar divides with scalar result types, so the vectorDivide
# rule skips them as "not this check's business" and the shader looks clean.
#
# Three such sites survived P9.36 in this corpus, all of them transcriptions of
# PxVec3::normalize():
#
#     pxgSlangFemRigidContact.slang  computeBasisVectors   (both branches)
#     psContactSolveBda.slang        normalize3
#     npArtiStep.slang               asNormalizeInPlace    (worldAngVel.normalize())
#
# The engine's normalize (PxVec3.h:293) scales through operator/= (PxVec3.h:260),
# which forms 1.0/m ONCE and multiplies three times.  Three correctly-rounded
# divides differ from that in at least one component for about half of all
# inputs -- the same 53.5% P9.36 measured.
#
# ---------------------------------------------------------------------------
# WHY NEITHER A FIXTURE NOR A MUTANT FOUND THEM
# ---------------------------------------------------------------------------
# The fixtures compare against a host mirror written from the same reading, and
# the mirrors here happened to be written the OTHER way -- so the shader and the
# mirror disagreed by one ULP, on inputs the fixtures did not contain.  The cloth
# rigid-contact fixture had exactly one contact normal for eight tranches, and for
# that one normal the two spellings agree in every component.  P9.45 widened it to
# six normals and the disagreement surfaced immediately.
#
# A mutant cannot find it either, for P9.36's reason: mutation asks whether the
# fixture notices a perturbation, and a fixture blind to the original is blind to
# the mutant.  The mutation sweep for this very tranche confirmed that twice --
# the three-divide mutant of the SECOND basis branch survived until a lane whose
# normal makes the two forms differ was searched for and added.
#
# ---------------------------------------------------------------------------
# THE RULE, and why it does not false-positive
# ---------------------------------------------------------------------------
# Flag a function when two or more SCALAR OpFDivs
#   * share the same divisor id, and
#   * take their dividends from OpCompositeExtract with DISTINCT component
#     indices.
#
# That second clause is what separates the defect from the faithful cases, and
# the corpus has plenty of the latter:
#
#   femClothPrimitives.cu:129-130   v = (d11*d20 - d01*d21) / denom;
#                                   w = (d00*d21 - d01*d20) / denom;
#   softBody.cu:558-569             v = detPcd / detBcd; w = detBpd / detBcd;
#                                   x = detBcp / detBcd;
#
# The engine really does divide two or three times by one scalar there -- but the
# dividends are independent dot/cross results, not the components of a vector, so
# they arrive as OpFSub / OpDot results and are not extracts.  Only a vector taken
# apart and put back together is flagged.
#
# ---------------------------------------------------------------------------
# WHAT THIS CHECK CANNOT SEE
# ---------------------------------------------------------------------------
# A normalisation whose components never live in one vector value -- three
# separate float locals divided by one scalar and never composited -- has no
# OpCompositeExtract to key on.  None exists in the corpus today.  As with every
# checker here, a green run means "no vector is taken apart and divided
# componentwise", not "the arithmetic matches".
#
# ---------------------------------------------------------------------------
# ALLOWLIST
# ---------------------------------------------------------------------------
# An entry is permitted only when the ENGINE at that site genuinely divides the
# components of one vector separately by the same scalar.  Each entry names the
# engine line that does it.
#
# Unlike gpuslang.vectorDivide, this list is NOT empty, and that is the finding
# rather than a weakness: the engine writes componentwise division out by hand in
# two places, and both are raw floats rather than PxVec3 arithmetic.
#
#   gridCal.cuh:38-40      calcGridPos divides x, y and z of a float4 by cellWidth
#                          in three statements.  It is transcribed in SEVEN shaders
#                          (the P9.38 consolidation has not reached it), and each
#                          copy is faithful.  The quotients go straight into
#                          floor() and become an int3, so they never reassemble a
#                          float vector.
#   CmConeLimitHelper.h:173  PxTan(clamped.y/4.0f), PxTan(clamped.z/4.0f) -- two
#                          divides by the literal 4, which is a power of two, so
#                          the two spellings are bit-identical here in any case.
#                          d6SolveQ inlines the helper twice, hence a count of 2.
#
# Everything else the first run reported was one of the three normalisations
# above, and all three were defects.

import os
import struct
import sys
from collections import defaultdict

SPV_DIR = os.environ.get("PXG_SHADER_DIR")
if not SPV_DIR:
    here = os.path.dirname(os.path.abspath(__file__))
    SPV_DIR = os.path.join(here, "..", "..", "..", "..", "build-gpuslang", "shaders")

OP_NAME = 5
OP_TYPE_FLOAT = 22
OP_FUNCTION = 54
OP_FUNCTION_END = 56
OP_COMPOSITE_EXTRACT = 81
OP_FDIV = 136

# shader stem -> {"count": n, "why": "..."}  (see ALLOWLIST above)
_GRID = {"count": 1, "why": "calcGridPos, gridCal.cuh:38-40 -- the engine divides "
                            "x/y/z by cellWidth in three statements on raw floats; "
                            "the quotients go into floor() and become an int3"}
ALLOWED = {
    "gridCal": dict(_GRID),
    "particleGridBda": dict(_GRID),
    "particleSelfCollisionBda": dict(_GRID),
    "psBounds": dict(_GRID),
    "psBoundsBda": dict(_GRID),
    "psContactGenBda": dict(_GRID),
    "psDiffuseAdvect": dict(_GRID, why=_GRID["why"] + " (spelled psdaGridPos here)"),
    "npD6JointPrep": {"count": 2,
                      "why": "coneGetLimitQ, CmConeLimitHelper.h:173 -- PxTan(clamped.y/4) "
                             "and PxTan(clamped.z/4), two divides by the literal 4 written "
                             "out in the engine; 4 is a power of two, so the reciprocal "
                             "form would be bit-identical anyway.  d6SolveQ inlines the "
                             "helper twice, hence 2."},
}


def parse(path):
    data = open(path, "rb").read()
    if len(data) < 20 or struct.unpack("<I", data[:4])[0] != 0x07230203:
        return None
    words = list(struct.unpack("<%dI" % (len(data) // 4), data))
    float_types = set()
    extracts = {}
    fdivs = []
    names = {}
    cur_fn = None
    i = 5
    while i < len(words):
        wc = words[i] >> 16
        op = words[i] & 0xFFFF
        if wc == 0:
            break
        if op == OP_NAME:
            raw = b"".join(struct.pack("<I", w) for w in words[i + 2 : i + wc])
            names[words[i + 1]] = raw.split(b"\0")[0].decode("utf-8", "replace")
        elif op == OP_TYPE_FLOAT:
            float_types.add(words[i + 1])
        elif op == OP_FUNCTION:
            cur_fn = words[i + 2]
        elif op == OP_FUNCTION_END:
            cur_fn = None
        elif op == OP_COMPOSITE_EXTRACT:
            extracts[words[i + 2]] = tuple(words[i + 4 : i + wc])   # the index chain
        elif op == OP_FDIV:
            fdivs.append((words[i + 1], words[i + 2], words[i + 3], words[i + 4], cur_fn))
        i += wc
    return float_types, extracts, fdivs, names


def group(float_types, extracts, fdivs):
    """(function, divisor) -> set of distinct component indices divided by it."""
    g = defaultdict(set)
    for rtype, _rid, a, b, fn in fdivs:
        if rtype not in float_types:
            continue                      # a vector divide is gpuslang.vectorDivide's job
        idx = extracts.get(a)
        if idx is None:
            continue                      # dividend is not a component of a vector
        g[(fn, b)].add(idx)
    return g


def offenders(path):
    parsed = parse(path)
    if parsed is None:
        return []
    float_types, extracts, fdivs, names = parsed
    out = []
    for (fn, _b), idxs in group(float_types, extracts, fdivs).items():
        if len(idxs) >= 2:
            out.append(names.get(fn, "fn%s" % fn))
    return out


def selftest():
    """Negative control.  A checker that cannot separate a componentwise vector
    divide from a shared denominator would pass everything silently -- which is
    exactly how the three sites this file exists for slipped past the P9.36 rule."""
    float_types = {9}
    # %40,%41,%42 extract components 0,1,2 of one vector; %50 is an OpFSub result.
    extracts = {40: (0,), 41: (1,), 42: (2,)}
    defect = [(9, 60, 40, 7, 1), (9, 61, 41, 7, 1), (9, 62, 42, 7, 1)]
    faithful = [(9, 63, 50, 7, 1), (9, 64, 51, 7, 1)]           # no extracts
    one_comp = [(9, 65, 40, 7, 1), (9, 66, 40, 7, 1)]           # same index twice
    vector = [(10, 67, 40, 7, 1), (10, 68, 41, 7, 1)]           # vector result type

    def run(fd):
        return sum(1 for _k, v in group(float_types, extracts, fd).items() if len(v) >= 2)

    ok = True
    if run(defect) != 1:
        print("SELFTEST FAIL: a componentwise vector divide was not flagged")
        ok = False
    if run(faithful) != 0:
        print("SELFTEST FAIL: a shared denominator over independent dividends WAS "
              "flagged -- femClothPrimitives.cu:129 would be a false positive")
        ok = False
    if run(one_comp) != 0:
        print("SELFTEST FAIL: the same component divided twice WAS flagged")
        ok = False
    if run(vector) != 0:
        print("SELFTEST FAIL: a vector-result divide was flagged -- that is "
              "gpuslang.vectorDivide's business, not this one's")
        ok = False
    return ok


def main():
    if not selftest():
        print("the checker's own control failed; its verdicts mean nothing")
        return 1
    if not os.path.isdir(SPV_DIR):
        print("no shader directory at %s -- set PXG_SHADER_DIR" % SPV_DIR)
        return 1

    bad = []
    scanned = 0
    for name in sorted(os.listdir(SPV_DIR)):
        if not name.endswith(".spv"):
            continue
        stem = name[:-4]
        scanned += 1
        found = offenders(os.path.join(SPV_DIR, name))
        allowed = ALLOWED.get(stem, {}).get("count", 0)
        if len(found) > allowed:
            bad.append((stem, found, allowed))

    if bad:
        print("THE COMPONENTS OF A VECTOR ARE BEING DIVIDED ONE AT A TIME BY THE SAME")
        print("SCALAR.  In the engine that expression is a reciprocal formed ONCE and")
        print("three multiplies (PxVec3.h:219/:260/:295); componentwise divides differ")
        print("from it for about half of all inputs.  gpuslang.vectorDivide cannot see")
        print("this spelling because each OpFDiv has a SCALAR result type.")
        print("Write  const float r = 1.0f / m;  and multiply -- or add an allowlist")
        print("entry naming the engine line that really divides each component.")
        print()
        for stem, found, allowed in bad:
            print("  %-34s %d componentwise divide(s), %d allowed  in: %s"
                  % (stem, len(found), allowed, ", ".join(sorted(set(found)))))
        return 1

    print("componentDivide: %d shaders scanned, no componentwise vector divides "
          "(%d allowlisted)" % (scanned, sum(v.get("count", 0) for v in ALLOWED.values())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
