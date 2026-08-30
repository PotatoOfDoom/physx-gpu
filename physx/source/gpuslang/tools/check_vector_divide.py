#!/usr/bin/env python3
# check_vector_divide.py -- P9.36.
#
# Flags every place in the built SPIR-V where a VECTOR is divided by a SPLAT OF A
# SCALAR, because in this port that is almost always a transcription defect.
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS
# ---------------------------------------------------------------------------
# PxVec3 has no three-divide form.  All three of its scalar-division spellings
# funnel into a reciprocal-then-multiply:
#
#   PxVec3.h:219  operator/(Type f)   { f = 1.0/f; return PxVec3(x*f, y*f, z*f); }
#   PxVec3.h:260  operator/=(Type f)  { f = 1.0/f; x *= f; y *= f; z *= f; }
#   PxVec3.h:295  normalize()         { m = magnitude(); if (m>0) *this /= m; ... }
#
# So `v / m`, `v /= m` and `normalize()` in the engine are ONE reciprocal and
# THREE MULTIPLIES.  Written in Slang as `v / m`, they become a genuine
# componentwise vector divide -- three correctly-rounded divisions -- and the two
# forms differ in at least one component for 53.5% of random vectors (measured,
# 3e6 samples).  Not a knife-edge: a coin flip.
#
# ---------------------------------------------------------------------------
# WHY A FIXTURE CANNOT SEE IT
# ---------------------------------------------------------------------------
# This is the same reason the other four static checkers exist.  The fixtures
# that cover these shaders compare bit-exactly against a HOST MIRROR, and the
# mirror is written by the same hand, from the same reading, at the same time --
# so it divides too.  A green suite then proves only that the shader and the
# mirror agree with each other.  P9.36 found six live sites this way, all of them
# under passing fixtures.
#
# A mutant cannot see it either: mutation testing perturbs the shader and asks
# whether the fixture notices, and a fixture blind to the original is blind to
# the mutant.
#
# ---------------------------------------------------------------------------
# THE RULE, and why it does not false-positive
# ---------------------------------------------------------------------------
# Flag `OpFDiv` when
#   * the RESULT TYPE is a vector, and
#   * the DIVISOR is an OpCompositeConstruct whose operands are all the SAME id.
#
# That second clause is what makes this precise.  A vector divided by a genuine
# VECTOR is faithful and common -- shape2Vertex (cudaNpCommon.h:284) really does
# divide each rotated component by its own scale component:
#
#     temp.x /= scale.x; temp.y /= scale.y; temp.z /= scale.z;
#
# three separate divides by three DIFFERENT scalars, which is exactly a
# componentwise vector divide.  Those have a vector divisor and are not flagged.
# Only a SPLAT -- the signature of `v / m` for scalar m -- is.
#
# ---------------------------------------------------------------------------
# IT WAS NOT A SCATTERING OF SLIPS -- IT WAS WRITTEN DOWN
# ---------------------------------------------------------------------------
# femClothClothContactBda.slang's header stated the WRONG rule as deliberate
# policy: "a PxVec3/float scalar divide is transcribed as Slang `float3 / float`
# (per-component OpFDiv), and an explicit PxVec3 * (1/x) as reciprocal-multiply --
# each is mirrored bit-for-bit by the CPU reference."  Both halves are the same
# operation in the engine, and that kernel's own source proves it:
# FEMClothExternalSolve.cu:748 writes `normal = disp / dist;` and :846 writes
# `normal = disp * (1.0f / dist);`, same type, same purpose.
#
# testFemClothSim.cpp carried a helper built to match -- `p72ref::divv`, commented
# "matching the shader's `float3 / float` ... NOT a reciprocal-multiply" -- and its
# trap A9 asserted the REVERSE of the truth as a checked property ("a multiplication
# FOLLOWED BY a division, not a multiplication by a reciprocal") for an expression
# whose operator is literally `float inv = 1.0f/s; return a * inv;`.
#
# Meanwhile five other files stated the CORRECT rule in their comments
# (npCvxCoreFace.slang:106 "NOT per-component division (review 165)",
# npIsoVecNormalize, npPsDiffuseAvg, npPsFluidDensity, npPsFluidVelocity), and
# pxgSlangFemFemAttach.slang:121 explains it exactly while a function thirty lines
# below it divided.  Two doctrines lived side by side across tranches.  Comments
# could not adjudicate between them; this check can.
#
# ---------------------------------------------------------------------------
# WHAT THIS CHECK CANNOT SEE
# ---------------------------------------------------------------------------
# Same operators, different association.  GuDistancePointTriangle.h:94-97 scales
# the two SCALARS by the reciprocal and only then forms the vector sum; three of
# the twenty-one copies of that tail in this corpus summed first and scaled after,
# and two of those had the reciprocal right.  That is invisible here and was found
# by reading the engine's statement order.  Do not read a green run as "the
# arithmetic matches" -- it means "no vector is divided by a splat".
#
# ---------------------------------------------------------------------------
# ALLOWLIST
# ---------------------------------------------------------------------------
# An entry is permitted only when the ENGINE at that site genuinely performs
# three separate divisions by the same scalar -- i.e. it is NOT going through
# PxVec3 or the cutil_math operators, but dividing raw floats it wrote out by
# hand.  Each entry must say which engine line does that.
#
# The list is EMPTY, and that is the finding rather than an accident of where the
# sweep stopped: the first run reported about sixty sites across 37 shaders, every
# one was resolved against the engine line it ports, and not a single one turned
# out to be a legitimate three-divide.

import os
import struct
import sys

SPV_DIR = os.environ.get("PXG_SHADER_DIR")
if not SPV_DIR:
    here = os.path.dirname(os.path.abspath(__file__))
    SPV_DIR = os.path.join(here, "..", "..", "..", "..", "build-gpuslang", "shaders")

OP_NAME = 5
OP_TYPE_FLOAT = 22
OP_TYPE_VECTOR = 23
OP_FUNCTION = 54
OP_FUNCTION_END = 56
OP_COMPOSITE_CONSTRUCT = 80
OP_FDIV = 136

# shader stem -> {"count": n, "why": "..."}  (see ALLOWLIST above)
ALLOWED = {}


def parse(path):
    data = open(path, "rb").read()
    if len(data) < 20 or struct.unpack("<I", data[:4])[0] != 0x07230203:
        return None
    words = list(struct.unpack("<%dI" % (len(data) // 4), data))
    vec_types = set()
    composites = {}
    findivs = []
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
        elif op == OP_TYPE_VECTOR:
            vec_types.add(words[i + 1])
        elif op == OP_FUNCTION:
            cur_fn = words[i + 2]
        elif op == OP_FUNCTION_END:
            cur_fn = None
        elif op == OP_COMPOSITE_CONSTRUCT:
            composites[words[i + 2]] = list(words[i + 3 : i + wc])
        elif op == OP_FDIV:
            findivs.append((words[i + 1], words[i + 2], words[i + 3], words[i + 4], cur_fn))
        i += wc
    return vec_types, composites, findivs, names


def offenders(path):
    parsed = parse(path)
    if parsed is None:
        return []
    vec_types, composites, findivs, names = parsed
    out = []
    for rtype, rid, _a, b, fn in findivs:
        if rtype not in vec_types:
            continue                      # scalar divide: not this check's business
        ops = composites.get(b)
        if ops is None:
            continue                      # divisor is a real vector, e.g. shape2Vertex
        if len(set(ops)) == 1:
            out.append(names.get(fn, "fn%s" % fn))
    return out


def selftest():
    """Negative control: synthesise the two shapes and confirm the rule separates
    them.  A checker that cannot tell a splat from a vector would pass everything
    silently, which is exactly the failure this file is meant to prevent."""
    vec_types = {10}
    # %20 = splat of %5 ; %21 = a genuine three-component vector
    composites = {20: [5, 5, 5], 21: [5, 6, 7]}
    splat = [(10, 30, 1, 20)]
    real = [(10, 31, 1, 21)]
    scalar = [(9, 32, 1, 20)]

    def run(fd):
        out = []
        for rtype, rid, _a, b in fd:
            if rtype not in vec_types:
                continue
            ops = composites.get(b)
            if ops is None:
                continue
            if len(set(ops)) == 1:
                out.append(rid)
        return out

    ok = True
    if run(splat) != [30]:
        print("SELFTEST FAIL: a splat divisor was not flagged")
        ok = False
    if run(real) != []:
        print("SELFTEST FAIL: a genuine vector divisor WAS flagged -- shape2Vertex "
              "would be a false positive")
        ok = False
    if run(scalar) != []:
        print("SELFTEST FAIL: a scalar-result divide was flagged")
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
        print("A VECTOR IS BEING DIVIDED BY A SPLAT SCALAR.  In the engine that is")
        print("always a reciprocal-then-multiply (PxVec3.h:219/:260/:295); three")
        print("componentwise divides differ from it for ~53% of vectors.")
        print("Write  v * (1.0f / m)  -- with the reciprocal formed ONCE -- or add an")
        print("allowlist entry naming the engine line that really divides three times.")
        print()
        for stem, found, allowed in bad:
            print("  %-34s %d splat-divide(s), %d allowed  in: %s"
                  % (stem, len(found), allowed, ", ".join(sorted(set(found)))))
        return 1

    print("vectorDivide: %d shaders scanned, no vector-by-splat divides "
          "(%d allowlisted)" % (scanned, sum(v.get("count", 0) for v in ALLOWED.values())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
