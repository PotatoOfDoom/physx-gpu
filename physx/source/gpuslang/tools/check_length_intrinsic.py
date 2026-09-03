#!/usr/bin/env python3
# check_length_intrinsic.py -- P9.47.  The SEVENTH static checker.
#
# Flags every use in the built SPIR-V of the GLSL.std.450 instructions whose
# result the specification does NOT pin to the last bit, where the engine
# computes the same quantity out of operations that ARE pinned:
#
#     Length(v)     (66)   vs  PxSqrt(v.magnitudeSquared()) / sqrtf(dot(v,v))
#     Distance(a,b) (67)   vs  the same on a difference
#     Normalize(v)  (69)   vs  v * PxRecipSqrt(m) / v * rsqrtf(dot(v,v))
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS
# ---------------------------------------------------------------------------
# The Vulkan environment spec gives Length, Distance and Normalize a precision
# ALLOWANCE.  Length and Distance are permitted 2 ULP plus whatever the
# intermediate squaring costs; Normalize is permitted 2 ULP per component and
# an implementation may lower it to `v / Length(v)` or to `v * InverseSqrt(m)`
# as it pleases.  Neither is a defined bit pattern.
#
# The engine, on the other side of the port, is exact:
#
#   PxVec3::magnitude()      = PxSqrt(magnitudeSquared())  -> sqrtf, which
#                              IEEE-754 requires to be correctly rounded.
#   cutil_math.h length()    = sqrtf(dot(v,v))             -> the same.
#   PxVec3::getNormalized()  = v * PxRecipSqrt(m)          -> one rsqrt and
#                              three multiplies.
#   PxVec3::normalize()      = *this /= magnitude()        -> ONE reciprocal
#                              (PxVec3.h:260) and three multiplies.
#   cutil_math.h normalize() = v * rsqrtf(dot(v,v)).
#
# So `length()` is a different expression from `sqrt(dot())` even where both
# are "the length", and `normalize()` is a different expression from all three
# engine spellings at once.  This checker asks the corpus to spell out which
# one the engine used instead of delegating the choice to the driver.
#
# It is the seventh in a line whose whole point is that no fixture and no
# mutant can see what they check:
#   gpuslang.barrierUniformity   (P9.16)
#   gpuslang.cudaTwinKernels     (P9.20)
#   gpuslang.speculatedLoads     (P9.23)
#   gpuslang.disabledKernels     (P9.26)
#   gpuslang.vectorDivide        (P9.36)
#   gpuslang.componentDivide     (P9.45)
#
# ---------------------------------------------------------------------------
# WHY NEITHER A FIXTURE NOR A MUTANT FINDS IT
# ---------------------------------------------------------------------------
# Same reason as P9.36 and P9.45, and it is worth writing down once more.  A
# fixture compares the shader against a host mirror; if the mirror was written
# from the same misreading, both say `length()` and agree exactly.  A mutant
# asks whether the fixture notices a perturbation, so a fixture blind to the
# original is blind to the mutant too.  Only a rule that reads the BUILT
# SPIR-V and knows what the engine spells can see this.
#
# ---------------------------------------------------------------------------
# THE RULE
# ---------------------------------------------------------------------------
# Count every OpExtInst that names the GLSL.std.450 set with instruction
# number 66 (Length), 67 (Distance) or 69 (Normalize).  A shader is an
# offender when it has more of them than its allowlist entry permits.
#
# InverseSqrt (32) is deliberately NOT in this rule.  P9.33a measured it
# against `1.0f/sqrtf` over 258442 inputs on llvmpipe and the two agree bit for
# bit; `gpuslang.npFloatIntrinRecipSqrt` keeps that measurement honest.  Sqrt
# (31) is not in it either -- Sqrt is correctly rounded by the same spec that
# leaves Length loose, so it IS the engine's operation.
#
# ---------------------------------------------------------------------------
# ALLOWLIST
# ---------------------------------------------------------------------------
# An entry is permitted only when the ENGINE at that site itself calls an
# intrinsic with the same latitude, or when the result provably cannot carry a
# last-bit difference into anything observable.  Each entry names the engine
# line.
#
# (populated by the P9.47 audit -- see RUNTIME_STATUS.md)

import os
import struct
import sys
from collections import defaultdict

SPV_DIR = os.environ.get("PXG_SHADER_DIR")
if not SPV_DIR:
    here = os.path.dirname(os.path.abspath(__file__))
    SPV_DIR = os.path.join(here, "..", "..", "..", "..", "build-gpuslang", "shaders")

OP_NAME = 5
OP_EXT_INST_IMPORT = 11
OP_EXT_INST = 12
OP_FUNCTION = 54
OP_FUNCTION_END = 56

# GLSL.std.450 instruction numbers whose result the spec does not pin.
LOOSE = {66: "Length", 67: "Distance", 69: "Normalize"}

# shader stem -> {"count": n, "why": "..."}
ALLOWED = {}


def scan(words):
    """-> [(instruction name, function id)] for every loose GLSL.std.450 call."""
    glsl_sets = set()
    names = {}
    found = []
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
        elif op == OP_EXT_INST_IMPORT:
            raw = b"".join(struct.pack("<I", w) for w in words[i + 2 : i + wc])
            if raw.split(b"\0")[0] == b"GLSL.std.450":
                glsl_sets.add(words[i + 1])
        elif op == OP_FUNCTION:
            cur_fn = words[i + 2]
        elif op == OP_FUNCTION_END:
            cur_fn = None
        elif op == OP_EXT_INST and wc >= 5 and words[i + 3] in glsl_sets:
            inst = words[i + 4]
            if inst in LOOSE:
                found.append((LOOSE[inst], cur_fn))
        i += wc
    return found, names


def offenders(path):
    data = open(path, "rb").read()
    if len(data) < 20 or struct.unpack("<I", data[:4])[0] != 0x07230203:
        return []
    words = list(struct.unpack("<%dI" % (len(data) // 4), data))
    found, names = scan(words)
    out = defaultdict(list)
    for inst, fn in found:
        out["%s in %s" % (inst, names.get(fn, "fn%s" % fn))].append(inst)
    return [(k, len(v)) for k, v in sorted(out.items())]


def _spv(instructions):
    """Build a minimal module: header, one GLSL.std.450 import as %1, one other
    extended set as %2, then the given instruction words."""
    words = [0x07230203, 0x00010000, 0, 99, 0]

    def ext_import(rid, name):
        b = name.encode() + b"\0"
        b += b"\0" * ((-len(b)) % 4)
        lit = list(struct.unpack("<%dI" % (len(b) // 4), b))
        return [((2 + len(lit)) << 16) | OP_EXT_INST_IMPORT, rid] + lit

    words += ext_import(1, "GLSL.std.450")
    words += ext_import(2, "OpenCL.std")
    words += instructions
    return words


def _extinst(set_id, inst_no, result_id=50):
    # OpExtInst: result type, result id, set, instruction, one operand
    return [(6 << 16) | OP_EXT_INST, 9, result_id, set_id, inst_no, 40]


def selftest():
    """Negative control.  A checker that counted every OpExtInst, or that
    ignored which extended-instruction SET an id belongs to, would be useless
    in opposite directions -- one screams at Sqrt, the other stays silent when
    the loose call comes through a second imported set."""
    ok = True

    def n(instrs):
        return len(scan(_spv(instrs))[0])

    if n(_extinst(1, 66)) != 1:
        print("SELFTEST FAIL: a GLSL.std.450 Length was not counted")
        ok = False
    if n(_extinst(1, 69)) != 1:
        print("SELFTEST FAIL: a GLSL.std.450 Normalize was not counted")
        ok = False
    if n(_extinst(1, 31)) != 0:
        print("SELFTEST FAIL: Sqrt WAS counted -- Sqrt is correctly rounded by "
              "the same spec that leaves Length loose, so it is the engine's "
              "own operation")
        ok = False
    if n(_extinst(1, 32)) != 0:
        print("SELFTEST FAIL: InverseSqrt WAS counted -- that is "
              "gpuslang.npFloatIntrinRecipSqrt's measured business, not this "
              "checker's")
        ok = False
    if n(_extinst(2, 66)) != 0:
        print("SELFTEST FAIL: instruction 66 of a DIFFERENT extended set was "
              "counted as GLSL.std.450 Length")
        ok = False
    if n(_extinst(1, 66) + _extinst(1, 67, 51) + _extinst(1, 69, 52)) != 3:
        print("SELFTEST FAIL: Length + Distance + Normalize did not add up")
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
        total = sum(c for _k, c in found)
        allowed = ALLOWED.get(stem, {}).get("count", 0)
        if total > allowed:
            bad.append((stem, found, total, allowed))

    if bad:
        print("A PRECISION-LOOSE GLSL.std.450 INSTRUCTION IS STANDING IN FOR AN EXACT")
        print("ENGINE EXPRESSION.  Length/Distance/Normalize are allowed several ULP")
        print("by the Vulkan environment spec and an implementation may lower them as")
        print("it likes; the engine computes sqrtf(dot(v,v)) and v * rsqrtf(dot(v,v)),")
        print("both of which ARE pinned.  Spell the engine's expression out -- or add")
        print("an allowlist entry naming the engine line that really calls an")
        print("intrinsic with the same latitude.")
        print()
        for stem, found, total, allowed in bad:
            print("  %-38s %d loose call(s), %d allowed" % (stem, total, allowed))
            for k, c in found:
                print("      %s%s" % (k, " x%d" % c if c > 1 else ""))
        return 1

    print("lengthIntrinsic: %d shaders scanned, no precision-loose Length/Distance/"
          "Normalize (%d allowlisted)"
          % (scanned, sum(v.get("count", 0) for v in ALLOWED.values())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
