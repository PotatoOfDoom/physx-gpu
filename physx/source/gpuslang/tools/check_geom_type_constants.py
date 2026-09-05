#!/usr/bin/env python3
# check_geom_type_constants.py -- P9.57.  The TENTH static checker.
#
# Every constant in the port that names a `PxGeometryType` member must carry that
# member's value, read out of the engine header rather than out of memory.
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS
# ---------------------------------------------------------------------------
# `PxGeometryType::Enum` (include/geometry/PxGeometry.h:48-64) is:
#
#     eSPHERE 0, ePLANE 1, eCAPSULE 2, eBOX 3, eCONVEXCORE 4, eCONVEXMESH 5,
#     ePARTICLESYSTEM 6, eTETRAHEDRONMESH 7, eTRIANGLEMESH 8, eHEIGHTFIELD 9,
#     eCUSTOM 10
#
# A Slang shader cannot include that header, so all 50-odd copies of these values
# in the corpus are hand-written -- and two of them were wrong:
#
#   pxgSlangCvxTriSetup.slang    CT_GEOM_TRIANGLEMESH = 4   (eCONVEXCORE)
#   softBodyContactGenBda.slang  GEOM_CONVEXMESH      = 4   (eCONVEXCORE)
#
# Both are on LIVE, MAPPED kernels.  The first drives the operand swap in
# `ctCommon`: convexMesh.cu:721 and :856 swap when slot 0 is an eTRIANGLEMESH, so
# with 4 the port never swaps a genuinely mesh-first pair -- and then reads the
# MESH shape's blob as a convex hull.  The second gates the softbody-vs-rigid
# GJK dispatch (softBodyContactGenBda:802): a rigid CONVEX MESH is type 5, fails
# the test, falls through `type != GEOM_PLANE` and its contact is dropped IN
# SILENCE, while an eCONVEXCORE shape is routed into a hull path it has no blob
# for.
#
# ---------------------------------------------------------------------------
# WHY NEITHER A FIXTURE NOR A MUTANT FINDS IT
# ---------------------------------------------------------------------------
# The same reason P9.36, P9.45 and P9.47 each needed a checker, and here it is
# especially sharp: the FIXTURES CARRY THE SAME NUMBER.  testCvxTriSetup.cpp
# declared `GEOM_TRIANGLEMESH = 4` and then set a shape's type to it and asserted
# the swap -- so shader and oracle agreed, exactly, on a value the engine never
# produces.  Mutating either one alone fails the test; the pair is consistent and
# wrong together, which is what a differential cannot see.
#
# pxgSlangCvxTriSetup's own header warns about this failure mode two lines above
# the defect, about the BV32 stride: "That is the failure mode of a fixture that
# DEFINES the layout it is meant to check: mutating the constant moves shader and
# oracle together."  The remedy there was to take the stride from the engine
# header with a static_assert.  This checker is that remedy for the enum, plus the
# part a static_assert cannot reach -- the Slang side.
#
# ---------------------------------------------------------------------------
# THE RULE
# ---------------------------------------------------------------------------
# Parse the enumerator values from PxGeometry.h.  Then, in every shader
# (`shaders/*.slang`) and every fixture (`tests/*.cpp`, `sdktest/*.cpp`), find
# constant definitions of the form
#
#     <qualifiers> <IDENT> = <integer literal>
#
# where IDENT contains one of the tokens GEOM, TYPE or EGT and ends in `_<MEMBER>`
# for some enumerator member.  Require the literal to equal the member's value.
#
# The token requirement is what keeps the rule honest: it matches CT_GEOM_SPHERE,
# PXG_GEOM_BOX, GEOM_HEIGHTFIELD, CF_TYPE_TRIANGLEMESH, TYPE_CAPSULE and
# EGT_PLANE, and it does not match a `MAX_SPHERE` or a `NUM_BOX` that happens to
# end in the same word.  A name that ends in a geometry type and calls itself a
# geometry type is claiming to BE that value; nothing else is flagged.
#
# Longest member wins, so eTRIANGLEMESH is never mistaken for a shorter suffix.

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # source/gpuslang
PHYSX = os.path.dirname(os.path.dirname(ROOT))                       # physx/
REPO = os.path.dirname(PHYSX)
GEOM_HEADER = os.path.join(PHYSX, "include", "geometry", "PxGeometry.h")


def parse_enum(path):
    """PxGeometryType::Enum -> {MEMBER_NAME_WITHOUT_E: value}."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    m = re.search(r"struct\s+PxGeometryType\s*\{(.*?)\n\};", text, re.S)
    if not m:
        m = re.search(r"PxGeometryType\s*\{(.*?)\n\};", text, re.S)
    if not m:
        sys.exit("check_geom_type_constants: cannot find PxGeometryType in %s" % path)
    body = m.group(1)
    out = {}
    nxt = 0
    for line in body.splitlines():
        line = re.sub(r"//.*$", "", line)
        line = re.sub(r"/\*.*?\*/", "", line)
        line = line.strip().rstrip(",").strip()
        em = re.match(r"^(e[A-Z_0-9]+)\s*(?:=\s*(-?\d+))?$", line)
        if not em:
            continue
        name = em.group(1)
        if em.group(2) is not None:
            nxt = int(em.group(2))
        # internal-use sentinels are not geometry types
        if name in ("eGEOMETRY_COUNT", "eINVALID"):
            nxt += 1
            continue
        out[name[1:]] = nxt
        nxt += 1
    if "eSPHERE"[1:] not in out or "eHEIGHTFIELD"[1:] not in out:
        sys.exit("check_geom_type_constants: PxGeometryType parse looks wrong: %r" % out)
    return out


MEMBERS = parse_enum(GEOM_HEADER)
# longest first, so TRIANGLEMESH is not read as a shorter suffix
MEMBER_ORDER = sorted(MEMBERS, key=len, reverse=True)

# `static const uint FOO = 8u;` / `static const uint32_t A = 0u, B = 3u;` /
# `const int FOO = 9;` -- one regex over the assignments, then the name filter.
ASSIGN = re.compile(r"\b([A-Z][A-Z0-9_]*)\s*=\s*(0[xX][0-9a-fA-F]+|\d+)\s*[uU]?\b")
TOKENS = ("GEOM", "TYPE", "EGT")


# A short trailing disambiguator is allowed and still counts as a claim: the
# fixtures spell the same value `GEOM_TRIANGLEMESH_H` (host mirror) and
# `GEOM_PARTICLESYSTEM_H2` (a second one in the same file).  Without this, the
# very copy that hid the softbody defect -- `GEOM_CONVEXMESH_H = 4u` in
# testSoftBodySim -- would not have been read.
SUFFIX = re.compile(r"_(?:[A-Z]{1,3})?\d{0,2}$")


def member_for(name):
    if not any(tok in name for tok in TOKENS):
        return None
    for mem in MEMBER_ORDER:
        if name == mem or name.endswith("_" + mem):
            return mem
        cut = SUFFIX.search(name)
        if cut and name[:cut.start()].endswith("_" + mem):
            return mem
    return None


def scan(path):
    """-> list of (line number, identifier, value found, value expected)"""
    bad = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    in_block_comment = False
    for i, raw in enumerate(lines, 1):
        line = raw
        if in_block_comment:
            end = line.find("*/")
            if end < 0:
                continue
            line = line[end + 2:]
            in_block_comment = False
        start = line.find("/*")
        if start >= 0 and line.find("*/", start) < 0:
            line = line[:start]
            in_block_comment = True
        line = re.sub(r"//.*$", "", line)
        # a comparison is not a definition
        line = line.replace("==", "\x00").replace("!=", "\x00")
        line = line.replace(">=", "\x00").replace("<=", "\x00")
        for m in ASSIGN.finditer(line):
            name, lit = m.group(1), m.group(2)
            mem = member_for(name)
            if mem is None:
                continue
            val = int(lit, 0)
            if val != MEMBERS[mem]:
                bad.append((i, name, val, MEMBERS[mem]))
    return bad


def main():
    roots = [
        (os.path.join(ROOT, "shaders"), ".slang"),
        (os.path.join(ROOT, "tests"), ".cpp"),
        (os.path.join(ROOT, "sdktest"), ".cpp"),
    ]
    files = []
    for d, ext in roots:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(ext):
                files.append(os.path.join(d, name))

    failures = []
    checked = 0
    for path in files:
        hits = scan(path)
        checked += 1
        for line, name, got, want in hits:
            failures.append((path, line, name, got, want))

    print("check_geom_type_constants: PxGeometryType = %s"
          % ", ".join("%s %d" % (k, MEMBERS[k]) for k in sorted(MEMBERS, key=MEMBERS.get)))
    print("check_geom_type_constants: %d files scanned" % checked)

    if failures:
        print("")
        print("FAIL: %d constant(s) name a PxGeometryType member and carry another "
              "value:" % len(failures))
        for path, line, name, got, want in failures:
            mem = member_for(name)
            print("  %s:%d  %s = %d   (PxGeometryType::e%s is %d)"
                  % (os.path.relpath(path, REPO), line, name, got, mem, want))
        return 1

    print("check_geom_type_constants: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
