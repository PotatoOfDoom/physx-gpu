#!/usr/bin/env python3
# check_groupshared.py -- P9.83b-1.  The EIGHTEENTH static checker, and the THIRD
# written because its defect had already happened.
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS
# ---------------------------------------------------------------------------
# llvmpipe reports maxComputeSharedMemorySize = 32768.  A shader that declares
# more Workgroup storage than that does not run: vkCreateComputePipelines fails,
# the kernel is never dispatched, and on this port a kernel that is never
# dispatched is SILENT -- the same failure mode the whole [pxg-unmapped] guard
# machinery exists for.
#
# It has already happened once.  P5's solveWholeIslandTGS had its shared staging
# capped 944 -> 640 entries for exactly this reason (RUNTIME_STATUS.md), and that
# cap was found by hand, late, after a pipeline creation failure.
#
# It was about to happen again.  P9.83b's audit of triangleTriangleCollision
# projected ~38.9 KB against the 32768 limit -- and the hand arithmetic that
# preceded the audit had said ~29.2 KB and concluded "it fits".  BOTH numbers were
# arithmetic on the CUDA's declarations.  Neither was a measurement of a compiled
# module, which is what this checker does instead.
#
# ---------------------------------------------------------------------------
# WHY THE HAND ARITHMETIC IS WRONG BY CONSTRUCTION
# ---------------------------------------------------------------------------
# Two effects the CUDA source cannot show:
#
#   STRUCT PADDING.  SPIR-V lays Workgroup structs out with the alignment its
#   member types demand.  `struct Patch { float4 KeepPoints[6]; float3 Normal;
#   float Deepest; uint PointCount; }` is 116 bytes in CUDA and 128 in Workgroup
#   storage, because the float4 array forces 16-byte alignment on the whole
#   record.  Over an array of 128 that is 1536 bytes nobody wrote down.
#
#   IMPORTED MODULES DECLARE THEIR OWN.  A Slang `import` or `#include` brings the
#   module's groupshared with it, so a caller's budget is its own declarations PLUS
#   every module it pulls in.  pxgSlangSdfFind alone contributes 8192 + 4096 + 136.
#
#   *** WHAT IS **NOT** TRUE, AND THIS CHECKER'S FIRST DRAFT SAID IT WAS. ***  The
#   P9.83b audit claimed a declaration that is DEAD on the caller's path still
#   occupies the budget "because SPIR-V keeps the OpVariable", and this header
#   repeated it.  Measured, it is false for this compiler:
#       femClothSdfMeshCgBda:cloth_sdfMeshContactGenLaunch   12424 B
#           gSdfIndices2 8192, gPxgWarpScratch 4096, ...     and NO gSdfIndices
#       softBodySdfMidphaseBda:sb_sdfMeshMidphase...          8328 B
#           gSdfIndices  4096, gPxgWarpScratch 4096, ...     and NO gSdfIndices2
#   The two are complementary: pxgSlangSdfFind declares both, and each caller's
#   entry point carries only the one it actually uses.  So an unused Workgroup
#   variable from an included file costs NOTHING, and the budget to plan against is
#   the measured one, not the sum of the declarations.  That is a 4096-byte
#   difference in the P9.83b projection, which is most of what stood between it and
#   the limit -- which is why it is written down here rather than quietly fixed.
#
# So the number has to come out of the compiled .spv.  This file parses it
# directly, the way tools/spv_nocontract.py already does for decorations: there is
# no spirv-dis in this tree and there does not need to be.
#
# ---------------------------------------------------------------------------
# THE LIMIT IS PER ENTRY POINT, NOT PER MODULE, AND THE FIRST DRAFT GOT THAT WRONG
# ---------------------------------------------------------------------------
# This checker's first run reported solverTGS at 34176 bytes -- over the measured
# limit by 1408 -- for a shader whose ctest is green and whose scenes rest boxes
# correctly.  The shader is fine; the CHECKER was summing every Workgroup variable
# in the module.  One .spv here carries ELEVEN entry points
# (tests/CMakeLists.txt:180 compiles solveBlockPartitionTGS, concludeBlocksTGS,
# writebackBlocksTGS and eight more into one file), and Vulkan applies
# maxComputeSharedMemorySize per PIPELINE -- so what counts is the storage
# reachable from ONE entry point, not the module's total.
#
# Slang emits SPIR-V 1.5, where OpEntryPoint's interface list is required to name
# every global variable the entry point statically uses, Workgroup included (it was
# Input/Output only before 1.4).  So the attribution is read straight out of the
# binary rather than inferred from a call graph.  Measured on solverTGS: eleven
# entry points, fourteen Workgroup variables, and the heaviest single entry point
# is well under the limit.
#
# THE MODULE TOTAL IS STILL REPORTED, because it is the number that matters if
# anyone ever links these as one pipeline -- but it is not the number that fails.
#
# ---------------------------------------------------------------------------
# THE RULE
# ---------------------------------------------------------------------------
# For every ENTRY POINT of every compiled shader in the build's shader directory:
#
#   1. sum the byte size of every Workgroup OpVariable in its interface list;
#   2. FAIL if the total exceeds LIMIT (32768, llvmpipe's reported maximum);
#   3. WARN -- and record on a ratchet -- anything above WATERMARK (75% of the
#      limit), because a shader at 80% is one module import away from failing and
#      the person who adds that import will not be looking here.
#
# The ratchet holds the measured size of each heavy shader.  It may shrink; it may
# not grow without the number being re-read and the entry updated, which is the
# point: "this got bigger and still fits" is exactly how a limit is walked into.

import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPV_DIR = os.path.normpath(
    os.path.join(HERE, "..", "..", "..", "..", "build-gpuslang", "shaders"))
KNOWN = os.path.join(HERE, "groupshared_known.txt")

# llvmpipe's VkPhysicalDeviceLimits::maxComputeSharedMemorySize.  Not a guess:
# the Vulkan spec's own minimum is 16384, and every desktop driver this port has
# been run on reports 32768 or more.  Set from the SMALLEST target, never the
# largest, because the limit that matters is the one that fails first.
LIMIT = 32768
WATERMARK = (LIMIT * 3) // 4          # 24576

STORAGE_WORKGROUP = 4

# --- opcodes -----------------------------------------------------------------
OP_TYPE_BOOL = 20
OP_TYPE_INT = 21
OP_TYPE_FLOAT = 22
OP_TYPE_VECTOR = 23
OP_TYPE_MATRIX = 24
OP_TYPE_ARRAY = 28
OP_TYPE_RUNTIME_ARRAY = 29
OP_TYPE_STRUCT = 30
OP_TYPE_POINTER = 32
OP_CONSTANT = 43
OP_VARIABLE = 59
OP_DECORATE = 71
OP_MEMBER_DECORATE = 72
OP_NAME = 5
OP_ENTRY_POINT = 15

DECOR_ARRAY_STRIDE = 6
DECOR_OFFSET = 35


def parse(path):
    """Returns (types, constants, variables, names, strides, member_offsets)."""
    blob = open(path, "rb").read()
    if len(blob) < 20:
        raise ValueError("too short to be SPIR-V")
    magic = struct.unpack("<I", blob[0:4])[0]
    endian = "<"
    if magic != 0x07230203:
        magic_be = struct.unpack(">I", blob[0:4])[0]
        if magic_be != 0x07230203:
            raise ValueError("not SPIR-V")
        endian = ">"
    words = struct.unpack(endian + "%dI" % (len(blob) // 4), blob[:len(blob) // 4 * 4])

    types, consts, variables, names = {}, {}, [], {}
    strides, moffsets, entries = {}, {}, []
    i = 5
    while i < len(words):
        w = words[i]
        count = w >> 16
        op = w & 0xFFFF
        if count == 0:
            break
        ops = words[i + 1: i + count]
        if op == OP_ENTRY_POINT and len(ops) >= 3:
            # execution model, entry id, then a literal string, then the interface
            # ids.  The string ends at the first word containing a zero byte --
            # including the case where the last word is exactly four characters and
            # the terminator lands alone in the NEXT word.
            k = 2
            while k < len(ops):
                word = ops[k]
                k += 1
                if (word & 0xFF) == 0 or ((word >> 8) & 0xFF) == 0 \
                   or ((word >> 16) & 0xFF) == 0 or ((word >> 24) & 0xFF) == 0:
                    break
            raw = b"".join(struct.pack(endian + "I", x) for x in ops[2:k])
            entries.append((raw.split(b"\0")[0].decode("utf-8", "replace"), list(ops[k:])))
        elif op == OP_NAME and len(ops) >= 2:
            raw = b"".join(struct.pack(endian + "I", x) for x in ops[1:])
            names[ops[0]] = raw.split(b"\0")[0].decode("utf-8", "replace")
        elif op == OP_DECORATE and len(ops) >= 3 and ops[1] == DECOR_ARRAY_STRIDE:
            strides[ops[0]] = ops[2]
        elif op == OP_MEMBER_DECORATE and len(ops) >= 4 and ops[2] == DECOR_OFFSET:
            moffsets.setdefault(ops[0], {})[ops[1]] = ops[3]
        elif op == OP_CONSTANT and len(ops) >= 3:
            consts[ops[1]] = ops[2]
        elif op in (OP_TYPE_BOOL, OP_TYPE_INT, OP_TYPE_FLOAT, OP_TYPE_VECTOR,
                    OP_TYPE_MATRIX, OP_TYPE_ARRAY, OP_TYPE_RUNTIME_ARRAY,
                    OP_TYPE_STRUCT, OP_TYPE_POINTER):
            types[ops[0]] = (op, list(ops[1:]))
        elif op == OP_VARIABLE and len(ops) >= 3:
            variables.append((ops[0], ops[1], ops[2]))   # result type, result id, storage class
        i += count
    return types, consts, variables, names, strides, moffsets, entries


def type_size(tid, types, consts, strides, moffsets, depth=0):
    """Byte size of a type id, or None when it cannot be resolved."""
    if depth > 24 or tid not in types:
        return None
    op, ops = types[tid]
    if op == OP_TYPE_BOOL:
        return 4                       # a Workgroup bool occupies a word
    if op in (OP_TYPE_INT, OP_TYPE_FLOAT):
        return max(1, ops[0] // 8)
    if op == OP_TYPE_VECTOR:
        e = type_size(ops[0], types, consts, strides, moffsets, depth + 1)
        return None if e is None else e * ops[1]
    if op == OP_TYPE_MATRIX:
        e = type_size(ops[0], types, consts, strides, moffsets, depth + 1)
        return None if e is None else e * ops[1]
    if op == OP_TYPE_ARRAY:
        n = consts.get(ops[1])
        if n is None:
            return None
        # THE DECORATED STRIDE WINS.  An array of float3 has element size 12 and
        # ArrayStride 16; using the element size under-counts by a third.
        stride = strides.get(tid)
        if stride is None:
            e = type_size(ops[0], types, consts, strides, moffsets, depth + 1)
            if e is None:
                return None
            stride = e
        return stride * n
    if op == OP_TYPE_RUNTIME_ARRAY:
        return 0                       # not legal in Workgroup; counted as nothing
    if op == OP_TYPE_STRUCT:
        if not ops:
            return 0
        offs = moffsets.get(tid)
        last = type_size(ops[-1], types, consts, strides, moffsets, depth + 1)
        if offs is not None and (len(ops) - 1) in offs and last is not None:
            # THE LAST MEMBER'S OFFSET PLUS ITS SIZE, then rounded to the struct's
            # own alignment -- which is the alignment of its most-aligned member.
            total = offs[len(ops) - 1] + last
            align = 1
            for m in ops:
                ms = type_size(m, types, consts, strides, moffsets, depth + 1)
                if ms:
                    align = max(align, min(16, ms if ms in (1, 2, 4, 8, 16) else 16))
            return (total + align - 1) // align * align
        # undecorated: lay it out the way SPIR-V's default rules do
        total, align = 0, 1
        for m in ops:
            ms = type_size(m, types, consts, strides, moffsets, depth + 1)
            if ms is None:
                return None
            a = min(16, ms if ms in (1, 2, 4, 8, 16) else 16)
            align = max(align, a)
            total = (total + a - 1) // a * a + ms
        return (total + align - 1) // align * align
    if op == OP_TYPE_POINTER:
        return 8
    return None


def workgroup_bytes(path):
    """(module total, [(entry point, bytes, [(name, bytes), ...]), ...])."""
    types, consts, variables, names, strides, moffsets, entries = parse(path)
    sizes = {}
    total = 0
    for rtype, rid, storage in variables:
        if storage != STORAGE_WORKGROUP or rtype not in types:
            continue
        op, ops = types[rtype]
        if op != OP_TYPE_POINTER:
            continue
        sz = type_size(ops[1], types, consts, strides, moffsets)
        sizes[rid] = (names.get(rid, "?%d" % rid), sz)
        if sz is not None:
            total += sz

    per_entry = []
    for ename, iface in entries:
        detail = [sizes[i] for i in iface if i in sizes]
        esz = sum(d[1] for d in detail if d[1] is not None)
        per_entry.append((ename, esz, detail))
    if not entries:
        # no entry point at all: fall back to the module total so a malformed or
        # library .spv is still visible rather than silently scoring zero
        per_entry.append(("<no entry point>", total, list(sizes.values())))
    return total, per_entry


def load_known():
    known = {}
    if not os.path.exists(KNOWN):
        return known
    for raw in open(KNOWN):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                known[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return known


def main():
    spv_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SPV_DIR
    if not os.path.isdir(spv_dir):
        print("check_groupshared: no shader directory at %s -- build gpuslang-shaders first"
              % spv_dir)
        return 0

    known = load_known()
    over, heavy, unresolved = [], [], []
    scanned = 0
    for name in sorted(os.listdir(spv_dir)):
        if not name.endswith(".spv"):
            continue
        path = os.path.join(spv_dir, name)
        try:
            _module_total, per_entry = workgroup_bytes(path)
        except Exception as exc:                       # a malformed .spv is a finding
            unresolved.append((name, str(exc)))
            continue
        scanned += 1
        stem = name[:-4]
        for ename, esz, detail in per_entry:
            if any(sz is None for _n, sz in detail):
                unresolved.append(("%s:%s" % (name, ename),
                                   "a Workgroup variable's type could not be sized"))
            key = "%s:%s" % (stem, ename)
            if esz > LIMIT:
                over.append((key, esz, detail))
            elif esz > WATERMARK:
                heavy.append((key, esz, detail))

    rc = 0
    if over:
        rc = 1
        print("A SHADER DECLARES MORE WORKGROUP STORAGE THAN THE DEVICE ALLOWS.")
        print("vkCreateComputePipelines fails, the kernel is never dispatched, and on")
        print("this port a kernel that is never dispatched is SILENT.")
        print()
        for stem, total, detail in over:
            print("  %-40s %7d bytes  (limit %d, over by %d)"
                  % (stem, total, LIMIT, total - LIMIT))
            for n, sz in sorted(detail, key=lambda d: -(d[1] or 0))[:8]:
                print("      %-34s %7s" % (n, sz if sz is not None else "?"))
        print()

    fresh = [(s, t, d) for s, t, d in heavy if known.get(s) is None]
    grew = [(s, t, d) for s, t, d in heavy if known.get(s) is not None and t > known[s]]
    if fresh or grew:
        rc = 1
        print("A SHADER IS ABOVE THE %d-BYTE WATERMARK (%d%% of the %d limit)."
              % (WATERMARK, WATERMARK * 100 // LIMIT, LIMIT))
        print("That is not a failure yet.  It is one module import away from one, and")
        print("the person who adds that import will not be looking here -- so the size")
        print("is recorded in tools/groupshared_known.txt and may not grow silently.")
        print()
        for stem, total, detail in fresh:
            print("  NEW   %-38s %7d bytes" % (stem, total))
            for n, sz in sorted(detail, key=lambda d: -(d[1] or 0))[:6]:
                print("          %-34s %7s" % (n, sz if sz is not None else "?"))
        for stem, total, detail in grew:
            print("  GREW  %-38s %7d bytes (recorded %d, +%d)"
                  % (stem, total, known[stem], total - known[stem]))
            for n, sz in sorted(detail, key=lambda d: -(d[1] or 0))[:6]:
                print("          %-34s %7s" % (n, sz if sz is not None else "?"))
        print()

    stale = sorted(set(known) - set(s for s, _t, _d in heavy) - set(s for s, _t, _d in over))
    if stale:
        rc = 1
        print("%d ratchet entry(ies) name a shader that is no longer above the watermark"
              % len(stale))
        print("(or no longer exists).  Remove them -- a ratchet that keeps entries for")
        print("sizes it can no longer see is not ratcheting.")
        for s in stale:
            print("  %-40s recorded %d" % (s, known[s]))
        print()

    if unresolved:
        rc = 1
        print("These shaders have a Workgroup variable this checker could not size.")
        print("An unsized variable is an UNKNOWN in the budget, not a zero:")
        for n, why in unresolved:
            print("  %-40s %s" % (n, why))
        print()

    if rc == 0:
        biggest = max(((s, t) for s, t, _d in heavy), key=lambda x: x[1], default=None)
        print("groupshared: %d shader(s) measured, none over %d bytes%s"
              % (scanned, LIMIT,
                 "; heaviest on the ratchet is %s at %d" % biggest if biggest else ""))
    return rc


if __name__ == "__main__":
    sys.exit(main())
