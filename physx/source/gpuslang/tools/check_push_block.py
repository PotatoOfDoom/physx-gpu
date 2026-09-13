#!/usr/bin/env python3
# check_push_block.py -- the SIXTEENTH static checker, and the FOURTH written
# because its defect had already happened.
#
# ---------------------------------------------------------------------------
# WHAT IT CHECKS
# ---------------------------------------------------------------------------
# Every mapped kernel has two independent declarations of its push-constant
# block, and nothing in the build compares them:
#
#   the ABI ROW   bridge/PxgVulkanKernelAbi.cpp -- the CUDA argument sizes, from
#                 which pushConstantSizeFor() computes the number of bytes the
#                 bridge actually pushes;
#   the SHADER    the [[vk::push_constant]] struct, and -- what matters -- the
#                 members of it the entry point REACHES.
#
# The invariant is NOT that the two sizes are equal.  Several shaders in this
# corpus share ONE params struct across several entry points, sized for the
# largest of them; the others then declare more than their own row pushes and
# are perfectly correct as long as they do not READ the surplus.  So this walks
# each entry point's call graph, collects the byte RANGE of every push-block
# member it reaches through an OpAccessChain, and asks two questions:
#
#     1. does the entry point read PAST the bytes the bridge pushes for it?
#        (never acceptable -- those bytes were never written)
#     2. is there an engine argument NO reached member overlaps?
#        (an argument the port never reads -- usually deliberate, never free)
#
# P9.78b REPLACED A HIGH-WATER MARK WITH RANGES, and that is the whole of it.
# The first version asked only how FAR into the block an entry point reached,
# which cannot see a dead argument in the MIDDLE of the pack.  Measured on the
# same corpus, the two questions do not agree:
#
#     reach    67 entry points stop short of their last argument
#     coverage 126 entry points leave 208 arguments unread
#
# ps_rigidAttachmentPrepareLaunch is the example that made the difference
# concrete: its `dt` (@40) and `isTGS` (@44) are read by NOTHING, and the reach
# measure scored it EXACT because activeParticleSystems (@48) sits after them.
#
# ---------------------------------------------------------------------------
# WHY: THE MEASUREMENT THAT PROMPTED IT (P9.76)
# ---------------------------------------------------------------------------
# pushConstantSizeFor() reads the ABI ROW and NEVER the SPIR-V
# (bridge/PhysXGpuVulkan.cpp:178-182).  Nothing anywhere compares the shader's
# block against the table.  P9.76 measured that directly: psRigidAttachBda's
# 72-byte solve block was shrunk to 56, rebuilt, and the engine printed
#
#     [pxg-launch] ps_solveRigidAttachmentsLaunch  grid=(1024,1,1) blob=72
#     [pxg-launch] sized ps_solveRigidAttachmentsLaunch OK grid(1024,1,1)
#
# -- accepted, with the scene still green.  Compare P9.76b, where a row that
# disagreed with the HOST pack produced a refusal on every launch and therefore
# a trace line someone could find.  A block that disagrees with its row produces
# nothing at all:
#   - the shim does not see it (it compares the blob against the row, and both
#     come from the row);
#   - a FIXTURE cannot see it either, because a fixture dispatches the shader
#     with its OWN host-side push struct and never consults the table;
#   - a SCENE sees only whatever the wrong bytes happened to change.
# The silent half of a defect pair is the half that needs a checker.
#
# ---------------------------------------------------------------------------
# WHAT IT DOES *NOT* CHECK
# ---------------------------------------------------------------------------
# Byte LAYOUT within the pushed range.  Two structs can agree on which bytes are
# read and still disagree on what lives there.  Coverage is a necessary
# condition, not a sufficient one.
#
# Nor can it tell a DELIBERATE omission from an accidental one -- that is a
# question about the CUDA, and the answer goes in push_block_known.txt beside
# the kernel, where a reader can check it.
#
# Nor does it check kernels with no ABI row: those are refused outright by the
# shim (pushConstantSizeFor returns 0), which check_launch_abi.py already holds.
#
# ---------------------------------------------------------------------------
# THE RATCHET
# ---------------------------------------------------------------------------
# An unread argument is usually not a defect.  The engine's argument pack is one
# shape for a whole family of kernels, and three distinct benign causes account
# for most of them:
#
#   - DEAD UPSTREAM.  The CUDA kernel does not use the parameter either.
#     `removeContactManagers_Stage1..4` take ONE parameter in CUDA
#     (pairManagement.cu:45-64) while the host builds a TWO-argument pack and
#     fires it at all five stages (PxgNarrowphaseCore.cpp:8527-8546), so
#     `copyManifold` is real in the row and absent from four of the kernels.
#     `materials` is threaded through four particle contact-gen kernels and
#     their whole helper chain (cudaParticleSystem.cu:1189-1679) and never
#     dereferenced once.
#   - ABI PARITY.  boxBoxNphase.slang says it in the struct itself:
#     `float toleranceLength; // @132 (unused by this kernel, present for ABI
#     parity)`.
#   - A DEFERRED BRANCH.  articulations, cone limits, SDF, friction materials,
#     a dynamic-rigid reaction -- documented in the shader's header.
#
# What the list is FOR is that the number stops being assumed.  The recorded
# SIGNATURE is checked, not just the name: a kernel that starts leaving a
# DIFFERENT argument unread is a regression the bare name would hide.  The list
# may SHRINK, never grow.  Reading PAST the row is never ratchetable.
#
# Usage: python3 check_push_block.py [shader-dir]
#        (default: ../../../../build-gpuslang/shaders)

import os
import re
import struct
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
ABI = os.path.join(ROOT, "physx", "source", "gpuslang", "bridge", "PxgVulkanKernelAbi.cpp")
MANIFEST = os.path.join(ROOT, "physx", "source", "gpuslang", "include", "PxgSlangKernelManifest.h")
KNOWN = os.path.join(os.path.dirname(__file__), "push_block_known.txt")

# ---------------------------------------------------------------------------
# SPIR-V
# ---------------------------------------------------------------------------
OP_ENTRY_POINT = 15
OP_MEMBER_NAME = 6
OP_MEMBER_DECORATE = 72
OP_DECORATE = 71
OP_TYPE_BOOL = 20
OP_TYPE_INT = 21
OP_TYPE_FLOAT = 22
OP_TYPE_VECTOR = 23
OP_TYPE_MATRIX = 24
OP_TYPE_ARRAY = 28
OP_TYPE_STRUCT = 30
OP_TYPE_POINTER = 32
OP_CONSTANT = 43
OP_FUNCTION = 54
OP_FUNCTION_END = 56
OP_FUNCTION_CALL = 57
OP_VARIABLE = 59
OP_LOAD = 61
OP_ACCESS_CHAIN = 65
OP_IN_BOUNDS_ACCESS_CHAIN = 66

DECOR_ARRAY_STRIDE = 6
DECOR_MATRIX_STRIDE = 7
DECOR_OFFSET = 35
SC_PUSH_CONSTANT = 9


def literal_string(words):
    """Decode a SPIR-V literal string; returns (text, words_consumed)."""
    raw = b""
    for n, w in enumerate(words):
        chunk = struct.pack("<I", w)
        raw += chunk
        if b"\0" in chunk:
            return raw.split(b"\0")[0].decode("utf-8", "replace"), n + 1
    return raw.split(b"\0")[0].decode("utf-8", "replace"), len(words)


class Module(object):
    def __init__(self, path):
        data = open(path, "rb").read()
        self.words = list(struct.unpack("<%dI" % (len(data) // 4), data))
        if not self.words or self.words[0] != 0x07230203:
            raise ValueError("not a SPIR-V module: %s" % path)
        self.types = {}          # id -> (opcode, operand words)
        self.constants = {}      # id -> value (32-bit literals only)
        self.member_offset = {}  # (struct id, member) -> offset
        self.array_stride = {}   # id -> stride
        self.matrix_stride = {}  # id -> stride
        self.pointers = {}       # id -> (storage class, pointee)
        self.push_vars = {}      # var id -> pointee type id
        self.entries = []        # (name, function id, [interface ids])
        self.member_name = {}    # (struct id, member) -> source name
        self.fn_members = {}     # function id -> {(push var, member index)}
        self.fn_whole = {}       # function id -> {push var} loaded entire
        self.fn_calls = {}       # function id -> {callee ids}
        self._parse()

    def _parse(self):
        w = self.words
        cur = None
        i = 5
        while i < len(w):
            wc = w[i] >> 16
            op = w[i] & 0xFFFF
            if wc == 0:
                break
            if op == OP_MEMBER_NAME:
                nm, _ = literal_string(w[i + 3:i + wc])
                self.member_name[(w[i + 1], w[i + 2])] = nm
            elif op == OP_ENTRY_POINT:
                name, used = literal_string(w[i + 3:i + wc])
                self.entries.append((name, w[i + 2], list(w[i + 3 + used:i + wc])))
            elif op == OP_MEMBER_DECORATE and w[i + 3] == DECOR_OFFSET:
                self.member_offset[(w[i + 1], w[i + 2])] = w[i + 4]
            elif op == OP_DECORATE and w[i + 2] == DECOR_ARRAY_STRIDE:
                self.array_stride[w[i + 1]] = w[i + 3]
            elif op == OP_DECORATE and w[i + 2] == DECOR_MATRIX_STRIDE:
                self.matrix_stride[w[i + 1]] = w[i + 3]
            elif op == OP_TYPE_POINTER:
                self.pointers[w[i + 1]] = (w[i + 2], w[i + 3])
                self.types[w[i + 1]] = (op, list(w[i + 2:i + wc]))
            elif op in (OP_TYPE_BOOL, OP_TYPE_INT, OP_TYPE_FLOAT, OP_TYPE_VECTOR,
                        OP_TYPE_MATRIX, OP_TYPE_ARRAY, OP_TYPE_STRUCT):
                self.types[w[i + 1]] = (op, list(w[i + 2:i + wc]))
            elif op == OP_CONSTANT and wc == 4:
                self.constants[w[i + 2]] = w[i + 3]
            elif op == OP_VARIABLE and w[i + 3] == SC_PUSH_CONSTANT:
                ptr = self.pointers.get(w[i + 1])
                if ptr:
                    self.push_vars[w[i + 2]] = ptr[1]
            elif op == OP_FUNCTION:
                cur = w[i + 2]
                self.fn_members.setdefault(cur, set())
                self.fn_whole.setdefault(cur, set())
                self.fn_calls.setdefault(cur, set())
            elif op == OP_FUNCTION_END:
                cur = None
            elif op == OP_FUNCTION_CALL and cur is not None:
                self.fn_calls[cur].add(w[i + 3])
            elif op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN) and cur is not None:
                base = w[i + 3]
                if base in self.push_vars:
                    # The FIRST index selects the struct member and must be a
                    # constant (SPIR-V requires it for a Block); a dynamic index
                    # into a push block is not representable, so a miss here is a
                    # walker bug rather than a shader shape.
                    idx = self.constants.get(w[i + 4]) if wc > 4 else None
                    self.fn_members[cur].add((base, idx))
            elif op == OP_LOAD and cur is not None and w[i + 3] in self.push_vars:
                self.fn_whole[cur].add(w[i + 3])   # the WHOLE block, every byte
            i += wc

    def type_size(self, tid):
        """Byte size of a type, or None when this walker cannot decide it."""
        t = self.types.get(tid)
        if not t:
            return None
        op, a = t
        if op == OP_TYPE_INT or op == OP_TYPE_FLOAT:
            return a[0] // 8
        if op == OP_TYPE_BOOL:
            # Not permitted in an interface block; Slang lowers bool to uint
            # before this point, so seeing one means something unmodelled.
            return None
        if op == OP_TYPE_VECTOR:
            c = self.type_size(a[0])
            return None if c is None else c * a[1]
        if op == OP_TYPE_MATRIX:
            stride = self.matrix_stride.get(tid)
            return None if stride is None else stride * a[1]
        if op == OP_TYPE_ARRAY:
            stride = self.array_stride.get(tid)
            n = self.constants.get(a[1])
            return None if (stride is None or n is None) else stride * n
        if op == OP_TYPE_STRUCT:
            return self.struct_size(tid)
        return None

    def struct_size(self, sid):
        t = self.types.get(sid)
        if not t or t[0] != OP_TYPE_STRUCT:
            return None
        total = 0
        for idx, member in enumerate(t[1]):
            off = self.member_offset.get((sid, idx))
            sz = self.type_size(member)
            if off is None or sz is None:
                return None
            total = max(total, off + sz)
        return (total + 3) & ~3          # Vulkan push ranges are multiples of 4

    def member_range(self, sid, idx):
        """[offset, offset+size) of one struct member, or None when undecidable."""
        t = self.types.get(sid)
        if not t or t[0] != OP_TYPE_STRUCT or idx is None or idx >= len(t[1]):
            return None
        off = self.member_offset.get((sid, idx))
        sz = self.type_size(t[1][idx])
        return None if (off is None or sz is None) else (off, off + sz)

    def block_members(self, name):
        """[(offset, end, source name)] for the entry point's push block, so the
        report can say WHICH field an unread argument overlaps rather than only
        where it sits."""
        for ename, _fn, iface in self.entries:
            if ename != name:
                continue
            pushes = [v for v in iface if v in self.push_vars]
            if len(pushes) != 1:
                return []
            sid = self.push_vars[pushes[0]]
            t = self.types.get(sid)
            if not t or t[0] != OP_TYPE_STRUCT:
                return []
            out = []
            for idx in range(len(t[1])):
                r = self.member_range(sid, idx)
                if r:
                    out.append((r[0], r[1], self.member_name.get((sid, idx), "?")))
            return out
        return []

    def coverage_for_entry(self, name):
        """(reached byte ranges, declared block size, note).

        Every push-block member the entry point can reach through its call
        graph, as a byte range.  RANGES, not a single high-water mark: a reach
        cannot see a dead argument in the MIDDLE of the pack, which is the blind
        spot P9.78 recorded and P9.78b closes."""
        for ename, fn, iface in self.entries:
            if ename != name:
                continue
            pushes = [v for v in iface if v in self.push_vars]
            if not pushes:
                return [], 0, "no push-constant variable in the interface"
            if len(pushes) > 1:
                return None, None, "%d push-constant variables in one interface" % len(pushes)
            var = pushes[0]
            sid = self.push_vars[var]
            declared = self.struct_size(sid)

            seen, stack = set(), [fn]
            while stack:
                f = stack.pop()
                if f in seen:
                    continue
                seen.add(f)
                stack.extend(self.fn_calls.get(f, ()))

            if any(var in self.fn_whole.get(f, ()) for f in seen):
                return [(0, declared)], declared, "whole-block OpLoad"

            ranges = []
            for f in seen:
                for (v, idx) in self.fn_members.get(f, ()):
                    if v != var:
                        continue
                    if idx is None:
                        return None, declared, "non-constant member index"
                    r = self.member_range(sid, idx)
                    if r is None:
                        return None, declared, "member %d undecidable" % idx
                    ranges.append(r)
            return ranges, declared, ""
        return None, None, "entry point not found in the module"


# ---------------------------------------------------------------------------
# the ABI table and the manifest
# ---------------------------------------------------------------------------
def _param_align(size):
    """paramAlign (PxgVulkanCudaContext.cpp:114) / abiAlign (PxgVulkanKernelAbi.cpp
    :2206): the largest power of two that DIVIDES the size, capped at 16.

    Transcribed as a modulo ladder, which it is.  P9.78b's first draft wrote it as
    `size if size in (1,2,4,8) else 16`, which is right for every power-of-two size
    and wrong for a 12-byte PxVec3 -- the by-value `gravity` argument.  That gave
    16 where the bridge gives 4 and shifted every argument after it, mis-sizing 16
    of the 358 rows and inventing two "unread arguments" that were nothing of the
    kind.  The same wrong copy shipped in check_launch_abi_pack.py and is corrected
    there too."""
    if size == 0:
        return 1
    for a in (16, 8, 4, 2):
        if size % a == 0:
            return a
    return 1


def _abi_arg_ranges(row):
    """[(start, end)] per CUDA argument, laid out exactly as the bridge lays the
    blob out (PxgVulkanCudaContext.cpp:443-462)."""
    out, off = [], 0
    for s in row:
        a = _param_align(s)
        off = (off + a - 1) & ~(a - 1)
        out.append((off, off + s))
        off += s
    return out


def abi_push_size(row):
    """pxgKernelAbiPushSize (PxgVulkanKernelAbi.cpp:2231), transcribed."""
    ranges = _abi_arg_ranges(row)
    end = ranges[-1][1] if ranges else 0
    ma = max([_param_align(s) for s in row] + [1])
    return (end + ma - 1) & ~(ma - 1)


def read_abi():
    src = open(ABI).read()
    arrays = {}
    for m in re.finditer(r"static\s+const\s+uint8_t\s+(\w+)\s*\[\]\s*=\s*\{([^}]*)\}", src):
        arrays[m.group(1)] = [int(v) for v in re.findall(r"\d+", m.group(2))]
    rows = {}
    for m in re.finditer(r'\{\s*"([^"]+)"\s*,\s*(\w+)\s*,', src):
        if m.group(2) in arrays:
            rows[m.group(1)] = arrays[m.group(2)]
    return rows


def read_manifest():
    out = []
    for m in re.finditer(r'\{\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\}', open(MANIFEST).read()):
        out.append((m.group(1), m.group(2), m.group(3)))   # cuda, spv, entry
    return out


def read_known():
    """kernel -> the unread-argument SIGNATURE ("index@offset+size;...").

    The signature is part of the ratchet, not just the name: a kernel that starts
    leaving a DIFFERENT argument unread is a regression the bare name would hide."""
    known = {}
    if not os.path.exists(KNOWN):
        return known
    for line in open(KNOWN):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        known[parts[0]] = parts[1] if len(parts) > 1 else None
    return known


def main():
    shader_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build-gpuslang", "shaders")
    if not os.path.isdir(shader_dir):
        print("no shader directory: %s" % shader_dir)
        return 2

    rows = read_abi()
    known = read_known()
    modules = {}

    full = 0
    no_row = 0
    missing_spv = 0
    undecided = []
    over = []      # the entry point can read PAST the pushed bytes
    nomember = []  # a row argument NO declared member overlaps -- a layout disagreement
    partial = []   # (cuda, entry, unread argument indices, nargs, note)

    for cuda, spv, entry in read_manifest():
        path = os.path.join(shader_dir, spv)
        if not os.path.exists(path):
            missing_spv += 1
            continue
        if spv not in modules:
            try:
                modules[spv] = Module(path)
            except ValueError as e:
                undecided.append((cuda, spv, entry, str(e)))
                continue
        row = rows.get(cuda)
        if row is None:
            no_row += 1
            continue
        ranges, declared, note = modules[spv].coverage_for_entry(entry)
        if ranges is None:
            undecided.append((cuda, spv, entry, note))
            continue

        want = abi_push_size(row)
        reach = max([e for (_s, e) in ranges] + [0])
        if reach > want:
            over.append((cuda, entry, reach, want, declared, note))
            continue

        args = _abi_arg_ranges(row)
        fields = modules[spv].block_members(entry)

        def name_at(a, b):
            hit = [nm for (ms, me, nm) in fields if ms < b and a < me]
            return "/".join(hit) if hit else "-"

        # An argument NO DECLARED MEMBER overlaps is a different animal from one
        # that is declared and simply never read: the shader's struct and the row
        # disagree on the LAYOUT, so every field after the divergence sits over
        # the wrong bytes.  P9.78b found its own alignment bug this way -- a
        # 12-byte PxVec3 given alignment 16 instead of 4 shifted every later
        # argument and produced exactly this signature.
        unmapped = [i for i, (a, b) in enumerate(args)
                    if fields and not any(ms < b and a < me for (ms, me, _n) in fields)]
        unread = [i for i, (a, b) in enumerate(args)
                  if i not in unmapped and not any(rs < b and a < re for (rs, re) in ranges)]

        if unmapped:
            nomember.append((cuda, entry,
                             ";".join("%d@%d+%d" % (i, args[i][0], args[i][1] - args[i][0])
                                      for i in unmapped)))
        if unread:
            partial.append((cuda, entry, unread, len(args),
                            ";".join("%d@%d+%d:%s" % (i, args[i][0], args[i][1] - args[i][0],
                                                      name_at(args[i][0], args[i][1]))
                                     for i in unread)))
        elif not unmapped:
            full += 1

    total = full + len(partial) + len(over) + len(undecided) + len(nomember)
    print("push-block COVERAGE vs ABI row over %d mapped kernels with a row "
          "(%d without a row, %d with no built .spv)" % (total, no_row, missing_spv))
    print("  FULL       %d  (every engine argument is read)" % full)
    print("  PARTIAL    %d  (at least one argument no code reads)" % len(partial))
    print("  OVER-READ  %d  (can read past the pushed bytes)" % len(over))
    print("  NO-MEMBER  %d  (a row argument no declared field covers -- layouts disagree)" % len(nomember))
    print("  UNDECIDED  %d" % len(undecided))
    print("  unread arguments in total: %d" % sum(len(p[2]) for p in partial))

    failures = 0

    if nomember:
        print("\nA row argument that NO declared push-block field covers means the shader")
        print("struct and the ABI row disagree on the LAYOUT, not merely on what is read:")
        for cuda, entry, sig in sorted(nomember):
            print("  %-52s %s" % (cuda, sig))
        failures += len(nomember)

    if over:
        print("\nAn entry point that reaches BEYOND its ABI row reads bytes the bridge")
        print("never pushed.  Nothing reports this at runtime -- see the header:")
        for cuda, entry, reach, want, declared, note in sorted(over):
            print("  %-52s reaches byte %d, the row pushes %d (block is %s)%s"
                  % (cuda, reach, want, declared, (" [%s]" % note) if note else ""))
        failures += len(over)

    seen_now = {c: sig for (c, _e, _u, _n, sig) in partial}
    fresh = [x for x in partial if x[0] not in known]
    if fresh:
        print("\n%d entry point(s) leave an engine argument UNREAD and are not on the" % len(fresh))
        print("ratchet.  That is usually deliberate -- an argument the port defers, or one")
        print("the kernel is documented not to use -- but it is never free: record each in")
        print("tools/push_block_known.txt with the reason, or read it:")
        for cuda, entry, unread, nargs, sig in sorted(fresh):
            print("  %-46s %d of %d arguments unread: %s" % (cuda, len(unread), nargs, sig))
        failures += len(fresh)

    moved = [(c, known[c], sig) for c, sig in seen_now.items()
             if known.get(c) is not None and known[c] != sig]
    if moved:
        print("\n%d recorded partial reader(s) MOVED.  Leaving MORE unread is a" % len(moved))
        print("regression; leaving less means the ratchet line is stale:")
        for cuda, was, now in sorted(moved):
            print("  %-46s was [%s], now [%s]" % (cuda, was, now))
        failures += len(moved)

    stale = [k for k in known if k not in seen_now]
    if stale:
        print("\n%d STALE entry(ies) in push_block_known.txt -- these kernels now read" % len(stale))
        print("every argument; remove the lines.  A ratchet that keeps finished kernels")
        print("on its list stops ratcheting:")
        for k in sorted(stale):
            print("  %s" % k)
        failures += len(stale)

    if undecided:
        print("\n%d kernel(s) this walker could not decide -- a recorded blind spot," % len(undecided))
        print("not a pass:")
        for cuda, spv, entry, note in sorted(undecided)[:20]:
            print("  %-48s %s:%s -- %s" % (cuda, spv, entry, note))
        if len(undecided) > 20:
            print("  ... and %d more" % (len(undecided) - 20))

    if failures:
        print("\ncheck_push_block: %d problem(s)" % failures)
        return 1
    print("\ncheck_push_block: clean (%d partial reader(s) on the ratchet, %d unread arguments)"
          % (len(partial), sum(len(p[2]) for p in partial)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
