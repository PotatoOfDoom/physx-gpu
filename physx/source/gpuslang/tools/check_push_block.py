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
# each entry point's call graph, collects every push-block member reached
# through an OpAccessChain, and asks the question that has an answer:
#
#     does the entry point read PAST the bytes the bridge pushes for it?
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
# Byte LAYOUT within the pushed range.  Two structs of equal size can still
# disagree on where a field sits, and this says nothing about that.  It checks
# the one thing that cannot be recovered from downstream: reading bytes that
# were never pushed.
#
# Nor does it check kernels with no ABI row: those are refused outright by the
# shim (pushConstantSizeFor returns 0), which check_launch_abi.py already holds.
#
# ---------------------------------------------------------------------------
# THE RATCHET
# ---------------------------------------------------------------------------
# An entry point that reaches only a PREFIX of its row is not a defect -- it is
# an argument the port defers, and several tranches did so deliberately.  Those
# are reported with their measured slack and carried in push_block_known.txt so
# the number is visible rather than assumed; the list may shrink, never grow.
# Reading PAST the row is never acceptable and is not ratchetable.
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
            if op == OP_ENTRY_POINT:
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

    def member_extent(self, sid, idx):
        """(offset + size) of one struct member, or None when undecidable."""
        t = self.types.get(sid)
        if not t or t[0] != OP_TYPE_STRUCT or idx is None or idx >= len(t[1]):
            return None
        off = self.member_offset.get((sid, idx))
        sz = self.type_size(t[1][idx])
        return None if (off is None or sz is None) else off + sz

    def reach_for_entry(self, name):
        """The highest push-constant BYTE the entry point can read, or (None, note).

        Returns (reach, declared, note): `reach` is max(offset + size) over every
        member reached through the entry point's call graph, `declared` is the
        whole block's size."""
        for ename, fn, iface in self.entries:
            if ename != name:
                continue
            pushes = [v for v in iface if v in self.push_vars]
            if not pushes:
                return 0, 0, "no push-constant variable in the interface"
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
                return declared, declared, "whole-block OpLoad"

            reach = 0
            for f in seen:
                for (v, idx) in self.fn_members.get(f, ()):
                    if v != var:
                        continue
                    if idx is None:
                        return None, declared, "non-constant member index"
                    ext = self.member_extent(sid, idx)
                    if ext is None:
                        return None, declared, "member %d undecidable" % idx
                    reach = max(reach, ext)
            return (reach + 3) & ~3, declared, ""
        return None, None, "entry point not found in the module"


# ---------------------------------------------------------------------------
# the ABI table and the manifest
# ---------------------------------------------------------------------------
def _abi_layout(row):
    """(end of the last argument, total blob size) for an ABI row.

    The second is pxgKernelAbiPushSize (PxgVulkanKernelAbi.cpp) transcribed; the
    FIRST is where a shader can legitimately stop.  The two differ by the
    TRAILING PADDING the blob alignment adds -- bytes the bridge pushes and
    nothing is meant to read.  Comparing a reach against the rounded total
    instead reported 166 "prefix readers" on this checker\'s first run, almost
    all of them exactly four bytes short: padding, not a deferral."""
    def align(s):
        return s if s in (1, 2, 4, 8) else 16
    off, ma = 0, 1
    for s in row:
        a = align(s)
        ma = max(ma, a)
        off = (off + a - 1) & ~(a - 1)
        off += s
    return (off + 3) & ~3, (off + ma - 1) & ~(ma - 1)


def abi_push_size(row):
    return _abi_layout(row)[1]


def abi_last_end(row):
    return _abi_layout(row)[0]


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
    """kernel -> (reach, last).  The RECORDED PAIR is part of the ratchet: a
    prefix reader that starts reading LESS than it used to is a regression the
    bare name would hide."""
    known = {}
    if not os.path.exists(KNOWN):
        return known
    for line in open(KNOWN):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 3:
            known[parts[0]] = (int(parts[1]), int(parts[2]))
        else:
            known[parts[0]] = None
    return known


def main():
    shader_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build-gpuslang", "shaders")
    if not os.path.isdir(shader_dir):
        print("no shader directory: %s" % shader_dir)
        return 2

    rows = read_abi()
    known = read_known()
    modules = {}

    exact = 0
    no_row = 0
    missing_spv = 0
    undecided = []
    over = []      # the entry point can read PAST the pushed bytes
    short = []     # it reaches only a prefix of them

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
        want = abi_push_size(row)
        last = abi_last_end(row)
        reach, declared, note = modules[spv].reach_for_entry(entry)
        if reach is None:
            undecided.append((cuda, spv, entry, note))
            continue
        if reach > want:
            over.append((cuda, entry, reach, want, declared, note))
        elif reach >= last:
            exact += 1          # reaches the last real argument; the rest is padding
        else:
            short.append((cuda, entry, reach, last, want, declared))

    total = exact + len(short) + len(over) + len(undecided)
    print("push-block REACH vs ABI row over %d mapped kernels with a row "
          "(%d without a row, %d with no built .spv)" % (total, no_row, missing_spv))
    print("  EXACT      %d  (reaches the last real argument; trailing padding is not read)" % exact)
    print("  PREFIX     %d  (stops short of the last real argument)" % len(short))
    print("  OVER-READ  %d  (can read past the pushed bytes)" % len(over))
    print("  UNDECIDED  %d" % len(undecided))

    failures = 0

    if over:
        print("\nAn entry point that reaches BEYOND its ABI row reads bytes the bridge")
        print("never pushed.  Nothing reports this at runtime -- see the header:")
        for cuda, entry, reach, want, declared, note in sorted(over):
            print("  %-52s %s reaches byte %d, the row pushes %d (block is %s)%s"
                  % (cuda, entry, reach, want, declared, (" [%s]" % note) if note else ""))
        failures += len(over)

    moved = []
    for cuda, entry, reach, last, want, declared in short:
        rec = known.get(cuda)
        if rec is not None and rec != (reach, last):
            moved.append((cuda, rec, (reach, last)))

    fresh = [x for x in short if x[0] not in known]
    if fresh:
        print("\n%d entry point(s) reach only a PREFIX of their row and are not on the" % len(fresh))
        print("ratchet.  That is usually an argument the port defers on purpose -- record")
        print("each in tools/push_block_known.txt with the reason, or read the tail:")
        for cuda, entry, reach, last, want, declared in sorted(fresh):
            print("  %-46s reaches %d, last argument ends at %d (%d pushed, block %s)"
                  % (cuda, reach, last, want, declared))
        failures += len(fresh)

    if moved:
        print("\n%d recorded prefix reader(s) MOVED.  Reading less than before is a" % len(moved))
        print("regression; reading more means the ratchet line is stale:")
        for cuda, was, now in sorted(moved):
            print("  %-52s was %d/%d, now %d/%d" % (cuda, was[0], was[1], now[0], now[1]))
        failures += len(moved)

    stale = [k for k in known if k not in {x[0] for x in short}]
    if stale:
        print("\n%d STALE entry(ies) in push_block_known.txt -- these kernels no longer" % len(stale))
        print("stop short of their row; remove the lines.  A ratchet that keeps finished")
        print("kernels on its list stops ratcheting:")
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
    print("\ncheck_push_block: clean (%d prefix reader(s) on the ratchet)" % len(short))
    return 0


if __name__ == "__main__":
    sys.exit(main())
