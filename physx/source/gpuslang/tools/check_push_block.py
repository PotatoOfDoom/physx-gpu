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
# THE JOIN: ROW == SHADER LAYOUT (P9.97b)
# ---------------------------------------------------------------------------
# Coverage alone is a necessary condition, not a sufficient one: two structs can
# agree on which bytes are read and still disagree on what lives there.  P9.97b
# found the case that proves it.  sb_sdfMeshMidphaseGeneratePairsLaunch's shader put
# its 80-byte by-value PxgFEMContactWriter at @88 in a 168-byte block while the row
# and the bridge put it at @96 (paramAlign(80) = 16; push 176).  Coverage was FULL
# and NO-MEMBER silent -- every argument was overlapped by SOME member -- while on the
# engine path w_outPoint read bridge padding and every later writer field sat one
# slot off.  The scene that finally reached it segfaulted.
#
# So every reached leaf member of the push block (nested Slang structs recursed) is
# now laid against the row's arguments AS THE BRIDGE PLACES THEM (paramAlign), with
# each by-value aggregate argument FLATTENED into its fields -- from the CUDA
# signature's struct (check_launch_abi_pack.struct_fields), from the vector types, or
# from HOST_ARG_FIELDS for a host-shaped row.  Failures, never ratcheted:
#
#   R0 PAD-READ    a REACHED member that overlaps no argument field: it reads bytes
#                  the bridge filled with zero padding (the P9.97b defect).
#   R1 STRADDLE    a member over more than one ARGUMENT, any of them >= 4 bytes.  A
#                  word over several SUB-WORD arguments is the packed-bool idiom
#                  (solveFlagsWord, boolsWord, extFlagsWord) and stays legal.
#   R2 CUT         a member edge strictly inside a field or a scalar/pointer argument.
#   R5 F-SUBWORD   a FLOAT member over an argument field narrower than 4 bytes (kills
#                  P9.97's M2, the kSapBoxInit {8,1,4} swap, at updateBodiesLaunch*'s
#                  `dt`, without flagging the bool-read-as-a-word idiom).
#
# Report-only: an UNREACHED member that is not named *pad* and lies wholly in padding
# (R0b).  A kernel whose push block or aggregate argument cannot be laid out goes to
# KNOWN_UNDECIDED_JOIN, a ratchet that is EMPTY.  A SELFTEST runs first (M1 -> CUT,
# M2 -> F-SUBWORD, the P9.97b writer@88 -> PAD-READ + CUT, plus synthetic cases).
#
# ---------------------------------------------------------------------------
# WHAT IT DOES *NOT* CHECK
# ---------------------------------------------------------------------------
# A permutation of same-sized fields INSIDE one aggregate (two pointers swapped):
# offsets cannot see names.  The bridge-path fixtures can (testSoftBodySdfMid's
# softBodySdfMidphaseBridgeSized launches the engine's host pack).
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

import collections
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

    def leaf_members(self, name):
        """[(offset, end, dotted name, kind)] of every LEAF member of the entry point's
        push block, nested structs recursed with their parent offsets added; kind is
        'f' for a float scalar/vector/matrix/array, 'i' otherwise.  None when undecidable."""
        for ename, _fn, iface in self.entries:
            if ename != name:
                continue
            pushes = [v for v in iface if v in self.push_vars]
            if len(pushes) != 1:
                return None
            out = []

            def walk(sid, base, prefix):
                t = self.types.get(sid)
                if not t or t[0] != OP_TYPE_STRUCT:
                    return False
                for idx, mt in enumerate(t[1]):
                    off = self.member_offset.get((sid, idx))
                    if off is None:
                        return False
                    nm = prefix + self.member_name.get((sid, idx), "?")
                    mtt = self.types.get(mt)
                    if mtt and mtt[0] == OP_TYPE_STRUCT:
                        if not walk(mt, base + off, nm + "."):
                            return False
                        continue
                    sz = self.type_size(mt)
                    if sz is None or not mtt:
                        return False
                    k = mtt
                    while k and k[0] in (OP_TYPE_VECTOR, OP_TYPE_MATRIX, OP_TYPE_ARRAY):
                        k = self.types.get(k[1][0])
                    out.append((base + off, base + off + sz, nm, "f" if k and k[0] == OP_TYPE_FLOAT else "i"))
                return True

            return out if walk(self.push_vars[pushes[0]], 0, "") else None
        return None

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
# P9.97b: ONE row model for both checkers.  check_launch_abi_pack.read_abi models
# array[:argCount] -- what the bridge uses -- and reports every row it cannot read
# (an unparsed array, a wrong count, a duplicate name) instead of skipping it.  The
# copy that stood here read whole arrays and kept the LAST of two rows with one name,
# where the bridge takes the FIRST.  Sibling import, the check_block_coverage.py
# precedent.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_launch_abi_pack as lap   # noqa: E402

_param_align = lap._param_align


def _abi_arg_ranges(row):
    """[(start, end)] per CUDA argument, laid out exactly as the bridge lays the
    blob out (paramAlign, PxgVulkanCudaContext.cpp sized launchKernel)."""
    return [(o, o + s) for (o, s) in lap.arg_layout(row)[0]]


def abi_push_size(row):
    """pxgKernelAbiPushSize (PxgVulkanKernelAbi.cpp), transcribed."""
    return lap.abi_push_size(row)


def read_abi(text=None):
    """(cuda name -> argument sizes, integrity problems) from the shared row model."""
    rows, bad = lap.read_abi(text)
    return collections.OrderedDict((k, r.sizes) for k, r in rows.items()), bad


# Kernels whose push block or aggregate argument this join cannot lay out.  RATCHET:
# may shrink, never grow.  Empty.
KNOWN_UNDECIDED_JOIN = set()


def arg_fields(name, row, sigs=None):
    """[(start, end, argument index)] -- the row's arguments as the bridge places them,
    every by-value aggregate FLATTENED into its fields; None when an aggregate cannot be
    laid out (an UNDECIDED join, never a pass)."""
    out = []
    shapes = lap.arg_fields(name, row, sigs)
    for i, ((a, b), shape) in enumerate(zip(_abi_arg_ranges(row), shapes)):
        if shape == "UNDECIDED":
            return None
        if shape is None:
            out.append((a, b, i))
        else:
            out.extend((a + fo, a + fo + fs, i) for (fo, fs, _k) in shape)
    return out


def join_check(row, fields, leaves, reached):
    """The layout join for one kernel.  Pure, so the selftest can feed it synthetic
    blocks.  Returns ([(rule, member, detail)] failures, [report-only notes])."""
    bad, notes = [], []
    args = _abi_arg_ranges(row)
    total = abi_push_size(row)
    for (ms, me, nm, kind) in leaves:
        if ms >= total:
            continue   # a shared struct sized for a larger sibling; over-reads are OVER-READ's
        hit_f = [(a, b, i) for (a, b, i) in fields if ms < b and a < me]
        hit_a = sorted(set(i for (a, b, i) in hit_f) |
                       set(i for i, (a, b) in enumerate(args) if ms < b and a < me))
        is_reached = reached is None or any(rs < me and ms < re_ for (rs, re_) in reached)
        if not hit_f:
            if is_reached:
                bad.append(("PAD-READ", nm, "@%d+%d is reached and overlaps no argument field -- it "
                            "reads bytes the bridge pads with zero" % (ms, me - ms)))
            elif "pad" not in nm.lower():
                notes.append("%s @%d+%d lies wholly in padding (unreached)" % (nm, ms, me - ms))
            continue
        if len(hit_a) > 1 and any(args[i][1] - args[i][0] >= 4 for i in hit_a):
            bad.append(("STRADDLE", nm, "@%d+%d spans arguments %s" % (ms, me - ms,
                        ", ".join("%d@%d+%d" % (i, args[i][0], args[i][1] - args[i][0]) for i in hit_a))))
        for (a, b, i) in hit_f:
            if a < ms < b or a < me < b:
                bad.append(("CUT", nm, "@%d+%d cuts argument %d's field @%d+%d" % (ms, me - ms, i, a, b - a)))
        if kind == "f" and any(b - a < 4 for (a, b, _i) in hit_f):
            bad.append(("F-SUBWORD", nm, "float @%d+%d over a %s-byte argument" % (
                ms, me - ms, "/".join(str(b - a) for (a, b, _i) in hit_f if b - a < 4))))
    return bad, notes


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


# ---------------------------------------------------------------------------
# SELFTEST (P9.97b): each case is built so that a join that cannot see it fails here.
# ---------------------------------------------------------------------------
SELFTEST_ROWS = [
    ("M1 vec3 split", "static const uint8_t kTwoPtrVec3[] = { 8, 8, 12 };",
     "static const uint8_t kTwoPtrVec3[] = { 8, 8, 8, 4 };", {"applyTGSSubstepGravity": "CUT"}),
    ("M2 sapbox swap", "static const uint8_t kSapBoxInit[] = { 8, 4, 1 };",
     "static const uint8_t kSapBoxInit[] = { 8, 1, 4 };",
     {"updateBodiesLaunch1T": "F-SUBWORD", "updateBodiesLaunch_Part2": "F-SUBWORD",
      "updateBodiesLaunch_Part3": "F-SUBWORD"}),
]


def _sdf_block(writer_at):
    """sb_sdfMeshMidphaseGeneratePairsLaunch's push block with the writer at `writer_at`."""
    out = [(0, 4, "tolerenceLength", "f"), (4, 8, "_pad4", "i")]
    out += [(8 + 8 * i, 16 + 8 * i, "ptr%d" % i, "i") for i in range(9)]
    out += [(80, 84, "nbFilterPairs", "i"), (84, 88, "_pad84", "i")]
    if writer_at == 96:
        out.append((88, 96, "_pad88", "i"))
    out += [(writer_at + 8 * i, writer_at + 8 + 8 * i, "w_ptr%d" % i, "i") for i in range(9)]
    out += [(writer_at + 72, writer_at + 76, "w_maxNumContacts", "i"),
            (writer_at + 76, writer_at + 80, "_pad%d" % (writer_at + 76), "i")]
    return out


def selftest(shader_dir):
    fails = []
    sigs = lap.cuda_signatures()
    abi_text = open(ABI).read()
    manifest = dict((c, (spv, e)) for (c, spv, e) in read_manifest())
    mods = {}

    def rules_for(rows, name):
        spv, entry = manifest[name]
        if spv not in mods:
            mods[spv] = Module(os.path.join(shader_dir, spv))
        m = mods[spv]
        leaves, flat = m.leaf_members(entry), arg_fields(name, rows[name], sigs)
        if leaves is None or flat is None:
            return None
        return set(r for (r, _m, _d) in join_check(rows[name], flat, leaves,
                                                    m.coverage_for_entry(entry)[0])[0])

    for label, old, new, expect in SELFTEST_ROWS:
        if abi_text.count(old) != 1:
            fails.append("SELFTEST BROKEN: %s -- anchor found %d times; re-anchor it" % (label, abi_text.count(old)))
            continue
        rows, _bad = read_abi(abi_text.replace(old, new))
        for k, rule in expect.items():
            try:
                got = rules_for(rows, k)
            except (OSError, ValueError, KeyError) as e:
                got = None
                fails.append("SELFTEST BROKEN: %s -- %s" % (label, e))
            if got is None or rule not in got:
                fails.append("SELFTEST: %s must give %s at %s; got %s" % (label, rule, k, got))
    # The P9.97b defect itself, synthetic: the writer at @88 over the row's writer at @96.
    name = "sb_sdfMeshMidphaseGeneratePairsLaunch"
    row = [4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 80]
    flat = arg_fields(name, row, sigs)
    if flat is None:
        fails.append("SELFTEST: the FEM writer of %s cannot be flattened" % name)
    else:
        for at in (88, 96):
            blk = _sdf_block(at)
            reached = [(a, b) for (a, b, n, _k) in blk if "pad" not in n]
            got = join_check(row, flat, blk, reached)[0]
            if at == 88:
                rules = set(r for (r, _m, _d) in got)
                if not {"PAD-READ", "CUT"} <= rules:
                    fails.append("SELFTEST: a writer @88 against the row's @96 must give PAD-READ and CUT; "
                                 "got %s" % rules)
            elif got:
                fails.append("SELFTEST: the writer @96 must join cleanly; got %s" % got)
    # STRADDLE over two 4-byte arguments, and the legal packed-bool word.
    got = set(r for (r, _m, _d) in join_check([8, 4, 4], [(0, 8, 0), (8, 12, 1), (12, 16, 2)],
                                               [(0, 8, "p", "i"), (8, 16, "two", "i")], None)[0])
    if "STRADDLE" not in got:
        fails.append("SELFTEST: one member over two 4-byte arguments must give STRADDLE; got %s" % got)
    got = join_check([8, 1, 1, 1, 1], [(0, 8, 0), (8, 9, 1), (9, 10, 2), (10, 11, 3), (11, 12, 4)],
                     [(0, 8, "p", "i"), (8, 12, "flagsWord", "i")], None)[0]
    if got:
        fails.append("SELFTEST: a word over four bool arguments is the packed-bool idiom; got %s" % got)
    # Nested Slang structs are recursed (a top-level-only walk would see one 8-byte member).
    fm = Module.__new__(Module)
    fm.types = {1: (OP_TYPE_INT, [32, 0]), 2: (OP_TYPE_FLOAT, [32]), 3: (OP_TYPE_INT, [64, 0]),
                10: (OP_TYPE_STRUCT, [2, 2]), 11: (OP_TYPE_STRUCT, [3, 10, 1])}
    fm.member_offset = {(10, 0): 0, (10, 1): 4, (11, 0): 0, (11, 1): 8, (11, 2): 16}
    fm.member_name = {(10, 0): "x", (10, 1): "y", (11, 0): "a", (11, 1): "b", (11, 2): "c"}
    fm.push_vars, fm.entries = {99: 11}, [("e", 0, [99])]
    fm.constants, fm.array_stride, fm.matrix_stride = {}, {}, {}
    want = [(0, 8, "a", "i"), (8, 12, "b.x", "f"), (12, 16, "b.y", "f"), (16, 20, "c", "i")]
    if fm.leaf_members("e") != want:
        fails.append("SELFTEST: nested push-block structs must be recursed: %s" % fm.leaf_members("e"))
    return fails


def main():
    shader_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build-gpuslang", "shaders")
    if not os.path.isdir(shader_dir):
        print("no shader directory: %s" % shader_dir)
        return 2

    fails = selftest(shader_dir)
    if fails:
        print("THE PUSH-BLOCK JOIN SELFTEST FAILED -- it cannot see what it claims to see:")
        for f in fails:
            print("  " + f)
        print()

    rows, integrity = read_abi()
    sigs = lap.cuda_signatures()
    known = read_known()
    modules = {}
    joinbad = []         # (cuda, rule, member, detail)
    joinnotes = []
    join_undecided = set()
    joined = 0

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

        # ---- the layout join (P9.97b) ----
        leaves = modules[spv].leaf_members(entry)
        flat = arg_fields(cuda, row, sigs)
        if leaves is None or flat is None:
            join_undecided.add(cuda)
        else:
            joined += 1
            jb, jn = join_check(row, flat, leaves, ranges)
            joinbad += [(cuda, r, m, d) for (r, m, d) in jb]
            joinnotes += ["%s: %s" % (cuda, n) for n in jn]

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

    print("  JOIN       %d kernels laid against their row field by field: %d PAD-READ, %d STRADDLE, "
          "%d CUT, %d F-SUBWORD; %d undecided (ratchet %d)"
          % (joined, sum(1 for x in joinbad if x[1] == "PAD-READ"),
             sum(1 for x in joinbad if x[1] == "STRADDLE"), sum(1 for x in joinbad if x[1] == "CUT"),
             sum(1 for x in joinbad if x[1] == "F-SUBWORD"), len(join_undecided), len(KNOWN_UNDECIDED_JOIN)))

    failures = len(fails)

    if integrity:
        print("\nThe ABI table has rows the shared row model cannot read as the bridge does")
        print("(check_launch_abi_pack.py reports the same, under I1/I4):")
        for name, cat, msg in integrity:
            print("  %-46s %s: %s" % (name, cat, msg))
        failures += len(integrity)

    if joinbad:
        print("\nTHE SHADER'S PUSH BLOCK AND THE ROW DISAGREE ON THE LAYOUT.  The bridge places")
        print("each argument at paramAlign(size) (an 80-byte writer at 16, not at its natural 8);")
        print("a member that reads padding or cuts a field reads the wrong bytes on the engine")
        print("path, while every fixture that dispatches its own blob stays green:")
        for cuda, rule, member, detail in sorted(joinbad):
            print("  %-46s %-9s %s %s" % (cuda, rule, member, detail))
        failures += len(joinbad)
    fresh_und = sorted(join_undecided - KNOWN_UNDECIDED_JOIN)
    if fresh_und:
        print("\n%d kernel(s) whose layout join is UNDECIDED (not a pass; teach struct_fields" % len(fresh_und))
        print("or HOST_ARG_FIELDS the aggregate, or add it to KNOWN_UNDECIDED_JOIN):")
        for k in fresh_und:
            print("  %s" % k)
        failures += len(fresh_und)
    stale_und = sorted(KNOWN_UNDECIDED_JOIN - join_undecided)
    if stale_und:
        print("\nSTALE KNOWN_UNDECIDED_JOIN entries (now decided; remove them): %s" % ", ".join(stale_und))
        failures += len(stale_und)
    if joinnotes:
        print("\nReport only -- unreached members that lie wholly in padding (R0b):")
        for n in sorted(joinnotes):
            print("  " + n)

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
