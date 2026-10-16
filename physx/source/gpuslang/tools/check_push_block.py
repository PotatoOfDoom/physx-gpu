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
#   R3 RUN-ON      (P9.99) a REACHED member that covers a field of >= 4 bytes and runs on
#                  into padding: a uint64_t over a PxU32 + pad, a float2 over a trailing
#                  PxReal.  Value-neutral on the bridge (it zero-fills padding), so it was
#                  P9.97b's documented gap -- but it is a layout the CUDA does not have.
#                  A word over sub-word fields + pad (68 reached and 18 unreached on the
#                  tree, all over 1-, 2- or 3 one-byte bools) is the packed-bool idiom and
#                  stays legal.  0 findings at P9.99; the row mutant kTwoPtr {8,8} -> {4,8}
#                  joined silently at all 37 of its kernels before this rule.
#
# Report-only: an UNREACHED member that is not named *pad* and lies wholly in padding
# (R0b), and an UNREACHED member R3 would flag -- each under its own heading.  A kernel
# whose push block or aggregate argument cannot be laid out goes to KNOWN_UNDECIDED_JOIN,
# a ratchet that is EMPTY.  A SELFTEST runs first (M1 -> CUT, M2 -> F-SUBWORD, M3 ->
# RUN-ON, the P9.97b writer@88 -> PAD-READ + CUT, plus synthetic cases).
#
# ---------------------------------------------------------------------------
# THE NAME LEG (P9.99)
# ---------------------------------------------------------------------------
# Offsets cannot see a permutation of members of EQUAL size -- two top-level pointers, two
# fields of a by-value writer, or a `uint` word over a 1-byte bool and a `uint` over a
# PxU32 swapped in the shader's struct: every offset and every member size stays, and no
# join rule fires.  P9.97b's sweep had two such mutants that only the bridge-sized fixture
# killed.  Names can see them.  Every REACHED leaf member is compared with the CUDA names
# of the fields it covers (signature parameter, dotted by-value field or vector component,
# or HOST_ARG_FIELD_NAMES -- check_launch_abi_pack.arg_field_names, laid out BY OFFSET
# through the row, not by position) and with the CUDA type, after a normalisation
# (camelCase/underscore tokens; prefixes g_ m w_ b_ gp_ d_; the Word/Byte suffix of the
# packed-bool idiom; nb~num, idx~index, cnt, vel, iter; the two UPSTREAM typos gBoftbodies
# and tolerenceLength -- documented, never reported).  Scores SAME > RELATED > UNRELATED.
#
#   N1 NAME-SWAP   a reached member whose own score -- the BEST over the fields it covers,
#                  so a packed word named after any one of its flags is SAME -- is below
#                  SAME, and whose name is RELATED or better, and strictly better, to
#                  ANOTHER named field of ANY size.  Positive evidence only: a mere
#                  mismatch (UNRELATED) is report-only.  Never ratcheted except by a named
#                  `swap` line in push_name_known.txt.
#   N2 NAME-BLIND  the swaps neither the join nor N1 would see, found by simulating them
#                  with BOTH as observers (blind = neither output changes):
#                    (a) SHADER side: every pair of reached members of equal MEMBER size,
#                        whatever fields they cover;
#                    (b) SHADER side: every reached x UNREACHED pair of equal size inside
#                        ONE row argument, the reach moving with the member;
#                    (c) CUDA side (check round): every pair of named CUDA fields of equal
#                        size, at least one under a reached member, exchanging name and
#                        type -- the CUDA declaration (and the host pack with it) reordered
#                        while the shader stays.  No offset, reach or push_block_known
#                        signature moves, so only N1 can see it.  Not simulated: two fields
#                        under ONE member (the byte order inside a packed word, below); not
#                        repeated: two fields under the two members of an (a)/(b) blind pair.
#                  Single members only: a swap of two whole NESTED structs (several leaves
#                  at once) is not simulated -- no push block on the tree has a nested
#                  struct member (P9.99 check round, 0 candidates).
#                  Held by tools/push_name_known.txt, signature checked (MOVED fails);
#                  it may shrink, never grow, and a stale line fails.
#   UNDECIDED      a reached member without an OpMemberName, a reached member over a CUDA
#                  field named `?` (a parameter whose name cannot be read, or a by-value
#                  struct defined twice with one layout and two spellings), a reached member
#                  over a row argument the CUDA does not declare (the row does not follow
#                  the signature there; check round C2 -- 7 kernels carry such an argument,
#                  none reaches it), or two CUDA declarations that agree on the sizes but
#                  not on the names: a failure, never a pass.  A kernel whose layout join is
#                  already undecided is counted there, once.
#
# Measured at P9.99 (415 kernels joined by name), after the check round: 2311 SAME / 91
# RELATED / 91 UNRELATED reached members (the 91 include 15 packed words with generic names
# such as flagsWord); 0 NAME-SWAP; 337 kernels have an equal-size reached pair, 8520 such
# pairs: 96 seen by the join, 8407 by names only, 17 blind in 13 kernels; of the 223
# reached x unreached pairs inside one argument 2 are blind, in 1 kernel; of 9728 CUDA-side
# pairs (335 kernels) 9 are blind, in 4 kernels (3 of them blind on nothing else).  Before
# the check round the CUDA side was not simulated and 17 such pairs in 12 kernels went
# unseen: 8 were made decidable by renaming the shader member to the CUDA name (numTests ->
# numWorkItems, prepDesc -> prepareDesc, clothAppliedForces -> softBodyAppliedForces;
# OpMemberName only), 9 are on the ratchet.  17 kernels on the ratchet with the evidence
# that settles them; 0 undecided.
#
# A RED RUN of the WHOLE checker follows the selftest (e2e_red): boxBoxNphase.spv with two
# member names swapped, in a scratch dir of symlinks, must exit 1 and its boxBoxNphase
# NAME-SWAP lines must name exactly those two members -- the only test that sees main's
# failure accounting (other output is not checked).
#
# ---------------------------------------------------------------------------
# WHAT IT DOES *NOT* CHECK
# ---------------------------------------------------------------------------
# A swap among the NAME-BLIND pairs (the 17 kernels of push_name_known.txt), and any swap of
# USES: the name leg checks the DECLARATION, so code that reads gSb.softbodies where the
# CUDA reads gpuShapes -- or that takes the bytes of a packed word in the wrong order
# (flagsWord byte 0 isSharedPartition, byte 1 isTGS) -- is a fixture's business (the
# bridge-path fixtures launch the engine's pack).
#
# A reached x unreached swap ACROSS arguments in the SHADER is not simulated: it moves the
# unread argument, and push_block_known.txt's signature carries the unread argument's index
# and the member name over it, so it fails there as MOVED.  If that signature ever loses its
# names or its indices, those swaps go blind.  The same swap in the CUDA declaration moves
# nothing the shader side records; it is simulation (c), observed by the names alone.
#
# Nor a swap of two whole nested push-block structs: the simulations exchange single leaf
# members (or single CUDA fields).  None exists on the tree today; a nested struct member
# added to a push block needs a group simulation (equal leaf profile) first.
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
# Usage: python3 check_push_block.py [shader-dir] [--selftest-only]
#        (default: ../../../../build-gpuslang/shaders)
#   --selftest-only  run the selftest alone and print `check_push_block selftest: ok`
#                    (exit 0/1) -- for fast checker mutation sweeps.
#   --no-e2e         INTERNAL: skip the end-to-end red run; only e2e_red's own
#                    subprocess passes it, so it does not recurse.  gpuslang.pushBlock
#                    never does.

import collections
import functools
import itertools
import os
import re
import struct
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
ABI = os.path.join(ROOT, "physx", "source", "gpuslang", "bridge", "PxgVulkanKernelAbi.cpp")
MANIFEST = os.path.join(ROOT, "physx", "source", "gpuslang", "include", "PxgSlangKernelManifest.h")
KNOWN = os.path.join(os.path.dirname(__file__), "push_block_known.txt")
NAME_KNOWN = os.path.join(os.path.dirname(__file__), "push_name_known.txt")

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


# The join's failure rules, in the order the JOIN line counts them.
JOIN_RULES = ("PAD-READ", "STRADDLE", "CUT", "F-SUBWORD", "RUN-ON")

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


def arg_field_names(name, row, sigs=None):
    """[(start, end, argument index, CUDA name, CUDA type)], parallel to arg_fields (P9.99);
    None when either cannot be laid out, the CUDA names are undecided, or the two disagree
    in length (an UNDECIDED name join, never zipped short)."""
    flat = arg_fields(name, row, sigs)
    per = lap.arg_field_names(name, row, sigs)
    if flat is None or per is None:
        return None
    nm = [x for p in per for x in p]
    if len(nm) != len(flat):
        return None
    return [(a, b, i, n, ty) for ((a, b, i), (n, ty)) in zip(flat, nm)]


def join_check(row, fields, leaves, reached):
    """The layout join for one kernel.  Pure, so the selftest can feed it synthetic
    blocks.  Returns ([(rule, member, detail)] failures, [(rule, note)] report-only notes
    -- R0b an unreached member wholly in padding, R3 an unreached RUN-ON)."""
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
                notes.append(("R0b", "%s @%d+%d lies wholly in padding (unreached)" % (nm, ms, me - ms)))
            continue
        # R3 RUN-ON (P9.99): covered bytes per OVERLAP (fields are disjoint), so a member
        # that also cuts a field is measured by what it really covers.
        cov = sum(min(b, me) - max(a, ms) for (a, b, _i) in hit_f)
        if cov < me - ms and any(b - a >= 4 for (a, b, _i) in hit_f):
            what = "@%d+%d covers %s and runs %d byte(s) on into padding" % (
                ms, me - ms, ", ".join("%d@%d+%d" % (i, a, b - a) for (a, b, i) in hit_f), me - ms - cov)
            if is_reached:
                bad.append(("RUN-ON", nm, what))
            else:
                notes.append(("R3", "%s %s (unreached)" % (nm, what)))
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


# ---------------------------------------------------------------------------
# the NAME leg (P9.99)
# ---------------------------------------------------------------------------
NAME_DROP_PREFIX = {"g", "m", "w", "b", "gp", "d"}
NAME_DROP_SUFFIX = {"word", "byte"}
NAME_SYNONYM = {"nb": "num", "idx": "index", "cnt": "count", "vel": "velocity", "iter": "iteration",
                # UPSTREAM TYPOS, documented, never reported: softBody.cu:100 `gBoftbodies`,
                # `tolerenceLength` in some forty signatures
                "boftbodies": "softbodies", "tolerence": "tolerance"}
SAME, RELATED, UNRELATED = 3, 2, 0


def name_tokens(n):
    toks = []
    for part in re.split(r"[._\[\]]+", n or ""):
        toks += [x.lower() for x in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", part)]
    while len(toks) > 1 and toks[0] in NAME_DROP_PREFIX:
        toks = toks[1:]
    while len(toks) > 1 and toks[-1] in NAME_DROP_SUFFIX:
        toks = toks[:-1]
    return [NAME_SYNONYM.get(x, x) for x in toks]


def name_key(n):
    return "".join(name_tokens(n))


def _type_key(ty):
    if not ty:
        return None
    w = re.sub(r"[\*&\[\]]", " ", re.sub(r"<.*>", "", ty)).split()
    return name_key(re.sub(r"^Pxg|^Px", "", w[-1])) if w else None


@functools.lru_cache(maxsize=None)
def name_score(s, c, ctype=None):
    """SAME, RELATED or UNRELATED: shader member name `s` against CUDA name `c` (dotted for
    a by-value field) and CUDA type `ctype`.  Pure; the selftest pins it with NAME_TABLE."""
    if not s or not c:
        return UNRELATED
    a = name_key(s)
    leaf = c.split(".")[-1]
    keys = {name_key(c), name_key(leaf)}
    if a in keys:
        return SAME
    if sorted(name_tokens(s)) in (sorted(name_tokens(c)), sorted(name_tokens(leaf))):
        return SAME                                   # dtInv / invDt
    tk = _type_key(ctype)
    if tk and len(tk) >= 4 and a == tk:
        return SAME                                   # solverCoreDesc / PxgSolverCoreDesc*
    m = re.match(r"^(?:.*\.)?(\w)\w*\.([xyzw])$", c)
    if m and a == (m.group(1) + m.group(2)).lower():
        return SAME                                   # gx / gravity.x
    for b in keys:
        if len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
            return RELATED                            # shapeToRigidRemap / ...Table
    st, ct = name_tokens(s), name_tokens(leaf)
    if len(st) >= 2 and all(x in ct for x in st):
        return RELATED                                # solverDesc / solverCoreDesc
    return UNRELATED


def _reached(ms, me, reached):
    return reached is None or any(rs < me and ms < re_ for (rs, re_) in reached)


def _name_members(leaves, fields, reached, total):
    """[(member name, (field indices))] for every REACHED leaf below the pushed size whose
    every overlapped field is named.  A member over SEVERAL fields counts: a word over a
    bool (or over four) trades places with a PxU32 without any size changing, so the
    packed-bool idiom is exactly where the name leg is needed."""
    out = []
    for (ms, me, nm, _k) in leaves:
        if ms >= total or not _reached(ms, me, reached):
            continue
        hit = tuple(j for j, f in enumerate(fields) if ms < f[1] and f[0] < me)
        if hit and all(fields[j][3] is not None for j in hit):
            out.append((nm, hit))
    return out


def name_undecided(leaves, fields, reached, total):
    """True when a REACHED member below the pushed size has no usable OpMemberName, or lies
    over a CUDA field whose name is '?' (a parameter, or the fields of a by-value struct,
    check_launch_abi_pack could not name), or over a row argument the CUDA does not declare
    (name None: the row does not follow the signature there): an UNDECIDED name join, never a
    pass -- and never a member silently left out of the name check."""
    for (ms, me, nm, _k) in leaves:
        if ms >= total or not _reached(ms, me, reached):
            continue
        if not nm or any(p in ("", "?") for p in nm.split(".")):
            return True
        if any(f[3] is None or "?" in f[3].split(".") for f in fields if ms < f[1] and f[0] < me):
            return True
    return False


def _name_fires(nm, hit, fields, thr):
    """(own score, other field, its score) when member `nm`, laid over the fields `hit`, is
    named at least `thr` -- and strictly better -- after ANOTHER named field than after the
    best of its own; else None.  Fields of ANY size are candidates."""
    own = max(name_score(nm, fields[j][3], fields[j][4]) for j in hit)
    if own == SAME:
        return None
    for j, g in enumerate(fields):
        if j in hit or g[3] is None:
            continue
        o = name_score(nm, g[3], g[4])
        if o >= thr and o > own:
            return (own, g, o)
    return None


def name_check(leaves, fields, reached, total, thr=RELATED):
    """N1 NAME-SWAP.  fields: [(start, end, argument index, CUDA name, CUDA type)]."""
    bad = []
    for (nm, hit) in _name_members(leaves, fields, reached, total):
        f = _name_fires(nm, hit, fields, thr)
        if f:
            own, g, o = f
            a, b = fields[hit[0]][0], fields[hit[-1]][1]
            bad.append(("NAME-SWAP", nm, "@%d+%d is CUDA `%s` (score %d) but is named like `%s` @%d (score %d)"
                        % (a, b - a, "/".join(fields[j][3] for j in hit), own, g[3], g[0], o)))
    return bad


def _observe(row, fields, leaves, reached, total):
    """What the join and N1 report for one placement of the members, as comparable data --
    the whole findings, offsets included, so a swap that only MOVES a finding a kernel
    already has (a shader that is already wrong) is seen, not blind."""
    flat = [(a, b, i) for (a, b, i, _n, _t) in fields]
    return (sorted(join_check(row, flat, leaves, reached)[0]), sorted(name_check(leaves, fields, reached, total)))


def _minus(ranges, a, b):
    """The byte ranges with [a, b) taken out."""
    out = []
    for (s, e) in ranges:
        if e <= a or b <= s:
            out.append((s, e))
            continue
        if s < a:
            out.append((s, a))
        if b < e:
            out.append((b, e))
    return out


def name_blind(row, leaves, fields, reached, total):
    """N2: (reached pairs, intra-argument pairs, CUDA-side pairs), each sorted -- the swaps no
    size rule can exclude and NEITHER observer, the join nor N1, would see:
      (a) every pair of REACHED members of equal MEMBER size, whatever fields they cover;
      (b) every REACHED x UNREACHED pair of equal size inside ONE row argument, the reach
          moving with the member.  A SHADER-side pair across arguments is not simulated: it
          moves the push_block_known.txt signature, which carries the unread argument's index
          and the member name over it;
      (c) every pair of named CUDA FIELDS of equal size, at least one under a reached member,
          exchanging (name, type) -- a swap in the CUDA declaration (and the host pack with
          it): the shader, its reach and every offset stay, so only N1 can see it.  Left out:
          two fields under ONE member (the byte order inside a packed word, documented), and
          two fields each under exactly one member when those two members are already an (a)
          or (b) blind pair (the same swap, seen from the CUDA side).
    (a) and (b) are 'a@off/b@off' over member names, (c) is 'cuda:x@off/y@off' over CUDA
    field names.  Single leaves only: a swap of two whole nested structs is not simulated
    (no push block on the tree has a nested struct member)."""
    base = _observe(row, fields, leaves, reached, total)
    args = _abi_arg_ranges(row)
    live = [x for x, lf in enumerate(leaves) if lf[0] < total]
    reach = dict((x, _reached(leaves[x][0], leaves[x][1], reached)) for x in live)

    def arg_of(lf):
        return tuple(i for i, (a, b) in enumerate(args) if lf[0] < b and a < lf[1])

    pa, pb, pc = [], [], []
    blind_members = set()
    for x, y in itertools.combinations(live, 2):
        if not reach[x]:
            x, y = y, x                        # x is the reached one, if either is
        a, b = leaves[x], leaves[y]
        if not reach[x] or a[1] - a[0] != b[1] - b[0] or a[2] == b[2]:
            continue
        if reach[y]:
            r2, out = reached, pa
        elif arg_of(a) and arg_of(a) == arg_of(b):
            r2, out = _minus(reached, a[0], a[1]) + [(b[0], b[1])], pb
        else:
            continue
        lv = list(leaves)
        lv[x], lv[y] = (a[0], a[1], b[2], b[3]), (b[0], b[1], a[2], a[3])
        if _observe(row, fields, lv, r2, total) == base:
            out.append("%s@%d/%s@%d" % (a[2], a[0], b[2], b[0]))
            blind_members.add(frozenset((x, y)))
    # (c) the CUDA side
    owners = [set(x for x in live if leaves[x][0] < f[1] and f[0] < leaves[x][1]) for f in fields]
    for i, j in itertools.combinations(range(len(fields)), 2):
        fi, fj = fields[i], fields[j]
        if fi[1] - fi[0] != fj[1] - fj[0] or fi[3] is None or fj[3] is None or fi[3] == fj[3]:
            continue
        if not any(reach[x] for x in owners[i] | owners[j]) or owners[i] & owners[j]:
            continue
        if len(owners[i]) == 1 and len(owners[j]) == 1 and frozenset(owners[i] | owners[j]) in blind_members:
            continue
        f2 = list(fields)
        f2[i], f2[j] = fi[:3] + (fj[3], fj[4]), fj[:3] + (fi[3], fi[4])
        if _observe(row, f2, leaves, reached, total) == base:
            pc.append("cuda:%s@%d/%s@%d" % (fi[3], fi[0], fj[3], fj[0]))
    return sorted(pa), sorted(pb), sorted(pc)


def name_stats(leaves, fields, reached, total):
    """Report-only numbers for the NAMES line: (Counter of each name-joined member's own
    score, whether two REACHED members below the pushed size have the same size)."""
    stat = collections.Counter(max(name_score(nm, fields[j][3], fields[j][4]) for j in hit)
                               for (nm, hit) in _name_members(leaves, fields, reached, total))
    sizes = collections.Counter(me - ms for (ms, me, _n, _k) in leaves
                                if ms < total and _reached(ms, me, reached))
    return stat, any(v > 1 for v in sizes.values())


def name_leg_kernel(row, leaves, named, reached):
    """The NAME leg for one kernel, pure: ('UNDECIDED', None, None) or
    ('OK', [(rule, member, detail)] NAME-SWAP findings, (reached pairs, intra pairs))."""
    total = abi_push_size(row)
    if named is None or name_undecided(leaves, named, reached, total):
        return ("UNDECIDED", None, None)
    return ("OK", name_check(leaves, named, reached, total), name_blind(row, leaves, named, reached, total))


def name_leg_entry(cuda, row, leaves, reached, sigs):
    """main's NAME-leg call for one join-decided kernel: the CUDA names from the signature(s)
    in `sigs`, then name_leg_kernel -- names that cannot be read (None) are UNDECIDED there,
    never an OK the selftest could not see (NS12b runs through this)."""
    return name_leg_kernel(row, leaves, arg_field_names(cuda, row, sigs), reached)


def read_name_known(path=None):
    """({kernel: blind signature}, {(kernel, member)}) from push_name_known.txt:
         blind <kernel> <a@off/b@off;...>   # why
         swap  <kernel> <member>            # why"""
    blind, swap = {}, set()
    p = NAME_KNOWN if path is None else path
    if not os.path.exists(p):
        return blind, swap
    for line in open(p):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        w = line.split()
        if w[0] == "blind" and len(w) == 3 and w[1] not in blind:
            blind[w[1]] = w[2]
        elif w[0] == "swap" and len(w) == 3 and (w[1], w[2]) not in swap:
            swap.add((w[1], w[2]))
        else:
            raise ValueError("push_name_known.txt: unparsed or duplicate line `%s`" % line)
    return blind, swap


def name_leg_failures(per_kernel, known_blind, known_swap):
    """[(category, kernel, detail)], every one a failure, from {kernel: name_leg_kernel(...)}:
    NAME-UNDECIDED; a NAME-SWAP no `swap` line waives; a FRESH NAME-BLIND kernel; a
    NAME-BLIND-MOVED signature; a STALE line of either kind (an UNDECIDED kernel's lines are
    not judged -- it already fails)."""
    out, undecided, now_blind, now_swap = [], set(), {}, []
    for k, (st, swaps, blind) in sorted(per_kernel.items()):
        if st != "OK":
            undecided.add(k)
            out.append(("NAME-UNDECIDED", k, "a reached member without an OpMemberName, or over a "
                        "CUDA field whose name cannot be read or that the CUDA does not declare"))
            continue
        now_swap += [(k, m, d) for (_r, m, d) in swaps]
        sig = ";".join(sorted(blind[0] + blind[1] + blind[2]))
        if sig:
            now_blind[k] = sig
    for k, sig in sorted(now_blind.items()):
        if k not in known_blind:
            out.append(("NAME-BLIND", k, sig))
        elif known_blind[k] != sig:
            out.append(("NAME-BLIND-MOVED", k, "was %s, now %s" % (known_blind[k], sig)))
    for k in sorted(set(known_blind) - set(now_blind) - undecided):
        out.append(("NAME-BLIND-STALE", k, known_blind[k]))
    for (k, m, d) in sorted(now_swap):
        if (k, m) not in known_swap:
            out.append(("NAME-SWAP", k, "%s %s" % (m, d)))
    for (k, m) in sorted(known_swap - set((k, m) for (k, m, _d) in now_swap)):
        if k not in undecided:
            out.append(("NAME-SWAP-STALE", k, m))
    return out


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
    # P9.99 R3: a pointer row entry narrowed to 4 -- the shader's 8-byte member now covers a
    # PxU32 and 4 pad bytes; no other join rule sees it (measured: all 37 kTwoPtr users)
    ("M3 ptr narrowed", "static const uint8_t kTwoPtr[] = { 8, 8 };",
     "static const uint8_t kTwoPtr[] = { 4, 8 };", {"ZeroBodies": "RUN-ON", "addAggPairsStage1": "RUN-ON"}),
]

# P9.99: the literal oracle of name_score -- every SAME/RELATED rule by a case the tree has,
# and every false positive the prototype met as an UNRELATED guard.
NAME_TABLE = [
    ("w_outPoint", "writer.outPoint", "float4*", 3), ("isTGSWord", "isTGS", "bool", 3),
    ("gravityX", "gravity.x", "PxVec3", 3), ("gx", "gravity.x", "PxVec3", 3),
    ("dtInv", "invDt", "PxReal", 3), ("solverCoreDesc", "constraintPrepDesc", "PxgSolverCoreDesc*", 3),
    ("prePrepDesc", "preDesc", "PxgPrePrepDesc*", 3), ("softBodies", "gBoftbodies", "PxgSoftBody*", 3),
    ("toleranceLength", "tolerenceLength", "PxReal", 3), ("startBit", "gStartBit", "PxU32", 3),
    ("velIterWord", "velocityIteration", "PxU32", 3), ("nbContacts", "numContacts", "PxU32", 3),
    ("shapeToRigidRemap", "shapeToRigidRemapTable", "PxNodeIndex*", 2),
    ("restDistance", "restDistances", "PxReal*", 2), ("solverDesc", "solverCoreDesc", "PxgSolverCoreDesc*", 2),
    ("w_contactIndexSortedByParticle", "contactSortedByParticle", "PxU64*", 0),   # ps_convexCollisionLaunch
    ("prepDesc", "preDesc", "PxgPrePrepDesc*", 0),                                 # sb_rigidAttachmentPrepare
    ("p0", "aggDesc", "PxgAggregateDesc*", 0), ("numTests", "numWorkItems", "PxU32", 0),
    ("maxN", "operation", "PxArticulationGPUAPIWriteType::Enum", 0),
    ("dt", "invDt", "PxReal", 0),
    # check-round guards: the RELATED substring rule needs 4 characters on both sides (a bare
    # `num` is not every num*); a name made only of droppable prefixes keeps its token (it is
    # never emptied into a match with another such name); a by-value argument's BARE name is
    # none of its fields
    ("num", "numContacts", "PxU32", 0), ("g", "d", "PxReal", 0), ("arg", "arg.count", "PxU32", 0),
]


def _swap_member_names(m, entry, a, b):
    """Swap two member NAMES of the entry point's push struct in a parsed Module -- the
    checker's view of a shader whose two declarations were swapped (P9.99 measured it equal
    to the recompiled mutant, leaf for leaf)."""
    for ename, _fn, iface in m.entries:
        if ename != entry:
            continue
        sid = [m.push_vars[v] for v in iface if v in m.push_vars][0]
        ka = [k for k, v in m.member_name.items() if k[0] == sid and v == a]
        kb = [k for k, v in m.member_name.items() if k[0] == sid and v == b]
        if len(ka) == 1 and len(kb) == 1:
            m.member_name[ka[0]], m.member_name[kb[0]] = b, a
            return True
    return False


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
    rows_now = read_abi(abi_text)[0]
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

    # ---- P9.99 R3 RUN-ON, synthetic ----
    def jr(row, fl, lv, rc=None):
        return set(r for (r, _m, _d) in join_check(row, fl, lv, rc)[0])
    f848 = [(0, 8, 0), (8, 12, 1), (16, 24, 2)]
    if "RUN-ON" not in jr([8, 4, 8], f848, [(0, 8, "p", "i"), (8, 16, "n64", "i"), (16, 24, "q", "i")]):
        fails.append("SELFTEST RO1: an 8-byte member over a 4-byte argument + pad must give RUN-ON")
    if jr([8, 4, 8], f848, [(0, 8, "p", "i"), (8, 12, "n", "i"), (16, 24, "q", "i")]):
        fails.append("SELFTEST RO2: the exact layout must join cleanly")
    if jr([8, 1], [(0, 8, 0), (8, 9, 1)], [(0, 8, "p", "i"), (8, 12, "isTGSWord", "i")]):
        fails.append("SELFTEST RO3: a word over one bool + pad is the packed-bool idiom")
    if jr([8, 4, 8], f848, [(0, 8, "p", "i"), (8, 16, "n64", "i"), (16, 24, "q", "i")], [(0, 8), (16, 24)]) or \
            [r for (r, _n) in join_check([8, 4, 8], f848, [(0, 8, "p", "i"), (8, 16, "n64", "i"),
                                                           (16, 24, "q", "i")], [(0, 8), (16, 24)])[1]] != ["R3"]:
        fails.append("SELFTEST RO4: an UNREACHED run-on is report-only, an R3 note")
    if "RUN-ON" not in jr([8, 8, 4], [(0, 8, 0), (8, 16, 1), (16, 20, 2)],
                          [(0, 8, "a", "i"), (8, 16, "b", "i"), (16, 24, "v", "f")]):
        fails.append("SELFTEST RO5: a float2 over a trailing PxReal + tail pad must give RUN-ON")
    if "RUN-ON" not in jr([8], [(0, 4, 0), (4, 5, 0)], [(0, 8, "abWide", "i")]):
        fails.append("SELFTEST RO8: a u64 over a by-value {PxU32, bool} + pad must give RUN-ON")
    if not {"CUT", "RUN-ON"} <= jr([8], [(0, 8, 0)], [(4, 12, "late", "i")]):
        fails.append("SELFTEST RO9: a member @4+8 over a field @0+8 and 4 pad bytes must give CUT and RUN-ON")
    # the real SDF block with nbFilterPairs widened over its PxU32 + pad (compiled: P9.99 mC)
    if flat is not None:
        blk = [(80, 88, "nbFilterPairs", "i") if x[2] == "nbFilterPairs" else x
               for x in _sdf_block(96) if x[2] != "_pad84"]
        if "RUN-ON" not in jr(row, flat, blk, [(a, b) for (a, b, n, _k) in blk if "pad" not in n]):
            fails.append("SELFTEST RO7: the SDF block with a 64-bit nbFilterPairs must give RUN-ON")

    # ---- P9.99 the NAME leg ----
    for s, c, ty, want in NAME_TABLE:
        if name_score(s, c, ty) != want:
            fails.append("SELFTEST NAME_TABLE: %s ~ %s (%s) must score %d, got %d" % (s, c, ty, want, name_score(s, c, ty)))
    fl = [(0, 8, 0, "aggDesc", "PxgAggregateDesc*"), (8, 16, 1, "bpDesc", "PxgBroadPhaseDesc*")]
    # (the CUDA-side swap of the two fields under a blind (a) pair is that same pair: not repeated)
    if name_blind([8, 8], [(0, 8, "p0", "i"), (8, 16, "p1", "i")], fl, None, 16) != (["p0@0/p1@8"], [], []):
        fails.append("SELFTEST NS4: positional p0/p1 must be name-blind")
    if name_blind([8, 8], [(0, 8, "aggDesc", "i"), (8, 16, "bpDesc", "i")], fl, None, 16) != ([], [], []):
        fails.append("SELFTEST NS4: named members must be decidable")
    if name_blind([8, 8], [(0, 8, "aggDesc", "i"), (8, 16, "p1", "i")], fl, None, 16) != ([], [], []):
        fails.append("SELFTEST NS4b: a pair with one named side is decidable")
    if name_blind([8, 4], [(0, 8, "p0", "i"), (8, 12, "n", "i")],
                  [(0, 8, 0, "aggDesc", "PxgAggregateDesc*"), (8, 12, 1, "count", "PxU32")], None, 12) != ([], [], []):
        fails.append("SELFTEST NS4c: members of different sizes are never a blind pair")
    # NS4d: a word over a bool and a PxU32 trade places with no size changing -- blind when the
    # names say nothing, although the two FIELDS differ in size
    if name_blind([4, 1], [(0, 4, "a", "i"), (4, 8, "b", "i")],
                  [(0, 4, 0, "x", "PxU32"), (4, 5, 1, "y", "bool")], None, 8) != (["a@0/b@4"], [], []):
        fails.append("SELFTEST NS4d: an equal-size pair over fields of different sizes must be simulated")
    # NS4e: ... unless the JOIN sees it: the float lands on the bool (F-SUBWORD)
    if name_blind([4, 1], [(0, 4, "u", "f"), (4, 8, "v", "i")],
                  [(0, 4, 0, "x", "PxReal"), (4, 5, 1, "y", "bool")], None, 8) != ([], [], []):
        fails.append("SELFTEST NS4e: a swap the join sees (F-SUBWORD) is not blind")
    # NS4h: ... but a float and an int member over two 4-byte fields trade places unseen: the
    # join has no rule for a float over a PxU32 (kinds are not compared), the names say nothing
    if name_blind([4, 4], [(0, 4, "u", "f"), (4, 8, "v", "i")],
                  [(0, 4, 0, "x", "PxU32"), (4, 8, 1, "y", "PxReal")], None, 8) != (["u@0/v@4"], [], []):
        fails.append("SELFTEST NS4h: a float/int pair of equal size the join cannot see must be name-blind")
    # NS4f: REACHED x UNREACHED inside ONE by-value argument -- reach moves with the member
    # (the unread `alloc` is named after the field it would land on: were the reach left behind,
    # N1 would flag it there and the pair would wrongly read as seen)
    if name_blind([16], [(0, 8, "raw", "i"), (8, 16, "alloc", "i")],
                  [(0, 8, 0, "s.mPtr", "T*"), (8, 16, 0, "s.mAlloc", "U*")], [(0, 8)],
                  16) != ([], ["raw@0/alloc@8"], []):
        fails.append("SELFTEST NS4f: a reached x unreached pair inside one argument must be simulated")
    # NS4i: the same pair over TWO arguments.  Swapped in the SHADER it moves push_block_known's
    # signature (not simulated); swapped in the CUDA declaration nothing moves, and with names
    # that say nothing N1 is silent too -- (c), the check-round C1 gap
    if name_blind([8, 8], [(0, 8, "raw", "i"), (8, 16, "alloc", "i")],
                  [(0, 8, 0, "mPtr", "T*"), (8, 16, 1, "mAlloc", "U*")], [(0, 8)],
                  16) != ([], [], ["cuda:mPtr@0/mAlloc@8"]):
        fails.append("SELFTEST NS4i: a CUDA-side reached x unreached swap across arguments must be simulated")
    if name_blind([8, 8], [(0, 8, "prepareDesc", "i"), (8, 16, "sharedDesc", "i")],
                  [(0, 8, 0, "prepareDesc", "T*"), (8, 16, 1, "sharedDesc", "U*")], [(0, 8)], 16) != ([], [], []):
        fails.append("SELFTEST NS4i: a CUDA-side swap of a member named after its field is decidable")
    # NS4j: two flags under ONE packed word swapped in the CUDA: the byte order inside a word,
    # documented, not a blind pair
    if name_blind([1, 1], [(0, 4, "flagsWord", "i")], [(0, 1, 0, "a", "bool"), (1, 2, 1, "b", "bool")],
                  None, 4) != ([], [], []):
        fails.append("SELFTEST NS4j: two fields under one member are the documented byte-order limitation")
    # NS4k: two CUDA fields with the SAME name -- exchanging them is no swap at all
    if name_blind([8, 8], [(0, 8, "p", "i"), (8, 16, "q", "i")], [(0, 8, 0, "p", "T*"), (8, 16, 1, "p", "T*")],
                  [(0, 8)], 16) != ([], [], []):
        fails.append("SELFTEST NS4k: two CUDA fields of one name are not a CUDA-side pair")
    # NS4g: in a kernel that ALREADY has two NAME-SWAPs, a swap that only MOVES a finding is
    # still seen (the observer compares whole findings, offsets included, not member names)
    if name_blind([8, 8, 8], [(0, 8, "bpDesc", "i"), (8, 16, "aggDesc", "i"), (16, 24, "q", "i")],
                  fl + [(16, 24, 2, "count", "PxU32*")], None, 24) != ([], [], []):
        fails.append("SELFTEST NS4g: a swap that moves an existing NAME-SWAP finding is not blind")
    swapped = [(0, 8, "bpDesc", "i"), (8, 16, "aggDesc", "i")]
    if name_check(swapped, fl, [], 16):
        fails.append("SELFTEST NS5: an UNREACHED swap must not fail")
    if len(name_check(swapped, fl, [(0, 16)], 16)) != 2:
        fails.append("SELFTEST NS5: a reached swap must give NAME-SWAP on both members")
    # NS6': a word over a bool swapped with a PxU32 -- the shape P9.99's first draft could not see
    f6 = [(0, 8, 0, "p", "T*"), (8, 12, 1, "islandIndex", "PxU32"), (12, 13, 2, "doFriction", "bool")]
    got = sorted(m for (_r, m, _d) in name_check([(0, 8, "p", "i"), (8, 12, "doFrictionWord", "i"),
                                                  (12, 16, "islandIndex", "i")], f6, None, 16))
    if got != ["doFrictionWord", "islandIndex"]:
        fails.append("SELFTEST NS6': a word-over-bool / PxU32 swap must give NAME-SWAP on both; got %s" % got)
    if name_check([(0, 8, "p", "i"), (8, 12, "islandIndex", "i"), (12, 16, "doFrictionWord", "i")], f6, None, 16):
        fails.append("SELFTEST NS6': the unswapped layout must be name-clean")
    if name_check([(0, 8, "restDistance", "i"), (8, 16, "x", "i")],
                  [(0, 8, 0, "restDistances", "PxReal*"), (8, 16, 1, "restDistancesScaled", "PxReal*")], None, 16):
        fails.append("SELFTEST NS7: a TIE with another field is not evidence")
    if not name_check([(0, 8, "restDistance", "i"), (8, 16, "contactDistance", "i")],
                      [(0, 8, 0, "contactDistances", "PxReal*"), (8, 16, 1, "restDistances", "PxReal*")], None, 16):
        fails.append("SELFTEST NS8: a swap seen only through RELATED names must give NAME-SWAP")
    f9 = [(0, 8, 0, "p", "T*")]
    if not name_undecided([(0, 8, "?", "i")], f9, None, 8) or name_undecided([(0, 8, "?", "i")], f9, [], 8):
        fails.append("SELFTEST NS9: a REACHED member without a name makes the name join UNDECIDED")
    # NS9b: through the pure per-kernel and failure functions main uses
    st = name_leg_kernel([8], [(0, 8, "?", "i")], f9, None)
    nf = name_leg_failures({"k": st}, {}, set())
    if st[0] != "UNDECIDED" or [c for (c, _k, _d) in nf] != ["NAME-UNDECIDED"]:
        fails.append("SELFTEST NS9b: an unnamed reached member must give NAME-UNDECIDED; got %s / %s" % (st, nf))
    # NS9c: a REACHED member over a row argument the CUDA does not declare (name None) is
    # UNDECIDED, never skipped; unreached, it is not judged
    f9c = [(0, 8, 0, "ptr", "T*"), (8, 12, 1, None, None)]
    if name_leg_kernel([8, 4], [(0, 8, "ptr", "i"), (8, 12, "bogusName", "i")], f9c, None)[0] != "UNDECIDED" or \
            name_leg_kernel([8, 4], [(0, 8, "ptr", "i"), (8, 12, "bogusName", "i")], f9c, [(0, 8)])[0] != "OK":
        fails.append("SELFTEST NS9c: a reached member over an undeclared CUDA argument must be NAME-UNDECIDED")
    # NS10: a word over two flags named after its SECOND flag is SAME (own = the best of its
    # fields), even with a RELATED decoy elsewhere
    if name_check([(0, 4, "isXWord", "i"), (4, 8, "n", "i")],
                  [(0, 1, 0, "isTGS", "bool"), (1, 2, 1, "isX", "bool"), (4, 8, 2, "isXFlag", "PxU32")], None, 8):
        fails.append("SELFTEST NS10: a word over several flags is the packed-bool idiom, not a swap")
    nf = name_leg_failures({"k": ("OK", [], (["a@0/b@8"], [], []))}, {"k": "a@0/b@8", "gone": "x@0/y@8"}, set())
    if [c for (c, _k, _d) in nf] != ["NAME-BLIND-STALE"]:
        fails.append("SELFTEST NS11: a stale blind line must fail; got %s" % nf)
    nf = name_leg_failures({"k": ("OK", [], (["a@0/c@16"], [], []))}, {"k": "a@0/b@8"}, set())
    if [c for (c, _k, _d) in nf] != ["NAME-BLIND-MOVED"]:
        fails.append("SELFTEST NS11: a MOVED blind signature must fail; got %s" % nf)
    nf = name_leg_failures({"k": ("OK", [("NAME-SWAP", "m", "d")], ([], [], []))}, {}, {("k", "m"), ("k", "old")})
    if sorted(c for (c, _k, _d) in nf) != ["NAME-SWAP-STALE"]:
        fails.append("SELFTEST NS11: a waived swap passes, a stale waiver fails; got %s" % nf)
    for bl in (([], ["r@0/u@8"], []), ([], [], ["cuda:x@0/y@8"])):
        nf = name_leg_failures({"k": ("OK", [], bl)}, {}, set())
        if [c for (c, _k, _d) in nf] != ["NAME-BLIND"]:
            fails.append("SELFTEST NS11b: a blind kernel (%s) that is not on the ratchet must fail; got %s"
                         % (bl, nf))
    # NS0-NS3 on the REAL shader and the REAL CUDA text (P9.97b's two surviving mutants and
    # their top-level sibling, measured red on recompiled shaders at P9.99)
    K = "sb_sdfMeshMidphaseGeneratePairsLaunch"
    try:
        spv, entry = manifest[K]
        total = abi_push_size(rows_now[K])
        named = arg_field_names(K, rows_now[K], sigs)
        base = Module(os.path.join(shader_dir, spv))
        rc = base.coverage_for_entry(entry)[0]
        if named is None or name_leg_kernel(rows_now[K], base.leaf_members(entry), named, rc) != \
                ("OK", [], ([], [], [])):
            fails.append("SELFTEST NS0: the tree's %s must be name-clean" % K)
        for a, b in (("w_outPoint", "w_outNormalPen"), ("gpuShapes", "softbodies")):
            m = Module(os.path.join(shader_dir, spv))
            if not _swap_member_names(m, entry, a, b):
                fails.append("SELFTEST BROKEN: NS1/2 -- members %s/%s not found; re-anchor" % (a, b))
                continue
            got = sorted(x[1] for x in name_check(m.leaf_members(entry), named, rc, total))
            if got != sorted((a, b)):
                fails.append("SELFTEST NS1/2: swapping %s/%s must give NAME-SWAP on both; got %s" % (a, b, got))
        cu = os.path.join(ROOT, "physx", "source", "gpunarrowphase", "src", "CUDA", "softbodySoftbodyMidPhase.cu")
        src = open(cu).read()
        at = src.find("void %s(" % K)
        win = src[at:at + 600]
        old = "\tconst PxgShape* PX_RESTRICT gpuShapes,\n\tconst PxgSoftBody* PX_RESTRICT softbodies,\n"
        new = "\tconst PxgShape* PX_RESTRICT softbodies,\n\tconst PxgSoftBody* PX_RESTRICT gpuShapes,\n"
        if at < 0 or win.count(old) != 1:
            fails.append("SELFTEST BROKEN: NS3 -- CUDA anchor not found once; re-anchor")
        else:
            mut = lap.signatures_in(src[:at] + win.replace(old, new) + src[at + 600:], "selftest:" + K)
            rel, line, _p = mut[K][0]
            nms = lap._SIG_NAMES[(rel, line)]
            named3 = [(a, b, i, (nms[i] if n is not None and "." not in n else n), ty)
                      for (a, b, i, n, ty) in named]
            got = sorted(x[1] for x in name_check(base.leaf_members(entry), named3, rc, total))
            if got != ["gpuShapes", "softbodies"]:
                fails.append("SELFTEST NS3: a CUDA-side name swap must give NAME-SWAP; got %s" % got)
        # NS3c: the check-round C1 proof -- sb_rigidAttachmentPrepareLaunch with the CUDA's
        # prepareDesc / sharedDesc parameters swapped (softBodyGM.cu, equal-size pointers): no
        # offset, reach or signature moves, so only the names of the reached members can see
        # it.  Until tranche C of task 35 the shader never read sharedDesc and the swap showed
        # on prepareDesc alone; the articulation arm reads it now, so BOTH members must flag
        K3c = "sb_rigidAttachmentPrepareLaunch"
        cu3 = os.path.join(ROOT, "physx", "source", "gpusimulationcontroller", "src", "CUDA", "softBodyGM.cu")
        src3 = open(cu3).read()
        at3 = src3.find("void %s(" % K3c)
        end3 = src3.find(")", at3)
        lines3 = src3[at3:end3].split("\n")
        ip = [n for n, l in enumerate(lines3) if re.search(r"\bprepareDesc,", l)]
        ish = [n for n, l in enumerate(lines3) if re.search(r"\bsharedDesc,", l)]
        if at3 < 0 or len(ip) != 1 or len(ish) != 1:
            fails.append("SELFTEST BROKEN: NS3c -- CUDA anchor not found once; re-anchor")
        else:
            lines3[ip[0]], lines3[ish[0]] = lines3[ish[0]], lines3[ip[0]]
            mut3 = lap.signatures_in(src3[:at3] + "\n".join(lines3) + src3[end3:], "selftest:" + K3c)
            fake3 = dict(sigs)
            fake3[K3c] = mut3[K3c]
            spv3, entry3 = manifest[K3c]
            m3 = Module(os.path.join(shader_dir, spv3))
            lv3, rc3 = m3.leaf_members(entry3), m3.coverage_for_entry(entry3)[0]
            if name_leg_entry(K3c, rows_now[K3c], lv3, rc3, sigs) != ("OK", [], ([], [], [])):
                fails.append("SELFTEST NS3c: the tree's %s must be name-clean and decidable" % K3c)
            st3 = name_leg_entry(K3c, rows_now[K3c], lv3, rc3, fake3)
            if st3[0] != "OK" or sorted(x[1] for x in st3[1]) != ["prepareDesc", "sharedDesc"]:
                fails.append("SELFTEST NS3c: %s with the CUDA's prepareDesc/sharedDesc swapped must give "
                             "NAME-SWAP on prepareDesc and sharedDesc; got %s" % (K3c, st3))
        # NS12b: a second declaration with the same sizes and renamed parameters -> UNDECIDED
        decl = sigs[K][0]
        fake = dict(sigs)
        fake[K] = list(sigs[K]) + [("selftest:fwd", 1, decl[2])]
        lap._SIG_NAMES[("selftest:fwd", 1)] = ["x%d" % i for i in range(len(decl[2]))]
        if lap.arg_field_names(K, rows_now[K], fake) is not None or \
                name_leg_entry(K, rows_now[K], base.leaf_members(entry), rc, fake)[0] != "UNDECIDED" or \
                name_leg_entry(K, rows_now[K], base.leaf_members(entry), rc, sigs) != ("OK", [], ([], [], [])):
            fails.append("SELFTEST NS12b: two declarations that disagree on the names make the name join UNDECIDED")
        # NS6r: the REAL solveBlockPartition with islandIndex / doFrictionWord swapped (== the compiled
        # solverPGS.slang mutant, leaf for leaf): the join is clean -- why the name leg exists --
        # and NAME-SWAP names exactly those two
        K6 = "solveBlockPartition"
        spv6, entry6 = manifest[K6]
        m6 = Module(os.path.join(shader_dir, spv6))
        named6 = arg_field_names(K6, rows_now[K6], sigs)
        rc6 = m6.coverage_for_entry(entry6)[0]
        if not _swap_member_names(m6, entry6, "islandIndex", "doFrictionWord") or named6 is None:
            fails.append("SELFTEST BROKEN: NS6r -- members not found; re-anchor")
        else:
            lv6 = m6.leaf_members(entry6)
            j6 = join_check(rows_now[K6], [(a, b, i) for (a, b, i, _n, _t) in named6], lv6, rc6)[0]
            st = name_leg_kernel(rows_now[K6], lv6, named6, rc6)
            if j6 or st[0] != "OK" or sorted(x[1] for x in st[1]) != ["doFrictionWord", "islandIndex"]:
                fails.append("SELFTEST NS6r: %s islandIndex/doFrictionWord swapped: join %s, names %s"
                             % (K6, j6, st))
        # NS10b: the REAL artiSolveInternalConstraints1T with solveFlagsWord (four bools) and
        # doVelLimit swapped: doVelLimit now lies over the four flags
        K10 = "artiSolveInternalConstraints1T"
        spv10, entry10 = manifest[K10]
        m10 = Module(os.path.join(shader_dir, spv10))
        named10 = arg_field_names(K10, rows_now[K10], sigs)
        if not _swap_member_names(m10, entry10, "solveFlagsWord", "doVelLimit") or named10 is None:
            fails.append("SELFTEST BROKEN: NS10b -- members not found; re-anchor")
        else:
            st = name_leg_kernel(rows_now[K10], m10.leaf_members(entry10), named10,
                                 m10.coverage_for_entry(entry10)[0])
            if st[0] != "OK" or [x[1] for x in st[1]] != ["doVelLimit"]:
                fails.append("SELFTEST NS10b: %s solveFlagsWord/doVelLimit swapped must give NAME-SWAP on "
                             "doVelLimit; got %s" % (K10, st))
    except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
        fails.append("SELFTEST BROKEN: NS0-NS3/NS6r/NS10b/NS12b -- %s" % e)
    finally:
        _drop_selftest_sig_names()

    # NS12: arg_field_names, literal oracle (real kernels + synthetic signatures)
    def names_of(k):
        return [n for per in lap.arg_field_names(k, rows_now[k], sigs) for (n, _t) in per]
    try:
        w = names_of(K)[-10:]
        if w != ["writer." + f for f in ("outPoint", "outNormalPen", "outBarycentric", "outContactInfo",
                                         "totalContactCount", "contactByX", "tempContactByX",
                                         "contactIndexSortedByX", "contactSortedByX", "maxNumContacts")]:
            fails.append("SELFTEST NS12: the FEM writer's field names, in layout order; got %s" % w)
        g = [n for n in names_of("preIntegrationLaunch") if n and n.startswith("gravity")]
        if g != ["gravity.x", "gravity.y", "gravity.z"]:
            fails.append("SELFTEST NS12: a by-value PxVec3 is named per component; got %s" % g)
        h = names_of("sb_copyOrApplySoftBodyDataDEPRECATED")[:4]
        if h != ["mPtr", "mHeapMemoryAllocator", "mSize", "mStatGroup"]:
            fails.append("SELFTEST NS12: the host-shaped PxgCudaBuffer argument is named from "
                         "HOST_ARG_FIELD_NAMES; got %s" % h)
    except (KeyError, TypeError) as e:
        fails.append("SELFTEST BROKEN: NS12 -- %s" % e)
    fake = {"SelftestK": [("selftest:K", 1, [(8, "PxU32*", None), (4, "PxU32", None)])],
            "SelftestK2": [("selftest:A", 1, [(8, "PxU32*", None)]), ("selftest:B", 2, [(8, "PxU32*", None)])],
            "SelftestK3": [("selftest:C", 1, [(8, "PxU32*", None), (4, "PxU32", None)])]}
    lap._SIG_NAMES[("selftest:K", 1)] = ["ptr", "n"]
    lap._SIG_NAMES[("selftest:A", 1)] = ["a"]
    lap._SIG_NAMES[("selftest:B", 2)] = ["b"]
    lap._SIG_NAMES[("selftest:C", 1)] = ["ptr", None]
    try:
        got = lap.arg_field_names("SelftestK", [4, 4], fake)
        if got != [[(None, None)], [("n", "PxU32")]]:
            fails.append("SELFTEST NS12: a row argument whose size is not the signature's has no name; got %s" % got)
        if lap.signature_names("SelftestK2", fake) is not None:
            fails.append("SELFTEST NS12: two declarations with different names must leave the names undecided")
        # NS12c: a parameter the row follows whose name cannot be read is '?' -> UNDECIDED once reached
        got = arg_field_names("SelftestK3", [8, 4], fake)
        if got != [(0, 8, 0, "ptr", "PxU32*"), (8, 12, 1, "?", "PxU32")] or \
                name_leg_kernel([8, 4], [(0, 8, "ptr", "i"), (8, 12, "n", "i")], got, None)[0] != "UNDECIDED":
            fails.append("SELFTEST NS12c: an unnamed parameter a member reaches must make the name join "
                         "UNDECIDED; got %s" % got)
        lap._SIG_NAMES[("selftest:B", 2)] = ["a"]
        if lap.signature_names("SelftestK2", fake) != ["a"]:
            fails.append("SELFTEST NS12: two declarations with the same names agree")
    finally:
        _drop_selftest_sig_names()
    # NS14: push_name_known.txt -- a duplicate line of either kind is refused, never "last wins"
    import tempfile
    for text, ok in (("blind k a@0/b@8   # why\nswap k m\n", True),
                     ("blind k a@0/b@8\nblind k a@0/c@16\n", False),
                     ("swap k m\nswap k m\n", False), ("blind k\n", False)):
        fd, tmp = tempfile.mkstemp(prefix="pxg_pnk_")
        try:
            os.write(fd, text.encode())
            os.close(fd)
            try:
                got = read_name_known(tmp)
                if not ok or got != ({"k": "a@0/b@8"}, {("k", "m")}):
                    fails.append("SELFTEST NS14: %r must %s; got %s" % (text, "parse" if ok else "be refused", got))
            except ValueError:
                if ok:
                    fails.append("SELFTEST NS14: %r must parse" % text)
        finally:
            os.unlink(tmp)
    key = ("sb_copyOrApplySoftBodyDataDEPRECATED", 0)
    saved = lap.HOST_ARG_FIELD_NAMES[key]
    lap.HOST_ARG_FIELD_NAMES[key] = saved[:3]
    try:
        if arg_field_names(key[0], rows_now[key[0]], sigs) is not None:
            fails.append("SELFTEST NS12: names that do not cover every field are UNDECIDED, never zipped short")
    finally:
        lap.HOST_ARG_FIELD_NAMES[key] = saved
    # NS13: one layout, two spellings -> the layout stands, the names are undecided
    hdr = lap._header_texts()
    extra = ["struct SelftestDup { PxU32 a; PxU32 b; };", "struct SelftestDup { PxU32 x; PxU32 y; };",
             "struct SelftestOne { PxU32 a; float b; };", "struct SelftestVec { PxVec3 v; PxU32 n; };"]
    hdr.extend(extra)
    try:
        if lap.struct_fields("SelftestDup") != (8, 4, [(0, 4, "i"), (4, 4, "i")]) or \
                lap.struct_field_names("SelftestDup") is not None:
            fails.append("SELFTEST NS13: a struct defined twice with one layout keeps the layout, not the names")
        if lap.struct_field_names("SelftestOne") != [("a", "PxU32"), ("b", "float")]:
            fails.append("SELFTEST NS13: struct field names; got %s" % lap.struct_field_names("SelftestOne"))
        if [n for (n, _t) in lap.struct_field_names("SelftestVec") or []] != ["v.x", "v.y", "v.z", "n"]:
            fails.append("SELFTEST NS13: a vector member is named per component, in order; got %s"
                         % lap.struct_field_names("SelftestVec"))
        # NS13b: that struct passed BY VALUE and reached: its fields are `d.?` -> UNDECIDED
        fake = {"SelftestK4": [("selftest:D", 1, [lap.param_shape("SelftestDup d")])]}
        lap._SIG_NAMES[("selftest:D", 1)] = ["d"]
        got = arg_field_names("SelftestK4", [8], fake)
        if got != [(0, 4, 0, "d.?", None), (4, 8, 0, "d.?", None)] or \
                name_leg_kernel([8], [(0, 4, "a", "i"), (4, 8, "b", "i")], got, None)[0] != "UNDECIDED":
            fails.append("SELFTEST NS13b: a by-value struct whose names are undecided must make the name "
                         "join UNDECIDED; got %s" % got)
    finally:
        del hdr[-len(extra):]
        for n in ("SelftestDup", "SelftestOne", "SelftestVec"):
            lap._STRUCT_CACHE.pop(n, None)
            lap._STRUCT_NAMES.pop(n, None)
        _drop_selftest_sig_names()
    return fails


def _drop_selftest_sig_names():
    """Remove every `selftest:` declaration the selftest put into lap._SIG_NAMES."""
    for k in [k for k in lap._SIG_NAMES if str(k[0]).startswith("selftest:")]:
        del lap._SIG_NAMES[k]


def e2e_red(shader_dir):
    """The WHOLE checker on a scratch shader dir whose boxBoxNphase.spv has the OpMemberName
    bytes of startContactPoints/startContactForces (equal length, both pointers) swapped must
    exit 1, and its boxBoxNphase_Kernel NAME-SWAP lines must name exactly those two members
    (P9.99).  Other output -- other kernels, other categories -- is not checked: the run
    proves main turns a NAME-SWAP into a failing exit, not that it reports nothing else."""
    import shutil
    import subprocess
    import tempfile
    d = tempfile.mkdtemp(prefix="pxg_pushblock_e2e_")
    try:
        for f in os.listdir(shader_dir):
            if f.endswith(".spv"):
                os.symlink(os.path.join(os.path.abspath(shader_dir), f), os.path.join(d, f))
        data = open(os.path.join(shader_dir, "boxBoxNphase.spv"), "rb").read()
        a, b = b"startContactPoints\0", b"startContactForces\0"
        if data.count(a) != 1 or data.count(b) != 1:
            return ["E2E BROKEN: boxBoxNphase.spv member names not found once; re-anchor"]
        os.unlink(os.path.join(d, "boxBoxNphase.spv"))
        open(os.path.join(d, "boxBoxNphase.spv"), "wb").write(
            data.replace(a, b"\1" * len(a)).replace(b, a).replace(b"\1" * len(a), b))
        p = subprocess.run([sys.executable, os.path.abspath(__file__), d, "--no-e2e"],
                           capture_output=True, text=True)
        out = p.stdout
        # exactly the two swapped members, and a failing exit -- whatever else the dir holds
        swaps = sorted(l.split()[2] for l in out.splitlines()
                       if l.split()[:2] == ["boxBoxNphase_Kernel", "NAME-SWAP"])
        if p.returncode != 1 or swaps != ["startContactForces", "startContactPoints"]:
            return ["E2E: the checker on a shader with two member names swapped must exit 1 with "
                    "NAME-SWAP on exactly those two; rc=%d, got %s" % (p.returncode, swaps)]
        return []
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    shader_dir = args[0] if args else os.path.join(ROOT, "build-gpuslang", "shaders")
    if not os.path.isdir(shader_dir):
        print("no shader directory: %s" % shader_dir)
        return 2

    fails = selftest(shader_dir)
    if fails:
        print("THE PUSH-BLOCK JOIN SELFTEST FAILED -- it cannot see what it claims to see:")
        for f in fails:
            print("  " + f)
        print()
    if "--selftest-only" in sys.argv:
        print("check_push_block selftest: %s" % ("%d failure(s)" % len(fails) if fails else "ok"))
        return 1 if fails else 0
    if "--no-e2e" not in sys.argv:
        e2e = e2e_red(shader_dir)
        if e2e:
            print("THE PUSH-BLOCK END-TO-END RED RUN FAILED -- a real swap would pass:")
            for f in e2e:
                print("  " + f)
            print()
        fails += e2e

    rows, integrity = read_abi()
    sigs = lap.cuda_signatures()
    known = read_known()
    modules = {}
    joinbad = []         # (cuda, rule, member, detail)
    joinnotes = []
    join_undecided = set()
    joined = 0
    name_leg = {}                       # P9.99: cuda -> name_leg_kernel(...)
    name_stat = collections.Counter()
    name_possible = 0

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
            joinnotes += [(r, "%s: %s" % (cuda, n)) for (r, n) in jn]

            # ---- the NAME leg (P9.99), on a decided join only: a join-undecided kernel
            # is counted once, there ----
            name_leg[cuda] = name_leg_entry(cuda, row, leaves, ranges, sigs)
            if name_leg[cuda][0] == "OK":
                st, pair = name_stats(leaves, arg_field_names(cuda, row, sigs), ranges, want)
                name_stat.update(st)
                name_possible += pair

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

    print("  JOIN       %d kernels laid against their row field by field: %s; %d undecided (ratchet %d)"
          % (joined, ", ".join("%d %s" % (sum(1 for x in joinbad if x[1] == r), r) for r in JOIN_RULES),
             len(join_undecided), len(KNOWN_UNDECIDED_JOIN)))
    known_blind, known_swap = read_name_known()
    ok = [v for v in name_leg.values() if v[0] == "OK"]
    blind_r = sum(1 for v in ok if v[2][0])
    blind_i = sum(1 for v in ok if v[2][1] and not v[2][0])
    blind_c = sum(1 for v in ok if v[2][2] and not v[2][0] and not v[2][1])
    print("  NAMES      %d kernels joined by name: %d reached members SAME, %d RELATED, %d UNRELATED "
          "(report-only); %d NAME-SWAP; %d of %d kernels with an equal-size reached pair "
          "swap-decidable, %d name-blind + %d intra-argument + %d CUDA-side only (%d CUDA-side pairs; "
          "ratchet %d); %d undecided (an unnamed member, or a reached one over an unreadable or "
          "undeclared CUDA field)"
          % (len(ok), name_stat[SAME], name_stat[RELATED], name_stat[UNRELATED], sum(len(v[1]) for v in ok),
             name_possible - blind_r, name_possible, blind_r, blind_i, blind_c,
             sum(len(v[2][2]) for v in ok), len(known_blind), len(name_leg) - len(ok)))

    failures = len(fails)

    name_fails = name_leg_failures(name_leg, known_blind, known_swap)
    if name_fails:
        print("\nTHE SHADER'S PUSH-BLOCK NAMES AND THE CUDA'S DISAGREE, or cannot be compared.  A")
        print("NAME-SWAP is a member laid over one CUDA field but named like another; a NAME-BLIND")
        print("kernel has an equal-size pair whose swap neither the join nor a name could show")
        print("(tools/push_name_known.txt, which may shrink, never grow):")
        for cat, k, d in name_fails:
            print("  %-46s %-17s %s" % (k, cat, d))
    failures += len(name_fails)

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
    for rule, heading in (("R0b", "unreached members that lie wholly in padding (R0b)"),
                          ("R3", "unreached members that cover a field of >= 4 bytes and run on "
                                 "into padding (R3)")):
        notes = sorted(n for (r, n) in joinnotes if r == rule)
        if notes:
            print("\nReport only -- %s:" % heading)
            for n in notes:
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
