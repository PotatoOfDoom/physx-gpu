#!/usr/bin/env python3
# check_subgroup_ops.py -- P9.99.  Pins the premise behind createPipelineEx's refusals:
# NOTHING in the engine depends on a subgroup size, so refusing a request (instead of
# quietly building without it) can never cost the engine a kernel.
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS
# ---------------------------------------------------------------------------
# P9.99 made createPipelineEx check an EXACT subgroup size against the entry's workgroup
# shape (VUID-02756 / 02757) and refuse what cannot be honoured, through the P9.98
# refusal path.  No fallback was added: the only real customer is the native warp-32
# path (npWarpNative, PXG_WARP_NATIVE 1, a P5 hardware item), which is correct only at
# exactly 32 lanes with full subgroups -- building it without the requirement would turn
# a refusal into wrong warp semantics.
#
# That decision is safe only while two facts hold, and nothing else in the build watches
# either of them:
#
#   (a) HOST.  No engine host path calls createPipelineEx.  The shim builds every engine
#       pipeline through createPipeline / createPipelineTex / createPipelineBindless
#       (src/PxgCudaShim.cpp, pipelineFor), and the bridge never asks for a size.  So in
#       src/, bridge/ and include/ the identifier may occur only where it is DECLARED or
#       DEFINED (`PxgPipelineHandle createPipelineEx(`) -- the base default and each
#       backend's override -- never as a call or an address.  Comments and string
#       literals are stripped first.  tests/ is not scanned: the fixtures call it.
#
#   (b) SPIR-V.  No ENGINE module (every module the kernel manifest maps) carries an
#       EXPLICIT subgroup dependency:
#         - a capability GroupNonUniform* (61-68), SubgroupBallotKHR (4423),
#           SubgroupVoteKHR (4431), GroupNonUniformRotateKHR (6026),
#           GroupNonUniformPartitionedNV (5297), QuadControlKHR (5087) or Groups (18);
#         - a BuiltIn decoration SubgroupSize .. SubgroupLocalInvocationId (36-41) or
#           Subgroup{Eq,Ge,Gt,Le,Lt}Mask (4416-4420);
#         - an instruction OpGroupNonUniform* (333-366), OpSubgroup*KHR (4421-4432) or
#           OpGroupNonUniformQuad{All,Any}KHR (5110-5111: Slang's QuadAll / QuadAny, which
#           declare QuadControlKHR and NO GroupNonUniform* capability);
#         - an OpControlBarrier / OpMemoryBarrier / OpAtomic* whose Scope operand is the
#           constant Subgroup (3).  These need NO capability, so a capability scan alone
#           cannot see them (P9.99 review D7).  A scope that is not a 32-bit OpConstant
#           is reported too: it cannot be shown not to be Subgroup.
#       The allowlist of modules that DO use subgroups is EXACT over the whole shader
#       directory: a listed module that is missing or no longer uses subgroups fails, a
#       module that uses them and is not listed fails, and no allowlisted module may be
#       an engine module.
#
# What it does NOT prove: that the engine is race-free at every subgroup width.  An op
# scan sees only EXPLICIT dependencies; RUNTIME_STATUS records races that llvmpipe's
# in-order 8-lane execution masks.  The no-fallback argument does not rest on that --
# it rests on (a).
#
# Measured at P9.99: 178 engine modules (428 manifest rows), 0 hits; whole build, hits
# only in the three probes below.  A selftest on synthetic module words and source text
# runs first; a selftest failure fails the run.
#
# Usage: check_subgroup_ops.py [shader-dir]   (default <repo>/build-gpuslang/shaders)
# Exit 0 clean, 1 findings or selftest failure, 2 usage / missing inputs.

import os
import re
import struct
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
GPUSLANG = os.path.join(ROOT, "physx", "source", "gpuslang")
MANIFEST = os.path.join(GPUSLANG, "include", "PxgSlangKernelManifest.h")
HOST_DIRS = ("src", "bridge", "include")
HOST_EXT = (".h", ".hpp", ".cpp", ".cc", ".inl")

# Test-only probes that use subgroup operations on purpose (never in the manifest).
ALLOWLIST = {
    "npSpikeSubgroup.spv": "RUNTIME_PLAN P0 spike: wave invariants at the device's subgroup size",
    "npWarpNative.spv": "the native warp-32 library probe (PXG_WARP_NATIVE 1, P5 hardware)",
    "p99SubgroupShape.spv": "P9.99 createPipelineEx shape rules (gpuslang.pipelineRefusal)",
}

SUBGROUP_CAPS = {18: "Groups", 61: "GroupNonUniform", 62: "GroupNonUniformVote",
                 63: "GroupNonUniformArithmetic", 64: "GroupNonUniformBallot",
                 65: "GroupNonUniformShuffle", 66: "GroupNonUniformShuffleRelative",
                 67: "GroupNonUniformClustered", 68: "GroupNonUniformQuad",
                 4423: "SubgroupBallotKHR", 4431: "SubgroupVoteKHR", 5087: "QuadControlKHR",
                 5297: "GroupNonUniformPartitionedNV", 6026: "GroupNonUniformRotateKHR"}
SUBGROUP_BUILTINS = {36: "SubgroupSize", 37: "SubgroupMaxSize", 38: "NumSubgroups",
                     39: "NumEnqueuedSubgroups", 40: "SubgroupId", 41: "SubgroupLocalInvocationId",
                     4416: "SubgroupEqMask", 4417: "SubgroupGeMask", 4418: "SubgroupGtMask",
                     4419: "SubgroupLeMask", 4420: "SubgroupLtMask"}
OP_CAPABILITY = 17
OP_ENTRY_POINT = 15
OP_DECORATE = 71
OP_CONSTANT = 43
DECORATION_BUILTIN = 11
SCOPE_SUBGROUP = 3
# opcode -> indices (into the operand words, opcode word excluded) of its Scope operands
SCOPE_OPERANDS = {224: (0, 1),   # OpControlBarrier: Execution, Memory
                  225: (0,),     # OpMemoryBarrier: Memory
                  227: (3,),     # OpAtomicLoad
                  228: (1,),     # OpAtomicStore
                  5614: (3,), 5615: (3,), 6035: (3,)}   # OpAtomicFMin/FMax/FAddEXT
for _op in range(229, 243):      # OpAtomicExchange .. OpAtomicXor
    SCOPE_OPERANDS[_op] = (3,)


def is_subgroup_op(op):
    return 333 <= op <= 366 or 4421 <= op <= 4432 or op in (5110, 5111)


def module_words(path):
    data = open(path, "rb").read()
    n = len(data) // 4
    words = list(struct.unpack("<%dI" % n, data[:n * 4]))
    if n < 5 or words[0] != 0x07230203:
        raise ValueError("not a SPIR-V module")
    return words


def scan_words(words):
    """Every explicit subgroup dependency in one module, as readable strings."""
    hits = []
    consts = {}
    scoped = []   # (opcode, scope id)
    i = 5
    while i < len(words):
        op = words[i] & 0xFFFF
        wc = words[i] >> 16
        if wc == 0 or i + wc > len(words):
            hits.append("malformed instruction stream at word %d" % i)
            break
        a = words[i + 1:i + wc]
        if op == OP_CAPABILITY and a and a[0] in SUBGROUP_CAPS:
            hits.append("OpCapability %s (%d)" % (SUBGROUP_CAPS[a[0]], a[0]))
        elif op == OP_DECORATE and len(a) >= 3 and a[1] == DECORATION_BUILTIN and a[2] in SUBGROUP_BUILTINS:
            hits.append("BuiltIn %s (%d)" % (SUBGROUP_BUILTINS[a[2]], a[2]))
        elif is_subgroup_op(op):
            hits.append("opcode %d (OpGroupNonUniform* / OpSubgroup*KHR / OpGroupNonUniformQuad*KHR)" % op)
        elif op == OP_CONSTANT and len(a) == 3:
            consts[a[1]] = a[2]
        if op in SCOPE_OPERANDS:
            for k in SCOPE_OPERANDS[op]:
                scoped.append((op, a[k] if k < len(a) else None))
        i += wc
    for op, sid in scoped:
        if sid is None or sid not in consts:
            hits.append("opcode %d: Scope operand %s is not a 32-bit OpConstant" % (op, sid))
        elif consts[sid] == SCOPE_SUBGROUP:
            hits.append("opcode %d with Scope Subgroup" % op)
    return hits, len(scoped)


def manifest_modules(text=None):
    if text is None:
        text = open(MANIFEST).read()
    rows = re.findall(r'\{\s*"([^"]+)"\s*,\s*"([^"]+\.spv)"\s*,\s*"([^"]+)"\s*\}', text)
    return rows, sorted({spv for (_c, spv, _e) in rows})


def strip_cpp(text):
    """Comments and string / character literals blanked (newlines kept)."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        elif c == '"' or c == "'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            j = min(j, n)
            if j < n and text[j] == c:
                j += 1
                out.append(c + " " * (j - i - 2) + c)
            else:   # unterminated on this line: keep the newline
                out.append(c + " " * (j - i - 1))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


DECL = re.compile(r"\bPxgPipelineHandle\s+createPipelineEx\s*\(")
IDENT = re.compile(r"\bcreatePipelineEx\b")


def host_uses(text):
    """(declarations, [line numbers of every other use]) of createPipelineEx in C++ text."""
    s = strip_cpp(text)
    decl_at = {m.end() - len("(") for m in DECL.finditer(s)}
    decls = 0
    others = []
    for m in IDENT.finditer(s):
        tail = s[m.end():]
        paren = m.end() + (len(tail) - len(tail.lstrip()))
        if paren in decl_at:
            decls += 1
        else:
            others.append(s.count("\n", 0, m.start()) + 1)
    return decls, others


def allowlist_problems(hits_by_module, present, engine):
    """The allowlist must be EXACT: listed <=> present and using subgroups, never an engine module."""
    probs = []
    for spv in sorted(ALLOWLIST):
        if spv in engine:
            probs.append("%s is allowlisted but is an ENGINE module (manifest)" % spv)
        if spv not in present:
            probs.append("%s is allowlisted but missing from the shader directory" % spv)
        elif not hits_by_module.get(spv):
            probs.append("%s is allowlisted but no longer uses subgroups (stale entry)" % spv)
    for spv in sorted(hits_by_module):
        if hits_by_module[spv] and spv not in ALLOWLIST and spv not in engine:
            probs.append("%s uses subgroups and is not allowlisted: %s" % (spv, "; ".join(hits_by_module[spv][:3])))
    return probs


def selftest(sample_words):
    """The checker on inputs whose answer is known: (failures)."""
    fails = []
    clean, _n = scan_words(sample_words)
    if clean:
        fails.append("SELFTEST: the clean sample module must have no hit: %s" % clean)
    bound = sample_words[3]
    hdr, body = sample_words[:5], sample_words[5:]

    def with_body(extra, bump=4):
        w = list(hdr) + list(body) + list(extra)
        w[3] = bound + bump
        return w

    # ids past the bound: c3 = the constant 3 (Subgroup), cw = the constant 2 (Workgroup, also
    # used as the semantics operand), other = a non-constant id.  Every operand of a scoped
    # instruction that is NOT its scope is cw or other, so a scope read from the wrong
    # operand index cannot find the 3.
    c3, other, cw = bound, bound + 1, bound + 2
    k3 = [(4 << 16) | OP_CONSTANT, 1, c3, SCOPE_SUBGROUP, (4 << 16) | OP_CONSTANT, 1, cw, 2]
    cases = [
        ("OpCapability GroupNonUniform", list(hdr) + [(2 << 16) | OP_CAPABILITY, 61] + list(body), "OpCapability"),
        ("OpCapability Groups", list(hdr) + [(2 << 16) | OP_CAPABILITY, 18] + list(body), "OpCapability"),
        ("BuiltIn SubgroupLocalInvocationId", with_body([(4 << 16) | OP_DECORATE, other, DECORATION_BUILTIN, 41]),
         "BuiltIn"),
        ("BuiltIn SubgroupLtMask", with_body([(4 << 16) | OP_DECORATE, other, DECORATION_BUILTIN, 4420]), "BuiltIn"),
        ("OpGroupNonUniformElect", with_body([(4 << 16) | 333, 1, other, c3]), "opcode 333"),
        ("OpSubgroupBallotKHR", with_body([(4 << 16) | 4421, 1, other, c3]), "opcode 4421"),
        # QuadAll / QuadAny (slangc 'QuadAll(gi > 3u)', measured): capability 5087 and
        # opcode 5110 / 5111, no GroupNonUniform* capability at all.
        ("OpCapability QuadControlKHR", list(hdr) + [(2 << 16) | OP_CAPABILITY, 5087] + list(body),
         "OpCapability QuadControlKHR"),
        ("OpGroupNonUniformQuadAllKHR", with_body([(4 << 16) | 5110, 1, other, other]), "opcode 5110"),
        ("OpGroupNonUniformQuadAnyKHR", with_body([(4 << 16) | 5111, 1, other, other]), "opcode 5111"),
        ("OpControlBarrier Subgroup execution", with_body(k3 + [(4 << 16) | 224, c3, cw, cw]), "Scope Subgroup"),
        ("OpControlBarrier Subgroup memory", with_body(k3 + [(4 << 16) | 224, cw, c3, cw]), "Scope Subgroup"),
        ("OpMemoryBarrier Subgroup", with_body(k3 + [(3 << 16) | 225, c3, cw]), "Scope Subgroup"),
        ("OpAtomicIAdd Subgroup", with_body(k3 + [(7 << 16) | 234, 1, other, cw, c3, cw, cw]), "Scope Subgroup"),
        ("OpAtomicStore Subgroup", with_body(k3 + [(5 << 16) | 228, cw, c3, cw, cw]), "Scope Subgroup"),
        ("OpAtomicLoad Subgroup", with_body(k3 + [(6 << 16) | 227, 1, other, cw, c3, cw]), "Scope Subgroup"),
        ("OpAtomicFAddEXT Subgroup", with_body(k3 + [(7 << 16) | 6035, 1, other, cw, c3, cw, cw]),
         "Scope Subgroup"),
        ("OpMemoryBarrier unresolved scope", with_body([(3 << 16) | 225, other, other]), "not a 32-bit OpConstant"),
    ]
    for name, words, want in cases:
        hits, _n = scan_words(words)
        if not any(want in h for h in hits):
            fails.append("SELFTEST: %s must be reported (%s), got %s" % (name, want, hits))
    # A Workgroup-scope (2) barrier is NOT a subgroup dependency.
    hits, _n = scan_words(with_body(k3 + [(4 << 16) | 224, cw, cw, cw, (7 << 16) | 234, 1, other, cw, cw, cw, cw]))
    if hits:
        fails.append("SELFTEST: a Workgroup-scope barrier must not be reported: %s" % hits)

    # Host leg.
    host_cases = [
        ("virtual PxgPipelineHandle createPipelineEx(const void* c) { return 0; }", 1, 0),
        ("PxgPipelineHandle createPipelineEx (const void* c) override\n{}", 1, 0),
        ("// dev->createPipelineEx(blob, 8)\n/* createPipelineEx( */ const char* s = \"createPipelineEx(\";", 0, 0),
        ("void f() { h = dev->createPipelineEx(b, n, e, 0, nullptr, 4, 8); }", 0, 1),
        ("void f() { h = mDevice->\n  createPipelineEx(b, n, e, 0, nullptr, 4, 0); }", 0, 1),
        ("auto p = &PxgSlangComputeDevice::createPipelineEx;", 0, 1),
        ("const char* s = \"a\\\"b\"; x = d.createPipelineEx(1);", 0, 1),
    ]
    for text, wd, wo in host_cases:
        d, o = host_uses(text)
        if d != wd or len(o) != wo:
            fails.append("SELFTEST: host scan of %r: %d declaration(s), %d use(s); expected %d, %d"
                         % (text, d, len(o), wd, wo))

    # Allowlist exactness.
    engine = {"eng.spv"}
    good = {spv: ["x"] for spv in ALLOWLIST}
    present = set(ALLOWLIST) | {"eng.spv", "probe.spv"}
    if allowlist_problems(good, present, engine):
        fails.append("SELFTEST: an exact allowlist must pass")
    idle = dict(good)
    idle["npWarpNative.spv"] = []
    if not any("stale" in p for p in allowlist_problems(idle, present, engine)):
        fails.append("SELFTEST: an allowlisted module without subgroup use must fail")
    if not any("missing" in p for p in allowlist_problems(good, present - {"p99SubgroupShape.spv"}, engine)):
        fails.append("SELFTEST: a missing allowlisted module must fail")
    extra = dict(good)
    extra["probe.spv"] = ["x"]
    if not any("not allowlisted" in p for p in allowlist_problems(extra, present, engine)):
        fails.append("SELFTEST: an unlisted subgroup module must fail")
    if not any("ENGINE" in p for p in allowlist_problems(good, present, engine | {"npSpikeSubgroup.spv"})):
        fails.append("SELFTEST: an allowlisted engine module must fail")
    return fails


def main():
    shader_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build-gpuslang", "shaders")
    if not os.path.isdir(shader_dir):
        print("check_subgroup_ops: no shader directory %s" % shader_dir)
        return 2
    rows, engine_list = manifest_modules()
    engine = set(engine_list)
    if not engine:
        print("check_subgroup_ops: no module rows parsed from %s" % MANIFEST)
        return 2
    present = sorted(f for f in os.listdir(shader_dir) if f.endswith(".spv"))
    bad = []

    missing = [m for m in engine_list if m not in present]
    for m in missing:
        bad.append("engine module %s is missing from %s" % (m, shader_dir))

    hits_by_module = {}
    scoped_total = 0
    for spv in present:
        try:
            hits, scoped = scan_words(module_words(os.path.join(shader_dir, spv)))
        except ValueError as e:
            bad.append("%s: %s" % (spv, e))
            continue
        scoped_total += scoped
        hits_by_module[spv] = hits
    sample = None
    for m in engine_list:
        if m in hits_by_module and not hits_by_module[m]:
            sample = module_words(os.path.join(shader_dir, m))
            break
    fails = selftest(sample) if sample is not None else ["SELFTEST: no clean engine module to start from"]

    engine_hits = [(m, hits_by_module[m]) for m in engine_list if hits_by_module.get(m)]
    for m, h in engine_hits:
        bad.append("ENGINE module %s has an explicit subgroup dependency: %s" % (m, "; ".join(h[:4])))
    bad += allowlist_problems(hits_by_module, set(present), engine)

    decls = 0
    host_files = 0
    for d in HOST_DIRS:
        for base, _dirs, files in os.walk(os.path.join(GPUSLANG, d)):
            for f in sorted(files):
                if not f.endswith(HOST_EXT):
                    continue
                path = os.path.join(base, f)
                host_files += 1
                nd, others = host_uses(open(path, errors="replace").read())
                decls += nd
                for line in others:
                    bad.append("%s:%d: createPipelineEx is used outside a declaration (an engine caller?)"
                               % (os.path.relpath(path, GPUSLANG), line))
    if decls < 2:
        bad.append("createPipelineEx: %d declaration(s) found in %s (the base default and the Vulkan override "
                   "were expected) -- the host scan is not seeing the code" % (decls, "/".join(HOST_DIRS)))

    print("check_subgroup_ops: %d manifest rows, %d engine modules, %d modules in %s, %d Scope operands resolved"
          % (len(rows), len(engine), len(present), shader_dir, scoped_total))
    print("  engine modules with an explicit subgroup dependency: %d" % len(engine_hits))
    print("  subgroup modules (allowlist, exact): %s"
          % ", ".join(sorted(m for m, h in hits_by_module.items() if h)))
    print("  host: %d files in %s, %d createPipelineEx declaration(s), %d other use(s)"
          % (host_files, ", ".join(HOST_DIRS), decls,
             sum(1 for b in bad if "used outside a declaration" in b)))
    if fails:
        print("THE SUBGROUP-OPS SELFTEST FAILED -- it cannot see what it claims to see:")
        for f in fails:
            print("  " + f)
    for b in bad:
        print("FAIL " + b)
    if fails or bad:
        return 1
    print("check_subgroup_ops: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
