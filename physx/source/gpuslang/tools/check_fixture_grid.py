#!/usr/bin/env python3
# Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.
#
# check_fixture_grid.py -- the TWELFTH static checker.
#
# A FIXTURE THAT SIZES ITS OWN GRID CANNOT SEE A MISSING GRID-STRIDE LOOP.
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS: THE SAME BLIND SPOT HID A REAL DEFECT THREE TIMES
# ---------------------------------------------------------------------------
# The engine launches most kernels with a FIXED grid -- a compile-time constant in
# Pxg*KernelIndices.h, not a number derived from the workload.  The Vulkan bridge
# then drops blockDim entirely (`PxgVulkanCudaContext::launchKernel` takes
# `unsigned int /*blockDimX*/, ...` with the names commented out and forwards only
# gridDim to vkCmdDispatch), so a shader that does not loop over the grid reaches
# at most gridDim * numthreads work items however many the engine hands it.
#
# A fixture that dispatches `ceil(count / 256)` -- its own divisor, sized to its own
# data -- guarantees one thread per work item.  Under that geometry the grid-stride
# loop is never needed, so its ABSENCE IS INVISIBLE, and so is a loop with a wrong
# constant.  The fixture passes; the engine, at its fixed grid, silently drops work.
#
# This has now hidden a real defect three separate times:
#
#   P9.52b-6  zeroNormals / normalizeNormals: the engine launches with a block of
#             1024 while the shader declares 256, so they reached a QUARTER of the
#             guide vertices on EVERY call.  The fixture sized its grid to the
#             shader (ceil(n/256)) and used meshes of 20-50 vertices -- below one
#             group either way.  With the engine divisor and meshes above 256 it
#             failed with 1820 mismatches.
#
#   P9.61     the five skinning entries, same shape.
#
#   P9.63     updateBodiesLaunchDirectAPI and updateBodyExternalVelocitiesLaunch
#             were one-pass ports (ceiling 64*256 = 16384 records) whose fixtures
#             dispatched ceil(count/256); and four thresholdStream entries strided
#             by 256*32 against the engine's 256*64 threads while their fixture
#             dispatched a literal 32 for all eight entries in the file.
#
# By the P9.57 precedent -- a checker is written when its defect has ALREADY
# happened -- three occurrences is well past the threshold.
#
# ---------------------------------------------------------------------------
# WHAT IS CHECKED
# ---------------------------------------------------------------------------
# For every fixture `dispatch()` whose kernel the engine launches with a FIXED grid
# constant at every one of its launch sites: the dispatched group count must EQUAL
# that constant.
#
# Note the rule is equality, not "at least".  A fixture dispatching FEWER groups
# than the engine leaves work uncovered on a correct shader; one dispatching MORE
# hides a too-small stride (the thresholdStream case ran the loop at half the
# engine's thread count and was output-equivalent only because its bodies happened
# to be idempotent).  Only the engine's own number reproduces what the engine does.
#
# ---------------------------------------------------------------------------
# THE EXEMPTION: A SHADER THAT SPELLS ITS STRIDE WITH WorkgroupCount()
# ---------------------------------------------------------------------------
# `WorkgroupCount()` is Slang's runtime spelling of gridDim, so a shader deriving
# its stride from it covers its work correctly AT ANY GRID -- there is no constant
# for the fixture to disagree with, and dispatching fewer groups is then a free and
# legitimate choice.  psHfMidphaseBda.slang is the case that taught this: its
# fixture dispatches 64 groups against the engine's 4096, says why in a comment
# ("stride 4096 > NP: one thread per particle"), separately ASSERTS
# `PxgParticleSystemKernelGridDim::PS_HEIGHTFIELD_COLLISION == HF_GRID`, and carries
# a `grid-real-4096` variant besides.  Flagging that would have been the checker
# manufacturing work, so entries whose body reaches WorkgroupCount() are skipped and
# counted as exempt.
#
# The exemption is deliberately narrow.  It does NOT cover a shader that partitions
# by a hard-coded grid constant, even a correct one:
# `cloth_clothContactPrepareLaunch` offsets each block by
# `nbIterationsPerBlock * groupId.x * RC_BLOCK` with `RC_GRID` baked into
# nbIterationsPerBlock.  RC_GRID does equal the engine's 1024 today -- but a fixture
# dispatching one group with fewer contacts than one block passes whatever RC_GRID
# says, so the constant is unobserved and the check still applies.
#
# ---------------------------------------------------------------------------
# WHAT THIS DELIBERATELY CANNOT DO
# ---------------------------------------------------------------------------
#  * THE GRID IS THE FLOOR OF THE PROPERTY, NOT THE WHOLE OF IT.  A fixture at the
#    engine's grid whose INPUT still fits in one pass observes nothing either -- the
#    loop never iterates.  P9.63's thresholdStream fixture and P9.64's
#    thresholdForceChange fixture both had to grow their element counts past one
#    pass before a deliberately wrong stride would fail, and the FEM prepare
#    fixtures still run a single contact, where no grid makes the partitioning
#    constant observable.  This checks the grid, which it can see; the input size is
#    per-kernel work-count reasoning it cannot do, and passing here is not a claim
#    that a fixture would catch a wrong stride.
#  * It says nothing about kernels the engine launches with a COMPUTED grid.  There
#    the fixture's own divisor is the honest thing to write, and the check skips it.
#  * It cannot resolve every expression on either side.  A dispatch argument that is
#    not a literal or a same-file constant, and a grid whose enum member it cannot
#    find, are reported as UNRESOLVED and counted separately -- they are neither
#    passes nor failures, and the summary prints how many there are so the number
#    that is not being checked stays visible.
#  * It reads the FIRST dispatch argument only.  A fixture is free to pass a y/z
#    extent; none currently does.
#  * Matching a launchKernel call to its PxgKernelIds enum uses the nearest
#    preceding getCuFunction in the same file, which is the corpus idiom
#    (assign kernelFunction, then launch it) but is a heuristic, not a parse.
#    A kernel whose launch site it cannot pair is simply absent from the engine
#    table and its fixtures go unchecked -- it under-reports, never over-reports.
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GS = os.path.dirname(HERE)                       # .../source/gpuslang
SRC = os.path.dirname(GS)                        # .../source
KNOWN = os.path.join(HERE, "fixture_grid_known.txt")


def kernel_names():
    """PxgKernelIds enum member -> CUDA kernel name."""
    out = {}
    path = os.path.join(SRC, "gpucommon/include/PxgKernelNames.h")
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r'\s*KERNEL_DEF\(\s*(\w+)\s*,\s*"([^"]+)"\s*\)', line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def grid_constants():
    """(enum type, member) -> int, over every *KernelGridDim enum in the headers.

    The headers put the grid and block dims in two enums with THE SAME member
    names, so the enum type has to be part of the key -- keying on the member
    alone silently reads a block dim as a grid dim."""
    out = {}
    for dirpath, _d, files in os.walk(SRC):
        if os.sep + "gpuslang" in dirpath:
            continue
        for fn in files:
            if not fn.endswith(".h"):
                continue
            text = open(os.path.join(dirpath, fn), encoding="utf-8", errors="replace").read()
            for em in re.finditer(r"\b(?:struct|namespace|enum(?:\s+class)?)\s+(\w*KernelGridDim)\b", text):
                brace = text.find("{", em.end())
                if brace < 0:
                    continue
                depth, end = 0, len(text)
                for j in range(brace, len(text)):
                    if text[j] == "{":
                        depth += 1
                    elif text[j] == "}":
                        depth -= 1
                        if depth == 0:
                            end = j
                            break
                for mm in re.finditer(r"\b(\w+)\s*=\s*(\d+)\s*[,}]", text[brace:end]):
                    out[(em.group(1), mm.group(1))] = int(mm.group(2))
    return out


GETFN = re.compile(r"getCuFunction\(\s*PxgKernelIds::(\w+)\s*\)")
LAUNCH = re.compile(r"launchKernel\(\s*(\w+)\s*,\s*([^,]+),")
GRIDREF = re.compile(r"(\w*KernelGridDim)::(\w+)")


def engine_grids(names, consts):
    """CUDA kernel name -> int grid, for kernels FIXED at every launch site.

    A kernel launched from several sites (PGS and TGS solver cores each launch the
    threshold-stream pipeline, for instance) is only usable here if every site
    agrees; a disagreement means there is no single number a fixture could use."""
    seen = {}
    for dirpath, _d, files in os.walk(SRC):
        if os.sep + "gpuslang" in dirpath:
            continue
        for fn in files:
            if not fn.endswith(".cpp"):
                continue
            text = open(os.path.join(dirpath, fn), encoding="utf-8", errors="replace").read()
            marks = [(m.start(), m.group(1)) for m in GETFN.finditer(text)]
            for m in LAUNCH.finditer(text):
                prev = [nm for pos, nm in marks if pos < m.start()]
                if not prev:
                    continue
                cuda = names.get(prev[-1])
                if cuda is None:
                    continue
                g = GRIDREF.search(m.group(2))
                val = consts.get((g.group(1), g.group(2))) if g else None
                seen.setdefault(cuda, set()).add(val)
    return {k: v.pop() for k, v in seen.items() if len(v) == 1 and None not in v}


KDECL = re.compile(r'\b(\w+)\s*\(\s*"[\w.]+\.spv"\s*,\s*"(\w+)"')
DISPATCH = re.compile(r"\b(\w+)\.dispatch\(\s*([^,]+),")
CONSTDECL = re.compile(r"\b(?:static\s+)?const(?:expr)?\s+\w+\s+(\w+)\s*=\s*([^;]+);")


def resolve(expr, scope):
    """An int for a literal or a same-file constant (one product deep), else None."""
    expr = expr.strip()
    if re.fullmatch(r"\d+u?", expr):
        return int(expr.rstrip("uU"))
    if expr in scope:
        return scope[expr]
    m = re.fullmatch(r"(\w+)\s*\*\s*(\w+)", expr)
    if m:
        a, b = (resolve(m.group(1), scope), resolve(m.group(2), scope))
        if a is not None and b is not None:
            return a * b
    return None


def brace_body(text, start):
    depth = 0
    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start:j]
    return None


def workgroup_count_entries():
    """Entry names whose Slang body reaches WorkgroupCount(), directly or through
    one level of same-file helper -- their stride adapts to any grid."""
    out = set()
    shaders = os.path.join(GS, "shaders")
    for fn in sorted(os.listdir(shaders)):
        if not fn.endswith(".slang"):
            continue
        text = open(os.path.join(shaders, fn), encoding="utf-8", errors="replace").read()
        if "WorkgroupCount" not in text:
            continue
        for m in re.finditer(r"\bvoid\s+(\w+)\s*\([^;{]*\)\s*\{", text):
            body = brace_body(text, m.end() - 1)
            if body is None:
                continue
            if "WorkgroupCount" in body:
                out.add(m.group(1))
                continue
            for cm in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", body):
                name = cm.group(1)
                if name in ("if", "for", "while", "switch", "return"):
                    continue
                hm = re.search(r"\b\w[\w<>,\s\*]*\s" + re.escape(name) + r"\s*\([^;{]*\)\s*\{", text)
                if hm is None:
                    continue
                hb = brace_body(text, hm.end() - 1)
                if hb and "WorkgroupCount" in hb:
                    out.add(m.group(1))
                    break
    return out


def load_known():
    out = set()
    if os.path.exists(KNOWN):
        for line in open(KNOWN, encoding="utf-8"):
            line = line.split("#")[0].strip()
            if line:
                out.add(line)
    return out


def main():
    consts = grid_constants()
    grids = engine_grids(kernel_names(), consts)
    adaptive = workgroup_count_entries()

    tests = os.path.join(GS, "tests")
    bad, checked, unresolved, exempt = [], 0, 0, 0

    for fn in sorted(os.listdir(tests)):
        if not fn.endswith(".cpp"):
            continue
        text = open(os.path.join(tests, fn), encoding="utf-8", errors="replace").read()
        # A Kernel variable name is reused across test functions -- testFemClothSim
        # declares `kPrep` for five different entries -- so a name->entry dict keyed
        # over the whole file resolves every dispatch to the LAST declaration and
        # attributes the wrong kernel.  (The first run of this checker did exactly
        # that and blamed cloth_clothContactPrepareLaunch for six dispatches, one of
        # which was cloth_rigidAttachmentPrepareLaunch.)  Pair each dispatch with
        # the nearest PRECEDING declaration of that name instead.
        decls = [(m.start(), m.group(1), m.group(2)) for m in KDECL.finditer(text)]
        scope = {}
        for m in CONSTDECL.finditer(text):
            v = resolve(m.group(2), scope)
            if v is not None:
                scope[m.group(1)] = v
        for m in DISPATCH.finditer(text):
            prior = [e for pos, var, e in decls if var == m.group(1) and pos < m.start()]
            entry = prior[-1] if prior else None
            if entry is None or entry not in grids:
                continue
            if entry in adaptive:
                exempt += 1
                continue
            want = grids[entry]
            got = resolve(m.group(2), scope)
            if got is None:
                unresolved += 1
                continue
            checked += 1
            if got != want:
                line = text.count("\n", 0, m.start()) + 1
                bad.append((entry, fn, line, got, want))

    known = load_known()
    fresh = [b for b in bad if b[0] not in known]
    stale = sorted(known - {b[0] for b in bad})

    print("fixture-grid check over %d fixed-grid engine kernels" % len(grids))
    print("  dispatches resolved and compared: %d   unresolved (not checked): %d"
          % (checked, unresolved))
    print("  exempt (shader strides by WorkgroupCount, correct at any grid): %d" % exempt)
    print("  disagreeing with the engine: %d   of those carried as known: %d"
          % (len(bad), len(bad) - len(fresh)))

    if fresh:
        print("\n%d FIXTURE DISPATCH(ES) DO NOT USE THE ENGINE'S GRID:\n" % len(fresh))
        for entry, fn, line, got, want in fresh:
            print("  %s\n    tests/%s:%d dispatches %d, engine launches %d"
                  % (entry, fn, line, got, want))
        print("\nA fixture at its own grid cannot see a missing or wrong grid-stride")
        print("loop.  Dispatch the engine's grid and size the input past one pass.")
    if stale:
        print("\n%d STALE ENTRY(IES) in fixture_grid_known.txt:\n" % len(stale))
        for name in stale:
            print("  %s -- now dispatches the engine's grid; remove the line" % name)
        print("\nA ratchet that keeps fixed kernels on its list stops ratcheting.")

    if fresh or stale:
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
