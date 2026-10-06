#!/usr/bin/env python3
# check_block_coverage.py -- P9.68.  The FOURTEENTH static checker, and the FOURTH
# written because its defect had already happened -- twice, three tranches apart.
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS
# ---------------------------------------------------------------------------
# `PxgVulkanCudaContext::launchKernel` takes
#
#     unsigned int /*blockDimX*/, unsigned int /*blockDimY*/, unsigned int /*blockDimZ*/
#
# -- the parameter names are commented out -- and forwards ONLY gridDimX/Y/Z to
# vkCmdDispatch.  So on this port:
#
#     the CUDA grid  becomes  the Vulkan workgroup count, verbatim;
#     the CUDA block  is      DISCARDED;
#     the Slang [numthreads] IS the block size.
#
# The host almost always sizes its grid FROM its block: `numBlocks = ceil(n /
# numThreadsPerBlock)`.  A port whose [numthreads] is SMALLER than that block and
# which makes one flat pass therefore reaches
#
#     ceil(n / cudaBlock) * slangThreads     instead of     n
#
# threads -- a fixed FRACTION of the work, on every call, at every scene size.
# This is not a ceiling past which the tail is lost.  It is slangThreads/cudaBlock
# of the work, forever.
#
# ---------------------------------------------------------------------------
# IT HAS HAPPENED TWICE, AND NEITHER TIME DID ANYTHING SAY SO
# ---------------------------------------------------------------------------
# P9.61: `zeroNormals` and `normalizeNormals` declared [numthreads(256,1,1)] against
# a host block of 1024, so three quarters of the guide vertices were never zeroed and
# never normalised -- on EVERY call.  The fixture passed because it sized its own grid
# to the shader and used meshes of 20-50 vertices, below one group either way.  Given
# the engine divisor and 300-700 vertex meshes it failed with 1820 mismatches.
#
# P9.67c: `convexCorePlaneNphase_Kernel` shipped [numthreads(64,1,1)] against a host
# block of 1024 (PxgNarrowphaseCore.cpp:848-852) -- ONE SIXTEENTH of the pairs.  The
# scene it was written for has a single cylinder, and test 0 is inside every ceiling
# there is.  P9.68 widened that scene to 80 cylinders with sleeping disabled and the
# shipped shader dropped exactly 16 of them through the plane (final y = -73.807,
# |v| = 39.240, 32 failed checks); with the loop, 0.
#
# BOTH WERE FOUND BY HAND, LATE, AND BY ACCIDENT.  check_grid_stride.py cannot see
# this class: it only fires when the CUDA ITSELF carries a grid-stride loop, and the
# CUDA here has none -- it does not need one, because its block is big enough.  The
# defect lives entirely in the gap between the two block sizes, which is a fact about
# the LAUNCH SITE, not about either body.
#
# ---------------------------------------------------------------------------
# THE RULE
# ---------------------------------------------------------------------------
# For every MAPPED row of PxgSlangKernelManifest.h that has an engine launch site:
#
#   1. resolve the launch site's blockDim product (x*y*z);
#   2. resolve the Slang entry's [numthreads] product;
#   3. PASS if slangThreads >= cudaBlock -- the group is at least as wide, so the
#      forwarded grid covers at least as much;
#   4. PASS if the GRID IS NOT SIZED FROM THE BLOCK (grid_from_block below).  One
#      block per object is the common shape and there the group width is parallelism,
#      not coverage;
#   5. PASS if the port carries a covering construct (grid_covers below:
#      WorkgroupCount(), a named stride in a for-increment WHOSE INIT CLAUSE IS ROOTED
#      IN SV_GroupID / SV_DispatchThreadID / WorkgroupCount (P9.97a), or a counted
#      loop whose variable scales a named product);
#   6. PASS if the port PARTITIONS THE BLOCK SERIALLY (serial_factor below) by enough:
#      slangThreads * tripCount >= cudaBlock;
#   7. otherwise FAIL, reporting the ratio.
#
# When a kernel has SEVERAL launch sites the LARGEST resolved block is used: the port
# has one [numthreads] and must satisfy the worst caller.
#
# EVERY ONE OF 4, 5 AND 6 WAS ADDED BECAUSE THE CHECK WITHOUT IT REPORTED PORTS THAT
# COVER THEIR WORK.  Successive readings took the finding count 35 -> 14 -> 7 -> 0,
# and each step was a false-positive shape with a written reason, not a loosened
# threshold.  P9.60b is the precedent and the warning: a marker that is merely
# plausible produces a list that is mostly noise, and a list that is mostly noise gets
# believed.
#
# ---------------------------------------------------------------------------
# WHAT IT DELIBERATELY CANNOT DECIDE
# ---------------------------------------------------------------------------
#  * THREADS ARE A PROXY FOR WORK ITEMS, NOT THE THING ITSELF.  A CUDA block of 1024
#    running a warp per item covers 32 items; a Slang group of 128 running a warp per
#    item covers 4.  The ratio is right there.  But a port that legitimately changes
#    the threads-per-item -- 32 in CUDA, 1 here -- can cover MORE items with FEWER
#    threads, and this check would call that a finding.  No such case is in the tree
#    today; if one arrives it belongs on the ratchet with the reason written down,
#    not silently exempted in code.
#  * IT READS ONLY LAUNCH SITES IT CAN PAIR AND RESOLVE.  A launchKernel call is paired
#    to its kernel through its FIRST ARGUMENT -- the CUfunction variable -- resolved to
#    that variable's nearest preceding getCuFunction assignment; proximity is only the
#    fallback for a CUfunction that is not a named variable (see GETFN_VAR for the
#    false finding proximity produced).  Block dims are resolved from literals,
#    *KernelBlockDim/*KernelGridDim enum members, WARP_SIZE, and the NEAREST PRECEDING
#    same-file declaration of a name (`numThreadsPerBlock` is redeclared in dozens of
#    functions; a file-wide symbol table reads the wrong one, which is how the first
#    run of this check reported 35 findings of which 31 were its own arithmetic).
#    Anything it cannot resolve is counted UNRESOLVED and printed, never passed
#    silently.
#  * `#if 0` BLOCKS ARE SKIPPED, and finding that out cost a false positive:
#    ps_applyDeltaLaunch's only launch site with a block of 256 is inside `#if 0` in
#    PxgPBDParticleSystemCore.cpp:757-776, while the live one
#    (PxgParticleSystemCore.cpp:1519) uses 64 and matches the shader exactly.  Only
#    `#if 0` / `#if 0 ... #endif` is elided -- other conditionals are compiled in
#    some configuration and are read normally.
#  * IT SAYS NOTHING ABOUT WHETHER A STRIDE IS CORRECT, only that one is present --
#    the same honest limit check_grid_stride.py states about itself.
#  * A NAMED INCREMENT IS NOT A STRIDE (P9.97a).  Up to P9.97 rule 5 accepted ANY
#    `for (...; ...; x += name)`, which is check_grid_stride.slang_covers' marker.
#    initializeSolverVelocitiesTGS, computeAverageSolverBodyVelocityTGS and
#    propagateAverageSolverBodyVelocityTGS all carry
#        for (uint b = threadIdInWorkUnit; b < numSlabs; b += numThreadsPerBody)
#    -- the loop over ONE body's slabs, not a grid stride -- and passed on it while
#    running [numthreads(32,1,1)] against the engine's (32, 8) block: one eighth of
#    the CUDA's warps, i.e. 1/2 .. 1/8 of the bodies for numSlabs >= 5 and 1/8 of the
#    velocity-iteration propagate at any numSlabs, measured in the engine as a
#    frozen hub (PXG_SCENE=slabgrid).  A grid-stride loop STARTS at a global index, so
#    grid_covers() now requires the init clause of a named-increment loop to be rooted
#    (through at most three local assignments) in a parameter declared SV_GroupID or
#    SV_DispatchThreadID, or in WorkgroupCount().  On the unfixed tree the tightened
#    rule flagged exactly those three kernels and nothing else among the mapped ones.
#    What it still cannot decide: a rooted init with a stride of the wrong size.
#
#  * FIXED-GRID LAUNCHES ARE A DIFFERENT CLASS AND ARE NOT CHECKED HERE.  When the
#    host launches a constant grid, CUDA reaches grid*cudaBlock threads and the port
#    grid*slangThreads, so the port needs a loop whose stride is ITS OWN thread count.
#    That is check_grid_stride.py's territory, and that checker says plainly it finds
#    MISSING loops rather than WRONG ones.  The two cloth vertex contact gens on its
#    ratchet (blockDim 256, fixed grid 512, [numthreads(32,1,1)]) are exactly such a
#    case and belong there, not here.
#
# THE LIST IS A RATCHET, exactly as gpuslang.launchAbi's and gridStride's are, and it
# is EMPTY: after the convexCorePlane fix no mapped kernel in the tree is short by
# this rule.  An empty ratchet is the strongest state one can be in -- there is
# nothing to argue about, and the NEXT occurrence fails on the day it is written.
# A name left in block_coverage_known.txt after its port is fixed also fails the
# check; a stale allowlist is how a ratchet quietly stops ratcheting.

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GPUSLANG = os.path.dirname(HERE)
SRC = os.path.dirname(GPUSLANG)                  # .../physx/source
MANIFEST = os.path.join(GPUSLANG, "include", "PxgSlangKernelManifest.h")
SHADERS = os.path.join(GPUSLANG, "shaders")
KNOWN = os.path.join(HERE, "block_coverage_known.txt")

sys.path.insert(0, HERE)
import check_grid_stride as cgs  # strip_comments, brace_body, slang_covers


def strip_if_zero(text):
    """Remove `#if 0 ... #endif` regions, honouring nested #if/#ifdef/#ifndef.

    Everything else is left in place: a block under `#if PX_SUPPORT_GPU_PHYSX` is
    compiled in the configuration this port runs, and eliding it would hide real
    launch sites."""
    out = []
    depth = None          # nesting depth inside the elided region, or None
    for line in text.split("\n"):
        s = line.strip()
        if depth is None:
            if re.match(r"#\s*if\s+0\s*$", s):
                depth = 1
                out.append("")          # keep line numbering stable
                continue
            out.append(line)
        else:
            if re.match(r"#\s*if(n?def)?\b", s):
                depth += 1
            elif re.match(r"#\s*endif\b", s):
                depth -= 1
                if depth == 0:
                    depth = None
            out.append("")
    return "\n".join(out)


def int_defines():
    """Object-like integer `#define`s across the engine headers, name -> literal."""
    out = {}
    pat = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+(\d+)\s*(?://.*)?$", re.M)
    for dirpath, _d, files in os.walk(SRC):
        if os.sep + "gpuslang" in dirpath:
            continue
        for fn in files:
            if not fn.endswith((".h", ".cuh")):
                continue
            text = open(os.path.join(dirpath, fn), encoding="utf-8", errors="replace").read()
            for m in pat.finditer(text):
                out.setdefault(m.group(1), m.group(2))
    return out


def kernel_names():
    """PxgKernelIds enum member -> CUDA kernel name."""
    out = {}
    path = os.path.join(SRC, "gpucommon", "include", "PxgKernelNames.h")
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r'\s*KERNEL_DEF\(\s*(\w+)\s*,\s*"([^"]+)"\s*\)', line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def dim_constants():
    """(enum type, member) -> int over every *KernelBlockDim / *KernelGridDim enum.

    The type is part of the key on purpose: the headers give the block-dim and
    grid-dim enums THE SAME member names, and keying on the member alone silently
    reads one as the other (check_fixture_grid.py learned this first)."""
    out = {}
    for dirpath, _d, files in os.walk(SRC):
        if os.sep + "gpuslang" in dirpath:
            continue
        for fn in files:
            if not fn.endswith(".h"):
                continue
            text = open(os.path.join(dirpath, fn), encoding="utf-8", errors="replace").read()
            # `PxgNarrowPhaseBlockDims` is plural where every sibling is singular,
            # and leaving the `s` off the pattern left 12 launch sites unresolved.
            for em in re.finditer(r"\b(?:struct|namespace|enum(?:\s+class)?)\s+"
                                  r"(\w*(?:BlockDim|GridDim)s?)\b", text):
                body = cgs.brace_body(text, em.end())
                if body is None:
                    continue
                for mm in re.finditer(r"\b(\w+)\s*=\s*(\d+)\s*[,}]", body):
                    out[(em.group(1), mm.group(1))] = int(mm.group(2))
    return out


CONSTS = {}
# Object-like `#define`s over the engine headers.  The warps-per-block numbers the
# narrowphase launches with are all macros -- MIDPHASE_WARPS_PER_BLOCK,
# NP_TRIMESH_WARPS_PER_BLOCK, CONVEX_PLANE_WARPS_PER_BLOCK -- so without these 20-odd
# launch sites resolve to nothing and the check silently covers less than it claims.
# Only integer-valued macros are taken; anything else stays unresolved.
DEFINES = {}
# PAIRING A LAUNCH TO ITS KERNEL BY THE FUNCTION VARIABLE, NOT BY PROXIMITY.
# check_fixture_grid.py pairs on the nearest preceding getCuFunction and warns that
# this is a heuristic; here it produced a false finding on the first run.
# PxgNarrowphaseCore.cpp fetches BOTH convex-convex stages before launching either:
#
#     CUfunction kernelFunction_stage2 = ...getCuFunction(CONVEX_CONVEX_KERNEL_MAIN);
#     result = launchKernel(kernelFunction_stage1, ..., WARP_SIZE, EARLY_OUT/WARP_SIZE, 1, ...)
#
# so the stage-1 launch pairs with the stage-2 kernel and reports its block of 128
# against a shader that correctly matches stage 2's 32.  Reading the launch's FIRST
# argument and resolving that variable's nearest preceding assignment gets it right,
# and falls back to proximity only when the CUfunction is not a named variable.
GETFN = re.compile(r"getCuFunction\(\s*PxgKernelIds::(\w+)\s*\)")
GETFN_VAR = re.compile(r"\b(\w+)\s*=\s*[^;=]*?getCuFunction\(\s*PxgKernelIds::(\w+)\s*\)")
LAUNCH = re.compile(r"launchKernel\(\s*(\w+)\s*,"
                    r"\s*([^,]+),\s*([^,]+),\s*([^,]+),"      # grid x y z
                    r"\s*([^,]+),\s*([^,]+),\s*([^,]+),")     # block x y z
# Any local integral declaration, const or not: `numWarpsPerBlock` is frequently a
# plain PxU32 and skipping those left 44 launch sites unresolved.
DECL = re.compile(r"\b(?:static\s+)?(?:const\s+|constexpr\s+)?"
                  r"(?:PxU32|PxU16|PxU8|uint32_t|unsigned\s+int|unsigned|int|size_t)\s+"
                  r"(\w+)\s*=\s*([^;]+);")
NAMED = {"WARP_SIZE": 32}


def resolve(expr, decls, pos, depth=0):
    """An int for `expr` at file offset `pos`, or None.

    A name resolves to its NEAREST PRECEDING declaration, which is what makes this
    usable at all: `numThreadsPerBlock` is declared in dozens of functions in one
    file with different values."""
    e = expr.strip()
    if depth > 5:
        return None
    # Integer-literal suffixes only.  A blanket rstrip("uUlL") truncates every enum
    # member whose name ENDS in L -- RIGID_DYNAMIC_GET_LINVEL, CLOTH_SOLVESHELL,
    # INTEGRATE_CORE_PARALLEL -- and left 14 launch sites unresolved while looking
    # like a gap in the headers rather than a bug here.
    m = re.fullmatch(r"(\d+)[uUlL]*", e)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"(\w*(?:BlockDim|GridDim)s?)::(\w+)", e)
    if m:
        return CONSTS.get((m.group(1), m.group(2)))
    if e in NAMED:
        return NAMED[e]
    if e in DEFINES:
        return resolve(DEFINES[e], decls, pos, depth + 1)
    if re.fullmatch(r"\(\s*(.*\S)\s*\)", e):
        return resolve(re.fullmatch(r"\(\s*(.*\S)\s*\)", e).group(1), decls, pos, depth + 1)
    if re.fullmatch(r"[A-Za-z_]\w*", e):
        cand = [d for d in decls if d[1] == e and d[0] < pos]
        if not cand:
            return None
        return resolve(cand[-1][2], decls, cand[-1][0], depth + 1)
    m = re.fullmatch(r"([^*/]+)\*([^*/]+)", e)
    if m:
        a = resolve(m.group(1), decls, pos, depth + 1)
        b = resolve(m.group(2), decls, pos, depth + 1)
        if a is not None and b is not None:
            return a * b
    m = re.fullmatch(r"([^*/]+)/([^*/]+)", e)
    if m:
        a = resolve(m.group(1), decls, pos, depth + 1)
        b = resolve(m.group(2), decls, pos, depth + 1)
        if a is not None and b:
            return a // b
    return None


def expander(decls):
    """Textual expansion of an expression at a file offset.

    Returns (expandedText, namesTouched).  THE NAME SET IS THE POINT: expanding
    `numBlocks` all the way turns `numThreadsPerBlock` into `1024`, so a test that
    looks for the block's IDENTIFIER in the finished string finds nothing and the
    derivation is missed -- which is exactly how the first version of this checker
    passed the very defect it was written for.  Recording every name substituted
    along the way keeps `numThreadsPerBlock` visible even after its value replaces
    it."""
    def expand(expr, pos, depth=0, seen=None):
        if seen is None:
            seen = set()
        if depth > 5:
            return expr, seen
        def sub(m):
            w = m.group(0)
            cand = [d for d in decls if d[1] == w and d[0] < pos]
            if cand:
                seen.add(w)
                return "(" + expand(cand[-1][2], cand[-1][0], depth + 1, seen)[0] + ")"
            seen.add(w)
            return w
        return re.sub(r"\b[A-Za-z_]\w*\b", sub, expr), seen
    return expand


def grid_from_block(m, decls, expand, block):
    """True when the launch site derives its GRID from its BLOCK.

    THIS IS THE DISCRIMINATOR, and without it the check reports the whole
    warp-per-object family as findings.  Two shapes exist in the engine:

      numBlocks = ceil(n / numThreadsPerBlock)   -- the grid ENCODES the block
          width.  The bridge forwards that grid to a group of a DIFFERENT width,
          so a narrower group covers a fraction of the work.  This is the defect.

      numBlocks = nbActiveSoftBodies             -- one block per object.  The
          block width is how much parallelism the object gets, not how much of
          the work is reached; a narrower group that loops over the object's
          items is output-identical.  softBodyRefitBda.slang:246 is exactly this
          ("each warp deals with one node ... one warp per group here ->
          warpIndex == 0, stride 1"), and flagging it would be manufacturing work.

    Two tests, either sufficient:
      * a NAME the block expression uses is touched while expanding the grid;
      * the expanded grid DIVIDES by a value equal to the resolved block product,
        which catches a site that writes the divisor as a literal.
    """
    grid, seen = expand(m.group(2).strip(), m.start())
    names = set()
    for i in (5, 6, 7):
        names |= {n for n in re.findall(r"[A-Za-z_]\w*", m.group(i)) if len(n) > 2}
    if names & seen:
        return True
    if block:
        for dm in re.finditer(r"/\s*(\(?[^)(/]+\)?)", grid):
            v = resolve(dm.group(1), decls, m.start())
            if v == block:
                return True
    return False


def engine_blocks(names):
    """CUDA kernel name -> (maxResolvedBlock or None, [unresolved exprs], gridFromBlock)."""
    out = {}
    for dirpath, _d, files in os.walk(SRC):
        if os.sep + "gpuslang" in dirpath:
            continue
        for fn in files:
            if not fn.endswith(".cpp"):
                continue
            path = os.path.join(dirpath, fn)
            text = strip_if_zero(open(path, encoding="utf-8", errors="replace").read())
            decls = [(m.start(), m.group(1), m.group(2)) for m in DECL.finditer(text)]
            expand = expander(decls)
            marks = [(m.start(), m.group(1)) for m in GETFN.finditer(text)]
            assigns = [(m.start(), m.group(1), m.group(2)) for m in GETFN_VAR.finditer(text)]
            for m in LAUNCH.finditer(text):
                var = m.group(1)
                byvar = [a for a in assigns if a[1] == var and a[0] < m.start()]
                if byvar:
                    enum = byvar[-1][2]
                else:
                    prev = [nm for pos, nm in marks if pos < m.start()]
                    if not prev:
                        continue
                    enum = prev[-1]
                cuda = names.get(enum)
                if cuda is None:
                    continue
                vals = [resolve(m.group(i), decls, m.start()) for i in (5, 6, 7)]
                slot = out.setdefault(cuda, [None, [], False])
                if all(v is not None for v in vals):
                    prod = vals[0] * vals[1] * vals[2]
                    slot[0] = prod if slot[0] is None else max(slot[0], prod)
                    if grid_from_block(m, decls, expand, prod):
                        slot[2] = True
                else:
                    slot[1].append("%s:%s" % (fn, ",".join(m.group(i).strip() for i in (5, 6, 7))))
    return out


NUMTHREADS = re.compile(r"\[numthreads\(\s*([^)]*)\)\s*\]\s*(?:\[[^\]]*\]\s*)*"
                        r"(?:void|\w[\w:<>]*)\s+(\w+)\s*\(", re.S)
# `public static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;` -- the `public` is what
# a module exports, and leaving it out of the pattern (together with not following
# `import`) made the checker report the two cloth VT midphases as findings when their
# partition loop is right there, using a constant from pxgSlangClothMidphase.
SLANG_CONST = re.compile(r"\b(?:public\s+)?(?:static\s+)?const\s+u?int\w*\s+(\w+)\s*=\s*([^;]+);")
SLANG_IMPORT = re.compile(r"^\s*import\s+(\w+)\s*;", re.M)


def slang_val(scope, expr, depth=0):
    """An int for a Slang expression using the file's `static const uint` scope."""
    e = expr.strip().rstrip("uU")
    if re.fullmatch(r"\d+", e):
        return int(e)
    if depth > 4:
        return None
    if e in scope:
        return slang_val(scope, scope[e], depth + 1)
    m = re.fullmatch(r"([^*]+)\*([^*]+)", e)
    if m:
        a = slang_val(scope, m.group(1), depth + 1)
        b = slang_val(scope, m.group(2), depth + 1)
        if a is not None and b is not None:
            return a * b
    return None


def slang_entries(mutate=None):
    """(shader stem, entry) -> (threadProduct or None, fileText, bodyOffset, constScope).

    `mutate(fileName, text)`, when given, rewrites a file's text in memory first
    (the selftest's hook)."""
    out = {}
    for fn in sorted(os.listdir(SHADERS)):
        if not fn.endswith(".slang"):
            continue
        raw = open(os.path.join(SHADERS, fn), encoding="utf-8", errors="replace").read()
        if mutate is not None:
            raw = mutate(fn, raw)
        text = cgs.strip_comments(raw)
        scope = {}
        for mod in SLANG_IMPORT.findall(text):          # one level of imports
            mpath = os.path.join(SHADERS, mod + ".slang")
            if os.path.exists(mpath):
                mtext = cgs.strip_comments(open(mpath, encoding="utf-8",
                                                errors="replace").read())
                scope.update({m.group(1): m.group(2) for m in SLANG_CONST.finditer(mtext)})
        scope.update({m.group(1): m.group(2) for m in SLANG_CONST.finditer(text)})
        for m in NUMTHREADS.finditer(text):
            prod, ok = 1, True
            for part in m.group(1).split(","):
                v = slang_val(scope, part)
                if v is None:
                    ok = False
                else:
                    prod *= v
            out[(fn[:-6], m.group(2))] = (prod if ok else None, text, m.end(), scope)
    return out


def serial_factor(text, body, val):
    """Largest constant trip count of a `for` in `body` (or a same-file helper).

    THE FIFTH ACCEPTANCE, and it exists because without it this check reported
    twelve findings of which twelve were ports that DO cover their work.  The
    corpus idiom for a CUDA block of `WARP_SIZE x numWarpsPerBlock` is a 32-lane
    Slang group -- this port's virtual warp -- that walks the block's other warps
    SERIALLY:

        for (uint sub = 0u; sub < MIDPHASE_WARPS_PER_BLOCK; ++sub)
        {
            const uint cmIdx = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub;

    convexTrimeshMidphaseBda.slang:227 says so in its own header ("the two warps
    never interact ... so the port walks them SERIALLY inside a 32-thread group").
    check_grid_stride.slang_covers() cannot see it: the increment is `++sub` and
    `sub` is ADDED to the block offset rather than scaling a named product, so
    neither its stride markers nor its counted-loop marker matches.

    A group of `slangThreads` running such a loop `N` times covers `slangThreads*N`
    of the block's work, which is why the caller multiplies rather than just
    accepting the presence of a loop."""
    best = 1
    scopes = [body]
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", body):
        name = m.group(1)
        if name in ("if", "for", "while", "switch", "return"):
            continue
        fm = re.search(r"\b\w[\w<>,\s\*]*\s" + re.escape(name) + r"\s*\([^;{]*\)\s*\{", text)
        if fm:
            hb = cgs.brace_body(text, fm.end() - 1)
            if hb:
                scopes.append(hb)
    for scope in scopes:
        for parts in cgs.for_headers(scope):
            m = re.search(r"<\s*([^;]+?)\s*$", parts[1])
            if not m:
                continue
            bound = m.group(1).strip()
            v = val(bound)
            if v is None or v <= best:
                continue
            vm = re.search(r"\b(\w+)\s*(?:=|<)", parts[0] + ";" + parts[1])
            if vm is None:
                continue
            var = vm.group(1)
            # THE LOOP MUST BE THE BLOCK PARTITION, NOT JUST ANY BOUNDED LOOP.
            # Taking the largest constant `for` bound anywhere in the body would
            # clear a kernel on an unrelated inner loop -- `for (w = 0; w <
            # BODYSIM_WORDS; ++w)` covers nothing.  What identifies the partition
            # is that the variable is ADDED to a group offset SCALED BY THE SAME
            # BOUND: `cmIdx = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub`.  Both halves
            # are required, so an inner loop over vertices cannot pass for one.
            b, x = re.escape(bound), re.escape(var)
            if re.search(r"\*\s*" + b + r"\s*\+\s*" + x + r"\b", scope) or \
               re.search(r"\b" + x + r"\s*\+\s*\w[\w.]*\s*\*\s*" + b + r"\b", scope):
                best = v
    return best


SV_GRID = re.compile(r"\b(\w+)\s*:\s*SV_(?:GroupID|DispatchThreadID)\b")


def entry_sig(text, off):
    """The parameter list of the entry whose `name(` ends at `off`."""
    i = text.find("{", off)
    return text[off:i] if i >= 0 else ""


def sv_rooted(body, sig, expr, depth=0):
    """True if `expr` is rooted in a grid-wide system value.

    A grid-stride loop starts at the GLOBAL index -- gid.x * W + lane, or the
    dispatch-thread id -- and steps by the grid.  An intra-item loop starts at a
    lane-in-work-unit and steps by the work unit.  Both are `x += name`; only the
    start tells them apart.  Local assignments are followed up to three levels
    (`warpIndex = gid.x * 8 + gtid.y; bodyId = warpIndex * bpw + ...; i = bodyId`)."""
    if re.search(r"WorkgroupCount\s*\(", expr):
        return True
    for n in SV_GRID.findall(sig):
        if re.search(r"\b" + re.escape(n) + r"\b", expr):
            return True
    if depth >= 3:
        return False
    for ident in set(re.findall(r"\b[A-Za-z_]\w*\b", expr)):
        for m in re.finditer(r"\b" + re.escape(ident) + r"\s*=\s*([^;=][^;]*);", body):
            if sv_rooted(body, sig, m.group(1), depth + 1):
                return True
    return False


def grid_covers(text, body, sig):
    """Rule 5: the port carries a construct that covers the whole grid's work.

    check_grid_stride.slang_covers() minus its blind spot (see the header's P9.97a
    note): a named-increment `for` counts only when its init clause is sv_rooted.
    The one level of same-file helpers is kept for WorkgroupCount() and counted
    loops; a helper's named-increment loop starts at a PARAMETER, which nothing
    here can root, so it is not accepted."""
    if cgs.SLANG_STRIDE.search(body):
        return True
    for parts in cgs.for_headers(body):
        if (cgs.NAMED_INC.search(parts[2]) or cgs.SLANG_STRIDE.search(parts[2])) \
                and sv_rooted(body, sig, parts[0]):
            return True
    if cgs.counted_covers(body):
        return True
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", body):
        name = m.group(1)
        if name in ("if", "for", "while", "switch", "return"):
            continue
        fm = re.search(r"\b\w[\w<>,\s\*]*\s" + re.escape(name) + r"\s*\([^;{]*\)\s*\{", text)
        if fm is None:
            continue
        hb = cgs.brace_body(text, fm.end() - 1)
        if hb is not None and (cgs.SLANG_STRIDE.search(hb) or cgs.counted_covers(hb)):
            return True
    return False


def load_manifest():
    rows = []
    with open(MANIFEST, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            m = re.match(r'\s*\{\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\}', line)
            if m:
                rows.append((m.group(1), m.group(2), m.group(3), n))
    return rows


def load_known():
    out = set()
    if os.path.exists(KNOWN):
        for line in open(KNOWN, encoding="utf-8"):
            line = line.split("#", 1)[0].strip()
            if line:
                out.add(line)
    return out


def scan(blocks, entries):
    """(findings, unresolved, checked) over every mapped row with a launch site."""
    findings, unresolved, checked = [], [], 0
    for cuda, spv, entry, line in load_manifest():
        if cuda not in blocks:
            continue                      # no engine launch site: nothing to compare
        cb, bad, derived = blocks[cuda]
        ent = entries.get((spv[:-4], entry))
        if ent is None:
            unresolved.append("%s: no [numthreads] found for %s in %s" % (cuda, entry, spv))
            continue
        st, text, off, scope = ent
        if st is None:
            unresolved.append("%s: [numthreads] not a constant in %s" % (cuda, spv))
            continue
        if cb is None:
            unresolved.append("%s: engine blockDim unresolved (%s)" % (cuda, "; ".join(bad[:2])))
            continue
        checked += 1
        if st >= cb:
            continue
        if not derived:
            continue          # grid is not sized from the block: see grid_from_block
        body = cgs.brace_body(text, off - 1)
        if body is not None and grid_covers(text, body, entry_sig(text, off)):
            continue
        if body is not None:
            factor = serial_factor(text, body, lambda e: slang_val(scope, e))
            if st * factor >= cb:
                continue
        findings.append((cuda, spv, entry, cb, st, line))
    return findings, unresolved, checked


# The negative control (P9.97a), in the style of check_barrier_uniformity.py: put the
# shipped P9.97 geometry back into initializeSolverVelocitiesTGS in memory -- a one-warp
# group whose body still carries the per-body slab loop `b += numThreadsPerBody` --
# and demand a finding.  That loop is exactly what the pre-P9.97a rule 5 took for a
# grid stride; a green run of this file therefore also says the tightening is awake.
SELFTEST_FILE = "solverInitVelocitiesTGS.slang"
SELFTEST_FROM = "[numthreads(32, INITVEL_WARPS_PER_BLOCK, 1)]"
SELFTEST_TO = "[numthreads(32, 1, 1)]"
SELFTEST_KERNEL = "initializeSolverVelocitiesTGS"


def selftest(blocks):
    def mutate(fn, text):
        if fn != SELFTEST_FILE:
            return text
        if text.count(SELFTEST_FROM) != 1:
            print("SELFTEST BROKEN: the anchor %r is gone from %s." % (SELFTEST_FROM, SELFTEST_FILE))
            print("  Re-anchor it on whatever now spells that kernel's [numthreads]; do NOT")
            print("  delete the selftest.")
            sys.exit(2)
        return text.replace(SELFTEST_FROM, SELFTEST_TO)
    findings, _, _ = scan(blocks, slang_entries(mutate))
    if SELFTEST_KERNEL not in {f[0] for f in findings}:
        print("SELFTEST FAILED: %s with a one-warp group was not reported short." % SELFTEST_KERNEL)
        print("  The per-body slab loop is being accepted as a grid stride again (P9.97a).")
        return 2
    print("selftest: %s with [numthreads(32,1,1)] -> reported short.  The checker can "
          "still see it." % SELFTEST_KERNEL)
    return 0


def main():
    global CONSTS, DEFINES
    CONSTS = dim_constants()
    DEFINES = int_defines()
    names = kernel_names()
    blocks = engine_blocks(names)
    rc = selftest(blocks)
    if rc:
        return rc
    entries = slang_entries()
    known = load_known()

    findings, unresolved, checked = scan(blocks, entries)

    found = {f[0] for f in findings}
    new = sorted(found - known)
    stale = sorted(known - found)

    for cuda, spv, entry, cb, st, line in sorted(findings, key=lambda f: -(f[3] // max(f[4], 1))):
        mark = "KNOWN" if cuda in known else "NEW  "
        print("  [%s] %-46s %s:%s  engine block %d, group %d -> reaches 1/%d of the work"
              % (mark, cuda, spv, entry, cb, st, cb // max(st, 1)))
    for u in unresolved:
        print("  [UNRES] %s" % u)
    print("checked %d mapped kernels with a resolved launch site; %d short, %d unresolved"
          % (checked, len(findings), len(unresolved)))

    rc = 0
    if new:
        print("FAIL: %d kernel(s) reach only a fraction of their work and are not on the "
              "ratchet:" % len(new))
        for n in new:
            print("    %s" % n)
        print("  Give the port a covering loop (stride = WorkgroupCount().x * BLOCK), or")
        print("  widen [numthreads] to the engine's block.  See this file's header.")
        rc = 1
    if stale:
        print("FAIL: %d name(s) on the ratchet no longer report short -- remove them from"
              " %s:" % (len(stale), os.path.basename(KNOWN)))
        for s in stale:
            print("    %s" % s)
        rc = 1
    if rc == 0:
        print("OK: no mapped kernel silently reaches a fraction of its work.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
