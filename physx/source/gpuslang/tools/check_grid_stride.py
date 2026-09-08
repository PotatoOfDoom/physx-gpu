#!/usr/bin/env python3
# check_grid_stride.py -- P9.59.  The ELEVENTH static checker, and the THIRD
# written because its defect had already happened.
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS
# ---------------------------------------------------------------------------
# Many PhysX CUDA kernels cover their work with a GRID-STRIDE LOOP:
#
#     for (PxU32 i = blockIdx.x * blockDim.x + threadIdx.x;
#          i < n;
#          i += gridDim.x * blockDim.x)
#
# The loop is not decoration.  The host may launch a grid SMALLER than the work --
# often because the trip count is a DEVICE value (`*nbPairsGlobal`,
# `*nbSecondPassPairs`) while the grid is computed on the host from a shape count --
# and the kernel covers the remainder by iterating.  A Slang port that drops the
# loop and makes exactly one pass processes only the first `gridDim.x * groupSize`
# items and leaves the rest untouched.
#
# THAT IS NOT A GRACEFUL DEGRADATION.  P9.52b-6 measured the sphere-vs-heightfield
# core without its loop: every pair past the ceiling left its `ni` / `interm` /
# `ordered` / `maxDepth` record UNWRITTEN, and `convexTrimeshCorrelate` -- which
# reads a contact count out of `ni.index` and then dereferences the contact stream --
# read uninitialised memory and dereferenced it.  SIGSEGV, deterministic, and inside
# llvmpipe's JIT where the backtrace is `?? ()`.  P9.58 measured the convex
# heightfield core: a half-3 box fell straight THROUGH the terrain, final y = -70.8
# and still accelerating at 39 m/s.
#
# ---------------------------------------------------------------------------
# WHY NEITHER A FIXTURE NOR A MUTANT FINDS IT
# ---------------------------------------------------------------------------
# A fixture dispatches the grid IT chooses, sized to its own case, so the ceiling is
# never approached: every probe in this suite launches a handful of items.  A mutant
# sweep is worse than blind here -- it mutates the shader against a fixture that
# cannot see the difference, so the mutant SURVIVES and reads as an equivalence.
#
# Scenes can find it, but only a scene whose work exceeds the ceiling, and the
# ceilings differ by four orders of magnitude between siblings.  P9.58's measurement:
# the convex heightfield core is launched with 32 warps per contact manager, the
# trimesh post process with at least 524288 threads.  A scene tuned to one says
# nothing about the other, and for the heightfield POST PROCESS the largest box the
# harness can place on the terrain does not reach its 256 either.  Only a static
# check covers the whole set at once.
#
# ---------------------------------------------------------------------------
# WHAT IT CHECKS
# ---------------------------------------------------------------------------
# For every MAPPED row of PxgSlangKernelManifest.h:
#   1. find the CUDA `__global__` of that name and brace-match its body;
#   2. decide whether that body carries a grid-stride construct;
#   3. find the named entry point in the named .slang and brace-match its body;
#   4. decide whether the port reproduces a stride -- see slang_covers() below, which
#      accepts `WorkgroupCount()`, a named stride in a `for` increment, and either of
#      those one level down in a same-file helper.
#
# A kernel with (1)-(2) but not (4) is a finding.
#
# THE MARKER IN (4) WAS WRONG IN THE FIRST VERSION and it mattered: looking only for
# `WorkgroupCount()` reported 38 omissions of which 26 were ports that loop.  The
# bridge forwards only gridDim, so the CUDA blockDim has no runtime spelling and the
# corpus writes the stride as a file-scope constant instead.  slang_covers() carries
# the full account; it is repeated here because a stale claim at the top of a file is
# what the next reader believes.
#
# THE LIST IS A RATCHET, exactly as gpuslang.launchAbi's is.  The kernels known to
# omit the loop live in grid_stride_known.txt.  A NEW omission fails the check.  An
# omission that has been FIXED but left in the file also fails it -- a stale
# allowlist is how a ratchet quietly stops ratcheting.

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GPUSLANG = os.path.dirname(HERE)
SRC = os.path.dirname(os.path.dirname(GPUSLANG))
MANIFEST = os.path.join(GPUSLANG, "include", "PxgSlangKernelManifest.h")
SHADERS = os.path.join(GPUSLANG, "shaders")
KNOWN = os.path.join(HERE, "grid_stride_known.txt")


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def brace_body(text, start):
    """Return the {...} body beginning at or after `start`, or None."""
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
    return None


# Finding the kernels is not the regex it looks like.  PhysX writes the declaration
# three ways, and the first version of this checker saw only the first:
#
#     extern "C" __global__ void name(                      -- plain
#     extern "C" __global__ __launch_bounds__(...) void name(   -- attribute BEFORE void
#     extern "C" __global__ void __launch_bounds__(...) name(   -- attribute AFTER void
#
# The second shape covers 50 mapped kernels including `convexHeightfieldNarrowphase`,
# which P9.58 had just given a grid-stride loop -- so the check had been silently
# skipping the very kernel whose defect motivated it.  A scanner that walks tokens
# after `__global__` and takes the last identifier before the parameter list handles
# all three, and cannot be quietly defeated by a fourth.
def kernel_decls(text):
    """Yield (kernelName, offsetJustPastTheOpenParen) for every __global__."""
    for m in re.finditer(r"__global__\b", text):
        i = m.end()
        last = None
        while i < len(text):
            ch = text[i]
            if ch.isspace():
                i += 1
                continue
            if ch == "(":
                depth = 0
                for j in range(i, len(text)):
                    if text[j] == "(":
                        depth += 1
                    elif text[j] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                else:
                    break
                # A parenthesis right after an identifier that is NOT an attribute
                # opens the parameter list; after an attribute it belongs to it.
                if last and not last.startswith("__"):
                    yield last, j + 1
                    break
                i = j + 1
                continue
            if ch == "_" or ch.isalnum():
                k = i
                while k < len(text) and (text[k] == "_" or text[k].isalnum()):
                    k += 1
                word = text[i:k]
                if word not in ("void", "static", "inline", "const"):
                    last = word
                i = k
                continue
            break


def load_manifest():
    rows = []
    with open(MANIFEST, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            m = re.match(r'\s*\{\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\}', line)
            if m:
                rows.append((m.group(1), m.group(2), m.group(3), n))
    return rows


def cuda_sources():
    out = []
    for root, _dirs, files in os.walk(SRC):
        if os.sep + "CUDA" not in root:
            continue
        for fn in files:
            if fn.endswith((".cu", ".cuh")):
                out.append(os.path.join(root, fn))
    return out


# A grid stride can only be written with gridDim -- but the expression appearing
# ANYWHERE in a kernel is not enough.  `gridDim.x * blockDim.x` also turns up as a
# total-thread count in single-pass kernels that then do `if (idx < n)`, and reading
# that as a loop over-reports by a factor of three.  What makes it a stride is that
# it sits in the INCREMENT clause of a `for`, so that is what is matched: the third
# clause of a brace-balanced `for` header.
STRIDE_EXPR = re.compile(r"gridDim\s*\.\s*x\s*\*\s*blockDim\s*\.\s*[xyz]"
                         r"|blockDim\s*\.\s*[xyz]\s*\*\s*gridDim\s*\.\s*x")


def for_increments(body):
    """Yield the third clause of every `for` header in `body`."""
    for m in re.finditer(r"\bfor\s*\(", body):
        depth = 0
        for j in range(m.end() - 1, len(body)):
            if body[j] == "(":
                depth += 1
            elif body[j] == ")":
                depth -= 1
                if depth == 0:
                    header = body[m.end():j]
                    parts = []
                    d = 0
                    cur = []
                    for ch in header:
                        if ch in "([":
                            d += 1
                        elif ch in ")]":
                            d -= 1
                        if ch == ";" and d == 0:
                            parts.append("".join(cur))
                            cur = []
                        else:
                            cur.append(ch)
                    parts.append("".join(cur))
                    if len(parts) == 3:
                        yield parts[2]
                    break


# The stride is not always written inline.  utility.cu's skinning kernels hoist it:
#     const PxU32 xDim = gridDim.x * blockDim.x;
#     for (i = blockIdx.x*blockDim.x + threadIdx.x; i < loopEnd; i += xDim)
# so a checker that only reads the increment clause misses five mapped kernels that
# an adversarial audit had already confirmed.  Named strides are resolved one level.
NAMED_STRIDE = re.compile(r"\b(\w+)\s*=\s*[^;]*?(?:gridDim\s*\.\s*x\s*\*\s*blockDim"
                          r"|blockDim\s*\.\s*[xyz]\s*\*\s*gridDim)[^;]*;")


def cuda_has_stride(body):
    incs = list(for_increments(body))
    if any(STRIDE_EXPR.search(inc) for inc in incs):
        return True
    names = {m.group(1) for m in NAMED_STRIDE.finditer(body)}
    if not names:
        return False
    return any(re.search(r"\+=\s*(" + "|".join(map(re.escape, names)) + r")\b", inc)
               for inc in incs)
# ---------------------------------------------------------------------------
# THE SLANG SIDE IS NOT `WorkgroupCount()`, AND ASSUMING IT WAS MADE THIS CHECKER
# REPORT MOSTLY NOISE (P9.60b)
# ---------------------------------------------------------------------------
# The first version looked for `WorkgroupCount(` in the entry body and called
# anything else a one-pass port.  It reported 38 omissions; a reading of 27 of them
# found almost all were PORTS THAT DO LOOP.
#
# The reason is in the bridge.  `PxgVulkanCudaContext::launchKernel` takes
# `unsigned int /*blockDimX*/, /*blockDimY*/, /*blockDimZ*/` -- the names are
# commented out -- and forwards only gridDimX/Y/Z to vkCmdDispatch.  THE CUDA
# blockDim HAS NO RUNTIME SPELLING ON THIS PORT.  So a shader reproducing
# `i += gridDim.x * blockDim.y` cannot write it as an expression at all; the corpus
# idiom is a file-scope `static const` holding the product for the ONE launch site,
# e.g. `BTRI_GRID * BTRI_WARPS_PER_BLOCK` = 4096*2.  Several shaders say so in their
# headers and warn that deriving the stride from the group count instead would
# double-visit or skip.
#
# Two further blind spots the same reading exposed:
#   * the loop is often in a HELPER in the same file, with the entry a two-line
#     wrapper -- reading only the entry body finds nothing;
#   * `WorkgroupCount()` is still a legitimate spelling (P9.52b-6 and P9.58 use it,
#     where the group count and the CUDA grid coincide), so both must be accepted.
#
# What is checked now: the entry, or any same-file function it calls, contains a
# `for` whose increment advances by a NAMED stride or by an expression involving
# WorkgroupCount.  A loop stepping by 1 does not count.
#
# THIS IS DELIBERATELY WEAKER THAN "the stride is correct".  A port looping with the
# WRONG constant passes.  Deciding that needs the launch site, and the launch site is
# in C++ the checker does not parse -- so the honest statement is that this finds
# MISSING loops, not wrong ones, and the file says so rather than implying more.
SLANG_STRIDE = re.compile(r"WorkgroupCount\s*\(")
NAMED_INC = re.compile(r"\+=\s*([A-Za-z_]\w*)")


def slang_covers(text, body):
    if SLANG_STRIDE.search(body):
        return True
    for inc in for_increments(body):
        if SLANG_STRIDE.search(inc) or NAMED_INC.search(inc):
            return True
    # one level of same-file helper calls
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", body):
        name = m.group(1)
        if name in ("if", "for", "while", "switch", "return"):
            continue
        fm = re.search(r"\b\w[\w<>,\s\*]*\s" + re.escape(name) + r"\s*\([^;{]*\)\s*\{", text)
        if fm is None:
            continue
        hb = brace_body(text, fm.end() - 1)
        if hb is None:
            continue
        if SLANG_STRIDE.search(hb) or any(
                SLANG_STRIDE.search(i) or NAMED_INC.search(i) for i in for_increments(hb)):
            return True
    return False


def main():
    rows = load_manifest()
    bodies = {}
    for path in cuda_sources():
        text = strip_comments(open(path, encoding="utf-8", errors="replace").read())
        for name, end in kernel_decls(text):
            if name in bodies:
                continue
            body = brace_body(text, end)
            if body is None:
                # The brace scan fails on a handful of kernels -- an asymmetric
                # `#if` arm, or a brace the comment stripper cannot see -- and
                # returning None there silently drops the kernel from the check.
                # Falling back to "everything up to the next kernel" is a coarser
                # body but a sound one for deciding whether a `for` increments by
                # the grid stride.
                nxt = next((e for n2, e in kernel_decls(text) if e > end), len(text))
                body = text[end:nxt]
            bodies[name] = (os.path.relpath(path, SRC), body)

    findings = []
    checked = strided = 0
    for cuda_name, spv, entry, line in rows:
        hit = bodies.get(cuda_name)
        if hit is None:
            continue                      # not a CUDA __global__ we can locate
        checked += 1
        cuda_file, cuda_body = hit
        if not cuda_has_stride(cuda_body):
            continue                      # nothing to reproduce
        strided += 1

        stem = spv[:-4] if spv.endswith(".spv") else spv
        slang = os.path.join(SHADERS, stem + ".slang")
        if not os.path.isfile(slang):
            findings.append((cuda_name, cuda_file, stem + ".slang",
                             "shader named by the manifest does not exist"))
            continue
        stext = strip_comments(open(slang, encoding="utf-8", errors="replace").read())
        em = re.search(r"\bvoid\s+" + re.escape(entry) + r"\s*\(", stext)
        if em is None:
            findings.append((cuda_name, cuda_file, stem + ".slang",
                             "entry point '%s' not found in the shader" % entry))
            continue
        ebody = brace_body(stext, em.end())
        if ebody is None:
            findings.append((cuda_name, cuda_file, stem + ".slang",
                             "entry point '%s' has no body" % entry))
            continue
        if not slang_covers(stext, ebody):
            findings.append((cuda_name, cuda_file, stem + ".slang",
                             "CUDA loops over the grid; the port makes ONE pass"))

    known = set()
    if os.path.isfile(KNOWN):
        for line in open(KNOWN, encoding="utf-8"):
            line = line.split("#", 1)[0].strip()
            if line:
                known.add(line)

    found = {f[0] for f in findings}
    new = sorted(found - known)
    stale = sorted(known - found)

    unlocated = [r[0] for r in rows if r[0] not in bodies]
    print("grid-stride check over %d mapped kernels" % len(rows))
    print("  located in CUDA: %d   of those looping over the grid: %d"
          % (checked, strided))
    print("  ports that reproduce the stride: %d   that do not: %d"
          % (strided - len(findings), len(findings)))

    if new:
        print("\n%d NEW GRID-STRIDE OMISSION(S):\n" % len(new))
        for name, cuda_file, slang, why in findings:
            if name in new:
                print("  %-44s %s" % (name, why))
                print("      CUDA  %s" % cuda_file)
                print("      Slang %s" % slang)
        print("\nEither give the port the loop -- an outer")
        print("  for (uint base = 0u; base < n; base += WorkgroupCount().x * BLOCK)")
        print("whose trip count is workgroup-uniform, so barriers inside stay in")
        print("uniform control flow -- or, if the omission is deliberate, add the")
        print("kernel to")
        print("  %s" % os.path.relpath(KNOWN, os.getcwd()))
        print("with the one-line reason.  An entry there is a claim somebody checked.")

    if stale:
        print("\n%d STALE ENTRY(IES) in grid_stride_known.txt:\n" % len(stale))
        for name in stale:
            print("  %s -- the port now reproduces the stride; remove the line" % name)
        print("\nA ratchet that keeps fixed kernels on its list stops ratcheting.")

    # A kernel the scanner cannot find is not checked AT ALL, which is the failure
    # mode that hid 52 of them behind a regex.  The count is reported and ratcheted
    # so the blind spot cannot grow back in silence.
    if unlocated:
        print("\n%d MAPPED KERNEL(S) NOT LOCATED IN CUDA -- these are NOT checked:\n"
              % len(unlocated))
        for n in unlocated:
            print("  %s" % n)
        print("\nEither the declaration has a shape kernel_decls() does not walk, or")
        print("the name is not a CUDA __global__ at all.  Both need a look: an")
        print("unlocated kernel is an unchecked one.")

    if new or stale or unlocated:
        return 1
    print("  known omissions carried: %d" % len(known))
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
