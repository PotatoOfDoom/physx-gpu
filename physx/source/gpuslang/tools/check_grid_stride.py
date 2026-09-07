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
#   4. decide whether the port reproduces a stride -- `WorkgroupCount()`, which is
#      the only way to spell `gridDim.x` in Slang, is the marker.
#
# A kernel with (1)-(2) but not (4) is a finding.
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
# WorkgroupCount() is Slang's only spelling of gridDim.
SLANG_STRIDE = re.compile(r"WorkgroupCount\s*\(")


def main():
    rows = load_manifest()
    bodies = {}
    for path in cuda_sources():
        text = strip_comments(open(path, encoding="utf-8", errors="replace").read())
        for m in re.finditer(r'__global__\s+void\s+(\w+)\s*\(', text):
            body = brace_body(text, m.end())
            if body is not None and m.group(1) not in bodies:
                bodies[m.group(1)] = (os.path.relpath(path, SRC), body)

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
        if not SLANG_STRIDE.search(ebody):
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

    if new or stale:
        return 1
    print("  known omissions carried: %d" % len(known))
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
