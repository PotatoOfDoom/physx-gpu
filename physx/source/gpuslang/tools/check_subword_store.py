#!/usr/bin/env python3
# check_subword_store.py -- P9.83a.  The SEVENTEENTH static checker, and the first
# one written because a mutant SURVIVED rather than because a defect shipped.
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS
# ---------------------------------------------------------------------------
# This device does not enable 8-bit or 16-bit storage (no VK_KHR_8bit_storage /
# storageBuffer8BitAccess is requested by PxgVulkanComputeDevice), so every PxU8
# and PxU16 the engine's buffers hold has to be written through the 32-bit word
# that contains it.  There are two ways to do that and they are NOT
# interchangeable:
#
#   THE PLAIN READ-MODIFY-WRITE
#       w[0] = (w[0] & ~(0xFFu << shift)) | ((value & 0xFFu) << shift);
#   is correct only when ONE invocation in the whole dispatch touches that word.
#   If two invocations own two different BYTES of the same word, each reads the
#   word before the other's write lands and the loser's byte is silently
#   reverted.  Four PxU8 lanes share a word, so four writers is the normal case,
#   not a corner one.
#
#   THE ATOMIC PAIR
#       InterlockedAnd(w[0], ~(0xFFu << shift), prev);
#       if (value) InterlockedOr(w[0], value << shift, prev);
#   touches only this lane's bits in each operation, so any interleaving of any
#   number of writers lands the same word.  The AND is what makes an UNINITIALISED
#   allocation safe; an OR on its own leaves stale bits in the lane whenever the
#   value is zero, and a non-zero byte reads as `true`.
#
# ---------------------------------------------------------------------------
# WHAT MADE IT A CHECKER RATHER THAN A COMMENT
# ---------------------------------------------------------------------------
# P9.83a's mutation sweep replaced triTriOverlapBda's atomic pair with the plain
# read-modify-write and the mutant SURVIVED.  It survived on merit: the fixture
# cannot see it.  The lost-update needs two workgroups to be inside the same word
# at the same instant, llvmpipe is free to run them one after the other, and a
# fixture that only fails when a race happens to manifest is a flaky test, not a
# measurement.  Every other survivor in that sweep was answered by strengthening
# the fixture; this one cannot be, because the defect is not a function of the
# input.
#
# That is exactly the situation a static rule is for.  The SHAPE of the write is
# decidable by reading it; whether the race fires is not.
#
# ---------------------------------------------------------------------------
# THE RULE
# ---------------------------------------------------------------------------
# For every shader and module source:
#
#   1. a masked sub-word write to an INDEXED destination (`w[0] = (w[0] & ...)`)
#      must be on the allowlist with a measured single-writer reason;
#   2. an InterlockedAnd that clears a sub-word lane must be followed, within a
#      short window, by an InterlockedOr -- a clear with no set is a byte
#      permanently stuck at zero, which is the failure mode of "the value is
#      already zero so the OR was dropped";
#   3. a pure FUNCTION that folds a byte into a value it returns (`return (p &
#      ~(0xffu << i*8)) | ...`) is not a store at all and is not considered.
#
# The allowlist may shrink and must not grow: a new sub-word RMW is either an
# atomic pair or a line in subword_store_known.txt saying who the single writer
# is.
#
# ---------------------------------------------------------------------------
# WHAT IT FOUND WHEN IT WAS WRITTEN
# ---------------------------------------------------------------------------
# Eleven sub-word sites in the corpus.  Seven are already the atomic pair
# (constraintPrepPGS, npArtiJointPrep, pxgSlangArticulationBlocks x2,
# triTriOverlapBda).  Two are pure value-folding helpers (npCvxCollide::s3,
# pxgSlangEpa::s3) and are not stores.  Two are plain read-modify-writes, and
# both already carry a single-writer argument in their own source -- which is why
# they are the allowlist's only entries rather than two new findings.

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHADER_DIR = os.path.normpath(os.path.join(HERE, "..", "shaders"))
KNOWN = os.path.join(HERE, "subword_store_known.txt")

# `dest = (dest & ~(MASK << ...)) | ...` where dest ends in an index.  The
# destination is captured so the two halves can be compared: a line that reads
# one place and writes another is not this pattern.
RMW = re.compile(
    r"""(?P<dest>[A-Za-z_][A-Za-z0-9_.\[\]\ ]*\])\s*=\s*\(\s*(?P<src>[A-Za-z_][A-Za-z0-9_.\[\]\ ]*\])\s*&\s*~\s*\(\s*0[xX](?:FF|ff|FFFF|ffff)u?\s*<<""",
    re.X,
)
CLEAR = re.compile(r"InterlockedAnd\s*\(\s*(?P<dest>[^,]+),\s*~\s*\(\s*0[xX](?:FF|ff|FFFF|ffff)u?\s*<<")
SET = re.compile(r"InterlockedOr\s*\(")

# How far after a clearing InterlockedAnd the matching InterlockedOr may sit.
# Six lines covers `uint prev;` + a comment + an `if` + the call; the two sites
# in the corpus use two and three.
SET_WINDOW = 6


# THE KEY IS file::function, NOT file:line, AND THAT WAS MEASURED.  The first
# draft keyed on the line number, reasoning that a site which moves should be
# re-read.  Correcting a COMMENT ten lines above femClothStepBda's storeByte
# immediately broke the entry -- and an entry that breaks on edits which cannot
# possibly change who writes the word is an entry that gets rubber-stamped back,
# which is worse than no ratchet.  The enclosing function is stable under that
# and still moves when the store is relocated into different code.
FUNC = re.compile(r"^\s*(?:public\s+|static\s+|\[[^\]]*\]\s*)*(?:void|uint|int|float\d?|bool|[A-Z][A-Za-z0-9_]*)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")


def enclosing_function(lines, idx):
    """The nearest function definition at or above `idx`.  Slang has no nested
    functions, so walking back to the first match is exact."""
    for j in range(idx, -1, -1):
        stripped = lines[j].strip()
        if stripped.startswith("//"):
            continue
        m = FUNC.match(lines[j])
        if m:
            return m.group("name")
    return "<file scope>"


def load_known():
    """file::function -> reason."""
    known = {}
    if not os.path.exists(KNOWN):
        return known
    for raw in open(KNOWN):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        key, _, reason = line.partition(" ")
        known[key.strip()] = (reason.strip() or raw.split("#", 1)[-1].strip())
    return known


def scan(path):
    """Returns (rmw_sites, dangling_clears) for one source file."""
    rmw, dangling = [], []
    lines = open(path, errors="replace").read().split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        m = RMW.search(line)
        if m and m.group("dest").strip() == m.group("src").strip():
            rmw.append((enclosing_function(lines, i), i + 1, stripped))
        if CLEAR.search(line):
            window = lines[i + 1: i + 1 + SET_WINDOW]
            if not any(SET.search(w) for w in window):
                dangling.append((enclosing_function(lines, i), i + 1, stripped))
    return rmw, dangling


def main():
    known = load_known()
    seen, fresh, dangling_all = set(), [], []
    scanned = 0

    for name in sorted(os.listdir(SHADER_DIR)):
        if not name.endswith(".slang"):
            continue
        scanned += 1
        rmw, dangling = scan(os.path.join(SHADER_DIR, name))
        for func, ln, text in rmw:
            key = "%s::%s" % (name, func)
            seen.add(key)
            if key not in known:
                fresh.append((key, ln, text))
        for func, ln, text in dangling:
            dangling_all.append(("%s::%s" % (name, func), ln, text))

    rc = 0
    if fresh:
        rc = 1
        print("A SUB-WORD STORE IS A PLAIN READ-MODIFY-WRITE.  Four PxU8 lanes share a")
        print("32-bit word, so unless ONE invocation in the dispatch owns that whole")
        print("word, each writer reads it before the others' writes land and the losing")
        print("bytes are silently reverted.  Use the atomic pair")
        print()
        print("    InterlockedAnd(w[0], ~(0xFFu << shift), prev);")
        print("    if (value) InterlockedOr(w[0], value << shift, prev);")
        print()
        print("or add the site to tools/subword_store_known.txt with the measured")
        print("reason why exactly one invocation writes that word.  A fixture cannot")
        print("settle this: the lost update only happens when two groups are inside the")
        print("word at the same instant, and it survived P9.83a's sweep for that reason.")
        print()
        for key, ln, text in fresh:
            print("  %-46s :%d  %s" % (key, ln, text))

    if dangling_all:
        rc = 1
        print()
        print("AN InterlockedAnd CLEARS A SUB-WORD LANE AND NOTHING SETS IT.  The clear")
        print("is what makes an un-zeroed allocation safe; without the matching")
        print("InterlockedOr the byte is stuck at zero forever, which is exactly what a")
        print("dropped 'the value is zero anyway' looks like.")
        print()
        for key, ln, text in dangling_all:
            print("  %-46s :%d  %s" % (key, ln, text))

    stale = sorted(set(known) - seen)
    if stale:
        rc = 1
        print()
        print("%d allowlist entry(ies) no longer match a sub-word read-modify-write." % len(stale))
        print("The function was RENAMED, the store moved out of it, or the site became")
        print("atomic.  Re-read it and update the file -- a ratchet that keeps entries")
        print("for sites it can no longer see is not ratcheting.")
        print()
        for key in stale:
            print("  %-46s %s" % (key, known[key]))

    if rc == 0:
        print("subwordStore: %d source(s) scanned, %d plain read-modify-write(s) on the "
              "allowlist, every other sub-word store atomic" % (scanned, len(seen)))
    return rc


if __name__ == "__main__":
    sys.exit(main())
