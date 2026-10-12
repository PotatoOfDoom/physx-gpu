#!/usr/bin/env python3
# vvl_triage.py -- P9.98.  Reads the stdout of runs made under the Khronos validation
# layer (tools/run_validation.sh), applies tools/validation_waivers.txt, and fails on
# any message the waivers do not cover.
#
# Each message is attributed through the shader-module NAME the device attaches under
# PXG_VK_DEBUG_NAMES=1 (the entry point).  A waiver scoped to a module therefore only
# matches a message that NAMES one of that module's entry points: an unnamed message
# (debug names off) is never waived.
#
# Fails CLOSED (P9.98 check round 0): a line that carries a validation-message header
# (`(ERROR|WARN|INFO|VERBOSE|PERF / <type>): msgNum:` or `MessageID = 0x...`) but whose
# `Validation <kind>: [ <ID> ]` part does not parse is counted as an unwaived message --
# a different VVL version that reformats its messages must not read as a pass.  Messages
# of every kind (Information too) are triaged, not only errors and warnings.
#
# Check round 1 adds three kinds of line the layer does not print, triaged exactly like
# its messages (every one is a failure unless a waiver covers it):
#   PXG-refusal           a device / shim / bridge refusal: `[PxgVulkan] ...; refused`,
#                         `nothing recorded`, `; returning 0`, `; no device`, a failed init
#                         step, `[PxgShim] ... InvalidValue` or `[pxg-refused]`.  A refused
#                         pipeline or launch is a kernel that never ran -- the layer cannot
#                         see it (F4's null pipeline used to be flagged AND crash; since P9.98
#                         it is a clean refusal), so a run with one must not PASS.  Only the
#                         fixtures that provoke refusals on purpose are waived (scope suite:).
#   PXG-push-zero-filled  `[PxgVulkan] createPipeline(<entry>): push block B B wider than
#                         caller C B; zero-filled` (PXG_VK_DEBUG_NAMES).  The F2 widening
#                         makes VUID-07987 unreachable, so a shader reading past the caller's
#                         bytes would read zeros silently; each entry allowed to be widened is
#                         listed (scope entry:).
#   PXG-alloc-over-limit  a `[pxg-alloc] peak N ... (maxMemoryAllocationCount M)` line
#                         (PXG_VK_ALLOC_STATS) with N > M.  The F5 waivers also carry the
#                         condition alloc-peak:<N>: they hold only for a log whose largest
#                         reported peak is at most N -- the bound their reason states.
# A line is attributed to a SUITE by the log's name (subset_v<m>_<suite>.log) or, in a
# `ctest -V` log, by its "<n>: " prefix and the "Start <n>: gpuslang.<suite>" line.
#
# Usage: vvl_triage.py [--shader-dir DIR] [--waivers FILE] [--summary-only] LOG [LOG ...]
# Exit:  0 no unwaived message;  1 unwaived (or unparsable) messages;  2 usage / unreadable
#        input -- a log, the waivers file (or a malformed line in it), or a module a
#        waiver names that is not in the shader directory.

import argparse
import collections
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_groupshared  # noqa: E402  (per-entry / module Workgroup sizes)

MSG = re.compile(r"Validation (Error|Warning|Performance Warning|Information|Verbose)\s*:\s*"
                 r"\[\s*([^\]\s][^\]]*?)\s*\]")
# What makes a line a validation message at all, whether or not MSG can parse it.
HEADER = re.compile(r"\((?:ERROR|WARN|INFO|VERBOSE|PERF)\s*/\s*\w+\)\s*:\s*msgNum:|MessageID\s*=\s*0x[0-9A-Fa-f]+")
NAME = re.compile(r"name = ([^,]+), type = VK_OBJECT_TYPE_SHADER_MODULE")
NAME_ANY = re.compile(r"name = ([^,]+), type = (VK_OBJECT_TYPE_\w+)")
SHARED = re.compile(r"uses (\d+) bytes of shared memory, which is more than "
                    r"maxComputeSharedMemorySize \((\d+)\)")
# Check round 1: the port's own lines (see the header).
REFUSAL = re.compile(r"\[PxgVulkan\].*(?:; refused|nothing recorded|; returning 0|; no device)"
                     r"|\[PxgVulkan\] init: .* failed"
                     r"|\[PxgShim\].*InvalidValue"
                     r"|\[pxg-refused\]")
WIDENED = re.compile(r"\[PxgVulkan\] createPipeline\(([^)]*)\): push block (\d+) B wider than caller (\d+) B; "
                     r"zero-filled")
ALLOC = re.compile(r"\[pxg-alloc\] peak (\d+) live VkDeviceMemory allocation\(s\) "
                   r"\(maxMemoryAllocationCount (\d+)\)")
CTEST_START = re.compile(r"^\s*Start\s+(\d+): gpuslang\.(\w+)")
CTEST_LINE = re.compile(r"^(\d+): ")
SUBSET_LOG = re.compile(r"subset_v\d+_(\w+)\.log$")


def load_waivers(path):
    """The waiver list; ValueError on a line that is not <ID> <scope> <condition> with a
    known scope and condition (a typo must not silently waive nothing -- or everything)."""
    out = []
    with open(path) as fh:
        for n, raw in enumerate(fh, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            f = line.split()
            if len(f) != 3:
                raise ValueError("%s:%d: malformed waiver line (want <ID> <scope> <condition>): %r"
                                 % (path, n, raw.rstrip("\n")))
            wid, scope, cond = f
            module = scope.startswith("module:") and scope.endswith(".spv") and len(scope) > len("module:.spv")
            named = (scope.startswith("suite:") and len(scope) > len("suite:")) or \
                    (scope.startswith("entry:") and len(scope) > len("entry:"))
            if not module and not named and scope != "any":
                raise ValueError("%s:%d: unknown scope %r (want any, module:<file>.spv, suite:<name> or "
                                 "entry:<name>)" % (path, n, scope))
            if cond == "module-sum" and not module:
                raise ValueError("%s:%d: module-sum needs a module: scope" % (path, n))
            if cond.startswith("alloc-peak:"):
                if not cond[len("alloc-peak:"):].isdigit():
                    raise ValueError("%s:%d: alloc-peak needs a number: %r" % (path, n, cond))
            elif cond not in ("always", "module-sum") and not (cond.startswith("text:") and len(cond) > 5):
                raise ValueError("%s:%d: unknown condition %r (want always, module-sum, text:<substring> or "
                                 "alloc-peak:<N>)" % (path, n, cond))
            out.append((wid, scope, cond))
    return out


class ModuleInfo(object):
    def __init__(self, shader_dir):
        self.dir = shader_dir
        self.cache = {}

    def get(self, spv):
        if spv not in self.cache:
            path = os.path.join(self.dir, spv)
            if not os.path.isfile(path):
                raise OSError("a waiver names %s, which is not in %s" % (spv, self.dir))
            total, entries = check_groupshared.workgroup_bytes(path)
            self.cache[spv] = (total, {e[0]: e[1] for e in entries})
        return self.cache[spv]


def condition_holds(cond, text, alloc_peak=None):
    """The scope-independent conditions.  alloc_peak: the largest [pxg-alloc] peak the
    message's log reports (None: the log reports none, and alloc-peak cannot hold)."""
    if cond == "always":
        return True
    if cond.startswith("text:"):
        return cond[len("text:"):] in text
    if cond.startswith("alloc-peak:"):
        return alloc_peak is not None and alloc_peak <= int(cond[len("alloc-peak:"):])
    return False


def waived(vuid, text, waivers, mods, suite=None, entry=None, alloc_peak=None):
    """suite: the gpuslang suite the line belongs to (None in a scene log); entry: the
    entry point a PXG-* line names (layer messages are attributed through NAME)."""
    if entry is None:
        name_m = NAME.search(text)
        entry = name_m.group(1).strip() if name_m else None
    for wid, scope, cond in waivers:
        if wid != vuid:
            continue
        if scope.startswith("suite:") or scope.startswith("entry:"):
            want = suite if scope.startswith("suite:") else entry
            if want is not None and want == scope.split(":", 1)[1] and condition_holds(cond, text, alloc_peak):
                return "%s %s" % (scope, cond)
            continue
        if scope.startswith("module:"):
            spv = scope[len("module:"):]
            if entry is None:
                continue
            total, per_entry = mods.get(spv)
            if entry not in per_entry:
                continue
            if cond == "module-sum":
                m = SHARED.search(text)
                if not m:
                    continue
                reported, limit = int(m.group(1)), int(m.group(2))
                if reported != total or per_entry[entry] > limit:
                    continue
            elif not condition_holds(cond, text, alloc_peak):
                continue
            return "%s %s" % (scope, cond)
        elif scope == "any" and condition_holds(cond, text, alloc_peak):
            return "any" if cond == "always" else "any %s" % cond
    return None


def scan_log(fh):
    """(lines, largest [pxg-alloc] peak or None, the ALLOC matches) of one log."""
    lines = fh.readlines()
    peaks = [ALLOC.search(l) for l in lines]
    peaks = [m for m in peaks if m]
    return lines, (max(int(m.group(1)) for m in peaks) if peaks else None), peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shader-dir", default=os.environ.get("PXG_SHADER_DIR", ""))
    ap.add_argument("--waivers", default=os.path.join(HERE, "validation_waivers.txt"))
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args()
    if not a.shader_dir or not os.path.isdir(a.shader_dir):
        print("vvl_triage: --shader-dir / PXG_SHADER_DIR must name the .spv directory", file=sys.stderr)
        return 2
    try:
        waivers = load_waivers(a.waivers)
    except (ValueError, OSError) as e:
        print("vvl_triage: %s" % e, file=sys.stderr)
        return 2
    mods = ModuleInfo(a.shader_dir)
    kept = collections.Counter()
    dropped = collections.Counter()
    examples = {}
    unparsed = []
    for log in a.logs:
        try:
            fh = open(log, errors="replace")
        except OSError as e:
            print("vvl_triage: %s" % e, file=sys.stderr)
            return 2
        with fh:
            lines, alloc_peak, allocs = scan_log(fh)
        sm = SUBSET_LOG.search(os.path.basename(log))
        log_suite = sm.group(1) if sm else None
        ctest_suites = {}
        for m in allocs:
            if int(m.group(1)) > int(m.group(2)):
                kept[("PXG-alloc-over-limit", "?")] += 1
                examples.setdefault(("PXG-alloc-over-limit", "?"), (log, m.group(0)))
        for line in lines:
            st = CTEST_START.search(line)
            if st:
                ctest_suites[st.group(1)] = st.group(2)
            cl = CTEST_LINE.search(line)
            suite = ctest_suites.get(cl.group(1)) if cl else log_suite
            entry = None
            m = MSG.search(line)
            if m:
                vuid = m.group(2).strip()
                names = NAME_ANY.findall(line)
                who = names[0][0].strip() if names else "?"
            else:
                if HEADER.search(line):
                    unparsed.append((log, line.strip()[:600]))
                    continue
                wm = WIDENED.search(line)
                if wm:
                    vuid, entry = "PXG-push-zero-filled", wm.group(1)
                    who = entry
                elif REFUSAL.search(line):
                    vuid, who = "PXG-refusal", suite or os.path.basename(log)
                else:
                    continue
            try:
                w = waived(vuid, line, waivers, mods, suite=suite, entry=entry, alloc_peak=alloc_peak)
            except (ValueError, OSError) as e:
                print("vvl_triage: %s" % e, file=sys.stderr)
                return 2
            if w:
                dropped[(vuid, w)] += 1
            else:
                kept[(vuid, who)] += 1
                examples.setdefault((vuid, who), (log, line.strip()[:600]))
    print("vvl_triage: %d message(s) waived, %d NOT waived, %d unparsable"
          % (sum(dropped.values()), sum(kept.values()), len(unparsed)))
    for (vuid, w), n in sorted(dropped.items()):
        print("  waived  %6d  %s  [%s]" % (n, vuid, w))
    for (vuid, who), n in sorted(kept.items()):
        print("  FAIL    %6d  %s  (object %s)" % (n, vuid, who))
        if not a.summary_only:
            log, text = examples[(vuid, who)]
            print("          e.g. %s: %s" % (os.path.basename(log), text))
    if unparsed:
        print("  FAIL    %6d  validation message(s) whose ID could not be parsed" % len(unparsed))
        log, text = unparsed[0]
        print("          e.g. %s: %s" % (os.path.basename(log), text))
    return 1 if kept or unparsed else 0


if __name__ == "__main__":
    sys.exit(main())
