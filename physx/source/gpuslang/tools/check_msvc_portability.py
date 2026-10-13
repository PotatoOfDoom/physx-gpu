#!/usr/bin/env python3
# check_msvc_portability.py -- guards the Windows build of the Vulkan PhysXGpu
# from Linux.
#
# The Slang/Vulkan PhysXGpu (source/compiler/cmakegpuvulkan, PX_GPU_BACKEND=VULKAN)
# is built by the official vc17win64-vulkan / vc18win64-vulkan presets, but this
# port is developed on Linux and MSVC has never compiled the bridge, the Vulkan
# device or the PXG_NO_CUDA_TOOLKIT branches of the gpu* host code.  A Linux
# compiler accepts several things MSVC rejects, and neither gcc nor clang (not
# even clang in MSVC-compatibility mode, which accepts `static __inline__` and a
# non-literal __declspec(align())) reports them.  This checker reads the exact
# source set the module is built from (cmake/PxgVulkanGpuSources.cmake passes it
# on the command line) plus the gpuslang headers, and fails on:
#
#   RTTI         dynamic_cast / typeid: PhysX compiles with /GR- (and -fno-rtti).
#   GNU          __attribute__, __builtin_* (but __builtin_align__, which
#                nocuda/host_defines.h defines), __typeof__, __extension__,
#                __int128, #pragma GCC -- outside a GNU-only preprocessor branch.
#   GNU-KEYWORD  __inline__ / __restrict__ while nocuda/host_defines.h does not map
#                them for MSVC (cutil_math.h's `static __inline__`).
#   POSIX        <dlfcn.h>, <unistd.h>, <pthread.h>, <sys/...>, and dlopen / dladdr /
#                setenv / posix_memalign / mmap calls -- outside a non-Windows branch.
#   WINDOWS-H    a raw #include <windows.h>: PhysX includes it only through
#                foundation/windows/PxWindowsInclude.h, which #errors otherwise.
#   ALIGN        __declspec(align(<non-literal>)): MSVC takes integer literals only.
#   WIN-HIDDEN   Win32 APIs that foundation/windows/PxWindowsInclude.h -- the only
#                allowed way to get <windows.h> -- hides with its NO* macros:
#                NONLS (MultiByteToWideChar, WideCharToMultiByte, CP_UTF8,
#                CompareString*, GetLocaleInfo*), NOUSER/NOMB (MessageBox*), NOGDI.
#
# Scanned: the given sources, the gpuslang include/bridge/nocuda headers, and the
# gpu* modules' include/ headers (their PXG_NO_CUDA_TOOLKIT branches are new to MSVC).
#
# A line is in a GNU-only / non-Windows branch when an enclosing #if names
# __GNUC__ / __clang__ / PX_GCC_FAMILY / PX_CLANG / PX_LINUX / PX_UNIX_FAMILY /
# __linux__ (or is the #else of, or negates, _MSC_VER / _WIN32 / PX_WINDOWS_FAMILY /
# PX_WINDOWS / PX_VC).  This is a guard, not a compiler: what it cannot see is left
# to a real MSVC build (README_WINDOWS.md).
#
#   check_msvc_portability.py <source.cpp> ...    check those files + gpuslang headers
#   check_msvc_portability.py --self-test         prove every rule can fail

import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GPUSLANG = os.path.dirname(HERE)
SOURCE_ROOT = os.path.dirname(GPUSLANG)
HEADER_DIRS = [os.path.join(GPUSLANG, d) for d in ("include", "bridge", "nocuda")] + \
    [os.path.join(SOURCE_ROOT, m, "include") for m in ("gpucommon", "gpubroadphase", "gpunarrowphase",
                                                          "gpusimulationcontroller", "gpusolver", "gpuarticulation")]

GNU_TOKENS = ("__GNUC__", "__clang__", "PX_GCC_FAMILY", "PX_CLANG", "PX_LINUX",
              "PX_UNIX_FAMILY", "__linux__", "PX_APPLE_FAMILY")
WIN_TOKENS = ("_MSC_VER", "_WIN32", "_WIN64", "PX_WINDOWS_FAMILY", "PX_WINDOWS", "PX_VC")


def strip_comments(text):
    """Blank out comments and string literals, keeping line structure."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("\n" * text.count("\n", i, j))
            i = j
        elif c == '"' or c == "'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append(c + c)
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def mentions(cond, tokens):
    return any(re.search(r"\b%s\b" % re.escape(t), cond) for t in tokens)


def branch_kind(directive, cond, is_else):
    """'gnu' when the branch is compiled only by non-MSVC compilers, else None."""
    negated = directive == "ifndef" or cond.lstrip().startswith("!")
    gnu = mentions(cond, GNU_TOKENS)
    win = mentions(cond, WIN_TOKENS)
    if directive in ("ifdef", "ifndef") and not gnu and not win:
        return None
    if gnu and not win:
        taken_gnu = not negated
    elif win and not gnu:
        taken_gnu = negated
    else:
        return None
    if is_else:
        taken_gnu = not taken_gnu
    return "gnu" if taken_gnu else None


RULES = [
    ("RTTI", re.compile(r"\b(dynamic_cast|typeid)\b"), False),
    ("GNU", re.compile(r"\b__attribute__\b|\b__builtin_(?!align__)\w+|\b__typeof__\b|\b__extension__\b|\b__int128\b|^\s*#\s*pragma\s+GCC\b"), True),
    ("POSIX", re.compile(r"#\s*include\s*<(dlfcn\.h|unistd\.h|pthread\.h|sys/[^>]*)>|\b(dlopen|dladdr|dlsym|setenv|unsetenv|posix_memalign|mmap|munmap)\s*\("), True),
    ("WINDOWS-H", re.compile(r"#\s*include\s*[<\"]windows\.h[>\"]", re.IGNORECASE), False),
    ("ALIGN", re.compile(r"__declspec\s*\(\s*align\s*\(\s*(?!\d+\s*\)\s*\))"), False),
    ("WIN-HIDDEN", re.compile(r"\b(MultiByteToWideChar|WideCharToMultiByte|CP_UTF8|CompareString\w*|GetLocaleInfo\w*|MessageBox\w*|CreateCompatibleDC|SelectObject|GetDeviceCaps)\b"), True),
]
GNU_KEYWORD = re.compile(r"\b(__inline__|__restrict__)\b")


def msvc_maps_gnu_keywords(host_defines):
    """The GNU keywords host_defines.h #defines inside an `#if defined(_MSC_VER)` block."""
    try:
        lines = strip_comments(open(host_defines).read()).split("\n")
    except OSError:
        return set()
    mapped, depth, msvc_depth = set(), 0, None
    for line in lines:
        d = re.match(r"\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)", line)
        if d:
            kind = d.group(1)
            if kind in ("if", "ifdef", "ifndef"):
                depth += 1
                if msvc_depth is None and kind == "if" and re.search(r"defined\s*\(\s*_MSC_VER\s*\)", d.group(2)) \
                        and not d.group(2).lstrip().startswith("!"):
                    msvc_depth = depth
            elif kind in ("elif", "else") and msvc_depth == depth:
                msvc_depth = None
            elif kind == "endif":
                if msvc_depth == depth:
                    msvc_depth = None
                depth -= 1
            continue
        if msvc_depth is not None:
            m = re.match(r"\s*#\s*define\s+(__inline__|__restrict__)\b", line)
            if m:
                mapped.add(m.group(1))
    return mapped


def check_file(path, mapped, findings):
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError as e:
        findings.append((path, 0, "IO", str(e)))
        return
    lines = strip_comments(text).split("\n")
    stack = []  # (directive, cond, is_else)
    for lineno, line in enumerate(lines, 1):
        d = re.match(r"\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)", line)
        if d:
            kind, cond = d.group(1), d.group(2).strip()
            if kind in ("if", "ifdef", "ifndef"):
                stack.append((kind, cond, False))
            elif kind in ("elif", "else") and stack:
                prev_kind, prev_cond, _ = stack[-1]
                # #elif X: treat as a fresh condition; #else: flips the previous one.
                stack[-1] = ("if", cond, False) if kind == "elif" else (prev_kind, prev_cond, True)
            elif kind == "endif" and stack:
                stack.pop()
            continue
        gnu_only = any(branch_kind(k, c, e) == "gnu" for k, c, e in stack)
        for name, rx, gnu_ok in RULES:
            if gnu_ok and gnu_only:
                continue
            if rx.search(line):
                findings.append((path, lineno, name, line.strip()))
        if not gnu_only:
            for m in GNU_KEYWORD.finditer(line):
                if m.group(1) not in mapped:
                    findings.append((path, lineno, "GNU-KEYWORD", line.strip()))


def run(paths, host_defines):
    mapped = msvc_maps_gnu_keywords(host_defines)
    findings = []
    for p in paths:
        check_file(p, mapped, findings)
    return findings


def collect(sources):
    files = list(sources)
    for d in HEADER_DIRS:
        for f in sorted(os.listdir(d)):
            if f.endswith((".h", ".hpp", ".cpp")):
                files.append(os.path.join(d, f))
    # gpuslang/include/*.cpp etc. may repeat a source: keep each file once.
    seen, out = set(), []
    for f in files:
        k = os.path.realpath(f)
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


SELF_TEST = [
    ("RTTI", "struct A{virtual ~A();};\nA* f(A* a){ return dynamic_cast<A*>(a); }\n"),
    ("GNU", "int x __attribute__((unused));\n"),
    ("GNU", "int f(int v){ return __builtin_popcount(v); }\n"),
    ("POSIX", "#include <dlfcn.h>\n"),
    ("POSIX", "void f(){ setenv(\"A\", \"B\", 1); }\n"),
    ("WINDOWS-H", "#include <windows.h>\n"),
    ("ALIGN", "struct __declspec(align(2 * sizeof(long))) S { long x; };\n"),
    ("WIN-HIDDEN", "#include \"foundation/windows/PxWindowsInclude.h\"\nint f(){ return MultiByteToWideChar(CP_UTF8, 0, 0, 0, 0, 0); }\n"),
    ("GNU-KEYWORD", "static __inline__ float f(float v){ return v; }\n"),
]
SELF_TEST_CLEAN = [
    "#if defined(__GNUC__)\nint x __attribute__((unused));\n#endif\n",
    "#if PX_WINDOWS_FAMILY\n#include \"foundation/windows/PxWindowsInclude.h\"\n#else\n#include <dlfcn.h>\n#endif\n",
    "#if !PX_WINDOWS_FAMILY\nvoid f(){ dladdr(0, 0); }\n#endif\n",
    "struct __declspec(align(16)) S { float x; };\n",
    "// dynamic_cast in a comment\nconst char* s = \"typeid\";\n",
    "#define __builtin_align__(n) __align__(n)\nstruct __builtin_align__(16) T { float x; };\n",
]


def self_test():
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        empty_defines = os.path.join(tmp, "host_defines.h")
        open(empty_defines, "w").write("#define __device__\n")
        for i, (rule, src) in enumerate(SELF_TEST):
            p = os.path.join(tmp, "bad%d.cpp" % i)
            open(p, "w").write(src)
            got = {f[2] for f in run([p], empty_defines)}
            if rule not in got:
                print("self-test: rule %s did not fire on:\n%s" % (rule, src))
                ok = False
        for i, src in enumerate(SELF_TEST_CLEAN):
            p = os.path.join(tmp, "good%d.cpp" % i)
            open(p, "w").write(src)
            got = run([p], empty_defines)
            if got:
                print("self-test: false positive %s on:\n%s" % (got, src))
                ok = False
        # The real host_defines.h must map the GNU keywords for MSVC.
        real = os.path.join(GPUSLANG, "nocuda", "host_defines.h")
        p = os.path.join(tmp, "kw.cpp")
        open(p, "w").write(SELF_TEST[-1][1])
        if run([p], real):
            print("self-test: nocuda/host_defines.h no longer maps __inline__/__restrict__ for MSVC")
            ok = False
    return ok


def main(argv):
    if argv[1:] == ["--self-test"]:
        if self_test():
            print("check_msvc_portability self-test: OK")
            return 0
        return 1
    if not self_test():
        print("check_msvc_portability: self-test FAILED, refusing to report")
        return 1
    sources = argv[1:]
    if not sources:
        print(__doc__ if __doc__ else "usage: check_msvc_portability.py <source.cpp> ...")
        return 2
    files = collect(sources)
    findings = run(files, os.path.join(GPUSLANG, "nocuda", "host_defines.h"))
    for path, lineno, rule, text in findings:
        print("%s:%d: %s: %s" % (os.path.relpath(path, os.path.dirname(GPUSLANG)), lineno, rule, text))
    print("check_msvc_portability: %d files, %d findings" % (len(files), len(findings)))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
