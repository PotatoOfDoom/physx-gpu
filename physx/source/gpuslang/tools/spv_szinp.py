# Post-processes a SPIR-V module (1.4+, where float_controls is core): adds
# OpCapability SignedZeroInfNanPreserve and, for every OpEntryPoint,
# OpExecutionMode <entry> SignedZeroInfNanPreserve 32.
#
# Rationale (P9.86): llvmpipe drops the sign of zero in arithmetic with a
# CONSTANT zero operand (0 - x, x + 0, 0 * x), NoContraction or not -- the
# classic a + 0 -> a / 0 - a -> -a / a * 0 -> 0 simplifications, which are
# only legal without a signed-zero guarantee.  CUDA keeps IEEE semantics.
# The execution mode is the SPIR-V way to demand that guarantee; this tool
# lets a fixture measure whether the driver honours it.
#
# P9.86 measured that the mode DOES stop llvmpipe folding x * 0 and x +/- a
# proven zero (the P9.43b class), but NOT OpFSub(constant 0, x) -> -x, and of
# course not slangc's own source-level folds of x + 0 / 0 + x / x - 0.  So every
# engine shader is stamped (tests/CMakeLists.txt, pxg_compile_shader), and the
# remaining spellings are handled in the source.  Float64 is stamped too when
# the module declares the Float64 capability.
#
# Usage: python3 spv_szinp.py <in.spv> [<out.spv>]   (in place when out omitted)

import struct
import sys

OP_CAPABILITY = 17
OP_ENTRY_POINT = 15
OP_EXECUTION_MODE = 16
CAP_SZINP = 4466
CAP_FLOAT64 = 10
MODE_SZINP = 4461


def main(src, dst):
    data = open(src, "rb").read()
    words = list(struct.unpack("<%dI" % (len(data) // 4), data))
    assert words[0] == 0x07230203, "not a SPIR-V module"
    assert words[1] >= 0x00010400, "float_controls needs SPIR-V 1.4+"

    last_cap = None
    last_ep_or_mode = None
    entries = []
    have_cap = False
    have_f64 = False
    have_modes = set()
    i = 5
    while i < len(words):
        wc = words[i] >> 16
        op = words[i] & 0xFFFF
        assert wc > 0
        if op == OP_CAPABILITY:
            last_cap = i + wc
            if words[i + 1] == CAP_SZINP:
                have_cap = True
            if words[i + 1] == CAP_FLOAT64:
                have_f64 = True
        elif op == OP_ENTRY_POINT:
            entries.append(words[i + 2])
            last_ep_or_mode = i + wc
        elif op == OP_EXECUTION_MODE:
            last_ep_or_mode = i + wc
            if wc == 4 and words[i + 2] == MODE_SZINP:
                have_modes.add((words[i + 1], words[i + 3]))
        i += wc
    assert last_cap is not None and last_ep_or_mode is not None and entries

    modes = []
    widths = [32, 64] if have_f64 else [32]
    for e in entries:
        for width in widths:
            if (e, width) not in have_modes:
                modes += [(4 << 16) | OP_EXECUTION_MODE, e, MODE_SZINP, width]
    out = words[:last_ep_or_mode] + modes + words[last_ep_or_mode:]
    if not have_cap:
        out = out[:last_cap] + [(2 << 16) | OP_CAPABILITY, CAP_SZINP] + out[last_cap:]
    open(dst, "wb").write(struct.pack("<%dI" % len(out), *out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
