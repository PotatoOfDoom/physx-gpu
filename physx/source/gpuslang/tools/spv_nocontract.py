# Post-processes a SPIR-V module, decorating every float arithmetic
# instruction result with NoContraction.
#
# Rationale: slangc (as shipped) emits no NoContraction decorations and its
# -fp-mode precise flag does not change the SPIR-V, while the Vulkan spec
# permits implementations to CONTRACT undecorated floating-point expressions.
# llvmpipe (LLVM 20) was observed to evaluate npCvxCoreEpa's makePlane
# triple-cross sum as a fused fma chain (1-ulp differences vs strict IEEE),
# breaking the suite's bitwise CPU-vs-GPU comparisons.  Decorating the ops
# forbids contraction and restores strict IEEE evaluation; on expressions the
# driver never contracted anyway this is a no-op.
#
# Usage: python3 spv_nocontract.py <file.spv>   (patches in place)

import struct
import sys

OP_DECORATE = 71
DECOR_NO_CONTRACTION = 42

# Arithmetic opcodes NoContraction may decorate (float forms)
FLOAT_ARITH = {
    129,  # OpFAdd
    131,  # OpFSub
    133,  # OpFMul
    136,  # OpFDiv
    138,  # OpFRem
    141,  # OpFMod
    142,  # OpVectorTimesScalar
    143,  # OpMatrixTimesScalar
    144,  # OpVectorTimesMatrix
    145,  # OpMatrixTimesVector
    146,  # OpMatrixTimesMatrix
    147,  # OpOuterProduct
    148,  # OpDot
}


def main(path):
    data = open(path, "rb").read()
    words = list(struct.unpack("<%dI" % (len(data) // 4), data))
    assert words[0] == 0x07230203, "not a SPIR-V module"

    ids = []
    first_decorate = None
    already = set()
    i = 5
    while i < len(words):
        wc = words[i] >> 16
        op = words[i] & 0xFFFF
        assert wc > 0
        if op == OP_DECORATE:
            if first_decorate is None:
                first_decorate = i
            if wc == 3 and words[i + 2] == DECOR_NO_CONTRACTION:
                already.add(words[i + 1])
        elif op in FLOAT_ARITH:
            # layout: result-type, result-id, operands...
            ids.append(words[i + 2])
        i += wc
    assert first_decorate is not None, "no annotation section found"

    new = []
    for rid in ids:
        if rid not in already:
            new.extend([(3 << 16) | OP_DECORATE, rid, DECOR_NO_CONTRACTION])

    out = words[:first_decorate] + new + words[first_decorate:]
    open(path, "wb").write(struct.pack("<%dI" % len(out), *out))
    print("spv_nocontract: %s: decorated %d ops" % (path, len(ids)))


if __name__ == "__main__":
    main(sys.argv[1])
