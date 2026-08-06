// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

// radixWarpBda.slang
//
// Test scaffolding for pxgSlangRadixWarp — NOT an engine kernel, absent from the
// manifest.  One workgroup of 32 threads per case; the keys live in a device
// buffer whose first half is the input (and, after eight passes, the answer) and
// whose second half is the sort's temp.

import pxgSlangRadixWarp;

struct RadixWarpParams
{
    uint64_t keys;      // @0  base of the per-case key regions
    uint64_t cases;     // @8  uint4 per case: (count, numUint4, slotBase, _)
    uint     numCases;  // @16
    uint     _pad20;
}

[[vk::push_constant]]
ConstantBuffer<RadixWarpParams> gRw;

[shader("compute")]
[numthreads(32, 1, 1)]
void radixWarpLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const uint c = gid.x;

    if (c >= gRw.numCases)
        return;

    const uint4 cs = Ptr<uint4>(gRw.cases)[c];
    const uint count = cs.x;
    const uint numUint4 = cs.y;
    const uint slotBase = cs.z;

    // The temp half sits immediately after the rounded key run, exactly as
    // sortTriangleIndices places it: input + numUint4 uint4s.
    const uint64_t input = gRw.keys + uint64_t(slotBase) * 4UL;
    const uint64_t temp = input + uint64_t(numUint4) * 16UL;

    rsSortSingleWarpKeysOnly(lane, input, count, numUint4, temp);
}
