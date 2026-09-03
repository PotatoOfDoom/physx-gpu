// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

// npRobustProbe.slang -- P9.50.  What happens to an OUT-OF-RANGE read through a
// DESCRIPTOR-BOUND buffer on this platform?
//
// WHY THIS EXISTS
//
// check_speculated_loads.py only tracks `Ptr<>` values, i.e. buffer-device
// addresses.  Every read through an `RWStructuredBuffer` is outside its scope,
// and that scope limit is not written down anywhere in the checker.
//
// Whether the limit is sound rests on one platform fact, and the log says two
// incompatible things about it:
//
//   * twice, that `robustBufferAccess` is NOT enabled -- which re-reading
//     PxgVulkanComputeDevice.cpp confirms: `enabled2.features` is zeroed and only
//     shaderInt64 and shaderFloat64 are ever set;
//   * once, in a fan-out audit, that a descriptor-bound SSBO is "robust
//     bounds-checked" -- with, as its evidence, a spirv-dis run showing no
//     PhysicalStorageBufferAddresses capability.  That evidence establishes "not
//     BDA".  It does not establish "bounds-checked", and 55 findings were
//     refuted partly on the strength of it.
//
// P9.49 is what makes this worth an instrument rather than an argument: there,
// a whole class was dismissed by a distance claim whose premise nobody had
// tested, and it had been faulting for four tranches.  So this measures the
// premise instead of asserting it.
//
// The kernel reads gSrc at an index the push constant supplies and publishes
// what came back, so the caller can walk the index outward from in-range to far
// past the allocation and see which of the three outcomes happens:
//   returns 0        -> the access is bounds-checked after all
//   returns garbage  -> unchecked, but the address happened to be mapped
//   the process dies -> unchecked, and reachable
//
// ONE INVOCATION ONLY, and the index is push-constant UNIFORM.  A per-lane index
// would be lowered to a masked gather, which is a different shape with different
// behaviour -- and the shape this port keeps faulting on is the uniform one.

struct RobustProbeParams
{
    uint index;      // element index to read -- may be far past the end
    uint pad0, pad1, pad2;
}

[[vk::push_constant]]
ConstantBuffer<RobustProbeParams> gParams;

RWStructuredBuffer<uint> gSrc;    // binding 0
RWStructuredBuffer<uint> gOut;    // binding 1

[shader("compute")]
[numthreads(1, 1, 1)]
void robustProbeRead(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x != 0u)
        return;
    gOut[0] = gSrc[gParams.index];
    gOut[1] = 0xD0DEu;   // a liveness marker: the kernel reached its last line
}
