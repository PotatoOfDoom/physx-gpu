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

// npSlabRefCount.slang
//
// Slang port of countActiveSlabs (solverBlock.cuh L84-97): the mass-splitting
// reference-count helper used by the (rigid and articulation) solve-block
// drivers to derive ref0/ref1.  For a solver-body `index`, it sums the
// population count of the encoded per-slab reference bitmask across
// num32Slabs = ceil(numSlabs/32) words (strided by numSolverBodies through the
// global encodedReferenceCounts buffer) and returns PxMax(1u, sum).
//
// __popc -> countbits; PxMax(1u, x) -> max(1u, x) (integer, exact).  Pure
// unsigned-integer arithmetic — bitwise-identical, no float exception classes.

struct CasIn
{
    uint4 cfg;   // index, numSlabs, numSolverBodies, 0
}

struct CasParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CasParams> gParams;

RWStructuredBuffer<uint> gWords;   // binding 0: encodedReferenceCounts
RWStructuredBuffer<CasIn> gIn;     // binding 1
RWStructuredBuffer<uint4> gOut;    // binding 2: .x = active-slab count

[shader("compute")]
[numthreads(64, 1, 1)]
void casCount(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const CasIn s = gIn[tid];

    const uint index = s.cfg.x;
    const uint numSlabs = s.cfg.y;
    const uint numSolverBodies = s.cfg.z;

    const uint num32Slabs = (numSlabs + 31u) / 32u;  // >32 slabs -> multiple words
    uint referenceCount = 0u;
    for (uint i = 0u; i < num32Slabs; ++i)
    {
        const uint id = gWords[index + i * numSolverBodies];
        referenceCount += countbits(id);  // __popc
    }

    gOut[tid] = uint4(max(1u, referenceCount), 0u, 0u, 0u);  // PxMax(1u, .)
}
