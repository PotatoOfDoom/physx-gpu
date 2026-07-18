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

// npSpikeSubgroup.slang — RUNTIME_PLAN.md P0 spike.
//
// Exercises the wave/subgroup intrinsics the warp-32 primitive library will
// be built on (ballot / arithmetic / shuffle-adjacent prefix ops) and reports
// per-thread invariants that hold at ANY subgroup size:
//   WavePrefixSum(1)     == WaveGetLaneIndex()          (exclusive scan)
//   popcount(ballot(1))  == WaveGetLaneCount()          (fully active groups)
//   WaveActiveSum(1)     == WaveGetLaneCount()
// The test dispatches exact multiples of the workgroup size so every lane is
// active (divergence-free), matching the REQUIRE_FULL_SUBGROUPS model.

struct SgOut
{
    uint4 v;   // x = laneIdx | (laneCount << 16), y = exclusive prefix of 1,
               // z = popcount of ballot(true), w = active sum of 1
}

struct SgParams { uint numThreads; }

[[vk::push_constant]]
ConstantBuffer<SgParams> gParams;

RWStructuredBuffer<SgOut> gOut;  // binding 0

[shader("compute")]
[numthreads(64, 1, 1)]
void spikeSubgroup(uint3 dtid : SV_DispatchThreadID)
{
    const uint laneIdx = WaveGetLaneIndex();
    const uint laneCount = WaveGetLaneCount();
    const uint prefix = WavePrefixSum(1u);
    const uint4 ballot = WaveActiveBallot(true);
    const uint ballotCount = countbits(ballot.x) + countbits(ballot.y) + countbits(ballot.z) + countbits(ballot.w);
    const uint sum = WaveActiveSum(1u);

    if (dtid.x < gParams.numThreads)
    {
        SgOut o;
        o.v = uint4(laneIdx | (laneCount << 16), prefix, ballotCount, sum);
        gOut[dtid.x] = o;
    }
}
