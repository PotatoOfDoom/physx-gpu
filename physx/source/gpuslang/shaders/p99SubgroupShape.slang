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


// p99SubgroupShape.slang -- P9.99 test-only probe (gpuslang.pipelineRefusal), never in the
// engine manifest.  One body at seven workgroup shapes, for createPipelineEx's
// required-subgroup-size rules on llvmpipe (window [8, 8], maxComputeWorkgroupSubgroups 32):
//   sgShape64     (64, 1, 1)   legal at 8                                  -> built, full subgroups
//   sgShape8x32   (8, 32, 1)   256 = 8 x 32 invocations, the 02756 boundary -> built
//   sgShape8x33   (8, 33, 1)   264 > 256                                   -> refused, VUID-02756
//   sgShape8x1x33 (8, 1, 33)   264 > 256 through Z                         -> refused, VUID-02756
//   sgShape512    (512, 1, 1)  512 > 256 (fits every P9.98 limit)          -> refused, VUID-02756
//   sgShape4      (4, 1, 1)    X not a multiple of 8                       -> refused, VUID-02757
//   sgShape12     (12, 1, 1)   X >= 8 but not a multiple of 8              -> refused, VUID-02757
// Measured on HEAD 2a97aa5 (P9.99 plan): every refused shape was BUILT, the layer reported
// 02756 / 02757, and sgShape4 ran 8-lane subgroups with 4 active lanes under
// REQUIRE_FULL_SUBGROUPS.  Vulkan-only (wave ops, like npSpikeSubgroup): absent from
// PXG_HLSL_ENTRIES and PXG_NOCONTRACT_SHADERS.

struct SgOut
{
    uint4 v;   // x = laneIdx | (laneCount << 16), y = WavePrefixSum(1), z = popcount(ballot(true)),
               // w = WaveActiveSum(1); in a full subgroup y == laneIdx and z == w == laneCount
}

struct SgParams { uint numThreads; }

[[vk::push_constant]]
ConstantBuffer<SgParams> gParams;

RWStructuredBuffer<SgOut> gOut;   // binding 0

void sgBody(uint linear)
{
    const uint laneIdx = WaveGetLaneIndex();
    const uint laneCount = WaveGetLaneCount();
    const uint prefix = WavePrefixSum(1u);
    const uint4 ballot = WaveActiveBallot(true);
    const uint bc = countbits(ballot.x) + countbits(ballot.y) + countbits(ballot.z) + countbits(ballot.w);
    const uint sum = WaveActiveSum(1u);
    if (linear < gParams.numThreads)
    {
        SgOut o;
        o.v = uint4(laneIdx | (laneCount << 16), prefix, bc, sum);
        gOut[linear] = o;
    }
}

[shader("compute")] [numthreads(64, 1, 1)]
void sgShape64(uint3 g : SV_GroupID, uint gi : SV_GroupIndex) { sgBody(g.x * 64u + gi); }
[shader("compute")] [numthreads(8, 32, 1)]
void sgShape8x32(uint3 g : SV_GroupID, uint gi : SV_GroupIndex) { sgBody(g.x * 256u + gi); }
[shader("compute")] [numthreads(8, 33, 1)]
void sgShape8x33(uint3 g : SV_GroupID, uint gi : SV_GroupIndex) { sgBody(g.x * 264u + gi); }
[shader("compute")] [numthreads(8, 1, 33)]
void sgShape8x1x33(uint3 g : SV_GroupID, uint gi : SV_GroupIndex) { sgBody(g.x * 264u + gi); }
[shader("compute")] [numthreads(512, 1, 1)]
void sgShape512(uint3 g : SV_GroupID, uint gi : SV_GroupIndex) { sgBody(g.x * 512u + gi); }
[shader("compute")] [numthreads(4, 1, 1)]
void sgShape4(uint3 g : SV_GroupID, uint gi : SV_GroupIndex) { sgBody(g.x * 4u + gi); }
[shader("compute")] [numthreads(12, 1, 1)]
void sgShape12(uint3 g : SV_GroupID, uint gi : SV_GroupIndex) { sgBody(g.x * 12u + gi); }
