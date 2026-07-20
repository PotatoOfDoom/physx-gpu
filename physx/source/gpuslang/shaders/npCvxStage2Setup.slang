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

// npCvxStage2Setup.slang
//
// P5 Step 4 (BDA, engine launch path): convexConvexNphase_stage2Kernel
// (cudaGJKEPA.cu L2645-2673 driving stage2Kernel_core L2390-2641) on the
// exact engine ABI —
//   (PxU32 numTests, PxReal toleranceLength, const PxU32* tempKeepPairsArray,
//    const PxgContactManagerInput*, PxgPersistentContactManifold*, PxgShape*,
//    const PxsCachedTransform*, const PxReal* contactDistance)
//   argSizes {4,4,8,8,8,8,8,8} -> 56-byte blob.
// Engine grid (PxgNarrowphaseCore.cpp ~L650): blockDim (32,
// GJK_EPA_WARPS_PER_BLOCK = 1), numBlocks = ceil(numTests / 1) — one warp
// per test, globalWarpIndex = blockIdx.x*blockDim.y + threadIdx.y, with the
// early-outs `globalWarpIndex >= numTests || !tempKeepPairsArray[gw]`.
//
// The setup/teardown arithmetic (npCvxStage2Core.stage2Body) is the
// tranche-154 bound-buffer port VERBATIM (bit-verified there); the parameter
// plumbing and data access are converted: shapes/manifold/transforms/
// contactDistance through Ptr<> from the raw engine pointers, and the
// cooked-hull blob header (uint4 at hull + 16) and extents (float4 at
// hull + 32) read through shape.hullOrMeshPtr.  The warp is serialized on
// lane 0 (llvmpipe has no 32-wide subgroup; the lane cooperation here is only
// the 16-lane moreData read + one shfl from lane 12, both straight-line when
// serialized).
//
// collide() ITSELF (the GJK/EPA/contact-gen core) IS NOT YET ASSEMBLED in
// Slang — its verified pieces live in the npGjk*/npEpa*/npManifold*
// tranches.  This kernel is the .cu with the single `collide(...)` call
// elided: `bool bIntersection = false` stays false and numManifoldContacts
// keeps the stage-1 refreshed count, so the epilogue store writes
//   mNbContacts = refreshedNb | 0x80000000
// — the manifold-invalidated / no-touch semantic that the downstream
// finishContactsKernel consumes safely (it clears the bit and treats the
// count as 0).  The quatB tail words 9..11 are stored back with the values
// read (the .cu's no-op rewrite).  When the assembled collide() lands it
// replaces that one call site in stage2Body.
//
// The setup arithmetic that feeds collide() (post-flip shapes, synthetic
// vertices, margins, blob-header decode, the shape1-gated minScale quirk,
// contactDistance + expandedMargin, replaceBreakingThreshold, counts unpack)
// is not observable in this collide-elided memory effect, so a sibling
// harness entry `convexConvexNphase_stage2KernelDebug`
// (npCvxStage2SetupDbg.slang, sharing stage2Body) dumps every scratch value
// for the bit-exact test to verify against the CPU reference.

import npCvxStage2Core;

// Engine ABI blob: {4,4,8,8,8,8,8,8} at natural alignment = 56 bytes.
struct Stage2Params
{
    uint numTests;               // @0
    float toleranceLength;       // @4
    uint64_t tempKeepPairsArray; // @8   const PxU32*
    uint64_t cmInputs;           // @16  const PxgContactManagerInput*
    uint64_t cmManifolds;        // @24  PxgPersistentContactManifold*
    uint64_t shapes;             // @32  PxgShape*
    uint64_t transformCache;     // @40  const PxsCachedTransform*
    uint64_t contactDistance;    // @48  const PxReal*
}

[[vk::push_constant]]
ConstantBuffer<Stage2Params> gS2;

// Production engine-ABI entry (56-byte blob).
[shader("compute")]
[numthreads(32, 1, 1)]
void convexConvexNphase_stage2Kernel(uint3 groupID : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    // globalWarpIndex = blockIdx.x * blockDim.y + threadIdx.y (blockDim.y = 1)
    const uint globalWarpIndex = groupID.x * 1u + tid.y;
    if (tid.x != 0u)
        return; // serialized warp
    stage2Body(globalWarpIndex, gS2.numTests, gS2.toleranceLength, gS2.tempKeepPairsArray,
               gS2.cmInputs, gS2.cmManifolds, gS2.shapes, gS2.transformCache,
               gS2.contactDistance, 0);
}
