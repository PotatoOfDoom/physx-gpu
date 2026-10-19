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

// npCvxCoreWarpAcc.slang
//
// Slang port of Gu::Contact32 -- the full-warp contact-patch accumulator of
// convexCoreCollision.cu (L686-917): addPoints + copyPoints + the warp
// reducePoints + findCreatePatch.  The warp collectives are emulated with
// deterministic lane loops over explicit 32-bit masks, preserving CUDA's
// tie conventions exactly:
//   - warpScan: inclusive prefix sum over the 32 per-lane point counts;
//     copyPoints writes each lane's transformed points at prefix - num and
//     broadcasts lane 31's prefix as totalPoints.  Each lane's own normal
//     is rotated (per-lane worldNormal).
//   - the addPoints round loop: ballot(i < totalPoints) as allMask; while
//     allMask: minIndex (LOWEST lane wins ties) finds the deepest point,
//     its normal is broadcast, sameNormalMask = allMask & ballot(dot >
//     PXC_SAME_NORMAL = 0.999, strict >); reducePoints picks <= 4 best points
//     (deepest, farthest, two side extremes along cross(n, p1-p0), indices
//     OR-ed so duplicates collapse); the picks land in tmpBuffer by
//     popcount offsets; the target patch's existing points are appended;
//     a second reducePoints over the union writes the patch back; lane 0
//     swaps the deepest pick to slot 0 and rebalances the counters.
//   - findCreatePatch: maxIndex over dot(normal, patchNormal[lane]) with
//     lanes >= mNumPatches contributing dot(normal, 0) = 0; > PXC_SAME_NORMAL
//     returns that patch; below MAX_PATCHES appends; at capacity each lane
//     t scans patches t+1..15 for its closest OTHER patch, maxIndex picks
//     the closest pair, and if that pair is closer than the new normal to
//     any patch, the pair member with the LARGER slot-0 dist (shallower) is
//     reset to receive the new patch, else the new patch is dropped
//     (0xffffffff).
// All dot/compare arithmetic: ALL outputs verify bitwise.


// P9.70 MOVED THE ARITHMETIC OUT.  What is left here is the descriptor-set
// binding, the WIn/WOut record layout and the entry point; the accumulator
// itself now lives in pxgSlangCvxCoreAcc.slang so that the engine kernel
// compiles THE SAME TEXT this fixture verifies bitwise.  An #include and not a
// module, for the reason measured in P9.69a.

#include "pxgSlangCvxCoreAcc.slang"

struct WIn
{
    uint4 meta;              // x = numCalls (1..2)
    float4 transformQ, transformP;
    uint4 nums[2][8];        // per call: 32 lane point counts (uint4-packed)
    float4 pts[2][128];      // per call: lane*4 + j (xyz = local point, w = dist)
    float4 nrm[2][32];       // per call: per-lane local normal
}

struct WOut
{
    uint4 counts;            // x = mNumPatches, y = mNumPoints
    float4 patchN[MAX_PATCHES];  // w = asfloat(numPoints)
    float4 pool[MAX_PATCHES * MAX_PATCH_POINTS];
}

struct WParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<WParams> gParams;

RWStructuredBuffer<WIn> gIn;    // binding 0
RWStructuredBuffer<WOut> gOut;  // binding 1

[shader("compute")]
[numthreads(32, 1, 1)]
void cwaAddPoints(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const WIn s = gIn[t];

    // Contact state
    uint mNumPatches = 0u, mNumPoints = 0u;
    float3 patchNormal[MAX_PATCHES];
    uint patchCount[MAX_PATCHES];
    float4 mPoints[MAX_PATCHES * MAX_PATCH_POINTS]; // (p, d)
    for (uint i = 0u; i < MAX_PATCHES; ++i) { patchNormal[i] = float3(0, 0, 0); patchCount[i] = 0u; }
    for (uint i = 0u; i < MAX_PATCHES * MAX_PATCH_POINTS; ++i) mPoints[i] = float4(0, 0, 0, 0);

    const uint numCalls = min(s.meta.x, 2u);
    for (uint call = 0u; call < numCalls; ++call)
    {
        // Unpack this call's lanes out of the fixture record into the flat
        // arrays the shared accumulator takes.  The uint4 packing is this
        // RECORD's business, not the accumulator's.
        uint laneCount[WARP_SIZE];
        float3 laneNrm[WARP_SIZE];
        for (uint l = 0u; l < WARP_SIZE; ++l)
        {
            laneCount[l] = s.nums[call][l / 4u][l & 3u];
            laneNrm[l] = s.nrm[call][l].xyz;
        }
        float4 lanePts[WARP_SIZE * MAX_CONVEX_CONTACTS];
        for (uint i = 0u; i < WARP_SIZE * MAX_CONVEX_CONTACTS; ++i)
            lanePts[i] = s.pts[call][i];

        cvxCoreAddPoints32(mNumPatches, mNumPoints, patchNormal, patchCount, mPoints,
                           laneCount, lanePts, laneNrm, s.transformQ, s.transformP.xyz);
    }

    WOut r;
    r.counts = uint4(mNumPatches, mNumPoints, 0u, 0u);
    for (uint i = 0u; i < MAX_PATCHES; ++i)
        r.patchN[i] = float4(i < mNumPatches ? patchNormal[i] : float3(0, 0, 0), asfloat(i < mNumPatches ? patchCount[i] : 0u));
    for (uint i = 0u; i < MAX_PATCHES * MAX_PATCH_POINTS; ++i)
    {
        const uint pi = i / MAX_PATCH_POINTS;
        const bool live = pi < mNumPatches && (i % MAX_PATCH_POINTS) < patchCount[pi];
        r.pool[i] = live ? mPoints[i] : float4(0, 0, 0, 0);
    }
    gOut[t] = r;
}
