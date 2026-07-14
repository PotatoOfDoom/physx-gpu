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
//     SAME_NORMAL = 0.999, strict >); reducePoints picks <= 4 best points
//     (deepest, farthest, two side extremes along cross(n, p1-p0), indices
//     OR-ed so duplicates collapse); the picks land in tmpBuffer by
//     popcount offsets; the target patch's existing points are appended;
//     a second reducePoints over the union writes the patch back; lane 0
//     swaps the deepest pick to slot 0 and rebalances the counters.
//   - findCreatePatch: maxIndex over dot(normal, patchNormal[lane]) with
//     lanes >= mNumPatches contributing dot(normal, 0) = 0; > SAME_NORMAL
//     returns that patch; below MAX_PATCHES appends; at capacity each lane
//     t scans patches t+1..15 for its closest OTHER patch, maxIndex picks
//     the closest pair, and if that pair is closer than the new normal to
//     any patch, the pair member with the LARGER slot-0 dist (shallower) is
//     reset to receive the new patch, else the new patch is dropped
//     (0xffffffff).
// All dot/compare arithmetic: ALL outputs verify bitwise.

static const float FLT_MAXQ = 3.402823466e+38f;
static const float SAME_NORMAL = 0.999f;
static const uint WARP_SIZE = 32u;
static const uint MAX_PATCHES = 16u;
static const uint MAX_PATCH_POINTS = 4u;
static const uint MAX_CONVEX_CONTACTS = 4u;
static const uint NOIDX = 0xFFFFFFFFu;

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

float dot3Q(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 crossQ(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
float magQ(float3 v) { return sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
float3 quatRotQ(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2);
}

// minIndex/maxIndex: LOWEST lane index achieving the min/max over mask lanes
uint minIndexQ(const float v[WARP_SIZE], uint mask, out float best)
{
    best = FLT_MAXQ;
    uint idx = NOIDX;
    for (uint l = 0u; l < WARP_SIZE; ++l)
        if ((mask & (1u << l)) != 0u && v[l] < best) { best = v[l]; idx = l; }
    return idx;
}
uint maxIndexQ(const float v[WARP_SIZE], uint mask, out float best)
{
    best = -FLT_MAXQ;
    uint idx = NOIDX;
    for (uint l = 0u; l < WARP_SIZE; ++l)
        if ((mask & (1u << l)) != 0u && v[l] > best) { best = v[l]; idx = l; }
    return idx;
}

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
        // ---- copyPoints ----
        uint laneNum[WARP_SIZE];
        for (uint l = 0u; l < WARP_SIZE; ++l)
            laneNum[l] = min(s.nums[call][l / 4u][l & 3u], MAX_CONVEX_CONTACTS);

        float4 newPts[WARP_SIZE * MAX_CONVEX_CONTACTS]; // (p world, d)
        float3 newNrm[WARP_SIZE * MAX_CONVEX_CONTACTS];
        for (uint i = 0u; i < WARP_SIZE * MAX_CONVEX_CONTACTS; ++i)
        { newPts[i] = float4(0, 0, 0, FLT_MAXQ); newNrm[i] = float3(0, 0, 0); }

        uint prefix = 0u;
        uint totalPoints = 0u;
        for (uint l = 0u; l < WARP_SIZE; ++l)
        {
            prefix += laneNum[l]; // inclusive warpScan
            const uint off = prefix - laneNum[l];
            const float3 worldNormal = quatRotQ(s.transformQ, s.nrm[call][l].xyz);
            for (uint j = 0u; j < laneNum[l]; ++j)
            {
                const float4 lp = s.pts[call][l * MAX_CONVEX_CONTACTS + j];
                const float3 wp = quatRotQ(s.transformQ, lp.xyz) + s.transformP.xyz;
                newPts[off + j] = float4(wp, lp.w);
                newNrm[off + j] = worldNormal;
            }
            if (l == WARP_SIZE - 1u) totalPoints = prefix; // shfl from lane 31
        }

        // ---- addPoints round loop ----
        const uint numFullWarps = (totalPoints + WARP_SIZE - 1u) / WARP_SIZE;
        for (uint round = 0u; round < numFullWarps; ++round)
        {
            // per-lane loads (dummy: d = FLT_MAX, n = 0)
            float4 lp[WARP_SIZE];
            float3 ln[WARP_SIZE];
            uint allMask = 0u;
            for (uint l = 0u; l < WARP_SIZE; ++l)
            {
                const uint i = round * WARP_SIZE + l;
                if (i < totalPoints)
                {
                    allMask |= (1u << l);
                    lp[l] = newPts[i]; ln[l] = newNrm[i];
                }
                else { lp[l] = float4(0, 0, 0, FLT_MAXQ); ln[l] = float3(0, 0, 0); }
            }

            uint guard = 0u;
            while (allMask != 0u && guard < 40u)
            {
                ++guard;
                // deepest point over allMask
                float dists[WARP_SIZE];
                for (uint l = 0u; l < WARP_SIZE; ++l) dists[l] = lp[l].w;
                float minDist;
                const uint minDistThread = minIndexQ(dists, allMask, minDist);
                const float3 minDistNormal = ln[minDistThread]; // shuffle

                // lanes whose normal matches the deepest one's
                uint sameNormalMask = 0u;
                for (uint l = 0u; l < WARP_SIZE; ++l)
                    if (dot3Q(ln[l], minDistNormal) > SAME_NORMAL) sameNormalMask |= (1u << l);
                sameNormalMask &= allMask;

                // ---- reducePoints over sameNormalMask ----
                float tmpv[WARP_SIZE];
                float md;
                for (uint l = 0u; l < WARP_SIZE; ++l) tmpv[l] = lp[l].w;
                const uint pi0 = minIndexQ(tmpv, sameNormalMask, md);
                const uint deepest = pi0;
                const float3 p0 = lp[pi0].xyz;
                for (uint l = 0u; l < WARP_SIZE; ++l) tmpv[l] = magQ(p0 - lp[l].xyz);
                const uint pi1 = maxIndexQ(tmpv, sameNormalMask, md);
                const float3 p1 = lp[pi1].xyz;
                const float3 sideN = crossQ(minDistNormal, p1 - p0);
                for (uint l = 0u; l < WARP_SIZE; ++l) tmpv[l] = dot3Q(sideN, lp[l].xyz - p0);
                const uint pi2 = minIndexQ(tmpv, sameNormalMask, md);
                const uint pi3 = maxIndexQ(tmpv, sameNormalMask, md);
                uint bestPointsMask = (1u << pi0) | (1u << pi1) | (1u << pi2) | (1u << pi3);

                // write picks to tmpBuffer by popcount offsets
                float4 tmpBuffer[8];
                for (uint i = 0u; i < 8u; ++i) tmpBuffer[i] = float4(0, 0, 0, 0);
                uint count = countbits(bestPointsMask);
                for (uint l = 0u; l < WARP_SIZE; ++l)
                    if ((bestPointsMask & (1u << l)) != 0u)
                        tmpBuffer[countbits(bestPointsMask & ((1u << l) - 1u))] = lp[l];

                allMask &= ~sameNormalMask;

                // ---- findCreatePatch(minDistNormal) ----
                uint patchIndex = NOIDX;
                {
                    float dots[WARP_SIZE];
                    for (uint l = 0u; l < WARP_SIZE; ++l)
                    {
                        const float3 pn = l < mNumPatches ? patchNormal[l] : float3(0, 0, 0);
                        dots[l] = dot3Q(minDistNormal, pn);
                    }
                    float maxDot;
                    const uint maxDotThread = maxIndexQ(dots, 0xFFFFFFFFu, maxDot);

                    if (maxDot > SAME_NORMAL)
                        patchIndex = maxDotThread;
                    else if (mNumPatches < MAX_PATCHES)
                    {
                        // lane 0 appends
                        patchNormal[mNumPatches] = minDistNormal;
                        patchCount[mNumPatches] = 0u;
                        ++mNumPatches;
                        patchIndex = mNumPatches - 1u;
                    }
                    else
                    {
                        // capacity: per-lane closest OTHER patch scan
                        float pMax[WARP_SIZE];
                        uint pClosest[WARP_SIZE];
                        for (uint l = 0u; l < WARP_SIZE; ++l)
                        {
                            const float3 pn = l < mNumPatches ? patchNormal[l] : float3(0, 0, 0);
                            float patchMaxDot = -FLT_MAXQ;
                            uint closestPatchThread = NOIDX;
                            for (uint i = l + 1u; i < MAX_PATCHES; ++i)
                            {
                                const float d = dot3Q(pn, patchNormal[i]);
                                if (patchMaxDot < d) { patchMaxDot = d; closestPatchThread = i; }
                            }
                            pMax[l] = patchMaxDot; pClosest[l] = closestPatchThread;
                        }
                        float patchMaxDot;
                        const uint patchMaxDotThread = maxIndexQ(pMax, 0xFFFFFFFFu, patchMaxDot);

                        if (patchMaxDot > maxDot)
                        {
                            const uint closestPatchThread = pClosest[patchMaxDotThread]; // shfl
                            const float dist0 = mPoints[patchMaxDotThread * MAX_PATCH_POINTS].w;
                            const float dist1 = mPoints[closestPatchThread * MAX_PATCH_POINTS].w;
                            const uint pi = dist0 > dist1 ? patchMaxDotThread : closestPatchThread;
                            // lane 0 resets the shallower patch
                            mNumPoints -= patchCount[pi];
                            patchCount[pi] = 0u;
                            patchNormal[pi] = minDistNormal;
                            patchIndex = pi;
                        }
                        // else: drop (patchIndex stays NOIDX)
                    }
                }

                if (patchIndex == NOIDX)
                    continue;

                // append existing patch points into tmpBuffer
                for (uint j = 0u; j < patchCount[patchIndex]; ++j)
                    tmpBuffer[count + j] = mPoints[patchIndex * MAX_PATCH_POINTS + j];
                count += patchCount[patchIndex];

                // second reducePoints over the union (lanes 0..count-1)
                const uint patchPointMask = count >= 32u ? 0xFFFFFFFFu : ((1u << count) - 1u);
                float4 pp[WARP_SIZE];
                for (uint l = 0u; l < WARP_SIZE; ++l)
                    pp[l] = l < count ? tmpBuffer[l] : float4(0, 0, 0, 0); // Point() zero-init
                for (uint l = 0u; l < WARP_SIZE; ++l) tmpv[l] = pp[l].w;
                float md2;
                const uint qi0 = minIndexQ(tmpv, patchPointMask, md2);
                const uint deepest2 = qi0;
                const float3 q0 = pp[qi0].xyz;
                for (uint l = 0u; l < WARP_SIZE; ++l) tmpv[l] = magQ(q0 - pp[l].xyz);
                const uint qi1 = maxIndexQ(tmpv, patchPointMask, md2);
                const float3 q1 = pp[qi1].xyz;
                const float3 sideN2 = crossQ(minDistNormal, q1 - q0);
                for (uint l = 0u; l < WARP_SIZE; ++l) tmpv[l] = dot3Q(sideN2, pp[l].xyz - q0);
                const uint qi2 = minIndexQ(tmpv, patchPointMask, md2);
                const uint qi3 = maxIndexQ(tmpv, patchPointMask, md2);
                bestPointsMask = (1u << qi0) | (1u << qi1) | (1u << qi2) | (1u << qi3);

                // write the best points back into the patch
                const uint newCount = countbits(bestPointsMask);
                for (uint l = 0u; l < WARP_SIZE; ++l)
                    if ((bestPointsMask & (1u << l)) != 0u)
                        mPoints[patchIndex * MAX_PATCH_POINTS + countbits(bestPointsMask & ((1u << l) - 1u))] = pp[l];

                // lane 0: move the deepest point to the beginning
                const uint dOff = countbits(bestPointsMask & ((1u << deepest2) - 1u));
                const float4 tv = mPoints[patchIndex * MAX_PATCH_POINTS];
                mPoints[patchIndex * MAX_PATCH_POINTS] = mPoints[patchIndex * MAX_PATCH_POINTS + dOff];
                mPoints[patchIndex * MAX_PATCH_POINTS + dOff] = tv;

                // adjust point count
                mNumPoints -= patchCount[patchIndex];
                patchCount[patchIndex] = newCount;
                mNumPoints += patchCount[patchIndex];
            }
        }
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
