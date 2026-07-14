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

// npCvxCoreContact.slang
//
// Slang port of Gu::Contact (GuConvexSupport.h L895-1048), the contact-patch
// accumulator the convex-core mesh kernels drive point-by-point:
//   - addPoint: first patch whose normal.dot(normal) >= SAME_NORMAL (0.999)
//     receives the point via addPatchPoint, else addPatch;
//   - addPatch: below MAX_PATCHES (16) appends (normal, 1 point); at
//     capacity it computes maxDot0 = max over existing normals of
//     normal.dot(n_i) and the closest existing PAIR (pi0, pi1) by n_i.n_j
//     over i < j; if maxDot0 > maxDot1 the new patch is dropped; otherwise
//     the shallower of the pair (greater min-depth over its points --
//     depth accumulated with PxMin from point 0) is REPLACED by the new
//     patch (mNumPoints adjusted by the replaced patch's count);
//   - addPatchPoint: the new point is prepended, then bubbled back by a
//     depth-sorted swap chain that BREAKS at the first non-swap; then
//     reducePolygon over the (p, d) points (stride 16, keep1st, normal =
//     patch normal) reindexes the slots, and mNumPoints is rebalanced.
// All dot/compare arithmetic: ALL outputs verify bitwise.

static const float FLT_EPS = 1.1920928955078125e-7f;
static const uint NOIDX = 0xFFFFFFFFu;
static const uint MAX_PATCHES = 16u;
static const uint MAX_PATCH_POINTS = 4u;
static const uint MAX_ADDS = 24u;

struct AccIn
{
    uint4 meta;             // x = numAdds
    float4 pts[MAX_ADDS];   // point positions (w = depth)
    float4 nrm[MAX_ADDS];   // point normals (w pad)
}

struct AccOut
{
    uint4 counts;           // x = mNumPatches, y = mNumPoints
    float4 patchN[MAX_PATCHES]; // patch normal (w = patch numPoints as float bits via asfloat)
    float4 pool[MAX_PATCHES * MAX_PATCH_POINTS]; // mPoints image (p, d)
}

struct AccParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<AccParams> gParams;

RWStructuredBuffer<AccIn> gIn;    // binding 0
RWStructuredBuffer<AccOut> gOut;  // binding 1

float dot3Q(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 crossQ(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
float magSq(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

// Gu::reducePolygon over (p, d) Point structs (stride 16; values only)
uint reducePolygonPQ(const float4 pts[MAX_PATCH_POINTS + 1u], uint count, float3 normal, out uint inds[4])
{
    inds[0] = NOIDX; inds[1] = NOIDX; inds[2] = NOIDX; inds[3] = NOIDX;
    uint pi0 = 0u, pi1 = NOIDX;
    {
        float maxDist = -3.402823466e+38f;
        const float3 p0 = pts[0].xyz;
        for (uint j = 0u; j < count; ++j)
        {
            const float dist = magSq(pts[j].xyz - p0);
            if (dist > maxDist + FLT_EPS)
            {
                maxDist = dist;
                pi1 = j;
            }
        }
    }

    if (pi0 == NOIDX || pi1 == NOIDX)
        return 0u;

    if (pi0 == pi1)
    {
        inds[0] = pi0;
        return 1u;
    }

    uint pi2 = NOIDX, pi3 = NOIDX;
    {
        const float3 p0 = pts[pi0].xyz;
        const float3 p1 = pts[pi1].xyz;
        const float3 n01 = crossQ(normal, p1 - p0);
        float minDist = 0.0f, maxDist = 0.0f;
        for (uint i = 0u; i < count; ++i)
        {
            const float d = dot3Q(n01, pts[i].xyz - p0);
            if (d < minDist - FLT_EPS)
            {
                pi2 = i;
                minDist = d;
            }
            if (d > maxDist + FLT_EPS)
            {
                pi3 = i;
                maxDist = d;
            }
        }
    }

    uint numPoints = 0u;
    inds[numPoints++] = pi0;
    inds[numPoints++] = pi1;
    if (pi2 != NOIDX && pi2 != pi0 && pi2 != pi1)
        inds[numPoints++] = pi2;
    if (pi3 != NOIDX && pi3 != pi0 && pi3 != pi1 && pi3 != pi2)
        inds[numPoints++] = pi3;

    return numPoints;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cctAddPoints(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const AccIn s = gIn[t];

    const float SAME_NORMAL = 0.999f; // Contact member init (L953)

    uint numPatches = 0u, numPoints = 0u;
    float3 patchNormal[MAX_PATCHES];
    uint patchCount[MAX_PATCHES];
    float4 pool[MAX_PATCHES * MAX_PATCH_POINTS];
    for (uint i = 0u; i < MAX_PATCHES; ++i) { patchNormal[i] = float3(0, 0, 0); patchCount[i] = 0u; }
    for (uint i = 0u; i < MAX_PATCHES * MAX_PATCH_POINTS; ++i) pool[i] = float4(0, 0, 0, 0);

    const uint numAdds = min(s.meta.x, MAX_ADDS);
    for (uint a = 0u; a < numAdds; ++a)
    {
        const float3 position = s.pts[a].xyz;
        const float depth = s.pts[a].w;
        const float3 normal = s.nrm[a].xyz;

        // addPoint (L936-949)
        uint target = NOIDX;
        for (uint i = 0u; i < numPatches && target == NOIDX; ++i)
        {
            if (dot3Q(patchNormal[i], normal) >= SAME_NORMAL)
                target = i;
        }

        if (target != NOIDX)
        {
            // addPatchPoint (L1022-1047)
            float4 points[MAX_PATCH_POINTS + 1u] =
                { float4(0, 0, 0, 0), float4(0, 0, 0, 0), float4(0, 0, 0, 0), float4(0, 0, 0, 0), float4(0, 0, 0, 0) };
            points[0] = float4(position, depth);
            for (uint i = 0u; i < patchCount[target]; ++i)
                points[i + 1u] = pool[target * MAX_PATCH_POINTS + i];
            for (uint i = 0u; i < patchCount[target]; ++i)
            {
                if (points[i].w > points[i + 1u].w)
                {
                    const float4 tv = points[i]; points[i] = points[i + 1u]; points[i + 1u] = tv;
                }
                else
                    break;
            }

            uint inds[4];
            const uint np = reducePolygonPQ(points, patchCount[target] + 1u, patchNormal[target], inds);
            for (uint i = 0u; i < np; ++i)
                pool[target * MAX_PATCH_POINTS + i] = points[inds[i]];

            numPoints -= patchCount[target];
            patchCount[target] = np;
            numPoints += np;
        }
        else if (numPatches < MAX_PATCHES)
        {
            // addPatch append (L966-977)
            const uint patchIndex = numPatches++;
            patchNormal[patchIndex] = normal;
            patchCount[patchIndex] = 1u;
            pool[patchIndex * MAX_PATCH_POINTS] = float4(position, depth);
            ++numPoints;
        }
        else
        {
            // addPatch at capacity (L979-1018)
            float maxDot0 = -3.402823466e+38f;
            for (uint i = 0u; i < numPatches; ++i)
                maxDot0 = max(maxDot0, dot3Q(normal, patchNormal[i]));

            float maxDot1 = -3.402823466e+38f;
            uint pi0 = NOIDX, pi1 = NOIDX;
            for (uint i = 0u; i < numPatches; ++i)
            {
                for (uint j = i + 1u; j < numPatches; ++j)
                {
                    const float d = dot3Q(patchNormal[i], patchNormal[j]);
                    if (d > maxDot1)
                    {
                        maxDot1 = d;
                        pi0 = i; pi1 = j;
                    }
                }
            }

            if (maxDot0 > maxDot1)
                continue; // drop the new patch

            float depth0 = pool[pi0 * MAX_PATCH_POINTS].w;
            for (uint i = 1u; i < patchCount[pi0]; ++i)
                depth0 = min(depth0, pool[pi0 * MAX_PATCH_POINTS + i].w);

            float depth1 = pool[pi1 * MAX_PATCH_POINTS].w;
            for (uint i = 1u; i < patchCount[pi1]; ++i)
                depth1 = min(depth1, pool[pi1 * MAX_PATCH_POINTS + i].w);

            const uint patchIndex = depth0 > depth1 ? pi0 : pi1;
            numPoints -= patchCount[patchIndex];
            patchNormal[patchIndex] = normal;
            patchCount[patchIndex] = 1u;
            pool[patchIndex * MAX_PATCH_POINTS] = float4(position, depth);
            ++numPoints;
        }
    }

    AccOut r;
    r.counts = uint4(numPatches, numPoints, 0u, 0u);
    for (uint i = 0u; i < MAX_PATCHES; ++i)
        r.patchN[i] = float4(i < numPatches ? patchNormal[i] : float3(0, 0, 0), asfloat(i < numPatches ? patchCount[i] : 0u));
    for (uint i = 0u; i < MAX_PATCHES * MAX_PATCH_POINTS; ++i)
    {
        const uint pi = i / MAX_PATCH_POINTS;
        const bool live = pi < numPatches && (i % MAX_PATCH_POINTS) < patchCount[pi];
        r.pool[i] = live ? pool[i] : float4(0, 0, 0, 0);
    }
    gOut[t] = r;
}
