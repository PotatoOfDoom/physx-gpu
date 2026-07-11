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

// npConvexTriOut.slang
//
// Slang port of the output stage of convexTriangleContactGen (convexTriangle.cuh
// :1826-1881) plus barycentricCoordinates (:1538) and getTriangleActiveMask
// (:1562) — tranche 110.  One warp handles ONE convex-triangle pair, so unlike
// the sphere/capsule warp output stages (tranche 106/107, 32 pairs per warp)
// there is NO per-lane contact aggregation and NO batch-drop; only thread 0
// writes and a single atomicAdd reserves the deferred second-pass slot.
//
// Given the contact-generation results for the pair (maxSeparation,
// nbOutputContacts, delayContacts, minNormal, the first contact point pa, the
// gpu/remapped triangle indices and the pair offsets) it writes the five
// per-pair outputs:
//  - cvxTriMaxDepth[pairOffset] = maxSeparation;
//  - if delayContacts: append (globalWarpIndex | getTriangleActiveMask(pa,tri))
//    to the second-pass list (reserved by atomicAdd) — the active mask
//    classifies the first contact's barycentric region (a vertex/edge above the
//    0.97 upper bound) so the second pass knows which neighbour triangle to
//    re-check;
//  - cvxTriInterm[pairOffset].gpuTriIndex = triangleIdx;
//  - orderedCvxTriInterm[paddedOffset] = (nb && !delay ? 1<<31 : 0) | triangleIdx;
//  - ConvexTriNormalAndIndex[pairOffset] = { minNormal, deferredMask +
//    (nb << NbContactsShift) + remapCpuTriangleIdx }.
//
// Determinism: the outputs are direct copies (minNormal, maxSeparation) or
// integer packing; the only computed value is the barycentric active mask, whose
// v/w classification against 0.97 is kept unambiguous by the fixtures (contacts
// placed at exact barycentric coordinates well off the 0.97 boundary), so the
// mask is bit-exact.  explicit dotX/crossX match the CPU FMA contraction.  The
// second-pass append order is nondeterministic (atomicAdd), so the test compares
// the appended set as a multiset.

import pxgSlangCommon;

static const uint DeferredContactMask = 0x80000000u;
static const uint NbContactsShift = 27u;
static const uint eV0 = 1u << 29;
static const uint eV1 = 2u << 29;
static const uint eV2 = 3u << 29;
static const uint eE01 = 4u << 29;
static const uint eE02 = 5u << 29;
static const uint eE12 = 6u << 29;

struct CoParams { uint numPairs; uint pad0, pad1, pad2; }

struct CoIn
{
    float4 minNormal; // xyz normal; w = maxSeparation
    float4 pa;        // xyz first contact point; w unused
    float4 triV0, triV1, triV2;
    uint4 misc;       // x=nbOutputContacts, y=delayContacts, z=triangleIdx, w=remapCpuTriangleIdx
    uint4 misc2;      // x=globalWarpIndex, y=pairOffset, z=paddedOffset, w unused
}

[[vk::push_constant]]
ConstantBuffer<CoParams> gParams;

RWStructuredBuffer<CoIn> gIn;         // binding 0
RWStructuredBuffer<float4> gNI;       // binding 1: per pairOffset (normal, asfloat(index))
RWStructuredBuffer<float> gMaxDepth;  // binding 2: per pairOffset
RWStructuredBuffer<uint> gInterm;     // binding 3: per pairOffset (gpuTriIndex)
RWStructuredBuffer<uint> gOrdered;    // binding 4: per paddedOffset
RWStructuredBuffer<uint> gSecondPass; // binding 5: deferred appends
RWStructuredBuffer<uint> gCounter;    // binding 6: [0] nbSecondPassPairs

float dotX(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 crossX(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

void barycentricCoordinates(float3 p, float3 a, float3 b, float3 c, out float v, out float w)
{
    const float3 ab = b - a, ac = c - a;
    const float3 n = crossX(ab, ac);
    const float3 ap = a - p, bp = b - p, cp = c - p;
    const float3 bCrossC = crossX(bp, cp);
    const float3 cCrossA = crossX(cp, ap);
    const float3 aCrossB = crossX(ap, bp);
    const float va = dotX(n, bCrossC);
    const float vb = dotX(n, cCrossA);
    const float vc = dotX(n, aCrossB);
    const float totalArea = va + vb + vc;
    const float denom = totalArea == 0.0f ? 0.0f : 1.0f / totalArea;
    v = vb * denom;
    w = vc * denom;
}

uint getTriangleActiveMaskVW(float v, float w)
{
    const float upperBound = 0.97f;
    const float u = 1.0f - v - w;
    if (v > upperBound) return eV1;
    if (w > upperBound) return eV2;
    if (u > upperBound) return eV0;
    if ((v + u) > upperBound) return eE01;
    if ((u + w) > upperBound) return eE02;
    if ((w + v) > upperBound) return eE12;
    return 0u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ctConvexTriOut(uint3 dtid : SV_DispatchThreadID)
{
    const uint p = dtid.x;
    if (p >= gParams.numPairs) return;

    const CoIn e = gIn[p];
    const float3 minNormal = e.minNormal.xyz;
    const float maxSeparation = e.minNormal.w;
    const uint nbOutputContacts = e.misc.x;
    const bool delayContacts = e.misc.y != 0u;
    const uint triangleIdx = e.misc.z;
    const uint remapCpuTriangleIdx = e.misc.w;
    const uint globalWarpIndex = e.misc2.x;
    const uint pairOffset = e.misc2.y;
    const uint paddedOffset = e.misc2.z;

    gMaxDepth[pairOffset] = maxSeparation;

    if (delayContacts)
    {
        uint startIndex = 0;
        InterlockedAdd(gCounter[0], 1u, startIndex);
        float v, w;
        barycentricCoordinates(e.pa.xyz, e.triV0.xyz, e.triV1.xyz, e.triV2.xyz, v, w);
        const uint mask = getTriangleActiveMaskVW(v, w);
        gSecondPass[startIndex] = globalWarpIndex | mask;
    }

    gInterm[pairOffset] = triangleIdx;
    const uint orderedMask = (nbOutputContacts != 0u && !delayContacts) ? (1u << 31) : 0u;
    gOrdered[paddedOffset] = orderedMask | triangleIdx;

    const uint delayContactMask = delayContacts ? DeferredContactMask : 0u;
    const uint niIndex = delayContactMask + (nbOutputContacts << NbContactsShift) + remapCpuTriangleIdx;
    gNI[pairOffset] = float4(minNormal, asfloat(niIndex));
}
