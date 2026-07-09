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

// npNpUtils.slang
//
// Slang port of a family of small narrow-phase predicate helpers: next3/prev3
// (the cyclic +/-1 index step over a triangle's 0,1,2 vertices, from
// nputils.cuh), isActiveEdge / isBoundaryEdge / isEdgeNonconvex (the
// adjacency-flag bit tests over an edge index, from convexTriangle.cuh /
// triangle.cuh — BOUNDARY is the grid/hull edge sentinel and NONCONVEX_FLAG
// marks a reflex/non-convex adjacent edge) and delayGenerateContact (when the
// minimum separating axis is a convex-hull face normal, defer contact generation
// to a second pass unless the axis is within ~45 degrees of the triangle normal,
// from convexTriangle.cuh).

struct NuParams { uint numTests; uint pad0, pad1, pad2; }

struct DelayIn
{
    float4 minNormal;         // xyz
    float4 triangleLocNormal; // xyz
    uint featureIndex;
    uint p0, p1, p2;
}

[[vk::push_constant]]
ConstantBuffer<NuParams> gParams;

static const uint BOUNDARY = 0xffffffffu;
static const uint NONCONVEX_FLAG = 0x80000000u;
static const uint EDGE_FEATURE_IDX = 0x80000000u;
static const uint TRI_FEATURE_IDX = 0x7fffffffu;

RWStructuredBuffer<uint> gX;        // binding 0  index for next3/prev3
RWStructuredBuffer<uint2> gX2Out;   // binding 1  (next3, prev3)
RWStructuredBuffer<uint> gEdge;     // binding 2  edge index
RWStructuredBuffer<uint> gEdgeOut;  // binding 3  packed: bit0 active, bit1 boundary, bit2 nonconvex
RWStructuredBuffer<DelayIn> gDelay; // binding 4
RWStructuredBuffer<uint> gDelayOut; // binding 5  (0/1)

int next3(int x) { return x == 2 ? 0 : x + 1; }
int prev3(int x) { return x == 0 ? 2 : x - 1; }

bool isActiveEdge(uint edgeIndex)   { return edgeIndex == BOUNDARY || (edgeIndex & NONCONVEX_FLAG) == 0u; }
bool isBoundaryEdge(uint edgeIndex) { return edgeIndex == BOUNDARY; }
bool isEdgeNonconvex(uint adjEdgeIndex) { return (adjEdgeIndex != BOUNDARY) && ((adjEdgeIndex & NONCONVEX_FLAG) != 0u); }

bool delayGenerateContact(float3 minNormal, float3 triangleLocNormal, uint featureIndex)
{
    if (featureIndex != EDGE_FEATURE_IDX && featureIndex != TRI_FEATURE_IDX)
    {
        const float3 triLocNormal = -triangleLocNormal;
        const float cosTheta = dot(minNormal, triLocNormal);
        const float threshold = 0.707106781f; // about 45 degrees
        return cosTheta < threshold;
    }
    return false;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void nuNext3Prev3(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const int x = int(gX[i]);
    gX2Out[i] = uint2(uint(next3(x)), uint(prev3(x)));
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void nuEdgeFlags(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const uint e = gEdge[i];
    uint packed = 0u;
    if (isActiveEdge(e))      packed |= 1u;
    if (isBoundaryEdge(e))    packed |= 2u;
    if (isEdgeNonconvex(e))   packed |= 4u;
    gEdgeOut[i] = packed;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void nuDelayContact(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const DelayIn e = gDelay[i];
    const bool r = delayGenerateContact(e.minNormal.xyz, e.triangleLocNormal.xyz, e.featureIndex);
    gDelayOut[i] = r ? 1u : 0u;
}
