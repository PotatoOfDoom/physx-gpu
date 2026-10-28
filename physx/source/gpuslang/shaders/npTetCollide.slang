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

// npTetCollide.slang
//
// The leaf test for pxgSlangTetCollide -- tetPrimitivesCollide2 and computeInnerSphere.
// GJK and EPA have their own leaf tests (ctGjk, ctEpa) that pin their arithmetic; what this
// one pins is the DRIVER's policy, which is where a plausible-looking port goes wrong
// without any single line looking suspicious: which result short-circuits, when EPA is
// entered at all, and that a degenerate EPA answer forces the whole query to be re-run from
// a different seed.
import pxgSlangGjk;
import pxgSlangTetCollide;
import pxgSlangGjkWarp;

struct TcParams { uint numTests; uint pad0, pad1, pad2; }
struct TcIn
{
    float4 searchDir;      // xyz = search direction, w = contactDistance
    float4 spheres;        // x = inSphereRadius0, y = inSphereRadius1
    uint4 counts;          // x = nbA, y = nbB
}
struct TcOut
{
    uint result; uint degenerate; uint size; uint pad;
    float lowerBound; float upperBound; float innerSphere; float pad2;
    float4 direction; float4 closestPointA; float4 closestPointB; float4 tetCenter;
}

[[vk::push_constant]]
ConstantBuffer<TcParams> gParams;

RWStructuredBuffer<float4> gVA;   // binding 0: rigid cloud, 64 per test
RWStructuredBuffer<float4> gVB;   // binding 1: tet cloud, 64 per test
RWStructuredBuffer<TcIn> gIn;     // binding 2
RWStructuredBuffer<TcOut> gOut;   // binding 3

[shader("compute")]
[numthreads(64, 1, 1)]
void ctTetCollide(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint vbase = t * GJK_MAX_CLOUD;

    const TcIn e = gIn[t];

    GjkCloud a, b;
    a.n = e.counts.x;
    b.n = e.counts.y;
    for (uint i = 0u; i < GJK_MAX_CLOUD; ++i)
    {
        a.v[i] = (i < a.n) ? gVA[vbase + i].xyz : float3(0.0f, 0.0f, 0.0f);
        b.v[i] = (i < b.n) ? gVB[vbase + i].xyz : float3(0.0f, 0.0f, 0.0f);
    }

    GjkOut o;
    const uint r = tetPrimitivesCollide2(a, b, e.searchDir.xyz, e.searchDir.w,
                                         e.spheres.x, e.spheres.y, o);

    float3 center;
    const float inner = computeInnerSphere(b.v[0], b.v[1], b.v[2], b.v[3], center);

    TcOut out0;
    out0.result = r;
    out0.degenerate = o.degenerate;
    out0.size = o.size;
    out0.pad = 0u; out0.pad2 = 0.0f;
    out0.lowerBound = o.lowerBound;
    out0.upperBound = o.upperBound;
    out0.innerSphere = inner;
    out0.direction = o.direction;
    out0.closestPointA = o.closestPointA;
    out0.closestPointB = o.closestPointB;
    out0.tetCenter = float4(center, 0.0f);
    gOut[t] = out0;
}

// P9.89: computeInnerSphere alone, one tetrahedron per thread, its four vertices at
// gVB[4t .. 4t+3].  testNpGjk's npTetCollide_p89InnerSphereSelect compares the result
// bit for bit with the CUDA's selectMin chain (softbodyPrimitives.cu:140/:147/:154),
// including the NaN the CUDA keeps once the first face's distance is NaN.  Nothing else
// is written, so the GJK fields of TcOut are zero.
[shader("compute")]
[numthreads(64, 1, 1)]
void ctInnerSphere(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;

    float3 center;
    const float inner = computeInnerSphere(gVB[t * 4u + 0u].xyz, gVB[t * 4u + 1u].xyz,
                                           gVB[t * 4u + 2u].xyz, gVB[t * 4u + 3u].xyz, center);

    TcOut out0;
    out0.result = 0u; out0.degenerate = 0u; out0.size = 0u; out0.pad = 0u;
    out0.lowerBound = 0.0f; out0.upperBound = 0.0f; out0.pad2 = 0.0f;
    out0.innerSphere = inner;
    out0.direction = float4(0.0f, 0.0f, 0.0f, 0.0f);
    out0.closestPointA = float4(0.0f, 0.0f, 0.0f, 0.0f);
    out0.closestPointB = float4(0.0f, 0.0f, 0.0f, 0.0f);
    out0.tetCenter = float4(center, 0.0f);
    gOut[t] = out0;
}

// The WARP-COOPERATIVE driver (pxgSlangGjkWarp) on the same inputs as ctTetCollide: one
// 32-thread group per test (dispatch exactly numTests groups), the clouds loaded into
// groupshared by all lanes, lane 0 writing the result.  testNpGjk's
// npTetCollide_warpDifferential compares the two entries bit for bit.  `pad` carries the
// driver's path bits (GW_PATH_*) for the fixture's coverage count; it is the only field
// that differs by design.
[shader("compute")]
[numthreads(32, 1, 1)]
void ctTetCollideWarp(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    const uint t = gid.x;
    const uint lane = gtid.x;
    if (t >= gParams.numTests) return;   // group-uniform
    const uint vbase = t * GJK_MAX_CLOUD;
    const TcIn e = gIn[t];

    for (uint i = lane; i < GJK_MAX_CLOUD; i += 32u)
    {
        if (i < e.counts.x) gwCloudPut(0u, i, gVA[vbase + i].xyz);
        if (i < e.counts.y) gwCloudPut(1u, i, gVB[vbase + i].xyz);
    }

    GjkOut o;
    uint path;
    const uint r = gwTetPrimitivesCollide2(lane, e.counts.x, e.counts.y, e.searchDir.xyz, e.searchDir.w,
                                           e.spheres.x, e.spheres.y, o, path);

    float3 bv[4];
    for (uint i = 0u; i < 4u; ++i) bv[i] = (i < e.counts.y) ? gVB[vbase + i].xyz : float3(0.0f, 0.0f, 0.0f);
    float3 center;
    const float inner = computeInnerSphere(bv[0], bv[1], bv[2], bv[3], center);

    if (lane == 0u)
    {
        TcOut out0;
        out0.result = r;
        out0.degenerate = o.degenerate;
        out0.size = o.size;
        out0.pad = path; out0.pad2 = 0.0f;
        out0.lowerBound = o.lowerBound;
        out0.upperBound = o.upperBound;
        out0.innerSphere = inner;
        out0.direction = o.direction;
        out0.closestPointA = o.closestPointA;
        out0.closestPointB = o.closestPointB;
        out0.tetCenter = float4(center, 0.0f);
        gOut[t] = out0;
    }
}
