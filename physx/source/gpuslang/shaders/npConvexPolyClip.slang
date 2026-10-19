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

// npConvexPolyClip.slang
//
// Slang port of the RIGID convex-polygon vs triangle clipper polyClip
// (convexTriangle.cuh :943-1058) plus addContacts (:918-940) — tranche 111.
// This is distinct from the cloth polyClip4 (tranche 95, femClothPrimitives.cu):
// it is a genuine warp-SIMD algorithm in which each LANE holds a distinct poly0
// (convex face) vertex/edge and the lanes communicate via shuffle / shuffleDot /
// __shfl_xor_sync / __ballot_sync / lowestSetIndex / lowestSetBit.  Poly0 (the
// convex face) has nbVerts0 verts in lanes 0..nbVerts0-1; poly1 (the triangle)
// has 3 verts in lanes 0..2.  The clip produces contacts where
//   (1) a triangle vertex lies inside every poly0 edge-plane (projected along
//       the axis to the poly0 plane) — phase 1;
//   (2) a poly0 vertex lies inside every triangle edge-plane (projected to the
//       triangle plane) — phase 2;
//   (3) a poly0 edge crosses a triangle edge (the poly0 edge is ray-traced to
//       the triangle plane) — phase 3.
// addContacts appends the surviving lanes' (pos, sep) to the transposed contact
// buffer, dropping any with sep > maxSep (NaN-safe) and compacting via an
// exclusive population scan up to NUM_TMP_CONTACTS_PER_PAIR = 32.
//
// Serialisation: one thread simulates the whole 32-lane warp with per-lane
// arrays; shuffle(x, k) -> x[k], shuffleDot(v, k, w[L]) -> dot(v[k], w[L]),
// __shfl_xor butterflies (AND/OR reductions) -> plain reductions, ballots ->
// bit loops.  The order-independent AND/OR reductions and the ballots are
// bit-identical to the warp; addContacts's exclusive scan uses the ENTERING
// contact count for every lane (matching the simultaneous warp scan).
//
// Not modelled (faithful to the active CUDA): triEdgeMask (the `e1cross &=
// triEdgeMask` masking is commented out in the source and the driver passes
// 0xFFFFFFFF), and s0d (computed but only used by commented-out code).
//
// Determinism: axis-aligned square poly0 + a coplanar triangle with axis =
// plane normal make the projections (s = +-t/dnom, dnom = +-1) exact and the
// edge-edge ray-trace lands on representable coordinates; explicit dotX/crossX
// match the CPU FMA contraction.  Contact order within a phase is deterministic
// (lane order), so the buffer is compared position-for-position.

import pxgSlangCommon;

static const uint NUM_TMP = 32u;
static const float PC_MAX_F32 = 3.402823466e38f;

struct PcParams
{
    uint numTests;
    uint maxVerts0; // stride (in float4s) of each test's poly0-vertex block
    uint pad0, pad1;
}

struct PcIn
{
    float4 plane0;  // convex face plane: n.xyz, d = w
    float4 plane1;  // triangle plane: n.xyz, d = w
    float4 axis;    // separating axis xyz; w = maxSep
    uint4 cfg;      // x=nbVerts0, y=initContacts, z/w unused
}

[[vk::push_constant]]
ConstantBuffer<PcParams> gParams;

RWStructuredBuffer<float4> gPoly0;    // binding 0: poly0 verts (maxVerts0/test)
RWStructuredBuffer<float4> gTri;      // binding 1: 3 triangle verts/test
RWStructuredBuffer<PcIn> gIn;         // binding 2
RWStructuredBuffer<float4> gContacts; // binding 3: NUM_TMP contacts/test (pos.xyz, sep)
RWStructuredBuffer<uint> gCount;      // binding 4: nbContacts/test

float dotX(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 crossX(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
float planeDist(float4 pl, float3 p) { return dotX(pl.xyz, p) + pl.w; }
uint popc(uint x) { return countbits(x); }
uint lowestSetIndex(uint x) { return firstbitlow(x); }
uint lowestSetBit(uint x) { return x & (0u - x); }

// per-thread warp state (one thread == one whole warp / one clip)
struct Warp
{
    float3 pos[32];
    float sep[32];
    uint contactCount;
    uint cbase; // contact-buffer base (t * NUM_TMP)
};

// addContacts: append lanes in `flags` (after the sep<=maxSep NaN-safe gate),
// compacted by exclusive pop-scan, up to NUM_TMP.
void addContacts(inout Warp wp, uint flags, float maxSep)
{
    uint keep = 0u;
    for (uint L = 0u; L < 32u; ++L)
        if ((flags & (1u << L)) != 0u && !(wp.sep[L] > maxSep))
            keep |= (1u << L);
    const uint entry = wp.contactCount;
    uint added = 0u;
    for (uint L = 0u; L < 32u; ++L)
    {
        if ((keep & (1u << L)) == 0u) continue;
        const uint index = popc(keep & ((1u << L) - 1u)) + entry;
        if (index < NUM_TMP)
        {
            gContacts[wp.cbase + index] = float4(wp.pos[L], wp.sep[L]);
            added++;
        }
    }
    wp.contactCount = entry + added;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ctPolyClip(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;

    const PcIn e = gIn[t];
    const float4 plane0 = e.plane0;
    const float4 plane1 = e.plane1;
    const float3 axis = e.axis.xyz;
    const float maxSep = e.axis.w;
    const uint nbVerts0 = e.cfg.x;
    const uint initContacts = e.cfg.y;
    const uint vbase = t * gParams.maxVerts0;

    // load lane data
    float3 v0[32], v1[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        v0[L] = (L < nbVerts0) ? gPoly0[vbase + L].xyz : float3(0, 0, 0);
        v1[L] = (L < 3u) ? gTri[t * 3u + L].xyz : float3(0, 0, 0);
    }

    Warp wp;
    wp.contactCount = initContacts;
    wp.cbase = t * NUM_TMP;

    const float dnom0 = dotX(plane1.xyz, axis);
    const float dnom1 = dotX(plane0.xyz, axis);

    float s0[32], s1[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        const float t0 = planeDist(plane1, v0[L]);
        const float t1 = planeDist(plane0, v1[L]);
        s0[L] = (t0 > maxSep || dnom0 == 0.0f) ? PC_MAX_F32 : -t0 / dnom0;
        s1[L] = (t1 > maxSep || dnom1 == 0.0f) ? PC_MAX_F32 : t1 / dnom1;
    }

    // poly0 edges + edge planes
    float3 e0[32], e0planeN[32];
    float e0planeD[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        const uint nxt = ((L + 1u == nbVerts0) ? 0u : (L + 1u)) & 31u; // & 31: match CUDA mod-32 shfl, avoid OOB at lane 31
        e0[L] = v0[nxt] - v0[L];
        e0planeN[L] = crossX(e0[L], axis);
        e0planeD[L] = dotX(e0planeN[L], v0[L]);
    }

    // ---- phase 1: triangle verts against poly0 edge-planes ----
    uint in1[32], out1[32];
    for (uint L = 0u; L < 32u; ++L) { in1[L] = 0u; out1[L] = 0u; }
    for (uint j = 0u; j < 3u; ++j)
    {
        const uint bj = 1u << j;
        for (uint L = 0u; L < 32u; ++L)
        {
            const float d = dotX(v1[j], e0planeN[L]) - e0planeD[L];
            if (d < 0.0f) in1[L] |= bj;
            if (d > 0.0f) out1[L] |= bj;
        }
    }
    for (uint L = 0u; L < 32u; ++L)
        if (L >= nbVerts0) { in1[L] = 0u; out1[L] = 0u; }

    // b = AND over lanes of ~out1, keep low 3 bits (triangle verts inside all planes)
    uint band = 0xffffffffu;
    for (uint L = 0u; L < 32u; ++L) band &= ~out1[L];
    for (uint L = 0u; L < 32u; ++L) { wp.pos[L] = v1[L] - s1[L] * axis; wp.sep[L] = s1[L]; }
    addContacts(wp, band & 7u, maxSep);

    bool anyOut1 = false;
    for (uint L = 0u; L < 32u; ++L) if (out1[L] != 0u) anyOut1 = true;
    if (!anyOut1) { gCount[t] = wp.contactCount; return; }

    // triangle edge crossings of poly0 planes (per poly0-plane lane)
    uint e1cross[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        const uint i1 = in1[L] | (in1[L] << 3);
        const uint o1 = out1[L] | (out1[L] << 3);
        e1cross[L] = ((i1 & (o1 >> 1)) | ((i1 >> 1) & o1)) & 7u;
    }

    // ---- phase 2: poly0 verts against triangle edge-planes ----
    float3 e1[32], e1planeN[32];
    float e1planeD[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        const uint nxt = ((L + 1u == 3u) ? 0u : (L + 1u)) & 31u; // & 31: match CUDA mod-32 shfl, avoid OOB at lane 31
        e1[L] = v1[nxt] - v1[L];
        e1planeN[L] = crossX(axis, e1[L]);
        e1planeD[L] = dotX(e1planeN[L], v1[L]);
    }
    uint in0[32], out0[32];
    for (uint L = 0u; L < 32u; ++L) { in0[L] = 0u; out0[L] = 0u; }
    for (uint j = 0u; j < nbVerts0; ++j)
    {
        uint inb = 0u, outb = 0u;
        for (uint L = 0u; L < 3u; ++L) // only the 3 triangle edge-planes matter (& 7 below)
        {
            const float d = dotX(v0[j], e1planeN[L]) - e1planeD[L];
            if (d < 0.0f) inb |= (1u << L);
            if (d > 0.0f) outb |= (1u << L);
        }
        in0[j] = inb; out0[j] = outb;
    }
    for (uint L = 0u; L < 32u; ++L) { in0[L] &= 7u; out0[L] &= 7u; }

    uint flag2 = 0u; // poly0 verts inside all triangle planes
    for (uint L = 0u; L < nbVerts0; ++L) if (out0[L] == 0u) flag2 |= (1u << L);
    flag2 &= (1u << nbVerts0) - 1u;
    for (uint L = 0u; L < 32u; ++L) { wp.pos[L] = v0[L]; wp.sep[L] = s0[L]; }
    addContacts(wp, flag2, maxSep);

    bool anyOut0 = false;
    for (uint L = 0u; L < nbVerts0; ++L) if (out0[L] != 0u) anyOut0 = true;
    if (!anyOut0) { gCount[t] = wp.contactCount; return; }

    // poly0 edge crossings of triangle planes (per poly0-edge lane)
    uint e0cross[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        const uint nxt = ((L + 1u == nbVerts0) ? 0u : (L + 1u)) & 31u; // & 31: match CUDA mod-32 shfl, avoid OOB at lane 31
        e0cross[L] = (in0[L] & out0[nxt]) | (in0[nxt] & out0[L]);
    }

    // ---- phase 3: edge-edge crossings ----
    uint c[32];
    uint a = 0u;
    for (uint L = 0u; L < 32u; ++L) { c[L] = e0cross[L] & e1cross[L]; a |= c[L]; }

    while (a != 0u)
    {
        const uint index = lowestSetIndex(a);   // a triangle-edge plane index (0..2)
        const uint lsb = lowestSetBit(a);
        const float3 m = e1planeN[index];
        const float3 triVert0 = v1[index];
        for (uint L = 0u; L < 32u; ++L)
        {
            const float denom = dotX(m, e0[L]);
            const float lambda = dotX(triVert0 - v0[L], m) / denom;
            const float3 p = v0[L] + lambda * e0[L];
            // 5.10+ (convexTriangle.cuh, 5.12 :1024-1103): t / dnom0 for
            // |dnom0| > 1e-4, else the Gram/Cramer separation in the basis
            // {e1, axis} with the sin^2 < 1e-12 discard -- see pxgSlangPolyClip.
            const float3 dv = p - triVert0;
            if (abs(dnom0) > 1.0e-4f)
            {
                const float d = -dotX(dv, plane1.xyz);
                wp.sep[L] = (denom == 0.0f) ? PC_MAX_F32 : d / dnom0;
            }
            else
            {
                const float3 e1FromShuffle = e1[index];   // shuffle(FULL_MASK, e1, index)
                const float e1MagnSqr = dotX(e1FromShuffle, e1FromShuffle);
                const float mMagnSqr = dotX(m, m);
                const float sepTmp = -((e1MagnSqr * dotX(dv, axis)) - (dotX(axis, e1FromShuffle) * dotX(dv, e1FromShuffle))) / mMagnSqr;
                wp.sep[L] = ((denom == 0.0f) || ((mMagnSqr / e1MagnSqr) < 1.0e-12f)) ? PC_MAX_F32 : sepTmp;
            }
            wp.pos[L] = p;
        }
        uint flag3 = 0u;
        for (uint L = 0u; L < 32u; ++L) if ((c[L] & lsb) != 0u) flag3 |= (1u << L);
        addContacts(wp, flag3, maxSep);
        a &= (a - 1u);
    }

    gCount[t] = wp.contactCount;
}
