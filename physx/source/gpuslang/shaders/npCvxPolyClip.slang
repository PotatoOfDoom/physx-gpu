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

// npCvxPolyClip.slang
//
// Slang port of the convex-vs-convex polyClip() from cudaGJKEPA.cu (L113-214) --
// the polygon clipper used by generateConvexContacts (distinct from the
// convex-vs-triangle clippers polyClip4 / npConvexPolyClip).  Warp-SIMD, one lane
// per polygon vertex; this port runs one thread per problem and serialises the
// 32-lane warp over per-lane arrays, replicating every ballot / shfl_xor / shuffle
// reduction bit-for-bit.
//
// Algorithm (Sutherland-Hodgman style, via in/out bit-masks):
//   1. project poly planes onto the clip axis (s0, s1);
//   2. poly1 points vs poly0 edge-planes -> in1/out1 masks; emit poly1 points
//      inside poly0 (AND-reduction of ~out1);
//   3. poly0 points vs poly1 edge-planes -> in0/out0 masks (ballot); emit poly0
//      points inside poly1;
//   4. edge-edge crossings (e0cross & e1cross): lambda intersection along e0;
//      mDote0==0 -> sep=PX_MAX_F32 (discarded by addContacts).

static const float PXG_MAX_F32 = 3.4028235e38f;
static const uint NUM_TMP = 32u;

struct PolyClipIn
{
    uint baseV;   // vertex-buffer base (float4 units): 32 v0 then 32 v1
    uint nbV0, nbV1, pad0;
    float4 plane0;   // n.xyz, d
    float4 plane1;   // n.xyz, d
    float4 axisCD;   // axis.xyz, .w = contactDist
}

struct PolyClipOut
{
    float4 cts[32];  // .xyz = pos, .w = sep  (first nbContacts valid)
    uint nbContacts;
    uint pad0, pad1, pad2;
}

struct PolyClipParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<PolyClipParams> gParams;

RWStructuredBuffer<PolyClipIn> gIn;    // binding 0
RWStructuredBuffer<float4>     gVerts; // binding 1 : 64 float4 per test (v0[32], v1[32])
RWStructuredBuffer<PolyClipOut> gOut;  // binding 2

uint popcnt(uint m) { return countbits(m); }
uint lowestSetIndex(uint m) { return firstbitlow(m); }

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxPolyClip(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const PolyClipIn s = gIn[t];
    const uint nbV0 = s.nbV0, nbV1 = s.nbV1;
    const float3 p0n = s.plane0.xyz; const float p0d = s.plane0.w;
    const float3 p1n = s.plane1.xyz; const float p1d = s.plane1.w;
    const float3 axis = s.axisCD.xyz; const float contactDist = s.axisCD.w;
    const uint bv = s.baseV;

    float3 v0[32], v1[32];
    for (uint L = 0u; L < 32u; ++L) { v0[L] = gVerts[bv + L].xyz; v1[L] = gVerts[bv + 32u + L].xyz; }

    // contacts accumulator
    float3 ctsPos[32]; float ctsSep[32];
    for (uint i = 0u; i < 32u; ++i) { ctsPos[i] = float3(0, 0, 0); ctsSep[i] = 0.0f; }
    uint nbContacts = 0u;

    // s0, s1 (per lane)
    const float p1DotAxis = dot(p1n, axis);
    float s0[32], s1[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        s0[L] = -(dot(p1n, v0[L]) + p1d) / p1DotAxis;
        s1[L] = dot(p0n, v1[L]) + p0d;
    }

    // e0, e0planeN, e0planeD (per lane = poly0 edge L)
    float3 e0[32], e0planeN[32]; float e0planeD[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        const uint nxt = (L + 1u == nbV0) ? 0u : ((L + 1u) & 31u);
        e0[L] = v0[nxt] - v0[L];
        e0planeN[L] = cross(e0[L], axis);
        e0planeD[L] = dot(e0planeN[L], v0[L]);
    }

    // poly1 points vs poly0 edge-planes -> in1/out1
    uint in1[32], out1[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        uint i1 = 0u, o1 = 0u;
        for (uint j = 0u; j < nbV1; ++j)
        {
            const float d = dot(v1[j], e0planeN[L]);
            const float tt = d - e0planeD[L];
            const uint bj = (1u << j);
            if (tt < 0.0f) i1 |= bj;
            if (tt > 0.0f) o1 |= bj;
        }
        in1[L] = i1; out1[L] = o1;
    }
    for (uint L = 0u; L < 32u; ++L) if (L >= nbV0) { in1[L] = 0u; out1[L] = 0u; }

    // b = AND over all lanes of (~out1)
    uint b = 0xffffffffu;
    for (uint L = 0u; L < 32u; ++L) b &= ~out1[L];

    // emit poly1 points inside poly0
    {
        const uint mask = b & ((nbV1 >= 32u) ? 0xffffffffu : ((1u << nbV1) - 1u));
        // addContacts(mask, v1[L]-s1[L]*axis, s1[L], contactDist)
        uint fmask = 0u;
        for (uint L = 0u; L < 32u; ++L) if ((mask & (1u << L)) != 0u && s1[L] < contactDist) fmask |= (1u << L);
        for (uint L = 0u; L < 32u; ++L)
            if ((fmask & (1u << L)) != 0u)
            {
                const uint idx = popcnt(fmask & ((1u << L) - 1u)) + nbContacts;
                if (idx < NUM_TMP) { ctsPos[idx] = v1[L] - s1[L] * axis; ctsSep[idx] = s1[L]; }
            }
        nbContacts += popcnt(fmask);
    }

    // any(out1)?
    uint anyOut1 = 0u;
    for (uint L = 0u; L < 32u; ++L) anyOut1 |= out1[L];
    if (anyOut1 == 0u)
    {
        PolyClipOut r;
        for (uint i = 0u; i < 32u; ++i) r.cts[i] = float4(ctsPos[i], ctsSep[i]);
        r.nbContacts = nbContacts; r.pad0 = 0u; r.pad1 = 0u; r.pad2 = 0u;
        gOut[t] = r; return;
    }

    // save poly1's edge crosses
    uint e1cross[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        const uint i1 = in1[L] | (in1[L] << nbV1);
        const uint o1 = out1[L] | (out1[L] << nbV1);
        e1cross[L] = (i1 & (o1 >> 1)) | ((i1 >> 1) & o1);
    }

    // e1, e1planeN, e1planeD (poly1 edge L)
    float3 e1[32], e1planeN[32]; float e1planeD[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        const uint nxt = (L + 1u == nbV1) ? 0u : ((L + 1u) & 31u);
        e1[L] = v1[nxt] - v1[L];
        e1planeN[L] = cross(axis, e1[L]);
        e1planeD[L] = dot(e1planeN[L], v1[L]);
    }

    // poly0 points vs poly1 edge-planes -> in0/out0 (ballot per j)
    uint in0[32], out0[32];
    for (uint L = 0u; L < 32u; ++L) { in0[L] = 0u; out0[L] = 0u; }
    for (uint j = 0u; j < nbV0; ++j)
    {
        uint inMask = 0u, outMask = 0u;
        for (uint L = 0u; L < 32u; ++L)
        {
            const float tt = dot(v0[j], e1planeN[L]) - e1planeD[L];
            if (tt < 0.0f) inMask |= (1u << L);
            if (tt > 0.0f) outMask |= (1u << L);
        }
        in0[j] = inMask; out0[j] = outMask;
    }

    const uint mask1 = (nbV1 >= 32u) ? 0xffffffffu : ((1u << nbV1) - 1u);
    for (uint L = 0u; L < 32u; ++L) { in0[L] &= mask1; out0[L] &= mask1; }

    // emit poly0 points inside poly1
    {
        uint ballotNotOut0 = 0u;
        for (uint L = 0u; L < 32u; ++L) if (out0[L] == 0u) ballotNotOut0 |= (1u << L);
        const uint mask = ballotNotOut0 & ((nbV0 >= 32u) ? 0xffffffffu : ((1u << nbV0) - 1u));
        uint fmask = 0u;
        for (uint L = 0u; L < 32u; ++L) if ((mask & (1u << L)) != 0u && s0[L] < contactDist) fmask |= (1u << L);
        for (uint L = 0u; L < 32u; ++L)
            if ((fmask & (1u << L)) != 0u)
            {
                const uint idx = popcnt(fmask & ((1u << L) - 1u)) + nbContacts;
                if (idx < NUM_TMP) { ctsPos[idx] = v0[L]; ctsSep[idx] = s0[L]; }
            }
        nbContacts += popcnt(fmask);
    }

    uint anyOut0 = 0u;
    for (uint L = 0u; L < 32u; ++L) anyOut0 |= out0[L];
    if (anyOut0 == 0u)
    {
        PolyClipOut r;
        for (uint i = 0u; i < 32u; ++i) r.cts[i] = float4(ctsPos[i], ctsSep[i]);
        r.nbContacts = nbContacts; r.pad0 = 0u; r.pad1 = 0u; r.pad2 = 0u;
        gOut[t] = r; return;
    }

    // poly0's edge crosses
    uint e0cross[32]; float s0d[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        const uint nxt = (L + 1u == nbV0) ? 0u : ((L + 1u) & 31u);
        const uint in0p = in0[nxt], out0p = out0[nxt];
        e0cross[L] = (in0[L] & out0p) | (in0p & out0[L]);
        s0d[L] = s0[nxt] - s0[L];
    }

    // c = e0cross & e1cross ; a = OR over lanes of c
    uint c[32]; uint a = 0u;
    for (uint L = 0u; L < 32u; ++L) { c[L] = e0cross[L] & e1cross[L]; a |= c[L]; }

    // edge-edge crossing loop
    for (; a != 0u; a &= (a - 1u))
    {
        const uint index = lowestSetIndex(a);
        const float3 m = e1planeN[index];
        const uint lsb = a & (uint)(-(int)a); // lowest set bit value
        // per-lane point/sep
        float3 pt[32]; float sepA[32];
        for (uint L = 0u; L < 32u; ++L)
        {
            const float mDote0 = dot(m, e0[L]);
            const float lambda = -(dot(m, v0[L]) - e1planeD[index]) / mDote0;
            pt[L] = v0[L] + lambda * e0[L];
            sepA[L] = (mDote0 != 0.0f) ? (s0[L] + lambda * s0d[L]) : PXG_MAX_F32;
        }
        uint flags = 0u;
        for (uint L = 0u; L < 32u; ++L) if ((c[L] & lsb) != 0u) flags |= (1u << L);
        // addContacts(flags, pt, sepA, contactDist)
        uint fmask = 0u;
        for (uint L = 0u; L < 32u; ++L) if ((flags & (1u << L)) != 0u && sepA[L] < contactDist) fmask |= (1u << L);
        for (uint L = 0u; L < 32u; ++L)
            if ((fmask & (1u << L)) != 0u)
            {
                const uint idx = popcnt(fmask & ((1u << L) - 1u)) + nbContacts;
                if (idx < NUM_TMP) { ctsPos[idx] = pt[L]; ctsSep[idx] = sepA[L]; }
            }
        nbContacts += popcnt(fmask);
    }

    PolyClipOut r;
    for (uint i = 0u; i < 32u; ++i) r.cts[i] = float4(ctsPos[i], ctsSep[i]);
    r.nbContacts = nbContacts; r.pad0 = 0u; r.pad1 = 0u; r.pad2 = 0u;
    gOut[t] = r;
}
