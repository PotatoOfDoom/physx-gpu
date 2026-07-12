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

// npEpaStart123.slang
//
// Slang port of epaStart123 (epa.cuh :330-385) — tranche 131.  When GJK ends on
// a lower-dimensional simplex (size 1/2/3), EPA cannot seed a tetrahedron; this
// routine builds a double-sided triangle polytope instead.  It unpacks the cached
// vertices/indices, then fills the triangle out with support points as needed:
//   * size<2: newPoint(dir=(1,0,0), ref=v0) -> vertex1
//   * size<3: newPoint(dir = axis x (v1-v0), ref=v0) -> vertex2   (axis = the
//             coordinate axis less aligned with (v1-v0))
// then forms the face normal from the better-conditioned pair of edges
//   normal = (|e01| <= |e02| ? e01 x e12 : e02 x e12).getNormalized();  d = n.v0
// and emits the two-triangle topology (sStartVertices3 / sStartAdjacency3), with
// the second (lane-1) triangle carrying the negated normal/distance (double-sided).
//
// newPoint (epa.cuh :306-324, ported in tranche 117) picks, of the two support
// points furthest along +/-dir, whichever is further from the reference along dir.
// It reuses the epa supportIndex argmax (a single vec3 dir, tranche 116).
//
// Determinism: with integer clouds and integer cached vertices every dot/cross is
// exact, magnitude/getNormalized use the correctly-rounded sqrt + 1/sqrt forms,
// and the support argmax is exact, so the shader and the CPU reference agree
// bit-for-bit.  One thread simulates the 32-lane warp with per-lane arrays.

struct Es3Params { uint numTests; uint maxV; uint pad0, pad1; }
struct Es3Scalars { uint size; uint nbA; uint nbB; uint pad; }
struct Es3Out
{
    float verts[9]; uint indices[3];
    float normal0[3]; float dist0;
    float normal1[3]; float dist1;
    uint v0; uint v1; uint a0; uint a1;
}

[[vk::push_constant]]
ConstantBuffer<Es3Params> gParams;

RWStructuredBuffer<float> gData;    // binding 0: GjkCachedData 32 floats/test
RWStructuredBuffer<float4> gVA;     // binding 1: convex A cloud (maxV/test)
RWStructuredBuffer<float4> gVB;     // binding 2: convex B cloud (maxV/test)
RWStructuredBuffer<Es3Scalars> gS;  // binding 3: uniforms (per test)
RWStructuredBuffer<Es3Out> gOut;    // binding 4

static const float ES_MAX_F32 = 3.402823466e38f;

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 cross3(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
float3 getNormalized(float3 v)
{
    const float m = dot3(v, v);
    if (m > 0.0f) { const float r = 1.0f / sqrt(m); return float3(v.x * r, v.y * r, v.z * r); }
    return float3(0.0f, 0.0f, 0.0f);
}
float magnitude3(float3 v) { return sqrt(dot3(v, v)); }

uint supportIdxA(float3 dir, uint vbase, uint nb)
{
    uint bestN = 0xFFFFFFFFu; float bestDist = -ES_MAX_F32;
    for (uint i = 0u; i < nb; i += 32u)
    {
        float d[32];
        for (uint lane = 0u; lane < 32u; ++lane) { const uint n = i + lane; d[lane] = (n < nb) ? dot3(dir, gVA[vbase + n].xyz) : -ES_MAX_F32; }
        float cmax = -ES_MAX_F32; for (uint lane = 0u; lane < 32u; ++lane) cmax = max(cmax, d[lane]);
        if (cmax > bestDist) { uint ll = 0u; for (uint lane = 0u; lane < 32u; ++lane) if (d[lane] == cmax) { ll = lane; break; } bestDist = cmax; bestN = i + ll; }
    }
    return bestN;
}
uint supportIdxB(float3 dir, uint vbase, uint nb)
{
    uint bestN = 0xFFFFFFFFu; float bestDist = -ES_MAX_F32;
    for (uint i = 0u; i < nb; i += 32u)
    {
        float d[32];
        for (uint lane = 0u; lane < 32u; ++lane) { const uint n = i + lane; d[lane] = (n < nb) ? dot3(dir, gVB[vbase + n].xyz) : -ES_MAX_F32; }
        float cmax = -ES_MAX_F32; for (uint lane = 0u; lane < 32u; ++lane) cmax = max(cmax, d[lane]);
        if (cmax > bestDist) { uint ll = 0u; for (uint lane = 0u; lane < 32u; ++lane) if (d[lane] == cmax) { ll = lane; break; } bestDist = cmax; bestN = i + ll; }
    }
    return bestN;
}

// newPoint: furthest support along +/-dir from ref, returns vertex + packed indices
void newPoint(float3 dir, float3 ref, uint vbase, uint nbA, uint nbB, out float3 vertex, out uint indices)
{
    const uint indexAp = supportIdxA(dir, vbase, nbA), indexBp = supportIdxB(float3(-dir.x, -dir.y, -dir.z), vbase, nbB);
    const uint indexAn = supportIdxA(float3(-dir.x, -dir.y, -dir.z), vbase, nbA), indexBn = supportIdxB(dir, vbase, nbB);
    const float3 vp = gVA[vbase + indexAp].xyz - gVB[vbase + indexBp].xyz;
    const float3 vn = gVA[vbase + indexAn].xyz - gVB[vbase + indexBn].xyz;
    if (abs(dot3(vp - ref, dir)) > abs(dot3(vn - ref, dir))) { vertex = vp; indices = (indexBp << 8) | (indexAp & 0xffu); }
    else { vertex = vn; indices = (indexBn << 8) | (indexAn & 0xffu); }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ctEpaStart123(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint base = t * 32u;
    const uint vbase = t * gParams.maxV;
    const Es3Scalars s = gS[t];

    float input[32];
    for (uint L = 0u; L < 32u; ++L) input[L] = gData[base + L];

    // unpack cached vertices (compacted) + index words from lanes 0-15
    float verts[12]; uint indices[4];
    for (uint L = 0u; L < 16u; ++L)
    {
        if ((L & 3u) < 3u) verts[L - L / 4u] = input[L];
        else indices[L >> 2] = asuint(input[L]) & 0xffffu; // CUDA S.indices is PxU16 -> 16-bit truncation
    }

    float3 vertex0 = float3(verts[0], verts[1], verts[2]);
    float3 vertex1 = float3(verts[3], verts[4], verts[5]);
    if (s.size < 2u)
    {
        float3 nv; uint ni; newPoint(float3(1.0f, 0.0f, 0.0f), vertex0, vbase, s.nbA, s.nbB, nv, ni);
        vertex1 = nv; verts[3] = nv.x; verts[4] = nv.y; verts[5] = nv.z; indices[1] = ni;
    }

    float3 vertex2 = float3(verts[6], verts[7], verts[8]);
    if (s.size < 3u)
    {
        const float3 dir = vertex1 - vertex0;
        const float3 axis = (abs(dir.x) < abs(dir.y)) ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
        const float3 sdir = cross3(axis, dir);
        float3 nv; uint ni; newPoint(sdir, vertex0, vbase, s.nbA, s.nbB, nv, ni);
        vertex2 = nv; verts[6] = nv.x; verts[7] = nv.y; verts[8] = nv.z; indices[2] = ni;
    }

    // Note (equivalent boundary): e02 == e01 + e12, and cross(e12,e12)==0 exactly for
    // integer coords, so cross(e01,e12) == cross(e02,e12) == cross(e01,e02) bit-for-bit
    // -- the triangle normal is the same regardless of which edge pair is crossed.  The
    // |e01|<=|e02| test therefore only selects the better-conditioned edge for float
    // stability; at the |e01|==|e02| boundary either branch yields the identical normal.
    // The `<=` and the specific (e01,e12) / (e02,e12) pairing are retained verbatim,
    // faithful to the CUDA (unobservable-under-integer mutants documented in the test).
    const float3 e01 = vertex1 - vertex0, e12 = vertex2 - vertex1, e02 = vertex2 - vertex0;
    float3 normal;
    if (magnitude3(e01) <= magnitude3(e02)) normal = getNormalized(cross3(e01, e12));
    else                                    normal = getNormalized(cross3(e02, e12));
    const float distance = dot3(normal, vertex0);

    Es3Out o;
    for (uint i = 0u; i < 9u; ++i) o.verts[i] = verts[i];
    o.indices[0] = indices[0]; o.indices[1] = indices[1]; o.indices[2] = indices[2];
    // lane 0: v = sStartVertices3[0], a = sStartAdjacency3[0], normal/distance as-is
    o.normal0[0] = normal.x; o.normal0[1] = normal.y; o.normal0[2] = normal.z; o.dist0 = distance;
    // lane 1 (tI==1): normal = -normal, distance = -distance
    o.normal1[0] = -normal.x; o.normal1[1] = -normal.y; o.normal1[2] = -normal.z; o.dist1 = -distance;
    o.v0 = 0x020100u; o.v1 = 0x020001u; // sStartVertices3 = {(0,1,2),(1,0,2)}
    o.a0 = 0x010101u; o.a1 = 0x000000u; // sStartAdjacency3 = {(1,1,1),(0,0,0)}
    gOut[t] = o;
}
