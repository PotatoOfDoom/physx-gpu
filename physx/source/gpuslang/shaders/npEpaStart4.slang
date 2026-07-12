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

// npEpaStart4.slang
//
// Slang port of epaStart4 (epa.cuh :254-278) — tranche 130, the first piece of
// the EPA (Expanding Polytope Algorithm) module.  When GJK terminates with a full
// tetrahedron simplex (size 4), EPA seeds its polytope directly from that simplex:
// this routine unpacks the 4 cached vertices + index words into the EpaScratch
// vertex/index arrays and normalises each of the 4 face normals the GJK simplex
// already carries (in lanes 16-31, the setNormal0/123 output) into a unit normal +
// plane distance for the 4 seed triangles, together with the fixed tetrahedron
// topology (sStartVertices4 / sStartAdjacencies4).
//
// Layout: the 32-float GjkCachedData is 4 CachedVertex (v.xyz + asfloat(indices))
// in lanes 0-15 then 4 { n.xyz, d } normals in lanes 16-31.  Lanes 0-15 pack the
// vertex components into S.vertices (compacted index tI - tI/4, skipping the comp-3
// index lanes) and the index words into S.indices[tI>>2].  For triangle tI<4 the
// normal is read from lanes tI*4+16..+18 and scaled by m = 1/sqrt(dot(n,n)) (read
// from the comp-0 lane tI*4+16); distance = n.w * m.
//
// Determinism: dot(n,n) is an exact integer sum of squares and m = 1/sqrt(dot)
// is a correctly-rounded OpFDiv over a correctly-rounded OpSqrt (NOT OpInverseSqrt),
// so with integer simplex data the unit normals and distances are bit-for-bit
// reproducible.  One thread simulates the 32-lane warp with per-lane arrays.

struct EsParams { uint numTests; uint pad0, pad1, pad2; }

// per test outputs
struct EsOut
{
    float verts[12]; // 4 vertices (xyz each)
    uint indices[4];
    float normal[12]; // 4 triangle normals (xyz each)
    float dist[4];
    uint a[4]; uint v[4];
}

[[vk::push_constant]]
ConstantBuffer<EsParams> gParams;

RWStructuredBuffer<float> gData;  // binding 0: GjkCachedData 32 floats/test — in
RWStructuredBuffer<EsOut> gOut;   // binding 1

// sStartVertices4 / sStartAdjacencies4 (MAKE_Vec3I8(A,B,C) = A | B<<8 | C<<16)
static const uint SV4_0 = 0x020100u, SV4_1 = 0x000103u, SV4_2 = 0x010203u, SV4_3 = 0x020003u;
static const uint SA4_0 = 0x010302u, SA4_1 = 0x020300u, SA4_2 = 0x030100u, SA4_3 = 0x010200u;

void dotW(float a[32], float b[32], out float o[32])
{
    float x[32];
    for (uint L = 0u; L < 32u; ++L) x[L] = a[L] * b[L];
    for (uint L = 0u; L < 32u; ++L)
    {
        const float s1 = (L + 1u < 32u) ? x[L + 1u] : x[L];
        const float s2 = (L + 2u < 32u) ? x[L + 2u] : x[L];
        o[L] = x[L] + s1 + s2;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ctEpaStart4(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint base = t * 32u;

    float input[32];
    for (uint L = 0u; L < 32u; ++L) input[L] = gData[base + L];

    EsOut o;

    // unpack vertices (compacted) and index words from lanes 0-15 (CUDA: if(tI<16)).
    // Note (unobservable mutant): widening this bound to 32 is caught only in real
    // CUDA (lanes 16-31 would corrupt the S.vertices/S.indices arrays past their 4
    // entries); in this bounds-checked per-lane model the extra writes land out of
    // range of o.verts/o.indices and are dropped, so the guard is faithful but its
    // removal is unobservable here.
    for (uint L = 0u; L < 16u; ++L)
    {
        if ((L & 3u) < 3u) o.verts[L - L / 4u] = input[L];
        else o.indices[L >> 2] = asuint(input[L]);
    }

    // d = dot(input,input); m = 1/sqrt(d) read at the comp-0 lane of each normal
    float d[32]; dotW(input, input, d);
    float rs[32]; for (uint L = 0u; L < 32u; ++L) rs[L] = 1.0f / sqrt(d[L]);

    for (uint tI = 0u; tI < 4u; ++tI)
    {
        const float m = rs[tI * 4u + 16u];
        o.normal[tI * 3u + 0u] = input[tI * 4u + 16u] * m;
        o.normal[tI * 3u + 1u] = input[tI * 4u + 17u] * m;
        o.normal[tI * 3u + 2u] = input[tI * 4u + 18u] * m;
        o.dist[tI] = input[tI * 4u + 19u] * m;
    }
    o.a[0] = SA4_0; o.a[1] = SA4_1; o.a[2] = SA4_2; o.a[3] = SA4_3;
    o.v[0] = SV4_0; o.v[1] = SV4_1; o.v[2] = SV4_2; o.v[3] = SV4_3;

    gOut[t] = o;
}
