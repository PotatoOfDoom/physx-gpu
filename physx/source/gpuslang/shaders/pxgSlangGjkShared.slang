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

// pxgSlangGjkShared -- the scalar helpers of squawk::gjk / squawk::epa (gjk.cuh,
// epa.cuh, schlock), shared by the SERIAL drivers (pxgSlangGjk, pxgSlangEpa: one
// thread simulates the warp) and the WARP-COOPERATIVE one (pxgSlangGjkWarp: 32
// threads, one CUDA lane each).
//
// The bodies are MOVED verbatim out of pxgSlangGjk and pxgSlangEpa, where each was a
// module-private copy, so that the cooperative driver compiles the same helpers the
// serial one does instead of a third copy.  crossX (pxgSlangGjk's name) and cross3
// (pxgSlangEpa's) are the same cross product under the two names their call sites
// already used; both are kept so that no call site's text had to change.
module pxgSlangGjkShared;

// ---- warp layout (gjk.cuh) -------------------------------------------------
public uint next3(uint x) { return x == 2u ? 0u : x + 1u; }
public uint prev3(uint x) { return x == 0u ? 2u : x - 1u; }
public uint ffs(uint m) { return m == 0u ? 0u : firstbitlow(m) + 1u; }
public uint encodeIndices(uint a, uint b) { return 0x3f800000u | ((b & 0xffu) << 8) | (a & 0xffu); }

// ---- PxVec3 arithmetic, spelled out so the association is fixed --------------
public float3 crossX(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
public float3 cross3(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
public float3 absX(float3 a) { return float3(abs(a.x), abs(a.y), abs(a.z)); }
public float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
public float magSq3(float3 v) { return dot3(v, v); }
public float mag3(float3 v) { return sqrt(dot3(v, v)); }
public float3 getNormalized(float3 v)
{
    const float m = dot3(v, v);
    if (m > 0.0f) { const float r = 1.0f / sqrt(m); return float3(v.x * r, v.y * r, v.z * r); }
    return float3(0.0f, 0.0f, 0.0f);
}

// ---- schlock::Vec3I8 -----------------------------------------------------------
public uint mk3(uint a, uint b, uint c) { return (a & 0xffu) | ((b & 0xffu) << 8) | ((c & 0xffu) << 16); }
public uint g3(uint p, uint i) { return (p >> (i * 8u)) & 0xffu; }
public uint s3(uint p, uint i, uint val) { return (p & ~(0xffu << (i * 8u))) | ((val & 0xffu) << (i * 8u)); }

// ---- schlock::barycentrics -------------------------------------------------------
public void barycentrics(float3 a, float3 b, float3 c, float3 p, out float u, out float v, out float w)
{
    const float3 m = cross3(b - a, c - a);
    const float3 ma = absX(m);
    const float3 a1 = cross3(b - p, c - p);
    const float3 a2 = cross3(c - p, a - p);
    float d;
    if (ma.x >= ma.y && ma.x >= ma.z) { d = 1.0f / m.x; u = a1.x * d; v = a2.x * d; }
    else if (ma.y >= ma.z)            { d = 1.0f / m.y; u = a1.y * d; v = a2.y * d; }
    else                              { d = 1.0f / m.z; u = a1.z * d; v = a2.z * d; }
    w = 1.0f - u - v;
}

// ---- epaStart4's seed polytope (epa.cuh sStartVertices4 / sStartAdjacencies4) -----
public static const uint SV4_0 = 0x020100u, SV4_1 = 0x000103u, SV4_2 = 0x010203u, SV4_3 = 0x020003u;
public static const uint SA4_0 = 0x010302u, SA4_1 = 0x020300u, SA4_2 = 0x030100u, SA4_3 = 0x010200u;
