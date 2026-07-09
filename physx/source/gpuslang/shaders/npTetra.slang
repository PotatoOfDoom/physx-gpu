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

// npTetra.slang
//
// Slang port of three deformable-tetrahedron primitives from
// gpunarrowphase/src/CUDA/deformableCollision.cuh: triBoundingBox and
// tetBoundingBox (the axis-aligned bounds of a triangle / tetrahedron, a
// componentwise min/max over the vertices) and computeTetBarycentric (the
// Cramer's-rule barycentric coordinates of a point relative to a tetrahedron,
// clamped to [0,1], returning whether the point lies inside up to an eps
// tolerance).  The bounds are exact (min/max select an input value); the
// barycentric coordinates are continuous, and the inside/outside boolean is a
// discrete decision gated on float comparisons at -eps / 1+eps.

struct TetParams { uint numTests; uint pad0, pad1, pad2; }

struct TriB { float4 v0, v1, v2; }
struct TetB { float4 v0, v1, v2, v3; }
struct BaryB { float4 v0, v1, v2, v3, p; }
struct BBox { float4 mn, mx; }
struct BaryO { float4 bary; float4 extra; } // extra.x = inside (1.0 / 0.0)

[[vk::push_constant]]
ConstantBuffer<TetParams> gParams;

RWStructuredBuffer<TriB> gTri;       // binding 0
RWStructuredBuffer<BBox> gTriBB;     // binding 1
RWStructuredBuffer<TetB> gTet;       // binding 2
RWStructuredBuffer<BBox> gTetBB;     // binding 3
RWStructuredBuffer<BaryB> gBary;     // binding 4
RWStructuredBuffer<BaryO> gBaryOut;  // binding 5

void triBoundingBox(float3 v0, float3 v1, float3 v2, out float3 mn, out float3 mx)
{
    mn = min(min(v0, v1), v2);
    mx = max(max(v0, v1), v2);
}

void tetBoundingBox(float3 v0, float3 v1, float3 v2, float3 v3, out float3 mn, out float3 mx)
{
    mn = min(min(v0, v1), min(v2, v3));
    mx = max(max(v0, v1), max(v2, v3));
}

// computeTetBarycentric: barycentric (u,v,w,x) of point vs tetrahedron abcd via
// Cramer's rule; the returned coords are clamped to [0,1], and the boolean is
// whether the unclamped coords all lie in [-eps, 1+eps].
bool computeTetBarycentric(float3 a, float3 b, float3 c, float3 d, float3 point, out float4 barycentric, float eps)
{
    const float minEps = -eps;
    const float maxEps = 1.0f + eps;
    const float3 ba = b - a;
    const float3 ca = c - a;
    const float3 da = d - a;
    const float3 pa = point - a;

    const float detBcd = dot(ba, cross(ca, da));
    const float detPcd = dot(pa, cross(ca, da));
    const float v = detPcd / detBcd;
    const float detBpd = dot(ba, cross(pa, da));
    const float w = detBpd / detBcd;
    const float detBcp = dot(ba, cross(ca, pa));
    const float x = detBcp / detBcd;

    const float u = 1.0f - v - w - x;
    barycentric = float4(u, v, w, x);

    barycentric.x = clamp(barycentric.x, 0.0f, 1.0f);
    barycentric.y = clamp(barycentric.y, 0.0f, 1.0f);
    barycentric.z = clamp(barycentric.z, 0.0f, 1.0f);
    barycentric.w = clamp(barycentric.w, 0.0f, 1.0f);

    return (u >= minEps && u <= maxEps) && (v >= minEps && v <= maxEps) && (w >= minEps && w <= maxEps) && (x >= minEps && x <= maxEps);
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctTriBBox(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const TriB e = gTri[i];
    float3 mn, mx;
    triBoundingBox(e.v0.xyz, e.v1.xyz, e.v2.xyz, mn, mx);
    BBox o;
    o.mn = float4(mn, 0.0f);
    o.mx = float4(mx, 0.0f);
    gTriBB[i] = o;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctTetBBox(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const TetB e = gTet[i];
    float3 mn, mx;
    tetBoundingBox(e.v0.xyz, e.v1.xyz, e.v2.xyz, e.v3.xyz, mn, mx);
    BBox o;
    o.mn = float4(mn, 0.0f);
    o.mx = float4(mx, 0.0f);
    gTetBB[i] = o;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctTetBary(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const BaryB e = gBary[i];
    float4 bary;
    const bool inside = computeTetBarycentric(e.v0.xyz, e.v1.xyz, e.v2.xyz, e.v3.xyz, e.p.xyz, bary, 1e-5f);
    BaryO o;
    o.bary = bary;
    o.extra = float4(inside ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);
    gBaryOut[i] = o;
}
