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

// npConvexTri.slang
//
// Slang port of the convex-triangle edge-edge separating-axis primitives from
// gpunarrowphase/src/CUDA/convexTriangle.cuh: isMinkowskiFace (the Gauss-map
// arc-intersection test that culls edge pairs which cannot build a face of the
// Minkowski sum) and isEdgePairPassed (the edge-edge SAT test built on it — the
// edge-cross axis, oriented convex->triangle, projected to a separation which
// updates the running max).

struct MkIn
{
    float4 A, B, BxA;   // edge 0: two adjacent normals + the edge (arc)
    float4 C, D, DxC;   // edge 1: two adjacent normals + the edge (arc)
}

// per-test inputs for isEdgePairPassed
struct EpIn
{
    float4 convexNormal0, convexNormal1, convexEdge, convexVertex;
    float4 triNormal0, triEdge, triVertex, convexCenterOfMass;
    float4 misc;          // x = contactDist, y = seed separation
    float4 seedFaceNormal; // xyz
}

struct EpOut
{
    float4 faceNormal;   // xyz
    float4 sr;           // x = separation, y = returned bool (0/1)
}

struct CtParams { uint numTests; uint pad0, pad1, pad2; }

[[vk::push_constant]]
ConstantBuffer<CtParams> gParams;

RWStructuredBuffer<MkIn> gMk;      // binding 0
RWStructuredBuffer<uint> gMkOut;   // binding 1  (0/1)
RWStructuredBuffer<EpIn> gEp;      // binding 2
RWStructuredBuffer<EpOut> gEpOut;  // binding 3

// isMinkowskiFace: two edges build a face of the Minkowski sum iff their arcs
// AB and CD intersect on the Gauss map.
bool isMinkowskiFace(float3 A, float3 B, float3 B_x_A, float3 C, float3 D, float3 D_x_C)
{
    const float CBA = dot(C, B_x_A);
    const float DBA = dot(D, B_x_A);
    const float ADC = dot(A, D_x_C);
    const float BDC = dot(B, D_x_C);
    // intersection test CBA*DBA < 0 && ADC*BDC < 0; hemisphere test CBA*BDC > 0
    return CBA * DBA < 0.0f && ADC * BDC < 0.0f && CBA * BDC > 0.0f;
}

// isEdgePairPassed: returns false only when this edge pair proves a separation
// beyond contactDist; otherwise returns true and, when the edge-cross axis beats
// the running separation, updates separation and faceNormal.
bool isEdgePairPassed(float3 convexNormal0, float3 convexNormal1, float3 convexEdge, float3 convexVertex,
                      float3 triNormal0, float3 triEdge, float3 triVertex, float3 convexCenterOfMass,
                      float contactDist, inout float separation, inout float3 faceNormal)
{
    if (!isMinkowskiFace(convexNormal0, convexNormal1, convexEdge, -triNormal0, triNormal0, triEdge))
        return true;

    float3 normal = cross(convexEdge, triEdge);

    // edge-parallel case (any near-zero cross component)
    const float eps = 1e-6f;
    if (abs(normal.x) < eps || abs(normal.y) < eps || abs(normal.z) < eps)
        return true;

    normal = normalize(normal);

    if (dot(normal, convexVertex - convexCenterOfMass) < 0.0f)
        normal = -normal; // point the normal from the convex hull toward the triangle

    const float dist = dot(normal, triVertex - convexVertex);

    if (dist > contactDist)
        return false;

    if (separation < dist)
    {
        separation = dist;
        faceNormal = normal;
    }

    return true;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctMinkowskiFace(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const MkIn m = gMk[t];
    const bool r = isMinkowskiFace(m.A.xyz, m.B.xyz, m.BxA.xyz, m.C.xyz, m.D.xyz, m.DxC.xyz);
    gMkOut[t] = r ? 1u : 0u;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctEdgePairPassed(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const EpIn e = gEp[t];

    float separation = e.misc.y;
    float3 faceNormal = e.seedFaceNormal.xyz;
    const bool r = isEdgePairPassed(e.convexNormal0.xyz, e.convexNormal1.xyz, e.convexEdge.xyz, e.convexVertex.xyz,
                                    e.triNormal0.xyz, e.triEdge.xyz, e.triVertex.xyz, e.convexCenterOfMass.xyz,
                                    e.misc.x, separation, faceNormal);

    EpOut o;
    o.faceNormal = float4(faceNormal, 0.0f);
    o.sr = float4(separation, r ? 1.0f : 0.0f, 0.0f, 0.0f);
    gEpOut[t] = o;
}
