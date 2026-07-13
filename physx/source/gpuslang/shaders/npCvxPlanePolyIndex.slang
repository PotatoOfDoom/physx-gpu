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

// npCvxPlanePolyIndex.slang
//
// Slang port of getPolygonIndexFromPlaneNormal (cudaGJKEPA.cu L2672-2773),
// the face selector of the convex-vs-plane narrowphase: given the plane
// normal n in SHAPE space, pick the hull polygon whose (covariant-transformed,
// normalized) normal best aligns with n, but prefer an EDGE when the edge's
// averaged normal beats the best face by more than the -5e-4 bias (edges
// yield two-face contacts):
//   - face pass: lane L handles polygons L, L+32, ...; planeN =
//     vertex2ShapeNormalVector(plane.n, scale, rot).getNormalized() (=
//     shape2Vertex: rotateInv(rotate(n)/scale)); per-lane best via strict
//     (bestProjection < proj) so the FIRST max in a lane wins ties; warp max
//     with the LOWEST winning lane; index = that lane's candidate (per-lane
//     init index = 0, bestProjection = -PX_MAX_F32);
//   - edge pass: lane L handles edges L, L+32, ...; f0/f1 = facesByEdges8[2i],
//     [2i+1]; edgeN = (loadNormal(f0) + loadNormal(f1)).getNormalized() where
//     loadNormal = shape2Vertex(plane.xyz).getNormalized(); per-lane best via
//     strict (dp > bestEdge); warp max (lowest winning lane) + 5e-4 bias;
//   - if bestEdge > bestProjection: the edge's better-aligned face (dp0 > dp1
//     picks f0, else f1) becomes the index and the other face becomes plane2
//     (caller-initialized to -1, i.e. 0xFFFFFFFF when untouched).
//
// One test per thread; the 32 warp lanes are serialized with exact per-lane
// bests and the established lowest-lane tie-break of warpReduction<MaxOpFloat>.

static const float PX_MAX_F32 = 3.4028235e38f;

float3 quatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
float3 quatRotateInv(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0, 0, 0);
}
// covariant vertex->shape normal transform (= shape2Vertex: divide by scale)
float3 shape2Vertex(float3 v, float3 scale, float4 rot)
{
    return quatRotateInv(rot, quatRotate(rot, v) / scale);
}
float3 loadNormal(float4 p, float3 scale, float4 rot)
{
    return getNormalized(shape2Vertex(p.xyz, scale, rot));
}

struct PlanePolyIn
{
    float4 planes[64];  // hull polygon planes (vertex space)
    uint4 fbe[12];      // facesByEdges8: 2 bytes per edge, byte-packed (max 96 edges)
    float4 scale;       // .xyz; .w = nbPolygons (as float)
    float4 rot;         // mesh-scale rotation quat
    float4 n;           // plane normal in SHAPE space (.xyz); .w = nbEdges (as float)
}

struct PlanePolyOut
{
    uint index;   // selected polygon
    uint plane2;  // second face on the edge path (0xFFFFFFFF when untouched)
    uint usedEdge;
    uint pad0;
}

struct PlanePolyParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<PlanePolyParams> gParams;

RWStructuredBuffer<PlanePolyIn> gIn;    // binding 0
RWStructuredBuffer<PlanePolyOut> gOut;  // binding 1

uint fbeByte(PlanePolyIn s, uint byteIdx)
{
    const uint word = s.fbe[byteIdx / 16u][(byteIdx / 4u) & 3u];
    return (word >> (8u * (byteIdx & 3u))) & 0xFFu;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxPlanePolyIndex(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const PlanePolyIn s = gIn[t];

    const uint nbPolygons = uint(s.scale.w);
    const uint nbEdges = uint(s.n.w);
    const float3 scale = s.scale.xyz;
    const float4 rot = s.rot;
    const float3 n = s.n.xyz;

    // ---- face pass (32 serialized lanes) ----
    float laneBest[32];
    uint laneIdx[32];
    for (uint L = 0u; L < 32u; ++L) { laneBest[L] = -PX_MAX_F32; laneIdx[L] = 0u; }
    for (uint i = 0u; i < nbPolygons; ++i)
    {
        const uint L = i & 31u; // lane i%32 handles polygon i (strided i += 32)
        const float3 planeN = getNormalized(shape2Vertex(s.planes[i].xyz, scale, rot));
        const float proj = dot(planeN, n);
        if (laneBest[L] < proj) { laneIdx[L] = i; laneBest[L] = proj; }
    }
    // warpReduction<MaxOpFloat>: max value, LOWEST lane holding it
    float bestProjection = -PX_MAX_F32;
    for (uint L = 0u; L < 32u; ++L) bestProjection = max(bestProjection, laneBest[L]);
    uint bestLane = 0u;
    for (uint L = 0u; L < 32u; ++L) if (laneBest[L] == bestProjection) { bestLane = L; break; }
    uint index = laneIdx[bestLane];

    // ---- edge pass ----
    float laneEdge[32];
    uint laneEdgeIdx[32];
    for (uint L = 0u; L < 32u; ++L) { laneEdge[L] = -PX_MAX_F32; laneEdgeIdx[L] = 0xFFFFFFFFu; }
    for (uint i = 0u; i < nbEdges; ++i)
    {
        const uint L = i & 31u;
        const uint f0 = fbeByte(s, 2u * i);
        const uint f1 = fbeByte(s, 2u * i + 1u);
        const float3 planeN0 = loadNormal(s.planes[f0], scale, rot);
        const float3 planeN1 = loadNormal(s.planes[f1], scale, rot);
        const float3 edgeN = getNormalized(planeN0 + planeN1);
        const float dp = dot(edgeN, n);
        if (dp > laneEdge[L]) { laneEdge[L] = dp; laneEdgeIdx[L] = i; }
    }
    float bestEdgeMax = -PX_MAX_F32;
    for (uint L = 0u; L < 32u; ++L) bestEdgeMax = max(bestEdgeMax, laneEdge[L]);
    uint bestEdgeLane = 0u;
    for (uint L = 0u; L < 32u; ++L) if (laneEdge[L] == bestEdgeMax) { bestEdgeLane = L; break; }
    const float bestEdge = bestEdgeMax + 5e-4f; // edge-preference bias
    const uint closestEdge = laneEdgeIdx[bestEdgeLane];

    uint plane2 = 0xFFFFFFFFu; // caller initializes to -1
    uint usedEdge = 0u;
    if (bestEdge > bestProjection)
    {
        usedEdge = 1u;
        const uint eIdx = closestEdge * 2u;
        const uint f0 = fbeByte(s, eIdx);
        const uint f1 = fbeByte(s, eIdx + 1u);
        const float3 planeN0 = loadNormal(s.planes[f0], scale, rot);
        const float3 planeN1 = loadNormal(s.planes[f1], scale, rot);
        const float dp0 = dot(planeN0, n);
        const float dp1 = dot(planeN1, n);
        if (dp0 > dp1) { index = f0; plane2 = f1; }
        else           { index = f1; plane2 = f0; }
    }

    PlanePolyOut r;
    r.index = index;
    r.plane2 = plane2;
    r.usedEdge = usedEdge;
    r.pad0 = 0u;
    gOut[t] = r;
}
