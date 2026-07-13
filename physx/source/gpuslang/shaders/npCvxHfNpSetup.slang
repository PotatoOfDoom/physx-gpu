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

// npCvxHfNpSetup.slang
//
// Slang port of the convexHeightfieldNarrowphaseCore setup stage
// (convexHeightfield.cu L120-269: buildTriangleInformations + the core's own
// prep, everything before the hand-off into the tranche-96/110/112
// convexTriangleContactGen pipeline).  One warp per pair, serialized:
//   - trimeshToConvexTransform = convexT.transformInv(heightfieldT);
//   - getTriangle (cu L83-116, the tranche-60 semantics): wrongHanded =
//     (columnScale < 0) XOR (rowScale < 0) (the handedness vector is dead),
//     vertex indices via the tranche-59 helper with the out-argument
//     permutation [0, 1+wh, 2-wh], adjacency via [wh?2:0, 1, wh?0:2], and
//     verts[vi] = hf2shapep(getVertex(idx), rowScale, heightScale,
//     columnScale) -- a pure per-component multiply, heights sign-extended
//     from the low 16 sample bits;
//   - triNormal = denormalizedNormal = (v1-v0) x (v2-v0) over the PERMUTED
//     verts; triangleLocNormal = t2c.rotate(triNormal).getNormalized()
//     (rotate ONLY, no translation, then normalize);
//   - triLocVerts[vi] = t2c.transform(verts[vi]);
//   - contactDist sum, convex CoM through vertex2Shape with the CONVEX
//     scale, the packed counts word, and the startIndex/roundedStartIndex
//     pair offsets (identical to the tranche-160 trimesh sibling; heightfield
//     pairs have no faceRemap -- the contact gen gets triangleIdx twice).

struct HfsIn
{
    float4 cq, cp;        // convex cached transform
    float4 tq, tp;        // heightfield cached transform
    float4 cScaleS, cScaleR; // convex PxMeshScale
    float4 hfScale;       // heightfield scale: x = rowScale, y = heightScale, z = columnScale
    float4 dists;         // x/y = contactDistance[ref0/ref1]
    uint4 pairInfo;       // x = triangleIdx, y = startIndex, z = roundedStartIndex, w = testOffset
    float4 cvxBlob[2];    // convex blob head: CoM float4, header uint4 (bit-cast)
    uint4 hf[17];         // heightfield words: [0]=rows, [1]=columns, samples from word 2 (max 66 words)
}

struct HfsOut
{
    float4 t2cq, t2cp;    // trimeshToConvexTransform; p.w = contactDist
    float4 com;           // shape-space CoM; w = header word (bit-cast)
    uint4 adj;            // xyz = triAdjTrisIdx (permuted), w = wrongHanded
    float4 locVerts[3];   // triLocVerts (w pad)
    float4 locNormal;     // triangleLocNormal (w pad)
    uint4 offsets;        // x = convexTriPairOffset, y = padded, z/w pad
}

struct HfsParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<HfsParams> gParams;

RWStructuredBuffer<HfsIn> gIn;    // binding 0
RWStructuredBuffer<HfsOut> gOut;  // binding 1

static const uint HF_BOUNDARY = 0xFFFFFFFFu;

float4 quatConjQ(float4 q) { return float4(-q.x, -q.y, -q.z, q.w); }
float4 quatMulQ(float4 a, float4 b)
{
    return float4(a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y,
                  a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z,
                  a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x,
                  a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z); // sequential like PxQuat.h:390
}
float3 quatRotQ(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
float3 quatRotInvQ(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
float3 vertex2ShapeQ(float3 v, float3 sc, float4 rot)
{
    const float3 r = quatRotQ(rot, v);
    return quatRotInvQ(rot, float3(r.x * sc.x, r.y * sc.y, r.z * sc.z));
}
float3 crossQ(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
float3 getNormalizedQ(float3 v)
{
    const float m = v.x * v.x + v.y * v.y + v.z * v.z;
    return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0, 0, 0);
}

uint hfWord(HfsIn s, uint idx) { return s.hf[idx / 4u][idx & 3u]; }

// ---- tranche-59 heightfield helpers (verbatim semantics) --------------------
bool hfShared(HfsIn s, uint vertexIndex)
{
    return (hfWord(s, 2u + vertexIndex) & 0x800000u) != 0u;
}
float hfHeight(HfsIn s, uint vertexIndex)
{
    const int h = int(hfWord(s, 2u + vertexIndex) << 16u) >> 16u; // sign-extended low 16 bits
    return float(h);
}
float3 hfGetVertex(HfsIn s, uint vertexIndex, uint columns)
{
    const uint row = vertexIndex / columns;
    const uint col = vertexIndex % columns;
    return float3(float(row), hfHeight(s, vertexIndex), float(col));
}
void hfTriVertexIndices(HfsIn s, uint triangleIndex, uint columns, out uint i0, out uint i1, out uint i2)
{
    const uint cell = triangleIndex >> 1u;
    const bool firstTri = (triangleIndex & 1u) == 0u;
    if (hfShared(s, cell))
    {
        if (firstTri) { i0 = cell + columns; i1 = cell; i2 = cell + columns + 1u; }
        else          { i0 = cell + 1u; i1 = cell + columns + 1u; i2 = cell; }
    }
    else
    {
        if (firstTri) { i0 = cell; i1 = cell + 1u; i2 = cell + columns; }
        else          { i0 = cell + columns + 1u; i1 = cell + columns; i2 = cell + 1u; }
    }
}
void hfTriAdjacencyIndices(HfsIn s, uint triangleIndex, uint columns, uint rows, out uint a0, out uint a1, out uint a2)
{
    const uint cell = triangleIndex >> 1u;
    const bool firstTri = (triangleIndex & 1u) == 0u;
    if (hfShared(s, cell))
    {
        if (firstTri)
        {
            a0 = HF_BOUNDARY; a1 = triangleIndex + 1u; a2 = HF_BOUNDARY;
            if (cell % columns != 0u)           a0 = triangleIndex - 1u;
            if (cell / columns != rows - 2u)    a2 = ((cell + columns) * 2u) + 1u;
        }
        else
        {
            a0 = HF_BOUNDARY; a1 = triangleIndex - 1u; a2 = HF_BOUNDARY;
            if (cell % columns < columns - 2u)  a0 = triangleIndex + 1u;
            if (cell >= columns - 1u)           a2 = (cell - columns) * 2u;
        }
    }
    else
    {
        if (firstTri)
        {
            a0 = HF_BOUNDARY; a1 = triangleIndex + 1u; a2 = HF_BOUNDARY;
            if (cell >= columns - 1u)           a0 = ((cell - columns) * 2u) + 1u;
            if (cell % columns != 0u)           a2 = triangleIndex - 1u;
        }
        else
        {
            a0 = HF_BOUNDARY; a1 = triangleIndex - 1u; a2 = HF_BOUNDARY;
            if (cell / columns != rows - 2u)    a0 = (cell + columns) * 2u;
            if (cell % columns < columns - 2u)  a2 = triangleIndex + 1u;
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxHfNpSetup(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const HfsIn s = gIn[t];

    HfsOut r;

    const float4 t2cq = quatMulQ(quatConjQ(s.cq), s.tq);
    const float3 t2cp = quatRotInvQ(s.cq, s.tp.xyz - s.cp.xyz);
    const float contactDist = s.dists.x + s.dists.y;
    r.t2cq = t2cq;
    r.t2cp = float4(t2cp, contactDist);

    const float3 com = vertex2ShapeQ(s.cvxBlob[0].xyz, s.cScaleS.xyz, s.cScaleR);
    r.com = float4(com, s.cvxBlob[1].x);

    const uint rows = s.hf[0].x;
    const uint columns = s.hf[0].y;
    const float rowScale = s.hfScale.x, heightScale = s.hfScale.y, columnScale = s.hfScale.z;

    // getTriangle (cu L83-116): the handedness vector is dead
    bool wrongHanded = false;
    if (columnScale < 0.0f) wrongHanded = !wrongHanded;
    if (rowScale < 0.0f) wrongHanded = !wrongHanded;
    const uint wh = wrongHanded ? 1u : 0u;

    const uint triangleIdx = s.pairInfo.x;
    uint tv[3];
    {
        uint i0, i1, i2;
        hfTriVertexIndices(s, triangleIdx, columns, i0, i1, i2);
        tv[0] = i0; tv[1u + wh] = i1; tv[2u - wh] = i2;
    }
    uint adj[3];
    {
        uint a0, a1, a2;
        hfTriAdjacencyIndices(s, triangleIdx, columns, rows, a0, a1, a2);
        adj[wh != 0u ? 2u : 0u] = a0; adj[1] = a1; adj[wh != 0u ? 0u : 2u] = a2;
    }
    r.adj = uint4(adj[0], adj[1], adj[2], wh);

    float3 verts[3];
    for (uint vi = 0u; vi < 3u; ++vi)
    {
        const float3 v = hfGetVertex(s, tv[vi], columns);
        verts[vi] = float3(v.x * rowScale, v.y * heightScale, v.z * columnScale); // hf2shapep
    }

    const float3 triNormal = crossQ(verts[1] - verts[0], verts[2] - verts[0]); // denormalizedNormal
    const float3 locNormal = getNormalizedQ(quatRotQ(t2cq, triNormal));        // rotate ONLY, then normalize
    r.locNormal = float4(locNormal, 0.0f);

    for (uint vi = 0u; vi < 3u; ++vi)
        r.locVerts[vi] = float4(quatRotQ(t2cq, verts[vi]) + t2cp, 0.0f);

    r.offsets = uint4(s.pairInfo.y + s.pairInfo.w, s.pairInfo.z + s.pairInfo.w, 0u, 0u);
    gOut[t] = r;
}
