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

// npCvxTriNpSetup.slang
//
// Slang port of the convexTrimeshNarrowphaseCore setup stage (convexMesh.cu
// L74-215): everything the wrapper computes before handing off to
// convexTriangleContactGen (which is the separately-verified tranche-96/110/
// 112 pipeline).  One warp per convex-triangle pair, serialized:
//   - trimeshToConvexTransform = convexT.transformInv(trimeshT) (established
//     exact quaternion forms, sequential quatMul w);
//   - contactDist = contactDistance[ref0] + contactDistance[ref1];
//   - convex blob header: center of mass (float4 at byte 0) pushed through
//     vertex2Shape with the CONVEX scale, and the packed
//     nbEdgesNbHullVerticesNbPolygons word (uint4.x at byte 16);
//   - trimesh blob walk (readTriangleMesh, same layout as tranche 157):
//     header {numVerts, numTris, -, nbPackedNodes}, nbPacked x 1168-byte
//     BV32DataPacked skip, float4 verts, uint4 triIndices, uint4 adjacencies,
//     u32 faceRemap; the pair's adjacency row and faceRemap entry are read;
//   - the three triangle vertices: vertex index from the uint4 row component
//     tid, vertex2Shape with the TRIMESH scale, then trimeshToConvexTransform
//     -- stored at writeIndex = tid, EXCEPT when the trimesh scale determinant
//     is negative: threads 1 and 2 swap targets (writeIndex = tid ^ 3,
//     thread 0 unaffected), flipping the winding;
//   - triangleLocNormal = cross(v1 - v0, v2 - v0).getNormalized() over the
//     PERMUTED vertices;
//   - convexTriPairOffset = pair.startIndex + testOffset and the padded
//     variant from pair.roundedStartIndex.

static const uint BV32_PACKED_BYTES = 1168u; // sizeof(Gu::BV32DataPacked)

struct SetIn
{
    float4 cq, cp;        // convex cached transform (q, p)
    float4 tq, tp;        // trimesh cached transform (q, p)
    float4 cScaleS, cScaleR; // convex PxMeshScale (scale xyz, rotation quat)
    float4 tScaleS, tScaleR; // trimesh PxMeshScale
    float4 dists;         // x = contactDistance[ref0], y = contactDistance[ref1]
    uint4 pairInfo;       // x = triangleIdx, y = startIndex, z = roundedStartIndex, w = testOffset
    float4 cvxBlob[2];    // convex blob head: [0] = CoM float4, [1] = header uint4 (bit-cast)
    uint4 blob[192];      // trimesh blob (uint-packed bytes)
}

struct SetOut
{
    float4 t2cq, t2cp;    // trimeshToConvexTransform (q, p); p.w = contactDist
    float4 com;           // shape-space convex center of mass; w = asfloat(header word)
    uint4 adjRemap;       // xyz = triAdjTrisIdx, w = remapCpuTriangleIdx
    float4 locVerts[3];   // triLocVerts after the flip permutation (w pad)
    float4 locNormal;     // triangleLocNormal; w = flipNormal flag
    uint4 offsets;        // x = convexTriPairOffset, y = convexTriPairOffsetPadded, z/w pad
}

struct SetParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<SetParams> gParams;

RWStructuredBuffer<SetIn> gIn;    // binding 0
RWStructuredBuffer<SetOut> gOut;  // binding 1

float4 quatConjQ(float4 q) { return float4(-q.x, -q.y, -q.z, q.w); }
float4 quatMulQ(float4 a, float4 b) // PxQuat::operator* (this = a)
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
float3 vertex2ShapeQ(float3 v, float3 sc, float4 rot) // multiply transform
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

uint blobWord(SetIn s, uint byteOff) { return s.blob[byteOff / 16u][(byteOff / 4u) & 3u]; }
float blobFloat(SetIn s, uint byteOff) { return asfloat(blobWord(s, byteOff)); }

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxTriNpSetup(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const SetIn s = gIn[t];

    SetOut r;

    // trimeshToConvexTransform = convexT.transformInv(trimeshT) (convexMesh.cu:139).
    // PxTransform::transformInv(PxTransform) is qinv = q.getConjugate();
    // (qinv.rotate(src.p - p), qinv * src.q) (PxTransform.h:162-169).  P9.94d: the
    // translation is the CONJUGATE ROTATED, not quatRotInvQ(s.cq, .) -- the two agree on
    // every nonzero word and differ in the sign of an exact zero (cq = identity,
    // tp - cp = (0,-0,-0): t2cp.y is -0 here, +0 under rotateInv).
    // npCvxTriNpSetup_p94dZeroSign pins it.
    const float4 t2cq = quatMulQ(quatConjQ(s.cq), s.tq);
    const float3 t2cp = quatRotQ(quatConjQ(s.cq), s.tp.xyz - s.cp.xyz);
    const float contactDist = s.dists.x + s.dists.y;
    r.t2cq = t2cq;
    r.t2cp = float4(t2cp, contactDist);

    // convex blob head: CoM into shape space; packed counts word
    const float3 com = vertex2ShapeQ(s.cvxBlob[0].xyz, s.cScaleS.xyz, s.cScaleR);
    r.com = float4(com, s.cvxBlob[1].x); // header word bit-cast through the float4

    // trimesh blob walk (readTriangleMesh)
    const uint numVerts = blobWord(s, 0u);
    const uint numTris = blobWord(s, 4u);
    const uint nbPacked = blobWord(s, 12u);
    uint off = 16u;
    off += BV32_PACKED_BYTES * nbPacked;
    const uint vertsOff = off;
    off += 16u * numVerts;
    const uint triOff = off;
    off += 16u * numTris;
    const uint adjOff = off;
    off += 16u * numTris;
    const uint remapOff = off;

    const uint triangleIdx = s.pairInfo.x;
    const uint adjX = blobWord(s, adjOff + 16u * triangleIdx);
    const uint adjY = blobWord(s, adjOff + 16u * triangleIdx + 4u);
    const uint adjZ = blobWord(s, adjOff + 16u * triangleIdx + 8u);
    const uint remap = blobWord(s, remapOff + 4u * triangleIdx);
    r.adjRemap = uint4(adjX, adjY, adjZ, remap);

    // the three vertices with the flip permutation (cu L180-199)
    const bool flipNormal = (s.tScaleS.x * s.tScaleS.y * s.tScaleS.z) < 0.0f;
    float3 locVerts[3];
    for (uint tid = 0u; tid < 3u; ++tid)
    {
        uint writeIndex = tid;
        if (tid > 0u && flipNormal)
            writeIndex = tid ^ 3u;

        const uint triVertIdx = blobWord(s, triOff + 16u * triangleIdx + 4u * tid);
        const float3 triV = float3(blobFloat(s, vertsOff + 16u * triVertIdx),
                                   blobFloat(s, vertsOff + 16u * triVertIdx + 4u),
                                   blobFloat(s, vertsOff + 16u * triVertIdx + 8u));
        const float3 vShape = vertex2ShapeQ(triV, s.tScaleS.xyz, s.tScaleR);
        locVerts[writeIndex] = quatRotQ(t2cq, vShape) + t2cp;
    }
    for (uint j = 0u; j < 3u; ++j) r.locVerts[j] = float4(locVerts[j], 0.0f);

    const float3 nrm = getNormalizedQ(crossQ(locVerts[1] - locVerts[0], locVerts[2] - locVerts[0]));
    r.locNormal = float4(nrm, flipNormal ? 1.0f : 0.0f);

    r.offsets = uint4(s.pairInfo.y + s.pairInfo.w, s.pairInfo.z + s.pairInfo.w, 0u, 0u);
    gOut[t] = r;
}
