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

// npCvxPlaneGen.slang
//
// Slang port of the convexPlaneNphase_kernel_core bRunNarrowphase contact
// generation (cudaGJKEPA.cu L2952-3059): face selection via
// getPolygonIndexFromPlaneNormal with -n (tranche 155a, inlined), the
// vertex-emission loop, the two schlock::reduce passes and the manifold write.
// Load-bearing quirks ported VERBATIM:
//   - the emission loop iterates planeIndex = bestPlaneBIdx, then plane2, but
//     the polygon descriptor is read as vRef8NbVertsMinIndex[bestPlaneBIdx] --
//     LOOP-INVARIANT (cu L2966): when the edge path supplies a second face,
//     the second iteration re-emits the FIRST face's vertices (duplicates that
//     the reduce then trims);
//   - per vertex: index = vertexData8[getVRef8(polyDesc) + lane]; pInShapeSpace
//     = vertex2Shape(v, scale, rot) (the MULTIPLY transform); pInPlaneSpace =
//     bToA.transform(...); signDist = pInPlaneSpace.x (the plane normal is +x
//     in plane space); flags = ballot(signDist < contactDist);
//   - if the ballot count exceeds 16 the SPARSE-mask reduce<false,false,4>
//     runs with the uniform localNormal (1,0,0); survivors are compacted into
//     the shared buffer at totalContacts + popc(flags & below);
//   - after the loop: mask = (1 << totalContacts) - 1, p/sep reloaded from the
//     shared buffer for lanes < totalContacts, a second (dense) reduce when
//     totalContacts > 4, and the manifold write at warpScanExclusive offsets:
//     B[off] = bToA.transformInv(p), A[off] = p with x -= sep (pushed onto the
//     plane), NP[off] = (n, sep); lanes 0..2 write moreData ((-bToA.p, 0),
//     planeQ, convexQ) and lane 0 writes mNbContacts;
//   - when totalContacts == 0 NOTHING is written (no manifold, no moreData,
//     no mNbContacts -- the memory keeps its previous values) and mask stays 0.
//
// One pair per test (32 lanes serialized).  Hull data arrives as direct
// buffers (blob byte reads are tranche 142).

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
// covariant vertex->shape normal transform (shape2Vertex: divide)
float3 shape2Vertex(float3 v, float3 scale, float4 rot)
{
    return quatRotateInv(rot, quatRotate(rot, v) / scale);
}
// contravariant vertex->shape point transform (multiply)
float3 vertex2Shape(float3 v, float3 scale, float4 rot)
{
    return quatRotateInv(rot, quatRotate(rot, v) * scale);
}
float3 loadNormal(float4 p, float3 scale, float4 rot)
{
    return getNormalized(shape2Vertex(p.xyz, scale, rot));
}

struct PlaneGenIn
{
    float4 planes[64];   // hull polygon planes (vertex space)
    float4 verts[64];    // hull vertices (vertex space)
    uint polyDesc[64];   // vRef8NbVertsMinIndex per polygon
    uint4 vtxData[16];   // polyData_vertexData8, byte-packed (256 bytes)
    uint4 fbe[12];       // facesByEdges8, byte-packed (max 96 edges)
    float4 scale;        // .xyz mesh scale; .w = nbPolygons
    float4 rot;          // mesh-scale rotation quat
    float4 n;            // working normal in convex space (.xyz, from 155b); .w = nbEdges
    float4 bToAq;
    float4 bToAp;        // .xyz; .w = contactDist
    float4 planeTq;      // plane transform quat (for the moreData write)
    float4 convexTq;     // convex transform quat
}

struct PlaneGenOut
{
    float4 mNp[4];       // manifold normal+pen written slots (unwritten slots 0)
    float4 mA[4];
    float4 mB[4];
    uint4 moreData0;     // written values (all-zero sentinel when nothing written)
    uint4 moreData1;
    uint4 moreData2;
    float4 p[32];        // final per-lane p (.xyz) / sep (.w) after the shared reload
    uint4 result;        // x = nbContacts (final), y = mask, z = totalContacts, w = wroteManifold
    uint4 aux;           // x = bestPlaneBIdx, y = plane2 (post-selector), z = mNbContacts written flag, w pad
}

struct PlaneGenParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<PlaneGenParams> gParams;

RWStructuredBuffer<PlaneGenIn> gIn;    // binding 0
RWStructuredBuffer<PlaneGenOut> gOut;  // binding 1

uint byteAt(uint4 words[16], uint byteIdx)
{
    return (words[byteIdx / 16u][(byteIdx / 4u) & 3u] >> (8u * (byteIdx & 3u))) & 0xFFu;
}
uint fbeByte(uint4 words[12], uint byteIdx)
{
    return (words[byteIdx / 16u][(byteIdx / 4u) & 3u] >> (8u * (byteIdx & 3u))) & 0xFFu;
}

// schlock::reduce<false,false,4> with a SPARSE lane mask and the UNIFORM
// normal (1,0,0) (tranche 137/146 semantics: lowest-lane argmin/argmax ties)
uint minIdxM(float v[32], uint mask)
{
    float mn = PX_MAX_F32;
    for (uint L = 0u; L < 32u; ++L) if ((mask & (1u << L)) != 0u) mn = min(mn, v[L]);
    for (uint L = 0u; L < 32u; ++L) if (((mask & (1u << L)) != 0u) && v[L] == mn) return L;
    return 0u;
}
uint maxIdxM(float v[32], uint mask)
{
    float mx = -PX_MAX_F32;
    for (uint L = 0u; L < 32u; ++L) if ((mask & (1u << L)) != 0u) mx = max(mx, v[L]);
    for (uint L = 0u; L < 32u; ++L) if (((mask & (1u << L)) != 0u) && v[L] == mx) return L;
    return 0u;
}
uint reduce4(float3 pt[32], float sep[32], float3 normal, uint allMask)
{
    const uint i0 = minIdxM(sep, allMask);
    uint mask = (1u << i0);
    float d1[32];
    for (uint L = 0u; L < 32u; ++L) { const float3 df = pt[i0] - pt[L]; d1[L] = dot(df, df); }
    const uint i1 = maxIdxM(d1, allMask & ~mask);
    mask |= (1u << i1);
    const float3 dir = cross(normal, pt[i1] - pt[i0]);
    float d[32];
    for (uint L = 0u; L < 32u; ++L) d[L] = dot(dir, pt[L] - pt[i0]);
    const uint f = maxIdxM(d, allMask & ~mask);
    mask |= (1u << f);
    uint g = minIdxM(d, allMask & ~mask);
    if (d[f] * d[g] > 0.0f) g = maxIdxM(d, allMask & ~mask);
    mask |= (1u << g);
    return mask;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxPlaneGen(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const PlaneGenIn s = gIn[t];

    const uint nbPolygons = uint(s.scale.w);
    const uint nbEdges = uint(s.n.w);
    const float3 scale = s.scale.xyz;
    const float4 rot = s.rot;
    const float3 n = s.n.xyz;
    const float4 bToAq = s.bToAq;
    const float3 bToAp = s.bToAp.xyz;
    const float contactDist = s.bToAp.w;
    const float3 localNormal = float3(1, 0, 0);

    // ---- getPolygonIndexFromPlaneNormal(nbPolygons, planes, scale, rot, -n, fbe, nbEdges, plane2) ----
    const float3 selN = -n;
    float laneBest[32]; uint laneIdx[32];
    for (uint L = 0u; L < 32u; ++L) { laneBest[L] = -PX_MAX_F32; laneIdx[L] = 0u; }
    for (uint i = 0u; i < nbPolygons; ++i)
    {
        const uint L = i & 31u;
        const float proj = dot(getNormalized(shape2Vertex(s.planes[i].xyz, scale, rot)), selN);
        if (laneBest[L] < proj) { laneIdx[L] = i; laneBest[L] = proj; }
    }
    float bestProjection = -PX_MAX_F32;
    for (uint L = 0u; L < 32u; ++L) bestProjection = max(bestProjection, laneBest[L]);
    uint bestLane = 0u;
    for (uint L = 0u; L < 32u; ++L) if (laneBest[L] == bestProjection) { bestLane = L; break; }
    uint bestPlaneBIdx = laneIdx[bestLane];

    float laneEdge[32]; uint laneEdgeIdx[32];
    for (uint L = 0u; L < 32u; ++L) { laneEdge[L] = -PX_MAX_F32; laneEdgeIdx[L] = 0xFFFFFFFFu; }
    for (uint i = 0u; i < nbEdges; ++i)
    {
        const uint L = i & 31u;
        const uint f0 = fbeByte(s.fbe, 2u * i);
        const uint f1 = fbeByte(s.fbe, 2u * i + 1u);
        const float3 edgeN = getNormalized(loadNormal(s.planes[f0], scale, rot) + loadNormal(s.planes[f1], scale, rot));
        const float dp = dot(edgeN, selN);
        if (dp > laneEdge[L]) { laneEdge[L] = dp; laneEdgeIdx[L] = i; }
    }
    float bestEdgeMax = -PX_MAX_F32;
    for (uint L = 0u; L < 32u; ++L) bestEdgeMax = max(bestEdgeMax, laneEdge[L]);
    uint bestEdgeLane = 0u;
    for (uint L = 0u; L < 32u; ++L) if (laneEdge[L] == bestEdgeMax) { bestEdgeLane = L; break; }
    const float bestEdge = bestEdgeMax + 5e-4f;
    const uint closestEdge = laneEdgeIdx[bestEdgeLane];

    uint plane2 = 0xFFFFFFFFu; // PxI32 plane2 = -1
    if (bestEdge > bestProjection)
    {
        const uint eIdx = closestEdge * 2u;
        const uint f0 = fbeByte(s.fbe, eIdx);
        const uint f1 = fbeByte(s.fbe, eIdx + 1u);
        const float dp0 = dot(loadNormal(s.planes[f0], scale, rot), selN);
        const float dp1 = dot(loadNormal(s.planes[f1], scale, rot), selN);
        if (dp0 > dp1) { bestPlaneBIdx = f0; plane2 = f1; }
        else           { bestPlaneBIdx = f1; plane2 = f0; }
    }
    const uint selPlane2 = plane2; // for aux output

    // ---- vertex emission loop (the polyDesc read is LOOP-INVARIANT: bestPlaneBIdx) ----
    float3 shPos[32]; float shSep[32]; // shared TemporaryContacts model
    uint totalContacts = 0u;

    for (uint planeIndex = bestPlaneBIdx; planeIndex != 0xFFFFFFFFu;)
    {
        float3 pInPlaneSpace[32];
        float signDist[32];
        const uint polyDesc = s.polyDesc[bestPlaneBIdx]; // cu L2966: NOT planeIndex
        const uint nbVertsPoly = (polyDesc & 0xFFFFu) >> 8u; // getNbVerts
        const uint vRef8 = polyDesc >> 16u;                  // getVRef8
        for (uint L = 0u; L < 32u; ++L)
        {
            pInPlaneSpace[L] = float3(0, 0, 0);
            signDist[L] = PX_MAX_F32;
            if (L < nbVertsPoly)
            {
                const uint index = byteAt(s.vtxData, vRef8 + L);
                const float3 pInShapeSpace = vertex2Shape(s.verts[index].xyz, scale, rot);
                pInPlaneSpace[L] = quatRotate(bToAq, pInShapeSpace) + bToAp;
                signDist[L] = pInPlaneSpace[L].x;
            }
        }

        uint flags = 0u;
        for (uint L = 0u; L < 32u; ++L)
            if (signDist[L] < contactDist) flags |= (1u << L);
        uint nbC = countbits(flags);

        if (nbC != 0u)
        {
            if (nbC > 16u)
            {
                flags = reduce4(pInPlaneSpace, signDist, localNormal, flags);
                nbC = countbits(flags);
            }
            for (uint L = 0u; L < 32u; ++L)
            {
                if ((flags & (1u << L)) != 0u)
                {
                    const uint index = countbits(flags & ((1u << L) - 1u));
                    shPos[totalContacts + index] = pInPlaneSpace[L];
                    shSep[totalContacts + index] = signDist[L];
                }
            }
            totalContacts += nbC;
        }

        planeIndex = plane2;
        plane2 = 0xFFFFFFFFu;
    }

    uint nbContacts = totalContacts;

    // ---- post-loop: reload, second reduce, manifold write ----
    PlaneGenOut r;
    for (uint j = 0u; j < 4u; ++j) { r.mNp[j] = float4(0, 0, 0, 0); r.mA[j] = float4(0, 0, 0, 0); r.mB[j] = float4(0, 0, 0, 0); }
    r.moreData0 = uint4(0, 0, 0, 0); r.moreData1 = uint4(0, 0, 0, 0); r.moreData2 = uint4(0, 0, 0, 0);
    for (uint L = 0u; L < 32u; ++L) r.p[L] = float4(0, 0, 0, 0);

    uint mask = 0u;
    uint wrote = 0u;
    float3 pL[32]; float sepL[32];
    for (uint L = 0u; L < 32u; ++L) { pL[L] = float3(0, 0, 0); sepL[L] = 0.0f; }

    if (nbContacts != 0u)
    {
        mask = (1u << totalContacts) - 1u;
        for (uint L = 0u; L < 32u; ++L)
        {
            if (L < nbContacts) { pL[L] = shPos[L]; sepL[L] = shSep[L]; }
        }
        if (nbContacts > 4u)
        {
            mask = reduce4(pL, sepL, localNormal, mask);
            nbContacts = countbits(mask);
        }

        for (uint L = 0u; L < 32u; ++L)
        {
            if ((mask & (1u << L)) != 0u)
            {
                const uint offset = countbits(mask & ((1u << L) - 1u)); // warpScanExclusive
                const float3 p = pL[L];
                const float sep = sepL[L];
                const float3 pInB = quatRotateInv(bToAq, p - bToAp); // bToA.transformInv(p)
                float3 pOnPlane = p;
                pOnPlane.x -= sep;
                r.mB[offset] = float4(pInB, 0.0f);
                r.mA[offset] = float4(pOnPlane, 0.0f);
                r.mNp[offset] = float4(n, sep);
            }
        }

        // lanes 0..2 moreData write + lane 0 mNbContacts
        r.moreData0 = uint4(asuint(-bToAp.x), asuint(-bToAp.y), asuint(-bToAp.z), asuint(0.0f));
        r.moreData1 = uint4(asuint(s.planeTq.x), asuint(s.planeTq.y), asuint(s.planeTq.z), asuint(s.planeTq.w));
        r.moreData2 = uint4(asuint(s.convexTq.x), asuint(s.convexTq.y), asuint(s.convexTq.z), asuint(s.convexTq.w));
        wrote = 1u;
    }

    for (uint L = 0u; L < 32u; ++L) r.p[L] = float4(pL[L], sepL[L]);
    r.result = uint4(nbContacts, mask, totalContacts, wrote);
    r.aux = uint4(bestPlaneBIdx, selPlane2, wrote, 0u);
    gOut[t] = r;
}
