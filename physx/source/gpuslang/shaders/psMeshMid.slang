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

// psMeshMid.slang
//
// Slang port of PsTreeContactGenTraverser — the particle-vs-triangle-mesh
// midphase (gpunarrowphase/src/CUDA/particleSystemMeshMidphase.cu): one
// 32-thread group = 32 particles walking ONE mesh BV32 tree together.
//
// - constructor (per lane): particle position transformed into the mesh
//   shape space, the speculative CCD contact volume
//   (getParticleSpeculativeContactVolume, isDiffuse/isTGS flags) or the
//   plain contact distance, and the volume position additionally taken to
//   VERTEX space (shape2Vertex).  The active-particle mask is the ballot of
//   valid lanes and — faithfully to the CUDA — is NEVER refined afterwards:
//   intersectBoxFullWarp computes a refined mask but only returns "any"
//   (the scratch's sBv32ActiveParticles stack is unused legacy upstream).
// - intersectBoxFullWarp: for each active particle (broadcast), clamp its
//   vertex-space volume position to the child box, take the delta back to
//   SHAPE space (vertex2Shape), and test squared distance < radius^2; the
//   node is visited when ANY active particle intersects.
// - intersectPrimitiveFullWarp: for each active particle p (broadcast of
//   its shape-space positions/radius), every lane tests ITS OWN leaf
//   triangle (verts vertex2Shape'd, V1/V2 swapped when the mesh scale
//   determinant is negative — MeshScaling::flipNormal) via
//   particleTriangleTest; contacts are compacted into p's per-particle list
//   (ballot + exclusive prefix), capped at MaxStaticContactsPerMesh = 6
//   with a serialized WORST-CONTACT REPLACEMENT pass for the overflow:
//   for each failed writer (in lane order), find the current worst stored
//   depth (maxIndex, lowest lane wins ties); if the failed depth is
//   smaller, it replaces that slot.  The owning lane's contactCount
//   saturates at 6 after any overflow round.
// - finalizeFullWarp: for a STATIC rigid, contacts append to the particle's
//   one-way list (per-particle atomicAdd, global cap
//   MaxStaticContactsPerParticle = 12, strided by the particle count,
//   pen = depth - restDistance, node index stored).  Whatever did not fit
//   (or everything, for dynamic rigids) goes through the shared
//   PxgParticleContactWriter path: warp scan of the remainders, ONE lane-31
//   atomicAdd, then per-lane sequential writes — suppressed for the diffuse
//   dispatch (isDiffuseParticlesThread).
//
// Engine glue not ported (documented): LoadShapePair/transform-cache reads,
// the SDF-mesh variant, the per-pair particle-system resolution, the
// grid-stride batching of particle groups (work items carry poses, scale,
// flags and the particle range directly), and the driver's PER-PARTICLE
// rigid-filter early-out (find(particleSystem, rigidId, particleMask)
// before the traverser runs) - the eventual driver must reapply it.
// NOTE: flag bit 8 (rigid is static) must stay consistent with the rigid
// node index in the real driver (isStaticBody == PX_INVALID_NODE).

import pxgSlangCommon;

static const float PX_MAX_F32 = 3.4028235e38f;
static const float PX_EPS_REAL = 1.192092896e-07f;
static const uint GU_BV4_CHILD_OFFSET_SHIFT_COUNT = 11;
static const uint MAX_MESH_CONTACTS = 6;   // PxgParticleContactInfo::MaxStaticContactsPerMesh
static const uint MAX_PARTICLE_CONTACTS = 12; // ...::MaxStaticContactsPerParticle
static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_PGS = 1.0f;
static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_TGS = 0.5f;
static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_DIFFUSE = 1.0f;

struct PmParams
{
    uint numParticles;   // particles in this dispatch (32 per group)
    uint maxContacts;    // writer capacity
    uint flags;          // 1 = enableCCD, 2 = isTGS, 4 = isDiffuseParticlesThread, 8 = rigid is static
    uint particleSystemId;
}

struct PmMesh
{
    float4 meshP;     // mesh pose position
    float4 meshQ;     // mesh pose quaternion
    float4 scale;     // mesh scale (w = contactDist)
    float4 scaleRot;  // mesh scale rotation quaternion (w unused)
    float4 misc;      // x = restDist
    uint4 rigid;      // xy = rigidId (u64)
}

// Gu::BV32DataPacked (1168 bytes)
struct Bv32Node
{
    float4 mMin[32];
    float4 mMax[32];
    uint mData[32];
    uint mNbNodes;
    uint mDepth;
    uint pad0, pad1;
}

// PxgParticlePrimitiveContact (32 bytes)
struct PsContact
{
    float4 normal_pen;
    uint2 rigidId;
    uint2 particleId;
}

[[vk::push_constant]]
ConstantBuffer<PmParams> gParams;

RWStructuredBuffer<PmMesh> gMesh;              // binding 0 (single element)
RWStructuredBuffer<Bv32Node> gNodes;           // binding 1
RWStructuredBuffer<uint4> gTriIndices;         // binding 2
RWStructuredBuffer<float4> gTriVerts;          // binding 3 (vertex space)
RWStructuredBuffer<float4> gCurrentPos;        // binding 4 (world)
RWStructuredBuffer<float4> gPredictedPos;      // binding 5 (world)
RWStructuredBuffer<float4> gOneWayInfos;       // binding 6 (strided)
RWStructuredBuffer<uint2> gOneWayNodeIndices;  // binding 7 (strided)
RWStructuredBuffer<uint> gOneWayCounts;        // binding 8 (per particle)
RWStructuredBuffer<PsContact> gContacts;       // binding 9 (writer)
RWStructuredBuffer<uint2> gContactByRigid;     // binding 10
RWStructuredBuffer<uint> gTempContactByRigid;  // binding 11
RWStructuredBuffer<uint> gContactIdxByRigid;   // binding 12
RWStructuredBuffer<uint2> gContactSortedByParticle; // binding 13
RWStructuredBuffer<uint> gTempContactByParticle;    // binding 14
RWStructuredBuffer<uint> gContactIdxByParticle;     // binding 15
RWStructuredBuffer<uint> gContactCount;        // binding 16

groupshared int sBv32Nodes[192];
groupshared uint sMask;
groupshared uint sStart;
groupshared float3 sBcastP;   // broadcast particle pos (shape space)
groupshared float3 sBcastCvp; // broadcast volume pos (shape space)
groupshared float3 sBcastCvv; // broadcast volume pos (vertex space)
groupshared float sBcastR;    // broadcast volume radius
groupshared uint sBcastCnt;   // broadcast contact count
groupshared float3 shLocalContacts[32][MAX_MESH_CONTACTS];
groupshared float shLocalContactDepths[32][MAX_MESH_CONTACTS];
groupshared float sRed[32];   // reduction scratch
groupshared uint sScan[32];

uint ballot32(bool pred, uint idx)
{
    GroupMemoryBarrierWithGroupSync();
    if (idx == 0) sMask = 0;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sMask, 1u << idx);
    GroupMemoryBarrierWithGroupSync();
    return sMask;
}

// vertex2Shape / shape2Vertex (cudaNpCommon.h): scale in a rotated frame
float3 vertex2Shape(float3 v, float3 scale, float4 rotation)
{
    return quatRotateInv(rotation, quatRotate(rotation, v) * scale);
}

float3 shape2Vertex(float3 v, float3 scale, float4 rotation)
{
    return quatRotateInv(rotation, quatRotate(rotation, v) / scale);
}

float getParticleSpeculativeContactVolume(out float3 cVolumePos,
    float3 currentPos, float3 predictedPos, float contactDist, bool isDiffuse, bool isTGS)
{
    const float sScaleInvPGS = 1.0f / PARTICLE_FORWARD_PROJECTION_STEP_SCALE_PGS;
    const float sScaleInvTGS = 1.0f / PARTICLE_FORWARD_PROJECTION_STEP_SCALE_TGS;
    const float sScaleInvDiffuse = 1.0f / PARTICLE_FORWARD_PROJECTION_STEP_SCALE_DIFFUSE;
    float scaleInv;
    if (isDiffuse)
        scaleInv = sScaleInvDiffuse;
    else
        scaleInv = isTGS ? sScaleInvTGS : sScaleInvPGS;
    const float3 cVolumeOffset = (predictedPos - currentPos) * scaleInv * 0.5f;
    cVolumePos = currentPos + cVolumeOffset;
    // P9.47: gridCal.cuh:139 is `cVolumeOffset.magnitude() + contactDist`, and
    // PxVec3::magnitude() is PxSqrt(magnitudeSquared()) -- not GLSL Length, which
    // the Vulkan environment spec leaves several ULP of room.
    return sqrt(dot(cVolumeOffset, cVolumeOffset)) + contactDist;
}

float3 closestPtPointTriangle2(float3 p, float3 a, float3 b, float3 c, float3 ab, float3 ac)
{
    const float3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) { const float v = d1 / (d1 - d3); return a + v * ab; }
    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) { const float w = d2 / (d2 - d6); return a + w * ac; }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    { const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return b + w * (c - b); }
    const float denom = 1.0f / (va + vb + vc);
    return a + ab * (vb * denom) + ac * (vc * denom);
}

float distancePointTriangleSquared(float3 p, float3 a, float3 b, float3 c, out float3 closestP)
{
    const float3 ab = b - a;
    const float3 ac = c - a;
    closestP = closestPtPointTriangle2(p, a, b, c, ab, ac);
    const float3 v = p - closestP;
    return dot(v, v);
}

bool particleTriangleTest(inout float3 normal, inout float distance,
                          float3 particlePos, float3 cVolumePos, float cVolumeRadius,
                          float3 triV0, float3 triV1, float3 triV2, bool enableCCD)
{
    const float3 cn = cross(triV1 - triV0, triV2 - triV0);
    const float m = dot(cn, cn);
    const float3 triNormal = m > 0.0f ? cn * rsqrt(m) : float3(0, 0, 0);
    const float cVolumeRadiusSq = cVolumeRadius * cVolumeRadius;

    const float cVolumePosPlaneDist = dot(cVolumePos - triV0, triNormal);
    float3 closestPos;
    if (cVolumePosPlaneDist < cVolumeRadius)
    {
        float distSq = distancePointTriangleSquared(cVolumePos, triV0, triV1, triV2, closestPos);
        if (distSq <= cVolumeRadiusSq)
        {
            const float particlePosPlaneDist = dot(particlePos - triV0, triNormal);
            if (particlePosPlaneDist > 0.0f)
            {
                if (enableCCD)
                    distSq = distancePointTriangleSquared(particlePos, triV0, triV1, triV2, closestPos);
                const float3 distVec = particlePos - closestPos;
                normal = (distSq > 0.0f) ? distVec * rsqrt(distSq) : triNormal;
                distance = sqrt(distSq);
                return true;
            }
        }
    }
    return false;
}

// PxgParticleContactWriter::writeContact
void writeContact(uint index, float4 normalPen, uint2 compressedParticleIndex, uint2 rigidId)
{
    if (index < gParams.maxContacts)
    {
        gContactByRigid[index] = rigidId;
        gTempContactByRigid[index] = rigidId.x;
        gContactIdxByRigid[index] = index;
        gContactSortedByParticle[index] = compressedParticleIndex;
        gTempContactByParticle[index] = compressedParticleIndex.x;
        gContactIdxByParticle[index] = index;
        PsContact ct;
        ct.normal_pen = normalPen;
        ct.rigidId = rigidId;
        ct.particleId = compressedParticleIndex;
        gContacts[index] = ct;
    }
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(32, 1, 1)]
void ctPsMeshMid(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const uint globalParticleIndex = gid.x * 32 + lane;
    const uint numParticles = gParams.numParticles;
    const bool enableCCD = (gParams.flags & 1) != 0;
    const bool isTGS = (gParams.flags & 2) != 0;
    const bool isDiffuseParticlesThread = (gParams.flags & 4) != 0;
    const bool rigidIsStatic = (gParams.flags & 8) != 0;

    const PmMesh mesh = gMesh[0];
    const float contactDist = mesh.scale.w;
    const float restDist = mesh.misc.x;
    const bool flipNormal = (mesh.scale.x * mesh.scale.y * mesh.scale.z) < 0.0f; // MeshScaling

    // ---- constructor ----
    float3 particlePosShapeSpace = float3(0, 0, 0);
    float3 cVolumePosShapeSpace = float3(0, 0, 0);
    float3 cVolumePosVertexSpace = float3(0, 0, 0);
    float cVolumeRadius = 0.0f;
    uint contactCount = 0;

    if (globalParticleIndex < numParticles)
    {
        const float3 currentPos = gCurrentPos[globalParticleIndex].xyz;
        particlePosShapeSpace = quatRotateInv(mesh.meshQ, currentPos - mesh.meshP.xyz);
        if (enableCCD)
        {
            const float3 predictedPos = gPredictedPos[globalParticleIndex].xyz;
            float3 cVolumePos;
            cVolumeRadius = getParticleSpeculativeContactVolume(cVolumePos, currentPos, predictedPos,
                                                                contactDist, isDiffuseParticlesThread, isTGS);
            cVolumePosShapeSpace = quatRotateInv(mesh.meshQ, cVolumePos - mesh.meshP.xyz);
        }
        else
        {
            cVolumePosShapeSpace = particlePosShapeSpace;
            cVolumeRadius = contactDist;
        }
        cVolumePosVertexSpace = shape2Vertex(cVolumePosShapeSpace, mesh.scale.xyz, mesh.scaleRot);
    }

    // the active mask is set ONCE and never refined (faithful to the CUDA)
    const uint activeParticles = ballot32(globalParticleIndex < numParticles, lane);

    // ---- BV32 walk (standard single stack) ----
    uint currentNodeIndex = 0;
    bool hasNode = true;
    uint nbStack = 0;

    while (hasNode)
    {
        const uint nodeIdx = currentNodeIndex >> 6;

        if ((currentNodeIndex & 1) != 0)
        {
            const uint leafOffset = (currentNodeIndex >> 1) & 31;
            const uint data = gNodes[nodeIdx].mData[leafOffset];
            const uint nbPrimitives = (data >> 1) & 63;
            const uint primStartIndex = data >> 7;
            uint primitiveIndex = 0xffffffff;
            if (lane < nbPrimitives)
                primitiveIndex = lane + primStartIndex;

            // per-triangle data for THIS lane
            float3 tv0 = float3(0, 0, 0), tv1 = float3(0, 0, 0), tv2 = float3(0, 0, 0);
            if (primitiveIndex != 0xffffffff)
            {
                const uint4 triIdx = gTriIndices[primitiveIndex];
                tv0 = vertex2Shape(gTriVerts[triIdx.x].xyz, mesh.scale.xyz, mesh.scaleRot);
                tv1 = vertex2Shape(gTriVerts[triIdx.y].xyz, mesh.scale.xyz, mesh.scaleRot);
                tv2 = vertex2Shape(gTriVerts[triIdx.z].xyz, mesh.scale.xyz, mesh.scaleRot);
                if (flipNormal)
                {
                    const float3 t = tv1;
                    tv1 = tv2;
                    tv2 = t;
                }
            }

            // serialize over the active particles (broadcast p's data)
            for (uint i = activeParticles; i != 0; i &= (i - 1))
            {
                const uint p = firstbitlow(i);
                GroupMemoryBarrierWithGroupSync();
                if (lane == p)
                {
                    sBcastP = particlePosShapeSpace;
                    sBcastCvp = cVolumePosShapeSpace;
                    sBcastR = cVolumeRadius;
                    sBcastCnt = contactCount;
                }
                GroupMemoryBarrierWithGroupSync();
                const float3 pPos = sBcastP;
                const float3 pCvp = sBcastCvp;
                const float pR = sBcastR;
                const uint contactStartIndex = sBcastCnt;

                bool intersect = false;
                float3 normal = float3(0, 0, 0);
                float distance = PX_MAX_F32;
                if (primitiveIndex != 0xffffffff)
                {
                    float3 normalShapeSpace = float3(0, 0, 0);
                    intersect = particleTriangleTest(normalShapeSpace, distance, pPos, pCvp, pR,
                                                     tv0, tv1, tv2, enableCCD);
                    if (intersect)
                        normal = quatRotate(mesh.meshQ, normalShapeSpace);
                }

                const uint resultWarp = ballot32(intersect, lane);
                const uint offset = countbits(resultWarp & ((1u << lane) - 1u));
                const uint validCount = countbits(resultWarp);

                if (lane == p)
                    contactCount += validCount;

                bool failedWrite = false;
                if (intersect)
                {
                    const uint index = contactStartIndex + offset;
                    if (index < MAX_MESH_CONTACTS)
                    {
                        shLocalContacts[p][index] = normal;
                        shLocalContactDepths[p][index] = distance;
                    }
                    else
                        failedWrite = true;
                }

                const uint failMask = ballot32(failedWrite, lane);
                if (failMask != 0)
                {
                    float outSep = -PX_MAX_F32;
                    if (lane < MAX_MESH_CONTACTS)
                        outSep = shLocalContactDepths[p][lane];

                    for (uint f = failMask; f != 0; f &= (f - 1))
                    {
                        const uint failLane = firstbitlow(f);
                        GroupMemoryBarrierWithGroupSync();
                        if (lane == failLane)
                            sBcastR = distance; // reuse as depth broadcast
                        GroupMemoryBarrierWithGroupSync();
                        const float depth = sBcastR;

                        const uint candidateSwapMask = ballot32(depth < outSep, lane);
                        if (candidateSwapMask != 0)
                        {
                            // maxIndex over outSep (lowest lane wins ties)
                            sRed[lane] = outSep;
                            GroupMemoryBarrierWithGroupSync();
                            uint bestIndex = 0;
                            {
                                float best = -PX_MAX_F32;
                                for (uint j = 0; j < 32; ++j)
                                    if (sRed[j] > best) { best = sRed[j]; bestIndex = j; }
                            }
                            GroupMemoryBarrierWithGroupSync();
                            if (lane == failLane)
                            {
                                shLocalContacts[p][bestIndex] = normal;
                                shLocalContactDepths[p][bestIndex] = distance;
                            }
                            GroupMemoryBarrierWithGroupSync();
                            if (lane == bestIndex)
                                outSep = depth;
                        }
                    }
                    if (lane == p)
                        contactCount = MAX_MESH_CONTACTS;
                }
            }
        }
        else
        {
            const uint nbChildren = gNodes[nodeIdx].mNbNodes;

            float3 mn = float3(0, 0, 0), mx = float3(0, 0, 0);
            const bool hasBox = lane < nbChildren;
            if (hasBox)
            {
                mn = gNodes[nodeIdx].mMin[lane].xyz;
                mx = gNodes[nodeIdx].mMax[lane].xyz;
            }

            // intersectBoxFullWarp: any active particle's volume touches
            bool intersect = false;
            for (uint i = activeParticles; i != 0; i &= (i - 1))
            {
                const uint p = firstbitlow(i);
                GroupMemoryBarrierWithGroupSync();
                if (lane == p)
                {
                    sBcastCvv = cVolumePosVertexSpace;
                    sBcastR = cVolumeRadius;
                }
                GroupMemoryBarrierWithGroupSync();
                const float3 sCvv = sBcastCvv;
                const float sR = sBcastR;
                if (hasBox)
                {
                    const float3 closestVertexSpace = max(mn, min(mx, sCvv));
                    const float3 distVecVertexSpace = closestVertexSpace - sCvv;
                    const float3 distVec = vertex2Shape(distVecVertexSpace, mesh.scale.xyz, mesh.scaleRot);
                    const float distSq = dot(distVec, distVec);
                    if (distSq < sR * sR)
                        intersect = true;
                }
            }

            uint currentIndex = (currentNodeIndex & 0xFFFFFFC0) | (lane << 1) | 1;

            bool isValidNode = false;
            if (lane < nbChildren)
            {
                const uint data = gNodes[nodeIdx].mData[lane];
                isValidNode = intersect;
                if (isValidNode && (data & 1) == 0)
                    currentIndex = (data >> GU_BV4_CHILD_OFFSET_SHIFT_COUNT) << 6;
            }

            const uint resultWarp = ballot32(isValidNode, lane);
            const uint offset = countbits(resultWarp & ((1u << lane) - 1u));
            const uint validCount = countbits(resultWarp);

            if (isValidNode)
                sBv32Nodes[nbStack + offset] = int(currentIndex);

            nbStack += validCount;
        }

        GroupMemoryBarrierWithGroupSync();

        if (nbStack > 0)
        {
            nbStack -= 1;
            currentNodeIndex = uint(sBv32Nodes[nbStack]);
        }
        else
            hasNode = false;
    }

    // ---- finalizeFullWarp ----
    uint id = 0;
    if (globalParticleIndex < numParticles)
    {
        if (rigidIsStatic)
        {
            if (contactCount != 0)
            {
                uint startIndex;
                InterlockedAdd(gOneWayCounts[globalParticleIndex], contactCount, startIndex);
                const uint endIndex = min(MAX_PARTICLE_CONTACTS, startIndex + contactCount);
                for (uint i2 = startIndex; i2 < endIndex; ++i2, ++id)
                {
                    const float3 normal = shLocalContacts[lane][id];
                    const float penetration = shLocalContactDepths[lane][id] - restDist;
                    const uint outputIndex = globalParticleIndex + i2 * numParticles;
                    gOneWayInfos[outputIndex] = float4(normal, penetration);
                    gOneWayNodeIndices[outputIndex] = mesh.rigid.xy;
                }
            }
        }
    }

    const uint remaining = contactCount - id;
    const uint anyMask = ballot32(remaining != 0, lane);
    if (anyMask != 0)
    {
        // inclusive warp scan of the remainders
        sScan[lane] = remaining;
        GroupMemoryBarrierWithGroupSync();
        uint count = 0;
        for (uint j = 0; j <= lane; ++j)
            count += sScan[j];
        uint total = count;
        for (uint j = lane + 1; j < 32; ++j)
            total += sScan[j];
        GroupMemoryBarrierWithGroupSync();

        if (lane == 31)
        {
            uint s;
            InterlockedAdd(gContactCount[0], total, s);
            sStart = s;
        }
        GroupMemoryBarrierWithGroupSync();
        uint startIndex = sStart;

        if (globalParticleIndex < numParticles)
        {
            startIndex = startIndex + count - remaining;

            if (remaining != 0 && !isDiffuseParticlesThread)
            {
                const uint2 compressedParticleIndex = uint2(globalParticleIndex, gParams.particleSystemId);
                for (uint i2 = id, index = startIndex; i2 < contactCount; ++i2, ++index)
                {
                    const float3 normal = shLocalContacts[lane][i2];
                    const float penetration = shLocalContactDepths[lane][i2] - restDist;
                    writeContact(index, float4(normal, penetration), compressedParticleIndex, mesh.rigid.xy);
                }
            }
        }
    }
}
