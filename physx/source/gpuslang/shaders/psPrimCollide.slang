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

// psPrimCollide.slang
//
// Slang port of psPrimitivesCollision — the particle-vs-primitive collision
// warp driver of gpunarrowphase/src/CUDA/cudaParticleSystem.cu (used by
// ps_primitivesCollisionLaunch):
//
// Each thread owns one work item (a contact-manager pair x grid cell): read
// cellStart/cellEnd at the item's grid hash (EMPTY_CELL = 0xffffffff), then
// the WARP iterates the maximum cell population across its 32 lanes
// (warp max + broadcast) so the per-round ballot stays full-warp; per round
// each lane tests one particle of its own cell: current position, the
// speculative CCD contact volume (getParticleSpeculativeContactVolume with
// isDiffuse = false) or the plain contact distance, the compressed particle
// index (PhysX 5.12: PxgEncodeParticleRef(systemId, sortedIndex); 5.6.1 used
// PxEncodeParticleIndex and then the rigid filter lookup find(particleSystem,
// rigidId, particleMask) over the UNSORTED id, which 5.12 removed -- its bindings
// and findPs stay for the 5.6.1 reference only) — and particlePrimitiveCollision (the tranche-87
// contact core: plane/sphere/capsule/box; eCONVEXCORE deferred).  Contacts
// are warp-aggregated: ballot, popc for the count and the lane's prefix,
// ONE lane-0 atomicAdd, broadcast, then writeContact
// (PxgParticleContactWriter): capacity-gated stores of the 32-byte
// PxgParticlePrimitiveContact plus the six radix-sort staging arrays
// (contactByRigid / tempContactByRigid / contactIndexSortedByRigid and the
// sorted-by-particle triple).  normal_pen.w = distance - restDistance.
//
// Engine glue not ported (documented): the workIndex -> (pair, cellOffset)
// binary search over the tranche-88 offsets and calcGridHashInBounds (work
// items carry the grid hash directly), LoadShapePair/transform cache reads,
// the eCONVEXCORE dispatch case, and the caller's grid-stride loop with its
// ballot/warpScan compaction of non-empty work items into shared batches
// (ps_primitivesCollisionLaunch) - the device function handles EMPTY/idle
// lanes identically either way.
//
// NOTE the warp-max range is the invariant that keeps the per-round ballot
// and barriers group-uniform - replacing it with each lane's own range is
// not an output change but a uniformity violation (UB in CUDA, barrier
// divergence here).

import pxgSlangCommon;

static const float PX_MAX_F32 = 3.4028235e38f;
static const float PX_EPS_REAL = 1.192092896e-07f;
static const uint EMPTY_CELL = 0xffffffff;
static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_PGS = 1.0f;
static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_TGS = 0.5f;
static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_DIFFUSE = 1.0f;

struct PclParams { uint numWork; uint numBuffers; uint maxContacts; uint isTGS; }

struct PclWork
{
    float4 shapeP; // w = cDistance (contactDist0 + contactDist1)
    float4 shapeQ;
    float4 scale;  // w = restDistance
    uint4 ids;     // x = gridHash, y = shape type (0 plane, 1 sphere, 2 capsule, 3 box), z = enableCCD, w = particleSystemId
    uint4 rigid;   // xy = rigidId (PxNodeIndex u64)
}

// PxgRigidFilterPair (32 bytes)
struct PairR { uint2 index0; uint2 index1; uint index2; uint pad0; uint pad1; uint pad2; }

// PxgParticlePrimitiveContact (32 bytes)
struct PsContact
{
    float4 normal_pen;
    uint2 rigidId;
    uint2 particleId;
}

[[vk::push_constant]]
ConstantBuffer<PclParams> gParams;

RWStructuredBuffer<PclWork> gWork;             // binding 0
RWStructuredBuffer<uint> gCellStart;           // binding 1
RWStructuredBuffer<uint> gCellEnd;             // binding 2
RWStructuredBuffer<uint> gSortedToUnsorted;    // binding 3
RWStructuredBuffer<float4> gCurrentPos;        // binding 4 (sorted origin pos)
RWStructuredBuffer<float4> gPredictedPos;      // binding 5 (sorted predicted pos)
RWStructuredBuffer<uint> gRunsum;              // binding 6 (particle buffer runsum)
RWStructuredBuffer<uint4> gFilterTable;        // binding 7 (per buffer: x = offset, y = count)
RWStructuredBuffer<PairR> gFilterPairs;        // binding 8 (flattened, sorted per buffer)
RWStructuredBuffer<PsContact> gContacts;       // binding 9
RWStructuredBuffer<uint2> gContactByRigid;     // binding 10
RWStructuredBuffer<uint> gTempContactByRigid;  // binding 11
RWStructuredBuffer<uint> gContactIdxByRigid;   // binding 12
RWStructuredBuffer<uint2> gContactSortedByParticle; // binding 13
RWStructuredBuffer<uint> gTempContactByParticle;    // binding 14
RWStructuredBuffer<uint> gContactIdxByParticle;     // binding 15
RWStructuredBuffer<uint> gContactCount;        // binding 16

groupshared uint sMask;
groupshared uint sRange[32];
groupshared uint sStart;

uint ballot32(bool pred, uint idx)
{
    GroupMemoryBarrierWithGroupSync();
    if (idx == 0) sMask = 0;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sMask, 1u << idx);
    GroupMemoryBarrierWithGroupSync();
    return sMask;
}

// ---- filter lookup (deformableElementFilter.cuh) ---------------------------
bool u64Lt(uint2 a, uint2 b) { return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x); }
bool u64Eq(uint2 a, uint2 b) { return a.x == b.x && a.y == b.y; }

// findRange: upper-bound search returning r-1 (see npDefFilter.slang)
uint findRange(uint value, uint length)
{
    if (length == 0)
        return 0;
    uint l = 0, r = length;
    while (l < r)
    {
        const uint m = (l + r) / 2;
        if (gRunsum[m] > value)
            r = m;
        else
            l = m + 1;
    }
    return r - 1;
}

// PxgRigidFilterPair::compare (index0 u64, index1 u64, index2 u32) with the
// ignoreIndex2 early accept of the (rigidId, particleId) wrapper
bool findRigid(uint offset, uint nbPairs, uint2 q0, uint2 q1)
{
    uint l = 0, r = nbPairs;
    while (l < r)
    {
        const uint mid = (l + r) / 2;
        const PairR p = gFilterPairs[offset + mid];
        int cmp = 0;
        if (u64Lt(q0, p.index0)) cmp = -1;
        else if (u64Lt(p.index0, q0)) cmp = 1;
        else if (u64Lt(q1, p.index1)) cmp = -1;
        else if (u64Lt(p.index1, q1)) cmp = 1;
        else if (0u < p.index2) cmp = -1;
        if (cmp == 0 || (u64Eq(q0, p.index0) && u64Eq(q1, p.index1)))
            return true;
        if (cmp < 0) r = mid; else l = mid + 1;
    }
    return false;
}

// find(particleSystem, rigidId, compressedParticleId): localize the UNSORTED
// particle id into its sim buffer, search that buffer's filter list
bool findPs(uint2 rigidId, uint2 particleMask, uint numBuffers)
{
    const uint particleId = particleMask.x; // PxGetParticleIndex = low 32 bits
    const uint bufId = findRange(particleId, numBuffers);
    const uint local = particleId - gRunsum[bufId];
    const uint4 tbl = gFilterTable[bufId];
    return findRigid(tbl.x, tbl.y, rigidId, uint2(local, 0));
}

// ---- speculative CCD contact volume (gridCal.cuh) ---------------------------
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

// ---- particle-primitive contact core (particleCollision.cuh, tranche 87) ---
float3 getBasisVector0(float4 q)
{
    const float x2 = q.x * 2.0f;
    const float w2 = q.w * 2.0f;
    return float3((q.w * w2) - 1.0f + q.x * x2,
                  (q.z * w2) + q.y * x2,
                  (-q.y * w2) + q.z * x2);
}

float distancePointSegmentSquared(float3 a, float3 b, float3 p, out float param)
{
    const float3 ap = p - a;
    const float3 ab = b - a;
    const float nom = dot(ap, ab);
    const float denom = dot(ab, ab);
    const float tValue = clamp(nom / denom, 0.0f, 1.0f);
    const float t = denom > 0.0f ? tValue : 0.0f;
    const float3 v = ap - (ab * t);
    param = t;
    return dot(v, v);
}

bool contactPointBox(inout float3 normal, inout float dist,
                     float3 pointPos, float3 boxP, float4 boxQ, float3 halfBoxExtents, float maxDist)
{
    const float3 relPointPos = quatRotateInv(boxQ, pointPos - boxP);
    const float3 closestPos = clamp(relPointPos, -halfBoxExtents, halfBoxExtents);
    if (maxDist < PX_MAX_F32)
    {
        const float maxDistSq = maxDist * maxDist;
        const float3 dv = relPointPos - closestPos;
        if (dot(dv, dv) > maxDistSq)
            return false;
    }
    const bool bX = halfBoxExtents.x >= abs(relPointPos.x);
    const bool bY = halfBoxExtents.y >= abs(relPointPos.y);
    const bool bZ = halfBoxExtents.z >= abs(relPointPos.z);
    const bool isInside = bX && bY && bZ;
    if (isInside)
    {
        const float3 distToSurface = halfBoxExtents - abs(closestPos);
        normal = closestPos.z >= 0.0f ? float3(0, 0, 1) : float3(0, 0, -1);
        dist = -distToSurface.z;
        if (distToSurface.x <= distToSurface.y && distToSurface.x <= distToSurface.z)
        {
            normal = closestPos.x >= 0.0f ? float3(1, 0, 0) : float3(-1, 0, 0);
            dist = -distToSurface.x;
        }
        else if (distToSurface.y <= distToSurface.z)
        {
            normal = closestPos.y >= 0.0f ? float3(0, 1, 0) : float3(0, -1, 0);
            dist = -distToSurface.y;
        }
    }
    else
    {
        normal = relPointPos - closestPos;
        const float mag = sqrt(dot(normal, normal)); // PxVec3::normalizeFast
        normal = normal * (1.0f / mag);
        dist = mag;
    }
    normal = quatRotate(boxQ, normal);
    return true;
}

bool contactPointSphere(inout float3 normal, inout float dist,
                        float3 pointPos, float3 spherePos, float sphereRadius, float maxDist)
{
    const float3 relPointPos = pointPos - spherePos;
    const float distCenterSq = dot(relPointPos, relPointPos);
    if (maxDist < PX_MAX_F32)
    {
        const float maxDistCenter = maxDist + sphereRadius;
        if (distCenterSq > maxDistCenter * maxDistCenter)
            return false;
    }
    const float distCenter = sqrt(distCenterSq);
    normal = (distCenter > PX_EPS_REAL) ? relPointPos * (1.0f / distCenter) : float3(1, 0, 0);
    dist = distCenter - sphereRadius;
    return true;
}

bool contactPointCapsule(inout float3 normal, inout float dist,
                         float3 pointPos, float3 capsulePos, float3 capsuleDir,
                         float capsuleRadius, float capsuleHalfHeight, float maxDist)
{
    const float3 capsuleAxisVec = capsuleDir * capsuleHalfHeight;
    const float3 relPointPos = pointPos - capsulePos;
    float t;
    const float distSegmentSq = distancePointSegmentSquared(-capsuleAxisVec, capsuleAxisVec, relPointPos, t);
    if (maxDist < PX_MAX_F32)
    {
        const float maxDistSegment = maxDist + capsuleRadius;
        if (distSegmentSq > maxDistSegment * maxDistSegment)
            return false;
    }
    const float3 distSegmentVec = relPointPos - capsuleAxisVec * (2.0f * t - 1.0f);
    const float distSegment = sqrt(distSegmentSq);
    normal = (distSegment > PX_EPS_REAL) ? (distSegmentVec * (1.0f / distSegment)) : float3(1, 0, 0);
    dist = distSegment - capsuleRadius;
    return true;
}

bool particlePrimitiveCollision(float3 particlePos, float3 cVolumePos, float cVolumeRadius,
                                PclWork w, bool enableCCD, inout float3 normal, inout float dist)
{
    bool intersect = false;
    switch (w.ids.y)
    {
    case 0: // ePLANE
    {
        const float3 pn = quatRotate(w.shapeQ, float3(1, 0, 0));
        const float pdd = -dot(pn, w.shapeP.xyz);
        dist = dot(pn, cVolumePos) + pdd;
        if (dist <= cVolumeRadius)
        {
            if (enableCCD)
                dist = dot(pn, particlePos) + pdd;
            normal = pn;
            intersect = true;
        }
        break;
    }
    case 1: // eSPHERE
    {
        intersect = contactPointSphere(normal, dist, cVolumePos, w.shapeP.xyz, w.scale.x, cVolumeRadius);
        if (intersect && enableCCD)
            contactPointSphere(normal, dist, particlePos, w.shapeP.xyz, w.scale.x, PX_MAX_F32);
        break;
    }
    case 2: // eCAPSULE
    {
        const float3 capsuleDir = getBasisVector0(w.shapeQ);
        intersect = contactPointCapsule(normal, dist, cVolumePos, w.shapeP.xyz, capsuleDir,
                                        w.scale.y, w.scale.x, cVolumeRadius);
        if (intersect && enableCCD)
            contactPointCapsule(normal, dist, particlePos, w.shapeP.xyz, capsuleDir,
                                w.scale.y, w.scale.x, PX_MAX_F32);
        break;
    }
    default: // 3: eBOX (glue only feeds 0-3; the CUDA default is no-intersect)
    {
        intersect = contactPointBox(normal, dist, cVolumePos, w.shapeP.xyz, w.shapeQ, w.scale.xyz, cVolumeRadius);
        if (intersect && enableCCD)
            contactPointBox(normal, dist, particlePos, w.shapeP.xyz, w.shapeQ, w.scale.xyz, PX_MAX_F32);
        break;
    }
    }
    return intersect;
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
        gTempContactByParticle[index] = compressedParticleIndex.x; // PxGetParticleIndex
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
void ctPsPrimCollide(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const uint workIndex = gid.x * 32 + lane;
    const bool isTGS = gParams.isTGS != 0;

    uint startIndex = EMPTY_CELL;
    uint endIndex = 0;
    uint range = 0;
    // clamped load keeps idle lanes initialized (their cell stays EMPTY)
    PclWork w = gWork[workIndex < gParams.numWork ? workIndex : 0];
    bool enableCCD = false;

    if (workIndex < gParams.numWork)
    {
        enableCCD = w.ids.z != 0;
        startIndex = gCellStart[w.ids.x];
        if (startIndex != EMPTY_CELL)
        {
            endIndex = gCellEnd[w.ids.x];
            range = endIndex - startIndex;
        }
    }

    // warp max + broadcast of the cell population
    sRange[lane] = range;
    GroupMemoryBarrierWithGroupSync();
    {
        uint m = 0;
        for (uint j = 0; j < 32; ++j)
            m = max(m, sRange[j]);
        range = m;
    }

    for (uint ind = 0; ind < range; ++ind)
    {
        bool intersect = false;
        float3 normal = float3(0, 0, 0);
        float distance = 0.0f;
        uint2 compressedParticleIndex = uint2(0, 0);

        if (startIndex != EMPTY_CELL)
        {
            const uint particleIndex = startIndex + ind;
            if (particleIndex < endIndex)
            {
                const float3 currentPos = gCurrentPos[particleIndex].xyz;

                float3 cVolumePos;
                float cVolumeRadius;
                if (enableCCD)
                {
                    const float3 predictedPos = gPredictedPos[particleIndex].xyz;
                    cVolumeRadius = getParticleSpeculativeContactVolume(cVolumePos,
                        currentPos, predictedPos, w.shapeP.w, false, isTGS);
                }
                else
                {
                    cVolumeRadius = w.shapeP.w;
                    cVolumePos = currentPos;
                }

                // PhysX 5.12: PxgEncodeParticleRef(systemId, sortedIndex) = (systemId << 34)
                // | index, i.e. the high word holds systemId << 2; and no filter lookup --
                // 5.12 removed the particle-rigid filter (findPs below is kept only as the
                // 5.6.1 reference and is not called).
                compressedParticleIndex = uint2(particleIndex, w.ids.w << 2u);
                intersect = particlePrimitiveCollision(currentPos, cVolumePos, cVolumeRadius,
                                                       w, enableCCD, normal, distance);
            }
        }

        const uint threadMask = (1u << lane) - 1;
        const uint mask = ballot32(intersect, lane);
        const uint nbToLock = countbits(mask);
        const uint contactIndex = countbits(mask & threadMask);

        if (lane == 0)
        {
            uint s = 0;
            if (nbToLock != 0)
                InterlockedAdd(gContactCount[0], nbToLock, s);
            sStart = s;
        }
        GroupMemoryBarrierWithGroupSync();
        const uint contactStartIndex = sStart;

        if (intersect)
        {
            const uint index = contactStartIndex + contactIndex;
            writeContact(index, float4(normal, distance - w.scale.w), compressedParticleIndex, w.rigid.xy);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
