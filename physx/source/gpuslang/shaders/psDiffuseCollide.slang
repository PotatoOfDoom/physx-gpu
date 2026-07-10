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

// psDiffuseCollide.slang
//
// Slang port of psDiffusePrimitivesCollision - the DIFFUSE-particle variant
// of the particle-vs-primitive collision driver
// (gpunarrowphase/src/CUDA/cudaParticleSystem.cu, used by
// ps_primitivesDiffuseCollisionLaunch):
//
// Unlike the rigid-contact driver there is no filter lookup and no warp
// aggregation: each thread owns one work item (pair x diffuse grid cell),
// iterates ITS OWN cell population, and per intersecting particle
// atomically bumps that particle's one-way contact count; slots below
// PxgParticleContactInfo::MaxStaticContactsPerParticle (12) write the
// contact info STRIDED by the diffuse particle count
// (outputIndex = particleIndex + slot * numParticles): normal_pen
// (w = distance - restDistance), a zeroed impulse and the rigid node index.
// The speculative CCD contact volume uses isDiffuse = TRUE
// (PARTICLE_FORWARD_PROJECTION_STEP_SCALE_DIFFUSE).  The whole kernel
// early-outs when the system has no diffuse particles
// (mMaxDiffuseParticles == 0).
//
// Engine glue not ported (documented): the workIndex -> (pair, cellOffset)
// binary search + calcGridHashInBounds (work items carry the hash),
// LoadShapePair/transform cache reads, the eCONVEXCORE dispatch case, and
// the per-pair particle-system resolution (the engine indexes
// particleSystems[particleSystemId] per work item; this port flattens the
// diffuse buffers and counts to dispatch-wide bindings/params, leaving
// ids.w unused).

import pxgSlangCommon;

static const float PX_MAX_F32 = 3.4028235e38f;
static const float PX_EPS_REAL = 1.192092896e-07f;
static const uint EMPTY_CELL = 0xffffffff;
static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_PGS = 1.0f;
static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_TGS = 0.5f;
static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_DIFFUSE = 1.0f;

struct PdParams { uint numWork; uint numParticles; uint maxDiffuseParticles; uint isTGS; }

struct PdWork
{
    float4 shapeP; // w = cDistance (contactDist0 + contactDist1)
    float4 shapeQ;
    float4 scale;  // w = restDistance
    uint4 ids;     // x = gridHash, y = shape type (0 plane, 1 sphere, 2 capsule, 3 box), z = enableCCD, w = particleSystemId
    uint4 rigid;   // xy = rigidId (PxNodeIndex u64)
}

[[vk::push_constant]]
ConstantBuffer<PdParams> gParams;

RWStructuredBuffer<PdWork> gWork;           // binding 0
RWStructuredBuffer<uint> gCellStart;        // binding 1 (diffuse cells)
RWStructuredBuffer<uint> gCellEnd;          // binding 2
RWStructuredBuffer<float4> gCurrentPos;     // binding 3 (diffuse sorted origin pos)
RWStructuredBuffer<float4> gPredictedPos;   // binding 4 (diffuse sorted pos)
RWStructuredBuffer<float4> gInfos;          // binding 5 (PxgParticleContactInfo, strided)
RWStructuredBuffer<float> gImpulses;        // binding 6
RWStructuredBuffer<uint2> gNodeIndices;     // binding 7
RWStructuredBuffer<uint> gCounts;           // binding 8 (per diffuse particle)

static const uint MAX_STATIC_CONTACTS_PER_PARTICLE = 12; // PxgParticleContactInfo

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
    return length(cVolumeOffset) + contactDist;
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
                                PdWork w, bool enableCCD, inout float3 normal, inout float dist)
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

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctPsDiffuseCollide(uint3 tid : SV_DispatchThreadID)
{
    const uint workIndex = tid.x;
    const bool isTGS = gParams.isTGS != 0;

    if (workIndex >= gParams.numWork)
        return;

    const PdWork w = gWork[workIndex];
    const bool enableCCD = w.ids.z != 0;

    // if there are no diffuse particles, skip this function
    if (gParams.maxDiffuseParticles == 0)
        return;

    uint startIndex = gCellStart[w.ids.x];
    uint endIndex = 0;
    uint range = 0;
    if (startIndex != EMPTY_CELL)
    {
        endIndex = gCellEnd[w.ids.x];
        range = endIndex - startIndex;
    }

    const uint numParticles = gParams.numParticles;

    for (uint ind = 0; ind < range; ++ind)
    {
        bool intersect = false;
        float3 normal = float3(0, 0, 0);
        float distance = 0.0f;
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
                    currentPos, predictedPos, w.shapeP.w, true, isTGS); // isDiffuse = true
            }
            else
            {
                cVolumeRadius = w.shapeP.w;
                cVolumePos = currentPos;
            }

            intersect = particlePrimitiveCollision(currentPos, cVolumePos, cVolumeRadius,
                                                   w, enableCCD, normal, distance);

            if (intersect)
            {
                uint index;
                InterlockedAdd(gCounts[particleIndex], 1, index);
                if (index < MAX_STATIC_CONTACTS_PER_PARTICLE)
                {
                    const uint outputIndex = particleIndex + index * numParticles;
                    gInfos[outputIndex] = float4(normal, distance - w.scale.w);
                    gImpulses[outputIndex] = 0.0f;
                    gNodeIndices[outputIndex] = w.rigid.xy;
                }
            }
        }
    }
}
