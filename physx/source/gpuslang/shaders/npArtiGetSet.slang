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

// npArtiGetSet.slang
//
// P8.118: the non-tendon half of the direct-GPU-API state get/set family
// (articulationDirectGpuApi.cu:350-819) -- nine mechanical gather/scatter
// kernels, one thread per element, no cross-thread traffic, no barriers.
// The tendon six follow in P8.119.
//
// TRAPS, each a sweep mutant unless marked:
//  1. Every kernel splits globalThreadIndex into groupIndex (the JOB) and a
//     local element index by the MAX pitch (maxDofs / maxLinks / 3 / 6);
//     the inner guard is the articulation's REAL count -- dead pitch slots
//     stay untouched (poison).
//  2. getArtiDofStates / setArtiDofStates map SIX read / FIVE write enum
//     values onto record arrays whose order differs between the two enums
//     (read: pos, vel, accel, force, targetVel, targetPos; write: pos, vel,
//     force, targetVel, targetPos -- no acceleration).
//  3. The setters OR dirty bits into gpuDirtyFlag from exactly ONE thread
//     per articulation (dofIndex/localIndex/elementIndex == 0) -- no RMW
//     race on subgroup-8 (unlike artiUpdateKinematic's all-lane RMW), and
//     each write type sets a DIFFERENT bit set (positions and velocities
//     also raise eNEEDS_KINEMATIC_UPDATE, forces/targets do not).
//  4. getArtiTransformStates: actor pose = body2World * body2Actor
//     .getInverse() -- the engine spelling (inv.p = q.rotateInv(-p), inv.q
//     = conjugate, then transform).  eROOT_GLOBAL_POSE (6) clamps numLinks
//     to 1; the OUTPUT pitch stays maxLinks either way.
//  5. setArtiRootGlobalPoseState writes body2World[0] = actorPose *
//     body2Actor -- NOT the inverse.
//  6. The velocity kernels address Cm::UnAlignedSpatialVector as raw
//     floats: ANGULAR is +0 (top), LINEAR is +3 (bottom), stride 6; the
//     root variants clamp to link 0 but keep the maxLinks output pitch.
//  7. setArtiLinkForceState writes externalAccelerations[link].TOP =
//     force * invMass (the .w lane); setArtiLinkTorqueState writes .BOTTOM
//     = q.rotate(invInertia .* q.rotateInv(torque)) -- the rotate-inv /
//     componentwise / rotate sandwich, per link.
//
// The module is NOCONTRACT: the transform composition and the torque
// sandwich must not fuse into FMAs (bitwise CPU-vs-GPU doctrine).

import pxgSlangCommon;
import pxgSlangArticulationDesc;

// PxArticulationGPUAPIReadType
static const uint GS_R_JOINT_POSITION = 0u;
static const uint GS_R_JOINT_VELOCITY = 1u;
static const uint GS_R_JOINT_ACCELERATION = 2u;
static const uint GS_R_JOINT_FORCE = 3u;
static const uint GS_R_JOINT_TARGET_VELOCITY = 4u;
static const uint GS_R_JOINT_TARGET_POSITION = 5u;
static const uint GS_R_ROOT_GLOBAL_POSE = 6u;
static const uint GS_R_ROOT_LINEAR_VELOCITY = 7u;
static const uint GS_R_ROOT_ANGULAR_VELOCITY = 8u;
static const uint GS_R_LINK_LINEAR_VELOCITY = 10u;
static const uint GS_R_LINK_ANGULAR_VELOCITY = 11u;
static const uint GS_R_LINK_LINEAR_ACCELERATION = 12u;
static const uint GS_R_LINK_ANGULAR_ACCELERATION = 13u;
// PxArticulationGPUAPIWriteType
static const uint GS_W_JOINT_POSITION = 0u;
static const uint GS_W_JOINT_VELOCITY = 1u;
static const uint GS_W_JOINT_FORCE = 2u;
static const uint GS_W_JOINT_TARGET_VELOCITY = 3u;
static const uint GS_W_JOINT_TARGET_POSITION = 4u;
static const uint GS_W_ROOT_LINEAR_VELOCITY = 6u;
static const uint GS_W_ROOT_ANGULAR_VELOCITY = 7u;

struct Gs6Params
{
    uint64_t scDesc;   // @0
    uint64_t data;     // @8
    uint64_t idx;      // @16
    uint nbElements;   // @24
    uint maxN;         // @28
    uint typeWord;     // @32 (enum)
}

struct Gs5Params
{
    uint64_t scDesc;   // @0
    uint64_t data;     // @8
    uint64_t idx;      // @16
    uint nbElements;   // @24
    uint maxN;         // @28
}

struct Gs4Params
{
    uint64_t scDesc;   // @0
    uint64_t data;     // @8
    uint64_t idx;      // @16
    uint nbElements;   // @24
}

[[vk::push_constant]]
ConstantBuffer<Gs6Params> gGs6;

[[vk::push_constant]]
ConstantBuffer<Gs5Params> gGs5;

[[vk::push_constant]]
ConstantBuffer<Gs4Params> gGs4;

uint64_t gsRecord(uint64_t scDesc, uint64_t gpuIndices, uint groupIndex)
{
    const uint articulationIndex = (Ptr<uint>(gpuIndices))[groupIndex];
    const uint64_t articulations = (Ptr<uint64_t>(scDesc + PXG_ARTI_CORE_ARTICULATIONS))[0];
    return articulations + uint64_t(articulationIndex) * PXG_ARTI_STRIDE;
}

// ---------------------------------------------------------------------------

[shader("compute")]
[numthreads(512, 1, 1)]
void getArtiDofStates(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint globalThreadIndex = dispatchThreadId.x;
    const uint maxDofs = gGs6.maxN;
    const uint groupIndex = globalThreadIndex / maxDofs;
    const uint dofIndex = globalThreadIndex % maxDofs;

    if (groupIndex < gGs6.nbElements)
    {
        const uint64_t record = gsRecord(gGs6.scDesc, gGs6.idx, groupIndex);
        uint srcField = PXG_ARTI_JOINT_POSITIONS;   // trap 2: the READ order
        switch (gGs6.typeWord)
        {
        case GS_R_JOINT_POSITION:        srcField = PXG_ARTI_JOINT_POSITIONS; break;
        case GS_R_JOINT_VELOCITY:        srcField = PXG_ARTI_JOINT_VELOCITIES; break;
        case GS_R_JOINT_ACCELERATION:    srcField = PXG_ARTI_JOINT_ACCELERATIONS; break;
        case GS_R_JOINT_FORCE:           srcField = PXG_ARTI_JOINT_FORCE; break;
        case GS_R_JOINT_TARGET_VELOCITY: srcField = PXG_ARTI_JOINT_TARGET_VELOCITIES; break;
        case GS_R_JOINT_TARGET_POSITION: srcField = PXG_ARTI_JOINT_TARGET_POSITIONS; break;
        default: break;
        }
        Ptr<float> src = Ptr<float>(artiPtrField(record, srcField));
        Ptr<float> dst = Ptr<float>(gGs6.data + uint64_t(groupIndex) * maxDofs * 4u);
        const uint artiDofs = artiU16Field(record, PXG_ARTI_DATA_NUM_JOINT_DOFS);
        if (dofIndex < artiDofs)   // trap 1
            dst[dofIndex] = src[dofIndex];
    }
}

[shader("compute")]
[numthreads(512, 1, 1)]
void getArtiTransformStates(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint globalThreadIndex = dispatchThreadId.x;
    const uint maxLinks = gGs6.maxN;
    const uint groupIndex = globalThreadIndex / maxLinks;
    const uint linkIndex = globalThreadIndex % maxLinks;

    if (groupIndex < gGs6.nbElements)
    {
        const uint64_t record = gsRecord(gGs6.scDesc, gGs6.idx, groupIndex);
        const uint numLinks = (gGs6.typeWord == GS_R_ROOT_GLOBAL_POSE)
            ? 1u
            : artiU16Field(record, PXG_ARTI_DATA_NUM_LINKS);   // trap 4

        if (linkIndex < numLinks)
        {
            Ptr<float> ba = Ptr<float>(artiPtrField(record, PXG_ARTI_LINK_BODY2ACTORS)
                                       + uint64_t(linkIndex) * PXG_ARTI_TRANSFORM_SIZE);
            Ptr<float> bw = Ptr<float>(artiPtrField(record, PXG_ARTI_LINK_BODY2WORLDS)
                                       + uint64_t(linkIndex) * PXG_ARTI_TRANSFORM_SIZE);
            const float4 qa = float4(ba[0], ba[1], ba[2], ba[3]);
            const float3 pa = float3(ba[4], ba[5], ba[6]);
            const float4 qw = float4(bw[0], bw[1], bw[2], bw[3]);
            const float3 pw = float3(bw[4], bw[5], bw[6]);
            // body2World * body2Actor.getInverse(), the engine spelling (trap 4)
            const float4 invQ = quatConjugate(qa);
            const float3 invP = quatRotateInv(qa, -pa);
            const float4 q = quatMul(qw, invQ);
            const float3 p = quatRotate(qw, invP) + pw;
            Ptr<float> dst = Ptr<float>(gGs6.data
                                        + (uint64_t(groupIndex) * maxLinks + linkIndex)
                                              * PXG_ARTI_TRANSFORM_SIZE);
            dst[0] = q.x; dst[1] = q.y; dst[2] = q.z; dst[3] = q.w;
            dst[4] = p.x; dst[5] = p.y; dst[6] = p.z;
        }
    }
}

[shader("compute")]
[numthreads(480, 1, 1)]
void getArtiVelocityStates(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint globalThreadIndex = dispatchThreadId.x;
    const uint maxLinks = gGs6.maxN;
    const uint threadPerArticulation = 3u * maxLinks;
    const uint groupIndex = globalThreadIndex / threadPerArticulation;
    const uint localIndex = globalThreadIndex % threadPerArticulation;
    const uint linkIndex = localIndex / 3u;
    const uint elementIndex = localIndex % 3u;

    if (groupIndex < gGs6.nbElements)
    {
        const uint64_t record = gsRecord(gGs6.scDesc, gGs6.idx, groupIndex);
        const uint artiLinks = artiU16Field(record, PXG_ARTI_DATA_NUM_LINKS);
        uint64_t srcBase = 0uL;
        uint numLinks = 0u;
        switch (gGs6.typeWord)   // trap 6
        {
        case GS_R_ROOT_LINEAR_VELOCITY:
            srcBase = artiPtrField(record, PXG_ARTI_MOTION_VELOCITIES) + 12u; numLinks = 1u; break;
        case GS_R_ROOT_ANGULAR_VELOCITY:
            srcBase = artiPtrField(record, PXG_ARTI_MOTION_VELOCITIES); numLinks = 1u; break;
        case GS_R_LINK_LINEAR_VELOCITY:
            srcBase = artiPtrField(record, PXG_ARTI_MOTION_VELOCITIES) + 12u;
            numLinks = artiLinks; break;
        case GS_R_LINK_ANGULAR_VELOCITY:
            srcBase = artiPtrField(record, PXG_ARTI_MOTION_VELOCITIES);
            numLinks = artiLinks; break;
        case GS_R_LINK_LINEAR_ACCELERATION:
            srcBase = artiPtrField(record, PXG_ARTI_MOTION_ACCELERATIONS) + 12u;
            numLinks = artiLinks; break;
        case GS_R_LINK_ANGULAR_ACCELERATION:
            srcBase = artiPtrField(record, PXG_ARTI_MOTION_ACCELERATIONS);
            numLinks = artiLinks; break;
        default: break;
        }
        if (linkIndex < numLinks)
        {
            Ptr<float> dst = Ptr<float>(gGs6.data
                                        + (uint64_t(groupIndex) * maxLinks + linkIndex) * 12u);
            Ptr<float> src = Ptr<float>(srcBase + uint64_t(linkIndex) * 24u);   // stride 6 floats
            dst[elementIndex] = src[elementIndex];
        }
    }
}

[shader("compute")]
[numthreads(512, 1, 1)]
void getArtiSpatialForceStates(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint globalThreadIndex = dispatchThreadId.x;
    const uint maxLinks = gGs5.maxN;
    const uint threadPerArticulation = 6u * maxLinks;
    const uint groupIndex = globalThreadIndex / threadPerArticulation;
    const uint localIndex = globalThreadIndex % threadPerArticulation;
    const uint linkIndex = localIndex / 6u;
    const uint elementIndex = localIndex % 6u;

    if (groupIndex < gGs5.nbElements)
    {
        const uint64_t record = gsRecord(gGs5.scDesc, gGs5.idx, groupIndex);
        const uint numLinks = artiU16Field(record, PXG_ARTI_DATA_NUM_LINKS);
        if (linkIndex < numLinks)
        {
            Ptr<float> dst = Ptr<float>(gGs5.data
                                        + (uint64_t(groupIndex) * threadPerArticulation
                                           + linkIndex * 6u) * 4u);
            Ptr<float> src = Ptr<float>(artiPtrField(record, PXG_ARTI_LINK_INCOMING_JOINT_FORCES)
                                        + uint64_t(linkIndex) * 24u);
            dst[elementIndex] = src[elementIndex];
        }
    }
}

// ---------------------------------------------------------------------------

[shader("compute")]
[numthreads(512, 1, 1)]
void setArtiDofStates(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint globalThreadIndex = dispatchThreadId.x;
    const uint maxDofs = gGs6.maxN;
    const uint groupIndex = globalThreadIndex / maxDofs;
    const uint dofIndex = globalThreadIndex % maxDofs;

    if (groupIndex < gGs6.nbElements)
    {
        const uint64_t record = gsRecord(gGs6.scDesc, gGs6.idx, groupIndex);
        Ptr<uint> dirty = Ptr<uint>(record + PXG_ARTI_DATA_gpuDirtyFlag);
        uint dstField = PXG_ARTI_JOINT_POSITIONS;   // trap 2: the WRITE order
        switch (gGs6.typeWord)
        {
        case GS_W_JOINT_POSITION:
            dstField = PXG_ARTI_JOINT_POSITIONS;
            if (dofIndex == 0u)   // trap 3
                dirty[0] = dirty[0] | PXG_ARTI_DIRTY_POSITIONS
                           | PXG_ARTI_DIRTY_NEEDS_KINEMATIC_UPDATE;
            break;
        case GS_W_JOINT_VELOCITY:
            dstField = PXG_ARTI_JOINT_VELOCITIES;
            if (dofIndex == 0u)
                dirty[0] = dirty[0] | PXG_ARTI_DIRTY_VELOCITIES
                           | PXG_ARTI_DIRTY_NEEDS_KINEMATIC_UPDATE;
            break;
        case GS_W_JOINT_FORCE:
            dstField = PXG_ARTI_JOINT_FORCE;
            if (dofIndex == 0u)
                dirty[0] = dirty[0] | PXG_ARTI_DIRTY_FORCES;
            break;
        case GS_W_JOINT_TARGET_VELOCITY:
            dstField = PXG_ARTI_JOINT_TARGET_VELOCITIES;
            if (dofIndex == 0u)
                dirty[0] = dirty[0] | PXG_ARTI_DIRTY_JOINT_TARGET_VEL;
            break;
        case GS_W_JOINT_TARGET_POSITION:
            dstField = PXG_ARTI_JOINT_TARGET_POSITIONS;
            if (dofIndex == 0u)
                dirty[0] = dirty[0] | PXG_ARTI_DIRTY_JOINT_TARGET_POS;
            break;
        default: break;
        }
        Ptr<float> dst = Ptr<float>(artiPtrField(record, dstField));
        Ptr<float> src = Ptr<float>(gGs6.data + uint64_t(groupIndex) * maxDofs * 4u);
        const uint artiDofs = artiU16Field(record, PXG_ARTI_DATA_NUM_JOINT_DOFS);
        if (dofIndex < artiDofs)   // trap 1
            dst[dofIndex] = src[dofIndex];
    }
}

[shader("compute")]
[numthreads(512, 1, 1)]
void setArtiRootGlobalPoseState(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint globalThreadIndex = dispatchThreadId.x;
    if (globalThreadIndex < gGs4.nbElements)
    {
        const uint64_t record = gsRecord(gGs4.scDesc, gGs4.idx, globalThreadIndex);
        Ptr<uint> dirty = Ptr<uint>(record + PXG_ARTI_DATA_gpuDirtyFlag);
        dirty[0] = dirty[0] | PXG_ARTI_DIRTY_ROOT_TRANSFORM
                   | PXG_ARTI_DIRTY_NEEDS_KINEMATIC_UPDATE;

        Ptr<float> src = Ptr<float>(gGs4.data
                                    + uint64_t(globalThreadIndex) * PXG_ARTI_TRANSFORM_SIZE);
        const float4 aq = float4(src[0], src[1], src[2], src[3]);
        const float3 ap = float3(src[4], src[5], src[6]);
        Ptr<float> ba = Ptr<float>(artiPtrField(record, PXG_ARTI_LINK_BODY2ACTORS));
        const float4 qa = float4(ba[0], ba[1], ba[2], ba[3]);
        const float3 pa = float3(ba[4], ba[5], ba[6]);
        // pose = actorPose * body2Actor -- NOT the inverse (trap 5)
        const float4 q = quatMul(aq, qa);
        const float3 p = quatRotate(aq, pa) + ap;
        Ptr<float> bw = Ptr<float>(artiPtrField(record, PXG_ARTI_LINK_BODY2WORLDS));
        bw[0] = q.x; bw[1] = q.y; bw[2] = q.z; bw[3] = q.w;
        bw[4] = p.x; bw[5] = p.y; bw[6] = p.z;
    }
}

[shader("compute")]
[numthreads(480, 1, 1)]
void setArtiRootVelocityState(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint globalThreadIndex = dispatchThreadId.x;
    const uint artiIndex = globalThreadIndex / 3u;
    const uint localIndex = globalThreadIndex % 3u;

    if (artiIndex < gGs5.nbElements)
    {
        const uint64_t record = gsRecord(gGs5.scDesc, gGs5.idx, artiIndex);
        if (localIndex == 0u)   // trap 3
        {
            Ptr<uint> dirty = Ptr<uint>(record + PXG_ARTI_DATA_gpuDirtyFlag);
            dirty[0] = dirty[0] | PXG_ARTI_DIRTY_ROOT_VELOCITIES
                       | PXG_ARTI_DIRTY_NEEDS_KINEMATIC_UPDATE;
        }
        // gGs5.maxN carries the enum for this signature (the 5th CUDA param)
        const uint64_t mvel = artiPtrField(record, PXG_ARTI_MOTION_VELOCITIES);
        const uint64_t dstBase =
            (gGs5.maxN == GS_W_ROOT_LINEAR_VELOCITY) ? (mvel + 12u) : mvel;   // trap 6
        Ptr<float> dst = Ptr<float>(dstBase);
        Ptr<float> src = Ptr<float>(gGs5.data + uint64_t(artiIndex) * 12u);
        dst[localIndex] = src[localIndex];
    }
}

[shader("compute")]
[numthreads(480, 1, 1)]
void setArtiLinkForceState(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint globalThreadIndex = dispatchThreadId.x;
    const uint maxLinks = gGs5.maxN;
    const uint threadPerArticulation = 3u * maxLinks;
    const uint artiSourceIndex = globalThreadIndex / threadPerArticulation;
    const uint localIndex = globalThreadIndex % threadPerArticulation;
    const uint linkIndex = localIndex / 3u;
    const uint elementIndex = localIndex % 3u;

    if (artiSourceIndex < gGs5.nbElements)
    {
        const uint64_t record = gsRecord(gGs5.scDesc, gGs5.idx, artiSourceIndex);
        const uint numLinks = artiU16Field(record, PXG_ARTI_DATA_NUM_LINKS);
        if (linkIndex < numLinks)
        {
            const float invMass =
                (Ptr<float>(artiPtrField(record, PXG_ARTI_LINK_PROPS)
                            + uint64_t(linkIndex) * 16u))[3];
            Ptr<float> src = Ptr<float>(gGs5.data
                                        + (uint64_t(artiSourceIndex) * maxLinks + linkIndex)
                                              * 12u);
            Ptr<float> dst = Ptr<float>(artiPtrField(record, PXG_ARTI_EXTERNAL_ACCELERATIONS)
                                        + uint64_t(linkIndex) * 24u);   // trap 7: .top
            dst[elementIndex] = src[elementIndex] * invMass;
        }
    }
}

[shader("compute")]
[numthreads(512, 1, 1)]
void setArtiLinkTorqueState(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint globalThreadIndex = dispatchThreadId.x;
    const uint maxLinks = gGs5.maxN;
    const uint groupIndex = globalThreadIndex / maxLinks;
    const uint linkIndex = globalThreadIndex % maxLinks;

    if (groupIndex < gGs5.nbElements)
    {
        const uint64_t record = gsRecord(gGs5.scDesc, gGs5.idx, groupIndex);
        const uint numLinks = artiU16Field(record, PXG_ARTI_DATA_NUM_LINKS);
        if (linkIndex < numLinks)
        {
            Ptr<float> bw = Ptr<float>(artiPtrField(record, PXG_ARTI_LINK_BODY2WORLDS)
                                       + uint64_t(linkIndex) * PXG_ARTI_TRANSFORM_SIZE);
            const float4 q = float4(bw[0], bw[1], bw[2], bw[3]);
            Ptr<float> src = Ptr<float>(gGs5.data
                                        + (uint64_t(groupIndex) * maxLinks + linkIndex) * 12u);
            const float3 linkTorque = float3(src[0], src[1], src[2]);
            const float3 localLinkTorque = quatRotateInv(q, linkTorque);   // trap 7
            Ptr<float> pp = Ptr<float>(artiPtrField(record, PXG_ARTI_LINK_PROPS)
                                       + uint64_t(linkIndex) * 16u);
            const float3 invInertia = float3(pp[0], pp[1], pp[2]);
            const float3 localAccel = invInertia * localLinkTorque;
            const float3 worldAccel = quatRotate(q, localAccel);
            Ptr<float> dst = Ptr<float>(artiPtrField(record, PXG_ARTI_EXTERNAL_ACCELERATIONS)
                                        + uint64_t(linkIndex) * 24u);
            dst[3] = worldAccel.x; dst[4] = worldAccel.y; dst[5] = worldAccel.z;   // .bottom
        }
    }
}

// ===========================================================================
// P8.119: the six tendon get/set kernels (articulationDirectGpuApi.cu:
// 822-1110).  Same launch family (512-wide blocks, one thread per element).
//
// TRAPS, each a sweep mutant unless marked:
//  8. setArtiTendonState raises the dirty bit at tendonIndex == 0 OUTSIDE
//     the tendonIndex < numTendons guard -- an articulation with ZERO
//     tendons of the written kind still gets its flag set.  Spatial and
//     fixed params have different strides (16 vs 32 bytes) and different
//     bits (1<<14 vs 1<<16).
//  9. The attachment / tendon-joint kernels read maxAttachments /
//     maxSpatialTendons / maxFixedTendons / maxTendonJoints from the
//     DESCRIPTOR (not from the kernel parameter), and their per-job source
//     pitch is desc.maxSpatialTendons * desc.maxAttachments.
//
// NINTH UPSTREAM-BUG CANDIDATE (transcribed verbatim, both sides): the
// attachment and tendon-joint copies index the USER buffer by
// attachmentIndex / tendonJointIndex ALONE -- the tendonIndex never enters
// the offset (:949/:997/:1049/:1095).  With more than one tendon per
// articulation the setter feeds every tendon from tendon 0's block of the
// user buffer, and the getter has a cross-tendon WRITE RACE into the same
// output slots.  The fixture keeps multi-tendon articulations' mod-element
// blocks IDENTICAL across tendons so the transcribed behavior stays
// deterministic, and documents the candidate instead of "fixing" it.
// ===========================================================================

[shader("compute")]
[numthreads(512, 1, 1)]
void getArtiTendonState(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint globalThreadIndex = dispatchThreadId.x;
    const uint maxTendons = gGs6.maxN;
    const uint groupIndex = globalThreadIndex / maxTendons;
    const uint tendonIndex = globalThreadIndex % maxTendons;

    if (groupIndex < gGs6.nbElements)
    {
        const uint64_t record = gsRecord(gGs6.scDesc, gGs6.idx, groupIndex);
        if (gGs6.typeWord == 17u)   // PxArticulationGPUAPIReadType::eSPATIAL_TENDON
        {
            const uint num = artiU16Field(record, PXG_ARTI_DATA_NUM_SPATIAL_TENDONS);
            if (tendonIndex < num)
            {
                Ptr<uint4> dst = Ptr<uint4>(gGs6.data
                                            + (uint64_t(groupIndex) * maxTendons + tendonIndex)
                                                  * PXG_ARTI_SPATIAL_TENDON_DATA_SIZE);
                Ptr<uint4> src = Ptr<uint4>(artiPtrField(record, PXG_ARTI_SPATIAL_TENDON_PARAMS)
                                            + uint64_t(tendonIndex)
                                                  * PXG_ARTI_SPATIAL_TENDON_DATA_SIZE);
                dst[0] = src[0];
            }
        }
        else if (gGs6.typeWord == 15u)   // eFIXED_TENDON
        {
            const uint num = artiU16Field(record, PXG_ARTI_DATA_NUM_FIXED_TENDONS);
            if (tendonIndex < num)
            {
                Ptr<uint4> dst = Ptr<uint4>(gGs6.data
                                            + (uint64_t(groupIndex) * maxTendons + tendonIndex)
                                                  * PXG_ARTI_FIXED_TENDON_DATA_SIZE);
                Ptr<uint4> src = Ptr<uint4>(artiPtrField(record, PXG_ARTI_FIXED_TENDON_PARAMS)
                                            + uint64_t(tendonIndex)
                                                  * PXG_ARTI_FIXED_TENDON_DATA_SIZE);
                dst[0] = src[0];
                dst[1] = src[1];
            }
        }
    }
}

[shader("compute")]
[numthreads(512, 1, 1)]
void setArtiTendonState(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint globalThreadIndex = dispatchThreadId.x;
    const uint maxTendons = gGs6.maxN;
    const uint groupIndex = globalThreadIndex / maxTendons;
    const uint tendonIndex = globalThreadIndex % maxTendons;

    if (groupIndex < gGs6.nbElements)
    {
        const uint64_t record = gsRecord(gGs6.scDesc, gGs6.idx, groupIndex);
        Ptr<uint> dirty = Ptr<uint>(record + PXG_ARTI_DATA_gpuDirtyFlag);
        if (gGs6.typeWord == 12u)   // PxArticulationGPUAPIWriteType::eSPATIAL_TENDON
        {
            const uint num = artiU16Field(record, PXG_ARTI_DATA_NUM_SPATIAL_TENDONS);
            if (tendonIndex < num)
            {
                Ptr<uint4> src = Ptr<uint4>(gGs6.data
                                            + (uint64_t(groupIndex) * maxTendons + tendonIndex)
                                                  * PXG_ARTI_SPATIAL_TENDON_DATA_SIZE);
                Ptr<uint4> dst = Ptr<uint4>(artiPtrField(record, PXG_ARTI_SPATIAL_TENDON_PARAMS)
                                            + uint64_t(tendonIndex)
                                                  * PXG_ARTI_SPATIAL_TENDON_DATA_SIZE);
                dst[0] = src[0];
            }
            if (tendonIndex == 0u)   // trap 8: OUTSIDE the count guard
                dirty[0] = dirty[0] | PXG_ARTI_DIRTY_SPATIAL_TENDON;
        }
        else if (gGs6.typeWord == 10u)   // eFIXED_TENDON
        {
            const uint num = artiU16Field(record, PXG_ARTI_DATA_NUM_FIXED_TENDONS);
            if (tendonIndex < num)
            {
                Ptr<uint4> src = Ptr<uint4>(gGs6.data
                                            + (uint64_t(groupIndex) * maxTendons + tendonIndex)
                                                  * PXG_ARTI_FIXED_TENDON_DATA_SIZE);
                Ptr<uint4> dst = Ptr<uint4>(artiPtrField(record, PXG_ARTI_FIXED_TENDON_PARAMS)
                                            + uint64_t(tendonIndex)
                                                  * PXG_ARTI_FIXED_TENDON_DATA_SIZE);
                dst[0] = src[0];
                dst[1] = src[1];
            }
            if (tendonIndex == 0u)
                dirty[0] = dirty[0] | PXG_ARTI_DIRTY_FIXED_TENDON;
        }
    }
}

[shader("compute")]
[numthreads(512, 1, 1)]
void getArtiSpatialTendonAttachmentState(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint maxTendonAttachments =
        (Ptr<uint>(gGs5.scDesc + PXG_ARTI_CORE_MAX_ATTACHMENTS))[0];   // trap 9
    const uint maxSpatialTendons =
        (Ptr<uint>(gGs5.scDesc + PXG_ARTI_CORE_MAX_SPATIAL_TENDONS))[0];

    const uint globalThreadIndex = dispatchThreadId.x;
    const uint maxTendons = gGs5.maxN;
    const uint groupIndex = globalThreadIndex / maxTendons;
    const uint elementIndex = globalThreadIndex % maxTendons;
    const uint tendonIndex = elementIndex / maxTendonAttachments;
    const uint attachmentIndex = elementIndex % maxTendonAttachments;

    if (groupIndex < gGs5.nbElements)
    {
        const uint64_t record = gsRecord(gGs5.scDesc, gGs5.idx, groupIndex);
        const uint numSpatialTendons = artiU16Field(record, PXG_ARTI_DATA_NUM_SPATIAL_TENDONS);
        if (tendonIndex < numSpatialTendons)
        {
            const uint64_t tendon = artiPtrField(record, PXG_ARTI_SPATIAL_TENDONS)
                                    + uint64_t(tendonIndex) * PXG_ARTI_TENDON_SIZE;
            const uint64_t modElements = (Ptr<uint64_t>(tendon + PXG_TENDON_MOD_ELEMENTS))[0];
            const uint numTendonAttachments = (Ptr<uint>(tendon + PXG_TENDON_NB_ELEMENTS))[0];
            if (attachmentIndex < numTendonAttachments)
            {
                // NINTH CANDIDATE: no tendonIndex in the user-buffer offset
                Ptr<uint4> dst = Ptr<uint4>(gGs5.data
                                            + (uint64_t(groupIndex) * maxSpatialTendons
                                                   * maxTendonAttachments
                                               + attachmentIndex) * 32u);
                Ptr<uint4> src = Ptr<uint4>(modElements + uint64_t(attachmentIndex) * 32u);
                dst[0] = src[0];
                dst[1] = src[1];
            }
        }
    }
}

[shader("compute")]
[numthreads(512, 1, 1)]
void setArtiSpatialTendonAttachmentState(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint maxTendonAttachments =
        (Ptr<uint>(gGs5.scDesc + PXG_ARTI_CORE_MAX_ATTACHMENTS))[0];   // trap 9
    const uint maxSpatialTendons =
        (Ptr<uint>(gGs5.scDesc + PXG_ARTI_CORE_MAX_SPATIAL_TENDONS))[0];

    const uint globalThreadIndex = dispatchThreadId.x;
    const uint maxTendons = gGs5.maxN;
    const uint groupIndex = globalThreadIndex / maxTendons;
    const uint elementIndex = globalThreadIndex % maxTendons;
    const uint tendonIndex = elementIndex / maxTendonAttachments;
    const uint attachmentIndex = elementIndex % maxTendonAttachments;

    if (groupIndex < gGs5.nbElements)
    {
        const uint64_t record = gsRecord(gGs5.scDesc, gGs5.idx, groupIndex);
        if (elementIndex == 0u)   // trap 8's shape: outside the tendon guard
        {
            Ptr<uint> dirty = Ptr<uint>(record + PXG_ARTI_DATA_gpuDirtyFlag);
            dirty[0] = dirty[0] | PXG_ARTI_DIRTY_SPATIAL_TENDON_ATTACHMENT;
        }
        const uint numSpatialTendons = artiU16Field(record, PXG_ARTI_DATA_NUM_SPATIAL_TENDONS);
        if (tendonIndex < numSpatialTendons)
        {
            const uint64_t tendon = artiPtrField(record, PXG_ARTI_SPATIAL_TENDONS)
                                    + uint64_t(tendonIndex) * PXG_ARTI_TENDON_SIZE;
            const uint64_t modElements = (Ptr<uint64_t>(tendon + PXG_TENDON_MOD_ELEMENTS))[0];
            const uint numTendonAttachments = (Ptr<uint>(tendon + PXG_TENDON_NB_ELEMENTS))[0];
            if (attachmentIndex < numTendonAttachments)
            {
                // NINTH CANDIDATE: no tendonIndex in the user-buffer offset
                Ptr<uint4> src = Ptr<uint4>(gGs5.data
                                            + (uint64_t(groupIndex) * maxSpatialTendons
                                                   * maxTendonAttachments
                                               + attachmentIndex) * 32u);
                Ptr<uint4> dst = Ptr<uint4>(modElements + uint64_t(attachmentIndex) * 32u);
                dst[0] = src[0];
                dst[1] = src[1];
            }
        }
    }
}

[shader("compute")]
[numthreads(512, 1, 1)]
void getArtiFixedTendonJointState(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint maxFixedTendons =
        (Ptr<uint>(gGs4.scDesc + PXG_ARTI_CORE_MAX_FIXED_TENDONS))[0];   // trap 9
    const uint maxTendonJoints =
        (Ptr<uint>(gGs4.scDesc + PXG_ARTI_CORE_MAX_TENDON_JOINTS))[0];
    const uint maxTendons = maxFixedTendons * maxTendonJoints;

    const uint globalThreadIndex = dispatchThreadId.x;
    const uint groupIndex = globalThreadIndex / maxTendons;
    const uint elementIndex = globalThreadIndex % maxTendons;
    const uint tendonIndex = elementIndex / maxTendonJoints;
    const uint tendonJointIndex = elementIndex % maxTendonJoints;

    if (groupIndex < gGs4.nbElements)
    {
        const uint64_t record = gsRecord(gGs4.scDesc, gGs4.idx, groupIndex);
        const uint numFixedTendons = artiU16Field(record, PXG_ARTI_DATA_NUM_FIXED_TENDONS);
        if (tendonIndex < numFixedTendons)
        {
            const uint64_t tendon = artiPtrField(record, PXG_ARTI_FIXED_TENDONS)
                                    + uint64_t(tendonIndex) * PXG_ARTI_TENDON_SIZE;
            const uint64_t modElements = (Ptr<uint64_t>(tendon + PXG_TENDON_MOD_ELEMENTS))[0];
            const uint numTendonJoints = (Ptr<uint>(tendon + PXG_TENDON_NB_ELEMENTS))[0];
            if (tendonJointIndex < numTendonJoints)
            {
                // NINTH CANDIDATE: no tendonIndex in the user-buffer offset
                Ptr<uint4> dst = Ptr<uint4>(gGs4.data
                                            + (uint64_t(groupIndex) * maxTendons
                                               + tendonJointIndex) * 16u);
                Ptr<uint4> src = Ptr<uint4>(modElements + uint64_t(tendonJointIndex) * 16u);
                dst[0] = src[0];
            }
        }
    }
}

[shader("compute")]
[numthreads(512, 1, 1)]
void setArtiFixedTendonJointState(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint maxFixedTendons =
        (Ptr<uint>(gGs4.scDesc + PXG_ARTI_CORE_MAX_FIXED_TENDONS))[0];   // trap 9
    const uint maxTendonJoints =
        (Ptr<uint>(gGs4.scDesc + PXG_ARTI_CORE_MAX_TENDON_JOINTS))[0];
    const uint maxTendons = maxFixedTendons * maxTendonJoints;

    const uint globalThreadIndex = dispatchThreadId.x;
    const uint groupIndex = globalThreadIndex / maxTendons;
    const uint elementIndex = globalThreadIndex % maxTendons;
    const uint tendonIndex = elementIndex / maxTendonJoints;
    const uint tendonJointIndex = elementIndex % maxTendonJoints;

    if (groupIndex < gGs4.nbElements)
    {
        const uint64_t record = gsRecord(gGs4.scDesc, gGs4.idx, groupIndex);
        if (elementIndex == 0u)   // trap 8's shape: outside the tendon guard
        {
            Ptr<uint> dirty = Ptr<uint>(record + PXG_ARTI_DATA_gpuDirtyFlag);
            dirty[0] = dirty[0] | PXG_ARTI_DIRTY_FIXED_TENDON_JOINT;
        }
        const uint numFixedTendons = artiU16Field(record, PXG_ARTI_DATA_NUM_FIXED_TENDONS);
        if (tendonIndex < numFixedTendons)
        {
            const uint64_t tendon = artiPtrField(record, PXG_ARTI_FIXED_TENDONS)
                                    + uint64_t(tendonIndex) * PXG_ARTI_TENDON_SIZE;
            const uint64_t modElements = (Ptr<uint64_t>(tendon + PXG_TENDON_MOD_ELEMENTS))[0];
            const uint numTendonJoints = (Ptr<uint>(tendon + PXG_TENDON_NB_ELEMENTS))[0];
            if (tendonJointIndex < numTendonJoints)
            {
                // NINTH CANDIDATE: no tendonIndex in the user-buffer offset
                Ptr<uint4> src = Ptr<uint4>(gGs4.data
                                            + (uint64_t(groupIndex) * maxTendons
                                               + tendonJointIndex) * 16u);
                Ptr<uint4> dst = Ptr<uint4>(modElements + uint64_t(tendonJointIndex) * 16u);
                dst[0] = src[0];
            }
        }
    }
}
