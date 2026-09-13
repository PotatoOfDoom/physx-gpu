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

// directApi.slang
//
// Slang port of the Direct GPU API rigid-body getter/setter kernels from
// gpusimulationcontroller/src/CUDA/updateBodiesAndShapes.cu:
//   - getRigidDynamicGlobalPose / getRigidDynamicLinearVelocity /
//     getRigidDynamicAngularVelocity
//   - getRigidDynamicLinearAcceleration / getRigidDynamicAngularAcceleration
//   - setRigidDynamicLinearVelocity / setRigidDynamicAngularVelocity
//   - setRigidDynamicForce / setRigidDynamicTorque
//
// The PxVec3/PxTransform user arrays are tightly packed (12/28 bytes per
// element) and therefore bound as raw float buffers.
// setRigidDynamicGlobalPose (which also refreshes bounds/transform cache) lives
// in updateBounds.slang next to the bounds machinery it shares.
//
// P9.79 -- A CORRECTION.  This header used to claim "copyUserData maps to the
// already-ported MemCopyBalanced".  That was FALSE and worth saying so plainly:
// copyUserData is a DIFFERENT CUDA kernel (updateBodiesAndShapes.cu:1280), with a
// different argument pack ({8,4} over PxgPtrPair, not MemCopyBalanced's CopyDesc),
// a different launch site, and its own unmapped manifest row.  Nothing maps it.
//
// What is TRUE about it, and is a measurement rather than a claim: its only launch
// site is PxgMemCopyDispatcher::flushCommands, and PxgMemCopyDispatcher is
// CONSTRUCTED NOWHERE in this tree -- not as a member, not as a local, not via
// PX_NEW.  The class compiles and is dead.  So the kernel is unreachable here for
// a reason no existing bucket names: not orphaned (a call site exists), not
// standalone (no app factory gates it), not feature-gated -- its OWNER is never
// built.  Recorded in tools/engine_reachable_unmapped.txt.
//
// P9.79 also moved the three GETTERS below onto the engine launch ABI, in
// shaders/directApiBda.slang.  The entries here keep the Phase-1 bound-buffer form
// and the testDirectApi fixture that drives it; the MANIFEST points the engine at
// the BDA twins.  Same split as radixSortGather.slang / femContactSortBda.slang.

import pxgSlangCommon;

static const uint BLOCK_SIZE = 256;

// PxgBodySim word layout (see bodyUpdate.slang)
static const uint BODYSIM_WORDS = 56;
static const uint W_LINVEL = 0;
static const uint W_ANGVEL = 4;
static const uint W_INV_INERTIA = 12;
static const uint W_BODY2WORLD = 28;
static const uint W_BODY2ACTOR = 36;
static const uint W_EXTERNAL_ACCEL = 48;

struct DirectApiParams
{
    uint nbElements;
    float oneOverDt;
    uint hasPrevVelocities;
};

[[vk::push_constant]]
ConstantBuffer<DirectApiParams> gParams;

RWStructuredBuffer<float> gData;           // binding 0: user data (PxVec3 / PxTransform, tightly packed)
RWStructuredBuffer<uint> gGpuIndices;      // binding 1: PxRigidDynamicGPUIndex
RWStructuredBuffer<uint> gBodySimPool;     // binding 2: PxgBodySim pool (word-addressed)
RWStructuredBuffer<float4> gPrevVelocities;// binding 3: PxgBodySimVelocities (2 float4 per body)

float4 loadF4(uint word)
{
    return float4(asfloat(gBodySimPool[word]), asfloat(gBodySimPool[word + 1]),
                  asfloat(gBodySimPool[word + 2]), asfloat(gBodySimPool[word + 3]));
}

void storeF4(uint word, float4 v)
{
    gBodySimPool[word] = asuint(v.x);
    gBodySimPool[word + 1] = asuint(v.y);
    gBodySimPool[word + 2] = asuint(v.z);
    gBodySimPool[word + 3] = asuint(v.w);
}

float3 loadUserVec3(uint index)
{
    return float3(gData[index * 3], gData[index * 3 + 1], gData[index * 3 + 2]);
}

void storeUserVec3(uint index, float3 v)
{
    gData[index * 3] = v.x;
    gData[index * 3 + 1] = v.y;
    gData[index * 3 + 2] = v.z;
}

// mirrors getRigidDynamicGlobalPose(): actor pose = body2World * body2Actor^-1
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void getRigidDynamicGlobalPose(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint i = dispatchThreadId.x;
    if (i >= gParams.nbElements)
        return;

    const uint body = gGpuIndices[i] * BODYSIM_WORDS;

    const float4 qW = loadF4(body + W_BODY2WORLD);
    const float4 pW = loadF4(body + W_BODY2WORLD + 4);
    const float4 qA = loadF4(body + W_BODY2ACTOR);
    const float4 pA = loadF4(body + W_BODY2ACTOR + 4);

    // PxAlignedTransform::getInverse() then transform(): see AlignedTransform.h
    const float4 qInv = quatConjugate(qA);
    const float3 pInv = quatRotateInv(qA, -pA.xyz);

    const float4 q = quatMul(qW, qInv);
    const float3 p = quatRotate(qW, pInv) + pW.xyz;

    // PxTransform output: 7 tightly packed floats (q, p)
    gData[i * 7 + 0] = q.x;
    gData[i * 7 + 1] = q.y;
    gData[i * 7 + 2] = q.z;
    gData[i * 7 + 3] = q.w;
    gData[i * 7 + 4] = p.x;
    gData[i * 7 + 5] = p.y;
    gData[i * 7 + 6] = p.z;
}

// mirrors getRigidDynamicLinearVelocity()
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void getRigidDynamicLinearVelocity(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint i = dispatchThreadId.x;
    if (i >= gParams.nbElements)
        return;

    const uint body = gGpuIndices[i] * BODYSIM_WORDS;
    storeUserVec3(i, loadF4(body + W_LINVEL).xyz);
}

// mirrors getRigidDynamicAngularVelocity()
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void getRigidDynamicAngularVelocity(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint i = dispatchThreadId.x;
    if (i >= gParams.nbElements)
        return;

    const uint body = gGpuIndices[i] * BODYSIM_WORDS;
    storeUserVec3(i, loadF4(body + W_ANGVEL).xyz);
}

// mirrors getRigidDynamicLinearAcceleration() (eENABLE_BODY_ACCELERATIONS)
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void getRigidDynamicLinearAcceleration(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint i = dispatchThreadId.x;
    if (i >= gParams.nbElements)
        return;

    const uint index = gGpuIndices[i];
    const float4 linVel = loadF4(index * BODYSIM_WORDS + W_LINVEL);
    const float4 deltaLinVel = linVel - gPrevVelocities[index * 2 + 0];
    storeUserVec3(i, deltaLinVel.xyz * gParams.oneOverDt);
}

// mirrors getRigidDynamicAngularAcceleration()
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void getRigidDynamicAngularAcceleration(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint i = dispatchThreadId.x;
    if (i >= gParams.nbElements)
        return;

    const uint index = gGpuIndices[i];
    const float4 angVel = loadF4(index * BODYSIM_WORDS + W_ANGVEL);
    const float4 deltaAngVel = angVel - gPrevVelocities[index * 2 + 1];
    storeUserVec3(i, deltaAngVel.xyz * gParams.oneOverDt);
}

// mirrors setRigidDynamicLinearVelocity()
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void setRigidDynamicLinearVelocity(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint i = dispatchThreadId.x;
    if (i >= gParams.nbElements)
        return;

    const uint index = gGpuIndices[i];
    const uint body = index * BODYSIM_WORDS;

    const float invMass = asfloat(gBodySimPool[body + W_LINVEL + 3]);
    const float4 lv = float4(loadUserVec3(i), invMass);
    storeF4(body + W_LINVEL, lv);
    if (gParams.hasPrevVelocities != 0)
        gPrevVelocities[index * 2 + 0] = lv;
}

// mirrors setRigidDynamicAngularVelocity()
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void setRigidDynamicAngularVelocity(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint i = dispatchThreadId.x;
    if (i >= gParams.nbElements)
        return;

    const uint index = gGpuIndices[i];
    const uint body = index * BODYSIM_WORDS;

    const float maxPenBias = asfloat(gBodySimPool[body + W_ANGVEL + 3]);
    const float4 av = float4(loadUserVec3(i), maxPenBias);
    storeF4(body + W_ANGVEL, av);
    if (gParams.hasPrevVelocities != 0)
        gPrevVelocities[index * 2 + 1] = av;
}

// mirrors setRigidDynamicForce(): externalLinearAcceleration = force * invMass
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void setRigidDynamicForce(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint i = dispatchThreadId.x;
    if (i >= gParams.nbElements)
        return;

    const uint body = gGpuIndices[i] * BODYSIM_WORDS;

    const float invMass = asfloat(gBodySimPool[body + W_LINVEL + 3]);
    const float3 deltaLin = loadUserVec3(i) * invMass;
    storeF4(body + W_EXTERNAL_ACCEL, float4(deltaLin, 0.0f));
}

// mirrors setRigidDynamicTorque(): externalAngularAcceleration =
// (R * diag(invInertia) * R^T) * torque
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void setRigidDynamicTorque(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint i = dispatchThreadId.x;
    if (i >= gParams.nbElements)
        return;

    const uint body = gGpuIndices[i] * BODYSIM_WORDS;

    const float4 invInertia = loadF4(body + W_INV_INERTIA);
    const float4 qW = loadF4(body + W_BODY2WORLD);

    const PxgMat33 inverseInertiaWorldSpace = transformInertiaTensor(invInertia.xyz, mat33FromQuat(qW));

    const float3 deltaAng = mat33Transform(inverseInertiaWorldSpace, loadUserVec3(i));
    storeF4(body + W_EXTERNAL_ACCEL + 4, float4(deltaAng, 0.0f));
}
