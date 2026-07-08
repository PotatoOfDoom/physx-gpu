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

// bodyUpdate.slang
//
// Slang port of the CPU->GPU rigid-body/shape mirror kernels from
// gpusimulationcontroller/src/CUDA/updateBodiesAndShapes.cu:
//   - updateBodiesLaunch
//   - updateBodiesLaunchDirectAPI
//   - updateShapesLaunch
//   - updateBodyExternalVelocitiesLaunch
//
// The CUDA kernels spread each 224-byte PxgBodySim across 16 threads and use
// warp shuffles to broadcast the embedded body index; the port copies one
// body per thread (the shuffle scheme is a memory-coalescing optimization
// with no semantic effect).  Buffers are bound word-addressed so partial
// float4 updates (the .w-only velocity lane merge of the direct-API path)
// stay race-free.

import pxgSlangCommon;

static const uint BLOCK_SIZE = 256;

// PxgBodySim word layout (224 bytes = 56 words)
static const uint BODYSIM_WORDS = 56;
static const uint W_LINVEL = 0;             // linearVelocityXYZ_inverseMassW
static const uint W_ANGVEL = 4;             // angularVelocityXYZ_maxPenBiasW
static const uint W_BODYSIM_INDEX = 27;     // freezeThreshold..._bodySimIndex.w
static const uint W_BODY2WORLD = 28;        // PxAlignedTransform (8 words)
static const uint W_BODY2ACTOR = 36;        // PxAlignedTransform (8 words)
static const uint W_INTERNAL_FLAGS = 45;    // internalFlags
static const uint W_EXTERNAL_ACCEL = 48;    // externalLinearAcceleration
static const uint MAX_LIN_VEL_UINT4 = 2;    // PXG_BODY_SIM_MAX_LIN_VEL_IND
static const uint SIZE_WITHOUT_ACCEL_UINT4 = 12; // PXG_BODY_SIM_SIZE_WITHOUT_ACCELERATION

// PxsRigidBody internal flags used by the direct-API DMA
static const uint PXG_BODY_FIRST_BODY_COPY_GPU = 1u << 9;
static const uint PXG_BODY_VELOCITY_COPY_GPU = 1u << 10;

struct BodyUpdateParams
{
    uint count; // bodies / shapes / velocity updates
};

[[vk::push_constant]]
ConstantBuffer<BodyUpdateParams> gParams;

RWStructuredBuffer<uint> gNewData;      // binding 0: staged source (bodies / shapes / velocity updates)
RWStructuredBuffer<uint> gBodySimPool;  // binding 1: persistent PxgBodySim pool (word-addressed)
RWStructuredBuffer<uint> gShapeSimPool; // binding 2: persistent PxgShapeSim pool (word-addressed)

float4 loadFloat4(RWStructuredBuffer<uint> buf, uint word)
{
    return float4(asfloat(buf[word]), asfloat(buf[word + 1]), asfloat(buf[word + 2]), asfloat(buf[word + 3]));
}

void storeFloat4(RWStructuredBuffer<uint> buf, uint word, float4 v)
{
    buf[word] = asuint(v.x);
    buf[word + 1] = asuint(v.y);
    buf[word + 2] = asuint(v.z);
    buf[word + 3] = asuint(v.w);
}

// PxAlignedTransform helpers on the word-addressed pool (q at +0, p at +4)
PxgAlignedTransform loadAlignedTransform(RWStructuredBuffer<uint> buf, uint word)
{
    PxgAlignedTransform t;
    t.q = loadFloat4(buf, word);
    t.p = loadFloat4(buf, word + 4);
    return t;
}

void storeAlignedTransform(RWStructuredBuffer<uint> buf, uint word, PxgAlignedTransform t)
{
    storeFloat4(buf, word, t.q);
    storeFloat4(buf, word + 4, t.p);
}

// mirrors PxAlignedQuat::rotate(float4): w component of the result is 0
float4 alignedQuatRotate(float4 q, float4 v)
{
    return float4(quatRotate(q, v.xyz), 0.0f);
}

float4 alignedQuatRotateInv(float4 q, float4 v)
{
    return float4(quatRotateInv(q, v.xyz), 0.0f);
}

// mirrors PxAlignedTransform::transform(PxAlignedTransform)
PxgAlignedTransform alignedTransformMul(PxgAlignedTransform a, PxgAlignedTransform b)
{
    PxgAlignedTransform r;
    r.p = alignedQuatRotate(a.q, b.p) + a.p;
    r.q = quatMul(a.q, b.q);
    return r;
}

// mirrors PxAlignedTransform::getInverse()
PxgAlignedTransform alignedTransformInverse(PxgAlignedTransform t)
{
    PxgAlignedTransform r;
    r.p = alignedQuatRotateInv(t.q, -t.p);
    r.q = quatConjugate(t.q);
    return r;
}

// mirrors PxAlignedTransform::operator!= (ignores p.w)
bool alignedTransformNotEqual(PxgAlignedTransform a, PxgAlignedTransform b)
{
    return !(a.p.x == b.p.x && a.p.y == b.p.y && a.p.z == b.p.z && all(a.q == b.q));
}

// mirrors updateBodiesLaunch(): scatter the staged PxgBodySim records into
// the persistent pool at their embedded body index
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void updateBodiesLaunch(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint i = dispatchThreadId.x;
    if (i >= gParams.count)
        return;

    const uint srcBase = i * BODYSIM_WORDS;
    const uint bodyIndex = gNewData[srcBase + W_BODYSIM_INDEX];
    const uint dstBase = bodyIndex * BODYSIM_WORDS;

    for (uint w = 0; w < BODYSIM_WORDS; ++w)
        gBodySimPool[dstBase + w] = gNewData[srcBase + w];
}

// mirrors updateBodiesLaunchDirectAPI(): first transfer copies everything;
// subsequent transfers keep the GPU velocities/accelerations and re-derive
// body2World when the CPU changed body2Actor (CMassLocalPose updates)
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void updateBodiesLaunchDirectAPI(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint i = dispatchThreadId.x;
    if (i >= gParams.count)
        return;

    const uint srcBase = i * BODYSIM_WORDS;
    const uint bodyIndex = gNewData[srcBase + W_BODYSIM_INDEX];
    const uint dstBase = bodyIndex * BODYSIM_WORDS;

    const uint internalFlags = gNewData[srcBase + W_INTERNAL_FLAGS];
    const bool firstTransfer = (internalFlags & PXG_BODY_FIRST_BODY_COPY_GPU) != 0;
    const bool copyVel = (internalFlags & PXG_BODY_VELOCITY_COPY_GPU) != 0;

    if (firstTransfer)
    {
        // on the first DMA, we copy the transform data that the user setup on CPU
        for (uint w = 0; w < BODYSIM_WORDS; ++w)
            gBodySimPool[dstBase + w] = gNewData[srcBase + w];
        return;
    }

    // store the old body2world and body2actor
    const PxgAlignedTransform body2WorldOld = loadAlignedTransform(gBodySimPool, dstBase + W_BODY2WORLD);
    const PxgAlignedTransform oldBody2Actor = loadAlignedTransform(gBodySimPool, dstBase + W_BODY2ACTOR);

    // copy everything below the external accelerations; keep linear and
    // angular velocity xyz intact (copy only inverseMass / maxPenBias .w)
    // unless the velocity-copy flag is raised
    for (uint u4 = 0; u4 < SIZE_WITHOUT_ACCEL_UINT4; ++u4)
    {
        const uint w0 = u4 * 4;
        if (u4 < MAX_LIN_VEL_UINT4 && !copyVel)
        {
            gBodySimPool[dstBase + w0 + 3] = gNewData[srcBase + w0 + 3];
        }
        else
        {
            for (uint c = 0; c < 4; ++c)
                gBodySimPool[dstBase + w0 + c] = gNewData[srcBase + w0 + c];
        }
    }

    // recalculate body2world with new body2actor
    const PxgAlignedTransform newBody2Actor = loadAlignedTransform(gBodySimPool, dstBase + W_BODY2ACTOR);

    if (alignedTransformNotEqual(oldBody2Actor, newBody2Actor))
    {
        const PxgAlignedTransform actor2World = alignedTransformMul(body2WorldOld, alignedTransformInverse(oldBody2Actor));
        const PxgAlignedTransform body2World = alignedTransformMul(actor2World, newBody2Actor);
        storeAlignedTransform(gBodySimPool, dstBase + W_BODY2WORLD, body2World);
    }
}

// mirrors updateShapesLaunch(): scatter PxgNewShapeSim records (a PxgShapeSim
// plus its element index, 80 bytes) into the shape-sim pool
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void updateShapesLaunch(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint NEW_SHAPE_WORDS = 20; // sizeof(PxgNewShapeSim) = 80
    const uint SHAPE_WORDS = 18;     // sizeof(PxgShapeSim) = 72
    const uint W_ELEMENT_INDEX = 18; // PxgNewShapeSim::mElementIndex

    const uint idx = dispatchThreadId.x;
    if (idx >= gParams.count)
        return;

    const uint srcBase = idx * NEW_SHAPE_WORDS;
    const uint elementIndex = gNewData[srcBase + W_ELEMENT_INDEX];
    const uint dstBase = elementIndex * SHAPE_WORDS;

    for (uint w = 0; w < SHAPE_WORDS; ++w)
        gShapeSimPool[dstBase + w] = gNewData[srcBase + w];
}

// mirrors updateBodyExternalVelocitiesLaunch(): applies user velocity /
// acceleration updates (PxgBodySimVelocityUpdate, 4 float4 per record) to
// the body-sim pool, preserving the pooled inverse mass in linVel.w
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void updateBodyExternalVelocitiesLaunch(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint UPDATE_WORDS = 16; // sizeof(PxgBodySimVelocityUpdate) = 64

    const uint i = dispatchThreadId.x;
    if (i >= gParams.count)
        return;

    const uint srcBase = i * UPDATE_WORDS;

    float4 linearVel = loadFloat4(gNewData, srcBase + 0);
    const uint bodyIndex = asuint(linearVel.w);
    const uint dstBase = bodyIndex * BODYSIM_WORDS;

    const float originalInvMass = asfloat(gBodySimPool[dstBase + W_LINVEL + 3]);
    linearVel.w = originalInvMass;

    storeFloat4(gBodySimPool, dstBase + W_LINVEL, linearVel);
    storeFloat4(gBodySimPool, dstBase + W_ANGVEL, loadFloat4(gNewData, srcBase + 4));
    storeFloat4(gBodySimPool, dstBase + W_EXTERNAL_ACCEL, loadFloat4(gNewData, srcBase + 8));
    storeFloat4(gBodySimPool, dstBase + W_EXTERNAL_ACCEL + 4, loadFloat4(gNewData, srcBase + 12));
}
