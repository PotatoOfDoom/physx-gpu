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
//   - updateBodiesLaunch          (ENGINE launch ABI, BDA — see below)
//   - updateBodiesLaunchDirectAPI (bound-buffer form)
//   - updateShapesLaunch          (ENGINE launch ABI, BDA — see below)
//   - updateBodyExternalVelocitiesLaunch (bound-buffer form)
//
// updateBodiesLaunch and updateShapesLaunch run on the ENGINE launch ABI
// (P5 Step 4): the push blob is the CUDA parameter list (one PxgNewBodiesDesc
// device pointer / two raw element pointers + a count) and the data is walked
// via BDA through the byte-exact pxgSlangSimCoreDesc mirrors, with the CUDA
// thread mapping preserved: updateBodiesLaunch keeps the 16-threads-per-body
// uint4 gather and the grid-stride loop `i += (blockDim.x * gridDim.x) / 16`.
// The only CUDA construct without a direct analogue is the __shfl_sync that
// broadcasts data.w from lane PXG_BODY_SIM_BODYSIM_INDEX_IND within each
// 16-lane group; since that lane's value is a uniform re-readable location,
// each participating thread re-reads the same uint4.w from memory instead
// (bit-identical result, no cross-lane op).
//
// The two direct-API entries below remain in bound-buffer form: buffers are
// bound word-addressed so partial float4 updates (the .w-only velocity lane
// merge of the direct-API path) stay race-free.

import pxgSlangCommon;
import pxgSlangSimCoreDesc;

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

// ENGINE launch ABI of updateBodiesLaunch: exactly the CUDA parameter list —
// one device pointer ({8} blob).
struct NewBodiesAbiParams
{
    uint64_t scDesc; // const PxgNewBodiesDesc*
};

[[vk::push_constant]]
ConstantBuffer<NewBodiesAbiParams> gNewBodiesAbi;

// ENGINE launch ABI of updateShapesLaunch: exactly the CUDA parameter list —
// two device pointers + a PxU32 count ({8,8,4} blob, padded to 24 bytes).
struct UpdateShapesAbiParams
{
    uint64_t newShapeSimsBufferDeviceData; // @0  const PxgNewShapeSim*
    uint64_t shapeSimsBufferDeviceData;    // @8  PxgShapeSim*
    uint nbNewShapes;                      // @16 (+4 bytes pad -> 24)
};

[[vk::push_constant]]
ConstantBuffer<UpdateShapesAbiParams> gUpdateShapesAbi;

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
// the persistent pool at their embedded body index — ENGINE launch ABI.
// Thread mapping is the CUDA's: 16 threads per body (14 of which move one
// uint4 each), grid-stride `i += (blockDim.x * gridDim.x) / 16`, launched
// with the engine's 64 x 256 grid.  The CUDA loop condition
// `(i < totalNbBodies) | (ballot(...) & 0)` evaluates to `i < totalNbBodies`
// (the ballot only refreshed the shuffle mask); the shuffle broadcast of
// lane PXG_BODY_SIM_BODYSIM_INDEX_IND's data.w becomes a re-read of that
// same uniform uint4.w from memory (see file header).
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void updateBodiesLaunch(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const PxgNewBodiesDescGpu scDesc = (Ptr<PxgNewBodiesDescGpu>(gNewBodiesAbi.scDesc))[0];

    Ptr<uint4> gBodySim = Ptr<uint4>(scDesc.mNewBodySim);
    const uint totalNbBodies = scDesc.mNbNewBodies;

    // persistent data
    Ptr<uint4> gBodySimPoolP = Ptr<uint4>(scDesc.mBodySimBufferDeviceData);

    const uint idx = dispatchThreadId.x;

    // each PxgBodySim has 224 bytes, so we need to use 16 threads to read one
    // body (14 threads move one 16-byte uint4 each; 16 keeps the CUDA's
    // shuffle-group alignment)
    for (uint i = idx / 16; i < totalNbBodies; i += (BLOCK_SIZE * WorkgroupCount().x) / 16)
    {
        const uint index = idx & 15;
        if (index < PXG_BODY_SIM_UINT4_SIZE)
        {
            const uint4 data = gBodySim[i * PXG_BODY_SIM_UINT4_SIZE + index];

            // __shfl_sync(mask, data.w, PXG_BODY_SIM_BODYSIM_INDEX_IND, 16)
            const uint bodyIndex = gBodySim[i * PXG_BODY_SIM_UINT4_SIZE + PXG_BODY_SIM_BODYSIM_INDEX_IND].w;

            gBodySimPoolP[bodyIndex * PXG_BODY_SIM_UINT4_SIZE + index] = data;
        }
    }
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
// plus its element index, 80 bytes) into the shape-sim pool — ENGINE launch
// ABI (two raw device pointers + count, grid-stride, engine 64 x 256 grid).
// Field-by-field copy like the CUDA: mTransform, mLocalBounds, mBodySimIndex,
// mHullDataIndex, mShapeFlags/mShapeType — the 4 padding bytes at offset 52
// are NOT copied (the CUDA leaves them untouched too).
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void updateShapesLaunch(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    Ptr<PxgNewShapeSimGpu> newShapeSimsBufferDeviceData =
        Ptr<PxgNewShapeSimGpu>(gUpdateShapesAbi.newShapeSimsBufferDeviceData);
    Ptr<PxgShapeSim> shapeSimsBufferDeviceData = Ptr<PxgShapeSim>(gUpdateShapesAbi.shapeSimsBufferDeviceData);
    const uint nbNewShapes = gUpdateShapesAbi.nbNewShapes;

    const uint startIndex = dispatchThreadId.x;

    for (uint idx = startIndex; idx < nbNewShapes; idx += BLOCK_SIZE * WorkgroupCount().x)
    {
        const PxgNewShapeSimGpu src = newShapeSimsBufferDeviceData[idx];
        const uint elementIndex = src.mElementIndex;

        shapeSimsBufferDeviceData[elementIndex].mTransform = src.base.mTransform;
        shapeSimsBufferDeviceData[elementIndex].lbMinX = src.base.lbMinX;
        shapeSimsBufferDeviceData[elementIndex].lbMinY = src.base.lbMinY;
        shapeSimsBufferDeviceData[elementIndex].lbMinZ = src.base.lbMinZ;
        shapeSimsBufferDeviceData[elementIndex].lbMaxX = src.base.lbMaxX;
        shapeSimsBufferDeviceData[elementIndex].lbMaxY = src.base.lbMaxY;
        shapeSimsBufferDeviceData[elementIndex].lbMaxZ = src.base.lbMaxZ;
        shapeSimsBufferDeviceData[elementIndex].mBodySimIndex = src.base.mBodySimIndex;
        shapeSimsBufferDeviceData[elementIndex].mHullDataIndex = src.base.mHullDataIndex;
        shapeSimsBufferDeviceData[elementIndex].mShapeFlags_mShapeType = src.base.mShapeFlags_mShapeType;
    }
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
