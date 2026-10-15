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

// ENGINE launch ABI of updateBodyExternalVelocitiesLaunch (P9.93): exactly the
// CUDA parameter list -- one device pointer to PxgUpdatedBodiesDesc ({8} blob).
struct UpdatedBodiesAbiParams
{
    uint64_t scDesc; // const PxgUpdatedBodiesDesc*
};

[[vk::push_constant]]
ConstantBuffer<UpdatedBodiesAbiParams> gUpdatedBodiesAbi;

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
//
// P9.63 -- GRID-STRIDE.  The CUDA loops
// `for (i = idx/16; i < totalNbBodies; i += (blockDim.x * gridDim.x) / 16)`;
// this port maps ONE thread per body rather than sixteen (there is no shuffle to
// keep a group aligned for -- the broadcast became a re-read), so its stride is
// the plain thread count `BLOCK_SIZE * WorkgroupCount().x`, not that over 16.
// Without the loop the port stopped at 64 x 256 = 16384 bodies and every body
// beyond that kept whatever the pool already held: on the first transfer that is
// an uninitialised PxgBodySim, and afterwards a body whose body2World silently
// stops following a CPU-side setCMassLocalPose.  The engine launch is
// UPDATE_BODIES_AND_SHAPES (grid 64, block 256), and nbNewBodies is unbounded --
// it is however many bodies were added since the last step.
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void updateBodiesLaunchDirectAPI(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    for (uint i = dispatchThreadId.x; i < gParams.count; i += BLOCK_SIZE * WorkgroupCount().x)
    {
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
            continue;   // CUDA: `continue` inside the grid-stride loop, not `return`
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
//
// P9.63 -- GRID-STRIDE.  The CUDA is `for (i = idx; i < gNbBodies; i += blockDim.x
// * gridDim.x)`, one thread per record, and this port maps threads the same way,
// so the stride is BLOCK_SIZE * WorkgroupCount().x with no divisor.  The engine
// launch is UPDATE_BODY_EXTERNAL_VELOCITIES (grid 64, block 256): without the loop
// the port applied the update to the first 16384 records and DROPPED the rest, so
// past that count a setLinearVelocity / addForce from the user simply did not
// arrive -- the body kept its previous velocity with no error anywhere.
//
// P9.93 -- ENGINE launch ABI.  This entry was a bound-buffer port ({count} push,
// gNewData / gBodySimPool bindings, like the Phase-1 updateBodiesLaunchDirectAPI
// twin above, whose engine entry lives in directApiBda.slang) and could never satisfy
// the engine's one-pointer pack; no scene had ever called PxRigidDynamic::addForce
// / addTorque / setForceAndTorque / clearForce / clearTorque, the only APIs that
// reach it (PxgSimulationCore::updateBodies launches it only when
// nbUpdatedBodies > 0 and the direct-GPU API is off), so the refusal never showed.
// It now reads the REAL PxgUpdatedBodiesDesc -- whose pool pointer comes FIRST,
// the reverse of PxgNewBodiesDesc -- and moves uint4s: the record's linVel.w is
// the body index as raw bits (PX_FR in PxgBodySimManager::updateBodies), never a
// float.  PXG_SCENE=extforce is the observer.
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void updateBodyExternalVelocitiesLaunch(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint UPDATE_UINT4 = 4; // sizeof(PxgBodySimVelocityUpdate) = 64

    const PxgUpdatedBodiesDescGpu scDesc = (Ptr<PxgUpdatedBodiesDescGpu>(gUpdatedBodiesAbi.scDesc))[0];
    Ptr<uint4> gBodySim = Ptr<uint4>(scDesc.mBodySimBufferDeviceData);
    Ptr<uint4> gUpdatedBodies = Ptr<uint4>(scDesc.mUpdatedBodySim);
    const uint gNbBodies = scDesc.mNbUpdatedBodies;

    for (uint i = dispatchThreadId.x; i < gNbBodies; i += BLOCK_SIZE * WorkgroupCount().x)
    {
        const uint srcBase = i * UPDATE_UINT4;

        uint4 linearVel = gUpdatedBodies[srcBase + 0];
        const uint bodyIndex = linearVel.w;
        const uint dstBase = bodyIndex * PXG_BODY_SIM_UINT4_SIZE;

        // linearVel.w = originalLinearV.w  (the pool's inverse mass survives)
        linearVel.w = gBodySim[dstBase + W_LINVEL / 4].w;

        gBodySim[dstBase + W_LINVEL / 4] = linearVel;
        gBodySim[dstBase + W_ANGVEL / 4] = gUpdatedBodies[srcBase + 1];
        gBodySim[dstBase + W_EXTERNAL_ACCEL / 4] = gUpdatedBodies[srcBase + 2];
        gBodySim[dstBase + W_EXTERNAL_ACCEL / 4 + 1] = gUpdatedBodies[srcBase + 3];
    }
}

// ---------------------------------------------------------------------------
// updateJointsLaunch (P5 Step 5 BDA engine ABI)
// ---------------------------------------------------------------------------
// Port of updateJointsLaunch() in updateBodiesAndShapes.cu L931 — the per-joint
// scatter that copies newly added / changed D6 joints from the CPU-mapped
// (PxPinnedArray) joint pools to the GPU device pools.  Without it the GPU joint
// buffers stay unpopulated (0xFFFFFFFF), so the D6 data-prep + block-prepare read
// garbage; it is the upstream producer that makes a JOINTED scene step (P5 Step 5
// tranche 12).  Launched from PxgSimulationCore::updateJoints whenever there are
// dirty rigid/arti joints: grid = (ceil(maxUpdatedJoints/8), 2, 1), block =
// (32, 8, 1) — blockIdx.y selects rigid (0) vs articulation (1).  One warp copies
// one joint; the copy is a pure byte scatter (no arithmetic), so the strided
// per-lane loop below is byte-identical to the CUDA warpCopy.
//
// Engine launch pack: a single (const PxgUpdatedJointsDesc*) -> {8}.  blockIdx.y
// == 1 is the articulation branch (L962): the same copy over the mD6Arti* pools.
// It was a deferred no-op until a D6 joint touching an articulation link was
// measured to have EXACTLY zero effect (SnippetDirectGPUAPIArticulation's scissor
// loops): the device arti pools kept the host's 0xFFFFFFFF fill.
static const uint PXG_D6_JOINT_DATA_WORDS = 112u;      // sizeof(PxgD6JointData) / 4 = 448/4
static const uint PXG_CONSTRAINT_PREPREP_WORDS = 8u;   // sizeof(PxgConstraintPrePrep) / 4 = 32/4

// mirrors PxgUpdatedJointsDesc (PxgSimulationCoreDesc.h L323), 96 bytes
struct PxgUpdatedJointsDescGpu
{
    uint64_t mD6RigidJointCPUPool;         // @0   const PxgD6JointData*  (pinned/mapped)
    uint64_t mD6RigidJointGPUPool;         // @8   PxgD6JointData*        (device)
    uint64_t mD6RigidJointPrePrepCPUPool;  // @16  const PxgConstraintPrePrep*
    uint64_t mD6RigidJointPrePrepGPUPool;  // @24  PxgConstraintPrePrep*
    uint64_t mD6ArtiJointCPUPool;          // @32
    uint64_t mD6ArtiJointGPUPool;          // @40
    uint64_t mD6ArtiJointPrePrepCPUPool;   // @48
    uint64_t mD6ArtiJointPrePrepGPUPool;   // @56
    uint64_t mUpdatedRigidJointIndices;    // @64  const PxU32*           (pinned/mapped)
    uint mNbUpdatedRigidJoints;            // @72
    uint pad0;                             // @76  (8-align)
    uint64_t mUpdatedArtiJointIndices;     // @80
    uint mNbUpdatedArtiJoints;             // @88
    uint pad1;                             // @92 -> size 96
}

struct UpdateJointsAbiParams
{
    uint64_t jointDesc; // @0 const PxgUpdatedJointsDesc*
}

[[vk::push_constant]]
ConstantBuffer<UpdateJointsAbiParams> gUpdateJointsAbi;

[shader("compute")]
[numthreads(32, 8, 1)]
void updateJointsLaunch(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    if (gUpdateJointsAbi.jointDesc == 0)
        return;

    Ptr<PxgUpdatedJointsDescGpu> desc = Ptr<PxgUpdatedJointsDescGpu>(gUpdateJointsAbi.jointDesc);

    // blockIdx.y == 0: rigid pools; otherwise the articulation pools (L962-995).
    const bool arti = groupId.y != 0u;
    const uint64_t jointCpuPool = arti ? desc[0].mD6ArtiJointCPUPool : desc[0].mD6RigidJointCPUPool;
    const uint64_t jointGpuPool = arti ? desc[0].mD6ArtiJointGPUPool : desc[0].mD6RigidJointGPUPool;
    const uint64_t prePrepCpuPool = arti ? desc[0].mD6ArtiJointPrePrepCPUPool : desc[0].mD6RigidJointPrePrepCPUPool;
    const uint64_t prePrepGpuPool = arti ? desc[0].mD6ArtiJointPrePrepGPUPool : desc[0].mD6RigidJointPrePrepGPUPool;
    const uint64_t updatedIndices = arti ? desc[0].mUpdatedArtiJointIndices : desc[0].mUpdatedRigidJointIndices;
    const uint nbUpdatedJoints = arti ? desc[0].mNbUpdatedArtiJoints : desc[0].mNbUpdatedRigidJoints;
    const uint lane = groupThreadId.x;                                  // threadIdx.x (0..31)
    const uint globalWarpIndex = groupId.x * 8u + groupThreadId.y;      // blockIdx.x*blockDim.y + threadIdx.y

    if (globalWarpIndex >= nbUpdatedJoints)
        return;

    Ptr<uint> updateJointIndices = Ptr<uint>(updatedIndices);
    const uint jointIndex = updateJointIndices[globalWarpIndex];

    // PxgD6JointData scatter: 448 bytes = 112 uints, strided across the 32 lanes.
    Ptr<uint> srcJoint = Ptr<uint>(jointCpuPool + uint64_t(jointIndex) * 448);
    Ptr<uint> dstJoint = Ptr<uint>(jointGpuPool + uint64_t(jointIndex) * 448);
    for (uint u = lane; u < PXG_D6_JOINT_DATA_WORDS; u += 32u)
        dstJoint[u] = srcJoint[u];

    // PxgConstraintPrePrep scatter: 32 bytes = 8 uints.
    Ptr<uint> srcPP = Ptr<uint>(prePrepCpuPool + uint64_t(jointIndex) * 32);
    Ptr<uint> dstPP = Ptr<uint>(prePrepGpuPool + uint64_t(jointIndex) * 32);
    for (uint u = lane; u < PXG_CONSTRAINT_PREPREP_WORDS; u += 32u)
        dstPP[u] = srcPP[u];
}
