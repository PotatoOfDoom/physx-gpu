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

// directApiBda.slang -- P9.79.
//
// The DIRECT-GPU API on the ENGINE launch ABI.  Every entry here is reachable
// only when PxSceneFlag::eENABLE_DIRECT_GPU_API is raised, and until
// PXG_SCENE=directapi existed nothing in this project had ever raised it.
//
// ---------------------------------------------------------------------------
// WHY THIS FILE EXISTS RATHER THAN AN EDIT TO directApi.slang
// ---------------------------------------------------------------------------
// directApi.slang holds the PHASE-1 bound-buffer ports of the rigid getters and
// setters: a 12-byte push block (nbElements / oneOverDt / hasPrevVelocities) and
// RWStructuredBuffer bindings, driven by tests/testDirectApi.cpp.  That form can
// never satisfy the engine, whose launch pack for a getter is four CUDA arguments
// ({8,8,8,4} -> a 32-byte blob), so the shim refused every engine launch of them.
//
// This is the same split the radix gathers already use: radixSortGather.slang
// keeps the Phase-1 form and its fixture, femContactSortBda.slang carries the
// engine-ABI twin, and the MANIFEST decides which one the engine gets.  Adding a
// file rather than converting one keeps both instruments: the bound-buffer
// fixture still proves the arithmetic in isolation, and the BDA entry is what
// the scene drives.
//
// ---------------------------------------------------------------------------
// WHAT WAS ACTUALLY WRONG, AND HOW IT WAS MEASURED
// ---------------------------------------------------------------------------
// P9.79's liveness audit of the twenty engine-reachable unmapped kernels found a
// whole family gated behind one scene flag.  PXG_SCENE=directapi was written
// first, before a line of shader, and its very first run said:
//
//     [pxg-refused] sized getRigidDynamicGlobalPose built a 32-byte push blob
//     [pxg-refused] sized getRigidDynamicLinearVelocity built a 32-byte push blob
//     [pxg-refused] sized getRigidDynamicAngularVelocity built a 32-byte push blob
//     [pxg-refused] sized radixSortCopyBits2 built a 32-byte push blob
//     [pxg-refused] sized radixSortCopy2 built a 32-byte push blob
//     [pxg-refused] sized updateBodiesLaunchDirectAPI built a 8-byte push blob
//     [pxg-launch]  sized getD6JointForces UNMAPPED -- skipped
//     [pxg-launch]  sized getD6JointTorques UNMAPPED -- skipped
//     [pxg-launch]  sized mergeChangedAABBMgrHandlesLaunch UNMAPPED -- skipped
//
// Six of those nine were already in check_launch_abi.py's KNOWN_UNWIRED ratchet,
// which is exactly what that ratchet is for; the scene is what turned a list of
// names into a failing measurement.
//
// ---------------------------------------------------------------------------
// WHY THE TWO RADIX GATHERS LIVE HERE
// ---------------------------------------------------------------------------
// radixSortCopy2 and radixSortCopyBits2 are not direct-API kernels by name, but
// in this engine they have exactly one caller --
// PxgGpuNarrowphaseCore::computeRigidsToShapes(), invoked from
// PxgSimulationController::updateBoundsAndShapes under
// `if (mDynamicContext->mEnableDirectGPUAPI && hasShapeInstanceChanged)`
// (PxgSimulationController.cpp:805).  They sort the rigid-node -> shape remap the
// direct-API getters index through, so they are part of this family whether or
// not their names say so, and they are refused for the same reason.

import pxgSlangCommon;
import pxgSlangSimCoreDesc;

static const uint BLOCK_SIZE = 256;   // every launch in this file: block 256

// ---------------------------------------------------------------------------
// updateBodiesLaunchDirectAPI (updateBodiesAndShapes.cu:105) -- the DMA that
// scatters staged PxgBodySim records into the persistent pool when the
// direct-GPU API is on.
//
// PxgSimulationCore::updateBodies (PxgSimulationCore.cpp:2346) picks this kernel
// over updateBodiesLaunch under `if (mGpuContext->getEnableDirectGPUAPI())`, so
// it belongs to this family even though it lives in bodyUpdate.slang's CUDA file.
// bodyUpdate.slang keeps its Phase-1 bound-buffer port and the testBodyUpdate
// fixture that drives it; the manifest points the ENGINE at this entry.
//
// ENGINE launch pack: one device pointer to PxgNewBodiesDesc ({8} blob), the
// same descriptor and the same 8-byte blob as updateBodiesLaunch.
//
// WHY THIS ONE MATTERS MOST OF THE NINE: it is the kernel that fills the body-sim
// pool every other direct-API getter then reads.  While it was refused,
// PXG_SCENE=directapi could not have produced a right answer even with every
// getter wired -- they would have read an unwritten pool.
// ---------------------------------------------------------------------------

// PxgBodySim word layout (224 bytes = 56 words) -- the same constants
// bodyUpdate.slang uses; duplicated rather than shared because they describe a
// C++ struct, not this file's code, and a single wrong copy is easier to see
// next to its use than behind an import.
static const uint BODYSIM_WORDS = 56;
static const uint W_BODYSIM_INDEX = 27;          // freezeThreshold..._bodySimIndex.w
static const uint W_BODY2WORLD = 28;             // PxAlignedTransform (8 words)
static const uint W_BODY2ACTOR = 36;             // PxAlignedTransform (8 words)
static const uint W_INTERNAL_FLAGS = 45;         // internalFlags
static const uint MAX_LIN_VEL_UINT4 = 2;         // PXG_BODY_SIM_MAX_LIN_VEL_IND
static const uint SIZE_WITHOUT_ACCEL_UINT4 = 12; // PXG_BODY_SIM_SIZE_WITHOUT_ACCELERATION

// PxsRigidBody internal flags used by the direct-API DMA
static const uint PXG_BODY_FIRST_BODY_COPY_GPU = 1u << 9;
static const uint PXG_BODY_VELOCITY_COPY_GPU = 1u << 10;

struct NewBodiesAbiParams
{
    uint64_t scDesc; // @0 const PxgNewBodiesDesc*  -> 8-byte blob
};

[[vk::push_constant]]
ConstantBuffer<NewBodiesAbiParams> gNewBodiesAbi;

float4 loadF4Ptr(Ptr<uint> buf, uint word)
{
    return float4(asfloat(buf[word]), asfloat(buf[word + 1]),
                  asfloat(buf[word + 2]), asfloat(buf[word + 3]));
}

void storeF4Ptr(Ptr<uint> buf, uint word, float4 v)
{
    buf[word] = asuint(v.x);
    buf[word + 1] = asuint(v.y);
    buf[word + 2] = asuint(v.z);
    buf[word + 3] = asuint(v.w);
}

PxgAlignedTransform loadAlignedTransformPtr(Ptr<uint> buf, uint word)
{
    PxgAlignedTransform t;
    t.q = loadF4Ptr(buf, word);
    t.p = loadF4Ptr(buf, word + 4);
    return t;
}

void storeAlignedTransformPtr(Ptr<uint> buf, uint word, PxgAlignedTransform t)
{
    storeF4Ptr(buf, word, t.q);
    storeF4Ptr(buf, word + 4, t.p);
}

// mirrors PxAlignedQuat::rotate(float4): the w component of the result is 0
float4 alignedQuatRotate(float4 q, float4 v) { return float4(quatRotate(q, v.xyz), 0.0f); }
float4 alignedQuatRotateInv(float4 q, float4 v) { return float4(quatRotateInv(q, v.xyz), 0.0f); }

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

// first transfer copies everything; subsequent transfers keep the GPU
// velocities/accelerations and re-derive body2World when the CPU changed
// body2Actor (setCMassLocalPose).
//
// GRID-STRIDE (inherited from the P9.63 finding on the bound-buffer twin): the
// CUDA loops `for (i = idx/16; i < totalNbBodies; i += (blockDim.x*gridDim.x)/16)`.
// This port maps ONE thread per body rather than sixteen -- there is no shuffle
// to keep a group aligned for, the broadcast became a re-read -- so its stride is
// the plain thread count, not that over 16.  Without the loop the port would stop
// at the engine's 64 x 256 = 16384 threads and every body beyond that would keep
// whatever the pool already held.
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void updateBodiesLaunchDirectAPI(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const PxgNewBodiesDescGpu scDesc = (Ptr<PxgNewBodiesDescGpu>(gNewBodiesAbi.scDesc))[0];

    Ptr<uint> gNewData = Ptr<uint>(scDesc.mNewBodySim);
    Ptr<uint> gBodySimPool = Ptr<uint>(scDesc.mBodySimBufferDeviceData);
    const uint totalNbBodies = scDesc.mNbNewBodies;

    for (uint i = dispatchThreadId.x; i < totalNbBodies; i += BLOCK_SIZE * WorkgroupCount().x)
    {
        const uint srcBase = i * BODYSIM_WORDS;
        const uint bodyIndex = gNewData[srcBase + W_BODYSIM_INDEX];
        const uint dstBase = bodyIndex * BODYSIM_WORDS;

        const uint internalFlags = gNewData[srcBase + W_INTERNAL_FLAGS];
        const bool firstTransfer = (internalFlags & PXG_BODY_FIRST_BODY_COPY_GPU) != 0;
        const bool copyVel = (internalFlags & PXG_BODY_VELOCITY_COPY_GPU) != 0;

        if (firstTransfer)
        {
            // on the first DMA, we copy the transform data the user set up on CPU
            for (uint w = 0; w < BODYSIM_WORDS; ++w)
                gBodySimPool[dstBase + w] = gNewData[srcBase + w];
            continue;   // CUDA: `continue` inside the grid-stride loop, not `return`
        }

        // store the old body2world and body2actor
        const PxgAlignedTransform body2WorldOld = loadAlignedTransformPtr(gBodySimPool, dstBase + W_BODY2WORLD);
        const PxgAlignedTransform oldBody2Actor = loadAlignedTransformPtr(gBodySimPool, dstBase + W_BODY2ACTOR);

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

        // recalculate body2world with the new body2actor
        const PxgAlignedTransform newBody2Actor = loadAlignedTransformPtr(gBodySimPool, dstBase + W_BODY2ACTOR);

        if (alignedTransformNotEqual(oldBody2Actor, newBody2Actor))
        {
            const PxgAlignedTransform actor2World =
                alignedTransformMul(body2WorldOld, alignedTransformInverse(oldBody2Actor));
            const PxgAlignedTransform body2World = alignedTransformMul(actor2World, newBody2Actor);
            storeAlignedTransformPtr(gBodySimPool, dstBase + W_BODY2WORLD, body2World);
        }
    }
}

// ---------------------------------------------------------------------------
// The rigid getters (updateBodiesAndShapes.cu:999 / :1019 / :1037).
//
// ENGINE launch pack, identical for all three
// (PxgSimulationCore::getRigidDynamicData, PxgSimulationCore.cpp:2755):
//   (void* data, const PxRigidDynamicGPUIndex* gpuIndices,
//    const PxgBodySim* bodySimBufferDeviceData, const PxU32 nbElements)
// -> argSizes {8,8,8,4}, natural-aligned to 28 and rounded to a 32-byte blob.
//
// `data` is the USER's device buffer and the element type differs per getter --
// a PxTransform (7 tightly packed floats) for the pose, a PxVec3 (3) for the two
// velocities.  That difference lives in the body, not the pack, so one push
// block serves all three, exactly as RadixCopyParams serves its two gathers.
// ---------------------------------------------------------------------------
struct RigidGetParams
{
    uint64_t data;       // @0  PxTransform* / PxVec3*  (user device memory)
    uint64_t gpuIndices; // @8  const PxRigidDynamicGPUIndex*
    uint64_t bodySims;   // @16 const PxgBodySim*
    uint nbElements;     // @24 (+4 pad -> 32)
};

[[vk::push_constant]]
ConstantBuffer<RigidGetParams> gRigidGet;

// mirrors getRigidDynamicGlobalPose():
//
//     PxAlignedTransform pose = bodySim.body2World * bodySim.body2Actor_maxImpulseW.getInverse();
//     data[globalThreadIndex] = pose.getTransform();
//
// PxAlignedTransform::getInverse() is (conj(q), rotateInv(q, -p)) and the
// composition is PxTransform::operator*, i.e. (q0*q1, p0 + rotate(q0, p1)).
// PxTransform is 7 tightly packed floats in (q.xyzw, p.xyz) order, so `data` is
// addressed as a raw float array at stride 7 rather than as a struct.
//
// NO grid-stride loop, deliberately: the CUDA is a bare
// `if (globalThreadIndex < nbElements)` and the host sizes the grid as
// (nbElements + 255) / 256, so the launch always covers every element.
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void getRigidDynamicGlobalPose(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint i = dispatchThreadId.x;
    if (i >= gRigidGet.nbElements)
        return;

    const uint index = (Ptr<uint>(gRigidGet.gpuIndices))[i];
    const PxgBodySim bodySim = (Ptr<PxgBodySim>(gRigidGet.bodySims))[index];

    const float4 qW = bodySim.body2World.q;
    const float3 pW = bodySim.body2World.p.xyz;
    const float4 qA = bodySim.body2Actor_maxImpulseW.q;
    const float3 pA = bodySim.body2Actor_maxImpulseW.p.xyz;

    const float4 qInv = quatConjugate(qA);
    const float3 pInv = quatRotateInv(qA, -pA);

    const float4 q = quatMul(qW, qInv);
    const float3 p = quatRotate(qW, pInv) + pW;

    Ptr<float> data = Ptr<float>(gRigidGet.data);
    data[i * 7 + 0] = q.x;
    data[i * 7 + 1] = q.y;
    data[i * 7 + 2] = q.z;
    data[i * 7 + 3] = q.w;
    data[i * 7 + 4] = p.x;
    data[i * 7 + 5] = p.y;
    data[i * 7 + 6] = p.z;
}

// mirrors getRigidDynamicLinearVelocity():
//     data[globalThreadIndex] = PxLoad3(bodySim.linearVelocityXYZ_inverseMassW);
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void getRigidDynamicLinearVelocity(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint i = dispatchThreadId.x;
    if (i >= gRigidGet.nbElements)
        return;

    const uint index = (Ptr<uint>(gRigidGet.gpuIndices))[i];
    const float3 v = (Ptr<PxgBodySim>(gRigidGet.bodySims))[index].linearVelocityXYZ_inverseMassW.xyz;

    Ptr<float> data = Ptr<float>(gRigidGet.data);
    data[i * 3 + 0] = v.x;
    data[i * 3 + 1] = v.y;
    data[i * 3 + 2] = v.z;
}

// mirrors getRigidDynamicAngularVelocity():
//     data[globalThreadIndex] = PxLoad3(bodySim.angularVelocityXYZ_maxPenBiasW);
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void getRigidDynamicAngularVelocity(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint i = dispatchThreadId.x;
    if (i >= gRigidGet.nbElements)
        return;

    const uint index = (Ptr<uint>(gRigidGet.gpuIndices))[i];
    const float3 v = (Ptr<PxgBodySim>(gRigidGet.bodySims))[index].angularVelocityXYZ_maxPenBiasW.xyz;

    Ptr<float> data = Ptr<float>(gRigidGet.data);
    data[i * 3 + 0] = v.x;
    data[i * 3 + 1] = v.y;
    data[i * 3 + 2] = v.z;
}

// ---------------------------------------------------------------------------
// The D6 joint getters (updateBodiesAndShapes.cu:1344 / :1367).
//
// ENGINE launch pack, identical for both
// (PxgSimulationCore::getD6JointData, PxgSimulationCore.cpp:3098):
//   (PxVec3* data, const PxD6JointGPUIndex* gpuIndices, PxU32 nbElements,
//    const PxgConstraintWriteback* constraintWriteBackBuffer, PxF32 oneOverDt,
//    const PxgConstraintIdMapEntry* constraintIdMap, PxU32 constraintIdMapSize)
// -> argSizes {8,8,4,8,4,8,4}, natural-aligned to
//    0, 8, 16, 24, 32, 40, 48 -> 52, rounded to a 56-byte blob.
//
// Note the pack is NOT sorted by size: nbElements (a PxU32) sits between two
// device pointers, so the 24-byte pointer that follows it is padded to 8.  That
// is the modulo ladder the shim reproduces, and getting it wrong here would put
// constraintWriteBackBuffer at 20 and hand the kernel a wild address.
//
// Both kernels are the same templated body
// (getD6JointForceOrTorque<tExtractOperation>) with one field selected, so they
// share this block and differ by a single member read -- transcribed twice
// rather than through a shared helper with a flag, because the CUDA's selection
// is a COMPILE-TIME template parameter and a runtime flag would be a different
// program.
// ---------------------------------------------------------------------------
static const uint PXG_CONSTRAINT_ID_INVALID = 0xffffffffu;

struct D6JointGetParams
{
    uint64_t data;                       // @0  PxVec3*  (user device memory)
    uint64_t gpuIndices;                 // @8  const PxD6JointGPUIndex*
    uint nbElements;                     // @16 (+4 pad)
    uint _pad0;                          // @20
    uint64_t constraintWriteBackBuffer;  // @24 const PxgConstraintWriteback*
    float oneOverDt;                     // @32 (+4 pad)
    uint _pad1;                          // @36
    uint64_t constraintIdMap;            // @40 const PxgConstraintIdMapEntry*
    uint constraintIdMapSize;            // @48 (+4 pad -> 56)
};

[[vk::push_constant]]
ConstantBuffer<D6JointGetParams> gD6Get;

// mirrors PxgConstraintWriteback (gpusolver/include/PxgConstraintWriteBack.h):
// two float4, 32 bytes.  The `broken` flag rides in linearImpulse_broken.w and
// the solver residual in angularImpulse_residual.w; the getters read .xyz only,
// which is what PxLoad3 does.
struct PxgConstraintWritebackGpu
{
    float4 linearImpulse_broken;    // @0
    float4 angularImpulse_residual; // @16
};                                  // = 32

// PxgConstraintIdMapEntry (gpusimulationcontroller/include/PxgConstraintIdMap.h)
// is a single PxU32 -- the internal joint-data id, or eINVALID_ID.  Its
// isJointDataIdValid() is `mJointDataId != 0xffffFFFF`, so the mirror is a plain
// uint array and no struct is needed.

// The shared body of both getters.  `wantTorque` selects which of the two
// writeback rows is read; in CUDA that choice is the template parameter
// tExtractOperation, resolved before codegen.
//
// The else-branch is NOT dead and is not an error path in the ordinary sense:
// the CUDA comments say it covers a joint that is inactive or sleeping, and a
// user who queries a joint before ever stepping.  It writes PxVec3(0), and that
// zero is a real answer the scene can observe -- which is why PXG_SCENE=directapi
// holds its joint OFF-CENTRE: with the joint frame on the centre of mass the
// expected torque would also be zero, and the scene could not tell an answered
// query from an unanswered one.
void d6JointGetForceOrTorque(uint i, bool wantTorque)
{
    if (i >= gD6Get.nbElements)
        return;

    // the gpu API index IS the constraint id
    const uint index = (Ptr<uint>(gD6Get.gpuIndices))[i];

    float3 out = float3(0.0f, 0.0f, 0.0f);

    if (index < gD6Get.constraintIdMapSize)
    {
        const uint jointDataId = (Ptr<uint>(gD6Get.constraintIdMap))[index];
        if (jointDataId != PXG_CONSTRAINT_ID_INVALID)
        {
            // the constraint id is the force-writeback index
            const PxgConstraintWritebackGpu wb =
                (Ptr<PxgConstraintWritebackGpu>(gD6Get.constraintWriteBackBuffer))[index];

            const float4 row = wantTorque ? wb.angularImpulse_residual : wb.linearImpulse_broken;
            out = row.xyz * gD6Get.oneOverDt;
        }
    }

    Ptr<float> data = Ptr<float>(gD6Get.data);
    data[i * 3 + 0] = out.x;
    data[i * 3 + 1] = out.y;
    data[i * 3 + 2] = out.z;
}

// mirrors getD6JointForces(): getD6JointForceOrTorque<gExtractForce>, i.e.
//     data[i] = PxLoad3(constraintWriteBack.linearImpulse_broken) * oneOverDt;
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void getD6JointForces(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    d6JointGetForceOrTorque(dispatchThreadId.x, false);
}

// mirrors getD6JointTorques(): getD6JointForceOrTorque<gExtractTorque>, i.e.
//     data[i] = PxLoad3(constraintWriteBack.angularImpulse_residual) * oneOverDt;
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void getD6JointTorques(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    d6JointGetForceOrTorque(dispatchThreadId.x, true);
}

// ---------------------------------------------------------------------------
// The two aggregate-aware radix gathers (radixSortImpl.cu:218 / :237).
//
// These differ from radixSortCopy / radixSortCopyHigh32Bits in
// femContactSortBda.slang in two ways that matter:
//
//   1. numKeys is a PxU32 BY VALUE, not a `const PxU32*` device pointer.  Same
//      spelling in the CUDA signature, different pack: {8,8,8,4} here versus
//      {8,8,8,8} there, and the rows must not be shared.
//   2. an AGGREGATE rank (0xffffffff) is a legal input, and the output for it is
//      an all-ones sentinel rather than a gather.  The non-"2" variants have no
//      such case, so dereferencing inValue[0xffffffff] would be their behaviour
//      and is not a faithful port of these.
//
// Both are launched from PxgGpuNarrowphaseCore::computeRigidsToShapes() with
// block 256 and grid (totalNumShapes + 255) / 256, and both guard with a bare
// `if (globalThreadIndex < numKeys)` -- so, as with the rigid getters above,
// there is no grid-stride loop to restore: the grid always covers the work.
// ---------------------------------------------------------------------------
static const uint RS_AGGREGATE = 0xffffffffu;

struct RadixCopy2Params
{
    uint64_t inValue;  // @0  const PxU64*
    uint64_t outValue; // @8  PxU64*
    uint64_t rank;     // @16 PxU32*
    uint numKeys;      // @24 BY VALUE (+4 pad -> 32)
};

[[vk::push_constant]]
ConstantBuffer<RadixCopy2Params> gRC2;

// mirrors radixSortCopy2 (radixSortImpl.cu:237):
//     const PxU32 index = rank[i];
//     outValue[i] = (index == 0xffffffff) ? 0xffffffffffffffff : inValue[index];
// 64-bit keys are uint2 (x = low word, y = high word), as everywhere else here.
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void radixSortCopy2(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint i = dispatchThreadId.x;
    if (i >= gRC2.numKeys)
        return;

    const uint index = (Ptr<uint>(gRC2.rank))[i];
    const uint2 v = (index == RS_AGGREGATE) ? uint2(RS_AGGREGATE, RS_AGGREGATE)
                                            : (Ptr<uint2>(gRC2.inValue))[index];
    (Ptr<uint2>(gRC2.outValue))[i] = v;
}

struct RadixCopyBits2Params
{
    uint64_t inValue;  // @0  const PxU64*
    uint64_t outValue; // @8  PxU32*
    uint64_t rank;     // @16 PxU32*
    uint numKeys;      // @24 BY VALUE
    uint lowBit;       // @28 a `const bool` in CUDA -- ONE byte in the pack
};                     //     ({8,8,8,4,1} -> 29 -> 32)

[[vk::push_constant]]
ConstantBuffer<RadixCopyBits2Params> gRCB2;

// mirrors radixSortCopyBits2 (radixSortImpl.cu:218):
//     if (index == 0xffffffff) outValue[i] = 0xffffffff;
//     else outValue[i] = lowBit ? PxU32(inValue[index] & 0xffffffff)
//                               : PxU32(inValue[index] >> 32);
//
// THE BOOL IS ONE BYTE.  CUDA passes `const bool lowBit` as a 1-byte argument,
// so the blob carries one meaningful byte at offset 28 and three bytes of
// padding after it.  Reading it as a `uint` here is safe ONLY because the three
// padding bytes are the tail of a 32-byte blob the host zero-extends -- the ABI
// row says 1, and the row is what the shim packs against.  Compare any non-zero
// value rather than testing == 1.
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void radixSortCopyBits2(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint i = dispatchThreadId.x;
    if (i >= gRCB2.numKeys)
        return;

    const uint index = (Ptr<uint>(gRCB2.rank))[i];
    uint outWord;
    if (index == RS_AGGREGATE)
    {
        outWord = RS_AGGREGATE;
    }
    else
    {
        const uint2 key = (Ptr<uint2>(gRCB2.inValue))[index];
        outWord = ((gRCB2.lowBit & 0xffu) != 0u) ? key.x : key.y;
    }
    (Ptr<uint>(gRCB2.outValue))[i] = outWord;
}
