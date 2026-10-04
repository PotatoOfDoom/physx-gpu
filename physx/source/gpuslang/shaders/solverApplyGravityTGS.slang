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

// solverApplyGravityTGS.slang
//
// Slang port of applyTGSSubstepGravity (+ its applyAcceleration helper) from
// gpusolver/src/CUDA/solverMultiBlockTGS.cu:1575-1683.  Under
// PxSceneFlag::eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS the host launches it once
// per POSITION iteration (PxgTGSCudaSolverCore.cpp:1354-1374) and preIntegration skips
// gravity and the external accelerations (skipGravityApplication); this kernel adds
// them per substep instead, into every slab copy of each dynamic body's solver
// velocity and into the accumulation-offset copy, honouring the lock flags.
//
// applyAcceleration (.cu:1575):
//   v = bodyVelocities[bodyIndex]                 // v.xyz = lin vel, v.w = angVel.x
//   v.xyz += accelScale*gravityTimesDt + linearAcceleration.xyz*stepDt
//   apply LINEAR_{X,Y,Z} locks (+0 for a locked component)
//   if hasAngularAcceleration:
//       a = bodyVelocities[bodyIndex + angVelIndexOffset]  // a.x = angVel.y, a.y = angVel.z
//       angAcc = sqrtInvInertia.getInverse() * angularAcceleration.xyz
//       v.w += angAcc.x*stepDt;  a.x += angAcc.y*stepDt;  a.y += angAcc.z*stepDt
//       apply ANGULAR_{X,Y,Z} locks
//       bodyVelocities[bodyIndex + angVelIndexOffset] = a
//   bodyVelocities[bodyIndex] = v
//
// P9.95 ENGINE ABI.  The kernel takes the engine's pack
// (PxgSolverCoreDesc*, PxgSolverSharedDesc<IterativeSolveDataTGS>*, PxVec3 gravity)
// -- PxgTGSCudaSolverCore.cpp:1362-1367, PxVec3 BY VALUE = 12 bytes, 28 -> 32-byte
// blob -- and reads every count, offset and pool pointer on-device through the
// byte-exact pxgSlangSolverDesc / pxgSlangCommon mirrors (testSolverDescLayout).
// The P5-era binding port (five SSBOs, a 36-byte GravityParams, pre-resolved node
// indices) could not run on the engine path, where the shim passes no buffers; it was
// rewritten in place, as solverInitVelocitiesTGS (P5 Step 4) and bodyUpdate (P9.93)
// were.
//
// Work geometry: the CUDA block is (32, 8) -- eight warps -- and the bridge drops
// blockDim and dispatches gridX = nbBlocksRequired = ceil(32*(mBodyCount +
// mBodyStartIndex)/256) unscaled.  So one workgroup is 256 threads = one CUDA block:
// warpIndex = blockIdx.x*blockDim.y + threadIdx.y = gid.x*8 + (gtid.x >> 5), lane =
// threadIdx.x = gtid.x & 31.  numThreadsPerBody = min(pow2ceil(numSlabs), 32) lanes
// cooperate on a body, each striding the body's slabs.
//
// Spelled as the IEEE C++ reading of the CUDA (the shader is in
// PXG_NOCONTRACT_SHADERS): getInverse()'s determinant column0.dot(column1.cross(
// column2)) is written out as (x + y) + z (P9.90: an OpDot summed x + (y + z) on
// llvmpipe), the inverse's cofactors keep PxMat33.h's `invDet * -(...)` signs, the
// transform is column0*v.x + column1*v.y + column2*v.z left to right, and the
// singular branch PxMat33(PxIdentity) * v keeps its computed zero terms (P9.86c).
//
// Two decisions recorded here:
//   * UPSTREAM QUIRK KEPT: the gate reads disableGravity (.cu:1656) but
//     applyAcceleration never does (.cu:1575-1583), so a gravity-disabled body that
//     has an external acceleration receives gravity as well.  The CPU twin zeroes it
//     (DyTGSDynamics.cpp:1490-1493).  This port reproduces the engine (P9.79b
//     precedent); PXG_SCENE=substepgravity pins it with body D.
//   * NO out-of-pool guard on the remapped index (CUDA literal).
//     solverInitVelocitiesTGS bounds the same index because the arti prep kernels
//     were unmapped when it was written (P8.55-57); they are wired now, and this
//     kernel runs after the init kernel has consumed the same references array.

import pxgSlangCommon;      // PxgBodySim, PxgSolverBodyData, PxgSolverTxIData, nodeIndexIndex
import pxgSlangSolverDesc;  // PxgSolverCoreDescGpu, PxgSolverSharedDescTGSGpu, PxgIslandContextGpu

struct ApplyTgsGravityAbiParams
{
    uint64_t solverCoreDesc; // @0  const PxgSolverCoreDesc*
    uint64_t sharedDesc;     // @8  const PxgSolverSharedDesc<IterativeSolveDataTGS>*
    float gravityX;          // @16 PxVec3 gravity BY VALUE (12 bytes, align 4)
    float gravityY;          // @20
    float gravityZ;          // @24
    uint pad28;              // @28 trailing pad: the blob is rounded to the 8-byte max alignment
}                            // = 32 == bridge blob (28 -> 32)

[[vk::push_constant]]
ConstantBuffer<ApplyTgsGravityAbiParams> gParams;

// solverBlock.cuh:62-70, literally (no x - 1: the caller only reaches it for a
// non-power of two, where the result is the same).
uint nextPowerOfTwo(uint x)
{
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return x + 1u;
}

// solverBlock.cuh:72-75
bool isPowerOfTwo(uint x)
{
    return (x & (x - 1u)) == 0u;
}

// PxMat33::getInverse() (PxMat33.h:215-242) then operator* / transform (PxMat33.h:344-347).
// m is column-major (PxgSolverTxIData.sqrtInvInertia): c0 = m[0..2], c1 = m[3..5], c2 = m[6..8].
float3 mat33InverseTransform(float m[9], float vx, float vy, float vz)
{
    const float c0x = m[0], c0y = m[1], c0z = m[2];
    const float c1x = m[3], c1y = m[4], c1z = m[5];
    const float c2x = m[6], c2y = m[7], c2z = m[8];
    // P9.90: getDeterminant() = column0.dot(column1.cross(column2)) (PxMat33.h:247), PxVec3::cross
    // and PxVec3::dot spelled out: the dot is (x + y) + z.  An OpDot here was summed x + (y + z)
    // by llvmpipe, and 1/det feeds every angular-acceleration lane.
    const float crx = c1y * c2z - c1z * c2y, cry = c1z * c2x - c1x * c2z, crz = c1x * c2y - c1y * c2x;
    const float det = (c0x * crx + c0y * cry) + c0z * crz;
    if (det != 0.0f)
    {
        const float invDet = 1.0f / det;
        const float i0x = invDet * (c1y * c2z - c2y * c1z), i0y = invDet * -(c0y * c2z - c2y * c0z), i0z = invDet * (c0y * c1z - c0z * c1y);
        const float i1x = invDet * -(c1x * c2z - c1z * c2x), i1y = invDet * (c0x * c2z - c0z * c2x), i1z = invDet * -(c0x * c1z - c0z * c1x);
        const float i2x = invDet * (c1x * c2y - c1y * c2x), i2y = invDet * -(c0x * c2y - c0y * c2x), i2z = invDet * (c0x * c1y - c1x * c0y);
        // column0*v.x + column1*v.y + column2*v.z, left to right per lane
        return float3((i0x * vx + i1x * vy) + i2x * vz, (i0y * vx + i1y * vy) + i2y * vz, (i0z * vx + i1z * vy) + i2z * vz);
    }
    // P9.86: PxMat33(PxIdentity) * v (PxMat33.h:240, transform :346; solverMultiBlockTGS.cu:1592)
    // keeps its zero terms: lane i is (1*v_i + 0*v_j) + 0*v_k, so a -0 in v_i becomes +0 unless
    // all three terms are -0, and an inf/NaN in v_j gives NaN.  Under the SignedZeroInfNanPreserve
    // stamp x*0 and x + (computed zero) stay IEEE; slangc leaves these adds alone (no literal-zero operand).
    return float3((vx + 0.0f * vy) + 0.0f * vz, (0.0f * vx + vy) + 0.0f * vz, (0.0f * vx + 0.0f * vy) + vz);
}

// solverMultiBlockTGS.cu:1575-1606.  disableGravity and hasLinearAcceleration are
// parameters of the CUDA helper that it never reads; they are not passed here.
void applyAcceleration(Ptr<float4> bodyVelocities, uint bodyIndex, float3 gravityTimesDt, float stepDt,
                       float4 linearAcceleration, float4 angularAcceleration, bool hasAngularAcceleration,
                       float accelScale, uint lockFlags, float sqrtInvInertia[9], uint angVelIndexOffset)
{
    float4 v = bodyVelocities[bodyIndex];
    v.x += accelScale * gravityTimesDt.x + linearAcceleration.x * stepDt;
    v.y += accelScale * gravityTimesDt.y + linearAcceleration.y * stepDt;
    v.z += accelScale * gravityTimesDt.z + linearAcceleration.z * stepDt;

    // PxRigidDynamicLockFlag: eLOCK_LINEAR_X/Y/Z = 1/2/4, eLOCK_ANGULAR_X/Y/Z = 8/16/32
    v.x = (lockFlags & 1u) != 0u ? 0.0f : v.x;
    v.y = (lockFlags & 2u) != 0u ? 0.0f : v.y;
    v.z = (lockFlags & 4u) != 0u ? 0.0f : v.z;

    if (hasAngularAcceleration)
    {
        float4 a = bodyVelocities[bodyIndex + angVelIndexOffset];
        const float3 angAcc = mat33InverseTransform(sqrtInvInertia, angularAcceleration.x, angularAcceleration.y, angularAcceleration.z);

        v.w += angAcc.x * stepDt;
        a.x += angAcc.y * stepDt;
        a.y += angAcc.z * stepDt;

        v.w = (lockFlags & 8u) != 0u ? 0.0f : v.w;
        a.x = (lockFlags & 16u) != 0u ? 0.0f : a.x;
        a.y = (lockFlags & 32u) != 0u ? 0.0f : a.y;

        bodyVelocities[bodyIndex + angVelIndexOffset] = a;
    }

    bodyVelocities[bodyIndex] = v;
}

[shader("compute")]
[numthreads(256, 1, 1)]   // CUDA block (32, 8, 1); the bridge drops blockDim and dispatches gridX
void applyTGSSubstepGravity(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gParams.solverCoreDesc))[0];
    PxgSolverSharedDescTGSGpu shared = (Ptr<PxgSolverSharedDescTGSGpu>(gParams.sharedDesc))[0];

    const float stepDt = shared.stepDt;

    Ptr<uint> solverReferences = Ptr<uint>(desc.solverBodyReferences); // PxgSolverReferences::mRemappedBodyIndex
    Ptr<float4> bodyVelocities = Ptr<float4>(shared.iter_solverBodyVelPool);
    const uint bodyOffset = (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[0].mBodyStartIndex;
    const uint numSlabs = desc.numSlabs;
    const uint allBodiesCount = desc.numSolverBodies;

    const uint warpIndex = gid.x * 8u + (gtid.x >> 5u);   // blockIdx.x * blockDim.y + threadIdx.y
    const uint lane = gtid.x & 31u;                       // threadIdx.x

    const uint pow2Slabs = isPowerOfTwo(numSlabs) ? numSlabs : nextPowerOfTwo(numSlabs);
    const uint numThreadsPerBody = min(pow2Slabs, 32u);
    const uint numBodiesPerWarp = 32u / numThreadsPerBody;
    const uint threadIdxInWorkUnit = lane & (numThreadsPerBody - 1u);

    const uint warpBodyStartIndex = warpIndex * numBodiesPerWarp;
    const uint bodyInWarp = lane / numThreadsPerBody;
    const uint bodyId = warpBodyStartIndex + bodyInWarp;

    const uint outputOffset = desc.accumulatedBodyDeltaVOffset;

    // PxVec3 gravityTimesDt = stepDt * gravity  (operator*(float, PxVec3): f*x, f*y, f*z)
    const float3 gravityTimesDt = float3(stepDt * gParams.gravityX, stepDt * gParams.gravityY, stepDt * gParams.gravityZ);

    Ptr<PxgSolverTxIData> iData = Ptr<PxgSolverTxIData>(desc.solverBodyTxIDataPool);

    if (bodyId < allBodiesCount)
    {
        if (bodyId >= bodyOffset)
        {
            const uint nodeIndex = nodeIndexIndex((Ptr<PxgSolverBodyData>(desc.solverBodyDataPool))[bodyId].islandNodeIndex);
            const PxgBodySim bodySim = (Ptr<PxgBodySim>(desc.mBodySimBufferDeviceData))[nodeIndex];
            const uint disableGravity = bodySim.lockFlags_disableGravity >> 16u;   // PxU16 @186
            const float4 linearAcceleration = bodySim.externalLinearAcceleration;
            const float4 angularAcceleration = bodySim.externalAngularAcceleration;
            const bool hasLinearAcceleration = linearAcceleration.x != 0.0f || linearAcceleration.y != 0.0f || linearAcceleration.z != 0.0f;
            const bool hasAngularAcceleration = angularAcceleration.x != 0.0f || angularAcceleration.y != 0.0f || angularAcceleration.z != 0.0f;

            // PxgSolverTxIData iData0: the CUDA reads it ONLY under hasAngularAcceleration,
            // and only the sqrtInvInertia half is used.  The CUDA leaves it uninitialised
            // otherwise; the zeros here are never read (applyAcceleration touches it only
            // under the same hasAngularAcceleration) and only keep slangc's E41035 quiet.
            float sqrtInvInertia[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
            if (hasAngularAcceleration)
                sqrtInvInertia = iData[bodyId /*+ bodyOffset*/].sqrtInvInertia;

            if (disableGravity == 0u || hasLinearAcceleration || hasAngularAcceleration)
            {
                const float accelScale = bodySim.sleepAngVelAccXYZ_accelScaleW.w;
                const uint lockFlags = bodySim.lockFlags_disableGravity & 0xFFFFu;   // PxU16 @184

                for (uint b = threadIdxInWorkUnit; b < numSlabs; b += numThreadsPerBody)
                {
                    const uint bodyIndex = numSlabs * (bodyId - bodyOffset) + b;
                    const uint remappedBodyIndex = solverReferences[bodyIndex];
                    if (remappedBodyIndex != 0xFFFFFFFFu)
                        applyAcceleration(bodyVelocities, remappedBodyIndex, gravityTimesDt, stepDt, linearAcceleration,
                                          angularAcceleration, hasAngularAcceleration, accelScale, lockFlags, sqrtInvInertia, 32u);
                }

                if (threadIdxInWorkUnit == 0u)
                    applyAcceleration(bodyVelocities, outputOffset + bodyId, gravityTimesDt, stepDt, linearAcceleration,
                                      angularAcceleration, hasAngularAcceleration, accelScale, lockFlags, sqrtInvInertia, allBodiesCount);
            }
        }
    }
}
