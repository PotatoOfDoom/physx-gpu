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
// gpusolver/src/CUDA/solverMultiBlockTGS.cu.  Each TGS substep, this adds the
// gravity- and external-acceleration-induced velocity change into every slab
// copy of each dynamic body's solver velocity (and the accumulation-offset
// copy), honouring the per-body linear/angular lock flags.
//
// applyAcceleration (cu:1575):
//   v = bodyVelocities[bodyIndex]                 // v.xyz = lin vel, v.w = angVel.x
//   v.xyz += accelScale*gravityTimesDt + linearAcceleration.xyz*stepDt
//   apply LINEAR_{X,Y,Z} locks (zero the locked component)
//   if hasAngularAcceleration:
//       a = bodyVelocities[bodyIndex + angVelIndexOffset]  // a.x = angVel.y, a.y = angVel.z
//       angAcc = sqrtInvInertia.getInverse() * angularAcceleration.xyz
//       v.w += angAcc.x*stepDt;  a.x += angAcc.y*stepDt;  a.y += angAcc.z*stepDt
//       apply ANGULAR_{X,Y,Z} locks
//       bodyVelocities[bodyIndex + angVelIndexOffset] = a
//   bodyVelocities[bodyIndex] = v
//
// Work layout is identical to initializeSolverVelocitiesTGS: one 32-thread group
// per CUDA warp, numThreadsPerBody cooperating lanes per body (push constant),
// scattering to each slab's remapped slot (angVelIndexOffset = 32) and once per
// body to the accumulation-offset copy (angVelIndexOffset = allBodiesCount).
//
// The per-body sim inputs are supplied pre-resolved: gNodeIndex[bodyId] indexes
// the body-sim inputs (mirroring solverBodyDataPool[bodyId].islandNodeIndex),
// and the sqrt-inverse-inertia (iData[bodyId]) is a PxMat33 (column-major).

struct GravityParams
{
    uint numSlabs;
    uint allBodiesCount;
    uint bodyOffset;
    uint outputOffset;
    uint numThreadsPerBody;
    float gravityX;
    float gravityY;
    float gravityZ;
    float stepDt;
}

// mirrors the accessed fields of PxgBodySim (disableGravity, lockFlags,
// sleepAngVelAccXYZ_accelScaleW.w, external{Linear,Angular}Acceleration)
struct BodySimInput
{
    uint disableGravity;  // PxU16 in CUDA (0/1)
    uint lockFlags;       // PxU16
    float accelScale;
    uint pad0;
    float4 linAccel;
    float4 angAccel;
}

// PxMat33 sqrtInvInertia, column-major (only xyz used)
struct Mat33
{
    float4 col0;
    float4 col1;
    float4 col2;
}

[[vk::push_constant]]
ConstantBuffer<GravityParams> gParams;

RWStructuredBuffer<uint>        gRemap;           // binding 0  solverReferences (mRemappedBodyIndex)
RWStructuredBuffer<float4>      gBodyVelocities;  // binding 1  solverBodyVelPool (in/out)
RWStructuredBuffer<uint>        gNodeIndex;       // binding 2  per body -> body-sim index
RWStructuredBuffer<BodySimInput> gBodySim;        // binding 3  indexed by node index
RWStructuredBuffer<Mat33>       gSqrtInvInertia;  // binding 4  iData[bodyId]

static const uint INVALID = 0xffffffffu;
// PxRigidDynamicLockFlag
static const uint LOCK_LIN_X = 1u, LOCK_LIN_Y = 2u, LOCK_LIN_Z = 4u;
static const uint LOCK_ANG_X = 8u, LOCK_ANG_Y = 16u, LOCK_ANG_Z = 32u;

// PxMat33::getInverse() * v  (column-major; identity if singular)
float3 invInertiaMul(Mat33 m, float3 v)
{
    const float3 c0 = m.col0.xyz, c1 = m.col1.xyz, c2 = m.col2.xyz;
    const float det = dot(c0, cross(c1, c2));
    if (det != 0.0f)
    {
        const float invDet = 1.0f / det;
        const float3 ic0 = float3(invDet *  (c1.y * c2.z - c2.y * c1.z),
                                  invDet * -(c0.y * c2.z - c2.y * c0.z),
                                  invDet *  (c0.y * c1.z - c0.z * c1.y));
        const float3 ic1 = float3(invDet * -(c1.x * c2.z - c1.z * c2.x),
                                  invDet *  (c0.x * c2.z - c0.z * c2.x),
                                  invDet * -(c0.x * c1.z - c0.z * c1.x));
        const float3 ic2 = float3(invDet *  (c1.x * c2.y - c1.y * c2.x),
                                  invDet * -(c0.x * c2.y - c0.y * c2.x),
                                  invDet *  (c0.x * c1.y - c1.x * c0.y));
        return ic0 * v.x + ic1 * v.y + ic2 * v.z;
    }
    return v; // identity * v
}

void applyAcceleration(uint bodyIndex, float3 gravityTimesDt, float stepDt,
                       float4 linAccel, float4 angAccel, bool hasAng,
                       float accelScale, uint lockFlags, Mat33 inertia, uint angVelIndexOffset)
{
    float4 v = gBodyVelocities[bodyIndex];
    v.x += accelScale * gravityTimesDt.x + linAccel.x * stepDt;
    v.y += accelScale * gravityTimesDt.y + linAccel.y * stepDt;
    v.z += accelScale * gravityTimesDt.z + linAccel.z * stepDt;

    v.x = (lockFlags & LOCK_LIN_X) != 0u ? 0.0f : v.x;
    v.y = (lockFlags & LOCK_LIN_Y) != 0u ? 0.0f : v.y;
    v.z = (lockFlags & LOCK_LIN_Z) != 0u ? 0.0f : v.z;

    if (hasAng)
    {
        float4 a = gBodyVelocities[bodyIndex + angVelIndexOffset];
        const float3 angAcc = invInertiaMul(inertia, float3(angAccel.x, angAccel.y, angAccel.z));

        v.w += angAcc.x * stepDt;
        a.x += angAcc.y * stepDt;
        a.y += angAcc.z * stepDt;

        v.w = (lockFlags & LOCK_ANG_X) != 0u ? 0.0f : v.w;
        a.x = (lockFlags & LOCK_ANG_Y) != 0u ? 0.0f : a.x;
        a.y = (lockFlags & LOCK_ANG_Z) != 0u ? 0.0f : a.y;

        gBodyVelocities[bodyIndex + angVelIndexOffset] = a;
    }

    gBodyVelocities[bodyIndex] = v;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void applyTGSSubstepGravity(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint numThreadsPerBody = gParams.numThreadsPerBody;
    const uint numBodiesPerWarp = 32u / numThreadsPerBody;
    const uint allBodiesCount = gParams.allBodiesCount;
    const uint numSlabs = gParams.numSlabs;
    const uint bodyOffset = gParams.bodyOffset;
    const uint outputOffset = gParams.outputOffset;
    const float stepDt = gParams.stepDt;
    const float3 gravityTimesDt = stepDt * float3(gParams.gravityX, gParams.gravityY, gParams.gravityZ);

    const uint warpIndex = gid.x;
    const uint lane = gtid.x;
    const uint threadIdxInWorkUnit = lane & (numThreadsPerBody - 1u);
    const uint bodyInWarp = lane / numThreadsPerBody;
    const uint bodyId = warpIndex * numBodiesPerWarp + bodyInWarp;

    if (bodyId < allBodiesCount && bodyId >= bodyOffset)
    {
        const uint nodeIndex = gNodeIndex[bodyId];
        const BodySimInput sim = gBodySim[nodeIndex];

        const float4 linAccel = sim.linAccel;
        const float4 angAccel = sim.angAccel;
        const bool hasLin = linAccel.x != 0.0f || linAccel.y != 0.0f || linAccel.z != 0.0f;
        const bool hasAng = angAccel.x != 0.0f || angAccel.y != 0.0f || angAccel.z != 0.0f;

        if (sim.disableGravity == 0u || hasLin || hasAng)
        {
            const float accelScale = sim.accelScale;
            const uint lockFlags = sim.lockFlags;
            const Mat33 inertia = gSqrtInvInertia[bodyId];

            for (uint b = threadIdxInWorkUnit; b < numSlabs; b += numThreadsPerBody)
            {
                const uint bodyIndex = numSlabs * (bodyId - bodyOffset) + b;
                const uint remapped = gRemap[bodyIndex];
                if (remapped != INVALID)
                    applyAcceleration(remapped, gravityTimesDt, stepDt, linAccel, angAccel, hasAng,
                                      accelScale, lockFlags, inertia, 32u);
            }

            if (threadIdxInWorkUnit == 0u)
                applyAcceleration(outputOffset + bodyId, gravityTimesDt, stepDt, linAccel, angAccel, hasAng,
                                  accelScale, lockFlags, inertia, allBodiesCount);
        }
    }
}
