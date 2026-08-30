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

// femClothPsSolveBda.slang
//
// P9.31 -- cloth_solveCPOutputClothDeltaVLaunch and
//          cloth_solveCPOutputParticleDeltaVLaunch
// gpusimulationcontroller/src/CUDA/FEMClothExternalSolve.cu:1159 and :1266
// (wrappers), :182 (the float4 computeTriangleContact the calls actually
// select), :59 (updateTrianglePosDelta), :53 (FEMCLOTH_BIAS_COEFFICIENT) --
// all read in full.  Launches: PxgFEMClothCore.cpp:2149 and :2217, both
// grid 1024, block 256.
//
// The last two links of the cloth-vs-particle chain.  Both solve the SAME
// constraint and differ only in where the result lands: one accumulates a
// position delta onto the cloth's three vertices, the other writes a velocity
// delta for the particle.
//
// ---------------------------------------------------------------------------
// THEY ARE NOT TWINS, and the difference is not only the output
// ---------------------------------------------------------------------------
// Everything from the work index down to `appliedForce.y += deltaFr` is the
// same text in both, so it is written ONCE here.  What differs:
//
//   * the cloth side computes  deltaPos = (( normal * deltaF) - tanDir * deltaFr) * BIAS
//     and scatters it with updateTrianglePosDelta;
//   * the particle side computes deltaV = ((-normal * deltaF) + tanDir * deltaFr)
//     * invMass0 * BIAS -- opposite signs AND an extra invMass0 factor -- and
//     stores it at deltaP[workIndex] with .w a 0/1 flag;
//   * the particle side takes one more argument (deltaP), so the two argument
//     packs are 56 and 64 bytes and get their own ABI rows.
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
// S1 -- THE WORK INDEX.  Same formula as the prepare kernel, and the same
//     correction: a grid-stride substitution is observation-equivalent (same
//     set, same lane), while dropping the `nbIterationsPerBlock *` factor is
//     not.  See P1 in femClothPsPrepareBda.slang for the argument.
//
// S2 -- THE CONSTRAINT BLOCK IS WARP-BLOCKED: constraints[workIndex / 32]
//     indexed by threadIdx.x & 31.
//
// S3 -- EVERYTHING IS GATED ON `invMass0 != 0`.  A particle with zero inverse
//     mass produces no force, no friction and no write -- and its
//     appliedForces entry keeps whatever the prepare kernel zeroed it to.
//
// S4 -- THE NORMAL IS NEGATED A THIRD TIME.  The contact gen stored -n, the
//     prepare negated it back to +n, and this negates the constraint's normal
//     again.
//
// S5 -- computeTriangleContact here is the FLOAT4 overload (:182), which
//     weights by (b.x, b.y, b.z) -- NOT the float2 overload (:155) twenty lines
//     above it, which weights by (1-b.x-b.y, b.x, b.y).  The call passes a
//     float4 and overload resolution picks the float4 one; reading the file
//     top-down finds the float2 form first and it is a different weighting.
//     Its out-parameter is the THREE per-vertex inverse masses, not the
//     interpolated one.
//
// S6 -- `tanDir.normalize()` is PxVec3::normalize: it returns the MAGNITUDE and
//     normalises in place, guarded on m > 0.  `fricDelta` is that magnitude.
//
// S7 -- THE FORCE CLAMPS.  deltaF = max(-appliedForce.x, -error * velMultiplier)
//     keeps the accumulated normal force non-negative; the friction step is
//     deltaFr = min(requiredForce + appliedForce.y, friction) - appliedForce.y.
//     Both read AND write appliedForces[workIndex], so the kernel is
//     iteration-stateful.
//
// S8 -- updateTrianglePosDelta skips a vertex when its inverse mass is zero OR
//     |barycentric| <= 1e-6, and adds 1.0 to .w for each vertex it does touch.
//
// S9 -- the element index is decoded with PxGetClothElementIndex on the cloth
//     side and PxGetSoftBodyElementIndex on the particle side.  Both masks are
//     0x000fffff and both shifts are 20, so the two produce the SAME value --
//     verified, not assumed.  It is a readability defect that would become a
//     real one the day either constant moves; upstream candidate 40, documented
//     and not reported.
// ===========================================================================

import pxgSlangFEMClothDesc;   // PxgFEMClothGpu
import pxgSlangParticleDesc;   // PxgParticleSystemGpu + the two material accessors
import pxgSlangAtomicFloat;    // pxgAtomicAddF4XYZ / pxgAtomicAddF4W

static const uint PSS_GRID = 1024u;
static const uint PSS_BLOCK = 256u;

// FEMCLOTH_BIAS_COEFFICIENT (FEMClothExternalSolve.cu:53).
static const float FEMCLOTH_BIAS_COEFFICIENT = 0.7f;

static const uint PSS_CONSTRAINT_STRIDE = 1152u;
static const uint PSS_OFF_NORMAL_PEN = 0u;
static const uint PSS_OFF_BARYCENTRIC = 512u;
static const uint PSS_OFF_VELMULT = 1024u;

static const uint PXG_CLOTH_ID_SHIFT = 20u;
static const uint PXG_CLOTH_ELEM_MASK = 0x000fffffu;   // S9

// PxsDeformableSurfaceMaterialData (PxsDeformableSurfaceMaterialCore.h:38), read
// against the header rather than from memory -- the first draft of this file said
// index 1 and that was wrong:
//     youngs@0  poissons@4  dynamicFriction@8  thickness@12  bendingStiffness@16
//     elasticityDamping@20  bendingDamping@24  padding@28   -> 32 bytes, 16-aligned
static const uint PSS_CLOTH_MAT_STRIDE = 32u;
static const uint PSS_CLOTH_MAT_DYNFRICTION = 2u;   // float index, i.e. byte 8

// PxsParticleMaterialData: friction is float index 0.
static const uint PSS_PS_MAT_FRICTION = 0u;

// ---------------------------------------------------------------------------
// Push constants.  The two kernels take DIFFERENT packs.
//   cloth side  (PxgFEMClothCore.cpp:2160): seven pointers -> 56 bytes.
//   particle side (:2228):                  eight pointers -> 64 bytes, with
//                                           deltaP inserted before appliedForces.
// ---------------------------------------------------------------------------
struct ClothPsSolveClothParams
{
    uint64_t clothes;              // @0
    uint64_t particleSystems;      // @8
    uint64_t contactInfos;         // @16
    uint64_t constraints;          // @24
    uint64_t numContacts;          // @32
    uint64_t appliedForces;        // @40  float2*
    uint64_t materials;            // @48 -> 56
}

struct ClothPsSolveParticleParams
{
    uint64_t clothes;              // @0
    uint64_t particleSystems;      // @8
    uint64_t contactInfos;         // @16
    uint64_t constraints;          // @24
    uint64_t numContacts;          // @32
    uint64_t deltaP;               // @40  float4* output -- the extra argument
    uint64_t appliedForces;        // @48  float2*
    uint64_t materials;            // @56 -> 64
}

[[vk::push_constant]] ConstantBuffer<ClothPsSolveClothParams> gSc;
[[vk::push_constant]] ConstantBuffer<ClothPsSolveParticleParams> gSp;

// computeTriangleContact, the FLOAT4 overload (:182).  S5.
float4 pssTriangleContact(uint64_t velsAddr, uint4 triVertId, float4 barycentric,
                          out float4 invMass)
{
    Ptr<float4> vels = Ptr<float4>(velsAddr);
    const float4 v0 = vels[triVertId.x];
    const float4 v1 = vels[triVertId.y];
    const float4 v2 = vels[triVertId.z];
    invMass = float4(v0.w, v1.w, v2.w, 0.0f);
    return v0 * barycentric.x + v1 * barycentric.y + v2 * barycentric.z;
}

// PxVec3::normalize (PxVec3.h:295-301) -- returns the magnitude and normalises in
// place when m > 0.  S6.
//
// THE SCALING IS A RECIPROCAL-MULTIPLY, NOT THREE DIVIDES, and that is not a
// stylistic choice.  normalize() delegates to operator/= (PxVec3.h:260-267),
// which is `f = 1.0/f; x *= f; y *= f; z *= f` -- ONE reciprocal and three
// multiplies.  Slang's `v / m` for scalar m is a genuine componentwise vector
// divide (verified: a single vec3-typed OpFDiv whose divisor is an
// OpCompositeConstruct splat), and the two forms differ in at least one
// component for 53.5% of random vectors.  P9.36.
float pssNormalize(inout float3 v)
{
    const float m = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (m > 0.0f)
        v = v * (1.0f / m);
    return m;
}

// updateTrianglePosDelta (FEMClothExternalSolve.cu:59).  S8.
void pssUpdateTrianglePosDelta(float4 invMasses, float4 barycentric, uint4 vertInds,
                               float3 deltaPos, uint64_t outDeltaAddr)
{
    if (invMasses.x > 0.0f && abs(barycentric.x) > 1e-6f)
    {
        pxgAtomicAddF4XYZ(outDeltaAddr, vertInds.x, deltaPos * invMasses.x * barycentric.x);
        pxgAtomicAddF4W(outDeltaAddr, vertInds.x, 1.0f);
    }
    if (invMasses.y > 0.0f && abs(barycentric.y) > 1e-6f)
    {
        pxgAtomicAddF4XYZ(outDeltaAddr, vertInds.y, deltaPos * invMasses.y * barycentric.y);
        pxgAtomicAddF4W(outDeltaAddr, vertInds.y, 1.0f);
    }
    if (invMasses.z > 0.0f && abs(barycentric.z) > 1e-6f)
    {
        pxgAtomicAddF4XYZ(outDeltaAddr, vertInds.z, deltaPos * invMasses.z * barycentric.z);
        pxgAtomicAddF4W(outDeltaAddr, vertInds.z, 1.0f);
    }
}

// Everything the two kernels share, up to and including the force update.
// `clothSide` selects only the final store.
void pssSolveBody(uint64_t clothes_, uint64_t particleSystems_, uint64_t contactInfos_,
                  uint64_t constraints_, uint64_t numContacts_, uint64_t appliedForces_,
                  uint64_t materials_, uint64_t deltaP_, bool clothSide,
                  uint3 gtid, uint3 gid)
{
    const uint tNumContacts = Ptr<uint>(numContacts_)[0];

    const uint nbBlocksRequired = (tNumContacts + PSS_BLOCK - 1u) / PSS_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + PSS_GRID - 1u) / PSS_GRID;

    const uint idx = gtid.x;
    const uint threadIndexInWarp = gtid.x & 31u;

    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(clothes_);
    Ptr<PxgParticleSystemGpu> particleSystems = Ptr<PxgParticleSystemGpu>(particleSystems_);

    for (uint i = 0u; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * PSS_BLOCK + idx + nbIterationsPerBlock * gid.x * PSS_BLOCK;  // S1

        if (workIndex >= tNumContacts)
            return;

        const uint64_t infoBase = contactInfos_ + uint64_t(workIndex) * 16UL;
        const uint2 pairInd0 = Ptr<uint2>(infoBase + 0UL)[0];
        const uint pairInd1 = Ptr<uint>(infoBase + 8UL)[0];

        const uint64_t cBase = constraints_ +
                               uint64_t(workIndex / 32u) * uint64_t(PSS_CONSTRAINT_STRIDE);      // S2
        const float velMultiplier =
            Ptr<float>(cBase + uint64_t(PSS_OFF_VELMULT))[threadIndexInWarp];

        const uint particleSystemId = pairInd0.y;
        const uint particleIndex = pairInd0.x;

        const float4 deltaP_invMass =
            Ptr<float4>(particleSystems[particleSystemId].mSortedDeltaP)[particleIndex];
        const float invMass0 = deltaP_invMass.w;

        if (invMass0 != 0.0f)                                                                   // S3
        {
            const uint phase =
                Ptr<uint>(particleSystems[particleSystemId].mSortedPhaseArray)[particleIndex];
            const uint group = phase & PXG_PARTICLE_PHASE_GROUP_MASK;
            const uint mi = pxgLoadPhaseToMaterial(
                particleSystems[particleSystemId].mPhaseGroupToMaterialHandle, group);
            const float dynamicFriction0 = pxgLoadParticleMaterialFloat(
                particleSystems[particleSystemId].mParticleMaterials, mi,
                particleSystems[particleSystemId].mParticleMaterialStride, PSS_PS_MAT_FRICTION);

            const float4 barycentric =
                Ptr<float4>(cBase + uint64_t(PSS_OFF_BARYCENTRIC))[threadIndexInWarp];

            const uint clothId = pairInd1 >> PXG_CLOTH_ID_SHIFT;
            const uint elementId = pairInd1 & PXG_CLOTH_ELEM_MASK;                              // S9
            const uint4 triVertId =
                Ptr<uint4>(clothes[clothId].mTriangleVertexIndices)[elementId];

            float4 invMasses1;
            const float4 delta1 = pssTriangleContact(clothes[clothId].mAccumulatedDeltaPos,
                                                     triVertId, barycentric, invMasses1);        // S5

            // mMaterialIndices is a PxU16 array; the cloth material's
            // dynamicFriction is float index 1 of a 32-byte record.
            const uint64_t miBase = clothes[clothId].mMaterialIndices;
            const uint miWord = Ptr<uint>(miBase)[elementId >> 1];
            const uint globalMaterialIndex = (miWord >> ((elementId & 1u) * 16u)) & 0xffffu;
            const float dynamicFriction1 =
                Ptr<float>(materials_ + uint64_t(globalMaterialIndex) *
                                        uint64_t(PSS_CLOTH_MAT_STRIDE))[PSS_CLOTH_MAT_DYNFRICTION];

            const float frictionCoefficient = (dynamicFriction0 + dynamicFriction1) * 0.5f;

            const float3 linDelta0 = float3(deltaP_invMass.x, deltaP_invMass.y, deltaP_invMass.z);
            const float3 linDelta1 = float3(delta1.x, delta1.y, delta1.z);
            const float3 delta = linDelta1 - linDelta0;

            const float4 normal_pen =
                Ptr<float4>(cBase + uint64_t(PSS_OFF_NORMAL_PEN))[threadIndexInWarp];
            const float3 normal = float3(-normal_pen.x, -normal_pen.y, -normal_pen.z);           // S4

            float2 appliedForce = Ptr<float2>(appliedForces_)[workIndex];

            const float normalDelta = dot(delta, normal);

            float3 tanDir = delta - normal * normalDelta;
            const float fricDelta = pssNormalize(tanDir);                                        // S6

            const float error = normal_pen.w + normalDelta;

            const float deltaF = max(-appliedForce.x, -error * velMultiplier);                   // S7
            appliedForce.x += deltaF;

            const float friction = appliedForce.x * frictionCoefficient;
            const float requiredForce = fricDelta * velMultiplier;

            const float deltaFr = min(requiredForce + appliedForce.y, friction) - appliedForce.y;
            appliedForce.y += deltaFr;

            if (clothSide)
            {
                const float3 deltaPos =
                    ((normal * deltaF) - tanDir * deltaFr) * FEMCLOTH_BIAS_COEFFICIENT;
                pssUpdateTrianglePosDelta(invMasses1, barycentric, triVertId, deltaPos,
                                          clothes[clothId].mDeltaPos);                           // S8
            }
            else
            {
                const float3 deltaV = ((-normal * deltaF) + tanDir * deltaFr) *
                                      invMass0 * FEMCLOTH_BIAS_COEFFICIENT;
                const float w = (deltaF != 0.0f || deltaFr != 0.0f) ? 1.0f : 0.0f;
                Ptr<float4>(deltaP_)[workIndex] = float4(deltaV.x, deltaV.y, deltaV.z, w);
            }

            Ptr<float2>(appliedForces_)[workIndex] = appliedForce;
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void cloth_solveCPOutputClothDeltaVLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    pssSolveBody(gSc.clothes, gSc.particleSystems, gSc.contactInfos, gSc.constraints,
                 gSc.numContacts, gSc.appliedForces, gSc.materials, 0UL, true, gtid, gid);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void cloth_solveCPOutputParticleDeltaVLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    pssSolveBody(gSp.clothes, gSp.particleSystems, gSp.contactInfos, gSp.constraints,
                 gSp.numContacts, gSp.appliedForces, gSp.materials, gSp.deltaP, false, gtid, gid);
}
