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

// softBodyPsSolveBda.slang
//
// P9.39e -- sb_solveOutputSPDeltaVLaunch and sb_solveOutputParticleDeltaVLaunch
// gpusimulationcontroller/src/CUDA/softBody.cu:1149 and :1286, softBody.cuh:61
// (computeTetraContact), :77 (updateTetraPosDelta), PxVec3.h:260/:295 (normalize),
// PxgFEMCore.h:184 (PxgFEMParticleConstraintBlock) -- all read in full.
// Launches: PxgSoftBodyCore.cpp:1412 and :1506, grid 1024, block 256.
//
// The LAST two links of the softbody-vs-particle contact chain, and a pair of near
// twins: identical friction accumulation, opposite conclusions.  Both are here in one
// file behind one shared body, as the cloth twin (P9.31) does, because the
// accumulation has to stay bit-identical between them and two copies would be two
// chances to drift.
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
// R1 -- THE SIGN IS NEWTON'S THIRD LAW, AND IT IS THE WHOLE DIFFERENCE.
//         softbody side (:1272): deltaPos = (( normal * deltaF) - tanDir * deltaFr) * biasCoefficient
//         particle side (:1407): deltaV   = ((-normal * deltaF) + tanDir * deltaFr) * invMass0 * relaxation
//     Exactly opposite, and nothing else in either expression differs.  Swapping them
//     compiles, runs, conserves nothing, and no type check can see it.
//
// R2 -- THE TRAILING SCALARS DIFFER TOO, AND NOT ONLY IN NAME.  The softbody side
//     multiplies by ONE runtime scalar; the particle side by TWO, invMass0 and then
//     relaxation, as two separate multiplies left to right.  Folding them into one
//     product changes the rounding.
//
// R3 -- READ POINTER AND WRITE POINTER ARE DIFFERENT SOFTBODY FIELDS.  The input is
//     mSimDeltaPos (@128); the softbody side scatters into mSimDelta (@272).  Both are
//     float4* on the same descriptor and both compile.
//
// R4 -- THE SOFTBODY SCATTER IS GUARDED, THE PARTICLE STORE IS NOT.
//     `if (deltaF != 0 || deltaFr != 0)` wraps updateTetraPosDelta (:1274), because the
//     .w it accumulates is the averaging denominator: an unguarded zero add would still
//     bump the count and shrink every other contribution to that vertex.  The particle
//     side has no such counter and stores unconditionally -- but writes `w = 1` under
//     the SAME condition, so the flag survives.  The cloth twin has NO guard at all on
//     its scatter, which is why this cannot be copied across.
//
// R5 -- THE ACCUMULATOR IS ITERATION-STATEFUL AND THE READ ORDER IS LOAD-BEARING:
//         deltaF  = max(-applied.x_old, -error * velMultiplier)
//         applied.x += deltaF
//         friction = applied.x_NEW * frictionCoefficient      <-- the NEW value
//         deltaFr  = min(requiredForce + applied.y_old, friction) - applied.y_old
//         applied.y += deltaFr
//     Hoisting `friction` above the update, or reusing a cached applied.y, changes the
//     result on every iteration after the first.
//
// R6 -- PxMax/PxMin ARE ORDERED TERNARIES, NOT IEEE min/max.  PxMax(a,b) is
//     `a > b ? a : b` (PxUnixMathIntrinsics.h:116), so PxMax(a, NaN) == NaN while
//     PxMax(NaN, b) == b.  Slang's max() is not required to agree, so both are written
//     out as the ternary the engine uses.
//
// R7 -- normalize() RETURNS THE MAGNITUDE AND SCALES BY ONE RECIPROCAL.  fricDelta is
//     the length of the tangential delta BEFORE normalisation, and a zero-length vector
//     is left untouched with length 0.  sqrt(dot(v,v)), never a length intrinsic.
//
// R8 -- BOTH RUNTIME SCALARS ARE ARGUMENTS HERE AND COMPILE-TIME 0.7f IN THE CLOTH
//     TWIN.  biasCoefficient and relaxation sit at @48 and @56 with a 4-byte hole after
//     each; the cloth kernels take neither.
//
// R9 -- SIM-MESH TET, SORTED PARTICLE INDEX -- the same pair of look-alike indices the
//     rest of this chain carries.  pairInd1 decodes to a SIMULATION-mesh tet because
//     the remap rewrote it; pairInd0's low dword is the SORTED particle index.
//
// R10 -- velMultiplier IS LOADED BEFORE THE invMass0 GATE, barycentric AFTER IT.  A
//     port that hoists both above the gate reads the constraint block for contacts it
//     then skips -- harmless today, and exactly the kind of "harmless" that stops being
//     so when the block is only partially initialised (see Q11 in the prepare shader).
//
// R11 -- invMasses IS A DEAD OUT-PARAMETER ON THE PARTICLE SIDE.  computeTetraContact
//     fills it and only the softbody side consumes it, in updateTetraPosDelta.
//
// R12 -- THE MATERIAL INDEX IS A PxU16 READ BY TET ID, and PxgSoftBody.h:138's comment
//     that the array is sized by mNumVertsGM is wrong: it is allocated and uploaded per
//     SIM TET.  Sizing a fixture by vertices would read past the end.

module softBodyPsSolveBda;

import pxgSlangSoftBodyDesc;     // PxgSoftBodyGpu (mSimTetIndices@80, mSimDeltaPos@128,
                                 //                 mSimDelta@272, mMaterialIndices@296)
import pxgSlangParticleDesc;     // PxgParticleSystemGpu + the material accessors
import pxgSlangSoftBodySolve;    // P9.39d: the nine shared solve helpers

static const uint SPS_GRID = 1024u;
static const uint SPS_BLOCK = 256u;

static const uint SPS_CONSTRAINT_STRIDE = 1152u;
static const uint SPS_OFF_NORMAL_PEN = 0u;
static const uint SPS_OFF_BARYCENTRIC = 512u;
static const uint SPS_OFF_VELMULT = 1024u;

// sb_solveOutputSPDeltaVLaunch (softBody.cu:1149): six pointers, then a float with a
// 4-byte hole, then the material pointer.  R8.
struct SbPsSolveSbParams
{
    uint64_t softbodies;           // @0
    uint64_t particleSystems;      // @8
    uint64_t contactInfos;         // @16
    uint64_t constraints;          // @24
    uint64_t numContacts;          // @32
    uint64_t appliedForces;        // @40  float2*
    float biasCoefficient;         // @48
    uint _pad52;                   // @52
    uint64_t materials;            // @56 -> 64
}

// sb_solveOutputParticleDeltaVLaunch (softBody.cu:1286): the same, plus deltaP, and
// `relaxation` where the other has biasCoefficient.
struct SbPsSolvePsParams
{
    uint64_t softbodies;           // @0
    uint64_t particleSystems;      // @8
    uint64_t contactInfos;         // @16
    uint64_t constraints;          // @24
    uint64_t numContacts;          // @32
    uint64_t deltaP;               // @40  float4* output
    uint64_t appliedForces;        // @48  float2*
    float relaxation;              // @56
    uint _pad60;                   // @60
    uint64_t materials;            // @64 -> 72
}

[[vk::push_constant]] ConstantBuffer<SbPsSolveSbParams> gSs;
[[vk::push_constant]] ConstantBuffer<SbPsSolvePsParams> gSq;

// PxsParticleMaterialData (PxParticleGpu.h:47): friction @0.
static const uint PS_MAT_FRICTION = 0u;

// PxGetGroup (PxgParticleSystem.h): the phase's low 20 bits.
static const uint PX_PARTICLE_PHASE_GROUP_MASK = 0x000fffffu;

// R6: the engine's ordered ternaries, not Slang's max/min.
float pxMax(float a, float b) { return a > b ? a : b; }
float pxMin(float a, float b) { return a < b ? a : b; }

void spsSolveBody(uint64_t softbodies_, uint64_t particleSystems_, uint64_t contactInfos_,
                  uint64_t constraints_, uint64_t numContacts_, uint64_t appliedForces_,
                  uint64_t materials_, uint64_t deltaP_, float scalar, bool softbodySide,
                  uint3 gtid, uint3 gid)
{
    const uint tNumContacts = Ptr<uint>(numContacts_)[0];

    const uint nbBlocksRequired = (tNumContacts + SPS_BLOCK - 1u) / SPS_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + SPS_GRID - 1u) / SPS_GRID;

    const uint idx = gtid.x;
    const uint threadIndexInWarp = gtid.x & 31u;

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(softbodies_);
    Ptr<PxgParticleSystemGpu> particleSystems = Ptr<PxgParticleSystemGpu>(particleSystems_);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * SPS_BLOCK + idx + nbIterationsPerBlock * gid.x * SPS_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        const uint64_t infoBase = contactInfos_ + uint64_t(workIndex) * 16UL;
        const uint2 pairInd0 = Ptr<uint2>(infoBase)[0];
        const uint pairInd1 = Ptr<uint>(infoBase + 8UL)[0];

        const uint64_t cBase = constraints_ + uint64_t(workIndex / 32u) * SPS_CONSTRAINT_STRIDE;

        // R10: velMultiplier BEFORE the gate.
        const float velMultiplier = Ptr<float>(cBase + uint64_t(SPS_OFF_VELMULT))[threadIndexInWarp];

        const uint particleSystemId = pairInd0.y;
        const uint particleIndex = pairInd0.x;            // R9: the SORTED index

        const float4 deltaP_invMass =
            Ptr<float4>(particleSystems[particleSystemId].mSortedDeltaP)[particleIndex];
        const float invMass0 = deltaP_invMass.w;

        if (invMass0 != 0.0f)
        {
            // the particle material, via the phase group
            const uint phase = Ptr<uint>(particleSystems[particleSystemId].mSortedPhaseArray)[particleIndex];
            const uint group = phase & PX_PARTICLE_PHASE_GROUP_MASK;
            const uint mi = pxgLoadPhaseToMaterial(
                particleSystems[particleSystemId].mPhaseGroupToMaterialHandle, group);
            const float dynamicFriction0 = pxgLoadParticleMaterialFloat(
                particleSystems[particleSystemId].mParticleMaterials, mi,
                particleSystems[particleSystemId].mParticleMaterialStride, PS_MAT_FRICTION);

            const float4 barycentric =                    // R10: AFTER the gate
                Ptr<float4>(cBase + uint64_t(SPS_OFF_BARYCENTRIC))[threadIndexInWarp];

            const uint softbodyId = getSoftBodyId(pairInd1);
            const uint tetId = getSoftBodyElementIndex(pairInd1);
            const uint4 tetrahedronId1 =
                Ptr<uint4>(softbodies[softbodyId].mSimTetIndices)[tetId];      // R9

            const uint globalMaterialIndex =
                materialIndexAt(softbodies[softbodyId].mMaterialIndices, tetId);  // R12
            const float dynamicFriction1 = materialDynamicFriction(materials_, globalMaterialIndex);

            float4 invMasses1;                            // R11: dead on the particle side
            const float4 delta1 = computeTetraContact(
                Ptr<float4>(softbodies[softbodyId].mSimDeltaPos), tetrahedronId1,
                barycentric, invMasses1);                 // R3: mSimDeltaPos is the INPUT

            const float frictionCoefficient = (dynamicFriction0 + dynamicFriction1) * 0.5f;
            const float3 linDelta0 = deltaP_invMass.xyz;
            const float3 linDelta1 = delta1.xyz;
            const float3 delta = linDelta1 - linDelta0;

            const float4 normal_pen = Ptr<float4>(cBase + uint64_t(SPS_OFF_NORMAL_PEN))[threadIndexInWarp];
            const float3 normal = float3(-normal_pen.x, -normal_pen.y, -normal_pen.z);

            float2 appliedForce = Ptr<float2>(appliedForces_)[workIndex];

            const float normalDelta = dot(delta, normal);

            float3 tanDir = delta - normal * normalDelta;
            const float fricDelta = vecNormalizeInPlace(tanDir);      // R7

            const float error = normal_pen.w + normalDelta;

            // R5: the order of these five lines is the algorithm.
            const float deltaF = pxMax(-appliedForce.x, -error * velMultiplier);
            appliedForce.x += deltaF;
            const float friction = appliedForce.x * frictionCoefficient;
            const float requiredForce = fricDelta * velMultiplier;
            const float deltaFr = pxMin(requiredForce + appliedForce.y, friction) - appliedForce.y;
            appliedForce.y += deltaFr;

            if (softbodySide)
            {
                // R1/R2: (+normal, -tanDir), ONE trailing scalar.
                const float3 deltaPos = ((normal * deltaF) - tanDir * deltaFr) * scalar;
                // R4: guarded, because updateTetraPosDelta also bumps the .w denominator.
                if (deltaF != 0.0f || deltaFr != 0.0f)
                    updateTetraPosDelta(invMasses1, barycentric, tetrahedronId1, deltaPos,
                                        softbodies[softbodyId].mSimDelta);      // R3: the OUTPUT
            }
            else
            {
                // R1/R2: (-normal, +tanDir), TWO trailing scalars, left to right.
                const float3 deltaV = ((-normal * deltaF) + tanDir * deltaFr) * invMass0 * scalar;
                float w = 0.0f;
                if (deltaF != 0.0f || deltaFr != 0.0f)
                    w = 1.0f;
                Ptr<float4>(deltaP_)[workIndex] = float4(deltaV.x, deltaV.y, deltaV.z, w);
            }

            Ptr<float2>(appliedForces_)[workIndex] = appliedForce;
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sb_solveOutputSPDeltaVLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    spsSolveBody(gSs.softbodies, gSs.particleSystems, gSs.contactInfos, gSs.constraints,
                 gSs.numContacts, gSs.appliedForces, gSs.materials, 0UL,
                 gSs.biasCoefficient, true, gtid, gid);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sb_solveOutputParticleDeltaVLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    spsSolveBody(gSq.softbodies, gSq.particleSystems, gSq.contactInfos, gSq.constraints,
                 gSq.numContacts, gSq.appliedForces, gSq.materials, gSq.deltaP,
                 gSq.relaxation, false, gtid, gid);
}
