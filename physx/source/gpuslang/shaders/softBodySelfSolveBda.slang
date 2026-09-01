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

// softBodySelfSolveBda.slang
//
// P8.4 tranche b — the SOFTBODY-vs-SOFTBODY constraint prepare and solve on the engine BDA
// ABI.  These are chain kernels 2 and 3 of the self-collision arc whose midphase landed in
// tranche a:
//
//   sb_softbodyContactPrepareLaunch      (softBody.cu:940,  PxgSoftBodyCore.cpp:2547,
//                                         SB_SS_CONTACTPREPARE)
//   sb_solveOutputSSDeltaVLaunch{,TGS}   (softBody.cu:1424 / :1691,
//                                         PxgSoftBodyCore.cpp:1636,
//                                         SB_SOLVE_SOFT_SOFT_BOTH_DELTA{,_TGS})
//
// Both sides of the pair are softbody TETS, which is the whole difference from the
// rigid-vs-softbody chain of P8.2 tranche d/e: there one side was a rigid body with a
// 6-DOF velocity and an inertia, here the two sides are symmetric and every step happens
// twice with the sign flipped.  `sb_solveRigidSoftCollisionLaunch` in
// softBodyRigidContactBda is the direct structural sibling and the arithmetic below is
// deliberately written to read alongside it.
//
// WHY THESE ARE A SEPARATE FILE FROM softBodyRigidContactBda.
// The two constraint BLOCKS are different structs -- PxgFemRigidConstraintBlock
// (PxgFEMCore.h:167, stride 3200) against PxgSoftBodySoftBodyConstraintBlock
// (PxgSoftBodyCore.h:51, stride 1792) -- and the two contact-info records are different too
// (PxgFemOtherContactInfo against PxgFemFemContactInfo).  Nothing is shared but the two
// tet helpers, which are small enough that a shared module would be more indirection than
// it saves.  That is the same call P8.3 made for the attachment kernel bodies, for the same
// stated reason: two CUDA functions that merely resemble each other stay two.
//
// *** THE GRID. ***
// Both launches are 1D: PxgSoftBodyKernelGridDim::SB_UPDATEROTATION = 1024 blocks of
// PxgSoftBodyKernelBlockDim::SB_UPDATEROTATION = 256 threads.  CUDA then computes
//
//     nbBlocksRequired      = (tNumContacts + blockDim.x - 1) / blockDim.x
//     nbIterationsPerBlock  = (nbBlocksRequired + gridDim.x - 1) / gridDim.x
//     workIndex             = i * blockDim.x + idx + nbIterationsPerBlock * blockIdx.x * blockDim.x
//
// which is a BLOCK-CONTIGUOUS partition, not a grid stride: block b owns the run
// [b * nbIterationsPerBlock * 256, (b+1) * nbIterationsPerBlock * 256).  It is reproduced
// exactly rather than replaced by a stride loop, because `constraints[workIndex / 32]` and
// `threadIndexInWarp = threadIdx.x & 31` mean the mapping from workIndex to a slot inside a
// constraint BLOCK is fixed by the index, not by the thread -- so any partition that covers
// every index once is equivalent, but reproducing the CUDA one keeps the fixture's expected
// values readable against the .cu.  The bridge drops blockDim, so [numthreads(256,1,1)]
// matches SB_UPDATEROTATION by construction and RC_GRID/RC_BLOCK carry the two constants,
// exactly as softBodyRigidContactBda does.
//
// DEVIATIONS from CUDA:
//   - CUDA `return`s out of the work loop when workIndex >= tNumContacts.  That is a real
//     early exit for the whole thread, and it is reproduced as a `break`, which is the same
//     thing here because nothing follows the loop;
//   - `AtomicAdd(float4&, PxVec3, PxReal)` becomes three float atomic adds plus a fourth on
//     the .w accumulator, using VK_EXT_shader_atomic_float exactly as softBodySolveBda and
//     femClothSolveBda already do.  The .w counter is what the averaging pass divides by, so
//     dropping it silently halves every delta;
//   - `PxVec3::normalize()` returns the LENGTH and leaves the vector unit; a zero-length
//     tangent leaves the vector untouched and returns 0, which is transcribed explicitly
//     rather than relying on a divide by zero producing something usable.
//
// NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the barycentric interpolation, the
// relative-position dot and the friction terms are all mul-feeding-add chains that would
// otherwise fuse to FMAs and break parity with the -ffp-contract=off CPU reference.  The one
// non-portable operation is the tangent normalise (a divide by a sqrt), and it only picks a
// tangential DIRECTION; the accept/reject and magnitude decisions above it are contraction-
// free products.

import pxgSlangSoftBodyDesc;   // PxgSoftBodyGpu (mSimTetIndices@80, mSimDelta@272, ...)
import pxgSlangAtomicFloat;    // pxgAtomicAddF4XYZ / pxgAtomicAddF4W (VK_EXT_shader_atomic_float)
import pxgSlangSoftBodySolve;  // P9.39d: the nine solve helpers, lifted OUT of this file

// PxgSoftBodyKernelGridDim / BlockDim::SB_UPDATEROTATION (PxgSoftBodyCoreKernelIndices.h).
static const uint SS_GRID = 1024u;
static const uint SS_BLOCK = 256u;

// PxEncodeSoftBodyIndex (PxgSimulationCoreDesc.h): (softBodyId << 20) | tetId.

// ---------------------------------------------------------------------------
// PxgFemFemContactInfo (PxgFEMCore.h:51), 16 B: pairInd0 PxU64 @0, pairInd1 PxU32 @8,
// auxInd @12.  Both halves carry a PxU32 encoding, so only the low word of pairInd0 is read.
// ---------------------------------------------------------------------------
static const uint64_t FFCI_STRIDE = 16UL;

// ---------------------------------------------------------------------------
// PxgSoftBodySoftBodyConstraintBlock (PxgSoftBodyCore.h:51) — block-of-32 SoA:
//   float4 barycentric0[32];  float4 barycentric1[32];  float4 normal_pen[32];
//   PxReal velMultiplier[32]; PxReal friction[32];
// -> 512 + 512 + 512 + 128 + 128 = 1792 bytes per block of 32 constraints.
// ---------------------------------------------------------------------------
static const uint64_t SSC_barycentric0  = 0UL;
static const uint64_t SSC_barycentric1  = 512UL;
static const uint64_t SSC_normal_pen    = 1024UL;
static const uint64_t SSC_velMultiplier = 1536UL;
static const uint64_t SSC_friction      = 1664UL;
static const uint64_t SSC_STRIDE        = 1792UL;

uint64_t ssBlockBase(uint64_t constraints, uint workIndex)
{
    return constraints + uint64_t(workIndex / 32u) * SSC_STRIDE;
}


// ===========================================================================
// sb_softbodyContactPrepareLaunch
//
// CUDA arg pack (PxgSoftBodyCore.cpp:2534): { PxgSoftBody* softbodies, float4* contacts,
// float4* normalPens, float4* barycentrics0, float4* barycentrics1,
// PxgFemFemContactInfo* contactInfos, PxU32* numContacts,
// PxgSoftBodySoftBodyConstraintBlock* constraints, float2* softBodyAppliedForces,
// const PxU32 maxContacts } -> nine pointers @0..64 and a PxU32 @72 -> 80-byte blob.
// (ABI row {8,8,8,8,8,8,8,8,8,4}.)
//
// `contacts` is carried for ABI parity and never read: the CUDA takes it and dereferences
// only normalPens, barycentrics0/1 and contactInfos (softBody.cu:940-1016).
// ===========================================================================
struct SbSsPrepParams
{
    uint64_t softbodies;        // @0
    uint64_t contacts;          // @8   (unused; ABI parity)
    uint64_t normalPens;        // @16
    uint64_t barycentrics0;     // @24
    uint64_t barycentrics1;     // @32
    uint64_t contactInfos;      // @40
    uint64_t numContacts;       // @48  in/out
    uint64_t constraints;       // @56  output
    uint64_t appliedForces;     // @64  output (float2 per contact)
    uint maxContacts;           // @72
    uint _pad76;                // @76 -> 80
}
[[vk::push_constant]]
ConstantBuffer<SbSsPrepParams> gPr;

[shader("compute")]
[numthreads(256, 1, 1)]   // == SS_BLOCK == PxgSoftBodyKernelBlockDim::SB_UPDATEROTATION
void sb_softbodyContactPrepareLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    // CUDA writes the clamp as `tNumContacts = PxMin(max, *n); if (thread0) *n = tNumContacts;`
    // -- already a `min` feeding a flat store, so unlike the remap kernels there is nothing
    // to restructure here.  The store is unconditional in CUDA (it writes back even when the
    // clamp did not bite, which is a no-op) and stays that way.
    const uint tNumContacts = min(gPr.maxContacts, Ptr<uint>(gPr.numContacts)[0]);
    if (gtid.x == 0u && gid.x == 0u)
        Ptr<uint>(gPr.numContacts)[0] = tNumContacts;

    if (gid.x >= SS_GRID)
        return;

    const uint nbBlocksRequired = (tNumContacts + SS_BLOCK - 1u) / SS_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + SS_GRID - 1u) / SS_GRID;

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gPr.softbodies);

    for (uint i = 0u; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * SS_BLOCK + gtid.x + nbIterationsPerBlock * gid.x * SS_BLOCK;
        if (workIndex >= tNumContacts)
            break;

        // CUDA's `threadIndexInWarp = threadIdx.x & 31` (softBody.cu:963).  Note this is
        // the THREAD's lane, while the constraint BLOCK is picked by workIndex / 32 -- the
        // two only address a consistent slot because every term of workIndex
        // (i * blockDim.x, idx, nbIterationsPerBlock * blockIdx.x * blockDim.x) is either
        // idx itself or a multiple of blockDim.x = 256, and 256 is a multiple of 32.  So
        // `gtid.x & 31` and `workIndex & 31` are the SAME value for every reachable
        // dispatch; the CUDA-literal form is kept, and a mutant swapping the two is
        // expected to survive because it cannot differ.
        const uint threadIndexInWarp = gtid.x & 31u;
        const uint64_t cb = ssBlockBase(gPr.constraints, workIndex);

        // initialize appliedForces to be zero
        Ptr<float2>(gPr.appliedForces)[workIndex] = float2(0.0f, 0.0f);

        const uint64_t ciBase = gPr.contactInfos + uint64_t(workIndex) * FFCI_STRIDE;
        const uint pairInd0 = Ptr<uint>(ciBase)[0];
        const uint pairInd1 = Ptr<uint>(ciBase + 8UL)[0];

        const uint softbodyId0 = getSoftBodyId(pairInd0);
        const uint tetInd0 = getSoftBodyElementIndex(pairInd0);
        const uint softbodyId1 = getSoftBodyId(pairInd1);
        const uint tetInd1 = getSoftBodyElementIndex(pairInd1);

        const float rest = softbodies[softbodyId0].mRestDistance + softbodies[softbodyId1].mRestDistance;

        const uint4 tetrahedronIdx0 = Ptr<uint4>(softbodies[softbodyId0].mSimTetIndices)[tetInd0];
        Ptr<float4> position_invmass0 = Ptr<float4>(softbodies[softbodyId0].mSimPosition_InvMass);
        const uint4 tetrahedronIdx1 = Ptr<uint4>(softbodies[softbodyId1].mSimTetIndices)[tetInd1];
        Ptr<float4> position_invmass1 = Ptr<float4>(softbodies[softbodyId1].mSimPosition_InvMass);

        const float4 normal_pen = Ptr<float4>(gPr.normalPens)[workIndex];
        const float4 barycentric0 = Ptr<float4>(gPr.barycentrics0)[workIndex];
        const float4 barycentric1 = Ptr<float4>(gPr.barycentrics1)[workIndex];

        const float invMass0 = computeInvMass(tetrahedronIdx0, position_invmass0, barycentric0);
        const float invMass1 = computeInvMass(tetrahedronIdx1, position_invmass1, barycentric1);

        Ptr<float4>(cb + SSC_barycentric0)[threadIndexInWarp] = barycentric0;
        Ptr<float4>(cb + SSC_barycentric1)[threadIndexInWarp] = barycentric1;
        // The NORMAL IS NEGATED here and the rest distance folded into .w.  Everything
        // downstream reads the constraint block, so the solve never sees the midphase's
        // sign convention.
        Ptr<float4>(cb + SSC_normal_pen)[threadIndexInWarp] =
            float4(-normal_pen.x, -normal_pen.y, -normal_pen.z, -rest + normal_pen.w);
        // Avoid division by zero: contacts between all-zero-invMass nodes DO get generated.
        Ptr<float>(cb + SSC_velMultiplier)[threadIndexInWarp] = 1.0f / max(1e-16f, invMass0 + invMass1);
    }
}

// ===========================================================================
// sb_solveOutputSSDeltaVLaunch{,TGS}
//
// CUDA arg pack (PxgSoftBodyCore.cpp:1624): { PxgSoftBody* softbodies,
// PxgFemFemContactInfo* contactInfos, PxgSoftBodySoftBodyConstraintBlock* constraints,
// PxU32* numContacts, const PxReal dt, const PxReal biasCoefficient, float2* appliedForces,
// PxsDeformableVolumeMaterialData* materials } -> four pointers @0..24, two PxReal @32/@36,
// two pointers @40/@48 -> 56-byte blob.  (ABI row {8,8,8,8,4,4,8,8}.)
//
// PGS and TGS differ in four places and nowhere else, so they share one body behind a
// compile-time-uniform `isTGS`, exactly as sb_solveRigidSoftCollisionLaunch{,TGS} does:
//   - PGS reads mSimPosition_InvMass AND mSimVelocity_InvMass; TGS reads mSimDeltaPos twice
//     (the accumulated position delta plays both roles);
//   - PGS scales the positional error by biasCoefficient, TGS does not (`biasedErr =
//     -projectedErr` outright);
//   - PGS drives the normal impulse from (biasedErr - normalVel), TGS from biasedErr alone;
//   - PGS's tangential magnitude is the relative VELOCITY along the tangent, TGS's is the
//     relative POSITION along it -- and TGS multiplies the final deltaPos by biasCoefficient.
// ===========================================================================
struct SbSsSolveParams
{
    uint64_t softbodies;        // @0
    uint64_t contactInfos;      // @8
    uint64_t constraints;       // @16
    uint64_t numContacts;       // @24
    float dt;                   // @32
    float biasCoefficient;      // @36
    uint64_t appliedForces;     // @40  in/out
    uint64_t materials;         // @48 -> 56
}
[[vk::push_constant]]
ConstantBuffer<SbSsSolveParams> gSv;

void solveOutputSSDeltaV(bool isTGS, uint lane, uint blockX)
{
    if (blockX >= SS_GRID)
        return;

    const uint tNumContacts = Ptr<uint>(gSv.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + SS_BLOCK - 1u) / SS_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + SS_GRID - 1u) / SS_GRID;

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gSv.softbodies);

    for (uint i = 0u; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * SS_BLOCK + lane + nbIterationsPerBlock * blockX * SS_BLOCK;
        if (workIndex >= tNumContacts)
            break;

        const uint threadIndexInWarp = lane & 31u;   // CUDA's threadIdx.x & 31; see the
                                                     // prepare above on why this equals
                                                     // workIndex & 31 for every dispatch
        const uint64_t cb = ssBlockBase(gSv.constraints, workIndex);
        const uint64_t ciBase = gSv.contactInfos + uint64_t(workIndex) * FFCI_STRIDE;

        const uint pairInd0 = Ptr<uint>(ciBase)[0];
        const float4 barycentric0 = Ptr<float4>(cb + SSC_barycentric0)[threadIndexInWarp];
        const uint softbodyId0 = getSoftBodyId(pairInd0);
        const uint tetId0 = getSoftBodyElementIndex(pairInd0);
        const uint4 tetrahedronId0 = Ptr<uint4>(softbodies[softbodyId0].mSimTetIndices)[tetId0];

        const uint pairInd1 = Ptr<uint>(ciBase + 8UL)[0];
        const float4 barycentric1 = Ptr<float4>(cb + SSC_barycentric1)[threadIndexInWarp];
        const uint softbodyId1 = getSoftBodyId(pairInd1);
        const uint tetId1 = getSoftBodyElementIndex(pairInd1);
        const uint4 tetrahedronId1 = Ptr<uint4>(softbodies[softbodyId1].mSimTetIndices)[tetId1];

        // TGS works entirely in accumulated position deltas; PGS in world positions.
        const uint64_t posAddr0 = isTGS ? softbodies[softbodyId0].mSimDeltaPos
                                        : softbodies[softbodyId0].mSimPosition_InvMass;
        const uint64_t posAddr1 = isTGS ? softbodies[softbodyId1].mSimDeltaPos
                                        : softbodies[softbodyId1].mSimPosition_InvMass;

        float4 invMasses0, invMasses1;
        const float4 pos0 = computeTetraContact(Ptr<float4>(posAddr0), tetrahedronId0, barycentric0, invMasses0);
        const float4 pos1 = computeTetraContact(Ptr<float4>(posAddr1), tetrahedronId1, barycentric1, invMasses1);

        float3 relVel = float3(0, 0, 0);
        if (!isTGS)
        {
            // NOTE the invMasses out-parameters are OVERWRITTEN by this second call in
            // CUDA too (softBody.cu:1471-1476).  mSimVelocity_InvMass carries the same .w
            // as mSimPosition_InvMass, so the value is unchanged -- but the write order is
            // transcribed rather than "cleaned up", because if the two ever diverge the
            // port must diverge with upstream, not silently disagree.
            float4 vm0, vm1;
            const float4 vel0 = computeTetraContact(Ptr<float4>(softbodies[softbodyId0].mSimVelocity_InvMass),
                                                    tetrahedronId0, barycentric0, vm0);
            const float4 vel1 = computeTetraContact(Ptr<float4>(softbodies[softbodyId1].mSimVelocity_InvMass),
                                                    tetrahedronId1, barycentric1, vm1);
            invMasses0 = vm0;
            invMasses1 = vm1;
            relVel = vel1.xyz - vel0.xyz;
        }

        const uint matIdx0 = materialIndexAt(softbodies[softbodyId0].mMaterialIndices, tetId0);
        const uint matIdx1 = materialIndexAt(softbodies[softbodyId1].mMaterialIndices, tetId1);
        const float frictionCoefficient =
            (materialDynamicFriction(gSv.materials, matIdx0) +
             materialDynamicFriction(gSv.materials, matIdx1)) * 0.5f;

        const float4 normal_pen = Ptr<float4>(cb + SSC_normal_pen)[threadIndexInWarp];
        const float3 normal = normal_pen.xyz;
        const float rest = normal_pen.w;
        const float velMultiplier = Ptr<float>(cb + SSC_velMultiplier)[threadIndexInWarp];

        const float4 relPos = pos1 - pos0;
        const float projectedErr = dot(relPos.xyz, normal) + rest;

        float2 appliedForce = Ptr<float2>(gSv.appliedForces)[workIndex];

        float3 tanDir;
        float tanMagnitude;
        float biasedErr;
        float normalVel;
        if (isTGS)
        {
            const float3 rPos = relPos.xyz;
            const float normalPos = dot(rPos, normal);
            tanDir = rPos - normal * normalPos;
            tanMagnitude = vecNormalizeInPlace(tanDir);
            biasedErr = -projectedErr;
            normalVel = 0.0f;
        }
        else
        {
            normalVel = dot(relVel, normal);
            tanDir = relVel - normal * normalVel;
            tanMagnitude = vecNormalizeInPlace(tanDir);
            biasedErr = -projectedErr * gSv.biasCoefficient;
        }

        // Normal force can only be +ve.
        const float deltaF = max(-appliedForce.x, (biasedErr - normalVel) * velMultiplier);
        appliedForce.x += deltaF;

        const float friction = appliedForce.x * frictionCoefficient;
        const float requiredForce = tanMagnitude * velMultiplier;
        // requiredForce is always positive.
        const float deltaFr = min(requiredForce + appliedForce.y, friction) - appliedForce.y;
        appliedForce.y += deltaFr;

        if (deltaF != 0.0f || deltaFr != 0.0f)
        {
            float3 deltaPos = (-(normal * deltaF) + tanDir * deltaFr) * gSv.dt;
            if (isTGS)
                deltaPos = (-(normal * deltaF) + tanDir * deltaFr) * gSv.biasCoefficient * gSv.dt;

            // The symmetry: the SAME impulse, opposite signs, into both tets' delta buffers.
            updateTetraPosDelta(invMasses0, barycentric0, tetrahedronId0, deltaPos,
                                softbodies[softbodyId0].mSimDelta);
            updateTetraPosDelta(invMasses1, barycentric1, tetrahedronId1, -deltaPos,
                                softbodies[softbodyId1].mSimDelta);
        }

        Ptr<float2>(gSv.appliedForces)[workIndex] = appliedForce;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sb_solveOutputSSDeltaVLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    solveOutputSSDeltaV(/*isTGS*/ false, gtid.x, gid.x);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sb_solveOutputSSDeltaVLaunchTGS(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    solveOutputSSDeltaV(/*isTGS*/ true, gtid.x, gid.x);
}
