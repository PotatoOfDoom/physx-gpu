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


// ===========================================================================
// femClothDampingBda.slang -- the cloth DAMPING trio (P9.19).
//
// Ports three kernels from gpusimulationcontroller/src/CUDA/FEMCloth.cu, all
// read in full:
//
//   cloth_accumulateInPlaneDampingDeltaVelocity  (:947)
//   cloth_accumulateBendingDampingDeltaVelocity  (:1046)
//   cloth_applyAccumulatedDeltaVelocity          (:918)
//
// They are one feature and land together.  PxgFEMClothCore::applyDamping
// (:2023) launches them back to back, in that order, and the third only makes
// sense after the first two: the accumulators atomically add into
// mAccumulatedDeltaVel and the applier folds that buffer into the velocities
// and clears it.  Splitting the tranche would land a producer with no consumer.
//
// THE PHYSICS, so the traps below read as consequences rather than trivia.
// Both accumulators implement the velocity-level damping of "Detailed Rigid
// Body Simulation with Extended Position Based Dynamics" (the CUDA cites its
// formula 33 for the mass weights).  The idea is to remove the part of the
// relative motion that is NOT a rigid motion of the element:
//
//   * in-plane, per TRIANGLE: subtract the mean velocity, project what is left
//     into the triangle's plane, extract the single rigid ROTATION rate about
//     the centroid that best fits the three projected velocities, and damp
//     everything except that rotation.
//   * bending, per TRIANGLE PAIR: the two shared-edge vertices define the hinge
//     axis; damp the DIFFERENCE between the two tips' rotation rates about it,
//     which is exactly the hinge's opening/closing speed.
//
// Both then subtract the mean of their own deltas so the correction adds no net
// linear momentum, and only then weight by inverse mass.
//
// All three block at 64 (PxgSoftBodyKernelBlockDim::SB_SOLVETETRA -- the CLOTH
// core reaches into the SOFT BODY block-dim table for these three, which is
// surprising enough to write down) and take a 2D grid whose y is the active
// cloth index.
// ===========================================================================

module femClothDampingBda;

import pxgSlangFEMClothDesc;   // PxgFEMClothGpu, PxsDeformableSurfaceMaterialGpu
import pxgSlangAtomicFloat;   // pxgAtomicAddF4XYZ -- x, y and z only, never w

// PxgSoftBodyKernelBlockDim::SB_SOLVETETRA (PxgSoftBodyCoreKernelIndices.h:43).
static const uint DAMP_BLOCK = 64u;

// ---------------------------------------------------------------------------
// PxVec3::normalize() (PxVec3.h:295) -- returns the magnitude BEFORE scaling,
// and divides ONLY when that magnitude is strictly positive.
//
// D3: every caller here tests the RETURN value against a threshold and keeps
// the normalised vector.  A port that returns the post-scaling length, or that
// divides unconditionally, changes both the guard and the value.
// ---------------------------------------------------------------------------
float dmpNormalize(inout float3 v)
{
    // sqrt(dot(v,v)), NOT length(v): PxVec3::magnitude() is PxSqrt of
    // magnitudeSquared, and Slang's `length` is free to lower to a different
    // sequence.  The first draft used `length` and disagreed with the CPU mirror
    // by one ULP on two of five vertices -- close enough to look like noise and
    // far enough to be a different number.
    const float m = sqrt(dot(v, v));
    // RECIPROCAL-MULTIPLY, not a componentwise divide: normalize() delegates to
    // operator/= (PxVec3.h:260-267), which forms 1.0/m once and multiplies.  P9.36.
    if (m > 0.0f)
        v = v * (1.0f / m);
    return m;
}

// deformableUtils.cuh:118.  D4: the helper divides by magnitudeSquared even
// though every call site here passes an already-normalised plane normal.  That
// division is transcribed rather than folded away: it is the CUDA's arithmetic,
// and 1/|n|^2 for a freshly normalised n is not exactly 1.0f.
float3 dmpProjectVectorOntoPlane(float3 v, float3 planeNormal)
{
    return v - (dot(planeNormal, v) / dot(planeNormal, planeNormal)) * planeNormal;
}

// D1: atomic.cuh:83 -- AtomicAdd(float4* p, PxU32 i, const PxVec3& v) touches x,
// y and z ONLY.  `w` carries the inverse mass in these buffers and must survive;
// a four-component add corrupts every vertex it touches.  pxgAtomicAddF4XYZ is
// that overload, already shared.

// ===========================================================================
// cloth_applyAccumulatedDeltaVelocity (FEMCloth.cu:918)
//
// CUDA arg pack (PxgFEMClothCore.cpp:2089): { PxgFEMCloth* femCloths,
// const PxU32* activeId } -> 16 bytes.
// ===========================================================================
struct ApplyDampParams
{
    uint64_t femCloths;   // @0
    uint64_t activeIds;   // @8  -> 16
}
[[vk::push_constant]]
ConstantBuffer<ApplyDampParams> gApply;

// IEEE `0 + x` and `0 - x` per component, as selects (P9.86): +0 for either zero,
// x / -x otherwise.  No arithmetic identity for slangc or the driver to fold.
float3 cdPlusZero3(float3 x)
{
    return float3((x.x == 0.0f) ? 0.0f : x.x, (x.y == 0.0f) ? 0.0f : x.y, (x.z == 0.0f) ? 0.0f : x.z);
}
float3 cdZeroMinus3(float3 x)
{
    return float3((x.x == 0.0f) ? 0.0f : -x.x, (x.y == 0.0f) ? 0.0f : -x.y, (x.z == 0.0f) ? 0.0f : -x.z);
}

[shader("compute")]
[numthreads(64, 1, 1)] // SB_SOLVETETRA
void cloth_applyAccumulatedDeltaVelocity(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint id = Ptr<uint>(gApply.activeIds)[gid.y];
    Ptr<PxgFEMClothGpu> femCloths = Ptr<PxgFEMClothGpu>(gApply.femCloths);

    const uint globalThreadIndex = gtid.x + DAMP_BLOCK * gid.x;
    if (globalThreadIndex >= femCloths[id].mNbVerts)
        return;

    Ptr<float4> velDelta = Ptr<float4>(femCloths[id].mAccumulatedDeltaVel);
    Ptr<float4> velocity = Ptr<float4>(femCloths[id].mVelocity_InvMass);

    const float4 d = velDelta[globalThreadIndex];
    const float4 v = velocity[globalThreadIndex];

    // D1 again, and here it is a plain store rather than an atomic: the CUDA
    // assigns .x, .y and .z one at a time and never touches .w, which holds the
    // inverse mass.  Writing the whole float4 would zero every vertex's mass.
    velocity[globalThreadIndex] = float4(v.x + d.x, v.y + d.y, v.z + d.z, v.w);

    // The delta buffer IS cleared in full -- all four components, including w.
    velDelta[globalThreadIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
}

// ===========================================================================
// cloth_accumulateInPlaneDampingDeltaVelocity (FEMCloth.cu:947)
//
// CUDA arg pack (PxgFEMClothCore.cpp:2046): { PxgFEMCloth* femCloths,
// const PxU32* activeId, const PxsDeformableSurfaceMaterialData* clothMaterials,
// PxReal dt } -> 8 + 8 + 8 + 4, natural alignment -> 32 bytes.
// ===========================================================================
struct InPlaneDampParams
{
    uint64_t femCloths;   // @0
    uint64_t activeIds;   // @8
    uint64_t materials;   // @16
    float dt;             // @24
    uint _pad28;          // @28 -> 32
}
[[vk::push_constant]]
ConstantBuffer<InPlaneDampParams> gInPlane;

[shader("compute")]
[numthreads(64, 1, 1)] // SB_SOLVETETRA
void cloth_accumulateInPlaneDampingDeltaVelocity(uint3 gtid : SV_GroupThreadID,
                                                 uint3 gid : SV_GroupID)
{
    const uint id = Ptr<uint>(gInPlane.activeIds)[gid.y];
    Ptr<PxgFEMClothGpu> femCloths = Ptr<PxgFEMClothGpu>(gInPlane.femCloths);

    const uint globalThreadIndex = gtid.x + DAMP_BLOCK * gid.x;
    if (globalThreadIndex >= femCloths[id].mNbTriangles)
        return;

    // mMaterialIndices is a PxU16*; llvmpipe has no 16-bit storage, so this is
    // the established word-read + half-extract (femClothSolveBda.slang:106).
    // D12: the index is the TRIANGLE index, not a vertex index -- the material
    // is a property of the triangle.
    const uint byteOff = 2u * globalThreadIndex;
    const uint word = Ptr<uint>(femCloths[id].mMaterialIndices)[byteOff >> 2];
    const uint materialIndex = ((byteOff >> 1) & 1u) != 0u ? (word >> 16) : (word & 0xffffu);

    Ptr<PxsDeformableSurfaceMaterialGpu> materials =
        Ptr<PxsDeformableSurfaceMaterialGpu>(gInPlane.materials);

    // D2: the early-out tests the CLAMPED factor against exactly zero, so a
    // NEGATIVE elasticityDamping also returns here (clamp pulls it to 0).  A
    // port that tests the raw product would run the whole body with a negative
    // factor and AMPLIFY the motion it is meant to damp.
    const float dampingFactor = clamp(materials[materialIndex].elasticityDamping * gInPlane.dt,
                                      0.0f, 1.0f);
    if (dampingFactor == 0.0f)
        return;

    const uint4 triangle = Ptr<uint4>(femCloths[id].mTriangleVertexIndices)[globalThreadIndex];

    Ptr<float4> positions = Ptr<float4>(femCloths[id].mPosition_InvMass);
    Ptr<float4> velocities = Ptr<float4>(femCloths[id].mVelocity_InvMass);

    const float4 pa4 = positions[triangle.x];
    const float4 pb4 = positions[triangle.y];
    const float4 pc4 = positions[triangle.z];
    const float invMassA = pa4.w, invMassB = pb4.w, invMassC = pc4.w;
    const float3 pA = pa4.xyz, pB = pb4.xyz, pC = pc4.xyz;

    const float3 vA = velocities[triangle.x].xyz;
    const float3 vB = velocities[triangle.y].xyz;
    const float3 vC = velocities[triangle.z].xyz;

    const float3 linearVelocity = (1.0f / 3.0f) * (vA + vB + vC);
    const float3 centroid = (1.0f / 3.0f) * (pA + pB + pC);

    float3 normal = cross(pB - pA, pC - pA);
    if (dmpNormalize(normal) < 1e-6f)
        return;

    // Only the in-plane part of each vertex's velocity relative to the mean.
    const float3 vAProj = dmpProjectVectorOntoPlane(vA - linearVelocity, normal);
    const float3 vBProj = dmpProjectVectorOntoPlane(vB - linearVelocity, normal);
    const float3 vCProj = dmpProjectVectorOntoPlane(vC - linearVelocity, normal);

    // D5: normal is unit and perpendicular to (p - centroid), which lies in the
    // plane, so |normal x (p - centroid)| IS the distance to the centroid.  The
    // CUDA relies on that identity and its comment says so ("Works because
    // edgeDir has unit length" at the bending twin).  rigidVelDir* comes back
    // normalised and dist* carries the radius.
    float3 rigidVelDirA = cross(normal, pA - centroid);
    float3 rigidVelDirB = cross(normal, pB - centroid);
    float3 rigidVelDirC = cross(normal, pC - centroid);
    const float distCenterA = dmpNormalize(rigidVelDirA);
    const float distCenterB = dmpNormalize(rigidVelDirB);
    const float distCenterC = dmpNormalize(rigidVelDirC);

    // D6 -- UPSTREAM CANDIDATE, transcribed bug-for-bug.  A vertex sitting
    // exactly on the centroid gives distCenter == 0 and this divides by it.  The
    // CUDA has no guard; neither does this.  Documented, not reported.
    const float rotVelA = dot(rigidVelDirA, vAProj) / distCenterA;
    const float rotVelB = dot(rigidVelDirB, vBProj) / distCenterB;
    const float rotVelC = dot(rigidVelDirC, vCProj) / distCenterC;

    const float rotVel = (1.0f / 3.0f) * (rotVelA + rotVelB + rotVelC);

    // What is left after removing the best-fit rigid rotation -- that is what
    // gets damped away.
    float3 deltaVelA = rotVel * distCenterA * rigidVelDirA - vAProj;
    float3 deltaVelB = rotVel * distCenterB * rigidVelDirB - vBProj;
    float3 deltaVelC = rotVel * distCenterC * rigidVelDirC - vCProj;

    // D7 -- UPSTREAM CANDIDATE.  Every vertex pinned gives a zero sum and an
    // infinite weightFactor.  No guard in the CUDA, none here.
    const float weightFactor = 1.0f / (invMassA + invMassB + invMassC);

    // D8 -- THE ORDER IS LOAD-BEARING.  The mean is removed BEFORE the mass
    // weighting, so the correction carries no net linear momentum in the RAW
    // deltas.  The CUDA carries a commented-out block that weights first; doing
    // it in that order leaves a residual force and is a different result.
    const float3 deltaAvg = (deltaVelA + deltaVelB + deltaVelC) * (1.0f / 3.0f);
    deltaVelA -= deltaAvg;
    deltaVelB -= deltaAvg;
    deltaVelC -= deltaAvg;

    deltaVelA *= invMassA * weightFactor * dampingFactor;
    deltaVelB *= invMassB * weightFactor * dampingFactor;
    deltaVelC *= invMassC * weightFactor * dampingFactor;

    const uint64_t velDelta = femCloths[id].mAccumulatedDeltaVel;
    if (invMassA != 0.0f) pxgAtomicAddF4XYZ(velDelta, triangle.x, deltaVelA);
    if (invMassB != 0.0f) pxgAtomicAddF4XYZ(velDelta, triangle.y, deltaVelB);
    if (invMassC != 0.0f) pxgAtomicAddF4XYZ(velDelta, triangle.z, deltaVelC);
}

// ===========================================================================
// cloth_accumulateBendingDampingDeltaVelocity (FEMCloth.cu:1046)
//
// CUDA arg pack (PxgFEMClothCore.cpp:2068): { PxgFEMCloth* femCloths,
// const PxU32* activeId, PxReal dt } -> 8 + 8 + 4, natural alignment -> 24.
//
// The hinge: the two triangles of a pair share an EDGE, and the pair's uint4
// names four vertices.  D9 -- THE COMPONENTS ARE NOT IN THE ORDER THEY READ.
// The CUDA binds
//
//     A = triangle.z,  B = triangle.w    the shared EDGE (the hinge axis)
//     C = triangle.x,  C2 = triangle.y   the two opposite TIPS
//
// so a port that takes .xyz as "the triangle" and .w as a tail -- which is what
// every OTHER cloth kernel in this corpus does -- hinges about the wrong pair
// and damps the wrong rotation.  It still runs, and it still produces plausible
// numbers.
// ===========================================================================
struct BendingDampParams
{
    uint64_t femCloths;   // @0
    uint64_t activeIds;   // @8
    float dt;             // @16
    uint _pad20;          // @20 -> 24
}
[[vk::push_constant]]
ConstantBuffer<BendingDampParams> gBend;

[shader("compute")]
[numthreads(64, 1, 1)] // SB_SOLVETETRA
void cloth_accumulateBendingDampingDeltaVelocity(uint3 gtid : SV_GroupThreadID,
                                                  uint3 gid : SV_GroupID)
{
    const uint id = Ptr<uint>(gBend.activeIds)[gid.y];
    Ptr<PxgFEMClothGpu> femCloths = Ptr<PxgFEMClothGpu>(gBend.femCloths);

    const uint globalThreadIndex = gtid.x + DAMP_BLOCK * gid.x;
    if (globalThreadIndex >= femCloths[id].mNbTrianglePairs)
        return;

    // D10: the shared and non-shared pairs live in SEPARATE ordered arrays and
    // the second is indexed from zero, not from the global index.  The split is
    // at mNbSharedTrianglePairs -- the same layout femClothSolvePairBda uses.
    uint4 triangle;
    float bendingDamping, bendingStiffness;
    const uint nbShared = femCloths[id].mNbSharedTrianglePairs;
    if (globalThreadIndex < nbShared)
    {
        triangle = Ptr<uint4>(femCloths[id].mOrderedSharedTrianglePairVertexIndices)[globalThreadIndex];
        const float4 c = Ptr<float4>(
            femCloths[id].mOrderedSharedRestBendingAngle_flexuralStiffness_damping)[globalThreadIndex];
        bendingDamping = c.z;
        bendingStiffness = c.y;
    }
    else
    {
        const uint k = globalThreadIndex - nbShared;
        triangle = Ptr<uint4>(femCloths[id].mOrderedNonSharedTrianglePairVertexIndices)[k];
        const float4 c = Ptr<float4>(
            femCloths[id].mOrderedNonSharedRestBendingAngle_flexuralStiffness_damping)[k];
        bendingDamping = c.z;
        bendingStiffness = c.y;
    }

    // D11: the gate is on the CLAMPED damping factor OR a zero flexural
    // stiffness.  A hinge with no stiffness is not a hinge, so damping it would
    // be damping a free fold.
    const float dampingFactor = clamp(bendingDamping * gBend.dt, 0.0f, 1.0f);
    if (dampingFactor == 0.0f || bendingStiffness == 0.0f)
        return;

    Ptr<float4> positions = Ptr<float4>(femCloths[id].mPosition_InvMass);
    Ptr<float4> velocities = Ptr<float4>(femCloths[id].mVelocity_InvMass);

    // D9, spelled out at the loads.
    const float3 velA = velocities[triangle.z].xyz;   // edge
    const float3 velB = velocities[triangle.w].xyz;   // edge
    const float3 velC = velocities[triangle.x].xyz;   // tip 1
    const float3 velC2 = velocities[triangle.y].xyz;  // tip 2

    const float4 pa4 = positions[triangle.z];
    const float4 pb4 = positions[triangle.w];
    const float4 pc4 = positions[triangle.x];
    const float4 pd4 = positions[triangle.y];
    const float invMassA = pa4.w, invMassB = pb4.w, invMassC = pc4.w, invMassC2 = pd4.w;
    const float3 posA = pa4.xyz, posB = pb4.xyz, posC = pc4.xyz, posC2 = pd4.xyz;

    // D13: the reference velocity is the mean of the TWO EDGE vertices only, not
    // of all four.  The CUDA's own comment flags the choice ("could also take
    // the average over all 4"), so it is a decision and not an oversight -- and
    // averaging four is a different, plausible-looking result.
    const float3 linearVelocity = 0.5f * (velA + velB);

    const float3 start = posA;

    float3 edgeDir = posB - posA;
    if (dmpNormalize(edgeDir) < 1e-6f)
        return;

    const float3 relVelTip1 = velC - linearVelocity;
    const float3 relVelTip2 = velC2 - linearVelocity;

    const float3 startToTip1 = posC - start;
    const float3 startToTip2 = posC2 - start;

    // Same identity as D5: edgeDir is unit, so the cross product's length is the
    // tip's perpendicular distance to the hinge axis.
    float3 tip1Dir = cross(edgeDir, startToTip1);
    const float distanceToTip1 = dmpNormalize(tip1Dir);
    float3 tip2Dir = cross(edgeDir, startToTip2);
    const float distanceToTip2 = dmpNormalize(tip2Dir);

    // D6 again: no guard against a tip ON the hinge axis.
    const float rotVelTip1 = dot(tip1Dir, relVelTip1) / distanceToTip1;
    const float rotVelTip2 = dot(tip2Dir, relVelTip2) / distanceToTip2;

    // The hinge's opening speed, and the ONLY thing this kernel damps.
    const float deltaRotVel = rotVelTip2 - rotVelTip1;
    const float deltaRotVelDamp = deltaRotVel * dampingFactor;

    // D14 -- THE SIGNS ARE OPPOSITE.  The two tips are pushed toward each other
    // in rotation, which is what closes the hinge; the same sign on both would
    // rotate the whole pair instead of damping its fold.
    float3 deltaVelTip1 = deltaRotVelDamp * distanceToTip1 * tip1Dir;
    float3 deltaVelTip2 = -deltaRotVelDamp * distanceToTip2 * tip2Dir;
    const float weightFactor = 1.0f / (invMassA + invMassB + invMassC + invMassC2);

    // D8's twin: the mean of FOUR deltas, removed before the mass weighting.
    // P9.86: deltaVelA and deltaVelB are PxVec3(0) (FEMCloth.cu:1131-1132), so the CUDA
    // computes ((+0 + +0) + tip1) + tip2 and then 0 - deltaAvg.  Spelled literally,
    // slangc folds the first add to `tip1` (it drops IEEE's -0 -> +0 on 0 + x, even
    // with -fp-mode precise) and emits the subtraction as OpFSub(0, x), which llvmpipe
    // rewrites to -x under SignedZeroInfNanPreserve too.  Both are spelled as selects.
    const float3 deltaAvg = (cdPlusZero3(deltaVelTip1) + deltaVelTip2) * 0.25f;
    float3 deltaVelA = cdZeroMinus3(deltaAvg);
    float3 deltaVelB = cdZeroMinus3(deltaAvg);
    deltaVelTip1 -= deltaAvg;
    deltaVelTip2 -= deltaAvg;

    // D15: NO dampingFactor here.  It was already folded in at deltaRotVelDamp,
    // and the in-plane twin -- which does multiply by it at this point -- makes
    // the asymmetry easy to "fix" into a squared factor.
    deltaVelTip1 *= invMassC * weightFactor;
    deltaVelTip2 *= invMassC2 * weightFactor;
    deltaVelA *= invMassA * weightFactor;
    deltaVelB *= invMassB * weightFactor;

    const uint64_t velDelta = femCloths[id].mAccumulatedDeltaVel;
    if (invMassA != 0.0f) pxgAtomicAddF4XYZ(velDelta, triangle.z, deltaVelA);
    if (invMassB != 0.0f) pxgAtomicAddF4XYZ(velDelta, triangle.w, deltaVelB);
    if (invMassC != 0.0f) pxgAtomicAddF4XYZ(velDelta, triangle.x, deltaVelTip1);
    if (invMassC2 != 0.0f) pxgAtomicAddF4XYZ(velDelta, triangle.y, deltaVelTip2);
}
