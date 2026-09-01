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

// npArtiUpdateBodies1T.slang
//
// P9.43b: `updateBodiesLaunch1T` (forwardDynamic2.cu:3047-3075, body
// updateBodiesInternal<integrateJointPositions> :2984-3045) -- the FIRST of the
// three end-of-step articulation stages, one thread per articulation, block
// (32, 2, 1), grid ceil(nbArticulations / 64), the {desc, dt, bool} arg pack the
// other two stages share.
//
// WHAT IT IS FOR.  This kernel is the ONLY writer of the per-articulation record
// arrays `jointPositions` / `jointVelocities` / `jointAccelerations` (:3035-3044):
// the DMA-back ships them to the host, PxgPostSolveArticulationTask copies them
// into Dy::ArticulationData (PxgContext.cpp:826-834), and PxArticulationCache
// reads them.  P9.40's guard found it skipped x240 in the articulation scene --
// the link visibly hangs at -pi/2 while copyInternalStateToCache reports 0.
// PXG_SCENE=articulationjoint measures exactly that.
//
// FOUR SECTIONS, IN THIS ORDER (trap T7: any reordering changes every output):
//   1. poses / joint positions.  integrate == false (TGS, the scene default):
//      computeAndEnforceJointPositions only WRAPS revolute positions, CLAMPS
//      limited prismatic ones and RE-DERIVES spherical ones from the poses
//      stepArticulation1TTGS already integrated -- no pose is written.
//      integrate == true (PGS): updatePoses integrates the root from
//      mPosMotionVelocity (NOT mMotionVelocity, trap T6) and the links through
//      propagateLink<false> (mPosJointVelocities, trap T5).
//   2. the deferred impulse flush -- byte for byte npArtiStepKernel's: average
//      when eHAS_IMPULSES, then the SIX-argument PxcFsFlushVelocity (the one that
//      accumulates mSolverSpatialDeltaVel for Part2, trap T1), then the atomic
//      byte clear of mStateDirty.
//   3. conserveMomentum<doIntegrate>: for a FREE base only, rewrite every
//      non-root link velocity by forward kinematics from the joint velocities,
//      then correct angular velocities (compound inertia), linear velocities and
//      link POSITIONS so the articulation's momentum and centre of mass match
//      the pre-step values.  Its arithmetic is pure add/mul/div/sqrt and is
//      bit-exact against the CPU when the association below is kept.
//   4. the dof writeback into the record: p, v, and a = accel + (v - vUnc) * invDt.
//
// THE MOMENTUM TAIL IS NOT npArtiLinkAccel's (trap T13).  It looks like section 5
// of alaComputeLinkAcceleration and differs in four places: the angular-ratio
// guard returns 0 here (1 there), velChange.bottom is MINUS offset x angDelta
// here (plus there), the compound inertia is transformInertiaTensor(1/invInertia,
// R(pose.q)) translated by mass here (the stored isolated inertia there), and
// this one additionally rewrites the link velocities by FK and moves the link
// positions.  The helpers are shared; the constants and signs are this kernel's.
//
// Accumulators start from asiRuntimeZero(dt), and the three `+ 0.0f` adds of
// addSpatialVector / the pose's p.w use the same runtime zero (trap T8: slangc
// folds `x + 0.0` away, IEEE does not -- the sign of a zero on a zero-mass link
// would come out wrong).
//
// Transcribed quirks, kept: computeAndEnforceJointPositions reads mJointOffset
// through a PxU8 local (:2463) -- the PxU32 field is truncated to its low byte
// there and only there (trap T3; unreachable below 256 dofs per articulation,
// recorded as an upstream candidate).  The reciprocals of the inverse inertia
// diagonal (:2882) are unguarded true divides (trap T12).  The prefetch of
// artiLinks[1].mParents (:2821) is unconditional in the .cu and reads past a
// one-link articulation's slice; it is guarded on linkCount > 1 here, which is
// write-equivalent (the value is dead when the loop has zero trips) and keeps
// llvmpipe off a possibly unmapped page (trap T9, the U01 class).

import pxgSlangCommon;
import pxgSlangArticulationDesc;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;
import npArtiStep;
import npArtiFlushVel;
import npArtiAvgImpulse;
import npArtiSpatialInertia;

static const uint AU1_WARPS_PER_BLOCK = 2u;      // COMPUTE_UNCONSTRAINED_VELOCITES / 32
static const float AU1_TWO_PI = 6.28318530717958647692f; // PxTwoPi
// PxArticulationJointType (PxSolverDefs.h:301-306) / PxArticulationMotion::eLIMITED
static const uint AU1_JOINT_PRISMATIC = 1u;
static const uint AU1_JOINT_REVOLUTE = 2u;
static const uint AU1_JOINT_SPHERICAL = 4u;
static const uint AU1_MOTION_LIMITED = 1u;

AbSpatialVector au1LoadSV(uint64_t base, uint fieldOffset, uint lane)
{
    float3 top, bottom;
    abLoadSpatialVector(base, fieldOffset, lane, top, bottom);
    return asmSVec(top, bottom);
}

// PxVec3::magnitude (PxVec3.h:179): sqrtf of the explicit sum, never a length intrinsic.
float au1Magnitude(float3 v)
{
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// `x + 0.0f` as the engine computes it -- and that is NOT the identity: IEEE-754
// gives (-0) + (+0) = +0, so the one input the add changes is a negative zero.
//
// A FIFTH PLATFORM RULE, measured in this tranche (P9.43b): the P8.87 runtime zero
// (`abs(dt) * 0.0f`, npArtiSpatialInertia.slang) survives slangc, but llvmpipe's
// LLVM proves it to be a zero anyway and, running with no-signed-zeros semantics,
// folds `x + thatZero` back to `x` whenever the zero reaches the add DIRECTLY.  It
// cannot do so through a loop-carried phi, which is why the P8.87 accumulator
// seeds work (and the momentum accumulators above do, verified bit for bit) and
// why addSpatialVector's three `+ 0.0f` and the pose's `p.w + 0.f` -- straight-line
// adds -- came back as -0.0 against the engine's +0.0 on every resting link.
// So the add is spelled as what it does: canonicalise -0.0 to +0.0 in integer
// space, where no floating-point flag applies.  NaN, infinities and every other
// value pass through unchanged, exactly as through the real add.
float au1PlusZero(float x)
{
    return asfloat(asuint(x) == 0x80000000u ? 0u : asuint(x));
}

// The CUDA arg pack: desc @0, dt @8, the bool in byte 12 -> 16-byte blob (the P8.83
// pattern; npArtiUpdateBodies.slang's AubParams verbatim).
struct Au1Params
{
    uint64_t desc;        // @0
    float dt;             // @8
    uint integrateWord;   // @12: the CUDA bool in byte 0
};

[[vk::push_constant]]
ConstantBuffer<Au1Params> gAu1;

// ---------------------------------------------------------------------------
// computeAndEnforceJointPositions (forwardDynamic2.cu:2452-2512), integrate == false
// ---------------------------------------------------------------------------
void au1ComputeAndEnforceJointPositions(uint64_t linkBlocks, uint64_t dofBlocks, uint linkCount,
                                        uint lane)
{
    for (uint linkID = 1u; linkID < linkCount; ++linkID)
    {
        const uint64_t link = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;
        const uint jointType = abU8(link, PXG_AB_LINK_mJointType, lane);
        // trap T3: `PxU8 jointOffset = link.mJointOffset[...]` (:2463)
        const uint jointOffset = abU32(link, PXG_AB_LINK_mJointOffset, lane) & 0xFFu;
        const uint64_t dof = dofBlocks + uint64_t(jointOffset) * PXG_AB_DOF_SIZE;

        // "We don't need to deal with eREVOLUTE_UNWRAPPED case because there are no
        // joint position changed" (:2466) -- eFIX, eREVOLUTE_UNWRAPPED, eUNDEFINED do nothing.
        if (jointType == AU1_JOINT_REVOLUTE)
        {
            float jPos = abF32(dof, PXG_AB_DOF_mJointPositions, lane);
            if (jPos > AU1_TWO_PI)
                jPos -= AU1_TWO_PI * 2.0f;
            else if (jPos < -AU1_TWO_PI)
                jPos += AU1_TWO_PI * 2.0f;
            // PxClamp(v, lo, hi) = PxMin(hi, PxMax(lo, v)) (PxMath.h:142-146), as ternaries
            const float lo = -AU1_TWO_PI * 2.0f;
            const float hi = AU1_TWO_PI * 2.0f;
            const float mx = (lo < jPos) ? jPos : lo;
            jPos = (hi < mx) ? hi : mx;
            abStoreF32(dof, PXG_AB_DOF_mJointPositions, lane, jPos);
        }
        else if (jointType == AU1_JOINT_PRISMATIC)
        {
            if (abU8(dof, PXG_AB_DOF_mMotion, lane) == AU1_MOTION_LIMITED)
            {
                // trap T15: a float2 at stride 8 inside mConstraintData
                const float2 limits = abF2(dof, PXG_AB_DOF_mConstraintData + PXG_AB_IC_mLimits_LowLimitX_highLimitY, lane);
                float jPosition = abF32(dof, PXG_AB_DOF_mJointPositions, lane);
                if (jPosition < limits.x)
                    jPosition = limits.x;
                if (jPosition > limits.y)
                    jPosition = limits.y;
                abStoreF32(dof, PXG_AB_DOF_mJointPositions, lane, jPosition);
            }
        }
        else if (jointType == AU1_JOINT_SPHERICAL)
        {
            const uint parent = abU32(link, PXG_AB_LINK_mParents, lane);
            const float4 pBody2WorldRot = abF4(linkBlocks + uint64_t(parent) * PXG_AB_LINK_SIZE,
                                               PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_Q, lane);
            const float4 newRot = abF4(link, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_Q, lane);
            const float4 relativeQuat = abF4(link, PXG_AB_LINK_mRelativeQuat, lane);
            // the return value is discarded; the side effect is dof[d].mJointPositions
            asComputeSphericalJointPositions(relativeQuat, newRot, pBody2WorldRot, dof,
                                             abU8(link, PXG_AB_LINK_mDofs, lane), lane);
        }
    }
}

// ---------------------------------------------------------------------------
// updatePoses (forwardDynamic2.cu:2688-2704) with propagateLinksDown (:2673-2685),
// integrate == true
// ---------------------------------------------------------------------------
void au1UpdatePoses(uint64_t linkBlocks, uint64_t dofBlocks, uint linkCount, bool fixBase, float dt,
                    uint lane)
{
    if (!fixBase)
    {
        // trap T6: the POSITION-iteration velocity, not mMotionVelocity
        const AbSpatialVector posVel = au1LoadSV(linkBlocks, PXG_AB_LINK_mPosMotionVelocity, lane);
        const AsTransform preTrans = asLoadTransform(linkBlocks, PXG_AB_LINK_mAccumulatedPose, lane);
        const AsTransform trans = asUpdateRootBody(posVel, preTrans, dt);
        abStoreSpatialTransform(linkBlocks, PXG_AB_LINK_mAccumulatedPose, lane, trans.q, trans.p);
    }
    // trap T5: isSubstep == false; strictly sequential (each link reads its parent's
    // pose, which the earlier iteration wrote), so the store is inside the loop
    for (uint linkID = 1u; linkID < linkCount; ++linkID)
    {
        const AsTransform body2World = asPropagateLink(linkBlocks, dofBlocks, linkID, dt, false, lane);
        abStoreSpatialTransform(linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE, PXG_AB_LINK_mAccumulatedPose,
                                lane, body2World.q, body2World.p);
    }
}

// ---------------------------------------------------------------------------
// conserveMomentum<doIntegrate> (forwardDynamic2.cu:2783-2981)
// ---------------------------------------------------------------------------
void au1ConserveMomentum(uint64_t artiBlock, uint64_t linkBlocks, uint64_t dofBlocks, uint linkCount,
                         bool fixBase, uint lane, float dt, bool doIntegrate)
{
    if (fixBase)
        return; // :2790 -- the whole function is inside `if (!fixBase)`

    const float rz = asiRuntimeZero(dt); // trap T8
    const float3 zero3 = float3(rz, rz, rz);

    const float4 COM_invMassW = abF4(artiBlock, PXG_AB_mCOM_TotalInvMassW, lane);
    const float3 preCOM = COM_invMassW.xyz;
    float3 postCOM = zero3;
    float3 preMomentumTop = zero3, preMomentumBottom = zero3;
    float3 postMomentumTop = zero3, postMomentumBottom = zero3;
    float3 posMomentum = zero3;

    // Loop A (:2800-2805): linear momentum from the OLD velocities, and the
    // position-iteration momentum
    for (uint linkID = 0u; linkID < linkCount; ++linkID)
    {
        const uint64_t link = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;
        const float mass = abF32(link, PXG_AB_LINK_mMass, lane);
        preMomentumTop = preMomentumTop + au1LoadSV(link, PXG_AB_LINK_mMotionVelocity, lane).bottom * mass;
        posMomentum = posMomentum + au1LoadSV(link, PXG_AB_LINK_mPosMotionVelocity, lane).bottom * mass;
    }

    // the root (:2808-2818): pre-momentum about the PRE-step pose (trap T11),
    // assignments, not accumulations
    float mass = abF32(linkBlocks, PXG_AB_LINK_mMass, lane);
    const float3 comPreLinVel = preMomentumTop * COM_invMassW.w;
    AbMat33 inertia = abLoadPxMat33(linkBlocks, PXG_AB_LINK_mIsolatedInertia, lane);
    AbSpatialVector vel = au1LoadSV(linkBlocks, PXG_AB_LINK_mMotionVelocity, lane);
    preMomentumBottom = asmM33Transform(inertia, vel.top)
                      + asmCross(abF4(linkBlocks, PXG_AB_LINK_mPreTransform + PXG_AB_STB_P, lane).xyz - preCOM,
                                 vel.bottom - comPreLinVel) * mass;
    postMomentumTop = vel.bottom * mass;
    postCOM = postCOM + abF4(linkBlocks, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_P, lane).xyz * mass;

    // trap T9: the .cu prefetches artiLinks[1].mParents unconditionally
    uint nextParent = 0u;
    if (linkCount > 1u)
        nextParent = abU32(linkBlocks + PXG_AB_LINK_SIZE, PXG_AB_LINK_mParents, lane);

    // Loop B (:2823-2866): pre-momentum of the links, then the FK rewrite of every
    // non-root velocity from the (already rewritten) parent's
    for (uint linkID = 1u; linkID < linkCount; ++linkID)
    {
        const uint64_t link = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;
        const uint parent = nextParent;
        if (linkID != linkCount - 1u)
            nextParent = abU32(linkBlocks + uint64_t(linkID + 1u) * PXG_AB_LINK_SIZE, PXG_AB_LINK_mParents, lane);

        const AsTransform childPose = asLoadTransform(link, PXG_AB_LINK_mAccumulatedPose, lane);
        const uint64_t parentLink = linkBlocks + uint64_t(parent) * PXG_AB_LINK_SIZE;
        const float3 parentPos = abF4(parentLink, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_P, lane).xyz;
        mass = abF32(link, PXG_AB_LINK_mMass, lane);
        const float3 rw = childPose.p - parentPos;

        const uint dofs = abU8(link, PXG_AB_LINK_mDofs, lane);
        const uint jointOffset = abU32(link, PXG_AB_LINK_mJointOffset, lane); // full width here (:2840)

        inertia = abLoadPxMat33(link, PXG_AB_LINK_mIsolatedInertia, lane);
        vel = au1LoadSV(link, PXG_AB_LINK_mMotionVelocity, lane);
        // `+=` of a two-term sum: bottom + (A + B), not (bottom + A) + B (trap T10)
        preMomentumBottom = preMomentumBottom
                          + (asmM33Transform(inertia, vel.top)
                             + asmCross(abF4(link, PXG_AB_LINK_mPreTransform + PXG_AB_STB_P, lane).xyz - preCOM,
                                        vel.bottom - comPreLinVel) * mass);

        // translateSpatialVector(-rw, parentVel) = (pv.top, pv.bottom + (-rw) x pv.top)
        // (DyFeatherstoneArticulation.h:828-831); the parent's velocity is the one an
        // earlier iteration rewrote, or the root's flushed one
        const AbSpatialVector pv = au1LoadSV(parentLink, PXG_AB_LINK_mMotionVelocity, lane);
        vel = asmSVec(pv.top, pv.bottom + asmCross(-rw, pv.top));

        for (uint ind = 0u; ind < 3u; ++ind)
        {
            if (ind < dofs)
            {
                const uint64_t dof = dofBlocks + uint64_t(jointOffset + ind) * PXG_AB_DOF_SIZE;
                const float jVel = abF32(dof, PXG_AB_DOF_mJointVelocities, lane);
                const AbSpatialVector mm = au1LoadSV(dof, PXG_AB_DOF_mLocalMotionMatrix, lane);
                // UnAlignedSpatialVector::rotate(PxTransform) = (q.rotate(top), q.rotate(bottom))
                // (CmSpatialVector.h:431), then * jVel, then +=
                const AbSpatialVector r = asmSVec(quatRotate(childPose.q, mm.top), quatRotate(childPose.q, mm.bottom));
                vel = asmSVecAdd(vel, asmSVecScale(r, jVel));
            }
        }

        postCOM = postCOM + childPose.p * mass;
        abStoreSpatialVector(link, PXG_AB_LINK_mMotionVelocity, lane, vel.top, vel.bottom);
        postMomentumTop = postMomentumTop + vel.bottom * mass;
    }

    postCOM = postCOM * COM_invMassW.w; // PxVec3::operator*=
    const float3 comPostLinVel = postMomentumTop * COM_invMassW.w;

    // PxMat33(PxZero) grown by `+=`: a runtime-zero matrix (trap T8)
    AbMat33 compoundInertia = asmM33(zero3, zero3, zero3);

    float4 nextInvInertiaDiag = abF4(linkBlocks, PXG_AB_LINK_mInvInertiaXYZ_invMassW, lane);
    float nextMass = abF32(linkBlocks, PXG_AB_LINK_mMass, lane);
    AsTransform nextPose = asLoadTransform(linkBlocks, PXG_AB_LINK_mAccumulatedPose, lane);

    // Loop C (:2877-2904): compound inertia about the POST COM from
    // transformInertiaTensor(1 / invInertia, R(pose.q)), and the post angular momentum
    for (uint linkID = 0u; linkID < linkCount; ++linkID)
    {
        const uint64_t link = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;
        const AbSpatialVector lvel = au1LoadSV(link, PXG_AB_LINK_mMotionVelocity, lane);

        // trap T12: three unguarded true divides
        const float3 inertiaDiag = float3(1.0f / nextInvInertiaDiag.x, 1.0f / nextInvInertiaDiag.y,
                                          1.0f / nextInvInertiaDiag.z);
        const float lmass = nextMass;
        const AsTransform pose = nextPose;
        if (linkID != linkCount - 1u)
        {
            const uint64_t next = linkBlocks + uint64_t(linkID + 1u) * PXG_AB_LINK_SIZE;
            nextInvInertiaDiag = abF4(next, PXG_AB_LINK_mInvInertiaXYZ_invMassW, lane);
            nextMass = abF32(next, PXG_AB_LINK_mMass, lane);
            nextPose = asLoadTransform(next, PXG_AB_LINK_mAccumulatedPose, lane);
        }

        const float3 offset = pose.p - postCOM;
        const AbMat33 R = asiMat33FromQuat(pose.q);
        // (offsetMass at :2894 is dead and not transcribed)
        const AbMat33 linkInertia = asiTransformInertiaTensor(inertiaDiag, R);
        compoundInertia = asmM33Add(compoundInertia, asmTranslateInertiaMass(linkInertia, lmass, offset));
        postMomentumBottom = postMomentumBottom
                           + (asmM33Transform(linkInertia, lvel.top)
                              + asmCross(offset, lvel.bottom - comPostLinVel) * lmass);
    }

    const AbMat33 invCompoundInertia = asmM33Inverse(compoundInertia);

    // trap T13(a): the guard returns ZERO here
    const float aDenom = au1Magnitude(postMomentumBottom);
    const float angRatio = aDenom == 0.0f ? 0.0f : au1Magnitude(preMomentumBottom) / aDenom;
    const float3 angMomDelta = postMomentumBottom * (angRatio - 1.0f);
    const float3 angDelta = asmM33Transform(invCompoundInertia, angMomDelta);

    // Loop D (:2914-2923): trap T13(b), velChange.bottom = MINUS offset x angDelta
    for (uint linkID = 0u; linkID < linkCount; ++linkID)
    {
        const uint64_t link = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;
        const AbSpatialVector lvel = au1LoadSV(link, PXG_AB_LINK_mMotionVelocity, lane);
        const float lmass = abF32(link, PXG_AB_LINK_mMass, lane);
        const float3 offset = abF4(link, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_P, lane).xyz - postCOM;
        const AbSpatialVector velChange = asmSVec(angDelta, -asmCross(offset, angDelta));
        const AbSpatialVector nv = asmSVecAdd(lvel, velChange);
        abStoreSpatialVector(link, PXG_AB_LINK_mMotionVelocity, lane, nv.top, nv.bottom);
        postMomentumTop = postMomentumTop + velChange.bottom * lmass;
    }

    const float3 deltaBottom = (preMomentumTop - postMomentumTop) * COM_invMassW.w;

    // (posMomentum * w) * dt when integrating; TGS's mPosMotionVelocity is already a
    // displacement over the step
    const float3 predictedCOM = doIntegrate ? preCOM + (posMomentum * COM_invMassW.w) * dt
                                            : preCOM + posMomentum * COM_invMassW.w;
    const float3 correction = predictedCOM - postCOM;

    // Loop E (:2940-2944): addSpatialVector adds (0, 0, 0, dx) to the float4 and
    // (dy, dz) to the float2 (PxgArticulation.h:214-218); the pose's p.w gets `+ 0.f`.
    // All three zeros are runtime zeros (trap T8).
    for (uint linkID = 0u; linkID < linkCount; ++linkID)
    {
        const uint64_t link = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;
        float4 t = abF4(link, PXG_AB_LINK_mMotionVelocity + PXG_AB_SVB_TOPXYZ_BX, lane);
        t = float4(au1PlusZero(t.x), au1PlusZero(t.y), au1PlusZero(t.z), t.w + deltaBottom.x);
        abStoreF4(link, PXG_AB_LINK_mMotionVelocity + PXG_AB_SVB_TOPXYZ_BX, lane, t);
        float2 b = abF2(link, PXG_AB_LINK_mMotionVelocity + PXG_AB_SVB_BYZ, lane);
        b = b + float2(deltaBottom.y, deltaBottom.z);
        abStoreF2(link, PXG_AB_LINK_mMotionVelocity + PXG_AB_SVB_BYZ, lane, b);

        float4 p = abF4(link, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_P, lane);
        p = float4(p.x + correction.x, p.y + correction.y, p.z + correction.z, au1PlusZero(p.w));
        abStoreF4(link, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_P, lane, p);
    }
    // (postCOM += correction at :2946 feeds only the `#if 0` validation block)
}

// ---------------------------------------------------------------------------
// the kernel (forwardDynamic2.cu:3047-3075 -> updateBodiesInternal :2984-3045)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(32, AU1_WARPS_PER_BLOCK, 1)]
void updateBodiesLaunch1T(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    const uint64_t d = gAu1.desc;
    const float dt = gAu1.dt;
    const bool integrate = (gAu1.integrateWord & 0xFFu) != 0u;

    const uint nbArticulations = artiU32Field(d, PXG_ARTI_CORE_NB_ARTICULATIONS);
    const uint warpIndex = groupThreadId.y;
    const uint globalWarpIndex = groupId.x * AU1_WARPS_PER_BLOCK + warpIndex;
    const uint lane = groupThreadId.x & 31u;
    const uint globalThreadIndex = globalWarpIndex * 32u + groupThreadId.x;

    if (globalThreadIndex >= nbArticulations)
        return;

    const uint maxLinks = artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS);
    const uint maxDofs = artiU32Field(d, PXG_ARTI_CORE_MAX_DOFS);

    // trap T14: the three block arrays by globalWarpIndex
    const uint64_t artiBlock = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS)
                             + uint64_t(globalWarpIndex) * PXG_AB_BLOCK_SIZE;
    const uint64_t linkBlocks = artiPtrField(d, PXG_ARTI_CORE_LINK_BLOCKS)
                              + uint64_t(globalWarpIndex) * uint64_t(maxLinks) * PXG_AB_LINK_SIZE;
    const uint64_t dofBlocks = artiPtrField(d, PXG_ARTI_CORE_DOF_BLOCKS)
                             + uint64_t(globalWarpIndex) * uint64_t(maxDofs) * PXG_AB_DOF_SIZE;

    const uint linkCount = abU32(artiBlock, PXG_AB_mNumLinks, lane);
    const bool fixBase = (abU32(artiBlock, PXG_AB_mFlags, lane) & PXG_ARTI_FLAG_FIX_BASE) != 0u;
    const float invDt = 1.0f / dt;

    // 1. poses / joint positions (trap T2: two different paths, not one with a flag)
    if (!integrate)
        au1ComputeAndEnforceJointPositions(linkBlocks, dofBlocks, linkCount, lane);
    else
        au1UpdatePoses(linkBlocks, dofBlocks, linkCount, fixBase, dt, lane);

    // 2. the deferred impulse flush -- npArtiStepKernel.slang:144-157 verbatim
    const uint stateDirty = abU8(artiBlock, PXG_AB_mStateDirty, lane);
    if (stateDirty != 0u)
    {
        if ((stateDirty & PXG_ARTI_STATE_HAS_IMPULSES) != 0u)
        {
            // the THREAD index is the articulation id (trap T14)
            aiAverageAndPropagate(artiPtrField(d, PXG_ARTI_CORE_SLAB_HAS_CHANGES),
                                  artiPtrField(d, PXG_ARTI_CORE_IMPULSES), artiBlock, linkBlocks,
                                  dofBlocks, globalThreadIndex, maxLinks, nbArticulations,
                                  artiU32Field(d, PXG_ARTI_CORE_NB_SLABS), lane);
        }
        fvFlushVelocity(artiBlock, linkBlocks, dofBlocks, linkCount, fixBase, lane); // trap T1
        abStoreU8(artiBlock, PXG_AB_mStateDirty, lane, 0u);                         // trap T4
    }

    // 3.
    au1ConserveMomentum(artiBlock, linkBlocks, dofBlocks, linkCount, fixBase, lane, dt, integrate);

    // 4. the dof writeback into the record selected by the BLOCK's mArticulationIndex
    const uint totalDofs = abU32(artiBlock, PXG_AB_mTotalDofs, lane);
    const uint articulationIndex = abU32(artiBlock, PXG_AB_mArticulationIndex, lane);
    const uint64_t a = artiPtrField(d, PXG_ARTI_CORE_ARTICULATIONS)
                     + uint64_t(articulationIndex) * PXG_ARTI_STRIDE;
    Ptr<float> jointP = Ptr<float>(artiPtrField(a, PXG_ARTI_JOINT_POSITIONS));
    Ptr<float> jointV = Ptr<float>(artiPtrField(a, PXG_ARTI_JOINT_VELOCITIES));
    Ptr<float> jointA = Ptr<float>(artiPtrField(a, PXG_ARTI_JOINT_ACCELERATIONS));
    for (uint i = 0u; i < totalDofs; ++i)
    {
        const uint64_t dof = dofBlocks + uint64_t(i) * PXG_AB_DOF_SIZE; // the FLAT dof index
        const float v = abF32(dof, PXG_AB_DOF_mJointVelocities, lane);
        jointP[i] = abF32(dof, PXG_AB_DOF_mJointPositions, lane);
        jointV[i] = v;
        jointA[i] = abF32(dof, PXG_AB_DOF_mJointAccel, lane)
                  + (v - abF32(dof, PXG_AB_DOF_mJointUnconstrainedVelocities, lane)) * invDt;
    }
}
