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

// npArtiStep.slang
//
// P8.94a: the POSITION INTEGRATION kernel's algebra — `updateRootBody`,
// `propagateLink<isSubstep>` and `computeSphericalJointPositions`
// (forwardDynamic2.cu:2514 / :2534 / :2410), plus the two quaternion helpers they
// need.  `stepArticulation1TTGS` itself is P8.94d; `PxcFsFlushVelocity` and
// `averageLinkImpulsesAndPropagate` are P8.94b and P8.94c.
//
// WHY THIS CHAIN.  P8.93 established that the articulation scene's link now
// reports a REAL pose (3.000, unchanged) and a REAL non-zero velocity (4.78).
// What stands between the velocity and a moving link is exactly two kernels:
// this one, which integrates the velocity into `mAccumulatedPose`, and
// `updateBodiesLaunch_Part2`, which copies that pose into the record's
// `linkBody2Worlds` (forwardDynamic2.cu:3189) — the buffer P8.93's DMA-back
// hands to the host.  Neither is mapped.
//
// Zero new ABI: `mPreTransform` (22528), `mDeltaQ` (26624), `mDeltaMotion`
// (11008), `mPosMotionVelocity` (10240), `mChildPose` (24576), `mParentPose`
// (25600), `mRelativeQuat` (21504), `mJointType` (28832),
// `mLocalMotionMatrix` (1920), `mJointPositions` (0) and
// `mPosJointVelocities` (384) were all mapped in P8.67/P8.68.
//
// ===========================================================================
// THE ORACLE
// ===========================================================================
//
// `PxExp` (PxMathUtils.h:143), the `PxQuat(angle, axis)` constructor,
// `PxQuat::rotate` / `getConjugate` / `getNormalized`, `PxVec3::normalize` and
// `PxTransform` are all PX_CUDA_CALLABLE, so the fixture computes the whole
// expectation with the engine's own arithmetic and compares bit patterns.
// `toRotationVector`, `computeSphericalJointPositions` and `propagateLink` are
// `static __device__`, so only their STRUCTURE is transcribed.
//
// ===========================================================================
// THE TRAPS
// ===========================================================================
//
//  1. **`isSubstep` chooses the velocity.**  A substep integrates
//     `mJointVelocities` and `mMotionVelocity`; a full step integrates
//     `mPosJointVelocities` and `mPosMotionVelocity`.  Four separate reads,
//     and getting one of them wrong is a plausible-looking wrong answer.
//  2. **`childOffset` is NEGATED at the load, `parentOffset` is not.**
//     `PxVec3 childOffset(-co.x, -co.y, -co.z)` against
//     `PxVec3 parentOffset(po.x, po.y, po.z)`.
//  3. **The three one-dof joint types differ in their position handling.**
//     Prismatic writes the position and builds an OFFSET, leaving the rotation
//     at `relativeQuat`.  Revolute wraps into `[-4pi, 4pi]` and then CLAMPS to
//     the same range — two separate operations, and the wrap is `-= 4pi` /
//     `+= 4pi`, not a modulo.  Revolute-unwrapped does neither.
//  4. **The revolute rotation angle is NEGATED**: `PxQuat(-jPos, axis)`, then
//     flipped to the shortest arc if `w < 0`.
//  5. **The spherical path integrates in the WORLD frame** and recovers the
//     joint positions from the resulting quaternion, rather than integrating the
//     joint positions directly — the CUDA's own commented-out three lines show
//     the alternative it does NOT take.
//  6. **`PxVec3::normalize` returns the magnitude and divides only when it is
//     positive**, and the division is componentwise `/=`, not a multiply by the
//     reciprocal.  `dist` is that magnitude times `dt`.
//  7. **`r = e + d + offset`** where `e` is the ROTATED parent offset and `d`
//     the raw child offset; the body's position is the parent's plus the
//     new orientation applied to `r`.
//  8. **`body2World.q = (pBody2World.q * newParentToChild.getConjugate())`** —
//     the CONJUGATE, and the parent's rotation on the left.
//
// And one the port drops with a note rather than transcribing:
// `computeSphericalJointPositions` has an `if (0)` branch using
// `getImaginaryPart` and `compAng`.  It is dead code in the engine and it is
// absent here.
//
// ===========================================================================
// WHY THERE ARE THREE PROBES AND NOT ONE FIXTURE
// ===========================================================================
//
// This is the FIRST articulation tranche whose oracle is NOT bit-exact, and the
// reason is measured rather than assumed.  `artiStepQuatProbe` isolates `PxExp`
// and the `PxQuat(angle, axis)` constructor and finds llvmpipe's `sin`/`cos`
// within **2 ULP** of glibc's but not equal to it; `artiStepRotVecProbe` does
// the same for `atan2` and finds **2.64e-6** absolute worst case on a
// near-identity quaternion, where a few ULP inside the `atan2` becomes a large
// RELATIVE error on a correspondingly tiny rotation vector.  So
// `artiStepAlgebra` tolerates a bounded drift on `mAccumulatedPose` and
// `mJointPositions` -- and ONLY there.  Every other word of the link and dof
// blocks, poison included, is compared bit for bit and matches exactly, which is
// what still pins the joint-type dispatch, the wrap, the clamp, the offsets and
// the `isSubstep` field selection.
//
// The two guards that decide branches rather than values do not survive that
// tolerance, so each got its own exact probe instead of a looser bound:
//
//  * `toRotationVector`'s tiny-axis branch is unreachable from the block fixture
//    (it needs a spherical joint whose relative quaternion exactly cancels its
//    parent's computed pose, and that pose is itself an output of the mirror);
//  * the spherical small-angle guard IS reachable -- the fixture puts a
//    sub-threshold angular velocity on a spherical link -- but the rotation it
//    skips is at most 1e-6 rad, i.e. ~4 ULP on a unit quaternion, which is
//    inside the noise floor the transcendentals force.  `artiStepSphGuardProbe`
//    checks the guard's EXACT claim instead: below the threshold the output is
//    the input, bit for bit.
//
// Sweep **41/42**.  The one survivor, S29, is proven equivalent at
// `asNormalizeInPlace` below.

import pxgSlangCommon;
import pxgSlangArticulationDesc;
// PxArticulationJointType, shared with npArtiJointFrame's AJ_* by value.  A
// mismatch cannot be silent: every joint of that type would take the wrong branch.
static const uint AS_J_FIX = 0u;
static const uint AS_J_PRISMATIC = 1u;
static const uint AS_J_REVOLUTE = 2u;
static const uint AS_J_REVOLUTE_UNWRAPPED = 3u;
static const uint AS_J_SPHERICAL = 4u;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;

public struct AsTransform
{
    public float4 q;
    public float3 p;
};

public AsTransform asTransform(float4 q, float3 p) { AsTransform t; t.q = q; t.p = p; return t; }

public AsTransform asLoadTransform(uint64_t base, uint fieldOffset, uint lane)
{
    return asTransform(abF4(base, fieldOffset + PXG_AB_STB_Q, lane),
                       abF4(base, fieldOffset + PXG_AB_STB_P, lane).xyz);
}

// PxQuat(angleRadians, unitAxis) (PxQuat.h:86): the half angle, sin on the axis
// and cos in w.
public float4 asQuatAngleAxis(float angleRadians, float3 unitAxis)
{
    const float a = angleRadians * 0.5f;
    const float s = sin(a);
    return float4(unitAxis.x * s, unitAxis.y * s, unitAxis.z * s, cos(a));
}

public float4 asQuatConjugate(float4 q) { return float4(-q.x, -q.y, -q.z, q.w); }

public float4 asQuatNormalize(float4 q)
{
    // PxQuat::getNormalized: `s = 1 / magnitude(); return q * s`, a MULTIPLY by
    // the reciprocal -- not a componentwise divide.
    const float s = 1.0f / sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return q * s;
}

// PxExp (PxMathUtils.h:143).  `PxRecipSqrt` is `1.0f / sqrtf(a)` on this
// platform (PxUnixMathIntrinsics.h:93), so the axis is `v / sqrt(m)` computed as
// a multiply by that reciprocal -- which is NOT the same last bit as dividing.
public float4 asQuatExp(float3 v)
{
    const float m = v.x * v.x + v.y * v.y + v.z * v.z;
    if (m < 1e-24f)
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    return asQuatAngleAxis(sqrt(m), v * (1.0f / sqrt(m)));
}

// toRotationVector (forwardDynamic2.cu:2388).
public float3 asToRotationVector(float4 q)
{
    const float quatEpsilon = 1.0e-8f;
    const float s2 = q.x * q.x + q.y * q.y + q.z * q.z;
    float angle;
    float3 axis;
    if (s2 < quatEpsilon * quatEpsilon)
    {
        // cannot extract a sensible axis
        angle = 0.0f;
        axis = float3(1.0f, 0.0f, 0.0f);
    }
    else
    {
        const float s = 1.0f / sqrt(s2);
        axis = float3(q.x, q.y, q.z) * s;
        angle = abs(q.w) < quatEpsilon ? 3.14159265358979323846f : atan2(s2 * s, q.w) * 2.0f;
    }
    return axis * angle;
}

// updateRootBody (forwardDynamic2.cu:2514).  The linear half integrates the
// position, the ANGULAR half becomes a quaternion through PxExp -- and the
// product is `deltaQ * preTransform.q`, delta on the LEFT.
public AsTransform asUpdateRootBody(AbSpatialVector motionVelocity, AsTransform preTransform,
                                    float dt)
{
    const float3 comLinVel = motionVelocity.bottom;
    const float3 newP = preTransform.p + comLinVel * dt;
    const float4 deltaQ = asQuatExp(motionVelocity.top * dt);
    return asTransform(asQuatNormalize(quatMul(deltaQ, preTransform.q)), newP);
}

// computeSphericalJointPositions (forwardDynamic2.cu:2409).  The engine's
// `if (0)` branch is dead and is not transcribed.
public float4 asComputeSphericalJointPositions(float4 relativeQuat, float4 newRot,
                                               float4 pBody2WorldRot, uint64_t dofData, uint dofs,
                                               uint lane)
{
    float4 newParentToChild = asQuatNormalize(quatMul(asQuatConjugate(newRot), pBody2WorldRot));

    if (newParentToChild.w < 0.0f)
        newParentToChild = -newParentToChild;

    float4 jointRotation = quatMul(newParentToChild, asQuatConjugate(relativeQuat));
    if (jointRotation.w < 0.0f)
        jointRotation = -jointRotation;

    const float3 axis = asToRotationVector(jointRotation);
    for (uint dd = 0u; dd < dofs; ++dd)
    {
        const uint64_t d = dofData + uint64_t(dd) * PXG_AB_DOF_SIZE;
        const float4 top = abF4(d, PXG_AB_DOF_mLocalMotionMatrix + PXG_AB_SVB_TOPXYZ_BX, lane);
        const float ang = -asmDot(float3(top.x, top.y, top.z), axis);
        abStoreF32(d, PXG_AB_DOF_mJointPositions, lane, ang);
    }

    return newParentToChild;
}

// PxVec3::normalize (PxVec3.h:295): returns the magnitude and divides IN PLACE
// only when it is positive -- and by a componentwise `/=`, not by a reciprocal.
//
// SWEEP SURVIVOR S29 ("normalize drops its positivity guard"), and it is a
// PROVEN equivalence rather than a fixture weakness.  The function has exactly
// one caller, `asIntegrateWorldQ` below, which uses the two results as
//
//     dist = asNormalizeInPlace(worldAngVel) * dt;
//     if (dist > 1e-6f)  ... quatMul(asQuatAngleAxis(dist, worldAngVel), oldQ)
//
// `m` is a `sqrt`, so it is never negative and the guard excludes exactly two
// inputs: `m == 0` and `m == NaN`.  In both, `dist` is `0 * dt == 0` or
// `NaN * dt == NaN`, and `0 > 1e-6f` and `NaN > 1e-6f` are BOTH false -- so the
// NaN the mutant writes into `worldAngVel` is never read.  The guard's only
// observable effect is on a value that is dead exactly when the guard fires.
//
// The fixture reaches the `m == 0` case anyway (`stZeroAngVel`, counted back as
// `sphZero`), which is what makes the proof checkable rather than merely
// argued: were the reasoning wrong, the mutant would die there.
float asNormalizeInPlace(inout float3 v)
{
    const float m = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (m > 0.0f)
    {
        // P9.45: ONE reciprocal and three multiplies, not three divides.  This mirrors
        // `worldAngVel.normalize()` (forwardDynamic2.cu:2630), and PxVec3::normalize
        // (PxVec3.h:293) scales through operator/= (PxVec3.h:260), which reciprocates once.
        // The P9.36 defect class in its componentwise spelling: `gpuslang.vectorDivide`
        // matches an OpFDiv with a VECTOR result type, and three scalar OpFDivs slip past
        // it.  Found by the same sweep that fixed computeBasisVectors.
        const float r = 1.0f / m;
        v = float3(v.x * r, v.y * r, v.z * r);
    }
    return m;
}

// The spherical branch's world-rotation integration, factored out of
// `asPropagateLink` so `artiStepSphGuardProbe` drives the SAME code the kernel
// does rather than a second transcription of it.
//
// The small-angle guard (trap 6) cannot be killed through `asPropagateLink`'s
// own output on this platform: below the threshold the rotation it skips is
// `dist <= 1e-6` radians, which perturbs a unit quaternion by ~4 ULP -- well
// inside the 384-ULP budget llvmpipe's `sin`/`cos` force on a chained pose (see
// `artiStepQuatProbe`).  The guard is not, however, an approximation: below the
// threshold the engine returns the OLD quaternion untouched, which is a
// bit-exact statement with no transcendental in it at all.  That is what the
// probe checks, and it is exact.
public float4 asIntegrateWorldQ(float4 oldQ, float3 angVel, float dt)
{
    float3 worldAngVel = angVel;
    const float dist = asNormalizeInPlace(worldAngVel) * dt;   // trap 6

    float4 newWorldQ = oldQ;
    if (dist > 1e-6f)
        newWorldQ = quatMul(asQuatAngleAxis(dist, worldAngVel), oldQ);
    return newWorldQ;
}

static const float AS_TWO_PI = 6.28318530717958647692f;

// propagateLink<isSubstep> (forwardDynamic2.cu:2534).
public AsTransform asPropagateLink(uint64_t linkBlocks, uint64_t dofBlocks, uint linkID, float dt,
                                   bool isSubstep, uint lane)
{
    const uint64_t link = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;
    const uint jointOffset = abU32(link, PXG_AB_LINK_mJointOffset, lane);
    const uint jointType = abU8(link, PXG_AB_LINK_mJointType, lane);
    const uint parent = abU32(link, PXG_AB_LINK_mParents, lane);
    const uint64_t pLink = linkBlocks + uint64_t(parent) * PXG_AB_LINK_SIZE;
    const uint64_t dof = dofBlocks + uint64_t(jointOffset) * PXG_AB_DOF_SIZE;

    const AsTransform pBody2World = asLoadTransform(pLink, PXG_AB_LINK_mAccumulatedPose, lane);

    const float4 co = abF4(link, PXG_AB_LINK_mChildPose + PXG_AB_STB_P, lane);
    const float4 po = abF4(link, PXG_AB_LINK_mParentPose + PXG_AB_STB_P, lane);
    const float4 relativeQuat = abF4(link, PXG_AB_LINK_mRelativeQuat, lane);

    // the child offset is NEGATED, the parent's is not -- trap 2
    const float3 childOffset = float3(-co.x, -co.y, -co.z);
    const float3 parentOffset = float3(po.x, po.y, po.z);

    float4 newParentToChild = relativeQuat;
    float3 offset = float3(0.0f, 0.0f, 0.0f);

    if (jointType == AS_J_PRISMATIC)
    {
        const float delta = abF32(dof, isSubstep ? PXG_AB_DOF_mJointVelocities
                                                 : PXG_AB_DOF_mPosJointVelocities, lane) * dt;
        const float pos = abF32(dof, PXG_AB_DOF_mJointPositions, lane) + delta;
        abStoreF32(dof, PXG_AB_DOF_mJointPositions, lane, pos);

        newParentToChild = relativeQuat;
        float3 mmTop, mmBottom;
        abLoadSpatialVector(dof, PXG_AB_DOF_mLocalMotionMatrix, lane, mmTop, mmBottom);
        offset = mmBottom * pos;
    }
    else if (jointType == AS_J_REVOLUTE)
    {
        const float delta = abF32(dof, isSubstep ? PXG_AB_DOF_mJointVelocities
                                                 : PXG_AB_DOF_mPosJointVelocities, lane) * dt;
        float jPos = abF32(dof, PXG_AB_DOF_mJointPositions, lane) + delta;

        // a WRAP by 4pi, then a CLAMP to the same range -- two operations, not one
        if (jPos > AS_TWO_PI)
            jPos -= AS_TWO_PI * 2.0f;
        else if (jPos < -AS_TWO_PI)
            jPos += AS_TWO_PI * 2.0f;

        jPos = clamp(jPos, -AS_TWO_PI * 2.0f, AS_TWO_PI * 2.0f);
        abStoreF32(dof, PXG_AB_DOF_mJointPositions, lane, jPos);

        const float4 u = abF4(dof, PXG_AB_DOF_mLocalMotionMatrix + PXG_AB_SVB_TOPXYZ_BX, lane);
        // the angle is NEGATED -- trap 4
        float4 jointRotation = asQuatAngleAxis(-jPos, float3(u.x, u.y, u.z));
        if (jointRotation.w < 0.0f)
            jointRotation = -jointRotation;

        newParentToChild = asQuatNormalize(quatMul(jointRotation, relativeQuat));
    }
    else if (jointType == AS_J_REVOLUTE_UNWRAPPED)
    {
        const float delta = abF32(dof, isSubstep ? PXG_AB_DOF_mJointVelocities
                                                 : PXG_AB_DOF_mPosJointVelocities, lane) * dt;
        // NO wrap and NO clamp -- that is what "unwrapped" means
        const float jPos = abF32(dof, PXG_AB_DOF_mJointPositions, lane) + delta;
        abStoreF32(dof, PXG_AB_DOF_mJointPositions, lane, jPos);

        const float4 u = abF4(dof, PXG_AB_DOF_mLocalMotionMatrix + PXG_AB_SVB_TOPXYZ_BX, lane);
        float4 jointRotation = asQuatAngleAxis(-jPos, float3(u.x, u.y, u.z));
        if (jointRotation.w < 0.0f)
            jointRotation = -jointRotation;

        newParentToChild = asQuatNormalize(quatMul(jointRotation, relativeQuat));
    }
    else if (jointType == AS_J_SPHERICAL)
    {
        const uint nbDofs = abU8(link, PXG_AB_LINK_mDofs, lane);
        const AsTransform oldTransform = asLoadTransform(link, PXG_AB_LINK_mAccumulatedPose, lane);

        float3 wvTop, wvBottom;
        abLoadSpatialVector(link, isSubstep ? PXG_AB_LINK_mMotionVelocity
                                            : PXG_AB_LINK_mPosMotionVelocity,
                            lane, wvTop, wvBottom);

        const float4 newWorldQ = asIntegrateWorldQ(oldTransform.q, wvTop, dt);

        newParentToChild = asComputeSphericalJointPositions(relativeQuat, newWorldQ,
                                                           pBody2World.q, dof, nbDofs, lane);
    }
    else
    {
        // eFIX (and eUNDEFINED): no velocity, so nothing to integrate
        newParentToChild = relativeQuat;
    }

    const float3 e = quatRotate(newParentToChild, parentOffset);
    const float3 d = childOffset;
    const float3 r = e + d + offset;   // trap 7

    AsTransform body2World;
    // the parent's rotation on the LEFT, the CONJUGATE on the right -- trap 8
    body2World.q = asQuatNormalize(quatMul(pBody2World.q, asQuatConjugate(newParentToChild)));
    body2World.p = pBody2World.p + quatRotate(body2World.q, r);
    return body2World;
}

// ---------------------------------------------------------------------------
// The fixture entry point
// ---------------------------------------------------------------------------
//
// The root-then-links shape `stepArticulation1TTGS` uses, minus the deltaQ /
// deltaMotion bookkeeping (that is P8.94d).  It is written this way rather than
// as a table of isolated calls because the link loop's PARENT DEPENDENCY is the
// thing worth testing: link i reads the parent's `mAccumulatedPose` after an
// earlier iteration has written it.

struct AsStepParams
{
    uint64_t linkBlocks;   // @0
    uint64_t dofBlocks;    // @8
    uint64_t numLinksPer;  // @16  uint32[nbLanes]
    uint nbLanes;          // @24
    uint isSubstep;        // @28
    float dt;              // @32
    uint pad0;             // @36
    uint pad1;             // @40
    uint pad2;             // @44 -> 48
};

[[vk::push_constant]]
ConstantBuffer<AsStepParams> gAsStep;

[shader("compute")]
[numthreads(32, 1, 1)]
void artiStepAlgebra(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gAsStep.nbLanes)
        return;

    const uint numLinks = (Ptr<uint>(gAsStep.numLinksPer))[lane];
    const bool isSubstep = gAsStep.isSubstep != 0u;
    const float dt = gAsStep.dt;
    const uint64_t linkBlocks = gAsStep.linkBlocks;

    // the root: integrated from its own motion velocity and its CURRENT pose
    float3 mvTop, mvBottom;
    abLoadSpatialVector(linkBlocks, PXG_AB_LINK_mMotionVelocity, lane, mvTop, mvBottom);
    const AsTransform root =
        asUpdateRootBody(asmSVec(mvTop, mvBottom),
                         asLoadTransform(linkBlocks, PXG_AB_LINK_mAccumulatedPose, lane), dt);
    abStoreSpatialTransform(linkBlocks, PXG_AB_LINK_mAccumulatedPose, lane, root.q, root.p);

    // the links, root to leaf: each reads the parent's pose the previous
    // iteration wrote
    for (uint linkID = 1u; linkID < numLinks; ++linkID)
    {
        const AsTransform b =
            asPropagateLink(linkBlocks, gAsStep.dofBlocks, linkID, dt, isSubstep, lane);
        abStoreSpatialTransform(linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE,
                                PXG_AB_LINK_mAccumulatedPose, lane, b.q, b.p);
    }
}

// A probe over the two quaternion helpers ALONE, so the question "is the
// difference from a transcendental or from the algebra?" has an answer rather
// than a hunch.  Dumps `asQuatExp(v)` and `asQuatAngleAxis(a, axis)` for a table
// of inputs; the fixture compares them against `PxExp` and the `PxQuat(angle,
// axis)` constructor, which are both PX_CUDA_CALLABLE.
struct AsQuatParams
{
    uint64_t inFloats;    // @0   float[nbCases * 7]  -- 3 exp + 1 angle + 3 axis
    uint64_t outFloats;   // @8   float[nbCases * 8]  -- two quaternions
    uint nbCases;         // @16
    uint pad0;            // @20
    uint pad1;            // @24
    uint pad2;            // @28 -> 32
};

[[vk::push_constant]]
ConstantBuffer<AsQuatParams> gAsQuat;

[shader("compute")]
[numthreads(64, 1, 1)]
void artiStepQuatProbe(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint tid = dispatchThreadId.x;
    if (tid >= gAsQuat.nbCases)
        return;

    Ptr<float> i = Ptr<float>(gAsQuat.inFloats + uint64_t(tid) * 7u * 4u);
    Ptr<float> o = Ptr<float>(gAsQuat.outFloats + uint64_t(tid) * 8u * 4u);

    const float4 e = asQuatExp(float3(i[0], i[1], i[2]));
    o[0] = e.x; o[1] = e.y; o[2] = e.z; o[3] = e.w;

    const float4 a = asQuatAngleAxis(i[3], float3(i[4], i[5], i[6]));
    o[4] = a.x; o[5] = a.y; o[6] = a.z; o[7] = a.w;
}

// The rotation-vector half of the probe.  `toRotationVector`'s tiny-axis guard
// is only reachable from `propagateLink` by constructing a spherical joint whose
// relative quaternion exactly cancels its parent's pose -- which the block
// fixture cannot arrange without first running the mirror.  Probing the function
// directly is the honest way to cover it.
struct AsRotVecParams
{
    uint64_t inFloats;    // @0   float[nbCases * 4]
    uint64_t outFloats;   // @8   float[nbCases * 3]
    uint nbCases;         // @16
    uint pad0;            // @20
    uint pad1;            // @24
    uint pad2;            // @28 -> 32
};

[[vk::push_constant]]
ConstantBuffer<AsRotVecParams> gAsRotVec;

[shader("compute")]
[numthreads(64, 1, 1)]
void artiStepRotVecProbe(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint tid = dispatchThreadId.x;
    if (tid >= gAsRotVec.nbCases)
        return;

    Ptr<float> i = Ptr<float>(gAsRotVec.inFloats + uint64_t(tid) * 4u * 4u);
    Ptr<float> o = Ptr<float>(gAsRotVec.outFloats + uint64_t(tid) * 3u * 4u);

    const float3 r = asToRotationVector(float4(i[0], i[1], i[2], i[3]));
    o[0] = r.x; o[1] = r.y; o[2] = r.z;
}

// The small-angle-guard half of the probe.  See `asIntegrateWorldQ` for why
// `artiStepAlgebra` cannot carry this one: below the threshold the skipped
// rotation is smaller than the transcendental noise the block comparison has to
// tolerate.  Here there is nothing to tolerate -- the claim under test is that
// the output is the INPUT, bit for bit.
struct AsSphGuardParams
{
    uint64_t inFloats;    // @0   float[nbCases * 8]  (q.xyzw, w.xyz, dt)
    uint64_t outFloats;   // @8   float[nbCases * 4]
    uint nbCases;         // @16
    uint pad0;            // @20
    uint pad1;            // @24
    uint pad2;            // @28 -> 32
};

[[vk::push_constant]]
ConstantBuffer<AsSphGuardParams> gAsSphGuard;

[shader("compute")]
[numthreads(64, 1, 1)]
void artiStepSphGuardProbe(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint tid = dispatchThreadId.x;
    if (tid >= gAsSphGuard.nbCases)
        return;

    Ptr<float> i = Ptr<float>(gAsSphGuard.inFloats + uint64_t(tid) * 8u * 4u);
    Ptr<float> o = Ptr<float>(gAsSphGuard.outFloats + uint64_t(tid) * 4u * 4u);

    const float4 r = asIntegrateWorldQ(float4(i[0], i[1], i[2], i[3]),
                                       float3(i[4], i[5], i[6]), i[7]);
    o[0] = r.x; o[1] = r.y; o[2] = r.z; o[3] = r.w;
}
