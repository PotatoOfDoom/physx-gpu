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

// npArtiLinkAccel.slang
//
// P8.92a: the last two pieces of `computeUnconstrainedVelocities`' fifth kernel
// that can stand on their own —
//
//   * `computeJointTransmittedFrictionForce` (forwardDynamic2.cu:2317), the
//     leaf-to-root bias-force sweep the acceleration pass ends with;
//   * `asmTranslateInertiaMass` (forwardDynamic2.cu:1569), the parallel-axis
//     theorem `computeLinkAcceleration`'s compound inertia is built out of,
//     which lives in `pxgSlangSpatialMatrix.slang` beside the rest of the algebra.
//
// `computeLinkAcceleration` itself is P8.92b and the wrapper is P8.92c.  **Zero
// new ABI for the fifth tranche running**: `mBiasForce` (20224),
// `mMotionAcceleration` (18688), `mMotionAccelerationInternal` (19456),
// `mInvInertiaXYZ_invMassW` (27648), `mAccumulatedPose` (23552), `mParents`
// (28672) and `mRw_{x,y,z}` (21120/21248/21376) were all mapped in P8.68.
//
// ===========================================================================
// THE ORACLE
// ===========================================================================
//
// Everything the body is built from is PX_CUDA_CALLABLE — `PxQuat::rotate` and
// `rotateInv`, `PxVec3::multiply`, `FeatherstoneArticulation::
// translateSpatialVector`, `loadSpatialVector` / `storeSpatialVector`,
// `PxMat33::getTranspose` and `operator*` — so the fixture computes the whole
// expectation with the ENGINE's own arithmetic on the host and compares bit
// patterns.  Only the loop's STRUCTURE is transcribed, which is exactly where its
// traps live.
//
// ===========================================================================
// THE SEVEN TRAPS, EACH OF WHICH GETS A MUTANT
// ===========================================================================
//
//  1. **The loop is leaf-to-root and strictly sequential** —
//     `for (linkID = nbLinks - 1; linkID > 0; --linkID)` — because every
//     iteration accumulates into its PARENT, which a later iteration then reads
//     as its own `Ia`.  Reversing it, or bounding it at 0, is wrong.
//  2. **The parent's bias force is LOADED AT THE TOP**, before this link's own
//     `Ia` is stored — the P8.88 read-modify-write shape.  With a strictly
//     sequential loop that is the same thing, but a port that re-loads at the
//     store is only accidentally right.
//  3. **The inertia is the per-component RECIPROCAL of
//     `mInvInertiaXYZ_invMassW`, each with its own zero guard**, and so is the
//     mass.  A static link has `invMass == 0`, so an unguarded divide gives `inf`
//     on exactly the case the scene contains.
//  4. **The dual basis, spelled out.**  `Ia.bottom` (the TORQUE half of a FORCE)
//     picks up the inertia applied to `motionAcceleration.top` (the ANGULAR half
//     of a MOTION), and `Ia.top` (the LINEAR FORCE half) picks up
//     `motionAcceleration.bottom * m`.  Both halves cross over; a port that keeps
//     them parallel produces plausible numbers.
//  5. **The inertia is applied IN THE BODY FRAME**:
//     `rot.rotate(rot.rotateInv(a.top).multiply(inertia))`.  Dropping either
//     rotation, or swapping them, is a rotation of the result rather than an
//     obvious error.
//  6. **`motionAcceleration` is the SUM** of `mMotionAcceleration` and
//     `mMotionAccelerationInternal`.  The two are separate fields precisely
//     because the acceleration pass computes them separately.
//  7. **The root's bias force is ZEROED after the loop**, which discards
//     everything the loop accumulated into link 0.  A port that skips it leaves
//     the accumulated value, which is a plausible-looking wrong answer.
//
// And the sign: `translateSpatialVector(+rw, Ia)` here, NOT negated — the
// opposite of the acceleration direction's `-parentToChild`.

import pxgSlangCommon;
import pxgSlangArticulationDesc;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;
import pxgSlangArticulationDynamic;

// The per-component reciprocal with its own zero guard -- trap 3.
float alaRecip(float v) { return v == 0.0f ? 0.0f : 1.0f / v; }

// computeJointTransmittedFrictionForce (forwardDynamic2.cu:2317).
public void alaJointTransmittedFrictionForce(uint64_t linkBlocks, uint numLinks, uint lane)
{
    // leaf to root, strictly sequential -- trap 1
    for (uint linkID = numLinks - 1u; linkID > 0u; --linkID)
    {
        const uint64_t link = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;
        const uint parent = abU32(link, PXG_AB_LINK_mParents, lane);
        const uint64_t parentLink = linkBlocks + uint64_t(parent) * PXG_AB_LINK_SIZE;

        // loaded BEFORE this link's own Ia is stored -- trap 2
        float3 bfTop, bfBottom;
        abLoadSpatialVector(parentLink, PXG_AB_LINK_mBiasForce, lane, bfTop, bfBottom);
        const AbSpatialVector biasForce = asmSVec(bfTop, bfBottom);

        const float3 rw = float3(abF32(link, PXG_AB_LINK_mRw_x, lane),
                                 abF32(link, PXG_AB_LINK_mRw_y, lane),
                                 abF32(link, PXG_AB_LINK_mRw_z, lane));

        const float4 invInertiaXYZ_invMassW = abF4(link, PXG_AB_LINK_mInvInertiaXYZ_invMassW, lane);
        const float4 rot = abF4(link, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_Q, lane);

        const float m = alaRecip(invInertiaXYZ_invMassW.w);
        const float3 inertia = float3(alaRecip(invInertiaXYZ_invMassW.x),
                                      alaRecip(invInertiaXYZ_invMassW.y),
                                      alaRecip(invInertiaXYZ_invMassW.z));

        // the SUM of the two acceleration fields -- trap 6
        float3 aTop, aBottom, iTop, iBottom;
        abLoadSpatialVector(link, PXG_AB_LINK_mMotionAcceleration, lane, aTop, aBottom);
        abLoadSpatialVector(link, PXG_AB_LINK_mMotionAccelerationInternal, lane, iTop, iBottom);
        const AbSpatialVector motionAcceleration =
            asmSVecAdd(asmSVec(aTop, aBottom), asmSVec(iTop, iBottom));

        float3 iaTop, iaBottom;
        abLoadSpatialVector(link, PXG_AB_LINK_mBiasForce, lane, iaTop, iaBottom);

        // the dual basis and the body-frame inertia -- traps 4 and 5.
        // `PxVec3::multiply` is the COMPONENTWISE product, not a dot.
        iaBottom = iaBottom
                 + quatRotate(rot, quatRotateInv(rot, motionAcceleration.top) * inertia);
        iaTop = iaTop + motionAcceleration.bottom * m;

        abStoreSpatialVector(link, PXG_AB_LINK_mBiasForce, lane, iaTop, iaBottom);

        // NOT negated -- into the parent's frame
        const AbSpatialVector translated =
            adTranslateSpatialVector(rw, asmSVec(iaTop, iaBottom));
        const AbSpatialVector parentBias = asmSVecAdd(biasForce, translated);
        abStoreSpatialVector(parentLink, PXG_AB_LINK_mBiasForce, lane, parentBias.top,
                             parentBias.bottom);
    }

    // the root's accumulation is DISCARDED -- trap 7
    abStoreSpatialVector(linkBlocks, PXG_AB_LINK_mBiasForce, lane, float3(0.0f, 0.0f, 0.0f),
                         float3(0.0f, 0.0f, 0.0f));
}

// ===========================================================================
// The fixture entry points
// ===========================================================================

struct AlaFrictionParams
{
    uint64_t linkBlocks;   // @0
    uint64_t numLinksPer;  // @8   uint32[nbLanes] -- a DIFFERENT count per lane
    uint nbLanes;          // @16
    uint pad0;             // @20 -> 24, padded to 32 by the shim's alignment
    uint pad1;             // @24
    uint pad2;             // @28
};

[[vk::push_constant]]
ConstantBuffer<AlaFrictionParams> gAlaFriction;

[shader("compute")]
[numthreads(32, 1, 1)]
void artiJointFriction(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gAlaFriction.nbLanes)
        return;

    const uint numLinks = (Ptr<uint>(gAlaFriction.numLinksPer))[lane];
    alaJointTransmittedFrictionForce(gAlaFriction.linkBlocks, numLinks, lane);
}

// The parallel-axis theorem, dumped so `asmTranslateInertiaMass` has a fixture of
// its own rather than being checked only through the compound inertia it feeds.
struct AlaInertiaParams
{
    uint64_t inFloats;    // @0   float[nbCases * 13]  -- 9 inertia + 1 mass + 3 offset
    uint64_t outFloats;   // @8   float[nbCases * 9]
    uint nbCases;         // @16
    uint pad0;            // @20
    uint pad1;            // @24
    uint pad2;            // @28
};

[[vk::push_constant]]
ConstantBuffer<AlaInertiaParams> gAlaInertia;

[shader("compute")]
[numthreads(64, 1, 1)]
void artiTranslateInertiaMass(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint tid = dispatchThreadId.x;
    if (tid >= gAlaInertia.nbCases)
        return;

    Ptr<float> i = Ptr<float>(gAlaInertia.inFloats + uint64_t(tid) * 13u * 4u);
    Ptr<float> o = Ptr<float>(gAlaInertia.outFloats + uint64_t(tid) * 9u * 4u);

    const AbMat33 inertia = asmM33(float3(i[0], i[1], i[2]), float3(i[3], i[4], i[5]),
                                   float3(i[6], i[7], i[8]));
    const AbMat33 r = asmTranslateInertiaMass(inertia, i[9], float3(i[10], i[11], i[12]));

    o[0] = r.c0.x; o[1] = r.c0.y; o[2] = r.c0.z;
    o[3] = r.c1.x; o[4] = r.c1.y; o[5] = r.c1.z;
    o[6] = r.c2.x; o[7] = r.c2.y; o[8] = r.c2.z;
}
