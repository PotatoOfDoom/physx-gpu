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

// npArtiJointFrame.slang
//
// The joint-frame math at the head of jcalc: `computeJointAxis`
// (forwardDynamic2.cu:337) and `ArticulationJointCore::setJointFrame` /
// `computeMotionMatrix` (DyArticulationJointCore.h:139-207).
//
// This is the first piece of real articulation DYNAMICS in the port, as opposed
// to bookkeeping or address maps, and it is deliberately taken first because it
// is the one piece of jcalc that can be checked against an ORACLE THAT IS THE
// ENGINE ITSELF: both functions are `PX_CUDA_CALLABLE`, so the fixture runs the
// real `Dy::ArticulationJointCore::setJointFrame` on the host over the same
// inputs and compares bit patterns.  Nothing is transcribed into the test.
//
// ===========================================================================
// WHY THE ARITHMETIC IS SPELLED OUT INSTEAD OF USING THE OBVIOUS INTRINSICS
// ===========================================================================
//
// `rotateAndNormalize` (DyArticulationJointCore.h:42) is a hand-fused
// rotate-then-normalize that avoids some multiplies, and its exact expression
// order is part of the result:
//
//     w2   = w*w - 0.5
//     dot2 = x*vx + y*vy + z*vz
//     rotated = (vx*w2 + (y*vz - z*vy)*w + x*dot2, ...)
//
// and `PxVec3::getNormalized()` is `m > 0 ? v * PxRecipSqrt(m) : (0,0,0)` --
// a RECIPROCAL square root times the vector, with an explicit zero-length
// guard, not a division and not `normalize()`.  Slang's `normalize()` is free
// to lower to anything the target likes, and `rsqrt` is permitted to be an
// approximation; either would put this off the engine's bit pattern.  So the
// expression is written out, `1.0f / sqrt(m)` is spelled as such, and the
// module is registered in PXG_NOCONTRACT_SHADERS so no `a*b+c` is fused into an
// FMA behind the transcription's back.
//
// The zero-length guard is not decoration either: a LOCKED axis produces a zero
// joint axis, `magnitudeSquared()` is then exactly 0, and the engine returns
// (0,0,0) rather than a NaN.  Dropping the guard would poison the motion matrix
// with NaNs for exactly the configurations a fixed joint produces.

import pxgSlangCommon;
import pxgSlangArticulationDesc;
import pxgSlangArticulationBlocks;   // the PXG_AR_* record offsets (P8.70)

// PxArticulationJointType (PxArticulationReducedCoordinate.h:301-306).
//
// THESE VALUES ARE MEASURED, NOT INFERRED, and the distinction cost five
// rewrites.  The first version took the order from the sequence the types
// appear in `computeMotionMatrix`'s switch -- prismatic, revolute, unwrapped,
// spherical, fix -- which is off by one from the enum, where eFIX is 0.  The
// failure that produced was diabolical: every case that landed on a
// neighbouring branch which happens to compute the same thing PASSED
// (eREVOLUTE hitting the unwrapped label, eREVOLUTE_UNWRAPPED hitting the
// spherical label with dofs == 1, eFIX hitting prismatic with a zero axis), and
// only prismatic and spherical failed.  Five structurally different rewrites of
// correct code later, echoing the shader's own inputs back to the host found it
// in one run.  The lesson is not about enums: when a fix does not move the
// failure count, stop fixing and start measuring.
static const uint AJ_FIX = 0u;
static const uint AJ_PRISMATIC = 1u;
static const uint AJ_REVOLUTE = 2u;
static const uint AJ_REVOLUTE_UNWRAPPED = 3u;
static const uint AJ_SPHERICAL = 4u;

// PxVec3::getNormalized(): m > 0 ? v * PxRecipSqrt(m) : PxVec3(0)
float3 ajGetNormalized(float3 v)
{
    const float m = v.x * v.x + v.y * v.y + v.z * v.z;
    return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0.0f, 0.0f, 0.0f);
}

// Dy::rotateAndNormalize(q, v) -- the fused form, expression order preserved
float3 ajRotateAndNormalize(float4 q, float3 v)
{
    const float vx = v.x, vy = v.y, vz = v.z;
    const float x = q.x, y = q.y, z = q.z, w = q.w;

    const float w2 = w * w - 0.5f;
    const float dot2 = (x * vx + y * vy + z * vz);
    const float3 rotated = float3(
        (vx * w2 + (y * vz - z * vy) * w + x * dot2),
        (vy * w2 + (z * vx - x * vz) * w + y * dot2),
        (vz * w2 + (x * vy - y * vx) * w + z * dot2));

    return ajGetNormalized(rotated);
}

// computeJointAxis: for each of the joint's dofs, a unit spatial vector whose
// single set component is named by dofIds[i].  Cm::UnAlignedSpatialVector's
// operator[] runs top.x, top.y, top.z, bottom.x, bottom.y, bottom.z, so an
// angular dof (index 0-2) sets a TOP component and a linear one (3-5) a BOTTOM
// component -- the same array, and the reason a prismatic joint reads
// `jointAxis.bottom` while a revolute reads `jointAxis.top`.
//
// The axis is computed on demand for one dof index rather than built into an
// array.  That is a style choice, not a workaround: an earlier version used
// `out float3 [3]` parameters and another a returned struct of arrays, both
// were rewritten while chasing the enum bug above, and NEITHER was ever shown
// to be at fault.  The array forms are not known to be broken and this comment
// does not claim they are — the record said so briefly and that was wrong.
//
// The CUDA's `if (i < dof)` guard goes away with the array, and that is safe
// rather than convenient: `computeMotionMatrix` reads index 0 only for
// prismatic and revolute (dof == 1) and indices < dofs for spherical, so no
// reader ever touches an index the guard would have left unwritten.  In the
// CUDA those entries are uninitialised stack; here they are never asked for.
float3 ajAxisTop(uint dofIdWord, uint i)
{
    const uint ind = (dofIdWord >> (i * 8u)) & 0xFFu;
    // ind == 0xFF marks a LOCKED axis: nothing matches, the axis is zero, and
    // that is the input the normalize guard exists for.
    return float3(ind == 0u ? 1.0f : 0.0f,
                  ind == 1u ? 1.0f : 0.0f,
                  ind == 2u ? 1.0f : 0.0f);
}

float3 ajAxisBottom(uint dofIdWord, uint i)
{
    const uint ind = (dofIdWord >> (i * 8u)) & 0xFFu;
    return float3(ind == 3u ? 1.0f : 0.0f,
                  ind == 4u ? 1.0f : 0.0f,
                  ind == 5u ? 1.0f : 0.0f);
}

// ArticulationJointCore::computeMotionMatrix, six named outputs.  `dofs` is
// only read by the spherical case; prismatic and revolute write column 0 and
// assert the count, and eFIX writes nothing at all -- so its columns keep the
// zeros set here, which is what the fixture's zero-seeded oracle checks.
struct AjMotionMatrix
{
    float3 top0, bottom0;
    float3 top1, bottom1;
    float3 top2, bottom2;
};

AjMotionMatrix ajComputeMotionMatrix(uint jointType, uint dofs, float4 childQ, float3 childP,
                                     uint dofIdWord)
{
    AjMotionMatrix mm;
    mm.top0 = float3(0.0f, 0.0f, 0.0f); mm.bottom0 = float3(0.0f, 0.0f, 0.0f);
    mm.top1 = float3(0.0f, 0.0f, 0.0f); mm.bottom1 = float3(0.0f, 0.0f, 0.0f);
    mm.top2 = float3(0.0f, 0.0f, 0.0f); mm.bottom2 = float3(0.0f, 0.0f, 0.0f);

    const float3 childOffset = -childP;

    if (jointType == AJ_PRISMATIC)
    {
        // the LINEAR axis: bottom, and the motion matrix's top stays zero
        mm.bottom0 = ajRotateAndNormalize(childQ, ajAxisBottom(dofIdWord, 0u));
    }
    else if (jointType == AJ_REVOLUTE || jointType == AJ_REVOLUTE_UNWRAPPED)
    {
        const float3 u = ajRotateAndNormalize(childQ, ajAxisTop(dofIdWord, 0u));
        mm.top0 = u;
        mm.bottom0 = cross(u, childOffset);
    }
    else if (jointType == AJ_SPHERICAL)
    {
        if (dofs > 0u)
        {
            const float3 u = ajRotateAndNormalize(childQ, ajAxisTop(dofIdWord, 0u));
            mm.top0 = u; mm.bottom0 = cross(u, childOffset);
        }
        if (dofs > 1u)
        {
            const float3 u = ajRotateAndNormalize(childQ, ajAxisTop(dofIdWord, 1u));
            mm.top1 = u; mm.bottom1 = cross(u, childOffset);
        }
        if (dofs > 2u)
        {
            const float3 u = ajRotateAndNormalize(childQ, ajAxisTop(dofIdWord, 2u));
            mm.top2 = u; mm.bottom2 = cross(u, childOffset);
        }
    }
    // AJ_FIX and anything unrecognised: the engine's switch falls through
    // without writing.
    return mm;
}

// ---------------------------------------------------------------------------
// Test entry: one joint per thread.  Input is a packed record per joint; output
// is the relative quat plus the three motion-matrix columns.
// ---------------------------------------------------------------------------
// NOTE the deliberate absence of vector types after the two float4s.  A `uint4
// dofIds` here would be ALIGNED TO 16 BYTES by Slang and land at @64, while the
// host's `uint8_t dofIds[16]` sits at @52 -- the first version of this record
// did exactly that, the shader read `dof` as its dof-id byte, and every
// prismatic and spherical case came back with a zero axis.  Scalars have
// 4-byte alignment and cannot drift.
struct AjJointIn
{
    float4 parentQ;   // @0
    float4 childQ;    // @16
    float childPx;    // @32
    float childPy;    // @36
    float childPz;    // @40
    uint jointType;   // @44
    uint dof;         // @48
    uint dofIds0;     // @52  bytes 0-3 of the host's dofIds[]
    uint dofIds1;     // @56  bytes 4-7
    uint dofIds2;     // @60
    uint dofIds3;     // @64
    uint pad0;        // @68
    uint pad1;        // @72
    uint pad2;        // @76
};                    // = 80

struct AjJointOut
{
    float4 relativeQuat; // @0
    float4 mm0Top;       // @16 (.w unused)
    float4 mm0Bottom;    // @32
    float4 mm1Top;       // @48
    float4 mm1Bottom;    // @64
    float4 mm2Top;       // @80
    float4 mm2Bottom;    // @96
};                       // = 112

struct AjParams
{
    uint64_t joints;
    uint64_t out;
    uint count;
};

[[vk::push_constant]]
ConstantBuffer<AjParams> gAj;

[shader("compute")]
[numthreads(64, 1, 1)]
void artiJointFrame(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadId.x;
    if (idx >= gAj.count)
        return;

    const AjJointIn j = (Ptr<AjJointIn>(gAj.joints))[idx];

    const float3 childP = float3(j.childPx, j.childPy, j.childPz);
    const AjMotionMatrix mm =
        ajComputeMotionMatrix(j.jointType, j.dof, j.childQ, childP, j.dofIds0);

    // setJointFrame: relativeQuat = (childPose.q * parentPose.q.getConjugate()).getNormalized()
    const float4 conj = float4(-j.parentQ.x, -j.parentQ.y, -j.parentQ.z, j.parentQ.w);
    const float4 prod = quatMul(j.childQ, conj);
    const float m = prod.x * prod.x + prod.y * prod.y + prod.z * prod.z + prod.w * prod.w;
    // PxQuat::getNormalized() is `*this * PxRecipSqrt(magnitudeSquared())` with
    // no zero guard -- a quaternion is never expected to be zero-length, and
    // adding a guard the engine does not have would diverge on the one input
    // where it matters.
    const float4 rq = prod * (1.0f / sqrt(m));

    Ptr<AjJointOut> o = Ptr<AjJointOut>(gAj.out);
    AjJointOut r;
    r.relativeQuat = rq;
    r.mm0Top = float4(mm.top0, 0.0f);
    r.mm0Bottom = float4(mm.bottom0, 0.0f);
    r.mm1Top = float4(mm.top1, 0.0f);
    r.mm1Bottom = float4(mm.bottom1, 0.0f);
    r.mm2Top = float4(mm.top2, 0.0f);
    r.mm2Bottom = float4(mm.bottom2, 0.0f);
    o[idx] = r;
}


// ===========================================================================
// P8.72: jcalc PHASE A -- the record-side recompute (forwardDynamic2.cu:379-432)
// ===========================================================================
//
// Guarded by `confiDirty || dataDirty`, one thread per articulation (the CUDA is
// single-threaded here too -- the "1T" in the caller's name).  It recomputes the
// joint axes, motion matrices and relative quats INTO THE ARTICULATION RECORD,
// and fills jointOffsets / cfmScale / cfms alongside.  The block-format copy is
// a separate phase.
//
// The per-joint maths is the code above, already proven bit-exact against the
// engine's own setJointFrame (P8.69).  What is new here is the loop, the dof
// accumulation and five record writes.
//
// THE FLAG THAT IS EASY TO MISS: setJointFrame is a NO-OP unless the joint's
// `jCalcUpdateFrames` byte is set, and it CLEARS that byte on the way out.  The
// CUDA calls it unconditionally and lets the flag decide, so the port must read
// and clear it too -- skipping that would recompute frames every frame for every
// joint, which is not wrong numerically but is not what the engine does, and
// leaving it set would be.

struct JcalcAParams
{
    uint64_t articulation;   // one PxgArticulation record
    float invLengthScale;
    uint pad0;
};

[[vk::push_constant]]
ConstantBuffer<JcalcAParams> gJcalcA;

// Cm::UnAlignedSpatialVector store: six tightly packed floats, no padding, so a
// float3 store would be wrong at the 12-byte boundary.
void ajStoreSpatialVector(uint64_t addr, float3 top, float3 bottom)
{
    Ptr<float> f = Ptr<float>(addr);
    f[0] = top.x;    f[1] = top.y;    f[2] = top.z;
    f[3] = bottom.x; f[4] = bottom.y; f[5] = bottom.z;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void artiJcalcPhaseA()
{
    const uint64_t a = gJcalcA.articulation;

    const uint flags = artiU8Field(a, PXG_ARTI_DATA_FLAGS);
    const bool confiDirty = artiU8Field(a, PXG_ARTI_DATA_CONFI_DIRTY) != 0u;
    const bool dataDirty = artiU32Field(a, PXG_ARTI_DATA_UPDATE_DIRTY) != 0u;
    if (!(confiDirty || dataDirty))
        return;

    const bool fixBase = (flags & PXG_ARTI_FLAG_FIX_BASE) != 0u;
    const uint numLinks = artiU16Field(a, PXG_ARTI_DATA_NUM_LINKS);
    const float invLengthScale = gJcalcA.invLengthScale;

    const uint64_t links = artiPtrField(a, PXG_ARTI_LINKS);
    const uint64_t props = artiPtrField(a, PXG_ARTI_LINK_PROPS);
    const uint64_t joints = artiPtrField(a, PXG_ARTI_JOINTS);
    const uint64_t jointData = artiPtrField(a, PXG_ARTI_JOINT_DATA);
    const uint64_t jointAxis = artiPtrField(a, PXG_ARTI_JOINT_AXIS);
    const uint64_t motionMatrix = artiPtrField(a, PXG_ARTI_MOTION_MATRIX);
    const uint64_t relativeQuat = artiPtrField(a, PXG_ARTI_RELATIVE_QUAT);
    Ptr<uint> jointOffsets = Ptr<uint>(artiPtrField(a, PXG_ARTI_JOINT_OFFSETS));
    Ptr<float> cfmScale = Ptr<float>(artiPtrField(a, PXG_ARTI_CFM_SCALE));
    Ptr<float> cfms = Ptr<float>(artiPtrField(a, PXG_ARTI_CFMS));

    // the root, whose cfm is zero when the base is fixed
    if (confiDirty)
        cfmScale[0] = fixBase ? 0.0f
            : asfloat((Ptr<uint>(links + PXG_ARTI_LINK_CFM_SCALE))[0]) * invLengthScale;
    if (dataDirty)
        cfms[0] = fixBase ? 0.0f
            : asfloat((Ptr<uint>(props + PXG_ARTI_LINK_PROP_INV_MASS))[0]) * cfmScale[0];

    uint totalDofs = 0u;
    for (uint linkID = 1u; linkID < numLinks; ++linkID)
    {
        const uint64_t jd = jointData + uint64_t(linkID) * PXG_ARTI_JOINT_CORE_DATA_SIZE;
        const uint dof = artiU8Field(jd, PXG_AR_JCD_nbDof);     // PxU8, not PxU32
        const uint offset = artiU32Field(jd, PXG_AR_JCD_jointOffset);

        const uint64_t jc = joints + uint64_t(linkID) * PXG_ARTI_JOINT_CORE_SIZE;
        const uint dofIdWord = artiU32Field(jc, PXG_AR_JOINT_dofIds);

        // computeJointAxis: dof unit spatial vectors at jointAxis[totalDofs..]
        for (uint i = 0u; i < 3u; ++i)
        {
            if (i < dof)
                ajStoreSpatialVector(
                    jointAxis + uint64_t(totalDofs + i) * PXG_ARTI_SPATIAL_VECTOR_SIZE,
                    ajAxisTop(dofIdWord, i), ajAxisBottom(dofIdWord, i));
        }

        // setJointFrame, flag and all
        if (artiU8Field(jc, PXG_AR_JOINT_jCalcUpdateFrames) != 0u)
        {
            const uint jointType = artiU8Field(jc, PXG_AR_JOINT_jointType);
            const float4 parentQ = float4(asfloat((Ptr<uint>(jc + PXG_AR_JOINT_parentPose))[0]),
                                          asfloat((Ptr<uint>(jc + PXG_AR_JOINT_parentPose))[1]),
                                          asfloat((Ptr<uint>(jc + PXG_AR_JOINT_parentPose))[2]),
                                          asfloat((Ptr<uint>(jc + PXG_AR_JOINT_parentPose))[3]));
            // PxTransform is { PxQuat q; PxVec3 p; } -- q first, then p at +16
            const float4 childQ = float4(asfloat((Ptr<uint>(jc + PXG_AR_JOINT_childPose))[0]),
                                         asfloat((Ptr<uint>(jc + PXG_AR_JOINT_childPose))[1]),
                                         asfloat((Ptr<uint>(jc + PXG_AR_JOINT_childPose))[2]),
                                         asfloat((Ptr<uint>(jc + PXG_AR_JOINT_childPose))[3]));
            const float3 childP = float3(asfloat((Ptr<uint>(jc + PXG_AR_JOINT_childPose))[4]),
                                         asfloat((Ptr<uint>(jc + PXG_AR_JOINT_childPose))[5]),
                                         asfloat((Ptr<uint>(jc + PXG_AR_JOINT_childPose))[6]));

            const float4 conj = float4(-parentQ.x, -parentQ.y, -parentQ.z, parentQ.w);
            const float4 prod = quatMul(childQ, conj);
            const float m = prod.x * prod.x + prod.y * prod.y + prod.z * prod.z + prod.w * prod.w;
            const float4 rq = prod * (1.0f / sqrt(m));

            Ptr<uint> rqOut = Ptr<uint>(relativeQuat + uint64_t(linkID) * PXG_ARTI_QUAT_SIZE);
            rqOut[0] = asuint(rq.x); rqOut[1] = asuint(rq.y);
            rqOut[2] = asuint(rq.z); rqOut[3] = asuint(rq.w);

            const AjMotionMatrix mm =
                ajComputeMotionMatrix(jointType, dof, childQ, childP, dofIdWord);
            const uint64_t mmBase = motionMatrix + uint64_t(linkID) * PXG_ARTI_SUBSPACE_STRIDE;
            ajStoreSpatialVector(mmBase + 0u * PXG_ARTI_SPATIAL_VECTOR_SIZE, mm.top0, mm.bottom0);
            ajStoreSpatialVector(mmBase + 1u * PXG_ARTI_SPATIAL_VECTOR_SIZE, mm.top1, mm.bottom1);
            ajStoreSpatialVector(mmBase + 2u * PXG_ARTI_SPATIAL_VECTOR_SIZE, mm.top2, mm.bottom2);

            artiStoreU8Field(jc, PXG_AR_JOINT_jCalcUpdateFrames, 0u);
        }

        jointOffsets[linkID] = offset;

        if (confiDirty)
            cfmScale[linkID] = asfloat((Ptr<uint>(links + uint64_t(linkID) * PXG_ARTI_LINK_SIZE
                                                  + PXG_ARTI_LINK_CFM_SCALE))[0]) * invLengthScale;
        if (dataDirty)
            cfms[linkID] = asfloat((Ptr<uint>(props + uint64_t(linkID) * PXG_ARTI_LINK_PROP_SIZE
                                              + PXG_ARTI_LINK_PROP_INV_MASS))[0]) * cfmScale[linkID];

        totalDofs += dof;
    }

    // confiDirty is cleared; updateDirty deliberately is NOT -- the CUDA holds on
    // to it until the block copy in phase C has consumed it.
    artiStoreU8Field(a, PXG_ARTI_DATA_CONFI_DIRTY, 0u);
}


// ===========================================================================
// P8.73: jcalc PHASE B part 1 -- the root link fill and the path-to-root
// bitfields (forwardDynamic2.cu:437-467 and :553-566)
// ===========================================================================
//
// Phase B copies the record into the BLOCK format, and this half of it does the
// root link (which has no joint, so it is filled outside the per-link loop) plus
// the whole path-to-root bitfield build.  The bitfield writes are lifted out of
// the link loop deliberately: they are the only thing in that loop that does not
// touch the link block, and keeping them together makes the wordSize indexing
// checkable on its own.
//
// ---------------------------------------------------------------------------
// THE HAZARD THAT IS NEW HERE: FOUR LANES PER WORD
// ---------------------------------------------------------------------------
//
// Every thread here is one articulation -- `threadIndexInWarp` is the lane, and
// the block arrays are shared by the whole warp.  mDofs, mDisableGravity and
// mRetainsAcceleration are PxU8[32] fields, so lanes 0-3 write four bytes of the
// SAME 32-bit word at the same time.  CUDA emits a byte store and the lanes never
// interact; SPIR-V has no byte store here, so the port needs the atomic and/or
// pair in abStoreU8 (pxgSlangArticulationBlocks.slang).  A plain read-modify-write
// compiles, runs, and silently loses three writes in four.
//
// The 64-bit bitfields are the benign case: PxgArticulationBitFieldStackData is
// PxU64[32], so each lane owns eight contiguous bytes and a load/or/store is
// race-free.
//
// ---------------------------------------------------------------------------
// VERIFICATION IS WEAKER HERE THAN IN PHASE A, AND THAT IS WORTH SAYING
// ---------------------------------------------------------------------------
//
// Phase A's maths came from `setJointFrame`, which is PX_CUDA_CALLABLE, so the
// fixture could call the engine.  `jcalc` itself is __device__-only: there is no
// engine function to compare phase B against, and the expectation in
// testArtiJcalcPhaseB1.cpp is therefore built by the fixture.  What carries the
// weight instead is (a) every offset used below being generated from the engine
// headers and pinned by static_assert, (b) the bitfields being derived a SECOND
// way in the fixture -- by walking the parent chain rather than replaying this
// loop -- and (c) a heavier mutation sweep.  That is not the same thing as an
// oracle and the tranche note says so.

struct JcalcB1Params
{
    uint64_t articulations;    // PxgArticulation[lane]
    uint64_t linkBlocks;       // this warp's PxgArticulationBlockLinkData[maxNumLinks]
    uint64_t bitFieldBlocks;   // this warp's bitfield rows [maxNumLinks * wordSize]
    uint maxNumLinks;
    uint nbArticulations;
};

[[vk::push_constant]]
ConstantBuffer<JcalcB1Params> gJcalcB1;

[shader("compute")]
[numthreads(32, 1, 1)]
void artiJcalcPhaseB1(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gJcalcB1.nbArticulations)
        return;

    const uint64_t a = gJcalcB1.articulations + uint64_t(lane) * PXG_ARTI_STRIDE;
    const uint numLinks = artiU16Field(a, PXG_ARTI_DATA_NUM_LINKS);
    const uint maxNumLinks = gJcalcB1.maxNumLinks;

    const uint64_t links = artiPtrField(a, PXG_ARTI_LINKS);
    const uint64_t body2Worlds = artiPtrField(a, PXG_ARTI_LINK_BODY2WORLDS);
    const Ptr<uint> pathToRootElems = Ptr<uint>(artiPtrField(a, PXG_ARTI_PATH_TO_ROOT));

    // ---- the root link block (element 0 of the warp's link block array) ----
    const uint64_t root = gJcalcB1.linkBlocks;

    abStoreU8(root, PXG_AB_LINK_mDisableGravity, lane,
              artiU8Field(links, PXG_AR_LINK_disableGravity));
    abStoreU8(root, PXG_AB_LINK_mRetainsAcceleration, lane,
              artiU8Field(links, PXG_AR_LINK_retainsAccelerations));
    // the root has no inbound joint, so its dof count is zero by definition and
    // is written rather than copied
    abStoreU8(root, PXG_AB_LINK_mDofs, lane, 0u);

    abStoreF4(root, PXG_AB_LINK_mLinDampingX_AngDampingY_maxLinVelSqZ_maxAngVelSqW, lane,
              float4(asfloat(artiU32Field(links, PXG_AR_LINK_linearDamping)),
                     asfloat(artiU32Field(links, PXG_AR_LINK_angularDamping)),
                     asfloat(artiU32Field(links, PXG_AR_LINK_maxLinearVelocitySq)),
                     asfloat(artiU32Field(links, PXG_AR_LINK_maxAngularVelocitySq))));

    // initialAngVelXYZ_penBiasClamp / initialLinVelXYZ_invMassW: the device sees
    // two float4s where the host declares PxVec3+PxReal pairs over the same bytes,
    // and only the xyz halves go into the motion velocity -- the .w components
    // (penBiasClamp, invMass) are unrelated fields that happen to share the vector.
    const float3 initAng = float3(asfloat(artiU32Field(links, PXG_AR_LINK_initialAngVel + 0u)),
                                  asfloat(artiU32Field(links, PXG_AR_LINK_initialAngVel + 4u)),
                                  asfloat(artiU32Field(links, PXG_AR_LINK_initialAngVel + 8u)));
    const float3 initLin = float3(asfloat(artiU32Field(links, PXG_AR_LINK_initialLinVel + 0u)),
                                  asfloat(artiU32Field(links, PXG_AR_LINK_initialLinVel + 4u)),
                                  asfloat(artiU32Field(links, PXG_AR_LINK_initialLinVel + 8u)));
    // Cm::UnAlignedSpatialVector(angular, linear) -- top is the ANGULAR half
    abStoreSpatialVector(root, PXG_AB_LINK_mMotionVelocity, lane, initAng, initLin);

    // PxTransform is { PxQuat q; PxVec3 p; }: q at +0, p at +16
    const uint64_t b2w0 = body2Worlds;
    abStoreSpatialTransform(root, PXG_AB_LINK_mAccumulatedPose, lane,
                            float4(asfloat(artiU32Field(b2w0, 0u)),
                                   asfloat(artiU32Field(b2w0, 4u)),
                                   asfloat(artiU32Field(b2w0, 8u)),
                                   asfloat(artiU32Field(b2w0, 12u))),
                            float3(asfloat(artiU32Field(b2w0, 16u)),
                                   asfloat(artiU32Field(b2w0, 20u)),
                                   asfloat(artiU32Field(b2w0, 24u))));

    abStoreU32(root, PXG_AB_LINK_mChildrenOffset, lane,
               artiU32Field(links, PXG_AR_LINK_childrenOffset));
    // numChildren is a PxU16 in the record and a PxU32 in the block
    abStoreU32(root, PXG_AB_LINK_mNumChildren, lane,
               artiU16Field(links, PXG_AR_LINK_numChildren));

    // ---- the path-to-root bitfields ----
    const uint wordSize = (maxNumLinks + 63u) / 64u;
    const uint64_t bf = gJcalcB1.bitFieldBlocks;

    for (uint i = 0u; i < maxNumLinks * wordSize; ++i)
        abStoreU64(bf + uint64_t(i) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane, uint64_t(0));

    // the root is on its own path
    abStoreU64(bf, 0u, lane, abU64(bf, 0u, lane) | uint64_t(1));

    for (uint linkID = 1u; linkID < numLinks; ++linkID)
    {
        const uint64_t lrec = links + uint64_t(linkID) * PXG_ARTI_LINK_SIZE;
        const uint offset = artiU32Field(lrec, PXG_AR_LINK_pathToRootOffset);
        const uint numPathToRoot = artiU16Field(lrec, PXG_AR_LINK_numPathToRoot);

        const uint64_t rowBase = bf + uint64_t(linkID * wordSize) * PXG_AB_BITFIELD_STACK_SIZE;

        for (uint j = 0u; j < numPathToRoot; ++j)
        {
            const uint index = pathToRootElems[offset + j];
            const uint word = index / 64u;
            const uint64_t slot = rowBase + uint64_t(word) * PXG_AB_BITFIELD_STACK_SIZE;
            // the shift is (index - 64*word), i.e. index % 64, and it is a 64-bit
            // shift -- writing it as `1 << index` would be the P8.47 trap, where a
            // 32-bit shift CLAMPS instead of wrapping
            const uint64_t bit = uint64_t(1) << (index - 64u * word);
            abStoreU64(slot, 0u, lane, abU64(slot, 0u, lane) | bit);
        }

        // and the root again, unconditionally, into word 0 of this link's row
        abStoreU64(rowBase, 0u, lane, abU64(rowBase, 0u, lane) | uint64_t(1));
    }
}
