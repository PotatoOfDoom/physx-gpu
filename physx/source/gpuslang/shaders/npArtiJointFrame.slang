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

// The phase bodies are FUNCTIONS, not entry points, so that assembling
// computeUnconstrainedVelocities1TLaunch later is composition rather than a
// second transcription of the same CUDA.  Each entry point below is a thin
// harness that supplies the lane and the block bases the engine kernel will
// supply from the core descriptor.
// PxTransform is { PxQuat q; PxVec3 p; } -- q at +0, p at +16, 28 bytes with no
// tail padding, so the two halves are read as seven scalars rather than as a
// float4 pair (`p` is not 16-byte aligned for odd link indices).
void jcalcStorePose(uint64_t blk, uint fieldOffset, uint lane, uint64_t src)
{
    abStoreSpatialTransform(blk, fieldOffset, lane,
                            float4(asfloat(artiU32Field(src, 0u)),
                                   asfloat(artiU32Field(src, 4u)),
                                   asfloat(artiU32Field(src, 8u)),
                                   asfloat(artiU32Field(src, 12u))),
                            float3(asfloat(artiU32Field(src, 16u)),
                                   asfloat(artiU32Field(src, 20u)),
                                   asfloat(artiU32Field(src, 24u))));
}

void jcalcFillRootLinkAndBitFields(uint64_t a, uint64_t linkBlocks, uint64_t bitFields,
                                   uint maxNumLinks, uint lane)
{
    const uint numLinks = artiU16Field(a, PXG_ARTI_DATA_NUM_LINKS);

    const uint64_t links = artiPtrField(a, PXG_ARTI_LINKS);
    const uint64_t body2Worlds = artiPtrField(a, PXG_ARTI_LINK_BODY2WORLDS);
    const Ptr<uint> pathToRootElems = Ptr<uint>(artiPtrField(a, PXG_ARTI_PATH_TO_ROOT));

    // ---- the root link block (element 0 of the warp's link block array) ----
    const uint64_t root = linkBlocks;

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

    jcalcStorePose(root, PXG_AB_LINK_mAccumulatedPose, lane, body2Worlds);

    abStoreU32(root, PXG_AB_LINK_mChildrenOffset, lane,
               artiU32Field(links, PXG_AR_LINK_childrenOffset));
    // numChildren is a PxU16 in the record and a PxU32 in the block
    abStoreU32(root, PXG_AB_LINK_mNumChildren, lane,
               artiU16Field(links, PXG_AR_LINK_numChildren));

    // ---- the path-to-root bitfields ----
    const uint wordSize = (maxNumLinks + 63u) / 64u;
    const uint64_t bf = bitFields;

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

[shader("compute")]
[numthreads(32, 1, 1)]
void artiJcalcPhaseB1(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gJcalcB1.nbArticulations)
        return;

    jcalcFillRootLinkAndBitFields(gJcalcB1.articulations + uint64_t(lane) * PXG_ARTI_STRIDE,
                                  gJcalcB1.linkBlocks, gJcalcB1.bitFieldBlocks,
                                  gJcalcB1.maxNumLinks, lane);
}


// ===========================================================================
// P8.74: jcalc PHASE B2 -- the per-link block fields (forwardDynamic2.cu:545-575)
// ===========================================================================
//
// The rest of phase B's link loop, minus the per-dof fill: for every link from 1
// up, the relative quat, joint offset, dof count, joint type, damping float4,
// parent index, children offsets, gravity flags and the three poses.  The
// bitfield writes that sit interleaved with these in the CUDA are P8.73's.
//
// Three shapes here that a uniform-looking run hides:
//
//  * mDofs and mJointType are PxU8[32] and mJointOffset/mParents/mChildrenOffset/
//    mNumChildren are PxU32[32] -- SAME LOOP, different element strides, so a
//    single "store a lane" helper used for all of them would be wrong for half.
//  * mRelativeQuat is a bare float4[32] and NOT a PxgSpatialTransformBlock, even
//    though it holds a quaternion and sits next to blocks that are; the CUDA
//    reinterpret_casts the PxQuat array to float4* and copies 16 bytes.
//  * nbDof is read out of ArticulationJointCoreData, where it is a PxU8 at
//    offset 4 -- the byte-not-word trap P8.70 and P8.72 both hit.

struct JcalcB2Params
{
    uint64_t articulations;
    uint64_t linkBlocks;
    uint nbArticulations;
    uint pad0;
};

[[vk::push_constant]]
ConstantBuffer<JcalcB2Params> gJcalcB2;

void jcalcFillLinkBlocks(uint64_t a, uint64_t linkBlocks, uint lane)
{
    const uint numLinks = artiU16Field(a, PXG_ARTI_DATA_NUM_LINKS);

    const uint64_t links = artiPtrField(a, PXG_ARTI_LINKS);
    const uint64_t joints = artiPtrField(a, PXG_ARTI_JOINTS);
    const uint64_t jointData = artiPtrField(a, PXG_ARTI_JOINT_DATA);
    const uint64_t body2Worlds = artiPtrField(a, PXG_ARTI_LINK_BODY2WORLDS);
    const uint64_t relativeQuats = artiPtrField(a, PXG_ARTI_RELATIVE_QUAT);
    const Ptr<uint> parents = Ptr<uint>(artiPtrField(a, PXG_ARTI_PARENTS));

    for (uint linkID = 1u; linkID < numLinks; ++linkID)
    {
        const uint64_t blk = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;
        const uint64_t lrec = links + uint64_t(linkID) * PXG_ARTI_LINK_SIZE;
        const uint64_t jc = joints + uint64_t(linkID) * PXG_ARTI_JOINT_CORE_SIZE;
        const uint64_t jd = jointData + uint64_t(linkID) * PXG_ARTI_JOINT_CORE_DATA_SIZE;

        // relativeQuats is a PxQuat array the CUDA reinterprets as float4 -- a
        // straight 16-byte copy, not a transform block
        const uint64_t rq = relativeQuats + uint64_t(linkID) * PXG_ARTI_QUAT_SIZE;
        abStoreF4(blk, PXG_AB_LINK_mRelativeQuat, lane,
                  float4(asfloat(artiU32Field(rq, 0u)), asfloat(artiU32Field(rq, 4u)),
                         asfloat(artiU32Field(rq, 8u)), asfloat(artiU32Field(rq, 12u))));

        abStoreU32(blk, PXG_AB_LINK_mJointOffset, lane,
                   artiU32Field(jd, PXG_AR_JCD_jointOffset));
        // nbDof is a PxU8 at offset 4 of an 8-byte struct
        abStoreU8(blk, PXG_AB_LINK_mDofs, lane, artiU8Field(jd, PXG_AR_JCD_nbDof));
        abStoreU8(blk, PXG_AB_LINK_mJointType, lane, artiU8Field(jc, PXG_AR_JOINT_jointType));

        abStoreF4(blk, PXG_AB_LINK_mLinDampingX_AngDampingY_maxLinVelSqZ_maxAngVelSqW, lane,
                  float4(asfloat(artiU32Field(lrec, PXG_AR_LINK_linearDamping)),
                         asfloat(artiU32Field(lrec, PXG_AR_LINK_angularDamping)),
                         asfloat(artiU32Field(lrec, PXG_AR_LINK_maxLinearVelocitySq)),
                         asfloat(artiU32Field(lrec, PXG_AR_LINK_maxAngularVelocitySq))));

        abStoreU32(blk, PXG_AB_LINK_mParents, lane, parents[linkID]);

        abStoreU32(blk, PXG_AB_LINK_mChildrenOffset, lane,
                   artiU32Field(lrec, PXG_AR_LINK_childrenOffset));
        abStoreU32(blk, PXG_AB_LINK_mNumChildren, lane,
                   artiU16Field(lrec, PXG_AR_LINK_numChildren));

        abStoreU8(blk, PXG_AB_LINK_mDisableGravity, lane,
                  artiU8Field(lrec, PXG_AR_LINK_disableGravity));
        abStoreU8(blk, PXG_AB_LINK_mRetainsAcceleration, lane,
                  artiU8Field(lrec, PXG_AR_LINK_retainsAccelerations));

        // the joint's own frames, then the link's world pose
        jcalcStorePose(blk, PXG_AB_LINK_mChildPose, lane, jc + PXG_AR_JOINT_childPose);
        jcalcStorePose(blk, PXG_AB_LINK_mParentPose, lane, jc + PXG_AR_JOINT_parentPose);
        jcalcStorePose(blk, PXG_AB_LINK_mAccumulatedPose, lane,
                       body2Worlds + uint64_t(linkID) * PXG_ARTI_TRANSFORM_SIZE);
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void artiJcalcPhaseB2(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gJcalcB2.nbArticulations)
        return;

    jcalcFillLinkBlocks(gJcalcB2.articulations + uint64_t(lane) * PXG_ARTI_STRIDE,
                        gJcalcB2.linkBlocks, lane);
}


// ===========================================================================
// P8.75: jcalc PHASE B3 -- the per-dof fill (forwardDynamic2.cu:505-544)
// ===========================================================================
//
// The innermost loop of phase B: for every dof of every link, one
// PxgArticulationBlockDofData and the 7296-byte
// PxgArticulationInternalConstraintData embedded in it at +5440.  Twenty-four
// writes per dof, and they are the parameters the internal-constraint solver
// reads back every substep, so this is the tranche that decides whether the
// solver sees the joint the user configured.
//
// TWO INDICES THAT ARE NOT THE SAME AND ARE EASY TO CONFUSE:
//
//   i      -- the dof's position within THIS joint, 0..nbDofs-1
//   dofId  -- joint.dofIds[i], the PxArticulationAxis the dof drives, 0..5
//
// The dof BLOCK is addressed by (jointOffset + i); every per-axis parameter --
// drives, limits, armature, friction, maxJointVelocity, motion -- is addressed by
// dofId.  And `mInvDofIds[dofId] = i` is the inverse map written back into the
// LINK block, which is the one write in this loop that does not go to the dof
// block at all.  Using i where dofId belongs is silent: for a revolute joint
// with dofIds[0] = eTWIST = 0 the two coincide, and only a spherical or a joint
// on a non-first axis tells them apart.  The fixture is built so they never
// coincide.
//
// The values are copied as raw 32-bit words rather than through floats.  Nothing
// here computes -- it moves — and a float round-trip would invite the compiler to
// canonicalise a NaN or flush a denormal that the engine kept.

struct JcalcB3Params
{
    uint64_t articulations;
    uint64_t linkBlocks;
    uint64_t dofBlocks;
    uint nbArticulations;
    uint pad0;
};

[[vk::push_constant]]
ConstantBuffer<JcalcB3Params> gJcalcB3;

void jcalcFillDofBlocks(uint64_t a, uint64_t linkBlocks, uint64_t dofBlocks, uint lane)
{
    const uint numLinks = artiU16Field(a, PXG_ARTI_DATA_NUM_LINKS);

    const uint64_t joints = artiPtrField(a, PXG_ARTI_JOINTS);
    const uint64_t jointData = artiPtrField(a, PXG_ARTI_JOINT_DATA);
    const uint64_t jointAxis = artiPtrField(a, PXG_ARTI_JOINT_AXIS);
    const uint64_t motionMatrix = artiPtrField(a, PXG_ARTI_MOTION_MATRIX);
    const Ptr<uint> jointPositions = Ptr<uint>(artiPtrField(a, PXG_ARTI_JOINT_POSITIONS));
    const Ptr<uint> jointVelocities = Ptr<uint>(artiPtrField(a, PXG_ARTI_JOINT_VELOCITIES));
    const Ptr<uint> jointTargetPos = Ptr<uint>(artiPtrField(a, PXG_ARTI_JOINT_TARGET_POSITIONS));
    const Ptr<uint> jointTargetVel = Ptr<uint>(artiPtrField(a, PXG_ARTI_JOINT_TARGET_VELOCITIES));

    for (uint linkID = 1u; linkID < numLinks; ++linkID)
    {
        const uint64_t blk = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;
        const uint64_t jc = joints + uint64_t(linkID) * PXG_ARTI_JOINT_CORE_SIZE;
        const uint64_t jd = jointData + uint64_t(linkID) * PXG_ARTI_JOINT_CORE_DATA_SIZE;

        const uint nbDofs = artiU8Field(jd, PXG_AR_JCD_nbDof);
        const uint jointOffset = artiU32Field(jd, PXG_AR_JCD_jointOffset);
        const uint64_t mmBase = motionMatrix + uint64_t(linkID) * PXG_ARTI_SUBSPACE_STRIDE;

        for (uint i = 0u; i < nbDofs; ++i)
        {
            const uint64_t dof = dofBlocks + uint64_t(jointOffset + i) * PXG_AB_DOF_SIZE;
            const uint64_t ic = dof + PXG_AB_DOF_mConstraintData;

            // the joint axis phase A computed, in block form
            const uint64_t ax =
                jointAxis + uint64_t(jointOffset + i) * PXG_ARTI_SPATIAL_VECTOR_SIZE;
            abStoreSpatialVector(dof, PXG_AB_DOF_mJointAxis, lane,
                                 float3(asfloat(artiU32Field(ax, 0u)),
                                        asfloat(artiU32Field(ax, 4u)),
                                        asfloat(artiU32Field(ax, 8u))),
                                 float3(asfloat(artiU32Field(ax, 12u)),
                                        asfloat(artiU32Field(ax, 16u)),
                                        asfloat(artiU32Field(ax, 20u))));

            // dofIds is a PxU8[6]; the axis it names indexes every per-axis array
            const uint dofId = artiU8Field(jc, PXG_AR_JOINT_dofIds + i);
            abStoreU8(dof, PXG_AB_DOF_mDofIds, lane, dofId);
            // the inverse map, and the only write here that lands in the LINK block
            abStoreU8(blk, PXG_AB_LINK_mInvDofIds + dofId * 32u, lane, i);

            const uint64_t dr = jc + PXG_AR_JOINT_drives + uint64_t(dofId) * PXG_AR_DRIVE_SIZE;
            const uint64_t env = dr + PXG_AR_DRIVE_envelope;
            const uint64_t lim = jc + PXG_AR_JOINT_limits + uint64_t(dofId) * PXG_AR_LIMIT_SIZE;
            const uint64_t fp = jc + PXG_AR_JOINT_frictionParams
                              + uint64_t(dofId) * PXG_AR_FRICTION_SIZE;

            abStoreU32(ic, PXG_AB_IC_mDriveStiffness, lane,
                       artiU32Field(dr, PXG_AR_DRIVE_stiffness));
            abStoreU32(ic, PXG_AB_IC_mDamping, lane, artiU32Field(dr, PXG_AR_DRIVE_damping));
            abStoreU32(ic, PXG_AB_IC_mDriveTargetVel, lane, jointTargetVel[jointOffset + i]);
            // written as a literal zero by the CUDA, not copied from anywhere
            abStoreF32(ic, PXG_AB_IC_mTargetPosBias, lane, 0.0f);
            abStoreU32(ic, PXG_AB_IC_mArmature, lane,
                       artiU32Field(jc, PXG_AR_JOINT_armature + dofId * 4u));
            abStoreU32(ic, PXG_AB_IC_mDriveTargetPos, lane, jointTargetPos[jointOffset + i]);
            // a float2[32] field: element stride 8
            abStoreF2(ic, PXG_AB_IC_mLimits_LowLimitX_highLimitY, lane,
                      float2(asfloat(artiU32Field(lim, PXG_AR_LIMIT_low)),
                             asfloat(artiU32Field(lim, PXG_AR_LIMIT_high))));

            // the old scalar friction coefficient is per JOINT, not per axis
            abStoreU32(ic, PXG_AB_IC_mFrictionCoefficient, lane,
                       artiU32Field(jc, PXG_AR_JOINT_frictionCoefficient));
            abStoreU32(ic, PXG_AB_IC_mStaticFrictionEffort, lane,
                       artiU32Field(fp, PXG_AR_FRICTION_staticFrictionEffort));
            abStoreU32(ic, PXG_AB_IC_mDynamicFrictionEffort, lane,
                       artiU32Field(fp, PXG_AR_FRICTION_dynamicFrictionEffort));
            abStoreU32(ic, PXG_AB_IC_mViscousFrictionCoefficient, lane,
                       artiU32Field(fp, PXG_AR_FRICTION_viscousFrictionCoefficient));

            abStoreU32(ic, PXG_AB_IC_mMaxForce, lane, artiU32Field(dr, PXG_AR_DRIVE_maxForce));
            abStoreU32(ic, PXG_AB_IC_mMaxEffort, lane, artiU32Field(env, PXG_AR_ENV_maxEffort));
            abStoreU32(ic, PXG_AB_IC_mMaxActuatorVelocity, lane,
                       artiU32Field(env, PXG_AR_ENV_maxActuatorVelocity));
            abStoreU32(ic, PXG_AB_IC_mVelocityDependentResistance, lane,
                       artiU32Field(env, PXG_AR_ENV_velocityDependentResistance));
            abStoreU32(ic, PXG_AB_IC_mSpeedEffortGradient, lane,
                       artiU32Field(env, PXG_AR_ENV_speedEffortGradient));
            abStoreU32(ic, PXG_AB_IC_mDriveType, lane, artiU32Field(dr, PXG_AR_DRIVE_driveType));
            abStoreU32(ic, PXG_AB_IC_mMaxJointVelocity, lane,
                       artiU32Field(jc, PXG_AR_JOINT_maxJointVelocity + dofId * 4u));

            abStoreU8(dof, PXG_AB_DOF_mMotion, lane,
                      artiU8Field(jc, PXG_AR_JOINT_motion + dofId));
            abStoreU32(dof, PXG_AB_DOF_mJointPositions, lane, jointPositions[jointOffset + i]);
            abStoreU32(dof, PXG_AB_DOF_mJointVelocities, lane, jointVelocities[jointOffset + i]);

            // motionMatrix[linkID].columns[i], the local motion matrix column
            const uint64_t mm = mmBase + uint64_t(i) * PXG_ARTI_SPATIAL_VECTOR_SIZE;
            abStoreSpatialVector(dof, PXG_AB_DOF_mLocalMotionMatrix, lane,
                                 float3(asfloat(artiU32Field(mm, 0u)),
                                        asfloat(artiU32Field(mm, 4u)),
                                        asfloat(artiU32Field(mm, 8u))),
                                 float3(asfloat(artiU32Field(mm, 12u)),
                                        asfloat(artiU32Field(mm, 16u)),
                                        asfloat(artiU32Field(mm, 20u))));
        }
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void artiJcalcPhaseB3(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gJcalcB3.nbArticulations)
        return;

    jcalcFillDofBlocks(gJcalcB3.articulations + uint64_t(lane) * PXG_ARTI_STRIDE,
                       gJcalcB3.linkBlocks, gJcalcB3.dofBlocks, lane);
}


// ===========================================================================
// P8.76: jcalc PHASE C -- the `else if (dataDirty)` refresh (fwdDynamic2.cu:586-736)
// ===========================================================================
//
// The path taken when the block layout is already valid and only VALUES have
// changed.  It is emphatically NOT a cheaper phase B: it writes a strict SUBSET
// of the same fields, and which fields it leaves alone is the whole content of
// this tranche.  Phase C does NOT touch
//
//     mRetainsAcceleration, mDofs, mJointType, mJointOffset, mParents,
//     mChildrenOffset, mNumChildren, mInvDofIds, mDofIds, mMotion, mJointAxis
//
// even though phase B writes every one of them, because those are structural and
// the structure has not changed.  Writing one of them here would be invisible in
// any test that only checks the fields that ARE written -- so the fixture poisons
// the destinations and requires every untouched field to come back poison.
//
// TWO BRANCHES THAT ARE EASY TO DROP:
//
//  * The tendon and mimic-joint dirty bits are masked OUT before asking whether
//    anything is left to do.  An articulation whose ONLY dirty bit is
//    eDIRTY_SPATIAL_TENDON must skip the whole block refresh -- the tendon blocks
//    are updated by their own helpers.  Forgetting the mask turns that into a full
//    refresh, which is not wrong numerically and is wrong in cost, every frame.
//  * `directAPI` suppresses the root motion-velocity write ONLY.  Under the direct
//    GPU state API the root velocity in the block is authoritative and the record's
//    initial velocity is stale; copying it would silently undo a user write.
//
// `updateDirty` is cleared here -- this is the consumer phase A deliberately left
// it for.
//
// The CUDA hand-preloads the next link's data into `next*` locals.  That is a
// latency transform with identical results, EXCEPT that it reads gJoints[1],
// links[1] and gJointData[1] before the loop guard, so a single-link articulation
// touches element [1] of a one-element array.  The port reads in the loop instead:
// same values, no out-of-bounds read.  Likewise `dofId = i ? joint.dofIds[i] :
// dofId0` is just `joint.dofIds[i]` with the first one preloaded.

struct JcalcCParams
{
    uint64_t articulations;
    uint64_t linkBlocks;
    uint64_t dofBlocks;
    uint nbArticulations;
    uint directAPI;
};

[[vk::push_constant]]
ConstantBuffer<JcalcCParams> gJcalcC;

void jcalcRefreshBlocks(uint64_t a, uint64_t linkBlocks, uint64_t dofBlocks,
                        bool directAPI, uint lane)
{
    const uint dirtyAll = artiU32Field(a, PXG_ARTI_DATA_UPDATE_DIRTY);

    const uint TENDON_BITS = PXG_ARTI_DIRTY_SPATIAL_TENDON
                           | PXG_ARTI_DIRTY_SPATIAL_TENDON_ATTACHMENT
                           | PXG_ARTI_DIRTY_FIXED_TENDON
                           | PXG_ARTI_DIRTY_FIXED_TENDON_JOINT;
    const uint updateDirty = dirtyAll & ~TENDON_BITS;

    if (updateDirty != 0u)
    {
        const uint numLinks = artiU16Field(a, PXG_ARTI_DATA_NUM_LINKS);

        const uint64_t links = artiPtrField(a, PXG_ARTI_LINKS);
        const uint64_t joints = artiPtrField(a, PXG_ARTI_JOINTS);
        const uint64_t jointData = artiPtrField(a, PXG_ARTI_JOINT_DATA);
        const uint64_t body2Worlds = artiPtrField(a, PXG_ARTI_LINK_BODY2WORLDS);
        const uint64_t motionMatrix = artiPtrField(a, PXG_ARTI_MOTION_MATRIX);
        const uint64_t relativeQuats = artiPtrField(a, PXG_ARTI_RELATIVE_QUAT);
        const Ptr<uint> jointPositions = Ptr<uint>(artiPtrField(a, PXG_ARTI_JOINT_POSITIONS));
        const Ptr<uint> jointVelocities = Ptr<uint>(artiPtrField(a, PXG_ARTI_JOINT_VELOCITIES));
        const Ptr<uint> jointTargetPos = Ptr<uint>(artiPtrField(a, PXG_ARTI_JOINT_TARGET_POSITIONS));
        const Ptr<uint> jointTargetVel = Ptr<uint>(artiPtrField(a, PXG_ARTI_JOINT_TARGET_VELOCITIES));

        // ---- the root: three fields, and the velocity only when not direct-API ----
        const uint64_t root = linkBlocks;
        abStoreU8(root, PXG_AB_LINK_mDisableGravity, lane,
                  artiU8Field(links, PXG_AR_LINK_disableGravity));
        abStoreF4(root, PXG_AB_LINK_mLinDampingX_AngDampingY_maxLinVelSqZ_maxAngVelSqW, lane,
                  float4(asfloat(artiU32Field(links, PXG_AR_LINK_linearDamping)),
                         asfloat(artiU32Field(links, PXG_AR_LINK_angularDamping)),
                         asfloat(artiU32Field(links, PXG_AR_LINK_maxLinearVelocitySq)),
                         asfloat(artiU32Field(links, PXG_AR_LINK_maxAngularVelocitySq))));

        if (!directAPI)
        {
            const float3 initAng =
                float3(asfloat(artiU32Field(links, PXG_AR_LINK_initialAngVel + 0u)),
                       asfloat(artiU32Field(links, PXG_AR_LINK_initialAngVel + 4u)),
                       asfloat(artiU32Field(links, PXG_AR_LINK_initialAngVel + 8u)));
            const float3 initLin =
                float3(asfloat(artiU32Field(links, PXG_AR_LINK_initialLinVel + 0u)),
                       asfloat(artiU32Field(links, PXG_AR_LINK_initialLinVel + 4u)),
                       asfloat(artiU32Field(links, PXG_AR_LINK_initialLinVel + 8u)));
            abStoreSpatialVector(root, PXG_AB_LINK_mMotionVelocity, lane, initAng, initLin);
        }

        jcalcStorePose(root, PXG_AB_LINK_mAccumulatedPose, lane, body2Worlds);

        for (uint linkID = 1u; linkID < numLinks; ++linkID)
        {
            const uint64_t blk = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;
            const uint64_t lrec = links + uint64_t(linkID) * PXG_ARTI_LINK_SIZE;
            const uint64_t jc = joints + uint64_t(linkID) * PXG_ARTI_JOINT_CORE_SIZE;
            const uint64_t jd = jointData + uint64_t(linkID) * PXG_ARTI_JOINT_CORE_DATA_SIZE;

            const uint nbDofs = artiU8Field(jd, PXG_AR_JCD_nbDof);
            const uint jointOffset = artiU32Field(jd, PXG_AR_JCD_jointOffset);
            const uint64_t mmBase = motionMatrix + uint64_t(linkID) * PXG_ARTI_SUBSPACE_STRIDE;
            const uint frictionCoefficient = artiU32Field(jc, PXG_AR_JOINT_frictionCoefficient);

            for (uint i = 0u; i < nbDofs; ++i)
            {
                const uint64_t dof = dofBlocks + uint64_t(jointOffset + i) * PXG_AB_DOF_SIZE;
                const uint64_t ic = dof + PXG_AB_DOF_mConstraintData;
                const uint dofId = artiU8Field(jc, PXG_AR_JOINT_dofIds + i);

                const uint64_t dr = jc + PXG_AR_JOINT_drives + uint64_t(dofId) * PXG_AR_DRIVE_SIZE;
                const uint64_t env = dr + PXG_AR_DRIVE_envelope;
                const uint64_t lim = jc + PXG_AR_JOINT_limits + uint64_t(dofId) * PXG_AR_LIMIT_SIZE;
                const uint64_t fp = jc + PXG_AR_JOINT_frictionParams
                                  + uint64_t(dofId) * PXG_AR_FRICTION_SIZE;

                abStoreU32(ic, PXG_AB_IC_mDriveStiffness, lane,
                           artiU32Field(dr, PXG_AR_DRIVE_stiffness));
                abStoreU32(ic, PXG_AB_IC_mDamping, lane, artiU32Field(dr, PXG_AR_DRIVE_damping));
                abStoreU32(ic, PXG_AB_IC_mDriveTargetVel, lane, jointTargetVel[jointOffset + i]);
                abStoreF32(ic, PXG_AB_IC_mTargetPosBias, lane, 0.0f);
                abStoreU32(ic, PXG_AB_IC_mArmature, lane,
                           artiU32Field(jc, PXG_AR_JOINT_armature + dofId * 4u));
                abStoreF2(ic, PXG_AB_IC_mLimits_LowLimitX_highLimitY, lane,
                          float2(asfloat(artiU32Field(lim, PXG_AR_LIMIT_low)),
                                 asfloat(artiU32Field(lim, PXG_AR_LIMIT_high))));
                abStoreU32(ic, PXG_AB_IC_mFrictionCoefficient, lane, frictionCoefficient);
                abStoreU32(ic, PXG_AB_IC_mStaticFrictionEffort, lane,
                           artiU32Field(fp, PXG_AR_FRICTION_staticFrictionEffort));
                abStoreU32(ic, PXG_AB_IC_mDynamicFrictionEffort, lane,
                           artiU32Field(fp, PXG_AR_FRICTION_dynamicFrictionEffort));
                abStoreU32(ic, PXG_AB_IC_mViscousFrictionCoefficient, lane,
                           artiU32Field(fp, PXG_AR_FRICTION_viscousFrictionCoefficient));
                abStoreU32(ic, PXG_AB_IC_mMaxForce, lane,
                           artiU32Field(dr, PXG_AR_DRIVE_maxForce));
                abStoreU32(ic, PXG_AB_IC_mMaxEffort, lane,
                           artiU32Field(env, PXG_AR_ENV_maxEffort));
                abStoreU32(ic, PXG_AB_IC_mMaxActuatorVelocity, lane,
                           artiU32Field(env, PXG_AR_ENV_maxActuatorVelocity));
                abStoreU32(ic, PXG_AB_IC_mVelocityDependentResistance, lane,
                           artiU32Field(env, PXG_AR_ENV_velocityDependentResistance));
                abStoreU32(ic, PXG_AB_IC_mSpeedEffortGradient, lane,
                           artiU32Field(env, PXG_AR_ENV_speedEffortGradient));
                abStoreU32(ic, PXG_AB_IC_mDriveType, lane,
                           artiU32Field(dr, PXG_AR_DRIVE_driveType));
                abStoreU32(ic, PXG_AB_IC_mMaxJointVelocity, lane,
                           artiU32Field(jc, PXG_AR_JOINT_maxJointVelocity + dofId * 4u));

                // these three are written regardless of directAPI -- the CUDA comment
                // above them says so explicitly and flags it as an open question
                abStoreU32(dof, PXG_AB_DOF_mJointPositions, lane,
                           jointPositions[jointOffset + i]);
                abStoreU32(dof, PXG_AB_DOF_mJointVelocities, lane,
                           jointVelocities[jointOffset + i]);
                abStoreU32(ic, PXG_AB_IC_mDriveTargetPos, lane, jointTargetPos[jointOffset + i]);

                const uint64_t mm = mmBase + uint64_t(i) * PXG_ARTI_SPATIAL_VECTOR_SIZE;
                abStoreSpatialVector(dof, PXG_AB_DOF_mLocalMotionMatrix, lane,
                                     float3(asfloat(artiU32Field(mm, 0u)),
                                            asfloat(artiU32Field(mm, 4u)),
                                            asfloat(artiU32Field(mm, 8u))),
                                     float3(asfloat(artiU32Field(mm, 12u)),
                                            asfloat(artiU32Field(mm, 16u)),
                                            asfloat(artiU32Field(mm, 20u))));
            }

            const uint64_t rq = relativeQuats + uint64_t(linkID) * PXG_ARTI_QUAT_SIZE;
            abStoreF4(blk, PXG_AB_LINK_mRelativeQuat, lane,
                      float4(asfloat(artiU32Field(rq, 0u)), asfloat(artiU32Field(rq, 4u)),
                             asfloat(artiU32Field(rq, 8u)), asfloat(artiU32Field(rq, 12u))));
            abStoreF4(blk, PXG_AB_LINK_mLinDampingX_AngDampingY_maxLinVelSqZ_maxAngVelSqW, lane,
                      float4(asfloat(artiU32Field(lrec, PXG_AR_LINK_linearDamping)),
                             asfloat(artiU32Field(lrec, PXG_AR_LINK_angularDamping)),
                             asfloat(artiU32Field(lrec, PXG_AR_LINK_maxLinearVelocitySq)),
                             asfloat(artiU32Field(lrec, PXG_AR_LINK_maxAngularVelocitySq))));
            abStoreU8(blk, PXG_AB_LINK_mDisableGravity, lane,
                      artiU8Field(lrec, PXG_AR_LINK_disableGravity));

            jcalcStorePose(blk, PXG_AB_LINK_mChildPose, lane, jc + PXG_AR_JOINT_childPose);
            jcalcStorePose(blk, PXG_AB_LINK_mParentPose, lane, jc + PXG_AR_JOINT_parentPose);
            jcalcStorePose(blk, PXG_AB_LINK_mAccumulatedPose, lane,
                           body2Worlds + uint64_t(linkID) * PXG_ARTI_TRANSFORM_SIZE);
        }
    }

    // the tendon and mimic-joint refreshes hang off the bits masked out above and
    // are separate subsystems; they are not part of this tranche.

    // updateDirty is cleared here, and only here -- phase A left it set on purpose
    (Ptr<uint>(a + PXG_ARTI_DATA_UPDATE_DIRTY))[0] = 0u;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void artiJcalcPhaseC(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gJcalcC.nbArticulations)
        return;

    jcalcRefreshBlocks(gJcalcC.articulations + uint64_t(lane) * PXG_ARTI_STRIDE,
                       gJcalcC.linkBlocks, gJcalcC.dofBlocks, gJcalcC.directAPI != 0u, lane);
}


// ===========================================================================
// P8.78: the TENDON and MIMIC-JOINT block helpers (forwardDynamic2.cu:106-350)
// ===========================================================================
//
// Six functions in the CUDA, three here: `initialize*Block` and `update*Block`
// differ by EXACTLY the structural fields, so each pair is one copy function
// plus an `initialize` guard -- which is how the CUDA itself writes the mimic
// pair, sharing `copyMimicJointsBlock`.
//
// What only the initialize side writes:
//
//   spatial tendon  mNumAttachments, and per attachment mParents / mChildrens /
//                   mLinkIndex; plus artiBlock.mNumSpatialTendons
//   fixed tendon    mNumTendonJoints, and per tendon joint mParents /
//                   mChildrens / mLinkIndex; plus artiBlock.mNumFixedTendons
//   mimic joint     nothing per joint -- only artiBlock.mNumMimicJoints
//
// The topology fields are structural: they come from `mFixedElements`, which the
// name says is fixed at initialisation, while the update path re-reads only
// `mModElements`, the run-time-modifiable half.  An update that also wrote the
// topology would be correct today and wrong the moment a tendon is re-pointed
// without a re-init, so the fixture poisons the destinations and requires the
// update path to leave those bytes alone.
//
// TWO-LEVEL INDEXING, and the inner bound is not the outer one:
//
//     attachmentBlocks[i * maxNumAttachments + j],  j < tendon.mNbElements
//
// `mNbElements` comes from the RECORD; `maxNumAttachments` is the per-warp
// allocation pitch.  With mNbElements == maxNumAttachments the two collapse and
// a wrong pitch is invisible, so the fixture uses mNbElements < maxNumAttachments
// and more than one tendon.

struct JcalcTendonParams
{
    uint64_t articulations;
    uint64_t artiBlocks;           // PxgArticulationBlockData for this warp
    uint64_t spatialTendonBlocks;
    uint64_t attachmentBlocks;
    uint64_t fixedTendonBlocks;
    uint64_t tendonJointBlocks;
    uint64_t mimicJointBlocks;
    uint maxNumAttachments;
    uint maxNumTendonJoints;
    uint nbArticulations;
    uint initialize;
};

[[vk::push_constant]]
ConstantBuffer<JcalcTendonParams> gJcalcTendon;

// The two halves are SEPARATELY GUARDED in the gpuDirtyFlag block (P8.79):
// eDIRTY_SPATIAL_TENDON refreshes only the params, eDIRTY_SPATIAL_TENDON_ATTACHMENT
// only the per-attachment mod fields.  jcalc runs both together.  Splitting them
// here keeps one field list per field rather than a third transcription.
void jcalcCopySpatialTendonParams(uint64_t a, uint64_t tendonBlocks, uint lane,
                                  bool withCounts)
{
    const uint numSpatialTendons = artiU16Field(a, PXG_ARTI_DATA_NUM_SPATIAL_TENDONS);
    const uint64_t params = artiPtrField(a, PXG_ARTI_SPATIAL_TENDON_PARAMS);
    const uint64_t tendons = artiPtrField(a, PXG_ARTI_SPATIAL_TENDONS);

    for (uint i = 0u; i < numSpatialTendons; ++i)
    {
        const uint64_t tp = params + uint64_t(i) * PXG_ARTI_SPATIAL_TENDON_DATA_SIZE;
        const uint64_t tb = tendonBlocks + uint64_t(i) * PXG_AB_ST_SIZE;

        if (withCounts)
        {
            const uint64_t tendon = tendons + uint64_t(i) * PXG_ARTI_TENDON_SIZE;
            abStoreU32(tb, PXG_AB_ST_mNumAttachments, lane,
                       artiU32Field(tendon, PXG_TENDON_NB_ELEMENTS));
        }

        abStoreU32(tb, PXG_AB_ST_mStiffness, lane, artiU32Field(tp, PXG_AR_STD_stiffness));
        abStoreU32(tb, PXG_AB_ST_mLimitStiffness, lane,
                   artiU32Field(tp, PXG_AR_STD_limitStiffness));
        abStoreU32(tb, PXG_AB_ST_mDamping, lane, artiU32Field(tp, PXG_AR_STD_damping));
        abStoreU32(tb, PXG_AB_ST_mOffset, lane, artiU32Field(tp, PXG_AR_STD_offset));
    }
}

void jcalcCopySpatialTendonAttachments(uint64_t a, uint64_t attachmentBlocks,
                                       uint maxNumAttachments, uint lane, bool withTopology)
{
    const uint numSpatialTendons = artiU16Field(a, PXG_ARTI_DATA_NUM_SPATIAL_TENDONS);
    const uint64_t tendons = artiPtrField(a, PXG_ARTI_SPATIAL_TENDONS);

    for (uint i = 0u; i < numSpatialTendons; ++i)
    {
        const uint64_t tendon = tendons + uint64_t(i) * PXG_ARTI_TENDON_SIZE;
        const uint nbElements = artiU32Field(tendon, PXG_TENDON_NB_ELEMENTS);

        const uint64_t ab = attachmentBlocks + uint64_t(i * maxNumAttachments) * PXG_AB_AT_SIZE;
        const uint64_t fixedData = artiPtrField(tendon, PXG_TENDON_FIXED_ELEMENTS);
        const uint64_t modData = artiPtrField(tendon, PXG_TENDON_MOD_ELEMENTS);

        for (uint j = 0u; j < nbElements; ++j)
        {
            const uint64_t slot = ab + uint64_t(j) * PXG_AB_AT_SIZE;
            const uint64_t md = modData + uint64_t(j) * PXG_ARTI_ATTACHMENT_MOD_SIZE;

            if (withTopology)
            {
                const uint64_t fd = fixedData + uint64_t(j) * PXG_ARTI_TENDON_FIXED_ELEM_SIZE;
                abStoreU32(slot, PXG_AB_AT_mParents, lane, artiU32Field(fd, PXG_AR_TEF_parent));
                abStoreU64(slot, PXG_AB_AT_mChildrens, lane,
                           (Ptr<uint64_t>(fd + PXG_AR_TEF_children))[0]);
                // linkInd is a PxU32 in the record and a PxU16 in the block
                abStoreU16(slot, PXG_AB_AT_mLinkIndex, lane,
                           artiU32Field(fd, PXG_AR_TEF_linkInd));
            }

            abStoreF3(slot, PXG_AB_AT_mRelativeOffset, lane,
                      float3(asfloat(artiU32Field(md, PXG_AR_TAD_relativeOffset + 0u)),
                             asfloat(artiU32Field(md, PXG_AR_TAD_relativeOffset + 4u)),
                             asfloat(artiU32Field(md, PXG_AR_TAD_relativeOffset + 8u))));
            abStoreU32(slot, PXG_AB_AT_mRestDistance, lane,
                       artiU32Field(md, PXG_AR_TAD_restLength));
            abStoreU32(slot, PXG_AB_AT_mCoefficient, lane,
                       artiU32Field(md, PXG_AR_TAD_coefficient));
            abStoreU32(slot, PXG_AB_AT_mLowLimit, lane, artiU32Field(md, PXG_AR_TAD_lowLimit));
            abStoreU32(slot, PXG_AB_AT_mHighLimit, lane, artiU32Field(md, PXG_AR_TAD_highLimit));
        }
    }
}

void jcalcCopySpatialTendonsBlock(uint64_t a, uint64_t artiBlock, uint64_t tendonBlocks,
                                  uint64_t attachmentBlocks, uint maxNumAttachments,
                                  uint lane, bool initialize)
{
    jcalcCopySpatialTendonParams(a, tendonBlocks, lane, initialize);
    jcalcCopySpatialTendonAttachments(a, attachmentBlocks, maxNumAttachments, lane, initialize);

    if (initialize)
        abStoreU32(artiBlock, PXG_AB_mNumSpatialTendons, lane,
                   artiU16Field(a, PXG_ARTI_DATA_NUM_SPATIAL_TENDONS));
}

void jcalcCopyFixedTendonParams(uint64_t a, uint64_t tendonBlocks, uint lane,
                                bool withCounts)
{
    const uint numFixedTendons = artiU16Field(a, PXG_ARTI_DATA_NUM_FIXED_TENDONS);
    const uint64_t params = artiPtrField(a, PXG_ARTI_FIXED_TENDON_PARAMS);
    const uint64_t tendons = artiPtrField(a, PXG_ARTI_FIXED_TENDONS);

    for (uint i = 0u; i < numFixedTendons; ++i)
    {
        const uint64_t tp = params + uint64_t(i) * PXG_ARTI_FIXED_TENDON_DATA_SIZE;
        const uint64_t tb = tendonBlocks + uint64_t(i) * PXG_AB_FT_SIZE;

        if (withCounts)
        {
            const uint64_t tendon = tendons + uint64_t(i) * PXG_ARTI_TENDON_SIZE;
            abStoreU32(tb, PXG_AB_FT_mNumTendonJoints, lane,
                       artiU32Field(tendon, PXG_TENDON_NB_ELEMENTS));
        }

        abStoreU32(tb, PXG_AB_FT_mStiffness, lane, artiU32Field(tp, PXG_AR_FTD_stiffness));
        abStoreU32(tb, PXG_AB_FT_mDamping, lane, artiU32Field(tp, PXG_AR_FTD_damping));
        abStoreU32(tb, PXG_AB_FT_mLimitStiffness, lane,
                   artiU32Field(tp, PXG_AR_FTD_limitStiffness));
        abStoreU32(tb, PXG_AB_FT_mOffset, lane, artiU32Field(tp, PXG_AR_FTD_offset));
        abStoreU32(tb, PXG_AB_FT_mLowLimit, lane, artiU32Field(tp, PXG_AR_FTD_lowLimit));
        abStoreU32(tb, PXG_AB_FT_mHighLimit, lane, artiU32Field(tp, PXG_AR_FTD_highLimit));
        abStoreU32(tb, PXG_AB_FT_mRestLength, lane, artiU32Field(tp, PXG_AR_FTD_restLength));
    }
}

void jcalcCopyFixedTendonJoints(uint64_t a, uint64_t tendonJointBlocks,
                                uint maxNumTendonJoints, uint lane, bool withTopology)
{
    const uint numFixedTendons = artiU16Field(a, PXG_ARTI_DATA_NUM_FIXED_TENDONS);
    const uint64_t tendons = artiPtrField(a, PXG_ARTI_FIXED_TENDONS);

    for (uint i = 0u; i < numFixedTendons; ++i)
    {
        const uint64_t tendon = tendons + uint64_t(i) * PXG_ARTI_TENDON_SIZE;
        const uint nbElements = artiU32Field(tendon, PXG_TENDON_NB_ELEMENTS);

        const uint64_t jb = tendonJointBlocks + uint64_t(i * maxNumTendonJoints) * PXG_AB_TJ_SIZE;
        const uint64_t fixedData = artiPtrField(tendon, PXG_TENDON_FIXED_ELEMENTS);
        const uint64_t modData = artiPtrField(tendon, PXG_TENDON_MOD_ELEMENTS);

        for (uint j = 0u; j < nbElements; ++j)
        {
            const uint64_t slot = jb + uint64_t(j) * PXG_AB_TJ_SIZE;
            const uint64_t cd = modData + uint64_t(j) * PXG_ARTI_TENDON_JOINT_COEFF_SIZE;

            if (withTopology)
            {
                const uint64_t fd = fixedData + uint64_t(j) * PXG_ARTI_TENDON_FIXED_ELEM_SIZE;
                abStoreU32(slot, PXG_AB_TJ_mParents, lane, artiU32Field(fd, PXG_AR_TEF_parent));
                abStoreU64(slot, PXG_AB_TJ_mChildrens, lane,
                           (Ptr<uint64_t>(fd + PXG_AR_TEF_children))[0]);
                abStoreU16(slot, PXG_AB_TJ_mLinkIndex, lane,
                           artiU32Field(fd, PXG_AR_TEF_linkInd));
            }

            abStoreU32(slot, PXG_AB_TJ_mCoefficient, lane,
                       artiU32Field(cd, PXG_AR_TJC_coefficient));
            abStoreU32(slot, PXG_AB_TJ_mRecipCoefficient, lane,
                       artiU32Field(cd, PXG_AR_TJC_recipCoefficient));
            abStoreU32(slot, PXG_AB_TJ_mAxis, lane, artiU32Field(cd, PXG_AR_TJC_axis));
        }
    }
}

void jcalcCopyFixedTendonsBlock(uint64_t a, uint64_t artiBlock, uint64_t tendonBlocks,
                                uint64_t tendonJointBlocks, uint maxNumTendonJoints,
                                uint lane, bool initialize)
{
    jcalcCopyFixedTendonParams(a, tendonBlocks, lane, initialize);
    jcalcCopyFixedTendonJoints(a, tendonJointBlocks, maxNumTendonJoints, lane, initialize);

    if (initialize)
        abStoreU32(artiBlock, PXG_AB_mNumFixedTendons, lane,
                   artiU16Field(a, PXG_ARTI_DATA_NUM_FIXED_TENDONS));
}

void jcalcCopyMimicJointsBlock(uint64_t a, uint64_t artiBlock, uint64_t mimicJointBlocks,
                               uint lane, bool initialize)
{
    const uint numMimicJoints = artiU16Field(a, PXG_ARTI_DATA_NUM_MIMIC_JOINTS);
    const uint64_t cores = artiPtrField(a, PXG_ARTI_MIMIC_JOINT_CORES);

    for (uint i = 0u; i < numMimicJoints; ++i)
    {
        const uint64_t c = cores + uint64_t(i) * PXG_ARTI_MIMIC_JOINT_CORE_SIZE;
        const uint64_t mb = mimicJointBlocks + uint64_t(i) * PXG_AB_MJ_SIZE;

        // the record is laid out linkA, axisA, linkB, axisB -- NOT in the order
        // the CUDA copy lists them (P8.77)
        abStoreU32(mb, PXG_AB_MJ_mLinkA, lane, artiU32Field(c, PXG_AR_MJC_linkA));
        abStoreU32(mb, PXG_AB_MJ_mLinkB, lane, artiU32Field(c, PXG_AR_MJC_linkB));
        abStoreU32(mb, PXG_AB_MJ_mAxisA, lane, artiU32Field(c, PXG_AR_MJC_axisA));
        abStoreU32(mb, PXG_AB_MJ_mAxisB, lane, artiU32Field(c, PXG_AR_MJC_axisB));
        abStoreU32(mb, PXG_AB_MJ_mGearRatio, lane, artiU32Field(c, PXG_AR_MJC_gearRatio));
        abStoreU32(mb, PXG_AB_MJ_mOffset, lane, artiU32Field(c, PXG_AR_MJC_offset));
        abStoreU32(mb, PXG_AB_MJ_mNaturalFrequency, lane,
                   artiU32Field(c, PXG_AR_MJC_naturalFrequency));
        abStoreU32(mb, PXG_AB_MJ_mDampingRatio, lane, artiU32Field(c, PXG_AR_MJC_dampingRatio));
    }

    // the ONLY difference between initializeMimicJointsBlock and
    // updateMimicJointsBlock in the CUDA
    if (initialize)
        abStoreU32(artiBlock, PXG_AB_mNumMimicJoints, lane, numMimicJoints);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void artiJcalcTendons(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gJcalcTendon.nbArticulations)
        return;

    const uint64_t a = gJcalcTendon.articulations + uint64_t(lane) * PXG_ARTI_STRIDE;
    const bool init = gJcalcTendon.initialize != 0u;

    jcalcCopySpatialTendonsBlock(a, gJcalcTendon.artiBlocks, gJcalcTendon.spatialTendonBlocks,
                                 gJcalcTendon.attachmentBlocks, gJcalcTendon.maxNumAttachments,
                                 lane, init);
    jcalcCopyFixedTendonsBlock(a, gJcalcTendon.artiBlocks, gJcalcTendon.fixedTendonBlocks,
                               gJcalcTendon.tendonJointBlocks, gJcalcTendon.maxNumTendonJoints,
                               lane, init);
    jcalcCopyMimicJointsBlock(a, gJcalcTendon.artiBlocks, gJcalcTendon.mimicJointBlocks,
                              lane, init);
}


// ===========================================================================
// P8.79: the gpuDirtyFlag block (forwardDynamic2.cu:754-905)
// ===========================================================================
//
// The direct-GPU-API refresh: the host has poked values straight into the
// articulation record and raised bits saying which.  TEN branches, not nine --
// eDIRTY_POSITIONS drives two of them -- and then the flag is cleared.
//
// TWO THINGS THAT LOOK LIKE ONE:
//
//   eDIRTY_POSITIONS                        -> every dof's mJointPositions
//   eDIRTY_ROOT_TRANSFORM | eDIRTY_POSITIONS -> every LINK's mAccumulatedPose
//
// so raising positions alone rewrites the poses too, and a fixture that only
// tests the flags one at a time in the obvious pairing would never separate them.
// That second loop also runs from linkID **0**, unlike the phase B2 and phase C
// link loops which start at 1 because the root has no joint -- here the root's
// pose is exactly what a root-transform poke changes.
//
// The four tendon branches are the two halves of the P8.78 copy functions,
// SEPARATELY guarded: eDIRTY_SPATIAL_TENDON refreshes only the params and
// eDIRTY_SPATIAL_TENDON_ATTACHMENT only the per-attachment mod fields, where
// jcalc always runs both.  That is why those functions were split rather than
// their field lists transcribed a third time -- and the halves are called here
// with the topology/count guard FALSE, because a direct-API poke changes values,
// never structure.

struct JcalcGpuDirtyParams
{
    uint64_t articulations;
    uint64_t linkBlocks;
    uint64_t dofBlocks;
    uint64_t spatialTendonBlocks;
    uint64_t attachmentBlocks;
    uint64_t fixedTendonBlocks;
    uint64_t tendonJointBlocks;
    uint maxNumAttachments;
    uint maxNumTendonJoints;
    uint nbArticulations;
    uint pad0;
};

[[vk::push_constant]]
ConstantBuffer<JcalcGpuDirtyParams> gJcalcGpuDirty;

void jcalcApplyGpuDirty(uint64_t a, uint64_t linkBlocks, uint64_t dofBlocks,
                        uint64_t spatialTendonBlocks, uint64_t attachmentBlocks,
                        uint64_t fixedTendonBlocks, uint64_t tendonJointBlocks,
                        uint maxNumAttachments, uint maxNumTendonJoints, uint lane)
{
    const uint dirty = artiU32Field(a, PXG_ARTI_DATA_gpuDirtyFlag);
    if (dirty == 0u)
        return;

    const uint totalDofs = artiU16Field(a, PXG_ARTI_DATA_NUM_JOINT_DOFS);
    const uint numLinks = artiU16Field(a, PXG_ARTI_DATA_NUM_LINKS);

    if ((dirty & PXG_ARTI_DIRTY_POSITIONS) != 0u)
    {
        const Ptr<uint> jointPositions = Ptr<uint>(artiPtrField(a, PXG_ARTI_JOINT_POSITIONS));
        for (uint i = 0u; i < totalDofs; ++i)
            abStoreU32(dofBlocks + uint64_t(i) * PXG_AB_DOF_SIZE,
                       PXG_AB_DOF_mJointPositions, lane, jointPositions[i]);
    }

    if ((dirty & PXG_ARTI_DIRTY_VELOCITIES) != 0u)
    {
        const Ptr<uint> jointVelocities = Ptr<uint>(artiPtrField(a, PXG_ARTI_JOINT_VELOCITIES));
        for (uint i = 0u; i < totalDofs; ++i)
            abStoreU32(dofBlocks + uint64_t(i) * PXG_AB_DOF_SIZE,
                       PXG_AB_DOF_mJointVelocities, lane, jointVelocities[i]);
    }

    // NOTE the `| POSITIONS`: a joint-position poke moves every link, so the
    // accumulated poses are re-read even when only positions were raised
    if ((dirty & (PXG_ARTI_DIRTY_ROOT_TRANSFORM | PXG_ARTI_DIRTY_POSITIONS)) != 0u)
    {
        const uint64_t body2Worlds = artiPtrField(a, PXG_ARTI_LINK_BODY2WORLDS);
        for (uint linkID = 0u; linkID < numLinks; ++linkID)   // from 0, root included
            jcalcStorePose(linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE,
                           PXG_AB_LINK_mAccumulatedPose, lane,
                           body2Worlds + uint64_t(linkID) * PXG_ARTI_TRANSFORM_SIZE);
    }

    if ((dirty & PXG_ARTI_DIRTY_ROOT_VELOCITIES) != 0u)
    {
        // motionVelocities[0], a Cm::UnAlignedSpatialVector: six packed floats
        const uint64_t mv = artiPtrField(a, PXG_ARTI_MOTION_VELOCITIES);
        abStoreSpatialVector(linkBlocks, PXG_AB_LINK_mMotionVelocity, lane,
                             float3(asfloat(artiU32Field(mv, 0u)),
                                    asfloat(artiU32Field(mv, 4u)),
                                    asfloat(artiU32Field(mv, 8u))),
                             float3(asfloat(artiU32Field(mv, 12u)),
                                    asfloat(artiU32Field(mv, 16u)),
                                    asfloat(artiU32Field(mv, 20u))));
    }

    if ((dirty & PXG_ARTI_DIRTY_JOINT_TARGET_VEL) != 0u)
    {
        const Ptr<uint> tv = Ptr<uint>(artiPtrField(a, PXG_ARTI_JOINT_TARGET_VELOCITIES));
        for (uint i = 0u; i < totalDofs; ++i)
            abStoreU32(dofBlocks + uint64_t(i) * PXG_AB_DOF_SIZE + PXG_AB_DOF_mConstraintData,
                       PXG_AB_IC_mDriveTargetVel, lane, tv[i]);
    }

    if ((dirty & PXG_ARTI_DIRTY_JOINT_TARGET_POS) != 0u)
    {
        const Ptr<uint> tp = Ptr<uint>(artiPtrField(a, PXG_ARTI_JOINT_TARGET_POSITIONS));
        for (uint i = 0u; i < totalDofs; ++i)
            abStoreU32(dofBlocks + uint64_t(i) * PXG_AB_DOF_SIZE + PXG_AB_DOF_mConstraintData,
                       PXG_AB_IC_mDriveTargetPos, lane, tp[i]);
    }

    // the tendon halves, with the structural guard OFF: a direct-API poke changes
    // values, never topology
    if ((dirty & PXG_ARTI_DIRTY_SPATIAL_TENDON) != 0u)
        jcalcCopySpatialTendonParams(a, spatialTendonBlocks, lane, false);

    if ((dirty & PXG_ARTI_DIRTY_SPATIAL_TENDON_ATTACHMENT) != 0u)
        jcalcCopySpatialTendonAttachments(a, attachmentBlocks, maxNumAttachments, lane, false);

    if ((dirty & PXG_ARTI_DIRTY_FIXED_TENDON) != 0u)
        jcalcCopyFixedTendonParams(a, fixedTendonBlocks, lane, false);

    if ((dirty & PXG_ARTI_DIRTY_FIXED_TENDON_JOINT) != 0u)
        jcalcCopyFixedTendonJoints(a, tendonJointBlocks, maxNumTendonJoints, lane, false);

    (Ptr<uint>(a + PXG_ARTI_DATA_gpuDirtyFlag))[0] = 0u;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void artiJcalcGpuDirty(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gJcalcGpuDirty.nbArticulations)
        return;

    jcalcApplyGpuDirty(gJcalcGpuDirty.articulations + uint64_t(lane) * PXG_ARTI_STRIDE,
                       gJcalcGpuDirty.linkBlocks, gJcalcGpuDirty.dofBlocks,
                       gJcalcGpuDirty.spatialTendonBlocks, gJcalcGpuDirty.attachmentBlocks,
                       gJcalcGpuDirty.fixedTendonBlocks, gJcalcGpuDirty.tendonJointBlocks,
                       gJcalcGpuDirty.maxNumAttachments, gJcalcGpuDirty.maxNumTendonJoints,
                       lane);
}


// ===========================================================================
// P8.80: initialize() and the block-header fill
// ===========================================================================
//
// Two functions that bracket jcalc in computeUnconstrainedVelocitiesInternal1T:
// the header fill (:941-957, guarded by `refillBlockData || confiDirty`) runs
// before it, `initialize()` (:62-102) between them.  Both are plain SoA writes --
// but between them they touch FOUR element strides, which is the only thing here
// that can go quietly wrong:
//
//   PxU32[32]/PxReal[32]  4 bytes   the counters, the header words
//   PxU8[32]             1 byte    mStateDirty -- four lanes per word, so the
//                                  atomic and/or pair, not a plain store
//   float4[32]           16 bytes  mDeltaQ
//   pointer[32]          8 bytes   mMotionVelocitiesPtr
//
// THE ONE NON-ZERO WRITE IN A FUNCTION WHOSE JOB IS ZEROING: `mDeltaQ` is set to
// the IDENTITY QUATERNION (0,0,0,1), not to zero.  A memset-shaped port produces a
// zero quaternion, which is not a rotation at all and will not announce itself
// until something rotates by it.
//
// Same oracle caveat as P8.73-P8.79: __device__-only, expectation from the
// fixture, carried by generated offsets, whole-region poison comparison and the
// sweep.

struct JcalcInitParams
{
    uint64_t articulations;
    uint64_t artiBlocks;
    uint64_t linkBlocks;
    uint64_t dofBlocks;
    uint numLinks;
    uint totalDofs;
    uint articulationIndex;
    float sleepThreshold;
    uint nbArticulations;
    uint doHeader;
};

[[vk::push_constant]]
ConstantBuffer<JcalcInitParams> gJcalcInit;

// zero a PxgSpatialVectorBlock lane
void jcalcZeroSpatialVector(uint64_t blk, uint fieldOffset, uint lane)
{
    abStoreSpatialVector(blk, fieldOffset, lane, float3(0.0f, 0.0f, 0.0f),
                         float3(0.0f, 0.0f, 0.0f));
}

void jcalcInitializeBlocks(uint64_t artiBlock, uint64_t linkBlocks, uint64_t dofBlocks,
                           uint numLinks, uint totalDofs, uint lane)
{
    for (uint i = 0u; i < numLinks; ++i)
    {
        const uint64_t blk = linkBlocks + uint64_t(i) * PXG_AB_LINK_SIZE;

        jcalcZeroSpatialVector(blk, PXG_AB_LINK_mDeltaMotion, lane);
        jcalcZeroSpatialVector(blk, PXG_AB_LINK_mPosMotionVelocity, lane);
        // the identity quaternion, NOT zero
        abStoreF4(blk, PXG_AB_LINK_mDeltaQ, lane, float4(0.0f, 0.0f, 0.0f, 1.0f));
        jcalcZeroSpatialVector(blk, PXG_AB_LINK_mScratchImpulse, lane);
        jcalcZeroSpatialVector(blk, PXG_AB_LINK_mSolverSpatialDeltaVel, lane);
        jcalcZeroSpatialVector(blk, PXG_AB_LINK_mSolverSpatialImpulse, lane);
        jcalcZeroSpatialVector(blk, PXG_AB_LINK_mSolverSpatialInternalConstraintImpulse, lane);
        jcalcZeroSpatialVector(blk, PXG_AB_LINK_mConstraintForces, lane);

        abStoreU32(blk, PXG_AB_LINK_mStaticContactStartIndex, lane, 0u);
        abStoreU32(blk, PXG_AB_LINK_mNbStaticContacts, lane, 0u);
        abStoreU32(blk, PXG_AB_LINK_mStaticJointStartIndex, lane, 0u);
        abStoreU32(blk, PXG_AB_LINK_mNbStaticJoints, lane, 0u);
        abStoreF32(blk, PXG_AB_LINK_mDeltaScale, lane, 0.0f);
    }

    jcalcZeroSpatialVector(artiBlock, PXG_AB_mRootDeferredZ, lane);
    jcalcZeroSpatialVector(artiBlock, PXG_AB_mCommonLinkDeltaVelocity, lane);
    abStoreU32(artiBlock, PXG_AB_mLinkWithDeferredImpulse, lane, 0u);

    for (uint i = 0u; i < totalDofs; ++i)
    {
        const uint64_t dof = dofBlocks + uint64_t(i) * PXG_AB_DOF_SIZE;
        abStoreF32(dof, PXG_AB_DOF_mDeferredQstZ, lane, 0.0f);
        abStoreF32(dof, PXG_AB_DOF_mInvStIsT_x, lane, 0.0f);
        abStoreF32(dof, PXG_AB_DOF_mInvStIsT_y, lane, 0.0f);
        abStoreF32(dof, PXG_AB_DOF_mInvStIsT_z, lane, 0.0f);
    }
}

// `articulationIndex` and `sleepThreshold` come from OUTSIDE the articulation
// record -- the remap id and the body sim's sleep threshold, both of which the
// launch wrapper resolves -- so they are parameters rather than reads.
void jcalcFillBlockHeader(uint64_t a, uint64_t artiBlock, uint articulationIndex,
                          float sleepThreshold, uint lane)
{
    abStoreU32(artiBlock, PXG_AB_mNumLinks, lane,
               artiU16Field(a, PXG_ARTI_DATA_NUM_LINKS));
    abStoreU32(artiBlock, PXG_AB_mTotalDofs, lane,
               artiU16Field(a, PXG_ARTI_DATA_NUM_JOINT_DOFS));
    // flags is a PxU8 in the record and a PxU32 in the block
    abStoreU32(artiBlock, PXG_AB_mFlags, lane, artiU8Field(a, PXG_ARTI_DATA_FLAGS));
    abStoreU32(artiBlock, PXG_AB_mArticulationIndex, lane, articulationIndex);
    abStoreF32(artiBlock, PXG_AB_mSleepThreshold, lane, sleepThreshold);
    // mStateDirty is a PxU8[32]: four lanes per word
    abStoreU8(artiBlock, PXG_AB_mStateDirty, lane, 0u);
    // a device POINTER lane array: 8-byte stride among 4-byte neighbours
    abStoreU64(artiBlock, PXG_AB_mMotionVelocitiesPtr, lane,
               artiPtrField(a, PXG_ARTI_MOTION_VELOCITIES));
}

[shader("compute")]
[numthreads(32, 1, 1)]
void artiJcalcInit(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gJcalcInit.nbArticulations)
        return;

    const uint64_t a = gJcalcInit.articulations + uint64_t(lane) * PXG_ARTI_STRIDE;

    if (gJcalcInit.doHeader != 0u)
        jcalcFillBlockHeader(a, gJcalcInit.artiBlocks, gJcalcInit.articulationIndex + lane,
                             gJcalcInit.sleepThreshold + float(lane), lane);

    jcalcInitializeBlocks(gJcalcInit.artiBlocks, gJcalcInit.linkBlocks, gJcalcInit.dofBlocks,
                          gJcalcInit.numLinks, gJcalcInit.totalDofs, lane);
}
