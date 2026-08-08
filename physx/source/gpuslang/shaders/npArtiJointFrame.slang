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
