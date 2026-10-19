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

// pxgSlangSoftBodySolve.slang
//
// P9.39d.  The nine device helpers every softbody SOLVE kernel needs, in one place.
//
// They were written for softBodySelfSolveBda in P8.x and were file-local there.  The
// softbody-vs-PARTICLE solve half (P9.39d..f) needs all nine of them, and copying them
// would have made a second -- then a third -- transcription of arithmetic that has to
// stay bit-identical across every caller.  P9.38 spent a whole tranche collapsing
// fifteen such copies of one binary search, and its finding was that all fifteen were
// CORRECT and that this was the reason to consolidate, not the reason to leave them:
// fifteen identical copies are fifteen future chances to re-derive one of them wrongly
// from memory.  So the module is created BEFORE the second copy exists rather than
// after the fifteenth.
//
// softBodySelfSolveBda now imports these rather than defining them; the change is
// verified inert with the SPIR-V DAG screen from P9.37, which reports one arithmetic
// graph per function name across every copy in the built corpus.
//
// WHAT IS DELIBERATELY NOT HERE: the constraint-block offsets.  Those describe
// PxgSoftBodySoftBodyConstraintBlock, whose layout is specific to the self-collision
// solve; the particle solve uses PxgFEMParticleConstraintBlock, a DIFFERENT struct with
// a different stride.  Putting both in one module would invite exactly the confusion
// this file exists to prevent.

module pxgSlangSoftBodySolve;

import pxgSlangAtomicFloat;    // pxgAtomicAddF4XYZ / pxgAtomicAddF4W

// ---------------------------------------------------------------------------
// The softbody id encoding (PxgSimulationCoreDesc.h:110/:127): (id << 20) | tetId.
// ---------------------------------------------------------------------------
public static const uint PXG_BITSHIFT_TET_ID = 20u;
public static const uint PX_MAX_NB_DEFORMABLE_VOLUME_TET = 0x000fffffu;

public uint getSoftBodyId(uint compressedId) { return compressedId >> PXG_BITSHIFT_TET_ID; }
public uint getSoftBodyElementIndex(uint compressedId) { return PX_MAX_NB_DEFORMABLE_VOLUME_TET & compressedId; }

// ---------------------------------------------------------------------------
// PxsDeformableVolumeMaterialData (PxsDeformableVolumeMaterialCore.h:50), PX_ALIGN(16):
// youngs @0, poissons @4, dynamicFriction @8, elasticityDamping @12, materialModel (PxU16)
// @16, _pad (PxU16) @18, deformThreshold @20, deformLowLimitRatio @24,
// deformHighLimitRatio @28 -> 32 bytes, and 32 is already a multiple of 16 so the array
// stride is 32.
// ---------------------------------------------------------------------------
public static const uint64_t MAT_STRIDE = 32UL;
public static const uint64_t MAT_dynamicFriction = 8UL;

public float materialDynamicFriction(uint64_t materials, uint materialIndex)
{
    return Ptr<float>(materials + uint64_t(materialIndex) * MAT_STRIDE + MAT_dynamicFriction)[0];
}

// mMaterialIndices is a PxU16 array; llvmpipe has no 16-bit storage in these shaders, so the
// containing 32-bit word is read and the half extracted.  NOTE `& ~uint64_t(3)`: writing
// `& ~3UL` MISCOMPILES -- Slang evaluates the complement in 32 bits and clears the top half
// of the device address (documented at length in softBodySolveBda.slang).
public uint materialIndexAt(uint64_t base, uint tetIdx)
{
    const uint64_t addr = base + uint64_t(tetIdx) * 2UL;
    const uint word = Ptr<uint>(addr & ~uint64_t(3))[0];
    return (addr & 2UL) != 0UL ? (word >> 16u) : (word & 0xffffu);
}

// ---------------------------------------------------------------------------
// computeTetraContact (softBody.cuh:61) — the barycentric blend of the four tet corners,
// with the four .w components handed back as the per-corner inverse masses.
// ---------------------------------------------------------------------------
// barycentricProject (softBody.cu:640) -- the blend on its own.  Upstream has TWO
// functions with this arithmetic in it: barycentricProject, and computeTetraContact
// (softBody.cuh:61), which additionally hands back the four .w components as the
// per-corner inverse masses.  Here there is ONE expression and the second function is a
// wrapper over it, so the two cannot drift.
//
// P9.39d wrote a third copy of this blend as a file-local `sppTetProject` in
// softBodyPsPrepareBda -- in the very tranche that created this module to stop exactly
// that.  The module sweep found it, by reporting a mutation of computeTetraContact as
// "not run: the .spv did not change", which was true and was the finding: the prepare
// shader was not calling this function at all.
//
// Association: ((a*bx + b*by) + c*bz) + d*bw, left to right, over WHOLE float4s -- so
// the fourth component that falls out is the barycentrically blended inverse mass.
public float4 barycentricProject(Ptr<float4> vals, uint4 tetId, float4 bary)
{
    const float4 v0 = vals[tetId.x];
    const float4 v1 = vals[tetId.y];
    const float4 v2 = vals[tetId.z];
    const float4 v3 = vals[tetId.w];
    return v0 * bary.x + v1 * bary.y + v2 * bary.z + v3 * bary.w;
}

public float4 computeTetraContact(Ptr<float4> vals, uint4 tetId, float4 bary, out float4 invMass)
{
    invMass = float4(vals[tetId.x].w, vals[tetId.y].w, vals[tetId.z].w, vals[tetId.w].w);
    return barycentricProject(vals, tetId, bary);
}

// AtomicAdd(float4& out, const PxVec3& v, PxReal addition) — three component adds plus the
// .w weight the later averaging pass divides by.  Dropping the .w add silently halves every
// delta, so it is not optional bookkeeping.
public void atomicAddDelta(uint64_t base, uint vertIdx, float3 dP, float addition)
{
    pxgAtomicAddF4XYZ(base, vertIdx, dP);
    pxgAtomicAddF4W(base, vertIdx, addition);
}

// updateTetraPosDelta (softBody.cuh:77).
public void updateTetraPosDelta(float4 invMasses, float4 bary, uint4 tetId, float3 deltaPos,
                         uint64_t outDeltasAddr)
{
    if (invMasses.x > 0.0f && abs(bary.x) > 1e-6f)
        atomicAddDelta(outDeltasAddr, tetId.x, deltaPos * (invMasses.x * bary.x), 1.0f);
    if (invMasses.y > 0.0f && abs(bary.y) > 1e-6f)
        atomicAddDelta(outDeltasAddr, tetId.y, deltaPos * (invMasses.y * bary.y), 1.0f);
    if (invMasses.z > 0.0f && abs(bary.z) > 1e-6f)
        atomicAddDelta(outDeltasAddr, tetId.z, deltaPos * (invMasses.z * bary.z), 1.0f);
    if (invMasses.w > 0.0f && abs(bary.w) > 1e-6f)
        atomicAddDelta(outDeltasAddr, tetId.w, deltaPos * (invMasses.w * bary.w), 1.0f);
}

// getSoftBodyInvMass (softBody.cuh:38) — the blended inverse mass of a TET contact is scaled
// by the LENGTH of the barycentric vector.  Identical to the factor P8.3 found in the
// attachment prepare, and just as load-bearing: dropping the square root gives every
// self-contact the wrong stiffness.
public float getSoftBodyInvMass(float baryMass, float4 bary)
{
    const float scale = sqrt(bary.x * bary.x + bary.y * bary.y + bary.z * bary.z + bary.w * bary.w);
    return baryMass * scale;
}

// computeInvMass (softBody.cu:608).
public float computeInvMass(uint4 tetId, Ptr<float4> position_invmass, float4 bary)
{
    const float4 a = position_invmass[tetId.x];
    const float4 b = position_invmass[tetId.y];
    const float4 c = position_invmass[tetId.z];
    const float4 d = position_invmass[tetId.w];
    const float invMass = a.w * bary.x + b.w * bary.y + c.w * bary.z + d.w * bary.w;
    return getSoftBodyInvMass(invMass, bary);
}

// PxVec3::normalize (PxVec3.h:299) — returns the LENGTH and leaves the vector unit; a
// zero-length vector is left alone and the length is 0.
public float vecNormalizeInPlace(inout float3 v)
{
    const float m = v.x * v.x + v.y * v.y + v.z * v.z;
    if (m > 0.0f)
    {
        // P9.36: normalize() scales through operator/= (PxVec3.h:260) -- ONE reciprocal,
        // three multiplies.  The header two lines up already named the right function.
        const float len = sqrt(m);
        v = v * (1.0f / len);
        return len;
    }
    return 0.0f;
}
