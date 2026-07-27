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

// softBodySolveBda.slang
//
// P8.0 tranche 4 — the first two kernels of the softbody FEM-volume SOLVE loop on the
// Vulkan engine ABI (BDA).  Ported line-for-line from
// gpusimulationcontroller/src/CUDA/softBodyGM.cu:
//
//   sb_gm_zeroTetMultipliers       (:1801, launch PxgSoftBodyCore.cpp:196)
//       clears every tetrahedron's Lagrange-multiplier spatial vector at the top of the
//       solve.  Without it the constraint solver accumulates into last frame's
//       multipliers and the material response drifts frame over frame.
//   sb_gm_applyExternalDeltasLaunch (:2609, launch PxgSoftBodyCore.cpp:872)
//       folds the accumulated per-vertex position correction (mSimDelta, written by the
//       contact/attachment solves) back into position, velocity and mSimDeltaPos, then
//       zeroes the delta buffer for the next iteration.  This is the softbody twin of
//       cloth_applyExternalDeltasLaunch, which the cloth arc has relied on since P7.1.
//
// They bracket the solve: zeroTetMultipliers opens it, applyExternalDeltas closes each
// iteration.  Both are pure load/scale/store — no transcendental, no warp op, no
// cross-lane communication at all.
//
// Doctrine (identical to the prior softbody tranches):
//   - the PxgSoftBody descriptor arrives as a device-pointer ARRAY read through Ptr<>
//     (the byte-exact pxgSlangSoftBodyDesc mirror); mSimVelocity_InvMass @88,
//     mSimPosition_InvMass @96, mSimDeltaPos @128, mSimTetraMultipliers @264,
//     mSimDelta @272, mNumVertsGM @372 and mNumTetsGM @376 are the fields touched;
//   - PxgSpatialVectorBlock (PxgArticulation.h:77) is the 32-lane SoA block the
//     multipliers live in: float4 mTopxyz_bx[32] @0 (512 B) then float2 mbyz[32] @512
//     (256 B) -> 768-byte stride.  storeSpatialVector(block, v, lane) writes
//     mTopxyz_bx[lane] = (v.top.xyz, v.bottom.x) and mbyz[lane] = v.bottom.yz, so
//     zeroing a tet means two stores at block (tetIdx/32), lane (tetIdx&31).
//
// DEVIATIONS from CUDA:
//   - sb_gm_applyExternalDeltasLaunch opens with a `blockCopy` of the WHOLE 432-byte
//     PxgSoftBody descriptor into __shared__ followed by __syncthreads, then reads the
//     shared copy.  That is a pure read-caching optimisation for a block whose 64
//     threads would otherwise each pull the same descriptor words; it is dropped here
//     and every lane reads the descriptor directly through Ptr<> (the same fan-out
//     treatment the earlier softbody/cloth ports document).  Dropping it also removes
//     the kernel's only barrier, so there is no groupshared and no uniformity concern;
//   - CUDA's `delta.x *= scale; delta.y *= scale, delta.z *= scale;` (note the comma
//     operator on the second statement — a typo that happens to behave identically) is
//     written as a float3 multiply.
//
// Both kernels are NoContraction-stamped (PXG_NOCONTRACT_SHADERS): applyExternalDeltas
// computes `vel + delta * invDt`, a multiply feeding an add that would otherwise fuse
// into an FMA and break bitwise parity with the -ffp-contract=off CPU reference.  There
// is NO transcendental in either kernel — this tranche is FULLY bit-exact, no tolerance
// seam.

import pxgSlangSoftBodyDesc;   // PxgSoftBodyGpu (mSimPosition_InvMass@96, ...)

// PxgSpatialVectorBlock (PxgArticulation.h:77): float4 mTopxyz_bx[32] @0, float2 mbyz[32]
// @512 -> 768-byte stride.
static const uint64_t SPATIAL_BLOCK_STRIDE = 768UL;
static const uint64_t SPATIAL_BLOCK_MBYZ_OFFSET = 512UL;

// ---------------------------------------------------------------------------
// sb_gm_zeroTetMultipliers (softBodyGM.cu:1801)
// ---------------------------------------------------------------------------
// CUDA arg pack (PxgSoftBodyCore.cpp:198): { PxgSoftBody* gSoftbodies,
// const PxU32* activeSoftbodies } -> 16-byte blob.  (ABI row {8,8}.)
struct ZeroTetMultipliersParams
{
    uint64_t gSoftbodies;       // @0
    uint64_t activeSoftbodies;  // @8  -> 16
}
[[vk::push_constant]]
ConstantBuffer<ZeroTetMultipliersParams> gZero;

[shader("compute")]
[numthreads(64, 1, 1)] // SB_SOLVETETRA; grid (nbBlocks, nbActiveSoftBodies, 1)
void sb_gm_zeroTetMultipliers(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSoftBodyGpu> gSoftbodies = Ptr<PxgSoftBodyGpu>(gZero.gSoftbodies);
    const uint softbodyId = (Ptr<uint>(gZero.activeSoftbodies))[groupId.y];

    const uint groupThreadIdx = groupThreadId.x + groupId.x * 64u;

    const uint numTets = gSoftbodies[softbodyId].mNumTetsGM;

    if (groupThreadIdx < numTets)
    {
        // storeSpatialVector(mSimTetraMultipliers[idx/32], UnAlignedSpatialVector::Zero(),
        //                    idx & 31)
        const uint64_t blockBase = gSoftbodies[softbodyId].mSimTetraMultipliers +
                                   uint64_t(groupThreadIdx >> 5) * SPATIAL_BLOCK_STRIDE;
        const uint lane = groupThreadIdx & 31u;

        Ptr<float4>(blockBase + uint64_t(lane) * 16UL)[0] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        Ptr<float2>(blockBase + SPATIAL_BLOCK_MBYZ_OFFSET + uint64_t(lane) * 8UL)[0] =
            float2(0.0f, 0.0f);
    }
}

// ---------------------------------------------------------------------------
// sb_gm_applyExternalDeltasLaunch (softBodyGM.cu:2609)
// ---------------------------------------------------------------------------
// CUDA arg pack (PxgSoftBodyCore.cpp:874): { PxgSoftBody* gSoftbodies,
// const PxU32* activeSoftbodies, const PxReal invDt } -> 20, rounded up to the pointer
// alignment -> 24-byte blob.  (ABI row {8,8,4}.)
struct ApplyExternalDeltasParams
{
    uint64_t gSoftbodies;       // @0
    uint64_t activeSoftbodies;  // @8
    float invDt;                // @16
    uint _pad20;                // @20  -> 24
}
[[vk::push_constant]]
ConstantBuffer<ApplyExternalDeltasParams> gApply;

[shader("compute")]
[numthreads(64, 1, 1)] // SB_SOLVETETRA; grid (nbBlocks, nbActiveSoftbodies, 1)
void sb_gm_applyExternalDeltasLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSoftBodyGpu> gSoftbodies = Ptr<PxgSoftBodyGpu>(gApply.gSoftbodies);
    const uint softbodyId = (Ptr<uint>(gApply.activeSoftbodies))[groupId.y];

    // The CUDA blockCopy of the whole descriptor into __shared__ + __syncthreads is a
    // read-caching optimisation only (see the header) — read it directly instead.
    Ptr<float4> curPositions = Ptr<float4>(gSoftbodies[softbodyId].mSimPosition_InvMass);
    Ptr<float4> vels = Ptr<float4>(gSoftbodies[softbodyId].mSimVelocity_InvMass);
    Ptr<float4> simDelta = Ptr<float4>(gSoftbodies[softbodyId].mSimDelta);
    Ptr<float4> simDeltaPos = Ptr<float4>(gSoftbodies[softbodyId].mSimDeltaPos);

    const uint vertIdx = groupThreadId.x + groupId.x * 64u;
    const uint nbVerts = gSoftbodies[softbodyId].mNumVertsGM;

    if (vertIdx < nbVerts)
    {
        const float4 pos = curPositions[vertIdx];
        const float4 vel = vels[vertIdx];

        float4 delta = simDelta[vertIdx];

        // delta.w counts how many constraints contributed; averaging divides by it, but
        // never by less than 1 (an untouched vertex has delta.w == 0 and delta == 0).
        const float scale = 1.0f / max(delta.w, 1.0f);

        delta.x *= scale; delta.y *= scale; delta.z *= scale;
        delta.w = 0.0f;

        const float4 newVel = vel + delta * gApply.invDt;

        curPositions[vertIdx] = pos + delta;

        vels[vertIdx] = newVel;

        simDeltaPos[vertIdx] = simDeltaPos[vertIdx] + delta;

        // clear delta
        simDelta[vertIdx] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}
