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
//
// ===========================================================================
// softBodyParticleAttachBda — sb_solveOutputParticleAttachmentDeltaVLaunch
// (softBodyGM.cu:2522), on the engine BDA ABI.  P9.48b.
//
// Launch: PxgSoftBodyCore.cpp:1968, FOUR arguments
// {softbodies(8), particleSystems(8), attachments(8), numAttachments(4)}, grid
// SB_UPDATEROTATION = 1024, block SB_UPDATEROTATION = 256 — the same shape as the
// cloth sibling next door, with `particleSystems` where `clothes` was.
//
// This is the FOURTH consumer of PxgFEMFEMAttachmentConstraint and the odd one out: it
// pins a TETRAHEDRON of a softbody to a single PARTICLE.  The other three landed at
// P8.19 (softbody-softbody), P9.32 (cloth-cloth) and the cloth sibling in
// softBodyClothAttachBda; the block's byte layout and the shared helpers live in
// `pxgSlangFemFemAttach`.
//
// WHAT IS THE SAME, and must not be "corrected" because side 0 became a point: elemId1 is
// still the side that receives -delta, elemId0 still receives +delta, the relative
// position is still `pos1 - pos0`, the scale is still `error * (1.0f/wTot)`, and the tet
// scatter still passes a constraint weight of 1.
//
// ---------------------------------------------------------------------------
// WHAT ACTUALLY DIFFERS — seven things, and every one of them is load-bearing
// ---------------------------------------------------------------------------
// E1  NO CONE LIMIT.  All three siblings branch on isConeLimitedEnabled and can replace
//     the error with computeConeLimitedError.  This one does not: `error` is the raw
//     `pos1 - pos0` unconditionally (softBodyGM.cu:2587-2588).  The constraint block still
//     CARRIES low_high_angles and attachmentBarycentric at their usual offsets; the kernel
//     simply never reads them.  Transcribing the sibling's branch here would be a
//     plausible-looking addition that changes the answer whenever a limit is configured.
//
// E2  `barycentric0.x` IS NOT A BARYCENTRIC COORDINATE.  It is a user BUFFER ID, read as
//     `PxU32(constraint.barycentric0[offset].x)` (:2574) — a C float-to-uint conversion,
//     i.e. truncation toward zero, not a bit cast.  Nothing else of barycentric0 is read.
//
// E3  THE PARTICLE INDEX TAKES THREE INDIRECTIONS (:2574-2579):
//       o          = mParticleBufferRunsum[findBufferIndexFromUniqueId(system, bufferId)]
//       particleId = PxGetParticleIndex(elemId0) + o
//       particleId = mUnsortedToSortedMapping[particleId]        <- the SORTED index
//     and the position is then mSortedPositions_InvMass[particleId].  Skipping any one of
//     the three still indexes a live array.
//
// E4  findBufferIndexFromUniqueId IS NOT dfFindRange.  It is an upper-bound binary search
//     over mParticleBufferSortedUniqueIds returning
//     mParticleBufferSortedUniqueIdsOriginalIndex[r - 1], with an early 0 for an empty
//     table (particleSystem.cuh:99-117).  `r - 1` on r == 0 would wrap; the length-0 guard
//     is what makes that unreachable.
//
// E5  wTot0 IS ASYMMETRIC.  Side 1 keeps the four-term barycentric sum, but side 0 is
//     simply `pos0.w` — the particle's inverse mass, raw (:2592).  Every sibling computes
//     wTot0 as a weighted sum over corners; here there is one corner and no weight.
//
// E6  A ZERO-MASS PAIR IS SKIPPED WITH `continue`, NOT `return` (:2584-2585): a lane whose
//     particle AND tet are both infinite-mass moves to the next iteration rather than
//     abandoning the rest of its grid-stride work.  The `workIndex >= numAttachments`
//     test above it IS a `return`, and the two must not be confused.
//
// ---------------------------------------------------------------------------
// THREE GUARDS THAT CANNOT BE KILLED, AND WHY (P9.48b sweep)
// ---------------------------------------------------------------------------
// Nineteen of the twenty-two mutants of this kernel die.  The three that live are the
// same shape: an UPSTREAM guard whose condition a DOWNSTREAM guard already implies, so
// removing it changes nothing that can be observed.  Each proof rests on an invariant the
// engine states itself, not on the fixture's choice of numbers.
//
//   * `invMass1 == 0.0f && invMass0 == 0.0f` (E6).  Attachment barycentrics are clamped
//     non-negative on construction -- softBody.cuh:108 says so where it explains why
//     updateTetPositionDelta tests only the products.  Under that invariant `pos1.w == 0`
//     forces every weighted corner mass to 0, hence wTot1 = 0; and `pos0.w == 0` IS
//     wTot0 = 0.  So the skipped lane always has wTot == 0, which `wTot > 0.0f` already
//     suppresses.  The guard is defensive, and it is kept because the .cu keeps it.
//   * `pos0.w > 0.0f` (E7's second guard).  What it suppresses is `delta * 0.0f`, the zero
//     vector, added component-wise to x, y and z.  It saves three atomics, not a result.
//   * `wTot > 0.0f`.  Reaching wTot == 0 needs wTot0 == 0 AND wTot1 == 0, i.e. pos0.w == 0
//     and every b1w component 0 -- exactly the state in which both scatters' OWN guards
//     (`invMassBary.* > 0` per corner, `pos0.w > 0` for the particle) are false.  Whatever
//     `1.0f / 0.0f` produced would reach no store.
//
// All three are written the .cu's way regardless.  A guard that is redundant today stops
// being redundant the moment one of the invariants moves.
//
// E7  THE PARTICLE SCATTER IS THE THREE-COMPONENT AtomicAdd.  `AtomicAdd(mDelta[id],
//     delta*pos0.w)` resolves to the (float4&, PxVec3) overload (atomic.cuh:54), which
//     touches x, y and z and LEAVES .w ALONE — unlike updateTetPositionDelta, whose
//     (float4&, PxVec3, PxReal) overload adds the constraint weight into .w.  It also
//     carries a SECOND guard, `pos0.w > 0.0f`, nested inside the `wTot > 0.0f` one.
// ===========================================================================

import pxgSlangSoftBodyDesc;   // PxgSoftBodyGpu (mSimTetIndices@80, mSimPosition_InvMass@96, mSimDelta@272)
import pxgSlangParticleDesc;   // PxgParticleSystemGpu
import pxgSlangFemFemAttach;   // FAC_* layout + facSlot + updateTetPositionDelta
import pxgSlangAtomicFloat;    // pxgAtomicAddF4XYZ (three components, .w untouched)

// PxgSoftBodyKernelGridDim/BlockDim::SB_UPDATEROTATION.
static const uint SBPA_GRID  = 1024u;
static const uint SBPA_BLOCK = 256u;

// PxgSimulationCoreDesc.h:40/41, PxDeformableVolume.h:57.
static const uint PXG_BITSHIFT_TET_ID = 20u;
static const uint PXG_VOLUME_TET_MASK = 0x000fffffu;

// findBufferIndexFromUniqueId (particleSystem.cuh:99).  E4: an UPPER-BOUND binary search
// (`values[m] > id` moves r down, everything else moves l up), then a lookup of r - 1 in
// the ORIGINAL-INDEX table.  The length-0 early return is what keeps r - 1 from wrapping.
uint findBufferIndexFromUniqueId(Ptr<PxgParticleSystemGpu> systems, uint systemId, uint uniqueBufferId)
{
    const uint length = systems[systemId].mCommonData_NumParticleBuffers;
    if (length == 0u)
        return 0u;

    Ptr<uint> values = Ptr<uint>(systems[systemId].mParticleBufferSortedUniqueIds);

    uint l = 0u, r = length;
    while (l < r)
    {
        const uint m = (l + r) / 2u;
        if (values[m] > uniqueBufferId)
            r = m;
        else
            l = m + 1u;
    }
    return Ptr<uint>(systems[systemId].mParticleBufferSortedUniqueIdsOriginalIndex)[r - 1u];
}

// findBufferOffset (softBodyGM.cu:2515).
uint findBufferOffset(Ptr<PxgParticleSystemGpu> systems, uint systemId, uint uniqueBufferId)
{
    const uint bufferIndex = findBufferIndexFromUniqueId(systems, systemId, uniqueBufferId);
    return Ptr<uint>(systems[systemId].mParticleBufferRunsum)[bufferIndex];
}

struct ParticleSbAttachSolveParams
{
    uint64_t softbodies;      // @0
    uint64_t particleSystems; // @8
    uint64_t attachments;     // @16
    uint numAttachments;      // @24
    uint _pad28;              // @28 -> 32
}

[[vk::push_constant]] ConstantBuffer<ParticleSbAttachSolveParams> gPs;

[shader("compute")]
[numthreads(256, 1, 1)]   // SB_UPDATEROTATION
void sb_solveOutputParticleAttachmentDeltaVLaunch(uint3 gtid : SV_GroupThreadID,
                                                  uint3 gid : SV_GroupID)
{
    const uint numAttachments = gPs.numAttachments;
    const uint nbBlocksRequired = (numAttachments + SBPA_BLOCK - 1u) / SBPA_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + SBPA_GRID - 1u) / SBPA_GRID;
    const uint idx = gtid.x;

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gPs.softbodies);
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gPs.particleSystems);

    for (uint i = 0u; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * SBPA_BLOCK + idx + nbIterationsPerBlock * gid.x * SBPA_BLOCK;
        if (workIndex >= numAttachments)
            break;   // CUDA `return`; nothing follows the loop.  NOT E6's `continue`.

        // ---- side 1: the softbody's simulation TETRAHEDRON -------------------------------
        // The .cu reads this PxU64 slot into a PxU32, i.e. the LOW half only.
        const uint elemId1 = Ptr<uint>(facSlot(gPs.attachments, workIndex, FAC_ELEMID1, 8UL))[0];
        const uint softBodyId = elemId1 >> PXG_BITSHIFT_TET_ID;      // PxGetSoftBodyId
        const uint tetrahedronId = elemId1 & PXG_VOLUME_TET_MASK;    // PxGetSoftBodyElementIndex

        Ptr<float4> simPos = Ptr<float4>(softbodies[softBodyId].mSimPosition_InvMass);
        const uint4 tetInd = Ptr<uint4>(softbodies[softBodyId].mSimTetIndices)[tetrahedronId];
        const float4 bary1 = Ptr<float4>(facSlot(gPs.attachments, workIndex, FAC_BARYCENTRIC1, 16UL))[0];

        const uint64_t accumDeltaV = softbodies[softBodyId].mSimDelta;

        const float4 sp0 = simPos[tetInd.x];
        const float4 sp1 = simPos[tetInd.y];
        const float4 sp2 = simPos[tetInd.z];
        const float4 sp3 = simPos[tetInd.w];
        const float4 pos1 = sp0 * bary1.x + sp1 * bary1.y + sp2 * bary1.z + sp3 * bary1.w;

        const float invMass1 = pos1.w;

        // ---- side 0: ONE PARTICLE, reached through three indirections (E2, E3, E4) -------
        // elemId0 is read at the FULL 64 bits here, unlike elemId1 above.
        const uint2 elemId0 = Ptr<uint2>(facSlot(gPs.attachments, workIndex, FAC_ELEMID0, 8UL))[0];
        const uint particleSystemId = elemId0.y;                     // PxGetParticleSystemId
        // E2: a buffer id, converted from float by TRUNCATION, not reinterpreted.
        const uint userBufferId = uint(Ptr<float4>(facSlot(gPs.attachments, workIndex,
                                                           FAC_BARYCENTRIC0, 16UL))[0].x);
        const uint o = findBufferOffset(systems, particleSystemId, userBufferId);
        const uint unsortedId = elemId0.x + o;                       // PxGetParticleIndex + offset
        const uint particleId =
            Ptr<uint>(systems[particleSystemId].mUnsortedToSortedMapping)[unsortedId];
        const float4 pos0 = Ptr<float4>(systems[particleSystemId].mSortedPositions_InvMass)[particleId];

        const float invMass0 = pos0.w;

        // E6: `continue`, not `return` -- the lane keeps its remaining grid-stride work.
        if (invMass1 == 0.0f && invMass0 == 0.0f)
            continue;

        // E1: no cone limit anywhere in this kernel.
        const float4 diff = pos1 - pos0;
        const float3 error = float3(diff.x, diff.y, diff.z);

        const float4 b1w = float4(bary1.x * sp0.w, bary1.y * sp1.w, bary1.z * sp2.w, bary1.w * sp3.w);

        const float wTot0 = pos0.w;                                  // E5: raw, not a sum
        const float wTot1 = b1w.x * bary1.x + b1w.y * bary1.y + b1w.z * bary1.z + b1w.w * bary1.w;
        const float wTot = wTot0 + wTot1;

        if (wTot > 0.0f)
        {
            const float3 delta = error * (1.0f / wTot);
            updateTetPositionDelta(accumDeltaV, tetInd, -delta, b1w, 1.0f);
            // E7: the three-component AtomicAdd, under a SECOND guard, into mDelta.
            if (pos0.w > 0.0f)
            {
                const float3 d = delta * pos0.w;
                pxgAtomicAddF4XYZ(systems[particleSystemId].mDelta, particleId, d);
            }
        }
    }
}
