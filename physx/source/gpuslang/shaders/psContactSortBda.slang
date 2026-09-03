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

// psContactSortBda — P6.4b-1: the contact-ORGANIZATION half of the PBD
// particle/primitive consume path on the BDA engine ABI.  Three kernels that run
// AFTER the plain radix sort has produced the by-rigid/by-particle contact ranks:
//
//   ps_reorderPrimitiveContactsLaunch (cudaParticleSystem.cu:1687) — materializes
//     the two sorted PxgParticlePrimitiveContact arrays by gathering the original
//     contacts through the two rank remap tables.  Launch: grid BOUNDCELLUPDATE
//     (32), block BOUNDCELLUPDATE (512); PxgParticleSystemCore.cpp:1302.
//   ps_findStartEndParticleFirstLaunch (particlesystem.cu:2277) — pass 1: writes,
//     for every contact, the GLOBAL 0-based range index (which particle run the
//     contact belongs to) into `offsets`, and the total range-start count into
//     `blockOffsets[0]`.
//   ps_findStartEndParticleSecondLaunch (particlesystem.cu:2402) — pass 2: from the
//     GLOBAL `offsets` writes each particle's [start,end) range + the total pair
//     count.  Launches: grid ACCUMULATE_DELTA (32), block ACCUMULATE_DELTA (512);
//     PxgParticleSystemCore.cpp:1648/:1690.
//
// P9.48a adds the FEM TWINS of the two range-find passes:
//
//   ps_findStartEndFEMFirstLaunch  (particlesystem.cu:2519)
//   ps_findStartEndFEMSecondLaunch (particlesystem.cu:2644)
//
// They build the same per-particle [start,end) ranges over the FEM-vs-particle
// contact stream instead of the primitive one, and they are launched from BOTH
// PxgSoftBodyCore.cpp (:2356/:2416) and PxgFEMClothCore.cpp (:426/:456) with the
// same dims (block SB_ACCUMULATE_DELTA 512, grid 32 — numerically identical to
// the particle path's ACCUMULATE_DELTA).  Diffed line for line against their
// twins; the differences are exactly three, and only the first is behavioural:
//
//   1. the record read is PxgFemOtherContactInfo (16 B) and the id is its
//      pairInd0, not PxgParticlePrimitiveContact::particleId (@24 of 32 B).
//   2. the FEM pass 1 leaves `PxU64 particleId` UNINITIALISED where the particle
//      pass 1 seeds it with 0xffffffffffffffff.  Dead either way: every read of
//      it sits inside `if (workIndex < tNumContacts)`, the same guard that
//      assigns it.  The port keeps the particle twin's sentinel, which is the
//      defined-behaviour spelling of the same dead value.
//   3. UPSTREAM CANDIDATE: the FEM pass 2 has NO trailing __syncthreads() at the
//      end of its iteration loop, while the particle twin has one carrying the
//      comment "sParticleId (shared memory) is read and written in the same loop
//      - read and write must be separated with syncs".  The FEM loop reads
//      sParticleId[tid] and writes sParticleId[tid+1] in the same body, so
//      iteration i+1's write races iteration i's read across warps whenever
//      numIterationPerBlock > 1.  Reproducing that would be reproducing a race,
//      not a behaviour — the port keeps the barrier.  DOCUMENTED, NOT REPORTED.
//
// llvmpipe deviations (flagged):
//   - The CUDA range-find pass uses __ballot_sync + __popc(mask & threadMask) for
//     the within-warp exclusive rank of the isRangeStart flag, and a per-warp
//     warpScan<Add>.  Vulkan/llvmpipe make no 32-lane subgroup guarantee, so both
//     are transcribed to the PROVEN portable warp32InclusiveScan (pxgSlangScan,
//     verified bit-exact by testRadixSortDesc): popc(mask & threadMask) - (1 -
//     isRangeStart) is algebraically inclusiveScan(isRangeStart) - 1, and the
//     per-warp warpScan is the same segmented inclusive scan minus its input.
//   - The cross-warp scan the CUDA runs INSIDE `if (warpIndex == 0)` (single-warp,
//     shuffle-only, no block barrier) is hoisted to a block-uniform call: every
//     thread feeds warp32InclusiveScan (warp 0 the real per-warp totals, other
//     warps zero) so the barrier-based scan stays in uniform control flow; only
//     warp 0's result is consumed, identical to the CUDA.
//   - GLOBAL vs WITHIN-BLOCK offsets: the CUDA pass 1 splits the contact array
//     across 32 independent thread blocks, each restarting its range counter at 0
//     (a WITHIN-BLOCK offset), and defers the cross-block prefix to pass 2 (which
//     exclusive-scans the per-block counts and adds that base).  A global running
//     range index cannot be produced across 32 concurrent Vulkan workgroups in a
//     single dispatch — there is no grid-wide sync, so block b cannot know block
//     b-1's count.  The port therefore folds pass 1 onto ONE workgroup (block 0),
//     carrying a single global accumulator gsBlockAccum1 across every 512-contact
//     chunk of the whole array, so `offsets` is already GLOBAL; blocks 1..31 are
//     no-ops.  The two-level warp scan is otherwise unchanged and stays bit-exact.
//     Pass 2 consequently consumes the GLOBAL `offsets` directly: no per-block
//     exclusive scan, no block-boundary correction — each range-start writes its
//     own global slot.  Observable outputs (rangeStart/rangeEnd/totalNumPairs) are
//     identical to the CUDA; only the intermediate `offsets` encoding differs.
//   - PxU64 particleId compared as its uint2 halves; the 0xffff.. sentinel is
//     uint2(~0,~0).  Every BDA Ptr<> load sits behind a uniform-scalar guard
//     (workIndex < tNumContacts) so nothing hoists out of a zero-trip loop.

import pxgSlangScan;

// PxgParticleSystemKernelBlockDim/GridDim (PxgParticleSystemCoreKernelIndices.h):
//   BOUNDCELLUPDATE = 512 block / 32 grid   (reorder)
//   ACCUMULATE_DELTA = 512 block / 32 grid  (find start/end)
static const uint RC_BLOCK = 512u;   // BOUNDCELLUPDATE block
static const uint RC_GRID = 32u;     // BOUNDCELLUPDATE grid
static const uint AD_BLOCK = 512u;   // ACCUMULATE_DELTA block
static const uint AD_GRID = 32u;     // ACCUMULATE_DELTA grid
static const uint WARP = 32u;
static const uint AD_NUM_WARPS = AD_BLOCK / WARP;  // 16

// PxgParticlePrimitiveContact (PxgParticleSystem.h, PX_ALIGN(16), 32 bytes) —
// float4 normal_pen @0, PxU64 rigidId @16, PxU64 particleId @24.  The reorder
// copies whole structs; the range-find reads only particleId.
struct PsContact
{
    float4 normal_pen; // @0
    uint2 rigidId;     // @16
    uint2 particleId;  // @24 -> 32
}

// PxgFemOtherContactInfo (PxgFEMCore.h:128, PX_ALIGN(16), 16 bytes) —
// PxU64 pairInd0 @0 (the PARTICLE side: systemId<<32 | particleIndex),
// PxU32 pairInd1 @8 (the FEM id), PxU32 rigidMatInd_isInCollision @12.
// The range-find reads only pairInd0; the accumulate reads it once more.
struct FemOtherContactInfo
{
    uint2 pairInd0;                  // @0
    uint pairInd1;                   // @8
    uint rigidMatInd_isInCollision;  // @12 -> 16
}

// ===========================================================================
// 1. ps_reorderPrimitiveContactsLaunch (cudaParticleSystem.cu:1687)
//    CUDA ABI: (const PxgParticlePrimitiveContact* particleContacts,
//               const PxI32* numParticleContacts, const PxU32* remapByRigid,
//               const PxU32* remapByParticle,
//               PxgParticlePrimitiveContact* sortedContactsByRigid,
//               PxgParticlePrimitiveContact* sortedContactsByParticle)
//    -> 6 device pointers = {8,8,8,8,8,8} = 48-byte blob.
// ===========================================================================
struct PsReorderParams
{
    uint64_t particleContacts;         // @0
    uint64_t numParticleContacts;      // @8  PxI32* device pointer
    uint64_t remapByRigid;             // @16
    uint64_t remapByParticle;          // @24
    uint64_t sortedContactsByRigid;    // @32 output
    uint64_t sortedContactsByParticle; // @40 output
}

[[vk::push_constant]]
ConstantBuffer<PsReorderParams> gRc;

[shader("compute")]
[numthreads(RC_BLOCK, 1, 1)]
void ps_reorderPrimitiveContactsLaunch(uint3 groupThreadId : SV_GroupThreadID,
                                       uint3 groupId : SV_GroupID)
{
    const uint tid = groupThreadId.x;
    const uint blockId = groupId.x;

    const uint totalNumContacts = Ptr<uint>(gRc.numParticleContacts)[0];

    // CUDA operator precedence preserved verbatim: ((n + B*G - 1)/B) * G.
    const uint numIternations = (totalNumContacts + RC_BLOCK * RC_GRID - 1u) / RC_BLOCK * RC_GRID;

    Ptr<PsContact> src = Ptr<PsContact>(gRc.particleContacts);
    Ptr<uint> remapByRigid = Ptr<uint>(gRc.remapByRigid);
    Ptr<uint> remapByParticle = Ptr<uint>(gRc.remapByParticle);
    Ptr<PsContact> outByRigid = Ptr<PsContact>(gRc.sortedContactsByRigid);
    Ptr<PsContact> outByParticle = Ptr<PsContact>(gRc.sortedContactsByParticle);

    for (uint i = 0; i < numIternations; ++i)
    {
        const uint index = tid + blockId * RC_BLOCK + i * RC_BLOCK * RC_GRID;
        if (index >= totalNumContacts)
            return;

        outByParticle[index] = src[remapByParticle[index]];
        outByRigid[index] = src[remapByRigid[index]];
    }
}

// ===========================================================================
// 2. ps_findStartEndParticleFirstLaunch (particlesystem.cu:2277)
//    CUDA ABI: (PxgParticleSystem* particleSystems, PxgParticlePrimitiveContact*
//               sortedContacts, PxU32* numContacts, PxU32* blockOffsets,
//               PxU32* offsets) -> 5 device pointers = {8,8,8,8,8} = 40-byte blob.
//    particleSystems is passed for ABI parity but never dereferenced.
// ===========================================================================
struct PsFindFirstParams
{
    uint64_t particleSystems; // @0  (unused, ABI parity)
    uint64_t sortedContacts;  // @8
    uint64_t numContacts;     // @16
    uint64_t blockOffsets;    // @24 output
    uint64_t offsets;         // @32 output
}

[[vk::push_constant]]
ConstantBuffer<PsFindFirstParams> gF1;

groupshared uint2 gsPid1[AD_BLOCK + 1];
groupshared uint gsWarpAccum1[WARP];   // CUDA sizes [WARP_SIZE]; only [0..15] used
groupshared uint gsBlockAccum1;

[shader("compute")]
[numthreads(AD_BLOCK, 1, 1)]
void ps_findStartEndParticleFirstLaunch(uint3 groupThreadId : SV_GroupThreadID,
                                        uint3 groupId : SV_GroupID)
{
    const uint tid = groupThreadId.x;
    const uint blockId = groupId.x;
    const uint lane = tid & (WARP - 1u);
    const uint warpIndex = tid / WARP;

    const uint tNumContacts = Ptr<uint>(gF1.numContacts)[0];

    // A global running range index cannot be produced across 32 concurrent
    // workgroups in one dispatch (no grid-wide sync), so the whole scan is folded
    // onto block 0, which carries the single global accumulator gsBlockAccum1
    // across every 512-contact chunk of the array.  Blocks 1..31 are no-ops (the
    // early return is uniform over the workgroup, so no barrier is skipped by only
    // some lanes).  See the header for the full rationale.
    if (blockId != 0u)
        return;

    const uint nbTotalChunks = (tNumContacts + AD_BLOCK - 1u) / AD_BLOCK;

    Ptr<PsContact> contacts = Ptr<PsContact>(gF1.sortedContacts);
    Ptr<uint> offsets = Ptr<uint>(gF1.offsets);
    Ptr<uint> blockOffsets = Ptr<uint>(gF1.blockOffsets);

    if (tid == 0)
        gsBlockAccum1 = 0u;

    GroupMemoryBarrierWithGroupSync();

    for (uint i = 0; i < nbTotalChunks; ++i)
    {
        const uint workIndex = i * AD_BLOCK + tid;

        uint2 particleId = uint2(0xffffffffu, 0xffffffffu);
        if (workIndex < tNumContacts)
        {
            particleId = contacts[workIndex].particleId;
            gsPid1[tid + 1] = particleId;
            if (workIndex > 0u && tid == 0u)
                gsPid1[0] = contacts[workIndex - 1].particleId;
        }

        GroupMemoryBarrierWithGroupSync();

        uint isRangeStart = 0u;
        if (workIndex < tNumContacts)
        {
            const uint2 prev = gsPid1[tid];
            if (workIndex == 0u || any(particleId != prev))
                isRangeStart = 1u;
        }

        // within-warp inclusive scan of isRangeStart (block-uniform); the CUDA
        // offset = popc(mask & threadMask) - (1 - isRangeStart) == inclusive - 1.
        const uint inclusive = warp32InclusiveScan(tid, isRangeStart);
        const int offset = int(inclusive) - 1;

        if (lane == (WARP - 1u))
            gsWarpAccum1[warpIndex] = inclusive; // popc(mask) = warp total

        const uint prevBlockAccumulator = gsBlockAccum1;

        GroupMemoryBarrierWithGroupSync();

        // cross-warp exclusive scan: warp 0 feeds the real per-warp totals, other
        // warps feed 0, so only warp 0's segment result is meaningful (== CUDA's
        // single-warp warpScan run inside `if (warpIndex == 0)`).
        const uint tOffset = (warpIndex == 0u && lane < AD_NUM_WARPS) ? gsWarpAccum1[lane] : 0u;
        const uint incl2 = warp32InclusiveScan(tid, tOffset);
        const uint output = incl2 - tOffset;
        if (warpIndex == 0u)
        {
            if (lane == (WARP - 1u))
                gsBlockAccum1 += (output + tOffset); // inclusive total of the block
            gsWarpAccum1[lane] = output;
        }

        GroupMemoryBarrierWithGroupSync();

        if (workIndex < tNumContacts)
            offsets[workIndex] = uint(offset) + prevBlockAccumulator + gsWarpAccum1[warpIndex];
    }

    if (tid == 0)
        blockOffsets[0] = gsBlockAccum1; // global total range-start count (== totalNumPairs)
}

// ===========================================================================
// 3. ps_findStartEndParticleSecondLaunch (particlesystem.cu:2402)
//    CUDA ABI: (PxgParticlePrimitiveContact* sortedContacts, PxU32* numContacts,
//               PxU32* blockOffsets, PxU32* offsets, PxU32* totalNumPairs,
//               PxU32* rangeStart, PxU32* rangeEnd)
//    -> 7 device pointers = {8,8,8,8,8,8,8} = 56-byte blob.
// ===========================================================================
struct PsFindSecondParams
{
    uint64_t sortedContacts;  // @0
    uint64_t numContacts;     // @8
    uint64_t blockOffsets;    // @16
    uint64_t offsets;         // @24
    uint64_t totalNumPairs;   // @32 output
    uint64_t rangeStart;      // @40 output
    uint64_t rangeEnd;        // @48 output
}

[[vk::push_constant]]
ConstantBuffer<PsFindSecondParams> gF2;

groupshared uint2 gsPid2[AD_BLOCK + 1];

[shader("compute")]
[numthreads(AD_BLOCK, 1, 1)]
void ps_findStartEndParticleSecondLaunch(uint3 groupThreadId : SV_GroupThreadID,
                                         uint3 groupId : SV_GroupID)
{
    const uint tid = groupThreadId.x;
    const uint blockId = groupId.x;

    const uint tNumContacts = Ptr<uint>(gF2.numContacts)[0];

    Ptr<PsContact> contacts = Ptr<PsContact>(gF2.sortedContacts);
    Ptr<uint> offsets = Ptr<uint>(gF2.offsets);
    Ptr<uint> rangeStart = Ptr<uint>(gF2.rangeStart);
    Ptr<uint> rangeEnd = Ptr<uint>(gF2.rangeEnd);

    // `offsets` is already a GLOBAL running range index (pass 1 folds the whole
    // scan onto one workgroup), so pass 2 no longer needs the per-block exclusive
    // scan of blockOffsets nor the block-boundary correction the CUDA used to lift
    // within-block offsets to global.  Each range-start writes its own global slot
    // directly; the range that ends at a start is the previous contact's global
    // range index.  Kept parallel over the 32 blocks (each reads any global slot).
    const uint totalBlockRequired = (tNumContacts + (AD_BLOCK - 1u)) / AD_BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (AD_GRID - 1u)) / AD_GRID;

    for (uint i = 0; i < numIterationPerBlock; ++i)
    {
        const uint workIndex = i * AD_BLOCK + tid + numIterationPerBlock * blockId * AD_BLOCK;

        uint2 particleId = uint2(0u, 0u);
        if (workIndex < tNumContacts)
        {
            particleId = contacts[workIndex].particleId;
            gsPid2[tid + 1] = particleId;
            if (workIndex > 0u && tid == 0u)
                gsPid2[0] = contacts[workIndex - 1].particleId;
        }

        GroupMemoryBarrierWithGroupSync();

        if (workIndex < tNumContacts)
        {
            const uint2 prev = gsPid2[tid];
            if (workIndex == 0u || any(particleId != prev))
            {
                rangeStart[offsets[workIndex]] = workIndex;

                if (workIndex > 0u)
                    rangeEnd[offsets[workIndex - 1u]] = workIndex;
            }

            if (workIndex == tNumContacts - 1u)
                rangeEnd[offsets[workIndex]] = workIndex + 1u;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (blockId == 0u && tid == 0u)
    {
        // total pair count = global range index of the last contact + 1.
        const uint totalPairs = (tNumContacts > 0u) ? (offsets[tNumContacts - 1u] + 1u) : 0u;
        Ptr<uint>(gF2.totalNumPairs)[0] = totalPairs;
    }
}

// ===========================================================================
// 4. ps_findStartEndFEMFirstLaunch (particlesystem.cu:2519)
//    The FEM twin of kernel 2.  CUDA ABI: (PxgParticleSystem* particleSystems,
//    PxgFemOtherContactInfo* sortedContactsInfo, PxU32* numContacts,
//    PxU32* blockOffsets, PxU32* offsets) -> 5 pointers = 40-byte blob.
//    particleSystems is passed for ABI parity but never dereferenced, exactly as
//    in the particle twin.
//    Launched from PxgSoftBodyCore.cpp:2356 and PxgFEMClothCore.cpp:426, both
//    with block SB_ACCUMULATE_DELTA (512) and grid SB_ACCUMULATE_DELTA (32).
//
//    The whole restructuring of the particle twin applies unchanged and for the
//    same reason: pass 1 is folded onto block 0 so that `offsets` comes out
//    GLOBAL rather than within-block (see the header).  Only the record type and
//    the field offset differ from ps_findStartEndParticleFirstLaunch.
// ===========================================================================
struct PsFindFemFirstParams
{
    uint64_t particleSystems;    // @0  (unused, ABI parity)
    uint64_t sortedContactsInfo; // @8
    uint64_t numContacts;        // @16
    uint64_t blockOffsets;       // @24 output
    uint64_t offsets;            // @32 output -> 40
}

[[vk::push_constant]]
ConstantBuffer<PsFindFemFirstParams> gFF1;

groupshared uint2 gsFemPid1[AD_BLOCK + 1];
groupshared uint gsFemWarpAccum1[WARP];   // CUDA sizes [WARP_SIZE]; only [0..15] used
groupshared uint gsFemBlockAccum1;

[shader("compute")]
[numthreads(AD_BLOCK, 1, 1)]
void ps_findStartEndFEMFirstLaunch(uint3 groupThreadId : SV_GroupThreadID,
                                   uint3 groupId : SV_GroupID)
{
    const uint tid = groupThreadId.x;
    const uint blockId = groupId.x;
    const uint lane = tid & (WARP - 1u);
    const uint warpIndex = tid / WARP;

    const uint tNumContacts = Ptr<uint>(gFF1.numContacts)[0];

    // See kernel 2: the global running range index is folded onto block 0.
    //
    // This early return is an efficiency guard, not a correctness one, and the
    // P9.48a sweep proved it: the loop below indexes `i * AD_BLOCK + tid` with no
    // blockId term and gsFemBlockAccum1 is per-workgroup, so every block would
    // compute the SAME values and write them to the SAME slots.  Removing the
    // return leaves the outputs bit-identical and only burns 31 workgroups of
    // duplicate work -- the one mutant of this kernel that cannot be killed, and
    // it cannot be killed because there is nothing there to kill.
    if (blockId != 0u)
        return;

    const uint nbTotalChunks = (tNumContacts + AD_BLOCK - 1u) / AD_BLOCK;

    Ptr<FemOtherContactInfo> infos = Ptr<FemOtherContactInfo>(gFF1.sortedContactsInfo);
    Ptr<uint> offsets = Ptr<uint>(gFF1.offsets);
    Ptr<uint> blockOffsets = Ptr<uint>(gFF1.blockOffsets);

    if (tid == 0)
        gsFemBlockAccum1 = 0u;

    GroupMemoryBarrierWithGroupSync();

    for (uint i = 0; i < nbTotalChunks; ++i)
    {
        const uint workIndex = i * AD_BLOCK + tid;

        // The .cu leaves this UNINITIALISED here (its particle twin seeds the
        // 0xffff.. sentinel); every read is behind the same guard that assigns
        // it, so the sentinel is the defined-behaviour spelling of a dead value.
        uint2 particleId = uint2(0xffffffffu, 0xffffffffu);
        if (workIndex < tNumContacts)
        {
            particleId = infos[workIndex].pairInd0;
            gsFemPid1[tid + 1] = particleId;
            if (workIndex > 0u && tid == 0u)
                gsFemPid1[0] = infos[workIndex - 1].pairInd0;
        }

        GroupMemoryBarrierWithGroupSync();

        uint isRangeStart = 0u;
        if (workIndex < tNumContacts)
        {
            const uint2 prev = gsFemPid1[tid];
            if (workIndex == 0u || any(particleId != prev))
                isRangeStart = 1u;
        }

        const uint inclusive = warp32InclusiveScan(tid, isRangeStart);
        const int offset = int(inclusive) - 1;

        if (lane == (WARP - 1u))
            gsFemWarpAccum1[warpIndex] = inclusive; // popc(mask) = warp total

        const uint prevBlockAccumulator = gsFemBlockAccum1;

        GroupMemoryBarrierWithGroupSync();

        const uint tOffset = (warpIndex == 0u && lane < AD_NUM_WARPS) ? gsFemWarpAccum1[lane] : 0u;
        const uint incl2 = warp32InclusiveScan(tid, tOffset);
        const uint output = incl2 - tOffset;
        if (warpIndex == 0u)
        {
            if (lane == (WARP - 1u))
                gsFemBlockAccum1 += (output + tOffset);
            gsFemWarpAccum1[lane] = output;
        }

        GroupMemoryBarrierWithGroupSync();

        if (workIndex < tNumContacts)
            offsets[workIndex] = uint(offset) + prevBlockAccumulator + gsFemWarpAccum1[warpIndex];
    }

    if (tid == 0)
        blockOffsets[0] = gsFemBlockAccum1; // global total range-start count
}

// ===========================================================================
// 5. ps_findStartEndFEMSecondLaunch (particlesystem.cu:2644)
//    The FEM twin of kernel 3.  CUDA ABI: (PxgFemOtherContactInfo*
//    sortedContactInfo, PxU32* numContacts, PxU32* blockOffsets, PxU32* offsets,
//    PxU32* totalNumPairs, PxU32* rangeStart, PxU32* rangeEnd)
//    -> 7 pointers = 56-byte blob.
//    Launched from PxgSoftBodyCore.cpp:2416 and PxgFEMClothCore.cpp:456.
//
//    The trailing GroupMemoryBarrierWithGroupSync() below has NO counterpart in
//    the .cu -- see the header, upstream candidate.  Its particle twin has it and
//    says why; without it the port's own reuse of gsFemPid2 across iterations is
//    a race, so it stays.
// ===========================================================================
struct PsFindFemSecondParams
{
    uint64_t sortedContactInfo; // @0
    uint64_t numContacts;       // @8
    uint64_t blockOffsets;      // @16
    uint64_t offsets;           // @24
    uint64_t totalNumPairs;     // @32 output
    uint64_t rangeStart;        // @40 output
    uint64_t rangeEnd;          // @48 output -> 56
}

[[vk::push_constant]]
ConstantBuffer<PsFindFemSecondParams> gFF2;

groupshared uint2 gsFemPid2[AD_BLOCK + 1];

[shader("compute")]
[numthreads(AD_BLOCK, 1, 1)]
void ps_findStartEndFEMSecondLaunch(uint3 groupThreadId : SV_GroupThreadID,
                                    uint3 groupId : SV_GroupID)
{
    const uint tid = groupThreadId.x;
    const uint blockId = groupId.x;

    const uint tNumContacts = Ptr<uint>(gFF2.numContacts)[0];

    Ptr<FemOtherContactInfo> infos = Ptr<FemOtherContactInfo>(gFF2.sortedContactInfo);
    Ptr<uint> offsets = Ptr<uint>(gFF2.offsets);
    Ptr<uint> rangeStart = Ptr<uint>(gFF2.rangeStart);
    Ptr<uint> rangeEnd = Ptr<uint>(gFF2.rangeEnd);

    // `offsets` is already GLOBAL (pass 1 folds the scan onto one workgroup), so
    // no per-block exclusive scan and no block-boundary correction -- identical
    // to the particle twin's treatment.
    const uint totalBlockRequired = (tNumContacts + (AD_BLOCK - 1u)) / AD_BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (AD_GRID - 1u)) / AD_GRID;

    for (uint i = 0; i < numIterationPerBlock; ++i)
    {
        const uint workIndex = i * AD_BLOCK + tid + numIterationPerBlock * blockId * AD_BLOCK;

        uint2 particleId = uint2(0u, 0u);
        if (workIndex < tNumContacts)
        {
            particleId = infos[workIndex].pairInd0;
            gsFemPid2[tid + 1] = particleId;
            if (workIndex > 0u && tid == 0u)
                gsFemPid2[0] = infos[workIndex - 1].pairInd0;
        }

        GroupMemoryBarrierWithGroupSync();

        if (workIndex < tNumContacts)
        {
            const uint2 prev = gsFemPid2[tid];
            if (workIndex == 0u || any(particleId != prev))
            {
                rangeStart[offsets[workIndex]] = workIndex;

                if (workIndex > 0u)
                    rangeEnd[offsets[workIndex - 1u]] = workIndex;
            }

            if (workIndex == tNumContacts - 1u)
                rangeEnd[offsets[workIndex]] = workIndex + 1u;
        }

        // NOT in the .cu (upstream candidate); required here, see the header.
        GroupMemoryBarrierWithGroupSync();
    }

    if (blockId == 0u && tid == 0u)
    {
        const uint totalPairs = (tNumContacts > 0u) ? (offsets[tNumContacts - 1u] + 1u) : 0u;
        Ptr<uint>(gFF2.totalNumPairs)[0] = totalPairs;
    }
}
