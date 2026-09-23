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

// npTriTriSharedProbe.slang -- P9.83b-4.
//
// THIS SHADER COMPUTES NOTHING.  It exists to MEASURE the Workgroup-storage
// footprint of doTriangleTriangleCollision (convexMeshMidphase.cu:972-1231)
// before that kernel is written, because the answer decides whether it can be
// one entry point at all: llvmpipe's maxComputeSharedMemorySize is 32768 and
// Vulkan applies it PER PIPELINE.
//
// P9.83b-pre projected 34816 bytes and called the kernel blocked on it.  P9.83b-2
// then argued the blocker dissolves once the contact reduction is serial, because
// the warp-emulated form's gPxgWarpScratch (4096 B) disappears: 34816 - 4096 =
// 30720.  That is an ARGUMENT, and this tranche has just spent three corrections
// learning what arguments are worth here (see the survivor list in
// pxgSlangContactReduceSerial.slang).  So the declarations below are the CUDA's,
// one for one, and gpuslang.groupshared reads the number out of the compiled
// SPIR-V.  A projection that agrees with a measurement is a measurement; a
// projection that stands alone is a guess with a number in it.
//
// The CUDA declarations being mirrored, in order (convexMeshMidphase.cu:993-1013):
//
//   Patch    patchesS[MAX_MESH_MESH_PATCHES2]      128 patches
//   PxVec4   tempPointsS[33 * MESH_MESH_CONTACT_LIMIT]   33 * 6 = 198
//   PxU32    countersS[32]
//   PxReal   shMinSepS[32]
//   uint2    triangleIndicesS[1024]
//   uint2    trianglesToSubdivideS[256]
//   PxU32    numTrianglesToSubdivideS
//   char     sSdfTexture[sizeof(SparseSDFTexture)]
//   bool     anyKeepS
//
// plus two that doTriangleTriangleCollision does NOT declare, because the
// helpers it calls SELF-ALLOCATE them -- and reading only the kernel's own
// declarations therefore undercounts:
//
//   PxU32    offset            findInterestingTrianglesA, sdfCollision.cuh:624
//   PxU32    perWarpCountS[32] threadBlockScanExclusive<32>, reduction.cuh:628,
//                              which is the overload addIndexToBuffer calls
//
// NOT declared here, deliberately: gPxgWarpScratch.  Its absence is the whole
// claim under test -- it is what the warp-emulated reduction needs and the serial
// one does not.

// P9.83b-5 (11/n): the warp-emulation scratch, at this kernel's block size.  The
// third entry point below is the only one that calls into it, so the first two
// measurements are unchanged -- Workgroup storage is charged per entry point by
// reachability, which P9.83b-5 (10/n) measured directly.
#define PXG_WARP_BLOCK 1024
#include "pxgSlangWarpInc.slang"

import pxgSlangSdfTexture;

static const uint MAX_MESH_MESH_PATCHES2 = 128u;
static const uint MESH_MESH_CONTACT_LIMIT = 6u;

// convexMeshMidphase.cu:432-438.  116 bytes in CUDA; SPIR-V Workgroup layout
// rounds the float3 to a 16-byte slot, so the measurement below is the number
// that matters, not the CUDA sizeof.
struct Patch
{
    float4 keepPoints[MESH_MESH_CONTACT_LIMIT];   // w = separation
    float3 normal;
    float deepest;
    uint pointCount;
};

groupshared Patch gPatches[MAX_MESH_MESH_PATCHES2];
groupshared float4 gTempPoints[33 * MESH_MESH_CONTACT_LIMIT];
groupshared uint gCounters[32];
groupshared float gMinSep[32];
groupshared uint2 gTriangleIndices[1024];
groupshared uint2 gTrianglesToSubdivide[256];
groupshared uint gNumTrianglesToSubdivide;
groupshared SparseSdfTexture gSdfTexture;
groupshared uint gAnyKeep;

// TWO MORE BUFFERS THAT ARE NOT DECLARED IN doTriangleTriangleCollision, and
// that the first version of this probe therefore missed.  They are SELF-ALLOCATED
// inside the helpers it calls, which is exactly why reading only the kernel's own
// declarations undercounts:
//
//   findInterestingTrianglesA (sdfCollision.cuh:624)  `__shared__ PxU32 offset;`
//   threadBlockScanExclusive<32> (reduction.cuh:628)  `__shared__ PxU32
//       perWarpCountS[NbWarps];` -- the two-argument overload allocates its own,
//       and addIndexToBuffer calls exactly that one (sdfCollision.cuh:502).
//
// A probe that measures nine of eleven buffers is the kind of partial measurement
// that reads like a complete one.
groupshared uint gFindOffset;
groupshared uint gScanPerWarpCount[32];

struct ProbeParams
{
    uint numItems;
    uint pad0, pad1, pad2;
}

[[vk::push_constant]] ConstantBuffer<ProbeParams> gP;
RWStructuredBuffer<uint> gOut;   // binding 0

// Every declaration above must be READ and WRITTEN, or SPIR-V's OpEntryPoint
// interface list will not name it and the measurement will read low.  P9.83b-1
// established that an UNUSED Workgroup variable costs nothing; this probe would
// be worthless if it accidentally relied on that.
void probeBody(uint3 dtid, uint3 gtid)
{
    const uint t = gtid.x;

    if (t == 0u)
    {
        gNumTrianglesToSubdivide = 0u;
        gAnyKeep = 0u;
        gSdfTexture.sdfBoxLower = float3(0, 0, 0);
        gSdfTexture.sdfDx = 1.0f;
        gSdfTexture.sdfBoxHigher = float3(1, 1, 1);
        gSdfTexture.invSdfDx = 1.0f;
        gSdfTexture.texture = 0u;
        gSdfTexture.textureSubgrids = 0u;
        gSdfTexture.coarseSize = uint3(1, 1, 1);
        gSdfTexture.subgridSize = 0u;
        gSdfTexture.subgridStartSlots = 0UL;
        gSdfTexture.fineToCoarse = 1.0f;
        gSdfTexture.sdfDims = uint4(1, 1, 1, 0);
        gSdfTexture.subgridsMinSdfValue = 0.0f;
        gSdfTexture.subgridsSdfValueRange = 1.0f;
    }

    if (t == 0u) gFindOffset = 0u;
    if (t < 32u) { gCounters[t] = t; gMinSep[t] = float(t); gScanPerWarpCount[t] = t; }
    for (uint p = t; p < MAX_MESH_MESH_PATCHES2; p += 1024u)
    {
        for (uint k = 0; k < MESH_MESH_CONTACT_LIMIT; ++k)
            gPatches[p].keepPoints[k] = float4(float(k), 0, 0, 0);
        gPatches[p].normal = float3(0, 1, 0);
        gPatches[p].deepest = 0.0f;
        gPatches[p].pointCount = 0u;
    }
    for (uint q = t; q < 33u * MESH_MESH_CONTACT_LIMIT; q += 1024u)
        gTempPoints[q] = float4(float(q), 0, 0, 0);
    gTriangleIndices[t] = uint2(t, 0u);
    for (uint r = t; r < 256u; r += 1024u)
        gTrianglesToSubdivide[r] = uint2(r, 0u);

    GroupMemoryBarrierWithGroupSync();

    if (dtid.x < gP.numItems)
    {
        uint acc = gCounters[t & 31u] + uint(gMinSep[t & 31u])
                 + gTriangleIndices[t].x + gNumTrianglesToSubdivide + gAnyKeep
                 + gFindOffset + gScanPerWarpCount[t & 31u]
                 + uint(gPatches[t & 127u].keepPoints[0].x)
                 + uint(gPatches[t & 127u].normal.y) + uint(gPatches[t & 127u].deepest)
                 + gPatches[t & 127u].pointCount
                 + uint(gTempPoints[t % 198u].x)
                 + gTrianglesToSubdivide[t & 255u].x
                 + uint(gSdfTexture.sdfBoxLower.x) + uint(gSdfTexture.sdfDx)
                 + uint(gSdfTexture.sdfBoxHigher.x) + uint(gSdfTexture.invSdfDx)
                 + gSdfTexture.texture + gSdfTexture.textureSubgrids
                 + gSdfTexture.coarseSize.x + gSdfTexture.subgridSize
                 + uint(gSdfTexture.subgridStartSlots) + uint(gSdfTexture.fineToCoarse)
                 + gSdfTexture.sdfDims.x + uint(gSdfTexture.subgridsMinSdfValue)
                 + uint(gSdfTexture.subgridsSdfValueRange);
        gOut[dtid.x] = acc;
    }
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void triTriSharedProbe(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    probeBody(dtid, gtid);
}

// ---------------------------------------------------------------------------
// P9.83b-4 (3/n): THE WARP-STAGING QUESTION P9.83b-2 DID NOT ASK
// ---------------------------------------------------------------------------
// P9.83b-2 argued the 34816-byte blocker dissolves once the contact reduction is
// serial, because the warp-emulated form's gPxgWarpScratch goes away.  That
// covers ONE of the THREE contactReduce2 call sites.
//
//   :472  inside reduceContactsSingleWarp, under `if (warpIndex == 0)` -- one
//         warp, so the serial body of P9.83b-5 replaces it outright.
//   :810  inside `if (anyKeepS)`, block-uniform, called by ALL 32 WARPS, each on
//         its own 32 lanes.
//   :931  inside `if (countersS[minIBlock] != 0)`, same shape.
//
// The last two still need a per-warp reduction, and a serial one needs the warp's
// 32 lanes somewhere addressable.  Two schemes, and the difference decides the
// design, so both are measured rather than reasoned about:
//
//   ALL WARPS AT ONCE   32 warps * 32 lanes * 16 B  -- every warp reduces in
//                       parallel, one barrier per call site
//   ONE WARP AT A TIME  32 lanes * 16 B             -- 32 sequential reduces per
//                       call site, 32 barriers, but a tiny buffer
//
// BOTH WERE MEASURED, as two entry points over the same probe body:
//
//   triTriSharedProbeWideStage     46684 bytes -- OVER THE LIMIT BY 13916
//   triTriSharedProbeNarrowStage   30812 bytes -- fits, 1956 bytes of headroom
//
// So the design is FORCED: the two warp-parallel reduce sites must be serialised
// one warp at a time.  The wide entry point has been DELETED rather than left
// here, because an entry point that is over the limit by construction makes
// gpuslang.groupshared fail for the whole corpus; its number is recorded above
// and in tools/groupshared_known.txt, which is where a number belongs.
// ---------------------------------------------------------------------------

groupshared float4 gWarpStageSmall[32];          // one warp at a time

[shader("compute")]
[numthreads(1024, 1, 1)]
void triTriSharedProbeNarrowStage(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    probeBody(dtid, gtid);
    if (gtid.x < 32u) gWarpStageSmall[gtid.x] = float4(float(gtid.x), 0, 0, 0);
    GroupMemoryBarrierWithGroupSync();
    if (dtid.x < gP.numItems)
        gOut[dtid.x] += uint(gWarpStageSmall[gtid.x & 31u].x);
}

// ---------------------------------------------------------------------------
// P9.83b-5 (11/n): THE BUFFER THE FIRST TWO MEASUREMENTS STILL DID NOT CARRY
// ---------------------------------------------------------------------------
// The note at the head of this file says gPxgWarpScratch is left out deliberately,
// "because it is what the warp-emulated reduction needs and the serial one does
// not".  That is true of the REDUCTION and FALSE OF THE KERNEL.
// doTriangleTriangleCollision has block collectives that have nothing to do with
// contactReduce2, and every one of them goes through the emulation:
//
//   :788   __ballot_sync(FULL_MASK, accept)
//   :812   threadBlockCountAndBroadcast(accept2, countersS)
//   :820   threadBlockScanExclusive<32>(threadEmitsElement, ...)
//   :881   __ballot_sync(FULL_MASK, contact.candidateContact)
//   :885   minIndex(minSep, candidateMask, minSep)        -- five shuffles
//   :899   minIndex(minSep, FULL_MASK, minSep)            -- five more
//   :1202  threadBlockCountAndBroadcast(contact.candidateContact, countersS)
//
// They are all block-uniform, so the emulation is legal -- and its scratch is
// sized by PXG_WARP_BLOCK, which for this 1024-thread block is 1024 uints.  The
// native subgroup path would make it vanish, and it is NOT available here:
// npWarpNative.slang records that CI's llvmpipe has a fixed subgroup size of 8,
// while the 32-lane CUDA semantics need requiredSubgroupSize == 32.  On this
// device the 4096 bytes are not optional.
//
// MEASURED, as an entry point that calls a warp primitive so the scratch lands in
// its SPIR-V interface list:
//
//   triTriSharedProbeWarpEmu   34908 bytes -- OVER THE LIMIT BY 2140
//
// The entry point has been DELETED for the same reason the wide-stage one was: an
// entry that is over the limit by construction makes gpuslang.groupshared fail for
// the whole corpus, so its number belongs in this note and in
// tools/groupshared_known.txt rather than in a shader that cannot be built into a
// pipeline.  Leaving gPxgWarpScratch out of the first two measurements was the
// same kind of partial measurement P9.83b-4 already corrected once, when it found
// `offset` and `perWarpCountS` missing: a probe that measures eleven of twelve
// buffers reads exactly like a complete one.
//
// ---------------------------------------------------------------------------
// P9.83b-5 (11/n): THE SPILL, MEASURED BESIDE THE PACKING TRICKS
// ---------------------------------------------------------------------------
// triTriSharedProbeWarpEmu is OVER the limit, so something has to go.  The three
// reclamations that keep every buffer in Workgroup storage were added up before
// they were measured, and together they are 2176 against a 2140-byte deficit --
// 36 bytes of headroom, which is a number nobody should build on:
//
//   gPatches 16384 -> 14848   pointCount out of the struct, so 6*float4 + float4
//                             lands exactly on 112 instead of rounding 120 to 128
//   gWarpStageSmall 512 -> 0  the per-warp stage is live strictly BEFORE
//                             tempPointsS is written (:810 vs :828), so it can be
//                             the first 32 slots of gTempPoints
//   gScanPerWarpCount 128 -> 0 findInterestingTrianglesA's scan finishes before
//                             countersS is first used in the same iteration
//
// CORRECTION (P9.83b-5, 14/n): the third one is a RACE in the CUDA's shape, and
// "finishes before" was the wrong test.  The two-argument scan overload
// (reduction.cuh:625-630) owns ONE static perWarpCountS per instantiation, and
// the same instantiation is reached by addToRefinementBuffer
// (sdfCollision.cuh:750), called at :1198 -- directly before
// threadBlockCountAndBroadcast(.., countersS) at :1202.  The scan reads all 32
// slots after its only barrier (reduction.cuh:612, reads :614/:618) and has no
// trailing one; the count writes sharedMem[warpIndex] from lane 0 (:583-584) with
// no barrier before it.  Aliased, a warp that leaves the scan early overwrites a
// slot another warp is still reading.  In THIS port the ballot emulation inside
// warpCountAndBroadcast is itself a workgroup barrier and would order the two --
// safety borrowed from the emulation, gone with a native subgroup path.  The
// 32732 below stays a correct byte count (one extra barrier between :1198 and
// :1202 would make the reclamation valid and costs no bytes); the reasoning was
// what was wrong.
//
// This entry point measures the OTHER answer: leave every buffer alone and put
// patchesS in a device-side per-workgroup scratch allocation instead.  It is the
// CUDA's largest shared array by a factor of two and the only one that is written
// by one warp and read by all -- so it is also the one whose move costs a
// barrier upgrade, from GroupMemoryBarrierWithGroupSync to
// AllMemoryBarrierWithGroupSync, at every point where the CUDA has __syncthreads.
// That upgrade is the price, and it is a real one: it is stated here so the choice
// is made on two measured numbers rather than on which trick was thought of first.

[shader("compute")]
[numthreads(1024, 1, 1)]
void triTriSharedProbeSpilled(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    const uint t = gtid.x;

    if (t == 0u)
    {
        gNumTrianglesToSubdivide = 0u;
        gAnyKeep = 0u;
        gFindOffset = 0u;
        gSdfTexture.sdfBoxLower = float3(0, 0, 0);
        gSdfTexture.sdfDx = 1.0f;
        gSdfTexture.sdfBoxHigher = float3(1, 1, 1);
        gSdfTexture.invSdfDx = 1.0f;
        gSdfTexture.texture = 0u;
        gSdfTexture.textureSubgrids = 0u;
        gSdfTexture.coarseSize = uint3(1, 1, 1);
        gSdfTexture.subgridSize = 0u;
        gSdfTexture.subgridStartSlots = 0UL;
        gSdfTexture.fineToCoarse = 1.0f;
        gSdfTexture.sdfDims = uint4(1, 1, 1, 0);
        gSdfTexture.subgridsMinSdfValue = 0.0f;
        gSdfTexture.subgridsSdfValueRange = 1.0f;
    }
    if (t < 32u) { gCounters[t] = t; gMinSep[t] = float(t); gScanPerWarpCount[t] = t; }
    for (uint q = t; q < 33u * MESH_MESH_CONTACT_LIMIT; q += 1024u)
        gTempPoints[q] = float4(float(q), 0, 0, 0);
    gTriangleIndices[t] = uint2(t, 0u);
    for (uint r = t; r < 256u; r += 1024u)
        gTrianglesToSubdivide[r] = uint2(r, 0u);
    if (t < 32u) gWarpStageSmall[t] = float4(float(t), 0, 0, 0);

    GroupMemoryBarrierWithGroupSync();

    const uint ballot = pxgBallotSync(PXG_FULL_MASK, t, (t & 3u) == 0u);
    const uint rank = pxgWarpScanAddInclusiveU(PXG_FULL_MASK, t, ballot & 1u, 5u);

    if (dtid.x < gP.numItems)
    {
        gOut[dtid.x] = gCounters[t & 31u] + uint(gMinSep[t & 31u])
                     + gTriangleIndices[t].x + gNumTrianglesToSubdivide + gAnyKeep
                     + gFindOffset + gScanPerWarpCount[t & 31u]
                     + uint(gTempPoints[t % 198u].x)
                     + gTrianglesToSubdivide[t & 255u].x
                     + uint(gWarpStageSmall[t & 31u].x)
                     + uint(gSdfTexture.sdfBoxLower.x) + uint(gSdfTexture.sdfDx)
                     + uint(gSdfTexture.sdfBoxHigher.x) + uint(gSdfTexture.invSdfDx)
                     + gSdfTexture.texture + gSdfTexture.textureSubgrids
                     + gSdfTexture.coarseSize.x + gSdfTexture.subgridSize
                     + uint(gSdfTexture.subgridStartSlots) + uint(gSdfTexture.fineToCoarse)
                     + gSdfTexture.sdfDims.x + uint(gSdfTexture.subgridsMinSdfValue)
                     + uint(gSdfTexture.subgridsSdfValueRange)
                     + pxgPopc(ballot) + rank;
    }
}

// ---------------------------------------------------------------------------
// P9.83b-5 (12/n): THE THREE RECLAMATIONS, BUILT RATHER THAN ADDED UP -- AND
// THEN OVER THE LIMIT ANYWAY
// ---------------------------------------------------------------------------
// 2176 against a 2140-byte deficit is arithmetic on three numbers each read off a
// different measurement, and this tranche had already been wrong twice about what
// a shader costs.  So the packing answer was BUILT as its own entry point and the
// checker read it out of the SPIR-V:
//
//   1. Patch loses pointCount.  6*float4 + float4 is 112 and needs no rounding;
//      with pointCount inside, 120 rounds up to 128.  The counts move to their own
//      uint[128].  16384 -> 14848.
//   2. The per-warp reduce stage becomes the first 32 slots of tempPointsS.  Legal
//      because the stage is dead before tempPointsS is first written -- :810 runs
//      the reduce, :828-835 fill tempPointsS.  512 -> 0.
//   3. threadBlockScanExclusive's self-allocated perWarpCountS becomes countersS.
//      findInterestingTrianglesA's scan finishes before countersS is first used in
//      the same iteration.  128 -> 0.
//
//   triTriSharedProbePacked   32732 bytes -- fits, 36 bytes spare
//
// AND THEN IT DOES NOT FIT, because 36 bytes is smaller than the port's OWN
// requirement.  contactReduce2 at :810 and :931 is a WARP-scoped call: each of the
// 32 warps gets its reduceMask back in its own lanes' registers.  A serialised
// per-warp reduce computes that mask on ONE thread, so it has to reach the warp's
// 32 lanes through memory -- one uint per warp, 128 bytes that the CUDA does not
// need and the port cannot do without.  Measured with it:
//
//   triTriSharedProbePacked   32860 bytes -- OVER THE LIMIT BY 92
//
// So the spill is FORCED, the same way the narrow staging was forced in 4/n: not
// chosen because it is tidier, but because the alternative was built and measured
// and does not fit.  The 36-byte margin was never a margin -- it was smaller than
// the first buffer the port had to add to it, and adding numbers from three
// measurements would never have shown that.
//
// Both entry points are DELETED.  The packed one is over the limit by
// construction, which fails gpuslang.groupshared for the whole corpus; the
// 32732-byte variant without gReduceMask is not worth keeping either, because it
// measures a design that cannot be finished.  The numbers are in
// tools/groupshared_known.txt.
// ---------------------------------------------------------------------------
