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

// pxgSlangSdfFind.slang -- P9.12e, the block-wide candidate gather.
//
// Ports three things that stack:
//   threadBlockScanExclusive<NbWarps>  reduction.cuh:597-623
//   addIndexToBuffer<PxU32, ...>       sdfCollision.cuh:496-520
//   findInterestingVertices<32, 1024>  sdfCollision.cuh:690-726
//
// This is the OUTER loop of ps_sdfMeshCollisonLaunch: before the kernel does any
// contact work it sweeps the whole particle array, keeps the ones whose
// PxSdfDistance is under the cull tolerance, and hands the collision body a
// packed list of at most 1024 of them.  When more than 1024 qualify it records
// where to resume and the kernel calls it again -- so the gather is a RESUMABLE
// stream, and that protocol is the substance of this tranche.
//
// A #include HEADER, NOT AN IMPORTED MODULE, and that is forced rather than
// chosen: the warp emulation's scratch array is sized by the PXG_WARP_BLOCK
// macro, which an `import` cannot carry across.  The includer sets
// PXG_WARP_BLOCK to 1024 and includes pxgSlangWarpInc.slang first, so there is
// exactly one 1024-entry scratch array in the translation unit.  Same
// arrangement npArtiMassMatrix.slang already uses.
//
// ---------------------------------------------------------------------------
// WHY 1024 THREADS IS A NEW SHAPE FOR THIS PORT
// ---------------------------------------------------------------------------
// Every particle kernel so far has been ONE or TWO warps per workgroup (32 or
// 64 threads).  This one is 32 emulated warps in a single 1024-thread
// workgroup, which is the maximum llvmpipe allows
// (maxComputeWorkGroupInvocations = 1024, measured by pxg-caps), and it needs a
// TWO-LEVEL scan: a ballot per warp, then a scan over the 32 per-warp counts.
//
// On the emulation path every warp primitive is a workgroup barrier, so EVERY
// CALL SITE MUST BE WORKGROUP-UNIFORM -- stronger than CUDA's warp convergence.
// That holds here by construction and it is worth saying why: the loop bound is
// `i < nbVertices` with `i` advancing by blockDim.x, so every thread runs the
// same number of iterations regardless of what it found.  The CUDA's own
// __syncthreads() inside the loop is what makes that true for it too.
//
// ---------------------------------------------------------------------------
// THE TRAPS
// ---------------------------------------------------------------------------
//  S1. The per-warp count array is indexed TWO DIFFERENT WAYS in the same
//      function: once by LANE (`perWarpCountS[threadIndexInWarp]`, to load the
//      value each lane scans) and once by WARP (`perWarpCountS[warpIndex]`, to
//      subtract the caller's own warp's count).  Swapping them compiles and is
//      wrong for every thread outside warp 0.
//  S2. `carryExclusiveScan = shfl(total, warpIndex) - perWarpCountS[warpIndex]`
//      -- an INCLUSIVE scan turned exclusive by subtracting the own value.
//      Using an exclusive scan there and still subtracting double-counts; using
//      the inclusive value without subtracting shifts every warp by its own
//      count.
//  S3. `totalSum = shfl(total, 31)` -- lane THIRTY-ONE, not NbWarps-1.  At the
//      NbWarps == 32 this kernel uses those coincide, and the fixture cannot
//      tell them apart; the engine's spelling is kept because at NbWarps < 32
//      lane 31 is still right (lanes past NbWarps contribute 0) while
//      NbWarps-1 would be too, and only lane 31 is what the CUDA wrote.
//  S4. addIndexToBuffer records the resume point from the thread whose
//      writeOffset is exactly TargetCount-1 -- the LAST SLOT, not the first
//      overflow -- and it records `index + 1`, one PAST the vertex that filled
//      the buffer.  `((PxU32*)(&index))[0] + 1` in the CUDA is the generic-T
//      spelling of that; at T == PxU32 it is just `index + 1`.
//  S5. The return is `min(total + nbElementsInBuffer, TargetCount)` --
//      SATURATING.  That is what makes S4 fire exactly once: after the buffer
//      is full every writeOffset is >= TargetCount, so no later thread can
//      match TargetCount-1 and overwrite the resume point.
//  S6. findInterestingVertices' loop has NO count guard.  Its sibling
//      findInterestingTrianglesA (sdfCollision.cuh:631) has
//      `&& sharedIndicesCount < TargetCount`; this one does not, so it walks
//      EVERY remaining vertex on every call and simply drops what does not fit.
//      Transcribed as-is -- and it is the reason S5 is load-bearing rather than
//      merely tidy.
//  S7. The stride is `blockDim.x` (1024) and the index is `i + threadIdx.x`
//      with threadIdx.x the BLOCK-wide id, not a lane.  A port that strides by
//      the warp size processes 1/32 of the array per iteration and still
//      terminates, so it fails quietly rather than loudly.
//  S8. The "did it overflow?" test is `startIndex != offset` -- comparing the
//      shared resume slot against the value it was SEEDED with.  No flag, no
//      count comparison.  If nothing overflowed, offset still equals the
//      original startIndex and the function reports `nbVertices`, which is what
//      terminates the caller's loop.
//  S9. `offset` is seeded by thread 0 from startIndex BEFORE the loop, and
//      startIndex is an inout that every thread carries.  The seed and the
//      final read are separated from every write by the loop's own barrier and
//      the one after it.
// S10. The point is transformed TWICE and in this order: `aToB.transform(p)`
//      into trimesh-local space, then `shape2Vertex(.., scale, rotation)` into
//      the SDF's vertex space.  shape2Vertex is a per-component DIVIDE by the
//      scale between two quaternion rotations (cudaNpCommon.h:284-296), not a
//      multiply by a reciprocal.
// S11. The emit predicate is `dist < tolerance`, strict.
// S12. THE BUFFER ORDER IS DETERMINISTIC AND ASCENDING.  writeOffset is
//      nbElementsInBuffer + the block-exclusive rank, and the rank runs in
//      thread order, so the buffer holds the qualifying indices sorted.  That
//      is not an accident to be preserved by luck -- it is what lets the
//      fixture predict the exact contents rather than a set.
//
// TWO DELIBERATE DIVERGENCES, recorded rather than hidden:
//   * The CUDA passes the output buffer in as `PxU32* sharedIndices`.  Slang
//     has no groupshared pointer parameter, so it is declared here.  There is
//     exactly one per block in the engine too.
//   * The CUDA templates over <NbWarps, TargetCount> and T.  Only
//     <32, 1024> at T == PxU32 exists in the engine's SDF particle path, so
//     that instantiation is spelled out.  Both numbers are named constants and
//     the fixture asserts the block size against them.

#ifndef PXG_SLANG_SDF_FIND
#define PXG_SLANG_SDF_FIND

// findInterestingVertices<32, 1024>, and blockDim.x == 1024 from the host
// launch (PxgNarrowphaseCore.cpp:3903).  All three must agree: NB_WARPS *
// WARP_SIZE == BLOCK_DIM.
static const uint SDFF_NB_WARPS = 32u;
static const uint SDFF_TARGET_COUNT = 1024u;
static const uint SDFF_BLOCK_DIM = 1024u;

// reduction.cuh:628 declares perWarpCountS inside the function; the engine's
// `sharedIndices` is the caller's triangleIndicesS[1024]
// (particleSystemMeshMidphase.cu:394); `offset` is addIndexToBuffer's shared
// resume slot (sdfCollision.cuh:695).
groupshared uint gSdfPerWarpCount[SDFF_NB_WARPS];
groupshared uint gSdfIndices[SDFF_TARGET_COUNT];
groupshared uint gSdfOffset;

// ---------------------------------------------------------------------------
// threadBlockScanExclusive<NbWarps>(bool, totalSum, perWarpCountS)
// reduction.cuh:597-623.
// ---------------------------------------------------------------------------
uint sdfBlockScanExclusive(uint tid, bool threadContributesElement, out uint totalSum)
{
    const uint threadIndexInWarp = tid & (PXG_WARP_SIZE - 1u);
    const uint warpIndex = tid >> 5;

    const uint ballotMask = pxgBallotSync(PXG_FULL_MASK, tid, threadContributesElement);
    const uint nbInteresting = pxgPopc(ballotMask);   // elements emitted per warp

    if (threadIndexInWarp == 0u)
        gSdfPerWarpCount[warpIndex] = nbInteresting;

    GroupMemoryBarrierWithGroupSync();

    // S1: indexed by LANE here...
    const uint warpCount = threadIndexInWarp < SDFF_NB_WARPS ? gSdfPerWarpCount[threadIndexInWarp] : 0u;

    // The inclusive cumulative sum per warp; lane 31 holds the block total.
    const uint total = pxgWarpScanAddInclusiveU(PXG_FULL_MASK, tid, warpCount, 5u);

    // S1 again: ...and by WARP here.  S2: inclusive minus own == exclusive.
    const uint carryExclusiveScan = pxgShflSyncU(PXG_FULL_MASK, tid, total, int(warpIndex), PXG_WARP_SIZE)
                                    - gSdfPerWarpCount[warpIndex];

    // S3: lane 31.
    totalSum = pxgShflSyncU(PXG_FULL_MASK, tid, total, 31, PXG_WARP_SIZE);

    return carryExclusiveScan + pxgWarpRankInMask(ballotMask, threadIndexInWarp);
}

// ---------------------------------------------------------------------------
// addIndexToBuffer<PxU32, NbWarps, TargetCount>, sdfCollision.cuh:496-520.
// ---------------------------------------------------------------------------
uint sdfAddIndexToBuffer(uint tid, bool threadEmitsElement, uint index, uint nbElementsInBuffer)
{
    uint total;
    const uint exclusiveSum = sdfBlockScanExclusive(tid, threadEmitsElement, total);

    if (threadEmitsElement)
    {
        const uint writeOffset = nbElementsInBuffer + exclusiveSum;

        if (writeOffset < SDFF_TARGET_COUNT)
            gSdfIndices[writeOffset] = index;

        // S4: the LAST slot, and the resume point is one PAST it.
        if (writeOffset == (SDFF_TARGET_COUNT - 1u))
            gSdfOffset = index + 1u;
    }

    // S5: saturating.
    return min(total + nbElementsInBuffer, SDFF_TARGET_COUNT);
}

// The per-vertex test, factored out so the *Ex trace and the live path cannot
// disagree about it.
float sdfVertexCullDistance(uint64_t positions, uint ind, float3 aToBp, float4 aToBq,
                            float3 scale1, float4 scale1Rot, SparseSdfTexture sdfMesh1,
                            out float3 vertexSpacePoint)
{
    // S10, and the P8.96c CLAMP RULE: the caller's `ind < nbVertices` guard is
    // applied to the INDEX, not to the load, because llvmpipe compiles a
    // guarded uniform BDA load to an unconditional load plus a select.
    const float3 v = sdfTransformPoint(aToBp, aToBq, Ptr<float4>(positions)[ind].xyz);
    vertexSpacePoint = sdfShape2Vertex(v, scale1, scale1Rot);
    return sdfDistance(sdfMesh1, vertexSpacePoint);
}

// ---------------------------------------------------------------------------
// findInterestingVertices<32, 1024>, sdfCollision.cuh:690-726.
//
// `echoOut`, when non-zero, receives float4(vertexSpacePoint, cullDistance) at
// [ind] for every vertex the sweep touches.  It exists because the fixture
// cannot evaluate Sample() on the host: the gather expectation is built from
// the distances THIS function reports, so what is under test is the gathering,
// not the distance (that is P9.12d's claim, already measured).  The point
// itself IS host-computable and is compared bit-for-bit.
// ---------------------------------------------------------------------------
uint sdfFindInterestingVerticesEx(uint tid, uint nbVertices, uint64_t positions,
                                  float3 scale1, float4 scale1Rot, float tolerance,
                                  SparseSdfTexture sdfMesh1, float3 aToBp, float4 aToBq,
                                  inout uint startIndex, uint64_t echoOut)
{
    // S9.
    if (tid == 0u)
        gSdfOffset = startIndex;

    uint sharedIndicesCount = 0u;

    // S6: no count guard -- every remaining vertex is visited on every call.
    // S7: the stride is the BLOCK size.
    for (uint i = startIndex; i < nbVertices; )
    {
        const uint ind = i + tid;
        i += SDFF_BLOCK_DIM;

        const uint safeInd = ind < nbVertices ? ind : 0u;
        float3 vtx;
        const float distInSDFSpace1 = sdfVertexCullDistance(positions, safeInd, aToBp, aToBq,
                                                            scale1, scale1Rot, sdfMesh1, vtx);
        // S11: strict.
        const bool addToBuffer = (ind < nbVertices) && (distInSDFSpace1 < tolerance);

        if (echoOut != 0UL && ind < nbVertices)
            Ptr<float4>(echoOut)[ind] = float4(vtx, distInSDFSpace1);

        GroupMemoryBarrierWithGroupSync();
        sharedIndicesCount = sdfAddIndexToBuffer(tid, addToBuffer, ind, sharedIndicesCount);
    }

    GroupMemoryBarrierWithGroupSync();

    // S8.
    if (startIndex != gSdfOffset)
        startIndex = gSdfOffset;
    else
        startIndex = nbVertices;

    return sharedIndicesCount;
}

uint sdfFindInterestingVertices(uint tid, uint nbVertices, uint64_t positions,
                                float3 scale1, float4 scale1Rot, float tolerance,
                                SparseSdfTexture sdfMesh1, float3 aToBp, float4 aToBq,
                                inout uint startIndex)
{
    return sdfFindInterestingVerticesEx(tid, nbVertices, positions, scale1, scale1Rot, tolerance,
                                        sdfMesh1, aToBp, aToBq, startIndex, 0UL);
}

#endif // PXG_SLANG_SDF_FIND
