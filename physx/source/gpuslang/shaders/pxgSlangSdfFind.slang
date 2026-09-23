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
// P9.34b: the TRIANGLE gather's buffer element is a uint2 -- the primitive index
// and its subdivision level.  With SUPPORT_SUBDIVISION 0 the level is always 0,
// but the CUDA still stores and reads both halves, so both are carried here.
groupshared uint2 gSdfIndices2[SDFF_TARGET_COUNT];
groupshared uint gSdfOffset;

// TWO SLOTS HERE STAND FOR SEVERAL IN THE CUDA, AND BOTH ARE SAFE -- checked, not
// assumed (P9.83b-5, 14/n; flagged by a read-only map of the mesh-mesh kernel).
//
//   gSdfOffset is the tet, triangle AND vertex gathers' resume slot, where the
//   CUDA has three separate `__shared__ PxU32 offset` (sdfCollision.cuh:569, :624,
//   :695).  Every engine includer runs exactly one gather kind
//   (femClothSdfMeshCgBda: triangles, psSdfMidphaseBda: vertices,
//   softBodySdfMidphaseBda: tets), and each gather seeds the slot from startIndex
//   before its first use, so no two resume points are ever live together.
//
//   gSdfPerWarpCount is both the two-argument scan's perWarpCountS
//   (reduction.cuh:628) and sdfGlobalScanExclusive's, where the CUDA gives the
//   global scan its own (reduction.cuh:674).  The writes are ordered anyway: the
//   gathers put a barrier before every sdfAddIndexToBuffer* call, and
//   sdfGlobalScanExclusive opens and closes with one, so no lane-0 write to a slot
//   can land while another warp is still reading the previous scan's value.
//
// A kernel that ran two gather kinds, or called a scan without a barrier after a
// previous one's reads, would break either argument.  None does today.

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

// ===========================================================================
// findInterestingTrianglesA<32, 1024>, sdfCollision.cuh:620-686 -- P9.34b.
//
// The TRIANGLE sibling of findInterestingVertices above, and the outer loop of
// cloth_sdfMeshContactGenLaunch.  Same resumable-stream protocol, same block scan,
// same offset handshake -- and three differences that matter.
//
// ---------------------------------------------------------------------------
//  T1. THE LOOP HAS A COUNT GUARD AND THE VERTEX ONE DOES NOT.  Thirty lines
//      apart in the same header:
//          vertices   for (i = startIndex; (i < nbVertices); )
//          triangles  for (i = startIndex; (i < nbPrimitives || ...) &&
//                          sharedIndicesCount < TargetCount; )
//      The vertex sweep visits every remaining primitive on every call and lets
//      addIndexToBuffer saturate; the triangle sweep STOPS as soon as the buffer
//      is full.
//
//      BE PRECISE ABOUT WHAT THAT CHANGES, because it is tempting to call it a
//      correctness trap and it is not.  Once the buffer is full,
//      `nbElementsInBuffer` is saturated at TargetCount, so every later
//      writeOffset is >= TargetCount: no slot is written, the
//      `writeOffset == TargetCount - 1` handshake cannot fire, and the return
//      saturates again.  The gathered set, the resume point and the round
//      structure are therefore IDENTICAL with and without the guard.  What
//      differs is WORK: without it the block keeps sampling the SDF for the whole
//      remainder of the mesh and throwing the results away, every round.
//
//      So the guard is an efficiency property, and the only thing that can
//      observe it is which primitives the function TOUCHED -- which is exactly
//      what the fixture's echo buffer records, and how it is checked.
//
//  T2. THE CULL TEST IS A BOUNDING SPHERE, NOT A POINT.  The vertex form asks
//      `distance(v) < tolerance`; this one asks
//          distance(centroid) < boundingSphereRadius / min(scale) + tolerance
//      where the radius is `sqrt(max(|v0-c|^2, max(|v1-c|^2, |v2-c|^2)))` -- the
//      same nested-max-then-one-sqrt shape as doTriangleSDFCollision's
//      `difference`, and wrong if spelled as a max of three lengths.  The DIVISION
//      by the smallest scale component happens to the radius alone, before the
//      tolerance is added.
//
//  T3. THE SUBDIVISION MACHINERY IS DEAD.  `SUPPORT_SUBDIVISION` is 0 at the one
//      call site (femClothPrimitives.cu:2306), so nbSubdivisionIndices is always
//      0, `PxMin(TargetCount, 4 * 0)` is 0, and the index reduces to
//      `i + threadIdx.x` with a stride of blockDim.x -- identical to the vertex
//      form.  The subdivision level is stored as a literal 0 rather than dropped,
//      because the consumer reads `.y` and passes it to getTriangleVertices.
// ===========================================================================

uint sdfAddIndexToBuffer2(uint tid, bool threadEmitsElement, uint2 index, uint nbElementsInBuffer)
{
    uint total;
    const uint exclusiveSum = sdfBlockScanExclusive(tid, threadEmitsElement, total);

    if (threadEmitsElement)
    {
        const uint writeOffset = nbElementsInBuffer + exclusiveSum;

        if (writeOffset < SDFF_TARGET_COUNT)
            gSdfIndices2[writeOffset] = index;

        // S4, and note the resume point is `index.x + 1` -- the FIRST word of the
        // pair, which the CUDA spells `((PxU32*)(&index))[0] + 1`.
        if (writeOffset == (SDFF_TARGET_COUNT - 1u))
            gSdfOffset = index.x + 1u;
    }

    return min(total + nbElementsInBuffer, SDFF_TARGET_COUNT);
}

// T2, factored out so the *Ex trace and the live path cannot disagree about it.
float sdfTriangleCullDistance(uint64_t meshInd, uint64_t meshVerts, uint ind,
                              float3 scale0, float4 scale0Rot, float3 aToBp, float4 aToBq,
                              float3 scale1, float4 scale1Rot, SparseSdfTexture sdfMesh1,
                              out float radiusOverScale, out float3 centroidVertexSpace)
{
    // getTriangleVertices (sdfCollision.cuh:521): vertex2Shape by scale0, then the
    // a-to-b transform.  The P8.96c CLAMP RULE applies to the index, not the load.
    const uint4 tri = Ptr<uint4>(meshInd)[ind];
    Ptr<float4> vp = Ptr<float4>(meshVerts);
    const float3 v0 = sdfTransformPoint(aToBp, aToBq, sdfVertex2Shape(vp[tri.x].xyz, scale0, scale0Rot));
    const float3 v1 = sdfTransformPoint(aToBp, aToBq, sdfVertex2Shape(vp[tri.y].xyz, scale0, scale0Rot));
    const float3 v2 = sdfTransformPoint(aToBp, aToBq, sdfVertex2Shape(vp[tri.z].xyz, scale0, scale0Rot));

    const float third = 1.0f / 3.0f;
    const float3 centroid = (v0 + v1 + v2) * third;

    centroidVertexSpace = sdfShape2Vertex(centroid, scale1, scale1Rot);
    const float distInSDFSpace1 = sdfDistance(sdfMesh1, centroidVertexSpace);

    const float3 e0 = v0 - centroid, e1 = v1 - centroid, e2 = v2 - centroid;
    float r = sqrt(max(dot(e0, e0), max(dot(e1, e1), dot(e2, e2))));
    r = r / min(scale1.x, min(scale1.y, scale1.z));
    radiusOverScale = r;

    return distInSDFSpace1;
}

uint sdfFindInterestingTrianglesEx(uint tid, uint nbPrimitives, uint64_t meshInd, uint64_t meshVerts,
                                   float3 scale0, float4 scale0Rot, float3 scale1, float4 scale1Rot,
                                   float tolerance, SparseSdfTexture sdfMesh1,
                                   float3 aToBp, float4 aToBq, inout uint startIndex,
                                   uint64_t echoOut)
{
    if (tid == 0u)
        gSdfOffset = startIndex;

    uint sharedIndicesCount = 0u;

    // T1: the count guard, which the vertex sibling does not have.
    for (uint i = startIndex; i < nbPrimitives && sharedIndicesCount < SDFF_TARGET_COUNT; )
    {
        // T3: with the subdivision machinery dead this is `i + tid`, stride blockDim.
        const uint ind = i + tid;
        i += SDFF_BLOCK_DIM;

        const uint safeInd = ind < nbPrimitives ? ind : 0u;
        float radiusOverScale;
        float3 centroidVtx;
        const float distInSDFSpace1 =
            sdfTriangleCullDistance(meshInd, meshVerts, safeInd, scale0, scale0Rot, aToBp, aToBq,
                                    scale1, scale1Rot, sdfMesh1, radiusOverScale, centroidVtx);

        // T2, and strict, exactly as the vertex form's tolerance test is.
        const bool addToBuffer = (ind < nbPrimitives) &&
                                 (distInSDFSpace1 < (radiusOverScale + tolerance));

        if (echoOut != 0UL && ind < nbPrimitives)
            Ptr<float4>(echoOut)[ind] = float4(centroidVtx, distInSDFSpace1);

        GroupMemoryBarrierWithGroupSync();
        sharedIndicesCount = sdfAddIndexToBuffer2(tid, addToBuffer, uint2(ind, 0u),
                                                  sharedIndicesCount);
    }

    GroupMemoryBarrierWithGroupSync();

    if (startIndex != gSdfOffset)
        startIndex = gSdfOffset;
    else
        startIndex = nbPrimitives;

    return sharedIndicesCount;
}

uint sdfFindInterestingTriangles(uint tid, uint nbPrimitives, uint64_t meshInd, uint64_t meshVerts,
                                 float3 scale0, float4 scale0Rot, float3 scale1, float4 scale1Rot,
                                 float tolerance, SparseSdfTexture sdfMesh1,
                                 float3 aToBp, float4 aToBq, inout uint startIndex)
{
    return sdfFindInterestingTrianglesEx(tid, nbPrimitives, meshInd, meshVerts, scale0, scale0Rot,
                                         scale1, scale1Rot, tolerance, sdfMesh1, aToBp, aToBq,
                                         startIndex, 0UL);
}

// ---------------------------------------------------------------------------
// globalScanExclusive<NbWarps>, reduction.cuh:669-687 -- P9.34c.
//
// The block scan above plus ONE atomic add by thread 0 and the two barriers
// around it.  psSdfMidphaseBda carried a private copy of this from P9.12h; it
// became the second consumer's problem at P9.34c, so it lives here now and takes
// the counter's address as a parameter rather than reaching for a push constant.
// ---------------------------------------------------------------------------
groupshared uint gSdfGlobalOffset;

uint sdfGlobalScanExclusive(uint tid, bool threadContributesElement, uint64_t counterAddr)
{
    GroupMemoryBarrierWithGroupSync();

    uint threadBlockTotalSum;
    const uint indexInThreadBlock = sdfBlockScanExclusive(tid, threadContributesElement,
                                                          threadBlockTotalSum);
    // Only ONE thread per block performs an atomic add.
    if (tid == 0u)
        InterlockedAdd(Ptr<uint>(counterAddr)[0], threadBlockTotalSum, gSdfGlobalOffset);

    GroupMemoryBarrierWithGroupSync();
    return indexInThreadBlock + gSdfGlobalOffset;
}


// ---------------------------------------------------------------------------
// findInterestingTets<32, 1024>, sdfCollision.cuh:565-616 -- P9.35b.
//
// The TET gather, and the third member of this family after the vertex form
// (P9.12e) and the triangle form (P9.34b).  It is closest to the VERTEX one --
// same plain-uint buffer, same addIndexToBuffer, same resume protocol -- and its
// cull test is the TRIANGLE one's, widened to four corners.  What is worth
// writing down is where it differs from BOTH:
//
//  U1. THERE IS NO COUNT GUARD.  The loop condition is `i < nbTets` alone, where
//      the triangle form carries `&& sharedIndicesCount < TargetCount`.  This is
//      the exact inverse of P9.34b's T1, and it is upstream's asymmetry, not a
//      transcription slip -- the two functions sit forty lines apart in the same
//      header.  P9.34b established that the guard is output-neutral and costs
//      only work (once the buffer saturates, every later write offset is >=
//      TargetCount and the resume handshake cannot fire again), so the same
//      argument says its ABSENCE here is output-neutral too.  Transcribed as
//      written; the fixture measures the difference in triangles TOUCHED, which
//      is the only place it shows.
//
//  U2. THE SOFTBODY SIDE HAS NO SCALE AT ALL.  The triangle form applies
//      vertex2Shape with scale0 before the a-to-b transform; here the tet
//      vertices go straight through `aToB.transform(...)`.  Not a unit scale that
//      happens to be the identity -- there is no call.
//
//  U3. THE RADIUS DIVIDE TAKES NO ABSOLUTE VALUE: `PxMin(scale1.scale.x, ...)`.
//      And the KERNEL that calls this divides its own contact distance by
//      `scale.abs().minElement()` -- WITH the absolute value.  Two spellings of
//      the same idea, forty lines apart, inside one call chain, disagreeing on
//      sign.  The mirror image of P9.34c's S3, where the cloth kernel omits the
//      abs and its particle sibling takes it.
//
//  U4. THE SURFACE HINT IS DEAD BY CALLER.  `surfaceHint` defaults to NULL and
//      the only call site (softbodySoftbodyMidPhase.cu:1202) passes nothing --
//      the argument is commented out there.  So `!surfaceHint || surfaceHint[ind]`
//      is always true.  Not transcribed, and named here so the next reader does
//      not go looking for the parameter.
//
// TWO THINGS THE SWEEP LEFT STANDING, and both are proofs rather than gaps:
//
//  * `sqrt(max(a2, b2, ...))` against `max(sqrt(a2), ...)` is an EQUIVALENT.
//    sqrt is monotone and correctly rounded, so it commutes with a maximum --
//    P9.34a established this for the triangle form and it carries over unchanged.
//
//  * SWAPPING TWO CORNERS is an equivalent HERE and only here.  This function
//    reads the four vertices through exactly two symmetric aggregates: the
//    centroid `((a + b) + c) + d`, where float addition is commutative so
//    `(a + b)` and `(b + a)` are the same bits, and the radius, a nest of
//    `max` over the four squared offsets, where max is commutative too.  The
//    DESCENT is a different matter -- there the cascade tests the corners in
//    order and the barycentric blend weights them individually, and its
//    corresponding mutants both fall.  Two functions over the same four points,
//    one order-blind and one not.
// ---------------------------------------------------------------------------

// U3, factored out so the *Ex trace and the live path cannot disagree about it.
float sdfTetCullDistance(uint64_t tetInd, uint64_t tetVerts, uint ind,
                         float3 aToBp, float4 aToBq,
                         float3 scale1, float4 scale1Rot, SparseSdfTexture sdfMesh1,
                         out float radiusOverScale, out float3 centroidVertexSpace)
{
    const uint4 tet = Ptr<uint4>(tetInd)[ind];
    Ptr<float4> vp = Ptr<float4>(tetVerts);
    // U2: no vertex2Shape on this side.
    const float3 a = sdfTransformPoint(aToBp, aToBq, vp[tet.x].xyz);
    const float3 b = sdfTransformPoint(aToBp, aToBq, vp[tet.y].xyz);
    const float3 c = sdfTransformPoint(aToBp, aToBq, vp[tet.z].xyz);
    const float3 d = sdfTransformPoint(aToBp, aToBq, vp[tet.w].xyz);

    const float3 centroid = (a + b + c + d) * 0.25f;

    centroidVertexSpace = sdfShape2Vertex(centroid, scale1, scale1Rot);
    const float distInSDFSpace1 = sdfDistance(sdfMesh1, centroidVertexSpace);

    const float3 e0 = a - centroid, e1 = b - centroid, e2 = c - centroid, e3 = d - centroid;
    // Nested max of SQUARES, one sqrt, and the nesting is max(max, max).
    float r = sqrt(max(max(dot(e0, e0), dot(e1, e1)), max(dot(e2, e2), dot(e3, e3))));
    r = r / min(scale1.x, min(scale1.y, scale1.z));     // U3: NO abs
    radiusOverScale = r;

    return distInSDFSpace1;
}

uint sdfFindInterestingTetsEx(uint tid, uint nbTets, uint64_t tetInd, uint64_t tetVerts,
                              float3 scale1, float4 scale1Rot,
                              float tolerance, SparseSdfTexture sdfMesh1,
                              float3 aToBp, float4 aToBq, inout uint startIndex,
                              uint64_t echoOut, uint64_t touchedOut)
{
    if (tid == 0u)
        gSdfOffset = startIndex;

    uint sharedIndicesCount = 0u;

    // U1: `i < nbTets` ALONE -- no count guard.
    for (uint i = startIndex; i < nbTets; )
    {
        const uint ind = i + tid;
        i += SDFF_BLOCK_DIM;

        const uint safeInd = ind < nbTets ? ind : 0u;
        float radiusOverScale;
        float3 centroidVtx;
        const float distInSDFSpace1 =
            sdfTetCullDistance(tetInd, tetVerts, safeInd, aToBp, aToBq,
                               scale1, scale1Rot, sdfMesh1, radiusOverScale, centroidVtx);

        const bool addToBuffer = (ind < nbTets) &&
                                 (distInSDFSpace1 < (radiusOverScale + tolerance));

        if (echoOut != 0UL && ind < nbTets)
            Ptr<float4>(echoOut)[ind] = float4(centroidVtx, distInSDFSpace1);
        // U1 is observable ONLY as work: which tets were sampled at all.  Without
        // this the missing count guard is invisible, because the gathered set, the
        // resume point and the round structure are all identical with or without it.
        if (touchedOut != 0UL && ind < nbTets)
            Ptr<uint>(touchedOut)[ind] = 1u;

        GroupMemoryBarrierWithGroupSync();
        sharedIndicesCount = sdfAddIndexToBuffer(tid, addToBuffer, ind, sharedIndicesCount);
    }

    GroupMemoryBarrierWithGroupSync();

    if (startIndex != gSdfOffset)
        startIndex = gSdfOffset;
    else
        startIndex = nbTets;

    return sharedIndicesCount;
}

uint sdfFindInterestingTets(uint tid, uint nbTets, uint64_t tetInd, uint64_t tetVerts,
                            float3 scale1, float4 scale1Rot,
                            float tolerance, SparseSdfTexture sdfMesh1,
                            float3 aToBp, float4 aToBq, inout uint startIndex)
{
    return sdfFindInterestingTetsEx(tid, nbTets, tetInd, tetVerts, scale1, scale1Rot,
                                    tolerance, sdfMesh1, aToBp, aToBq, startIndex, 0UL, 0UL);
}

#endif // PXG_SLANG_SDF_FIND
