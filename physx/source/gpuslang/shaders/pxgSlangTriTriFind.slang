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

// pxgSlangTriTriFind -- P9.83b-8 (3/n).  A TEXTUAL INCLUDE, not a module: it
// declares Workgroup storage, exactly like pxgSlangTriTriBlock and pxgSlangSdfFind.
//
//   The refinement half of the mesh-mesh kernel's gather, which the cloth port of
//   the same function (sdfFindInterestingTrianglesEx) left out as dead code (T3
//   there):
//
//     ttfFindInterestingTriangles   findInterestingTrianglesA<32, 1024>
//                                   (sdfCollision.cuh:619-687) WITH the
//                                   nbSubdivisionIndices / sharedSubdivisionIndices
//                                   input, the child fetch and elevation, the shifted
//                                   index and stride, and the `|| n > 0` loop term
//     ttfAddIndexToBuffer           addIndexToBuffer<uint2, 32, 1024> (:498-520),
//                                   writing the kernel's index buffer
//     ttfTriangleCullSphere         the cull's texture-free half (:661-670), on the
//                                   SUB-triangle
//     ttfAddToRefinementBuffer      addToRefinementBuffer<32, 256> (:741-763)
//
// Include order: pxgSlangWarpInc (PXG_WARP_BLOCK 1024), pxgSlangSdfFind (the scan
// and its gSdfPerWarpCount, the resume slot gSdfOffset), pxgSlangTriTriBlock
// (gTtbScratch), and `import pxgSlangTriTriGeom` in the includer.
//
// ---------------------------------------------------------------------------
// STORAGE -- three CUDA buffers, one of them borrowed
// ---------------------------------------------------------------------------
//   triangleIndicesS[1024] (uint2, cu:1007) is gTtbScratch, the 8192-byte alias
//     P9.83b-5 (15a/n) measured: entry k is components 2(k&1), 2(k&1)+1 of
//     gTtbScratch[k >> 1].  The halves are written by DIFFERENT threads (k and k^1),
//     so they are stored one scalar component at a time; a vector read-modify-write
//     of the uint4 would lose one of the two.
//   trianglesToSubdivideS[256] (uint2, cu:1010) and numTrianglesToSubdivideS
//     (cu:1011) are declared here: gTtfSubdivide / gTtfNumSubdivide.
//   The gather's `offset` (sdfCollision.cuh:624) is gSdfOffset, and both scans'
//     perWarpCountS (reduction.cuh:628, one static per instantiation, shared by
//     addIndexToBuffer and addToRefinementBuffer) is gSdfPerWarpCount -- the same
//     sharing pxgSlangSdfFind's note already argues.  The mesh-mesh kernel runs
//     only the triangle gather, so gSdfOffset has one owner.
//
// ---------------------------------------------------------------------------
// THE SDF SAMPLE IS A HOOK
// ---------------------------------------------------------------------------
// The cull samples the SDF at the sub-triangle's centroid.  That sample is
// PxSdfDistance, already certified (gpuslang.sdfCollision); what is new here is
// everything around it.  So the one texture read goes through TTF_SDF_DISTANCE(t,
// ind, sub, p), which defaults to sdfDistance(t, p).  The fixture defines it
// before including this file -- to a hash of (ind, sub) that lets it drive the
// control flow into any corner, or to an analytic plane whose float arithmetic the
// host can mirror bit for bit, which checks the point the cull samples.  The
// DEFAULT line is therefore not covered by the fixture: it is the kernel fixture's
// obligation (P9.83b-8, 15).
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
// F1  The children come FIRST.  Threads tid < 4n take sharedSubdivisionIndices[tid
//     >> 2] and elevate its sub-id by child tid & 3; the rest take
//     `i + tid - min(1024, 4n)`, and i advances by `blockDim - min(1024, 4n)`.
//     With n = 256 that stride is ZERO: the whole block is children.  n is then
//     zeroed -- the children are all consumed by the first pass.
// F2  The loop runs while `(i < nbPrimitives || n > 0) && count < 1024`: a pass
//     with only children runs even when the sweep is finished (i == nbPrimitives).
// F3  The cull is on the SUB-triangle: getTriangleVertices with the sub-id
//     (pxgSlangTriTriGeom), then centroid, then
//         r = sqrt(PxMax(|v0-c|^2, PxMax(|v1-c|^2, |v2-c|^2))) / PxMin(s1.x, PxMin(s1.y, s1.z))
//     and the test is strict, `dist < r + tolerance`.  The cloth port spells the
//     max with the builtin; PxMax and PxMin are selectMax/selectMin here, spelled.
// F4  THE RESUME HANDSHAKE (upstream candidate 60).  Whoever writes slot 1023
//     sets offset = index.x + 1 -- and for a CHILD, index.x is its PARENT's index
//     from an earlier round, not a position in the sweep.  It can only happen when
//     n == 256 and all 1024 children pass, but then the next start index is
//     parent + 1.  In the kernel every refinement entry is BELOW startIndex and the
//     buffer is sorted by index (children fill the index buffer first, in buffer
//     order, and sweep entries follow in ascending order), so parent + 1 <=
//     startIndex and only two outcomes exist: a REWIND to parent + 1 < startIndex
//     (the level-0 triangles in between are gathered again -- their contacts are
//     duplicated, and the refinement requests the 256 cap dropped for them are
//     issued again), or, when parent + 1 == startIndex, "no overflow" and
//     startIndex = nbPrimitives (the rest of the mesh is SKIPPED, its contacts
//     lost; with no SDF on mesh0 there is no second pass to recover them).
//     Consecutive all-children rounds can chain rewinds and skips before any sweep
//     runs.  A forward jump would need an entry at or above startIndex, which the
//     kernel never produces.  The port reproduces the handshake; the fixture
//     constructs the rewind, the skip, and (as a function-level check on an input
//     the kernel cannot produce) the jump.
// F5  addToRefinementBuffer drops requests at level >= maxRefinementLevel BEFORE
//     the scan, stores the sub-id UN-elevated (the gather elevates on the way
//     out), rebuilds from 0 every round, and saturates at 256 with no resume:
//     requests beyond 256 are silently lost.
// F6  The count is `min(total + n, TargetCount)`, and that clamped value is
//     always exactly the buffer's occupancy -- every emitter below TargetCount
//     writes its slot.  Only the UNCLAMPED `total + n` runs past the slots written
//     (after saturation), which is why the min is there (compare S5 in
//     pxgSlangSdfFind).  (Corrected in review: this note first said the clamped
//     count and the slots written could differ.)
// F7  THE INDEX IS CLAMPED BEFORE THE READS, AND THE CUDA HAS NO CLAMP (P9.83b-8
//     T1).  The sweep threads past the end have ind >= nbPrimitives, and on this
//     platform `if (ind < nbPrimitives)` is a MASK, not a branch: the loads inside
//     are issued for every lane and only their values are discarded (P9.22,
//     check_speculated_loads.py).  Here that is a two-level read -- three indices
//     at 3*ind, up to ~12 KB past the index buffer, then vertices AT THOSE
//     INDICES.  Past the triangles lie the adjacency words, whose flag bits make
//     them vertex indices far outside the mesh (P9.83b-8's critic, N7), so a
//     speculated second read would have unbounded reach.  The speculated-loads
//     checker cannot see it: the reads sit in ttgGetTriangleVertices, whose index is
//     a parameter, and the guard sits here.  So the reads take `safeInd`, clamped to
//     0 past the end exactly as pxgSlangSdfFind does at each of its three culls; the
//     guard still decides addToBuffer, so no emitted value changes.  A child thread
//     reads gTtfSubdivide, which only ever holds indices below nbPrimitives, and
//     passes the clamp unchanged.
//     MEASURED, AND THE HAZARD DID NOT MATERIALISE HERE.  The fixture's index buffer
//     carries 1024 poison entries past the mesh (vertex indices near 2^31, ~32 GiB
//     past the vertex buffer).  A build WITHOUT the clamp passed 5 runs of 5: on this
//     llvmpipe these reads are not issued for inactive lanes -- whether as masked
//     gathers or by skipping an all-inactive body was not read off the JIT code.
//     Unlike P9.49's sites, this body holds no barrier.  The clamp stays anyway:
//     whether a masked read is issued is the driver's choice, not the code's, and
//     the same shape did fault where the index was uniform (P9.22).  A mutant that
//     removes it survives, and that is the measurement above, not a gap.

#ifndef PXG_SLANG_TRI_TRI_FIND
#define PXG_SLANG_TRI_TRI_FIND

static const uint TTF_TARGET_COUNT = 1024u;      // findInterestingTrianglesA<32, 1024>
static const uint TTF_BLOCK_DIM = 1024u;         // the host launch (PxgNarrowphaseCore.cpp:1450)
static const uint TTF_SUBDIVIDE_COUNT = 256u;    // addToRefinementBuffer<32, 256>

groupshared uint2 gTtfSubdivide[TTF_SUBDIVIDE_COUNT];   // trianglesToSubdivideS, cu:1010
groupshared uint gTtfNumSubdivide;                      // numTrianglesToSubdivideS, cu:1011

#ifndef TTF_SDF_DISTANCE
#define TTF_SDF_DISTANCE(t, ind, sub, p) sdfDistance(t, p)
#endif

// ---- the index buffer view over gTtbScratch ----------------------------------
void ttfIndexSet(uint k, uint2 v)
{
    const uint slot = k >> 1;
    const uint c = (k & 1u) * 2u;
    gTtbScratch[slot][c] = v.x;          // one scalar store per component: see STORAGE
    gTtbScratch[slot][c + 1u] = v.y;
}

uint2 ttfIndexGet(uint k)
{
    const uint slot = k >> 1;
    const uint c = (k & 1u) * 2u;
    return uint2(gTtbScratch[slot][c], gTtbScratch[slot][c + 1u]);
}

// ---- addIndexToBuffer<uint2, 32, 1024> (sdfCollision.cuh:498-520) -------------
uint ttfAddIndexToBuffer(uint tid, bool threadEmitsElement, uint2 index, uint nbElementsInBuffer)
{
    uint total;
    const uint exclusiveSum = sdfBlockScanExclusive(tid, threadEmitsElement, total);

    if (threadEmitsElement)
    {
        const uint writeOffset = nbElementsInBuffer + exclusiveSum;

        if (writeOffset < TTF_TARGET_COUNT)
            ttfIndexSet(writeOffset, index);

        if (writeOffset == (TTF_TARGET_COUNT - 1u))
            gSdfOffset = index.x + 1u;   // F4: the FIRST word, which for a child is its parent
    }

    return min(total + nbElementsInBuffer, TTF_TARGET_COUNT);   // F6
}

// ---- the cull's texture-free half (sdfCollision.cuh:661-670) ------------------
// Returns boundingSphereRadius / PxMin(scale1); centroidVtx is the point the SDF is
// sampled at, shape2Vertex(centroid, scale1).  F3.
float ttfTriangleCullSphere(uint64_t meshInd, uint64_t meshVerts, float3 scale0, float4 scale0Rot,
                            float3 aToBp, float4 aToBq, uint ind, uint subInd,
                            float3 scale1, float4 scale1Rot, out float3 centroidVtx)
{
    float3 v0, v1, v2;
    ttgGetTriangleVertices(meshInd, meshVerts, scale0, scale0Rot, aToBp, aToBq, ind, subInd, v0, v1, v2);

    const float third = 1.0f / 3.0f;
    const float3 centroid = (v0 + v1 + v2) * third;

    centroidVtx = sdfShape2Vertex(centroid, scale1, scale1Rot);

    float boundingSphereRadius = sqrt(ttgPxMax(ttgMagnitudeSquared(v0 - centroid),
                                               ttgPxMax(ttgMagnitudeSquared(v1 - centroid),
                                                        ttgMagnitudeSquared(v2 - centroid))));
    // PxMin(float) = intrinsics::selectMin, `a < b ? a : b`.
    const float minYZ = scale1.y < scale1.z ? scale1.y : scale1.z;
    const float minScale = scale1.x < minYZ ? scale1.x : minYZ;
    boundingSphereRadius /= minScale;
    return boundingSphereRadius;
}

// ---- findInterestingTrianglesA<32, 1024> (sdfCollision.cuh:619-687) ----------
// nbSubdivisionIndices is the caller's REGISTER copy (cu:1062, :1209); the entries
// are read from gTtfSubdivide.  Must be called by the whole 1024-thread block.
uint ttfFindInterestingTriangles(uint tid, uint nbPrimitives, uint64_t meshInd, uint64_t meshVerts,
                                 float3 scale0, float4 scale0Rot, float3 scale1, float4 scale1Rot,
                                 float tolerance, SparseSdfTexture sdfMesh1,
                                 float3 aToBp, float4 aToBq, inout uint startIndex,
                                 uint nbSubdivisionIndices)
{
    if (tid == 0u)
        gSdfOffset = startIndex;

    uint sharedIndicesCount = 0u;
    for (uint i = startIndex; (i < nbPrimitives || nbSubdivisionIndices > 0u) && sharedIndicesCount < TTF_TARGET_COUNT; )
    {
        // F1
        uint ind;
        uint subdivisionLevel = 0u;
        const uint childThreads = min(TTF_TARGET_COUNT, 4u * nbSubdivisionIndices);
        if (tid < 4u * nbSubdivisionIndices)
        {
            const uint2 v = gTtfSubdivide[tid >> 2];
            ind = v.x;
            subdivisionLevel = v.y;
            subdivisionLevel = ttgElevateSubdivisionId(subdivisionLevel, tid & 3u);
        }
        else
        {
            ind = i + tid - childThreads;
        }
        i += TTF_BLOCK_DIM - childThreads;
        nbSubdivisionIndices = 0u;

        bool addToBuffer = false;
        const uint safeInd = ind < nbPrimitives ? ind : 0u;   // F7
        if (ind < nbPrimitives)
        {
            float3 centroidVtx;
            const float radius = ttfTriangleCullSphere(meshInd, meshVerts, scale0, scale0Rot, aToBp, aToBq,
                                                       safeInd, subdivisionLevel, scale1, scale1Rot, centroidVtx);
            const float distInSDFSpace1 = TTF_SDF_DISTANCE(sdfMesh1, safeInd, subdivisionLevel, centroidVtx);
            addToBuffer = distInSDFSpace1 < (radius + tolerance);   // F3: strict
        }

        GroupMemoryBarrierWithGroupSync();
        sharedIndicesCount = ttfAddIndexToBuffer(tid, addToBuffer, uint2(ind, subdivisionLevel), sharedIndicesCount);
    }

    GroupMemoryBarrierWithGroupSync();

    // F4
    if (startIndex != gSdfOffset)
        startIndex = gSdfOffset;
    else
        startIndex = nbPrimitives;

    return sharedIndicesCount;
}

// ---- addToRefinementBuffer<32, 256> (sdfCollision.cuh:741-763) ----------------
// nbElementsInBuffer is 0 at every call site (cu:1198; femClothPrimitives.cu:2537
// compiles the refinement out), so dropping it from the write offset or the count
// is equivalent everywhere the engine calls this -- and the fixture, which passes 0
// too, cannot see such a mutant.  F5.
uint ttfAddToRefinementBuffer(uint tid, bool addToBuffer, uint ind, uint subInd, uint nbElementsInBuffer,
                              uint maxRefinementLevel)
{
    uint subdivisionLevel, subTriangleIndex;
    ttgDecodeSubdivisionId(subInd, subdivisionLevel, subTriangleIndex);
    addToBuffer = addToBuffer && subdivisionLevel < maxRefinementLevel;

    uint total;
    const uint exclusiveSum = sdfBlockScanExclusive(tid, addToBuffer, total);

    if (addToBuffer)
    {
        const uint writeOffset = nbElementsInBuffer + exclusiveSum;

        if (writeOffset < TTF_SUBDIVIDE_COUNT)
            gTtfSubdivide[writeOffset] = uint2(ind, subInd);
    }

    return min(total + nbElementsInBuffer, TTF_SUBDIVIDE_COUNT);
}

#endif // PXG_SLANG_TRI_TRI_FIND
