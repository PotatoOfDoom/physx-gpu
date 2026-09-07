// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

// cvxTriPostProcessBda.slang
//
// `convexTrimeshPostProcess` (convexMeshPostProcess.cu:369, core at :182) on the
// engine BDA ABI: the second-pass SILHOUETTE CULL.  Contact generation emits
// contacts on every triangle it touches, including on edges and vertices shared
// with a neighbouring triangle that also produced contacts.  Those duplicates
// would fight each other in the solver, so this kernel re-examines each pair that
// contact-gen flagged for a second pass and, when the feature it sat on is
// already covered by a neighbour, CLEARS the contact count on that pair.
//
// The logic is a lift of `npCvxPostProcess.slang`, which ports the same core
// verbatim against a fixture-supplied blob.  What changes here is only where the
// bytes come from: fixture arrays become `Ptr<>` reads through the engine's real
// pointers.  The heightfield half of that module is a DIFFERENT kernel
// (`convexHeightfieldPostProcess`) and is not part of this one.
//
// ===========================================================================
// THE DISPATCH: NO LOOP AT ALL, WHICH IS WHY THE CHAINED LOADS ARE SAFE
// ===========================================================================
//
// The CUDA is a grid-stride loop over `*nbSecondPassPairs` whose body chains
// through `gpuShapes[shapeID].hullOrMeshPtr` — a 64-bit address read out of
// memory and then dereferenced.  That is exactly the P8.40 pattern that segfaults
// llvmpipe inside the JIT-compiled shader.
//
// Here it costs nothing to avoid, because the CUDA's grid is enormous relative to
// its work: the host launches `max(8192, 4 * numTests)` blocks of 64 threads
// (PxgNarrowphaseCore.cpp:2667-2670), so the first pass alone covers at least
// 524288 second-pass pairs.  One workgroup per CUDA block and ONE PAIR PER THREAD
// reproduces that first pass exactly, with no loop for a chained load to sit in.
//
// The ceiling is the same one P8.46c carries and for the same reason — the trip
// count is the DEVICE value `*nbSecondPassPairs` while the grid is computed on the
// host — but here the headroom is four orders of magnitude, not two.  A scene that
// exceeded it would need more than half a million convex-triangle pairs flagged
// for a second pass in a single frame.
//
// ===========================================================================
// THE READS THIS SHADER MAKES, AND WHY EACH OFFSET IS WHAT IT IS
// ===========================================================================
//
// ConvexMeshPair (convexNpCommon.h:70-81) is 56 bytes with `startIndex` @28,
// `count` @32 and `roundedStartIndex` @40.  The trailing numbers in that
// declaration are running END offsets, not field offsets — see P8.46b.  All of
// them are now pinned to the engine header by static_assert in testNpAbiLayout.cpp
// rather than trusted to a comment.
//
// ConvexTriNormalAndIndex is 16 bytes: `PxVec3 normal` @0, `int index` @12.  The
// index word packs a deferred flag in bit 31, the contact count in bits 27-30 and
// the CPU triangle index in bits 0-26.
//
// The cooked mesh blob (readTriangleMesh, triangleMesh.cuh:120) — note that this
// kernel walks TWO SECTIONS PAST what the setup stage needs, the per-vertex
// triangle-reference table:
//
//   @0                            uint4 (numVerts, numTris, _, nbPackedNodes)
//   @16                           BV32DataPacked[nbPackedNodes]   -- 1168 B each
//   +1168*nbPackedNodes           float4 verts[numVerts]
//   +16*numVerts                  uint4  indices[numTris]
//   +16*numTris                   uint4  adjacencies[numTris]
//   +16*numTris                   uint32 faceRemap[numTris]
//   +4*numTris                    uint32 accumulatedTriangleRefs[numVerts + 1]
//   +4*(numVerts+1)               uint32 triangleReferences[...]
//
// `.w` sizes the node array; `.z` is a different count entirely.  The 1168 is
// `sizeof(Gu::BV32DataPacked)` and is pinned by static_assert — it read 160 in
// `pxgSlangCvxTriSetup.slang` until P8.49, with the fixture carrying the same
// wrong number, so the two agreed with each other and disagreed with the engine.
//
// ===========================================================================
// THE THREE SWEEP SURVIVORS, AND WHY NO FIXTURE REACHES THEM
// ===========================================================================
//
// 48 of 51 mutants die.  The three that live are not gaps:
//
//   * PP_THREADS_PER_BLOCK lowered to 32.  `numthreads` stays 64, so a smaller
//     stride makes each workgroup's index range OVERLAP its neighbour's: coverage
//     becomes a strict SUPERSET and some pairs are processed twice.  The only
//     effect of this kernel is `index &= 0x00FFFFFF`, which is IDEMPOTENT, so the
//     second visit is invisible by construction.  The constant does matter in the
//     other direction and that is tested: raising it to 128 loses the pairs
//     between 64 and 127 and dies.
//   * PP_EMASK widened from 0xe0000000 to 0xf8000000.  The two differ only in bits
//     27 and 28, and the low half of the same word is the pair index.  Reaching
//     those bits needs a second-pass pair index of at least 2^27 — 134 million
//     entries in the stack array — so no fixture can allocate the input that would
//     separate them.
//   * the searches bounded by the ROUNDED count instead of `count`.  The entries
//     past `count` are the radix sort's 0xFFFFFFFF padding, and 0xFFFFFFFF is
//     greater than or equal to every target the searches can form (`index | flag`,
//     with the flag in the top two bits and a triangle index far below 2^27).  So
//     extending the range can neither add a match nor move the index binarySearch
//     returns.  Equality would need a target of exactly 0xFFFFFFFF, i.e. a
//     triangleIndex of 0xFFFFFFFF — which the boundary branch short-circuits
//     before any search runs.

import pxgSlangCvxTriSetup;   // CtShape, the PxgShape mirror

static const uint PP_THREADS_PER_BLOCK = 64u;   // WARP_SIZE * NP_TRIMESH_WARPS_PER_BLOCK

static const uint PP_BV32_STRIDE = 1168u;       // sizeof(Gu::BV32DataPacked)

// ConvexTriIntermediateData::VoronoiState (convexNpCommon.h:118-127).  NOTE that
// the enum declares eE02 BEFORE eE12 — 5<<29 and 6<<29 respectively — which is NOT
// the order the core's branch chain reads.  Getting these two the wrong way round
// swaps which adjacency word an edge contact consults.
static const uint PP_EV0  = 1u << 29;
static const uint PP_EV1  = 2u << 29;
static const uint PP_EV2  = 3u << 29;
static const uint PP_EE01 = 4u << 29;
static const uint PP_EE02 = 5u << 29;
static const uint PP_EE12 = 6u << 29;
static const uint PP_EMASK = 0xe0000000u;

static const uint PP_BOUNDARY = 0xffffffffu;             // triangle.cuh:35
static const uint PP_FACE_CONTACT = 0x80000000u;         // TriangleSearchFlags
static const uint PP_EDGE_CONTACT = 0xc0000000u;

// (ConvexMeshPair*, ConvexTriNormalAndIndex**, ConvexTriIntermediateData**,
//  const PxU32**, PxU32**, PxU8*, PxU32*, PxgShape*) -- eight pointers, 64 bytes.
struct PpParams
{
    uint64_t cvxTrimeshPair;           // @0
    uint64_t cvxTriNIPtr;              // @8
    uint64_t cvxTriIntermPtr;          // @16  read by the core's commented-out
                                       //      variant only; kept for ABI shape
    uint64_t orderedCvxTriIntermPtr;   // @24
    uint64_t cvxTriSecondPassPairsPtr; // @32
    uint64_t stackPtr;                 // @40  sPairsGPU is just this, cast
    uint64_t nbSecondPassPairs;        // @48
    uint64_t shapes;                   // @56
}                                      // = 64

[[vk::push_constant]]
ConstantBuffer<PpParams> gPp;

uint ppGetNbContacts(uint index) { return (index & 0x7FFFFFFFu) >> 27u; }

bool ppIsBoundaryEdge(uint edgeIndex) { return edgeIndex == PP_BOUNDARY; }

// reduction.cuh:729 -- the largest index with data[index] <= value, or 0 when
// data[0] is already greater.  Note the loop loads a VALUE, not an address: this
// is a plain BDA read in a loop, not the chained form P8.40 rules out.
uint ppBinarySearch(uint64_t data, uint numElements, uint value)
{
    uint left = 0u;
    uint right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (Ptr<uint>(data + uint64_t(pos) * 4UL)[0] <= value)
            left = pos + 1u;
        else
            right = pos;
    }
    return left != 0u ? left - 1u : 0u;
}

// findTriangle / findFaceContactTriangle / findEdgeContactTriangle
// (convexMeshPostProcess.cu:154-179) differ only in the flag they OR in.
bool ppFindWithFlag(uint64_t data, uint count, uint targetIndexWithMask, uint flag)
{
    const uint target = targetIndexWithMask | flag;
    const uint idx = ppBinarySearch(data, count, target);
    return idx < count && Ptr<uint>(data + uint64_t(idx) * 4UL)[0] == target;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void convexTrimeshPostProcess(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint nbTpProcess = Ptr<uint>(gPp.nbSecondPassPairs)[0];
    // convexMeshPostProcess.cu:391 / :603 -- the CUDA covers the second-pass queue
    // with a grid-stride loop, and its trip count `*nbSecondPassPairs` is a DEVICE
    // value while the grid is computed on the host from a shape count.  Restored in
    // P9.59.  The trip count `ceil(nbTpProcess / (WorkgroupCount().x * 64))` is
    // WORKGROUP-UNIFORM; there are no barriers in this entry point, but keeping the
    // property true costs nothing and the next edit may add one.
    const uint ppStride = WorkgroupCount().x * PP_THREADS_PER_BLOCK;

    for (uint ppBase = 0u; ppBase < nbTpProcess; ppBase += ppStride)
    {
    const uint globalThreadIdx = ppBase + gid.x * PP_THREADS_PER_BLOCK + gtid.x;
    if (globalThreadIdx >= nbTpProcess)
        continue;

    const uint64_t secondPassPairs = Ptr<uint64_t>(gPp.cvxTriSecondPassPairsPtr)[0];
    const uint64_t cvxTriNI = Ptr<uint64_t>(gPp.cvxTriNIPtr)[0];

    const uint pairIdxPlusMask = Ptr<uint>(secondPassPairs + uint64_t(globalThreadIdx) * 4UL)[0];
    const uint pairIdx = pairIdxPlusMask & 0x1fffffffu;

    // sPairsGPU is `stackPtr` reinterpreted as uint4* -- not a chained load, the
    // __shared__ hand-off in the CUDA is only there to share one cast.
    const uint4 curPair = Ptr<uint4>(gPp.stackPtr + uint64_t(pairIdx) * 16UL)[0];
    const uint cmIdx = curPair.x;
    const uint triangleIdx = curPair.y;
    const uint testOffset = curPair.z;
    const uint shapeID = curPair.w;

    const uint64_t pairAddr = gPp.cvxTrimeshPair + uint64_t(cmIdx) * 56UL;
    const uint nbPairsPerCM = Ptr<uint>(pairAddr + 32UL)[0];           // count
    const uint pairStartIndex = Ptr<uint>(pairAddr + 28UL)[0];         // startIndex
    const uint orderedPairStartIndex = Ptr<uint>(pairAddr + 40UL)[0];  // roundedStartIndex

    const uint convexTriPairOffset = pairStartIndex + testOffset;

    // ConvexTriNormalAndIndex: PxVec3 normal @0, int index @12, stride 16.
    const uint64_t niIndexAddr = cvxTriNI + uint64_t(convexTriPairOffset) * 16UL + 12UL;
    const uint niWord = Ptr<uint>(niIndexAddr)[0];

    if (ppGetNbContacts(niWord) == 0u)
        return;

    // ---- the cooked mesh blob ---------------------------------------------
    const CtShape trimeshShape = Ptr<CtShape>(gPp.shapes)[shapeID];
    const uint64_t meshPtr = uint64_t(trimeshShape.hullOrMeshOffset.x) |
                             (uint64_t(trimeshShape.hullOrMeshOffset.y) << 32);

    const uint4 meshHeader = Ptr<uint4>(meshPtr)[0];
    const uint numVerts = meshHeader.x;
    const uint numTris = meshHeader.y;

    uint64_t off = meshPtr + 16UL;
    off += uint64_t(meshHeader.w) * uint64_t(PP_BV32_STRIDE);
    off += uint64_t(numVerts) * 16UL;              // verts, skipped
    const uint64_t triIndicesAddr = off;
    off += uint64_t(numTris) * 16UL;
    const uint64_t adjAddr = off;
    off += uint64_t(numTris) * 16UL;
    off += uint64_t(numTris) * 4UL;                // mGRB_faceRemap, skipped
    const uint64_t accRefsAddr = off;
    off += uint64_t(numVerts + 1u) * 4UL;
    const uint64_t refsAddr = off;

    const uint64_t orderedBase = Ptr<uint64_t>(gPp.orderedCvxTriIntermPtr)[0];
    const uint64_t orderedTriIndices = orderedBase + uint64_t(orderedPairStartIndex) * 4UL;

    uint4 adjTriangles = Ptr<uint4>(adjAddr + uint64_t(triangleIdx) * 16UL)[0];
    uint4 indices = Ptr<uint4>(triIndicesAddr + uint64_t(triangleIdx) * 16UL)[0];

    // A negative scale determinant reverses the winding.  PxMeshScale
    // ::hasNegativeDeterminant() is the product of the three scale components.
    const bool flipNormal =
        (trimeshShape.scaleX * trimeshShape.scaleY * trimeshShape.scaleZ) < 0.0f;
    if (flipNormal)
    {
        const uint ta = adjTriangles.x; adjTriangles.x = adjTriangles.z; adjTriangles.z = ta;
        const uint ti = indices.y;      indices.y = indices.z;           indices.z = ti;
    }

    const uint mask = pairIdxPlusMask & PP_EMASK;

    bool needsProcessing = false;
    uint triangleIndex = 0xFFFFFFFFu;
    uint vertexIdx = 0xFFFFFFFFu;
    uint2 adjTriangles2 = uint2(0u, 0u);

    if (mask == PP_EV0)
    {
        vertexIdx = indices.x;
        adjTriangles2 = uint2(adjTriangles.x, adjTriangles.z);
    }
    else if (mask == PP_EV1)
    {
        vertexIdx = indices.y;
        adjTriangles2 = uint2(adjTriangles.x, adjTriangles.y);
    }
    else if (mask == PP_EV2)
    {
        vertexIdx = indices.z;
        adjTriangles2 = uint2(adjTriangles.y, adjTriangles.z);
    }
    else if (mask == PP_EE01)
    {
        triangleIndex = adjTriangles.x;
        if (triangleIndex == 0xFFFFFFFFu) needsProcessing = true;
    }
    else if (mask == PP_EE12)
    {
        triangleIndex = adjTriangles.y;
        if (triangleIndex == 0xFFFFFFFFu) needsProcessing = true;
    }
    else if (mask == PP_EE02)
    {
        triangleIndex = adjTriangles.z;
        if (triangleIndex == 0xFFFFFFFFu) needsProcessing = true;
    }

    if (triangleIndex != 0xFFFFFFFFu)
    {
        // `isBoundaryEdge` is false here BY CONSTRUCTION -- the branches above
        // already sent 0xFFFFFFFF down the other path -- so the OR is dead.
        // Ported verbatim anyway; this is not the place to be clever.
        needsProcessing = ppIsBoundaryEdge(triangleIndex) ||
                          !ppFindWithFlag(orderedTriIndices, nbPairsPerCM,
                                          triangleIndex, PP_FACE_CONTACT);
    }
    else if (vertexIdx != 0xFFFFFFFFu)
    {
        // A vertex contact survives only if some triangle sharing the vertex
        // already produced a face or edge contact.
        needsProcessing = true;

        if (!(ppIsBoundaryEdge(adjTriangles2.x) && ppIsBoundaryEdge(adjTriangles2.y)))
        {
            const uint startIdx = Ptr<uint>(accRefsAddr + uint64_t(vertexIdx) * 4UL)[0];
            const uint endIdx = Ptr<uint>(accRefsAddr + uint64_t(vertexIdx + 1u) * 4UL)[0];

            for (uint idx = startIdx; idx < endIdx; ++idx)
            {
                const uint triIdx = Ptr<uint>(refsAddr + uint64_t(idx) * 4UL)[0];
                if (ppFindWithFlag(orderedTriIndices, nbPairsPerCM, triIdx, PP_FACE_CONTACT) ||
                    ppFindWithFlag(orderedTriIndices, nbPairsPerCM, triIdx, PP_EDGE_CONTACT))
                {
                    needsProcessing = false;
                    break;
                }
            }
        }
    }

    if (!needsProcessing)
    {
        // THE MASK IS 0x00FFFFFF -- SIX F's in the CUDA's `0x0ffffff`, not seven.
        // It clears the contact count and the deferred flag as intended, but it
        // ALSO clears bits 24-26 of the 27-bit triangle index.  That is what the
        // engine does and what everything downstream is calibrated against, so it
        // is reproduced exactly rather than "fixed".
        Ptr<uint>(niIndexAddr)[0] = niWord & 0x0ffffffu;
    }
    }
}
