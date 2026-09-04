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

// convexHfPostProcessBda.slang
//
// `convexHeightfieldPostProcess` (convexMeshPostProcess.cu:585-620, whose per-pair
// body is convexHeightfieldPostProcessCore at :410-582) on the engine BDA ABI --
// the last kernel of the rigid-vs-heightfield narrowphase chain.  P9.51c.
//
// It decides, for each contact the second pass flagged, whether a NEIGHBOURING
// triangle has already produced a better contact for the same feature; when one
// has, this pair's contact count is cleared.  Without it a box on a heightfield
// gets duplicate contacts along every shared edge and vertex of the grid.
//
// ===========================================================================
// THE TWO PLACES IT IS NOT ITS TRIMESH SIBLING
// ===========================================================================
//
// `cvxTriPostProcessBda` is the same kernel over a cooked mesh, and the parts that
// differ are exactly the parts that have to differ:
//
//  1. THE EDGE TEST IS `isActiveEdge && !findTriangle`, NOT `isBoundaryEdge ||
//     !findFaceContactTriangle`.  Same shape, different operator, different
//     predicate (convexMeshPostProcess.cu:522 against :286).  `isActiveEdge`
//     (convexTriangle.cuh:556) is `edgeIndex == BOUNDARY || !(edgeIndex &
//     NONCONVEX_FLAG)` -- it admits a boundary edge AND any edge not marked
//     non-convex, where `isBoundaryEdge` admits only the first.  Carrying the
//     trimesh spelling over would keep contacts on non-convex heightfield edges
//     that the engine drops.
//
//  1a. AND THE BOUNDARY-EDGE GUARD IS ABSENT.  Each of the trimesh core's three
//     edge arms ends with `if (triangleIndex == 0xFFFFFFFF) needsProcessing =
//     true;` (convexMeshPostProcess.cu:298, :306, :314).  The heightfield core's
//     arms do not (:506-516).  The consequence is not cosmetic: with the guard, a
//     BOUNDARY adjacency KEEPS the contact; without it, `triangleIndex` is still
//     0xFFFFFFFF when the branch chain ends, so BOTH the edge and the vertex arm
//     are skipped, `needsProcessing` stays false, and the contact is CLEARED.
//
//     So an edge contact on the RIM of a heightfield is dropped, while the same
//     contact on the rim of a cooked mesh is kept.  UPSTREAM CANDIDATE (the 47th
//     recorded in this port, and like every one of them it is DOCUMENTED, NOT
//     REPORTED).  It is transcribed exactly as upstream has it -- the downstream
//     contact counts are calibrated against this behaviour, and "the sibling does
//     it the other way" is a reason to write this note, not to change the code.
//
//  2. THE VERTEX TEST IS A 3x3 CELL WALK, NOT A REFERENCE-TABLE LOOKUP.  A cooked
//     mesh ships an `accumulatedTriangleRefs` / `triangleReferences` pair that
//     lists every triangle touching a vertex; a heightfield has no such table, so
//     the .cu walks the up-to-nine cells around the vertex's own cell and tests
//     both triangles of each (:527-570).  The clamps are upstream's own and are
//     asymmetric on purpose: `PxMax(row,1)-1` and `PxMin(row, nbRows-2)+1`.
//
// Everything else -- the second-pass index and its mask, the ConvexMeshPair
// offsets, the contact-count gate, the binary search, and the final
// `index & 0x0ffffff` -- is the sibling's, and is spelled the same way here.
//
// ===========================================================================
// THE DISPATCH SHAPE, AND A CEILING THAT IS NOT THE SIBLING'S
// ===========================================================================
//
// One workgroup per CUDA block and ONE PAIR PER THREAD, which reproduces the
// CUDA's first grid-stride pass exactly and leaves no loop for a chained
// buffer-device-address load to sit in (the P8.40 hazard).
//
// **THE HEADROOM HERE IS FOUR ORDERS OF MAGNITUDE SMALLER THAN THE TRIMESH
// SIBLING'S, AND THAT IS WORTH STATING PLAINLY.**  The trimesh postprocess is
// launched with `max(8192, ...)` blocks (PxgNarrowphaseCore.cpp:2671), so its
// first pass alone covers at least 524288 second-pass pairs.  This one is launched
// with `(numTests * 8 + 1) / 2` blocks of 64 threads (:2971-2973), i.e. 256 threads
// per contact manager -- so the first pass covers 256 * numTests second-pass pairs
// and the CUDA's grid-stride loop is what covers the rest.  A contact manager
// whose box straddles more than 256 flagged heightfield triangles in one frame
// would lose the surplus, and those pairs keep contacts the engine would have
// cleared: duplicate contacts along shared edges, not missing ones.
//
// That is a real limit, it is smaller than any other in this chain, and closing it
// needs the same thing the rest of the chain needs -- indirect dispatch in the
// shim, or a host-side grid derived from `*nbSecondPassPairs`.  It is recorded
// here rather than left to be discovered.
//
// The ABI row is SHARED with `convexTrimeshPostProcess`: both kernels declare the
// same eight pointers in the same order (convexMeshPostProcess.cu:585-593 against
// :240-248), so the 64-byte blob is identical.

module convexHfPostProcessBda;

import pxgSlangCvxTriSetup;   // CtShape, the PxgShape mirror
import pxgSlangHeightfield;   // the sample decoders and the two index helpers

static const uint PP_THREADS_PER_BLOCK = 64u;   // WARP_SIZE * NP_TRIMESH_WARPS_PER_BLOCK

// ConvexTriIntermediateData::VoronoiState (convexNpCommon.h:118-127).  NOTE that
// the enum declares eE02 BEFORE eE12 -- 5<<29 and 6<<29 -- which is NOT the order
// the core's branch chain reads.  Getting these two the wrong way round swaps
// which adjacency word an edge contact consults.
static const uint PP_EV0  = 1u << 29;
static const uint PP_EV1  = 2u << 29;
static const uint PP_EV2  = 3u << 29;
static const uint PP_EE01 = 4u << 29;
static const uint PP_EE02 = 5u << 29;
static const uint PP_EE12 = 6u << 29;
static const uint PP_EMASK = 0xe0000000u;

static const uint PP_BOUNDARY = 0xffffffffu;             // triangle.cuh:35
static const uint PP_FACE_CONTACT = 0x80000000u;         // TriangleSearchFlags
// GuCookingGrbTriangleMesh.h:56 -- the adjacency word's BIT 31 marks an edge the
// cooker found non-convex.  `isActiveEdge` admits an edge only when that bit is
// clear, or when the whole word is BOUNDARY.
//
// It is the SAME BIT as PP_FACE_CONTACT, and that is not a coincidence to tidy
// away: the search key ORs 0x80000000 into a triangle index, and the cooker sets
// 0x80000000 in an adjacency word, so the two constants must stay separately named
// even though they are equal.  Collapsing them would make a later change to either
// silently change the other.
static const uint PP_NONCONVEX_FLAG = 0x80000000u;

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

// convexTriangle.cuh:556.  NOT isBoundaryEdge: this admits every edge the cooker
// did not mark non-convex, and a boundary edge on top of that.
bool ppIsActiveEdge(uint edgeIndex)
{
    return edgeIndex == PP_BOUNDARY || (edgeIndex & PP_NONCONVEX_FLAG) == 0u;
}

// reduction.cuh:729 -- the largest index with data[index] <= value, or 0 when
// data[0] is already greater.  The loop loads a VALUE, not an address: a plain BDA
// read in a loop, not the chained form P8.40 rules out.
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

// findTriangle (convexMeshPostProcess.cu:154).  It ORs 0x80000000, which is
// TriangleSearchFlags::eFACE_CONTACT -- the same constant the trimesh sibling's
// findFaceContactTriangle uses, written a different way upstream.
bool ppFindTriangle(uint64_t data, uint count, uint targetIndexWithMask)
{
    const uint target = targetIndexWithMask | PP_FACE_CONTACT;
    const uint idx = ppBinarySearch(data, count, target);
    return idx < count && Ptr<uint>(data + uint64_t(idx) * 4UL)[0] == target;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void convexHeightfieldPostProcess(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint nbTpProcess = Ptr<uint>(gPp.nbSecondPassPairs)[0];
    const uint globalThreadIdx = gid.x * PP_THREADS_PER_BLOCK + gtid.x;
    if (globalThreadIdx >= nbTpProcess)
        return;

    const uint64_t secondPassPairs = Ptr<uint64_t>(gPp.cvxTriSecondPassPairsPtr)[0];
    const uint64_t cvxTriNI = Ptr<uint64_t>(gPp.cvxTriNIPtr)[0];

    const uint pairIdxPlusMask = Ptr<uint>(secondPassPairs + uint64_t(globalThreadIdx) * 4UL)[0];
    const uint pairIdx = pairIdxPlusMask & 0x1fffffffu;

    // sPairsGPU is `stackPtr` reinterpreted as uint4* -- not a chained load; the
    // __shared__ hand-off in the CUDA exists only to share one cast.
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

    // ---- the heightfield blob ---------------------------------------------
    // THE SHAPE COMES FROM curPair.w, NOT FROM THE CONTACT MANAGER.  The midphase
    // wrote the HEIGHTFIELD's shape ref into the stack record (`cin.shapeRef1`),
    // precisely so this kernel does not have to re-derive which operand is which.
    const CtShape hfShape = Ptr<CtShape>(gPp.shapes)[shapeID];
    const uint64_t meshPtr = uint64_t(hfShape.hullOrMeshOffset.x) |
                             (uint64_t(hfShape.hullOrMeshOffset.y) << 32);

    const uint nbRows = Ptr<uint>(meshPtr)[0];
    const uint nbCols = Ptr<uint>(meshPtr + 4UL)[0];
    const uint64_t samples = meshPtr + 8UL;

    // The handedness xor, from the ROW and COLUMN scales only -- the height scale
    // has no say (convexMeshPostProcess.cu:455-467, and the `handedness` vector it
    // also builds there is never read).
    bool wrongHanded = false;
    if (hfShape.scaleZ < 0.0f) wrongHanded = !wrongHanded;
    if (hfShape.scaleX < 0.0f) wrongHanded = !wrongHanded;
    const uint wh = wrongHanded ? 1u : 0u;

    uint tv[3];
    uint adj[3];
    {
        uint i0, i1, i2;
        hfTriVertexIndices(samples, triangleIdx, nbCols, i0, i1, i2);
        tv[0] = i0; tv[1u + wh] = i1; tv[2u - wh] = i2;
        uint a0, a1, a2;
        hfTriAdjacencyIndices(samples, triangleIdx, nbRows, nbCols, a0, a1, a2);
        adj[wh != 0u ? 2u : 0u] = a0; adj[1] = a1; adj[wh != 0u ? 0u : 2u] = a2;
    }

    const uint64_t orderedBase = Ptr<uint64_t>(gPp.orderedCvxTriIntermPtr)[0];
    const uint64_t orderedTriIndices = orderedBase + uint64_t(orderedPairStartIndex) * 4UL;

    const uint mask = pairIdxPlusMask & PP_EMASK;

    bool needsProcessing = false;
    uint triangleIndex = 0xFFFFFFFFu;
    uint vertexIdx = 0xFFFFFFFFu;

    if (mask == PP_EV0)       vertexIdx = tv[0];
    else if (mask == PP_EV1)  vertexIdx = tv[1];
    else if (mask == PP_EV2)  vertexIdx = tv[2];
    else if (mask == PP_EE01) triangleIndex = adj[0];
    else if (mask == PP_EE12) triangleIndex = adj[1];
    else if (mask == PP_EE02) triangleIndex = adj[2];

    if (triangleIndex != 0xFFFFFFFFu)
    {
        // `isActiveEdge && !findTriangle` -- see the header.  The trimesh sibling's
        // `isBoundaryEdge || !findFaceContactTriangle` is a different predicate
        // joined by a different operator, and both differences are upstream's.
        //
        // AND `ppIsActiveEdge` IS ALWAYS TRUE HERE.  A heightfield adjacency word
        // is a plain triangle index or BOUNDARY -- the NONCONVEX bit is set by the
        // MESH COOKER (GuCookingGrbTriangleMesh.h:141) and nothing sets it on this
        // path, while BOUNDARY is caught by the first clause.  So a sweep mutant
        // that drops the call will SURVIVE.  It is transcribed anyway: the branch
        // is upstream's, and "no input we can build reaches it" is a reason to
        // record the fact, not to delete the code.
        needsProcessing = ppIsActiveEdge(triangleIndex) &&
                          !ppFindTriangle(orderedTriIndices, nbPairsPerCM, triangleIndex);
    }
    else if (vertexIdx != 0xFFFFFFFFu)
    {
        // The 3x3 cell neighbourhood of the vertex's own cell.  The clamps are
        // upstream's and are ASYMMETRIC: the low end is `max(x,1) - 1` (so 0 stays
        // 0 instead of wrapping to 0xffffffff) and the high end is
        // `min(x, n-2) + 1`, which is n-1 at the far edge -- the last row and
        // column of VERTICES, which is one past the last row and column of CELLS.
        const uint cellIdx = triangleIdx / 2u;
        const uint column = cellIdx % nbCols;
        const uint row = cellIdx / nbCols;
        const uint startRow = max(row, 1u) - 1u;
        const uint endRow = min(row, nbRows - 2u) + 1u;
        const uint startCol = max(column, 1u) - 1u;
        const uint endCol = min(column, nbCols - 2u) + 1u;

        needsProcessing = true;

        // The `&& needsProcessing` in the outer condition and the `break` in the
        // inner loop are upstream's own early-out (:537, :551, :565).  The `break`
        // leaves only the INNER loop, so the outer condition is what stops the
        // walk -- transcribed with both, because a single flag test after the loops
        // would keep searching and the searches are the expensive part.
        //
        // The walk's last row and column are VERTEX rows, one past the last CELL:
        // `min(row, nbRows-2) + 1` is nbRows-1 at the far edge.  The only read
        // those iterations make is `hfZerothVertexShared(samples, cell2)`, and
        // cell2 is at most nbRows*nbCols - 1 -- the last valid sample -- so they
        // are in range.  What they produce is vertex indices naming triangles that
        // do not exist, which then match nothing.  Checked rather than assumed,
        // because the same clamp on a path that DEREFERENCED those indices would
        // read past the grid.
        for (uint rr = startRow; rr <= endRow && needsProcessing; ++rr)
        {
            for (uint cc = startCol; cc <= endCol; ++cc)
            {
                const uint cell2 = rr * nbCols + cc;

                uint triIdx = cell2 * 2u;
                {
                    uint i0, i1, i2;
                    hfTriVertexIndices(samples, triIdx, nbCols, i0, i1, i2);
                    uint w[3];
                    w[0] = i0; w[1u + wh] = i1; w[2u - wh] = i2;
                    if (w[0] == vertexIdx || w[1] == vertexIdx || w[2] == vertexIdx)
                    {
                        if (ppFindTriangle(orderedTriIndices, nbPairsPerCM, triIdx))
                        {
                            needsProcessing = false;
                            break;
                        }
                    }
                }

                triIdx = triIdx + 1u;
                {
                    uint i0, i1, i2;
                    hfTriVertexIndices(samples, triIdx, nbCols, i0, i1, i2);
                    uint w[3];
                    w[0] = i0; w[1u + wh] = i1; w[2u - wh] = i2;
                    if (w[0] == vertexIdx || w[1] == vertexIdx || w[2] == vertexIdx)
                    {
                        if (ppFindTriangle(orderedTriIndices, nbPairsPerCM, triIdx))
                        {
                            needsProcessing = false;
                            break;
                        }
                    }
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
