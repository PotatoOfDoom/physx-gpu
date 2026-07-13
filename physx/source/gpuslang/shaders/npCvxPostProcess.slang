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

// npCvxPostProcess.slang
//
// Slang port of convexTrimeshPostProcessCore (convexMeshPostProcess.cu
// L182-365) and convexHeightfieldPostProcessCore (L410-582): the scalar
// per-thread second-pass silhouette culling that clears contacts generated on
// inactive mesh features.  One second-pass pair per test.  Ported verbatim:
//   - pairIdx = word & 0x1FFFFFFF, voronoi mask = word & eMASK (0xE0000000),
//     with eV0/eV1/eV2 = 1/2/3 << 29 and eE01/eE12/eE02 = 4/5/6 << 29;
//   - the early return when getNbContacts(cvxni.index) == 0;
//   - TRIMESH: the real byte-offset blob walk (header uint4, skip
//     hdr.w * sizeof(BV32DataPacked) = 1168 bytes, skip numVerts float4
//     verts, triIndices, adjacencies, skip faceRemap, accumulatedTriangleRefs
//     (numVerts+1), triangleReferences); flipNormal (scale determinant < 0)
//     swaps adjTriangles.x<->z and indices.y<->z; edge branches take the
//     adjacent triangle (needsProcessing immediately when it is 0xFFFFFFFF);
//     an edge with a neighbour: needsProcessing = isBoundaryEdge(idx) ||
//     !findFaceContactTriangle (note isBoundaryEdge is false there by
//     construction -- ported verbatim; the OR'd eFACE_CONTACT flag does NOT
//     strip a NONCONVEX bit 31 from the adjacency, so flagged neighbours
//     never match); a vertex: needsProcessing = true unless (not both
//     adjacent edges boundary) and some triangle referencing the vertex is
//     found as face or edge contact;
//   - HEIGHTFIELD: wrongHanded from rowScale/columnScale signs permutes the
//     vertex-index outputs [0, 1+wh, 2-wh] and adjacency outputs
//     [wh?2:0, 1, wh?0:2] (the handedness vector at cu L453 is dead);
//     edge branches: needsProcessing = isActiveEdge && !findTriangle
//     (isActiveEdge(idx != BOUNDARY) = !(idx & NONCONVEX) -- always true for
//     HF adjacencies, ported verbatim); vertex branches walk the 3x3 cell
//     neighbourhood (clamped rows/cols) over both cell triangles, matching
//     any of the three (permuted) vertex indices, findTriangle with the
//     plain 0x80000000 flag;
//   - the clear is cvxni.index &= 0x0FFFFFF -- SEVEN F's: it zeroes the
//     nbContacts/deferred bits AND bits 24-26 of the 27-bit triangle index.
// findTriangle/findFaceContactTriangle/findEdgeContactTriangle and
// binarySearch are the tranche-113 ports; the HF vertex/adjacency helpers are
// the tranche-59 ports (copied verbatim from npHeightfield.slang).

static const uint EMASK = 0xE0000000u;
static const uint EV0 = 1u << 29;
static const uint EV1 = 2u << 29;
static const uint EV2 = 3u << 29;
static const uint EE01 = 4u << 29;
static const uint EE12 = 5u << 29;
static const uint EE02 = 6u << 29;
static const uint FACE_CONTACT_FLAG = 0x80000000u;  // TriangleSearchFlags::eFACE_CONTACT
static const uint EDGE_CONTACT_FLAG = 0xC0000000u;  // TriangleSearchFlags::eEDGE_CONTACT
static const uint BOUNDARY = 0xFFFFFFFFu;
static const uint NONCONVEX_FLAG = 0x80000000u;
static const uint BV32_PACKED_BYTES = 1168u;        // sizeof(Gu::BV32DataPacked)

struct PostIn
{
    uint4 hdr;        // x = mode (0 trimesh, 1 heightfield), y = pairIdxPlusMask word, z pad, w pad
    uint4 pairsGPU[4];// curPair candidates: x = cmIdx (0/1), y = triangleIdx, z = testOffset, w pad
    uint4 pairA;      // ConvexMeshPair 0: x = count, y = startIndex, z = roundedStartIndex, w pad
    uint4 pairB;      // ConvexMeshPair 1
    float4 scale;     // trimesh: mesh scale xyz | heightfield: x = rowScale, z = columnScale
    uint4 niIdx[8];   // cvxTriNI[..].index words [32]
    uint4 ordered[8]; // orderedTriIndices base table [32] (per-pair slice starts at roundedStartIndex)
    uint4 blob[192];  // trimesh blob bytes (uint-packed) | heightfield: [0]=nbRows, [1]=nbCols, samples from [2]
}

struct PostOut
{
    uint4 res;        // x = cvxni.index after, y = needsProcessing, z = triangleIndex, w = vertexIdx
    uint4 aux;        // x = earlyOut (nbContacts == 0), y = mask, z = convexTriPairOffset, w pad
}

struct PostParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<PostParams> gParams;

RWStructuredBuffer<PostIn> gIn;    // binding 0
RWStructuredBuffer<PostOut> gOut;  // binding 1

uint getNbContactsNic(uint index) { return (index & 0x7FFFFFFFu) >> 27u; }

uint blobWord(PostIn s, uint byteOff) { return s.blob[byteOff / 16u][(byteOff / 4u) & 3u]; }

// nputils binarySearch (tranche 113): last index with data <= value, else 0
uint bsearchOrdered(PostIn s, uint base, uint count, uint value)
{
    uint left = 0u, right = count;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (s.ordered[(base + pos) / 4u][(base + pos) & 3u] <= value) left = pos + 1u;
        else right = pos;
    }
    return left != 0u ? left - 1u : 0u;
}

bool findWithFlag(PostIn s, uint base, uint count, uint targetIndexWithMask, uint flag)
{
    const uint target = targetIndexWithMask | flag;
    const uint idx = bsearchOrdered(s, base, count, target);
    return idx < count && s.ordered[(base + idx) / 4u][(base + idx) & 3u] == target;
}

bool isBoundaryEdge(uint edgeIndex) { return edgeIndex == BOUNDARY; }
bool isActiveEdge(uint edgeIndex) { return edgeIndex == BOUNDARY || (edgeIndex & NONCONVEX_FLAG) == 0u; }

// ---- heightfield helpers (tranche 59 ports, verbatim from npHeightfield) ----
bool hfZerothShared(PostIn s, uint vertexIndex)
{
    return (s.blob[(2u + vertexIndex) / 4u][(2u + vertexIndex) & 3u] & 0x800000u) != 0u;
}

void hfTriVertexIndices(PostIn s, uint triangleIndex, uint columns, out uint i0, out uint i1, out uint i2)
{
    const uint cell = triangleIndex >> 1u;
    const bool firstTri = (triangleIndex & 1u) == 0u;
    if (hfZerothShared(s, cell))
    {
        if (firstTri) { i0 = cell + columns; i1 = cell; i2 = cell + columns + 1u; }
        else          { i0 = cell + 1u; i1 = cell + columns + 1u; i2 = cell; }
    }
    else
    {
        if (firstTri) { i0 = cell; i1 = cell + 1u; i2 = cell + columns; }
        else          { i0 = cell + columns + 1u; i1 = cell + columns; i2 = cell + 1u; }
    }
}

void hfTriAdjacencyIndices(PostIn s, uint triangleIndex, uint columns, uint rows, out uint a0, out uint a1, out uint a2)
{
    const uint cell = triangleIndex >> 1u;
    const bool firstTri = (triangleIndex & 1u) == 0u;
    if (hfZerothShared(s, cell))
    {
        if (firstTri)
        {
            a0 = BOUNDARY; a1 = triangleIndex + 1u; a2 = BOUNDARY;
            if (cell % columns != 0u)           a0 = triangleIndex - 1u;
            if (cell / columns != rows - 2u)    a2 = ((cell + columns) * 2u) + 1u;
        }
        else
        {
            a0 = BOUNDARY; a1 = triangleIndex - 1u; a2 = BOUNDARY;
            if (cell % columns < columns - 2u)  a0 = triangleIndex + 1u;
            if (cell >= columns - 1u)           a2 = (cell - columns) * 2u;
        }
    }
    else
    {
        if (firstTri)
        {
            a0 = BOUNDARY; a1 = triangleIndex + 1u; a2 = BOUNDARY;
            if (cell >= columns - 1u)           a0 = ((cell - columns) * 2u) + 1u;
            if (cell % columns != 0u)           a2 = triangleIndex - 1u;
        }
        else
        {
            a0 = BOUNDARY; a1 = triangleIndex - 1u; a2 = BOUNDARY;
            if (cell / columns != rows - 2u)    a0 = (cell + columns) * 2u;
            if (cell % columns < columns - 2u)  a2 = triangleIndex + 1u;
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxPostProcess(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const PostIn s = gIn[t];

    PostOut r;
    r.res = uint4(0, 0, 0xFFFFFFFFu, 0xFFFFFFFFu);
    r.aux = uint4(0, 0, 0, 0);

    const uint mode = s.hdr.x;
    const uint pairIdxPlusMask = s.hdr.y;
    const uint pairIdx = pairIdxPlusMask & 0x1FFFFFFFu;

    const uint4 curPair = s.pairsGPU[pairIdx & 3u]; // model: 4 candidate pairs
    const uint cmIdx = curPair.x;
    const uint triangleIdx = curPair.y;
    const uint testOffset = curPair.z;

    const uint4 pair = (cmIdx == 0u) ? s.pairA : s.pairB;
    const uint nbPairsPerCM = pair.x;
    const uint pairStartIndex = pair.y;
    const uint orderedPairStartIndex = pair.z;
    const uint convexTriPairOffset = pairStartIndex + testOffset;
    r.aux.z = convexTriPairOffset;

    uint niWord = s.niIdx[convexTriPairOffset / 4u][convexTriPairOffset & 3u];
    r.res.x = niWord;

    const uint nbContacts = getNbContactsNic(niWord);
    if (nbContacts == 0u)
    {
        r.aux.x = 1u; // early out, nothing written
        gOut[t] = r;
        return;
    }

    const uint mask = pairIdxPlusMask & EMASK;
    r.aux.y = mask;

    bool needsProcessing = false;
    uint triangleIndex = 0xFFFFFFFFu;
    uint vertexIdx = 0xFFFFFFFFu;

    if (mode == 0u)
    {
        // ---- trimesh core (cu L182-365): real byte-offset blob walk ----
        const uint numVerts = blobWord(s, 0u);
        const uint numTris = blobWord(s, 4u);
        const uint numBV32 = blobWord(s, 12u); // header .w
        uint off = 16u;                        // sizeof(uint4)
        off += BV32_PACKED_BYTES * numBV32;
        off += 16u * numVerts;                 // float4 verts (skipped)
        const uint triIndicesOff = off;
        off += 16u * numTris;                  // uint4 per triangle
        const uint adjOff = off;
        off += 16u * numTris;
        off += 4u * numTris;                   // mGRB_faceRemap
        const uint accRefsOff = off;
        off += 4u * (numVerts + 1u);
        const uint refsOff = off;

        uint adjX = blobWord(s, adjOff + 16u * triangleIdx);
        uint adjY = blobWord(s, adjOff + 16u * triangleIdx + 4u);
        uint adjZ = blobWord(s, adjOff + 16u * triangleIdx + 8u);
        uint indX = blobWord(s, triIndicesOff + 16u * triangleIdx);
        uint indY = blobWord(s, triIndicesOff + 16u * triangleIdx + 4u);
        uint indZ = blobWord(s, triIndicesOff + 16u * triangleIdx + 8u);

        const bool flipNormal = (s.scale.x * s.scale.y * s.scale.z) < 0.0f; // hasNegativeDeterminant
        if (flipNormal)
        {
            const uint ta = adjX; adjX = adjZ; adjZ = ta;
            const uint ti = indY; indY = indZ; indZ = ti;
        }

        uint2 adjTriangles2 = uint2(0, 0);
        if (mask == EV0) { vertexIdx = indX; adjTriangles2 = uint2(adjX, adjZ); }
        else if (mask == EV1) { vertexIdx = indY; adjTriangles2 = uint2(adjX, adjY); }
        else if (mask == EV2) { vertexIdx = indZ; adjTriangles2 = uint2(adjY, adjZ); }
        else if (mask == EE01) { triangleIndex = adjX; if (triangleIndex == 0xFFFFFFFFu) needsProcessing = true; }
        else if (mask == EE12) { triangleIndex = adjY; if (triangleIndex == 0xFFFFFFFFu) needsProcessing = true; }
        else if (mask == EE02) { triangleIndex = adjZ; if (triangleIndex == 0xFFFFFFFFu) needsProcessing = true; }

        if (triangleIndex != 0xFFFFFFFFu)
        {
            needsProcessing = isBoundaryEdge(triangleIndex) ||
                              !findWithFlag(s, orderedPairStartIndex, nbPairsPerCM, triangleIndex, FACE_CONTACT_FLAG);
        }
        else if (vertexIdx != 0xFFFFFFFFu)
        {
            needsProcessing = true;
            if (!(isBoundaryEdge(adjTriangles2.x) && isBoundaryEdge(adjTriangles2.y)))
            {
                const uint startIdx = blobWord(s, accRefsOff + 4u * vertexIdx);
                const uint endIdx = blobWord(s, accRefsOff + 4u * (vertexIdx + 1u));
                for (uint idx = startIdx; idx < endIdx; ++idx)
                {
                    const uint triIdx = blobWord(s, refsOff + 4u * idx);
                    if (findWithFlag(s, orderedPairStartIndex, nbPairsPerCM, triIdx, FACE_CONTACT_FLAG) ||
                        findWithFlag(s, orderedPairStartIndex, nbPairsPerCM, triIdx, EDGE_CONTACT_FLAG))
                    {
                        needsProcessing = false;
                        break;
                    }
                }
            }
        }
    }
    else
    {
        // ---- heightfield core (cu L410-582) ----
        const uint nbRows = s.blob[0].x;
        const uint nbCols = s.blob[0].y;

        const float rowScale = s.scale.x;
        const float columnScale = s.scale.z;
        bool wrongHanded = false;
        if (columnScale < 0.0f) wrongHanded = !wrongHanded;
        if (rowScale < 0.0f) wrongHanded = !wrongHanded;
        const uint wh = wrongHanded ? 1u : 0u;
        // the handedness vector (cu L453-464) is computed but never used

        uint tv[3];
        uint adj[3];
        {
            uint i0, i1, i2;
            hfTriVertexIndices(s, triangleIdx, nbCols, i0, i1, i2);
            tv[0] = i0; tv[1u + wh] = i1; tv[2u - wh] = i2;
            uint a0, a1, a2;
            hfTriAdjacencyIndices(s, triangleIdx, nbCols, nbRows, a0, a1, a2);
            adj[wh != 0u ? 2u : 0u] = a0; adj[1] = a1; adj[wh != 0u ? 0u : 2u] = a2;
        }

        if (mask == EV0) vertexIdx = tv[0];
        else if (mask == EV1) vertexIdx = tv[1];
        else if (mask == EV2) vertexIdx = tv[2];
        else if (mask == EE01) triangleIndex = adj[0];
        else if (mask == EE12) triangleIndex = adj[1];
        else if (mask == EE02) triangleIndex = adj[2];

        if (triangleIndex != 0xFFFFFFFFu)
        {
            needsProcessing = isActiveEdge(triangleIndex) &&
                              !findWithFlag(s, orderedPairStartIndex, nbPairsPerCM, triangleIndex, FACE_CONTACT_FLAG);
        }
        else if (vertexIdx != 0xFFFFFFFFu)
        {
            const uint cellIdx = triangleIdx / 2u;
            const uint column = cellIdx % nbCols;
            const uint row = cellIdx / nbCols;
            const uint startRow = max(row, 1u) - 1u;
            const uint endRow = min(row, nbRows - 2u) + 1u;
            const uint startCol = max(column, 1u) - 1u;
            const uint endCol = min(column, nbCols - 2u) + 1u;

            needsProcessing = true;
            for (uint rr = startRow; rr <= endRow && needsProcessing; ++rr)
            {
                for (uint cc = startCol; cc <= endCol; ++cc)
                {
                    const uint cell2 = rr * nbCols + cc;
                    uint triIdx = cell2 * 2u;
                    uint i0, i1, i2;
                    hfTriVertexIndices(s, triIdx, nbCols, i0, i1, i2);
                    tv[0] = i0; tv[1u + wh] = i1; tv[2u - wh] = i2;
                    if (tv[0] == vertexIdx || tv[1] == vertexIdx || tv[2] == vertexIdx)
                    {
                        if (findWithFlag(s, orderedPairStartIndex, nbPairsPerCM, triIdx, FACE_CONTACT_FLAG))
                        {
                            needsProcessing = false;
                            break;
                        }
                    }
                    triIdx = triIdx + 1u;
                    hfTriVertexIndices(s, triIdx, nbCols, i0, i1, i2);
                    tv[0] = i0; tv[1u + wh] = i1; tv[2u - wh] = i2;
                    if (tv[0] == vertexIdx || tv[1] == vertexIdx || tv[2] == vertexIdx)
                    {
                        if (findWithFlag(s, orderedPairStartIndex, nbPairsPerCM, triIdx, FACE_CONTACT_FLAG))
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
        niWord = niWord & 0x0FFFFFFu; // SEVEN F's: clears count/deferred AND triangle-index bits 24-26

    r.res = uint4(niWord, needsProcessing ? 1u : 0u, triangleIndex, vertexIdx);
    gOut[t] = r;
}
