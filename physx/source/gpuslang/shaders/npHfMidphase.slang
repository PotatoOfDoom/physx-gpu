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

// npHfMidphase.slang
//
// Slang port of the heightfield midphase pair-generation core from
// gpunarrowphase/src/CUDA/heightfieldUtil.cuh: heightfieldComputePairs (count)
// and heightfieldOutputPairs (emit).  Given the overlap AABB projected onto the
// heightfield grid (minRow..maxRow x minColumn..maxColumn) and a y-range
// (miny..maxy), each covered quad is visited twice (once per triangle/material).
// A quad whose four corner heights are all strictly above maxy (con0) or all
// strictly below miny (con1) is culled, as is a triangle whose material is
// eHOLE (127).  Each surviving triangle contributes one pair — counted by
// hfmComputePairs, and emitted as uint4(cmInd, elementInd, triangleIdx, 0) by
// hfmOutputPairs (with the per-query output-slot cap).  triangleIdx =
// (vertexIndex<<1) for the lower (material0) triangle, +1 for the upper
// (material1) triangle.
//
// A PxHeightFieldSample is 4 bytes — PxI16 height, PxBitAndByte materialIndex0
// (top bit 0x80 = tessellation flag), PxBitAndByte materialIndex1 (top bit =
// reserved flag) — modelled here as one uint word: bits 0-15 = height (signed),
// 16-23 = materialIndex0 byte, 24-31 = materialIndex1 byte.  A PxBitAndByte's
// value masks off the top flag bit, so the material index is byte & 0x7f (and
// eHOLE = 127 = 0x7f is the largest representable index).

struct HfmParams
{
    uint numTests;
    uint nbRows;
    uint nbCols;
    uint pairsPerTest; // output-slot cap per query (hfmOutputPairs)
}

// One overlap query per test.
struct HfQuery
{
    uint minColumn;
    uint maxColumn;
    uint minRow;
    uint maxRow;
    uint miny;
    uint maxy;
    uint cmInd;
    uint elementInd;
}

[[vk::push_constant]]
ConstantBuffer<HfmParams> gParams;

static const uint HF_HOLE = 127u;

RWStructuredBuffer<uint> gSamples;      // binding 0  packed PxHeightFieldSample per vertex
RWStructuredBuffer<HfQuery> gQueries;   // binding 1  one overlap query per test
RWStructuredBuffer<uint> gCounts;       // binding 2  pair count per test
RWStructuredBuffer<uint4> gPairs;       // binding 3  emitted pairs, pairsPerTest slots per test

// getHeight: sign-extended PxI16 in the low 16 bits
float getHeight(uint vertexIndex)
{
    const int h = int(gSamples[vertexIndex] << 16) >> 16;
    return float(h);
}

// material index (flag bit masked off, matching PxBitAndByte::operator storageType)
uint getMaterial0(uint vertexIndex) { return (gSamples[vertexIndex] >> 16) & 0x7fu; }
uint getMaterial1(uint vertexIndex) { return (gSamples[vertexIndex] >> 24) & 0x7fu; }

// ---------------------------------------------------------------------------
// hfmComputePairs: mirrors heightfieldComputePairs — counts the surviving
// (non-culled, non-hole) triangle pairs in the query's grid range.
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void hfmComputePairs(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint nbCols = gParams.nbCols;
    const HfQuery q = gQueries[t];

    const float maxy = float(q.maxy);
    const float miny = float(q.miny);

    uint nbPairs = 0u;
    const uint columnSpan = q.maxColumn - q.minColumn;
    const uint totalNumProcessed = (q.maxRow - q.minRow) * columnSpan * 2u;

    for (uint i = 0u; i < totalNumProcessed; ++i)
    {
        const uint index = i / 2u;
        const uint vertexIndex = (q.minRow + index / columnSpan) * nbCols + (q.minColumn + index % columnSpan);
        const float h0 = getHeight(vertexIndex);
        const float h1 = getHeight(vertexIndex + 1u);
        const float h2 = getHeight(vertexIndex + nbCols);
        const float h3 = getHeight(vertexIndex + nbCols + 1u);
        const bool con0 = maxy < h0 && maxy < h1 && maxy < h2 && maxy < h3;
        const bool con1 = miny > h0 && miny > h1 && miny > h2 && miny > h3;

        if (!(con0 || con1))
        {
            const bool isMaterial1 = (i & 1u) != 0u;
            const uint material = isMaterial1 ? getMaterial1(vertexIndex) : getMaterial0(vertexIndex);
            if (material != HF_HOLE)
                nbPairs++;
        }
    }

    gCounts[t] = nbPairs;
}

// ---------------------------------------------------------------------------
// hfmOutputPairs: mirrors heightfieldOutputPairs — emits each surviving pair as
// uint4(cmInd, elementInd, triangleIdx, 0) into the query's output slots, up to
// the per-query cap, and records how many were written.
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void hfmOutputPairs(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint nbCols = gParams.nbCols;
    const uint pairsPerTest = gParams.pairsPerTest;
    const HfQuery q = gQueries[t];

    const float maxy = float(q.maxy);
    const float miny = float(q.miny);

    const uint startOffset = t * pairsPerTest;
    const uint size = startOffset + pairsPerTest;
    uint writeIndex = startOffset;

    const uint columnSpan = q.maxColumn - q.minColumn;
    const uint totalNumProcessed = (q.maxRow - q.minRow) * columnSpan * 2u;

    for (uint i = 0u; i < totalNumProcessed && writeIndex < size; ++i)
    {
        const uint index = i / 2u;
        const uint vertexIndex = (q.minRow + index / columnSpan) * nbCols + (q.minColumn + index % columnSpan);
        const float h0 = getHeight(vertexIndex);
        const float h1 = getHeight(vertexIndex + 1u);
        const float h2 = getHeight(vertexIndex + nbCols);
        const float h3 = getHeight(vertexIndex + nbCols + 1u);
        const bool con0 = maxy < h0 && maxy < h1 && maxy < h2 && maxy < h3;
        const bool con1 = miny > h0 && miny > h1 && miny > h2 && miny > h3;

        if (!(con0 || con1))
        {
            const bool isMaterial1 = (i & 1u) != 0u;
            const uint material = isMaterial1 ? getMaterial1(vertexIndex) : getMaterial0(vertexIndex);
            if (material != HF_HOLE)
            {
                const uint triangleIdx = isMaterial1 ? ((vertexIndex << 1) + 1u) : (vertexIndex << 1);
                gPairs[writeIndex] = uint4(q.cmInd, q.elementInd, triangleIdx, 0u);
                writeIndex++;
            }
        }
    }

    gCounts[t] = writeIndex - startOffset;
}
