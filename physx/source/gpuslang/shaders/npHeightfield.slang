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

// npHeightfield.slang
//
// Slang port of the heightfield triangle-extraction core from
// gpunarrowphase/src/CUDA/heightfieldUtil.cuh: getTriangleVertexIndices (the
// per-triangle vertex-index computation, with the quad's diagonal chosen by the
// tessellation flag), getVertex (row/height/column of a sample) and hf2shapep
// (scale into the shape's local space).  Given a triangle index it produces the
// three sample vertex indices and the three shape-space triangle vertices.
//
// A PxHeightFieldSample is 4 bytes — PxI16 height, PxBitAndByte materialIndex0
// (whose top bit 0x80 is the tessellation flag), PxBitAndByte materialIndex1 —
// modelled here as one uint word: bits 0-15 = height (signed), 16-23 =
// materialIndex0, 24-31 = materialIndex1.

struct HfParams
{
    uint numTests;
    uint columns;
    uint rows;
    uint pad;
    float rowScale;
    float heightScale;
    float columnScale;
    float pad2;
}

struct HfOut
{
    uint4 idx;         // (v0idx, v1idx, v2idx, 0)
    float4 v0, v1, v2; // shape-space verts (.xyz)
}

[[vk::push_constant]]
ConstantBuffer<HfParams> gParams;

static const uint HF_BOUNDARY = 0xffffffffu;

RWStructuredBuffer<uint> gSamples;   // binding 0  packed PxHeightFieldSample per vertex
RWStructuredBuffer<uint> gTriIndex;  // binding 1  triangle index per test
RWStructuredBuffer<HfOut> gOut;      // binding 2
RWStructuredBuffer<uint4> gAdjOut;   // binding 3  (adj0, adj1, adj2, 0)

// getHeight: sign-extended PxI16 in the low 16 bits
float getHeight(uint vertexIndex)
{
    const int h = int(gSamples[vertexIndex] << 16) >> 16;
    return float(h);
}

bool isZerothVertexShared(uint vertexIndex)
{
    // tessFlag = materialIndex0 (byte 2) & 0x80
    return (gSamples[vertexIndex] & 0x800000u) != 0u;
}

float3 getVertex(uint vertexIndex, uint columns)
{
    const uint row = vertexIndex / columns;
    const uint col = vertexIndex % columns;
    return float3(float(row), getHeight(vertexIndex), float(col));
}

float3 hf2shapep(float3 v, float rowScale, float heightScale, float columnScale)
{
    return float3(v.x * rowScale, v.y * heightScale, v.z * columnScale);
}

// ---------------------------------------------------------------------------
// hfGetTriangle: mirrors getTriangleVertexIndices + getVertex + hf2shapep
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void hfGetTriangle(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint columns = gParams.columns;
    const uint triangleIndex = gTriIndex[t];

    const uint cell = triangleIndex >> 1;
    const bool firstTri = (triangleIndex & 0x1u) == 0u;

    uint i0, i1, i2;
    if (isZerothVertexShared(cell))
    {
        if (firstTri)
        {
            i0 = cell + columns;
            i1 = cell;
            i2 = cell + columns + 1u;
        }
        else
        {
            i0 = cell + 1u;
            i1 = cell + columns + 1u;
            i2 = cell;
        }
    }
    else
    {
        if (firstTri)
        {
            i0 = cell;
            i1 = cell + 1u;
            i2 = cell + columns;
        }
        else
        {
            i0 = cell + columns + 1u;
            i1 = cell + columns;
            i2 = cell + 1u;
        }
    }

    const float rs = gParams.rowScale, hs = gParams.heightScale, cs = gParams.columnScale;
    const float3 p0 = hf2shapep(getVertex(i0, columns), rs, hs, cs);
    const float3 p1 = hf2shapep(getVertex(i1, columns), rs, hs, cs);
    const float3 p2 = hf2shapep(getVertex(i2, columns), rs, hs, cs);

    HfOut o;
    o.idx = uint4(i0, i1, i2, 0u);
    o.v0 = float4(p0, 0.0f);
    o.v1 = float4(p1, 0.0f);
    o.v2 = float4(p2, 0.0f);
    gOut[t] = o;
}

// ---------------------------------------------------------------------------
// hfGetAdjacency: mirrors getTriangleAdjacencyIndices — the three edge-adjacent
// triangle indices of a heightfield triangle.  A missing neighbour (grid edge)
// is HF_BOUNDARY.  The diagonal orientation (shared vs unshared zeroth vertex)
// and the sub-triangle parity select one of four index patterns.
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void hfGetAdjacency(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint columns = gParams.columns;
    const uint rows = gParams.rows;
    const uint triangleIndex = gTriIndex[t];

    const uint cell = triangleIndex >> 1;
    const bool firstTri = (triangleIndex & 0x1u) == 0u;

    uint a0, a1, a2;
    if (isZerothVertexShared(cell))
    {
        if (firstTri)
        {
            a0 = HF_BOUNDARY;
            a1 = triangleIndex + 1u;
            a2 = HF_BOUNDARY;
            if (cell % columns != 0u)          a0 = triangleIndex - 1u;
            if (cell / columns != rows - 2u)    a2 = ((cell + columns) * 2u) + 1u;
        }
        else
        {
            a0 = HF_BOUNDARY;
            a1 = triangleIndex - 1u;
            a2 = HF_BOUNDARY;
            if (cell % columns < columns - 2u)  a0 = triangleIndex + 1u;
            if (cell >= columns - 1u)           a2 = (cell - columns) * 2u;
        }
    }
    else
    {
        if (firstTri)
        {
            a0 = HF_BOUNDARY;
            a1 = triangleIndex + 1u;
            a2 = HF_BOUNDARY;
            if (cell >= columns - 1u)           a0 = ((cell - columns) * 2u) + 1u;
            if (cell % columns != 0u)          a2 = triangleIndex - 1u;
        }
        else
        {
            a0 = HF_BOUNDARY;
            a1 = triangleIndex - 1u;
            a2 = HF_BOUNDARY;
            if (cell / columns != rows - 2u)    a0 = (cell + columns) * 2u;
            if (cell % columns < columns - 2u)  a2 = triangleIndex + 1u;
        }
    }

    gAdjOut[t] = uint4(a0, a1, a2, 0u);
}
