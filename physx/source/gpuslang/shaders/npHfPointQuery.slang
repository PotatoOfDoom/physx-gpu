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

// npHfPointQuery.slang
//
// Slang port of the heightfield grid-coordinate helpers and point query from
// gpunarrowphase/src/CUDA/heightfieldUtil.cuh: getMinRow / getMaxRow /
// getMinColumn / getMaxColumn (clamp floor/ceil of a local coordinate into the
// valid cell/vertex range) and testLocalPoint (bilinear interpolation of the
// four surrounding sample heights, and whether the query point sits below the
// interpolated surface).
//
// The sample word packs a signed PxI16 height in bits 0-15 (the material bytes
// in 16-31 are not consulted here).

struct HfqParams
{
    uint numTests;
    uint rows;
    uint columns;
    uint pad;
}

[[vk::push_constant]]
ConstantBuffer<HfqParams> gParams;

RWStructuredBuffer<uint> gSamples;    // binding 0  packed PxHeightFieldSample per vertex
RWStructuredBuffer<float4> gPoints;   // binding 1  query point (.xyz; grid range uses .x/.z)
RWStructuredBuffer<uint4> gRange;     // binding 2  (minRow, maxRow, minColumn, maxColumn)
RWStructuredBuffer<float4> gHeightHit; // binding 3  (height, resultBool, insideBool, 0)

// getHeight: sign-extended PxI16 in the low 16 bits
float getHeight(uint vertexIndex)
{
    const int h = int(gSamples[vertexIndex] << 16) >> 16;
    return float(h);
}

// getMinRow/getMaxRow/getMinColumn/getMaxColumn:
//   (PxU32)PxClamp(PxI32(floor/ceil(coord)), 0, rows/cols - {2,1})
uint getMinRow(float x, uint rows)      { return uint(clamp(int(floor(x)), 0, int(rows) - 2)); }
uint getMaxRow(float x, uint rows)      { return uint(clamp(int(ceil(x)),  0, int(rows) - 1)); }
uint getMinColumn(float z, uint cols)   { return uint(clamp(int(floor(z)), 0, int(cols) - 2)); }
uint getMaxColumn(float z, uint cols)   { return uint(clamp(int(ceil(z)),  0, int(cols) - 1)); }

// ---------------------------------------------------------------------------
// hfGridRange: the four clamped grid-coordinate helpers for one (x,z) point.
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void hfGridRange(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint rows = gParams.rows, cols = gParams.columns;
    const float4 p = gPoints[t];
    gRange[t] = uint4(getMinRow(p.x, rows), getMaxRow(p.x, rows),
                      getMinColumn(p.z, cols), getMaxColumn(p.z, cols));
}

// ---------------------------------------------------------------------------
// hfTestLocalPoint: mirrors testLocalPoint — bilinear height at (x,z) and
// whether the interpolated surface is above the query point's y.  When the
// point is outside [0,rows-1] x [0,columns-1] the function returns false and
// the height is undefined; insideBool records which case applies so the caller
// only trusts the height when inside.
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void hfTestLocalPoint(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint rows = gParams.rows, cols = gParams.columns;
    const float4 pnt = gPoints[t];

    if (pnt.x < 0.0f || pnt.x > float(rows - 1u) || pnt.z < 0.0f || pnt.z > float(cols - 1u))
    {
        gHeightHit[t] = float4(0.0f, 0.0f, 0.0f, 0.0f); // outside: result=false, inside=false
        return;
    }

    const uint row = getMinRow(pnt.x, rows);
    const uint column = getMinColumn(pnt.z, cols);
    const uint vertex = row * cols + column;
    const float h0 = getHeight(vertex);
    const float h1 = getHeight(vertex + 1u);
    const float h2 = getHeight(vertex + cols);
    const float h3 = getHeight(vertex + cols + 1u);
    const float s = pnt.x - floor(pnt.x);
    const float tt = pnt.z - floor(pnt.z);
    const float h = h0 * (1.0f - s) * (1.0f - tt) + h1 * s * (1.0f - tt)
                  + h2 * (1.0f - s) * tt + h3 * s * tt;

    const float result = (h > pnt.y) ? 1.0f : 0.0f;
    gHeightHit[t] = float4(h, result, 1.0f, 0.0f);
}
