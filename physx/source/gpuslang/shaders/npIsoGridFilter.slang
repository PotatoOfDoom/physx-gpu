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

// npIsoGridFilter.slang
//
// Slang port of the two isosurface dense-grid filters (isosurfaceExtraction.cu)
// over a self-contained grid of up to 64 cells:
//
//   gridFilterGauss (L1097-1125):
//     w[3] = { neighborWeight, 1, neighborWeight }
//     for x in -1..1: for y in -1..1: for z in -1..1:      // x outer, z inner
//       weight = w[x+1] * w[y+1] * w[z+1]                  // left-to-right
//       weightSum += weight                                 // ALL taps counted,
//       m += weight * getGridValueSafe(source, xyz, x,y,z)  // even OOB (value 0)
//     dest = m / weightSum                                  // plain division
//
//   gridFilterDilateErode (L1138-1153):
//     m = sign * source[threadIndex]
//     for the same 27 taps: m = PxMax(m, sign * getGridValueSafe(...))
//     dest = sign * m
//
// Shared pieces: getCellCoords (denseGridStandalone.cuh L47-56: x = t % nx;
// t /= nx; y = t % ny; z = t / ny) and getGridValueSafe (L91-96: any
// coordinate+offset outside [0, numCells) returns 0.0f, else
// source[getCellNr] with getCellNr = (zi*ny + yi)*nx + xi).  Both loops are
// strict x-outer/y-mid/z-inner folds (order-sensitive float accumulation).
// PxMax float = selectMax = a>b?a:b.  Float mul/add/div NoContraction-stamped;
// +-0 compared with eqFZ.

float pmax(float a, float b) { return (a > b) ? a : b; }   // PxMax = selectMax

struct GfIn
{
    float4 vals[16];   // up to 64 grid cells, vals[i>>2][i&3]
    uint4 dims;        // numCellsX, numCellsY, numCellsZ, threadIndex
    float4 prm;        // neighborWeight, sign (+1 dilate / -1 erode), _, _
}

struct GfOut
{
    float4 r;          // gauss result, dilateErode result, _, _
}

struct GfParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<GfParams> gParams;

RWStructuredBuffer<GfIn> gIn;    // binding 0
RWStructuredBuffer<GfOut> gOut;  // binding 1

float valAt(GfIn s, int i) { return s.vals[uint(i) >> 2][uint(i) & 3u]; }

// denseGridStandalone.cuh getCellNr (L38-41).
int getCellNr(int nx, int ny, int xi, int yi, int zi) { return (zi * ny + yi) * nx + xi; }

// denseGridStandalone.cuh getGridValueSafe (L91-96): 0 outside the grid.
float getGridValueSafe(GfIn s, int nx, int ny, int nz, int x, int y, int z, int ox, int oy, int oz)
{
    if (x + ox < 0 || y + oy < 0 || z + oz < 0 || x + ox >= nx || y + oy >= ny || z + oz >= nz)
        return 0.0f;
    return valAt(s, getCellNr(nx, ny, x + ox, y + oy, z + oz));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void isoGridFilter(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const GfIn s = gIn[tid];

    const int nx = int(s.dims.x), ny = int(s.dims.y), nz = int(s.dims.z);
    const int threadIndex = int(s.dims.w);
    const float neighborWeight = s.prm.x;
    const float sign = s.prm.y;

    // getCellCoords (L47-56)
    int t = threadIndex;
    const int cx = t % nx; t /= nx;
    const int cy = t % ny;
    const int cz = t / ny;

    // gridFilterGauss
    float w0 = neighborWeight;   // w[0]
    float w1 = 1.0f;             // w[1]
    float w2 = w0;               // w[2]

    float weightSum = 0.0f;
    float m = 0.0f;
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y)
            for (int z = -1; z <= 1; ++z)
            {
                const float wx = (x == -1) ? w0 : (x == 0) ? w1 : w2;
                const float wy = (y == -1) ? w0 : (y == 0) ? w1 : w2;
                const float wz = (z == -1) ? w0 : (z == 0) ? w1 : w2;
                const float weight = wx * wy * wz;   // w[x+1]*w[y+1]*w[z+1], left-to-right
                weightSum += weight;
                m += weight * getGridValueSafe(s, nx, ny, nz, cx, cy, cz, x, y, z);
            }
    const float gauss = m / weightSum;

    // gridFilterDilateErode
    float md = sign * valAt(s, threadIndex);
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y)
            for (int z = -1; z <= 1; ++z)
                md = pmax(md, sign * getGridValueSafe(s, nx, ny, nz, cx, cy, cz, x, y, z));
    const float dilateErode = sign * md;

    GfOut o;
    o.r = float4(gauss, dilateErode, 0.0f, 0.0f);
    gOut[tid] = o;
}
