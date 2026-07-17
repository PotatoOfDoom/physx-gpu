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

// npGridIndex.slang
//
// Slang port of the dense-grid cell-index arithmetic from
// denseGridStandalone.cuh: getCellNr (L38-41), getCellCoords (L47-56) and
// getCellIndexSafe (L78-83).
//   getCellNr(nx, ny, xi, yi, zi) = (zi*ny + yi)*nx + xi
//   getCellCoords(nx, ny, cellNr) = { x = cellNr % nx; cellNr /= nx;
//                                     y = cellNr % ny; z = cellNr / ny; w = -1 }
//   getCellIndexSafe(...) = (any (idx+off) out of [0, numCells)) ? EMPTY_SUBGRID
//                           : getCellNr(nx, ny, idx.x+ox, idx.y+oy, idx.z+oz)
//
// Pure integer arithmetic (no float, so exact by construction).

static const uint EMPTY_SUBGRID = 0xffffffffu;

int getCellNr(int nx, int ny, int xi, int yi, int zi)
{
    return (zi * ny + yi) * nx + xi;
}

int4 getCellCoords(int nx, int ny, int cellNr)
{
    int4 result;
    result.x = cellNr % nx;
    cellNr /= nx;
    result.y = cellNr % ny;
    result.z = cellNr / ny;
    result.w = -1;
    return result;
}

uint getCellIndexSafe(int nx, int ny, int nz, int xi, int yi, int zi, int ox, int oy, int oz)
{
    if (xi + ox < 0 || yi + oy < 0 || zi + oz < 0 ||
        xi + ox >= nx || yi + oy >= ny || zi + oz >= nz)
        return EMPTY_SUBGRID;
    return uint(getCellNr(nx, ny, xi + ox, yi + oy, zi + oz));
}

struct GiIn
{
    int4 grid;    // nx, ny, nz, _
    int4 idx;     // xi, yi, zi, _
    int4 off;     // ox, oy, oz, _
}

struct GiOut
{
    int4 cellNr;   // cellNr, _, _, _
    int4 coords;   // getCellCoords(cellNr)
    uint4 safe;    // getCellIndexSafe, _, _, _
}

struct GiParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<GiParams> gParams;

RWStructuredBuffer<GiIn> gIn;    // binding 0
RWStructuredBuffer<GiOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void gridIndex(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const GiIn s = gIn[tid];

    const int nx = s.grid.x, ny = s.grid.y, nz = s.grid.z;
    const int xi = s.idx.x, yi = s.idx.y, zi = s.idx.z;

    const int cellNr = getCellNr(nx, ny, xi, yi, zi);
    const int4 coords = getCellCoords(nx, ny, cellNr);
    const uint safe = getCellIndexSafe(nx, ny, nz, xi, yi, zi, s.off.x, s.off.y, s.off.z);

    GiOut o;
    o.cellNr = int4(cellNr, 0, 0, 0);
    o.coords = coords;
    o.safe = uint4(safe, 0u, 0u, 0u);
    gOut[tid] = o;
}
