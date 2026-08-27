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

// pxgGridCalRef.h -- the particle GRID HASH, transcribed from the ENGINE.
//
// WHY THIS IS ADMISSIBLE AS AN ORACLE, which is the only thing that matters
// about a reference: every line below comes from the CUDA in this repository,
// cited per function.  A host mirror of our own Slang would only prove the two
// copies agree with each other -- which a differential between two copies of the
// same mistake also proves.  gridCal.cuh is an ENGINE header and
// cudaParticleSystem.cu an ENGINE kernel file, so this is the same relationship
// every Phase-1 battery has to its reference.
//
// WHY IT IS A HEADER (P9.24).  P9.18 wrote this mirror inside
// testPsConvexContactGen.cpp to pin `calcGridHashInBounds`, which until then had
// four live call sites in psContactGenBda and was verified by none of them --
// every particle contact-gen fixture ran a 1x1x1 grid and said so out loud
// ("gridHash is always 0").  With one cell the whole composition collapses:
// calcWrappedGridRangeSize clamps the range to (1,1,1), calcGridOffsetInRange
// returns (0,0,0) for every cell offset, and calcGridHash masks each component
// with (1-1) == 0, so ANY implementation that returns zero passes.
//
// P9.18 closed one of the three fixtures and wrote down that the other two were
// still open, with the reason a shared home was needed: "The engine-derived host
// mirror already exists and should be SHARED rather than transcribed again."  A
// third transcription would be a third place for the same reading of the same
// CUDA to drift.
//
// Integer arithmetic throughout except the initial floor.  Callers are expected
// to keep their overlap bounds away from a cell boundary -- otherwise the
// rounding is a knife edge and the oracle stops being exact.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace pxgtest
{
namespace gridcal
{

struct I3 { int32_t x, y, z; };
struct U3 { uint32_t x, y, z; };

// gridCal.cuh:34
inline I3 calcGridPos(const float p[3], float cellWidth)
{
    return I3{ int32_t(std::floor(p[0] / cellWidth)),
               int32_t(std::floor(p[1] / cellWidth)),
               int32_t(std::floor(p[2] / cellWidth)) };
}

// gridCal.cuh:46 -- the mask is SIGNED, exactly as the CUDA writes it.
inline uint32_t calcGridHash(I3 g, U3 s)
{
    g.x = g.x & int32_t(s.x - 1u);
    g.y = g.y & int32_t(s.y - 1u);
    g.z = g.z & int32_t(s.z - 1u);
    return uint32_t((g.z * int32_t(s.y)) * int32_t(s.x) + g.y * int32_t(s.x) + g.x);
}

// gridCal.cuh:86
inline U3 calcWrappedGridRangeSize(I3 mn, I3 mx, U3 s)
{
    auto cl = [](int32_t v, int32_t hi) { return uint32_t(v < 0 ? 0 : (v > hi ? hi : v)); };
    return U3{ cl(mx.x - mn.x + 1, int32_t(s.x)),
               cl(mx.y - mn.y + 1, int32_t(s.y)),
               cl(mx.z - mn.z + 1, int32_t(s.z)) };
}

// gridCal.cuh:97
inline U3 calcGridOffsetInRange(U3 rs, uint32_t off)
{
    return U3{ off % rs.x, (off / rs.x) % rs.y, off / (rs.x * rs.y) };
}

// cudaParticleSystem.cu:96 -- the composition, which is the thing being pinned.
inline uint32_t calcGridHashInBounds(const float bmn[3], const float bmx[3],
                                     float cellWidth, uint32_t cellOffset, U3 gridSize)
{
    const I3 gpMin = calcGridPos(bmn, cellWidth);          // calcGridRange, gridCal.cuh:105
    const I3 gpMax = calcGridPos(bmx, cellWidth);
    const U3 rs = calcWrappedGridRangeSize(gpMin, gpMax, gridSize);
    const U3 off = calcGridOffsetInRange(rs, cellOffset);
    const I3 gp = I3{ gpMin.x + int32_t(off.x), gpMin.y + int32_t(off.y), gpMin.z + int32_t(off.z) };
    return calcGridHash(gp, gridSize);
}

// The NUMBER of cells a pair's overlap region spans -- the work-item count the
// engine allocates for it (calcWrappedGridRangeSize's product).
inline uint32_t cellsInBounds(const float bmn[3], const float bmx[3],
                              float cellWidth, U3 gridSize)
{
    const U3 rs = calcWrappedGridRangeSize(calcGridPos(bmn, cellWidth),
                                           calcGridPos(bmx, cellWidth), gridSize);
    return rs.x * rs.y * rs.z;
}

// combine() + fattenFast(): cudaParticleSystem.cu:84-93 and PxBounds3.h:442.
// The OVERLAP region of the two shapes, then grown by the pair's contact
// distance.  Takes the two bounds as arguments rather than indexing a fixture's
// table, which is what lets three fixtures share it.
inline void overlapBound(const float bmnA[3], const float bmxA[3],
                         const float bmnB[3], const float bmxB[3],
                         float cDistance, float omn[3], float omx[3])
{
    for (int k = 0; k < 3; ++k)
    {
        omn[k] = std::max(bmnA[k], bmnB[k]) - cDistance;
        omx[k] = std::min(bmxA[k], bmxB[k]) + cDistance;
    }
}

} // namespace gridcal
} // namespace pxgtest
