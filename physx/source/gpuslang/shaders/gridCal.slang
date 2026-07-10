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

// gridCal.slang
//
// Slang port of the particle hash-grid helpers from
// gpucommon/src/CUDA/gridCal.cuh — the foundation of every particle-system
// spatial query:
//
// - calcGridPos: floor(pos / cellWidth) per component (int cell coords).
// - calcGridHash: power-of-two wrap via `& (gridSize-1)`, then the linear
//   z-major cell address.
// - calcGridHashPeriodic: single-wrap periodic adjustment (+/- periodGridSize
//   on < 0 / >= period), then the same linear address over the NON-periodic
//   grid size.
// - calcWrappedGridRangeSize: (max - min + 1) per component clamped to
//   [0, gridSize].
// - calcGridOffsetInRange: linear offset -> 3D offset within a range
//   (x = off % sx, y = (off / sx) % sy, z = off / (sx*sy)).
// - calcGridRange: calcGridPos of the bounds' min and max corners.
// - getParticleSpeculativeContactVolume: the forward-projection midpoint and
//   covering radius (PGS scale 1, TGS scale 0.5, diffuse scale 1):
//   offset = (predicted-current)*scaleInv*0.5, pos = current+offset,
//   radius = |offset| + contactDist.
//
// All grid outputs are integers (floor decisions are deterministic away from
// cell boundaries); the speculative volume is continuous.

struct GcParams { uint numTests; uint pad0, pad1, pad2; }

struct GcIn
{
    float4 pos_cw;          // xyz = particle pos, w = cellWidth
    float4 bmin;            // bounds minimum
    float4 bmax;            // bounds maximum
    int4 gridPosIn_flags;   // xyz = gridPos for the periodic hash, w = flags (1 = isDiffuse, 2 = isTGS)
    int4 gridSize_off;      // xyz = grid size (pow2 for calcGridHash), w = linear offset for calcGridOffsetInRange
    int4 periodSize;        // xyz = periodic grid size
    float4 curPos;          // current position (speculative volume)
    float4 predPos_cd;      // xyz = predicted position, w = contactDist
}

struct GcOut
{
    int4 gridPos_hash;      // xyz = calcGridPos(pos, cw), w = calcGridHash(that, gridSize)
    int4 rangeMin_ph;       // xyz = calcGridRange min, w = calcGridHashPeriodic(gridPosIn, gridSize, periodSize)
    int4 rangeMax;          // xyz = calcGridRange max
    uint4 rangeSize;        // xyz = calcWrappedGridRangeSize(rangeMin, rangeMax, gridSize)
    uint4 offsetInRange;    // xyz = calcGridOffsetInRange(rangeSize, offset)
    float4 cVolume;         // xyz = cVolumePos, w = radius
}

[[vk::push_constant]]
ConstantBuffer<GcParams> gParams;

RWStructuredBuffer<GcIn> gIn;    // binding 0
RWStructuredBuffer<GcOut> gOut;  // binding 1

static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_PGS = 1.0f;
static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_TGS = 0.5f;
static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_DIFFUSE = 1.0f;

int3 calcGridPos(float3 particlePos, float cellWidth)
{
    return int3(floor(particlePos.x / cellWidth),
                floor(particlePos.y / cellWidth),
                floor(particlePos.z / cellWidth));
}

// wrap-around address (gridSize must be a power of two)
uint calcGridHash(int3 gridPos, uint3 gridSize)
{
    gridPos.x = gridPos.x & int(gridSize.x - 1);
    gridPos.y = gridPos.y & int(gridSize.y - 1);
    gridPos.z = gridPos.z & int(gridSize.z - 1);
    return uint((gridPos.z * int(gridSize.y)) * int(gridSize.x) + gridPos.y * int(gridSize.x) + gridPos.x);
}

uint calcGridHashPeriodic(int3 gridPos, int3 gridSize, int3 periodGridSize)
{
    if (gridPos.x < 0) gridPos.x += periodGridSize.x;
    else if (gridPos.x >= periodGridSize.x) gridPos.x -= periodGridSize.x;
    if (gridPos.y < 0) gridPos.y += periodGridSize.y;
    else if (gridPos.y >= periodGridSize.y) gridPos.y -= periodGridSize.y;
    if (gridPos.z < 0) gridPos.z += periodGridSize.z;
    else if (gridPos.z >= periodGridSize.z) gridPos.z -= periodGridSize.z;
    return uint((gridPos.z * gridSize.y) * gridSize.x + gridPos.y * gridSize.x + gridPos.x);
}

uint3 calcWrappedGridRangeSize(int3 gridRangeMin, int3 gridRangeMax, uint3 gridSize)
{
    const uint x = uint(clamp(gridRangeMax.x - gridRangeMin.x + 1, 0, int(gridSize.x)));
    const uint y = uint(clamp(gridRangeMax.y - gridRangeMin.y + 1, 0, int(gridSize.y)));
    const uint z = uint(clamp(gridRangeMax.z - gridRangeMin.z + 1, 0, int(gridSize.z)));
    return uint3(x, y, z);
}

uint3 calcGridOffsetInRange(uint3 gridRangeSize, uint offset)
{
    const uint x = offset % gridRangeSize.x;
    const uint y = (offset / gridRangeSize.x) % gridRangeSize.y;
    const uint z = offset / (gridRangeSize.x * gridRangeSize.y);
    return uint3(x, y, z);
}

void calcGridRange(out int3 gridPosMin, out int3 gridPosMax, float3 bmin, float3 bmax, float cellWidth)
{
    gridPosMin = calcGridPos(bmin, cellWidth);
    gridPosMax = calcGridPos(bmax, cellWidth);
}

float getParticleSpeculativeContactVolume(out float3 cVolumePos,
    float3 currentPos, float3 predictedPos, float contactDist, bool isDiffuse, bool isTGS)
{
    const float sScaleInvPGS = 1.0f / PARTICLE_FORWARD_PROJECTION_STEP_SCALE_PGS;
    const float sScaleInvTGS = 1.0f / PARTICLE_FORWARD_PROJECTION_STEP_SCALE_TGS;
    const float sScaleInvDiffuse = 1.0f / PARTICLE_FORWARD_PROJECTION_STEP_SCALE_DIFFUSE;
    float scaleInv;
    if (isDiffuse)
        scaleInv = sScaleInvDiffuse;
    else
        scaleInv = isTGS ? sScaleInvTGS : sScaleInvPGS;
    const float3 cVolumeOffset = (predictedPos - currentPos) * scaleInv * 0.5f;
    cVolumePos = currentPos + cVolumeOffset;
    return length(cVolumeOffset) + contactDist;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctGridCal(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const GcIn e = gIn[i];
    const float cw = e.pos_cw.w;
    const uint3 gridSize = uint3(e.gridSize_off.xyz);

    GcOut o;
    const int3 gp = calcGridPos(e.pos_cw.xyz, cw);
    o.gridPos_hash = int4(gp, int(calcGridHash(gp, gridSize)));

    int3 rMin, rMax;
    calcGridRange(rMin, rMax, e.bmin.xyz, e.bmax.xyz, cw);
    o.rangeMin_ph = int4(rMin, int(calcGridHashPeriodic(e.gridPosIn_flags.xyz, e.gridSize_off.xyz, e.periodSize.xyz)));
    o.rangeMax = int4(rMax, 0);

    const uint3 rs = calcWrappedGridRangeSize(rMin, rMax, gridSize);
    o.rangeSize = uint4(rs, 0);
    o.offsetInRange = uint4(calcGridOffsetInRange(rs, uint(e.gridSize_off.w)), 0);

    float3 cvp;
    const float rad = getParticleSpeculativeContactVolume(cvp, e.curPos.xyz, e.predPos_cd.xyz, e.predPos_cd.w,
                                                          (e.gridPosIn_flags.w & 1) != 0, (e.gridPosIn_flags.w & 2) != 0);
    o.cVolume = float4(cvp, rad);
    gOut[i] = o;
}
