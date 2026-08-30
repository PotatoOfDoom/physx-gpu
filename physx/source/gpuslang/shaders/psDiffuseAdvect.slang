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

// psDiffuseAdvect.slang -- P9.13e, the two kernels that ADVECT and AGE diffuse
// particles.  Ported from gpusimulationcontroller/src/CUDA/diffuseParticles.cu,
// both read in full:
//
//   ps_diffuseParticleUpdatePBF (:164)  sample the FLUID's velocity field around
//       each diffuse particle and store the weighted average plus the neighbour
//       COUNT
//   ps_diffuseParticleCompact   (:275)  classify by that count, integrate, age,
//       and compact the survivors back into the user buffer
//
// They are one tranche because the second consumes exactly what the first
// produces, and because the buffer they pass it through is not the one its name
// suggests (V3 below).
//
// ---------------------------------------------------------------------------
// THE OVERLOADED BUFFER, which is the thing to understand first
// ---------------------------------------------------------------------------
// UpdatePBF writes its velocity average into
// `mDiffuseSortedOriginPos_LifeTime` -- the SORTED ORIGIN POSITION array -- and
// the CUDA says so in a one-line comment ("Overloading this buffer to store the
// new velocity...").  Compact then reads that same array as `velAvgs`.  So
// between these two kernels the array does not hold origin positions at all,
// and its `.w` holds a NEIGHBOUR COUNT rather than a lifetime.
//
// A port that writes the velocity into `mDiffuseSortedVel` -- the obvious
// choice, and a real array that exists -- compiles, runs, and leaves Compact
// reading last frame's origin positions as if they were velocities.
//
// ---------------------------------------------------------------------------
// TRAPS -- ps_diffuseParticleUpdatePBF
// ---------------------------------------------------------------------------
// V1 -- grid (ceil(maxDiffuse/256), nbActiveSystems, 1): the system comes from
//     blockIdx.**y**.  Compact, twenty lines below, takes it from blockIdx.z.
//     Two adjacent kernels, two different axes.
//
// V2 -- the count is the SYSTEM's `*mNumDiffuseParticles` (what P9.13c's Sum
//     published), not a per-buffer count: this kernel works on the flat sorted
//     array and knows nothing about buffers.
//
// V3 -- the output is the overloaded origin array; see above.
//
// V4 -- THE NEIGHBOURS ARE FLUID PARTICLES, NOT DIFFUSE ONES.  cellStart/
//     cellEnd, mSortedPositions_InvMass and mSortedVelocities are the FLUID
//     grid and the FLUID arrays.  The diffuse particle only contributes its own
//     position.  The diffuse grid (mDiffuseCellStart/...) is not touched here.
//
// V5 -- `end = (mFlags & eFULL_DIFFUSE_ADVECTION) ? 3 : 1`, and with end == 1
//     the loops run x = y = z = 0 only, i.e. offset[0] == 0: the particle's OWN
//     cell and nothing else.  eFULL_DIFFUSE_ADVECTION is 1<<2 -- the constant
//     P9.10 pinned as a trap against eENABLE_SPECULATIVE_CCD's 1<<3.
//
// V6 -- the offsets are `{0, -1, +1}` IN THAT ORDER, not {-1, 0, +1}.  With
//     end == 3 the set covered is the same 3x3x3, but the ITERATION ORDER
//     differs -- and the loop can stop early (V7), so the order decides WHICH
//     16 neighbours are taken.
//
// V7 -- `maxNeighbors = 16` with a `goto` out of the triple loop.  The count
//     that reaches the output is therefore capped at 16, and the 16 taken are
//     the first 16 in the order V6 fixes.
//
// V8 -- the weight is WDiffuse(sqrt(dSq), invContactDistance) = 1 - h * invR,
//     a linear falloff, applied to the SQUARE-ROOTED distance.
//
// V9 -- the stored `.w` is `PxReal(numNeighbors)` -- an integer count widened
//     to float, which Compact then compares against 4 and 8.
//
// ---------------------------------------------------------------------------
// TRAPS -- ps_diffuseParticleCompact
// ---------------------------------------------------------------------------
// K1 -- grid (x, maxDiffuseBuffersPerSystem, nbActiveSystems): system from
//     blockIdx.**z**, buffer from .y.  See V1.
//
// K2 -- the count is the buffer's PUBLISHED slot `mNumDiffuseParticles[0]`, and
//     the survivors are appended into the STAGING slot [1] -- so this kernel
//     re-uses the two-slot protocol P9.13c documented, and P9.13c's Copy is
//     what publishes the compacted count next frame.
//
// K3 -- `index = pi + buffer.mStartIndex`, and mStartIndex is the field P9.13c's
//     UpdateUnsorted writes (and, for an empty buffer, deliberately does not).
//
// K4 -- THREE ARRAYS, THREE INDEX SPACES, in four consecutive lines:
//         diffusePositions[sortedInd]     the SORTED current position
//         diffusePositionsOld[index]      the UNSORTED origin position
//         velAvgs[sortedInd]              the SORTED overloaded average
//     Using one index for all three compiles and produces plausible motion.
//
// K5 -- the velocity is RECONSTRUCTED, `vel = (pos - oldPos) / dt`, not read
//     from any velocity array.
//
// K6 -- the classification is on the NEIGHBOUR COUNT: < 4 spray, < 8 foam, else
//     bubble.  Three different integrations, and the thresholds are on a float
//     that holds an integer.
//
// K7 -- the bubble term is `vel - (1 + buoyancy) * gravity * dt +
//     bubbleDrag * (velAvg - vel)`.  The `1 +` and the MINUS are both
//     load-bearing: gravity points down, so subtracting it is what makes a
//     bubble rise.
//
// K8 -- the speed clamp is `PxMin(|v|, maxVel) * v.getNormalized()`, guarded by
//     `magnitudeSquared() > 0`, and PhysX's getNormalized() itself returns zero
//     for a zero vector.
//
// K9 -- `newPosCorr = pos + (newVel - vel) * dt` is a CORRECTION to the
//     already-predicted position, not a fresh integration from oldPos.
//
// K10 -- `lifeTime = max(xi4.w - dt, 0)` and the survivor test is
//     `lifeTime > 0`, so a particle whose lifetime lands exactly on zero dies
//     this frame.
//
// K11 -- the compaction is WARP-COOPERATIVE: ballot the survivors, lane 0
//     atomicAdds the popcount into the staging slot, the offset is
//     shfl-broadcast from lane 0, and each survivor writes at
//     `offset + warpScanExclusive(ballot, lane)`.  The order WITHIN a warp is
//     therefore deterministic; the base offset across warps is a race.

import pxgSlangCommon;
import pxgSlangParticleDesc;

#define PXG_WARP_BLOCK 256
#include "pxgSlangWarpInc.slang"

static const uint PSDA_SOLVE = 256u;
static const uint PSDA_WARP = 32u;
static const uint PSDA_EMPTY_CELL = 0xffffffffu;
static const uint PSDA_MAX_NEIGHBORS = 16u;
// PxParticleFlag::eFULL_DIFFUSE_ADVECTION is 1<<2 -- NOT 1<<3, which is
// eENABLE_SPECULATIVE_CCD (the constant psHfMidphaseBda and psSdfMidphaseBda pin).
static const uint PSDA_FLAG_FULL_ADVECTION = 4u;

struct PsDiffAdvectPush
{
    uint64_t particleSystems;        // @0
    uint64_t activeParticleSystems;  // @8
    float gx, gy, gz;                // @16 PxVec3 gravity
    float dt;                        // @28 -> 32
}
[[vk::push_constant]]
ConstantBuffer<PsDiffAdvectPush> gDa;

// gridCal.cuh:34 / :46, as transcribed in particleGridBda.slang.
int3 psdaGridPos(float4 p, float cellWidth)
{
    int3 g;
    g.x = int(floor(p.x / cellWidth));
    g.y = int(floor(p.y / cellWidth));
    g.z = int(floor(p.z / cellWidth));
    return g;
}
uint psdaGridHash(int3 g, uint sx, uint sy, uint sz)
{
    const uint gx = asuint(g.x) & (sx - 1u);
    const uint gy = asuint(g.y) & (sy - 1u);
    const uint gz = asuint(g.z) & (sz - 1u);
    return ((gz * sy) * sx) + (gy * sx) + gx;
}
// diffuseParticles.cu:56 -- WDiffuse(h, invR) = 1 - h * invR.
float psdaWDiffuse(float h, float invR) { return 1.0f - h * invR; }

// ---------------------------------------------------------------------------
// ps_diffuseParticleUpdatePBF (diffuseParticles.cu:164)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ps_diffuseParticleUpdatePBF(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gDa.particleSystems);
    const uint id = Ptr<uint>(gDa.activeParticleSystems)[gid.y];            // V1: .y

    const uint numDiffuse = Ptr<uint>(systems[id].mNumDiffuseParticles)[0]; // V2
    const uint pi = gtid.x + gid.x * PSDA_SOLVE;
    if (pi >= numDiffuse)
        return;

    Ptr<uint> cellStarts = Ptr<uint>(systems[id].mCellStart);               // V4: the FLUID grid
    Ptr<uint> cellEnds = Ptr<uint>(systems[id].mCellEnd);
    Ptr<float4> sortedPose = Ptr<float4>(systems[id].mSortedPositions_InvMass);
    Ptr<float4> sortedVel = Ptr<float4>(systems[id].mSortedVelocities);

    Ptr<float4> diffusePositions = Ptr<float4>(systems[id].mDiffuseSortedPos_LifeTime);
    // V3: the ORIGIN array, overloaded as the velocity-average output.
    Ptr<float4> newVel = Ptr<float4>(systems[id].mDiffuseSortedOriginPos_LifeTime);

    const float4 xi4 = diffusePositions[pi];
    const float3 pos = float3(xi4.x, xi4.y, xi4.z);

    const float cellWidth = systems[id].mCommonData_GridCellWidth;
    const float contactDistanceSq = systems[id].mCommonData_ParticleContactDistanceSq;
    const float invContactDistance = systems[id].mCommonData_ParticleContactDistanceInv;
    const int3 gridPos = psdaGridPos(xi4, cellWidth);
    const uint sx = systems[id].mCommonData_GridSizeX;
    const uint sy = systems[id].mCommonData_GridSizeY;
    const uint sz = systems[id].mCommonData_GridSizeZ;

    float weightSum = 0.0f;
    float3 velocitySum = float3(0.0f, 0.0f, 0.0f);
    uint numNeighbors = 0u;

    // V5: one cell, or the full 3x3x3.
    const uint end = ((systems[id].mData_Flags & PSDA_FLAG_FULL_ADVECTION) != 0u) ? 3u : 1u;
    // V6: {0, -1, +1} IN THAT ORDER.
    int offs[3];
    offs[0] = 0; offs[1] = -1; offs[2] = 1;

    bool full = false;                                                      // V7: the `goto`
    for (uint z = 0u; z < end && !full; ++z)
        for (uint y = 0u; y < end && !full; ++y)
            for (uint x = 0u; x < end && !full; ++x)
            {
                const int3 np = int3(gridPos.x + offs[x], gridPos.y + offs[y], gridPos.z + offs[z]);
                const uint gridHash = psdaGridHash(np, sx, sy, sz);
                const uint startIndex = cellStarts[gridHash];
                if (startIndex == PSDA_EMPTY_CELL)
                    continue;

                const uint endIndex = cellEnds[gridHash];
                for (uint q = startIndex; q < endIndex; ++q)
                {
                    const float4 xj4 = sortedPose[q];
                    const float3 xij = pos - float3(xj4.x, xj4.y, xj4.z);
                    const float dSq = dot(xij, xij);
                    if (dSq < contactDistanceSq)
                    {
                        const float4 vj4 = sortedVel[q];
                        const float w = psdaWDiffuse(sqrt(dSq), invContactDistance);  // V8
                        weightSum += w;
                        velocitySum += float3(vj4.x, vj4.y, vj4.z) * w;
                        ++numNeighbors;
                        if (numNeighbors == PSDA_MAX_NEIGHBORS) { full = true; break; }
                    }
                }
            }

    float3 velAvg = float3(0.0f, 0.0f, 0.0f);
    if (weightSum > 0.0f)
        // P9.36: diffuseParticles.cu:269 divides a PxVec3 by a PxReal, i.e.
        // PxVec3::operator/ (PxVec3.h:219) -- one reciprocal, three multiplies.
        velAvg = velocitySum * (1.0f / weightSum);

    newVel[pi] = float4(velAvg.x, velAvg.y, velAvg.z, float(numNeighbors));  // V9
}

// ---------------------------------------------------------------------------
// ps_diffuseParticleCompact (diffuseParticles.cu:275)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ps_diffuseParticleCompact(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gDa.particleSystems);
    const uint id = Ptr<uint>(gDa.activeParticleSystems)[gid.z];            // K1: .z

    const uint bufferIndex = gid.y;
    if (bufferIndex >= systems[id].mNumDiffuseBuffers)
        return;

    Ptr<PxgParticleDiffuseSimBufferGpu> bufs =
        Ptr<PxgParticleDiffuseSimBufferGpu>(systems[id].mDiffuseSimBuffers);

    Ptr<uint> numDiffuseParticles = Ptr<uint>(bufs[bufferIndex].mNumDiffuseParticles);
    const uint numDiffuse = numDiffuseParticles[0];                         // K2: PUBLISHED

    const uint tid = gtid.x;
    const uint pi = tid + gid.x * PSDA_SOLVE;
    const uint lane = tid & (PSDA_WARP - 1u);

    // The warp primitives below are reached by every lane of the block, so the
    // early return has to be a MASK rather than a `return` -- the CUDA can
    // return because __ballot_sync's mask is the ACTIVE set; the emulation's
    // ballot is built out of workgroup barriers (P9.12e), so a lane that left
    // would strand the rest.  `live` carries what the CUDA's return does.
    const bool live = (pi < numDiffuse);

    float3 newPosCorr = float3(0.0f, 0.0f, 0.0f);
    float3 newVelCorr = float3(0.0f, 0.0f, 0.0f);
    float lifeTime = 0.0f;

    if (live)
    {
        Ptr<float4> velAvgs = Ptr<float4>(systems[id].mDiffuseSortedOriginPos_LifeTime);  // V3
        Ptr<float4> diffusePositions = Ptr<float4>(systems[id].mDiffuseSortedPos_LifeTime);
        Ptr<float4> diffusePositionsOld = Ptr<float4>(systems[id].mDiffuseOriginPos_LifeTime);
        Ptr<uint> reverseLookup = Ptr<uint>(systems[id].mDiffuseUnsortedToSortedMapping);

        const uint index = pi + bufs[bufferIndex].mStartIndex;              // K3
        const uint sortedInd = reverseLookup[index];

        // K4: three arrays, three index spaces.
        const float4 xi4 = diffusePositions[sortedInd];
        const float4 vi4Old = diffusePositionsOld[index];
        const float4 xiva4 = velAvgs[sortedInd];

        const float3 pos = float3(xi4.x, xi4.y, xi4.z);
        const float3 oldPos = float3(vi4Old.x, vi4Old.y, vi4Old.z);
        const float3 velAvg = float3(xiva4.x, xiva4.y, xiva4.z);

        const float lifeDelta = gDa.dt;
        const float3 vel = (pos - oldPos) * (1.0f / gDa.dt);                // K5

        const float3 gravity = float3(gDa.gx, gDa.gy, gDa.gz);
        float3 newVel;
        if (xiva4.w < 4.0f)                                                 // K6: spray
            newVel = vel * (1.0f - bufs[bufferIndex].mParams_airDrag * gDa.dt);
        else if (xiva4.w < 8.0f)                                            // foam
            newVel = velAvg;
        else                                                                // K7: bubble
            newVel = vel - (1.0f + bufs[bufferIndex].mParams_buoyancy) * gravity * gDa.dt
                   + bufs[bufferIndex].mParams_bubbleDrag * (velAvg - vel);

        const float maxVel = systems[id].mData_MaxVelocity;
        const float magSq = dot(newVel, newVel);
        if (magSq > 0.0f)                                                   // K8
        {
            const float mag = sqrt(magSq);
            const float scale = (mag < maxVel) ? mag : maxVel;
            // PxVec3::getNormalized(): m2 > 0 ? v * PxRecipSqrt(m2) : 0.
            const float3 unit = newVel * (1.0f / sqrt(magSq));
            newVel = scale * unit;
        }

        newPosCorr = pos + (newVel - vel) * gDa.dt;                         // K9
        newVelCorr = newVel;
        lifeTime = max(xi4.w - lifeDelta, 0.0f);                            // K10
    }

    // K11: ballot, one atomic per warp by lane 0, broadcast, exclusive scan.
    const bool survives = live && (lifeTime > 0.0f);
    const uint res = pxgBallotSync(PXG_FULL_MASK, tid, survives);

    uint offset = 0u;
    if (lane == 0u)
        InterlockedAdd(numDiffuseParticles[1], pxgPopc(res), offset);
    offset = pxgShflSyncU(PXG_FULL_MASK, tid, offset, 0, PSDA_WARP);

    if (survives)
    {
        const uint newIndex = offset + pxgWarpRankInMask(res, lane);
        Ptr<float4>(bufs[bufferIndex].mDiffusePositions_LifeTime)[newIndex] =
            float4(newPosCorr.x, newPosCorr.y, newPosCorr.z, lifeTime);
        Ptr<float4>(bufs[bufferIndex].mDiffuseVelocities)[newIndex] =
            float4(newVelCorr.x, newVelCorr.y, newVelCorr.z, 0.0f);
    }
}
