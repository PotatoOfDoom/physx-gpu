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

// psBoundsBda.slang
//
// P6.4a-1 — the four PBD particle bound / cell-range bookkeeping kernels on the
// Vulkan engine ABI (BDA), ported line-for-line from CUDA:
//
//   - ps_updateBoundFirstPassLaunch  (gpusimulationcontroller/src/CUDA/
//     particlesystem.cu:450) : per particle system, each block reduces its
//     slab of predicted (or speculative-CCD) particle + diffuse-particle
//     positions to a world AABB -> one PxBounds3 per block.
//   - ps_updateBoundSecondPassLaunch (particlesystem.cu:562) : a single block
//     reduces the per-block bounds to the global AABB, fattens it by the
//     element contact distance into the broadphase bound array, and writes the
//     AABB centre back into mData.mBoundCenter.  Launch site
//     PxgParticleSystemCore.cpp:856 (PS_UPDATE_BOUND_FRIST) / :907
//     (PS_UPDATE_BOUND_SECOND).
//   - ps_primitivesBoundFirstPassLaunch  (gpunarrowphase/src/CUDA/
//     cudaParticleSystem.cu:106) : per broadphase-found particle-shape pair,
//     combine the rigid + particle AABBs, fattenFast by the summed contact
//     distance, count the wrapped grid cells the overlap covers, and emit a
//     segmented EXCLUSIVE scan of those counts (per-warp scan, per-block
//     warp-total scan, running block accumulator), per-block totals ->
//     blockOffsets.  BLOCK = 512 (BOUNDCELLUPDATE), grid = 32.
//   - ps_primitivesBoundSecondPassLaunch (cudaParticleSystem.cu:217) :
//     exclusive scan over the 32 block totals, folded into every offset, grand
//     total -> totalNumPairs.  Launch site PxgNarrowphaseCore.cpp:3151
//     (PS_PRIMITIVES_BOUND_FIRST) / :3190 (PS_PRIMITIVES_BOUND_SECOND).
//
// Reused cores / mirrors:
//   - PxgParticleSystemGpu descriptor mirror: pxgSlangParticleDesc (P6.0);
//     every field these kernels touch (mCommonData grid fields, the diffuse
//     slab pointers, mData_Flags @312, mData_BoundCenter @248) is already
//     present, so no mirror extension was needed.
//   - the segmented-scan / cell-count core is the tranche-88 psBounds.slang
//     (harness) logic transcribed here; the primitivesBound wrappers add the
//     LoadShapePair descriptor reads (dataReadWriteHelper.cuh) in front of it.
//   - PxgContactManagerInput (16 B) mirrors boxBoxNphase.slang:89; PxgShape
//     (48 B) mirrors pxgSlangCommon.slang:358.  Transcribed locally (not
//     imported) to keep the rigid-common module out of the particle-descriptor
//     TU; the byte layout is pinned by static_asserts against the REAL engine
//     structs in testPsBounds.cpp.
//
// Deviations from CUDA (recipe policy):
//   - the two-level 32-lane warpReduction + inter-warp shared reduce of the
//     updateBound passes is lowered to a PORTABLE full-block groupshared
//     min/max tree reduction (blockReduceMinF/blockReduceMaxF).  min/max of a
//     finite float set is associative and rounding-free, so the reduced AABB is
//     BIT-IDENTICAL regardless of association order or subgroup width — this is
//     the llvmpipe guard (its subgroup width != 32, so the CUDA FULL_MASK warp
//     shuffle cannot be assumed).
//   - the primitivesBound warpScan / warp-total scan is emulated with the same
//     groupshared barrier scan psBounds.slang already uses (WARP=32 is a LOGICAL
//     tile, not a hardware subgroup), so it is output-identical without a
//     32-lane assumption.
//   - the CUDA blockCopy of the whole descriptor into shared memory is replaced
//     by direct field loads through Ptr<> (semantically identical read-fanout).
//   - descriptor scalar writeback (mData.mBoundCenter, the per-block PxBounds3,
//     the broadphase bound array) is done through a raw-byte-offset Ptr<float>
//     (stride 656 for PxgParticleSystem, 24 for PxBounds3), matching the BDA
//     nested-pointer walk.

import pxgSlangParticleDesc;

// PxgParticleSystemKernelBlockDim::UPDATEBOUND (PxgParticleSystemCoreKernelIndices.h:39)
static const uint UPDATEBOUND = 1024u;
// PxgParticleSystemKernelBlockDim::BOUNDCELLUPDATE (:41) / GridDim::BOUNDCELLUPDATE (:58)
static const uint BC_BLOCK = 512u;
static const uint BC_GRID = 32u;
static const uint WARP = 32u;
static const uint WARPS_PER_BLOCK = BC_BLOCK / WARP; // 16

static const float PX_MAX_F32 = 3.4028235e38f;

// PxGeometryType::ePARTICLESYSTEM (PxGeometry.h:56, 0-indexed)
static const uint GEOM_PARTICLESYSTEM = 6u;
// PxParticleFlag::eENABLE_SPECULATIVE_CCD (PxPBDParticleSystem.h:185)
static const uint PS_FLAG_ENABLE_CCD = 8u;

// particlesystem.cu forward-projection step scales (gridCal.cuh:113-115).
static const float FWD_PGS = 1.0f;
static const float FWD_TGS = 0.5f;

// PxsCachedTransform stride bytes (unused field for the first pass, present only
// for ABI parity / documentation).
// PxgParticleSystem array stride (sizeof) — pins the descriptor walk.
static const uint64_t PS_STRIDE = 656ull;
// mData.mBoundCenter byte offset within PxgParticleSystem.
static const uint64_t BOUNDCENTER_OFF = 248ull;
// sizeof(PxBounds3) in floats.
static const uint BOUNDS_FLOATS = 6u;

// ---------------------------------------------------------------------------
// Local element-struct mirrors (byte-exact vs the engine C++ layouts).
// ---------------------------------------------------------------------------
// PxgContactManagerInput (PxgContactManager.h, PX_ALIGN(16), 16 bytes)
struct PxgContactManagerInput
{
    uint shapeRef0;         // @0
    uint shapeRef1;         // @4
    uint transformCacheRef0;// @8
    uint transformCacheRef1;// @12
}

// PxgShape (PxgConvexConvexShape.h, 48 bytes) — only type / particleOrSoftbodyId
// are read by these kernels; the leading fields keep the 48-byte stride.
struct PxgShape
{
    float scaleX, scaleY, scaleZ;                 // @0
    float scaleRotX, scaleRotY, scaleRotZ, scaleRotW; // @12
    uint materialIndex;                           // @28
    uint2 hullOrMeshOffset;                       // @32 (size_t hullOrMeshPtr)
    uint type;                                    // @40
    uint particleOrSoftbodyId;                    // @44
}                                                 // = 48

// ---- gridCal.cuh helpers (see gridCal.slang / psBounds.slang) -------------
int3 calcGridPos(float3 p, float cellWidth)
{
    return int3(int(floor(p.x / cellWidth)), int(floor(p.y / cellWidth)), int(floor(p.z / cellWidth)));
}

uint3 calcWrappedGridRangeSize(int3 mn, int3 mx, uint3 gridSize)
{
    return uint3(uint(clamp(mx.x - mn.x + 1, 0, int(gridSize.x))),
                 uint(clamp(mx.y - mn.y + 1, 0, int(gridSize.y))),
                 uint(clamp(mx.z - mn.z + 1, 0, int(gridSize.z))));
}

// combine(): intersection box (component max of minima, min of maxima).
void combineBounds(float3 mn0, float3 mx0, float3 mn1, float3 mx1, out float3 mn, out float3 mx)
{
    mn = max(mn0, mn1);
    mx = min(mx0, mx1);
}

// getParticleSpeculativeContactVolume (gridCal.cuh:122).
float getSpecVolume(out float3 cVolumePos, float3 currentPos, float3 predictedPos,
                    float contactDist, bool isDiffuse, bool isTGS)
{
    float scaleInv;
    if (isDiffuse)
        scaleInv = 1.0f / FWD_PGS;         // FWD_DIFFUSE == FWD_PGS == 1.0
    else
        scaleInv = isTGS ? (1.0f / FWD_TGS) : (1.0f / FWD_PGS);
    const float3 cVolumeOffset = (predictedPos - currentPos) * scaleInv * 0.5f;
    cVolumePos = currentPos + cVolumeOffset;
    return length(cVolumeOffset) + contactDist;
}

// ---------------------------------------------------------------------------
// Portable full-block min/max reduction (llvmpipe guard: no 32-lane subgroup).
// ---------------------------------------------------------------------------
groupshared float sRed[UPDATEBOUND];

float blockReduceMinF(uint tid, float v)
{
    sRed[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = UPDATEBOUND / 2u; s > 0u; s >>= 1)
    {
        if (tid < s)
            sRed[tid] = min(sRed[tid], sRed[tid + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    const float r = sRed[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

float blockReduceMaxF(uint tid, float v)
{
    sRed[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = UPDATEBOUND / 2u; s > 0u; s >>= 1)
    {
        if (tid < s)
            sRed[tid] = max(sRed[tid], sRed[tid + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    const float r = sRed[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// ===========================================================================
// ps_updateBoundFirstPassLaunch — per-block world AABB of the particle system.
// argSizes (PxgParticleSystemCore.cpp:856): { particleSystems*, PxU32 id,
//   bool isTGS, PxBounds3* bounds } -> {8,4,1,8} = 24-byte blob (bool @12).
// ===========================================================================
struct PsUpdateBound1Params
{
    uint64_t particleSystems; // @0
    uint id;                  // @8
    uint isTGSWord;           // @12 (byte0 = bool isTGS)
    uint64_t bounds;          // @16 -> 24 (PxBounds3* tempBuf, one per block)
}
[[vk::push_constant]]
ConstantBuffer<PsUpdateBound1Params> gUB1;

[shader("compute")]
[numthreads(1024, 1, 1)]
void ps_updateBoundFirstPassLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gUB1.particleSystems);
    const uint id = gUB1.id;
    const bool isTGS = (gUB1.isTGSWord & 0xffu) != 0u;

    const uint maxDiffuse = systems[id].mCommonData_MaxDiffuseParticles;
    uint numDiffuseParticles = 0u;
    if (maxDiffuse > 0u)
        numDiffuseParticles = uint((Ptr<int>(systems[id].mNumDiffuseParticles))[0]);
    const uint numRegular = systems[id].mCommonData_NumParticles;
    const uint numParticles = max(numRegular, numDiffuseParticles);
    const uint globalThreadIndex = tid + gid.x * UPDATEBOUND;

    Ptr<float4> currentPositions = Ptr<float4>(systems[id].mOriginPos_InvMass);
    Ptr<float4> predictedPositions = Ptr<float4>(systems[id].mUnsortedPositions_InvMass);
    Ptr<float4> currentDiffusePositions = Ptr<float4>(systems[id].mDiffuseOriginPos_LifeTime);
    Ptr<float4> predictedDiffusePositions = Ptr<float4>(systems[id].mDiffusePosition_LifeTime);

    float3 minPos = float3(PX_MAX_F32, PX_MAX_F32, PX_MAX_F32);
    float3 maxPos = float3(-PX_MAX_F32, -PX_MAX_F32, -PX_MAX_F32);

    if (globalThreadIndex < numParticles)
    {
        const bool enableCCD = (systems[id].mData_Flags & PS_FLAG_ENABLE_CCD) > 0u;
        if (globalThreadIndex < numRegular)
        {
            const float3 predictedPos = predictedPositions[globalThreadIndex].xyz;
            if (enableCCD)
            {
                const float3 currentPos = currentPositions[globalThreadIndex].xyz;
                float3 cVolumePos;
                const float cVolumeRadius = getSpecVolume(cVolumePos, currentPos, predictedPos, 0.0f, false, isTGS);
                minPos = cVolumePos - float3(cVolumeRadius, cVolumeRadius, cVolumeRadius);
                maxPos = cVolumePos + float3(cVolumeRadius, cVolumeRadius, cVolumeRadius);
            }
            else
            {
                minPos = predictedPos;
                maxPos = predictedPos;
            }
        }

        if (globalThreadIndex < numDiffuseParticles)
        {
            const float3 predictedPos = predictedDiffusePositions[globalThreadIndex].xyz;
            if (enableCCD)
            {
                const float3 currentPos = currentDiffusePositions[globalThreadIndex].xyz;
                float3 cVolumePos;
                const float cVolumeRadius = getSpecVolume(cVolumePos, currentPos, predictedPos, 0.0f, true, isTGS);
                minPos = min(minPos, cVolumePos - float3(cVolumeRadius, cVolumeRadius, cVolumeRadius));
                maxPos = max(maxPos, cVolumePos + float3(cVolumeRadius, cVolumeRadius, cVolumeRadius));
            }
            else
            {
                minPos = min(minPos, predictedPos);
                maxPos = max(maxPos, predictedPos);
            }
        }
    }

    const float mnx = blockReduceMinF(tid, minPos.x);
    const float mny = blockReduceMinF(tid, minPos.y);
    const float mnz = blockReduceMinF(tid, minPos.z);
    const float mxx = blockReduceMaxF(tid, maxPos.x);
    const float mxy = blockReduceMaxF(tid, maxPos.y);
    const float mxz = blockReduceMaxF(tid, maxPos.z);

    if (tid == 0u)
    {
        Ptr<float> bounds = Ptr<float>(gUB1.bounds);
        const uint b = gid.x * BOUNDS_FLOATS;
        bounds[b + 0u] = mnx; bounds[b + 1u] = mny; bounds[b + 2u] = mnz;
        bounds[b + 3u] = mxx; bounds[b + 4u] = mxy; bounds[b + 5u] = mxz;
    }
}

// ===========================================================================
// ps_updateBoundSecondPassLaunch — global AABB + fatten + centre writeback.
// argSizes (PxgParticleSystemCore.cpp:907): { particleSystems*, PxU32 id,
//   PxBounds3* bounds, PxU32 numBounds, PxBounds3* boundArray,
//   PxReal* contactDists, PxU32 elemIndex } -> {8,4,8,4,8,8,4} = 56-byte blob.
// ===========================================================================
struct PsUpdateBound2Params
{
    uint64_t particleSystems; // @0
    uint id;                  // @8
    uint _pad12;              // @12
    uint64_t bounds;          // @16 (input per-block bounds)
    uint numBounds;           // @24
    uint _pad28;              // @28
    uint64_t boundArray;      // @32 (output broadphase bound array)
    uint64_t contactDists;    // @40
    uint elemIndex;           // @48
    uint _pad52;              // @52 -> 56
}
[[vk::push_constant]]
ConstantBuffer<PsUpdateBound2Params> gUB2;

[shader("compute")]
[numthreads(1024, 1, 1)]
void ps_updateBoundSecondPassLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint id = gUB2.id;
    const uint numBounds = gUB2.numBounds;

    Ptr<float> boundsIn = Ptr<float>(gUB2.bounds);

    float3 accMin = float3(PX_MAX_F32, PX_MAX_F32, PX_MAX_F32);
    float3 accMax = float3(-PX_MAX_F32, -PX_MAX_F32, -PX_MAX_F32);

    const uint numIterations = (numBounds + UPDATEBOUND - 1u) / UPDATEBOUND;
    for (uint i = 0u; i < numIterations; ++i)
    {
        const uint elemIndex = tid + i * UPDATEBOUND;

        // PxBounds3::setEmpty()
        float3 bmn = float3(PX_MAX_F32, PX_MAX_F32, PX_MAX_F32);
        float3 bmx = float3(-PX_MAX_F32, -PX_MAX_F32, -PX_MAX_F32);
        if (elemIndex < numBounds)
        {
            const uint b = elemIndex * BOUNDS_FLOATS;
            bmn = float3(boundsIn[b + 0u], boundsIn[b + 1u], boundsIn[b + 2u]);
            bmx = float3(boundsIn[b + 3u], boundsIn[b + 4u], boundsIn[b + 5u]);
        }

        accMin.x = min(accMin.x, blockReduceMinF(tid, bmn.x));
        accMin.y = min(accMin.y, blockReduceMinF(tid, bmn.y));
        accMin.z = min(accMin.z, blockReduceMinF(tid, bmn.z));
        accMax.x = max(accMax.x, blockReduceMaxF(tid, bmx.x));
        accMax.y = max(accMax.y, blockReduceMaxF(tid, bmx.y));
        accMax.z = max(accMax.z, blockReduceMaxF(tid, bmx.z));
    }

    if (tid == 0u)
    {
        const uint elemIndex = gUB2.elemIndex;
        const float contactDist = (Ptr<float>(gUB2.contactDists))[elemIndex];

        // resBound = min - cd (axes 0..2), max + cd (axes 3..5); centre cancels cd.
        const float3 resMin = accMin - float3(contactDist, contactDist, contactDist);
        const float3 resMax = accMax + float3(contactDist, contactDist, contactDist);

        Ptr<float> boundArray = Ptr<float>(gUB2.boundArray);
        const uint o = elemIndex * BOUNDS_FLOATS;
        boundArray[o + 0u] = resMin.x; boundArray[o + 1u] = resMin.y; boundArray[o + 2u] = resMin.z;
        boundArray[o + 3u] = resMax.x; boundArray[o + 4u] = resMax.y; boundArray[o + 5u] = resMax.z;

        Ptr<float> boundCenter = Ptr<float>(gUB2.particleSystems + uint64_t(id) * PS_STRIDE + BOUNDCENTER_OFF);
        boundCenter[0u] = (resMax.x + resMin.x) * 0.5f;
        boundCenter[1u] = (resMax.y + resMin.y) * 0.5f;
        boundCenter[2u] = (resMax.z + resMin.z) * 0.5f;
    }
}

// ===========================================================================
// ps_primitivesBoundFirstPassLaunch — per-pair cell-range count + scan.
// argSizes (PxgNarrowphaseCore.cpp:3151): { PxU32 numTests,
//   PxgContactManagerInput* cmInputs, PxgShape* shapes,
//   PxsCachedTransform* transformCache, PxBounds3* bounds,
//   PxReal* contactDistance, PxgParticleSystem* particleSystems,
//   PxU32* blockOffsets (out), PxU32* offsets (out) }
//   -> {4,8,8,8,8,8,8,8,8} = 72-byte blob.  transformCache is unused by this
//   pass (present for ABI parity — the CUDA reads only bounds/contactDistance).
// ===========================================================================
struct PsPrimBound1Params
{
    uint numTests;            // @0
    uint _pad4;               // @4
    uint64_t cmInputs;        // @8
    uint64_t shapes;          // @16
    uint64_t transformCache;  // @24 (unused)
    uint64_t bounds;          // @32
    uint64_t contactDistance; // @40
    uint64_t particleSystems; // @48
    uint64_t blockOffsets;    // @56 (output)
    uint64_t offsets;         // @64 -> 72 (output)
}
[[vk::push_constant]]
ConstantBuffer<PsPrimBound1Params> gPB1;

// scan plumbing (same shapes as psBounds.slang / broadphaseRegionHistogram).
groupshared uint gsLane[BC_BLOCK];
groupshared uint gsWarp[WARPS_PER_BLOCK];
groupshared uint gsBlockAccum;

void warpScanLane(uint tid)
{
    for (uint off = 1u; off < WARP; off <<= 1)
    {
        const uint lane = tid & (WARP - 1u);
        const uint add = (lane >= off) ? gsLane[tid - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gsLane[tid] += add;
        GroupMemoryBarrierWithGroupSync();
    }
}

void scanWarpTotals(uint tid)
{
    const bool active = tid < WARPS_PER_BLOCK;
    const uint p0 = active ? gsWarp[tid] : 0u;
    uint pInc = p0;
    for (uint off = 1u; off < WARPS_PER_BLOCK; off <<= 1)
    {
        const bool take = active && tid >= off;
        const uint pa = take ? gsWarp[tid - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        pInc += pa;
        if (active) gsWarp[tid] = pInc;
        GroupMemoryBarrierWithGroupSync();
    }
    GroupMemoryBarrierWithGroupSync();
    if (active) gsWarp[tid] = pInc - p0;
    if (tid == WARPS_PER_BLOCK - 1u) gsBlockAccum += pInc;
}

[shader("compute")]
[numthreads(512, 1, 1)]
void ps_primitivesBoundFirstPassLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint warpIndex = tid / WARP;
    const uint lane = tid & (WARP - 1u);

    const uint numTests = gPB1.numTests;
    const uint nbBlocksRequired = (numTests + BC_BLOCK - 1u) / BC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + BC_GRID - 1u) / BC_GRID;

    Ptr<PxgContactManagerInput> cmInputs = Ptr<PxgContactManagerInput>(gPB1.cmInputs);
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gPB1.shapes);
    Ptr<float> boundsArr = Ptr<float>(gPB1.bounds);
    Ptr<float> contactDistance = Ptr<float>(gPB1.contactDistance);
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gPB1.particleSystems);
    Ptr<uint> blockOffsets = Ptr<uint>(gPB1.blockOffsets);
    Ptr<uint> offsets = Ptr<uint>(gPB1.offsets);

    if (tid == 0u)
        gsBlockAccum = 0u;
    GroupMemoryBarrierWithGroupSync();

    for (uint it = 0u; it < nbIterationsPerBlock; ++it)
    {
        const uint workIndex = it * BC_BLOCK + tid + nbIterationsPerBlock * blockIdx * BC_BLOCK;

        uint cellNumCount = 0u;
        if (workIndex < numTests)
        {
            // LoadShapePair<ePARTICLESYSTEM> (dataReadWriteHelper.cuh:292): pick
            // the particle-system shape into slot 0, keep both cache refs.
            const PxgContactManagerInput cin = cmInputs[workIndex];
            uint shapeRefP = cin.shapeRef0;
            uint particleCacheRef = cin.transformCacheRef0;
            uint rigidCacheRef = cin.transformCacheRef1;
            if (shapes[cin.shapeRef0].type != GEOM_PARTICLESYSTEM)
            {
                shapeRefP = cin.shapeRef1;
                particleCacheRef = cin.transformCacheRef1;
                rigidCacheRef = cin.transformCacheRef0;
            }

            const uint particleInd = shapes[shapeRefP].particleOrSoftbodyId;

            const float cellWidth = systems[particleInd].mCommonData_GridCellWidth;
            const uint3 wrappedGridSize = uint3(systems[particleInd].mCommonData_GridSizeX,
                                                systems[particleInd].mCommonData_GridSizeY,
                                                systems[particleInd].mCommonData_GridSizeZ);

            const float cDistance = contactDistance[particleCacheRef] + contactDistance[rigidCacheRef];

            const uint r = rigidCacheRef * BOUNDS_FLOATS;
            const uint p = particleCacheRef * BOUNDS_FLOATS;
            const float3 b0min = float3(boundsArr[r + 0u], boundsArr[r + 1u], boundsArr[r + 2u]);
            const float3 b0max = float3(boundsArr[r + 3u], boundsArr[r + 4u], boundsArr[r + 5u]);
            const float3 b1min = float3(boundsArr[p + 0u], boundsArr[p + 1u], boundsArr[p + 2u]);
            const float3 b1max = float3(boundsArr[p + 3u], boundsArr[p + 4u], boundsArr[p + 5u]);

            float3 mn, mx;
            combineBounds(b0min, b0max, b1min, b1max, mn, mx);
            mn -= float3(cDistance, cDistance, cDistance); // PxBounds3::fattenFast
            mx += float3(cDistance, cDistance, cDistance);

            const int3 gridPosMin = calcGridPos(mn, cellWidth);
            const int3 gridPosMax = calcGridPos(mx, cellWidth);
            const uint3 rangeSize = calcWrappedGridRangeSize(gridPosMin, gridPosMax, wrappedGridSize);
            cellNumCount = rangeSize.x * rangeSize.y * rangeSize.z;
        }

        gsLane[tid] = cellNumCount;
        GroupMemoryBarrierWithGroupSync();
        warpScanLane(tid);
        const uint offset = gsLane[tid] - cellNumCount; // exclusive prefix within warp
        if (lane == WARP - 1u)
            gsWarp[warpIndex] = gsLane[tid];

        const uint prevBlockAccumulator = gsBlockAccum;
        GroupMemoryBarrierWithGroupSync();

        scanWarpTotals(tid);

        GroupMemoryBarrierWithGroupSync();

        if (workIndex < numTests)
            offsets[workIndex] = offset + prevBlockAccumulator + gsWarp[warpIndex];

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u)
        blockOffsets[blockIdx] = gsBlockAccum;
}

// ===========================================================================
// ps_primitivesBoundSecondPassLaunch — block-total scan + fold + grand total.
// argSizes (PxgNarrowphaseCore.cpp:3190): { PxU32 numTests, PxU32* blockOffsets,
//   PxU32* offsets, PxU32* totalNumPairs } -> {4,8,8,8} = 32-byte blob.
// ===========================================================================
struct PsPrimBound2Params
{
    uint numTests;          // @0
    uint _pad4;             // @4
    uint64_t blockOffsets;  // @8 (input)
    uint64_t offsets;       // @16 (output)
    uint64_t totalNumPairs; // @24 -> 32 (output)
}
[[vk::push_constant]]
ConstantBuffer<PsPrimBound2Params> gPB2;

groupshared uint gsBlockPrefix[BC_GRID];
groupshared uint gsTotalPairs;

[shader("compute")]
[numthreads(512, 1, 1)]
void ps_primitivesBoundSecondPassLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint numTests = gPB2.numTests;

    Ptr<uint> blockOffsets = Ptr<uint>(gPB2.blockOffsets);
    Ptr<uint> offsets = Ptr<uint>(gPB2.offsets);
    Ptr<uint> totalNumPairs = Ptr<uint>(gPB2.totalNumPairs);

    // exclusive scan over the 32 per-block totals (CUDA does this with one warp;
    // a serial lane-0 scan is output-identical — no 32-lane assumption).
    if (tid == 0u)
    {
        uint acc = 0u;
        for (uint b = 0u; b < BC_GRID; ++b)
        {
            const uint v = blockOffsets[b];
            gsBlockPrefix[b] = acc;
            acc += v;
        }
        gsTotalPairs = acc;
    }
    GroupMemoryBarrierWithGroupSync();

    const uint totalBlockRequired = (numTests + BC_BLOCK - 1u) / BC_BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + BC_GRID - 1u) / BC_GRID;
    const uint blockAccum = gsBlockPrefix[blockIdx];

    for (uint i = 0u; i < numIterationPerBlock; ++i)
    {
        const uint workIndex = i * BC_BLOCK + tid + numIterationPerBlock * blockIdx * BC_BLOCK;
        if (workIndex < numTests)
            offsets[workIndex] = offsets[workIndex] + blockAccum;
    }

    if (blockIdx == 0u && tid == 0u)
        totalNumPairs[0u] = gsTotalPairs;
}
