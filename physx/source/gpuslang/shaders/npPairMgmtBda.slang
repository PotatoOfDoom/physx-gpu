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

// npPairMgmtBda.slang
//
// The whole contact-manager removal chain (pairManagement.cu:45-145) on the
// Vulkan engine ABI: `removeContactManagers_Stage1` .. `_Stage5` plus the
// wide-manifold variant `_Stage5_CvxTri`.
//
// THIS HAS TO BE ONE TRANCHE, not five.  `removeLostPairsGpuInternal`
// (PxgNarrowphaseCore.cpp:8521-8553) launches all five back to back with the
// SAME argument pack, and stages 1-4 exist only to build the run-sum buffer that
// stage 5 reads.  Stages 1-4 were unmapped in the manifest, so their launches
// failed silently (`fromFunc()` fails and `launchKernel` returns before the trace
// point — no trace line, just the engine's stderr) while stage 5 was mapped to a
// bound-buffer Phase-1 shader and showed up as REFUSED.  Porting stage 5 alone
// would have made the trace clean and the RESULT WRONG: it would binary-search a
// run-sum buffer nobody had filled in.
//
// THE ARGUMENT PACK is `(const PxgPairManagementData* pairData, bool copyManifold)`
// for all five -> ABI row {8,1}, a 9-byte blob.  The 1-byte bool is read back as
// the low byte of a uint, the same way `broadphaseEndPtHistogram` reads
// `isIncremental`: the shim writes 9 bytes and the three bytes above the bool are
// whatever was already in the push-constant range.
//
// EVERYTHING ELSE COMES THROUGH THE STRUCT POINTER.  PxgPairManagementData
// (cudaNpCommon.h:65) is nine pointers, two PxU32, and one more pointer that the
// PxU32 pair does NOT pack in with — `mPersistentContactManagers` re-aligns to 8,
// so it sits at 80 and the struct is 88 bytes, not 84.  testPairManagement pins
// every offset with a static_assert against the real header.
//
// WHAT THE CHAIN DOES.  Given N pairs of which K are being removed (the removal
// indices are sorted ascending by the caller):
//
//   Stage1  keep/drop buffer: buf[i] = (i < N-K) ? 0 : 1
//   Stage2  for each removal index r: buf[r] = (r < N-K) ? 1 : 0
//           — a removed pair in the KEPT region is a hole to fill (1); a removed
//             pair already in the tail is not (0), and neither is a kept pair in
//             the kept region.  What survives as 1 is exactly {holes in [0,N-K)}
//             union {survivors in [N-K,N)}, and those two sets have equal size.
//   Stage3  per-block exclusive scan of buf, block totals to the accumulator
//   Stage4  add the cross-block exclusive prefix -> buf is the device-wide
//           exclusive prefix sum of the marks
//   Stage5  nbToSwap = buf[N-K]; for swap i, dstIndex = the i-th marked slot and
//           srcIndex = the (i+nbToSwap)-th, both found by binary search over the
//           run-sum.  Copy the six contact-manager arrays (and the manifold) from
//           src down to dst.
//
// The binary search is `binarySearch` from reduction.cuh:729 — the LARGEST index
// with data[index] <= value, and 0 when data[0] > value.  Transcribed verbatim;
// the `left ? left-1 : 0` tail is not the same as `max(left,1)-1` for the
// underflow case and the port keeps the original shape.
//
// DIMENSIONS.  Grid 32 x block 512 — `PxgNarrowPhaseGridDims::REMOVE_CONTACT_MANAGERS`
// is 32 and `...BlockDims::...` is 512 (P8.36's lesson: the launch takes grid
// first, then block, and the shim uses the SPIR-V's declared workgroup size, so a
// transposition here is silent).  Both scan stages must agree on the same
// decomposition, and the cross-block accumulator the engine allocates is exactly
// `sizeof(PxU32) * 32` (PxgNarrowphaseCore.cpp:8446), so NUM_BLOCKS is not free.
//
// The CUDA blocks are 2D — (WARP_SIZE, blockDim/WARP_SIZE) — and the kernels
// index with `threadIdx.x + WARP_SIZE*threadIdx.y`, which is the linear thread id.
// `threadIdx.x & 15` in stage 5 is therefore `linear & 15` and `threadIdx.x` in
// the _CvxTri variant is `linear & 31`; the 1D port says so directly.

static const uint BLOCK = 512u;      // PxgNarrowPhaseBlockDims::REMOVE_CONTACT_MANAGERS
static const uint NUM_BLOCKS = 32u;  // PxgNarrowPhaseGridDims::REMOVE_CONTACT_MANAGERS
static const uint WARP = 32u;
static const uint WARPS_PER_BLOCK = BLOCK / WARP;   // 16

// PxgPairManagementData, x86-64.
static const uint PMD_INPUT      = 0u;    // PxgContactManagerInput*
static const uint PMD_OUTPUT     = 8u;    // PxsContactManagerOutput*
static const uint PMD_CMS        = 16u;   // PxsContactManager**
static const uint PMD_SIS        = 24u;   // Sc::ShapeInteraction**
static const uint PMD_REST       = 32u;   // PxReal*
static const uint PMD_TORSIONAL  = 40u;   // PxsTorsionalFrictionData*
static const uint PMD_TEMPACC    = 48u;   // PxU32*  (the run-sum buffer)
static const uint PMD_BLOCKACC   = 56u;   // PxU32*  [NUM_BLOCKS]
static const uint PMD_REMOVEIDX  = 64u;   // PxU32*
static const uint PMD_NBPAIRS    = 72u;   // PxU32
static const uint PMD_NBTOREMOVE = 76u;   // PxU32
static const uint PMD_PERSISTENT = 80u;   // void*

// Payload element sizes, in bytes.
static const uint SZ_INPUT     = 16u;   // PxgContactManagerInput
static const uint SZ_OUTPUT    = 48u;   // PxsContactManagerOutput
static const uint SZ_REST      = 4u;    // PxReal
static const uint SZ_TORSIONAL = 8u;    // PxsTorsionalFrictionData

// sizeof(manifold)/sizeof(float4) for the two stage-5 variants.
static const uint MANIFOLD_F4      = 16u;   // PxgPersistentContactManifold      (256)
static const uint MULTIMANIFOLD_F4 = 76u;   // PxgPersistentContactMultiManifold (1216)

struct PairMgmtParams
{
    uint64_t pairData;      // @0 const PxgPairManagementData*
    uint     copyManifold;  // @8 a 1-byte bool; only the low byte is written
}

[[vk::push_constant]]
ConstantBuffer<PairMgmtParams> gPm;

uint pmU32(uint offset) { return Ptr<uint>(gPm.pairData + offset)[0]; }
uint64_t pmPtr(uint offset) { return Ptr<uint64_t>(gPm.pairData + offset)[0]; }

// reduction.cuh:729 — largest index with data[index] <= value, else 0.
uint binarySearchRunSum(uint64_t data, uint numElements, uint value)
{
    uint left = 0u;
    uint right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (Ptr<uint>(data)[pos] <= value)
            left = pos + 1u;
        else
            right = pos;
    }
    return left != 0u ? left - 1u : 0u;
}

// ---------------------------------------------------------------------------
// Stage 1: initializeKeepDropBuffer
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(512, 1, 1)]
void removeContactManagers_Stage1(uint3 dtid : SV_DispatchThreadID)
{
    const uint totalCount = pmU32(PMD_NBPAIRS);
    const uint newArraySize = totalCount - pmU32(PMD_NBTOREMOVE);
    const uint64_t runSum = pmPtr(PMD_TEMPACC);
    const uint stride = BLOCK * WorkgroupCount().x;

    for (uint i = dtid.x; i < totalCount; i += stride)
        Ptr<uint>(runSum)[i] = (i < newArraySize) ? 0u : 1u;
}

// ---------------------------------------------------------------------------
// Stage 2: markKeepDropBuff
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(512, 1, 1)]
void removeContactManagers_Stage2(uint3 dtid : SV_DispatchThreadID)
{
    const uint totalCount = pmU32(PMD_NBPAIRS);
    const uint nbToRemove = pmU32(PMD_NBTOREMOVE);
    const uint newArraySize = totalCount - nbToRemove;
    const uint64_t runSum = pmPtr(PMD_TEMPACC);
    const uint64_t removeIdx = pmPtr(PMD_REMOVEIDX);
    const uint stride = BLOCK * WorkgroupCount().x;

    for (uint i = dtid.x; i < nbToRemove; i += stride)
    {
        const uint index = Ptr<uint>(removeIdx)[i];
        Ptr<uint>(runSum)[index] = (index < newArraySize) ? 1u : 0u;
    }
}

// ---------------------------------------------------------------------------
// the two-pass exclusive scan (reduction.cuh scanKernel1of2 / 2of2)
//
// llvmpipe's subgroup is 8 lanes, so the CUDA warp scans are done as 32-lane
// VIRTUAL warps over groupshared, with every barrier in uniform control flow.
// The partition is the CUDA one: block b owns the contiguous run of
// `numIterationPerBlock` chunks starting at chunk `b * numIterationPerBlock`, so
// stages 3 and 4 walk the same elements in the same order.
// ---------------------------------------------------------------------------
groupshared uint gsVals[BLOCK];
groupshared uint gsWarp[WARPS_PER_BLOCK];
groupshared uint gsBlockTotal;
groupshared uint gsBlockAccum[NUM_BLOCKS];

void warpScanInclusive(uint tid)
{
    for (uint off = 1u; off < WARP; off <<= 1)
    {
        const uint lane = tid & (WARP - 1u);
        const uint add = (lane >= off) ? gsVals[tid - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gsVals[tid] += add;
        GroupMemoryBarrierWithGroupSync();
    }
}

// Exclusive scan of the per-warp totals, leaving gsWarp[w] = sum of warps < w and
// folding the block total into gsBlockTotal.
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
    if (tid == WARPS_PER_BLOCK - 1u) gsBlockTotal += pInc;
}

// ---------------------------------------------------------------------------
// Stage 3: processKeepDropBuff — per-block exclusive scan, block totals out
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(512, 1, 1)]
void removeContactManagers_Stage3(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint n = pmU32(PMD_NBPAIRS);
    const uint64_t runSum = pmPtr(PMD_TEMPACC);
    const uint64_t blockTotals = pmPtr(PMD_BLOCKACC);
    const uint warpIndex = tid / WARP;

    const uint totalBlockRequired = (n + (BLOCK - 1u)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1u)) / NUM_BLOCKS;

    if (tid == 0u)
        gsBlockTotal = 0u;
    GroupMemoryBarrierWithGroupSync();

    for (uint it = 0u; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = tid + it * BLOCK + BLOCK * blockIdx * numIterationPerBlock;

        uint val = 0u;
        if (workIndex < n)
            val = Ptr<uint>(runSum)[workIndex];

        gsVals[tid] = val;
        GroupMemoryBarrierWithGroupSync();
        warpScanInclusive(tid);
        const uint accum = gsVals[tid] - val;   // exclusive within the warp

        if ((tid & (WARP - 1u)) == WARP - 1u)
            gsWarp[warpIndex] = accum + val;

        // Read BEFORE the barrier: this is the carry from the previous
        // iterations, which is what makes the whole thing exclusive.
        const uint prevBlock = gsBlockTotal;
        GroupMemoryBarrierWithGroupSync();

        scanWarpTotals(tid);

        GroupMemoryBarrierWithGroupSync();

        if (workIndex < n)
            Ptr<uint>(runSum)[workIndex] = accum + prevBlock + gsWarp[warpIndex];
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u)
        Ptr<uint>(blockTotals)[blockIdx] = gsBlockTotal;
}

// ---------------------------------------------------------------------------
// Stage 4: accumulateKeepDrop — add the cross-block exclusive prefix
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(512, 1, 1)]
void removeContactManagers_Stage4(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint n = pmU32(PMD_NBPAIRS);
    const uint64_t runSum = pmPtr(PMD_TEMPACC);
    const uint64_t blockTotals = pmPtr(PMD_BLOCKACC);

    if (tid < NUM_BLOCKS)
        gsBlockAccum[tid] = Ptr<uint>(blockTotals)[tid];
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0u)
    {
        uint acc = 0u;
        for (uint i = 0u; i < NUM_BLOCKS; ++i)
        {
            const uint t = gsBlockAccum[i];
            gsBlockAccum[i] = acc;
            acc += t;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const uint totalBlockRequired = (n + (BLOCK - 1u)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1u)) / NUM_BLOCKS;
    const uint blockAccum = gsBlockAccum[blockIdx];

    for (uint it = 0u; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = tid + it * BLOCK + BLOCK * blockIdx * numIterationPerBlock;
        if (workIndex < n)
            Ptr<uint>(runSum)[workIndex] = Ptr<uint>(runSum)[workIndex] + blockAccum;
    }
}

// ---------------------------------------------------------------------------
// the stage-5 payload move
//
// The six contact-manager arrays are copied as SCALARS rather than as their
// element types.  PxsContactManagerOutput is 48 bytes, so odd indices land 0 mod
// 16 only by luck of the base pointer, and the P8.26 lesson (PxBounds3 / 24 B)
// applies: anything whose stride is not a multiple of 16 gets read and written
// component-wise.  The manifold, whose stride IS a multiple of 16, moves as uint4
// — uint and not float4, so a manifold slot holding a bit pattern that happens to
// be a signalling NaN comes back byte-identical instead of canonicalised.
// ---------------------------------------------------------------------------
void copyBytes(uint64_t base, uint stride, uint dstIndex, uint srcIndex, uint nbWords)
{
    const uint64_t d = base + uint64_t(dstIndex) * uint64_t(stride);
    const uint64_t s = base + uint64_t(srcIndex) * uint64_t(stride);
    for (uint w = 0u; w < nbWords; ++w)
        Ptr<uint>(d)[w] = Ptr<uint>(s)[w];
}

void swapPayload(uint dstIndex, uint srcIndex)
{
    copyBytes(pmPtr(PMD_INPUT), SZ_INPUT, dstIndex, srcIndex, SZ_INPUT / 4u);
    copyBytes(pmPtr(PMD_OUTPUT), SZ_OUTPUT, dstIndex, srcIndex, SZ_OUTPUT / 4u);
    copyBytes(pmPtr(PMD_REST), SZ_REST, dstIndex, srcIndex, SZ_REST / 4u);
    copyBytes(pmPtr(PMD_TORSIONAL), SZ_TORSIONAL, dstIndex, srcIndex, SZ_TORSIONAL / 4u);

    // The two pointer arrays are 8-byte elements, so index*8 keeps 8-alignment
    // and they move as whole pointers.
    const uint64_t cms = pmPtr(PMD_CMS);
    Ptr<uint64_t>(cms)[dstIndex] = Ptr<uint64_t>(cms)[srcIndex];
    const uint64_t sis = pmPtr(PMD_SIS);
    Ptr<uint64_t>(sis)[dstIndex] = Ptr<uint64_t>(sis)[srcIndex];
}

void copyManifold(uint64_t manifolds, uint nbFloat4, uint dstIndex, uint srcIndex,
                  uint lane, uint laneCount)
{
    const uint64_t d = manifolds + uint64_t(dstIndex) * uint64_t(nbFloat4) * 16UL;
    const uint64_t s = manifolds + uint64_t(srcIndex) * uint64_t(nbFloat4) * 16UL;
    for (uint idx = lane; idx < nbFloat4; idx += laneCount)
        Ptr<uint4>(d)[idx] = Ptr<uint4>(s)[idx];
}

// ---------------------------------------------------------------------------
// Stage 5: 16 threads per swap, single manifold
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(512, 1, 1)]
void removeContactManagers_Stage5(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint totalSize = pmU32(PMD_NBPAIRS);
    const uint64_t runSum = pmPtr(PMD_TEMPACC);
    const uint nbToSwap = Ptr<uint>(runSum)[totalSize - pmU32(PMD_NBTOREMOVE)];
    const bool doManifold = (gPm.copyManifold & 0xffu) != 0u;

    const uint linear = gtid.x;
    const uint lane = linear & 15u;
    const uint stride = (BLOCK * WorkgroupCount().x) / 16u;

    for (uint i = (linear + gid.x * BLOCK) / 16u; i < nbToSwap; i += stride)
    {
        const uint dstIndex = binarySearchRunSum(runSum, totalSize, i);
        const uint srcIndex = binarySearchRunSum(runSum, totalSize, i + nbToSwap);

        if (lane == 0u)
            swapPayload(dstIndex, srcIndex);

        if (doManifold)
            copyManifold(pmPtr(PMD_PERSISTENT), MANIFOLD_F4, dstIndex, srcIndex, lane, 16u);
    }
}

// ---------------------------------------------------------------------------
// Stage 5 _CvxTri: 32 threads per swap, multi-manifold
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(512, 1, 1)]
void removeContactManagers_Stage5_CvxTri(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint totalSize = pmU32(PMD_NBPAIRS);
    const uint64_t runSum = pmPtr(PMD_TEMPACC);
    const uint nbToSwap = Ptr<uint>(runSum)[totalSize - pmU32(PMD_NBTOREMOVE)];
    const bool doManifold = (gPm.copyManifold & 0xffu) != 0u;

    const uint linear = gtid.x;
    const uint lane = linear & (WARP - 1u);
    const uint stride = (BLOCK * WorkgroupCount().x) / WARP;

    for (uint i = (linear + gid.x * BLOCK) / WARP; i < nbToSwap; i += stride)
    {
        const uint dstIndex = binarySearchRunSum(runSum, totalSize, i);
        const uint srcIndex = binarySearchRunSum(runSum, totalSize, i + nbToSwap);

        if (lane == 0u)
            swapPayload(dstIndex, srcIndex);

        if (doManifold)
            copyManifold(pmPtr(PMD_PERSISTENT), MULTIMANIFOLD_F4, dstIndex, srcIndex, lane, WARP);
    }
}
