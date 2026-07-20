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

// broadphaseSetup.slang
//
// P5 Step 4 (BDA, engine launch path): the broad-phase foundation kernels
// (gpubroadphase/src/CUDA/broadphase.cu), all driven exactly as the engine
// launches them.  translateAABBsLaunch takes explicit buffer pointers + PxU8
// shift bytes; the other four take the PxgBroadPhaseDesc aggregate BY DEVICE
// POINTER, walked through the byte-exact PxgBroadPhaseDescGpu mirror
// (pxgSlangBpDesc.slang, proven by testBpDescLayout).  Thread mappings,
// grid-stride loops and the radix-sort padding tails mirror the .cu verbatim.

import pxgSlangBpDesc;

// ---- constants ----
static const uint PX_SIGN_BITMASK = 0x80000000u;
static const uint PX_INVALID_U32 = 0xffffffffu;
static const uint PXG_INVALID_BP_PROJECTION = 0xffffffffu;
static const uint PXG_INVALID_BP_HANDLE = 0x3fffffffu;
static const uint gEncodedMinExtent = 0x01800000u; // encodeFloat(-PX_MAX_BOUNDS_EXTENTS) & (~1)
static const uint gEncodedMaxExtent = 0xfe7fffffu; // (encodeFloat(PX_MAX_BOUNDS_EXTENTS) & (~1)) | 1

// mirrors encodeFloat(): monotonic float -> uint mapping (order-preserving)
uint encodeFloat(float f)
{
    const uint i = asuint(f);
    return (i & PX_SIGN_BITMASK) != 0 ? ~i : (i | PX_SIGN_BITMASK);
}

// ---------------------------------------------------------------------------
// translateAABBsLaunch (BP_TRANSLATE_AABBS) — explicit-pointer ABI
// ---------------------------------------------------------------------------
// Push constant mirrors the CUDA parameter list EXACTLY — four device pointers
// then boxesCapacity (PxU32) then six PxU8 shift bytes (bytes 36..41, read as
// two packed words).  hasEnvIds is the .cu's `envIds != NULL` pointer check.
struct TranslateAabbParams
{
    uint64_t inArray;          // @0  const PxBounds3*
    uint64_t outArray;         // @8  PxgIntegerAABB*
    uint64_t contactDistances; // @16 const PxReal*
    uint64_t envIds;           // @24 const PxU32*  (0 => no env ids)
    uint boxesCapacity;        // @32 PxU32
    uint packedShift0;         // @36 bytes [shiftX, shiftY, shiftZ, envIDX]
    uint packedShift1;         // @40 bytes [envIDY, envIDZ, pad, pad]
}
[[vk::push_constant]]
ConstantBuffer<TranslateAabbParams> gTranslate;

// 8 threads per box (boxIndex = tid>>3, a = tid&7, a<6 work / 2 idle), exactly
// the engine's grid arithmetic (nbBlocks = capacity / (blockDim/8)).
[shader("compute")]
[numthreads(256, 1, 1)]
void translateAABBsLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint boxIndex = dtid.x >> 3;
    if (boxIndex >= gTranslate.boxesCapacity)
        return;

    const uint shiftX = gTranslate.packedShift0 & 0xffu;
    const uint shiftY = (gTranslate.packedShift0 >> 8) & 0xffu;
    const uint shiftZ = (gTranslate.packedShift0 >> 16) & 0xffu;
    const uint envIDX = (gTranslate.packedShift0 >> 24) & 0xffu;
    const uint envIDY = gTranslate.packedShift1 & 0xffu;
    const uint envIDZ = (gTranslate.packedShift1 >> 8) & 0xffu;

    const uint shifts[6] = { shiftX, shiftY, shiftZ, shiftX, shiftY, shiftZ };
    const uint envShifts[6] = { envIDX, envIDY, envIDZ, envIDX, envIDY, envIDZ };

    Ptr<float> inBounds = Ptr<float>(gTranslate.inArray);            // PxBounds3 = 6 floats/box
    Ptr<float> contactDistances = Ptr<float>(gTranslate.contactDistances);
    Ptr<uint> outBounds = Ptr<uint>(gTranslate.outArray);            // PxgIntegerAABB = 6 uints/box

    const float contactDistance = contactDistances[boxIndex];
    const uint a = dtid.x & 7;
    if (a < 6)
    {
        const float inV = inBounds[boxIndex * 6 + a];
        const uint mask = a < 3 ? 0u : 1u;
        const float projection = a < 3 ? inV - contactDistance : inV + contactDistance;

        const uint shift = shifts[a];
        const uint envIdShift = envShifts[a];

        const uint b = (encodeFloat(projection) >> shift) & (~1u);
        uint outV = b | mask;

        if (gTranslate.envIds != 0 && envIdShift != 0)
        {
            const uint envId = (Ptr<uint>(gTranslate.envIds))[boxIndex];
            if (envId != PX_INVALID_U32)
                outV |= envId << (32 - envIdShift);
            else
                outV = (mask != 0) ? gEncodedMaxExtent : gEncodedMinExtent;
        }

        outBounds[boxIndex * 6 + a] = outV;
    }
}

// ---------------------------------------------------------------------------
// The bpDesc-aggregate kernels: ONE PxgBroadPhaseDesc device pointer.
// ---------------------------------------------------------------------------
struct BpDescParams
{
    uint64_t bpDesc;
}
[[vk::push_constant]]
ConstantBuffer<BpDescParams> gBp;

// mirrors markCreatedPairsLaunch (BP_UPDATE_CREATEDPAIRS): append each created
// box's per-axis min/max integer projections + start/end handles (isNew=true)
// at numPreviousHandles+i, setEmpty() its old bounds, then pad the projection/
// handle arrays to an even count for the radix sort.  uint2 writes -> 2 flat
// uints.  Grid-stride, blockDim 256, grid 32.
[shader("compute")]
[numthreads(256, 1, 1)]
void markCreatedPairsLaunch(uint3 dtid : SV_DispatchThreadID)
{
    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gBp.bpDesc);
    const uint numCreatedHandleSize = desc[0].numCreatedHandles;
    const uint startIndex = desc[0].numPreviousHandles;

    Ptr<uint> createdHandles = Ptr<uint>(desc[0].updateData_createdHandles);
    Ptr<uint> boxProjX = Ptr<uint>(desc[0].boxProjections[0]);
    Ptr<uint> boxProjY = Ptr<uint>(desc[0].boxProjections[1]);
    Ptr<uint> boxProjZ = Ptr<uint>(desc[0].boxProjections[2]);
    Ptr<uint> boxHandlesX = Ptr<uint>(desc[0].boxHandles[0]);
    Ptr<uint> boxHandlesY = Ptr<uint>(desc[0].boxHandles[1]);
    Ptr<uint> boxHandlesZ = Ptr<uint>(desc[0].boxHandles[2]);
    Ptr<uint> newBounds = Ptr<uint>(desc[0].newIntegerBounds);   // 6 uints/box
    Ptr<uint> oldBounds = Ptr<uint>(desc[0].oldIntegerBounds);

    const uint stride = WorkgroupCount().x * 256u;
    for (uint i = dtid.x; i < numCreatedHandleSize; i += stride)
    {
        const uint handle = createdHandles[i];
        const uint index = startIndex + i;

        const uint startHandle = bpCreateHandle(handle, true, true);
        const uint endHandle = bpCreateHandle(handle, false, true);

        // per-axis (min, max) projections from the new integer bounds
        [unroll]
        for (uint ax = 0; ax < 3; ++ax)
        {
            const uint mn = newBounds[handle * 6 + ax];
            const uint mx = newBounds[handle * 6 + 3 + ax];
            Ptr<uint> proj = ax == 0 ? boxProjX : (ax == 1 ? boxProjY : boxProjZ);
            Ptr<uint> hnd = ax == 0 ? boxHandlesX : (ax == 1 ? boxHandlesY : boxHandlesZ);
            proj[index * 2 + 0] = mn;
            proj[index * 2 + 1] = mx;
            hnd[index * 2 + 0] = startHandle;
            hnd[index * 2 + 1] = endHandle;
        }

        // oldBounds[handle].setEmpty()
        oldBounds[handle * 6 + 0] = 0xff7fffffu;
        oldBounds[handle * 6 + 1] = 0xff7fffffu;
        oldBounds[handle * 6 + 2] = 0xff7fffffu;
        oldBounds[handle * 6 + 3] = 0x00800000u;
        oldBounds[handle * 6 + 4] = 0x00800000u;
        oldBounds[handle * 6 + 5] = 0x00800000u;
    }

    // pad the handle count to even (radix-sort requirement), one uint2 per axis
    const uint numExistingProjections = numCreatedHandleSize + startIndex;
    const uint remainingProjections = numExistingProjections & 1u;
    if (dtid.x < remainingProjections)
    {
        const uint index = numExistingProjections + dtid.x;
        [unroll]
        for (uint ax = 0; ax < 3; ++ax)
        {
            Ptr<uint> proj = ax == 0 ? boxProjX : (ax == 1 ? boxProjY : boxProjZ);
            Ptr<uint> hnd = ax == 0 ? boxHandlesX : (ax == 1 ? boxHandlesY : boxHandlesZ);
            proj[index * 2 + 0] = PXG_INVALID_BP_PROJECTION;
            proj[index * 2 + 1] = PXG_INVALID_BP_PROJECTION;
            hnd[index * 2 + 0] = PXG_INVALID_BP_HANDLE;
            hnd[index * 2 + 1] = PXG_INVALID_BP_HANDLE;
        }
    }
}

// mirrors initializeSapBox1DLaunch (BP_INITIALIZE_SAPBOX): 1 thread per
// projection (no grid-stride); writes sapBox[handle].mMinMax[isStart?0:1] = i
// into the new or current sap boxes (VALUE branch on isNew, not a null check).
// ABI: (bpDesc*, PxU32 numHandles, bool isNew) — the bool is byte 12 of the
// blob; bytes 13..15 are zero padding, so the masked word read is exact.
struct SapBoxParams
{
    uint64_t bpDesc;   // @0
    uint numHandles;   // @8
    uint isNewWord;    // @12: bool in byte 0
}
[[vk::push_constant]]
ConstantBuffer<SapBoxParams> gSapInit;

[shader("compute")]
[numthreads(256, 1, 1)]
void initializeSapBox1DLaunch(uint3 dtid : SV_DispatchThreadID)
{
    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gSapInit.bpDesc);
    const uint numProjections = gSapInit.numHandles * 2u;
    const bool isNew = (gSapInit.isNewWord & 0xffu) != 0;
    if (dtid.x >= numProjections)
        return;

    [unroll]
    for (uint ax = 0; ax < 3; ++ax)
    {
        Ptr<uint> sapBox = Ptr<uint>(isNew ? desc[0].boxNewSapBox1D[ax] : desc[0].boxSapBox1D[ax]);
        Ptr<uint> handles = Ptr<uint>(desc[0].boxHandles[ax]); // boxHandles[0][ax]
        const uint sortedHandle = handles[dtid.x];
        const uint handle = bpGetHandle(sortedHandle);
        const uint slot = bpIsStartProjection(sortedHandle) ? 0u : 1u;
        sapBox[handle * 2u + slot] = dtid.x;
    }
}

// mirrors initializeRadixRanks (BP_INITIALIZE_RANKS): identity ranks written
// uint4-at-a-time to all three axes' rank arrays, grid-stride over
// ceil(nbProjections/4) groups (output padded to a multiple of 4).
[shader("compute")]
[numthreads(256, 1, 1)]
void initializeRadixRanks(uint3 dtid : SV_DispatchThreadID)
{
    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gBp.bpDesc);
    const uint nbProjections = desc[0].numPreviousHandles * 2u;
    const uint nbProjectionsOver4 = (nbProjections + 3u) / 4u;

    Ptr<uint> ranksX = Ptr<uint>(desc[0].boxProjectionRanks[0]);
    Ptr<uint> ranksY = Ptr<uint>(desc[0].boxProjectionRanks[1]);
    Ptr<uint> ranksZ = Ptr<uint>(desc[0].boxProjectionRanks[2]);

    const uint stride = WorkgroupCount().x * 256u;
    for (uint i = dtid.x; i < nbProjectionsOver4; i += stride)
    {
        [unroll]
        for (uint c = 0; c < 4; ++c)
        {
            ranksX[i * 4u + c] = i * 4u + c;
            ranksY[i * 4u + c] = i * 4u + c;
            ranksZ[i * 4u + c] = i * 4u + c;
        }
    }
}

// mirrors updateHandles (BP_UDPATE_HANDLES): gather-permute per axis —
// boxHandles[0][ax][i] = boxHandles[1][ax][ranks[ax][i]], grid-stride over
// nbProjections.
[shader("compute")]
[numthreads(256, 1, 1)]
void updateHandles(uint3 dtid : SV_DispatchThreadID)
{
    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gBp.bpDesc);
    const uint nbProjections = desc[0].numPreviousHandles * 2u;

    const uint stride = WorkgroupCount().x * 256u;
    for (uint i = dtid.x; i < nbProjections; i += stride)
    {
        [unroll]
        for (uint ax = 0; ax < 3; ++ax)
        {
            Ptr<uint> ranks = Ptr<uint>(desc[0].boxProjectionRanks[ax]);
            Ptr<uint> cur = Ptr<uint>(desc[0].boxHandles[ax]);      // boxHandles[0][ax]
            Ptr<uint> tmp = Ptr<uint>(desc[0].boxHandles[3 + ax]);  // boxHandles[1][ax]
            cur[i] = tmp[ranks[i]];
        }
    }
}
