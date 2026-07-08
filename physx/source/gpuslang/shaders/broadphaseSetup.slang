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
// Slang port of the GPU broad-phase (incremental SAP) foundation kernels
// (gpubroadphase/src/CUDA/broadphase.cu):
//   - translateAABBsLaunch: quantize float AABBs (PxBounds3) into the sortable
//     integer projection layout (PxgIntegerAABB), applying the contact-distance
//     inflation, the monotonic float->uint encoding, the axis bit-shift and the
//     environment-ID packing.
//   - markCreatedPairsLaunch: append the newly inserted bounds' min/max integer
//     projections and their start/end sort handles to the per-axis arrays (and
//     pad the tail to an even count for the radix sort).
//   - initializeSapBox1DLaunch: after the projections are sorted, record each
//     box's min/max endpoint indices into its 1D SAP box.
//
// These establish the integer-AABB + SAP-box data structures the rest of the
// broad-phase builds on.  The descriptor pointer members are bound as explicit
// buffers and the scalar members arrive as push constants (same policy as the
// solver ports).

// ---- constants ----
static const uint PX_SIGN_BITMASK = 0x80000000u;
static const uint PX_INVALID_U32 = 0xffffffffu;
static const uint PXG_INVALID_BP_PROJECTION = 0xffffffffu;
static const uint PXG_INVALID_BP_HANDLE = 0x3fffffffu;
static const uint gEncodedMinExtent = 0x01800000u; // encodeFloat(-PX_MAX_BOUNDS_EXTENTS) & (~1)
static const uint gEncodedMaxExtent = 0xfe7fffffu; // (encodeFloat(PX_MAX_BOUNDS_EXTENTS) & (~1)) | 1

// mirrors PxgIntegerAABB (24 bytes): mMinMax[6] = {MIN_X,MIN_Y,MIN_Z,MAX_X,MAX_Y,MAX_Z}
struct PxgIntegerAABB
{
    uint mMinMax[6];
}

// mirrors PxgSapBox1D (8 bytes): mMinMax[0] = min endpoint index, [1] = max
struct PxgSapBox1D
{
    uint mMinMax[2];
}

struct BpParams
{
    // translateAABBsLaunch
    uint boxesCapacity;
    uint shiftX, shiftY, shiftZ;
    uint envShiftX, envShiftY, envShiftZ;
    uint hasEnvIds;
    // markCreatedPairsLaunch
    uint numCreatedHandles;
    uint numPreviousHandles;
    // initializeSapBox1DLaunch
    uint numHandles;
}

[[vk::push_constant]]
ConstantBuffer<BpParams> gParams;

RWStructuredBuffer<float> gInBounds;               // binding 0 (PxBounds3: 6 floats/box)
RWStructuredBuffer<PxgIntegerAABB> gOutBounds;     // binding 1
RWStructuredBuffer<float> gContactDistances;       // binding 2
RWStructuredBuffer<uint> gEnvIds;                  // binding 3
RWStructuredBuffer<uint> gCreatedHandles;          // binding 4
RWStructuredBuffer<uint> gBoxProjX;                // binding 5 (flat uint, 2 per box)
RWStructuredBuffer<uint> gBoxProjY;                // binding 6
RWStructuredBuffer<uint> gBoxProjZ;                // binding 7
RWStructuredBuffer<uint> gBoxHandlesX;             // binding 8 (flat uint, 2 per box)
RWStructuredBuffer<uint> gBoxHandlesY;             // binding 9
RWStructuredBuffer<uint> gBoxHandlesZ;             // binding 10
RWStructuredBuffer<PxgIntegerAABB> gNewIntegerBounds; // binding 11
RWStructuredBuffer<PxgIntegerAABB> gOldIntegerBounds; // binding 12
RWStructuredBuffer<PxgSapBox1D> gSapBoxX;          // binding 13
RWStructuredBuffer<PxgSapBox1D> gSapBoxY;          // binding 14
RWStructuredBuffer<PxgSapBox1D> gSapBoxZ;          // binding 15

// mirrors encodeFloat(): monotonic float -> uint mapping (order-preserving)
uint encodeFloat(float f)
{
    const uint i = asuint(f);
    return (i & PX_SIGN_BITMASK) != 0 ? ~i : (i | PX_SIGN_BITMASK);
}

// handle helpers (PxgBroadPhaseDesc.h)
uint createHandle(uint handle, bool isStart, bool isNew)
{
    return (handle << 3) | (isStart ? 1u : 0u) | (isNew ? 2u : 0u);
}
uint getHandle(uint sortedHandle) { return sortedHandle >> 3; }
bool isStartProjection(uint sortedHandle) { return (sortedHandle & 1u) != 0; }

void setEmpty(uint boundIndex)
{
    gOldIntegerBounds[boundIndex].mMinMax[0] = 0xff7fffffu;
    gOldIntegerBounds[boundIndex].mMinMax[1] = 0xff7fffffu;
    gOldIntegerBounds[boundIndex].mMinMax[2] = 0xff7fffffu;
    gOldIntegerBounds[boundIndex].mMinMax[3] = 0x00800000u;
    gOldIntegerBounds[boundIndex].mMinMax[4] = 0x00800000u;
    gOldIntegerBounds[boundIndex].mMinMax[5] = 0x00800000u;
}

// mirrors translateAABBsLaunch(): one thread per box computes all 6 components
// (the CUDA original uses 8 threads/box; the result is identical)
[shader("compute")]
[numthreads(256, 1, 1)]
void translateAABBsLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint boxIndex = dtid.x;
    if (boxIndex >= gParams.boxesCapacity)
        return;

    const uint shifts[6] = { gParams.shiftX, gParams.shiftY, gParams.shiftZ,
                             gParams.shiftX, gParams.shiftY, gParams.shiftZ };
    const uint envShifts[6] = { gParams.envShiftX, gParams.envShiftY, gParams.envShiftZ,
                                gParams.envShiftX, gParams.envShiftY, gParams.envShiftZ };

    const float contactDistance = gContactDistances[boxIndex];

    for (uint a = 0; a < 6; ++a)
    {
        const float inV = gInBounds[boxIndex * 6 + a];
        const uint mask = a < 3 ? 0u : 1u;
        const float projection = a < 3 ? inV - contactDistance : inV + contactDistance;

        const uint shift = shifts[a];
        const uint envIdShift = envShifts[a];

        const uint b = (encodeFloat(projection) >> shift) & (~1u);
        uint outV = b | mask;

        if (gParams.hasEnvIds != 0 && envIdShift != 0)
        {
            const uint envId = gEnvIds[boxIndex];
            if (envId != PX_INVALID_U32)
                outV |= envId << (32 - envIdShift);
            else
                outV = (mask != 0) ? gEncodedMaxExtent : gEncodedMinExtent;
        }

        gOutBounds[boxIndex].mMinMax[a] = outV;
    }
}

// mirrors markCreatedPairsLaunch(): append the created bounds' projections and
// sort handles at [numPreviousHandles + i], padding the tail to an even count
[shader("compute")]
[numthreads(256, 1, 1)]
void markCreatedPairsLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint numCreated = gParams.numCreatedHandles;
    const uint startIndex = gParams.numPreviousHandles;
    const uint stride = 256 * 64; // arbitrary grid-stride upper bound

    for (uint i = dtid.x; i < numCreated; i += stride)
    {
        const uint handle = gCreatedHandles[i];
        const uint index = startIndex + i;

        const uint startHandle = createHandle(handle, true, true);
        const uint endHandle = createHandle(handle, false, true);

        // uint2 projection = (getMin(axis), getMax(axis))
        gBoxProjX[index * 2 + 0] = gNewIntegerBounds[handle].mMinMax[0]; // MIN_X
        gBoxProjX[index * 2 + 1] = gNewIntegerBounds[handle].mMinMax[3]; // MAX_X
        gBoxProjY[index * 2 + 0] = gNewIntegerBounds[handle].mMinMax[1]; // MIN_Y
        gBoxProjY[index * 2 + 1] = gNewIntegerBounds[handle].mMinMax[4]; // MAX_Y
        gBoxProjZ[index * 2 + 0] = gNewIntegerBounds[handle].mMinMax[2]; // MIN_Z
        gBoxProjZ[index * 2 + 1] = gNewIntegerBounds[handle].mMinMax[5]; // MAX_Z

        gBoxHandlesX[index * 2 + 0] = startHandle;
        gBoxHandlesX[index * 2 + 1] = endHandle;
        gBoxHandlesY[index * 2 + 0] = startHandle;
        gBoxHandlesY[index * 2 + 1] = endHandle;
        gBoxHandlesZ[index * 2 + 0] = startHandle;
        gBoxHandlesZ[index * 2 + 1] = endHandle;

        setEmpty(handle);
    }

    // pad the projection/handle count to a multiple of 2 for the radix sort
    const uint numExistingProjections = numCreated + startIndex;
    const uint remaining = numExistingProjections & 1u;
    if (dtid.x < remaining)
    {
        const uint index = numExistingProjections + dtid.x;
        gBoxProjX[index * 2 + 0] = PXG_INVALID_BP_PROJECTION;
        gBoxProjX[index * 2 + 1] = PXG_INVALID_BP_PROJECTION;
        gBoxHandlesX[index * 2 + 0] = PXG_INVALID_BP_HANDLE;
        gBoxHandlesX[index * 2 + 1] = PXG_INVALID_BP_HANDLE;
        gBoxProjY[index * 2 + 0] = PXG_INVALID_BP_PROJECTION;
        gBoxProjY[index * 2 + 1] = PXG_INVALID_BP_PROJECTION;
        gBoxHandlesY[index * 2 + 0] = PXG_INVALID_BP_HANDLE;
        gBoxHandlesY[index * 2 + 1] = PXG_INVALID_BP_HANDLE;
        gBoxProjZ[index * 2 + 0] = PXG_INVALID_BP_PROJECTION;
        gBoxProjZ[index * 2 + 1] = PXG_INVALID_BP_PROJECTION;
        gBoxHandlesZ[index * 2 + 0] = PXG_INVALID_BP_HANDLE;
        gBoxHandlesZ[index * 2 + 1] = PXG_INVALID_BP_HANDLE;
    }
}

// mirrors initializeSapBox1DLaunch(): from the sorted per-axis handle arrays,
// store each box's min/max endpoint index into its SAP box.  The caller binds
// either the "new" or the regular SAP-box arrays (the CUDA isNew pointer
// select becomes a binding choice).
[shader("compute")]
[numthreads(256, 1, 1)]
void initializeSapBox1DLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint numProjections = gParams.numHandles * 2;
    const uint i = dtid.x;
    if (i >= numProjections)
        return;

    const uint sortedHandleX = gBoxHandlesX[i];
    const uint sortedHandleY = gBoxHandlesY[i];
    const uint sortedHandleZ = gBoxHandlesZ[i];

    const uint handleX = getHandle(sortedHandleX);
    const uint handleY = getHandle(sortedHandleY);
    const uint handleZ = getHandle(sortedHandleZ);

    gSapBoxX[handleX].mMinMax[isStartProjection(sortedHandleX) ? 0u : 1u] = i;
    gSapBoxY[handleY].mMinMax[isStartProjection(sortedHandleY) ? 0u : 1u] = i;
    gSapBoxZ[handleZ].mMinMax[isStartProjection(sortedHandleZ) ? 0u : 1u] = i;
}
