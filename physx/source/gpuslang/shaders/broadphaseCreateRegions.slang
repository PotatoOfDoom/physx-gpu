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

// broadphaseCreateRegions.slang
//
// Slang port of createRegionsKernel (BP_CREATE_REGIONS) from
// gpubroadphase/src/CUDA/broadphase.cu.  This is the region producer of the
// one-shot broad phase: it partitions the world into REGION_SIZE_PER_AXIS^3
// coarse regions and computes, for every box, which region span its bounds
// cover, plus the per-projection region-overlap counts that the region
// prefix-sum scan pair (broadphaseRegionHistogram) then accumulates.
//
// For each projection slot on the (current-frame) X-sorted handle list, only a
// start projection does work.  For its box it maps the box's min/max endpoint
// positions on each axis (from boxSapBox1D) through the total-end-point
// histogram (totalEndPtHistogram, produced by broadphaseEndPtHistogram) into an
// integer coordinate, divides by elementsPerRegions to get the [min,max] region
// index span, and writes:
//   - regionRange[handle]      = { minRange = (sRegionXYZ), maxRange = (eRegionXYZ) };
//   - startRegionAccum at the box's X min position = (eRegionY-sRegionY+1)*(eRegionZ-sRegionZ+1),
//     and 0 at the box's X max position;
//   - regionAccum[workIndex]   = (eRegionX-sRegionX+1) * that yz product (0 if eRegionX < sRegionX).
// End projections (and every non-start slot) write regionAccum[workIndex] = 0.
//
// Pure per-thread gather/scatter — no groupshared, scans, atomics or barriers.
// The fixed 32-block grid and its contiguous per-block partition are preserved.

static const uint BLOCK = 256;
static const uint NUM_BLOCKS = 32;
static const uint REGION_SIZE_PER_AXIS = 4;

struct CreateRegionsParams
{
    uint numHandles; // #boxes; numProjections = numHandles * 2
}

[[vk::push_constant]]
ConstantBuffer<CreateRegionsParams> gParams;

RWStructuredBuffer<uint> gTotalEndHist;     // binding 0  totalEndPtHistogram[axis] flat [3][numProj]  (R)
RWStructuredBuffer<uint> gHandlesX;         // binding 1  boxHandles[0][AXIS_X] [numProj]              (R)
RWStructuredBuffer<uint> gSapBox;           // binding 2  boxSapBox1D[axis] flat [3][numHandles][2]     (R)
RWStructuredBuffer<uint> gRegionRange;      // binding 3  regionRange[handle] flat [numHandles][8]       (W)
RWStructuredBuffer<uint> gStartRegionAccum; // binding 4  startRegionAccum [numProj]                     (W)
RWStructuredBuffer<uint> gRegionAccum;      // binding 5  regionAccum [numProj]                          (W)

bool isStartProjection(uint sortedHandle) { return (sortedHandle & 1u) != 0u; }
uint getHandle(uint sortedHandle) { return sortedHandle >> 3; }

[shader("compute")]
[numthreads(256, 1, 1)]
void createRegionsKernel(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;

    const uint numHandles = gParams.numHandles;
    const uint numProjections = numHandles * 2u;
    const uint totalBlockRequired = (numProjections + BLOCK - 1u) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + NUM_BLOCKS - 1u) / NUM_BLOCKS;
    const uint elementsPerRegions = (numHandles + (REGION_SIZE_PER_AXIS - 1u)) / REGION_SIZE_PER_AXIS;

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        if (workIndex >= numProjections)
            continue;

        const uint sortedHandle = gHandlesX[workIndex];
        uint regionCount = 0;

        if (isStartProjection(sortedHandle))
        {
            const uint handle = getHandle(sortedHandle);

            // X axis
            const uint sapXBase = (0u * numHandles + handle) * 2u;
            const uint startIndexX = gSapBox[sapXBase + 0u];
            const uint endIndexX = gSapBox[sapXBase + 1u];
            const uint sRegionX = gTotalEndHist[0u * numProjections + startIndexX] / elementsPerRegions;
            const uint eRegionX = gTotalEndHist[0u * numProjections + endIndexX] / elementsPerRegions;

            // Y axis
            const uint sapYBase = (1u * numHandles + handle) * 2u;
            const uint sRegionY = gTotalEndHist[1u * numProjections + gSapBox[sapYBase + 0u]] / elementsPerRegions;
            const uint eRegionY = gTotalEndHist[1u * numProjections + gSapBox[sapYBase + 1u]] / elementsPerRegions;

            // Z axis
            const uint sapZBase = (2u * numHandles + handle) * 2u;
            const uint sRegionZ = gTotalEndHist[2u * numProjections + gSapBox[sapZBase + 0u]] / elementsPerRegions;
            const uint eRegionZ = gTotalEndHist[2u * numProjections + gSapBox[sapZBase + 1u]] / elementsPerRegions;

            const uint rr = handle * 8u;
            gRegionRange[rr + 0u] = sRegionX;
            gRegionRange[rr + 1u] = sRegionY;
            gRegionRange[rr + 2u] = sRegionZ;
            gRegionRange[rr + 3u] = 0u;
            gRegionRange[rr + 4u] = eRegionX;
            gRegionRange[rr + 5u] = eRegionY;
            gRegionRange[rr + 6u] = eRegionZ;
            gRegionRange[rr + 7u] = 0u;

            const uint yzRegionAccum = (eRegionY - sRegionY + 1u) * (eRegionZ - sRegionZ + 1u);
            gStartRegionAccum[startIndexX] = yzRegionAccum;
            gStartRegionAccum[endIndexX] = 0u;

            if (eRegionX >= sRegionX)
                regionCount = (eRegionX - sRegionX + 1u) * yzRegionAccum;
        }

        gRegionAccum[workIndex] = regionCount;
    }
}
