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
// P5 Step 4 (BDA, engine launch path): createRegionsKernel (BP_CREATE_REGIONS)
// from gpubroadphase/src/CUDA/broadphase.cu, driven exactly as the engine
// launches it — ONE PxgBroadPhaseDesc device pointer, walked through the
// byte-exact PxgBroadPhaseDescGpu mirror (pxgSlangBpDesc.slang, proven by
// testBpDescLayout).  numHandles comes from the desc's
// numPreviousHandles + numCreatedHandles - numRemovedHandles, as the .cu reads
// it; the fixed 32-block grid's contiguous per-block partition
// (numIterationPerBlock uses the BLOCK_SIZE constant, not gridDim) mirrors the
// .cu verbatim.
//
// This is the region producer of the one-shot broad phase: it partitions the
// world into REGION_SIZE_PER_AXIS^3 coarse regions and computes, for every box,
// which region span its bounds cover, plus the per-projection region-overlap
// counts that the region prefix-sum scan pair (broadphaseRegionHistogram) then
// accumulates.  For each projection slot on the current-frame X-sorted handle
// list, only a start projection does work.  For its box it maps the box's
// min/max endpoint positions on each axis (from boxSapBox1D) through the
// total-end-point histogram (totalEndPtHistogram) into an integer coordinate,
// divides by elementsPerRegions to get the [min,max] region index span, and
// writes:
//   - regionRange[handle]      = { minRange = (sRegionXYZ), maxRange = (eRegionXYZ) };
//   - startRegionAccum at the box's X min position = (eRegionY-sRegionY+1)*(eRegionZ-sRegionZ+1),
//     and 0 at the box's X max position;
//   - regionAccum[workIndex]   = (eRegionX-sRegionX+1) * that yz product (0 if eRegionX < sRegionX).
// End projections (and every non-start slot) write regionAccum[workIndex] = 0.
//
// Pure per-thread gather/scatter — no groupshared, scans, atomics or barriers.
// PxgIntegerRegion = { uint4 minRange; uint4 maxRange } = 8 flat uints;
// PxgSapBox1D = { PxU32 mMinMax[2] } = 2 flat uints.

import pxgSlangBpDesc;

static const uint BLOCK = 256;      // PxgBPKernelBlockDim::BP_CREATE_REGIONS
static const uint NUM_BLOCKS = 32;  // BLOCK_SIZE in the .cu == PxgBPKernelGridDim::BP_CREATE_REGIONS
static const uint REGION_SIZE_PER_AXIS = 4;

struct BpDescParams
{
    uint64_t bpDesc;
}
[[vk::push_constant]]
ConstantBuffer<BpDescParams> gBp;

[shader("compute")]
[numthreads(256, 1, 1)]
void createRegionsKernel(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gBp.bpDesc);

    Ptr<uint> endPtsHistogramX = Ptr<uint>(desc[0].totalEndPtHistogram[0]);
    Ptr<uint> endPtsHistogramY = Ptr<uint>(desc[0].totalEndPtHistogram[1]);
    Ptr<uint> endPtsHistogramZ = Ptr<uint>(desc[0].totalEndPtHistogram[2]);

    Ptr<uint> handlesX = Ptr<uint>(desc[0].boxHandles[0]); // boxHandles[0][AXIS_X], current frame
    Ptr<uint> sapBoxX = Ptr<uint>(desc[0].boxSapBox1D[0]); // PxgSapBox1D = 2 uints/box
    Ptr<uint> sapBoxY = Ptr<uint>(desc[0].boxSapBox1D[1]);
    Ptr<uint> sapBoxZ = Ptr<uint>(desc[0].boxSapBox1D[2]);

    Ptr<uint> regionRange = Ptr<uint>(desc[0].regionRange); // PxgIntegerRegion = 8 uints/box
    Ptr<uint> startRegionAccum = Ptr<uint>(desc[0].startRegionAccum);
    Ptr<uint> regionAccum = Ptr<uint>(desc[0].regionAccum);

    const uint numHandles = desc[0].numPreviousHandles + desc[0].numCreatedHandles - desc[0].numRemovedHandles;
    const uint numProjections = numHandles * 2u;

    const uint totalBlockRequired = (numProjections + (BLOCK - 1u)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1u)) / NUM_BLOCKS;
    const uint elementsPerRegions = (numHandles + (REGION_SIZE_PER_AXIS - 1u)) / REGION_SIZE_PER_AXIS;

    const uint idx = gtid.x;

    for (uint i = 0; i < numIterationPerBlock; ++i)
    {
        const uint workIndex = i * BLOCK + idx + numIterationPerBlock * gid.x * BLOCK;
        if (workIndex < numProjections)
        {
            // get handle
            const uint sortedHandle = handlesX[workIndex];
            uint regionCount = 0;
            if (bpIsStartProjection(sortedHandle))
            {
                const uint handle = bpGetHandle(sortedHandle);

                // calculate region index in x axis
                const uint startIndex = sapBoxX[handle * 2u + 0u];
                const uint endIndex = sapBoxX[handle * 2u + 1u];

                const uint startIndexX = endPtsHistogramX[startIndex];
                const uint endIndexX = endPtsHistogramX[endIndex];
                const uint sRegionX = startIndexX / elementsPerRegions;
                const uint eRegionX = endIndexX / elementsPerRegions;

                // calculate region index in y axis
                const uint startIndexY = endPtsHistogramY[sapBoxY[handle * 2u + 0u]];
                const uint endIndexY = endPtsHistogramY[sapBoxY[handle * 2u + 1u]];
                const uint sRegionY = startIndexY / elementsPerRegions;
                const uint eRegionY = endIndexY / elementsPerRegions;

                // calculate region index in z axis
                const uint startIndexZ = endPtsHistogramZ[sapBoxZ[handle * 2u + 0u]];
                const uint endIndexZ = endPtsHistogramZ[sapBoxZ[handle * 2u + 1u]];
                const uint sRegionZ = startIndexZ / elementsPerRegions;
                const uint eRegionZ = endIndexZ / elementsPerRegions;

                // regionRange[handle].minRange / .maxRange (uint4 pair, w = 0)
                regionRange[handle * 8u + 0u] = sRegionX;
                regionRange[handle * 8u + 1u] = sRegionY;
                regionRange[handle * 8u + 2u] = sRegionZ;
                regionRange[handle * 8u + 3u] = 0u;
                regionRange[handle * 8u + 4u] = eRegionX;
                regionRange[handle * 8u + 5u] = eRegionY;
                regionRange[handle * 8u + 6u] = eRegionZ;
                regionRange[handle * 8u + 7u] = 0u;

                const uint yzRegionAccum = (eRegionY - sRegionY + 1u) * (eRegionZ - sRegionZ + 1u);
                startRegionAccum[startIndex] = yzRegionAccum;
                startRegionAccum[endIndex] = 0u;

                if (eRegionX >= sRegionX)
                    regionCount = (eRegionX - sRegionX + 1u) * yzRegionAccum;
            }
            regionAccum[workIndex] = regionCount;
        }
    }
}
