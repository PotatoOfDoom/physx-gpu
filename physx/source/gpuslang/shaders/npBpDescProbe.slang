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

// npBpDescProbe — P5 Step 4: the empirical layout proof for the
// PxgBroadPhaseDescGpu mirror (pxgSlangBpDesc.slang).  The host test fills a
// REAL engine-side PxgBroadPhaseDesc with distinct sentinels, uploads it, and
// this kernel reads fields spanning the whole struct through the mirror into
// an output array the host compares bitwise.  A single wrong offset (missing
// padding, wrong array stride, drifted engine struct) mismatches.

import pxgSlangBpDesc;

struct ProbeParams
{
    uint64_t desc; // device address of the engine-uploaded PxgBroadPhaseDesc
    uint64_t out;  // device address of uint64_t[PROBE_COUNT]
}

[[vk::push_constant]]
ConstantBuffer<ProbeParams> gProbe;

[shader("compute")]
[numthreads(1, 1, 1)]
void bpDescProbe(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x != 0)
        return;
    PxgBroadPhaseDescGpu d = (Ptr<PxgBroadPhaseDescGpu>(gProbe.desc))[0];
    Ptr<uint64_t> o = Ptr<uint64_t>(gProbe.out);
    uint i = 0;
    o[i++] = d.updateData_createdHandles;
    o[i++] = uint64_t(d.numCreatedHandles);
    o[i++] = d.updateData_removedHandles;
    o[i++] = uint64_t(d.numRemovedHandles);
    o[i++] = d.updateData_updatedHandles;
    o[i++] = uint64_t(d.numUpdatedHandles);
    o[i++] = d.aabbMngr_changedHandleMap;
    o[i++] = uint64_t(d.aabbMngr_changedHandleBitMapWordCounts);
    o[i++] = d.aabbMngr_addedHandleMap;
    o[i++] = d.aabbMngr_aggregatedBoundHandles;
    o[i++] = d.updateData_fpBounds;
    o[i++] = d.updateData_contactDistances;
    o[i++] = d.newIntegerBounds;
    o[i++] = d.updateData_envIDs;
    o[i++] = d.oldIntegerBounds;
    o[i++] = d.boxSapBox1D[0];
    o[i++] = d.boxSapBox1D[2];
    o[i++] = d.boxNewSapBox1D[1];
    o[i++] = d.boxProjections[2];
    o[i++] = d.boxHandles[0];      // [0][0]
    o[i++] = d.boxHandles[5];      // [1][2]
    o[i++] = uint64_t(d.numPreviousHandles);
    o[i++] = uint64_t(d.sharedFoundPairIndex);
    o[i++] = uint64_t(d.sharedLostAggPairIndex);
    o[i++] = d.foundPairReport;
    o[i++] = d.lostActorPairReport;
    o[i++] = d.lostPairReportMap;
    o[i++] = d.aabbMngr_volumeData;
    o[i++] = d.activeRegionsHistogram;
    o[i++] = d.orderedStartRegionHandles;
    o[i++] = d.blockOverlapChecksRegion;
    o[i++] = d.overlapChecksTotalRegion;
    o[i++] = d.overlapChecksHandleRegiones;
    o[i++] = d.regionRange;
    o[i++] = d.blockStartRegionAccum;
    o[i++] = uint64_t(d.startRegionAccumTotal);
    o[i++] = d.regionAccum;
    o[i++] = uint64_t(d.regionAccumTotal);
    o[i++] = d.endPtHistogram[0];     // [0][0]
    o[i++] = d.endPtHistogram[5];     // [1][2]
    o[i++] = d.blockEndPtHistogram[4];// [1][1]
    o[i++] = d.endPointHandles[1];    // [0][1]
    o[i++] = d.totalEndPtHistogram[2];
    o[i++] = d.blockTotalEndPtHistogram[0];
    o[i++] = d.startPtHistogram[3];   // [1][0]
    o[i++] = d.blockStartPtHistogram[2]; // [0][2]
    o[i++] = d.startPointHandles[5];  // [1][2]
    o[i++] = d.boxProjectionRanks[1];
    o[i++] = d.incrementalComparisons[0];
    o[i++] = d.incrementalBlockComparisons[2];
    o[i++] = d.aggReportBlock[1];
    o[i++] = d.actorReportBlock[0];
    o[i++] = uint64_t(d.totalIncrementalComparisons[0]);
    o[i++] = uint64_t(d.totalIncrementalComparisons[2]);
    o[i++] = uint64_t(d.max_found_lost_pairs);
    o[i++] = uint64_t(d.max_found_lost_agg_pairs);
    o[i++] = uint64_t(d.overflowFlagsWord & 0xffu); // the trailing bool byte
}
