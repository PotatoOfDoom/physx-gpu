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

// pxgSlangBpDesc — RUNTIME_PLAN.md P5 Step 4: the BDA mirror of the engine's
// PxgBroadPhaseDesc (gpubroadphase/include/PxgBroadPhaseDesc.h).  Most of the
// broad-phase kernels receive ONE device pointer to this aggregate and walk
// its nested device pointers, so the Slang side needs a byte-exact mirror.
//
// Field-for-field transcription with SUPPORT_UPDATE_HANDLES_ARRAY_FOR_GPU
// defined (it is, PxgBroadPhaseDesc.h:38).  Pointers -> uint64_t, PxU32 ->
// uint, regionOverlapType = PxU64 -> uint64_t, trailing bool -> read via
// overflowFlagsWord (& 0xff).  Natural C layout: uint64 fields align 8, so a
// pointer following an odd number of uints gets 4 bytes of padding — the
// explicit padN fields reproduce that, because Slang packs physical-storage
// structs tightly unless told otherwise.  The layout is verified EMPIRICALLY,
// not by inspection: tests/testBpDescLayout.cpp fills a real engine-side
// PxgBroadPhaseDesc with sentinels and the bpDescProbe kernel reads fields
// across the whole struct through this mirror (bitwise compare).  Any drift
// between compiler and mirror fails that test.
module pxgSlangBpDesc;

public struct PxgBroadPhaseDescGpu
{
    public uint64_t updateData_createdHandles;              // @0
    public uint     numCreatedHandles;                      // @8
    uint pad0;                                              // @12 (align next ptr)
    public uint64_t updateData_removedHandles;              // @16
    public uint     numRemovedHandles;                      // @24
    uint pad1;                                              // @28
    public uint64_t updateData_updatedHandles;              // @32 (SUPPORT_UPDATE_HANDLES...)
    public uint     numUpdatedHandles;                      // @40
    uint pad2;                                              // @44
    public uint64_t aabbMngr_changedHandleMap;              // @48
    public uint     aabbMngr_changedHandleBitMapWordCounts; // @56
    uint pad3;                                              // @60
    public uint64_t aabbMngr_addedHandleMap;                // @64
    public uint64_t aabbMngr_removedHandleMap;              // @72
    public uint64_t aabbMngr_aggregatedBoundHandles;        // @80
    public uint64_t updateData_fpBounds;                    // @88
    public uint64_t updateData_contactDistances;            // @96
    public uint64_t newIntegerBounds;                       // @104
    public uint64_t updateData_groups;                      // @112
    public uint64_t updateData_envIDs;                      // @120
    public uint64_t oldIntegerBounds;                       // @128
    public uint64_t boxSapBox1D[3];                         // @136
    public uint64_t boxNewSapBox1D[3];                      // @160
    public uint64_t boxProjections[3];                      // @184
    public uint64_t boxHandles[6];                          // @208  [2][3] flattened: [cur X,Y,Z, new X,Y,Z]
    public uint     numPreviousHandles;                     // @256
    public uint     sharedFoundPairIndex;                   // @260
    public uint     sharedLostPairIndex;                    // @264
    public uint     sharedFoundAggPairIndex;                // @268
    public uint     sharedLostAggPairIndex;                 // @272
    uint pad4;                                              // @276
    public uint64_t foundPairReport;                        // @280
    public uint64_t lostPairReport;                         // @288
    public uint64_t foundAggPairReport;                     // @296
    public uint64_t lostAggPairReport;                      // @304
    public uint64_t foundActorPairReport;                   // @312
    public uint64_t lostActorPairReport;                    // @320
    public uint64_t foundPairReportMap;                     // @328
    public uint64_t lostPairReportMap;                      // @336
    public uint64_t aabbMngr_volumeData;                    // @344
    public uint64_t activeRegionsHistogram;                 // @352
    public uint64_t startRegionsHistogram;                  // @360
    public uint64_t orderedActiveRegionHandles;             // @368
    public uint64_t orderedStartRegionHandles;              // @376
    public uint64_t blockOverlapChecksRegion;               // @384
    public uint64_t overlapChecksRegion;                    // @392
    public uint64_t overlapChecksTotalRegion;               // @400  (regionOverlapType = PxU64 VALUE)
    public uint64_t overlapChecksHandleRegiones;            // @408
    public uint64_t regionRange;                            // @416
    public uint64_t startRegionAccum;                       // @424
    public uint64_t blockStartRegionAccum;                  // @432
    public uint     startRegionAccumTotal;                  // @440
    uint pad5;                                              // @444
    public uint64_t regionAccum;                            // @448
    public uint64_t blockRegionAccum;                       // @456
    public uint     regionAccumTotal;                       // @464
    uint pad6;                                              // @468
    public uint64_t endPtHistogram[6];                      // @472  [2][3] flattened
    public uint64_t blockEndPtHistogram[6];                 // @520
    public uint64_t endPointHandles[6];                     // @568
    public uint64_t totalEndPtHistogram[3];                 // @616
    public uint64_t blockTotalEndPtHistogram[3];            // @640
    public uint64_t startPtHistogram[6];                    // @664  [2][3] flattened
    public uint64_t blockStartPtHistogram[6];               // @712
    public uint64_t startPointHandles[6];                   // @760
    public uint64_t boxProjectionRanks[3];                  // @808
    public uint64_t incrementalComparisons[3];              // @832
    public uint64_t incrementalBlockComparisons[3];         // @856
    public uint64_t aggReportBlock[2];                      // @880
    public uint64_t actorReportBlock[2];                    // @896
    public uint     totalIncrementalComparisons[3];         // @912
    public uint     max_found_lost_pairs;                   // @924
    public uint     max_found_lost_agg_pairs;               // @928
    public uint     overflowFlagsWord;                      // @932  bool found_lost_pairs_overflow_flags in byte 0
    // struct size rounds to 936+pad -> 8-aligned (940 -> 944)
}

// handle helpers (PxgBroadPhaseDesc.h) shared by the bpDesc kernel family
public uint bpCreateHandle(uint handle, bool isStart, bool isNew)
{
    return (handle << 3) | (isStart ? 1u : 0u) | (isNew ? 2u : 0u);
}
public uint bpMarkDeleted(uint handle) { return handle | (1u << 2); }
public uint bpGetHandle(uint sortedHandle) { return sortedHandle >> 3; }
public bool bpIsStartProjection(uint sortedHandle) { return (sortedHandle & 1u) != 0; }
public bool bpIsNewProjection(uint sortedHandle) { return ((sortedHandle >> 1) & 1u) != 0; }
public bool bpIsDeletedProjection(uint sortedHandle) { return (sortedHandle & 4u) != 0; }
