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

// pxgSlangTriTriTail.slang -- P9.83b-8 T2.
//
// The mesh-mesh kernel's OUTPUT TAIL as one block function:
//
//   writePatchesAndContacts<32>          convexMeshMidphase.cu:673-751
//   createPatchWriterFullThreadBlock     :504-629 (called from :712)
//
// TEXTUAL INCLUDE, after pxgSlangTriTriBlock.slang.  It reads gTtbPatches, and a
// module cannot see its includer's Workgroup storage.  The includer must also
// `import pxgSlangTriTriPatch;` and `import pxgSlangNpAbi;` -- pxgSlangNpPatchTail
// imports the ABI types without re-exporting them.
//
// ---------------------------------------------------------------------------
// DESIGN A: THE WHOLE TAIL RUNS ON THREAD 0
// ---------------------------------------------------------------------------
// The CUDA spreads the tail over the block:
//
//   * a block scan of the patch point counts (:706, the PxU32 overload of
//     threadBlockScanExclusive, reduction.cuh:658-663 -> :634-656, with its own
//     __shared__ perWarpCountS[32] at :661 and a barrier at :645);
//   * patchStartIndexS[128] (:697), filled by threads < 128 (:707-708);
//   * three __shared__ byte offsets (:521-523), reserved by thread 0 (:550-620);
//   * two barriers publishing them (:532, :622);
//   * one header per thread < N (:726-734), one point per thread with
//     tid/6 < N and tid%6 < PointCount (:736-750).
//
// Everything the barriers publish is produced by thread 0 or feeds only the
// writes, so here thread 0 does all of it from registers: the certified serial
// scan (ttpPatchStartIndices, fixture ttrScan), the certified reservation
// (ttpReserveBlock, fixture ttrReserve), and the certified writer
// (ttpWritePatchAndContacts, fixture ttrWrite).  That costs NO Workgroup bytes
// (design C, CUDA-faithful, would add 652: 512 + 12 + the scan's 128), and the
// point write loops over the patches instead of mapping threads, which is the
// blockDim-independence the writer already documents (its note 2).
//
// ONE BARRIER, B0, AT ENTRY, AND THE CUDA HAS NO SUCH BARRIER.  gTtbPatches' last
// writes are already followed by barriers in the block functions (createNewPatches
// ends every pass with one, tryAssign ends with one), so B0 is redundant today.
// It is here so this function does not depend on what its caller did last -- the
// rule pxgSlangTriTriBlock.slang states for its own entry points.
//
// CALL IT UNCONDITIONALLY, from every thread, at kernel top level after the outer
// loop (cu:1224-1230) -- INCLUDING when generateContacts is false.  cu:1034 skips
// only the contact loop; the tail still writes eHAS_NO_TOUCH, both change flags,
// nbContacts = 0 and the base + 0xFFFFFFFF pointers for a pair with no overlap.
//
// ---------------------------------------------------------------------------
// FIVE THINGS A TRANSCRIPTION GETS WRONG HERE
// ---------------------------------------------------------------------------
// TT1  THE ATOMICS ARE GATED ON THE SCAN TOTAL, NOT ON N.  :554 tests
//      totalNumContacts.  A block with no patches reserves nothing and leaves all
//      three offsets at 0xFFFFFFFF WITHOUT overflow -- candidate 55's pointers.
//      (N > 0 with a total of 0 is the same arm, but the kernel cannot produce it:
//      createNewPatches keeps a patch only if its PointCount is nonzero, :955.)
//
// TT2  THE OVERFLOW ERROR IS AN OR, NOT A STORE.  setOverflowError is an
//      atomicOr (cudaNpCommon.h:103-110) of up to three bits, one per failed
//      check.  One InterlockedOr of their union is value-identical.  A store
//      would erase another block's bits.
//
// TT3  THE WRITE LOOPS USE THE CALLER'S N.  The overflow zeroing at :580-586
//      changes thread 0's BY-VALUE copy inside createPatchWriterFullThreadBlock;
//      :726 and :737 test the caller's numPatchesToKeep.  So an overflow of ONE
//      stream still writes the other: points with no headers, or headers that
//      point at nothing (the writer's note 3).  r.numPatchesOut is for the
//      published record only.
//
// TT4  nbContacts IS A PxU16 STORE.  :605 writes 2 bytes at offset 36; the
//      `flags` PxU16 at 38 must survive.  One read-modify-write of the word, as
//      pxgSlangNpPatchTail does, by the one thread that owns the record.
//
// TT5  THE WRITER TAKES THE STREAM BASES, THE RECORD TAKES THE START BASES.  The
//      published pointers are startContactPatches/Points/Forces + offset
//      (:609-611, push @104/@112/@120); the writer's members are patchStream and
//      contactStream + offset (:625-626, push @72/@64).  They are different
//      buffers in the engine, and a fixture that passes the same address for both
//      cannot tell them apart.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// THE SWEEP: 23 MUTANTS, 21 KILLED, 2 SURVIVE -- BOTH PROVABLY EQUIVALENT
// ---------------------------------------------------------------------------
// Fixture: npTriTriTail.slang:tttTail (tests/testNpTriTriTail.cpp), forced fresh.
//
//   tt2-or-ungated     `if (r.overflowBits != 0u)` dropped before the OR
//       EQUIVALENT: an atomic OR of zero leaves the word as it was.  The guard
//       saves an atomic, not a value.
//   counts-unguarded   `(i < numPatchesToKeep) ? ... : 0u` dropped
//       EQUIVALENT: ttpPatchStartIndices tests `i < numPatchesToKeep` itself
//       before it adds a count, so the fixture's poisoned records past N (point
//       count 5) never reach the total either way.  The guard is kept because
//       it is the CUDA's `count = 0` for threadIdx.x >= N (:699-701).
//
// Worth naming among the 21:
//   b0-removed         killed, because the fixture loads the patches with NO
//       barrier of its own -- thread 0 then reads records other threads have not
//       written.  In the kernel B0 is redundant; this is where its job shows.
//   tt1-gate-on-N      killed only by the function-level cases N > 0 with no
//       points, the one input that separates the CUDA's gate on the scan TOTAL.
//   tt5-writer-from-start / tt5-points-writer-from-start   killed through the
//       decoys: the start bases are real buffers, so writing through them is a
//       visible stray write, not undefined behaviour.
//   tt4-nbcontacts-whole-word   killed by the random high half of the word.

#ifndef PXG_SLANG_TRI_TRI_TAIL
#define PXG_SLANG_TRI_TRI_TAIL

struct TtbTailArgs
{
    uint64_t cmOutputs;            // push @40   PxsContactManagerOutput*
    uint64_t materials;            // push @56   PxsMaterialData*
    uint64_t contactStream;        // push @64   the writer's contact base (:626)
    uint64_t patchStream;          // push @72   the writer's patch base (:625)
    uint64_t counters;             // push @80   PxgPatchAndContactCounters*
    uint64_t touchChangeFlags;     // push @88   PxU32*
    uint64_t patchChangeFlags;     // push @96   PxU32*
    uint64_t startContactPatches;  // push @104  published (:610)
    uint64_t startContactPoints;   // push @112  published (:611)
    uint64_t startContactForces;   // push @120  published (:609)
    uint patchBytesLimit;          // push @128
    uint contactBytesLimit;        // push @132
    uint forceBytesLimit;          // push @136
}

// ALL threads of the block, block-uniformly.  numPatchesToKeep is the block-uniform
// count from the assignment.  localQ1/localP1 and the two material indices are the
// LOCAL transform1 and shapes -- the arguments as doTriangleTriangleCollision sees
// them after the flipped arm exchanged them (cu:1365); ttpWritePatchAndContacts
// undoes the material exchange itself (its note 4), so the caller must not.
void ttbWritePatchesAndContacts(uint tid, uint pairIndex, uint numPatchesToKeep,
                                float4 localQ1, float3 localP1, bool isFlipped,
                                uint localMaterialIndex0, uint localMaterialIndex1,
                                TtbTailArgs a)
{
    GroupMemoryBarrierWithGroupSync();                   // B0 (port-only, see above)

    if (tid == 0u)
    {
        // :699-706.  Entries at or past N are never written by the block functions,
        // so they are read as 0 -- the CUDA's `count = 0` for threadIdx.x >= N.
        uint counts[TTP_MAX_MESH_MESH_PATCHES];
        for (uint i = 0u; i < TTP_MAX_MESH_MESH_PATCHES; ++i)
            counts[i] = (i < numPatchesToKeep) ? gTtbPatches[i].pointCount : 0u;
        uint starts[TTP_MAX_MESH_MESH_PATCHES];
        const uint total = ttpPatchStartIndices(counts, numPatchesToKeep, starts);

        Ptr<PxgPatchAndContactCountersGpu> counters = Ptr<PxgPatchAndContactCountersGpu>(a.counters);
        Ptr<PxsContactManagerOutputGpu> cmOutputs = Ptr<PxsContactManagerOutputGpu>(a.cmOutputs);

        // :554-558 (TT1).  The adds are 32-bit and wrap; the checks inside
        // ttpReserveBlock are 64-bit.
        uint patchAdd = 0xFFFFFFFFu, contactAdd = 0xFFFFFFFFu, forceAdd = 0xFFFFFFFFu;
        if (total != 0u)
        {
            InterlockedAdd(counters[0].patchesBytes, TTP_PATCH_SIZE * numPatchesToKeep, patchAdd);
            InterlockedAdd(counters[0].contactsBytes, TTP_CONTACT_SIZE * total, contactAdd);
            InterlockedAdd(counters[0].forceAndIndiceBytes, TTP_FORCE_SIZE * total, forceAdd);
        }

        const TtpReservation r = ttpReserveBlock(cmOutputs[pairIndex].allflagsWord, numPatchesToKeep, total,
                                                 patchAdd, contactAdd, forceAdd,
                                                 a.patchBytesLimit, a.contactBytesLimit, a.forceBytesLimit);

        // :562/:569/:575 (TT2).
        if (r.overflowBits != 0u)
            InterlockedOr(counters[0].overflowError, r.overflowBits);

        // :598-600.
        Ptr<uint>(a.touchChangeFlags)[pairIndex] = r.touchChange;
        Ptr<uint>(a.patchChangeFlags)[pairIndex] = r.patchChange;

        // :603-605 (TT4).
        cmOutputs[pairIndex].allflagsWord = r.allFlagsWord;
        {
            const uint nf = cmOutputs[pairIndex].nbContactsFlags;
            cmOutputs[pairIndex].nbContactsFlags = (nf & 0xFFFF0000u) | (r.numContactsOut & 0xFFFFu);
        }

        // :607-618 (TT5).  The sentinel rides into the pointer on the zero-contact
        // path (candidate 55); only overflow writes zeros, and then all three.
        if (!r.overflow)
        {
            cmOutputs[pairIndex].contactForces = a.startContactForces + uint64_t(r.forceByteOffset);
            cmOutputs[pairIndex].contactPatches = a.startContactPatches + uint64_t(r.patchByteOffset);
            cmOutputs[pairIndex].contactPoints = a.startContactPoints + uint64_t(r.contactByteOffset);
        }
        else
        {
            cmOutputs[pairIndex].contactForces = 0uL;
            cmOutputs[pairIndex].contactPatches = 0uL;
            cmOutputs[pairIndex].contactPoints = 0uL;
        }

        // :726-750 (TT3): the caller's N, not r.numPatchesOut.
        for (uint p = 0u; p < numPatchesToKeep; ++p)
            ttpWritePatchAndContacts(a.patchStream, r.patchByteOffset,
                                     a.contactStream, r.contactByteOffset,
                                     a.materials, p, starts[p], gTtbPatches[p],
                                     localQ1, localP1, isFlipped,
                                     localMaterialIndex0, localMaterialIndex1);
    }
}

#endif // PXG_SLANG_TRI_TRI_TAIL
