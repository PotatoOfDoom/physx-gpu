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

// femClothMeshVertContactGenBda.slang -- P9.26
//
// cloth_meshVertexContactGenLaunch (femClothPrimitives.cu:2910), read in full.
//
// ===========================================================================
// THIS KERNEL EMITS NO CONTACTS, AND THAT IS THE ENGINE'S DOING, NOT THIS PORT'S
// ===========================================================================
//
// The CUDA computes a closest point, a normal, a penetration and a barycentric
// for every (cloth vertex, mesh triangle) pair the midphase produced -- and then
// throws all of it away.  The write is gated on `hasContacts`, and the ONLY
// assignment to `hasContacts` in the whole kernel is commented out:
//
//     bool hasContacts = false;                       // :2938
//     ...
//         if ((pen <= cDistance) && normal.dot(n) > 0.f)
//         {
//             pairInd0 = rigidId.getInd();
//             pairInd1 = PxEncodeClothIndex(clothId, clothVertIndex);
//             contact  = make_float4(closestPt.x, closestPt.y, closestPt.z,
//                                    restDistance[cmIdx]);
//             //hasContacts = true;                   // :3056  <-- COMMENTED OUT
//         }
//     }
//     PxgOutputIndex indexer(__ballot_sync(FULL_MASK, hasContacts),
//                            writer.totalContactCount);
//     if (hasContacts) { ... writeRigidVsDeformableContact32(...) }
//
// PxgOutputIndex (:79) only performs its atomicAdd when the popcount is non-zero,
// so with an always-empty ballot the contact COUNTER does not move either.  The
// kernel's entire observable output is the one atomicMax at the end:
//
//     atomicMax(stackSizeNeededOnDevice, numMidphasePairsFound * sizeof(uint4));
//
// Upstream candidate 33: a contact-generation kernel disabled by a commented-out
// line, while its producer midphase, its argument pack and its launch site all
// remain live and run every step.  Documented, NOT reported.
//
// ---------------------------------------------------------------------------
// WHY THE GEOMETRY IS NOT TRANSCRIBED HERE, stated rather than left to be noticed
// ---------------------------------------------------------------------------
// The closest-point chain (Gu::closestPtPointTriangle2, computeBarycentric, the
// vertex-space round trip, the `normal.dot(n) > 0` sidedness gate) is DEAD CODE
// in the CUDA: nothing it computes can reach an output.  Transcribing it would
// add roughly two hundred lines that NO fixture in this port could verify --
// the only observable is "no contacts", which every implementation of the dead
// code satisfies equally.  Untestable code that looks tested is worse than
// absent code that says it is absent.
//
// So this port implements the kernel's observable behaviour exactly, and the
// risk that upstream re-enables the line is handled where it can actually be
// caught: tools/check_disabled_kernels.py re-derives the commented state from
// the vendored CUDA on every ctest run and fails the moment it changes.  That is
// the same device P9.20 used for the twin kernels -- a fact about someone else's
// source that this port depends on gets a check, not a comment.
//
// ---------------------------------------------------------------------------
// THE LOOP IS NOT HERE EITHER, AND FOR THE SAME REASON
// ---------------------------------------------------------------------------
// The CUDA grid-strides `i += gridDim.x * blockDim.x` = 131072 over
// `paddedNumPairs = (numPairs + 31) & ~31`, where `numPairs` is
// min(numMidphasePairsFound, stackSizeBytes / sizeof(uint4)); the engine launches
// (512,1,1) x (256,1,1) at PxgNarrowphaseCore.cpp:6831.  An earlier draft of this
// file carried that loop as an empty walk "so the shape matches".
//
// The P9.26 sweep is what removed it.  `stackSizeBytes / 16` -> `/ 4` SURVIVED --
// of course it did: the clamp feeds only the bound of a loop with no body, so
// nothing downstream could tell.  That is the identical objection the header
// above raises against transcribing the geometry, and keeping four lines of it
// because they are short would have been the same mistake in miniature.  The
// shape of the loop is described here, where prose belongs; what remains in the
// code is what produces output.
//
// When the geometry comes back, so do the loop, the clamp and their mutants.
// ===========================================================================
module femClothMeshVertContactGenBda;

// ---------------------------------------------------------------------------
// Push-constant blob -- BYTE-IDENTICAL to the CUDA pack
// (PxgNarrowphaseCore.cpp:6814): { PxReal toleranceLength, cmInputs*,
// transformCache*, contactDistance*, restDistance*, gpuShapes*, femClothes*,
// shapeToRigidRemapTable*, PxU32 stackSizeBytes, stackPtr*, midphasePairsNum*,
// stackSizeNeededOnDevice*, PxgFEMContactWriter writer (by value, 80 B) }.
//
// The writer lands at @96, and that is 16-aligned already, so this pack does NOT
// hit the paramAlign(80) == 16 trap P9.15e had to find -- the arguments before it
// happen to end on a 16-byte boundary.  Recorded because the next kernel with a
// by-value tail will have to check it again.
// ABI row {4,8,8,8,8,8,8,8,4,8,8,8,80}.
// ---------------------------------------------------------------------------
struct ClothMeshVertCgParams
{
    float toleranceLength;         // @0  (ABI parity only)
    uint _pad4;                    // @4
    uint64_t cmInputs;             // @8   (ABI parity only -- see the note above)
    uint64_t transformCache;       // @16  (ABI parity only)
    uint64_t contactDistance;      // @24  (ABI parity only)
    uint64_t restDistance;         // @32  (ABI parity only)
    uint64_t gpuShapes;            // @40  (ABI parity only)
    uint64_t femClothes;           // @48  (ABI parity only)
    uint64_t shapeToRigidRemap;    // @56  (ABI parity only)
    uint stackSizeBytes;           // @64
    uint _pad68;                   // @68
    uint64_t stackPtr;             // @72  (ABI parity only)
    uint64_t midphasePairsNum;     // @80
    uint64_t stackSizeNeeded;      // @88  atomicMax output
    // ---- PxgFEMContactWriter (by value, 80 B) @96 ----
    uint64_t w_outPoint;           // @96   (ABI parity only)
    uint64_t w_outNormalPen;       // @104  (ABI parity only)
    uint64_t w_outBarycentric;     // @112  (ABI parity only)
    uint64_t w_outContactInfo;     // @120  (ABI parity only)
    uint64_t w_totalContactCount;  // @128  (never incremented -- see the header)
    uint64_t w_contactByX;         // @136  (ABI parity only)
    uint64_t w_tempContactByX;     // @144  (ABI parity only)
    uint64_t w_contactIndexSortedByX; // @152 (ABI parity only)
    uint64_t w_contactSortedByX;   // @160  (ABI parity only)
    uint w_maxNumContacts;         // @168
    uint _pad172;                  // @172 -> 176
}

[[vk::push_constant]] ConstantBuffer<ClothMeshVertCgParams> gVc;

// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void cloth_meshVertexContactGenLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    // The kernel's ONE observable output: the UNCLAMPED pair count, in bytes, by
    // one thread of the grid.  This is how the engine sizes next frame's
    // intermediate stack, so it is not cosmetic even though no contact is --
    // clamping it to the stack that already exists would mean it could never ask
    // for a bigger one.
    //
    // The `globalThreadIdx == 0` guard is transcribed from the CUDA and is a
    // CONTENTION measure, not a semantic one: the argument depends on neither the
    // block nor the lane, and atomicMax with a uniform value is idempotent.  All
    // three sweep mutants that widen the guard therefore survive, and are recorded
    // as one equivalence rather than chased.
    const uint numFound = Ptr<uint>(gVc.midphasePairsNum)[0];
    if (gid.x == 0u && gtid.x == 0u)
        InterlockedMax(Ptr<uint>(gVc.stackSizeNeeded)[0], numFound * 16u);
}
