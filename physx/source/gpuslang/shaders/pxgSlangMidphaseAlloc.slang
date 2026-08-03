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
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS \'\'AS IS\'\' AND ANY
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
//
// ===========================================================================
// pxgSlangMidphaseAlloc — midphaseAllocate (midphaseAllocate.cuh:129), once.
//
// The convex-vs-trimesh chain does not receive its seven intermediate buffers.
// `convexTrimeshNarrowphase` (convexMesh.cu:670) CARVES THEM OUT of the single
// flat collision stack at runtime and publishes the addresses back through
// double pointers, and the five kernels after it re-derive the same layout.  So
// this arithmetic is a shared contract, not a private detail of one kernel —
// extracted first, and on its own, for the same reason P8.23 extracted the PCM
// gate: everything downstream inherits whatever it gets wrong.
//
// Element sizes, verified against convexNpCommon.h:
//   uint4                       16      the midphase's pair records
//   ConvexTriNormalAndIndex     16      PxVec3 normal + int index
//   ConvexTriContacts            4      a single PxU32 index into the temp buffer
//   PxU32 (ordered interm)       4
//   PxReal (max depth)           4
//   ConvexTriIntermediateData    4      a single PxU32 gpuTriIndex
//   PxU32 (second-pass flags)    4
//                        total  56  ==  calculateMaxPairs' memRequiredPerPair
//
// TWO THINGS THAT ONLY LITERAL TRANSCRIPTION KEEPS:
//
//   1. THE ALLOCATION ORDER IS NOT THE ARGUMENT ORDER.  The signature lists
//      NI, Contacts, MaxDepth, Interm, Ordered, SecondPass, Pairs; the body
//      allocates Pairs, NI, Contacts, ORDERED, MaxDepth, Interm, SecondPass.
//      `ordered` moves from fifth in the signature to fourth in memory, and
//      `maxDepth` and `interm` shift down behind it.  Following the signature
//      instead of the body puts three of the seven buffers at wrong addresses —
//      and every one of them would still be inside the stack allocation, so the
//      corruption reads as bad contacts rather than a fault.
//
//   2. `ordered` IS SIZED BY THE OTHER COUNTER.  Six of the seven use
//      `numPairsAfterMP` = (unpadded + 3) & ~3.  The ordered array alone uses
//      `numPaddedPairsAfterMP`, the separately-accumulated counter the midphase
//      bumps with `((nbPairsPerCM + 3) & ~3) * 2` — already padded AND doubled,
//      because the radix sort needs its temp space inline.  That is why
//      `midphasePairsNumPadded` exists as a second atomic at all.
// ===========================================================================

module pxgSlangMidphaseAlloc;

public static const uint64_t MA_PAIR_BYTES        = 16UL;  // uint4
public static const uint64_t MA_NI_BYTES          = 16UL;  // ConvexTriNormalAndIndex
public static const uint64_t MA_CONTACTS_BYTES    =  4UL;  // ConvexTriContacts
public static const uint64_t MA_ORDERED_BYTES     =  4UL;  // PxU32
public static const uint64_t MA_MAXDEPTH_BYTES    =  4UL;  // PxReal
public static const uint64_t MA_INTERM_BYTES      =  4UL;  // ConvexTriIntermediateData
public static const uint64_t MA_SECONDPASS_BYTES  =  4UL;  // PxU32

// The seven device addresses, in ALLOCATION order.
public struct MidphaseAlloc
{
    public uint64_t pairs;
    public uint64_t ni;
    public uint64_t contacts;
    public uint64_t ordered;
    public uint64_t maxDepth;
    public uint64_t interm;
    public uint64_t secondPass;
}

// mirrors midphaseAllocate(): pure address arithmetic, no loads, no stores.
public MidphaseAlloc maCompute(uint64_t stackPtr, uint numPairsUnpadded, uint numPaddedPairs)
{
    // "Pad up to a multiple of 4 so we don't get misaligned addresses"
    const uint64_t n = uint64_t((numPairsUnpadded + 3u) & ~3u);
    const uint64_t nPadded = uint64_t(numPaddedPairs);

    MidphaseAlloc a;
    uint64_t p = stackPtr;
    a.pairs      = p; p += n * MA_PAIR_BYTES;
    a.ni         = p; p += n * MA_NI_BYTES;
    a.contacts   = p; p += n * MA_CONTACTS_BYTES;
    a.ordered    = p; p += nPadded * MA_ORDERED_BYTES;   // the OTHER counter
    a.maxDepth   = p; p += n * MA_MAXDEPTH_BYTES;
    a.interm     = p; p += n * MA_INTERM_BYTES;
    a.secondPass = p;
    return a;
}
