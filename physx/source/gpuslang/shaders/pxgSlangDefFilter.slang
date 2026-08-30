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

// ===========================================================================
// pxgSlangDefFilter — the deformable/rigid element filter, ONCE.
//
// `find(const PxgRigidFilterPair*, nbPairs, rigidId, compressedId)`
// (deformableElementFilter.cuh:89, over the :67 overload) is spelled NINE times
// across this corpus, and P9.34c added the ninth while saying so.  This module
// exists so P9.35c does not add a tenth: the softbody SDF midphase imports it
// instead of carrying its own.
//
// It is deliberately NOT a refactor of the other nine.  Converting eight live
// call sites is its own change with its own sweep -- some of them search
// PxParticleRigidFilterPair, a DIFFERENT 16-byte record, and merging the two
// searches is the failure mode that change has to avoid.  This is the
// destination for that work, not the work itself.
//
// WHAT THE TRANSCRIPTION HAS TO PRESERVE, both established in P9.34c:
//
//  F1. THE COMPARISON ORDER is index0, then index1, then index2, each as an
//      UNSIGNED 64-BIT value.  The pairs are sorted by that order and the search
//      is a binary one, so a comparison that treats the halves as two
//      independent 32-bit words walks the wrong way.
//
//  F2. THE `ignoreIndex2` DISJUNCT accepts a hit on (index0, index1) alone.
//      For the cloth-rigid and softbody-rigid filter arrays it is redundant --
//      they are built from default-constructed pairs, whose constructor is
//      `index2(0)` (PxgSimulationCoreDesc.h:164), and the query carries 0 too --
//      but the SAME overload is called elsewhere with attachment handles in
//      index2, where it decides the answer.  Transcribed because the CUDA has
//      it, and because this module is meant to serve those callers too.
//
// PxgRigidFilterPair is 32 bytes: PxU64 index0, PxU64 index1, PxU32 index2,
// PxU32 padding[3].
// ===========================================================================
module pxgSlangDefFilter;

public struct PxgRigidFilterPairRec
{
    public uint2 index0;
    public uint2 index1;
    public uint index2;
    public uint pad0, pad1, pad2;
}

// F1: an unsigned 64-bit comparison, high word first.
public bool dfU64Lt(uint2 a, uint2 b) { return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x); }
public bool dfU64Eq(uint2 a, uint2 b) { return a.x == b.x && a.y == b.y; }

// `find(pairs, nbPairs, rigidId, compressedId)` -- the query's index1 is a PxU32
// widened to PxU64, so its high word is zero, and index2 is zero.
public bool dfFindRigidPair(uint64_t pairsAddr, uint nbPairs, uint2 rigidId, uint compressedId)
{
    Ptr<PxgRigidFilterPairRec> pairs = Ptr<PxgRigidFilterPairRec>(pairsAddr);
    const uint2 q0 = rigidId;
    const uint2 q1 = uint2(compressedId, 0u);

    uint l = 0u, r = nbPairs;
    while (l < r)
    {
        const uint mid = (l + r) / 2u;
        const PxgRigidFilterPairRec p = pairs[mid];

        // PxgRigidFilterPair::compare, with the query on the left.
        int cmp = 0;
        if (dfU64Lt(q0, p.index0)) cmp = -1;
        else if (dfU64Lt(p.index0, q0)) cmp = 1;
        else if (dfU64Lt(q1, p.index1)) cmp = -1;
        else if (dfU64Lt(p.index1, q1)) cmp = 1;
        else if (0u < p.index2) cmp = -1;      // the query's index2 is 0

        // F2.
        if (cmp == 0 || (dfU64Eq(q0, p.index0) && dfU64Eq(q1, p.index1)))
            return true;

        if (cmp < 0) r = mid; else l = mid + 1u;
    }
    return false;
}
