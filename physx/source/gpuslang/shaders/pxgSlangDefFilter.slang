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
// (deformableElementFilter.cuh:89, over the :67 overload) was spelled NINE times
// across this corpus when P9.34c added the ninth and said so.  P9.35c created this
// module so it would not become a tenth, deliberately leaving the conversion of the
// others to its own change.
//
// P9.38 IS THAT CHANGE.  Every BDA-addressed copy now calls in here: seven callers on
// dfFindRigidPair and four on dfFindNonRigidPair, all eleven compiling to one
// arithmetic DAG apiece (gpuslang.helperUniformity checks it).
//
// Before converting anything, all fifteen searches in the corpus were read against
// deformableElementFilter.cuh and PxgRigidFilterPair::compare, and every one agreed
// on the comparison order, the 64-bit keys and the ignoreIndex2 disjunct.  That is
// the reason to collapse them rather than a reason not to: fifteen identical
// transcriptions are fifteen future chances to re-derive one of them wrong, and the
// two preceding tranches each ended on exactly that.  One difference did turn up --
// see the non-rigid section below for the two algorithms.
//
// WHAT DID NOT MOVE, and why it cannot:
//   * the searches that read their pairs through a StructuredBuffer BINDING rather
//     than a device address -- npSbSb, npSbSelf, npSbPrim, npClothCloth,
//     fcStaticTriCloth, fcVertexConvex, psPrimCollide.  A module taking a uint64_t
//     address cannot be handed a binding, and giving them one would mean changing
//     how those kernels bind their filter arrays: a fixture-and-ABI change with no
//     behavioural gain.  They stay, and they are listed here so the next reader knows
//     the omission is a decision rather than an oversight;
//   * npDefFilter.slang, which is the FIXTURE shader for these searches -- the
//     reference implementation testNpDefFilter.cpp drives.  It reads through bindings
//     for the same reason;
//   * findParticleRigid over PxParticleRigidFilterPair, a DIFFERENT 16-byte record
//     with its own operator== / operator< navigation, and npSbPrim's
//     findParticleFilter, which passes ignoreIndex2 = FALSE with a bucket's mUniqueId
//     as the third key.  Merging those into these searches is the failure mode this
//     work had to avoid.
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
// ---------------------------------------------------------------------------
// TWO THINGS HERE ARE UNTESTABLE, and both were established by PROOF after a mutation
// sweep left them standing, rather than by widening a rig.  They were written for the
// copy in femClothSdfMeshCgBda; P9.38 moved them here with the code they describe.
//
//   * `l = mid + 1` against `l = l + 1` -- the second still RETURNS the same answer.
//     The invariant `l <= k < r` (k = the target's index) survives both updates:
//     cmp > 0 means pairs[mid] < query, so k > mid >= l and therefore k >= l + 1.  The
//     loop only exits at l >= r, which contradicts the invariant, so a present key is
//     always found; and cmp == 0 is a real match, so an absent one is never claimed.
//     The two differ in WORK -- O(n) against O(log n) -- and in nothing else.  A
//     fixture cannot separate them; only this argument can.
//
//   * the `ignoreIndex2` disjunct is redundant FOR THE CLOTH-RIGID ARRAY.  That list
//     is built at PxgSimulationController.cpp:4094 from a default-constructed
//     PxgRigidFilterPair, whose constructor is `index2(0)`
//     (PxgSimulationCoreDesc.h:164), and nothing else writes index2 into
//     mClothRigidFilterPairs.  The query carries index2 = 0 too, so `compare` never
//     reaches its index2 branch with a difference and `cmp == 0` already holds.  The
//     clause is transcribed because the CUDA has it and because the SAME overload is
//     called elsewhere WITH attachment handles in index2, where it decides the answer.
//     That is the reason it cannot be dropped from a shared implementation even though
//     one caller's data makes it dead.
// ---------------------------------------------------------------------------
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


// ===========================================================================
// The NON-RIGID record.  PxgNonRigidFilterPair (PxgSimulationCoreDesc.h:235) is 16
// bytes -- PxU64 index0, PxU32 index1, PxU32 index2 -- and its find()
// (deformableElementFilter.cuh:130) is the same probe-and-accept loop as the rigid
// one over its own compare().
//
// P9.38 FOUND TWO ALGORITHMS HERE, and that is why this exists.  Of the four
// BDA-addressed non-rigid searches in the corpus, softBodySelfCollisionBda and
// softBodySoftBodyMidphaseBda transcribed the engine's loop, while
// softBodyClothContactGenBda and softBodyClothVertContactGenBda wrote a LOWER-BOUND
// search instead -- narrow to the first element >= (index0, index1), then test that
// one element.  Both give the right answer, and the reason is worth writing down
// rather than trusting: entries sharing an (index0, index1) form a CONTIGUOUS run
// because index2 is the last sort key, the engine's loop keeps that run inside
// [l, r) at every step, and a loop that only exits at l == r must therefore probe
// into it -- where the ignoreIndex2 disjunct fires.  Equivalent, but two algorithms
// where the engine has one is a maintenance hazard, so both are gone.
//
// THE NULL GUARD IS NOT HERE, DELIBERATELY.  mFilteringPairs is null whenever a body
// registered no filter pairs, which is the common case, and on llvmpipe a zero-trip
// `while (l < r)` still speculates the first iteration's load and faults at address
// 0 -- with per-lane garbage that breaks warp uniformity and then a workgroup
// barrier.  The fix is to select a non-null POINTER, and only the caller knows which
// of its own buffers is safe to point at, so each caller keeps that choice and hands
// this function an address it has already made safe together with r = 0.  The full
// diagnosis is in softBodySelfCollisionBda.slang.
public static const uint64_t DF_NRPAIR_STRIDE = 16UL;

public bool dfFindNonRigidPair(uint64_t pairsAddr, uint nbPairs, uint id0, uint id1)
{
    const uint2 q0 = uint2(id0, 0u);        // the query's index0 is a widened PxU32
    uint l = 0u, r = nbPairs;
    while (l < r)
    {
        const uint mid = (l + r) / 2u;
        const uint64_t e = pairsAddr + uint64_t(mid) * DF_NRPAIR_STRIDE;
        const uint2 pIndex0 = uint2(Ptr<uint>(e)[0], Ptr<uint>(e + 4UL)[0]);
        const uint pIndex1 = Ptr<uint>(e + 8UL)[0];
        const uint pIndex2 = Ptr<uint>(e + 12UL)[0];

        // PxgNonRigidFilterPair::compare, with the query's index2 fixed at 0.
        int cmp = 0;
        if (dfU64Lt(q0, pIndex0)) cmp = -1;
        else if (dfU64Lt(pIndex0, q0)) cmp = 1;
        else if (id1 < pIndex1) cmp = -1;
        else if (id1 > pIndex1) cmp = 1;
        else if (0u < pIndex2) cmp = -1;

        // F2 again: a hit on (index0, index1) alone is accepted.
        if (cmp == 0 || (dfU64Eq(q0, pIndex0) && id1 == pIndex1))
            return true;

        if (cmp < 0) r = mid; else l = mid + 1u;
    }
    return false;
}
