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

// pxgSlangParticleContact.slang -- P9.12g, the plumbing every particle-vs-shape
// midphase needs and that three of them had grown their own copy of.
//
// WHY THIS EXISTS NOW.  psMeshMidphaseBda's own comment (P9.10) said it:
//
//     "This is a THIRD copy in the corpus ... Extracting all three is its own
//      tranche; duplicating a fourth time would not be."
//
// ps_sdfMeshCollisonLaunch is the fourth caller, and it needs BOTH the contact
// writer and the rigid element filter, so it would have made copy four of the
// writer and copy two of the filter.  This is that tranche.
//
// WHAT MOVED, and from where:
//   PxgParticleContactWriter::writeContact  psMeshMidphaseBda, psHfMidphaseBda
//   the rigid element filter (find/findRange/findRigid)   psMeshMidphaseBda
//   LoadShapePair<A, B>                     psMeshMidphaseBda, psHfMidphaseBda
//   PxgParticlePrimitiveContact             both
//   PxgContactManagerInput                  both
//
// WHAT DID NOT MOVE, deliberately: psContactGenBda and femClothContactGenBda
// carry their own filter variants over PRIVATE query structs, and neither is a
// module.  Folding those in is a change to two shaders this tranche has no
// fixture for; they are left alone and named here so the next person knows the
// extraction is still partial.
//
// THE ONE GENERALISATION.  LoadShapePair is a template on TWO geometry types in
// the CUDA, and the two callers do not agree on which one it tests: the mesh
// midphase tests the TRIANGLE MESH (particleSystemMeshMidphase.cu:403) while the
// heightfield midphase tests the PARTICLE SYSTEM
// (particleSystemHFMidPhaseCG.cu:293).  Rather than pick one and bury the
// difference, psLoadShapePair takes the tested type as an argument and returns
// the matching half first.  The choice therefore stays visible at each call
// site, which is where it is a trap.

import pxgSlangCommon;
import pxgSlangParticleDesc;

// PxgContactManagerInput, gpunarrowphase/include/PxgContactManager.h (16 B).
public struct PxgContactManagerInput
{
    public uint shapeRef0, shapeRef1, transformCacheRef0, transformCacheRef1;
}

// PxgParticlePrimitiveContact (32 bytes).
public struct PsContact
{
    public float4 normal_pen;
    public uint2 rigidId;
    public uint2 particleId;
}

// PxgParticleContactWriter (PxgParticleSystem.h:604-635): eight pointers and a
// capacity, passed to the kernel BY VALUE.  Held as a struct here because a
// module cannot reach into its importer's push-constant block -- which is
// exactly what the three private copies were doing.
public struct PsContactWriter
{
    public uint64_t particleContacts;
    public uint64_t numTotalContacts;
    public uint64_t contactSortedByParticle;
    public uint64_t tempContactByParticle;
    public uint64_t contactIndexSortedByParticle;
    public uint64_t contactByRigid;
    public uint64_t tempContactByRigid;
    public uint64_t contactIndexSortedByRigid;
    public uint maxContacts;
}

// PxgParticleContactWriter::writeContact (PxgParticleSystem.h:616-634).
// The capacity test is the CUDA's: over capacity the contact is silently
// DROPPED, and the caller's index -- which came from a global atomic -- is
// simply not used.
public void psWriteContact(PsContactWriter w, uint index, float4 normalPen,
                           uint2 compressedParticleIndex, uint2 rigidId)
{
    if (index < w.maxContacts)
    {
        (Ptr<uint2>(w.contactByRigid))[index] = rigidId;
        // PxU32(rigidId.getInd() & 0xffffffff) -- the LOW word of the node index.
        (Ptr<uint>(w.tempContactByRigid))[index] = rigidId.x;
        (Ptr<uint>(w.contactIndexSortedByRigid))[index] = index;

        (Ptr<uint2>(w.contactSortedByParticle))[index] = compressedParticleIndex;
        // PxGetParticleIndex(compressed) -- also the low word.
        (Ptr<uint>(w.tempContactByParticle))[index] = compressedParticleIndex.x;
        (Ptr<uint>(w.contactIndexSortedByParticle))[index] = index;

        PsContact ct;
        ct.normal_pen = normalPen;
        ct.rigidId = rigidId;
        ct.particleId = compressedParticleIndex;
        (Ptr<PsContact>(w.particleContacts))[index] = ct;
    }
}

// ---------------------------------------------------------------------------
// The rigid filter list, deformableElementFilter.cuh:118.
//
// THE STRUCT IS **PxParticleRigidFilterPair** (PxFiltering.h:740, 16 BYTES):
// two PxU64 and nothing else.  NOT PxgRigidFilterPair (PxgSimulationCoreDesc.h,
// 32 B with an index2 and three padding words) -- that one belongs to the
// softbody/cloth attachment paths.
//
// deformableElementFilter.cuh carries BOTH families and an overload for each,
// and `find(PxgParticleSystem&, PxU64, PxU64)` (:118) hands
// `PxgParticleSimBuffer::mFilterPairs` -- a `PxParticleRigidFilterPair*` -- to
// the 16-byte overload at :39, which compares with the struct's own
// `operator==` / `operator<`, i.e. over (mID0, mID1) ONLY.  There is no index2
// on this path and no `ignoreIndex2` argument.
//
// P9.10 got this wrong in both directions at once: the shader read a 32-byte
// stride and its fixture WROTE one, so the two agreed and the sweep had nothing
// to find -- the P8.49a failure mode exactly.  P9.12h's fixture builds the
// filter list out of the engine's own `PxParticleRigidFilterPair`, which is
// what exposed it.  A 32-byte reader over a 16-byte list picks up every other
// entry and runs off the end of the allocation.
// ---------------------------------------------------------------------------
public struct PsPairR { public uint2 mID0; public uint2 mID1; }

public bool psU64Lt(uint2 a, uint2 b) { return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x); }
public bool psU64Eq(uint2 a, uint2 b) { return a.x == b.x && a.y == b.y; }

public uint psFindRange(Ptr<uint> values, uint length, uint value)
{
    if (length == 0u) return 0u;
    uint l = 0u, r = length;
    while (l < r)
    {
        const uint m = (l + r) / 2u;
        if (values[m] > value) r = m; else l = m + 1u;
    }
    return r - 1u;
}

// deformableElementFilter.cuh:39.  `pair == p` first, then `pair < p` -- and
// `operator<` (PxFiltering.h:749) is NOT a total order's strict less-than
// spelled the usual way: it is
//     mID0 < o.mID0 || (mID0 == o.mID0 && mID1 < o.mID1)
// which for an equal-mID0 pair falls through to the mID1 test.  Transcribed as
// written rather than as `psU64Lt(q0,p.mID0) || (eq && psU64Lt(q1,p.mID1))`
// collapsed into a comparator, because the equality test above it already
// consumed the `==` case.
public bool psFindRigid(uint64_t pairsAddr, uint nbPairs, uint2 q0, uint2 q1)
{
    Ptr<PsPairR> pairs = Ptr<PsPairR>(pairsAddr);
    uint l = 0u, r = nbPairs;
    while (l < r)
    {
        const uint mid = (l + r) / 2u;
        const PsPairR p = pairs[mid];
        if (psU64Eq(q0, p.mID0) && psU64Eq(q1, p.mID1))
            return true;
        const bool lt = psU64Lt(q0, p.mID0) || (psU64Eq(q0, p.mID0) && psU64Lt(q1, p.mID1));
        if (lt) r = mid; else l = mid + 1u;
    }
    return false;
}

public bool psFindFilter(Ptr<PxgParticleSystemGpu> systems, uint psId, uint2 rigidId, uint2 particleMask)
{
    const uint particleId = particleMask.x;   // PxGetParticleIndex = the low word
    const uint numBuffers = systems[psId].mCommonData_NumParticleBuffers;
    Ptr<uint> runsum = Ptr<uint>(systems[psId].mParticleBufferRunsum);
    const uint id = psFindRange(runsum, numBuffers, particleId);
    Ptr<PxgParticleSimBufferGpu> simBufs = Ptr<PxgParticleSimBufferGpu>(systems[psId].mParticleSimBuffers);
    const uint64_t pairsAddr = simBufs[id].mFilterPairs;
    const uint nbPairs = simBufs[id].mNumFilterPairs;
    const uint localId = particleId - runsum[id];
    return psFindRigid(pairsAddr, nbPairs, rigidId, uint2(localId, 0u));
}

// ---------------------------------------------------------------------------
// LoadShapePair<A, B>.  `testedType` is A: the pair's slot 0 is checked against
// it, and whichever half matches comes back FIRST.  See the note at the head of
// this file about why that is a parameter and not a convention.
// ---------------------------------------------------------------------------
public void psLoadShapePair(uint64_t cmInputs, uint64_t gpuShapes, uint cmIdx, uint testedType,
                            out PxgShape matchShape, out uint matchCacheRef,
                            out PxgShape otherShape, out uint otherCacheRef)
{
    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(cmInputs)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gpuShapes);
    uint sRefA = cin.shapeRef0, cRefA = cin.transformCacheRef0;
    uint sRefB = cin.shapeRef1, cRefB = cin.transformCacheRef1;
    if (shapes[cin.shapeRef0].type != testedType)
    {
        sRefA = cin.shapeRef1; cRefA = cin.transformCacheRef1;
        sRefB = cin.shapeRef0; cRefB = cin.transformCacheRef0;
    }
    matchShape = shapes[sRefA]; matchCacheRef = cRefA;
    otherShape = shapes[sRefB]; otherCacheRef = cRefB;
}
