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
// PxgRigidFilterPair (PxgSimulationCoreDesc.h, 32 B): index0 u64, index1 u64,
// index2 u32, padding[3].
// ---------------------------------------------------------------------------
public struct PsPairR { public uint2 index0; public uint2 index1; public uint index2; public uint pad0, pad1, pad2; }

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

public bool psFindRigid(uint64_t pairsAddr, uint nbPairs, uint2 q0, uint2 q1)
{
    Ptr<PsPairR> pairs = Ptr<PsPairR>(pairsAddr);
    uint l = 0u, r = nbPairs;
    while (l < r)
    {
        const uint mid = (l + r) / 2u;
        const PsPairR p = pairs[mid];
        int cmp = 0;
        if (psU64Lt(q0, p.index0)) cmp = -1;
        else if (psU64Lt(p.index0, q0)) cmp = 1;
        else if (psU64Lt(q1, p.index1)) cmp = -1;
        else if (psU64Lt(p.index1, q1)) cmp = 1;
        else if (0u < p.index2) cmp = -1;
        if (cmp == 0 || (psU64Eq(q0, p.index0) && psU64Eq(q1, p.index1)))
            return true;
        if (cmp < 0) r = mid; else l = mid + 1u;
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
