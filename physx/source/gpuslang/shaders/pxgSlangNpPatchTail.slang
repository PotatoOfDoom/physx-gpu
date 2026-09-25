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

// pxgSlangNpPatchTail.slang -- P9.82.
//
// registerContactPatch and insertIntoPatchStream (contactPatchUtils.cuh:76 and
// :139) on the engine's BDA ABI -- the output tail every in-kernel rigid
// narrowphase shares.
//
// WORD 13 IS HALF A STORE (P9.87).  insertIntoPatchStream assigns the header
// FIELD BY FIELD through a PxContactPatch* into the stream (:161-178): words 0-12
// whole, materialIndex1 a PxU16 store (:171, bytes 52-53), and pad[0..4] (bytes
// 54-63) never.  This tail stored word 13 whole and so zeroed pad[0]; its
// fixture could not tell, because the stream was zero-filled and word 13 was
// compared through a 16-bit mask.  Word 13 is now a read-modify-write and words
// 14-15 stay unwritten; testConvexPlaneNphase poisons the stream per word and
// compares all sixteen.
//
// WHY A MODULE.  Five CUDA kernels end with this exact pair: boxBoxNphase,
// sphereNphase, convexPlaneNphase, trimeshPlaneNarrowphase and
// trimeshHeightfieldNarrowphase.  npCvxPlaneNphase.slang carried a working BDA
// transcription of it inline, verified by PXG_SCENE=convexplane; P9.82 needed the
// same tail for triPlaneNpBda, and writing a second copy is how two transcriptions
// of one body start drifting (the P9.15d lesson).  So the tail was MOVED here and
// npCvxPlaneNphase.slang now calls it -- the bodies below are that file's, not a
// retyping, and its scene is what certifies them.
//
// THE THREE THINGS THAT ARE EASY TO GET WRONG, all preserved:
//
//   * THE OVERFLOW TEST IS ON ONE RECORD, NOT THE WHOLE RESERVATION.  Upstream
//     compares `offset + sizeof(PxContact)` against the limit, not
//     `offset + sizeof(PxContact) * nbContacts`.  It is a weaker test than it
//     looks, and it is upstream's.
//   * A PATCH OVERFLOW REWRITES THE STATUS WORD THAT WAS ALREADY WRITTEN.  When
//     the patch reservation fails the code goes back and clears the touch flag,
//     zeroes the contact count and re-derives BOTH change flags -- so the pair
//     ends up reported as "no touch", not as "touch with a lost patch".
//   * THE PACKED STATUS WORD IS FOUR BYTES IN ONE UINT: prevPatches << 24,
//     statusFlags << 16, numPatches << 8, allflagsStart in byte 0 -- and the
//     success path writes numPatches while the overflow path writes the word
//     WITHOUT it, which is how byte 1 gets cleared.

module pxgSlangNpPatchTail;

import pxgSlangNpAbi;

public static const uint NPT_INVALID = 0xFFFFFFFFu;

// PxsContactManagerStatusFlag
public static const uint NPT_FLAG_HAS_NO_TOUCH = 1u;
public static const uint NPT_FLAG_HAS_TOUCH = 2u;
public static const uint NPT_FLAG_TOUCH_KNOWN = 3u;

// PxgPatchAndContactCounters::OverflowError
public static const uint NPT_OVERFLOW_CONTACT = 1u;
public static const uint NPT_OVERFLOW_FORCE = 2u;
public static const uint NPT_OVERFLOW_PATCH = 4u;

public static const uint NPT_CONTACT_SIZE = 16u; // sizeof(PxContact)
public static const uint NPT_FORCE_SIZE = 4u;    // sizeof(PxU32)
public static const uint NPT_PATCH_SIZE = 64u;   // sizeof(PxContactPatch)

// PxCombineMode / PxMaterialFlag
public static const uint NPT_MODE_AVERAGE = 0u, NPT_MODE_MIN = 1u,
                         NPT_MODE_MULTIPLY = 2u, NPT_MODE_MAX = 3u;
public static const uint NPT_MAT_DISABLE_FRICTION = 1u,
                         NPT_MAT_DISABLE_STRONG_FRICTION = 2u,
                         NPT_MAT_COMPLIANT_ACC_SPRING = 16u;

// PxsCombineMaterials scalar combine (npInsertPatch combineMaterials).
public float nptCombineScalars(float a, float b, uint mode)
{
    if (mode == NPT_MODE_AVERAGE) return 0.5f * (a + b);
    if (mode == NPT_MODE_MIN) return min(a, b);
    if (mode == NPT_MODE_MULTIPLY) return a * b;
    if (mode == NPT_MODE_MAX) return max(a, b);
    return 0.0f;
}

// The CONTACT/FORCE reservation and the cmOutputs pointer write-back.
// Returns the contact byte offset, or NPT_INVALID when nothing was reserved.
public uint nptReserveContacts(uint64_t countersAddr, uint64_t cmOutputsAddr,
                               uint64_t startContactPoints, uint64_t startContactForces,
                               uint contactBytesLimit, uint forceBytesLimit,
                               uint workIndex, uint nbContacts, out uint forceByteOffset)
{
    Ptr<PxgPatchAndContactCountersGpu> counters = Ptr<PxgPatchAndContactCountersGpu>(countersAddr);
    Ptr<PxsContactManagerOutputGpu> cmOutputs = Ptr<PxsContactManagerOutputGpu>(cmOutputsAddr);

    uint contactByteOffset = NPT_INVALID;
    forceByteOffset = NPT_INVALID;
    if (nbContacts != 0u)
    {
        InterlockedAdd(counters[0].contactsBytes, NPT_CONTACT_SIZE * nbContacts, contactByteOffset);
        InterlockedAdd(counters[0].forceAndIndiceBytes, NPT_FORCE_SIZE * nbContacts, forceByteOffset);
        // ONE RECORD, not the whole reservation -- see the header.
        if ((contactByteOffset + NPT_CONTACT_SIZE) > contactBytesLimit)
        {
            InterlockedOr(counters[0].overflowError, NPT_OVERFLOW_CONTACT);
            contactByteOffset = NPT_INVALID;
        }
        if ((forceByteOffset + NPT_FORCE_SIZE) > forceBytesLimit)
        {
            InterlockedOr(counters[0].overflowError, NPT_OVERFLOW_FORCE);
            forceByteOffset = NPT_INVALID;
        }
    }

    if (contactByteOffset != NPT_INVALID)
    {
        cmOutputs[workIndex].contactForces = startContactForces + uint64_t(forceByteOffset);
        cmOutputs[workIndex].contactPoints = startContactPoints + uint64_t(contactByteOffset);
    }
    else
    {
        cmOutputs[workIndex].contactForces = 0uL;
        cmOutputs[workIndex].contactPoints = 0uL;
    }
    return contactByteOffset;
}

// registerContactPatch: the status word, the two change flags and the PATCH
// reservation.  Returns the patch byte offset, or NPT_INVALID.
public uint nptRegisterContactPatch(uint64_t countersAddr, uint64_t cmOutputsAddr,
                                    uint64_t touchChangeAddr, uint64_t patchChangeAddr,
                                    uint64_t startContactPatches, uint patchBytesLimit,
                                    uint workIndex, uint nbContacts)
{
    Ptr<PxgPatchAndContactCountersGpu> counters = Ptr<PxgPatchAndContactCountersGpu>(countersAddr);
    Ptr<PxsContactManagerOutputGpu> cmOutputs = Ptr<PxsContactManagerOutputGpu>(cmOutputsAddr);
    Ptr<uint> touchChangeFlags = Ptr<uint>(touchChangeAddr);
    Ptr<uint> patchChangeFlags = Ptr<uint>(patchChangeAddr);

    const uint allflags = cmOutputs[workIndex].allflagsWord;
    const uint oldStatusFlags = (allflags >> 16u) & 0xFFu;
    uint statusFlags = oldStatusFlags & (~NPT_FLAG_TOUCH_KNOWN);
    if (nbContacts != 0u) statusFlags |= NPT_FLAG_HAS_TOUCH;
    else                  statusFlags |= NPT_FLAG_HAS_NO_TOUCH;

    const uint prevPatches = (allflags >> 8u) & 0xFFu;
    const bool previouslyHadTouch = (oldStatusFlags & NPT_FLAG_HAS_TOUCH) != 0u;
    const bool prevTouchKnown = (oldStatusFlags & NPT_FLAG_TOUCH_KNOWN) != 0u;
    const uint numPatches = (nbContacts > 0u) ? 1u : 0u;
    const bool currentlyHasTouch = nbContacts > 0u;
    const bool change = (previouslyHadTouch != currentlyHasTouch) || (!prevTouchKnown);

    touchChangeFlags[workIndex] = change ? 1u : 0u;
    patchChangeFlags[workIndex] = (prevPatches != numPatches) ? 1u : 0u;

    cmOutputs[workIndex].allflagsWord = (prevPatches << 24u) | (statusFlags << 16u) | (numPatches << 8u);
    {
        const uint nf = cmOutputs[workIndex].nbContactsFlags;
        cmOutputs[workIndex].nbContactsFlags = (nf & 0xFFFF0000u) | (nbContacts & 0xFFFFu);
    }

    uint patchIndex = NPT_INVALID;
    if (nbContacts != 0u)
    {
        InterlockedAdd(counters[0].patchesBytes, NPT_PATCH_SIZE, patchIndex);
        if ((patchIndex + NPT_PATCH_SIZE) > patchBytesLimit)
        {
            InterlockedOr(counters[0].overflowError, NPT_OVERFLOW_PATCH);
            patchIndex = NPT_INVALID;

            // THE REWRITE -- see the header.
            statusFlags = (statusFlags & (~NPT_FLAG_TOUCH_KNOWN)) | NPT_FLAG_HAS_NO_TOUCH;
            cmOutputs[workIndex].allflagsWord = (prevPatches << 24u) | (statusFlags << 16u);
            const uint nf = cmOutputs[workIndex].nbContactsFlags;
            cmOutputs[workIndex].nbContactsFlags = (nf & 0xFFFF0000u);
            touchChangeFlags[workIndex] = previouslyHadTouch ? 1u : 0u;
            patchChangeFlags[workIndex] = (prevPatches != 0u) ? 1u : 0u;
        }
        else
        {
            cmOutputs[workIndex].contactPatches = startContactPatches + uint64_t(patchIndex);
        }
    }
    return patchIndex;
}

// insertIntoPatchStream: combine the two materials and write the 64-byte patch.
// ---------------------------------------------------------------------------
// THE MATERIAL COMBINE, FACTORED OUT (P9.83b-5).
//
// It used to live inside nptInsertIntoPatchStream, which hardcodes
// startContactIndex = 0 because its CUDA sibling (contactPatchUtils.cuh:163)
// does.  The mesh-mesh writer needs the SAME combine with a REAL
// startContactIndex (convexMeshMidphase.cu:656, `patchStartIndexS[threadIdx.x]`),
// so the choice was one body with a parameter or two bodies that drift.  This
// port has already paid for the second option once: contactReduceShared had
// SEVEN transcriptions before P9.83b-3 folded six of them, and two had quietly
// come to differ.
//
// materialCombiner.cuh's combineMaterials is what both sides transcribe.
// ---------------------------------------------------------------------------
public struct NptCombinedMaterial
{
    public float restitution;
    public float dynamicFriction;
    public float staticFriction;
    public float damping;
    public uint materialFlags;
}

// P9.83b-7's SWEEP, AND WHAT IT PROVED.  Ten mutants of this combine, driven
// through gpuslang.npInsertPatch: 9 killed outright.  Before the fold every one
// of them would have SURVIVED, because that fixture executed its own copy and
// never this body -- which is the whole point of folding it.
//
// The tenth, "delete `dyn = 0; sta = 0;` in the DISABLE_FRICTION arm", survived
// and was a MALFORMED MUTANT, not a finding: deleting the assignment leaves the
// two locals uninitialised, so the mutated program is undefined and its survival
// says nothing about the original.  Replacing the deletion with a SENTINEL
// (dyn = sta = 0.25f, and separately sta = 0.5f) kills it both ways, so the
// assignment is load-bearing.
//
// A MUTANT WHOSE MUTATED FORM IS UNDEFINED BEHAVIOUR TESTS NOTHING.  This is the
// second one this tranche -- the first skipped lanes whose bit could not yet be
// set -- and both sat in a results list looking like coverage gaps.  Mutate to a
// DIFFERENT DEFINED VALUE, never to an absent one.
//
// THE UNPACKED FORM IS THE BODY; THE BDA FORM IS AN ADAPTER (P9.83b-7).
//
// A SECOND full transcription of this combine lived in npInsertPatch.slang, which
// imported nothing -- so gpuslang.npInsertPatch, 5000 cases with every field
// compared against a C++ reference, certified THAT copy while the body the engine
// runs (reached from triPlaneNpBda and npCvxPlaneNphase) was covered only by
// scenes.  A fixture that passes while testing a different body is the failure
// this port paid for once already with contactReduceShared.
//
// The two could not simply share an entry point because the LAYOUTS differ:
// PxsMaterialDataGpu packs the 16-bit flags and the friction and restitution
// modes into flagsModes0 and the damping mode into modes1, while the fixture
// carries them as separate fields.  So the pure combine takes UNPACKED scalars
// and the BDA entry unpacks and delegates.  Now both reach the same body.
public struct NptMaterialIn
{
    public float dynamicFriction;
    public float staticFriction;
    public float restitution;
    public float damping;
    public uint flags;                // the 16-bit material flags
    public uint fricCombineMode;
    public uint restCombineMode;
    public uint dampingCombineMode;
}

public NptCombinedMaterial nptCombineMaterialsRaw(NptMaterialIn m0, NptMaterialIn m1)
{
    const float r0f = m0.restitution, r1f = m1.restitution;
    const bool compliant0 = r0f < 0.0f, compliant1 = r1f < 0.0f;
    const bool exactlyOneCompliant = compliant0 != compliant1;
    const bool bothCompliant = compliant0 && compliant1;
    const uint m0flags = m0.flags;
    const uint m1flags = m1.flags;
    const uint m0fric = m0.fricCombineMode;
    const uint m1fric = m1.fricCombineMode;
    const uint m0rest = m0.restCombineMode;
    const uint m1rest = m1.restCombineMode;
    const uint m0damp = m0.dampingCombineMode;
    const uint m1damp = m1.dampingCombineMode;
    const bool cAcc0 = (m0flags & NPT_MAT_COMPLIANT_ACC_SPRING) != 0u;
    const bool cAcc1 = (m1flags & NPT_MAT_COMPLIANT_ACC_SPRING) != 0u;
    const bool exactlyOneAcc = cAcc0 != cAcc1;

    float rest, damping;
    if (bothCompliant && exactlyOneAcc)
        rest = cAcc0 ? r0f : r1f;
    else
    {
        const uint mode = exactlyOneCompliant ? NPT_MODE_MIN : max(m0rest, m1rest);
        const float flip = (bothCompliant && (mode == NPT_MODE_MULTIPLY)) ? -1.0f : 1.0f;
        rest = flip * nptCombineScalars(r0f, r1f, mode);
    }
    if (bothCompliant && exactlyOneAcc)
        damping = cAcc0 ? m0.damping : m1.damping;
    else
    {
        const uint mode = exactlyOneCompliant ? NPT_MODE_MAX : max(m0damp, m1damp);
        damping = nptCombineScalars(m0.damping, m1.damping, mode);
    }

    const uint combineFlags = m0flags | m1flags;
    float sta, dyn;
    uint matFlags;
    if ((combineFlags & NPT_MAT_DISABLE_FRICTION) == 0u)
    {
        const uint fm = max(m0fric, m1fric);
        const float dynF = nptCombineScalars(m0.dynamicFriction, m1.dynamicFriction, fm);
        const float staF = nptCombineScalars(m0.staticFriction, m1.staticFriction, fm);
        const float fDyn = max(dynF, 0.0f);
        dyn = fDyn;
        sta = (staF - fDyn) >= 0.0f ? staF : fDyn;
        matFlags = combineFlags;
    }
    else
    {
        matFlags = combineFlags | NPT_MAT_DISABLE_STRONG_FRICTION;
        dyn = 0.0f; sta = 0.0f;
    }

    NptCombinedMaterial out;
    out.restitution = rest;
    out.dynamicFriction = dyn;
    out.staticFriction = sta;
    out.damping = damping;
    out.materialFlags = matFlags;
    return out;
}

// The BDA adapter: read two PxsMaterialDataGpu, unpack, delegate.
public NptCombinedMaterial nptCombineMaterials(uint64_t materialsAddr, uint mi0, uint mi1)
{
    Ptr<PxsMaterialDataGpu> materials = Ptr<PxsMaterialDataGpu>(materialsAddr);
    // P9.88: combineMaterials (materialCombiner.cuh:35) takes PxU16 index0/index1, and
    // every CUDA caller hands it the PxU32 PxgShape::materialIndex (contactPatchUtils.cuh
    // :149-150 explicitly, convexMeshMidphase.cu:644 implicitly), so the lookup is by the
    // low 16 bits.  One mask here covers convexPlane, trimeshPlane and triangleTriangle.
    const PxsMaterialDataGpu p0 = materials[mi0 & 0xFFFFu];
    const PxsMaterialDataGpu p1 = materials[mi1 & 0xFFFFu];

    NptMaterialIn a;
    a.dynamicFriction = p0.dynamicFriction;
    a.staticFriction = p0.staticFriction;
    a.restitution = p0.restitution;
    a.damping = p0.damping;
    a.flags = p0.flagsModes0 & 0xFFFFu;
    a.fricCombineMode = (p0.flagsModes0 >> 16u) & 0xFFu;
    a.restCombineMode = (p0.flagsModes0 >> 24u) & 0xFFu;
    a.dampingCombineMode = p0.modes1 & 0xFFu;

    NptMaterialIn b;
    b.dynamicFriction = p1.dynamicFriction;
    b.staticFriction = p1.staticFriction;
    b.restitution = p1.restitution;
    b.damping = p1.damping;
    b.flags = p1.flagsModes0 & 0xFFFFu;
    b.fricCombineMode = (p1.flagsModes0 >> 16u) & 0xFFu;
    b.restCombineMode = (p1.flagsModes0 >> 24u) & 0xFFu;
    b.dampingCombineMode = p1.modes1 & 0xFFu;

    return nptCombineMaterialsRaw(a, b);
}

public void nptInsertIntoPatchStream(uint64_t materialsAddr, uint64_t patchStreamAddr,
                                     uint mi0, uint mi1, uint patchIndex,
                                     float3 worldNormal, uint nbContacts)
{
    if (patchIndex == NPT_INVALID)
        return;

    const NptCombinedMaterial cm = nptCombineMaterials(materialsAddr, mi0, mi1);
    const float rest = cm.restitution;
    const float dyn = cm.dynamicFriction;
    const float sta = cm.staticFriction;
    const float damping = cm.damping;
    const uint matFlags = cm.materialFlags;

    Ptr<uint> patchBase = Ptr<uint>(patchStreamAddr + uint64_t(patchIndex));
    const uint one = asuint(1.0f);
    patchBase[0] = one; patchBase[1] = one; patchBase[2] = one; patchBase[3] = one;
    patchBase[4] = asuint(worldNormal.x);
    patchBase[5] = asuint(worldNormal.y);
    patchBase[6] = asuint(worldNormal.z);
    patchBase[7] = asuint(rest);
    patchBase[8] = asuint(dyn);
    patchBase[9] = asuint(sta);
    patchBase[10] = asuint(damping);
    patchBase[11] = ((nbContacts & 0xFFu) << 16u) | ((matFlags & 0xFFu) << 24u);
    patchBase[12] = (mi0 & 0xFFFFu) << 16u;
    // materialIndex1 is a PxU16 store (contactPatchUtils.cuh:171): pad[0] (bits
    // 16-31) keeps the stream's content.  A plain read-modify-write is exact: the
    // 64-byte slot is this pair's own InterlockedAdd reservation (nptRegisterContactPatch)
    // and both engine callers run the tail on one invocation (npCvxPlaneNphase
    // tid.x == 0, triPlaneNpBda gtid.x == 0).  Same form as pxgSlangTriTriPatch.
    patchBase[13] = (patchBase[13] & 0xFFFF0000u) | (mi1 & 0xFFFFu);
    // patchBase[14], patchBase[15] are pad[1..4] and are NOT written: the CUDA never assigns pad.
}
