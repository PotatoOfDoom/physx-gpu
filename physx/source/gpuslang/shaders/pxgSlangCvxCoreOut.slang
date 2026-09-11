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

// pxgSlangCvxCoreOut.slang -- TestOutput::writeOutput (convexCoreCollision.cu:177-290),
// the contact/patch/force emission tail every convexCore*Nphase kernel ends with.
// P9.69b lifted it out of cvxCorePlaneNpBda so that the convex arm does not get a
// SECOND transcription of it.
//
// WHY THIS IS AN INCLUDE AND NOT A MODULE: the same measurement that decided
// pxgSlangCvxCoreCvx (see that file's header) -- a module boundary took
// npCvxCoreCvxGen's compile from 1.07 seconds to over forty minutes.  Textual
// inclusion has no boundary and no cost.
//
// WHY THE ADDRESSES ARRIVE IN A STRUCT RATHER THAN THROUGH `gParams`: an include
// that reached for the includer's push-constant block by NAME would compile, and
// would silently bind to whatever that file happened to call its parameters.  The
// struct makes the thirteen things this tail touches an explicit contract, which is
// what a reader needs when the same bytes are written from two kernels.
//
// P9.37's finding applies here and is the reason to share rather than copy: a
// replicated engine tail drifts in ASSOCIATION, not just in operators, and the
// drift is invisible until a scene disagrees.

// The constants and the material combiner the tail needs, moved here with it
// (P9.69b) so an includer only has to supply the EGT_* geometry types.
static const uint INVALID = 0xFFFFFFFFu;



// PxsContactManagerStatusFlag
static const uint FLAG_HAS_NO_TOUCH = 1u;
static const uint FLAG_HAS_TOUCH = 2u;
static const uint FLAG_TOUCH_KNOWN = 3u;

// PxgPatchAndContactCounters::OverflowError
static const uint OVERFLOW_CONTACT = 1u;
static const uint OVERFLOW_FORCE = 2u;
static const uint OVERFLOW_PATCH = 4u;

static const uint CONTACT_SIZE = 16u; // sizeof(PxContact)
static const uint FORCE_SIZE = 4u;    // sizeof(PxU32)
static const uint PATCH_SIZE = 64u;   // sizeof(PxContactPatch)

// PxCombineMode / PxMaterialFlag
static const uint MODE_AVERAGE = 0u, MODE_MIN = 1u, MODE_MULTIPLY = 2u, MODE_MAX = 3u;
static const uint MAT_DISABLE_FRICTION = 1u, MAT_DISABLE_STRONG_FRICTION = 2u, MAT_COMPLIANT_ACC_SPRING = 16u;

// PxsCombineMaterials scalar combine (npInsertPatch combineMaterials).
float combineScalars(float a, float b, uint mode)
{
    if (mode == MODE_AVERAGE) return 0.5f * (a + b);
    if (mode == MODE_MIN) return min(a, b);
    if (mode == MODE_MULTIPLY) return a * b;
    if (mode == MODE_MAX) return max(a, b);
    return 0.0f;
}

// ---------------------------------------------------------------------------
// combineMaterials (npInsertPatch), factored out of cvxCoreWriteOutput by P9.70c
// so the MULTI-PATCH tail (pxgSlangCvxCoreOut32.slang) shares these instructions
// instead of carrying a second transcription of them.  That is P9.37's finding
// applied a second time: the danger in a replicated tail is not that an operator
// is mistyped, it is that an ASSOCIATION drifts, and a drifted association is
// invisible until a scene disagrees.
void cvxCoreCombineMaterials(uint64_t materialsAddr, uint mi0, uint mi1,
                             out float sta, out float dyn, out float rest,
                             out float damping, out uint matFlags)
{
    Ptr<PxsMaterialDataGpu> materials = Ptr<PxsMaterialDataGpu>(materialsAddr);
    const PxsMaterialDataGpu m0 = materials[mi0];
    const PxsMaterialDataGpu m1 = materials[mi1];

    const float r0f = m0.restitution, r1f = m1.restitution;
    const bool compliant0 = r0f < 0.0f, compliant1 = r1f < 0.0f;
    const bool exactlyOneCompliant = compliant0 != compliant1;
    const bool bothCompliant = compliant0 && compliant1;
    const uint m0flags = m0.flagsModes0 & 0xFFFFu;
    const uint m1flags = m1.flagsModes0 & 0xFFFFu;
    const uint m0fric = (m0.flagsModes0 >> 16u) & 0xFFu;
    const uint m1fric = (m1.flagsModes0 >> 16u) & 0xFFu;
    const uint m0rest = (m0.flagsModes0 >> 24u) & 0xFFu;
    const uint m1rest = (m1.flagsModes0 >> 24u) & 0xFFu;
    const uint m0damp = m0.modes1 & 0xFFu;
    const uint m1damp = m1.modes1 & 0xFFu;
    const bool cAcc0 = (m0flags & MAT_COMPLIANT_ACC_SPRING) != 0u;
    const bool cAcc1 = (m1flags & MAT_COMPLIANT_ACC_SPRING) != 0u;
    const bool exactlyOneAcc = cAcc0 != cAcc1;

    if (bothCompliant && exactlyOneAcc)
        rest = cAcc0 ? r0f : r1f;
    else
    {
        const uint mode = exactlyOneCompliant ? MODE_MIN : max(m0rest, m1rest);
        const float flip = (bothCompliant && (mode == MODE_MULTIPLY)) ? -1.0f : 1.0f;
        rest = flip * combineScalars(r0f, r1f, mode);
    }
    if (bothCompliant && exactlyOneAcc)
        damping = cAcc0 ? m0.damping : m1.damping;
    else
    {
        const uint mode = exactlyOneCompliant ? MODE_MAX : max(m0damp, m1damp);
        damping = combineScalars(m0.damping, m1.damping, mode);
    }

    const uint combineFlags = m0flags | m1flags;
    if ((combineFlags & MAT_DISABLE_FRICTION) == 0u)
    {
        const uint fm = max(m0fric, m1fric);
        const float dynF = combineScalars(m0.dynamicFriction, m1.dynamicFriction, fm);
        const float staF = combineScalars(m0.staticFriction, m1.staticFriction, fm);
        const float fDyn = max(dynF, 0.0f);
        dyn = fDyn;
        sta = (staF - fDyn) >= 0.0f ? staF : fDyn;
        matFlags = combineFlags;
    }
    else
    {
        matFlags = combineFlags | MAT_DISABLE_STRONG_FRICTION;
        dyn = 0.0f; sta = 0.0f;
    }
}

struct CvxCoreOutRefs
{
    uint64_t cmOutputs;
    uint64_t materials;
    uint64_t contactStream;
    uint64_t patchStream;
    uint64_t patchAndContactCounters;
    uint64_t touchChangeFlags;
    uint64_t patchChangeFlags;
    uint64_t startContactPatches;
    uint64_t startContactPoints;
    uint64_t startContactForces;
    uint patchBytesLimit;
    uint contactBytesLimit;
    uint forceBytesLimit;
}

// `patchNormal` is what the CUDA passes as the patch's normal, which is the NEGATED
// contact normal at both call sites (convexCoreCollision.cu:492 for the plane arm).
// The negation stays with the caller so this file never has to know which side the
// generator considered "first".
void cvxCoreWriteOutput(uint testIndex, PxgShapeGpu shape0, PxgShapeGpu shape1,
                        float3 patchNormal, float3 pointsOut[4], float distsOut[4],
                        uint nbContacts, CvxCoreOutRefs o)
{
    // ---- TestOutput::writeOutput (:177-290) --------------------------------
    Ptr<PxgPatchAndContactCountersGpu> counters =
        Ptr<PxgPatchAndContactCountersGpu>(o.patchAndContactCounters);
    Ptr<PxsContactManagerOutputGpu> cmOutputs =
        Ptr<PxsContactManagerOutputGpu>(o.cmOutputs);
    Ptr<uint> touchChangeFlags = Ptr<uint>(o.touchChangeFlags);
    Ptr<uint> patchChangeFlags = Ptr<uint>(o.patchChangeFlags);

    uint contactByteOffset = INVALID;
    uint forceByteOffset = INVALID;
    if (nbContacts != 0u)
    {
        InterlockedAdd(counters[0].contactsBytes, CONTACT_SIZE * nbContacts, contactByteOffset);
        InterlockedAdd(counters[0].forceAndIndiceBytes, FORCE_SIZE * nbContacts, forceByteOffset);
        if ((uint64_t(contactByteOffset) + uint64_t(CONTACT_SIZE)) > uint64_t(o.contactBytesLimit))
        {
            InterlockedOr(counters[0].overflowError, OVERFLOW_CONTACT);
            contactByteOffset = INVALID;
        }
        if ((uint64_t(forceByteOffset) + uint64_t(FORCE_SIZE)) > uint64_t(o.forceBytesLimit))
        {
            InterlockedOr(counters[0].overflowError, OVERFLOW_FORCE);
            forceByteOffset = INVALID;
        }
    }

    if (contactByteOffset != INVALID)
    {
        // The points are ALREADY in world space here -- generateContacts works on
        // the world-space plane and the posed convex -- so unlike the convex-plane
        // kernel's tail there is no transform on the way out.
        Ptr<float4> baseContactStream = Ptr<float4>(o.contactStream + uint64_t(contactByteOffset));
        for (uint i = 0u; i < 4u; ++i)
            if (i < nbContacts)
                baseContactStream[i] = float4(pointsOut[i], distsOut[i]);

        cmOutputs[testIndex].contactForces = o.startContactForces + uint64_t(forceByteOffset);
        cmOutputs[testIndex].contactPoints = o.startContactPoints + uint64_t(contactByteOffset);
    }
    else
    {
        cmOutputs[testIndex].contactForces = 0uL;
        cmOutputs[testIndex].contactPoints = 0uL;
    }

    const uint allflags = cmOutputs[testIndex].allflagsWord;
    const uint oldStatusFlags = (allflags >> 16u) & 0xFFu;
    uint statusFlags = oldStatusFlags & (~FLAG_TOUCH_KNOWN);
    if (nbContacts != 0u) statusFlags |= FLAG_HAS_TOUCH;
    else                  statusFlags |= FLAG_HAS_NO_TOUCH;

    const uint prevPatches = (allflags >> 8u) & 0xFFu;
    const bool previouslyHadTouch = (oldStatusFlags & FLAG_HAS_TOUCH) != 0u;
    const bool prevTouchKnown = (oldStatusFlags & FLAG_TOUCH_KNOWN) != 0u;
    const uint numPatches = (nbContacts > 0u) ? 1u : 0u;
    const bool currentlyHasTouch = nbContacts > 0u;
    const bool change = (previouslyHadTouch != currentlyHasTouch) || (!prevTouchKnown);

    touchChangeFlags[testIndex] = change ? 1u : 0u;
    patchChangeFlags[testIndex] = (prevPatches != numPatches) ? 1u : 0u;

    cmOutputs[testIndex].allflagsWord = (prevPatches << 24u) | (statusFlags << 16u) | (numPatches << 8u);
    {
        const uint nf = cmOutputs[testIndex].nbContactsFlags;
        cmOutputs[testIndex].nbContactsFlags = (nf & 0xFFFF0000u) | (nbContacts & 0xFFFFu);
    }

    uint patchIndex = INVALID;
    if (nbContacts != 0u)
    {
        InterlockedAdd(counters[0].patchesBytes, PATCH_SIZE, patchIndex);
        if ((uint64_t(patchIndex) + uint64_t(PATCH_SIZE)) > uint64_t(o.patchBytesLimit))
        {
            InterlockedOr(counters[0].overflowError, OVERFLOW_PATCH);
            patchIndex = INVALID;

            statusFlags = (statusFlags & (~FLAG_TOUCH_KNOWN)) | FLAG_HAS_NO_TOUCH;
            cmOutputs[testIndex].allflagsWord = (prevPatches << 24u) | (statusFlags << 16u);
            const uint nf = cmOutputs[testIndex].nbContactsFlags;
            cmOutputs[testIndex].nbContactsFlags = (nf & 0xFFFF0000u);
            touchChangeFlags[testIndex] = previouslyHadTouch ? 1u : 0u;
            patchChangeFlags[testIndex] = (prevPatches != 0u) ? 1u : 0u;
        }
        else
        {
            cmOutputs[testIndex].contactPatches = o.startContactPatches + uint64_t(patchIndex);
        }
    }

    // insertIntoPatchStream(materials, patchStream, shape0, shape1, patchIndex,
    //   normal, nbContacts) -- inlined, as npCvxPlaneNphase inlines it.
    if (patchIndex != INVALID)
    {
        // PxU16 materialIndex0 = shape0.materialIndex; (cu:415-416).  PxgShape's
        // field is a PxU32 (PxgConvexConvexShape.h:53), and the CUDA narrows it
        // BEFORE it is used -- so the truncated value is what indexes materials[],
        // not just what is stored in the patch.  P9.70c: both tails passed the full
        // uint, which differs only above 65535 materials and therefore only where no
        // scene will ever go.  Transcribed anyway; the point of a bit-exact port is
        // that "unreachable" is not a reason to diverge.
        const uint mi0 = shape0.materialIndex & 0xFFFFu;
        const uint mi1 = shape1.materialIndex & 0xFFFFu;
        float sta, dyn, rest, damping;
        uint matFlags;
        cvxCoreCombineMaterials(o.materials, mi0, mi1, sta, dyn, rest, damping, matFlags);

        Ptr<uint> patchBase = Ptr<uint>(o.patchStream + uint64_t(patchIndex));
        const uint one = asuint(1.0f);
        patchBase[0] = one; patchBase[1] = one; patchBase[2] = one; patchBase[3] = one;
        patchBase[4] = asuint(patchNormal.x);
        patchBase[5] = asuint(patchNormal.y);
        patchBase[6] = asuint(patchNormal.z);
        patchBase[7] = asuint(rest);
        patchBase[8] = asuint(dyn);
        patchBase[9] = asuint(sta);
        patchBase[10] = asuint(damping);
        patchBase[11] = ((nbContacts & 0xFFu) << 16u) | ((matFlags & 0xFFu) << 24u);
        patchBase[12] = (mi0 & 0xFFFFu) << 16u;
        patchBase[13] = (mi1 & 0xFFFFu);
    }
}
