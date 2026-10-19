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

// pxgSlangCvxCoreOut32.slang -- TestOutput::writeOutput, the Gu::Contact OVERLOAD
// (convexCoreCollision.cu:292-445), on the engine's BDA sinks.  P9.70c.
//
// ===========================================================================
// WHY THIS IS A TRANSCRIPTION AND NOT A LIFT, WHICH IS THE POINT OF THE FILE
// ===========================================================================
//
// P9.70a and P9.70b were both WITHDRAWN after reading the corpus instead of the
// CUDA: Contact/Contact32 and TrimeshBvh were already ported and already verified.
// This tail looked like the third of those.  It is not, and the difference is
// exactly the kind that a plan made from the .cu cannot see.
//
// `ccoWriteContact` (npCvxCoreOut.slang:281-402) IS a faithful port of these 121
// lines -- and it writes into a FIXTURE RECORD:
//
//     r.stream[64]     a float4 array in a RWStructuredBuffer
//     r.patchImg[64]   a uint4 array in the same record
//     r.counters/.outputs/.flags   what the atomics WOULD have produced
//
// A fixture does that deliberately: it makes every produced byte comparable to a
// CPU mirror without needing the engine's ten device buffers.  But it means there
// is no function in the corpus that writes these bytes THROUGH Ptr<> into
// contactStream / patchStream / cmOutputs, and no function that runs the
// InterlockedAdd against PxgPatchAndContactCounters.  That half is genuinely new
// and is written here, once, for the three multi-patch convexCore kernels
// (Trimesh, Tetmesh, Clothmesh) that will each need it.
//
// The fixture stays the ARBITER of the arithmetic: every counter, flag word and
// patch word below is the same expression ccoWriteContact computes and
// testNpCvxCoreOut compares bitwise against its CPU mirror.  What is unverified by
// that fixture -- and is stated rather than implied -- is the ADDRESSING: which
// device byte each value lands on.  Until P9.87 only the scenes checked that (see
// below).
//
// ONE WORD DIFFERS FROM ccoWriteContact ON PURPOSE (P9.87): word 13's high half.
// ccoWriteContact models a zero-initialised RECORD image (npCvxCoreOut.slang) and
// writes 0 there; this tail writes into the stream and, like cu:434 (a PxU16
// store), leaves pad[0] as the stream holds it.  Since P9.87 the ADDRESSING is
// verified too: cvxCoreTailBda.slang (cctMulti) #includes this tail and
// tests/testCvxCoreTailBda.cpp runs it over per-word-poisoned streams, cmOutputs and
// counters, compared whole against a mirror of cu:292-445.
//
// ===========================================================================
// FIVE DIFFERENCES FROM THE SINGLE-PATCH TAIL, ALL OF THEM LOAD-BEARING
// ===========================================================================
//
// pxgSlangCvxCoreOut.slang's cvxCoreWriteOutput is the :177-290 overload, for a
// generator that produces ONE patch of up to four points.  Reusing it here would
// be wrong in five separate places, which is why it is not parameterised:
//
//  1. TOUCH GATES ON numPatches, NOT numPoints.  `currentlyHasTouch = numPatches > 0`
//     (:374).  The single-patch tail writes `nbContacts > 0`.  P9.87 CORRECTED the
//     reason this note used to give ("a contact can hold points in a patch that was
//     dropped"): Contact32 keeps mNumPoints == the sum of its patches' counts --
//     findCreatePatch opens or resets a patch at 0 points and the same addPoints
//     round fills it with at least the deepest point (unit normals put that lane in
//     sameNormalMask), a reset patch's points are subtracted first (cu:899-900), and
//     dropped points are never added.  So numPatches > 0 and numPoints > 0 are the
//     SAME predicate on every state the accumulator reaches, and the mutant that
//     swaps them survives testCvxCoreTailBda as an equivalent.  Transcribed as the
//     CUDA spells it anyway.
//  2. patchChange COMPARES prevPatches AGAINST numPatches (:378), not against 1.
//  3. THE STREAM IS WRITTEN PATCH-MAJOR (:340-347): for each patch, for each of
//     ITS points, in patch order -- `baseContactStream++` walks a single cursor.
//     Point i of the record is NOT point i of the stream.
//  4. THE PATCH RESERVATION IS numPatches * 64 BYTES (:386), and the overflow test
//     uses that same product (:388) rather than one patch's worth.  Since 5.12 it
//     is the REQUESTED count (nbRequestedPatches), taken before a failed contact
//     or force allocation zeroes numPatches and numPoints.
//  (Line numbers in this header are 5.6.1's; the 5.12 bodies sit about 20 lines
//  higher after the licence-header change.)
//  5. THE PxContactPatch ARRAY IS WRITTEN INLINE, numPatches of them, each with its
//     OWN normal and nbContacts and a RUNNING startContactIndex (:421-443).  The
//     single-patch tail inlines `insertIntoPatchStream`, which writes exactly one.
//
// ===========================================================================
// REQUIREMENTS ON THE INCLUDER
// ===========================================================================
//
// Include AFTER pxgSlangCvxCoreOut.slang (CvxCoreOutRefs, the OVERFLOW_*/FLAG_*/
// *_SIZE constants, cvxCoreCombineMaterials) and AFTER pxgSlangCvxCoreAcc.slang
// (MAX_PATCHES, MAX_PATCH_POINTS -- this tail reads the accumulator's state
// directly, so the two must agree on those bounds by construction rather than by
// a matching literal).

// Gu::Contact's state, as pxgSlangCvxCoreAcc leaves it.  Passed as one struct so a
// caller cannot silently hand the four arrays over in the wrong order.
struct CvxCoreContact32
{
    uint numPatches;
    uint numPoints;
    float3 patchNormal[MAX_PATCHES];
    uint patchCount[MAX_PATCHES];
    float4 points[MAX_PATCHES * MAX_PATCH_POINTS]; // xyz = world point, w = dist
}

void cvxCoreWriteOutput32(uint testIndex, PxgShapeGpu shape0, PxgShapeGpu shape1,
                          CvxCoreContact32 c, CvxCoreOutRefs o)
{
    Ptr<PxgPatchAndContactCountersGpu> counters =
        Ptr<PxgPatchAndContactCountersGpu>(o.patchAndContactCounters);
    Ptr<PxsContactManagerOutputGpu> cmOutputs =
        Ptr<PxsContactManagerOutputGpu>(o.cmOutputs);
    Ptr<uint> touchChangeFlags = Ptr<uint>(o.touchChangeFlags);
    Ptr<uint> patchChangeFlags = Ptr<uint>(o.patchChangeFlags);

    // PxU8(contact.numPatches()) / PxU16(contact.numPoints()) (:313-314): the
    // NARROWING IS PART OF THE SEMANTICS.  numPatches is capped at 16 by the
    // accumulator so the PxU8 can never bite, but numPoints is written into an
    // eight-bit patch field further down and into a sixteen-bit output field here;
    // the masks are transcribed rather than assumed away.
    uint numPatches = c.numPatches & 0xFFu;
    uint numPoints = c.numPoints & 0xFFFFu;
    // 5.12 (:296): the patch demand survives a failed contact/force allocation.
    const uint nbRequestedPatches = numPatches;

    uint contactByteOffset = INVALID;
    uint forceByteOffset = INVALID;
    if (numPoints != 0u)
    {
        InterlockedAdd(counters[0].contactsBytes, CONTACT_SIZE * numPoints, contactByteOffset);
        InterlockedAdd(counters[0].forceAndIndiceBytes, FORCE_SIZE * numPoints, forceByteOffset);

        // 5.12 (:304-315): the limit test covers the WHOLE reservation (it was ONE
        // PxContact / ONE PxU32 up to 5.6.1), and a force overflow also drops the
        // contact offset.
        if ((uint64_t(contactByteOffset) + uint64_t(CONTACT_SIZE) * uint64_t(numPoints)) > uint64_t(o.contactBytesLimit))
        {
            InterlockedOr(counters[0].overflowError, OVERFLOW_CONTACT);
            contactByteOffset = INVALID;
        }
        if ((uint64_t(forceByteOffset) + uint64_t(FORCE_SIZE) * uint64_t(numPoints)) > uint64_t(o.forceBytesLimit))
        {
            InterlockedOr(counters[0].overflowError, OVERFLOW_FORCE);
            forceByteOffset = INVALID;
            contactByteOffset = INVALID;
        }
    }

    if (contactByteOffset != INVALID)
    {
        // (:338-347) `float4* baseContactStream = contactStream + contactByteOffset`
        // then ONE cursor walked patch-major.  The points are already in WORLD space:
        // Contact32::addPoints applied transform1 on the way in (pxgSlangCvxCoreAcc),
        // so there is no transform here, exactly as in the .cu.
        Ptr<float4> baseContactStream = Ptr<float4>(o.contactStream + uint64_t(contactByteOffset));
        uint slot = 0u;
        for (uint i = 0u; i < numPatches; ++i)
        {
            const uint n = c.patchCount[i];
            for (uint j = 0u; j < n; ++j)
                baseContactStream[slot++] = c.points[i * MAX_PATCH_POINTS + j];
        }

        // Since 5.12 a force overflow also invalidates contactByteOffset, so the
        // force offset is always valid here (5.6.1 stored startContactForces +
        // 0xFFFFFFFF after a force-only overflow).
        cmOutputs[testIndex].contactForces = o.startContactForces + uint64_t(forceByteOffset);
        cmOutputs[testIndex].contactPoints = o.startContactPoints + uint64_t(contactByteOffset);
    }
    else
    {
        cmOutputs[testIndex].contactForces = 0uL;
        cmOutputs[testIndex].contactPoints = 0uL;
        numPoints = 0u;  // 5.12 (:337-338)
        numPatches = 0u;
    }

    const uint allflags = cmOutputs[testIndex].allflagsWord;
    const uint oldStatusFlags = (allflags >> 16u) & 0xFFu;
    uint statusFlags = oldStatusFlags & (~FLAG_TOUCH_KNOWN);
    if (numPoints != 0u) statusFlags |= FLAG_HAS_TOUCH;    // gates on numPoints (:365)
    else                 statusFlags |= FLAG_HAS_NO_TOUCH;

    const uint prevPatches = (allflags >> 8u) & 0xFFu;
    const bool previouslyHadTouch = (oldStatusFlags & FLAG_HAS_TOUCH) != 0u;
    const bool prevTouchKnown = (oldStatusFlags & FLAG_TOUCH_KNOWN) != 0u;
    // ...but the touch CHANGE gates on numPatches (:374).  Difference 1 (equivalent
    // under Contact32's invariant; see the header).
    const bool currentlyHasTouch = numPatches > 0u;
    const bool change = (previouslyHadTouch != currentlyHasTouch) || (!prevTouchKnown);

    touchChangeFlags[testIndex] = change ? 1u : 0u;
    patchChangeFlags[testIndex] = (prevPatches != numPatches) ? 1u : 0u;  // difference 2

    cmOutputs[testIndex].allflagsWord = (prevPatches << 24u) | (statusFlags << 16u) | (numPatches << 8u);
    {
        const uint nf = cmOutputs[testIndex].nbContactsFlags;
        cmOutputs[testIndex].nbContactsFlags = (nf & 0xFFFF0000u) | numPoints;
    }

    uint patchIndex = INVALID;
    if (nbRequestedPatches != 0u)
    {
        // difference 4: the REQUESTED patches reserved, and the SAME product tested.
        InterlockedAdd(counters[0].patchesBytes, nbRequestedPatches * PATCH_SIZE, patchIndex);
        if ((uint64_t(patchIndex) + uint64_t(nbRequestedPatches) * uint64_t(PATCH_SIZE)) > uint64_t(o.patchBytesLimit))
        {
            InterlockedOr(counters[0].overflowError, OVERFLOW_PATCH);
            patchIndex = INVALID;

            // The in-gate rollback (:393-400).  Note what it does NOT undo: the
            // contact stream write above stands, and contactPoints/contactForces
            // still point at it.  nbContacts going to zero is what makes those
            // bytes unreachable.  Transcribed, not tidied.
            statusFlags = (statusFlags & (~FLAG_TOUCH_KNOWN)) | FLAG_HAS_NO_TOUCH;
            cmOutputs[testIndex].allflagsWord = (prevPatches << 24u) | (statusFlags << 16u);
            const uint nf = cmOutputs[testIndex].nbContactsFlags;
            cmOutputs[testIndex].nbContactsFlags = (nf & 0xFFFF0000u);
            touchChangeFlags[testIndex] = previouslyHadTouch ? 1u : 0u;
            patchChangeFlags[testIndex] = (prevPatches != 0u) ? 1u : 0u;
        }
        else if (numPatches != 0u)
        {
            cmOutputs[testIndex].contactPatches = o.startContactPatches + uint64_t(patchIndex);
        }
        else
            patchIndex = INVALID;
    }

    // difference 5: the PxContactPatch ARRAY (:408-444), 64 bytes = 16 words each:
    //   w0-w3   mMassModification (linear0, angular0, linear1, angular1), all 1.0f
    //   w4-w6   normal;   w7 restitution;  w8 dynamicFriction;  w9 staticFriction;
    //   w10     damping
    //   w11     startContactIndex(0:16) | nbContacts(16:24) | materialFlags(24:32)
    //   w12     internalFlags(=0)(0:16) | materialIndex0(16:32)
    //   w13     materialIndex1(0:16), pad[0](16:32) NOT WRITTEN (a PxU16 store, cu:434);
    //   w14/w15 NOT WRITTEN by the CUDA
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

        const uint one = asuint(1.0f);
        uint startContactIndex = 0u;
        for (uint i = 0u; i < numPatches; ++i)
        {
            const uint nbPatchPoints = c.patchCount[i];
            const float3 n = c.patchNormal[i];
            Ptr<uint> p = Ptr<uint>(o.patchStream + uint64_t(patchIndex) + uint64_t(i) * uint64_t(PATCH_SIZE));
            p[0] = one; p[1] = one; p[2] = one; p[3] = one;
            p[4] = asuint(n.x);
            p[5] = asuint(n.y);
            p[6] = asuint(n.z);
            p[7] = asuint(rest);
            p[8] = asuint(dyn);
            p[9] = asuint(sta);
            p[10] = asuint(damping);
            p[11] = (startContactIndex & 0xFFFFu) | ((nbPatchPoints & 0xFFu) << 16u)
                  | ((matFlags & 0xFFu) << 24u);
            p[12] = (mi0 & 0xFFFFu) << 16u;   // internalFlags = 0 in the low half
            // P9.87: cu:434 is a PxU16 store, so pad[0] (bits 16-31) keeps the stream's
            // content.  Single writer: lane 0 alone calls this, and patch i is its own
            // 64-byte slot inside this invocation's InterlockedAdd reservation.
            p[13] = (p[13] & 0xFFFF0000u) | (mi1 & 0xFFFFu);
            // p[14], p[15] are pad[1..4] and are NOT written, matching cu:425-441.
            startContactIndex += nbPatchPoints;
        }
    }
}
