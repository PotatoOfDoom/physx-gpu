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

// npCvxCoreOut.slang
//
// Slang port of the two TestOutput::writeOutput overloads of the convex-core
// narrowphase family (convexCoreCollision.cu L177-445), the output stage
// shared by all five convexCore*Nphase kernels.
//
// Entry ccoWriteContacts (cu L177-290, the PxContact-array variant used by the
// plane and convex-convex kernels):
//   - contact/force reservations of 16*nb / 4*nb bytes; since 5.12 the
//     64-bit-promoted overflow tests cover the WHOLE reservation (offset +
//     sizeof(PxContact) * nb resp. sizeof(PxU32) * nb; independent ifs, both
//     flags can set) and a force overflow also drops the contact offset;
//   - on a valid contact offset the nb contacts are written serially as
//     (p.xyz, separation) float4s; outputs get startForces+forceOffset /
//     startPoints+contactOffset, else both NULL and (5.12) nbContacts = 0,
//     while the patch demand (patchRequested) is kept for the reservation
//     below, whose index is then returned as invalid;
//   - allflags rewrite (clear eTOUCH_KNOWN, set eHAS_TOUCH/eHAS_NO_TOUCH,
//     ((prevPatches<<8|status)<<16)|(numPatches<<8)) with numPatches =
//     nbContacts ? 1 : 0 and touch/patch change flags;
//   - single 64-byte patch reservation with the rollback (PATCH_BUFFER_
//     OVERFLOW, HAS_NO_TOUCH rewrite with numPatches=0, nbContacts=0,
//     touchChange=previouslyHadTouch, patchChange=prevPatches!=0);
//   - the trailing insertIntoPatchStream is the separately-verified
//     tranche-45 shader, so this port emits its call arguments (patchIndex,
//     the caller's normal, and nbContacts after the 5.12 zeroing).
//
// Entry ccoWriteContact (cu L292-445, the Gu::Contact variant used by the
// trimesh/heightfield/cloth kernels): same skeleton with the multi-patch
// differences ported verbatim:
//   - numPatches = u8(contact.numPatches()), numPoints = u16(numPoints());
//     reservations and eHAS_TOUCH gate on numPoints, but currentlyHasTouch =
//     numPatches > 0 (unlike the array variant!) and patchChange compares
//     prevPatches != numPatches;
//   - the contact stream is written patch-major (i < numPatches, j <
//     numPatchPoints(i), sequential slots) from mPoints[i*4+j];
//   - patch reservation of numPatches*64 bytes, 64-bit overflow compare; since
//     5.12 the REQUESTED count, taken before a failed allocation zeroes
//     numPatches/numPoints;
//   - on a valid patchIndex the PxContactPatch array is written INLINE (no
//     insertIntoPatchStream): combineMaterials (the tranche-43 core,
//     reproduced verbatim) then per patch mass 1.0f x4, patch normal,
//     restitution/dynamic/static/damping, startContactIndex accumulating
//     numPatchPoints, nbContacts, materialFlags u8, internalFlags 0,
//     materialIndex0/1.
//
// One test per thread.  NULL pointers are modelled as the 0xFFFFFFFE
// sentinel.

static const uint HAS_NO_TOUCH = 1u;
static const uint HAS_TOUCH = 2u;
static const uint TOUCH_KNOWN = 3u;
static const uint CONTACT_BUFFER_OVERFLOW = 1u;
static const uint FORCE_BUFFER_OVERFLOW = 2u;
static const uint PATCH_BUFFER_OVERFLOW = 4u;
static const uint NULL_SENTINEL = 0xFFFFFFFEu;

static const uint MODE_AVERAGE = 0u, MODE_MIN = 1u, MODE_MULTIPLY = 2u, MODE_MAX = 3u;
static const uint FLAG_DISABLE_FRICTION = 1u, FLAG_DISABLE_STRONG_FRICTION = 2u, FLAG_COMPLIANT_ACC_SPRING = 16u;

struct CoreOutIn
{
    float4 contacts[4];   // array variant: (p.xyz, separation)
    float4 normal;        // array variant: insertIntoPatchStream normal arg
    uint4 misc;           // x = nbContacts (array variant), y = allflags word,
                          // z = mNumPatches, w = mNumPoints (Contact variant)
    uint4 cfg;            // x = contactBytesLimit, y = forceBytesLimit, z = patchBytesLimit, w pad
    uint4 counters;       // x/y/z = initial contacts/force/patches bytes
    uint4 mat;            // x = materialIndex0, y = materialIndex1 (Contact variant), z/w pad
    float4 mat0F, mat1F;  // dynamicFriction, staticFriction, restitution, damping
    uint4 mat0U, mat1U;   // flags, fricCombineMode, restCombineMode, dampingCombineMode
    uint4 patchCounts[4]; // Contact variant: numPatchPoints per patch (16 uints)
    float4 patchN[16];    // Contact variant: patch normals (w pad)
    float4 points[64];    // Contact variant: mPoints[patch*4+point] = (p.xyz, d)
}

struct CoreOutOut
{
    float4 stream[64];    // contact stream slots relative to the initial contactsBytes
    uint4 counters;       // x/y/z = counters after, w = overflow flags
    uint4 outputs;        // x = contactForces, y = contactPoints, z = contactPatches (offsets,
                          // NULL_SENTINEL when NULL/unwritten), w = nbContacts stored
    uint4 flags;          // x = touchChange, y = patchChange, z = allflags after, w = patchIndex
    float4 patchArgs;     // array variant: insertIntoPatchStream normal (.xyz), nbContacts (.w)
    uint4 patchImg[64];   // Contact variant: PxContactPatch images (4 uint4 = 64 bytes each)
}

struct CoreOutParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CoreOutParams> gParams;

RWStructuredBuffer<CoreOutIn> gIn;    // binding 0
RWStructuredBuffer<CoreOutOut> gOut;  // binding 1

// PxsCombineMaterials (tranche-43 verified core), returning
// (staticFriction, dynamicFriction, restitution, damping) + flags
float combineScalars(float a, float b, uint mode)
{
    if (mode == MODE_AVERAGE) return 0.5f * (a + b);
    // P9.86e: PxMin/PxMax<float> (PxsMaterialCombiner.h:43/47, :60/:64) are selectMin/selectMax
    // (PxMath.h:81-101, PxUnixMathIntrinsics.h:116-125): on a +-0 tie the SECOND operand wins.  GLSL
    // FMin/FMax give min(+0,-0) = -0 / max(+0,-0) = +0 on llvmpipe in either order, so spell the selects.
    if (mode == MODE_MIN) return (a < b) ? a : b;
    if (mode == MODE_MULTIPLY) return a * b;
    if (mode == MODE_MAX) return (a > b) ? a : b;
    return 0.0f;
}
void combineMaterialsQ(float4 m0F, uint4 m0U, float4 m1F, uint4 m1U,
                       out float staFriction, out float dynFriction,
                       out float restitution, out float damping, out uint matFlags)
{
    const float r0 = m0F.z, r1 = m1F.z;
    const bool compliant0 = r0 < 0.0f;
    const bool compliant1 = r1 < 0.0f;
    const bool exactlyOneCompliant = compliant0 != compliant1;
    const bool bothCompliant = compliant0 && compliant1;
    const bool compliantAcc0 = (m0U.x & FLAG_COMPLIANT_ACC_SPRING) != 0u;
    const bool compliantAcc1 = (m1U.x & FLAG_COMPLIANT_ACC_SPRING) != 0u;
    const bool exactlyOneAccCompliant = compliantAcc0 != compliantAcc1;

    if (bothCompliant && exactlyOneAccCompliant)
    {
        restitution = compliantAcc0 ? r0 : r1;
    }
    else
    {
        const uint combineMode = exactlyOneCompliant ? MODE_MIN : max(m0U.z, m1U.z);
        const float flipSign = (bothCompliant && (combineMode == MODE_MULTIPLY)) ? -1.0f : 1.0f;
        restitution = flipSign * combineScalars(r0, r1, combineMode);
    }

    {
        const float d0 = m0F.w, d1 = m1F.w;
        if (bothCompliant && exactlyOneAccCompliant)
        {
            damping = compliantAcc0 ? d0 : d1;
        }
        else
        {
            const uint combineMode = exactlyOneCompliant ? MODE_MAX : max(m0U.w, m1U.w);
            damping = combineScalars(d0, d1, combineMode);
        }
    }

    {
        const uint combineFlags = m0U.x | m1U.x;
        if ((combineFlags & FLAG_DISABLE_FRICTION) == 0u)
        {
            const uint frictionCombineMode = max(m0U.y, m1U.y);
            const float dyn = combineScalars(m0F.x, m1F.x, frictionCombineMode);
            const float sta = combineScalars(m0F.y, m1F.y, frictionCombineMode);
            // PxMax(dynFriction, 0.0f) = selectMax (PxsMaterialCombiner.h:161): +0 for a +-0 dyn.
            const float fDyn = (dyn > 0.0f) ? dyn : 0.0f;
            const float fSta = (sta - fDyn) >= 0.0f ? sta : fDyn;
            dynFriction = fDyn;
            staFriction = fSta;
            matFlags = combineFlags;
        }
        else
        {
            matFlags = combineFlags | FLAG_DISABLE_STRONG_FRICTION;
            dynFriction = 0.0f;
            staFriction = 0.0f;
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ccoWriteContacts(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const CoreOutIn s = gIn[t];

    uint nbContacts = s.misc.x;
    const uint allflags = s.misc.y;
    const bool patchRequested = nbContacts > 0u; // 5.12 (cu :173)

    uint contactsBytes = s.counters.x, forceBytes = s.counters.y, patchesBytes = s.counters.z;
    uint ovf = 0u;

    CoreOutOut r;
    for (uint j = 0u; j < 64u; ++j) { r.stream[j] = float4(0, 0, 0, 0); r.patchImg[j] = uint4(0, 0, 0, 0); }

    // contact/force reservation, 5.12 whole-reservation 64-bit overflow tests, independent
    // ifs, a force overflow also drops the contact offset
    uint contactByteOffset = 0xFFFFFFFFu;
    uint forceByteOffset = 0xFFFFFFFFu;
    if (nbContacts != 0u)
    {
        contactByteOffset = contactsBytes;
        contactsBytes += 16u * nbContacts;
        forceByteOffset = forceBytes;
        forceBytes += 4u * nbContacts;
        if ((uint64_t(contactByteOffset) + 16ULL * uint64_t(nbContacts)) > uint64_t(s.cfg.x)) { ovf |= CONTACT_BUFFER_OVERFLOW; contactByteOffset = 0xFFFFFFFFu; }
        if ((uint64_t(forceByteOffset) + 4ULL * uint64_t(nbContacts)) > uint64_t(s.cfg.y))
        {
            ovf |= FORCE_BUFFER_OVERFLOW; forceByteOffset = 0xFFFFFFFFu;
            contactByteOffset = 0xFFFFFFFFu;
        }
    }

    uint outForces, outPoints;
    if (contactByteOffset != 0xFFFFFFFFu)
    {
        const uint base = (contactByteOffset - s.counters.x) / 16u;
        for (uint i = 0u; i < nbContacts; ++i)
            r.stream[base + i] = s.contacts[i]; // make_float4(p.contact, p.separation)
        outForces = forceByteOffset;
        outPoints = contactByteOffset;
    }
    else
    {
        outForces = NULL_SENTINEL; // NULL
        outPoints = NULL_SENTINEL;
        nbContacts = 0u; // 5.12 (cu :213)
    }

    const uint oldStatusFlags = (allflags >> 16u) & 0xFFu;
    uint status = oldStatusFlags;
    status &= ~TOUCH_KNOWN;
    if (nbContacts != 0u) status |= HAS_TOUCH;
    else status |= HAS_NO_TOUCH;

    const uint prevPatches = (allflags & 0xFFFFu) >> 8u;
    const bool previouslyHadTouch = (oldStatusFlags & HAS_TOUCH) != 0u;
    const bool prevTouchKnown = (oldStatusFlags & TOUCH_KNOWN) != 0u;
    const uint numPatches = (nbContacts > 0u) ? 1u : 0u;
    const bool currentlyHasTouch = nbContacts > 0u;

    uint touchChange = ((previouslyHadTouch != currentlyHasTouch) || !prevTouchKnown) ? 1u : 0u;
    uint patchChange = (prevPatches != numPatches) ? 1u : 0u;
    uint allflagsOut = (((prevPatches << 8u) | status) << 16u) | (numPatches << 8u);
    uint nbOut = nbContacts;

    // single 64-byte patch reservation with the in-gate rollback
    uint patchIndex = 0xFFFFFFFFu;
    uint cpOut = NULL_SENTINEL;
    if (patchRequested)
    {
        patchIndex = patchesBytes;
        patchesBytes += 64u; // sizeof(PxContactPatch)
        if ((uint64_t(patchIndex) + 64ULL) > uint64_t(s.cfg.z))
        {
            ovf |= PATCH_BUFFER_OVERFLOW;
            patchIndex = 0xFFFFFFFFu;

            status &= ~TOUCH_KNOWN;
            status |= HAS_NO_TOUCH;
            allflagsOut = ((prevPatches << 8u) | status) << 16u; // numPatches = 0
            nbOut = 0u;
            touchChange = previouslyHadTouch ? 1u : 0u;
            patchChange = (prevPatches != 0u) ? 1u : 0u;
        }
        else if (nbContacts != 0u)
        {
            cpOut = patchIndex;
        }
        else
            patchIndex = 0xFFFFFFFFu;
    }

    r.counters = uint4(contactsBytes, forceBytes, patchesBytes, ovf);
    r.outputs = uint4(outForces, outPoints, cpOut, nbOut);
    r.flags = uint4(touchChange, patchChange, allflagsOut, patchIndex);
    r.patchArgs = float4(s.normal.xyz, float(nbContacts)); // insertIntoPatchStream args (tranche-45 shader)
    gOut[t] = r;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ccoWriteContact(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const CoreOutIn s = gIn[t];

    uint numPatches = s.misc.z & 0xFFu;  // PxU8(contact.numPatches())
    uint numPoints = s.misc.w & 0xFFFFu; // PxU16(contact.numPoints())
    const uint nbRequestedPatches = numPatches; // 5.12 (cu :296)
    const uint allflags = s.misc.y;

    uint contactsBytes = s.counters.x, forceBytes = s.counters.y, patchesBytes = s.counters.z;
    uint ovf = 0u;

    CoreOutOut r;
    for (uint j = 0u; j < 64u; ++j) { r.stream[j] = float4(0, 0, 0, 0); r.patchImg[j] = uint4(0, 0, 0, 0); }

    uint contactByteOffset = 0xFFFFFFFFu;
    uint forceByteOffset = 0xFFFFFFFFu;
    if (numPoints != 0u)
    {
        contactByteOffset = contactsBytes;
        contactsBytes += 16u * numPoints;
        forceByteOffset = forceBytes;
        forceBytes += 4u * numPoints;
        // 5.12 (cu :304-315): whole reservation; force overflow drops the contacts too
        if ((uint64_t(contactByteOffset) + 16ULL * uint64_t(numPoints)) > uint64_t(s.cfg.x)) { ovf |= CONTACT_BUFFER_OVERFLOW; contactByteOffset = 0xFFFFFFFFu; }
        if ((uint64_t(forceByteOffset) + 4ULL * uint64_t(numPoints)) > uint64_t(s.cfg.y))
        {
            ovf |= FORCE_BUFFER_OVERFLOW; forceByteOffset = 0xFFFFFFFFu;
            contactByteOffset = 0xFFFFFFFFu;
        }
    }

    uint outForces, outPoints;
    if (contactByteOffset != 0xFFFFFFFFu)
    {
        // patch-major sequential stream write (baseContactStream++)
        uint slot = (contactByteOffset - s.counters.x) / 16u;
        for (uint i = 0u; i < numPatches; ++i)
            for (uint j = 0u; j < s.patchCounts[i / 4u][i & 3u]; ++j)
                r.stream[slot++] = s.points[i * 4u + j]; // patchPoint(i, j) = mPoints[i*MAX_PATCH_POINTS+j]
        outForces = forceByteOffset;
        outPoints = contactByteOffset;
    }
    else
    {
        outForces = NULL_SENTINEL;
        outPoints = NULL_SENTINEL;
        numPoints = 0u;  // 5.12 (cu :337-338)
        numPatches = 0u;
    }

    const uint oldStatusFlags = (allflags >> 16u) & 0xFFu;
    uint status = oldStatusFlags;
    status &= ~TOUCH_KNOWN;
    if (numPoints != 0u) status |= HAS_TOUCH; // gates on numPoints...
    else status |= HAS_NO_TOUCH;

    const uint prevPatches = (allflags & 0xFFFFu) >> 8u;
    const bool previouslyHadTouch = (oldStatusFlags & HAS_TOUCH) != 0u;
    const bool prevTouchKnown = (oldStatusFlags & TOUCH_KNOWN) != 0u;
    const bool currentlyHasTouch = numPatches > 0u; // ...but touch change gates on numPatches (cu L374)

    uint touchChange = ((previouslyHadTouch != currentlyHasTouch) || !prevTouchKnown) ? 1u : 0u;
    uint patchChange = (prevPatches != numPatches) ? 1u : 0u;
    uint allflagsOut = (((prevPatches << 8u) | status) << 16u) | (numPatches << 8u);
    uint nbOut = numPoints;

    // numPatches * 64-byte patch reservation with the in-gate rollback
    uint patchIndex = 0xFFFFFFFFu;
    uint cpOut = NULL_SENTINEL;
    if (nbRequestedPatches != 0u)
    {
        patchIndex = patchesBytes;
        patchesBytes += nbRequestedPatches * 64u;
        if ((uint64_t(patchIndex) + uint64_t(nbRequestedPatches) * 64ULL) > uint64_t(s.cfg.z))
        {
            ovf |= PATCH_BUFFER_OVERFLOW;
            patchIndex = 0xFFFFFFFFu;

            status &= ~TOUCH_KNOWN;
            status |= HAS_NO_TOUCH;
            allflagsOut = ((prevPatches << 8u) | status) << 16u; // numPatches = 0
            nbOut = 0u;
            touchChange = previouslyHadTouch ? 1u : 0u;
            patchChange = (prevPatches != 0u) ? 1u : 0u;
        }
        else if (numPatches != 0u)
        {
            cpOut = patchIndex;
        }
        else
            patchIndex = 0xFFFFFFFFu;
    }

    // inline PxContactPatch array write (cu L408-444), 4 uint4 words per patch:
    //   w0-w3   mMassModification (linear0, angular0, linear1, angular1) -- all four 1.0f
    //   w4-w6   normal; w7 restitution; w8 dynamicFriction; w9 staticFriction; w10 damping
    //   w11     startContactIndex(0:16) | nbContacts(16:24) | materialFlags(24:32)
    //   w12     internalFlags(=0)(0:16) | materialIndex0(16:32)
    //   w13     materialIndex1(0:16); w14/w15 pad (untouched by the CUDA)
    if (patchIndex != 0xFFFFFFFFu)
    {
        float staFriction, dynFriction, restitution, damping;
        uint matFlags;
        combineMaterialsQ(s.mat0F, s.mat0U, s.mat1F, s.mat1U,
                          staFriction, dynFriction, restitution, damping, matFlags);

        uint startContactIndex = 0u;
        for (uint i = 0u; i < numPatches; ++i)
        {
            const uint nbPatchPoints = s.patchCounts[i / 4u][i & 3u];
            const float3 n = s.patchN[i].xyz;
            r.patchImg[i * 4u + 0u] = uint4(asuint(1.0f), asuint(1.0f), asuint(1.0f), asuint(1.0f));
            r.patchImg[i * 4u + 1u] = uint4(asuint(n.x), asuint(n.y), asuint(n.z), asuint(restitution));
            r.patchImg[i * 4u + 2u] = uint4(asuint(dynFriction), asuint(staFriction), asuint(damping),
                                            (startContactIndex & 0xFFFFu) | ((nbPatchPoints & 0xFFu) << 16u) | ((matFlags & 0xFFu) << 24u));
            r.patchImg[i * 4u + 3u] = uint4(((s.mat.x & 0xFFFFu) << 16u), // internalFlags = 0 in the low half
                                            (s.mat.y & 0xFFFFu), 0u, 0u);
            startContactIndex += nbPatchPoints;
        }
    }

    r.counters = uint4(contactsBytes, forceBytes, patchesBytes, ovf);
    r.outputs = uint4(outForces, outPoints, cpOut, nbOut);
    r.flags = uint4(touchChange, patchChange, allflagsOut, patchIndex);
    r.patchArgs = float4(0, 0, 0, 0);
    gOut[t] = r;
}
