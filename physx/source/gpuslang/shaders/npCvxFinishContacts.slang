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

// npCvxFinishContacts.slang
//
// Slang port of the finishContactsKernel tail (cudaGJKEPA.cu L2052-2167) plus
// writePatchesToStream4threads (L1942-2007) and the patchConstants/patchOffsets
// tables: everything that runs after the tranche-151 contact-stream write.
// One test = one warp (32 pairs).  Per lane:
//   - scratch contactParams.startIdx = 0, materialParams.internalFlags = 0;
//   - allflags (cmOutputs->allflagsStart word) decode: oldStatusFlags = byte 2
//     (u16Low(u32High)), prevPatches = byte 1 (u16High(u32Low)); statusFlags
//     clear eTOUCH_KNOWN (= eHAS_NO_TOUCH|eHAS_TOUCH = 3), then set eHAS_TOUCH
//     (2) when nbContacts != 0 else eHAS_NO_TOUCH (1);
//   - valid lanes write touchChangeFlags = (previouslyHadTouch ^
//     currentlyHasTouch) || !prevTouchKnown, patchChangeFlags = (prevPatches !=
//     numPatches) with numPatches = nbContacts ? 1 : 0, then allflagsStart =
//     merge(merge(prevPatches, statusFlags), merge(numPatches, 0)) =
//     ((prevPatches<<8 | status) << 16) | (numPatches << 8) and
//     cmOutputs.nbContacts = nbContacts;
//   - patchMask = ballot(nbContacts > 0); warp lane 31 reserves
//     sizeof(PxContactPatch)=64 bytes per patch; overflow when (patchIndex +
//     totalSz) is STRICTLY > patchBytesLimit (unlike the contact stream's >=),
//     setting PATCH_BUFFER_OVERFLOW (4) and 0xFFFFFFFF;
//   - on overflow with a non-empty mask: patchMask = 0, scratch nbContacts = 0,
//     statusFlags forced to eHAS_NO_TOUCH, allflagsStart rewritten with
//     numPatches = 0, cmOutputs.nbContacts = 0, touchChangeFlags =
//     previouslyHadTouch, patchChangeFlags = prevPatches != 0;
//   - valid && patchIndex valid: cmOutputs->contactPatches = patchIndex +
//     64 * popc(patchMask & belowLane);
//   - combineMaterials (PxsCombineMaterials, the tranche-43 core) on the
//     scratch material indices -> restitution/dynamicFriction/staticFriction/
//     damping (float bits) + materialFlags (u8) in scratch;
//   - writePatchesToStream4threads: each masked lane emits one 64-byte
//     PxContactPatch as 4 uint4s via the patchOffsets table: [0] = the four
//     1.0f invMass/invInertia scale constants, [1] = normx/normy/normz/
//     restitution, [2] = dynamicFriction/staticFriction/damping/contactUnion
//     (startIdx | nbContacts<<16 | materialFlags<<24), [3] = materialUnion
//     (internalFlags | materialIndex0<<16), materialUnion2 (materialIndex1,
//     pad modelled 0), 0.0f, 0.0f.
//
// Invalid lanes: CUDA leaves their combine outputs as uninitialized registers
// stored to scratch; modelled as 0 (never read back by valid outputs).

static const uint HAS_NO_TOUCH = 1u;
static const uint HAS_TOUCH = 2u;
static const uint TOUCH_KNOWN = 3u; // eHAS_NO_TOUCH | eHAS_TOUCH
static const uint PATCH_BUFFER_OVERFLOW = 4u;

static const uint MODE_AVERAGE = 0u, MODE_MIN = 1u, MODE_MULTIPLY = 2u, MODE_MAX = 3u;
static const uint FLAG_DISABLE_FRICTION = 1u, FLAG_DISABLE_STRONG_FRICTION = 2u, FLAG_COMPLIANT_ACC_SPRING = 16u;

struct MaterialData
{
    float dynamicFriction;
    float staticFriction;
    float restitution;
    float damping;
    uint flags;
    uint fricCombineMode;
    uint restCombineMode;
    uint dampingCombineMode;
}

float combineScalars(float a, float b, uint mode)
{
    if (mode == MODE_AVERAGE) return 0.5f * (a + b);
    if (mode == MODE_MIN) return min(a, b);
    if (mode == MODE_MULTIPLY) return a * b;
    if (mode == MODE_MAX) return max(a, b);
    return 0.0f;
}

void combineMaterials(MaterialData m0, MaterialData m1,
                      out uint materialFlags, out float staFriction, out float dynFriction,
                      out float restitution, out float damping)
{
    const float r0 = m0.restitution;
    const float r1 = m1.restitution;
    const bool compliant0 = r0 < 0.0f;
    const bool compliant1 = r1 < 0.0f;
    const bool exactlyOneCompliant = compliant0 != compliant1;
    const bool bothCompliant = compliant0 && compliant1;
    const bool compliantAcc0 = (m0.flags & FLAG_COMPLIANT_ACC_SPRING) != 0u;
    const bool compliantAcc1 = (m1.flags & FLAG_COMPLIANT_ACC_SPRING) != 0u;
    const bool exactlyOneAccCompliant = compliantAcc0 != compliantAcc1;

    if (bothCompliant && exactlyOneAccCompliant)
    {
        restitution = compliantAcc0 ? r0 : r1;
    }
    else
    {
        const uint combineMode = exactlyOneCompliant ? MODE_MIN : max(m0.restCombineMode, m1.restCombineMode);
        const float flipSign = (bothCompliant && (combineMode == MODE_MULTIPLY)) ? -1.0f : 1.0f;
        restitution = flipSign * combineScalars(r0, r1, combineMode);
    }

    {
        const float d0 = m0.damping;
        const float d1 = m1.damping;
        if (bothCompliant && exactlyOneAccCompliant)
        {
            damping = compliantAcc0 ? d0 : d1;
        }
        else
        {
            const uint combineMode = exactlyOneCompliant ? MODE_MAX : max(m0.dampingCombineMode, m1.dampingCombineMode);
            damping = combineScalars(d0, d1, combineMode);
        }
    }

    {
        const uint combineFlags = m0.flags | m1.flags;
        if ((combineFlags & FLAG_DISABLE_FRICTION) == 0u)
        {
            const uint frictionCombineMode = max(m0.fricCombineMode, m1.fricCombineMode);
            const float dynF = combineScalars(m0.dynamicFriction, m1.dynamicFriction, frictionCombineMode);
            const float staF = combineScalars(m0.staticFriction, m1.staticFriction, frictionCombineMode);
            const float fDynFriction = max(dynF, 0.0f);
            const float fStaFriction = (staF - fDynFriction) >= 0.0f ? staF : fDynFriction;
            dynFriction = fDynFriction;
            staFriction = fStaFriction;
            materialFlags = combineFlags;
        }
        else
        {
            materialFlags = combineFlags | FLAG_DISABLE_STRONG_FRICTION;
            dynFriction = 0.0f;
            staFriction = 0.0f;
        }
    }
}

struct FinishIn
{
    // per lane: x = nbContacts (scratch, from the stream write)
    //           y = allflagsStart word (cmOutputs)
    //           z = valid (elementGlobalIndex < numTests)
    //           w = materialIndex0
    uint4 lane[32];
    // per lane: x = materialIndex1, y/z/w = norm x/y/z float bits (scratch)
    uint4 lane2[32];
    uint4 cfg; // x = patchBytesLimit, y = initial patchesBytes, z/w pad
}

struct FinishOut
{
    // per lane: x = touchChangeFlag, y = patchChangeFlag,
    //           z = allflagsStart after, w = nbContacts after
    uint4 lane[32];
    // per lane: x = contactPatches byte offset (0xFFFFFFFF sentinel when unset),
    //           y = scratch materialFlags (u8), z/w pad
    uint4 lane2[32];
    // per lane: combined restitution / dynamicFriction / staticFriction / damping
    float4 combined[32];
    uint4 counters; // x = patchesBytes after, y = overflowError, z/w pad
    uint4 patches[128]; // 32 patches x 4 uint4 (64 bytes each), relative to initial patchesBytes
}

struct FinishParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<FinishParams> gParams;

RWStructuredBuffer<FinishIn> gIn;           // binding 0
RWStructuredBuffer<FinishOut> gOut;         // binding 1
RWStructuredBuffer<MaterialData> gMaterials; // binding 2

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxFinishContacts(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const FinishIn s = gIn[t];

    const uint patchBytesLimit = s.cfg.x;
    uint patchesBytes = s.cfg.y;
    uint overflowError = 0u;

    FinishOut r;
    for (uint i = 0u; i < 128u; ++i) r.patches[i] = uint4(0, 0, 0, 0);

    uint scratchNb[32];    // sscratch.contactParams[..].nbContacts
    uint statusFlags[32];
    uint prevPatches[32];
    bool previouslyHadTouch[32];
    uint touchChange[32], patchChange[32], allflagsOut[32], nbOut[32];
    uint cpOffset[32];

    for (uint L = 0u; L < 32u; ++L)
    {
        const bool valid = s.lane[L].z != 0u;
        const uint nbContacts = valid ? s.lane[L].x : 0u;
        scratchNb[L] = s.lane[L].x; // scratch holds the stream-write value regardless

        uint allflags = 0u;
        if (valid)
            allflags = s.lane[L].y;

        const uint oldStatusFlags = (allflags >> 16u) & 0xFFu; // u16Low(u32High)
        uint status = oldStatusFlags;
        status &= ~TOUCH_KNOWN;
        if (nbContacts != 0u) status |= HAS_TOUCH;
        else status |= HAS_NO_TOUCH;

        prevPatches[L] = (allflags & 0xFFFFu) >> 8u; // u16High(u32Low)
        previouslyHadTouch[L] = (oldStatusFlags & HAS_TOUCH) != 0u;
        const bool prevTouchKnown = (oldStatusFlags & TOUCH_KNOWN) != 0u;

        touchChange[L] = 0u; patchChange[L] = 0u; allflagsOut[L] = s.lane[L].y; nbOut[L] = 0u;
        cpOffset[L] = 0xFFFFFFFFu;

        if (valid)
        {
            const uint numPatches = (nbContacts != 0u) ? 1u : 0u;
            const bool currentlyHasTouch = nbContacts != 0u;
            const bool change = (previouslyHadTouch[L] != currentlyHasTouch) || (!prevTouchKnown);
            touchChange[L] = change ? 1u : 0u;
            patchChange[L] = (prevPatches[L] != numPatches) ? 1u : 0u;
            allflagsOut[L] = (((prevPatches[L] << 8u) | status) << 16u) | (numPatches << 8u);
            nbOut[L] = nbContacts;
        }
        statusFlags[L] = status;
    }

    // patchMask ballot + lane-31 reservation
    uint patchMask = 0u;
    for (uint L = 0u; L < 32u; ++L)
    {
        const bool valid = s.lane[L].z != 0u;
        const uint nbContacts = valid ? s.lane[L].x : 0u;
        if (nbContacts > 0u) patchMask |= (1u << L);
    }

    uint patchIndex = 0xFFFFFFFFu;
    if (patchMask != 0u)
    {
        const uint totalSz = 64u * countbits(patchMask); // sizeof(PxContactPatch)
        patchIndex = patchesBytes;
        patchesBytes += totalSz;
        if ((patchIndex + totalSz) > patchBytesLimit) // STRICTLY > (unlike the contact stream)
        {
            overflowError |= PATCH_BUFFER_OVERFLOW;
            patchIndex = 0xFFFFFFFFu;
        }
    }

    if (patchIndex == 0xFFFFFFFFu && patchMask != 0u) // overflow rollback
    {
        patchMask = 0u;
        for (uint L = 0u; L < 32u; ++L)
        {
            scratchNb[L] = 0u;
            const bool valid = s.lane[L].z != 0u;
            if (valid)
            {
                uint status = statusFlags[L];
                status &= ~TOUCH_KNOWN;
                status |= HAS_NO_TOUCH;
                allflagsOut[L] = ((prevPatches[L] << 8u) | status) << 16u; // numPatches = 0
                nbOut[L] = 0u;
                touchChange[L] = previouslyHadTouch[L] ? 1u : 0u;
                patchChange[L] = (prevPatches[L] != 0u) ? 1u : 0u;
                statusFlags[L] = status;
            }
        }
    }

    for (uint L = 0u; L < 32u; ++L)
    {
        const bool valid = s.lane[L].z != 0u;
        if (valid && patchIndex != 0xFFFFFFFFu)
            cpOffset[L] = patchIndex + 64u * countbits(patchMask & ((1u << L) - 1u));
    }

    // combineMaterials into scratch (invalid lanes modelled 0)
    float rest[32], dynF[32], staF[32], damp[32];
    uint matFlags[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        rest[L] = 0.0f; dynF[L] = 0.0f; staF[L] = 0.0f; damp[L] = 0.0f; matFlags[L] = 0u;
        const bool valid = s.lane[L].z != 0u;
        if (valid)
        {
            const MaterialData m0 = gMaterials[s.lane[L].w];
            const MaterialData m1 = gMaterials[s.lane2[L].x];
            uint mf; float sf, df, re, da;
            combineMaterials(m0, m1, mf, sf, df, re, da);
            rest[L] = re; dynF[L] = df; staF[L] = sf; damp[L] = da;
            matFlags[L] = mf & 0xFFu; // static_cast<PxU8>
        }
    }

    // writePatchesToStream4threads
    const uint ONE_BITS = asuint(1.0f);
    for (uint L = 0u; L < 32u; ++L)
    {
        if ((patchMask & (1u << L)) != 0u)
        {
            const uint slot = countbits(patchMask & ((1u << L) - 1u)); // patch index
            const uint base = slot * 4u;
            // contactUnion: startIdx (0) | nbContacts<<16 | materialFlags<<24
            const uint contactUnion = (scratchNb[L] & 0xFFu) << 16u | (matFlags[L] << 24u);
            // materialUnion: internalFlags (0) | materialIndex0<<16
            const uint materialUnion = (s.lane[L].w & 0xFFFFu) << 16u;
            // materialUnion2: materialIndex1 | pad (modelled 0) << 16
            const uint materialUnion2 = s.lane2[L].x & 0xFFFFu;
            r.patches[base + 0u] = uint4(ONE_BITS, ONE_BITS, ONE_BITS, ONE_BITS);
            r.patches[base + 1u] = uint4(s.lane2[L].y, s.lane2[L].z, s.lane2[L].w, asuint(rest[L]));
            r.patches[base + 2u] = uint4(asuint(dynF[L]), asuint(staF[L]), asuint(damp[L]), contactUnion);
            r.patches[base + 3u] = uint4(materialUnion, materialUnion2, 0u, 0u);
        }
    }

    for (uint L = 0u; L < 32u; ++L)
    {
        r.lane[L] = uint4(touchChange[L], patchChange[L], allflagsOut[L], nbOut[L]);
        r.lane2[L] = uint4(cpOffset[L], matFlags[L], 0u, 0u);
        r.combined[L] = float4(rest[L], dynF[L], staF[L], damp[L]);
    }
    r.counters = uint4(patchesBytes, overflowError, 0u, 0u);
    gOut[t] = r;
}
