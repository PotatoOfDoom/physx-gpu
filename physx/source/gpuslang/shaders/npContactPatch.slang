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

// npContactPatch.slang
//
// Slang port of the narrow-phase contact-stream allocation glue from
// gpunarrowphase/src/CUDA/contactPatchUtils.cuh:
//   - setContactPointAndForcePointers: atomically reserves the per-pair contact
//     and force/indice byte ranges from the shared counters (with overflow
//     detection), returning the contact byte offset.
//   - registerContactPatch (5.12: keeps the patch reservation of a pair whose
//     contact or force allocation failed, but reports it with no contacts):
//     recomputes the per-pair status-flag word packed into
//     PxsContactManagerOutput (allflagsStart / nbPatches / statusFlag /
//     prevPatches), the touch- and patch-change flags, the contact count, and
//     atomically reserves the contact patch.
//
// The packed status word is the uint32 aliased over the four bytes
// {allflagsStart, nbPatches, statusFlag, prevPatches} of PxsContactManagerOutput
// (little-endian: byte0=allflagsStart, byte1=nbPatches, byte2=statusFlag,
// byte3=prevPatches).  The struct's device pointers become byte offsets here; the
// per-pair flag computation is deterministic, while the atomic byte/patch
// reservations form a valid partition regardless of thread ordering.
//
// gCounters layout: [0]=patchesBytes, [1]=contactsBytes, [2]=forceAndIndiceBytes,
//                    [3]=overflowError.  sizeof(PxContact)/PxU32/PxContactPatch
// are supplied as push constants (the allocation is size-agnostic).

struct ContactPatchParams
{
    uint numTests;
    uint contactSize;       // sizeof(PxContact)
    uint forceSize;         // sizeof(PxU32)
    uint patchSize;         // sizeof(PxContactPatch)
    uint contactBytesLimit;
    uint forceBytesLimit;
    uint patchBytesLimit;
}

[[vk::push_constant]]
ConstantBuffer<ContactPatchParams> gParams;

RWStructuredBuffer<uint> gNbContacts;    // binding 0  nbContacts per pair (input)
RWStructuredBuffer<uint> gCounters;      // binding 1  [patchesBytes, contactsBytes, forceBytes, overflow]
RWStructuredBuffer<uint> gPriorAllflags; // binding 2  packed status word per pair (input, register stage)
RWStructuredBuffer<uint> gOutAllflags;   // binding 3  packed status word per pair (output)
RWStructuredBuffer<uint> gTouchChange;   // binding 4
RWStructuredBuffer<uint> gPatchChange;   // binding 5
RWStructuredBuffer<uint> gContactOffset; // binding 6  returned contact byte offset (set stage)
RWStructuredBuffer<uint> gForceOffset;   // binding 7  returned force byte offset (set stage)
RWStructuredBuffer<uint> gPatchIndex;    // binding 8  returned patch byte offset (register stage)

static const uint OVERFLOW_CONTACT = 1u; // PxgPatchAndContactCounters::CONTACT_BUFFER_OVERFLOW
static const uint OVERFLOW_FORCE = 2u;   // FORCE_BUFFER_OVERFLOW
static const uint OVERFLOW_PATCH = 4u;   // PATCH_BUFFER_OVERFLOW

// PxsContactManagerStatusFlag
static const uint FLAG_HAS_NO_TOUCH = 1u;
static const uint FLAG_HAS_TOUCH = 2u;
static const uint FLAG_TOUCH_KNOWN = 3u; // eHAS_NO_TOUCH | eHAS_TOUCH

static const uint INVALID = 0xffffffffu;

// bit helpers (cutil_math.h)
uint u32High(uint v) { return (v >> 16) & 0xffffu; }
uint u32Low(uint v) { return v & 0xffffu; }
uint u16High(uint v) { return (v >> 8) & 0xffu; }
uint u16Low(uint v) { return v & 0xffu; }
uint mergeU16(uint hi, uint lo) { return ((hi & 0xffffu) << 16) | (lo & 0xffffu); }
uint mergeU8(uint hi, uint lo) { return ((hi & 0xffu) << 8) | (lo & 0xffu); }

// ---------------------------------------------------------------------------
// setContactPointAndForcePointers
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void setContactPointAndForcePointers(uint3 dtid : SV_DispatchThreadID)
{
    const uint workIndex = dtid.x;
    if (workIndex >= gParams.numTests) return;

    const uint nbContacts = gNbContacts[workIndex];
    uint contactByteOffset = INVALID;
    uint forceByteOffset = INVALID;

    if (nbContacts != 0u)
    {
        InterlockedAdd(gCounters[1], gParams.contactSize * nbContacts, contactByteOffset);
        InterlockedAdd(gCounters[2], gParams.forceSize * nbContacts, forceByteOffset);

        // 5.12: the WHOLE reservation is tested (size_t sums in the CUDA), and
        // a force overflow also invalidates the contact offset.
        if ((uint64_t(contactByteOffset) + uint64_t(gParams.contactSize) * uint64_t(nbContacts)) > uint64_t(gParams.contactBytesLimit))
        {
            InterlockedOr(gCounters[3], OVERFLOW_CONTACT);
            contactByteOffset = INVALID;
        }
        if ((uint64_t(forceByteOffset) + uint64_t(gParams.forceSize) * uint64_t(nbContacts)) > uint64_t(gParams.forceBytesLimit))
        {
            InterlockedOr(gCounters[3], OVERFLOW_FORCE);
            forceByteOffset = INVALID;
            contactByteOffset = INVALID;
        }
    }

    gContactOffset[workIndex] = contactByteOffset;
    gForceOffset[workIndex] = forceByteOffset;
}

// ---------------------------------------------------------------------------
// registerContactPatch
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void registerContactPatch(uint3 dtid : SV_DispatchThreadID)
{
    const uint workIndex = dtid.x;
    if (workIndex >= gParams.numTests) return;

    // 5.12: preserve patch demand even if contact or force allocation failed.
    // The CUDA tests cmOutputs[workIndex].contactPoints / contactForces for
    // NULL; setContactPointAndForcePointers NULLs both exactly when it returns
    // an invalid contact offset, which is what the set stage left in
    // gContactOffset (this stage runs after it on the same buffers).
    uint nbContacts = gNbContacts[workIndex];
    const bool patchRequested = nbContacts > 0u;
    if (patchRequested && gContactOffset[workIndex] == INVALID)
        nbContacts = 0u;
    const uint allflags = gPriorAllflags[workIndex];

    const uint oldStatusFlags = u16Low(u32High(allflags));
    uint statusFlags = oldStatusFlags & (~FLAG_TOUCH_KNOWN);
    if (nbContacts != 0u)
        statusFlags |= FLAG_HAS_TOUCH;
    else
        statusFlags |= FLAG_HAS_NO_TOUCH;

    const uint prevPatches = u16High(u32Low(allflags));
    const bool previouslyHadTouch = (oldStatusFlags & FLAG_HAS_TOUCH) != 0u;
    const bool prevTouchKnown = (oldStatusFlags & FLAG_TOUCH_KNOWN) != 0u;

    const uint numPatches = (nbContacts > 0u) ? 1u : 0u;
    const bool currentlyHasTouch = nbContacts > 0u;

    const bool change = (previouslyHadTouch != currentlyHasTouch) || (!prevTouchKnown);
    gTouchChange[workIndex] = change ? 1u : 0u;
    gPatchChange[workIndex] = (prevPatches != numPatches) ? 1u : 0u;

    gOutAllflags[workIndex] = mergeU16(mergeU8(prevPatches, statusFlags), mergeU8(numPatches, 0u));

    uint patchIndex = INVALID;
    if (patchRequested)
    {
        InterlockedAdd(gCounters[0], gParams.patchSize, patchIndex);
        if ((patchIndex + gParams.patchSize) > gParams.patchBytesLimit)
        {
            InterlockedOr(gCounters[3], OVERFLOW_PATCH);
            patchIndex = INVALID;

            statusFlags &= (~FLAG_TOUCH_KNOWN);
            statusFlags |= FLAG_HAS_NO_TOUCH;
            gOutAllflags[workIndex] = mergeU16(mergeU8(prevPatches, statusFlags), 0u);

            gTouchChange[workIndex] = previouslyHadTouch ? 1u : 0u;
            gPatchChange[workIndex] = (prevPatches != 0u) ? 1u : 0u;
        }
        else if (nbContacts == 0u)
            patchIndex = INVALID; // reserved, but the pair has no contacts to patch
    }
    gPatchIndex[workIndex] = patchIndex;
}
