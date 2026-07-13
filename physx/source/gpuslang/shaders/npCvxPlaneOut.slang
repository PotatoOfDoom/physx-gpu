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

// npCvxPlaneOut.slang
//
// Slang port of the convexPlaneNphase_kernel_core output tail (cudaGJKEPA.cu
// L3072-3181).  Differs from the convex-convex finish path in several quirks,
// all ported verbatim:
//   - the contact/force reservations add sizeof(PxContact)*nb / 4*nb, but the
//     overflow tests only add ONE element size ((offset + 16) > limit and
//     (offset + 4) > limit -- L3080/L3086) and are INDEPENDENT ifs (no
//     else-if: both flags can set in one pass);
//   - on a valid contact offset the masked lanes write the WORLD-space point
//     (planeTransform.transform(p), sep) at warpScanExclusive slots and the
//     outputs get startForces + forceOffset / startPoints + contactOffset (the
//     force offset can be the wrapped 0xFFFFFFFF value when only the force
//     reservation failed); on contact overflow BOTH output pointers are NULL;
//   - allflags/touch/patch-change logic as in the finish kernel (clear
//     eTOUCH_KNOWN, set eHAS_TOUCH/eHAS_NO_TOUCH, rewrite
//     ((prevPatches<<8|status)<<16)|(numPatches<<8), nbContacts store);
//   - patch reservation (64 bytes, overflow (patchIndex + 64) > limit) with
//     the rollback INSIDE the lane-0 gate: PATCH_BUFFER_OVERFLOW flag,
//     HAS_NO_TOUCH status rewrite with numPatches = 0, nbContacts = 0,
//     touchChange = previouslyHadTouch, patchChange = prevPatches != 0 -- no
//     ballot rollback (single patch, single lane); else contactPatches =
//     startPatches + patchIndex;
//   - lane 0 then calls insertIntoPatchStream(materials, patchStream,
//     planeShape, convexShape, patchIndex, planeTransform.rotate(localNormal),
//     nbContacts) -- the stream write itself is the separately-verified
//     tranche-45 shader, so this port emits the call arguments (patchIndex,
//     the world normal, the surviving nbContacts and the material indices).
//
// One pair per test.  NULL pointers are modelled as the 0xFFFFFFFE sentinel
// (distinct from the 0xFFFFFFFF overflow value that can legitimately wrap).

static const uint HAS_NO_TOUCH = 1u;
static const uint HAS_TOUCH = 2u;
static const uint TOUCH_KNOWN = 3u;
static const uint CONTACT_BUFFER_OVERFLOW = 1u;
static const uint FORCE_BUFFER_OVERFLOW = 2u;
static const uint PATCH_BUFFER_OVERFLOW = 4u;
static const uint NULL_SENTINEL = 0xFFFFFFFEu;

float3 quatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2);
}

struct PlaneOutIn
{
    float4 planeTq, planeTp;  // plane transform
    float4 p[32];             // per-lane p (.xyz) / sep (.w) from 155b/c
    uint4 misc;               // x = mask, y = nbContacts, z = allflags word, w = materialIndices packed (mat0 | mat1<<16)
    uint4 cfg;                // x = contactBytesLimit, y = forceBytesLimit, z = patchBytesLimit, w pad
    uint4 counters;           // x = initial contactsBytes, y = initial forceBytes, z = initial patchesBytes, w pad
}

struct PlaneOutOut
{
    float4 stream[4];         // world contact stream slots (relative to initial contactsBytes)
    uint4 counters;           // x = contactsBytes after, y = forceBytes after, z = patchesBytes after, w = overflow flags
    uint4 outputs;            // x = contactForces offset (NULL_SENTINEL when NULL), y = contactPoints offset,
                              // z = contactPatches offset (NULL_SENTINEL when unwritten), w = nbContacts stored
    uint4 flags;              // x = touchChange, y = patchChange, z = allflags after, w = patchIndex arg
    float4 patchArgs;         // world normal (.xyz) passed to insertIntoPatchStream; .w = nbContacts arg
}

struct PlaneOutParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<PlaneOutParams> gParams;

RWStructuredBuffer<PlaneOutIn> gIn;    // binding 0
RWStructuredBuffer<PlaneOutOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxPlaneOut(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const PlaneOutIn s = gIn[t];

    const uint mask = s.misc.x;
    uint nbContacts = s.misc.y;
    const uint allflags = s.misc.z;
    const float4 planeTq = s.planeTq;
    const float3 planeTp = s.planeTp.xyz;

    uint contactsBytes = s.counters.x, forceBytes = s.counters.y, patchesBytes = s.counters.z;
    uint ovf = 0u;

    PlaneOutOut r;
    for (uint j = 0u; j < 4u; ++j) r.stream[j] = float4(0, 0, 0, 0);

    // contact/force reservation (lane 0), single-element overflow tests, NO else-if
    uint contactByteOffset = 0xFFFFFFFFu;
    uint forceByteOffset = 0xFFFFFFFFu;
    if (nbContacts != 0u)
    {
        contactByteOffset = contactsBytes;
        contactsBytes += 16u * nbContacts;
        forceByteOffset = forceBytes;
        forceBytes += 4u * nbContacts;
        if ((contactByteOffset + 16u) > s.cfg.x) { ovf |= CONTACT_BUFFER_OVERFLOW; contactByteOffset = 0xFFFFFFFFu; }
        if ((forceByteOffset + 4u) > s.cfg.y) { ovf |= FORCE_BUFFER_OVERFLOW; forceByteOffset = 0xFFFFFFFFu; }
    }

    uint outForces, outPoints;
    if (contactByteOffset != 0xFFFFFFFFu)
    {
        const uint streamBase = s.counters.x;
        for (uint L = 0u; L < 32u; ++L)
        {
            if ((mask & (1u << L)) != 0u)
            {
                const uint index = countbits(mask & ((1u << L) - 1u)); // warpScanExclusive
                const float3 worldP = quatRotate(planeTq, s.p[L].xyz) + planeTp;
                r.stream[(contactByteOffset - streamBase) / 16u + index] = float4(worldP, s.p[L].w);
            }
        }
        outForces = forceByteOffset; // may be the wrapped 0xFFFFFFFF -- ported exactly
        outPoints = contactByteOffset;
    }
    else
    {
        outForces = NULL_SENTINEL; // NULL
        outPoints = NULL_SENTINEL;
    }

    // allflags / touch / patch change
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

    // patch reservation (lane 0), rollback inside the gate
    uint patchIndex = 0xFFFFFFFFu;
    uint cpOut = NULL_SENTINEL;
    if (nbContacts != 0u)
    {
        patchIndex = patchesBytes;
        patchesBytes += 64u; // sizeof(PxContactPatch)
        if ((patchIndex + 64u) > s.cfg.z)
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
        else
        {
            cpOut = patchIndex;
        }
    }

    // insertIntoPatchStream args (the stream write is the tranche-45 shader)
    const float3 worldNormal = quatRotate(planeTq, float3(1, 0, 0));

    r.counters = uint4(contactsBytes, forceBytes, patchesBytes, ovf);
    r.outputs = uint4(outForces, outPoints, cpOut, nbOut);
    r.flags = uint4(touchChange, patchChange, allflagsOut, patchIndex);
    r.patchArgs = float4(worldNormal, float(nbContacts));
    gOut[t] = r;
}
