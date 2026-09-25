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

// npSphereNphase.slang
//
// Slang port of sphereNphase_Kernel (gpunarrowphase/src/CUDA/cudaSphere.cu) on the
// ENGINE launch ABI (BDA).  This is the sphere/capsule/plane/box primitive-primitive
// narrow-phase dispatcher: one thread per contact manager reads the pair's two shape
// types, orders them (sphere/lower type first), calls the matching contact routine in
// npSpherePrimitives (the shared math also exercised by the harness sub-case entries),
// negates the normal for the flipped ordering, and emits contacts/patches to the
// contact- and patch-streams with the atomic offset reservation on
// PxgPatchAndContactCounters — mirroring contactPatchUtils.cuh
// (setContactPointAndForcePointers / registerContactPatch / insertIntoPatchStream).
//
// Unlike the 4-lane box-box kernel this is embarrassingly parallel (one contact
// manager per thread, no groupshared / barriers), so the CUDA's early out-of-range
// return is kept as a real early return.  Element structs read/written through device
// pointers (PxgContactManagerInput / PxgShape / PxsCachedTransform /
// PxsContactManagerOutput / PxgPatchAndContactCounters / PxsMaterialData / the
// contact + patch stream byte layout) mirror the engine C++ layouts byte-for-byte and
// are the SAME mirrors the box-box kernel uses.

import pxgSlangCommon;       // PxgShape, PxsCachedTransform, PXG_GEOM_*, quat helpers
import npSpherePrimitives;   // PrimContact + the six sphere/capsule contact routines

static const uint  INVALID = 0xffffffffu;
static const uint  CONTACT_SIZE = 16u;        // sizeof(PxContact)
static const uint  FORCE_SIZE = 4u;           // sizeof(PxU32)
static const uint  PATCH_SIZE = 64u;          // sizeof(PxContactPatch)

// PxgPatchAndContactCounters::OverflowError
static const uint OVERFLOW_CONTACT = 1u;
static const uint OVERFLOW_FORCE = 2u;
static const uint OVERFLOW_PATCH = 4u;

// PxsContactManagerStatusFlag
static const uint FLAG_HAS_NO_TOUCH = 1u;
static const uint FLAG_HAS_TOUCH = 2u;
static const uint FLAG_TOUCH_KNOWN = 3u; // eHAS_NO_TOUCH | eHAS_TOUCH

// PxMaterialFlag / PxCombineMode
static const uint MODE_AVERAGE = 0u, MODE_MIN = 1u, MODE_MULTIPLY = 2u, MODE_MAX = 3u;
static const uint MAT_DISABLE_FRICTION = 1u, MAT_DISABLE_STRONG_FRICTION = 2u, MAT_COMPLIANT_ACC_SPRING = 16u;

// ---------------------------------------------------------------------------
// Element-struct mirrors (byte-exact vs the engine C++ layouts; identical to the
// mirrors in boxBoxNphase.slang).
// ---------------------------------------------------------------------------

// PxgContactManagerInput (PxgContactManager.h, PX_ALIGN(16), 16 bytes)
struct PxgContactManagerInput
{
    uint shapeRef0;
    uint shapeRef1;
    uint transformCacheRef0;
    uint transformCacheRef1;
}

// PxgPatchAndContactCounters (cudaNpCommon.h, PX_ALIGN(16), 16 bytes)
struct PxgPatchAndContactCounters
{
    uint patchesBytes;
    uint contactsBytes;
    uint forceAndIndiceBytes;
    uint overflowError;
}

// PxsContactManagerOutput (PxsContactManagerState.h, PX_ALIGN(16), 48 bytes).
struct PxsContactManagerOutput
{
    uint64_t contactPatches;    // @0
    uint64_t contactPoints;     // @8
    uint64_t contactForces;     // @16
    uint64_t frictionPatches;   // @24
    uint allflagsWord;          // @32 : allflagsStart(0:8)|nbPatches(8:16)|statusFlag(16:24)|prevPatches(24:32)
    uint nbContacts_flags;      // @36 : nbContacts(0:16)|flags(16:32)
    uint2 pad;                  // @40 : PxU8 pad[8]
}

// PxsMaterialData (PxsMaterialCore.h, 24 bytes, align 4).
struct PxsMaterialData
{
    float dynamicFriction;      // @0
    float staticFriction;       // @4
    float restitution;          // @8
    float damping;              // @12
    uint packedFlags;           // @16 : flags(0:16 PxU16) | fricCombineMode(16:24) | restCombineMode(24:32)
    uint packedDamp;            // @20 : dampingCombineMode(0:8) | pad
}

// ---------------------------------------------------------------------------
// Push constant: mirrors the CUDA __global__ parameter list EXACTLY.
// numTests(4) @0, toleranceLength(4) @4, then 14 device pointers (8-aligned) @8..112,
// then the three PxU32 byte-limits @120,124,128 => 136 bytes (maxAlign 8).
// ---------------------------------------------------------------------------
struct SphereNphaseParams
{
    uint numTests;                    // @0
    float toleranceLength;            // @4  (unused by this kernel, present for ABI parity)
    uint64_t cmInputs;                // @8   const PxgContactManagerInput*
    uint64_t cmOutputs;               // @16  PxsContactManagerOutput*
    uint64_t shapes;                  // @24  PxgShape*
    uint64_t transformCache;          // @32  const PxsCachedTransform*
    uint64_t contactDistance;         // @40  const PxReal*
    uint64_t materials;               // @48  const PxsMaterialData*
    uint64_t contactStream;           // @56  PxU8*
    uint64_t patchStream;             // @64  PxU8*
    uint64_t patchAndContactCounters; // @72  PxgPatchAndContactCounters*
    uint64_t touchChangeFlags;        // @80  PxU32*
    uint64_t patchChangeFlags;        // @88  PxU32*
    uint64_t startContactPatches;     // @96  PxU8*
    uint64_t startContactPoints;      // @104 PxU8*
    uint64_t startContactForces;      // @112 PxU8*
    uint patchBytesLimit;             // @120
    uint contactBytesLimit;           // @124
    uint forceBytesLimit;             // @128
}                                     // = 136

[[vk::push_constant]]
ConstantBuffer<SphereNphaseParams> gParams;

// PxsCombineMaterials scalar combine.
float combineScalars(float a, float b, uint mode)
{
    if (mode == MODE_AVERAGE) return 0.5f * (a + b);
    if (mode == MODE_MIN) return min(a, b);
    if (mode == MODE_MULTIPLY) return a * b;
    if (mode == MODE_MAX) return max(a, b);
    return 0.0f;
}

// mirrors materialCombiner.cuh combineMaterials (same logic as boxBoxNphase.slang).
void combineMaterials(PxsMaterialData m0, PxsMaterialData m1,
                      out float sta, out float dyn, out float rest, out float damping, out uint matFlags)
{
    const float r0f = m0.restitution, r1f = m1.restitution;
    const bool compliant0 = r0f < 0.0f, compliant1 = r1f < 0.0f;
    const bool exactlyOneCompliant = compliant0 != compliant1;
    const bool bothCompliant = compliant0 && compliant1;
    const uint m0flags = m0.packedFlags & 0xffffu;
    const uint m1flags = m1.packedFlags & 0xffffu;
    const uint m0fric = (m0.packedFlags >> 16u) & 0xffu;
    const uint m1fric = (m1.packedFlags >> 16u) & 0xffu;
    const uint m0rest = (m0.packedFlags >> 24u) & 0xffu;
    const uint m1rest = (m1.packedFlags >> 24u) & 0xffu;
    const uint m0damp = m0.packedDamp & 0xffu;
    const uint m1damp = m1.packedDamp & 0xffu;
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

// ---------------------------------------------------------------------------
// sphereNphase_Kernel : sphere/capsule primitive narrow phase on the engine ABI.
// Grid: block 64, one thread per contact manager (SV_DispatchThreadID.x = workIndex).
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void sphereNphase_Kernel(uint3 dtid : SV_DispatchThreadID)
{
    const uint workIndex = dtid.x;
    if (workIndex >= gParams.numTests)
        return;

    // ---- device pointers ----
    Ptr<PxgContactManagerInput> cmInputs = Ptr<PxgContactManagerInput>(gParams.cmInputs);
    Ptr<PxsContactManagerOutput> cmOutputs = Ptr<PxsContactManagerOutput>(gParams.cmOutputs);
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gParams.shapes);
    Ptr<PxsCachedTransform> transformCache = Ptr<PxsCachedTransform>(gParams.transformCache);
    Ptr<float> contactDistance = Ptr<float>(gParams.contactDistance);
    Ptr<PxsMaterialData> materials = Ptr<PxsMaterialData>(gParams.materials);
    Ptr<PxgPatchAndContactCounters> counters = Ptr<PxgPatchAndContactCounters>(gParams.patchAndContactCounters);
    Ptr<uint> touchChangeFlags = Ptr<uint>(gParams.touchChangeFlags);
    Ptr<uint> patchChangeFlags = Ptr<uint>(gParams.patchChangeFlags);

    // ---- per-pair inputs ----
    const PxgContactManagerInput cin = cmInputs[workIndex];
    const uint tcRef0 = cin.transformCacheRef0;
    const uint tcRef1 = cin.transformCacheRef1;
    const PxsCachedTransform tc0 = transformCache[tcRef0];
    const PxsCachedTransform tc1 = transformCache[tcRef1];
    const float cDist = contactDistance[tcRef0] + contactDistance[tcRef1];

    // material indices come from the ORIGINAL (unswapped) shapes (as in the CUDA).
    const PxgShape shape0 = shapes[cin.shapeRef0];
    const PxgShape shape1 = shapes[cin.shapeRef1];

    uint type0 = shape0.type;
    uint type1 = shape1.type;
    float3 scale0 = shape0.scale();
    float3 scale1 = shape1.scale();
    float4 q0 = tc0.transform.q();
    float3 tp0 = tc0.transform.p();
    float4 q1 = tc1.transform.q();
    float3 tp1 = tc1.transform.p();

    // shape0 should always be the lower geometry type (sphere first).
    const bool flip = (type1 < type0);
    if (flip)
    {
        const uint tt = type0; type0 = type1; type1 = tt;
        const float3 ts = scale0; scale0 = scale1; scale1 = ts;
        const float4 tq = q0; q0 = q1; q1 = tq;
        const float3 tpp = tp0; tp0 = tp1; tp1 = tpp;
    }

    const float sphereRadius = scale0.x;

    PrimContact c;
    c.nbContacts = 0u;
    c.pointPen0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    c.pointPen1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    c.normal = float3(0.0f, 0.0f, 0.0f);

    if (type0 == PXG_GEOM_SPHERE)
    {
        if (type1 == PXG_GEOM_SPHERE)
            c = sphereSphereContact(q0, tp0, sphereRadius, q1, tp1, scale1.x, cDist);
        else if (type1 == PXG_GEOM_PLANE)
            c = spherePlaneContact(q0, tp0, sphereRadius, q1, tp1, cDist);
        else if (type1 == PXG_GEOM_CAPSULE)
            c = sphereCapsuleContact(q0, tp0, sphereRadius, q1, tp1, scale1.y, scale1.x, cDist);
        else if (type1 == PXG_GEOM_BOX)
            c = sphereBoxContact(q0, tp0, sphereRadius, q1, tp1, scale1, cDist);
    }
    else
    {
        // type0 != sphere => type1 must be capsule (CUDA asserts this).
        if (type0 == PXG_GEOM_PLANE)
            c = planeCapsuleContact(q0, tp0, q1, tp1, scale1.y, scale1.x, cDist);
        else if (type0 == PXG_GEOM_CAPSULE)
            c = capsuleCapsuleContact(q0, tp0, scale0.y, scale0.x, q1, tp1, scale1.y, scale1.x, cDist);
    }

    const uint nbContacts = c.nbContacts;
    float3 normal = c.normal;
    if (nbContacts != 0u && flip)
        normal = -normal;

    // =======================================================================
    // Output tail — mirrors contactPatchUtils.cuh.
    // =======================================================================

    // --- setContactPointAndForcePointers ---
    uint contactByteOffset = INVALID;
    if (nbContacts != 0u)
    {
        uint cOff, fOff;
        InterlockedAdd(counters[0].contactsBytes, CONTACT_SIZE * nbContacts, cOff);
        InterlockedAdd(counters[0].forceAndIndiceBytes, FORCE_SIZE * nbContacts, fOff);
        contactByteOffset = cOff;
        uint forceOffset = fOff;
        if ((cOff + CONTACT_SIZE) > gParams.contactBytesLimit)
        {
            InterlockedOr(counters[0].overflowError, OVERFLOW_CONTACT);
            contactByteOffset = INVALID;
        }
        if ((fOff + FORCE_SIZE) > gParams.forceBytesLimit)
        {
            InterlockedOr(counters[0].overflowError, OVERFLOW_FORCE);
            forceOffset = INVALID;
        }
        if (contactByteOffset != INVALID)
        {
            cmOutputs[workIndex].contactForces = gParams.startContactForces + uint64_t(forceOffset);
            cmOutputs[workIndex].contactPoints = gParams.startContactPoints + uint64_t(contactByteOffset);
        }
        else
        {
            cmOutputs[workIndex].contactForces = 0uL;
            cmOutputs[workIndex].contactPoints = 0uL;
        }
    }

    // --- registerContactPatch ---
    const uint allflags = cmOutputs[workIndex].allflagsWord;
    const uint oldStatusFlags = (allflags >> 16u) & 0xffu;    // statusFlag byte
    uint statusFlags = oldStatusFlags & (~FLAG_TOUCH_KNOWN);
    if (nbContacts != 0u) statusFlags |= FLAG_HAS_TOUCH;
    else                  statusFlags |= FLAG_HAS_NO_TOUCH;

    const uint prevPatches = (allflags >> 8u) & 0xffu;        // current nbPatches byte
    const bool previouslyHadTouch = (oldStatusFlags & FLAG_HAS_TOUCH) != 0u;
    const bool prevTouchKnown = (oldStatusFlags & FLAG_TOUCH_KNOWN) != 0u;
    const uint numPatches = (nbContacts > 0u) ? 1u : 0u;
    const bool currentlyHasTouch = nbContacts > 0u;
    const bool change = (previouslyHadTouch != currentlyHasTouch) || (!prevTouchKnown);

    touchChangeFlags[workIndex] = change ? 1u : 0u;
    patchChangeFlags[workIndex] = (prevPatches != numPatches) ? 1u : 0u;

    cmOutputs[workIndex].allflagsWord =
        (prevPatches << 24u) | (statusFlags << 16u) | (numPatches << 8u);
    {
        const uint nf = cmOutputs[workIndex].nbContacts_flags;
        cmOutputs[workIndex].nbContacts_flags = (nf & 0xffff0000u) | (nbContacts & 0xffffu);
    }

    uint patchIndex = INVALID;
    if (nbContacts != 0u)
    {
        InterlockedAdd(counters[0].patchesBytes, PATCH_SIZE, patchIndex);
        if ((patchIndex + PATCH_SIZE) > gParams.patchBytesLimit)
        {
            InterlockedOr(counters[0].overflowError, OVERFLOW_PATCH);
            patchIndex = INVALID;

            statusFlags = (statusFlags & (~FLAG_TOUCH_KNOWN)) | FLAG_HAS_NO_TOUCH;
            cmOutputs[workIndex].allflagsWord = (prevPatches << 24u) | (statusFlags << 16u);
            const uint nf = cmOutputs[workIndex].nbContacts_flags;
            cmOutputs[workIndex].nbContacts_flags = (nf & 0xffff0000u);
            touchChangeFlags[workIndex] = previouslyHadTouch ? 1u : 0u;
            patchChangeFlags[workIndex] = (prevPatches != 0u) ? 1u : 0u;
        }
        else
        {
            cmOutputs[workIndex].contactPatches = gParams.startContactPatches + uint64_t(patchIndex);
        }
    }

    // --- insertIntoPatchStream ---
    if (patchIndex != INVALID)
    {
        const uint mi0 = shape0.materialIndex;
        const uint mi1 = shape1.materialIndex;
        const PxsMaterialData m0 = materials[mi0];
        const PxsMaterialData m1 = materials[mi1];

        float sta, dyn, rest, damping;
        uint matFlags;
        combineMaterials(m0, m1, sta, dyn, rest, damping, matFlags);

        Ptr<uint> patchBase = Ptr<uint>(gParams.patchStream + uint64_t(patchIndex));
        const uint one = asuint(1.0f);
        patchBase[0] = one; patchBase[1] = one; patchBase[2] = one; patchBase[3] = one;
        patchBase[4] = asuint(normal.x);
        patchBase[5] = asuint(normal.y);
        patchBase[6] = asuint(normal.z);
        patchBase[7] = asuint(rest);
        patchBase[8] = asuint(dyn);
        patchBase[9] = asuint(sta);
        patchBase[10] = asuint(damping);
        patchBase[11] = ((nbContacts & 0xffu) << 16u) | ((matFlags & 0xffu) << 24u);
        patchBase[12] = (mi0 & 0xffffu) << 16u;
        // P9.87: materialIndex1 is a PxU16 store (contactPatchUtils.cuh:171), so pad[0]
        // (bytes 54-55) keeps the stream's content and words 14-15 (pad[1..4]) are never
        // written (:161-178).  Single writer: patchIndex is this invocation's own
        // InterlockedAdd(PATCH_SIZE) reservation.
        patchBase[13] = (patchBase[13] & 0xffff0000u) | (mi1 & 0xffffu);
    }

    // --- write the contact points/penetrations to the contact stream ---
    if (contactByteOffset != INVALID)
    {
        Ptr<float4> baseStream = Ptr<float4>(gParams.contactStream + uint64_t(contactByteOffset));
        if (nbContacts > 0u) baseStream[0] = c.pointPen0;
        if (nbContacts > 1u) baseStream[1] = c.pointPen1;
    }
}
