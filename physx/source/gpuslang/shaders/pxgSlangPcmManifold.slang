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
//
// ===========================================================================
// pxgSlangPcmManifold — the PERSISTENT contact-manifold gate, on the BDA ABI.
//
// This is the half of midphaseGeneratePairs (convexMeshMidphase.cu:378) that
// decides whether the BV32 traversal runs at all:
//
//     bool invalidate    = invalidateManifold(shapeToMeshNoScale, multiManifolds[cmIdx],
//                                             minMargin, ratio);          // :250
//     bool lostContacts  = refreshManifolds(...);                         // :254
//     bool fullContactGen = invalidate || lostContacts;                   // :263
//
// It is extracted FIRST, and on its own, for a reason the rest of the kernel
// does not share: it is the only STATEFUL part.  `refreshManifolds`
// (manifold.cuh:84) writes a recomputed penetration back into a persistent
// 1216-byte record and (:97) destructively zeroes mNbManifolds.  Everything
// else in that kernel is a pure function of the current frame.
//
// WHY THAT MATTERS FOR TESTING, stated before the code so it cannot be
// forgotten later: a single-dispatch golden-vector fixture CANNOT detect a bug
// here.  A port that simply always reported fullContactGen = true would produce
// correct contacts on frame one and in every isolated fixture, while having
// silently destroyed PCM caching — the damage would surface many frames later
// as solver warm-start drift.  The fixtures for this module therefore dispatch
// REPEATEDLY over an evolving transform and assert on the record BETWEEN
// dispatches.
//
// ---------------------------------------------------------------------------
// PxgPersistentContactMultiManifold, 1216 bytes
// (gpunarrowphase/include/PxgPersistentContactManifold.h:80).  Every offset
// below is taken from the struct's own trailing comments, which are END offsets:
//
//   mContacts[4][6]        @0     ... 1152   PxgContact, stride 48, row stride 288
//        .pointA  float3   @+0
//        .pointB  float3   @+12
//        .normal  float3   @+24
//        .penetration      @+36
//        .triIndex  uint   @+40
//        .pad       uint   @+44
//   mNbContacts[4]  uint   @1152  ... 1168
//   mRelativeTransform     @1168  ... 1200   PxAlignedTransform { PxAlignedQuat q; float4 p; }
//        .q  float4        @1168                (AlignedTransform.h:43 — q FIRST, then p)
//        .p  float4        @1184
//   mNbManifolds    uint   @1200  ( struct padded to 1216 )
//
// The record is 16-byte aligned and every field above lands on its natural
// alignment, so unlike PxBounds3 (24 B) and ConvexMeshPair (56 B) this one is
// safe to read as float4 where the field IS a float4.  PxgContact is 48 bytes,
// so contact c of manifold m starts at (m*6 + c)*48 — a multiple of 16, but the
// float3 members inside it are NOT 16-aligned, so they are read as scalars.
//
// ---------------------------------------------------------------------------
// WHAT THIS MODULE DELIBERATELY DOES NOT DO
//
// `refreshManifolds` ends with a warp-wide `__any_sync(FULL_MASK, lostContact)`
// followed by a lane-0 write of `mNbManifolds = 0` (manifold.cuh:93-99).  Both
// stay in the CALLER, and the module returns the PER-LANE `lostContact` instead.
// Two reasons:
//   - the group-wide OR is the caller's groupshared-scan emulation of a warp
//     ballot (this port has no subgroup-native path; see gpuslang/README.md),
//     and a module that owned its own groupshared would fix the group width;
//   - keeping the barrier in the caller keeps it in provably UNIFORM control
//     flow.  In the CUDA the `__any_sync` sits OUTSIDE `if (curManifoldIdx <
//     numManifolds)` precisely so every lane reaches it; a
//     GroupMemoryBarrierWithGroupSync that lands in divergent flow does not
//     error on llvmpipe, it hangs.
//
// LANE COUNT.  mmRefreshSlot mirrors `PxU32 i = threadIdxInWarp;` with NO loop
// (manifold.cuh:52), which covers 24 slots (4 manifolds x 6 contacts) and needs
// at least 24 lanes.  Under this port's sanctioned mapping — 32-lane virtual
// warps, [numthreads(32,1,1)], four subgroups of 8 per group — that holds, and
// lanes 24..31 are no-ops exactly as in the CUDA.  It is recorded here because
// narrowing the group to 8 would not merely drop work: 16 of the 24 cached
// contacts would keep the PREVIOUS frame's penetration while 8 were current.
// ===========================================================================

module pxgSlangPcmManifold;

public static const uint MULTIMANIFOLD_MAX_MANIFOLDS = 4u;   // manifold.cuh:36
public static const uint SUBMANIFOLD_MAX_CONTACTS    = 6u;   // manifold.cuh:37
public static const uint PCM_MAX_SLOTS =
    MULTIMANIFOLD_MAX_MANIFOLDS * SUBMANIFOLD_MAX_CONTACTS;  // 24

public static const uint64_t PCM_CONTACT_STRIDE   = 48UL;
public static const uint64_t PCM_NBCONTACTS_OFFSET = 1152UL;
public static const uint64_t PCM_RELTRANSFORM_Q   = 1168UL;
public static const uint64_t PCM_RELTRANSFORM_P   = 1184UL;
public static const uint64_t PCM_NBMANIFOLDS      = 1200UL;
public static const uint64_t PCM_STRIDE           = 1216UL;

// Contact field offsets within a PxgContact.
public static const uint64_t PCM_C_POINTA = 0UL;
public static const uint64_t PCM_C_POINTB = 12UL;
public static const uint64_t PCM_C_NORMAL = 24UL;
public static const uint64_t PCM_C_PEN    = 36UL;

// A float3 read from a NON-16-aligned address: three scalar loads, never a
// Ptr<float3>, because the underlying float3 straddles the 16-byte grid inside
// a 48-byte PxgContact.
float3 loadF3(uint64_t a)
{
    return float3(Ptr<float>(a)[0], Ptr<float>(a + 4UL)[0], Ptr<float>(a + 8UL)[0]);
}

// ---------------------------------------------------------------------------
// quatRotate / quatDot — the two quaternion operations the gate needs.
// PxQuat stores (x, y, z, w); the rotation is the standard
//   v + 2w(q x v) + 2(q x (q x v))  form PxQuat::rotate() uses.
// ---------------------------------------------------------------------------
public float3 pcmQuatRotate(float4 q, float3 v)
{
    const float3 qv = float3(q.x, q.y, q.z);
    const float3 t = 2.0f * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}

// aToB.transform(p) for a PxTransform { PxQuat q; PxVec3 p; }.
public float3 pcmTransform(float4 q, float3 t, float3 v)
{
    return pcmQuatRotate(q, v) + t;
}

// ---------------------------------------------------------------------------
// invalidateManifold (manifold.cuh:104).
//
//   const PxReal thresholdP = minMargin * ratio;
//   const PxReal thresholdQ = 0.9998f;                       // about 1 degree
//   deltaP = max(|aToB.p - prev.p| componentwise)
//   deltaQ = aToB.q.dot(prev.q)
//   return (deltaP > thresholdP) || (deltaQ < thresholdQ);
//
// The CUDA reads the previous relative transform with PxAlignedTransform_ReadWarp,
// a warp-cooperative load whose RESULT is identical in every lane; read here as
// two plain float4s, which is the same value and, unlike the shuffle-based
// helper, carries no lane-count assumption.
//
// NOTE the asymmetry, mirrored verbatim: position uses STRICTLY GREATER, and
// orientation uses STRICTLY LESS.  A record that has moved exactly thresholdP
// stays valid; one whose quaternion dot is exactly 0.9998 stays valid too.
// ---------------------------------------------------------------------------
// The PREDICATE, on values.  Split out in P9.82 because the same arithmetic is
// needed by a caller that holds the previous transform in registers rather than in
// a manifold record -- npTriPlaneNp's fixture reads it from a StructuredBuffer --
// and P9.82's first attempt gave it a second home in pxgSlangTriPlane before
// noticing this one.  One spelling; the wrappers differ only in where the bytes
// come from.
//
// deltaP is the LARGEST ABSOLUTE COMPONENT of the translation difference, not its
// length, and deltaQ is a RAW quaternion dot with no abs -- so an antipodal but
// geometrically equal rotation invalidates.  Both are upstream's.
public bool mmInvalidatePredicate(float4 aToB_q, float3 aToB_p,
                                  float4 prevQ, float3 prevP, float thresholdP)
{
    const float thresholdQ = 0.9998f;

    const float3 delta = aToB_p - prevP;
    const float3 absDelta = float3(abs(delta.x), abs(delta.y), abs(delta.z));

    const float deltaP = max(absDelta.x, max(absDelta.y, absDelta.z));
    const float deltaQ = dot(aToB_q, prevQ);

    return (deltaP > thresholdP) || (deltaQ < thresholdQ);
}

// refreshManifolds' per-contact ARITHMETIC, on values -- see mmRefreshSlot below
// for the two details it preserves (the projection uses the OLD penetration; the
// write-back is unconditional within the slot guard).
public void mmRefreshMath(float4 aToB_q, float3 aToB_p, float3 pointA, float3 pointB,
                          float3 normal, float penOld, float sqProjectBreakingThreshold,
                          out float penNew, out bool lost)
{
    const float3 localAInB = pcmTransform(aToB_q, aToB_p, pointA);
    const float3 v = localAInB - pointB;
    penNew = dot(v, normal);

    const float3 projectedPoint = localAInB - normal * penOld;
    const float3 projectedDifference = pointB - projectedPoint;
    const float distance2d = dot(projectedDifference, projectedDifference);
    lost = distance2d > sqProjectBreakingThreshold;
}

public bool mmInvalidateManifold(uint64_t manifold, float4 aToB_q, float3 aToB_p,
                                 float minMargin, float ratio)
{
    const float4 prevQ = Ptr<float4>(manifold + PCM_RELTRANSFORM_Q)[0];
    const float4 prevP = Ptr<float4>(manifold + PCM_RELTRANSFORM_P)[0];
    return mmInvalidatePredicate(aToB_q, aToB_p, prevQ,
                                 float3(prevP.x, prevP.y, prevP.z), minMargin * ratio);
}

// ---------------------------------------------------------------------------
// refreshManifolds, the per-lane body (manifold.cuh:40-92).
//
//   PxU32 i = threadIdxInWarp;                  // NO loop — 24 slots, >=24 lanes
//   curManifoldIdx = i / SUBMANIFOLD_MAX_CONTACTS;
//   if (curManifoldIdx < numManifolds) {
//       nbContacts   = mNbContacts[curManifoldIdx];
//       curContactIdx = i % SUBMANIFOLD_MAX_CONTACTS;
//       if (curContactIdx < nbContacts) {
//           localAInB = aToB.transform(pointA);
//           v         = localAInB - pointB;
//           penNew    = v.dot(normal);
//           projectedPoint      = localAInB - normal * penOld;   // penOld, NOT penNew
//           projectedDifference = pointB - projectedPoint;
//           curContact.penetration = penNew;                     // WRITE-BACK
//           if (projectedDifference.magnitudeSquared() > sqProjectBreakingThreshold)
//               lostContact = true;
//       }
//   }
//
// Two details that a paraphrase loses and a mutant would exploit:
//   - the projection uses the OLD penetration, and the write-back of the NEW one
//     happens after it.  Reordering those two lines changes which contacts break.
//   - the write-back is UNCONDITIONAL within the slot guard: a contact whose
//     drift exceeds the threshold still gets its new penetration stored before
//     the caller zeroes mNbManifolds.
//
// Returns this lane's `lostContact`.  The caller ORs it across the group.
// ---------------------------------------------------------------------------
public bool mmRefreshSlot(uint64_t manifold, float4 aToB_q, float3 aToB_p,
                          float projectBreakingThreshold, uint lane)
{
    const float sqProjectBreakingThreshold = projectBreakingThreshold * projectBreakingThreshold;
    const uint numManifolds = Ptr<uint>(manifold + PCM_NBMANIFOLDS)[0];

    const uint curManifoldIdx = lane / SUBMANIFOLD_MAX_CONTACTS;
    if (curManifoldIdx >= numManifolds)
        return false;

    const uint nbContacts =
        Ptr<uint>(manifold + PCM_NBCONTACTS_OFFSET + uint64_t(curManifoldIdx) * 4UL)[0];
    const uint curContactIdx = lane % SUBMANIFOLD_MAX_CONTACTS;
    if (curContactIdx >= nbContacts)
        return false;

    // mContacts[m][c] lives at (m*6 + c)*48, and with m = lane/6, c = lane%6 that
    // index is identically `lane` -- so the slot offset is lane*48, not a
    // two-dimensional walk.  Written out because the identity is easy to "fix"
    // into (curManifoldIdx*6 + curContactIdx)*48, which is the same number but
    // invites a stride mistake on the row.
    const uint64_t c = manifold + uint64_t(lane) * PCM_CONTACT_STRIDE;

    const float3 pointA = loadF3(c + PCM_C_POINTA);
    const float3 pointB = loadF3(c + PCM_C_POINTB);
    const float3 normal = loadF3(c + PCM_C_NORMAL);
    const float penOld = Ptr<float>(c + PCM_C_PEN)[0];

    float penNew;
    bool lost;
    mmRefreshMath(aToB_q, aToB_p, pointA, pointB, normal, penOld,
                  sqProjectBreakingThreshold, penNew, lost);

    Ptr<float>(c + PCM_C_PEN)[0] = penNew;

    return lost;
}
