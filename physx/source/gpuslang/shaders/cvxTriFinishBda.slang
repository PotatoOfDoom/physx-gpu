// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

// cvxTriFinishBda.slang
//
// `convexTrimeshFinishContacts` (convexMeshOutput.cu:116-346) on the engine BDA
// ABI — the LAST unmapped kernel of the convex-vs-trimesh chain, and the one that
// turns a multi-manifold into the contact patches, points and face indices the
// solver actually reads.
//
// Lifted from `npCvxTrimeshOut.slang`, the Phase-1 verbatim port of the same
// kernel, with the same quirks; what changes is where the bytes come from and go.
// The reservations become real `InterlockedAdd`s on the shared
// `PxgPatchAndContactCounters`, and the three streams become real writes.
//
// ONE WARP PER PAIR.  The host launches `ceil(numTests / 2)` blocks of two warps
// (PxgNarrowphaseCore.cpp:2742-2744) and derives the pair from
// `globalThreadIndex / WARP_SIZE` — no grid-stride loop, so one workgroup per CUDA
// block and a constant-trip-2 loop over its warps covers every pair exactly once,
// the shape P8.48 and P8.50b already use.
//
// THE LANE LAYOUT is the CUDA's: `singleManifoldIndex = lane / 6` and
// `threadIndexInManifold = lane % 6`, so lanes 0..23 cover four manifolds of six
// and lanes 24..31 map to manifold 4 or 5 and can never hold a contact.
//
// ===========================================================================
// THE QUIRKS, PORTED VERBATIM
// ===========================================================================
//
//   * THE RESERVATION IS LANE 0's ALONE, and `insertAveragePoint` inflates
//     `totalNumContacts` by `nbManifolds` ON LANE 0 ONLY (cu:190-191).  The
//     inflated count feeds the reservation sizes, the stored `nbContacts`, and
//     LANE 0's face-index base — while lanes 1..31 keep the un-inflated popcount.
//     The face-index region base is therefore PER-LANE DIVERGENT, and lane 0's
//     slot can collide with another lane's.  That is a genuine unordered
//     intra-warp write conflict in the CUDA; it is reproduced, not repaired.
//     The average contacts themselves are never written (`getAvgContactValue`,
//     cu:84-98, is dead code) — the slots are reserved and left stale.
//   * THE OVERFLOW TESTS ARE A CHAINED else-if over the FULL reservation sizes,
//     so at most one error bit is ever set even when two limits are blown.
//     Rollback zeroes `nbManifolds` AND `totalNumContacts`, rewrites
//     HAS_NO_TOUCH, and NULLs the three output pointers.
//   * A PAIR WITH MANIFOLDS BUT NO CONTACTS and no average insertion skips the
//     reservation entirely, leaving the offsets at 0xFFFFFFFF — yet still passes
//     the cu:257 gate and stores `base + 0xFFFFFFFF` as its output pointers
//     (wrapped).  The `hasContacts` guards then suppress every stream write, so
//     nothing is dereferenced.  Ported exactly.
//   * `getWorldNormal` (cu:100-111) does NOT average despite its comment: it
//     takes the manifold's FIRST contact normal, rotates it by the TRIMESH
//     transform and normalizes.  Lanes past `numContacts` feed it a zero vector
//     and get zero back.
//   * `flip` is `type(shapeRef0) == eTRIANGLEMESH`; it swaps the shape and
//     transform refs and negates the patch normal, but NOT the per-contact
//     world point.
//   * THE PATCH HEADER IS WRITTEN FIELD BY FIELD in the CUDA (cu:65-81):
//     bytes 54-63 (pad[0..4]) are never stored and keep the stream's content.
//     So word 12 is stored whole, word 13 is a read-modify-write of its low
//     half (cu:75, a PxU16 store), and words 14-15 are not touched.  P9.87:
//     this used to be one uint4 store that zeroed all ten pad bytes, and the
//     fixture's zero-filled stream and zero-expecting oracle agreed with it.

// ===========================================================================
// THE ONE SWEEP SURVIVOR, AND WHY IT IS UNREACHABLE
// ===========================================================================
//
// 46 of 47 mutants die.  The one that lives drops the `nbManifolds == 0` term
// from the cu:257 gate.  It cannot be observed: with no manifolds every lane's
// `numContacts` is zero (the `mIdx >= nbManifolds` test), so `hasContacts` is
// false across the whole warp and every write below sits behind it.  The body
// still runs, reads valid memory and produces nothing.  The term is the CUDA's
// and is kept, but the per-lane guards already do its job.

import pxgSlangNpAbi;

static const uint CF_WARP = 32u;
static const uint CF_WARPS_PER_BLOCK = 2u;          // CORRELATE_WARPS_PER_BLOCK
static const uint CF_MAX_CONTACTS = 6u;             // PXG_SUBMANIFOLD_MAX_CONTACTS

static const uint CF_TOUCH_KNOWN = 3u;
static const uint CF_HAS_TOUCH = 2u;
static const uint CF_HAS_NO_TOUCH = 1u;
static const uint CF_PATCH_OVF = 4u, CF_CONTACT_OVF = 1u, CF_FORCE_OVF = 2u;
static const uint CF_TYPE_SPHERE = 0u, CF_TYPE_CAPSULE = 2u, CF_TYPE_TRIANGLEMESH = 8u;

// PxgPersistentContactMultiManifold, 1216 B (see cvxTriCorrelateBda.slang).
static const uint CF_MANIFOLD_STRIDE = 1216u;
static const uint CF_NB_CONTACTS_OFF = 1152u;
static const uint CF_NB_MANIFOLDS_OFF = 1200u;

// (pairs*, transformCache*, gpuShapes*, cmInputs*, cmOutputs*, cmMultiManifold*,
//  numPairs, materials*, contactStream*, patchStream*, forceAndIndiceStream*,
//  insertAveragePoint(bool), patchAndContactCounters*, touchChangeFlags*,
//  patchChangeFlags*, startContactPatches*, startContactPoints*,
//  startContactForces*, patchBytesLimit, contactBytesLimit, forceBytesLimit)
struct CfParams
{
    uint64_t pairs;                    // @0
    uint64_t transformCache;           // @8
    uint64_t gpuShapes;                // @16
    uint64_t cmInputs;                 // @24
    uint64_t cmOutputs;                // @32
    uint64_t cmMultiManifold;          // @40
    uint     numPairs;                 // @48
    uint64_t materials;                // @56
    uint64_t contactStream;            // @64
    uint64_t patchStream;              // @72
    uint64_t forceAndIndiceStream;     // @80
    uint     insertAveragePointWord;   // @88  bool byte, upper pad zero-filled
    uint64_t patchAndContactCounters;  // @96
    uint64_t touchChangeFlags;         // @104
    uint64_t patchChangeFlags;         // @112
    uint64_t startContactPatches;      // @120
    uint64_t startContactPoints;       // @128
    uint64_t startContactForces;       // @136
    uint     patchBytesLimit;          // @144
    uint     contactBytesLimit;        // @148
    uint     forceBytesLimit;          // @152
}                                      // = 160

[[vk::push_constant]]
ConstantBuffer<CfParams> gCf;

groupshared uint gsCfStage[CF_WARP];

uint cfExchange(uint lane, uint value, uint src)
{
    gsCfStage[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    const uint r = gsCfStage[src & (CF_WARP - 1u)];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

uint cfBallot(uint lane, bool pred)
{
    gsCfStage[lane] = pred ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    uint m = 0u;
    for (uint i = 0u; i < CF_WARP; ++i)
        m |= gsCfStage[i] << i;
    GroupMemoryBarrierWithGroupSync();
    return m;
}

float3 cfQuatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2);
}

float3 cfNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0, 0, 0);
}

float cfCombineScalars(float a, float b, uint mode)
{
    if (mode == 0u) return 0.5f * (a + b);
    if (mode == 1u) return min(a, b);
    if (mode == 2u) return a * b;
    if (mode == 3u) return max(a, b);
    return 0.0f;
}

// PxsCombineMaterials, exactly as the Phase-1 port has it.
void cfCombineMaterials(PxsMaterialDataGpu m0, PxsMaterialDataGpu m1,
                        out uint matFlags, out float staF, out float dynF,
                        out float rest, out float damp)
{
    const uint f0 = m0.flagsModes0 & 0xFFFFu, f1 = m1.flagsModes0 & 0xFFFFu;
    const uint fric0 = (m0.flagsModes0 >> 16u) & 0xFFu, fric1 = (m1.flagsModes0 >> 16u) & 0xFFu;
    const uint rst0 = (m0.flagsModes0 >> 24u) & 0xFFu, rst1 = (m1.flagsModes0 >> 24u) & 0xFFu;
    const uint dmp0 = m0.modes1 & 0xFFu, dmp1 = m1.modes1 & 0xFFu;

    const float r0 = m0.restitution, r1 = m1.restitution;
    const bool c0 = r0 < 0.0f, c1 = r1 < 0.0f;
    const bool oneC = c0 != c1, bothC = c0 && c1;
    const bool a0 = (f0 & 16u) != 0u, a1 = (f1 & 16u) != 0u;
    const bool oneA = a0 != a1;

    if (bothC && oneA) rest = a0 ? r0 : r1;
    else
    {
        const uint mode = oneC ? 1u : max(rst0, rst1);
        const float flip = (bothC && mode == 2u) ? -1.0f : 1.0f;
        rest = flip * cfCombineScalars(r0, r1, mode);
    }
    if (bothC && oneA) damp = a0 ? m0.damping : m1.damping;
    else
    {
        const uint mode = oneC ? 3u : max(dmp0, dmp1);
        damp = cfCombineScalars(m0.damping, m1.damping, mode);
    }
    const uint cf = f0 | f1;
    if ((cf & 1u) == 0u)
    {
        const uint mode = max(fric0, fric1);
        const float d = cfCombineScalars(m0.dynamicFriction, m1.dynamicFriction, mode);
        const float s2 = cfCombineScalars(m0.staticFriction, m1.staticFriction, mode);
        const float fd = max(d, 0.0f);
        dynF = fd; staF = (s2 - fd) >= 0.0f ? s2 : fd;
        matFlags = cf;
    }
    else { matFlags = cf | 2u; dynF = 0.0f; staF = 0.0f; }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void convexTrimeshFinishContacts(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const bool insertAveragePoint = (gCf.insertAveragePointWord & 0xFFu) != 0u;

    for (uint sub = 0u; sub < CF_WARPS_PER_BLOCK; ++sub)
    {
        AllMemoryBarrierWithGroupSync();

        const uint pairIndex = gid.x * CF_WARPS_PER_BLOCK + sub;
        if (pairIndex >= gCf.numPairs)
            continue;

        const uint64_t manBase = gCf.cmMultiManifold + uint64_t(pairIndex) * uint64_t(CF_MANIFOLD_STRIDE);
        uint nbManifolds = Ptr<uint>(manBase + uint64_t(CF_NB_MANIFOLDS_OFF))[0];

        const uint mIdx = lane / CF_MAX_CONTACTS;
        const uint tIdx = lane % CF_MAX_CONTACTS;
        const uint numContacts = (mIdx >= nbManifolds)
            ? 0u
            : Ptr<uint>(manBase + uint64_t(CF_NB_CONTACTS_OFF) + uint64_t(min(mIdx, 3u)) * 4UL)[0];

        const bool hasContacts = tIdx < numContacts;
        const uint contactMask = cfBallot(lane, hasContacts);
        const uint popcContacts = countbits(contactMask);
        const uint writeIndex = countbits(contactMask & ((1u << lane) - 1u));

        Ptr<PxsContactManagerOutputGpu> cmOutputs = Ptr<PxsContactManagerOutputGpu>(gCf.cmOutputs);
        const uint allflags = cmOutputs[pairIndex].allflagsWord;
        const uint oldStatusFlags = (allflags >> 16u) & 0xFFu;
        uint statusFlags = oldStatusFlags & ~CF_TOUCH_KNOWN;
        statusFlags |= (popcContacts != 0u) ? CF_HAS_TOUCH : CF_HAS_NO_TOUCH;
        const uint prevPatches = (allflags & 0xFFFFu) >> 8u;

        // ---- the lane-0 reservation block (cu:188-255) ----------------------
        uint patchByteOffset = 0xFFFFFFFFu;
        uint contactByteOffset = 0xFFFFFFFFu;
        uint forceByteOffset = 0xFFFFFFFFu;
        bool overflow = false;
        uint totalNumContacts = popcContacts;   // lane 0 inflates this; others do not
        uint nbManifoldsL0 = nbManifolds;

        if (lane == 0u)
        {
            totalNumContacts += insertAveragePoint ? nbManifoldsL0 : 0u;

            if (totalNumContacts != 0u)
            {
                Ptr<PxgPatchAndContactCountersGpu> ctr =
                    Ptr<PxgPatchAndContactCountersGpu>(gCf.patchAndContactCounters);
                InterlockedAdd(ctr[0].patchesBytes, 64u * nbManifoldsL0, patchByteOffset);
                InterlockedAdd(ctr[0].contactsBytes, 16u * totalNumContacts, contactByteOffset);
                InterlockedAdd(ctr[0].forceAndIndiceBytes, 4u * totalNumContacts * 2u, forceByteOffset);

                // A CHAINED else-if, so at most one error bit is set even when two
                // limits are blown.  The 64-bit widening keeps the comparison
                // honest when a cursor is already near 2^32.
                if ((uint64_t(patchByteOffset) + 64UL * uint64_t(nbManifoldsL0)) > uint64_t(gCf.patchBytesLimit))
                {
                    InterlockedOr(ctr[0].overflowError, CF_PATCH_OVF);
                    patchByteOffset = 0xFFFFFFFFu; overflow = true;
                }
                else if ((uint64_t(contactByteOffset) + 16UL * uint64_t(totalNumContacts)) > uint64_t(gCf.contactBytesLimit))
                {
                    InterlockedOr(ctr[0].overflowError, CF_CONTACT_OVF);
                    contactByteOffset = 0xFFFFFFFFu; overflow = true;
                }
                else if ((uint64_t(forceByteOffset) + 4UL * uint64_t(totalNumContacts) * 2UL) > uint64_t(gCf.forceBytesLimit))
                {
                    InterlockedOr(ctr[0].overflowError, CF_FORCE_OVF);
                    forceByteOffset = 0xFFFFFFFFu; overflow = true;
                }

                if (overflow)
                {
                    nbManifoldsL0 = 0u;
                    totalNumContacts = 0u;
                    statusFlags = (statusFlags & ~CF_TOUCH_KNOWN) | CF_HAS_NO_TOUCH;
                }
            }

            const bool previouslyHadTouch = (oldStatusFlags & CF_HAS_TOUCH) != 0u;
            const bool prevTouchKnown = (oldStatusFlags & CF_TOUCH_KNOWN) != 0u;
            const bool currentlyHasTouch = nbManifoldsL0 != 0u;

            Ptr<uint>(gCf.touchChangeFlags)[pairIndex] =
                ((previouslyHadTouch != currentlyHasTouch) || !prevTouchKnown) ? 1u : 0u;
            Ptr<uint>(gCf.patchChangeFlags)[pairIndex] = (prevPatches != nbManifoldsL0) ? 1u : 0u;

            cmOutputs[pairIndex].allflagsWord =
                (((prevPatches << 8u) | statusFlags) << 16u) | (nbManifoldsL0 << 8u);
            cmOutputs[pairIndex].nbContactsFlags =
                (cmOutputs[pairIndex].nbContactsFlags & 0xFFFF0000u) | (totalNumContacts & 0xFFFFu);

            if (!overflow)
            {
                // The wrapped `base + 0xFFFFFFFF` of a no-contact pair is stored as
                // the CUDA stores it; nothing dereferences it because every stream
                // write below is behind `hasContacts`.
                cmOutputs[pairIndex].contactForces = gCf.startContactForces + uint64_t(forceByteOffset);
                cmOutputs[pairIndex].contactPatches = gCf.startContactPatches + uint64_t(patchByteOffset);
                cmOutputs[pairIndex].contactPoints = gCf.startContactPoints + uint64_t(contactByteOffset);
            }
            else
            {
                cmOutputs[pairIndex].contactForces = 0UL;
                cmOutputs[pairIndex].contactPatches = 0UL;
                cmOutputs[pairIndex].contactPoints = 0UL;
            }
        }

        // cu:257 -- `__any_sync(overflow) || nbManifolds == 0`.  `nbManifolds` here
        // is the UNIFORM pre-rollback value; the rollback is lane 0's private copy
        // and is covered by the any().
        const uint ovfBallot = cfBallot(lane, overflow);
        if (ovfBallot != 0u || nbManifolds == 0u)
            continue;

        // ---- the post-gate body (cu:259-345) --------------------------------
        Ptr<PxgContactManagerInputGpu> cmInputs = Ptr<PxgContactManagerInputGpu>(gCf.cmInputs);
        Ptr<PxgShapeGpu> shapes = Ptr<PxgShapeGpu>(gCf.gpuShapes);
        Ptr<PxsCachedTransformGpu> cache = Ptr<PxsCachedTransformGpu>(gCf.transformCache);

        const PxgContactManagerInputGpu inp = cmInputs[pairIndex];
        uint shapeRef0 = inp.shapeRef0, shapeRef1 = inp.shapeRef1;
        uint tcRef0 = inp.transformCacheRef0, tcRef1 = inp.transformCacheRef1;

        const bool flip = shapes[shapeRef0].type == CF_TYPE_TRIANGLEMESH;
        if (flip)
        {
            const uint sr = shapeRef0; shapeRef0 = shapeRef1; shapeRef1 = sr;
            const uint tr = tcRef0;    tcRef0 = tcRef1;       tcRef1 = tr;
        }

        const PxgShapeGpu shape0 = shapes[shapeRef0];
        const uint2 materialIndices = uint2(
            Ptr<uint>(gCf.pairs + uint64_t(pairIndex) * 56UL + 48UL)[0],
            Ptr<uint>(gCf.pairs + uint64_t(pairIndex) * 56UL + 52UL)[0]);

        // The three offsets are lane 0's; every other lane gets them by shuffle.
        patchByteOffset = cfExchange(lane, patchByteOffset, 0u);
        contactByteOffset = cfExchange(lane, contactByteOffset, 0u);
        forceByteOffset = cfExchange(lane, forceByteOffset, 0u);
        // `totalNumContacts` is NOT shuffled: lane 0 keeps the inflated value and
        // the others the plain popcount, which is what makes the face-index base
        // per-lane divergent (cu:191 vs cu:290).

        const PxsCachedTransformGpu triXf = cache[tcRef1];
        const PxsCachedTransformGpu sphXf = cache[tcRef0];
        const float4 triQ = float4(triXf.qx, triXf.qy, triXf.qz, triXf.qw);
        const float3 triP = float3(triXf.px, triXf.py, triXf.pz);
        const float4 sphQ = float4(sphXf.qx, sphXf.qy, sphXf.qz, sphXf.qw);
        const float3 sphP = float3(sphXf.px, sphXf.py, sphXf.pz);

        // getWorldNormal: the manifold's FIRST contact normal, not an average.
        const uint64_t cp0 = manBase + uint64_t(mIdx * CF_MAX_CONTACTS) * 48UL;
        float3 nrm = float3(0, 0, 0);
        if (hasContacts)
            nrm = float3(Ptr<float>(cp0 + 24UL)[0], Ptr<float>(cp0 + 28UL)[0], Ptr<float>(cp0 + 32UL)[0]);
        const float3 worldNormal = cfNormalized(cfQuatRotate(triQ, nrm));

        if (hasContacts)
        {
            const uint64_t cp = manBase + uint64_t(mIdx * CF_MAX_CONTACTS + tIdx) * 48UL;
            const float3 pointA = float3(Ptr<float>(cp + 0UL)[0], Ptr<float>(cp + 4UL)[0], Ptr<float>(cp + 8UL)[0]);
            const float3 pointB = float3(Ptr<float>(cp + 12UL)[0], Ptr<float>(cp + 16UL)[0], Ptr<float>(cp + 20UL)[0]);
            float pen = Ptr<float>(cp + 36UL)[0];
            const uint triIndex = Ptr<uint>(cp + 40UL)[0];

            // The CUDA gates this on `threadIndexInManifold == 0` alone.  The
            // extra offset test is provably unreachable -- the reservation runs
            // whenever totalNumContacts != 0, and reaching here needs
            // popcContacts > 0 -- but it is kept because the alternative failure
            // is a WILD WRITE at patchStream + 0xFFFFFFFF rather than a wrong value.
            if (tIdx == 0u && patchByteOffset != 0xFFFFFFFFu)
            {
                Ptr<PxsMaterialDataGpu> materials = Ptr<PxsMaterialDataGpu>(gCf.materials);
                uint matFlags; float staF, dynF, rest, damp;
                cfCombineMaterials(materials[materialIndices.x & 0xFFFFu], materials[materialIndices.y & 0xFFFFu],   // P9.88: cu:61 static_cast<PxU16>
                                   matFlags, staF, dynF, rest, damp);

                const float3 pn = flip ? -worldNormal : worldNormal;
                const uint64_t pb = gCf.patchStream + uint64_t(patchByteOffset) + uint64_t(mIdx) * 64UL;
                Ptr<float4>(pb + 0UL)[0] = float4(1.0f, 1.0f, 1.0f, 1.0f);   // mass modification
                Ptr<float4>(pb + 16UL)[0] = float4(pn, rest);
                Ptr<float4>(pb + 32UL)[0] = float4(dynF, staF, damp,
                    asfloat((writeIndex & 0xFFFFu) | ((numContacts & 0xFFu) << 16u) | ((matFlags & 0xFFu) << 24u)));
                // Word 12 whole: internalFlags (cu:81) | materialIndex0 (cu:74).  Word 13
                // only its low half (cu:75); single writer -- lane tIdx == 0 of manifold
                // mIdx, inside this pair's own InterlockedAdd reservation.
                Ptr<uint> pw = Ptr<uint>(pb);
                pw[12] = 1u /* PxContactPatch::eHAS_FACE_INDICES */ | ((materialIndices.x & 0xFFFFu) << 16u);
                pw[13] = (pw[13] & 0xFFFF0000u) | (materialIndices.y & 0xFFFFu);
            }

            float3 worldPt;
            if (shape0.type == CF_TYPE_SPHERE)
            {
                const float radius = shape0.scaleY;
                pen = pen - radius;
                worldPt = sphP - worldNormal * radius;
            }
            else if (shape0.type == CF_TYPE_CAPSULE)
            {
                const float radius = shape0.scaleY;
                pen = pen - radius;
                worldPt = cfQuatRotate(sphQ, pointA) + sphP - worldNormal * radius;
            }
            else
            {
                worldPt = cfQuatRotate(triQ, pointB) + triP;
            }

            if (contactByteOffset != 0xFFFFFFFFu)
                Ptr<float4>(gCf.contactStream + uint64_t(contactByteOffset) +
                            uint64_t(writeIndex) * 16UL)[0] = float4(worldPt, pen);

            if (forceByteOffset != 0xFFFFFFFFu)
                Ptr<uint>(gCf.forceAndIndiceStream + uint64_t(forceByteOffset) +
                          uint64_t(totalNumContacts) * 4UL)[writeIndex] = triIndex;
        }
    }
}
