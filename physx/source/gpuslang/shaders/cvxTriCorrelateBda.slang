// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

// cvxTriCorrelateBda.slang
//
// `convexTrimeshCorrelate` (convexMeshCorrelate.cu:348, core at :95) on the engine
// BDA ABI: the convex-trimesh MULTI-MANIFOLD PATCH BUILDER, and the last stage of
// the chain before contacts reach the solver.
//
// Contact generation has by now produced, per contact manager, a run of triangles
// each carrying a normal, a contact count and a slice of contact points.  This
// kernel turns that into at most four PATCHES of at most six contacts: repeatedly
// pick the deepest triangle still holding contacts, accept every triangle whose
// normal agrees with it to within PATCH_ACCEPTANCE_EPS, pull all their contacts
// into the warp, and reduce to six with the clustering `contactReduce` landed in
// P8.50a.
//
// ONE WARP PER CONTACT MANAGER, AND THE COVERAGE IS EXACT.  The host launches
// `ceil(numTests / CORRELATE_WARPS_PER_BLOCK)` blocks of two warps and passes
// numTests as `numPairs` (PxgNarrowphaseCore.cpp:2702-2705) — there is no
// grid-stride loop, so one workgroup per CUDA block with a constant-trip-2 loop
// over its warps covers every pair exactly once, the same shape P8.48 uses.  No
// ceiling, nothing silently skipped.
//
// `[numthreads(32,1,1)]` IS MANDATORY, not a preference: `crContactReduce6Cluster`
// carries barriers in warp-uniform control flow, and llvmpipe does not error on a
// divergent barrier — it HANGS.  The two warps of a CUDA block are therefore
// serialised through one 32-thread group rather than run as a 64-thread one.
//
// ===========================================================================
// THE QUIRKS, ALL PORTED VERBATIM
// ===========================================================================
//
//   * `purgeEmptyTriangles` writes `tempCounts = (slot << 4) | nbContacts` with
//     nbContacts taken from `getNbContacts`, which is FOUR bits — but the patch
//     loop reads it back with `& 7`, which is THREE.  A triangle with 8..15
//     contacts therefore purges as valid yet contributes `nbContacts & 7`, and at
//     exactly 8 it contributes ZERO while keeping a nonzero tempCounts forever:
//     never accepted, never cleared, so `remainingTriangles` never reaches zero
//     and the patch loop runs out its four iterations.
//   * the zero-triangle early-out writes `mNbManifolds` of `outBuffer[pairIndex]`,
//     while every other output goes to `outBuffer[pair.cmIndex]`.  Two different
//     manifolds whenever cmIndex != pairIndex.
//   * `mNbContacts[patch]` is written for every patch the loop runs, including
//     ones that end up empty.
//   * `numContactsTotal` (cu:301) is accumulated and never used; `cmOutputs` and
//     `pTempContactIndex` are parameters the core never touches.  All three are
//     kept in the ABI because the launch passes them.
//
// ===========================================================================
// THE ONE DELIBERATE DEVIATION
// ===========================================================================
//
// `purgeEmptyTriangles` guards its depth load with `tI < nbToLoad` but its nic
// load with `nbToLoad` — the block size, not the lane test (cu:70-71).  Lanes past
// the end therefore read one nic record beyond the run.  The value is PROVABLY
// DISCARDED (their `d` is FLT_MAX, so `valid` is false), so this port CLAMPS the
// index instead: the result is identical, and an unclamped read would be an
// out-of-bounds buffer-device-address load that can take llvmpipe down with it
// rather than returning garbage.

// ===========================================================================
// THE TWO SWEEP SURVIVORS, BOTH UNREACHABLE RATHER THAN UNTESTED
// ===========================================================================
//
// 35 of 37 mutants die.  The two that live cannot be reached by any input:
//
//   * firing the reduce at `>= 6` instead of `> 6`.  At exactly six held contacts
//     the mutant calls `crContactReduce6Cluster` — which opens with
//     `countbits(allMask) <= 6 -> return allMask`.  The extra call is a no-op, so
//     the two thresholds are indistinguishable by construction.
//   * writing `mRelativeTransform` unconditionally instead of under
//     `nbPatches > 0`.  The zero-triangle path has already returned by the time
//     the patch loop is entered, so `remainingTriangles` is at least one and the
//     loop always runs at least once.  `nbPatches == 0` never happens here.  The
//     guard is the CUDA's and is kept, but nothing can observe it.
//
// A third mutant is worth recording because of HOW it died: removing the reduce
// entirely does not produce wrong contacts, it HANGS.  With more than six contacts
// held and nothing to cull them, `readMask = ~currentContactMask` can empty out and
// `c += countbits(readMask)` stops advancing.  The reduction is load-bearing for
// TERMINATION, not only for manifold quality.

import pxgSlangContactReduce;   // crContactReduce6Cluster

static const uint CC_WARP = 32u;
static const float CC_MAX_F32 = 3.4028234663852886e+38f;
static const uint CORRELATE_WARPS_PER_BLOCK = 2u;
static const uint CC_CACHED = 0xFFFFFFFFu;                 // CONVEX_TRIMESH_CACHED
static const uint CC_MAX_MANIFOLDS = 4u;                   // PXG_MULTIMANIFOLD_MAX_SUBMANIFOLDS
static const uint CC_MAX_CONTACTS = 6u;                    // PXG_SUBMANIFOLD_MAX_CONTACTS
static const float CC_PATCH_ACCEPTANCE_EPS = 0.996194698f;

// PxgPersistentContactMultiManifold, 1216 B:
//   @0    PxgContact mContacts[4][6]      (48 B each: pointA @0, pointB @12,
//                                          normal @24, penetration @36,
//                                          triIndex @40, pad @44)
//   @1152 PxU32 mNbContacts[4]
//   @1168 PxAlignedTransform mRelativeTransform  (q @1168, p @1184)
//   @1200 PxU32 mNbManifolds
static const uint CC_MANIFOLD_STRIDE = 1216u;
static const uint CC_NB_CONTACTS_OFF = 1152u;
static const uint CC_REL_XF_OFF = 1168u;
static const uint CC_NB_MANIFOLDS_OFF = 1200u;

// (ConvexMeshPair*, PxReal**, ConvexTriNormalAndIndex**, ConvexTriContacts**,
//  PxgPersistentContactMultiManifold*, PxsContactManagerOutput*, PxU32*,
//  PxU32, PxReal, ConvexTriContact*, PxU32*)  ->  80 bytes.
struct CcParams
{
    uint64_t pairs;                    // @0
    uint64_t depthsPtr;                // @8
    uint64_t normalAndIndexPtr;        // @16
    uint64_t contactsPtr;              // @24
    uint64_t output;                   // @32
    uint64_t cmOutputs;                // @40  unused by the core
    uint64_t tempCountAndOffsetBuffer; // @48
    uint     numPairs;                 // @56
    float    clusterBias;              // @60
    uint64_t tempConvexTriContacts;    // @64
    uint64_t pTempContactIndex;        // @72  unused by the core
}                                      // = 80

[[vk::push_constant]]
ConstantBuffer<CcParams> gCc;

// --------------------------------------------------------------------------
// Warp primitives.  Self-contained and prefixed `cc`/`gsCc` for the reason
// pxgSlangContactReduce is: this shader composes with that module, a Slang
// `import` is a module boundary, and two copies of one groupshared array cost
// nothing next to a name collision that would silently share one.
// --------------------------------------------------------------------------
groupshared uint gsCcStage[CC_WARP];
groupshared uint gsCcRunSums[CC_WARP];
groupshared uint gsCcStartIdx[CC_WARP];
groupshared uint gsCcTriInds[CC_WARP];

uint ccExchange(uint lane, uint value, uint src)
{
    gsCcStage[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    const uint r = gsCcStage[src & (CC_WARP - 1u)];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

float ccShflF(uint lane, float value, uint src)
{
    return asfloat(ccExchange(lane, asuint(value), src));
}

uint ccBallot(uint lane, bool pred)
{
    gsCcStage[lane] = pred ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    uint m = 0u;
    for (uint i = 0u; i < CC_WARP; ++i)
        m |= gsCcStage[i] << i;
    GroupMemoryBarrierWithGroupSync();
    return m;
}

int ccLowestSetIndex(uint v) { return v == 0u ? -1 : int(firstbitlow(v)); }

// minIndex(v, mask, out) -- the LOWEST lane holding the minimum, which is what
// makes the deepest-triangle pick deterministic.
uint ccMinIndex(uint lane, float v, uint mask, out float minV)
{
    float m = ((mask >> lane) & 1u) != 0u ? v : CC_MAX_F32;
    m = min(m, ccShflF(lane, m, lane ^ 16u));
    m = min(m, ccShflF(lane, m, lane ^ 8u));
    m = min(m, ccShflF(lane, m, lane ^ 4u));
    m = min(m, ccShflF(lane, m, lane ^ 2u));
    m = min(m, ccShflF(lane, m, lane ^ 1u));
    minV = m;
    return uint(ccLowestSetIndex(ccBallot(lane, m == v) & mask));
}

// warpScan<AddOpPxU32> -- an INCLUSIVE add scan (reduction.cuh:262).
uint ccScanInclusive(uint lane, uint v)
{
    for (uint r = 1u; r < CC_WARP; r <<= 1)
    {
        const uint n = ccExchange(lane, v, lane - r);
        if (lane >= r)
            v += n;
    }
    return v;
}

// nputils binarySearch over the 32 exclusive run-sums: the last index whose
// entry is <= value, or 0 when even the first is greater.  With zero-contact
// lanes producing duplicate run-sums, the contact-bearing lane is always the LAST
// of an equal group, which is exactly what this returns.
uint ccBinarySearch32(uint value)
{
    uint left = 0u, right = CC_WARP;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (gsCcRunSums[pos] <= value) left = pos + 1u;
        else right = pos;
    }
    return left != 0u ? left - 1u : 0u;
}

uint ccGetNbContacts(uint index) { return (index & 0x7FFFFFFFu) >> 27u; }
uint ccGetTriangleIndex(uint index) { return index & ((1u << 27) - 1u); }

// P9.89: PxQuat::rotate LITERALLY (PxQuat.h:286-295, reached through
// PxTransform::rotate/transform at convexMeshCorrelate.cu:304 and :313).  This
// used to be v*(w*w - |u|^2) + u*(2 u.v) + (u x v)*(2w), which equals the PhysX
// form only algebraically and only for an exactly unit q: 2v(w^2 - 1/2) is not
// v(w^2 - |u|^2) once q is a rounded float quaternion, and the two sum their
// products in different orders.  Every written normal and pointB came out a few
// ulps off.  Same class as the rotate P9.37 removed from softBodyContactGenBda;
// npCvxCorrelate.slang's quatRotate is this same literal body.
float3 ccQuatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}

// --------------------------------------------------------------------------
// purgeEmptyTriangles (cu:58-87): compact the depth / nic runs down to the
// triangles that actually carry contacts, and record each survivor's ORIGINAL
// slot and contact count in tempCounts.
//
// The compaction is IN PLACE and `dest <= loaded + lane` always, so on CUDA the
// warp's loads all retire before its stores.  A 32-thread workgroup on llvmpipe
// gives no such guarantee, so the record is read into registers, a barrier is
// taken, and only then written.
// --------------------------------------------------------------------------
uint ccPurgeEmptyTriangles(uint lane, uint64_t depthBuf, uint64_t nicBuf, uint count,
                           uint64_t tempCounts)
{
    uint nbTriangles = 0u;

    for (uint loaded = 0u; loaded < count; loaded += CC_WARP)
    {
        const uint nbToLoad = min(count - loaded, CC_WARP);
        const uint src = loaded + lane;
        const uint srcClamped = min(src, count - 1u);   // see the header note

        const float d = lane < nbToLoad ? Ptr<float>(depthBuf + uint64_t(src) * 4UL)[0]
                                        : CC_MAX_F32;
        const float4 nicRec = Ptr<float4>(nicBuf + uint64_t(srcClamped) * 16UL)[0];
        const uint nbContacts = ccGetNbContacts(asuint(nicRec.w));

        const bool valid = (d != CC_MAX_F32) && (nbContacts > 0u);
        const uint validBits = ccBallot(lane, valid);
        const uint dest = countbits(validBits & ((1u << lane) - 1u)) + nbTriangles;

        // DEVICE memory, not groupshared: every lane's load of its own record must
        // retire before any lane's store to a compacted slot.
        AllMemoryBarrierWithGroupSync();

        if (valid)
        {
            Ptr<float>(depthBuf + uint64_t(dest) * 4UL)[0] = d;
            Ptr<uint>(tempCounts + uint64_t(dest) * 4UL)[0] = (src << 4u) | nbContacts;
            Ptr<float4>(nicBuf + uint64_t(dest) * 16UL)[0] = nicRec;
        }

        AllMemoryBarrierWithGroupSync();

        nbTriangles += countbits(validBits);
    }

    return nbTriangles;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void convexTrimeshCorrelate(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    for (uint sub = 0u; sub < CORRELATE_WARPS_PER_BLOCK; ++sub)
    {
        AllMemoryBarrierWithGroupSync();

        const uint pairIndex = gid.x * CORRELATE_WARPS_PER_BLOCK + sub;
        if (pairIndex >= gCc.numPairs)
            continue;

        // ConvexMeshPair, 56 B: PxTransform aToB (PxQuat q @0, PxVec3 p @16),
        // startIndex @28, count @32, cmIndex @36, roundedStartIndex @40.
        // The CUDA stages this through __shared__ so one warp loads it once; every
        // lane reading it directly is the same 56 bytes out of the same cache line
        // and saves a barrier.
        const uint64_t pairAddr = gCc.pairs + uint64_t(pairIndex) * 56UL;
        const float4 aToBq = Ptr<float4>(pairAddr)[0];
        const float3 aToBp = float3(Ptr<float>(pairAddr + 16UL)[0],
                                    Ptr<float>(pairAddr + 20UL)[0],
                                    Ptr<float>(pairAddr + 24UL)[0]);
        const uint start = Ptr<uint>(pairAddr + 28UL)[0];
        const uint count = Ptr<uint>(pairAddr + 32UL)[0];
        const uint cmIndex = Ptr<uint>(pairAddr + 36UL)[0];

        // Cached from last frame: nothing to do, and NOTHING is written.
        if (count == CC_CACHED)
            continue;

        const uint64_t nicBuffer = Ptr<uint64_t>(gCc.normalAndIndexPtr)[0];
        const uint64_t contactBuffer = Ptr<uint64_t>(gCc.contactsPtr)[0];
        const uint64_t depthBuffer = Ptr<uint64_t>(gCc.depthsPtr)[0];

        const uint64_t depthRun = depthBuffer + uint64_t(start) * 4UL;
        const uint64_t nicRun = nicBuffer + uint64_t(start) * 16UL;
        const uint64_t tempRun = gCc.tempCountAndOffsetBuffer + uint64_t(start) * 4UL;

        const uint nbTriangles = ccPurgeEmptyTriangles(lane, depthRun, nicRun, count, tempRun);

        if (nbTriangles == 0u)
        {
            // NOTE THE SLOT: pairIndex, not cmIndex.  Every other write below goes
            // to the cmIndex manifold.
            if (lane == 0u)
                Ptr<uint>(gCc.output + uint64_t(pairIndex) * uint64_t(CC_MANIFOLD_STRIDE)
                          + uint64_t(CC_NB_MANIFOLDS_OFF))[0] = 0u;
            continue;
        }

        const uint64_t outBase = gCc.output + uint64_t(cmIndex) * uint64_t(CC_MANIFOLD_STRIDE);

        uint nbPatches = 0u;
        uint remainingTriangles = nbTriangles;

        for (; nbPatches < CC_MAX_MANIFOLDS && remainingTriangles != 0u; nbPatches++)
        {
            // ---- (1) the deepest triangle still holding contacts ------------
            // Per-lane strided scan with a STRICT `<`, so within a lane the first
            // strided index wins a tie; then an exact min across lanes, lowest
            // lane wins.  Both tie-breaks are load-bearing.
            //
            // P9.89: bestIndex is an ABSOLUTE index into nicBuffer, `start + i`,
            // exactly as convexMeshCorrelate.cu:182 stores it and :194 reads it --
            // not run-relative.  The two agree whenever some lane has a winner.
            // They part when NO lane has one: purge (cu:72) only drops d == FLT_MAX,
            // so a surviving triangle with a +inf or NaN depth never passes
            // `d < bestDepth`, every lane keeps the identity (FLT_MAX, 0), minIndex
            // picks lane 0, and the CUDA reads bestNormal from GLOBAL slot 0 --
            // another pair's record.  Run-relative, the port read nicBuffer[start],
            // this pair's own first survivor, which accepts itself and builds one
            // patch where the CUDA accepts by a foreign normal (often: none, four
            // empty patches).  Slot 0 is written by whichever pair owns it, so on
            // the CUDA the read races that pair's purge; the port reproduces the
            // read, and the fixture keeps the slot-0 owner cached so it is stable.
            float bestDepth = CC_MAX_F32;
            uint bestIndex = 0u;
            for (uint i = lane; i < nbTriangles; i += CC_WARP)
            {
                if (Ptr<uint>(tempRun + uint64_t(i) * 4UL)[0] != 0u)
                {
                    const float d = Ptr<float>(depthRun + uint64_t(i) * 4UL)[0];
                    if (d < bestDepth)
                    {
                        bestDepth = d;
                        bestIndex = start + i;
                    }
                }
            }

            float minD;
            const uint i0 = ccMinIndex(lane, bestDepth, 0xFFFFFFFFu, minD);
            bestIndex = ccExchange(lane, bestIndex, i0);

            const float3 bestNormal = Ptr<float4>(nicBuffer + uint64_t(bestIndex) * 16UL)[0].xyz;

            uint currentContactMask = 0u;
            float3 pA = float3(0.0f, 0.0f, 0.0f);
            float separation = 0.0f;
            uint contactTriangleIndex = 0u;
            bool hasContact = false;

            for (uint i = 0u; i < nbTriangles; i += CC_WARP)
            {
                float3 normal = float3(0.0f, 0.0f, 0.0f);
                uint triIndex = 0u;
                uint nbContacts = 0u;
                uint startIndex = 0u;

                if ((i + lane) < nbTriangles)
                {
                    const uint index = Ptr<uint>(tempRun + uint64_t(i + lane) * 4UL)[0];
                    // THREE bits here; purge stored FOUR.  See the header.
                    nbContacts = index & 7u;
                    startIndex = index >> 4u;
                    if (nbContacts != 0u)
                    {
                        const float4 nic = Ptr<float4>(nicRun + uint64_t(i + lane) * 16UL)[0];
                        normal = nic.xyz;
                        triIndex = ccGetTriangleIndex(asuint(nic.w));
                    }
                }

                const bool accept = dot(normal, bestNormal) > CC_PATCH_ACCEPTANCE_EPS;
                if (!accept)
                    nbContacts = 0u;

                const uint acceptMask = ccBallot(lane, accept);
                remainingTriangles -= countbits(acceptMask);

                const uint summedContacts = ccScanInclusive(lane, nbContacts);

                gsCcRunSums[lane] = summedContacts - nbContacts;   // exclusive
                gsCcStartIdx[lane] = startIndex;                   // kept even when rejected
                gsCcTriInds[lane] = triIndex;

                const uint totalContacts = ccExchange(lane, summedContacts, CC_WARP - 1u);
                GroupMemoryBarrierWithGroupSync();

                // `totalContacts` and `currentContactMask` are warp-uniform, so
                // this loop's trip count is too and every barrier below is safe.
                for (uint c = 0u; c < totalContacts;)
                {
                    const uint readMask = ~currentContactMask;
                    const bool needsContact = (readMask & (1u << lane)) != 0u;
                    const uint readIndex = c + countbits(readMask & ((1u << lane) - 1u));

                    if (needsContact && readIndex < totalContacts)
                    {
                        const uint idx = ccBinarySearch32(readIndex);
                        const uint offset = readIndex - gsCcRunSums[idx];
                        const uint startIdx = gsCcStartIdx[idx] + start;
                        const uint contactStartIdx =
                            Ptr<uint>(contactBuffer + uint64_t(startIdx) * 4UL)[0];

                        const float4 v = Ptr<float4>(gCc.tempConvexTriContacts +
                                                     uint64_t(contactStartIdx + offset) * 16UL)[0];
                        pA = v.xyz;
                        separation = v.w;
                        contactTriangleIndex = gsCcTriInds[idx];
                        hasContact = true;
                    }

                    currentContactMask = ccBallot(lane, hasContact);

                    // `c` advances by the number of FREE LANES, not by the number
                    // of contacts actually read.
                    c += countbits(readMask);

                    if (countbits(currentContactMask) > CC_MAX_CONTACTS)
                        currentContactMask = crContactReduce6Cluster(lane, pA, separation,
                                                                     bestNormal,
                                                                     currentContactMask,
                                                                     gCc.clusterBias);

                    hasContact = (currentContactMask & (1u << lane)) != 0u;
                }

                // gsCcRunSums is read and written across iterations of the
                // enclosing loop, so the write below must not race the reads above.
                GroupMemoryBarrierWithGroupSync();

                if (accept)
                    Ptr<uint>(tempRun + uint64_t(i + lane) * 4UL)[0] = 0u;   // triangle done
            }

            // The next patch's deepest scan reads every tempCounts slot, including
            // ones other lanes just cleared -- device ordering, not groupshared.
            AllMemoryBarrierWithGroupSync();

            // ---- output this patch ------------------------------------------
            const uint nbContacts = countbits(currentContactMask);
            const uint slot = countbits(currentContactMask & ((1u << lane) - 1u));

            // Written even when the patch is empty.
            if (lane == 0u)
                Ptr<uint>(outBase + uint64_t(CC_NB_CONTACTS_OFF) + uint64_t(nbPatches) * 4UL)[0]
                    = nbContacts;

            if ((currentContactMask & (1u << lane)) != 0u)
            {
                const float3 nor = ccQuatRotate(aToBq, bestNormal);
                const float3 pointB = ccQuatRotate(aToBq, pA) + aToBp + nor * separation;

                const uint64_t cp = outBase + uint64_t(nbPatches * CC_MAX_CONTACTS + slot) * 48UL;
                Ptr<float>(cp + 0UL)[0] = pA.x;
                Ptr<float>(cp + 4UL)[0] = pA.y;
                Ptr<float>(cp + 8UL)[0] = pA.z;
                Ptr<float>(cp + 12UL)[0] = pointB.x;
                Ptr<float>(cp + 16UL)[0] = pointB.y;
                Ptr<float>(cp + 20UL)[0] = pointB.z;
                Ptr<float>(cp + 24UL)[0] = -nor.x;
                Ptr<float>(cp + 28UL)[0] = -nor.y;
                Ptr<float>(cp + 32UL)[0] = -nor.z;
                Ptr<float>(cp + 36UL)[0] = separation;
                Ptr<uint>(cp + 40UL)[0] = contactTriangleIndex;
            }
        }

        // mRelativeTransform is written ONLY when a patch was produced; the CUDA
        // stages it through PxAlignedTransform_WriteWarp's eight lanes, which is
        // the same thirty-two bytes.  q comes first (AlignedTransform.h:43).
        if (nbPatches > 0u && lane == 0u)
        {
            Ptr<float4>(outBase + uint64_t(CC_REL_XF_OFF))[0] = aToBq;
            Ptr<float4>(outBase + uint64_t(CC_REL_XF_OFF) + 16UL)[0] = float4(aToBp, 0.0f);
        }

        if (lane == 0u)
            Ptr<uint>(outBase + uint64_t(CC_NB_MANIFOLDS_OFF))[0] = nbPatches;
    }
}
