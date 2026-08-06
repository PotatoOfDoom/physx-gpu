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

// pxgSlangContactReduce.slang
//
// `contactReduceShared` (contactReduction.cuh:55-165) as a 32-lane virtual warp,
// at the ONE instantiation the convex-vs-triangle emit tail reaches:
// `contactReduce<false, true, 5, true>` (convexTriangle.cuh:1789), i.e.
// TClustering = false, TKeepAnotherDeepestForPCM = true, TMaxPoints = 5,
// TDoDupeTest = true, and — through the `contactReduce` wrapper at
// contactReduction.cuh:168-178 — clusterBias = 0, distanceAdjustment = 0 and
// initialPointCriteria = separation.
//
// WHY A SECOND PORT OF SOMETHING ALREADY GREEN.  `contactReduce.slang` covers the
// same algorithm and its test drives 4096 random cases through it, but it is a
// SCALAR harness: one thread walks all 32 contacts out of per-lane arrays.  That
// proves the ALGORITHM and nothing about the warp.  The emit tail needs the real
// thing — one lane per contact, every cross-lane read a groupshared exchange —
// and the two places a warp version can diverge from a scalar one (the argmax
// tie-break and the dupe test's lane-index rule) are precisely the places the
// scalar form cannot exercise.  So this module is checked against the SAME C++
// oracle the scalar port uses, which is what makes a divergence attributable to
// the warp emulation rather than to the algorithm.
//
// ONE WARP PER GROUP.  Every consumer MUST declare `[numthreads(32,1,1)]`, for
// the same reason pxgSlangPolyClip does: the reductions here carry barriers, and
// llvmpipe does not error on a divergent barrier, it HANGS.
//
// ALLMASK IS A UNIFORMITY PRECONDITION.  The dupe loop iterates over the set bits
// of `allMask` and every iteration performs four shuffles, i.e. eight barriers.
// The trip count is therefore warp-uniform only if `allMask` is.  At the emit-tail
// call site it is: `allMask` starts life as `(1 << numContactsGenerated) - 1` and
// `numContactsGenerated` is a popcount of a ballot (convexTriangle.cuh:939), which
// is uniform by construction.  A caller that passes a per-lane mask hangs.
//
// THE THREE STAGES AFTER THE DUPE CULL are all argmax/argmin over the warp, and
// `mask` stays warp-uniform through all of them because every lane computes the
// same anchors — which is what makes the `popc(mask) == 4` branch and the fill
// loop uniform, and therefore safe to contain barriers.

// SELF-CONTAINED, no `#include "pxgSlangWarpInc.slang"`, for the reason
// pxgSlangPolyClip is: the emit tail composes THIS module with
// pxgSlangConvexSat, which does include it.  A Slang `import` is a module
// boundary, so each importer would carry its own copy of the warp library's
// groupshared scratch, and the primitives here are three functions.  Everything
// is prefixed `cr`/`gsCr` so nothing can collide with `gsPc*` (pxgSlangPolyClip)
// or `gsCs*` (pxgSlangConvexSat) when all three land in one shader.

static const uint CR_WARP = 32u;
static const float CR_MAX_F32 = 3.4028234663852886e+38f;   // PX_MAX_F32 / FLT_MAX
public static const uint CR_MAX_POINTS = 5u;               // TMaxPoints at this call site
public static const uint CR_MAX_POINTS_CLUSTER = 6u;       // TMaxPoints at the correlate call site

groupshared uint gsCrStage[CR_WARP];

// __shfl_sync / __shfl_xor_sync: store / barrier / read / barrier, so the array
// is reusable immediately and the NEXT call needs no leading barrier of its own.
uint crExchange(uint lane, uint value, uint src)
{
    gsCrStage[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    const uint r = gsCrStage[src & (CR_WARP - 1u)];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

float crShflF(uint lane, float value, uint src)
{
    return asfloat(crExchange(lane, asuint(value), src));
}

float crShflXorF(uint lane, float value, uint xorMask)
{
    return asfloat(crExchange(lane, asuint(value), lane ^ xorMask));
}

// __ballot_sync(FULL_MASK, pred).  The CUDA's set-mask AND is applied at the call
// site, not here, because that is where it appears in the original.
uint crBallot(uint lane, bool pred)
{
    gsCrStage[lane] = pred ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    uint b = 0u;
    for (uint i = 0u; i < CR_WARP; ++i)
        b |= gsCrStage[i] != 0u ? (1u << i) : 0u;
    GroupMemoryBarrierWithGroupSync();
    return b;
}

// __ffs(v) - 1, with the CUDA convention that an empty word gives -1.
int crLowestSetIndex(uint v) { return v == 0u ? -1 : int(firstbitlow(v)); }

// --------------------------------------------------------------------------
// maxIndex / minIndex (nputils.cuh:137-149 and :151-163)
//
// TRANSCRIBED BY HAND RATHER THAN CALLED FROM pxgSlangWarpInc, and the difference
// is load-bearing.  `pxgWarpReduceMaxWinnerF` (pxgSlangWarpInc.slang:313) resolves
// the winner as `lowestSetIndex(ballot(mask, best == value))` — it passes ONE mask
// to both the sync role and the selection role.  The CUDA passes FULL_MASK to the
// ballot and ANDs the RESULT with the set mask:
//
//     maxV = mask & (1 << lane) ? v : -FLT_MAX;      // out-of-set lanes neutralised
//     ... butterfly over all 32 lanes ...
//     return lowestSetIndex(__ballot_sync(FULL_MASK, maxV == v) & mask);
//
// The equality is tested against the lane's ORIGINAL `v`, which for an out-of-set
// lane is its raw value, not -FLT_MAX.  So an out-of-set lane holding a value equal
// to the in-set maximum sets its ballot bit, and only the `& mask` removes it.
// Ballotting over the set mask instead — which is what the generic helper does —
// happens to agree here, but only because the neutralised lanes cannot WIN; the
// spellings stop agreeing the moment the caller's mask and sync mask differ.  The
// CUDA's composition is reproduced literally.
//
// `maxV` / `minV` are out parameters in the CUDA and every call site at this
// instantiation discards them.  They are kept so the signature still mirrors the
// original.
uint crMaxIndex(uint lane, float v, uint mask, out float maxV)
{
    float m = ((mask >> lane) & 1u) != 0u ? v : -CR_MAX_F32;

    m = max(m, crShflXorF(lane, m, 16u));
    m = max(m, crShflXorF(lane, m, 8u));
    m = max(m, crShflXorF(lane, m, 4u));
    m = max(m, crShflXorF(lane, m, 2u));
    m = max(m, crShflXorF(lane, m, 1u));

    maxV = m;
    return uint(crLowestSetIndex(crBallot(lane, m == v) & mask));
}

uint crMinIndex(uint lane, float v, uint mask, out float minV)
{
    float m = ((mask >> lane) & 1u) != 0u ? v : CR_MAX_F32;

    m = min(m, crShflXorF(lane, m, 16u));
    m = min(m, crShflXorF(lane, m, 8u));
    m = min(m, crShflXorF(lane, m, 4u));
    m = min(m, crShflXorF(lane, m, 2u));
    m = min(m, crShflXorF(lane, m, 1u));

    minV = m;
    return uint(crLowestSetIndex(crBallot(lane, m == v) & mask));
}

// `shuffle(FULL_MASK, PxVec3, i)` — three scalar shuffles, exactly as the CUDA's
// PxVec3 overload expands to.
float3 crShfl3(uint lane, float3 v, uint src)
{
    return float3(crShflF(lane, v.x, src),
                  crShflF(lane, v.y, src),
                  crShflF(lane, v.z, src));
}

// --------------------------------------------------------------------------
// contactReduceShared<false, true, 5, true>
//
// Returns the surviving lane mask.  `allMask` must be warp-uniform (see header).
// --------------------------------------------------------------------------
public uint crContactReduce5(uint lane, float3 pointRaw, float separation, float3 normal, uint allMask)
{
    // ---- TDoDupeTest (contactReduction.cuh:64-77) --------------------------
    // A lane is a dupe of an earlier-or-deeper coincident lane.  The tie-break
    // `sep == separation && i > threadIndexInWarp` keeps the LOWER lane when two
    // coincident contacts share a separation exactly, and it is the only thing
    // that makes the cull deterministic — without it two lanes cull each other and
    // both vanish.
    {
        bool imADupe = false;
        for (uint m = allMask; m != 0u; m = m & (m - 1u))
        {
            const uint i = uint(crLowestSetIndex(m));
            const float3 pi = crShfl3(lane, pointRaw, i);
            const float3 dv = pi - pointRaw;
            const float d = dot(dv, dv);
            const float sep = crShflF(lane, separation, i);

            if (d < 1e-8f && (sep < separation || (sep == separation && i > lane)))
                imADupe = true;
        }
        allMask = allMask & ~crBallot(lane, imADupe);
    }

    // ---- the early-out (contactReduction.cuh:79-84) -------------------------
    // Warp-uniform: allMask is uniform, so every lane leaves together.
    if (countbits(allMask) <= CR_MAX_POINTS)
        return allMask;

    // The clipped point, projected onto the normal plane.  Everything after this
    // measures distances in that plane; only the PCM step and the fill loop go
    // back to the raw separation.
    const float3 point = pointRaw - normal * dot(pointRaw, normal);

    float v, w;

    // THE `distanceAdjustment` TERM IS DROPPED, not overlooked.  The CUDA writes
    // `maxIndex(initialPointCriteria + distanceAdjustment, ...)` and three more of
    // the same shape; at this instantiation `distanceAdjustment` is the literal 0
    // the `contactReduce` wrapper passes (contactReduction.cuh:174).  Adding it
    // cannot move a decision: it changes at most the SIGN of a zero, and every use
    // of the result is an ordering comparison, under which -0 and +0 are equal.
    // `contactReduce2` (:180-190) passes -separation instead and would need the
    // term back — it is not reached from the convex-vs-triangle emit tail.

    // p0 — the deepest contact.  initialPointCriteria is `separation` here, so the
    // criterion is the raw separation and the comparison is exact.
    const uint i0 = crMaxIndex(lane, separation, allMask, v);
    uint mask = 1u << i0;

    // p1 — furthest from p0 in the normal plane.
    const float3 dv0 = crShfl3(lane, point, i0) - point;
    const float dist = sqrt(dot(dv0, dv0));
    const uint i1 = crMaxIndex(lane, dist, allMask & ~mask, v);
    mask |= 1u << i1;

    // p2 / p3 — the two extremes either side of the p0-p1 edge, measured by the
    // signed distance along `normal x (p1 - p0)`.  The CUDA re-shuffles point[i0]
    // here rather than reusing it; same value, and the shuffle is kept so the
    // barrier count matches the original's structure.
    const float3 dir = cross(normal, crShfl3(lane, point, i1) - crShfl3(lane, point, i0));
    const float d = dot(dir, point - crShfl3(lane, point, i0));

    const uint f = crMaxIndex(lane, d, allMask & ~mask, v);
    mask |= 1u << f;

    const uint g = crMinIndex(lane, d, allMask & ~mask, w);
    mask |= 1u << g;

    // ---- TKeepAnotherDeepestForPCM (contactReduction.cuh:118-126) -----------
    // `popc(mask) == 4` means the four anchors landed on four DISTINCT lanes.
    // They need not: maxIndex over `allMask & ~mask` can return a lane already in
    // `mask` only if the set is exhausted, which it is not here, but the four
    // picks can still collide through the `1 << i` when two anchors coincide — the
    // CUDA tests the popcount rather than assuming, and so does this.
    //
    // `mask` is warp-uniform (every lane computed the same i0/i1/f/g), so this
    // branch is uniform and the barriers inside crMinIndex are safe.
    if (countbits(mask) == 4u)
    {
        const uint i4 = crMinIndex(lane, separation, allMask & ~mask, v);
        mask |= 1u << i4;
    }

    // ---- TClustering == false (contactReduction.cuh:154-160) ----------------
    // Fill the remaining slots with the deepest unselected contacts.  The trip
    // count is uniform for the same reason.
    const uint count = countbits(mask);
    for (uint i = count; i < CR_MAX_POINTS; ++i)
        mask |= 1u << crMinIndex(lane, separation, allMask & ~mask, v);

    return mask;
}

// ---------------------------------------------------------------------------
// THE SECOND INSTANTIATION: `contactReduce<true, false, 6, false>`
// (convexMeshCorrelate.cu:277), which is what `convexTrimeshCorrelate` reaches.
//
// Three differences from `crContactReduce5`, and the third is a whole branch the
// module did not have:
//   * NO DUPE TEST.  Coincident contacts survive into the anchor search.
//   * NO PCM-DEEPEST STEP.  `TKeepAnotherDeepestForPCM` is false, so the
//     `popc(mask) == 4` block is gone entirely — with six slots to fill the fill
//     stage reaches those points anyway.
//   * TClustering IS TRUE.  Instead of filling the remaining slots with the
//     deepest unselected contacts, every contact is LABELLED with the anchor it
//     is closest to in the normal plane, one representative is taken from each
//     non-empty cluster, and only then are the leftover slots filled.  The anchors
//     themselves are biased deeper by `clusterBias` so a cluster prefers the
//     anchor that defined it, which is what stops the manifold flickering between
//     near-equal contacts frame to frame.
//
// `clusterBias` is a real value here, not the zero the emit tail passes: the host
// hands `convexTrimeshCorrelate` a per-scene bias (PxgNarrowphaseCore.cpp:2718).
//
// EVERY LOOP BELOW HAS A WARP-UNIFORM TRIP COUNT, which is what makes the
// barriers inside crShfl3/crBallot/crMinIndex safe.  `mask` after the four anchors
// is uniform (every lane computed the same i0/i1/f/g), `nbClusters` is derived
// from it, and `cluster` is a ballot ANDed with a uniform mask — so the
// `if (cluster != 0u)` guard is uniform too.  llvmpipe does not error on a
// divergent barrier, it HANGS.
//
// FOUR OF THIRTY-TWO SWEEP MUTANTS SURVIVE, and each has a proof rather than an
// excuse:
//
//   * `label` initialised to 0 instead of -1.  The initial value is never read:
//     the first anchor's distance is finite and the test is `dd < CR_MAX_F32`, so
//     every lane is assigned on the first iteration.
//   * the edge direction reversed, `p0 - p1` for `p1 - p0`.  That negates `d`,
//     which swaps argmax and argmin — so `f` and `g` trade places and the anchor
//     SET is identical.  `mask` is a bitmask and the cluster walk goes by LANE
//     order, not insertion order, so nothing downstream can tell.
//   * `d` measured from `point[i1]` instead of `point[i0]`.  The two differ by
//     `dot(dir, p1 - p0)`, and `dir` is `normal x (p1 - p0)` — perpendicular to
//     it by construction, so that offset is exactly zero in real arithmetic.  In
//     floats it is a residue many orders of magnitude below any tie the fixture
//     constructs, and it shifts every lane equally, so no ordering moves.
//   * the p1 distance left squared.  This one is the deliberate control: argmax is
//     invariant under any monotone transform, and `sqrt` is one.  It must survive,
//     or the sweep is only proving that the fixture rejects every edit.
//
// THE FILL LOOP CANNOT RUN OUT OF CANDIDATES.  `allMask` has more than six bits
// (the early-out saw to that) and at most six ever get selected, so
// `allMask & ~mask` is never empty and `crMinIndex` is never asked to reduce over
// nothing — which would hand back `crLowestSetIndex(0) == -1` and shift by it.
// ---------------------------------------------------------------------------
public uint crContactReduce6Cluster(uint lane, float3 pointRaw, float separation, float3 normal,
                                    uint allMask, float clusterBias)
{
    // Warp-uniform: allMask is uniform, so every lane leaves together.
    if (countbits(allMask) <= CR_MAX_POINTS_CLUSTER)
        return allMask;

    const float3 point = pointRaw - normal * dot(pointRaw, normal);

    float v, w;

    // p0 — the deepest contact.  `initialPointCriteria` is `separation` and
    // `distanceAdjustment` is the literal 0 the wrapper passes, so the criterion
    // is the raw separation and the comparison is exact.
    const uint i0 = crMaxIndex(lane, separation, allMask, v);
    uint mask = 1u << i0;

    // p1 — furthest from p0 in the normal plane.
    const float3 dv0 = crShfl3(lane, point, i0) - point;
    const float dist = sqrt(dot(dv0, dv0));
    const uint i1 = crMaxIndex(lane, dist, allMask & ~mask, v);
    mask |= 1u << i1;

    // p2 / p3 — the extremes either side of the p0-p1 edge.
    const float3 dir = cross(normal, crShfl3(lane, point, i1) - crShfl3(lane, point, i0));
    const float d = dot(dir, point - crShfl3(lane, point, i0));

    const uint f = crMaxIndex(lane, d, allMask & ~mask, v);
    mask |= 1u << f;

    const uint g = crMinIndex(lane, d, allMask & ~mask, w);
    mask |= 1u << g;

    // ---- TClustering == true (contactReduction.cuh:130-153) -----------------
    // The bias is subtracted from the ANCHORS only, and only for the cluster
    // representative search below — `separation` itself is untouched.
    float sep = separation;
    if (((mask >> lane) & 1u) != 0u)
        sep -= clusterBias;

    // Label every lane with the nearest anchor.  The CUDA consumes `mask` as it
    // walks it and then zeroes it; walking a copy and assigning zero afterwards is
    // the same thing said once.  Ties go to the EARLIER anchor: the test is a
    // strict `<`, so the first anchor at a given distance keeps the label.
    int nbClusters = 0;
    int label = -1;
    {
        float t = CR_MAX_F32;
        for (uint m = mask; m != 0u; ++nbClusters, m = m & (m - 1u))
        {
            const uint anchor = uint(crLowestSetIndex(m));
            const float3 dv = point - crShfl3(lane, point, anchor);
            const float dd = dot(dv, dv);
            if (dd < t)
            {
                t = dd;
                label = nbClusters;
            }
        }
    }

    // One representative per non-empty cluster: the deepest, with the anchor's
    // bias already applied.  A cluster can be empty of IN-SET lanes even though it
    // has an anchor, because labels are assigned to every lane while the pick is
    // restricted to `allMask` — hence the guard rather than an unconditional pick.
    mask = 0u;
    for (int i = 0; i < nbClusters; ++i)
    {
        const uint cluster = crBallot(lane, label == i) & allMask;
        if (cluster != 0u)
            mask |= 1u << crMinIndex(lane, sep, cluster, v);
    }

    // Fill the rest.  NOTE the loop starts at `nbClusters`, NOT at
    // `countbits(mask)` — the CUDA counts CLUSTERS here, not selected points, so
    // an empty cluster costs a slot that is never filled.  That is the engine's
    // behaviour and it is reproduced rather than tidied.
    for (int i = nbClusters; i < int(CR_MAX_POINTS_CLUSTER); ++i)
        mask |= 1u << crMinIndex(lane, sep, allMask & ~mask, v);

    return mask;
}
