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

// pxgSlangContactReduceSerial -- P9.83b-2.
//
//   contactReduceShared (contactReduction.cuh:55-165) as ONE serial body, with
//   its four template arguments and its two wrapper inputs as parameters.
//
// ---------------------------------------------------------------------------
// WHY THIS EXISTS: ONE CUDA FUNCTION, FOUR INSTANTIATIONS, FOUR TRANSCRIPTIONS
// ---------------------------------------------------------------------------
// contactReduceShared is a single CUDA template.  This corpus had transcribed it
// FOUR times, once per instantiation it needed, and the four had already started
// to differ in ways that are invisible from any one of them:
//
//   pxgSlangTriPlane::contactReduce6       <true,  true,  6, false>  serial
//   npCvxCorrelate::contactReduce6         <true,  false, 6, false>  serial
//   npTriHfNp::contactReduce6Dupe          <true,  true,  6, true >  serial
//   pxgSlangContactReduce (the warp form)  <false, true,  5, true >  barriers
//
// The three SERIAL ones are now thin adapters to this body.  The warp form is
// not, and cannot be without restructuring its callers: its calling convention is
// per-lane with barriers inside, so it is a different shape, not a different
// instantiation.  See the note at the end of this header.
//
// Template instantiations of one body ARE one body -- that is precisely the case
// this port's sharing rule names -- and a fifth was about to be written, because
// triangleTriangleCollision needs `contactReduce2<true, true, 6, false>`, which
// is the SAME body reached through the OTHER wrapper.
//
// ---------------------------------------------------------------------------
// THE TWO WRAPPERS, AND WHY THE DIFFERENCE IS NOT COSMETIC
// ---------------------------------------------------------------------------
// contactReduction.cuh has two entry wrappers over one shared body:
//
//   contactReduce  (:171-177)   distanceAdjustment = 0
//                               initialPointCriteria = separation
//   contactReduce2 (:180-188)   distanceAdjustment = -separation
//                               initialPointCriteria = |point| (the point
//                                 PROJECTED onto the normal plane, :185)
//
// `distanceAdjustment` enters the body at FOUR sites, and THE FOURTH IS
// SUBTRACTED WHERE THE FIRST THREE ARE ADDED:
//
//   :91   i0 = argmax(initialPointCriteria + adj)
//   :94   i1 = argmax(dist            + adj)
//   :107  f  = argmax(d               + adj)
//   :110  g  = argmin(d               - adj)     <-- the sign flip
//
// Under contactReduce the adjustment is zero, so all four collapse and every
// existing transcription reuses ONE array for f and g (pxgSlangTriPlane's old
// body wrote `vals[j] = dval[j]` twice).  Under contactReduce2 they are two
// different criteria: f is argmax(d - sep) and g is argmin(d + sep).  Copying the
// collapsed form into the mesh-mesh kernel picks the wrong p3 anchor on most
// non-degenerate manifolds, and the result is a plausible-looking 6-point patch.
//
// THE PCM PICK AND THE CLUSTERING USE THE RAW SEPARATION, unadjusted (:126, :164),
// while the clustering's own pick uses sep biased by clusterBias (:135-137,
// :153, :157).  Both spellings were already right in the serial body; they are
// carried over unchanged.
//
// ---------------------------------------------------------------------------
// WHY SERIAL AND NOT THE WARP FORM
// ---------------------------------------------------------------------------
// The warp-emulated transcription (pxgSlangContactReduce) implements every
// __shfl_sync/__ballot_sync as a GroupMemoryBarrierWithGroupSync.  That is
// correct where the whole block reaches it, and a HANG where it does not:
// convexMeshMidphase.cu calls reduceContactsSingleWarp under `if (warpIndex == 0)`
// (:842, :950), which in Slang would put thousands of workgroup barriers under a
// warp-varying guard in a 1024-thread block.  check_barrier_uniformity.py pass 3
// exists to catch exactly that shape.  The serial body has no barriers at all, so
// a caller may run it on one thread -- which is what triPlaneNpBda already does
// and what the mesh-mesh kernel will do.
// ---------------------------------------------------------------------------
// FIVE MUTANTS SURVIVE THIS MODULE'S SWEEP, AND FOUR OF THEM PROVABLY MUST
// ---------------------------------------------------------------------------
// P9.83b-2's sweep killed 20 of 25.  All five survivors are recorded here with
// their arguments, so a later sweep does not re-investigate them and -- more
// importantly -- so nobody "simplifies" the CUDA's spelling on the grounds that no
// test objects.  The spellings below are kept because they are the ENGINE's, not
// because a fixture can tell.
//
// FOUR ARE PROVABLY EQUIVALENT AND ONE IS EQUIVALENT ON EVERY REACHABLE INPUT.
// The fifth was first diagnosed as a fixture gap and it was not; see
// earlyout-strict below, which is the one worth reading.
//
//   i1-anchor-order   `point[i0] - point[j]`  ->  `point[j] - point[i0]`
//       Only the MAGNITUDE of that difference is used.  IEEE subtraction gives
//       exactly opposite signs, so the squares and their sum are bit-identical.
//       Provably equivalent.
//
//   dir-cross-order   `cross(normal, p1 - p0)`  ->  `cross(p1 - p0, normal)`
//       Negates `dir`, hence every dval.  argmax(-d + a) is argmin(d - a) and
//       argmin(-d - a) is argmax(d + a), so f and g simply SWAP and the mask is
//       the same set.  Equivalent except where f and g would be the SAME lane (a
//       total tie in dval), where the pick order decides which lane the second
//       draw excludes.  Random float inputs do not produce that tie.
//
//   dval-origin       `point[j] - point[i0]`  ->  `point[j] - point[i1]`
//       Adds the SAME constant dot(dir, point[i1] - point[i0]) to every lane's
//       dval.  A constant added to all lanes preserves every argmax and argmin
//       over dval +/- adj, so both picks are unchanged.  (Mathematically that
//       constant is zero -- dir is perpendicular to p1 - p0 by construction -- but
//       the equivalence does not depend on that; it only needs the shift to be
//       uniform, which it is even with the cross product rounded.)
//
//   cluster-strict-vs-loose   `d < tbest[j]`  ->  `d <= tbest[j]`
//       Changes which anchor claims a lane only on an EXACT tie of squared
//       distance.  The CUDA writes `<` (contactReduction.cuh:147) and so does
//       this; no random-float fixture will separate them.
//
//   earlyout-strict   `newCount <= maxPoints`  ->  `newCount < maxPoints`
//       DIAGNOSED WRONG THE FIRST TIME, and the correction is the interesting
//       part.  It was called a fixture gap: the two spellings differ only at
//       newCount == maxPoints, and the fixture had been built to give every case
//       MORE than maxPoints live lanes, so nothing touched the boundary.  One case
//       in seven was given exactly six live lanes -- and the mutant survived
//       ANYWAY, which is the answer.
//
//       At newCount == maxPoints the reduction cannot do anything but return the
//       live set.  It picks i0, i1, f, g (four distinct lanes), possibly i4 (a
//       fifth), then labels every lane with its nearest anchor and takes one lane
//       per cluster -- clusters are disjoint, so those picks are distinct -- and
//       finally fills `for i in nbClusters..maxPoints` with one new lane each.
//       The total is exactly maxPoints lanes drawn from a pool of exactly
//       maxPoints, so the result IS allMask.  The early-out is a shortcut, not a
//       behaviour, and the two spellings are provably equivalent.
//
//       The boundary cases stay.  They no longer kill a mutant, but they assert
//       the trivial-return contract of contactReduction.cuh:85 directly, for both
//       wrappers, which nothing else did.
module pxgSlangContactReduceSerial;

public static const float CRS_FLT_MAX = 3.402823466e38f;

// NOTE: with an empty mask (or all masked values at the sentinel) these return
// idx 0, unlike CUDA maxIndex/minIndex (lowest MASKED index); every call site
// provably passes a nonempty mask with finite values.  Moved here verbatim from
// pxgSlangTriPlane, which now imports them back.
public uint crsArgMax(float vals[32], uint mask, out float best)
{
    best = -CRS_FLT_MAX;
    uint idx = 0;
    for (uint j = 0; j < 32; ++j)
        if (((mask >> j) & 1) != 0 && vals[j] > best) { best = vals[j]; idx = j; }
    return idx;
}
public uint crsArgMin(float vals[32], uint mask, out float best)
{
    best = CRS_FLT_MAX;
    uint idx = 0;
    for (uint j = 0; j < 32; ++j)
        if (((mask >> j) & 1) != 0 && vals[j] < best) { best = vals[j]; idx = j; }
    return idx;
}

// The template arguments, as a value rather than a specialisation.  Slang has
// generics, but a struct of four flags keeps every instantiation visible at its
// call site -- which is the property that was missing when four transcriptions
// drifted apart.
public struct CrsConfig
{
    public bool clustering;        // TClustering
    public bool keepDeepestForPCM; // TKeepAnotherDeepestForPCM
    public int maxPoints;          // TMaxPoints
    public bool dupeTest;          // TDoDupeTest
}

public CrsConfig crsConfig(bool clustering, bool keepDeepestForPCM, int maxPoints, bool dupeTest)
{
    CrsConfig c;
    c.clustering = clustering;
    c.keepDeepestForPCM = keepDeepestForPCM;
    c.maxPoints = maxPoints;
    c.dupeTest = dupeTest;
    return c;
}

// TDoDupeTest IS A FIELD, AND FOR ONE COMMIT IT DELIBERATELY WAS NOT.
// When this module first landed the dupe cull was not implemented, and the
// tempting shape was a fourth bool the body would accept and ignore.  That is the
// precise failure the module exists to end -- a caller would pass `true`, get no
// cull, and have every reason to believe it had one -- so the field was left OUT
// and asking for the cull was a compile error rather than a wrong manifold.
// Now the cull is implemented (contactReduction.cuh:65-79) and the field is back.
// The sequence is recorded because "add the flag now, implement it later" is the
// shape that produces the silent version.

// ---------------------------------------------------------------------------
// The shared body (contactReduction.cuh:55-165), serial.
//
// `adj` is the CUDA's distanceAdjustment, per lane; `initial` is its
// initialPointCriteria, per lane.  Both are arrays and not scalars because
// contactReduce2 derives them PER LANE from that lane's separation and point.
// ---------------------------------------------------------------------------
public uint crsReduceShared(float3 ptRaw[32], float sep[32], float3 normal,
                            uint allMask, float clusterBias,
                            float adj[32], float initial[32], CrsConfig cfg)
{
    // THE DUPE CULL RUNS FIRST, BEFORE newCount (contactReduction.cuh:65-79).
    // A lane dies to a COINCIDENT lane (squared distance < 1e-8f on the RAW
    // points, not the projected ones) that is either strictly deeper, or equally
    // deep with a HIGHER index.  The index tie-break is what makes it
    // deterministic: without it two exactly-coincident, exactly-equal contacts
    // would each cull the other and both would vanish.
    if (cfg.dupeTest)
    {
        uint dupes = 0;
        for (uint l = 0; l < 32; ++l)
        {
            if (((allMask >> l) & 1) == 0) continue;
            bool imADupe = false;
            for (uint m = allMask; m != 0; m &= (m - 1))
            {
                const uint i = firstbitlow(m);
                const float3 dv = ptRaw[i] - ptRaw[l];
                const float d = dot(dv, dv);
                if (d < 1e-8f && (sep[i] < sep[l] || (sep[i] == sep[l] && i > l)))
                    imADupe = true;
            }
            if (imADupe) dupes |= 1u << l;
        }
        allMask &= ~dupes;
    }

    float3 point[32];
    for (uint j = 0; j < 32; ++j)
        point[j] = ptRaw[j] - normal * dot(ptRaw[j], normal);

    const uint newCount = countbits(allMask);
    if (newCount <= uint(cfg.maxPoints))
        return allMask;

    float v;
    float vals[32];

    // p0: argmax(initialPointCriteria + distanceAdjustment)   (:91)
    for (uint j = 0; j < 32; ++j) vals[j] = initial[j] + adj[j];
    const uint i0 = crsArgMax(vals, allMask, v);
    uint mask = 1u << i0;

    // p1: argmax(|anchor - lane| + distanceAdjustment)        (:93-94)
    // THE SUBTRACTION ORDER IS ANCHOR MINUS LANE, and the magnitude is
    // PxVec3::magnitude() = sqrt(dot), never OpLength -- check_length_intrinsic.py
    // exists for that distinction.
    for (uint j = 0; j < 32; ++j)
    {
        const float3 d = point[i0] - point[j];
        vals[j] = sqrt(dot(d, d)) + adj[j];
    }
    const uint i1 = crsArgMax(vals, allMask & ~mask, v);
    mask |= 1u << i1;

    // p2/p3: the two extremes along dir = normal x (p1 - p0)  (:105-110)
    const float3 dir = cross(normal, point[i1] - point[i0]);
    float dval[32];
    for (uint j = 0; j < 32; ++j) dval[j] = dot(dir, point[j] - point[i0]);

    for (uint j = 0; j < 32; ++j) vals[j] = dval[j] + adj[j];
    const uint f = crsArgMax(vals, allMask & ~mask, v);
    mask |= 1u << f;

    // *** THE SIGN FLIP (:110 is `minIndex(d - distanceAdjustment, ...)`). ***
    for (uint j = 0; j < 32; ++j) vals[j] = dval[j] - adj[j];
    const uint g = crsArgMin(vals, allMask & ~mask, v);
    mask |= 1u << g;

    // the PCM extra-deepest pick uses the RAW separation (:121-128)
    if (cfg.keepDeepestForPCM && countbits(mask) == 4 && cfg.maxPoints > 4)
    {
        const uint i4 = crsArgMin(sep, allMask & ~mask, v);
        mask |= 1u << i4;
    }

    if (cfg.clustering)
    {
        float sepB[32];
        for (uint j = 0; j < 32; ++j)
            sepB[j] = sep[j] - (((mask >> j) & 1) != 0 ? clusterBias : 0.0f);

        int label[32];
        int nbClusters = 0;
        {
            uint m = mask;
            float tbest[32];
            for (uint j = 0; j < 32; ++j) { tbest[j] = CRS_FLT_MAX; label[j] = -1; }
            for (; m != 0; ++nbClusters, m &= (m - 1))
            {
                const uint anchor = firstbitlow(m);
                for (uint j = 0; j < 32; ++j)
                {
                    const float3 dv = point[j] - point[anchor];
                    const float d = dot(dv, dv);
                    if (d < tbest[j]) { tbest[j] = d; label[j] = nbClusters; }
                }
            }
        }

        mask = 0;
        for (int i = 0; i < nbClusters; ++i)
        {
            uint cluster = 0;
            for (uint j = 0; j < 32; ++j)
                if (label[j] == i) cluster |= 1u << j;
            cluster &= allMask;
            if (cluster != 0)
                mask |= 1u << crsArgMin(sepB, cluster, v);
        }
        for (int i = nbClusters; i < cfg.maxPoints; ++i)
            mask |= 1u << crsArgMin(sepB, allMask & ~mask, v);
    }
    else
    {
        // contactReduction.cuh:159-165.  THIS BRANCH WAS MISSING FOR TWO
        // COMMITS and the omission was invisible, because all three callers
        // folded in so far pass TClustering = true.  It bites the moment a
        // clustering=false caller is folded -- npCvxTriEmit::contactReduce5 and
        // pxgSlangConvexTriCloth::contactReduce5, which are <false,true,5,true>
        // and BOTH implement this loop correctly in their own bodies.  Folding
        // them onto the module as it stood would have silently dropped the
        // fill-out and emitted a short manifold: the shared body less complete
        // than the copies it replaced, which is the failure mode consolidation
        // is supposed to remove rather than centralise.
        //
        // The fill uses the RAW separation, not the cluster-biased sepB -- sepB
        // does not exist on this path at all.
        uint count = countbits(mask);
        for (uint i = count; i < uint(cfg.maxPoints); ++i)
            mask |= 1u << crsArgMin(sep, allMask & ~mask, v);
    }
    return mask;
}

// ---------------------------------------------------------------------------
// WRAPPER 1: contactReduce (contactReduction.cuh:171-177).
//   distanceAdjustment = 0, initialPointCriteria = separation.
// ---------------------------------------------------------------------------
public uint crsContactReduce(float3 ptRaw[32], float sep[32], float3 normal,
                             uint allMask, float clusterBias, CrsConfig cfg)
{
    float adj[32], initial[32];
    for (uint j = 0; j < 32; ++j) { adj[j] = 0.0f; initial[j] = sep[j]; }
    return crsReduceShared(ptRaw, sep, normal, allMask, clusterBias, adj, initial, cfg);
}

// ---------------------------------------------------------------------------
// WRAPPER 2: contactReduce2 (contactReduction.cuh:180-188).
//   distanceAdjustment = -separation,
//   initialPointCriteria = |pointRaw projected onto the normal plane|.
//
// The projection is the SAME one the body computes for `point`, so the criterion
// is the projected point's distance from the ORIGIN -- not from any anchor, and
// not the separation.  That makes p0 the most EXTREME contact rather than the
// DEEPEST one, which is a different anchor set and therefore a different patch.
// contactReduction.cuh:43 flags this: its "ALGORITHM BELOW TAKES DEEPEST CONTACT
// AS p0" describes wrapper 1 only.
// ---------------------------------------------------------------------------
public uint crsContactReduce2(float3 ptRaw[32], float sep[32], float3 normal,
                              uint allMask, float clusterBias, CrsConfig cfg)
{
    float adj[32], initial[32];
    for (uint j = 0; j < 32; ++j)
    {
        const float3 p = ptRaw[j] - normal * dot(ptRaw[j], normal);
        adj[j] = -sep[j];
        initial[j] = sqrt(dot(p, p));
    }
    return crsReduceShared(ptRaw, sep, normal, allMask, clusterBias, adj, initial, cfg);
}
