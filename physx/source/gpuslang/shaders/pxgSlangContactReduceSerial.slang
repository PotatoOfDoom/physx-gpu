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
}

public CrsConfig crsConfig(bool clustering, bool keepDeepestForPCM, int maxPoints)
{
    CrsConfig c;
    c.clustering = clustering;
    c.keepDeepestForPCM = keepDeepestForPCM;
    c.maxPoints = maxPoints;
    return c;
}

// *** TDoDupeTest IS DELIBERATELY NOT A FIELD HERE. ***
// contactReduceShared's fourth template argument guards a fifteen-line duplicate
// cull (contactReduction.cuh:65-79) that only ONE instantiation in this corpus
// turns on: npTriHfNp::contactReduce6Dupe, <true, true, 6, TRUE>.  That body is
// still its own transcription and is NOT folded in here yet.
//
// The tempting shape was a fourth bool that this body ignores.  A config field
// that is accepted and silently does nothing is exactly the failure this module
// exists to end: a future caller would pass `true`, get no dupe cull, and have
// every reason to believe it had one.  So the field does not exist, and a caller
// that needs the dupe test cannot express it -- which is a compile error rather
// than a wrong manifold.
//
// Folding npTriHfNp in is the remaining half of this consolidation and needs its
// fixture to prove the dupe arm, not just the arms it shares.

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
