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

// pxgCrsRef.h -- the host mirror of contactReduceShared (contactReduction.cuh:55-165)
// and its wrappers, kept in ONE place.  P9.83b-5 (14/n).
//
// WHY IT IS A HEADER.  Two fixtures had grown a private copy of the same oracle:
// testNpCrsWrapper2.cpp (the full body -- dupe prologue, both clustering arms, all
// four instantiations the port uses) and testNpTriTriReduce.cpp (only the
// <clustering=true, dupeTest=false, maxPoints=6> path reduceContactsSingleWarp
// needs).  A third fixture -- npTriTriBlock's per-warp reduce -- needed it too,
// and a third private copy is how this port got seven transcriptions of
// contactReduceShared on the SHADER side before P9.83b-2/-3 folded them.
//
// BEFORE CONSOLIDATING, the two copies were compared expression by expression on
// the one path they share: the normal-plane projection, the maxPoints early-out,
// i0/i1 (:91, :93-94), the edge direction (:105), f and g with the sign flip at
// :110, the PCM fifth point (:121-128), the biased separations and the anchor
// clustering (:135-158).  Identical, down to the loop that finds the anchor.  The
// only differences are the parts the reduced copy never needed: the TDoDupeTest
// prologue (:65-79) and the TClustering=false fill (:159-165), and `newCount`
// spelled as a loop in one and as popcount32 in the other.  So the FULL copy is
// the one moved here, VERBATIM -- `static` became `inline`, nothing else.
//
// WHAT REPLACES THE REDUNDANCY.  One mistake here now reaches three fixtures
// instead of one.  What guards it is unchanged: the mirror is written from the
// CUDA, the shader is written from the CUDA, separately, and both existing
// fixtures re-ran BIT-EXACT across the move -- which is the evidence that the
// oracle they certify against did not change when it moved.

#ifndef PXG_CRS_REF_H
#define PXG_CRS_REF_H

#include <cmath>
#include <cstdint>

namespace pxgtest
{
namespace crsref
{

struct V3 { float x, y, z; };
struct V4 { float x, y, z, w; };

inline V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
inline V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
inline float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross3(V3 a, V3 b)
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
inline V3 scale3(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
// PxVec3::magnitude() is PxSqrt(x*x + y*y + z*z) -- sqrt(dot), never a length
// intrinsic.  tools/check_length_intrinsic.py exists for that distinction and the
// mirror has to make the same choice or the comparison measures the choice.
inline float mag3(V3 a) { return std::sqrt(dot3(a, a)); }

inline constexpr float FLT_MAX_LOCAL = 3.402823466e38f;

inline uint32_t argMax(const float* vals, uint32_t mask, float& best)
{
    best = -FLT_MAX_LOCAL;
    uint32_t idx = 0;
    for (uint32_t j = 0; j < 32; ++j)
        if (((mask >> j) & 1) != 0 && vals[j] > best) { best = vals[j]; idx = j; }
    return idx;
}
inline uint32_t argMin(const float* vals, uint32_t mask, float& best)
{
    best = FLT_MAX_LOCAL;
    uint32_t idx = 0;
    for (uint32_t j = 0; j < 32; ++j)
        if (((mask >> j) & 1) != 0 && vals[j] < best) { best = vals[j]; idx = j; }
    return idx;
}

// contactReduceShared<TClustering=true, TKeepAnotherDeepestForPCM=true,
//                     TMaxPoints=6, TDoDupeTest=dupeTest>  (contactReduction.cuh:55-165)
inline uint32_t reduceShared(const V3* ptRaw, const float* sep, V3 normal,
                             uint32_t allMask, float clusterBias,
                             const float* adj, const float* initial, int maxPoints,
                             bool dupeTest, bool clustering)
{
    // contactReduction.cuh:65-79, transcribed from the CUDA and not from the
    // Slang.  In CUDA every lane runs this against every set bit of allMask and
    // the verdicts are collected with __ballot_sync, so the cull is simultaneous:
    // the mask a lane tests against is the PRE-cull mask, not one thinned by
    // earlier verdicts.  That is why the verdicts are accumulated here and applied
    // only afterwards -- culling in place would let a lane be saved by the death
    // of the lane that killed it.
    if (dupeTest)
    {
        uint32_t dupes = 0;
        for (uint32_t l = 0; l < 32; ++l)                       // threadIndexInWarp
        {
            if (((allMask >> l) & 1) == 0) continue;
            bool imADupe = false;
            for (uint32_t m = allMask; m; m &= (m - 1))
            {
                uint32_t i = 0;
                while (((m >> i) & 1u) == 0u) ++i;              // lowestSetIndex
                const V3 dv = sub(ptRaw[i], ptRaw[l]);          // RAW points, :71
                const float d = dot3(dv, dv);                   // magnitudeSquared
                if (d < 1e-8f && (sep[i] < sep[l] || (sep[i] == sep[l] && i > l)))
                    imADupe = true;
            }
            if (imADupe) dupes |= 1u << l;
        }
        allMask &= ~dupes;                                      // :78
    }

    V3 point[32];
    for (uint32_t j = 0; j < 32; ++j)
        point[j] = sub(ptRaw[j], scale3(normal, dot3(ptRaw[j], normal)));

    uint32_t newCount = 0;
    for (uint32_t j = 0; j < 32; ++j) newCount += (allMask >> j) & 1u;
    if (newCount <= uint32_t(maxPoints))
        return allMask;

    float v;
    float vals[32];

    for (uint32_t j = 0; j < 32; ++j) vals[j] = initial[j] + adj[j];      // :91
    const uint32_t i0 = argMax(vals, allMask, v);
    uint32_t mask = 1u << i0;

    for (uint32_t j = 0; j < 32; ++j)                                     // :93-94
        vals[j] = mag3(sub(point[i0], point[j])) + adj[j];
    const uint32_t i1 = argMax(vals, allMask & ~mask, v);
    mask |= 1u << i1;

    const V3 dir = cross3(normal, sub(point[i1], point[i0]));             // :105
    float dval[32];
    for (uint32_t j = 0; j < 32; ++j) dval[j] = dot3(dir, sub(point[j], point[i0]));

    for (uint32_t j = 0; j < 32; ++j) vals[j] = dval[j] + adj[j];         // :107
    const uint32_t f = argMax(vals, allMask & ~mask, v);
    mask |= 1u << f;

    for (uint32_t j = 0; j < 32; ++j) vals[j] = dval[j] - adj[j];         // :110 SIGN FLIP
    const uint32_t g = argMin(vals, allMask & ~mask, v);
    mask |= 1u << g;

    uint32_t bits = 0;
    for (uint32_t j = 0; j < 32; ++j) bits += (mask >> j) & 1u;
    if (bits == 4 && maxPoints > 4)                                       // :121-128
    {
        const uint32_t i4 = argMin(sep, allMask & ~mask, v);              // RAW sep
        mask |= 1u << i4;
    }

    if (clustering)                                                       // :135-158
    {
        float sepB[32];
        for (uint32_t j = 0; j < 32; ++j)
            sepB[j] = sep[j] - (((mask >> j) & 1) != 0 ? clusterBias : 0.0f);

        int label[32];
        int nbClusters = 0;
        {
            uint32_t m = mask;
            float tbest[32];
            for (uint32_t j = 0; j < 32; ++j) { tbest[j] = FLT_MAX_LOCAL; label[j] = -1; }
            for (; m != 0; ++nbClusters, m &= (m - 1))
            {
                uint32_t anchor = 0;
                while (((m >> anchor) & 1u) == 0u) ++anchor;
                for (uint32_t j = 0; j < 32; ++j)
                {
                    const V3 dv = sub(point[j], point[anchor]);
                    const float d = dot3(dv, dv);
                    if (d < tbest[j]) { tbest[j] = d; label[j] = nbClusters; }
                }
            }
        }

        mask = 0;
        for (int i = 0; i < nbClusters; ++i)
        {
            uint32_t cluster = 0;
            for (uint32_t j = 0; j < 32; ++j)
                if (label[j] == i) cluster |= 1u << j;
            cluster &= allMask;
            if (cluster != 0)
                mask |= 1u << argMin(sepB, cluster, v);
        }
        for (int i = nbClusters; i < maxPoints; ++i)
            mask |= 1u << argMin(sepB, allMask & ~mask, v);
    }
    else                                                              // :159-165
    {
        // The fill uses the RAW separation; sepB does not exist on this path.
        uint32_t count = 0;
        for (uint32_t j = 0; j < 32; ++j) count += (mask >> j) & 1u;
        for (uint32_t i = count; i < uint32_t(maxPoints); ++i)
            mask |= 1u << argMin(sep, allMask & ~mask, v);
    }
    return mask;
}

// wrapper 1 (contactReduction.cuh:171-177): adj = 0, initial = separation
inline uint32_t refWrapper1(const V3* pt, const float* sep, V3 n, uint32_t allMask, float bias)
{
    float adj[32], initial[32];
    for (uint32_t j = 0; j < 32; ++j) { adj[j] = 0.0f; initial[j] = sep[j]; }
    return reduceShared(pt, sep, n, allMask, bias, adj, initial, 6, false, true);
}

// wrapper 1 with the DEFAULT TDoDupeTest = true -- npTriHfNp's instantiation.
inline uint32_t refWrapper1Dupe(const V3* pt, const float* sep, V3 n, uint32_t allMask, float bias)
{
    float adj[32], initial[32];
    for (uint32_t j = 0; j < 32; ++j) { adj[j] = 0.0f; initial[j] = sep[j]; }
    return reduceShared(pt, sep, n, allMask, bias, adj, initial, 6, true, true);
}

// wrapper 2 (contactReduction.cuh:180-188): adj = -separation,
// initial = |pointRaw projected onto the normal plane| (:185 point.magnitude())
inline uint32_t refWrapper2(const V3* pt, const float* sep, V3 n, uint32_t allMask, float bias)
{
    float adj[32], initial[32];
    for (uint32_t j = 0; j < 32; ++j)
    {
        const V3 p = sub(pt[j], scale3(n, dot3(pt[j], n)));
        adj[j] = -sep[j];
        initial[j] = mag3(p);
    }
    return reduceShared(pt, sep, n, allMask, bias, adj, initial, 6, false, true);
}

// <clustering=FALSE, keepAnotherDeepestForPCM=true, maxPoints=5, dupeTest=true>
// -- npCvxTriEmit::contactReduce5 and pxgSlangConvexTriCloth::contactReduce5.
inline uint32_t refNoCluster5(const V3* pt, const float* sep, V3 n, uint32_t allMask, float bias)
{
    float adj[32], initial[32];
    for (uint32_t j = 0; j < 32; ++j) { adj[j] = 0.0f; initial[j] = sep[j]; }
    return reduceShared(pt, sep, n, allMask, bias, adj, initial, 5, true, false);
}

inline uint32_t popcount32(uint32_t v) { uint32_t c = 0; while (v) { c += v & 1u; v >>= 1; } return c; }

} // namespace crsref
} // namespace pxgtest

#endif
