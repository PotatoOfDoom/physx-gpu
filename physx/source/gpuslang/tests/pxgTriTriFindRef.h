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

// pxgTriTriFindRef.h -- the host mirror of the mesh-mesh kernel's GATHER
// (pxgSlangTriTriFind): findInterestingTrianglesA<32, 1024> with addIndexToBuffer
// inlined, the cull's texture-free half, and addToRefinementBuffer<32, 256>.
//
// P9.83b-8 (4/n) moved these here from testNpTriTriFind.cpp, where the gather
// fixture had them, because the kernel fixture replays the same gather.  Two
// changes, neither touching a value or a branch: the SDF sample is a CALLABLE,
// float(ind, sub, centroidVtx), where the gather fixture used a hash or an
// analytic plane (the gather fixture passes those; the kernel fixture passes
// distances measured by the certified device entry npSdfCollision:sdfCollDistance),
// and FindStats counts the samples taken.

#ifndef PXG_TRI_TRI_FIND_REF_H
#define PXG_TRI_TRI_FIND_REF_H

#include "pxgTriTriGeomRef.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace pxgtest
{
namespace tritrifindref
{
using namespace pxgtest::tritrigeomref;

struct U2 { uint32_t x, y; };

static const uint32_t TT_TARGET = 1024u;    // findInterestingTrianglesA<32, 1024>
static const uint32_t TT_BLOCK = 1024u;     // the host launch
static const uint32_t TT_SUBCAP = 256u;     // addToRefinementBuffer<32, 256>

// What one gather call sees: the mesh whose triangles are culled, the two scale
// frames, the pose that carries mesh0 into mesh1's frame, and the tolerance.
struct GatherGeom
{
    const std::vector<U4>* inds;
    const std::vector<F4>* verts;
    PxVec3 scale0;
    PxQuat rot0;
    PxVec3 scale1;
    PxQuat rot1;
    PxTransform aToB;
    float tolerance;
};

// sdfCollision.cuh:661-670, the texture-free half.
inline float refCullSphere(const GatherGeom& g, uint32_t ind, uint32_t sub, PxVec3& centroidVtx)
{
    PxVec3 v0, v1, v2;
    refGetTriangleVertices(*g.inds, *g.verts, g.scale0, g.rot0, g.aToB, ind, sub, v0, v1, v2);

    const physx::PxReal third = 1.f / 3.f;
    PxVec3 centroid = (v0 + v1 + v2) * third;

    centroidVtx = refShape2Vertex(centroid, g.scale1, g.rot1);
    physx::PxReal boundingSphereRadius = physx::PxSqrt(physx::PxMax((v0 - centroid).magnitudeSquared(),
        physx::PxMax((v1 - centroid).magnitudeSquared(), (v2 - centroid).magnitudeSquared())));
    boundingSphereRadius /= physx::PxMin(g.scale1.x, physx::PxMin(g.scale1.y, g.scale1.z));
    return boundingSphereRadius;
}

struct FindStats
{
    size_t childLastSlot = 0, mixedPasses = 0, refineOverflow = 0, levelCapDrops = 0, childEntries = 0,
           boundaryHits = 0, roundedBoundaryHits = 0, samples = 0;
};

// findInterestingTrianglesA<32, 1024> with addIndexToBuffer inlined, one block.
// `dist` is float(uint32_t ind, uint32_t sub, const PxVec3& centroidVtx).
template <typename Dist>
inline uint32_t refFind(const GatherGeom& g, uint32_t nbPrimitives, uint32_t& startIndex,
                        uint32_t nbSubdivisionIndices, const U2* subBuf, U2* idxBuf, Dist&& dist, FindStats& st)
{
    uint32_t offset = startIndex;   // :626-627
    const float tolerance = g.tolerance;
    uint32_t sharedIndicesCount = 0;
    for (uint32_t i = startIndex; (i < nbPrimitives || nbSubdivisionIndices > 0) && sharedIndicesCount < TT_TARGET; )
    {
        std::vector<uint32_t> ind(TT_BLOCK), sub(TT_BLOCK);
        std::vector<uint8_t> add(TT_BLOCK, 0);
        const uint32_t childThreads = std::min(TT_TARGET, 4u * nbSubdivisionIndices);
        if (nbSubdivisionIndices > 0 && childThreads < TT_BLOCK) ++st.mixedPasses;
        for (uint32_t t = 0; t < TT_BLOCK; ++t)
        {
            if (t < 4u * nbSubdivisionIndices)
            {
                const U2 v = subBuf[t >> 2];
                ind[t] = v.x;
                sub[t] = physx::Gu::elevateSubdivisionId(v.y, t & 3u);
            }
            else
            {
                ind[t] = i + t - childThreads;
                sub[t] = 0;
            }
        }
        i += TT_BLOCK - childThreads;
        nbSubdivisionIndices = 0;

        for (uint32_t t = 0; t < TT_BLOCK; ++t)
        {
            if (ind[t] < nbPrimitives)
            {
                PxVec3 cv;
                const float r = refCullSphere(g, ind[t], sub[t], cv);
                const float d = dist(ind[t], sub[t], cv);
                ++st.samples;
                add[t] = d < (r + tolerance);
                if (d == (r + tolerance)) ++st.boundaryHits;   // `<` and `<=` disagree here
                if (d == (r + tolerance) && double(r) + double(tolerance) > double(d)) ++st.roundedBoundaryHits;
            }
        }

        // addIndexToBuffer: the block scan in thread order.
        uint32_t excl = 0;
        for (uint32_t t = 0; t < TT_BLOCK; ++t)
        {
            if (!add[t]) continue;
            const uint32_t w = sharedIndicesCount + excl;
            if (w < TT_TARGET) { idxBuf[w] = U2{ ind[t], sub[t] }; if (sub[t] != 0) ++st.childEntries; }
            if (w == TT_TARGET - 1u)
            {
                offset = ind[t] + 1u;
                if (sub[t] != 0 && t < childThreads) ++st.childLastSlot;
            }
            ++excl;
        }
        sharedIndicesCount = std::min(excl + sharedIndicesCount, TT_TARGET);
    }

    if (startIndex != offset) startIndex = offset;   // :681-684
    else startIndex = nbPrimitives;
    return sharedIndicesCount;
}

// addToRefinementBuffer<32, 256> (:741-763), with nbElementsInBuffer = 0 as every
// call site passes it.
inline uint32_t refAddToRefinement(const std::vector<uint8_t>& want, const std::vector<uint32_t>& ind,
                                   const std::vector<uint32_t>& sub, uint32_t maxLevel, U2* subBuf, FindStats& st)
{
    uint32_t excl = 0;
    for (uint32_t t = 0; t < TT_BLOCK; ++t)
    {
        uint32_t level, idx;
        physx::Gu::decodeSubdivisionId(sub[t], level, idx);
        const bool add = want[t] && level < maxLevel;
        if (want[t] && !add) ++st.levelCapDrops;
        if (!add) continue;
        if (excl < TT_SUBCAP) subBuf[excl] = U2{ ind[t], sub[t] };
        ++excl;
    }
    if (excl > TT_SUBCAP) ++st.refineOverflow;
    return std::min(excl, TT_SUBCAP);
}

} // namespace tritrifindref
} // namespace pxgtest

#endif // PXG_TRI_TRI_FIND_REF_H
