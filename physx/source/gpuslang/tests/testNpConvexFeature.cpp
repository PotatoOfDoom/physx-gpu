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

// Verification of npConvexFeature.slang: ctSelectFeature (delayGenerateContact +
// selectPolygonFeatureIndex) — the convex-triangle feature-selection decision
// layer.  Uses axis-aligned box hulls (6 faces / 12 edges, axis-aligned plane
// normals) so the face/edge argmaxes are over exact dot products; the three
// integer outputs (polyFaceIndex, polyIndex2, delay) are discrete and compared
// BIT-EXACTLY against an exact CPU transcription that mirrors the shader's
// 32-lane bucketed argmax (lowest-lane tie-break, tranche 108).  Diagonal
// minNormals (edge bisectors, irrational) drive the closest-edge branch — their
// argmax decisions are unambiguous (or exact ties resolved identically), so the
// discrete outputs still match exactly.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct F4 { float x, y, z, w; };
struct U2 { uint32_t x, y; };
struct U4 { uint32_t x, y, z, w; };
struct FiIn { F4 minNormal, triLocNormal; U4 misc; };
struct FiOut { uint32_t polyFaceIndex, polyIndex2, delay, pad0; };
struct FiParams { uint32_t numTests, maxPolys, maxEdges, pad0; };

static const uint32_t EDGE_FEATURE_IDX = 0x80000000u;
static const uint32_t TRI_FEATURE_IDX = 0x7FffFFffu;
static const uint32_t INVALID = 0xffffffffu;
static const uint32_t MAX_POLYS = 6, MAX_EDGES = 12;
static const float FI_MAX = 3.402823466e38f;
static const float COS45 = 0.707106781f;

static float dotX(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 normX(V3 v) { const float s = 1.0f / std::sqrt(dotX(v, v)); return { v.x * s, v.y * s, v.z * s }; }
static F4 f4(V3 v) { return { v.x, v.y, v.z, 0.0f }; }

// axis-aligned box: faces 0=+x 1=-x 2=+y 3=-y 4=+z 5=-z; 12 edges, each with its
// two bordering face indices (matching tranche 108's makeBox topology).
struct Box { F4 planeN[6]; U2 edgeF[12]; };
static Box makeBox()
{
    Box b;
    b.planeN[0] = { 1, 0, 0, 0 }; b.planeN[1] = { -1, 0, 0, 0 };
    b.planeN[2] = { 0, 1, 0, 0 }; b.planeN[3] = { 0, -1, 0, 0 };
    b.planeN[4] = { 0, 0, 1, 0 }; b.planeN[5] = { 0, 0, -1, 0 };
    uint32_t e = 0;
    for (uint32_t k = 0; k < 2; ++k) for (uint32_t j = 0; j < 2; ++j) { b.edgeF[e++] = { j ? 2u : 3u, k ? 4u : 5u }; } // x-edges
    for (uint32_t k = 0; k < 2; ++k) for (uint32_t i = 0; i < 2; ++i) { b.edgeF[e++] = { i ? 0u : 1u, k ? 4u : 5u }; } // y-edges
    for (uint32_t j = 0; j < 2; ++j) for (uint32_t i = 0; i < 2; ++i) { b.edgeF[e++] = { i ? 0u : 1u, j ? 2u : 3u }; } // z-edges
    return b;
}

struct FiRef { uint32_t polyFaceIndex, polyIndex2, delay; };
static FiRef featureRef(const Box& box, uint32_t nbPolys, uint32_t nbEdges,
                        V3 minNormal, V3 triLocNormal, uint32_t featureIndex)
{
    FiRef r;
    r.delay = 0;
    if (featureIndex != EDGE_FEATURE_IDX && featureIndex != TRI_FEATURE_IDX)
    {
        const V3 tln = { -triLocNormal.x, -triLocNormal.y, -triLocNormal.z };
        r.delay = (dotX(minNormal, tln) < COS45) ? 1u : 0u;
    }
    uint32_t polyFaceIndex = featureIndex, polyIndex2 = INVALID;
    if (featureIndex == TRI_FEATURE_IDX || featureIndex == EDGE_FEATURE_IDX)
    {
        auto pn = [&](uint32_t i) { return V3{ box.planeN[i].x, box.planeN[i].y, box.planeN[i].z }; };
        float laneProj[32]; uint32_t laneFace[32];
        for (uint32_t L = 0; L < 32; ++L) { laneProj[L] = -FI_MAX; laneFace[L] = 0; }
        for (uint32_t i = 0; i < nbPolys; ++i)
        { const float proj = dotX(minNormal, pn(i)); const uint32_t L = i % 32u; if (laneProj[L] < proj) { laneProj[L] = proj; laneFace[L] = i; } }
        float minProj = laneProj[0]; polyFaceIndex = laneFace[0];
        for (uint32_t L = 1; L < 32; ++L) if (laneProj[L] > minProj) { minProj = laneProj[L]; polyFaceIndex = laneFace[L]; }

        float laneMaxDpSq[32], laneMaxDpSq2[32]; uint32_t laneClosest[32], lanePoly2[32];
        for (uint32_t L = 0; L < 32; ++L) { laneMaxDpSq[L] = minProj * minProj; laneClosest[L] = INVALID; laneMaxDpSq2[L] = -10.0f; lanePoly2[L] = INVALID; }
        for (uint32_t i = 0; i < nbEdges; ++i)
        {
            const uint32_t L = i % 32u;
            const U2 f = box.edgeF[i];
            const V3 edgeN = add(pn(f.x), pn(f.y));
            const float enMagSq = dotX(edgeN, edgeN);
            const float dp = dotX(edgeN, minNormal);
            const float sqDp = dp * dp;
            if ((f.x == polyFaceIndex || f.y == polyFaceIndex) && sqDp > laneMaxDpSq2[L])
            { laneMaxDpSq2[L] = sqDp; lanePoly2[L] = (f.x == polyFaceIndex) ? f.y : f.x; }
            if (dp > 0.0f && sqDp > laneMaxDpSq[L] * enMagSq)
            { laneMaxDpSq[L] = sqDp / enMagSq; laneClosest[L] = i; }
        }
        float maxDpSq = laneMaxDpSq[0]; uint32_t closestEdge = laneClosest[0];
        for (uint32_t L = 1; L < 32; ++L) if (laneMaxDpSq[L] > maxDpSq) { maxDpSq = laneMaxDpSq[L]; closestEdge = laneClosest[L]; }
        if (closestEdge != INVALID)
        {
            const U2 f = box.edgeF[closestEdge];
            const float dp0 = dotX(pn(f.x), minNormal), dp1 = dotX(pn(f.y), minNormal);
            if (dp0 > dp1) { polyFaceIndex = f.x; polyIndex2 = f.y; }
            else           { polyFaceIndex = f.y; polyIndex2 = f.x; }
        }
        else
        {
            float maxDpSq2 = laneMaxDpSq2[0]; polyIndex2 = lanePoly2[0];
            for (uint32_t L = 1; L < 32; ++L) if (laneMaxDpSq2[L] > maxDpSq2) { maxDpSq2 = laneMaxDpSq2[L]; polyIndex2 = lanePoly2[L]; }
        }
    }
    r.polyFaceIndex = polyFaceIndex; r.polyIndex2 = polyIndex2;
    return r;
}

PXG_TEST(npConvexFeature_selectFeature)
{
    std::mt19937 rng(1090901);
    const Box box = makeBox();
    const V3 axes[6] = { {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1} };

    std::vector<F4> planeN;
    std::vector<U2> edgeF;
    std::vector<FiIn> in;
    std::vector<FiRef> ref;

    auto emit = [&](V3 minNormal, V3 triN, uint32_t feat)
    {
        for (uint32_t i = 0; i < MAX_POLYS; ++i) planeN.push_back(box.planeN[i]);
        for (uint32_t i = 0; i < MAX_EDGES; ++i) edgeF.push_back(box.edgeF[i]);
        FiIn e;
        e.minNormal = f4(minNormal); e.triLocNormal = f4(triN);
        e.misc = { feat, MAX_POLYS, MAX_EDGES, 0 };
        in.push_back(e);
        ref.push_back(featureRef(box, MAX_POLYS, MAX_EDGES, minNormal, triN, feat));
    };

    // family A: FACE feature (featureIndex in 0..5) -> delay gate + passthrough.
    // minNormal / triNormal axis-aligned so cosTheta is exact (0, +-1), well away
    // from the 0.707 threshold.
    for (uint32_t iter = 0; iter < 400; ++iter)
    {
        const uint32_t feat = rng() % 6;
        const V3 mn = axes[rng() % 6];
        const V3 tn = axes[rng() % 6];
        emit(mn, tn, feat);
    }
    // family B: EDGE/TRI feature with an AXIS-ALIGNED minNormal -> unique best
    // face, no edge beats it, exercises the best-NEIGHBOUR (else) branch.
    for (uint32_t iter = 0; iter < 400; ++iter)
    {
        const uint32_t feat = (rng() & 1) ? EDGE_FEATURE_IDX : TRI_FEATURE_IDX;
        const V3 mn = axes[rng() % 6];
        const V3 tn = axes[rng() % 6];
        emit(mn, tn, feat);
    }
    // family C: EDGE/TRI feature with a DIAGONAL minNormal (an edge bisector,
    // irrational) -> the closest-EDGE branch fires and both edge faces are
    // returned.  Bisector of two perpendicular box faces = their summed normal.
    const uint32_t perpPairs[12][2] = {
        {0,2},{0,3},{0,4},{0,5},{1,2},{1,3},{1,4},{1,5},{2,4},{2,5},{3,4},{3,5}
    };
    for (uint32_t iter = 0; iter < 400; ++iter)
    {
        const uint32_t feat = (rng() & 1) ? EDGE_FEATURE_IDX : TRI_FEATURE_IDX;
        const uint32_t* pr = perpPairs[rng() % 12];
        const V3 bis = normX(add(axes[pr[0]], axes[pr[1]]));
        const V3 tn = axes[rng() % 6];
        emit(bis, tn, feat);
    }

    // family D: EDGE/TRI feature with an ASYMMETRIC diagonal minNormal (weighted
    // 2:1 toward one face of an edge, e.g. (2,1,0)/sqrt5) so the closest-EDGE
    // branch fires AND its two faces have DISTINCT projections (dp0 != dp1),
    // pinning the dp0>dp1 face-ordering.  A signed-permutation of (2,1,0) keeps
    // it aligned to a box edge while varying which edge/orientation wins.
    for (uint32_t iter = 0; iter < 400; ++iter)
    {
        const uint32_t feat = (rng() & 1) ? EDGE_FEATURE_IDX : TRI_FEATURE_IDX;
        // base (2,1,0): 2*axis a + 1*axis b, a != b and not opposite
        float w[3] = { 0, 0, 0 };
        const uint32_t a = rng() % 3;
        uint32_t b = rng() % 3; while (b == a) b = rng() % 3;
        w[a] = ((rng() & 1) ? 2.0f : -2.0f);
        w[b] = ((rng() & 1) ? 1.0f : -1.0f);
        const V3 mn = normX(V3{ w[0], w[1], w[2] });
        const V3 tn = axes[rng() % 6];
        emit(mn, tn, feat);
    }

    // family E: EDGE/TRI feature with a CORNER minNormal (+-1,+-1,+-1)/sqrt3, so
    // three edges tie at the max closest-edge metric — pinning the lowest-lane
    // tie-break of the closest-edge reduction (a `>=` would pick a different
    // edge and hence different faces).
    for (uint32_t iter = 0; iter < 300; ++iter)
    {
        const uint32_t feat = (rng() & 1) ? EDGE_FEATURE_IDX : TRI_FEATURE_IDX;
        const V3 corner = normX(V3{ (rng() & 1) ? 1.0f : -1.0f,
                                    (rng() & 1) ? 1.0f : -1.0f,
                                    (rng() & 1) ? 1.0f : -1.0f });
        const V3 tn = axes[rng() % 6];
        emit(corner, tn, feat);
    }

    // KNOWN-EQUIVALENT mutants under the exact box-hull regime (each would need a
    // non-box hull to observe, which forfeits the exact integer-output match):
    //  * the FACE-argmax tie-break: a minNormal that ties two faces is exactly an
    //    edge bisector, which always makes that edge beat the face, so the
    //    closest-edge branch overwrites polyFaceIndex and the face tie-break is
    //    never observed;
    //  * the closest-edge metric normalisation (dp^2 / enMagSq vs * enMagSq):
    //    every box edge joins two perpendicular unit faces, so enMagSq == 2 is
    //    CONSTANT and the ordering is unchanged;
    //  * the closest-edge threshold init (minProj^2 vs a large negative): for a
    //    box the closest-edge and best-neighbour branches converge to the same
    //    two faces, so forcing the closest-edge branch is a no-op;
    //  * the face-loop lane bucketing (index % 32 vs a single accumulator): a box
    //    has 6 faces / 12 edges < 32, so lane == index and the 32-lane structure
    //    is a no-op (it only diverges for hulls with > 32 faces/edges).
    const uint32_t N = (uint32_t)in.size();
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<FiOut> outInit(N);

    PxgBufferHandle bP = createBufferFrom(dev, planeN);
    PxgBufferHandle bE = createBufferFrom(dev, edgeF);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    FiParams prm = { N, MAX_POLYS, MAX_EDGES, 0 };
    Kernel k("npConvexFeature.spv", "ctSelectFeature",
             { (uint32_t)sizeof(F4), (uint32_t)sizeof(U2), (uint32_t)sizeof(FiIn), (uint32_t)sizeof(FiOut) },
             sizeof(FiParams));
    k.dispatch((N + 63) / 64, { bP, bE, bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<FiOut>(dev, bOut, N);

    size_t mm = 0, nDelay = 0, nFace = 0, nNeighbour = 0, nClosestEdge = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const FiRef& r = ref[t];
        const FiOut& o = g[t];
        if (o.polyFaceIndex != r.polyFaceIndex || o.polyIndex2 != r.polyIndex2 || o.delay != r.delay) ++mm;
        if (r.delay) ++nDelay;
        const uint32_t feat = in[t].misc.x;
        if (feat != EDGE_FEATURE_IDX && feat != TRI_FEATURE_IDX) ++nFace;
        else if (r.polyIndex2 != INVALID && t >= 800) ++nClosestEdge; // family C region
        else if (r.polyIndex2 != INVALID) ++nNeighbour;
    }
    PXG_CHECK_MSG(mm == 0, "ctSelectFeature: %zu mismatches", mm);
    PXG_CHECK_MSG(nDelay > 40 && nFace > 100 && nNeighbour > 100 && nClosestEdge > 100,
                  "ctSelectFeature coverage thin (delay=%zu face=%zu neighbour=%zu closestEdge=%zu)",
                  nDelay, nFace, nNeighbour, nClosestEdge);

    for (PxgBufferHandle h : { bP, bE, bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
