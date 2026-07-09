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

// Verification of npConvexFaceSat.slang: ctSatConvexFaceNormals
// (satConvexFaceNormals) — the convex-face-normal half of the convex-vs-triangle
// SAT.  The C++ reference transcribes the per-polygon plane transform, the
// hull/triangle projection and the running-max, matching the Slang serial order.
// The continuous outputs (separation, faceNormal) are compared with almostEqual;
// the discrete outputs (the disjoint return and the winning feature index) are
// gated on float comparisons, so cases within MARG of the per-polygon disjoint
// boundaries, the final separation-update boundary, or a near-tie between the two
// best polygons are skipped (measure-zero GPU-vs-CPU flips / warp-tie-break).

#include "pxgTestFramework.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct V4 { float x, y, z, w; };
struct F4 { float x, y, z, w; };
struct CfIn { F4 convexScale, convexScaleRot, tri0, tri1, tri2, misc, seedFaceNormal; uint32_t seedFeature, p0, p1, p2; };
struct CfOut { F4 sepRet, faceNormal; uint32_t feature, q0, q1, q2; };
struct CfParams { uint32_t numTests, maxPoly, maxVerts, pad; };

static const uint32_t MAX_POLY = 12;
static const uint32_t MAX_VERTS = 24;
static const float MARG = 1e-3f;

static V3 operator-(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static float len(V3 a) { return std::sqrt(dot(a, a)); }
static V3 norm(V3 a) { const float m = len(a); return { a.x / m, a.y / m, a.z / m }; }
static F4 f4(V3 v) { return { v.x, v.y, v.z, 0.0f }; }

// quaternion rotate / rotateInv, byte-identical to pxgSlangCommon's quatRotate*
static V3 quatRotate(V4 q, V3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return { vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
             vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
             vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2 };
}
static V3 quatRotateInv(V4 q, V3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return { vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * dot2,
             vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * dot2,
             vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * dot2 };
}
static V3 vertex2ShapeNormalVector(V3 n, V3 scale, V4 rot)
{
    V3 t = quatRotate(rot, n);
    t = { t.x / scale.x, t.y / scale.y, t.z / scale.z };
    return quatRotateInv(rot, t);
}

static V4 randUnitQuat(std::mt19937& rng)
{
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    V4 q; float m;
    do { q = { u(rng), u(rng), u(rng), u(rng) }; m = std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w); } while (m < 1e-3f);
    return { q.x/m, q.y/m, q.z/m, q.w/m };
}
static V3 randUnit(std::mt19937& rng)
{
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    V3 v; do { v = { u(rng), u(rng), u(rng) }; } while (dot(v, v) < 1e-4f);
    return norm(v);
}

PXG_TEST(npConvexFaceSat_satConvexFaceNormals)
{
    std::mt19937 rng(65011);
    const uint32_t N = 16000;
    std::vector<uint32_t> polyDesc(N * MAX_POLY);
    std::vector<F4> planes(N * MAX_POLY);
    std::vector<F4> verts(N * MAX_VERTS);
    std::vector<CfIn> in(N);

    std::vector<uint8_t> refRet(N), skip(N, 0);
    std::vector<float> refSep(N);
    std::vector<V3> refFN(N);
    std::vector<uint32_t> refFeat(N);

    std::uniform_real_distribution<float> uv(-2.0f, 2.0f);
    std::uniform_real_distribution<float> usc(0.4f, 2.0f);
    std::uniform_real_distribution<float> upd(-2.0f, 2.0f);
    std::uniform_real_distribution<float> ucd(0.5f, 12.0f); // wide, so overlap & disjoint both occur
    std::uniform_int_distribution<uint32_t> unp(3, MAX_POLY);

    size_t disjointC = 0, overlapC = 0, updated = 0, notUpd = 0, skipped = 0;

    for (uint32_t t = 0; t < N; ++t)
    {
        const uint32_t nbPoly = unp(rng);
        for (uint32_t i = 0; i < MAX_VERTS; ++i) verts[t * MAX_VERTS + i] = { uv(rng), uv(rng), uv(rng), 0.0f };
        for (uint32_t i = 0; i < MAX_POLY; ++i)
        {
            const V3 n = randUnit(rng);
            planes[t * MAX_POLY + i] = { n.x, n.y, n.z, upd(rng) };
            polyDesc[t * MAX_POLY + i] = rng() % MAX_VERTS; // minIndex in low byte
        }
        const V3 scale = { usc(rng), usc(rng), usc(rng) };
        const V4 rot = randUnitQuat(rng);
        // triangle near the vertex cloud (so overlaps are common)
        const V3 tv0 = { uv(rng), uv(rng), uv(rng) };
        const V3 tv1 = { uv(rng), uv(rng), uv(rng) };
        const V3 tv2 = { uv(rng), uv(rng), uv(rng) };
        const float contactDist = ucd(rng);
        const float seedSep = (rng() & 1u) ? -1e30f : (upd(rng) * 0.3f);
        const V3 seedFN = randUnit(rng);
        const uint32_t seedFeat = 0xABCDu;
        in[t] = { f4(scale), { rot.x, rot.y, rot.z, rot.w }, f4(tv0), f4(tv1), f4(tv2),
                  { float(nbPoly), contactDist, seedSep, 0 }, f4(seedFN), seedFeat, 0, 0, 0 };

        // ---- reference (mirrors the Slang serial order) ----
        float _sep = -FLT_MAX; uint32_t _feat = 0; V3 _fn = seedFN;
        bool disjoint = false, doSkip = false;
        float bestTop = -FLT_MAX, best2 = -FLT_MAX; // top-two tempSeparation for near-tie skip
        for (uint32_t i = 0; i < nbPoly; ++i)
        {
            const V3 vsN = { planes[t * MAX_POLY + i].x, planes[t * MAX_POLY + i].y, planes[t * MAX_POLY + i].z };
            const float planeDist = planes[t * MAX_POLY + i].w;
            const V3 minVert = { verts[t * MAX_VERTS + (polyDesc[t * MAX_POLY + i] & 0xffu)].x,
                                 verts[t * MAX_VERTS + (polyDesc[t * MAX_POLY + i] & 0xffu)].y,
                                 verts[t * MAX_VERTS + (polyDesc[t * MAX_POLY + i] & 0xffu)].z };
            V3 ssN = vertex2ShapeNormalVector(vsN, scale, rot);
            const float mag = len(ssN);
            const float invMag = 1.0f / mag;
            ssN = { ssN.x * invMag, ssN.y * invMag, ssN.z * invMag };
            const float min0 = dot(ssN, minVert);
            const float max0 = -planeDist * invMag;
            const float d0 = dot(tv0, ssN), d1 = dot(tv1, ssN), d2 = dot(tv2, ssN);
            const float max1 = std::max(d0, std::max(d1, d2));
            const float min1 = std::min(d0, std::min(d1, d2));
            // near disjoint-boundary skip (either comparison)
            if (std::fabs(min1 - (max0 + contactDist)) < MARG || std::fabs(min0 - (max1 + contactDist)) < MARG) doSkip = true;
            if ((min1 > max0 + contactDist) || (min0 > max1 + contactDist)) { disjoint = true; break; }
            const float tempSep = min1 - max0;
            if (_sep < tempSep) { _sep = tempSep; _feat = i; _fn = ssN; }
            if (tempSep > bestTop) { best2 = bestTop; bestTop = tempSep; }
            else if (tempSep > best2) { best2 = tempSep; }
        }
        // near-tie between the two best faces -> feature/faceNormal ambiguous (warp tie-break)
        if (!disjoint && std::fabs(bestTop - best2) < MARG) doSkip = true;

        float sep = seedSep; V3 fn = seedFN; uint32_t feat = seedFeat; uint8_t ret = 1;
        if (disjoint) { ret = 0; ++disjointC; }
        else
        {
            ++overlapC;
            if (std::fabs(sep - _sep) < MARG) doSkip = true; // final update boundary
            if (sep < _sep) { fn = _fn; sep = _sep; feat = _feat; ++updated; } else ++notUpd;
        }
        refRet[t] = ret; refSep[t] = sep; refFN[t] = fn; refFeat[t] = feat;
        if (doSkip) { skip[t] = 1; ++skipped; }
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<CfOut> outInit(N);
    PxgBufferHandle bPD = createBufferFrom(dev, polyDesc);
    PxgBufferHandle bPL = createBufferFrom(dev, planes);
    PxgBufferHandle bV = createBufferFrom(dev, verts);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    CfParams prm = { N, MAX_POLY, MAX_VERTS, 0 };
    Kernel k("npConvexFaceSat.spv", "ctSatConvexFaceNormals",
             { 4, (uint32_t)sizeof(F4), (uint32_t)sizeof(F4), (uint32_t)sizeof(CfIn), (uint32_t)sizeof(CfOut) }, sizeof(CfParams));
    k.dispatch(GROUPS, { bPD, bPL, bV, bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CfOut>(dev, bOut, N);
    size_t mmRet = 0, mmSep = 0, mmFeat = 0, mmFN = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        if (skip[t]) continue;
        const CfOut& v = g[t];
        const uint8_t gret = (v.sepRet.y != 0.0f) ? 1 : 0;
        if (gret != refRet[t]) { ++mmRet; continue; }
        if (refRet[t] == 0) continue; // disjoint: separation/feature/faceNormal are the (unchanged) seeds
        if (!almostEqual(v.sepRet.x, refSep[t], 1e-4f, 1e-4f)) { ++mmSep; continue; }
        if (v.feature != refFeat[t]) { ++mmFeat; continue; }
        if (!almostEqual(v.faceNormal.x, refFN[t].x, 1e-4f, 1e-4f) ||
            !almostEqual(v.faceNormal.y, refFN[t].y, 1e-4f, 1e-4f) ||
            !almostEqual(v.faceNormal.z, refFN[t].z, 1e-4f, 1e-4f)) ++mmFN;
    }
    PXG_CHECK_MSG(mmRet == 0, "satConvexFaceNormals: %zu return mismatches", mmRet);
    PXG_CHECK_MSG(mmSep == 0, "satConvexFaceNormals: %zu separation mismatches", mmSep);
    PXG_CHECK_MSG(mmFeat == 0, "satConvexFaceNormals: %zu feature mismatches", mmFeat);
    PXG_CHECK_MSG(mmFN == 0, "satConvexFaceNormals: %zu faceNormal mismatches", mmFN);
    PXG_CHECK_MSG(disjointC > 500 && overlapC > N / 8 && updated > 100 && notUpd > 100,
                  "satConvexFaceNormals: coverage thin (disjoint=%zu overlap=%zu upd=%zu notUpd=%zu skip=%zu)",
                  disjointC, overlapC, updated, notUpd, skipped);

    dev->destroyBuffer(bPD); dev->destroyBuffer(bPL); dev->destroyBuffer(bV); dev->destroyBuffer(bIn); dev->destroyBuffer(bOut);
}

} // namespace
