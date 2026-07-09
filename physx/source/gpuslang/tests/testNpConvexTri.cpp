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

// Verification of npConvexTri.slang: ctMinkowskiFace (isMinkowskiFace) and
// ctEdgePairPassed (isEdgePairPassed) — the convex-triangle edge-edge SAT
// primitives.  isMinkowskiFace's boolean is gated on three float-product sign
// tests, so cases where any product sits within MARG of zero are skipped.
// isEdgePairPassed's continuous outputs (separation, faceNormal) are compared
// with tolerance and its discrete return is skipped near the Minkowski-face,
// edge-parallel and dist>contactDist boundaries (measure-zero GPU-vs-CPU flips).

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct F4 { float x, y, z, w; };

static V3 operator-(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 operator-(V3 a) { return { -a.x, -a.y, -a.z }; }
static float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 normalize(V3 a) { const float m = std::sqrt(dot(a, a)); return { a.x / m, a.y / m, a.z / m }; }

static const float MARG = 1e-3f;

static bool refMinkowskiFace(V3 A, V3 B, V3 BxA, V3 C, V3 D, V3 DxC,
                             float& p0, float& p1, float& p2)
{
    const float CBA = dot(C, BxA), DBA = dot(D, BxA), ADC = dot(A, DxC), BDC = dot(B, DxC);
    p0 = CBA * DBA; p1 = ADC * BDC; p2 = CBA * BDC;
    return CBA * DBA < 0.0f && ADC * BDC < 0.0f && CBA * BDC > 0.0f;
}

struct MkIn { F4 A, B, BxA, C, D, DxC; };
struct EpIn { F4 cN0, cN1, cEdge, cVert, tN0, tEdge, tVert, cCoM, misc, seedFN; };
struct EpOut { F4 faceNormal, sr; };
struct CtParams { uint32_t numTests, pad0, pad1, pad2; };

static V3 randUnit(std::mt19937& rng)
{
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    V3 v;
    do { v = { u(rng), u(rng), u(rng) }; } while (dot(v, v) < 1e-4f);
    return normalize(v);
}
static F4 f4(V3 v) { return { v.x, v.y, v.z, 0.0f }; }

PXG_TEST(npConvexTri_minkowskiFace)
{
    std::mt19937 rng(63011);
    const uint32_t N = 12000;
    std::vector<MkIn> in(N);
    std::vector<uint8_t> ref(N), skip(N, 0);
    size_t trueC = 0, falseC = 0, skipped = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        // The predicate is pure arithmetic on its six vector inputs: A,B and C,D
        // are adjacent face normals, and BxA/DxC are the two edge vectors (arcs).
        // Independent random unit vectors exercise the arithmetic across both
        // outcomes with no degeneracy.
        const V3 A = randUnit(rng), B = randUnit(rng), BxA = randUnit(rng);
        const V3 C = randUnit(rng), D = randUnit(rng), DxC = randUnit(rng);
        in[t] = { f4(A), f4(B), f4(BxA), f4(C), f4(D), f4(DxC) };
        float p0, p1, p2;
        ref[t] = refMinkowskiFace(A, B, BxA, C, D, DxC, p0, p1, p2) ? 1 : 0;
        if (std::fabs(p0) < MARG || std::fabs(p1) < MARG || std::fabs(p2) < MARG) { skip[t] = 1; ++skipped; continue; }
        if (ref[t]) ++trueC; else ++falseC;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<uint32_t> outInit(N, 7);
    std::vector<EpIn> epDummy(N);
    std::vector<EpOut> epoDummy(N);
    PxgBufferHandle bMk = createBufferFrom(dev, in);
    PxgBufferHandle bMkOut = createBufferFrom(dev, outInit);
    PxgBufferHandle bEp = createBufferFrom(dev, epDummy);
    PxgBufferHandle bEpOut = createBufferFrom(dev, epoDummy);

    CtParams prm = { N, 0, 0, 0 };
    Kernel k("npConvexTri.spv", "ctMinkowskiFace",
             { (uint32_t)sizeof(MkIn), 4, (uint32_t)sizeof(EpIn), (uint32_t)sizeof(EpOut) }, sizeof(CtParams));
    k.dispatch(GROUPS, { bMk, bMkOut, bEp, bEpOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<uint32_t>(dev, bMkOut, N);
    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t) { if (skip[t]) continue; if ((g[t] ? 1 : 0) != ref[t]) ++mm; }
    PXG_CHECK_MSG(mm == 0, "ctMinkowskiFace: %zu mismatches", mm);
    PXG_CHECK_MSG(trueC > 200 && falseC > N / 4, "ctMinkowskiFace: coverage thin (true=%zu false=%zu skip=%zu)", trueC, falseC, skipped);

    dev->destroyBuffer(bMk); dev->destroyBuffer(bMkOut); dev->destroyBuffer(bEp); dev->destroyBuffer(bEpOut);
}

PXG_TEST(npConvexTri_edgePairPassed)
{
    std::mt19937 rng(63029);
    const uint32_t N = 16000;
    std::vector<EpIn> in(N);
    std::vector<uint8_t> refRet(N), skip(N, 0);
    std::vector<float> refSep(N);
    std::vector<V3> refFN(N);
    std::uniform_real_distribution<float> uc(0.01f, 0.3f);   // contactDist
    std::uniform_real_distribution<float> uv(-3.0f, 3.0f);   // vertex/com coords
    size_t retTrue = 0, retFalse = 0, updated = 0, cullC = 0, skipped = 0;

    for (uint32_t t = 0; t < N; ++t)
    {
        const V3 cN0 = randUnit(rng), cN1 = randUnit(rng);
        const V3 cEdge = cross(cN1, cN0);       // convex edge = cross of adjacent normals
        const V3 tN0 = randUnit(rng);
        const V3 tEdge = randUnit(rng);
        const V3 cVert = { uv(rng), uv(rng), uv(rng) };
        const V3 tVert = { uv(rng), uv(rng), uv(rng) };
        const V3 cCoM = { uv(rng), uv(rng), uv(rng) };
        const float contactDist = uc(rng);
        const float seedSep = (rng() & 1u) ? -1e30f : (uv(rng) * 0.5f); // sometimes low, sometimes mid
        const V3 seedFN = randUnit(rng);
        in[t] = { f4(cN0), f4(cN1), f4(cEdge), f4(cVert), f4(tN0), f4(tEdge), f4(tVert), f4(cCoM),
                  { contactDist, seedSep, 0, 0 }, f4(seedFN) };

        // C++ reference mirroring isEdgePairPassed
        float sep = seedSep; V3 fn = seedFN; uint8_t ret; bool doSkip = false;
        {
            // isMinkowskiFace(cN0, cN1, cEdge, -tN0, tN0, tEdge)
            const V3 A = cN0, B = cN1, BxA = cEdge, C = -tN0, D = tN0, DxC = tEdge;
            const float CBA = dot(C, BxA), DBA = dot(D, BxA), ADC = dot(A, DxC), BDC = dot(B, DxC);
            const bool mk = CBA * DBA < 0.0f && ADC * BDC < 0.0f && CBA * BDC > 0.0f;
            if (std::fabs(CBA * DBA) < MARG || std::fabs(ADC * BDC) < MARG || std::fabs(CBA * BDC) < MARG) doSkip = true;
            if (!mk) { ret = 1; ++cullC; }
            else
            {
                V3 normal = cross(cEdge, tEdge);
                const float eps = 1e-6f;
                if (std::fabs(normal.x) < eps || std::fabs(normal.y) < eps || std::fabs(normal.z) < eps)
                {
                    ret = 1; // parallel
                    // near the eps boundary the branch could flip
                    if (std::fabs(std::fabs(normal.x) - eps) < 1e-7f || std::fabs(std::fabs(normal.y) - eps) < 1e-7f ||
                        std::fabs(std::fabs(normal.z) - eps) < 1e-7f) doSkip = true;
                }
                else
                {
                    normal = normalize(normal);
                    const float od = dot(normal, cVert - cCoM);
                    if (std::fabs(od) < MARG) doSkip = true; // orientation-flip boundary (sign cascades into dist)
                    if (od < 0.0f) normal = -normal;
                    const float dist = dot(normal, tVert - cVert);
                    if (std::fabs(dist - contactDist) < MARG) doSkip = true; // dist>contactDist boundary
                    // separation<dist update boundary: a rounding flip there would leave
                    // faceNormal at seed vs normal, so skip when seed sits within MARG of dist
                    if (std::fabs(sep - dist) < MARG) doSkip = true;
                    if (dist > contactDist) ret = 0;
                    else { if (sep < dist) { sep = dist; fn = normal; } ret = 1; }
                }
            }
        }
        refRet[t] = ret; refSep[t] = sep; refFN[t] = fn;
        if (doSkip) { skip[t] = 1; ++skipped; continue; }
        if (ret) ++retTrue; else ++retFalse;
        if (ret && (refSep[t] != seedSep)) ++updated;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<MkIn> mkDummy(N);
    std::vector<uint32_t> mkoDummy(N);
    std::vector<EpOut> outInit(N);
    PxgBufferHandle bMk = createBufferFrom(dev, mkDummy);
    PxgBufferHandle bMkOut = createBufferFrom(dev, mkoDummy);
    PxgBufferHandle bEp = createBufferFrom(dev, in);
    PxgBufferHandle bEpOut = createBufferFrom(dev, outInit);

    CtParams prm = { N, 0, 0, 0 };
    Kernel k("npConvexTri.spv", "ctEdgePairPassed",
             { (uint32_t)sizeof(MkIn), 4, (uint32_t)sizeof(EpIn), (uint32_t)sizeof(EpOut) }, sizeof(CtParams));
    k.dispatch(GROUPS, { bMk, bMkOut, bEp, bEpOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<EpOut>(dev, bEpOut, N);
    size_t mmRet = 0, mmSep = 0, mmFN = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        if (skip[t]) continue;
        const EpOut& v = g[t];
        const uint8_t gret = (v.sr.y != 0.0f) ? 1 : 0;
        if (gret != refRet[t]) { ++mmRet; continue; }
        if (!almostEqual(v.sr.x, refSep[t], 1e-4f, 1e-4f)) { ++mmSep; continue; }
        // faceNormal only meaningful (updated) when ret==1 and sep advanced; compare regardless,
        // it should equal the reference (seed or updated) to tolerance
        if (!almostEqual(v.faceNormal.x, refFN[t].x, 1e-4f, 1e-4f) ||
            !almostEqual(v.faceNormal.y, refFN[t].y, 1e-4f, 1e-4f) ||
            !almostEqual(v.faceNormal.z, refFN[t].z, 1e-4f, 1e-4f)) ++mmFN;
    }
    PXG_CHECK_MSG(mmRet == 0, "ctEdgePairPassed: %zu return mismatches", mmRet);
    PXG_CHECK_MSG(mmSep == 0, "ctEdgePairPassed: %zu separation mismatches", mmSep);
    PXG_CHECK_MSG(mmFN == 0, "ctEdgePairPassed: %zu faceNormal mismatches", mmFN);
    PXG_CHECK_MSG(retTrue > N / 8 && retFalse > 100 && updated > 100 && cullC > 100,
                  "ctEdgePairPassed: coverage thin (true=%zu false=%zu upd=%zu cull=%zu skip=%zu)",
                  retTrue, retFalse, updated, cullC, skipped);

    dev->destroyBuffer(bMk); dev->destroyBuffer(bMkOut); dev->destroyBuffer(bEp); dev->destroyBuffer(bEpOut);
}

} // namespace
