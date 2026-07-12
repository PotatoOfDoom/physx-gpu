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

// Verification of npGjkSupportStep.slang: ctGjkSupportStep — the cold-path front
// half of one gjk() iteration (gjk.cuh :272,:296-310).  One thread per test
// simulates the 32-lane warp.  The support-vertex indices, the updated simplex,
// the support vertex, the lower bound, the stored direction and the convergence
// flag are compared bit-exactly against a CPU transcription.  Integer convex
// clouds and integer closest points keep magnitude (sqrt), the division
// dV=cpv/cm, the support dot products / argmax and the lb comparison
// bit-deterministic.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct SsParams { uint32_t numTests, maxV, pad0, pad1; };
struct SsScalars { float lowerBound, upperBound, minSep, convergenceRatio; uint32_t nbA, nbB, size, pad; };
struct SsResult { uint32_t indexA, indexB, converged, directionWritten; float lowerBoundOut, dir0, dir1, dir2; };

static const float MAXF = 3.402823466e38f;
static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float asF32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static float dotX(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

static void dotW(const float a[32], const float b[32], float o[32])
{
    float x[32]; for (uint32_t L = 0; L < 32; ++L) x[L] = a[L] * b[L];
    for (uint32_t L = 0; L < 32; ++L)
    {
        const float s1 = (L + 1 < 32) ? x[L + 1] : x[L];
        const float s2 = (L + 2 < 32) ? x[L + 2] : x[L];
        o[L] = x[L] + s1 + s2;
    }
}
static uint32_t supportIdx(V3 dir, const std::vector<float>& V, uint32_t vbase, uint32_t nb)
{
    uint32_t bestN = 0xFFFFFFFFu;
    float bestDist = -MAXF;
    for (uint32_t i = 0; i < nb; i += 32)
    {
        float d[32];
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            const uint32_t n = i + lane;
            V3 v = { V[(vbase + n) * 4], V[(vbase + n) * 4 + 1], V[(vbase + n) * 4 + 2] };
            d[lane] = (n < nb) ? dotX(dir, v) : -MAXF;
        }
        float chunkMax = -MAXF;
        for (uint32_t lane = 0; lane < 32; ++lane) chunkMax = std::fmax(chunkMax, d[lane]);
        if (chunkMax > bestDist)
        {
            uint32_t lowestLane = 0;
            for (uint32_t lane = 0; lane < 32; ++lane) if (d[lane] == chunkMax) { lowestLane = lane; break; }
            bestDist = chunkMax; bestN = i + lowestLane;
        }
    }
    return bestN;
}
static uint32_t encodeIndices(uint32_t a, uint32_t b) { return 0x3f800000u | ((b & 0xffu) << 8) | (a & 0xffu); }

struct Ref { float z[32], sup[32]; SsResult r; };
static Ref stepRef(const float cpvIn[32], const float zin[32],
                   const std::vector<float>& VA, const std::vector<float>& VB, uint32_t vbase, const SsScalars& s)
{
    Ref out;
    float cpv[32], zV[32]; for (uint32_t L = 0; L < 32; ++L) { cpv[L] = cpvIn[L]; zV[L] = zin[L]; }

    float dcp[32]; dotW(cpv, cpv, dcp);
    float cm[32]; for (uint32_t L = 0; L < 32; ++L) cm[L] = std::sqrt(dcp[L & 28u]);
    float dV[32]; for (uint32_t L = 0; L < 32; ++L) dV[L] = cpv[L] / cm[L];

    const V3 dvVec = { dV[0], dV[1], dV[2] };
    const uint32_t indexA = supportIdx({ -dvVec.x, -dvVec.y, -dvVec.z }, VA, vbase, s.nbA);
    const uint32_t indexB = supportIdx(dvVec, VB, vbase, s.nbB);

    const V3 vA = { VA[(vbase + indexA) * 4], VA[(vbase + indexA) * 4 + 1], VA[(vbase + indexA) * 4 + 2] };
    const V3 vB = { VB[(vbase + indexB) * 4], VB[(vbase + indexB) * 4 + 1], VB[(vbase + indexB) * 4 + 2] };
    const V3 sv = { vA.x - vB.x, vA.y - vB.y, vA.z - vB.z };
    float supportVertexV[32];
    for (uint32_t L = 0; L < 32; ++L)
    {
        const uint32_t c = L & 3u;
        supportVertexV[L] = (c == 0u) ? sv.x : (c == 1u) ? sv.y : (c == 2u) ? sv.z : 0.0f;
    }

    const float encoded = asF32(encodeIndices(indexA, indexB));
    for (uint32_t L = 0; L < 32; ++L)
        if ((L >> 2) == s.size) zV[L] = ((L & 3u) < 3u) ? supportVertexV[L] : encoded;

    float dds[32]; dotW(dV, supportVertexV, dds);
    const float lb = dds[0];

    float lowerBoundOut = s.lowerBound;
    uint32_t converged = 0, directionWritten = 0;
    float dir0 = 0, dir1 = 0, dir2 = 0;
    if (lb > s.lowerBound)
    {
        dir0 = -dV[0]; dir1 = -dV[1]; dir2 = -dV[2];
        directionWritten = 1u;
        lowerBoundOut = lb;
        if (lowerBoundOut > s.minSep || lowerBoundOut >= s.upperBound * s.convergenceRatio)
            converged = 1u;
    }

    for (uint32_t L = 0; L < 32; ++L) { out.z[L] = zV[L]; out.sup[L] = supportVertexV[L]; }
    out.r.indexA = indexA; out.r.indexB = indexB; out.r.converged = converged; out.r.directionWritten = directionWritten;
    out.r.lowerBoundOut = lowerBoundOut; out.r.dir0 = dir0; out.r.dir1 = dir1; out.r.dir2 = dir2;
    return out;
}

PXG_TEST(npGjkSupportStep_step)
{
    std::mt19937 rng(1270127);
    const uint32_t MAXV = 40;
    std::vector<float> CPV, Z, VA, VB;
    std::vector<SsScalars> S;
    std::vector<Ref> ref;
    size_t nDirWr = 0, nNoWr = 0, nConv = 0;

    auto iv = [&](int r) -> V3 { return { float(int(rng() % (2 * r + 1)) - r), float(int(rng() % (2 * r + 1)) - r), float(int(rng() % (2 * r + 1)) - r) }; };
    auto randCloud = [&](uint32_t nb) -> std::vector<V3>
    {
        std::vector<V3> c(MAXV, V3{ 0, 0, 0 });
        for (uint32_t k = 0; k < nb; ++k) c[k] = iv(10);
        return c;
    };
    // standalone argmax over a V3 cloud (matches supportIdx over the flat buffer)
    auto supportIdxV = [&](V3 dir, const std::vector<V3>& V, uint32_t nb) -> uint32_t
    {
        uint32_t bestN = 0xFFFFFFFFu; float bestDist = -MAXF;
        for (uint32_t i = 0; i < nb; i += 32)
        {
            float d[32];
            for (uint32_t lane = 0; lane < 32; ++lane) { const uint32_t n = i + lane; d[lane] = (n < nb) ? dotX(dir, V[n]) : -MAXF; }
            float chunkMax = -MAXF; for (uint32_t lane = 0; lane < 32; ++lane) chunkMax = std::fmax(chunkMax, d[lane]);
            if (chunkMax > bestDist) { uint32_t ll = 0; for (uint32_t lane = 0; lane < 32; ++lane) if (d[lane] == chunkMax) { ll = lane; break; } bestDist = chunkMax; bestN = i + ll; }
        }
        return bestN;
    };
    // lb = splatX(dot(dV, supportVertexV)) for a given cp + clouds (independent of the bounds)
    auto peekLb = [&](V3 cp, const std::vector<V3>& A, const std::vector<V3>& B, uint32_t nbA, uint32_t nbB) -> float
    {
        float cpv[32]; for (uint32_t sl = 0; sl < 8; ++sl) { cpv[sl * 4] = cp.x; cpv[sl * 4 + 1] = cp.y; cpv[sl * 4 + 2] = cp.z; cpv[sl * 4 + 3] = 0.0f; }
        float dcp[32]; dotW(cpv, cpv, dcp);
        float cm[32]; for (uint32_t L = 0; L < 32; ++L) cm[L] = std::sqrt(dcp[L & 28u]);
        float dV[32]; for (uint32_t L = 0; L < 32; ++L) dV[L] = cpv[L] / cm[L];
        V3 dvVec = { dV[0], dV[1], dV[2] };
        uint32_t iA = supportIdxV({ -dvVec.x, -dvVec.y, -dvVec.z }, A, nbA);
        uint32_t iB = supportIdxV(dvVec, B, nbB);
        V3 sv = { A[iA].x - B[iB].x, A[iA].y - B[iB].y, A[iA].z - B[iB].z };
        float supp[32]; for (uint32_t L = 0; L < 32; ++L) { const uint32_t c = L & 3u; supp[L] = (c == 0) ? sv.x : (c == 1) ? sv.y : (c == 2) ? sv.z : 0.0f; }
        float dds[32]; dotW(dV, supp, dds); return dds[0];
    };

    auto emit = [&](V3 cp, const std::vector<V3>& cloudA, const std::vector<V3>& cloudB, uint32_t nbA, uint32_t nbB, SsScalars s)
    {
        float cpv[32], z[32];
        for (uint32_t sl = 0; sl < 8; ++sl) { cpv[sl * 4] = cp.x; cpv[sl * 4 + 1] = cp.y; cpv[sl * 4 + 2] = cp.z; cpv[sl * 4 + 3] = 0.0f; }
        for (uint32_t L = 0; L < 32; ++L) z[L] = float(500 + L); // distinct sentinels
        const uint32_t vbase = (uint32_t)(VA.size() / 4);
        for (uint32_t k = 0; k < MAXV; ++k)
        {
            VA.push_back(cloudA[k].x); VA.push_back(cloudA[k].y); VA.push_back(cloudA[k].z); VA.push_back(0);
            VB.push_back(cloudB[k].x); VB.push_back(cloudB[k].y); VB.push_back(cloudB[k].z); VB.push_back(0);
        }
        s.nbA = nbA; s.nbB = nbB;
        for (uint32_t L = 0; L < 32; ++L) { CPV.push_back(cpv[L]); Z.push_back(z[L]); }
        S.push_back(s);
        Ref r = stepRef(cpv, z, VA, VB, vbase, s);
        ref.push_back(r);
        if (r.r.directionWritten) { ++nDirWr; if (r.r.converged) ++nConv; } else ++nNoWr;
    };

    // family A: fresh iteration (lowerBound very low -> direction written), varied sizes
    for (uint32_t it = 0; it < 900; ++it)
    {
        V3 cp; do { cp = iv(8); } while (cp.x == 0 && cp.y == 0 && cp.z == 0);
        const uint32_t nbA = 4 + rng() % 30, nbB = 4 + rng() % 30;
        SsScalars s{}; s.lowerBound = -MAXF; s.upperBound = float(1 + rng() % 50);
        s.minSep = float(1 + rng() % 3) * 0.25f; s.convergenceRatio = 0.98f; s.size = rng() % 4;
        emit(cp, randCloud(nbA), randCloud(nbB), nbA, nbB, s);
    }
    // family B: high lowerBound so lb <= lowerBound (no direction write)
    for (uint32_t it = 0; it < 300; ++it)
    {
        V3 cp; do { cp = iv(8); } while (cp.x == 0 && cp.y == 0 && cp.z == 0);
        const uint32_t nbA = 4 + rng() % 30, nbB = 4 + rng() % 30;
        SsScalars s{}; s.lowerBound = 1.0e30f; s.upperBound = float(1 + rng() % 50);
        s.minSep = 0.5f; s.convergenceRatio = 0.98f; s.size = rng() % 4;
        emit(cp, randCloud(nbA), randCloud(nbB), nbA, nbB, s);
    }
    // family C: guaranteed convergence via the lb>minSep branch (minSep below any lb)
    for (uint32_t it = 0; it < 200; ++it)
    {
        V3 cp; do { cp = iv(6); } while (cp.x == 0 && cp.y == 0 && cp.z == 0);
        const uint32_t nbA = 4 + rng() % 30, nbB = 4 + rng() % 30;
        SsScalars s{}; s.lowerBound = -MAXF; s.upperBound = float(1 + rng() % 50);
        s.minSep = -1.0e30f; s.convergenceRatio = 0.98f; s.size = rng() % 4;
        emit(cp, randCloud(nbA), randCloud(nbB), nbA, nbB, s);
    }
    // family D: convergence via the ratio branch (upperBound*ratio below any lb)
    for (uint32_t it = 0; it < 200; ++it)
    {
        V3 cp; do { cp = iv(6); } while (cp.x == 0 && cp.y == 0 && cp.z == 0);
        const uint32_t nbA = 4 + rng() % 30, nbB = 4 + rng() % 30;
        SsScalars s{}; s.lowerBound = -MAXF; s.upperBound = -1.0e20f; s.minSep = 1.0e30f; s.convergenceRatio = 1.0f;
        s.size = rng() % 4;
        emit(cp, randCloud(nbA), randCloud(nbB), nbA, nbB, s);
    }
    // family E: lowerBound == lb exactly -> pins the strict `lb > lowerBound` (direction write flips)
    for (uint32_t it = 0; it < 150; ++it)
    {
        V3 cp; do { cp = iv(7); } while (cp.x == 0 && cp.y == 0 && cp.z == 0);
        const uint32_t nbA = 4 + rng() % 20, nbB = 4 + rng() % 20;
        std::vector<V3> A = randCloud(nbA), B = randCloud(nbB);
        const float lb = peekLb(cp, A, B, nbA, nbB);
        SsScalars s{}; s.lowerBound = lb; s.upperBound = float(1 + rng() % 50); s.minSep = 0.5f; s.convergenceRatio = 0.98f; s.size = rng() % 4;
        emit(cp, A, B, nbA, nbB, s);
    }
    // family F: minSep == lb exactly, ratio branch disabled -> pins `lb > minSep`
    for (uint32_t it = 0; it < 150; ++it)
    {
        V3 cp; do { cp = iv(7); } while (cp.x == 0 && cp.y == 0 && cp.z == 0);
        const uint32_t nbA = 4 + rng() % 20, nbB = 4 + rng() % 20;
        std::vector<V3> A = randCloud(nbA), B = randCloud(nbB);
        const float lb = peekLb(cp, A, B, nbA, nbB);
        SsScalars s{}; s.lowerBound = -MAXF; s.upperBound = 1.0e30f; s.minSep = lb; s.convergenceRatio = 1.0f; s.size = rng() % 4;
        emit(cp, A, B, nbA, nbB, s);
    }
    // family G: upperBound*convergenceRatio == lb exactly, minSep branch disabled -> pins `lb >= upperBound*ratio`
    for (uint32_t it = 0; it < 150; ++it)
    {
        V3 cp; do { cp = iv(7); } while (cp.x == 0 && cp.y == 0 && cp.z == 0);
        const uint32_t nbA = 4 + rng() % 20, nbB = 4 + rng() % 20;
        std::vector<V3> A = randCloud(nbA), B = randCloud(nbB);
        const float lb = peekLb(cp, A, B, nbA, nbB);
        SsScalars s{}; s.lowerBound = -MAXF; s.upperBound = lb; s.minSep = 1.0e30f; s.convergenceRatio = 1.0f; s.size = rng() % 4;
        emit(cp, A, B, nbA, nbB, s);
    }
    // family H: multi-chunk (nb=36) with the max support vertex duplicated in chunk 0 and chunk 1 ->
    // pins the strict cross-chunk `chunkMax > bestDist` tie-break (keeps the LOWER global index).
    for (uint32_t it = 0; it < 60; ++it)
    {
        const V3 cp = { 1, 0, 0 };                       // dV = (1,0,0); dirA = (-1,0,0)
        const uint32_t nbA = 36;
        std::vector<V3> A(MAXV, V3{ 0, 0, 0 });
        for (uint32_t k = 0; k < nbA; ++k) A[k] = { float(int(rng() % 9) - 9), float(int(rng() % 5) - 2), float(int(rng() % 5) - 2) }; // x in [-9,-1]
        const uint32_t lo = 3u, hi = 35u;                // duplicated extreme in both chunks
        A[lo] = { -10, 0, 0 }; A[hi] = { -10, 0, 0 };    // dot(dirA,·) = 10 at both -> tie
        const uint32_t nbB = 4 + rng() % 10;
        std::vector<V3> B = randCloud(nbB);
        SsScalars s{}; s.lowerBound = -MAXF; s.upperBound = float(1 + rng() % 50); s.minSep = 0.5f; s.convergenceRatio = 0.98f; s.size = rng() % 4;
        emit(cp, A, B, nbA, nbB, s);
    }

    const uint32_t N = (uint32_t)ref.size();
    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bCPV = createBufferFrom(dev, CPV);
    PxgBufferHandle bZ = createBufferFrom(dev, Z);
    PxgBufferHandle bVA = createBufferFrom(dev, VA);
    PxgBufferHandle bVB = createBufferFrom(dev, VB);
    PxgBufferHandle bS = createBufferFrom(dev, S);
    std::vector<float> zoInit(N * 32, -777.0f), supInit(N * 32, -777.0f);
    std::vector<SsResult> resInit(N);
    std::memset(resInit.data(), 0xEE, resInit.size() * sizeof(SsResult));
    PxgBufferHandle bZO = createBufferFrom(dev, zoInit);
    PxgBufferHandle bSup = createBufferFrom(dev, supInit);
    PxgBufferHandle bRes = createBufferFrom(dev, resInit);

    SsParams prm = { N, MAXV, 0, 0 };
    Kernel k("npGjkSupportStep.spv", "ctGjkSupportStep",
             { 4, 4, 16, 16, (uint32_t)sizeof(SsScalars), 4, 4, (uint32_t)sizeof(SsResult) }, sizeof(SsParams));
    k.dispatch((N + 63) / 64, { bCPV, bZ, bVA, bVB, bS, bZO, bSup, bRes }, prm);
    dev->flush();

    const auto gZO = downloadBuffer<float>(dev, bZO, N * 32);
    const auto gSup = downloadBuffer<float>(dev, bSup, N * 32);
    const auto gRes = downloadBuffer<SsResult>(dev, bRes, N);

    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const Ref& r = ref[t];
        const SsResult& g = gRes[t];
        if (g.indexA != r.r.indexA || g.indexB != r.r.indexB) ++mm;
        if (g.converged != r.r.converged || g.directionWritten != r.r.directionWritten) ++mm;
        if (asU32(g.lowerBoundOut) != asU32(r.r.lowerBoundOut)) ++mm;
        if (r.r.directionWritten)
            if (asU32(g.dir0) != asU32(r.r.dir0) || asU32(g.dir1) != asU32(r.r.dir1) || asU32(g.dir2) != asU32(r.r.dir2)) ++mm;
        for (uint32_t L = 0; L < 32; ++L)
        {
            if (asU32(gZO[t * 32 + L]) != asU32(r.z[L])) ++mm;
            if (asU32(gSup[t * 32 + L]) != asU32(r.sup[L])) ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "gjkSupportStep: %zu mismatches", mm);
    PXG_CHECK_MSG(nDirWr > 500 && nNoWr > 100 && nConv > 100,
                  "gjkSupportStep coverage thin (dirWr=%zu noWr=%zu conv=%zu)", nDirWr, nNoWr, nConv);

    for (PxgBufferHandle h : { bCPV, bZ, bVA, bVB, bS, bZO, bSup, bRes }) dev->destroyBuffer(h);
}

} // namespace
