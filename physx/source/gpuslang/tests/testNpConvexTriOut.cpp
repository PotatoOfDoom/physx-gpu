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

// Verification of npConvexTriOut.slang: ctConvexTriOut — the output stage of
// convexTriangleContactGen plus barycentricCoordinates / getTriangleActiveMask.
// One thread per pair writes five per-pair outputs; the deferred second-pass
// appends race via atomicAdd, so they are checked as a MULTISET.  All outputs
// are direct copies or integer packing except the barycentric active mask, whose
// classification is kept off the 0.97 boundary; the CPU reference is a bit-exact
// transcription so the mask matches.

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
struct U4 { uint32_t x, y, z, w; };
struct CoIn { F4 minNormal, pa, triV0, triV1, triV2; U4 misc, misc2; };
struct CoParams { uint32_t numPairs, pad0, pad1, pad2; };

static const uint32_t DeferredContactMask = 0x80000000u;
static const uint32_t NbContactsShift = 27u;
static const uint32_t POISON = 0xCDCDCDCDu;
static const uint32_t eMasks[7] = { 0u, 1u << 29, 2u << 29, 3u << 29, 4u << 29, 5u << 29, 6u << 29 }; // 0,eV0..

static float dotX(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 crossX(V3 a, V3 b)
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static F4 f4(V3 v, float w = 0.0f) { return { v.x, v.y, v.z, w }; }

static void barycentricRef(V3 p, V3 a, V3 b, V3 c, float& v, float& w)
{
    const V3 ab = sub(b, a), ac = sub(c, a);
    const V3 n = crossX(ab, ac);
    const V3 ap = sub(a, p), bp = sub(b, p), cp = sub(c, p);
    const float va = dotX(n, crossX(bp, cp));
    const float vb = dotX(n, crossX(cp, ap));
    const float vc = dotX(n, crossX(ap, bp));
    const float total = va + vb + vc;
    const float denom = total == 0.0f ? 0.0f : 1.0f / total;
    v = vb * denom; w = vc * denom;
}
static uint32_t activeMaskRef(float v, float w)
{
    const float ub = 0.97f, u = 1.0f - v - w;
    if (v > ub) return 2u << 29;      // eV1
    if (w > ub) return 3u << 29;      // eV2
    if (u > ub) return 1u << 29;      // eV0
    if ((v + u) > ub) return 4u << 29; // eE01
    if ((u + w) > ub) return 5u << 29; // eE02
    if ((w + v) > ub) return 6u << 29; // eE12
    return 0u;
}

static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct CoRef { V3 niNormal; uint32_t niIndex, gpuTri, ordered; float maxDepth; uint32_t pairOffset, padOffset; bool delay; uint32_t secondPassVal; };

PXG_TEST(npConvexTriOut_output)
{
    const uint32_t N = 1000;
    std::mt19937 rng(1100110);
    const uint32_t poolN = N + 16;

    // unique per-pair offsets (permutations) so every pair owns its slots
    std::vector<uint32_t> perm1(poolN), perm2(poolN);
    for (uint32_t i = 0; i < poolN; ++i) { perm1[i] = i; perm2[i] = i; }
    std::shuffle(perm1.begin(), perm1.end(), rng);
    std::shuffle(perm2.begin(), perm2.end(), rng);

    // barycentric (a,b,c) placements hitting each active-mask region
    // (a=u -> v0, b=v -> v1, c=w -> v2); a+b+c = 1, well off the 0.97 bound
    const float baryABC[7][3] = {
        { 1.0f / 3, 1.0f / 3, 1.0f / 3 }, // interior -> mask 0
        { 0.99f, 0.005f, 0.005f },        // u -> eV0
        { 0.005f, 0.99f, 0.005f },        // v -> eV1
        { 0.005f, 0.005f, 0.99f },        // w -> eV2
        { 0.48f, 0.5f, 0.02f },           // v+u -> eE01
        { 0.5f, 0.02f, 0.48f },           // u+w -> eE02
        { 0.02f, 0.48f, 0.5f },           // w+v -> eE12
    };

    std::vector<CoIn> in(N);
    std::vector<CoRef> ref(N);
    for (uint32_t p = 0; p < N; ++p)
    {
        CoIn& e = in[p];
        std::memset(&e, 0, sizeof(e));
        const uint32_t fam = p % 4;
        // a triangle with exact verts
        const V3 v0 = { float(int(rng() % 5) - 2), float(int(rng() % 5) - 2), 0 };
        const V3 v1 = { v0.x + 4, v0.y, 1 };
        const V3 v2 = { v0.x, v0.y + 4, -1 };
        const V3 minN = { float(int(rng() % 7) - 3), float(int(rng() % 7) - 3), float(int(rng() % 7) - 3) };
        const float maxSep = float(int(rng() % 21) - 10) * 0.25f;
        const uint32_t triIdx = rng() % 100000;
        const uint32_t remapIdx = rng() % ((1u << NbContactsShift) - 1);
        const uint32_t gti = (1u << 28) | (rng() % 4096);
        const uint32_t pairOffset = perm1[p];
        const uint32_t padOffset = perm2[p];

        bool delay; uint32_t nb;
        uint32_t region = 0;
        if (fam == 0) { delay = false; nb = 0; }               // zero contact
        else if (fam == 1) { delay = false; nb = 1 + rng() % 4; } // face, not deferred
        else { delay = true; nb = 1 + rng() % 4; region = 1 + (p % 6); } // deferred, cycle regions incl interior via p
        if (fam >= 2) region = p % 7; // spread all 7 regions across deferred pairs

        // build pa from barycentric coords of the chosen region
        const float* abc = baryABC[region];
        const V3 pa = { abc[0] * v0.x + abc[1] * v1.x + abc[2] * v2.x,
                        abc[0] * v0.y + abc[1] * v1.y + abc[2] * v2.y,
                        abc[0] * v0.z + abc[1] * v1.z + abc[2] * v2.z };

        e.minNormal = f4(minN, maxSep);
        e.pa = f4(pa);
        e.triV0 = f4(v0); e.triV1 = f4(v1); e.triV2 = f4(v2);
        e.misc = { nb, delay ? 1u : 0u, triIdx, remapIdx };
        e.misc2 = { gti, pairOffset, padOffset, 0 };

        CoRef& r = ref[p];
        r.niNormal = minN;
        r.maxDepth = maxSep;
        r.gpuTri = triIdx;
        r.pairOffset = pairOffset;
        r.padOffset = padOffset;
        r.delay = delay;
        const uint32_t orderedMask = (nb != 0 && !delay) ? (1u << 31) : 0u;
        r.ordered = orderedMask | triIdx;
        const uint32_t deferredMask = delay ? DeferredContactMask : 0u;
        r.niIndex = deferredMask + (nb << NbContactsShift) + remapIdx;
        if (delay)
        {
            float v, w; barycentricRef(pa, v0, v1, v2, v, w);
            r.secondPassVal = gti | activeMaskRef(v, w);
        }
    }

    uint32_t totalSP = 0;
    for (uint32_t p = 0; p < N; ++p) if (ref[p].delay) ++totalSP;

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<F4> niInit(poolN, { -7, -7, -7, -7 });
    std::vector<float> depthInit(poolN, -7.0f);
    std::vector<uint32_t> intermInit(poolN, POISON), orderedInit(poolN, POISON);
    std::vector<uint32_t> spInit(totalSP + 16, POISON);
    std::vector<uint32_t> ctrInit = { 0 };

    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bNI = createBufferFrom(dev, niInit);
    PxgBufferHandle bD = createBufferFrom(dev, depthInit);
    PxgBufferHandle bI = createBufferFrom(dev, intermInit);
    PxgBufferHandle bO = createBufferFrom(dev, orderedInit);
    PxgBufferHandle bSp = createBufferFrom(dev, spInit);
    PxgBufferHandle bCt = createBufferFrom(dev, ctrInit);

    CoParams prm = { N, 0, 0, 0 };
    Kernel k("npConvexTriOut.spv", "ctConvexTriOut",
             { (uint32_t)sizeof(CoIn), (uint32_t)sizeof(F4), 4, 4, 4, 4, 4 }, sizeof(CoParams));
    k.dispatch((N + 63) / 64, { bIn, bNI, bD, bI, bO, bSp, bCt }, prm);
    dev->flush();

    const auto gNI = downloadBuffer<F4>(dev, bNI, poolN);
    const auto gD = downloadBuffer<float>(dev, bD, poolN);
    const auto gI = downloadBuffer<uint32_t>(dev, bI, poolN);
    const auto gO = downloadBuffer<uint32_t>(dev, bO, poolN);
    const auto gSp = downloadBuffer<uint32_t>(dev, bSp, totalSP + 16);
    const auto gCtr = downloadBuffer<uint32_t>(dev, bCt, 1);

    PXG_CHECK_MSG(gCtr[0] == totalSP, "convexTriOut: second-pass count %u (want %u)", gCtr[0], totalSP);

    size_t mm = 0, nDelay = 0, nFace = 0, nZero = 0;
    std::vector<bool> usedPo(poolN, false), usedPad(poolN, false);
    std::vector<uint32_t> wantSP, gotSP;
    for (uint32_t p = 0; p < N; ++p)
    {
        const CoRef& r = ref[p];
        usedPo[r.pairOffset] = true; usedPad[r.padOffset] = true;
        const F4 ni = gNI[r.pairOffset];
        if (ni.x != r.niNormal.x || ni.y != r.niNormal.y || ni.z != r.niNormal.z || asU32(ni.w) != r.niIndex) ++mm;
        if (gD[r.pairOffset] != r.maxDepth) ++mm;
        if (gI[r.pairOffset] != r.gpuTri) ++mm;
        if (gO[r.padOffset] != r.ordered) ++mm;
        if (r.delay) { ++nDelay; wantSP.push_back(r.secondPassVal); }
        else if ((r.niIndex >> NbContactsShift & 0xf) != 0) ++nFace; else ++nZero;
    }
    for (uint32_t s = 0; s < totalSP; ++s) gotSP.push_back(gSp[s]);
    std::sort(wantSP.begin(), wantSP.end());
    std::sort(gotSP.begin(), gotSP.end());
    if (wantSP != gotSP) ++mm;

    // untouched pool slots keep their init
    for (uint32_t s = 0; s < poolN; ++s)
    {
        if (!usedPo[s] && (asU32(gNI[s].x) != asU32(-7.0f) || gD[s] != -7.0f || gI[s] != POISON)) ++mm;
        if (!usedPad[s] && gO[s] != POISON) ++mm;
    }
    for (uint32_t s = totalSP; s < totalSP + 16; ++s) if (gSp[s] != POISON) ++mm;

    PXG_CHECK_MSG(mm == 0, "convexTriOut: %zu mismatches", mm);
    PXG_CHECK_MSG(nDelay > 100 && nFace > 100 && nZero > 100,
                  "convexTriOut coverage thin (delay=%zu face=%zu zero=%zu)", nDelay, nFace, nZero);

    for (PxgBufferHandle h : { bIn, bNI, bD, bI, bO, bSp, bCt }) dev->destroyBuffer(h);
}

} // namespace
