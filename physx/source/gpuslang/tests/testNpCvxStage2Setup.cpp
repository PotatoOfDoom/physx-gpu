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

// P5 Step 4: convexConvexNphase_stage2Kernel on the ENGINE ABI, driven
// through PxgCudaShim exactly as PxgNarrowphaseCore.cpp launches it —
// argument blob {4,4,8,8,8,8,8,8} (56 bytes), grid numTests x (32,1)
// blocks (GJK_EPA_WARPS_PER_BLOCK = 1), REAL engine arrays + cooked-hull
// blobs (header uint4 at hull+16, extents float4 at hull+32) at live device
// addresses.  Two entries share one body:
//   - the production entry's memory effects: the tempKeepPairsArray /
//     numTests early-outs and the moreData words 9..12 store — every pair
//     here keeps the elision-contract writeback refreshedNb | 0x80000000
//     (convex type0 pairs are gated off collide(); sphere/capsule type0
//     pairs are pinned behind the transform-validity gate below because the
//     stub blobs are not real hulls — assembled-collide output is covered by
//     testCvxCollide.cpp) and the quatB tail is a bit-identical rewrite;
//   - the Debug entry (engine blob + trailing debug pointer) dumps every
//     scratch value the .cu computes; compared against the tranche-154
//     CPU transcription (shape flip, synthetic vertices, margins, blob
//     header decode, the shape1-gated minScale quirk, contactDistance +
//     expandedMargin, replaceBreakingThreshold, counts unpack, writeback
//     word) — same expected outputs as the bound-buffer test, minus the
//     injected collide() results, plus the transform-validity gate.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h"
#include "PxgContactManager.h"
#include "PxgConvexConvexShape.h"
#include "PxgPersistentContactManifold.h"
#include "PxsTransformCache.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct V3 { float x, y, z; };
// fixture record (the bound-buffer test's S2In, toleranceLength now per launch)
struct S2In { U4 meta0, meta1; F4 scale0, rot0, scale1, rot1, extents0, extents1, t0q, t0p, t1q, t1p; U4 misc; };
// the Debug entry's record (matches Stage2Debug in the shader, 13x16 = 208 B)
struct S2Dbg { F4 rotOut0, rotOut1, tBq, tBp, aToBq, aToBp, verts0[2]; U4 counts0, counts1; F4 params; U4 result, materials; };

static inline V3 v3(const F4& f) { return V3{ f.x, f.y, f.z }; }
static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 mul(const V3& a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 unit(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? mul(a, 1.f / l) : a; }
static V3 qRotInv(const F4& q, const V3& v) { const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z, w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz; return V3{ vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2, vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2, vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2 }; }
static F4 qMul(const F4& a, const F4& b)
{
    return F4{ a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y,
               a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z,
               a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x,
               a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}
static float pcmMargin(const V3& e, float tol) { return std::fmin(std::fmin(std::fmin(e.x, e.y), e.z) * 0.25f, tol * 0.05f); }
static bool quatIsUnit(const F4& q)
{
    const float mag = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return std::isfinite(mag) && std::fabs(mag - 1.f) < 1e-3f;
}

// CUDA transcription (the bound-buffer refSetup, collide() outputs removed).
static void refSetup(const S2In& s, float tol, S2Dbg& o)
{
    uint32_t type0 = s.meta0.x, type1 = s.meta1.x, hdr0 = s.meta0.z, hdr1 = s.meta1.z;
    uint32_t mat0 = s.meta0.y, mat1 = s.meta1.y;
    F4 rot0 = s.rot0, rot1 = s.rot1;
    V3 sc0 = v3(s.scale0), sc1 = v3(s.scale1);
    F4 e0 = s.extents0, e1 = s.extents1;
    F4 t0q = s.t0q, t1q = s.t1q;
    V3 t0p = v3(s.t0p), t1p = v3(s.t1p);
    float cd0 = s.t0p.w, cd1 = s.t1p.w;

    const bool flip = type1 < type0;
    if (flip)
    {
        std::swap(type0, type1); std::swap(hdr0, hdr1); std::swap(sc0, sc1);
        std::swap(mat0, mat1); std::swap(rot0, rot1);
        std::swap(e0, e1); std::swap(t0q, t1q); std::swap(t0p, t1p); std::swap(cd0, cd1);
    }

    const F4 c{ -t1q.x, -t1q.y, -t1q.z, t1q.w };
    const F4 aq = qMul(c, t0q);
    const V3 ap = qRotInv(t1q, sub(t0p, t1p));

    o.verts0[0] = F4{ 0, 0, 0, 0 }; o.verts0[1] = F4{ 0, 0, 0, 0 };
    float em, cm0, is0;
    uint32_t nv0, np0, ne0;
    if (type0 == 0)
    {
        em = sc0.x; cm0 = sc0.x; nv0 = 1; np0 = 0; ne0 = 0; is0 = 0.f;
    }
    else if (type0 == 2)
    {
        o.verts0[0] = F4{ 1, 0, 0, 0 }; o.verts0[1] = F4{ -1, 0, 0, 0 };
        em = sc0.y; cm0 = sc0.y; nv0 = 2; np0 = 0; ne0 = 0; is0 = 0.f;
    }
    else
    {
        em = 0.f;
        cm0 = pcmMargin(V3{ e0.x * sc0.x, e0.y * sc0.y, e0.z * sc0.z }, tol);
        nv0 = (hdr0 & 0xFFFF) >> 8; np0 = hdr0 & 0xFF; ne0 = hdr0 >> 16;
        float ms0 = 1.f;
        if (type1 == 3) ms0 = std::fmin(sc0.z, std::fmin(sc0.x, sc0.y)); // shape1-gated quirk
        is0 = e0.w * ms0;
    }
    const uint32_t nv1 = (hdr1 & 0xFFFF) >> 8, np1 = hdr1 & 0xFF, ne1 = hdr1 >> 16;
    float ms1 = 1.f;
    if (type1 == 3) ms1 = std::fmin(sc1.z, std::fmin(sc1.x, sc1.y));
    const float is1 = e1.w * ms1;

    const float cm1 = pcmMargin(V3{ e1.x * sc1.x, e1.y * sc1.y, e1.z * sc1.z }, tol);
    const float rbt = std::fmin(cm0, cm1) * 0.05f;
    const float cd = cd0 + cd1 + em;

    const uint32_t cw = s.misc.x;
    // collide() elided: bIntersection = false, count = refreshed count
    const uint32_t writeback = (cw & 0xFF) | 0x80000000u;
    const bool gate = quatIsUnit(t0q) && quatIsUnit(t1q); // fixture p is always finite

    o.rotOut0 = rot0; o.rotOut1 = rot1;
    o.tBq = t1q; o.tBp = F4{ t1p.x, t1p.y, t1p.z, 0 };
    o.materials = U4{ mat0, mat1, gate ? 1u : 0u, 0 };
    o.aToBq = aq;
    o.aToBp = F4{ ap.x, ap.y, ap.z, cd };
    o.counts0 = U4{ nv0, np0, ne0, type0 };
    o.counts1 = U4{ nv1, np1, ne1, type1 };
    o.params = F4{ is0, is1, rbt, em };
    o.result = U4{ cw & 0xFF, cw >> 8, writeback, flip ? 1u : 0u };
}

static bool eqF4(const F4& a, const F4& b) { return almostEqual(a.x, b.x, 1e-5f, 1e-6f) && almostEqual(a.y, b.y, 1e-5f, 1e-6f) && almostEqual(a.z, b.z, 1e-5f, 1e-6f) && almostEqual(a.w, b.w, 1e-5f, 1e-6f); }
static bool eqU4(const U4& a, const U4& b) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

static std::vector<uint8_t> s2Loader(const char* file) { return loadShaderBlob(file); }

PXG_TEST(npCvxStage2Setup_engineAbi)
{
    using namespace physx;

    std::mt19937 rng(154001);
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { V3 ax = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }); const float a = rf(-3.1f, 3.1f); return F4{ ax.x * std::sin(a * .5f), ax.y * std::sin(a * .5f), ax.z * std::sin(a * .5f), std::cos(a * .5f) }; };

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npCvxStage2Setup.spv", "convexConvexNphase_stage2Kernel", 56u, 0 },         // engine ABI
        { "npCvxStage2SetupDbg.spv", "convexConvexNphase_stage2KernelDebug", 64u, 0 }, // + debug ptr
    };
    PxgCudaShim shim(dev, manifest, 2, &s2Loader);

    const float tols[2] = { 1.1f, 2.7f };
    const uint32_t N = 3000;   // pairs per launch (incl. GUARD tail)
    const uint32_t GUARD = 4;

    size_t cSphere = 0, cCapsule = 0, cBox1 = 0, cFlip = 0, cSkipPair = 0, cBadQuat = 0;

    for (uint32_t launch = 0; launch < 2; ++launch)
    {
        const float tol = tols[launch];
        const uint32_t numTests = N - GUARD;

        std::vector<S2In> in(N);
        std::vector<uint32_t> keep(N);
        std::vector<S2Dbg> ref(N);

        for (uint32_t i = 0; i < N; ++i)
        {
            S2In s{};
            const uint32_t kinds[4] = { 0u, 2u, 3u, 5u }; // sphere, capsule, box, convex mesh
            const uint32_t ty0 = kinds[rng() % 4], ty1 = kinds[rng() % 4];
            if (ty0 == 0 || ty1 == 0) ++cSphere;
            if (ty0 == 2 || ty1 == 2) ++cCapsule;
            if (ty1 == 3 || (ty1 < ty0 && ty0 == 3)) ++cBox1;
            if (ty1 < ty0) ++cFlip;
            s.meta0 = U4{ ty0, rng() % 512, (rng() % 65536) << 16 | (rng() % 64) << 8 | (rng() % 64), 0 };
            s.meta1 = U4{ ty1, rng() % 512, (rng() % 65536) << 16 | (rng() % 64) << 8 | (rng() % 64), 0 };
            s.scale0 = F4{ rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 2), 0 };
            s.scale1 = F4{ rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 2), 0 };
            s.rot0 = randQuat(); s.rot1 = randQuat();
            s.extents0 = F4{ rf(0.3f, 2), rf(0.3f, 2), rf(0.3f, 2), rf(0.1f, 1) };
            s.extents1 = F4{ rf(0.3f, 2), rf(0.3f, 2), rf(0.3f, 2), rf(0.1f, 1) };
            s.t0q = randQuat(); s.t1q = randQuat();
            if (i % 16 == 5) // non-unit quat -> transform-validity gate off
            {
                s.t0q = F4{ s.t0q.x * 1.2f, s.t0q.y * 1.2f, s.t0q.z * 1.2f, s.t0q.w * 1.2f };
                ++cBadQuat;
            }
            // The assembled collide() (npCvxCollide) now runs for post-flip
            // sphere/capsule type0 — but this fixture's 64-byte stub blobs are
            // NOT valid hulls (the collide vertex/plane reads would run off the
            // stub).  Pin those pairs behind the transform-validity gate with a
            // non-unit t0q: bIntersection stays false, the elision-contract
            // writeback expectations hold, and the debug entry still verifies
            // the full sphere/capsule SETUP arithmetic (which runs regardless
            // of the gate).  Assembled-collide contact output is covered by
            // testCvxCollide.cpp on a REAL cooked hull.
            const uint32_t minType = ty0 < ty1 ? ty0 : ty1;
            if (minType == 0u || minType == 2u)
                s.t0q = F4{ s.t0q.x * 1.2f, s.t0q.y * 1.2f, s.t0q.z * 1.2f, s.t0q.w * 1.2f };
            s.t0p = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), rf(0.02f, 0.3f) };
            s.t1p = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), rf(0.02f, 0.3f) };
            const uint32_t refreshed = rng() % 5, initial = rng() % 5;
            s.misc = U4{ refreshed | (initial << 8), 0, 0, 0 };
            in[i] = s;
            keep[i] = (i % 7 == 3) ? 0u : 1u; // stage-1 early-out coverage
            if (i < numTests && keep[i] == 0) ++cSkipPair;
            refSetup(s, tol, ref[i]);
        }

        // ---- marshal into REAL engine arrays ----
        std::vector<PxgContactManagerInput> cmIn(N);
        std::vector<PxgShape> shapes(2 * N);
        std::vector<PxsCachedTransform> cache(2 * N);
        std::vector<float> cdArr(2 * N);
        std::vector<PxgPersistentContactManifold> mfs(N);
        std::vector<uint8_t> blobs(size_t(2) * N * 64, 0);
        std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
        std::memset(cache.data(), 0, cache.size() * sizeof(PxsCachedTransform));
        std::memset(mfs.data(), 0xCD, mfs.size() * sizeof(PxgPersistentContactManifold));

        PxgDevicePtr dBlobs = 0;
        PXG_CHECK_MSG(shim.memAlloc(&dBlobs, blobs.size()) == PxgResult::Success, "memAlloc blobs");
        if (!dBlobs) { std::printf("  NOTE: no BDA -> skip\n"); return; }

        for (uint32_t i = 0; i < N; ++i)
        {
            const S2In& s = in[i];
            cmIn[i].shapeRef0 = 2 * i;
            cmIn[i].shapeRef1 = 2 * i + 1;
            cmIn[i].transformCacheRef0 = 2 * i;
            cmIn[i].transformCacheRef1 = 2 * i + 1;

            for (int k = 0; k < 2; ++k)
            {
                const U4& meta = k == 0 ? s.meta0 : s.meta1;
                const F4& sc = k == 0 ? s.scale0 : s.scale1;
                const F4& ro = k == 0 ? s.rot0 : s.rot1;
                const F4& ex = k == 0 ? s.extents0 : s.extents1;
                PxgShape& sh = shapes[2 * i + k];
                sh.type = meta.x;
                sh.materialIndex = meta.y;
                sh.scale.scale = PxVec3(sc.x, sc.y, sc.z);
                sh.scale.rotation = PxQuat(ro.x, ro.y, ro.z, ro.w);
                sh.hullOrMeshPtr = size_t(dBlobs + uint64_t(2 * i + k) * 64u);
                // blob: header uint4 at +16 (x = the packed counts word), extents at +32
                U4 hdr{ meta.z, 0, 0, 0 };
                std::memcpy(&blobs[size_t(2 * i + k) * 64 + 16], &hdr, 16);
                std::memcpy(&blobs[size_t(2 * i + k) * 64 + 32], &ex, 16);
            }

            cache[2 * i].transform.q = PxQuat(s.t0q.x, s.t0q.y, s.t0q.z, s.t0q.w);
            cache[2 * i].transform.p = PxVec3(s.t0p.x, s.t0p.y, s.t0p.z);
            cache[2 * i + 1].transform.q = PxQuat(s.t1q.x, s.t1q.y, s.t1q.z, s.t1q.w);
            cache[2 * i + 1].transform.p = PxVec3(s.t1p.x, s.t1p.y, s.t1p.z);
            cdArr[2 * i] = s.t0p.w;
            cdArr[2 * i + 1] = s.t1p.w;

            mfs[i].mNbContacts = s.misc.x; // the stage-1 counts word
            mfs[i].mQuatB = PxAlignedQuat(rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-1, 1));
        }

        PxgDevicePtr dCmIn = 0, dShapes = 0, dCache = 0, dCd = 0, dMf = 0, dKeep = 0, dDbg = 0;
        shim.memAlloc(&dCmIn, cmIn.size() * sizeof(PxgContactManagerInput));
        shim.memAlloc(&dShapes, shapes.size() * sizeof(PxgShape));
        shim.memAlloc(&dCache, cache.size() * sizeof(PxsCachedTransform));
        shim.memAlloc(&dCd, cdArr.size() * 4);
        shim.memAlloc(&dMf, mfs.size() * sizeof(PxgPersistentContactManifold));
        shim.memAlloc(&dKeep, N * 4);
        shim.memAlloc(&dDbg, size_t(N) * sizeof(S2Dbg));
        shim.memcpyHtoD(dBlobs, blobs.data(), blobs.size());
        shim.memcpyHtoD(dCmIn, cmIn.data(), cmIn.size() * sizeof(PxgContactManagerInput));
        shim.memcpyHtoD(dShapes, shapes.data(), shapes.size() * sizeof(PxgShape));
        shim.memcpyHtoD(dCache, cache.data(), cache.size() * sizeof(PxsCachedTransform));
        shim.memcpyHtoD(dCd, cdArr.data(), cdArr.size() * 4);
        shim.memcpyHtoD(dMf, mfs.data(), mfs.size() * sizeof(PxgPersistentContactManifold));
        shim.memcpyHtoD(dKeep, keep.data(), N * 4);
        shim.memsetD32(dDbg, 0xCCCCCCCC, size_t(N) * sizeof(S2Dbg) / 4);

        // ---- 1. the PRODUCTION entry: engine blob, real memory effects ----
        struct Stage2Blob
        {
            uint32_t numTests;
            float toleranceLength;
            uint64_t tempKeepPairsArray, cmInputs, cmManifolds, shapes, transformCache, contactDistance;
        } blob = { numTests, tol, dKeep, dCmIn, dMf, dShapes, dCache, dCd };
        static_assert(sizeof(Stage2Blob) == 56, "stage2 engine blob is 56 bytes");

        // engine grid: numBlocks = ceil(numTests / warpsPerBlock), warpsPerBlock = 1
        PXG_CHECK_MSG(shim.launchKernel(0, numTests, 1, 1, &blob, sizeof(blob)) == PxgResult::Success,
                      "launch stage2 (engine ABI)");
        shim.streamSynchronize();

        std::vector<PxgPersistentContactManifold> gMf(N);
        shim.memcpyDtoH(gMf.data(), dMf, gMf.size() * sizeof(PxgPersistentContactManifold));

        size_t mmW = 0, mmU = 0;
        for (uint32_t i = 0; i < N; ++i)
        {
            PxgPersistentContactManifold expectMf = mfs[i];
            if (i < numTests && keep[i] != 0)
                expectMf.mNbContacts = (in[i].misc.x & 0xFF) | 0x80000000u; // collide-elided writeback
            if (std::memcmp(&gMf[i], &expectMf, sizeof(PxgPersistentContactManifold)) != 0)
            {
                if (i < numTests && keep[i] != 0) ++mmW;
                else ++mmU; // early-out pairs / guard tail must be untouched
            }
        }
        PXG_CHECK_MSG(mmW == 0, "stage2 launch %u: %zu writeback mismatches", launch, mmW);
        PXG_CHECK_MSG(mmU == 0, "stage2 launch %u: %zu early-out/guard violations", launch, mmU);

        // ---- 2. the DEBUG entry: full setup arithmetic, bit-verified ----
        shim.memcpyHtoD(dMf, mfs.data(), mfs.size() * sizeof(PxgPersistentContactManifold)); // reset
        struct Stage2DebugBlob
        {
            uint32_t numTests;
            float toleranceLength;
            uint64_t tempKeepPairsArray, cmInputs, cmManifolds, shapes, transformCache, contactDistance;
            uint64_t debugOut;
        } dblob = { numTests, tol, dKeep, dCmIn, dMf, dShapes, dCache, dCd, dDbg };
        static_assert(sizeof(Stage2DebugBlob) == 64, "stage2 debug blob is 64 bytes");
        PXG_CHECK_MSG(shim.launchKernel(1, numTests, 1, 1, &dblob, sizeof(dblob)) == PxgResult::Success,
                      "launch stage2 (debug entry)");
        shim.streamSynchronize();

        std::vector<S2Dbg> gDbg(N);
        shim.memcpyDtoH(gDbg.data(), dDbg, gDbg.size() * sizeof(S2Dbg));

        size_t mmD = 0, mmS = 0;
        for (uint32_t i = 0; i < N; ++i)
        {
            const S2Dbg& g = gDbg[i];
            if (i >= numTests || keep[i] == 0)
            {
                // untouched: 0xCC sentinel fill
                const uint32_t* w = reinterpret_cast<const uint32_t*>(&g);
                for (size_t k = 0; k < sizeof(S2Dbg) / 4; ++k)
                    if (w[k] != 0xCCCCCCCCu) { ++mmS; break; }
                continue;
            }
            const S2Dbg& r = ref[i];
            if (!eqF4(r.rotOut0, g.rotOut0) || !eqF4(r.rotOut1, g.rotOut1) ||
                !eqF4(r.tBq, g.tBq) || !eqF4(r.tBp, g.tBp) ||
                !eqU4(r.materials, g.materials) ||
                !eqF4(r.aToBq, g.aToBq) || !eqF4(r.aToBp, g.aToBp) ||
                !eqF4(r.verts0[0], g.verts0[0]) || !eqF4(r.verts0[1], g.verts0[1]) ||
                !eqU4(r.counts0, g.counts0) || !eqU4(r.counts1, g.counts1) ||
                !eqF4(r.params, g.params) || !eqU4(r.result, g.result)) ++mmD;
        }
        PXG_CHECK_MSG(mmD == 0, "stage2 launch %u: %zu debug-record mismatches", launch, mmD);
        PXG_CHECK_MSG(mmS == 0, "stage2 launch %u: %zu debug sentinel violations", launch, mmS);

        // debug run performs the same manifold writes
        shim.memcpyDtoH(gMf.data(), dMf, gMf.size() * sizeof(PxgPersistentContactManifold));
        size_t mmW2 = 0;
        for (uint32_t i = 0; i < N; ++i)
        {
            PxgPersistentContactManifold expectMf = mfs[i];
            if (i < numTests && keep[i] != 0)
                expectMf.mNbContacts = (in[i].misc.x & 0xFF) | 0x80000000u;
            if (std::memcmp(&gMf[i], &expectMf, sizeof(PxgPersistentContactManifold)) != 0) ++mmW2;
        }
        PXG_CHECK_MSG(mmW2 == 0, "stage2 launch %u: %zu debug-run writeback mismatches", launch, mmW2);

        for (PxgDevicePtr pfree : { dBlobs, dCmIn, dShapes, dCache, dCd, dMf, dKeep, dDbg })
            shim.memFree(pfree);
    }

    const size_t total = size_t(2) * N;
    PXG_CHECK_MSG(cSphere > total / 10 && cCapsule > total / 10 && cBox1 > total / 10 && cFlip > total / 10 &&
                  cSkipPair > total / 20 && cBadQuat > total / 40,
                  "stage2: coverage thin (sph %zu cap %zu box1 %zu flip %zu skip %zu badQuat %zu)",
                  cSphere, cCapsule, cBox1, cFlip, cSkipPair, cBadQuat);
}

} // namespace
