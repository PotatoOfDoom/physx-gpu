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

// testPsInflatable.cpp -- P9.6 fixture for ps_calculateInflatableVolume and
// ps_solveInflatableVolume, ported as npPsInflatable.slang.
//
// ORACLE: host transcription, carried by generated-and-pinned offsets, whole-region
// 0xAB poison compare, and the sweep.  Neither kernel body has a PX_CUDA_CALLABLE
// original, so this is the P8.73-P8.80 arrangement, not a true oracle.
//
// WHAT IS BIT-EXACT HERE, AND WHAT IS NOT.  Two things force a tolerance, and both
// are named rather than smoothed over:
//
//   1. FLOAT ATOMICS.  Kernel 1 accumulates a vertex normal with one atomic add per
//      (triangle, vertex).  A vertex shared by k triangles gets k additions in an
//      order the host cannot predict.  The fixture therefore gives one inflatable
//      DISJOINT triangles -- every vertex in exactly one triangle, so exactly one
//      addition, bit-exact -- and a separate one a CLOSED octahedron, where the
//      sharing is real and the comparison uses a relative epsilon.
//   2. THE CUBE ROOT.  The CUDA calls cbrtf; SPIR-V has none, and CUDA's cbrtf,
//      glibc's cbrtf and pow(x,1/3) are three implementations that agree to a few
//      ULP and not to the bit.  So `lambda` is compared bit-exactly ONLY where the
//      argument is exactly zero (every implementation returns 0), and with a
//      relative epsilon otherwise.  Everything kernel 2 derives from lambda
//      inherits that.
//
// THE BLOCK REDUCTION IS BIT-EXACT and is the load-bearing part of the mirror: it
// is a fixed XOR butterfly over fixed lanes, so the host can and does reproduce the
// exact summation tree (warp butterfly -> one slot per warp -> warp-0 butterfly).
// Getting that tree wrong would show up immediately as a volume mismatch.
//
// GEOMETRY: the standard four-system pool with a DECOY that is never active, an
// active list that is NOT a permutation, cloth arrays padded past mNumClothBuffers
// with real records, and -- as P9.5's audit taught -- a cloth slot whose
// mParticleBufferIndex is NOT its own slot, so that indirection is under test.

#include "pxgTestFramework.h"

#include "host_defines.h"
#include "PxgParticleSystem.h"
#include "PxParticleBuffer.h"

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

struct F4 { float x, y, z, w; };
struct V3 { float x, y, z; };
static V3 v3(float x, float y, float z) { V3 r = { x, y, z }; return r; }
static V3 operator-(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static V3 operator+(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static V3 operator*(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b)
{
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
// PxVec3::normalize(): scales in place by a reciprocal only when m > 0.
static V3 normalize3(V3 v)
{
    const float m = std::sqrt(dot3(v, v));
    return (m > 0.0f) ? v * (1.0f / m) : v;
}
static float hMax(float a, float b) { return a > b ? a : b; }
static float hMin(float a, float b) { return a < b ? a : b; }

// ---------------------------------------------------------------------------
// The block reduction, reproduced EXACTLY: reduction.cuh:188-200 (the XOR
// butterfly) and :226-247 (warp slots then a warp-0 butterfly).  Float addition
// is commutative bit-for-bit, so every lane of a butterfly step ends with the same
// value and the tree is well defined without knowing which lane "ran first".
// ---------------------------------------------------------------------------
static const uint32_t INF_BLOCK = 256u;
static const uint32_t INF_WARPS = INF_BLOCK / 32u;

static void warpButterflyAdd(float lane[32])
{
    for (uint32_t r = 16u; r > 0u; r >>= 1)
    {
        float nx[32];
        for (uint32_t i = 0; i < 32u; ++i) nx[i] = lane[i] + lane[i ^ r];
        for (uint32_t i = 0; i < 32u; ++i) lane[i] = nx[i];
    }
}

// Returns what thread 0 sees.
static float blockReduceAdd(const float value[INF_BLOCK])
{
    float warpSum[INF_WARPS];
    for (uint32_t w = 0; w < INF_WARPS; ++w)
    {
        float lane[32];
        for (uint32_t i = 0; i < 32u; ++i) lane[i] = value[w * 32u + i];
        warpButterflyAdd(lane);
        warpSum[w] = lane[0];
    }
    float lane[32];
    for (uint32_t i = 0; i < 32u; ++i) lane[i] = (i < INF_WARPS) ? warpSum[i] : 0.0f;
    warpButterflyAdd(lane);
    return lane[0];
}

struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };

template <typename T>
DevBuf makeDev(PxgSlangComputeDevice* dev, const std::vector<T>& data)
{
    DevBuf b;
    b.h = createBufferFrom(dev, data);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

struct CalcPush { uint64_t particleSystems, activeParticleSystems; };
struct SolvePush { uint64_t particleSystems, activeParticleSystems; float dt, coefficient; };
static_assert(sizeof(CalcPush) == 16, "kTwoPtr blob is 16 bytes");
static_assert(sizeof(SolvePush) == 24, "the {8,8,4,4} blob is 24 bytes");

static const uint32_t POISON_U32 = 0xABABABABu;

// ---------------------------------------------------------------------------
PXG_TEST(psInflatableLayout)
{
    PXG_CHECK_MSG(sizeof(PxParticleCloth) == 28,
                  "sizeof(PxParticleCloth) == 28 (got %zu)", sizeof(PxParticleCloth));
    PXG_CHECK(offsetof(PxParticleCloth, startVertexIndex) == 0);
    PXG_CHECK(offsetof(PxParticleCloth, numVertices) == 4);
    PXG_CHECK(offsetof(PxParticleCloth, clothBlendScale) == 8);
    PXG_CHECK(offsetof(PxParticleCloth, restVolume) == 12);
    PXG_CHECK(offsetof(PxParticleCloth, pressure) == 16);
    PXG_CHECK(offsetof(PxParticleCloth, startTriangleIndex) == 20);
    PXG_CHECK(offsetof(PxParticleCloth, numTriangles) == 24);

    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mCloths) == 48);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mInflatableLambda) == 80);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mTriangles) == 32);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mNumCloths) == 100);

    PXG_CHECK(offsetof(PxgParticleSystem, mNormalArray) == 384);
    PXG_CHECK(offsetof(PxgParticleSystem, mAccumDeltaP) == 152);
    PXG_CHECK(offsetof(PxgParticleSystem, mClothSimBuffers) == 432);
    PXG_CHECK(offsetof(PxgParticleSystem, mNumClothBuffers) == 464);
    PXG_CHECK(offsetof(PxgParticleSystem, mParticleBufferRunsum) == 408);
    PXG_CHECK_MSG(offsetof(PxgParticleSystem, mCommonData)
                      + offsetof(PxGpuParticleData, mParticleContactDistance) == 104,
                  "mCommonData.mParticleContactDistance must sit at 104 (got %zu)",
                  offsetof(PxgParticleSystem, mCommonData)
                      + offsetof(PxGpuParticleData, mParticleContactDistance));
}

// ---------------------------------------------------------------------------

struct Infl
{
    uint32_t startVertexIndex = 0, numVertices = 0;
    float clothBlendScale = 0.0f, restVolume = 0.0f, pressure = 0.0f;
    uint32_t startTriangleIndex = 0, numTriangles = 0;
    bool closed = false;      // true -> shared vertices, tolerance region
};

struct ClothBuf
{
    uint32_t particleBufferIndex = 0;
    std::vector<Infl> infl;
    std::vector<uint32_t> tris;      // buffer-LOCAL vertex indices
    DevBuf bTris, bCloths, bLambda;
};

struct Sys
{
    uint32_t numClothBuffers = 0;
    std::vector<ClothBuf> cloths;
    std::vector<F4> pos;
    std::vector<uint32_t> unsortedToSorted, runsum;
    float particleContactDistance = 0.0f;
    DevBuf bPos, bU2S, bRunsum, bClothArr, bNormals, bAccum;
};

PXG_TEST(psInflatable)
{
    PxgSlangComputeDevice* dev = testDevice();

    // Sized so each of the three particle buffers gets a DISJOINT 850-vertex
    // stride: with every buffer at offset 0 the cloth buffers' vertex ranges
    // overlap, they accumulate into the same normals, and the result is both
    // double-counted and racy.  That was the first run's failure.
    const uint32_t NUM_PARTICLES = 2600u;
    const uint32_t PAD = 32u;
    const uint32_t CLOTH_ARRAY_SLOTS = 3u;
    const uint32_t BUFFER_STRIDE = 850u;
    const uint32_t INFL_SLOTS = 6u;          // allocated per cloth buffer
    const float COEFF = 0.75f;               // not 1, so the scale is observable
    const float DT = 1.0f / 90.0f;           // the DEAD parameter

    // The big inflatable has 300 DISJOINT triangles, so numPasses == 2 and the
    // pass loop is a real loop rather than a single iteration -- without that,
    // every mutant touching the pass decomposition is an identity.
    // > 256 so the reduction runs more than one pass, and small enough that
    // 3 * BIG_TRIS still fits inside one particle buffer's stride.
    const uint32_t BIG_TRIS = 270u;

    auto buildSys = [&](uint32_t salt, float contactDistance,
                        std::vector<uint32_t> inflCountPerBuffer)
    {
        Sys s;
        s.numClothBuffers = uint32_t(inflCountPerBuffer.size());
        while (inflCountPerBuffer.size() < CLOTH_ARRAY_SLOTS)
            inflCountPerBuffer.push_back(1u);
        s.particleContactDistance = contactDistance;

        s.pos.resize(NUM_PARTICLES);
        for (uint32_t i = 0; i < NUM_PARTICLES; ++i)
        {
            // Spread in all three axes and OFF THE ORIGIN, so the centre-of-mass
            // subtraction is not an identity.
            s.pos[i] = { 11.0f + float(salt) * 0.5f + float((i * 7u) % 23u) * 0.31f,
                         -7.0f + float((i * 11u) % 19u) * 0.27f,
                         5.0f + float((i * 13u) % 17u) * 0.23f,
                         0.5f + float(i % 7u) * 0.1f };
        }
        s.unsortedToSorted.resize(NUM_PARTICLES);
        for (uint32_t i = 0; i < NUM_PARTICLES; ++i)
            s.unsortedToSorted[i] = (i * 37u + 11u) % NUM_PARTICLES;
        s.runsum = { 0u, BUFFER_STRIDE, 2u * BUFFER_STRIDE };

        uint32_t bIdx = 0;
        for (uint32_t nInf : inflCountPerBuffer)
        {
            ClothBuf c;
            // As P9.5's audit taught: NOT the identity, and a bijection so no two
            // cloth slots share a particle buffer.
            c.particleBufferIndex = (bIdx + 1u) % CLOTH_ARRAY_SLOTS;
            ++bIdx;

            uint32_t vCursor = 0, tCursor = 0;
            for (uint32_t k = 0; k < nInf; ++k)
            {
                Infl inf;
                inf.clothBlendScale = 0.25f + float(k) * 0.05f;
                if (k == 0u)
                {
                    // DISJOINT: triangle t uses vertices 3t, 3t+1, 3t+2.  Every
                    // vertex belongs to exactly one triangle -> exactly one atomic
                    // add -> bit-exact.
                    inf.numTriangles = BIG_TRIS;
                    inf.numVertices = BIG_TRIS * 3u;
                    inf.startVertexIndex = vCursor;
                    inf.startTriangleIndex = tCursor;   // an INDEX-ARRAY offset
                    inf.closed = false;
                    inf.pressure = 1.5f + float(salt) * 0.001f;
                    for (uint32_t t = 0; t < inf.numTriangles; ++t)
                    {
                        c.tris.push_back(vCursor + 3u * t + 0u);
                        c.tris.push_back(vCursor + 3u * t + 1u);
                        c.tris.push_back(vCursor + 3u * t + 2u);
                    }
                    // restVolume chosen so restVolume*pressure - volume is POSITIVE
                    // for this one (lambda > 0 drives kernel 2).
                    // MEASURED: this inflatable's signed volume is ~155.  With
                    // restVolume*pressure four orders of magnitude larger the volume
                    // term vanishes into the rounding and the i3/i4 mutants survive.
                    inf.restVolume = 266.0f;
                }
                else if (k == 1u || k == 2u)
                {
                    // A CLOSED octahedron: 8 triangles over 6 shared vertices.
                    inf.numTriangles = 8u;
                    inf.numVertices = 6u;
                    inf.startVertexIndex = vCursor;
                    inf.startTriangleIndex = tCursor;
                    inf.closed = true;
                    inf.pressure = 2.25f;
                    // k == 1: a POSITIVE argument, so the closed mesh drives kernel 2
                    // through the shared-normal path.
                    // k == 2: a NEGATIVE restVolume drives restVolume*pressure - volume
                    // hard negative, so PxMax(0, ..) returns EXACTLY ZERO -- and cbrt(0)
                    // is 0 in every implementation, which is the only place `lambda`
                    // can be compared bit-exactly.  It also makes kernel 2's second
                    // early-out (lambda == 0) fire.
                    // MEASURED: the octahedron's signed volume is ~-4.75.
                    inf.restVolume = (k == 1u) ? 4.0f : -1.0e6f;
                    static const uint32_t OCT[8][3] = {
                        {0,2,4},{2,1,4},{1,3,4},{3,0,4},
                        {2,0,5},{1,2,5},{3,1,5},{0,3,5}
                    };
                    for (uint32_t t = 0; t < 8u; ++t)
                        for (uint32_t j = 0; j < 3u; ++j)
                            c.tris.push_back(vCursor + OCT[t][j]);
                }
                else
                {
                    // pressure == 0 -> BOTH kernels early-out; its lambda slot must
                    // come back POISONED and its normals untouched.
                    inf.numTriangles = 5u;
                    inf.numVertices = 15u;
                    inf.startVertexIndex = vCursor;
                    inf.startTriangleIndex = tCursor;
                    inf.closed = false;
                    inf.pressure = 0.0f;
                    inf.restVolume = 9.0f;
                    for (uint32_t t = 0; t < inf.numTriangles; ++t)
                    {
                        c.tris.push_back(vCursor + 3u * t + 0u);
                        c.tris.push_back(vCursor + 3u * t + 1u);
                        c.tris.push_back(vCursor + 3u * t + 2u);
                    }
                }
                vCursor += inf.numVertices;
                tCursor += inf.numTriangles * 3u;
                c.infl.push_back(inf);
            }
            s.cloths.push_back(c);
        }
        return s;
    };

    // A: two live cloth buffers (3 and 2 inflatables).  DECOY never active.
    // B: one.  C: two.  Contact distances differ so maxDelta differs per system.
    Sys sysA = buildSys(100u, 0.40f, { 4u, 2u });
    Sys sysD = buildSys(200u, 9.90f, { 1u, 1u, 1u });
    Sys sysB = buildSys(300u, 0.005f, { 4u });         // tiny -> maxDelta BINDS
    Sys sysC = buildSys(400u, 0.55f, { 3u, 1u });

    Sys* order[4] = { &sysA, &sysD, &sysB, &sysC };

    uint32_t maxCloths = 0, maxClothBuffers = 0;
    for (Sys* s : order)
    {
        maxClothBuffers = PxMax(maxClothBuffers, uint32_t(s->cloths.size()));
        for (auto& c : s->cloths) maxCloths = PxMax(maxCloths, uint32_t(c.infl.size()));
    }
    PXG_CHECK_MSG((BIG_TRIS + 255u) / 256u > 1u,
                  "the big inflatable must span more than one reduction pass");
    PXG_CHECK_MSG(sysB.numClothBuffers < CLOTH_ARRAY_SLOTS,
                  "B must have real cloth records past its own bound");

    for (Sys* s : order)
    {
        s->bPos = makeDev(dev, s->pos);
        s->bU2S = makeDev(dev, s->unsortedToSorted);
        s->bRunsum = makeDev(dev, s->runsum);

        std::vector<uint32_t> nrm(size_t(NUM_PARTICLES + PAD) * 4u, POISON_U32);
        std::vector<uint32_t> acc(size_t(NUM_PARTICLES + PAD) * 4u, POISON_U32);
        for (uint32_t i = 0; i < NUM_PARTICLES * 4u; ++i) acc[i] = f2u(0.0f);
        s->bNormals = makeDev(dev, nrm);
        s->bAccum = makeDev(dev, acc);

        std::vector<PxgParticleClothSimBuffer> arr(s->cloths.size());
        for (uint32_t b = 0; b < uint32_t(s->cloths.size()); ++b)
        {
            ClothBuf& c = s->cloths[b];
            c.bTris = makeDev(dev, c.tris);

            std::vector<PxParticleCloth> recs(INFL_SLOTS);
            std::memset(recs.data(), 0xCD, recs.size() * sizeof(PxParticleCloth));
            for (uint32_t k = 0; k < uint32_t(c.infl.size()); ++k)
            {
                PxParticleCloth r;
                r.startVertexIndex = c.infl[k].startVertexIndex;
                r.numVertices = c.infl[k].numVertices;
                r.clothBlendScale = c.infl[k].clothBlendScale;
                r.restVolume = c.infl[k].restVolume;
                r.pressure = c.infl[k].pressure;
                r.startTriangleIndex = c.infl[k].startTriangleIndex;
                r.numTriangles = c.infl[k].numTriangles;
                recs[k] = r;
            }
            c.bCloths = makeDev(dev, recs);

            std::vector<uint32_t> lam(INFL_SLOTS + 4u, POISON_U32);
            c.bLambda = makeDev(dev, lam);

            PxgParticleClothSimBuffer cb;
            std::memset(&cb, 0xEE, sizeof(cb));
            cb.mTriangles = reinterpret_cast<PxU32*>(size_t(c.bTris.addr));
            cb.mCloths = reinterpret_cast<PxParticleCloth*>(size_t(c.bCloths.addr));
            cb.mInflatableLambda = reinterpret_cast<PxReal*>(size_t(c.bLambda.addr));
            cb.mParticleBufferIndex = c.particleBufferIndex;
            cb.mNumCloths = uint32_t(c.infl.size());
            arr[b] = cb;
        }
        s->bClothArr = makeDev(dev, arr);
    }

    std::vector<uint32_t> active = { 2u, 0u, 3u };   // NOT a permutation
    DevBuf bActive = makeDev(dev, active);

    std::vector<PxgParticleSystem> descArr;
    for (Sys* s : order)
    {
        PxgParticleSystem ps;
        std::memset(&ps, 0xEE, sizeof(ps));
        ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(s->bPos.addr));
        ps.mUnsortedToSortedMapping = reinterpret_cast<PxU32*>(size_t(s->bU2S.addr));
        ps.mParticleBufferRunsum = reinterpret_cast<PxU32*>(size_t(s->bRunsum.addr));
        ps.mClothSimBuffers = reinterpret_cast<PxgParticleClothSimBuffer*>(size_t(s->bClothArr.addr));
        ps.mNumClothBuffers = s->numClothBuffers;
        ps.mNormalArray = reinterpret_cast<float4*>(size_t(s->bNormals.addr));
        ps.mAccumDeltaP = reinterpret_cast<float4*>(size_t(s->bAccum.addr));
        ps.mCommonData.mParticleContactDistance = s->particleContactDistance;
        descArr.push_back(ps);
    }
    DevBuf bDesc = makeDev(dev, descArr);

    CalcPush pushC = { bDesc.addr, bActive.addr };
    SolvePush pushS = { bDesc.addr, bActive.addr, DT, COEFF };

    {
        Kernel k("npPsInflatable.spv", "ps_calculateInflatableVolume", {}, sizeof(CalcPush));
        k.dispatch(maxCloths, {}, pushC, maxClothBuffers, uint32_t(active.size()));
        dev->flush();
    }
    {
        Kernel k("npPsInflatable.spv", "ps_solveInflatableVolume", {}, sizeof(SolvePush));
        k.dispatch(maxCloths, {}, pushS, maxClothBuffers, uint32_t(active.size()));
        dev->flush();
    }

    // ---------------------------------------------------------------------
    // The host mirror.
    // ---------------------------------------------------------------------
    size_t mm = 0, tolMM = 0;
    size_t nLambdaZero = 0, nLambdaPos = 0, nMaxDeltaBound = 0, nSkipped = 0, nMultiPass = 0;

    for (Sys* s : order)
    {
        const bool isActive = (s != &sysD);

        std::vector<V3> refNrm(NUM_PARTICLES, v3(0, 0, 0));
        std::vector<bool> nrmTouched(NUM_PARTICLES, false);
        std::vector<bool> nrmShared(NUM_PARTICLES, false);
        std::vector<F4> refAcc(NUM_PARTICLES, F4{ 0, 0, 0, 0 });
        std::vector<bool> accTol(NUM_PARTICLES, false);
        // per cloth buffer, per inflatable slot
        std::vector<std::vector<float>> refLam(s->cloths.size());
        std::vector<std::vector<bool>> lamLive(s->cloths.size()), lamTol(s->cloths.size());
        for (size_t b = 0; b < s->cloths.size(); ++b)
        {
            refLam[b].assign(INFL_SLOTS + 4u, 0.0f);
            lamLive[b].assign(INFL_SLOTS + 4u, false);
            lamTol[b].assign(INFL_SLOTS + 4u, false);
        }

        if (isActive)
        {
            // ---- kernel 1 ----
            for (uint32_t b = 0; b < s->numClothBuffers; ++b)
            {
                ClothBuf& c = s->cloths[b];
                const uint32_t offset = s->runsum[c.particleBufferIndex];
                for (uint32_t k = 0; k < uint32_t(c.infl.size()); ++k)
                {
                    const Infl& inf = c.infl[k];
                    if (inf.pressure == 0.0f) { ++nSkipped; continue; }

                    const uint32_t numPasses = (inf.numTriangles + INF_BLOCK - 1u) / INF_BLOCK;
                    if (numPasses > 1u) ++nMultiPass;
                    const float comScale = 1.0f / (3.0f * float(inf.numTriangles));

                    // normals zeroed over the GLOBAL UNSORTED vertex range
                    const uint32_t vs = inf.startVertexIndex + offset;
                    for (uint32_t i = vs; i < vs + inf.numVertices; ++i)
                    {
                        refNrm[i] = v3(0, 0, 0);
                        nrmTouched[i] = true;
                        nrmShared[i] = false;
                    }

                    // pass 1: centre of mass
                    float com0 = 0.0f, com1 = 0.0f, com2 = 0.0f;
                    for (uint32_t pass = 0; pass < numPasses; ++pass)
                    {
                        float vx[INF_BLOCK], vy[INF_BLOCK], vz[INF_BLOCK];
                        for (uint32_t t = 0; t < INF_BLOCK; ++t)
                        {
                            const uint32_t gi = pass * INF_BLOCK + t;
                            V3 tc = v3(0, 0, 0);
                            if (gi < inf.numTriangles)
                            {
                                const uint32_t tri = inf.startTriangleIndex + gi * 3u;
                                const uint32_t a = s->unsortedToSorted[c.tris[tri + 0] + offset];
                                const uint32_t bb = s->unsortedToSorted[c.tris[tri + 1] + offset];
                                const uint32_t cc = s->unsortedToSorted[c.tris[tri + 2] + offset];
                                const V3 va = v3(s->pos[a].x, s->pos[a].y, s->pos[a].z);
                                const V3 vb = v3(s->pos[bb].x, s->pos[bb].y, s->pos[bb].z);
                                const V3 vc = v3(s->pos[cc].x, s->pos[cc].y, s->pos[cc].z);
                                tc = (va + vb + vc) * comScale;
                            }
                            vx[t] = tc.x; vy[t] = tc.y; vz[t] = tc.z;
                        }
                        com0 = com0 + blockReduceAdd(vx);
                        com1 = com1 + blockReduceAdd(vy);
                        com2 = com2 + blockReduceAdd(vz);
                    }

                    // pass 2: signed volume + normals
                    float volume = 0.0f;
                    for (uint32_t pass = 0; pass < numPasses; ++pass)
                    {
                        float sv[INF_BLOCK];
                        for (uint32_t t = 0; t < INF_BLOCK; ++t)
                        {
                            const uint32_t gi = pass * INF_BLOCK + t;
                            float signedVolume = 0.0f;
                            if (gi < inf.numTriangles)
                            {
                                const uint32_t tri = inf.startTriangleIndex + gi * 3u;
                                const uint32_t ua = c.tris[tri + 0] + offset;
                                const uint32_t ub = c.tris[tri + 1] + offset;
                                const uint32_t uc = c.tris[tri + 2] + offset;
                                const uint32_t a = s->unsortedToSorted[ua];
                                const uint32_t bb = s->unsortedToSorted[ub];
                                const uint32_t cc = s->unsortedToSorted[uc];
                                const V3 com = v3(com0, com1, com2);
                                const V3 va = v3(s->pos[a].x, s->pos[a].y, s->pos[a].z) - com;
                                const V3 vb = v3(s->pos[bb].x, s->pos[bb].y, s->pos[bb].z) - com;
                                const V3 vc = v3(s->pos[cc].x, s->pos[cc].y, s->pos[cc].z) - com;
                                V3 nor = cross3(vb - va, vc - va);
                                signedVolume = dot3(va, nor);
                                nor = normalize3(nor);
                                const uint32_t vs3[3] = { ua, ub, uc };
                                for (uint32_t j = 0; j < 3u; ++j)
                                {
                                    const uint32_t vi = vs3[j];
                                    if (inf.closed) nrmShared[vi] = true;
                                    refNrm[vi] = refNrm[vi] + nor;
                                }
                            }
                            sv[t] = signedVolume;
                        }
                        volume = volume + blockReduceAdd(sv);
                    }

                    if (std::getenv("PXG_INFL_DUMP"))
                        std::printf("  [dump] sys %p buf %u infl %u: volume=%.6g "
                                    "restVolume*pressure=%.6g numTris=%u\n",
                                    (void*)s, b, k, volume, inf.restVolume * inf.pressure,
                                    inf.numTriangles);
                    const float constraintScale = 1.0f / float(inf.numVertices);
                    const float arg = hMax(0.0f, inf.restVolume * inf.pressure - volume);
                    const float lambda = std::cbrt(arg) * constraintScale * 0.25f;
                    const float maxDelta = s->particleContactDistance * 0.1f;
                    const float stored = hMin(lambda, maxDelta);
                    refLam[b][k] = stored;
                    lamLive[b][k] = true;
                    // Bit-exact only when the cube root's argument is exactly zero.
                    lamTol[b][k] = (arg != 0.0f);
                    if (arg == 0.0f) ++nLambdaZero; else ++nLambdaPos;
                    if (stored == maxDelta && lambda > maxDelta) ++nMaxDeltaBound;
                }
            }

            // ---- kernel 2 ----
            for (uint32_t b = 0; b < s->numClothBuffers; ++b)
            {
                ClothBuf& c = s->cloths[b];
                const uint32_t offset = s->runsum[c.particleBufferIndex];
                for (uint32_t k = 0; k < uint32_t(c.infl.size()); ++k)
                {
                    const Infl& inf = c.infl[k];
                    const float lambda = lamLive[b][k] ? refLam[b][k] : 0.0f;
                    if (inf.pressure == 0.0f || lambda == 0.0f) continue;
                    const uint32_t vb = inf.startVertexIndex + offset;
                    for (uint32_t t = 0; t < inf.numVertices; ++t)
                    {
                        const uint32_t vertId = t + vb;
                        const V3 normal = normalize3(refNrm[vertId]);
                        const uint32_t ind = s->unsortedToSorted[vertId];
                        const V3 d = normal * lambda * COEFF;
                        refAcc[ind].x += d.x; refAcc[ind].y += d.y;
                        refAcc[ind].z += d.z; refAcc[ind].w += 1.0f;
                        if (lamTol[b][k] || nrmShared[vertId]) accTol[ind] = true;
                    }
                }
            }
        }

        const auto gNrm = downloadBuffer<uint32_t>(dev, s->bNormals.h, (NUM_PARTICLES + PAD) * 4u);
        const auto gAcc = downloadBuffer<uint32_t>(dev, s->bAccum.h, (NUM_PARTICLES + PAD) * 4u);

        auto cmpF = [&](float ref, uint32_t got, bool tolerant, const char* what, uint32_t idx)
        {
            if (!tolerant)
            {
                if (f2u(ref) != got)
                {
                    if (mm < 12)
                        std::printf("  sys %p %s[%u] MM: ref %.9g (0x%08x) got %.9g (0x%08x)\n",
                                    (void*)s, what, idx, ref, f2u(ref), u2f(got), got);
                    ++mm;
                }
            }
            else
            {
                const float g = u2f(got), d = std::fabs(g - ref);
                const float sc = hMax(1.0f, hMax(std::fabs(g), std::fabs(ref)));
                if (!(d <= 1e-5f * sc))
                {
                    if (tolMM < 8)
                        std::printf("  sys %p %s[%u] TOL MM: ref %.9g got %.9g\n",
                                    (void*)s, what, idx, ref, g);
                    ++tolMM;
                }
            }
        };

        for (uint32_t w = NUM_PARTICLES * 4u; w < (NUM_PARTICLES + PAD) * 4u; ++w)
        {
            if (gNrm[w] != POISON_U32) { if (mm < 8) std::printf("  normals TAIL at %u\n", w); ++mm; }
            if (gAcc[w] != POISON_U32) { if (mm < 8) std::printf("  accum TAIL at %u\n", w); ++mm; }
        }

        for (uint32_t i = 0; i < NUM_PARTICLES; ++i)
        {
            if (nrmTouched[i])
            {
                const bool tol = nrmShared[i];
                cmpF(refNrm[i].x, gNrm[i * 4u + 0u], tol, "normals.x", i);
                cmpF(refNrm[i].y, gNrm[i * 4u + 1u], tol, "normals.y", i);
                cmpF(refNrm[i].z, gNrm[i * 4u + 2u], tol, "normals.z", i);
                cmpF(0.0f, gNrm[i * 4u + 3u], false, "normals.w", i);
            }
            else
            {
                // never visited -> still poisoned
                for (uint32_t j = 0; j < 4u; ++j)
                    if (gNrm[i * 4u + j] != POISON_U32)
                    {
                        if (mm < 12) std::printf("  sys %p normals[%u].%u not poisoned\n",
                                                 (void*)s, i, j);
                        ++mm;
                    }
            }
            const bool at = accTol[i];
            cmpF(refAcc[i].x, gAcc[i * 4u + 0u], at, "accum.x", i);
            cmpF(refAcc[i].y, gAcc[i * 4u + 1u], at, "accum.y", i);
            cmpF(refAcc[i].z, gAcc[i * 4u + 2u], at, "accum.z", i);
            cmpF(refAcc[i].w, gAcc[i * 4u + 3u], false, "accum.w", i);
        }

        for (uint32_t b = 0; b < uint32_t(s->cloths.size()); ++b)
        {
            const auto gLam = downloadBuffer<uint32_t>(dev, s->cloths[b].bLambda.h, INFL_SLOTS + 4u);
            for (uint32_t k = 0; k < INFL_SLOTS + 4u; ++k)
            {
                if (b < s->numClothBuffers && k < uint32_t(s->cloths[b].infl.size()) && lamLive[b][k])
                    cmpF(refLam[b][k], gLam[k], lamTol[b][k], "lambda", k);
                else if (gLam[k] != POISON_U32)
                {
                    if (mm < 12)
                        std::printf("  sys %p buf %u lambda[%u] not poisoned: 0x%08x\n",
                                    (void*)s, b, k, gLam[k]);
                    ++mm;
                }
            }
        }
    }

    PXG_CHECK_MSG(nMultiPass > 0, "an inflatable must span more than one reduction pass");
    PXG_CHECK_MSG(nSkipped > 0, "a zero-pressure inflatable must be present");
    PXG_CHECK_MSG(nLambdaZero > 0 && nLambdaPos > 0,
                  "the cube-root argument must be BOTH zero and positive (zero %zu, pos %zu)",
                  nLambdaZero, nLambdaPos);
    PXG_CHECK_MSG(nMaxDeltaBound > 0, "the maxDelta clamp must BIND somewhere");

    PXG_CHECK_MSG(mm == 0, "the inflatable pair is bit-exact where it can be (%zu mismatches)", mm);
    PXG_CHECK_MSG(tolMM == 0, "the atomic/cbrt region matches within tolerance (%zu)", tolMM);

    for (Sys* s : order)
    {
        dev->destroyBuffer(s->bPos.h); dev->destroyBuffer(s->bU2S.h);
        dev->destroyBuffer(s->bRunsum.h); dev->destroyBuffer(s->bClothArr.h);
        dev->destroyBuffer(s->bNormals.h); dev->destroyBuffer(s->bAccum.h);
        for (auto& c : s->cloths)
        {
            dev->destroyBuffer(c.bTris.h); dev->destroyBuffer(c.bCloths.h);
            dev->destroyBuffer(c.bLambda.h);
        }
    }
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

} // namespace
