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

// testPsAerodynamics.cpp -- P9.5 fixture for ps_solveAerodynamics1Launch and
// ps_solveAerodynamics2Launch, ported as npPsAerodynamics.slang.
//
// ORACLE: host transcription, carried by generated-and-pinned offsets, whole-region
// 0xAB poison compare, and the sweep.  There is no PX_CUDA_CALLABLE original for
// either kernel body, so this is the P8.73-P8.80 arrangement, not a true oracle --
// and this file says so rather than presenting the sweep as one.
//
// THE ATOMIC PROBLEM, AND HOW THIS FIXTURE HANDLES IT.  Kernel 1 accumulates into
// mNormalArray and mDelta with FLOAT ATOMICS, once per (triangle, vertex) pair.  A
// vertex shared by k triangles therefore receives k additions IN AN ORDER THE HOST
// CANNOT PREDICT, and float addition is not associative, so a bit-exact host mirror
// is impossible for shared vertices in general.  Rather than weaken every check to
// a tolerance, the fixture SPLITS the geometry:
//
//   * cloth buffer 0 -- DISJOINT triangles, no two sharing a vertex.  Every
//     accumulator receives EXACTLY ONE addition, so the result is bit-exact
//     regardless of order, and this is the region every mutant must die in.  It is
//     compared byte for byte against a 0xAB poison image.
//   * cloth buffer 1 -- a FAN of triangles around one shared centre vertex.  Its
//     job is to exercise the fan-in that buffer 0 cannot: mDelta.w > 1 and the
//     `1/delta.w` divide in kernel 2.  Here the COUNT (.w) is still compared bit-
//     exactly -- repeatedly adding 1.0f to a small integer is exact in any order --
//     while the accumulated xyz is compared to a relative epsilon.  That epsilon is
//     the ONLY tolerance in this file and it exists because of float atomics, not
//     because the port is approximate.
//
// GEOMETRY: the standard four-system pool with a DECOY that is never active, an
// active list that is NOT a permutation, and every system's cloth ARRAY padded past
// its own mNumClothBuffers with REAL records so the per-system bound has something
// poisoned to hit on the wrong side of it.
//
// TWO ROUNDS, over two independent accumulator sets and two descriptors:
//   round A -- kernel 1 ALONE, compared against the mirror's post-kernel-1 state;
//   round B -- kernel 1 THEN kernel 2, as the host does, compared against the
//              final state.
// Round A exists because the first sweep found that kernel 2 CLEARS mDelta, which
// hid every mutant whose only effect was on the fan-in COUNT -- dropping the
// invMass term from calculateDelta, dropping the triangle guard, and adding a
// spurious .w to the normal accumulation all survived for that one reason.  With
// the accumulators observed before the clear, all three die.  Round B still tests
// the composition, including upstream candidate 15 (kernel 1 writes normals at the
// SORTED index, kernel 2 reads them at the UNSORTED one), which the mirror
// reproduces deliberately.

#include "pxgTestFramework.h"

#include "host_defines.h"
#include "PxgParticleSystem.h"
#include "PxParticleBuffer.h"
#include "PxsPBDMaterialCore.h"

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
// PxVec3::cross (PxVec3.h:280-283): positive product first, per component.
static V3 cross3(V3 a, V3 b)
{
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static float mag3(V3 v) { return std::sqrt(dot3(v, v)); }
// PxVec3::getNormalized(): guards magnitudeSquared() > 0, ZERO fallback.
static V3 normalized3(V3 v)
{
    const float m = dot3(v, v);
    return (m > 0.0f) ? v * (1.0f / std::sqrt(m)) : v3(0.0f, 0.0f, 0.0f);
}
// SafeNormalize(v, fallback): same guard, CALLER's fallback.
static V3 safeNormalize3(V3 v, V3 fallback)
{
    const float l = dot3(v, v);
    return (l > 0.0f) ? v * (1.0f / std::sqrt(l)) : fallback;
}
// PxMax/PxMin for float are ternaries, and PxClamp is PxMin(hi, PxMax(lo, v)).
static float hMax(float a, float b) { return a > b ? a : b; }
static float hMin(float a, float b) { return a < b ? a : b; }
static float hClamp(float v, float lo, float hi) { return hMin(hi, hMax(lo, v)); }

struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };

template <typename T>
DevBuf makeDev(PxgSlangComputeDevice* dev, const std::vector<T>& data)
{
    DevBuf b;
    b.h = createBufferFrom(dev, data);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

struct AeroPush { uint64_t particleSystems, activeParticleSystems; float dt; uint32_t pad; };
static_assert(sizeof(AeroPush) == 24, "the {8,8,4} blob rounds to 24 bytes");

static const uint32_t POISON_U32 = 0xABABABABu;
static const uint32_t BLOCK = 256u;   // PxgParticleSystemKernelBlockDim::PS_SOLVE

// PxsPBDMaterialData float indices used here.
static const uint32_t MAT_F_LIFT = 10u;
static const uint32_t MAT_F_DRAG = 11u;

// ---------------------------------------------------------------------------
// Layout proof -- every constant npPsAerodynamics.slang hard-codes, pinned
// against the real engine headers.
// ---------------------------------------------------------------------------
PXG_TEST(psAerodynamicsLayout)
{
    PXG_CHECK_MSG(sizeof(PxgParticleSimBuffer) == 88,
                  "sizeof(PxgParticleSimBuffer) == 88 (got %zu)", sizeof(PxgParticleSimBuffer));
    PXG_CHECK(offsetof(PxgParticleSimBuffer, mPhases) == 24);
    PXG_CHECK(offsetof(PxgParticleSimBuffer, mNumActiveParticles) == 56);

    PXG_CHECK_MSG(sizeof(PxgParticleClothSimBuffer) == 112,
                  "sizeof(PxgParticleClothSimBuffer) == 112 (got %zu)",
                  sizeof(PxgParticleClothSimBuffer));
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mTriangles) == 32);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mParticleBufferIndex) == 88);
    PXG_CHECK(offsetof(PxgParticleClothSimBuffer, mNumTriangles) == 104);

    // the descriptor fields the two kernels walk
    PXG_CHECK(offsetof(PxgParticleSystem, mSortedPhaseArray) == 40);
    PXG_CHECK(offsetof(PxgParticleSystem, mParticleMaterials) == 80);
    PXG_CHECK(offsetof(PxgParticleSystem, mDelta) == 368);
    PXG_CHECK(offsetof(PxgParticleSystem, mNormalArray) == 384);
    PXG_CHECK(offsetof(PxgParticleSystem, mParticleSimBuffers) == 400);
    PXG_CHECK(offsetof(PxgParticleSystem, mParticleBufferRunsum) == 408);
    PXG_CHECK(offsetof(PxgParticleSystem, mClothSimBuffers) == 432);
    PXG_CHECK(offsetof(PxgParticleSystem, mNumClothBuffers) == 464);
    PXG_CHECK(offsetof(PxgParticleSystem, mParticleMaterialStride) == 632);
    PXG_CHECK(offsetof(PxgParticleSystem, mPhaseGroupToMaterialHandle) == 640);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mUnsortedToSortedMapping) == 48);
    PXG_CHECK(offsetof(PxGpuParticleSystem, mSortedPositions_InvMass) == 24);

    // mWind is a PxVec3 nested in mData; the shader reads it as three floats.
    PXG_CHECK_MSG(offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mWind) == 328,
                  "mData.mWind must sit at 328 (got %zu)",
                  offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mWind));

    // the two material coefficients, by float index
    PXG_CHECK_MSG(offsetof(PxsPBDMaterialData, lift) == MAT_F_LIFT * 4u,
                  "lift must be float index %u (got byte %zu)", MAT_F_LIFT,
                  offsetof(PxsPBDMaterialData, lift));
    PXG_CHECK_MSG(offsetof(PxsPBDMaterialData, drag) == MAT_F_DRAG * 4u,
                  "drag must be float index %u (got byte %zu)", MAT_F_DRAG,
                  offsetof(PxsPBDMaterialData, drag));
    PXG_CHECK_MSG(sizeof(PxsPBDMaterialData) == 60,
                  "sizeof(PxsPBDMaterialData) == 60 (got %zu)", sizeof(PxsPBDMaterialData));
}

// ---------------------------------------------------------------------------

struct ClothBuf
{
    uint32_t numTriangles = 0;
    uint32_t particleBufferIndex = 0;
    bool sharing = false;               // true -> the fan, tolerance region
    std::vector<uint32_t> tris;         // 3 per triangle, buffer-LOCAL indices
    DevBuf bTris;
};

struct Sys
{
    uint32_t numClothBuffers = 0;
    std::vector<ClothBuf> cloths;
    std::vector<F4> pos, vel;           // SORTED arrays
    std::vector<uint32_t> sortedPhase;  // SORTED
    // ONE PARTICLE BUFFER PER CLOTH BUFFER, as the engine has it: every
    // PxParticleClothBuffer IS a particle buffer, so cloth buffer b maps to
    // particle buffer b at runsum offset b*40.  That matters for correctness of
    // this fixture, not just realism -- kernel 2's grid.Y runs one block per cloth
    // buffer CONCURRENTLY, and if two of them shared a particle buffer they would
    // race on sortedVel and on the mDelta clear.  The engine never has that; a
    // fixture that did would be testing a hazard the port cannot be responsible for.
    std::vector<std::vector<uint32_t>> unsortedPhase;  // [buffer][local]
    std::vector<uint32_t> numActivePerBuf;
    std::vector<uint32_t> unsortedToSorted, runsum;
    V3 wind = v3(0, 0, 0);
    // TWO accumulator sets: A is written by kernel 1 ALONE, B by the pair.
    DevBuf bPos, bVel, bSortedPhase, bU2S, bRunsum, bClothArr, bPartArr;
    DevBuf bNormalsA, bDeltaA, bNormalsB, bDeltaB, bVelB;
    std::vector<DevBuf> bUnsortedPhase;
};

static uint32_t hashu(uint32_t k, uint32_t salt)
{
    uint32_t h = k * 2654435761u + salt * 40503u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return h;
}

PXG_TEST(psAerodynamics)
{
    PxgSlangComputeDevice* dev = testDevice();

    // The array is deliberately LARGE: both grids must span more than one block or
    // every block-decomposition mutant (halving the block dim, dropping the
    // blockIdx.x term) is an identity -- the lesson P9.1a's psm-block-dim-128 cost
    // once already.  With PS_SOLVE = 256 that needs > 256 triangles in some cloth
    // buffer AND > 256 active particles in some particle buffer.
    const uint32_t NUM_PARTICLES = 900u;  // per system, the SORTED array size
    const uint32_t PAD = 32u;             // poisoned tail on every output array
    const uint32_t CLOTH_ARRAY_SLOTS = 3u;
    // Every triangle array is allocated well past its live triangle count and the
    // tail filled with a DISTINCTIVE vertex, so a dropped triangle guard reads
    // something real and observable rather than falling off the allocation.
    const uint32_t TRI_SLOTS = 520u;
    // Three DISTINCT, in-range, non-collinear pad vertices.  Three copies of one
    // vertex would make every pad triangle zero-area, so a dropped triangle guard
    // would move only the fan-in count and nothing else.
    const uint32_t TRI_PAD_V0 = 271u, TRI_PAD_V1 = 277u, TRI_PAD_V2 = 283u;
    const float DT = 1.0f / 90.0f;        // not 1, so dt is observable

    // Two materials with DIFFERENT lift/drag, plus one with BOTH ZERO so the
    // calculateDelta gate (trap Y6) has a case on each side.  Material 2's drag is
    // zero but its lift is NOT -- that is the case a mutant spelling the gate as
    // (drag != 0 && lift != 0) dies on.
    const uint32_t NUM_MATS = 5u;
    // The RUNTIME stride, deliberately NOT sizeof(PxsPBDMaterialData) == 60: the
    // engine walks materials by a byte stride carried in the descriptor (the
    // largest material type's size), so a port that hard-codes 60 must die.  With
    // a 60-byte stride in the fixture it would be an identity.
    const uint32_t MAT_FLOATS = 16u;          // 64-byte stride
    const uint32_t MAT_STRIDE = MAT_FLOATS * 4u;
    std::vector<float> mats(NUM_MATS * MAT_FLOATS, 0.0f);
    auto setMat = [&](uint32_t m, float lift, float drag)
    {
        mats[m * MAT_FLOATS + MAT_F_LIFT] = lift;
        mats[m * MAT_FLOATS + MAT_F_DRAG] = drag;
    };
    setMat(0, 0.0f, 0.0f);      // both zero -> gate closed
    setMat(1, 2.5f, 1.75f);     // both non-zero
    setMat(2, 3.25f, 0.0f);     // lift only  -> gate OPEN (kills the && mutant)
    setMat(3, 0.0f, 1.25f);     // drag only  -> gate OPEN
    // A drag so large that dt*drag*cosTheta*invMass runs past 0.75 and the clamp
    // actually BINDS.  Without it the drag factor peaks around 0.035 and the
    // clamp bounds are never exercised -- the first sweep's y8b survivor.
    setMat(4, 0.5f, 900.0f);
    DevBuf bMats = makeDev(dev, mats);

    // phaseGroupToMaterialHandle: PxU16 per group, packed two per word.  Group g
    // maps to material (g + 1) % NUM_MATS, so group != material anywhere.
    //
    // THE TABLE IS DELIBERATELY HUGE.  PxGetGroup masks the phase with
    // 0x000fffff (PxParticleGpu.h:64) because bits 20+ are BEHAVIOUR FLAGS, and
    // the phases below really do carry them -- as the live scene's do.  For the
    // mask to be under test, an unmasked read must land somewhere REAL and
    // DIFFERENT rather than faulting, so the table is sized past the largest
    // unmasked index and every entry beyond the live groups maps to material 0,
    // whose lift and drag are both zero.  Drop the mask and the gate closes.
    const uint32_t NUM_GROUPS = 8u;
    const uint32_t PHASE_FLAGS = (1u << 20) | (1u << 21);   // SelfCollide | Filter
    const uint32_t TABLE_WORDS = (((NUM_GROUPS + PHASE_FLAGS) >> 1) + 2u);
    std::vector<uint32_t> phaseToMat(TABLE_WORDS, 0u);   // material 0 = lift/drag 0
    for (uint32_t g = 0; g < NUM_GROUPS; ++g)
    {
        const uint32_t m = (g + 1u) % NUM_MATS;
        phaseToMat[g >> 1] |= (m & 0xffffu) << ((g & 1u) * 16u);
    }
    DevBuf bPhaseToMat = makeDev(dev, phaseToMat);

    auto buildSys = [&](uint32_t salt, uint32_t numActive0, V3 wind,
                        std::vector<std::pair<uint32_t, bool>> triCountsAndSharing)
    {
        Sys s;
        s.numClothBuffers = uint32_t(triCountsAndSharing.size());
        while (triCountsAndSharing.size() < CLOTH_ARRAY_SLOTS)
            triCountsAndSharing.push_back({ 2u + uint32_t(triCountsAndSharing.size()), false });
        s.wind = wind;

        s.pos.resize(NUM_PARTICLES);
        s.vel.resize(NUM_PARTICLES);
        s.sortedPhase.resize(NUM_PARTICLES);
        for (uint32_t i = 0; i < NUM_PARTICLES; ++i)
        {
            // Positions spread in all three axes so no cross product is degenerate
            // by accident, and a couple of DEGENERATE triangles are planted below.
            s.pos[i] = { float(salt) * 0.5f + float(i) * 0.37f,
                         float((i * 7u) % 11u) * 0.29f,
                         float((i * 13u) % 17u) * 0.23f,
                         // trap Y6: a FIXED particle (invMass 0) every 8th slot
                         (i % 8u == 3u) ? 0.0f : (0.75f + float(i) * 0.011f) };
            // .w is DELIBERATELY NON-ZERO: kernel 2 stores a literal 0 there, and
            // with a zero already in place that store is invisible (the first
            // sweep's y11b survivor).
            s.vel[i] = { float((i * 5u) % 9u) * 0.4f - 1.3f,
                         float(salt) * 0.25f - float(i) * 0.06f,
                         float((i * 3u) % 7u) * 0.31f,
                         7.5f + float(i) * 0.01f };
            // SORTED phase: group varies so several materials are hit.
            // Flags on a SUBSET, so both flagged and unflagged phases exist.
            s.sortedPhase[i] = ((i + salt) % NUM_GROUPS)
                             | ((i % 3u == 0u) ? PHASE_FLAGS : 0u);
        }
        // A REAL permutation, so sorted and unsorted indices never coincide --
        // which is what makes upstream candidate 15 (trap Y12) observable.
        s.unsortedToSorted.resize(NUM_PARTICLES);
        for (uint32_t i = 0; i < NUM_PARTICLES; ++i)
            s.unsortedToSorted[i] = (i * 37u + 11u) % NUM_PARTICLES;
        s.runsum = { 0u, 300u, 600u };

        // UNSORTED phases, buffer-local, DELIBERATELY DIFFERENT from the sorted
        // array -- trap Y10.  If a port reads the wrong one it lands on a
        // different material and dies.  One array PER particle buffer, and the
        // counts differ per buffer so a bound mix-up is visible.
        s.unsortedPhase.resize(CLOTH_ARRAY_SLOTS);
        s.numActivePerBuf.resize(CLOTH_ARRAY_SLOTS);
        for (uint32_t b = 0; b < CLOTH_ARRAY_SLOTS; ++b)
        {
            s.numActivePerBuf[b] = numActive0 - b;   // distinct per buffer
            s.unsortedPhase[b].resize(NUM_PARTICLES);
            for (uint32_t i = 0; i < NUM_PARTICLES; ++i)
                s.unsortedPhase[b][i] = ((i * 3u + salt + 2u + b * 5u) % NUM_GROUPS)
                                      | ((i % 4u == 1u) ? PHASE_FLAGS : 0u);
        }

        uint32_t bIdx = 0;
        for (auto& pr : triCountsAndSharing)
        {
            ClothBuf c;
            c.numTriangles = pr.first;
            c.sharing = pr.second;
            // NOT its own slot: cloth slot b maps to particle buffer (b+1)%N.
            // With the identity mapping the whole mParticleBufferIndex
            // indirection -- which drives the runsum offset AND
            // mParticleSimBuffers -- is untested.  Kept a BIJECTION so no two
            // cloth slots share a particle buffer (that would race).
            c.particleBufferIndex = (bIdx + 1u) % CLOTH_ARRAY_SLOTS;
            ++bIdx;
            c.tris.resize(size_t(TRI_SLOTS) * 3u);
            for (uint32_t t = 0; t < TRI_SLOTS; ++t)
            {
                c.tris[t * 3u + 0u] = TRI_PAD_V0;
                c.tris[t * 3u + 1u] = TRI_PAD_V1;
                c.tris[t * 3u + 2u] = TRI_PAD_V2;
            }
            if (!c.sharing)
            {
                // DISJOINT: triangle t uses vertices 3t, 3t+1, 3t+2 -- no sharing,
                // so every accumulator gets exactly one add and the compare is
                // bit-exact.  Kept inside [0, 39] since runsum[0] == 0 and the
                // permutation is over NUM_PARTICLES.
                for (uint32_t t = 0; t < c.numTriangles; ++t)
                {
                    c.tris[t * 3u + 0u] = 3u * t + 0u;
                    c.tris[t * 3u + 1u] = 3u * t + 1u;
                    c.tris[t * 3u + 2u] = 3u * t + 2u;
                }
                // One DEGENERATE triangle (two identical vertices) so the zero-area
                // cross product exercises getNormalized's zero guard (trap Y9a).
                if (c.numTriangles >= 2u)
                {
                    c.tris[3u] = 5u; c.tris[4u] = 5u; c.tris[5u] = 9u;
                }
            }
            else
            {
                // A FAN around centre vertex 0: triangle t = (0, t+1, t+2).  Vertex
                // 0 receives one contribution per triangle, so mDelta.w reaches
                // numTriangles there -- the fan-in kernel 2 divides by.
                for (uint32_t t = 0; t < c.numTriangles; ++t)
                {
                    c.tris[t * 3u + 0u] = 0u;
                    c.tris[t * 3u + 1u] = (t + 1u) % 288u;
                    c.tris[t * 3u + 2u] = (t + 2u) % 288u;
                }
            }
            s.cloths.push_back(c);
        }
        return s;
    };

    // A: buffer 0 disjoint (7 triangles), buffer 1 the fan (5 triangles).
    // DECOY: never active.  B: one disjoint buffer.  C: a fan only.
    // The wind differs per system so a system-index error lands on other numbers.
    // Buffer 1 of A and C's only buffer carry 300 triangles, so gridTri == 2 and a
    // block-stride mutant duplicates work instead of being an identity.  Every
    // system's particle buffers hold ~290 active particles, so gridPart == 2 too.
    Sys sysA = buildSys(100u, 290u, v3(1.7f, -0.4f, 0.9f), { { 90u, false }, { 300u, true } });
    Sys sysD = buildSys(200u, 260u, v3(-9.0f, 9.0f, -9.0f), { { 5u, false }, { 5u, false }, { 5u, false } });
    Sys sysB = buildSys(300u, 285u, v3(0.0f, 0.0f, 0.0f), { { 70u, false } });   // ZERO wind
    Sys sysC = buildSys(400u, 288u, v3(-2.2f, 1.1f, 3.3f), { { 300u, true } });

    Sys* order[4] = { &sysA, &sysD, &sysB, &sysC };

    uint32_t maxTriangles = 0, maxClothBuffers = 0, maxParticles = 0;
    for (Sys* s : order)
    {
        maxClothBuffers = PxMax(maxClothBuffers, uint32_t(s->cloths.size()));
        for (uint32_t n : s->numActivePerBuf) maxParticles = PxMax(maxParticles, n);
        for (auto& c : s->cloths) maxTriangles = PxMax(maxTriangles, c.numTriangles);
    }
    PXG_CHECK_MSG(sysB.numClothBuffers < CLOTH_ARRAY_SLOTS,
                  "B must have real cloth records past its own bound");

    for (Sys* s : order)
    {
        s->bPos = makeDev(dev, s->pos);
        s->bVel = makeDev(dev, s->vel);
        s->bSortedPhase = makeDev(dev, s->sortedPhase);
        s->bU2S = makeDev(dev, s->unsortedToSorted);
        s->bRunsum = makeDev(dev, s->runsum);
        s->bUnsortedPhase.resize(CLOTH_ARRAY_SLOTS);
        for (uint32_t b = 0; b < CLOTH_ARRAY_SLOTS; ++b)
            s->bUnsortedPhase[b] = makeDev(dev, s->unsortedPhase[b]);

        // The two accumulators start ZEROED over the live range (the engine clears
        // mDelta in ps_solveVelocityLaunch and mNormalArray is scratch), with a
        // POISONED tail so an overrun past the particle count is caught.
        std::vector<uint32_t> nrm(size_t(NUM_PARTICLES + PAD) * 4u, POISON_U32);
        std::vector<uint32_t> dlt(size_t(NUM_PARTICLES + PAD) * 4u, POISON_U32);
        for (uint32_t i = 0; i < NUM_PARTICLES * 4u; ++i) { nrm[i] = f2u(0.0f); dlt[i] = f2u(0.0f); }
        s->bNormalsA = makeDev(dev, nrm);
        s->bDeltaA = makeDev(dev, dlt);
        s->bNormalsB = makeDev(dev, nrm);
        s->bDeltaB = makeDev(dev, dlt);
        // Round B needs its OWN velocity array -- kernel 2 writes into it, and
        // round A must be able to read the untouched original.
        std::vector<F4> velB = s->vel;
        s->bVelB = makeDev(dev, velB);

        // one particle sim buffer PER CLOTH BUFFER, each with its own phases and
        // its own active count
        std::vector<PxgParticleSimBuffer> parr(CLOTH_ARRAY_SLOTS);
        std::memset(parr.data(), 0xEE, parr.size() * sizeof(PxgParticleSimBuffer));
        for (uint32_t b = 0; b < CLOTH_ARRAY_SLOTS; ++b)
        {
            parr[b].mPhases = reinterpret_cast<PxU32*>(size_t(s->bUnsortedPhase[b].addr));
            parr[b].mNumActiveParticles = s->numActivePerBuf[b];
        }
        s->bPartArr = makeDev(dev, parr);

        std::vector<PxgParticleClothSimBuffer> arr(s->cloths.size());
        for (uint32_t b = 0; b < uint32_t(s->cloths.size()); ++b)
        {
            ClothBuf& c = s->cloths[b];
            c.bTris = makeDev(dev, c.tris);
            PxgParticleClothSimBuffer cb;
            std::memset(&cb, 0xEE, sizeof(cb));
            cb.mTriangles = reinterpret_cast<PxU32*>(size_t(c.bTris.addr));
            cb.mParticleBufferIndex = c.particleBufferIndex;
            cb.mNumTriangles = c.numTriangles;
            arr[b] = cb;
        }
        s->bClothArr = makeDev(dev, arr);
    }

    // NOT a permutation: index 1 (the decoy) is never named.
    std::vector<uint32_t> active = { 2u, 0u, 3u };
    DevBuf bActive = makeDev(dev, active);

    std::vector<PxgParticleSystem> descArrA, descArrB;
    for (Sys* s : order)
    {
        PxgParticleSystem ps;
        std::memset(&ps, 0xEE, sizeof(ps));
        ps.mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(s->bPos.addr));
        ps.mSortedVelocities = reinterpret_cast<float4*>(size_t(s->bVel.addr));
        ps.mSortedPhaseArray = reinterpret_cast<PxU32*>(size_t(s->bSortedPhase.addr));
        ps.mUnsortedToSortedMapping = reinterpret_cast<PxU32*>(size_t(s->bU2S.addr));
        ps.mParticleBufferRunsum = reinterpret_cast<PxU32*>(size_t(s->bRunsum.addr));
        ps.mClothSimBuffers = reinterpret_cast<PxgParticleClothSimBuffer*>(size_t(s->bClothArr.addr));
        ps.mParticleSimBuffers = reinterpret_cast<PxgParticleSimBuffer*>(size_t(s->bPartArr.addr));
        ps.mNumClothBuffers = s->numClothBuffers;
        ps.mNormalArray = reinterpret_cast<float4*>(size_t(s->bNormalsA.addr));
        ps.mDelta = reinterpret_cast<float4*>(size_t(s->bDeltaA.addr));
        ps.mParticleMaterials = reinterpret_cast<PxsParticleMaterialData*>(size_t(bMats.addr));
        ps.mParticleMaterialStride = MAT_STRIDE;
        ps.mPhaseGroupToMaterialHandle = reinterpret_cast<PxU16*>(size_t(bPhaseToMat.addr));
        ps.mData.mWind = PxVec3(s->wind.x, s->wind.y, s->wind.z);
        descArrA.push_back(ps);
        // Round B: the same descriptor with the OTHER accumulator set and its own
        // velocity array.
        ps.mNormalArray = reinterpret_cast<float4*>(size_t(s->bNormalsB.addr));
        ps.mDelta = reinterpret_cast<float4*>(size_t(s->bDeltaB.addr));
        ps.mSortedVelocities = reinterpret_cast<float4*>(size_t(s->bVelB.addr));
        descArrB.push_back(ps);
    }
    DevBuf bDescA = makeDev(dev, descArrA);
    DevBuf bDescB = makeDev(dev, descArrB);

    AeroPush pushA = { bDescA.addr, bActive.addr, DT, 0u };
    AeroPush pushB = { bDescB.addr, bActive.addr, DT, 0u };

    // The two real grids.  Note kernel 2's Y extent is the PARTICLE buffer max in
    // the engine; here every system has one particle buffer, so the faithful value
    // is 1 -- but the cloth guard is against mNumClothBuffers, so passing 1 would
    // silently skip cloth buffer 1.  The engine has the same shape (maxBuffers >=
    // maxClothBuffers always holds there because every cloth buffer IS a particle
    // buffer); the fixture uses maxClothBuffers so the guard is the thing under
    // test rather than the grid.
    const uint32_t gridTri = (maxTriangles + BLOCK - 1u) / BLOCK;
    const uint32_t gridPart = (maxParticles + BLOCK - 1u) / BLOCK;
    {   // round A: kernel 1 ALONE, so the raw accumulators are observable
        Kernel k("npPsAerodynamics.spv", "ps_solveAerodynamics1Launch", {}, sizeof(AeroPush));
        k.dispatch(gridTri, {}, pushA, maxClothBuffers, uint32_t(active.size()));
        dev->flush();
    }
    {   // round B: the pair, in the host's order
        Kernel k("npPsAerodynamics.spv", "ps_solveAerodynamics1Launch", {}, sizeof(AeroPush));
        k.dispatch(gridTri, {}, pushB, maxClothBuffers, uint32_t(active.size()));
        dev->flush();
    }
    {
        Kernel k("npPsAerodynamics.spv", "ps_solveAerodynamics2Launch", {}, sizeof(AeroPush));
        k.dispatch(gridPart, {}, pushB, maxClothBuffers, uint32_t(active.size()));
        dev->flush();
    }

    // ---------------------------------------------------------------------
    // The host mirror, replaying both kernels in the same order.
    // ---------------------------------------------------------------------
    size_t mm = 0, tolMM = 0;
    size_t nFanIn = 0, nGateOpen = 0, nGateClosed = 0, nDegenerate = 0, nFixed = 0;

    for (Sys* s : order)
    {
        const bool isActive = (s != &sysD);

        std::vector<V3> refNrm(NUM_PARTICLES, v3(0, 0, 0));
        std::vector<V3> refDlt(NUM_PARTICLES, v3(0, 0, 0));
        std::vector<float> refCnt(NUM_PARTICLES, 0.0f);
        std::vector<F4> refVel = s->vel;
        std::vector<bool> shared(NUM_PARTICLES, false);
        std::vector<V3> snapNrm(NUM_PARTICLES, v3(0, 0, 0));
        std::vector<V3> snapDlt(NUM_PARTICLES, v3(0, 0, 0));
        std::vector<float> snapCnt(NUM_PARTICLES, 0.0f);

        auto matFor = [&](uint32_t group, uint32_t fi)
        {
            const uint32_t m = (group + 1u) % NUM_MATS;
            return mats[m * MAT_FLOATS + fi];
        };

        if (isActive)
        {
            // ---- kernel 1 ----
            for (uint32_t b = 0; b < s->numClothBuffers; ++b)
            {
                ClothBuf& c = s->cloths[b];
                const uint32_t offset = s->runsum[c.particleBufferIndex];
                for (uint32_t t = 0; t < c.numTriangles; ++t)
                {
                    const uint32_t triBase = t * 3u;
                    for (uint32_t i = 0; i < 3u; ++i)
                    {
                        const uint32_t ind0 = c.tris[triBase + (i % 3u)] + offset;
                        const uint32_t ind1 = c.tris[triBase + ((i + 1u) % 3u)] + offset;
                        const uint32_t ind2 = c.tris[triBase + ((i + 2u) % 3u)] + offset;

                        const uint32_t a = s->unsortedToSorted[ind0];
                        const uint32_t bb = s->unsortedToSorted[ind1];
                        const uint32_t cc = s->unsortedToSorted[ind2];

                        const F4 p14 = s->pos[a];
                        const V3 p1 = v3(p14.x, p14.y, p14.z);
                        const V3 va = v3(s->vel[a].x, s->vel[a].y, s->vel[a].z);

                        const uint32_t group = s->sortedPhase[a] & 0x000fffffu;
                        const float lift = matFor(group, MAT_F_LIFT);
                        const float drag = matFor(group, MAT_F_DRAG);
                        const bool calculateDelta = ((drag != 0.0f) || (lift != 0.0f))
                                                 && (p14.w != 0.0f);
                        if (calculateDelta) ++nGateOpen; else ++nGateClosed;
                        if (p14.w == 0.0f) ++nFixed;

                        const V3 p2 = v3(s->pos[bb].x, s->pos[bb].y, s->pos[bb].z);
                        const V3 p3 = v3(s->pos[cc].x, s->pos[cc].y, s->pos[cc].z);
                        const V3 vb = v3(s->vel[bb].x, s->vel[bb].y, s->vel[bb].z);
                        const V3 vc = v3(s->vel[cc].x, s->vel[cc].y, s->vel[cc].z);

                        const V3 e1 = p2 - p1;
                        const V3 e2 = p3 - p1;
                        const V3 rawN = cross3(e2, e1);
                        if (dot3(rawN, rawN) == 0.0f) ++nDegenerate;
                        V3 n = normalized3(rawN);

                        if (refCnt[a] > 0.0f || dot3(refNrm[a], refNrm[a]) != 0.0f) shared[a] = true;
                        refNrm[a] = refNrm[a] + n;

                        if (calculateDelta)
                        {
                            const V3 v = (va + vb + vc) * 0.3333f;
                            const V3 vrel = v3(s->wind.x - v.x, s->wind.y - v.y, s->wind.z - v.z);
                            const V3 vdir = normalized3(vrel);
                            float cosTheta = dot3(vdir, n);
                            if (cosTheta < 0.0f) { n = n * -1.0f; cosTheta = cosTheta * -1.0f; }

                            const float dragFactor =
                                hClamp(DT * drag * cosTheta * p14.w, -0.75f, 0.75f);
                            const V3 dragV = vrel * dragFactor;
                            const float liftCoefficient =
                                hClamp(p14.w * DT * lift * cosTheta
                                           * std::sqrt(hMax(0.0f, 1.0f - (cosTheta * cosTheta))),
                                       -1.0f, 1.0f) * mag3(vrel);
                            const V3 liftDir = normalized3(cross3(cross3(vrel, n), vrel));
                            const V3 j = dragV + liftDir * liftCoefficient;

                            refDlt[a] = refDlt[a] + j;
                            refCnt[a] += 1.0f;
                            if (refCnt[a] > 1.0f) ++nFanIn;
                        }
                    }
                }
            }

            // Snapshot: what round A must show, before kernel 2 touches anything.
            snapNrm = refNrm; snapDlt = refDlt; snapCnt = refCnt;

            // ---- kernel 2 ----
            for (uint32_t b = 0; b < s->numClothBuffers; ++b)
            {
                ClothBuf& c = s->cloths[b];
                const uint32_t offset = s->runsum[c.particleBufferIndex];
                for (uint32_t g = 0; g < s->numActivePerBuf[c.particleBufferIndex]; ++g)
                {
                    const uint32_t unsortedIndex = g + offset;
                    const uint32_t group =
                        s->unsortedPhase[c.particleBufferIndex][g] & 0x000fffffu;   // Y10
                    const float lift = matFor(group, MAT_F_LIFT);
                    const float drag = matFor(group, MAT_F_DRAG);
                    const uint32_t index = s->unsortedToSorted[unsortedIndex];
                    const F4 p14 = s->pos[index];
                    const bool calculateDelta = ((drag != 0.0f) || (lift != 0.0f))
                                             && (p14.w != 0.0f);
                    if (calculateDelta && refCnt[index] > 0.0f)
                    {
                        const float scale = 1.0f / refCnt[index];
                        refVel[index].x += refDlt[index].x * scale;
                        refVel[index].y += refDlt[index].y * scale;
                        refVel[index].z += refDlt[index].z * scale;
                        refVel[index].w = 0.0f;
                    }
                    refDlt[index] = v3(0, 0, 0);
                    refCnt[index] = 0.0f;
                    // Y12: the UNSORTED index, exactly as the CUDA spells it.
                    const V3 nn = refNrm[unsortedIndex];
                    const V3 un = safeNormalize3(nn, v3(0.0f, 1.0f, 0.0f));
                    refNrm[unsortedIndex] = v3(-un.x, -un.y, -un.z);
                }
            }
        }

        const auto gNrmA = downloadBuffer<uint32_t>(dev, s->bNormalsA.h, (NUM_PARTICLES + PAD) * 4u);
        const auto gDltA = downloadBuffer<uint32_t>(dev, s->bDeltaA.h, (NUM_PARTICLES + PAD) * 4u);
        const auto gNrm = downloadBuffer<uint32_t>(dev, s->bNormalsB.h, (NUM_PARTICLES + PAD) * 4u);
        const auto gDlt = downloadBuffer<uint32_t>(dev, s->bDeltaB.h, (NUM_PARTICLES + PAD) * 4u);
        const auto gVel = downloadBuffer<uint32_t>(dev, s->bVelB.h, NUM_PARTICLES * 4u);
        // Round A must not have touched the ORIGINAL velocity array at all.
        const auto gVelA = downloadBuffer<uint32_t>(dev, s->bVel.h, NUM_PARTICLES * 4u);

        // The poisoned tails must survive on both accumulators.
        for (uint32_t w = NUM_PARTICLES * 4u; w < (NUM_PARTICLES + PAD) * 4u; ++w)
        {
            if (gNrm[w] != POISON_U32) { if (mm < 8) std::printf("  normals TAIL clobbered at %u\n", w); ++mm; }
            if (gDlt[w] != POISON_U32) { if (mm < 8) std::printf("  delta TAIL clobbered at %u\n", w); ++mm; }
            if (gNrmA[w] != POISON_U32) { if (mm < 8) std::printf("  normalsA TAIL clobbered at %u\n", w); ++mm; }
            if (gDltA[w] != POISON_U32) { if (mm < 8) std::printf("  deltaA TAIL clobbered at %u\n", w); ++mm; }
        }

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
                const float g = u2f(got);
                const float d = std::fabs(g - ref);
                const float scale = hMax(1.0f, hMax(std::fabs(g), std::fabs(ref)));
                if (!(d <= 1e-5f * scale))
                {
                    if (tolMM < 8)
                        std::printf("  sys %p %s[%u] TOL MM: ref %.9g got %.9g\n",
                                    (void*)s, what, idx, ref, g);
                    ++tolMM;
                }
            }
        };

        for (uint32_t i = 0; i < NUM_PARTICLES; ++i)
        {
            const bool tol = shared[i];
            // --- round A: kernel 1 alone, BEFORE kernel 2 clears anything ---
            cmpF(snapNrm[i].x, gNrmA[i * 4u + 0u], tol, "A.normals.x", i);
            cmpF(snapNrm[i].y, gNrmA[i * 4u + 1u], tol, "A.normals.y", i);
            cmpF(snapNrm[i].z, gNrmA[i * 4u + 2u], tol, "A.normals.z", i);
            // normals.w must STILL BE ZERO: kernel 1 uses the THREE-argument
            // atomic and must never touch it (trap Y4).
            cmpF(0.0f, gNrmA[i * 4u + 3u], false, "A.normals.w", i);
            cmpF(snapDlt[i].x, gDltA[i * 4u + 0u], tol, "A.delta.x", i);
            cmpF(snapDlt[i].y, gDltA[i * 4u + 1u], tol, "A.delta.y", i);
            cmpF(snapDlt[i].z, gDltA[i * 4u + 2u], tol, "A.delta.z", i);
            // The fan-in COUNT, uncleared -- this is what makes a count-only
            // mutant observable at all.
            cmpF(snapCnt[i], gDltA[i * 4u + 3u], false, "A.delta.w(count)", i);
            // round A ran kernel 1 only, so the velocities must be untouched
            cmpF(s->vel[i].x, gVelA[i * 4u + 0u], false, "A.vel.x", i);
            cmpF(s->vel[i].w, gVelA[i * 4u + 3u], false, "A.vel.w", i);

            // --- round B: the pair ---
            cmpF(refNrm[i].x, gNrm[i * 4u + 0u], tol, "normals.x", i);
            cmpF(refNrm[i].y, gNrm[i * 4u + 1u], tol, "normals.y", i);
            cmpF(refNrm[i].z, gNrm[i * 4u + 2u], tol, "normals.z", i);
            cmpF(0.0f, gNrm[i * 4u + 3u], false, "normals.w", i);
            cmpF(refDlt[i].x, gDlt[i * 4u + 0u], tol, "delta.x", i);
            cmpF(refDlt[i].y, gDlt[i * 4u + 1u], tol, "delta.y", i);
            cmpF(refDlt[i].z, gDlt[i * 4u + 2u], tol, "delta.z", i);
            // the COUNT is exact in any order -- repeated +1.0f on a small integer
            cmpF(refCnt[i], gDlt[i * 4u + 3u], false, "delta.w(count)", i);
            cmpF(refVel[i].x, gVel[i * 4u + 0u], tol, "vel.x", i);
            cmpF(refVel[i].y, gVel[i * 4u + 1u], tol, "vel.y", i);
            cmpF(refVel[i].z, gVel[i * 4u + 2u], tol, "vel.z", i);
            cmpF(refVel[i].w, gVel[i * 4u + 3u], false, "vel.w", i);
        }
    }

    // Coverage: every trap must actually have been exercised, or the sweep is
    // measuring nothing.
    PXG_CHECK_MSG(nFanIn > 0, "the fan must produce delta.w > 1 (got %zu)", nFanIn);
    PXG_CHECK_MSG(nGateOpen > 0 && nGateClosed > 0,
                  "calculateDelta must fire BOTH ways (open %zu, closed %zu)",
                  nGateOpen, nGateClosed);
    PXG_CHECK_MSG(nDegenerate > 0, "a degenerate (zero-area) triangle must be present");
    PXG_CHECK_MSG(nFixed > 0, "a fixed (invMass 0) vertex must be present");

    PXG_CHECK_MSG(mm == 0, "the PBD aerodynamics pair is bit-exact on the disjoint "
                           "region (%zu mismatches)", mm);
    PXG_CHECK_MSG(tolMM == 0, "the fan-in region matches within the atomic-order "
                              "tolerance (%zu mismatches)", tolMM);

    for (Sys* s : order)
    {
        dev->destroyBuffer(s->bPos.h); dev->destroyBuffer(s->bVel.h);
        dev->destroyBuffer(s->bSortedPhase.h); dev->destroyBuffer(s->bU2S.h);
        dev->destroyBuffer(s->bRunsum.h); dev->destroyBuffer(s->bClothArr.h);
        dev->destroyBuffer(s->bNormalsA.h); dev->destroyBuffer(s->bDeltaA.h);
        dev->destroyBuffer(s->bNormalsB.h); dev->destroyBuffer(s->bDeltaB.h);
        dev->destroyBuffer(s->bVelB.h);
        dev->destroyBuffer(s->bPartArr.h);
        for (auto& u : s->bUnsortedPhase) dev->destroyBuffer(u.h);
        for (auto& c : s->cloths) dev->destroyBuffer(c.bTris.h);
    }
    dev->destroyBuffer(bDescA.h); dev->destroyBuffer(bDescB.h); dev->destroyBuffer(bActive.h);
    dev->destroyBuffer(bMats.h); dev->destroyBuffer(bPhaseToMat.h);
}

} // namespace
