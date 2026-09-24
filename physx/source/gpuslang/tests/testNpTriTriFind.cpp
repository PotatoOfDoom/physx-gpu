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

// Verification of pxgSlangTriTriFind (P9.83b-8, 3/n) through npTriTriFind.slang.
//
// The host mirror below is written FROM THE CUDA -- findInterestingTrianglesA
// (sdfCollision.cuh:619-687), addIndexToBuffer (:498-520), addToRefinementBuffer
// (:741-763) and the kernel's inner-loop control (convexMeshMidphase.cu:1062-1209)
// -- with a block of 1024 threads unrolled into a thread-ordered loop: the block
// scan is an exclusive running count in thread order, so the mirror needs no warp
// model.  The cull's geometry goes through pxgTriTriGeomRef.h (the engine's
// PxVec3/PxQuat/PxTransform/Gu:: underneath), bit for bit.
//
// What is compared, per case and per round: the count, the start index before and
// after, the refinement count, the index buffer (every slot any earlier round has
// written; slots nobody has written yet hold undefined groupshared), and the whole
// refinement buffer (the case seeds all 256 slots, so every slot is defined).
//
// UPSTREAM CANDIDATE 60 is constructed, not argued: 256 refinement entries whose
// 1024 children all pass make a child write slot 1023, and the start index after
// that round is its PARENT + 1.  Two cases pin the two outcomes the kernel can
// reach -- a rewind when the sweep had finished, and a silent skip when parent + 1
// equals the start -- and "refine everything" reaches it from the kernel's OWN
// starting state (start 0, no refinement), chaining rewinds round after round.  A
// third case (the forward jump) starts from a buffer the kernel cannot produce:
// in the kernel every entry is below startIndex and the buffer is sorted.  It is
// kept as a function-level equivalence check and labelled as such.
//
// The randomly seeded cases are function-level too: their seed buffers are
// unsorted and may hold entries at or above startIndex, which is fine for
// comparing the port with the mirror and says nothing about kernel states.

#include "pxgTestFramework.h"
#include "pxgTriTriGeomRef.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace pxgtest::tritrigeomref;

struct TtfPush
{
    uint64_t cases, out, meshInd, meshVerts;
    uint32_t numCases, pad0, pad1, pad2;
};
static_assert(sizeof(TtfPush) == 48, "TtfPush layout");

struct U2 { uint32_t x, y; };

struct TtfCase
{
    F4 scale0, rot0, scale1, rot1, aToBp, aToBq;
    F4 plane;
    U4 cfg0;   // nbTriangles, startIndex, initialN, maxRounds
    U4 cfg1;   // mode, passPermille, childPermille, refinePermille
    U4 cfg2;   // seed, maxRefinementLevel, tolerance bits, refineAll
    U4 cfgPad;
    U2 seedSub[256];
    uint32_t pad[4];
};
static_assert(sizeof(TtfCase) == 2240, "TtfCase stride");

struct TtfCullCase { F4 scale0, rot0, scale1, rot1, aToBp, aToBq; U4 misc; };
static_assert(sizeof(TtfCullCase) == 112, "TtfCullCase layout");

const uint32_t TARGET = 1024u, BLOCK = 1024u, SUBCAP = 256u, MAX_ROUNDS = 16u;
const uint32_t OUT_IDX = 64u, OUT_SUB = OUT_IDX + MAX_ROUNDS * 2048u, OUT_TAIL = OUT_SUB + MAX_ROUNDS * 512u;
const uint32_t OUT_STRIDE = OUT_TAIL + 4u;

uint32_t bitsOf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
float floatOf(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

uint32_t ttfHash(uint32_t a, uint32_t b, uint32_t seed)
{
    uint32_t x = a * 0x9E3779B1u ^ (b * 0x85EBCA77u) ^ seed;
    x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15;
    return x;
}

struct Mesh
{
    std::vector<F4> verts;
    std::vector<U4> inds;
};

// sdfCollision.cuh:661-670, the texture-free half.
float refCullSphere(const Mesh& m, const TtfCase& c, uint32_t ind, uint32_t sub, PxVec3& centroidVtx)
{
    const PxTransform aToB(v3(c.aToBp), q4(c.aToBq));
    PxVec3 v0, v1, v2;
    refGetTriangleVertices(m.inds, m.verts, v3(c.scale0), q4(c.rot0), aToB, ind, sub, v0, v1, v2);

    const physx::PxReal third = 1.f / 3.f;
    PxVec3 centroid = (v0 + v1 + v2) * third;

    centroidVtx = refShape2Vertex(centroid, v3(c.scale1), q4(c.rot1));
    physx::PxReal boundingSphereRadius = physx::PxSqrt(physx::PxMax((v0 - centroid).magnitudeSquared(),
        physx::PxMax((v1 - centroid).magnitudeSquared(), (v2 - centroid).magnitudeSquared())));
    boundingSphereRadius /= physx::PxMin(c.scale1.x, physx::PxMin(c.scale1.y, c.scale1.z));
    return boundingSphereRadius;
}

float refDistance(const TtfCase& c, uint32_t ind, uint32_t sub, const PxVec3& p)
{
    if (c.cfg1.x == 0u)
    {
        const uint32_t permille = sub != 0u ? c.cfg1.z : c.cfg1.y;
        return (ttfHash(ind, sub, c.cfg2.x) % 1000u) < permille ? -1.0f : 1.0e30f;
    }
    return p.x * c.plane.x + p.y * c.plane.y + p.z * c.plane.z - c.plane.w;
}

bool refRefineDecision(const TtfCase& c, uint32_t ind, uint32_t sub)
{
    if (c.cfg2.w != 0u) return true;
    return (ttfHash(ind, sub, c.cfg2.x ^ 0x5EEDu) % 1000u) < c.cfg1.w;
}

struct Stats { size_t childLastSlot = 0, mixedPasses = 0, refineOverflow = 0, levelCapDrops = 0, childEntries = 0, boundaryHits = 0, roundedBoundaryHits = 0; };

// findInterestingTrianglesA<32, 1024> with addIndexToBuffer inlined, one block.
uint32_t refFind(const Mesh& m, const TtfCase& c, uint32_t nbPrimitives, uint32_t& startIndex,
                 uint32_t nbSubdivisionIndices, const U2* subBuf, U2* idxBuf, Stats& st)
{
    uint32_t offset = startIndex;   // :626-627
    const float tolerance = floatOf(c.cfg2.z);
    uint32_t sharedIndicesCount = 0;
    for (uint32_t i = startIndex; (i < nbPrimitives || nbSubdivisionIndices > 0) && sharedIndicesCount < TARGET; )
    {
        std::vector<uint32_t> ind(BLOCK), sub(BLOCK);
        std::vector<uint8_t> add(BLOCK, 0);
        const uint32_t childThreads = std::min(TARGET, 4u * nbSubdivisionIndices);
        if (nbSubdivisionIndices > 0 && childThreads < BLOCK) ++st.mixedPasses;
        for (uint32_t t = 0; t < BLOCK; ++t)
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
        i += BLOCK - childThreads;
        nbSubdivisionIndices = 0;

        for (uint32_t t = 0; t < BLOCK; ++t)
        {
            if (ind[t] < nbPrimitives)
            {
                PxVec3 cv;
                const float r = refCullSphere(m, c, ind[t], sub[t], cv);
                const float d = refDistance(c, ind[t], sub[t], cv);
                add[t] = d < (r + tolerance);
                if (d == (r + tolerance)) ++st.boundaryHits;   // `<` and `<=` disagree here
                if (d == (r + tolerance) && double(r) + double(tolerance) > double(d)) ++st.roundedBoundaryHits;
            }
        }

        // addIndexToBuffer: the block scan in thread order.
        uint32_t excl = 0;
        for (uint32_t t = 0; t < BLOCK; ++t)
        {
            if (!add[t]) continue;
            const uint32_t w = sharedIndicesCount + excl;
            if (w < TARGET) { idxBuf[w] = U2{ ind[t], sub[t] }; if (sub[t] != 0) ++st.childEntries; }
            if (w == TARGET - 1u)
            {
                offset = ind[t] + 1u;
                if (sub[t] != 0 && t < childThreads) ++st.childLastSlot;
            }
            ++excl;
        }
        sharedIndicesCount = std::min(excl + sharedIndicesCount, TARGET);
    }

    if (startIndex != offset) startIndex = offset;   // :681-684
    else startIndex = nbPrimitives;
    return sharedIndicesCount;
}

uint32_t refAddToRefinement(const std::vector<uint8_t>& want, const std::vector<uint32_t>& ind,
                            const std::vector<uint32_t>& sub, uint32_t maxLevel, U2* subBuf, Stats& st)
{
    uint32_t excl = 0;
    for (uint32_t t = 0; t < BLOCK; ++t)
    {
        uint32_t level, idx;
        physx::Gu::decodeSubdivisionId(sub[t], level, idx);
        const bool add = want[t] && level < maxLevel;
        if (want[t] && !add) ++st.levelCapDrops;
        if (!add) continue;
        if (excl < SUBCAP) subBuf[excl] = U2{ ind[t], sub[t] };
        ++excl;
    }
    if (excl > SUBCAP) ++st.refineOverflow;
    return std::min(excl, SUBCAP);
}

struct RefRun
{
    std::vector<U4> headers;                    // per round
    std::vector<std::vector<U2>> idxImages;     // per round, 1024
    std::vector<std::vector<uint32_t>> idxValid;// per round: how many leading slots are defined
    std::vector<std::vector<U2>> subImages;     // per round, 256
    uint32_t rounds = 0, finalStart = 0, finalNumSub = 0;
};

RefRun refRounds(const Mesh& m, const TtfCase& c, Stats& st)
{
    RefRun R;
    std::vector<U2> idxBuf(TARGET, U2{ 0xDEADu, 0xDEADu });
    std::vector<U2> subBuf(c.seedSub, c.seedSub + SUBCAP);
    uint32_t written = 0;   // slots [0, written) have been written by some round
    const uint32_t nbTriangles = c.cfg0.x;
    uint32_t reg = c.cfg0.z;
    uint32_t startIndex = c.cfg0.y;
    const uint32_t maxRounds = std::min(c.cfg0.w, MAX_ROUNDS);
    uint32_t round = 0;
    while ((startIndex < nbTriangles || reg > 0) && round < maxRounds)
    {
        const uint32_t before = startIndex;
        const uint32_t nbFound = refFind(m, c, nbTriangles, startIndex, reg, subBuf.data(), idxBuf.data(), st);
        reg = 0;
        written = std::max(written, nbFound);

        std::vector<uint32_t> ind(BLOCK), sub(BLOCK);
        std::vector<uint8_t> want(BLOCK);
        for (uint32_t t = 0; t < BLOCK; ++t)
        {
            ind[t] = t < nbFound ? idxBuf[t].x : nbTriangles;
            sub[t] = t < nbFound ? idxBuf[t].y : 0u;
            want[t] = ind[t] < nbTriangles && refRefineDecision(c, ind[t], sub[t]);
        }
        const uint32_t newCount = refAddToRefinement(want, ind, sub, c.cfg2.y, subBuf.data(), st);
        reg = newCount;

        R.headers.push_back(U4{ nbFound, before, startIndex, reg });
        R.idxImages.push_back(idxBuf);
        R.idxValid.push_back(std::vector<uint32_t>{ written });
        R.subImages.push_back(subBuf);
        ++round;
    }
    R.rounds = round; R.finalStart = startIndex; R.finalNumSub = reg;
    return R;
}

struct Rng
{
    std::mt19937 g;
    explicit Rng(uint32_t s) : g(s) {}
    uint32_t u() { return g(); }
    float f(float lo, float hi) { return lo + (hi - lo) * (float(g() % 1000003u) / 1000003.f); }
    F4 unitQuat()
    {
        float a[4], m2 = 0.f;
        for (int k = 0; k < 4; ++k) { a[k] = f(-1.f, 1.f); m2 += a[k] * a[k]; }
        const float inv = 1.f / std::sqrt(m2 > 1e-4f ? m2 : 1.f);
        return F4{ a[0] * inv, a[1] * inv, a[2] * inv, a[3] * inv };
    }
    F4 point(float r) { return F4{ f(-r, r), f(-r, r), f(-r, r), 0.f }; }
};

const F4 IDQ{ 0, 0, 0, 1 };
const F4 ONE3{ 1, 1, 1, 0 };

// THE POISON TAIL (P9.83b-8 T1, F7 in pxgSlangTriTriFind).  One sweep pass reaches
// 1023 entries past the last triangle, and on this platform the guard on those
// threads is a mask, not a branch.  So the index buffer carries 1024 entries past
// the mesh whose "vertex indices" sit near 2^31: a speculated read of one of them
// would fetch a vertex ~32 GiB past the vertex buffer instead of landing quietly in
// whatever follows the allocation.  The clamp keeps every such read at triangle 0.
// Measured without the clamp: 5 runs of 5 pass, so this llvmpipe does not issue
// those reads today; the tail is a tripwire for the day it does.
static const uint32_t POISON_TAIL = 1024u;

Mesh makeMesh(Rng& rng, uint32_t nv, uint32_t nt)
{
    Mesh m;
    m.verts.resize(nv);
    for (uint32_t v = 0; v < nv; ++v) { m.verts[v] = rng.point(3.f); m.verts[v].w = 99.f; }
    m.inds.resize(nt + POISON_TAIL);
    for (uint32_t t = 0; t < nt; ++t)
    {
        uint32_t x = rng.u() % nv, y, z;
        do y = rng.u() % nv; while (y == x);
        do z = rng.u() % nv; while (z == x || z == y);
        m.inds[t] = U4{ x, y, z, (x + 1u) % nv };
    }
    for (uint32_t t = nt; t < nt + POISON_TAIL; ++t)
        m.inds[t] = U4{ 0x7FFFFFF0u + (t & 7u), 0x7FFFFFF8u, 0x7FFFFFFCu, 0xFFFFFFFFu };
    return m;
}

} // namespace

// ---------------------------------------------------------------------------
PXG_TEST(npTriTriFind_rounds)
{
    Rng rng(833001u);
    const uint32_t NV = 1500u, NT = 3000u;
    const Mesh mesh = makeMesh(rng, NV, NT);

    std::vector<TtfCase> cs;
    std::vector<const char*> names;
    auto base = [&](uint32_t nbTri, uint32_t start, uint32_t n) {
        TtfCase c{};
        c.scale0 = ONE3; c.rot0 = IDQ; c.scale1 = ONE3; c.rot1 = IDQ;
        c.aToBp = F4{ 0, 0, 0, 0 }; c.aToBq = IDQ;
        c.plane = F4{ 0, 0, 1, 0 };
        c.cfg0 = U4{ nbTri, start, n, MAX_ROUNDS };
        c.cfg1 = U4{ 0u, 500u, 500u, 0u };
        c.cfg2 = U4{ rng.u(), 8u, bitsOf(0.1f), 0u };
        for (uint32_t k = 0; k < SUBCAP; ++k)
        {
            const uint32_t level = rng.u() % 4u;
            c.seedSub[k] = U2{ rng.u() % (nbTri ? nbTri : 1u), level ? ((level << 26) | (rng.u() & ((1u << (2u * level)) - 1u))) : 0u };
        }
        return c;
    };
    auto push = [&](const TtfCase& c, const char* n) { cs.push_back(c); names.push_back(n); };

    // Hash mode: the control flow, driven into its corners.
    { TtfCase c = base(NT, 0, 0); c.cfg1 = U4{ 0, 500, 500, 0 }; push(c, "sweep, no refinement"); }
    { TtfCase c = base(NT, 0, 0); c.cfg1 = U4{ 0, 600, 600, 150 }; push(c, "sweep + refinement"); }
    { TtfCase c = base(NT, 0, 0); c.cfg1 = U4{ 0, 900, 900, 400 }; push(c, "dense refinement"); }
    { TtfCase c = base(NT, 0, 0); c.cfg1 = U4{ 0, 1000, 1000, 0 }; c.cfg2.w = 1u; push(c, "refine everything: overflow and the level-8 cap"); }
    { TtfCase c = base(NT, 0, 0); c.cfg1 = U4{ 0, 1000, 1000, 0 }; c.cfg2.w = 1u; c.cfg2.y = 3u; push(c, "refine everything, cap 3"); }
    { TtfCase c = base(NT, 1500, 0); c.cfg1 = U4{ 0, 400, 700, 200 }; push(c, "start mid-mesh"); }
    { TtfCase c = base(NT, 0, 100); c.cfg1 = U4{ 0, 500, 800, 100 }; push(c, "100 seeded: children beside the sweep"); }
    { TtfCase c = base(NT, 0, 256); c.cfg1 = U4{ 0, 500, 900, 100 }; push(c, "256 seeded, 90% children pass"); }
    { TtfCase c = base(0, 0, 50); push(c, "no triangles, 50 seeded"); }
    { TtfCase c = base(NT, NT, 37); c.cfg1 = U4{ 0, 500, 700, 300 }; push(c, "sweep done, refinement only"); }
    { TtfCase c = base(700, 0, 0); c.cfg1 = U4{ 0, 1000, 1000, 0 }; push(c, "fewer triangles than one buffer"); }
    { TtfCase c = base(NT, 0, 0); c.cfg1 = U4{ 0, 0, 0, 0 }; push(c, "nothing passes"); }

    // Upstream candidate 60, constructed.  256 parents 3k + 7 (the last is 772), every
    // child passes, so child 1023 writes the last slot and sets offset = 773.
    auto cand60 = [&](uint32_t start, const char* n) {
        TtfCase c = base(2000, start, 256);
        c.cfg1 = U4{ 0, 500, 1000, 0 };
        for (uint32_t k = 0; k < SUBCAP; ++k) c.seedSub[k] = U2{ 3u * k + 7u, (k % 3u) ? ((1u << 26) | (k & 3u)) : 0u };
        push(c, n);
    };
    cand60(2000, "candidate 60: sweep done -> REWIND to 773");
    cand60(773, "candidate 60: start == parent + 1 -> SKIP the rest");
    cand60(100, "candidate 60 (UNREACHABLE input, function-level): jump forward to 773");

    // Plane mode: the cull's geometry decides, mirrored bit for bit.
    for (uint32_t k = 0; k < 8; ++k)
    {
        TtfCase c = base(NT, 0, (k & 1u) ? 60u : 0u);
        const F4 n = rng.unitQuat();   // any unit 4-vector's xyz is a fine non-unit normal
        c.plane = F4{ n.x, n.y, n.z, rng.f(-0.5f, 0.5f) };
        c.cfg1 = U4{ 1u, 0, 0, 120u };
        c.cfg2.z = bitsOf(rng.f(0.02f, 0.4f));
        if (k >= 2) { c.scale0 = F4{ rng.f(0.5f, 2.f), rng.f(0.5f, 2.f), rng.f(0.5f, 2.f), 0 }; c.rot0 = rng.unitQuat(); }
        if (k >= 4) { c.scale1 = F4{ rng.f(0.5f, 2.f), rng.f(0.5f, 2.f), rng.f(0.5f, 2.f), 0 }; c.rot1 = rng.unitQuat(); }
        if (k >= 6) { c.aToBp = rng.point(1.f); c.aToBq = rng.unitQuat(); }
        push(c, "plane cull");
    }
    // The strict cull test ON its boundary: pick a triangle (and in two cases a
    // child) and put its distance exactly on r + tol.  k < 3: r + tol is EXACT
    // (Sterbenz), which pins `<` against `<=`.  k >= 3: r + tol ROUNDS DOWN onto d
    // (tol one ulp above d - r, whose ulp is a quarter of d's), which also pins the
    // CUDA's rounded sum against the rearranged `d - r < tol` -- the latter subtracts
    // exactly and sees d below the true sum.
    for (uint32_t k = 0; k < 6; ++k)
    {
        TtfCase c = base(NT, 0, k == 0 ? 0u : 64u);
        c.plane = F4{ 0.48f, -0.6f, 0.64f, 0.1f };
        c.cfg1 = U4{ 1u, 0, 0, 0u };
        if (k % 3u == 2u) { c.scale1 = F4{ 1.5f, 0.75f, 2.f, 0 }; c.rot1 = rng.unitQuat(); c.aToBp = rng.point(0.5f); c.aToBq = rng.unitQuat(); }
        // k % 3 == 0: triangle 11 itself; otherwise child 2 of the first seeded entry.
        uint32_t ind = 11u, sub = 0u;
        if (k % 3u != 0u) { ind = c.seedSub[0].x; sub = physx::Gu::elevateSubdivisionId(c.seedSub[0].y, 2u); }
        // Sterbenz: move the plane so d lands in [r/2, 2r]; then tol = d - r is EXACT
        // and fl(r + tol) == d with no search.  (Searching tol by ulps fails whenever
        // |tol| >> |d|: its steps jump over d.)
        PxVec3 cv;
        const float r = refCullSphere(mesh, c, ind, sub, cv);
        const float dotN = cv.x * c.plane.x + cv.y * c.plane.y + cv.z * c.plane.z;
        c.plane.w = dotN - 1.25f * r;
        const float d = refDistance(c, ind, sub, cv);
        float tol = d - r;
        if (k >= 3u) tol = std::nextafter(tol, 1e30f);   // the exact sum is now d + ulp(tol) < d + ulp(d)/2
        const bool rounded = double(r) + double(tol) > double(d);
        const bool hit = d >= 0.5f * r && d <= 2.f * r && (r + tol) == d && (rounded == (k >= 3u));
        PXG_CHECK_MSG(hit, "case %u is not on the cull boundary as intended (r %.9g, d %.9g, tol %.9g)", k, r, d, tol);
        c.cfg2.z = bitsOf(tol);
        push(c, k >= 3u ? "cull boundary, rounded sum" : "cull boundary, exact sum");
    }
    const uint32_t N = uint32_t(cs.size());

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> outInit(size_t(N) * OUT_STRIDE, 0xA5A5A5A5u);
    PxgBufferHandle bIn = createBufferFrom(dev, cs), bOut = createBufferFrom(dev, outInit);
    PxgBufferHandle bInd = createBufferFrom(dev, mesh.inds), bVert = createBufferFrom(dev, mesh.verts);
    TtfPush p{};
    p.cases = dev->getBufferDeviceAddress(bIn); p.out = dev->getBufferDeviceAddress(bOut);
    p.meshInd = dev->getBufferDeviceAddress(bInd); p.meshVerts = dev->getBufferDeviceAddress(bVert);
    p.numCases = N;
    {
        Kernel k("npTriTriFind.spv", "ttfRounds", {}, sizeof(TtfPush));
        PXG_CHECK(k.isValid());
        k.dispatch(N, {}, p);
        dev->flush();
    }
    const std::vector<uint32_t> got = downloadBuffer<uint32_t>(dev, bOut, size_t(N) * OUT_STRIDE);

    Stats st;
    size_t mmHdr = 0, mmIdx = 0, mmSub = 0, mmTail = 0, totalRounds = 0;
    std::vector<RefRun> runs;
    for (uint32_t c = 0; c < N; ++c)
    {
        const RefRun R = refRounds(mesh, cs[c], st);
        const uint32_t* o = &got[size_t(c) * OUT_STRIDE];
        totalRounds += R.rounds;
        if (o[OUT_TAIL] != R.rounds || o[OUT_TAIL + 1] != R.finalStart || o[OUT_TAIL + 2] != R.finalNumSub || o[OUT_TAIL + 3] != 0x51DEu)
        {
            std::printf("  case %u (%s): tail (%u, %u, %u) want (%u, %u, %u)\n", c, names[c], o[OUT_TAIL], o[OUT_TAIL + 1],
                        o[OUT_TAIL + 2], R.rounds, R.finalStart, R.finalNumSub);
            ++mmTail;
        }
        for (uint32_t r = 0; r < R.rounds && r < MAX_ROUNDS; ++r)
        {
            const U4& h = R.headers[r];
            if (o[r * 4] != h.x || o[r * 4 + 1] != h.y || o[r * 4 + 2] != h.z || o[r * 4 + 3] != h.w)
            {
                if (mmHdr < 6)
                    std::printf("  case %u (%s) round %u: (count %u, start %u -> %u, sub %u) want (%u, %u -> %u, %u)\n",
                                c, names[c], r, o[r * 4], o[r * 4 + 1], o[r * 4 + 2], o[r * 4 + 3], h.x, h.y, h.z, h.w);
                ++mmHdr;
            }
            const uint32_t valid = R.idxValid[r][0];
            for (uint32_t s = 0; s < valid; ++s)
            {
                const uint32_t gx = o[OUT_IDX + r * 2048u + 2u * s], gy = o[OUT_IDX + r * 2048u + 2u * s + 1u];
                if (gx != R.idxImages[r][s].x || gy != R.idxImages[r][s].y)
                {
                    if (mmIdx < 6) std::printf("  case %u round %u slot %u: (%u, 0x%08x) want (%u, 0x%08x)\n", c, r, s, gx, gy,
                                               R.idxImages[r][s].x, R.idxImages[r][s].y);
                    ++mmIdx;
                }
            }
            for (uint32_t s = 0; s < SUBCAP; ++s)
            {
                const uint32_t gx = o[OUT_SUB + r * 512u + 2u * s], gy = o[OUT_SUB + r * 512u + 2u * s + 1u];
                if (gx != R.subImages[r][s].x || gy != R.subImages[r][s].y) ++mmSub;
            }
        }
        runs.push_back(R);
    }

    // Candidate 60, stated as outcomes rather than as a mirror match.
    auto find = [&](const char* prefix) { for (uint32_t c = 0; c < N; ++c) if (std::strncmp(names[c], prefix, std::strlen(prefix)) == 0) return c; return N; };
    const uint32_t cRewind = find("candidate 60: sweep done"), cSkip = find("candidate 60: start =="), cJump = find("candidate 60 (UNREACHABLE");
    const uint32_t* oR = &got[size_t(cRewind) * OUT_STRIDE];
    const uint32_t* oS = &got[size_t(cSkip) * OUT_STRIDE];
    const uint32_t* oJ = &got[size_t(cJump) * OUT_STRIDE];
    std::printf("  candidate 60: rewind case round 0 start %u -> %u (count %u); skip case %u -> %u; jump case %u -> %u\n",
                oR[1], oR[2], oR[0], oS[1], oS[2], oJ[1], oJ[2]);
    PXG_CHECK_MSG(oR[0] == TARGET && oR[1] == 2000u && oR[2] == 773u,
                  "candidate 60 rewind: a child in slot 1023 sends the finished sweep back to parent + 1");
    PXG_CHECK_MSG(oS[0] == TARGET && oS[1] == 773u && oS[2] == 2000u,
                  "candidate 60 skip: offset == startIndex reads as 'no overflow' and ends the sweep");
    PXG_CHECK_MSG(oJ[0] == TARGET && oJ[1] == 100u && oJ[2] == 773u,
                  "handshake on an unreachable buffer (entries above startIndex): the port still follows the CUDA");

    std::printf("  rounds: %u cases, %zu rounds; child in last slot %zu, mixed passes %zu, refinement overflow %zu, "
                "level-cap drops %zu, child entries %zu, cull-boundary hits %zu (rounded sum %zu)\n",
                N, totalRounds, st.childLastSlot, st.mixedPasses, st.refineOverflow, st.levelCapDrops, st.childEntries,
                st.boundaryHits, st.roundedBoundaryHits);
    PXG_CHECK_MSG(mmHdr == 0, "%zu round-header mismatches", mmHdr);
    PXG_CHECK_MSG(mmIdx == 0, "%zu index-buffer mismatches", mmIdx);
    PXG_CHECK_MSG(mmSub == 0, "%zu refinement-buffer mismatches", mmSub);
    PXG_CHECK_MSG(mmTail == 0, "%zu tail mismatches", mmTail);
    PXG_CHECK_MSG(st.childLastSlot >= 3, "a child reached slot 1023 only %zu times", st.childLastSlot);
    PXG_CHECK_MSG(st.mixedPasses >= 5, "children beside a sweep only %zu times", st.mixedPasses);
    PXG_CHECK_MSG(st.refineOverflow >= 5, "the 256 cap overflowed only %zu times", st.refineOverflow);
    PXG_CHECK_MSG(st.levelCapDrops >= 100, "the level cap dropped only %zu requests", st.levelCapDrops);
    PXG_CHECK_MSG(st.childEntries >= 2000, "only %zu child entries gathered", st.childEntries);
    PXG_CHECK_MSG(st.boundaryHits >= 6, "the strict cull test met its boundary only %zu times", st.boundaryHits);
    PXG_CHECK_MSG(st.roundedBoundaryHits >= 3, "a ROUNDED r + tol met d only %zu times", st.roundedBoundaryHits);
    for (PxgBufferHandle h : { bIn, bOut, bInd, bVert }) dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
PXG_TEST(npTriTriFind_cullSphere)
{
    Rng rng(833002u);
    const Mesh mesh = makeMesh(rng, 400u, 300u);
    const uint32_t N = 3000u;
    std::vector<TtfCullCase> cs(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        TtfCullCase s{};
        s.scale0 = F4{ rng.f(0.3f, 3.f), rng.f(0.3f, 3.f), rng.f(0.3f, 3.f), 0 }; s.rot0 = rng.unitQuat();
        s.scale1 = F4{ rng.f(0.3f, 3.f), rng.f(0.3f, 3.f), rng.f(0.3f, 3.f), 0 }; s.rot1 = rng.unitQuat();
        s.aToBp = rng.point(4.f); s.aToBq = rng.unitQuat();
        // Which scale1 component is smallest, in every position, so a min that drops
        // or misreads a component shows.  The ORDER and NESTING of the two selectMins
        // cannot show: on finite positive scales every order returns the same bits
        // (only NaN or a +0/-0 pair tells them apart, and a PxMeshScale has neither).
        switch (i % 6u)
        {
        case 0: s.scale1.x = 0.25f; break;
        case 1: s.scale1.y = 0.25f; break;
        case 2: s.scale1.z = 0.25f; break;
        case 3: s.scale1.y = s.scale1.z = 0.25f; break;
        default: break;
        }
        const uint32_t level = i % 9u;   // 0..8, the engine's range
        s.misc = U4{ rng.u() % 300u, level ? ((level << 26) | (rng.u() & ((1u << (2u * level)) - 1u))) : 0u, 0, 0 };
        cs[i] = s;
    }
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<U4> outInit(N, U4{ 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu });
    PxgBufferHandle bIn = createBufferFrom(dev, cs), bOut = createBufferFrom(dev, outInit);
    PxgBufferHandle bInd = createBufferFrom(dev, mesh.inds), bVert = createBufferFrom(dev, mesh.verts);
    TtfPush p{};
    p.cases = dev->getBufferDeviceAddress(bIn); p.out = dev->getBufferDeviceAddress(bOut);
    p.meshInd = dev->getBufferDeviceAddress(bInd); p.meshVerts = dev->getBufferDeviceAddress(bVert);
    p.numCases = N;
    {
        Kernel k("npTriTriFind.spv", "ttfCullProbe", {}, sizeof(TtfPush));
        PXG_CHECK(k.isValid());
        k.dispatch((N + 63u) / 64u, {}, p);
        dev->flush();
    }
    const std::vector<U4> got = downloadBuffer<U4>(dev, bOut, N);
    size_t mm = 0, discDistributedMax = 0, discBuiltinSqrtLen = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        TtfCase c{};
        c.scale0 = cs[i].scale0; c.rot0 = cs[i].rot0; c.scale1 = cs[i].scale1; c.rot1 = cs[i].rot1;
        c.aToBp = cs[i].aToBp; c.aToBq = cs[i].aToBq;
        PxVec3 cv;
        const float r = refCullSphere(mesh, c, cs[i].misc.x, cs[i].misc.y, cv);
        if (got[i].x != bitsOf(cv.x) || got[i].y != bitsOf(cv.y) || got[i].z != bitsOf(cv.z) || got[i].w != bitsOf(r))
        {
            if (mm < 4) std::printf("  cull case %u: r 0x%08x want 0x%08x\n", i, got[i].w, bitsOf(r));
            ++mm;
        }
        // A max of three LENGTHS is the SAME number as one sqrt of the max of squares
        // (sqrt is monotone and correctly rounded, so it picks the same element and
        // rounds it the same way) -- asserted per case, because pxgSlangSdfFind's T2
        // note claimed the opposite.  A reciprocal multiply for the scale division is
        // a real difference, and is the discriminator.
        const PxTransform aToB(v3(c.aToBp), q4(c.aToBq));
        PxVec3 v0, v1, v2;
        refGetTriangleVertices(mesh.inds, mesh.verts, v3(c.scale0), q4(c.rot0), aToB, cs[i].misc.x, cs[i].misc.y, v0, v1, v2);
        const PxVec3 ctr = (v0 + v1 + v2) * (1.f / 3.f);
        const float minS = physx::PxMin(c.scale1.x, physx::PxMin(c.scale1.y, c.scale1.z));
        const float alt = physx::PxMax((v0 - ctr).magnitude(), physx::PxMax((v1 - ctr).magnitude(), (v2 - ctr).magnitude())) / minS;
        if (bitsOf(alt) != bitsOf(r)) ++discDistributedMax;   // expected: never
        const float alt2 = physx::PxSqrt(physx::PxMax((v0 - ctr).magnitudeSquared(), physx::PxMax((v1 - ctr).magnitudeSquared(),
                                         (v2 - ctr).magnitudeSquared()))) * (1.f / minS);
        if (bitsOf(alt2) != bitsOf(r)) ++discBuiltinSqrtLen;
    }
    std::printf("  cull sphere: %u cases, max-of-lengths differing %zu (must be 0), reciprocal-scale discriminating %zu\n",
                N, discDistributedMax, discBuiltinSqrtLen);
    PXG_CHECK_MSG(mm == 0, "%zu cull-sphere mismatches", mm);
    PXG_CHECK_MSG(discDistributedMax == 0, "max-of-lengths differed in %zu cases -- the equivalence is wrong", discDistributedMax);
    PXG_CHECK_MSG(discBuiltinSqrtLen > 150, "reciprocal scale visible in only %zu cases (the 0.25 ties are exact either way)", discBuiltinSqrtLen);
    for (PxgBufferHandle h : { bIn, bOut, bInd, bVert }) dev->destroyBuffer(h);
}
