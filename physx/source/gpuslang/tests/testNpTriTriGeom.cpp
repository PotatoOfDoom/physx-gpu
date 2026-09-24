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

// Verification of pxgSlangTriTriGeom (P9.83b-8, 1/n) through npTriTriGeom.slang.
//
// THE ORACLE IS THE ENGINE, wherever the engine is host-compilable:
//   GuTriangleRefinement.h   getSubTriangleEncoded / decode / encode / elevate,
//                            included, not transcribed
//   foundation PxTransform   transform, transformInv, getInverse -- the real
//                            methods, not re-typed arithmetic
//   foundation PxMath        PxMax(float) = intrinsics::selectMax
// Only the __device__ functions are transcribed, each next to its CUDA line, in
// the shared pxgTriTriGeomRef.h: vertex2Shape (cudaNpCommon.h:270-281),
// getTriangleVertices (sdfCollision.cuh:522-538) and triangleRadiusSquared /
// sdfRadiusSquared (sdfCollision.cuh:729-739) -- and those transcriptions call the
// engine's PxQuat/PxTransform/Gu:: pieces for everything below them.
//
// Every comparison is bitwise.  The shader is stamped NoContraction, this file
// is compiled with -ffp-contract=off, and the arithmetic is the same sequence of
// separately rounded IEEE operations on both sides.  NaN compares equal to NaN
// (the payload an inf - inf produces is the hardware's business, not the port's);
// every other value, including the sign of zero, must match to the bit.
//
// A bitwise fixture only certifies an ORDER if the wrong order would give a
// different bit pattern on the chosen inputs.  So each claim about order comes
// with a DISCRIMINATOR counted on the host: how many cases the tempting wrong
// spelling would change.  A discriminator of zero means the fixture could not
// see that mistake, and the test fails rather than passes.
//
// Levels.  The encoding gives the level six bits at << 26 (its comment says
// four), and getSubTriangle shifts by 2*(level-1), which is undefined past level
// 16.  The engine never exceeds level 8 (maxRefinementLevel, cu:1059); this
// fixture covers 0..13, the range in which a level's index still fits the
// 26-bit field, and it also feeds indices with junk ABOVE bit 2*level, which the
// engine never makes but which are defined -- getSubTriangle must ignore them and
// elevate must carry them.

#include "pxgTestFramework.h"
#include "pxgTriTriGeomRef.h"   // the __device__ mirrors; GuTriangleRefinement.h / PxTransform.h under them

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace pxgtest::tritrigeomref;

struct TtgParams
{
    uint32_t numCases, pad0, pad1, pad2;
    uint64_t cases, outBuf, meshInd, meshVerts;
};
static_assert(sizeof(TtgParams) == 48, "TtgParams layout");

struct SubdivCase { F4 a, b, c; U4 misc; };
struct VertsCase { F4 scale, scaleRot, tp, tq; U4 misc; };
struct MathCase { F4 v0, v1, v2, boxLo, boxHi, tp, tq, v; };
static_assert(sizeof(SubdivCase) == 64, "SubdivCase layout");
static_assert(sizeof(VertsCase) == 80, "VertsCase layout");
static_assert(sizeof(MathCase) == 128, "MathCase layout");

const float QNAN = std::numeric_limits<float>::quiet_NaN();
const float PINF = std::numeric_limits<float>::infinity();

uint32_t bitsOf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
float floatOf(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// Bitwise, except that any NaN equals any NaN (see the header).
bool sameBits(uint32_t got, float want)
{
    if (std::isnan(want)) return std::isnan(floatOf(got));
    return got == bitsOf(want);
}
bool sameVec(const U4& got, const PxVec3& want)
{
    return sameBits(got.x, want.x) && sameBits(got.y, want.y) && sameBits(got.z, want.z) && got.w == 0u;
}
bool differs(const PxVec3& p, const PxVec3& q)
{
    return bitsOf(p.x) != bitsOf(q.x) || bitsOf(p.y) != bitsOf(q.y) || bitsOf(p.z) != bitsOf(q.z);
}

// ---- inputs ------------------------------------------------------------------

struct Rng
{
    std::mt19937 g;
    explicit Rng(uint32_t seed) : g(seed) {}
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

uint32_t runKernel(const char* entry, uint32_t numCases, uint64_t cases, uint64_t outBuf,
                   uint64_t meshInd = 0, uint64_t meshVerts = 0)
{
    TtgParams p{};
    p.numCases = numCases;
    p.cases = cases;
    p.outBuf = outBuf;
    p.meshInd = meshInd;
    p.meshVerts = meshVerts;
    Kernel k("npTriTriGeom.spv", entry, {}, sizeof(TtgParams));
    if (!k.isValid()) return 0;
    k.dispatch((numCases + 63u) / 64u, {}, p);
    testDevice()->flush();
    return 1;
}

// The engine's getSubTriangle with the digit order REVERSED (least significant
// first): the discriminator for the most plausible transcription slip.
void subTriangleLsdFirst(PxVec3& a, PxVec3& b, PxVec3& c, uint32_t id, uint32_t steps)
{
    for (uint32_t i = 0; i < steps; ++i)
        physx::Gu::getSubTriangle4(int((id >> (2u * i)) & 3u), a, b, c);
}

// The engine's getSubTriangle4 / getSubTriangle with the DISTRIBUTED midpoint
// 0.5f*a + 0.5f*b: the discriminator for the midpoint order, which only an
// overflowing a + b can show (see pxgSlangTriTriGeom.slang).
void subTriangle4Distributed(uint32_t id, PxVec3& a, PxVec3& b, PxVec3& c)
{
    const PxVec3 ab = 0.5f * a + 0.5f * b, bc = 0.5f * b + 0.5f * c, ca = 0.5f * c + 0.5f * a;
    if (id == 0) { b = ab; c = ca; }
    else if (id == 1) { a = ab; c = bc; }
    else if (id == 2) { a = ca; b = bc; }
    else { a = ab; b = bc; c = ca; }
}
void subTriangleDistributed(PxVec3& a, PxVec3& b, PxVec3& c, uint32_t id, uint32_t steps)
{
    for (uint32_t i = 0; i < steps; ++i)
        subTriangle4Distributed((id >> (2u * (steps - i - 1u))) & 3u, a, b, c);
}
bool differsNaN(const PxVec3& p, const PxVec3& q)   // NaN == NaN, as the comparison does
{
    const float x[3] = { p.x, p.y, p.z }, y[3] = { q.x, q.y, q.z };
    for (int k = 0; k < 3; ++k)
        if (!(std::isnan(x[k]) && std::isnan(y[k])) && bitsOf(x[k]) != bitsOf(y[k])) return true;
    return false;
}

} // namespace

// ---------------------------------------------------------------------------
// Subdivision: getSubTriangleEncoded, decode, encode(decode), elevate x4.
// ---------------------------------------------------------------------------
PXG_TEST(npTriTriGeom_subdivision)
{
    Rng rng(831001u);
    std::vector<SubdivCase> cs;

    // misc.x drives getSubTriangleEncoded (levels 0..13), misc.y drives the
    // integer trio decode/encode/elevate, whose domain is every uint -- see the
    // case layout in npTriTriGeom.slang.  Most cases use one id for both.
    auto push = [&](uint32_t enc, uint32_t idId) {
        SubdivCase s{};
        s.a = rng.point(4.f); s.b = rng.point(4.f); s.c = rng.point(4.f);
        s.misc = U4{ enc, idId, 0u, 0u };
        cs.push_back(s);
    };
    auto pushCorners = [&](F4 a, F4 b, F4 c, uint32_t enc) {
        SubdivCase s{};
        s.a = a; s.b = b; s.c = c;
        s.misc = U4{ enc, enc, 0u, 0u };
        cs.push_back(s);
    };

    // Named ids first: the unrefined triangle, each level-1 child, the one id
    // whose digits read differently in the two orders, the deepest fully-set id.
    const uint32_t SH = 26u;
    auto push1 = [&](uint32_t enc) { push(enc, enc); };
    push1(0u);
    for (uint32_t k = 0; k < 4; ++k) push1((1u << SH) | k);
    push1((2u << SH) | 0x1u);                 // digits (0,1) MSD-first
    push1((2u << SH) | 0x4u);                 // digits (1,0)
    push1((2u << SH) | 0xEu);                 // digits (3,2)
    push1((13u << SH) | ((1u << SH) - 1u));   // thirteen 3s: the centre, thirteen times
    push1((13u << SH) | 0x1B1B1B1u);          // mixed, every digit value present
    push1((12u << SH) | ((1u << 24) - 1u));   // level 12 at its full index width
    push1((1u << SH) | 0x3FFFFFCu);           // level 1, digit 0, junk above bit 2
    push1(0x3FFFFFFu);                        // level 0 with a full junk index
    // The integer trio past what "4 bits for subdivision level" would hold: the
    // level field is six bits, and 16, 31, 32 and 63 are where a four- or five-bit
    // reading of it, or an elevate that wraps the level, would show.
    for (uint32_t L : { 14u, 15u, 16u, 17u, 31u, 32u, 62u, 63u })
    {
        push(0u, (L << SH) | 0x2D2D2Du);
        push(0u, (L << SH) | ((1u << SH) - 1u));
    }
    push(0u, 0xFFFFFFFFu);
    // The midpoint order's only witness: corners whose pairwise sums overflow.
    // 0.5f * (3e38 + 3e38) = 0.5f * inf = inf; 0.5f*3e38 + 0.5f*3e38 = 3e38.
    {
        const float H = 3e38f;
        const F4 corners[3][3] = {
            { { H, -H, 1.f, 0 }, { H, 1.f, -H, 0 }, { -H, 2.f, 3.f, 0 } },
            { { H, H, H, 0 }, { H, H, -H, 0 }, { -H, H, H, 0 } },
            { { -H, 0.5f, H, 0 }, { -H, -H, 0.25f, 0 }, { 1.f, -H, H, 0 } },
        };
        for (const auto& t : corners)
            for (uint32_t enc : { (1u << SH) | 0u, (1u << SH) | 1u, (1u << SH) | 2u, (1u << SH) | 3u,
                                  (2u << SH) | 0x6u, (3u << SH) | 0x1Bu })
                pushCorners(t[0], t[1], t[2], enc);
    }

    const uint32_t N = 4096u;
    while (cs.size() < N)
    {
        const uint32_t i = uint32_t(cs.size());
        const uint32_t level = i % 14u;                              // 0..13
        const uint32_t digitMask = (1u << (2u * level)) - 1u;                  // 2*13 = 26: still defined
        const uint32_t index = (i % 4u == 3u) ? (rng.u() & ((1u << SH) - 1u))   // junk above 2*level
                                              : (rng.u() & digitMask);
        const uint32_t enc = (level << SH) | index;
        push(enc, (i % 8u == 6u) ? rng.u() : enc);   // 1 in 8: the whole uint domain
    }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<U4> outInit(N * 5u, U4{ 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu });
    PxgBufferHandle bIn = createBufferFrom(dev, cs);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    PXG_CHECK(runKernel("ttgSubdivProbe", N, dev->getBufferDeviceAddress(bIn), dev->getBufferDeviceAddress(bOut)));
    const std::vector<U4> got = downloadBuffer<U4>(dev, bOut, N * 5u);

    size_t mmVerts = 0, mmDecode = 0, mmElevate = 0;
    size_t discOrder = 0, discJunk = 0, discMidpoint = 0, moved = 0, elevOverflow = 0, wideLevel = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const SubdivCase& s = cs[i];
        const uint32_t enc = s.misc.x, idId = s.misc.y;
        PxVec3 a = v3(s.a), b = v3(s.b), c = v3(s.c);
        physx::Gu::getSubTriangleEncoded(a, b, c, enc);

        physx::PxU32 level, index;
        physx::Gu::decodeSubdivisionId(idId, level, index);
        physx::PxU32 subLevel, subIndex;   // what getSubTriangleEncoded walked
        physx::Gu::decodeSubdivisionId(enc, subLevel, subIndex);

        const U4* o = &got[i * 5u];
        if (!sameVec(o[0], a) || !sameVec(o[1], b) || !sameVec(o[2], c))
        {
            if (mmVerts < 4)
                std::printf("  subdiv case %u enc 0x%08x: vertex mismatch (got a.x 0x%08x want 0x%08x)\n",
                            i, enc, o[0].x, bitsOf(a.x));
            ++mmVerts;
        }
        if (o[3].x != level || o[3].y != index || o[3].z != physx::Gu::encodeSubdivisionId(level, index) || o[3].w != 0u)
        {
            if (mmDecode < 4)
                std::printf("  subdiv case %u id 0x%08x: decode (%u,%u,0x%08x) want (%u,%u)\n",
                            i, idId, o[3].x, o[3].y, o[3].z, level, index);
            ++mmDecode;
        }
        const uint32_t e[4] = { physx::Gu::elevateSubdivisionId(idId, 0u), physx::Gu::elevateSubdivisionId(idId, 1u),
                                physx::Gu::elevateSubdivisionId(idId, 2u), physx::Gu::elevateSubdivisionId(idId, 3u) };
        if (o[4].x != e[0] || o[4].y != e[1] || o[4].z != e[2] || o[4].w != e[3])
        {
            if (mmElevate < 4)
                std::printf("  subdiv case %u id 0x%08x: elevate 0x%08x want 0x%08x\n", i, idId, o[4].x, e[0]);
            ++mmElevate;
        }

        // Discriminators.
        PxVec3 la = v3(s.a), lb = v3(s.b), lc = v3(s.c);
        subTriangleLsdFirst(la, lb, lc, subIndex, subLevel);
        if (differs(la, a) || differs(lb, b) || differs(lc, c)) ++discOrder;
        // Junk above bit 2*level: the port must ignore it.  What checks that is
        // mmVerts on these cases -- this only counts that they exist.
        const uint32_t digitsOnly = subIndex & ((1u << (2u * subLevel)) - 1u);
        if (digitsOnly != subIndex) ++discJunk;
        PxVec3 da = v3(s.a), db = v3(s.b), dc = v3(s.c);
        subTriangleDistributed(da, db, dc, subIndex, subLevel);
        if (differsNaN(da, a) || differsNaN(db, b) || differsNaN(dc, c)) ++discMidpoint;
        if (differs(a, v3(s.a)) || differs(b, v3(s.b)) || differs(c, v3(s.c))) ++moved;
        if ((e[3] >> 26) != ((level + 1u) & 63u)) ++elevOverflow;   // 4*index+3 spilled into the level field
        if (level >= 16u) ++wideLevel;
    }

    std::printf("  subdivision: %u cases, %zu moved, order-discriminating %zu, junk-index %zu, "
                "midpoint-order discriminating %zu, elevate-into-level %zu, level field >= 16 %zu\n",
                N, moved, discOrder, discJunk, discMidpoint, elevOverflow, wideLevel);
    PXG_CHECK_MSG(mmVerts == 0, "%zu vertex mismatches", mmVerts);
    PXG_CHECK_MSG(mmDecode == 0, "%zu decode/encode mismatches", mmDecode);
    PXG_CHECK_MSG(mmElevate == 0, "%zu elevate mismatches", mmElevate);
    PXG_CHECK_MSG(discOrder > 1000, "only %zu cases can tell MSD-first from LSD-first", discOrder);
    PXG_CHECK_MSG(discJunk > 500, "only %zu cases carry index bits above 2*level", discJunk);
    PXG_CHECK_MSG(discMidpoint >= 10, "only %zu cases can tell 0.5f*(a+b) from 0.5f*a+0.5f*b", discMidpoint);
    PXG_CHECK_MSG(moved > 3000, "only %zu cases subdivide at all", moved);
    PXG_CHECK_MSG(elevOverflow > 50, "only %zu elevations overflow into the level field", elevOverflow);
    PXG_CHECK_MSG(wideLevel > 100, "only %zu ids have a level field of 16 or more", wideLevel);
    dev->destroyBuffer(bIn);
    dev->destroyBuffer(bOut);
}

// ---------------------------------------------------------------------------
// getTriangleVertices (with and without sub-index) over a mesh.
// ---------------------------------------------------------------------------
PXG_TEST(npTriTriGeom_vertices)
{
    Rng rng(831002u);

    // The mesh.  `w` of every vertex is a poison that must never reach xyz, and
    // `w` of every triangle is a VALID vertex index different from x, y and z, so
    // a port that read inds.w for a corner would load a real, wrong vertex.
    const uint32_t NV = 97u, NT = 61u;
    std::vector<F4> verts(NV);
    for (uint32_t v = 0; v < NV; ++v) { verts[v] = rng.point(3.f); verts[v].w = 12345.f; }
    std::vector<U4> inds(NT);
    for (uint32_t t = 0; t < NT; ++t)
    {
        uint32_t x = rng.u() % NV, y, z, w;
        do y = rng.u() % NV; while (y == x);
        do z = rng.u() % NV; while (z == x || z == y);
        do w = rng.u() % NV; while (w == x || w == y || w == z);
        inds[t] = U4{ x, y, z, w };
    }

    const uint32_t N = 4096u;
    std::vector<VertsCase> cs(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        VertsCase s{};
        s.scale = F4{ rng.f(0.25f, 3.f), rng.f(0.25f, 3.f), rng.f(0.25f, 3.f), 0.f };
        if (i % 8u == 5u) s.scale.y = -s.scale.y;                 // a mirrored mesh scale
        s.scaleRot = rng.unitQuat();
        s.tp = rng.point(5.f);
        s.tq = rng.unitQuat();
        if (i % 16u == 1u) { s.scale = F4{ 1.f, 1.f, 1.f, 0.f }; s.scaleRot = F4{ 0.f, 0.f, 0.f, 1.f }; }
        if (i % 16u == 2u) { s.tp = F4{ 0.f, 0.f, 0.f, 0.f }; s.tq = F4{ 0.f, 0.f, 0.f, 1.f }; }

        uint32_t enc = 0u;
        if (i % 4u != 0u)
        {
            const uint32_t level = (i % 11u == 7u) ? 9u + rng.u() % 5u : 1u + rng.u() % 8u;   // 1..8, some 9..13
            enc = (level << 26) | (rng.u() & ((1u << (2u * level)) - 1u));
        }
        s.misc = U4{ rng.u() % NT, enc, 0u, 0u };
        cs[i] = s;
    }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<U4> outInit(N * 3u, U4{ 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu });
    PxgBufferHandle bIn = createBufferFrom(dev, cs);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    PxgBufferHandle bInd = createBufferFrom(dev, inds);
    PxgBufferHandle bVert = createBufferFrom(dev, verts);
    PXG_CHECK(runKernel("ttgVertsProbe", N, dev->getBufferDeviceAddress(bIn), dev->getBufferDeviceAddress(bOut),
                        dev->getBufferDeviceAddress(bInd), dev->getBufferDeviceAddress(bVert)));
    const std::vector<U4> got = downloadBuffer<U4>(dev, bOut, N * 3u);

    size_t mmTri = 0, discSubFirst = 0, discNoScaleRot = 0, refined = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const VertsCase& s = cs[i];
        const PxVec3 scale = v3(s.scale);
        const PxQuat scaleRot = q4(s.scaleRot);
        const PxTransform t(v3(s.tp), q4(s.tq));
        PxVec3 a, b, c;
        refGetTriangleVertices(inds, verts, scale, scaleRot, t, s.misc.x, s.misc.y, a, b, c);

        const U4* o = &got[i * 3u];
        if (!sameVec(o[0], a) || !sameVec(o[1], b) || !sameVec(o[2], c))
        {
            if (mmTri < 4)
                std::printf("  verts case %u tri %u enc 0x%08x: a.x got 0x%08x want 0x%08x\n",
                            i, s.misc.x, s.misc.y, o[0].x, bitsOf(a.x));
            ++mmTri;
        }

        // Discriminator: subdividing in VERTEX space and transforming afterwards
        // is the same triangle in exact arithmetic and a different one in float.
        if (s.misc.y > 0u)
        {
            ++refined;
            const U4 ti = inds[s.misc.x];
            PxVec3 sa = v3(verts[ti.x]), sb = v3(verts[ti.y]), sc = v3(verts[ti.z]);
            physx::Gu::getSubTriangleEncoded(sa, sb, sc, s.misc.y);
            sa = t.transform(refVertex2Shape(sa, scale, scaleRot));
            sb = t.transform(refVertex2Shape(sb, scale, scaleRot));
            sc = t.transform(refVertex2Shape(sc, scale, scaleRot));
            if (differs(sa, a) || differs(sb, b) || differs(sc, c)) ++discSubFirst;
        }
        // Discriminator: vertex2Shape without the rotated frame (scale applied
        // on the mesh axes) -- the rotation must be visible.  On the unrefined
        // cases, where corner a IS vertex inds.x.
        if (s.misc.y == 0u)
        {
            const PxVec3 raw = v3(verts[inds[s.misc.x].x]);
            const PxVec3 naive = t.transform(PxVec3(raw.x * scale.x, raw.y * scale.y, raw.z * scale.z));
            if (differs(naive, a)) ++discNoScaleRot;
        }
    }

    std::printf("  vertices: %u cases (%zu refined), subdivide-first discriminating %zu, "
                "unrotated-scale discriminating %zu\n", N, refined, discSubFirst, discNoScaleRot);
    PXG_CHECK_MSG(mmTri == 0, "%zu getTriangleVertices mismatches", mmTri);
    PXG_CHECK_MSG(refined > 2500 && discSubFirst > refined / 2, "subdivide-first visible in only %zu of %zu", discSubFirst, refined);
    PXG_CHECK_MSG(discNoScaleRot > N / 8, "unrotated scale visible in only %zu cases", discNoScaleRot);
    dev->destroyBuffer(bIn);
    dev->destroyBuffer(bOut);
    dev->destroyBuffer(bInd);
    dev->destroyBuffer(bVert);
}

// ---------------------------------------------------------------------------
// triangleRadiusSquared, sdfRadiusSquared, PxMax, transformInv, getInverse.
// ---------------------------------------------------------------------------
PXG_TEST(npTriTriGeom_math)
{
    Rng rng(831003u);

    // PxMax operand pairs where selectMax (`a > b ? a : b`) differs from every
    // plausible stand-in: NaN on either side, the two zeros both ways, equal
    // values, infinities.
    const float NZ = -0.0f;
    const float pairs[][2] = {
        { QNAN, 1.f }, { 1.f, QNAN }, { QNAN, -2.f }, { -2.f, QNAN }, { QNAN, QNAN },
        { 0.f, NZ }, { NZ, 0.f }, { 0.f, 0.f }, { NZ, NZ },
        { 3.f, 3.f }, { -PINF, 5.f }, { 5.f, PINF }, { PINF, QNAN }, { QNAN, -PINF },
    };
    const uint32_t NP = uint32_t(sizeof(pairs) / sizeof(pairs[0]));

    const uint32_t N = 4096u;
    std::vector<MathCase> cs(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        MathCase s{};
        s.v0 = rng.point(3.f); s.v1 = rng.point(3.f); s.v2 = rng.point(3.f);
        s.boxLo = rng.point(4.f); s.boxHi = rng.point(4.f);
        s.tp = rng.point(5.f); s.tq = rng.unitQuat();
        s.v = rng.point(6.f);
        if (i < 4u * NP) { s.v0.w = pairs[i % NP][0]; s.v1.w = pairs[i % NP][1]; }
        else { s.v0.w = rng.f(-3.f, 3.f); s.v1.w = (i % 5u == 0u) ? s.v0.w : rng.f(-3.f, 3.f); }

        // Non-finite triangles: the only inputs on which the OPERAND ORDER of the
        // two PxMax calls shows.  selectMax passes b through when a is NaN and
        // returns NaN when b is, so which edge sits in which slot decides whether
        // a NaN edge is swallowed or propagated.  NaN in v1 makes e01 and e12 NaN;
        // in v2, e12 and e20; in v0, e20 and e01; a shared inf makes the edge
        // between the two vertices NaN (inf - inf) and the other two inf.
        switch (i % 32u)
        {
        case 3: s.v1.x = PINF; s.v2.x = PINF; break;
        case 7: s.v0.x = PINF; s.v1.x = PINF; break;
        case 11: s.v2.x = PINF; s.v0.x = PINF; break;
        case 15: s.v1.y = QNAN; break;
        case 29: s.v2.z = QNAN; break;
        case 31: s.v0.z = QNAN; break;
        case 19: { const float w0 = s.v0.w; s.v0 = s.v1; s.v0.w = w0; break; }   // a degenerate edge: exactly 0
        case 23: s.v0 = F4{ 0, 0, 0, s.v0.w }; s.v1 = F4{ 1, 0, 0, s.v1.w }; s.v2 = F4{ 0, 1, 0, 0 }; break;   // ties
        case 27: s.tq = F4{ 0.f, 0.f, 0.f, 1.f }; break;
        // Signed zeros in the two inverses: tp == 0 and v == tp, under identity
        // and under a quaternion whose products cancel exactly.  Without these no
        // output component is ever zero, and three spellings that differ from the
        // engine only in the sign of a zero pass (see the discriminators below).
        case 13: s.tp = F4{ 0.f, 0.f, 0.f, 0.f }; s.tq = F4{ 0.5f, -0.5f, 0.5f, 0.5f }; break;
        case 17: s.v = s.tp; s.tq = F4{ 0.5f, -0.5f, 0.5f, 0.5f }; break;
        case 21: s.tp = F4{ 0.f, 0.f, 0.f, 0.f }; s.tq = F4{ 0.f, 0.f, 0.f, 1.f }; break;
        case 25: s.v = s.tp; break;
        default: break;
        }
        cs[i] = s;
    }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<U4> outInit(N * 4u, U4{ 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu });
    PxgBufferHandle bIn = createBufferFrom(dev, cs);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    PXG_CHECK(runKernel("ttgMathProbe", N, dev->getBufferDeviceAddress(bIn), dev->getBufferDeviceAddress(bOut)));
    const std::vector<U4> got = downloadBuffer<U4>(dev, bOut, N * 4u);

    size_t mmTri = 0, mmSdf = 0, mmMax = 0, mmInv = 0, mmGetInv = 0;
    size_t discOrder = 0, discChain = 0, discFold = 0, discMaxBuiltin = 0;
    size_t discNegOutside = 0, discInvAsTinv = 0, discConjRotate = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const MathCase& s = cs[i];
        const PxVec3 v0 = v3(s.v0), v1 = v3(s.v1), v2 = v3(s.v2);
        const PxReal tri = refTriangleRadiusSquared(v0, v1, v2);
        const PxReal sdf = refSdfRadiusSquared(v3(s.boxLo), v3(s.boxHi));
        const PxReal mx = physx::PxMax(s.v0.w, s.v1.w);
        const PxTransform t(v3(s.tp), q4(s.tq));
        const PxVec3 ti = t.transformInv(v3(s.v));
        const PxTransform inv = t.getInverse();

        const U4* o = &got[i * 4u];
        if (!sameBits(o[0].x, tri)) { if (mmTri < 4) std::printf("  math case %u: triRadSq 0x%08x want 0x%08x\n", i, o[0].x, bitsOf(tri)); ++mmTri; }
        if (!sameBits(o[0].y, sdf)) ++mmSdf;
        if (!sameBits(o[0].z, mx) || o[0].w != 0u)
        {
            if (mmMax < 4) std::printf("  math case %u: PxMax(0x%08x, 0x%08x) = 0x%08x want 0x%08x\n",
                                       i, bitsOf(s.v0.w), bitsOf(s.v1.w), o[0].z, bitsOf(mx));
            ++mmMax;
        }
        if (!sameVec(o[1], ti)) ++mmInv;
        if (!sameVec(o[2], inv.p) || !sameBits(o[3].x, inv.q.x) || !sameBits(o[3].y, inv.q.y) ||
            !sameBits(o[3].z, inv.q.z) || !sameBits(o[3].w, inv.q.w))
            ++mmGetInv;

        // Discriminators.
        const PxReal e01 = (v1 - v0).magnitudeSquared(), e12 = (v2 - v1).magnitudeSquared(), e20 = (v0 - v2).magnitudeSquared();
        // The RE-ASSOCIATION PxMax(e01, PxMax(e12, e20)) is not a discriminator:
        // it is equivalent for every triangle.  selectMax is associative except
        // when the MIDDLE operand is NaN, and e12 NaN forces e01 NaN, e20 NaN, or
        // e01 = e20 = inf -- on each of which both spellings agree (brute-forced
        // over 10^6 two-component triangles from {0, -0, 1, -1, 2, +-3e38, +-inf,
        // NaN} as well; RUNTIME_STATUS P9.83b-8 (1/n)).  Asserted per case rather
        // than assumed, so the claim cannot rot silently.
        const PxReal reassoc = physx::PxMax(e01, physx::PxMax(e12, e20));
        PXG_CHECK(sameBits(bitsOf(reassoc), tri));
        // ...and so is swapping the INNER pair (same argument: a lone NaN among
        // e01/e12 drags e20 to NaN or both others to inf).
        const PxReal innerSwap = physx::PxMax(physx::PxMax(e12, e01), e20);
        PXG_CHECK(sameBits(bitsOf(innerSwap), tri));
        // What IS observable: the operand order (outer slots swapped), and a
        // maxNum chain in place of selectMax.
        const PxReal outerSwap = physx::PxMax(e20, physx::PxMax(e01, e12));
        if (!sameBits(bitsOf(outerSwap), tri)) ++discOrder;
        const PxReal chain = std::fmax(std::fmax(e01, e12), e20);
        if (!sameBits(bitsOf(chain), tri)) ++discChain;
        if (differs(inv.transform(v3(s.v)), ti)) ++discFold;
        // The sign-of-zero spellings, each bitwise equal to the engine unless a
        // component of the result is zero.
        const PxQuat q = q4(s.tq);
        const PxVec3 p = v3(s.tp);
        if (differs(-q.rotateInv(p), inv.p)) ++discNegOutside;                    // -rotateInv(p)
        if (differs(t.transformInv(PxVec3(0.f)), inv.p)) ++discInvAsTinv;          // getInverse().p = transformInv(0)
        if (differs(q.getConjugate().rotate(v3(s.v) - p), ti)) ++discConjRotate;   // PxTransform.h:167-168's spelling
        const float fm = std::fmax(s.v0.w, s.v1.w);   // IEEE maxNum, what a builtin max tends to be
        if (!(std::isnan(fm) && std::isnan(mx)) && bitsOf(fm) != bitsOf(mx)) ++discMaxBuiltin;
    }

    std::printf("  math: %u cases, operand-order discriminating %zu, maxNum-chain discriminating %zu, "
                "inverse-fold discriminating %zu, builtin-max discriminating %zu; signed-zero: "
                "-rotateInv(p) %zu, transformInv(0) %zu, conjugate-rotate %zu\n",
                N, discOrder, discChain, discFold, discMaxBuiltin, discNegOutside, discInvAsTinv, discConjRotate);
    PXG_CHECK_MSG(mmTri == 0, "%zu triangleRadiusSquared mismatches", mmTri);
    PXG_CHECK_MSG(mmSdf == 0, "%zu sdfRadiusSquared mismatches", mmSdf);
    PXG_CHECK_MSG(mmMax == 0, "%zu PxMax mismatches", mmMax);
    PXG_CHECK_MSG(mmInv == 0, "%zu transformInv mismatches", mmInv);
    PXG_CHECK_MSG(mmGetInv == 0, "%zu getInverse mismatches", mmGetInv);
    PXG_CHECK_MSG(discOrder >= 100, "only %zu cases can tell the PxMax operand order", discOrder);
    PXG_CHECK_MSG(discChain >= 100, "only %zu cases can tell selectMax from a maxNum chain", discChain);
    PXG_CHECK_MSG(discFold > N / 4, "only %zu cases can tell transformInv from getInverse().transform", discFold);
    PXG_CHECK_MSG(discMaxBuiltin >= 12, "only %zu cases can tell selectMax from maxNum", discMaxBuiltin);
    PXG_CHECK_MSG(discNegOutside >= 20, "only %zu cases can tell rotateInv(-p) from -rotateInv(p)", discNegOutside);
    PXG_CHECK_MSG(discInvAsTinv >= 20, "only %zu cases can tell getInverse().p from transformInv(0)", discInvAsTinv);
    PXG_CHECK_MSG(discConjRotate >= 20, "only %zu cases can tell rotateInv from conjugate().rotate", discConjRotate);
    dev->destroyBuffer(bIn);
    dev->destroyBuffer(bOut);
}
