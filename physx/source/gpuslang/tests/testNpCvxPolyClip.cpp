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

// Verification of npCvxPolyClip.slang: the convex-vs-convex polyClip() from
// cudaGJKEPA.cu.  The C++ reference transcribes the same 32-lane serialised
// warp algorithm; the emitted contacts (count + ordered pos/sep) are compared
// (positions to tolerance), with groups whose in/out ballots, sep gates or
// mDote0 sit within a small margin of a boundary skipped (measure-zero flips).

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct V3 { float x, y, z; };
struct PolyClipIn { uint32_t baseV, nbV0, nbV1, pad0; F4 plane0, plane1, axisCD; };
struct PolyClipOut { F4 cts[32]; uint32_t nbContacts, pad0, pad1, pad2; };

const uint32_t MAXVERT = 64; // 32 v0 + 32 v1
const float MAXF = 3.4028235e38f;
const float TIE = 1e-3f;

static inline V3 add(const V3& a, const V3& b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 mul(const V3& a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 crossp(const V3& a, const V3& b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static inline V3 nrm(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? mul(a, 1.f / l) : a; }
static uint32_t popc(uint32_t m) { uint32_t c = 0; while (m) { c += m & 1u; m >>= 1; } return c; }
static uint32_t lowIdx(uint32_t m) { uint32_t i = 0; while (!(m & (1u << i))) ++i; return i; }

// exact transcription of the shader; fills out, sets marg on near-boundary cases.
static void refPolyClip(const PolyClipIn& s, const std::vector<F4>& verts, PolyClipOut& o, bool& marg)
{
    const uint32_t nbV0 = s.nbV0, nbV1 = s.nbV1, bv = s.baseV;
    const V3 p0n{ s.plane0.x, s.plane0.y, s.plane0.z }; const float p0d = s.plane0.w;
    const V3 p1n{ s.plane1.x, s.plane1.y, s.plane1.z }; const float p1d = s.plane1.w;
    const V3 axis{ s.axisCD.x, s.axisCD.y, s.axisCD.z }; const float contactDist = s.axisCD.w;

    V3 v0[32], v1[32];
    for (uint32_t L = 0; L < 32; ++L) { v0[L] = V3{ verts[bv + L].x, verts[bv + L].y, verts[bv + L].z }; v1[L] = V3{ verts[bv + 32 + L].x, verts[bv + 32 + L].y, verts[bv + 32 + L].z }; }

    V3 ctsPos[32]; float ctsSep[32];
    for (uint32_t i = 0; i < 32; ++i) { ctsPos[i] = V3{ 0, 0, 0 }; ctsSep[i] = 0; }
    uint32_t nbContacts = 0;

    const float p1DotAxis = dot3(p1n, axis);
    float s0[32], s1[32];
    for (uint32_t L = 0; L < 32; ++L) { s0[L] = -(dot3(p1n, v0[L]) + p1d) / p1DotAxis; s1[L] = dot3(p0n, v1[L]) + p0d; }

    V3 e0[32], e0planeN[32]; float e0planeD[32];
    for (uint32_t L = 0; L < 32; ++L) { const uint32_t nxt = (L + 1 == nbV0) ? 0 : ((L + 1) & 31); e0[L] = sub(v0[nxt], v0[L]); e0planeN[L] = crossp(e0[L], axis); e0planeD[L] = dot3(e0planeN[L], v0[L]); }

    uint32_t in1[32], out1[32];
    for (uint32_t L = 0; L < 32; ++L)
    {
        uint32_t i1 = 0, o1 = 0;
        for (uint32_t j = 0; j < nbV1; ++j)
        {
            const float tt = dot3(v1[j], e0planeN[L]) - e0planeD[L];
            if (L < nbV0 && std::fabs(tt) < TIE) marg = true;
            if (tt < 0.f) i1 |= (1u << j);
            if (tt > 0.f) o1 |= (1u << j);
        }
        in1[L] = i1; out1[L] = o1;
    }
    for (uint32_t L = 0; L < 32; ++L) if (L >= nbV0) { in1[L] = 0; out1[L] = 0; }

    uint32_t b = 0xffffffffu;
    for (uint32_t L = 0; L < 32; ++L) b &= ~out1[L];

    auto addContacts = [&](uint32_t flags, const V3* pos, const float* sep, float maxSep) {
        uint32_t fmask = 0;
        for (uint32_t L = 0; L < 32; ++L) if ((flags & (1u << L)) && sep[L] < maxSep) fmask |= (1u << L);
        // margin: any flagged lane's sep near maxSep
        for (uint32_t L = 0; L < 32; ++L) if ((flags & (1u << L)) && std::fabs(sep[L] - maxSep) < TIE) marg = true;
        for (uint32_t L = 0; L < 32; ++L) if (fmask & (1u << L)) { const uint32_t idx = popc(fmask & ((1u << L) - 1)) + nbContacts; if (idx < 32) { ctsPos[idx] = pos[L]; ctsSep[idx] = sep[L]; } }
        nbContacts += popc(fmask);
    };

    {
        const uint32_t mask = b & ((nbV1 >= 32) ? 0xffffffffu : ((1u << nbV1) - 1));
        V3 pos[32]; float sep[32];
        for (uint32_t L = 0; L < 32; ++L) { pos[L] = sub(v1[L], mul(axis, s1[L])); sep[L] = s1[L]; }
        addContacts(mask, pos, sep, contactDist);
    }

    uint32_t anyOut1 = 0; for (uint32_t L = 0; L < 32; ++L) anyOut1 |= out1[L];
    if (anyOut1 == 0) { for (uint32_t i = 0; i < 32; ++i) o.cts[i] = F4{ ctsPos[i].x, ctsPos[i].y, ctsPos[i].z, ctsSep[i] }; o.nbContacts = nbContacts; return; }

    uint32_t e1cross[32];
    for (uint32_t L = 0; L < 32; ++L) { const uint32_t i1 = in1[L] | (in1[L] << nbV1); const uint32_t o1 = out1[L] | (out1[L] << nbV1); e1cross[L] = (i1 & (o1 >> 1)) | ((i1 >> 1) & o1); }

    V3 e1[32], e1planeN[32]; float e1planeD[32];
    for (uint32_t L = 0; L < 32; ++L) { const uint32_t nxt = (L + 1 == nbV1) ? 0 : ((L + 1) & 31); e1[L] = sub(v1[nxt], v1[L]); e1planeN[L] = crossp(axis, e1[L]); e1planeD[L] = dot3(e1planeN[L], v1[L]); }

    uint32_t in0[32], out0[32];
    for (uint32_t L = 0; L < 32; ++L) { in0[L] = 0; out0[L] = 0; }
    for (uint32_t j = 0; j < nbV0; ++j)
    {
        uint32_t inMask = 0, outMask = 0;
        for (uint32_t L = 0; L < 32; ++L)
        {
            const float tt = dot3(v0[j], e1planeN[L]) - e1planeD[L];
            if (L < nbV1 && std::fabs(tt) < TIE) marg = true;
            if (tt < 0.f) inMask |= (1u << L);
            if (tt > 0.f) outMask |= (1u << L);
        }
        in0[j] = inMask; out0[j] = outMask;
    }
    const uint32_t mask1 = (nbV1 >= 32) ? 0xffffffffu : ((1u << nbV1) - 1);
    for (uint32_t L = 0; L < 32; ++L) { in0[L] &= mask1; out0[L] &= mask1; }

    {
        uint32_t ballotNotOut0 = 0; for (uint32_t L = 0; L < 32; ++L) if (out0[L] == 0) ballotNotOut0 |= (1u << L);
        const uint32_t mask = ballotNotOut0 & ((nbV0 >= 32) ? 0xffffffffu : ((1u << nbV0) - 1));
        V3 pos[32]; float sep[32];
        for (uint32_t L = 0; L < 32; ++L) { pos[L] = v0[L]; sep[L] = s0[L]; }
        addContacts(mask, pos, sep, contactDist);
    }

    uint32_t anyOut0 = 0; for (uint32_t L = 0; L < 32; ++L) anyOut0 |= out0[L];
    if (anyOut0 == 0) { for (uint32_t i = 0; i < 32; ++i) o.cts[i] = F4{ ctsPos[i].x, ctsPos[i].y, ctsPos[i].z, ctsSep[i] }; o.nbContacts = nbContacts; return; }

    uint32_t e0cross[32]; float s0d[32];
    for (uint32_t L = 0; L < 32; ++L) { const uint32_t nxt = (L + 1 == nbV0) ? 0 : ((L + 1) & 31); e0cross[L] = (in0[L] & out0[nxt]) | (in0[nxt] & out0[L]); s0d[L] = s0[nxt] - s0[L]; }

    uint32_t c[32]; uint32_t a = 0;
    for (uint32_t L = 0; L < 32; ++L) { c[L] = e0cross[L] & e1cross[L]; a |= c[L]; }

    for (; a != 0; a &= (a - 1))
    {
        const uint32_t index = lowIdx(a);
        const V3 m = e1planeN[index];
        const uint32_t lsb = a & (uint32_t)(-(int32_t)a);
        V3 pt[32]; float sepA[32];
        for (uint32_t L = 0; L < 32; ++L)
        {
            const float mDote0 = dot3(m, e0[L]);
            if ((c[L] & lsb) && std::fabs(mDote0) < TIE) marg = true;
            const float lambda = -(dot3(m, v0[L]) - e1planeD[index]) / mDote0;
            pt[L] = add(v0[L], mul(e0[L], lambda));
            sepA[L] = (mDote0 != 0.f) ? (s0[L] + lambda * s0d[L]) : MAXF;
        }
        uint32_t flags = 0;
        for (uint32_t L = 0; L < 32; ++L) if (c[L] & lsb) flags |= (1u << L);
        addContacts(flags, pt, sepA, contactDist);
    }

    for (uint32_t i = 0; i < 32; ++i) o.cts[i] = F4{ ctsPos[i].x, ctsPos[i].y, ctsPos[i].z, ctsSep[i] };
    o.nbContacts = nbContacts;
}

static bool eqC(const F4& r, const F4& v) { return almostEqual(v.x, r.x, 5e-4f, 5e-5f) && almostEqual(v.y, r.y, 5e-4f, 5e-5f) && almostEqual(v.z, r.z, 5e-4f, 5e-5f) && almostEqual(v.w, r.w, 5e-4f, 5e-4f); }

PXG_TEST(npCvxPolyClip_clip)
{
    std::mt19937 rng(145001);
    const uint32_t N = 5000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<PolyClipIn> in(N);
    std::vector<F4> verts(size_t(N) * MAXVERT, F4{ 0, 0, 0, 0 });
    std::vector<uint8_t> skip(N, 0);
    std::vector<PolyClipOut> ref(N);
    size_t nSkip = 0, withC = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t nbV0 = 3 + (rng() % 6), nbV1 = 3 + (rng() % 6);
        const uint32_t bv = i * MAXVERT;
        const V3 axis = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
        V3 u = nrm(crossp(axis, V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }));
        if (dot3(u, u) < 0.5f) u = nrm(crossp(axis, V3{ 1, 0, 0 }));
        const V3 vv = crossp(axis, u); // u x vv = axis, CCW around axis

        const V3 base{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        const float z0 = rf(-0.2f, 0.2f), z1 = z0 + rf(0.02f, 0.4f);
        const V3 c0 = add(base, mul(axis, z0));
        const float r0 = rf(0.6f, 1.2f), r1 = rf(0.6f, 1.2f);
        const V3 off = add(mul(u, rf(-0.7f, 0.7f)), mul(vv, rf(-0.7f, 0.7f)));
        const V3 c1 = add(add(base, mul(axis, z1)), off);
        const float ph0 = rf(0, 6.28f), ph1 = rf(0, 6.28f);

        V3 poly0[32], poly1[32];
        for (uint32_t k = 0; k < nbV0; ++k) { const float th = ph0 + 6.2831853f * k / nbV0; poly0[k] = add(c0, add(mul(u, r0 * std::cos(th)), mul(vv, r0 * std::sin(th)))); }
        for (uint32_t k = 0; k < nbV1; ++k) { const float th = ph1 + 6.2831853f * k / nbV1; poly1[k] = add(c1, add(mul(u, r1 * std::cos(th)), mul(vv, r1 * std::sin(th)))); }
        // inactive lanes: fixed to the centre (deterministic; masked out)
        for (uint32_t k = nbV0; k < 32; ++k) poly0[k] = c0;
        for (uint32_t k = nbV1; k < 32; ++k) poly1[k] = c1;
        for (uint32_t k = 0; k < 32; ++k) { verts[bv + k] = F4{ poly0[k].x, poly0[k].y, poly0[k].z, 0 }; verts[bv + 32 + k] = F4{ poly1[k].x, poly1[k].y, poly1[k].z, 0 }; }

        // planes: tilted slightly from axis so seps vary per vertex
        const V3 tilt0 = add(mul(u, rf(-0.15f, 0.15f)), mul(vv, rf(-0.15f, 0.15f)));
        const V3 tilt1 = add(mul(u, rf(-0.15f, 0.15f)), mul(vv, rf(-0.15f, 0.15f)));
        const V3 n0 = nrm(add(axis, tilt0)), n1 = nrm(add(axis, tilt1));
        const F4 plane0{ n0.x, n0.y, n0.z, -dot3(n0, c0) };
        const F4 plane1{ n1.x, n1.y, n1.z, -dot3(n1, c1) };
        const float contactDist = rf(0.05f, 0.6f);

        PolyClipIn s{};
        s.baseV = bv; s.nbV0 = nbV0; s.nbV1 = nbV1;
        s.plane0 = plane0; s.plane1 = plane1; s.axisCD = { axis.x, axis.y, axis.z, contactDist };
        in[i] = s;

        bool marg = false;
        PolyClipOut r{};
        refPolyClip(s, verts, r, marg);
        ref[i] = r;
        skip[i] = marg ? 1u : 0u;
        if (marg) { ++nSkip; continue; }
        if (r.nbContacts > 0) ++withC;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<PolyClipOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bV = createBufferFrom(dev, verts);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct PolyClipParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxPolyClip.spv", "cvxPolyClip",
             { (uint32_t)sizeof(PolyClipIn), (uint32_t)sizeof(F4), (uint32_t)sizeof(PolyClipOut) }, sizeof(PolyClipParams));
    k.dispatch(GROUPS, { bIn, bV, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<PolyClipOut>(dev, bOut, N);
    size_t mmN = 0, mmC = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        const uint32_t rc = ref[i].nbContacts, gc = g[i].nbContacts;
        if (rc != gc) { ++mmN; continue; }
        const uint32_t lim = rc < 32 ? rc : 32;
        for (uint32_t j = 0; j < lim; ++j) if (!eqC(ref[i].cts[j], g[i].cts[j])) { ++mmC; break; }
    }
    PXG_CHECK_MSG(mmN == 0, "polyClip: %zu contact-count mismatches", mmN);
    PXG_CHECK_MSG(mmC == 0, "polyClip: %zu contact mismatches", mmC);
    PXG_CHECK_MSG(nSkip < (size_t)N * 4 / 5, "polyClip: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(withC > N / 20, "polyClip: contact coverage thin (%zu)", withC);

    for (PxgBufferHandle h : { bIn, bV, bOut }) dev->destroyBuffer(h);
}

} // namespace
