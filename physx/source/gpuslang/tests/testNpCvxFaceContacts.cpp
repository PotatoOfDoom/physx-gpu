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

// Verification of npCvxFaceContacts.slang: generatedFaceContacts() from
// cudaGJKEPA.cu.  Two GJK witness verts cast vertex-space rays through a convex
// polyhedron (box / tilted hull); each ray that enters within contactDistance
// emits a contact (va = aToB.transformInv(vA), vb = vA - dir*tEnter, normalPen
// = (dir, tEnter)).  The C++ reference transcribes the CUDA (shape2Vertex +
// serialised intersectRayPolyhedron + transformInv); the contact count and
// present contacts are compared (to tolerance) with boundary-margin groups
// skipped.

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
struct FaceContactsIn { F4 vA0, vA1, gjkDir, scale1, rot1, aToBq, aToBp; };
struct FaceContactsOut { F4 va0, vb0, np0, va1, vb1, np1, meta; };

const uint32_t MAXPOLYS = 40;
const float PXG_MAXF = 3.4028235e38f;
const float MARG = 1e-4f;

static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 add(const V3& a, const V3& b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline V3 mul(const V3& a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static inline V3 divc(const V3& a, const V3& b) { return V3{ a.x / b.x, a.y / b.y, a.z / b.z }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 crossp(const V3& a, const V3& b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static inline V3 nrm(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? mul(a, 1.f / l) : a; }

static V3 quatRotate(const F4& q, const V3& v)
{
    const float vx = 2.f * v.x, vy = 2.f * v.y, vz = 2.f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = q.x * vx + q.y * vy + q.z * vz;
    return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
               vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
               vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2 };
}
static V3 quatRotateInv(const F4& q, const V3& v)
{
    const float vx = 2.f * v.x, vy = 2.f * v.y, vz = 2.f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = q.x * vx + q.y * vy + q.z * vz;
    return V3{ vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * dot2,
               vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * dot2,
               vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * dot2 };
}
static V3 shape2Vertex(const V3& v, const V3& scale, const F4& rot) { return quatRotateInv(rot, divc(quatRotate(rot, v), scale)); }
static V3 transformInvPoint(const F4& q, const V3& p, const V3& v) { return quatRotateInv(q, sub(v, p)); }

// serialised intersectRayPolyhedron (block-of-32 warp reduction)
static bool intersectRayPolyhedron(const V3& a, const V3& dir, const std::vector<F4>& planes,
                                   uint32_t nbPolygons, float& tEnter, float& tExit, bool& marg)
{
    const float eps = 1e-7f;
    float tFirst = 0.f, tLast = PXG_MAXF;
    tEnter = 0.f; tExit = 0.f;
    const uint32_t WARP = 32;
    for (uint32_t i = 0; i < nbPolygons; i += WARP)
    {
        bool blockSep = false;
        float newFirst = tFirst, newLast = tLast;
        for (uint32_t lane = 0; lane < WARP; ++lane)
        {
            const uint32_t k = i + lane;
            if (k >= nbPolygons) continue;
            const F4& pl = planes[k];
            const V3 n{ pl.x, pl.y, pl.z };
            const float d = pl.w;
            const float denom = dot3(n, dir);
            const float distToPlane = dot3(n, a) + d;
            if (std::fabs(std::fabs(denom) - eps) < 1e-8f) marg = true;
            if (std::fabs(distToPlane) < MARG && std::fabs(denom) < eps) marg = true;
            if (std::fabs(denom) < eps)
            {
                if (distToPlane > 0.f) blockSep = true;
            }
            else
            {
                const float tTemp = -distToPlane / denom;
                const bool con = 0.f > denom;
                const bool con0 = tTemp > tFirst;
                const bool con1 = tLast > tTemp;
                if (std::fabs(tTemp - tFirst) < MARG) marg = true;
                if (tLast < PXG_MAXF && std::fabs(tLast - tTemp) < MARG) marg = true;
                if (con && con0) newFirst = std::fmax(newFirst, tTemp);
                if (con1 && !con) newLast = std::fmin(newLast, tTemp);
            }
        }
        if (blockSep) return false;
        tFirst = newFirst; tLast = newLast;
        if (std::fabs(tFirst - tLast) < MARG) marg = true;
        if (tFirst > tLast) return false;
    }
    tEnter = tFirst; tExit = tLast;
    return true;
}

// Exact transcription of generatedFaceContacts. Fills up to 2 contacts.
static uint32_t refFaceContacts(const FaceContactsIn& s, const std::vector<F4>& planes, uint32_t nbPolygons,
                                FaceContactsOut& o, bool& marg)
{
    const V3 vA0{ s.vA0.x, s.vA0.y, s.vA0.z };
    const V3 vA1{ s.vA1.x, s.vA1.y, s.vA1.z };
    const V3 gjkDir{ s.gjkDir.x, s.gjkDir.y, s.gjkDir.z };
    const float inflatedRadius = s.gjkDir.w;
    const V3 scale1{ s.scale1.x, s.scale1.y, s.scale1.z };
    const V3 aToBp{ s.aToBp.x, s.aToBp.y, s.aToBp.z };

    const V3 dir = mul(gjkDir, -1.f);
    const V3 vertexSpaceDir = shape2Vertex(mul(dir, -1.f), scale1, s.rot1);
    const V3 p0 = shape2Vertex(vA0, scale1, s.rot1);
    const V3 p1 = shape2Vertex(vA1, scale1, s.rot1);

    uint32_t numContacts = 0;
    float te, tx;
    if (intersectRayPolyhedron(p0, vertexSpaceDir, planes, nbPolygons, te, tx, marg))
    {
        if (std::fabs(inflatedRadius - te) < MARG) marg = true;
        if (inflatedRadius >= te)
        {
            const V3 va = transformInvPoint(s.aToBq, aToBp, vA0);
            const V3 vb = sub(vA0, mul(dir, te));
            o.va0 = F4{ va.x, va.y, va.z, 0 }; o.vb0 = F4{ vb.x, vb.y, vb.z, 0 };
            o.np0 = F4{ dir.x, dir.y, dir.z, te };
            numContacts++;
        }
    }
    if (intersectRayPolyhedron(p1, vertexSpaceDir, planes, nbPolygons, te, tx, marg))
    {
        if (std::fabs(inflatedRadius - te) < MARG) marg = true;
        if (inflatedRadius >= te)
        {
            const V3 va = transformInvPoint(s.aToBq, aToBp, vA1);
            const V3 vb = sub(vA1, mul(dir, te));
            const F4 vaf{ va.x, va.y, va.z, 0 }, vbf{ vb.x, vb.y, vb.z, 0 }, npf{ dir.x, dir.y, dir.z, te };
            if (numContacts == 0) { o.va0 = vaf; o.vb0 = vbf; o.np0 = npf; }
            else { o.va1 = vaf; o.vb1 = vbf; o.np1 = npf; }
            numContacts++;
        }
    }
    return numContacts;
}

static uint32_t boxPlanes(const V3& c, const V3& u, const V3& v, const V3& w,
                          float hu, float hv, float hw, std::vector<F4>& out)
{
    const float cu = dot3(u, c), cv = dot3(v, c), cw = dot3(w, c);
    out.push_back(F4{ u.x, u.y, u.z, -cu - hu });
    out.push_back(F4{ -u.x, -u.y, -u.z, cu - hu });
    out.push_back(F4{ v.x, v.y, v.z, -cv - hv });
    out.push_back(F4{ -v.x, -v.y, -v.z, cv - hv });
    out.push_back(F4{ w.x, w.y, w.z, -cw - hw });
    out.push_back(F4{ -w.x, -w.y, -w.z, cw - hw });
    return 6;
}

static F4 randQuat(std::mt19937& rng, float maxAng)
{
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    V3 ax = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
    const float a = rf(-maxAng, maxAng);
    const float hs = std::sin(a * 0.5f), hc = std::cos(a * 0.5f);
    return F4{ ax.x * hs, ax.y * hs, ax.z * hs, hc };
}

static bool eqPt(const F4& r, const F4& v, float wt)
{
    return almostEqual(v.x, r.x, 1e-4f, 1e-4f) && almostEqual(v.y, r.y, 1e-4f, 1e-4f) &&
           almostEqual(v.z, r.z, 1e-4f, 1e-4f) && almostEqual(v.w, r.w, 1e-4f, wt);
}

PXG_TEST(npCvxFaceContacts_generate)
{
    std::mt19937 rng(136001);
    const uint32_t N = 14000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<FaceContactsIn> in(N);
    std::vector<F4> planesFlat(size_t(N) * MAXPOLYS, F4{ 0, 0, 0, 0 });
    std::vector<uint8_t> skip(N, 0);
    std::vector<FaceContactsOut> ref(N);
    size_t nSkip = 0, c0 = 0, c1 = 0, c2 = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        // convex polyhedron (in vertex space) as a box around the origin
        const bool ident = (i % 2 == 0);
        F4 rot1 = ident ? F4{ 0, 0, 0, 1 } : randQuat(rng, 0.6f);
        V3 scale1 = ident ? V3{ 1, 1, 1 } : V3{ rf(0.7f, 1.4f), rf(0.7f, 1.4f), rf(0.7f, 1.4f) };

        // box in vertex space
        V3 bu = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
        V3 btmp = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
        V3 bw = nrm(crossp(bu, btmp));
        V3 bv = nrm(crossp(bw, bu));
        std::vector<F4> planes;
        uint32_t nb = boxPlanes(V3{ 0, 0, 0 }, bu, bv, bw, rf(0.5f, 1.2f), rf(0.5f, 1.2f), rf(0.5f, 1.2f), planes);

        // gjk direction (contact normal, shape space of A)
        V3 gjkDir = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
        // dir = -gjkDir ; vertexSpaceDir = shape2Vertex(gjkDir)
        // Aim the witness verts so that, in vertex space, the ray (p_i, vSDir)
        // heads into the box.  For identity scale/rot p_i = vA_i, vSDir = gjkDir,
        // so pick vA0 outside the box, dir pointing toward origin.
        V3 vd = shape2Vertex(gjkDir, scale1, rot1);           // vertexSpaceDir
        vd = nrm(vd);
        // place p0 so the ray p0 + t*vd passes through the origin region
        V3 p0v = add(mul(vd, -rf(1.5f, 3.5f)), mul(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }, 0.4f));
        // vA0 = vertex2Shape(p0v) so shape2Vertex(vA0) = p0v (approx round-trip)
        // vertex2Shape(p) = rotInv(rot(p) * scale)
        auto vertex2Shape = [&](const V3& p) { return quatRotateInv(rot1, V3{ quatRotate(rot1, p).x * scale1.x, quatRotate(rot1, p).y * scale1.y, quatRotate(rot1, p).z * scale1.z }); };
        V3 vA0 = vertex2Shape(p0v);
        // vA1 near vA0 (an edge), sometimes offset enough to miss
        V3 vA1 = add(vA0, mul(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }, rf(0.1f, 1.2f)));

        const float contactDistance = rf(0.3f, 4.0f);

        F4 aToBq = randQuat(rng, 3.14159f);
        V3 aToBp{ rf(-3, 3), rf(-3, 3), rf(-3, 3) };

        FaceContactsIn s{};
        s.vA0 = { vA0.x, vA0.y, vA0.z, float(nb) };
        s.vA1 = { vA1.x, vA1.y, vA1.z, 0 };
        s.gjkDir = { gjkDir.x, gjkDir.y, gjkDir.z, contactDistance };
        s.scale1 = { scale1.x, scale1.y, scale1.z, 0 };
        s.rot1 = rot1;
        s.aToBq = aToBq;
        s.aToBp = { aToBp.x, aToBp.y, aToBp.z, 0 };
        in[i] = s;
        for (uint32_t f = 0; f < nb; ++f) planesFlat[size_t(i) * MAXPOLYS + f] = planes[f];

        FaceContactsOut r{}; bool marg = false;
        const uint32_t cnt = refFaceContacts(s, planes, nb, r, marg);
        r.meta.x = float(cnt);
        skip[i] = marg ? 1u : 0u; ref[i] = r;
        if (marg) { ++nSkip; continue; }
        if (cnt == 0) ++c0; else if (cnt == 1) ++c1; else ++c2;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<FaceContactsOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bPlanes = createBufferFrom(dev, planesFlat);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct FaceContactsParams { uint32_t numTests, maxPolys; } prm = { N, MAXPOLYS };
    Kernel k("npCvxFaceContacts.spv", "cvxFaceContacts",
             { (uint32_t)sizeof(FaceContactsIn), (uint32_t)sizeof(F4), (uint32_t)sizeof(FaceContactsOut) }, sizeof(FaceContactsParams));
    k.dispatch(GROUPS, { bIn, bPlanes, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<FaceContactsOut>(dev, bOut, N);
    size_t mmCount = 0, mmPt = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        const uint32_t rc = (uint32_t)(ref[i].meta.x + 0.5f), gc = (uint32_t)(g[i].meta.x + 0.5f);
        if (rc != gc) { ++mmCount; continue; }
        if (rc >= 1)
        {
            if (!eqPt(ref[i].va0, g[i].va0, 1e-4f)) ++mmPt;
            if (!eqPt(ref[i].vb0, g[i].vb0, 1e-4f)) ++mmPt;
            if (!eqPt(ref[i].np0, g[i].np0, 1e-4f)) ++mmPt;
        }
        if (rc >= 2)
        {
            if (!eqPt(ref[i].va1, g[i].va1, 1e-4f)) ++mmPt;
            if (!eqPt(ref[i].vb1, g[i].vb1, 1e-4f)) ++mmPt;
            if (!eqPt(ref[i].np1, g[i].np1, 1e-4f)) ++mmPt;
        }
    }
    PXG_CHECK_MSG(mmCount == 0, "faceContacts: %zu count mismatches", mmCount);
    PXG_CHECK_MSG(mmPt == 0, "faceContacts: %zu contact mismatches", mmPt);
    PXG_CHECK_MSG(nSkip < N / 2, "faceContacts: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(c0 > N / 30 && c1 > N / 30 && c2 > N / 30, "faceContacts: count coverage thin (%zu/%zu/%zu)", c0, c1, c2);

    for (PxgBufferHandle h : { bIn, bPlanes, bOut }) dev->destroyBuffer(h);
}

} // namespace
