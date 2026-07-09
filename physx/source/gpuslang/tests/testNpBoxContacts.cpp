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

// Verification of npBoxContacts.slang: calculateContacts — the box-box clipping
// core.  Each group of 4 lanes carries an incident polygon (4 points in box A's
// reference-face frame) + the reference-face half-extents, incident-face normal
// and contactDist; the kernel emits up to 3 contacts per lane (area / corner /
// edge-clip) in the CUDA's interleaved slot layout (contact c of lane L at
// 4*c+L).  The C++ reference transcribes calculateContacts including the two
// group-wide early exits.  Because the per-lane contact COUNT is an integer that
// turns on float predicates (penetration/area thresholds, point-in-polygon,
// segment-slab hit), groups where any predicate sits within a small margin of
// its decision boundary are skipped — a GPU-vs-CPU rounding flip there would
// change the count and is measure-zero for the fixed seed; skipped groups are
// asserted to stay a small minority.

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
struct ContactsIn { F4 faceNormal, misc; };

const float MAXF = 3.4028235e38f;

// --- helpers identical to the shader ---------------------------------------
bool contains4(const V3 v[4], float px, float py, const V3& bmin, const V3& bmax)
{
    if (px < bmin.x || px > bmax.x || py < bmin.y || py > bmax.y) return false;
    const float eps = 1e-6f, zero = 0.f;
    uint32_t ip = 0;
    bool inside = false, done = false;
    uint32_t j = 3;
    for (uint32_t i = 0; i < 4 && !done; ++i)
    {
        const float jx = v[j].x, jy = v[j].y, ix = v[i].x, iy = v[i].y;
        if ((px == jx && py == jy) || (px == ix && py == iy)) { inside = true; done = true; }
        else
        {
            const bool yflag0 = jy > py, yflag1 = iy > py;
            if (yflag0 != yflag1)
            {
                const float jix = ix - jx, jiy = iy - jy, jty = py - jy;
                const float part1 = jty * jix, part2 = (jx + eps) * jiy, part3 = px * jiy;
                const bool comp = jiy > zero;
                const float tmp = part1 + part2;
                const float comp1 = comp ? tmp : part3, comp2 = comp ? part3 : tmp;
                if (comp1 >= comp2) { if (ip == 1) { inside = false; done = true; } else ip++; }
            }
            j = i;
        }
    }
    if (!done) inside = ip > 0;
    return inside;
}

bool segAABB(const V3& p0, const V3& d, const V3& mx, const V3& mn, float& tmin, float& tmax)
{
    const float eps = 1e-6f;
    const float ax = std::fabs(d.x), ay = std::fabs(d.y), az = std::fabs(d.z);
    float tminf = 0.f, tmaxf = 1.f; tmin = 0.f; tmax = 1.f;
    const bool pX = eps > ax, pY = eps > ay, pZ = eps > az;
    if ((pX && (p0.x > mx.x || p0.x < mn.x)) || (pY && (p0.y > mx.y || p0.y < mn.y)) ||
        (pZ && (p0.z > mx.z || p0.z < mn.z))) return false;
    const float ox = 1.f / d.x, oy = 1.f / d.y, oz = 1.f / d.z;
    const float t1x = pX ? 0.f : (mn.x - p0.x) * ox, t1y = pY ? 0.f : (mn.y - p0.y) * oy, t1z = pZ ? 0.f : (mn.z - p0.z) * oz;
    const float t2x = pX ? MAXF : (mx.x - p0.x) * ox, t2y = pY ? MAXF : (mx.y - p0.y) * oy, t2z = pZ ? MAXF : (mx.z - p0.z) * oz;
    const float ft1 = std::max(std::min(t1x, t2x), std::max(std::min(t1y, t2y), std::min(t1z, t2z)));
    const float ft2 = std::min(std::max(t1x, t2x), std::min(std::max(t1y, t2y), std::max(t1z, t2z)));
    tminf = std::max(ft1, tminf); tmaxf = std::min(tmaxf, ft2);
    tmin = tminf; tmax = tmaxf;
    return !((tminf > tmaxf) || (tminf > 1.f));
}

float minEdgeDist2D(float px, float py, const V3 v[4])
{
    float best = 1e30f;
    uint32_t j = 3;
    for (uint32_t i = 0; i < 4; ++i)
    {
        const float ax = v[j].x, ay = v[j].y, bx = v[i].x, by = v[i].y;
        const float ex = bx - ax, ey = by - ay;
        const float len2 = ex * ex + ey * ey;
        float t = len2 > 0.f ? ((px - ax) * ex + (py - ay) * ey) / len2 : 0.f;
        t = std::max(0.f, std::min(1.f, t));
        const float dx = px - (ax + t * ex), dy = py - (ay + t * ey);
        best = std::min(best, std::sqrt(dx * dx + dy * dy));
        j = i;
    }
    return best;
}

// reference for one group.  Returns true if the group is "marginal" (any
// predicate near its decision boundary) — such groups are skipped by the caller.
bool refGroup(float eX, float eY, float cd, const V3& fn, const V3 pts[4],
              uint32_t count[4], F4 cout[12], bool& hadArea, bool& hadCorner, bool& hadEdge)
{
    hadArea = hadCorner = hadEdge = false;
    const float MARG = 1e-3f;
    const float extentX = 1.00001f * eX, extentY = 1.00001f * eY;
    const V3 ext{ extentX, extentY, MAXF };
    const V3 negExt{ -extentX, -extentY, -cd - 1e-7f };
    V3 bmin = pts[0], bmax = pts[0];
    for (int k = 1; k < 4; ++k)
    {
        bmin.x = std::min(bmin.x, pts[k].x); bmin.y = std::min(bmin.y, pts[k].y); bmin.z = std::min(bmin.z, pts[k].z);
        bmax.x = std::max(bmax.x, pts[k].x); bmax.y = std::max(bmax.y, pts[k].y); bmax.z = std::max(bmax.z, pts[k].z);
    }
    for (int i = 0; i < 12; ++i) cout[i] = F4{ 9, 9, 9, 9 };

    bool marginal = false;
    bool pPen[4], pArea[4];
    uint32_t cnt[4] = { 0, 0, 0, 0 };

    for (int L = 0; L < 4; ++L)
    {
        const float z = -pts[L].z;
        pPen[L] = cd > z;
        if (std::fabs(cd - z) < MARG) marginal = true;
        pArea[L] = false;
        if (pPen[L])
        {
            if (std::fabs(extentX - std::fabs(pts[L].x)) < MARG || std::fabs(extentY - std::fabs(pts[L].y)) < MARG) marginal = true;
            if (extentX >= std::fabs(pts[L].x) && extentY >= std::fabs(pts[L].y))
            {
                pArea[L] = true;
                cout[0 * 4 + L] = F4{ pts[L].x, pts[L].y, pts[L].z, z };
                cnt[L] = 1;
                hadArea = true;
            }
        }
    }
    const bool allArea = pArea[0] && pArea[1] && pArea[2] && pArea[3];

    bool pArea2[4] = { false, false, false, false };
    if (!allArea)
    {
        for (int L = 0; L < 4; ++L)
        {
            const float denom = fn.z;
            const float qx = (L == 0 || L == 3) ? -extentX : extentX;
            const float qy = (L & 2) ? -extentY : extentY;
            // margins: q0 near incident-polygon AABB boundary, or near a polygon edge
            if (std::fabs(qx - bmin.x) < MARG || std::fabs(bmax.x - qx) < MARG ||
                std::fabs(qy - bmin.y) < MARG || std::fabs(bmax.y - qy) < MARG) marginal = true;
            if (minEdgeDist2D(qx, qy, pts) < MARG) marginal = true;
            if (contains4(pts, qx, qy, bmin, bmax))
            {
                const float nom = fn.x * (pts[0].x - qx) + fn.y * (pts[0].y - qy) + fn.z * (pts[0].z);
                const float tval = nom / denom;
                if (std::fabs(cd - (-tval)) < MARG) marginal = true;
                if (cd > -tval)
                {
                    pArea2[L] = true;
                    cout[cnt[L] * 4 + L] = F4{ qx, qy, tval, -tval };
                    cnt[L]++;
                    hadCorner = true;
                }
            }
        }
    }
    const bool allArea2 = pArea2[0] && pArea2[1] && pArea2[2] && pArea2[3];
    const bool done2 = allArea || allArea2;

    if (!done2)
    {
        for (int L = 0; L < 4; ++L)
        {
            const uint32_t rStart = L, rEnd = (L + 1) & 3;
            const V3 p0 = pts[rStart], p1 = pts[rEnd];
            const bool pPenStart = pPen[rStart], pPenEnd = pPen[rEnd];
            const bool pAreaStart = pArea[rStart], pAreaEnd = pArea[rEnd];
            if (pPenStart || pPenEnd)
            {
                const bool con0 = pPenStart && pAreaStart, con1 = pPenEnd && pAreaEnd;
                if (!(con0 && con1))
                {
                    const V3 d{ p1.x - p0.x, p1.y - p0.y, p1.z - p0.z };
                    if (std::fabs(d.x) < MARG || std::fabs(d.y) < MARG || std::fabs(d.z) < MARG) marginal = true;
                    float tmn, tmx;
                    const bool hit = segAABB(p0, d, ext, negExt, tmn, tmx);
                    if (std::fabs(tmx - tmn) < MARG) marginal = true;
                    if (hit)
                    {
                        if (!con0)
                        {
                            const V3 ip{ p0.x + d.x * tmn, p0.y + d.y * tmn, p0.z + d.z * tmn };
                            cout[cnt[L] * 4 + L] = F4{ ip.x, ip.y, ip.z, -ip.z }; cnt[L]++; hadEdge = true;
                        }
                        if (!con1)
                        {
                            const V3 ip{ p0.x + d.x * tmx, p0.y + d.y * tmx, p0.z + d.z * tmx };
                            cout[cnt[L] * 4 + L] = F4{ ip.x, ip.y, ip.z, -ip.z }; cnt[L]++; hadEdge = true;
                        }
                    }
                }
            }
        }
    }
    for (int L = 0; L < 4; ++L) count[L] = cnt[L];
    return marginal;
}

PXG_TEST(npBoxContacts_calculateContacts)
{
    std::mt19937 rng(49010);
    const uint32_t NG = 6000;
    const uint32_t N = NG * 4;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<ContactsIn> in(NG);
    std::vector<F4> pts(N);
    std::vector<uint8_t> skip(NG, 0);
    std::vector<uint32_t> refCount(N);
    std::vector<F4> refC(NG * 12);

    size_t nSkip = 0, pathArea = 0, pathArea2 = 0, pathEdge = 0;
    for (uint32_t g = 0; g < NG; ++g)
    {
        const float eX = rf(0.6f, 1.8f), eY = rf(0.6f, 1.8f);
        const float cd = rf(0.02f, 0.2f);
        // incident-face normal: mostly +z so denom is well away from 0
        V3 fn{ rf(-0.35f, 0.35f), rf(-0.35f, 0.35f), rf(0.7f, 1.3f) };
        // incident polygon: a rotated quad, size spanning smaller/larger than the
        // face, offset so it lands inside / straddling / covering the face
        const float hx = rf(0.3f, 2.4f), hy = rf(0.3f, 2.4f);
        const float ox = rf(-1.6f, 1.6f), oy = rf(-1.6f, 1.6f);
        const float ang = rf(0.0f, 3.14159f);
        const float ca = std::cos(ang), sa = std::sin(ang);
        // base z (penetration depth) and a slight tilt to give the 4 verts distinct z
        const float z0 = rf(-0.25f, 0.15f);
        auto corner = [&](float lx, float ly) {
            const float x = ox + ca * lx - sa * ly;
            const float y = oy + sa * lx + ca * ly;
            // coplanar on the incident plane through (ox,oy,z0) with normal fn
            const float z = z0 - (fn.x * (x - ox) + fn.y * (y - oy)) / fn.z;
            return V3{ x, y, z };
        };
        V3 P[4] = { corner(-hx, -hy), corner(hx, -hy), corner(hx, hy), corner(-hx, hy) };

        ContactsIn c{};
        c.faceNormal = F4{ fn.x, fn.y, fn.z, 0 };
        c.misc = F4{ eX, eY, cd, 0 };
        in[g] = c;
        for (int L = 0; L < 4; ++L) pts[g * 4 + L] = F4{ P[L].x, P[L].y, P[L].z, 0 };

        uint32_t cnt[4]; F4 co[12];
        bool hadArea, hadCorner, hadEdge;
        const bool marg = refGroup(eX, eY, cd, fn, P, cnt, co, hadArea, hadCorner, hadEdge);
        skip[g] = marg ? 1u : 0u;
        if (marg) { ++nSkip; }
        for (int L = 0; L < 4; ++L) refCount[g * 4 + L] = cnt[L];
        for (int i = 0; i < 12; ++i) refC[g * 12 + i] = co[i];

        if (!marg)
        {
            if (hadArea) ++pathArea;     // >=1 area (in-face vertex) contact
            if (hadCorner) ++pathArea2;  // >=1 reference-corner-in-incident contact
            if (hadEdge) ++pathEdge;     // >=1 edge-clip contact
        }
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<F4> outInit(NG * 12, F4{ 9, 9, 9, 9 });
    std::vector<uint32_t> cntInit(N, 999);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bPts = createBufferFrom(dev, pts);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    PxgBufferHandle bCnt = createBufferFrom(dev, cntInit);

    struct BoxContactsParams { uint32_t numGroups; } p = { NG };
    Kernel k("npBoxContacts.spv", "boxCalculateContacts", { 32, 16, 16, 4 }, sizeof(BoxContactsParams));
    k.dispatch(GROUPS, { bIn, bPts, bOut, bCnt }, p);
    dev->flush();

    const auto gCnt = downloadBuffer<uint32_t>(dev, bCnt, N);
    const auto gC = downloadBuffer<F4>(dev, bOut, NG * 12);

    size_t mmCount = 0, mmContact = 0, maxCount = 0;
    for (uint32_t g = 0; g < NG; ++g)
    {
        if (skip[g]) continue;
        for (uint32_t L = 0; L < 4; ++L)
        {
            const uint32_t rc = refCount[g * 4 + L], gc = gCnt[g * 4 + L];
            maxCount = std::max<size_t>(maxCount, gc);
            if (rc != gc) { ++mmCount; continue; }
            for (uint32_t c = 0; c < rc; ++c)
            {
                const F4& r = refC[g * 12 + 4 * c + L];
                const F4& v = gC[g * 12 + 4 * c + L];
                if (!almostEqual(v.x, r.x, 1e-4f, 1e-5f) || !almostEqual(v.y, r.y, 1e-4f, 1e-5f) ||
                    !almostEqual(v.z, r.z, 1e-4f, 1e-5f) || !almostEqual(v.w, r.w, 1e-4f, 1e-5f)) ++mmContact;
            }
        }
    }
    PXG_CHECK_MSG(mmCount == 0, "calculateContacts: %zu per-lane count mismatches", mmCount);
    PXG_CHECK_MSG(mmContact == 0, "calculateContacts: %zu contact-value mismatches", mmContact);
    PXG_CHECK_MSG(maxCount <= 3, "calculateContacts: count exceeded 3 (%zu)", maxCount);
    PXG_CHECK_MSG(nSkip < NG / 4, "calculateContacts: too many marginal groups skipped (%zu/%u)", nSkip, NG);
    PXG_CHECK_MSG(pathArea > NG / 40 && pathArea2 > NG / 40 && pathEdge > NG / 20,
                  "calculateContacts: path coverage thin (area=%zu corner=%zu edge=%zu)", pathArea, pathArea2, pathEdge);

    for (PxgBufferHandle h : { bIn, bPts, bOut, bCnt }) dev->destroyBuffer(h);
}

} // namespace
