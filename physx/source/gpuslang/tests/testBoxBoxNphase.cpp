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

// Verification of boxBoxNphase.slang (boxBoxNphase_Kernel) on the ENGINE launch
// ABI against a C++ transcription of boxBoxNphase_Kernel / doBoxBoxGenerate-
// Contacts (gpunarrowphase/src/CUDA/cudaBox.cu) plus the contact-output tail
// (contactPatchUtils.cuh).  The kernel is driven through PxgCudaShim as the
// 19-argument CUDA parameter pack packed to a 136-byte natural-alignment blob;
// the element structs (PxgContactManagerInput / PxsCachedTransform / PxgShape /
// PxsContactManagerOutput / PxgPatchAndContactCounters / PxsMaterialData /
// PxContact stream / PxContactPatch stream) are filled with the real engine byte
// layouts so the read/write comparison also validates the ABI strides.  The
// reference reproduces the 4-lane collaborative algorithm sequentially (SAT ->
// feature -> incident polygon -> clip -> 4-point reduction).

#include "pxgTestFramework.h"
#include "pxgGpuStructs.h"
#include "pxgRefMath.h"
#include "PxgCudaShim.h"

#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

static const uint32_t INVALID = 0xffffffffu;
static const float FMAX = 3.4028234663852885981170418348452e+38f; // PX_MAX_F32

// ---------------------------------------------------------------------------
// Host mirrors of the engine element structs (byte-exact vs the C++ headers).
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct HInput { uint32_t shapeRef0, shapeRef1, transformCacheRef0, transformCacheRef1; }; // 16
struct HCachedTransform { float qx, qy, qz, qw, px, py, pz; uint32_t flags; };             // 32
struct HShape { float sx, sy, sz, srx, sry, srz, srw; uint32_t materialIndex;
                uint32_t hullLo, hullHi, type, particleId; };                              // 48
struct HOutput { uint64_t contactPatches, contactPoints, contactForces, frictionPatches;
                 uint32_t allflagsWord, nbContacts_flags; uint32_t pad0, pad1; };          // 48
struct HCounters { uint32_t patchesBytes, contactsBytes, forceAndIndiceBytes, overflowError; }; // 16
struct HMaterial { float dyn, sta, rest, damp; uint32_t packedFlags, packedDamp; };        // 24
#pragma pack(pop)

static_assert(sizeof(HInput) == 16, "HInput");
static_assert(sizeof(HCachedTransform) == 32, "HCachedTransform");
static_assert(sizeof(HShape) == 48, "HShape");
static_assert(sizeof(HOutput) == 48, "HOutput");
static_assert(sizeof(HCounters) == 16, "HCounters");
static_assert(sizeof(HMaterial) == 24, "HMaterial");

// ---------------------------------------------------------------------------
// CPU reference : sequential simulation of the 4-lane box-box algorithm.
// ---------------------------------------------------------------------------
struct RefPair
{
    bool separated = false;
    Vec3 normal;                 // mtd
    int nbContacts = 0;
    int total = 0;               // pre-reduction contact count (> 6 => caseB reduction)
    Vec3 pts[6];                 // global-space, ordered as written to the stream
    float pen[6] = {};
};

static float comp3(const Vec3& v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }
static Vec3 vmul(const Vec3& a, const Vec3& b) { return Vec3(a.x * b.x, a.y * b.y, a.z * b.z); }

static bool contains4(Vec3 P0, Vec3 P1, Vec3 P2, Vec3 P3, float px, float py, Vec3 bmin, Vec3 bmax)
{
    if (px < bmin.x || px > bmax.x || py < bmin.y || py > bmax.y) return false;
    const float Vx[4] = { P0.x, P1.x, P2.x, P3.x };
    const float Vy[4] = { P0.y, P1.y, P2.y, P3.y };
    const float eps = 1e-6f, zero = 0.f;
    uint32_t inter = 0; bool inside = false, done = false;
    uint32_t j = 3;
    for (uint32_t i = 0; i < 4 && !done; ++i)
    {
        const float jx = Vx[j], jy = Vy[j], ix = Vx[i], iy = Vy[i];
        if ((px == jx && py == jy) || (px == ix && py == iy)) { inside = true; done = true; }
        else
        {
            const bool y0 = jy > py, y1 = iy > py;
            if (y0 != y1)
            {
                const float jix = ix - jx, jiy = iy - jy, jty = py - jy;
                const float part1 = jty * jix, part2 = (jx + eps) * jiy, part3 = px * jiy;
                const bool comp = jiy > zero; const float tmp = part1 + part2;
                const float c1 = comp ? tmp : part3, c2 = comp ? part3 : tmp;
                if (c1 >= c2) { if (inter == 1) { inside = false; done = true; } else inter++; }
            }
            j = i;
        }
    }
    if (!done) inside = inter > 0;
    return inside;
}

static bool segAABB(Vec3 p0, Vec3 d, Vec3 mx, Vec3 mn, float& tmin, float& tmax)
{
    const float eps = 1e-6f;
    const Vec3 absV(std::fabs(d.x), std::fabs(d.y), std::fabs(d.z));
    float tminf = 0.f, tmaxf = 1.f; tmin = 0.f; tmax = 1.f;
    const bool pX = eps > absV.x, pY = eps > absV.y, pZ = eps > absV.z;
    if ((pX && (p0.x > mx.x || p0.x < mn.x)) || (pY && (p0.y > mx.y || p0.y < mn.y)) ||
        (pZ && (p0.z > mx.z || p0.z < mn.z)))
        return false;
    const Vec3 odd(1.f / d.x, 1.f / d.y, 1.f / d.z);
    const Vec3 t1(pX ? 0.f : (mn.x - p0.x) * odd.x, pY ? 0.f : (mn.y - p0.y) * odd.y, pZ ? 0.f : (mn.z - p0.z) * odd.z);
    const Vec3 t2(pX ? FMAX : (mx.x - p0.x) * odd.x, pY ? FMAX : (mx.y - p0.y) * odd.y, pZ ? FMAX : (mx.z - p0.z) * odd.z);
    const Vec3 tt1 = t1.minimum(t2), tt2 = t1.maximum(t2);
    const float ft1 = std::fmax(tt1.x, std::fmax(tt1.y, tt1.z));
    const float ft2 = std::fmin(tt2.x, std::fmin(tt2.y, tt2.z));
    tminf = std::fmax(ft1, tminf); tmaxf = std::fmin(tmaxf, ft2);
    tmin = tminf; tmax = tmaxf;
    return !((tminf > tmaxf) || (tminf > 1.f));
}

// lowest-lane argmin over 4 candidate values (minIndex4 semantics)
static int argmin4(const float v[4])
{
    float m = std::fmin(v[0], std::fmin(v[1], std::fmin(v[2], v[3])));
    return (v[0] == m) ? 0 : ((v[1] == m) ? 1 : ((v[2] == m) ? 2 : 3));
}

static RefPair refBoxBox(Vec3 box0e, Vec3 box1e, const TransformV& xf0, const TransformV& xf1, float cDist)
{
    RefPair R;
    const Mat33 rot0(xf0.q), rot1(xf1.q);
    const Vec3 c00 = rot0.col0, c01 = rot0.col1, c02 = rot0.col2, tp0 = xf0.p;
    const Vec3 c10 = rot1.col0, c11 = rot1.col1, c12 = rot1.col2, tp1 = xf1.p;
    const float eps = 1e-6f;

    // ---- Stage 1 : SAT ----
    const Vec3 dP = tp1 - tp0;
    const Vec3 r0(c00.dot(c10), c01.dot(c10), c02.dot(c10));
    const Vec3 r1(c00.dot(c11), c01.dot(c11), c02.dot(c11));
    const Vec3 r2(c00.dot(c12), c01.dot(c12), c02.dot(c12));
    const Vec3 tp(c00.dot(dP), c01.dot(dP), c02.dot(dP));
    const Vec3 ab0(std::fabs(r0.x) + eps, std::fabs(r1.x) + eps, std::fabs(r2.x) + eps);
    const Vec3 ab1(std::fabs(r0.y) + eps, std::fabs(r1.y) + eps, std::fabs(r2.y) + eps);
    const Vec3 ab2(std::fabs(r0.z) + eps, std::fabs(r1.z) + eps, std::fabs(r2.z) + eps);

    float minOv[3]; uint32_t featv[3]; float sgnv[3]; bool sepv[3];
    for (int tid = 0; tid < 3; ++tid)
    {
        const Vec3 col = (tid == 0) ? r0 : (tid == 1 ? r1 : r2);
        const Vec3 abCol(std::fabs(col.x) + eps, std::fabs(col.y) + eps, std::fabs(col.z) + eps);
        const Vec3 abT = (tid == 0) ? ab0 : (tid == 1 ? ab1 : ab2);
        const float tpc = comp3(tp, tid), ea = comp3(box0e, tid), eb = comp3(box1e, tid);
        const float rbA = abT.dot(box1e);
        const float oA = (ea + rbA) - std::fabs(tpc) + cDist, sA = tpc;
        const float sB = tp.dot(col), raB = abCol.dot(box0e);
        const float oB = (raB + eb) - std::fabs(sB) + cDist;
        float minO = oA; uint32_t feat = uint32_t(tid); float sgn = sA;
        if (minO > oB) { minO = oB; sgn = sB; feat = uint32_t(tid) + 3; }
        bool sepFace = (0.f > oA) || (0.f > oB);
        const int iHi = (tid == 2) ? 1 : 2, iLo = (tid == 0) ? 1 : 0;
        const float b1Lo = comp3(box1e, iLo), b1Hi = comp3(box1e, iHi);
        const float as1 = std::fabs(col.y * tp.z - col.z * tp.y);
        const float ra1 = abCol.z * box0e.y + abCol.y * box0e.z;
        const float rb1 = comp3(ab0, iHi) * b1Lo + comp3(ab0, iLo) * b1Hi;
        const bool sep1 = as1 > (ra1 + rb1 + cDist);
        const float as2 = std::fabs(col.z * tp.x - col.x * tp.z);
        const float ra2 = abCol.z * box0e.x + abCol.x * box0e.z;
        const float rb2 = comp3(ab1, iHi) * b1Lo + comp3(ab1, iLo) * b1Hi;
        const bool sep2 = as2 > (ra2 + rb2 + cDist);
        const float as3 = std::fabs(col.x * tp.y - col.y * tp.x);
        const float ra3 = abCol.y * box0e.x + abCol.x * box0e.y;
        const float rb3 = comp3(ab2, iHi) * b1Lo + comp3(ab2, iLo) * b1Hi;
        const bool sep3 = as3 > (ra3 + rb3 + cDist);
        minOv[tid] = minO; featv[tid] = feat; sgnv[tid] = sgn;
        sepv[tid] = sepFace || sep1 || sep2 || sep3;
    }
    R.separated = sepv[0] || sepv[1] || sepv[2];
    int pick = (minOv[0] <= minOv[1] && minOv[0] <= minOv[2]) ? 0 : (minOv[1] <= minOv[2] ? 1 : 2);
    const uint32_t feature = featv[pick];
    const float sign = sgnv[pick];
    if (R.separated) { R.normal = Vec3(0, 0, 0); R.nbContacts = 0; return R; }

    // ---- Stage 2 : feature switch ----
    const bool neg = 0.f >= sign;
    Vec3 mtd, nc0, nc1, nc2, np; float e0, e1;
    if (feature == 0) { if (neg) { mtd = c00; nc0 = -c02; nc1 = c01; nc2 = c00; } else { mtd = -c00; nc0 = c02; nc1 = c01; nc2 = -c00; } np = tp0 - mtd * box0e.x; e0 = box0e.z; e1 = box0e.y; }
    else if (feature == 1) { if (neg) { mtd = c01; nc0 = c00; nc1 = -c02; nc2 = c01; } else { mtd = -c01; nc0 = c00; nc1 = c02; nc2 = -c01; } np = tp0 - mtd * box0e.y; e0 = box0e.x; e1 = box0e.z; }
    else if (feature == 2) { if (neg) { mtd = c02; nc0 = c00; nc1 = c01; nc2 = c02; } else { mtd = -c02; nc0 = c00; nc1 = -c01; nc2 = -c02; } np = tp0 - mtd * box0e.z; e0 = box0e.x; e1 = box0e.y; }
    else if (feature == 3) { if (neg) { mtd = c10; nc0 = c12; nc1 = c11; nc2 = -c10; } else { mtd = -c10; nc0 = -c12; nc1 = c11; nc2 = c10; } np = tp1 + mtd * box1e.x; e0 = box1e.z; e1 = box1e.y; }
    else if (feature == 4) { if (neg) { mtd = c11; nc0 = c10; nc1 = c12; nc2 = -c11; } else { mtd = -c11; nc0 = c10; nc1 = -c12; nc2 = c11; } np = tp1 + mtd * box1e.y; e0 = box1e.x; e1 = box1e.z; }
    else { if (neg) { mtd = c12; nc0 = c10; nc1 = -c11; nc2 = -c12; } else { mtd = -c12; nc0 = c10; nc1 = c11; nc2 = c12; } np = tp1 + mtd * box1e.z; e0 = box1e.x; e1 = box1e.y; }

    const bool useT1 = feature < 3;
    const Vec3 oc0 = useT1 ? c10 : c00, oc1 = useT1 ? c11 : c01, oc2 = useT1 ? c12 : c02, oP = useT1 ? tp1 : tp0;
    auto matTT = [](Vec3 a, Vec3 b, Vec3 c, Vec3 v) { return Vec3(a.dot(v), b.dot(v), c.dot(v)); };
    const Vec3 t1n0 = matTT(nc0, nc1, nc2, oc0), t1n1 = matTT(nc0, nc1, nc2, oc1), t1n2 = matTT(nc0, nc1, nc2, oc2);
    const Vec3 t1nP = matTT(nc0, nc1, nc2, oP - np);
    const Vec3 localNormal = matTT(nc0, nc1, nc2, mtd);
    auto xform = [&](Vec3 c0, Vec3 c1, Vec3 c2, Vec3 p, Vec3 v) { return c0 * v.x + c1 * v.y + c2 * v.z + p; };

    // ---- Stage 3 : incident polygon ----
    const Vec3 axis = useT1 ? -localNormal : localNormal;
    const Vec3 iext = useT1 ? box1e : box0e;
    float dv[3], absd[3];
    for (int l = 0; l < 3; ++l)
    {
        const Vec3 col = (l == 0) ? t1n0 : (l == 1 ? t1n1 : t1n2);
        dv[l] = col.dot(axis); absd[l] = std::fabs(dv[l]);
    }
    Vec3 faceNormal;
    { // group-uniform incident face normal
        if (absd[0] >= absd[1] && absd[0] >= absd[2]) faceNormal = t1n0 * ((dv[0] > 0.f) ? -1.f : 1.f);
        else if (absd[1] >= absd[2]) faceNormal = t1n1 * ((dv[1] > 0.f) ? -1.f : 1.f);
        else faceNormal = t1n2 * ((dv[2] > 0.f) ? -1.f : 1.f);
    }
    Vec3 pts[4];
    for (int l = 0; l < 4; ++l)
    {
        const float s0 = (l == 0 || l == 3) ? -1.f : 1.f;
        const float s1 = (l & 2) ? 1.f : -1.f;
        Vec3 isgn;
        if (absd[0] >= absd[1] && absd[0] >= absd[2]) { isgn.x = (dv[0] > 0.f) ? -1.f : 1.f; isgn.y = s0; isgn.z = s1; }
        else if (absd[1] >= absd[2]) { isgn.y = (dv[1] > 0.f) ? -1.f : 1.f; isgn.x = s0; isgn.z = s1; }
        else { isgn.z = (dv[2] > 0.f) ? -1.f : 1.f; isgn.x = s0; isgn.y = s1; }
        pts[l] = xform(t1n0, t1n1, t1n2, t1nP, vmul(iext, isgn));
    }
    R.normal = mtd;

    // ---- Stage 4 : calculateContacts ----
    Vec3 bmin = pts[0], bmax = pts[0];
    for (int l = 1; l < 4; ++l) { bmin = bmin.minimum(pts[l]); bmax = bmax.maximum(pts[l]); }
    const float extentX = 1.00001f * e0, extentY = 1.00001f * e1;
    const Vec3 ext(extentX, extentY, FMAX), negExt(-extentX, -extentY, -cDist - 1e-7f);

    struct Vec4 { float x, y, z, w; };
    Vec4 localC[4][3];
    int cnt[4] = { 0, 0, 0, 0 };
    bool pPen[4], pArea[4] = { false, false, false, false }, pArea2[4] = { false, false, false, false };

    for (int l = 0; l < 4; ++l)
    {
        const float z = -pts[l].z;
        pPen[l] = cDist > z;
        if (pPen[l] && extentX >= std::fabs(pts[l].x) && extentY >= std::fabs(pts[l].y))
        {
            pArea[l] = true; localC[l][0] = { pts[l].x, pts[l].y, pts[l].z, z }; cnt[l] = 1;
        }
    }
    const bool allArea = pArea[0] && pArea[1] && pArea[2] && pArea[3];
    if (!allArea)
    {
        for (int l = 0; l < 4; ++l)
        {
            const float denom = faceNormal.z;
            const float qx = (l == 0 || l == 3) ? -extentX : extentX;
            const float qy = (l & 2) ? -extentY : extentY;
            if (contains4(pts[0], pts[1], pts[2], pts[3], qx, qy, bmin, bmax))
            {
                const Vec3 diff(pts[0].x - qx, pts[0].y - qy, pts[0].z);
                const float tval = faceNormal.dot(diff) / denom;
                if (cDist > (-tval)) { pArea2[l] = true; localC[l][cnt[l]] = { qx, qy, tval, -tval }; cnt[l]++; }
            }
        }
    }
    const bool allArea2 = pArea2[0] && pArea2[1] && pArea2[2] && pArea2[3];
    const bool done2 = allArea || allArea2;
    if (!done2)
    {
        for (int l = 0; l < 4; ++l)
        {
            const int rStart = l, rEnd = (l + 1) & 3;
            const Vec3 p0 = pts[rStart], p1 = pts[rEnd];
            if (pPen[rStart] || pPen[rEnd])
            {
                const bool con0 = pPen[rStart] && pArea[rStart];
                const bool con1 = pPen[rEnd] && pArea[rEnd];
                if (!(con0 && con1))
                {
                    const Vec3 p0p1 = p1 - p0; float tmn, tmx;
                    if (segAABB(p0, p0p1, ext, negExt, tmn, tmx))
                    {
                        if (!con0) { const Vec3 ip = p0 + p0p1 * tmn; localC[l][cnt[l]] = { ip.x, ip.y, ip.z, -ip.z }; cnt[l]++; }
                        if (!con1) { const Vec3 ip = p0 + p0p1 * tmx; localC[l][cnt[l]] = { ip.x, ip.y, ip.z, -ip.z }; cnt[l]++; }
                    }
                }
            }
        }
    }
    for (int l = 0; l < 4; ++l) if (cnt[l] > 3) cnt[l] = 3;

    // ---- Stage 5 : reduction ----
    const int total = cnt[0] + cnt[1] + cnt[2] + cnt[3];
    R.total = total;
    auto gl = [&](const Vec4& c) { return xform(nc0, nc1, nc2, np, Vec3(c.x, c.y, c.z)); };
    if (total <= 6)
    {
        int excl[4]; int acc = 0;
        for (int l = 0; l < 4; ++l) { excl[l] = acc; acc += cnt[l]; }
        for (int l = 0; l < 4; ++l)
            for (int i = 0; i < cnt[l]; ++i)
            { const Vec3 g = gl(localC[l][i]); R.pts[excl[l] + i] = g; R.pen[excl[l] + i] = localC[l][i].w; }
        R.nbContacts = total;
    }
    else
    {
        // deepest
        float dWv[4]; int dIv[4];
        for (int l = 0; l < 4; ++l) { float dW = FMAX; int dI = 0; for (int i = 0; i < cnt[l]; ++i) if (localC[l][i].w < dW) { dW = localC[l][i].w; dI = i; } dWv[l] = dW; dIv[l] = dI; }
        int wD = argmin4(dWv);
        const Vec4 deepest = localC[wD][dIv[wD]];
        const Vec3 deepestPt(deepest.x, deepest.y, deepest.z);
        // furthest
        const Vec3 dpp = deepestPt - faceNormal * faceNormal.dot(deepestPt);
        float fWv[4]; int fIv[4];
        for (int l = 0; l < 4; ++l) { float fW = -FMAX; int fI = 0; for (int i = 0; i < cnt[l]; ++i) { const Vec3 d = dpp - Vec3(localC[l][i].x, localC[l][i].y, localC[l][i].z); const float ds = d.dot(d); if (ds > fW) { fW = ds; fI = i; } } fWv[l] = -fW; fIv[l] = fI; }
        int wF = argmin4(fWv);
        const Vec4 furthest = localC[wF][fIv[wF]];
        const Vec3 furthestPt(furthest.x, furthest.y, furthest.z);
        // min/max perpendicular
        const Vec3 dir = faceNormal.cross(furthestPt - deepestPt);
        float mnVv[4], mxVv[4]; int mnIv[4], mxIv[4];
        for (int l = 0; l < 4; ++l)
        {
            float mxV = -FMAX, mnV = FMAX; int mxI = 0, mnI = 0;
            for (int i = 0; i < cnt[l]; ++i) { const float dp = dir.dot(Vec3(localC[l][i].x, localC[l][i].y, localC[l][i].z) - deepestPt); if (dp > mxV) { mxV = dp; mxI = i; } if (dp < mnV) { mnV = dp; mnI = i; } }
            mnVv[l] = mnV; mxVv[l] = -mxV; mnIv[l] = mnI; mxIv[l] = mxI;
        }
        int wA = argmin4(mnVv), wB = argmin4(mxVv);
        const Vec4 cMin = localC[wA][mnIv[wA]], cMax = localC[wB][mxIv[wB]];
        const Vec4 chosen[4] = { deepest, furthest, cMin, cMax };
        for (int k = 0; k < 4; ++k) { R.pts[k] = gl(chosen[k]); R.pen[k] = chosen[k].w; }
        R.nbContacts = 4;
    }
    return R;
}

// PxsCombineMaterials (mirrors npInsertPatch combineMaterials)
static void combineMaterials(const HMaterial& m0, const HMaterial& m1, float& sta, float& dyn,
                             float& rest, float& damping, uint32_t& flags)
{
    const uint32_t DISABLE_FRICTION = 1u, DISABLE_STRONG = 2u, ACC_SPRING = 16u;
    auto combine = [](float a, float b, uint32_t mode) -> float {
        if (mode == 0) return 0.5f * (a + b); if (mode == 1) return std::fmin(a, b);
        if (mode == 2) return a * b; if (mode == 3) return std::fmax(a, b); return 0.f; };
    const uint32_t f0 = m0.packedFlags & 0xffffu, f1 = m1.packedFlags & 0xffffu;
    const uint32_t fr0 = (m0.packedFlags >> 16) & 0xffu, fr1 = (m1.packedFlags >> 16) & 0xffu;
    const uint32_t re0 = (m0.packedFlags >> 24) & 0xffu, re1 = (m1.packedFlags >> 24) & 0xffu;
    const uint32_t dm0 = m0.packedDamp & 0xffu, dm1 = m1.packedDamp & 0xffu;
    const bool comp0 = m0.rest < 0.f, comp1 = m1.rest < 0.f;
    const bool exactlyOne = comp0 != comp1, both = comp0 && comp1;
    const bool a0 = (f0 & ACC_SPRING) != 0, a1 = (f1 & ACC_SPRING) != 0, exactlyOneAcc = a0 != a1;
    if (both && exactlyOneAcc) rest = a0 ? m0.rest : m1.rest;
    else { const uint32_t mode = exactlyOne ? 1u : std::max(re0, re1); const float flip = (both && mode == 2u) ? -1.f : 1.f; rest = flip * combine(m0.rest, m1.rest, mode); }
    if (both && exactlyOneAcc) damping = a0 ? m0.damp : m1.damp;
    else { const uint32_t mode = exactlyOne ? 3u : std::max(dm0, dm1); damping = combine(m0.damp, m1.damp, mode); }
    const uint32_t cf = f0 | f1;
    if ((cf & DISABLE_FRICTION) == 0) { const uint32_t fm = std::max(fr0, fr1); const float d = combine(m0.dyn, m1.dyn, fm), s = combine(m0.sta, m1.sta, fm); const float fd = std::fmax(d, 0.f); dyn = fd; sta = (s - fd) >= 0.f ? s : fd; flags = cf; }
    else { flags = cf | DISABLE_STRONG; dyn = 0.f; sta = 0.f; }
}

// ---------------------------------------------------------------------------
// test driver
// ---------------------------------------------------------------------------
struct PairCfg { Vec3 box0e, box1e; TransformV xf0, xf1; float cDist; uint32_t mat0, mat1; };

static HCachedTransform makeXf(const TransformV& t)
{
    return { t.q.x, t.q.y, t.q.z, t.q.w, t.p.x, t.p.y, t.p.z, 0u };
}
static HShape makeShape(const Vec3& scale, uint32_t mat)
{
    return { scale.x, scale.y, scale.z, 0.f, 0.f, 0.f, 1.f, mat, 0u, 0u, 3u /*eBOX*/, 0u };
}

static std::vector<uint8_t> boxShimLoader(const char* file) { return loadShaderBlob(file); }

static void checkBoxBox(const std::vector<PairCfg>& cfgs, uint32_t priorAllflags, bool mutate, const char* label)
{
    using namespace physx::gpuslang;
    const uint32_t n = uint32_t(cfgs.size());

    // materials (indexed by shape.materialIndex).  0x1235 entries so a pair can carry indices with
    // non-zero high bytes (0x0123/0x1234): words 12/13 store the full PxU16 (contactPatchUtils.cuh:149-150,
    // :170-171), which indices 0..7 cannot tell from an 8-bit mask.
    std::vector<HMaterial> materials(0x10000u + 8u);
    for (uint32_t i = 0; i < 0x1235u; ++i)
    {
        const float f = 0.2f + 0.1f * i;
        materials[i] = { f, f + 0.3f, (i % 3 == 2) ? -0.5f : 0.4f, 0.05f * i,
                         uint32_t((i & 1) * 16u) | (uint32_t(i % 4) << 16) | (uint32_t((i + 1) % 4) << 24),
                         uint32_t((i + 2) % 4) };
    }
    // P9.88: a second bank at [0x10000 + i] holds DIFFERENT materials, so an index k | 0x10000
    // tells a lookup that narrows to PxU16 first (contactPatchUtils.cuh:149-150,
    // materialCombiner.cuh:35) from one that does not; both stay in bounds.
    for (uint32_t i = 0; i < 8u; ++i)
    {
        const float g = 1.1f - 0.07f * i;
        materials[0x10000u + i] = { g, g + 0.15f, (i % 3 == 1) ? -0.3f : 0.1f + 0.05f * i, 0.9f - 0.1f * i,
                                    uint32_t(((i + 1) & 1) * 16u) | (uint32_t((i + 3) % 4) << 16) | (uint32_t((i + 2) % 4) << 24),
                                    uint32_t((i + 1) % 4) };
    }

    // transform cache + contact distances (2 refs per pair)
    std::vector<HCachedTransform> xcache(2 * n);
    std::vector<float> cdist(2 * n);
    std::vector<HInput> inputs(n);
    std::vector<HShape> shapes(2 * n);
    for (uint32_t i = 0; i < n; ++i)
    {
        xcache[2 * i + 0] = makeXf(cfgs[i].xf0);
        xcache[2 * i + 1] = makeXf(cfgs[i].xf1);
        cdist[2 * i + 0] = 0.5f * cfgs[i].cDist;
        cdist[2 * i + 1] = 0.5f * cfgs[i].cDist;
        shapes[2 * i + 0] = makeShape(cfgs[i].box0e, cfgs[i].mat0);
        shapes[2 * i + 1] = makeShape(cfgs[i].box1e, cfgs[i].mat1);
        inputs[i] = { 2 * i + 0, 2 * i + 1, 2 * i + 0, 2 * i + 1 };
    }

    // outputs seeded with a known prior status word (exercises the flag logic)
    std::vector<HOutput> outputs(n);
    std::memset(outputs.data(), 0, outputs.size() * sizeof(HOutput));
    for (auto& o : outputs) { o.allflagsWord = priorAllflags; o.nbContacts_flags = 0xAAAA0000u; /*prior flags hi*/ }

    HCounters counters = { 0, 0, 0, 0 };
    const uint32_t CONTACT_LIMIT = 1u << 20, FORCE_LIMIT = 1u << 20, PATCH_LIMIT = 1u << 20;
    std::vector<uint8_t> contactStream(CONTACT_LIMIT, 0), patchStream(PATCH_LIMIT), forceStream(FORCE_LIMIT, 0);
    // Poisoned patch stream: insertIntoPatchStream (contactPatchUtils.cuh:161-178) stores field by field
    // and never writes pad bytes 54-63, and the engine's mCompressedPatches is never cleared, so those
    // bytes must come back exactly as uploaded.  The fill varies by byte so a store that copies another
    // poisoned word into the pad is visible; the host vector is not modified after the upload and is
    // the expectation.
    for (uint32_t k = 0; k < PATCH_LIMIT; ++k) patchStream[k] = uint8_t(0xCDu ^ (k * 37u));
    std::vector<uint32_t> touchFlags(n, 7u), patchFlags(n, 7u); // seeded with junk

    // ---- device ----
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = { { "boxBoxNphase.spv", "boxBoxNphase_Kernel", 136u, 0 } };
    PxgCudaShim shim(dev, manifest, 1, &boxShimLoader);

    PxgDevicePtr dIn = 0, dOut = 0, dShapes = 0, dXf = 0, dCd = 0, dMat = 0, dCStream = 0, dPStream = 0,
                 dCounters = 0, dTouch = 0, dPatch = 0, dStartPatch = 0, dStartPoint = 0, dStartForce = 0;
    shim.memAlloc(&dIn, inputs.size() * sizeof(HInput));
    shim.memAlloc(&dOut, outputs.size() * sizeof(HOutput));
    shim.memAlloc(&dShapes, shapes.size() * sizeof(HShape));
    shim.memAlloc(&dXf, xcache.size() * sizeof(HCachedTransform));
    shim.memAlloc(&dCd, cdist.size() * sizeof(float));
    shim.memAlloc(&dMat, materials.size() * sizeof(HMaterial));
    shim.memAlloc(&dCStream, contactStream.size());
    shim.memAlloc(&dPStream, patchStream.size());
    shim.memAlloc(&dCounters, sizeof(HCounters));
    shim.memAlloc(&dTouch, touchFlags.size() * 4);
    shim.memAlloc(&dPatch, patchFlags.size() * 4);
    shim.memAlloc(&dStartForce, forceStream.size());
    if (!dIn || !dOut || !dShapes || !dXf || !dCd || !dMat || !dCStream || !dPStream || !dCounters || !dTouch || !dPatch || !dStartForce)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    // start* pointers are the base device addresses of the streams.
    dStartPatch = dPStream; dStartPoint = dCStream;

    shim.memcpyHtoD(dIn, inputs.data(), inputs.size() * sizeof(HInput));
    shim.memcpyHtoD(dOut, outputs.data(), outputs.size() * sizeof(HOutput));
    shim.memcpyHtoD(dShapes, shapes.data(), shapes.size() * sizeof(HShape));
    shim.memcpyHtoD(dXf, xcache.data(), xcache.size() * sizeof(HCachedTransform));
    shim.memcpyHtoD(dCd, cdist.data(), cdist.size() * sizeof(float));
    shim.memcpyHtoD(dMat, materials.data(), materials.size() * sizeof(HMaterial));
    shim.memcpyHtoD(dCStream, contactStream.data(), contactStream.size());
    shim.memcpyHtoD(dPStream, patchStream.data(), patchStream.size());
    shim.memcpyHtoD(dCounters, &counters, sizeof(HCounters));
    shim.memcpyHtoD(dTouch, touchFlags.data(), touchFlags.size() * 4);
    shim.memcpyHtoD(dPatch, patchFlags.data(), patchFlags.size() * 4);
    shim.memcpyHtoD(dStartForce, forceStream.data(), forceStream.size());

    struct BlobT
    {
        uint32_t numTests;
        uint64_t cmInputs, cmOutputs, shapes, transformCache, contactDistance, materials, contactStream,
                 patchStream, counters, touch, patch, startPatch, startPoint, startForce;
        uint32_t patchLimit, contactLimit, forceLimit; float tolerance;
    } blob = { n, dIn, dOut, dShapes, dXf, dCd, dMat, dCStream, dPStream, dCounters, dTouch, dPatch,
               dStartPatch, dStartPoint, dStartForce, PATCH_LIMIT, CONTACT_LIMIT, FORCE_LIMIT, 1.0f };
    static_assert(sizeof(BlobT) == 136, "boxBoxNphase blob is 136 bytes");

    if (mutate) { const uint64_t s = blob.startPoint; blob.startPoint = blob.startForce; blob.startForce = s; }

    // engine grid: numThreadsPerBlock=64, numPairsPerBlock=16, numBlocks=ceil(n/16)
    const uint32_t numBlocks = (n + 15u) / 16u;
    PXG_CHECK_MSG(shim.launchKernel(0, numBlocks, 1, 1, &blob, sizeof(blob)) == PxgResult::Success,
                  "launch boxBoxNphase_Kernel (%s)", label);
    shim.streamSynchronize();

    // ---- download ----
    std::vector<HOutput> gOut(n);
    HCounters gCnt;
    std::vector<uint8_t> gCStream(contactStream.size()), gPStream(patchStream.size());
    std::vector<uint32_t> gTouch(n), gPatch(n);
    shim.memcpyDtoH(gOut.data(), dOut, gOut.size() * sizeof(HOutput));
    shim.memcpyDtoH(&gCnt, dCounters, sizeof(HCounters));
    shim.memcpyDtoH(gCStream.data(), dCStream, gCStream.size());
    shim.memcpyDtoH(gPStream.data(), dPStream, gPStream.size());
    shim.memcpyDtoH(gTouch.data(), dTouch, gTouch.size() * 4);
    shim.memcpyDtoH(gPatch.data(), dPatch, gPatch.size() * 4);

    // ---- reference ----
    size_t bad = 0;
    uint32_t expTotalContacts = 0, expTotalPatches = 0;
    const uint32_t oldStatus = (priorAllflags >> 16) & 0xffu;
    const uint32_t oldPrevPatches = (priorAllflags >> 8) & 0xffu;
    const bool prevHadTouch = (oldStatus & 2u) != 0u;
    const bool prevKnown = (oldStatus & 3u) != 0u;

    uint32_t caseBPairs = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        const RefPair R = refBoxBox(cfgs[i].box0e, cfgs[i].box1e, cfgs[i].xf0, cfgs[i].xf1, cfgs[i].cDist);
        const uint32_t nb = uint32_t(R.nbContacts);
        if (R.total > 6) caseBPairs++;
        expTotalContacts += nb; if (nb) expTotalPatches++;

        // cmOutput nbContacts
        const uint32_t gNb = gOut[i].nbContacts_flags & 0xffffu;
        if (gNb != nb) { ++bad; if (bad < 12) std::printf("  [%s] pair %u nbContacts gpu=%u ref=%u\n", label, i, gNb, nb); continue; }
        // flags hi-16 preserved
        if ((gOut[i].nbContacts_flags >> 16) != 0xAAAAu) ++bad;

        // packed status word
        // registerContactPatch (contactPatchUtils.cuh:81-89): the old status byte survives with
        // eTOUCH_KNOWN cleared, then eHAS_TOUCH or eHAS_NO_TOUCH is set.
        const uint32_t status = (oldStatus & ~3u) | (nb ? 2u : 1u);
        const uint32_t numPatches = nb ? 1u : 0u;
        const uint32_t expAllflags = (oldPrevPatches << 24) | (status << 16) | (numPatches << 8);
        if (gOut[i].allflagsWord != expAllflags) { ++bad; if (bad < 12) std::printf("  [%s] pair %u allflags gpu=%08x ref=%08x\n", label, i, gOut[i].allflagsWord, expAllflags); }
        // change flags
        const bool curTouch = nb > 0;
        const uint32_t expTouch = ((prevHadTouch != curTouch) || !prevKnown) ? 1u : 0u;
        const uint32_t expPatch = (oldPrevPatches != numPatches) ? 1u : 0u;
        if (gTouch[i] != expTouch) ++bad;
        if (gPatch[i] != expPatch) ++bad;

        if (nb == 0) continue;

        // locate this pair's contacts via cmOutput.contactPoints (byte offset from stream base)
        const uint64_t cpAddr = gOut[i].contactPoints;
        const uint64_t patchAddr = gOut[i].contactPatches;
        // contactForces address must be startContactForces + forceOffset (== forceOffset from base)
        const uint64_t cfAddr = gOut[i].contactForces;
        const bool cpInRange = cpAddr >= dCStream && cpAddr + nb * 16u <= dCStream + contactStream.size();
        const bool pInRange = patchAddr >= dPStream && patchAddr + 64u <= dPStream + patchStream.size();
        const bool cfInRange = cfAddr >= dStartForce && cfAddr + nb * 4u <= dStartForce + forceStream.size();
        if (!cpInRange || !pInRange || !cfInRange)
        {
            ++bad;
            if (bad < 12) std::printf("  [%s] pair %u output ptr out of range (cp=%d p=%d cf=%d)\n",
                                      label, i, cpInRange ? 1 : 0, pInRange ? 1 : 0, cfInRange ? 1 : 0);
            continue;
        }
        const uint64_t cOff = cpAddr - dCStream;
        const uint64_t pOff = patchAddr - dPStream;

        // compare contacts
        for (uint32_t k = 0; k < nb; ++k)
        {
            float f4[4];
            std::memcpy(f4, &gCStream[cOff + k * 16], 16);
            if (!(almostEqual(f4[0], R.pts[k].x, 2e-4f, 2e-5f) && almostEqual(f4[1], R.pts[k].y, 2e-4f, 2e-5f) &&
                  almostEqual(f4[2], R.pts[k].z, 2e-4f, 2e-5f) && almostEqual(f4[3], R.pen[k], 2e-4f, 2e-5f)))
            {
                ++bad;
                if (bad < 12) std::printf("  [%s] pair %u contact %u gpu=(%g,%g,%g,%g) ref=(%g,%g,%g,%g)\n",
                                          label, i, k, f4[0], f4[1], f4[2], f4[3], R.pts[k].x, R.pts[k].y, R.pts[k].z, R.pen[k]);
            }
        }

        // compare patch header -- all 16 words
        float pf[16];
        std::memcpy(pf, &gPStream[pOff], 64);
        const uint32_t* pu = reinterpret_cast<const uint32_t*>(pf);
        uint32_t init[16];
        std::memcpy(init, &patchStream[pOff], 64);   // the header's bytes as uploaded
        for (uint32_t w = 0; w < 4u; ++w)             // mMassModification = 1.0f x4 (contactPatchUtils.cuh:175-178)
            if (pu[w] != 0x3f800000u) { ++bad; if (bad < 12) std::printf("  [%s] pair %u patch word %u gpu=%08x ref=3f800000\n", label, i, w, pu[w]); }
        if (!(almostEqual(pf[4], R.normal.x, 2e-4f, 2e-5f) && almostEqual(pf[5], R.normal.y, 2e-4f, 2e-5f) &&
              almostEqual(pf[6], R.normal.z, 2e-4f, 2e-5f)))
        { ++bad; if (bad < 12) std::printf("  [%s] pair %u patch normal gpu=(%g,%g,%g) ref=(%g,%g,%g)\n", label, i, pf[4], pf[5], pf[6], R.normal.x, R.normal.y, R.normal.z); }

        float sta, dyn, rest, damping; uint32_t mflags;
        combineMaterials(materials[cfgs[i].mat0 & 0xffffu], materials[cfgs[i].mat1 & 0xffffu], sta, dyn, rest, damping, mflags);
        if (!almostEqual(pf[7], rest, 2e-5f, 2e-6f)) ++bad;   // restitution
        if (!almostEqual(pf[8], dyn, 2e-5f, 2e-6f)) ++bad;    // dynamicFriction
        if (!almostEqual(pf[9], sta, 2e-5f, 2e-6f)) ++bad;    // staticFriction
        if (!almostEqual(pf[10], damping, 2e-5f, 2e-6f)) ++bad; // damping
        const uint32_t word11 = ((nb & 0xffu) << 16) | ((mflags & 0xffu) << 24);
        if (pu[11] != word11) ++bad;
        if (pu[12] != ((cfgs[i].mat0 & 0xffffu) << 16)) { ++bad; if (bad < 12) std::printf("  [%s] pair %u patch word 12 gpu=%08x ref=%08x\n", label, i, pu[12], (cfgs[i].mat0 & 0xffffu) << 16); }
        // :171 is a PxU16 store: pad[0] (bytes 54-55) keeps the upload; pad[1..4] (words 14-15) are
        // never written (P9.87).
        const uint32_t expW13 = (init[13] & 0xffff0000u) | (cfgs[i].mat1 & 0xffffu);
        if (pu[13] != expW13) { ++bad; if (bad < 12) std::printf("  [%s] pair %u patch word 13 gpu=%08x ref=%08x\n", label, i, pu[13], expW13); }
        if (pu[14] != init[14] || pu[15] != init[15])
        { ++bad; if (bad < 12) std::printf("  [%s] pair %u patch pad words gpu=%08x,%08x ref=%08x,%08x\n", label, i, pu[14], pu[15], init[14], init[15]); }
    }

    // Patches are packed from offset 0 by the atomic reservation (counters start at 0): nothing past
    // the last reserved slot may be written.
    for (size_t b = size_t(expTotalPatches) * 64u; b < gPStream.size(); ++b)
        if (gPStream[b] != patchStream[b]) { ++bad; std::printf("  [%s] patch stream written past reserved slots at byte %zu\n", label, b); break; }

    // counters : deterministic totals (order-independent partition)
    if (gCnt.contactsBytes != expTotalContacts * 16u) { ++bad; std::printf("  [%s] contactsBytes gpu=%u ref=%u\n", label, gCnt.contactsBytes, expTotalContacts * 16u); }
    if (gCnt.patchesBytes != expTotalPatches * 64u) { ++bad; std::printf("  [%s] patchesBytes gpu=%u ref=%u\n", label, gCnt.patchesBytes, expTotalPatches * 64u); }
    if (gCnt.forceAndIndiceBytes != expTotalContacts * 4u) ++bad;
    if (gCnt.overflowError != 0u) ++bad;

    std::printf("  [%s] pairs=%u contacts=%u patches=%u caseB(>6->4)=%u\n",
                label, n, expTotalContacts, expTotalPatches, caseBPairs);

    if (mutate)
        PXG_CHECK_MSG(bad != 0, "[%s] mutation must FAIL (swapped startPoint/startForce)", label);
    else
        PXG_CHECK_MSG(bad == 0, "[%s] %zu mismatches", label, bad);

    for (PxgDevicePtr p : { dIn, dOut, dShapes, dXf, dCd, dMat, dCStream, dPStream, dCounters, dTouch, dPatch, dStartForce })
        shim.memFree(p);
}

// axis-aligned box with a translation
static TransformV xfT(float x, float y, float z) { return TransformV(QuatV(0, 0, 0, 1), Vec3(x, y, z)); }
static TransformV xfQ(const QuatV& q, Vec3 p) { return TransformV(q.getNormalized(), p); }

PXG_TEST(boxBoxNphase_axisAligned)
{
    std::vector<PairCfg> cfgs;
    // face-face overlap along x (equal unit boxes) -> 4 area contacts
    cfgs.push_back({ Vec3(1, 1, 1), Vec3(1, 1, 1), xfT(0, 0, 0), xfT(1.8f, 0, 0), 0.02f, 1, 2 });
    // face-face overlap along y (different sizes)
    cfgs.push_back({ Vec3(1.5f, 1, 0.8f), Vec3(0.6f, 0.7f, 1.2f), xfT(0, 0, 0), xfT(0.3f, 1.5f, 0.2f), 0.05f, 3, 4 });
    // clearly separated pair -> zero contacts, eHAS_NO_TOUCH
    cfgs.push_back({ Vec3(1, 1, 1), Vec3(1, 1, 1), xfT(0, 0, 0), xfT(5.0f, 0, 0), 0.01f, 0, 5 });
    // box B rotated 45 deg about Y, overlapping along Y: the incident square is
    // rotated 45 deg vs the reference square -> 8 clip points -> total > 6 ->
    // exercises the 4-point manifold reduction (caseB).
    {
        const float s = std::sin(0.39269908f), c = std::cos(0.39269908f); // 22.5 deg
        cfgs.push_back({ Vec3(1, 1, 1), Vec3(1, 1, 1), xfT(0, 0, 0),
                         xfQ(QuatV(0, s, 0, c), Vec3(0.0f, 1.5f, 0.0f)), 0.05f, 6, 1 });
    }
    // touching pair whose material indices have non-zero high bytes (full PxU16 in words 12/13)
    cfgs.push_back({ Vec3(1, 1, 1), Vec3(1, 1, 1), xfT(0, 0, 0), xfT(1.8f, 0, 0), 0.02f, 0x0123u, 0x1234u });
    // The prior allflags word is allflagsStart(byte 0) | nbPatches(byte 1) | statusFlag(byte 2)
    // | prevPatches(byte 3) (PxsContactManagerState.h:74-77; the CUDA reads the status with
    // u16Low(u32High(allflags)), contactPatchUtils.cuh:81).  0x00000203 and 0x00000001 put
    // 3 / 1 in byte 0, NOT in the status: the status-byte priors are the three below.
    checkBoxBox(cfgs, 0x00000000u, false, "axisAligned/prior0");
    checkBoxBox(cfgs, 0x00000203u, false, "axisAligned/prior203");   // nbPatches=2 -> prevPatches=2; status 0
    checkBoxBox(cfgs, 0x00020103u, false, "axisAligned/priorKnownTouch");    // eHAS_TOUCH, 1 patch
    checkBoxBox(cfgs, 0x00010000u, false, "axisAligned/priorKnownNoTouch");  // eHAS_NO_TOUCH, 0 patches
    checkBoxBox(cfgs, 0x7C4A0200u, false, "axisAligned/priorHighBits");      // status 0x4A: bits 3 and 6 must survive

    // P9.88: bit 16 on the material indices (k < 8 so the second bank is hit).
    std::vector<PairCfg> hi;
    hi.push_back({ Vec3(1, 1, 1), Vec3(1, 1, 1), xfT(0, 0, 0), xfT(1.8f, 0, 0), 0.02f, 0x10001u, 2 });
    hi.push_back({ Vec3(1, 1, 1), Vec3(1, 1, 1), xfT(0, 0, 0), xfT(1.8f, 0, 0), 0.02f, 3, 0x10006u });
    hi.push_back({ Vec3(1.5f, 1, 0.8f), Vec3(0.6f, 0.7f, 1.2f), xfT(0, 0, 0), xfT(0.3f, 1.5f, 0.2f), 0.05f, 0x10004u, 0x10005u });
    checkBoxBox(hi, 0x00000000u, false, "axisAligned/matHi");
}

PXG_TEST(boxBoxNphase_rotated)
{
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> ang(-0.35f, 0.35f);
    std::uniform_real_distribution<float> off(-0.4f, 0.4f);
    std::vector<PairCfg> cfgs;
    for (int i = 0; i < 40; ++i)
    {
        // small random rotation about a random axis for box B, overlapping box A
        const float a = ang(rng);
        Vec3 axisv(off(rng), off(rng), off(rng));
        axisv = axisv.getNormalized();
        const float s = std::sin(a * 0.5f), c = std::cos(a * 0.5f);
        QuatV q(axisv.x * s, axisv.y * s, axisv.z * s, c);
        const Vec3 b0(1.0f, 1.2f, 0.9f), b1(0.8f, 1.0f, 1.1f);
        // place B so the boxes overlap along one axis with a healthy margin
        const int axisSel = i % 3;
        Vec3 p(off(rng), off(rng), off(rng));
        const float sep = 1.5f; // < b0+b1 half-extent sum on that axis -> overlap
        if (axisSel == 0) p.x = sep; else if (axisSel == 1) p.y = sep; else p.z = sep;
        cfgs.push_back({ b0, b1, xfQ(QuatV(0, 0, 0, 1), Vec3(0, 0, 0)), xfQ(q, p), 0.03f,
                         uint32_t(i % 7), uint32_t((i + 3) % 7) });
    }
    checkBoxBox(cfgs, 0x00000001u, false, "rotated");
}

PXG_TEST(boxBoxNphase_mutation)
{
    std::vector<PairCfg> cfgs;
    cfgs.push_back({ Vec3(1, 1, 1), Vec3(1, 1, 1), xfT(0, 0, 0), xfT(1.7f, 0, 0), 0.02f, 1, 2 });
    checkBoxBox(cfgs, 0x0u, true, "mutation");
}

} // namespace
