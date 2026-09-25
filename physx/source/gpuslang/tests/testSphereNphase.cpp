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

// Verification of npSphereNphase.slang (sphereNphase_Kernel) on the ENGINE launch
// ABI against a C++ transcription of sphereNphase_Kernel (gpunarrowphase/src/CUDA/
// cudaSphere.cu) + the sphere/capsule primitive routines (sphereCollision.cuh) +
// the contact-output tail (contactPatchUtils.cuh).  The kernel is driven through
// PxgCudaShim as the 19-argument CUDA parameter pack packed to a 136-byte
// natural-alignment blob; the element structs (PxgContactManagerInput /
// PxsCachedTransform / PxgShape / PxsContactManagerOutput / PxgPatchAndContact-
// Counters / PxsMaterialData / PxContact stream / PxContactPatch stream) are filled
// with the real engine byte layouts so the read/write comparison validates the ABI
// strides too.  Coverage: sphere-sphere (overlap + separated), sphere-plane (rotated
// "ground"), sphere-box (inside-face + outside-corner), sphere-capsule, plane-capsule
// (two contacts), capsule-capsule, and the shape-order flip path (box-vs-sphere ->
// sphere first, contact normal negated).

#include "pxgTestFramework.h"
#include "pxgGpuStructs.h"
#include "pxgRefMath.h"
#include "PxgCudaShim.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

namespace
{
using namespace pxgtest;

static const uint32_t INVALID = 0xffffffffu;
static const float PX_EPS_REAL = 1.192092896e-07f;

// PxGeometryType
enum GeomType { GT_SPHERE = 0, GT_PLANE = 1, GT_CAPSULE = 2, GT_BOX = 3 };

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

static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ---------------------------------------------------------------------------
// CPU reference : direct transcription of sphereCollision.cuh + the kernel
// dispatch/flip in cudaSphere.cu.
// ---------------------------------------------------------------------------
struct RC
{
    int nb = 0;
    Vec3 pt[2];
    float pen[2] = { 0.f, 0.f };
    Vec3 normal;
};

static float distancePointSegmentSquared(Vec3 a, Vec3 b, Vec3 p, float& param)
{
    const Vec3 ap = p - a, ab = b - a;
    const float nom = ap.dot(ab), denom = ab.dot(ab);
    const float tValue = clampf(nom / denom, 0.f, 1.f);
    const float t = denom > 0.f ? tValue : 0.f;
    const Vec3 v = ap - ab * t;
    param = t;
    return v.dot(v);
}

static void distanceSegmentSegmentSquared(Vec3 p1, Vec3 d1, Vec3 p2, Vec3 d2, float& s, float& t)
{
    const float eps = 1e-6f;
    const Vec3 r = p1 - p2;
    const float a = d1.dot(d1), e = d2.dot(d2), b = d1.dot(d2), c = d1.dot(r);
    const float aRecip = a > eps ? (1.f / a) : 0.f;
    const float eRecip = e > eps ? (1.f / e) : 0.f;
    const float f = d2.dot(r);
    const float denom = a * e - b * b;
    const float temp = b * f - c * e;
    const float s0 = clampf(temp / denom, 0.f, 1.f);
    const float sTmp = eps > denom ? 0.5f : s0;
    const float tTmp = (b * sTmp + f) * eRecip;
    const float t2 = clampf(tTmp, 0.f, 1.f);
    const float comp = (b * t2 - c) * aRecip;
    s = clampf(comp, 0.f, 1.f);
    t = t2;
}

static void pcmDistancePointSegmentTValue22(Vec3 a0, Vec3 b0, Vec3 a1, Vec3 b1,
                                            Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3,
                                            float& t0, float& t1, float& t2, float& t3)
{
    const Vec3 ab0 = b0 - a0, ab1 = b1 - a1;
    const float nom00 = (p0 - a0).dot(ab0), nom10 = (p1 - a0).dot(ab0);
    const float nom01 = (p2 - a1).dot(ab1), nom11 = (p3 - a1).dot(ab1);
    const float denom0 = ab0.dot(ab0), denom1 = ab1.dot(ab1);
    const float eps = 1e-6f;
    t0 = (std::fabs(denom0) < eps) ? 0.f : (nom00 / denom0);
    t1 = (std::fabs(denom0) < eps) ? 0.f : (nom10 / denom0);
    t2 = (std::fabs(denom1) < eps) ? 0.f : (nom01 / denom1);
    t3 = (std::fabs(denom1) < eps) ? 0.f : (nom11 / denom1);
}

static RC refSphereSphere(TransformV t0, TransformV t1, float r0, float r1, float cd)
{
    RC R;
    const Vec3 delta = t0.p - t1.p;
    const float distanceSq = delta.dot(delta);
    const float radiusSum = r0 + r1, inflatedSum = radiusSum + cd;
    if ((inflatedSum * inflatedSum) > distanceSq)
    {
        const float eps = 1e-4f, dist = std::sqrt(distanceSq);
        const Vec3 normal = (eps >= dist) ? Vec3(1, 0, 0) : delta * (1.f / dist);
        R.pt[0] = t1.p + normal * r1; R.pen[0] = dist - radiusSum; R.normal = normal; R.nb = 1;
    }
    return R;
}

static RC refSpherePlane(TransformV t0, TransformV t1, float r, float cd)
{
    RC R;
    const Vec3 sphereCenterInPlaneSpace = t1.q.rotateInv(t0.p - t1.p);
    const float separation = sphereCenterInPlaneSpace.x - r;
    if (cd >= separation)
    {
        const Vec3 worldNormal = t1.q.getBasisVector0();
        R.pt[0] = t0.p - worldNormal * r; R.pen[0] = separation; R.normal = worldNormal; R.nb = 1;
    }
    return R;
}

static RC refSphereCapsule(TransformV t0, TransformV t1, float sphereR, float capR, float hh, float cd)
{
    RC R;
    const Vec3 sphereCenter = t0.p;
    const Vec3 tmp0 = t1.q.getBasisVector0() * hh;
    const Vec3 s = t1.p + tmp0, e = t1.p - tmp0;
    const float radiusSum = sphereR + capR, inflatedSum = radiusSum + cd;
    float t; const float squareDist = distancePointSegmentSquared(s, e, sphereCenter, t);
    if ((inflatedSum * inflatedSum) > squareDist)
    {
        const Vec3 p = s + (e - s) * t;
        const Vec3 dir = sphereCenter - p;
        const float length = std::sqrt(dir.dot(dir));
        const Vec3 normal = length > PX_EPS_REAL ? dir * (1.f / length) : Vec3(1, 0, 0);
        R.pt[0] = sphereCenter - normal * sphereR;
        R.pen[0] = std::sqrt(squareDist) - radiusSum; R.normal = normal; R.nb = 1;
    }
    return R;
}

static RC refSphereBox(TransformV t0, TransformV t1, float sphereR, Vec3 boxExtents, float cd)
{
    RC R;
    const Vec3 sphereOrigin = t0.p;
    const Vec3 sphereCenter = t1.q.rotateInv(sphereOrigin - t1.p);
    const Vec3 nBox = -boxExtents;
    const float inflatedSum = sphereR + cd, sqInflatedSum = inflatedSum * inflatedSum;
    const Vec3 p(clampf(sphereCenter.x, nBox.x, boxExtents.x),
                clampf(sphereCenter.y, nBox.y, boxExtents.y),
                clampf(sphereCenter.z, nBox.z, boxExtents.z));
    const Vec3 v = sphereCenter - p;
    const float lengthSq = v.dot(v);
    if (sqInflatedSum > lengthSq)
    {
        const bool bInside = boxExtents.x >= std::fabs(sphereCenter.x) &&
                             boxExtents.y >= std::fabs(sphereCenter.y) &&
                             boxExtents.z >= std::fabs(sphereCenter.z);
        if (bInside)
        {
            const Vec3 absP(std::fabs(p.x), std::fabs(p.y), std::fabs(p.z));
            const Vec3 distToSurface = boxExtents - absP;
            Vec3 localNorm = p.z >= 0.f ? Vec3(0, 0, 1) : Vec3(0, 0, -1);
            float dist = -distToSurface.z;
            if (distToSurface.x <= distToSurface.y && distToSurface.x <= distToSurface.z)
            { localNorm = p.x >= 0.f ? Vec3(1, 0, 0) : Vec3(-1, 0, 0); dist = -distToSurface.x; }
            else if (distToSurface.y <= distToSurface.z)
            { localNorm = p.y >= 0.f ? Vec3(0, 1, 0) : Vec3(0, -1, 0); dist = -distToSurface.y; }
            const Vec3 normal = t1.q.rotate(localNorm);
            R.pt[0] = sphereOrigin - normal * dist; R.pen[0] = dist - sphereR; R.normal = normal;
        }
        else
        {
            const float length = std::sqrt(lengthSq);
            const Vec3 locNorm = v * (1.f / length);
            R.normal = t1.q.rotate(locNorm);
            R.pt[0] = t1.q.rotate(p) + t1.p; R.pen[0] = length - sphereR;
        }
        R.nb = 1;
    }
    return R;
}

static RC refPlaneCapsule(TransformV t0, TransformV t1, float radius, float hh, float cd)
{
    RC R;
    const TransformV bToA = t0.transformInv(t1);
    const Vec3 planeNormal = t0.q.getBasisVector0().getNormalized();
    const Vec3 outN = -planeNormal;
    const Vec3 basis = bToA.q.getBasisVector0();
    const Vec3 tmp = basis * hh;
    const Vec3 s = bToA.p + tmp, e = bToA.p - tmp;
    const float inflatedRadius = radius + cd;
    int nb = 0;
    // t0.transform(point) == t0.q.rotate(point) + t0.p
    const float sd0 = s.x;
    if (inflatedRadius >= sd0) { R.pt[nb] = t0.q.rotate(s) + t0.p - planeNormal * radius; R.pen[nb] = sd0 - radius; nb++; }
    const float sd1 = e.x;
    if (inflatedRadius >= sd1) { R.pt[nb] = t0.q.rotate(e) + t0.p - planeNormal * radius; R.pen[nb] = sd1 - radius; nb++; }
    if (nb > 0) R.normal = outN;
    R.nb = nb;
    return R;
}

static RC refCapsuleCapsule(TransformV t0, TransformV t1, float r0, float hh0, float r1, float hh1, float cd)
{
    RC R;
    const Vec3 positionOffset = (t0.p + t1.p) * 0.5f;
    const Vec3 p0 = t0.p - positionOffset, p1 = t1.p - positionOffset;
    const Vec3 tmp0 = t0.q.getBasisVector0() * hh0;
    const Vec3 s0 = p0 + tmp0, e0 = p0 - tmp0, d0 = e0 - s0;
    const Vec3 tmp1 = t1.q.getBasisVector0() * hh1;
    const Vec3 s1 = p1 + tmp1, e1 = p1 - tmp1, d1 = e1 - s1;
    const float sumRadius = r0 + r1, inflatedSum = sumRadius + cd;
    const float inflatedSumSquared = inflatedSum * inflatedSum;
    const float a = d0.dot(d0), e = d1.dot(d1), eps = 1e-6f;
    float t0v, t1v; distanceSegmentSegmentSquared(s0, d0, s1, d1, t0v, t1v);
    const Vec3 closestA = s0 + d0 * t0v, closestB = s1 + d1 * t1v;
    const Vec3 vv = closestA - closestB;
    const float sqDist0 = vv.dot(vv);
    int nb = 0; Vec3 outN;
    if (eps > sqDist0)
    {
        const Vec3 normal = t0.q.rotateInv(Vec3(0, 1, 0)).getNormalized();
        R.pt[0] = (closestA - normal * r0) + positionOffset; R.pen[0] = -r0; outN = normal; nb = 1;
    }
    else if (inflatedSumSquared >= sqDist0)
    {
        const float parallelTolerance = 0.9998f;
        const Vec3 dir0 = (eps > a) ? Vec3(0) : d0 * (1.f / std::sqrt(a));
        const Vec3 dir1 = (eps > e) ? Vec3(0) : d1 * (1.f / std::sqrt(e));
        const float cosv = std::fabs(dir0.dot(dir1));
        if (cosv > parallelTolerance)
        {
            float tV0, tV1, tV2, tV3;
            pcmDistancePointSegmentTValue22(s0, e0, s1, e1, s1, e1, s0, e0, tV0, tV1, tV2, tV3);
            if (tV0 >= 0.f && tV0 <= 1.f) { const Vec3 pr = s0 + d0 * tV0; const Vec3 v = pr - s1; const float d = std::sqrt(v.dot(v)); const Vec3 n = v * (1.f / d); R.pt[nb] = (pr - n * r0) + positionOffset; R.pen[nb] = d - sumRadius; outN = n; nb++; }
            if (tV1 >= 0.f && tV1 <= 1.f) { const Vec3 pr = s0 + d0 * tV1; const Vec3 v = pr - e1; const float d = std::sqrt(v.dot(v)); const Vec3 n = v * (1.f / d); R.pt[nb] = (pr - n * r0) + positionOffset; R.pen[nb] = d - sumRadius; outN = n; nb++; }
            if (nb < 2 && tV2 >= 0.f && tV2 <= 1.f) { const Vec3 pr = s1 + d1 * tV2; const Vec3 v = s0 - pr; const float d = std::sqrt(v.dot(v)); const Vec3 n = v * (1.f / d); R.pt[nb] = (s0 - n * r0) + positionOffset; R.pen[nb] = d - sumRadius; outN = n; nb++; }
            if (nb < 2 && tV3 >= 0.f && tV3 <= 1.f) { const Vec3 pr = s1 + d1 * tV3; const Vec3 v = e0 - pr; const float d = std::sqrt(v.dot(v)); const Vec3 n = v * (1.f / d); R.pt[nb] = (e0 - n * r0) + positionOffset; R.pen[nb] = d - sumRadius; outN = n; nb++; }
        }
        if (nb == 0)
        {
            const Vec3 normal = vv.getNormalized();
            R.pt[0] = (closestA - normal * r0) + positionOffset;
            R.pen[0] = std::sqrt(sqDist0) - sumRadius; outN = normal; nb = 1;
        }
    }
    if (nb > 0) R.normal = outN;
    R.nb = nb;
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
struct PairCfg
{
    uint32_t type0; Vec3 scale0; TransformV xf0;
    uint32_t type1; Vec3 scale1; TransformV xf1;
    float cDist; uint32_t mat0, mat1;
};

// Mirrors cudaSphere.cu's dispatch: order shapes (lower type first), call the
// matching routine, negate the normal for the flipped ordering.
static RC refDispatch(const PairCfg& cfg)
{
    uint32_t type0 = cfg.type0, type1 = cfg.type1;
    Vec3 scale0 = cfg.scale0, scale1 = cfg.scale1;
    TransformV xf0 = cfg.xf0, xf1 = cfg.xf1;
    const bool flip = type1 < type0;
    if (flip) { std::swap(type0, type1); std::swap(scale0, scale1); std::swap(xf0, xf1); }
    const float sphereRadius = scale0.x;

    RC R;
    if (type0 == GT_SPHERE)
    {
        if (type1 == GT_SPHERE) R = refSphereSphere(xf0, xf1, sphereRadius, scale1.x, cfg.cDist);
        else if (type1 == GT_PLANE) R = refSpherePlane(xf0, xf1, sphereRadius, cfg.cDist);
        else if (type1 == GT_CAPSULE) R = refSphereCapsule(xf0, xf1, sphereRadius, scale1.y, scale1.x, cfg.cDist);
        else if (type1 == GT_BOX) R = refSphereBox(xf0, xf1, sphereRadius, scale1, cfg.cDist);
    }
    else if (type0 == GT_PLANE) R = refPlaneCapsule(xf0, xf1, scale1.y, scale1.x, cfg.cDist);
    else if (type0 == GT_CAPSULE) R = refCapsuleCapsule(xf0, xf1, scale0.y, scale0.x, scale1.y, scale1.x, cfg.cDist);

    if (R.nb && flip) R.normal = -R.normal;
    return R;
}

static HCachedTransform makeXf(const TransformV& t)
{
    return { t.q.x, t.q.y, t.q.z, t.q.w, t.p.x, t.p.y, t.p.z, 0u };
}
static HShape makeShape(const Vec3& scale, uint32_t mat, uint32_t type)
{
    return { scale.x, scale.y, scale.z, 0.f, 0.f, 0.f, 1.f, mat, 0u, 0u, type, 0u };
}

static std::vector<uint8_t> sphereShimLoader(const char* file) { return loadShaderBlob(file); }

static void checkSphere(const std::vector<PairCfg>& cfgs, uint32_t priorAllflags, bool mutate, const char* label)
{
    using namespace physx::gpuslang;
    const uint32_t n = uint32_t(cfgs.size());

    std::vector<HMaterial> materials(8);
    for (uint32_t i = 0; i < materials.size(); ++i)
    {
        const float f = 0.2f + 0.1f * i;
        materials[i] = { f, f + 0.3f, (i % 3 == 2) ? -0.5f : 0.4f, 0.05f * i,
                         uint32_t((i & 1) * 16u) | (uint32_t(i % 4) << 16) | (uint32_t((i + 1) % 4) << 24),
                         uint32_t((i + 2) % 4) };
    }

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
        shapes[2 * i + 0] = makeShape(cfgs[i].scale0, cfgs[i].mat0, cfgs[i].type0);
        shapes[2 * i + 1] = makeShape(cfgs[i].scale1, cfgs[i].mat1, cfgs[i].type1);
        inputs[i] = { 2 * i + 0, 2 * i + 1, 2 * i + 0, 2 * i + 1 };
    }

    std::vector<HOutput> outputs(n);
    std::memset(outputs.data(), 0, outputs.size() * sizeof(HOutput));
    for (auto& o : outputs) { o.allflagsWord = priorAllflags; o.nbContacts_flags = 0xAAAA0000u; }

    HCounters counters = { 0, 0, 0, 0 };
    const uint32_t CONTACT_LIMIT = 1u << 20, FORCE_LIMIT = 1u << 20, PATCH_LIMIT = 1u << 20;
    std::vector<uint8_t> contactStream(CONTACT_LIMIT, 0), patchStream(PATCH_LIMIT), forceStream(FORCE_LIMIT, 0);
    // Poisoned: insertIntoPatchStream (contactPatchUtils.cuh:161-178) stores field by field and never
    // writes pad bytes 54-63, and the engine's patch stream (mCompressedPatches) is never cleared, so
    // those bytes must come back exactly as uploaded.  The fill varies by byte, not a uniform 0xCD, so a
    // store that copies ANOTHER poisoned word into the pad is visible too.  The host vector is not
    // modified after the upload, so it is the expectation for every byte the kernel must not write.
    for (uint32_t k = 0; k < PATCH_LIMIT; ++k) patchStream[k] = uint8_t(0xCDu ^ (k * 37u));
    std::vector<uint32_t> touchFlags(n, 7u), patchFlags(n, 7u);

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = { { "npSphereNphase.spv", "sphereNphase_Kernel", 136u, 0 } };
    PxgCudaShim shim(dev, manifest, 1, &sphereShimLoader);

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
        uint32_t numTests; float tolerance;
        uint64_t cmInputs, cmOutputs, shapes, transformCache, contactDistance, materials, contactStream,
                 patchStream, counters, touch, patch, startPatch, startPoint, startForce;
        uint32_t patchLimit, contactLimit, forceLimit;
    } blob = { n, 1.0f, dIn, dOut, dShapes, dXf, dCd, dMat, dCStream, dPStream, dCounters, dTouch, dPatch,
               dStartPatch, dStartPoint, dStartForce, PATCH_LIMIT, CONTACT_LIMIT, FORCE_LIMIT };
    static_assert(sizeof(BlobT) == 136, "sphereNphase blob is 136 bytes");

    if (mutate) { const uint64_t s = blob.startPoint; blob.startPoint = blob.startForce; blob.startForce = s; }

    // engine grid: numthreads=64, one pair per thread -> numBlocks = ceil(n/64)
    const uint32_t numBlocks = (n + 63u) / 64u;
    PXG_CHECK_MSG(shim.launchKernel(0, numBlocks, 1, 1, &blob, sizeof(blob)) == PxgResult::Success,
                  "launch sphereNphase_Kernel (%s)", label);
    shim.streamSynchronize();

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

    size_t bad = 0;
    uint32_t expTotalContacts = 0, expTotalPatches = 0;
    const uint32_t oldStatus = (priorAllflags >> 16) & 0xffu;
    const uint32_t oldPrevPatches = (priorAllflags >> 8) & 0xffu;
    const bool prevHadTouch = (oldStatus & 2u) != 0u;
    const bool prevKnown = (oldStatus & 3u) != 0u;

    for (uint32_t i = 0; i < n; ++i)
    {
        const RC R = refDispatch(cfgs[i]);
        const uint32_t nb = uint32_t(R.nb);
        expTotalContacts += nb; if (nb) expTotalPatches++;

        const uint32_t gNb = gOut[i].nbContacts_flags & 0xffffu;
        if (gNb != nb) { ++bad; if (bad < 12) std::printf("  [%s] pair %u nbContacts gpu=%u ref=%u\n", label, i, gNb, nb); continue; }
        if ((gOut[i].nbContacts_flags >> 16) != 0xAAAAu) ++bad;

        // registerContactPatch (contactPatchUtils.cuh:81-89): the old status byte survives with
        // eTOUCH_KNOWN cleared, then eHAS_TOUCH or eHAS_NO_TOUCH is set.
        const uint32_t status = (oldStatus & ~3u) | (nb ? 2u : 1u);
        const uint32_t numPatches = nb ? 1u : 0u;
        const uint32_t expAllflags = (oldPrevPatches << 24) | (status << 16) | (numPatches << 8);
        if (gOut[i].allflagsWord != expAllflags) { ++bad; if (bad < 12) std::printf("  [%s] pair %u allflags gpu=%08x ref=%08x\n", label, i, gOut[i].allflagsWord, expAllflags); }
        const bool curTouch = nb > 0;
        const uint32_t expTouch = ((prevHadTouch != curTouch) || !prevKnown) ? 1u : 0u;
        const uint32_t expPatch = (oldPrevPatches != numPatches) ? 1u : 0u;
        if (gTouch[i] != expTouch) ++bad;
        if (gPatch[i] != expPatch) ++bad;

        if (nb == 0) continue;

        const uint64_t cpAddr = gOut[i].contactPoints;
        const uint64_t patchAddr = gOut[i].contactPatches;
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

        for (uint32_t k = 0; k < nb; ++k)
        {
            float f4[4];
            std::memcpy(f4, &gCStream[cOff + k * 16], 16);
            if (!(almostEqual(f4[0], R.pt[k].x, 2e-4f, 2e-5f) && almostEqual(f4[1], R.pt[k].y, 2e-4f, 2e-5f) &&
                  almostEqual(f4[2], R.pt[k].z, 2e-4f, 2e-5f) && almostEqual(f4[3], R.pen[k], 2e-4f, 2e-5f)))
            {
                ++bad;
                if (bad < 12) std::printf("  [%s] pair %u contact %u gpu=(%g,%g,%g,%g) ref=(%g,%g,%g,%g)\n",
                                          label, i, k, f4[0], f4[1], f4[2], f4[3], R.pt[k].x, R.pt[k].y, R.pt[k].z, R.pen[k]);
            }
        }

        float pf[16];
        std::memcpy(pf, &gPStream[pOff], 64);
        const uint32_t* pu = reinterpret_cast<const uint32_t*>(pf);
        if (!(almostEqual(pf[4], R.normal.x, 2e-4f, 2e-5f) && almostEqual(pf[5], R.normal.y, 2e-4f, 2e-5f) &&
              almostEqual(pf[6], R.normal.z, 2e-4f, 2e-5f)))
        { ++bad; if (bad < 12) std::printf("  [%s] pair %u patch normal gpu=(%g,%g,%g) ref=(%g,%g,%g)\n", label, i, pf[4], pf[5], pf[6], R.normal.x, R.normal.y, R.normal.z); }

        float sta, dyn, rest, damping; uint32_t mflags;
        combineMaterials(materials[cfgs[i].mat0], materials[cfgs[i].mat1], sta, dyn, rest, damping, mflags);
        if (!almostEqual(pf[7], rest, 2e-5f, 2e-6f)) ++bad;
        if (!almostEqual(pf[8], dyn, 2e-5f, 2e-6f)) ++bad;
        if (!almostEqual(pf[9], sta, 2e-5f, 2e-6f)) ++bad;
        if (!almostEqual(pf[10], damping, 2e-5f, 2e-6f)) ++bad;
        const uint32_t word11 = ((nb & 0xffu) << 16) | ((mflags & 0xffu) << 24);
        if (pu[11] != word11) ++bad;
        if (pu[12] != ((cfgs[i].mat0 & 0xffffu) << 16)) ++bad;
        // mMassModification = 1.0f x4 (contactPatchUtils.cuh:175-178).
        for (uint32_t w = 0; w < 4u; ++w)
            if (pu[w] != 0x3f800000u) { ++bad; if (bad < 12) std::printf("  [%s] pair %u patch word %u gpu=%08x ref=3f800000\n", label, i, w, pu[w]); }
        uint32_t init[16];
        std::memcpy(init, &patchStream[pOff], 64);   // the header's bytes as uploaded
        // :171 is a PxU16 store: pad[0] (bytes 54-55) keeps the upload; pad[1..4] (words 14-15) are
        // never written (P9.87).
        const uint32_t expW13 = (init[13] & 0xffff0000u) | (cfgs[i].mat1 & 0xffffu);
        if (pu[13] != expW13) { ++bad; if (bad < 12) std::printf("  [%s] pair %u patch word 13 gpu=%08x ref=%08x\n", label, i, pu[13], expW13); }
        if (pu[14] != init[14] || pu[15] != init[15])
        { ++bad; if (bad < 12) std::printf("  [%s] pair %u patch pad words gpu=%08x,%08x ref=%08x,%08x\n", label, i, pu[14], pu[15], init[14], init[15]); }
    }

    // Nothing past the reserved patches may be written.
    for (size_t k = std::min<size_t>(gCnt.patchesBytes, gPStream.size()); k < gPStream.size(); ++k)
        if (gPStream[k] != patchStream[k]) { ++bad; std::printf("  [%s] patch stream byte %zu written past patchesBytes\n", label, k); break; }

    if (gCnt.contactsBytes != expTotalContacts * 16u) { ++bad; std::printf("  [%s] contactsBytes gpu=%u ref=%u\n", label, gCnt.contactsBytes, expTotalContacts * 16u); }
    if (gCnt.patchesBytes != expTotalPatches * 64u) { ++bad; std::printf("  [%s] patchesBytes gpu=%u ref=%u\n", label, gCnt.patchesBytes, expTotalPatches * 64u); }
    if (gCnt.forceAndIndiceBytes != expTotalContacts * 4u) ++bad;
    if (gCnt.overflowError != 0u) ++bad;

    std::printf("  [%s] pairs=%u contacts=%u patches=%u\n", label, n, expTotalContacts, expTotalPatches);

    if (mutate)
        PXG_CHECK_MSG(bad != 0, "[%s] mutation must FAIL (swapped startPoint/startForce)", label);
    else
        PXG_CHECK_MSG(bad == 0, "[%s] %zu mismatches", label, bad);

    for (PxgDevicePtr p : { dIn, dOut, dShapes, dXf, dCd, dMat, dCStream, dPStream, dCounters, dTouch, dPatch, dStartForce })
        shim.memFree(p);
}

static TransformV xfT(float x, float y, float z) { return TransformV(QuatV(0, 0, 0, 1), Vec3(x, y, z)); }
static TransformV xfQ(const QuatV& q, Vec3 p) { return TransformV(q.getNormalized(), p); }

// Build a set of pairs spanning every dispatched sub-case.
static std::vector<PairCfg> buildCases()
{
    std::vector<PairCfg> cfgs;
    const float s45 = std::sin(0.7853981634f), c45 = std::cos(0.7853981634f); // 45 deg

    // 0: sphere-sphere overlapping
    cfgs.push_back({ GT_SPHERE, Vec3(1, 1, 1), xfT(0, 0, 0), GT_SPHERE, Vec3(1, 1, 1), xfT(1.5f, 0, 0), 0.02f, 1, 2 });
    // 1: sphere-sphere separated -> no contact
    cfgs.push_back({ GT_SPHERE, Vec3(1, 1, 1), xfT(0, 0, 0), GT_SPHERE, Vec3(1, 1, 1), xfT(5.0f, 0, 0), 0.02f, 0, 3 });
    // 2: sphere-plane, plane rotated so its normal is +Y (a "ground" a dynamic sphere rests on).
    //    q = +90 deg about Z maps local +x -> world +y.
    cfgs.push_back({ GT_SPHERE, Vec3(0.5f, 0.5f, 0.5f), xfT(0.0f, 0.45f, 0.0f),
                     GT_PLANE, Vec3(1, 1, 1), xfQ(QuatV(0, 0, s45, c45), Vec3(0, 0, 0)), 0.02f, 4, 5 });
    // 3: sphere-box, sphere centre inside the box near the +Z face
    cfgs.push_back({ GT_SPHERE, Vec3(0.5f, 0.5f, 0.5f), xfT(0, 0, 0.9f),
                     GT_BOX, Vec3(1, 1, 1), xfT(0, 0, 0), 0.02f, 6, 1 });
    // 4: sphere-box, sphere centre outside the box (corner region) -> closest-point branch
    cfgs.push_back({ GT_SPHERE, Vec3(0.5f, 0.5f, 0.5f), xfT(1.2f, 1.2f, 0),
                     GT_BOX, Vec3(1, 1, 1), xfT(0, 0, 0), 0.05f, 2, 4 });
    // 5: sphere-capsule (capsule axis = local x)
    cfgs.push_back({ GT_SPHERE, Vec3(0.5f, 0.5f, 0.5f), xfT(0, 0, 0),
                     GT_CAPSULE, Vec3(1.0f /*hh*/, 0.4f /*r*/, 0), xfT(0, 1.0f, 0), 0.2f, 3, 6 });
    // 6: flip path: box vs sphere (shape0=box, shape1=sphere) -> ordered sphere-first, normal negated
    cfgs.push_back({ GT_BOX, Vec3(1, 1, 1), xfT(0, 0, 0),
                     GT_SPHERE, Vec3(0.5f, 0.5f, 0.5f), xfT(0, 0, 0.9f), 0.02f, 5, 6 });
    // 7: plane-capsule, capsule parallel to the plane surface -> two contacts.
    //    plane normal +x (identity); capsule axis rotated to world +y so both ends touch.
    cfgs.push_back({ GT_PLANE, Vec3(1, 1, 1), xfT(0, 0, 0),
                     GT_CAPSULE, Vec3(1.0f /*hh*/, 0.3f /*r*/, 0), xfQ(QuatV(0, 0, s45, c45), Vec3(0.1f, 0, 0)), 0.05f, 1, 2 });
    // 8: capsule-capsule, perpendicular axes -> general single-contact branch
    cfgs.push_back({ GT_CAPSULE, Vec3(1.0f, 0.3f, 0), xfT(0, 0, 0),
                     GT_CAPSULE, Vec3(1.0f, 0.3f, 0), xfQ(QuatV(0, 0, s45, c45), Vec3(0, 0, 0.4f)), 0.05f, 4, 0 });
    return cfgs;
}

PXG_TEST(sphereNphase_allSubCases)
{
    const std::vector<PairCfg> cfgs = buildCases();
    // The prior allflags word is allflagsStart(byte 0) | nbPatches(byte 1) | statusFlag(byte 2)
    // | prevPatches(byte 3) (PxsContactManagerState.h:74-77; the CUDA reads the status with
    // u16Low(u32High(allflags)), contactPatchUtils.cuh:81).  0x00000203 and 0x00000001 put
    // 3 / 1 in byte 0, NOT in the status: the status-byte priors are the three below.
    checkSphere(cfgs, 0x00000000u, false, "subCases/prior0");
    checkSphere(cfgs, 0x00000203u, false, "subCases/prior203");   // nbPatches=2 -> prevPatches=2; status 0
    checkSphere(cfgs, 0x00000001u, false, "subCases/prior1");     // allflagsStart=1; status 0
    checkSphere(cfgs, 0x00020103u, false, "subCases/priorKnownTouch");    // eHAS_TOUCH, 1 patch
    checkSphere(cfgs, 0x00010000u, false, "subCases/priorKnownNoTouch");  // eHAS_NO_TOUCH, 0 patches
    checkSphere(cfgs, 0x7C4A0200u, false, "subCases/priorHighBits");      // status 0x4A: bits 3 and 6 must survive
}

PXG_TEST(sphereNphase_mutation)
{
    std::vector<PairCfg> cfgs;
    // a single overlapping sphere-box pair; swapping startPoint/startForce must be caught.
    cfgs.push_back({ GT_SPHERE, Vec3(0.5f, 0.5f, 0.5f), xfT(0, 0, 0.9f),
                     GT_BOX, Vec3(1, 1, 1), xfT(0, 0, 0), 0.02f, 1, 2 });
    checkSphere(cfgs, 0x0u, true, "mutation");
}

} // namespace
