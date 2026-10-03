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

#include <array>
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

// P9.90: every PxVec3 dot of the reference below goes through rdot.  gDotOld = false is the
// CUDA's PxVec3::dot, x*x' + y*y' + z*z' = (x + y) + z (PxVec3.h:173); true is the old-order
// twin x + (y + z), how llvmpipe (Mesa lower_fdot) sums an UNDECORATED OpDot.
// sphereNphase_p90DotOrder requires the twin to change every observed word.
static bool gDotOld = false;
static float rdot(const Vec3& a, const Vec3& b)
{
    return gDotOld ? a.x * b.x + (a.y * b.y + a.z * b.z) : a.dot(b);
}

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
    const float nom = rdot(ap, ab), denom = rdot(ab, ab);
    const float tValue = clampf(nom / denom, 0.f, 1.f);
    const float t = denom > 0.f ? tValue : 0.f;
    const Vec3 v = ap - ab * t;
    param = t;
    return rdot(v, v);
}

// P9.86c coverage for sphereNphase_capsuleCapsuleZeroSign: how often a -0 reaches the two PxClamp
// selects below (distanceSegmentSegment.cuh:76/:81).  clampf(v, 0, 1) is PxClamp's
// selectMin(1, selectMax(0, v)) for every float, -0 and NaN included.
static size_t gSegTTmpNegZero = 0, gSegCompNegZero = 0;

static void distanceSegmentSegmentSquared(Vec3 p1, Vec3 d1, Vec3 p2, Vec3 d2, float& s, float& t)
{
    const float eps = 1e-6f;
    const Vec3 r = p1 - p2;
    const float a = rdot(d1, d1), e = rdot(d2, d2), b = rdot(d1, d2), c = rdot(d1, r);
    const float aRecip = a > eps ? (1.f / a) : 0.f;
    const float eRecip = e > eps ? (1.f / e) : 0.f;
    const float f = rdot(d2, r);
    const float denom = a * e - b * b;
    const float temp = b * f - c * e;
    const float s0 = clampf(temp / denom, 0.f, 1.f);
    const float sTmp = eps > denom ? 0.5f : s0;
    const float tTmp = (b * sTmp + f) * eRecip;
    if (tTmp == 0.f && std::signbit(tTmp)) ++gSegTTmpNegZero;
    const float t2 = clampf(tTmp, 0.f, 1.f);
    const float comp = (b * t2 - c) * aRecip;
    if (comp == 0.f && std::signbit(comp)) ++gSegCompNegZero;
    s = clampf(comp, 0.f, 1.f);
    t = t2;
}

static void pcmDistancePointSegmentTValue22(Vec3 a0, Vec3 b0, Vec3 a1, Vec3 b1,
                                            Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3,
                                            float& t0, float& t1, float& t2, float& t3)
{
    const Vec3 ab0 = b0 - a0, ab1 = b1 - a1;
    const float nom00 = rdot((p0 - a0), ab0), nom10 = rdot((p1 - a0), ab0);
    const float nom01 = rdot((p2 - a1), ab1), nom11 = rdot((p3 - a1), ab1);
    const float denom0 = rdot(ab0, ab0), denom1 = rdot(ab1, ab1);
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
    const float distanceSq = rdot(delta, delta);
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
        const float length = std::sqrt(rdot(dir, dir));
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
    const float lengthSq = rdot(v, v);
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

// P9.94b twin of planeCapsuleContact's bToA.p (sphereCollision.cuh:527): false = the CUDA's
// PxTransform::transformInv (conj(q).rotate(p1 - p0), pxgRefMath.h TransformV::transformInv),
// true = the pre-P9.94b npSpherePrimitives spelling q.rotateInv(p1 - p0).
static bool gPlaneCapsuleRotInvTwin = false;

static RC refPlaneCapsule(TransformV t0, TransformV t1, float radius, float hh, float cd)
{
    RC R;
    TransformV bToA = t0.transformInv(t1);
    if (gPlaneCapsuleRotInvTwin) bToA.p = t0.q.rotateInv(t1.p - t0.p);
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
    const float a = rdot(d0, d0), e = rdot(d1, d1), eps = 1e-6f;
    float t0v, t1v; distanceSegmentSegmentSquared(s0, d0, s1, d1, t0v, t1v);
    const Vec3 closestA = s0 + d0 * t0v, closestB = s1 + d1 * t1v;
    const Vec3 vv = closestA - closestB;
    const float sqDist0 = rdot(vv, vv);
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
        const float cosv = std::fabs(rdot(dir0, dir1));
        if (cosv > parallelTolerance)
        {
            float tV0, tV1, tV2, tV3;
            pcmDistancePointSegmentTValue22(s0, e0, s1, e1, s1, e1, s0, e0, tV0, tV1, tV2, tV3);
            if (tV0 >= 0.f && tV0 <= 1.f) { const Vec3 pr = s0 + d0 * tV0; const Vec3 v = pr - s1; const float d = std::sqrt(rdot(v, v)); const Vec3 n = v * (1.f / d); R.pt[nb] = (pr - n * r0) + positionOffset; R.pen[nb] = d - sumRadius; outN = n; nb++; }
            if (tV1 >= 0.f && tV1 <= 1.f) { const Vec3 pr = s0 + d0 * tV1; const Vec3 v = pr - e1; const float d = std::sqrt(rdot(v, v)); const Vec3 n = v * (1.f / d); R.pt[nb] = (pr - n * r0) + positionOffset; R.pen[nb] = d - sumRadius; outN = n; nb++; }
            if (nb < 2 && tV2 >= 0.f && tV2 <= 1.f) { const Vec3 pr = s1 + d1 * tV2; const Vec3 v = s0 - pr; const float d = std::sqrt(rdot(v, v)); const Vec3 n = v * (1.f / d); R.pt[nb] = (s0 - n * r0) + positionOffset; R.pen[nb] = d - sumRadius; outN = n; nb++; }
            if (nb < 2 && tV3 >= 0.f && tV3 <= 1.f) { const Vec3 pr = s1 + d1 * tV3; const Vec3 v = e0 - pr; const float d = std::sqrt(rdot(v, v)); const Vec3 n = v * (1.f / d); R.pt[nb] = (e0 - n * r0) + positionOffset; R.pen[nb] = d - sumRadius; outN = n; nb++; }
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

static uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// GLSL.std.450 FMin/FMax as llvmpipe evaluates them under the SignedZeroInfNanPreserve stamp
// (P9.86 F4; nir_lower_alu with lower_fminmax_signed_zero: feq(a,b) ? imin/imax(bits) : fmin/fmax):
// ordered min/max, and on a +-0 tie min gives -0 and max +0 in either operand order.
static float fminLp(float a, float b) { return (a == b) ? (std::signbit(a) ? a : b) : (a < b ? a : b); }
static float fmaxLp(float a, float b) { return (a == b) ? (std::signbit(a) ? b : a) : (a > b ? a : b); }

// P9.86c mutant twins of the combine below (0 = the CUDA): 1 replays PxMin as FMin
// (npSphereNphase-r0), 2 PxMax as FMax (npSphereNphase-r1), both as llvmpipe evaluates them.
// sphereNphase_materialZeroSign requires each twin to change the expected patch words.
// There is deliberately NO twin for the fDynFriction clamp (npSphereNphase-r2,
// `max(dynF, 0.0f)`): FMax with a literal +0 source is lowered by nir_lower_alu
// (lower_fminmax_signed_zero, constant pos_zero && max branch) to bcsel(flt(+0, dynF), dynF, +0),
// which is the select `dynF > 0 ? dynF : 0` itself, so that revert is equivalent on this
// driver for every 32-bit dynF.  The case still delivers dynFriction = -0 to the clamp.
static int gMatVariant = 0;

// PxsCombineMaterials (PxsMaterialCombiner.h:36-170, reached through materialCombiner.cuh:46 from
// insertIntoPatchStream, contactPatchUtils.cuh:149-159, with the UNSWAPPED shape0/shape1).
// PxMin/PxMax<float> are intrinsics::selectMin/selectMax (PxMath.h:82-101,
// PxUnixMathIntrinsics.h:116-125): `a < b ? a : b` / `a > b ? a : b`, so a +-0 tie keeps b --
// NOT std::fmin/std::fmax, whose zero sign is unspecified.
static void combineMaterials(const HMaterial& m0, const HMaterial& m1, float& sta, float& dyn,
                             float& rest, float& damping, uint32_t& flags, float* dynRaw = nullptr)
{
    const uint32_t DISABLE_FRICTION = 1u, DISABLE_STRONG = 2u, ACC_SPRING = 16u;
    auto combine = [](float a, float b, uint32_t mode) -> float {
        if (mode == 0) return 0.5f * (a + b);                                                // :41/:57
        if (mode == 1) return (gMatVariant == 1) ? fminLp(a, b) : ((a < b) ? a : b);        // :43/:60
        if (mode == 2) return a * b;                                                         // :45/:62
        if (mode == 3) return (gMatVariant == 2) ? fmaxLp(a, b) : ((a > b) ? a : b);        // :47/:64
        return 0.f; };
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
    if ((cf & DISABLE_FRICTION) == 0)
    {
        const uint32_t fm = std::max(fr0, fr1);
        const float d = combine(m0.dyn, m1.dyn, fm), s = combine(m0.sta, m1.sta, fm);
        const float fd = (d > 0.f) ? d : 0.f;          // :161 PxMax(dynFriction, 0.0f) = selectMax
        if (dynRaw) *dynRaw = d;
        dyn = fd; sta = (s - fd) >= 0.f ? s : fd; flags = cf;   // :164 (CUDA branch)
    }
    else { flags = cf | DISABLE_STRONG; dyn = 0.f; sta = 0.f; if (dynRaw) *dynRaw = 0.f; }
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

// P9.86c signed-zero material bank at ZS_BANK + k (below 0x10000, clear of bank 1 at 0..7 and
// bank 2 at 0x10000..7).  Every value is accepted by the PxMaterial setters: NpMaterial.cpp:110/:125
// check only PxIsFinite (friction; k6's negative frictions pass that check but lie outside the
// documented range [0, PX_MAX_F32) of include/PxMaterial.h:152/:174, and feed only the r2
// coverage, never a kill), :140-141 finite and <= 1 (restitution; -0 < 0 is false, so -0 is a
// rigid, non-compliant material), :161 finite and >= 0 (damping; -0 >= 0 holds).  Combine modes:
// packedFlags bits 16-23 friction, 24-31 restitution, packedDamp bits 0-7 damping
// (eAVERAGE 0, eMIN 1, eMULTIPLY 2, eMAX 3).
static const uint32_t ZS_BANK = 0x2000u;
static const uint32_t ZS_COUNT = 8u;
static HMaterial zeroSignMaterial(uint32_t k)
{
    const float pz = 0.0f, nz = -0.0f;
    auto modes = [](uint32_t fric, uint32_t restm) { return (fric << 16) | (restm << 24); };
    switch (k)
    {
    case 0: return { pz, pz, pz, pz, modes(1, 1), 1u };          // +0 everywhere, eMIN
    case 1: return { nz, nz, nz, nz, modes(1, 1), 1u };          // -0 everywhere, eMIN
    case 2: return { pz, pz, pz, pz, modes(3, 3), 3u };          // +0 everywhere, eMAX
    case 3: return { nz, nz, nz, nz, modes(3, 3), 3u };          // -0 everywhere, eMAX
    case 4: return { 0.3f, 0.5f, -0.5f, pz, modes(0, 0), 0u };   // compliant (rest < 0), damping +0
    case 5: return { 0.3f, 0.5f, 0.4f, nz, modes(0, 0), 0u };    // rigid, damping -0
    case 6: return { -0.5f, -0.25f, nz, 0.2f, modes(2, 2), 2u }; // eMULTIPLY, negative friction
    default: return { pz, 0.3f, 0.5f, nz, modes(2, 2), 2u };     // eMULTIPLY, zero dynamic friction
    }
}

// strictZero (P9.86c): additionally require the exact bits on every contact-point and normal lane
// whose CUDA-mirror value is +-0 (the sign of zero is an output bit).  Opt-in: only the directed
// configurations are derived to be exact on their zero lanes (every product reaching them is exact,
// so fma contraction cannot change them); the other configurations keep the tolerance compare.
// comparedWords (optional, P9.86c): per pair, whether patch words 7-10 were bitwise compared
// (nb > 0 and the output pointers in range) and the GPU's words 7-10, so a caller can tie its
// coverage to the GPU output rather than to the host mirror alone.
struct ComparedPatch { bool compared; uint32_t w[4]; };
// gpuWords (optional, P9.90): per pair, the GPU's contact count, contact words (point, pen) and
// patch normal words 4-6, for a caller that compares them bitwise.
struct GpuContactWords { uint32_t nb; uint32_t c[2][4]; uint32_t n[3]; };
static void checkSphere(const std::vector<PairCfg>& cfgs, uint32_t priorAllflags, bool mutate, const char* label,
                        bool strictZero = false, std::vector<ComparedPatch>* comparedWords = nullptr,
                        std::vector<GpuContactWords>* gpuWords = nullptr)
{
    using namespace physx::gpuslang;
    const uint32_t n = uint32_t(cfgs.size());

    std::vector<HMaterial> materials(0x10000u + 8u);
    for (uint32_t i = 0; i < 8u; ++i)
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
    for (uint32_t k = 0; k < ZS_COUNT; ++k) materials[ZS_BANK + k] = zeroSignMaterial(k);

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
    if (comparedWords) comparedWords->assign(n, ComparedPatch{ false, { 0u, 0u, 0u, 0u } });
    if (gpuWords) gpuWords->assign(n, GpuContactWords{});

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
        if (gpuWords)
        {
            GpuContactWords& g = (*gpuWords)[i];
            g.nb = nb;
            std::memcpy(g.c, &gCStream[cOff], nb * 16u);
            std::memcpy(g.n, &gPStream[pOff + 16u], 12u);
        }

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
            if (strictZero)
            {
                const float rv[4] = { R.pt[k].x, R.pt[k].y, R.pt[k].z, R.pen[k] };
                for (uint32_t l = 0; l < 4u; ++l)
                    if (rv[l] == 0.f && fbits(f4[l]) != fbits(rv[l]))
                    { ++bad; if (bad < 12) std::printf("  [%s] pair %u contact %u lane %u gpu=%08x ref=%08x\n", label, i, k, l, fbits(f4[l]), fbits(rv[l])); }
            }
        }

        float pf[16];
        std::memcpy(pf, &gPStream[pOff], 64);
        const uint32_t* pu = reinterpret_cast<const uint32_t*>(pf);
        if (!(almostEqual(pf[4], R.normal.x, 2e-4f, 2e-5f) && almostEqual(pf[5], R.normal.y, 2e-4f, 2e-5f) &&
              almostEqual(pf[6], R.normal.z, 2e-4f, 2e-5f)))
        { ++bad; if (bad < 12) std::printf("  [%s] pair %u patch normal gpu=(%g,%g,%g) ref=(%g,%g,%g)\n", label, i, pf[4], pf[5], pf[6], R.normal.x, R.normal.y, R.normal.z); }
        if (strictZero)
        {
            const float rn[3] = { R.normal.x, R.normal.y, R.normal.z };
            for (uint32_t l = 0; l < 3u; ++l)
                if (rn[l] == 0.f && pu[4 + l] != fbits(rn[l]))
                { ++bad; if (bad < 12) std::printf("  [%s] pair %u patch word %u gpu=%08x ref=%08x\n", label, i, 4 + l, pu[4 + l], fbits(rn[l])); }
        }

        float sta, dyn, rest, damping; uint32_t mflags;
        combineMaterials(materials[cfgs[i].mat0 & 0xffffu], materials[cfgs[i].mat1 & 0xffffu], sta, dyn, rest, damping, mflags);
        // P9.86c: the combine is exact IEEE arithmetic plus selects (0.5f * (a + b), a * b, flip * x,
        // PxMin/PxMax, the :161/:164 clamps), so words 7-10 (restitution, dynamicFriction,
        // staticFriction, damping) are compared BITWISE, the sign of zero included.
        {
            const float want[4] = { rest, dyn, sta, damping };
            if (comparedWords)
            {
                (*comparedWords)[i].compared = true;
                for (uint32_t w = 0; w < 4u; ++w) (*comparedWords)[i].w[w] = pu[7 + w];
            }
            for (uint32_t w = 0; w < 4u; ++w)
                if (pu[7 + w] != fbits(want[w]))
                { ++bad; if (bad < 12) std::printf("  [%s] pair %u patch word %u gpu=%08x ref=%08x\n", label, i, 7 + w, pu[7 + w], fbits(want[w])); }
        }
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

    // P9.88: bit 16 on mat0 of even pairs and on mat1 of every third.
    std::vector<PairCfg> hi = cfgs;
    for (size_t i = 0; i < hi.size(); ++i)
    {
        if (i % 2 == 0) hi[i].mat0 |= 0x10000u;
        if (i % 3 == 0) hi[i].mat1 |= 0x10000u;
    }
    checkSphere(hi, 0x00000000u, false, "subCases/matHi");
}

// P9.86c: the material combine on signed zeros.  PxsCombineMaterials (PxsMaterialCombiner.h)
// combines with PxMin/PxMax = selectMin/selectMax (b on a tie) and clamps
// fDynFriction = PxMax(dynFriction, 0.0f) (:161); the port spells all three as selects
// (npSphereNphase.slang combineScalars, fDyn).  The regular material banks never produce a +-0
// tie, so these touching sphere pairs (one contact, one patch each) combine the ZS_BANK
// materials: +-0 ties under eMIN/eMAX in both material orders, the damping forced to eMAX by
// exactly one compliant material, eMULTIPLY giving dynFriction = -0, and the box-vs-sphere flip
// path (cudaSphere.cu:108-111), which still combines the UNSWAPPED shape0/shape1 materials
// (cudaSphere.cu:215 -> contactPatchUtils.cuh:149-159).  checkSphere compares patch words 7-10
// bitwise against the CUDA mirror; each mutant twin of the mirror (gMatVariant) must change the
// expected words, so the case keeps observing both select fixes.
PXG_TEST(sphereNphase_materialZeroSign)
{
    struct MatPair { uint32_t a, b; bool flip; uint32_t img[4]; };   // img = CUDA words 7-10
    const MatPair mp[] = {
        // eMIN (-0,+0): PxMin keeps b = +0; FMin gives -0 (restitution, staticFriction, damping)
        { 1, 0, false, { 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u } },
        // eMIN (+0,-0): -0 either way; dynFriction -0 reaches the :161 clamp
        { 0, 1, false, { 0x80000000u, 0x00000000u, 0x80000000u, 0x80000000u } },
        // eMAX (+0,-0): PxMax keeps b = -0; FMax gives +0; dynFriction -0 reaches :161
        { 2, 3, false, { 0x80000000u, 0x00000000u, 0x80000000u, 0x80000000u } },
        // eMAX (-0,+0): +0 either way
        { 3, 2, false, { 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u } },
        // exactly one compliant: restitution forced eMIN(-0.5, 0.4), damping forced eMAX(+0,-0) = -0
        { 4, 5, false, { 0xBF000000u, 0x3E99999Au, 0x3F000000u, 0x80000000u } },
        // forced eMAX(-0,+0) = +0
        { 5, 4, false, { 0xBF000000u, 0x3E99999Au, 0x3F000000u, 0x00000000u } },
        // eMULTIPLY: dynFriction -0.5 * +0 = -0; staF -0.075 < fDyn, so sta = fDyn = +0
        { 6, 7, false, { 0x80000000u, 0x00000000u, 0x00000000u, 0x80000000u } },
        // eMULTIPLY: dynFriction +0 * -0.5 = -0
        { 7, 6, false, { 0x80000000u, 0x00000000u, 0x00000000u, 0x80000000u } },
        // flip path (box shape0, sphere shape1): eMIN (-0,+0) and eMAX (+0,-0), unswapped order
        { 1, 0, true,  { 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u } },
        { 2, 3, true,  { 0x80000000u, 0x00000000u, 0x80000000u, 0x80000000u } },
    };
    const uint32_t np = uint32_t(sizeof(mp) / sizeof(mp[0]));

    std::vector<PairCfg> cfgs;
    for (uint32_t p = 0; p < np; ++p)
    {
        if (mp[p].flip)
            cfgs.push_back({ GT_BOX, Vec3(1, 1, 1), xfT(0, 0, 0), GT_SPHERE, Vec3(0.5f, 0.5f, 0.5f), xfT(0, 0, 0.9f),
                             0.02f, ZS_BANK + mp[p].a, ZS_BANK + mp[p].b });
        else
            cfgs.push_back({ GT_SPHERE, Vec3(1, 1, 1), xfT(0, 0, 0), GT_SPHERE, Vec3(1, 1, 1), xfT(1.5f, 0, 0),
                             0.02f, ZS_BANK + mp[p].a, ZS_BANK + mp[p].b });
    }

    // Coverage: the mirror must reproduce the hand-derived CUDA image; the FMin twin must change
    // restitution, staticFriction and damping (pair 0 and its flip twin), the FMax twin the same
    // words (pair 2, its flip twin) plus the forced-eMAX damping (pair 4); dynFriction = -0 must
    // reach the clamp, including the eMULTIPLY pairs where staticFriction falls back to fDyn.
    uint32_t imageBad = 0, twinWord[3][4] = {}, twinPairs[3] = {}, flipTwin[3] = {};
    uint32_t forcedMaxTie = 0, dynNegZero = 0, staFromClamp = 0;
    std::vector<std::array<uint32_t, 4>> twinBits[3];   // [variant][pair] expected words 7-10 of each twin
    for (int v = 1; v <= 2; ++v) twinBits[v].resize(np);
    for (uint32_t p = 0; p < np; ++p)
    {
        const HMaterial m0 = zeroSignMaterial(mp[p].a), m1 = zeroSignMaterial(mp[p].b);
        uint32_t fl;
        float w0[4], d0;
        combineMaterials(m0, m1, w0[2], w0[1], w0[0], w0[3], fl, &d0);   // w = { rest, dyn, sta, damping }
        for (uint32_t w = 0; w < 4u; ++w)
            if (fbits(w0[w]) != mp[p].img[w])
            {
                ++imageBad;
                std::printf("  [materialZeroSign] pair %u word %u mirror=%08x cuda=%08x\n", p, 7 + w, fbits(w0[w]), mp[p].img[w]);
            }
        if (fbits(d0) == 0x80000000u)
        {
            ++dynNegZero;
            if (fbits(w0[2]) == fbits(w0[1])) ++staFromClamp;   // sta = fDyn (:164 fallback)
        }
        for (int v = 1; v <= 2; ++v)
        {
            float wv[4];
            gMatVariant = v;
            combineMaterials(m0, m1, wv[2], wv[1], wv[0], wv[3], fl);
            gMatVariant = 0;
            for (uint32_t w = 0; w < 4u; ++w) twinBits[v][p][w] = fbits(wv[w]);
            bool changed = false;
            for (uint32_t w = 0; w < 4u; ++w)
                if (fbits(wv[w]) != fbits(w0[w])) { ++twinWord[v][w]; changed = true; }
            if (changed) { ++twinPairs[v]; if (mp[p].flip) ++flipTwin[v]; }
            if (v == 2 && ((m0.rest < 0.f) != (m1.rest < 0.f)) && fbits(m0.damp) == 0u &&
                fbits(m1.damp) == 0x80000000u && fbits(wv[3]) != fbits(w0[3]))
                ++forcedMaxTie;
        }
    }
    std::printf("  [materialZeroSign] FMin twin: pairs=%u words(7..10)=%u,%u,%u,%u flip=%u; "
                "FMax twin: pairs=%u words=%u,%u,%u,%u flip=%u; forced-eMAX tie=%u; "
                "dynFriction -0 at clamp=%u (sta=fDyn %u)\n",
                twinPairs[1], twinWord[1][0], twinWord[1][1], twinWord[1][2], twinWord[1][3], flipTwin[1],
                twinPairs[2], twinWord[2][0], twinWord[2][1], twinWord[2][2], twinWord[2][3], flipTwin[2],
                forcedMaxTie, dynNegZero, staFromClamp);
    PXG_CHECK_MSG(imageBad == 0u, "the CUDA mirror differs from the hand-derived CUDA image in %u words", imageBad);
    PXG_CHECK_MSG(twinWord[1][0] && twinWord[1][2] && twinWord[1][3],
                  "FMin twin (npSphereNphase-r0) must change restitution, staticFriction and damping");
    PXG_CHECK_MSG(twinWord[2][0] && twinWord[2][2] && twinWord[2][3],
                  "FMax twin (npSphereNphase-r1) must change restitution, staticFriction and damping");
    PXG_CHECK_MSG(flipTwin[1] >= 1u && flipTwin[2] >= 1u, "the flip path must observe both twins (%u, %u)",
                  flipTwin[1], flipTwin[2]);
    PXG_CHECK_MSG(forcedMaxTie >= 1u, "forced-eMAX damping tie (+0,-0) not observed by the FMax twin (%u)", forcedMaxTie);
    PXG_CHECK_MSG(dynNegZero >= 5u && staFromClamp >= 2u,
                  "dynFriction -0 reached the :161 clamp on %u pairs (expected 5), sta = fDyn on %u (expected 2)",
                  dynNegZero, staFromClamp);

    // GPU-side coverage: every directed pair must have its patch words 7-10 bitwise compared, and a
    // twin counts as observed on a pair only where the GPU words equal the CUDA image AND differ from
    // that twin's words -- so a pair that stopped touching, or a loosened compare, fails here.
    std::vector<ComparedPatch> gw;
    checkSphere(cfgs, 0x00000000u, false, "materialZeroSign", false, &gw);
    uint32_t comparedPairs = 0, gpuObserved[3] = { 0, 0, 0 };
    for (uint32_t p = 0; p < np && p < gw.size(); ++p)
    {
        if (!gw[p].compared) continue;
        ++comparedPairs;
        bool matchesImage = true;
        for (uint32_t w = 0; w < 4u; ++w) if (gw[p].w[w] != mp[p].img[w]) matchesImage = false;
        if (!matchesImage) continue;
        for (int v = 1; v <= 2; ++v)
        {
            bool differs = false;
            for (uint32_t w = 0; w < 4u; ++w) if (gw[p].w[w] != twinBits[v][p][w]) differs = true;
            if (differs) ++gpuObserved[v];
        }
    }
    std::printf("  [materialZeroSign] GPU: compared pairs=%u/%u, FMin twin observed on %u, FMax twin on %u\n",
                comparedPairs, np, gpuObserved[1], gpuObserved[2]);
    PXG_CHECK_MSG(comparedPairs == np, "materialZeroSign: patch words 7-10 compared for only %u of %u pairs",
                  comparedPairs, np);
    PXG_CHECK_MSG(gpuObserved[1] >= 2u, "GPU output observes the FMin twin (r0) on %u pairs, need 2", gpuObserved[1]);
    PXG_CHECK_MSG(gpuObserved[2] >= 3u, "GPU output observes the FMax twin (r1) on %u pairs, need 3", gpuObserved[2]);
}

// P9.86c engine route of npCapsuleCollision_capsuleCapsuleZeroSign: capsule-capsule pairs whose
// degenerate capsule (halfHeight 2^-13, |d|^2 = 2^-24 < 1e-6, reciprocal = the select's +0) drives
// tTmp = -0 (capsule1) or comp = -0 (capsule0) into the PxClamp selects of
// distanceSegmentSegmentSquared (distanceSegmentSegment.cuh:76/:81; npSpherePrimitives.slang t2/s2).
// capsule0 is turned 180 deg about z, so getBasisVector0 = (-1, +0, -0), and sits at z = -0 with
// capsule1 at z = +0: the sign of s reaches closestA.z, vv.z and the patch normal word 6
// (cudaSphere.cu:215, npSphereNphase.slang patchBase[6]).  FClamp would turn each -0 into +0 and flip
// word 6 (pair 0: +0 -> -0, pairs 1 and 2: -0 -> +0).  Exact binary fractions throughout.
PXG_TEST(sphereNphase_capsuleCapsuleZeroSign)
{
    const float D = 0.50006103515625f;   // 0.5 + 2^-14
    const float A = 0.499969482421875f;  // 0.5 - 2^-15
    const float H = 1.220703125e-4f;     // 2^-13
    const float Z = 0.8660254f;          // q = (0,0,Z,0.5): 120 deg about z
    const QuatV qz180(0, 0, -1, 0), qId(0, 0, 0, 1), qz120(0, 0, Z, 0.5f);
    std::vector<PairCfg> cfgs;
    cfgs.push_back({ GT_CAPSULE, Vec3(1, 1, 0), TransformV(qz180, Vec3(D, 0.75f, -0.0f)),
                     GT_CAPSULE, Vec3(H, 1, 0), TransformV(qId, Vec3(-D, -0.75f, 0.0f)), 0.0f, 1, 2 });
    cfgs.push_back({ GT_CAPSULE, Vec3(H, 1, 0), TransformV(qz180, Vec3(1.0f, 0.0f, -0.0f)),
                     GT_CAPSULE, Vec3(1, 1, 0), TransformV(qId, Vec3(-1.0f, 0.0f, 0.0f)), 0.0f, 3, 4 });
    cfgs.push_back({ GT_CAPSULE, Vec3(1, 1, 0), TransformV(qz180, Vec3(A, 0.75f, -0.0f)),
                     GT_CAPSULE, Vec3(H, 1, 0), TransformV(qz120, Vec3(-A, -0.75f, 0.0f)), 0.0f, 5, 6 });
    const bool tTmpPair[3] = { true, false, true };
    const uint32_t cudaW6[3] = { 0x00000000u, 0x80000000u, 0x80000000u };   // hand-derived CUDA word 6
    for (uint32_t i = 0; i < 3u; ++i)
    {
        gSegTTmpNegZero = gSegCompNegZero = 0;
        const RC R = refDispatch(cfgs[i]);
        const size_t hit = tTmpPair[i] ? gSegTTmpNegZero : gSegCompNegZero;
        std::printf("  [capsZeroSign] pair %u: %s = -0 seen %zu, normal.z=%08x nb=%d\n", i,
                    tTmpPair[i] ? "tTmp" : "comp", hit, fbits(R.normal.z), R.nb);
        PXG_CHECK_MSG(hit != 0u, "pair %u no longer drives %s = -0 into its clamp", i, tTmpPair[i] ? "tTmp" : "comp");
        PXG_CHECK_MSG(R.nb == 1 && fbits(R.normal.z) == cudaW6[i], "pair %u: mirror normal.z %08x, CUDA %08x",
                      i, fbits(R.normal.z), cudaW6[i]);
    }
    checkSphere(cfgs, 0x00000000u, false, "capsZeroSign", true);
}

// P9.94b engine route of npCapsuleCollision_planeCapsuleZeroSign: plane-capsule's bToA =
// transform0.transformInv(transform1) (sphereCollision.cuh:527) is conj(q0).rotate(p1 - p0)
// (PxTransform.h:162-168), not q0.rotateInv(p1 - p0).  Plane q0 = (1,0,0,0) (180 deg about x, the
// normal stays +x) at p0 = (0,-0,0); capsule q1 = q0 at p1 = (0,0,-1), halfHeight 1, radius 0.5,
// cDist 0.0625 + 0.0625.  CUDA bToAp = (+0,-0,1), old (+0,+0,1); the y sign survives segE and
// q0.rotate into the stored contact point y: CUDA 0x00000000, old 0x80000000 (one contact; the
// patch normal is (-1,-0,-0)).  strictZero compares that lane and normal words 5-6 bitwise.
PXG_TEST(sphereNphase_planeCapsuleZeroSign)
{
    std::vector<PairCfg> cfgs;
    cfgs.push_back({ GT_PLANE, Vec3(1, 1, 1), TransformV(QuatV(1, 0, 0, 0), Vec3(0, -0.0f, 0)),
                     GT_CAPSULE, Vec3(1.0f /*hh*/, 0.5f /*r*/, 0), TransformV(QuatV(1, 0, 0, 0), Vec3(0, 0, -1)),
                     0.125f, 1, 2 });
    const RC R = refDispatch(cfgs[0]);
    gPlaneCapsuleRotInvTwin = true;
    const RC T = refDispatch(cfgs[0]);
    gPlaneCapsuleRotInvTwin = false;
    std::printf("  [planeCapsuleZeroSign] nb cuda=%d twin=%d, pt[0].y cuda=%08x twin=%08x, normal=(%08x,%08x,%08x)\n",
                R.nb, T.nb, fbits(R.pt[0].y), fbits(T.pt[0].y), fbits(R.normal.x), fbits(R.normal.y), fbits(R.normal.z));
    PXG_CHECK_MSG(R.nb == 1 && T.nb == 1 && fbits(R.pt[0].y) == 0x00000000u && fbits(T.pt[0].y) == 0x80000000u,
                  "planeCapsuleZeroSign: the rotateInv twin no longer flips pt[0].y (cuda %08x, twin %08x)",
                  fbits(R.pt[0].y), fbits(T.pt[0].y));
    checkSphere(cfgs, 0x00000000u, false, "planeCapsuleZeroSign", true);

    // The composition order of bToA.q = conj(q0) * q1 as well: allSubCases' only plane-capsule
    // pair has an identity plane, under which the product commutes.  Plane turned 90 deg about z
    // (normal +y), capsule turned 90 deg about x at p1 = (0, 0.5, 0): conj(q0) * q1 lays the
    // segment along the plane's -y, q1 * conj(q0) along its -z -- two contacts either way, at
    // different points (tolerance compare).
    const float h = 0.70710678f;
    std::vector<PairCfg> order;
    order.push_back({ GT_PLANE, Vec3(1, 1, 1), TransformV(QuatV(0, 0, h, h), Vec3(0, 0, 0)),
                      GT_CAPSULE, Vec3(1.0f /*hh*/, 0.5f /*r*/, 0), TransformV(QuatV(h, 0, 0, h), Vec3(0, 0.5f, 0)),
                      0.1f, 1, 2 });
    PXG_CHECK_MSG(refDispatch(order[0]).nb == 2, "planeCapsuleOrder: the pair no longer makes two contacts");
    checkSphere(order, 0x00000000u, false, "planeCapsuleOrder");
}

// P9.90: the dot order of the primitive routines, observed bitwise.  npSpherePrimitives spells
// every PxVec3 dot of sphereCollision.cuh as a Slang dot(), and llvmpipe sums an undecorated
// OpDot as x + (y + z) where the CUDA's PxVec3::dot is (x + y) + z; the NoContraction stamp on
// npSphereNphase.spv (tests/CMakeLists.txt, PXG_NOCONTRACT_SHADERS) makes it forward.  The
// fixtures above use axis-aligned or 45-degree configurations and a tolerance on the points, so
// they pass either way.  Here: general orientations and non-round positions, each pair drawn
// until the old-order twin changes its first penetration.  Sub-cases: sphere-sphere (distanceSq,
// sphereCollision.cuh:45), sphere-capsule (distancePointSegmentSquared nom/denom/v.v, :96-102),
// and capsule-capsule in the general single-contact branch (distanceSegmentSegmentSquared's five
// dots and sqDist0, :668-686, distanceSegmentSegment.cuh), both shape orders.  Every penetration
// is sqrt(dot) - radiusSum, correctly rounded IEEE on both sides, so it is compared BITWISE, and
// so are the contact points and normals.  Built unstamped, all 36 penetrations differ and each
// equals the old-order twin (measured, both device modes).
static uint32_t gP90Rng = 0x9e3779b9u;
static float p90Rand(float lo, float hi)
{
    gP90Rng = gP90Rng * 1664525u + 1013904223u;
    return lo + (hi - lo) * float(gP90Rng >> 8) * (1.0f / 16777216.0f);
}
static QuatV p90Quat()
{
    return QuatV(p90Rand(-1, 1), p90Rand(-1, 1), p90Rand(-1, 1), p90Rand(0.2f, 1)).getNormalized();
}
static Vec3 p90Vec(float h) { return Vec3(p90Rand(-h, h), p90Rand(-h, h), p90Rand(-h, h)); }

PXG_TEST(sphereNphase_p90DotOrder)
{
    std::vector<PairCfg> cfgs;
    uint32_t kinds[3] = { 0, 0, 0 };
    const uint32_t PER_KIND = 12u;
    for (uint32_t attempt = 0; attempt < 20000u && (kinds[0] < PER_KIND || kinds[1] < PER_KIND || kinds[2] < PER_KIND); ++attempt)
    {
        const uint32_t kind = attempt % 3u;
        if (kinds[kind] >= PER_KIND) continue;
        PairCfg c;
        const Vec3 base = p90Vec(3.0f);
        const float cd = p90Rand(0.01f, 0.07f);
        const uint32_t m0 = attempt & 7u, m1 = (attempt >> 3) & 7u;
        if (kind == 0u)
        {
            const float r0 = p90Rand(0.3f, 0.9f), r1 = p90Rand(0.3f, 0.9f);
            const Vec3 d = p90Vec(1.0f).getNormalized() * ((r0 + r1) * p90Rand(0.6f, 0.99f));
            c = { GT_SPHERE, Vec3(r0, r0, r0), TransformV(p90Quat(), base + d),
                  GT_SPHERE, Vec3(r1, r1, r1), TransformV(p90Quat(), base), cd, m0, m1 };
        }
        else if (kind == 1u)
        {
            const float r = p90Rand(0.3f, 0.8f), hh = p90Rand(0.4f, 1.3f), cr = p90Rand(0.2f, 0.5f);
            const QuatV q = p90Quat();
            const Vec3 axis = q.getBasisVector0();
            const Vec3 off = p90Vec(1.0f);
            const Vec3 perp = (off - axis * off.dot(axis)).getNormalized();
            const Vec3 sc = base + axis * (hh * p90Rand(-0.8f, 0.8f)) + perp * ((r + cr) * p90Rand(0.6f, 0.99f));
            const bool flipped = (attempt / 3u) & 1u;
            const PairCfg sph = { GT_SPHERE, Vec3(r, r, r), TransformV(p90Quat(), sc),
                                  GT_CAPSULE, Vec3(hh, cr, 0), TransformV(q, base), cd, m0, m1 };
            c = flipped ? PairCfg{ sph.type1, sph.scale1, sph.xf1, sph.type0, sph.scale0, sph.xf0, cd, m0, m1 } : sph;
        }
        else
        {
            const float r0 = p90Rand(0.2f, 0.5f), r1 = p90Rand(0.2f, 0.5f);
            const float h0 = p90Rand(0.4f, 1.2f), h1 = p90Rand(0.4f, 1.2f);
            const QuatV q0 = p90Quat(), q1 = p90Quat();
            const Vec3 n = q0.getBasisVector0().cross(q1.getBasisVector0());
            if (n.magnitude() < 0.3f) continue;   // stay clear of the parallel branch
            const Vec3 p1 = base + q1.getBasisVector0() * (h1 * p90Rand(-0.7f, 0.7f));
            const Vec3 p0 = p1 - q0.getBasisVector0() * (h0 * p90Rand(-0.7f, 0.7f)) +
                            n.getNormalized() * ((r0 + r1) * p90Rand(0.6f, 0.99f));
            c = { GT_CAPSULE, Vec3(h0, r0, 0), TransformV(q0, p0), GT_CAPSULE, Vec3(h1, r1, 0), TransformV(q1, p1), cd, m0, m1 };
        }
        gDotOld = false; const RC R = refDispatch(c);
        gDotOld = true;  const RC T = refDispatch(c);
        gDotOld = false;
        if (R.nb != 1 || T.nb != 1 || fbits(R.pen[0]) == fbits(T.pen[0])) continue;
        cfgs.push_back(c);
        ++kinds[kind];
    }
    PXG_CHECK_MSG(kinds[0] == PER_KIND && kinds[1] == PER_KIND && kinds[2] == PER_KIND,
                  "p90DotOrder: drew %u/%u/%u order-sensitive pairs, need %u each", kinds[0], kinds[1], kinds[2], PER_KIND);

    std::vector<GpuContactWords> gw;
    checkSphere(cfgs, 0x00000000u, false, "p90DotOrder", false, nullptr, &gw);
    if (gw.size() != cfgs.size()) return;   // no BDA

    uint32_t penBad = 0, penOld = 0, laneBad = 0, twinSame = 0;
    for (uint32_t i = 0; i < uint32_t(cfgs.size()); ++i)
    {
        gDotOld = false; const RC R = refDispatch(cfgs[i]);
        gDotOld = true;  const RC T = refDispatch(cfgs[i]);
        gDotOld = false;
        if (fbits(T.pen[0]) == fbits(R.pen[0])) ++twinSame;
        const GpuContactWords& g = gw[i];
        if (g.nb != 1u) { ++penBad; continue; }
        if (g.c[0][3] != fbits(R.pen[0]))
        {
            ++penBad;
            if (g.c[0][3] == fbits(T.pen[0])) ++penOld;
            if (penBad < 12) std::printf("  [p90DotOrder] pair %u pen gpu=%08x cuda=%08x oldOrder=%08x\n",
                                         i, g.c[0][3], fbits(R.pen[0]), fbits(T.pen[0]));
        }
        const float rv[6] = { R.pt[0].x, R.pt[0].y, R.pt[0].z, R.normal.x, R.normal.y, R.normal.z };
        const uint32_t gv[6] = { g.c[0][0], g.c[0][1], g.c[0][2], g.n[0], g.n[1], g.n[2] };
        for (uint32_t l = 0; l < 6u; ++l) if (gv[l] != fbits(rv[l])) ++laneBad;
    }
    std::printf("  [p90DotOrder] %zu pairs: pen mismatches %u (%u carry the old-order value), point/normal lanes off %u\n",
                cfgs.size(), penBad, penOld, laneBad);
    PXG_CHECK_MSG(twinSame == 0u, "the old-order twin must change every penetration (%u unchanged)", twinSame);
    PXG_CHECK_MSG(penBad == 0u, "p90DotOrder: %u penetrations differ bitwise from the CUDA order", penBad);
    // The points and normals inherit the same dots (normal = delta / dist, v / |v|, the segment
    // parameters); they are bitwise equal to the mirror too, so they are held to it.
    PXG_CHECK_MSG(laneBad == 0u, "p90DotOrder: %u point/normal lanes differ bitwise from the CUDA order", laneBad);
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
