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

// P7.0 FEMCloth bring-up test: the three cloth orchestration kernels
// (femClothSimBda.spv) on the engine ABI (BDA), driven by the REAL engine
// descriptor.  A genuine PxgFEMCloth (gpusimulationcontroller header) is filled
// with device buffer addresses + scalars, DMA'd to the device, and read back
// through the Slang pxgSlangFEMClothDesc mirror — the first-ever PxgFEMCloth BDA
// round-trip.  static_asserts / offsetof checks pin the mirror's byte offsets
// against the real engine struct (the highest-risk item, sizeof == 400), and each
// kernel's output is compared BIT-EXACT against a CPU transcription of the
// FEMCloth.cu math.
//
// Suites (substring-matched by the "femClothSim" ctest filter):
//   femClothSimDescLayout        — byte-offset proof of the descriptor mirror
//   femClothSimPreIntegratePGS   — cloth_preIntegrateLaunch PGS predict+integrate
//   femClothSimPreIntegrateTGS   — cloth_preIntegrateLaunch TGS branch (vel + zero posDelta)
//   femClothSimFinalize          — cloth_finalizeVelocitiesLaunch settling + wake flag
//   femClothSimSleeping          — cloth_sleeping wake-counter bookkeeping

#include "pxgTestFramework.h"

// no-CUDA-toolkit stand-in for the __device__/__align__ qualifiers the engine
// descriptor header decorates its inline members with; must precede PxgFEMCloth.h.
#include "host_defines.h"
#include "PxgFEMCloth.h"                     // real engine per-cloth descriptor
#include "PxsDeformableSurfaceMaterialCore.h" // real engine surface material data
#include "PxNodeIndex.h"                     // PxNodeIndex (rigid/world anchor node index)
#include "PxgSimulationCoreDesc.h"           // PxgFEMRigidAttachment (P7.1 tranche 3a) / PxgRigidFilterPair (4a)
#include "PxgSolverBody.h"                   // PxgSolverBodyData / PxgSolverTxIData / PxAlignedTransform
#include "PxgContactManager.h"               // PxgContactManagerInput (P7.1 tranche 4a contact gen)
#include "PxgConvexConvexShape.h"            // PxgShape (P7.1 tranche 4a contact gen)
#include "PxsTransformCache.h"               // PxsCachedTransform (P7.1 tranche 4a contact gen)
#include "geometry/PxGeometry.h"             // PxGeometryType (P7.1 tranche 4a contact gen)
#include "foundation/PxBounds3.h"            // PxBounds3 (P7.1 tranche 4a midphase AABB)

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

namespace
{
using namespace pxgtest;
using namespace physx;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct F4 { float x, y, z, w; };

// A device buffer + its buffer-device-address (BDA).
struct DevBuf
{
    PxgBufferHandle h = 0;
    uint64_t addr = 0;
};

template <typename T>
DevBuf makeDev(PxgSlangComputeDevice* dev, const std::vector<T>& data)
{
    DevBuf b;
    b.h = createBufferFrom(dev, data);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

static const float PX_MAX_REAL_F = 3.40282347e+38f;

// The three push-constant blobs, byte-identical to femClothSimBda.slang's
// [[vk::push_constant]] structs (which the shim reproduces from the CUDA arg pack).
struct PrePush
{
    uint64_t femCloths, activeIds;
    float gx, gy, gz, dt;
    uint32_t isTGSWord, pad36;
    uint64_t speculativeCCD;
    uint32_t extFlagsWord, pad52;
    uint64_t updateContactPairs, vtCounts, eeCounts;
};
static_assert(sizeof(PrePush) == 80, "PreIntegrate push-constant blob must be 80 bytes");

struct FinPush
{
    uint64_t femCloths, activeId;
    float invDt, dt;
    uint32_t alwaysAvgWord, pad28;
};
static_assert(sizeof(FinPush) == 32, "Finalize push-constant blob must be 32 bytes");

struct SleepPush
{
    uint64_t femCloths;
    uint32_t numActive, pad12;
    uint64_t activeId;
    float dt, resetCounter;
    uint64_t wakeCounters, stateChangedMask;
};
static_assert(sizeof(SleepPush) == 48, "Sleeping push-constant blob must be 48 bytes");

// Zeroed real PxgFEMCloth (no vtable — memset-safe, like PxgParticleSystem).
static PxgFEMCloth makeZeroDesc()
{
    PxgFEMCloth c;
    std::memset(&c, 0, sizeof(c));
    return c;
}

// P7.1: the in-plane membrane energy-solve push-constant blob, byte-identical to
// femClothSolveBda.slang's SolveNonSharedTriParams (ABI row {8,8,8,4,4,1}).
struct SolvePush
{
    uint64_t femCloths, activeFEMCloths, materials;
    float dt;
    uint32_t partitionId, isTGSWord, pad36;
};
static_assert(sizeof(SolvePush) == 40, "SolveNonSharedTri push-constant blob must be 40 bytes");

// P7.1 tranche 2: the triangle-pair energy solve push-constant blob, byte-identical
// to femClothSolvePairBda.slang's SolveTriPairParams (ABI row {8,8,8,4,4,1,1}).
struct SolveTriPairPush
{
    uint64_t femCloths, activeFEMCloths, materials;
    float dt;
    uint32_t partitionId;
    uint32_t flagsWord;   // byte0 = isSharedTrianglePair, byte1 = isTGS
    uint32_t pad36;
};
static_assert(sizeof(SolveTriPairPush) == 40, "SolveTriPair push-constant blob must be 40 bytes");

// P7.1 tranche 2: the Jacobi average push-constant blob, byte-identical to
// femClothSolvePairBda.slang's AvgPairVertsParams (ABI row {8,8,4,1}).
struct AvgPairVertsPush
{
    uint64_t femCloths, activeFEMCloths;
    float dtInv;
    uint32_t isSharedWord;   // byte0 = isSharedPartition
};
static_assert(sizeof(AvgPairVertsPush) == 24, "AvgPairVerts push-constant blob must be 24 bytes");

// ---------------------------------------------------------------------------
// P7.1: bit-exact CPU reference for the in-plane membrane energy solve.  The
// membrane leaf (refMembrane) + its 2x2 constraints (extractRotation2D /
// arapConstraint2D / areaConstraint2D) are lifted VERBATIM from the Phase-1-
// verified testNpClothSharedEnergy.cpp — which already pins them bit-exact vs the
// SAME Slang membraneSolve this tranche imports (pxgSlangClothMembrane, a byte-
// for-byte copy of npClothSharedEnergy's).  Kept in a private namespace so it
// cannot collide with the file-scope helpers / F4.
// ---------------------------------------------------------------------------
namespace p71ref
{
struct F4 { float x, y, z, w; };
struct V2 { float x, y; };
struct V3 { float x, y, z; };
const float FEMCLOTH_THRESHOLD = 1.0e-14f;
const float FEMCLOTH_HALF_PI = 1.57079632679489661923f;
const float FEMCLOTH_2PI = 6.28318530717958647692f;
static float pabs(float a) { return a < 0.f ? -a : a; }
static float selMin(float a, float b) { return a < b ? a : b; }
static float selMax(float a, float b) { return a > b ? a : b; }

static V2 add2(V2 a, V2 b) { return V2{ a.x + b.x, a.y + b.y }; }
static V2 sub2(V2 a, V2 b) { return V2{ a.x - b.x, a.y - b.y }; }
static V2 muls2(V2 a, float s) { return V2{ a.x * s, a.y * s }; }
static V2 neg2(V2 a) { return V2{ -a.x, -a.y }; }
static float dot2(V2 a, V2 b) { return a.x * b.x + a.y * b.y; }
static float magSq2(V2 v) { return v.x * v.x + v.y * v.y; }
static V3 add3(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 sub3(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 muls3(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b)
{
    return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
static float magSq3(V3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

static V3 getNormalized3(V3 v)
{
    const float m = magSq3(v);
    return m > 0.f ? muls3(v, 1.0f / std::sqrt(m)) : V3{ 0.f, 0.f, 0.f };
}

static void extractRotation2D(V2& r0, V2& r1, V2 f0, V2 f1)
{
    const float s2x = dot2(f0, f0), s2y = dot2(f1, f1), s2z = dot2(f0, f1);
    const float det = s2x * s2y - s2z * s2z;
    if (det < FEMCLOTH_THRESHOLD) { r0 = V2{ 1.f, 0.f }; r1 = V2{ 0.f, 1.f }; return; }

    const float s0 = std::sqrt(det);
    const float t = std::sqrt(s2x + s2y + 2.0f * s0);
    if (t < FEMCLOTH_THRESHOLD) { r0 = V2{ 1.f, 0.f }; r1 = V2{ 0.f, 1.f }; return; }
    const float tInv = 1.0f / t;

    const float Sx = (s2x + s0) * tInv, Sy = (s2y + s0) * tInv, Sz = s2z * tInv;

    const float sDet = Sx * Sy - Sz * Sz;
    if (sDet < FEMCLOTH_THRESHOLD) { r0 = V2{ 1.f, 0.f }; r1 = V2{ 0.f, 1.f }; return; }
    const float sDetInv = 1.0f / sDet;

    const float ix = Sy * sDetInv, iy = Sx * sDetInv, iz = (-Sz) * sDetInv;

    r0 = add2(muls2(f0, ix), muls2(f1, iz));
    r1 = add2(muls2(f0, iz), muls2(f1, iy));
}

static float queryDeltaLambda3(float C, V2 g0, V2 g1, V2 g2, float alphaTilde, float lambda,
                               float m0, float m1, float m2)
{
    const float denom = (m0 * magSq2(g0) + m1 * magSq2(g1) + m2 * magSq2(g2)) + alphaTilde;
    if (denom < FEMCLOTH_THRESHOLD)
        return 0.0f;
    return (-C - alphaTilde * lambda) / denom;
}

static void arapConstraint2D(float& lambda, V2& dx0, V2& dx1, V2& dx2, float alphaTilde,
                             const F4& QInv, V2 x01, V2 x02, float m0, float m1, float m2)
{
    const V2 f0 = add2(muls2(x01, QInv.x), muls2(x02, QInv.y));
    const V2 f1 = add2(muls2(x01, QInv.z), muls2(x02, QInv.w));

    V2 r0, r1;
    extractRotation2D(r0, r1, f0, f1);

    const V2 FMinusR0 = sub2(f0, r0);
    const V2 FMinusR1 = sub2(f1, r1);

    const float C = std::sqrt(dot2(FMinusR0, FMinusR0) + dot2(FMinusR1, FMinusR1));

    if (C > FEMCLOTH_THRESHOLD)
    {
        const float CInv = 1.0f / C;

        const V2 pC_pF0 = muls2(FMinusR0, CInv);
        const V2 pC_pF1 = muls2(FMinusR1, CInv);

        const V2 grad1 = add2(muls2(pC_pF0, QInv.x), muls2(pC_pF1, QInv.z));
        const V2 grad2 = add2(muls2(pC_pF0, QInv.y), muls2(pC_pF1, QInv.w));
        const V2 grad0 = sub2(neg2(grad1), grad2);

        const float deltaLambda = queryDeltaLambda3(C, grad0, grad1, grad2, alphaTilde, lambda, m0, m1, m2);
        lambda = lambda + deltaLambda;

        dx0 = add2(dx0, muls2(grad0, m0 * deltaLambda));
        dx1 = add2(dx1, muls2(grad1, m1 * deltaLambda));
        dx2 = add2(dx2, muls2(grad2, m2 * deltaLambda));
    }
}

static void areaConstraint2D(float& lambda, V2& dx0, V2& dx1, V2& dx2, float alphaTilde,
                             V2 x01, V2 x02, float m0, float m1, float m2, float area)
{
    const float x01CrossX02 = x01.x * x02.y - x01.y * x02.x;
    const float undeformedAreaInv = 1.0f / area;

    const float C = 0.5f * x01CrossX02 * undeformedAreaInv - 1.0f;

    const V2 grad1 = V2{ 0.5f * undeformedAreaInv * x02.y, -(0.5f * undeformedAreaInv * x02.x) };
    const V2 grad2 = V2{ -(0.5f * undeformedAreaInv * x01.y), 0.5f * undeformedAreaInv * x01.x };
    const V2 grad0 = sub2(neg2(grad1), grad2);

    const float deltaLambda = queryDeltaLambda3(C, grad0, grad1, grad2, alphaTilde, lambda, m0, m1, m2);
    lambda = lambda + deltaLambda;

    dx0 = add2(dx0, muls2(grad0, m0 * deltaLambda));
    dx1 = add2(dx1, muls2(grad1, m1 * deltaLambda));
    dx2 = add2(dx2, muls2(grad2, m2 * deltaLambda));
}

// membraneEnergySolvePerTriangle (FEMClothUtil.cuh L436-527).
static void refMembrane(F4& xx0, F4& xx1, F4& xx2, float dt, float youngs, float poissons,
                        float thickness, const F4& QInv, float vs0, float vs1, float vs2,
                        float& slotX, float& slotY, bool isTGS)
{
    if (youngs < FEMCLOTH_THRESHOLD)
        return;

    V3 x0 = V3{ xx0.x, xx0.y, xx0.z };
    V3 x1 = V3{ xx1.x, xx1.y, xx1.z };
    V3 x2 = V3{ xx2.x, xx2.y, xx2.z };

    const V3 x01 = sub3(x1, x0);
    const V3 x02 = sub3(x2, x0);

    const V3 axis0 = getNormalized3(x01);
    const V3 normal = cross3(x01, x02);
    const V3 axis1 = getNormalized3(cross3(normal, axis0));

    const float dt2 = dt * dt;

    const float det = QInv.x * QInv.w - QInv.y * QInv.z;
    const float area = 1.0f / (2.0f * det);
    const float volume = area * thickness;

    V2 dx0 = V2{ 0.f, 0.f }, dx1 = V2{ 0.f, 0.f }, dx2 = V2{ 0.f, 0.f };
    float lambda0 = 0.f, lambda1 = 0.f;

    if (!isTGS)
    {
        lambda0 = slotX;
        lambda1 = slotY;
    }

    const float lamL = youngs * poissons / ((1.0f + poissons) * (1.0f - 2.0f * poissons));
    const float mu = youngs / (2.0f * (1.0f + poissons));

    V2 xp01 = V2{ dot3(axis0, x01), dot3(axis1, x01) };
    V2 xp02 = V2{ dot3(axis0, x02), dot3(axis1, x02) };

    const float alphaTilde0 = 1.0f / (2.0f * mu * volume * dt2);

    arapConstraint2D(lambda0, dx0, dx1, dx2, alphaTilde0, QInv, xp01, xp02,
                     vs0 * xx0.w, vs1 * xx1.w, vs2 * xx2.w);

    if (poissons > FEMCLOTH_THRESHOLD)
    {
        float alphaTilde1 = 0.0f;

        if (poissons < 0.5f - FEMCLOTH_THRESHOLD)
        {
            alphaTilde1 = 1.0f / (lamL * volume * dt2);
        }

        xp01 = add2(xp01, sub2(dx1, dx0));
        xp02 = add2(xp02, sub2(dx2, dx0));

        areaConstraint2D(lambda1, dx0, dx1, dx2, alphaTilde1, xp01, xp02,
                         vs0 * xx0.w, vs1 * xx1.w, vs2 * xx2.w, area);
    }

    x0 = add3(x0, add3(muls3(axis0, dx0.x), muls3(axis1, dx0.y)));
    x1 = add3(x1, add3(muls3(axis0, dx1.x), muls3(axis1, dx1.y)));
    x2 = add3(x2, add3(muls3(axis0, dx2.x), muls3(axis1, dx2.y)));

    if (!isTGS)
    {
        slotX = lambda0;
        slotY = lambda1;
    }

    xx0.x = x0.x; xx0.y = x0.y; xx0.z = x0.z;
    xx1.x = x1.x; xx1.y = x1.y; xx1.z = x1.z;
    xx2.x = x2.x; xx2.y = x2.y; xx2.z = x2.z;
}

// bendingEnergySolvePerTrianglePair (FEMClothUtil.cuh L543-655).  The angle is
// computed with the HOST std::atan2 (the documented seam — the engine kernel calls
// atan2 ON-DEVICE, which is not cross-backend bit-exact).  Everything else is a
// verbatim transcription (lifted from the Phase-1-verified testNpClothBending.cpp).
// vrc = the four vertex reference counts; lambdaSlot is the per-pair bending lambda.
static void refBendingPair(F4& x0, F4& x1, F4& x2, F4& x3, F4 vrc,
                           float restBendingAngle, float kInv, float dt,
                           float& lambdaSlot, bool isTGS)
{
    const V3 p0 = { x0.x, x0.y, x0.z }, p1 = { x1.x, x1.y, x1.z };
    const V3 p2 = { x2.x, x2.y, x2.z }, p3 = { x3.x, x3.y, x3.z };
    const V3 x02 = sub3(p2, p0), x03 = sub3(p3, p0), x13 = sub3(p3, p1), x12 = sub3(p2, p1), x23 = sub3(p3, p2);
    const float x23Len = std::sqrt(magSq3(x23));

    if (x23Len < FEMCLOTH_THRESHOLD)
        return;
    if (kInv <= 0.f)
        return;

    const float x23LenInv = 1.f / x23Len;
    const V3 x23n = muls3(x23, x23LenInv);
    const V3 scaledN0 = cross3(x02, x03);
    const V3 scaledN1 = cross3(x13, x12);
    const float n0LenInv = 1.f / std::sqrt(magSq3(scaledN0));
    const float n1LenInv = 1.f / std::sqrt(magSq3(scaledN1));
    const V3 n0 = muls3(scaledN0, n0LenInv);
    const V3 n1 = muls3(scaledN1, n1LenInv);

    const float cosAngle = dot3(n0, n1);
    const float sinAngle = dot3(cross3(n0, n1), x23n);
    const float angle = std::atan2(sinAngle, cosAngle);   // THE SEAM

    const float dtInv = 1.f / dt;
    const float alphaTilde = kInv * dtInv * dtInv;
    float C = angle - restBendingAngle;

    if (pabs(C + FEMCLOTH_2PI) < pabs(C))
        C += FEMCLOTH_2PI;
    else if (pabs(C - FEMCLOTH_2PI) < pabs(C))
        C -= FEMCLOTH_2PI;

    C = selMin(FEMCLOTH_HALF_PI, selMax(-FEMCLOTH_HALF_PI, C));

    const V3 temp0 = muls3(n0, n0LenInv);
    const V3 temp1 = muls3(n1, n1LenInv);
    const V3 dCdx0 = muls3(temp0, -x23Len);
    const V3 dCdx1 = muls3(temp1, -x23Len);
    const V3 dCdx2 = V3{ dot3(x03, x23n) * temp0.x + dot3(x13, x23n) * temp1.x,
                         dot3(x03, x23n) * temp0.y + dot3(x13, x23n) * temp1.y,
                         dot3(x03, x23n) * temp0.z + dot3(x13, x23n) * temp1.z };
    const V3 s3 = V3{ dot3(x02, x23n) * temp0.x + dot3(x12, x23n) * temp1.x,
                      dot3(x02, x23n) * temp0.y + dot3(x12, x23n) * temp1.y,
                      dot3(x02, x23n) * temp0.z + dot3(x12, x23n) * temp1.z };
    const V3 dCdx3 = V3{ -s3.x, -s3.y, -s3.z };

    float lambda = 0.f;
    if (!isTGS)
        lambda = lambdaSlot;

    const float m0 = vrc.x * x0.w;
    const float m1 = vrc.y * x1.w;
    const float m2 = vrc.z * x2.w;
    const float m3 = vrc.w * x3.w;
    const float denom = (m0 * magSq3(dCdx0) + m1 * magSq3(dCdx1) + m2 * magSq3(dCdx2) + m3 * magSq3(dCdx3)) + alphaTilde;
    float deltaLambda;
    if (denom < FEMCLOTH_THRESHOLD)
        deltaLambda = 0.f;
    else
        deltaLambda = (-C - alphaTilde * lambda) / denom;

    if (!isTGS)
        lambdaSlot = lambda + deltaLambda;

    const float scale0 = vrc.x * x0.w * deltaLambda;
    const float scale1 = vrc.y * x1.w * deltaLambda;
    const float scale2 = vrc.z * x2.w * deltaLambda;
    const float scale3 = vrc.w * x3.w * deltaLambda;

    x0.x += scale0 * dCdx0.x; x0.y += scale0 * dCdx0.y; x0.z += scale0 * dCdx0.z;
    x1.x += scale1 * dCdx1.x; x1.y += scale1 * dCdx1.y; x1.z += scale1 * dCdx1.z;
    x2.x += scale2 * dCdx2.x; x2.y += scale2 * dCdx2.y; x2.z += scale2 * dCdx2.z;
    x3.x += scale3 * dCdx3.x; x3.y += scale3 * dCdx3.y; x3.z += scale3 * dCdx3.z;
}

// clothSharedEnergySolvePerTrianglePair (FEMClothUtil.cuh:661): membrane on both
// triangles (x2,x3,x0)/(x2,x3,x1) threaded in place, then bending.  restEdge holds
// (restEdge0.x, restEdge0.y, restEdge1.x, restEdge1.y).
static void refSharedPair(F4& x0, F4& x1, F4& x2, F4& x3, F4 vrc,
                          F4 restEdge, float restSharedEdgeLength,
                          float youngs0, float poissons0, float thickness0,
                          float youngs1, float poissons1, float thickness1,
                          float& m0x, float& m0y, float& m1x, float& m1y,
                          float restBendingAngle, float kInv, float dt,
                          float& bendLambda, bool isTGS)
{
    const float det0 = restSharedEdgeLength * restEdge.y;
    const float det1 = restSharedEdgeLength * restEdge.w;

    if (pabs(det0) > FEMCLOTH_THRESHOLD)
    {
        const float inv0 = 1.f / det0;
        const F4 QInv0 = F4{ restEdge.y * inv0, 0.f * inv0, -restEdge.x * inv0, restSharedEdgeLength * inv0 };
        refMembrane(x2, x3, x0, dt, youngs0, poissons0, thickness0, QInv0, vrc.z, vrc.w, vrc.x, m0x, m0y, isTGS);
    }
    if (pabs(det1) > FEMCLOTH_THRESHOLD)
    {
        const float inv1 = 1.f / det1;
        const F4 QInv1 = F4{ restEdge.w * inv1, 0.f * inv1, -restEdge.z * inv1, restSharedEdgeLength * inv1 };
        refMembrane(x2, x3, x1, dt, youngs1, poissons1, thickness1, QInv1, vrc.z, vrc.w, vrc.y, m1x, m1y, isTGS);
    }

    refBendingPair(x0, x1, x2, x3, vrc, restBendingAngle, kInv, dt, bendLambda, isTGS);
}
} // namespace p71ref

// ---------------------------------------------------------------------------
// Descriptor mirror layout proof — pins pxgSlangFEMClothDesc against the real
// engine PxgFEMCloth (offsets the three kernels dereference).  sizeof == 400.
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimDescLayout)
{
    // sizeof / array stride: the cloth kernels index femCloths[id], so the STRIDE
    // must match (verified in SPIR-V as ArrayStride 400).
    PXG_CHECK_MSG(sizeof(PxgFEMCloth) == 400, "PxgFEMCloth sizeof == 400 (got %zu)", sizeof(PxgFEMCloth));

    // device-pointer members
    PXG_CHECK(offsetof(PxgFEMCloth, mVelocity_InvMass) == 8);
    PXG_CHECK(offsetof(PxgFEMCloth, mPosition_InvMass) == 16);
    PXG_CHECK(offsetof(PxgFEMCloth, mRestPosition) == 24);
    PXG_CHECK(offsetof(PxgFEMCloth, mPrevPositionInContactOffset) == 32);
    PXG_CHECK(offsetof(PxgFEMCloth, mNonSharedTriPair_hasActiveBending) == 160);
    PXG_CHECK(offsetof(PxgFEMCloth, mOrderedSharedRestEdge0_edge1) == 168); // +8 => 7 pad after the bool
    PXG_CHECK(offsetof(PxgFEMCloth, mAccumulatedDeltaPos) == 256);

    // scalar tail
    PXG_CHECK(offsetof(PxgFEMCloth, mLinearDamping) == 288);
    PXG_CHECK(offsetof(PxgFEMCloth, mMaxLinearVelocity) == 292);
    PXG_CHECK(offsetof(PxgFEMCloth, mSettlingThreshold) == 300);
    PXG_CHECK(offsetof(PxgFEMCloth, mSleepThreshold) == 304);
    PXG_CHECK(offsetof(PxgFEMCloth, mSettlingDamping) == 308);
    PXG_CHECK(offsetof(PxgFEMCloth, mNbVerts) == 316);
    PXG_CHECK(offsetof(PxgFEMCloth, mActorFlags) == 376);
    PXG_CHECK(offsetof(PxgFEMCloth, mBodyFlags) == 377);
    PXG_CHECK(offsetof(PxgFEMCloth, mSurfaceFlags) == 378);
    PXG_CHECK(offsetof(PxgFEMCloth, mIsActive) == 380);
    PXG_CHECK(offsetof(PxgFEMCloth, mRestDistance) == 384);

    // P7.1: the in-plane membrane solve (cloth_solveNonSharedTriangleEnergy*) reads
    // these fields through the pxgSlangFEMClothDesc mirror — pin their offsets.
    PXG_CHECK(offsetof(PxgFEMCloth, mMaterialIndices) == 56);                                // PxU16*
    PXG_CHECK(offsetof(PxgFEMCloth, mOrderedNonSharedTriangleVertexIndices_triIndex) == 80); // uint4*
    PXG_CHECK(offsetof(PxgFEMCloth, mOrderedNonSharedTriangleLambdas) == 96);                // float2*
    PXG_CHECK(offsetof(PxgFEMCloth, mOrderedNonSharedTriangleRestPoseInv) == 104);           // float4* (QInv)
    PXG_CHECK(offsetof(PxgFEMCloth, mNonSharedTriAccumulatedPartitionsCP) == 192);           // PxU32*
    PXG_CHECK(offsetof(PxgFEMCloth, mNbNonSharedTriPartitions) == 344);                      // PxU32

    // the byte-packed flag word (@376): low byte == mActorFlags (eDISABLE_GRAVITY test)
    PXG_CHECK(PxActorFlag::eDISABLE_GRAVITY == (1 << 1));

    // P7.1: PxsDeformableSurfaceMaterialGpu mirror (Slang) vs the real engine
    // PxsDeformableSurfaceMaterialData (32-byte array element read per triangle).
    PXG_CHECK_MSG(sizeof(PxsDeformableSurfaceMaterialData) == 32,
                  "PxsDeformableSurfaceMaterialData sizeof == 32 (got %zu)",
                  sizeof(PxsDeformableSurfaceMaterialData));
    PXG_CHECK(offsetof(PxsDeformableSurfaceMaterialData, youngs) == 0);
    PXG_CHECK(offsetof(PxsDeformableSurfaceMaterialData, poissons) == 4);
    PXG_CHECK(offsetof(PxsDeformableSurfaceMaterialData, dynamicFriction) == 8);
    PXG_CHECK(offsetof(PxsDeformableSurfaceMaterialData, thickness) == 12);
    PXG_CHECK(offsetof(PxsDeformableSurfaceMaterialData, bendingStiffness) == 16);
    PXG_CHECK(offsetof(PxsDeformableSurfaceMaterialData, elasticityDamping) == 20);
    PXG_CHECK(offsetof(PxsDeformableSurfaceMaterialData, bendingDamping) == 24);

    // P7.1 tranche 2: the triangle-pair (shared shell-energy + bending) solve
    // (cloth_solveTrianglePairEnergy* / cloth_averageTrianglePairVertsLaunch) reads
    // these pair-scheme fields through the pxgSlangFEMClothDesc mirror — pin their
    // offsets against the real engine PxgFEMCloth (all already present in the mirror
    // from tranche 1's byte-exact round-trip; this locks the ones now dereferenced).
    PXG_CHECK(offsetof(PxgFEMCloth, mOrderedSharedTriangleLambdas) == 88);                          // float2* (membrane lambdas)
    PXG_CHECK(offsetof(PxgFEMCloth, mOrderedSharedTrianglePairVertexIndices) == 112);               // uint4*
    PXG_CHECK(offsetof(PxgFEMCloth, mOrderedNonSharedTrianglePairVertexIndices) == 120);            // uint4*
    PXG_CHECK(offsetof(PxgFEMCloth, mOrderedSharedRestBendingAngle_flexuralStiffness_damping) == 128);    // float4*
    PXG_CHECK(offsetof(PxgFEMCloth, mOrderedNonSharedRestBendingAngle_flexuralStiffness_damping) == 136); // float4*
    PXG_CHECK(offsetof(PxgFEMCloth, mSharedBendingLambdas) == 144);                                 // float*
    PXG_CHECK(offsetof(PxgFEMCloth, mNonSharedBendingLambdas) == 152);                              // float*
    PXG_CHECK(offsetof(PxgFEMCloth, mNonSharedTriPair_hasActiveBending) == 160);                    // bool (low byte of the @160 word)
    PXG_CHECK(offsetof(PxgFEMCloth, mOrderedSharedRestEdge0_edge1) == 168);                         // float4*
    PXG_CHECK(offsetof(PxgFEMCloth, mOrderedSharedRestEdgeLength_material0_material1) == 176);      // float4*
    PXG_CHECK(offsetof(PxgFEMCloth, mPosition_InvMassCP) == 184);                                   // float4* (copies buffer)
    PXG_CHECK(offsetof(PxgFEMCloth, mSharedTriPairRemapOutputCP) == 200);                           // PxU32* (writeIndices)
    PXG_CHECK(offsetof(PxgFEMCloth, mNonSharedTriPairRemapOutputCP) == 208);                        // PxU32*
    PXG_CHECK(offsetof(PxgFEMCloth, mSharedTriPairAccumulatedCopiesCP) == 216);                     // PxU32*
    PXG_CHECK(offsetof(PxgFEMCloth, mNonSharedTriPairAccumulatedCopiesCP) == 224);                  // PxU32*
    PXG_CHECK(offsetof(PxgFEMCloth, mSharedTriPairAccumulatedPartitionsCP) == 232);                 // PxU32*
    PXG_CHECK(offsetof(PxgFEMCloth, mNonSharedTriPairAccumulatedPartitionsCP) == 240);              // PxU32*
    PXG_CHECK(offsetof(PxgFEMCloth, mNbSharedTrianglePairs) == 336);                                // PxU32
    PXG_CHECK(offsetof(PxgFEMCloth, mNbNonSharedTrianglePairs) == 340);                             // PxU32
    PXG_CHECK(offsetof(PxgFEMCloth, mNbSharedTriPairPartitions) == 348);                            // PxU32
    PXG_CHECK(offsetof(PxgFEMCloth, mNbNonSharedTriPairPartitions) == 352);                         // PxU32
}

// ---------------------------------------------------------------------------
// Shared preIntegrate driver: dispatch + bit-exact CPU compare for a given
// isTGS mode.  invMass encoded in .w; vertex 3 has w==0 (skip branch).
// ---------------------------------------------------------------------------
static void runPreIntegrate(bool isTGS)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 8;
    const float dt = 1.0f / 60.0f;
    const float3 g = { 0.0f, -9.81f, 0.0f };
    const float linearDamping = 0.1f;
    const float maxLinVel = 1.0e6f;    // finite but never reached -> no clamp
    const float restDistance = 0.05f;
    const uint32_t actorFlags = 0u;    // gravity enabled

    std::vector<F4> positions(N), vels(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        const float w = (i == 3) ? 0.0f : 1.0f;
        positions[i] = { float(i) * 1.0f, 5.0f + float(i) * 0.25f, -float(i) * 0.5f, w };
        vels[i] = { 0.1f * float(i), -0.2f * float(i), 0.05f * float(i), 0.0f };
    }
    std::vector<F4> posDelta(N, { 9.f, 9.f, 9.f, 9.f });
    std::vector<F4> prevContact(N, { -1.f, -1.f, -1.f, -1.f });
    std::vector<uint32_t> vtCounts = { 123u }, eeCounts = { 456u };
    std::vector<float> speculativeCCD = { 7.0f };
    std::vector<uint8_t> updatePairs = { 9u };

    DevBuf bPos = makeDev(dev, positions);
    DevBuf bVel = makeDev(dev, vels);
    DevBuf bDelta = makeDev(dev, posDelta);
    DevBuf bPrev = makeDev(dev, prevContact);
    DevBuf bVt = makeDev(dev, vtCounts);
    DevBuf bEe = makeDev(dev, eeCounts);
    DevBuf bCcd = makeDev(dev, speculativeCCD);
    DevBuf bUpd = makeDev(dev, updatePairs);

    PxgFEMCloth c = makeZeroDesc();
    c.mVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVel.addr));
    c.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    c.mAccumulatedDeltaPos = reinterpret_cast<float4*>(size_t(bDelta.addr));
    c.mPrevPositionInContactOffset = reinterpret_cast<float4*>(size_t(bPrev.addr));
    c.mLinearDamping = linearDamping;
    c.mMaxLinearVelocity = maxLinVel;
    c.mRestDistance = restDistance;
    c.mNbVerts = N;
    c.mActorFlags = PxU8(actorFlags);

    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    PrePush push = {};
    push.femCloths = bDesc.addr;
    push.activeIds = bActive.addr;
    push.gx = g.x; push.gy = g.y; push.gz = g.z; push.dt = dt;
    push.isTGSWord = isTGS ? 1u : 0u;
    push.speculativeCCD = bCcd.addr;
    push.extFlagsWord = 0u;                 // extForces=false, adaptivePairUpd=false
    push.updateContactPairs = bUpd.addr;
    push.vtCounts = bVt.addr;
    push.eeCounts = bEe.addr;

    Kernel k("femClothSimBda.spv", "cloth_preIntegrateLaunch", {}, sizeof(PrePush));
    k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    // CPU reference (bit-exact expression order, matching FEMCloth.cu:48).
    std::vector<F4> refPos = positions, refVel = vels, refDelta = posDelta, refPrev = prevContact;
    for (uint32_t i = 0; i < N; ++i)
    {
        const F4 tPos = positions[i];
        float vx = vels[i].x, vy = vels[i].y, vz = vels[i].z;
        if (tPos.w != 0.0f) // !externalForces && w!=0
        {
            vx += g.x * dt; vy += g.y * dt; vz += g.z * dt;   // gravity (actorFlags has no DISABLE_GRAVITY)
            const float damping = 1.0f - std::fmin(1.0f, linearDamping * dt);
            vx *= damping; vy *= damping; vz *= damping;
        }
        if (isTGS)
        {
            refVel[i] = { vx, vy, vz, tPos.w };
            refDelta[i] = { 0.f, 0.f, 0.f, tPos.w };
            // positions + prevContact unchanged
        }
        else
        {
            const float maxVel = (maxLinVel == PX_MAX_REAL_F) ? PX_MAX_REAL_F
                                 : (maxLinVel < 0.0f) ? (0.8f * restDistance / dt) : maxLinVel;
            const float velMagSq = vx * vx + vy * vy + vz * vz;
            if (maxVel != PX_MAX_REAL_F && maxVel * maxVel < velMagSq)
            {
                const float s = maxVel / std::sqrt(velMagSq);
                vx *= s; vy *= s; vz *= s;
            }
            const float dx = vx * dt, dy = vy * dt, dz = vz * dt;
            refVel[i] = { vx, vy, vz, tPos.w };
            refDelta[i] = { dx, dy, dz, tPos.w };
            refPos[i] = { tPos.x + dx, tPos.y + dy, tPos.z + dz, tPos.w };
            refPrev[i] = { tPos.x + dx, tPos.y + dy, tPos.z + dz, 1.0f };
        }
    }

    const auto gPos = downloadBuffer<F4>(dev, bPos.h, N);
    const auto gVel = downloadBuffer<F4>(dev, bVel.h, N);
    const auto gDelta = downloadBuffer<F4>(dev, bDelta.h, N);
    const auto gPrev = downloadBuffer<F4>(dev, bPrev.h, N);
    const auto gVt = downloadBuffer<uint32_t>(dev, bVt.h, 1);
    const auto gEe = downloadBuffer<uint32_t>(dev, bEe.h, 1);

    size_t mm = 0;
    auto cmp4 = [&](const F4& a, const F4& b, const char* what, uint32_t i) {
        if (f2u(a.x) != f2u(b.x) || f2u(a.y) != f2u(b.y) || f2u(a.z) != f2u(b.z) || f2u(a.w) != f2u(b.w))
        {
            if (mm < 8)
                std::printf("  %s MM %u: ref(%g %g %g %g) got(%g %g %g %g)\n", what, i,
                            a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w);
            ++mm;
        }
    };
    for (uint32_t i = 0; i < N; ++i)
    {
        cmp4(refPos[i], gPos[i], "pos", i);
        cmp4(refVel[i], gVel[i], "vel", i);
        cmp4(refDelta[i], gDelta[i], "delta", i);
        cmp4(refPrev[i], gPrev[i], "prevContact", i);
    }
    // globalThreadIndex==0, groupId.y==0, non-adaptive -> VT/EE counts reset to 0.
    if (gVt[0] != 0u) { ++mm; std::printf("  VTContactCounts MM: %u (expected 0)\n", gVt[0]); }
    if (gEe[0] != 0u) { ++mm; std::printf("  EEContactCounts MM: %u (expected 0)\n", gEe[0]); }

    PXG_CHECK_MSG(mm == 0, "cloth_preIntegrateLaunch (%s) bit-exact vs CPU (%zu mismatches)",
                  isTGS ? "TGS" : "PGS", mm);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bDelta.h);
    dev->destroyBuffer(bPrev.h); dev->destroyBuffer(bVt.h); dev->destroyBuffer(bEe.h);
    dev->destroyBuffer(bCcd.h); dev->destroyBuffer(bUpd.h); dev->destroyBuffer(bDesc.h);
    dev->destroyBuffer(bActive.h);
}

PXG_TEST(femClothSimPreIntegratePGS) { runPreIntegrate(/*isTGS*/ false); }
PXG_TEST(femClothSimPreIntegrateTGS) { runPreIntegrate(/*isTGS*/ true); }

// ---------------------------------------------------------------------------
// cloth_finalizeVelocitiesLaunch — velocity settling + wake flag
// ---------------------------------------------------------------------------
static void runFinalize(bool alwaysAvg)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 8;
    const float dt = 1.0f / 60.0f;
    const float settlingThreshold = 2.0f;
    const float sleepThreshold = 0.5f;
    const float settlingDamping = 4.0f;

    // Mixed deltas: some large (awake, scaling 1), some tiny (asleep, scaled),
    // vertex 5 has w==0 (skipped).  posDelta.w carries invMass (unused here).
    std::vector<F4> posDelta(N), positions(N), vels(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        const float w = (i == 5) ? 0.0f : 1.0f;
        const float dmag = (i % 3 == 0) ? 0.2f : (i % 3 == 1) ? 0.0005f : 0.02f;
        posDelta[i] = { dmag, dmag * 0.5f, -dmag, w };
        positions[i] = { float(i), float(i) + 1.0f, float(i) + 2.0f, w };
        vels[i] = { 0.3f * float(i) + 0.1f, -0.2f * float(i) - 0.05f, 0.15f, w };
    }

    DevBuf bDelta = makeDev(dev, posDelta);
    DevBuf bPos = makeDev(dev, positions);
    DevBuf bVel = makeDev(dev, vels);

    PxgFEMCloth c = makeZeroDesc();
    c.mVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVel.addr));
    c.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    c.mAccumulatedDeltaPos = reinterpret_cast<float4*>(size_t(bDelta.addr));
    c.mSettlingThreshold = settlingThreshold;
    c.mSleepThreshold = sleepThreshold;
    c.mSettlingDamping = settlingDamping;
    c.mNbVerts = N;
    c.mIsActive = 0u;

    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    FinPush push = {};
    push.femCloths = bDesc.addr;
    push.activeId = bActive.addr;
    push.invDt = 1.0f / dt;
    push.dt = dt;
    push.alwaysAvgWord = alwaysAvg ? 1u : 0u;

    Kernel k("femClothSimBda.spv", "cloth_finalizeVelocitiesLaunch", {}, sizeof(FinPush));
    k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    // CPU reference (FEMCloth.cu:1227).
    std::vector<F4> refVel = vels;
    bool anyAwake = false;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float settleTolerance = settlingThreshold * dt;
        const float tolerance = sleepThreshold * dt;
        const float sleepDamping = 1.0f - std::fmin(1.0f, settlingDamping * dt);

        const F4 delta = posDelta[i];
        const float tdx = delta.x, tdy = delta.y, tdz = delta.z;
        const float dvx = tdx / dt, dvy = tdy / dt, dvz = tdz / dt;
        const F4 pos = positions[i];

        if (pos.w != 0.0f)
        {
            float velocityScaling = 1.0f;
            const float magSq = tdx * tdx + tdy * tdy + tdz * tdz;
            bool awake;
            if (magSq < settleTolerance * settleTolerance)
            {
                awake = magSq >= tolerance * tolerance;
                velocityScaling = sleepDamping;
            }
            else
            {
                awake = true;
            }
            anyAwake = anyAwake || awake;

            if (alwaysAvg || velocityScaling != 1.0f)
            {
                const F4 vel = vels[i];
                float tvx = vel.x, tvy = vel.y, tvz = vel.z;
                const float deltaVelMagSqr = dvx * dvx + dvy * dvy + dvz * dvz;
                const float velMagSqr = tvx * tvx + tvy * tvy + tvz * tvz;
                if (alwaysAvg && deltaVelMagSqr < (tvx * tvx + tvy * tvy + tvz * tvz))
                {
                    const float s = std::sqrt(deltaVelMagSqr / velMagSqr);
                    tvx *= s; tvy *= s; tvz *= s;
                }
                refVel[i] = { velocityScaling * tvx, velocityScaling * tvy, velocityScaling * tvz, vel.w };
            }
        }
    }

    const auto gVel = downloadBuffer<F4>(dev, bVel.h, N);
    const auto gDesc = downloadBuffer<PxgFEMCloth>(dev, bDesc.h, 1);

    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const F4& a = refVel[i]; const F4& b = gVel[i];
        if (f2u(a.x) != f2u(b.x) || f2u(a.y) != f2u(b.y) || f2u(a.z) != f2u(b.z) || f2u(a.w) != f2u(b.w))
        {
            if (mm < 8) std::printf("  vel MM %u: ref(%g %g %g %g) got(%g %g %g %g)\n", i,
                                    a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w);
            ++mm;
        }
    }
    const uint32_t expectActive = anyAwake ? 1u : 0u;
    if (gDesc[0].mIsActive != expectActive)
    { ++mm; std::printf("  mIsActive MM: %u (expected %u)\n", gDesc[0].mIsActive, expectActive); }

    PXG_CHECK_MSG(mm == 0, "cloth_finalizeVelocitiesLaunch (alwaysAvg=%d) bit-exact vs CPU (%zu mismatches)",
                  int(alwaysAvg), mm);

    dev->destroyBuffer(bDelta.h); dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

PXG_TEST(femClothSimFinalize)
{
    runFinalize(/*alwaysAvg*/ false); // the PGS free-fall path (velocities untouched when awake)
    runFinalize(/*alwaysAvg*/ true);  // the TGS averaging path
}

// ---------------------------------------------------------------------------
// cloth_sleeping — wake-counter bookkeeping + state-change ballot
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimSleeping)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 1.0f / 60.0f;
    const float resetCounter = 0.4f;
    const uint32_t numActive = 3;

    // Three cloths exercising each transition:
    //  c0: active this frame (mIsActive=1), was asleep (wake=0)   -> reset->0.4, pred=true (bit0)
    //  c1: inactive, still awake (wake=0.1 -> 0.1-dt>0)           -> decay,     pred=false
    //  c2: inactive, falls asleep (wake=0.005 < dt -> 0)          -> decay->0,  pred=true (bit2)
    std::vector<uint32_t> isActive = { 1u, 0u, 0u };
    std::vector<float> wake = { 0.0f, 0.1f, 0.005f };

    std::vector<PxgFEMCloth> descArr(numActive);
    for (uint32_t i = 0; i < numActive; ++i)
    {
        descArr[i] = makeZeroDesc();
        descArr[i].mIsActive = isActive[i];
    }
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<uint32_t> active = { 0u, 1u, 2u };
    DevBuf bActive = makeDev(dev, active);
    DevBuf bWake = makeDev(dev, wake);
    // stateChangedMask: cloth_sleeping fills one word per warp, but only for the warps that
    // hold a live cloth — the rest of the map must keep this sentinel.
    static const uint32_t MASK_SENTINEL = 0xDEADBEEFu;
    std::vector<uint32_t> stateMask(32, MASK_SENTINEL);
    DevBuf bMask = makeDev(dev, stateMask);

    SleepPush push = {};
    push.femCloths = bDesc.addr;
    push.numActive = numActive;
    push.activeId = bActive.addr;
    push.dt = dt;
    push.resetCounter = resetCounter;
    push.wakeCounters = bWake.addr;
    push.stateChangedMask = bMask.addr;

    Kernel k("femClothSimBda.spv", "cloth_sleeping", {}, sizeof(SleepPush));
    k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    // CPU reference.
    std::vector<float> refWake = wake;
    uint32_t refMask0 = 0u;
    for (uint32_t i = 0; i < numActive; ++i)
    {
        const bool reset = isActive[i] != 0u;
        float counter = wake[i];
        const bool wasActive = counter > 0.0f;
        counter = reset ? resetCounter : std::fmax(0.0f, counter - dt);
        const bool nowActive = counter > 0.0f;
        refWake[i] = counter;
        if (nowActive != wasActive)
            refMask0 |= (1u << i); // laneId == i, warpId == 0
    }

    const auto gWake = downloadBuffer<float>(dev, bWake.h, numActive);
    const auto gMask = downloadBuffer<uint32_t>(dev, bMask.h, 32);

    size_t mm = 0;
    for (uint32_t i = 0; i < numActive; ++i)
        if (f2u(refWake[i]) != f2u(gWake[i]))
        { ++mm; std::printf("  wake MM %u: ref %g got %g\n", i, refWake[i], gWake[i]); }

    if (gMask[0] != refMask0)
    { ++mm; std::printf("  stateMask[0] MM: got 0x%x expected 0x%x\n", gMask[0], refMask0); }
    // CUDA gates the store on `(startIdx + threadIdx.x) < ((numActive + 31) / 32)`, so with
    // one live word only word 0 is written and the rest of the PxgBitMap must be LEFT ALONE
    // — writing them is an out-of-bounds store past the end of the map.  This assertion used
    // to expect zeros here, which encoded the missing guard; it was corrected together with
    // the shader while porting the softbody twin (sb_sleeping).
    const uint32_t nbWords = (numActive + 31u) / 32u;
    for (uint32_t w = nbWords; w < 32; ++w)
        if (gMask[w] != MASK_SENTINEL)
        { ++mm; std::printf("  stateMask[%u] clobbered: got 0x%x expected the untouched sentinel 0x%x "
                            "(the (numActive+31)/32 word guard is missing)\n", w, gMask[w], MASK_SENTINEL); }

    PXG_CHECK_MSG(mm == 0, "cloth_sleeping bit-exact vs CPU (%zu mismatches)", mm);

    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
    dev->destroyBuffer(bWake.h); dev->destroyBuffer(bMask.h);
}

// ---------------------------------------------------------------------------
// P7.1: cloth_solveNonSharedTriangleEnergyLaunch — the in-plane membrane energy
// solve.  One STRETCHED triangle (edge x01 pulled to 1.5x rest) in a single
// non-shared partition [0,1); dispatch, then bit-exact compare positions /
// velocities / accumulated-delta / lambdas vs a CPU transcription of
// updateNonSharedTriangle (FEMCloth.cu:610) driving the verified refMembrane.
// ---------------------------------------------------------------------------
static void runSolveNonShared(bool isTGS)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 3;                 // one triangle
    const float dt = 1.0f / 60.0f;
    const float invDt = 1.0f / dt;

    struct F2 { float x, y; };
    struct U4 { uint32_t x, y, z, w; };

    // Stretched current positions (rest edges are unit; x01 pulled to 1.5).
    std::vector<F4> positions = {
        { 0.0f, 0.0f, 0.0f, 1.0f },
        { 1.5f, 0.0f, 0.0f, 1.0f },
        { 0.0f, 1.0f, 0.0f, 1.0f },
    };
    std::vector<F4> vels = {
        { 0.10f, -0.20f, 0.05f, 0.0f },
        { -0.03f, 0.07f, -0.11f, 0.0f },
        { 0.02f, -0.09f, 0.13f, 0.0f },
    };
    std::vector<F4> accum = {
        { 0.30f, -0.10f, 0.20f, 0.0f },
        { -0.40f, 0.50f, -0.60f, 0.0f },
        { 0.70f, -0.80f, 0.90f, 0.0f },
    };
    // Rest-pose inverse: Q = [[1,0],[0,1]] -> QInv = (1,0,0,1); det=1, area=0.5.
    std::vector<F4> qinv = { { 1.0f, 0.0f, 0.0f, 1.0f } };
    std::vector<F2> lambdas = { { 0.0f, 0.0f } };
    std::vector<U4> triVtx = { { 0u, 1u, 2u, 0u } };   // verts (0,1,2), triIndex 0
    std::vector<uint32_t> matIdx = { 0u };              // mMaterialIndices[0] = 0
    std::vector<uint32_t> partCP = { 1u };              // partition 0 -> [0,1)

    PxsDeformableSurfaceMaterialData mat;
    std::memset(&mat, 0, sizeof(mat));
    mat.youngs = 1.0e5f;
    mat.poissons = 0.3f;
    mat.thickness = 0.01f;
    std::vector<PxsDeformableSurfaceMaterialData> mats = { mat };

    DevBuf bPos = makeDev(dev, positions);
    DevBuf bVel = makeDev(dev, vels);
    DevBuf bAcc = makeDev(dev, accum);
    DevBuf bQ = makeDev(dev, qinv);
    DevBuf bLam = makeDev(dev, lambdas);
    DevBuf bTri = makeDev(dev, triVtx);
    DevBuf bMatIdx = makeDev(dev, matIdx);
    DevBuf bPart = makeDev(dev, partCP);
    DevBuf bMat = makeDev(dev, mats);

    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bPos.addr));
    c.mVelocity_InvMass = reinterpret_cast<decltype(c.mVelocity_InvMass)>(size_t(bVel.addr));
    c.mAccumulatedDeltaPos = reinterpret_cast<decltype(c.mAccumulatedDeltaPos)>(size_t(bAcc.addr));
    c.mMaterialIndices = reinterpret_cast<decltype(c.mMaterialIndices)>(size_t(bMatIdx.addr));
    c.mOrderedNonSharedTriangleVertexIndices_triIndex =
        reinterpret_cast<decltype(c.mOrderedNonSharedTriangleVertexIndices_triIndex)>(size_t(bTri.addr));
    c.mOrderedNonSharedTriangleLambdas =
        reinterpret_cast<decltype(c.mOrderedNonSharedTriangleLambdas)>(size_t(bLam.addr));
    c.mOrderedNonSharedTriangleRestPoseInv =
        reinterpret_cast<decltype(c.mOrderedNonSharedTriangleRestPoseInv)>(size_t(bQ.addr));
    c.mNonSharedTriAccumulatedPartitionsCP =
        reinterpret_cast<decltype(c.mNonSharedTriAccumulatedPartitionsCP)>(size_t(bPart.addr));
    c.mNbNonSharedTriPartitions = 1u;
    c.mNbVerts = N;

    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    SolvePush push = {};
    push.femCloths = bDesc.addr;
    push.activeFEMCloths = bActive.addr;
    push.materials = bMat.addr;
    push.dt = dt;
    push.partitionId = 0u;
    push.isTGSWord = isTGS ? 1u : 0u;

    Kernel k("femClothSolveBda.spv", "cloth_solveNonSharedTriangleEnergyLaunch", {}, sizeof(SolvePush));
    k.dispatch(/*gridX numBlocks*/ 1u, {}, push, /*gridY nbActive*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    // CPU reference: updateNonSharedTriangle (FEMCloth.cu:610) wiring + refMembrane.
    p71ref::F4 rx0 = { positions[0].x, positions[0].y, positions[0].z, positions[0].w };
    p71ref::F4 rx1 = { positions[1].x, positions[1].y, positions[1].z, positions[1].w };
    p71ref::F4 rx2 = { positions[2].x, positions[2].y, positions[2].z, positions[2].w };
    const p71ref::F4 rQ = { qinv[0].x, qinv[0].y, qinv[0].z, qinv[0].w };
    float slotX = isTGS ? 0.0f : lambdas[0].x;
    float slotY = isTGS ? 0.0f : lambdas[0].y;

    p71ref::refMembrane(rx0, rx1, rx2, dt, mat.youngs, mat.poissons, mat.thickness, rQ,
                        1.0f, 1.0f, 1.0f, slotX, slotY, isTGS);

    std::vector<F4> refPos = positions, refVel = vels, refAcc = accum;
    const p71ref::F4 nx[3] = { rx0, rx1, rx2 };
    const F4 xcp[3] = { positions[0], positions[1], positions[2] };
    for (uint32_t v = 0; v < 3; ++v)
    {
        const F4 newPos = { nx[v].x, nx[v].y, nx[v].z, xcp[v].w }; // membrane preserves .w
        const F4 d = { newPos.x - xcp[v].x, newPos.y - xcp[v].y, newPos.z - xcp[v].z, newPos.w - xcp[v].w };
        refVel[v] = { vels[v].x + d.x * invDt, vels[v].y + d.y * invDt, vels[v].z + d.z * invDt, vels[v].w + d.w * invDt };
        refAcc[v] = { accum[v].x + d.x, accum[v].y + d.y, accum[v].z + d.z, accum[v].w + d.w };
        refPos[v] = newPos;
    }
    const F2 refLam = { slotX, slotY };

    const auto gPos = downloadBuffer<F4>(dev, bPos.h, N);
    const auto gVel = downloadBuffer<F4>(dev, bVel.h, N);
    const auto gAcc = downloadBuffer<F4>(dev, bAcc.h, N);
    const auto gLam = downloadBuffer<F2>(dev, bLam.h, 1);

    // The solve must actually move the stretched vertex (teeth: not a no-op).
    PXG_CHECK_MSG(f2u(gPos[1].x) != f2u(positions[1].x),
                  "energy solve should correct the stretched triangle (x1.x moved)");

    size_t mm = 0;
    auto cmp4 = [&](const char* nm, uint32_t i, const F4& r, const F4& g) {
        if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) || f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
        { ++mm; std::printf("  %s[%u] MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", nm, i,
                            r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w); }
    };
    for (uint32_t v = 0; v < N; ++v)
    {
        cmp4("pos", v, refPos[v], gPos[v]);
        cmp4("vel", v, refVel[v], gVel[v]);
        cmp4("acc", v, refAcc[v], gAcc[v]);
    }
    if (!isTGS && (f2u(refLam.x) != f2u(gLam[0].x) || f2u(refLam.y) != f2u(gLam[0].y)))
    { ++mm; std::printf("  lambda MM: ref(%g,%g) got(%g,%g)\n", refLam.x, refLam.y, gLam[0].x, gLam[0].y); }

    PXG_CHECK_MSG(mm == 0, "cloth_solveNonSharedTriangleEnergy bit-exact vs CPU (%zu mismatches, isTGS=%d)",
                  mm, int(isTGS));

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bAcc.h);
    dev->destroyBuffer(bQ.h); dev->destroyBuffer(bLam.h); dev->destroyBuffer(bTri.h);
    dev->destroyBuffer(bMatIdx.h); dev->destroyBuffer(bPart.h); dev->destroyBuffer(bMat.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

PXG_TEST(femClothSimSolveNonSharedPGS) { runSolveNonShared(/*isTGS*/ false); }
PXG_TEST(femClothSimSolveNonSharedTGS) { runSolveNonShared(/*isTGS*/ true); }

// ---------------------------------------------------------------------------
// P7.1 tranche 2: cloth_solveTrianglePairEnergyLaunch (+ cluster twin) +
// cloth_averageTrianglePairVertsLaunch — the triangle-PAIR shared-energy + bending
// solve.  A SINGLE triangle pair (4 verts, one dihedral) with the copies/remap
// arrays set up for one pair in one partition; dispatch the pair solve then the
// Jacobi average, and compare positions / velocities / accumulated-delta / lambdas
// against a CPU transcription of updateTrianglePairs (FEMCloth.cu:768) driving the
// verified refMembrane + refBendingPair.
//
// atan2 seam (see femClothSolvePairBda.slang header): the MEMBRANE half + the
// copies/remap/average machinery are BIT-EXACT; the on-device atan2 BENDING half is
// a TOLERANCE path.  The configs below cover both:
//   - bending OFF (kInv<=0): everything bit-exact  -> pins membrane + copies/average.
//   - non-shared FLAT pair (symmetric coplanar): atan2(0,+x)==+0 is exact, so the
//     whole bending downstream (dCdx / deltaLambda / scale) is bit-exact too.
//   - genuinely-BENT pair with bending on: positions within a tight relative
//     tolerance (the atan2-only divergence); membrane lambdas stay bit-exact.
// ---------------------------------------------------------------------------

// One triangle pair, 4 vertices (indices 0..3), pair vertexIndex = (0,1,2,3).
struct TriPairConfig
{
    bool isShared;      // shared partition (membrane+bending) vs non-shared (bending only)
    bool bendingOn;     // kInv > 0 (else the bending term early-outs)
    bool flat;          // symmetric coplanar pair (atan2==+0 exact -> bending bit-exact)
    bool isTGS;
    bool useCluster;    // dispatch the cluster twin instead of the per-partition launch
};

static void runSolveTriPair(const TriPairConfig& cfg)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 4;                 // one pair, 4 verts
    const float dt = 1.0f / 60.0f;
    const float invDt = 1.0f / dt;

    struct F2 { float x, y; };
    struct U4 { uint32_t x, y, z, w; };

    // --- geometry ---------------------------------------------------------
    std::vector<F4> positions(N);
    if (cfg.flat)
    {
        // symmetric coplanar pair: sinAngle==0, cosAngle==1 exactly -> atan2==+0.
        positions[0] = { -1.0f, 0.0f, 0.0f, 1.0f };
        positions[1] = { 1.0f, 0.0f, 0.0f, 1.0f };
        positions[2] = { 0.0f, -1.0f, 0.0f, 1.0f };
        positions[3] = { 0.0f, 1.0f, 0.0f, 1.0f };
    }
    else
    {
        // a stretched, out-of-plane (bent) pair; verts 2,3 are the shared edge.
        positions[0] = { -0.5f, 0.5f, 0.30f, 1.0f };
        positions[1] = { 0.5f, 0.5f, -0.20f, 1.0f };
        positions[2] = { 0.0f, 0.0f, 0.0f, 1.0f };
        positions[3] = { 0.0f, 1.2f, 0.0f, 1.0f };
    }
    std::vector<F4> vels(N), accum(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        vels[i] = { 0.10f * float(i) + 0.03f, -0.20f * float(i) - 0.01f, 0.05f * float(i), 1.0f };
        accum[i] = { 0.30f - 0.10f * float(i), -0.20f + 0.05f * float(i), 0.15f * float(i), 1.0f };
    }

    // --- material + rest data --------------------------------------------
    PxsDeformableSurfaceMaterialData mat;
    std::memset(&mat, 0, sizeof(mat));
    mat.youngs = 1.0e5f;
    mat.poissons = 0.3f;
    mat.thickness = 0.01f;
    std::vector<PxsDeformableSurfaceMaterialData> mats = { mat };

    const float restSharedEdgeLength = 1.0f;
    // restEdge0 = (0,1), restEdge1 = (0,1) -> det0=det1=1 (both membranes run).
    const F4 restEdge = { 0.0f, 1.0f, 0.0f, 1.0f };
    // restData1 = (restLen, material0, material1, _) -> both index material 0.
    const F4 restEdgeLen_mat = { restSharedEdgeLength, 0.0f, 0.0f, 0.0f };

    const float restBendingAngle = cfg.flat ? 0.5f : 0.4f;
    const float kInv = cfg.bendingOn ? 2.0e-3f : -1.0f;   // <=0 disables bending
    const F4 restBend = { restBendingAngle, kInv, 0.0f, 0.0f };

    // per-pair lambda seeds (PGS reads/updates them; TGS ignores them).
    std::vector<F2> membraneLambdas = { { 0.011f, -0.022f }, { -0.033f, 0.044f } }; // 2 float2
    std::vector<float> bendingLambda = { 0.055f };

    // --- copies/remap scheme for a single pair ---------------------------
    // CP buffer: [0..3] current per-pair positions (w=0 -> read from positions[]),
    // [4..7] accumulated poses (written by solve, read+zeroed by average).
    std::vector<F4> cp(8, F4{ 0.0f, 0.0f, 0.0f, 0.0f });
    std::vector<U4> pairVtx = { { 0u, 1u, 2u, 3u } };
    std::vector<uint32_t> writeIndices = { 4u, 5u, 6u, 7u };   // pair-slot k -> CP[4+k]
    std::vector<uint32_t> accumCopies = { 1u, 2u, 3u, 4u };    // per-vertex prefix sum (1 copy each)
    std::vector<uint32_t> accumPartitions = { 1u };            // partition 0 -> [0,1)

    // --- upload ----------------------------------------------------------
    DevBuf bPos = makeDev(dev, positions);
    DevBuf bVel = makeDev(dev, vels);
    DevBuf bAcc = makeDev(dev, accum);
    DevBuf bCP = makeDev(dev, cp);
    DevBuf bPairVtx = makeDev(dev, pairVtx);
    DevBuf bWrite = makeDev(dev, writeIndices);
    DevBuf bCopies = makeDev(dev, accumCopies);
    DevBuf bParts = makeDev(dev, accumPartitions);
    DevBuf bRestEdge = makeDev(dev, std::vector<F4>{ restEdge });
    DevBuf bRestEdgeLenMat = makeDev(dev, std::vector<F4>{ restEdgeLen_mat });
    DevBuf bRestBend = makeDev(dev, std::vector<F4>{ restBend });
    DevBuf bMemLam = makeDev(dev, membraneLambdas);
    DevBuf bBendLam = makeDev(dev, bendingLambda);
    DevBuf bMat = makeDev(dev, mats);

    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bPos.addr));
    c.mVelocity_InvMass = reinterpret_cast<decltype(c.mVelocity_InvMass)>(size_t(bVel.addr));
    c.mAccumulatedDeltaPos = reinterpret_cast<decltype(c.mAccumulatedDeltaPos)>(size_t(bAcc.addr));
    c.mPosition_InvMassCP = reinterpret_cast<decltype(c.mPosition_InvMassCP)>(size_t(bCP.addr));
    c.mNbVerts = N;

    if (cfg.isShared)
    {
        c.mOrderedSharedTrianglePairVertexIndices =
            reinterpret_cast<decltype(c.mOrderedSharedTrianglePairVertexIndices)>(size_t(bPairVtx.addr));
        c.mSharedTriPairRemapOutputCP =
            reinterpret_cast<decltype(c.mSharedTriPairRemapOutputCP)>(size_t(bWrite.addr));
        c.mSharedTriPairAccumulatedCopiesCP =
            reinterpret_cast<decltype(c.mSharedTriPairAccumulatedCopiesCP)>(size_t(bCopies.addr));
        c.mSharedTriPairAccumulatedPartitionsCP =
            reinterpret_cast<decltype(c.mSharedTriPairAccumulatedPartitionsCP)>(size_t(bParts.addr));
        c.mOrderedSharedRestEdge0_edge1 =
            reinterpret_cast<decltype(c.mOrderedSharedRestEdge0_edge1)>(size_t(bRestEdge.addr));
        c.mOrderedSharedRestEdgeLength_material0_material1 =
            reinterpret_cast<decltype(c.mOrderedSharedRestEdgeLength_material0_material1)>(size_t(bRestEdgeLenMat.addr));
        c.mOrderedSharedRestBendingAngle_flexuralStiffness_damping =
            reinterpret_cast<decltype(c.mOrderedSharedRestBendingAngle_flexuralStiffness_damping)>(size_t(bRestBend.addr));
        c.mOrderedSharedTriangleLambdas =
            reinterpret_cast<decltype(c.mOrderedSharedTriangleLambdas)>(size_t(bMemLam.addr));
        c.mSharedBendingLambdas =
            reinterpret_cast<decltype(c.mSharedBendingLambdas)>(size_t(bBendLam.addr));
        c.mNbSharedTrianglePairs = 1u;
        c.mNbSharedTriPairPartitions = 1u;
    }
    else
    {
        c.mOrderedNonSharedTrianglePairVertexIndices =
            reinterpret_cast<decltype(c.mOrderedNonSharedTrianglePairVertexIndices)>(size_t(bPairVtx.addr));
        c.mNonSharedTriPairRemapOutputCP =
            reinterpret_cast<decltype(c.mNonSharedTriPairRemapOutputCP)>(size_t(bWrite.addr));
        c.mNonSharedTriPairAccumulatedCopiesCP =
            reinterpret_cast<decltype(c.mNonSharedTriPairAccumulatedCopiesCP)>(size_t(bCopies.addr));
        c.mNonSharedTriPairAccumulatedPartitionsCP =
            reinterpret_cast<decltype(c.mNonSharedTriPairAccumulatedPartitionsCP)>(size_t(bParts.addr));
        c.mOrderedNonSharedRestBendingAngle_flexuralStiffness_damping =
            reinterpret_cast<decltype(c.mOrderedNonSharedRestBendingAngle_flexuralStiffness_damping)>(size_t(bRestBend.addr));
        c.mNonSharedBendingLambdas =
            reinterpret_cast<decltype(c.mNonSharedBendingLambdas)>(size_t(bBendLam.addr));
        c.mNbNonSharedTrianglePairs = 1u;
        c.mNbNonSharedTriPairPartitions = 1u;
        c.mNonSharedTriPair_hasActiveBending = true;   // else the kernel early-returns
    }

    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    // --- dispatch solve ---------------------------------------------------
    SolveTriPairPush push = {};
    push.femCloths = bDesc.addr;
    push.activeFEMCloths = bActive.addr;
    push.materials = bMat.addr;
    push.dt = dt;
    push.partitionId = 0u;   // partitionId (launch) / firstClusterId (cluster)
    push.flagsWord = (cfg.isShared ? 1u : 0u) | ((cfg.isTGS ? 1u : 0u) << 8u);

    const char* solveEntry = cfg.useCluster ? "cloth_solveTrianglePairEnergyClusterLaunch"
                                            : "cloth_solveTrianglePairEnergyLaunch";
    Kernel kSolve("femClothSolvePairBda.spv", solveEntry, {}, sizeof(SolveTriPairPush));
    kSolve.dispatch(/*gridX numBlocks*/ 1u, {}, push, /*gridY nbActive*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    // --- dispatch average -------------------------------------------------
    AvgPairVertsPush apush = {};
    apush.femCloths = bDesc.addr;
    apush.activeFEMCloths = bActive.addr;
    apush.dtInv = invDt;
    apush.isSharedWord = cfg.isShared ? 1u : 0u;
    Kernel kAvg("femClothSolvePairBda.spv", "cloth_averageTrianglePairVertsLaunch", {}, sizeof(AvgPairVertsPush));
    kAvg.dispatch(/*gridX numBlocks*/ 1u, {}, apush, /*gridY nbActive*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    // --- CPU reference ----------------------------------------------------
    p71ref::F4 rx0 = { positions[0].x, positions[0].y, positions[0].z, positions[0].w };
    p71ref::F4 rx1 = { positions[1].x, positions[1].y, positions[1].z, positions[1].w };
    p71ref::F4 rx2 = { positions[2].x, positions[2].y, positions[2].z, positions[2].w };
    p71ref::F4 rx3 = { positions[3].x, positions[3].y, positions[3].z, positions[3].w };
    const p71ref::F4 vrc = { 1.0f, 1.0f, 1.0f, 1.0f };   // 1 copy per vertex

    float m0x = membraneLambdas[0].x, m0y = membraneLambdas[0].y;
    float m1x = membraneLambdas[1].x, m1y = membraneLambdas[1].y;
    float bendLam = bendingLambda[0];

    if (cfg.isShared)
    {
        const p71ref::F4 re = { restEdge.x, restEdge.y, restEdge.z, restEdge.w };
        p71ref::refSharedPair(rx0, rx1, rx2, rx3, vrc, re, restSharedEdgeLength,
                              mat.youngs, mat.poissons, mat.thickness,
                              mat.youngs, mat.poissons, mat.thickness,
                              m0x, m0y, m1x, m1y, restBendingAngle, kInv, dt, bendLam, cfg.isTGS);
    }
    else
    {
        p71ref::refBendingPair(rx0, rx1, rx2, rx3, vrc, restBendingAngle, kInv, dt, bendLam, cfg.isTGS);
    }

    const p71ref::F4 solved[4] = { rx0, rx1, rx2, rx3 };
    std::vector<F4> refPos(N), refVel(N), refAcc(N);
    for (uint32_t v = 0; v < N; ++v)
    {
        const F4 orig = positions[v];
        const F4 d = { solved[v].x - orig.x, solved[v].y - orig.y, solved[v].z - orig.z, 0.0f };
        refPos[v] = { orig.x + d.x, orig.y + d.y, orig.z + d.z, orig.w };
        refVel[v] = { vels[v].x + d.x * invDt, vels[v].y + d.y * invDt, vels[v].z + d.z * invDt, vels[v].w };
        refAcc[v] = { accum[v].x + d.x, accum[v].y + d.y, accum[v].z + d.z, accum[v].w };
    }

    // --- download + compare ----------------------------------------------
    const auto gPos = downloadBuffer<F4>(dev, bPos.h, N);
    const auto gVel = downloadBuffer<F4>(dev, bVel.h, N);
    const auto gAcc = downloadBuffer<F4>(dev, bAcc.h, N);
    const auto gMemLam = downloadBuffer<F2>(dev, bMemLam.h, 2);
    const auto gBendLam = downloadBuffer<float>(dev, bBendLam.h, 1);

    const char* tag = cfg.isShared ? "shared" : "nonShared";
    // Bit-exact whenever bending is disabled OR the atan2 hits its exact +0 special
    // value (flat pair); otherwise the atan2 downstream is a tolerance path.
    const bool bitExact = !cfg.bendingOn || cfg.flat;

    // Teeth: the solve must actually move at least one vertex.
    bool moved = false;
    for (uint32_t v = 0; v < N; ++v)
        moved = moved || f2u(gPos[v].x) != f2u(positions[v].x) || f2u(gPos[v].y) != f2u(positions[v].y)
                      || f2u(gPos[v].z) != f2u(positions[v].z);
    PXG_CHECK_MSG(moved, "%s tri-pair solve should move a vertex (teeth, isTGS=%d, bendOn=%d, flat=%d)",
                  tag, int(cfg.isTGS), int(cfg.bendingOn), int(cfg.flat));

    // Tight documented tolerance for the atan2 seam.  atan2 diverges by at most a
    // few ulp (~1e-7 rel) between llvmpipe's GLSL.std.450 Atan2 and glibc; that
    // propagates LINEARLY through C -> deltaLambda -> the per-vertex bending
    // displacement `d`, whose absolute error is bounded by ~5e-7 here.  Position and
    // accumulated-delta inherit `d` directly, so a 1e-6 absolute floor (+ 1e-5
    // relative) covers them.  Velocity inherits `d * invDt` (invDt == 60), so its
    // absolute floor is scaled by invDt (1e-6 * 60 == 6e-5) — the SAME underlying
    // seam bound, just amplified by the fixed invDt, not a looser tolerance.
    const float relTol = 1.0e-5f;
    const float posAbs = 1.0e-6f;
    const float velAbs = posAbs * invDt;   // velocity = position-change * invDt
    auto close = [&](float a, float b, float absTol) { return std::fabs(a - b) <= absTol + relTol * std::fabs(a); };

    size_t mm = 0;
    float maxErr = 0.0f;
    auto cmp4 = [&](const char* nm, uint32_t i, const F4& r, const F4& g, float absTol) {
        bool ok;
        if (bitExact)
            ok = f2u(r.x) == f2u(g.x) && f2u(r.y) == f2u(g.y) && f2u(r.z) == f2u(g.z) && f2u(r.w) == f2u(g.w);
        else
            ok = close(r.x, g.x, absTol) && close(r.y, g.y, absTol) && close(r.z, g.z, absTol) && close(r.w, g.w, absTol);
        maxErr = std::fmax(maxErr, std::fmax(std::fabs(r.x - g.x), std::fmax(std::fabs(r.y - g.y), std::fabs(r.z - g.z))));
        if (!ok)
        {
            ++mm;
            if (mm <= 8)
                std::printf("  %s %s[%u] MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", tag, nm, i,
                            r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w);
        }
    };
    for (uint32_t v = 0; v < N; ++v)
    {
        cmp4("pos", v, refPos[v], gPos[v], posAbs);
        cmp4("vel", v, refVel[v], gVel[v], velAbs);
        cmp4("acc", v, refAcc[v], gAcc[v], posAbs);
    }

    // Lambdas (PGS only — TGS never reads/writes them).  The MEMBRANE lambdas are
    // always bit-exact (the membrane half is bit-exact); the BENDING lambda is
    // bit-exact when the bending is bit-exact, else within tolerance.
    if (!cfg.isTGS)
    {
        if (cfg.isShared)
        {
            if (f2u(gMemLam[0].x) != f2u(m0x) || f2u(gMemLam[0].y) != f2u(m0y)
                || f2u(gMemLam[1].x) != f2u(m1x) || f2u(gMemLam[1].y) != f2u(m1y))
            {
                ++mm;
                std::printf("  %s membrane lambda MM: ref(%g,%g / %g,%g) got(%g,%g / %g,%g)\n", tag,
                            m0x, m0y, m1x, m1y, gMemLam[0].x, gMemLam[0].y, gMemLam[1].x, gMemLam[1].y);
            }
        }
        bool bendOk = bitExact ? (f2u(gBendLam[0]) == f2u(bendLam)) : close(bendLam, gBendLam[0], posAbs);
        if (!bendOk)
        {
            ++mm;
            std::printf("  %s bending lambda MM: ref %g got %g\n", tag, bendLam, gBendLam[0]);
        }
    }

    PXG_CHECK_MSG(mm == 0, "%s tri-pair solve %s vs CPU (%zu mismatches, isTGS=%d, bendOn=%d, flat=%d, cluster=%d, maxErr=%g)",
                  tag, bitExact ? "bit-exact" : "within-tol", mm, int(cfg.isTGS), int(cfg.bendingOn),
                  int(cfg.flat), int(cfg.useCluster), maxErr);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bAcc.h);
    dev->destroyBuffer(bCP.h); dev->destroyBuffer(bPairVtx.h); dev->destroyBuffer(bWrite.h);
    dev->destroyBuffer(bCopies.h); dev->destroyBuffer(bParts.h); dev->destroyBuffer(bRestEdge.h);
    dev->destroyBuffer(bRestEdgeLenMat.h); dev->destroyBuffer(bRestBend.h); dev->destroyBuffer(bMemLam.h);
    dev->destroyBuffer(bBendLam.h); dev->destroyBuffer(bMat.h); dev->destroyBuffer(bDesc.h);
    dev->destroyBuffer(bActive.h);
}

// Shared pair, bending OFF: membrane + copies/remap/average are BIT-EXACT.
PXG_TEST(femClothSimSolveTriPairSharedMembranePGS)
{
    runSolveTriPair({ /*isShared*/ true, /*bendingOn*/ false, /*flat*/ false, /*isTGS*/ false, /*useCluster*/ false });
}
PXG_TEST(femClothSimSolveTriPairSharedMembraneTGS)
{
    runSolveTriPair({ /*isShared*/ true, /*bendingOn*/ false, /*flat*/ false, /*isTGS*/ true, /*useCluster*/ false });
}

// Shared pair, bending ON (bent): membrane bit-exact, bending within tolerance.
PXG_TEST(femClothSimSolveTriPairSharedBendPGS)
{
    runSolveTriPair({ /*isShared*/ true, /*bendingOn*/ true, /*flat*/ false, /*isTGS*/ false, /*useCluster*/ false });
}
PXG_TEST(femClothSimSolveTriPairSharedBendTGS)
{
    runSolveTriPair({ /*isShared*/ true, /*bendingOn*/ true, /*flat*/ false, /*isTGS*/ true, /*useCluster*/ false });
}

// Non-shared FLAT pair, bending ON: atan2(0,+x)==+0 exact -> bending BIT-EXACT.
PXG_TEST(femClothSimSolveTriPairNonSharedFlatPGS)
{
    runSolveTriPair({ /*isShared*/ false, /*bendingOn*/ true, /*flat*/ true, /*isTGS*/ false, /*useCluster*/ false });
}
PXG_TEST(femClothSimSolveTriPairNonSharedFlatTGS)
{
    runSolveTriPair({ /*isShared*/ false, /*bendingOn*/ true, /*flat*/ true, /*isTGS*/ true, /*useCluster*/ false });
}

// Non-shared BENT pair, bending ON: the general on-device atan2 path, within tolerance.
PXG_TEST(femClothSimSolveTriPairNonSharedBentPGS)
{
    runSolveTriPair({ /*isShared*/ false, /*bendingOn*/ true, /*flat*/ false, /*isTGS*/ false, /*useCluster*/ false });
}
PXG_TEST(femClothSimSolveTriPairNonSharedBentTGS)
{
    runSolveTriPair({ /*isShared*/ false, /*bendingOn*/ true, /*flat*/ false, /*isTGS*/ true, /*useCluster*/ false });
}

// Cluster twin (the barrier-looped partition path): shared membrane bit-exact +
// non-shared flat bending bit-exact, dispatched through the *ClusterLaunch entry.
PXG_TEST(femClothSimSolveTriPairClusterPGS)
{
    runSolveTriPair({ /*isShared*/ true, /*bendingOn*/ false, /*flat*/ false, /*isTGS*/ false, /*useCluster*/ true });
    runSolveTriPair({ /*isShared*/ false, /*bendingOn*/ true, /*flat*/ true, /*isTGS*/ false, /*useCluster*/ true });
}

// ===========================================================================
// P7.1 tranche 3a: the cloth RIGID-ATTACHMENT constraint kernels — the machinery
// to PIN a cloth vertex to a rigid / world anchor (toward a hanging cloth).  ONE
// cloth vertex + ONE static/world attachment: run prepare -> solve, and compare the
// prepared constraint block AND the solved mDeltaPos (the cloth-side pull toward the
// anchor) BIT-EXACT against a CPU transcription of FEMClothConstraintPrep.cu:387 +
// FEMClothExternalSolve.cu:1371/:1571 + attachments.cuh.  The attachment solve is
// plain linear algebra, so every field is fully bit-exact (no transcendental seam).
// ===========================================================================

// Host replica of the block-of-32 PxgFEMRigidAttachmentConstraint (PxgFEMCore.h:191,
// verbatim hand-copy — PxgFEMCore.h is not includable in this CUDA-toolkit-free test
// TU: it pulls PxgCudaBuffer.h -> cuda.h).  The layout is a trivial fixed-array SoA
// (float4[32] blocks + PxU32[32] + PxU64[32], each 16-byte-aligned), pinned by the
// self static_asserts below.  This is the byte image the prepare WRITES and the
// solve READS, matching the shader's RAC_* byte offsets (femClothRigidAttachBda.slang).
struct RigidAttachConstraintHost
{
    F4 baryOrType[32];                  // @0
    F4 raXn0_biasW[32];                 // @512
    F4 raXn1_biasW[32];                 // @1024
    F4 raXn2_biasW[32];                 // @1536
    F4 velMultiplierXYZ_invMassW[32];   // @2048
    F4 low_high_limits[32];             // @2560
    F4 axis_angle[32];                  // @3072
    uint32_t elemId[32];                // @3584
    uint64_t rigidId[32];               // @3712
};
static_assert(sizeof(RigidAttachConstraintHost) == 3968, "PxgFEMRigidAttachmentConstraint sizeof == 3968");
static_assert(offsetof(RigidAttachConstraintHost, raXn0_biasW) == 512, "raXn0_biasW @512");
static_assert(offsetof(RigidAttachConstraintHost, raXn1_biasW) == 1024, "raXn1_biasW @1024");
static_assert(offsetof(RigidAttachConstraintHost, raXn2_biasW) == 1536, "raXn2_biasW @1536");
static_assert(offsetof(RigidAttachConstraintHost, velMultiplierXYZ_invMassW) == 2048, "velMult @2048");
static_assert(offsetof(RigidAttachConstraintHost, low_high_limits) == 2560, "low_high_limits @2560");
static_assert(offsetof(RigidAttachConstraintHost, axis_angle) == 3072, "axis_angle @3072");
static_assert(offsetof(RigidAttachConstraintHost, elemId) == 3584, "elemId @3584");
static_assert(offsetof(RigidAttachConstraintHost, rigidId) == 3712, "rigidId @3712");

// ---------------------------------------------------------------------------
// bit-exact CPU reference for the prepare + PGS/TGS attachment solves.  The rigid /
// world-anchor math (PxQuat::rotate, getNormalized via 1/sqrt, PxMat33*PxVec3) and
// the calculateAttachmentDeltaImpulsePGS/TGS leaves (attachments.cuh) are transcribed
// verbatim into a private namespace.
// ---------------------------------------------------------------------------
namespace p73ref
{
struct V3 { float x, y, z; };
static V3 cross3(V3 a, V3 b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// PxQuat::rotate (PxQuat.h:286).
static V3 quatRotate(float qx, float qy, float qz, float qw, V3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = qw * qw - 0.5f;
    const float d2 = qx * vx + qy * vy + qz * vz;
    return V3{ vx * w2 + (qy * vz - qz * vy) * qw + qx * d2,
               vy * w2 + (qz * vx - qx * vz) * qw + qy * d2,
               vz * w2 + (qx * vy - qy * vx) * qw + qz * d2 };
}
// PxVec3::getNormalized (PxVec3.h:286); PxRecipSqrt == 1.0/sqrt (unix intrinsic).
static V3 getNormalized(V3 v)
{
    const float m = v.x * v.x + v.y * v.y + v.z * v.z;
    if (m > 0.0f) { const float rs = 1.0f / std::sqrt(m); return V3{ v.x * rs, v.y * rs, v.z * rs }; }
    return V3{ 0.0f, 0.0f, 0.0f };
}
// PxMat33 (column-major [9]) * PxVec3.
static V3 mat33Mul(const float m[9], V3 v)
{
    return V3{ m[0] * v.x + m[3] * v.y + m[6] * v.z,
               m[1] * v.x + m[4] * v.y + m[7] * v.z,
               m[2] * v.x + m[5] * v.y + m[8] * v.z };
}

// cloth_rigidAttachmentPrepareLaunch (FEMClothConstraintPrep.cu:387) — the rigid /
// static (non-articulation) path.  Fills constraint slot [0].
static void refPrepare(RigidAttachConstraintHost& c,
                       F4 attachmentPose,  // cloth vertex pos_invMass (invMass1 in .w)
                       V3 ra4, V3 axisIn, float axisAngleW, F4 lowHigh, F4 baryOrType1,
                       uint32_t elemId, uint64_t rigidId,
                       float qx, float qy, float qz, float qw, V3 bodyFrameP,
                       float invMass0, const float sqrtInvInertia[9], bool isStatic)
{
    const float invMass1 = attachmentPose.w;
    const V3 point = V3{ attachmentPose.x, attachmentPose.y, attachmentPose.z };
    const V3 normal0 = V3{ 1.0f, 0.0f, 0.0f };
    const V3 normal1 = V3{ 0.0f, 1.0f, 0.0f };
    const V3 normal2 = V3{ 0.0f, 0.0f, 1.0f };

    float invSqrt[9];
    float inertiaScale = 1.0f;
    if (invMass0 == 0.0f && !isStatic)
    {
        invSqrt[0] = 1.0f; invSqrt[1] = 0.0f; invSqrt[2] = 0.0f;
        invSqrt[3] = 0.0f; invSqrt[4] = 1.0f; invSqrt[5] = 0.0f;
        invSqrt[6] = 0.0f; invSqrt[7] = 0.0f; invSqrt[8] = 1.0f;
        inertiaScale = 0.0f;
    }
    else
    {
        for (int k = 0; k < 9; ++k) invSqrt[k] = sqrtInvInertia[k];
    }

    const V3 ra = quatRotate(qx, qy, qz, qw, ra4);
    const V3 error = V3{ ra.x + bodyFrameP.x - point.x, ra.y + bodyFrameP.y - point.y, ra.z + bodyFrameP.z - point.z };
    const V3 worldAxis = getNormalized(quatRotate(qx, qy, qz, qw, axisIn));

    const V3 raXn0 = cross3(ra, normal0);
    const V3 raXn1 = cross3(ra, normal1);
    const V3 raXn2 = cross3(ra, normal2);

    const V3 rs0 = mat33Mul(invSqrt, raXn0);
    const V3 rs1 = mat33Mul(invSqrt, raXn1);
    const V3 rs2 = mat33Mul(invSqrt, raXn2);

    const float resp0 = dot3(rs0, rs0) * inertiaScale + invMass0 + invMass1;
    const float resp1 = dot3(rs1, rs1) * inertiaScale + invMass0 + invMass1;
    const float resp2 = dot3(rs2, rs2) * inertiaScale + invMass0 + invMass1;

    const float vm0 = (resp0 > 0.0f) ? (1.0f / resp0) : 0.0f;
    const float vm1 = (resp1 > 0.0f) ? (1.0f / resp1) : 0.0f;
    const float vm2 = (resp2 > 0.0f) ? (1.0f / resp2) : 0.0f;

    const float be0 = dot3(error, normal0);
    const float be1 = dot3(error, normal1);
    const float be2 = dot3(error, normal2);

    c.raXn0_biasW[0] = F4{ rs0.x, rs0.y, rs0.z, be0 };
    c.raXn1_biasW[0] = F4{ rs1.x, rs1.y, rs1.z, be1 };
    c.raXn2_biasW[0] = F4{ rs2.x, rs2.y, rs2.z, be2 };
    c.velMultiplierXYZ_invMassW[0] = F4{ vm0, vm1, vm2, invMass0 };
    c.elemId[0] = elemId;
    c.rigidId[0] = rigidId;
    c.baryOrType[0] = baryOrType1;
    c.low_high_limits[0] = lowHigh;
    c.axis_angle[0] = F4{ worldAxis.x, worldAxis.y, worldAxis.z, axisAngleW };
}

// calculateAttachmentDeltaImpulsePGS (attachments.cuh:83) + cloth-side scatter
// (FEMClothExternalSolve.cu:1470) — NOTE the CUDA passes raXn0_biasW THREE TIMES
// (:1451), so raXn0/1/2 and all three position-error components come from raXn0_biasW.
static F4 refSolvePGS(const RigidAttachConstraintHost& c, F4 clothVel, V3 rigidLinVel, V3 rigidAngVel, float dt)
{
    const float invDt = 1.0f / dt;
    const V3 linVel1 = V3{ clothVel.x, clothVel.y, clothVel.z };
    const F4 r0 = c.raXn0_biasW[0];
    const F4 vm = c.velMultiplierXYZ_invMassW[0];
    const V3 raXn0 = V3{ r0.x, r0.y, r0.z };

    const float velOfRigidX = rigidLinVel.x + dot3(rigidAngVel, raXn0);
    const float velOfRigidY = rigidLinVel.y + dot3(rigidAngVel, raXn0);
    const float velOfRigidZ = rigidLinVel.z + dot3(rigidAngVel, raXn0);

    const float positionError = r0.w;
    const float biasFactor = 0.5f;
    const V3 velError = V3{ linVel1.x - positionError * biasFactor * invDt,
                           linVel1.y - positionError * biasFactor * invDt,
                           linVel1.z - positionError * biasFactor * invDt };

    const float dF0 = (velError.x - velOfRigidX) * vm.x;
    const float dF1 = (velError.y - velOfRigidY) * vm.y;
    const float dF2 = (velError.z - velOfRigidZ) * vm.z;

    F4 out = F4{ 0.0f, 0.0f, 0.0f, 0.0f };
    if (dF0 != 0.0f || dF1 != 0.0f || dF2 != 0.0f)
    {
        const V3 deltaPos = V3{ (-dF0) * dt, (-dF1) * dt, (-dF2) * dt };
        out = F4{ deltaPos.x * clothVel.w, deltaPos.y * clothVel.w, deltaPos.z * clothVel.w, 1.0f };
    }
    return out;
}

// calculateAttachmentDeltaImpulseTGS (attachments.cuh:147, offset overload — distinct
// raXn rows) + cloth-side scatter (FEMClothExternalSolve.cu:1667).
static F4 refSolveTGS(const RigidAttachConstraintHost& c, F4 clothAccumDelta,
                      V3 rigidLinVel, V3 rigidAngVel, V3 rigidLinDelta, V3 rigidAngDelta, float dt)
{
    const V3 linDelta1 = V3{ clothAccumDelta.x, clothAccumDelta.y, clothAccumDelta.z };
    const F4 r0 = c.raXn0_biasW[0], r1 = c.raXn1_biasW[0], r2 = c.raXn2_biasW[0];
    const F4 vm = c.velMultiplierXYZ_invMassW[0];
    const V3 raXn0 = V3{ r0.x, r0.y, r0.z }, raXn1 = V3{ r1.x, r1.y, r1.z }, raXn2 = V3{ r2.x, r2.y, r2.z };

    const float vo0 = rigidLinVel.x + dot3(rigidAngVel, raXn0);
    const float vo1 = rigidLinVel.y + dot3(rigidAngVel, raXn1);
    const float vo2 = rigidLinVel.z + dot3(rigidAngVel, raXn2);

    const V3 linDelta = V3{ linDelta1.x - rigidLinDelta.x, linDelta1.y - rigidLinDelta.y, linDelta1.z - rigidLinDelta.z };
    const float peX = r0.w, peY = r1.w, peZ = r2.w;
    const V3 tgsError = V3{ linDelta.x - peX - dot3(rigidAngDelta, raXn0),
                           linDelta.y - peY - dot3(rigidAngDelta, raXn1),
                           linDelta.z - peZ - dot3(rigidAngDelta, raXn2) };

    const float velDt = dt; // isVelocityIteration == false
    const float BIAS = 0.7f; // FEMCLOTH_BIAS_COEFFICIENT
    const float dF0 = (tgsError.x - vo0 * velDt) * vm.x * BIAS;
    const float dF1 = (tgsError.y - vo1 * velDt) * vm.y * BIAS;
    const float dF2 = (tgsError.z - vo2 * velDt) * vm.z * BIAS;

    F4 out = F4{ 0.0f, 0.0f, 0.0f, 0.0f };
    if (dF0 != 0.0f || dF1 != 0.0f || dF2 != 0.0f)
    {
        const V3 deltaPos = V3{ -dF0, -dF1, -dF2 };
        out = F4{ deltaPos.x * clothAccumDelta.w, deltaPos.y * clothAccumDelta.w, deltaPos.z * clothAccumDelta.w, 1.0f };
    }
    return out;
}
} // namespace p73ref

static void pokePtr(std::vector<uint8_t>& buf, size_t off, uint64_t v) { std::memcpy(buf.data() + off, &v, 8); }
static void pokeU32(std::vector<uint8_t>& buf, size_t off, uint32_t v) { std::memcpy(buf.data() + off, &v, 4); }

// ---------------------------------------------------------------------------
// Descriptor-mirror layout proof for the NEW tranche-3a mirrors.
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimRigidAttachDescLayout)
{
    // PxNodeIndex: static/world anchor is mID == PX_INVALID_NODE (low 32 bits).
    PXG_CHECK_MSG(sizeof(PxNodeIndex) == 8, "PxNodeIndex sizeof == 8 (got %zu)", sizeof(PxNodeIndex));
    PxNodeIndex staticNode; // default ctor -> PX_INVALID_NODE
    PXG_CHECK(staticNode.isStaticBody());
    PXG_CHECK(staticNode.getInd() == 0x00000000FFFFFFFFull);

    // PxgFEMRigidAttachment (PxgSimulationCoreDesc.h:208) — the shader's RA_* offsets.
    PXG_CHECK_MSG(sizeof(PxgFEMRigidAttachment) == 96, "PxgFEMRigidAttachment sizeof == 96 (got %zu)", sizeof(PxgFEMRigidAttachment));
    PXG_CHECK(offsetof(PxgFEMRigidAttachment, index0) == 0);
    PXG_CHECK(offsetof(PxgFEMRigidAttachment, index1) == 8);
    PXG_CHECK(offsetof(PxgFEMRigidAttachment, localPose0) == 32);
    PXG_CHECK(offsetof(PxgFEMRigidAttachment, coneLimitParams) + offsetof(PxgConeLimitParams, low_high_limits) == 48);
    PXG_CHECK(offsetof(PxgFEMRigidAttachment, coneLimitParams) + offsetof(PxgConeLimitParams, axis_angle) == 64);
    PXG_CHECK(offsetof(PxgFEMRigidAttachment, baryOrType1) == 80);

    // PxgSolverBodyData / PxgSolverTxIData (PxgSolverBody.h) — the prepare's anchor
    // frame reads.  The device float4 initialLinVelXYZ_invMassW @16 == { initialLinVel
    // (@16, 12B), invMass (@28, 4B) } on the host (non-CUDA union branch).
    PXG_CHECK_MSG(sizeof(PxgSolverBodyData) == 96, "PxgSolverBodyData sizeof == 96 (got %zu)", sizeof(PxgSolverBodyData));
    PXG_CHECK(offsetof(PxgSolverBodyData, initialLinVel) == 16);
    PXG_CHECK(offsetof(PxgSolverBodyData, invMass) == 28);
    PXG_CHECK_MSG(sizeof(PxgSolverTxIData) == 64, "PxgSolverTxIData sizeof == 64 (got %zu)", sizeof(PxgSolverTxIData));
    PXG_CHECK(offsetof(PxgSolverTxIData, sqrtInvInertia) == 28);
    PXG_CHECK_MSG(sizeof(PxAlignedTransform) == 32, "PxAlignedTransform sizeof == 32 (got %zu)", sizeof(PxAlignedTransform));

    // block-of-32 constraint replica pinned above by static_asserts; re-affirm the stride.
    PXG_CHECK(sizeof(RigidAttachConstraintHost) == 3968);
}

// ---------------------------------------------------------------------------
// cloth_rigidAttachmentPrepareLaunch -> cloth_solveRigidClothAttachment{,TGS}Launch:
// pin ONE cloth vertex to a fixed world anchor.
// ---------------------------------------------------------------------------
static void runRigidAttach(bool isTGS)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 1.0f / 60.0f;

    // --- cloth vertex (id 0, elemIdx 0), invMass = 0.5 (movable) ----------
    const float invMass = 0.5f;
    std::vector<F4> positions = { { 1.0f, 2.0f, 3.0f, invMass } };
    std::vector<F4> velocities = { { 0.10f, -0.20f, 0.05f, invMass } };
    std::vector<F4> accumDelta = { { 0.02f, -0.03f, 0.04f, invMass } };
    std::vector<F4> deltaPos = { { 0.0f, 0.0f, 0.0f, 0.0f } };   // solve output
    std::vector<uint32_t> triVtx = { 0u, 0u, 0u, 0u };           // unused (vertex attach)

    DevBuf bPos = makeDev(dev, positions);
    DevBuf bVel = makeDev(dev, velocities);
    DevBuf bAcc = makeDev(dev, accumDelta);
    DevBuf bDelta = makeDev(dev, deltaPos);
    DevBuf bTri = makeDev(dev, triVtx);

    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bPos.addr));
    c.mVelocity_InvMass = reinterpret_cast<decltype(c.mVelocity_InvMass)>(size_t(bVel.addr));
    c.mAccumulatedDeltaPos = reinterpret_cast<decltype(c.mAccumulatedDeltaPos)>(size_t(bAcc.addr));
    c.mDeltaPos = reinterpret_cast<decltype(c.mDeltaPos)>(size_t(bDelta.addr));
    c.mTriangleVertexIndices = reinterpret_cast<decltype(c.mTriangleVertexIndices)>(size_t(bTri.addr));
    c.mNbVerts = 1u;
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    // --- the world anchor: static node, identity body frame at the origin -
    const F4 anchor = { 2.0f, 0.5f, 3.5f, 0.0f };   // localPose0 (world = local, identity frame)
    const uint64_t staticNodeInd = 0x00000000FFFFFFFFull;

    // PxgFEMRigidAttachment (96 bytes) via the real engine struct.
    PxgFEMRigidAttachment att;
    std::memset(&att, 0, sizeof(att));
    att.index0 = staticNodeInd;                          // rigid node index (static)
    att.index1 = 0ull;                                   // elemId = (clothId 0 << 20) | elemIdx 0
    att.localPose0 = make_float4(anchor.x, anchor.y, anchor.z, anchor.w);
    att.coneLimitParams.low_high_limits = make_float4(-1.0f, -1.0f, -1.0f, -1.0f); // cone disabled
    att.coneLimitParams.axis_angle = make_float4(1.0f, 0.0f, 0.0f, -1.0f);         // axis (1,0,0), angle -1
    att.baryOrType1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);                         // vertex type
    std::vector<PxgFEMRigidAttachment> atts = { att };
    DevBuf bAtt = makeDev(dev, atts);

    std::vector<uint32_t> activeAtt = { 0u };
    DevBuf bActiveAtt = makeDev(dev, activeAtt);
    std::vector<uint64_t> attachIds = { 0xBADBADBADull };   // output (PxNodeIndex)
    DevBuf bAttachIds = makeDev(dev, attachIds);

    // constraint block (3968 bytes) seeded with a sentinel pattern.
    std::vector<uint8_t> constraintBytes(sizeof(RigidAttachConstraintHost), 0xCD);
    DevBuf bConstraint = makeDev(dev, constraintBytes);

    // rigidDeltaVel: prepare zeroes [0] and [0+numAttachments]; deferred otherwise.
    std::vector<F4> rigidDeltaVel = { { 9.f, 9.f, 9.f, 9.f }, { 9.f, 9.f, 9.f, 9.f } };
    DevBuf bRigidDeltaVel = makeDev(dev, rigidDeltaVel);

    // --- anchor rigid frame / mass / inertia (solver body 0 = world/zero) --
    // body2WorldPool[0] = identity quat (0,0,0,1), p = (0,0,0,0).
    std::vector<F4> body2World = { { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } };
    DevBuf bBody2World = makeDev(dev, body2World);
    std::vector<uint8_t> solverBodyData(96, 0);   // initialLinVelXYZ_invMassW @16 = 0 -> invMass0 = 0
    DevBuf bSolverBodyData = makeDev(dev, solverBodyData);
    std::vector<uint8_t> solverTxIData(64, 0);    // sqrtInvInertia @28 = 0 matrix
    DevBuf bSolverTxIData = makeDev(dev, solverTxIData);

    // --- solver-body velocity buffers (world anchor at rest) --------------
    // numSolverBodies = 1, accumulatedBodyDeltaVOffset = 0.  Read slots (0,1 for
    // outSolverVelocity; 0,1,2 for solverBodyVelPool) are ZERO; the trailing slots
    // carry sentinels so a wrong reader offset would read nonzero and mismatch.
    const uint32_t numSolverBodies = 1u;
    const uint32_t accOffset = 0u;
    std::vector<F4> outSolverVelocity = { { 0.f, 0.f, 0.f, 0.f }, { 0.f, 0.f, 0.f, 0.f },
                                          { 77.f, 78.f, 79.f, 80.f }, { 81.f, 82.f, 83.f, 84.f } };
    DevBuf bOutSolverVel = makeDev(dev, outSolverVelocity);
    std::vector<F4> solverBodyVelPool = { { 0.f, 0.f, 0.f, 0.f }, { 0.f, 0.f, 0.f, 0.f },
                                          { 0.f, 0.f, 0.f, 0.f }, { 61.f, 62.f, 63.f, 64.f } };
    DevBuf bSolverBodyVelPool = makeDev(dev, solverBodyVelPool);
    std::vector<uint32_t> solverBodyIndices = { 0u };   // never dereferenced for a static anchor
    DevBuf bSolverBodyIndices = makeDev(dev, solverBodyIndices);

    // --- raw descriptor buffers, pointers poked at the mirror offsets -----
    std::vector<uint8_t> prepDesc(304, 0);          // PxgConstraintPrepareDescGpu
    pokePtr(prepDesc, 64, bSolverBodyData.addr);    // solverBodyDataPool @64
    pokePtr(prepDesc, 72, bSolverTxIData.addr);     // solverBodyTxIDataPool @72
    pokePtr(prepDesc, 168, bBody2World.addr);       // body2WorldPool @168
    DevBuf bPrepDesc = makeDev(dev, prepDesc);

    std::vector<uint8_t> prePrepDesc(760, 0);       // PxgPrePrepDescGpu
    pokePtr(prePrepDesc, 272, bSolverBodyIndices.addr); // solverBodyIndices @272 (static -> unused)
    DevBuf bPrePrepDesc = makeDev(dev, prePrepDesc);

    std::vector<uint8_t> solverCoreDesc(320, 0);    // PxgSolverCoreDescGpu
    pokePtr(solverCoreDesc, 0, bOutSolverVel.addr);        // outSolverVelocity @0
    pokeU32(solverCoreDesc, 268, numSolverBodies);         // numSolverBodies @268
    pokeU32(solverCoreDesc, 276, accOffset);               // accumulatedBodyDeltaVOffset @276
    DevBuf bSolverCoreDesc = makeDev(dev, solverCoreDesc);

    std::vector<uint8_t> sharedDesc(224, 0);        // PxgSolverSharedDesc<IterativeSolveData(TGS)>
    pokePtr(sharedDesc, 160, bSolverBodyVelPool.addr);     // iter_solverBodyVelPool @160
    DevBuf bSharedDesc = makeDev(dev, sharedDesc);

    // ======================= dispatch PREPARE ==========================
    struct PrepPush
    {
        uint64_t clothes, rigidAttachments, activeRigidAttachments, rigidAttachmentIds;
        uint32_t numRigidAttachments, pad36;
        uint64_t constraints, prePrepDesc, prepDesc, sharedDesc, rigidDeltaVel;
    };
    static_assert(sizeof(PrepPush) == 80, "prepare push-constant blob must be 80 bytes");
    PrepPush pp = {};
    pp.clothes = bDesc.addr;
    pp.rigidAttachments = bAtt.addr;
    pp.activeRigidAttachments = bActiveAtt.addr;
    pp.rigidAttachmentIds = bAttachIds.addr;
    pp.numRigidAttachments = 1u;
    pp.constraints = bConstraint.addr;
    pp.prePrepDesc = bPrePrepDesc.addr;
    pp.prepDesc = bPrepDesc.addr;
    pp.sharedDesc = bSharedDesc.addr;
    pp.rigidDeltaVel = bRigidDeltaVel.addr;

    Kernel kPrep("femClothRigidAttachBda.spv", "cloth_rigidAttachmentPrepareLaunch", {}, sizeof(PrepPush));
    kPrep.dispatch(1u, {}, pp, 1u, 1u);
    dev->flush();

    // ---- CPU reference for the prepared constraint ----
    RigidAttachConstraintHost refC;
    std::memset(&refC, 0xCD, sizeof(refC));   // same sentinel seed
    float zeroInertia[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    p73ref::refPrepare(refC, positions[0], p73ref::V3{ anchor.x, anchor.y, anchor.z },
                       p73ref::V3{ 1.0f, 0.0f, 0.0f }, -1.0f,
                       F4{ -1.0f, -1.0f, -1.0f, -1.0f }, F4{ 0.0f, 0.0f, 0.0f, 0.0f },
                       /*elemId*/ 0u, /*rigidId*/ staticNodeInd,
                       /*q*/ 0.0f, 0.0f, 0.0f, 1.0f, p73ref::V3{ 0.0f, 0.0f, 0.0f },
                       /*invMass0*/ 0.0f, zeroInertia, /*isStatic*/ true);

    const auto gConstraintBytes = downloadBuffer<uint8_t>(dev, bConstraint.h, sizeof(RigidAttachConstraintHost));
    RigidAttachConstraintHost gotC;
    std::memcpy(&gotC, gConstraintBytes.data(), sizeof(gotC));
    const auto gAttachIds = downloadBuffer<uint64_t>(dev, bAttachIds.h, 1);
    const auto gRigidDeltaVel = downloadBuffer<F4>(dev, bRigidDeltaVel.h, 2);

    size_t mm = 0;
    auto cmp4 = [&](const char* nm, const F4& r, const F4& g) {
        if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) || f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
        { ++mm; std::printf("  prep %s MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", nm,
                            r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w); }
    };
    cmp4("raXn0_biasW", refC.raXn0_biasW[0], gotC.raXn0_biasW[0]);
    cmp4("raXn1_biasW", refC.raXn1_biasW[0], gotC.raXn1_biasW[0]);
    cmp4("raXn2_biasW", refC.raXn2_biasW[0], gotC.raXn2_biasW[0]);
    cmp4("velMult_invMass", refC.velMultiplierXYZ_invMassW[0], gotC.velMultiplierXYZ_invMassW[0]);
    cmp4("baryOrType", refC.baryOrType[0], gotC.baryOrType[0]);
    cmp4("low_high_limits", refC.low_high_limits[0], gotC.low_high_limits[0]);
    cmp4("axis_angle", refC.axis_angle[0], gotC.axis_angle[0]);
    if (gotC.elemId[0] != refC.elemId[0])
    { ++mm; std::printf("  prep elemId MM: got %u ref %u\n", gotC.elemId[0], refC.elemId[0]); }
    if (gotC.rigidId[0] != refC.rigidId[0])
    { ++mm; std::printf("  prep rigidId MM: got %llu ref %llu\n",
                        (unsigned long long)gotC.rigidId[0], (unsigned long long)refC.rigidId[0]); }
    if (gAttachIds[0] != staticNodeInd)
    { ++mm; std::printf("  rigidAttachmentIds MM: got %llu ref %llu\n",
                        (unsigned long long)gAttachIds[0], (unsigned long long)staticNodeInd); }
    // rigidDeltaVel zeroed by the prepare (deferred-but-zeroed).
    for (uint32_t j = 0; j < 2; ++j)
        if (f2u(gRigidDeltaVel[j].x) != 0u || f2u(gRigidDeltaVel[j].y) != 0u ||
            f2u(gRigidDeltaVel[j].z) != 0u || f2u(gRigidDeltaVel[j].w) != 0u)
        { ++mm; std::printf("  rigidDeltaVel[%u] not zeroed\n", j); }

    // teeth: the prepared position error must equal (anchor - clothPos) componentwise
    // (the pull direction), and be non-zero.
    PXG_CHECK_MSG(f2u(gotC.raXn0_biasW[0].w) == f2u(anchor.x - positions[0].x) &&
                  f2u(gotC.raXn1_biasW[0].w) == f2u(anchor.y - positions[0].y) &&
                  f2u(gotC.raXn2_biasW[0].w) == f2u(anchor.z - positions[0].z),
                  "prepared position error must be (anchor - clothPos)");

    PXG_CHECK_MSG(mm == 0, "cloth_rigidAttachmentPrepareLaunch bit-exact vs CPU (%zu mismatches)", mm);

    // ======================= dispatch SOLVE ============================
    struct SolvePush
    {
        uint64_t clothes, attachments;
        uint32_t numAttachments, pad20;
        uint64_t prePrepDesc, solverCoreDesc, artiCoreDesc, sharedDesc;
        float dt;
        uint32_t pad60;
        uint64_t rigidDeltaVel;
    };
    static_assert(sizeof(SolvePush) == 72, "solve push-constant blob must be 72 bytes");
    SolvePush sp = {};
    sp.clothes = bDesc.addr;
    sp.attachments = bConstraint.addr;
    sp.numAttachments = 1u;
    sp.prePrepDesc = bPrePrepDesc.addr;
    sp.solverCoreDesc = bSolverCoreDesc.addr;
    sp.artiCoreDesc = 0ull;   // never dereferenced (articulation deferred)
    sp.sharedDesc = bSharedDesc.addr;
    sp.dt = dt;
    sp.rigidDeltaVel = bRigidDeltaVel.addr;

    const char* solveEntry = isTGS ? "cloth_solveRigidClothAttachmentTGSLaunch"
                                   : "cloth_solveRigidClothAttachmentLaunch";
    Kernel kSolve("femClothRigidAttachBda.spv", solveEntry, {}, sizeof(SolvePush));
    kSolve.dispatch(1u, {}, sp, 1u, 1u);
    dev->flush();

    // ---- CPU reference for the solved mDeltaPos (rigid at rest -> vel0 = 0) ----
    const p73ref::V3 z3 = { 0.0f, 0.0f, 0.0f };
    F4 refDelta = isTGS ? p73ref::refSolveTGS(gotC, accumDelta[0], z3, z3, z3, z3, dt)
                        : p73ref::refSolvePGS(gotC, velocities[0], z3, z3, dt);

    const auto gDelta = downloadBuffer<F4>(dev, bDelta.h, 1);

    // teeth: the solve must actually pull the vertex (mDeltaPos != 0, w == 1).
    PXG_CHECK_MSG(f2u(gDelta[0].x) != 0u || f2u(gDelta[0].y) != 0u || f2u(gDelta[0].z) != 0u,
                  "attachment solve should pull the cloth vertex (mDeltaPos != 0, isTGS=%d)", int(isTGS));

    size_t mm2 = 0;
    if (f2u(refDelta.x) != f2u(gDelta[0].x) || f2u(refDelta.y) != f2u(gDelta[0].y) ||
        f2u(refDelta.z) != f2u(gDelta[0].z) || f2u(refDelta.w) != f2u(gDelta[0].w))
    { ++mm2; std::printf("  solve mDeltaPos MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n",
                         refDelta.x, refDelta.y, refDelta.z, refDelta.w,
                         gDelta[0].x, gDelta[0].y, gDelta[0].z, gDelta[0].w); }
    PXG_CHECK_MSG(mm2 == 0, "cloth_solveRigidClothAttachment%s bit-exact vs CPU (%zu mismatches)",
                  isTGS ? "TGS" : "PGS", mm2);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bAcc.h);
    dev->destroyBuffer(bDelta.h); dev->destroyBuffer(bTri.h); dev->destroyBuffer(bDesc.h);
    dev->destroyBuffer(bAtt.h); dev->destroyBuffer(bActiveAtt.h); dev->destroyBuffer(bAttachIds.h);
    dev->destroyBuffer(bConstraint.h); dev->destroyBuffer(bRigidDeltaVel.h);
    dev->destroyBuffer(bBody2World.h); dev->destroyBuffer(bSolverBodyData.h); dev->destroyBuffer(bSolverTxIData.h);
    dev->destroyBuffer(bOutSolverVel.h); dev->destroyBuffer(bSolverBodyVelPool.h); dev->destroyBuffer(bSolverBodyIndices.h);
    dev->destroyBuffer(bPrepDesc.h); dev->destroyBuffer(bPrePrepDesc.h);
    dev->destroyBuffer(bSolverCoreDesc.h); dev->destroyBuffer(bSharedDesc.h);
}

PXG_TEST(femClothSimRigidAttachPGS) { runRigidAttach(/*isTGS*/ false); }
PXG_TEST(femClothSimRigidAttachTGS) { runRigidAttach(/*isTGS*/ true); }

// ---------------------------------------------------------------------------
// P7.1 tranche 3c: cloth_applyExternalDeltasLaunch — folds the accumulated
// external correction (mDeltaPos, .w = per-vertex count) into position / velocity
// / accumulated-delta, then clears it.  The missing link that lets an attachment
// (or contact) actually move a cloth vertex.  Bit-exact vs FEMClothExternalSolve.cu:1053.
// ---------------------------------------------------------------------------
struct ApplyExtPush { uint64_t clothes, activeClothes; float dt; uint32_t pad20; };
static_assert(sizeof(ApplyExtPush) == 24, "applyExternalDeltas push blob must be 24 bytes");

PXG_TEST(femClothSimApplyExternalDeltas)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 3;
    const float dt = 1.0f / 60.0f;
    const float invDt = 1.0f / dt;

    std::vector<F4> positions = { {1.f,2.f,3.f,1.f}, {4.f,5.f,6.f,1.f}, {7.f,8.f,9.f,1.f} };
    std::vector<F4> vels = { {0.1f,0.2f,0.3f,0.f}, {-0.1f,-0.2f,-0.3f,0.f}, {0.05f,-0.05f,0.15f,0.f} };
    std::vector<F4> accum = { {0.5f,0.5f,0.5f,0.f}, {1.f,1.f,1.f,0.f}, {-1.f,-1.f,-1.f,0.f} };
    // vert0 count=2 (apply), vert1 count=0 (skip, must stay untouched), vert2 count=3 (apply)
    std::vector<F4> deltaPos = { {0.3f,-0.6f,0.9f,2.0f}, {9.f,9.f,9.f,0.0f}, {1.5f,0.3f,-0.9f,3.0f} };

    DevBuf bPos = makeDev(dev, positions);
    DevBuf bVel = makeDev(dev, vels);
    DevBuf bAcc = makeDev(dev, accum);
    DevBuf bDelta = makeDev(dev, deltaPos);

    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bPos.addr));
    c.mVelocity_InvMass = reinterpret_cast<decltype(c.mVelocity_InvMass)>(size_t(bVel.addr));
    c.mAccumulatedDeltaPos = reinterpret_cast<decltype(c.mAccumulatedDeltaPos)>(size_t(bAcc.addr));
    c.mDeltaPos = reinterpret_cast<decltype(c.mDeltaPos)>(size_t(bDelta.addr));
    c.mNbVerts = N;

    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    ApplyExtPush push = {};
    push.clothes = bDesc.addr;
    push.activeClothes = bActive.addr;
    push.dt = dt;

    Kernel k("femClothRigidAttachBda.spv", "cloth_applyExternalDeltasLaunch", {}, sizeof(ApplyExtPush));
    k.dispatch(/*numBlocks*/ 1u, {}, push, /*nbActive*/ 1u, 1u);
    dev->flush();

    // CPU reference (FEMClothExternalSolve.cu:1053).
    std::vector<F4> refPos = positions, refVel = vels, refAcc = accum, refDelta = deltaPos;
    for (uint32_t v = 0; v < N; ++v)
    {
        const F4 d = deltaPos[v];
        if (d.w != 0.0f)
        {
            const float scale = 1.0f / d.w;
            const F4 sd = { d.x * scale, d.y * scale, d.z * scale, 0.0f };
            refVel[v] = { vels[v].x + sd.x * invDt, vels[v].y + sd.y * invDt, vels[v].z + sd.z * invDt, vels[v].w + sd.w * invDt };
            refPos[v] = { positions[v].x + sd.x, positions[v].y + sd.y, positions[v].z + sd.z, positions[v].w + sd.w };
            refAcc[v] = { accum[v].x + sd.x, accum[v].y + sd.y, accum[v].z + sd.z, accum[v].w + sd.w };
            refDelta[v] = { 0.0f, 0.0f, 0.0f, 0.0f };
        }
    }

    const auto gPos = downloadBuffer<F4>(dev, bPos.h, N);
    const auto gVel = downloadBuffer<F4>(dev, bVel.h, N);
    const auto gAcc = downloadBuffer<F4>(dev, bAcc.h, N);
    const auto gDelta = downloadBuffer<F4>(dev, bDelta.h, N);

    PXG_CHECK_MSG(f2u(gPos[0].x) != f2u(positions[0].x), "applyExternalDeltas should move vert0 (teeth)");
    PXG_CHECK_MSG(f2u(gPos[1].x) == f2u(positions[1].x), "vert1 (count 0) must stay untouched");

    size_t mm = 0;
    auto cmp4 = [&](const char* nm, uint32_t i, const F4& r, const F4& g) {
        if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) || f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
        { ++mm; std::printf("  %s[%u] MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", nm, i, r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w); }
    };
    for (uint32_t v = 0; v < N; ++v)
    {
        cmp4("pos", v, refPos[v], gPos[v]);
        cmp4("vel", v, refVel[v], gVel[v]);
        cmp4("acc", v, refAcc[v], gAcc[v]);
        cmp4("delta", v, refDelta[v], gDelta[v]);
    }
    PXG_CHECK_MSG(mm == 0, "cloth_applyExternalDeltas bit-exact vs CPU (%zu mismatches)", mm);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bAcc.h);
    dev->destroyBuffer(bDelta.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

// ===========================================================================
// P7.1 tranche 4a: cloth-vertex-vs-PRIMITIVE contact GENERATION.
//   femClothContactGenBda.spv:
//     cloth_midphaseGenerateVertexPairsLaunch  (femClothMidPhase.cu)
//     cloth_planeVertContactGenLaunch          (femClothPrimitives.cu:853)
//     cloth_SphereVertexContactGenLaunch       (femClothPrimitives.cu:700)
// Bit-exact vs a CPU transcription of the .cu math (both memory modes).
// ===========================================================================

// PxgFemOtherContactInfo host mirror (PxgFEMCore.h:128, PX_ALIGN(16), 16 bytes).
// PxgFEMCore.h is not test-includable (it pulls the CUDA buffer stack), so the
// contact-writer's info tail is mirrored locally and pinned self-consistently
// here (the layout the shader writes at index*16 must equal what we read back).
struct FemOtherContactInfoHost
{
    uint64_t pairInd0;                 // @0
    uint32_t pairInd1;                 // @8
    uint32_t rigidMatInd_isInCollision; // @12 -> 16
};
static_assert(sizeof(FemOtherContactInfoHost) == 16, "PxgFemOtherContactInfo mirror == 16B");
static_assert(offsetof(FemOtherContactInfoHost, pairInd0) == 0, "pairInd0 @0");
static_assert(offsetof(FemOtherContactInfoHost, pairInd1) == 8, "pairInd1 @8");
static_assert(offsetof(FemOtherContactInfoHost, rigidMatInd_isInCollision) == 12, "rigidMatInd @12");

// midphase push blob — byte-identical to MidphaseParams (ABI row {4,4,8,8,8,8,8,8,8,8}).
struct MidphasePush
{
    uint32_t numTests;
    float toleranceLength;
    uint64_t cmInputs, transformCache, bounds, contactDistance, gpuShapes, femClothes;
    uint64_t stackPtr, midphasePairsNum;
};
static_assert(sizeof(MidphasePush) == 72, "midphase push blob must be 72 bytes");

// plane / sphere contact-gen push blob — byte-identical to ContactGenParams
// (ABI row {4,8,8,8,8,8,8,8,8,4,80}).  The PxgFEMContactWriter (80B) inlined @80.
struct ContactGenPush
{
    uint32_t numTests, pad4;
    uint64_t cmInputs, transformCache, contactDistance, restDistance, gpuShapes, femClothes;
    uint64_t shapeToRigidRemap, filterPairs;
    uint32_t nbFilterPairs, pad76;
    uint64_t w_outPoint, w_outNormalPen, w_outBarycentric, w_outContactInfo, w_totalContactCount;
    uint64_t w_contactByX, w_tempContactByX, w_contactIndexSortedByX, w_contactSortedByX;
    uint32_t w_maxNumContacts, pad156;
};
static_assert(sizeof(ContactGenPush) == 160, "contact-gen push blob must be 160 bytes");
static_assert(offsetof(ContactGenPush, w_outPoint) == 80, "writer @80");
static_assert(offsetof(ContactGenPush, w_maxNumContacts) == 152, "maxNumContacts @152");

// PxQuat::rotateInv / getBasisVector0 CPU transcriptions (bit-exact; the test TU
// is built -ffp-contract=off, matching the NoContraction shader stamp).
struct V3c { float x, y, z; };
static V3c cpuQuatRotateInv(const float q[4], V3c v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q[3] * q[3] - 0.5f;
    const float dot2 = (q[0] * vx + q[1] * vy + q[2] * vz);
    return V3c{ vx * w2 - (q[1] * vz - q[2] * vy) * q[3] + q[0] * dot2,
                vy * w2 - (q[2] * vx - q[0] * vz) * q[3] + q[1] * dot2,
                vz * w2 - (q[0] * vy - q[1] * vx) * q[3] + q[2] * dot2 };
}
static V3c cpuGetBasisVector0(const float q[4])
{
    const float x2 = q[0] * 2.0f, w2 = q[3] * 2.0f;
    return V3c{ (q[3] * w2) - 1.0f + q[0] * x2, (q[2] * w2) + q[1] * x2, (-q[1] * w2) + q[2] * x2 };
}

// ---------------------------------------------------------------------------
// Descriptor / layout proof for the narrowphase mirrors the contact-gen kernels
// dereference (byte-exact vs the real engine structs).
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimContactGenDescLayout)
{
    static_assert(sizeof(PxgContactManagerInput) == 16, "PxgContactManagerInput stride");
    static_assert(sizeof(PxgShape) == 48, "PxgShape stride");
    static_assert(offsetof(PxgShape, type) == 40, "PxgShape::type @40");
    static_assert(offsetof(PxgShape, particleOrSoftbodyId) == 44, "PxgShape::particleOrSoftbodyId @44");
    static_assert(offsetof(PxgShape, materialIndex) == 28, "PxgShape::materialIndex @28");
    static_assert(sizeof(PxsCachedTransform) == 32, "PxsCachedTransform stride");
    static_assert(sizeof(PxgRigidFilterPair) == 32, "PxgRigidFilterPair stride");

    PXG_CHECK_MSG(sizeof(PxBounds3) == 24, "PxBounds3 sizeof == 24 (got %zu)", sizeof(PxBounds3));
    PXG_CHECK(offsetof(PxBounds3, minimum) == 0);
    PXG_CHECK(offsetof(PxBounds3, maximum) == 12);

    // PxGeometryType values the kernels key on.
    PXG_CHECK(uint32_t(PxGeometryType::eSPHERE) == 0u);
    PXG_CHECK(uint32_t(PxGeometryType::ePLANE) == 1u);
    PXG_CHECK(uint32_t(PxGeometryType::eTRIANGLEMESH) == 8u);
    // PxNodeIndex round-trip: contactByRigid low-32 == getInd() & 0xffffffff.
    PXG_CHECK(sizeof(PxNodeIndex) == 8);
}

// Build a single-cloth descriptor referencing a position buffer + nbVerts.
static PxgFEMCloth makeClothDesc(uint64_t posAddr, uint32_t nbVerts)
{
    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(posAddr));
    c.mNbVerts = nbVerts;
    return c;
}

// ---------------------------------------------------------------------------
// cloth_midphaseGenerateVertexPairsLaunch — vertices vs the rigid world AABB.
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimMidphaseVertexPairs)
{
    PxgSlangComputeDevice* dev = testDevice();

    // rigid AABB [-1,1]^3 at cache ref 1; total contactDist 0.1 -> tolerance 0.01.
    // vert0 inside (hit), vert1 far (miss), vert2 0.05 outside (hit), vert3 far (miss).
    std::vector<F4> verts = {
        { 0.00f,  0.00f, 0.00f, 1.0f },   // id0 HIT (magSq 0)
        { 5.00f,  0.00f, 0.00f, 1.0f },   // id1 MISS (magSq 16)
        { 1.05f,  0.00f, 0.00f, 1.0f },   // id2 HIT (magSq 0.0025 <= 0.01)
        { 0.00f, -5.00f, 0.00f, 1.0f },   // id3 MISS (magSq 16)
    };
    const uint32_t nbVerts = uint32_t(verts.size());
    DevBuf bVerts = makeDev(dev, verts);

    PxgFEMCloth c = makeClothDesc(bVerts.addr, nbVerts);
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<PxgContactManagerInput> cmInputs(1);
    std::memset(cmInputs.data(), 0, sizeof(PxgContactManagerInput));
    cmInputs[0].shapeRef0 = 0u; cmInputs[0].shapeRef1 = 1u;
    cmInputs[0].transformCacheRef0 = 0u; cmInputs[0].transformCacheRef1 = 1u;
    DevBuf bCin = makeDev(dev, cmInputs);

    std::vector<PxgShape> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    shapes[0].type = PxU32(PxGeometryType::eTRIANGLEMESH); // cloth
    shapes[0].particleOrSoftbodyId = 0u;
    shapes[1].type = PxU32(PxGeometryType::ePLANE);        // rigid (any non-mesh)
    DevBuf bShapes = makeDev(dev, shapes);

    // bounds: [1] = the rigid AABB [-1,1]^3 (min @0, max @12).
    std::vector<float> boundsF(2 * 6, 0.0f);
    boundsF[6 + 0] = -1.0f; boundsF[6 + 1] = -1.0f; boundsF[6 + 2] = -1.0f;
    boundsF[6 + 3] =  1.0f; boundsF[6 + 4] =  1.0f; boundsF[6 + 5] =  1.0f;
    DevBuf bBounds = makeDev(dev, boundsF);

    std::vector<float> contactDist = { 0.05f, 0.05f }; // total 0.1 -> tolerance 0.01
    DevBuf bCd = makeDev(dev, contactDist);
    std::vector<PxsCachedTransform> tcache(2);
    std::memset(tcache.data(), 0, tcache.size() * sizeof(PxsCachedTransform));
    DevBuf bTc = makeDev(dev, tcache); // unused by the vertex midphase (ABI parity)

    const uint32_t STACK_MAX = 64u;
    std::vector<uint32_t> stack(STACK_MAX * 4, 0xffffffffu); // uint4[]
    std::vector<uint32_t> pairsNum = { 0u };
    DevBuf bStack = makeDev(dev, stack);
    DevBuf bPairsNum = makeDev(dev, pairsNum);

    MidphasePush push = {};
    push.numTests = 1u;
    push.toleranceLength = 1.0f;
    push.cmInputs = bCin.addr;
    push.transformCache = bTc.addr;
    push.bounds = bBounds.addr;
    push.contactDistance = bCd.addr;
    push.gpuShapes = bShapes.addr;
    push.femClothes = bDesc.addr;
    push.stackPtr = bStack.addr;
    push.midphasePairsNum = bPairsNum.addr;

    // engine launch geometry: grid (32, numTests, 1), block (32, 2, 1).
    Kernel k("femClothContactGenBda.spv", "cloth_midphaseGenerateVertexPairsLaunch", {}, sizeof(MidphasePush));
    k.dispatch(/*gridX*/ 32u, {}, push, /*gridY numTests*/ 1u, 1u);
    dev->flush();

    const uint32_t gNum = downloadBuffer<uint32_t>(dev, bPairsNum.h, 1)[0];
    PXG_CHECK_MSG(gNum == 2u, "midphase emitted %u pairs (expected 2)", gNum);

    const auto gStack = downloadBuffer<uint32_t>(dev, bStack.h, STACK_MAX * 4);
    // Single warp -> deterministic order: pairs written in vertex-id order.
    // pair0 = (cmIdx 0, id 0), pair1 = (cmIdx 0, id 2).
    size_t mm = 0;
    auto chkPair = [&](uint32_t slot, uint32_t cm, uint32_t id) {
        const uint32_t gx = gStack[slot * 4 + 0], gy = gStack[slot * 4 + 1];
        const uint32_t gz = gStack[slot * 4 + 2], gw = gStack[slot * 4 + 3];
        if (gx != cm || gy != id || gz != 0u || gw != 0u)
        { ++mm; std::printf("  midphase pair[%u] MM: got(%u,%u,%u,%u) ref(%u,%u,0,0)\n", slot, gx, gy, gz, gw, cm, id); }
    };
    chkPair(0u, 0u, 0u);
    chkPair(1u, 0u, 2u);
    PXG_CHECK_MSG(mm == 0, "cloth_midphaseGenerateVertexPairs bit-exact vs CPU (%zu mismatches)", mm);

    // ---- non-penetrating case: all verts far outside -> zero pairs ----
    std::vector<F4> vertsOut = {
        { 10.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 10.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 10.0f, 1.0f },
    };
    DevBuf bVertsOut = makeDev(dev, vertsOut);
    PxgFEMCloth c2 = makeClothDesc(bVertsOut.addr, uint32_t(vertsOut.size()));
    std::vector<PxgFEMCloth> descArr2 = { c2 };
    DevBuf bDesc2 = makeDev(dev, descArr2);
    std::vector<uint32_t> pairsNum2 = { 0u };
    DevBuf bPairsNum2 = makeDev(dev, pairsNum2);
    MidphasePush push2 = push;
    push2.femClothes = bDesc2.addr;
    push2.midphasePairsNum = bPairsNum2.addr;
    k.dispatch(32u, {}, push2, 1u, 1u);
    dev->flush();
    const uint32_t gNum2 = downloadBuffer<uint32_t>(dev, bPairsNum2.h, 1)[0];
    PXG_CHECK_MSG(gNum2 == 0u, "midphase (no penetration) emitted %u pairs (expected 0)", gNum2);

    dev->destroyBuffer(bVerts.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bCin.h);
    dev->destroyBuffer(bShapes.h); dev->destroyBuffer(bBounds.h); dev->destroyBuffer(bCd.h);
    dev->destroyBuffer(bTc.h); dev->destroyBuffer(bStack.h); dev->destroyBuffer(bPairsNum.h);
    dev->destroyBuffer(bVertsOut.h); dev->destroyBuffer(bDesc2.h); dev->destroyBuffer(bPairsNum2.h);
}

// Shared writer-buffer set for the plane / sphere contact-gen tests.
struct WriterBufs
{
    DevBuf point, normalPen, barycentric, info, count, byRigid, tempByRigid, idxByRigid, sortedByRigid;
    uint32_t maxContacts;
};
static WriterBufs makeWriter(PxgSlangComputeDevice* dev, uint32_t maxContacts)
{
    WriterBufs w; w.maxContacts = maxContacts;
    std::vector<F4> zf4(maxContacts, F4{ 0, 0, 0, 0 });
    std::vector<FemOtherContactInfoHost> zinfo(maxContacts);
    std::memset(zinfo.data(), 0, zinfo.size() * sizeof(FemOtherContactInfoHost));
    std::vector<uint32_t> zc = { 0u };
    std::vector<uint64_t> zu64(maxContacts, 0xffffffffffffffffull);
    std::vector<uint32_t> zu32(maxContacts, 0xffffffffu);
    w.point = makeDev(dev, zf4); w.normalPen = makeDev(dev, zf4); w.barycentric = makeDev(dev, zf4);
    w.info = makeDev(dev, zinfo); w.count = makeDev(dev, zc);
    w.byRigid = makeDev(dev, zu64); w.tempByRigid = makeDev(dev, zu32);
    w.idxByRigid = makeDev(dev, zu32); w.sortedByRigid = makeDev(dev, zu32);
    return w;
}
static void fillWriter(ContactGenPush& p, const WriterBufs& w)
{
    p.w_outPoint = w.point.addr; p.w_outNormalPen = w.normalPen.addr; p.w_outBarycentric = w.barycentric.addr;
    p.w_outContactInfo = w.info.addr; p.w_totalContactCount = w.count.addr;
    p.w_contactByX = w.byRigid.addr; p.w_tempContactByX = w.tempByRigid.addr;
    p.w_contactIndexSortedByX = w.idxByRigid.addr; p.w_contactSortedByX = w.sortedByRigid.addr;
    p.w_maxNumContacts = w.maxContacts;
}
static void destroyWriter(PxgSlangComputeDevice* dev, const WriterBufs& w)
{
    dev->destroyBuffer(w.point.h); dev->destroyBuffer(w.normalPen.h); dev->destroyBuffer(w.barycentric.h);
    dev->destroyBuffer(w.info.h); dev->destroyBuffer(w.count.h); dev->destroyBuffer(w.byRigid.h);
    dev->destroyBuffer(w.tempByRigid.h); dev->destroyBuffer(w.idxByRigid.h); dev->destroyBuffer(w.sortedByRigid.h);
}

// ---------------------------------------------------------------------------
// cloth_planeVertContactGenLaunch — one cloth vertex penetrating a plane.
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimPlaneVertContactGen)
{
    PxgSlangComputeDevice* dev = testDevice();

    // plane: rotate the canonical +X normal onto +Y (q = +90deg about z), at origin.
    const float sq = std::sin(0.78539816339744830961f);
    const float cq = std::cos(0.78539816339744830961f);
    const float planeQ[4] = { 0.0f, 0.0f, sq, cq };
    const float planeP[3] = { 0.0f, 0.0f, 0.0f };
    const float contactOffsetCloth = 0.2f, contactOffsetRigid = 0.1f; // total 0.3
    const float contactDist = contactOffsetCloth + contactOffsetRigid;
    const float restDistance = 0.05f;
    const uint32_t materialIndex = 5u;
    const uint64_t rigidNodeInd = 0x0000000200000007ull; // low32 = 7

    // vert0 penetrating (below plane), vert1/vert2 above (no contact).
    std::vector<F4> verts = {
        { 0.5f, -0.1f, 0.3f, 1.0f },   // HIT (pInPlane.x ~ -0.1 <= 0.3)
        { 0.0f,  1.0f, 0.0f, 1.0f },   // MISS
        { -0.3f, 2.0f, 0.1f, 1.0f },   // MISS
    };
    const uint32_t nbVerts = uint32_t(verts.size());
    DevBuf bVerts = makeDev(dev, verts);
    PxgFEMCloth c = makeClothDesc(bVerts.addr, nbVerts);
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<PxgContactManagerInput> cmInputs(1);
    std::memset(cmInputs.data(), 0, sizeof(PxgContactManagerInput));
    cmInputs[0].shapeRef0 = 0u; cmInputs[0].shapeRef1 = 1u;
    cmInputs[0].transformCacheRef0 = 0u; cmInputs[0].transformCacheRef1 = 1u;
    DevBuf bCin = makeDev(dev, cmInputs);

    std::vector<PxgShape> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    shapes[0].type = PxU32(PxGeometryType::eTRIANGLEMESH);
    shapes[0].particleOrSoftbodyId = 0u;
    shapes[1].type = PxU32(PxGeometryType::ePLANE);
    shapes[1].materialIndex = materialIndex;
    DevBuf bShapes = makeDev(dev, shapes);

    std::vector<PxsCachedTransform> tcache(2);
    std::memset(tcache.data(), 0, tcache.size() * sizeof(PxsCachedTransform));
    tcache[1].transform = PxTransform(PxVec3(planeP[0], planeP[1], planeP[2]),
                                      PxQuat(planeQ[0], planeQ[1], planeQ[2], planeQ[3]));
    DevBuf bTc = makeDev(dev, tcache);

    std::vector<float> contactDistArr = { contactOffsetCloth, contactOffsetRigid };
    DevBuf bCd = makeDev(dev, contactDistArr);
    std::vector<float> restArr = { restDistance };
    DevBuf bRd = makeDev(dev, restArr);
    std::vector<uint64_t> remap = { 0ull, rigidNodeInd };
    DevBuf bRemap = makeDev(dev, remap);
    DevBuf bFilters = makeDev(dev, std::vector<uint32_t>{ 0u }); // dummy buffer (nbFilterPairs=0 -> find() false)

    WriterBufs w = makeWriter(dev, 64u);

    ContactGenPush push = {};
    push.numTests = 1u;
    push.cmInputs = bCin.addr; push.transformCache = bTc.addr; push.contactDistance = bCd.addr;
    push.restDistance = bRd.addr; push.gpuShapes = bShapes.addr; push.femClothes = bDesc.addr;
    push.shapeToRigidRemap = bRemap.addr; push.filterPairs = bFilters.addr; push.nbFilterPairs = 0u;
    fillWriter(push, w);

    Kernel k("femClothContactGenBda.spv", "cloth_planeVertContactGenLaunch", {}, sizeof(ContactGenPush));
    k.dispatch(/*gridX numTests*/ 1u, {}, push, 1u, 1u);
    dev->flush();

    // ---- CPU reference for vert0 ----
    const V3c n = cpuGetBasisVector0(planeQ);
    const V3c pin = cpuQuatRotateInv(planeQ, V3c{ verts[0].x - planeP[0], verts[0].y - planeP[1], verts[0].z - planeP[2] });
    PXG_CHECK_MSG(contactDist >= pin.x, "reference: vert0 must be within the plane contact offset");
    const F4 refNormalPen = { n.x, n.y, n.z, pin.x };
    const float rpx = verts[0].x - pin.x * n.x, rpy = verts[0].y - pin.x * n.y, rpz = verts[0].z - pin.x * n.z;
    const F4 refPoint = { rpx, rpy, rpz, restDistance };
    const uint32_t clothMask0 = (0u << 20) | 0u;

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, w.count.h, 1)[0];
    PXG_CHECK_MSG(gCount == 1u, "plane contact-gen emitted %u contacts (expected 1)", gCount);

    const F4 gPoint = downloadBuffer<F4>(dev, w.point.h, 1)[0];
    const F4 gNP = downloadBuffer<F4>(dev, w.normalPen.h, 1)[0];
    const F4 gBary = downloadBuffer<F4>(dev, w.barycentric.h, 1)[0];
    const FemOtherContactInfoHost gInfo = downloadBuffer<FemOtherContactInfoHost>(dev, w.info.h, 1)[0];
    const uint64_t gByRigid = downloadBuffer<uint64_t>(dev, w.byRigid.h, 1)[0];
    const uint32_t gTempByRigid = downloadBuffer<uint32_t>(dev, w.tempByRigid.h, 1)[0];
    const uint32_t gIdxByRigid = downloadBuffer<uint32_t>(dev, w.idxByRigid.h, 1)[0];

    size_t mm = 0;
    auto cmpF4 = [&](const char* nm, const F4& r, const F4& g) {
        if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) || f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
        { ++mm; std::printf("  plane %s MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", nm, r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w); }
    };
    cmpF4("outPoint", refPoint, gPoint);
    cmpF4("outNormalPen", refNormalPen, gNP);
    cmpF4("outBarycentric", F4{ 0, 0, 0, 1 }, gBary);
    if (gByRigid != rigidNodeInd) { ++mm; std::printf("  plane contactByRigid MM got %llu ref %llu\n", (unsigned long long)gByRigid, (unsigned long long)rigidNodeInd); }
    if (gTempByRigid != uint32_t(rigidNodeInd & 0xffffffffu)) { ++mm; std::printf("  plane tempByRigid MM got %u\n", gTempByRigid); }
    if (gIdxByRigid != 0u) { ++mm; std::printf("  plane contactIndexSortedByRigid MM got %u\n", gIdxByRigid); }
    if (gInfo.pairInd0 != rigidNodeInd) { ++mm; std::printf("  plane info.pairInd0 MM got %llu\n", (unsigned long long)gInfo.pairInd0); }
    if (gInfo.pairInd1 != clothMask0) { ++mm; std::printf("  plane info.pairInd1 MM got %u ref %u\n", gInfo.pairInd1, clothMask0); }
    if (gInfo.rigidMatInd_isInCollision != (materialIndex & 0x7fffffffu)) { ++mm; std::printf("  plane info.rigidMatInd MM got %u\n", gInfo.rigidMatInd_isInCollision); }
    PXG_CHECK_MSG(mm == 0, "cloth_planeVertContactGen bit-exact vs CPU (%zu mismatches)", mm);

    // teeth: a wrong normal would not satisfy this human-readable sanity.
    PXG_CHECK_MSG(almostEqual(gNP.x, 0.0f, 1e-5f, 1e-6f) && almostEqual(gNP.y, 1.0f, 1e-5f, 1e-6f) &&
                  almostEqual(gNP.z, 0.0f, 1e-5f, 1e-6f) && almostEqual(gNP.w, -0.1f, 1e-5f, 1e-6f),
                  "plane contact normal ~ (0,1,0), separation ~ -0.1");

    destroyWriter(dev, w);
    dev->destroyBuffer(bVerts.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bCin.h);
    dev->destroyBuffer(bShapes.h); dev->destroyBuffer(bTc.h); dev->destroyBuffer(bCd.h);
    dev->destroyBuffer(bRd.h); dev->destroyBuffer(bRemap.h); dev->destroyBuffer(bFilters.h);
}

// non-penetrating plane case: all verts above the plane -> zero contacts.
PXG_TEST(femClothSimPlaneVertContactGenNoContact)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float sq = std::sin(0.78539816339744830961f), cq = std::cos(0.78539816339744830961f);
    const float planeQ[4] = { 0.0f, 0.0f, sq, cq };

    std::vector<F4> verts = { { 0.0f, 1.0f, 0.0f, 1.0f }, { 0.0f, 2.0f, 0.0f, 1.0f } }; // all above
    DevBuf bVerts = makeDev(dev, verts);
    PxgFEMCloth c = makeClothDesc(bVerts.addr, uint32_t(verts.size()));
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<PxgContactManagerInput> cmInputs(1);
    std::memset(cmInputs.data(), 0, sizeof(PxgContactManagerInput));
    cmInputs[0].shapeRef1 = 1u; cmInputs[0].transformCacheRef1 = 1u;
    DevBuf bCin = makeDev(dev, cmInputs);
    std::vector<PxgShape> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    shapes[0].type = PxU32(PxGeometryType::eTRIANGLEMESH);
    shapes[1].type = PxU32(PxGeometryType::ePLANE);
    DevBuf bShapes = makeDev(dev, shapes);
    std::vector<PxsCachedTransform> tcache(2);
    std::memset(tcache.data(), 0, tcache.size() * sizeof(PxsCachedTransform));
    tcache[1].transform = PxTransform(PxVec3(0, 0, 0), PxQuat(planeQ[0], planeQ[1], planeQ[2], planeQ[3]));
    DevBuf bTc = makeDev(dev, tcache);
    std::vector<float> contactDistArr = { 0.2f, 0.1f };
    DevBuf bCd = makeDev(dev, contactDistArr);
    std::vector<float> restArr = { 0.0f };
    DevBuf bRd = makeDev(dev, restArr);
    std::vector<uint64_t> remap = { 0ull, 7ull };
    DevBuf bRemap = makeDev(dev, remap);
    DevBuf bFilters = makeDev(dev, std::vector<uint32_t>{ 0u });
    WriterBufs w = makeWriter(dev, 64u);

    ContactGenPush push = {};
    push.numTests = 1u;
    push.cmInputs = bCin.addr; push.transformCache = bTc.addr; push.contactDistance = bCd.addr;
    push.restDistance = bRd.addr; push.gpuShapes = bShapes.addr; push.femClothes = bDesc.addr;
    push.shapeToRigidRemap = bRemap.addr; push.filterPairs = bFilters.addr; push.nbFilterPairs = 0u;
    fillWriter(push, w);

    Kernel k("femClothContactGenBda.spv", "cloth_planeVertContactGenLaunch", {}, sizeof(ContactGenPush));
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, w.count.h, 1)[0];
    PXG_CHECK_MSG(gCount == 0u, "plane contact-gen (no penetration) emitted %u contacts (expected 0)", gCount);

    destroyWriter(dev, w);
    dev->destroyBuffer(bVerts.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bCin.h);
    dev->destroyBuffer(bShapes.h); dev->destroyBuffer(bTc.h); dev->destroyBuffer(bCd.h);
    dev->destroyBuffer(bRd.h); dev->destroyBuffer(bRemap.h); dev->destroyBuffer(bFilters.h);
}

// ---------------------------------------------------------------------------
// cloth_SphereVertexContactGenLaunch — one cloth vertex penetrating a sphere.
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimSphereVertContactGen)
{
    PxgSlangComputeDevice* dev = testDevice();

    const float sphereRadius = 1.0f;
    const float contactOffsetCloth = 0.05f, contactOffsetRigid = 0.05f; // total 0.1
    const float contactDist = contactOffsetCloth + contactOffsetRigid;
    const float restDistance = 0.03f;
    const uint32_t materialIndex = 9u;
    const uint64_t rigidNodeInd = 0x0000000300000011ull; // low32 = 0x11
    const float sphereCenter[3] = { 0.0f, 0.0f, 0.0f };

    // vert0 inside the sphere (hit), vert1/vert2 outside (miss).
    std::vector<F4> verts = {
        { 0.5f, 0.0f, 0.0f, 1.0f },   // HIT (dist 0.5 < 1.1)
        { 3.0f, 0.0f, 0.0f, 1.0f },   // MISS
        { 0.0f, 2.0f, 0.0f, 1.0f },   // MISS
    };
    const uint32_t nbVerts = uint32_t(verts.size());
    DevBuf bVerts = makeDev(dev, verts);
    PxgFEMCloth c = makeClothDesc(bVerts.addr, nbVerts);
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<PxgContactManagerInput> cmInputs(1);
    std::memset(cmInputs.data(), 0, sizeof(PxgContactManagerInput));
    cmInputs[0].shapeRef0 = 0u; cmInputs[0].shapeRef1 = 1u;
    cmInputs[0].transformCacheRef0 = 0u; cmInputs[0].transformCacheRef1 = 1u;
    DevBuf bCin = makeDev(dev, cmInputs);

    std::vector<PxgShape> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    shapes[0].type = PxU32(PxGeometryType::eTRIANGLEMESH);
    shapes[0].particleOrSoftbodyId = 0u;
    shapes[1].type = PxU32(PxGeometryType::eSPHERE);
    shapes[1].materialIndex = materialIndex;
    shapes[1].scale.scale = PxVec3(sphereRadius, sphereRadius, sphereRadius); // radius = scale.y
    DevBuf bShapes = makeDev(dev, shapes);

    std::vector<PxsCachedTransform> tcache(2);
    std::memset(tcache.data(), 0, tcache.size() * sizeof(PxsCachedTransform));
    tcache[1].transform = PxTransform(PxVec3(sphereCenter[0], sphereCenter[1], sphereCenter[2]), PxQuat(PxIdentity));
    DevBuf bTc = makeDev(dev, tcache);

    std::vector<float> contactDistArr = { contactOffsetCloth, contactOffsetRigid };
    DevBuf bCd = makeDev(dev, contactDistArr);
    std::vector<float> restArr = { restDistance };
    DevBuf bRd = makeDev(dev, restArr);
    std::vector<uint64_t> remap = { 0ull, rigidNodeInd };
    DevBuf bRemap = makeDev(dev, remap);
    DevBuf bFilters = makeDev(dev, std::vector<uint32_t>{ 0u });
    WriterBufs w = makeWriter(dev, 64u);

    ContactGenPush push = {};
    push.numTests = 1u;
    push.cmInputs = bCin.addr; push.transformCache = bTc.addr; push.contactDistance = bCd.addr;
    push.restDistance = bRd.addr; push.gpuShapes = bShapes.addr; push.femClothes = bDesc.addr;
    push.shapeToRigidRemap = bRemap.addr; push.filterPairs = bFilters.addr; push.nbFilterPairs = 0u;
    fillWriter(push, w);

    Kernel k("femClothContactGenBda.spv", "cloth_SphereVertexContactGenLaunch", {}, sizeof(ContactGenPush));
    k.dispatch(/*gridX numTests*/ 1u, {}, push, 1u, 1u);
    dev->flush();

    // ---- CPU reference (spheresphere) for vert0 ----
    const float dx = verts[0].x - sphereCenter[0], dy = verts[0].y - sphereCenter[1], dz = verts[0].z - sphereCenter[2];
    const float distanceSq = dx * dx + dy * dy + dz * dz;
    const float radiusSum = sphereRadius; // 0 + radius
    const float inflatedSum = radiusSum + contactDist;
    PXG_CHECK_MSG((inflatedSum * inflatedSum) > distanceSq, "reference: vert0 must penetrate the inflated sphere");
    const float dist = std::sqrt(distanceSq);
    V3c refN = (1e-4f >= dist) ? V3c{ 1.0f, 0.0f, 0.0f } : V3c{ dx * (1.0f / dist), dy * (1.0f / dist), dz * (1.0f / dist) };
    const float refPen = dist - radiusSum;
    const F4 refNormalPen = { refN.x, refN.y, refN.z, refPen };
    const F4 refPoint = { verts[0].x, verts[0].y, verts[0].z, restDistance };
    const uint32_t clothMask0 = (0u << 20) | 0u;

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, w.count.h, 1)[0];
    PXG_CHECK_MSG(gCount == 1u, "sphere contact-gen emitted %u contacts (expected 1)", gCount);

    const F4 gPoint = downloadBuffer<F4>(dev, w.point.h, 1)[0];
    const F4 gNP = downloadBuffer<F4>(dev, w.normalPen.h, 1)[0];
    const F4 gBary = downloadBuffer<F4>(dev, w.barycentric.h, 1)[0];
    const FemOtherContactInfoHost gInfo = downloadBuffer<FemOtherContactInfoHost>(dev, w.info.h, 1)[0];
    const uint64_t gByRigid = downloadBuffer<uint64_t>(dev, w.byRigid.h, 1)[0];
    const uint32_t gTempByRigid = downloadBuffer<uint32_t>(dev, w.tempByRigid.h, 1)[0];

    size_t mm = 0;
    auto cmpF4 = [&](const char* nm, const F4& r, const F4& g) {
        if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) || f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
        { ++mm; std::printf("  sphere %s MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", nm, r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w); }
    };
    cmpF4("outPoint", refPoint, gPoint);
    cmpF4("outNormalPen", refNormalPen, gNP);
    cmpF4("outBarycentric", F4{ 0, 0, 0, 1 }, gBary);
    if (gByRigid != rigidNodeInd) { ++mm; std::printf("  sphere contactByRigid MM got %llu\n", (unsigned long long)gByRigid); }
    if (gTempByRigid != uint32_t(rigidNodeInd & 0xffffffffu)) { ++mm; std::printf("  sphere tempByRigid MM got %u\n", gTempByRigid); }
    if (gInfo.pairInd0 != rigidNodeInd) { ++mm; std::printf("  sphere info.pairInd0 MM got %llu\n", (unsigned long long)gInfo.pairInd0); }
    if (gInfo.pairInd1 != clothMask0) { ++mm; std::printf("  sphere info.pairInd1 MM got %u\n", gInfo.pairInd1); }
    if (gInfo.rigidMatInd_isInCollision != (materialIndex & 0x7fffffffu)) { ++mm; std::printf("  sphere info.rigidMatInd MM got %u\n", gInfo.rigidMatInd_isInCollision); }
    PXG_CHECK_MSG(mm == 0, "cloth_SphereVertexContactGen bit-exact vs CPU (%zu mismatches)", mm);

    PXG_CHECK_MSG(almostEqual(gNP.x, 1.0f, 1e-5f, 1e-6f) && almostEqual(gNP.w, -0.5f, 1e-5f, 1e-6f),
                  "sphere contact normal ~ (1,0,0), separation ~ -0.5");

    destroyWriter(dev, w);
    dev->destroyBuffer(bVerts.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bCin.h);
    dev->destroyBuffer(bShapes.h); dev->destroyBuffer(bTc.h); dev->destroyBuffer(bCd.h);
    dev->destroyBuffer(bRd.h); dev->destroyBuffer(bRemap.h); dev->destroyBuffer(bFilters.h);
}

// ---------------------------------------------------------------------------
// P9.15a -- cloth_boxVertexContactGenLaunch.
//
// THE ORACLE, STATED HONESTLY.  `spherebox` is __device__-only
// (sphereCollision.cuh:371), so there is no engine function to call: the mirror
// below is a CPU TRANSCRIPTION, compared bit for bit.  What carries it is the
// same arrangement P8.73-P8.80 used -- every offset generated and static_asserted,
// the whole writer region compared, and a heavy sweep -- plus one thing those had
// not: the SAME body is already swept in Phase 1 as fcStaticTriCloth's mode 1,
// and P9.15a LIFTED it into pxgSlangSphereBox rather than transcribing it again,
// so that battery keeps verifying the shader half of this comparison.
//
// WHAT THE GEOMETRY IS BUILT TO DISCRIMINATE
//   * a vertex strictly INSIDE the box, so the emitted point is the point on the
//     RIGID and not the cloth vertex (V1).  At pen == 0 the two coincide, so a
//     merely-touching vertex cannot tell the box kernel from its sphere sibling;
//   * a vertex on the box's DIAGONAL, where all three faces tie at once -- the
//     only place the `<=` chain and its Z seed are separable (SB1);
//   * a vertex OUTSIDE but within the contact distance, for the 1/rsqrt branch
//     (SB4), and one PAST it, which must not contact at all.  The gate's
//     STRICTNESS (SB5, `>` vs `>=`) is separable only at exact equality, which a
//     rotated round trip cannot hold -- that probe lives in the companion test
//     femClothSimBoxVertContactGenExactGate, on an axis-aligned box;
//   * a NON-IDENTITY box rotation, so quatRotate and quatRotateInv cannot be
//     swapped (SB6), and three DISTINCT half-extents so an axis mix-up shows (V4);
//   * a FILTER PAIR naming a vertex that would otherwise contact (V3);
//   * 700 vertices -- more than one 512-thread tile, and deliberately not a
//     multiple of either 32 or 512 -- with contacts in BOTH tiles, so the loop
//     bound is exercised past its first iteration.
// ---------------------------------------------------------------------------
struct BoxNP { float nx, ny, nz, pen; bool hit; };

// CPU mirror of spherebox (sphereCollision.cuh:371), radius 0.
static BoxNP cpuSphereBox(V3c sphereOrigin, const float q[4], V3c boxP,
                          float sphereRadius, V3c ext, float contactDist)
{
    BoxNP r = { 0, 0, 0, 0, false };
    const V3c c = cpuQuatRotateInv(q, V3c{ sphereOrigin.x - boxP.x, sphereOrigin.y - boxP.y,
                                           sphereOrigin.z - boxP.z });
    const float inflatedSum = sphereRadius + contactDist;
    const float sqInflatedSum = inflatedSum * inflatedSum;
    auto cl = [](float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); };
    const V3c p = { cl(c.x, -ext.x, ext.x), cl(c.y, -ext.y, ext.y), cl(c.z, -ext.z, ext.z) };
    const V3c v = { c.x - p.x, c.y - p.y, c.z - p.z };
    const float lengthSq = v.x * v.x + v.y * v.y + v.z * v.z;
    if (!(sqInflatedSum > lengthSq)) return r;      // SB5: strictly greater

    V3c localNorm; float dist;
    const bool inside = (ext.x >= std::fabs(c.x)) && (ext.y >= std::fabs(c.y)) && (ext.z >= std::fabs(c.z));
    if (inside)
    {
        const V3c absP = { std::fabs(p.x), std::fabs(p.y), std::fabs(p.z) };
        const V3c d = { ext.x - absP.x, ext.y - absP.y, ext.z - absP.z };
        localNorm = (p.z >= 0.0f) ? V3c{ 0, 0, 1 } : V3c{ 0, 0, -1 };
        dist = -d.z;                                 // SB3
        if (d.x <= d.y && d.x <= d.z)                // SB1
        { localNorm = (p.x >= 0.0f) ? V3c{ 1, 0, 0 } : V3c{ -1, 0, 0 }; dist = -d.x; }
        else if (d.y <= d.z)
        { localNorm = (p.y >= 0.0f) ? V3c{ 0, 1, 0 } : V3c{ 0, -1, 0 }; dist = -d.y; }
    }
    else
    {
        const float recipLength = 1.0f / std::sqrt(lengthSq);   // SB4: 1/rsqrt round trip
        const float len = 1.0f / recipLength;
        localNorm = V3c{ v.x * recipLength, v.y * recipLength, v.z * recipLength };
        dist = len;
    }
    // quatRotate (forward) -- pxgSlangCommon's spelling, operation for operation.
    const float vx = 2.0f * localNorm.x, vy = 2.0f * localNorm.y, vz = 2.0f * localNorm.z;
    const float w2 = q[3] * q[3] - 0.5f;
    const float dot2 = (q[0] * vx + q[1] * vy + q[2] * vz);
    r.nx = vx * w2 + (q[1] * vz - q[2] * vy) * q[3] + q[0] * dot2;
    r.ny = vy * w2 + (q[2] * vx - q[0] * vz) * q[3] + q[1] * dot2;
    r.nz = vz * w2 + (q[0] * vy - q[1] * vx) * q[3] + q[2] * dot2;
    r.pen = dist - sphereRadius;
    r.hit = true;
    return r;
}

PXG_TEST(femClothSimBoxVertContactGen)
{
    PxgSlangComputeDevice* dev = testDevice();

    // V4: three DISTINCT half-extents, none equal to any index, count or bound.
    const V3c ext = { 1.3f, 0.7f, 2.1f };
    const V3c boxP = { 0.4f, -0.25f, 0.6f };
    // SB6: a non-identity rotation (30 degrees about a non-axis direction).
    const float ax = 0.4082482905f, ay = 0.8164965809f, az = 0.4082482905f; // normalized (1,2,1)
    const float half = 0.2617993878f; // 15 deg -> 30 deg rotation
    const float sh = std::sin(half), ch = std::cos(half);
    const float boxQ[4] = { ax * sh, ay * sh, az * sh, ch };

    const float contactOffsetCloth = 0.18f, contactOffsetRigid = 0.07f;   // total 0.25
    const float contactDist = contactOffsetCloth + contactOffsetRigid;
    const float restDistance = 0.031f;         // V6: distinct from every contact offset
    const uint32_t materialIndex = 9u;
    const uint64_t rigidNodeInd = 0x0000000500000013ull;   // low32 = 19

    // Box-space anchor points, mapped into world so the oracle stays exact.
    auto toWorld = [&](V3c local) {
        // world = quatRotate(q, local) + boxP  (forward rotation, spelled as above)
        const float vx = 2.0f * local.x, vy = 2.0f * local.y, vz = 2.0f * local.z;
        const float w2 = boxQ[3] * boxQ[3] - 0.5f;
        const float dot2 = (boxQ[0] * vx + boxQ[1] * vy + boxQ[2] * vz);
        return V3c{ vx * w2 + (boxQ[1] * vz - boxQ[2] * vy) * boxQ[3] + boxQ[0] * dot2 + boxP.x,
                    vy * w2 + (boxQ[2] * vx - boxQ[0] * vz) * boxQ[3] + boxQ[1] * dot2 + boxP.y,
                    vz * w2 + (boxQ[0] * vy - boxQ[1] * vx) * boxQ[3] + boxQ[2] * dot2 + boxP.z };
    };

    const uint32_t NB_VERTS = 700u;            // > one 512-tile, multiple of neither 32 nor 512
    const uint32_t V_INSIDE_DEEP   = 3u;       // V1/SB3: strictly inside, nearest face +y
    const uint32_t V_DIAGONAL      = 11u;      // SB1: all three faces tie
    const uint32_t V_OUTSIDE_NEAR  = 40u;      // SB4: outside, within contactDist
    const uint32_t V_OUTSIDE_FAR   = 57u;      // just PAST the contact distance -> NO contact
    const uint32_t V_FILTERED      = 90u;      // V3: would contact, filter suppresses it
    const uint32_t V_SECOND_TILE   = 613u;     // in the SECOND 512-thread tile
    // SB4 needs more than one sample.  With a single outside vertex the mutant that
    // replaces `1.0f / rsqrt(lengthSq)` with `sqrt(lengthSq)` SURVIVED: the two
    // spellings round-trip to the same float for that one input.  They are not the
    // same function -- one rounds twice -- so the fix is more samples, at varied
    // distances and off corners and edges where lengthSq has two or three non-zero
    // components rather than one.
    static const uint32_t V_OUT_EXTRA[6] = { 101u, 137u, 208u, 259u, 331u, 404u };
    const V3c FAR_AWAY = { 900.0f, 900.0f, 900.0f };

    std::vector<F4> verts(NB_VERTS, F4{ FAR_AWAY.x, FAR_AWAY.y, FAR_AWAY.z, 1.0f });
    auto place = [&](uint32_t i, V3c local) {
        const V3c w = toWorld(local);
        verts[i] = F4{ w.x, w.y, w.z, 1.0f };
    };
    // strictly inside: nearest face is +y (dist to +y = 0.7-0.5 = 0.2; to x = 1.3-0.2 = 1.1;
    // to z = 2.1-0.3 = 1.8) -> localNorm (0,1,0), pen = -0.2
    place(V_INSIDE_DEEP,  V3c{ 0.2f, 0.5f, 0.3f });
    // on the diagonal of the INSCRIBED cube: distToSurface equal on all three axes
    // (ext - t = same) -> ext.x-t = ext.y-t = ext.z-t is impossible for distinct
    // extents, so tie the two SMALLEST by construction instead: pick t so that
    // ext.x-|x| == ext.y-|y| == ext.z-|z|.
    place(V_DIAGONAL,     V3c{ 1.3f - 0.4f, 0.7f - 0.4f, 2.1f - 0.4f });
    // outside the +x face by 0.15 (< 0.25), inside the other two spans
    place(V_OUTSIDE_NEAR, V3c{ ext.x + 0.15f, 0.1f, -0.2f });
    // outside the +x face by MORE than contactDist -> rejected.
    //
    // The first draft put this vertex EXACTLY at the inflated distance, to separate
    // the gate's strict `>` from `>=`.  It does not survive: the vertex is authored
    // in box space and pushed through a rotation to world, and the shader rotates it
    // back, so the round trip lands a few ulps inside and the mirror accepts it.
    // Same knife edge P9.14b hit with GJK.  The strictness probe therefore moved to
    // femClothSimBoxVertContactGenExactGate below, which uses an axis-aligned box at
    // the origin and binary-exact numbers so equality is reachable.
    place(V_OUTSIDE_FAR, V3c{ ext.x + contactDist + 0.05f, 0.0f, 0.0f });
    // would contact (inside), but the filter names it
    place(V_FILTERED,     V3c{ -0.3f, -0.2f, 0.4f });
    // second tile, strictly inside, nearest face -z
    place(V_SECOND_TILE,  V3c{ 0.1f, 0.05f, -1.9f });
    // SB4's extra samples: faces, edges and corners at varied separations.
    place(V_OUT_EXTRA[0], V3c{ ext.x + 0.07f,  0.30f,          -0.50f });          // +x face
    place(V_OUT_EXTRA[1], V3c{ -0.20f,         -ext.y - 0.11f,  0.80f });          // -y face
    place(V_OUT_EXTRA[2], V3c{ 0.60f,           0.10f,          ext.z + 0.19f });  // +z face
    place(V_OUT_EXTRA[3], V3c{ ext.x + 0.09f,   ext.y + 0.13f, -0.40f });          // xy edge
    place(V_OUT_EXTRA[4], V3c{ -ext.x - 0.06f,  0.20f,          ext.z + 0.08f });  // xz edge
    place(V_OUT_EXTRA[5], V3c{ ext.x + 0.05f,  -ext.y - 0.06f,  ext.z + 0.07f });  // corner

    DevBuf bVerts = makeDev(dev, verts);

    // THE CLOTH ID IS 1, NOT 0, AND SLOT 0 IS A DECOY.  With id 0 the index encode
    // `(femClothId << 20) | i` is the identity on i, so a wrong shift is invisible
    // and so is a wrong femClothes[] subscript.  Slot 0 therefore holds a real,
    // live descriptor over a DIFFERENT vertex buffer parked far from the box: a
    // port that reads it produces a deterministic empty set rather than a fault.
    std::vector<F4> decoyVerts(9u, F4{ -800.0f, -800.0f, -800.0f, 1.0f });
    DevBuf bDecoy = makeDev(dev, decoyVerts);
    const uint32_t CLOTH_ID = 1u;
    std::vector<PxgFEMCloth> descArr = { makeClothDesc(bDecoy.addr, uint32_t(decoyVerts.size())),
                                         makeClothDesc(bVerts.addr, NB_VERTS) };
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<PxgContactManagerInput> cmInputs(1);
    std::memset(cmInputs.data(), 0, sizeof(PxgContactManagerInput));
    cmInputs[0].shapeRef0 = 0u; cmInputs[0].shapeRef1 = 1u;
    cmInputs[0].transformCacheRef0 = 0u; cmInputs[0].transformCacheRef1 = 1u;
    DevBuf bCin = makeDev(dev, cmInputs);

    std::vector<PxgShape> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    shapes[0].type = PxU32(PxGeometryType::eTRIANGLEMESH);
    shapes[0].particleOrSoftbodyId = CLOTH_ID;
    shapes[1].type = PxU32(PxGeometryType::eBOX);
    shapes[1].materialIndex = materialIndex;
    shapes[1].scale = PxMeshScale(PxVec3(ext.x, ext.y, ext.z), PxQuat(PxIdentity));
    DevBuf bShapes = makeDev(dev, shapes);

    std::vector<PxsCachedTransform> tcache(2);
    std::memset(tcache.data(), 0, tcache.size() * sizeof(PxsCachedTransform));
    tcache[1].transform = PxTransform(PxVec3(boxP.x, boxP.y, boxP.z),
                                      PxQuat(boxQ[0], boxQ[1], boxQ[2], boxQ[3]));
    DevBuf bTc = makeDev(dev, tcache);

    std::vector<float> contactDistArr = { contactOffsetCloth, contactOffsetRigid };
    DevBuf bCd = makeDev(dev, contactDistArr);
    std::vector<float> restArr = { restDistance };
    DevBuf bRd = makeDev(dev, restArr);
    std::vector<uint64_t> remap = { 0ull, rigidNodeInd };
    DevBuf bRemap = makeDev(dev, remap);

    // V3: one filter pair, keyed (rigidNodeInd, clothMask(V_FILTERED)).
    const uint32_t filteredMask = (CLOTH_ID << 20) | V_FILTERED;
    std::vector<PxgRigidFilterPair> fpairs(1);
    std::memset(fpairs.data(), 0, sizeof(PxgRigidFilterPair));
    fpairs[0].index0 = rigidNodeInd;
    fpairs[0].index1 = uint64_t(filteredMask);
    DevBuf bFilters = makeDev(dev, fpairs);

    WriterBufs w = makeWriter(dev, 64u);

    ContactGenPush push = {};
    push.numTests = 1u;
    push.cmInputs = bCin.addr; push.transformCache = bTc.addr; push.contactDistance = bCd.addr;
    push.restDistance = bRd.addr; push.gpuShapes = bShapes.addr; push.femClothes = bDesc.addr;
    push.shapeToRigidRemap = bRemap.addr; push.filterPairs = bFilters.addr; push.nbFilterPairs = 1u;
    fillWriter(push, w);

    Kernel k("femClothContactGenBda.spv", "cloth_boxVertexContactGenLaunch", {}, sizeof(ContactGenPush));
    PXG_CHECK_MSG(k.isValid(), "the box vertex contact-gen pipeline must build");
    if (!k.isValid()) { destroyWriter(dev, w); return; }
    k.dispatch(/*gridX numTests*/ 1u, {}, push, 1u, 1u);
    dev->flush();

    // ---- the analytic expectation ----
    struct Want { uint32_t vert; BoxNP np; };
    std::vector<Want> want;
    uint32_t coveredInside = 0, coveredOutside = 0, coveredTile1 = 0, wouldFilter = 0;
    for (uint32_t i = 0; i < NB_VERTS; ++i)
    {
        const BoxNP r = cpuSphereBox(V3c{ verts[i].x, verts[i].y, verts[i].z }, boxQ, boxP,
                                     0.0f, ext, contactDist);
        if (!r.hit) continue;
        if (i == V_FILTERED) { ++wouldFilter; continue; }
        if (r.pen < 0.0f) ++coveredInside; else ++coveredOutside;
        if (i >= 512u) ++coveredTile1;
        want.push_back(Want{ i, r });
    }

    // Coverage, demanded rather than designed.
    PXG_CHECK_MSG(wouldFilter == 1u,
                  "the filtered vertex %u does not otherwise produce a contact -- V3 is unprobed",
                  V_FILTERED);
    PXG_CHECK_MSG(coveredInside > 0 && coveredOutside >= 6,
                  "the contact set has %u inside and %u outside -- SB4's 1/rsqrt round trip "
                  "needs SEVERAL outside samples before it separates from a plain sqrt",
                  coveredInside, coveredOutside);
    PXG_CHECK_MSG(coveredTile1 > 0,
                  "no contact comes from a vertex past index 511 -- the loop never runs a second "
                  "512-thread tile and the block-rounded bound is unprobed");
    {
        const BoxNP e = cpuSphereBox(V3c{ verts[V_OUTSIDE_FAR].x, verts[V_OUTSIDE_FAR].y,
                                          verts[V_OUTSIDE_FAR].z }, boxQ, boxP, 0.0f, ext, contactDist);
        PXG_CHECK_MSG(!e.hit, "vertex %u was meant to sit past the contact distance and be "
                      "rejected, but the mirror accepts it", V_OUTSIDE_FAR);
    }
    {
        const V3c cl = cpuQuatRotateInv(boxQ, V3c{ verts[V_DIAGONAL].x - boxP.x,
                                                   verts[V_DIAGONAL].y - boxP.y,
                                                   verts[V_DIAGONAL].z - boxP.z });
        const float dx = ext.x - std::fabs(cl.x), dy = ext.y - std::fabs(cl.y), dz = ext.z - std::fabs(cl.z);
        PXG_CHECK_MSG(std::fabs(dx - dy) < 1e-5f && std::fabs(dy - dz) < 1e-5f,
                      "vertex %u was meant to tie all three faces (%g, %g, %g) -- SB1's `<=` chain "
                      "is unprobed", V_DIAGONAL, dx, dy, dz);
    }

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, w.count.h, 1)[0];
    PXG_CHECK_MSG(gCount == uint32_t(want.size()),
                  "box contact-gen emitted %u contacts, expected %zu", gCount, want.size());
    if (gCount != uint32_t(want.size()))
    {
        destroyWriter(dev, w);
        dev->destroyBuffer(bVerts.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bCin.h);
        dev->destroyBuffer(bShapes.h); dev->destroyBuffer(bTc.h); dev->destroyBuffer(bCd.h);
        dev->destroyBuffer(bRd.h); dev->destroyBuffer(bRemap.h); dev->destroyBuffer(bFilters.h);
        dev->destroyBuffer(bDecoy.h);
        return;
    }

    const auto gPoint = downloadBuffer<F4>(dev, w.point.h, w.maxContacts);
    const auto gNP = downloadBuffer<F4>(dev, w.normalPen.h, w.maxContacts);
    const auto gBary = downloadBuffer<F4>(dev, w.barycentric.h, w.maxContacts);
    const auto gInfo = downloadBuffer<FemOtherContactInfoHost>(dev, w.info.h, w.maxContacts);
    const auto gByRigid = downloadBuffer<uint64_t>(dev, w.byRigid.h, w.maxContacts);
    const auto gTempByRigid = downloadBuffer<uint32_t>(dev, w.tempByRigid.h, w.maxContacts);
    const auto gIdxByRigid = downloadBuffer<uint32_t>(dev, w.idxByRigid.h, w.maxContacts);

    // The slot a contact lands in is the atomicAdd's order across tiles, i.e. a
    // race (the P9.11 rule), so the emitted contacts are matched as a SET by the
    // cloth vertex the contact-info names, and claimed BY INDEX so duplicates
    // would not cancel out.
    std::vector<bool> claimed(want.size(), false);
    size_t mm = 0;
    for (uint32_t sidx = 0; sidx < gCount; ++sidx)
    {
        const uint32_t pid = gInfo[sidx].pairInd1;
        size_t at = want.size();
        for (size_t q = 0; q < want.size(); ++q)
            if (!claimed[q] && ((CLOTH_ID << 20) | want[q].vert) == pid) { at = q; break; }
        if (at == want.size())
        { if (mm < 4) std::printf("  box slot %u names cloth vertex mask %u, which is not an "
                                  "unclaimed expected contact\n", sidx, pid);
          ++mm; continue; }
        claimed[at] = true;
        const Want& e = want[at];

        const F4 refNP = { e.np.nx, e.np.ny, e.np.nz, e.np.pen };
        // V1: the point on the RIGID -- clothP - pen * n -- NOT the cloth vertex.
        const F4 refPoint = { verts[e.vert].x - e.np.pen * e.np.nx,
                              verts[e.vert].y - e.np.pen * e.np.ny,
                              verts[e.vert].z - e.np.pen * e.np.nz,
                              restDistance };
        auto cmpF4 = [&](const char* nm, const F4& r, const F4& g) {
            if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) ||
                f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
            { ++mm; if (mm < 6) std::printf("  box vert %u %s MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n",
                                            e.vert, nm, r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w); }
        };
        cmpF4("outNormalPen", refNP, gNP[sidx]);
        cmpF4("outPoint", refPoint, gPoint[sidx]);
        cmpF4("outBarycentric", F4{ 0, 0, 0, 1 }, gBary[sidx]);
        if (gByRigid[sidx] != rigidNodeInd) { ++mm; std::printf("  box vert %u contactByRigid MM\n", e.vert); }
        if (gTempByRigid[sidx] != uint32_t(rigidNodeInd & 0xffffffffu)) { ++mm; std::printf("  box vert %u tempByRigid MM\n", e.vert); }
        if (gIdxByRigid[sidx] != sidx) { ++mm; std::printf("  box vert %u idxByRigid MM got %u want %u\n", e.vert, gIdxByRigid[sidx], sidx); }
        if (gInfo[sidx].pairInd0 != rigidNodeInd) { ++mm; std::printf("  box vert %u info.pairInd0 MM\n", e.vert); }
        if (gInfo[sidx].rigidMatInd_isInCollision != (materialIndex & 0x7fffffffu)) { ++mm; std::printf("  box vert %u info.rigidMatInd MM\n", e.vert); }
    }
    PXG_CHECK_MSG(mm == 0, "cloth_boxVertexContactGen bit-exact vs the CPU mirror (%zu mismatches)", mm);

    // Nothing past the emitted contacts may have been touched.
    size_t badPoison = 0;
    for (uint32_t sidx = gCount; sidx < w.maxContacts; ++sidx)
    {
        if (gByRigid[sidx] != 0xffffffffffffffffull) ++badPoison;
        if (gTempByRigid[sidx] != 0xffffffffu || gIdxByRigid[sidx] != 0xffffffffu) ++badPoison;
    }
    PXG_CHECK_MSG(badPoison == 0, "%zu writer slots past the emitted contacts were written", badPoison);

    // The V1 teeth: for the deep-inside vertex the emitted point must differ from
    // the cloth vertex by a measurable amount, or the sphere sibling's convention
    // would pass this fixture too.
    {
        size_t at = want.size();
        for (size_t q = 0; q < want.size(); ++q) if (want[q].vert == V_INSIDE_DEEP) { at = q; break; }
        PXG_CHECK_MSG(at != want.size(), "the deep-inside vertex produced no contact");
        if (at != want.size())
            PXG_CHECK_MSG(std::fabs(want[at].np.pen) > 0.05f,
                          "the deep-inside vertex's penetration is only %g -- the emitted point and "
                          "the cloth vertex are too close for V1 to be separable", want[at].np.pen);
    }

    std::printf("    [ note ] %zu contacts over %u verts (%u inside, %u outside, %u past index 511, "
                "1 filtered); box extents (%g, %g, %g)\n",
                want.size(), NB_VERTS, coveredInside, coveredOutside, coveredTile1,
                ext.x, ext.y, ext.z);

    destroyWriter(dev, w);
    dev->destroyBuffer(bVerts.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bCin.h);
    dev->destroyBuffer(bShapes.h); dev->destroyBuffer(bTc.h); dev->destroyBuffer(bCd.h);
    dev->destroyBuffer(bRd.h); dev->destroyBuffer(bRemap.h); dev->destroyBuffer(bFilters.h);
    dev->destroyBuffer(bDecoy.h);
}

// ---------------------------------------------------------------------------
// P9.15a companion -- SB5, the gate's STRICTNESS, on numbers that can hold it.
//
// `spherebox` accepts on `sqInflatedSum > lengthSq`, strictly.  `>` and `>=`
// differ at EXACTLY one input: equality.  The rotated fixture above cannot reach
// it -- a vertex authored in box space, rotated to world and rotated back lands a
// few ulps off -- so this companion removes every source of rounding:
//
//   * the box sits at the ORIGIN with an IDENTITY rotation, so quatRotateInv is
//     the identity map (with q = (0,0,0,1): w2 = 0.5, dot2 = 0, result = v);
//   * every extent and offset is a negative power of two or a sum of them, so the
//     clamp, the subtraction and the square are all exact in binary32.
//
// ext.x = 1.0, contactDist = 0.125 + 0.125 = 0.25, vertex at x = 1.25:
//     clamp -> 1.0, v.x = 0.25, lengthSq = 0.0625
//     inflatedSum = 0 + 0.25,   sqInflatedSum = 0.0625
// so the test is `0.0625 > 0.0625` -> FALSE -> no contact.  A `>=` mutant emits
// one.  Its neighbour one ulp closer must contact under both, which is what makes
// this a knife edge rather than a coarse rejection.
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimBoxVertContactGenExactGate)
{
    PxgSlangComputeDevice* dev = testDevice();

    const V3c ext = { 1.0f, 0.5f, 2.0f };            // exact
    const float contactOffsetCloth = 0.125f, contactOffsetRigid = 0.125f;
    const float contactDist = contactOffsetCloth + contactOffsetRigid;   // 0.25, exact
    const float restDistance = 0.0625f;
    const uint32_t materialIndex = 3u;
    const uint64_t rigidNodeInd = 0x0000000100000029ull;

    // vert0: EXACTLY at the inflated distance -> rejected by `>`.
    // vert1: one ulp closer -> accepted under both spellings, so the pair brackets
    //        the boundary instead of merely sitting on one side of it.
    // vert2: comfortably inside, so the dispatch emits something either way and a
    //        zero count cannot be mistaken for "the gate worked".
    // vert3: SB1's SECOND tie.  The rotated fixture's diagonal vertex ties all three
    //        faces at once, so the FIRST test (`dx <= dy && dx <= dz`) fires and the
    //        `else if (dy <= dz)` is never reached -- the mutant that makes that one
    //        strict SURVIVED the first sweep.  This vertex ties only Y and Z, with X
    //        strictly further, so the second test alone decides:
    //            |x| = 0.5   -> dx = 1.0 - 0.5  = 0.5
    //            |y| = 0.25  -> dy = 0.5 - 0.25 = 0.25
    //            |z| = 1.75  -> dz = 2.0 - 1.75 = 0.25
    //        `dy <= dz` takes +Y; `dy < dz` falls through to the Z seed and reports
    //        +Z instead.  Same depth, different normal.  Every value is exact.
    const float atGate = ext.x + contactDist;                       // 1.25
    const float justInside = std::nextafter(atGate, 0.0f);
    std::vector<F4> verts = {
        { atGate,     0.0f,  0.0f,  1.0f },
        { justInside, 0.0f,  0.0f,  1.0f },
        { 0.25f,      0.0f,  0.0f,  1.0f },
        { 0.5f,       0.25f, 1.75f, 1.0f },
    };
    const uint32_t NB = uint32_t(verts.size());

    DevBuf bVerts = makeDev(dev, verts);
    PxgFEMCloth c = makeClothDesc(bVerts.addr, NB);
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<PxgContactManagerInput> cmInputs(1);
    std::memset(cmInputs.data(), 0, sizeof(PxgContactManagerInput));
    cmInputs[0].shapeRef0 = 0u; cmInputs[0].shapeRef1 = 1u;
    cmInputs[0].transformCacheRef0 = 0u; cmInputs[0].transformCacheRef1 = 1u;
    DevBuf bCin = makeDev(dev, cmInputs);

    std::vector<PxgShape> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    shapes[0].type = PxU32(PxGeometryType::eTRIANGLEMESH);
    shapes[0].particleOrSoftbodyId = 0u;
    shapes[1].type = PxU32(PxGeometryType::eBOX);
    shapes[1].materialIndex = materialIndex;
    shapes[1].scale = PxMeshScale(PxVec3(ext.x, ext.y, ext.z), PxQuat(PxIdentity));
    DevBuf bShapes = makeDev(dev, shapes);

    std::vector<PxsCachedTransform> tcache(2);
    std::memset(tcache.data(), 0, tcache.size() * sizeof(PxsCachedTransform));
    tcache[1].transform = PxTransform(PxVec3(0.0f, 0.0f, 0.0f), PxQuat(PxIdentity));
    DevBuf bTc = makeDev(dev, tcache);

    std::vector<float> contactDistArr = { contactOffsetCloth, contactOffsetRigid };
    DevBuf bCd = makeDev(dev, contactDistArr);
    std::vector<float> restArr = { restDistance };
    DevBuf bRd = makeDev(dev, restArr);
    std::vector<uint64_t> remap = { 0ull, rigidNodeInd };
    DevBuf bRemap = makeDev(dev, remap);
    DevBuf bFilters = makeDev(dev, std::vector<uint32_t>{ 0u });

    WriterBufs w = makeWriter(dev, 16u);

    ContactGenPush push = {};
    push.numTests = 1u;
    push.cmInputs = bCin.addr; push.transformCache = bTc.addr; push.contactDistance = bCd.addr;
    push.restDistance = bRd.addr; push.gpuShapes = bShapes.addr; push.femClothes = bDesc.addr;
    push.shapeToRigidRemap = bRemap.addr; push.filterPairs = bFilters.addr; push.nbFilterPairs = 0u;
    fillWriter(push, w);

    Kernel k("femClothContactGenBda.spv", "cloth_boxVertexContactGenLaunch", {}, sizeof(ContactGenPush));
    PXG_CHECK_MSG(k.isValid(), "the box vertex contact-gen pipeline must build");
    if (!k.isValid()) { destroyWriter(dev, w); return; }
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    // The arithmetic must actually be exact, or this test proves nothing.
    const float identQ[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    const BoxNP g0 = cpuSphereBox(V3c{ verts[0].x, 0.0f, 0.0f }, identQ, V3c{ 0, 0, 0 }, 0.0f, ext, contactDist);
    const BoxNP g1 = cpuSphereBox(V3c{ verts[1].x, 0.0f, 0.0f }, identQ, V3c{ 0, 0, 0 }, 0.0f, ext, contactDist);
    PXG_CHECK_MSG(!g0.hit, "vertex 0 must sit EXACTLY on the gate and be rejected by `>` -- the "
                  "mirror accepts it, so the numbers are not exact after all");
    PXG_CHECK_MSG(g1.hit, "vertex 1 (one ulp closer) must be accepted, or the pair does not "
                  "bracket the boundary");

    // SB1's second tie must actually be a tie, and X must not be in the running.
    {
        const float dx = ext.x - 0.5f, dy = ext.y - 0.25f, dz = ext.z - 1.75f;
        PXG_CHECK_MSG(dy == dz && dx > dy,
                      "vertex 3 was meant to tie Y against Z with X strictly further "
                      "(dx %g, dy %g, dz %g) -- SB1's SECOND `<=` is unprobed", dx, dy, dz);
    }

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, w.count.h, 1)[0];
    PXG_CHECK_MSG(gCount == 3u,
                  "the exact-gate dispatch emitted %u contacts, expected 3 (vertex 0 sits exactly "
                  "on the gate and the test is strictly greater; a `>=` spelling emits 4)", gCount);

    if (gCount == 3u)
    {
        const auto gInfo = downloadBuffer<FemOtherContactInfoHost>(dev, w.info.h, 3);
        const auto gNP = downloadBuffer<F4>(dev, w.normalPen.h, 3);
        bool sawVert0 = false; int at3 = -1;
        for (uint32_t i = 0; i < 3u; ++i)
        {
            if (gInfo[i].pairInd1 == 0u) sawVert0 = true;
            if (gInfo[i].pairInd1 == 3u) at3 = int(i);
        }
        PXG_CHECK_MSG(!sawVert0, "the vertex sitting exactly on the gate produced a contact");
        PXG_CHECK_MSG(at3 >= 0, "the Y/Z tie vertex produced no contact");
        if (at3 >= 0)
            PXG_CHECK_MSG(f2u(gNP[at3].x) == f2u(0.0f) && f2u(gNP[at3].y) == f2u(1.0f) &&
                          f2u(gNP[at3].z) == f2u(0.0f) && f2u(gNP[at3].w) == f2u(-0.25f),
                          "the Y/Z tie must resolve to +Y at depth -0.25 (the `<=` chain), got "
                          "(%g, %g, %g) pen %g -- +Z means the second test went strict",
                          gNP[at3].x, gNP[at3].y, gNP[at3].z, gNP[at3].w);
    }

    destroyWriter(dev, w);
    dev->destroyBuffer(bVerts.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bCin.h);
    dev->destroyBuffer(bShapes.h); dev->destroyBuffer(bTc.h); dev->destroyBuffer(bCd.h);
    dev->destroyBuffer(bRd.h); dev->destroyBuffer(bRemap.h); dev->destroyBuffer(bFilters.h);
}

// ===========================================================================
// P7.1 tranche 4b-1: cloth-vs-rigid contact SORT / REORDER.
//   femClothContactSortBda.spv:
//     fem_reorderRigidContactsLaunch  (softbodyPrimitives.cu:1655)
// The reorder gathers the 4a-generated unsorted cloth-vs-rigid contact arrays
// (point / normalPen / barycentric / info) into the `…Sorted…` buffers through
// the by-rigid radix ranks (remapByRigid) produced upstream by the already-mapped
// plain radix sort, and splits the sorted contact-info's pairInd0 (PxU64 rigid id)
// into a separate sortedRigidIds array.  A PURE permutation: no float arithmetic,
// so every moved float4 / info byte must survive verbatim.  Checked bit-exact vs a
// CPU gather in BOTH memory modes (the FemOtherContactInfoHost mirror @2033 pins
// the 16-byte contact-info layout the shader gathers).
// ===========================================================================

// fem_reorderRigidContactsLaunch push blob — byte-identical to
// femClothContactSortBda.slang's FemReorderParams (ABI row {8}x11 = 88 bytes).
struct FemReorderPush
{
    uint64_t contacts, normalpens, barycentrics, contactInfos, numContacts, remapByRigid;
    uint64_t sortedContacts, sortedNormalPens, sortedBarycentrics, sortedContactInfos, sortedRigidIds;
};
static_assert(sizeof(FemReorderPush) == 88, "fem reorder push blob must be 88 bytes");

// Bit-exact reorder-gather proof for a given contact count.  NC is chosen by the
// callers to cover both the single-iteration common case and the grid-stride loop
// (NC > SB_REORDERCONTACTS grid*block = 262144 forces a second loop iteration,
// exercising the CUDA-precedence-preserved numIternations term).
static void runReorderRigidContacts(uint32_t NC, uint32_t seed)
{
    std::mt19937 rng(seed);
    PxgSlangComputeDevice* dev = testDevice();

    std::vector<F4> contacts(NC), normalpens(NC), barycentrics(NC);
    std::vector<FemOtherContactInfoHost> infos(NC);
    for (uint32_t i = 0; i < NC; ++i)
    {
        // Distinct finite floats per lane (rng() value-cast, the psContactSort
        // precedent) — a wrong permutation is caught, no NaN-canonicalization risk.
        contacts[i]     = { float(rng()), float(rng()), float(rng()), float(rng()) };
        normalpens[i]   = { float(rng()), float(rng()), float(rng()), float(rng()) };
        barycentrics[i] = { float(rng()), float(rng()), float(rng()), float(rng()) };
        // The 16-byte contact info carries raw integer bit patterns (a true bit
        // test); pairInd0 is the PxU64 rigid id the kernel also splits out.
        infos[i].pairInd0 = (uint64_t(rng()) << 32) | uint64_t(rng());
        infos[i].pairInd1 = rng();
        infos[i].rigidMatInd_isInCollision = rng();
    }

    // by-rigid sort ranks: an arbitrary permutation of [0, NC) (what the upstream
    // radix sort emits into mContactRemapSortedByRigidBuf).
    std::vector<uint32_t> remapByRigid(NC);
    std::iota(remapByRigid.begin(), remapByRigid.end(), 0u);
    std::shuffle(remapByRigid.begin(), remapByRigid.end(), rng);

    DevBuf bContacts = makeDev(dev, contacts);
    DevBuf bNormalPens = makeDev(dev, normalpens);
    DevBuf bBarycentrics = makeDev(dev, barycentrics);
    DevBuf bInfos = makeDev(dev, infos);
    DevBuf bNum = makeDev(dev, std::vector<uint32_t>(1, NC));
    DevBuf bRemap = makeDev(dev, remapByRigid);
    DevBuf bSortedContacts = makeDev(dev, std::vector<F4>(NC));
    DevBuf bSortedNormalPens = makeDev(dev, std::vector<F4>(NC));
    DevBuf bSortedBarycentrics = makeDev(dev, std::vector<F4>(NC));
    DevBuf bSortedInfos = makeDev(dev, std::vector<FemOtherContactInfoHost>(NC));
    DevBuf bSortedRigidIds = makeDev(dev, std::vector<uint64_t>(NC));
    if (!bContacts.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    FemReorderPush push = {
        bContacts.addr, bNormalPens.addr, bBarycentrics.addr, bInfos.addr, bNum.addr, bRemap.addr,
        bSortedContacts.addr, bSortedNormalPens.addr, bSortedBarycentrics.addr,
        bSortedInfos.addr, bSortedRigidIds.addr };
    Kernel k("femClothContactSortBda.spv", "fem_reorderRigidContactsLaunch", {}, sizeof(FemReorderPush));
    k.dispatch(/*gridX SB_REORDERCONTACTS*/ 1024u, {}, push);
    dev->flush();

    const auto gC = downloadBuffer<F4>(dev, bSortedContacts.h, NC);
    const auto gN = downloadBuffer<F4>(dev, bSortedNormalPens.h, NC);
    const auto gB = downloadBuffer<F4>(dev, bSortedBarycentrics.h, NC);
    const auto gI = downloadBuffer<FemOtherContactInfoHost>(dev, bSortedInfos.h, NC);
    const auto gR = downloadBuffer<uint64_t>(dev, bSortedRigidIds.h, NC);

    size_t bad = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const uint32_t r = remapByRigid[i];
        if (std::memcmp(&gC[i], &contacts[r], sizeof(F4)) != 0) ++bad;
        if (std::memcmp(&gN[i], &normalpens[r], sizeof(F4)) != 0) ++bad;
        if (std::memcmp(&gB[i], &barycentrics[r], sizeof(F4)) != 0) ++bad;
        if (std::memcmp(&gI[i], &infos[r], sizeof(FemOtherContactInfoHost)) != 0) ++bad;
        if (gR[i] != infos[r].pairInd0) ++bad; // sortedRigidIds[i] == sortedInfo.pairInd0
    }
    PXG_CHECK_MSG(bad == 0, "cloth reorder gather bit-exact NC=%u (%zu mismatches)", NC, bad);

    dev->destroyBuffer(bContacts.h); dev->destroyBuffer(bNormalPens.h); dev->destroyBuffer(bBarycentrics.h);
    dev->destroyBuffer(bInfos.h); dev->destroyBuffer(bNum.h); dev->destroyBuffer(bRemap.h);
    dev->destroyBuffer(bSortedContacts.h); dev->destroyBuffer(bSortedNormalPens.h);
    dev->destroyBuffer(bSortedBarycentrics.h); dev->destroyBuffer(bSortedInfos.h);
    dev->destroyBuffer(bSortedRigidIds.h);
}

// Common case: ~20 active device blocks, single grid-stride iteration.
PXG_TEST(femClothSimReorderRigidContacts) { runReorderRigidContacts(5000u, 0x5EED4B01u); }

// Grid-stride coverage: NC just past SB_REORDERCONTACTS grid*block (256*1024 =
// 262144) so the last 777 contacts are gathered on the SECOND loop iteration,
// exercising the numIternations / index += blockDim*gridDim stride verbatim.
PXG_TEST(femClothSimReorderRigidContactsMultiIter) { runReorderRigidContacts(262144u + 777u, 0x4B02F00Du); }

// ===========================================================================
// P7.1 tranche 4b-2/3: cloth-vs-rigid contact PREPARE + query/SOLVE.
//   femClothRigidContactBda.spv:
//     cloth_rigidContactPrepareLaunch                       (FEMClothConstraintPrep.cu:70)
//     cloth_queryRigidClothContactReferenceCount{,TGS}Launch(FEMClothExternalSolve.cu:414/:558)
//     cloth_solveRigidClothCollision{,TGS}Launch            (FEMClothExternalSolve.cu:535/:578)
// ONE cloth vertex penetrating a STATIC floor -> a sorted contact -> prepare (build the
// PxgFemRigidConstraintBlock) -> queryRefCount (bump mDeltaPos.w) -> solve (push the
// vertex OUT along the plane normal into mDeltaPos).  Bit-exact vs a CPU transcription of
// the .cu math (PGS and TGS), both memory modes.
// ===========================================================================

// Host replica of the block-of-32 PxgFemRigidConstraintBlock (PxgFEMCore.h:167, verbatim
// hand-copy — PxgFEMCore.h is not includable in this CUDA-toolkit-free TU).  A trivial
// fixed-array SoA (float4[32] blocks + float[32]); the byte image the prepare WRITES and
// the solve READS, matching the shader's FRC_* byte offsets (femClothRigidContactBda.slang).
struct RigidContactConstraintHost
{
    F4 raXn_resp[32];          // @0
    F4 raXnF0_resp[32];        // @512
    F4 raXnF1_resp[32];        // @1024
    F4 normal_errorW[32];      // @1536
    F4 fricTan0_invMass0[32];  // @2048
    F4 barycentric[32];        // @2560
    float maxPenBias[32];      // @3072 -> 3200
};
static_assert(sizeof(RigidContactConstraintHost) == 3200, "PxgFemRigidConstraintBlock sizeof == 3200");
static_assert(offsetof(RigidContactConstraintHost, raXnF0_resp) == 512, "raXnF0_resp @512");
static_assert(offsetof(RigidContactConstraintHost, raXnF1_resp) == 1024, "raXnF1_resp @1024");
static_assert(offsetof(RigidContactConstraintHost, normal_errorW) == 1536, "normal_errorW @1536");
static_assert(offsetof(RigidContactConstraintHost, fricTan0_invMass0) == 2048, "fricTan0_invMass0 @2048");
static_assert(offsetof(RigidContactConstraintHost, barycentric) == 2560, "barycentric @2560");
static_assert(offsetof(RigidContactConstraintHost, maxPenBias) == 3072, "maxPenBias @3072");

// ---------------------------------------------------------------------------
// bit-exact CPU reference for the prepare + query/solve.  prepareFEMContacts
// (deformableUtils.cuh:132) and FEMCollision::initialize / readCloth / computeFEMChange /
// writeCloth (deformableUtils.cuh:280 / :488) are transcribed verbatim for the STATIC /
// null-material (frictionless) path.
// ---------------------------------------------------------------------------
namespace p714ref
{
struct V3 { float x, y, z; };
static V3 cross3(V3 a, V3 b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static float rmax(float a, float b) { return a > b ? a : b; }
static float rmin(float a, float b) { return a < b ? a : b; }
static float rabs(float a) { return a < 0.f ? -a : a; }

// PxMat33 (column-major [9]) * PxVec3.
static V3 mat33Mul(const float m[9], V3 v)
{
    return V3{ m[0] * v.x + m[3] * v.y + m[6] * v.z,
               m[1] * v.x + m[4] * v.y + m[7] * v.z,
               m[2] * v.x + m[5] * v.y + m[8] * v.z };
}

// PxComputeBasisVectors (PxMathUtils.h:297); right/up .normalize() is v / sqrt(magSq).
static void computeBasis(V3 dir, V3& right, V3& up)
{
    if (rabs(dir.y) <= 0.9999f)
    {
        right = V3{ dir.z, 0.0f, -dir.x };
        const float m = std::sqrt(right.x * right.x + right.y * right.y + right.z * right.z);
        if (m > 0.0f) right = V3{ right.x / m, right.y / m, right.z / m };
        up = V3{ dir.y * right.z, dir.z * right.x - dir.x * right.z, -dir.y * right.x };
    }
    else
    {
        right = V3{ 1.0f, 0.0f, 0.0f };
        up = V3{ 0.0f, dir.z, -dir.y };
        const float m = std::sqrt(up.x * up.x + up.y * up.y + up.z * up.z);
        if (m > 0.0f) up = V3{ up.x / m, up.y / m, up.z / m };
    }
}

// prepareFEMContacts (deformableUtils.cuh:132) — rigid / static (non-articulation) path.
static void refPrepare(RigidContactConstraintHost& c, int slot,
                       V3 normal, V3 p, float pen, V3 delta, F4 barycentric,
                       V3 bodyFrameP, float invMass0, float penBiasClampRigid,
                       const float sqrtInvInertia[9], float penBiasClampFEM, bool isStatic)
{
    V3 t0, t1;
    computeBasis(normal, t0, t1);

    const bool isKinematic = (invMass0 == 0.0f) && (!isStatic);

    const V3 ra = V3{ p.x - bodyFrameP.x, p.y - bodyFrameP.y, p.z - bodyFrameP.z };
    const V3 raXn = cross3(ra, normal);
    const V3 raXF0 = cross3(ra, t0);
    const V3 raXF1 = cross3(ra, t1);

    const V3 raXnSI = mat33Mul(sqrtInvInertia, raXn);
    const float resp0 = dot3(raXnSI, raXnSI) + invMass0;
    const V3 raXF0SI = mat33Mul(sqrtInvInertia, raXF0);
    const V3 raXF1SI = mat33Mul(sqrtInvInertia, raXF1);
    const float respF0 = dot3(raXF0SI, raXF0SI) + invMass0;
    const float respF1 = dot3(raXF1SI, raXF1SI) + invMass0;

    F4 raXn_resp, raXnF0_resp, raXnF1_resp;
    if (isKinematic)
    {
        raXn_resp = F4{ raXn.x, raXn.y, raXn.z, resp0 };
        raXnF0_resp = F4{ raXF0.x, raXF0.y, raXF0.z, respF0 };
        raXnF1_resp = F4{ raXF1.x, raXF1.y, raXF1.z, respF1 };
    }
    else
    {
        raXn_resp = F4{ raXnSI.x, raXnSI.y, raXnSI.z, resp0 };
        raXnF0_resp = F4{ raXF0SI.x, raXF0SI.y, raXF0SI.z, respF0 };
        raXnF1_resp = F4{ raXF1SI.x, raXF1SI.y, raXF1SI.z, respF1 };
    }

    const float maxPenBias = rmax(penBiasClampRigid, penBiasClampFEM);
    const float error = pen + dot3(delta, normal);

    c.normal_errorW[slot] = F4{ normal.x, normal.y, normal.z, error };
    c.raXn_resp[slot] = raXn_resp;
    c.raXnF0_resp[slot] = raXnF0_resp;
    c.raXnF1_resp[slot] = raXnF1_resp;
    c.fricTan0_invMass0[slot] = F4{ t0.x, t0.y, t0.z, invMass0 };
    c.maxPenBias[slot] = maxPenBias;
    c.barycentric[slot] = barycentric;
}

// FEMCollision::initialize (deformableUtils.cuh:280) — static / null-material (frictionless)
// path.  Rigid velocities are all zero for a static body at rest.
struct InitResult { bool isActive; float deltaLambdaN; float deltaLambdaT; V3 normal; V3 tangent; float accum; };
static InitResult refInitialize(const RigidContactConstraintHost& c, int slot, bool isTGS,
                                float appliedForceRef, float dt, V3 bcVec, V3 deformableLinDelta,
                                V3 deformableInvMasses, float rigidRefCount, bool wasActive, bool checkOnly)
{
    InitResult r;
    r.isActive = false; r.deltaLambdaN = 0.0f; r.deltaLambdaT = 0.0f;
    r.tangent = V3{ 0, 0, 0 }; r.accum = appliedForceRef;

    const float invDt = 1.0f / dt;
    const F4 raXn_resp = c.raXn_resp[slot];
    const F4 normal_biasW = c.normal_errorW[slot];
    const F4 raXnF0_resp = c.raXnF0_resp[slot];
    const F4 raXnF1_resp = c.raXnF1_resp[slot];
    const float maxPenBias = c.maxPenBias[slot];
    const F4 fricTan0 = c.fricTan0_invMass0[slot];

    const V3 normal = V3{ normal_biasW.x, normal_biasW.y, normal_biasW.z };
    r.normal = normal;
    const V3 fric0 = V3{ fricTan0.x, fricTan0.y, fricTan0.z };
    const V3 fric1 = cross3(normal, fric0);
    const float initPen = normal_biasW.w;
    const V3 raXn = V3{ raXn_resp.x, raXn_resp.y, raXn_resp.z };

    // static body at rest -> zero velocities / deltas.
    const V3 linVel = V3{ 0, 0, 0 }, angVel = V3{ 0, 0, 0 };
    const V3 rigidLinDelta = V3{ 0, 0, 0 }, rigidAngDelta = V3{ 0, 0, 0 };

    float CN; V3 relLinDelta; V3 angDelta;
    if (isTGS)
    {
        const float normalVel = dot3(linVel, normal) + dot3(angVel, raXn);
        relLinDelta = V3{ rigidLinDelta.x - deformableLinDelta.x, rigidLinDelta.y - deformableLinDelta.y,
                          rigidLinDelta.z - deformableLinDelta.z };
        angDelta = rigidAngDelta;
        const float error = (initPen + dot3(relLinDelta, normal) + dot3(angDelta, raXn)) * invDt;
        const float errorBiased = rmax(maxPenBias, error);
        CN = errorBiased + normalVel;
    }
    else
    {
        const float normalVel = dot3(linVel, normal) + dot3(angVel, raXn);
        relLinDelta = V3{ -deformableLinDelta.x, -deformableLinDelta.y, -deformableLinDelta.z };
        angDelta = V3{ 0, 0, 0 };
        const float error = (initPen + dot3(relLinDelta, normal)) * invDt;
        const float errorBiased = rmax(maxPenBias, error);
        CN = errorBiased + normalVel;
    }

    r.isActive = wasActive || (CN < 0.0f);
    if (checkOnly) return r;
    if (!r.isActive) return r;

    const V3 bcSq = V3{ bcVec.x * bcVec.x, bcVec.y * bcVec.y, bcVec.z * bcVec.z };
    const float deformableInvMass = dot3(bcSq, deformableInvMasses);
    const float unitResponse = rigidRefCount * raXn_resp.w + deformableInvMass;
    const float invDenom = (unitResponse > 0.0f) ? (1.0f / unitResponse) : 0.0f;

    r.deltaLambdaN = rmax(-CN * invDenom, -r.accum);
    r.accum += r.deltaLambdaN;

    const V3 raXnF0 = V3{ raXnF0_resp.x, raXnF0_resp.y, raXnF0_resp.z };
    const V3 raXnF1 = V3{ raXnF1_resp.x, raXnF1_resp.y, raXnF1_resp.z };
    const float tanVel0 = dot3(linVel, fric0) + dot3(angVel, raXnF0);
    const float tanVel1 = dot3(linVel, fric1) + dot3(angVel, raXnF1);
    const float CT0 = (dot3(fric0, relLinDelta) + dot3(angDelta, raXnF0)) * invDt + tanVel0;
    const float CT1 = (dot3(fric1, relLinDelta) + dot3(angDelta, raXnF1)) * invDt + tanVel1;
    const V3 relTanDelta = V3{ CT0 * fric0.x + CT1 * fric1.x, CT0 * fric0.y + CT1 * fric1.y,
                               CT0 * fric0.z + CT1 * fric1.z };
    const float tanMagSq = dot3(relTanDelta, relTanDelta);
    if (tanMagSq > 1.0e-14f)
    {
        const float CT = std::sqrt(tanMagSq);
        const float invTanMag = 1.0f / CT;
        r.tangent = V3{ relTanDelta.x * invTanMag, relTanDelta.y * invTanMag, relTanDelta.z * invTanMag };
        const float uT0 = rigidRefCount * raXnF0_resp.w + deformableInvMass;
        const float invT0 = (uT0 > 0.0f) ? (1.0f / uT0) : 0.0f;
        const float uT1 = rigidRefCount * raXnF1_resp.w + deformableInvMass;
        const float invT1 = (uT1 > 0.0f) ? (1.0f / uT1) : 0.0f;
        const float dLT0 = CT0 * invT0;
        const float dLT1 = CT1 * invT1;
        const float dLT = std::sqrt(dLT0 * dLT0 + dLT1 * dLT1);
        const float combinedFriction = 0.0f;   // null-material path (DEFERRED)
        r.deltaLambdaT = -rmin(dLT, combinedFriction * rabs(r.deltaLambdaN));
    }
    return r;
}
} // namespace p714ref

// ---------------------------------------------------------------------------
// cloth_rigidContactPrepareLaunch -> query -> solve: push one cloth vertex OUT of a
// static floor.  isTGS selects the PGS / TGS solve twin.
// ---------------------------------------------------------------------------
static void runRigidContact(bool isTGS)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 1.0f / 60.0f;

    // --- cloth vertex (id 0, elemIdx 0), invMass = 0.5 (movable) ----------
    const float invMass = 0.5f;
    // accumulated delta (xyz) nonzero -> exercises the initialize relLinDelta term; .w = invMass.
    const F4 accum0 = { 0.0010f, -0.0020f, 0.0015f, invMass };
    std::vector<F4> positions = { { 0.5f, -0.02f, 0.7f, invMass } };
    std::vector<F4> velocities = { { 0.0f, 0.0f, 0.0f, invMass } };
    std::vector<F4> accumDelta = { accum0 };
    std::vector<F4> deltaPos = { { 0.0f, 0.0f, 0.0f, 0.0f } };   // query/solve output
    std::vector<uint32_t> triVtx = { 0u, 0u, 0u, 0u };           // unused (vertex contact)

    DevBuf bPos = makeDev(dev, positions);
    DevBuf bVel = makeDev(dev, velocities);
    DevBuf bAcc = makeDev(dev, accumDelta);
    DevBuf bDelta = makeDev(dev, deltaPos);
    DevBuf bTri = makeDev(dev, triVtx);

    const float penBiasClampFEM = -1.0e10f;
    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bPos.addr));
    c.mVelocity_InvMass = reinterpret_cast<decltype(c.mVelocity_InvMass)>(size_t(bVel.addr));
    c.mAccumulatedDeltaPos = reinterpret_cast<decltype(c.mAccumulatedDeltaPos)>(size_t(bAcc.addr));
    c.mDeltaPos = reinterpret_cast<decltype(c.mDeltaPos)>(size_t(bDelta.addr));
    c.mTriangleVertexIndices = reinterpret_cast<decltype(c.mTriangleVertexIndices)>(size_t(bTri.addr));
    c.mPenBiasClamp = penBiasClampFEM;
    c.mNbVerts = 1u;
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    // --- the STATIC floor contact -----------------------------------------
    // plane normal (unit, tilted, dominant +Y so "out of floor" ~ +Y).  Stored in
    // normalPens.xyz; the solver normal is -planeNormal.
    float rawN[3] = { 0.2f, 0.9f, 0.3f };
    const float nmag = std::sqrt(rawN[0] * rawN[0] + rawN[1] * rawN[1] + rawN[2] * rawN[2]);
    const p714ref::V3 planeNormal = { rawN[0] / nmag, rawN[1] / nmag, rawN[2] / nmag };
    const float penW = -0.02f;   // pInPlaneSpace.x < 0 -> penetrating (pen = penW - restW).

    std::vector<F4> contacts = { { positions[0].x, positions[0].y, positions[0].z, 0.0f } }; // restW = 0
    std::vector<F4> normalPens = { { planeNormal.x, planeNormal.y, planeNormal.z, penW } };
    std::vector<F4> barycentrics = { { 0.0f, 0.0f, 0.0f, 1.0f } };   // vertex (bc.w != 0)
    const uint64_t staticNodeInd = 0x00000000FFFFFFFFull;
    std::vector<FemOtherContactInfoHost> infos(1);
    infos[0].pairInd0 = staticNodeInd;         // rigid node (static)
    infos[0].pairInd1 = 0u;                     // (clothId 0 << 20) | elemIdx 0
    infos[0].rigidMatInd_isInCollision = 0u;    // not-yet-in-collision, rigidMatIndex 0

    DevBuf bContacts = makeDev(dev, contacts);
    DevBuf bNormalPens = makeDev(dev, normalPens);
    DevBuf bBary = makeDev(dev, barycentrics);
    DevBuf bInfos = makeDev(dev, infos);
    std::vector<uint32_t> numContacts = { 1u };
    DevBuf bNum = makeDev(dev, numContacts);

    // constraint block (3200 bytes) seeded with a sentinel pattern.
    std::vector<uint8_t> constraintBytes(sizeof(RigidContactConstraintHost), 0xCD);
    DevBuf bConstraint = makeDev(dev, constraintBytes);

    std::vector<float> rigidLambdaNs = { 123.0f };   // prepare zeroes it (appliedForces)
    DevBuf bLambda = makeDev(dev, rigidLambdaNs);
    std::vector<F4> rigidDeltaVel = { { 9.f, 9.f, 9.f, 9.f }, { 9.f, 9.f, 9.f, 9.f } };
    DevBuf bRigidDeltaVel = makeDev(dev, rigidDeltaVel);
    std::vector<uint32_t> refCounts = { 0u, 0u };    // femRigidContactCount (unused for static)
    DevBuf bRefCounts = makeDev(dev, refCounts);

    // --- static-body frame / mass / inertia (solver body 0) ---------------
    // body2WorldPool[0] = identity quat, p = origin -> ra == contact point.
    std::vector<F4> body2World = { { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } };
    DevBuf bBody2World = makeDev(dev, body2World);
    // solverBodyData[0]: initialLinVelXYZ_invMassW @16 = 0 (invMass0 = 0),
    //   initialAngVelXYZ_penBiasClamp.w @12 = penBiasClampRigid.
    const float penBiasClampRigid = -1.0e10f;
    std::vector<uint8_t> solverBodyData(96, 0);
    std::memcpy(solverBodyData.data() + 12, &penBiasClampRigid, 4);
    DevBuf bSolverBodyData = makeDev(dev, solverBodyData);
    // solverTxIData[0].sqrtInvInertia @28: a NON-zero symmetric matrix so the rigid
    // response rows (raXn x sqrtInvInertia) have teeth (a static body reads slot 0's
    // matrix verbatim; velocities stay zero, so only the response denominator is affected).
    const float sqrtInvInertia[9] = { 2.0f, 0.1f, 0.0f, 0.1f, 1.5f, 0.2f, 0.0f, 0.2f, 1.0f };
    std::vector<uint8_t> solverTxIData(64, 0);
    std::memcpy(solverTxIData.data() + 28, sqrtInvInertia, 9 * 4);
    DevBuf bSolverTxIData = makeDev(dev, solverTxIData);

    // --- solver-body velocity buffers (static floor at rest) --------------
    const uint32_t numSolverBodies = 1u;
    const uint32_t accOffset = 0u;
    std::vector<F4> outSolverVelocity = { { 0.f, 0.f, 0.f, 0.f }, { 0.f, 0.f, 0.f, 0.f },
                                          { 77.f, 78.f, 79.f, 80.f }, { 81.f, 82.f, 83.f, 84.f } };
    DevBuf bOutSolverVel = makeDev(dev, outSolverVelocity);
    std::vector<F4> solverBodyVelPool = { { 0.f, 0.f, 0.f, 0.f }, { 0.f, 0.f, 0.f, 0.f },
                                          { 0.f, 0.f, 0.f, 0.f }, { 61.f, 62.f, 63.f, 64.f } };
    DevBuf bSolverBodyVelPool = makeDev(dev, solverBodyVelPool);
    std::vector<uint32_t> solverBodyIndices = { 0u };   // never dereferenced for a static body
    DevBuf bSolverBodyIndices = makeDev(dev, solverBodyIndices);

    // --- raw descriptor buffers, pointers poked at the mirror offsets -----
    std::vector<uint8_t> prepDesc(304, 0);          // PxgConstraintPrepareDescGpu
    pokePtr(prepDesc, 64, bSolverBodyData.addr);    // solverBodyDataPool @64
    pokePtr(prepDesc, 72, bSolverTxIData.addr);     // solverBodyTxIDataPool @72
    pokePtr(prepDesc, 168, bBody2World.addr);       // body2WorldPool @168
    DevBuf bPrepDesc = makeDev(dev, prepDesc);

    std::vector<uint8_t> prePrepDesc(760, 0);       // PxgPrePrepDescGpu
    pokePtr(prePrepDesc, 272, bSolverBodyIndices.addr); // solverBodyIndices @272 (static -> unused)
    DevBuf bPrePrepDesc = makeDev(dev, prePrepDesc);

    std::vector<uint8_t> solverCoreDesc(320, 0);    // PxgSolverCoreDescGpu
    pokePtr(solverCoreDesc, 0, bOutSolverVel.addr);        // outSolverVelocity @0
    pokeU32(solverCoreDesc, 268, numSolverBodies);         // numSolverBodies @268
    pokeU32(solverCoreDesc, 276, accOffset);               // accumulatedBodyDeltaVOffset @276
    DevBuf bSolverCoreDesc = makeDev(dev, solverCoreDesc);

    std::vector<uint8_t> sharedDesc(224, 0);        // PxgSolverSharedDesc<IterativeSolveData(TGS)>
    pokePtr(sharedDesc, 160, bSolverBodyVelPool.addr);     // iter_solverBodyVelPool @160
    DevBuf bSharedDesc = makeDev(dev, sharedDesc);

    // ======================= dispatch PREPARE ==========================
    struct PrepPush
    {
        uint64_t femClothes, contacts, normalPens, barycentrics, contactInfos, numContacts;
        uint64_t constraints, prePrepDesc, prepDesc, rigidLambdaNs;
        float invDt; uint32_t pad84;
        uint64_t sharedDesc;
        uint32_t isTGS, pad100;
    };
    static_assert(sizeof(PrepPush) == 104, "prepare push-constant blob must be 104 bytes");
    PrepPush pp = {};
    pp.femClothes = bDesc.addr; pp.contacts = bContacts.addr; pp.normalPens = bNormalPens.addr;
    pp.barycentrics = bBary.addr; pp.contactInfos = bInfos.addr; pp.numContacts = bNum.addr;
    pp.constraints = bConstraint.addr; pp.prePrepDesc = bPrePrepDesc.addr; pp.prepDesc = bPrepDesc.addr;
    pp.rigidLambdaNs = bLambda.addr; pp.invDt = 1.0f / dt; pp.sharedDesc = bSharedDesc.addr;
    pp.isTGS = isTGS ? 1u : 0u;

    Kernel kPrep("femClothRigidContactBda.spv", "cloth_rigidContactPrepareLaunch", {}, sizeof(PrepPush));
    kPrep.dispatch(1u, {}, pp, 1u, 1u);
    dev->flush();

    // ---- CPU reference for the prepared constraint ----
    RigidContactConstraintHost refC;
    std::memset(&refC, 0xCD, sizeof(refC));
    const p714ref::V3 solverNormal = { -planeNormal.x, -planeNormal.y, -planeNormal.z };
    const float pen = penW - contacts[0].w;
    p714ref::refPrepare(refC, 0, solverNormal, p714ref::V3{ positions[0].x, positions[0].y, positions[0].z },
                        pen, p714ref::V3{ accum0.x, accum0.y, accum0.z }, barycentrics[0],
                        p714ref::V3{ 0, 0, 0 }, /*invMass0*/ 0.0f, penBiasClampRigid,
                        sqrtInvInertia, penBiasClampFEM, /*isStatic*/ true);

    const auto gConstraintBytes = downloadBuffer<uint8_t>(dev, bConstraint.h, sizeof(RigidContactConstraintHost));
    RigidContactConstraintHost gotC;
    std::memcpy(&gotC, gConstraintBytes.data(), sizeof(gotC));
    const auto gLambda = downloadBuffer<float>(dev, bLambda.h, 1);

    size_t mm = 0;
    auto cmp4 = [&](const char* nm, const F4& r, const F4& g) {
        if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) || f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
        { ++mm; std::printf("  prep %s MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", nm,
                            r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w); }
    };
    cmp4("raXn_resp", refC.raXn_resp[0], gotC.raXn_resp[0]);
    cmp4("raXnF0_resp", refC.raXnF0_resp[0], gotC.raXnF0_resp[0]);
    cmp4("raXnF1_resp", refC.raXnF1_resp[0], gotC.raXnF1_resp[0]);
    cmp4("normal_errorW", refC.normal_errorW[0], gotC.normal_errorW[0]);
    cmp4("fricTan0_invMass0", refC.fricTan0_invMass0[0], gotC.fricTan0_invMass0[0]);
    cmp4("barycentric", refC.barycentric[0], gotC.barycentric[0]);
    if (f2u(refC.maxPenBias[0]) != f2u(gotC.maxPenBias[0]))
    { ++mm; std::printf("  prep maxPenBias MM: ref %g got %g\n", refC.maxPenBias[0], gotC.maxPenBias[0]); }
    // appliedForces zeroed by the prepare.
    if (f2u(gLambda[0]) != 0u)
    { ++mm; std::printf("  rigidLambdaNs not zeroed: %g\n", gLambda[0]); }

    // teeth: the response rows must be non-zero (the sqrtInvInertia matrix is exercised).
    PXG_CHECK_MSG(f2u(gotC.raXn_resp[0].x) != 0u || f2u(gotC.raXn_resp[0].y) != 0u ||
                  f2u(gotC.raXn_resp[0].z) != 0u,
                  "prepared raXn response must be non-zero (sqrtInvInertia teeth)");
    PXG_CHECK_MSG(mm == 0, "cloth_rigidContactPrepareLaunch bit-exact vs CPU (%zu mismatches)", mm);

    // ======================= dispatch QUERY (refcount) =================
    struct QueryPush
    {
        uint64_t femClothes, contactInfos, constraints, numContacts, prePrepDesc, solverCoreDesc;
        uint64_t artiCoreDesc, sharedDesc;
        float dt; uint32_t pad68;
        uint64_t appliedForces, rigidBodyReferenceCounts;
    };
    static_assert(sizeof(QueryPush) == 88, "query push-constant blob must be 88 bytes");
    QueryPush qp = {};
    qp.femClothes = bDesc.addr; qp.contactInfos = bInfos.addr; qp.constraints = bConstraint.addr;
    qp.numContacts = bNum.addr; qp.prePrepDesc = bPrePrepDesc.addr; qp.solverCoreDesc = bSolverCoreDesc.addr;
    qp.artiCoreDesc = 0ull; qp.sharedDesc = bSharedDesc.addr; qp.dt = dt;
    qp.appliedForces = bLambda.addr; qp.rigidBodyReferenceCounts = bRefCounts.addr;

    const char* queryEntry = isTGS ? "cloth_queryRigidClothContactReferenceCountTGSLaunch"
                                   : "cloth_queryRigidClothContactReferenceCountLaunch";
    Kernel kQuery("femClothRigidContactBda.spv", queryEntry, {}, sizeof(QueryPush));
    kQuery.dispatch(1u, {}, qp, 1u, 1u);
    dev->flush();

    // ---- CPU reference for the query (mark in-collision + bump mDeltaPos.w) ----
    const p714ref::V3 bcVec = { 1.0f, 0.0f, 0.0f };   // vertex
    const p714ref::V3 accumV = { accum0.x, accum0.y, accum0.z };
    const p714ref::InitResult qir = p714ref::refInitialize(refC, 0, isTGS, /*appliedForceRef*/ 0.0f, dt,
                                                           bcVec, accumV, p714ref::V3{ invMass, 0, 0 },
                                                           /*rigidRefCount*/ 1.0f, /*wasActive*/ false,
                                                           /*checkOnly*/ true);
    PXG_CHECK_MSG(qir.isActive, "query CPU ref: the penetrating contact must be active");
    F4 refDeltaAfterQuery = { 0.0f, 0.0f, 0.0f, (qir.isActive && invMass != 0.0f) ? 1.0f : 0.0f };

    const auto gDeltaQ = downloadBuffer<F4>(dev, bDelta.h, 1);
    const auto gInfoQ = downloadBuffer<FemOtherContactInfoHost>(dev, bInfos.h, 1);
    if (f2u(gDeltaQ[0].w) != f2u(refDeltaAfterQuery.w))
    { ++mm; std::printf("  query mDeltaPos.w MM: ref %g got %g\n", refDeltaAfterQuery.w, gDeltaQ[0].w); }
    PXG_CHECK_MSG(f2u(gDeltaQ[0].w) == f2u(1.0f), "query must bump mDeltaPos.w to 1 (reference count)");
    PXG_CHECK_MSG((gInfoQ[0].rigidMatInd_isInCollision & 0x80000000u) != 0u,
                  "query must mark the contact in-collision (bit31)");
    PXG_CHECK_MSG(mm == 0, "cloth_queryRigidClothContactReferenceCount%s bit-exact vs CPU (%zu mismatches)",
                  isTGS ? "TGS" : "PGS", mm);

    // ======================= dispatch SOLVE ============================
    struct SolvePush
    {
        uint64_t femClothes, contactInfos, constraints, numContacts, prePrepDesc, solverCoreDesc;
        uint64_t artiCoreDesc, sharedDesc, rigidDeltaVel, appliedForces, rigidBodyReferenceCounts;
        float dt; uint32_t pad92;
        uint64_t materials, rigidBodyMaterials;
    };
    static_assert(sizeof(SolvePush) == 112, "solve push-constant blob must be 112 bytes");
    SolvePush sp = {};
    sp.femClothes = bDesc.addr; sp.contactInfos = bInfos.addr; sp.constraints = bConstraint.addr;
    sp.numContacts = bNum.addr; sp.prePrepDesc = bPrePrepDesc.addr; sp.solverCoreDesc = bSolverCoreDesc.addr;
    sp.artiCoreDesc = 0ull; sp.sharedDesc = bSharedDesc.addr; sp.rigidDeltaVel = bRigidDeltaVel.addr;
    sp.appliedForces = bLambda.addr; sp.rigidBodyReferenceCounts = bRefCounts.addr; sp.dt = dt;
    sp.materials = 0ull; sp.rigidBodyMaterials = 0ull;   // DEFERRED null-material path

    const char* solveEntry = isTGS ? "cloth_solveRigidClothCollisionTGSLaunch"
                                   : "cloth_solveRigidClothCollisionLaunch";
    Kernel kSolve("femClothRigidContactBda.spv", solveEntry, {}, sizeof(SolvePush));
    kSolve.dispatch(1u, {}, sp, 1u, 1u);
    dev->flush();

    // ---- CPU reference for the solved mDeltaPos ----
    // readCloth (checkOnly=false): mass-splitting by the query's refCount (mDeltaPos.w == 1).
    const float refCount = refDeltaAfterQuery.w;   // 1
    const p714ref::V3 deformableInvMasses = { invMass * refCount, 0.0f, 0.0f };
    const p714ref::InitResult sir = p714ref::refInitialize(refC, 0, isTGS, /*appliedForceRef*/ 0.0f, dt,
                                                           bcVec, accumV, deformableInvMasses,
                                                           /*rigidRefCount*/ 1.0f, /*wasActive*/ true,
                                                           /*checkOnly*/ false);
    // computeFEMChange + writeCloth (vertex).
    const p714ref::V3 deltaPosR = {
        -(sir.deltaLambdaN * sir.normal.x + sir.deltaLambdaT * sir.tangent.x) * dt,
        -(sir.deltaLambdaN * sir.normal.y + sir.deltaLambdaT * sir.tangent.y) * dt,
        -(sir.deltaLambdaN * sir.normal.z + sir.deltaLambdaT * sir.tangent.z) * dt };
    const float sScale = deformableInvMasses.x;   // > 0
    F4 refDeltaAfterSolve = { refDeltaAfterQuery.x + deltaPosR.x * sScale,
                              refDeltaAfterQuery.y + deltaPosR.y * sScale,
                              refDeltaAfterQuery.z + deltaPosR.z * sScale,
                              refDeltaAfterQuery.w };   // .w unchanged (writeCloth touches xyz only)

    const auto gDelta = downloadBuffer<F4>(dev, bDelta.h, 1);
    const auto gLambda2 = downloadBuffer<float>(dev, bLambda.h, 1);
    const auto gRigidDeltaVel = downloadBuffer<F4>(dev, bRigidDeltaVel.h, 2);

    // teeth: the solve must push the vertex OUT along the plane normal (dot > 0).
    const float pushDot = gDelta[0].x * planeNormal.x + gDelta[0].y * planeNormal.y + gDelta[0].z * planeNormal.z;
    PXG_CHECK_MSG(pushDot > 0.0f, "solve must push the cloth vertex OUT of the floor (dot(mDeltaPos, planeNormal) > 0, got %g, isTGS=%d)",
                  pushDot, int(isTGS));

    size_t mm2 = 0;
    if (f2u(refDeltaAfterSolve.x) != f2u(gDelta[0].x) || f2u(refDeltaAfterSolve.y) != f2u(gDelta[0].y) ||
        f2u(refDeltaAfterSolve.z) != f2u(gDelta[0].z) || f2u(refDeltaAfterSolve.w) != f2u(gDelta[0].w))
    { ++mm2; std::printf("  solve mDeltaPos MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n",
                         refDeltaAfterSolve.x, refDeltaAfterSolve.y, refDeltaAfterSolve.z, refDeltaAfterSolve.w,
                         gDelta[0].x, gDelta[0].y, gDelta[0].z, gDelta[0].w); }
    if (f2u(gLambda2[0]) != f2u(sir.accum))
    { ++mm2; std::printf("  solve appliedForces MM: ref %g got %g\n", sir.accum, gLambda2[0]); }
    // dynamic-rigid reaction DEFERRED: static contact writes zeros with count 0.
    for (uint32_t j = 0; j < 2; ++j)
        if (f2u(gRigidDeltaVel[j].x) != 0u || f2u(gRigidDeltaVel[j].y) != 0u ||
            f2u(gRigidDeltaVel[j].z) != 0u || f2u(gRigidDeltaVel[j].w) != 0u)
        { ++mm2; std::printf("  rigidDeltaVel[%u] not zeroed (static reaction)\n", j); }
    PXG_CHECK_MSG(mm2 == 0, "cloth_solveRigidClothCollision%s bit-exact vs CPU (%zu mismatches)",
                  isTGS ? "TGS" : "PGS", mm2);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bAcc.h);
    dev->destroyBuffer(bDelta.h); dev->destroyBuffer(bTri.h); dev->destroyBuffer(bDesc.h);
    dev->destroyBuffer(bContacts.h); dev->destroyBuffer(bNormalPens.h); dev->destroyBuffer(bBary.h);
    dev->destroyBuffer(bInfos.h); dev->destroyBuffer(bNum.h); dev->destroyBuffer(bConstraint.h);
    dev->destroyBuffer(bLambda.h); dev->destroyBuffer(bRigidDeltaVel.h); dev->destroyBuffer(bRefCounts.h);
    dev->destroyBuffer(bBody2World.h); dev->destroyBuffer(bSolverBodyData.h); dev->destroyBuffer(bSolverTxIData.h);
    dev->destroyBuffer(bOutSolverVel.h); dev->destroyBuffer(bSolverBodyVelPool.h); dev->destroyBuffer(bSolverBodyIndices.h);
    dev->destroyBuffer(bPrepDesc.h); dev->destroyBuffer(bPrePrepDesc.h);
    dev->destroyBuffer(bSolverCoreDesc.h); dev->destroyBuffer(bSharedDesc.h);
}

PXG_TEST(femClothSimRigidContactPGS) { runRigidContact(/*isTGS*/ false); }
PXG_TEST(femClothSimRigidContactTGS) { runRigidContact(/*isTGS*/ true); }

// ===========================================================================
// P7.2 tranche 1: cloth SELF-collision midphase + contact GENERATION.
//   femClothSelfCollisionBda.spv:
//     cloth_selfCollisionMidphaseVTLaunch  (femClothClothMidPhase.cu:1163)
//     cloth_selfCollisionMidphaseEELaunch  (femClothClothMidPhase.cu:1196)
// Both kernels walk the cloth's own BV32 triangle BVH (one 32-thread group per
// query) and append a PxgFemFemContactInfo for every penetrating vertex-triangle
// (VT) / crossing edge-edge (EE) pair.  The midphase emits ONLY the integer contact
// record (pairInd0 / pairInd1 / auxInd — the pair-type + validity bits and, for EE,
// the local edge indices); the normal / distance / barycentric are computed later in
// the (out-of-scope) self-contact prepare.  FULLY bit-exact: the VT/EE closest-point
// tests are pure mul/add/sub/div/sqrt/min/max — no transcendental, no tolerance seam.
// Checked bit-exact vs a CPU transcription of the .cu math in BOTH memory modes.
// ===========================================================================

// PxgFemFemContactInfo host mirror (PxgFEMCore.h:51, PX_ALIGN(16), 16 bytes).
// PxgFEMCore.h is not test-includable (pulls the CUDA buffer stack), so the layout
// the shader writes at index*16 is mirrored locally and pinned self-consistently.
struct FemFemContactInfoHost
{
    uint64_t pairInd0;   // @0
    uint32_t pairInd1;   // @8
    uint32_t auxInd;     // @12 -> 16
};
static_assert(sizeof(FemFemContactInfoHost) == 16, "PxgFemFemContactInfo mirror == 16B");
static_assert(offsetof(FemFemContactInfoHost, pairInd0) == 0, "pairInd0 @0");
static_assert(offsetof(FemFemContactInfoHost, pairInd1) == 8, "pairInd1 @8");
static_assert(offsetof(FemFemContactInfoHost, auxInd) == 12, "auxInd @12");

// self-collision push blob — byte-identical to SelfCollisionParams
// (ABI row {4,8,8,8,8,4,8,8,8} -> 72 bytes).
struct SelfCollisionPush
{
    uint32_t maxContacts, pad4;
    uint64_t activeClothes, contactDistance, clothes, clothClothPairs;
    uint32_t numClothClothPairs, pad44;
    uint64_t updateContactPairs, outContactInfo, totalContactCount;
};
static_assert(sizeof(SelfCollisionPush) == 72, "self-collision push blob must be 72 bytes");
static_assert(offsetof(SelfCollisionPush, updateContactPairs) == 48, "updateContactPairs @48");
static_assert(offsetof(SelfCollisionPush, totalContactCount) == 64, "totalContactCount @64");

// A four-uint triangle-index record (uint4: x/y/z vertex ids, w = edge authorship).
struct TriIdx4 { uint32_t x, y, z, w; };

// PxgFemFemContactInfo::auxInd bit packing (PxgFEMCore.h:69-82) — the exact constants
// the shader uses, pinned here so the CPU reference builds the same word.
static const uint32_t AUX_IND0_BITS_H = 14u;
static const uint32_t AUX_IND1_BITS_H = 15u;
static const uint32_t AUX_IND0_MASK_H = (1u << AUX_IND0_BITS_H) - 1u;
static const uint32_t AUX_IND1_MASK_H = ((1u << AUX_IND1_BITS_H) - 1u) << AUX_IND0_BITS_H;
static const uint32_t PAIR_TYPE_FLAG_MASK_H = 1u << 30;

// Gu::BV32DataPacked (1168 bytes) with a SINGLE leaf child covering all nbTriangles
// triangles (primStart 0), boxed by [boxLo, boxHi]^3.  The buffer is prefixed with a
// 16-byte header (sizeof(uint4)) the kernel skips (bv32PackedNodes = mTriMeshData+16).
static std::vector<uint32_t> makeBv32TreeWords(uint32_t nbTriangles, float boxLo, float boxHi)
{
    std::vector<uint32_t> w(296, 0u); // 4 header words + 292 node words = 1184 bytes
    const uint32_t node = 4u;         // node begins at byte 16
    // mMin[0].xyz @node ; mMax[0].xyz @node+128 (byte +512) ; mData[0] @node+256 (byte +1024)
    w[node + 0] = f2u(boxLo); w[node + 1] = f2u(boxLo); w[node + 2] = f2u(boxLo);
    w[node + 128 + 0] = f2u(boxHi); w[node + 128 + 1] = f2u(boxHi); w[node + 128 + 2] = f2u(boxHi);
    w[node + 256] = (0u << 7) | (nbTriangles << 1) | 1u;  // leaf: primStart 0, nbPrims, isLeaf
    w[node + 288] = 1u;                                    // mNbNodes @byte 1152
    w[node + 289] = 1u;                                    // mDepth  @byte 1156
    return w;
}

// ---------------------------------------------------------------------------
// Bit-exact CPU transcription of the VT / EE self-collision predicates
// (femClothClothMidPhase.cu ClothTree{Vertex,Edge}Traverser + deformableCollision.cuh
// closestPtPointTriangle / closestPtLineLine).  Built -ffp-contract=off (matching the
// NoContraction shader stamp).
// ---------------------------------------------------------------------------
namespace p72ref
{
struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static float fmin3(float a, float b) { return a < b ? a : b; }
static float fmax3(float a, float b) { return a > b ? a : b; }
static V3 vmin(V3 a, V3 b) { return V3{ fmin3(a.x, b.x), fmin3(a.y, b.y), fmin3(a.z, b.z) }; }
static V3 vmax(V3 a, V3 b) { return V3{ fmax3(a.x, b.x), fmax3(a.y, b.y), fmax3(a.z, b.z) }; }

static const float BARY_THR = 1.0e-6f;
static const float ONE_MINUS_BARY_THR = 1.0f - 1.0e-6f;

static V3 closestPtPointTriangle(V3 p, V3 a, V3 b, V3 c, float& s, float& t)
{
    const V3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    const float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) { s = 0.0f; t = 0.0f; return a; }
    const V3 bp = sub(p, b);
    const float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) { s = 1.0f; t = 0.0f; return b; }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) { const float v = d1 / (d1 - d3); s = v; t = 0.0f; return add(a, mul(ab, v)); }
    const V3 cp = sub(p, c);
    const float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) { s = 0.0f; t = 1.0f; return c; }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) { const float wv = d2 / (d2 - d6); s = 0.0f; t = wv; return add(a, mul(ac, wv)); }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    { const float wv = (d4 - d3) / ((d4 - d3) + (d5 - d6)); s = 1.0f - wv; t = wv; return add(b, mul(sub(c, b), wv)); }
    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom, wv = vc * denom;
    s = v; t = wv;
    return add(a, add(mul(ab, v), mul(ac, wv)));
}

// closestPtLineLine (deformableCollision.cuh:244) — matches npEdgeEdge.slang:57.  The
// vector/scalar `r0 / r0Len` is transcribed as PER-COMPONENT division (divv), matching
// the shader's `float3 / float` (3x OpFDiv at -O0), NOT a reciprocal-multiply.
static V3 divv(V3 a, float s) { return V3{ a.x / s, a.y / s, a.z / s }; }
static void closestPtLineLine(V3 e0v0, V3 e0v1, V3 e1v0, V3 e1v1, float& s, float& t, float& distSq)
{
    const V3 r0 = sub(e0v1, e0v0), r1 = sub(e1v1, e1v0), d = sub(e1v0, e0v0);
    const float r0Len2 = dot(r0, r0), r1Len2 = dot(r1, r1);
    if (r0Len2 < 1.0e-18f || r1Len2 < 1.0e-18f) { s = 0.0f; t = 0.0f; distSq = dot(d, d); return; }
    const float r0Len = std::sqrt(r0Len2), r1Len = std::sqrt(r1Len2);
    const V3 u0 = divv(r0, r0Len), u1 = divv(r1, r1Len);
    const float cosTheta = std::fabs(dot(u0, u1));
    if (cosTheta > 0.9999f)
    {
        const float tProj = dot(sub(e0v0, e1v0), r1) / r1Len2;
        t = tProj; s = 0.0f;
        const V3 pt1 = add(e1v0, mul(r1, tProj));
        distSq = dot(sub(pt1, e0v0), sub(pt1, e0v0));
        return;
    }
    const float a01 = -dot(u0, u1);
    const float b0 = dot(u0, d) / r0Len, b1 = -dot(u1, d) / r1Len;
    const float det = 1.0f - a01 * a01;
    if (det < 1.0e-9f) { s = 0.0f; t = 0.0f; distSq = dot(d, d); return; }
    const float invDet = 1.0f / det;
    s = (b0 - b1 * a01) * invDet;
    t = (b1 - a01 * b0) * invDet;
    const V3 pt0 = add(e0v0, mul(r0, s)), pt1 = add(e1v0, mul(r1, t));
    distSq = dot(sub(pt1, pt0), sub(pt1, pt0));
}
} // namespace p72ref

// Build a single-active-cloth self-collision descriptor over the given buffers.
static SelfCollisionPush makeSelfCollisionPush(uint32_t maxContacts, uint64_t activeClothes,
                                               uint64_t contactDistance, uint64_t clothes,
                                               uint64_t updateContactPairs, uint64_t outContactInfo,
                                               uint64_t totalContactCount)
{
    SelfCollisionPush p = {};
    p.maxContacts = maxContacts;
    p.activeClothes = activeClothes;
    p.contactDistance = contactDistance;
    p.clothes = clothes;
    p.clothClothPairs = 0ull;      // unused by the traversers (ABI parity)
    p.numClothClothPairs = 0u;
    p.updateContactPairs = updateContactPairs;
    p.outContactInfo = outContactInfo;
    p.totalContactCount = totalContactCount;
    return p;
}

// ---------------------------------------------------------------------------
// cloth_selfCollisionMidphaseVTLaunch — one cloth vertex penetrating a triangle.
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimSelfCollisionVT)
{
    PxgSlangComputeDevice* dev = testDevice();

    // Triangle T0 = {0,1,2} in the z=0 plane; vertex 3 hovers 0.05 above, projecting
    // inside T0 (a VT self-contact); vertex 3 is NOT a member of T0.
    std::vector<F4> pos = {
        { 0.00f, 0.00f, 0.00f, 1.0f },   // v0 (T0)
        { 1.00f, 0.00f, 0.00f, 1.0f },   // v1 (T0)
        { 0.00f, 1.00f, 0.00f, 1.0f },   // v2 (T0)
        { 0.25f, 0.25f, 0.05f, 1.0f },   // v3 penetrating
    };
    std::vector<F4> rest = pos; // distinct rest positions -> rest-pose filter (dist>0) passes
    std::vector<TriIdx4> tris = { { 0u, 1u, 2u, 0x7u } }; // edge authorship 0x7 (VT ignores .w)
    std::vector<uint32_t> tree = makeBv32TreeWords(/*nbTriangles*/ 1u, -1000.0f, 1000.0f);

    DevBuf bPos = makeDev(dev, pos);
    DevBuf bRest = makeDev(dev, rest);
    DevBuf bTri = makeDev(dev, tris);
    DevBuf bTree = makeDev(dev, tree);

    PxgFEMCloth c = makeZeroDesc();
    c.mTriMeshData = reinterpret_cast<void*>(size_t(bTree.addr));
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bPos.addr));
    c.mRestPosition = reinterpret_cast<decltype(c.mRestPosition)>(size_t(bRest.addr));
    c.mTriangleVertexIndices = reinterpret_cast<decltype(c.mTriangleVertexIndices)>(size_t(bTri.addr));
    c.mNbVerts = 4u;
    c.mElementIndex = 0u;
    c.mSelfCollisionFilterDistance = 0.0f; // filterDistSq 0 -> any positive rest distance passes
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);
    std::vector<float> cd = { 0.05f };            // contactDist = 2*0.05 = 0.1 -> contactDistSq 0.01
    DevBuf bCd = makeDev(dev, cd);
    std::vector<uint32_t> upd = { 1u };
    DevBuf bUpd = makeDev(dev, upd);

    const uint32_t MAXC = 64u;
    std::vector<FemFemContactInfoHost> outInfo(MAXC);
    std::memset(outInfo.data(), 0xAB, outInfo.size() * sizeof(FemFemContactInfoHost)); // poison
    DevBuf bOut = makeDev(dev, outInfo);
    std::vector<uint32_t> count = { 0u };
    DevBuf bCount = makeDev(dev, count);

    if (!bPos.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    SelfCollisionPush push = makeSelfCollisionPush(MAXC, bActive.addr, bCd.addr, bDesc.addr,
                                                   bUpd.addr, bOut.addr, bCount.addr);
    Kernel k("femClothSelfCollisionBda.spv", "cloth_selfCollisionMidphaseVTLaunch", {}, sizeof(SelfCollisionPush));
    // GRID MIRRORS THE ENGINE, and that is the whole point of this line.  The engine
    // packs MIDPHASE_WARPS_PER_BLOCK (=2) warps into one block and launches only
    // ceil(N/2) blocks (PxgFEMClothCore.cpp:714); the bridge drops blockDim and passes
    // gridDim straight through.  A test that dispatched one group per query would
    // exercise a grid the engine NEVER issues -- which is exactly how a half-coverage
    // bug survived here: the shader indexed by gid.x alone and this line handed it the
    // generous grid that made that correct.  Unit tests choose their own grid, so this
    // is the ONLY place that mismatch can be caught.
    k.dispatch(/*gridX ceil(numVerts/2)*/ 2u, {}, push, /*gridY nbActive*/ 1u, 1u);
    dev->flush();

    // ---- CPU predicate: exactly vertex 3 penetrates T0 ----
    using p72ref::V3;
    auto v3f = [&](const F4& f) { return V3{ f.x, f.y, f.z }; };
    const float contactDist = 2.0f * cd[0];
    const float contactDistSq = contactDist * contactDist;
    const V3 x1 = v3f(pos[0]), x2 = v3f(pos[1]), x3 = v3f(pos[2]), vtx = v3f(pos[3]);
    const V3 bMin = p72ref::sub(p72ref::vmin(p72ref::vmin(x1, x2), x3), V3{ contactDist, contactDist, contactDist });
    const V3 bMax = p72ref::add(p72ref::vmax(p72ref::vmax(x1, x2), x3), V3{ contactDist, contactDist, contactDist });
    const bool contains = !(vtx.x < bMin.x || vtx.x > bMax.x || vtx.y < bMin.y || vtx.y > bMax.y || vtx.z < bMin.z || vtx.z > bMax.z);
    float s, t; const V3 cp = p72ref::closestPtPointTriangle(vtx, x1, x2, x3, s, t);
    const V3 dir = p72ref::sub(vtx, cp);
    const float distSq = p72ref::dot(dir, dir);
    const bool penetrates = contains && (distSq < contactDistSq) && (distSq > 1.0e-14f);
    PXG_CHECK_MSG(penetrates, "reference: vertex 3 must penetrate T0 (distSq=%g < %g)", distSq, contactDistSq);

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
    PXG_CHECK_MSG(gCount == 1u, "VT self-collision emitted %u contacts (expected 1)", gCount);

    const FemFemContactInfoHost gi = downloadBuffer<FemFemContactInfoHost>(dev, bOut.h, 1)[0];
    size_t mm = 0;
    const uint64_t refPair0 = (0ull << 20) | 3ull;   // encode(cloth 0, vertex 3)
    const uint32_t refPair1 = (0u << 20) | 0u;        // encode(cloth 0, triangle 0)
    const uint32_t refAux = 0u;                       // VT pair (bit30=0), invalid (self, bit31=0), zero-init
    if (gi.pairInd0 != refPair0) { ++mm; std::printf("  VT pairInd0 MM got %llu ref %llu\n", (unsigned long long)gi.pairInd0, (unsigned long long)refPair0); }
    if (gi.pairInd1 != refPair1) { ++mm; std::printf("  VT pairInd1 MM got %u ref %u\n", gi.pairInd1, refPair1); }
    if (gi.auxInd != refAux) { ++mm; std::printf("  VT auxInd MM got 0x%08x ref 0x%08x\n", gi.auxInd, refAux); }
    PXG_CHECK_MSG(mm == 0, "cloth_selfCollisionMidphaseVT bit-exact vs CPU (%zu mismatches)", mm);

    // ---- non-penetrating case: lift vertex 3 above the fattened bound -> zero contacts ----
    std::vector<F4> posOut = pos;
    posOut[3] = F4{ 0.25f, 0.25f, 0.5f, 1.0f }; // z=0.5 > 0.2 fattened bound -> not contained
    DevBuf bPosOut = makeDev(dev, posOut);
    PxgFEMCloth c2 = c;
    c2.mPosition_InvMass = reinterpret_cast<decltype(c2.mPosition_InvMass)>(size_t(bPosOut.addr));
    std::vector<PxgFEMCloth> descArr2 = { c2 };
    DevBuf bDesc2 = makeDev(dev, descArr2);
    std::vector<uint32_t> count2 = { 0u };
    DevBuf bCount2 = makeDev(dev, count2);
    SelfCollisionPush push2 = push;
    push2.clothes = bDesc2.addr;
    push2.totalContactCount = bCount2.addr;
    k.dispatch(/*ceil(numVerts/2)*/ 2u, {}, push2, 1u, 1u);
    dev->flush();
    const uint32_t gCount2 = downloadBuffer<uint32_t>(dev, bCount2.h, 1)[0];
    PXG_CHECK_MSG(gCount2 == 0u, "VT self-collision (no penetration) emitted %u contacts (expected 0)", gCount2);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bRest.h); dev->destroyBuffer(bTri.h);
    dev->destroyBuffer(bTree.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
    dev->destroyBuffer(bCd.h); dev->destroyBuffer(bUpd.h); dev->destroyBuffer(bOut.h); dev->destroyBuffer(bCount.h);
    dev->destroyBuffer(bPosOut.h); dev->destroyBuffer(bDesc2.h); dev->destroyBuffer(bCount2.h);
}

// ---------------------------------------------------------------------------
// cloth_selfCollisionMidphaseEELaunch — two triangles whose edges cross.
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimSelfCollisionEE)
{
    PxgSlangComputeDevice* dev = testDevice();

    // T0 = {0,1,2}: edge0 (v0->v1) is the x-axis segment at (y=0,z=0); its other two
    //   edges dive to v2=(0,0,-1) (far from T1).
    // T1 = {3,4,5} at z=0.03: edge0 (v3->v4) and edge2 (v5->v3) BOTH cross the x-axis
    //   within contactDist (two EE crossings); edge1 (v4->v5) stays at y=-0.3 (far).
    std::vector<F4> pos = {
        { -0.5f, 0.0f,  0.0f, 1.0f },   // v0 (T0)
        {  0.5f, 0.0f,  0.0f, 1.0f },   // v1 (T0)
        {  0.0f, 0.0f, -1.0f, 1.0f },   // v2 (T0)
        {  0.0f,  0.3f, 0.03f, 1.0f },  // v3 (T1)
        {  0.2f, -0.3f, 0.03f, 1.0f },  // v4 (T1)
        { -0.2f, -0.3f, 0.03f, 1.0f },  // v5 (T1)
    };
    std::vector<F4> rest = pos;
    std::vector<TriIdx4> tris = { { 0u, 1u, 2u, 0x7u }, { 3u, 4u, 5u, 0x7u } }; // all edges active
    std::vector<uint32_t> twae = { 0u, 1u }; // both triangles have active edges
    std::vector<uint32_t> tree = makeBv32TreeWords(/*nbTriangles*/ 2u, -1000.0f, 1000.0f);

    DevBuf bPos = makeDev(dev, pos);
    DevBuf bRest = makeDev(dev, rest);
    DevBuf bTri = makeDev(dev, tris);
    DevBuf bTwae = makeDev(dev, twae);
    DevBuf bTree = makeDev(dev, tree);

    PxgFEMCloth c = makeZeroDesc();
    c.mTriMeshData = reinterpret_cast<void*>(size_t(bTree.addr));
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bPos.addr));
    c.mRestPosition = reinterpret_cast<decltype(c.mRestPosition)>(size_t(bRest.addr));
    c.mTriangleVertexIndices = reinterpret_cast<decltype(c.mTriangleVertexIndices)>(size_t(bTri.addr));
    c.mTrianglesWithActiveEdges = reinterpret_cast<decltype(c.mTrianglesWithActiveEdges)>(size_t(bTwae.addr));
    c.mNbTrianglesWithActiveEdges = 2u;
    c.mElementIndex = 0u;
    c.mSelfCollisionFilterDistance = 0.0f;
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);
    std::vector<float> cd = { 0.05f };            // contactDist 0.1 -> contactDistSq 0.01
    DevBuf bCd = makeDev(dev, cd);
    std::vector<uint32_t> upd = { 1u };
    DevBuf bUpd = makeDev(dev, upd);

    const uint32_t MAXC = 64u;
    std::vector<FemFemContactInfoHost> outInfo(MAXC);
    std::memset(outInfo.data(), 0xAB, outInfo.size() * sizeof(FemFemContactInfoHost));
    DevBuf bOut = makeDev(dev, outInfo);
    std::vector<uint32_t> count = { 0u };
    DevBuf bCount = makeDev(dev, count);

    if (!bPos.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    SelfCollisionPush push = makeSelfCollisionPush(MAXC, bActive.addr, bCd.addr, bDesc.addr,
                                                   bUpd.addr, bOut.addr, bCount.addr);
    Kernel k("femClothSelfCollisionBda.spv", "cloth_selfCollisionMidphaseEELaunch", {}, sizeof(SelfCollisionPush));
    // GRID MIRRORS THE ENGINE, and that is the whole point of this line.  The engine
    // packs MIDPHASE_WARPS_PER_BLOCK (=2) warps into one block and launches only
    // ceil(N/2) blocks (PxgFEMClothCore.cpp:714); the bridge drops blockDim and passes
    // gridDim straight through.  A test that dispatched one group per query would
    // exercise a grid the engine NEVER issues -- which is exactly how a half-coverage
    // bug survived here: the shader indexed by gid.x alone and this line handed it the
    // generous grid that made that correct.  Unit tests choose their own grid, so this
    // is the ONLY place that mismatch can be caught.
    k.dispatch(/*gridX ceil(numTrisWithActiveEdges/2)*/ 1u, {}, push, /*gridY nbActive*/ 1u, 1u);
    dev->flush();

    // ---- CPU transcription: the T1(query)-vs-T0(found) 3x3 edge loop ----
    // (only the higher-indexed triangle's warp processes the pair: t0_index 1 > primIdx 0.)
    using p72ref::V3;
    auto v3f = [&](const F4& f) { return V3{ f.x, f.y, f.z }; };
    const float contactDist = 2.0f * cd[0];
    const float contactDistSq = contactDist * contactDist;
    const uint32_t nextE[3] = { 1u, 2u, 0u };
    const uint32_t vIndices0[3] = { tris[1].x, tris[1].y, tris[1].z }; // query = T1
    const uint32_t vIndices1[3] = { tris[0].x, tris[0].y, tris[0].z }; // found = T0
    V3 t0pts[3] = { v3f(pos[vIndices0[0]]), v3f(pos[vIndices0[1]]), v3f(pos[vIndices0[2]]) };
    V3 t1pts[3] = { v3f(pos[vIndices1[0]]), v3f(pos[vIndices1[1]]), v3f(pos[vIndices1[2]]) };
    std::vector<FemFemContactInfoHost> refContacts;
    const uint64_t refPair0 = (0ull << 20) | 1ull;   // encode(cloth 0, t0_index 1)
    const uint32_t refPair1 = (0u << 20) | 0u;        // encode(cloth 0, primitive 0)
    for (uint32_t e1 = 0; e1 < 3; ++e1)           // edges of found T0
    {
        const uint32_t e1v0 = vIndices1[e1], e1v1 = vIndices1[nextE[e1]];
        const V3 e1x0 = t1pts[e1], e1x1 = t1pts[nextE[e1]];
        for (uint32_t e0 = 0; e0 < 3; ++e0)       // edges of query T1
        {
            const uint32_t e0v0 = vIndices0[e0], e0v1 = vIndices0[nextE[e0]];
            if (e0v0 == e1v0 || e0v0 == e1v1 || e0v1 == e1v0 || e0v1 == e1v1) continue;
            const V3 e0x0 = t0pts[e0], e0x1 = t0pts[nextE[e0]];
            float s, t, dq;
            p72ref::closestPtLineLine(e0x0, e0x1, e1x0, e1x1, s, t, dq);
            const bool isOutside = (s < p72ref::BARY_THR) || (s > p72ref::ONE_MINUS_BARY_THR) ||
                                   (t < p72ref::BARY_THR) || (t > p72ref::ONE_MINUS_BARY_THR);
            if (isOutside) continue;
            if (dq < contactDistSq && dq > 1.0e-14f)
            {
                // filterDistSq 0, distinct rest positions -> passes; build the auxInd.
                uint32_t aux = PAIR_TYPE_FLAG_MASK_H;
                aux = (aux & ~AUX_IND0_MASK_H) | (e0 & AUX_IND0_MASK_H);
                aux = (aux & ~AUX_IND1_MASK_H) | ((e1 & ((1u << AUX_IND1_BITS_H) - 1u)) << AUX_IND0_BITS_H);
                refContacts.push_back(FemFemContactInfoHost{ refPair0, refPair1, aux });
            }
        }
    }
    PXG_CHECK_MSG(refContacts.size() == 2u, "reference: expected 2 EE crossings, got %zu", refContacts.size());

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
    PXG_CHECK_MSG(gCount == uint32_t(refContacts.size()), "EE self-collision emitted %u contacts (expected %zu)",
                  gCount, refContacts.size());

    const auto gi = downloadBuffer<FemFemContactInfoHost>(dev, bOut.h, gCount ? gCount : 1u);
    size_t mm = 0;
    for (uint32_t i = 0; i < gCount && i < refContacts.size(); ++i)
    {
        if (std::memcmp(&gi[i], &refContacts[i], sizeof(FemFemContactInfoHost)) != 0)
        {
            ++mm;
            std::printf("  EE contact[%u] MM: got(p0=%llu,p1=%u,aux=0x%08x) ref(p0=%llu,p1=%u,aux=0x%08x)\n", i,
                        (unsigned long long)gi[i].pairInd0, gi[i].pairInd1, gi[i].auxInd,
                        (unsigned long long)refContacts[i].pairInd0, refContacts[i].pairInd1, refContacts[i].auxInd);
        }
    }
    PXG_CHECK_MSG(mm == 0, "cloth_selfCollisionMidphaseEE bit-exact vs CPU (%zu mismatches)", mm);

    // ---- non-crossing case: lift T1 to z=1.0 -> no edge crossing -> zero contacts ----
    std::vector<F4> posOut = pos;
    posOut[3].z = 1.0f; posOut[4].z = 1.0f; posOut[5].z = 1.0f;
    DevBuf bPosOut = makeDev(dev, posOut);
    PxgFEMCloth c2 = c;
    c2.mPosition_InvMass = reinterpret_cast<decltype(c2.mPosition_InvMass)>(size_t(bPosOut.addr));
    std::vector<PxgFEMCloth> descArr2 = { c2 };
    DevBuf bDesc2 = makeDev(dev, descArr2);
    std::vector<uint32_t> count2 = { 0u };
    DevBuf bCount2 = makeDev(dev, count2);
    SelfCollisionPush push2 = push;
    push2.clothes = bDesc2.addr;
    push2.totalContactCount = bCount2.addr;
    k.dispatch(/*ceil(numTrisWithActiveEdges/2)*/ 1u, {}, push2, 1u, 1u);
    dev->flush();
    const uint32_t gCount2 = downloadBuffer<uint32_t>(dev, bCount2.h, 1)[0];
    PXG_CHECK_MSG(gCount2 == 0u, "EE self-collision (no crossing) emitted %u contacts (expected 0)", gCount2);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bRest.h); dev->destroyBuffer(bTri.h);
    dev->destroyBuffer(bTwae.h); dev->destroyBuffer(bTree.h); dev->destroyBuffer(bDesc.h);
    dev->destroyBuffer(bActive.h); dev->destroyBuffer(bCd.h); dev->destroyBuffer(bUpd.h);
    dev->destroyBuffer(bOut.h); dev->destroyBuffer(bCount.h);
    dev->destroyBuffer(bPosOut.h); dev->destroyBuffer(bDesc2.h); dev->destroyBuffer(bCount2.h);
}

// ===========================================================================
// P7.2 tranche 2: cloth-vs-CLOTH self-contact SOLVE.
//   femClothClothContactBda.spv:
//     cloth_clothContactPrepareLaunch            (FEMClothConstraintPrep.cu:155)
//     cloth_queryClothClothContact{VT,EE}CountLaunch (FEMClothExternalSolve.cu:902/:964)
//     cloth_solveClothClothDelta{VT,EE}Launch    (FEMClothExternalSolve.cu:651/:760)
//     cloth_updateClothContactValidityLaunch     (FEMClothExternalSolve.cu:1114)
// The prepare marks a not-yet-valid self-contact valid (rest-pose distance filter); the
// query counts each referenced cloth vertex into mDeltaPos.w and marks the pair
// in-collision; the VT/EE solve scatters BOTH endpoints' corrections into mDeltaPos so the
// penetrating vertex + its triangle (VT) / the two crossing edges (EE) get pushed APART.
// Checked bit-exact vs a CPU transcription of the .cu math (PGS; the cloth self-contact
// solve has no TGS variant) in BOTH memory modes.  FULLY bit-exact: the closest-point +
// constraint math is pure mul/add/sub/div/sqrt/abs — the only transcendental (PxRecipSqrt)
// is on the friction branch, DEAD for the zero-friction self-fold target.
// ===========================================================================

// auxInd flag bits beyond the tranche-1 masks (@3235): isInCollision (bit29), Valid (bit31).
static const uint32_t CONTACT_FLAG_MASK_H  = 1u << 29;
static const uint32_t VALIDITY_FLAG_MASK_H = 1u << 31;

// Push blobs, byte-identical to femClothClothContactBda.slang's [[vk::push_constant]]
// structs (ABI rows {8,8,8,4,8,8} / {8,8,8} / {8,8,8,4} / {8,8,1,1,8,8,8,4}).
struct PrepClothClothPush
{
    uint64_t clothes, contactInfos, numContacts;
    uint32_t maxContacts, pad28;
    uint64_t clothMaterials, updateContactPairs;
};
static_assert(sizeof(PrepClothClothPush) == 48, "clothContactPrepare push blob must be 48 bytes");
static_assert(offsetof(PrepClothClothPush, clothMaterials) == 32, "clothMaterials @32");

struct QueryClothClothPush
{
    uint64_t cloths, contactInfos, numContacts;
};
static_assert(sizeof(QueryClothClothPush) == 24, "queryClothCloth push blob must be 24 bytes");

struct SolveClothClothPush
{
    uint64_t cloths, contactInfos, numContacts;
    float dt;
    uint32_t pad28;
};
static_assert(sizeof(SolveClothClothPush) == 32, "solveClothCloth push blob must be 32 bytes");

struct UpdateValidityPush
{
    uint64_t femCloths, activeIds;
    uint32_t adaptiveWord, forceWord;
    uint64_t updateContactPairs, VTContactCount, EEContactCount;
    float dt;
    uint32_t pad52;
};
static_assert(sizeof(UpdateValidityPush) == 56, "updateContactValidity push blob must be 56 bytes");
static_assert(offsetof(UpdateValidityPush, updateContactPairs) == 24, "updateContactPairs @24");
static_assert(offsetof(UpdateValidityPush, dt) == 48, "dt @48");

// ---------------------------------------------------------------------------
// Bit-exact CPU transcription of the count + solve arithmetic
// (FEMClothExternalSolve.cu countContact + triangleCollisionConstraint +
// applyContactDelta).  Reuses the p72ref vector helpers; built -ffp-contract=off.
// ---------------------------------------------------------------------------
namespace p72solve
{
using p72ref::V3;
using p72ref::sub;
using p72ref::add;
using p72ref::mul;    // mul(v, s) = { v.x*s, v.y*s, v.z*s }
using p72ref::dot;
using p72ref::divv;   // divv(v, s) = { v.x/s, v.y/s, v.z/s } (per-component)
static const float THR = 1.0e-14f;

// weights-weighted displacement (cloth0 - cloth1), left-associative like the shader.
static V3 dispFromWeights(const float w[4], V3 xx0, V3 xx1, V3 xx2, V3 xx3)
{
    return add(add(add(mul(xx0, w[0]), mul(xx1, w[1])), mul(xx2, w[2])), mul(xx3, w[3]));
}

// triangleCollisionConstraint (frictionless path): out dx[4].  invM = count-split masses.
static void constraint(V3& dx0, V3& dx1, V3& dx2, V3& dx3, V3 disp, V3 normal, const float invM[4],
                       float restDist, const float w[4])
{
    dx0 = dx1 = dx2 = dx3 = V3{ 0.0f, 0.0f, 0.0f };
    const float CN = dot(disp, normal) - restDist;
    float denom = invM[0] * w[0] * w[0];
    denom += invM[1] * w[1] * w[1];
    denom += invM[2] * w[2] * w[2];
    denom += invM[3] * w[3] * w[3];
    if (denom < THR)
        return;
    const float invDenom = 1.0f / denom;
    const float deltaLambdaN = -CN * invDenom;
    const V3 commonVec = mul(normal, deltaLambdaN);
    dx0 = mul(commonVec, invM[0] * w[0]);
    dx1 = mul(commonVec, invM[1] * w[1]);
    dx2 = mul(commonVec, invM[2] * w[2]);
    dx3 = mul(commonVec, invM[3] * w[3]);
}
} // namespace p72solve

// A self-contact reference outcome: the four endpoints' (vertexSlot, mDeltaPos.w count,
// mDeltaPos.xyz correction).
struct SelfContactRef
{
    uint32_t slot[4];
    float countW[4];
    p72ref::V3 dx[4];
    bool inCollision;
};

// Transcribe the query (countContact) + solve (constraint + applyContactDelta) for one
// self-contact given the four endpoint positions / inverse masses / weights, the normal
// and the total rest distance.  `vertInvMass[i]` == the endpoint's mPosition_InvMass.w.
static SelfContactRef refSelfContact(const uint32_t slot[4], const p72ref::V3 xx[4],
                                     const float vertInvMass[4], const float weights[4],
                                     p72ref::V3 normal, float restDistTotal)
{
    using p72ref::V3;
    SelfContactRef r;
    for (int i = 0; i < 4; ++i) { r.slot[i] = slot[i]; r.countW[i] = 0.0f; r.dx[i] = V3{ 0, 0, 0 }; }

    // countContact: mark in-collision iff distSq <= restDist^2, then bump |coeff|.
    const V3 disp = p72solve::dispFromWeights(weights, xx[0], xx[1], xx[2], xx[3]);
    const float distSq = p72solve::dot(disp, disp);
    r.inCollision = !(distSq > restDistTotal * restDistTotal);
    if (!r.inCollision)
        return r;
    for (int i = 0; i < 4; ++i)
    {
        const float coeff = std::fabs(weights[i]);
        if (coeff > 1e-3f)
            r.countW[i] = coeff;   // one contact -> distinct slots, so no accumulation
    }

    // solve: split masses by the query counts, run the constraint, scatter if count > 0.
    float splitted[4];
    for (int i = 0; i < 4; ++i)
        splitted[i] = r.countW[i] * vertInvMass[i];
    V3 dx0, dx1, dx2, dx3;
    p72solve::constraint(dx0, dx1, dx2, dx3, disp, normal, splitted, restDistTotal, weights);
    const V3 dxs[4] = { dx0, dx1, dx2, dx3 };
    for (int i = 0; i < 4; ++i)
        if (r.countW[i] > 0.0f)
            r.dx[i] = dxs[i];
    return r;
}

// cloth_clothContactPrepare -> query -> solve: push a penetrating VT self-contact apart.
static void runClothSelfContactVT()
{
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 1.0f / 60.0f;
    const float invMass = 0.5f;
    const float restDistScalar = 0.05f;   // total rest distance = 0.1

    // Triangle T0 = {0,1,2} in a TILTED plane (so the contact normal is off-axis and no
    // delta component lands on an exact signed zero — the femClothSimRigidContact
    // precedent); vertex 3 hovers 0.05 above the centroid along the triangle normal,
    // projecting to the interior (barycentric ~1/3,1/3,1/3).
    std::vector<F4> pos = {
        { 0.00f, 0.00f, 0.00f, invMass },              // v0 (T0)
        { 1.00f, 0.00f, 0.20f, invMass },              // v1 (T0) tilted
        { 0.00f, 1.00f, 0.10f, invMass },              // v2 (T0) tilted
        { 0.3235743f, 0.3284538f, 0.1487950f, invMass },  // v3 penetrating vertex
    };
    std::vector<F4> rest = pos;   // distinct rest -> rest-pose filter passes
    std::vector<F4> prev = pos;   // prevPositionInContactOffset == pos -> relDelta 0
    std::vector<F4> deltaPos(4, F4{ 0, 0, 0, 0 });
    std::vector<float> friction(4, 0.0f);   // frictionless -> the tangent branch is dead
    std::vector<TriIdx4> tris = { { 0u, 1u, 2u, 0u } };

    DevBuf bPos = makeDev(dev, pos);
    DevBuf bRest = makeDev(dev, rest);
    DevBuf bPrev = makeDev(dev, prev);
    DevBuf bDelta = makeDev(dev, deltaPos);
    DevBuf bFric = makeDev(dev, friction);
    DevBuf bTri = makeDev(dev, tris);

    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bPos.addr));
    c.mRestPosition = reinterpret_cast<decltype(c.mRestPosition)>(size_t(bRest.addr));
    c.mPrevPositionInContactOffset = reinterpret_cast<decltype(c.mPrevPositionInContactOffset)>(size_t(bPrev.addr));
    c.mDeltaPos = reinterpret_cast<decltype(c.mDeltaPos)>(size_t(bDelta.addr));
    c.mDynamicFrictions = reinterpret_cast<decltype(c.mDynamicFrictions)>(size_t(bFric.addr));
    c.mTriangleVertexIndices = reinterpret_cast<decltype(c.mTriangleVertexIndices)>(size_t(bTri.addr));
    c.mNbVerts = 4u;
    c.mSelfCollisionFilterDistance = 0.0f;
    c.mRestDistance = restDistScalar;
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    // contactInfo: pairInd0 = encode(0,3) (vertex), pairInd1 = encode(0,0) (triangle),
    // auxInd = VT (bit30=0), NOT-yet-valid (bit31=0).
    std::vector<FemFemContactInfoHost> infos(1);
    infos[0].pairInd0 = (0ull << 20) | 3ull;
    infos[0].pairInd1 = (0u << 20) | 0u;
    infos[0].auxInd = 0u;
    DevBuf bInfos = makeDev(dev, infos);
    std::vector<uint32_t> numContacts = { 1u };
    DevBuf bNum = makeDev(dev, numContacts);
    std::vector<uint32_t> upd = { 1u };
    DevBuf bUpd = makeDev(dev, upd);

    if (!bPos.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    const uint32_t MAXC = 64u;

    // ---- prepare ----
    PrepClothClothPush pp = {};
    pp.clothes = bDesc.addr; pp.contactInfos = bInfos.addr; pp.numContacts = bNum.addr;
    pp.maxContacts = MAXC; pp.clothMaterials = 0ull; pp.updateContactPairs = bUpd.addr;
    Kernel kPrep("femClothClothContactBda.spv", "cloth_clothContactPrepareLaunch", {}, sizeof(PrepClothClothPush));
    kPrep.dispatch(1u, {}, pp, 1u, 1u);
    dev->flush();
    const FemFemContactInfoHost giPrep = downloadBuffer<FemFemContactInfoHost>(dev, bInfos.h, 1)[0];
    PXG_CHECK_MSG((giPrep.auxInd & VALIDITY_FLAG_MASK_H) != 0u,
                  "prepare must mark the VT self-contact valid (auxInd=0x%08x)", giPrep.auxInd);

    // ---- query VT (count) ----
    QueryClothClothPush qp = {};
    qp.cloths = bDesc.addr; qp.contactInfos = bInfos.addr; qp.numContacts = bNum.addr;
    Kernel kQuery("femClothClothContactBda.spv", "cloth_queryClothClothContactVTCountLaunch", {}, sizeof(QueryClothClothPush));
    kQuery.dispatch(1u, {}, qp, 1u, 1u);
    dev->flush();

    // ---- CPU reference: closestPtPointTriangle(p=v3, a=v0, b=v1, c=v2) -> weights ----
    using p72ref::V3;
    auto v3f = [&](const F4& f) { return V3{ f.x, f.y, f.z }; };
    float s, t;
    p72ref::closestPtPointTriangle(v3f(pos[3]), v3f(pos[0]), v3f(pos[1]), v3f(pos[2]), s, t);
    const float rBary = 1.0f - s - t;
    const float weightsVT[4] = { 1.0f, -rBary, -s, -t };
    const V3 xxVT[4] = { v3f(pos[3]), v3f(pos[0]), v3f(pos[1]), v3f(pos[2]) };
    // normal = disp / dist (per-component division; VT path).
    const V3 dispVT = p72solve::dispFromWeights(weightsVT, xxVT[0], xxVT[1], xxVT[2], xxVT[3]);
    const float distVT = std::sqrt(p72solve::dot(dispVT, dispVT));
    const V3 normalVT = p72ref::divv(dispVT, distVT);
    const float restTotal = restDistScalar + restDistScalar;
    const uint32_t slotVT[4] = { 3u, 0u, 1u, 2u };   // vertex, tri.x, tri.y, tri.z
    const float vimVT[4] = { invMass, invMass, invMass, invMass };
    const SelfContactRef ref = refSelfContact(slotVT, xxVT, vimVT, weightsVT, normalVT, restTotal);
    PXG_CHECK_MSG(ref.inCollision, "reference: the VT self-contact must be in-collision");

    // per-slot reference mDeltaPos after query (.w) — check the count pass.
    F4 refDelta[4] = { F4{ 0, 0, 0, 0 }, F4{ 0, 0, 0, 0 }, F4{ 0, 0, 0, 0 }, F4{ 0, 0, 0, 0 } };
    for (int i = 0; i < 4; ++i) refDelta[ref.slot[i]].w = ref.countW[i];

    const auto gDeltaQ = downloadBuffer<F4>(dev, bDelta.h, 4);
    const FemFemContactInfoHost giQ = downloadBuffer<FemFemContactInfoHost>(dev, bInfos.h, 1)[0];
    size_t mmQ = 0;
    for (uint32_t v = 0; v < 4; ++v)
        if (f2u(gDeltaQ[v].w) != f2u(refDelta[v].w))
        { ++mmQ; std::printf("  VT query mDeltaPos[%u].w MM: ref %g got %g\n", v, refDelta[v].w, gDeltaQ[v].w); }
    PXG_CHECK_MSG((giQ.auxInd & CONTACT_FLAG_MASK_H) != 0u, "query must mark the VT contact in-collision");
    PXG_CHECK_MSG(mmQ == 0, "cloth_queryClothClothContactVTCount bit-exact vs CPU (%zu mismatches)", mmQ);

    // ---- solve VT ----
    SolveClothClothPush sp = {};
    sp.cloths = bDesc.addr; sp.contactInfos = bInfos.addr; sp.numContacts = bNum.addr; sp.dt = dt;
    Kernel kSolve("femClothClothContactBda.spv", "cloth_solveClothClothDeltaVTLaunch", {}, sizeof(SolveClothClothPush));
    kSolve.dispatch(1u, {}, sp, 1u, 1u);
    dev->flush();

    for (int i = 0; i < 4; ++i)
    {
        refDelta[ref.slot[i]].x = ref.dx[i].x;
        refDelta[ref.slot[i]].y = ref.dx[i].y;
        refDelta[ref.slot[i]].z = ref.dx[i].z;
    }

    const auto gDelta = downloadBuffer<F4>(dev, bDelta.h, 4);
    size_t mm = 0;
    for (uint32_t v = 0; v < 4; ++v)
        if (f2u(gDelta[v].x) != f2u(refDelta[v].x) || f2u(gDelta[v].y) != f2u(refDelta[v].y) ||
            f2u(gDelta[v].z) != f2u(refDelta[v].z) || f2u(gDelta[v].w) != f2u(refDelta[v].w))
        { ++mm; std::printf("  VT solve mDeltaPos[%u] MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", v,
                            refDelta[v].x, refDelta[v].y, refDelta[v].z, refDelta[v].w,
                            gDelta[v].x, gDelta[v].y, gDelta[v].z, gDelta[v].w); }

    // teeth: the vertex is pushed OUT along +normal and the triangle vertices are pushed
    // along -normal -> the vertex-triangle penetration is resolved (both endpoints move apart).
    auto dotN = [](const F4& d, const V3& nn) { return d.x * nn.x + d.y * nn.y + d.z * nn.z; };
    PXG_CHECK_MSG(dotN(gDelta[3], normalVT) > 0.0f,
                  "solve must push the penetrating VERTEX out along +normal (dot %g)", dotN(gDelta[3], normalVT));
    PXG_CHECK_MSG(dotN(gDelta[0], normalVT) < 0.0f && dotN(gDelta[1], normalVT) < 0.0f && dotN(gDelta[2], normalVT) < 0.0f,
                  "solve must push the TRIANGLE along -normal (dots %g,%g,%g)",
                  dotN(gDelta[0], normalVT), dotN(gDelta[1], normalVT), dotN(gDelta[2], normalVT));
    PXG_CHECK_MSG(mm == 0, "cloth_solveClothClothDeltaVT bit-exact vs CPU (%zu mismatches)", mm);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bRest.h); dev->destroyBuffer(bPrev.h);
    dev->destroyBuffer(bDelta.h); dev->destroyBuffer(bFric.h); dev->destroyBuffer(bTri.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bInfos.h); dev->destroyBuffer(bNum.h); dev->destroyBuffer(bUpd.h);
}

// cloth_clothContactPrepare -> query -> solve: push a penetrating EE self-contact apart.
static void runClothSelfContactEE()
{
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 1.0f / 60.0f;
    const float invMass = 0.5f;
    const float restDistScalar = 0.05f;   // total rest distance = 0.1

    // Two perpendicular crossing edges with a 0.03 gap, ROTATED off-axis (Rz30.Ry25.Rx20)
    // so the contact normal has no zero component (the femClothSimRigidContact precedent —
    // an axis-aligned separation would land the transverse delta components on signed zero).
    // T_a = {0,1,2} (index 0): edge0 = v0->v1.  T_b = {3,4,5} (index 1): edge0 = v3->v4.
    std::vector<F4> pos = {
        { -0.3924428f, -0.2265770f,  0.2113091f, invMass },  // v0 (T_a, edge1 v0)
        {  0.3924428f,  0.2265770f, -0.2113091f, invMass },  // v1 (T_a, edge1 v1)
        { -0.5149359f,  0.0976325f, -0.8516507f, invMass },  // v2 (T_a, far)
        { -0.0879522f,  0.2628919f,  0.1185422f, invMass },  // v3 (T_b, edge0 v0)
        {  0.1188483f, -0.2687499f, -0.0674431f, invMass },  // v4 (T_b, edge0 v1)
        {  0.5149359f, -0.0976325f,  0.8516507f, invMass },  // v5 (T_b, far)
    };
    std::vector<F4> rest = pos;
    std::vector<F4> prev = pos;
    std::vector<F4> deltaPos(6, F4{ 0, 0, 0, 0 });
    std::vector<float> friction(6, 0.0f);
    std::vector<TriIdx4> tris = { { 0u, 1u, 2u, 0x7u }, { 3u, 4u, 5u, 0x7u } };

    DevBuf bPos = makeDev(dev, pos);
    DevBuf bRest = makeDev(dev, rest);
    DevBuf bPrev = makeDev(dev, prev);
    DevBuf bDelta = makeDev(dev, deltaPos);
    DevBuf bFric = makeDev(dev, friction);
    DevBuf bTri = makeDev(dev, tris);

    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bPos.addr));
    c.mRestPosition = reinterpret_cast<decltype(c.mRestPosition)>(size_t(bRest.addr));
    c.mPrevPositionInContactOffset = reinterpret_cast<decltype(c.mPrevPositionInContactOffset)>(size_t(bPrev.addr));
    c.mDeltaPos = reinterpret_cast<decltype(c.mDeltaPos)>(size_t(bDelta.addr));
    c.mDynamicFrictions = reinterpret_cast<decltype(c.mDynamicFrictions)>(size_t(bFric.addr));
    c.mTriangleVertexIndices = reinterpret_cast<decltype(c.mTriangleVertexIndices)>(size_t(bTri.addr));
    c.mNbVerts = 6u;
    c.mSelfCollisionFilterDistance = 0.0f;
    c.mRestDistance = restDistScalar;
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    // contactInfo: pairInd0 = encode(0,1) (triangle0 = T_b, edge0 index 0),
    // pairInd1 = encode(0,0) (triangle1 = T_a, edge index 0), auxInd = EE (bit30), aux0=aux1=0,
    // NOT-yet-valid.
    std::vector<FemFemContactInfoHost> infos(1);
    infos[0].pairInd0 = (0ull << 20) | 1ull;
    infos[0].pairInd1 = (0u << 20) | 0u;
    infos[0].auxInd = PAIR_TYPE_FLAG_MASK_H;   // bit30 = EE, aux0=aux1=0
    DevBuf bInfos = makeDev(dev, infos);
    std::vector<uint32_t> numContacts = { 1u };
    DevBuf bNum = makeDev(dev, numContacts);
    std::vector<uint32_t> upd = { 1u };
    DevBuf bUpd = makeDev(dev, upd);

    if (!bPos.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    const uint32_t MAXC = 64u;

    // ---- prepare ----
    PrepClothClothPush pp = {};
    pp.clothes = bDesc.addr; pp.contactInfos = bInfos.addr; pp.numContacts = bNum.addr;
    pp.maxContacts = MAXC; pp.clothMaterials = 0ull; pp.updateContactPairs = bUpd.addr;
    Kernel kPrep("femClothClothContactBda.spv", "cloth_clothContactPrepareLaunch", {}, sizeof(PrepClothClothPush));
    kPrep.dispatch(1u, {}, pp, 1u, 1u);
    dev->flush();
    const FemFemContactInfoHost giPrep = downloadBuffer<FemFemContactInfoHost>(dev, bInfos.h, 1)[0];
    PXG_CHECK_MSG((giPrep.auxInd & VALIDITY_FLAG_MASK_H) != 0u,
                  "prepare must mark the EE self-contact valid (auxInd=0x%08x)", giPrep.auxInd);

    // ---- query EE (count) ----
    QueryClothClothPush qp = {};
    qp.cloths = bDesc.addr; qp.contactInfos = bInfos.addr; qp.numContacts = bNum.addr;
    Kernel kQuery("femClothClothContactBda.spv", "cloth_queryClothClothContactEECountLaunch", {}, sizeof(QueryClothClothPush));
    kQuery.dispatch(1u, {}, qp, 1u, 1u);
    dev->flush();

    // ---- CPU reference: closestPtLineLine(edge0 = v3->v4, edge1 = v0->v1) -> weights ----
    using p72ref::V3;
    auto v3f = [&](const F4& f) { return V3{ f.x, f.y, f.z }; };
    // vertIndices = (T_b[0]=3, T_b[1]=4, T_a[0]=0, T_a[1]=1).
    const uint32_t slotEE[4] = { 3u, 4u, 0u, 1u };
    const V3 xxEE[4] = { v3f(pos[3]), v3f(pos[4]), v3f(pos[0]), v3f(pos[1]) };
    float s, t, distSq;
    p72ref::closestPtLineLine(xxEE[0], xxEE[1], xxEE[2], xxEE[3], s, t, distSq);
    const float weightsEE[4] = { 1.0f - s, s, -(1.0f - t), -t };
    // normal = disp * (1/dist) (reciprocal-multiply; EE path).
    const V3 dispEE = p72solve::dispFromWeights(weightsEE, xxEE[0], xxEE[1], xxEE[2], xxEE[3]);
    const float distEE = std::sqrt(distSq);
    const V3 normalEE = p72ref::mul(dispEE, 1.0f / distEE);
    const float restTotal = restDistScalar + restDistScalar;
    const float vimEE[4] = { invMass, invMass, invMass, invMass };
    const SelfContactRef ref = refSelfContact(slotEE, xxEE, vimEE, weightsEE, normalEE, restTotal);
    PXG_CHECK_MSG(ref.inCollision, "reference: the EE self-contact must be in-collision");

    F4 refDelta[6];
    for (int v = 0; v < 6; ++v) refDelta[v] = F4{ 0, 0, 0, 0 };
    for (int i = 0; i < 4; ++i) refDelta[ref.slot[i]].w = ref.countW[i];

    const auto gDeltaQ = downloadBuffer<F4>(dev, bDelta.h, 6);
    const FemFemContactInfoHost giQ = downloadBuffer<FemFemContactInfoHost>(dev, bInfos.h, 1)[0];
    size_t mmQ = 0;
    for (uint32_t v = 0; v < 6; ++v)
        if (f2u(gDeltaQ[v].w) != f2u(refDelta[v].w))
        { ++mmQ; std::printf("  EE query mDeltaPos[%u].w MM: ref %g got %g\n", v, refDelta[v].w, gDeltaQ[v].w); }
    PXG_CHECK_MSG((giQ.auxInd & CONTACT_FLAG_MASK_H) != 0u, "query must mark the EE contact in-collision");
    PXG_CHECK_MSG(mmQ == 0, "cloth_queryClothClothContactEECount bit-exact vs CPU (%zu mismatches)", mmQ);

    // ---- solve EE ----
    SolveClothClothPush sp = {};
    sp.cloths = bDesc.addr; sp.contactInfos = bInfos.addr; sp.numContacts = bNum.addr; sp.dt = dt;
    Kernel kSolve("femClothClothContactBda.spv", "cloth_solveClothClothDeltaEELaunch", {}, sizeof(SolveClothClothPush));
    kSolve.dispatch(1u, {}, sp, 1u, 1u);
    dev->flush();

    for (int i = 0; i < 4; ++i)
    {
        refDelta[ref.slot[i]].x = ref.dx[i].x;
        refDelta[ref.slot[i]].y = ref.dx[i].y;
        refDelta[ref.slot[i]].z = ref.dx[i].z;
    }

    const auto gDelta = downloadBuffer<F4>(dev, bDelta.h, 6);
    size_t mm = 0;
    for (uint32_t v = 0; v < 6; ++v)
        if (f2u(gDelta[v].x) != f2u(refDelta[v].x) || f2u(gDelta[v].y) != f2u(refDelta[v].y) ||
            f2u(gDelta[v].z) != f2u(refDelta[v].z) || f2u(gDelta[v].w) != f2u(refDelta[v].w))
        { ++mm; std::printf("  EE solve mDeltaPos[%u] MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", v,
                            refDelta[v].x, refDelta[v].y, refDelta[v].z, refDelta[v].w,
                            gDelta[v].x, gDelta[v].y, gDelta[v].z, gDelta[v].w); }

    // teeth: edge0 (v3,v4) is pushed along +normal and edge1 (v0,v1) along -normal -> the
    // crossing edges separate (both endpoints move apart).
    auto dotN = [](const F4& d, const V3& nn) { return d.x * nn.x + d.y * nn.y + d.z * nn.z; };
    PXG_CHECK_MSG(dotN(gDelta[3], normalEE) > 0.0f && dotN(gDelta[4], normalEE) > 0.0f,
                  "solve must push edge0 out along +normal (dots %g,%g)", dotN(gDelta[3], normalEE), dotN(gDelta[4], normalEE));
    PXG_CHECK_MSG(dotN(gDelta[0], normalEE) < 0.0f && dotN(gDelta[1], normalEE) < 0.0f,
                  "solve must push edge1 along -normal (dots %g,%g)", dotN(gDelta[0], normalEE), dotN(gDelta[1], normalEE));
    PXG_CHECK_MSG(mm == 0, "cloth_solveClothClothDeltaEE bit-exact vs CPU (%zu mismatches)", mm);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bRest.h); dev->destroyBuffer(bPrev.h);
    dev->destroyBuffer(bDelta.h); dev->destroyBuffer(bFric.h); dev->destroyBuffer(bTri.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bInfos.h); dev->destroyBuffer(bNum.h); dev->destroyBuffer(bUpd.h);
}

// ===========================================================================
// P8.1: the SELF-CONTACT REDUNDANCY fixture — is the applied correction invariant
// in the number of contacts describing ONE geometric overlap?
//
// WHY THIS EXISTS.  Fixing the midphase grid defect DOUBLES the self-collision contact
// count (VT 76 -> 152 measured) and makes the clothselfcollide scene markedly WORSE: the
// draping layer is pushed above its own spawn height, the layers separate, and the upper
// one tunnels through the shelf.  Buffer overflow, contact duplication, the count pass,
// the divisor, the clear, a downstream grid mismatch and the iteration budget were all
// excluded by measurement, and every link that can be compared against the .cu by reading
// matches it.  What was missing was a fixture with an ANALYTICALLY KNOWN answer -- the
// existing self-contact tests compare the port against a CPU transcription of the same
// logic, so they agree with each other whether or not that logic is right.
//
// THE ALGEBRA, for a free vertex above a PINNED triangle.  Weights are (1, -r, -s, -t)
// with r+s+t = 1.  countContact bumps mDeltaPos.w by |coeff_i|, so with N identical
// contacts w[free] = N.  The solve multiplies the inverse masses by those counts
// (splittedInvMasses, femClothClothContactBda.slang:474) -- so denom = N*m and
// deltaLambdaN = -CN/(N*m), while dx = normal * deltaLambdaN * (N*m) * 1 = -CN * normal,
// i.e. each contact contributes the SAME correction regardless of N.  N of them accumulate
// to N * (-CN*normal) with .w = N, and applyExternalDeltas divides by .w:
//
//     displacement = (restDistance - gap) * normal,   INDEPENDENT OF N.
//
// The pinned triangle drops out entirely (invMass 0), so the closed form is exact: after
// one full pass the free vertex must sit at EXACTLY the rest distance from the triangle
// plane.  That is the assertion -- an analytic outcome, not a transcription.
//
// If the measured displacement instead SCALES with N, that is the over-response, and it
// is what makes the corrected (doubled) contact set destabilise the scene.
// ===========================================================================
static void runClothSelfContactRedundancy(uint32_t nbContacts)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 1.0f / 60.0f;
    const float freeInvMass = 0.5f;
    const float restDistScalar = 0.05f;    // total rest distance = 2 * 0.05 = 0.1
    const float totalRestDist = 2.0f * restDistScalar;

    // Triangle T0 = {0,1,2} in the z = 0 plane, PINNED (invMass 0 -> it cannot move, so
    // the free vertex's displacement is the whole story).  v3 hovers `gap` above the
    // centroid along +z, well inside the triangle.
    const float gap = 0.04f;               // < totalRestDist -> penetrating
    std::vector<F4> pos = {
        { 0.00f, 0.00f, 0.00f, 0.0f },     // v0 (T0) PINNED
        { 1.00f, 0.00f, 0.00f, 0.0f },     // v1 (T0) PINNED
        { 0.00f, 1.00f, 0.00f, 0.0f },     // v2 (T0) PINNED
        { 1.0f / 3.0f, 1.0f / 3.0f, gap, freeInvMass },   // v3, above the centroid
    };
    std::vector<F4> rest = pos;            // rest-pose filter distance 0 -> passes
    std::vector<F4> prev = pos;            // relDelta 0
    std::vector<F4> vel(4, F4{ 0, 0, 0, 0 });
    std::vector<F4> accum(4, F4{ 0, 0, 0, 0 });
    std::vector<F4> deltaPos(4, F4{ 0, 0, 0, 0 });
    std::vector<float> friction(4, 0.0f);
    std::vector<TriIdx4> tris = { { 0u, 1u, 2u, 0u } };

    DevBuf bPos = makeDev(dev, pos);
    DevBuf bRest = makeDev(dev, rest);
    DevBuf bPrev = makeDev(dev, prev);
    DevBuf bVel = makeDev(dev, vel);
    DevBuf bAcc = makeDev(dev, accum);
    DevBuf bDelta = makeDev(dev, deltaPos);
    DevBuf bFric = makeDev(dev, friction);
    DevBuf bTri = makeDev(dev, tris);

    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bPos.addr));
    c.mRestPosition = reinterpret_cast<decltype(c.mRestPosition)>(size_t(bRest.addr));
    c.mPrevPositionInContactOffset = reinterpret_cast<decltype(c.mPrevPositionInContactOffset)>(size_t(bPrev.addr));
    c.mVelocity_InvMass = reinterpret_cast<decltype(c.mVelocity_InvMass)>(size_t(bVel.addr));
    c.mAccumulatedDeltaPos = reinterpret_cast<decltype(c.mAccumulatedDeltaPos)>(size_t(bAcc.addr));
    c.mDeltaPos = reinterpret_cast<decltype(c.mDeltaPos)>(size_t(bDelta.addr));
    c.mDynamicFrictions = reinterpret_cast<decltype(c.mDynamicFrictions)>(size_t(bFric.addr));
    c.mTriangleVertexIndices = reinterpret_cast<decltype(c.mTriangleVertexIndices)>(size_t(bTri.addr));
    c.mNbVerts = 4u;
    c.mSelfCollisionFilterDistance = 0.0f;
    c.mRestDistance = restDistScalar;
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    // N IDENTICAL VT contacts for the one overlap (v3 vs T0).  The midphase is skipped on
    // purpose: the contact set is the independent variable here, so it is controlled
    // exactly rather than produced.
    std::vector<FemFemContactInfoHost> infos(nbContacts);
    for (uint32_t i = 0; i < nbContacts; ++i)
    {
        infos[i].pairInd0 = (0ull << 20) | 3ull;   // encode(cloth 0, vertex 3)
        infos[i].pairInd1 = (0u << 20) | 0u;       // encode(cloth 0, triangle 0)
        infos[i].auxInd = 0u;                      // VT (bit30=0), not-yet-valid (bit31=0)
    }
    DevBuf bInfos = makeDev(dev, infos);
    std::vector<uint32_t> numContacts = { nbContacts };
    DevBuf bNum = makeDev(dev, numContacts);
    std::vector<uint32_t> upd = { 1u };
    DevBuf bUpd = makeDev(dev, upd);

    if (!bPos.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    const uint32_t MAXC = 64u;

    // ---- the real chain: prepare -> count -> solve -> apply --------------------
    PrepClothClothPush pp = {};
    pp.clothes = bDesc.addr; pp.contactInfos = bInfos.addr; pp.numContacts = bNum.addr;
    pp.maxContacts = MAXC; pp.clothMaterials = 0ull; pp.updateContactPairs = bUpd.addr;
    Kernel kPrep("femClothClothContactBda.spv", "cloth_clothContactPrepareLaunch", {}, sizeof(PrepClothClothPush));
    kPrep.dispatch(1u, {}, pp, 1u, 1u);
    dev->flush();

    QueryClothClothPush qp = {};
    qp.cloths = bDesc.addr; qp.contactInfos = bInfos.addr; qp.numContacts = bNum.addr;
    Kernel kQuery("femClothClothContactBda.spv", "cloth_queryClothClothContactVTCountLaunch", {}, sizeof(QueryClothClothPush));
    kQuery.dispatch(1u, {}, qp, 1u, 1u);
    dev->flush();

    const auto afterCount = downloadBuffer<F4>(dev, bDelta.h, 4);

    SolveClothClothPush sp = {};
    sp.cloths = bDesc.addr; sp.contactInfos = bInfos.addr; sp.numContacts = bNum.addr; sp.dt = dt;
    Kernel kSolve("femClothClothContactBda.spv", "cloth_solveClothClothDeltaVTLaunch", {}, sizeof(SolveClothClothPush));
    kSolve.dispatch(1u, {}, sp, 1u, 1u);
    dev->flush();

    const auto afterSolve = downloadBuffer<F4>(dev, bDelta.h, 4);

    ApplyExtPush ap = {};
    ap.clothes = bDesc.addr; ap.activeClothes = 0ull; ap.dt = dt;
    std::vector<uint32_t> activeIds = { 0u };
    DevBuf bActive = makeDev(dev, activeIds);
    ap.activeClothes = bActive.addr;
    Kernel kApply("femClothRigidAttachBda.spv", "cloth_applyExternalDeltasLaunch", {}, sizeof(ApplyExtPush));
    kApply.dispatch(1u, {}, ap, 1u, 1u);
    dev->flush();

    const auto finalPos = downloadBuffer<F4>(dev, bPos.h, 4);

    // ---- the analytic expectation ---------------------------------------------
    // The triangle lies in z = 0 and is pinned, so the contact normal is +z, the gap is
    // v3.z, and one full pass must place v3 at EXACTLY the total rest distance.
    const float movedTo = finalPos[3].z;
    const float displacement = movedTo - gap;
    const float expectDisp = totalRestDist - gap;

    std::printf("  N=%u: mDeltaPos[3].w after count = %.4f, .z after solve = %.6f -> v3.z %.6f -> %.6f "
                "(moved %.6f, expected %.6f)\n",
                nbContacts, afterCount[3].w, afterSolve[3].z, gap, movedTo, displacement, expectDisp);

    // The pinned triangle must not have moved at all (invMass 0).
    for (uint32_t v = 0; v < 3u; ++v)
        PXG_CHECK_MSG(f2u(finalPos[v].z) == f2u(0.0f),
                      "pinned triangle vertex %u must not move (z = %g)", v, finalPos[v].z);

    // The count must scale with N -- if it does not, the fixture is not testing what it
    // claims to and every later assertion is meaningless.
    PXG_CHECK_MSG(std::fabs(afterCount[3].w - float(nbContacts)) < 1e-4f,
                  "the count pass must accumulate |coeff|=1 per contact: N=%u gave .w=%g",
                  nbContacts, afterCount[3].w);

    // THE ASSERTION: the applied displacement is the analytic one, independent of N.
    PXG_CHECK_MSG(std::fabs(displacement - expectDisp) < 1.0e-5f,
                  "N=%u: one pass must move the free vertex to the rest distance -- moved %.6f, "
                  "expected %.6f (ratio %.3f)", nbContacts, displacement, expectDisp,
                  expectDisp != 0.0f ? displacement / expectDisp : 0.0f);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bRest.h); dev->destroyBuffer(bPrev.h);
    dev->destroyBuffer(bVel.h); dev->destroyBuffer(bAcc.h); dev->destroyBuffer(bDelta.h);
    dev->destroyBuffer(bFric.h); dev->destroyBuffer(bTri.h); dev->destroyBuffer(bDesc.h);
    dev->destroyBuffer(bInfos.h); dev->destroyBuffer(bNum.h); dev->destroyBuffer(bUpd.h);
    dev->destroyBuffer(bActive.h);
}

// ===========================================================================
// P8.1: the TWO-DIRECTION fixture — what the grid fix actually adds.
//
// The redundancy fixture above proves that N copies of ONE contact are perfectly
// mass-averaged.  But the grid fix does not add copies: it lets the OTHER layer's
// vertices act as query vertices too, so each overlap region ends up described from BOTH
// directions -- (layerA vertex, layerB triangle) AND (layerB vertex, layerA triangle).
// Those are genuinely different constraints on different endpoint sets, so the averaging
// above does not relate them.
//
// Two parallel FREE triangles a gap apart, gap < restDistance.  Direction 1 is a0 vs TB;
// direction 2 is b0 vs TA.  The test reports the separation achieved after one full pass
// with direction 1 alone (what the buggy half-coverage midphase produced) and with both
// (what the corrected midphase produces).  The correct answer for a contact solve is that
// the pair ends up at the rest distance EITHER WAY -- resolving one overlap twice from
// opposite sides must not separate them twice as far.
// ===========================================================================
static float runClothSelfContactTwoWay(bool bothDirections, bool verbose)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 1.0f / 60.0f;
    const float m = 0.5f;
    const float restDistScalar = 0.05f;
    const float totalRestDist = 2.0f * restDistScalar;
    const float gap = 0.04f;

    // TA = {0,1,2} in z = 0;  TB = {3,4,5} in z = gap, directly above and congruent.
    std::vector<F4> pos = {
        { 0.00f, 0.00f, 0.0f, m }, { 1.00f, 0.00f, 0.0f, m }, { 0.00f, 1.00f, 0.0f, m },
        { 0.00f, 0.00f, gap, m }, { 1.00f, 0.00f, gap, m }, { 0.00f, 1.00f, gap, m },
    };
    // a0 (v0) sits directly under b0 (v3); to make each query vertex project INSIDE the
    // opposing triangle rather than onto a corner, both are nudged to the centroid in xy.
    pos[0] = F4{ 1.0f / 3.0f, 1.0f / 3.0f, 0.0f, m };
    pos[3] = F4{ 1.0f / 3.0f, 1.0f / 3.0f, gap, m };

    std::vector<F4> rest = pos, prev = pos;
    std::vector<F4> vel(6, F4{ 0, 0, 0, 0 }), accum(6, F4{ 0, 0, 0, 0 }), deltaPos(6, F4{ 0, 0, 0, 0 });
    std::vector<float> friction(6, 0.0f);
    std::vector<TriIdx4> tris = { { 0u, 1u, 2u, 0u }, { 3u, 4u, 5u, 0u } };

    DevBuf bPos = makeDev(dev, pos), bRest = makeDev(dev, rest), bPrev = makeDev(dev, prev);
    DevBuf bVel = makeDev(dev, vel), bAcc = makeDev(dev, accum), bDelta = makeDev(dev, deltaPos);
    DevBuf bFric = makeDev(dev, friction), bTri = makeDev(dev, tris);

    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bPos.addr));
    c.mRestPosition = reinterpret_cast<decltype(c.mRestPosition)>(size_t(bRest.addr));
    c.mPrevPositionInContactOffset = reinterpret_cast<decltype(c.mPrevPositionInContactOffset)>(size_t(bPrev.addr));
    c.mVelocity_InvMass = reinterpret_cast<decltype(c.mVelocity_InvMass)>(size_t(bVel.addr));
    c.mAccumulatedDeltaPos = reinterpret_cast<decltype(c.mAccumulatedDeltaPos)>(size_t(bAcc.addr));
    c.mDeltaPos = reinterpret_cast<decltype(c.mDeltaPos)>(size_t(bDelta.addr));
    c.mDynamicFrictions = reinterpret_cast<decltype(c.mDynamicFrictions)>(size_t(bFric.addr));
    c.mTriangleVertexIndices = reinterpret_cast<decltype(c.mTriangleVertexIndices)>(size_t(bTri.addr));
    c.mNbVerts = 6u;
    c.mSelfCollisionFilterDistance = 0.0f;
    c.mRestDistance = restDistScalar;
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<FemFemContactInfoHost> infos;
    FemFemContactInfoHost ci = {};
    ci.pairInd0 = (0ull << 20) | 0ull;   // vertex a0 = v0
    ci.pairInd1 = (0u << 20) | 1u;       // triangle TB = index 1
    ci.auxInd = 0u;
    infos.push_back(ci);
    if (bothDirections)
    {
        FemFemContactInfoHost ci2 = {};
        ci2.pairInd0 = (0ull << 20) | 3ull;  // vertex b0 = v3
        ci2.pairInd1 = (0u << 20) | 0u;      // triangle TA = index 0
        ci2.auxInd = 0u;
        infos.push_back(ci2);
    }
    DevBuf bInfos = makeDev(dev, infos);
    std::vector<uint32_t> numContacts = { uint32_t(infos.size()) };
    DevBuf bNum = makeDev(dev, numContacts);
    std::vector<uint32_t> upd = { 1u };
    DevBuf bUpd = makeDev(dev, upd);
    std::vector<uint32_t> activeIds = { 0u };
    DevBuf bActive = makeDev(dev, activeIds);

    if (!bPos.addr) { std::printf("  NOTE: no BDA -> skip\n"); return 0.0f; }

    PrepClothClothPush pp = {};
    pp.clothes = bDesc.addr; pp.contactInfos = bInfos.addr; pp.numContacts = bNum.addr;
    pp.maxContacts = 64u; pp.clothMaterials = 0ull; pp.updateContactPairs = bUpd.addr;
    Kernel("femClothClothContactBda.spv", "cloth_clothContactPrepareLaunch", {}, sizeof(PrepClothClothPush))
        .dispatch(1u, {}, pp, 1u, 1u);
    dev->flush();

    QueryClothClothPush qp = {};
    qp.cloths = bDesc.addr; qp.contactInfos = bInfos.addr; qp.numContacts = bNum.addr;
    Kernel("femClothClothContactBda.spv", "cloth_queryClothClothContactVTCountLaunch", {}, sizeof(QueryClothClothPush))
        .dispatch(1u, {}, qp, 1u, 1u);
    dev->flush();

    SolveClothClothPush sp = {};
    sp.cloths = bDesc.addr; sp.contactInfos = bInfos.addr; sp.numContacts = bNum.addr; sp.dt = dt;
    Kernel("femClothClothContactBda.spv", "cloth_solveClothClothDeltaVTLaunch", {}, sizeof(SolveClothClothPush))
        .dispatch(1u, {}, sp, 1u, 1u);
    dev->flush();

    ApplyExtPush ap = {};
    ap.clothes = bDesc.addr; ap.activeClothes = bActive.addr; ap.dt = dt;
    Kernel("femClothRigidAttachBda.spv", "cloth_applyExternalDeltasLaunch", {}, sizeof(ApplyExtPush))
        .dispatch(1u, {}, ap, 1u, 1u);
    dev->flush();

    const auto fin = downloadBuffer<F4>(dev, bPos.h, 6);
    const float sep = fin[3].z - fin[0].z;   // the two facing query vertices
    if (verbose)
        std::printf("  %-14s a0.z %.6f -> %.6f | b0.z %.6f -> %.6f | separation %.6f -> %.6f "
                    "(rest distance %.6f)\n",
                    bothDirections ? "BOTH dirs:" : "ONE dir:",
                    pos[0].z, fin[0].z, pos[3].z, fin[3].z, gap, sep, totalRestDist);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bRest.h); dev->destroyBuffer(bPrev.h);
    dev->destroyBuffer(bVel.h); dev->destroyBuffer(bAcc.h); dev->destroyBuffer(bDelta.h);
    dev->destroyBuffer(bFric.h); dev->destroyBuffer(bTri.h); dev->destroyBuffer(bDesc.h);
    dev->destroyBuffer(bInfos.h); dev->destroyBuffer(bNum.h); dev->destroyBuffer(bUpd.h);
    dev->destroyBuffer(bActive.h);
    return sep;
}

// ===========================================================================
// P8.1: the K-DISTINCT-TRIANGLES fixture — the configuration the other two miss.
//
// The redundancy fixture uses N copies of ONE contact and the two-way fixture uses one
// contact per direction; both have every participant appearing in exactly ONE contact per
// role.  That is the ONE case where the count-based mass split is neutral.
//
// The realistic configuration is different: one vertex penetrating K DISTINCT opposing
// triangles at once, which is what a dense fold produces.  Then the vertex's accumulated
// count is c_v = K (weight 1 per contact) while each triangle corner carries only c_a ~ 1/3
// from its single contact -- a 3K-fold asymmetry.  Since the scatter is
// dx_j = c_j * m_j * w_j * lambda * n (femClothClothContactBda.slang:474 x :440), the
// count enters the correction as a MULTIPLIER, so the separation migrates onto the
// vertex side as K grows and the triangle side stops moving.
//
// This test measures that migration directly.  It asserts nothing about which behaviour is
// "right" -- upstream PhysX 5.6.1 does the same arithmetic and the port reproduces it
// faithfully -- it records the K-dependence so the effect is a measured number rather than
// an argument.
// ===========================================================================
static float lastOvershootFactor = 0.0f;   // set by runClothSelfContactManyTriangles

static void runClothSelfContactManyTriangles(uint32_t K)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 1.0f / 60.0f;
    const float m = 0.5f;
    const float restDistScalar = 0.05f;
    const float totalRestDist = 2.0f * restDistScalar;
    const float gap = 0.04f;

    // v0 is the free penetrating vertex.  K congruent FREE triangles sit `gap` above it,
    // stacked in the vertex buffer but all at the same height, so each contact describes
    // the same overlap depth while touching a DISTINCT vertex set.
    std::vector<F4> pos;
    pos.push_back(F4{ 1.0f / 3.0f, 1.0f / 3.0f, 0.0f, m });   // v0, the vertex side
    for (uint32_t k = 0; k < K; ++k)
    {
        pos.push_back(F4{ 0.0f, 0.0f, gap, m });
        pos.push_back(F4{ 1.0f, 0.0f, gap, m });
        pos.push_back(F4{ 0.0f, 1.0f, gap, m });
    }
    const uint32_t NV = uint32_t(pos.size());
    std::vector<F4> rest = pos, prev = pos;
    std::vector<F4> vel(NV, F4{ 0, 0, 0, 0 }), accum(NV, F4{ 0, 0, 0, 0 }), deltaPos(NV, F4{ 0, 0, 0, 0 });
    std::vector<float> friction(NV, 0.0f);
    std::vector<TriIdx4> tris;
    for (uint32_t k = 0; k < K; ++k)
        tris.push_back(TriIdx4{ 1u + 3u * k, 2u + 3u * k, 3u + 3u * k, 0u });

    DevBuf bPos = makeDev(dev, pos), bRest = makeDev(dev, rest), bPrev = makeDev(dev, prev);
    DevBuf bVel = makeDev(dev, vel), bAcc = makeDev(dev, accum), bDelta = makeDev(dev, deltaPos);
    DevBuf bFric = makeDev(dev, friction), bTri = makeDev(dev, tris);

    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bPos.addr));
    c.mRestPosition = reinterpret_cast<decltype(c.mRestPosition)>(size_t(bRest.addr));
    c.mPrevPositionInContactOffset = reinterpret_cast<decltype(c.mPrevPositionInContactOffset)>(size_t(bPrev.addr));
    c.mVelocity_InvMass = reinterpret_cast<decltype(c.mVelocity_InvMass)>(size_t(bVel.addr));
    c.mAccumulatedDeltaPos = reinterpret_cast<decltype(c.mAccumulatedDeltaPos)>(size_t(bAcc.addr));
    c.mDeltaPos = reinterpret_cast<decltype(c.mDeltaPos)>(size_t(bDelta.addr));
    c.mDynamicFrictions = reinterpret_cast<decltype(c.mDynamicFrictions)>(size_t(bFric.addr));
    c.mTriangleVertexIndices = reinterpret_cast<decltype(c.mTriangleVertexIndices)>(size_t(bTri.addr));
    c.mNbVerts = NV;
    c.mSelfCollisionFilterDistance = 0.0f;
    c.mRestDistance = restDistScalar;
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<FemFemContactInfoHost> infos(K);
    for (uint32_t k = 0; k < K; ++k)
    {
        infos[k].pairInd0 = (0ull << 20) | 0ull;    // vertex v0
        infos[k].pairInd1 = (0u << 20) | k;         // triangle k
        infos[k].auxInd = 0u;
    }
    DevBuf bInfos = makeDev(dev, infos);
    std::vector<uint32_t> numContacts = { K };
    DevBuf bNum = makeDev(dev, numContacts);
    std::vector<uint32_t> upd = { 1u };
    DevBuf bUpd = makeDev(dev, upd);
    std::vector<uint32_t> activeIds = { 0u };
    DevBuf bActive = makeDev(dev, activeIds);

    if (!bPos.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PrepClothClothPush pp = {};
    pp.clothes = bDesc.addr; pp.contactInfos = bInfos.addr; pp.numContacts = bNum.addr;
    pp.maxContacts = 256u; pp.clothMaterials = 0ull; pp.updateContactPairs = bUpd.addr;
    Kernel("femClothClothContactBda.spv", "cloth_clothContactPrepareLaunch", {}, sizeof(PrepClothClothPush))
        .dispatch(1u, {}, pp, 1u, 1u);
    dev->flush();

    QueryClothClothPush qp = {};
    qp.cloths = bDesc.addr; qp.contactInfos = bInfos.addr; qp.numContacts = bNum.addr;
    Kernel("femClothClothContactBda.spv", "cloth_queryClothClothContactVTCountLaunch", {}, sizeof(QueryClothClothPush))
        .dispatch(1u, {}, qp, 1u, 1u);
    dev->flush();
    const auto afterCount = downloadBuffer<F4>(dev, bDelta.h, NV);

    SolveClothClothPush sp = {};
    sp.cloths = bDesc.addr; sp.contactInfos = bInfos.addr; sp.numContacts = bNum.addr; sp.dt = dt;
    Kernel("femClothClothContactBda.spv", "cloth_solveClothClothDeltaVTLaunch", {}, sizeof(SolveClothClothPush))
        .dispatch(1u, {}, sp, 1u, 1u);
    dev->flush();

    ApplyExtPush ap = {};
    ap.clothes = bDesc.addr; ap.activeClothes = bActive.addr; ap.dt = dt;
    Kernel("femClothRigidAttachBda.spv", "cloth_applyExternalDeltasLaunch", {}, sizeof(ApplyExtPush))
        .dispatch(1u, {}, ap, 1u, 1u);
    dev->flush();

    const auto fin = downloadBuffer<F4>(dev, bPos.h, NV);
    const float deficit = totalRestDist - gap;                 // 0.06, what must be closed
    const float dVert = fin[0].z - pos[0].z;                   // vertex side (moves -z)
    const float dCorner = fin[1].z - pos[1].z;                 // one triangle corner (+z)
    const float sep = (fin[1].z - fin[0].z) - gap;             // achieved extra separation

    lastOvershootFactor = sep / deficit;
    std::printf("  K=%-3u count[v]=%.4f count[corner]=%.4f | vertex %+.6f  corner %+.6f | "
                "closed %.6f of %.6f (%.4fx)\n",
                K, afterCount[0].w, afterCount[1].w, dVert, dCorner, sep, deficit, sep / deficit);

    // The count asymmetry is the mechanism, so pin it: the vertex side accumulates 1 per
    // contact (K total) while each corner sees only its own contact's |w| (~1/3).
    PXG_CHECK_MSG(std::fabs(afterCount[0].w - float(K)) < 1e-4f,
                  "vertex-side count must be K=%u, got %g", K, afterCount[0].w);
    PXG_CHECK_MSG(afterCount[1].w < 0.5f,
                  "each corner must carry only its own contact's weight (~1/3), got %g", afterCount[1].w);
    // MEASURED: the pass OVERSHOOTS, and the overshoot SHRINKS as K grows --
    //   K = 1  2  4  8  16  ->  1.2000  1.1053  1.0541  1.0274  1.0138
    // The cause is that mDeltaPos.w accumulates |coeff| rather than 1, so the effective
    // denominator is sum_j m_j|w_j|^3 instead of the XPBD-correct sum_j m_j w_j^2; the ratio
    // between them is the overshoot factor.  It is bounded by the vertex side's weight of 1
    // (so <= 1.2 for a centroid VT hit) and decays as the vertex-side count comes to dominate
    // the denominator.  This is UPSTREAM PhysX 5.6.1 arithmetic -- the .cu bumps by
    // PxAbs(coeffs[j]) at FEMClothExternalSolve.cu:882-896 despite its own comment calling it
    // a reference count -- and the Slang port reproduces it faithfully.  So the bound is a
    // REGRESSION FENCE, not a correctness claim: it pins the inherited behaviour so a future
    // port change that alters it fails loudly.
    PXG_CHECK_MSG(sep <= deficit * 1.25f,
                  "K=%u: separation %.6f exceeds the inherited overshoot envelope (%.4fx > 1.25x)",
                  K, sep, deficit, sep / deficit);
    // The corner side must lose its share as K grows -- that migration IS the mechanism.
    PXG_CHECK_MSG(dCorner > 0.0f && dCorner < deficit,
                  "K=%u: the triangle corner must move the other way and by less than the deficit "
                  "(got %+.6f)", K, dCorner);
}

PXG_TEST(femClothSimSelfContactManyTriangles)
{
    // The K-sweep is the measurement; the per-K checks above are the teeth.  The sweep must
    // also be MONOTONE: the overshoot decays as the vertex-side count grows, and that shape
    // is the signature of the |coeff|-vs-1 denominator.  A port change that broke the count
    // accumulation would keep the individual bounds but destroy the ordering.
    float prevFactor = 1.0e30f;
    for (uint32_t K : { 1u, 2u, 4u, 8u, 16u })
    {
        runClothSelfContactManyTriangles(K);
        const float f = lastOvershootFactor;
        PXG_CHECK_MSG(f < prevFactor,
                      "the overshoot must decay with contact density: K=%u gave %.4fx, "
                      "not below the previous %.4fx", K, f, prevFactor);
        prevFactor = f;
    }
    PXG_CHECK_MSG(prevFactor < 1.02f,
                  "at K=16 the overshoot should have decayed below 1.02x (got %.4fx)", prevFactor);
}

PXG_TEST(femClothSimSelfContactTwoWay)
{
    const float totalRestDist = 0.1f;
    const float sepOne = runClothSelfContactTwoWay(/*bothDirections*/ false, /*verbose*/ true);
    const float sepBoth = runClothSelfContactTwoWay(/*bothDirections*/ true, /*verbose*/ true);
    if (sepOne == 0.0f && sepBoth == 0.0f) return;   // no-BDA skip

    // A single overlap resolved from ONE side lands exactly on the rest distance -- the
    // redundancy fixture already pins that mechanism analytically.
    PXG_CHECK_MSG(std::fabs(sepOne - totalRestDist) < 1.0e-5f,
                  "one direction must separate the layers to the rest distance: got %.6f, expected %.6f",
                  sepOne, totalRestDist);

    // THE QUESTION.  Resolving the SAME overlap additionally from the opposite side must
    // not push it further apart: the second constraint sees a pair that direction 1 has
    // already separated to the rest distance, so it should contribute nothing.  If the
    // separation grows instead, the two directions add rather than agree -- and that is
    // precisely what doubles the push when the midphase grid defect is fixed.
    std::printf("  over-separation factor: %.4f\n", totalRestDist != 0.0f ? sepBoth / totalRestDist : 0.0f);
    PXG_CHECK_MSG(std::fabs(sepBoth - totalRestDist) < 1.0e-5f,
                  "both directions must ALSO land on the rest distance, not overshoot: got %.6f, "
                  "expected %.6f (factor %.3f)", sepBoth, totalRestDist, sepBoth / totalRestDist);
}

PXG_TEST(femClothSimSelfContactRedundancy1) { runClothSelfContactRedundancy(1u); }
PXG_TEST(femClothSimSelfContactRedundancy2) { runClothSelfContactRedundancy(2u); }
PXG_TEST(femClothSimSelfContactRedundancy4) { runClothSelfContactRedundancy(4u); }

PXG_TEST(femClothSimSelfContactSolveVT) { runClothSelfContactVT(); }
PXG_TEST(femClothSimSelfContactSolveEE) { runClothSelfContactEE(); }

// cloth_updateClothContactValidityLaunch — snapshot each vertex into
// mPrevPositionInContactOffset (w = 1) and reset the VT/EE contact counts on an adaptive
// update.  Pure copy + set-w — trivially bit-exact; the forced-fresh mutation check flips
// `adaptive` off to prove the write is guarded (a stub that always wrote would fail).
PXG_TEST(femClothSimUpdateClothContactValidity)
{
    PxgSlangComputeDevice* dev = testDevice();

    std::vector<F4> pos = {
        { 1.5f, -2.5f, 3.5f, 0.25f },
        { -4.0f, 5.0f, -6.0f, 0.5f },
        { 7.0f, 8.0f, 9.0f, 1.0f },
    };
    const uint32_t N = uint32_t(pos.size());
    std::vector<F4> prev(N, F4{ -111.0f, -222.0f, -333.0f, -444.0f });   // poison
    std::vector<float> fric(N, 0.0f);

    DevBuf bPos = makeDev(dev, pos);
    DevBuf bPrev = makeDev(dev, prev);

    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bPos.addr));
    c.mPrevPositionInContactOffset = reinterpret_cast<decltype(c.mPrevPositionInContactOffset)>(size_t(bPrev.addr));
    c.mNbVerts = N;
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);
    std::vector<uint32_t> upd = { 1u };   // updateContactPairs == 1
    DevBuf bUpd = makeDev(dev, upd);
    std::vector<uint32_t> vtCount = { 77u };
    DevBuf bVt = makeDev(dev, vtCount);
    std::vector<uint32_t> eeCount = { 88u };
    DevBuf bEe = makeDev(dev, eeCount);

    if (!bPos.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    UpdateValidityPush up = {};
    up.femCloths = bDesc.addr; up.activeIds = bActive.addr;
    up.adaptiveWord = 1u;   // adaptiveCollisionPairUpdate = true
    up.forceWord = 0u;      // forceUpdateClothContactPairs (unused)
    up.updateContactPairs = bUpd.addr; up.VTContactCount = bVt.addr; up.EEContactCount = bEe.addr;
    up.dt = 1.0f / 60.0f;   // unused
    Kernel k("femClothClothContactBda.spv", "cloth_updateClothContactValidityLaunch", {}, sizeof(UpdateValidityPush));
    k.dispatch(1u, {}, up, 1u, 1u);
    dev->flush();

    const auto gPrev = downloadBuffer<F4>(dev, bPrev.h, N);
    const uint32_t gVt = downloadBuffer<uint32_t>(dev, bVt.h, 1)[0];
    const uint32_t gEe = downloadBuffer<uint32_t>(dev, bEe.h, 1)[0];
    size_t mm = 0;
    for (uint32_t v = 0; v < N; ++v)
        if (f2u(gPrev[v].x) != f2u(pos[v].x) || f2u(gPrev[v].y) != f2u(pos[v].y) ||
            f2u(gPrev[v].z) != f2u(pos[v].z) || f2u(gPrev[v].w) != f2u(1.0f))
        { ++mm; std::printf("  update prev[%u] MM: ref(%g,%g,%g,1) got(%g,%g,%g,%g)\n", v,
                            pos[v].x, pos[v].y, pos[v].z, gPrev[v].x, gPrev[v].y, gPrev[v].z, gPrev[v].w); }
    PXG_CHECK_MSG(gVt == 0u && gEe == 0u, "adaptive update must reset VT/EE counts (got %u,%u)", gVt, gEe);
    PXG_CHECK_MSG(mm == 0, "cloth_updateClothContactValidity bit-exact vs CPU (%zu mismatches)", mm);

    // forced-fresh mutation check: adaptive OFF -> the snapshot MUST NOT happen and the
    // counts MUST survive (proves the write is genuinely gated, not an unconditional stub).
    std::vector<F4> prev2(N, F4{ -111.0f, -222.0f, -333.0f, -444.0f });
    DevBuf bPrev2 = makeDev(dev, prev2);
    std::vector<uint32_t> vt2 = { 77u };
    DevBuf bVt2 = makeDev(dev, vt2);
    std::vector<uint32_t> ee2 = { 88u };
    DevBuf bEe2 = makeDev(dev, ee2);
    PxgFEMCloth c2 = c;
    c2.mPrevPositionInContactOffset = reinterpret_cast<decltype(c2.mPrevPositionInContactOffset)>(size_t(bPrev2.addr));
    std::vector<PxgFEMCloth> descArr2 = { c2 };
    DevBuf bDesc2 = makeDev(dev, descArr2);
    UpdateValidityPush up2 = up;
    up2.femCloths = bDesc2.addr; up2.adaptiveWord = 0u;   // adaptive OFF
    up2.VTContactCount = bVt2.addr; up2.EEContactCount = bEe2.addr;
    k.dispatch(1u, {}, up2, 1u, 1u);
    dev->flush();
    const auto gPrev2 = downloadBuffer<F4>(dev, bPrev2.h, N);
    const uint32_t gVt2 = downloadBuffer<uint32_t>(dev, bVt2.h, 1)[0];
    const uint32_t gEe2 = downloadBuffer<uint32_t>(dev, bEe2.h, 1)[0];
    PXG_CHECK_MSG(f2u(gPrev2[0].w) == f2u(-444.0f), "non-adaptive update must leave prev untouched (got w=%g)", gPrev2[0].w);
    PXG_CHECK_MSG(gVt2 == 77u && gEe2 == 88u, "non-adaptive update must not reset counts (got %u,%u)", gVt2, gEe2);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bPrev.h); dev->destroyBuffer(bDesc.h);
    dev->destroyBuffer(bActive.h); dev->destroyBuffer(bUpd.h); dev->destroyBuffer(bVt.h); dev->destroyBuffer(bEe.h);
    dev->destroyBuffer(bPrev2.h); dev->destroyBuffer(bVt2.h); dev->destroyBuffer(bEe2.h); dev->destroyBuffer(bDesc2.h);
}


// ---------------------------------------------------------------------------
// P7.2 tranche 3a — cloth_refitBoundLaunch (FEMCloth.cu:338, femClothRefitBda.spv).
//
// The BVH bounds refit: walk the cloth's Gu::BV32DataPacked tree bottom-up (deepest
// BV32DataDepthInfo level first), rewrite every node's 32 per-child AABBs from the
// CURRENT vertex positions, aggregate each node's own bound into mPackedNodeBounds,
// and finally expand the root bound into the broad-phase boundArray slot.  Without
// this the tranche-1 self-collision midphase queries the rest-pose tree it was cooked
// with and returns zero contacts.
//
// The test tree is deliberately TWO levels deep so BOTH aggregation paths run:
//   node 0 (root, level 0): 2 INTERNAL children -> nodes 1 and 2
//   node 1 (level 1): 2 LEAF children (prims 0-1 and prim 2)
//   node 2 (level 1): 1 LEAF child  (prims 3-5)
// Level 1 exercises the leaf triangle-vertex min/max ballot walk; level 0 exercises
// the `!isLeaf -> read mPackedNodeBounds[childOffset]` path, which is only correct if
// level 1 really ran first (a top-down or single-level port fails here).
// ---------------------------------------------------------------------------
struct RefitPush
{
    uint64_t femClothes, activeFemClothes;
    uint32_t nbActive, pad20;
    uint64_t contactDists, speculativeCCD, boundArray, elemIndex;
    float nbCollisionPairUpdatesPerTimestep;
    uint32_t pad60;
    uint64_t updateContactPairs;
};
static_assert(sizeof(RefitPush) == 72, "refitBound push blob must be 72 bytes");
static_assert(offsetof(RefitPush, contactDists) == 24, "contactDists @24");
static_assert(offsetof(RefitPush, boundArray) == 40, "boundArray @40");
static_assert(offsetof(RefitPush, nbCollisionPairUpdatesPerTimestep) == 56, "nbUpdates @56");
static_assert(offsetof(RefitPush, updateContactPairs) == 64, "updateContactPairs @64");

namespace p72refit
{
// Gu::BV32DataPacked = 1168 B = 292 uint32 words.
static const uint32_t BV32_WORDS = 292u;
static const uint32_t W_MMIN  = 0u;    // mMin[32] float4 @byte 0
static const uint32_t W_MMAX  = 128u;  // mMax[32] float4 @byte 512
static const uint32_t W_MDATA = 256u;  // mData[32] @byte 1024
static const uint32_t W_NBNODES = 288u;// mNbNodes @byte 1152
static const uint32_t W_DEPTH = 289u;  // mDepth  @byte 1156

static uint32_t leafWord(uint32_t startIndex, uint32_t nbPrims)
{
    return (startIndex << 7) | (nbPrims << 1) | 1u;   // isLeaf bit 0
}
static uint32_t internalWord(uint32_t childOffset)
{
    return childOffset << 11;                          // GU_BV4_CHILD_OFFSET_SHIFT_COUNT, isLeaf 0
}

struct Bounds { p72ref::V3 mn, mx; };

// The whole tree blob: uint4 header + nbNodes x BV32DataPacked + maxDepth x
// BV32DataDepthInfo + nbNodes x remapPackedNodeIndex.
struct TreeBlob
{
    std::vector<uint32_t> w;
    uint32_t nodeWord(uint32_t node) const { return 4u + node * BV32_WORDS; }
};

// The 32-lane semantics of the CUDA kernel, transcribed exactly (-ffp-contract=off):
// per node, the leaf ballot walk broadcasts a warp-wide min/max into EVERY lane, the
// non-leaf lanes then OVERWRITE their own lane value with the child's stored bound,
// and the node's aggregate reduces over all 32 lanes (lanes >= nbChildren still carry
// the leaf aggregate / the +-MAX identity, which is exactly what CUDA reduces too).
static void refitNodeRef(uint32_t nodeIndex, TreeBlob& tree, std::vector<Bounds>& packed,
                         const std::vector<F4>& pos, const std::vector<TriIdx4>& tris)
{
    const uint32_t nw = tree.nodeWord(nodeIndex);
    const uint32_t nbChildren = tree.w[nw + W_NBNODES];
    const float MAXF = PX_MAX_REAL_F;

    std::vector<p72ref::V3> laneMin(32, p72ref::V3{ MAXF, MAXF, MAXF });
    std::vector<p72ref::V3> laneMax(32, p72ref::V3{ -MAXF, -MAXF, -MAXF });

    uint32_t resultWarp = 0u;
    for (uint32_t k = 0; k < 32u; ++k)
        if (k < nbChildren && (tree.w[nw + W_MDATA + k] & 1u))
            resultWarp |= (1u << k);

    for (uint32_t k = resultWarp; k != 0u; k &= (k - 1u))
    {
        uint32_t indexInWarp = 0;
        while (((k >> indexInWarp) & 1u) == 0u) ++indexInWarp;
        const uint32_t d = tree.w[nw + W_MDATA + indexInWarp];
        const uint32_t nbPrimitives = (d >> 1) & 63u;
        const uint32_t startIndex = d >> 7;

        std::vector<p72ref::V3> tMin = laneMin, tMax = laneMax;
        for (uint32_t lane = 0; lane < 32u; ++lane)
        {
            if (lane >= nbPrimitives) continue;
            const TriIdx4& t = tris[startIndex + lane];
            const F4& a = pos[t.x]; const F4& b = pos[t.y]; const F4& c = pos[t.z];
            float tx = p72ref::fmin3(a.x, b.x), ty = p72ref::fmin3(a.y, b.y), tz = p72ref::fmin3(a.z, b.z);
            tx = p72ref::fmin3(tx, c.x); ty = p72ref::fmin3(ty, c.y); tz = p72ref::fmin3(tz, c.z);
            tMin[lane] = p72ref::vmin(tMin[lane], p72ref::V3{ tx, ty, tz });
            tx = p72ref::fmax3(a.x, b.x); ty = p72ref::fmax3(a.y, b.y); tz = p72ref::fmax3(a.z, b.z);
            tx = p72ref::fmax3(tx, c.x); ty = p72ref::fmax3(ty, c.y); tz = p72ref::fmax3(tz, c.z);
            tMax[lane] = p72ref::vmax(tMax[lane], p72ref::V3{ tx, ty, tz });
        }
        p72ref::V3 rn = tMin[0], rx = tMax[0];
        for (uint32_t lane = 1; lane < 32u; ++lane)
        { rn = p72ref::vmin(rn, tMin[lane]); rx = p72ref::vmax(rx, tMax[lane]); }
        for (uint32_t lane = 0; lane < 32u; ++lane) { laneMin[lane] = rn; laneMax[lane] = rx; }
    }

    for (uint32_t lane = 0; lane < 32u; ++lane)
    {
        if (lane < nbChildren && !(tree.w[nw + W_MDATA + lane] & 1u))
        {
            const uint32_t childOffset = tree.w[nw + W_MDATA + lane] >> 11;
            laneMin[lane] = packed[childOffset].mn;
            laneMax[lane] = packed[childOffset].mx;
        }
    }

    for (uint32_t lane = 0; lane < nbChildren; ++lane)
    {
        std::memcpy(&tree.w[nw + W_MMIN + lane * 4 + 0], &laneMin[lane].x, 4);
        std::memcpy(&tree.w[nw + W_MMIN + lane * 4 + 1], &laneMin[lane].y, 4);
        std::memcpy(&tree.w[nw + W_MMIN + lane * 4 + 2], &laneMin[lane].z, 4);
        tree.w[nw + W_MMIN + lane * 4 + 3] = f2u(0.0f);
        std::memcpy(&tree.w[nw + W_MMAX + lane * 4 + 0], &laneMax[lane].x, 4);
        std::memcpy(&tree.w[nw + W_MMAX + lane * 4 + 1], &laneMax[lane].y, 4);
        std::memcpy(&tree.w[nw + W_MMAX + lane * 4 + 2], &laneMax[lane].z, 4);
        tree.w[nw + W_MMAX + lane * 4 + 3] = f2u(0.0f);
    }

    p72ref::V3 an = laneMin[0], ax = laneMax[0];
    for (uint32_t lane = 1; lane < 32u; ++lane)
    { an = p72ref::vmin(an, laneMin[lane]); ax = p72ref::vmax(ax, laneMax[lane]); }
    packed[nodeIndex].mn = an;
    packed[nodeIndex].mx = ax;
}
} // namespace p72refit

PXG_TEST(femClothSimRefitBound)
{
    using namespace p72refit;
    PxgSlangComputeDevice* dev = testDevice();

    // 8 vertices with deliberately irregular coordinates (no exact-representable grid,
    // so a min/max mix-up cannot accidentally agree), 6 triangles.
    std::vector<F4> pos = {
        { -0.31718f,  0.90211f, -0.14477f, 0.5f },
        {  1.17093f, -0.42388f,  0.63512f, 0.5f },
        {  0.55401f,  1.30877f,  0.21939f, 0.5f },
        { -1.02664f,  0.13205f,  1.44170f, 0.5f },
        {  2.31882f,  0.77519f, -0.90613f, 0.5f },
        {  0.04471f, -1.15938f,  0.38204f, 0.5f },
        { -0.68350f,  2.02117f,  1.09926f, 0.5f },
        {  1.83007f,  0.29644f, -1.27735f, 0.5f },
    };
    std::vector<TriIdx4> tris = {
        { 0u, 1u, 2u, 0u }, { 1u, 3u, 4u, 0u }, { 2u, 5u, 6u, 0u },
        { 3u, 6u, 7u, 0u }, { 4u, 0u, 5u, 0u }, { 5u, 7u, 1u, 0u },
    };

    const uint32_t nbNodes = 3u, maxDepth = 2u;

    // --- assemble the tree blob -------------------------------------------------
    TreeBlob tree;
    tree.w.assign(4u + nbNodes * BV32_WORDS + maxDepth * 2u + nbNodes, 0u);
    tree.w[0] = uint32_t(pos.size());   // nbVerts
    tree.w[1] = uint32_t(tris.size());  // nbTriangles
    tree.w[2] = maxDepth;
    tree.w[3] = nbNodes;

    // Poison every node AABB slot with a recognisable rest-pose stale value: a correct
    // refit must overwrite slots 0..nbChildren-1 and leave the rest untouched.
    const float POISON = -7777.0f;
    for (uint32_t n = 0; n < nbNodes; ++n)
        for (uint32_t k = 0; k < 32u * 4u; ++k)
        {
            tree.w[tree.nodeWord(n) + W_MMIN + k] = f2u(POISON);
            tree.w[tree.nodeWord(n) + W_MMAX + k] = f2u(POISON);
        }

    // node 0: two internal children -> nodes 1, 2
    tree.w[tree.nodeWord(0) + W_MDATA + 0] = internalWord(1u);
    tree.w[tree.nodeWord(0) + W_MDATA + 1] = internalWord(2u);
    tree.w[tree.nodeWord(0) + W_NBNODES] = 2u;
    tree.w[tree.nodeWord(0) + W_DEPTH] = 0u;
    // node 1: two leaf children (tris 0-1, tri 2)
    tree.w[tree.nodeWord(1) + W_MDATA + 0] = leafWord(0u, 2u);
    tree.w[tree.nodeWord(1) + W_MDATA + 1] = leafWord(2u, 1u);
    tree.w[tree.nodeWord(1) + W_NBNODES] = 2u;
    tree.w[tree.nodeWord(1) + W_DEPTH] = 1u;
    // node 2: one leaf child (tris 3-5)
    tree.w[tree.nodeWord(2) + W_MDATA + 0] = leafWord(3u, 3u);
    tree.w[tree.nodeWord(2) + W_NBNODES] = 1u;
    tree.w[tree.nodeWord(2) + W_DEPTH] = 1u;

    // BV32DataDepthInfo[maxDepth]: level 0 -> {offset 0, count 1}; level 1 -> {1, 2}
    const uint32_t wDepthInfo = 4u + nbNodes * BV32_WORDS;
    tree.w[wDepthInfo + 0] = 0u; tree.w[wDepthInfo + 1] = 1u;
    tree.w[wDepthInfo + 2] = 1u; tree.w[wDepthInfo + 3] = 2u;
    // remapPackedNodeIndex[nbNodes]: identity (level 0 = node 0, level 1 = nodes 1,2)
    const uint32_t wRemap = wDepthInfo + maxDepth * 2u;
    tree.w[wRemap + 0] = 0u; tree.w[wRemap + 1] = 1u; tree.w[wRemap + 2] = 2u;

    // --- device state -----------------------------------------------------------
    const uint32_t clothId = 0u, elemIdx = 2u;
    const float origContactOffset = 0.037f, specCCD = 0.011f, nbUpdatesPerStep = 3.0f;

    std::vector<float> packedBoundsInit(nbNodes * 6u, POISON);
    std::vector<float> boundArrayInit((elemIdx + 2u) * 6u, POISON);

    // The tree blob gets a 16-byte trailer.  llvmpipe widens the final
    // remapPackedNodeIndex load past the last element, so a device buffer that ends
    // EXACTLY on that word faults when it happens to sit at the end of a mapped region
    // (reproduces only when this case runs after the rest of the suite, never in
    // isolation).  It is a load-width artifact, not a kernel off-by-one: with the
    // trailer present the refit is still bit-exact, which it could not be if the kernel
    // were really consuming a 4th remap entry (it would refit an extra node).  The real
    // engine blob is a cooked mesh-data block with its own trailing sections, so it
    // never ends on that boundary.
    std::vector<uint32_t> treePad = tree.w; treePad.resize(tree.w.size() + 4u, 0u);
    DevBuf bPos = makeDev(dev, pos);
    DevBuf bTri = makeDev(dev, tris);
    DevBuf bTree = makeDev(dev, treePad);
    DevBuf bPacked = makeDev(dev, packedBoundsInit);
    DevBuf bBounds = makeDev(dev, boundArrayInit);
    std::vector<uint32_t> active = { clothId };
    DevBuf bActive = makeDev(dev, active);
    std::vector<uint32_t> elemIndices = { elemIdx };
    DevBuf bElem = makeDev(dev, elemIndices);
    std::vector<float> spec = { specCCD };
    DevBuf bSpec = makeDev(dev, spec);
    std::vector<float> contactDists = { 0.0f };
    DevBuf bCd = makeDev(dev, contactDists);
    std::vector<uint32_t> upd = { 1u };
    DevBuf bUpd = makeDev(dev, upd);

    PxgFEMCloth c = makeZeroDesc();
    c.mTriMeshData = reinterpret_cast<decltype(c.mTriMeshData)>(size_t(bTree.addr));
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bPos.addr));
    c.mTriangleVertexIndices = reinterpret_cast<decltype(c.mTriangleVertexIndices)>(size_t(bTri.addr));
    c.mPackedNodeBounds = reinterpret_cast<decltype(c.mPackedNodeBounds)>(size_t(bPacked.addr));
    c.mNbVerts = uint32_t(pos.size());
    c.mNbTriangles = uint32_t(tris.size());
    c.mOriginalContactOffset = origContactOffset;
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    RefitPush rp{};
    rp.femClothes = bDesc.addr;
    rp.activeFemClothes = bActive.addr;
    rp.nbActive = 1u;
    rp.contactDists = bCd.addr;
    rp.speculativeCCD = bSpec.addr;
    rp.boundArray = bBounds.addr;
    rp.elemIndex = bElem.addr;
    rp.nbCollisionPairUpdatesPerTimestep = nbUpdatesPerStep;
    rp.updateContactPairs = bUpd.addr;

    Kernel k("femClothRefitBda.spv", "cloth_refitBoundLaunch", {}, sizeof(RefitPush));
    k.dispatch(1u, {}, rp, 1u, 1u);
    dev->flush();

    // --- CPU reference (same bottom-up order the kernel walks) ------------------
    TreeBlob refTree = tree;
    std::vector<Bounds> refPacked(nbNodes);
    for (uint32_t n = 0; n < nbNodes; ++n)
    { refPacked[n].mn = p72ref::V3{ POISON, POISON, POISON }; refPacked[n].mx = refPacked[n].mn; }
    for (uint32_t i = maxDepth; i > 0u; --i)
    {
        const uint32_t off = refTree.w[wDepthInfo + (i - 1u) * 2u + 0];
        const uint32_t cnt = refTree.w[wDepthInfo + (i - 1u) * 2u + 1];
        for (uint32_t j = 0; j < cnt; ++j)
            refitNodeRef(refTree.w[wRemap + j + off], refTree, refPacked, pos, tris);
    }

    const auto gTree = downloadBuffer<uint32_t>(dev, bTree.h, treePad.size());
    const auto gPacked = downloadBuffer<float>(dev, bPacked.h, packedBoundsInit.size());
    const auto gBounds = downloadBuffer<float>(dev, bBounds.h, boundArrayInit.size());

    size_t mm = 0;
    for (uint32_t n = 0; n < nbNodes; ++n)
    {
        const uint32_t nw = refTree.nodeWord(n);
        for (uint32_t k2 = 0; k2 < 32u * 4u; ++k2)
        {
            if (gTree[nw + W_MMIN + k2] != refTree.w[nw + W_MMIN + k2])
            { ++mm; std::printf("  node %u mMin word %u MM: ref %08x got %08x\n", n, k2,
                                refTree.w[nw + W_MMIN + k2], gTree[nw + W_MMIN + k2]); }
            if (gTree[nw + W_MMAX + k2] != refTree.w[nw + W_MMAX + k2])
            { ++mm; std::printf("  node %u mMax word %u MM: ref %08x got %08x\n", n, k2,
                                refTree.w[nw + W_MMAX + k2], gTree[nw + W_MMAX + k2]); }
        }
    }
    for (uint32_t n = 0; n < nbNodes; ++n)
    {
        const float r[6] = { refPacked[n].mn.x, refPacked[n].mn.y, refPacked[n].mn.z,
                             refPacked[n].mx.x, refPacked[n].mx.y, refPacked[n].mx.z };
        for (uint32_t e = 0; e < 6u; ++e)
            if (f2u(gPacked[n * 6 + e]) != f2u(r[e]))
            { ++mm; std::printf("  packedNodeBounds[%u][%u] MM: ref %g got %g\n", n, e, r[e], gPacked[n * 6 + e]); }
    }

    // The refit MUST have actually run: node 0's aggregate is the whole mesh AABB.
    p72ref::V3 wholeMin{ PX_MAX_REAL_F, PX_MAX_REAL_F, PX_MAX_REAL_F };
    p72ref::V3 wholeMax{ -PX_MAX_REAL_F, -PX_MAX_REAL_F, -PX_MAX_REAL_F };
    for (const TriIdx4& t : tris)
        for (uint32_t vi : { t.x, t.y, t.z })
        {
            wholeMin = p72ref::vmin(wholeMin, p72ref::V3{ pos[vi].x, pos[vi].y, pos[vi].z });
            wholeMax = p72ref::vmax(wholeMax, p72ref::V3{ pos[vi].x, pos[vi].y, pos[vi].z });
        }
    PXG_CHECK_MSG(f2u(refPacked[0].mn.x) == f2u(wholeMin.x) && f2u(refPacked[0].mx.z) == f2u(wholeMax.z),
                  "reference root bound must equal the whole-mesh AABB (self-check)");
    PXG_CHECK_MSG(f2u(gPacked[0]) != f2u(POISON), "root packedNodeBounds must be overwritten (refit did not run)");

    // broad-phase slot: root -/+ PxMax(nbUpdates,1) * (originalContactOffset + specCCD)
    const float contactDist = origContactOffset + specCCD;
    const float expansion = (nbUpdatesPerStep > 1.0f ? nbUpdatesPerStep : 1.0f) * contactDist;
    const float refBound[6] = {
        refPacked[0].mn.x - expansion, refPacked[0].mn.y - expansion, refPacked[0].mn.z - expansion,
        refPacked[0].mx.x + expansion, refPacked[0].mx.y + expansion, refPacked[0].mx.z + expansion };
    for (uint32_t e = 0; e < 6u; ++e)
        if (f2u(gBounds[elemIdx * 6 + e]) != f2u(refBound[e]))
        { ++mm; std::printf("  boundArray[%u][%u] MM: ref %g got %g\n", elemIdx, e, refBound[e],
                            gBounds[elemIdx * 6 + e]); }
    // neighbouring broad-phase slots must be untouched (elemIndex indirection proof)
    for (uint32_t e = 0; e < 6u; ++e)
    {
        if (f2u(gBounds[0 * 6 + e]) != f2u(POISON)) { ++mm; std::printf("  boundArray[0][%u] clobbered\n", e); }
        if (f2u(gBounds[(elemIdx + 1u) * 6 + e]) != f2u(POISON))
        { ++mm; std::printf("  boundArray[%u][%u] clobbered\n", elemIdx + 1u, e); }
    }
    PXG_CHECK_MSG(mm == 0, "cloth_refitBoundLaunch bit-exact vs CPU (%zu mismatches)", mm);

    // forced-fresh gating check: updateContactPairs == 0 -> the kernel MUST early-out and
    // leave every output at its poison value (a stub that always refit would fail here).
    std::vector<float> packed2(nbNodes * 6u, POISON);
    DevBuf bPacked2 = makeDev(dev, packed2);
    std::vector<uint32_t> upd0 = { 0u };
    DevBuf bUpd0 = makeDev(dev, upd0);
    PxgFEMCloth c2 = c;
    c2.mPackedNodeBounds = reinterpret_cast<decltype(c2.mPackedNodeBounds)>(size_t(bPacked2.addr));
    std::vector<PxgFEMCloth> descArr2 = { c2 };
    DevBuf bDesc2 = makeDev(dev, descArr2);
    RefitPush rp2 = rp;
    rp2.femClothes = bDesc2.addr; rp2.updateContactPairs = bUpd0.addr;
    k.dispatch(1u, {}, rp2, 1u, 1u);
    dev->flush();
    const auto gPacked2 = downloadBuffer<float>(dev, bPacked2.h, packed2.size());
    for (uint32_t e = 0; e < nbNodes * 6u; ++e)
        PXG_CHECK_MSG(f2u(gPacked2[e]) == f2u(POISON),
                      "updateContactPairs==0 must skip the refit entirely (slot %u = %g)", e, gPacked2[e]);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bTri.h); dev->destroyBuffer(bTree.h);
    dev->destroyBuffer(bPacked.h); dev->destroyBuffer(bBounds.h); dev->destroyBuffer(bActive.h);
    dev->destroyBuffer(bElem.h); dev->destroyBuffer(bSpec.h); dev->destroyBuffer(bCd.h);
    dev->destroyBuffer(bUpd.h); dev->destroyBuffer(bDesc.h);
    dev->destroyBuffer(bPacked2.h); dev->destroyBuffer(bUpd0.h); dev->destroyBuffer(bDesc2.h);
}

// ===========================================================================
// P8.12 -- cloth_stepLaunch, the TGS cloth substep integrator.
//
// This is the kernel whose absence made every cloth scene refuse to move on the TGS solver.
// The live proof is PXG_SCENE=clothdrape with PXG_SOLVER unset, which was RED before this port
// (dropped 0.0000 over 3.00s) and settles afterwards.  This fixture covers the parts a settling
// scene cannot separate:
//
//   * the 64-byte blob, whose PxVec3 gravity sits at @20 UNPADDED and whose three bools pack at
//     @32/@33/@34 -- get the padding wrong and every pointer after it is off;
//   * the contact-pair bookkeeping: who writes updateContactPairs, and that a forced update
//     ZEROES both contact counts;
//   * the gravity/damping predict, its invMass gate, its eDISABLE_GRAVITY gate, and the
//     min(1, damping*dt) clamp that stops a large damping REVERSING the velocity;
//   * all three regimes of the maximum-displacement clamp, its isPrevPosSet gate, and that the
//     correction reaches the velocity as well as the position;
//   * that mPrevPositionInRestOffset is written ALWAYS and mPrevPositionInContactOffset only on
//     a forced update, with .w marked 1.
// ===========================================================================
namespace p812step
{
struct ClothStepPush
{
    uint64_t femCloths, activeId;
    float dt;
    float gx, gy, gz;
    uint8_t externalForces, adaptive, forceUpdate, pad35;
    uint32_t pad36;
    uint64_t updateContactPairs, vtContactCount, eeContactCount;
};
static_assert(sizeof(ClothStepPush) == 64, "cloth_stepLaunch blob must be 64 bytes");
static_assert(offsetof(ClothStepPush, dt) == 16, "dt @16");
static_assert(offsetof(ClothStepPush, gx) == 20, "PxVec3 gravity @20, NOT padded to 16");
static_assert(offsetof(ClothStepPush, externalForces) == 32, "the three bools pack at @32");
static_assert(offsetof(ClothStepPush, updateContactPairs) == 40, "the next pointer realigns @40");
} // namespace p812step

PXG_TEST(femClothSimStep)
{
    using namespace p812step;
    PxgSlangComputeDevice* dev = testDevice();

    const float DT = 0.25f;                 // large and exact, so hand arithmetic is exact
    const float GRAV_X = 0.0f, GRAV_Y = -8.0f, GRAV_Z = 0.0f;

    // ---- the vertices -----------------------------------------------------------------
    // v0  free, plain integration               v1  PINNED (invMass 0)
    // v2  clamped by mMaxLinearVelocity         v3  clamp ARMED but isPrevPosSet == 0
    const uint32_t NV = 4u;
    std::vector<F4> pos = {
        F4{ 0.0f, 10.0f, 0.0f, 1.0f },       // invMass 1 -> gravity + damping apply
        F4{ 1.0f, 10.0f, 0.0f, 0.0f },       // invMass 0 -> PINNED
        // *** v2's INVERSE MASS IS 2, NOT 1, ON PURPOSE. ***  mPrevPositionInContactOffset is
        // written as the final position with .w OVERWRITTEN to 1 as a "reference is set" mark.
        // With invMass 1 that mark is indistinguishable from the mass that was already there,
        // and a port that drops it passes — measured, the mutant survived.  At invMass 2 the
        // position keeps .w = 2 and only the contact-offset copy reads 1.
        F4{ 2.0f, 10.0f, 0.0f, 2.0f },
        F4{ 3.0f, 10.0f, 0.0f, 1.0f },
    };
    std::vector<F4> vel = {
        F4{ 0.0f, 0.0f, 0.0f, 7.0f },        // .w carries a marker that must be PRESERVED
        F4{ 0.0f, 0.0f, 0.0f, 8.0f },
        F4{ 0.0f, -40.0f, 0.0f, 9.0f },      // fast: 40 * 0.25 = 10 of travel, way over the cap
        F4{ 0.0f, -40.0f, 0.0f, 10.0f },     // the same speed, but with no reference set
    };
    std::vector<F4> accum(NV, F4{ 0.0f, 0.0f, 0.0f, 5.0f });   // .w must be PRESERVED
    // mPrevPositionInContactOffset: .w == 1 marks the reference as set.  v3's is 0.
    std::vector<F4> prevContact = {
        F4{ 0.0f, 10.0f, 0.0f, 1.0f },
        F4{ 1.0f, 10.0f, 0.0f, 1.0f },
        F4{ 2.0f, 10.0f, 0.0f, 1.0f },
        F4{ 3.0f, 10.0f, 0.0f, 0.0f },       // NOT set
    };
    std::vector<F4> prevRest = pos;          // the clamp measures against THIS
    DevBuf bPos = makeDev(dev, pos), bVel = makeDev(dev, vel), bAcc = makeDev(dev, accum),
           bPrevC = makeDev(dev, prevContact), bPrevR = makeDev(dev, prevRest);
    if (!bPos.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    std::vector<uint32_t> vtCount = { 1234u }, eeCount = { 5678u };
    std::vector<uint32_t> updPairs = { 0xCDCDCDCDu };   // a PxU8 in a poisoned word
    DevBuf bVt = makeDev(dev, vtCount), bEe = makeDev(dev, eeCount), bUpd = makeDev(dev, updPairs);

    PxgFEMCloth cloth = makeZeroDesc();
    cloth.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    cloth.mVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVel.addr));
    cloth.mAccumulatedDeltaPos = reinterpret_cast<float4*>(size_t(bAcc.addr));
    cloth.mPrevPositionInContactOffset = reinterpret_cast<float4*>(size_t(bPrevC.addr));
    cloth.mPrevPositionInRestOffset = reinterpret_cast<float4*>(size_t(bPrevR.addr));
    cloth.mNbVerts = NV;
    cloth.mLinearDamping = 2.0f;             // damping * dt = 0.5 -> the factor is 0.5
    cloth.mMaxLinearVelocity = 8.0f;         // 8 * 0.25 = 2.0 of allowed travel
    cloth.mRestDistance = 0.0f;
    cloth.mOriginalContactOffset = 0.0f;

    // activeId is DELIBERATELY not the identity: gid.y 0 -> cloth 2.
    std::vector<PxgFEMCloth> descArr(3u, makeZeroDesc());
    descArr[2] = cloth;
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 2u };
    DevBuf bActive = makeDev(dev, active);

    ClothStepPush push = {};
    push.femCloths = bDesc.addr;
    push.activeId = bActive.addr;
    push.dt = DT;
    push.gx = GRAV_X; push.gy = GRAV_Y; push.gz = GRAV_Z;
    push.externalForces = 1u;
    push.adaptive = 0u;
    push.forceUpdate = 1u;
    push.updateContactPairs = bUpd.addr;
    push.vtContactCount = bVt.addr;
    push.eeContactCount = bEe.addr;

    Kernel k("femClothStepBda.spv", "cloth_stepLaunch", {}, sizeof(ClothStepPush));
    // The engine's grid (PxgFEMClothCore.cpp:1521): x = ceil(maxClothVerts / CLOTH_STEP),
    // y = nbActiveFEMCloths.
    k.dispatch(/*gridX*/ 1u, {}, push, /*gridY*/ 1u, 1u);
    dev->flush();

    const auto gPos = downloadBuffer<F4>(dev, bPos.h, NV);
    const auto gVel = downloadBuffer<F4>(dev, bVel.h, NV);
    const auto gAcc = downloadBuffer<F4>(dev, bAcc.h, NV);
    const auto gPrevC = downloadBuffer<F4>(dev, bPrevC.h, NV);
    const auto gPrevR = downloadBuffer<F4>(dev, bPrevR.h, NV);
    const uint32_t gVt = downloadBuffer<uint32_t>(dev, bVt.h, 1)[0];
    const uint32_t gEe = downloadBuffer<uint32_t>(dev, bEe.h, 1)[0];
    const uint32_t gUpd = downloadBuffer<uint32_t>(dev, bUpd.h, 1)[0];

    for (uint32_t i = 0; i < NV; ++i)
        std::printf("  v%u: pos=(%.5f %.5f %.5f | %.2f) vel=(%.5f %.5f %.5f | %.2f) "
                    "accum=(%.5f %.5f %.5f | %.2f)\n", i,
                    gPos[i].x, gPos[i].y, gPos[i].z, gPos[i].w,
                    gVel[i].x, gVel[i].y, gVel[i].z, gVel[i].w,
                    gAcc[i].x, gAcc[i].y, gAcc[i].z, gAcc[i].w);

    auto near = [](float a, float b, float t = 1e-5f) { return std::fabs(a - b) <= t; };

    // ---- v0: the plain path ------------------------------------------------------------
    // vel = (0 + (-8)*0.25) * (1 - min(1, 2*0.25)) = -2 * 0.5 = -1
    // pos = 10 + (-1)*0.25 = 9.75; the trajectory is 0.25, well inside the 2.0 cap.
    PXG_CHECK_MSG(near(gVel[0].y, -1.0f),
                  "v0: gravity gives -2 and the damping factor is 1 - min(1, 2*0.25) = 0.5, so "
                  "the velocity is -1; got %.6f (-2 means the damping was skipped)", gVel[0].y);
    PXG_CHECK_MSG(near(gPos[0].y, 9.75f), "v0: pos = 10 + (-1)*0.25; got %.6f", gPos[0].y);
    PXG_CHECK_MSG(near(gAcc[0].y, -0.25f),
                  "v0: the accumulated delta takes the SAME delta as the position; got %.6f",
                  gAcc[0].y);
    PXG_CHECK_MSG(near(gPos[0].w, 1.0f) && near(gVel[0].w, 7.0f) && near(gAcc[0].w, 5.0f),
                  "v0: the .w of all three buffers is carried through untouched (invMass, and "
                  "two markers); got %.2f %.2f %.2f", gPos[0].w, gVel[0].w, gAcc[0].w);

    // ---- v1: PINNED --------------------------------------------------------------------
    // invMass 0, so no gravity and no damping -- but the integration still runs, and with a
    // zero velocity that leaves it exactly where it was.
    PXG_CHECK_MSG(near(gVel[1].y, 0.0f) && near(gPos[1].y, 10.0f),
                  "v1 has invMass 0: the gravity/damping block must skip it entirely; got vel "
                  "%.6f pos %.6f", gVel[1].y, gPos[1].y);

    // ---- v2: the maximum-displacement clamp --------------------------------------------
    // vel = (-40 - 2) * 0.5 = -21; unclamped pos = 10 - 5.25 = 4.75, a trajectory of -5.25
    // against a cap of mMaxLinearVelocity * dt = 2.0.  correctionScale = 2/5.25, so the
    // correction is (2/5.25 - 1) * (-5.25) = +3.25 and the clamped position is 8.0.
    PXG_CHECK_MSG(near(gPos[2].y, 8.0f, 2e-5f),
                  "v2: the trajectory is clamped to mMaxLinearVelocity * dt = 2.0 below its "
                  "REST-offset reference of 10, i.e. 8.0; got %.6f (4.75 means the clamp never "
                  "fired)", gPos[2].y);
    // The correction reaches the velocity too: -21 + 3.25/0.25 = -21 + 13 = -8.
    PXG_CHECK_MSG(near(gVel[2].y, -8.0f, 2e-4f),
                  "v2: the clamp correction must be applied to the VELOCITY as well, scaled by "
                  "1/dt: -21 + 13 = -8; got %.6f (-21 means only the position was clamped)",
                  gVel[2].y);
    PXG_CHECK_MSG(near(gAcc[2].y, -2.0f, 2e-5f),
                  "v2: the accumulated delta takes the correction too; got %.6f", gAcc[2].y);

    // ---- v3: the same speed, but no reference ------------------------------------------
    PXG_CHECK_MSG(near(gPos[3].y, 4.75f, 2e-5f),
                  "v3 has isPrevPosSet == 0, so the clamp must NOT fire and the position is the "
                  "unclamped 4.75; got %.6f", gPos[3].y);

    // ---- the previous-position buffers -------------------------------------------------
    PXG_CHECK_MSG(near(gPrevR[0].y, gPos[0].y) && near(gPrevR[2].y, gPos[2].y),
                  "mPrevPositionInRestOffset is written for EVERY vertex, with the final "
                  "position");
    PXG_CHECK_MSG(near(gPrevC[0].y, gPos[0].y) && near(gPrevC[0].w, 1.0f),
                  "on a forced update mPrevPositionInContactOffset takes the final position "
                  "with .w MARKED 1; got y %.6f w %.2f", gPrevC[0].y, gPrevC[0].w);
    // v2 is where the mark is actually separable from the inverse mass — see the note above.
    PXG_CHECK_MSG(near(gPos[2].w, 2.0f),
                  "v2's position keeps its inverse mass 2 in .w; got %.2f", gPos[2].w);
    PXG_CHECK_MSG(near(gPrevC[2].w, 1.0f),
                  "v2's contact-offset copy has .w OVERWRITTEN to the mark 1, not left at its "
                  "inverse mass 2; got %.2f", gPrevC[2].w);
    PXG_CHECK_MSG(near(gPrevR[2].w, 2.0f),
                  "the mark belongs to the CONTACT-offset copy only: the rest-offset copy keeps "
                  "the inverse mass 2; got %.2f", gPrevR[2].w);
    PXG_CHECK_MSG(near(gPrevC[3].w, 1.0f),
                  "the mark is set for v3 too -- the forced update does not care whether a "
                  "reference existed before");

    // ---- the contact-pair bookkeeping --------------------------------------------------
    PXG_CHECK_MSG((gUpd & 0xffu) == 1u,
                  "updateContactPairs is a PxU8 and must read 1 after a forced update; got "
                  "0x%02x", gUpd & 0xffu);
    PXG_CHECK_MSG((gUpd & 0xffffff00u) == 0xCDCDCD00u,
                  "the PxU8 store must leave the other three bytes of the word alone; got "
                  "0x%08x", gUpd);
    PXG_CHECK_MSG(gVt == 0u && gEe == 0u,
                  "a forced update ZEROES both contact counts (1234/5678 before); got %u/%u",
                  gVt, gEe);

    for (PxgBufferHandle h : { bPos.h, bVel.h, bAcc.h, bPrevC.h, bPrevR.h, bVt.h, bEe.h,
                               bUpd.h, bDesc.h, bActive.h })
        dev->destroyBuffer(h);
}

// A second run of the same kernel with the OTHER settings of the three flags: external forces
// OFF, no forced update, and gravity disabled on the actor.  Kept separate because the flags
// are push constants, so one dispatch can only ever exercise one combination.
PXG_TEST(femClothSimStepFlags)
{
    using namespace p812step;
    PxgSlangComputeDevice* dev = testDevice();

    const float DT = 0.25f;
    const uint32_t NV = 2u;
    std::vector<F4> pos = { F4{ 0.0f, 10.0f, 0.0f, 1.0f }, F4{ 1.0f, 10.0f, 0.0f, 1.0f } };
    std::vector<F4> vel = { F4{ 0.0f, -4.0f, 0.0f, 0.0f }, F4{ 0.0f, -4.0f, 0.0f, 0.0f } };
    std::vector<F4> accum(NV, F4{ 0.0f, 0.0f, 0.0f, 0.0f });
    std::vector<F4> prevContact(NV, F4{ 0.0f, 10.0f, 0.0f, 1.0f });
    std::vector<F4> prevRest = pos;
    DevBuf bPos = makeDev(dev, pos), bVel = makeDev(dev, vel), bAcc = makeDev(dev, accum),
           bPrevC = makeDev(dev, prevContact), bPrevR = makeDev(dev, prevRest);
    if (!bPos.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    std::vector<uint32_t> vtCount = { 1234u }, eeCount = { 5678u };
    std::vector<uint32_t> updPairs = { 0x000000FFu };
    DevBuf bVt = makeDev(dev, vtCount), bEe = makeDev(dev, eeCount), bUpd = makeDev(dev, updPairs);

    PxgFEMCloth cloth = makeZeroDesc();
    cloth.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    cloth.mVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVel.addr));
    cloth.mAccumulatedDeltaPos = reinterpret_cast<float4*>(size_t(bAcc.addr));
    cloth.mPrevPositionInContactOffset = reinterpret_cast<float4*>(size_t(bPrevC.addr));
    cloth.mPrevPositionInRestOffset = reinterpret_cast<float4*>(size_t(bPrevR.addr));
    cloth.mNbVerts = NV;
    cloth.mLinearDamping = 2.0f;
    // PX_MAX_REAL: the maximum-displacement block is skipped ENTIRELY, however far the vertex
    // travels.  Without this regime the port could clamp everything and still look right above.
    cloth.mMaxLinearVelocity = 3.40282347e+38f;
    std::vector<PxgFEMCloth> descArr = { cloth };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    ClothStepPush push = {};
    push.femCloths = bDesc.addr;
    push.activeId = bActive.addr;
    push.dt = DT;
    push.gx = 0.0f; push.gy = -8.0f; push.gz = 0.0f;
    push.externalForces = 0u;      // NO gravity predict at all
    push.adaptive = 0u;
    push.forceUpdate = 0u;         // NO forced update
    push.updateContactPairs = bUpd.addr;
    push.vtContactCount = bVt.addr;
    push.eeContactCount = bEe.addr;

    Kernel k("femClothStepBda.spv", "cloth_stepLaunch", {}, sizeof(ClothStepPush));
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    const auto gPos = downloadBuffer<F4>(dev, bPos.h, NV);
    const auto gVel = downloadBuffer<F4>(dev, bVel.h, NV);
    const auto gPrevC = downloadBuffer<F4>(dev, bPrevC.h, NV);
    const uint32_t gVt = downloadBuffer<uint32_t>(dev, bVt.h, 1)[0];
    const uint32_t gEe = downloadBuffer<uint32_t>(dev, bEe.h, 1)[0];
    const uint32_t gUpd = downloadBuffer<uint32_t>(dev, bUpd.h, 1)[0];

    auto near = [](float a, float b, float t = 1e-5f) { return std::fabs(a - b) <= t; };

    PXG_CHECK_MSG(near(gVel[0].y, -4.0f),
                  "with externalForcesEveryTgsIterationEnabled FALSE the velocity is left "
                  "alone -- no gravity, no damping; got %.6f", gVel[0].y);
    // pos = 10 + (-4)*0.25 = 9.0, and with mMaxLinearVelocity == PX_MAX_REAL nothing clamps it
    // even though the travel is 1.0.
    PXG_CHECK_MSG(near(gPos[0].y, 9.0f),
                  "the integration still runs with the predict off, and PX_MAX_REAL disables "
                  "the displacement clamp entirely; got %.6f", gPos[0].y);
    PXG_CHECK_MSG(near(gPrevC[0].y, 10.0f),
                  "without a forced update mPrevPositionInContactOffset must be LEFT ALONE at "
                  "its old value 10; got %.6f", gPrevC[0].y);
    PXG_CHECK_MSG((gUpd & 0xffu) == 0u,
                  "updateContactPairs must be cleared to 0 when forceUpdate is false; got "
                  "0x%02x", gUpd & 0xffu);
    PXG_CHECK_MSG(gVt == 1234u && gEe == 5678u,
                  "the contact counts must NOT be zeroed when forceUpdate is false; got %u/%u",
                  gVt, gEe);

    for (PxgBufferHandle h : { bPos.h, bVel.h, bAcc.h, bPrevC.h, bPrevR.h, bVt.h, bEe.h,
                               bUpd.h, bDesc.h, bActive.h })
        dev->destroyBuffer(h);
}

// A third run, for the three branches the first two cannot reach: the damping CLAMP (which only
// bites when linearDamping * dt exceeds 1), the eDISABLE_GRAVITY actor flag, and the NEGATIVE
// mMaxLinearVelocity regime that measures against 0.8 * mRestDistance instead of a velocity.
// Two clothes and gridY = 2, since all three live on the descriptor rather than in the blob.
PXG_TEST(femClothSimStepDampingAndFlags)
{
    using namespace p812step;
    PxgSlangComputeDevice* dev = testDevice();

    const float DT = 0.25f;

    // ---- cloth A: gravity ON, damping 8 (so damping*dt = 2, past the clamp), and the
    // NEGATIVE mMaxLinearVelocity regime with restDistance 5 -> maxDistance = 0.8 * 5 = 4.
    std::vector<F4> posA = { F4{ 0.0f, 10.0f, 0.0f, 1.0f } };
    std::vector<F4> velA = { F4{ 0.0f, -40.0f, 0.0f, 0.0f } };
    std::vector<F4> accA = { F4{ 0.0f, 0.0f, 0.0f, 0.0f } };
    std::vector<F4> prevCA = { F4{ 0.0f, 10.0f, 0.0f, 1.0f } };
    std::vector<F4> prevRA = posA;
    DevBuf bPosA = makeDev(dev, posA), bVelA = makeDev(dev, velA), bAccA = makeDev(dev, accA),
           bPrevCA = makeDev(dev, prevCA), bPrevRA = makeDev(dev, prevRA);
    if (!bPosA.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // ---- cloth B: gravity DISABLED on the actor, no damping, no clamp.
    std::vector<F4> posB = { F4{ 5.0f, 10.0f, 0.0f, 1.0f } };
    std::vector<F4> velB = { F4{ 0.0f, 0.0f, 0.0f, 0.0f } };
    std::vector<F4> accB = { F4{ 0.0f, 0.0f, 0.0f, 0.0f } };
    std::vector<F4> prevCB = { F4{ 5.0f, 10.0f, 0.0f, 1.0f } };
    std::vector<F4> prevRB = posB;
    DevBuf bPosB = makeDev(dev, posB), bVelB = makeDev(dev, velB), bAccB = makeDev(dev, accB),
           bPrevCB = makeDev(dev, prevCB), bPrevRB = makeDev(dev, prevRB);

    // ---- cloth C: the one vertex that separates the clamp's three moving parts.
    // It starts at y = 10 with velocity -40, so dt = 0.25 gives a raw delta of -10 and an
    // unclamped position of 0.  Its REST-offset reference is deliberately NOT its start
    // position but y = 12, and mMaxLinearVelocity is -1 with mRestDistance 5, so the cap is
    // 0.8 * 5 = 4.  Three readings, three distinct answers:
    //     correct        trajectory = 0 - 12 = -12, cap 4   -> pos 8.0
    //     cap = maxVel*dt (-0.25)                            -> pos 11.75
    //     trajectory = this substep's delta (-10)            -> pos 6.0
    // Cloth A cannot do this job: its damping kills the velocity, so it never travels.
    std::vector<F4> posC = { F4{ 9.0f, 10.0f, 0.0f, 1.0f } };
    std::vector<F4> velC = { F4{ 0.0f, -40.0f, 0.0f, 0.0f } };
    std::vector<F4> accC = { F4{ 0.0f, 0.0f, 0.0f, 0.0f } };
    std::vector<F4> prevCC = { F4{ 9.0f, 10.0f, 0.0f, 1.0f } };   // .w = 1: the reference IS set
    std::vector<F4> prevRC = { F4{ 9.0f, 12.0f, 0.0f, 1.0f } };   // NOT the start position
    DevBuf bPosC = makeDev(dev, posC), bVelC = makeDev(dev, velC), bAccC = makeDev(dev, accC),
           bPrevCC = makeDev(dev, prevCC), bPrevRC = makeDev(dev, prevRC);

    std::vector<uint32_t> vtCount = { 0u }, eeCount = { 0u }, updPairs = { 0u };
    DevBuf bVt = makeDev(dev, vtCount), bEe = makeDev(dev, eeCount), bUpd = makeDev(dev, updPairs);

    PxgFEMCloth cA = makeZeroDesc();
    cA.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bPosA.addr));
    cA.mVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVelA.addr));
    cA.mAccumulatedDeltaPos = reinterpret_cast<float4*>(size_t(bAccA.addr));
    cA.mPrevPositionInContactOffset = reinterpret_cast<float4*>(size_t(bPrevCA.addr));
    cA.mPrevPositionInRestOffset = reinterpret_cast<float4*>(size_t(bPrevRA.addr));
    cA.mNbVerts = 1u;
    cA.mLinearDamping = 8.0f;            // 8 * 0.25 = 2 -> min() clamps it to 1, factor 0
    cA.mMaxLinearVelocity = -1.0f;       // the NEGATIVE regime
    cA.mRestDistance = 5.0f;             // -> maxDistance = 4
    cA.mActorFlags = 0u;

    PxgFEMCloth cB = makeZeroDesc();
    cB.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bPosB.addr));
    cB.mVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVelB.addr));
    cB.mAccumulatedDeltaPos = reinterpret_cast<float4*>(size_t(bAccB.addr));
    cB.mPrevPositionInContactOffset = reinterpret_cast<float4*>(size_t(bPrevCB.addr));
    cB.mPrevPositionInRestOffset = reinterpret_cast<float4*>(size_t(bPrevRB.addr));
    cB.mNbVerts = 1u;
    cB.mLinearDamping = 0.0f;
    cB.mMaxLinearVelocity = 3.40282347e+38f;
    cB.mActorFlags = PxU8(PxActorFlag::eDISABLE_GRAVITY);

    PxgFEMCloth cC = makeZeroDesc();
    cC.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bPosC.addr));
    cC.mVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVelC.addr));
    cC.mAccumulatedDeltaPos = reinterpret_cast<float4*>(size_t(bAccC.addr));
    cC.mPrevPositionInContactOffset = reinterpret_cast<float4*>(size_t(bPrevCC.addr));
    cC.mPrevPositionInRestOffset = reinterpret_cast<float4*>(size_t(bPrevRC.addr));
    cC.mNbVerts = 1u;
    cC.mLinearDamping = 0.0f;
    cC.mMaxLinearVelocity = -1.0f;       // the NEGATIVE regime, WITH travel this time
    cC.mRestDistance = 5.0f;             // -> maxDistance = 0.8 * 5 = 4
    cC.mActorFlags = PxU8(PxActorFlag::eDISABLE_GRAVITY);   // keep the arithmetic exact

    std::vector<PxgFEMCloth> descArr = { cA, cB, cC };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u, 1u, 2u };
    DevBuf bActive = makeDev(dev, active);

    ClothStepPush push = {};
    push.femCloths = bDesc.addr;
    push.activeId = bActive.addr;
    push.dt = DT;
    push.gx = 0.0f; push.gy = -8.0f; push.gz = 0.0f;
    push.externalForces = 1u;
    push.adaptive = 0u;
    push.forceUpdate = 0u;
    push.updateContactPairs = bUpd.addr;
    push.vtContactCount = bVt.addr;
    push.eeContactCount = bEe.addr;

    Kernel k("femClothStepBda.spv", "cloth_stepLaunch", {}, sizeof(ClothStepPush));
    k.dispatch(/*gridX*/ 1u, {}, push, /*gridY*/ 3u, 1u);
    dev->flush();

    const auto gPosA = downloadBuffer<F4>(dev, bPosA.h, 1);
    const auto gVelA = downloadBuffer<F4>(dev, bVelA.h, 1);
    const auto gPosB = downloadBuffer<F4>(dev, bPosB.h, 1);
    const auto gVelB = downloadBuffer<F4>(dev, bVelB.h, 1);
    std::printf("  A: pos.y=%.5f vel.y=%.5f   B: pos.y=%.5f vel.y=%.5f\n",
                gPosA[0].y, gVelA[0].y, gPosB[0].y, gVelB[0].y);

    auto near = [](float a, float b, float t = 1e-5f) { return std::fabs(a - b) <= t; };

    // ---- cloth A ------------------------------------------------------------------------
    // vel = (-40 + (-8)*0.25) * (1 - min(1, 8*0.25)) = -42 * 0 = 0.
    // WITHOUT the min() the factor is 1 - 2 = -1 and the velocity comes out at +42 -- the
    // damping would REVERSE the motion instead of killing it.
    PXG_CHECK_MSG(near(gVelA[0].y, 0.0f),
                  "cloth A: linearDamping * dt = 2, so min(1, .) clamps the factor to 0 and the "
                  "velocity is killed outright; got %.6f (+42 means the min() is missing and "
                  "the damping REVERSED the velocity)", gVelA[0].y);
    // With the velocity zeroed there is no travel, so the clamp cannot be seen on A's position;
    // what IS seen is that a maxLinearVelocity of -1 did not disable the block or divide by it.
    PXG_CHECK_MSG(near(gPosA[0].y, 10.0f),
                  "cloth A does not move once the damping has killed its velocity; got %.6f",
                  gPosA[0].y);

    // ---- cloth B ------------------------------------------------------------------------
    // eDISABLE_GRAVITY, damping 0: the velocity stays exactly 0 and so does the position.
    PXG_CHECK_MSG(near(gVelB[0].y, 0.0f),
                  "cloth B has eDISABLE_GRAVITY, so gravity must NOT be added; got %.6f "
                  "(-2 means the actor flag was ignored)", gVelB[0].y);
    PXG_CHECK_MSG(near(gPosB[0].y, 10.0f),
                  "cloth B therefore does not move; got %.6f", gPosB[0].y);

    // ---- cloth C: the clamp's three moving parts, separated --------------------------
    const auto gPosC = downloadBuffer<F4>(dev, bPosC.h, 1);
    const auto gVelC = downloadBuffer<F4>(dev, bVelC.h, 1);
    std::printf("  C: pos.y=%.5f vel.y=%.5f\n", gPosC[0].y, gVelC[0].y);
    PXG_CHECK_MSG(near(gPosC[0].y, 8.0f, 2e-5f),
                  "cloth C: the cap is 0.8 * mRestDistance = 4 and the trajectory is measured "
                  "from the REST-offset reference at 12, so the clamped position is 8.0; got "
                  "%.6f.  11.75 means the negative-maxVel regime used maxVel*dt instead; 6.0 "
                  "means the trajectory was measured from this substep's start", gPosC[0].y);
    // vel = -40 + 8/0.25 = -8.
    PXG_CHECK_MSG(near(gVelC[0].y, -8.0f, 2e-4f),
                  "cloth C's velocity takes the same correction scaled by 1/dt; got %.6f",
                  gVelC[0].y);

    // ---- ONE MUTANT SURVIVES, AND THE ARITHMETIC SAYS WHY ---------------------------
    // Deleting the `mMaxLinearVelocity < PX_MAX_REAL` guard that skips the whole clamp block
    // changes nothing measurable, because the block is a no-op at PX_MAX_REAL anyway: the cap
    // becomes PX_MAX_REAL * dt, its SQUARE overflows to infinity, and `displacementSq > inf` is
    // false for every finite displacement.  The guard is an early-out, not a behaviour switch.
    // No fixture can separate the two without a displacement of order 1e19, which no scene
    // produces.  Kept as the faithful transcription and recorded rather than papered over.

    for (PxgBufferHandle h : { bPosA.h, bVelA.h, bAccA.h, bPrevCA.h, bPrevRA.h,
                               bPosB.h, bVelB.h, bAccB.h, bPrevCB.h, bPrevRB.h,
                               bPosC.h, bVelC.h, bAccC.h, bPrevCC.h, bPrevRC.h,
                               bVt.h, bEe.h, bUpd.h, bDesc.h, bActive.h })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P8.16 -- cloth_differentClothCollision{VT,EE}Launch, the cloth-vs-DIFFERENT-cloth
// midphase.  The sibling femClothSelfCollisionBda deferred these as "a SEPARATE arg
// pack / ABI"; the traverser bodies are the same C++ structs, so what this fixture has
// to pin is exactly the five things that differ (femClothClothMidPhase.cu:1240-1384):
//
//   * the 88-byte narrowphase-driven blob (cmInputs / gpuShapes / transformCache instead
//     of an activeClothes list), and that the two cloth ids come from
//     PxgShape::particleOrSoftbodyId;
//   * contactDist is the SUM of the two shapes' cached distances, not one doubled --
//     covered by a pair that is inside the sum but outside 2*cd[ref1], and another that
//     is outside the sum but inside 2*cd[ref0];
//   * VT sweeps BOTH directions in one launch (globalWarpIdx < numVerts0 + numVerts1),
//     each half walking the OTHER cloth's tree; EE runs min(numTri0, numTri1) queries
//     with the SMALLER cloth always driving, which the second EE dispatch proves by
//     swapping the pair order and demanding the identical output;
//   * every `clothId0 != clothId1` guard is live and TRUE, so the self path's three skips
//     are disabled.  Both are exercised on purpose: the VT scene makes a penetrating
//     vertex share its INDEX with a corner of the triangle it hits, and the EE scene gives
//     the two cloths triangles with identical vertex indices, so a mutant that folds
//     either guard the self way loses contacts;
//   * markValidity(clothId0 != clothId1) -> bit 31 SET, where the self path leaves it 0.
//
// The rest-pose filter is absent by construction (the call passes NULL rest buffers and
// filterDistSq 0), so the descriptors below leave mRestPosition null -- a port that kept
// the self-collision lookup would fault on the first contact rather than mis-filter.
// ===========================================================================
namespace p816dc
{
struct DifferentClothPush
{
    uint32_t maxContacts;
    float toleranceLength;
    uint64_t cmInputs, transformCache, contactDistance, gpuShapes, femClothes, clothClothPairs;
    uint32_t numClothClothPairs, pad60;
    uint64_t updateContactPairs, outContactInfo, totalContactCount;
};
static_assert(sizeof(DifferentClothPush) == 88, "different-cloth push blob must be 88 bytes");
static_assert(offsetof(DifferentClothPush, cmInputs) == 8, "cmInputs @8");
static_assert(offsetof(DifferentClothPush, femClothes) == 40, "femClothes @40");
static_assert(offsetof(DifferentClothPush, numClothClothPairs) == 56, "numClothClothPairs @56");
static_assert(offsetof(DifferentClothPush, updateContactPairs) == 64, "the next pointer realigns @64");
static_assert(offsetof(DifferentClothPush, totalContactCount) == 80, "totalContactCount @80");

struct CmInputHost { uint32_t shapeRef0, shapeRef1, cacheRef0, cacheRef1; };
struct PxgShapeHost
{
    float sx, sy, sz, srx, sry, srz, srw;
    uint32_t materialIndex, hullLo, hullHi, type, particleOrSoftbodyId;
};
static_assert(sizeof(PxgShapeHost) == 48, "PxgShape mirror must be 48 bytes");
static const uint32_t GEOM_TRIANGLEMESH_H = 8u;   // PxGeometryType::eTRIANGLEMESH

static const uint32_t VALID_H = 1u << 31;         // markValidity(true)

static PxgShapeHost makeClothShape(uint32_t clothId)
{
    PxgShapeHost s = {};
    s.sx = s.sy = s.sz = 1.0f;
    s.srw = 1.0f;
    s.type = GEOM_TRIANGLEMESH_H;
    s.particleOrSoftbodyId = clothId;
    return s;
}

// Sort key for the SET comparison: unlike the self-collision fixtures (one query warp, one
// leaf) this kernel runs several query warps against a shared atomic counter, so the output
// ORDER is not deterministic.  The contact set is.
static bool infoLess(const FemFemContactInfoHost& a, const FemFemContactInfoHost& b)
{
    if (a.pairInd0 != b.pairInd0) return a.pairInd0 < b.pairInd0;
    if (a.pairInd1 != b.pairInd1) return a.pairInd1 < b.pairInd1;
    return a.auxInd < b.auxInd;
}

static size_t compareSets(std::vector<FemFemContactInfoHost> got,
                          std::vector<FemFemContactInfoHost> ref, const char* tag)
{
    std::sort(got.begin(), got.end(), infoLess);
    std::sort(ref.begin(), ref.end(), infoLess);
    size_t mm = 0;
    for (size_t i = 0; i < got.size() && i < ref.size(); ++i)
    {
        if (std::memcmp(&got[i], &ref[i], sizeof(FemFemContactInfoHost)) != 0)
        {
            ++mm;
            std::printf("  %s contact[%zu] MM: got(p0=%llu,p1=%u,aux=0x%08x) ref(p0=%llu,p1=%u,aux=0x%08x)\n",
                        tag, i, (unsigned long long)got[i].pairInd0, got[i].pairInd1, got[i].auxInd,
                        (unsigned long long)ref[i].pairInd0, ref[i].pairInd1, ref[i].auxInd);
        }
    }
    return mm;
}
} // namespace p816dc

// ---------------------------------------------------------------------------
// cloth_differentClothCollisionVTLaunch — both directions in one launch.
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimDifferentClothVT)
{
    using namespace p816dc;
    using p72ref::V3;
    PxgSlangComputeDevice* dev = testDevice();

    // contactDist = cd[cacheRef0] + cd[cacheRef1] = 0.05 + 0.02 = 0.07.
    const float CD0 = 0.05f, CD1 = 0.02f;
    const float contactDist = CD0 + CD1;
    const float contactDistSq = contactDist * contactDist;

    // Cloth 0: triangle A = {0,1,2} in the z=0 plane, plus two probe vertices above cloth 1's
    // triangle.  v4 is 0.03 away (inside 0.07); v3 is 0.085 away -- OUTSIDE the summed
    // distance but INSIDE 2*CD0 = 0.10, which is the mutant that vertex exists to kill.
    //
    // The PENETRATING probe is the LAST vertex on purpose.  Cloth 0 has five vertices and
    // cloth 1 four, so the two halves of the query range have different lengths -- and an
    // inverted branchless select (which swaps which half is which) then drops index 4
    // entirely.  With the probe at index 3 that mutant SURVIVES: both halves still cover
    // index 3, and the emitted ids come out identical because the select swaps my/other
    // consistently.  The asymmetry is the only thing that separates them.
    std::vector<F4> pos0 = {
        {  0.00f, 0.00f, 0.000f, 1.0f },   // v0 (triangle A)
        {  1.00f, 0.00f, 0.000f, 1.0f },   // v1 (triangle A)
        {  0.00f, 1.00f, 0.000f, 1.0f },   // v2 (triangle A)
        {  0.25f, 0.25f, 0.585f, 1.0f },   // v3 -> dist 0.085, must MISS
        {  0.25f, 0.25f, 0.530f, 1.0f },   // v4 -> hits cloth 1's triangle B (dist 0.03)
    };
    std::vector<TriIdx4> tris0 = { { 0u, 1u, 2u, 0x7u } };

    // Cloth 1: the penetrating vertex is index 0 ON PURPOSE -- it equals triangle A's .x, so
    // the self path's "vertex is a corner of this triangle" skip would fire.  Here it must not.
    std::vector<F4> pos1 = {
        {  0.25f, 0.25f, 0.040f, 1.0f },   // w0 -> hits cloth 0's triangle A (dist 0.04)
        {  0.00f, 0.00f, 0.500f, 1.0f },   // w1 (triangle B)
        {  1.00f, 0.00f, 0.500f, 1.0f },   // w2 (triangle B)
        {  0.00f, 1.00f, 0.500f, 1.0f },   // w3 (triangle B)
    };
    std::vector<TriIdx4> tris1 = { { 1u, 2u, 3u, 0x7u } };

    std::vector<uint32_t> tree0 = makeBv32TreeWords(/*nbTriangles*/ 1u, -1000.0f, 1000.0f);
    std::vector<uint32_t> tree1 = makeBv32TreeWords(/*nbTriangles*/ 1u, -1000.0f, 1000.0f);

    DevBuf bPos0 = makeDev(dev, pos0), bTri0 = makeDev(dev, tris0), bTree0 = makeDev(dev, tree0);
    DevBuf bPos1 = makeDev(dev, pos1), bTri1 = makeDev(dev, tris1), bTree1 = makeDev(dev, tree1);
    if (!bPos0.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgFEMCloth c0 = makeZeroDesc();
    c0.mTriMeshData = reinterpret_cast<void*>(size_t(bTree0.addr));
    c0.mPosition_InvMass = reinterpret_cast<decltype(c0.mPosition_InvMass)>(size_t(bPos0.addr));
    c0.mTriangleVertexIndices = reinterpret_cast<decltype(c0.mTriangleVertexIndices)>(size_t(bTri0.addr));
    c0.mNbVerts = uint32_t(pos0.size());
    c0.mElementIndex = 0u;   // unused on this path (contactDist comes from the cache refs)

    PxgFEMCloth c1 = makeZeroDesc();
    c1.mTriMeshData = reinterpret_cast<void*>(size_t(bTree1.addr));
    c1.mPosition_InvMass = reinterpret_cast<decltype(c1.mPosition_InvMass)>(size_t(bPos1.addr));
    c1.mTriangleVertexIndices = reinterpret_cast<decltype(c1.mTriangleVertexIndices)>(size_t(bTri1.addr));
    c1.mNbVerts = uint32_t(pos1.size());
    c1.mElementIndex = 1u;
    // mRestPosition stays NULL: the different-cloth call has no rest-pose filter.

    std::vector<PxgFEMCloth> descArr = { c0, c1 };
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<PxgShapeHost> shapes = { makeClothShape(0u), makeClothShape(1u) };
    DevBuf bShapes = makeDev(dev, shapes);
    std::vector<CmInputHost> cms = { { 0u, 1u, 0u, 1u } };   // shapeRef0/1 = 0/1, cacheRef0/1 = 0/1
    DevBuf bCm = makeDev(dev, cms);
    std::vector<float> cd = { CD0, CD1 };
    DevBuf bCd = makeDev(dev, cd);
    std::vector<uint32_t> upd = { 1u };
    DevBuf bUpd = makeDev(dev, upd);

    const uint32_t MAXC = 64u;
    std::vector<FemFemContactInfoHost> outInfo(MAXC);
    std::memset(outInfo.data(), 0xAB, outInfo.size() * sizeof(FemFemContactInfoHost));   // poison
    DevBuf bOut = makeDev(dev, outInfo);
    std::vector<uint32_t> count = { 0u };
    DevBuf bCount = makeDev(dev, count);

    DifferentClothPush push = {};
    push.maxContacts = MAXC;
    push.toleranceLength = 1.0f;
    push.cmInputs = bCm.addr;
    push.transformCache = 0ull;      // unused (ABI parity)
    push.contactDistance = bCd.addr;
    push.gpuShapes = bShapes.addr;
    push.femClothes = bDesc.addr;
    push.clothClothPairs = 0ull;     // unused (ABI parity)
    push.numClothClothPairs = 0u;
    push.updateContactPairs = bUpd.addr;
    push.outContactInfo = bOut.addr;
    push.totalContactCount = bCount.addr;

    Kernel k("femClothDifferentClothBda.spv", "cloth_differentClothCollisionVTLaunch", {},
             sizeof(DifferentClothPush));
    // GRID MIRRORS THE ENGINE: numBlocks = ceil(2*maxClothVerts / MIDPHASE_WARPS_PER_BLOCK),
    // gridY = numTests (PxgFEMClothCore.cpp:790).  maxClothVerts = 5 -> 5 blocks, covering
    // globalWarpIdx 0..9 against numQueries = 5 + 4 = 9.  Handing the shader a per-query grid
    // instead would hide exactly the half-coverage bug the self-collision port shipped with.
    const uint32_t maxClothVerts = uint32_t(pos0.size() > pos1.size() ? pos0.size() : pos1.size());
    const uint32_t gridVT = (2u * maxClothVerts + 1u) / 2u;
    k.dispatch(gridVT, {}, push, /*gridY numTests*/ 1u, 1u);
    dev->flush();

    // ---- CPU reference: the full bidirectional sweep, predicate for predicate ----
    auto v3f = [](const F4& f) { return V3{ f.x, f.y, f.z }; };
    std::vector<FemFemContactInfoHost> ref;
    for (uint32_t q = 0; q < uint32_t(pos0.size() + pos1.size()); ++q)
    {
        const bool isCloth0 = q < uint32_t(pos0.size());
        const uint32_t vertexIdx = isCloth0 ? q : q - uint32_t(pos0.size());
        const uint32_t myId = isCloth0 ? 0u : 1u, otherId = isCloth0 ? 1u : 0u;
        const std::vector<F4>& myPos = isCloth0 ? pos0 : pos1;
        const std::vector<F4>& otherPos = isCloth0 ? pos1 : pos0;
        const std::vector<TriIdx4>& otherTris = isCloth0 ? tris1 : tris0;
        const V3 vtx = v3f(myPos[vertexIdx]);
        for (uint32_t p = 0; p < uint32_t(otherTris.size()); ++p)
        {
            const TriIdx4& tri = otherTris[p];
            const bool vtInTri = (vertexIdx == tri.x) || (vertexIdx == tri.y) || (vertexIdx == tri.z);
            if (!((myId != otherId) || !vtInTri)) continue;   // the skip that is DEAD here
            const V3 x1 = v3f(otherPos[tri.x]), x2 = v3f(otherPos[tri.y]), x3 = v3f(otherPos[tri.z]);
            const V3 fat{ contactDist, contactDist, contactDist };
            const V3 bMin = p72ref::sub(p72ref::vmin(p72ref::vmin(x1, x2), x3), fat);
            const V3 bMax = p72ref::add(p72ref::vmax(p72ref::vmax(x1, x2), x3), fat);
            if (vtx.x < bMin.x || vtx.x > bMax.x || vtx.y < bMin.y || vtx.y > bMax.y ||
                vtx.z < bMin.z || vtx.z > bMax.z) continue;
            float s, t;
            const V3 cp = p72ref::closestPtPointTriangle(vtx, x1, x2, x3, s, t);
            const V3 dir = p72ref::sub(vtx, cp);
            const float dq = p72ref::dot(dir, dir);
            if (!(dq < contactDistSq && dq > 1.0e-14f)) continue;
            ref.push_back(FemFemContactInfoHost{ (uint64_t(myId) << 20) | vertexIdx,
                                                 (otherId << 20) | p, VALID_H });
        }
    }
    PXG_CHECK_MSG(ref.size() == 2u, "reference: expected 2 bidirectional VT contacts, got %zu", ref.size());
    // Pin what each of the two IS, so a reference that drifts with the shader is caught.
    PXG_CHECK_MSG(ref[0].pairInd0 == 4ull && ref[0].pairInd1 == (1u << 20),
                  "reference[0] must be cloth0 vertex 4 vs cloth1 triangle 0");
    PXG_CHECK_MSG(ref[1].pairInd0 == (1ull << 20) && ref[1].pairInd1 == 0u,
                  "reference[1] must be cloth1 vertex 0 vs cloth0 triangle 0 (the corner-index case)");

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
    PXG_CHECK_MSG(gCount == uint32_t(ref.size()), "different-cloth VT emitted %u contacts (expected %zu)",
                  gCount, ref.size());
    const auto got = downloadBuffer<FemFemContactInfoHost>(dev, bOut.h, gCount ? gCount : 1u);
    const size_t mm = compareSets(std::vector<FemFemContactInfoHost>(got.begin(), got.begin() + gCount),
                                  ref, "VT");
    PXG_CHECK_MSG(mm == 0, "cloth_differentClothCollisionVT bit-exact vs CPU (%zu mismatches)", mm);

    // ---- *updateContactPairs == 0 -> the whole launch is a no-op ----
    std::vector<uint32_t> upd0 = { 0u };
    DevBuf bUpd0 = makeDev(dev, upd0);
    std::vector<uint32_t> count0 = { 0u };
    DevBuf bCount0 = makeDev(dev, count0);
    DifferentClothPush pushOff = push;
    pushOff.updateContactPairs = bUpd0.addr;
    pushOff.totalContactCount = bCount0.addr;
    k.dispatch(gridVT, {}, pushOff, 1u, 1u);
    dev->flush();
    PXG_CHECK_MSG(downloadBuffer<uint32_t>(dev, bCount0.h, 1)[0] == 0u,
                  "different-cloth VT must early-exit when updateContactPairs is 0");

    for (PxgBufferHandle h : { bPos0.h, bTri0.h, bTree0.h, bPos1.h, bTri1.h, bTree1.h, bDesc.h,
                               bShapes.h, bCm.h, bCd.h, bUpd.h, bOut.h, bCount.h, bUpd0.h, bCount0.h })
        dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// cloth_differentClothCollisionEELaunch — the SMALLER cloth drives, both pair orders.
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimDifferentClothEE)
{
    using namespace p816dc;
    using p72ref::V3;
    PxgSlangComputeDevice* dev = testDevice();

    // contactDist = 0.06 + 0.04 = 0.10.  The threshold is chosen against the two crossings'
    // ACTUAL distSq values (0.006791 and 0.004851) rather than their geometric separation:
    // closestPtLineLine normalizes b0/b1 by DIFFERENT edge lengths (deformableCollision.cuh:244),
    // so its distSq is not the true minimum distance between the segments.  Halving to
    // 2*CD1 = 0.08 drops the first crossing -- which is what makes the summed contact distance
    // observable here.
    const float CD0 = 0.06f, CD1 = 0.04f;
    const float contactDist = CD0 + CD1;              // 0.10
    const float contactDistSq = contactDist * contactDist;

    // Cloth 0 (ONE triangle-with-active-edges -> always the query side): edge0 is the x-axis
    // segment at y=z=0; the other two edges dive to z=-1.
    std::vector<F4> pos0 = {
        { -0.5f, 0.0f,  0.0f, 1.0f },
        {  0.5f, 0.0f,  0.0f, 1.0f },
        {  0.0f, 0.0f, -1.0f, 1.0f },
    };
    std::vector<TriIdx4> tris0 = { { 0u, 1u, 2u, 0x7u } };
    std::vector<uint32_t> twae0 = { 0u };

    // Cloth 1 (TWO triangles-with-active-edges, so it is never the query side):
    //   tri 0 far away -- proves the leaf walk covers more than one primitive;
    //   tri 1 at z=0.03 -- its edges 0 and 2 both cross cloth 0's edge0, and BOTH land on the
    //     same lane, so this is also the multi-element write loop's coverage.  Its vertex
    //     indices are 0/1/2, IDENTICAL to cloth 0's triangle, so a mutant that keeps the self
    //     path's connected-edge skip drops both crossings.  z=0.03 is beyond cloth 0's RAW
    //     bounds (z-max 0), so it also pins the t0_bounds fattening.
    //
    // The crossing triangle is primitive ONE, not zero, on purpose.  Cloth 0's tree holds a
    // single primitive, so a mutant that walks the QUERY cloth's tree instead of the other
    // one still visits index 0 -- and with the crossing triangle at index 0 it reads exactly
    // the right triangle and SURVIVES.  Putting the crossing at index 1 makes the wrong tree
    // find the far triangle instead.
    std::vector<F4> pos1 = {
        {  0.0f,  0.3f, 0.030f, 1.0f },
        {  0.2f, -0.3f, 0.030f, 1.0f },
        { -0.2f, -0.3f, 0.030f, 1.0f },
        { 10.0f, 10.0f, 10.0f,  1.0f },
        { 11.0f, 10.0f, 10.0f,  1.0f },
        { 10.0f, 11.0f, 10.0f,  1.0f },
    };
    std::vector<TriIdx4> tris1 = { { 3u, 4u, 5u, 0x7u }, { 0u, 1u, 2u, 0x7u } };
    std::vector<uint32_t> twae1 = { 0u, 1u };

    std::vector<uint32_t> tree0 = makeBv32TreeWords(/*nbTriangles*/ 1u, -1000.0f, 1000.0f);
    std::vector<uint32_t> tree1 = makeBv32TreeWords(/*nbTriangles*/ 2u, -1000.0f, 1000.0f);

    DevBuf bPos0 = makeDev(dev, pos0), bTri0 = makeDev(dev, tris0), bTwae0 = makeDev(dev, twae0),
           bTree0 = makeDev(dev, tree0);
    DevBuf bPos1 = makeDev(dev, pos1), bTri1 = makeDev(dev, tris1), bTwae1 = makeDev(dev, twae1),
           bTree1 = makeDev(dev, tree1);
    if (!bPos0.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgFEMCloth c0 = makeZeroDesc();
    c0.mTriMeshData = reinterpret_cast<void*>(size_t(bTree0.addr));
    c0.mPosition_InvMass = reinterpret_cast<decltype(c0.mPosition_InvMass)>(size_t(bPos0.addr));
    c0.mTriangleVertexIndices = reinterpret_cast<decltype(c0.mTriangleVertexIndices)>(size_t(bTri0.addr));
    c0.mTrianglesWithActiveEdges = reinterpret_cast<decltype(c0.mTrianglesWithActiveEdges)>(size_t(bTwae0.addr));
    c0.mNbTrianglesWithActiveEdges = uint32_t(twae0.size());
    c0.mElementIndex = 0u;

    PxgFEMCloth c1 = makeZeroDesc();
    c1.mTriMeshData = reinterpret_cast<void*>(size_t(bTree1.addr));
    c1.mPosition_InvMass = reinterpret_cast<decltype(c1.mPosition_InvMass)>(size_t(bPos1.addr));
    c1.mTriangleVertexIndices = reinterpret_cast<decltype(c1.mTriangleVertexIndices)>(size_t(bTri1.addr));
    c1.mTrianglesWithActiveEdges = reinterpret_cast<decltype(c1.mTrianglesWithActiveEdges)>(size_t(bTwae1.addr));
    c1.mNbTrianglesWithActiveEdges = uint32_t(twae1.size());
    c1.mElementIndex = 1u;

    std::vector<PxgFEMCloth> descArr = { c0, c1 };
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<PxgShapeHost> shapes = { makeClothShape(0u), makeClothShape(1u) };
    DevBuf bShapes = makeDev(dev, shapes);
    std::vector<float> cd = { CD0, CD1 };
    DevBuf bCd = makeDev(dev, cd);
    std::vector<uint32_t> upd = { 1u };
    DevBuf bUpd = makeDev(dev, upd);

    const uint32_t MAXC = 64u;
    std::vector<FemFemContactInfoHost> outInfo(MAXC);
    std::memset(outInfo.data(), 0xAB, outInfo.size() * sizeof(FemFemContactInfoHost));
    DevBuf bOut = makeDev(dev, outInfo);

    // ---- CPU reference: the 3x3 edge loop, query = cloth 0's triangle 0 ----
    auto v3f = [](const F4& f) { return V3{ f.x, f.y, f.z }; };
    const uint32_t nextE[3] = { 1u, 2u, 0u };
    const uint32_t vIndices0[3] = { tris0[0].x, tris0[0].y, tris0[0].z };
    const V3 t0pts[3] = { v3f(pos0[vIndices0[0]]), v3f(pos0[vIndices0[1]]), v3f(pos0[vIndices0[2]]) };
    const V3 t0min = p72ref::sub(p72ref::vmin(p72ref::vmin(t0pts[0], t0pts[1]), t0pts[2]),
                                 V3{ contactDist, contactDist, contactDist });
    const V3 t0max = p72ref::add(p72ref::vmax(p72ref::vmax(t0pts[0], t0pts[1]), t0pts[2]),
                                 V3{ contactDist, contactDist, contactDist });
    std::vector<FemFemContactInfoHost> ref;
    for (uint32_t p = 0; p < uint32_t(tris1.size()); ++p)
    {
        const uint32_t vIndices1[3] = { tris1[p].x, tris1[p].y, tris1[p].z };
        const V3 t1pts[3] = { v3f(pos1[vIndices1[0]]), v3f(pos1[vIndices1[1]]), v3f(pos1[vIndices1[2]]) };
        const V3 t1min = p72ref::vmin(p72ref::vmin(t1pts[0], t1pts[1]), t1pts[2]);
        const V3 t1max = p72ref::vmax(p72ref::vmax(t1pts[0], t1pts[1]), t1pts[2]);
        if (t1min.x > t0max.x || t0min.x > t1max.x || t1min.y > t0max.y || t0min.y > t1max.y ||
            t1min.z > t0max.z || t0min.z > t1max.z) continue;
        for (uint32_t e1 = 0; e1 < 3; ++e1)
        {
            const V3 e1x0 = t1pts[e1], e1x1 = t1pts[nextE[e1]];
            for (uint32_t e0 = 0; e0 < 3; ++e0)
            {
                // The connected-edge skip is gated on clothId0 == clothId1 and is DEAD here,
                // even though cloth 1's triangle 0 reuses cloth 0's vertex indices.
                const V3 e0x0 = t0pts[e0], e0x1 = t0pts[nextE[e0]];
                float s, t, dq;
                p72ref::closestPtLineLine(e0x0, e0x1, e1x0, e1x1, s, t, dq);
                if ((s < p72ref::BARY_THR) || (s > p72ref::ONE_MINUS_BARY_THR) ||
                    (t < p72ref::BARY_THR) || (t > p72ref::ONE_MINUS_BARY_THR)) continue;
                if (!(dq < contactDistSq && dq > 1.0e-14f)) continue;
                uint32_t aux = PAIR_TYPE_FLAG_MASK_H | VALID_H;
                aux = (aux & ~AUX_IND0_MASK_H) | (e0 & AUX_IND0_MASK_H);
                aux = (aux & ~AUX_IND1_MASK_H) | ((e1 & ((1u << AUX_IND1_BITS_H) - 1u)) << AUX_IND0_BITS_H);
                ref.push_back(FemFemContactInfoHost{ 0ull /*encode(cloth 0, triangle 0)*/,
                                                     (1u << 20) | p, aux });
            }
        }
    }
    PXG_CHECK_MSG(ref.size() == 2u, "reference: expected 2 EE crossings against cloth 1's triangle 1, got %zu",
                  ref.size());
    PXG_CHECK_MSG(ref[0].pairInd0 == 0ull && ref[0].pairInd1 == ((1u << 20) | 1u),
                  "reference: the crossings must be cloth0 triangle 0 vs cloth1 PRIMITIVE 1");

    // GRID MIRRORS THE ENGINE: ceil(maxClothTrianglesWithActiveEdges / 2) x numTests.
    const uint32_t maxTwae = uint32_t(twae0.size() > twae1.size() ? twae0.size() : twae1.size());
    const uint32_t gridEE = (maxTwae + 1u) / 2u;
    Kernel k("femClothDifferentClothBda.spv", "cloth_differentClothCollisionEELaunch", {},
             sizeof(DifferentClothPush));

    // Two dispatches with the pair order SWAPPED.  `isCloth0 = numTri0 <= numTri1` makes the
    // smaller cloth drive either way, so both must produce the identical contact set -- which
    // is what proves the branchless id select in its mask == 1 branch.
    for (uint32_t swapped = 0; swapped < 2u; ++swapped)
    {
        std::vector<CmInputHost> cms = swapped ? std::vector<CmInputHost>{ { 1u, 0u, 1u, 0u } }
                                               : std::vector<CmInputHost>{ { 0u, 1u, 0u, 1u } };
        DevBuf bCm = makeDev(dev, cms);
        std::vector<uint32_t> count = { 0u };
        DevBuf bCount = makeDev(dev, count);

        DifferentClothPush push = {};
        push.maxContacts = MAXC;
        push.toleranceLength = 1.0f;
        push.cmInputs = bCm.addr;
        push.contactDistance = bCd.addr;
        push.gpuShapes = bShapes.addr;
        push.femClothes = bDesc.addr;
        push.updateContactPairs = bUpd.addr;
        push.outContactInfo = bOut.addr;
        push.totalContactCount = bCount.addr;

        k.dispatch(gridEE, {}, push, /*gridY numTests*/ 1u, 1u);
        dev->flush();

        const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
        PXG_CHECK_MSG(gCount == uint32_t(ref.size()),
                      "different-cloth EE (%s pair order) emitted %u contacts (expected %zu)",
                      swapped ? "swapped" : "natural", gCount, ref.size());
        const auto got = downloadBuffer<FemFemContactInfoHost>(dev, bOut.h, gCount ? gCount : 1u);
        const size_t mm = compareSets(std::vector<FemFemContactInfoHost>(got.begin(), got.begin() + gCount),
                                      ref, swapped ? "EE(swap)" : "EE");
        PXG_CHECK_MSG(mm == 0, "cloth_differentClothCollisionEE (%s pair order) bit-exact vs CPU (%zu mismatches)",
                      swapped ? "swapped" : "natural", mm);

        dev->destroyBuffer(bCm.h); dev->destroyBuffer(bCount.h);
    }

    for (PxgBufferHandle h : { bPos0.h, bTri0.h, bTwae0.h, bTree0.h, bPos1.h, bTri1.h, bTwae1.h,
                               bTree1.h, bDesc.h, bShapes.h, bCd.h, bUpd.h, bOut.h })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P8.17 step 1 -- cloth_meshMidphaseGeneratePairsLaunch, the cloth-vs-COOKED-TRIANGLE-MESH
// midphase.  softBodyMeshMidphaseBda's sibling: same cooked blob, same top-4 extraction, same
// grid, with a cloth TRIANGLE query.  What this fixture has to pin is the four things that
// differ from the softbody path, because every one of them is a plausible copy-paste:
//
//   * the SAT is TRIANGLE-vs-TRIANGLE with NINE edge-cross axes on top of the two plane terms.
//     One pair exists purely to kill a missing cross block: two triangles whose planes each cut
//     the other (both plane terms come out at -0.80) but which are separated by 0.25 along an
//     edge-cross axis.  With the cross block it is rejected, without it accepted;
//   * the contact distance is NOT divided by the scale.  min|scale| is 0.5, so the softbody
//     reading would give a 0.40 band against the correct 0.20 -- and one cloth triangle sits at
//     0.28, inside the wrong band and outside the right one;
//   * the box test is a plain AABB overlap against a PRE-FATTENED query bound.  The leaf box is
//     tight in y (max 0.06), so dropping the fattening loses the 0.12 pair outright;
//   * there is no isValidTet gate and no surface hint -- every cloth triangle is a query.
//
// The expected pair set is computed by a CPU transcription of the leaf predicate rather than
// written down, so a mutant anywhere in satIntersect fails without the fixture having to
// anticipate which axis it broke.
// ===========================================================================
namespace p817mesh
{
struct ClothMeshMidPush
{
    float toleranceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, bounds, contactDistance, gpuShapes, femClothes;
    uint32_t stackSizeBytes, pad60;
    uint64_t stackPtr, midphasePairsNum;
};
static_assert(sizeof(ClothMeshMidPush) == 80, "cloth mesh midphase blob must be 80 bytes");
static_assert(offsetof(ClothMeshMidPush, femClothes) == 48, "femClothes @48");
static_assert(offsetof(ClothMeshMidPush, stackSizeBytes) == 56, "stackSizeBytes @56");
static_assert(offsetof(ClothMeshMidPush, stackPtr) == 64, "stackPtr @64");

using p72ref::V3;
using p72ref::sub;
using p72ref::dot;
static V3 crs(V3 a, V3 b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 scl(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }

// PxPlane(p0,p1,p2) — normalize(cross(p1-p0, p2-p0)), d = -p0.dot(n).
static void makePlane(V3 p0, V3 p1, V3 p2, V3& n, float& d)
{
    const V3 c = crs(sub(p1, p0), sub(p2, p0));
    const float m = dot(c, c);
    n = (m > 0.0f) ? scl(c, 1.0f / std::sqrt(m)) : V3{ 0, 0, 0 };
    d = -dot(p0, n);
}
// minProject (triangletriangle.cuh:43): the `+ d` lands ONCE, after the min.
static float minProjectTri(V3 n, float d, V3 v0, V3 v1, V3 v2)
{
    return std::fmin(dot(n, v0), std::fmin(dot(n, v1), dot(n, v2))) + d;
}
static float testSeparatingAxis(V3 ax, V3 a0, V3 a1, V3 a2, V3 b0, V3 b1, V3 b2)
{
    float min0, max0, min1, max1;
    min0 = max0 = dot(a0, ax);
    min1 = max1 = dot(b0, ax);
    const float pa1 = dot(a1, ax), pb1 = dot(b1, ax);
    min0 = std::fmin(pa1, min0); max0 = std::fmax(pa1, max0);
    min1 = std::fmin(pb1, min1); max1 = std::fmax(pb1, max1);
    const float pa2 = dot(a2, ax), pb2 = dot(b2, ax);
    min0 = std::fmin(pa2, min0); max0 = std::fmax(pa2, max0);
    min1 = std::fmin(pb2, min1); max1 = std::fmax(pb2, max1);
    return std::fmax(min1 - max0, min0 - max1);
}
// Gu::satIntersect<true>, early-outs included.  `crossRaised` reports whether the cross block
// pushed sep above what the two plane terms alone gave — the fixture asserts it fires.
static float satIntersect(V3 a0, V3 a1, V3 a2, V3 nA, float dA,
                          V3 b0, V3 b1, V3 b2, V3 nB, float dB, float tol, bool* crossRaised)
{
    float sep = minProjectTri(nA, dA, b0, b1, b2);
    if (sep > tol) return sep;
    sep = std::fmax(sep, minProjectTri(nB, dB, a0, a1, a2));
    if (sep > tol) return sep;
    const float planeSep = sep;
    const V3 ax0[3] = { sub(a1, a0), sub(a2, a0), sub(a2, a1) };
    const V3 ax1[3] = { sub(b1, b0), sub(b2, b0), sub(b2, b1) };
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            V3 sa = crs(ax0[i], ax1[j]);
            const float m = dot(sa, sa);
            if (m > 1e-5f)
            {
                sa = scl(sa, 1.0f / std::sqrt(m));
                sep = std::fmax(sep, testSeparatingAxis(sa, a0, a1, a2, b0, b1, b2));
                if (sep > tol) { if (crossRaised && sep > planeSep) *crossRaised = true; return sep; }
            }
        }
    if (crossRaised && sep > planeSep) *crossRaised = true;
    return sep;
}
} // namespace p817mesh

PXG_TEST(femClothSimMeshMidphase)
{
    using namespace p816dc;        // PxgShapeHost, CmInputHost
    using namespace p817mesh;
    using p817mesh::V3;
    PxgSlangComputeDevice* dev = testDevice();

    // The mesh's placement.  scaleRot stays identity so shape2Vertex is a plain componentwise
    // divide; the SCALE is non-uniform (min |scale| = 0.5, which is what the softbody path would
    // divide the contact distance by) and the rigid rotation is NINETY degrees about y:
    //     world = q.rotate(vertex * scale) + T
    // Ninety, not the more convenient hundred and eighty, because a 180-degree rotation IS ITS
    // OWN INVERSE -- with it, swapping the kernel's world->local `rotateInv` for `rotate` changes
    // nothing and that mutant survives.  Measured.
    const float SX = 2.0f, SY = 1.0f, SZ = 0.5f;
    const float TX = 10.0f, TY = 20.0f, TZ = 30.0f;
    const float QS = 0.70710678f;                       // sin(45) = cos(45)
    const float QX = 0.0f, QY = QS, QZ = 0.0f, QW = QS; // 90 degrees about y
    // PxQuat::rotate (PxQuat.h:310), the same formula the shader carries.
    auto quatRot = [&](float x, float y, float z) {
        const float vx = 2.0f * x, vy = 2.0f * y, vz = 2.0f * z;
        const float w2 = QW * QW - 0.5f;
        const float d2 = QX * vx + QY * vy + QZ * vz;
        return V3{ vx * w2 + (QY * vz - QZ * vy) * QW + QX * d2,
                   vy * w2 + (QZ * vx - QX * vz) * QW + QY * d2,
                   vz * w2 + (QX * vy - QY * vx) * QW + QZ * d2 };
    };
    auto toWorld = [&](float vx, float vy, float vz) {
        const V3 r = quatRot(vx * SX, vy * SY, vz * SZ);
        return F4{ r.x + TX, r.y + TY, r.z + TZ, 1.0f };
    };

    // ---- the cooked mesh, in VERTEX space --------------------------------------------------
    // ONE BV32 node with SIX LEAF CHILDREN, one per query slot.  The per-slot boxes are what
    // ISOLATE the slots, and they have to: `satIntersect`'s axis set is the two face normals plus
    // the nine edge crosses, and for two PARALLEL triangles every one of those nine crosses is
    // parallel to the common normal.  So the SAT reports the plane gap for a pair of horizontal
    // triangles no matter how far apart they are in x — measured, not assumed (the first version
    // of this fixture put every horizontal triangle in one leaf and slot 4 matched three of its
    // neighbours' triangles at their y-gap).  The BVH boxes are the only thing that separates
    // them, which is exactly their job in the engine too.
    //
    // prim 0        : a DUMMY parked 1000 away and in NO leaf, so every primStartIndex is nonzero
    //                 -- with a zero start `data >> 7` and `data >> 8` agree and a shift mutant is
    //                 invisible;
    // prims 1,2     : slot 0's triangle, and a BACK-FACING one that is CLOSER (0.06 against 0.12)
    //                 -- only the front-face gate keeps it out;
    // prims 3..8    : slot 1's triangle plus a stack of five, so that slot sees SIX qualifying
    //                 triangles and the MaxNumPairs = 4 cap actually binds;
    // prims 9,10,11 : slots 2, 3 and 4;
    // prim 12       : the CROSS-AXIS triangle, in a deliberately LOOSE box (see below).
    std::vector<F4> mv;
    std::vector<TriIdx4> mi;
    auto pushTri = [&](F4 p0, F4 p1, F4 p2) {
        const uint32_t b = uint32_t(mv.size());
        mv.push_back(p0); mv.push_back(p1); mv.push_back(p2);
        mi.push_back(TriIdx4{ b, b + 1u, b + 2u, 0u });
    };
    // A front-facing (n = +y) unit right triangle at x in [ox, ox+1], height y.
    auto pushSlab = [&](float ox, float y) {
        pushTri(F4{ ox, y, 0, 0 }, F4{ ox, y, 1.0f, 0 }, F4{ ox + 1.0f, y, 0, 0 });
    };
    pushTri(F4{ 1000.0f, 0, 0, 0 }, F4{ 1000.0f, 0, 1.0f, 0 }, F4{ 1001.0f, 0, 0, 0 });   // prim 0
    pushSlab(0.0f, 0.0f);                                                                 // prim 1 (slot 0)
    pushTri(F4{ 0, 0.06f, 0, 0 }, F4{ 1.0f, 0.06f, 0, 0 }, F4{ 0, 0.06f, 1.0f, 0 });       // prim 2, n = -y
    const uint32_t SLOT1_PRIM = uint32_t(mi.size());                                       // prim 3
    pushSlab(2.0f, 0.0f);
    for (uint32_t k = 0; k < 5u; ++k)                                                      // prims 4..8
        pushSlab(2.0f, -0.02f * float(k + 1u));
    const uint32_t SLOT2_PRIM = uint32_t(mi.size());  pushSlab(4.0f, 0.0f);                // prim 9
    const uint32_t SLOT3_PRIM = uint32_t(mi.size());  pushSlab(6.0f, 0.0f);                // prim 10
    const uint32_t SLOT4_PRIM = uint32_t(mi.size());  pushSlab(8.0f, 0.0f);                // prim 11
    // The CROSS-AXIS triangle.  Its edges are deliberately SHORT (0.5, not 2.0): the separating
    // cross product then has |axis| = 0.25, so leaving out the 1/sqrt(magSq) normalization scales
    // the reported separation down by 4x -- from 0.25, outside the band, to 0.0625, inside it.
    // With long edges the unnormalized axis is LONGER than unit and the verdict does not change,
    // which is how that mutant survived the first sweep.
    const uint32_t CROSS_PRIM = uint32_t(mi.size());                                       // prim 12
    pushTri(F4{ 12.0f, 0.30f, -0.25f, 0 }, F4{ 12.0f, 0.30f, 0.25f, 0 }, F4{ 12.0f, 0.35f, 0, 0 });
    // The TINY pair, parked at x = 30.  Every edge is 0.03 long, so EVERY edge cross has
    // magSq <= 8.1e-7 -- below satIntersect's 1e-5 floor.  All nine cross axes are therefore
    // skipped and the two PLANE terms alone decide, which is the only configuration in which
    // applying them to the wrong triangle is observable (it turns 0.28 into 0.0).
    const float TINY_E = 0.03f;
    const uint32_t TINY_PRIM = uint32_t(mi.size());                                        // prim 13
    pushTri(F4{ 30.0f, 0, 0, 0 }, F4{ 30.0f, 0, TINY_E, 0 }, F4{ 30.0f + TINY_E, 0, 0, 0 });
    // The DIAGONAL pair, parked at x = 20.75.  Its leaf box is offset from the query bound in
    // BOTH x and y by 0.15 -- so the fattened-bound OVERLAP test (this kernel's) accepts it while
    // the softbody traverser's unfattened-bound DISPLACEMENT test rejects it
    // (0.15^2 + 0.15^2 = 0.045 > 0.20^2 = 0.04).  Axis-aligned offsets cannot tell those two
    // forms apart; only a diagonal one can.
    const uint32_t DIAG_PRIM = uint32_t(mi.size());                                        // prim 14
    pushSlab(20.75f, -0.15f);

    const uint32_t NPRIM = uint32_t(mi.size());
    const uint32_t NVERT = uint32_t(mv.size());

    // ---- the BV32 node: ONE node, SIX leaf children ----------------------------------------
    // Every slot box is TIGHT IN Y (its own triangle's height), so dropping the query bound's
    // fattening loses those pairs outright.  Child 5 is deliberately LOOSE -- a BVH box only has
    // to CONTAIN its primitive, and a loose one decouples the box test from the cross-axis SAT
    // that child exists to exercise.
    struct LeafSpec { uint32_t start, count; float mn[3], mx[3]; };
    const LeafSpec leafSpec[8] = {
        { 1u,          2u, { 0.0f, 0.00f, 0.0f }, { 1.0f, 0.06f, 1.0f } },   // slot 0 + back-facing
        { SLOT1_PRIM,  6u, { 2.0f, -0.10f, 0.0f }, { 3.0f, 0.00f, 1.0f } },  // slot 1 + the stack
        { SLOT2_PRIM,  1u, { 4.0f, 0.00f, 0.0f }, { 5.0f, 0.00f, 1.0f } },
        { SLOT3_PRIM,  1u, { 6.0f, 0.00f, 0.0f }, { 7.0f, 0.00f, 1.0f } },
        { SLOT4_PRIM,  1u, { 8.0f, 0.00f, 0.0f }, { 9.0f, 0.00f, 1.0f } },
        { CROSS_PRIM,  1u, { 10.0f, -1.0f, -2.0f }, { 14.0f, 1.0f, 2.0f } }, // loose, on purpose
        { TINY_PRIM,   1u, { 29.5f, -0.5f, -0.5f }, { 30.5f, 0.5f, 0.5f } }, // loose, on purpose
        { DIAG_PRIM,   1u, { 20.75f, -0.15f, 0.0f }, { 21.75f, -0.15f, 1.0f } },
    };
    std::vector<uint32_t> blob(4u + 292u, 0u);   // header (4 words) + one 1168-byte node
    {
        const uint32_t n = 4u;   // the node starts at byte 16
        for (uint32_t c = 0; c < 8u; ++c)
        {
            for (uint32_t a = 0; a < 3u; ++a)
            {
                blob[n + c * 4u + a] = f2u(leafSpec[c].mn[a]);              // mMin[c]
                blob[n + 128u + c * 4u + a] = f2u(leafSpec[c].mx[a]);       // mMax[c] (byte +512)
            }
            // mData: isLeaf | nbPrimitives<<1 | primStartIndex<<7
            blob[n + 256u + c] = 1u | (leafSpec[c].count << 1) | (leafSpec[c].start << 7);
        }
        blob[n + 288] = 8u;   // mNbNodes @byte 1152 — eight children
        blob[n + 289] = 1u;   // mDepth
    }
    const size_t vBase = blob.size();
    blob.resize(vBase + NVERT * 4u, 0u);
    std::memcpy(blob.data() + vBase, mv.data(), NVERT * sizeof(F4));
    const size_t iBase = blob.size();
    blob.resize(iBase + NPRIM * 4u, 0u);
    std::memcpy(blob.data() + iBase, mi.data(), NPRIM * sizeof(TriIdx4));
    blob[0] = NVERT;
    blob[1] = NPRIM;
    // header.z is nbBv32TreeNodes and header.w is nbPackedNodes.  They are DIFFERENT counts and
    // only .w sizes the node array, so .z is a deliberately wrong 7 here: a port that sizes with
    // .z walks the vertex pointer seven nodes too far and reads garbage.
    blob[2] = 7u;
    blob[3] = 1u;

    DevBuf bBlob = makeDev(dev, blob);
    if (!bBlob.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // ---- the cloth: one query triangle per slot, plus the cross-axis one -------------------
    const uint32_t NB_SLOTS = 5u;
    const float gaps[5] = { 0.12f, 0.04f, 0.28f, 0.45f, 0.02f };
    std::vector<F4> cv;
    std::vector<TriIdx4> ci;
    std::vector<V3> clothVs;   // the same corners in VERTEX space, for the CPU reference
    auto pushCloth = [&](V3 p0, V3 p1, V3 p2) {
        const uint32_t b = uint32_t(cv.size());
        cv.push_back(toWorld(p0.x, p0.y, p0.z));
        cv.push_back(toWorld(p1.x, p1.y, p1.z));
        cv.push_back(toWorld(p2.x, p2.y, p2.z));
        ci.push_back(TriIdx4{ b, b + 1u, b + 2u, 0u });
        clothVs.push_back(p0); clothVs.push_back(p1); clothVs.push_back(p2);
    };
    for (uint32_t t = 0; t < NB_SLOTS; ++t)
    {
        const float ox = float(t) * 2.0f, g = gaps[t];
        pushCloth(V3{ ox + 0.2f, g, 0.2f }, V3{ ox + 0.2f, g, 0.6f }, V3{ ox + 0.6f, g, 0.2f });
    }
    // slot 5 — the cross-axis query: its plane cuts prim 12 and prim 12's plane cuts it (the two
    // plane terms come out at -0.05), and only an edge-cross axis separates them, by 0.25.
    pushCloth(V3{ 11.55f, 0.0f, 0.0f }, V3{ 12.05f, 0.0f, 0.0f }, V3{ 11.8f, 0.05f, 0.0f });
    // slot 6 — the tiny query, 0.28 above prim 13 with every edge cross below the 1e-5 floor.
    pushCloth(V3{ 30.0f, 0.28f, 0.0f }, V3{ 30.0f, 0.28f, TINY_E }, V3{ 30.0f + TINY_E, 0.28f, 0.0f });
    // slot 7 — the diagonal query, 0.15 above and 0.15 to the left of prim 14's leaf box.
    pushCloth(V3{ 20.2f, 0.0f, 0.2f }, V3{ 20.2f, 0.0f, 0.6f }, V3{ 20.6f, 0.0f, 0.2f });
    const uint32_t NB_CLOTH_TRI = uint32_t(ci.size());

    DevBuf bCv = makeDev(dev, cv), bCi = makeDev(dev, ci);

    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bCv.addr));
    c.mTriangleVertexIndices = reinterpret_cast<decltype(c.mTriangleVertexIndices)>(size_t(bCi.addr));
    c.mNbTriangles = NB_CLOTH_TRI;
    c.mNbVerts = uint32_t(cv.size());
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- the pair.  The TRIMESH is shapeRef0 and the CLOTH shapeRef1, i.e. the SWAPPED order.
    // BOTH shapes are eTRIANGLEMESH here, so the cloth can only be found by its body id.
    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_TRIANGLEMESH_H; shapes[0].particleOrSoftbodyId = 0xFFFFFFFFu;
    shapes[0].sx = SX; shapes[0].sy = SY; shapes[0].sz = SZ; shapes[0].srw = 1.0f;
    shapes[0].hullLo = uint32_t(bBlob.addr & 0xFFFFFFFFull);
    shapes[0].hullHi = uint32_t(bBlob.addr >> 32);
    shapes[1].type = GEOM_TRIANGLEMESH_H; shapes[1].particleOrSoftbodyId = 0u;
    shapes[1].sx = 1.0f; shapes[1].sy = 1.0f; shapes[1].sz = 1.0f; shapes[1].srw = 1.0f;
    DevBuf bShapes = makeDev(dev, shapes);
    std::vector<CmInputHost> cms = { { 0u, 1u, /*trimesh cache*/ 1u, /*cloth cache*/ 0u } };
    DevBuf bCms = makeDev(dev, cms);

    // PxsCachedTransform, 32 B: PxQuat q @0, PxVec3 p @16.  Slot 0 = the cloth (unused), slot 1
    // = the trimesh, rotated 180 degrees about y and translated.
    std::vector<float> xf(2 * 8, 0.0f);
    xf[3] = 1.0f;                                                              // slot 0 identity
    xf[8 + 0] = QX; xf[8 + 1] = QY; xf[8 + 2] = QZ; xf[8 + 3] = QW;             // 90 deg about y
    xf[8 + 4] = TX; xf[8 + 5] = TY; xf[8 + 6] = TZ;
    DevBuf bXf = makeDev(dev, xf);

    // contactDistance is summed over BOTH cache refs: 0.05 + 0.15 = 0.20, and NOT divided.
    const float CD_CLOTH = 0.05f, CD_MESH = 0.15f;
    const float contactDist = CD_CLOTH + CD_MESH;
    std::vector<float> cd = { CD_CLOTH, CD_MESH };
    DevBuf bCd = makeDev(dev, cd);

    const uint32_t CAP = 64u;
    std::vector<uint32_t> stack(CAP * 4u, 0xFFFFFFFFu);
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> pairNum = { 0u };
    DevBuf bPairNum = makeDev(dev, pairNum);

    ClothMeshMidPush push = {};
    push.toleranceLength = 1.0f;
    push.cmInputs = bCms.addr;
    push.transformCache = bXf.addr;
    push.bounds = 0ull;                 // ABI parity only — the kernel never reads it
    push.contactDistance = bCd.addr;
    push.gpuShapes = bShapes.addr;
    push.femClothes = bDesc.addr;
    push.stackSizeBytes = CAP * 16u;
    push.stackPtr = bStack.addr;
    push.midphasePairsNum = bPairNum.addr;

    Kernel k("femClothMeshMidphaseBda.spv", "cloth_meshMidphaseGeneratePairsLaunch",
             {}, sizeof(ClothMeshMidPush));
    // The engine's grid (PxgNarrowphaseCore.cpp:6597): x = SB_SBMIDPHASE = 1024, y = numTests.
    k.dispatch(/*gridX*/ 1024u, {}, push, /*gridY*/ 1u, 1u);
    dev->flush();

    // ---- CPU reference: the leaf predicate, per lane, exactly as the warp runs it -----------
    // Each leaf gives lane L the primitive primStart + L, so with <= 32 primitives per leaf every
    // lane sees at most one primitive per leaf and its tolerance is min(PX_MAX, contactDist).
    // The reference walks the SAME six leaves the tree carries.
    auto meshVert = [&](uint32_t i) { return V3{ mv[i].x, mv[i].y, mv[i].z }; };

    bool crossFired = false;
    std::vector<std::pair<uint32_t, uint32_t>> ref;   // (clothTri, prim)
    for (uint32_t t = 0; t < NB_CLOTH_TRI; ++t)
    {
        const V3 a0 = clothVs[t * 3 + 0], a1 = clothVs[t * 3 + 1], a2 = clothVs[t * 3 + 2];
        V3 nA; float dA; p817mesh::makePlane(a0, a1, a2, nA, dA);
        const V3 cen{ (a0.x + a1.x + a2.x) / 3.0f, (a0.y + a1.y + a2.y) / 3.0f, (a0.z + a1.z + a2.z) / 3.0f };
        const V3 tMin{ std::fmin(std::fmin(a0.x, a1.x), a2.x) - contactDist,
                       std::fmin(std::fmin(a0.y, a1.y), a2.y) - contactDist,
                       std::fmin(std::fmin(a0.z, a1.z), a2.z) - contactDist };
        const V3 tMax{ std::fmax(std::fmax(a0.x, a1.x), a2.x) + contactDist,
                       std::fmax(std::fmax(a0.y, a1.y), a2.y) + contactDist,
                       std::fmax(std::fmax(a0.z, a1.z), a2.z) + contactDist };

        // per-lane best, persisting across the leaves the warp visits
        std::vector<float> best(32, 3.4028235e38f);
        std::vector<uint32_t> bestPrim(32, 0xFFFFFFFFu);
        for (const LeafSpec& lf : leafSpec)
        {
            const bool overlap = !(lf.mn[0] > tMax.x || tMin.x > lf.mx[0] ||
                                   lf.mn[1] > tMax.y || tMin.y > lf.mx[1] ||
                                   lf.mn[2] > tMax.z || tMin.z > lf.mx[2]);
            if (!overlap) continue;
            for (uint32_t lane = 0; lane < lf.count && lane < 32u; ++lane)
            {
                const uint32_t p = lf.start + lane;
                const V3 b0 = meshVert(mi[p].x), b1 = meshVert(mi[p].y), b2 = meshVert(mi[p].z);
                V3 nB; float dB; p817mesh::makePlane(b0, b1, b2, nB, dB);
                if (!(p817mesh::dot(nB, cen) + dB >= 0.0f)) continue;   // the front-face gate
                bool raised = false;
                const float sep = p817mesh::satIntersect(a0, a1, a2, nA, dA, b0, b1, b2, nB, dB,
                                                         std::fmin(best[lane], contactDist), &raised);
                if (raised && sep >= contactDist) crossFired = true;
                if (sep < best[lane]) { best[lane] = sep; bestPrim[lane] = p; }
            }
        }
        // the top-4 extraction: lanes inside the band, four smallest, lowest lane wins ties
        std::vector<std::pair<float, uint32_t>> keep;   // (sep, lane)
        for (uint32_t lane = 0; lane < 32u; ++lane)
            if (best[lane] < contactDist) keep.push_back({ best[lane], lane });
        std::sort(keep.begin(), keep.end());
        if (keep.size() > 4u) keep.resize(4u);
        for (auto& e : keep) ref.push_back({ t, bestPrim[e.second] });
    }
    std::sort(ref.begin(), ref.end());

    PXG_CHECK_MSG(crossFired,
                  "the fixture must contain a pair the EDGE-CROSS axes reject and the two plane "
                  "terms alone would accept — without one, deleting satIntersect's cross block "
                  "is invisible");
    PXG_CHECK_MSG(ref.size() == 7u,
                  "reference: expected 7 pairs (slot 0 one, slot 1 four via the cap, slot 4 one, "
                  "slot 7 one, and none for slots 2, 3, the cross pair or the tiny pair), got %zu",
                  ref.size());

    const uint32_t gNum = downloadBuffer<uint32_t>(dev, bPairNum.h, 1)[0];
    const auto gStack = downloadBuffer<uint32_t>(dev, bStack.h, CAP * 4u);
    std::vector<std::pair<uint32_t, uint32_t>> got;
    for (uint32_t i = 0; i < gNum && i < CAP; ++i)
    {
        PXG_CHECK_MSG(gStack[i * 4u + 0u] == 0u,
                      "pair %u carries cmIdx %u; blockIdx.y is the ONLY contact-manager index here",
                      i, gStack[i * 4u + 0u]);
        got.push_back({ gStack[i * 4u + 1u], gStack[i * 4u + 2u] });
    }
    std::sort(got.begin(), got.end());
    for (auto& e : got) std::printf("  pair: clothTri %u  meshPrim %u\n", e.first, e.second);

    PXG_CHECK_MSG(gNum == uint32_t(ref.size()),
                  "emitted %u pairs, reference says %zu.  0 usually means the blob walk or the "
                  "vertex-space transform is wrong; more means the contact distance picked up the "
                  "softbody path's scale division (the 0.28, cross and tiny pairs all come inside "
                  "a 0.40 band); fewer means the query bound lost its fattening",
                  gNum, ref.size());
    PXG_CHECK_MSG(got == ref, "the emitted pair SET does not match the CPU reference");

    // Pin what the six ARE, so a reference that drifts with the shader is caught too.
    auto has = [&](uint32_t t, uint32_t p) {
        return std::find(got.begin(), got.end(), std::make_pair(t, p)) != got.end();
    };
    auto countFor = [&](uint32_t t) {
        uint32_t n = 0; for (auto& e : got) if (e.first == t) ++n; return n;
    };
    PXG_CHECK_MSG(has(0u, 1u), "slot 0 must match the triangle in ITS OWN leaf (prim 1)");
    PXG_CHECK_MSG(countFor(0u) == 1u,
                  "slot 0 must keep exactly ONE: the back-facing prim 2 shares its leaf and is "
                  "CLOSER (0.06 against 0.12), and only the front-face gate keeps it out (got %u)",
                  countFor(0u));
    PXG_CHECK_MSG(countFor(1u) == 4u,
                  "slot 1 sees six qualifying triangles in its leaf and the MaxNumPairs cap must "
                  "hold it to four (got %u)", countFor(1u));
    PXG_CHECK_MSG(has(1u, SLOT1_PRIM) && has(1u, SLOT1_PRIM + 1u) &&
                  has(1u, SLOT1_PRIM + 2u) && has(1u, SLOT1_PRIM + 3u),
                  "slot 1 must keep the FOUR CLOSEST — its own triangle plus the three nearest of "
                  "the stack, not the first four the leaf happens to hand out");
    PXG_CHECK_MSG(countFor(2u) == 0u,
                  "slot 2 sits 0.28 away: outside the correct 0.20 band and inside the 0.40 the "
                  "softbody path's scale division would give");
    PXG_CHECK_MSG(countFor(3u) == 0u, "slot 3 sits 0.45 away, outside either reading");
    PXG_CHECK_MSG(countFor(4u) == 1u && has(4u, SLOT4_PRIM),
                  "slot 4 sits 0.02 away and must match its own leaf's triangle");
    PXG_CHECK_MSG(countFor(5u) == 0u,
                  "the cross-axis pair must be REJECTED: the two plane terms give -0.05 and only "
                  "the nine edge-cross axes separate it, by 0.25 -- and only once the axis is "
                  "NORMALIZED (unnormalized it reports 0.0625, inside the 0.20 band)");
    PXG_CHECK_MSG(countFor(6u) == 0u,
                  "the tiny pair must be REJECTED at 0.28: every edge cross is below the 1e-5 "
                  "floor, so the two plane terms decide alone -- and they only give 0.28 when "
                  "each is applied to the OTHER triangle's vertices");
    PXG_CHECK_MSG(countFor(7u) == 1u && has(7u, DIAG_PRIM),
                  "the diagonal pair must be KEPT: its leaf box clears the query bound by 0.15 in "
                  "x AND y, which the fattened-bound overlap test accepts and the softbody "
                  "traverser's unfattened displacement test (0.045 > 0.04) would reject");

    for (PxgBufferHandle h : { bBlob.h, bCv.h, bCi.h, bDesc.h, bShapes.h, bCms.h, bXf.h, bCd.h,
                               bStack.h, bPairNum.h })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P8.17 step 2 -- cloth_meshContactGenLaunch, the cloth-vs-COOKED-TRIANGLE-MESH contact
// generation.
//
// SCOPE, STATED UP FRONT.  The clipping core (trimTriangleByTriangle ->
// getDeepestPointSimplified -> computeBarycentric) is a LIFT of fcStaticTriCloth.slang's mode 0,
// which testFcStaticTriCloth.cpp already checks against a CPU transcription -- the sign-change
// bookkeeping, the in-place insert branch, the +-tolerance averaging band and the degenerate
// paths.  Re-transcribing all of that here would duplicate that fixture, not extend it.  What
// this fixture covers is the part that is NEW: the BDA plumbing.
//
//   * the 192-byte blob, whose stackSizeBytes sits at @64 BEFORE the two stack pointers -- the
//     softbody sibling puts it at @80 AFTER them, and reading one pack with the other's layout
//     hands the kernel a byte count where a device pointer belongs;
//   * the pair-stack iteration, the ceil-to-a-warp `l` and the per-warp slot packing.  40 pairs
//     straddle a warp boundary, with a contact in EACH warp, so both reservations are exercised
//     and their slots must come out in warp order;
//   * cloth/mesh resolution by particleOrSoftbodyId with the pair in SWAPPED shape order;
//   * `meshTriangleIdx` really indexing the mesh (primitive 0 is a decoy 1000 units away);
//   * the cooked-blob walk sized by header `.w`, not `.z`;
//   * the transform chain -- vertex2Shape (which MULTIPLIES) then
//     clothTransform.transformInv(trimeshToWorld) -- with a non-uniform mesh scale and two
//     DIFFERENT non-trivial rigid transforms;
//   * the filter skip needing ALL THREE cloth vertices;
//   * restDistance indexed by cmIdx (not by a cache ref, and not by 0);
//   * the writer's field mapping, including that the u64 contactByX/tempContactByX keys are
//     left UNTOUCHED by this variant.
//
// The geometry is chosen so the clip outcome is ANALYTIC: each query triangle is built from the
// mesh triangle's own cloth-local image, offset along its normal by a chosen gap and sideways by
// a little.  The clip then leaves the triangle whole, all three corners sit at the same depth,
// and the deepest point is their centroid with barycentric (1/3, 1/3, 1/3).  Crucially the
// positions are DERIVED FROM the transform chain, so a shader whose chain differs computes a
// different plane and a different separation.
// ===========================================================================
namespace p817cg
{
struct ClothMeshCgPush
{
    float toleranceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, restDistance, gpuShapes, femClothes,
             shapeToRigidRemapTable;
    uint32_t stackSizeBytes, pad68;
    uint64_t stackPtr, midphasePairsNum, filterPairs;
    uint32_t nbFilterPairs, pad100;
    uint64_t stackSizeNeededOnDevice;
    // PxgFEMContactWriter, by value
    uint64_t wPoint, wNormalPen, wBary, wInfo, wCount, wByX, wTempByX, wIdxByX, wSorted32;
    uint32_t wMaxContacts, pad188;
};
static_assert(sizeof(ClothMeshCgPush) == 192, "cloth mesh CG blob must be 192 bytes");
static_assert(offsetof(ClothMeshCgPush, shapeToRigidRemapTable) == 56, "remap table @56");
static_assert(offsetof(ClothMeshCgPush, stackSizeBytes) == 64, "stackSizeBytes @64, BEFORE the stack pointers");
static_assert(offsetof(ClothMeshCgPush, stackPtr) == 72, "stackPtr @72");
static_assert(offsetof(ClothMeshCgPush, midphasePairsNum) == 80, "midphasePairsNum @80");
static_assert(offsetof(ClothMeshCgPush, nbFilterPairs) == 96, "nbFilterPairs @96");
static_assert(offsetof(ClothMeshCgPush, stackSizeNeededOnDevice) == 104, "stackSizeNeeded @104");
static_assert(offsetof(ClothMeshCgPush, wPoint) == 112, "the writer starts @112");
static_assert(offsetof(ClothMeshCgPush, wMaxContacts) == 184, "maxNumContacts @184");

// PxgRigidFilterPair (32 B): u64 index0, u64 index1, u32 index2, padding.
struct RigidFilterPairHost { uint64_t index0, index1; uint32_t index2, pad0, pad1, pad2; };
static_assert(sizeof(RigidFilterPairHost) == 32, "PxgRigidFilterPair mirror must be 32 bytes");

using p817mesh::V3;
using p817mesh::crs;
using p817mesh::scl;
using p72ref::sub;
using p72ref::add;
using p72ref::dot;

static V3 norm(V3 v) { const float m = dot(v, v); return m > 0.0f ? scl(v, 1.0f / std::sqrt(m)) : V3{ 0, 0, 0 }; }

// PxQuat::rotate / rotateInv, the same formula the shader carries.
static V3 qRot(const float q[4], V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q[3] * q[3] - 0.5f;
    const float d2 = q[0] * vx + q[1] * vy + q[2] * vz;
    return V3{ vx * w2 + (q[1] * vz - q[2] * vy) * q[3] + q[0] * d2,
               vy * w2 + (q[2] * vx - q[0] * vz) * q[3] + q[1] * d2,
               vz * w2 + (q[0] * vy - q[1] * vx) * q[3] + q[2] * d2 };
}
static V3 qRotInv(const float q[4], V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q[3] * q[3] - 0.5f;
    const float d2 = q[0] * vx + q[1] * vy + q[2] * vz;
    return V3{ vx * w2 - (q[1] * vz - q[2] * vy) * q[3] + q[0] * d2,
               vy * w2 - (q[2] * vx - q[0] * vz) * q[3] + q[1] * d2,
               vz * w2 - (q[0] * vy - q[1] * vx) * q[3] + q[2] * d2 };
}
// q = a * b (PxQuat::operator*)
static void qMul(const float a[4], const float b[4], float out[4])
{
    out[0] = a[3] * b[0] + b[3] * a[0] + a[1] * b[2] - a[2] * b[1];
    out[1] = a[3] * b[1] + b[3] * a[1] + a[2] * b[0] - a[0] * b[2];
    out[2] = a[3] * b[2] + b[3] * a[2] + a[0] * b[1] - a[1] * b[0];
    out[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2];
}
// vertex2Shape(v, scale, rot) = rot.rotateInv(rot.rotate(v) * scale) -- MULTIPLIES.
static V3 vertex2Shape(V3 v, V3 scale, const float rot[4])
{
    const V3 t = qRot(rot, v);
    return qRotInv(rot, V3{ t.x * scale.x, t.y * scale.y, t.z * scale.z });
}
} // namespace p817cg

PXG_TEST(femClothSimMeshContactGen)
{
    using namespace p816dc;        // PxgShapeHost, CmInputHost
    using namespace p817cg;
    using p817cg::V3;
    PxgSlangComputeDevice* dev = testDevice();

    // ---- the two rigid transforms and the mesh scale ---------------------------------------
    // Deliberately DIFFERENT and both non-trivial: a wrong transformInv composition, a dropped
    // vertex2Shape or a rotate/rotateInv swap all move the static plane in cloth space.
    const float QS = 0.70710678f;
    const float clothQ[4] = { 0.0f, 0.0f, QS, QS };   // 90 degrees about z
    const V3 clothP{ 7.0f, 8.0f, 9.0f };
    const float meshQ[4] = { 0.0f, QS, 0.0f, QS };    // 90 degrees about y
    const V3 meshP{ 1.0f, 2.0f, 3.0f };
    const V3 meshScale{ 2.0f, 1.0f, 0.5f };
    const float meshScaleRot[4] = { 0.0f, 0.0f, 0.0f, 1.0f };   // identity

    // ---- the cooked mesh: a decoy at index 0, the real triangle at index 1 -------------------
    // Primitive 0 is parked 1000 up, so a kernel that ignores meshTriangleIdx and always reads
    // primitive 0 finds nothing at all.
    std::vector<F4> mv = {
        { -5.0f, 1000.0f, -5.0f, 0 }, { -5.0f, 1000.0f, 5.0f, 0 }, { 5.0f, 1000.0f, -5.0f, 0 },
        { -5.0f,    0.0f, -5.0f, 0 }, { -5.0f,    0.0f, 5.0f, 0 }, { 5.0f,    0.0f, -5.0f, 0 },
    };
    std::vector<TriIdx4> mi = { { 0u, 1u, 2u, 0u }, { 3u, 4u, 5u, 0u } };
    const uint32_t REAL_PRIM = 1u;

    std::vector<uint32_t> blob(4u + 292u, 0u);     // header + one (unused) BV32 node
    const size_t vBase = blob.size();
    blob.resize(vBase + mv.size() * 4u, 0u);
    std::memcpy(blob.data() + vBase, mv.data(), mv.size() * sizeof(F4));
    const size_t iBase = blob.size();
    blob.resize(iBase + mi.size() * 4u, 0u);
    std::memcpy(blob.data() + iBase, mi.data(), mi.size() * sizeof(TriIdx4));
    blob[0] = uint32_t(mv.size());
    blob[1] = uint32_t(mi.size());
    blob[2] = 7u;      // header.z: a deliberately WRONG node count -- only .w sizes the array
    blob[3] = 1u;      // header.w: nbPackedNodes

    DevBuf bBlob = makeDev(dev, blob);
    if (!bBlob.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // ---- the mesh triangle's image in CLOTH-LOCAL space -------------------------------------
    // meshToCloth = clothT.transformInv(meshT):  q = conj(clothQ) * meshQ,
    //                                            p = conj(clothQ).rotate(meshP - clothP)
    float clothQConj[4] = { -clothQ[0], -clothQ[1], -clothQ[2], clothQ[3] };
    float m2cQ[4];
    qMul(clothQConj, meshQ, m2cQ);
    const V3 m2cP = qRotInv(clothQ, sub(meshP, clothP));
    auto toClothLocal = [&](F4 v) {
        return add(qRot(m2cQ, vertex2Shape(V3{ v.x, v.y, v.z }, meshScale, meshScaleRot)), m2cP);
    };
    const V3 A0 = toClothLocal(mv[3]), B0 = toClothLocal(mv[4]), C0 = toClothLocal(mv[5]);
    const V3 fixNormal = norm(crs(sub(B0, A0), sub(C0, A0)));
    // A point safely inside the triangle, and an in-plane basis to spread the corners over.
    const V3 centre{ (A0.x + B0.x + C0.x) / 3.0f, (A0.y + B0.y + C0.y) / 3.0f, (A0.z + B0.z + C0.z) / 3.0f };
    const V3 t1 = norm(sub(B0, A0));
    const V3 t2 = crs(fixNormal, t1);

    // ---- the cloth: one query triangle per pair, at a chosen gap along fixNormal -------------
    const float CD_CLOTH = 0.05f, CD_MESH = 0.15f;
    const float cDistance = CD_CLOTH + CD_MESH;      // 0.20
    const uint32_t NB_PAIRS = 40u;                   // straddles a warp boundary: l = 64
    // Pairs 0, 2 and 4 make contact in warp 0; pair 35 makes the only contact in warp 1.
    auto gapFor = [&](uint32_t i) -> float {
        if (i == 0u) return 0.10f;
        if (i == 2u) return -0.05f;     // penetrating: kept by the -cDistance slab
        if (i == 4u) return 0.10f;
        if (i == 3u) return 0.10f;      // would make contact, but all three verts are filtered
        if (i == 35u) return 0.02f;
        return 0.25f;                   // outside the +cDistance slab -> clipped to nothing
    };
    std::vector<F4> cv;
    std::vector<TriIdx4> ci;
    std::vector<V3> corner(NB_PAIRS * 3u);
    for (uint32_t i = 0; i < NB_PAIRS; ++i)
    {
        const float g = gapFor(i);
        const V3 base = add(centre, scl(fixNormal, g));
        const V3 c0 = add(base, scl(t1, 0.20f));
        const V3 c1 = add(base, scl(t2, 0.20f));
        const V3 c2 = add(base, scl(t1, -0.20f));
        corner[i * 3 + 0] = c0; corner[i * 3 + 1] = c1; corner[i * 3 + 2] = c2;
        const uint32_t b = uint32_t(cv.size());
        cv.push_back(F4{ c0.x, c0.y, c0.z, 1.0f });
        cv.push_back(F4{ c1.x, c1.y, c1.z, 1.0f });
        cv.push_back(F4{ c2.x, c2.y, c2.z, 1.0f });
        ci.push_back(TriIdx4{ b, b + 1u, b + 2u, 0u });
    }
    DevBuf bCv = makeDev(dev, cv), bCi = makeDev(dev, ci);

    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bCv.addr));
    c.mTriangleVertexIndices = reinterpret_cast<decltype(c.mTriangleVertexIndices)>(size_t(bCi.addr));
    c.mNbTriangles = NB_PAIRS;
    c.mNbVerts = uint32_t(cv.size());
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- shapes, contact managers, transforms ----------------------------------------------
    // SWAPPED order: shapeRef0 is the TRIMESH and shapeRef1 the cloth, so the kernel has to find
    // the cloth by particleOrSoftbodyId rather than by position.
    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_TRIANGLEMESH_H; shapes[0].particleOrSoftbodyId = 0xFFFFFFFFu;
    shapes[0].sx = meshScale.x; shapes[0].sy = meshScale.y; shapes[0].sz = meshScale.z;
    shapes[0].srw = 1.0f;
    shapes[0].materialIndex = 0x2Au;
    shapes[0].hullLo = uint32_t(bBlob.addr & 0xFFFFFFFFull);
    shapes[0].hullHi = uint32_t(bBlob.addr >> 32);
    shapes[1].type = GEOM_TRIANGLEMESH_H; shapes[1].particleOrSoftbodyId = 0u;
    shapes[1].sx = shapes[1].sy = shapes[1].sz = 1.0f; shapes[1].srw = 1.0f;
    DevBuf bShapes = makeDev(dev, shapes);

    // cmIdx 1 is the live pair; entry 0 exists only so restDistance[cmIdx] can be told from
    // restDistance[0].
    const uint32_t CM_IDX = 1u;
    std::vector<CmInputHost> cms = { { 9u, 9u, 9u, 9u }, { 0u, 1u, /*trimesh cache*/ 1u, /*cloth cache*/ 0u } };
    DevBuf bCms = makeDev(dev, cms);

    // PxsCachedTransform, 32 B each: slot 0 = the cloth, slot 1 = the trimesh.
    std::vector<float> xf(2 * 8, 0.0f);
    xf[0] = clothQ[0]; xf[1] = clothQ[1]; xf[2] = clothQ[2]; xf[3] = clothQ[3];
    xf[4] = clothP.x;  xf[5] = clothP.y;  xf[6] = clothP.z;
    xf[8 + 0] = meshQ[0]; xf[8 + 1] = meshQ[1]; xf[8 + 2] = meshQ[2]; xf[8 + 3] = meshQ[3];
    xf[8 + 4] = meshP.x;  xf[8 + 5] = meshP.y;  xf[8 + 6] = meshP.z;
    DevBuf bXf = makeDev(dev, xf);

    std::vector<float> cd = { CD_CLOTH, CD_MESH };
    DevBuf bCd = makeDev(dev, cd);
    const float REST_D = 0.33f;
    std::vector<float> rd = { 9.9f, REST_D };     // rd[CM_IDX] is the one that must be written
    DevBuf bRd = makeDev(dev, rd);

    // shapeToRigidRemapTable is indexed by the TRIMESH cache ref (1).
    const uint64_t RIGID_ID = 0x1234000056780000ull;
    std::vector<uint64_t> remap = { 0xDEADull, RIGID_ID };
    DevBuf bRemap = makeDev(dev, remap);

    // ---- the filter list: all three verts of pair 3, only two of pair 4 ---------------------
    std::vector<RigidFilterPairHost> filters;
    auto addFilter = [&](uint32_t vertIdx) {
        RigidFilterPairHost f = {};
        f.index0 = RIGID_ID;
        f.index1 = uint64_t(vertIdx);   // clothId 0 -> the mask IS the vertex index
        f.index2 = 0u;
        filters.push_back(f);
    };
    for (uint32_t v = 9u; v <= 11u; ++v) addFilter(v);    // pair 3's three vertices
    addFilter(12u); addFilter(13u);                        // TWO of pair 4's three
    std::sort(filters.begin(), filters.end(),
              [](const RigidFilterPairHost& a, const RigidFilterPairHost& b) {
                  return a.index0 != b.index0 ? a.index0 < b.index0 : a.index1 < b.index1;
              });
    DevBuf bFilters = makeDev(dev, filters);

    // ---- the pair stack ---------------------------------------------------------------------
    std::vector<uint32_t> stack(NB_PAIRS * 4u, 0u);
    for (uint32_t i = 0; i < NB_PAIRS; ++i)
    {
        stack[i * 4 + 0] = CM_IDX;
        stack[i * 4 + 1] = i;            // the cloth triangle
        stack[i * 4 + 2] = REAL_PRIM;    // the mesh triangle
        stack[i * 4 + 3] = 0u;
    }
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> pairNum = { NB_PAIRS };
    DevBuf bPairNum = makeDev(dev, pairNum);
    std::vector<uint32_t> needed = { 0u };
    DevBuf bNeeded = makeDev(dev, needed);

    // ---- the writer's output buffers, poisoned so an untouched lane is visible --------------
    const uint32_t MAXC = 32u;
    std::vector<F4> outPoint(MAXC), outNormalPen(MAXC), outBary(MAXC);
    std::memset(outPoint.data(), 0xAB, MAXC * sizeof(F4));
    std::memset(outNormalPen.data(), 0xAB, MAXC * sizeof(F4));
    std::memset(outBary.data(), 0xAB, MAXC * sizeof(F4));
    std::vector<uint32_t> outInfo(MAXC * 4u, 0xABABABABu);
    std::vector<uint64_t> outByX(MAXC, 0xCDCDCDCDCDCDCDCDull);   // must stay UNTOUCHED
    std::vector<uint32_t> outTempByX(MAXC, 0xCDCDCDCDu);         // must stay UNTOUCHED
    std::vector<uint32_t> outIdxByX(MAXC, 0xABABABABu);
    std::vector<uint32_t> outSorted32(MAXC, 0xABABABABu);
    std::vector<uint32_t> outCount = { 0u };
    DevBuf bPoint = makeDev(dev, outPoint), bNormalPen = makeDev(dev, outNormalPen),
           bBary = makeDev(dev, outBary), bInfo = makeDev(dev, outInfo),
           bByX = makeDev(dev, outByX), bTempByX = makeDev(dev, outTempByX),
           bIdxByX = makeDev(dev, outIdxByX), bSorted32 = makeDev(dev, outSorted32),
           bCount = makeDev(dev, outCount);

    ClothMeshCgPush push = {};
    push.toleranceLength = 1.0f;
    push.cmInputs = bCms.addr;
    push.transformCache = bXf.addr;
    push.contactDistance = bCd.addr;
    push.restDistance = bRd.addr;
    push.gpuShapes = bShapes.addr;
    push.femClothes = bDesc.addr;
    push.shapeToRigidRemapTable = bRemap.addr;
    push.stackSizeBytes = NB_PAIRS * 16u;
    push.stackPtr = bStack.addr;
    push.midphasePairsNum = bPairNum.addr;
    push.filterPairs = bFilters.addr;
    push.nbFilterPairs = uint32_t(filters.size());
    push.stackSizeNeededOnDevice = bNeeded.addr;
    push.wPoint = bPoint.addr; push.wNormalPen = bNormalPen.addr; push.wBary = bBary.addr;
    push.wInfo = bInfo.addr;   push.wCount = bCount.addr;
    push.wByX = bByX.addr;     push.wTempByX = bTempByX.addr;
    push.wIdxByX = bIdxByX.addr; push.wSorted32 = bSorted32.addr;
    push.wMaxContacts = MAXC;

    Kernel k("femClothMeshContactGenBda.spv", "cloth_meshContactGenLaunch", {}, sizeof(ClothMeshCgPush));
    // The engine's grid (PxgNarrowphaseCore.cpp:6694): 2048 blocks of 128 threads.  Only block 0
    // has pairs in range here, so the emitted slot order is deterministic.
    k.dispatch(/*gridX*/ 2048u, {}, push, 1u, 1u);
    dev->flush();

    // ---- the expected contacts, analytically -------------------------------------------------
    // The clip leaves each in-band triangle whole, so all three corners sit at the same depth,
    // getDeepestPointSimplified averages them and the barycentric is (1/3, 1/3, 1/3).
    struct Expect { uint32_t tri; float sep; V3 point; };
    std::vector<Expect> expect;
    for (uint32_t i = 0; i < NB_PAIRS; ++i)
    {
        if (i == 3u) continue;                       // filtered on all three vertices
        const float g = gapFor(i);
        if (!(g < cDistance) || g > cDistance) continue;
        if (g >= cDistance) continue;
        if (g > 0.20f) continue;
        const V3 c0 = corner[i * 3 + 0], c1 = corner[i * 3 + 1], c2 = corner[i * 3 + 2];
        // sep = planeDistance(plane, c0) -- the FIRST corner sets minDist and the other two only
        // accumulate into the average.
        const float sep = dot(fixNormal, c0) - dot(fixNormal, A0);
        const V3 deepest{ (c0.x + c1.x + c2.x) / 3.0f, (c0.y + c1.y + c2.y) / 3.0f,
                          (c0.z + c1.z + c2.z) / 3.0f };
        expect.push_back({ i, sep, add(qRot(clothQ, deepest), clothP) });
    }
    PXG_CHECK_MSG(expect.size() == 4u,
                  "reference: expected 4 contacts (pairs 0, 2, 4 in warp 0 and pair 35 in warp 1), "
                  "got %zu", expect.size());

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
    PXG_CHECK_MSG(gCount == uint32_t(expect.size()),
                  "emitted %u contacts, expected %zu.  0 usually means the blob walk or the "
                  "transform chain is wrong; 5 means the all-three-vertices filter skip became "
                  "an any-vertex one; 3 means the second warp's reservation was lost",
                  gCount, expect.size());

    const auto gPoint = downloadBuffer<F4>(dev, bPoint.h, MAXC);
    const auto gNormalPen = downloadBuffer<F4>(dev, bNormalPen.h, MAXC);
    const auto gBary = downloadBuffer<F4>(dev, bBary.h, MAXC);
    const auto gInfo = downloadBuffer<uint32_t>(dev, bInfo.h, MAXC * 4u);
    const auto gIdx = downloadBuffer<uint32_t>(dev, bIdxByX.h, MAXC);
    const auto gSorted = downloadBuffer<uint32_t>(dev, bSorted32.h, MAXC);

    const V3 wn = qRot(clothQ, fixNormal);
    size_t mm = 0;
    for (uint32_t s = 0; s < gCount && s < expect.size(); ++s)
    {
        const Expect& e = expect[s];
        std::printf("  slot %u: tri %u sep %.6f point (%.4f %.4f %.4f)\n", s, e.tri,
                    gNormalPen[s].w, gPoint[s].x, gPoint[s].y, gPoint[s].z);
        auto near1 = [&](float a, float b) { return std::fabs(a - b) < 2.0e-4f; };
        if (!near1(gPoint[s].x, e.point.x) || !near1(gPoint[s].y, e.point.y) ||
            !near1(gPoint[s].z, e.point.z))
        { ++mm; std::printf("  slot %u point MM: got (%.6f %.6f %.6f) ref (%.6f %.6f %.6f)\n", s,
                            gPoint[s].x, gPoint[s].y, gPoint[s].z, e.point.x, e.point.y, e.point.z); }
        if (!near1(gPoint[s].w, REST_D))
        { ++mm; std::printf("  slot %u point.w MM: got %.6f ref %.6f (restDistance[cmIdx])\n",
                            s, gPoint[s].w, REST_D); }
        if (!near1(gNormalPen[s].x, wn.x) || !near1(gNormalPen[s].y, wn.y) || !near1(gNormalPen[s].z, wn.z))
        { ++mm; std::printf("  slot %u normal MM: got (%.6f %.6f %.6f) ref (%.6f %.6f %.6f)\n", s,
                            gNormalPen[s].x, gNormalPen[s].y, gNormalPen[s].z, wn.x, wn.y, wn.z); }
        if (!near1(gNormalPen[s].w, e.sep))
        { ++mm; std::printf("  slot %u separation MM: got %.6f ref %.6f\n", s, gNormalPen[s].w, e.sep); }
        const float third = 1.0f / 3.0f;
        if (!near1(gBary[s].x, third) || !near1(gBary[s].y, third) || !near1(gBary[s].z, third))
        { ++mm; std::printf("  slot %u barycentric MM: got (%.6f %.6f %.6f), expected thirds\n",
                            s, gBary[s].x, gBary[s].y, gBary[s].z); }
        // PxgFemOtherContactInfo: pairInd0 = the rigid node index (u64), pairInd1 = the encoded
        // cloth TRIANGLE, then the rigid material with bit 31 cleared.
        const uint64_t gotPair0 = uint64_t(gInfo[s * 4 + 0]) | (uint64_t(gInfo[s * 4 + 1]) << 32);
        if (gotPair0 != RIGID_ID)
        { ++mm; std::printf("  slot %u pairInd0 MM: got 0x%llx ref 0x%llx\n", s,
                            (unsigned long long)gotPair0, (unsigned long long)RIGID_ID); }
        if (gInfo[s * 4 + 2] != e.tri)
        { ++mm; std::printf("  slot %u pairInd1 MM: got %u ref %u (encoded cloth TRIANGLE)\n",
                            s, gInfo[s * 4 + 2], e.tri); }
        if (gInfo[s * 4 + 3] != 0x2Au)
        { ++mm; std::printf("  slot %u material MM: got 0x%x ref 0x2A\n", s, gInfo[s * 4 + 3]); }
        if (gSorted[s] != uint32_t(RIGID_ID & 0xFFFFFFFFull))
        { ++mm; std::printf("  slot %u contactSortedByX MM: got 0x%x ref 0x%x (the LOW word)\n",
                            s, gSorted[s], uint32_t(RIGID_ID & 0xFFFFFFFFull)); }
        if (gIdx[s] != s)
        { ++mm; std::printf("  slot %u contactIndexSortedByX MM: got %u ref %u\n", s, gIdx[s], s); }
    }
    PXG_CHECK_MSG(mm == 0, "cloth_meshContactGen contact fields (%zu mismatches)", mm);

    // The u64 sort keys belong to the OTHER writer variant and must be left alone.
    const auto gByX = downloadBuffer<uint64_t>(dev, bByX.h, MAXC);
    const auto gTempByX = downloadBuffer<uint32_t>(dev, bTempByX.h, MAXC);
    size_t touched = 0;
    for (uint32_t s = 0; s < MAXC; ++s)
    {
        if (gByX[s] != 0xCDCDCDCDCDCDCDCDull) ++touched;
        if (gTempByX[s] != 0xCDCDCDCDu) ++touched;
    }
    PXG_CHECK_MSG(touched == 0,
                  "writeRigidVsDeformableContact32 must NOT write contactByX / tempContactByX -- "
                  "%zu of those words changed", touched);

    // The slot BEYOND the last contact must still be poison: the reservation is per warp, so a
    // lost or double reservation shows up as a gap or an overwrite here.
    PXG_CHECK_MSG(gIdx[expect.size()] == 0xABABABABu,
                  "slot %zu was written; the two warps must occupy exactly %zu consecutive slots",
                  expect.size(), expect.size());

    // stackSizeNeededOnDevice is an atomicMax of the BYTE count.
    const uint32_t gNeeded = downloadBuffer<uint32_t>(dev, bNeeded.h, 1)[0];
    PXG_CHECK_MSG(gNeeded == NB_PAIRS * 16u,
                  "stackSizeNeededOnDevice must be the pair count in BYTES (got %u, expected %u)",
                  gNeeded, NB_PAIRS * 16u);

    for (PxgBufferHandle h : { bBlob.h, bCv.h, bCi.h, bDesc.h, bShapes.h, bCms.h, bXf.h, bCd.h,
                               bRd.h, bRemap.h, bFilters.h, bStack.h, bPairNum.h, bNeeded.h,
                               bPoint.h, bNormalPen.h, bBary.h, bInfo.h, bByX.h, bTempByX.h,
                               bIdxByX.h, bSorted32.h, bCount.h })
        dev->destroyBuffer(h);
}

} // namespace
