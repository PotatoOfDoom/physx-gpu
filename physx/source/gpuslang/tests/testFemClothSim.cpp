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
#include "PxgParticleSystem.h"               // real engine per-particle-system descriptor (P9.30)
#include "PxsDeformableSurfaceMaterialCore.h" // real engine surface material data
#include "PxNodeIndex.h"                     // PxNodeIndex (rigid/world anchor node index)
#include "PxgSimulationCoreDesc.h"           // PxgFEMRigidAttachment (P7.1 tranche 3a) / PxgRigidFilterPair (4a)
#include "PxgSolverBody.h"                   // PxgSolverBodyData / PxgSolverTxIData / PxAlignedTransform
#include "PxgContactManager.h"               // PxgContactManagerInput (P7.1 tranche 4a contact gen)
#include "PxgConvexConvexShape.h"            // PxgShape (P7.1 tranche 4a contact gen)
#include "PxsTransformCache.h"               // PxsCachedTransform (P7.1 tranche 4a contact gen)
#include "geometry/PxGeometry.h"             // PxGeometryType (P7.1 tranche 4a contact gen)
#include "geometry/PxConvexCoreGeometry.h"  // PxConvexCore::Type (P9.75: eBOX read from the enum,
                                            // never a remembered 2 -- the P9.57 rule)
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

// P9.64 -- the FEM constraint-prepare and cloth-cloth entries are launched by the
// engine with a FIXED grid of 1024 (PxgKernelGridDim), and their shaders partition
// the work by that constant: each block takes
// `nbIterationsPerBlock * groupId.x * RC_BLOCK` as its offset, with the grid baked
// into nbIterationsPerBlock.  A fixture dispatching ONE group therefore reproduces
// neither the engine's launch nor the partitioning, and -- with 1023 groups absent
// -- nothing checks that those groups write nothing.
//
// Honest limit: at these fixtures' single contact the loop still runs one iteration
// whatever the grid, so the engine's grid alone does not make the partitioning
// CONSTANT observable; that needs more contacts than one block's share.  The grid
// is the floor of the property, not the whole of it.  tools/check_fixture_grid.py
// checks the floor, which is what it can see.
static const uint32_t FEM_PREP_GRID = 1024;

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

// P9.86c: mutant twin of membraneSolve's +0-seeded dx accumulators (FEMClothUtil.cuh:464,
// PxVec2 dx0(0.0f) ... then dx += massInv*deltaLambda*grad at :364-366 / :416-418).
//   0 = the CUDA: dx = +0 + term, so a -0 term lane lands +0.
//   1 = the port with its post-ARAP select (pxgSlangClothMembrane.slang, P9.86) reverted:
//       llvmpipe folds the forwarded literal seed, dx = term; the post-area select is kept.
//   2 = the P9.86 audit's hypothesised AREA-side fold without the post-area select: the
//       ARAP branch left dx at the seed and the area add folds against it, dx = term.
//       Meaningful only while the ARAP branch is not taken (the fixture asserts that).
// femClothSimSolveNonSharedZeroSign* / *RestZeroSign* require their twin to change the
// stored image.
static int gMembraneSeedVariant = 0;
static unsigned gMembraneNegZeroTermLanes = 0;       // coverage: -0 lanes among the ARAP terms
static unsigned gMembraneArapTaken = 0;              // coverage: ARAP C > threshold branch entries
static unsigned gMembraneNegZeroAreaTermLanes = 0;   // coverage: -0 lanes among the area terms
static V2 plusZero2(V2 v) { return V2{ v.x == 0.f ? 0.f : v.x, v.y == 0.f ? 0.f : v.y }; }
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

        ++gMembraneArapTaken;
        const V2 t0 = muls2(grad0, m0 * deltaLambda);
        const V2 t1 = muls2(grad1, m1 * deltaLambda);
        const V2 t2 = muls2(grad2, m2 * deltaLambda);
        const float lanes[6] = { t0.x, t0.y, t1.x, t1.y, t2.x, t2.y };
        for (float l : lanes)
            if (f2u(l) == 0x80000000u)
                ++gMembraneNegZeroTermLanes;
        // refMembrane calls this on the +0 seed, so variant 1 is exactly the folded `0 + term`.
        if (gMembraneSeedVariant == 1) { dx0 = t0; dx1 = t1; dx2 = t2; }
        else { dx0 = add2(dx0, t0); dx1 = add2(dx1, t1); dx2 = add2(dx2, t2); }
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

    const V2 t0 = muls2(grad0, m0 * deltaLambda);
    const V2 t1 = muls2(grad1, m1 * deltaLambda);
    const V2 t2 = muls2(grad2, m2 * deltaLambda);
    const float lanes[6] = { t0.x, t0.y, t1.x, t1.y, t2.x, t2.y };
    for (float l : lanes)
        if (f2u(l) == 0x80000000u)
            ++gMembraneNegZeroAreaTermLanes;
    // Variant 2: dx is still the literal seed (ARAP branch not taken), the add folds.
    if (gMembraneSeedVariant == 2) { dx0 = t0; dx1 = t1; dx2 = t2; }
    else { dx0 = add2(dx0, t0); dx1 = add2(dx1, t1); dx2 = add2(dx2, t2); }
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
        if (gMembraneSeedVariant == 1) { dx0 = plusZero2(dx0); dx1 = plusZero2(dx1); dx2 = plusZero2(dx2); }
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
        // P9.36: FEMCloth.cu:1258 is `PxVec3 deltaVel = tDelta / dt;` -- PxVec3's
        // operator/ (PxVec3.h:219), one reciprocal and three multiplies.
        const float invDt = 1.0f / dt;
        const float dvx = tdx * invDt, dvy = tdy * invDt, dvz = tdz * invDt;
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
//
// P9.86c zero-sign cases -- the +0-seeded dx accumulators of pxgSlangClothMembrane
// (CUDA FEMClothUtil.cuh:464 PxVec2 dx0(0.0f), then dx += massInv*deltaLambda*grad).
//
// ZS_STRETCHED: the same triangle lying in the z = -0 plane, with poissons = 0.
// x01.z = x02.z = -0 - -0 = +0, so the whole ARAP solve is bit-identical to the plain
// case, and its terms (massInv*deltaLambda)*grad carry -0 in t1.y, t2.x and t2.y
// (deltaLambda < 0 against grad1.y = grad2 = +0).  The CUDA adds them to the +0 seed
// (:364-366), so dx1.y = dx2 = +0 and x1.z = x2.z = -0 + +0 = +0.  Without
// pxgSlangClothMembrane's post-ARAP select the seed add folds (dx = term), the -0 lanes
// reach x_i += dx.x*axis0 + dx.y*axis1 (:506-508) and x1.z = x2.z stay -0.  poissons = 0
// skips the area step (:488), whose own select would otherwise re-canonicalise.
//
// ZS_REST: an obtuse triangle AT REST in the z = -0 plane (x2 = (-0.5, 1, -0), QInv from
// that rest pose), poissons 0.3.  F = I exactly, so the ARAP branch is NOT taken and dx
// reaches the area step still holding the seed.  The area C is +0 with lambda 0, so
// deltaLambda = -0 and t1 = -0*(1, 0.5) = (-0, -0): the CUDA's +0 + t1 keeps dx1 = +0
// and x1.z = -0 + +0 = +0.  This is the input on which the P9.86 audit's hypothesised
// AREA-side fold (the L156 add seeing the literal seed) would leave x1.z at -0; it keeps
// the post-area select's revert honest (its survival is the EQUIVALENT verdict, observed).
// ---------------------------------------------------------------------------
enum { ZS_NONE = 0, ZS_STRETCHED = 1, ZS_REST = 2 };
static void runSolveNonShared(bool isTGS, int zeroSign = ZS_NONE)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 3;                 // one triangle
    const float dt = 1.0f / 60.0f;
    const float invDt = 1.0f / dt;

    struct F2 { float x, y; };
    struct U4 { uint32_t x, y, z, w; };

    // Stretched current positions (rest edges are unit; x01 pulled to 1.5).
    const float zPlane = zeroSign != ZS_NONE ? -0.0f : 0.0f;   // -0: the flat cloth lies in the z = -0 plane
    std::vector<F4> positions = {
        { 0.0f, 0.0f, zPlane, 1.0f },
        { 1.5f, 0.0f, zPlane, 1.0f },
        { 0.0f, 1.0f, zPlane, 1.0f },
    };
    if (zeroSign == ZS_REST)
    {
        positions[1].x = 1.0f;     // x01 = (1, 0, +0): unstretched
        positions[2].x = -0.5f;    // x02 = (-0.5, 1, +0): obtuse at vertex 0 -> grad1.y = 0.5 > 0
    }
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
    // ZS_REST: Q = [u01 u02] = [[1,-0.5],[0,1]] -> QInv = [[1,0.5],[0,1]], stored (Qi00, Qi10,
    // Qi01, Qi11) as queryDeformationGradient_F2x2 reads it: f0 = x01, f1 = 0.5*x01 + x02 = (+0, 1).
    if (zeroSign == ZS_REST)
        qinv[0] = { 1.0f, 0.0f, 0.5f, 1.0f };
    std::vector<F2> lambdas = { { 0.0f, 0.0f } };
    std::vector<U4> triVtx = { { 0u, 1u, 2u, 0u } };   // verts (0,1,2), triIndex 0
    std::vector<uint32_t> matIdx = { 0u };              // mMaterialIndices[0] = 0
    std::vector<uint32_t> partCP = { 1u };              // partition 0 -> [0,1)

    PxsDeformableSurfaceMaterialData mat;
    std::memset(&mat, 0, sizeof(mat));
    mat.youngs = 1.0e5f;
    mat.poissons = zeroSign == ZS_STRETCHED ? 0.0f : 0.3f;   // 0 skips the area step (FEMClothUtil.cuh:488)
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
    const float slotX0 = slotX, slotY0 = slotY;

    p71ref::gMembraneNegZeroTermLanes = 0;
    p71ref::gMembraneArapTaken = 0;
    p71ref::gMembraneNegZeroAreaTermLanes = 0;
    p71ref::refMembrane(rx0, rx1, rx2, dt, mat.youngs, mat.poissons, mat.thickness, rQ,
                        1.0f, 1.0f, 1.0f, slotX, slotY, isTGS);

    if (zeroSign == ZS_STRETCHED)
    {
        // Coverage: the ARAP terms must carry exactly the three -0 lanes t1.y, t2.x, t2.y.
        // (Checked before the twin below, which re-enters arapConstraint2D.)
        PXG_CHECK_MSG(p71ref::gMembraneNegZeroTermLanes == 3u,
                      "zeroSign: %u -0 ARAP term lanes (want 3: t1.y, t2.x, t2.y), isTGS=%d",
                      p71ref::gMembraneNegZeroTermLanes, int(isTGS));
        // The CUDA (+0 seed add) lands both z coordinates on +0.
        PXG_CHECK_MSG(f2u(rx1.z) == 0u && f2u(rx2.z) == 0u,
                      "zeroSign: CUDA x1.z/x2.z %08x %08x, want +0, isTGS=%d", f2u(rx1.z), f2u(rx2.z), int(isTGS));
        // The folded-seed twin (the port without its post-ARAP select) must keep them at -0,
        // or this case no longer observes the fold.
        p71ref::F4 tw0 = { positions[0].x, positions[0].y, positions[0].z, positions[0].w };
        p71ref::F4 tw1 = { positions[1].x, positions[1].y, positions[1].z, positions[1].w };
        p71ref::F4 tw2 = { positions[2].x, positions[2].y, positions[2].z, positions[2].w };
        float tx = slotX0, ty = slotY0;
        p71ref::gMembraneSeedVariant = 1;
        p71ref::refMembrane(tw0, tw1, tw2, dt, mat.youngs, mat.poissons, mat.thickness, rQ,
                            1.0f, 1.0f, 1.0f, tx, ty, isTGS);
        p71ref::gMembraneSeedVariant = 0;
        PXG_CHECK_MSG(f2u(tw1.z) == 0x80000000u && f2u(tw2.z) == 0x80000000u,
                      "zeroSign twin: x1.z/x2.z %08x %08x, want -0 -- the case no longer observes the "
                      "folded ARAP seed, isTGS=%d", f2u(tw1.z), f2u(tw2.z), int(isTGS));
    }
    else if (zeroSign == ZS_REST)
    {
        // Coverage (before the twin re-enters the mirror): the ARAP branch must stay untaken so
        // dx reaches the area step as the seed, and the area terms must carry -0 in t1.x, t1.y
        // (deltaLambda = -0 against grad1 = (1, 0.5)) and t2.y (against grad2.y = 1).
        PXG_CHECK_MSG(p71ref::gMembraneArapTaken == 0u,
                      "restZeroSign: the ARAP branch ran %u time(s) -- the rest pose is no longer exact, isTGS=%d",
                      p71ref::gMembraneArapTaken, int(isTGS));
        PXG_CHECK_MSG(p71ref::gMembraneNegZeroAreaTermLanes == 3u,
                      "restZeroSign: %u -0 area term lanes (want 3: t1.x, t1.y, t2.y), isTGS=%d",
                      p71ref::gMembraneNegZeroAreaTermLanes, int(isTGS));
        PXG_CHECK_MSG(f2u(rx1.z) == 0u, "restZeroSign: CUDA x1.z %08x, want +0, isTGS=%d", f2u(rx1.z), int(isTGS));
        // The audit's hypothesised area-side fold (no post-area select) must leave x1.z at -0.
        p71ref::F4 tw0 = { positions[0].x, positions[0].y, positions[0].z, positions[0].w };
        p71ref::F4 tw1 = { positions[1].x, positions[1].y, positions[1].z, positions[1].w };
        p71ref::F4 tw2 = { positions[2].x, positions[2].y, positions[2].z, positions[2].w };
        float tx = slotX0, ty = slotY0;
        p71ref::gMembraneSeedVariant = 2;
        p71ref::refMembrane(tw0, tw1, tw2, dt, mat.youngs, mat.poissons, mat.thickness, rQ,
                            1.0f, 1.0f, 1.0f, tx, ty, isTGS);
        p71ref::gMembraneSeedVariant = 0;
        PXG_CHECK_MSG(f2u(tw1.z) == 0x80000000u,
                      "restZeroSign twin: x1.z %08x, want -0 -- the case no longer observes an area-side "
                      "seed fold, isTGS=%d", f2u(tw1.z), int(isTGS));
    }

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

    // The solve must actually move the stretched vertex (teeth: not a no-op).  ZS_REST is at
    // rest by construction; its teeth are the coverage checks above.
    if (zeroSign != ZS_REST)
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
// P9.86c: pxgSlangClothMembrane's +0-seeded ARAP dx accumulators -- flat cloth at z = -0.
PXG_TEST(femClothSimSolveNonSharedZeroSignPGS) { runSolveNonShared(/*isTGS*/ false, ZS_STRETCHED); }
PXG_TEST(femClothSimSolveNonSharedZeroSignTGS) { runSolveNonShared(/*isTGS*/ true, ZS_STRETCHED); }
PXG_TEST(femClothSimSolveNonSharedRestZeroSignPGS) { runSolveNonShared(/*isTGS*/ false, ZS_REST); }
PXG_TEST(femClothSimSolveNonSharedRestZeroSignTGS) { runSolveNonShared(/*isTGS*/ true, ZS_REST); }

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

// ---------------------------------------------------------------------------
// P9.15b -- cloth_midphaseGeneratePairsLaunch.
//
// THE ORACLE, AND WHY IT IS NOT A SECOND TRANSCRIPTION.  The traversal walks a
// BV32 tree, and modelling that walk on the host would be exactly the duplication
// this port avoids.  It is not needed: the tree is an INPUT, authored here, so the
// expected pair set follows from the tree's definition applied ONCE --
//
//     expected = { i : i lies in a REACHED leaf and triBound(i) meets the box }
//
// where "reached" is a property of the node boxes this fixture wrote, evaluated
// directly rather than by simulating a stack.
//
// THE LYING NODE, and why the fixture needs one.  In a real BV32 tree every node
// box contains its primitives, which makes child culling a pure optimisation:
// skipping a child can never change the emitted set, because the per-primitive
// test would have rejected those primitives anyway.  A fixture built only from
// honest nodes therefore cannot kill a single child-culling mutant.  So one child
// here is deliberately NON-CONSERVATIVE: its box misses the rigid box while its
// triangles overlap it.  The kernel must traverse the tree AS GIVEN and emit
// nothing for it; a port that drops the child test emits three pairs.
//
// WHAT THE TREE IS BUILT TO DISCRIMINATE
//   * TWO LEVELS: the root has an INTERNAL child pointing at node 1, so the
//     `(data & 1) == 0` branch and the `>> 11 << 6` child-offset decode both run;
//   * children that hit and children that miss, INTERLEAVED, so the ballot and the
//     exclusive prefix over the child mask carry a non-trivial pattern;
//   * leaves with DIFFERENT primitive counts and non-zero, non-contiguous
//     primStart values, so `(data >> 1) & 63` and `data >> 7` cannot be confused;
//   * a leaf whose primitives STRADDLE the box, so the per-primitive test is not
//     all-or-nothing within one tile;
//   * a stack bound that CUTS the emitted pairs, so M7's contract -- the counter
//     overshoots, the write is dropped -- is measured in the same run.
// ---------------------------------------------------------------------------
struct TriMidPush
{
    uint32_t numTests; float toleranceLength;
    uint64_t cmInputs, transformCache, bounds, contactDistance, gpuShapes, femClothes;
    uint64_t stackPtr, midphasePairsNum;
    uint32_t stackSizeBytes, pad76;
};
static_assert(sizeof(TriMidPush) == 80, "cloth tri-midphase blob must be 80 bytes");
static_assert(offsetof(TriMidPush, stackPtr) == 56, "stackPtr @56");
static_assert(offsetof(TriMidPush, stackSizeBytes) == 72, "stackSizeBytes @72");

namespace tmid
{
static const uint32_t NODE_WORDS = 292u;      // sizeof(Gu::BV32DataPacked) / 4
static const uint32_t HDR_WORDS = 4u;         // the uint4 the kernel skips

// A four-uint triangle-index record.  Declared here rather than reusing TriIdx4,
// which this file introduces further down, after this test.
struct Tri4 { uint32_t x, y, z, w; };
struct Box { float lo[3], hi[3]; };
struct Child { Box box; bool isLeaf; uint32_t primStart, nbPrims, childNode; };

// One BV32 node's worth of words, written into `w` at node index `k`.
static void writeNode(std::vector<uint32_t>& w, uint32_t k, const std::vector<Child>& kids)
{
    const uint32_t base = HDR_WORDS + k * NODE_WORDS;
    for (uint32_t j = 0; j < kids.size(); ++j)
    {
        for (int c = 0; c < 3; ++c)
        {
            w[base + j * 4u + uint32_t(c)]         = f2u(kids[j].box.lo[c]);
            w[base + 128u + j * 4u + uint32_t(c)]  = f2u(kids[j].box.hi[c]);
        }
        w[base + 256u + j] = kids[j].isLeaf
            ? ((kids[j].primStart << 7) | (kids[j].nbPrims << 1) | 1u)
            : (kids[j].childNode << 11);          // bit 0 clear == internal
    }
    w[base + 288u] = uint32_t(kids.size());       // mNbNodes
    w[base + 289u] = 1u;                          // mDepth (unread by the traversal)
}
} // namespace tmid

PXG_TEST(femClothSimTriMidphase)
{
    using namespace tmid;
    PxgSlangComputeDevice* dev = testDevice();

    // The rigid AABB and the contact distances.  Distinct offsets so a port that
    // reads only one, or the wrong cache ref, lands on a different fattened box.
    const float contactOffsetCloth = 0.30f, contactOffsetRigid = 0.10f;
    const float contactDist = contactOffsetCloth + contactOffsetRigid;   // 0.40
    const float rbLo[3] = { 2.0f, -1.0f, -1.0f };
    const float rbHi[3] = { 8.0f,  1.0f,  1.0f };
    const float fatLo[3] = { rbLo[0] - contactDist, rbLo[1] - contactDist, rbLo[2] - contactDist };
    const float fatHi[3] = { rbHi[0] + contactDist, rbHi[1] + contactDist, rbHi[2] + contactDist };

    // Triangles 0..17: a strip along +x, triangle i spanning x in [i, i+0.5].  With
    // the fattened box covering x in [1.6, 8.4] the strip STRADDLES both ends.
    //
    // Triangle 18 is the BAND triangle: x in [8.15, 8.35], INSIDE the fattened box
    // and OUTSIDE the raw one [2, 8].  Without it the fattening is unprobed -- the
    // first sweep lost `m5-box-not-fattened` and `m5-contactdist-rigid-only` to
    // exactly that, because no triangle sat in the band either spelling opens.
    //
    // Triangles 19..50 are a THIRTY-TWO primitive leaf.  A BV32 leaf's count lives
    // in `(data >> 1) & 63`, and with at most four primitives per leaf bit five is
    // never set, so `& 31` was indistinguishable.  Thirty-two is the largest count
    // a leaf can carry (one lane each) and the smallest that sets bit five.
    //
    // Triangle 51 is the HIGH-SLOT triangle, and it exists for the CHILD SLOT, not
    // for its geometry.  A leaf's slot lives in `(currentNodeIndex >> 1) & 31` -- five
    // bits, because a BV32 node has thirty-two children -- and this tree's root had
    // six, so slots never passed 5 and `& 15` was indistinguishable from `& 31`.  The
    // P9.17a module sweep lost `bv3-leafslot-mask-15` to exactly that.  This triangle
    // hangs off root child SEVENTEEN, and seventeen masked to four bits is ONE, which
    // is the LYING child -- so a four-bit mask does not merely mis-index, it reads the
    // liar's primitives (3..5, all overlapping) instead of this one.
    const uint32_t NB_STRIP = 18u;
    const uint32_t BAND_TRI = 18u;
    const uint32_t BIG_FIRST = 19u, BIG_COUNT = 32u;
    const uint32_t HIGH_TRI = BIG_FIRST + BIG_COUNT;     // 51
    const uint32_t HIGH_SLOT = 17u;                      // >= 16, and 17 & 15 == 1
    const uint32_t NB_TRIS = HIGH_TRI + 1u;              // 52
    std::vector<F4> cverts;
    std::vector<Tri4> ctris;
    auto addTri = [&](float x0, float x1) {
        const uint32_t b = uint32_t(cverts.size());
        cverts.push_back(F4{ x0, 0.0f, 0.0f, 1.0f });
        cverts.push_back(F4{ x1, 0.0f, 0.0f, 1.0f });
        cverts.push_back(F4{ 0.5f * (x0 + x1), 0.2f, 0.0f, 1.0f });
        ctris.push_back(Tri4{ b, b + 1u, b + 2u, 0u });
    };
    for (uint32_t i = 0; i < NB_STRIP; ++i) addTri(float(i), float(i) + 0.5f);
    addTri(8.15f, 8.35f);                                 // 18: the band triangle
    for (uint32_t k = 0; k < BIG_COUNT; ++k)              // 19..50: the big leaf
    { const float x = 1.0f + 0.25f * float(k); addTri(x, x + 0.2f); }
    addTri(4.6f, 4.8f);                                   // 51: the high-slot triangle
    // .w is NOT part of the triangle (M1): point it at a vertex far outside every
    // strip, so a port that folds it into the bound picks up 40 units of slack.
    cverts.push_back(F4{ 0.0f, 40.0f, 0.0f, 1.0f });
    const uint32_t decoyVert = uint32_t(cverts.size()) - 1u;
    for (auto& t : ctris) t.w = decoyVert;

    auto triBox = [&](uint32_t i) {
        Box b; const Tri4& t = ctris[i];
        const F4 v[3] = { cverts[t.x], cverts[t.y], cverts[t.z] };
        b.lo[0] = PxMin(v[0].x, PxMin(v[1].x, v[2].x)); b.hi[0] = PxMax(v[0].x, PxMax(v[1].x, v[2].x));
        b.lo[1] = PxMin(v[0].y, PxMin(v[1].y, v[2].y)); b.hi[1] = PxMax(v[0].y, PxMax(v[1].y, v[2].y));
        b.lo[2] = PxMin(v[0].z, PxMin(v[1].z, v[2].z)); b.hi[2] = PxMax(v[0].z, PxMax(v[1].z, v[2].z));
        return b;
    };
    auto meets = [&](const Box& b) {
        return !(b.lo[0] > fatHi[0] || fatLo[0] > b.hi[0] ||
                 b.lo[1] > fatHi[1] || fatLo[1] > b.hi[1] ||
                 b.lo[2] > fatHi[2] || fatLo[2] > b.hi[2]);
    };
    auto hull = [&](uint32_t from, uint32_t to) {
        Box b = triBox(from);
        for (uint32_t i = from + 1u; i < to; ++i)
        { const Box t = triBox(i);
          for (int c = 0; c < 3; ++c) { b.lo[c] = PxMin(b.lo[c], t.lo[c]); b.hi[c] = PxMax(b.hi[c], t.hi[c]); } }
        return b;
    };

    // ---- the tree ----
    // root: 5 children, interleaved hit/miss, one INTERNAL, one LYING.
    const Box FAR_BOX = { { 100.0f, 100.0f, 100.0f }, { 101.0f, 101.0f, 101.0f } };
    std::vector<Child> root = {
        { hull(0u, 3u),   true,  0u,        3u,        0u },  // 0: honest, prims 0..2
        { FAR_BOX,        true,  3u,        3u,        0u },  // 1: LYING -- 3..5 DO overlap
        { hull(6u, 10u),  false, 0u,        0u,        1u },  // 2: INTERNAL -> node 1
        { hull(BIG_FIRST, BIG_FIRST + BIG_COUNT),
                          true,  BIG_FIRST, BIG_COUNT, 0u },  // 3: the 32-primitive leaf
        { hull(BAND_TRI, BAND_TRI + 1u),
                          true,  BAND_TRI,  1u,        0u },  // 4: the band triangle
        { FAR_BOX,        true,  14u,       1u,        0u },  // 5: honest miss
    };
    // Children 6..16 pad the root out so the high-slot leaf lands at slot 17.  They
    // are honest misses on triangle 14 (x in [14, 14.5], far right of the fattened
    // box), so they add nothing to the expected set and cannot perturb the lie count.
    while (root.size() < HIGH_SLOT)
        root.push_back({ FAR_BOX, true, 14u, 1u, 0u });
    root.push_back({ hull(HIGH_TRI, HIGH_TRI + 1u), true, HIGH_TRI, 1u, 0u });   // 17
    // node 1: 3 leaves under the internal child, covering prims 6..9.
    std::vector<Child> n1 = {
        { hull(6u, 8u),  true, 6u, 2u, 0u },
        { FAR_BOX,       true, 16u, 2u, 0u },    // honest miss (prims 16,17 are far right)
        { hull(8u, 10u), true, 8u, 2u, 0u },
    };
    // Child 4 and node1's child 1 must be honest misses, so their primitives must
    // NOT overlap either -- checked below rather than assumed.

    std::vector<uint32_t> tree(HDR_WORDS + 2u * NODE_WORDS, 0u);
    writeNode(tree, 0u, root);
    writeNode(tree, 1u, n1);
    tree[0] = uint32_t(cverts.size()); tree[3] = 2u;   // the skipped header

    // ---- the expected pair set, from the tree's definition ----
    std::vector<uint32_t> want;
    uint32_t suppressedByLie = 0, straddleHit = 0, straddleMiss = 0;
    auto addLeaf = [&](const Child& c, bool reached) {
        for (uint32_t i = c.primStart; i < c.primStart + c.nbPrims; ++i)
        {
            const bool ov = meets(triBox(i));
            if (!reached) { if (ov) ++suppressedByLie; continue; }
            if (ov) { want.push_back(i); ++straddleHit; } else ++straddleMiss;
        }
    };
    for (const Child& c : root)
    {
        if (!c.isLeaf) continue;
        addLeaf(c, meets(c.box));
    }
    for (const Child& c : n1) addLeaf(c, meets(root[2].box) && meets(c.box));

    PXG_CHECK_MSG(suppressedByLie == 3u,
                  "the LYING node must hide exactly 3 overlapping triangles, it hides %u -- "
                  "without it no child-culling mutant can die", suppressedByLie);
    PXG_CHECK_MSG(straddleHit > 0 && straddleMiss > 0,
                  "the reached leaves are all-hit or all-miss (%u / %u) -- the per-primitive "
                  "test is not being exercised inside a tile", straddleHit, straddleMiss);
    PXG_CHECK_MSG(!want.empty(), "the expected pair set is EMPTY");
    // The band triangle must be IN the fattened box and OUT of the raw one, or the
    // fattening is unprobed whichever way a mutant spells it.
    {
        const Box b = triBox(BAND_TRI);
        const bool inFat = meets(b);
        const bool inRaw = !(b.lo[0] > rbHi[0] || rbLo[0] > b.hi[0] ||
                             b.lo[1] > rbHi[1] || rbLo[1] > b.hi[1] ||
                             b.lo[2] > rbHi[2] || rbLo[2] > b.hi[2]);
        PXG_CHECK_MSG(inFat && !inRaw,
                      "the band triangle must sit inside the FATTENED box and outside the raw "
                      "one (fat %d, raw %d) -- otherwise contactDistance is unprobed",
                      int(inFat), int(inRaw));
        PXG_CHECK_MSG(std::find(want.begin(), want.end(), BAND_TRI) != want.end(),
                      "the band triangle is not in the expected set");
    }
    // The high-slot leaf must sit at a slot with bit four set AND be reached, or the
    // five-bit slot mask is unprobed however a mutant spells it.
    {
        PXG_CHECK_MSG(root.size() == HIGH_SLOT + 1u && HIGH_SLOT >= 16u,
                      "the high-slot leaf must be at a slot >= 16 (root has %u children, "
                      "slot %u) -- below that, a four-bit mask is indistinguishable",
                      uint32_t(root.size()), HIGH_SLOT);
        PXG_CHECK_MSG(meets(root[HIGH_SLOT].box),
                      "the high-slot leaf is not reached, so its slot index is never decoded");
        PXG_CHECK_MSG(std::find(want.begin(), want.end(), HIGH_TRI) != want.end(),
                      "the high-slot triangle is not in the expected set");
        // and the slot it would be MISREAD as must say something different
        PXG_CHECK_MSG(root[HIGH_SLOT & 15u].primStart != root[HIGH_SLOT].primStart,
                      "slot %u masked to four bits is %u, and both name primStart %u -- "
                      "the mask would be unobservable", HIGH_SLOT, HIGH_SLOT & 15u,
                      root[HIGH_SLOT].primStart);
    }
    PXG_CHECK_MSG(BIG_COUNT == 32u,
                  "the big leaf carries %u primitives, not 32 -- bit five of `(data >> 1) & 63` "
                  "is only set at 32 and `& 31` is otherwise indistinguishable", BIG_COUNT);

    // ---- the rig ----
    DevBuf bTree = makeDev(dev, tree);
    DevBuf bCVerts = makeDev(dev, cverts);
    DevBuf bCTris = makeDev(dev, ctris);

    PxgFEMCloth fc; std::memset(&fc, 0, sizeof(fc));
    fc.mTriMeshData = reinterpret_cast<void*>(size_t(bTree.addr));
    fc.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bCVerts.addr));
    fc.mTriangleVertexIndices = reinterpret_cast<uint4*>(size_t(bCTris.addr));
    fc.mNbVerts = uint32_t(cverts.size());
    fc.mNbTriangles = NB_TRIS;
    // Cloth id 1, with a live decoy on slot 0.  The first draft made the decoy a COPY
    // of the real descriptor with only mNbTriangles changed -- and this kernel never
    // reads mNbTriangles, so `femClothes[0]` was an exact equivalent and the mutant
    // survived.  The decoy now owns its OWN tree, vertices and indices, parked far
    // from the rigid box, so reading slot 0 produces a deterministic empty set.
    std::vector<uint32_t> decoyTree(HDR_WORDS + NODE_WORDS, 0u);
    std::vector<Child> decoyRoot = { { FAR_BOX, true, 0u, 1u, 0u } };
    writeNode(decoyTree, 0u, decoyRoot);
    std::vector<F4> decoyVerts = { F4{ 100.0f, 100.0f, 100.0f, 1.0f },
                                   F4{ 100.5f, 100.0f, 100.0f, 1.0f },
                                   F4{ 100.2f, 100.2f, 100.0f, 1.0f } };
    std::vector<Tri4> decoyTris = { Tri4{ 0u, 1u, 2u, 0u } };
    DevBuf bDTree = makeDev(dev, decoyTree);
    DevBuf bDVerts = makeDev(dev, decoyVerts);
    DevBuf bDTris = makeDev(dev, decoyTris);
    PxgFEMCloth decoy; std::memset(&decoy, 0, sizeof(decoy));
    decoy.mTriMeshData = reinterpret_cast<void*>(size_t(bDTree.addr));
    decoy.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bDVerts.addr));
    decoy.mTriangleVertexIndices = reinterpret_cast<uint4*>(size_t(bDTris.addr));
    decoy.mNbVerts = 3u; decoy.mNbTriangles = 1u;
    std::vector<PxgFEMCloth> descArr = { decoy, fc };
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<PxgContactManagerInput> cmInputs(1);
    std::memset(cmInputs.data(), 0, sizeof(PxgContactManagerInput));
    cmInputs[0].shapeRef0 = 0u; cmInputs[0].shapeRef1 = 1u;
    cmInputs[0].transformCacheRef0 = 0u; cmInputs[0].transformCacheRef1 = 1u;
    DevBuf bCin = makeDev(dev, cmInputs);

    std::vector<PxgShape> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    shapes[0].type = PxU32(PxGeometryType::eTRIANGLEMESH);
    shapes[0].particleOrSoftbodyId = 1u;
    shapes[1].type = PxU32(PxGeometryType::eBOX);
    DevBuf bShapes = makeDev(dev, shapes);

    // bounds: slot 0 is the CLOTH's (never read by this kernel -- poisoned so a port
    // that reads it produces nonsense), slot 1 is the rigid's.
    std::vector<float> boundsArr = { -9e9f, -9e9f, -9e9f, 9e9f, 9e9f, 9e9f,
                                     rbLo[0], rbLo[1], rbLo[2], rbHi[0], rbHi[1], rbHi[2] };
    DevBuf bBounds = makeDev(dev, boundsArr);
    std::vector<float> cdArr = { contactOffsetCloth, contactOffsetRigid };
    DevBuf bCd = makeDev(dev, cdArr);
    DevBuf bTc = makeDev(dev, std::vector<PxsCachedTransform>(2));

    // M7: the stack is DELIBERATELY too small, so the counter must overshoot it.
    // JUST below the expected set, not far below.  A stack of 2 against 33 pairs was
    // tried and reverted: it fires the overshoot contract but leaves 31 of the 33
    // pairs uncompared, and a cap that hides most of the output hides most of the
    // errors (the P9.14b lesson).  28 keeps the drop observable and still compares
    // the great majority.
    const uint32_t STACK_PAIRS = 28u;
    const uint32_t SLACK_PAIRS = 6u;
    PXG_CHECK_MSG(want.size() > STACK_PAIRS,
                  "the expected set (%zu) does not exceed the stack (%u) -- M7's overshoot "
                  "contract is unprobed", want.size(), STACK_PAIRS);
    struct U4 { uint32_t x, y, z, w; };
    std::vector<U4> stack(STACK_PAIRS + SLACK_PAIRS, U4{ 0xEEEEEEEEu, 0xEEEEEEEEu, 0xEEEEEEEEu, 0xEEEEEEEEu });
    DevBuf bStack = makeDev(dev, stack);
    DevBuf bCount = makeDev(dev, std::vector<uint32_t>{ 0u });

    TriMidPush push = {};
    push.numTests = 1u; push.toleranceLength = 1.0f;
    push.cmInputs = bCin.addr; push.transformCache = bTc.addr; push.bounds = bBounds.addr;
    push.contactDistance = bCd.addr; push.gpuShapes = bShapes.addr; push.femClothes = bDesc.addr;
    push.stackPtr = bStack.addr; push.midphasePairsNum = bCount.addr;
    push.stackSizeBytes = STACK_PAIRS * 16u;

    Kernel k("femClothContactGenBda.spv", "cloth_midphaseGeneratePairsLaunch", {}, sizeof(TriMidPush));
    PXG_CHECK_MSG(k.isValid(), "the cloth tri-midphase pipeline must build");
    if (!k.isValid()) return;
    k.dispatch(/*gridX ceil(numTests / 2)*/ 1u, {}, push, 1u, 1u);
    dev->flush();

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
    const auto gStack = downloadBuffer<U4>(dev, bStack.h, STACK_PAIRS + SLACK_PAIRS);

    size_t mm = 0;
    // M7: the counter counts every reserved slot, INCLUDING the ones past the stack.
    PXG_CHECK_MSG(gCount == uint32_t(want.size()),
                  "the midphase counted %u pairs, expected %zu (the counter must NOT be capped "
                  "by the stack -- its overshoot is what sizes the next frame's stack)",
                  gCount, want.size());

    // The slot each pair lands in is the atomicAdd's order across tiles, so the
    // WRITTEN prefix is matched as a set against the expected triangles.
    const uint32_t written = PxMin(gCount, STACK_PAIRS);
    std::vector<bool> claimed(want.size(), false);
    for (uint32_t i = 0; i < written; ++i)
    {
        if (gStack[i].x != 0u) { ++mm; std::printf("  tri-midphase slot %u: cmIdx %u, expected 0\n", i, gStack[i].x); continue; }
        if (gStack[i].z != 0u || gStack[i].w != 0u) { ++mm; std::printf("  tri-midphase slot %u: .zw not zeroed\n", i); }
        size_t at = want.size();
        for (size_t q = 0; q < want.size(); ++q)
            if (!claimed[q] && want[q] == gStack[i].y) { at = q; break; }
        if (at == want.size())
        { ++mm; std::printf("  tri-midphase slot %u names triangle %u, which is not an unclaimed "
                            "expected pair\n", i, gStack[i].y); continue; }
        claimed[at] = true;
    }
    PXG_CHECK_MSG(mm == 0, "cloth_midphaseGeneratePairs emitted %zu wrong pairs", mm);

    // Nothing past the stack bound may have been written -- that is where the
    // overshoot would have landed had the drop not held.
    size_t badPoison = 0;
    for (uint32_t i = written; i < STACK_PAIRS + SLACK_PAIRS; ++i)
        if (gStack[i].x != 0xEEEEEEEEu || gStack[i].y != 0xEEEEEEEEu) ++badPoison;
    PXG_CHECK_MSG(badPoison == 0, "%zu stack slots past stackSize were written", badPoison);

    std::printf("    [ note ] %zu pairs expected over %u triangles (%u written, %zu dropped by the "
                "stack bound, %u hidden by the lying node, %u rejected inside reached leaves)\n",
                want.size(), NB_TRIS, written, want.size() - written, suppressedByLie, straddleMiss);

    dev->destroyBuffer(bTree.h); dev->destroyBuffer(bCVerts.h); dev->destroyBuffer(bCTris.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bCin.h); dev->destroyBuffer(bShapes.h);
    dev->destroyBuffer(bBounds.h); dev->destroyBuffer(bCd.h); dev->destroyBuffer(bTc.h);
    dev->destroyBuffer(bStack.h); dev->destroyBuffer(bCount.h);
    dev->destroyBuffer(bDTree.h); dev->destroyBuffer(bDVerts.h); dev->destroyBuffer(bDTris.h);
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
// P9.86c: PxAbs(float) is intrinsics::abs = ::fabsf (PxMath.h:112, PxUnixMathIntrinsics.h:57),
// which clears the sign of -0.  The old select spelling `a < 0 ? -a : a` returned -0 for -0,
// so mu * |dLN| came out -0 on a dLN == -0 lane and -min(dLT, -0) gave +0 where the CUDA
// gives -0.
static float rabs(float a) { return std::fabs(a); }

// PxMat33 (column-major [9]) * PxVec3.
static V3 mat33Mul(const float m[9], V3 v)
{
    return V3{ m[0] * v.x + m[3] * v.y + m[6] * v.z,
               m[1] * v.x + m[4] * v.y + m[7] * v.z,
               m[2] * v.x + m[5] * v.y + m[8] * v.z };
}

// PxComputeBasisVectors (PxMathUtils.h:297).  P9.36: this comment used to say
// "right/up .normalize() is v / sqrt(magSq)" -- it is not.  PxVec3::normalize scales
// through operator/= (PxVec3.h:260): ONE reciprocal, three multiplies.
static void computeBasis(V3 dir, V3& right, V3& up)
{
    if (rabs(dir.y) <= 0.9999f)
    {
        right = V3{ dir.z, 0.0f, -dir.x };
        const float m = std::sqrt(right.x * right.x + right.y * right.y + right.z * right.z);
        if (m > 0.0f) { const float r = 1.0f / m; right = V3{ right.x * r, right.y * r, right.z * r }; }
        up = V3{ dir.y * right.z, dir.z * right.x - dir.x * right.z, -dir.y * right.x };
    }
    else
    {
        right = V3{ 1.0f, 0.0f, 0.0f };
        up = V3{ 0.0f, dir.z, -dir.y };
        const float m = std::sqrt(up.x * up.x + up.y * up.y + up.z * up.z);
        if (m > 0.0f) { const float r = 1.0f / m; up = V3{ up.x * r, up.y * r, up.z * r }; }
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

// FEMCollision::initialize (deformableUtils.cuh:280) — null-material (frictionless) path.
// The rigid velocity state is a PARAMETER (P9.45): it defaults to all zeros, which is what a
// static body at rest reads, and the dynamic case passes what PxgVelocityReader would return.
struct RefRigidVel { V3 linVel{ 0, 0, 0 }, angVel{ 0, 0, 0 }, linDelta{ 0, 0, 0 }, angDelta{ 0, 0, 0 }; };
struct InitResult { bool isActive; float deltaLambdaN; float deltaLambdaT; V3 normal; V3 tangent;
                    V3 raXn; V3 raXt; float accum;
                    // P9.46: did the FRICTION term bind the tangent clamp, or was the raw
                    // tangent magnitude already the smaller of the two?  A test that only
                    // checks deltaLambdaT != 0 cannot tell, and every mutant of the
                    // deformable-material read survived precisely because of that.
                    bool frictionBound; float dLTraw; float dLTlimit;
                    // P9.86c coverage: deltaLambdaN met the (+0, -0) PxMax tie, and the
                    // tangent branch ran (the only place the tie reaches an output).
                    bool dlnZeroTie; bool tanBranch; };

// P9.86c mutant twin of the deltaLambdaN PxMax (deformableUtils.cuh:374): 1 = the pre-fix
// port's max(), i.e. GLSL FMax as llvmpipe evaluates it (F4: +0 on a (+0,-0) tie in either
// operand order, the non-NaN operand for a NaN).  0 = selectMax, the CUDA.
static int gDlnVariant = 0;
static float llvmpipeFMax(float a, float b)
{
    if (a != a) return b;
    if (b != b) return a;
    if (a == 0.0f && b == 0.0f) return (std::signbit(a) && std::signbit(b)) ? -0.0f : 0.0f;
    return a > b ? a : b;
}
// PxsCombinePxReal (PxsMaterialCombiner.h:53).  PxMin/PxMax are the intrinsic selects
// (PxUnixMathIntrinsics.h:113/:122), not the generic templates.
static float refCombineFriction(float val0, float val1, uint32_t mode)
{
    if (mode == 0u) return 0.5f * (val0 + val1);            // eAVERAGE
    if (mode == 1u) return (val0 < val1) ? val0 : val1;     // eMIN
    if (mode == 2u) return val0 * val1;                     // eMULTIPLY
    if (mode == 3u) return (val0 > val1) ? val0 : val1;     // eMAX
    return 0.0f;
}

static InitResult refInitialize(const RigidContactConstraintHost& c, int slot, bool isTGS,
                                float appliedForceRef, float dt, V3 bcVec, V3 deformableLinDelta,
                                V3 deformableInvMasses, float rigidRefCount, bool wasActive, bool checkOnly,
                                RefRigidVel rv = RefRigidVel{}, float combinedFriction = 0.0f)
{
    InitResult r;
    r.isActive = false; r.deltaLambdaN = 0.0f; r.deltaLambdaT = 0.0f;
    r.tangent = V3{ 0, 0, 0 }; r.raXt = V3{ 0, 0, 0 }; r.accum = appliedForceRef;
    r.frictionBound = false; r.dLTraw = 0.0f; r.dLTlimit = 0.0f;
    r.dlnZeroTie = false; r.tanBranch = false;

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
    r.raXn = raXn;   // :306, live on both early-return paths

    const V3 linVel = rv.linVel, angVel = rv.angVel;
    const V3 rigidLinDelta = rv.linDelta, rigidAngDelta = rv.angDelta;

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

    // PxMax(float) = intrinsics::selectMax (PxUnixMathIntrinsics.h:122): on a tie the
    // SECOND operand survives, so (+0, -0) gives -0.
    const float dlnA = -CN * invDenom, dlnB = -r.accum;
    r.dlnZeroTie = (f2u(dlnA) == 0x00000000u && f2u(dlnB) == 0x80000000u);
    r.deltaLambdaN = (gDlnVariant == 1) ? llvmpipeFMax(dlnA, dlnB) : rmax(dlnA, dlnB);
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
        r.tanBranch = true;
        const float CT = std::sqrt(tanMagSq);
        const float invTanMag = 1.0f / CT;
        r.tangent = V3{ relTanDelta.x * invTanMag, relTanDelta.y * invTanMag, relTanDelta.z * invTanMag };
        // raXt = frac0*raXnF0 + frac1*raXnF1 (:391-394) -- P9.45; zero outside this branch.
        const float frac0 = dot3(r.tangent, fric0);
        const float frac1 = dot3(r.tangent, fric1);
        r.raXt = V3{ frac0 * raXnF0.x + frac1 * raXnF1.x,
                     frac0 * raXnF0.y + frac1 * raXnF1.y,
                     frac0 * raXnF0.z + frac1 * raXnF1.z };
        const float uT0 = rigidRefCount * raXnF0_resp.w + deformableInvMass;
        const float invT0 = (uT0 > 0.0f) ? (1.0f / uT0) : 0.0f;
        const float uT1 = rigidRefCount * raXnF1_resp.w + deformableInvMass;
        const float invT1 = (uT1 > 0.0f) ? (1.0f / uT1) : 0.0f;
        const float dLT0 = CT0 * invT0;
        const float dLT1 = CT1 * invT1;
        const float dLT = std::sqrt(dLT0 * dLT0 + dLT1 * dLT1);
        const float lim = combinedFriction * rabs(r.deltaLambdaN);
        r.dLTraw = dLT; r.dLTlimit = lim; r.frictionBound = (lim < dLT);
        r.deltaLambdaT = -rmin(dLT, lim);
    }
    return r;
}

// FEMCollision::computeRigidChange (deformableUtils.cuh:418-429) — P9.45.  The reference
// count multiplies the SUM; the linear row multiplies by rigidInvMass FIRST.
static void refComputeRigidChange(const InitResult& r, bool isArticulation, float rigidInvMass,
                                  float rigidRefCount, V3& deltaLinVel0, V3& deltaAngVel0)
{
    const V3 angBase = V3{ r.raXn.x * r.deltaLambdaN + r.raXt.x * r.deltaLambdaT,
                           r.raXn.y * r.deltaLambdaN + r.raXt.y * r.deltaLambdaT,
                           r.raXn.z * r.deltaLambdaN + r.raXt.z * r.deltaLambdaT };
    deltaAngVel0 = isArticulation
                 ? angBase
                 : V3{ angBase.x * rigidRefCount, angBase.y * rigidRefCount, angBase.z * rigidRefCount };

    const V3 linBase = V3{ r.normal.x * r.deltaLambdaN + r.tangent.x * r.deltaLambdaT,
                           r.normal.y * r.deltaLambdaN + r.tangent.y * r.deltaLambdaT,
                           r.normal.z * r.deltaLambdaN + r.tangent.z * r.deltaLambdaT };
    deltaLinVel0 = V3{ (linBase.x * rigidInvMass) * rigidRefCount,
                       (linBase.y * rigidInvMass) * rigidRefCount,
                       (linBase.z * rigidInvMass) * rigidRefCount };
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
    kPrep.dispatch(FEM_PREP_GRID, {}, pp, 1u, 1u);
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

// ---------------------------------------------------------------------------
// P9.45 — the DYNAMIC-rigid reaction.  runRigidContact above seeds one contact against a
// STATIC floor, which is the only geometry the drape scenes needed and the reason
// FEMCollision::computeRigidChange (deformableUtils.cuh:418) could be deferred: with
// rigidId.isStaticBody() the .cu's own guard never fires.  This case seeds SIX contacts
// that between them take every branch of that guard, plus the rigid reference-count
// increment the query pass now performs:
//
//   lanes 0..2  DYNAMIC rigid body (node index 3 -> solver body 1, invMass 0.25).  Three
//               contacts on ONE body, so the query's atomic must reach 3 and the solve must
//               read 3 back -- a single contact would leave the count at 1, where a dropped
//               reference-count multiply is invisible.
//   lane 3      STATIC (node 0xFFFFFFFF): globalRigidBodyId == -1, no increment, zeros.
//   lane 4      KINEMATIC (node index 4 -> solver body 2, invMass 0): the invMass half of
//               the guard, which the static lane cannot test on its own.
//   lane 5      ARTICULATION LINK (node index 5 with the articulation bit): the count IS
//               incremented, exactly as the .cu does, but the reaction is NOT written.  That
//               is this module's articulation deferral, and pinning it here means the day
//               prepareFEMContacts and readRigidVelocity grow their articulation branches,
//               this assertion goes red and forces the reaction on with them.
//
// Every cloth vertex is distinct, so the cloth-side mass split stays 1 per vertex and the
// RIGID reference count is the only multi-contact quantity under test.
// ---------------------------------------------------------------------------
static void runRigidContactDynamic(bool isTGS)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 1.0f / 60.0f;

    const uint32_t N = 11u;         // 8 VERTEX + 2 TRIANGLE + 1 SEPARATING-but-active
    const uint32_t NV = 15u;        // 8 vertex lanes + 3 + 3 for the triangles + 1 for lane 10
    const uint32_t numSolverBodies = 3u;
    const uint32_t accOffset = 0u;
    const float rigidInvMass = 0.25f;

    // ---- cloth vertices.  Lanes 0..7 own vertices 0..7 one-to-one; triangle lane 8 owns
    // vertices 8,9,10 and lane 9 owns 11,12,13.  DISJOINT on purpose: every vertex then has
    // exactly one writer, so the scattered deltas stay bit-exactly comparable instead of
    // needing the tolerance an atomically-shared vertex would force.
    const float invMass = 0.5f;
    std::vector<F4> positions(NV), velocities(NV), accumDelta(NV), deltaPos(NV);
    for (uint32_t i = 0; i < NV; ++i)
    {
        const float f = float(i);
        positions[i] = { 0.5f + 0.11f * f, -0.02f - 0.003f * f, 0.7f - 0.07f * f, invMass };
        velocities[i] = { 0.0f, 0.0f, 0.0f, invMass };
        accumDelta[i] = { 0.0010f + 0.0004f * f, -0.0020f + 0.0003f * f, 0.0015f - 0.0002f * f, invMass };
        deltaPos[i] = { 0.0f, 0.0f, 0.0f, 0.0f };
    }
    // Two triangles, as uint4 rows (the fourth component is unread).
    const uint32_t triVerts[2][3] = { { 8u, 9u, 10u }, { 11u, 12u, 13u } };
    std::vector<uint32_t> triVtx(8u, 0u);
    for (uint32_t t = 0; t < 2u; ++t)
        for (uint32_t k = 0; k < 3u; ++k) triVtx[t * 4u + k] = triVerts[t][k];

    // P9.46: the per-vertex friction array a VERTEX contact reads (PxgFEMCloth.h:87).  A
    // cloth has TWO friction sources and only the .cu says which applies: a TRIANGLE contact
    // goes through mMaterialIndices into the surface material table, a VERTEX contact reads
    // this array directly (deformableUtils.cuh:514 vs :533).  Every lane here is a vertex
    // contact, so this is the live source; the table below still has to exist, because the
    // read is gated on `materials != NULL` rather than on this pointer.
    std::vector<float> clothVertFriction(NV);
    for (uint32_t i = 0; i < NV; ++i)
        clothVertFriction[i] = 0.10f + 0.07f * float(i);   // all distinct
    // The triangle lanes' vertices carry POISON: a TRIANGLE contact must take its friction
    // from the material table, never from this array.
    for (uint32_t v = 8u; v < NV - 1u; ++v) clothVertFriction[v] = -7.5f;
    // ...but vertex 14 belongs to the SEPARATING vertex lane 10, so it keeps a real value.
    clothVertFriction[NV - 1u] = 0.22f;

    DevBuf bPos = makeDev(dev, positions);
    DevBuf bVel = makeDev(dev, velocities);
    DevBuf bAcc = makeDev(dev, accumDelta);
    DevBuf bDelta = makeDev(dev, deltaPos);
    DevBuf bTri = makeDev(dev, triVtx);
    DevBuf bClothFric = makeDev(dev, clothVertFriction);

    const float penBiasClampFEM = -1.0e10f;
    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bPos.addr));
    c.mVelocity_InvMass = reinterpret_cast<decltype(c.mVelocity_InvMass)>(size_t(bVel.addr));
    c.mAccumulatedDeltaPos = reinterpret_cast<decltype(c.mAccumulatedDeltaPos)>(size_t(bAcc.addr));
    c.mDeltaPos = reinterpret_cast<decltype(c.mDeltaPos)>(size_t(bDelta.addr));
    c.mTriangleVertexIndices = reinterpret_cast<decltype(c.mTriangleVertexIndices)>(size_t(bTri.addr));
    c.mDynamicFrictions = reinterpret_cast<decltype(c.mDynamicFrictions)>(size_t(bClothFric.addr));
    c.mPenBiasClamp = penBiasClampFEM;
    c.mNbVerts = NV;
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    // P9.46 -- MATERIALS.
    // Rigid table: four PxsMaterialData records (24 B), one per PxCombineMode, so every arm
    // of PxsCombinePxReal runs.  dynamicFriction @0; the packed word @16 is
    // flags | fricCombineMode<<16 | restCombineMode<<24, and ONLY bits 16..23 may be read --
    // the other bytes carry decoys, so a shader that took the whole word or the wrong byte
    // lands on an out-of-range mode and PxsCombinePxReal's fall-through 0.
    // Material 4 carries an OUT-OF-RANGE combine mode.  PxCombineMode has four values and
    // the engine asserts that, so this is not a state a valid scene reaches -- but
    // PxsCombinePxReal has a fall-through `return 0.0f` for it, and that arm is the port's
    // answer to a corrupt or uninitialised material byte: NO friction, rather than silently
    // one side's.  Without a lane pointing here that arm is unreachable and its mutant lives.
    const float rigidMatFriction[5] = { 0.30f, 0.80f, 0.50f, 0.20f, 0.55f };
    const uint32_t rigidMatMode[5] = { 0u, 1u, 2u, 3u, 7u };   // eAVERAGE, eMIN, eMULTIPLY, eMAX, invalid
    std::vector<uint32_t> rigidMats(5u * 6u, 0u);
    for (uint32_t m = 0; m < 5u; ++m)
    {
        uint32_t* w = &rigidMats[m * 6u];
        std::memcpy(&w[0], &rigidMatFriction[m], 4);
        w[1] = 0x7f7f7f7fu; w[2] = 0x7f7f7f7fu; w[3] = 0x7f7f7f7fu;
        w[4] = 0xBEEFu | (rigidMatMode[m] << 16) | (0xA5u << 24);
        w[5] = 0x000000C3u;
    }
    DevBuf bRigidMats = makeDev(dev, rigidMats);

    // Surface table: PxsDeformableSurfaceMaterialData, 32 B, dynamicFriction @8.  Slot 0 is
    // a DECOY with friction 0 -- an index read that ignored mMaterialIndices and took slot 0
    // would silently produce a frictionless triangle contact and nothing else would say so.
    // Slots 2 and 3 are the ones the two TRIANGLE lanes read, and their values are chosen
    // so the FRICTION binds the tangent clamp there: deltaLambdaT = -min(dLT, mu*|dLN|), and
    // with a large mu the raw tangent magnitude is the smaller term and mu never reaches the
    // output at all.  The first version of this fixture had 0.25 and 0.85, both large enough
    // that every mutant of the deformable-material read survived -- measured, not guessed:
    // lane 8 had dLT 0.0484 against a limit of 0.258, lane 9 dLT 0.0437 against 1.00.
    // They are also DIFFERENT from each other, so the PxU16 half-selection is observable.
    const float clothMatFriction[4] = { 0.00f, 0.65f, 0.05f, 0.03f };
    std::vector<float> clothMats(4u * 8u, 0.0f);
    for (uint32_t m = 0; m < 4u; ++m)
    {
        clothMats[m * 8u + 0u] = 1.0e6f;                 // youngs
        clothMats[m * 8u + 1u] = 0.45f;                  // poissons (the neighbouring field)
        clothMats[m * 8u + 2u] = clothMatFriction[m];    // dynamicFriction @8
    }
    DevBuf bClothMats = makeDev(dev, clothMats);

    // mMaterialIndices is a PxU16 array indexed by TRIANGLE: tri 0 -> material 2 (low half
    // of word 0), tri 1 -> material 3 (HIGH half), so femMaterialIndexAt's halving is under
    // test as well as the table read.
    const uint16_t clothMatIdx[2] = { 2u, 3u };
    std::vector<uint32_t> clothMatIdxWords(1u, uint32_t(clothMatIdx[0]) | (uint32_t(clothMatIdx[1]) << 16));
    DevBuf bClothMatIdx = makeDev(dev, clothMatIdxWords);
    descArr[0].mMaterialIndices = reinterpret_cast<decltype(descArr[0].mMaterialIndices)>(size_t(bClothMatIdx.addr));
    dev->upload(bDesc.h, descArr.data(), descArr.size() * sizeof(PxgFEMCloth));

    // Lanes 0..7 cover all four combine modes twice.  The two TRIANGLE lanes take MULTIPLY
    // and MIN, which keep their combined friction small enough for the clamp to bind.
    const uint32_t rigidMatOf[11] = { 0u, 1u, 2u, 3u, 0u, 1u, 4u, 3u, 2u, 1u, 0u };
    // Which cloth element each lane names, and whether it is a TRIANGLE contact.
    const uint32_t elemOf[11] = { 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 0u, 1u, 14u };
    const bool isTriOf[11] = { false, false, false, false, false,
                               false, false, false, true, true, false };
    // Lane 10 is SEPARATING and pre-marked in-collision, and carries an accumulated lambda
    // from an earlier solver iteration.  That is the one shape in which deltaLambdaN goes
    // NEGATIVE -- dLN = max(-CN*invDenom, -accum) with CN > 0 -- and it is the only way the
    // abs() in the tangent clamp is observable.  It also covers the .cu's "once active, stay
    // active" branch on the cloth side, which until now only the softbody fixture drove.
    const bool separatingOf[11] = { false, false, false, false, false, false,
                                    false, false, false, false, true };
    const float seededLambda[11] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                     0.0f, 0.0f, 0.0f, 0.0f, 0.75f };

    // ---- the six rigid ids, and which solver body each resolves to -------
    const uint64_t staticNodeInd = 0x00000000FFFFFFFFull;
    const uint64_t dynNodeInd = 3ull;                    // articulation bit (high word bit0) clear
    const uint64_t kinNodeInd = 4ull;
    const uint64_t artiNodeInd = (1ull << 32) | 5ull;    // articulation bit SET
    // Lanes 6 and 7 are dynamic as well; they exist for computeBasisVectors' SECOND branch
    // (see the normal table below), and being dynamic gives them reaction coverage too.
    const uint64_t nodeInds[11] = { dynNodeInd, dynNodeInd, dynNodeInd,
                                    staticNodeInd, kinNodeInd, artiNodeInd,
                                    dynNodeInd, dynNodeInd,
                                    dynNodeInd, dynNodeInd,     // the two TRIANGLE lanes
                                    dynNodeInd };               // the separating lane
    // solverBodyIndices[nodeIndex]: node 3 -> body 1 (dynamic), node 4 -> body 2 (kinematic),
    // node 5 -> body 1 as well, so the articulation lane carries a NON-ZERO invMass and only
    // the articulation test can stop its reaction.
    std::vector<uint32_t> solverBodyIndices = { 0u, 0u, 0u, 1u, 2u, 1u };
    const uint32_t bodyOf[11] = { 1u, 1u, 1u, 0u, 2u, 1u, 1u, 1u, 1u, 1u, 1u }; // static reads slot 0
    // getGlobalRigidBodyId: -1 static, solverBodyIdx rigid, numSolverBodies+idx articulation.
    const int globalIdOf[11] = { 1, 1, 1, -1, 2, int(numSolverBodies) + 1, 1, 1, 1, 1, 1 };
    DevBuf bSolverBodyIndices = makeDev(dev, solverBodyIndices);

    // ---- contact geometry: a distinct tilted normal per lane -------------
    std::vector<F4> contacts(N), normalPens(N), barycentrics(N);
    std::vector<FemOtherContactInfoHost> infos(N);
    // PxComputeBasisVectors branches on |dir.y| <= 0.9999f, and dir is the SOLVER normal,
    // which prepare forms by negating the plane normal.  Lanes 0..5 spread six tilted
    // normals across the FIRST branch (right = normalize(dir.z, 0, -dir.x)); lane 6 is
    // near-vertical so the SECOND branch runs (right = (1,0,0), up normalised instead);
    // lane 7 sits EXACTLY on the 0.9999 boundary, where <= and < disagree, and is left
    // un-renormalised so that |y| is exactly the literal.  Without 6 and 7 the second
    // normalisation and the comparison operator are both unreachable, and mutants of each
    // survived the first sweep.
    p714ref::V3 planeNormals[11];
    for (uint32_t i = 0; i < N; ++i)
    {
        // Lane 6's exact digits matter: the second branch normalises up = (0, dir.z, -dir.y),
        // and for most near-vertical normals `up.k / m` and `up.k * (1/m)` happen to agree
        // in every component, so the first near-vertical lane tried here still let the
        // three-divide mutant through.  This one was searched for: it differs in the z
        // component (-0.999999702 divided against -0.999999762 reciprocated).
        if (i == 6u) { planeNormals[i] = { -0.000200001727f, -0.999999642f, -0.000800006907f }; continue; }
        if (i == 7u) { planeNormals[i] = { 0.0099999f, -0.9999f, 0.0099999f }; continue; }
        float raw[3] = { 0.2f + 0.09f * float(i), 0.9f - 0.05f * float(i), 0.3f - 0.04f * float(i) };
        const float m = std::sqrt(raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]);
        planeNormals[i] = { raw[0] / m, raw[1] / m, raw[2] / m };
    }
    for (uint32_t i = 0; i < N; ++i)
    {
        contacts[i] = { positions[i].x, positions[i].y, positions[i].z, 0.0f };   // restW = 0
        // Deep enough that the contact stays ACTIVE in BOTH modes: TGS folds the rigid
        // position delta into the error term at invDt, so a shallow penetration flips to
        // inactive there while staying active under PGS.
        // A SEPARATING lane needs a POSITIVE penetration so CN comes out positive.
        normalPens[i] = { planeNormals[i].x, planeNormals[i].y, planeNormals[i].z,
                          separatingOf[i] ? 0.02f : (-0.05f - 0.004f * float(i)) };
        // A TRIANGLE contact is bc.w == 0 and carries three real weights; a VERTEX contact
        // is bc.w != 0 and only bc.x is used (as 1.0).  The two take DIFFERENT friction
        // sources, which is the whole reason lanes 8 and 9 exist.
        barycentrics[i] = isTriOf[i]
                        ? (i == 8u ? F4{ 0.5f, 0.3f, 0.2f, 0.0f } : F4{ 0.25f, 0.6f, 0.15f, 0.0f })
                        : F4{ 0.0f, 0.0f, 0.0f, 1.0f };
        infos[i].pairInd0 = nodeInds[i];
        infos[i].pairInd1 = elemOf[i];               // (clothId 0 << 20) | element index
        // bits 0..30 are the rigid material index; bit 31 is isInCollision, and the
        // separating lane is pre-marked so the query keeps it active without re-testing.
        infos[i].rigidMatInd_isInCollision = rigidMatOf[i] | (separatingOf[i] ? 0x80000000u : 0u);
    }

    DevBuf bContacts = makeDev(dev, contacts);
    DevBuf bNormalPens = makeDev(dev, normalPens);
    DevBuf bBary = makeDev(dev, barycentrics);
    DevBuf bInfos = makeDev(dev, infos);
    std::vector<uint32_t> numContacts = { N };
    DevBuf bNum = makeDev(dev, numContacts);

    std::vector<uint8_t> constraintBytes(sizeof(RigidContactConstraintHost), 0xCD);
    DevBuf bConstraint = makeDev(dev, constraintBytes);

    std::vector<float> rigidLambdaNs(N, 123.0f);
    DevBuf bLambda = makeDev(dev, rigidLambdaNs);
    std::vector<F4> rigidDeltaVel(2u * N, F4{ 9.f, 9.f, 9.f, 9.f });
    DevBuf bRigidDeltaVel = makeDev(dev, rigidDeltaVel);
    // numSolverBodies + numArticulations entries, exactly as PxgFEMClothCore.cpp:320 sizes it.
    std::vector<uint32_t> refCounts(numSolverBodies + 2u, 0u);
    DevBuf bRefCounts = makeDev(dev, refCounts);

    // ---- three solver bodies: frames, masses, sqrt inverse inertias ------
    std::vector<F4> body2World = {
        { 0.f, 0.f, 0.f, 1.f }, { 0.00f, 0.00f, 0.00f, 0.f },   // body 0 (static slot)
        { 0.f, 0.f, 0.f, 1.f }, { 0.10f, -0.30f, 0.20f, 0.f },  // body 1 (dynamic)
        { 0.f, 0.f, 0.f, 1.f }, { -0.05f, 0.40f, 0.15f, 0.f },  // body 2 (kinematic)
    };
    DevBuf bBody2World = makeDev(dev, body2World);

    const float penBiasClampRigid = -1.0e10f;
    const float invMassOf[3] = { 0.0f, rigidInvMass, 0.0f };
    std::vector<uint8_t> solverBodyData(96u * numSolverBodies, 0);
    for (uint32_t b = 0; b < numSolverBodies; ++b)
    {
        std::memcpy(solverBodyData.data() + 96u * b + 12, &penBiasClampRigid, 4);  // penBiasClamp @12
        std::memcpy(solverBodyData.data() + 96u * b + 28, &invMassOf[b], 4);       // invMass @28
    }
    DevBuf bSolverBodyData = makeDev(dev, solverBodyData);

    const float sqrtInvInertia0[9] = { 2.0f, 0.1f, 0.0f, 0.1f, 1.5f, 0.2f, 0.0f, 0.2f, 1.0f };
    const float sqrtInvInertia1[9] = { 1.3f, -0.2f, 0.05f, -0.2f, 0.9f, 0.15f, 0.05f, 0.15f, 1.7f };
    const float sqrtInvInertia2[9] = { 0.7f, 0.3f, -0.1f, 0.3f, 1.1f, 0.0f, -0.1f, 0.0f, 0.8f };
    const float* sqrtInvOf[3] = { sqrtInvInertia0, sqrtInvInertia1, sqrtInvInertia2 };
    std::vector<uint8_t> solverTxIData(64u * numSolverBodies, 0);
    for (uint32_t b = 0; b < numSolverBodies; ++b)
        std::memcpy(solverTxIData.data() + 64u * b + 28, sqrtInvOf[b], 9 * 4);   // sqrtInvInertia @28
    DevBuf bSolverTxIData = makeDev(dev, solverTxIData);

    // ---- velocities.  Body 0 stays at rest so the STATIC lane reads zeros,
    //      exactly as it does in runRigidContact; bodies 1 and 2 move.
    std::vector<F4> outSolverVelocity(2u * numSolverBodies, F4{ 0.f, 0.f, 0.f, 0.f });
    outSolverVelocity[1] = { 0.30f, -0.70f, 0.20f, 0.f };                      // body 1 linear
    outSolverVelocity[2] = { 0.10f, 0.20f, 0.30f, 0.f };                       // body 2 linear
    outSolverVelocity[1 + numSolverBodies] = { 0.11f, 0.22f, -0.33f, 0.f };    // body 1 angular
    outSolverVelocity[2 + numSolverBodies] = { 0.04f, 0.05f, 0.06f, 0.f };     // body 2 angular
    DevBuf bOutSolverVel = makeDev(dev, outSolverVelocity);

    // The TGS rows carry velocity in v0/v1.x-y and POSITION DELTA in v1.z-w / v2, and the
    // deltas enter the error at invDt (x60 here), so they are deliberately two orders
    // smaller than the velocities -- large ones drive every contact inactive.
    std::vector<F4> solverBodyVelPool(3u * numSolverBodies, F4{ 0.f, 0.f, 0.f, 0.f });
    solverBodyVelPool[1] = { 0.05f, -0.09f, 0.03f, 0.14f };
    solverBodyVelPool[2] = { -0.02f, 0.07f, 0.01f, -0.06f };
    solverBodyVelPool[1 + numSolverBodies] = { 0.21f, -0.12f, 0.0033f, -0.0044f };
    solverBodyVelPool[2 + numSolverBodies] = { 0.08f, 0.09f, -0.0010f, 0.0011f };
    solverBodyVelPool[1 + 2u * numSolverBodies] = { 0.0055f, -0.0066f, 0.0077f, -0.0088f };
    solverBodyVelPool[2 + 2u * numSolverBodies] = { -0.0015f, 0.0016f, -0.0017f, 0.0018f };
    DevBuf bSolverBodyVelPool = makeDev(dev, solverBodyVelPool);

    // readRigidVelocity (pxgSlangFemRigidContact.slang:222) mirrored on the host.
    auto rigidVelOf = [&](uint32_t body) {
        p714ref::RefRigidVel rv;
        if (isTGS)
        {
            const F4 v0 = solverBodyVelPool[accOffset + body];
            const F4 v1 = solverBodyVelPool[accOffset + body + numSolverBodies];
            const F4 v2 = solverBodyVelPool[accOffset + body + 2u * numSolverBodies];
            rv.linVel = { v0.x, v0.y, v0.z };
            rv.angVel = { v0.w, v1.x, v1.y };
            rv.linDelta = { v1.z, v1.w, v2.x };
            rv.angDelta = { v2.y, v2.z, v2.w };
        }
        else
        {
            const F4 iv0 = outSolverVelocity[body];
            const F4 iv1 = outSolverVelocity[body + numSolverBodies];
            const F4 dv0 = solverBodyVelPool[accOffset + body];
            const F4 dv1 = solverBodyVelPool[accOffset + body + numSolverBodies];
            rv.linVel = { iv0.x + dv0.x, iv0.y + dv0.y, iv0.z + dv0.z };
            rv.angVel = { iv1.x + dv1.x, iv1.y + dv1.y, iv1.z + dv1.z };
        }
        return rv;
    };

    // ---- descriptors ------------------------------------------------------
    std::vector<uint8_t> prepDesc(304, 0);
    pokePtr(prepDesc, 64, bSolverBodyData.addr);
    pokePtr(prepDesc, 72, bSolverTxIData.addr);
    pokePtr(prepDesc, 168, bBody2World.addr);
    DevBuf bPrepDesc = makeDev(dev, prepDesc);

    std::vector<uint8_t> prePrepDesc(760, 0);
    pokePtr(prePrepDesc, 272, bSolverBodyIndices.addr);
    DevBuf bPrePrepDesc = makeDev(dev, prePrepDesc);

    std::vector<uint8_t> solverCoreDesc(320, 0);
    pokePtr(solverCoreDesc, 0, bOutSolverVel.addr);
    pokeU32(solverCoreDesc, 268, numSolverBodies);
    pokeU32(solverCoreDesc, 276, accOffset);
    DevBuf bSolverCoreDesc = makeDev(dev, solverCoreDesc);

    std::vector<uint8_t> sharedDesc(224, 0);
    pokePtr(sharedDesc, 160, bSolverBodyVelPool.addr);
    DevBuf bSharedDesc = makeDev(dev, sharedDesc);

    // ---- per-lane element shape.  A VERTEX lane owns one cloth vertex and blends nothing;
    // a TRIANGLE lane owns three and blends them by the barycentric.  readCloth and the
    // prepare kernel both branch on bc.w, so the reference has to as well.
    auto laneVerts = [&](uint32_t i, uint32_t out[3]) -> uint32_t {
        if (!isTriOf[i]) { out[0] = elemOf[i]; return 1u; }
        for (uint32_t k = 0; k < 3u; ++k) out[k] = triVerts[elemOf[i]][k];
        return 3u;
    };
    auto laneBcVec = [&](uint32_t i) {
        return isTriOf[i] ? p714ref::V3{ barycentrics[i].x, barycentrics[i].y, barycentrics[i].z }
                          : p714ref::V3{ 1.0f, 0.0f, 0.0f };
    };
    // The accumulated delta the prepare and the solve both use: the blend for a triangle
    // (femClothRigidContactBda.slang:288), the vertex's own row otherwise.
    auto laneAccum = [&](uint32_t i) {
        if (!isTriOf[i])
            return p714ref::V3{ accumDelta[elemOf[i]].x, accumDelta[elemOf[i]].y, accumDelta[elemOf[i]].z };
        const F4 bc = barycentrics[i];
        const uint32_t* tv = triVerts[elemOf[i]];
        return p714ref::V3{
            accumDelta[tv[0]].x * bc.x + accumDelta[tv[1]].x * bc.y + accumDelta[tv[2]].x * bc.z,
            accumDelta[tv[0]].y * bc.x + accumDelta[tv[1]].y * bc.y + accumDelta[tv[2]].y * bc.z,
            accumDelta[tv[0]].z * bc.x + accumDelta[tv[1]].z * bc.y + accumDelta[tv[2]].z * bc.z };
    };
    // The cloth half of getCombinedFriction: the material table through mMaterialIndices for
    // a triangle, the per-vertex array for a vertex (deformableUtils.cuh:514 vs :533).
    auto laneClothFriction = [&](uint32_t i) {
        return isTriOf[i] ? clothMatFriction[clothMatIdx[elemOf[i]]] : clothVertFriction[elemOf[i]];
    };

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
    kPrep.dispatch(FEM_PREP_GRID, {}, pp, 1u, 1u);
    dev->flush();

    RigidContactConstraintHost refC;
    std::memset(&refC, 0xCD, sizeof(refC));
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t b = bodyOf[i];
        const p714ref::V3 solverNormal = { -planeNormals[i].x, -planeNormals[i].y, -planeNormals[i].z };
        const float pen = normalPens[i].w - contacts[i].w;
        const p714ref::V3 frameP = { body2World[2u * b + 1u].x, body2World[2u * b + 1u].y,
                                     body2World[2u * b + 1u].z };
        p714ref::refPrepare(refC, int(i), solverNormal,
                            p714ref::V3{ positions[i].x, positions[i].y, positions[i].z },
                            pen, laneAccum(i),
                            barycentrics[i], frameP, invMassOf[b], penBiasClampRigid,
                            sqrtInvOf[b], penBiasClampFEM, /*isStatic*/ nodeInds[i] == staticNodeInd);
    }

    const auto gConstraintBytes = downloadBuffer<uint8_t>(dev, bConstraint.h, sizeof(RigidContactConstraintHost));
    RigidContactConstraintHost gotC;
    std::memcpy(&gotC, gConstraintBytes.data(), sizeof(gotC));

    size_t mm = 0;
    auto cmp4 = [&](const char* nm, uint32_t lane, const F4& r, const F4& g) {
        if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) || f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
        { ++mm; std::printf("  prep lane %u %s MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", lane, nm,
                            r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w); }
    };
    for (uint32_t i = 0; i < N; ++i)
    {
        cmp4("raXn_resp", i, refC.raXn_resp[i], gotC.raXn_resp[i]);
        cmp4("raXnF0_resp", i, refC.raXnF0_resp[i], gotC.raXnF0_resp[i]);
        cmp4("raXnF1_resp", i, refC.raXnF1_resp[i], gotC.raXnF1_resp[i]);
        cmp4("normal_errorW", i, refC.normal_errorW[i], gotC.normal_errorW[i]);
        cmp4("fricTan0_invMass0", i, refC.fricTan0_invMass0[i], gotC.fricTan0_invMass0[i]);
        cmp4("barycentric", i, refC.barycentric[i], gotC.barycentric[i]);
        if (f2u(refC.maxPenBias[i]) != f2u(gotC.maxPenBias[i]))
        { ++mm; std::printf("  prep lane %u maxPenBias MM\n", i); }
    }
    PXG_CHECK_MSG(f2u(gotC.fricTan0_invMass0[0].w) == f2u(rigidInvMass),
                  "lane 0 must carry the DYNAMIC inverse mass (got %g)", gotC.fricTan0_invMass0[0].w);
    PXG_CHECK_MSG(f2u(gotC.fricTan0_invMass0[4].w) == 0u,
                  "lane 4 must carry the KINEMATIC zero inverse mass (got %g)", gotC.fricTan0_invMass0[4].w);
    PXG_CHECK_MSG(mm == 0, "dynamic prepare bit-exact vs CPU (%zu mismatches)", mm);

    // The prepare zeroes rigidLambdaNs for every contact, so the accumulated lambda the
    // separating lane needs has to be written back afterwards -- which is exactly the state
    // the engine is in from the second solver iteration onward.
    {
        std::vector<float> seeded(seededLambda, seededLambda + N);
        dev->upload(bLambda.h, seeded.data(), seeded.size() * sizeof(float));
    }

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

    // ---- CPU reference for the query ----
    std::vector<uint32_t> refRefCounts(refCounts.size(), 0u);
    std::vector<float> refVertW(NV, 0.0f);        // per-VERTEX mDeltaPos.w after the query
    for (uint32_t i = 0; i < N; ++i)
    {
        // countReferenceOnly: readCloth returns the RAW inverse masses, without the
        // reference-count multiply the solve applies.
        const p714ref::V3 invMq = isTriOf[i] ? p714ref::V3{ invMass, invMass, invMass }
                                             : p714ref::V3{ invMass, 0.0f, 0.0f };
        const p714ref::InitResult qir =
            p714ref::refInitialize(refC, int(i), isTGS, seededLambda[i], dt, laneBcVec(i),
                                   laneAccum(i), invMq, /*rigidRefCount*/ 1.0f,
                                   /*wasActive*/ separatingOf[i], /*checkOnly*/ true,
                                   rigidVelOf(bodyOf[i]), /*combinedFriction*/ 0.0f);
        // Faithful to the .cu: everything below hangs off isActive.  The assertion is a
        // COVERAGE claim about the seeded data, not a step of the reference -- if it ever
        // fires, the data drifted, and the comparison stays honest either way.
        PXG_CHECK_MSG(qir.isActive, "query CPU ref: lane %u must be active (seeded data)", i);
        if (!qir.isActive)
            continue;
        // The vertex bumps: a triangle bumps each of its three, gated on bc.k > 1e-3.
        uint32_t vs[3];
        const uint32_t nv = laneVerts(i, vs);
        const float bcArr[3] = { barycentrics[i].x, barycentrics[i].y, barycentrics[i].z };
        for (uint32_t k = 0; k < nv; ++k)
            if (!isTriOf[i] || bcArr[k] > 1e-3f)
                refVertW[vs[k]] += 1.0f;
        if (globalIdOf[i] != -1 && invMassOf[bodyOf[i]] != 0.0f)
            refRefCounts[size_t(globalIdOf[i])] += 1u;
    }

    const auto gDeltaQ = downloadBuffer<F4>(dev, bDelta.h, NV);
    const auto gRefCountsQ = downloadBuffer<uint32_t>(dev, bRefCounts.h, uint32_t(refCounts.size()));
    for (uint32_t v = 0; v < NV; ++v)
        if (f2u(gDeltaQ[v].w) != f2u(refVertW[v]))
        { ++mm; std::printf("  query vertex %u mDeltaPos.w MM: ref %g got %g\n", v, refVertW[v], gDeltaQ[v].w); }
    for (size_t k = 0; k < refRefCounts.size(); ++k)
        if (gRefCountsQ[k] != refRefCounts[k])
        { ++mm; std::printf("  query refCounts[%zu] MM: ref %u got %u\n", k, refRefCounts[k], gRefCountsQ[k]); }

    // teeth: seven contacts share solver body 1, so its count must be 7 -- with 1 a dropped
    // reference-count multiply in computeRigidChange would be invisible.
    PXG_CHECK_MSG(gRefCountsQ[1] == 8u, "the eight dynamic contacts must bump body 1 to 8 (got %u)",
                  gRefCountsQ[1]);
    PXG_CHECK_MSG(gRefCountsQ[2] == 0u, "the KINEMATIC body must not be counted (got %u)", gRefCountsQ[2]);
    PXG_CHECK_MSG(gRefCountsQ[numSolverBodies + 1u] == 1u,
                  "the articulation link IS counted, as in the .cu (got %u)",
                  gRefCountsQ[numSolverBodies + 1u]);
    PXG_CHECK_MSG(mm == 0, "dynamic query bit-exact vs CPU (%zu mismatches)", mm);

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
    sp.materials = bClothMats.addr; sp.rigidBodyMaterials = bRigidMats.addr;

    const char* solveEntry = isTGS ? "cloth_solveRigidClothCollisionTGSLaunch"
                                   : "cloth_solveRigidClothCollisionLaunch";
    Kernel kSolve("femClothRigidContactBda.spv", solveEntry, {}, sizeof(SolvePush));
    kSolve.dispatch(1u, {}, sp, 1u, 1u);
    dev->flush();

    // ---- CPU reference for the solve ----
    std::vector<p714ref::V3> refVertDelta(NV, p714ref::V3{ 0.0f, 0.0f, 0.0f });
    std::vector<F4> refRigidDeltaVel(2u * N);
    std::vector<float> refAccum(N);
    std::vector<bool> reactsOf(N, false);
    uint32_t dynamicLanes = 0u, zeroLanes = 0u;
    uint32_t frictionLanes = 0u, boundLanes = 0u, boundTriLanes = 0u, boundVertLanes = 0u;
    uint32_t negativeLambdaLanes = 0u;
    uint32_t modeSeen[5] = { 0u, 0u, 0u, 0u, 0u };
    std::vector<float> laneFriction(N, 0.0f);
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t b = bodyOf[i];
        // readRigidBody: the count is read back only for a non-static, finite-mass body.
        const float rigidRefCount =
            (globalIdOf[i] != -1 && invMassOf[b] != 0.0f) ? float(refRefCounts[size_t(globalIdOf[i])]) : 1.0f;
        // readCloth's mass split: each of the lane's vertices carries invMass times ITS OWN
        // reference count.
        uint32_t vs[3];
        const uint32_t nv = laneVerts(i, vs);
        p714ref::V3 deformableInvMasses = { 0.0f, 0.0f, 0.0f };
        {
            float im[3] = { 0.0f, 0.0f, 0.0f };
            for (uint32_t k = 0; k < nv; ++k) im[k] = invMass * refVertW[vs[k]];
            deformableInvMasses = { im[0], im[1], im[2] };
        }
        // P9.46: getCombinedFriction() = PxsCombinePxReal(rigid, cloth, the RIGID material's
        // mode).  The cloth half comes from a DIFFERENT source per contact kind, which is
        // what lanes 8 and 9 are here to prove.
        const float combinedFriction = p714ref::refCombineFriction(
            rigidMatFriction[rigidMatOf[i]], laneClothFriction(i), rigidMatMode[rigidMatOf[i]]);
        laneFriction[i] = combinedFriction;
        // modeSeen[4] counts the out-of-range mode, whatever its numeric value.
        ++modeSeen[rigidMatMode[rigidMatOf[i]] < 4u ? rigidMatMode[rigidMatOf[i]] : 4u];
        const p714ref::InitResult sir =
            p714ref::refInitialize(refC, int(i), isTGS, seededLambda[i], dt, laneBcVec(i),
                                   laneAccum(i), deformableInvMasses, rigidRefCount,
                                   /*wasActive*/ true, /*checkOnly*/ false, rigidVelOf(b),
                                   combinedFriction);
        if (sir.deltaLambdaN < 0.0f)
            ++negativeLambdaLanes;
        if (sir.deltaLambdaT != 0.0f)
            ++frictionLanes;
        if (sir.frictionBound)
        {
            ++boundLanes;
            if (isTriOf[i]) ++boundTriLanes; else ++boundVertLanes;
        }
        const p714ref::V3 deltaPosR = {
            -(sir.deltaLambdaN * sir.normal.x + sir.deltaLambdaT * sir.tangent.x) * dt,
            -(sir.deltaLambdaN * sir.normal.y + sir.deltaLambdaT * sir.tangent.y) * dt,
            -(sir.deltaLambdaN * sir.normal.z + sir.deltaLambdaT * sir.tangent.z) * dt };
        // writeCloth: a vertex contact scales by its own inverse mass, a triangle contact by
        // invM[k] * bc[k] per vertex, each guarded on invM[k] > 0.
        {
            const float imArr[3] = { deformableInvMasses.x, deformableInvMasses.y, deformableInvMasses.z };
            const float bcArr[3] = { barycentrics[i].x, barycentrics[i].y, barycentrics[i].z };
            for (uint32_t k = 0; k < nv; ++k)
            {
                if (!(imArr[k] > 0.0f))
                    continue;
                const float sc = isTriOf[i] ? (imArr[k] * bcArr[k]) : imArr[k];
                refVertDelta[vs[k]].x += deltaPosR.x * sc;
                refVertDelta[vs[k]].y += deltaPosR.y * sc;
                refVertDelta[vs[k]].z += deltaPosR.z * sc;
            }
        }
        refAccum[i] = sir.accum;

        // The .cu's guard, plus this port's articulation extension.
        const bool isArti = (nodeInds[i] >> 32) & 1ull;
        const bool reacts = (invMassOf[b] != 0.0f) && (nodeInds[i] != staticNodeInd) && !isArti;
        reactsOf[i] = reacts;
        p714ref::V3 dLin = { 0, 0, 0 }, dAng = { 0, 0, 0 };
        float count = 0.0f;
        if (reacts)
        {
            count = 1.0f;
            p714ref::refComputeRigidChange(sir, /*isArticulation*/ false, invMassOf[b], rigidRefCount,
                                           dLin, dAng);
            ++dynamicLanes;
        }
        else
        {
            ++zeroLanes;
        }
        refRigidDeltaVel[i] = { dLin.x, dLin.y, dLin.z, count };
        refRigidDeltaVel[i + N] = { dAng.x, dAng.y, dAng.z, 0.0f };
    }
    PXG_CHECK_MSG(dynamicLanes == 8u && zeroLanes == 3u,
                  "coverage: expected 8 reacting and 3 silent lanes, got %u / %u", dynamicLanes, zeroLanes);

    // P9.46 TEETH.  Without these the material path could be silently inert again:
    // deltaLambdaT is exactly -0 whenever the combined friction is 0, and a comparison
    // against a reference that ALSO produced -0 stays green either way.
    // N - 1, not N: lane 6 carries the OUT-OF-RANGE combine mode, whose whole point is that
    // PxsCombinePxReal returns 0 there and the tangent impulse collapses to -0.
    PXG_CHECK_MSG(frictionLanes == N - 1u,
                  "all but the invalid-mode lane must produce a non-zero tangent impulse, "
                  "got %u of %u", frictionLanes, N);
    PXG_CHECK_MSG(f2u(laneFriction[6]) == 0u,
                  "the out-of-range combine mode must give exactly 0 friction (got %g)",
                  laneFriction[6]);
    // The abs() in the tangent clamp is only observable where deltaLambdaN goes negative.
    PXG_CHECK_MSG(negativeLambdaLanes >= 1u,
                  "at least one lane must reach a NEGATIVE deltaLambdaN (separating contact "
                  "with an accumulated lambda), got %u", negativeLambdaLanes);
    for (uint32_t m = 0; m < 4u; ++m)
        PXG_CHECK_MSG(modeSeen[m] > 0u, "PxCombineMode %u is never exercised", m);
    PXG_CHECK_MSG(modeSeen[4] > 0u, "the out-of-range combine mode is never exercised");
    {
        uint32_t distinct = 0u;
        for (uint32_t a = 0; a < N; ++a)
        {
            bool dup = false;
            for (uint32_t b2 = 0; b2 < a; ++b2) if (laneFriction[b2] == laneFriction[a]) dup = true;
            if (!dup) ++distinct;
        }
        PXG_CHECK_MSG(distinct >= 6u, "the seeded frictions must take at least 6 distinct "
                                      "values across the lanes, got %u", distinct);
        // BOTH cloth friction sources must be reached.  A triangle lane's vertices carry
        // -7.5 in mDynamicFrictions, so a port that read the per-vertex array for a triangle
        // contact would produce a NEGATIVE combined friction and this comparison would go
        // red -- but only if a triangle lane exists at all, which is what this pins.
        uint32_t triLanes = 0u, vertLanes = 0u;
        for (uint32_t a = 0; a < N; ++a) { if (isTriOf[a]) ++triLanes; else ++vertLanes; }
        PXG_CHECK_MSG(triLanes >= 2u && vertLanes >= 6u,
                      "coverage: %u TRIANGLE and %u VERTEX lanes -- both friction sources "
                      "must be live", triLanes, vertLanes);
        // AND the friction must actually BIND the clamp on both kinds.  deltaLambdaT is
        // -min(dLT, mu*|dLN|); where the raw magnitude is the smaller term the material
        // value never reaches the output, and a mutant of the read survives.  That is not a
        // hypothetical: the first version of this fixture bound on no triangle lane at all,
        // and every one of the six deformable-material mutants lived through the sweep.
        PXG_CHECK_MSG(boundTriLanes >= 2u && boundVertLanes >= 3u,
                      "the friction must BIND the tangent clamp on both contact kinds: "
                      "%u triangle and %u vertex lanes bound (of %u total)",
                      boundTriLanes, boundVertLanes, boundLanes);
        std::printf("  friction per lane:");
        for (uint32_t a = 0; a < N; ++a) std::printf(" %.4g", laneFriction[a]);
        std::printf("\n");
    }

    const auto gDelta = downloadBuffer<F4>(dev, bDelta.h, NV);
    const auto gLambda2 = downloadBuffer<float>(dev, bLambda.h, N);
    const auto gRigidDeltaVel = downloadBuffer<F4>(dev, bRigidDeltaVel.h, 2u * N);

    size_t mm2 = 0;
    auto cmpRow = [&](const char* nm, uint32_t k, const F4& r, const F4& g) {
        if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) || f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
        { ++mm2; std::printf("  solve %s[%u] MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", nm, k,
                             r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w); }
    };
    // Per VERTEX, because the two triangle lanes each scatter onto three of them.  Every
    // vertex still has exactly ONE writing lane, so this stays bit-exact.
    for (uint32_t v = 0; v < NV; ++v)
        cmpRow("mDeltaPos", v,
               F4{ refVertDelta[v].x, refVertDelta[v].y, refVertDelta[v].z, refVertW[v] }, gDelta[v]);
    for (uint32_t i = 0; i < N; ++i)
    {
        cmpRow("rigidDeltaVel", i, refRigidDeltaVel[i], gRigidDeltaVel[i]);
        cmpRow("rigidDeltaVel", i + N, refRigidDeltaVel[i + N], gRigidDeltaVel[i + N]);
        if (f2u(gLambda2[i]) != f2u(refAccum[i]))
        { ++mm2; std::printf("  solve appliedForces[%u] MM: ref %g got %g\n", i, refAccum[i], gLambda2[i]); }
    }

    // teeth 1: every reacting lane must produce a REACTION, and it must point INTO the
    // floor -- the cloth is pushed out along the plane normal, the body the other way.
    for (uint32_t i = 0; i < N; ++i)
    {
        if (!reactsOf[i])
            continue;
        PXG_CHECK_MSG(f2u(gRigidDeltaVel[i].w) == f2u(1.0f),
                      "dynamic lane %u must report count 1 (got %g)", i, gRigidDeltaVel[i].w);
        const float lin = gRigidDeltaVel[i].x * planeNormals[i].x + gRigidDeltaVel[i].y * planeNormals[i].y
                        + gRigidDeltaVel[i].z * planeNormals[i].z;
        // Lane 10 is the SEPARATING lane: it unwinds an already-accumulated lambda, so its
        // deltaLambdaN is negative and the reaction points the OTHER way -- the sign flip is
        // the whole reason that lane exists, so it is asserted, not exempted.
        if (separatingOf[i])
            PXG_CHECK_MSG(lin > 0.0f,
                          "separating lane %u must UNWIND, i.e. react along the normal "
                          "(dot = %g, isTGS=%d)", i, lin, int(isTGS));
        else
            PXG_CHECK_MSG(lin < 0.0f,
                          "dynamic lane %u reaction must oppose the cloth push (dot = %g, isTGS=%d)",
                          i, lin, int(isTGS));
        const F4 a = gRigidDeltaVel[i + N];
        PXG_CHECK_MSG(f2u(a.x) != 0u || f2u(a.y) != 0u || f2u(a.z) != 0u,
                      "dynamic lane %u angular reaction must be non-zero", i);
    }
    // teeth 2: every silent lane stays exactly zero, count included.
    for (uint32_t i = 0; i < N; ++i)
    {
        if (reactsOf[i])
            continue;
        for (uint32_t half = 0; half < 2u; ++half)
        {
            const F4 v = gRigidDeltaVel[i + half * N];
            PXG_CHECK_MSG(f2u(v.x) == 0u && f2u(v.y) == 0u && f2u(v.z) == 0u && f2u(v.w) == 0u,
                          "silent lane %u half %u must stay zero (got %g,%g,%g,%g)",
                          i, half, v.x, v.y, v.z, v.w);
        }
    }
    // teeth 3: the two basis branches are both reached, and the boundary lane sits exactly
    // on the literal -- otherwise a mutant of either normalisation, or of <= vs <, survives.
    {
        uint32_t firstBranch = 0u, secondBranch = 0u;
        for (uint32_t i = 0; i < N; ++i)
        {
            const float dy = -planeNormals[i].y;   // the SOLVER normal's y
            if (std::fabs(dy) <= 0.9999f) ++firstBranch; else ++secondBranch;
        }
        PXG_CHECK_MSG(firstBranch >= 6u && secondBranch >= 1u,
                      "computeBasisVectors coverage: %u lanes on the first branch, %u on the second",
                      firstBranch, secondBranch);
        PXG_CHECK_MSG(f2u(std::fabs(-planeNormals[7].y)) == f2u(0.9999f),
                      "lane 7 must sit EXACTLY on the 0.9999 boundary (got %.9g)",
                      double(std::fabs(-planeNormals[7].y)));
    }
    PXG_CHECK_MSG(mm2 == 0, "dynamic cloth_solveRigidClothCollision%s bit-exact vs CPU (%zu mismatches)",
                  isTGS ? "TGS" : "PGS", mm2);

    // ================= second solve: the NULL rigid material table ==============
    // The .cu passes `&rigidBodyMaterials[contactInfo.getRigidMaterialIndex()]`, so the
    // pointer readRigidBody tests is NULL only when the table is null AND the index is 0 --
    // which is why the port guards on `base + index * stride`, not on the base.  On that
    // path the rigid half is friction 0 with combine mode eMAX, so the combined friction is
    // max(0, deformableFriction) = the DEFORMABLE value, unhalved.
    //
    // The old single-lane static fixture already ran with both tables null, but there the
    // deformable friction was 0 too, and every combine mode returns 0 from (0, 0) -- the
    // eMAX default was unobservable.  Here `materials` stays live and the VERTEX lanes read
    // cloth.mDynamicFrictions, which readCloth reaches without the material table at all,
    // so the default mode finally reaches the output.
    // Pass 2 repeats with the DEFORMABLE table null as well.  There the .cu's
    // `materials ? ... : 0.0f` is the only thing standing between the port and a read off
    // address 8, and both readCloth branches carry that guard -- the vertex branch guards
    // the mDynamicFrictions read on `materials` too, which is easy to miss because the
    // array it reads does not come from that pointer at all.
    for (uint32_t pass = 0; pass < 2u; ++pass)
    {
        const bool nullFemMats = (pass == 1u);
        auto infosNull = downloadBuffer<FemOtherContactInfoHost>(dev, bInfos.h, N);
        for (uint32_t i = 0; i < N; ++i)
            infosNull[i].rigidMatInd_isInCollision &= 0x80000000u;   // index 0, activity kept
        dev->upload(bInfos.h, infosNull.data(), infosNull.size() * sizeof(FemOtherContactInfoHost));

        // Restore the exact post-QUERY state the first solve started from.
        {
            std::vector<F4> d(NV);
            for (uint32_t v = 0; v < NV; ++v) d[v] = F4{ 0.0f, 0.0f, 0.0f, refVertW[v] };
            dev->upload(bDelta.h, d.data(), d.size() * sizeof(F4));
            std::vector<float> seeded(seededLambda, seededLambda + N);
            dev->upload(bLambda.h, seeded.data(), seeded.size() * sizeof(float));
            std::vector<F4> z(2u * N, F4{ 0.0f, 0.0f, 0.0f, 0.0f });
            dev->upload(bRigidDeltaVel.h, z.data(), z.size() * sizeof(F4));
        }

        SolvePush spn = sp;
        spn.rigidBodyMaterials = 0ull;
        if (nullFemMats)
        {
            spn.materials = 0ull;
            // With the table null the .cu never dereferences it, so the per-triangle index
            // is free -- and a port that dropped the guard would compute
            // 0 + 65535 * 32 + 8 and fault.  That is the ONLY observable a dropped null
            // guard has: with a small index the wild read lands in mapped memory and reads
            // back 0, which is exactly what the guard would have produced.
            const uint32_t poison = 65535u | (65535u << 16);
            dev->upload(bClothMatIdx.h, &poison, sizeof(poison));
        }
        Kernel kSolveNull("femClothRigidContactBda.spv", solveEntry, {}, sizeof(SolvePush));
        kSolveNull.dispatch(1u, {}, spn, 1u, 1u);
        dev->flush();

        std::vector<F4> refNullVel(2u * N);
        std::vector<float> refNullAccum(N);
        uint32_t nullBound = 0u, nullHalved = 0u;
        for (uint32_t i = 0; i < N; ++i)
        {
            const uint32_t b = bodyOf[i];
            const float rigidRefCount =
                (globalIdOf[i] != -1 && invMassOf[b] != 0.0f) ? float(refRefCounts[size_t(globalIdOf[i])]) : 1.0f;
            uint32_t vs[3];
            const uint32_t nv = laneVerts(i, vs);
            float im[3] = { 0.0f, 0.0f, 0.0f };
            for (uint32_t k = 0; k < nv; ++k) im[k] = invMass * refVertW[vs[k]];
            const p714ref::V3 dim = { im[0], im[1], im[2] };
            // rigid friction 0, mode eMAX -- the null default, spelled out.  With the
            // deformable table null BOTH readCloth branches return 0, vertex included.
            const float mu = p714ref::refCombineFriction(
                0.0f, nullFemMats ? 0.0f : laneClothFriction(i), 3u);
            const p714ref::InitResult sir =
                p714ref::refInitialize(refC, int(i), isTGS, seededLambda[i], dt, laneBcVec(i),
                                       laneAccum(i), dim, rigidRefCount, /*wasActive*/ true,
                                       /*checkOnly*/ false, rigidVelOf(b), mu);
            if (sir.frictionBound) ++nullBound;
            // The mutant that defaults to eAVERAGE would halve mu here; count the lanes on
            // which that is actually observable, i.e. where the clamp binds on a non-zero mu.
            if (sir.frictionBound && mu != 0.0f) ++nullHalved;
            const bool isArti = (nodeInds[i] >> 32) & 1ull;
            const bool reacts = (invMassOf[b] != 0.0f) && (nodeInds[i] != staticNodeInd) && !isArti;
            p714ref::V3 dLin = { 0, 0, 0 }, dAng = { 0, 0, 0 };
            float count = 0.0f;
            if (reacts)
            {
                count = 1.0f;
                p714ref::refComputeRigidChange(sir, /*isArticulation*/ false, invMassOf[b],
                                               rigidRefCount, dLin, dAng);
            }
            refNullVel[i] = { dLin.x, dLin.y, dLin.z, count };
            refNullVel[i + N] = { dAng.x, dAng.y, dAng.z, 0.0f };
            refNullAccum[i] = sir.accum;
        }
        // Pass 1 must bind, or eMAX and eAVERAGE are indistinguishable.  Pass 2 cannot
        // bind at all -- every friction is 0 there, which is exactly the point.
        if (!nullFemMats)
            PXG_CHECK_MSG(nullHalved >= 2u,
                          "the null-rigid-material pass must BIND the clamp on at least two "
                          "lanes, else eMAX and eAVERAGE are indistinguishable there "
                          "(bound %u)", nullHalved);
        else
            PXG_CHECK_MSG(nullHalved == 0u,
                          "with both material tables null every friction must be 0 (%u bound)",
                          nullHalved);

        const auto gNullVel = downloadBuffer<F4>(dev, bRigidDeltaVel.h, 2u * N);
        const auto gNullAccum = downloadBuffer<float>(dev, bLambda.h, N);
        size_t mmn = 0;
        for (uint32_t k = 0; k < 2u * N; ++k)
        {
            const F4& r = refNullVel[k]; const F4& g = gNullVel[k];
            if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) || f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
            { ++mmn; std::printf("  null-mat pass %u rigidDeltaVel[%u] MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n",
                                 pass, k, r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w); }
        }
        for (uint32_t i = 0; i < N; ++i)
            if (f2u(refNullAccum[i]) != f2u(gNullAccum[i]))
            { ++mmn; std::printf("  null-mat pass %u appliedForces[%u] MM: ref %g got %g\n",
                                 pass, i, refNullAccum[i], gNullAccum[i]); }
        PXG_CHECK_MSG(mmn == 0, "null material pass %u bit-exact vs CPU (%zu mismatches)", pass, mmn);
        if (nullFemMats)
            dev->upload(bClothMatIdx.h, clothMatIdxWords.data(), clothMatIdxWords.size() * sizeof(uint32_t));
    }

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bAcc.h);
    dev->destroyBuffer(bDelta.h); dev->destroyBuffer(bTri.h); dev->destroyBuffer(bDesc.h);
    dev->destroyBuffer(bContacts.h); dev->destroyBuffer(bNormalPens.h); dev->destroyBuffer(bBary.h);
    dev->destroyBuffer(bInfos.h); dev->destroyBuffer(bNum.h); dev->destroyBuffer(bConstraint.h);
    dev->destroyBuffer(bLambda.h); dev->destroyBuffer(bRigidDeltaVel.h); dev->destroyBuffer(bRefCounts.h);
    dev->destroyBuffer(bBody2World.h); dev->destroyBuffer(bSolverBodyData.h); dev->destroyBuffer(bSolverTxIData.h);
    dev->destroyBuffer(bOutSolverVel.h); dev->destroyBuffer(bSolverBodyVelPool.h);
    dev->destroyBuffer(bSolverBodyIndices.h);
    dev->destroyBuffer(bPrepDesc.h); dev->destroyBuffer(bPrePrepDesc.h);
    dev->destroyBuffer(bSolverCoreDesc.h); dev->destroyBuffer(bSharedDesc.h);
    dev->destroyBuffer(bClothFric.h); dev->destroyBuffer(bRigidMats.h); dev->destroyBuffer(bClothMats.h);
}

PXG_TEST(femClothSimRigidContactDynamicPGS) { runRigidContactDynamic(/*isTGS*/ false); }
PXG_TEST(femClothSimRigidContactDynamicTGS) { runRigidContactDynamic(/*isTGS*/ true); }

// ---------------------------------------------------------------------------
// P9.86c -- the deltaLambdaN PxMax zero tie (pxgSlangFemRigidContact.slang contactInitialize,
// CUDA FEMCollision::initialize, deformableUtils.cuh:374):
//
//   deltaLambdaN = PxMax(-CN * invDenom, -accumulatedDeltaLambdaN)
//
// PxMax(float) is intrinsics::selectMax, `a > b ? a : b` (PxUnixMathIntrinsics.h:122): on the
// tie (a, b) = (+0, -0) it keeps b = -0.  The pre-fix port spelled it max(), GLSL FMax, which
// llvmpipe answers with +0 (F4).  For a DYNAMIC body invDenom > 0, so a = +0 needs CN == -0
// exactly, and b = -0 is the ordinary appliedForces = +0.  That is a cloth vertex RESTING
// exactly on a box (pen = -0, zero relative normal motion) which slides in +x, with the
// contact already in collision from an earlier iteration.  The tie alone is absorbed
// (appliedForces = +0 + +-0, |dLN| in the friction clamp, mDeltaPos is a +0-seeded atomic
// add); it surfaces in computeRigidChange, where deltaLambdaT = -min(dLT, mu*|dLN|) = -0 on
// the live tangent branch and every n_i*dLN + t_i*(-0) / raXn_i*dLN + raXt_i*(-0) with a
// sign-clear second factor carries the sign of dLN into rigidDeltaVel:
//   CUDA / HEAD : rigidDeltaVel[0].xyz = (+0,+0,+0), [1].xyz = (-0,-0,+0)
//   pre-fix port: rigidDeltaVel[0].xyz = (-0,-0,-0), [1].xyz = (+0,+0,+0)
//
// Every zero sign is chosen on purpose:
//   plane normal (0,1,0), pen w = -0          -> solver normal (-0,-1,-0), initPen = -0
//   body 1 frame p = contact p - (0,0.5,0)    -> ra = (+0,0.5,+0), raXn = (+0,+0,+0)
//   diagonal sqrtInvInertia                   -> raXnF0 = (+0,+0,-0.6), raXnF1 = (0.75,+0,+0)
//   PGS linVel = (+0,+0,+0) + (0.5,+0,+0), angVel = (-0)+(-0)
//   TGS linVel = (0.5,+0,+0), angVel = linDelta = angDelta = (-0,-0,-0)
// and the cloth vertex's accumulated delta is (+0,+0,+0) at the prepare but (-0,-0,-0) at the
// solve (a later iteration: a zero reached through a negation).  BOTH modes need that
// re-upload: d.n at the prepare and (-d).n (PGS) or (linDelta - d).n (TGS, with linDelta
// = -0) at the solve cannot both be -0 for one unchanged d, and with d = +0 at the solve the
// solve error is +0, CN = +0, a = -0 and there is no tie.
// ---------------------------------------------------------------------------
static void runRigidContactZeroSign(bool isTGS)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 1.0f / 60.0f;
    const uint32_t N = 1u, NV = 1u;
    const uint32_t numSolverBodies = 2u;
    const uint32_t accOffset = 0u;
    const float rigidInvMass = 0.25f;
    const float invMass = 0.5f;

    std::vector<F4> positions = { F4{ 0.25f, 0.5f, -0.75f, invMass } };
    std::vector<F4> velocities = { F4{ 0.0f, 0.0f, 0.0f, invMass } };
    std::vector<F4> accumDelta = { F4{ 0.0f, 0.0f, 0.0f, invMass } };
    std::vector<F4> deltaPos = { F4{ 0.0f, 0.0f, 0.0f, 0.0f } };
    std::vector<uint32_t> triVtx(4u, 0u);
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
    c.mNbVerts = NV;
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    // node 1 -> solver body 1 (dynamic).
    std::vector<uint32_t> solverBodyIndices = { 0u, 1u };
    DevBuf bSolverBodyIndices = makeDev(dev, solverBodyIndices);

    const F4 planeNormal = { 0.0f, 1.0f, 0.0f, -0.0f };   // pen w = -0: exactly touching
    std::vector<F4> contacts = { F4{ 0.25f, 0.5f, -0.75f, 0.0f } };
    std::vector<F4> normalPens = { planeNormal };
    std::vector<F4> barycentrics = { F4{ 0.0f, 0.0f, 0.0f, 1.0f } };   // VERTEX contact
    std::vector<FemOtherContactInfoHost> infos(N);
    infos[0].pairInd0 = 1ull;
    infos[0].pairInd1 = 0u;
    infos[0].rigidMatInd_isInCollision = 0x80000000u;   // in collision since an earlier iteration
    DevBuf bContacts = makeDev(dev, contacts);
    DevBuf bNormalPens = makeDev(dev, normalPens);
    DevBuf bBary = makeDev(dev, barycentrics);
    DevBuf bInfos = makeDev(dev, infos);
    std::vector<uint32_t> numContacts = { N };
    DevBuf bNum = makeDev(dev, numContacts);

    std::vector<uint8_t> constraintBytes(sizeof(RigidContactConstraintHost), 0xCD);
    DevBuf bConstraint = makeDev(dev, constraintBytes);
    std::vector<float> rigidLambdaNs(N, 123.0f);
    DevBuf bLambda = makeDev(dev, rigidLambdaNs);
    std::vector<F4> rigidDeltaVel(2u * N, F4{ 9.f, 9.f, 9.f, 9.f });
    DevBuf bRigidDeltaVel = makeDev(dev, rigidDeltaVel);
    std::vector<uint32_t> refCounts(numSolverBodies + 2u, 0u);
    DevBuf bRefCounts = makeDev(dev, refCounts);

    std::vector<F4> body2World = {
        { 0.f, 0.f, 0.f, 1.f }, { 0.0f, 0.0f, 0.0f, 0.f },     // body 0 (static slot)
        { 0.f, 0.f, 0.f, 1.f }, { 0.25f, 0.0f, -0.75f, 0.f },  // body 1: ra = (+0, 0.5, +0)
    };
    DevBuf bBody2World = makeDev(dev, body2World);

    const float penBiasClampRigid = -1.0e10f;
    const float invMassOf[2] = { 0.0f, rigidInvMass };
    std::vector<uint8_t> solverBodyData(96u * numSolverBodies, 0);
    for (uint32_t b = 0; b < numSolverBodies; ++b)
    {
        std::memcpy(solverBodyData.data() + 96u * b + 12, &penBiasClampRigid, 4);
        std::memcpy(solverBodyData.data() + 96u * b + 28, &invMassOf[b], 4);
    }
    DevBuf bSolverBodyData = makeDev(dev, solverBodyData);
    // Diagonal (axis-aligned) sqrt inverse inertia, +0 off the diagonal.
    const float sqrtInvInertia1[9] = { 1.5f, 0.0f, 0.0f, 0.0f, 0.8f, 0.0f, 0.0f, 0.0f, 1.2f };
    std::vector<uint8_t> solverTxIData(64u * numSolverBodies, 0);
    std::memcpy(solverTxIData.data() + 64u * 1u + 28, sqrtInvInertia1, 9 * 4);
    DevBuf bSolverTxIData = makeDev(dev, solverTxIData);

    // PGS: initial (+0) plus iteration delta; TGS reads the pool rows alone.
    std::vector<F4> outSolverVelocity(2u * numSolverBodies, F4{ 0.f, 0.f, 0.f, 0.f });
    outSolverVelocity[1] = { 0.0f, 0.0f, 0.0f, 0.f };
    outSolverVelocity[1 + numSolverBodies] = { -0.0f, -0.0f, -0.0f, 0.f };
    DevBuf bOutSolverVel = makeDev(dev, outSolverVelocity);
    std::vector<F4> solverBodyVelPool(3u * numSolverBodies, F4{ 0.f, 0.f, 0.f, 0.f });
    solverBodyVelPool[1] = { 0.5f, 0.0f, 0.0f, -0.0f };
    solverBodyVelPool[1 + numSolverBodies] = { -0.0f, -0.0f, -0.0f, -0.0f };
    solverBodyVelPool[1 + 2u * numSolverBodies] = { -0.0f, -0.0f, -0.0f, -0.0f };
    DevBuf bSolverBodyVelPool = makeDev(dev, solverBodyVelPool);

    p714ref::RefRigidVel rv;
    if (isTGS)
    {
        const F4 v0 = solverBodyVelPool[accOffset + 1u];
        const F4 v1 = solverBodyVelPool[accOffset + 1u + numSolverBodies];
        const F4 v2 = solverBodyVelPool[accOffset + 1u + 2u * numSolverBodies];
        rv.linVel = { v0.x, v0.y, v0.z };
        rv.angVel = { v0.w, v1.x, v1.y };
        rv.linDelta = { v1.z, v1.w, v2.x };
        rv.angDelta = { v2.y, v2.z, v2.w };
    }
    else
    {
        const F4 iv0 = outSolverVelocity[1], iv1 = outSolverVelocity[1 + numSolverBodies];
        const F4 dv0 = solverBodyVelPool[accOffset + 1u], dv1 = solverBodyVelPool[accOffset + 1u + numSolverBodies];
        rv.linVel = { iv0.x + dv0.x, iv0.y + dv0.y, iv0.z + dv0.z };
        rv.angVel = { iv1.x + dv1.x, iv1.y + dv1.y, iv1.z + dv1.z };
    }

    std::vector<uint8_t> prepDesc(304, 0);
    pokePtr(prepDesc, 64, bSolverBodyData.addr);
    pokePtr(prepDesc, 72, bSolverTxIData.addr);
    pokePtr(prepDesc, 168, bBody2World.addr);
    DevBuf bPrepDesc = makeDev(dev, prepDesc);
    std::vector<uint8_t> prePrepDesc(760, 0);
    pokePtr(prePrepDesc, 272, bSolverBodyIndices.addr);
    DevBuf bPrePrepDesc = makeDev(dev, prePrepDesc);
    std::vector<uint8_t> solverCoreDesc(320, 0);
    pokePtr(solverCoreDesc, 0, bOutSolverVel.addr);
    pokeU32(solverCoreDesc, 268, numSolverBodies);
    pokeU32(solverCoreDesc, 276, accOffset);
    DevBuf bSolverCoreDesc = makeDev(dev, solverCoreDesc);
    std::vector<uint8_t> sharedDesc(224, 0);
    pokePtr(sharedDesc, 160, bSolverBodyVelPool.addr);
    DevBuf bSharedDesc = makeDev(dev, sharedDesc);

    // ======================= PREPARE ==========================
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
    kPrep.dispatch(FEM_PREP_GRID, {}, pp, 1u, 1u);
    dev->flush();

    RigidContactConstraintHost refC;
    std::memset(&refC, 0xCD, sizeof(refC));
    const p714ref::V3 solverNormal = { -planeNormal.x, -planeNormal.y, -planeNormal.z };
    p714ref::refPrepare(refC, 0, solverNormal, p714ref::V3{ positions[0].x, positions[0].y, positions[0].z },
                        planeNormal.w - contacts[0].w,
                        p714ref::V3{ accumDelta[0].x, accumDelta[0].y, accumDelta[0].z }, barycentrics[0],
                        p714ref::V3{ body2World[3].x, body2World[3].y, body2World[3].z }, rigidInvMass,
                        penBiasClampRigid, sqrtInvInertia1, penBiasClampFEM, /*isStatic*/ false);
    const auto gConstraintBytes = downloadBuffer<uint8_t>(dev, bConstraint.h, sizeof(RigidContactConstraintHost));
    RigidContactConstraintHost gotC;
    std::memcpy(&gotC, gConstraintBytes.data(), sizeof(gotC));
    size_t mm = 0;
    auto cmp4 = [&](const char* nm, const F4& r, const F4& g) {
        if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) || f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
        { ++mm; std::printf("  zero-sign %s MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", nm,
                            r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w); }
    };
    cmp4("raXn_resp", refC.raXn_resp[0], gotC.raXn_resp[0]);
    cmp4("raXnF0_resp", refC.raXnF0_resp[0], gotC.raXnF0_resp[0]);
    cmp4("raXnF1_resp", refC.raXnF1_resp[0], gotC.raXnF1_resp[0]);
    cmp4("normal_errorW", refC.normal_errorW[0], gotC.normal_errorW[0]);
    cmp4("fricTan0_invMass0", refC.fricTan0_invMass0[0], gotC.fricTan0_invMass0[0]);
    cmp4("barycentric", refC.barycentric[0], gotC.barycentric[0]);
    if (f2u(refC.maxPenBias[0]) != f2u(gotC.maxPenBias[0])) { ++mm; std::printf("  zero-sign maxPenBias MM\n"); }
    PXG_CHECK_MSG(mm == 0, "zero-sign prepare bit-exact vs CPU (%zu mismatches)", mm);
    PXG_CHECK_MSG(f2u(gotC.normal_errorW[0].w) == 0x80000000u,
                  "coverage: the resting contact must prepare initPen = -0 (got 0x%08x)",
                  f2u(gotC.normal_errorW[0].w));
    PXG_CHECK_MSG(f2u(gotC.raXn_resp[0].x) == 0u && f2u(gotC.raXn_resp[0].y) == 0u
                      && f2u(gotC.raXn_resp[0].z) == 0u,
                  "coverage: raXn must be (+0,+0,+0)");

    // A later iteration: the vertex's accumulated delta is a zero reached by negation.
    const std::vector<F4> accumSolve = { F4{ -0.0f, -0.0f, -0.0f, invMass } };
    dev->upload(bAcc.h, accumSolve.data(), accumSolve.size() * sizeof(F4));
    const p714ref::V3 accumS = { accumSolve[0].x, accumSolve[0].y, accumSolve[0].z };

    // ======================= QUERY (refcount) =================
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
    Kernel kQuery("femClothRigidContactBda.spv",
                  isTGS ? "cloth_queryRigidClothContactReferenceCountTGSLaunch"
                        : "cloth_queryRigidClothContactReferenceCountLaunch", {}, sizeof(QueryPush));
    kQuery.dispatch(1u, {}, qp, 1u, 1u);
    dev->flush();
    const auto gDeltaQ = downloadBuffer<F4>(dev, bDelta.h, NV);
    const auto gRefCountsQ = downloadBuffer<uint32_t>(dev, bRefCounts.h, uint32_t(refCounts.size()));
    const auto gInfosQ = downloadBuffer<FemOtherContactInfoHost>(dev, bInfos.h, N);
    PXG_CHECK_MSG(f2u(gDeltaQ[0].w) == f2u(1.0f), "query must bump the vertex count to 1 (got %g)", gDeltaQ[0].w);
    PXG_CHECK_MSG(gRefCountsQ[1] == 1u, "query must count the dynamic body once (got %u)", gRefCountsQ[1]);
    PXG_CHECK_MSG((gInfosQ[0].rigidMatInd_isInCollision & 0x80000000u) != 0u, "contact must stay in collision");

    // ======================= SOLVE (null material tables: mu = 0) =====
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
    sp.materials = 0ull; sp.rigidBodyMaterials = 0ull;
    Kernel kSolve("femClothRigidContactBda.spv",
                  isTGS ? "cloth_solveRigidClothCollisionTGSLaunch" : "cloth_solveRigidClothCollisionLaunch",
                  {}, sizeof(SolvePush));
    kSolve.dispatch(1u, {}, sp, 1u, 1u);
    dev->flush();

    // ---- reference image, CUDA semantics (variant 0) or the pre-fix port (variant 1) ----
    struct Image { F4 lin, ang, vert; float accum; };
    p714ref::InitResult sir0{};
    auto image = [&](int variant, p714ref::InitResult* out) {
        p714ref::gDlnVariant = variant;
        const p714ref::InitResult sir =
            p714ref::refInitialize(refC, 0, isTGS, /*appliedForceRef*/ 0.0f, dt, p714ref::V3{ 1.0f, 0.0f, 0.0f },
                                   accumS, p714ref::V3{ invMass * 1.0f, 0.0f, 0.0f }, /*rigidRefCount*/ 1.0f,
                                   /*wasActive*/ true, /*checkOnly*/ false, rv,
                                   p714ref::refCombineFriction(0.0f, 0.0f, 3u));
        p714ref::gDlnVariant = 0;
        if (out) *out = sir;
        p714ref::V3 dLin, dAng;
        p714ref::refComputeRigidChange(sir, /*isArticulation*/ false, rigidInvMass, 1.0f, dLin, dAng);
        const p714ref::V3 dp = {
            -(sir.deltaLambdaN * sir.normal.x + sir.deltaLambdaT * sir.tangent.x) * dt,
            -(sir.deltaLambdaN * sir.normal.y + sir.deltaLambdaT * sir.tangent.y) * dt,
            -(sir.deltaLambdaN * sir.normal.z + sir.deltaLambdaT * sir.tangent.z) * dt };
        const float s = invMass * 1.0f;
        Image im;
        im.lin = F4{ dLin.x, dLin.y, dLin.z, 1.0f };
        im.ang = F4{ dAng.x, dAng.y, dAng.z, 0.0f };
        // mDeltaPos is +0-seeded (the query only bumped w): 0 + deltaPos * s per lane.
        im.vert = F4{ 0.0f + dp.x * s, 0.0f + dp.y * s, 0.0f + dp.z * s, 1.0f };
        im.accum = sir.accum;
        return im;
    };
    const Image want = image(0, &sir0);
    const Image twin = image(1, nullptr);

    // Coverage: the tie is presented AND the tangent branch carries it into the reaction.
    const uint32_t zeroTieLanes = (sir0.dlnZeroTie && sir0.tanBranch && f2u(sir0.deltaLambdaN) == 0x80000000u
                                   && f2u(sir0.deltaLambdaT) == 0x80000000u) ? 1u : 0u;
    PXG_CHECK_MSG(zeroTieLanes == 1u,
                  "coverage: the resting lane must present the (+0,-0) PxMax tie with the tangent branch "
                  "live (tie %d tan %d dLN 0x%08x dLT 0x%08x, isTGS=%d)", int(sir0.dlnZeroTie),
                  int(sir0.tanBranch), f2u(sir0.deltaLambdaN), f2u(sir0.deltaLambdaT), int(isTGS));
    PXG_CHECK_MSG(std::memcmp(&want.lin, &twin.lin, sizeof(F4)) != 0
                      || std::memcmp(&want.ang, &twin.ang, sizeof(F4)) != 0,
                  "the FMax twin must change the rigid reaction image");
    PXG_CHECK_MSG(f2u(want.lin.x) == 0u && f2u(want.lin.y) == 0u && f2u(want.lin.z) == 0u
                      && f2u(want.ang.x) == 0x80000000u && f2u(want.ang.y) == 0x80000000u
                      && f2u(want.ang.z) == 0u,
                  "CUDA image: lin (+0,+0,+0), ang (-0,-0,+0)");
    PXG_CHECK_MSG(f2u(twin.lin.x) == 0x80000000u && f2u(twin.lin.y) == 0x80000000u
                      && f2u(twin.lin.z) == 0x80000000u
                      && f2u(twin.ang.x) == 0u && f2u(twin.ang.y) == 0u && f2u(twin.ang.z) == 0u,
                  "twin image: lin (-0,-0,-0), ang (+0,+0,+0)");

    // ---- GPU vs the CUDA image, bitwise ----
    const auto gDelta = downloadBuffer<F4>(dev, bDelta.h, NV);
    const auto gLambda = downloadBuffer<float>(dev, bLambda.h, N);
    const auto gRigidDeltaVel = downloadBuffer<F4>(dev, bRigidDeltaVel.h, 2u * N);
    size_t mm2 = 0;
    auto cmpRow = [&](const char* nm, const F4& r, const F4& g) {
        if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) || f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
        { ++mm2; std::printf("  zero-sign solve %s MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g) "
                             "bits got %08x %08x %08x\n", nm, r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w,
                             f2u(g.x), f2u(g.y), f2u(g.z)); }
    };
    cmpRow("rigidDeltaVel[0]", want.lin, gRigidDeltaVel[0]);
    cmpRow("rigidDeltaVel[1]", want.ang, gRigidDeltaVel[1]);
    cmpRow("mDeltaPos[0]", want.vert, gDelta[0]);
    if (f2u(gLambda[0]) != f2u(want.accum))
    { ++mm2; std::printf("  zero-sign appliedForces MM: ref %g got %g\n", want.accum, gLambda[0]); }
    PXG_CHECK_MSG(f2u(gRigidDeltaVel[1].x) == 0x80000000u && f2u(gRigidDeltaVel[0].y) == 0u,
                  "GPU reaction must carry dLN = -0 (ang.x -0, lin.y +0; got 0x%08x 0x%08x, isTGS=%d)",
                  f2u(gRigidDeltaVel[1].x), f2u(gRigidDeltaVel[0].y), int(isTGS));
    PXG_CHECK_MSG(mm2 == 0, "zero-sign cloth_solveRigidClothCollision%s bit-exact vs CPU (%zu mismatches)",
                  isTGS ? "TGS" : "PGS", mm2);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bAcc.h);
    dev->destroyBuffer(bDelta.h); dev->destroyBuffer(bTri.h); dev->destroyBuffer(bDesc.h);
    dev->destroyBuffer(bContacts.h); dev->destroyBuffer(bNormalPens.h); dev->destroyBuffer(bBary.h);
    dev->destroyBuffer(bInfos.h); dev->destroyBuffer(bNum.h); dev->destroyBuffer(bConstraint.h);
    dev->destroyBuffer(bLambda.h); dev->destroyBuffer(bRigidDeltaVel.h); dev->destroyBuffer(bRefCounts.h);
    dev->destroyBuffer(bBody2World.h); dev->destroyBuffer(bSolverBodyData.h); dev->destroyBuffer(bSolverTxIData.h);
    dev->destroyBuffer(bOutSolverVel.h); dev->destroyBuffer(bSolverBodyVelPool.h);
    dev->destroyBuffer(bSolverBodyIndices.h);
    dev->destroyBuffer(bPrepDesc.h); dev->destroyBuffer(bPrePrepDesc.h);
    dev->destroyBuffer(bSolverCoreDesc.h); dev->destroyBuffer(bSharedDesc.h);
}

PXG_TEST(femClothSimRigidContactZeroSignPGS) { runRigidContactZeroSign(/*isTGS*/ false); }
PXG_TEST(femClothSimRigidContactZeroSignTGS) { runRigidContactZeroSign(/*isTGS*/ true); }

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
    // LEFT TO RIGHT, like the shader's `a + ab * v + ac * w`.  This mirror used to
    // group the two scaled edges first, which differs in the last bit of the
    // returned POINT.  No check here reads that point bit-exactly -- all three
    // callers take only s/t or a distance predicate -- so nothing was wrong, and
    // that is exactly why it could sit here: a mirror that is only ever half-read
    // drifts from the thing it mirrors without anything going red.  Found while
    // writing P9.30's mirror, after P9.29 lost a measurement to the same slip.
    return add(add(a, mul(ab, v)), mul(ac, wv));
}

// closestPtLineLine (deformableCollision.cuh:244) — matches npEdgeEdge.slang:57.
//
// P9.36 CORRECTED THIS HELPER, and the old comment is worth keeping because it is the
// clearest statement of why the defect was corpus-wide.  It read:
//
//     "The vector/scalar `r0 / r0Len` is transcribed as PER-COMPONENT division (divv),
//      matching the shader's `float3 / float` (3x OpFDiv at -O0), NOT a
//      reciprocal-multiply."
//
// "matching the shader" is the whole problem: the mirror was written to agree with the
// port rather than with PhysX.  deformableCollision.cuh:265 is
// `const PxVec3 u0 = r0 / r0Len;`, and PxVec3::operator/ (PxVec3.h:219) is
// `f = Type(1.0)/f; return PxVec3T(x*f, y*f, z*f);` -- ONE reciprocal, three multiplies.
// Three componentwise divides are a different function, disagreeing in at least one
// component for 53.4595% of random vectors.  Both sides now do what the engine does.
static V3 divv(V3 a, float s) { const float r = 1.0f / s; return V3{ a.x * r, a.y * r, a.z * r }; }
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
using p72ref::divv;   // P9.36: divv(v, s) = v * (1/s), as PxVec3::operator/ is
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
    kPrep.dispatch(FEM_PREP_GRID, {}, pp, 1u, 1u);
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

// P9.86f -- the FRICTION clamp of triangleCollisionConstraint (FEMClothExternalSolve.cu:274-275):
//   deltaLambdaT = PxMax(0.f, CT * invDenom);  deltaLambdaT = -PxMin(deltaLambdaT, fc * |deltaLambdaN|);
// PxMax/PxMin<float> are selectMax/selectMin (PxUnixMathIntrinsics.h:116-125): a +-0 tie keeps the SECOND
// operand, where llvmpipe's FMax gives +0 and FMin -0 in either order.  One VT self-contact, every value a
// binary fraction so no bit depends on rounding:
//   triangle a=(0,0,0) b=(1,-1,0) c=(1,1,0), vertex p=(-1/16, 2^-17, -0): the closest point is a (s=t=0),
//   weights (1,-1,-0,-0), disp = (-1/16, 2^-17, -0), |disp| rounds to 1/16, normal = (-1, 2^-13, -0);
//   prev(p) = (3/16, 2^-17, +0) and the triangle at rest give relDelta = (-1/4, +0, -0), so
//   tanDir = (+0, -1, +0) (tanMagSq 2^-30) and CT = (-1/4)(+0) + (+0)(-1) + (-0)(+0) = -0.
// Case A, rest distance 0.1: lim = 0.5 * 0.0375 > 0.  CUDA: PxMax(0,-0) = -0, PxMin(-0, lim) = -0, so
//   deltaLambdaT = +0 and commonVec.z = (dLN * -0) + (+0 * +0) = +0; the FMax port gives -0 there.
// Case B, rest distance 1/16 == |disp|: CN = +0, dLN = -0, lim = +0.  CUDA: PxMin(-0, +0) = +0, so
//   deltaLambdaT = -0 and commonVec.y = (-0 * 2^-13) + (-0 * -1) = +0; the FMin port gives -0 there.
// mDeltaPos only ever holds +0 or sums in the engine (cleared with make_float4(0), FEMClothExternalSolve.cu:1109,
// then atomically added to), and +0 + (-0) = +0 -- so there the flipped zero lane of a contact's dx cannot show.
// The fixture therefore seeds the accumulator with -0, which makes the kernel's per-contact dx bits visible:
// -0 + dx keeps dx's sign.
static void runClothSelfContactFrictionZeroSign(float restDistScalar, int killsVariant)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 1.0f / 60.0f;
    const float invMass = 0.5f;
    const float NZ = -0.0f, E = 7.62939453125e-06f;   // 2^-17
    std::vector<F4> pos = {
        { 0.0f, 0.0f, 0.0f, invMass },       // v0 = a
        { 1.0f, -1.0f, 0.0f, invMass },      // v1 = b
        { 1.0f, 1.0f, 0.0f, invMass },       // v2 = c
        { -0.0625f, E, NZ, invMass },        // v3 = the vertex
    };
    std::vector<F4> rest = pos;
    std::vector<F4> prev = pos;
    prev[3] = F4{ 0.1875f, E, 0.0f, invMass };
    std::vector<F4> deltaPos(4, F4{ NZ, NZ, NZ, 0.0f });
    std::vector<float> friction(4, 0.5f);
    std::vector<TriIdx4> tris = { { 0u, 1u, 2u, 0u } };

    // ---- host mirror (FEMClothExternalSolve.cu:243 + applyContactDelta), variant twins ----
    using p72ref::V3;
    auto v3f = [&](const F4& f) { return V3{ f.x, f.y, f.z }; };
    float s, t;
    p72ref::closestPtPointTriangle(v3f(pos[3]), v3f(pos[0]), v3f(pos[1]), v3f(pos[2]), s, t);
    const float rB = 1.0f - s - t;
    const float w[4] = { 1.0f, -rB, -s, -t };
    const V3 xx[4] = { v3f(pos[3]), v3f(pos[0]), v3f(pos[1]), v3f(pos[2]) };
    const V3 pv[4] = { v3f(prev[3]), v3f(prev[0]), v3f(prev[1]), v3f(prev[2]) };
    const uint32_t slot[4] = { 3u, 0u, 1u, 2u };
    const V3 disp = p72solve::dispFromWeights(w, xx[0], xx[1], xx[2], xx[3]);
    const V3 dl[4] = { p72ref::sub(xx[0], pv[0]), p72ref::sub(xx[1], pv[1]), p72ref::sub(xx[2], pv[2]), p72ref::sub(xx[3], pv[3]) };
    const V3 relDelta = p72solve::dispFromWeights(w, dl[0], dl[1], dl[2], dl[3]);
    const float dist = std::sqrt(p72ref::dot(disp, disp));
    const V3 normal = p72ref::mul(disp, 1.0f / dist);
    const float restTotal = restDistScalar + restDistScalar;
    float count[4];
    for (int i = 0; i < 4; ++i) count[i] = (std::fabs(w[i]) > 1e-3f) ? std::fabs(w[i]) : 0.0f;
    const float fc = 0.25f * (friction[0] + friction[1] + friction[2] + friction[3]);
    auto solveRef = [&](int variant, F4 out[4]) {
        float m[4];
        for (int i = 0; i < 4; ++i) m[i] = count[i] * invMass;
        const float CN = p72ref::dot(disp, normal) - restTotal;
        float denom = m[0] * w[0] * w[0];
        denom += m[1] * w[1] * w[1];
        denom += m[2] * w[2] * w[2];
        denom += m[3] * w[3] * w[3];
        const float invDenom = 1.0f / denom;
        const float dLN = -CN * invDenom;
        V3 cv = p72ref::mul(normal, dLN);
        V3 tanDir = p72ref::sub(relDelta, p72ref::mul(normal, p72ref::dot(relDelta, normal)));
        const float tanMagSq = p72ref::dot(tanDir, tanDir);
        if (fc != 0.0f && tanMagSq > p72solve::THR)
        {
            tanDir = p72ref::mul(tanDir, 1.0f / std::sqrt(tanMagSq));
            const float x = p72ref::dot(relDelta, tanDir) * invDenom;
            // 0 = the CUDA selects; 1 = llvmpipe FMax at :274; 2 = llvmpipe FMin at :275
            float d = (variant == 1) ? ((x > 0.0f) ? x : 0.0f) : ((0.0f > x) ? 0.0f : x);
            const float lim = fc * std::fabs(dLN);
            d = (variant == 2) ? -((d == lim) ? (std::signbit(d) ? d : lim) : (d < lim ? d : lim))
                               : -((d < lim) ? d : lim);
            cv = p72ref::add(cv, p72ref::mul(tanDir, d));
        }
        for (int i = 0; i < 4; ++i) out[i] = deltaPos[i];
        for (int i = 0; i < 4; ++i)
        {
            out[slot[i]].w = count[i];
            if (count[i] > 0.0f)
            {
                const V3 dx = p72ref::mul(cv, m[i] * w[i]);
                out[slot[i]].x = out[slot[i]].x + dx.x;
                out[slot[i]].y = out[slot[i]].y + dx.y;
                out[slot[i]].z = out[slot[i]].z + dx.z;
            }
        }
    };
    F4 ref[4], twin[4];
    solveRef(0, ref);
    PXG_CHECK_MSG(s == 0.0f && t == 0.0f && f2u(normal.x) == 0xBF800000u && f2u(normal.z) == 0x80000000u &&
                  f2u(relDelta.y) == 0u && f2u(relDelta.z) == 0x80000000u,
                  "friction zero-sign rig drifted (s %g t %g n %08x %08x rd %08x %08x)", s, t,
                  f2u(normal.x), f2u(normal.z), f2u(relDelta.y), f2u(relDelta.z));
    for (int v = 1; v <= 2; ++v)
    {
        solveRef(v, twin);
        bool differs = false;
        for (int i = 0; i < 4; ++i)
            differs |= f2u(twin[i].x) != f2u(ref[i].x) || f2u(twin[i].y) != f2u(ref[i].y) || f2u(twin[i].z) != f2u(ref[i].z);
        PXG_CHECK_MSG(differs == (v == killsVariant), "rest %g: the friction-clamp twin %d %s the scattered deltas",
                      restDistScalar, v, differs ? "changes" : "does not change");
    }

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

    PrepClothClothPush pp = {};
    pp.clothes = bDesc.addr; pp.contactInfos = bInfos.addr; pp.numContacts = bNum.addr;
    pp.maxContacts = 64u; pp.clothMaterials = 0ull; pp.updateContactPairs = bUpd.addr;
    Kernel kPrep("femClothClothContactBda.spv", "cloth_clothContactPrepareLaunch", {}, sizeof(PrepClothClothPush));
    kPrep.dispatch(FEM_PREP_GRID, {}, pp, 1u, 1u);
    dev->flush();
    QueryClothClothPush qp = {};
    qp.cloths = bDesc.addr; qp.contactInfos = bInfos.addr; qp.numContacts = bNum.addr;
    Kernel kQuery("femClothClothContactBda.spv", "cloth_queryClothClothContactVTCountLaunch", {}, sizeof(QueryClothClothPush));
    kQuery.dispatch(1u, {}, qp, 1u, 1u);
    dev->flush();
    SolveClothClothPush sp = {};
    sp.cloths = bDesc.addr; sp.contactInfos = bInfos.addr; sp.numContacts = bNum.addr; sp.dt = dt;
    Kernel kSolve("femClothClothContactBda.spv", "cloth_solveClothClothDeltaVTLaunch", {}, sizeof(SolveClothClothPush));
    kSolve.dispatch(1u, {}, sp, 1u, 1u);
    dev->flush();

    const auto g = downloadBuffer<F4>(dev, bDelta.h, 4);
    size_t mm = 0;
    for (uint32_t v = 0; v < 4; ++v)
        if (f2u(g[v].x) != f2u(ref[v].x) || f2u(g[v].y) != f2u(ref[v].y) || f2u(g[v].z) != f2u(ref[v].z) || f2u(g[v].w) != f2u(ref[v].w))
        { ++mm; std::printf("  rest %g mDeltaPos[%u] MM: ref(%08x,%08x,%08x,%08x) got(%08x,%08x,%08x,%08x)\n", restDistScalar, v,
                            f2u(ref[v].x), f2u(ref[v].y), f2u(ref[v].z), f2u(ref[v].w), f2u(g[v].x), f2u(g[v].y), f2u(g[v].z), f2u(g[v].w)); }
    PXG_CHECK_MSG(mm == 0, "cloth_solveClothClothDeltaVT friction zero sign (rest %g): %zu slots differ bitwise from the CUDA selects",
                  restDistScalar, mm);

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
    kPrep.dispatch(FEM_PREP_GRID, {}, pp, 1u, 1u);
    dev->flush();
    const FemFemContactInfoHost giPrep = downloadBuffer<FemFemContactInfoHost>(dev, bInfos.h, 1)[0];
    PXG_CHECK_MSG((giPrep.auxInd & VALIDITY_FLAG_MASK_H) != 0u,
                  "prepare must mark the EE self-contact valid (auxInd=0x%08x)", giPrep.auxInd);

    // ---- query EE (count) ----
    QueryClothClothPush qp = {};
    qp.cloths = bDesc.addr; qp.contactInfos = bInfos.addr; qp.numContacts = bNum.addr;
    Kernel kQuery("femClothClothContactBda.spv", "cloth_queryClothClothContactEECountLaunch", {}, sizeof(QueryClothClothPush));
    kQuery.dispatch(FEM_PREP_GRID, {}, qp, 1u, 1u);
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
    kSolve.dispatch(FEM_PREP_GRID, {}, sp, 1u, 1u);
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
    kPrep.dispatch(FEM_PREP_GRID, {}, pp, 1u, 1u);
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
PXG_TEST(femClothSimSelfContactFrictionZeroSign)
{
    runClothSelfContactFrictionZeroSign(0.05f, 1);      // case A: the FMax twin (:274)
    runClothSelfContactFrictionZeroSign(0.03125f, 2);   // case B: the FMin twin (:275)
}

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

// P9.86c: the two warp reductions (FEMCloth.cu:470-471 leaf walk, :490-491 node aggregate)
// are warpShuffleMin/Max (shuffle.cuh:49-72): a 5-step __shfl_xor_sync butterfly with
// CUDA fminf/fmaxf, NOT the PxMin/PxMax selects the per-triangle gather uses.  On NVIDIA
// fminf/fmaxf are minNum/maxNum with -0 < +0 in either operand order (fminf(+0,-0) = -0,
// fmaxf(+0,-0) = +0), NaN ignored, both-NaN -> canonical NaN.  This mirror used to reduce
// with a lane-0-seeded p72ref::vmin/vmax scan -- the port's pre-P9.86 behaviour, where a
// +/-0 tie keeps the LATER lane's sign -- so it would have expected the old port's bits.
//
// gRefitRedVariant: mutant twins of that old reduction, one bit per (call site x min/max):
//   bit0 leaf-walk min, bit1 leaf-walk max, bit2 node-aggregate min, bit3 node-aggregate max.
// gRefitZeroTie[bit index]: mixed-sign zero ties the CUDA reduction resolved at that site.
static int gRefitRedVariant = 0;
static uint32_t gRefitZeroTie[4] = { 0u, 0u, 0u, 0u };
static float refitU2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static float cuFminfRef(float a, float b, uint32_t& ties)
{
    const bool na = a != a, nb = b != b;
    if (na && nb) return refitU2f(0x7FFFFFFFu);
    if (na) return b;
    if (nb) return a;
    if (a == b) { if (f2u(a) != f2u(b)) ++ties; return refitU2f(f2u(a) | f2u(b)); }
    return a < b ? a : b;
}
static float cuFmaxfRef(float a, float b, uint32_t& ties)
{
    const bool na = a != a, nb = b != b;
    if (na && nb) return refitU2f(0x7FFFFFFFu);
    if (na) return b;
    if (nb) return a;
    if (a == b) { if (f2u(a) != f2u(b)) ++ties; return refitU2f(f2u(a) & f2u(b)); }
    return a > b ? a : b;
}
// shuffle.cuh:49/62 -- v = f(v, __shfl_xor_sync(v, r)) for r = 1, 2, 4, 8, 16; every lane
// ends with the same value, lane 0's is returned.
static p72ref::V3 warpShuffleRef(std::vector<p72ref::V3> v, bool isMax, uint32_t& ties)
{
    for (uint32_t r = 1u; r < 32u; r <<= 1)
    {
        std::vector<p72ref::V3> n(32);
        for (uint32_t l = 0; l < 32u; ++l)
        {
            const p72ref::V3 a = v[l], b = v[l ^ r];
            n[l] = isMax ? p72ref::V3{ cuFmaxfRef(a.x, b.x, ties), cuFmaxfRef(a.y, b.y, ties), cuFmaxfRef(a.z, b.z, ties) }
                         : p72ref::V3{ cuFminfRef(a.x, b.x, ties), cuFminfRef(a.y, b.y, ties), cuFminfRef(a.z, b.z, ties) };
        }
        v = n;
    }
    return v[0];
}
static p72ref::V3 warpRedRef(const std::vector<p72ref::V3>& v, bool isMax, uint32_t bitIndex)
{
    if (gRefitRedVariant & (1 << bitIndex))
    {
        p72ref::V3 r = v[0];
        for (uint32_t l = 1; l < 32u; ++l) r = isMax ? p72ref::vmax(r, v[l]) : p72ref::vmin(r, v[l]);
        return r;
    }
    return warpShuffleRef(v, isMax, gRefitZeroTie[bitIndex]);
}

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
        // min = warpShuffleMin(tMin); max = warpShuffleMax(tMax);  (FEMCloth.cu:470-471)
        const p72ref::V3 rn = warpRedRef(tMin, false, 0u), rx = warpRedRef(tMax, true, 1u);
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

    // sPackedNodeBounds[nodeIndex] = { warpShuffleMin(min), warpShuffleMax(max) }  (:490-491)
    const p72ref::V3 an = warpRedRef(laneMin, false, 2u), ax = warpRedRef(laneMax, true, 3u);
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

// ---------------------------------------------------------------------------
// P9.86c -- the sign of a zero extreme through cloth_refitBoundLaunch's warp reductions.
//
// femClothSimRefitBound's irregular positions never put a zero on an AABB extreme, so the
// two warpShuffleMin/Max reductions (FEMCloth.cu:470-471 leaf walk, :490-491 node
// aggregate; shuffle.cuh fminf/fmaxf) never see a +/-0 tie and the port's pre-P9.86
// PxMin/PxMax select scan passed it.  Here a planar cloth rests in y = 0 with some
// coordinates -0 (a negated zero, or pos + vel*dt with pos = -0 and vel = 0), and the x/z
// extremes are zeros too:
//   node 0 (root): 2 INTERNAL children -> nodes 1, 2
//   node 1: 1 LEAF child, tris 0-1 (lanes 0, 1 of the leaf walk)
//   node 2: 1 LEAF child, tri 2
// Per-triangle PxMin/PxMax selects (FEMCloth.cu:445-467, tie keeps the 2nd operand):
//   T0 min (-0, -0, -0.75)  max (1, -0, +0)      T1 min (0.25, +0, -1)  max (1.25, +0, -0)
//   T2 min (+0, -0, -0.4)   max (0.6, -0, -0.2)
// CUDA: packed[1] = (-0, -0, -1 | 1.25, +0, +0), packed[0] = (-0, -0, -1 | 1.25, +0, +0).
// A lane-order select scan gives node 1 min.y = +0, max.z = -0 (leaf walk) and root
// min.x = +0, max.y = -0 (aggregate) -- one mixed tie per (site x min/max).
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimRefitBoundZeroSign)
{
    using namespace p72refit;
    PxgSlangComputeDevice* dev = testDevice();
    const float N0 = -0.0f;

    std::vector<F4> pos = {
        { N0,    0.0f,  -0.75f, 0.5f },   // v0
        { 0.5f,  0.0f,  0.0f,   0.5f },   // v1
        { 1.0f,  N0,    -0.25f, 0.5f },   // v2
        { 0.25f, N0,    -0.5f,  0.5f },   // v3
        { 0.75f, N0,    N0,     0.5f },   // v4
        { 1.25f, 0.0f,  -1.0f,  0.5f },   // v5
        { 0.0f,  N0,    -0.2f,  0.5f },   // v6
        { 0.6f,  N0,    -0.4f,  0.5f },   // v7
        { 0.3f,  N0,    -0.3f,  0.5f },   // v8
    };
    std::vector<TriIdx4> tris = { { 0u, 1u, 2u, 0u }, { 3u, 4u, 5u, 0u }, { 6u, 7u, 8u, 0u } };

    const uint32_t nbNodes = 3u, maxDepth = 2u;
    TreeBlob tree;
    tree.w.assign(4u + nbNodes * BV32_WORDS + maxDepth * 2u + nbNodes, 0u);
    tree.w[0] = uint32_t(pos.size());
    tree.w[1] = uint32_t(tris.size());
    tree.w[2] = maxDepth;
    tree.w[3] = nbNodes;
    const float POISON = -7777.0f;
    for (uint32_t n = 0; n < nbNodes; ++n)
        for (uint32_t k = 0; k < 32u * 4u; ++k)
        {
            tree.w[tree.nodeWord(n) + W_MMIN + k] = f2u(POISON);
            tree.w[tree.nodeWord(n) + W_MMAX + k] = f2u(POISON);
        }
    tree.w[tree.nodeWord(0) + W_MDATA + 0] = internalWord(1u);
    tree.w[tree.nodeWord(0) + W_MDATA + 1] = internalWord(2u);
    tree.w[tree.nodeWord(0) + W_NBNODES] = 2u;
    tree.w[tree.nodeWord(0) + W_DEPTH] = 0u;
    tree.w[tree.nodeWord(1) + W_MDATA + 0] = leafWord(0u, 2u);
    tree.w[tree.nodeWord(1) + W_NBNODES] = 1u;
    tree.w[tree.nodeWord(1) + W_DEPTH] = 1u;
    tree.w[tree.nodeWord(2) + W_MDATA + 0] = leafWord(2u, 1u);
    tree.w[tree.nodeWord(2) + W_NBNODES] = 1u;
    tree.w[tree.nodeWord(2) + W_DEPTH] = 1u;
    const uint32_t wDepthInfo = 4u + nbNodes * BV32_WORDS;
    tree.w[wDepthInfo + 0] = 0u; tree.w[wDepthInfo + 1] = 1u;
    tree.w[wDepthInfo + 2] = 1u; tree.w[wDepthInfo + 3] = 2u;
    const uint32_t wRemap = wDepthInfo + maxDepth * 2u;
    tree.w[wRemap + 0] = 0u; tree.w[wRemap + 1] = 1u; tree.w[wRemap + 2] = 2u;

    // --- CPU reference + mutant twins -------------------------------------------
    TreeBlob refTree;
    std::vector<Bounds> refPacked;
    auto runRef = [&](int variant, TreeBlob& t, std::vector<Bounds>& pk) {
        gRefitRedVariant = variant;
        t = tree;
        pk.assign(nbNodes, Bounds{ p72ref::V3{ POISON, POISON, POISON }, p72ref::V3{ POISON, POISON, POISON } });
        for (uint32_t i = maxDepth; i > 0u; --i)
        {
            const uint32_t off = t.w[wDepthInfo + (i - 1u) * 2u + 0];
            const uint32_t cnt = t.w[wDepthInfo + (i - 1u) * 2u + 1];
            for (uint32_t j = 0; j < cnt; ++j)
                refitNodeRef(t.w[wRemap + j + off], t, pk, pos, tris);
        }
        gRefitRedVariant = 0;
    };
    auto image = [&](const TreeBlob& t, const std::vector<Bounds>& pk) {
        std::vector<uint32_t> img;
        for (uint32_t n = 0; n < nbNodes; ++n)
            for (uint32_t k2 = 0; k2 < 32u * 4u; ++k2)
            { img.push_back(t.w[t.nodeWord(n) + W_MMIN + k2]); img.push_back(t.w[t.nodeWord(n) + W_MMAX + k2]); }
        for (uint32_t n = 0; n < nbNodes; ++n)
            for (float f : { pk[n].mn.x, pk[n].mn.y, pk[n].mn.z, pk[n].mx.x, pk[n].mx.y, pk[n].mx.z })
                img.push_back(f2u(f));
        return img;
    };

    for (uint32_t s = 0; s < 4u; ++s) gRefitZeroTie[s] = 0u;
    runRef(0, refTree, refPacked);
    const std::vector<uint32_t> want = image(refTree, refPacked);
    // coverage: every (site x min/max) reduction must resolve a mixed +/-0 tie
    PXG_CHECK_MSG(gRefitZeroTie[0] > 0u && gRefitZeroTie[1] > 0u && gRefitZeroTie[2] > 0u && gRefitZeroTie[3] > 0u,
                  "refit zero-sign case must hit mixed +/-0 ties in all four reductions (leafMin %u leafMax %u aggMin %u aggMax %u)",
                  gRefitZeroTie[0], gRefitZeroTie[1], gRefitZeroTie[2], gRefitZeroTie[3]);
    // the hand-derived CUDA bits (NVIDIA fminf/fmaxf zero rule)
    PXG_CHECK_MSG(f2u(refPacked[1].mn.x) == 0x80000000u && f2u(refPacked[1].mn.y) == 0x80000000u &&
                  f2u(refPacked[1].mx.y) == 0u && f2u(refPacked[1].mx.z) == 0u,
                  "reference node 1 bound must be (-0, -0, . | ., +0, +0)");
    PXG_CHECK_MSG(f2u(refPacked[0].mn.x) == 0x80000000u && f2u(refPacked[0].mn.y) == 0x80000000u &&
                  f2u(refPacked[0].mx.y) == 0u && f2u(refPacked[0].mx.z) == 0u,
                  "reference root bound must be (-0, -0, . | ., +0, +0)");
    PXG_CHECK_MSG(f2u(refPacked[2].mn.x) == 0u && f2u(refPacked[2].mn.y) == 0x80000000u &&
                  f2u(refPacked[2].mx.y) == 0x80000000u,
                  "reference node 2 bound must be (+0, -0, . | ., -0, .)");
    for (int v : { 1, 2, 4, 8, 15 })
    {
        TreeBlob tt; std::vector<Bounds> pp;
        runRef(v, tt, pp);
        PXG_CHECK_MSG(image(tt, pp) != want,
                      "select-scan twin %d gives the CUDA image -- the case does not observe it", v);
    }

    // --- device ---------------------------------------------------------------------
    const uint32_t clothId = 0u, elemIdx = 2u;
    const float origContactOffset = 0.037f, specCCD = 0.011f, nbUpdatesPerStep = 3.0f;
    std::vector<float> packedBoundsInit(nbNodes * 6u, POISON);
    std::vector<float> boundArrayInit((elemIdx + 2u) * 6u, POISON);
    // 16-byte trailer: see femClothSimRefitBound (llvmpipe widens the last remap load).
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
            { ++mm; std::printf("  packedNodeBounds[%u][%u] MM: ref %08x got %08x\n", n, e, f2u(r[e]),
                                f2u(gPacked[n * 6 + e])); }
    }
    const float contactDist = origContactOffset + specCCD;
    const float expansion = (nbUpdatesPerStep > 1.0f ? nbUpdatesPerStep : 1.0f) * contactDist;
    const float refBound[6] = {
        refPacked[0].mn.x - expansion, refPacked[0].mn.y - expansion, refPacked[0].mn.z - expansion,
        refPacked[0].mx.x + expansion, refPacked[0].mx.y + expansion, refPacked[0].mx.z + expansion };
    for (uint32_t e = 0; e < 6u; ++e)
    {
        if (f2u(gBounds[elemIdx * 6 + e]) != f2u(refBound[e]))
        { ++mm; std::printf("  boundArray[%u][%u] MM: ref %g got %g\n", elemIdx, e, refBound[e],
                            gBounds[elemIdx * 6 + e]); }
        if (f2u(gBounds[0 * 6 + e]) != f2u(POISON)) { ++mm; std::printf("  boundArray[0][%u] clobbered\n", e); }
        if (f2u(gBounds[(elemIdx + 1u) * 6 + e]) != f2u(POISON))
        { ++mm; std::printf("  boundArray[%u][%u] clobbered\n", elemIdx + 1u, e); }
    }
    PXG_CHECK_MSG(mm == 0, "cloth_refitBoundLaunch zero-sign case bit-exact vs CPU (%zu mismatches)", mm);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bTri.h); dev->destroyBuffer(bTree.h);
    dev->destroyBuffer(bPacked.h); dev->destroyBuffer(bBounds.h); dev->destroyBuffer(bActive.h);
    dev->destroyBuffer(bElem.h); dev->destroyBuffer(bSpec.h); dev->destroyBuffer(bCd.h);
    dev->destroyBuffer(bUpd.h); dev->destroyBuffer(bDesc.h);
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
static const uint32_t GEOM_TRIANGLEMESH_H = uint32_t(physx::PxGeometryType::eTRIANGLEMESH);   // PxGeometryType::eTRIANGLEMESH

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


// ===========================================================================
// P9.19 -- the cloth DAMPING trio.
//
// Three kernels, one feature, and they are tested in the order the engine runs
// them (PxgFEMClothCore::applyDamping, :2023): the two accumulators atomically
// add into mAccumulatedDeltaVel, the applier folds that buffer into the
// velocities and clears it.
//
// THE ORACLE IS A HOST TRANSCRIPTION OF THE CUDA, not of our Slang -- the same
// relationship every Phase-1 battery has to its reference, and the reason it is
// admissible.  Line numbers are cited at each function.
//
// BIT-EXACT DESPITE THE ATOMICS, and the reasoning took a correction worth
// keeping.  The first draft said "two contributors commute, so the sum is
// order-independent" and seeded the buffer with a non-zero value.  That is
// WRONG: with a seed s the two orders give (s+a)+b and (s+b)+a, which differ
// because addition is not ASSOCIATIVE -- commutativity only says a+b == b+a.
// The fixture measured it: one vertex, the contended one, off by one ULP.
//
// So the seed decides the rule, and this fixture uses both:
//   * a vertex with TWO contributors is seeded with exactly ZERO, which makes
//     the sum (0+a)+b == a+b == b+a and therefore order-independent;
//   * a vertex with ONE contributor keeps a NON-ZERO seed, where seed+a is
//     deterministic and a port that ASSIGNS instead of adding is visible.
// Both paths are compared bit-exactly, and no race outcome is asserted.
// ===========================================================================
namespace dmp
{
struct V3 { float x, y, z; };
static V3 v3(float a, float b, float c) { return V3{ a, b, c }; }
static V3 add(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 sub(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 mul(float s, V3 a) { return V3{ s * a.x, s * a.y, s * a.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b)
{ return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }

// PxVec3::normalize (PxVec3.h:295): returns the magnitude BEFORE scaling and
// divides only when it is strictly positive.
//
// P9.36: `*this /= m` is operator/=(Type) (PxVec3.h:260), which forms the
// RECIPROCAL ONCE and multiplies three times.  Three componentwise divides are a
// different function -- they disagree with it in at least one component for 53.5%
// of vectors.  This mirror used to divide, and so did the shader, so the fixture
// was comparing two equally wrong things and stayed green.
static float normalize3(V3& v)
{
    const float m = std::sqrt(dot3(v, v));
    if (m > 0.0f) { const float r = 1.0f / m; v.x *= r; v.y *= r; v.z *= r; }
    return m;
}
// deformableUtils.cuh:118 -- divides by magnitudeSquared even for a unit normal.
static V3 projectOntoPlane(V3 v, V3 n) { return sub(v, mul(dot3(n, v) / dot3(n, n), n)); }
static float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

// ---- the scene -------------------------------------------------------------
// Eight vertices and four triangles.  Vertices 1, 2, 4 and 6 are each shared by
// exactly TWO triangles (contended, still bit-exact); vertex 3 is PINNED
// (invMass 0) so the per-vertex atomic gate and the weightFactor's use of a zero
// term are both live.
// Triangle 4 is ENTIRELY PINNED, and it is not decoration.  Both accumulators
// guard each atomic with `if (invMass != 0)`, which reads like a pure saving --
// the delta is already scaled by invMass, so a pinned vertex would add zero.  It
// is not a saving: with all three masses zero, weightFactor is 1/0 == +inf and
// the delta becomes 0 * inf == NaN.  The guard is what stops three NaNs from
// being atomically added into the buffer.  Without a fully pinned element in the
// fixture, deleting the guard is invisible.
static const uint32_t NB_V = 11u;
static const uint32_t NB_T = 5u;
static const uint32_t TRIS[NB_T][3] = { { 0u, 1u, 2u }, { 2u, 3u, 4u },
                                        { 5u, 6u, 7u }, { 1u, 4u, 6u },
                                        { 8u, 9u, 10u } };
// Off-axis and non-degenerate: an axis-aligned triangle makes the plane normal a
// basis vector and projectVectorOntoPlane a component drop, which hides a wrong
// normal entirely.
static const float POS[NB_V][3] = {
    { 0.00f, 0.00f, 0.00f }, { 1.30f, 0.20f, 0.10f }, { 0.40f, 1.10f, 0.30f },
    { 1.70f, 1.40f, 0.90f }, { 0.10f, 2.20f, 0.50f }, { 2.30f, 0.70f, 1.60f },
    { 1.10f, 2.60f, 0.80f }, { 2.90f, 1.90f, 0.20f },
    { 4.10f, 0.30f, 2.10f }, { 3.40f, 1.20f, 2.80f }, { 4.60f, 1.70f, 1.90f },
};
static const float INV_MASS[NB_V] = { 1.0f, 0.5f, 2.0f, 0.0f, 1.5f, 0.25f, 0.75f, 1.25f,
                                      0.0f, 0.0f, 0.0f };
// Distinct per vertex and NOT a rigid motion of any triangle -- a velocity field
// that is already a rigid rotation has nothing to damp and every mutant lives.
static const float VEL[NB_V][3] = {
    {  0.30f, -0.70f,  0.20f }, { -0.40f,  0.90f, -0.10f }, {  0.60f,  0.10f, -0.50f },
    { -0.20f, -0.30f,  0.80f }, {  0.90f,  0.40f,  0.15f }, { -0.60f,  0.25f, -0.35f },
    {  0.15f, -0.85f,  0.45f }, {  0.70f,  0.55f, -0.25f },
    { -0.35f,  0.65f,  0.05f }, {  0.45f, -0.15f, -0.75f }, {  0.20f,  0.80f,  0.35f },
};
// Four materials: one live, one ZERO damping (the D2 early-out), one NEGATIVE
// (which clamp pulls to zero -- the same early-out by a different route, and the
// one a port testing the RAW product would run with an amplifying factor), and a
// second live one so two triangles with different factors both contribute.
static const uint32_t NB_MAT = 4u;
static const float MAT_DAMP[NB_MAT] = { 3.0f, 0.0f, -5.0f, 7.0f };
// Triangle 4 (the all-pinned one) takes material 0, so it is LIVE and actually
// reaches the atomic gates.
static const uint16_t TRI_MAT[NB_T] = { 0u, 1u, 2u, 3u, 0u };

// --- cloth_accumulateInPlaneDampingDeltaVelocity (FEMCloth.cu:947) ----------
static void inPlaneRef(float dt, std::vector<float>& delta /* NB_V*4 */)
{
    for (uint32_t t = 0; t < NB_T; ++t)
    {
        const float dampingFactor = clamp01(MAT_DAMP[TRI_MAT[t]] * dt);
        if (dampingFactor == 0.0f) continue;

        const uint32_t ia = TRIS[t][0], ib = TRIS[t][1], ic = TRIS[t][2];
        const float invMassA = INV_MASS[ia], invMassB = INV_MASS[ib], invMassC = INV_MASS[ic];
        const V3 pA = v3(POS[ia][0], POS[ia][1], POS[ia][2]);
        const V3 pB = v3(POS[ib][0], POS[ib][1], POS[ib][2]);
        const V3 pC = v3(POS[ic][0], POS[ic][1], POS[ic][2]);
        const V3 vA = v3(VEL[ia][0], VEL[ia][1], VEL[ia][2]);
        const V3 vB = v3(VEL[ib][0], VEL[ib][1], VEL[ib][2]);
        const V3 vC = v3(VEL[ic][0], VEL[ic][1], VEL[ic][2]);

        const V3 linearVelocity = mul(1.0f / 3.0f, add(add(vA, vB), vC));
        const V3 centroid = mul(1.0f / 3.0f, add(add(pA, pB), pC));

        V3 normal = cross3(sub(pB, pA), sub(pC, pA));
        if (normalize3(normal) < 1e-6f) continue;

        const V3 vAProj = projectOntoPlane(sub(vA, linearVelocity), normal);
        const V3 vBProj = projectOntoPlane(sub(vB, linearVelocity), normal);
        const V3 vCProj = projectOntoPlane(sub(vC, linearVelocity), normal);

        V3 rdA = cross3(normal, sub(pA, centroid));
        V3 rdB = cross3(normal, sub(pB, centroid));
        V3 rdC = cross3(normal, sub(pC, centroid));
        const float dA = normalize3(rdA), dB = normalize3(rdB), dC = normalize3(rdC);

        const float rotVel = (1.0f / 3.0f) * (dot3(rdA, vAProj) / dA + dot3(rdB, vBProj) / dB
                                              + dot3(rdC, vCProj) / dC);

        V3 dvA = sub(mul(rotVel * dA, rdA), vAProj);
        V3 dvB = sub(mul(rotVel * dB, rdB), vBProj);
        V3 dvC = sub(mul(rotVel * dC, rdC), vCProj);

        const float weightFactor = 1.0f / (invMassA + invMassB + invMassC);
        // The mean is removed BEFORE the mass weighting (D8).
        const V3 avg = mul(1.0f / 3.0f, add(add(dvA, dvB), dvC));
        dvA = sub(dvA, avg); dvB = sub(dvB, avg); dvC = sub(dvC, avg);
        dvA = mul(invMassA * weightFactor * dampingFactor, dvA);
        dvB = mul(invMassB * weightFactor * dampingFactor, dvB);
        dvC = mul(invMassC * weightFactor * dampingFactor, dvC);

        const uint32_t idx[3] = { ia, ib, ic };
        const V3 dv[3] = { dvA, dvB, dvC };
        const float im[3] = { invMassA, invMassB, invMassC };
        for (int k = 0; k < 3; ++k)
            if (im[k] != 0.0f)
            { delta[idx[k] * 4 + 0] += dv[k].x;
              delta[idx[k] * 4 + 1] += dv[k].y;
              delta[idx[k] * 4 + 2] += dv[k].z; }
    }
}
} // namespace dmp


// The three push-constant blobs, byte-identical to femClothDampingBda.slang's.
struct ApplyDampPush { uint64_t femCloths; uint64_t activeIds; };
struct InPlaneDampPush { uint64_t femCloths; uint64_t activeIds; uint64_t materials;
                         float dt; uint32_t pad; };
struct BendingDampPush { uint64_t femCloths; uint64_t activeIds; float dt; uint32_t pad; };
static_assert(sizeof(ApplyDampPush) == 16, "engine ABI blob size");
static_assert(sizeof(InPlaneDampPush) == 32, "engine ABI blob size");
static_assert(sizeof(BendingDampPush) == 24, "engine ABI blob size");

// ---------------------------------------------------------------------------
// cloth_applyAccumulatedDeltaVelocity -- the applier, on its own.
//
// It is three lines of arithmetic and one of them is a trap: the CUDA assigns
// .x, .y and .z INDIVIDUALLY and never touches .w, which carries the inverse
// mass.  A port that stores the whole float4 zeroes every vertex's mass, and the
// scene would then behave like a cloth of infinitely heavy particles -- plausible
// enough to survive a "does it fall" check.
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimApplyDampingDelta)
{
    using namespace dmp;
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t N = NB_V;
    const uint32_t SLACK = 5u;   // vertices past mNbVerts: must be untouched
    std::vector<F4> vels(N + SLACK), deltas(N + SLACK);
    for (uint32_t i = 0; i < N + SLACK; ++i)
    {
        // .w is the inverse mass and is NOT part of the sum.
        vels[i] = F4{ VEL[i % NB_V][0], VEL[i % NB_V][1], VEL[i % NB_V][2], INV_MASS[i % NB_V] };
        // A distinctive delta including a NON-ZERO w, so a four-component add is
        // visible in the mass rather than only in the velocity.
        deltas[i] = F4{ 0.125f * float(i + 1), -0.25f * float(i + 1), 0.5f * float(i + 1),
                        7.0f + float(i) };
    }
    const std::vector<F4> velsIn = vels, deltasIn = deltas;

    DevBuf bVel = makeDev(dev, vels);
    DevBuf bDelta = makeDev(dev, deltas);

    PxgFEMCloth c = makeZeroDesc();
    c.mVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVel.addr));
    c.mAccumulatedDeltaVel = reinterpret_cast<float4*>(size_t(bDelta.addr));
    c.mNbVerts = N;
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    ApplyDampPush push = {};
    push.femCloths = bDesc.addr;
    push.activeIds = bActive.addr;

    Kernel k("femClothDampingBda.spv", "cloth_applyAccumulatedDeltaVelocity", {},
             sizeof(ApplyDampPush));
    k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    const auto gotVel = downloadBuffer<F4>(dev, bVel.h, N + SLACK);
    const auto gotDelta = downloadBuffer<F4>(dev, bDelta.h, N + SLACK);

    for (uint32_t i = 0; i < N; ++i)
    {
        PXG_CHECK_MSG(f2u(gotVel[i].x) == f2u(velsIn[i].x + deltasIn[i].x) &&
                      f2u(gotVel[i].y) == f2u(velsIn[i].y + deltasIn[i].y) &&
                      f2u(gotVel[i].z) == f2u(velsIn[i].z + deltasIn[i].z),
                      "vertex %u velocity is (%g %g %g), expected (%g %g %g)", i,
                      double(gotVel[i].x), double(gotVel[i].y), double(gotVel[i].z),
                      double(velsIn[i].x + deltasIn[i].x), double(velsIn[i].y + deltasIn[i].y),
                      double(velsIn[i].z + deltasIn[i].z));
        PXG_CHECK_MSG(f2u(gotVel[i].w) == f2u(velsIn[i].w),
                      "vertex %u lost its inverse mass: %g became %g -- the CUDA writes x, y "
                      "and z one at a time and never touches w", i,
                      double(velsIn[i].w), double(gotVel[i].w));
        PXG_CHECK_MSG(gotDelta[i].x == 0.0f && gotDelta[i].y == 0.0f &&
                      gotDelta[i].z == 0.0f && gotDelta[i].w == 0.0f,
                      "vertex %u delta not cleared: (%g %g %g %g) -- the delta buffer IS "
                      "zeroed in full, all four components", i, double(gotDelta[i].x),
                      double(gotDelta[i].y), double(gotDelta[i].z), double(gotDelta[i].w));
    }
    for (uint32_t i = N; i < N + SLACK; ++i)
    {
        PXG_CHECK_MSG(f2u(gotVel[i].x) == f2u(velsIn[i].x) && f2u(gotVel[i].w) == f2u(velsIn[i].w),
                      "vertex %u is past mNbVerts (%u) and was written anyway", i, N);
        PXG_CHECK_MSG(f2u(gotDelta[i].w) == f2u(deltasIn[i].w),
                      "delta %u is past mNbVerts and was cleared anyway", i);
    }

    std::printf("    [ note ] %u vertices summed and cleared, %u past the count untouched; "
                "inverse masses preserved\n", N, SLACK);
    dev->destroyBuffer(bVel.h); dev->destroyBuffer(bDelta.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}


// ---------------------------------------------------------------------------
// cloth_accumulateInPlaneDampingDeltaVelocity, against a host transcription of
// FEMCloth.cu:947.
//
// BIT-EXACT, and the fixture earns that: no vertex is touched more than twice,
// so the two atomic adds commute and the sum is order-independent.  Four of the
// eight vertices ARE touched twice, so the contention is real.
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimInPlaneDamping)
{
    using namespace dmp;
    PxgSlangComputeDevice* dev = testDevice();

    const float dt = 0.1f;   // MAT_DAMP 3 -> 0.3, 7 -> 0.7; both strictly inside [0,1]
    const uint32_t SLACK = 4u;

    std::vector<F4> positions(NB_V), vels(NB_V), deltas(NB_V + SLACK);
    for (uint32_t i = 0; i < NB_V; ++i)
    {
        positions[i] = F4{ POS[i][0], POS[i][1], POS[i][2], INV_MASS[i] };
        vels[i] = F4{ VEL[i][0], VEL[i][1], VEL[i][2], INV_MASS[i] };
    }
    // How many triangles will actually write to each vertex -- needed BEFORE the
    // seeding, because the seed is what makes the comparison legitimate (see the
    // header): zero where two atomics contend, non-zero where only one writes.
    uint32_t hits[NB_V] = { 0 };
    for (uint32_t t = 0; t < NB_T; ++t)
    {
        if (clamp01(MAT_DAMP[TRI_MAT[t]] * dt) == 0.0f) continue;
        for (int k2 = 0; k2 < 3; ++k2) if (INV_MASS[TRIS[t][k2]] != 0.0f) hits[TRIS[t][k2]]++;
    }
    for (uint32_t i = 0; i < NB_V + SLACK; ++i)
    {
        const bool contended = (i < NB_V) && (hits[i] > 1u);
        deltas[i] = contended
            ? F4{ 0.0f, 0.0f, 0.0f, 9.5f + float(i) }
            : F4{ 0.03f * float(i), -0.02f * float(i), 0.05f * float(i), 9.5f + float(i) };
    }
    const std::vector<F4> deltasIn = deltas;

    std::vector<uint32_t> triIdx(NB_T * 4u, 0u);
    for (uint32_t t = 0; t < NB_T; ++t)
    { triIdx[t * 4 + 0] = TRIS[t][0]; triIdx[t * 4 + 1] = TRIS[t][1];
      triIdx[t * 4 + 2] = TRIS[t][2];
      // .w is NOT part of the triangle here -- point it at a vertex that would
      // change the answer if a port folded it in.
      triIdx[t * 4 + 3] = (t + 5u) % NB_V; }

    // mMaterialIndices is a PxU16 array; pack two per word.
    std::vector<uint32_t> matIdxWords((NB_T + 1u) / 2u, 0u);
    for (uint32_t t = 0; t < NB_T; ++t)
        matIdxWords[t >> 1] |= uint32_t(TRI_MAT[t]) << (16u * (t & 1u));

    std::vector<PxsDeformableSurfaceMaterialData> mats(NB_MAT);
    for (uint32_t m = 0; m < NB_MAT; ++m)
    {
        std::memset(&mats[m], 0, sizeof(mats[m]));
        mats[m].elasticityDamping = MAT_DAMP[m];
        // Poison the neighbouring fields: a port reading thickness (@12) or
        // bendingStiffness (@16) instead of elasticityDamping (@20) lands here.
        mats[m].youngs = 1.0e6f; mats[m].poissons = 0.45f;
        mats[m].dynamicFriction = 0.31f; mats[m].thickness = 0.41f;
        mats[m].bendingStiffness = 0.51f; mats[m].bendingDamping = 0.61f;
    }

    DevBuf bPos = makeDev(dev, positions), bVel = makeDev(dev, vels);
    DevBuf bDelta = makeDev(dev, deltas), bTri = makeDev(dev, triIdx);
    DevBuf bMatIdx = makeDev(dev, matIdxWords), bMat = makeDev(dev, mats);

    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    c.mVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVel.addr));
    c.mAccumulatedDeltaVel = reinterpret_cast<float4*>(size_t(bDelta.addr));
    c.mTriangleVertexIndices = reinterpret_cast<uint4*>(size_t(bTri.addr));
    c.mMaterialIndices = reinterpret_cast<PxU16*>(size_t(bMatIdx.addr));
    c.mNbVerts = NB_V;
    c.mNbTriangles = NB_T;
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    InPlaneDampPush push = {};
    push.femCloths = bDesc.addr;
    push.activeIds = bActive.addr;
    push.materials = bMat.addr;
    push.dt = dt;

    Kernel k("femClothDampingBda.spv", "cloth_accumulateInPlaneDampingDeltaVelocity", {},
             sizeof(InPlaneDampPush));
    k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    std::vector<float> want(size_t(NB_V + SLACK) * 4u);
    for (uint32_t i = 0; i < NB_V + SLACK; ++i)
    { want[i * 4 + 0] = deltasIn[i].x; want[i * 4 + 1] = deltasIn[i].y;
      want[i * 4 + 2] = deltasIn[i].z; want[i * 4 + 3] = deltasIn[i].w; }
    inPlaneRef(dt, want);

    const auto got = downloadBuffer<F4>(dev, bDelta.h, NB_V + SLACK);
    uint32_t touched = 0, contended = 0, seeded = 0;
    for (uint32_t i = 0; i < NB_V; ++i)
    {
        if (hits[i]) ++touched;
        if (hits[i] > 1u) ++contended;
        if (hits[i] == 1u &&
            (deltasIn[i].x != 0.0f || deltasIn[i].y != 0.0f || deltasIn[i].z != 0.0f)) ++seeded;
        PXG_CHECK_MSG(hits[i] <= 2u,
                      "vertex %u has %u contributors -- with three the sum is (s+a)+b+c in "
                      "SOME order and no seed makes that order-independent", i, hits[i]);
        PXG_CHECK_MSG(hits[i] <= 1u || (deltasIn[i].x == 0.0f && deltasIn[i].y == 0.0f &&
                                        deltasIn[i].z == 0.0f),
                      "vertex %u is contended and was seeded non-zero -- (s+a)+b != (s+b)+a "
                      "and this comparison would be asserting a race", i);
    }
    PXG_CHECK_MSG(touched > 0 && contended > 0 && seeded > 0,
                  "%u vertices touched, %u contended, %u single-writer-and-seeded -- the "
                  "fixture must exercise the contended path AND the add-not-assign path",
                  touched, contended, seeded);

    for (uint32_t i = 0; i < NB_V + SLACK; ++i)
    {
        PXG_CHECK_MSG(f2u(got[i].x) == f2u(want[i * 4 + 0]) &&
                      f2u(got[i].y) == f2u(want[i * 4 + 1]) &&
                      f2u(got[i].z) == f2u(want[i * 4 + 2]),
                      "vertex %u delta is (%.9g %.9g %.9g), expected (%.9g %.9g %.9g)", i,
                      double(got[i].x), double(got[i].y), double(got[i].z),
                      double(want[i * 4 + 0]), double(want[i * 4 + 1]), double(want[i * 4 + 2]));
        PXG_CHECK_MSG(f2u(got[i].w) == f2u(deltasIn[i].w),
                      "vertex %u delta .w changed (%g -> %g) -- AtomicAdd(float4*, i, PxVec3) "
                      "touches x, y and z only", i, double(deltasIn[i].w), double(got[i].w));
    }

    uint32_t live = 0, pinnedTris = 0;
    for (uint32_t t = 0; t < NB_T; ++t)
    {
        if (clamp01(MAT_DAMP[TRI_MAT[t]] * dt) == 0.0f) continue;
        ++live;
        if (INV_MASS[TRIS[t][0]] == 0.0f && INV_MASS[TRIS[t][1]] == 0.0f &&
            INV_MASS[TRIS[t][2]] == 0.0f) ++pinnedTris;
    }
    PXG_CHECK_MSG(pinnedTris > 0,
                  "no LIVE triangle has all three vertices pinned -- weightFactor never goes "
                  "infinite, the deltas never go NaN, and deleting the per-vertex atomic "
                  "guard is invisible");
    std::printf("    [ note ] %u triangles (%u live, %u gated by clamp, %u of the live ones "
                "fully pinned), %u vertices touched: %u contended (zero-seeded), %u "
                "single-writer with a live seed; bit-exact\n", NB_T, live, NB_T - live,
                pinnedTris, touched, contended, seeded);
    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bDelta.h);
    dev->destroyBuffer(bTri.h); dev->destroyBuffer(bMatIdx.h); dev->destroyBuffer(bMat.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}


namespace dmp
{
// ---- the bending scene -----------------------------------------------------
// Four triangle pairs over the same eight vertices, split shared/non-shared at
// NB_SHARED.  The uint4 order is the CUDA's and NOT the obvious one:
//     .x, .y = the two TIPS      .z, .w = the shared EDGE (the hinge axis)
static const uint32_t NB_TP = 4u;
static const uint32_t NB_SHARED_TP = 2u;
static const uint32_t PAIRS[NB_TP][4] = {
    { 0u, 3u, 1u, 2u },   // shared 0   -- vertex 3 is PINNED, so its atomic is gated
    { 4u, 5u, 6u, 7u },   // shared 1   -- gated out by zero flexural stiffness
    { 2u, 6u, 0u, 4u },   // non-shared 0
    { 1u, 7u, 3u, 5u },   // non-shared 1 -- gated out by zero damping
};
// .x rest angle (unread here), .y flexural stiffness, .z bending damping.
static const float PAIR_COEFF[NB_TP][3] = {
    { 0.21f, 1.5f, 3.0f },   // live, factor 0.3
    { 0.31f, 0.0f, 4.0f },   // stiffness 0 -> gated
    { 0.41f, 2.5f, 6.0f },   // live, factor 0.6
    { 0.51f, 1.0f, 0.0f },   // damping 0 -> gated
};

// --- cloth_accumulateBendingDampingDeltaVelocity (FEMCloth.cu:1046) ---------
// one pair of FEMCloth.cu:1046-1165; idx = { iA, iB, iC, iC2 } (D9: A and B are the
// EDGE, .z/.w; C and C2 the TIPS, .x/.y), pos/vel/im indexed the same way
// P9.86 mutant twins (0 = the CUDA): 1 replays `deltaVelA/B = -deltaAvg` (the literal
// `0 - x` llvmpipe folds to a negate), 2 replays `(tip1 + tip2) * 0.25` without the
// +0 seed (slangc's `0 + x` fold).  femClothSimBendingDampingZeroSign requires both
// to change its expected image.
static int gBendVariant = 0;
static void bendingPairRef(const V3 pos[4], const V3 vel[4], const float im[4],
                           float stiffness, float damping, float dt, const uint32_t idx[4],
                           std::vector<float>& delta)
{
    const float dampingFactor = clamp01(damping * dt);
    if (dampingFactor == 0.0f || stiffness == 0.0f) return;
    const float imA = im[0], imB = im[1], imC = im[2], imC2 = im[3];
    const V3 velA = vel[0], velB = vel[1], velC = vel[2], velC2 = vel[3];
    const V3 posA = pos[0], posB = pos[1], posC = pos[2], posC2 = pos[3];

    // D13: the mean of the TWO EDGE velocities, not of all four.
    const V3 linearVelocity = mul(0.5f, add(velA, velB));

    V3 edgeDir = sub(posB, posA);
    if (normalize3(edgeDir) < 1e-6f) return;

    const V3 relVelTip1 = sub(velC, linearVelocity);
    const V3 relVelTip2 = sub(velC2, linearVelocity);

    V3 tip1Dir = cross3(edgeDir, sub(posC, posA));
    const float d1 = normalize3(tip1Dir);
    V3 tip2Dir = cross3(edgeDir, sub(posC2, posA));
    const float d2 = normalize3(tip2Dir);

    const float rot1 = dot3(tip1Dir, relVelTip1) / d1;
    const float rot2 = dot3(tip2Dir, relVelTip2) / d2;
    const float deltaRotVelDamp = (rot2 - rot1) * dampingFactor;

    // D14: opposite signs on the two tips.
    V3 dT1 = mul(deltaRotVelDamp * d1, tip1Dir);
    V3 dT2 = mul(-deltaRotVelDamp * d2, tip2Dir);
    V3 dA = v3(0.0f, 0.0f, 0.0f), dB = v3(0.0f, 0.0f, 0.0f);

    const float weightFactor = 1.0f / (imA + imB + imC + imC2);
    // (deltaVelA + deltaVelB + deltaVelTip1 + deltaVelTip2) * 0.25f, left to right
    const V3 avg = (gBendVariant == 2) ? mul(0.25f, add(dT1, dT2))
                                       : mul(0.25f, add(add(add(dA, dB), dT1), dT2));
    if (gBendVariant == 1) { dA = v3(-avg.x, -avg.y, -avg.z); dB = dA; }
    else { dA = sub(dA, avg); dB = sub(dB, avg); }
    dT1 = sub(dT1, avg); dT2 = sub(dT2, avg);
    // D15: NO second dampingFactor -- it is already inside deltaRotVelDamp.
    dT1 = mul(imC * weightFactor, dT1);
    dT2 = mul(imC2 * weightFactor, dT2);
    dA = mul(imA * weightFactor, dA);
    dB = mul(imB * weightFactor, dB);

    const V3 dv[4] = { dA, dB, dT1, dT2 };
    for (int k = 0; k < 4; ++k)
        if (im[k] != 0.0f)
        { delta[idx[k] * 4 + 0] += dv[k].x;
          delta[idx[k] * 4 + 1] += dv[k].y;
          delta[idx[k] * 4 + 2] += dv[k].z; }
}

static void bendingRef(float dt, std::vector<float>& delta /* (NB_V+slack)*4 */)
{
    for (uint32_t t = 0; t < NB_TP; ++t)
    {
        const uint32_t idx[4] = { PAIRS[t][2], PAIRS[t][3], PAIRS[t][0], PAIRS[t][1] };
        V3 pos[4], vel[4];
        float im[4];
        for (int k = 0; k < 4; ++k)
        {
            pos[k] = v3(POS[idx[k]][0], POS[idx[k]][1], POS[idx[k]][2]);
            vel[k] = v3(VEL[idx[k]][0], VEL[idx[k]][1], VEL[idx[k]][2]);
            im[k] = INV_MASS[idx[k]];
        }
        bendingPairRef(pos, vel, im, PAIR_COEFF[t][1], PAIR_COEFF[t][2], dt, idx, delta);
    }
}
} // namespace dmp

// ---------------------------------------------------------------------------
// cloth_accumulateBendingDampingDeltaVelocity, against FEMCloth.cu:1046.
//
// Same seeding rule as the in-plane twin: zero at contended vertices, live seed
// at single-writer ones, both compared bit-exactly.
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimBendingDamping)
{
    using namespace dmp;
    PxgSlangComputeDevice* dev = testDevice();

    const float dt = 0.1f;   // 3.0 -> 0.3, 6.0 -> 0.6; both strictly inside [0,1]
    const uint32_t SLACK = 4u;

    std::vector<F4> positions(NB_V), vels(NB_V), deltas(NB_V + SLACK);
    for (uint32_t i = 0; i < NB_V; ++i)
    {
        positions[i] = F4{ POS[i][0], POS[i][1], POS[i][2], INV_MASS[i] };
        vels[i] = F4{ VEL[i][0], VEL[i][1], VEL[i][2], INV_MASS[i] };
    }

    uint32_t hits[NB_V] = { 0 };
    for (uint32_t t = 0; t < NB_TP; ++t)
    {
        if (clamp01(PAIR_COEFF[t][2] * dt) == 0.0f || PAIR_COEFF[t][1] == 0.0f) continue;
        for (int k2 = 0; k2 < 4; ++k2) if (INV_MASS[PAIRS[t][k2]] != 0.0f) hits[PAIRS[t][k2]]++;
    }
    for (uint32_t i = 0; i < NB_V + SLACK; ++i)
    {
        const bool contended = (i < NB_V) && (hits[i] > 1u);
        deltas[i] = contended
            ? F4{ 0.0f, 0.0f, 0.0f, 4.25f + float(i) }
            : F4{ -0.04f * float(i), 0.06f * float(i), -0.01f * float(i), 4.25f + float(i) };
    }
    const std::vector<F4> deltasIn = deltas;

    // Two ordered arrays, and the non-shared one is indexed from ZERO.
    std::vector<uint32_t> sharedIdx(NB_SHARED_TP * 4u), nonSharedIdx((NB_TP - NB_SHARED_TP) * 4u);
    std::vector<F4> sharedCoeff(NB_SHARED_TP), nonSharedCoeff(NB_TP - NB_SHARED_TP);
    for (uint32_t t = 0; t < NB_TP; ++t)
    {
        const bool shared = t < NB_SHARED_TP;
        const uint32_t k = shared ? t : t - NB_SHARED_TP;
        uint32_t* dst = shared ? &sharedIdx[k * 4] : &nonSharedIdx[k * 4];
        for (int c = 0; c < 4; ++c) dst[c] = PAIRS[t][c];
        const F4 co{ PAIR_COEFF[t][0], PAIR_COEFF[t][1], PAIR_COEFF[t][2], 0.0f };
        (shared ? sharedCoeff[k] : nonSharedCoeff[k]) = co;
    }

    DevBuf bPos = makeDev(dev, positions), bVel = makeDev(dev, vels);
    DevBuf bDelta = makeDev(dev, deltas);
    DevBuf bSi = makeDev(dev, sharedIdx), bNsi = makeDev(dev, nonSharedIdx);
    DevBuf bSc = makeDev(dev, sharedCoeff), bNsc = makeDev(dev, nonSharedCoeff);

    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    c.mVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVel.addr));
    c.mAccumulatedDeltaVel = reinterpret_cast<float4*>(size_t(bDelta.addr));
    c.mOrderedSharedTrianglePairVertexIndices = reinterpret_cast<uint4*>(size_t(bSi.addr));
    c.mOrderedNonSharedTrianglePairVertexIndices = reinterpret_cast<uint4*>(size_t(bNsi.addr));
    c.mOrderedSharedRestBendingAngle_flexuralStiffness_damping =
        reinterpret_cast<float4*>(size_t(bSc.addr));
    c.mOrderedNonSharedRestBendingAngle_flexuralStiffness_damping =
        reinterpret_cast<float4*>(size_t(bNsc.addr));
    c.mNbVerts = NB_V;
    c.mNbTrianglePairs = NB_TP;
    c.mNbSharedTrianglePairs = NB_SHARED_TP;
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    BendingDampPush push = {};
    push.femCloths = bDesc.addr;
    push.activeIds = bActive.addr;
    push.dt = dt;

    Kernel k("femClothDampingBda.spv", "cloth_accumulateBendingDampingDeltaVelocity", {},
             sizeof(BendingDampPush));
    k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    std::vector<float> want(size_t(NB_V + SLACK) * 4u);
    for (uint32_t i = 0; i < NB_V + SLACK; ++i)
    { want[i * 4 + 0] = deltasIn[i].x; want[i * 4 + 1] = deltasIn[i].y;
      want[i * 4 + 2] = deltasIn[i].z; want[i * 4 + 3] = deltasIn[i].w; }
    bendingRef(dt, want);

    uint32_t touched = 0, contended = 0, seeded = 0, sharedLive = 0, nonSharedLive = 0;
    for (uint32_t t = 0; t < NB_TP; ++t)
        if (clamp01(PAIR_COEFF[t][2] * dt) != 0.0f && PAIR_COEFF[t][1] != 0.0f)
            (t < NB_SHARED_TP ? sharedLive : nonSharedLive)++;
    for (uint32_t i = 0; i < NB_V; ++i)
    {
        if (hits[i]) ++touched;
        if (hits[i] > 1u) ++contended;
        if (hits[i] == 1u && (deltasIn[i].x != 0.0f || deltasIn[i].y != 0.0f)) ++seeded;
        PXG_CHECK_MSG(hits[i] <= 2u, "vertex %u has %u contributors -- no seed makes three "
                      "atomic adds order-independent", i, hits[i]);
    }
    PXG_CHECK_MSG(sharedLive > 0 && nonSharedLive > 0,
                  "%u shared and %u non-shared pairs survive the gates -- BOTH ordered arrays "
                  "must be read, or the index split at mNbSharedTrianglePairs is untested",
                  sharedLive, nonSharedLive);
    PXG_CHECK_MSG(touched > 0 && contended > 0 && seeded > 0,
                  "%u touched, %u contended, %u seeded single-writer -- the fixture must "
                  "exercise all three", touched, contended, seeded);

    const auto got = downloadBuffer<F4>(dev, bDelta.h, NB_V + SLACK);
    for (uint32_t i = 0; i < NB_V + SLACK; ++i)
    {
        PXG_CHECK_MSG(f2u(got[i].x) == f2u(want[i * 4 + 0]) &&
                      f2u(got[i].y) == f2u(want[i * 4 + 1]) &&
                      f2u(got[i].z) == f2u(want[i * 4 + 2]),
                      "vertex %u delta is (%.9g %.9g %.9g), expected (%.9g %.9g %.9g)", i,
                      double(got[i].x), double(got[i].y), double(got[i].z),
                      double(want[i * 4 + 0]), double(want[i * 4 + 1]), double(want[i * 4 + 2]));
        PXG_CHECK_MSG(f2u(got[i].w) == f2u(deltasIn[i].w),
                      "vertex %u delta .w changed -- the atomic touches x, y and z only", i);
    }

    std::printf("    [ note ] %u pairs (%u shared live, %u non-shared live, %u gated), "
                "%u vertices touched: %u contended, %u single-writer seeded; bit-exact\n",
                NB_TP, sharedLive, nonSharedLive, NB_TP - sharedLive - nonSharedLive,
                touched, contended, seeded);
    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bDelta.h);
    dev->destroyBuffer(bSi.h); dev->destroyBuffer(bNsi.h);
    dev->destroyBuffer(bSc.h); dev->destroyBuffer(bNsc.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

// ---------------------------------------------------------------------------
// P9.86: the bending damper's two IEEE zero rules, observed.  deltaVelA/B start
// as PxVec3(0) (FEMCloth.cu:1131-1132): the mean is ((+0 + +0) + tip1) + tip2,
// which is +0 even when both tips are -0, and A/B become 0 - mean, which is +0
// for a +0 mean.  One pair, all four velocities zero (so deltaRotVelDamp is +0
// and the tip deltas are zeros signed by their directions), tips on opposite
// sides of the hinge so tip1Dir.y < 0 < tip2Dir.y (both tip deltas -0 in y), and
// every accumulator seeded with -0: -0 + (+0) = +0 but -0 + (-0) = -0, so both
// spellings the port once had (the folded seed, the folded `0 - x`) change bits.
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimBendingDampingZeroSign)
{
    using namespace dmp;
    PxgSlangComputeDevice* dev = testDevice();
    const float dt = 0.1f;

    // tips .x/.y = vertices 2, 3; edge .z/.w = vertices 0, 1
    const float P[4][3] = { { 0, 0, 0 }, { 1, 0, 0 }, { 0.5f, 1, 0.3f }, { 0.5f, -1, -0.2f } };
    const float IM[4] = { 0.5f, 2.0f, 1.0f, 0.25f };
    std::vector<F4> positions(4), vels(4), deltas(4);
    for (uint32_t i = 0; i < 4; ++i)
    {
        positions[i] = F4{ P[i][0], P[i][1], P[i][2], IM[i] };
        vels[i] = F4{ 0.0f, 0.0f, 0.0f, IM[i] };
        deltas[i] = F4{ -0.0f, -0.0f, -0.0f, 7.5f + float(i) };
    }
    const std::vector<F4> deltasIn = deltas;
    std::vector<uint32_t> sharedIdx = { 2u, 3u, 0u, 1u };
    std::vector<uint32_t> nonSharedIdx = { 0u, 0u, 0u, 0u };
    std::vector<F4> sharedCoeff = { F4{ 0.1f, 3.0f, 6.0f, 0.0f } };
    std::vector<F4> nonSharedCoeff = { F4{ 0, 0, 0, 0 } };

    DevBuf bPos = makeDev(dev, positions), bVel = makeDev(dev, vels);
    DevBuf bDelta = makeDev(dev, deltas);
    DevBuf bSi = makeDev(dev, sharedIdx), bNsi = makeDev(dev, nonSharedIdx);
    DevBuf bSc = makeDev(dev, sharedCoeff), bNsc = makeDev(dev, nonSharedCoeff);
    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bPos.addr));
    c.mVelocity_InvMass = reinterpret_cast<float4*>(size_t(bVel.addr));
    c.mAccumulatedDeltaVel = reinterpret_cast<float4*>(size_t(bDelta.addr));
    c.mOrderedSharedTrianglePairVertexIndices = reinterpret_cast<uint4*>(size_t(bSi.addr));
    c.mOrderedNonSharedTrianglePairVertexIndices = reinterpret_cast<uint4*>(size_t(bNsi.addr));
    c.mOrderedSharedRestBendingAngle_flexuralStiffness_damping =
        reinterpret_cast<float4*>(size_t(bSc.addr));
    c.mOrderedNonSharedRestBendingAngle_flexuralStiffness_damping =
        reinterpret_cast<float4*>(size_t(bNsc.addr));
    c.mNbVerts = 4u;
    c.mNbTrianglePairs = 1u;
    c.mNbSharedTrianglePairs = 1u;
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);
    std::vector<uint32_t> active = { 0u };
    DevBuf bActive = makeDev(dev, active);

    BendingDampPush push = {};
    push.femCloths = bDesc.addr;
    push.activeIds = bActive.addr;
    push.dt = dt;
    Kernel k("femClothDampingBda.spv", "cloth_accumulateBendingDampingDeltaVelocity", {},
             sizeof(BendingDampPush));
    k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    auto ref = [&](int variant) {
        std::vector<float> w(16);
        for (uint32_t i = 0; i < 4; ++i)
        { w[i * 4 + 0] = deltasIn[i].x; w[i * 4 + 1] = deltasIn[i].y;
          w[i * 4 + 2] = deltasIn[i].z; w[i * 4 + 3] = deltasIn[i].w; }
        const uint32_t idx[4] = { 0u, 1u, 2u, 3u };
        V3 pos[4], vel[4];
        for (int q = 0; q < 4; ++q) { pos[q] = v3(P[q][0], P[q][1], P[q][2]); vel[q] = v3(0, 0, 0); }
        gBendVariant = variant;
        bendingPairRef(pos, vel, IM, sharedCoeff[0].y, sharedCoeff[0].z, dt, idx, w);
        gBendVariant = 0;
        return w;
    };
    const std::vector<float> want = ref(0);
    for (int v = 1; v <= 2; ++v)
        PXG_CHECK_MSG(std::memcmp(want.data(), ref(v).data(), want.size() * 4) != 0,
                      "mutant twin %d gives the same image -- the case does not observe it", v);

    const auto got = downloadBuffer<F4>(dev, bDelta.h, 4);
    for (uint32_t i = 0; i < 4; ++i)
        PXG_CHECK_MSG(f2u(got[i].x) == f2u(want[i * 4 + 0]) &&
                      f2u(got[i].y) == f2u(want[i * 4 + 1]) &&
                      f2u(got[i].z) == f2u(want[i * 4 + 2]) &&
                      f2u(got[i].w) == f2u(deltasIn[i].w),
                      "vertex %u delta bits %08x %08x %08x, expected %08x %08x %08x", i,
                      f2u(got[i].x), f2u(got[i].y), f2u(got[i].z), f2u(want[i * 4 + 0]),
                      f2u(want[i * 4 + 1]), f2u(want[i * 4 + 2]));
    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bVel.h); dev->destroyBuffer(bDelta.h);
    dev->destroyBuffer(bSi.h); dev->destroyBuffer(bNsi.h);
    dev->destroyBuffer(bSc.h); dev->destroyBuffer(bNsc.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h);
}

// ===========================================================================
// P9.26 -- cloth_midphaseVertexMeshLaunch, the cloth-VERTEX-vs-COOKED-TRIANGLE-MESH
// midphase.  The other half of the block whose TRIANGLE half femClothSimMeshMidphase
// above pins: the engine runs both, one after the other, off the same bounds and the
// same intermediate stack (PxgNarrowphaseCore.cpp:6581/:6671 then :6724/:6814).
//
// The two kernels share a cooked blob, a BV32 walk and an argument shape, and differ
// in four ways that are each a plausible copy-paste from the file next door.  Every
// one of them gets a query in this fixture whose verdict FLIPS if the sibling's rule
// is used instead:
//
//   * V1/V3 THERE IS NO FRONT-FACE GATE and no SAT.  The leaf test is the triangle's
//     own AABB against a sphere bound, and that is all it is.  Vertex 0's leaf holds a
//     BACK-FACING triangle that the sibling's `dot(nB, centre) + dB >= 0` throws away;
//     here it must be KEPT, so vertex 0 expects TWO pairs and not one.
//   * V2 EVERY HIT IS PUSHED.  The sibling keeps the four closest by separation;
//     `pushOntoStackFullWarp` has no reduction at all.  Vertex 1's leaf holds SIX
//     qualifying triangles and all six must come out -- a borrowed top-4 gives four.
//   * V5 THE RADIUS IS NOT DIVIDED BY THE SCALE.  min|scale| is 0.5, so the softbody
//     reading gives a 0.40 band against the correct 0.20, and vertex 2 sits at 0.28 --
//     inside the wrong band, outside the right one.
//   * the query is built from ONE VERTEX, so the mesh-space round trip is the whole
//     of the geometry.  The rigid rotation is NINETY degrees about y, not a hundred
//     and eighty, because 180 is its own inverse and would let `rotate` stand in for
//     `rotateInv` unnoticed -- the same measurement the sibling records.
//
// The expected pair set is computed by a CPU transcription of the walk rather than
// written down, so a mutant anywhere in the traversal fails without the fixture
// having to anticipate which node it broke.  What IS written down, separately, is
// what each vertex's count has to be and why -- a reference that drifts with the
// shader would otherwise agree with it.
// ===========================================================================
namespace p926vm
{
struct ClothVertMidPush
{
    uint32_t numWorkItems; float toleranceLength;
    uint64_t cmInputs, transformCache, bounds, contactDistance, gpuShapes, femClothes;
    uint32_t stackSizeBytes, pad60;
    uint64_t stackPtr, midphasePairsNum;
};
static_assert(sizeof(ClothVertMidPush) == 80, "cloth vertex midphase blob must be 80 bytes");
static_assert(offsetof(ClothVertMidPush, cmInputs) == 8, "cmInputs @8");
static_assert(offsetof(ClothVertMidPush, femClothes) == 48, "femClothes @48");
static_assert(offsetof(ClothVertMidPush, stackSizeBytes) == 56, "stackSizeBytes @56");
static_assert(offsetof(ClothVertMidPush, stackPtr) == 64, "stackPtr @64");
static_assert(offsetof(ClothVertMidPush, midphasePairsNum) == 72, "midphasePairsNum @72");

using p817mesh::V3;

// One BV32 child: either a leaf over [start, start+count) or an internal pointer.
struct ChildSpec
{
    bool internal; uint32_t a, b;    // leaf: (start, count).  internal: (childNodeIdx, 0)
    float mn[3], mx[3];
};
} // namespace p926vm

PXG_TEST(femClothSimMeshVertMidphase)
{
    using namespace p816dc;        // PxgShapeHost, CmInputHost
    using namespace p926vm;
    using p817mesh::V3;
    PxgSlangComputeDevice* dev = testDevice();

    // The mesh's placement -- the sibling's, deliberately, so a difference between the
    // two fixtures' results is a difference between the two KERNELS.
    const float SX = 2.0f, SY = 1.0f, SZ = 0.5f;
    const float TX = 10.0f, TY = 20.0f, TZ = 30.0f;
    const float QS = 0.70710678f;
    const float QX = 0.0f, QY = QS, QZ = 0.0f, QW = QS;   // 90 degrees about y
    auto quatRot = [&](float x, float y, float z) {
        const float vx = 2.0f * x, vy = 2.0f * y, vz = 2.0f * z;
        const float w2 = QW * QW - 0.5f;
        const float d2 = QX * vx + QY * vy + QZ * vz;
        return V3{ vx * w2 + (QY * vz - QZ * vy) * QW + QX * d2,
                   vy * w2 + (QZ * vx - QX * vz) * QW + QY * d2,
                   vz * w2 + (QX * vy - QY * vx) * QW + QZ * d2 };
    };
    auto toWorld = [&](V3 v) {
        const V3 r = quatRot(v.x * SX, v.y * SY, v.z * SZ);
        return F4{ r.x + TX, r.y + TY, r.z + TZ, 1.0f };
    };

    // The contact distance is summed over BOTH cache refs and NOT divided by the scale
    // (V5).  It is declared here, before the mesh, because one triangle's height is
    // defined AS it -- see TOUCH_PRIM below.
    const float CD_CLOTH = 0.05f, CD_MESH = 0.15f;
    const float contactDist = CD_CLOTH + CD_MESH;

    // ---- the cooked mesh, in VERTEX space --------------------------------------------
    std::vector<F4> mv;
    std::vector<TriIdx4> mi;
    auto pushTri = [&](F4 p0, F4 p1, F4 p2) {
        const uint32_t b = uint32_t(mv.size());
        mv.push_back(p0); mv.push_back(p1); mv.push_back(p2);
        mi.push_back(TriIdx4{ b, b + 1u, b + 2u, 0u });
    };
    // front-facing (n = +y) unit right triangle at x in [ox, ox+1], height y
    auto pushSlab = [&](float ox, float y) {
        pushTri(F4{ ox, y, 0, 0 }, F4{ ox, y, 1.0f, 0 }, F4{ ox + 1.0f, y, 0, 0 });
    };
    // prim 0: a DUMMY parked 1000 away and in NO leaf, so every primStartIndex is
    // NONZERO -- with a zero start `data >> 7` and `data >> 8` agree and a shift
    // mutant in the leaf decode is invisible.
    pushTri(F4{ 1000.0f, 0, 0, 0 }, F4{ 1000.0f, 0, 1.0f, 0 }, F4{ 1001.0f, 0, 0, 0 });
    // The slab family starts at x = 1, not x = 0: the origin has to stay clear for the
    // EXACTLY-TOUCHING query below, whose whole point is that its bound reaches 0.20 and
    // no further.
    const uint32_t V0_PRIM = uint32_t(mi.size());          // prim 1
    pushSlab(1.0f, 0.0f);
    // prim 2: BACK-FACING (n = -y) and CLOSER than prim 1.  The triangle sibling's
    // front-face gate drops it; this kernel has no such gate and must keep it.
    const uint32_t V0_BACK_PRIM = uint32_t(mi.size());     // prim 2
    pushTri(F4{ 1.0f, 0.06f, 0, 0 }, F4{ 2.0f, 0.06f, 0, 0 }, F4{ 1.0f, 0.06f, 1.0f, 0 });
    // The THIRD member of vertex 0's leaf, separated from its query in Z ALONE, so a leaf
    // overlap test that drops the z terms accepts it.  It goes here and not further down
    // because a leaf is a CONSECUTIVE run [start, start + count): pushing it after the next
    // leaf's stack put a foreign triangle inside this leaf's range, which is precisely what
    // the containment invariant below caught on its first run.
    const uint32_t V0_Z_PRIM = uint32_t(mi.size());
    pushTri(F4{ 1.0f, 0, 2.0f, 0 }, F4{ 1.0f, 0, 3.0f, 0 }, F4{ 2.0f, 0, 2.0f, 0 });
    const uint32_t V1_PRIM = uint32_t(mi.size());          // prims 3..8 -- SIX of them
    pushSlab(2.0f, 0.0f);
    for (uint32_t k = 0; k < 5u; ++k)
        pushSlab(2.0f, -0.02f * float(k + 1u));
    const uint32_t V2_PRIM = uint32_t(mi.size());  pushSlab(4.0f, 0.0f);
    const uint32_t V3_PRIM = uint32_t(mi.size());  pushSlab(6.0f, 0.0f);
    const uint32_t V4_PRIM = uint32_t(mi.size());  pushSlab(8.0f, 0.0f);
    // The prim IMMEDIATELY AFTER vertex 4's single-primitive leaf, in NO leaf of its own
    // and placed right on top of that leaf's query.  `if (lane < nbPrimitives)` is the only
    // thing that keeps lane 1 from reading it -- without the guard it becomes a pair.
    // prims 12..14 hang off an INTERNAL child, so the walk has to decode
    // bvDataChildNodeIndex and carry a node across two rounds of the stack.
    const uint32_t V5_PRIM = uint32_t(mi.size());                          // prims 12..14
    pushSlab(12.0f, 0.0f);
    pushSlab(12.0f, -0.03f);
    pushSlab(12.0f, 0.03f);
    // prim 15 sits in the internal node's OTHER leaf, far away: the second-level box
    // test has to reject it, or descending is indistinguishable from taking everything.
    pushSlab(8.0f, 0.01f);
    const uint32_t FAR_PRIM = uint32_t(mi.size());  pushSlab(40.0f, 0.0f);
    // prim 16 sits at EXACTLY the contact distance above the vertex-space ORIGIN, and its
    // query vertex is the origin itself.  That vertex is the one point whose world round
    // trip is bit-exact -- toWorld(0) is the translation and nothing else, so the kernel
    // recovers (0,0,0) with no rounding at all -- which is what makes `>` and `>=` in the
    // leaf overlap test distinguishable.  Anywhere else the round trip through a
    // translation of 20 loses the last bits and the two spellings agree.
    const uint32_t TOUCH_PRIM = uint32_t(mi.size());
    pushTri(F4{ -0.5f, contactDist, -0.5f, 0 }, F4{ -0.5f, contactDist, 0.5f, 0 },
            F4{ 0.5f, contactDist, -0.5f, 0 });
    // A SECOND leaf over vertex 5's query, beside the internal child, so that ONE query
    // puts two entries on the node stack.  Without it every query in this fixture hits
    // exactly one child, and then popping the stack in the wrong order -- or resetting
    // `nbStack` instead of accumulating it -- changes nothing.
    const uint32_t V5B_PRIM = uint32_t(mi.size());  pushSlab(12.0f, 0.02f);
    // Two triangles whose THIRD vertex carries the extreme, one for each end.  Every
    // triangle above is wound so that b0 and b1 already bracket b2, which makes
    // `min(b0, b1)` and `min(min(b0, b1), b2)` agree -- the winding, not the arithmetic,
    // was hiding the mutant.
    const uint32_t MINB2_PRIM = uint32_t(mi.size());                       // b2 has min x
    pushTri(F4{ 17.0f, 0, 0, 0 }, F4{ 17.0f, 0, 0.5f, 0 }, F4{ 16.0f, 0, 0, 0 });
    const uint32_t MAXB2_PRIM = uint32_t(mi.size());                       // b2 has max x
    pushTri(F4{ 20.0f, 0, 0, 0 }, F4{ 20.0f, 0, 0.5f, 0 }, F4{ 21.0f, 0, 0, 0 });

    const uint32_t NPRIM = uint32_t(mi.size());
    const uint32_t NVERT = uint32_t(mv.size());

    // ---- the tree: root with SIX children, one of them internal ----------------------
    // Six, not eight, so lanes 6..31 have `hasBox` false and mNbNodes is load-bearing.
    // Every leaf box is TIGHT IN Y around its own triangles, which is what isolates the
    // queries from one another -- and what makes the 0.28 vertex depend on the band.
    // The INTERNAL child is child 6 and the leaf that shares vertex 5's query is child 5,
    // deliberately in that order: children are pushed in lane order, so the internal one
    // lands ON TOP of the stack and is expanded while another entry is still pending.
    // That is what makes `nbStack = validCount` differ from `nbStack += validCount`.
    const ChildSpec root[10] = {
        { false, V0_PRIM,     3u, { 1.0f,  0.00f, 0.0f }, { 2.0f,  0.06f, 3.0f } },
        { false, V1_PRIM,     6u, { 2.0f, -0.10f, 0.0f }, { 3.0f,  0.00f, 1.0f } },
        { false, V2_PRIM,     1u, { 4.0f,  0.00f, 0.0f }, { 5.0f,  0.00f, 1.0f } },
        { false, V3_PRIM,     1u, { 6.0f,  0.00f, 0.0f }, { 7.0f,  0.00f, 1.0f } },
        { false, V4_PRIM,     1u, { 8.0f,  0.00f, 0.0f }, { 9.0f,  0.00f, 1.0f } },
        { false, V5B_PRIM,    1u, { 12.0f, 0.02f, 0.0f }, { 13.0f, 0.02f, 1.0f } },
        { true,  1u,          0u, { 12.0f, -0.03f, 0.0f }, { 41.0f, 0.03f, 1.0f } },
        // deliberately LOOSE in y, so the NODE box test is a clear accept and the exactly
        // touching comparison lives only in the leaf predicate this mutant class targets
        { false, TOUCH_PRIM,  1u, { -0.5f, -0.10f, -0.5f }, { 0.5f, contactDist + 0.1f, 0.5f } },
        { false, MINB2_PRIM,  1u, { 16.0f, 0.00f, 0.0f }, { 17.0f, 0.00f, 0.5f } },
        { false, MAXB2_PRIM,  1u, { 20.0f, 0.00f, 0.0f }, { 21.0f, 0.00f, 0.5f } },
    };
    const ChildSpec node1[2] = {
        { false, V5_PRIM,  3u, { 12.0f, -0.03f, 0.0f }, { 13.0f, 0.03f, 1.0f } },
        { false, FAR_PRIM, 1u, { 40.0f,  0.00f, 0.0f }, { 41.0f, 0.00f, 1.0f } },
    };

    const uint32_t NODE_WORDS = 292u;   // 1168 bytes
    std::vector<uint32_t> blob(4u + 2u * NODE_WORDS, 0u);
    auto writeNode = [&](uint32_t nodeIdx, const ChildSpec* cs, uint32_t n) {
        const uint32_t base = 4u + nodeIdx * NODE_WORDS;
        for (uint32_t c = 0; c < n; ++c)
        {
            for (uint32_t a = 0; a < 3u; ++a)
            {
                blob[base + c * 4u + a] = f2u(cs[c].mn[a]);
                blob[base + 128u + c * 4u + a] = f2u(cs[c].mx[a]);
            }
            // leaf:     isLeaf(1) | nbPrimitives<<1 | primStartIndex<<7
            // internal: isLeaf(0) | childNodeIndex<<GU_BV4_CHILD_OFFSET_SHIFT_COUNT(11)
            blob[base + 256u + c] = cs[c].internal ? (cs[c].a << 11)
                                                   : (1u | (cs[c].b << 1) | (cs[c].a << 7));
        }
        blob[base + 288u] = n;    // mNbNodes
        blob[base + 289u] = 1u;   // mDepth
    };
    writeNode(0u, root, 10u);
    writeNode(1u, node1, 2u);

    // THE TREE INVARIANT, checked rather than assumed.
    //
    // Every child box must CONTAIN its subtree.  A cooked BV32 always does, and the whole
    // node-box test rests on it: given containment, dropping an axis from that test can
    // only cause extra DESCENTS -- a primitive whose own bound clears the query cannot be
    // hiding inside a box that clears it too.  That is why a mutant which LOOSENS the node
    // test (say, deleting its y terms) is genuinely equivalent here, while one that
    // TIGHTENS it (comparing against the box's min twice) is killed by this fixture.
    //
    // The equivalence is therefore an equivalence UNDER THIS INVARIANT, and an invariant
    // that is only believed is not one.  If a later edit makes a box too small, the node
    // test starts deciding outcomes, the "loosening is invisible" reasoning stops holding,
    // and this check is what says so.
    {
        auto contains = [&](const ChildSpec& box, const ChildSpec& inner) {
            for (uint32_t a = 0; a < 3u; ++a)
                if (inner.mn[a] < box.mn[a] || inner.mx[a] > box.mx[a]) return false;
            return true;
        };
        auto primBox = [&](uint32_t start, uint32_t count) {
            ChildSpec b = {}; 
            for (uint32_t a = 0; a < 3u; ++a) { b.mn[a] = 3.4e38f; b.mx[a] = -3.4e38f; }
            for (uint32_t k = 0; k < count; ++k)
            {
                const uint32_t idx[3] = { mi[start + k].x, mi[start + k].y, mi[start + k].z };
                for (uint32_t t = 0; t < 3u; ++t)
                {
                    const float c[3] = { mv[idx[t]].x, mv[idx[t]].y, mv[idx[t]].z };
                    for (uint32_t a = 0; a < 3u; ++a)
                    { b.mn[a] = std::fmin(b.mn[a], c[a]); b.mx[a] = std::fmax(b.mx[a], c[a]); }
                }
            }
            return b;
        };
        for (uint32_t k = 0; k < 10u; ++k)
        {
            if (!root[k].internal)
            {
                PXG_CHECK_MSG(contains(root[k], primBox(root[k].a, root[k].b)),
                              "root child %u does not contain its own primitives", k);
                continue;
            }
            for (uint32_t j = 0; j < 2u; ++j)
            {
                PXG_CHECK_MSG(contains(root[k], node1[j]),
                              "root child %u does not contain node 1's child %u", k, j);
                PXG_CHECK_MSG(contains(node1[j], primBox(node1[j].a, node1[j].b)),
                              "node 1 child %u does not contain its own primitives", j);
            }
        }
    }

    const size_t vBase = blob.size();
    blob.resize(vBase + NVERT * 4u, 0u);
    std::memcpy(blob.data() + vBase, mv.data(), NVERT * sizeof(F4));
    const size_t iBase = blob.size();
    blob.resize(iBase + NPRIM * 4u, 0u);
    std::memcpy(blob.data() + iBase, mi.data(), NPRIM * sizeof(TriIdx4));
    blob[0] = NVERT;
    blob[1] = NPRIM;
    // header.z is nbBv32TreeNodes, header.w is nbPackedNodes.  Only .w sizes the node
    // array, so .z is a deliberately wrong 9: a port that sizes with it walks the vertex
    // pointer seven nodes too far and reads garbage.  The sibling makes the same point.
    blob[2] = 9u;
    blob[3] = 2u;

    DevBuf bBlob = makeDev(dev, blob);
    if (!bBlob.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // ---- the cloth: ONE VERTEX per case, and no triangles at all ---------------------
    // This kernel reads mPosition_InvMass and mNbVerts and nothing else -- the triangle
    // indices the sibling needs are not even bound.
    std::vector<V3> vs = {
        { 1.5f,  0.12f, 0.5f },   // 0: one leaf, THREE triangles -- one back-facing, one
                                  //    separated in Z ALONE
        { 2.5f,  0.00f, 0.5f },   // 1: SIX qualifying triangles in one leaf
        { 4.5f,  0.28f, 0.5f },   // 2: outside 0.20, inside the scale-divided 0.40
        { 6.5f,  0.45f, 0.5f },   // 3: outside either reading
        { 8.5f,  0.02f, 0.5f },   // 4: just inside, and its leaf's NEXT primitive is a
                                  //    decoy only the lane guard keeps out
        { 12.5f, 0.05f, 0.5f },   // 5: TWO children -- a leaf and the internal one
        { 0.0f,  0.00f, 0.0f },   // 6: the ORIGIN -- a triangle sits at EXACTLY 0.20 above
        { 15.85f, 0.00f, 0.1f },  // 7: reaches a triangle only via its THIRD vertex's min x
        { 21.15f, 0.00f, 0.1f },  // 8: reaches a triangle only via its THIRD vertex's max x
    };
    // The vertex loop strides by blockDim.y * gridDim.x = 2048.  With a handful of
    // vertices no index ever advances, so a stride of 1024 -- the grid width alone, and the
    // obvious wrong reading -- is indistinguishable from the right one.  The cloth is
    // therefore padded past 1024: everything from PAD_START up sits on vertex 4's query and
    // would be visited TWICE by the halved stride, and the rest is parked where nothing
    // overlaps.
    const uint32_t NB_CASES = uint32_t(vs.size());
    const uint32_t PAD_START = 1024u, NB_VERTS = 1100u;
    for (uint32_t i = NB_CASES; i < NB_VERTS; ++i)
        vs.push_back(i < PAD_START ? V3{ 500.0f, 0.0f, 0.0f } : V3{ 8.5f, 0.02f, 0.5f });
    std::vector<F4> cv;
    for (const V3& v : vs) cv.push_back(toWorld(v));
    DevBuf bCv = makeDev(dev, cv);

    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bCv.addr));
    c.mNbVerts = NB_VERTS;
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    // The TRIMESH is shapeRef0 and the CLOTH shapeRef1 -- the SWAPPED order, so the
    // cloth can only be found by its body id (both shapes are eTRIANGLEMESH).
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

    std::vector<float> xf(2 * 8, 0.0f);
    xf[3] = 1.0f;                                                    // slot 0 identity
    xf[8 + 0] = QX; xf[8 + 1] = QY; xf[8 + 2] = QZ; xf[8 + 3] = QW;
    xf[8 + 4] = TX; xf[8 + 5] = TY; xf[8 + 6] = TZ;
    DevBuf bXf = makeDev(dev, xf);

    std::vector<float> cd = { CD_CLOTH, CD_MESH };
    DevBuf bCd = makeDev(dev, cd);

    // ---- CPU reference: the SAME walk, node by node ----------------------------------
    auto meshVert = [&](uint32_t i) { return V3{ mv[i].x, mv[i].y, mv[i].z }; };
    auto boxHit = [&](const float* mn, const float* mx, V3 qn, V3 qx) {
        return !(mn[0] > qx.x || qn.x > mx[0] || mn[1] > qx.y || qn.y > mx[1] ||
                 mn[2] > qx.z || qn.z > mx[2]);
    };
    // `dropNodeY` runs the walk with the y terms removed from the NODE box test, and
    // nothing else changed -- see the note under the results for what it is for.
    auto walk = [&](bool dropNodeY) {
        std::vector<std::pair<uint32_t, uint32_t>> out;   // (vertexIdx, prim)
        for (uint32_t v = 0; v < NB_VERTS; ++v)
        {
            const V3 ctr = vs[v];                              // the round trip is identity
            const V3 qn{ ctr.x - contactDist, ctr.y - contactDist, ctr.z - contactDist };
            const V3 qx{ ctr.x + contactDist, ctr.y + contactDist, ctr.z + contactDist };
            auto nodeHit = [&](const ChildSpec& c) {
                if (!dropNodeY) return boxHit(c.mn, c.mx, qn, qx);
                return !(c.mn[0] > qx.x || qn.x > c.mx[0] || c.mn[2] > qx.z || qn.z > c.mx[2]);
            };
            auto visit = [&](const ChildSpec* cs, uint32_t n, auto&& self) -> void {
                for (uint32_t k = 0; k < n; ++k)
                {
                    if (!nodeHit(cs[k])) continue;
                    if (cs[k].internal) { self(node1, 2u, self); continue; }
                    for (uint32_t lane = 0; lane < cs[k].b && lane < 32u; ++lane)
                    {
                        const uint32_t p = cs[k].a + lane;
                        const V3 b0 = meshVert(mi[p].x), b1 = meshVert(mi[p].y),
                                 b2 = meshVert(mi[p].z);
                        const float tn[3] = { std::fmin(std::fmin(b0.x, b1.x), b2.x),
                                              std::fmin(std::fmin(b0.y, b1.y), b2.y),
                                              std::fmin(std::fmin(b0.z, b1.z), b2.z) };
                        const float tx[3] = { std::fmax(std::fmax(b0.x, b1.x), b2.x),
                                              std::fmax(std::fmax(b0.y, b1.y), b2.y),
                                              std::fmax(std::fmax(b0.z, b1.z), b2.z) };
                        if (boxHit(tn, tx, qn, qx)) out.push_back({ v, p });
                    }
                }
            };
            visit(root, 10u, visit);
        }
        std::sort(out.begin(), out.end());
        return out;
    };
    const std::vector<std::pair<uint32_t, uint32_t>> ref = walk(false);

    // THE ONE SURVIVOR OF THE SWEEP, EXECUTED RATHER THAN ARGUED.
    //
    // Deleting the y terms from the NODE box test survives, and the reason is real: a
    // cooked BV32 child box CONTAINS its subtree, so loosening the node test can only add
    // DESCENTS, never emissions -- a primitive whose own bound clears the query cannot hide
    // inside a box that clears it too.  Tightening it is a different matter and IS killed
    // (comparing against the box's min twice).
    //
    // But that is an argument, and P9.24 paid four sweeps to learn that an argument about a
    // mutant can be wrong where running it cannot.  So the walk above runs the mutant and
    // this asserts the two agree.  It is not a tautology: it holds because the boxes
    // contain their subtrees, and the moment an edit breaks that -- which the containment
    // check above would also report -- these two sets diverge and the verdict "equivalent"
    // is withdrawn by the fixture rather than by anyone remembering to re-derive it.
    PXG_CHECK_MSG(walk(true) == ref,
                  "dropping the y terms from the NODE box test changes the emitted pair set, "
                  "so it is NOT the equivalent this fixture records it as.  Some child box no "
                  "longer contains its subtree; add the mutant back to the sweep list");

    // ---- run 1: a capacity that cannot bind -----------------------------------------
    const uint32_t CAP = 256u;
    const uint32_t SENTINEL = 0xFFFFFFFFu;
    std::vector<uint32_t> stack(CAP * 4u, SENTINEL);
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> pairNum = { 0u };
    DevBuf bPairNum = makeDev(dev, pairNum);

    ClothVertMidPush push = {};
    push.numWorkItems = 1u;
    push.toleranceLength = 1.0f;
    push.cmInputs = bCms.addr;
    push.transformCache = bXf.addr;
    push.bounds = 0ull;                 // ABI parity only -- the kernel never reads it
    push.contactDistance = bCd.addr;
    push.gpuShapes = bShapes.addr;
    push.femClothes = bDesc.addr;
    push.stackSizeBytes = CAP * 16u;
    push.stackPtr = bStack.addr;
    push.midphasePairsNum = bPairNum.addr;

    Kernel k("femClothMeshVertMidphaseBda.spv", "cloth_midphaseVertexMeshLaunch",
             {}, sizeof(ClothVertMidPush));
    // The engine's grid (PxgNarrowphaseCore.cpp:6742): x = SB_SBMIDPHASE = 1024,
    // y = numTests.  One dispatched group is ONE warp; the block's two are serial.
    k.dispatch(/*gridX*/ 1024u, {}, push, /*gridY*/ 1u, 1u);
    dev->flush();

    const uint32_t gNum = downloadBuffer<uint32_t>(dev, bPairNum.h, 1)[0];
    const auto gStack = downloadBuffer<uint32_t>(dev, bStack.h, CAP * 4u);
    std::vector<std::pair<uint32_t, uint32_t>> got;
    for (uint32_t i = 0; i < gNum && i < CAP; ++i)
    {
        PXG_CHECK_MSG(gStack[i * 4u + 0u] == 0u,
                      "pair %u carries cmIdx %u; blockIdx.y is the ONLY contact-manager "
                      "index here", i, gStack[i * 4u + 0u]);
        PXG_CHECK_MSG(gStack[i * 4u + 3u] == 0u,
                      "pair %u has a non-zero fourth word; the CUDA writes make_uint4(..., 0)", i);
        got.push_back({ gStack[i * 4u + 1u], gStack[i * 4u + 2u] });
    }
    std::sort(got.begin(), got.end());

    auto countFor = [&](uint32_t v) {
        uint32_t n = 0; for (auto& e : got) if (e.first == v) ++n; return n;
    };
    auto has = [&](uint32_t v, uint32_t p) {
        return std::find(got.begin(), got.end(), std::make_pair(v, p)) != got.end();
    };

    const uint32_t PAD_PAIRS = NB_VERTS - PAD_START;      // one each, on vertex 4's query
    PXG_CHECK_MSG(ref.size() == 16u + PAD_PAIRS,
                  "reference: expected %u pairs -- vertex 0 two, vertex 1 six, vertex 4 one, "
                  "vertex 5 four, vertices 6, 7 and 8 one each, none for 2 and 3, plus %u from "
                  "the padding -- got %zu", 16u + PAD_PAIRS, PAD_PAIRS, ref.size());
    PXG_CHECK_MSG(gNum == uint32_t(ref.size()),
                  "emitted %u pairs, reference says %zu.  0 usually means the blob walk or "
                  "the vertex-space transform is wrong; 10 means a borrowed top-4 cap; "
                  "more means the contact distance picked up the softbody path's scale "
                  "division (vertex 2 comes inside a 0.40 band)", gNum, ref.size());
    PXG_CHECK_MSG(got == ref, "the emitted pair SET does not match the CPU reference");

    // Pin what the twelve ARE, so a reference that drifts with the shader is caught too.
    PXG_CHECK_MSG(countFor(0u) == 2u && has(0u, V0_PRIM) && has(0u, V0_BACK_PRIM),
                  "vertex 0 must keep BOTH triangles in its leaf, including the BACK-FACING "
                  "one -- the triangle sibling's front-face gate would drop it and this "
                  "kernel has no gate at all (got %u)", countFor(0u));
    PXG_CHECK_MSG(countFor(1u) == 6u,
                  "vertex 1 must keep ALL SIX qualifying triangles: pushOntoStackFullWarp has "
                  "no reduction, and a borrowed MaxNumPairs = 4 gives four (got %u)",
                  countFor(1u));
    PXG_CHECK_MSG(countFor(2u) == 0u,
                  "vertex 2 sits 0.28 away: outside the correct 0.20 band and inside the 0.40 "
                  "the softbody path's scale division would give");
    PXG_CHECK_MSG(countFor(3u) == 0u, "vertex 3 sits 0.45 away, outside either reading");
    PXG_CHECK_MSG(countFor(4u) == 1u && has(4u, V4_PRIM),
                  "vertex 4 sits 0.02 away and must match its own leaf's triangle");
    PXG_CHECK_MSG(countFor(5u) == 4u && has(5u, V5_PRIM) && has(5u, V5_PRIM + 1u) &&
                  has(5u, V5_PRIM + 2u) && has(5u, V5B_PRIM),
                  "vertex 5 must reach BOTH of its children: the leaf beside the internal one "
                  "AND the three triangles behind it.  Two entries on the node stack at once is "
                  "what makes the pop order and the `nbStack +=` accumulation observable; three "
                  "means the internal child was never expanded, one means the stack was reset "
                  "(got %u)", countFor(5u));
    PXG_CHECK_MSG(countFor(6u) == 1u && has(6u, TOUCH_PRIM),
                  "vertex 6 sits at the vertex-space ORIGIN and prim 16 at EXACTLY the contact "
                  "distance above it, so the leaf bound TOUCHES: `tMin.y > qMax.y` is false and "
                  "the pair is kept, where `>=` would drop it.  This is the only query in the "
                  "fixture whose world round trip is bit-exact, and therefore the only one that "
                  "can tell the two spellings apart (got %u)", countFor(6u));
    PXG_CHECK_MSG(!has(5u, FAR_PRIM),
                  "the internal child's OTHER leaf sits at x = 40 and its box must reject the "
                  "query -- descending is otherwise indistinguishable from taking everything");
    PXG_CHECK_MSG(!has(0u, V0_Z_PRIM),
                  "vertex 0's leaf holds a triangle separated from its query in Z ALONE; if it "
                  "comes out, the leaf overlap test is not testing all three axes");
    PXG_CHECK_MSG(countFor(4u) == 1u && !has(4u, V4_PRIM + 1u),
                  "the primitive after vertex 4's single-primitive leaf sits right on that "
                  "leaf's query and belongs to no leaf.  Only `lane < nbPrimitives` keeps it "
                  "out -- a raw `lane + primStartIndex` for all 32 lanes emits it");
    PXG_CHECK_MSG(countFor(7u) == 1u && has(7u, MINB2_PRIM),
                  "vertex 7 reaches its triangle only through that triangle's THIRD vertex, "
                  "which carries the minimum x: `min(b0, b1)` alone puts the bound 1.0 too far "
                  "right and the pair disappears (got %u)", countFor(7u));
    PXG_CHECK_MSG(countFor(8u) == 1u && has(8u, MAXB2_PRIM),
                  "vertex 8 reaches its triangle only through that triangle's THIRD vertex, "
                  "which carries the maximum x: `max(b0, b1)` alone puts the bound 1.0 too far "
                  "left and the pair disappears (got %u)", countFor(8u));
    uint32_t padSeen = 0;
    for (uint32_t v = PAD_START; v < NB_VERTS; ++v)
    {
        PXG_CHECK_MSG(countFor(v) == 1u,
                      "padding vertex %u produced %u pairs, expected exactly one.  Two means "
                      "the vertex loop strides by the grid width instead of "
                      "blockDim.y * gridDim.x and visits it twice", v, countFor(v));
        ++padSeen;
    }
    PXG_CHECK_MSG(padSeen == PAD_PAIRS && PAD_START >= 1024u,
                  "the padding must start at or past 1024, or a halved vertex stride is "
                  "unobservable");
    for (uint32_t i = gNum; i < CAP; ++i)
        PXG_CHECK_MSG(gStack[i * 4u + 0u] == SENTINEL,
                      "slot %u past the emitted count was written", i);

    // ---- run 2: a capacity that DOES bind -------------------------------------------
    // "If we run out of stacksize, drop contacts" -- but the COUNTER still advances past
    // the capacity, because the engine reads it back to size next frame's stack.  The
    // buffer keeps its 64 slots and only the kernel's bound moves, so a mutant that drops
    // the `stackIndex < stackCapacity` guard writes into slack and is caught here rather
    // than corrupting the heap.
    // FOUR, not six: the largest single ballot in this fixture pushes SIX (vertex 1's
    // leaf), so a capacity of four is the only setting under which clamping the atomicAdd
    // to the room left -- rather than counting what was actually wanted -- shows up.
    const uint32_t TIGHT = 4u;
    std::vector<uint32_t> stack2(CAP * 4u, SENTINEL);
    DevBuf bStack2 = makeDev(dev, stack2);
    std::vector<uint32_t> pairNum2 = { 0u };
    DevBuf bPairNum2 = makeDev(dev, pairNum2);
    push.stackSizeBytes = TIGHT * 16u;
    push.stackPtr = bStack2.addr;
    push.midphasePairsNum = bPairNum2.addr;
    k.dispatch(/*gridX*/ 1024u, {}, push, /*gridY*/ 1u, 1u);
    dev->flush();

    const uint32_t gNum2 = downloadBuffer<uint32_t>(dev, bPairNum2.h, 1)[0];
    const auto gStack2 = downloadBuffer<uint32_t>(dev, bStack2.h, CAP * 4u);
    PXG_CHECK_MSG(gNum2 == uint32_t(ref.size()),
                  "with a capacity of %u the counter must still reach %zu -- the engine reads "
                  "it back to size next frame's stack, so clamping it there loses the growth "
                  "signal (got %u)", TIGHT, ref.size(), gNum2);
    uint32_t written = 0;
    for (uint32_t i = 0; i < CAP; ++i)
        if (gStack2[i * 4u + 0u] != SENTINEL) ++written;
    PXG_CHECK_MSG(written == TIGHT,
                  "with a capacity of %u exactly %u slots may be written; %u were.  More means "
                  "the `stackIndex < stackCapacity` guard is gone", TIGHT, TIGHT, written);
    for (uint32_t i = TIGHT; i < CAP; ++i)
        PXG_CHECK_MSG(gStack2[i * 4u + 0u] == SENTINEL,
                      "slot %u is past the capacity and was written anyway", i);

    std::printf("    [ note ] %u verts (%u cases + %u padding past 1024), %u prims over "
                "2 nodes (10 + 2 children, one internal); %u pairs: v0 %u (one back-facing, "
                "one z-only reject), v1 %u (no top-4), v4 %u, v5 %u (leaf + internal), "
                "v6 %u (exactly touching), v7/v8 %u/%u (third-vertex extremes); capacity %u "
                "writes %u and still counts %u\n",
                NB_VERTS, NB_CASES, PAD_PAIRS, NPRIM, gNum, countFor(0u), countFor(1u),
                countFor(4u), countFor(5u), countFor(6u), countFor(7u), countFor(8u),
                TIGHT, written, gNum2);

    dev->destroyBuffer(bBlob.h); dev->destroyBuffer(bCv.h); dev->destroyBuffer(bDesc.h);
    dev->destroyBuffer(bShapes.h); dev->destroyBuffer(bCms.h); dev->destroyBuffer(bXf.h);
    dev->destroyBuffer(bCd.h); dev->destroyBuffer(bStack.h); dev->destroyBuffer(bPairNum.h);
    dev->destroyBuffer(bStack2.h); dev->destroyBuffer(bPairNum2.h);
}

// ===========================================================================
// P9.26 -- cloth_meshVertexContactGenLaunch, the consumer of the midphase above.
//
// THIS KERNEL EMITS NO CONTACTS, AND THAT IS THE ENGINE'S DOING.  The CUDA computes
// a closest point, a normal, a penetration and a barycentric for every pair the
// midphase produced and then throws all of it away: the write is gated on
// `hasContacts`, and the only assignment to `hasContacts` in the kernel is commented
// out (femClothPrimitives.cu:3056).  PxgOutputIndex (:79) skips its atomicAdd on an
// empty ballot, so the contact COUNTER does not move either.  The kernel's entire
// observable output is one atomicMax that sizes next frame's intermediate stack.
//
// So this fixture cannot check a contact, and pretending otherwise would be the whole
// mistake.  What it CAN check is every property of the one live output, and each of
// the four below is a mutant that a "just call atomicMax somewhere" port would pass:
//
//   * the count is UNCLAMPED.  `numPairs` is min(found, stackSize) and drives the
//     loop; the atomicMax takes `numMidphasePairsFound`, the raw value.  Run 1 asks
//     for 37 pairs into a 16-deep stack -- the clamped reading gives 256 where the
//     right one gives 592.  This is the whole point of the output: the engine reads
//     it back to grow a stack that was too small, so a clamped value can never ask
//     for more than it already has.
//   * it is a MAX, not a store.  Run 2 pre-seeds a larger value and it must survive.
//   * the unit is BYTES.  sizeof(uint4) = 16, not 1 and not 4.
//   * ONE thread of the grid does it (`globalThreadIdx == 0`), across 512 blocks.
//
// and, negatively, that nothing else moves: the contact counter is pre-seeded with a
// value the kernel must not touch, and every writer buffer carries a sentinel.
// ===========================================================================
namespace p926cg
{
struct ClothVertCgPush
{
    float toleranceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, restDistance, gpuShapes, femClothes;
    uint64_t shapeToRigidRemap;
    uint32_t stackSizeBytes, pad68;
    uint64_t stackPtr, midphasePairsNum, stackSizeNeeded;
    uint64_t w_outPoint, w_outNormalPen, w_outBarycentric, w_outContactInfo, w_totalContactCount;
    uint64_t w_contactByX, w_tempContactByX, w_contactIndexSortedByX, w_contactSortedByX;
    uint32_t w_maxNumContacts, pad172;
};
static_assert(sizeof(ClothVertCgPush) == 176, "cloth vertex CG blob must be 176 bytes");
static_assert(offsetof(ClothVertCgPush, stackSizeBytes) == 64, "stackSizeBytes @64");
static_assert(offsetof(ClothVertCgPush, stackPtr) == 72, "stackPtr @72");
static_assert(offsetof(ClothVertCgPush, midphasePairsNum) == 80, "midphasePairsNum @80");
static_assert(offsetof(ClothVertCgPush, stackSizeNeeded) == 88, "stackSizeNeeded @88");
// The writer is passed BY VALUE and its paramAlign is 16 (P9.15e).  @96 is already
// 16-aligned, so nothing moves -- asserted rather than assumed, because the next
// kernel with a by-value tail will not be so lucky.
static_assert(offsetof(ClothVertCgPush, w_outPoint) == 96, "writer @96, 16-aligned");
static_assert(offsetof(ClothVertCgPush, w_maxNumContacts) == 168, "maxNumContacts @168");
} // namespace p926cg

PXG_TEST(femClothSimMeshVertContactGen)
{
    using namespace p926cg;
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t SENTINEL32 = 0xFEEDFACEu;
    const uint32_t SEED_COUNT = 7u;            // the contact counter must not move
    const uint32_t NB_FOUND = 37u;             // what the midphase says it produced
    const uint32_t STACK_PAIRS = 16u;          // ... into a stack that holds sixteen

    WriterBufs w = makeWriter(dev, 64u);
    std::vector<uint32_t> seedCount = { SEED_COUNT };
    DevBuf bCount = makeDev(dev, seedCount);
    if (!bCount.addr) { std::printf("  NOTE: no BDA -> skip\n"); destroyWriter(dev, w); return; }

    // The pair stack the kernel would read.  Its contents are irrelevant -- nothing
    // this kernel computes can reach an output -- but it is bound and sentinel-filled
    // so a port that WRITES to the pair stack (the midphase's buffer, not its own) is
    // caught rather than silently corrupting it.
    std::vector<uint32_t> pairs(STACK_PAIRS * 4u, SENTINEL32);
    DevBuf bPairs = makeDev(dev, pairs);
    std::vector<uint32_t> found = { NB_FOUND };
    DevBuf bFound = makeDev(dev, found);
    std::vector<uint32_t> needed = { 0u };
    DevBuf bNeeded = makeDev(dev, needed);

    ClothVertCgPush p = {};
    p.toleranceLength = 1.0f;
    p.stackSizeBytes = STACK_PAIRS * 16u;
    p.stackPtr = bPairs.addr;
    p.midphasePairsNum = bFound.addr;
    p.stackSizeNeeded = bNeeded.addr;
    p.w_outPoint = w.point.addr; p.w_outNormalPen = w.normalPen.addr;
    p.w_outBarycentric = w.barycentric.addr; p.w_outContactInfo = w.info.addr;
    p.w_totalContactCount = bCount.addr;
    p.w_contactByX = w.byRigid.addr; p.w_tempContactByX = w.tempByRigid.addr;
    p.w_contactIndexSortedByX = w.idxByRigid.addr; p.w_contactSortedByX = w.sortedByRigid.addr;
    p.w_maxNumContacts = w.maxContacts;
    // cmInputs, transformCache, contactDistance, restDistance, gpuShapes, femClothes and
    // shapeToRigidRemap stay null.  They are in the pack for ABI parity and the kernel
    // reads none of them -- which is itself a claim, and a null address is how it gets
    // tested: a port that transcribes the dead geometry faults here instead of passing.

    Kernel k("femClothMeshVertContactGenBda.spv", "cloth_meshVertexContactGenLaunch",
             {}, sizeof(ClothVertCgPush));
    k.dispatch(/*gridX*/ 512u, {}, p, 1u, 1u);   // PxgNarrowphaseCore.cpp:6831
    dev->flush();

    const uint32_t need1 = downloadBuffer<uint32_t>(dev, bNeeded.h, 1)[0];
    PXG_CHECK_MSG(need1 == NB_FOUND * 16u,
                  "stackSizeNeeded is %u, expected %u = %u found * 16 bytes.  %u would mean the "
                  "count was clamped to the stack depth first -- and a value that cannot exceed "
                  "the stack it is meant to grow is useless; %u would mean the unit is pairs, "
                  "not bytes", need1, NB_FOUND * 16u, NB_FOUND, STACK_PAIRS * 16u, NB_FOUND);

    const uint32_t cnt1 = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
    PXG_CHECK_MSG(cnt1 == SEED_COUNT,
                  "the contact counter moved from %u to %u.  It must not: every lane's "
                  "`hasContacts` is false, so the ballot is empty and PxgOutputIndex performs "
                  "no atomicAdd at all (femClothPrimitives.cu:89 gates it on popcount > 0)",
                  SEED_COUNT, cnt1);

    const auto gPoint = downloadBuffer<F4>(dev, w.point.h, w.maxContacts);
    const auto gNp = downloadBuffer<F4>(dev, w.normalPen.h, w.maxContacts);
    const auto gBary = downloadBuffer<F4>(dev, w.barycentric.h, w.maxContacts);
    for (uint32_t i = 0; i < w.maxContacts; ++i)
        PXG_CHECK_MSG(gPoint[i].x == 0.0f && gPoint[i].w == 0.0f &&
                      gNp[i].x == 0.0f && gNp[i].w == 0.0f &&
                      gBary[i].x == 0.0f && gBary[i].w == 0.0f,
                      "contact slot %u was written.  This kernel emits NO contacts upstream; if "
                      "that has changed, tools/check_disabled_kernels.py will say so and the "
                      "geometry has to be ported for real", i);
    const auto gPairs = downloadBuffer<uint32_t>(dev, bPairs.h, STACK_PAIRS * 4u);
    for (uint32_t i = 0; i < STACK_PAIRS * 4u; ++i)
        PXG_CHECK_MSG(gPairs[i] == SENTINEL32,
                      "word %u of the midphase pair stack was overwritten; this kernel only "
                      "READS it", i);

    // ---- run 2: it is a MAX ----------------------------------------------------------
    const uint32_t BIGGER = NB_FOUND * 16u + 1000u;
    std::vector<uint32_t> needed2 = { BIGGER };
    DevBuf bNeeded2 = makeDev(dev, needed2);
    p.stackSizeNeeded = bNeeded2.addr;
    k.dispatch(512u, {}, p, 1u, 1u);
    dev->flush();
    const uint32_t need2 = downloadBuffer<uint32_t>(dev, bNeeded2.h, 1)[0];
    PXG_CHECK_MSG(need2 == BIGGER,
                  "stackSizeNeeded was %u before the launch and is %u after.  atomicMax must "
                  "leave a larger value alone -- a plain store would drop another contact "
                  "manager's larger request in the same frame", BIGGER, need2);

    // ---- run 3: nothing found, nothing asked for --------------------------------------
    std::vector<uint32_t> none = { 0u };
    DevBuf bNone = makeDev(dev, none);
    std::vector<uint32_t> needed3 = { 0u };
    DevBuf bNeeded3 = makeDev(dev, needed3);
    p.midphasePairsNum = bNone.addr;
    p.stackSizeNeeded = bNeeded3.addr;
    k.dispatch(512u, {}, p, 1u, 1u);
    dev->flush();
    const uint32_t need3 = downloadBuffer<uint32_t>(dev, bNeeded3.h, 1)[0];
    PXG_CHECK_MSG(need3 == 0u,
                  "with no pairs found, stackSizeNeeded must stay 0, got %u", need3);
    const uint32_t cnt3 = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
    PXG_CHECK_MSG(cnt3 == SEED_COUNT,
                  "the contact counter moved to %u across three launches", cnt3);

    std::printf("    [ note ] %u found into a %u-pair stack -> stackSizeNeeded %u bytes "
                "(unclamped); max keeps %u; empty keeps 0; contact count %u untouched over "
                "three launches; no contact slot and no pair-stack word written\n",
                NB_FOUND, STACK_PAIRS, need1, need2, cnt3);

    dev->destroyBuffer(bPairs.h); dev->destroyBuffer(bFound.h); dev->destroyBuffer(bNeeded.h);
    dev->destroyBuffer(bNeeded2.h); dev->destroyBuffer(bNone.h); dev->destroyBuffer(bNeeded3.h);
    dev->destroyBuffer(bCount.h);
    destroyWriter(dev, w);
}

// ===========================================================================
// P9.27 -- the cloth-vs-HEIGHTFIELD midphase pair, both entry points.
//
// The CUDA shares one __device__ core between them (femClothHFMidPhase.cu:76) and
// this port shares one Slang body, so the fixture drives BOTH names over the SAME
// heightfield and asserts each against its own CPU reference.  If the shared body
// were wrong in a way that happened to suit one element type, the other would show
// it.
//
// WHAT IT PINS, and each of these is a mutant that would otherwise live:
//
//   * the ELEMENT bound differs and nothing else does.  The triangle path takes
//     triBoundingBox fattened by the contact distance; the vertex path takes a
//     SPHERE bound of that distance around one position.  A vertex whose sphere
//     reaches a quad its triangle's bound does not, and a triangle whose bound
//     reaches one no single vertex does, separate the two;
//   * the y range reaches the quad cull through a SATURATING float->PxU32
//     conversion.  One element sits with its local bound below zero, which is the
//     case P9.27a showed nothing in the port could previously tell apart;
//   * eHOLE is compared against the material byte with the tessellation flag
//     MASKED OFF, and the two material slots of one quad are independent;
//   * the payload is uint4(cmIdx, elementIdx, triangleIdx, 0) with
//     triangleIdx = (vertexIndex << 1) + isMaterial1;
//   * *** THE DUPLICATION. ***  The engine launches gridDim.y = numTests and
//     neither kernel reads blockIdx.y, so every pair is emitted numTests times.
//     The second dispatch drives numTests = 2 with gridDim.y = 2 and requires
//     EXACTLY two copies of every pair.  That is upstream candidate 37, measured
//     rather than described -- and a port that "fixed" it would fail here, which
//     is the point.
// ===========================================================================
namespace p927hf
{
struct ClothHfMidPush
{
    uint32_t numNPWorkItems; float toleranceLength;
    uint64_t cmInputs, transformCache, contactDistance, gpuShapes, femClothes;
    uint64_t stackPtr, midphasePairsNum;
    uint32_t stackSizeBytes, pad68;
};
static_assert(sizeof(ClothHfMidPush) == 72, "cloth hf midphase blob must be 72 bytes");
static_assert(offsetof(ClothHfMidPush, femClothes) == 40, "femClothes @40");
static_assert(offsetof(ClothHfMidPush, stackPtr) == 48, "stackPtr @48");
static_assert(offsetof(ClothHfMidPush, midphasePairsNum) == 56, "midphasePairsNum @56");
static_assert(offsetof(ClothHfMidPush, stackSizeBytes) == 64, "stackSizeBytes @64 -- AFTER "
              "the output pointers, unlike the softbody pack where it precedes them");

static uint32_t hfSample(int16_t height, uint8_t mat0, uint8_t mat1)
{
    return (uint32_t(uint16_t(height))) | (uint32_t(mat0) << 16) | (uint32_t(mat1) << 24);
}
// cvt.rzi.u32.f32 -- see pxgSlangHeightfield and P9.27a.
static uint32_t f2uSatRef(float x)
{
    if (!(x > 0.0f)) return 0u;
    if (x >= 4294967296.0f) return 0xffffffffu;
    return uint32_t(x);
}
} // namespace p927hf

PXG_TEST(femClothSimHeightfieldMidphase)
{
    using namespace p816dc;        // PxgShapeHost, CmInputHost
    using namespace p927hf;
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t MAT_SOLID = 3u, MAT_HOLE = 127u;
    const uint32_t MAT_HOLE_TESS = 127u | 0x80u;   // eHOLE with the tessellation flag set
    // FOUR rows and FIVE columns.  A square field cannot tell nbRows from nbCols, and a
    // port that swapped them would index the samples identically -- measured, as a survivor.
    const uint32_t NB_ROWS = 4u, NB_COLS = 5u;

    // blob: nbRows @0, nbCols @4, then NB_ROWS * NB_COLS samples.
    std::vector<uint32_t> blob(2u + NB_ROWS * NB_COLS, 0u);
    blob[0] = NB_ROWS; blob[1] = NB_COLS;
    auto setSample = [&](uint32_t r, uint32_t c, int16_t h, uint8_t m0, uint8_t m1) {
        blob[2u + r * NB_COLS + c] = hfSample(h, m0, m1);
    };
    for (uint32_t r = 0; r < NB_ROWS; ++r)
        for (uint32_t c = 0; c < NB_COLS; ++c)
            setSample(r, c, 0, uint8_t(MAT_SOLID), uint8_t(MAT_SOLID));
    // quad (0,0): one slot a HOLE, the other solid -- the two slots are independent.
    setSample(0u, 0u, 0, uint8_t(MAT_HOLE), uint8_t(MAT_SOLID));
    // quad (0,1): a hole in BOTH slots, with the TESSELLATION flag set.  Comparing the
    // raw byte against eHOLE would miss it and emit two pairs.
    setSample(0u, 1u, 0, uint8_t(MAT_HOLE_TESS), uint8_t(MAT_HOLE_TESS));
    // rows 2 and 3 raised well above everything, so quads there are culled by con0.
    for (uint32_t c = 0; c < NB_COLS; ++c) { setSample(2u, c, 400, uint8_t(MAT_SOLID), uint8_t(MAT_SOLID)); }
    for (uint32_t c = 0; c < NB_COLS; ++c) { setSample(3u, c, 400, uint8_t(MAT_SOLID), uint8_t(MAT_SOLID)); }
    // quad (1,2)'s far corner is NEGATIVE, so the signed height read matters here.
    setSample(1u, 2u, -1, uint8_t(MAT_SOLID), uint8_t(MAT_SOLID));
    DevBuf bBlob = makeDev(dev, blob);
    if (!bBlob.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // ---- the heightfield's placement -------------------------------------------------
    // Non-unit scales on all three axes and a 180-degree rotation about y, which is exact
    // in floating point:  world = R * (hf * scale) + T,  R(x, y, z) = (-x, y, -z).
    const float ROW_SCALE = 2.0f, HEIGHT_SCALE = 0.01f, COL_SCALE = 0.5f;
    const float TX = 5.0f, TY = 7.0f, TZ = 9.0f;
    auto toWorld = [&](float hr, float hh, float hc) {
        return F4{ -(hr * ROW_SCALE) + TX, (hh * HEIGHT_SCALE) + TY, -(hc * COL_SCALE) + TZ, 1.0f };
    };

    // ---- the cloth ------------------------------------------------------------------
    // Vertices are placed in HEIGHTFIELD units and converted; triangles index them.
    // The vertex count is 9 -- deliberately NOT a multiple of 2048, so lanes past the end
    // of the array are exercised on every launch.  That is trap H1: the CUDA loads
    // positions[vertIndex] before its own bounds check, and on this platform an
    // unclamped index is a fault rather than a wasted load.
    struct HV { float r, h, c; };
    // The heights are in HEIGHTFIELD units and the contact distance of 0.1 world units
    // is 10 of them (heightScale 0.01), so a vertex at h = 0 carries a y band of
    // -10..+10 around the surface.  It has to STRADDLE the surface: `con1` culls an
    // element lying entirely above every corner, so a vertex parked at h = 30 over a
    // quad at height 0 generates nothing -- measured, on the first version of this
    // fixture, where "over quad (0,0)" produced no pair at all.
    std::vector<HV> vs = {
        { 0.5f,    0.0f, 0.5f },   // 0: on quad (0,0) -- one slot is a hole
        { 0.5f,    0.0f, 1.5f },   // 1: on quad (0,1) -- both slots tessellated holes
        { 1.5f,    0.0f, 0.5f },   // 2: on quad (1,0) -- both solid
        { 1.5f,  -10.0f, 2.5f },   // 3: on quad (1,2), LOCAL BOUND BELOW ZERO
        { 2.5f,    0.0f, 0.5f },   // 4: under a quad raised to +400 -- con0 culls it
        { 0.5f,  900.0f, 2.5f },   // 5: far above a quad at height 0 -- con1 culls it
        { 0.5f,    0.0f, 2.5f },   // 6: on quad (0,2) -- both solid, the plain case
        { 1.5f,    0.0f, 1.5f },   // 7: on quad (1,1) -- both solid
        { 1.2f,    0.0f, 0.9f },   // 8: the TRIANGLE partner -- see the triangles below
        // *** THE TWO VERTICES WHOSE RADIUS DECIDES THE RECTANGLE. ***  Every vertex above
        // sits in the middle of a quad, so its bound covers that quad whatever the radius
        // is -- which left "drop the radius", "halve it" and "read only one of the two
        // contact distances" all alive.  The contact distance is 0.10 world units, i.e.
        // 0.20 of a COLUMN at colScale 0.5.
        //   9  at column 0.85: the correct radius reaches 1.05 and covers TWO quads; half
        //      of it reaches 0.95 and covers one, and so does the cloth's 0.02 alone.
        //   10 at column 1.25: the correct radius reaches 1.05..1.45 and covers ONE quad,
        //      while doubling the heightfield's 0.08 reaches 0.93..1.57 and covers two.
        // The two together pin the sum from both sides.
        { 1.5f,    0.0f, 0.85f },  // 9
        { 1.5f,    0.0f, 1.25f },  // 10
        // 11: the TRIANGLE fattening's own case.  Every triangle above spans enough columns
        // that fattening the bound by 0.2 of one crosses no integer, so dropping
        // fattenFast entirely changed nothing -- measured, as a survivor.  A degenerate
        // triangle sitting EXACTLY on column 1.0 has an unfattened column span of zero,
        // which the area gate rejects outright; fattened it spans 0.8..1.2 and covers two
        // quads.  Some pairs against none.
        { 1.5f,    0.0f, 1.0f }    // 11
    };
    // *** THE CLOTH IS PADDED PAST 2048. ***  The work partition assigns 64 warps (2048
    // threads) per contact manager and each lane owns one element, so with a handful of
    // vertices `nbIter` is 1, the outer element stride is never taken, and half the warps
    // do nothing.  That left the element stride and the two-warps-per-block loop both
    // unobservable -- measured, as survivors.  Three of the padding vertices are LIVE and
    // placed to force both:
    //   * index 40 and index 100 fall in the 32-element blocks that only the SECOND warp
    //     of each block reaches, so dropping the `sub` loop loses them;
    //   * index 2050 is past 2048, so it is reached only on the second `j` iteration --
    //     and only if the element stride is 2048 and not 32.
    const uint32_t NB_VERTS = 2100u;
    const uint32_t LIVE_LATE[3] = { 40u, 100u, 2050u };
    const HV FAR_AWAY{ 100.0f, 0.0f, 100.0f };
    const HV LATE_SPOT{ 1.5f, 0.0f, 0.5f };          // quad (1,0), both slots solid
    while (vs.size() < NB_VERTS) vs.push_back(FAR_AWAY);
    for (uint32_t i : LIVE_LATE) vs[i] = LATE_SPOT;
    std::vector<F4> cv;
    for (const HV& v : vs) cv.push_back(toWorld(v.r, v.h, v.c));

    // Triangles.  Triangle 0 spans vertices 0, 6 and 8, so its bound covers quads (0,0),
    // (0,1) and (0,2) at once -- a footprint NO single vertex sphere reaches, which is
    // what separates the two entry points.  Triangle 1 is the degenerate-in-footprint
    // case: all three corners inside quad (1,0).
    // Triangle 0's THIRD vertex (8) carries the maximum ROW, and it decides the rectangle:
    // rows 0.5, 0.5, 1.2 fattened by 0.05 give maxRow = ceil(1.25) = 2, while a bound taken
    // from the first two only gives ceil(0.55) = 1.
    // Triangle 2's third vertex (9) carries the minimum COLUMN the same way: columns 2.5,
    // 2.5 and 0.85 give minCol = floor(0.65) = 0 against floor(2.3) = 2.
    // Without them, `min(w0, w1)` and `max(w0, w1)` are indistinguishable from the real
    // three-way bound -- measured, as two survivors.
    std::vector<TriIdx4> ci = {
        { 0u, 6u, 8u, 0u },
        { 2u, 2u, 2u, 0u },
        { 6u, 6u, 9u, 0u },
        { 11u, 11u, 11u, 0u },
    };
    const uint32_t NB_TRIS = uint32_t(ci.size());
    DevBuf bCv = makeDev(dev, cv), bCi = makeDev(dev, ci);

    const uint32_t CLOTH_ID = 1u;
    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bCv.addr));
    c.mTriangleVertexIndices = reinterpret_cast<decltype(c.mTriangleVertexIndices)>(size_t(bCi.addr));
    c.mNbVerts = NB_VERTS;
    c.mNbTriangles = NB_TRIS;
    std::vector<PxgFEMCloth> descArr(CLOTH_ID + 1u, makeZeroDesc());
    descArr[CLOTH_ID] = c;
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- the pair.  HEIGHTFIELD is shapeRef0 and CLOTH shapeRef1 -- the SWAPPED order,
    // so LoadShapePair has to route by TYPE.
    const uint32_t GEOM_HEIGHTFIELD_H = uint32_t(physx::PxGeometryType::eHEIGHTFIELD);
    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_HEIGHTFIELD_H;  shapes[0].particleOrSoftbodyId = 0xFFFFFFFFu;
    shapes[0].sx = ROW_SCALE; shapes[0].sy = HEIGHT_SCALE; shapes[0].sz = COL_SCALE;
    shapes[0].srw = 1.0f;
    shapes[0].hullLo = uint32_t(bBlob.addr & 0xFFFFFFFFull);
    shapes[0].hullHi = uint32_t(bBlob.addr >> 32);
    shapes[1].type = GEOM_TRIANGLEMESH_H; shapes[1].particleOrSoftbodyId = CLOTH_ID;
    shapes[1].sx = 1.0f; shapes[1].sy = 1.0f; shapes[1].sz = 1.0f; shapes[1].srw = 1.0f;
    DevBuf bShapes = makeDev(dev, shapes);
    // TWO contact managers over the same shapes, for the duplication run below.
    std::vector<CmInputHost> cms = { { 0u, 1u, /*hf cache*/ 1u, /*cloth cache*/ 0u },
                                     { 0u, 1u, 1u, 0u } };
    DevBuf bCms = makeDev(dev, cms);

    std::vector<float> xf(2 * 8, 0.0f);
    xf[3] = 1.0f;                                             // slot 0 (cloth) identity
    xf[8 + 0] = 0.0f; xf[8 + 1] = 1.0f; xf[8 + 2] = 0.0f; xf[8 + 3] = 0.0f;  // 180 about y
    xf[8 + 4] = TX; xf[8 + 5] = TY; xf[8 + 6] = TZ;
    DevBuf bXf = makeDev(dev, xf);

    // ASYMMETRIC on purpose: with 0.05 and 0.05 a port that reads only the cloth's, or
    // doubles the heightfield's, produces the same sum and both mutants live.
    const float CD_CLOTH = 0.02f, CD_HF = 0.08f;
    const float contactDist = CD_CLOTH + CD_HF;
    std::vector<float> cd = { CD_CLOTH, CD_HF };
    DevBuf bCd = makeDev(dev, cd);

    // ---- CPU reference: the same chain, in the same order ----------------------------
    auto heightAt = [&](uint32_t vi) {
        return float(int16_t(uint16_t(blob[2u + vi] & 0xffffu)));
    };
    auto matAt = [&](uint32_t vi, bool slot1) {
        return (blob[2u + vi] >> (slot1 ? 24 : 16)) & 0x7fu;
    };
    // world -> heightfield row/height/column units, the hfLocalBound chain
    auto localBound = [&](F4 wMin, F4 wMax, float out_mn[3], float out_mx[3]) {
        const float cen[3] = { (wMin.x + wMax.x) * 0.5f, (wMin.y + wMax.y) * 0.5f,
                               (wMin.z + wMax.z) * 0.5f };
        const float ext[3] = { (wMax.x - wMin.x) * 0.5f, (wMax.y - wMin.y) * 0.5f,
                               (wMax.z - wMin.z) * 0.5f };
        // rotateInv by 180 about y is its own inverse: (x, y, z) -> (-x, y, -z)
        const float lc[3] = { -(cen[0] - TX), cen[1] - TY, -(cen[2] - TZ) };
        const float w[3] = { ext[0], ext[1], ext[2] };   // abs-sum of the mirrored basis
        out_mn[0] = (lc[0] - w[0]) / std::fabs(ROW_SCALE);
        out_mn[1] = (lc[1] - w[1]) / HEIGHT_SCALE;
        out_mn[2] = (lc[2] - w[2]) / std::fabs(COL_SCALE);
        out_mx[0] = (lc[0] + w[0]) / std::fabs(ROW_SCALE);
        out_mx[1] = (lc[1] + w[1]) / HEIGHT_SCALE;
        out_mx[2] = (lc[2] + w[2]) / std::fabs(COL_SCALE);
    };
    auto clampMinRow = [&](float x, uint32_t n) {
        return uint32_t(std::max(0, std::min(int(std::floor(x)), int(n) - 2))); };
    auto clampMaxRow = [&](float x, uint32_t n) {
        return uint32_t(std::max(0, std::min(int(std::ceil(x)), int(n) - 1))); };

    // pairs of (elementIdx, triangleIdx) for one element's bound
    auto pairsFor = [&](F4 wMin, F4 wMax, std::vector<uint32_t>& out) {
        float mn[3], mx[3];
        localBound(wMin, wMax, mn, mx);
        if (!(!(mn[0] > float(NB_ROWS) - 1.0f) || (mn[2] > float(NB_COLS) - 1.0f)
              || (mx[0] < 0.0f) || (mx[2] < 0.0f)))
            return;
        const uint32_t minRow = clampMinRow(mn[0], NB_ROWS), maxRow = clampMaxRow(mx[0], NB_ROWS);
        const uint32_t minCol = clampMinRow(mn[2], NB_COLS), maxCol = clampMaxRow(mx[2], NB_COLS);
        if ((2u * (maxCol - minCol) * (maxRow - minRow)) == 0u) return;
        const float minyF = float(f2uSatRef(mn[1])), maxyF = float(f2uSatRef(mx[1]));
        const uint32_t columnSpan = maxCol - minCol;
        const uint32_t total = (maxRow - minRow) * columnSpan * 2u;
        for (uint32_t i = 0; i < total; ++i)
        {
            const uint32_t index = i / 2u;
            const uint32_t vi = (minRow + index / columnSpan) * NB_COLS + (minCol + index % columnSpan);
            const float h0 = heightAt(vi), h1 = heightAt(vi + 1u);
            const float h2 = heightAt(vi + NB_COLS), h3 = heightAt(vi + NB_COLS + 1u);
            const bool con0 = maxyF < h0 && maxyF < h1 && maxyF < h2 && maxyF < h3;
            const bool con1 = minyF > h0 && minyF > h1 && minyF > h2 && minyF > h3;
            if (con0 || con1) continue;
            const bool slot1 = (i & 1u) != 0u;
            if (matAt(vi, slot1) == MAT_HOLE) continue;
            out.push_back((vi << 1) + (slot1 ? 1u : 0u));
        }
    };

    std::vector<std::pair<uint32_t, uint32_t>> refVert, refTri;   // (elementIdx, triangleIdx)
    for (uint32_t v = 0; v < NB_VERTS; ++v)
    {
        const F4 p = cv[v];
        const F4 mn{ p.x - contactDist, p.y - contactDist, p.z - contactDist, 0 };
        const F4 mx{ p.x + contactDist, p.y + contactDist, p.z + contactDist, 0 };
        std::vector<uint32_t> got; pairsFor(mn, mx, got);
        for (uint32_t t : got) refVert.push_back({ v, t });
    }
    for (uint32_t t = 0; t < NB_TRIS; ++t)
    {
        const F4 a = cv[ci[t].x], b = cv[ci[t].y], d = cv[ci[t].z];
        const F4 mn{ std::fmin(std::fmin(a.x, b.x), d.x) - contactDist,
                     std::fmin(std::fmin(a.y, b.y), d.y) - contactDist,
                     std::fmin(std::fmin(a.z, b.z), d.z) - contactDist, 0 };
        const F4 mx{ std::fmax(std::fmax(a.x, b.x), d.x) + contactDist,
                     std::fmax(std::fmax(a.y, b.y), d.y) + contactDist,
                     std::fmax(std::fmax(a.z, b.z), d.z) + contactDist, 0 };
        std::vector<uint32_t> got; pairsFor(mn, mx, got);
        for (uint32_t g : got) refTri.push_back({ t, g });
    }
    std::sort(refVert.begin(), refVert.end());
    std::sort(refTri.begin(), refTri.end());

    // ---- run both entry points -------------------------------------------------------
    const uint32_t CAP = 512u;
    const uint32_t SENTINEL = 0xFFFFFFFFu;

    uint32_t lastWritten = 0u;
    auto runOne = [&](const char* entry, uint32_t numTests, uint32_t gridY,
                      std::vector<std::pair<uint32_t, uint32_t>>& emitted,
                      std::vector<uint32_t>& cmSeen, uint32_t stackPairs) {
        // The BUFFER always holds CAP entries; only the size the KERNEL is told changes,
        // so a mutant that drops the write cap lands in slack instead of the heap.
        std::vector<uint32_t> stack(CAP * 4u, SENTINEL);
        DevBuf bStack = makeDev(dev, stack);
        std::vector<uint32_t> pairNum = { 0u };
        DevBuf bPairNum = makeDev(dev, pairNum);

        ClothHfMidPush p = {};
        p.numNPWorkItems = numTests;
        p.toleranceLength = 1.0f;
        p.cmInputs = bCms.addr;
        p.transformCache = bXf.addr;
        p.contactDistance = bCd.addr;
        p.gpuShapes = bShapes.addr;
        p.femClothes = bDesc.addr;
        p.stackPtr = bStack.addr;
        p.midphasePairsNum = bPairNum.addr;
        p.stackSizeBytes = stackPairs * 16u;

        Kernel k("femClothHfMidphaseBda.spv", entry, {}, sizeof(ClothHfMidPush));
        // The engine's grid (PxgNarrowphaseCore.cpp:6927 and :7078):
        // x = SB_SBMIDPHASE = 1024, y = numTests.
        k.dispatch(/*gridX*/ 1024u, {}, p, gridY, 1u);
        dev->flush();

        const uint32_t n = downloadBuffer<uint32_t>(dev, bPairNum.h, 1)[0];
        const auto s = downloadBuffer<uint32_t>(dev, bStack.h, CAP * 4u);
        emitted.clear(); cmSeen.clear();
        lastWritten = 0u;
        for (uint32_t i = 0; i < CAP; ++i)
            if (s[i * 4u + 0u] != SENTINEL) ++lastWritten;
        for (uint32_t i = 0; i < n && i < stackPairs; ++i)
        {
            cmSeen.push_back(s[i * 4u + 0u]);
            emitted.push_back({ s[i * 4u + 1u], s[i * 4u + 2u] });
            PXG_CHECK_MSG(s[i * 4u + 3u] == 0u,
                          "pair %u has a non-zero fourth word; the CUDA writes "
                          "make_uint4(..., 0)", i);
        }
        std::sort(emitted.begin(), emitted.end());
        return n;
    };

    std::vector<std::pair<uint32_t, uint32_t>> gotTri, gotVert;
    std::vector<uint32_t> cmTri, cmVert;
    const uint32_t nTri = runOne("cloth_heightfieldMidphaseGeneratePairsLaunch", 1u, 1u,
                                 gotTri, cmTri, CAP);
    const uint32_t nVert = runOne("cloth_midphaseVertexHeightfieldLaunch", 1u, 1u,
                                  gotVert, cmVert, CAP);

    for (uint32_t x : cmTri) PXG_CHECK_MSG(x == 0u, "triangle pass: cmIdx %u, expected 0", x);
    for (uint32_t x : cmVert) PXG_CHECK_MSG(x == 0u, "vertex pass: cmIdx %u, expected 0", x);

    PXG_CHECK_MSG(nTri == uint32_t(refTri.size()),
                  "triangle pass emitted %u pairs, reference says %zu", nTri, refTri.size());
    PXG_CHECK_MSG(gotTri == refTri, "the triangle pass's pair SET does not match the reference");
    PXG_CHECK_MSG(nVert == uint32_t(refVert.size()),
                  "vertex pass emitted %u pairs, reference says %zu", nVert, refVert.size());
    PXG_CHECK_MSG(gotVert == refVert, "the vertex pass's pair SET does not match the reference");

    auto countFor = [&](const std::vector<std::pair<uint32_t, uint32_t>>& v, uint32_t e) {
        uint32_t n = 0; for (auto& x : v) if (x.first == e) ++n; return n;
    };

    // The two entry points must DIFFER, or one body serving both proves nothing.
    PXG_CHECK_MSG(gotTri != gotVert,
                  "the triangle and vertex passes emitted the same set; the fixture cannot "
                  "tell the two element bounds apart and the shared body is untested");
    PXG_CHECK_MSG(countFor(gotTri, 0u) > countFor(gotVert, 0u),
                  "triangle 0 spans three quads that no single vertex sphere reaches, so the "
                  "triangle pass must emit MORE for it than the vertex pass does for vertex 0 "
                  "(%u against %u)", countFor(gotTri, 0u), countFor(gotVert, 0u));
    PXG_CHECK_MSG(countFor(gotVert, 0u) == 1u,
                  "vertex 0 sits over quad (0,0), whose FIRST material slot is a hole, so "
                  "exactly one of its two triangles survives; got %u", countFor(gotVert, 0u));
    PXG_CHECK_MSG(countFor(gotVert, 1u) == 0u,
                  "vertex 1's quad is a hole in BOTH slots, and its material byte carries the "
                  "TESSELLATION flag.  Getting %u instead of 0 means the flag was not masked "
                  "off before the eHOLE comparison", countFor(gotVert, 1u));
    PXG_CHECK_MSG(countFor(gotVert, 2u) == 2u,
                  "vertex 2's quad is solid in both slots; got %u", countFor(gotVert, 2u));
    PXG_CHECK_MSG(countFor(gotVert, 3u) == 2u,
                  "vertex 3's LOCAL BOUND GOES BELOW ZERO in y, and the y range reaches the "
                  "quad cull through a float->PxU32 conversion that SATURATES.  Two pairs "
                  "under the correct reading; 0 means the conversion was transcribed with C++ "
                  "host semantics, where a negative becomes a huge unsigned and con1 rejects "
                  "everything (P9.27a).  Got %u", countFor(gotVert, 3u));
    PXG_CHECK_MSG(countFor(gotVert, 4u) == 0u,
                  "vertex 4 sits far BELOW a quad raised to +400, so con0 culls it; got %u",
                  countFor(gotVert, 4u));
    PXG_CHECK_MSG(countFor(gotVert, 5u) == 0u,
                  "vertex 5 sits far ABOVE a quad at height 0, so con1 culls it; got %u",
                  countFor(gotVert, 5u));

    // ---- TWO CONTACT MANAGERS, ONE y-BLOCK --------------------------------------------
    // cmIdx is `workIndex / 64`, and with gridDim.y = numTests the two happen to agree --
    // block 0 handles cm 0 and block 1 cm 1 -- so a port that read cmIdx from blockIdx.y
    // passes the duplication run below and fails here.  Measured, as a survivor.
    std::vector<std::pair<uint32_t, uint32_t>> twoCm;
    std::vector<uint32_t> cmTwo;
    const uint32_t nTwo = runOne("cloth_heightfieldMidphaseGeneratePairsLaunch", 2u, 1u,
                                 twoCm, cmTwo, CAP);
    {
        uint32_t a = 0, b = 0;
        for (uint32_t x : cmTwo) { if (x == 0u) ++a; else if (x == 1u) ++b; }
        PXG_CHECK_MSG(nTwo == 2u * nTri && a == nTri && b == nTri,
                      "with numTests = 2 and ONE y-block each contact manager must contribute "
                      "its own %u pairs: cmIdx comes from workIndex / 64, not from the block "
                      "index.  Got %u total, %u on cm 0 and %u on cm 1", nTri, nTwo, a, b);
    }

    // ---- A STACK THAT ACTUALLY BINDS ---------------------------------------------------
    // "If we run out of stacksize, drop pairs" -- the writes stop at the capacity while the
    // COUNTER has already been committed, which is how the host learns it needs a bigger
    // one.  With a 512-entry stack and twenty pairs neither the cap nor the byte-to-entry
    // division was ever exercised; both mutants lived.
    {
        const uint32_t TIGHT = 6u;
        std::vector<std::pair<uint32_t, uint32_t>> tightSet;
        std::vector<uint32_t> tightCm;
        const uint32_t nTight = runOne("cloth_midphaseVertexHeightfieldLaunch", 1u, 1u,
                                       tightSet, tightCm, TIGHT);
        PXG_CHECK_MSG(nTight == nVert,
                      "with a capacity of %u the counter must still reach %u -- the engine "
                      "reads it back to size next frame's stack, so clamping it there loses "
                      "the growth signal (got %u)", TIGHT, nVert, nTight);
        PXG_CHECK_MSG(lastWritten == TIGHT,
                      "with a capacity of %u exactly %u entries may be written; %u were.  "
                      "More means the `writeIndex < stackSize` cap is gone, or that "
                      "stackSizeBytes was divided by something other than sizeof(uint4)",
                      TIGHT, TIGHT, lastWritten);
    }

    // ---- THE DUPLICATION, measured ---------------------------------------------------
    std::vector<std::pair<uint32_t, uint32_t>> dupTri;
    std::vector<uint32_t> cmDup;
    const uint32_t nDup = runOne("cloth_heightfieldMidphaseGeneratePairsLaunch", 2u, 2u,
                                 dupTri, cmDup, CAP);
    PXG_CHECK_MSG(nDup == 4u * nTri,
                  "with numTests = 2 the engine launches gridDim.y = 2 and NEITHER kernel "
                  "reads blockIdx.y, so each y-block repeats the whole work partition: two "
                  "contact managers times two y-blocks is FOUR copies of every pair.  Expected "
                  "%u, got %u.  Half that means the duplication was 'fixed' -- which would put "
                  "this port out of step with the engine it replaces (upstream candidate 37)",
                  4u * nTri, nDup);
    uint32_t cm0 = 0, cm1 = 0;
    for (uint32_t x : cmDup) { if (x == 0u) ++cm0; else if (x == 1u) ++cm1; }
    PXG_CHECK_MSG(cm0 == 2u * nTri && cm1 == 2u * nTri,
                  "both contact managers must appear, %u times each: cmIdx comes from "
                  "workIndex / 64 and a port that read it from the block index would put every "
                  "pair on cm 0 (got %u and %u)", 2u * nTri, cm0, cm1);

    std::printf("    [ note ] hf %ux%u, %u verts (not a multiple of 2048 -- trap H1), %u tris; "
                "triangle pass %u pairs, vertex pass %u pairs, sets differ; v3 below zero in y "
                "gives %u; numTests=2 x gridY=2 gives %u = 4x%u, split %u/%u over the two "
                "contact managers\n",
                NB_ROWS, NB_COLS, NB_VERTS, NB_TRIS, nTri, nVert, countFor(gotVert, 3u),
                nDup, nTri, cm0, cm1);

    for (PxgBufferHandle h : { bBlob.h, bCv.h, bCi.h, bDesc.h, bShapes.h, bCms.h, bXf.h, bCd.h })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P9.27b -- cloth_heightfieldContactGenLaunch, checked DIFFERENTIALLY against the
// cloth-vs-cooked-mesh contact gen it shares a core with.
//
// The two kernels call the same upstream __device__ function, and since P9.27b the
// same Slang one (pxgSlangClothTriContact).  femClothSimMeshContactGen already
// pins that core against a CPU transcription, so writing a second CPU mirror here
// would test the mirror, not the kernel.  What is UNPINNED is the heightfield
// GLUE, and a differential isolates exactly that:
//
//   * feed the heightfield kernel a heightfield whose triangle T is at some place;
//   * feed the mesh kernel a cooked mesh whose triangle IS those three vertices,
//     with scale (1,1,1) and identity scaleRot so its vertex2Shape is the identity;
//   * give both the same rigid transforms, the same cloth and the same distances.
//
// Every contact must then be BIT-IDENTICAL.  A wrong sample decode, a doubled
// scale (C2), a wrong hfToCloth composition (C4) or a mis-read triangle index all
// break the equality while leaving each kernel internally plausible.
//
// FOUR SWEEP SURVIVORS, all four verified equivalents rather than gaps:
//
//   * C1, the positional shape read.  The mesh sibling routes cloth-vs-rigid by
//     body id; this kernel takes shapeRef0 as the cloth outright.  With the
//     engine's own ordering the two rules AGREE, so no input this fixture can
//     build separates them -- the difference appears only for a pair the
//     broadphase emitted the other way round, and feeding one deliberately would
//     have the kernel read `clothes[0xffffffff]`.  Stated rather than papered
//     over: this is the one difference between the two kernels that is documented
//     and not measured.
//   * rounding `l` up to a whole warp.  The trailing lanes have `i >= numPairs`,
//     so they ballot false and write nothing; the rounding exists so the CUDA's
//     __ballot_sync is reached by a full warp, which the groupshared emulation
//     does not need.
//   * `gid.x * (4 * WARP) + sub * WARP` against `gid.x * WARP + sub * (4 * WARP)`.
//     Both enumerate exactly the multiples of 32, each once -- a permutation of
//     the same work, not a different set of it.
//   * reserving without the `validCount > 0` guard.  InterlockedAdd of zero
//     cannot move a counter, and no lane writes when the ballot is empty.
// ===========================================================================
namespace p927cg
{
struct ClothHfCgPush
{
    float toleranceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, restDistance, gpuShapes, femClothes,
             shapeToRigidRemapTable, stackPtr, midphasePairsNum;
    uint32_t stackSizeBytes, pad84;
    uint64_t stackSizeNeededOnDevice;
    uint64_t wPoint, wNormalPen, wBary, wInfo, wCount, wByX, wTempByX, wIdxByX, wSorted32;
    uint32_t wMaxContacts, pad172;
};
static_assert(sizeof(ClothHfCgPush) == 176, "cloth hf CG blob must be 176 bytes");
static_assert(offsetof(ClothHfCgPush, stackPtr) == 64, "stackPtr @64");
static_assert(offsetof(ClothHfCgPush, midphasePairsNum) == 72, "midphasePairsNum @72");
static_assert(offsetof(ClothHfCgPush, stackSizeBytes) == 80, "stackSizeBytes @80");
static_assert(offsetof(ClothHfCgPush, wPoint) == 96, "writer @96");

// The DISABLED sibling's pack: the same thirteen arguments, writer BEFORE the two
// tail scalars.  Both are 176 bytes and the layouts differ -- see the ABI note.
struct ClothHfVertCgPush
{
    float toleranceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, restDistance, gpuShapes, femClothes,
             shapeToRigidRemapTable, stackPtr, midphasePairsNum;
    uint64_t wPoint, wNormalPen, wBary, wInfo, wCount, wByX, wTempByX, wIdxByX, wSorted32;
    uint32_t wMaxContacts, pad156;
    uint32_t stackSizeBytes, pad164;
    uint64_t stackSizeNeededOnDevice;
};
static_assert(sizeof(ClothHfVertCgPush) == 176, "cloth hf VERTEX CG blob must be 176 bytes");
static_assert(offsetof(ClothHfVertCgPush, wPoint) == 80, "writer @80 -- NOT @96");
static_assert(offsetof(ClothHfVertCgPush, stackSizeBytes) == 160, "stackSizeBytes @160");
static_assert(offsetof(ClothHfVertCgPush, stackSizeNeededOnDevice) == 168, "needed @168");

using p817cg::V3;
static uint32_t hfSample(int16_t h, uint8_t m0, uint8_t m1)
{
    return (uint32_t(uint16_t(h))) | (uint32_t(m0) << 16) | (uint32_t(m1) << 24);
}
} // namespace p927cg

PXG_TEST(femClothSimHeightfieldContactGen)
{
    using namespace p816dc;
    using namespace p927cg;
    using p817cg::V3;
    using p817cg::qRot; using p817cg::qRotInv; using p817cg::qMul;
    using p817cg::norm; using p817cg::crs; using p817cg::sub; using p817cg::add;
    using p817cg::scl; using p817cg::dot;
    PxgSlangComputeDevice* dev = testDevice();

    // ---- the heightfield: 2 rows x 3 columns, one row of quads -----------------------
    // Column 1's sample carries the TESSELLATION bit and column 0's does not, so the two
    // quads are cut along DIFFERENT diagonals -- which is what makes hfTriVertexIndices'
    // branch observable rather than a coin flip that happens to agree.
    const uint32_t NB_ROWS = 2u, NB_COLS = 3u;
    const uint8_t MAT = 3u;
    std::vector<uint32_t> blob(2u + NB_ROWS * NB_COLS, 0u);
    blob[0] = NB_ROWS; blob[1] = NB_COLS;
    auto setS = [&](uint32_t r, uint32_t c, int16_t h, uint8_t m0, uint8_t m1) {
        blob[2u + r * NB_COLS + c] = hfSample(h, m0, m1);
    };
    setS(0u, 0u, 0,  MAT, MAT);
    setS(0u, 1u, 0,  uint8_t(MAT | 0x80u), MAT);   // tessellation bit set
    setS(0u, 2u, 0,  MAT, MAT);
    setS(1u, 0u, 10, MAT, MAT);                    // a NON-ZERO height, so the decode matters
    setS(1u, 1u, 10, MAT, MAT);
    setS(1u, 2u, 10, MAT, MAT);
    DevBuf bHfBlob = makeDev(dev, blob);
    if (!bHfBlob.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // Non-unit scales on all three axes, so a dropped or doubled scale moves the terrain.
    const V3 hfScale{ 2.0f, 0.05f, 0.5f };

    // ---- hfGetTriangle, transcribed ---------------------------------------------------
    auto heightAt = [&](uint32_t vi) { return float(int16_t(uint16_t(blob[2u + vi] & 0xffffu))); };
    auto tessAt = [&](uint32_t vi) { return (blob[2u + vi] & 0x800000u) != 0u; };
    auto hfVertex = [&](uint32_t vi) {
        return V3{ float(vi / NB_COLS), heightAt(vi), float(vi % NB_COLS) };
    };
    auto hfTriangle = [&](uint32_t triIdx, V3& a, V3& b, V3& c) {
        const uint32_t cell = triIdx >> 1;
        const bool first = (triIdx & 1u) == 0u;
        uint32_t t0, t1, t2;
        if (tessAt(cell))
        {
            if (first) { t0 = cell + NB_COLS; t1 = cell; t2 = cell + NB_COLS + 1u; }
            else       { t0 = cell + 1u; t1 = cell + NB_COLS + 1u; t2 = cell; }
        }
        else
        {
            if (first) { t0 = cell; t1 = cell + 1u; t2 = cell + NB_COLS; }
            else       { t0 = cell + NB_COLS + 1u; t1 = cell + NB_COLS; t2 = cell + 1u; }
        }
        auto sc = [&](V3 v) { return V3{ v.x * hfScale.x, v.y * hfScale.y, v.z * hfScale.z }; };
        a = sc(hfVertex(t0)); b = sc(hfVertex(t1)); c = sc(hfVertex(t2));
    };

    // Triangle 3 = cell 1's SECOND triangle: cell 1 carries the tessellation bit, so it
    // takes the other diagonal, and it is the second slot, so it takes the other winding.
    const uint32_t HF_TRI = 3u;
    V3 hfA, hfB, hfC;
    hfTriangle(HF_TRI, hfA, hfB, hfC);

    // ---- the cooked mesh: the SAME triangle, scale (1,1,1), identity scaleRot ---------
    // With those two, the mesh kernel's vertex2Shape is the identity, so both kernels
    // must produce byte-identical static triangles in cloth space.
    std::vector<F4> mv = {
        { -50.0f, 1000.0f, -50.0f, 0 }, { -50.0f, 1000.0f, 50.0f, 0 }, { 50.0f, 1000.0f, -50.0f, 0 },
        { hfA.x, hfA.y, hfA.z, 0 }, { hfB.x, hfB.y, hfB.z, 0 }, { hfC.x, hfC.y, hfC.z, 0 },
    };
    std::vector<TriIdx4> mi = { { 0u, 1u, 2u, 0u }, { 3u, 4u, 5u, 0u } };
    const uint32_t MESH_TRI = 1u;
    std::vector<uint32_t> mblob(4u + 292u, 0u);
    const size_t vBase = mblob.size();
    mblob.resize(vBase + mv.size() * 4u, 0u);
    std::memcpy(mblob.data() + vBase, mv.data(), mv.size() * sizeof(F4));
    const size_t iBase = mblob.size();
    mblob.resize(iBase + mi.size() * 4u, 0u);
    std::memcpy(mblob.data() + iBase, mi.data(), mi.size() * sizeof(TriIdx4));
    mblob[0] = uint32_t(mv.size()); mblob[1] = uint32_t(mi.size());
    mblob[2] = 7u; mblob[3] = 1u;
    DevBuf bMeshBlob = makeDev(dev, mblob);

    // ---- the two rigid transforms, both non-trivial -----------------------------------
    const float QS = 0.70710678f;
    const float clothQ[4] = { 0.0f, 0.0f, QS, QS };   // 90 about z
    const V3 clothP{ 7.0f, 8.0f, 9.0f };
    const float rigidQ[4] = { 0.0f, QS, 0.0f, QS };   // 90 about y
    const V3 rigidP{ 1.0f, 2.0f, 3.0f };

    // The static triangle in CLOTH-local space, for placing the cloth on top of it.
    float clothQConj[4] = { -clothQ[0], -clothQ[1], -clothQ[2], clothQ[3] };
    float r2cQ[4]; qMul(clothQConj, rigidQ, r2cQ);
    const V3 r2cP = qRotInv(clothQ, sub(rigidP, clothP));
    auto toClothLocal = [&](V3 v) { return add(qRot(r2cQ, v), r2cP); };
    const V3 A0 = toClothLocal(hfA), B0 = toClothLocal(hfB), C0 = toClothLocal(hfC);
    const V3 fixNormal = norm(crs(sub(B0, A0), sub(C0, A0)));
    const V3 centre{ (A0.x + B0.x + C0.x) / 3.0f, (A0.y + B0.y + C0.y) / 3.0f,
                     (A0.z + B0.z + C0.z) / 3.0f };
    const V3 t1 = norm(sub(B0, A0));
    const V3 t2 = crs(fixNormal, t1);

    // ---- the cloth: enough triangles to straddle a warp boundary ----------------------
    const float CD_CLOTH = 0.05f, CD_RIGID = 0.15f;
    const uint32_t NB_PAIRS = 40u;
    auto gapFor = [&](uint32_t i) -> float {
        if (i == 0u) return 0.10f;
        if (i == 2u) return -0.05f;     // penetrating
        if (i == 4u) return 0.10f;
        if (i == 35u) return 0.02f;     // the only contact in warp 1
        return 0.25f;                   // outside the slab
    };
    std::vector<F4> cv; std::vector<TriIdx4> ci;
    for (uint32_t i = 0; i < NB_PAIRS; ++i)
    {
        const V3 base = add(centre, scl(fixNormal, gapFor(i)));
        const V3 c0 = add(base, scl(t1, 0.20f));
        const V3 c1 = add(base, scl(t2, 0.20f));
        const V3 c2 = add(base, scl(t1, -0.20f));
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
    // Cloth id ONE, not zero.  `pairInd1` is `(clothId << 20) | triangleIdx`, and with
    // id 0 every shift produces the same word -- a mutant moving the shift to 16 lived.
    const uint32_t CLOTH_ID = 1u;
    std::vector<PxgFEMCloth> descArr(CLOTH_ID + 1u, makeZeroDesc());
    descArr[CLOTH_ID] = c;
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- shapes.  HF run: cloth is shapeRef0 (positional, C1).  Mesh run: SWAPPED,
    // because that kernel routes by body id and the existing fixture proves it.
    const uint32_t GEOM_HEIGHTFIELD_H = uint32_t(physx::PxGeometryType::eHEIGHTFIELD);
    std::vector<PxgShapeHost> hfShapes(2), meshShapes(2);
    std::memset(hfShapes.data(), 0, hfShapes.size() * sizeof(PxgShapeHost));
    std::memset(meshShapes.data(), 0, meshShapes.size() * sizeof(PxgShapeHost));
    hfShapes[0].type = GEOM_TRIANGLEMESH_H; hfShapes[0].particleOrSoftbodyId = CLOTH_ID;
    hfShapes[0].sx = hfShapes[0].sy = hfShapes[0].sz = 1.0f; hfShapes[0].srw = 1.0f;
    hfShapes[1].type = GEOM_HEIGHTFIELD_H;  hfShapes[1].particleOrSoftbodyId = 0xFFFFFFFFu;
    hfShapes[1].sx = hfScale.x; hfShapes[1].sy = hfScale.y; hfShapes[1].sz = hfScale.z;
    hfShapes[1].srw = 1.0f; hfShapes[1].materialIndex = 0x2Au;
    hfShapes[1].hullLo = uint32_t(bHfBlob.addr & 0xFFFFFFFFull);
    hfShapes[1].hullHi = uint32_t(bHfBlob.addr >> 32);

    meshShapes[0].type = GEOM_TRIANGLEMESH_H; meshShapes[0].particleOrSoftbodyId = 0xFFFFFFFFu;
    meshShapes[0].sx = meshShapes[0].sy = meshShapes[0].sz = 1.0f; meshShapes[0].srw = 1.0f;
    meshShapes[0].materialIndex = 0x2Au;
    meshShapes[0].hullLo = uint32_t(bMeshBlob.addr & 0xFFFFFFFFull);
    meshShapes[0].hullHi = uint32_t(bMeshBlob.addr >> 32);
    meshShapes[1].type = GEOM_TRIANGLEMESH_H; meshShapes[1].particleOrSoftbodyId = CLOTH_ID;
    meshShapes[1].sx = meshShapes[1].sy = meshShapes[1].sz = 1.0f; meshShapes[1].srw = 1.0f;
    DevBuf bHfShapes = makeDev(dev, hfShapes), bMeshShapes = makeDev(dev, meshShapes);

    // cache ref 0 = cloth, 1 = rigid, in both runs.
    // TWO contact managers, identical in content.  With one, cmIdx is always 0 and so is
    // the pair's fourth word, so a kernel reading cmIdx out of the WRONG word produced the
    // same answer -- measured, as a survivor.
    std::vector<CmInputHost> hfCms = { { 0u, 1u, 0u, 1u }, { 0u, 1u, 0u, 1u } };
    std::vector<CmInputHost> meshCms = { { 0u, 1u, 1u, 0u }, { 0u, 1u, 1u, 0u } };
    DevBuf bHfCms = makeDev(dev, hfCms), bMeshCms = makeDev(dev, meshCms);

    std::vector<float> xf(2 * 8, 0.0f);
    xf[0] = clothQ[0]; xf[1] = clothQ[1]; xf[2] = clothQ[2]; xf[3] = clothQ[3];
    xf[4] = clothP.x; xf[5] = clothP.y; xf[6] = clothP.z;
    xf[8 + 0] = rigidQ[0]; xf[8 + 1] = rigidQ[1]; xf[8 + 2] = rigidQ[2]; xf[8 + 3] = rigidQ[3];
    xf[8 + 4] = rigidP.x; xf[8 + 5] = rigidP.y; xf[8 + 6] = rigidP.z;
    DevBuf bXf = makeDev(dev, xf);

    std::vector<float> cd = { CD_CLOTH, CD_RIGID };
    std::vector<float> rd = { 0.125f, 0.0f };
    DevBuf bCd = makeDev(dev, cd), bRd = makeDev(dev, rd);
    std::vector<uint64_t> remap = { 0ull, 0x00000007ull };      // PxNodeIndex for cache ref 1
    DevBuf bRemap = makeDev(dev, remap);

    // ---- the pair stack, shared in shape by both runs ----------------------------------
    // The BUFFER holds more than the kernel is told, and the midphase reports MORE FOUND
    // than the stack holds.  Both matter: `numPairs = min(found, stackSize)` bounds the
    // walk, while the high-water mark takes the RAW count -- and with found == stackSize
    // neither clamp was observable.  The surplus entries are addressable (so an unclamped
    // walk lands in slack rather than the heap) and carry triangle 0, which makes contact,
    // so reading them shows up as extra contacts rather than as silence.
    const uint32_t NB_STACK = NB_PAIRS;            // what the kernel is told
    const uint32_t NB_FOUND = NB_PAIRS + 20u;      // what the midphase claims
    const uint32_t NB_ALLOC = NB_FOUND;            // what is actually addressable
    std::vector<uint32_t> hfStack(NB_ALLOC * 4u, 0u), meshStack(NB_ALLOC * 4u, 0u);
    for (uint32_t i = 0; i < NB_ALLOC; ++i)
    {
        // The SECOND contact manager owns the odd pairs.
        const uint32_t cm = (i & 1u);
        const uint32_t tri = (i < NB_PAIRS) ? i : 0u;
        hfStack[i * 4 + 0] = cm; hfStack[i * 4 + 1] = tri; hfStack[i * 4 + 2] = HF_TRI;
        meshStack[i * 4 + 0] = cm; meshStack[i * 4 + 1] = tri; meshStack[i * 4 + 2] = MESH_TRI;
    }
    DevBuf bHfStack = makeDev(dev, hfStack), bMeshStack = makeDev(dev, meshStack);
    std::vector<uint32_t> nFound = { NB_FOUND };
    DevBuf bHfNum = makeDev(dev, nFound), bMeshNum = makeDev(dev, nFound);

    const uint32_t MAXC = 64u;
    WriterBufs wHf = makeWriter(dev, MAXC), wMesh = makeWriter(dev, MAXC);
    std::vector<uint32_t> zeroNeeded = { 0u };
    DevBuf bHfNeeded = makeDev(dev, zeroNeeded), bMeshNeeded = makeDev(dev, zeroNeeded);

    // ---- run the HEIGHTFIELD kernel ----------------------------------------------------
    ClothHfCgPush hp = {};
    hp.toleranceLength = 1.0f;
    hp.cmInputs = bHfCms.addr; hp.transformCache = bXf.addr;
    hp.contactDistance = bCd.addr; hp.restDistance = bRd.addr;
    hp.gpuShapes = bHfShapes.addr; hp.femClothes = bDesc.addr;
    hp.shapeToRigidRemapTable = bRemap.addr;
    hp.stackPtr = bHfStack.addr; hp.midphasePairsNum = bHfNum.addr;
    hp.stackSizeBytes = NB_STACK * 16u;
    hp.stackSizeNeededOnDevice = bHfNeeded.addr;
    hp.wPoint = wHf.point.addr; hp.wNormalPen = wHf.normalPen.addr;
    hp.wBary = wHf.barycentric.addr; hp.wInfo = wHf.info.addr; hp.wCount = wHf.count.addr;
    hp.wByX = wHf.byRigid.addr; hp.wTempByX = wHf.tempByRigid.addr;
    hp.wIdxByX = wHf.idxByRigid.addr; hp.wSorted32 = wHf.sortedByRigid.addr;
    hp.wMaxContacts = MAXC;
    Kernel kh("femClothHfContactGenBda.spv", "cloth_heightfieldContactGenLaunch",
              {}, sizeof(ClothHfCgPush));
    kh.dispatch(2048u, {}, hp, 1u, 1u);      // PxgNarrowphaseCore.cpp:7024
    dev->flush();

    // ---- run the MESH kernel over the identical geometry --------------------------------
    p817cg::ClothMeshCgPush mp = {};
    mp.toleranceLength = 1.0f;
    mp.cmInputs = bMeshCms.addr; mp.transformCache = bXf.addr;
    mp.contactDistance = bCd.addr; mp.restDistance = bRd.addr;
    mp.gpuShapes = bMeshShapes.addr; mp.femClothes = bDesc.addr;
    mp.shapeToRigidRemapTable = bRemap.addr;
    mp.stackSizeBytes = NB_STACK * 16u;
    mp.stackPtr = bMeshStack.addr; mp.midphasePairsNum = bMeshNum.addr;
    mp.filterPairs = 0ull; mp.nbFilterPairs = 0u;
    mp.stackSizeNeededOnDevice = bMeshNeeded.addr;
    mp.wPoint = wMesh.point.addr; mp.wNormalPen = wMesh.normalPen.addr;
    mp.wBary = wMesh.barycentric.addr; mp.wInfo = wMesh.info.addr; mp.wCount = wMesh.count.addr;
    mp.wByX = wMesh.byRigid.addr; mp.wTempByX = wMesh.tempByRigid.addr;
    mp.wIdxByX = wMesh.idxByRigid.addr; mp.wSorted32 = wMesh.sortedByRigid.addr;
    mp.wMaxContacts = MAXC;
    Kernel km("femClothMeshContactGenBda.spv", "cloth_meshContactGenLaunch",
              {}, sizeof(p817cg::ClothMeshCgPush));
    km.dispatch(2048u, {}, mp, 1u, 1u);
    dev->flush();

    // ---- compare ------------------------------------------------------------------------
    const uint32_t nHf = downloadBuffer<uint32_t>(dev, wHf.count.h, 1)[0];
    const uint32_t nMesh = downloadBuffer<uint32_t>(dev, wMesh.count.h, 1)[0];
    PXG_CHECK_MSG(nHf > 0u,
                  "the heightfield contact gen produced NO contacts; the differential below "
                  "is meaningless without at least one");
    PXG_CHECK_MSG(nHf == nMesh,
                  "the two kernels disagree on the contact COUNT (%u against %u) over the "
                  "identical static triangle.  They share staticTriangleVsClothCollision, so a "
                  "difference is in the heightfield glue: the sample decode, the missing "
                  "vertex2Shape (C2) or the hfToCloth composition (C4)", nHf, nMesh);

    // Contacts are written at a warp-scan index, and the warps race, so compare as a
    // MULTISET keyed on the cloth triangle rather than slot by slot.
    const auto pHf = downloadBuffer<F4>(dev, wHf.point.h, MAXC);
    const auto pMesh = downloadBuffer<F4>(dev, wMesh.point.h, MAXC);
    const auto nHfBuf = downloadBuffer<F4>(dev, wHf.normalPen.h, MAXC);
    const auto nMeshBuf = downloadBuffer<F4>(dev, wMesh.normalPen.h, MAXC);
    const auto bHfBuf = downloadBuffer<F4>(dev, wHf.barycentric.h, MAXC);
    const auto bMeshBuf = downloadBuffer<F4>(dev, wMesh.barycentric.h, MAXC);
    const auto iHf = downloadBuffer<uint32_t>(dev, wHf.info.h, MAXC * 4u);
    const auto iMesh = downloadBuffer<uint32_t>(dev, wMesh.info.h, MAXC * 4u);

    struct Rec { uint32_t tri; uint32_t w[12]; };
    auto gather = [&](uint32_t n, const std::vector<F4>& pt, const std::vector<F4>& np,
                      const std::vector<F4>& ba, const std::vector<uint32_t>& info) {
        std::vector<Rec> out;
        for (uint32_t i = 0; i < n && i < MAXC; ++i)
        {
            Rec r{};
            r.tri = info[i * 4 + 2] & ((1u << 20) - 1u);
            const float src[12] = { pt[i].x, pt[i].y, pt[i].z, pt[i].w,
                                    np[i].x, np[i].y, np[i].z, np[i].w,
                                    ba[i].x, ba[i].y, ba[i].z, ba[i].w };
            for (int k = 0; k < 12; ++k) r.w[k] = f2u(src[k]);
            out.push_back(r);
        }
        std::sort(out.begin(), out.end(), [](const Rec& a, const Rec& b) {
            if (a.tri != b.tri) return a.tri < b.tri;
            for (int k = 0; k < 12; ++k) if (a.w[k] != b.w[k]) return a.w[k] < b.w[k];
            return false;
        });
        return out;
    };
    const auto recHf = gather(nHf, pHf, nHfBuf, bHfBuf, iHf);
    const auto recMesh = gather(nMesh, pMesh, nMeshBuf, bMeshBuf, iMesh);

    uint32_t mismatches = 0;
    for (size_t i = 0; i < recHf.size() && i < recMesh.size(); ++i)
    {
        if (recHf[i].tri != recMesh[i].tri) { ++mismatches; continue; }
        for (int k = 0; k < 12; ++k)
            if (recHf[i].w[k] != recMesh[i].w[k]) { ++mismatches; break; }
    }
    PXG_CHECK_MSG(mismatches == 0u,
                  "%u of %zu contacts differ BIT-EXACTLY between the heightfield and mesh "
                  "kernels over the identical static triangle.  Both call the same core, so "
                  "the difference is the heightfield glue", mismatches, recHf.size());

    // The contact INFO words must agree too -- the rigid id, the encoded cloth index and
    // the material all come from the glue, not from the shared core.
    for (uint32_t i = 0; i < nHf && i < MAXC; ++i)
        PXG_CHECK_MSG(iHf[i * 4 + 3] == 0x2Au,
                      "contact %u carries material 0x%X, expected the HEIGHTFIELD shape's "
                      "0x2A -- a port reading the CLOTH's materialIndex would get 0", i,
                      iHf[i * 4 + 3]);

    // The stack high-water mark: UNCLAMPED, in bytes.
    const uint32_t needHf = downloadBuffer<uint32_t>(dev, bHfNeeded.h, 1)[0];
    PXG_CHECK_MSG(needHf == NB_FOUND * 16u,
                  "stackSizeNeededOnDevice is %u, expected %u = %u FOUND pairs * 16 bytes.  "
                  "%u would mean the count was clamped to the %u-pair stack first, and a "
                  "value that cannot exceed the stack it is meant to grow could never ask "
                  "for a bigger one", needHf, NB_FOUND * 16u, NB_FOUND,
                  NB_STACK * 16u, NB_STACK);

    // ---- a writer bound that actually BINDS -------------------------------------------
    // The writer drops a contact whose index reaches maxNumContacts.  With room for
    // sixty-four and four contacts the guard never fired, so deleting it survived.
    {
        const uint32_t TIGHT = 2u;
        WriterBufs wt = makeWriter(dev, MAXC);      // the BUFFERS stay full size ...
        ClothHfCgPush tp = hp;
        tp.wPoint = wt.point.addr; tp.wNormalPen = wt.normalPen.addr;
        tp.wBary = wt.barycentric.addr; tp.wInfo = wt.info.addr; tp.wCount = wt.count.addr;
        tp.wByX = wt.byRigid.addr; tp.wTempByX = wt.tempByRigid.addr;
        tp.wIdxByX = wt.idxByRigid.addr; tp.wSorted32 = wt.sortedByRigid.addr;
        tp.wMaxContacts = TIGHT;                    // ... only the kernel's bound moves
        std::vector<uint32_t> nd2 = { 0u };
        DevBuf bNd2 = makeDev(dev, nd2);
        tp.stackSizeNeededOnDevice = bNd2.addr;
        kh.dispatch(2048u, {}, tp, 1u, 1u);
        dev->flush();
        const auto tp4 = downloadBuffer<F4>(dev, wt.point.h, MAXC);
        uint32_t written = 0;
        for (uint32_t i = 0; i < MAXC; ++i)
            if (!(tp4[i].x == 0.0f && tp4[i].y == 0.0f && tp4[i].z == 0.0f && tp4[i].w == 0.0f))
                ++written;
        PXG_CHECK_MSG(written <= TIGHT,
                      "with maxNumContacts = %u the writer may fill at most %u slots; %u "
                      "carry data.  More means the `index >= maxNumContacts` guard is gone",
                      TIGHT, TIGHT, written);
        const uint32_t cnt2 = downloadBuffer<uint32_t>(dev, wt.count.h, 1)[0];
        PXG_CHECK_MSG(cnt2 == nHf,
                      "the contact COUNTER must still reach %u even when the writer drops "
                      "contacts past %u -- the count is reserved before the bound is "
                      "checked (got %u)", nHf, TIGHT, cnt2);
        destroyWriter(dev, wt);
        dev->destroyBuffer(bNd2.h);
    }

    std::printf("    [ note ] hf %ux%u, triangle %u (tessellated cell, second slot); "
                "%u contacts over 2 contact managers, bit-identical to the mesh kernel "
                "over the same triangle; %u found into a %u-pair stack -> stackSizeNeeded "
                "%u (unclamped)\n", NB_ROWS, NB_COLS, HF_TRI, nHf, NB_FOUND, NB_STACK,
                needHf);

    // ---- the DISABLED sibling ------------------------------------------------------------
    {
        const uint32_t SEED = 11u, FOUND = 29u, STACK_PAIRS = 8u;
        std::vector<uint32_t> seedCount = { SEED };
        DevBuf bCount = makeDev(dev, seedCount);
        std::vector<uint32_t> found = { FOUND };
        DevBuf bFound = makeDev(dev, found);
        // PRE-SEEDED above what the kernel will ask for, so `atomicMax` and a plain
        // store are distinguishable.  With a seed of zero they agree and the mutant lived.
        const uint32_t SEED_NEEDED = FOUND * 16u + 1000u;
        std::vector<uint32_t> needed = { SEED_NEEDED };
        DevBuf bNeeded = makeDev(dev, needed);
        WriterBufs w2 = makeWriter(dev, MAXC);

        ClothHfVertCgPush vp = {};
        vp.toleranceLength = 1.0f;
        vp.stackPtr = bHfStack.addr;
        vp.midphasePairsNum = bFound.addr;
        vp.stackSizeBytes = STACK_PAIRS * 16u;
        vp.stackSizeNeededOnDevice = bNeeded.addr;
        vp.wPoint = w2.point.addr; vp.wNormalPen = w2.normalPen.addr;
        vp.wBary = w2.barycentric.addr; vp.wInfo = w2.info.addr;
        vp.wCount = bCount.addr;
        vp.wByX = w2.byRigid.addr; vp.wTempByX = w2.tempByRigid.addr;
        vp.wIdxByX = w2.idxByRigid.addr; vp.wSorted32 = w2.sortedByRigid.addr;
        vp.wMaxContacts = MAXC;
        // cmInputs, transformCache, contactDistance, restDistance, gpuShapes, femClothes and
        // shapeToRigidRemapTable stay NULL.  They are in the pack for ABI parity and the
        // kernel reads none of them -- which a null address is how to test, not assert.
        Kernel kv("femClothHfContactGenBda.spv", "cloth_heightfieldVertexContactGenLaunch",
                  {}, sizeof(ClothHfVertCgPush));
        kv.dispatch(512u, {}, vp, 1u, 1u);       // PxgNarrowphaseCore.cpp:7176
        dev->flush();

        const uint32_t nd = downloadBuffer<uint32_t>(dev, bNeeded.h, 1)[0];
        PXG_CHECK_MSG(nd == SEED_NEEDED,
                      "stackSizeNeeded was seeded at %u and the kernel asks for %u, so an "
                      "atomicMax must leave it alone; got %u.  A plain store would drop "
                      "another contact manager's larger request in the same frame",
                      SEED_NEEDED, FOUND * 16u, nd);
        // ... and now with a seed BELOW the request, so the clamp is observable too.
        std::vector<uint32_t> needed2 = { 0u };
        DevBuf bNeeded2 = makeDev(dev, needed2);
        vp.stackSizeNeededOnDevice = bNeeded2.addr;
        kv.dispatch(512u, {}, vp, 1u, 1u);
        dev->flush();
        const uint32_t nd2 = downloadBuffer<uint32_t>(dev, bNeeded2.h, 1)[0];
        PXG_CHECK_MSG(nd2 == FOUND * 16u,
                      "stackSizeNeeded is %u, expected %u = %u found * 16 bytes.  %u would "
                      "mean the count was clamped to the %u-pair stack first -- a value that "
                      "cannot exceed the stack it is meant to grow is useless",
                      nd2, FOUND * 16u, FOUND, STACK_PAIRS * 16u, STACK_PAIRS);
        dev->destroyBuffer(bNeeded2.h);
        const uint32_t cnt = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
        PXG_CHECK_MSG(cnt == SEED,
                      "the contact counter moved from %u to %u.  It must not: every lane's "
                      "`hasContacts` is false upstream, so the ballot is empty and "
                      "PxgOutputIndex performs no atomicAdd at all", SEED, cnt);
        const auto gp = downloadBuffer<F4>(dev, w2.point.h, MAXC);
        for (uint32_t i = 0; i < MAXC; ++i)
            PXG_CHECK_MSG(gp[i].x == 0.0f && gp[i].w == 0.0f,
                          "contact slot %u was written.  This kernel emits NO contacts "
                          "upstream; if that has changed, gpuslang.disabledKernels says so "
                          "and the geometry has to be ported for real", i);
        std::printf("    [ note ] disabled sibling: %u found into an %u-pair stack -> "
                    "stackSizeNeeded %u (unclamped); a larger seed survives the max; "
                    "contact count %u untouched; no slot written\n",
                    FOUND, STACK_PAIRS, nd2, cnt);
        destroyWriter(dev, w2);
        for (PxgBufferHandle h : { bCount.h, bFound.h, bNeeded.h }) dev->destroyBuffer(h);
    }

    destroyWriter(dev, wHf); destroyWriter(dev, wMesh);
    for (PxgBufferHandle h : { bHfBlob.h, bMeshBlob.h, bCv.h, bCi.h, bDesc.h, bHfShapes.h,
                               bMeshShapes.h, bHfCms.h, bMeshCms.h, bXf.h, bCd.h, bRd.h,
                               bRemap.h, bHfStack.h, bMeshStack.h, bHfNum.h, bMeshNum.h,
                               bHfNeeded.h, bMeshNeeded.h })
        dev->destroyBuffer(h);
}


// ===========================================================================
// P9.29 -- cloth_SphereContactGenLaunch, against a full HOST MIRROR.
//
// Unlike P9.15d/P9.20/P9.21, this kernel's collision cores are NOT already
// covered by a Phase-1 battery on the engine path: triSphereCollide exists in
// fcPrim.slang, but fcCapsuleCollision (and the whole
// distanceSegmentTriangleSquared / distanceSegmentSegmentSquared chain under
// it) had no engine-side port before this tranche.  So the reference here is a
// C++ transcription of BOTH cores plus the glue, and the comparison is
// BIT-EXACT -- not a differential against a sibling kernel.
//
// WHAT THE RIG IS BUILT TO DISCRIMINATE
//   * TWO contact managers with DIFFERENT rigid TYPES -- a sphere and a capsule
//     -- so S2's bare `else` is exercised on both sides of the branch, and a
//     port that wrote `else if (type == eCAPSULE)` would still pass while one
//     that routed both to the sphere core would not;
//   * DIFFERENT rest distances per manager (S6), so `restDistance[cmIdx]` is
//     separable from a cache ref;
//   * 40 pairs, well past one warp: a per-WARP driver -- which is what the box
//     and convex siblings in the same file use, and what the host's stale
//     comment at the launch site claims -- would see 1 pair in 32 (S1);
//   * a midphase count that OVERSHOOTS the stack bound, so S4's clamp and S8's
//     UNCLAMPED high-water mark pull opposite ways in one run;
//   * a filter list naming ALL THREE corners of one triangle and TWO of
//     another: the first must vanish, the second must survive (S7);
//   * capsule-vs-triangle cases that reach THREE different exits of
//     distanceSegmentTriangleSquared -- the plane-crossing early out, the
//     both-projections-interior branch, and the edge fallback -- because the
//     three are 60 lines apart and a transcription slip in one is invisible in
//     the others.
// ===========================================================================
namespace p929st
{
using p817cg::V3;
using p817cg::qRot;
using p817cg::norm;
using p817cg::crs;
using p817cg::sub;
using p817cg::add;
using p817cg::scl;
using p817cg::dot;

using p817cg::RigidFilterPairHost;

// The kernel shares cloth_boxTriangleContactGenLaunch's fifteen-argument pack --
// and NOT cloth_meshContactGenLaunch's, which this fixture reached for first and
// paid for with a deterministic segfault.  Both are 192 bytes AND both put the
// writer at @112, so sizeof and the one static_assert everybody checks agree.
// The MIDDLE does not:
//
//              mesh (ClothMeshCgPush)        box-triangle (this one)
//     @64      stackSizeBytes                stackPtr
//     @72      stackPtr                      midphasePairsNum
//     @80      midphasePairsNum              filterPairs
//     @88      filterPairs                   nbFilterPairs
//     @96      nbFilterPairs                 stackSizeNeeded
//     @104     stackSizeNeeded               (pad -- paramAlign(80) is 16)
//
// Every asserted offset below is therefore load-bearing, not decoration.
struct ClothSphereTriPush
{
    float toleranceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, restDistance, gpuShapes, femClothes,
             shapeToRigidRemapTable, stackPtr, midphasePairsNum, filterPairs;
    uint32_t nbFilterPairs, stackSizeBytes;
    uint64_t stackSizeNeededOnDevice;
    uint64_t pad104;                 // paramAlign(80) == 16, so the writer lands at 112
    uint64_t wPoint, wNormalPen, wBary, wInfo, wCount, wByX, wTempByX, wIdxByX, wSorted32;
    uint32_t wMaxContacts, pad188;
};
static_assert(sizeof(ClothSphereTriPush) == 192, "cloth sphere/capsule TRIANGLE blob must be 192 bytes");
static_assert(offsetof(ClothSphereTriPush, stackPtr) == 64, "stackPtr @64 -- NOT stackSizeBytes");
static_assert(offsetof(ClothSphereTriPush, midphasePairsNum) == 72, "midphasePairsNum @72");
static_assert(offsetof(ClothSphereTriPush, filterPairs) == 80, "filterPairs @80");
static_assert(offsetof(ClothSphereTriPush, nbFilterPairs) == 88, "nbFilterPairs @88");
static_assert(offsetof(ClothSphereTriPush, stackSizeBytes) == 92, "stackSizeBytes @92");
static_assert(offsetof(ClothSphereTriPush, stackSizeNeededOnDevice) == 96, "stackSizeNeeded @96");
static_assert(offsetof(ClothSphereTriPush, wPoint) == 112, "the writer starts @112");
static_assert(offsetof(ClothSphereTriPush, wMaxContacts) == 184, "maxNumContacts @184");

static const uint32_t GEOM_SPHERE_H = uint32_t(physx::PxGeometryType::eSPHERE);    // PxGeometryType::eSPHERE
static const uint32_t GEOM_CAPSULE_H = uint32_t(physx::PxGeometryType::eCAPSULE);   // PxGeometryType::eCAPSULE

// ---- the host mirror -------------------------------------------------------
// PxQuat::getBasisVector0.
static V3 basis0(const float q[4])
{
    const float x2 = q[0] * 2.0f, w2 = q[3] * 2.0f;
    return V3{ (q[3] * w2) - 1.0f + q[0] * x2,
               (q[2] * w2) + q[1] * x2,
               (-q[1] * w2) + q[2] * x2 };
}

// closestPtPointTriangle (deformableCollision.cuh:146) with the s/t out params.
static V3 cpTri(V3 p, V3 a, V3 b, V3 c, float& s, float& t)
{
    const V3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    const float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) { s = 0.0f; t = 0.0f; return a; }
    const V3 bp = sub(p, b);
    const float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) { s = 1.0f; t = 0.0f; return b; }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        const float v = d1 / (d1 - d3);
        s = v; t = 0.0f; return add(a, scl(ab, v));
    }
    const V3 cp = sub(p, c);
    const float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) { s = 0.0f; t = 1.0f; return c; }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        const float w = d2 / (d2 - d6);
        s = 0.0f; t = w; return add(a, scl(ac, w));
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        s = 1.0f - w; t = w; return add(b, scl(sub(c, b), w));
    }
    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom, w = vc * denom;
    s = v; t = w;
    // LEFT TO RIGHT: the shader writes `a + ab * v + ac * w`, which is
    // (a + ab*v) + ac*w.  Grouping the two scaled edges first instead costs one
    // ULP on the interior-face case and nothing anywhere else -- 15 of 28
    // contacts, measured, before this line read the way it does now.
    return add(add(a, scl(ab, v)), scl(ac, w));
}

static bool validBary(float v, float w)
{
    const float eps = 1e-6f, zero = -eps, one = 1.0f + eps;
    return (v >= zero && v <= one) && (w >= zero && w <= one) && ((v + w) <= one);
}

// PxClamp<float>(x, 0, 1) = PxMin(1, PxMax(0, x)) with the float specialisations
// selectMin/selectMax (PxMath.h:80-101, :142-146; PxUnixMathIntrinsics.h:116-125):
// a -0 (and a NaN) passes through unchanged.
static float pxClamp01(float x)
{
    const float lo = (0.0f > x) ? 0.0f : x;     // selectMax(0, x)
    return (1.0f < lo) ? 1.0f : lo;             // selectMin(1, lo)
}

// P9.86c mutant twins of the three clamps in segSeg (0 = the CUDA).  Variant k+1
// replays the port's pre-P9.86 spelling of site k -- Slang clamp(), i.e. GLSL FClamp,
// which llvmpipe evaluates as maxps/minps: fmax(x, +0) = (x > 0) ? x : +0, so a -0
// (and a NaN) comes out +0.  Sites: 0 = s0 (distanceSegmentSegment.cuh:63), 1 = t2
// (:76), 2 = s (:81).  gSegSegNegZeroIn[k] counts the -0 arguments site k consumed;
// femClothSimCapsuleTriClampZeroSign requires every twin to change its image.
static int gSegSegClampVariant = 0;
static uint32_t gSegSegNegZeroIn[3] = { 0u, 0u, 0u };
static float segClamp(int site, float x, bool used)
{
    if (used && x == 0.0f && std::signbit(x)) ++gSegSegNegZeroIn[site];
    if (gSegSegClampVariant == site + 1)
    {
        const float lo = (x > 0.0f) ? x : 0.0f;  // FMax on llvmpipe: -0 and NaN -> +0
        return (lo < 1.0f) ? lo : 1.0f;
    }
    return pxClamp01(x);
}

static void segSeg(V3 p1, V3 d1, V3 p2, V3 d2, float& s, float& t)
{
    const float eps = 1e-6f;
    const V3 r = sub(p1, p2);
    const float a = dot(d1, d1), e = dot(d2, d2), b = dot(d1, d2), c = dot(d1, r);
    const float aRecip = a > eps ? (1.0f / a) : 0.0f;
    const float eRecip = e > eps ? (1.0f / e) : 0.0f;
    const float f = dot(d2, r);
    const float denom = a * e - b * b;
    const float temp = b * f - c * e;
    const float s0 = segClamp(0, temp / denom, !(eps > denom));   // s0 is dead when parallel
    const float sTmp = eps > denom ? 0.5f : s0;
    const float tTmp = (b * sTmp + f) * eRecip;
    const float t2 = segClamp(1, tTmp, true);
    const float comp = (b * t2 - c) * aRecip;
    s = segClamp(2, comp, true);
    t = t2;
}

static const float MAX_F32_H = 3.4028234663852886e+38f;

// Which of distanceSegmentTriangleSquared's five exits the last call took.  The
// mirror is line-for-line the shader, so counting exits HERE is evidence about the
// shader's coverage -- and it turns "the rig reaches three exits" from a claim in a
// comment into a number the fixture prints and checks.
enum SegTriExit { EXIT_PLANE_CROSS = 0, EXIT_BOTH_INTERIOR, EXIT_CON0_EDGE, EXIT_CON1_EDGE, EXIT_NEITHER };
static int gSegTriExit = -1;

static float segTriSq(V3 p, V3 q, V3 a, V3 b, V3 c, float& outT, float& outU, float& outV)
{
    outT = 0.0f; outU = 0.0f; outV = 0.0f;
    gSegTriExit = -1;
    const V3 pq = sub(q, p), ab = sub(b, a), ac = sub(c, a), bc = sub(c, b);
    const V3 ap = sub(p, a), aq = sub(q, a);
    const float d00 = dot(ab, ab), d01 = dot(ab, ac), d11 = dot(ac, ac);
    const float tDenom = d00 * d11 - d01 * d01;
    const float bdenom = tDenom > 0.0f ? 1.0f / tDenom : MAX_F32_H;
    const V3 n = norm(crs(ab, ac));
    const float dist3 = dot(ap, n), sqDist3 = dist3 * dist3;
    const float dist4 = dot(aq, n), sqDist4 = dist4 * dist4;
    const float dMul = dist3 * dist4;

    if (dMul <= 0.0f)
    {
        const float nom = -dot(n, ap);
        const float denom = 1.0f / dot(n, pq);
        const float t0 = nom * denom;
        const V3 ip = add(p, scl(pq, t0));
        const V3 v2 = sub(ip, a);
        const float d20 = dot(v2, ab), d21 = dot(v2, ac);
        const float v00 = (d11 * d20 - d01 * d21) * bdenom;
        const float w00 = (d00 * d21 - d01 * d20) * bdenom;
        if (validBary(v00, w00))
        {
            gSegTriExit = EXIT_PLANE_CROSS;
            outT = t0; outU = v00; outV = w00; return 0.0f;
        }
    }

    const V3 pV20 = sub(sub(p, scl(n, dist3)), a);
    const float v0 = (d11 * dot(pV20, ab) - d01 * dot(pV20, ac)) * bdenom;
    const float w0 = (d00 * dot(pV20, ac) - d01 * dot(pV20, ab)) * bdenom;
    const bool con0 = validBary(v0, w0);

    const V3 qV20 = sub(sub(q, scl(n, dist4)), a);
    const float v1 = (d11 * dot(qV20, ab) - d01 * dot(qV20, ac)) * bdenom;
    const float w1 = (d00 * dot(qV20, ac) - d01 * dot(qV20, ab)) * bdenom;
    const bool con1 = validBary(v1, w1);

    if (con0 && con1)
    {
        gSegTriExit = EXIT_BOTH_INTERIOR;
        if (sqDist4 > sqDist3) { outT = 0.0f; outU = v0; outV = w0; return sqDist3; }
        outT = 1.0f; outU = v1; outV = w1; return sqDist4;
    }

    float t00, t10, t01, t11, t02, t12;
    segSeg(p, pq, a, ab, t00, t10);
    segSeg(p, pq, b, bc, t01, t11);
    segSeg(p, pq, a, ac, t02, t12);
    V3 tv = sub(add(p, scl(pq, t00)), add(a, scl(ab, t10)));
    const float sqDist0 = dot(tv, tv);
    tv = sub(add(p, scl(pq, t01)), add(b, scl(bc, t11)));
    const float sqDist1 = dot(tv, tv);
    tv = sub(add(p, scl(pq, t02)), add(a, scl(ac, t12)));
    const float sqDist2 = dot(tv, tv);

    const float u10 = t10, v10 = 0.0f;
    const float u11 = 1.0f - t11, v11 = t11;
    const float u12 = 0.0f, v12 = t12;

    float sqDistPE, uEdge, vEdge, tSeg;
    if (sqDist1 > sqDist0 && sqDist2 > sqDist0) { sqDistPE = sqDist0; uEdge = u10; vEdge = v10; tSeg = t00; }
    else if (sqDist2 > sqDist1)                 { sqDistPE = sqDist1; uEdge = u11; vEdge = v11; tSeg = t01; }
    else                                        { sqDistPE = sqDist2; uEdge = u12; vEdge = v12; tSeg = t02; }

    if (con0)
    {
        gSegTriExit = EXIT_CON0_EDGE;
        if (sqDistPE > sqDist3) { outT = 0.0f; outU = v0; outV = w0; return sqDist3; }
        outT = tSeg; outU = uEdge; outV = vEdge; return sqDistPE;
    }
    if (con1)
    {
        gSegTriExit = EXIT_CON1_EDGE;
        if (sqDistPE > sqDist4) { outT = 1.0f; outU = v1; outV = w1; return sqDist4; }
        outT = tSeg; outU = uEdge; outV = vEdge; return sqDistPE;
    }
    gSegTriExit = EXIT_NEITHER;
    outT = tSeg; outU = uEdge; outV = vEdge; return sqDistPE;
}

struct Hit { F4 point, normalPen, bary; };

static bool triSphere(V3 a, V3 b, V3 c, V3 centre, float radius, float cDist, Hit& out)
{
    const float inflated = radius + cDist;
    const float sqInflated = inflated * inflated;
    float s, t;
    const V3 closest = cpTri(centre, a, b, c, s, t);
    const V3 v = sub(closest, centre);
    const float sqDist = dot(v, v);
    if (!(sqInflated > sqDist)) return false;
    const float dist = std::sqrt(sqDist);
    const V3 n = (dist > 1e-8f) ? scl(v, 1.0f / dist) : norm(crs(sub(b, a), sub(c, a)));
    const V3 p = add(centre, scl(n, radius));
    out.point = F4{ p.x, p.y, p.z, 0.0f };
    out.normalPen = F4{ n.x, n.y, n.z, dist - radius };
    out.bary = F4{ 1.0f - s - t, s, t, 0.0f };
    return true;
}

static bool triCapsule(const float rq[4], V3 rp, float halfHeight, float radius,
                       V3 a, V3 b, V3 c, float cDist, Hit& out)
{
    const float inflated = radius + cDist;
    const float sqInflated = inflated * inflated;
    const V3 dir = basis0(rq);
    const V3 tmp = scl(dir, halfHeight);
    const V3 p = add(rp, tmp), q = sub(rp, tmp);
    float t, u, v;
    const float sqDist = segTriSq(p, q, a, b, c, t, u, v);
    if (!(sqInflated > sqDist)) return false;
    const V3 pq = sub(q, p);
    const V3 pointOnSegment = add(p, scl(pq, t));
    const float w = 1.0f - u - v;
    const V3 pointOnTriangle = add(add(scl(a, w), scl(b, u)), scl(c, v));
    const V3 normal = norm(sub(pointOnTriangle, pointOnSegment));
    const float dist = std::sqrt(sqDist);
    const V3 pOnCap = add(pointOnSegment, scl(normal, radius));
    out.point = F4{ pOnCap.x, pOnCap.y, pOnCap.z, 0.0f };
    out.normalPen = F4{ normal.x, normal.y, normal.z, dist - radius };
    out.bary = F4{ w, u, v, 0.0f };
    return true;
}
} // namespace p929st

PXG_TEST(femClothSimSphereTriContactGen)
{
    using namespace p816dc;
    using namespace p929st;
    using p817cg::V3;
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t CLOTH_ID = 1u;
    const float CD_CLOTH = 0.05f, CD_SPHERE = 0.15f, CD_CAPSULE = 0.10f;
    const float REST_SPHERE = 0.031f, REST_CAPSULE = 0.077f;   // S6: per manager
    const uint32_t MAT_SPHERE = 0x2Au, MAT_CAPSULE = 0x35u;

    // ---- the two rigids --------------------------------------------------------------
    // The sphere: radius comes from scale.y, and scale.x/scale.z are DIFFERENT so a port
    // reading the wrong component moves the surface.
    const V3 spP{ 0.30f, 0.20f, -0.10f };
    const float spQ[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    const float SP_RADIUS = 0.40f;

    // The capsule: halfHeight from scale.x, radius from scale.y, axis = basis vector 0 of
    // its pose -- rotated so the axis is NOT a coordinate axis and a dropped rotation is
    // visible in every contact.
    const float ch = std::sin(0.35f), cc = std::cos(0.35f);
    const float capQ[4] = { 0.0f, 0.0f, ch, cc };              // 0.7 rad about +z
    const V3 capP{ -0.20f, 0.55f, 0.25f };
    const float CAP_HALF = 0.60f, CAP_RADIUS = 0.25f;

    // ---- the cloth: a fan plus four hand-placed triangles ------------------------------
    // S1: well past one warp, so a per-warp driver processes 1 in 32 of them.
    const uint32_t NB_FAN = 40u;
    std::vector<F4> cv;
    std::vector<TriIdx4> ci;
    auto addTri = [&](V3 a, V3 b, V3 c) {
        const uint32_t base = uint32_t(cv.size());
        cv.push_back(F4{ a.x, a.y, a.z, 1.0f });
        cv.push_back(F4{ b.x, b.y, b.z, 1.0f });
        cv.push_back(F4{ c.x, c.y, c.z, 1.0f });
        ci.push_back(TriIdx4{ base, base + 1u, base + 2u, 0u });
    };
    for (uint32_t i = 0; i < NB_FAN; ++i)
    {
        // A fan of small triangles swept through both rigids: the offset walks in y so
        // some triangles cut the capsule's axis plane and some hang off an edge.
        const float u = float(i) * 0.055f - 1.0f;
        const V3 a{ -0.35f + u * 0.10f, 0.10f + u * 0.55f, -0.30f + u * 0.05f };
        const V3 b{ 0.45f + u * 0.10f,  0.05f + u * 0.55f, -0.20f + u * 0.05f };
        const V3 c{ 0.05f + u * 0.10f,  0.15f + u * 0.55f,  0.50f + u * 0.05f };
        addTri(a, b, c);
    }
    // Four more, placed by hand rather than swept, because the fan alone reached only
    // three of distanceSegmentTriangleSquared's five exits (measured: plane-crossing,
    // con0-edge and neither).  The capsule's axis is basis vector 0 of a 0.7 rad turn
    // about +z, so its segment runs from about (0.259, 0.937, 0.25) to (-0.659, 0.164,
    // 0.25) with radius 0.25 and a 0.40 inflated reach.
    //
    // ODD indices belong to the capsule manager -- 41 and 43 -- so those are the two
    // that matter; 40 and 42 are their sphere-side partners and are placed out of the
    // sphere's reach so they add nothing but the parity.
    const float PLANE_Z = -0.05f;                 // 0.30 below the segment: inside 0.40
    // 40 -- the SPHERE's near-tangent case, and the only reason it is hand-placed:
    // triSphereCollide guards its normal with `dist > 1e-8`, falling back to the
    // triangle's own surface normal below that.  Every swept case has the sphere centre
    // millimetres from the surface, so the guard was reached and never decided anything
    // -- a mutant moving the threshold to 1e-4 survived.  Here the centre sits 1e-5
    // BELOW a plane whose winding makes the surface normal point UP, so the two
    // branches disagree in SIGN and not merely in the last bits.
    {
        const float Y = spP.y - 1.0e-5f;
        addTri(V3{ spP.x - 1.0f, Y, spP.z - 1.0f },
               V3{ spP.x,        Y, spP.z + 1.0f },
               V3{ spP.x + 1.0f, Y, spP.z - 1.0f });
    }
    // 41 -- BOTH projections interior: a slab wide enough to contain both endpoints,
    // parallel to the segment so the plane is never crossed (dMul > 0).
    addTri(V3{ -2.0f, -2.0f, PLANE_Z }, V3{ 2.0f, -2.0f, PLANE_Z }, V3{ 0.0f, 2.5f, PLANE_Z });
    addTri(V3{ 9.0f, 9.0f, 9.2f }, V3{ 9.5f, 9.0f, 9.2f }, V3{ 9.0f, 9.5f, 9.2f });   // 42, far away
    // 43 -- only q's projection interior: the same plane, but a triangle that covers the
    // q end of the segment and not the p end.
    addTri(V3{ -1.2f, -0.3f, PLANE_Z }, V3{ -0.2f, -0.3f, PLANE_Z }, V3{ -0.7f, 0.7f, PLANE_Z });
    // A SECOND fan, aimed at the capsule from many directions with SMALL triangles.
    // The first fan's triangles are large enough that the capsule almost always lands
    // on their interior, so the edge machinery under distanceSegmentTriangleSquared --
    // the three segment-segment calls, their clamps, and the edge pick -- was reached
    // but never decided anything.  Six mutants survived on exactly that.  Small
    // triangles ringing the axis put the closest feature on an EDGE or a VERTEX.
    {
        const V3 dir{ std::cos(0.35f) * std::cos(0.35f) - std::sin(0.35f) * std::sin(0.35f),
                      2.0f * std::sin(0.35f) * std::cos(0.35f), 0.0f };   // basis0(capQ)
        const V3 up{ 0.0f, 0.0f, 1.0f };
        const V3 side = crs(dir, up);
        for (uint32_t j = 0; j < 24u; ++j)
        {
            const float along = (float(j) / 23.0f * 2.0f - 1.0f) * CAP_HALF * 1.15f;
            const float ang = float(j) * 0.5236f;                 // 30 degrees apart
            const float rad = 0.30f + 0.04f * float(j % 3u);
            const V3 centre = add(add(capP, scl(dir, along)),
                                  add(scl(side, std::cos(ang) * rad), scl(up, std::sin(ang) * rad)));
            const float sz = 0.11f + 0.01f * float(j % 4u);
            const V3 e0 = add(scl(side, std::cos(ang + 1.1f)), scl(up, std::sin(ang + 1.1f)));
            const V3 e1 = add(scl(dir, 0.7f), scl(up, 0.3f));
            addTri(add(centre, scl(e0, sz)),
                   add(centre, scl(e1, sz)),
                   add(centre, scl(add(e0, e1), -sz * 0.6f)));
        }
    }
    // One more, for the last branch a swept triangle cannot reach:
    // distanceSegmentSegmentSquared's PARALLEL case, where the closest-point system is
    // singular and the CUDA picks the segment midpoint (0.5) rather than an endpoint.
    // Edge ab below runs exactly along the capsule's own axis, so denom cancels to
    // zero and `eps > denom` fires.  A mutant replacing that 0.5 with 0.0 survived
    // every swept triangle, because none of them had an edge parallel to the axis.
    {
        const V3 dir{ std::cos(0.35f) * std::cos(0.35f) - std::sin(0.35f) * std::sin(0.35f),
                      2.0f * std::sin(0.35f) * std::cos(0.35f), 0.0f };
        const V3 up{ 0.0f, 0.0f, 1.0f };
        const V3 side = crs(dir, up);
        const V3 base = add(capP, scl(side, 0.30f));
        addTri(sub(base, scl(dir, 0.25f)), add(base, scl(dir, 0.25f)), add(base, scl(up, 0.55f)));
    }
    const uint32_t NB_TRIS = uint32_t(ci.size());

    // Which contact manager each triangle is tested against.  The fan alternates, as
    // before; every hand-placed triangle from index NB_FAN on is aimed at the CAPSULE,
    // because they exist for the capsule core's branches.  Index parity used to decide
    // this, which meant a case could only be aimed by choosing its position in the list.
    std::vector<uint32_t> triCm(NB_TRIS, 1u);
    for (uint32_t i = 0; i < NB_FAN; ++i) triCm[i] = (i & 1u);
    triCm[40u] = 0u;                     // the near-tangent SPHERE case, below
    DevBuf bCv = makeDev(dev, cv);
    if (!bCv.addr) { std::printf("  NOTE: no BDA -> skip\n"); dev->destroyBuffer(bCv.h); return; }
    DevBuf bCi = makeDev(dev, ci);

    PxgFEMCloth cloth = makeZeroDesc();
    cloth.mPosition_InvMass = reinterpret_cast<decltype(cloth.mPosition_InvMass)>(size_t(bCv.addr));
    cloth.mTriangleVertexIndices = reinterpret_cast<decltype(cloth.mTriangleVertexIndices)>(size_t(bCi.addr));
    cloth.mNbTriangles = NB_TRIS;
    cloth.mNbVerts = uint32_t(cv.size());
    std::vector<PxgFEMCloth> descArr(CLOTH_ID + 1u, makeZeroDesc());
    descArr[CLOTH_ID] = cloth;
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- shapes: 0 = cloth, 1 = sphere, 2 = capsule -----------------------------------
    std::vector<PxgShapeHost> shapes(3);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0] = makeClothShape(CLOTH_ID);
    shapes[1].type = GEOM_SPHERE_H;  shapes[1].particleOrSoftbodyId = 0xFFFFFFFFu;
    shapes[1].sx = 0.77f; shapes[1].sy = SP_RADIUS; shapes[1].sz = 0.11f;   // only .y is the radius
    shapes[1].srw = 1.0f; shapes[1].materialIndex = MAT_SPHERE;
    shapes[2].type = GEOM_CAPSULE_H; shapes[2].particleOrSoftbodyId = 0xFFFFFFFFu;
    shapes[2].sx = CAP_HALF; shapes[2].sy = CAP_RADIUS; shapes[2].sz = 0.13f;
    shapes[2].srw = 1.0f; shapes[2].materialIndex = MAT_CAPSULE;
    DevBuf bShapes = makeDev(dev, shapes);

    // cache refs: 0 = cloth, 1 = sphere, 2 = capsule.  Manager 0 puts the cloth FIRST,
    // manager 1 puts it SECOND -- loadClothRigidPair routes by TYPE, and a port that
    // trusted slot order would fail exactly one of the two.
    std::vector<CmInputHost> cms = { { 0u, 1u, 0u, 1u }, { 2u, 0u, 2u, 0u } };
    DevBuf bCms = makeDev(dev, cms);

    std::vector<float> xf(3 * 8, 0.0f);
    xf[3] = 1.0f;                                     // cloth pose: identity, at the origin
    xf[8 + 0] = spQ[0]; xf[8 + 1] = spQ[1]; xf[8 + 2] = spQ[2]; xf[8 + 3] = spQ[3];
    xf[8 + 4] = spP.x; xf[8 + 5] = spP.y; xf[8 + 6] = spP.z;
    xf[16 + 0] = capQ[0]; xf[16 + 1] = capQ[1]; xf[16 + 2] = capQ[2]; xf[16 + 3] = capQ[3];
    xf[16 + 4] = capP.x; xf[16 + 5] = capP.y; xf[16 + 6] = capP.z;
    DevBuf bXf = makeDev(dev, xf);

    std::vector<float> cd = { CD_CLOTH, CD_SPHERE, CD_CAPSULE };
    std::vector<float> rd = { REST_SPHERE, REST_CAPSULE };
    DevBuf bCd = makeDev(dev, cd), bRd = makeDev(dev, rd);

    const uint64_t NODE_SPHERE = 0x0000000700000031ull, NODE_CAPSULE = 0x0000000200000019ull;
    std::vector<uint64_t> remap = { 0ull, NODE_SPHERE, NODE_CAPSULE };
    DevBuf bRemap = makeDev(dev, remap);

    // ---- the filter list (S7) ---------------------------------------------------------
    // Triangle 6: ALL THREE corners against BOTH rigids -> must vanish from both runs.
    // Triangle 9: TWO corners only -> must survive, and its absence would mean the port
    // used `||`.
    // Both are triangles this rig MEASURED as contacting: triangle 12 is even, so
    // it belongs to manager 0 (the SPHERE), and 21 is odd, so it belongs to
    // manager 1 (the CAPSULE).  Picking them by eye is what the first version did,
    // and it drew a filtered triangle that never touched anything -- a check that
    // cannot fail is not a check, so the non-vacuity of BOTH is asserted below.
    const uint32_t FILTERED_TRI = 12u, PARTIAL_TRI = 21u;
    std::vector<RigidFilterPairHost> filters;
    auto pushFilter = [&](uint64_t node, uint32_t vert) {
        RigidFilterPairHost f{};
        f.index0 = node;
        f.index1 = uint64_t((CLOTH_ID << 20) | vert);
        filters.push_back(f);
    };
    for (uint64_t node : { NODE_SPHERE, NODE_CAPSULE })
    {
        for (uint32_t k = 0; k < 3u; ++k) pushFilter(node, FILTERED_TRI * 3u + k);
        for (uint32_t k = 0; k < 2u; ++k) pushFilter(node, PARTIAL_TRI * 3u + k);
    }
    // findRigid does a BINARY search: the list must be sorted by (index0, index1).
    std::sort(filters.begin(), filters.end(),
              [](const RigidFilterPairHost& a, const RigidFilterPairHost& b) {
                  if (a.index0 != b.index0) return a.index0 < b.index0;
                  return a.index1 < b.index1;
              });
    DevBuf bFilters = makeDev(dev, filters);

    // ---- the pair stack ----------------------------------------------------------------
    // S4/S8: the midphase claims MORE than the stack holds.  The surplus entries are
    // addressable and carry triangle 0, which makes contact, so an unclamped walk shows up
    // as EXTRA contacts rather than as a crash.
    const uint32_t NB_STACK = NB_TRIS;
    const uint32_t NB_FOUND = NB_TRIS + 24u;
    std::vector<uint32_t> stack(NB_FOUND * 4u, 0u);
    for (uint32_t i = 0; i < NB_FOUND; ++i)
    {
        const uint32_t tri = (i < NB_TRIS) ? i : 0u;
        stack[i * 4 + 0] = triCm[tri];
        stack[i * 4 + 1] = tri;
    }
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> nFound = { NB_FOUND };
    DevBuf bNum = makeDev(dev, nFound);

    const uint32_t MAXC = 96u;
    WriterBufs w = makeWriter(dev, MAXC);
    std::vector<uint32_t> zeroNeeded = { 0u };
    DevBuf bNeeded = makeDev(dev, zeroNeeded);

    ClothSphereTriPush p = {};
    p.toleranceLength = 1.0f;
    p.cmInputs = bCms.addr; p.transformCache = bXf.addr;
    p.contactDistance = bCd.addr; p.restDistance = bRd.addr;
    p.gpuShapes = bShapes.addr; p.femClothes = bDesc.addr;
    p.shapeToRigidRemapTable = bRemap.addr;
    p.stackSizeBytes = NB_STACK * 16u;
    p.stackPtr = bStack.addr; p.midphasePairsNum = bNum.addr;
    p.filterPairs = bFilters.addr; p.nbFilterPairs = uint32_t(filters.size());
    p.stackSizeNeededOnDevice = bNeeded.addr;
    p.wPoint = w.point.addr; p.wNormalPen = w.normalPen.addr;
    p.wBary = w.barycentric.addr; p.wInfo = w.info.addr; p.wCount = w.count.addr;
    p.wByX = w.byRigid.addr; p.wTempByX = w.tempByRigid.addr;
    p.wIdxByX = w.idxByRigid.addr; p.wSorted32 = w.sortedByRigid.addr;
    p.wMaxContacts = MAXC;

    Kernel k("femClothContactGenBda.spv", "cloth_SphereContactGenLaunch",
             {}, sizeof(ClothSphereTriPush));
    k.dispatch(1024u, {}, p, 1u, 1u);      // PxgNarrowphaseCore.cpp:5760
    dev->flush();

    // ---- the expected set, from the mirror ---------------------------------------------
    struct Exp { uint32_t tri, cm; uint64_t node; uint32_t mat; Hit h; float rest; };
    std::vector<Exp> expected;
    uint32_t nSphere = 0u, nCapsule = 0u;
    uint32_t exitHits[5] = { 0u, 0u, 0u, 0u, 0u };   // segTriSq coverage, measured
    for (uint32_t i = 0; i < NB_STACK; ++i)   // S4: the CLAMPED walk
    {
        const uint32_t cm = stack[i * 4 + 0], tri = stack[i * 4 + 1];
        const bool isSphere = (cm == 0u);
        const uint64_t node = isSphere ? NODE_SPHERE : NODE_CAPSULE;

        uint32_t nFiltered = 0u;
        for (uint32_t kk = 0; kk < 3u; ++kk)
        {
            const uint64_t key = uint64_t((CLOTH_ID << 20) | (tri * 3u + kk));
            for (const auto& f : filters)
                if (f.index0 == node && f.index1 == key) { ++nFiltered; break; }
        }
        if (nFiltered == 3u) continue;        // S7: AND over all three

        const V3 a{ cv[tri * 3 + 0].x, cv[tri * 3 + 0].y, cv[tri * 3 + 0].z };
        const V3 b{ cv[tri * 3 + 1].x, cv[tri * 3 + 1].y, cv[tri * 3 + 1].z };
        const V3 c{ cv[tri * 3 + 2].x, cv[tri * 3 + 2].y, cv[tri * 3 + 2].z };

        Hit h{};
        bool hit;
        if (isSphere)
            hit = triSphere(a, b, c, spP, SP_RADIUS, CD_CLOTH + CD_SPHERE, h);
        else
        {
            hit = triCapsule(capQ, capP, CAP_HALF, CAP_RADIUS, a, b, c, CD_CLOTH + CD_CAPSULE, h);
            if (gSegTriExit >= 0 && gSegTriExit < 5) ++exitHits[gSegTriExit];
        }
        if (!hit) continue;

        h.point.w = isSphere ? REST_SPHERE : REST_CAPSULE;
        expected.push_back(Exp{ tri, cm, node, isSphere ? MAT_SPHERE : MAT_CAPSULE, h,
                                h.point.w });
        if (isSphere) ++nSphere; else ++nCapsule;
    }

    const uint32_t got = downloadBuffer<uint32_t>(dev, w.count.h, 1)[0];
    PXG_CHECK_MSG(nSphere > 0u && nCapsule > 0u,
                  "the rig produced %u sphere and %u capsule contacts; BOTH must be "
                  "non-zero or S2's bare `else` is not exercised", nSphere, nCapsule);
    PXG_CHECK_MSG(got == uint32_t(expected.size()),
                  "contact COUNT %u, host mirror says %zu (%u sphere + %u capsule over %u "
                  "clamped pairs of %u claimed)",
                  got, expected.size(), nSphere, nCapsule, NB_STACK, NB_FOUND);

    // Contacts are reserved with a racing atomicAdd, so compare as a MULTISET.
    const auto pt = downloadBuffer<F4>(dev, w.point.h, MAXC);
    const auto np = downloadBuffer<F4>(dev, w.normalPen.h, MAXC);
    const auto ba = downloadBuffer<F4>(dev, w.barycentric.h, MAXC);
    const auto info = downloadBuffer<uint32_t>(dev, w.info.h, MAXC * 4u);

    struct Rec { uint32_t key[16]; };
    auto less = [](const Rec& x, const Rec& y) {
        for (int i = 0; i < 16; ++i) if (x.key[i] != y.key[i]) return x.key[i] < y.key[i];
        return false;
    };
    std::vector<Rec> gpu, ref;
    for (uint32_t i = 0; i < got && i < MAXC; ++i)
    {
        Rec r{};
        r.key[0] = info[i * 4 + 2];                     // pairInd1 = (clothId<<20)|tri
        r.key[1] = info[i * 4 + 0]; r.key[2] = info[i * 4 + 1];   // pairInd0 (u64)
        r.key[3] = info[i * 4 + 3];                     // material
        const float src[12] = { pt[i].x, pt[i].y, pt[i].z, pt[i].w,
                                np[i].x, np[i].y, np[i].z, np[i].w,
                                ba[i].x, ba[i].y, ba[i].z, ba[i].w };
        for (int j = 0; j < 12; ++j) r.key[4 + j] = f2u(src[j]);
        gpu.push_back(r);
    }
    for (const Exp& e : expected)
    {
        Rec r{};
        r.key[0] = (CLOTH_ID << 20) | e.tri;
        r.key[1] = uint32_t(e.node & 0xffffffffull); r.key[2] = uint32_t(e.node >> 32);
        r.key[3] = e.mat & 0x7fffffffu;
        const float src[12] = { e.h.point.x, e.h.point.y, e.h.point.z, e.h.point.w,
                                e.h.normalPen.x, e.h.normalPen.y, e.h.normalPen.z, e.h.normalPen.w,
                                e.h.bary.x, e.h.bary.y, e.h.bary.z, e.h.bary.w };
        for (int j = 0; j < 12; ++j) r.key[4 + j] = f2u(src[j]);
        ref.push_back(r);
    }
    std::sort(gpu.begin(), gpu.end(), less);
    std::sort(ref.begin(), ref.end(), less);

    static const char* const KEY_NAME[16] = {
        "pairInd1", "pairInd0.lo", "pairInd0.hi", "material",
        "point.x", "point.y", "point.z", "point.w(rest)",
        "normal.x", "normal.y", "normal.z", "penetration",
        "bary.x", "bary.y", "bary.z", "bary.w"
    };
    uint32_t mismatches = 0u;
    int firstField = -1; size_t firstAt = 0;
    for (size_t i = 0; i < gpu.size() && i < ref.size(); ++i)
        for (int j = 0; j < 16; ++j)
            if (gpu[i].key[j] != ref[i].key[j])
            {
                ++mismatches;
                if (firstField < 0) { firstField = j; firstAt = i; }
                break;
            }
    if (mismatches != 0u && firstField >= 0)
    {
        const uint32_t gw = gpu[firstAt].key[firstField], rw = ref[firstAt].key[firstField];
        float gf, rf; std::memcpy(&gf, &gw, 4); std::memcpy(&rf, &rw, 4);
        std::printf("    first difference at sorted slot %zu, field %s: gpu 0x%08X (%.9g) "
                    "vs mirror 0x%08X (%.9g); tri %u\n",
                    firstAt, KEY_NAME[firstField], gw, gf, rw, rf,
                    gpu[firstAt].key[0] & ((1u << 20) - 1u));
    }
    PXG_CHECK_MSG(mismatches == 0u,
                  "%u of %zu contacts differ BIT-EXACTLY from the host mirror of "
                  "triSphereCollide / fcCapsuleCollision and the caller glue",
                  mismatches, gpu.size());

    // S7 both ways, read off the emitted set rather than argued.
    uint32_t sawFiltered = 0u, sawPartial = 0u;
    for (const Rec& r : gpu)
    {
        const uint32_t tri = r.key[0] & ((1u << 20) - 1u);
        if (tri == FILTERED_TRI) ++sawFiltered;
        if (tri == PARTIAL_TRI) ++sawPartial;
    }
    // NON-VACUITY: the filtered triangle must be one that WOULD have produced a
    // contact.  Run the mirror over it with the filter ignored and demand a hit --
    // otherwise `sawFiltered == 0` passes for a triangle that simply misses, which
    // is exactly what the first version of this rig did.
    {
        const V3 fa{ cv[FILTERED_TRI * 3 + 0].x, cv[FILTERED_TRI * 3 + 0].y, cv[FILTERED_TRI * 3 + 0].z };
        const V3 fb{ cv[FILTERED_TRI * 3 + 1].x, cv[FILTERED_TRI * 3 + 1].y, cv[FILTERED_TRI * 3 + 1].z };
        const V3 fc{ cv[FILTERED_TRI * 3 + 2].x, cv[FILTERED_TRI * 3 + 2].y, cv[FILTERED_TRI * 3 + 2].z };
        Hit fh{};
        const bool wouldHit = (triCm[FILTERED_TRI] == 0u)
            ? triSphere(fa, fb, fc, spP, SP_RADIUS, CD_CLOTH + CD_SPHERE, fh)
            : triCapsule(capQ, capP, CAP_HALF, CAP_RADIUS, fa, fb, fc, CD_CLOTH + CD_CAPSULE, fh);
        PXG_CHECK_MSG(wouldHit,
                      "triangle %u produces no contact even UNFILTERED, so the filter check "
                      "below cannot fail and proves nothing", FILTERED_TRI);
    }
    PXG_CHECK_MSG(sawFiltered == 0u,
                  "triangle %u has ALL THREE corners filtered against both rigids and still "
                  "produced %u contacts", FILTERED_TRI, sawFiltered);
    PXG_CHECK_MSG(sawPartial > 0u,
                  "triangle %u has only TWO of three corners filtered and produced nothing: "
                  "the filter is an AND (S7), and this is what an `||` port loses", PARTIAL_TRI);

    // The capsule core's branch coverage, measured rather than asserted in a comment.
    // distanceSegmentTriangleSquared has five exits and they are sixty lines apart; a
    // transcription slip in one is invisible from the others, so the rig has to reach
    // more than one or the bit-exact comparison above is only testing a single path.
    {
        static const char* const EXIT_NAME[5] = { "plane-crossing", "both-interior",
                                                  "con0-edge", "con1-edge", "neither" };
        uint32_t distinct = 0u;
        std::printf("    segTriSq exits:");
        for (int e = 0; e < 5; ++e)
        {
            std::printf(" %s=%u", EXIT_NAME[e], exitHits[e]);
            if (exitHits[e] != 0u) ++distinct;
        }
        std::printf("\n");
        // ALL FIVE, not "at least a few".  The fan alone reached three; triangles 41
        // and 43 were added by hand for the two it missed, so the honest gate is the
        // number the rig actually achieves.
        PXG_CHECK_MSG(distinct == 5u,
                      "the capsule cases reached %u of distanceSegmentTriangleSquared's five "
                      "exits, not all five; the bit-exact comparison above would then be "
                      "checking a subset of the branches and calling it the function", distinct);
    }

    // S8: the high-water mark takes the UNCLAMPED count.
    const uint32_t needed = downloadBuffer<uint32_t>(dev, bNeeded.h, 1)[0];
    PXG_CHECK_MSG(needed == NB_FOUND * 16u,
                  "stackSizeNeeded is %u, expected the UNCLAMPED %u * 16 = %u; the clamped "
                  "count would give %u and the stack would never grow",
                  needed, NB_FOUND, NB_FOUND * 16u, NB_STACK * 16u);

    std::printf("    [ note ] %zu contacts (%u sphere, %u capsule) over %u clamped pairs of "
                "%u claimed; triangle %u filtered out, %u survived on a partial filter\n",
                expected.size(), nSphere, nCapsule, NB_STACK, NB_FOUND, FILTERED_TRI, PARTIAL_TRI);

    // ---- S1, second half: the GRID STRIDE ------------------------------------------
    // The first dispatch cannot see the stride at all.  With 44 pairs every thread runs
    // one iteration and stops, so `i += SPHTRI_STRIDE` is never taken and a mutant that
    // replaces 1024*128 with 1024*4 survives -- measured, as a survivor, before this
    // block existed.  A survivor that the rig cannot reach is a hole in the rig.
    //
    // So: hand the kernel MORE pairs than the grid has threads.  With
    // numPairs = SPHTRI_STRIDE + 8, threads 0..7 -- and only they -- come back for a
    // second iteration, and the eight contacting pairs are put exactly there.  Every
    // pair below the stride names a triangle that is far from both rigids, so the whole
    // first pass is silent and the count is the second pass alone.
    //
    // A SMALLER stride does not merely shift the window: with stride S' < S every index
    // is visited by S/S' different threads, so the eight tail hits come back S/S' times.
    // The check is therefore on the exact count, which separates "wrong window" from
    // "right window, wrong multiplicity" without needing to know which mutant it is.
    {
        const uint32_t GRID = 1024u, BLOCK = 128u;
        const uint32_t STRIDE = GRID * BLOCK;             // 131072, the shader's constant
        const uint32_t TAIL = 8u;
        const uint32_t FAR_TRI = 0u;                      // measured below as contact-free
        const uint32_t HIT_TRI = 4u;                      // measured above as contacting

        std::vector<uint32_t> big(size_t(STRIDE + TAIL) * 4u, 0u);
        for (uint32_t i = 0; i < STRIDE + TAIL; ++i)
            big[size_t(i) * 4 + 1] = (i < STRIDE) ? FAR_TRI : HIT_TRI;   // cm 0 = the sphere
        DevBuf bBig = makeDev(dev, big);
        std::vector<uint32_t> bigFound = { STRIDE + TAIL };
        DevBuf bBigNum = makeDev(dev, bigFound);
        DevBuf bBigNeeded = makeDev(dev, zeroNeeded);
        WriterBufs w2 = makeWriter(dev, 64u);

        ClothSphereTriPush q = p;
        q.stackPtr = bBig.addr; q.midphasePairsNum = bBigNum.addr;
        q.stackSizeBytes = (STRIDE + TAIL) * 16u;
        q.stackSizeNeededOnDevice = bBigNeeded.addr;
        q.filterPairs = 0ull; q.nbFilterPairs = 0u;
        q.wPoint = w2.point.addr; q.wNormalPen = w2.normalPen.addr;
        q.wBary = w2.barycentric.addr; q.wInfo = w2.info.addr; q.wCount = w2.count.addr;
        q.wByX = w2.byRigid.addr; q.wTempByX = w2.tempByRigid.addr;
        q.wIdxByX = w2.idxByRigid.addr; q.wSorted32 = w2.sortedByRigid.addr;
        q.wMaxContacts = 64u;
        k.dispatch(GRID, {}, q, 1u, 1u);
        dev->flush();

        const uint32_t n2 = downloadBuffer<uint32_t>(dev, w2.count.h, 1)[0];
        PXG_CHECK_MSG(n2 == TAIL,
                      "the stride run produced %u contacts, expected exactly %u.  The "
                      "kernel was given %u pairs against a grid of %u threads, so only "
                      "threads 0..%u come back for a second iteration -- more than %u "
                      "means the stride is too small and pairs are being visited by "
                      "several threads, fewer means the second iteration never happened",
                      n2, TAIL, STRIDE + TAIL, STRIDE, TAIL - 1u, TAIL);

        std::printf("    [ note ] stride run: %u pairs over %u threads -> %u contacts "
                    "from the %u-pair tail\n", STRIDE + TAIL, STRIDE, n2, TAIL);

        destroyWriter(dev, w2);
        for (PxgBufferHandle h : { bBig.h, bBigNum.h, bBigNeeded.h }) dev->destroyBuffer(h);
    }

    destroyWriter(dev, w);
    for (PxgBufferHandle h : { bCv.h, bCi.h, bDesc.h, bShapes.h, bCms.h, bXf.h, bCd.h,
                               bRd.h, bRemap.h, bFilters.h, bStack.h, bNum.h, bNeeded.h })
        dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// P9.86c -- the three PxClamp(., 0, 1) of distanceSegmentSegmentSquared
// (distanceSegmentSegment.cuh:63/:76/:81) fed a -0.
//
// PxClamp<float> is selectMin(1, selectMax(0, v)) and passes a -0 through.  The port
// once spelled all three as Slang clamp() -- GLSL FClamp, which llvmpipe evaluates
// with maxps/minps, so fmax(-0, +0) = +0.  femClothSimSphereTriContactGen compares
// bit-exactly but its capsule pose and triangles are generic: none of the three
// clamps ever sees a -0 there, and reverting each select survived the P9.86c sweep.
//
// Here: ONE capsule at rest on the world axes, its pose carrying -0 in the z lanes
// (q = (0,0,-0,1), p = (-0.5,-0.5,-0)).  Basis vector 0 is then (1, +0, -0), so the
// segment runs from p = (+0,-0.5,-0) to q = (-1,-0.5,+0), pq = (-1,+0,+0).  Two cloth
// triangles lie in the plane y = 0, 0.5 above it (radius 0.375 + contact 0.25):
//   A  a=(0,0,-1) b=(0,0,-0) c=(-1,0,-1).  p projects onto vertex b; edges ab and bc
//      tie at b (0.25 each), bc wins the strict `>` chain, and the con0 test ties with
//      it (0.25 > 0.25 is false), so the EDGE result is kept.  On bc, c = +0, f = -0 and
//      b = 1, so temp = -0 -> s0 = -0 -> tTmp = -0 -> t2 = -0 -> comp = -0:
//      bary.z = t11 = -0 and normal.z = (+0) - (-0) = +0.  With s0 OR t2 through FClamp
//      both parameters come out +0: bary.z = +0, normal.z = (-0) - (+0) = -0.
//   B  a=(0,0,-0) b=(1,0,-0) c=(1,0,-1).  p projects onto vertex a; edges ab and ac
//      tie, ac wins.  On ac, comp = (-1)(+0) - (+0) = -0 = t02, so pointOnSegment.z =
//      -0 + (+0)(-0) = -0 and normal.z = (-0) - (-0) = +0.  With s through FClamp,
//      pointOnSegment.z = +0 and normal.z = (-0) - (+0) = -0.
// Every product on these paths is exact, so fma contraction (CUDA) or its absence
// (llvmpipe, the host) cannot move a zero's sign.  The mirror is pinned against the
// hand-derived CUDA bits as well, so it cannot drift to the port's old behaviour.
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimCapsuleTriClampZeroSign)
{
    using namespace p816dc;
    using namespace p929st;
    using p817cg::V3;
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t CLOTH_ID = 1u;
    const float CD_CLOTH = 0.125f, CD_CAPSULE = 0.125f, REST = 0.0625f;
    const uint32_t MAT_CAPSULE = 0x35u;
    const uint64_t NODE_CAPSULE = 0x0000000200000019ull;
    const float CAP_HALF = 0.5f, CAP_RADIUS = 0.375f;

    // The -0 lanes are ASSIGNED, and their sign is checked, so no initializer rewrite
    // can quietly turn the case into a +0 one.
    float capQ[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    capQ[2] = -0.0f;
    V3 capP{ -0.5f, -0.5f, 0.0f };
    capP.z = -0.0f;

    std::vector<F4> cv = {
        F4{ 0.0f, 0.0f, -1.0f, 1.0f }, F4{ 0.0f, 0.0f, 0.0f, 1.0f }, F4{ -1.0f, 0.0f, -1.0f, 1.0f },  // A
        F4{ 0.0f, 0.0f, 0.0f, 1.0f },  F4{ 1.0f, 0.0f, 0.0f, 1.0f }, F4{ 1.0f, 0.0f, -1.0f, 1.0f },   // B
    };
    cv[1].z = -0.0f;                          // A.b: makes r.z = (-0) - (-0) = +0, so f = -0
    cv[3].z = -0.0f; cv[4].z = -0.0f;         // B.a, B.b: pointOnTriangle.z = -0
    PXG_CHECK_MSG(std::signbit(capQ[2]) && std::signbit(capP.z) && std::signbit(cv[1].z) &&
                  std::signbit(cv[3].z) && std::signbit(cv[4].z),
                  "a -0 input lane lost its sign before the test ran");
    std::vector<TriIdx4> ci = { TriIdx4{ 0u, 1u, 2u, 0u }, TriIdx4{ 3u, 4u, 5u, 0u } };
    const uint32_t NB_TRIS = 2u;

    // ---- the mirror, the CUDA image, and the twins ------------------------------------
    struct Img { uint32_t f[12]; };
    auto image = [](const Hit& h) {
        Img m;
        const float src[12] = { h.point.x, h.point.y, h.point.z, h.point.w,
                                h.normalPen.x, h.normalPen.y, h.normalPen.z, h.normalPen.w,
                                h.bary.x, h.bary.y, h.bary.z, h.bary.w };
        for (int j = 0; j < 12; ++j) m.f[j] = f2u(src[j]);
        return m;
    };
    auto runMirror = [&](int variant, Img out[2], int exits[2]) {
        bool all = true;
        gSegSegClampVariant = variant;
        for (uint32_t t = 0; t < NB_TRIS; ++t)
        {
            const V3 a{ cv[t * 3 + 0].x, cv[t * 3 + 0].y, cv[t * 3 + 0].z };
            const V3 b{ cv[t * 3 + 1].x, cv[t * 3 + 1].y, cv[t * 3 + 1].z };
            const V3 c{ cv[t * 3 + 2].x, cv[t * 3 + 2].y, cv[t * 3 + 2].z };
            Hit h{};
            const bool hit = triCapsule(capQ, capP, CAP_HALF, CAP_RADIUS, a, b, c,
                                        CD_CLOTH + CD_CAPSULE, h);
            exits[t] = gSegTriExit;
            h.point.w = REST;
            out[t] = image(h);
            all = all && hit;
        }
        gSegSegClampVariant = 0;
        return all;
    };

    for (int k = 0; k < 3; ++k) gSegSegNegZeroIn[k] = 0u;
    Img want[2];
    int exits[2];
    const bool bothHit = runMirror(0, want, exits);
    uint32_t negZeroIn[3];                    // snapshot BEFORE any twin run adds to it
    for (int k = 0; k < 3; ++k) negZeroIn[k] = gSegSegNegZeroIn[k];

    PXG_CHECK_MSG(bothHit, "the mirror reports no contact for one of the two triangles");
    PXG_CHECK_MSG(exits[0] == EXIT_CON0_EDGE && exits[1] == EXIT_CON0_EDGE,
                  "segTriSq exits %d/%d, both cases are built for the con0 branch (%d)",
                  exits[0], exits[1], int(EXIT_CON0_EDGE));
    // Coverage: s0 = -0 on A.bc and B.ac, t2 = -0 on A.bc, s = -0 on A.bc, B.ab, B.ac.
    PXG_CHECK_MSG(negZeroIn[0] == 2u && negZeroIn[1] == 1u && negZeroIn[2] == 3u,
                  "-0 reached the clamps %u/%u/%u times (s0/t2/s), expected 2/1/3 -- the "
                  "rig no longer exercises the P9.86 select spellings",
                  negZeroIn[0], negZeroIn[1], negZeroIn[2]);

    // The CUDA's bits, derived by hand (see the banner).  The edge sub-exit (not the
    // con0 vertex one) is what puts t11 = -0 into A's bary.z, so this pins that tie too.
    static const uint32_t CUDA_IMG[2][12] = {
        { 0x00000000u, 0xBE000000u, 0x00000000u, 0x3D800000u,     // point, rest
          0x00000000u, 0x3F800000u, 0x00000000u, 0x3E000000u,     // normal, pen 0.125
          0x00000000u, 0x3F800000u, 0x80000000u, 0x00000000u },   // bary (w, u, v = -0)
        { 0x00000000u, 0xBE000000u, 0x00000000u, 0x3D800000u,
          0x00000000u, 0x3F800000u, 0x00000000u, 0x3E000000u,
          0x3F800000u, 0x00000000u, 0x00000000u, 0x00000000u },
    };
    static const char* const FIELD[12] = {
        "point.x", "point.y", "point.z", "point.w(rest)", "normal.x", "normal.y", "normal.z",
        "penetration", "bary.x", "bary.y", "bary.z", "bary.w" };
    for (uint32_t t = 0; t < NB_TRIS; ++t)
        for (int j = 0; j < 12; ++j)
            PXG_CHECK_MSG(want[t].f[j] == CUDA_IMG[t][j],
                          "mirror tri %c %s = 0x%08X, the CUDA gives 0x%08X", 'A' + char(t),
                          FIELD[j], want[t].f[j], CUDA_IMG[t][j]);

    // Each twin must change exactly the fields its revert changes, and nothing else.
    //   v1 (s0) and v2 (t2): A's normal.z -> 0x80000000 and bary.z -> 0x00000000; B intact.
    //   v3 (s):              B's normal.z -> 0x80000000; A intact (on A.bc it moves t01
    //                        alone, and p + pq*t01 absorbs it: normal.z stays +0).
    for (int v = 1; v <= 3; ++v)
    {
        Img twin[2];
        int twinExits[2];
        runMirror(v, twin, twinExits);
        for (uint32_t t = 0; t < NB_TRIS; ++t)
        {
            Img expect = want[t];
            if (v <= 2 && t == 0u) { expect.f[6] = 0x80000000u; expect.f[10] = 0x00000000u; }
            if (v == 3 && t == 1u) { expect.f[6] = 0x80000000u; }
            for (int j = 0; j < 12; ++j)
                PXG_CHECK_MSG(twin[t].f[j] == expect.f[j],
                              "mutant twin %d, tri %c %s = 0x%08X, expected 0x%08X", v,
                              'A' + char(t), FIELD[j], twin[t].f[j], expect.f[j]);
        }
    }

    // ---- the kernel ---------------------------------------------------------------------
    DevBuf bCv = makeDev(dev, cv);
    if (!bCv.addr) { std::printf("  NOTE: no BDA -> skip\n"); dev->destroyBuffer(bCv.h); return; }
    DevBuf bCi = makeDev(dev, ci);

    PxgFEMCloth cloth = makeZeroDesc();
    cloth.mPosition_InvMass = reinterpret_cast<decltype(cloth.mPosition_InvMass)>(size_t(bCv.addr));
    cloth.mTriangleVertexIndices = reinterpret_cast<decltype(cloth.mTriangleVertexIndices)>(size_t(bCi.addr));
    cloth.mNbTriangles = NB_TRIS;
    cloth.mNbVerts = uint32_t(cv.size());
    std::vector<PxgFEMCloth> descArr(CLOTH_ID + 1u, makeZeroDesc());
    descArr[CLOTH_ID] = cloth;
    DevBuf bDesc = makeDev(dev, descArr);

    // shapes: 0 = cloth, 1 = capsule
    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0] = makeClothShape(CLOTH_ID);
    shapes[1].type = GEOM_CAPSULE_H; shapes[1].particleOrSoftbodyId = 0xFFFFFFFFu;
    shapes[1].sx = CAP_HALF; shapes[1].sy = CAP_RADIUS; shapes[1].sz = 0.13f;
    shapes[1].srw = 1.0f; shapes[1].materialIndex = MAT_CAPSULE;
    DevBuf bShapes = makeDev(dev, shapes);

    std::vector<CmInputHost> cms = { { 0u, 1u, 0u, 1u } };
    DevBuf bCms = makeDev(dev, cms);

    std::vector<float> xf(2 * 8, 0.0f);
    xf[3] = 1.0f;                                     // cloth pose: identity, at the origin
    for (int k = 0; k < 4; ++k) xf[8 + k] = capQ[k];
    xf[12] = capP.x; xf[13] = capP.y; xf[14] = capP.z;
    DevBuf bXf = makeDev(dev, xf);

    std::vector<float> cd = { CD_CLOTH, CD_CAPSULE };
    std::vector<float> rd = { REST };
    DevBuf bCd = makeDev(dev, cd), bRd = makeDev(dev, rd);
    std::vector<uint64_t> remap = { 0ull, NODE_CAPSULE };
    DevBuf bRemap = makeDev(dev, remap);

    std::vector<uint32_t> stack = { 0u, 0u, 0u, 0u, 0u, 1u, 0u, 0u };   // (cm 0, tri 0/1)
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> nFound = { NB_TRIS };
    DevBuf bNum = makeDev(dev, nFound);
    std::vector<uint32_t> zeroNeeded = { 0u };
    DevBuf bNeeded = makeDev(dev, zeroNeeded);

    const uint32_t MAXC = 8u;
    WriterBufs w = makeWriter(dev, MAXC);

    ClothSphereTriPush p = {};
    p.toleranceLength = 1.0f;
    p.cmInputs = bCms.addr; p.transformCache = bXf.addr;
    p.contactDistance = bCd.addr; p.restDistance = bRd.addr;
    p.gpuShapes = bShapes.addr; p.femClothes = bDesc.addr;
    p.shapeToRigidRemapTable = bRemap.addr;
    p.stackSizeBytes = NB_TRIS * 16u;
    p.stackPtr = bStack.addr; p.midphasePairsNum = bNum.addr;
    p.filterPairs = 0ull; p.nbFilterPairs = 0u;
    p.stackSizeNeededOnDevice = bNeeded.addr;
    p.wPoint = w.point.addr; p.wNormalPen = w.normalPen.addr;
    p.wBary = w.barycentric.addr; p.wInfo = w.info.addr; p.wCount = w.count.addr;
    p.wByX = w.byRigid.addr; p.wTempByX = w.tempByRigid.addr;
    p.wIdxByX = w.idxByRigid.addr; p.wSorted32 = w.sortedByRigid.addr;
    p.wMaxContacts = MAXC;

    Kernel k("femClothContactGenBda.spv", "cloth_SphereContactGenLaunch",
             {}, sizeof(ClothSphereTriPush));
    k.dispatch(1024u, {}, p, 1u, 1u);      // PxgNarrowphaseCore.cpp:5760
    dev->flush();

    const uint32_t got = downloadBuffer<uint32_t>(dev, w.count.h, 1)[0];
    PXG_CHECK_MSG(got == NB_TRIS, "contact COUNT %u, expected one per triangle (%u)", got, NB_TRIS);

    const auto pt = downloadBuffer<F4>(dev, w.point.h, MAXC);
    const auto np = downloadBuffer<F4>(dev, w.normalPen.h, MAXC);
    const auto ba = downloadBuffer<F4>(dev, w.barycentric.h, MAXC);
    const auto info = downloadBuffer<uint32_t>(dev, w.info.h, MAXC * 4u);
    uint32_t seen[2] = { 0u, 0u };
    for (uint32_t i = 0; i < got && i < MAXC; ++i)
    {
        const uint32_t pairInd1 = info[i * 4 + 2];
        const uint32_t tri = pairInd1 & ((1u << 20) - 1u);
        PXG_CHECK_MSG(tri < NB_TRIS && (pairInd1 >> 20) == CLOTH_ID,
                      "contact %u names pairInd1 0x%08X", i, pairInd1);
        if (tri >= NB_TRIS) continue;
        ++seen[tri];
        PXG_CHECK_MSG(info[i * 4 + 0] == uint32_t(NODE_CAPSULE & 0xffffffffull) &&
                      info[i * 4 + 1] == uint32_t(NODE_CAPSULE >> 32) &&
                      info[i * 4 + 3] == (MAT_CAPSULE & 0x7fffffffu),
                      "tri %c info record %08X %08X / mat %08X", 'A' + char(tri),
                      info[i * 4 + 1], info[i * 4 + 0], info[i * 4 + 3]);
        const Hit gh{ pt[i], np[i], ba[i] };
        const Img g = image(gh);
        for (int j = 0; j < 12; ++j)
            PXG_CHECK_MSG(g.f[j] == want[tri].f[j],
                          "GPU tri %c %s = 0x%08X, the CUDA (mirror) gives 0x%08X",
                          'A' + char(tri), FIELD[j], g.f[j], want[tri].f[j]);
    }
    PXG_CHECK_MSG(seen[0] == 1u && seen[1] == 1u,
                  "contacts per triangle %u/%u, expected exactly one each", seen[0], seen[1]);

    destroyWriter(dev, w);
    for (PxgBufferHandle h : { bCv.h, bCi.h, bDesc.h, bShapes.h, bCms.h, bXf.h, bCd.h,
                               bRd.h, bRemap.h, bStack.h, bNum.h, bNeeded.h })
        dev->destroyBuffer(h);
}


// ===========================================================================
// P9.30 -- the cloth-vs-PARTICLE chain, both halves.
//
// The midphase is checked as a SET against a host mirror of the traversal's
// PREDICATE rather than of the traversal: with a single-leaf tree the walk
// visits every triangle, so what the kernel decides is exactly
// "triangle bound intersects the particle's cube", and a mirror of the walk
// itself would be the duplication the sweep exists to avoid.  The contact gen
// is checked BIT-EXACTLY against a full transcription, because there the
// arithmetic is the thing.
// ===========================================================================
namespace p930ps
{
using p817cg::V3;
using p817cg::sub;
using p817cg::add;
using p817cg::dot;
using p817cg::scl;

// cloth_psMidphaseGeneratePairsLaunch: twelve arguments -> 88 bytes.
struct ClothPsMidphasePush
{
    uint32_t numWorkItems; float toleranceLength;
    uint64_t cmInputs, gpuShapes, transformCache, bounds, contactDistance, femClothes,
             particleSystems;
    uint32_t stackSizeBytes, pad68;
    uint64_t stackPtr, midphasePairsNum;
};
static_assert(sizeof(ClothPsMidphasePush) == 88, "cloth ps midphase blob must be 88 bytes");
static_assert(offsetof(ClothPsMidphasePush, gpuShapes) == 16, "gpuShapes @16 -- THIRD, not after bounds");
static_assert(offsetof(ClothPsMidphasePush, particleSystems) == 56, "particleSystems @56");
static_assert(offsetof(ClothPsMidphasePush, stackSizeBytes) == 64, "stackSizeBytes @64");
static_assert(offsetof(ClothPsMidphasePush, stackPtr) == 72, "stackPtr @72");
static_assert(offsetof(ClothPsMidphasePush, midphasePairsNum) == 80, "midphasePairsNum @80");

// cloth_psContactGenLaunch: thirteen arguments, writer by value -> 176 bytes.
// The cloth-vs-heightfield contact gen's blob is ALSO 176 with its writer ALSO
// at 96 and a different middle, so every offset here is asserted.
struct ClothPsCgPush
{
    float toleranceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, restDistance, gpuShapes,
             particleSystems, femClothes;
    uint32_t stackSizeBytes, pad68;
    uint64_t stackPtr, midphasePairsNum, stackSizeNeeded;
    uint64_t wPoint, wNormalPen, wBary, wInfo, wCount, wByX, wTempByX, wIdxByX, wSorted32;
    uint32_t wMaxContacts, pad172;
};
static_assert(sizeof(ClothPsCgPush) == 176, "cloth ps CG blob must be 176 bytes");
static_assert(offsetof(ClothPsCgPush, particleSystems) == 48, "particleSystems @48");
static_assert(offsetof(ClothPsCgPush, femClothes) == 56, "femClothes @56 -- AFTER particleSystems");
static_assert(offsetof(ClothPsCgPush, stackSizeBytes) == 64, "stackSizeBytes @64");
static_assert(offsetof(ClothPsCgPush, stackPtr) == 72, "stackPtr @72");
static_assert(offsetof(ClothPsCgPush, midphasePairsNum) == 80, "midphasePairsNum @80");
static_assert(offsetof(ClothPsCgPush, stackSizeNeeded) == 88, "stackSizeNeeded @88");
static_assert(offsetof(ClothPsCgPush, wPoint) == 96, "the writer starts @96 -- paramAlign(80) is 16");
static_assert(offsetof(ClothPsCgPush, wMaxContacts) == 168, "maxNumContacts @168");

static const uint32_t GEOM_PARTICLESYSTEM_H = uint32_t(physx::PxGeometryType::ePARTICLESYSTEM);   // PxGeometryType::ePARTICLESYSTEM

// ---- the contact gen's host mirror ----------------------------------------
// Gu::closestPtPointTriangle2 (GuDistancePointTriangle.h:44), the ab/ac form.
static V3 cpTri2(V3 p, V3 a, V3 b, V3 c, V3 ab, V3 ac)
{
    const V3 ap = sub(p, a);
    const float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    const V3 bp = sub(p, b);
    const float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) { const float v = d1 / (d1 - d3); return add(a, scl(ab, v)); }
    const V3 cp = sub(p, c);
    const float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) { const float w = d2 / (d2 - d6); return add(a, scl(ac, w)); }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    { const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return add(b, scl(sub(c, b), w)); }
    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom, w = vc * denom;
    return add(add(a, scl(ab, v)), scl(ac, w));   // left to right, like the shader
}

// computeBarycentric (femClothPrimitives.cu:114); the bool is not mirrored (C7).
static F4 computeBary(V3 a, V3 b, V3 c, V3 p)
{
    const V3 v0 = sub(b, a), v1 = sub(c, a), v2 = sub(p, a);
    const float d00 = dot(v0, v0), d01 = dot(v0, v1), d11 = dot(v1, v1);
    const float d20 = dot(v2, v0), d21 = dot(v2, v1);
    const float denom = d00 * d11 - d01 * d01;
    const float v = (d11 * d20 - d01 * d21) / denom;
    const float w = (d00 * d21 - d01 * d20) / denom;
    return F4{ 1.0f - v - w, v, w, 0.0f };
}
} // namespace p930ps

PXG_TEST(femClothSimPsMidphase)
{
    using namespace p816dc;
    using namespace p930ps;
    using p817cg::V3;
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t CLOTH_ID = 1u, PS_ID = 2u;
    const float CD_PARTICLE = 0.30f, CD_CLOTH = 0.10f;   // the SUM is what the query uses
    const float R = CD_PARTICLE + CD_CLOTH;

    // ---- a cloth of 12 triangles laid along x, each a small fixed-size patch -------
    const uint32_t NB_TRIS = 12u;
    std::vector<F4> cv; std::vector<TriIdx4> ci;
    for (uint32_t t = 0; t < NB_TRIS; ++t)
    {
        const float x = float(t) * 1.0f;
        const uint32_t base = uint32_t(cv.size());
        cv.push_back(F4{ x - 0.2f, 0.0f, -0.2f, 1.0f });
        cv.push_back(F4{ x + 0.2f, 0.0f, -0.2f, 1.0f });
        cv.push_back(F4{ x + 0.0f, 0.0f,  0.3f, 1.0f });
        ci.push_back(TriIdx4{ base, base + 1u, base + 2u, 0u });
    }
    DevBuf bCv = makeDev(dev, cv);
    if (!bCv.addr) { std::printf("  NOTE: no BDA -> skip\n"); dev->destroyBuffer(bCv.h); return; }
    DevBuf bCi = makeDev(dev, ci);

    // A single-leaf tree over all 12 triangles: the walk then reaches every one, so
    // what the kernel decides IS the per-triangle predicate the mirror computes.
    std::vector<uint32_t> tree = makeBv32TreeWords(NB_TRIS, -1000.0f, 1000.0f);
    DevBuf bTree = makeDev(dev, tree);

    PxgFEMCloth cloth = makeZeroDesc();
    cloth.mTriMeshData = reinterpret_cast<decltype(cloth.mTriMeshData)>(size_t(bTree.addr));
    cloth.mPosition_InvMass = reinterpret_cast<decltype(cloth.mPosition_InvMass)>(size_t(bCv.addr));
    cloth.mTriangleVertexIndices = reinterpret_cast<decltype(cloth.mTriangleVertexIndices)>(size_t(bCi.addr));
    cloth.mNbTriangles = NB_TRIS;
    cloth.mNbVerts = uint32_t(cv.size());
    std::vector<PxgFEMCloth> descArr(CLOTH_ID + 1u, makeZeroDesc());
    descArr[CLOTH_ID] = cloth;
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- particles: some straddling triangles, some clear of every one -------------
    std::vector<F4> parts = {
        { 0.00f, 0.20f, 0.00f, 1.0f },   // over triangle 0, inside R
        { 1.00f, 0.80f, 0.00f, 1.0f },   // over triangle 1, OUTSIDE R in y
        { 2.45f, 0.10f, 0.00f, 1.0f },   // between triangles 2 and 3: reaches BOTH
        { 5.00f, 0.00f, 0.00f, 1.0f },   // exactly on triangle 5's plane
        { 50.0f, 50.0f, 50.0f, 1.0f },   // nowhere near anything
        { 11.0f, -0.35f, 0.0f, 1.0f },   // under the LAST triangle, inside R
    };
    const uint32_t NB_PARTICLES = uint32_t(parts.size());
    DevBuf bParts = makeDev(dev, parts);

    std::vector<PxgParticleSystem> psArr(PS_ID + 1u);
    std::memset(psArr.data(), 0, psArr.size() * sizeof(PxgParticleSystem));
    psArr[PS_ID].mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bParts.addr));
    psArr[PS_ID].mCommonData.mNumParticles = NB_PARTICLES;
    DevBuf bPs = makeDev(dev, psArr);

    // ---- shapes.  TWO managers, and the particle system sits in the OTHER slot in
    // the second one, so M6's routing by TYPE is separable from slot order.
    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_PARTICLESYSTEM_H; shapes[0].particleOrSoftbodyId = PS_ID;
    shapes[0].sx = shapes[0].sy = shapes[0].sz = 1.0f; shapes[0].srw = 1.0f;
    shapes[1] = makeClothShape(CLOTH_ID);
    DevBuf bShapes = makeDev(dev, shapes);

    // cache ref 0 = particle system (CD 0.30), 1 = cloth (CD 0.10).
    std::vector<CmInputHost> cms = { { 0u, 1u, 0u, 1u }, { 1u, 0u, 1u, 0u } };
    DevBuf bCms = makeDev(dev, cms);
    std::vector<float> cd = { CD_PARTICLE, CD_CLOTH };
    DevBuf bCd = makeDev(dev, cd);
    std::vector<float> xf(2 * 8, 0.0f); xf[3] = 1.0f; xf[8 + 3] = 1.0f;
    DevBuf bXf = makeDev(dev, xf);
    std::vector<float> dummyBounds(2 * 6, 0.0f);
    DevBuf bBounds = makeDev(dev, dummyBounds);

    // The stack deliberately holds FEWER pairs than the query will find, so the
    // "drop the pair, keep counting" behaviour is measured rather than assumed.
    const uint32_t NB_ALLOC = 128u;
    const uint32_t STACK_PAIRS = 6u;
    std::vector<uint32_t> stack(NB_ALLOC * 4u, 0xEEEEEEEEu);
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> zero = { 0u };
    DevBuf bNum = makeDev(dev, zero);

    ClothPsMidphasePush p = {};
    p.numWorkItems = uint32_t(cms.size());
    p.toleranceLength = 1.0f;
    p.cmInputs = bCms.addr; p.gpuShapes = bShapes.addr; p.transformCache = bXf.addr;
    p.bounds = bBounds.addr; p.contactDistance = bCd.addr;
    p.femClothes = bDesc.addr; p.particleSystems = bPs.addr;
    p.stackSizeBytes = STACK_PAIRS * 16u;
    p.stackPtr = bStack.addr; p.midphasePairsNum = bNum.addr;

    Kernel k("femClothPsMidphaseBda.spv", "cloth_psMidphaseGeneratePairsLaunch",
             {}, sizeof(ClothPsMidphasePush));
    k.dispatch(1024u, {}, p, uint32_t(cms.size()), 1u);   // grid (1024, numTests, 1)
    dev->flush();

    // ---- the expected set ----------------------------------------------------------
    struct Pair { uint32_t cm, particle, tri; };
    std::vector<Pair> expected;
    for (uint32_t cm = 0; cm < uint32_t(cms.size()); ++cm)
        for (uint32_t pi = 0; pi < NB_PARTICLES; ++pi)
        {
            const V3 c{ parts[pi].x, parts[pi].y, parts[pi].z };
            for (uint32_t t = 0; t < NB_TRIS; ++t)
            {
                const F4& a = cv[t * 3 + 0]; const F4& b = cv[t * 3 + 1]; const F4& d = cv[t * 3 + 2];
                const V3 tMin{ std::min(std::min(a.x, b.x), d.x), std::min(std::min(a.y, b.y), d.y),
                               std::min(std::min(a.z, b.z), d.z) };
                const V3 tMax{ std::max(std::max(a.x, b.x), d.x), std::max(std::max(a.y, b.y), d.y),
                               std::max(std::max(a.z, b.z), d.z) };
                const bool hit = !(tMin.x > c.x + R || c.x - R > tMax.x ||
                                   tMin.y > c.y + R || c.y - R > tMax.y ||
                                   tMin.z > c.z + R || c.z - R > tMax.z);
                if (hit) expected.push_back(Pair{ cm, pi, t });
            }
        }

    const uint32_t found = downloadBuffer<uint32_t>(dev, bNum.h, 1)[0];
    PXG_CHECK_MSG(expected.size() > STACK_PAIRS,
                  "the rig found %zu pairs against a %u-pair stack; it must OVERFLOW or the "
                  "drop-but-keep-counting behaviour is not measured", expected.size(), STACK_PAIRS);
    PXG_CHECK_MSG(found == uint32_t(expected.size()),
                  "the midphase COUNTER reads %u, the mirror says %zu.  The counter takes every "
                  "hit even when the stack cannot hold it", found, expected.size());

    // Only the first STACK_PAIRS slots may have been written, and each must be a
    // member of the expected set with the fields in the right places (M1).
    const auto st = downloadBuffer<uint32_t>(dev, bStack.h, NB_ALLOC * 4u);
    uint32_t written = 0u, bad = 0u;
    for (uint32_t i = 0; i < NB_ALLOC; ++i)
    {
        if (st[i * 4] == 0xEEEEEEEEu) continue;
        ++written;
        if (i >= STACK_PAIRS) { ++bad; continue; }
        const uint32_t cm = st[i * 4 + 0], pi = st[i * 4 + 1], tri = st[i * 4 + 2];
        bool member = false;
        for (const Pair& e : expected)
            if (e.cm == cm && e.particle == pi && e.tri == tri) { member = true; break; }
        if (!member)
        {
            ++bad;
            std::printf("  slot %u holds (cm %u, particle %u, tri %u), which is not an expected "
                        "pair -- M1 puts the PARTICLE in .y and the triangle in .z\n", i, cm, pi, tri);
        }
    }
    PXG_CHECK_MSG(bad == 0u, "%u written pairs are wrong or past the stack bound", bad);
    PXG_CHECK_MSG(written == STACK_PAIRS,
                  "%u slots written, expected exactly the %u the stack holds", written, STACK_PAIRS);

    std::printf("    [ note ] %zu pairs found over %u particles x %u triangles x %zu managers; "
                "%u written into a %u-pair stack, counter %u\n",
                expected.size(), NB_PARTICLES, NB_TRIS, cms.size(), written, STACK_PAIRS, found);

    // ---- the numWorkItems GUARD --------------------------------------------------
    // `if (cmIdx >= numWorkItems) return;` never fires when the dispatch has exactly
    // numWorkItems y-groups, which is what the engine launches and what the run above
    // does -- so a port that dropped the guard passed, and the sweep said so.  Here
    // the dispatch is deliberately WIDER than the work: the extra y-groups must
    // contribute nothing, and without the guard they read a contact manager that is
    // not there.
    {
        std::vector<uint32_t> z3 = { 0u };
        DevBuf bWideNum = makeDev(dev, z3);
        std::vector<uint32_t> wideStack(NB_ALLOC * 4u, 0xEEEEEEEEu);
        DevBuf bWideStack = makeDev(dev, wideStack);
        ClothPsMidphasePush q = p;
        q.numWorkItems = 1u;                       // only manager 0 is real work
        q.stackPtr = bWideStack.addr; q.midphasePairsNum = bWideNum.addr;
        q.stackSizeBytes = NB_ALLOC * 16u;
        k.dispatch(1024u, {}, q, 4u, 1u);          // FOUR y-groups for ONE work item
        dev->flush();

        uint32_t oneMgr = 0u;
        for (const Pair& e : expected) if (e.cm == 0u) ++oneMgr;
        const uint32_t nWide = downloadBuffer<uint32_t>(dev, bWideNum.h, 1)[0];
        PXG_CHECK_MSG(nWide == oneMgr,
                      "a dispatch of 4 y-groups against numWorkItems=1 found %u pairs, expected "
                      "the %u of manager 0 alone.  The three extra groups must return at the "
                      "guard; without it they index a contact manager past the end of the array",
                      nWide, oneMgr);
        std::printf("    [ note ] guard run: 4 y-groups, numWorkItems 1 -> %u pairs (manager 0 "
                    "only)\n", nWide);
        for (PxgBufferHandle h : { bWideNum.h, bWideStack.h }) dev->destroyBuffer(h);
    }

    // ---- M8, the WARP STRIDE, which the run above cannot reach --------------------
    // The loop strides by gridDim.x * blockDim.y = 1024 * 2 warps.  With six
    // particles every warp runs at most one iteration and the stride is never taken,
    // so a mutant that replaces it with gridDim.x alone survives -- a survivor the
    // rig cannot reach is a hole in the rig, which is what P9.29 measured the hard
    // way.  So: more particles than the grid has warps, with the only contacting
    // ones in the tail that a second iteration reaches.
    {
        const uint32_t NB_WARPS = 1024u * 2u;          // the shader's PS_NB_WARPS
        const uint32_t TAIL = 8u;
        const uint32_t BIG = NB_WARPS + TAIL;
        std::vector<F4> big(BIG, F4{ 500.0f, 500.0f, 500.0f, 1.0f });   // far from everything
        for (uint32_t i = NB_WARPS; i < BIG; ++i)
            big[i] = F4{ 0.0f, 0.15f, 0.0f, 1.0f };    // over triangle 0, inside R
        DevBuf bBigPart = makeDev(dev, big);
        std::vector<PxgParticleSystem> psBig(PS_ID + 1u);
        std::memset(psBig.data(), 0, psBig.size() * sizeof(PxgParticleSystem));
        psBig[PS_ID].mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bBigPart.addr));
        psBig[PS_ID].mCommonData.mNumParticles = BIG;
        DevBuf bPsBig = makeDev(dev, psBig);
        std::vector<uint32_t> bigStack(BIG * 4u, 0xEEEEEEEEu);
        DevBuf bBigStack = makeDev(dev, bigStack);
        std::vector<uint32_t> z2 = { 0u };
        DevBuf bBigNum = makeDev(dev, z2);

        ClothPsMidphasePush q = p;
        q.numWorkItems = 1u;                            // one manager keeps the count clean
        q.particleSystems = bPsBig.addr;
        q.stackPtr = bBigStack.addr; q.midphasePairsNum = bBigNum.addr;
        q.stackSizeBytes = BIG * 16u;
        k.dispatch(1024u, {}, q, 1u, 1u);
        dev->flush();

        const uint32_t n2 = downloadBuffer<uint32_t>(dev, bBigNum.h, 1)[0];
        PXG_CHECK_MSG(n2 == TAIL,
                      "the stride run found %u pairs, expected exactly %u.  %u particles were "
                      "given to a grid of %u warps and only the last %u touch a triangle, so "
                      "only a second iteration at the right stride finds them -- more means "
                      "particles are being visited by several warps, fewer means the second "
                      "iteration never happened", n2, TAIL, BIG, NB_WARPS, TAIL);
        std::printf("    [ note ] stride run: %u particles over %u warps -> %u pairs from the "
                    "%u-particle tail\n", BIG, NB_WARPS, n2, TAIL);

        for (PxgBufferHandle h : { bBigPart.h, bPsBig.h, bBigStack.h, bBigNum.h })
            dev->destroyBuffer(h);
    }

    for (PxgBufferHandle h : { bCv.h, bCi.h, bTree.h, bDesc.h, bParts.h, bPs.h, bShapes.h,
                               bCms.h, bCd.h, bXf.h, bBounds.h, bStack.h, bNum.h })
        dev->destroyBuffer(h);
}


PXG_TEST(femClothSimPsContactGen)
{
    using namespace p816dc;
    using namespace p930ps;
    using p817cg::V3;
    using p817cg::sub; using p817cg::add; using p817cg::dot; using p817cg::scl;
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t CLOTH_ID = 1u, PS_ID = 2u;
    // DIFFERENT contact distances per cache ref, so the SUM (C5) is separable from
    // either one, and a rest distance that is neither (C4).
    const float CD_PARTICLE = 0.30f, CD_CLOTH = 0.10f;
    const float REST_0 = 0.07f, REST_1 = 0.19f;      // per contact manager (C4)

    // ---- one cloth triangle per pair, tilted so no axis is special ----------------
    const uint32_t NB_TRIS = 8u;
    std::vector<F4> cv; std::vector<TriIdx4> ci;
    for (uint32_t t = 0; t < NB_TRIS; ++t)
    {
        const float x = float(t) * 2.0f;
        const uint32_t base = uint32_t(cv.size());
        cv.push_back(F4{ x - 0.5f, 0.05f * float(t), -0.4f, 1.0f });
        cv.push_back(F4{ x + 0.6f, -0.03f * float(t), -0.2f, 1.0f });
        cv.push_back(F4{ x + 0.1f, 0.02f * float(t),  0.5f, 1.0f });
        ci.push_back(TriIdx4{ base, base + 1u, base + 2u, 0u });
    }
    DevBuf bCv = makeDev(dev, cv);
    if (!bCv.addr) { std::printf("  NOTE: no BDA -> skip\n"); dev->destroyBuffer(bCv.h); return; }
    DevBuf bCi = makeDev(dev, ci);

    PxgFEMCloth cloth = makeZeroDesc();
    cloth.mPosition_InvMass = reinterpret_cast<decltype(cloth.mPosition_InvMass)>(size_t(bCv.addr));
    cloth.mTriangleVertexIndices = reinterpret_cast<decltype(cloth.mTriangleVertexIndices)>(size_t(bCi.addr));
    cloth.mNbTriangles = NB_TRIS; cloth.mNbVerts = uint32_t(cv.size());
    std::vector<PxgFEMCloth> descArr(CLOTH_ID + 1u, makeZeroDesc());
    descArr[CLOTH_ID] = cloth;
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- particles: chosen to reach several exits of closestPtPointTriangle2 ------
    // face interior, each of the three vertex regions, and two edge regions -- the
    // returned point differs structurally between them, and a transcription slip in
    // one branch is invisible from the others.
    std::vector<F4> parts;
    auto triCentroid = [&](uint32_t t) {
        const F4& a = cv[t * 3 + 0]; const F4& b = cv[t * 3 + 1]; const F4& c = cv[t * 3 + 2];
        return V3{ (a.x + b.x + c.x) / 3.0f, (a.y + b.y + c.y) / 3.0f, (a.z + b.z + c.z) / 3.0f };
    };
    auto vtx = [&](uint32_t t, uint32_t k) {
        const F4& v = cv[t * 3 + k]; return V3{ v.x, v.y, v.z };
    };
    // PARTICLE p IS PAIRED WITH TRIANGLE p+1, never with triangle p.  The first
    // version of this rig paired p with p, which made C1's `.y`/`.z` swap a NO-OP --
    // the sweep said so, by reporting `c1-pair-swap` as a survivor against the very
    // trap the kernel's headline note is about.  With the two index spaces offset,
    // a swap points at a different triangle and the contact set changes.
    parts.push_back(F4{ triCentroid(1).x, triCentroid(1).y + 0.20f, triCentroid(1).z, 1.0f }); // face
    parts.push_back(F4{ vtx(2, 0).x - 0.15f, vtx(2, 0).y + 0.10f, vtx(2, 0).z - 0.15f, 1.0f }); // vertex a
    parts.push_back(F4{ vtx(3, 1).x + 0.18f, vtx(3, 1).y + 0.08f, vtx(3, 1).z - 0.12f, 1.0f }); // vertex b
    parts.push_back(F4{ vtx(4, 2).x + 0.05f, vtx(4, 2).y + 0.09f, vtx(4, 2).z + 0.20f, 1.0f }); // vertex c
    {   // edge ab of triangle 5
        const V3 a = vtx(5, 0), b = vtx(5, 1);
        const V3 mid{ (a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f, (a.z + b.z) * 0.5f };
        parts.push_back(F4{ mid.x, mid.y + 0.05f, mid.z - 0.22f, 1.0f });
    }
    {   // edge ac of triangle 6
        const V3 a = vtx(6, 0), c = vtx(6, 2);
        const V3 mid{ (a.x + c.x) * 0.5f, (a.y + c.y) * 0.5f, (a.z + c.z) * 0.5f };
        parts.push_back(F4{ mid.x - 0.20f, mid.y + 0.05f, mid.z, 1.0f });
    }
    parts.push_back(F4{ triCentroid(7).x + 9.0f, 9.0f, 9.0f, 1.0f });   // far: no contact
    // THE BAND, for C5.  cDistance is 0.40 and cDistance^2 is 0.16, so a particle
    // 0.50 above its triangle has sqDist 0.25: rejected by `sqDist <= cDistance^2`
    // and ACCEPTED by a port that forgot to square.  Without it that mutant lives --
    // measured, as a survivor.
    parts.push_back(F4{ triCentroid(0).x, triCentroid(0).y + 0.50f, triCentroid(0).z, 1.0f });
    const uint32_t NB_PARTICLES = uint32_t(parts.size());
    auto triFor = [&](uint32_t particle) { return (particle + 1u) % NB_TRIS; };
    DevBuf bParts = makeDev(dev, parts);

    std::vector<PxgParticleSystem> psArr(PS_ID + 1u);
    std::memset(psArr.data(), 0, psArr.size() * sizeof(PxgParticleSystem));
    psArr[PS_ID].mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bParts.addr));
    psArr[PS_ID].mCommonData.mNumParticles = NB_PARTICLES;
    DevBuf bPs = makeDev(dev, psArr);

    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_PARTICLESYSTEM_H; shapes[0].particleOrSoftbodyId = PS_ID;
    shapes[0].sx = shapes[0].sy = shapes[0].sz = 1.0f; shapes[0].srw = 1.0f;
    shapes[1] = makeClothShape(CLOTH_ID);
    DevBuf bShapes = makeDev(dev, shapes);
    // Manager 1 puts the particle system in the OTHER slot: routing is by TYPE.
    std::vector<CmInputHost> cms = { { 0u, 1u, 0u, 1u }, { 1u, 0u, 1u, 0u } };
    DevBuf bCms = makeDev(dev, cms);
    std::vector<float> cd = { CD_PARTICLE, CD_CLOTH };
    std::vector<float> rd = { REST_0, REST_1 };
    DevBuf bCd = makeDev(dev, cd), bRd = makeDev(dev, rd);
    std::vector<float> xf(2 * 8, 0.0f); xf[3] = 1.0f; xf[8 + 3] = 1.0f;
    DevBuf bXf = makeDev(dev, xf);

    // ---- the pair stack.  The midphase CLAIMS more than the stack holds (C8). -----
    const uint32_t NB_STACK = NB_PARTICLES;
    const uint32_t NB_FOUND = NB_PARTICLES + 13u;
    std::vector<uint32_t> stack(NB_FOUND * 4u, 0u);
    for (uint32_t i = 0; i < NB_FOUND; ++i)
    {
        stack[i * 4 + 0] = (i & 1u);                               // alternate managers
        stack[i * 4 + 1] = (i < NB_PARTICLES) ? i : 0u;                  // C1: particle in .y
        stack[i * 4 + 2] = (i < NB_PARTICLES) ? triFor(i) : triFor(0u);  //     triangle in .z
    }
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> nFound = { NB_FOUND };
    DevBuf bNum = makeDev(dev, nFound);
    std::vector<uint32_t> zero = { 0u };
    DevBuf bNeeded = makeDev(dev, zero);

    const uint32_t MAXC = 32u;
    WriterBufs w = makeWriter(dev, MAXC);

    ClothPsCgPush p = {};
    p.toleranceLength = 1.0f;
    p.cmInputs = bCms.addr; p.transformCache = bXf.addr;
    p.contactDistance = bCd.addr; p.restDistance = bRd.addr;
    p.gpuShapes = bShapes.addr; p.particleSystems = bPs.addr; p.femClothes = bDesc.addr;
    p.stackSizeBytes = NB_STACK * 16u;
    p.stackPtr = bStack.addr; p.midphasePairsNum = bNum.addr;
    p.stackSizeNeeded = bNeeded.addr;
    p.wPoint = w.point.addr; p.wNormalPen = w.normalPen.addr; p.wBary = w.barycentric.addr;
    p.wInfo = w.info.addr; p.wCount = w.count.addr;
    p.wByX = w.byRigid.addr; p.wTempByX = w.tempByRigid.addr; p.wIdxByX = w.idxByRigid.addr;
    p.wSorted32 = 0ull;                 // C2: NULL for the particle writer, and never written
    p.wMaxContacts = MAXC;

    Kernel k("femClothPsContactGenBda.spv", "cloth_psContactGenLaunch",
             {}, sizeof(ClothPsCgPush));
    k.dispatch(512u, {}, p, 1u, 1u);
    dev->flush();

    // ---- the mirror ----------------------------------------------------------------
    struct Exp { F4 point, normalPen, bary; uint64_t pairInd0; uint32_t pairInd1; };
    std::vector<Exp> expected;
    uint32_t faceHits = 0u;
    for (uint32_t i = 0; i < NB_STACK; ++i)          // C8: the CLAMPED walk
    {
        const uint32_t cm = stack[i * 4 + 0], pi = stack[i * 4 + 1], tri = stack[i * 4 + 2];
        const float cDistance = CD_PARTICLE + CD_CLOTH;
        const float rest = (cm == 0u) ? REST_0 : REST_1;
        const V3 a{ cv[tri * 3 + 0].x, cv[tri * 3 + 0].y, cv[tri * 3 + 0].z };
        const V3 b{ cv[tri * 3 + 1].x, cv[tri * 3 + 1].y, cv[tri * 3 + 1].z };
        const V3 c{ cv[tri * 3 + 2].x, cv[tri * 3 + 2].y, cv[tri * 3 + 2].z };
        const V3 pp{ parts[pi].x, parts[pi].y, parts[pi].z };
        const V3 ab = sub(b, a), ac = sub(c, a);
        const V3 closest = cpTri2(pp, a, b, c, ab, ac);
        const V3 v = sub(pp, closest);
        const float sqDist = dot(v, v);
        if (!(sqDist <= cDistance * cDistance)) continue;       // C5
        ++faceHits;
        const float m = std::sqrt(sqDist);
        // C6: unguarded.  P9.36: `v / m` on a PxVec3 is PxVec3.h:219 -- one
        // reciprocal, three multiplies -- not three divides.
        const V3 n = scl(v, 1.0f / m);
        const float pen = m - rest;                             // C4
        const V3 contact = sub(pp, scl(n, rest));               // C4
        Exp e{};
        e.point = F4{ contact.x, contact.y, contact.z, 0.0f };
        e.normalPen = F4{ -n.x, -n.y, -n.z, pen };              // C4: NEGATED
        e.bary = computeBary(a, b, c, closest);
        e.pairInd0 = (uint64_t(PS_ID) << 32) | pi;              // C3
        e.pairInd1 = (CLOTH_ID << 20) | tri;
        expected.push_back(e);
    }

    const uint32_t got = downloadBuffer<uint32_t>(dev, w.count.h, 1)[0];
    PXG_CHECK_MSG(faceHits >= 6u,
                  "only %u of the %u seeded pairs made contact; the rig must reach the face, "
                  "the three vertex regions and two edge regions or the bit-exact comparison "
                  "below is testing one branch of closestPtPointTriangle2", faceHits, NB_STACK);
    PXG_CHECK_MSG(got == uint32_t(expected.size()),
                  "contact COUNT %u, the mirror says %zu over %u clamped pairs of %u claimed",
                  got, expected.size(), NB_STACK, NB_FOUND);

    const auto pt = downloadBuffer<F4>(dev, w.point.h, MAXC);
    const auto np = downloadBuffer<F4>(dev, w.normalPen.h, MAXC);
    const auto ba = downloadBuffer<F4>(dev, w.barycentric.h, MAXC);
    const auto info = downloadBuffer<uint32_t>(dev, w.info.h, MAXC * 4u);
    const auto byX = downloadBuffer<uint64_t>(dev, w.byRigid.h, MAXC);
    const auto tmpX = downloadBuffer<uint32_t>(dev, w.tempByRigid.h, MAXC);

    struct Rec { uint32_t key[15]; };
    auto less = [](const Rec& x, const Rec& y) {
        for (int i = 0; i < 15; ++i) if (x.key[i] != y.key[i]) return x.key[i] < y.key[i];
        return false;
    };
    std::vector<Rec> gpu, ref;
    for (uint32_t i = 0; i < got && i < MAXC; ++i)
    {
        Rec r{};
        r.key[0] = info[i * 4 + 0]; r.key[1] = info[i * 4 + 1];   // pairInd0 (u64)
        r.key[2] = info[i * 4 + 2];                               // pairInd1
        const float src[12] = { pt[i].x, pt[i].y, pt[i].z, pt[i].w,
                                np[i].x, np[i].y, np[i].z, np[i].w,
                                ba[i].x, ba[i].y, ba[i].z, ba[i].w };
        for (int j = 0; j < 12; ++j) r.key[3 + j] = f2u(src[j]);
        gpu.push_back(r);
    }
    for (const Exp& e : expected)
    {
        Rec r{};
        r.key[0] = uint32_t(e.pairInd0 & 0xffffffffull);
        r.key[1] = uint32_t(e.pairInd0 >> 32);
        r.key[2] = e.pairInd1;
        const float src[12] = { e.point.x, e.point.y, e.point.z, e.point.w,
                                e.normalPen.x, e.normalPen.y, e.normalPen.z, e.normalPen.w,
                                e.bary.x, e.bary.y, e.bary.z, e.bary.w };
        for (int j = 0; j < 12; ++j) r.key[3 + j] = f2u(src[j]);
        ref.push_back(r);
    }
    std::sort(gpu.begin(), gpu.end(), less);
    std::sort(ref.begin(), ref.end(), less);

    static const char* const KEY_NAME[15] = {
        "pairInd0.lo", "pairInd0.hi", "pairInd1",
        "point.x", "point.y", "point.z", "point.w",
        "normal.x", "normal.y", "normal.z", "penetration",
        "bary.x", "bary.y", "bary.z", "bary.w"
    };
    uint32_t mismatches = 0u; int firstField = -1; size_t firstAt = 0;
    for (size_t i = 0; i < gpu.size() && i < ref.size(); ++i)
        for (int j = 0; j < 15; ++j)
            if (gpu[i].key[j] != ref[i].key[j])
            { ++mismatches; if (firstField < 0) { firstField = j; firstAt = i; } break; }
    if (mismatches != 0u && firstField >= 0)
    {
        const uint32_t gw = gpu[firstAt].key[firstField], rw = ref[firstAt].key[firstField];
        float gf, rf; std::memcpy(&gf, &gw, 4); std::memcpy(&rf, &rw, 4);
        std::printf("    first difference at sorted slot %zu, field %s: gpu 0x%08X (%.9g) vs "
                    "mirror 0x%08X (%.9g)\n", firstAt, KEY_NAME[firstField], gw, gf, rw, rf);
    }
    PXG_CHECK_MSG(mismatches == 0u,
                  "%u of %zu contacts differ BIT-EXACTLY from the host mirror of "
                  "clothParticleCollision", mismatches, gpu.size());

    // C3: the sort key is the ENCODED PARTICLE, not a rigid node index -- and the
    // 32-bit companion is its LOW half.
    uint32_t keyBad = 0u;
    for (uint32_t i = 0; i < got && i < MAXC; ++i)
    {
        const uint64_t pair0 = (uint64_t(info[i * 4 + 1]) << 32) | info[i * 4 + 0];
        if (byX[i] != pair0 || tmpX[i] != uint32_t(pair0 & 0xffffffffull)) ++keyBad;
    }
    PXG_CHECK_MSG(keyBad == 0u,
                  "%u contacts sort by something other than their own encoded particle index "
                  "(C3); the rigid callers pass a node index here and this one must not", keyBad);

    // C8: the high-water mark is the UNCLAMPED count.
    const uint32_t needed = downloadBuffer<uint32_t>(dev, bNeeded.h, 1)[0];
    PXG_CHECK_MSG(needed == NB_FOUND * 16u,
                  "stackSizeNeeded is %u, expected the UNCLAMPED %u * 16 = %u (the clamped "
                  "count would give %u)", needed, NB_FOUND, NB_FOUND * 16u, NB_STACK * 16u);

    std::printf("    [ note ] %zu contacts over %u clamped pairs of %u claimed; %u reached the "
                "closest-point branches\n", expected.size(), NB_STACK, NB_FOUND, faceHits);

    // ---- C6, on its own: a particle EXACTLY on the triangle ------------------------
    // `n = v / m` has no zero guard upstream, so m == 0 gives a non-finite normal.
    // Compared as "both sides are non-finite" rather than bit-for-bit: NaN payloads
    // are not something this port controls, and pinning one would be a claim about
    // the driver rather than about the kernel.
    //
    // The particle sits on VERTEX a, not on the centroid.  The centroid was the first
    // attempt and it does not work: closestPtPointTriangle2's face branch recomputes
    // the point through barycentric weights, so the result differs from the centroid
    // in the last bits and `m` comes out around 1e-8 rather than 0 -- the check then
    // failed on a FINITE normal and was right to.  On a vertex the function returns
    // `a` verbatim from its first branch, so `v` is exactly zero by construction.
    {
        const V3 ctr = vtx(0, 0);
        std::vector<F4> one = { F4{ ctr.x, ctr.y, ctr.z, 1.0f } };
        DevBuf bOnePart = makeDev(dev, one);
        std::vector<PxgParticleSystem> psOne(PS_ID + 1u);
        std::memset(psOne.data(), 0, psOne.size() * sizeof(PxgParticleSystem));
        psOne[PS_ID].mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bOnePart.addr));
        psOne[PS_ID].mCommonData.mNumParticles = 1u;
        DevBuf bPsOne = makeDev(dev, psOne);
        std::vector<uint32_t> oneStack = { 0u, 0u, 0u, 0u };
        DevBuf bOneStack = makeDev(dev, oneStack);
        std::vector<uint32_t> oneFound = { 1u };
        DevBuf bOneNum = makeDev(dev, oneFound);
        DevBuf bOneNeeded = makeDev(dev, zero);
        WriterBufs w2 = makeWriter(dev, 4u);

        ClothPsCgPush q = p;
        q.particleSystems = bPsOne.addr;
        q.stackPtr = bOneStack.addr; q.midphasePairsNum = bOneNum.addr;
        q.stackSizeBytes = 16u; q.stackSizeNeeded = bOneNeeded.addr;
        q.wPoint = w2.point.addr; q.wNormalPen = w2.normalPen.addr; q.wBary = w2.barycentric.addr;
        q.wInfo = w2.info.addr; q.wCount = w2.count.addr;
        q.wByX = w2.byRigid.addr; q.wTempByX = w2.tempByRigid.addr; q.wIdxByX = w2.idxByRigid.addr;
        q.wMaxContacts = 4u;
        k.dispatch(512u, {}, q, 1u, 1u);
        dev->flush();

        const uint32_t n2 = downloadBuffer<uint32_t>(dev, w2.count.h, 1)[0];
        PXG_CHECK_MSG(n2 == 1u,
                      "a particle exactly ON the triangle produced %u contacts, expected 1: the "
                      "gate is `sqDist <= cDistance^2` (C5) and 0 passes it", n2);
        const F4 gotNp = downloadBuffer<F4>(dev, w2.normalPen.h, 4)[0];
        const bool nonFinite = !std::isfinite(gotNp.x) || !std::isfinite(gotNp.y) ||
                               !std::isfinite(gotNp.z);
        PXG_CHECK_MSG(nonFinite,
                      "the degenerate contact's normal is (%g, %g, %g), which is finite.  "
                      "Upstream divides by the distance with no zero guard (C6, upstream "
                      "candidate 39); a port that added one would pass every other check here "
                      "and silently stop matching the engine",
                      gotNp.x, gotNp.y, gotNp.z);
        std::printf("    [ note ] degenerate case: normal (%g, %g, %g), penetration %g -- "
                    "unguarded divide, as upstream\n", gotNp.x, gotNp.y, gotNp.z, gotNp.w);

        destroyWriter(dev, w2);
        for (PxgBufferHandle h : { bOnePart.h, bPsOne.h, bOneStack.h, bOneNum.h, bOneNeeded.h })
            dev->destroyBuffer(h);
    }

    // ---- the WRITER'S CAPACITY GATE ----------------------------------------------
    // `if (index >= maxNumContacts) return;` never fires above -- 6 contacts against
    // a 32-slot writer -- so a port that dropped it passed, and the sweep said so.
    // Here the same pairs run against a writer that holds fewer contacts than they
    // make: the COUNTER must still climb past the cap (the atomicAdd happens before
    // the gate), and the buffers past the cap must keep their seeded values.
    {
        const uint32_t CAP = 3u;
        WriterBufs wc = makeWriter(dev, 8u);   // allocate 8, TELL the kernel 3
        DevBuf bCapNeeded = makeDev(dev, zero);
        ClothPsCgPush q = p;
        q.stackSizeNeeded = bCapNeeded.addr;
        q.wPoint = wc.point.addr; q.wNormalPen = wc.normalPen.addr; q.wBary = wc.barycentric.addr;
        q.wInfo = wc.info.addr; q.wCount = wc.count.addr;
        q.wByX = wc.byRigid.addr; q.wTempByX = wc.tempByRigid.addr; q.wIdxByX = wc.idxByRigid.addr;
        q.wMaxContacts = CAP;
        k.dispatch(512u, {}, q, 1u, 1u);
        dev->flush();

        const uint32_t nc = downloadBuffer<uint32_t>(dev, wc.count.h, 1)[0];
        PXG_CHECK_MSG(nc == uint32_t(expected.size()),
                      "with a %u-slot cap the counter reads %u; it must still reach %zu, because "
                      "the atomicAdd happens BEFORE the capacity gate and the engine sizes the "
                      "next frame from it", CAP, nc, expected.size());
        const auto capIdx = downloadBuffer<uint32_t>(dev, wc.idxByRigid.h, 8u);
        uint32_t past = 0u;
        for (uint32_t i = CAP; i < 8u; ++i)
            if (capIdx[i] != 0xffffffffu) ++past;
        PXG_CHECK_MSG(past == 0u,
                      "%u slots past the %u-slot cap were written; the writer's gate is what "
                      "stops a contact stream from running off its buffer", past, CAP);
        std::printf("    [ note ] capacity run: cap %u, counter %u, %u slots past the cap "
                    "written\n", CAP, nc, past);

        destroyWriter(dev, wc);
        dev->destroyBuffer(bCapNeeded.h);
    }

    // ---- the THREAD STRIDE, which the run above cannot reach ----------------------
    // 512 blocks x 256 threads = 131072, and seven pairs never reach it.  Hand the
    // kernel more pairs than the grid has threads, with the only contacting ones in
    // the tail: a smaller stride does not shift the window, it visits every index
    // several times over, so the count -- not the membership -- is what separates
    // them.  Same rig, and same reasoning, as P9.29's.
    {
        const uint32_t STRIDE = 512u * 256u;
        const uint32_t TAIL = 8u;
        const uint32_t FAR_PARTICLE = 6u;      // the one placed far from everything
        std::vector<uint32_t> big(size_t(STRIDE + TAIL) * 4u, 0u);
        for (uint32_t i = 0; i < STRIDE + TAIL; ++i)
        {
            // Every pair uses the SAME (particle, triangle) mapping the run above
            // does -- particle p against triangle triFor(p) -- so the tail contacts
            // and the body does not.  Writing (0, 0) here instead was silent until
            // the index spaces were offset: it had been a contacting pair by
            // accident, and became a non-contacting one for the same reason.
            const bool tail = (i >= STRIDE);
            const uint32_t particle = tail ? 0u : FAR_PARTICLE;
            big[size_t(i) * 4 + 1] = particle;
            big[size_t(i) * 4 + 2] = triFor(particle);
        }
        DevBuf bBig = makeDev(dev, big);
        std::vector<uint32_t> bigFound = { STRIDE + TAIL };
        DevBuf bBigNum = makeDev(dev, bigFound);
        DevBuf bBigNeeded = makeDev(dev, zero);
        WriterBufs w2 = makeWriter(dev, 64u);

        ClothPsCgPush q = p;
        q.stackPtr = bBig.addr; q.midphasePairsNum = bBigNum.addr;
        q.stackSizeBytes = (STRIDE + TAIL) * 16u;
        q.stackSizeNeeded = bBigNeeded.addr;
        q.wPoint = w2.point.addr; q.wNormalPen = w2.normalPen.addr; q.wBary = w2.barycentric.addr;
        q.wInfo = w2.info.addr; q.wCount = w2.count.addr;
        q.wByX = w2.byRigid.addr; q.wTempByX = w2.tempByRigid.addr; q.wIdxByX = w2.idxByRigid.addr;
        q.wMaxContacts = 64u;
        k.dispatch(512u, {}, q, 1u, 1u);
        dev->flush();

        const uint32_t n2 = downloadBuffer<uint32_t>(dev, w2.count.h, 1)[0];
        PXG_CHECK_MSG(n2 == TAIL,
                      "the stride run produced %u contacts, expected exactly %u over %u pairs "
                      "against %u threads; more means the stride is too small and pairs are "
                      "visited several times, fewer means the second iteration never happened",
                      n2, TAIL, STRIDE + TAIL, STRIDE);
        std::printf("    [ note ] stride run: %u pairs over %u threads -> %u contacts from the "
                    "%u-pair tail\n", STRIDE + TAIL, STRIDE, n2, TAIL);

        destroyWriter(dev, w2);
        for (PxgBufferHandle h : { bBig.h, bBigNum.h, bBigNeeded.h }) dev->destroyBuffer(h);
    }

    destroyWriter(dev, w);
    for (PxgBufferHandle h : { bCv.h, bCi.h, bDesc.h, bParts.h, bPs.h, bShapes.h, bCms.h,
                               bCd.h, bRd.h, bXf.h, bStack.h, bNum.h, bNeeded.h })
        dev->destroyBuffer(h);
}


// ===========================================================================
// P9.31 -- cloth_particleContactPrepareLaunch, bit-exact against a host mirror,
// plus a COVERAGE run large enough to make the work-index formula observable.
//
// The small run pins the arithmetic.  The large one exists because of what the
// shader's P1 note says: a grid-stride substitution is observation-equivalent,
// but dropping the `nbIterationsPerBlock *` factor is not -- and neither can be
// told apart below 262144 contacts, because that is gridDim * blockDim and
// nbIterationsPerBlock is 1 for anything smaller.
// ===========================================================================
namespace p931prep
{
using p817cg::V3;

// Ten pointers -> 80 bytes.
struct ClothPsPreparePush
{
    uint64_t clothes, particleSystems, contacts, normalPens, barycentrics,
             contactInfos, numContacts, constraints, clothAppliedForces,
             particleAppliedForces;
};
static_assert(sizeof(ClothPsPreparePush) == 80, "cloth ps prepare blob must be 80 bytes");
static_assert(offsetof(ClothPsPreparePush, contactInfos) == 40, "contactInfos @40");
static_assert(offsetof(ClothPsPreparePush, constraints) == 56, "constraints @56");

// PxgFEMParticleConstraintBlock (PxgFEMCore.h:184): float4[32], float4[32],
// float[32] = 1152 bytes, and it is a struct of LANE ARRAYS, not an array of
// per-contact records.
static const uint32_t CB_STRIDE = 1152u;
static const uint32_t CB_WORDS = CB_STRIDE / 4u;
static const uint32_t CB_OFF_NORMAL_PEN = 0u;
static const uint32_t CB_OFF_BARY = 128u;      // 512 bytes / 4
static const uint32_t CB_OFF_VELMULT = 256u;   // 1024 bytes / 4
// PxgFEMCore.h is not in this TU's include set, so the layout is pinned by
// arithmetic against the header's declaration rather than by sizeof:
//   float4[32] + float4[32] + float[32] = 512 + 512 + 128 = 1152.
static_assert(CB_OFF_BARY * 4u == 512u, "barycentric lane array starts at byte 512");
static_assert(CB_OFF_VELMULT * 4u == 1024u, "velMultiplier lane array starts at byte 1024");
static_assert(CB_STRIDE == 512u + 512u + 128u, "the block is three lane arrays, 1152 bytes");
} // namespace p931prep

PXG_TEST(femClothSimPsContactPrepare)
{
    using namespace p816dc;
    using namespace p931prep;
    using p817cg::V3;
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t CLOTH_ID = 1u, PS_ID = 2u;
    const float REST_DISTANCE = 0.037f;

    // ---- the cloth: 24 triangles, each with its own accumulated delta ---------------
    const uint32_t NB_TRIS = 24u;
    std::vector<F4> accum;      // mAccumulatedDeltaPos, .w = inverse mass per vertex
    std::vector<TriIdx4> ci;
    for (uint32_t t = 0; t < NB_TRIS; ++t)
    {
        const uint32_t base = uint32_t(accum.size());
        // Distinct per-vertex deltas AND distinct inverse masses, so P6's float4
        // interpolation of .w is separable from any single vertex's mass.
        // TRIANGLE 23 IS ENTIRELY PINNED.  Contact 4 pairs it with particle 28,
        // whose inverse mass is also zero, so that contact -- and only that one --
        // has unitResponse == 0 and takes the velMultiplier guard's else branch.
        // Without it `(unitResponse > 0) ? 1/unitResponse : 0` and a bare
        // reciprocal agree everywhere and the guard is untested; the sweep said so.
        const bool pinnedTri = (t == NB_TRIS - 1u);
        const float w0 = pinnedTri ? 0.0f : (0.5f + 0.1f * float(t % 3));
        const float w1 = pinnedTri ? 0.0f : (0.7f + 0.1f * float(t % 5));
        const float w2 = pinnedTri ? 0.0f : (0.9f + 0.1f * float(t % 7));
        accum.push_back(F4{ 0.01f * float(t), 0.02f * float(t), -0.03f * float(t), w0 });
        accum.push_back(F4{ -0.04f * float(t), 0.015f * float(t), 0.02f * float(t), w1 });
        accum.push_back(F4{ 0.025f * float(t), -0.01f * float(t), 0.035f * float(t), w2 });
        ci.push_back(TriIdx4{ base, base + 1u, base + 2u, 0u });
    }
    DevBuf bAccum = makeDev(dev, accum);
    if (!bAccum.addr) { std::printf("  NOTE: no BDA -> skip\n"); dev->destroyBuffer(bAccum.h); return; }
    DevBuf bCi = makeDev(dev, ci);

    PxgFEMCloth cloth = makeZeroDesc();
    cloth.mAccumulatedDeltaPos = reinterpret_cast<decltype(cloth.mAccumulatedDeltaPos)>(size_t(bAccum.addr));
    cloth.mTriangleVertexIndices = reinterpret_cast<decltype(cloth.mTriangleVertexIndices)>(size_t(bCi.addr));
    cloth.mNbTriangles = NB_TRIS;
    cloth.mNbVerts = uint32_t(accum.size());
    cloth.mRestDistance = REST_DISTANCE;
    std::vector<PxgFEMCloth> descArr(CLOTH_ID + 1u, makeZeroDesc());
    descArr[CLOTH_ID] = cloth;
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- the particle system: sorted deltas with distinct inverse masses ------------
    const uint32_t NB_PARTICLES = 48u;
    std::vector<F4> sortedDeltaP(NB_PARTICLES);
    for (uint32_t i = 0; i < NB_PARTICLES; ++i)
        sortedDeltaP[i] = F4{ 0.005f * float(i), -0.007f * float(i), 0.009f * float(i),
                              // every fourth particle is IMMOVABLE, so unitResponse
                              // is separable from invMass1 alone
                              (i % 4u == 0u) ? 0.0f : (0.3f + 0.05f * float(i % 6)) };
    DevBuf bDeltaP = makeDev(dev, sortedDeltaP);
    std::vector<PxgParticleSystem> psArr(PS_ID + 1u);
    std::memset(psArr.data(), 0, psArr.size() * sizeof(PxgParticleSystem));
    psArr[PS_ID].mSortedDeltaP = reinterpret_cast<float4*>(size_t(bDeltaP.addr));
    psArr[PS_ID].mCommonData.mNumParticles = NB_PARTICLES;
    DevBuf bPs = makeDev(dev, psArr);

    // ---- the contacts.  40 of them, so P2's constraint blocks are 0 and 1 and the
    // lane index wraps.
    const uint32_t NB_CONTACTS = 40u;
    std::vector<FemOtherContactInfoHost> infos(NB_CONTACTS);
    std::vector<F4> contacts(NB_CONTACTS), normalPens(NB_CONTACTS), barys(NB_CONTACTS);
    for (uint32_t i = 0; i < NB_CONTACTS; ++i)
    {
        const uint32_t particle = (i * 7u) % NB_PARTICLES;   // NOT i: P-index vs work index
        const uint32_t tri = (i * 5u + 3u) % NB_TRIS;        // and NOT the particle either
        infos[i].pairInd0 = (uint64_t(PS_ID) << 32) | particle;
        infos[i].pairInd1 = (CLOTH_ID << 20) | tri;
        infos[i].rigidMatInd_isInCollision = 0xABCDEF01u;    // must be left alone
        // P3: read and discarded -- seeded with something recognisable so a port
        // that used it produces garbage rather than something plausible.
        contacts[i] = F4{ 1e9f, -1e9f, 1e9f, -1e9f };
        normalPens[i] = F4{ 0.6f - 0.02f * float(i), -0.8f + 0.01f * float(i),
                            0.1f + 0.005f * float(i), -0.05f + 0.004f * float(i) };
        const float v = 0.15f + 0.01f * float(i % 5);
        const float w = 0.25f + 0.01f * float(i % 7);
        barys[i] = F4{ 1.0f - v - w, v, w, 0.0f };
    }
    DevBuf bInfos = makeDev(dev, infos), bContacts = makeDev(dev, contacts);
    DevBuf bNormalPens = makeDev(dev, normalPens), bBarys = makeDev(dev, barys);
    std::vector<uint32_t> nc = { NB_CONTACTS };
    DevBuf bNum = makeDev(dev, nc);

    const uint32_t NB_BLOCKS = (NB_CONTACTS + 31u) / 32u;
    std::vector<uint32_t> constraintWords(size_t(NB_BLOCKS) * CB_WORDS, 0xEEEEEEEEu);
    DevBuf bConstraints = makeDev(dev, constraintWords);
    // The applied-force arrays are float2, so 2 floats per contact -- but they are
    // allocated with FOUR and the tail seeded, because a port that wrote them as
    // float4 zeroes bytes [16i, 16i+16) instead of [8i, 8i+8) and every word this
    // fixture checks still comes back zero.  The sweep survived exactly that; the
    // tail is what tells the two apart (and what a float4 write runs off the end
    // of in the engine's own allocation).
    std::vector<float> forcesSeed(NB_CONTACTS * 4u, -12345.0f);
    DevBuf bClothForces = makeDev(dev, forcesSeed), bPartForces = makeDev(dev, forcesSeed);

    ClothPsPreparePush p = {};
    p.clothes = bDesc.addr; p.particleSystems = bPs.addr;
    p.contacts = bContacts.addr; p.normalPens = bNormalPens.addr; p.barycentrics = bBarys.addr;
    p.contactInfos = bInfos.addr; p.numContacts = bNum.addr;
    p.constraints = bConstraints.addr;
    p.clothAppliedForces = bClothForces.addr; p.particleAppliedForces = bPartForces.addr;

    Kernel k("femClothPsPrepareBda.spv", "cloth_particleContactPrepareLaunch",
             {}, sizeof(ClothPsPreparePush));
    k.dispatch(1024u, {}, p, 1u, 1u);
    dev->flush();

    // ---- the mirror ------------------------------------------------------------------
    const auto gotC = downloadBuffer<uint32_t>(dev, bConstraints.h, uint32_t(constraintWords.size()));
    const auto gotClothF = downloadBuffer<float>(dev, bClothForces.h, NB_CONTACTS * 4u);
    const auto gotPartF = downloadBuffer<float>(dev, bPartForces.h, NB_CONTACTS * 4u);

    uint32_t mismatches = 0u;
    for (uint32_t i = 0; i < NB_CONTACTS; ++i)
    {
        const uint32_t particle = uint32_t(infos[i].pairInd0 & 0xffffffffull);
        const uint32_t tri = infos[i].pairInd1 & 0xfffffu;
        const TriIdx4& tv = ci[tri];
        const F4& b = barys[i];
        // P6: the whole float4, so .w is the interpolated inverse mass.
        const F4 v0 = accum[tv.x], v1 = accum[tv.y], v2 = accum[tv.z];
        const F4 delta1 = F4{ v0.x * b.x + v1.x * b.y + v2.x * b.z,
                              v0.y * b.x + v1.y * b.y + v2.y * b.z,
                              v0.z * b.x + v1.z * b.y + v2.z * b.z,
                              v0.w * b.x + v1.w * b.y + v2.w * b.z };
        const float invMass1 = delta1.w;
        const F4& np = normalPens[i];
        const V3 normal{ -np.x, -np.y, -np.z };                        // P4
        const F4& dp = sortedDeltaP[particle];
        const float invMass0 = dp.w;
        const V3 delta{ delta1.x - dp.x, delta1.y - dp.y, delta1.z - dp.z };  // P5 order
        const float pen = np.w + (normal.x * delta.x + normal.y * delta.y + normal.z * delta.z)
                          - REST_DISTANCE;
        const float unitResponse = invMass0 + invMass1;
        const float velMultiplier = (unitResponse > 0.0f) ? (1.0f / unitResponse) : 0.0f;

        const size_t blk = size_t(i / 32u) * CB_WORDS;
        const uint32_t lane = i & 31u;                                  // P2
        const uint32_t* gn = &gotC[blk + CB_OFF_NORMAL_PEN + lane * 4u];
        const uint32_t* gb = &gotC[blk + CB_OFF_BARY + lane * 4u];
        const uint32_t gv = gotC[blk + CB_OFF_VELMULT + lane];
        const uint32_t refN[4] = { f2u(normal.x), f2u(normal.y), f2u(normal.z), f2u(pen) };
        const uint32_t refB[4] = { f2u(b.x), f2u(b.y), f2u(b.z), f2u(b.w) };
        bool bad = (gv != f2u(velMultiplier));
        for (int j = 0; j < 4; ++j) { if (gn[j] != refN[j] || gb[j] != refB[j]) bad = true; }
        if (bad)
        {
            if (mismatches == 0u)
            {
                float gnp; std::memcpy(&gnp, &gn[3], 4);
                float gvm; std::memcpy(&gvm, &gv, 4);
                std::printf("    first difference at contact %u (block %u lane %u): "
                            "pen gpu %.9g vs %.9g, velMult gpu %.9g vs %.9g\n",
                            i, i / 32u, lane, gnp, pen, gvm, velMultiplier);
            }
            ++mismatches;
        }
        // P8: both force arrays zeroed, as float2 (stride 8, not 16).
        if (gotClothF[i * 2] != 0.0f || gotClothF[i * 2 + 1] != 0.0f ||
            gotPartF[i * 2] != 0.0f || gotPartF[i * 2 + 1] != 0.0f)
        {
            std::printf("    applied forces at %u not zeroed: cloth (%g, %g) particle (%g, %g)\n",
                        i, gotClothF[i * 2], gotClothF[i * 2 + 1],
                        gotPartF[i * 2], gotPartF[i * 2 + 1]);
            ++mismatches;
        }
    }
    PXG_CHECK_MSG(mismatches == 0u,
                  "%u of %u prepared constraints differ BIT-EXACTLY from the host mirror",
                  mismatches, NB_CONTACTS);

    // Nothing past the contact count may have been touched.
    uint32_t past = 0u;
    for (uint32_t i = NB_CONTACTS; i < NB_BLOCKS * 32u; ++i)
    {
        const size_t blk = size_t(i / 32u) * CB_WORDS;
        if (gotC[blk + CB_OFF_VELMULT + (i & 31u)] != 0xEEEEEEEEu) ++past;
    }
    PXG_CHECK_MSG(past == 0u, "%u constraint lanes past the contact count were written", past);

    // The info records must come back untouched -- this kernel reads them only.
    const auto gotInfos = downloadBuffer<FemOtherContactInfoHost>(dev, bInfos.h, NB_CONTACTS);
    uint32_t infoTouched = 0u;
    for (uint32_t i = 0; i < NB_CONTACTS; ++i)
        if (gotInfos[i].rigidMatInd_isInCollision != 0xABCDEF01u) ++infoTouched;
    PXG_CHECK_MSG(infoTouched == 0u, "%u contact infos were modified by a read-only kernel",
                  infoTouched);

    // The float2 stride, measured: everything from 2*NB_CONTACTS on must still hold
    // the seed.
    uint32_t tailTouched = 0u;
    for (uint32_t i = NB_CONTACTS * 2u; i < NB_CONTACTS * 4u; ++i)
    {
        if (gotClothF[i] != -12345.0f) ++tailTouched;
        if (gotPartF[i] != -12345.0f) ++tailTouched;
    }
    PXG_CHECK_MSG(tailTouched == 0u,
                  "%u words past the float2 region were written; the applied-force arrays are "
                  "float2 (stride 8), and a float4 store covers twice the ground and runs off "
                  "the end of the engine's allocation", tailTouched);

    // The velMultiplier guard, measured rather than assumed to be reachable.
    uint32_t zeroResponse = 0u;
    for (uint32_t i = 0; i < NB_CONTACTS; ++i)
    {
        const uint32_t particle = uint32_t(infos[i].pairInd0 & 0xffffffffull);
        const uint32_t tri = infos[i].pairInd1 & 0xfffffu;
        const TriIdx4& tv = ci[tri];
        const F4& b = barys[i];
        const float im1 = accum[tv.x].w * b.x + accum[tv.y].w * b.y + accum[tv.z].w * b.z;
        if (sortedDeltaP[particle].w + im1 <= 0.0f) ++zeroResponse;
    }
    PXG_CHECK_MSG(zeroResponse >= 1u,
                  "no contact has unitResponse == 0, so the velMultiplier guard's else branch "
                  "is unreachable and a bare reciprocal would pass");

    std::printf("    [ note ] %u constraints over %u blocks, %u lanes past the count untouched, "
                "%u with zero unit response\n",
                NB_CONTACTS, NB_BLOCKS, NB_BLOCKS * 32u - NB_CONTACTS, zeroResponse);

    // ---- COVERAGE, past the point where the work-index formula matters -------------
    // gridDim * blockDim is 1024 * 256 = 262144.  Below that nbIterationsPerBlock is 1
    // and the formula collapses to a flat index, so NOTHING about it is observable --
    // which is why the small run above cannot see it and why the shader's P1 note now
    // says a grid-stride substitution is equivalent rather than wrong.
    //
    // Above it the factor starts to matter.  With 300000 contacts
    // nbIterationsPerBlock is 2, so each block owns 512 contiguous indices and the
    // grid covers 524288.  Drop the `nbIterationsPerBlock *` factor and the per-block
    // ranges overlap instead: the grid then reaches only 1024*256 + 256 = 262400, and
    // everything from there to 299999 is never written.  That is what this run sees.
    {
        const uint32_t BIG = 300000u;
        const uint32_t BIG_BLOCKS = (BIG + 31u) / 32u;
        std::vector<FemOtherContactInfoHost> bigInfos(BIG);
        for (uint32_t i = 0; i < BIG; ++i)
        {
            bigInfos[i].pairInd0 = (uint64_t(PS_ID) << 32) | 1u;   // particle 1 is movable
            bigInfos[i].pairInd1 = (CLOTH_ID << 20) | 0u;
            bigInfos[i].rigidMatInd_isInCollision = 0u;
        }
        DevBuf bBigInfos = makeDev(dev, bigInfos);
        std::vector<F4> bigF4(BIG, F4{ 0.1f, 0.2f, 0.3f, 0.4f });
        DevBuf bBigContacts = makeDev(dev, bigF4);
        DevBuf bBigNormals = makeDev(dev, bigF4);
        std::vector<F4> bigBary(BIG, F4{ 0.5f, 0.25f, 0.25f, 0.0f });
        DevBuf bBigBary = makeDev(dev, bigBary);
        std::vector<uint32_t> bigNum = { BIG };
        DevBuf bBigNum = makeDev(dev, bigNum);
        std::vector<uint32_t> bigConstraints(size_t(BIG_BLOCKS) * CB_WORDS, 0xEEEEEEEEu);
        DevBuf bBigConstraints = makeDev(dev, bigConstraints);
        std::vector<float> bigForces(size_t(BIG) * 2u, -1.0f);
        DevBuf bBigCF = makeDev(dev, bigForces), bBigPF = makeDev(dev, bigForces);

        if (!bBigConstraints.addr)
        {
            std::printf("    NOTE: the %u-contact coverage run could not allocate; skipped\n", BIG);
        }
        else
        {
            ClothPsPreparePush q = p;
            q.contacts = bBigContacts.addr; q.normalPens = bBigNormals.addr;
            q.barycentrics = bBigBary.addr; q.contactInfos = bBigInfos.addr;
            q.numContacts = bBigNum.addr; q.constraints = bBigConstraints.addr;
            q.clothAppliedForces = bBigCF.addr; q.particleAppliedForces = bBigPF.addr;
            k.dispatch(1024u, {}, q, 1u, 1u);
            dev->flush();

            const auto bigGot = downloadBuffer<uint32_t>(dev, bBigConstraints.h,
                                                         uint32_t(bigConstraints.size()));
            uint32_t unwritten = 0u, firstUnwritten = 0xffffffffu;
            for (uint32_t i = 0; i < BIG; ++i)
            {
                const size_t blk = size_t(i / 32u) * CB_WORDS;
                if (bigGot[blk + CB_OFF_VELMULT + (i & 31u)] == 0xEEEEEEEEu)
                {
                    if (firstUnwritten == 0xffffffffu) firstUnwritten = i;
                    ++unwritten;
                }
            }
            PXG_CHECK_MSG(unwritten == 0u,
                          "%u of %u constraint lanes were never written, first at %u.  The grid "
                          "reaches %u threads, so with %u contacts nbIterationsPerBlock is 2 and "
                          "each block must own 512 contiguous indices; dropping that factor "
                          "leaves everything past %u untouched",
                          unwritten, BIG, firstUnwritten, 1024u * 256u, BIG, 1024u * 256u + 256u);
            // Report the MEASURED count, not the hoped-for one: PXG_CHECK_MSG records a
            // failure and carries on, so a note that says "all lanes written"
            // unconditionally contradicts the check three lines above it.
            std::printf("    [ note ] coverage run: %u contacts over %u threads, %u lanes "
                        "unwritten\n", BIG, 1024u * 256u, unwritten);
        }

        for (PxgBufferHandle h : { bBigInfos.h, bBigContacts.h, bBigNormals.h, bBigBary.h,
                                   bBigNum.h, bBigConstraints.h, bBigCF.h, bBigPF.h })
            dev->destroyBuffer(h);
    }

    for (PxgBufferHandle h : { bAccum.h, bCi.h, bDesc.h, bDeltaP.h, bPs.h, bInfos.h, bContacts.h,
                               bNormalPens.h, bBarys.h, bNum.h, bConstraints.h, bClothForces.h,
                               bPartForces.h })
        dev->destroyBuffer(h);
}


// ===========================================================================
// P9.31 -- the two cloth_solveCPOutput*DeltaVLaunch kernels, bit-exact.
//
// One rig, two dispatches over the SAME constraints: the cloth side accumulates
// a position delta onto the triangle's vertices, the particle side writes a
// velocity delta per contact.  They share everything up to the force update, so
// running both over identical inputs is also a check that the shared half stayed
// shared.
//
// The cloth side writes through ATOMICS, so contacts are laid out one per
// triangle with no shared vertices: every vertex has exactly one writer and the
// comparison can be bit-exact.  One deliberately CONTENDED pair sits at the end
// and is checked on its .w count alone, which is order-invariant.
// ===========================================================================
namespace p931solve
{
using p817cg::V3;

struct ClothPsSolveClothPush
{
    uint64_t clothes, particleSystems, contactInfos, constraints, numContacts,
             appliedForces, materials;
};
static_assert(sizeof(ClothPsSolveClothPush) == 56, "cloth-side solve blob must be 56 bytes");

struct ClothPsSolveParticlePush
{
    uint64_t clothes, particleSystems, contactInfos, constraints, numContacts,
             deltaP, appliedForces, materials;
};
static_assert(sizeof(ClothPsSolveParticlePush) == 64, "particle-side solve blob must be 64 bytes");
static_assert(offsetof(ClothPsSolveParticlePush, deltaP) == 40,
              "deltaP @40 -- inserted BEFORE appliedForces, which is what makes the two "
              "packs different lengths");

static const float BIAS = 0.7f;                  // FEMCLOTH_BIAS_COEFFICIENT
static const uint32_t CLOTH_MAT_STRIDE = 32u;    // PxsDeformableSurfaceMaterialData
static const uint32_t CLOTH_MAT_DYNFRICTION_F = 2u;   // youngs, poissons, dynamicFriction
static const uint32_t PS_MAT_STRIDE = 32u;
static const uint32_t PS_MAT_FRICTION_F = 0u;

using p931prep::CB_STRIDE;
using p931prep::CB_WORDS;
using p931prep::CB_OFF_NORMAL_PEN;
using p931prep::CB_OFF_BARY;
using p931prep::CB_OFF_VELMULT;

// PxVec3::normalize -- returns the magnitude, normalises in place if m > 0.
// P9.36: reciprocal-then-multiply, per PxVec3.h:260.  See normalize3 above.
static float pxNormalize(V3& v)
{
    const float m = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (m > 0.0f) { const float r = 1.0f / m; v.x *= r; v.y *= r; v.z *= r; }
    return m;
}
} // namespace p931solve

PXG_TEST(femClothSimPsSolveCP)
{
    using namespace p816dc;
    using namespace p931prep;
    using namespace p931solve;
    using p817cg::V3;
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t CLOTH_ID = 1u, PS_ID = 2u;

    // ---- contacts: one per triangle, no shared vertices, plus a contended pair -----
    const uint32_t NB_SINGLE = 10u;      // contacts 0..9, one triangle each
    const uint32_t NB_CONTENDED = 2u;    // contacts 10,11 share triangle NB_SINGLE
    const uint32_t NB_CONTACTS = NB_SINGLE + NB_CONTENDED;
    const uint32_t NB_TRIS = NB_SINGLE + 1u;

    std::vector<F4> accum; std::vector<TriIdx4> ci;
    std::vector<uint16_t> matIdx(NB_TRIS);
    for (uint32_t t = 0; t < NB_TRIS; ++t)
    {
        const uint32_t base = uint32_t(accum.size());
        // Vertex 1 of triangle 3 is PINNED (inverse mass 0) and vertex 2 of
        // triangle 5 gets a barycentric of zero below -- S8's two skip conditions,
        // one each.
        const float m0 = 0.4f + 0.05f * float(t);
        const float m1 = (t == 3u) ? 0.0f : (0.6f + 0.03f * float(t));
        const float m2 = 0.8f - 0.02f * float(t);
        accum.push_back(F4{ 0.011f * float(t), -0.021f * float(t), 0.031f * float(t), m0 });
        accum.push_back(F4{ -0.013f * float(t), 0.023f * float(t), -0.033f * float(t), m1 });
        accum.push_back(F4{ 0.017f * float(t), 0.027f * float(t), 0.037f * float(t), m2 });
        ci.push_back(TriIdx4{ base, base + 1u, base + 2u, 0u });
        matIdx[t] = uint16_t(t % 3u);          // three distinct cloth materials
    }
    DevBuf bAccum = makeDev(dev, accum);
    if (!bAccum.addr) { std::printf("  NOTE: no BDA -> skip\n"); dev->destroyBuffer(bAccum.h); return; }
    DevBuf bCi = makeDev(dev, ci);
    std::vector<uint32_t> matIdxWords((NB_TRIS + 1u) / 2u, 0u);
    for (uint32_t t = 0; t < NB_TRIS; ++t)
        matIdxWords[t >> 1] |= uint32_t(matIdx[t]) << ((t & 1u) * 16u);
    DevBuf bMatIdx = makeDev(dev, matIdxWords);

    // Cloth materials: three records, dynamicFriction the THIRD float of 32 bytes.
    std::vector<float> clothMats(3u * (CLOTH_MAT_STRIDE / 4u), 0.0f);
    const float CLOTH_FRICTION[3] = { 0.20f, 0.55f, 0.00f };
    for (uint32_t m = 0; m < 3u; ++m)
    {
        clothMats[m * 8u + 0u] = 1.0e6f;                    // youngs
        clothMats[m * 8u + 1u] = 0.45f;                     // poissons -- the field the
                                                            // first draft of the shader read
        clothMats[m * 8u + CLOTH_MAT_DYNFRICTION_F] = CLOTH_FRICTION[m];
    }
    DevBuf bClothMats = makeDev(dev, clothMats);

    PxgFEMCloth cloth = makeZeroDesc();
    cloth.mAccumulatedDeltaPos = reinterpret_cast<decltype(cloth.mAccumulatedDeltaPos)>(size_t(bAccum.addr));
    cloth.mTriangleVertexIndices = reinterpret_cast<decltype(cloth.mTriangleVertexIndices)>(size_t(bCi.addr));
    cloth.mMaterialIndices = reinterpret_cast<decltype(cloth.mMaterialIndices)>(size_t(bMatIdx.addr));
    cloth.mNbTriangles = NB_TRIS; cloth.mNbVerts = uint32_t(accum.size());
    std::vector<F4> deltaPosSeed(accum.size(), F4{ 0, 0, 0, 0 });
    DevBuf bDeltaPos = makeDev(dev, deltaPosSeed);
    cloth.mDeltaPos = reinterpret_cast<decltype(cloth.mDeltaPos)>(size_t(bDeltaPos.addr));
    std::vector<PxgFEMCloth> descArr(CLOTH_ID + 1u, makeZeroDesc());
    descArr[CLOTH_ID] = cloth;
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- the particle system -------------------------------------------------------
    const uint32_t NB_PARTICLES = NB_CONTACTS;
    std::vector<F4> sortedDeltaP(NB_PARTICLES);
    std::vector<uint32_t> phases(NB_PARTICLES);
    for (uint32_t i = 0; i < NB_PARTICLES; ++i)
    {
        // S3: contact 7's particle is IMMOVABLE, so that contact must produce no
        // write at all and must leave its applied force exactly as seeded.
        const float im = (i == 7u) ? 0.0f : (0.25f + 0.04f * float(i % 5));
        sortedDeltaP[i] = F4{ 0.006f * float(i), -0.004f * float(i), 0.008f * float(i), im };
        // Two groups -- but with the NON-GROUP bits set, so `phase & GROUP_MASK` is
        // not the identity.  With bare 0/1 phases the mask is a no-op and the mutant
        // that drops it survives, which is what the sweep reported.
        phases[i] = (i % 2u) | 0x40000000u | 0x00100000u;
    }
    DevBuf bDeltaP = makeDev(dev, sortedDeltaP), bPhases = makeDev(dev, phases);
    std::vector<uint32_t> phaseToMat = { (1u << 16) | 0u };     // group 0 -> mat 0, group 1 -> mat 1
    DevBuf bPhaseToMat = makeDev(dev, phaseToMat);
    std::vector<float> psMats(2u * (PS_MAT_STRIDE / 4u), 0.0f);
    const float PS_FRICTION[2] = { 0.30f, 0.90f };
    psMats[0u * 8u + PS_MAT_FRICTION_F] = PS_FRICTION[0];
    psMats[1u * 8u + PS_MAT_FRICTION_F] = PS_FRICTION[1];
    DevBuf bPsMats = makeDev(dev, psMats);

    std::vector<PxgParticleSystem> psArr(PS_ID + 1u);
    std::memset(psArr.data(), 0, psArr.size() * sizeof(PxgParticleSystem));
    psArr[PS_ID].mSortedDeltaP = reinterpret_cast<float4*>(size_t(bDeltaP.addr));
    psArr[PS_ID].mSortedPhaseArray = reinterpret_cast<PxU32*>(size_t(bPhases.addr));
    psArr[PS_ID].mPhaseGroupToMaterialHandle = reinterpret_cast<PxU16*>(size_t(bPhaseToMat.addr));
    psArr[PS_ID].mParticleMaterials = reinterpret_cast<PxsParticleMaterialData*>(size_t(bPsMats.addr));
    psArr[PS_ID].mParticleMaterialStride = PS_MAT_STRIDE;
    psArr[PS_ID].mCommonData.mNumParticles = NB_PARTICLES;
    DevBuf bPs = makeDev(dev, psArr);

    // ---- contact infos and constraints (as the prepare would have left them) --------
    std::vector<FemOtherContactInfoHost> infos(NB_CONTACTS);
    for (uint32_t i = 0; i < NB_CONTACTS; ++i)
    {
        const uint32_t tri = (i < NB_SINGLE) ? i : NB_SINGLE;      // the last two share one
        infos[i].pairInd0 = (uint64_t(PS_ID) << 32) | i;
        infos[i].pairInd1 = (CLOTH_ID << 20) | tri;
        infos[i].rigidMatInd_isInCollision = 0u;
    }
    DevBuf bInfos = makeDev(dev, infos);

    const uint32_t NB_BLOCKS = (NB_CONTACTS + 31u) / 32u;
    std::vector<uint32_t> cw(size_t(NB_BLOCKS) * CB_WORDS, 0u);
    std::vector<F4> refNormalPen(NB_CONTACTS), refBary(NB_CONTACTS);
    std::vector<float> refVelMult(NB_CONTACTS);
    for (uint32_t i = 0; i < NB_CONTACTS; ++i)
    {
        // A tilted normal so no component is zero, and a penetration that changes
        // sign across the set so S7's clamp bites on some contacts and not others.
        V3 n{ 0.3f + 0.05f * float(i), 0.8f - 0.03f * float(i), -0.5f + 0.02f * float(i) };
        pxNormalize(n);
        const float pen = 0.02f - 0.008f * float(i);
        // S8's second skip condition, once in EACH of the three slots.  The first
        // version zeroed only the third, and the mutant that drops the epsilon test
        // from the FIRST branch then survived -- a survivor that says nothing about
        // the code and everything about which branch the rig happened to reach.
        //   contact 5 -> bary.z == 0   (w = 0)
        //   contact 6 -> bary.y == 0   (v = 0)
        //   contact 8 -> bary.x == 0   (v + w = 1)
        float v, w;
        if (i == 5u)      { v = 0.30f; w = 0.0f; }
        else if (i == 6u) { v = 0.0f;  w = 0.25f; }
        else if (i == 8u) { v = 0.40f; w = 0.60f; }
        else              { w = 0.20f + 0.01f * float(i % 4); v = 0.30f + 0.02f * float(i % 3); }
        refNormalPen[i] = F4{ n.x, n.y, n.z, pen };
        refBary[i] = F4{ 1.0f - v - w, v, w, 0.0f };
        refVelMult[i] = 0.7f + 0.05f * float(i % 6);
        const size_t blk = size_t(i / 32u) * CB_WORDS;
        const uint32_t lane = i & 31u;
        cw[blk + CB_OFF_NORMAL_PEN + lane * 4u + 0] = f2u(n.x);
        cw[blk + CB_OFF_NORMAL_PEN + lane * 4u + 1] = f2u(n.y);
        cw[blk + CB_OFF_NORMAL_PEN + lane * 4u + 2] = f2u(n.z);
        cw[blk + CB_OFF_NORMAL_PEN + lane * 4u + 3] = f2u(pen);
        cw[blk + CB_OFF_BARY + lane * 4u + 0] = f2u(refBary[i].x);
        cw[blk + CB_OFF_BARY + lane * 4u + 1] = f2u(refBary[i].y);
        cw[blk + CB_OFF_BARY + lane * 4u + 2] = f2u(refBary[i].z);
        cw[blk + CB_OFF_BARY + lane * 4u + 3] = f2u(refBary[i].w);
        cw[blk + CB_OFF_VELMULT + lane] = f2u(refVelMult[i]);
    }
    DevBuf bConstraints = makeDev(dev, cw);

    // S7: NON-ZERO seeds, so the two clamps are exercised rather than trivially
    // satisfied.  A zero seed makes deltaF = max(0, ...) and the friction step
    // degenerate, and the whole clamp structure becomes invisible.
    std::vector<float> forceSeed(NB_CONTACTS * 2u);
    for (uint32_t i = 0; i < NB_CONTACTS; ++i)
    { forceSeed[i * 2] = 0.004f * float(i % 5); forceSeed[i * 2 + 1] = 0.002f * float(i % 3); }
    DevBuf bForcesA = makeDev(dev, forceSeed), bForcesB = makeDev(dev, forceSeed);
    std::vector<uint32_t> ncv = { NB_CONTACTS };
    DevBuf bNum = makeDev(dev, ncv);
    std::vector<F4> deltaPOut(NB_CONTACTS, F4{ -9, -9, -9, -9 });
    DevBuf bDeltaPOut = makeDev(dev, deltaPOut);

    // ---- run the CLOTH side ---------------------------------------------------------
    ClothPsSolveClothPush pc = {};
    pc.clothes = bDesc.addr; pc.particleSystems = bPs.addr; pc.contactInfos = bInfos.addr;
    pc.constraints = bConstraints.addr; pc.numContacts = bNum.addr;
    pc.appliedForces = bForcesA.addr; pc.materials = bClothMats.addr;
    Kernel kc("femClothPsSolveBda.spv", "cloth_solveCPOutputClothDeltaVLaunch",
              {}, sizeof(ClothPsSolveClothPush));
    kc.dispatch(1024u, {}, pc, 1u, 1u);
    dev->flush();

    // ---- run the PARTICLE side over the same constraints -----------------------------
    ClothPsSolveParticlePush pp = {};
    pp.clothes = bDesc.addr; pp.particleSystems = bPs.addr; pp.contactInfos = bInfos.addr;
    pp.constraints = bConstraints.addr; pp.numContacts = bNum.addr;
    pp.deltaP = bDeltaPOut.addr; pp.appliedForces = bForcesB.addr; pp.materials = bClothMats.addr;
    Kernel kp("femClothPsSolveBda.spv", "cloth_solveCPOutputParticleDeltaVLaunch",
              {}, sizeof(ClothPsSolveParticlePush));
    kp.dispatch(1024u, {}, pp, 1u, 1u);
    dev->flush();

    // ---- the mirror ------------------------------------------------------------------
    std::vector<F4> refDeltaPos(accum.size(), F4{ 0, 0, 0, 0 });
    std::vector<F4> refDeltaP(NB_CONTACTS, F4{ -9, -9, -9, -9 });
    std::vector<float> refForce = forceSeed;
    uint32_t clamped = 0u, skippedByMass = 0u;
    for (uint32_t i = 0; i < NB_CONTACTS; ++i)
    {
        const uint32_t particle = uint32_t(infos[i].pairInd0 & 0xffffffffull);
        const uint32_t tri = infos[i].pairInd1 & 0xfffffu;
        const F4& dp = sortedDeltaP[particle];
        const float invMass0 = dp.w;
        if (invMass0 == 0.0f) { ++skippedByMass; continue; }        // S3

        const uint32_t group = phases[particle] & 0x000fffffu;
        const float friction0 = PS_FRICTION[group];
        const float friction1 = CLOTH_FRICTION[matIdx[tri]];
        const float frictionCoefficient = (friction0 + friction1) * 0.5f;

        const TriIdx4& tv = ci[tri];
        const F4& b = refBary[i];
        const F4 v0 = accum[tv.x], v1 = accum[tv.y], v2 = accum[tv.z];
        const F4 invMasses1 = F4{ v0.w, v1.w, v2.w, 0.0f };          // S5
        const F4 delta1 = F4{ v0.x * b.x + v1.x * b.y + v2.x * b.z,
                              v0.y * b.x + v1.y * b.y + v2.y * b.z,
                              v0.z * b.x + v1.z * b.y + v2.z * b.z,
                              v0.w * b.x + v1.w * b.y + v2.w * b.z };
        const V3 delta{ delta1.x - dp.x, delta1.y - dp.y, delta1.z - dp.z };
        const F4& np = refNormalPen[i];
        const V3 normal{ -np.x, -np.y, -np.z };                      // S4

        float afx = refForce[i * 2], afy = refForce[i * 2 + 1];
        const float normalDelta = delta.x * normal.x + delta.y * normal.y + delta.z * normal.z;
        V3 tanDir{ delta.x - normal.x * normalDelta,
                   delta.y - normal.y * normalDelta,
                   delta.z - normal.z * normalDelta };
        const float fricDelta = pxNormalize(tanDir);                 // S6
        const float error = np.w + normalDelta;
        const float cand = -error * refVelMult[i];
        const float deltaF = (-afx > cand) ? -afx : cand;            // S7
        if (deltaF == -afx && cand < -afx) ++clamped;
        afx += deltaF;
        const float frictionMax = afx * frictionCoefficient;
        const float requiredForce = fricDelta * refVelMult[i];
        const float sum = requiredForce + afy;
        const float deltaFr = ((sum < frictionMax) ? sum : frictionMax) - afy;
        afy += deltaFr;
        refForce[i * 2] = afx; refForce[i * 2 + 1] = afy;

        // cloth side
        const V3 deltaPos{ (normal.x * deltaF - tanDir.x * deltaFr) * BIAS,
                           (normal.y * deltaF - tanDir.y * deltaFr) * BIAS,
                           (normal.z * deltaF - tanDir.z * deltaFr) * BIAS };
        const uint32_t vi[3] = { tv.x, tv.y, tv.z };
        const float im[3] = { invMasses1.x, invMasses1.y, invMasses1.z };
        const float bw[3] = { b.x, b.y, b.z };
        for (int j = 0; j < 3; ++j)
            if (im[j] > 0.0f && std::fabs(bw[j]) > 1e-6f)             // S8
            {
                refDeltaPos[vi[j]].x += deltaPos.x * im[j] * bw[j];
                refDeltaPos[vi[j]].y += deltaPos.y * im[j] * bw[j];
                refDeltaPos[vi[j]].z += deltaPos.z * im[j] * bw[j];
                refDeltaPos[vi[j]].w += 1.0f;
            }

        // particle side: opposite signs AND an extra invMass0 factor
        const V3 deltaV{ (-normal.x * deltaF + tanDir.x * deltaFr) * invMass0 * BIAS,
                         (-normal.y * deltaF + tanDir.y * deltaFr) * invMass0 * BIAS,
                         (-normal.z * deltaF + tanDir.z * deltaFr) * invMass0 * BIAS };
        const float wflag = (deltaF != 0.0f || deltaFr != 0.0f) ? 1.0f : 0.0f;
        refDeltaP[i] = F4{ deltaV.x, deltaV.y, deltaV.z, wflag };
    }

    PXG_CHECK_MSG(skippedByMass == 1u,
                  "%u contacts were skipped for zero inverse mass, expected exactly 1 -- S3 is "
                  "not exercised otherwise", skippedByMass);
    PXG_CHECK_MSG(clamped > 0u,
                  "the normal-force clamp never bit; S7 is then untested and the rig's "
                  "penetrations need re-choosing");

    // Compare: the single-writer vertices bit-exactly, the contended triangle on .w.
    const auto gotDeltaPos = downloadBuffer<F4>(dev, bDeltaPos.h, uint32_t(accum.size()));
    const auto gotDeltaP = downloadBuffer<F4>(dev, bDeltaPOut.h, NB_CONTACTS);
    const auto gotForceA = downloadBuffer<float>(dev, bForcesA.h, NB_CONTACTS * 2u);
    const auto gotForceB = downloadBuffer<float>(dev, bForcesB.h, NB_CONTACTS * 2u);

    uint32_t mm = 0u;
    const uint32_t contendedBase = NB_SINGLE * 3u;    // the shared triangle's vertices
    for (uint32_t v = 0; v < uint32_t(accum.size()); ++v)
    {
        const bool contended = (v >= contendedBase);
        if (contended)
        {
            // Order-invariant: the count is an integer sum of 1.0f.
            if (f2u(gotDeltaPos[v].w) != f2u(refDeltaPos[v].w))
            { ++mm; std::printf("    contended vertex %u count %g, expected %g\n",
                                v, gotDeltaPos[v].w, refDeltaPos[v].w); }
            continue;
        }
        if (f2u(gotDeltaPos[v].x) != f2u(refDeltaPos[v].x) ||
            f2u(gotDeltaPos[v].y) != f2u(refDeltaPos[v].y) ||
            f2u(gotDeltaPos[v].z) != f2u(refDeltaPos[v].z) ||
            f2u(gotDeltaPos[v].w) != f2u(refDeltaPos[v].w))
        {
            if (mm == 0u)
                std::printf("    first cloth delta difference at vertex %u: gpu (%.9g %.9g %.9g | "
                            "%.9g) vs (%.9g %.9g %.9g | %.9g)\n", v,
                            gotDeltaPos[v].x, gotDeltaPos[v].y, gotDeltaPos[v].z, gotDeltaPos[v].w,
                            refDeltaPos[v].x, refDeltaPos[v].y, refDeltaPos[v].z, refDeltaPos[v].w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0u, "%u cloth-side vertex deltas differ from the mirror", mm);

    uint32_t mp = 0u;
    for (uint32_t i = 0; i < NB_CONTACTS; ++i)
        if (f2u(gotDeltaP[i].x) != f2u(refDeltaP[i].x) || f2u(gotDeltaP[i].y) != f2u(refDeltaP[i].y) ||
            f2u(gotDeltaP[i].z) != f2u(refDeltaP[i].z) || f2u(gotDeltaP[i].w) != f2u(refDeltaP[i].w))
        {
            if (mp == 0u)
                std::printf("    first particle delta difference at contact %u: gpu (%.9g %.9g "
                            "%.9g | %.9g) vs (%.9g %.9g %.9g | %.9g)\n", i,
                            gotDeltaP[i].x, gotDeltaP[i].y, gotDeltaP[i].z, gotDeltaP[i].w,
                            refDeltaP[i].x, refDeltaP[i].y, refDeltaP[i].z, refDeltaP[i].w);
            ++mp;
        }
    PXG_CHECK_MSG(mp == 0u, "%u particle-side deltas differ from the mirror", mp);

    // Both runs must leave the SAME applied forces: the force update is the shared
    // half of the two kernels, and a divergence there means the split moved.
    uint32_t mf = 0u;
    for (uint32_t i = 0; i < NB_CONTACTS * 2u; ++i)
    {
        if (f2u(gotForceA[i]) != f2u(refForce[i])) ++mf;
        if (f2u(gotForceA[i]) != f2u(gotForceB[i])) ++mf;
    }
    PXG_CHECK_MSG(mf == 0u,
                  "%u applied-force words disagree with the mirror or between the two kernels; "
                  "everything up to the force update is one shared body", mf);

    std::printf("    [ note ] %u contacts (%u skipped for zero inverse mass, %u normal-force "
                "clamps bit); %u single-writer vertices bit-exact, %u contended checked on count\n",
                NB_CONTACTS, skippedByMass, clamped, contendedBase,
                uint32_t(accum.size()) - contendedBase);

    for (PxgBufferHandle h : { bAccum.h, bCi.h, bMatIdx.h, bClothMats.h, bDeltaPos.h, bDesc.h,
                               bDeltaP.h, bPhases.h, bPhaseToMat.h, bPsMats.h, bPs.h, bInfos.h,
                               bConstraints.h, bForcesA.h, bForcesB.h, bNum.h, bDeltaPOut.h })
        dev->destroyBuffer(h);
}


// ---------------------------------------------------------------------------
// P9.32: cloth_solveOutputAttachmentClothClothDeltaVLaunch — the cloth-vs-cloth
// ATTACHMENT solve (FEMClothExternalSolve.cu:1493, launch PxgFEMClothCore.cpp:1915).
// ---------------------------------------------------------------------------
namespace p932att
{
struct ClothAttachSolvePush
{
    uint64_t clothes, attachments;
    uint32_t numAttachments, pad20;
};
static_assert(sizeof(ClothAttachSolvePush) == 24, "cloth attach solve blob must be 24 bytes");
static_assert(offsetof(ClothAttachSolvePush, attachments) == 8, "attachments @8");
static_assert(offsetof(ClothAttachSolvePush, numAttachments) == 16, "numAttachments @16");

// PxgFEMFEMAttachmentConstraint (PxgFEMCore.h:204) — the warp-strided block.  This TU
// cannot offsetof against the real struct (PxgFEMCore.h pulls the CUDA buffer stack), and
// the layout already exists twice — in the pxgSlangFemFemAttach module and in
// testSoftBodySim.cpp's p819att.  What this copy can do that those cannot is refuse to let
// the offsets be TYPED: they are ACCUMULATED from the field declarations
//
//   union { float4 low_high_limits[32];  float4 low_high_angles[32];       }
//   union { float4 axis_angle[32];       float4 attachmentBarycentric[32]; }
//   float4 barycentric0[32];   float4 barycentric1[32];
//   PxU64  elemId0[32];        PxU64  elemId1[32];
//   float  constraintOffset[32];
//
// and only then compared against the numbers the SHADER carries.  A field whose size or
// position is misread here therefore fails to agree with the shader instead of quietly
// agreeing with a second wrong copy.
static const size_t FAC_LANES = 32;
static const size_t FAC_F4_ARR  = 16 * FAC_LANES;   // float4[32]
static const size_t FAC_U64_ARR =  8 * FAC_LANES;   // PxU64[32]
static const size_t FAC_F32_ARR =  4 * FAC_LANES;   // float[32]
static const size_t FAC_LOW_HIGH_ANGLES = 0;
static const size_t FAC_ATT_BARYCENTRIC = FAC_LOW_HIGH_ANGLES + FAC_F4_ARR;
static const size_t FAC_BARYCENTRIC0    = FAC_ATT_BARYCENTRIC + FAC_F4_ARR;
static const size_t FAC_BARYCENTRIC1    = FAC_BARYCENTRIC0 + FAC_F4_ARR;
static const size_t FAC_ELEMID0         = FAC_BARYCENTRIC1 + FAC_F4_ARR;
static const size_t FAC_ELEMID1         = FAC_ELEMID0 + FAC_U64_ARR;
static const size_t FAC_CONSTR_OFFSET   = FAC_ELEMID1 + FAC_U64_ARR;
static const size_t FAC_STRIDE          = FAC_CONSTR_OFFSET + FAC_F32_ARR;
// The literals in pxgSlangFemFemAttach.slang, restated so a change to either side breaks.
static_assert(FAC_ATT_BARYCENTRIC == 512, "attachmentBarycentric @512");
static_assert(FAC_BARYCENTRIC0 == 1024, "barycentric0 @1024");
static_assert(FAC_BARYCENTRIC1 == 1536, "barycentric1 @1536");
static_assert(FAC_ELEMID0 == 2048, "elemId0 @2048");
static_assert(FAC_ELEMID1 == 2304, "elemId1 @2304");
static_assert(FAC_CONSTR_OFFSET == 2560, "constraintOffset @2560");
static_assert(FAC_STRIDE == 2688, "the block is 2688 bytes");

// PxgSimulationCoreDesc.h:41 / PxDeformableSurface.h:56 — the encoding BOTH sides of the
// kernel decode, one through the accessors and one open-coded (trap A4).
static const uint32_t CLOTH_ID_SHIFT = 20u;
static const uint32_t CLOTH_ELEM_MASK = 0x000fffffu;
static uint64_t encodeCloth(uint32_t clothId, uint32_t elem)
{
    return uint64_t((clothId << CLOTH_ID_SHIFT) | elem);
}

// PxgSoftBodyKernelGridDim/BlockDim::SB_UPDATEROTATION.
static const uint32_t FCA_GRID = 1024u, FCA_BLOCK = 256u;
} // namespace p932att

PXG_TEST(femClothSimClothClothAttachment)
{
    using namespace p932att;
    PxgSlangComputeDevice* dev = testDevice();

    // Three clothes at ids 1, 2 and 5 — never 0.  A port that forgets the >> 20 reads
    // descriptor 0, which stays all-zero here, so the mistake is a null dereference rather
    // than a plausible number.
    const uint32_t CLOTH_A = 1u, CLOTH_B = 2u, CLOTH_C = 5u;

    // 70 attachments = three warp blocks of 32/32/6.  Fewer than 33 and trap A2 is
    // invisible: `attachments[workIndex]` and `attachments[workIndex/32]` agree exactly on
    // the first block.
    const uint32_t NB_ATT = 70u;
    const uint32_t NB_BLOCKS = (NB_ATT + 31u) / 32u;

    // Attachment 9 has BOTH sides fully pinned -> the invMass guard fires (trap A6).
    // Attachments 44 and 45 have exactly ONE side pinned, which is what separates the
    // guard's `&&` from an `||`: with `&&` they are solved against the free side's mass,
    // with `||` they would be dropped.  The first sweep had neither, and the `||` mutant
    // survived — a survivor that says nothing about the code and everything about which
    // states the rig reached.
    // Attachments 33 and 58 are SELF-attachments inside cloth C and share vertices, so
    // their scatters contend (trap A10).
    const uint32_t PINNED_ATT = 9u, PINNED_A_ONLY = 44u, PINNED_B_ONLY = 45u;
    auto isSelf = [&](uint32_t a) { return a == 33u || a == 58u; };

    // Trap A7 needs a corner whose weight is zero in EVERY slot, for BOTH reasons — the
    // first sweep put its zeros in corners y and z only, so a mutant that guards corner x
    // was a no-op against this rig and survived.  Six explicit attachments now, spread over
    // two warp blocks so a slot cannot be reached by accident.
    auto zeroMassCorner = [](uint32_t a) -> int {   // -1 = none
        if (a == 5u || a == 40u) return 0;
        if (a == 3u || a == 41u) return 1;
        if (a == 7u || a == 42u) return 2;
        return -1;
    };
    auto zeroBaryCorner = [](uint32_t a) -> int {
        if (a == 20u) return 0;
        if (a == 21u) return 1;
        if (a == 12u) return 2;
        return -1;
    };

    // ---- the three clothes ----------------------------------------------------------
    // Every attachment owns its own triangle on each side, except the self ones, which
    // share cloth C's vertex 0..5 pool.  Positions and inverse masses are all distinct so
    // the interpolated inverse mass (trap A5) equals no single vertex's.
    std::vector<F4> vA, vB, vC;
    std::vector<TriIdx4> tA, tB, tC;
    auto pushTri = [](std::vector<F4>& v, std::vector<TriIdx4>& t, float ox, float oy,
                      const float m[3]) {
        const uint32_t b = uint32_t(v.size());
        v.push_back(F4{ ox,         oy,         0.10f, m[0] });
        v.push_back(F4{ ox + 1.3f,  oy + 0.2f,  0.30f, m[1] });
        v.push_back(F4{ ox + 0.4f,  oy + 1.7f, -0.20f, m[2] });
        t.push_back(TriIdx4{ b, b + 1u, b + 2u, 0u });
    };
    for (uint32_t a = 0; a < NB_ATT; ++a)
    {
        const bool pinA = (a == PINNED_ATT || a == PINNED_A_ONLY);
        const bool pinB = (a == PINNED_ATT || a == PINNED_B_ONLY);
        float mA[3] = { 0.40f + 0.011f * float(a),
                        0.62f + 0.007f * float(a),
                        0.85f - 0.006f * float(a % 30u) };
        float mB[3] = { 0.55f - 0.004f * float(a % 25u),
                        0.31f + 0.009f * float(a),
                        0.73f + 0.005f * float(a % 17u) };
        const int zm = zeroMassCorner(a);
        if (zm >= 0) mA[zm] = 0.0f;
        for (int c = 0; c < 3; ++c) { if (pinA) mA[c] = 0.0f; if (pinB) mB[c] = 0.0f; }
        if (isSelf(a))
        {
            // Both sides land in cloth C on triangles that SHARE vertices.
            continue;
        }
        pushTri(vA, tA, 2.0f * float(a), 0.0f, mA);
        pushTri(vB, tB, 2.0f * float(a), 5.0f, mB);
    }
    // Cloth C: six vertices, three triangles over them, so the two self-attachments write
    // the same vertices twice from opposite sides.
    {
        const float mC[6] = { 0.37f, 0.0f, 0.91f, 0.48f, 0.66f, 0.24f };
        for (uint32_t i = 0; i < 6u; ++i)
            vC.push_back(F4{ 0.7f * float(i), 1.1f * float(i % 3u), -0.4f * float(i % 2u), mC[i] });
        tC.push_back(TriIdx4{ 0u, 1u, 2u, 0u });
        tC.push_back(TriIdx4{ 2u, 3u, 4u, 0u });
        tC.push_back(TriIdx4{ 4u, 5u, 0u, 0u });
    }

    DevBuf bVA = makeDev(dev, vA);
    if (!bVA.addr) { std::printf("  NOTE: no BDA -> skip\n"); dev->destroyBuffer(bVA.h); return; }
    DevBuf bVB = makeDev(dev, vB), bVC = makeDev(dev, vC);
    DevBuf bTA = makeDev(dev, tA), bTB = makeDev(dev, tB), bTC = makeDev(dev, tC);
    std::vector<F4> dA(vA.size(), F4{ 0, 0, 0, 0 }), dB(vB.size(), F4{ 0, 0, 0, 0 }),
                    dC(vC.size(), F4{ 0, 0, 0, 0 });
    DevBuf bDA = makeDev(dev, dA), bDB = makeDev(dev, dB), bDC = makeDev(dev, dC);

    std::vector<PxgFEMCloth> descArr(CLOTH_C + 1u, makeZeroDesc());
    auto wire = [](PxgFEMCloth& c, DevBuf& pos, DevBuf& tri, DevBuf& delta,
                   size_t nbVerts, size_t nbTris) {
        c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(pos.addr));
        c.mTriangleVertexIndices =
            reinterpret_cast<decltype(c.mTriangleVertexIndices)>(size_t(tri.addr));
        c.mDeltaPos = reinterpret_cast<decltype(c.mDeltaPos)>(size_t(delta.addr));
        c.mNbVerts = uint32_t(nbVerts);
        c.mNbTriangles = uint32_t(nbTris);
    };
    wire(descArr[CLOTH_A], bVA, bTA, bDA, vA.size(), tA.size());
    wire(descArr[CLOTH_B], bVB, bTB, bDB, vB.size(), tB.size());
    wire(descArr[CLOTH_C], bVC, bTC, bDC, vC.size(), tC.size());
    DevBuf bDesc = makeDev(dev, descArr);

    // ---- the constraint block ---------------------------------------------------------
    // Seeded with a recognisable pattern so a read that misses its slot lands on garbage
    // rather than on a zero that could pass for an answer.
    std::vector<uint32_t> con(NB_BLOCKS * FAC_STRIDE / 4, 0xCDCDCDCDu);
    auto putF4 = [&](size_t byteOff, F4 v) {
        const size_t w = byteOff / 4;
        con[w] = f2u(v.x); con[w + 1] = f2u(v.y); con[w + 2] = f2u(v.z); con[w + 3] = f2u(v.w);
    };
    struct Side { uint32_t clothId, elem; F4 bary; };
    std::vector<Side> s0(NB_ATT), s1(NB_ATT);
    uint32_t triA = 0u, triB = 0u;
    for (uint32_t a = 0; a < NB_ATT; ++a)
    {
        // Barycentrics that sum to one but are never equal between the two sides — with
        // equal weights the swap of barycentric0/barycentric1 is invisible.  Attachment 12
        // zeroes corner 2 on side 0: trap A7's second half.
        F4 b0{ 0.5f - 0.004f * float(a % 20u), 0.2f + 0.003f * float(a % 13u), 0.0f, 0.0f };
        b0.z = 1.0f - b0.x - b0.y;
        switch (zeroBaryCorner(a))
        {
        case 0:  b0 = F4{ 0.0f, 0.45f, 0.55f, 0.0f }; break;
        case 1:  b0 = F4{ 0.45f, 0.0f, 0.55f, 0.0f }; break;
        case 2:  b0 = F4{ 0.45f, 0.55f, 0.0f, 0.0f }; break;
        default: break;
        }
        F4 b1{ 0.15f + 0.005f * float(a % 11u), 0.55f - 0.006f * float(a % 9u), 0.0f, 0.0f };
        b1.z = 1.0f - b1.x - b1.y;

        if (isSelf(a))
        {
            s0[a] = Side{ CLOTH_C, (a == 33u) ? 0u : 1u, b0 };
            s1[a] = Side{ CLOTH_C, (a == 33u) ? 1u : 2u, b1 };
        }
        else
        {
            s0[a] = Side{ CLOTH_A, triA++, b0 };
            s1[a] = Side{ CLOTH_B, triB++, b1 };
        }

        const size_t base = size_t(a / 32u) * FAC_STRIDE, off = a & 31u;
        // Trap A3: the slots are PxU64 and the kernel takes the LOW word.  The upper word
        // carries a non-zero pattern, so a port that reads a uint2, or that steps by 4
        // bytes, produces a cloth id far outside the descriptor array.
        const size_t w0 = (base + FAC_ELEMID0 + off * 8) / 4;
        con[w0] = uint32_t(encodeCloth(s0[a].clothId, s0[a].elem));
        con[w0 + 1] = 0xA5A50000u | a;
        const size_t w1 = (base + FAC_ELEMID1 + off * 8) / 4;
        con[w1] = uint32_t(encodeCloth(s1[a].clothId, s1[a].elem));
        con[w1 + 1] = 0x5A5A0000u | a;
        putF4(base + FAC_BARYCENTRIC0 + off * 16, s0[a].bary);
        putF4(base + FAC_BARYCENTRIC1 + off * 16, s1[a].bary);
        // The three fields this kernel must NOT read (the cone-limit trio) keep the 0xCD
        // fill, so a body copied from the softbody solve reads NaNs rather than zeros.
    }
    DevBuf bCon = makeDev(dev, con);

    ClothAttachSolvePush push = {};
    push.clothes = bDesc.addr;
    push.attachments = bCon.addr;
    push.numAttachments = NB_ATT;
    Kernel k("femClothAttachBda.spv", "cloth_solveOutputAttachmentClothClothDeltaVLaunch",
             {}, sizeof(ClothAttachSolvePush));
    k.dispatch(FCA_GRID, {}, push, 1u, 1u);
    dev->flush();

    // ---- the mirror, transcribed from the .cu ------------------------------------------
    std::vector<F4> rA(vA.size(), F4{ 0, 0, 0, 0 }), rB(vB.size(), F4{ 0, 0, 0, 0 }),
                    rC(vC.size(), F4{ 0, 0, 0, 0 });
    auto vertsOf = [&](uint32_t clothId) -> const std::vector<F4>& {
        return clothId == CLOTH_A ? vA : (clothId == CLOTH_B ? vB : vC);
    };
    auto trisOf = [&](uint32_t clothId) -> const std::vector<TriIdx4>& {
        return clothId == CLOTH_A ? tA : (clothId == CLOTH_B ? tB : tC);
    };
    auto refOf = [&](uint32_t clothId) -> std::vector<F4>& {
        return clothId == CLOTH_A ? rA : (clothId == CLOTH_B ? rB : rC);
    };
    uint32_t skippedByMass = 0u, contendedWrites = 0u;
    for (uint32_t a = 0; a < NB_ATT; ++a)
    {
        const TriIdx4& t0 = trisOf(s0[a].clothId)[s0[a].elem];
        const TriIdx4& t1 = trisOf(s1[a].clothId)[s1[a].elem];
        const std::vector<F4>& v0 = vertsOf(s0[a].clothId);
        const std::vector<F4>& v1 = vertsOf(s1[a].clothId);
        const F4 p00 = v0[t0.x], p01 = v0[t0.y], p02 = v0[t0.z];
        const F4 p10 = v1[t1.x], p11 = v1[t1.y], p12 = v1[t1.z];
        const F4& b0 = s0[a].bary; const F4& b1 = s1[a].bary;
        // The float4 blend — .w included, and .w is what divides (trap A5).
        const F4 pos0{ p00.x * b0.x + p01.x * b0.y + p02.x * b0.z,
                       p00.y * b0.x + p01.y * b0.y + p02.y * b0.z,
                       p00.z * b0.x + p01.z * b0.y + p02.z * b0.z,
                       p00.w * b0.x + p01.w * b0.y + p02.w * b0.z };
        const F4 pos1{ p10.x * b1.x + p11.x * b1.y + p12.x * b1.z,
                       p10.y * b1.x + p11.y * b1.y + p12.y * b1.z,
                       p10.z * b1.x + p11.z * b1.y + p12.z * b1.z,
                       p10.w * b1.x + p11.w * b1.y + p12.w * b1.z };
        if (pos1.w == 0.0f && pos0.w == 0.0f) { ++skippedByMass; continue; }   // trap A6

        // The shape of the arithmetic is the observable, not just its value -- and P9.36
        // corrected which shape.  This comment used to read:
        //
        //     "the CUDA writes `diff * coefficient / (invMass0 + invMass1)`, so the scale
        //      is a multiplication FOLLOWED BY a division, not a multiplication by a
        //      reciprocal."
        //
        // It is the reverse.  `diff` is a CUDA float4 (FEMClothExternalSolve.cu:1553) and
        // `float4 operator/(float4 a, float s)` (cutil_math.h:485) is
        // `float inv = 1.0f/s; return a * inv;` -- a multiplication BY A RECIPROCAL, which
        // is precisely what the old text ruled out.  The trap was pointed the wrong way and
        // pinned the port's mistake in place.
        const float coefficient = 0.5f;                                        // trap A9
        const float denom = pos0.w + pos1.w;
        const float invDenom = 1.0f / denom;
        const float dix = (pos1.x - pos0.x) * coefficient * invDenom;
        const float diy = (pos1.y - pos0.y) * coefficient * invDenom;
        const float diz = (pos1.z - pos0.z) * coefficient * invDenom;

        // Trap A7: NO skip.  Trap A8: cloth0 gets +, cloth1 gets -.
        auto scatter = [&](uint32_t clothId, const TriIdx4& t, const F4& b,
                           float sx, float sy, float sz) {
            std::vector<F4>& r = refOf(clothId);
            const std::vector<F4>& v = vertsOf(clothId);
            const uint32_t iv[3] = { t.x, t.y, t.z };
            const float bw[3] = { b.x, b.y, b.z };
            for (int c = 0; c < 3; ++c)
            {
                const float im = v[iv[c]].w;                                   // trap A5
                r[iv[c]].x += sx * bw[c] * im;
                r[iv[c]].y += sy * bw[c] * im;
                r[iv[c]].z += sz * bw[c] * im;
                r[iv[c]].w += 1.0f;
            }
        };
        scatter(s0[a].clothId, t0, b0, dix, diy, diz);
        scatter(s1[a].clothId, t1, b1, -dix, -diy, -diz);
        if (isSelf(a)) ++contendedWrites;
    }

    const auto gA = downloadBuffer<F4>(dev, bDA.h, uint32_t(dA.size()));
    const auto gB = downloadBuffer<F4>(dev, bDB.h, uint32_t(dB.size()));
    const auto gC = downloadBuffer<F4>(dev, bDC.h, uint32_t(dC.size()));

    // Cloth A and B: every vertex has exactly ONE writer, so the comparison is bitwise.
    uint32_t mm = 0u;
    auto cmpExact = [&](const std::vector<F4>& got, const std::vector<F4>& ref, const char* tag) {
        for (uint32_t v = 0; v < uint32_t(ref.size()); ++v)
            if (f2u(got[v].x) != f2u(ref[v].x) || f2u(got[v].y) != f2u(ref[v].y) ||
                f2u(got[v].z) != f2u(ref[v].z) || f2u(got[v].w) != f2u(ref[v].w))
            {
                if (mm == 0u)
                    std::printf("    first %s difference at vertex %u: gpu (%.9g %.9g %.9g | %.9g)"
                                " vs (%.9g %.9g %.9g | %.9g)\n", tag, v,
                                got[v].x, got[v].y, got[v].z, got[v].w,
                                ref[v].x, ref[v].y, ref[v].z, ref[v].w);
                ++mm;
            }
    };
    cmpExact(gA, rA, "cloth A");
    cmpExact(gB, rB, "cloth B");
    PXG_CHECK_MSG(mm == 0u, "%u single-writer vertex deltas differ from the mirror", mm);

    // Cloth C carries the two SELF-attachments, whose six scatters overlap; the atomics make
    // the order free, so the totals are compared with a tolerance and the counters exactly.
    uint32_t mc = 0u;
    for (uint32_t v = 0; v < uint32_t(rC.size()); ++v)
    {
        if (f2u(gC[v].w) != f2u(rC[v].w))
        { ++mc; std::printf("    cloth C vertex %u writer count %g, expected %g\n",
                            v, gC[v].w, rC[v].w); }
        if (std::fabs(gC[v].x - rC[v].x) > 1e-6f || std::fabs(gC[v].y - rC[v].y) > 1e-6f ||
            std::fabs(gC[v].z - rC[v].z) > 1e-6f)
        { ++mc; std::printf("    cloth C vertex %u delta (%.9g %.9g %.9g) expected (%.9g %.9g %.9g)\n",
                            v, gC[v].x, gC[v].y, gC[v].z, rC[v].x, rC[v].y, rC[v].z); }
    }
    PXG_CHECK_MSG(mc == 0u, "%u contended cloth-C vertices disagree with the mirror", mc);

    // The structural facts the mirror alone would not pin, because the mirror shares the
    // port's reading of them.
    PXG_CHECK_MSG(skippedByMass == 1u,
                  "exactly one attachment must be skipped for zero inverse mass, not %u",
                  skippedByMass);
    const TriIdx4& tp = tA[s0[PINNED_ATT].elem];
    PXG_CHECK_MSG(gA[tp.x].w == 0.0f && gA[tp.y].w == 0.0f && gA[tp.z].w == 0.0f,
                  "the fully pinned attachment must leave its triangle untouched");
    // Trap A6's `&&`: a HALF-pinned attachment is solved, not skipped.  Both of its
    // triangles take writer counts, and the free side actually moves.
    for (uint32_t a : { PINNED_A_ONLY, PINNED_B_ONLY })
    {
        const TriIdx4& ha = tA[s0[a].elem];
        const TriIdx4& hb = tB[s1[a].elem];
        PXG_CHECK_MSG(gA[ha.x].w == 1.0f && gB[hb.x].w == 1.0f,
                      "attachment %u has exactly one pinned side and must still be solved "
                      "(counts %g and %g)", a, gA[ha.x].w, gB[hb.x].w);
    }
    // Trap A7 in the affirmative: a corner whose weight is zero — for EITHER reason, in
    // EACH of the three slots — still takes a writer count and a zero delta.
    uint32_t zeroWeightChecked = 0u;
    for (uint32_t a = 0; a < NB_ATT; ++a)
    {
        if (isSelf(a) || a == PINNED_ATT || a == PINNED_A_ONLY) continue;
        const int slots[2] = { zeroMassCorner(a), zeroBaryCorner(a) };
        for (int which = 0; which < 2; ++which)
        {
            const int c = slots[which];
            if (c < 0) continue;
            const TriIdx4& t = tA[s0[a].elem];
            const uint32_t v = (c == 0) ? t.x : (c == 1 ? t.y : t.z);
            PXG_CHECK_MSG(gA[v].w == 1.0f && gA[v].x == 0.0f && gA[v].y == 0.0f &&
                          gA[v].z == 0.0f,
                          "attachment %u corner %d carries zero weight (%s) and must still "
                          "take a writer count and a zero delta (got count %g, xyz %g %g %g)",
                          a, c, which == 0 ? "inverse mass" : "barycentric",
                          gA[v].w, gA[v].x, gA[v].y, gA[v].z);
            ++zeroWeightChecked;
        }
    }
    PXG_CHECK_MSG(zeroWeightChecked == 9u,
                  "the rig must place a zero weight in each of the three corner slots for "
                  "both reasons; %u were found", zeroWeightChecked);
    // Trap A8: the two sides moved in OPPOSITE directions.  Attachment 0's side-0 anchor is
    // below its side-1 anchor in y, so cloth A's corner moves +y and cloth B's -y.
    const TriIdx4& t0a = tA[s0[0u].elem];
    const TriIdx4& t0b = tB[s1[0u].elem];
    PXG_CHECK_MSG(gA[t0a.x].y > 0.0f && gB[t0b.x].y < 0.0f,
                  "cloth0 must move toward cloth1 and cloth1 toward cloth0 (got %g and %g)",
                  gA[t0a.x].y, gB[t0b.x].y);

    std::printf("    [ note ] %u attachments over %u warp blocks, %u skipped for zero inverse "
                "mass, 2 half-pinned, %u zero-weight corners checked, %u self-attachments "
                "contending in cloth C; %zu single-writer vertices bit-exact\n",
                NB_ATT, NB_BLOCKS, skippedByMass, zeroWeightChecked, contendedWrites,
                vA.size() + vB.size());

    for (PxgBufferHandle h : { bVA.h, bVB.h, bVC.h, bTA.h, bTB.h, bTC.h, bDA.h, bDB.h, bDC.h,
                               bDesc.h, bCon.h })
        dev->destroyBuffer(h);
}


// ---------------------------------------------------------------------------
// P9.32 coverage rig.  Two properties of the work-index formula are invisible below
// gridDim*blockDim = 1024*256 = 262144 attachments, because `nbIterationsPerBlock` is 1
// there and the formula collapses to a flat index:
//   - dropping the `nbIterationsPerBlock *` factor makes the per-block ranges OVERLAP,
//     so the tail is never visited and everything else is visited twice (trap A1);
//   - the zero-mass guard being a `return` rather than a `continue` abandons every LATER
//     iteration of a thread that meets one fully pinned attachment (trap A6).
// Both need nbIterationsPerBlock >= 2, so this rig runs 262400 attachments: 1025 blocks
// required over a 1024-wide grid -> exactly two iterations per thread.
//
// Every attachment uses the SAME triangle on both sides with the SAME barycentric, so the
// position delta is exactly zero and the only observable is the .w WRITER COUNT — which is
// precisely what a coverage question asks.  Attachments 0..255 are fully pinned, which is
// block 0's first iteration; their threads' second iteration is attachments 256..511.
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimClothClothAttachmentCoverage)
{
    using namespace p932att;
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t CLOTH_ID = 3u;
    const uint32_t NB_ATT = 262400u;                 // 1025 blocks over a 1024 grid
    const uint32_t NB_PINNED = FCA_BLOCK;            // the whole of block 0's iteration 0
    const uint32_t NB_BLOCKS = NB_ATT / 32u;
    // Derived, not assumed: the dispatch this rig needs must really take two iterations.
    const uint32_t nbBlocksRequired = (NB_ATT + FCA_BLOCK - 1u) / FCA_BLOCK;
    const uint32_t nbIterationsPerBlock = (nbBlocksRequired + FCA_GRID - 1u) / FCA_GRID;
    PXG_CHECK_MSG(nbIterationsPerBlock == 2u,
                  "the rig must reach two iterations per thread, got %u", nbIterationsPerBlock);

    std::vector<F4> verts(size_t(NB_ATT) * 3u);
    std::vector<TriIdx4> tris(NB_ATT);
    for (uint32_t a = 0; a < NB_ATT; ++a)
    {
        const uint32_t b = a * 3u;
        const float im = (a < NB_PINNED) ? 0.0f : 0.5f;
        verts[b + 0] = F4{ float(a), 0.0f, 0.0f, im };
        verts[b + 1] = F4{ float(a) + 1.0f, 0.0f, 0.0f, im };
        verts[b + 2] = F4{ float(a), 1.0f, 0.0f, im };
        tris[a] = TriIdx4{ b, b + 1u, b + 2u, 0u };
    }
    DevBuf bV = makeDev(dev, verts);
    if (!bV.addr) { std::printf("  NOTE: no BDA -> skip\n"); dev->destroyBuffer(bV.h); return; }
    DevBuf bT = makeDev(dev, tris);
    std::vector<F4> delta(verts.size(), F4{ 0, 0, 0, 0 });
    DevBuf bD = makeDev(dev, delta);

    std::vector<PxgFEMCloth> descArr(CLOTH_ID + 1u, makeZeroDesc());
    descArr[CLOTH_ID].mPosition_InvMass =
        reinterpret_cast<decltype(descArr[0].mPosition_InvMass)>(size_t(bV.addr));
    descArr[CLOTH_ID].mTriangleVertexIndices =
        reinterpret_cast<decltype(descArr[0].mTriangleVertexIndices)>(size_t(bT.addr));
    descArr[CLOTH_ID].mDeltaPos =
        reinterpret_cast<decltype(descArr[0].mDeltaPos)>(size_t(bD.addr));
    descArr[CLOTH_ID].mNbVerts = uint32_t(verts.size());
    descArr[CLOTH_ID].mNbTriangles = NB_ATT;
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<uint32_t> con(size_t(NB_BLOCKS) * FAC_STRIDE / 4, 0u);
    const F4 BARY{ 0.25f, 0.35f, 0.40f, 0.0f };
    for (uint32_t a = 0; a < NB_ATT; ++a)
    {
        const size_t base = size_t(a / 32u) * FAC_STRIDE, off = a & 31u;
        const uint32_t code = uint32_t(encodeCloth(CLOTH_ID, a));
        con[(base + FAC_ELEMID0 + off * 8) / 4] = code;
        con[(base + FAC_ELEMID1 + off * 8) / 4] = code;
        for (int c = 0; c < 3; ++c)
        {
            const float w = (c == 0) ? BARY.x : (c == 1 ? BARY.y : BARY.z);
            con[(base + FAC_BARYCENTRIC0 + off * 16) / 4 + c] = f2u(w);
            con[(base + FAC_BARYCENTRIC1 + off * 16) / 4 + c] = f2u(w);
        }
    }
    DevBuf bCon = makeDev(dev, con);

    ClothAttachSolvePush push = {};
    push.clothes = bDesc.addr;
    push.attachments = bCon.addr;
    push.numAttachments = NB_ATT;
    Kernel k("femClothAttachBda.spv", "cloth_solveOutputAttachmentClothClothDeltaVLaunch",
             {}, sizeof(ClothAttachSolvePush));
    k.dispatch(FCA_GRID, {}, push, 1u, 1u);
    dev->flush();

    const auto got = downloadBuffer<F4>(dev, bD.h, uint32_t(verts.size()));

    // Both sides scatter into the SAME triangle, so a visited attachment leaves 2.0 on each
    // of its three corners; a skipped or unvisited one leaves 0.0.
    uint32_t badCount = 0u, unvisited = 0u, firstUnvisited = NB_ATT, nonZeroDelta = 0u;
    for (uint32_t a = 0; a < NB_ATT; ++a)
    {
        const float expect = (a < NB_PINNED) ? 0.0f : 2.0f;
        for (uint32_t c = 0; c < 3u; ++c)
        {
            const F4& g = got[size_t(a) * 3u + c];
            if (g.w != expect)
            {
                if (a >= NB_PINNED && g.w == 0.0f)
                { if (firstUnvisited == NB_ATT) firstUnvisited = a; ++unvisited; }
                else ++badCount;
            }
            if (g.x != 0.0f || g.y != 0.0f || g.z != 0.0f) ++nonZeroDelta;
        }
    }
    PXG_CHECK_MSG(unvisited == 0u,
                  "%u corners were never written; the first belongs to attachment %u "
                  "(the formula's first uncovered index would be %u)",
                  unvisited, firstUnvisited, FCA_GRID * FCA_BLOCK);
    PXG_CHECK_MSG(badCount == 0u,
                  "%u corners carry a writer count that is neither the expected 2 nor 0 — a "
                  "count above 2 means a work index was visited more than once", badCount);
    PXG_CHECK_MSG(nonZeroDelta == 0u,
                  "%u corners moved, but both sides of every attachment interpolate the same "
                  "anchor, so every delta must be exactly zero", nonZeroDelta);

    std::printf("    [ note ] %u attachments, %u iterations per thread; %u pinned (block 0's "
                "first iteration, whose SECOND iteration is attachments %u..%u), all %zu "
                "remaining corners written exactly twice\n",
                NB_ATT, nbIterationsPerBlock, NB_PINNED, NB_PINNED, NB_PINNED * 2u - 1u,
                (size_t(NB_ATT) - NB_PINNED) * 3u);

    for (PxgBufferHandle h : { bV.h, bT.h, bD.h, bDesc.h, bCon.h })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P9.75 -- convexCoreClothmeshNphase_Kernel32 on the SHIPPED engine .spv
// ===========================================================================
//
// WHY THIS EXISTS, AND WHAT IT IS ANSWERING.
//
// P9.74 mapped the kernel and swept it against PXG_SCENE=clothcvxcore.  After the
// scene's verdict was sharpened the sweep killed 9 of 25 mutants, and the sixteen
// survivors were classified with source evidence rather than filed as noise:
//
//   A  provably equivalent IN THAT SCENE (7)   -- nothing to fix, but nothing proved
//   B  real change, below the scene's resolution (8)
//   C  dead consumer (0)                        -- every field is read by a MAPPED kernel
//   D  observationally identical by ALLOCATOR ACCIDENT, not construction (1)
//
// Classes A, B and D are all "the scene cannot see this", for three different
// reasons, and no harder version of that scene reaches them: the scene has ONE
// cloth (so clothId is 0 and the shift constant is unobservable), NO attachments
// (so nbFilterPairs is 0 and `&&` cannot be told from `||`), ONE rigid (so the
// 64-bit sort key is constant), and a resting contact (so half-penetration and the
// ERP scaling are both fractions of a millimetre).
//
// A fixture is not a smaller scene.  It is the observer that CHOOSES those values,
// and that is the whole argument for writing one here -- the same argument P9.73
// made when P9.72's softbody scene could not see a tetrahedron's fourth vertex.
//
// WHAT IT DRIVES.  The SHIPPED cvxCoreClothNpBda.spv, not a copy.  P9.73 recorded
// why that distinction is load-bearing: Phase-1 fixtures are self-contained by
// convention, carrying their own copies of the shared functions, so mutants in the
// shared modules survive them.  This one synthesises the fourteen device-side
// arguments and dispatches the real entry point, exactly as the
// cloth_planeVertContactGenLaunch fixture above does.
//
// THE GEOMETRY, AND WHY IT IS THE GEOMETRY.
//
//   The convex core is an analytic BOX of FULL extents (2,2,2) at the origin, so it
//   spans [-1,1]^3 and its top face is the plane y = +1.  Analytic means no cooked
//   hull and no device-side hull data -- the core's parameters ride in the
//   PxMeshScale bytes (convexCoreCollision.cu:114-169), which is the one arm of
//   makeConvexShape that needs nothing allocated.
//
//   The cloth's transform cache entry is IDENTITY, because a deformable surface has
//   no rigid pose and its vertices are already world-space.  That makes
//   transform0in1 == the box's pose and keeps the whole test in one frame.
//
//   Two triangles at y = 0.9, i.e. 0.1 INSIDE the top face.  0.1 is chosen to be
//   far larger than the margins (the triangle margin is 0.001 * the fattened box
//   dimension, about 0.002) and far smaller than every other way out of the box,
//   so EPA's minimum-penetration axis is unambiguously +/-Y.
//
// WHAT MAKES no-half-penetration CHECKABLE -- the fact that had to be read, not
// assumed.  faceClipQ's points are ALREADY midway between the two surfaces:
//
//     polyDists[i] = float4((pr0 + pr1) * 0.5f, dot3Q(axis, pr0 - pr1));
//                                    -- pxgSlangCvxCoreCvx.slang, getContact
//
// where pr0 and pr1 are the point projected onto face0's and face1's planes along
// the axis.  So the kernel's `point = patch[i].xyz + normal * patch[i].w * 0.5f`
// moves that midpoint by another half separation and lands it ON ONE OF THE TWO
// PLANES -- here y == 0.9 (the triangle) or y == 1.0 (the box face).  Dropping the
// shift leaves it at y == 0.95, which is on neither.  That is an exact, analytic
// discrimination, and it exists only because both surfaces here are planes.
// ===========================================================================

// Push blob, byte-identical to cvxCoreClothNpBda.slang's CvxCoreClothParams and to
// the kCvxCoreCloth ABI row { 8,8,8,8,8,8,8,8,8,8,4,4,8,80 }.
struct CvxCoreClothPush
{
    uint64_t stackPtr, midphasePairsNum, cmInputs, shapes, femClothes;
    uint64_t shapeToRigidRemapTable, transformCache, contactDistance, restDistance, filterPairs;
    uint32_t nbFilterPairs, stackSizeBytes;
    uint64_t stackSizeNeededOnDevice;
    uint64_t w_outPoint, w_outNormalPen, w_outBarycentric, w_outContactInfo, w_totalContactCount;
    uint64_t w_contactByX, w_tempContactByX, w_contactIndexSortedByX, w_contactSortedByX;
    uint32_t w_maxNumContacts, pad172;
};
static_assert(sizeof(CvxCoreClothPush) == 176, "convex-core cloth push blob must be 176 bytes");
static_assert(offsetof(CvxCoreClothPush, nbFilterPairs) == 80, "nbFilterPairs @80");
static_assert(offsetof(CvxCoreClothPush, stackSizeNeededOnDevice) == 88, "stackSizeNeeded @88");
static_assert(offsetof(CvxCoreClothPush, w_outPoint) == 96, "writer @96");
static_assert(offsetof(CvxCoreClothPush, w_maxNumContacts) == 168, "maxNumContacts @168");

// PxComputeBarycentric, the TRIANGLE overload (PxMathUtils.h:390-409), transcribed
// operation for operation so the comparison can be BIT-EXACT.  This TU is built
// -ffp-contract=off, which is what the shaders' NoContraction stamp matches.
//
// NOT the corpus's `computeBarycentric` (fcPrim2.slang:309, fcStaticTriCloth.slang:215):
// those are femClothPrimitives.cu's function, the same Cramer arithmetic plus a
// [-1e-4, 1+1e-4] acceptance window whose bool gates a contact.  The kernel under
// test has no such gate, and transcribing the windowed one would import a decision
// that does not belong on this path.
static F4 cpuTriBarycentric(V3c a, V3c b, V3c c, V3c p)
{
    const V3c v0{ b.x - a.x, b.y - a.y, b.z - a.z };
    const V3c v1{ c.x - a.x, c.y - a.y, c.z - a.z };
    const V3c v2{ p.x - a.x, p.y - a.y, p.z - a.z };
    auto d3 = [](const V3c& u, const V3c& v) { return u.x * v.x + u.y * v.y + u.z * v.z; };
    const float d00 = d3(v0, v0), d01 = d3(v0, v1), d11 = d3(v1, v1);
    const float d20 = d3(v2, v0), d21 = d3(v2, v1);
    const float denom = d00 * d11 - d01 * d01;
    const float v = (d11 * d20 - d01 * d21) / denom;
    const float w = (d00 * d21 - d01 * d20) / denom;
    const float u = 1.0f - v - w;
    return F4{ u, v, w, 0.0f };   // bary.w = 0 on the TRIANGLE overload (PxMathUtils.h:409)
}

// PxEncodeClothIndex (PxgSimulationCoreDesc.h:101-108), spelled from its own line --
// PXG_BITSHIFT_ELEMENT_ID, not PXG_BITSHIFT_TET_ID.
static uint32_t cpuEncodeCloth(uint32_t clothId, uint32_t elementId)
{
    return (clothId << PXG_BITSHIFT_ELEMENT_ID) | elementId;
}

PXG_TEST(cvxCoreClothNp)
{
    PxgSlangComputeDevice* dev = testDevice();

    // ---- the values the SCENE cannot choose ------------------------------
    // Each of these makes at least one P9.74 class-A survivor observable.
    const uint32_t CLOTH_ID = 3u;              // scene has one cloth, so clothId is 0 there
    const uint32_t MATERIAL_INDEX = 0x1234u;
    const uint64_t RIGID_NODE_IND = 0x00000007ABCDEF01ull;  // BOTH words non-zero and unequal
    const float REST_DIST_CM0 = 0.05f;         // restDistance[0], the only correct read
    const float REST_DIST_POISON = 999.0f;     // restDistance[1]: what pairIndex would read
    const float CONTACT_OFF = 0.01f;           // per shape, so contactDist == 0.02

    // ---- geometry --------------------------------------------------------
    const float TRI_Y = 0.9f;                  // 0.1 inside the box's top face at y = 1
    const float BOX_TOP_Y = 1.0f;
    std::vector<F4> verts = {
        F4{ -0.5f, TRI_Y, -0.5f, 1.0f }, F4{ 0.5f, TRI_Y, -0.5f, 1.0f }, F4{ 0.0f, TRI_Y, 0.5f, 1.0f },
        F4{ -0.4f, TRI_Y,  0.2f, 1.0f }, F4{ 0.4f, TRI_Y,  0.2f, 1.0f }, F4{ 0.0f, TRI_Y, -0.6f, 1.0f },
    };
    // Triangle 2 is TILTED: its plane is not perpendicular to the contact normal.
    // That is the whole point of case D below -- with a normal perpendicular to the
    // triangle, shifting the contact point along it does NOT move the in-plane
    // projection, so the barycentric of the raw point and of the shifted point are
    // the same number and `bary-of-raw-point` is unobservable.  A tilted triangle
    // breaks that degeneracy, and nothing about the resting scene ever could.
    verts.push_back(F4{ -0.5f, 0.85f, -0.5f, 1.0f });
    verts.push_back(F4{  0.5f, 0.95f, -0.5f, 1.0f });
    verts.push_back(F4{  0.0f, 0.90f,  0.5f, 1.0f });
    std::vector<uint32_t> triInds = { 0,1,2,0,  3,4,5,0,  6,7,8,0 };   // three uint4 rows
    DevBuf bVerts = makeDev(dev, verts);
    DevBuf bTris = makeDev(dev, triInds);

    PxgFEMCloth cloth;
    std::memset(&cloth, 0, sizeof(cloth));
    cloth.mPosition_InvMass = reinterpret_cast<float4*>(bVerts.addr);
    cloth.mTriangleVertexIndices = reinterpret_cast<uint4*>(bTris.addr);
    // mTriMeshData is deliberately left NULL: this kernel never traverses, so
    // getNodes() and scaleBounds() are unreachable from it (convexCoreCollision.cu:1195-1279).
    // The kernel indexes femClothes[clothId] directly, so the array must REACH
    // index CLOTH_ID.  Uploading one descriptor and then asking for index 3 reads
    // off the end -- which is how the first run of this fixture segfaulted, and it
    // is precisely the mistake a scene with one cloth can never teach.
    std::vector<PxgFEMCloth> clothArr(CLOTH_ID + 1u);
    std::memset(clothArr.data(), 0, clothArr.size() * sizeof(PxgFEMCloth));
    clothArr[CLOTH_ID] = cloth;
    DevBuf bCloth = makeDev(dev, clothArr);

    std::vector<PxgShape> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    // shape0: the convex CORE box.  The core TYPE rides in hullOrMeshPtr and the
    // core DATA in the PxMeshScale bytes (convexCoreCollision.cu:114-169); eBOX is 2
    // (PxConvexCoreGeometry.h:59), which is also the layer's CORE_BOX.  BoxCore
    // carries FULL extents, and the margin is the scale rotation's w.
    shapes[0].type = PxU32(PxGeometryType::eCONVEXCORE);
    shapes[0].hullOrMeshPtr = size_t(PxConvexCore::Type::eBOX);
    shapes[0].scale.scale = PxVec3(2.0f, 2.0f, 2.0f);
    shapes[0].scale.rotation = PxQuat(0.0f, 0.0f, 0.0f, 0.0f);   // .w IS the margin: 0
    shapes[0].materialIndex = MATERIAL_INDEX;
    // shape1: the cloth's collision shape is a TRIANGLE mesh.
    shapes[1].type = PxU32(PxGeometryType::eTRIANGLEMESH);
    shapes[1].particleOrSoftbodyId = CLOTH_ID;
    DevBuf bShapes = makeDev(dev, shapes);

    std::vector<PxsCachedTransform> tcache(2);
    std::memset(tcache.data(), 0, tcache.size() * sizeof(PxsCachedTransform));
    tcache[0].transform = PxTransform(PxIdentity);   // the box, at the origin
    tcache[1].transform = PxTransform(PxIdentity);   // the cloth: world-space verts
    DevBuf bTc = makeDev(dev, tcache);

    std::vector<PxgContactManagerInput> cmInputs(1);
    std::memset(cmInputs.data(), 0, sizeof(PxgContactManagerInput));
    cmInputs[0].shapeRef0 = 0u; cmInputs[0].shapeRef1 = 1u;
    cmInputs[0].transformCacheRef0 = 0u; cmInputs[0].transformCacheRef1 = 1u;
    DevBuf bCin = makeDev(dev, cmInputs);

    std::vector<float> contactDistArr = { CONTACT_OFF, CONTACT_OFF };
    DevBuf bCd = makeDev(dev, contactDistArr);
    // THE POISON SLOT.  restDistance is indexed by the CONTACT MANAGER index, which
    // is 0 for both pairs; a port that indexed by the PAIR index would read slot 1
    // for the second pair.  In the scene that mutant is invisible -- it reads past a
    // 4-byte allocation into a 128-byte buddy slot that happens to hold zeros, which
    // is exactly the accident P9.74 recorded as class D.  Here slot 1 is loud.
    std::vector<float> restArr = { REST_DIST_CM0, REST_DIST_POISON };
    DevBuf bRd = makeDev(dev, restArr);
    std::vector<uint64_t> remap = { RIGID_NODE_IND, 0ull };
    DevBuf bRemap = makeDev(dev, remap);

    // ---- the filter, which the scene has none of -------------------------
    // PxgRigidFilterPair: PxU64 index0, PxU64 index1, PxU32 index2, PxU32 pad[3].
    // ONE entry, matching triangle 0's FIRST vertex only.  find() is then true for
    // vertex .x and false for .y and .z, so:
    //     A && B && C  ==  false  -> the pair is NOT skipped   (the engine's meaning)
    //     A || (B && C) == true   -> the pair IS skipped       (the mutant)
    // Note the parse: replacing only the first `&&` gives `A || (B && C)`, not a
    // flat three-way OR -- C precedence, which Slang follows.
    struct FilterPairHost { uint64_t index0, index1; uint32_t index2, pad0, pad1, pad2; };
    static_assert(sizeof(FilterPairHost) == 32, "PxgRigidFilterPair mirror == 32B");
    std::vector<FilterPairHost> filters(1);
    std::memset(filters.data(), 0, filters.size() * sizeof(FilterPairHost));
    filters[0].index0 = RIGID_NODE_IND;
    filters[0].index1 = cpuEncodeCloth(CLOTH_ID, triInds[0]);   // triangle 0's .x
    DevBuf bFilters = makeDev(dev, filters);

    // ---- the midphase pairs ---------------------------------------------
    // uint4(cmIdx, primitiveIndex, 0, 0) -- femClothMidPhase.cu:98, and the Slang
    // port agrees (femClothContactGenBda.slang:1087).  TWO pairs on the SAME contact
    // manager with DIFFERENT triangles: that is what separates triIndex from the
    // zeroed .w slot, and what separates restDistance[cmIndex] from [pairIndex].
    std::vector<uint32_t> pairs = { 0u,0u,0u,0u,  0u,1u,0u,0u };
    DevBuf bPairs = makeDev(dev, pairs);
    std::vector<uint32_t> pairCount = { 2u };
    DevBuf bPairNum = makeDev(dev, pairCount);
    std::vector<uint32_t> stackNeeded = { 0u };
    DevBuf bStackNeeded = makeDev(dev, stackNeeded);

    WriterBufs w = makeWriter(dev, 64u);

    CvxCoreClothPush push = {};
    push.stackPtr = bPairs.addr;
    push.midphasePairsNum = bPairNum.addr;
    push.cmInputs = bCin.addr;
    push.shapes = bShapes.addr;
    push.femClothes = bCloth.addr;
    push.shapeToRigidRemapTable = bRemap.addr;
    push.transformCache = bTc.addr;
    push.contactDistance = bCd.addr;
    push.restDistance = bRd.addr;
    push.filterPairs = bFilters.addr;
    push.nbFilterPairs = 1u;
    push.stackSizeBytes = 4096u;           // far above 2 pairs: the clamp is a no-op here
    push.stackSizeNeededOnDevice = bStackNeeded.addr;
    push.w_outPoint = w.point.addr; push.w_outNormalPen = w.normalPen.addr;
    push.w_outBarycentric = w.barycentric.addr; push.w_outContactInfo = w.info.addr;
    push.w_totalContactCount = w.count.addr; push.w_contactByX = w.byRigid.addr;
    push.w_tempContactByX = w.tempByRigid.addr; push.w_contactIndexSortedByX = w.idxByRigid.addr;
    push.w_contactSortedByX = w.sortedByRigid.addr; push.w_maxNumContacts = w.maxContacts;

    Kernel k("cvxCoreClothNpBda.spv", "convexCoreClothmeshNphase_Kernel32", {}, sizeof(CvxCoreClothPush));
    k.dispatch(/*gridX*/ 1u, {}, push, 1u, 1u);
    dev->flush();

    // ---- read back --------------------------------------------------------
    const uint32_t gCount = downloadBuffer<uint32_t>(dev, w.count.h, 1)[0];
    PXG_CHECK_MSG(gCount > 0u && gCount <= w.maxContacts,
                  "the kernel emitted %u contacts; a triangle 0.1 inside the box must produce "
                  "some, and the filter must NOT suppress it (only one of three vertices is "
                  "filtered, and the engine's test is &&)", gCount);

    // A mutant that suppresses every contact (the filter always skipping, the
    // convex arm never built, the swap gone) leaves gCount == 0, and
    // downloadBuffer(.., 0) faults.  The first version of this fixture CRASHED on
    // four such mutants: they were counted as kills, which is right, but a crash
    // says only "something broke" where a check says what.  Bail out with the
    // failure already recorded.
    if (gCount == 0u)
    {
        std::printf("  cvxCoreCloth: no contacts at all -- nothing further can be checked\n");
        destroyWriter(dev, w);
        for (PxgBufferHandle h : { bVerts.h, bTris.h, bCloth.h, bShapes.h, bTc.h, bCin.h,
                                   bCd.h, bRd.h, bRemap.h, bFilters.h, bPairs.h, bPairNum.h,
                                   bStackNeeded.h })
            dev->destroyBuffer(h);
        return;
    }

    const std::vector<F4> gPoint = downloadBuffer<F4>(dev, w.point.h, gCount);
    const std::vector<F4> gNP = downloadBuffer<F4>(dev, w.normalPen.h, gCount);
    const std::vector<F4> gBary = downloadBuffer<F4>(dev, w.barycentric.h, gCount);
    const std::vector<FemOtherContactInfoHost> gInfo =
        downloadBuffer<FemOtherContactInfoHost>(dev, w.info.h, gCount);
    const std::vector<uint64_t> gByRigid = downloadBuffer<uint64_t>(dev, w.byRigid.h, gCount);
    const std::vector<uint32_t> gTempByRigid = downloadBuffer<uint32_t>(dev, w.tempByRigid.h, gCount);
    const std::vector<uint32_t> gIdxByRigid = downloadBuffer<uint32_t>(dev, w.idxByRigid.h, gCount);
    const uint32_t gStackNeeded = downloadBuffer<uint32_t>(dev, bStackNeeded.h, 1)[0];

    // ---- the per-contact invariants ---------------------------------------
    size_t mm = 0;
    uint32_t sawTri0 = 0u, sawTri1 = 0u, onTriPlane = 0u, onBoxPlane = 0u;
    const uint32_t pairInd1Tri0 = cpuEncodeCloth(CLOTH_ID, 0u);
    const uint32_t pairInd1Tri1 = cpuEncodeCloth(CLOTH_ID, 1u);

    for (uint32_t i = 0; i < gCount; ++i)
    {
        // 1. THE REST DISTANCE IS THE CONTACT MANAGER's, for every contact.
        if (f2u(gPoint[i].w) != f2u(REST_DIST_CM0))
        { ++mm; std::printf("  cvxCoreCloth[%u] outPoint.w = %g, expected restDistance[cmIndex] = %g "
                            "(restDistance[1] is %g -- reading it means indexing by pairIndex)\n",
                            i, gPoint[i].w, REST_DIST_CM0, REST_DIST_POISON); }

        // 2. THE FOURTH BARYCENTRIC SLOT IS A MODE SELECTOR, and this is the
        //    TRIANGLE generator, so it must be 0.  `if (bc.w == 0.f) // Cloth triangle`
        //    else `// Cloth vertex` -- FEMClothExternalSolve.cu:375,391,501 and
        //    deformableUtils.cuh:501,549,568.  The mirror-image assertion for the
        //    VERTEX path is above at cmpF4("outBarycentric", F4{0,0,0,1}, ...).
        if (f2u(gBary[i].w) != f2u(0.0f))
        { ++mm; std::printf("  cvxCoreCloth[%u] outBarycentric.w = %g, expected 0 -- a 1 here makes "
                            "the solver read the contact as a VERTEX contact and index mDeltaPos "
                            "with a TRIANGLE index\n", i, gBary[i].w); }

        // 3. THE BARYCENTRIC IS OF THE WRITTEN CONTACT POINT.  Recomputed on the host
        //    from the kernel's OWN outPoint, so this is a cross-check and not a
        //    re-derivation: a port that took the barycentric of the pre-shift point,
        //    or permuted u/v/w, disagrees with its own output.
        const uint32_t tri = (gInfo[i].pairInd1 == pairInd1Tri1) ? 1u : 0u;
        const uint32_t b0 = triInds[tri * 4u + 0u], b1 = triInds[tri * 4u + 1u], b2 = triInds[tri * 4u + 2u];
        const V3c a{ verts[b0].x, verts[b0].y, verts[b0].z };
        const V3c b{ verts[b1].x, verts[b1].y, verts[b1].z };
        const V3c c{ verts[b2].x, verts[b2].y, verts[b2].z };
        const F4 refBary = cpuTriBarycentric(a, b, c, V3c{ gPoint[i].x, gPoint[i].y, gPoint[i].z });
        if (f2u(refBary.x) != f2u(gBary[i].x) || f2u(refBary.y) != f2u(gBary[i].y) ||
            f2u(refBary.z) != f2u(gBary[i].z))
        { ++mm; std::printf("  cvxCoreCloth[%u] outBarycentric (%g,%g,%g) != PxComputeBarycentric of "
                            "its own outPoint (%g,%g,%g)\n", i, gBary[i].x, gBary[i].y, gBary[i].z,
                            refBary.x, refBary.y, refBary.z); }

        // 4. THE MATERIAL INDEX, WITH BIT 31 CLEARED.  setRigidMaterialIndex(mat)
        //    followed by markInCollision(false) collapses to (mat & 0x7fffffff);
        //    bit 31 is the isInCollision flag (PxgFEMCore.h:144-158).
        if (gInfo[i].rigidMatInd_isInCollision != (MATERIAL_INDEX & 0x7fffffffu))
        { ++mm; std::printf("  cvxCoreCloth[%u] rigidMatInd = 0x%08x, expected 0x%08x\n",
                            i, gInfo[i].rigidMatInd_isInCollision, MATERIAL_INDEX & 0x7fffffffu); }

        // 5. THE SORT KEY IS THE WHOLE 64-BIT NODE INDEX.  Both words of
        //    RIGID_NODE_IND are non-zero and unequal, so a swapped pair is visible --
        //    which it is not in a scene with one rigid.
        if (gByRigid[i] != RIGID_NODE_IND)
        { ++mm; std::printf("  cvxCoreCloth[%u] contactByRigid = 0x%016llx, expected 0x%016llx\n",
                            i, (unsigned long long)gByRigid[i], (unsigned long long)RIGID_NODE_IND); }
        if (gTempByRigid[i] != uint32_t(RIGID_NODE_IND & 0xffffffffull))
        { ++mm; std::printf("  cvxCoreCloth[%u] tempContactByRigid = 0x%08x\n", i, gTempByRigid[i]); }
        if (gIdxByRigid[i] != i)
        { ++mm; std::printf("  cvxCoreCloth[%u] contactIndexSortedByRigid = %u\n", i, gIdxByRigid[i]); }
        if (gInfo[i].pairInd0 != RIGID_NODE_IND)
        { ++mm; std::printf("  cvxCoreCloth[%u] info.pairInd0 = 0x%016llx\n",
                            i, (unsigned long long)gInfo[i].pairInd0); }

        // 6. THE CLOTH ELEMENT ID.  CLOTH_ID is 3, not 0, so both the id itself and
        //    the shift constant are observable here.
        if (gInfo[i].pairInd1 != pairInd1Tri0 && gInfo[i].pairInd1 != pairInd1Tri1)
        { ++mm; std::printf("  cvxCoreCloth[%u] info.pairInd1 = 0x%08x, expected 0x%08x (tri 0) or "
                            "0x%08x (tri 1) -- clothId %u shifted by PXG_BITSHIFT_ELEMENT_ID\n",
                            i, gInfo[i].pairInd1, pairInd1Tri0, pairInd1Tri1, CLOTH_ID); }
        if (gInfo[i].pairInd1 == pairInd1Tri0) ++sawTri0;
        if (gInfo[i].pairInd1 == pairInd1Tri1) ++sawTri1;

        // 7. THE NORMAL IS NEGATED ON WRITE (convexCoreCollision.cu:1269), and the
        //    SIGN is the point -- an abs() here would pin the axis and miss the
        //    negation entirely, which is how the first version of this fixture let
        //    `normal-not-negated` survive while the SCENE killed it.
        //
        //    The expected sign is derived, not fitted: the cloth sits 0.1 BELOW the
        //    box's top face, inside the solid, so the direction that expels it is
        //    +Y.  The written normal is that direction.
        if (!(almostEqual(gNP[i].x, 0.0f, 1e-4f, 1e-5f) && almostEqual(gNP[i].z, 0.0f, 1e-4f, 1e-5f) &&
              almostEqual(gNP[i].y, 1.0f, 1e-4f, 1e-5f)))
        { ++mm; std::printf("  cvxCoreCloth[%u] outNormalPen.xyz = (%g,%g,%g), expected (0,1,0) -- "
                            "the cloth is inside the box and must be pushed OUT, along +Y\n",
                            i, gNP[i].x, gNP[i].y, gNP[i].z); }

        // 8. THE PENETRATION IS WRITTEN RAW -- no 40%% ERP scaling.  The triangle sits
        //    0.1 inside the top face, and the only margins in play are the convex's
        //    (0) and the triangle's (0.001 * the fattened box dimension, ~0.002), so
        //    the separation must land near -0.1.  The tetmesh kernel's
        //    `if (d < 0) d *= 0.4f` would put it near -0.04, which this window
        //    excludes by a factor of two.
        //    The value is DERIVED to its last term, not bounded loosely:
        //      penetration   = BOX_TOP_Y - TRI_Y                     = 0.1
        //      convex margin = shape0.scale.rotation.w               = 0
        //      box bounds    = the full extent, FATTENED by contactDist on each side
        //                    = 2.0 + 2*0.02                          = 2.04
        //      triangle margin = triMarginK * maxElement(dims)       = 0.001 * 2.04
        //      separation    = -(penetration + triangle margin)      = -0.10204
        //    and `dist -= marginA + marginB` is where the two margins leave.  A
        //    window this tight also separates the three margin mutants, which a
        //    +/-0.02 window could not: triMarginK = 0 gives -0.1, the unfattened
        //    bounds give -0.102, and a one-sided contactDist gives -0.10202.
        //    ONE FACTOR IN THIS LINE WAS WRONG AND THE SWEEP IS WHAT FOUND IT.  The
        //    fattening is by `contactDist`, the SUM of the two shapes' offsets, on
        //    each side -- not by one shape's CONTACT_OFF.  With 2.02 instead of 2.04
        //    the reference landed exactly BETWEEN the clean value (-0.10204) and two
        //    of the margin mutants (-0.102 and -0.10202), and tolerated both.  A
        //    reference that sits in the middle of the mutants it is meant to exclude
        //    is the worst possible kind: it passes, and it passes them too.
        const float contactDistSum = 2.0f * CONTACT_OFF;              // == the kernel's contactDist
        const float fattenedDim = 2.0f + 2.0f * contactDistSum;       // full extent + both sides
        const float refSep = -( (BOX_TOP_Y - TRI_Y) + 0.001f * fattenedDim );
        if (!almostEqual(gNP[i].w, refSep, 1e-4f, 2e-6f))
        { ++mm; std::printf("  cvxCoreCloth[%u] separation = %.8g, expected %.8g raw (the 0.4x ERP "
                            "hack of the TETMESH kernel would give %.8g)\n",
                            i, gNP[i].w, refSep, refSep * 0.4f); }

        // 9. THE HALF-PENETRATION SHIFT LANDS THE POINT ON A SURFACE.  faceClipQ
        //    already returns the midpoint between the two projections, so one more
        //    half-separation puts it on the triangle plane (0.9) or the box face
        //    (1.0).  Omitting the shift leaves 0.95, which is neither.
        const bool onTri = almostEqual(gPoint[i].y, TRI_Y, 1e-3f, 1e-4f);
        const bool onBox = almostEqual(gPoint[i].y, BOX_TOP_Y, 1e-3f, 1e-4f);
        if (onTri) ++onTriPlane;
        if (onBox) ++onBoxPlane;
        if (!onTri && !onBox)
        { ++mm; std::printf("  cvxCoreCloth[%u] outPoint.y = %g, expected the triangle plane %g or "
                            "the box face %g; %g would be the un-shifted midpoint\n",
                            i, gPoint[i].y, TRI_Y, BOX_TOP_Y, 0.5f * (TRI_Y + BOX_TOP_Y)); }
    }

    // ---- the whole-run invariants -----------------------------------------
    // BOTH triangles must appear.  With `triIndex = testPair.w` -- and the midphase
    // writes uint4(cmIdx, prim, 0, 0) -- every contact would carry triangle 0.
    PXG_CHECK_MSG(sawTri0 > 0u && sawTri1 > 0u,
                  "contacts for triangle 0: %u, triangle 1: %u -- both midphase pairs must be "
                  "consumed with their own triangle index", sawTri0, sawTri1);

    // The atomicMax tail (convexCoreCollision.cu:1277-1278) reports the UNCLAMPED
    // pair count in BYTES, which is how the engine learns to grow the stack.
    PXG_CHECK_MSG(gStackNeeded == 2u * 16u,
                  "stackSizeNeededOnDevice = %u, expected %u (numMidphasePairsFound * sizeof(uint4))",
                  gStackNeeded, 2u * 16u);

    PXG_CHECK_MSG(mm == 0, "convexCoreClothmeshNphase_Kernel32 contact invariants (%zu mismatches)", mm);

    // Teeth: a run that produced contacts for only one of the two planes, or none on
    // either, would satisfy every per-contact check above vacuously.
    std::printf("    [ info ] cvxCoreCloth: %u contacts (tri0 %u, tri1 %u); on triangle plane %u, "
                "on box face %u; normal (%g,%g,%g) sep %g point0 (%g,%g,%g)\n",
                gCount, sawTri0, sawTri1, onTriPlane, onBoxPlane,
                gNP[0].x, gNP[0].y, gNP[0].z, gNP[0].w, gPoint[0].x, gPoint[0].y, gPoint[0].z);

    // =======================================================================
    // CASE B -- THE PAIR ARRIVES MESH-FIRST, so checkTypes' swap arm has to fire.
    //
    // The scene reaches this arm and the first version of this fixture did not,
    // because it put the core in slot 0 where the kernel already wants it.  A
    // fixture that only ever feeds the easy order is asserting that the easy order
    // works.  With the swap disabled, shape0 stays the TRIANGLE mesh,
    // makeConvexShapeCvxCore returns false for it, and every pair is skipped -- so
    // the count going to zero IS the discrimination.
    // =======================================================================
    {
        std::vector<PxgContactManagerInput> cmSwapped(1);
        std::memset(cmSwapped.data(), 0, sizeof(PxgContactManagerInput));
        cmSwapped[0].shapeRef0 = 1u; cmSwapped[0].shapeRef1 = 0u;          // mesh first
        cmSwapped[0].transformCacheRef0 = 1u; cmSwapped[0].transformCacheRef1 = 0u;
        DevBuf bCinSwap = makeDev(dev, cmSwapped);
        std::vector<uint32_t> sn2 = { 0u };
        DevBuf bSn2 = makeDev(dev, sn2);
        WriterBufs w2 = makeWriter(dev, 64u);

        CvxCoreClothPush p2 = push;
        p2.cmInputs = bCinSwap.addr;
        p2.stackSizeNeededOnDevice = bSn2.addr;
        p2.w_outPoint = w2.point.addr; p2.w_outNormalPen = w2.normalPen.addr;
        p2.w_outBarycentric = w2.barycentric.addr; p2.w_outContactInfo = w2.info.addr;
        p2.w_totalContactCount = w2.count.addr; p2.w_contactByX = w2.byRigid.addr;
        p2.w_tempContactByX = w2.tempByRigid.addr; p2.w_contactIndexSortedByX = w2.idxByRigid.addr;
        p2.w_contactSortedByX = w2.sortedByRigid.addr;

        k.dispatch(1u, {}, p2, 1u, 1u);
        dev->flush();

        const uint32_t c2 = downloadBuffer<uint32_t>(dev, w2.count.h, 1)[0];
        PXG_CHECK_MSG(c2 == gCount,
                      "mesh-first gave %u contacts, core-first gave %u -- checkTypes must swap the "
                      "two slots so the arm is order-independent", c2, gCount);
        if (c2 == gCount)
        {
            const std::vector<F4> p2Point = downloadBuffer<F4>(dev, w2.point.h, c2);
            const std::vector<F4> p2NP = downloadBuffer<F4>(dev, w2.normalPen.h, c2);
            size_t sm = 0;
            for (uint32_t i = 0; i < c2; ++i)
                if (f2u(p2Point[i].x) != f2u(gPoint[i].x) || f2u(p2Point[i].y) != f2u(gPoint[i].y) ||
                    f2u(p2Point[i].z) != f2u(gPoint[i].z) || f2u(p2NP[i].w) != f2u(gNP[i].w))
                    ++sm;
            PXG_CHECK_MSG(sm == 0, "mesh-first and core-first disagree on %zu of %u contacts", sm, c2);
        }
        destroyWriter(dev, w2);
        dev->destroyBuffer(bCinSwap.h); dev->destroyBuffer(bSn2.h);
    }

    // =======================================================================
    // CASE C -- THE MIDPHASE OVERFLOWED, so the stack clamp has to bite.
    //
    //     stackSize = stackSizeBytes / sizeof(uint4);
    //     numPairs  = PxMin(numMidphasePairsFound, stackSize);
    //
    // stackSizeBytes = 16 means room for ONE uint4, so only the first of the two
    // pairs is actually on the stack and only triangle 0 may be consumed.  The
    // count reported through the atomicMax tail is the UNCLAMPED one -- that is how
    // the engine learns to grow the buffer -- so it stays at 2 * 16.
    //
    // The scene cannot reach this: 32 pairs against a multi-megabyte stack make the
    // clamp a no-op, which is why `no-clamp-to-stack` and `stack-size-in-bytes` are
    // provably equivalent there and discriminating here.
    // =======================================================================
    {
        std::vector<uint32_t> sn3 = { 0u };
        DevBuf bSn3 = makeDev(dev, sn3);
        WriterBufs w3 = makeWriter(dev, 64u);

        CvxCoreClothPush p3 = push;
        p3.stackSizeBytes = 16u;              // exactly one uint4
        p3.stackSizeNeededOnDevice = bSn3.addr;
        p3.w_outPoint = w3.point.addr; p3.w_outNormalPen = w3.normalPen.addr;
        p3.w_outBarycentric = w3.barycentric.addr; p3.w_outContactInfo = w3.info.addr;
        p3.w_totalContactCount = w3.count.addr; p3.w_contactByX = w3.byRigid.addr;
        p3.w_tempContactByX = w3.tempByRigid.addr; p3.w_contactIndexSortedByX = w3.idxByRigid.addr;
        p3.w_contactSortedByX = w3.sortedByRigid.addr;

        k.dispatch(1u, {}, p3, 1u, 1u);
        dev->flush();

        const uint32_t c3 = downloadBuffer<uint32_t>(dev, w3.count.h, 1)[0];
        const uint32_t sn3v = downloadBuffer<uint32_t>(dev, bSn3.h, 1)[0];
        const std::vector<FemOtherContactInfoHost> i3 =
            downloadBuffer<FemOtherContactInfoHost>(dev, w3.info.h, c3 ? c3 : 1u);
        uint32_t tri1Leaked = 0u;
        for (uint32_t i = 0; i < c3; ++i)
            if (i3[i].pairInd1 == pairInd1Tri1) ++tri1Leaked;

        PXG_CHECK_MSG(c3 == sawTri0 && tri1Leaked == 0u,
                      "with room for ONE pair the kernel produced %u contacts (%u of them from the "
                      "SECOND pair); expected %u, all from triangle 0 -- stackSizeBytes / 16 is the "
                      "clamp, and the second uint4 is not on the stack",
                      c3, tri1Leaked, sawTri0);
        PXG_CHECK_MSG(sn3v == 2u * 16u,
                      "stackSizeNeededOnDevice = %u under the clamp, expected %u -- the tail reports "
                      "the UNCLAMPED count so the engine can grow the stack", sn3v, 2u * 16u);

        destroyWriter(dev, w3);
        dev->destroyBuffer(bSn3.h);
    }

    // =======================================================================
    // CASE D -- A TILTED TRIANGLE, so the half-penetration shift moves the
    // barycentric.
    //
    // Cases A-C all use triangles whose plane is perpendicular to the contact
    // normal.  For those, `point = raw + normal * w * 0.5` slides along the
    // triangle's own normal, the in-plane projection is unchanged, and
    // PxComputeBarycentric returns the SAME three numbers for the raw point and the
    // shifted one.  `bary-of-raw-point` is therefore not a different program there,
    // and no amount of tightening reaches it.
    //
    // Triangle 2 is tilted (y from 0.85 to 0.95) while the contact normal stays +Y,
    // because the way out of the box is still straight up.  Now the shift has a
    // component in the triangle's plane and the two barycentrics differ.
    // =======================================================================
    {
        std::vector<uint32_t> pairsD = { 0u, 2u, 0u, 0u };
        DevBuf bPairsD = makeDev(dev, pairsD);
        std::vector<uint32_t> pairNumD = { 1u };
        DevBuf bPairNumD = makeDev(dev, pairNumD);
        std::vector<uint32_t> snD = { 0u };
        DevBuf bSnD = makeDev(dev, snD);
        WriterBufs wD = makeWriter(dev, 64u);

        CvxCoreClothPush pD = push;
        pD.stackPtr = bPairsD.addr;
        pD.midphasePairsNum = bPairNumD.addr;
        pD.stackSizeNeededOnDevice = bSnD.addr;
        pD.w_outPoint = wD.point.addr; pD.w_outNormalPen = wD.normalPen.addr;
        pD.w_outBarycentric = wD.barycentric.addr; pD.w_outContactInfo = wD.info.addr;
        pD.w_totalContactCount = wD.count.addr; pD.w_contactByX = wD.byRigid.addr;
        pD.w_tempContactByX = wD.tempByRigid.addr; pD.w_contactIndexSortedByX = wD.idxByRigid.addr;
        pD.w_contactSortedByX = wD.sortedByRigid.addr;

        k.dispatch(1u, {}, pD, 1u, 1u);
        dev->flush();

        const uint32_t cD = downloadBuffer<uint32_t>(dev, wD.count.h, 1)[0];
        PXG_CHECK_MSG(cD > 0u, "the tilted triangle produced no contacts (%u)", cD);
        if (cD > 0u)
        {
            const std::vector<F4> dPoint = downloadBuffer<F4>(dev, wD.point.h, cD);
            const std::vector<F4> dBary = downloadBuffer<F4>(dev, wD.barycentric.h, cD);
            const std::vector<F4> dNP = downloadBuffer<F4>(dev, wD.normalPen.h, cD);
            const V3c ta{ verts[6].x, verts[6].y, verts[6].z };
            const V3c tb{ verts[7].x, verts[7].y, verts[7].z };
            const V3c tc{ verts[8].x, verts[8].y, verts[8].z };
            size_t dm = 0; uint32_t moved = 0u;
            for (uint32_t i = 0; i < cD; ++i)
            {
                const F4 refB = cpuTriBarycentric(ta, tb, tc,
                                                  V3c{ dPoint[i].x, dPoint[i].y, dPoint[i].z });
                if (f2u(refB.x) != f2u(dBary[i].x) || f2u(refB.y) != f2u(dBary[i].y) ||
                    f2u(refB.z) != f2u(dBary[i].z) || f2u(dBary[i].w) != f2u(0.0f))
                { ++dm; std::printf("  cvxCoreCloth tilted[%u] bary (%g,%g,%g,%g) != of its own point "
                                    "(%g,%g,%g,0)\n", i, dBary[i].x, dBary[i].y, dBary[i].z,
                                    dBary[i].w, refB.x, refB.y, refB.z); }
                // The non-triviality guard: if the shift did NOT move the barycentric
                // -- which is what happens when the normal is perpendicular to the
                // triangle -- then this case proves nothing and must say so rather
                // than pass quietly.  P9.73 wrote the same kind of counter after a
                // survivor showed a fixture was measuring a degenerate configuration.
                const V3c raw{ dPoint[i].x - 0.0f * dNP[i].w,
                               dPoint[i].y - (-dNP[i].y) * dNP[i].w * 0.5f,
                               dPoint[i].z };
                const F4 rawB = cpuTriBarycentric(ta, tb, tc, raw);
                if (f2u(rawB.x) != f2u(refB.x) || f2u(rawB.y) != f2u(refB.y)) ++moved;
            }
            PXG_CHECK_MSG(dm == 0, "tilted-triangle barycentric mismatches: %zu of %u", dm, cD);
            PXG_CHECK_MSG(moved > 0u,
                          "the tilted case is DEGENERATE: for all %u contacts the barycentric of "
                          "the shifted point equals that of the unshifted one, so this case cannot "
                          "tell the two apart and proves nothing", cD);
            std::printf("    [ info ] cvxCoreCloth tilted: %u contacts, %u of them with a "
                        "shift-sensitive barycentric\n", cD, moved);
        }
        destroyWriter(dev, wD);
        dev->destroyBuffer(bPairsD.h); dev->destroyBuffer(bPairNumD.h); dev->destroyBuffer(bSnD.h);
    }

    destroyWriter(dev, w);
    for (PxgBufferHandle h : { bVerts.h, bTris.h, bCloth.h, bShapes.h, bTc.h, bCin.h,
                               bCd.h, bRd.h, bRemap.h, bFilters.h, bPairs.h, bPairNum.h,
                               bStackNeeded.h })
        dev->destroyBuffer(h);
}

} // namespace
