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
#include "PxgSimulationCoreDesc.h"           // PxgDeformablePairInfo / PxgDeformableRef (5.12)
#include "PxgDeformableContactWriters.h"     // PxgRigidDbContactWriter (5.12)
#include "PxgSolverBody.h"                   // PxgSolverBodyData / PxgSolverTxIData / PxAlignedTransform
#include "CmSpatialVector.h"                 // Cm::UnAlignedSpatialVector (tranche A: articulation prepare oracle)
#include "PxSpatialMatrix.h"                 // PxSpatialMatrix::operator* -- the engine's own product
#include "PxgArticulation.h"                 // PxgArticulation record (tranche A)
#include "PxgArticulationLink.h"             // PxgArticulationLink (penBiasClamp)
#include "PxgArticulationCoreDesc.h"         // mMaxLinksPerArticulation / nbArticulations
#include "PxgBodySim.h"                      // articulationRemapId
#include "PxgContactManager.h"               // PxgContactManagerInput (P7.1 tranche 4a contact gen)
#include "PxgConvexConvexShape.h"            // PxgShape (P7.1 tranche 4a contact gen)
#include "PxsTransformCache.h"               // PxsCachedTransform (P7.1 tranche 4a contact gen)
#include "geometry/PxGeometry.h"             // PxGeometryType (P7.1 tranche 4a contact gen)
#include "geometry/PxConvexCoreGeometry.h"  // PxConvexCore::Type (P9.75: eBOX read from the enum,
                                            // never a remembered 2 -- the P9.57 rule)
#include "foundation/PxBounds3.h"            // PxBounds3 (P7.1 tranche 4a midphase AABB)
#include "geometry/PxMeshScale.h"           // P9.94d: getTriBarycentric's scale.toMat33()
#include "pxgHfLocalBoundRef.h"              // P9.89: transformFast(getInverse()) oracle + twins
#include "pxgBv32TreeGen.h"                // synthetic BV32 trees for the multi-workgroup refit

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <array>
#include <limits>
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
// engine PxgFEMCloth (offsets the three kernels dereference).  sizeof == 416 (5.12).
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimDescLayout)
{
    // sizeof / array stride: the cloth kernels index femCloths[id], so the STRIDE
    // must match (verified in SPIR-V as ArrayStride 416; 5.12 added the two filter-pair
    // arrays and their counts).
    PXG_CHECK_MSG(sizeof(PxgFEMCloth) == 416, "PxgFEMCloth sizeof == 416 (got %zu)", sizeof(PxgFEMCloth));

    // device-pointer members
    PXG_CHECK(offsetof(PxgFEMCloth, mVelocity_InvMass) == 8);
    PXG_CHECK(offsetof(PxgFEMCloth, mPosition_InvMass) == 16);
    PXG_CHECK(offsetof(PxgFEMCloth, mRestPosition) == 24);
    PXG_CHECK(offsetof(PxgFEMCloth, mPrevPositionInContactOffset) == 32);
    PXG_CHECK(offsetof(PxgFEMCloth, mNonSharedTriPair_hasActiveBending) == 160);
    PXG_CHECK(offsetof(PxgFEMCloth, mOrderedSharedRestEdge0_edge1) == 168); // +8 => 7 pad after the bool
    PXG_CHECK(offsetof(PxgFEMCloth, mAccumulatedDeltaPos) == 256);

    // 5.12: the per-cloth filter pairs (sorted PxgDeformablePairInfo) and their counts
    PXG_CHECK(offsetof(PxgFEMCloth, mRigidSurfaceFilterPairs) == 280);
    PXG_CHECK(offsetof(PxgFEMCloth, mSurfaceSurfaceFilterPairs) == 288);
    PXG_CHECK(offsetof(PxgFEMCloth, mNumRigidSurfaceFilterPairs) == 296);
    PXG_CHECK(offsetof(PxgFEMCloth, mNumSurfaceSurfaceFilterPairs) == 300);

    // scalar tail
    PXG_CHECK(offsetof(PxgFEMCloth, mLinearDamping) == 304);
    PXG_CHECK(offsetof(PxgFEMCloth, mMaxLinearVelocity) == 308);
    PXG_CHECK(offsetof(PxgFEMCloth, mSettlingThreshold) == 316);
    PXG_CHECK(offsetof(PxgFEMCloth, mSleepThreshold) == 320);
    PXG_CHECK(offsetof(PxgFEMCloth, mSettlingDamping) == 324);
    PXG_CHECK(offsetof(PxgFEMCloth, mNbVerts) == 332);
    PXG_CHECK(offsetof(PxgFEMCloth, mActorFlags) == 392);
    PXG_CHECK(offsetof(PxgFEMCloth, mBodyFlags) == 393);
    PXG_CHECK(offsetof(PxgFEMCloth, mSurfaceFlags) == 394);
    PXG_CHECK(offsetof(PxgFEMCloth, mIsActive) == 396);
    PXG_CHECK(offsetof(PxgFEMCloth, mRestOffset) == 400);

    // P7.1: the in-plane membrane solve (cloth_solveNonSharedTriangleEnergy*) reads
    // these fields through the pxgSlangFEMClothDesc mirror — pin their offsets.
    PXG_CHECK(offsetof(PxgFEMCloth, mMaterialIndices) == 56);                                // PxU16*
    PXG_CHECK(offsetof(PxgFEMCloth, mOrderedNonSharedTriangleVertexIndices_triIndex) == 80); // uint4*
    PXG_CHECK(offsetof(PxgFEMCloth, mOrderedNonSharedTriangleLambdas) == 96);                // float2*
    PXG_CHECK(offsetof(PxgFEMCloth, mOrderedNonSharedTriangleRestPoseInv) == 104);           // float4* (QInv)
    PXG_CHECK(offsetof(PxgFEMCloth, mNonSharedTriAccumulatedPartitionsCP) == 192);           // PxU32*
    PXG_CHECK(offsetof(PxgFEMCloth, mNbNonSharedTriPartitions) == 360);                      // PxU32

    // the byte-packed flag word (@392): low byte == mActorFlags (eDISABLE_GRAVITY test)
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
    PXG_CHECK(offsetof(PxgFEMCloth, mNbSharedTrianglePairs) == 352);                                // PxU32
    PXG_CHECK(offsetof(PxgFEMCloth, mNbNonSharedTrianglePairs) == 356);                             // PxU32
    PXG_CHECK(offsetof(PxgFEMCloth, mNbSharedTriPairPartitions) == 364);                            // PxU32
    PXG_CHECK(offsetof(PxgFEMCloth, mNbNonSharedTriPairPartitions) == 368);                         // PxU32
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
    c.mRestOffset = restDistance;
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
// P7.1 tranche 4a: cloth-vertex-vs-PRIMITIVE contact GENERATION.
//   femClothContactGenBda.spv:
//     cloth_midphaseGenerateVertexPairsLaunch  (femClothMidPhase.cu)
//     cloth_planeVertContactGenLaunch          (femClothPrimitives.cu:853)
//     cloth_SphereVertexContactGenLaunch       (femClothPrimitives.cu:700)
// Bit-exact vs a CPU transcription of the .cu math (both memory modes).
// ===========================================================================

// PxgDeformablePairInfo host mirror (PxgSimulationCoreDesc.h, 5.12): ref0 / ref1, 16 bytes.
// The 5.6.1 PxgFemOtherContactInfo (pairInd0 / packed pairInd1 / rigidMatInd_isInCollision)
// is gone: the rigid material index has its own PxU16 stream and the cloth feature is a
// 64-bit PxgDeformableRef.
struct FemPairInfoHost
{
    uint64_t ref0;   // @0  the rigid PxNodeIndex (or another participant's ref)
    uint64_t ref1;   // @8  the PxgDeformableRef
};
static_assert(sizeof(FemPairInfoHost) == 16, "PxgDeformablePairInfo mirror == 16B");
static_assert(sizeof(FemPairInfoHost) == sizeof(PxgDeformablePairInfo), "PxgDeformablePairInfo stride");

// PxgDeformableRef [deformable:30 @34][active][invalid][type:3 @29][localEdge:2 @27][feature:27].
static uint64_t encClothRef(uint32_t clothId, uint32_t type, uint32_t feature)
{
    return (uint64_t(clothId) << 34) | (uint64_t(type) << 29) | uint64_t(feature);
}
static uint64_t encClothVertex(uint32_t clothId, uint32_t v) { return encClothRef(clothId, 0u, v); }
static uint64_t encClothTriangle(uint32_t clothId, uint32_t t) { return encClothRef(clothId, 1u, t); }
static uint64_t encClothEdge(uint32_t clothId, uint32_t t, uint32_t localEdge)
{
    return encClothRef(clothId, 3u, t) | (uint64_t(localEdge) << 27);
}

// midphase push blob — byte-identical to MidphaseParams (ABI row {4,4,8,8,8,8,8,8,8,8}).
struct MidphasePush
{
    uint32_t numTests;
    float toleranceLength;
    uint64_t cmInputs, transformCache, bounds, contactDistance, gpuShapes, femClothes;
    uint64_t stackPtr, midphasePairsNum;
};
static_assert(sizeof(MidphasePush) == 72, "midphase push blob must be 72 bytes");

// plane / sphere / box vertex contact-gen push blob — byte-identical to ContactGenParams
// (ABI row {4,8,8,8,8,8,8,8,80}, 5.12).  The PxgRigidDbContactWriter (80B) inlined @64; the
// 5.6.1 filterPairs / nbFilterPairs arguments are gone (the filter is per cloth,
// PxgFEMCloth::mRigidSurfaceFilterPairs).
struct ContactGenPush
{
    uint32_t numTests, pad4;
    uint64_t cmInputs, transformCache, contactDistance, restDistance, gpuShapes, femClothes;
    uint64_t shapeToRigidRemap;
    uint64_t w_outPoint, w_outNormalPen, w_outBarycentric, w_outContactInfo, w_outRigidMaterialIndex;
    uint64_t w_totalContactCount, w_contactByRigid, w_tempContactByRigid, w_contactIndexSortedByRigid;
    uint32_t w_maxNumContacts, pad140;
};
static_assert(sizeof(ContactGenPush) == 144, "contact-gen push blob must be 144 bytes");
static_assert(offsetof(ContactGenPush, w_outPoint) == 64, "writer @64");
static_assert(offsetof(ContactGenPush, w_maxNumContacts) == 136, "maxNumContacts @136");
static_assert(sizeof(PxgRigidDbContactWriter) == 80, "PxgRigidDbContactWriter sizeof");

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
    static_assert(sizeof(PxgDeformablePairInfo) == 16, "PxgDeformablePairInfo stride");

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

// Shared writer-buffer set for the plane / sphere / box contact-gen tests (the 5.12
// PxgRigidDbContactWriter).  The barycentric slots start at a sentinel: the vertex writers
// leave them untouched.
static const F4 kBarySentinel = { 7.0f, -7.0f, 7.0f, -7.0f };
struct WriterBufs
{
    DevBuf point, normalPen, barycentric, info, rigidMat, count, byRigid, tempByRigid, idxByRigid;
    uint32_t maxContacts;
};
static WriterBufs makeWriter(PxgSlangComputeDevice* dev, uint32_t maxContacts)
{
    WriterBufs w; w.maxContacts = maxContacts;
    std::vector<F4> zf4(maxContacts, F4{ 0, 0, 0, 0 });
    std::vector<F4> bary(maxContacts, kBarySentinel);
    std::vector<FemPairInfoHost> zinfo(maxContacts, FemPairInfoHost{ 0ull, 0ull });
    std::vector<uint32_t> zmat((maxContacts + 1u) / 2u, 0xffffffffu);   // PxU16 stream, two per word
    std::vector<uint32_t> zc = { 0u };
    std::vector<uint64_t> zu64(maxContacts, 0xffffffffffffffffull);
    std::vector<uint32_t> zu32(maxContacts, 0xffffffffu);
    w.point = makeDev(dev, zf4); w.normalPen = makeDev(dev, zf4); w.barycentric = makeDev(dev, bary);
    w.info = makeDev(dev, zinfo); w.rigidMat = makeDev(dev, zmat); w.count = makeDev(dev, zc);
    w.byRigid = makeDev(dev, zu64); w.tempByRigid = makeDev(dev, zu32);
    w.idxByRigid = makeDev(dev, zu32);
    return w;
}
static void fillWriter(ContactGenPush& p, const WriterBufs& w)
{
    p.w_outPoint = w.point.addr; p.w_outNormalPen = w.normalPen.addr; p.w_outBarycentric = w.barycentric.addr;
    p.w_outContactInfo = w.info.addr; p.w_outRigidMaterialIndex = w.rigidMat.addr; p.w_totalContactCount = w.count.addr;
    p.w_contactByRigid = w.byRigid.addr; p.w_tempContactByRigid = w.tempByRigid.addr;
    p.w_contactIndexSortedByRigid = w.idxByRigid.addr;
    p.w_maxNumContacts = w.maxContacts;
}
// The PxU16 rigid material index of contact i.
static uint32_t readRigidMat(PxgSlangComputeDevice* dev, const WriterBufs& w, uint32_t i)
{
    const std::vector<uint32_t> words = downloadBuffer<uint32_t>(dev, w.rigidMat.h, (w.maxContacts + 1u) / 2u);
    return (words[i / 2u] >> (16u * (i & 1u))) & 0xffffu;
}
static void destroyWriter(PxgSlangComputeDevice* dev, const WriterBufs& w)
{
    dev->destroyBuffer(w.point.h); dev->destroyBuffer(w.normalPen.h); dev->destroyBuffer(w.barycentric.h);
    dev->destroyBuffer(w.info.h); dev->destroyBuffer(w.rigidMat.h); dev->destroyBuffer(w.count.h);
    dev->destroyBuffer(w.byRigid.h); dev->destroyBuffer(w.tempByRigid.h); dev->destroyBuffer(w.idxByRigid.h);
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

    WriterBufs w = makeWriter(dev, 64u);

    ContactGenPush push = {};
    push.numTests = 1u;
    push.cmInputs = bCin.addr; push.transformCache = bTc.addr; push.contactDistance = bCd.addr;
    push.restDistance = bRd.addr; push.gpuShapes = bShapes.addr; push.femClothes = bDesc.addr;
    push.shapeToRigidRemap = bRemap.addr;
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
    const uint64_t clothRef0 = encClothVertex(0u, 0u);

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, w.count.h, 1)[0];
    PXG_CHECK_MSG(gCount == 1u, "plane contact-gen emitted %u contacts (expected 1)", gCount);

    const F4 gPoint = downloadBuffer<F4>(dev, w.point.h, 1)[0];
    const F4 gNP = downloadBuffer<F4>(dev, w.normalPen.h, 1)[0];
    const F4 gBary = downloadBuffer<F4>(dev, w.barycentric.h, 1)[0];
    const FemPairInfoHost gInfo = downloadBuffer<FemPairInfoHost>(dev, w.info.h, 1)[0];
    const uint32_t gMat = readRigidMat(dev, w, 0u);
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
    cmpF4("outBarycentric (a vertex contact writes none)", kBarySentinel, gBary);
    if (gByRigid != rigidNodeInd) { ++mm; std::printf("  plane contactByRigid MM got %llu ref %llu\n", (unsigned long long)gByRigid, (unsigned long long)rigidNodeInd); }
    if (gTempByRigid != uint32_t(rigidNodeInd & 0xffffffffu)) { ++mm; std::printf("  plane tempByRigid MM got %u\n", gTempByRigid); }
    if (gIdxByRigid != 0u) { ++mm; std::printf("  plane contactIndexSortedByRigid MM got %u\n", gIdxByRigid); }
    if (gInfo.ref0 != rigidNodeInd) { ++mm; std::printf("  plane info.ref0 MM got %llu\n", (unsigned long long)gInfo.ref0); }
    if (gInfo.ref1 != clothRef0) { ++mm; std::printf("  plane info.ref1 MM got 0x%llx ref 0x%llx\n", (unsigned long long)gInfo.ref1, (unsigned long long)clothRef0); }
    if (gMat != (materialIndex & 0xffffu)) { ++mm; std::printf("  plane rigid material MM got %u\n", gMat); }
    PXG_CHECK_MSG(mm == 0, "cloth_planeVertContactGen bit-exact vs CPU (%zu mismatches)", mm);

    // teeth: a wrong normal would not satisfy this human-readable sanity.
    PXG_CHECK_MSG(almostEqual(gNP.x, 0.0f, 1e-5f, 1e-6f) && almostEqual(gNP.y, 1.0f, 1e-5f, 1e-6f) &&
                  almostEqual(gNP.z, 0.0f, 1e-5f, 1e-6f) && almostEqual(gNP.w, -0.1f, 1e-5f, 1e-6f),
                  "plane contact normal ~ (0,1,0), separation ~ -0.1");

    destroyWriter(dev, w);
    dev->destroyBuffer(bVerts.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bCin.h);
    dev->destroyBuffer(bShapes.h); dev->destroyBuffer(bTc.h); dev->destroyBuffer(bCd.h);
    dev->destroyBuffer(bRd.h); dev->destroyBuffer(bRemap.h);
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
    WriterBufs w = makeWriter(dev, 64u);

    ContactGenPush push = {};
    push.numTests = 1u;
    push.cmInputs = bCin.addr; push.transformCache = bTc.addr; push.contactDistance = bCd.addr;
    push.restDistance = bRd.addr; push.gpuShapes = bShapes.addr; push.femClothes = bDesc.addr;
    push.shapeToRigidRemap = bRemap.addr;
    fillWriter(push, w);

    Kernel k("femClothContactGenBda.spv", "cloth_planeVertContactGenLaunch", {}, sizeof(ContactGenPush));
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, w.count.h, 1)[0];
    PXG_CHECK_MSG(gCount == 0u, "plane contact-gen (no penetration) emitted %u contacts (expected 0)", gCount);

    destroyWriter(dev, w);
    dev->destroyBuffer(bVerts.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bCin.h);
    dev->destroyBuffer(bShapes.h); dev->destroyBuffer(bTc.h); dev->destroyBuffer(bCd.h);
    dev->destroyBuffer(bRd.h); dev->destroyBuffer(bRemap.h);
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
    WriterBufs w = makeWriter(dev, 64u);

    ContactGenPush push = {};
    push.numTests = 1u;
    push.cmInputs = bCin.addr; push.transformCache = bTc.addr; push.contactDistance = bCd.addr;
    push.restDistance = bRd.addr; push.gpuShapes = bShapes.addr; push.femClothes = bDesc.addr;
    push.shapeToRigidRemap = bRemap.addr;
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
    const uint64_t clothRef0 = encClothVertex(0u, 0u);

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, w.count.h, 1)[0];
    PXG_CHECK_MSG(gCount == 1u, "sphere contact-gen emitted %u contacts (expected 1)", gCount);

    const F4 gPoint = downloadBuffer<F4>(dev, w.point.h, 1)[0];
    const F4 gNP = downloadBuffer<F4>(dev, w.normalPen.h, 1)[0];
    const F4 gBary = downloadBuffer<F4>(dev, w.barycentric.h, 1)[0];
    const FemPairInfoHost gInfo = downloadBuffer<FemPairInfoHost>(dev, w.info.h, 1)[0];
    const uint32_t gMat = readRigidMat(dev, w, 0u);
    const uint64_t gByRigid = downloadBuffer<uint64_t>(dev, w.byRigid.h, 1)[0];
    const uint32_t gTempByRigid = downloadBuffer<uint32_t>(dev, w.tempByRigid.h, 1)[0];

    size_t mm = 0;
    auto cmpF4 = [&](const char* nm, const F4& r, const F4& g) {
        if (f2u(r.x) != f2u(g.x) || f2u(r.y) != f2u(g.y) || f2u(r.z) != f2u(g.z) || f2u(r.w) != f2u(g.w))
        { ++mm; std::printf("  sphere %s MM: ref(%g,%g,%g,%g) got(%g,%g,%g,%g)\n", nm, r.x, r.y, r.z, r.w, g.x, g.y, g.z, g.w); }
    };
    cmpF4("outPoint", refPoint, gPoint);
    cmpF4("outNormalPen", refNormalPen, gNP);
    cmpF4("outBarycentric (a vertex contact writes none)", kBarySentinel, gBary);
    if (gByRigid != rigidNodeInd) { ++mm; std::printf("  sphere contactByRigid MM got %llu\n", (unsigned long long)gByRigid); }
    if (gTempByRigid != uint32_t(rigidNodeInd & 0xffffffffu)) { ++mm; std::printf("  sphere tempByRigid MM got %u\n", gTempByRigid); }
    if (gInfo.ref0 != rigidNodeInd) { ++mm; std::printf("  sphere info.ref0 MM got %llu\n", (unsigned long long)gInfo.ref0); }
    if (gInfo.ref1 != clothRef0) { ++mm; std::printf("  sphere info.ref1 MM got 0x%llx\n", (unsigned long long)gInfo.ref1); }
    if (gMat != (materialIndex & 0xffffu)) { ++mm; std::printf("  sphere rigid material MM got %u\n", gMat); }
    PXG_CHECK_MSG(mm == 0, "cloth_SphereVertexContactGen bit-exact vs CPU (%zu mismatches)", mm);

    PXG_CHECK_MSG(almostEqual(gNP.x, 1.0f, 1e-5f, 1e-6f) && almostEqual(gNP.w, -0.5f, 1e-5f, 1e-6f),
                  "sphere contact normal ~ (1,0,0), separation ~ -0.5");

    destroyWriter(dev, w);
    dev->destroyBuffer(bVerts.h); dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bCin.h);
    dev->destroyBuffer(bShapes.h); dev->destroyBuffer(bTc.h); dev->destroyBuffer(bCd.h);
    dev->destroyBuffer(bRd.h); dev->destroyBuffer(bRemap.h);
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
    // V3: one filter pair on the cloth, keyed (rigid filter ref, the cloth vertex V_FILTERED):
    // 5.12 keeps the rigid-surface filter per cloth (mRigidSurfaceFilterPairs, sorted
    // PxgDeformablePairInfo, binary-searched), not as a kernel argument.  A dynamic rigid's
    // filter ref is its node.
    std::vector<FemPairInfoHost> fpairs = { FemPairInfoHost{ rigidNodeInd, encClothVertex(CLOTH_ID, V_FILTERED) } };
    DevBuf bFilters = makeDev(dev, fpairs);
    std::vector<PxgFEMCloth> descArr = { makeClothDesc(bDecoy.addr, uint32_t(decoyVerts.size())),
                                         makeClothDesc(bVerts.addr, NB_VERTS) };
    descArr[1].mRigidSurfaceFilterPairs = reinterpret_cast<PxgDeformablePairInfo*>(size_t(bFilters.addr));
    descArr[1].mNumRigidSurfaceFilterPairs = 1u;
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

    WriterBufs w = makeWriter(dev, 64u);

    ContactGenPush push = {};
    push.numTests = 1u;
    push.cmInputs = bCin.addr; push.transformCache = bTc.addr; push.contactDistance = bCd.addr;
    push.restDistance = bRd.addr; push.gpuShapes = bShapes.addr; push.femClothes = bDesc.addr;
    push.shapeToRigidRemap = bRemap.addr;
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
        dev->destroyBuffer(bRd.h); dev->destroyBuffer(bRemap.h);
        dev->destroyBuffer(bDecoy.h);
        return;
    }

    const auto gPoint = downloadBuffer<F4>(dev, w.point.h, w.maxContacts);
    const auto gNP = downloadBuffer<F4>(dev, w.normalPen.h, w.maxContacts);
    const auto gBary = downloadBuffer<F4>(dev, w.barycentric.h, w.maxContacts);
    const auto gInfo = downloadBuffer<FemPairInfoHost>(dev, w.info.h, w.maxContacts);
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
        const uint64_t pid = gInfo[sidx].ref1;
        size_t at = want.size();
        for (size_t q = 0; q < want.size(); ++q)
            if (!claimed[q] && encClothVertex(CLOTH_ID, want[q].vert) == pid) { at = q; break; }
        if (at == want.size())
        { if (mm < 4) std::printf("  box slot %u names cloth vertex ref 0x%llx, which is not an "
                                  "unclaimed expected contact\n", sidx, (unsigned long long)pid);
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
        cmpF4("outBarycentric (a vertex contact writes none)", kBarySentinel, gBary[sidx]);
        if (gByRigid[sidx] != rigidNodeInd) { ++mm; std::printf("  box vert %u contactByRigid MM\n", e.vert); }
        if (gTempByRigid[sidx] != uint32_t(rigidNodeInd & 0xffffffffu)) { ++mm; std::printf("  box vert %u tempByRigid MM\n", e.vert); }
        if (gIdxByRigid[sidx] != sidx) { ++mm; std::printf("  box vert %u idxByRigid MM got %u want %u\n", e.vert, gIdxByRigid[sidx], sidx); }
        if (gInfo[sidx].ref0 != rigidNodeInd) { ++mm; std::printf("  box vert %u info.ref0 MM\n", e.vert); }
        if (readRigidMat(dev, w, sidx) != (materialIndex & 0xffffu)) { ++mm; std::printf("  box vert %u rigid material MM\n", e.vert); }
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
    dev->destroyBuffer(bRd.h); dev->destroyBuffer(bRemap.h);
    dev->destroyBuffer(bDecoy.h); dev->destroyBuffer(bFilters.h);
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

    WriterBufs w = makeWriter(dev, 16u);

    ContactGenPush push = {};
    push.numTests = 1u;
    push.cmInputs = bCin.addr; push.transformCache = bTc.addr; push.contactDistance = bCd.addr;
    push.restDistance = bRd.addr; push.gpuShapes = bShapes.addr; push.femClothes = bDesc.addr;
    push.shapeToRigidRemap = bRemap.addr;
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
        const auto gInfo = downloadBuffer<FemPairInfoHost>(dev, w.info.h, 3);
        const auto gNP = downloadBuffer<F4>(dev, w.normalPen.h, 3);
        bool sawVert0 = false; int at3 = -1;
        for (uint32_t i = 0; i < 3u; ++i)
        {
            if (gInfo[i].ref1 == encClothVertex(0u, 0u)) sawVert0 = true;
            if (gInfo[i].ref1 == encClothVertex(0u, 3u)) at3 = int(i);
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
    dev->destroyBuffer(bRd.h); dev->destroyBuffer(bRemap.h);
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
    fc.mTriangleVertexIndices_ownedEdges = reinterpret_cast<uint4*>(size_t(bCTris.addr));
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
    decoy.mTriangleVertexIndices_ownedEdges = reinterpret_cast<uint4*>(size_t(bDTris.addr));
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
// CPU gather in BOTH memory modes (the FemPairInfoHost mirror pins the 16-byte
// PxgDeformablePairInfo the shader gathers).  5.12 adds the PxU16 rigid material
// stream, gathered alongside.
// ===========================================================================

// fem_reorderRigidContactsLaunch push blob — byte-identical to
// femClothContactSortBda.slang's FemReorderParams (ABI row {8}x13 = 104 bytes, 5.12).
struct FemReorderPush
{
    uint64_t contacts, normalpens, barycentrics, contactInfos, contactMaterials, numContacts, remapByRigid;
    uint64_t sortedContacts, sortedNormalPens, sortedBarycentrics, sortedContactInfos, sortedContactMaterials,
        sortedRigidIds;
};
static_assert(sizeof(FemReorderPush) == 104, "fem reorder push blob must be 104 bytes");

// Bit-exact reorder-gather proof for a given contact count.  NC is chosen by the
// callers to cover both the single-iteration common case and the grid-stride loop
// (NC > SB_REORDERCONTACTS grid*block = 262144 forces a second loop iteration,
// exercising the CUDA-precedence-preserved numIternations term).
static void runReorderRigidContacts(uint32_t NC, uint32_t seed)
{
    std::mt19937 rng(seed);
    PxgSlangComputeDevice* dev = testDevice();

    std::vector<F4> contacts(NC), normalpens(NC), barycentrics(NC);
    std::vector<FemPairInfoHost> infos(NC);
    std::vector<uint16_t> mats(NC + 1u);
    for (uint32_t i = 0; i < NC; ++i)
    {
        // Distinct finite floats per lane (rng() value-cast, the psContactSort
        // precedent) — a wrong permutation is caught, no NaN-canonicalization risk.
        contacts[i]     = { float(rng()), float(rng()), float(rng()), float(rng()) };
        normalpens[i]   = { float(rng()), float(rng()), float(rng()), float(rng()) };
        barycentrics[i] = { float(rng()), float(rng()), float(rng()), float(rng()) };
        // The 16-byte pair info carries raw integer bit patterns (a true bit test);
        // ref0 is the PxU64 rigid id the kernel also splits out.
        infos[i].ref0 = (uint64_t(rng()) << 32) | uint64_t(rng());
        infos[i].ref1 = (uint64_t(rng()) << 32) | uint64_t(rng());
        mats[i] = uint16_t(rng());
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
    std::vector<uint32_t> matWords((NC + 1u) / 2u);
    std::memcpy(matWords.data(), mats.data(), matWords.size() * 4u);
    DevBuf bMats = makeDev(dev, matWords);
    DevBuf bSortedMats = makeDev(dev, std::vector<uint32_t>((NC + 1u) / 2u, 0xA5A5A5A5u));
    DevBuf bNum = makeDev(dev, std::vector<uint32_t>(1, NC));
    DevBuf bRemap = makeDev(dev, remapByRigid);
    DevBuf bSortedContacts = makeDev(dev, std::vector<F4>(NC));
    DevBuf bSortedNormalPens = makeDev(dev, std::vector<F4>(NC));
    DevBuf bSortedBarycentrics = makeDev(dev, std::vector<F4>(NC));
    DevBuf bSortedInfos = makeDev(dev, std::vector<FemPairInfoHost>(NC));
    DevBuf bSortedRigidIds = makeDev(dev, std::vector<uint64_t>(NC));
    if (!bContacts.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    FemReorderPush push = {
        bContacts.addr, bNormalPens.addr, bBarycentrics.addr, bInfos.addr, bMats.addr, bNum.addr, bRemap.addr,
        bSortedContacts.addr, bSortedNormalPens.addr, bSortedBarycentrics.addr,
        bSortedInfos.addr, bSortedMats.addr, bSortedRigidIds.addr };
    Kernel k("femClothContactSortBda.spv", "fem_reorderRigidContactsLaunch", {}, sizeof(FemReorderPush));
    k.dispatch(/*gridX SB_REORDERCONTACTS*/ 1024u, {}, push);
    dev->flush();

    const auto gC = downloadBuffer<F4>(dev, bSortedContacts.h, NC);
    const auto gN = downloadBuffer<F4>(dev, bSortedNormalPens.h, NC);
    const auto gB = downloadBuffer<F4>(dev, bSortedBarycentrics.h, NC);
    const auto gI = downloadBuffer<FemPairInfoHost>(dev, bSortedInfos.h, NC);
    const auto gR = downloadBuffer<uint64_t>(dev, bSortedRigidIds.h, NC);
    const auto gM = downloadBuffer<uint32_t>(dev, bSortedMats.h, (NC + 1u) / 2u);

    size_t bad = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const uint32_t r = remapByRigid[i];
        if (std::memcmp(&gC[i], &contacts[r], sizeof(F4)) != 0) ++bad;
        if (std::memcmp(&gN[i], &normalpens[r], sizeof(F4)) != 0) ++bad;
        if (std::memcmp(&gB[i], &barycentrics[r], sizeof(F4)) != 0) ++bad;
        if (std::memcmp(&gI[i], &infos[r], sizeof(FemPairInfoHost)) != 0) ++bad;
        if (gR[i] != infos[r].ref0) ++bad; // sortedRigidIds[i] == sortedInfo.ref0
        if (((gM[i / 2u] >> (16u * (i & 1u))) & 0xffffu) != mats[r]) ++bad;
    }
    PXG_CHECK_MSG(bad == 0, "cloth reorder gather bit-exact NC=%u (%zu mismatches)", NC, bad);

    dev->destroyBuffer(bContacts.h); dev->destroyBuffer(bNormalPens.h); dev->destroyBuffer(bBarycentrics.h);
    dev->destroyBuffer(bInfos.h); dev->destroyBuffer(bNum.h); dev->destroyBuffer(bRemap.h);
    dev->destroyBuffer(bSortedContacts.h); dev->destroyBuffer(bSortedNormalPens.h);
    dev->destroyBuffer(bSortedBarycentrics.h); dev->destroyBuffer(bSortedInfos.h);
    dev->destroyBuffer(bSortedRigidIds.h); dev->destroyBuffer(bMats.h); dev->destroyBuffer(bSortedMats.h);
}

// Common case: ~20 active device blocks, single grid-stride iteration.
PXG_TEST(femClothSimReorderRigidContacts) { runReorderRigidContacts(5000u, 0x5EED4B01u); }

// Grid-stride coverage: NC just past SB_REORDERCONTACTS grid*block (256*1024 =
// 262144) so the last 777 contacts are gathered on the SECOND loop iteration,
// exercising the numIternations / index += blockDim*gridDim stride verbatim.
PXG_TEST(femClothSimReorderRigidContactsMultiIter) { runReorderRigidContacts(262144u + 777u, 0x4B02F00Du); }

// ===========================================================================
// P7.2 tranche 1: cloth SELF-collision midphase + contact GENERATION.
//   femClothSelfCollisionBda.spv:
//     cloth_selfCollisionMidphaseVTLaunch  (femClothClothMidPhase.cu:1163)
//     cloth_selfCollisionMidphaseEELaunch  (femClothClothMidPhase.cu:1196)
// Both kernels walk the cloth's own BV32 triangle BVH (one 32-thread group per
// query) and append a PxgDeformablePairInfo (5.12) for every penetrating vertex-triangle
// (VT) / crossing edge-edge (EE) pair.  The midphase emits ONLY the two references
// (VT: vertex + triangle; EE: two triangle-edge refs carrying the local edge indices);
// the normal / distance / barycentric are computed later in
// the (out-of-scope) self-contact prepare.  FULLY bit-exact: the VT/EE closest-point
// tests are pure mul/add/sub/div/sqrt/min/max — no transcendental, no tolerance seam.
// Checked bit-exact vs a CPU transcription of the .cu math in BOTH memory modes.
// ===========================================================================

// self-collision push blob — byte-identical to SelfCollisionParams
// (ABI row {4,8,8,8,8,8,8} -> 56 bytes, 5.12: the clothClothPairs / numClothClothPairs
// arguments are gone and the output is a PxgDeformablePairInfo stream).
struct SelfCollisionPush
{
    uint32_t maxContacts, pad4;
    uint64_t activeClothes, contactDistance, clothes;
    uint64_t updateContactPairs, outContactInfo, totalContactCount;
};
static_assert(sizeof(SelfCollisionPush) == 56, "self-collision push blob must be 56 bytes");
static_assert(offsetof(SelfCollisionPush, updateContactPairs) == 32, "updateContactPairs @32");
static_assert(offsetof(SelfCollisionPush, totalContactCount) == 48, "totalContactCount @48");

// A four-uint triangle-index record (uint4: x/y/z vertex ids, w = edge authorship).
struct TriIdx4 { uint32_t x, y, z, w; };

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
    std::vector<TriIdx4> tris = { { 0u, 1u, 2u, 0x7u } }; // all three edges owned (VT ignores .w)
    std::vector<uint32_t> tree = makeBv32TreeWords(/*nbTriangles*/ 1u, -1000.0f, 1000.0f);

    DevBuf bPos = makeDev(dev, pos);
    DevBuf bRest = makeDev(dev, rest);
    DevBuf bTri = makeDev(dev, tris);
    DevBuf bTree = makeDev(dev, tree);

    PxgFEMCloth c = makeZeroDesc();
    c.mTriMeshData = reinterpret_cast<void*>(size_t(bTree.addr));
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bPos.addr));
    c.mRestPosition = reinterpret_cast<decltype(c.mRestPosition)>(size_t(bRest.addr));
    c.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(c.mTriangleVertexIndices_ownedEdges)>(size_t(bTri.addr));
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
    std::vector<FemPairInfoHost> outInfo(MAXC);
    std::memset(outInfo.data(), 0xAB, outInfo.size() * sizeof(FemPairInfoHost)); // poison
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

    const FemPairInfoHost gi = downloadBuffer<FemPairInfoHost>(dev, bOut.h, 1)[0];
    size_t mm = 0;
    const uint64_t refRef0 = encClothVertex(0u, 3u);     // cloth 0, vertex 3
    const uint64_t refRef1 = encClothTriangle(0u, 0u);   // cloth 0, triangle 0 (no flag bits)
    if (gi.ref0 != refRef0) { ++mm; std::printf("  VT ref0 MM got 0x%llx ref 0x%llx\n", (unsigned long long)gi.ref0, (unsigned long long)refRef0); }
    if (gi.ref1 != refRef1) { ++mm; std::printf("  VT ref1 MM got 0x%llx ref 0x%llx\n", (unsigned long long)gi.ref1, (unsigned long long)refRef1); }
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
    std::vector<TriIdx4> tris = { { 0u, 1u, 2u, 0x7u }, { 3u, 4u, 5u, 0x7u } }; // all edges owned (5.12 ownsEdge bits)
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
    c.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(c.mTriangleVertexIndices_ownedEdges)>(size_t(bTri.addr));
    c.mTrianglesWithOwnedEdges = reinterpret_cast<decltype(c.mTrianglesWithOwnedEdges)>(size_t(bTwae.addr));
    c.mNbTrianglesWithOwnedEdges = 2u;
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
    std::vector<FemPairInfoHost> outInfo(MAXC);
    std::memset(outInfo.data(), 0xAB, outInfo.size() * sizeof(FemPairInfoHost));
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
    std::vector<FemPairInfoHost> refContacts;
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
                // filterDistSq 0, distinct rest positions -> passes; 5.12 encodes both
                // sides as eTRIANGLE_EDGE refs (query triangle 1, found triangle 0).
                refContacts.push_back(FemPairInfoHost{ encClothEdge(0u, 1u, e0), encClothEdge(0u, 0u, e1) });
            }
        }
    }
    PXG_CHECK_MSG(refContacts.size() == 2u, "reference: expected 2 EE crossings, got %zu", refContacts.size());

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
    PXG_CHECK_MSG(gCount == uint32_t(refContacts.size()), "EE self-collision emitted %u contacts (expected %zu)",
                  gCount, refContacts.size());

    const auto gi = downloadBuffer<FemPairInfoHost>(dev, bOut.h, gCount ? gCount : 1u);
    size_t mm = 0;
    for (uint32_t i = 0; i < gCount && i < refContacts.size(); ++i)
    {
        if (std::memcmp(&gi[i], &refContacts[i], sizeof(FemPairInfoHost)) != 0)
        {
            ++mm;
            std::printf("  EE contact[%u] MM: got(0x%llx, 0x%llx) ref(0x%llx, 0x%llx)\n", i,
                        (unsigned long long)gi[i].ref0, (unsigned long long)gi[i].ref1,
                        (unsigned long long)refContacts[i].ref0, (unsigned long long)refContacts[i].ref1);
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

// cloth_updateClothContactValidityLaunch push blob (femClothClothContactBda.slang
// UpdateValidityParams, ABI row {8,8,1,1,8,8,8,4}): the two bools are bytes 16 / 17 of one
// word; the word at @20 is padding the kernel never reads.
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

// Every PhysX tree has mMin[k].w == mMax[k].w == 0.0f in its VALID child slots
// (BV32Tree::createSOAformatNode writes PxVec4(min, 0.f); the refit itself writes
// make_float4(min, 0.f)).  The multi-workgroup refit keeps its per-node arrival counter
// in mMin[0].w and relies on that, so the poisoned fixtures keep those words cooked.
static void zeroValidW(TreeBlob& tree, uint32_t nbNodes)
{
    for (uint32_t n = 0; n < nbNodes; ++n)
        for (uint32_t k = 0; k < tree.w[tree.nodeWord(n) + W_NBNODES]; ++k)
        {
            tree.w[tree.nodeWord(n) + W_MMIN + k * 4u + 3u] = 0u;
            tree.w[tree.nodeWord(n) + W_MMAX + k * 4u + 3u] = 0u;
        }
}

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

    zeroValidW(tree, nbNodes);

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
    c.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(c.mTriangleVertexIndices_ownedEdges)>(size_t(bTri.addr));
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

    zeroValidW(tree, nbNodes);

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
    c.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(c.mTriangleVertexIndices_ownedEdges)>(size_t(bTri.addr));
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

// ---------------------------------------------------------------------------
// cloth_refitBoundLaunch over a WIDE tree: more nodes per depth level than the kernel has
// warps.  The two cases above have at most two nodes per level, so they cannot tell a
// group that walks a level's nodes in parallel warps from one that walks them serially.
// This tree has 3 levels with 1, 6 and 75 nodes (the bottom level is two full 32-node
// trips plus a partial one), node leaf counts from 1 to 32 with gaps in the slot pattern,
// leaf sizes from 1 to 32 triangles, a level-1 node with leaf AND internal children, and
// a shuffled remap -- so the leaf walks of the warps sharing a trip have different trip
// counts and different slots.  Bit-exact against the same CPU transcription, node by node.
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimRefitBoundWide)
{
    using namespace p72refit;
    PxgSlangComputeDevice* dev = testDevice();

    uint32_t rng = 0x2545F491u;
    auto next = [&rng]() { rng = rng * 1664525u + 1013904223u; return rng >> 8; };
    auto frand = [&]() { return float(next() % 200001u) / 100000.0f - 1.0f; };

    const uint32_t nbMid = 6u, nbBottom = 75u;
    const uint32_t nbNodes = 1u + nbMid + nbBottom, maxDepth = 3u;
    // node numbering: 0 = root, 1..6 = level 1, 7..81 = level 2
    std::vector<TriIdx4> tris;
    std::vector<F4> pos;
    auto addLeaf = [&](uint32_t nbPrims) {
        const uint32_t start = uint32_t(tris.size());
        for (uint32_t p = 0; p < nbPrims; ++p)
        {
            const uint32_t v = uint32_t(pos.size());
            for (uint32_t q = 0; q < 3u; ++q)
                pos.push_back(F4{ frand() * 4.0f, frand() * 4.0f, frand() * 4.0f, 0.5f });
            tris.push_back(TriIdx4{ v, v + 1u, v + 2u, 0u });
        }
        return leafWord(start, nbPrims);
    };

    TreeBlob tree;
    tree.w.assign(4u + nbNodes * BV32_WORDS + maxDepth * 2u + nbNodes, 0u);
    const float POISON = -7777.0f;
    for (uint32_t n = 0; n < nbNodes; ++n)
        for (uint32_t k = 0; k < 32u * 4u; ++k)
        {
            tree.w[tree.nodeWord(n) + W_MMIN + k] = f2u(POISON);
            tree.w[tree.nodeWord(n) + W_MMAX + k] = f2u(POISON);
        }
    auto setChild = [&](uint32_t node, uint32_t slot, uint32_t word) { tree.w[tree.nodeWord(node) + W_MDATA + slot] = word; };

    // root: the six level-1 nodes
    for (uint32_t c = 0; c < nbMid; ++c) setChild(0u, c, internalWord(1u + c));
    tree.w[tree.nodeWord(0) + W_NBNODES] = nbMid;
    // level 1: the bottom nodes dealt out 15/14/13/12/11/10; node 1 also gets three LEAF
    // children in slots between its internal ones
    uint32_t bottom = 1u + nbMid;
    for (uint32_t m = 0; m < nbMid; ++m)
    {
        const uint32_t node = 1u + m, nbInternal = 15u - m;
        uint32_t slot = 0u;
        for (uint32_t c = 0; c < nbInternal; ++c)
        {
            if (m == 0u && (c == 2u || c == 5u || c == 9u))
                setChild(node, slot++, addLeaf(1u + c));
            setChild(node, slot++, internalWord(bottom++));
        }
        tree.w[tree.nodeWord(node) + W_NBNODES] = slot;
    }
    PXG_CHECK_MSG(bottom == nbNodes, "wide tree: %u bottom nodes dealt, expected %u", bottom - 1u - nbMid, nbBottom);
    // level 2: leaf children only, 1..32 of them, leaf sizes 1..32
    for (uint32_t b = 1u + nbMid; b < nbNodes; ++b)
    {
        const uint32_t nbLeaves = (b * 7u) % 32u + 1u;
        for (uint32_t c = 0; c < nbLeaves; ++c)
            setChild(b, c, addLeaf((b + c * 5u) % 32u + 1u));
        tree.w[tree.nodeWord(b) + W_NBNODES] = nbLeaves;
    }
    tree.w[0] = uint32_t(pos.size());
    tree.w[1] = uint32_t(tris.size());
    tree.w[2] = maxDepth;
    tree.w[3] = nbNodes;

    // depth info {offset, count} and a remap that is shuffled within each level
    const uint32_t wDepthInfo = 4u + nbNodes * BV32_WORDS;
    const uint32_t levelOffset[3] = { 0u, 1u, 1u + nbMid }, levelCount[3] = { 1u, nbMid, nbBottom };
    const uint32_t wRemap = wDepthInfo + maxDepth * 2u;
    for (uint32_t l = 0; l < 3u; ++l)
    {
        tree.w[wDepthInfo + l * 2u + 0] = levelOffset[l];
        tree.w[wDepthInfo + l * 2u + 1] = levelCount[l];
        std::vector<uint32_t> ids(levelCount[l]);
        for (uint32_t i = 0; i < levelCount[l]; ++i) ids[i] = levelOffset[l] + i;
        for (uint32_t i = levelCount[l]; i > 1u; --i) std::swap(ids[i - 1u], ids[next() % i]);
        for (uint32_t i = 0; i < levelCount[l]; ++i) tree.w[wRemap + levelOffset[l] + i] = ids[i];
    }

    zeroValidW(tree, nbNodes);

    // --- CPU reference ---------------------------------------------------------------
    TreeBlob refTree = tree;
    std::vector<Bounds> refPacked(nbNodes, Bounds{ p72ref::V3{ POISON, POISON, POISON }, p72ref::V3{ POISON, POISON, POISON } });
    for (uint32_t i = maxDepth; i > 0u; --i)
    {
        const uint32_t off = refTree.w[wDepthInfo + (i - 1u) * 2u + 0];
        const uint32_t cnt = refTree.w[wDepthInfo + (i - 1u) * 2u + 1];
        for (uint32_t j = 0; j < cnt; ++j)
            refitNodeRef(refTree.w[wRemap + j + off], refTree, refPacked, pos, tris);
    }

    // --- device --------------------------------------------------------------------------
    const uint32_t clothId = 0u, elemIdx = 1u;
    const float origContactOffset = 0.021f, specCCD = 0.004f, nbUpdatesPerStep = 1.0f;
    std::vector<float> packedBoundsInit(nbNodes * 6u, POISON);
    std::vector<float> boundArrayInit((elemIdx + 1u) * 6u, POISON);
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
    c.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(c.mTriangleVertexIndices_ownedEdges)>(size_t(bTri.addr));
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

    size_t mm = 0, printed = 0;
    for (uint32_t n = 0; n < nbNodes; ++n)
    {
        const uint32_t nw = refTree.nodeWord(n);
        for (uint32_t k2 = 0; k2 < 32u * 4u; ++k2)
            for (uint32_t base : { W_MMIN, W_MMAX })
                if (gTree[nw + base + k2] != refTree.w[nw + base + k2])
                {
                    ++mm;
                    if (printed++ < 16)
                        std::printf("  node %u %s word %u MM: ref %08x got %08x\n", n, base == W_MMIN ? "mMin" : "mMax",
                                    k2, refTree.w[nw + base + k2], gTree[nw + base + k2]);
                }
        const float r[6] = { refPacked[n].mn.x, refPacked[n].mn.y, refPacked[n].mn.z,
                             refPacked[n].mx.x, refPacked[n].mx.y, refPacked[n].mx.z };
        for (uint32_t e = 0; e < 6u; ++e)
            if (f2u(gPacked[n * 6 + e]) != f2u(r[e]))
            {
                ++mm;
                if (printed++ < 16)
                    std::printf("  packedNodeBounds[%u][%u] MM: ref %08x got %08x\n", n, e, f2u(r[e]), f2u(gPacked[n * 6 + e]));
            }
    }
    const float expansion = (nbUpdatesPerStep > 1.0f ? nbUpdatesPerStep : 1.0f) * (origContactOffset + specCCD);
    const float refBound[6] = {
        refPacked[0].mn.x - expansion, refPacked[0].mn.y - expansion, refPacked[0].mn.z - expansion,
        refPacked[0].mx.x + expansion, refPacked[0].mx.y + expansion, refPacked[0].mx.z + expansion };
    for (uint32_t e = 0; e < 6u; ++e)
    {
        if (f2u(gBounds[elemIdx * 6 + e]) != f2u(refBound[e]))
        { ++mm; std::printf("  boundArray[%u][%u] MM: ref %g got %g\n", elemIdx, e, refBound[e], gBounds[elemIdx * 6 + e]); }
        if (f2u(gBounds[e]) != f2u(POISON)) { ++mm; std::printf("  boundArray[0][%u] clobbered\n", e); }
    }
    PXG_CHECK_MSG(f2u(gPacked[0]) != f2u(POISON), "root packedNodeBounds must be overwritten (refit did not run)");
    PXG_CHECK_MSG(mm == 0, "cloth_refitBoundLaunch wide tree (%u nodes, %zu triangles) bit-exact vs CPU (%zu mismatches)",
                  nbNodes, tris.size(), mm);

    dev->destroyBuffer(bPos.h); dev->destroyBuffer(bTri.h); dev->destroyBuffer(bTree.h);
    dev->destroyBuffer(bPacked.h); dev->destroyBuffer(bBounds.h); dev->destroyBuffer(bActive.h);
    dev->destroyBuffer(bElem.h); dev->destroyBuffer(bSpec.h); dev->destroyBuffer(bCd.h);
    dev->destroyBuffer(bUpd.h); dev->destroyBuffer(bDesc.h);
}

// ---------------------------------------------------------------------------
// cloth_refitBoundLaunch's multi-workgroup schedule (femClothRefitBda.slang header):
// grid (nbActive, nbSlices), one warp per group, and a node is refit by the LAST of its
// internal children to arrive (a device-scope atomic counter in mMin[0].w), never
// behind a level barrier.  The cases above run one slice (gridY = 1), where one group
// walks the whole tree; this one drives the schedule the engine launches and the ways
// it can break:
//   - a parent refit before all of its children finished would read a stale child
//     bound.  Every launch after the first moves every vertex, so a stale bound is the
//     PREVIOUS launch's and cannot match the reference by accident.  This catches a
//     parent refit that is NOT redone later (wrong child count, see the mutations in
//     the shader review).  It cannot catch an EXTRA early refit: "refit on first
//     arrival" or "every arrival refits" still pass, because each early refit resets
//     the counter and the chronologically last arrival refits again from complete
//     children; only arrivals overlapping a parent refit (real GPU concurrency, which
//     llvmpipe practically never produces) expose them.  Missing release/acquire
//     barriers are invisible on llvmpipe as well (coherent CPU memory);
//   - the counter must be back to 0.0f after every launch (the parent's own
//     mMin[0] = float4(min, 0) store is the reset): five launches in a row, with 1, 3,
//     256 (the engine's), 2000 (more slices than nodes) and 256 slices again;
//   - the trees: the SnippetDeformableSurface shape (levels of 1, 32 and 1024 nodes), a
//     12-level unbalanced chain whose nodes have ONE internal child among leaves, with
//     leaf-only side nodes refit at launch start next to the deepest node, one of them
//     filed two levels below its depth (the depth info only promises "deeper"), a
//     random tree, and a one-node tree (the root refits itself at once).  Four cloths
//     in one launch, activated out of order, and a fifth INACTIVE one that must stay
//     untouched;
//   - the parent-link cache in padding[0] starts as garbage, including real internal
//     slots of the WRONG node, and is scrambled again before the third launch so the
//     parent search also runs under 256 concurrent slices; after every launch it must
//     hold each node's true parent link.
// Every word of every tree (mMin/mMax incl. the w words and the poisoned unused slots,
// mData, mNbNodes, mDepth, padding[1], depth info, remap), mPackedNodeBounds and the
// broad-phase slots are compared bit-exact against the CPU transcription, launch by
// launch.
// ---------------------------------------------------------------------------
PXG_TEST(femClothSimRefitBoundLastArriver)
{
    using namespace p72refit;
    namespace g = pxgbv32gen;
    PxgSlangComputeDevice* dev = testDevice();
    g::Rng rng{ 0x6A09E667u };
    const float POISON = -7777.0f;

    std::vector<g::Spec> specs;
    specs.push_back(g::wideSpec(32u, 32u, rng));   // levels 1 / 32 / 1024
    specs.push_back(g::chainSpec(12u, rng));       // 12 levels, unbalanced
    specs.push_back(g::randomSpec(5u, rng));
    specs.push_back(g::singleSpec(rng));           // the root alone
    specs.push_back(g::randomSpec(4u, rng));       // INACTIVE
    const uint32_t nbBodies = uint32_t(specs.size());
    const std::vector<uint32_t> active = { 3u, 0u, 2u, 1u };
    const std::vector<uint32_t> elemIndices = { 5u, 1u, 7u, 3u, 0u };
    const uint32_t nbSlots = 9u;

    struct Body
    {
        g::Tree tree;
        std::vector<F4> pos;
        std::vector<TriIdx4> tris;
        TreeBlob ref;
        std::vector<Bounds> refPacked;
        DevBuf bPos, bTri, bTree, bPacked;
        bool isActive = false;
    };
    std::vector<Body> bodies(nbBodies);
    auto randomPos = [&](Body& b) {
        for (F4& p : b.pos) p = F4{ rng.unit() * 5.0f, rng.unit() * 5.0f, rng.unit() * 5.0f, 0.5f };
    };
    uint32_t wrongLinks = 0u;
    for (uint32_t i = 0; i < nbBodies; ++i)
    {
        Body& b = bodies[i];
        b.tree = g::build(specs[i], rng, POISON);
        const uint32_t nv = 3u + b.tree.nbPrims;
        b.pos.resize(nv);
        randomPos(b);
        for (uint32_t t = 0; t < b.tree.nbPrims; ++t)
            b.tris.push_back(TriIdx4{ rng.below(nv), rng.below(nv), rng.below(nv), 0u });
        b.tree.w[0] = nv;
        b.ref.w = b.tree.w;
        b.refPacked.assign(b.tree.nbNodes, Bounds{ p72ref::V3{ POISON, POISON, POISON }, p72ref::V3{ POISON, POISON, POISON } });
        b.isActive = std::find(active.begin(), active.end(), i) != active.end();
        for (uint32_t n = 0; n < b.tree.nbNodes; ++n)
        {
            const uint32_t link = b.tree.w[b.tree.nodeWord(n) + g::W_PADDING0];
            const uint32_t p = link >> 5, k = link & 31u;
            if (link != b.tree.parentLink[n] && p < b.tree.nbNodes && k < b.tree.w[b.tree.nodeWord(p) + g::W_NBNODES] &&
                (b.tree.w[b.tree.nodeWord(p) + g::W_MDATA + k] & 1u) == 0u)
                ++wrongLinks;
        }
    }
    // the shapes this case claims
    const g::Tree& wide = bodies[0].tree;
    PXG_CHECK_MSG(wide.maxDepth == 3u && wide.levelCount[0] == 1u && wide.levelCount[2] == 1024u,
                  "wide tree must have levels of 1 and 1024 nodes (%u levels)", wide.maxDepth);
    PXG_CHECK_MSG(bodies[1].tree.maxDepth >= 12u, "chain tree must be at least 12 levels deep (%u)", bodies[1].tree.maxDepth);
    PXG_CHECK_MSG(bodies[3].tree.nbNodes == 1u, "single-node tree must be one node");
    PXG_CHECK_MSG(wrongLinks > 0u, "the padding garbage must include links to a real slot of the wrong node");

    for (Body& b : bodies)
    {
        b.bPos = makeDev(dev, b.pos);
        b.bTri = makeDev(dev, b.tris);
        b.bTree = makeDev(dev, b.tree.w);
        b.bPacked = makeDev(dev, std::vector<float>(b.tree.nbNodes * 6u, POISON));
    }
    std::vector<float> specCCD(nbBodies), origOffset(nbBodies);
    std::vector<PxgFEMCloth> descArr;
    for (uint32_t i = 0; i < nbBodies; ++i)
    {
        specCCD[i] = 0.001f * float(i + 1u);
        origOffset[i] = 0.013f * float(i + 2u);
        PxgFEMCloth c = makeZeroDesc();
        c.mTriMeshData = reinterpret_cast<decltype(c.mTriMeshData)>(size_t(bodies[i].bTree.addr));
        c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bodies[i].bPos.addr));
        c.mTriangleVertexIndices_ownedEdges =
            reinterpret_cast<decltype(c.mTriangleVertexIndices_ownedEdges)>(size_t(bodies[i].bTri.addr));
        c.mPackedNodeBounds = reinterpret_cast<decltype(c.mPackedNodeBounds)>(size_t(bodies[i].bPacked.addr));
        c.mNbVerts = uint32_t(bodies[i].pos.size());
        c.mNbTriangles = uint32_t(bodies[i].tris.size());
        c.mOriginalContactOffset = origOffset[i];
        descArr.push_back(c);
    }
    DevBuf bDesc = makeDev(dev, descArr);
    DevBuf bActive = makeDev(dev, active);
    DevBuf bElem = makeDev(dev, elemIndices);
    DevBuf bSpec = makeDev(dev, specCCD);
    DevBuf bBounds = makeDev(dev, std::vector<float>(nbSlots * 6u, POISON));
    DevBuf bCd = makeDev(dev, std::vector<float>(nbBodies, 0.0f));
    DevBuf bUpd = makeDev(dev, std::vector<uint32_t>(1, 1u));

    const float nbUpdatesPerStep = 2.0f;
    RefitPush rp{};
    rp.femClothes = bDesc.addr;
    rp.activeFemClothes = bActive.addr;
    rp.nbActive = uint32_t(active.size());
    rp.contactDists = bCd.addr;
    rp.speculativeCCD = bSpec.addr;
    rp.boundArray = bBounds.addr;
    rp.elemIndex = bElem.addr;
    rp.nbCollisionPairUpdatesPerTimestep = nbUpdatesPerStep;
    rp.updateContactPairs = bUpd.addr;

    Kernel k("femClothRefitBda.spv", "cloth_refitBoundLaunch", {}, sizeof(RefitPush));
    const uint32_t slicesPerLaunch[] = { 1u, 3u, 256u, 2000u, 256u };
    std::vector<float> refBounds(nbSlots * 6u, POISON);
    for (uint32_t launch = 0; launch < 5u; ++launch)
    {
        const uint32_t slices = slicesPerLaunch[launch];
        for (uint32_t i = 0; i < nbBodies && launch > 0u; ++i)
        {
            if (!bodies[i].isActive) continue;
            randomPos(bodies[i]);
            dev->upload(bodies[i].bPos.h, bodies[i].pos.data(), bodies[i].pos.size() * sizeof(F4));
            if (launch == 2u)   // fresh garbage in the parent-link cache, searched by 256 slices
                for (uint32_t n = 0; n < bodies[i].tree.nbNodes; ++n)
                {
                    const uint32_t gword = (n & 1u) ? rng.next() : 0xFFFFFFFFu;
                    dev->upload(bodies[i].bTree.h, &gword, 4u, size_t(bodies[i].tree.nodeWord(n) + g::W_PADDING0) * 4u);
                }
        }
        k.dispatch(uint32_t(active.size()), {}, rp, slices, 1u);
        dev->flush();

        size_t mm = 0, printed = 0;
        for (uint32_t i = 0; i < nbBodies; ++i)
        {
            Body& b = bodies[i];
            const g::Tree& t = b.tree;
            if (b.isActive)
            {
                for (uint32_t l = t.maxDepth; l > 0u; --l)
                {
                    const uint32_t off = t.w[t.wDepthInfo + (l - 1u) * 2u + 0];
                    const uint32_t cnt = t.w[t.wDepthInfo + (l - 1u) * 2u + 1];
                    for (uint32_t j = 0; j < cnt; ++j)
                        refitNodeRef(t.w[t.wRemap + off + j], b.ref, b.refPacked, b.pos, b.tris);
                }
                const p72ref::V3 rn = b.refPacked[0].mn, rx = b.refPacked[0].mx;
                const float expansion = (nbUpdatesPerStep > 1.0f ? nbUpdatesPerStep : 1.0f) * (origOffset[i] + specCCD[i]);
                const float r6[6] = { rn.x - expansion, rn.y - expansion, rn.z - expansion,
                                      rx.x + expansion, rx.y + expansion, rx.z + expansion };
                for (uint32_t e = 0; e < 6u; ++e) refBounds[elemIndices[i] * 6u + e] = r6[e];
            }
            const auto gTree = downloadBuffer<uint32_t>(dev, b.bTree.h, t.w.size());
            const auto gPacked = downloadBuffer<float>(dev, b.bPacked.h, t.nbNodes * 6u);
            for (uint32_t w = 0; w < uint32_t(t.w.size()); ++w)
            {
                const bool pad0 = w >= 4u && w < t.wDepthInfo && (w - 4u) % g::BV32_WORDS == g::W_PADDING0;
                uint32_t want = b.ref.w[w];
                if (pad0 && b.isActive)
                {
                    const uint32_t n = (w - 4u) / g::BV32_WORDS;
                    if (n == 0u) continue;   // the root has no parent; its padding is never read
                    want = t.parentLink[n];
                }
                if (gTree[w] != want)
                {
                    ++mm;
                    if (printed++ < 16)
                        std::printf("  launch %u (%u slices) body %u tree word %u (node %u word %u) MM: ref %08x got %08x\n",
                                    launch, slices, i, w, w >= 4u ? (w - 4u) / g::BV32_WORDS : 0u,
                                    w >= 4u ? (w - 4u) % g::BV32_WORDS : w, want, gTree[w]);
                }
            }
            for (uint32_t n = 0; n < t.nbNodes; ++n)
            {
                const float r[6] = { b.refPacked[n].mn.x, b.refPacked[n].mn.y, b.refPacked[n].mn.z,
                                     b.refPacked[n].mx.x, b.refPacked[n].mx.y, b.refPacked[n].mx.z };
                for (uint32_t e = 0; e < 6u; ++e)
                    if (f2u(gPacked[n * 6u + e]) != f2u(r[e]))
                    {
                        ++mm;
                        if (printed++ < 16)
                            std::printf("  launch %u (%u slices) body %u packedNodeBounds[%u][%u] MM: ref %08x got %08x\n",
                                        launch, slices, i, n, e, f2u(r[e]), f2u(gPacked[n * 6u + e]));
                    }
            }
        }
        const auto gBounds = downloadBuffer<float>(dev, bBounds.h, nbSlots * 6u);
        for (uint32_t e = 0; e < nbSlots * 6u; ++e)
            if (f2u(gBounds[e]) != f2u(refBounds[e]))
            {
                ++mm;
                if (printed++ < 16)
                    std::printf("  launch %u boundArray[%u][%u] MM: ref %08x got %08x\n", launch, e / 6u, e % 6u,
                                f2u(refBounds[e]), f2u(gBounds[e]));
            }
        PXG_CHECK_MSG(mm == 0, "cloth_refitBoundLaunch last-arriver launch %u (%u slices) bit-exact vs CPU (%zu mismatches)",
                      launch, slices, mm);
    }

    for (Body& b : bodies)
    {
        dev->destroyBuffer(b.bPos.h); dev->destroyBuffer(b.bTri.h);
        dev->destroyBuffer(b.bTree.h); dev->destroyBuffer(b.bPacked.h);
    }
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bActive.h); dev->destroyBuffer(bElem.h);
    dev->destroyBuffer(bSpec.h); dev->destroyBuffer(bBounds.h); dev->destroyBuffer(bCd.h);
    dev->destroyBuffer(bUpd.h);
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
    cloth.mRestOffset = 0.0f;
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
// mMaxLinearVelocity regime that measures against 0.8 * mRestOffset instead of a velocity.
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
    // position but y = 12, and mMaxLinearVelocity is -1 with mRestOffset 5, so the cap is
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
    cA.mRestOffset = 5.0f;             // -> maxDistance = 4
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
    cC.mRestOffset = 5.0f;             // -> maxDistance = 0.8 * 5 = 4
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
                  "cloth C: the cap is 0.8 * mRestOffset = 4 and the trajectory is measured "
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
//   * the 72-byte narrowphase-driven blob (cmInputs / gpuShapes / transformCache instead
//     of an activeClothes list; 5.12 dropped 5.6.1's unread clothClothPairs /
//     numClothClothPairs), and that the two cloth ids come from
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
//   * 5.12: the records are PxgDeformablePairInfo -- VT (vertex of MY cloth, triangle of
//     the OTHER), EE two triangle-edge refs (MY triangle t0 / local edge e0, the OTHER's
//     primitive / e1).  5.6.1's markValidity bit is gone with the auxInd field.  The
//     per-cloth mSurfaceSurfaceFilterPairs are empty here (makeZeroDesc).
//
// The rest-pose filter is absent by construction (the call passes NULL rest buffers and
// filterDistSq 0), so the descriptors below leave mRestPosition null -- a port that kept
// the self-collision lookup would fault on the first contact rather than mis-filter.
// ===========================================================================
namespace p816dc
{
// Byte-identical to DifferentClothParams (ABI row kClothDifferentCloth {4,4,8,8,8,8,8,8,8,8}).
struct DifferentClothPush
{
    uint32_t maxContacts;
    float toleranceLength;
    uint64_t cmInputs, transformCache, contactDistance, gpuShapes, femClothes;
    uint64_t updateContactPairs, outContactInfo, totalContactCount;
};
static_assert(sizeof(DifferentClothPush) == 72, "different-cloth push blob must be 72 bytes");
static_assert(offsetof(DifferentClothPush, cmInputs) == 8, "cmInputs @8");
static_assert(offsetof(DifferentClothPush, femClothes) == 40, "femClothes @40");
static_assert(offsetof(DifferentClothPush, updateContactPairs) == 48, "updateContactPairs @48");
static_assert(offsetof(DifferentClothPush, totalContactCount) == 64, "totalContactCount @64");

struct CmInputHost { uint32_t shapeRef0, shapeRef1, cacheRef0, cacheRef1; };
struct PxgShapeHost
{
    float sx, sy, sz, srx, sry, srz, srw;
    uint32_t materialIndex, hullLo, hullHi, type, particleOrSoftbodyId;
};
static_assert(sizeof(PxgShapeHost) == 48, "PxgShape mirror must be 48 bytes");
static const uint32_t GEOM_TRIANGLEMESH_H = uint32_t(physx::PxGeometryType::eTRIANGLEMESH);   // PxGeometryType::eTRIANGLEMESH

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
static bool infoLess(const FemPairInfoHost& a, const FemPairInfoHost& b)
{
    if (a.ref0 != b.ref0) return a.ref0 < b.ref0;
    return a.ref1 < b.ref1;
}

static size_t compareSets(std::vector<FemPairInfoHost> got,
                          std::vector<FemPairInfoHost> ref, const char* tag)
{
    std::sort(got.begin(), got.end(), infoLess);
    std::sort(ref.begin(), ref.end(), infoLess);
    size_t mm = 0;
    for (size_t i = 0; i < got.size() && i < ref.size(); ++i)
    {
        if (std::memcmp(&got[i], &ref[i], sizeof(FemPairInfoHost)) != 0)
        {
            ++mm;
            std::printf("  %s contact[%zu] MM: got(r0=0x%016llx,r1=0x%016llx) ref(r0=0x%016llx,r1=0x%016llx)\n",
                        tag, i, (unsigned long long)got[i].ref0, (unsigned long long)got[i].ref1,
                        (unsigned long long)ref[i].ref0, (unsigned long long)ref[i].ref1);
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
    c0.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(c0.mTriangleVertexIndices_ownedEdges)>(size_t(bTri0.addr));
    c0.mNbVerts = uint32_t(pos0.size());
    c0.mElementIndex = 0u;   // unused on this path (contactDist comes from the cache refs)

    PxgFEMCloth c1 = makeZeroDesc();
    c1.mTriMeshData = reinterpret_cast<void*>(size_t(bTree1.addr));
    c1.mPosition_InvMass = reinterpret_cast<decltype(c1.mPosition_InvMass)>(size_t(bPos1.addr));
    c1.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(c1.mTriangleVertexIndices_ownedEdges)>(size_t(bTri1.addr));
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
    std::vector<FemPairInfoHost> outInfo(MAXC);
    std::memset(outInfo.data(), 0xAB, outInfo.size() * sizeof(FemPairInfoHost));   // poison
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
    std::vector<FemPairInfoHost> ref;
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
            ref.push_back(FemPairInfoHost{ encClothVertex(myId, vertexIdx), encClothTriangle(otherId, p) });
        }
    }
    PXG_CHECK_MSG(ref.size() == 2u, "reference: expected 2 bidirectional VT contacts, got %zu", ref.size());
    // Pin what each of the two IS, so a reference that drifts with the shader is caught.
    PXG_CHECK_MSG(ref[0].ref0 == encClothVertex(0u, 4u) && ref[0].ref1 == encClothTriangle(1u, 0u),
                  "reference[0] must be cloth0 vertex 4 vs cloth1 triangle 0");
    PXG_CHECK_MSG(ref[1].ref0 == encClothVertex(1u, 0u) && ref[1].ref1 == encClothTriangle(0u, 0u),
                  "reference[1] must be cloth1 vertex 0 vs cloth0 triangle 0 (the corner-index case)");

    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCount.h, 1)[0];
    PXG_CHECK_MSG(gCount == uint32_t(ref.size()), "different-cloth VT emitted %u contacts (expected %zu)",
                  gCount, ref.size());
    const auto got = downloadBuffer<FemPairInfoHost>(dev, bOut.h, gCount ? gCount : 1u);
    const size_t mm = compareSets(std::vector<FemPairInfoHost>(got.begin(), got.begin() + gCount),
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
    c0.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(c0.mTriangleVertexIndices_ownedEdges)>(size_t(bTri0.addr));
    c0.mTrianglesWithOwnedEdges = reinterpret_cast<decltype(c0.mTrianglesWithOwnedEdges)>(size_t(bTwae0.addr));
    c0.mNbTrianglesWithOwnedEdges = uint32_t(twae0.size());
    c0.mElementIndex = 0u;

    PxgFEMCloth c1 = makeZeroDesc();
    c1.mTriMeshData = reinterpret_cast<void*>(size_t(bTree1.addr));
    c1.mPosition_InvMass = reinterpret_cast<decltype(c1.mPosition_InvMass)>(size_t(bPos1.addr));
    c1.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(c1.mTriangleVertexIndices_ownedEdges)>(size_t(bTri1.addr));
    c1.mTrianglesWithOwnedEdges = reinterpret_cast<decltype(c1.mTrianglesWithOwnedEdges)>(size_t(bTwae1.addr));
    c1.mNbTrianglesWithOwnedEdges = uint32_t(twae1.size());
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
    std::vector<FemPairInfoHost> outInfo(MAXC);
    std::memset(outInfo.data(), 0xAB, outInfo.size() * sizeof(FemPairInfoHost));
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
    std::vector<FemPairInfoHost> ref;
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
                ref.push_back(FemPairInfoHost{ encClothEdge(0u, 0u, e0), encClothEdge(1u, p, e1) });
            }
        }
    }
    PXG_CHECK_MSG(ref.size() == 2u, "reference: expected 2 EE crossings against cloth 1's triangle 1, got %zu",
                  ref.size());
    PXG_CHECK_MSG((ref[0].ref0 & ~(3ull << 27)) == encClothEdge(0u, 0u, 0u) &&
                  (ref[0].ref1 & ~(3ull << 27)) == encClothEdge(1u, 1u, 0u),
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
        const auto got = downloadBuffer<FemPairInfoHost>(dev, bOut.h, gCount ? gCount : 1u);
        const size_t mm = compareSets(std::vector<FemPairInfoHost>(got.begin(), got.begin() + gCount),
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
// P9.89: every fold in triangletriangle.cuh is PxMin/PxMax<float>, i.e. selectMin(a, b) =
// a < b ? a : b and selectMax(a, b) = a > b ? a : b (PxMath.h:82-100,
// PxUnixMathIntrinsics.h:116-125).  An equal compare returns b, so a +-0 tie keeps b's zero, and
// a NaN in either operand fails the compare, so b is returned: a NaN b sticks, a NaN a is
// dropped.  This mirror used std::fmin/std::fmax before, which is the OLD port's spelling
// (GLSL.std.450 FMin/FMax: the non-NaN operand, -0/+0 on a zero tie), so it could not see the
// difference.  `revert` replays the old spelling one site at a time (P89SatSite); 0 is the CUDA.
static float selMin(float a, float b) { return a < b ? a : b; }
static float selMax(float a, float b) { return a > b ? a : b; }
static float oldFMin(float a, float b)
{
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a == 0.0f && b == 0.0f) return (std::signbit(a) || std::signbit(b)) ? -0.0f : 0.0f;
    return a < b ? a : b;
}
static float oldFMax(float a, float b)
{
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a == 0.0f && b == 0.0f) return (std::signbit(a) && std::signbit(b)) ? -0.0f : 0.0f;
    return a > b ? a : b;
}
enum P89SatSite : uint32_t
{
    SITE_MP_INNER = 1u << 0,    // minProject  PxMin(n.v1, n.v2)                 (:48-49)
    SITE_MP_OUTER = 1u << 1,    // minProject  PxMin(n.v0, inner)                (:47)
    SITE_MIN0_1 = 1u << 2,      // testSeparatingAxis i = 1: min0 = PxMin(proj0, min0)  (:64)
    SITE_MAX0_1 = 1u << 3,      //                           max0 = PxMax(proj0, max0)  (:65)
    SITE_MIN1_1 = 1u << 4,      //                           min1 = PxMin(proj1, min1)  (:66)
    SITE_MAX1_1 = 1u << 5,      //                           max1 = PxMax(proj1, max1)  (:67)
    SITE_MIN0_2 = 1u << 6,      // the same four at i = 2
    SITE_MAX0_2 = 1u << 7,
    SITE_MIN1_2 = 1u << 8,
    SITE_MAX1_2 = 1u << 9,
    SITE_TSA_RET = 1u << 10,    // PxMax(min1 - max0, min0 - max1)               (:70)
    SITE_SAT_PLANE = 1u << 11,  // satIntersect sep = PxMax(sep, minProject(triangle1.triPlane, triangle0)) (:80)
    SITE_SAT_CROSS = 1u << 12,  // satIntersect sep = PxMax(sep, tSep)           (:109)
    SITE_COUNT = 13,
    SITE_ALL = (1u << 13) - 1u
};
static float foldMin(uint32_t revert, uint32_t site, float a, float b) { return (revert & site) ? oldFMin(a, b) : selMin(a, b); }
static float foldMax(uint32_t revert, uint32_t site, float a, float b) { return (revert & site) ? oldFMax(a, b) : selMax(a, b); }

// minProject (triangletriangle.cuh:43): the `+ d` lands ONCE, after the min.
static float minProjectTri(V3 n, float d, V3 v0, V3 v1, V3 v2, uint32_t revert = 0u)
{
    return foldMin(revert, SITE_MP_OUTER, dot(n, v0),
                   foldMin(revert, SITE_MP_INNER, dot(n, v1), dot(n, v2))) + d;
}
// testSeparatingAxis (triangletriangle.cuh:56): the loop's i = 1, 2 unrolled, projection as a,
// running value as b.
static float testSeparatingAxis(V3 ax, V3 a0, V3 a1, V3 a2, V3 b0, V3 b1, V3 b2, uint32_t revert = 0u)
{
    float min0, max0, min1, max1;
    min0 = max0 = dot(a0, ax);
    min1 = max1 = dot(b0, ax);
    const float pa1 = dot(a1, ax), pb1 = dot(b1, ax);
    min0 = foldMin(revert, SITE_MIN0_1, pa1, min0); max0 = foldMax(revert, SITE_MAX0_1, pa1, max0);
    min1 = foldMin(revert, SITE_MIN1_1, pb1, min1); max1 = foldMax(revert, SITE_MAX1_1, pb1, max1);
    const float pa2 = dot(a2, ax), pb2 = dot(b2, ax);
    min0 = foldMin(revert, SITE_MIN0_2, pa2, min0); max0 = foldMax(revert, SITE_MAX0_2, pa2, max0);
    min1 = foldMin(revert, SITE_MIN1_2, pb2, min1); max1 = foldMax(revert, SITE_MAX1_2, pb2, max1);
    return foldMax(revert, SITE_TSA_RET, min1 - max0, min0 - max1);
}
// Gu::satIntersect<true>, early-outs included.  `crossRaised` reports whether the cross block
// pushed sep above what the two plane terms alone gave — the fixture asserts it fires.
// `nanCross` counts the cross axes whose testSeparatingAxis came out NaN.
static float satIntersect(V3 a0, V3 a1, V3 a2, V3 nA, float dA,
                          V3 b0, V3 b1, V3 b2, V3 nB, float dB, float tol, bool* crossRaised,
                          uint32_t revert = 0u, int* nanCross = nullptr)
{
    float sep = minProjectTri(nA, dA, b0, b1, b2, revert);
    if (sep > tol) return sep;
    sep = foldMax(revert, SITE_SAT_PLANE, sep, minProjectTri(nB, dB, a0, a1, a2, revert));
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
                const float tSep = testSeparatingAxis(sa, a0, a1, a2, b0, b1, b2, revert);
                if (nanCross && std::isnan(tSep)) ++*nanCross;
                sep = foldMax(revert, SITE_SAT_CROSS, sep, tSep);
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
    c.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(c.mTriangleVertexIndices_ownedEdges)>(size_t(bCi.addr));
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
                // PxMin(mBestDistance, mContactDistance) (femClothClothMidPhase.cu:140): a select.
                const float sep = p817mesh::satIntersect(a0, a1, a2, nA, dA, b0, b1, b2, nB, dB,
                                                         p817mesh::selMin(best[lane], contactDist), &raised);
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
// P9.89 -- femClothMeshMidphaseBda's triangle-triangle SAT folds are the CUDA's SELECTS.
//
// Every fold in triangletriangle.cuh is PxMin/PxMax<float>: minProject's
// PxMin(n.v0, PxMin(n.v1, n.v2)) (:48), testSeparatingAxis's running
// min0/max0/min1/max1 = PxMin/PxMax(proj, running) (:64-67) and its
// PxMax(min1 - max0, min0 - max1) (:70), and satIntersect's
// sep = PxMax(sep, minProject(...)) (:80) and sep = PxMax(sep, tSep) (:109).  PxMin/PxMax<float>
// are selectMin/selectMax, a < b ? a : b and a > b ? a : b (PxMath.h:82-100,
// PxUnixMathIntrinsics.h:116-125).  The port spelled all of them min()/max(), i.e.
// GLSL.std.450 FMin/FMax, which llvmpipe evaluates as "the non-NaN operand" with -0/+0 on a
// zero tie.
//
// Where the engine sees it: a NaN cross axis.  satIntersect normalizes each edge cross by
// 1/sqrt(magSq); when one component of the cross OVERFLOWS (edge lengths whose product passes
// FLT_MAX, ~2e19 each), magSq = inf, the scale is 0, inf * 0 = NaN, and every projection on
// that axis is NaN.  selectMax(sep, NaN) = NaN WIPES the running separation, and
// selectMax(NaN, t) = t lets the next finite axis reset it; FMax just skips the NaN.  So:
//   * DROP  -- the last evaluated axis is NaN.  The CUDA returns NaN, `separation < bestDist`
//              fails and the pair is never recorded; the old port kept the plane gap and
//              emitted it;
//   * RESET -- NaN axes in the middle, then one finite axis.  The CUDA forgets the plane gap
//              and returns that last axis's (negative) separation, so the pair jumps to the
//              FRONT of the warp's top-4; the old port kept the gap and ranked it fifth, i.e.
//              dropped it;
//   * CTRL  -- the same NaN axes in the middle, but the cloth is level, so the last finite
//              axis is vertical and recomputes the plane gap: the reset changes nothing, and
//              both spellings emit the same ordered list.
// femClothSimMeshMidphase_p89NaNAxis drives those through the engine kernel.  The other fold
// sites cannot change a pair set at all (a NaN projection needs a NaN axis, and then all six
// projections are NaN on both spellings; the sign of a zero never reaches a comparison), so
// femClothSimMeshMidphase_p89SatProbe reads every fold out bit for bit through the test-only
// cloth_meshMidphaseSatProbe entry.
// ===========================================================================
namespace p89cmm
{
using p817mesh::V3;

// PxQuat::rotate / rotateInv (PxQuat.h:310, :324) and shape2Vertex (cudaNpCommon.h:285), the
// formulas the shader carries, so the vertex-space query triangle is the kernel's bit for bit.
static V3 quatRotate(const float q[4], V3 v, bool inv)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q[3] * q[3] - 0.5f;
    const float dot2 = (q[0] * vx + q[1] * vy + q[2] * vz);
    if (!inv)
        return V3{ vx * w2 + (q[1] * vz - q[2] * vy) * q[3] + q[0] * dot2,
                   vy * w2 + (q[2] * vx - q[0] * vz) * q[3] + q[1] * dot2,
                   vz * w2 + (q[0] * vy - q[1] * vx) * q[3] + q[2] * dot2 };
    return V3{ vx * w2 - (q[1] * vz - q[2] * vy) * q[3] + q[0] * dot2,
               vy * w2 - (q[2] * vx - q[0] * vz) * q[3] + q[1] * dot2,
               vz * w2 - (q[0] * vy - q[1] * vx) * q[3] + q[2] * dot2 };
}

struct Leaf { uint32_t start, count; };

// One query warp of cloth_meshMidphaseGeneratePairsLaunch: the per-lane leaf walk and the
// top-4 extraction.  Returns the emitted prims IN EMISSION ORDER (ascending bestDist, lowest
// lane first on a tie).  Leaves are visited highest child first, as the LIFO stack pops them.
static std::vector<uint32_t> queryPairs(V3 a0, V3 a1, V3 a2, const std::vector<Leaf>& leaves,
                                        const std::vector<std::array<float, 3>>& leafMin,
                                        const std::vector<std::array<float, 3>>& leafMax,
                                        const std::vector<F4>& mv, const std::vector<TriIdx4>& mi,
                                        float contactDist, uint32_t revert,
                                        std::vector<float>* bestOut = nullptr)
{
    using namespace p817mesh;
    // triBoundingBox (deformableCollision.cuh:40) + fattenFast(contactDist)
    const float tMin[3] = { selMin(selMin(a0.x, a1.x), a2.x) - contactDist,
                            selMin(selMin(a0.y, a1.y), a2.y) - contactDist,
                            selMin(selMin(a0.z, a1.z), a2.z) - contactDist };
    const float tMax[3] = { selMax(selMax(a0.x, a1.x), a2.x) + contactDist,
                            selMax(selMax(a0.y, a1.y), a2.y) + contactDist,
                            selMax(selMax(a0.z, a1.z), a2.z) + contactDist };
    V3 nA; float dA; makePlane(a0, a1, a2, nA, dA);
    const float third = 1.0f / 3.0f;
    const V3 cen{ (a0.x + a1.x + a2.x) * third, (a0.y + a1.y + a2.y) * third, (a0.z + a1.z + a2.z) * third };

    std::vector<float> best(32, 3.4028235e38f);
    std::vector<uint32_t> bestPrim(32, 0xFFFFFFFFu);
    for (size_t c = leaves.size(); c-- > 0;)
    {
        bool overlap = true;
        for (int k = 0; k < 3; ++k)
            if (leafMin[c][k] > tMax[k] || tMin[k] > leafMax[c][k]) overlap = false;
        if (!overlap) continue;
        for (uint32_t lane = 0; lane < leaves[c].count && lane < 32u; ++lane)
        {
            const uint32_t p = leaves[c].start + lane;
            const V3 b0{ mv[mi[p].x].x, mv[mi[p].x].y, mv[mi[p].x].z };
            const V3 b1{ mv[mi[p].y].x, mv[mi[p].y].y, mv[mi[p].y].z };
            const V3 b2{ mv[mi[p].z].x, mv[mi[p].z].y, mv[mi[p].z].z };
            V3 nB; float dB; makePlane(b0, b1, b2, nB, dB);
            if (!(dot(nB, cen) + dB >= 0.0f)) continue;   // the front-face gate (:138)
            const float sep = satIntersect(a0, a1, a2, nA, dA, b0, b1, b2, nB, dB,
                                           selMin(best[lane], contactDist), nullptr, revert);
            if (sep < best[lane]) { best[lane] = sep; bestPrim[lane] = p; }
        }
    }
    if (bestOut) *bestOut = best;
    uint32_t count = 0;
    for (uint32_t l = 0; l < 32u; ++l) if (best[l] < contactDist) ++count;
    if (count > 4u) count = 4u;
    std::vector<uint32_t> out;
    for (uint32_t i = 0; i < count; ++i)
    {
        uint32_t w = 0; float b = 3.4028235e38f;
        for (uint32_t l = 0; l < 32u; ++l) if (best[l] < b) { b = best[l]; w = l; }
        if (best[w] < contactDist) out.push_back(bestPrim[w]);
        best[w] = 3.4028235e38f;
    }
    return out;
}
} // namespace p89cmm

PXG_TEST(femClothSimMeshMidphase_p89NaNAxis)
{
    using namespace p816dc;        // PxgShapeHost, CmInputHost
    using namespace p89cmm;
    using p817mesh::V3;
    PxgSlangComputeDevice* dev = testDevice();

    uint64_t rs = 0x9a5c0de5ull;
    auto u = [&]() { rs = rs * 6364136223846793005ull + 1442695040888963407ull; return float(uint32_t(rs >> 40)) / 16777216.0f; };

    // The trimesh sits at the identity with unit scale, so vertex space is world space (the
    // transform is still run, in the kernel's own formulas, by the reference).  Cases are
    // stacked in y, 4 apart; every leaf box is tight in y, so each query overlaps only its own
    // leaf.  A 90-degree turn about y, (x, z) -> (z, -x), is exact and keeps every winding, so
    // each case comes in four orientations.
    enum Kind { DROP = 0, RESET = 1, CTRL = 2 };
    const int NDROP = 8, NRESET = 8, NCTRL = 4, NC = NDROP + NRESET + NCTRL;
    const float CD_CLOTH = 0.05f, CD_MESH = 0.15f;
    const float contactDist = CD_CLOTH + CD_MESH;

    std::vector<F4> mv;
    std::vector<TriIdx4> mi;
    std::vector<Leaf> leaves;
    std::vector<int> kind;
    std::vector<std::array<V3, 3>> query;
    auto turn = [](V3 v, int r) {
        for (int i = 0; i < r; ++i) v = V3{ v.z, v.y, -v.x };
        return v;
    };
    auto pushTri = [&](V3 p0, V3 p1, V3 p2) {
        const uint32_t b = uint32_t(mv.size());
        mv.push_back(F4{ p0.x, p0.y, p0.z, 0 }); mv.push_back(F4{ p1.x, p1.y, p1.z, 0 });
        mv.push_back(F4{ p2.x, p2.y, p2.z, 0 });
        mi.push_back(TriIdx4{ b, b + 1u, b + 2u, 0u });
    };
    // prim 0: a dummy in no leaf, so no primStartIndex is zero.
    pushTri(V3{ 1000.0f, -500.0f, 0 }, V3{ 1000.0f, -500.0f, 1.0f }, V3{ 1001.0f, -500.0f, 0 });

    for (int c = 0; c < NC; ++c)
    {
        const int k = c < NDROP ? DROP : c < NDROP + NRESET ? RESET : CTRL;
        const int r = c & 3;
        const float Y = 4.0f * float(c);
        // X * L > FLT_MAX, so a cloth edge along X crossed with a mesh edge along L overflows.
        // s * X and m * L stay below 1.8e19, so both triangles' own plane normals are finite.
        const float X = 1.9e19f + 1.0e19f * u(), L = 1.9e19f + 1.0e19f * u();
        const float s = 0.25f + 0.25f * u(), m = 0.25f + 0.35f * u();
        const float xb = 0.02f + 0.28f * u(), zb = 0.02f + 0.28f * u();
        Leaf lf{ uint32_t(mi.size()), 0u };
        std::array<V3, 3> q;
        if (k == DROP)
        {
            // Horizontal pair, gap g.  Cloth edge a2 - a1 = (X, 0, -s) x mesh edge
            // b2 - b1 = (m, 0, -L) is the LAST axis, (2, 2), and it overflows.  (0, 1), (0, 2)
            // and (2, 1) are finite and vertical and give the gap.
            const float g = 0.04f + 0.13f * u();
            q = { turn(V3{ 0, Y + g, 0 }, r), turn(V3{ 0, Y + g, s }, r), turn(V3{ X, Y + g, 0 }, r) };
            pushTri(turn(V3{ xb, Y, zb }, r), turn(V3{ xb, Y, zb + L }, r), turn(V3{ xb + m, Y, zb }, r));
            lf.count = 1u;
        }
        else
        {
            // The cloth's long edge a1 - a0 = (X, 0, 0) runs along x; the mesh P is long in z.
            // Axes (0, 0), (0, 1), (2, 0), (2, 1) overflow, (0, 2) is parallel and skipped,
            // and the LAST axis, (2, 2) = (a2 - a1) x (m, 0, 0), is finite: the cloth normal.
            //   RESET: the cloth is tilted by h, and P runs off in -z, under the far side of
            //          the cloth's plane, so along the cloth normal it is NEGATIVELY separated.
            //          The CUDA's sep ends there; the old port kept the plane gap g;
            //   CTRL:  the cloth is level (h = 0) and P runs off in +z, so the last axis is
            //          vertical and recomputes the gap: both spellings agree.
            // P shares its leaf with small level slabs under the cloth's low edge, gaps
            // 0.02 .. 0.13, all closer than g: four for RESET (so the top-4 cap binds), two
            // for CTRL.
            const float g = 0.15f + 0.03f * u();
            const float h = (k == RESET) ? 0.04f + 0.26f * u() : 0.0f;
            q = { turn(V3{ 0, Y + g, 0 }, r), turn(V3{ X, Y + g, 0 }, r), turn(V3{ 0, Y + g + h, s }, r) };
            const uint32_t nSlab = (k == RESET) ? 4u : 2u;
            const uint32_t pLane = uint32_t(c) % 5u % (nSlab + 1u);
            uint32_t slab = 0;
            for (uint32_t lane = 0; lane < nSlab + 1u; ++lane)
            {
                if (lane == pLane)
                {
                    if (k == RESET)
                        pushTri(turn(V3{ xb, Y, zb }, r), turn(V3{ xb + m, Y, zb - L }, r), turn(V3{ xb, Y, zb - L }, r));
                    else
                        pushTri(turn(V3{ xb, Y, zb }, r), turn(V3{ xb, Y, zb + L }, r), turn(V3{ xb + m, Y, zb + L }, r));
                }
                else
                {
                    const float qd = 0.02f + 0.028f * float(slab++) + 0.01f * u();
                    const float yq = Y + g - qd;
                    pushTri(turn(V3{ 0.1f, yq, 0.05f }, r), turn(V3{ 0.1f, yq, 0.35f }, r), turn(V3{ 0.4f, yq, 0.05f }, r));
                }
            }
            lf.count = nSlab + 1u;
        }
        leaves.push_back(lf);
        kind.push_back(k);
        query.push_back(q);
    }
    const uint32_t NPRIM = uint32_t(mi.size()), NVERT = uint32_t(mv.size());

    // Leaf boxes: exactly the bounds of their primitives.
    std::vector<std::array<float, 3>> lmin(NC), lmax(NC);
    for (int c = 0; c < NC; ++c)
    {
        lmin[c] = { 3.4e38f, 3.4e38f, 3.4e38f };
        lmax[c] = { -3.4e38f, -3.4e38f, -3.4e38f };
        for (uint32_t p = leaves[c].start; p < leaves[c].start + leaves[c].count; ++p)
            for (uint32_t v : { mi[p].x, mi[p].y, mi[p].z })
            {
                const float cc[3] = { mv[v].x, mv[v].y, mv[v].z };
                for (int a = 0; a < 3; ++a)
                {
                    lmin[c][a] = std::min(lmin[c][a], cc[a]);
                    lmax[c][a] = std::max(lmax[c][a], cc[a]);
                }
            }
    }

    // ---- reference: CUDA selects, and the old FMin/FMax twin ------------------------------
    const float triQ[4] = { 0, 0, 0, 1 }, scaleRot[4] = { 0, 0, 0, 1 };
    auto toVertex = [&](V3 w) {
        V3 t = quatRotate(triQ, w, true);                         // world -> mesh local (triP = 0)
        t = quatRotate(scaleRot, t, false);                       // shape2Vertex, scale (1,1,1)
        t = V3{ t.x / 1.0f, t.y / 1.0f, t.z / 1.0f };
        return quatRotate(scaleRot, t, true);
    };
    std::vector<std::vector<uint32_t>> ref(NC), twin(NC), twinCross(NC);
    uint32_t nDropNaN = 0, nResetFront = 0, nCtrlSame = 0, nTwinDiff = 0, nCrossDiff = 0;
    for (int c = 0; c < NC; ++c)
    {
        const V3 a0 = toVertex(query[c][0]), a1 = toVertex(query[c][1]), a2 = toVertex(query[c][2]);
        std::vector<float> best;
        ref[c] = queryPairs(a0, a1, a2, leaves, lmin, lmax, mv, mi, contactDist, 0u, &best);
        twin[c] = queryPairs(a0, a1, a2, leaves, lmin, lmax, mv, mi, contactDist, p817mesh::SITE_ALL);
        twinCross[c] = queryPairs(a0, a1, a2, leaves, lmin, lmax, mv, mi, contactDist, p817mesh::SITE_SAT_CROSS);
        if (ref[c] != twin[c]) ++nTwinDiff;
        if (ref[c] != twinCross[c]) ++nCrossDiff;
        const uint32_t P = (kind[c] == DROP) ? leaves[c].start
                                            : leaves[c].start + uint32_t(c) % 5u % leaves[c].count;
        if (kind[c] == DROP && ref[c].empty() && twin[c] == std::vector<uint32_t>{ P })
            ++nDropNaN;
        if (kind[c] == RESET && ref[c].size() == 4u && ref[c][0] == P && best[P - leaves[c].start] < 0.0f &&
            twin[c].size() == 4u && std::find(twin[c].begin(), twin[c].end(), P) == twin[c].end())
            ++nResetFront;
        if (kind[c] == CTRL && ref[c] == twin[c] && ref[c].size() == 3u && ref[c][2] == P)
            ++nCtrlSame;
    }
    PXG_CHECK_MSG(nDropNaN == uint32_t(NDROP),
                  "p89 NaN axis: only %u of %d DROP cases have the CUDA rejecting the pair (NaN last "
                  "axis) while the old FMax twin emits it", nDropNaN, NDROP);
    PXG_CHECK_MSG(nResetFront == uint32_t(NRESET),
                  "p89 NaN axis: only %u of %d RESET cases put the NaN-reset pair FIRST in the CUDA "
                  "list and leave it out of the old twin's", nResetFront, NRESET);
    PXG_CHECK_MSG(nCtrlSame == uint32_t(NCTRL),
                  "p89 NaN axis: only %u of %d CTRL cases emit the same three pairs, the NaN-axis "
                  "pair last, on both spellings", nCtrlSame, NCTRL);
    PXG_CHECK_MSG(nTwinDiff == uint32_t(NDROP + NRESET) && nCrossDiff == uint32_t(NDROP + NRESET),
                  "p89 NaN axis: the old twin differs on %u cases and the sep = PxMax(sep, tSep) "
                  "revert alone on %u; both should be the %d DROP + RESET cases",
                  nTwinDiff, nCrossDiff, NDROP + NRESET);

    // ---- the cooked blob: header + ONE BV32 node with NC leaf children --------------------
    std::vector<uint32_t> blob(4u + 292u, 0u);
    {
        const uint32_t n = 4u;
        for (int c = 0; c < NC; ++c)
        {
            for (int a = 0; a < 3; ++a)
            {
                blob[n + c * 4u + a] = f2u(lmin[c][a]);
                blob[n + 128u + c * 4u + a] = f2u(lmax[c][a]);
            }
            blob[n + 256u + c] = 1u | (leaves[c].count << 1) | (leaves[c].start << 7);
        }
        blob[n + 288] = uint32_t(NC);   // mNbNodes
        blob[n + 289] = 1u;             // mDepth
    }
    const size_t vBase = blob.size();
    blob.resize(vBase + NVERT * 4u, 0u);
    std::memcpy(blob.data() + vBase, mv.data(), NVERT * sizeof(F4));
    const size_t iBase = blob.size();
    blob.resize(iBase + NPRIM * 4u, 0u);
    std::memcpy(blob.data() + iBase, mi.data(), NPRIM * sizeof(TriIdx4));
    blob[0] = NVERT; blob[1] = NPRIM; blob[2] = 1u; blob[3] = 1u;
    DevBuf bBlob = makeDev(dev, blob);
    if (!bBlob.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    std::vector<F4> cv;
    std::vector<TriIdx4> ci;
    for (int c = 0; c < NC; ++c)
    {
        const uint32_t b = uint32_t(cv.size());
        for (int i = 0; i < 3; ++i) cv.push_back(F4{ query[c][i].x, query[c][i].y, query[c][i].z, 1.0f });
        ci.push_back(TriIdx4{ b, b + 1u, b + 2u, 0u });
    }
    DevBuf bCv = makeDev(dev, cv), bCi = makeDev(dev, ci);
    PxgFEMCloth cl = makeZeroDesc();
    cl.mPosition_InvMass = reinterpret_cast<decltype(cl.mPosition_InvMass)>(size_t(bCv.addr));
    cl.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(cl.mTriangleVertexIndices_ownedEdges)>(size_t(bCi.addr));
    cl.mNbTriangles = uint32_t(NC);
    cl.mNbVerts = uint32_t(cv.size());
    std::vector<PxgFEMCloth> descArr = { cl };
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_TRIANGLEMESH_H; shapes[0].particleOrSoftbodyId = 0xFFFFFFFFu;
    shapes[0].sx = 1.0f; shapes[0].sy = 1.0f; shapes[0].sz = 1.0f; shapes[0].srw = 1.0f;
    shapes[0].hullLo = uint32_t(bBlob.addr & 0xFFFFFFFFull);
    shapes[0].hullHi = uint32_t(bBlob.addr >> 32);
    shapes[1].type = GEOM_TRIANGLEMESH_H; shapes[1].particleOrSoftbodyId = 0u;
    shapes[1].sx = 1.0f; shapes[1].sy = 1.0f; shapes[1].sz = 1.0f; shapes[1].srw = 1.0f;
    DevBuf bShapes = makeDev(dev, shapes);
    std::vector<CmInputHost> cms = { { 0u, 1u, 1u, 0u } };
    DevBuf bCms = makeDev(dev, cms);
    std::vector<float> xf(2 * 8, 0.0f);
    xf[3] = 1.0f; xf[8 + 3] = 1.0f;   // both identity
    DevBuf bXf = makeDev(dev, xf);
    std::vector<float> cd = { CD_CLOTH, CD_MESH };
    DevBuf bCd = makeDev(dev, cd);
    const uint32_t CAP = 256u;
    DevBuf bStack = makeDev(dev, std::vector<uint32_t>(CAP * 4u, 0xFFFFFFFFu));
    DevBuf bPairNum = makeDev(dev, std::vector<uint32_t>{ 0u });

    p817mesh::ClothMeshMidPush push = {};
    push.toleranceLength = 1.0f;
    push.cmInputs = bCms.addr;
    push.transformCache = bXf.addr;
    push.contactDistance = bCd.addr;
    push.gpuShapes = bShapes.addr;
    push.femClothes = bDesc.addr;
    push.stackSizeBytes = CAP * 16u;
    push.stackPtr = bStack.addr;
    push.midphasePairsNum = bPairNum.addr;
    Kernel k("femClothMeshMidphaseBda.spv", "cloth_meshMidphaseGeneratePairsLaunch",
             {}, sizeof(p817mesh::ClothMeshMidPush));
    k.dispatch(1024u, {}, push, 1u, 1u);
    dev->flush();

    // Each query's pairs occupy one contiguous block (one atomicAdd), written in emission
    // order, so the per-query sequence in stack order IS the kernel's ranking.
    const uint32_t gNum = downloadBuffer<uint32_t>(dev, bPairNum.h, 1)[0];
    const auto gStack = downloadBuffer<uint32_t>(dev, bStack.h, CAP * 4u);
    std::vector<std::vector<uint32_t>> got(NC);
    uint32_t badCm = 0, badTri = 0;
    for (uint32_t i = 0; i < gNum && i < CAP; ++i)
    {
        if (gStack[i * 4u + 0u] != 0u) ++badCm;
        const uint32_t t = gStack[i * 4u + 1u];
        if (t >= uint32_t(NC)) { ++badTri; continue; }
        got[t].push_back(gStack[i * 4u + 2u]);
    }
    size_t refTotal = 0;
    for (auto& r : ref) refTotal += r.size();
    PXG_CHECK_MSG(badCm == 0u && badTri == 0u, "p89 NaN axis: %u pairs with a bad cmIdx, %u with a bad cloth triangle",
                  badCm, badTri);
    PXG_CHECK_MSG(gNum == uint32_t(refTotal), "p89 NaN axis: emitted %u pairs, the CUDA reference %zu", gNum, refTotal);
    uint32_t bad[3] = { 0, 0, 0 }, asTwin = 0;
    for (int c = 0; c < NC; ++c)
    {
        if (got[c] == ref[c]) continue;
        ++bad[kind[c]];
        if (got[c] == twin[c]) ++asTwin;
        if (bad[0] + bad[1] + bad[2] <= 4)
        {
            std::printf("    p89 NaN axis case %d (kind %d): gpu [", c, kind[c]);
            for (uint32_t p : got[c]) std::printf(" %u", p);
            std::printf(" ] cuda [");
            for (uint32_t p : ref[c]) std::printf(" %u", p);
            std::printf(" ] old twin [");
            for (uint32_t p : twin[c]) std::printf(" %u", p);
            std::printf(" ]\n");
        }
    }
    std::printf("    [p89 NaN axis] %d queries (%d drop, %d reset, %d ctrl), %zu CUDA pairs\n",
                NC, NDROP, NRESET, NCTRL, refTotal);
    PXG_CHECK_MSG(bad[0] + bad[1] + bad[2] == 0u,
                  "p89 NaN axis: %u DROP, %u RESET and %u CTRL queries emit a different ordered pair "
                  "list than the CUDA's selectMax folds (%u of them the old FMax twin's)",
                  bad[0], bad[1], bad[2], asTwin);

    for (PxgBufferHandle h : { bBlob.h, bCv.h, bCi.h, bDesc.h, bShapes.h, bCms.h, bXf.h, bCd.h,
                               bStack.h, bPairNum.h })
        dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// P9.89 -- every fold of the SAT, bit for bit, through cloth_meshMidphaseSatProbe.
//
// Each record carries two triangles, two explicit planes and an explicit axis (with the SAT
// tolerance in axis.w); the probe returns
//   x = minProjectTri(planeA, b0, b1, b2)
//   y = testSeparatingAxis(axis, a0, a1, a2, b0, b1, b2)
//   z = satIntersectTriTri(explicit planes)
//   w = satIntersectTriTri(makePlane(a), makePlane(b)), the engine kernel's composition.
// The battery:
//   * PATTERN records, for the eleven minProject / testSeparatingAxis folds.  On an
//     axis-aligned axis sgn * e_k a vertex's projection is ((x*ax + y*ay) + z*az), so each
//     projection can be planted exactly: -0 (all three products -0), +0, a non-round value,
//     +-inf, or NaN (an inf in a zeroed component: inf * 0).  For every fold a sampler draws
//     projection patterns until it has four whose single-site revert (that ONE fold spelled
//     FMin/FMax) changes the output -- four with signed-zero ties only, four with a
//     non-finite operand;
//   * PLANE records for satIntersect's sep = PxMax(sep, minProject(B, a)): TINY triangles
//     (edges <= 4e-3, so every edge cross has magSq < 1e-5 and the nine cross axes are all
//     skipped) with axis-aligned explicit planes, so the plane fold decides the result
//     alone -- a -0 / +0 tie, or planeB NaN (the (0, NaN, 0) makePlane gives on overflow);
//   * CROSS records for sep = PxMax(sep, tSep): the engine's own NaN-axis geometry, two
//     triangles whose long edges (~2e19) overflow in the cross product, with makePlane planes;
//   * ORDINARY records (non-round, overlapping or near) and NON-FINITE records (random inf,
//     +-0 and ~2e19 / ~2e38 components), compared bit for bit like the rest.
// The reference is the CUDA transcribed (p817mesh, revert = 0).  NaN outputs are compared as
// NaN: which NaN a select hands on depends on operand order inside llvmpipe's arithmetic, and
// the CUDA's own NaN is canonical anyway.  Everything else is compared by bits.
// ---------------------------------------------------------------------------
namespace p89cmm
{
struct SpRec { F4 a0, a1, a2, b0, b1, b2, pA, pB, ax; };
static_assert(sizeof(SpRec) == 144, "probe record must be SP_IN_STRIDE = 144 bytes");
struct SpPush { uint64_t input, output; uint32_t numRecords, pad0, pad1, pad2; };
static_assert(sizeof(SpPush) == 32, "probe push block must be 32 bytes");

static F4 f4(V3 v, float w = 0.0f) { return F4{ v.x, v.y, v.z, w }; }
static V3 v3(F4 f) { return V3{ f.x, f.y, f.z }; }
static bool sameWord(float a, float b)
{
    return (std::isnan(a) && std::isnan(b)) || (!std::isnan(a) && !std::isnan(b) && f2u(a) == f2u(b));
}
static F4 probeRef(const SpRec& r, uint32_t revert)
{
    using namespace p817mesh;
    const V3 a0 = v3(r.a0), a1 = v3(r.a1), a2 = v3(r.a2), b0 = v3(r.b0), b1 = v3(r.b1), b2 = v3(r.b2);
    F4 o;
    o.x = minProjectTri(v3(r.pA), r.pA.w, b0, b1, b2, revert);
    o.y = testSeparatingAxis(v3(r.ax), a0, a1, a2, b0, b1, b2, revert);
    o.z = satIntersect(a0, a1, a2, v3(r.pA), r.pA.w, b0, b1, b2, v3(r.pB), r.pB.w, r.ax.w, nullptr, revert);
    V3 nA, nB; float dA, dB;
    makePlane(a0, a1, a2, nA, dA);
    makePlane(b0, b1, b2, nB, dB);
    o.w = satIntersect(a0, a1, a2, nA, dA, b0, b1, b2, nB, dB, r.ax.w, nullptr, revert);
    return o;
}
static uint32_t probeDiffWords(const SpRec& r, uint32_t revert)
{
    const F4 c = probeRef(r, 0u), m = probeRef(r, revert);
    return uint32_t(!sameWord(c.x, m.x)) + uint32_t(!sameWord(c.y, m.y)) + uint32_t(!sameWord(c.z, m.z)) +
           uint32_t(!sameWord(c.w, m.w));
}
} // namespace p89cmm

PXG_TEST(femClothSimMeshMidphase_p89SatProbe)
{
    using namespace p89cmm;
    using p817mesh::V3;
    namespace pm = p817mesh;
    PxgSlangComputeDevice* dev = testDevice();

    uint64_t rs = 0xc0ffee89ull;
    auto u = [&]() { rs = rs * 6364136223846793005ull + 1442695040888963407ull; return float(uint32_t(rs >> 40)) / 16777216.0f; };
    auto pick = [&](int n) { return int(u() * float(n)) % n; };
    auto sym = [&](float r) { return (u() * 2.0f - 1.0f) * r; };
    const float INF = std::numeric_limits<float>::infinity();
    const float NaN = std::numeric_limits<float>::quiet_NaN();
    const float MAXF = 3.4028235e38f;

    // A vertex whose projection on sgn * e_k is exactly p.
    auto realize = [&](float p, int k, float sgn) {
        float c[3];
        const bool zero = (p == 0.0f), neg = std::signbit(p);
        for (int i = 0; i < 3; ++i)
            c[i] = zero ? (neg ? -(0.3f + u()) : (0.3f + u())) : sym(1.5f);
        if (std::isnan(p)) { c[(k + 1) % 3] = (u() < 0.5f) ? INF : -INF; }
        else if (zero) c[k] = (neg ? -0.0f : 0.0f) * sgn;
        else c[k] = p * sgn;
        return V3{ c[0], c[1], c[2] };
    };
    auto axisOf = [](int k, float sgn) { V3 a{ 0.0f, 0.0f, 0.0f }; (&a.x)[k] = sgn; return a; };

    enum Fam { PATTERN = 0, PLANE = 1, CROSS = 2, ORDINARY = 3, NONFINITE = 4 };
    std::vector<SpRec> recs;
    std::vector<int> fam;
    std::vector<uint32_t> famSite;   // the site a planted record was drawn for (0 = none)

    // ---- PATTERN records ------------------------------------------------------------------
    const float ZP[6] = { -0.0f, 0.0f, -1.25f, -0.75f, 0.5f, 1.5f };
    const float FP[9] = { -0.0f, 0.0f, -1.25f, -0.75f, 0.5f, 1.5f, NaN, INF, -INF };
    const uint32_t tsaSites[9] = { pm::SITE_MIN0_1, pm::SITE_MAX0_1, pm::SITE_MIN1_1, pm::SITE_MAX1_1,
                                   pm::SITE_MIN0_2, pm::SITE_MAX0_2, pm::SITE_MIN1_2, pm::SITE_MAX1_2,
                                   pm::SITE_TSA_RET };
    const uint32_t mpSites[2] = { pm::SITE_MP_INNER, pm::SITE_MP_OUTER };
    uint32_t missing = 0;
    for (int si = 0; si < 11; ++si)
    {
        const bool isMp = si >= 9;
        const uint32_t site = isMp ? mpSites[si - 9] : tsaSites[si];
        for (int nonfin = 0; nonfin < 2; ++nonfin)
        {
            int found = 0;
            for (int tries = 0; tries < 400000 && found < 4; ++tries)
            {
                const int k = pick(3);
                const float sgn = (u() < 0.5f) ? 1.0f : -1.0f;
                const V3 ax = axisOf(k, sgn);
                float p[6];
                bool anyNonfin = false;
                for (int i = 0; i < 6; ++i)
                {
                    p[i] = nonfin ? FP[pick(9)] : ZP[pick(6)];
                    if (!std::isfinite(p[i])) anyNonfin = true;
                }
                if (bool(nonfin) != anyNonfin) continue;
                SpRec r;
                const float DP[4] = { 0.0f, -0.0f, 0.25f, -0.5f };
                if (isMp)
                {
                    r.a0 = f4(V3{ sym(1.0f), sym(1.0f), sym(1.0f) });
                    r.a1 = f4(V3{ sym(1.0f), sym(1.0f), sym(1.0f) });
                    r.a2 = f4(V3{ sym(1.0f), sym(1.0f), sym(1.0f) });
                    r.b0 = f4(realize(p[0], k, sgn)); r.b1 = f4(realize(p[1], k, sgn)); r.b2 = f4(realize(p[2], k, sgn));
                    r.pA = f4(ax, DP[pick(4)]);
                }
                else
                {
                    r.a0 = f4(realize(p[0], k, sgn)); r.a1 = f4(realize(p[1], k, sgn)); r.a2 = f4(realize(p[2], k, sgn));
                    r.b0 = f4(realize(p[3], k, sgn)); r.b1 = f4(realize(p[4], k, sgn)); r.b2 = f4(realize(p[5], k, sgn));
                    r.pA = f4(ax, DP[pick(2)]);
                }
                r.pB = f4(axisOf(pick(3), (u() < 0.5f) ? 1.0f : -1.0f), DP[pick(2)]);
                r.ax = f4(ax, (u() < 0.5f) ? 0.2f : MAXF);
                const F4 c = probeRef(r, 0u), m = probeRef(r, site);
                if (sameWord(isMp ? c.x : c.y, isMp ? m.x : m.y)) continue;
                recs.push_back(r); fam.push_back(PATTERN); famSite.push_back(site);
                ++found;
            }
            if (found < 4) ++missing;
        }
    }
    PXG_CHECK_MSG(missing == 0u, "p89 SAT probe: %u (site, zero/non-finite) pattern classes found fewer than four "
                  "distinguishing records", missing);

    // ---- PLANE records: tiny triangles, the plane fold alone ----------------------------------
    {
        auto tiny = [&]() {
            const int t = pick(4);
            return t == 0 ? 0.0f : t == 1 ? -0.0f : (u() < 0.5f ? -1.0f : 1.0f) * (0.5e-3f + 1.5e-3f * u());
        };
        for (int nonfin = 0; nonfin < 2; ++nonfin)
        {
            int found = 0;
            for (int tries = 0; tries < 400000 && found < 4; ++tries)
            {
                SpRec r;
                r.a0 = f4(V3{ tiny(), tiny(), tiny() }); r.a1 = f4(V3{ tiny(), tiny(), tiny() }); r.a2 = f4(V3{ tiny(), tiny(), tiny() });
                r.b0 = f4(V3{ tiny(), tiny(), tiny() }); r.b1 = f4(V3{ tiny(), tiny(), tiny() }); r.b2 = f4(V3{ tiny(), tiny(), tiny() });
                r.pA = f4(axisOf(pick(3), (u() < 0.5f) ? 1.0f : -1.0f), (u() < 0.5f) ? 0.0f : -0.0f);
                if (nonfin)
                {
                    V3 n{ 0.0f, 0.0f, 0.0f }; (&n.x)[pick(3)] = NaN;   // makePlane's overflow normal
                    r.pB = f4(n, NaN);
                }
                else
                    r.pB = f4(axisOf(pick(3), (u() < 0.5f) ? 1.0f : -1.0f), (u() < 0.5f) ? 0.0f : -0.0f);
                r.ax = f4(axisOf(pick(3), 1.0f), (u() < 0.5f) ? 0.2f : MAXF);
                const F4 c = probeRef(r, 0u), m = probeRef(r, pm::SITE_SAT_PLANE);
                if (sameWord(c.z, m.z)) continue;
                recs.push_back(r); fam.push_back(PLANE); famSite.push_back(pm::SITE_SAT_PLANE);
                ++found;
            }
            if (found < 4) ++missing;
        }
    }

    // ---- CROSS records: the engine's NaN-axis geometry -----------------------------------------
    {
        auto turn = [](V3 v, int r) { for (int i = 0; i < r; ++i) v = V3{ v.z, v.y, -v.x }; return v; };
        int found = 0;
        for (int tries = 0; tries < 4000 && found < 16; ++tries)
        {
            const int r = pick(4), shape = pick(3);
            const float X = 1.9e19f + 1.0e19f * u(), L = 1.9e19f + 1.0e19f * u();
            const float s = 0.3f + 0.5f * u(), m = 0.3f + 0.5f * u(), g = sym(0.3f), h = 0.04f + 0.4f * u();
            const float xb = sym(0.3f), zb = sym(0.3f);
            V3 a[3], b[3];
            if (shape == 0)
            {
                a[0] = { 0, g, 0 }; a[1] = { 0, g, s }; a[2] = { X, g, 0 };
                b[0] = { xb, 0, zb }; b[1] = { xb, 0, zb + L }; b[2] = { xb + m, 0, zb };
            }
            else
            {
                a[0] = { 0, g, 0 }; a[1] = { X, g, 0 }; a[2] = { 0, g + h, s };
                b[0] = { xb, 0, zb };
                if (shape == 1) { b[1] = { xb + m, 0, zb - L }; b[2] = { xb, 0, zb - L }; }
                else { b[1] = { xb, 0, zb + L }; b[2] = { xb + m, 0, zb + L }; }
            }
            SpRec rec;
            rec.a0 = f4(turn(a[0], r)); rec.a1 = f4(turn(a[1], r)); rec.a2 = f4(turn(a[2], r));
            rec.b0 = f4(turn(b[0], r)); rec.b1 = f4(turn(b[1], r)); rec.b2 = f4(turn(b[2], r));
            V3 nA, nB; float dA, dB;
            pm::makePlane(v3(rec.a0), v3(rec.a1), v3(rec.a2), nA, dA);
            pm::makePlane(v3(rec.b0), v3(rec.b1), v3(rec.b2), nB, dB);
            rec.pA = f4(nA, dA); rec.pB = f4(nB, dB);
            V3 ax{ sym(1.0f), sym(1.0f), sym(1.0f) };
            ax = pm::scl(ax, 1.0f / std::sqrt(pm::dot(ax, ax)));
            rec.ax = f4(ax, (u() < 0.5f) ? 0.2f : MAXF);
            if (probeDiffWords(rec, pm::SITE_SAT_CROSS) == 0u) continue;
            recs.push_back(rec); fam.push_back(CROSS); famSite.push_back(pm::SITE_SAT_CROSS);
            ++found;
        }
        if (found < 16) ++missing;
    }

    // ---- ORDINARY and NON-FINITE records ------------------------------------------------------
    for (int t = 0; t < 96; ++t)
    {
        SpRec r;
        const V3 o{ sym(2.0f), sym(2.0f), sym(2.0f) };
        const float sc = 0.2f + 2.0f * u();
        V3 a[3], b[3];
        for (int i = 0; i < 3; ++i) a[i] = { o.x + sym(sc), o.y + sym(sc), o.z + sym(sc) };
        const V3 o2{ o.x + sym(0.6f * sc), o.y + sym(0.6f * sc), o.z + sym(0.6f * sc) };
        for (int i = 0; i < 3; ++i) b[i] = { o2.x + sym(sc), o2.y + sym(sc), o2.z + sym(sc) };
        r.a0 = f4(a[0]); r.a1 = f4(a[1]); r.a2 = f4(a[2]); r.b0 = f4(b[0]); r.b1 = f4(b[1]); r.b2 = f4(b[2]);
        V3 nA, nB; float dA, dB;
        pm::makePlane(a[0], a[1], a[2], nA, dA);
        pm::makePlane(b[0], b[1], b[2], nB, dB);
        r.pA = f4(nA, dA); r.pB = f4(nB, dB);
        V3 ax{ sym(1.0f), sym(1.0f), sym(1.0f) };
        r.ax = f4(pm::scl(ax, 1.0f / std::sqrt(pm::dot(ax, ax))), (t & 1) ? 0.2f : MAXF);
        recs.push_back(r); fam.push_back(ORDINARY); famSite.push_back(0u);
    }
    for (int t = 0; t < 96; ++t)
    {
        auto comp = [&]() {
            const float c = u();
            if (c < 0.12f) return INF;
            if (c < 0.24f) return -INF;
            if (c < 0.36f) return ((u() < 0.5f) ? 1.0f : -1.0f) * (1.2e19f + 1.0e19f * u());
            if (c < 0.42f) return ((u() < 0.5f) ? 1.0f : -1.0f) * (2.0e38f + 1.0e38f * u());
            if (c < 0.50f) return (u() < 0.5f) ? 0.0f : -0.0f;
            return sym(2.0f);
        };
        SpRec r;
        V3 a[3], b[3];
        for (int i = 0; i < 3; ++i) { a[i] = { comp(), comp(), comp() }; b[i] = { comp(), comp(), comp() }; }
        r.a0 = f4(a[0]); r.a1 = f4(a[1]); r.a2 = f4(a[2]); r.b0 = f4(b[0]); r.b1 = f4(b[1]); r.b2 = f4(b[2]);
        V3 nA, nB; float dA, dB;
        pm::makePlane(a[0], a[1], a[2], nA, dA);
        pm::makePlane(b[0], b[1], b[2], nB, dB);
        r.pA = f4(nA, dA); r.pB = f4(nB, dB);
        V3 ax{ sym(1.0f), sym(1.0f), sym(1.0f) };
        ax = pm::scl(ax, 1.0f / std::sqrt(pm::dot(ax, ax)));
        if (u() < 0.4f) ax = axisOf(pick(3), (u() < 0.5f) ? 1.0f : -1.0f);
        r.ax = f4(ax, (u() < 0.5f) ? 0.2f : MAXF);
        recs.push_back(r); fam.push_back(NONFINITE); famSite.push_back(0u);
    }
    PXG_CHECK_MSG(missing == 0u, "p89 SAT probe: %u planted classes came up short", missing);
    const uint32_t N = uint32_t(recs.size());

    // ---- the battery has to SEE every fold: each single-site revert changes some word -------
    const char* siteName[pm::SITE_COUNT] = { "minProject inner", "minProject outer", "min0 i=1", "max0 i=1",
                                             "min1 i=1", "max1 i=1", "min0 i=2", "max0 i=2", "min1 i=2",
                                             "max1 i=2", "tsa return", "sat plane", "sat cross" };
    uint32_t siteWords[pm::SITE_COUNT] = {}, twinWords = 0, blindSites = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        for (int si = 0; si < int(pm::SITE_COUNT); ++si) siteWords[si] += probeDiffWords(recs[i], 1u << si);
        twinWords += probeDiffWords(recs[i], pm::SITE_ALL);
    }
    std::printf("    [p89 SAT probe] %u records; words the single-site revert changes:", N);
    for (int si = 0; si < int(pm::SITE_COUNT); ++si)
    {
        std::printf(" %s %u%s", siteName[si], siteWords[si], si + 1 < int(pm::SITE_COUNT) ? "," : "");
        if (siteWords[si] < 4u) ++blindSites;
    }
    std::printf("; old twin %u\n", twinWords);
    PXG_CHECK_MSG(blindSites == 0u, "p89 SAT probe: %u fold sites are changed on fewer than four words by their "
                  "own revert -- the battery cannot see them", blindSites);

    // ---- run the probe -----------------------------------------------------------------------
    DevBuf bIn = makeDev(dev, recs);
    if (!bIn.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }
    std::vector<F4> outInit(N, F4{ 1.0f, 2.0f, 3.0f, 4.0f });
    DevBuf bOut = makeDev(dev, outInit);
    SpPush push = { bIn.addr, bOut.addr, N, 0u, 0u, 0u };
    Kernel k("femClothMeshMidphaseBda.spv", "cloth_meshMidphaseSatProbe", {}, sizeof(SpPush));
    k.dispatch((N + 31u) / 32u, {}, push);
    dev->flush();
    const auto g = downloadBuffer<F4>(dev, bOut.h, N);

    uint32_t bad[5] = { 0, 0, 0, 0, 0 }, badWords = 0, asTwin = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const F4 c = probeRef(recs[i], 0u), t = probeRef(recs[i], pm::SITE_ALL);
        const float gv[4] = { g[i].x, g[i].y, g[i].z, g[i].w }, cv[4] = { c.x, c.y, c.z, c.w }, tv[4] = { t.x, t.y, t.z, t.w };
        bool recBad = false;
        for (int w = 0; w < 4; ++w)
        {
            if (sameWord(gv[w], cv[w])) continue;
            recBad = true;
            ++badWords;
            if (sameWord(gv[w], tv[w])) ++asTwin;
            if (badWords <= 6)
                std::printf("    p89 SAT probe rec %u (fam %d, site 0x%x) word %c: gpu %08x (%g) cuda %08x (%g) old twin %08x (%g)\n",
                            i, fam[i], famSite[i], "xyzw"[w], f2u(gv[w]), gv[w], f2u(cv[w]), cv[w], f2u(tv[w]), tv[w]);
        }
        if (recBad) ++bad[fam[i]];
    }
    PXG_CHECK_MSG(badWords == 0u,
                  "p89 SAT probe: %u words differ from the CUDA's select folds (%u of them equal the old "
                  "FMin/FMax twin) -- by family: pattern %u, plane %u, cross %u, ordinary %u, non-finite %u records",
                  badWords, asTwin, bad[PATTERN], bad[PLANE], bad[CROSS], bad[ORDINARY], bad[NONFINITE]);

    for (PxgBufferHandle h : { bIn.h, bOut.h }) dev->destroyBuffer(h);
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
//   * the 176-byte blob (5.12), whose stackSizeBytes sits at @64 BEFORE the two stack pointers -- the
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
//   * the 5.12 filter: one (rigid filter ref, cloth TRIANGLE) search of the cloth's
//     mRigidSurfaceFilterPairs.  A vertex ref of a triangle, or the triangle under another
//     rigid, must NOT suppress it (5.6.1 needed all three cloth vertices filtered);
//   * restDistance indexed by cmIdx (not by a cache ref, and not by 0);
//   * the PxgRigidDbContactWriter field mapping (5.12): pair info {rigid node, triangle ref},
//     the PxU16 material stream and the full rigid sort key (contactByRigid, its low word,
//     the identity index).
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
// Byte-identical to ClothMeshCgParams (ABI row kClothMeshContactGen, 5.12).
struct ClothMeshCgPush
{
    float toleranceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, restDistance, gpuShapes, femClothes,
             shapeToRigidRemapTable;
    uint32_t stackSizeBytes, pad68;
    uint64_t stackPtr, midphasePairsNum;
    uint64_t stackSizeNeededOnDevice;
    // PxgRigidDbContactWriter, by value
    uint64_t wPoint, wNormalPen, wBary, wInfo, wRigidMat, wCount, wByX, wTempByX, wIdxByX;
    uint32_t wMaxContacts, pad172;
};
static_assert(sizeof(ClothMeshCgPush) == 176, "cloth mesh CG blob must be 176 bytes");
static_assert(offsetof(ClothMeshCgPush, shapeToRigidRemapTable) == 56, "remap table @56");
static_assert(offsetof(ClothMeshCgPush, stackSizeBytes) == 64, "stackSizeBytes @64, BEFORE the stack pointers");
static_assert(offsetof(ClothMeshCgPush, stackPtr) == 72, "stackPtr @72");
static_assert(offsetof(ClothMeshCgPush, midphasePairsNum) == 80, "midphasePairsNum @80");
static_assert(offsetof(ClothMeshCgPush, stackSizeNeededOnDevice) == 88, "stackSizeNeeded @88");
static_assert(offsetof(ClothMeshCgPush, wPoint) == 96, "the writer starts @96");
static_assert(offsetof(ClothMeshCgPush, wMaxContacts) == 168, "maxNumContacts @168");

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
    // P9.94a: the conjugate's rotate, as written in PxTransform.h:166 -- not clothQ.rotateInv.
    const V3 m2cP = qRot(clothQConj, sub(meshP, clothP));
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
        if (i == 3u) return 0.10f;      // would make contact, but the triangle is filtered
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
    c.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(c.mTriangleVertexIndices_ownedEdges)>(size_t(bCi.addr));
    c.mNbTriangles = NB_PAIRS;
    c.mNbVerts = uint32_t(cv.size());
    // ---- the filter list (5.12, per cloth, sorted (ref0, ref1)) ------------------------------
    // (RIGID_ID, triangle 3) suppresses pair 3.  Pair 4 carries two near misses: a VERTEX ref of
    // one of its corners under the right rigid, and its TRIANGLE ref under another rigid.
    const uint64_t RIGID_ID = 0x1234000056780000ull;   // dynamic: the filter ref is the node
    std::vector<FemPairInfoHost> filters = {
        FemPairInfoHost{ RIGID_ID, encClothVertex(0u, 12u) },
        FemPairInfoHost{ RIGID_ID, encClothTriangle(0u, 3u) },
        FemPairInfoHost{ RIGID_ID + 1ull, encClothTriangle(0u, 4u) },
    };
    std::sort(filters.begin(), filters.end(), [](const FemPairInfoHost& a, const FemPairInfoHost& b) {
        return a.ref0 != b.ref0 ? a.ref0 < b.ref0 : a.ref1 < b.ref1;
    });
    DevBuf bFilters = makeDev(dev, filters);
    c.mRigidSurfaceFilterPairs = reinterpret_cast<PxgDeformablePairInfo*>(size_t(bFilters.addr));
    c.mNumRigidSurfaceFilterPairs = uint32_t(filters.size());
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
    std::vector<uint64_t> remap = { 0xDEADull, RIGID_ID };
    DevBuf bRemap = makeDev(dev, remap);

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
    std::vector<uint32_t> outMat(MAXC / 2u, 0xABABABABu);         // PxU16 stream
    std::vector<uint64_t> outByX(MAXC, 0xCDCDCDCDCDCDCDCDull);
    std::vector<uint32_t> outTempByX(MAXC, 0xCDCDCDCDu);
    std::vector<uint32_t> outIdxByX(MAXC, 0xABABABABu);
    std::vector<uint32_t> outCount = { 0u };
    DevBuf bPoint = makeDev(dev, outPoint), bNormalPen = makeDev(dev, outNormalPen),
           bBary = makeDev(dev, outBary), bInfo = makeDev(dev, outInfo), bMat = makeDev(dev, outMat),
           bByX = makeDev(dev, outByX), bTempByX = makeDev(dev, outTempByX),
           bIdxByX = makeDev(dev, outIdxByX), bCount = makeDev(dev, outCount);

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
    push.stackSizeNeededOnDevice = bNeeded.addr;
    push.wPoint = bPoint.addr; push.wNormalPen = bNormalPen.addr; push.wBary = bBary.addr;
    push.wInfo = bInfo.addr;   push.wRigidMat = bMat.addr; push.wCount = bCount.addr;
    push.wByX = bByX.addr; push.wTempByX = bTempByX.addr; push.wIdxByX = bIdxByX.addr;
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
        if (i == 3u) continue;                       // (RIGID_ID, triangle 3) is filtered
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
                  "transform chain is wrong; 5 means the triangle filter was lost; 3 means the "
                  "filter matched a vertex ref or the wrong rigid, or the second warp's reservation was lost",
                  gCount, expect.size());

    const auto gPoint = downloadBuffer<F4>(dev, bPoint.h, MAXC);
    const auto gNormalPen = downloadBuffer<F4>(dev, bNormalPen.h, MAXC);
    const auto gBary = downloadBuffer<F4>(dev, bBary.h, MAXC);
    const auto gInfo = downloadBuffer<uint32_t>(dev, bInfo.h, MAXC * 4u);
    const auto gIdx = downloadBuffer<uint32_t>(dev, bIdxByX.h, MAXC);
    const auto gMat = downloadBuffer<uint32_t>(dev, bMat.h, MAXC / 2u);
    const auto gByX = downloadBuffer<uint64_t>(dev, bByX.h, MAXC);
    const auto gTempByX = downloadBuffer<uint32_t>(dev, bTempByX.h, MAXC);

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
        // PxgDeformablePairInfo: ref0 = the rigid node index (u64), ref1 = the cloth TRIANGLE
        // ref; the material is its own PxU16 stream.
        const uint64_t gotRef0 = uint64_t(gInfo[s * 4 + 0]) | (uint64_t(gInfo[s * 4 + 1]) << 32);
        const uint64_t gotRef1 = uint64_t(gInfo[s * 4 + 2]) | (uint64_t(gInfo[s * 4 + 3]) << 32);
        if (gotRef0 != RIGID_ID)
        { ++mm; std::printf("  slot %u ref0 MM: got 0x%llx ref 0x%llx\n", s,
                            (unsigned long long)gotRef0, (unsigned long long)RIGID_ID); }
        if (gotRef1 != encClothTriangle(0u, e.tri))
        { ++mm; std::printf("  slot %u ref1 MM: got 0x%llx ref triangle %u\n",
                            s, (unsigned long long)gotRef1, e.tri); }
        const uint32_t mat = (gMat[s / 2u] >> (16u * (s & 1u))) & 0xffffu;
        if (mat != 0x2Au)
        { ++mm; std::printf("  slot %u material MM: got 0x%x ref 0x2A\n", s, mat); }
        if (gByX[s] != RIGID_ID)
        { ++mm; std::printf("  slot %u contactByRigid MM: got 0x%llx\n", s, (unsigned long long)gByX[s]); }
        if (gTempByX[s] != uint32_t(RIGID_ID & 0xFFFFFFFFull))
        { ++mm; std::printf("  slot %u tempContactByRigid MM: got 0x%x ref 0x%x (the LOW word)\n",
                            s, gTempByX[s], uint32_t(RIGID_ID & 0xFFFFFFFFull)); }
        if (gIdx[s] != s)
        { ++mm; std::printf("  slot %u contactIndexSortedByX MM: got %u ref %u\n", s, gIdx[s], s); }
    }
    PXG_CHECK_MSG(mm == 0, "cloth_meshContactGen contact fields (%zu mismatches)", mm);

    // Slots past the last contact keep their sort-key poison.
    size_t touched = 0;
    for (uint32_t s = uint32_t(expect.size()); s < MAXC; ++s)
    {
        if (gByX[s] != 0xCDCDCDCDCDCDCDCDull) ++touched;
        if (gTempByX[s] != 0xCDCDCDCDu) ++touched;
    }
    PXG_CHECK_MSG(touched == 0, "%zu sort-key words past the last contact changed", touched);

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
                               bPoint.h, bNormalPen.h, bBary.h, bInfo.h, bMat.h, bByX.h, bTempByX.h,
                               bIdxByX.h, bCount.h })
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
    c.mTriangleVertexIndices_ownedEdges = reinterpret_cast<uint4*>(size_t(bTri.addr));
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
    // PxgRigidDbContactWriter (5.12), by value
    uint64_t w_outPoint, w_outNormalPen, w_outBarycentric, w_outContactInfo, w_outRigidMaterialIndex;
    uint64_t w_totalContactCount, w_contactByRigid, w_tempContactByRigid, w_contactIndexSortedByRigid;
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
    p.w_outRigidMaterialIndex = w.rigidMat.addr;
    p.w_totalContactCount = bCount.addr;
    p.w_contactByRigid = w.byRigid.addr; p.w_tempContactByRigid = w.tempByRigid.addr;
    p.w_contactIndexSortedByRigid = w.idxByRigid.addr;
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
                      gBary[i].x == kBarySentinel.x && gBary[i].w == kBarySentinel.w,
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
    c.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(c.mTriangleVertexIndices_ownedEdges)>(size_t(bCi.addr));
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
// P9.89 -- the cloth-vs-heightfield midphase under a GENERIC rotation, pair set against
// the foundation's own PxBounds3::transformFast(hfTransform.getInverse(), bound).
//
// femClothSimHeightfieldMidphase above turns the field by 180 degrees about y, where
// every rotation entry is 0 or +-1 and all spellings of the inverse transform agree bit
// for bit -- so it could not see that the port rotated (centre - p) ONCE where the CUDA
// rotates centre and -p separately and adds (femClothHFMidPhase.cu:107 via
// PxTransform::getInverse), nor that it took the basis from rotateInv(q, e_i) where the
// CUDA builds PxMat33(conj q).  Both put the local bound an ULP or two off, and an ULP is
// a floor()/ceil()/truncation on the other side of an integer: a different pair SET.
//
// THE ORACLE IS NOT A TRANSCRIPTION.  pxgHfLocalBoundRef.h calls the foundation's
// transformFast and getInverse directly, in this -ffp-contract=off binary.  The TWINS --
// the old port whole, the old centre alone, the old basis whole and each old column
// alone -- are composed by hand, and the fixture proves the composition honest by
// requiring that with nothing substituted it reproduces transformFast bit for bit.
//
// THE POINTS ARE SEARCHED, NOT SAMPLED.  A random point separates the two orders about
// once in a million; `rotatedPoints` places a point's bound on a row, column or height
// integer and walks it ULP by ULP until a twin emits a different set.  The fixture then
// REQUIRES, per twin, that some point separates it and that its whole expected pair list
// differs from the CUDA's -- so this fixture cannot silently lose its teeth.
//
// Both entry points run over the same cloth: a vertex bound is (p - r, p + r)
// (femClothHFMidPhase.cu:379), and a triangle whose three indices name one vertex has
// triBoundingBox(p, p, p).fattenFast(r) -- the same bits -- so element i emits the same
// pairs through either kernel.  2300 elements: past 2048, so the `j` loop runs twice.
// ===========================================================================
PXG_TEST(femClothSimHeightfieldMidphaseRotated)
{
    using namespace p816dc;        // PxgShapeHost, CmInputHost
    using namespace p927hf;        // ClothHfMidPush
    PxgSlangComputeDevice* dev = testDevice();

    const hflb::Placement hf = hflb::rotatedPlacement();
    const std::vector<uint32_t> blob = hflb::rotatedBlob();
    const hflb::Field field{ blob.data() };
    DevBuf bBlob = makeDev(dev, blob);
    if (!bBlob.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // ASYMMETRIC and non-round; the kernel adds the two exactly as the host does.
    const float CD_CLOTH = 0.07f, CD_HF = 0.045f;
    const float radius = CD_CLOTH + CD_HF;

    const uint32_t NB = 2300u;
    std::vector<physx::PxVec3> pts;
    std::vector<uint8_t> isKnife;
    hflb::rotatedPoints(hf, field, radius, NB, pts, isKnife);
    uint32_t nbKnives = 0u;
    for (uint8_t k : isKnife) nbKnives += k;

    // ---- the oracle, its composition check and the twins ----------------------------
    std::vector<std::pair<uint32_t, uint32_t>> ref;           // (element, triangleIdx)
    std::vector<std::vector<uint32_t>> refPer(NB);
    uint32_t composedMismatch = 0u;
    for (uint32_t i = 0; i < NB; ++i)
    {
        const physx::PxBounds3 b = hflb::sphereBound(pts[i], radius);
        physx::PxVec3 mn, mx, cmn, cmx;
        hflb::localBound(hf, b, hflb::CUDA, mn, mx);
        hflb::localBound(hf, b, hflb::COMPOSED, cmn, cmx);
        for (uint32_t a = 0; a < 3u; ++a)
            if (f2u(mn[a]) != f2u(cmn[a]) || f2u(mx[a]) != f2u(cmx[a])) ++composedMismatch;
        hflb::pairs(field, mn, mx, refPer[i]);
        for (uint32_t t : refPer[i]) ref.push_back({ i, t });
    }
    std::sort(ref.begin(), ref.end());
    PXG_CHECK_MSG(composedMismatch == 0u,
                  "the hand-composed chain with nothing substituted differs from the "
                  "foundation's transformFast in %u bound components; the twins below would "
                  "then not be 'the CUDA with one piece changed'", composedMismatch);

    uint32_t sepCount[6] = {}, sepKnife[6] = {};
    for (uint32_t t = 0; t < 6u; ++t)
    {
        std::vector<std::pair<uint32_t, uint32_t>> twin;
        for (uint32_t i = 0; i < NB; ++i)
        {
            std::vector<uint32_t> got;
            hflb::pairsOfPoint(hf, field, pts[i], radius, hflb::ROT_TWINS[t], got);
            if (got != refPer[i]) { ++sepCount[t]; sepKnife[t] += isKnife[i]; }
            for (uint32_t g : got) twin.push_back({ i, g });
        }
        std::sort(twin.begin(), twin.end());
        PXG_CHECK_MSG(twin != ref && sepCount[t] >= 3u,
                      "twin '%s': only %u of %u elements emit a different pair set than the "
                      "CUDA (need >= 3).  The fixture can no longer tell that spelling from "
                      "transformFast(getInverse()) and would pass on the pre-P9.89 port",
                      hflb::ROT_TWIN_NAMES[t], sepCount[t], NB);
    }

    // ---- the cloth: one vertex per point, triangle i = (i, i, i) --------------------
    std::vector<F4> cv(NB);
    for (uint32_t i = 0; i < NB; ++i) cv[i] = F4{ pts[i].x, pts[i].y, pts[i].z, 1.0f };
    std::vector<TriIdx4> ci(NB);
    for (uint32_t i = 0; i < NB; ++i) ci[i] = TriIdx4{ i, i, i, 0u };
    DevBuf bCv = makeDev(dev, cv), bCi = makeDev(dev, ci);

    const uint32_t CLOTH_ID = 1u;
    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bCv.addr));
    c.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(c.mTriangleVertexIndices_ownedEdges)>(size_t(bCi.addr));
    c.mNbVerts = NB;
    c.mNbTriangles = NB;
    std::vector<PxgFEMCloth> descArr(CLOTH_ID + 1u, makeZeroDesc());
    descArr[CLOTH_ID] = c;
    DevBuf bDesc = makeDev(dev, descArr);

    const uint32_t GEOM_HEIGHTFIELD_H = uint32_t(physx::PxGeometryType::eHEIGHTFIELD);
    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0].type = GEOM_HEIGHTFIELD_H;  shapes[0].particleOrSoftbodyId = 0xFFFFFFFFu;
    shapes[0].sx = hf.scale.x; shapes[0].sy = hf.scale.y; shapes[0].sz = hf.scale.z;
    shapes[0].srw = 1.0f;
    shapes[0].hullLo = uint32_t(bBlob.addr & 0xFFFFFFFFull);
    shapes[0].hullHi = uint32_t(bBlob.addr >> 32);
    shapes[1].type = GEOM_TRIANGLEMESH_H; shapes[1].particleOrSoftbodyId = CLOTH_ID;
    shapes[1].sx = 1.0f; shapes[1].sy = 1.0f; shapes[1].sz = 1.0f; shapes[1].srw = 1.0f;
    DevBuf bShapes = makeDev(dev, shapes);
    std::vector<CmInputHost> cms = { { 0u, 1u, /*hf cache*/ 1u, /*cloth cache*/ 0u } };
    DevBuf bCms = makeDev(dev, cms);

    std::vector<float> xf(2 * 8, 0.0f);
    xf[3] = 1.0f;                                             // slot 0 (cloth) identity
    xf[8 + 0] = hf.q.x; xf[8 + 1] = hf.q.y; xf[8 + 2] = hf.q.z; xf[8 + 3] = hf.q.w;
    xf[8 + 4] = hf.p.x; xf[8 + 5] = hf.p.y; xf[8 + 6] = hf.p.z;
    DevBuf bXf = makeDev(dev, xf);
    std::vector<float> cd = { CD_CLOTH, CD_HF };
    DevBuf bCd = makeDev(dev, cd);

    const uint32_t CAP = 16384u;
    PXG_CHECK_MSG(ref.size() < CAP, "the reference has %zu pairs; raise CAP", ref.size());
    auto runOne = [&](const char* entry, std::vector<std::pair<uint32_t, uint32_t>>& emitted) {
        std::vector<uint32_t> stack(CAP * 4u, 0xFFFFFFFFu);
        DevBuf bStack = makeDev(dev, stack);
        std::vector<uint32_t> pairNum = { 0u };
        DevBuf bPairNum = makeDev(dev, pairNum);
        ClothHfMidPush p = {};
        p.numNPWorkItems = 1u;
        p.toleranceLength = 1.0f;
        p.cmInputs = bCms.addr;
        p.transformCache = bXf.addr;
        p.contactDistance = bCd.addr;
        p.gpuShapes = bShapes.addr;
        p.femClothes = bDesc.addr;
        p.stackPtr = bStack.addr;
        p.midphasePairsNum = bPairNum.addr;
        p.stackSizeBytes = CAP * 16u;
        Kernel k("femClothHfMidphaseBda.spv", entry, {}, sizeof(ClothHfMidPush));
        k.dispatch(/*gridX*/ 1024u, {}, p, 1u, 1u);       // PxgNarrowphaseCore.cpp:6927
        dev->flush();
        const uint32_t n = downloadBuffer<uint32_t>(dev, bPairNum.h, 1)[0];
        const auto s = downloadBuffer<uint32_t>(dev, bStack.h, CAP * 4u);
        emitted.clear();
        for (uint32_t i = 0; i < n && i < CAP; ++i)
            emitted.push_back({ s[i * 4u + 1u], s[i * 4u + 2u] });
        std::sort(emitted.begin(), emitted.end());
        dev->destroyBuffer(bStack.h); dev->destroyBuffer(bPairNum.h);
        return n;
    };

    const char* entries[2] = { "cloth_heightfieldMidphaseGeneratePairsLaunch",
                               "cloth_midphaseVertexHeightfieldLaunch" };
    for (const char* entry : entries)
    {
        std::vector<std::pair<uint32_t, uint32_t>> got;
        const uint32_t n = runOne(entry, got);
        uint32_t badElems = 0u, badKnives = 0u, firstBad = 0xFFFFFFFFu;
        {
            std::vector<std::vector<uint32_t>> gotPer(NB);
            for (auto& e : got) if (e.first < NB) gotPer[e.first].push_back(e.second);
            for (uint32_t i = 0; i < NB; ++i)
            {
                std::sort(gotPer[i].begin(), gotPer[i].end());
                std::vector<uint32_t> want = refPer[i];
                std::sort(want.begin(), want.end());
                if (gotPer[i] != want)
                {
                    ++badElems; badKnives += isKnife[i];
                    if (firstBad == 0xFFFFFFFFu) firstBad = i;
                }
            }
        }
        PXG_CHECK_MSG(n == uint32_t(ref.size()) && got == ref && badElems == 0u,
                      "%s: %u pairs against the foundation oracle's %zu; %u elements (%u of "
                      "them knife-edge points) emit a different set, first element %u.  A "
                      "knife-edge mismatch is the association order of transformFast(getInverse()) "
                      "-- femClothHFMidPhase.cu:107", entry, n, ref.size(), badElems, badKnives,
                      firstBad);
    }

    std::printf("    [ note ] %u elements (%u knife-edge), %zu pairs; elements separating each "
                "twin from the CUDA (knife-edge): all %u (%u), centre %u (%u), basis %u (%u), "
                "col0 %u (%u), col1 %u (%u), col2 %u (%u)\n",
                NB, nbKnives, ref.size(), sepCount[0], sepKnife[0], sepCount[1], sepKnife[1],
                sepCount[2], sepKnife[2], sepCount[3], sepKnife[3], sepCount[4], sepKnife[4],
                sepCount[5], sepKnife[5]);

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
    // PxgRigidDbContactWriter (5.12), by value
    uint64_t wPoint, wNormalPen, wBary, wInfo, wRigidMat, wCount, wByX, wTempByX, wIdxByX;
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
    uint64_t wPoint, wNormalPen, wBary, wInfo, wRigidMat, wCount, wByX, wTempByX, wIdxByX;
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
    using p817cg::qRot; using p817cg::qMul;
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
    const V3 r2cP = qRot(clothQConj, sub(rigidP, clothP));   // P9.94a: literal transformInv
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
    c.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(c.mTriangleVertexIndices_ownedEdges)>(size_t(bCi.addr));
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
    hp.wIdxByX = wHf.idxByRigid.addr; hp.wRigidMat = wHf.rigidMat.addr;
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
    mp.stackSizeNeededOnDevice = bMeshNeeded.addr;
    mp.wPoint = wMesh.point.addr; mp.wNormalPen = wMesh.normalPen.addr;
    mp.wBary = wMesh.barycentric.addr; mp.wInfo = wMesh.info.addr; mp.wCount = wMesh.count.addr;
    mp.wByX = wMesh.byRigid.addr; mp.wTempByX = wMesh.tempByRigid.addr;
    mp.wIdxByX = wMesh.idxByRigid.addr; mp.wRigidMat = wMesh.rigidMat.addr;
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
            r.tri = info[i * 4 + 2] & ((1u << 27) - 1u);   // the PxgDeformableRef feature
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

    // The contact INFO must agree too -- the rigid id, the cloth triangle ref and the
    // material all come from the glue, not from the shared core.
    for (uint32_t i = 0; i < nHf && i < MAXC; ++i)
    {
        const uint32_t mat = readRigidMat(dev, wHf, i);
        PXG_CHECK_MSG(mat == 0x2Au,
                      "contact %u carries material 0x%X, expected the HEIGHTFIELD shape's "
                      "0x2A -- a port reading the CLOTH's materialIndex would get 0", i, mat);
        const uint64_t r0 = uint64_t(iHf[i * 4 + 0]) | (uint64_t(iHf[i * 4 + 1]) << 32);
        const uint64_t r1 = uint64_t(iHf[i * 4 + 2]) | (uint64_t(iHf[i * 4 + 3]) << 32);
        PXG_CHECK_MSG(r0 == 0x00000007ull && r1 == encClothTriangle(CLOTH_ID, uint32_t(r1 & ((1u << 27) - 1u))),
                      "contact %u pair info (0x%llx, 0x%llx): expected the remapped rigid node "
                      "and a TRIANGLE ref of cloth %u", i, (unsigned long long)r0,
                      (unsigned long long)r1, CLOTH_ID);
    }

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
        tp.wIdxByX = wt.idxByRigid.addr; tp.wRigidMat = wt.rigidMat.addr;
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
        vp.wIdxByX = w2.idxByRigid.addr; vp.wRigidMat = w2.rigidMat.addr;
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
// P9.94a -- the SIGN OF ZERO through transformInvTransform, on both engine cloth
// kernels.
//
// femClothPrimitives.cu:3716 / :3876 build the static triangle's cloth-space
// transform as clothTransform.transformInv(rigidTransform), and PxTransform.h:
// 162-169 spells that `qinv = q.getConjugate(); qinv.rotate(src.p - p)`.  The
// port used to write q.rotateInv(src.p - p): identical bits for every nonzero
// component, but a component that is a sum of exact zeros can take the other
// sign.  The two fixtures above cannot see that -- femClothSimMeshContactGen
// compares at 2e-4, and femClothSimHeightfieldContactGen is a differential
// between two kernels that share the helper, so a revert moves both sides.
//
// So each test here is ABSOLUTE: one pair, geometry built so that every term of
// the separation is an exact zero (an axis-aligned triangle whose cloth-space
// normal has two zero lanes, a cloth triangle lying IN its plane and strictly
// inside all five trim planes), and the stored normalPen.w is compared BIT-EXACTLY
// against a short literal host chain.  The chain is valid because nothing is
// clipped and the three corners tie, so sep = planeDistance(plane, a1).  The twin
// flag switches only the chain's transformInv p to the old spelling, and each case
// asserts that the twin gives the other zero -- the proof that the case separates.
// ===========================================================================
namespace p994a
{
using p817cg::V3;
using p817cg::qRot; using p817cg::qRotInv; using p817cg::qMul;
using p817cg::norm; using p817cg::crs; using p817cg::sub; using p817cg::add; using p817cg::dot;

// Mutant twin: true selects the PRE-FIX spelling clothQ.rotateInv(rigidP - clothP).
static bool gP94TinvRotInv = false;

// The literal chain: hfToCloth / trimeshToClothTransform, the static triangle in cloth
// space (vertex2Shape first for the cooked mesh, not for the heightfield: C2), then
// staticTriangleVsClothCollision's plane and planeDistance(plane, a1).
static float sepRef(const float clothQ[4], V3 clothP, const float rigidQ[4], V3 rigidP,
                    const V3 tri[3], bool meshVertex2Shape, V3 a1)
{
    const float qinv[4] = { -clothQ[0], -clothQ[1], -clothQ[2], clothQ[3] };
    float relQ[4];
    qMul(qinv, rigidQ, relQ);
    const V3 relP = gP94TinvRotInv ? qRotInv(clothQ, sub(rigidP, clothP))
                                   : qRot(qinv, sub(rigidP, clothP));
    const float idRot[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    V3 l[3];
    for (int k = 0; k < 3; ++k)
    {
        const V3 v = meshVertex2Shape ? p817cg::vertex2Shape(tri[k], V3{ 1.0f, 1.0f, 1.0f }, idRot) : tri[k];
        l[k] = add(qRot(relQ, v), relP);
    }
    const V3 n = norm(crs(sub(l[1], l[0]), sub(l[2], l[0])));
    const float d = -dot(l[0], n);               // makePlane(localA0, fixNormal)
    return dot(n, a1) + d;                       // planeDistance(plane, a1)
}

struct Case
{
    const char* name;
    float rigidQ[4]; V3 rigidP;
    V3 cloth[3];
    uint32_t cudaBits;                           // hand-derived, re-checked against sepRef
};

// One pair through the MESH (hf == false) or HEIGHTFIELD kernel.  Cloth identity at
// cache ref 0 / shapeRef0, the rigid at 1 -- positional for the heightfield kernel (C1),
// and the mesh kernel finds the cloth by its body id either way.  Returns the contact
// count and the stored normalPen.w word of slot 0.
static void runOne(PxgSlangComputeDevice* dev, bool hf, const Case& cs, uint32_t& count, uint32_t& sepBits)
{
    using namespace p816dc;
    count = 0xFFFFFFFFu; sepBits = 0xFFFFFFFFu;

    // The rigid geometry.  Mesh: triangle (0,1,-1), (0,2,-1), (0,1,-2), cooked as one
    // triangle behind one (unused) BV32 node.  Heightfield: 3x3 samples at height 0, no
    // tessellation bit; triangle 8 = cell 4's first = (1,0,1), (1,0,2), (2,0,1).
    std::vector<uint32_t> blob;
    uint32_t rigidTri = 0u;
    if (hf)
    {
        blob.assign(2u + 9u, 0u);
        blob[0] = 3u; blob[1] = 3u;
        for (uint32_t s = 0; s < 9u; ++s) blob[2u + s] = p927cg::hfSample(0, 3u, 3u);
        rigidTri = 8u;
    }
    else
    {
        const std::vector<F4> mv = { { 0.0f, 1.0f, -1.0f, 0 }, { 0.0f, 2.0f, -1.0f, 0 }, { 0.0f, 1.0f, -2.0f, 0 } };
        const std::vector<TriIdx4> mi = { { 0u, 1u, 2u, 0u } };
        blob.assign(4u + 292u, 0u);
        const size_t vBase = blob.size();
        blob.resize(vBase + mv.size() * 4u, 0u);
        std::memcpy(blob.data() + vBase, mv.data(), mv.size() * sizeof(F4));
        const size_t iBase = blob.size();
        blob.resize(iBase + mi.size() * 4u, 0u);
        std::memcpy(blob.data() + iBase, mi.data(), mi.size() * sizeof(TriIdx4));
        blob[0] = uint32_t(mv.size()); blob[1] = uint32_t(mi.size()); blob[2] = 1u; blob[3] = 1u;
    }
    DevBuf bBlob = makeDev(dev, blob);
    if (!bBlob.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    std::vector<F4> cv = { { cs.cloth[0].x, cs.cloth[0].y, cs.cloth[0].z, 1.0f },
                           { cs.cloth[1].x, cs.cloth[1].y, cs.cloth[1].z, 1.0f },
                           { cs.cloth[2].x, cs.cloth[2].y, cs.cloth[2].z, 1.0f } };
    std::vector<TriIdx4> ci = { { 0u, 1u, 2u, 0u } };
    DevBuf bCv = makeDev(dev, cv), bCi = makeDev(dev, ci);
    PxgFEMCloth c = makeZeroDesc();
    c.mPosition_InvMass = reinterpret_cast<decltype(c.mPosition_InvMass)>(size_t(bCv.addr));
    c.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(c.mTriangleVertexIndices_ownedEdges)>(size_t(bCi.addr));
    c.mNbTriangles = 1u; c.mNbVerts = 3u;
    std::vector<PxgFEMCloth> descArr = { c };
    DevBuf bDesc = makeDev(dev, descArr);

    std::vector<PxgShapeHost> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShapeHost));
    shapes[0] = makeClothShape(0u);
    shapes[1].type = hf ? uint32_t(physx::PxGeometryType::eHEIGHTFIELD) : GEOM_TRIANGLEMESH_H;
    shapes[1].particleOrSoftbodyId = 0xFFFFFFFFu;
    shapes[1].sx = shapes[1].sy = shapes[1].sz = 1.0f; shapes[1].srw = 1.0f;
    shapes[1].materialIndex = 0x2Au;
    shapes[1].hullLo = uint32_t(bBlob.addr & 0xFFFFFFFFull);
    shapes[1].hullHi = uint32_t(bBlob.addr >> 32);
    DevBuf bShapes = makeDev(dev, shapes);
    std::vector<CmInputHost> cms = { { 0u, 1u, 0u, 1u } };
    DevBuf bCms = makeDev(dev, cms);

    // PxsCachedTransform: slot 0 the cloth (identity), slot 1 the rigid.  The -0 lanes are
    // the point, so they are stored as written.
    std::vector<float> xf(2 * 8, 0.0f);
    xf[3] = 1.0f;
    for (int k = 0; k < 4; ++k) xf[8 + k] = cs.rigidQ[k];
    xf[8 + 4] = cs.rigidP.x; xf[8 + 5] = cs.rigidP.y; xf[8 + 6] = cs.rigidP.z;
    DevBuf bXf = makeDev(dev, xf);

    std::vector<float> cd = { 0.05f, 0.05f };
    std::vector<float> rd = { 0.125f };
    DevBuf bCd = makeDev(dev, cd), bRd = makeDev(dev, rd);
    std::vector<uint64_t> remap = { 0ull, 0x00000007ull };
    DevBuf bRemap = makeDev(dev, remap);
    std::vector<uint32_t> stack = { 0u, 0u, rigidTri, 0u };
    DevBuf bStack = makeDev(dev, stack);
    std::vector<uint32_t> one = { 1u }, zero = { 0u };
    DevBuf bNum = makeDev(dev, one), bNeeded = makeDev(dev, zero);
    WriterBufs w = makeWriter(dev, 4u);

    if (hf)
    {
        p927cg::ClothHfCgPush p = {};
        p.toleranceLength = 1.0f;
        p.cmInputs = bCms.addr; p.transformCache = bXf.addr;
        p.contactDistance = bCd.addr; p.restDistance = bRd.addr;
        p.gpuShapes = bShapes.addr; p.femClothes = bDesc.addr;
        p.shapeToRigidRemapTable = bRemap.addr;
        p.stackPtr = bStack.addr; p.midphasePairsNum = bNum.addr;
        p.stackSizeBytes = 16u; p.stackSizeNeededOnDevice = bNeeded.addr;
        p.wPoint = w.point.addr; p.wNormalPen = w.normalPen.addr; p.wBary = w.barycentric.addr;
        p.wInfo = w.info.addr; p.wCount = w.count.addr;
        p.wByX = w.byRigid.addr; p.wTempByX = w.tempByRigid.addr;
        p.wIdxByX = w.idxByRigid.addr; p.wRigidMat = w.rigidMat.addr;
        p.wMaxContacts = w.maxContacts;
        Kernel k("femClothHfContactGenBda.spv", "cloth_heightfieldContactGenLaunch", {},
                 sizeof(p927cg::ClothHfCgPush));
        k.dispatch(2048u, {}, p, 1u, 1u);
    }
    else
    {
        p817cg::ClothMeshCgPush p = {};
        p.toleranceLength = 1.0f;
        p.cmInputs = bCms.addr; p.transformCache = bXf.addr;
        p.contactDistance = bCd.addr; p.restDistance = bRd.addr;
        p.gpuShapes = bShapes.addr; p.femClothes = bDesc.addr;
        p.shapeToRigidRemapTable = bRemap.addr;
        p.stackSizeBytes = 16u; p.stackPtr = bStack.addr; p.midphasePairsNum = bNum.addr;
        p.stackSizeNeededOnDevice = bNeeded.addr;
        p.wPoint = w.point.addr; p.wNormalPen = w.normalPen.addr; p.wBary = w.barycentric.addr;
        p.wInfo = w.info.addr; p.wCount = w.count.addr;
        p.wByX = w.byRigid.addr; p.wTempByX = w.tempByRigid.addr;
        p.wIdxByX = w.idxByRigid.addr; p.wRigidMat = w.rigidMat.addr;
        p.wMaxContacts = w.maxContacts;
        Kernel k("femClothMeshContactGenBda.spv", "cloth_meshContactGenLaunch", {},
                 sizeof(p817cg::ClothMeshCgPush));
        k.dispatch(2048u, {}, p, 1u, 1u);
    }
    dev->flush();

    count = downloadBuffer<uint32_t>(dev, w.count.h, 1)[0];
    sepBits = f2u(downloadBuffer<F4>(dev, w.normalPen.h, 1)[0].w);

    destroyWriter(dev, w);
    for (PxgBufferHandle h : { bBlob.h, bCv.h, bCi.h, bDesc.h, bShapes.h, bCms.h, bXf.h, bCd.h,
                               bRd.h, bRemap.h, bStack.h, bNum.h, bNeeded.h })
        dev->destroyBuffer(h);
}

static void runCases(bool hf, const Case* cases, size_t nbCases)
{
    PxgSlangComputeDevice* dev = testDevice();
    const V3 meshTri[3] = { { 0.0f, 1.0f, -1.0f }, { 0.0f, 2.0f, -1.0f }, { 0.0f, 1.0f, -2.0f } };
    const V3 hfTri[3] = { { 1.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 2.0f }, { 2.0f, 0.0f, 1.0f } };
    const float clothQ[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    const V3 clothP{ 0.0f, 0.0f, 0.0f };
    for (size_t i = 0; i < nbCases; ++i)
    {
        const Case& cs = cases[i];
        const V3* tri = hf ? hfTri : meshTri;
        const uint32_t lit = f2u(sepRef(clothQ, clothP, cs.rigidQ, cs.rigidP, tri, !hf, cs.cloth[0]));
        gP94TinvRotInv = true;
        const uint32_t twin = f2u(sepRef(clothQ, clothP, cs.rigidQ, cs.rigidP, tri, !hf, cs.cloth[0]));
        gP94TinvRotInv = false;
        PXG_CHECK_MSG(lit == cs.cudaBits,
                      "%s: the literal chain gives sep 0x%08X, the hand derivation 0x%08X",
                      cs.name, lit, cs.cudaBits);
        PXG_CHECK_MSG(twin == (lit ^ 0x80000000u),
                      "%s: the rotateInv twin gives sep 0x%08X against the literal 0x%08X -- "
                      "the case does not separate the two transformInv spellings", cs.name, twin, lit);

        uint32_t count = 0u, gSep = 0u;
        runOne(dev, hf, cs, count, gSep);
        if (count == 0xFFFFFFFFu) return;        // no BDA
        PXG_CHECK_MSG(count == 1u, "%s: %u contacts, expected exactly 1 (nothing clipped, sep 0 < "
                      "cDistance 0.1)", cs.name, count);
        PXG_CHECK_MSG(gSep == lit,
                      "%s: normalPen.w is 0x%08X, CUDA 0x%08X (the old rotateInv spelling gives "
                      "0x%08X).  transformInvTransform must be quatRotate(quatConjugate(a.q), "
                      "b.p - a.p)", cs.name, gSep, lit, twin);
        std::printf("    [ note ] %s: sep 0x%08X (twin 0x%08X), %u contact\n", cs.name, gSep, twin, count);
    }
}
} // namespace p994a

// Mesh: meshQ = 180 degrees about z; the cloth-space normal is (1,+0,+0) and every sep
// term is a zero.  A and B put the zero on opposite sides, so a fix that canonicalizes
// zeros to +0 dies on B.
PXG_TEST(femClothSimMeshContactGenP94ZeroSign)
{
    using p994a::Case;
    const float mz = -0.0f;
    const Case cases[] = {
        { "mesh A", { 0.0f, 0.0f, 1.0f, 0.0f }, { mz, mz, 0.5f },
          { { mz, -1.25f, -0.75f }, { mz, -1.5f, -0.75f }, { mz, -1.25f, -1.0f } }, 0x00000000u },
        { "mesh B", { 0.0f, 0.0f, 1.0f, 0.0f }, { mz, mz, mz },
          { { mz, -1.25f, -1.25f }, { mz, -1.5f, -1.25f }, { mz, -1.25f, -1.5f } }, 0x80000000u },
    };
    p994a::runCases(/*hf*/ false, cases, sizeof(cases) / sizeof(cases[0]));
}

// Heightfield: hfQ = 180 degrees about x with signed-zero lanes; the rotated y of every
// vertex is -0, the normal is (+0,-1,+0), and the literal chain gives +0 where the old one
// gives -0.
PXG_TEST(femClothSimHeightfieldContactGenP94ZeroSign)
{
    using p994a::Case;
    const float mz = -0.0f;
    const Case cases[] = {
        { "hf A", { 1.0f, mz, mz, 0.0f }, { -4.0f, mz, mz },
          { { -2.75f, 0.0f, -1.25f }, { -2.75f, 0.0f, -1.5f }, { -2.5f, 0.0f, -1.25f } }, 0x00000000u },
    };
    p994a::runCases(/*hf*/ true, cases, sizeof(cases) / sizeof(cases[0]));
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
//   * a filter list (5.12: the cloth's mRigidSurfaceFilterPairs) naming one
//     triangle against both rigids, and only the corner VERTICES of another:
//     the first must vanish, the second must survive (S7);
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

// The kernel shares cloth_boxTriangleContactGenLaunch's pack (ABI row kClothBoxTriCg)
// -- and NOT cloth_meshContactGenLaunch's, which this fixture reached for first and
// paid for with a deterministic segfault.  Both are 176 bytes (5.12) AND both put the
// writer at @96, so sizeof and the one static_assert everybody checks agree.
// The MIDDLE does not:
//
//              mesh (ClothMeshCgPush)        box-triangle (this one)
//     @64      stackSizeBytes                stackPtr
//     @72      stackPtr                      midphasePairsNum
//     @80      midphasePairsNum              stackSizeBytes
//     @88      stackSizeNeeded               stackSizeNeeded
//
// Every asserted offset below is therefore load-bearing, not decoration.
struct ClothSphereTriPush
{
    float toleranceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, restDistance, gpuShapes, femClothes,
             shapeToRigidRemapTable, stackPtr, midphasePairsNum;
    uint32_t stackSizeBytes, pad84;
    uint64_t stackSizeNeededOnDevice;
    // PxgRigidDbContactWriter (5.12), by value
    uint64_t wPoint, wNormalPen, wBary, wInfo, wRigidMat, wCount, wByX, wTempByX, wIdxByX;
    uint32_t wMaxContacts, pad172;
};
static_assert(sizeof(ClothSphereTriPush) == 176, "cloth sphere/capsule TRIANGLE blob must be 176 bytes");
static_assert(offsetof(ClothSphereTriPush, stackPtr) == 64, "stackPtr @64 -- NOT stackSizeBytes");
static_assert(offsetof(ClothSphereTriPush, midphasePairsNum) == 72, "midphasePairsNum @72");
static_assert(offsetof(ClothSphereTriPush, stackSizeBytes) == 80, "stackSizeBytes @80");
static_assert(offsetof(ClothSphereTriPush, stackSizeNeededOnDevice) == 88, "stackSizeNeeded @88");
static_assert(offsetof(ClothSphereTriPush, wPoint) == 96, "the writer starts @96");
static_assert(offsetof(ClothSphereTriPush, wMaxContacts) == 168, "maxNumContacts @168");

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
    cloth.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(cloth.mTriangleVertexIndices_ownedEdges)>(size_t(bCi.addr));
    cloth.mNbTriangles = NB_TRIS;
    cloth.mNbVerts = uint32_t(cv.size());
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

    // ---- the filter list (S7, 5.12) ---------------------------------------------------
    // The cloth's mRigidSurfaceFilterPairs, keyed (rigid filter ref, cloth feature ref); a
    // dynamic rigid's filter ref is its node.
    // FILTERED_TRI: its TRIANGLE ref against BOTH rigids -> must vanish.
    // PARTIAL_TRI: all three corner VERTEX refs against both rigids, but not the triangle
    // -> must survive: the triangle path searches the triangle ref only.
    // Both are triangles this rig MEASURED as contacting: triangle 12 is even, so
    // it belongs to manager 0 (the SPHERE), and 21 is odd, so it belongs to
    // manager 1 (the CAPSULE).  Picking them by eye is what the first version did,
    // and it drew a filtered triangle that never touched anything -- a check that
    // cannot fail is not a check, so the non-vacuity of BOTH is asserted below.
    const uint32_t FILTERED_TRI = 12u, PARTIAL_TRI = 21u;
    std::vector<FemPairInfoHost> filters;
    for (uint64_t node : { NODE_SPHERE, NODE_CAPSULE })
    {
        filters.push_back(FemPairInfoHost{ node, encClothTriangle(CLOTH_ID, FILTERED_TRI) });
        for (uint32_t k = 0; k < 3u; ++k)
            filters.push_back(FemPairInfoHost{ node, encClothVertex(CLOTH_ID, ci[PARTIAL_TRI].x + k) });
    }
    // find() does a BINARY search: the list must be sorted by (ref0, ref1).
    std::sort(filters.begin(), filters.end(), [](const FemPairInfoHost& a, const FemPairInfoHost& b) {
        return a.ref0 != b.ref0 ? a.ref0 < b.ref0 : a.ref1 < b.ref1;
    });
    DevBuf bFilters = makeDev(dev, filters);
    cloth.mRigidSurfaceFilterPairs = reinterpret_cast<PxgDeformablePairInfo*>(size_t(bFilters.addr));
    cloth.mNumRigidSurfaceFilterPairs = uint32_t(filters.size());
    std::vector<PxgFEMCloth> descArr(CLOTH_ID + 1u, makeZeroDesc());
    descArr[CLOTH_ID] = cloth;
    DevBuf bDesc = makeDev(dev, descArr);

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
    p.stackSizeNeededOnDevice = bNeeded.addr;
    p.wPoint = w.point.addr; p.wNormalPen = w.normalPen.addr;
    p.wBary = w.barycentric.addr; p.wInfo = w.info.addr; p.wCount = w.count.addr;
    p.wByX = w.byRigid.addr; p.wTempByX = w.tempByRigid.addr;
    p.wIdxByX = w.idxByRigid.addr; p.wRigidMat = w.rigidMat.addr;
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

        bool filtered = false;
        for (const auto& f : filters)
            if (f.ref0 == node && f.ref1 == encClothTriangle(CLOTH_ID, tri)) { filtered = true; break; }
        if (filtered) continue;               // S7: one (rigid, triangle) search

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

    struct Rec { uint32_t key[17]; };
    auto less = [](const Rec& x, const Rec& y) {
        for (int i = 0; i < 17; ++i) if (x.key[i] != y.key[i]) return x.key[i] < y.key[i];
        return false;
    };
    std::vector<Rec> gpu, ref;
    for (uint32_t i = 0; i < got && i < MAXC; ++i)
    {
        Rec r{};
        r.key[0] = info[i * 4 + 2];                     // ref1.lo = (TRIANGLE << 29) | tri
        r.key[1] = info[i * 4 + 0]; r.key[2] = info[i * 4 + 1];   // ref0 = the rigid node
        r.key[3] = readRigidMat(dev, w, i);             // the PxU16 material stream
        r.key[16] = info[i * 4 + 3];                    // ref1.hi = clothId << 2
        const float src[12] = { pt[i].x, pt[i].y, pt[i].z, pt[i].w,
                                np[i].x, np[i].y, np[i].z, np[i].w,
                                ba[i].x, ba[i].y, ba[i].z, ba[i].w };
        for (int j = 0; j < 12; ++j) r.key[4 + j] = f2u(src[j]);
        gpu.push_back(r);
    }
    for (const Exp& e : expected)
    {
        Rec r{};
        const uint64_t ref1 = encClothTriangle(CLOTH_ID, e.tri);
        r.key[0] = uint32_t(ref1 & 0xffffffffull);
        r.key[1] = uint32_t(e.node & 0xffffffffull); r.key[2] = uint32_t(e.node >> 32);
        r.key[3] = e.mat & 0xffffu;
        r.key[16] = uint32_t(ref1 >> 32);
        const float src[12] = { e.h.point.x, e.h.point.y, e.h.point.z, e.h.point.w,
                                e.h.normalPen.x, e.h.normalPen.y, e.h.normalPen.z, e.h.normalPen.w,
                                e.h.bary.x, e.h.bary.y, e.h.bary.z, e.h.bary.w };
        for (int j = 0; j < 12; ++j) r.key[4 + j] = f2u(src[j]);
        ref.push_back(r);
    }
    std::sort(gpu.begin(), gpu.end(), less);
    std::sort(ref.begin(), ref.end(), less);

    static const char* const KEY_NAME[17] = {
        "ref1.lo", "ref0.lo", "ref0.hi", "material",
        "point.x", "point.y", "point.z", "point.w(rest)",
        "normal.x", "normal.y", "normal.z", "penetration",
        "bary.x", "bary.y", "bary.z", "bary.w", "ref1.hi"
    };
    uint32_t mismatches = 0u;
    int firstField = -1; size_t firstAt = 0;
    for (size_t i = 0; i < gpu.size() && i < ref.size(); ++i)
        for (int j = 0; j < 17; ++j)
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
                    gpu[firstAt].key[0] & ((1u << 27) - 1u));
    }
    PXG_CHECK_MSG(mismatches == 0u,
                  "%u of %zu contacts differ BIT-EXACTLY from the host mirror of "
                  "triSphereCollide / fcCapsuleCollision and the caller glue",
                  mismatches, gpu.size());

    // S7 both ways, read off the emitted set rather than argued.
    uint32_t sawFiltered = 0u, sawPartial = 0u;
    for (const Rec& r : gpu)
    {
        const uint32_t tri = r.key[0] & ((1u << 27) - 1u);
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
                  "triangle %u is filtered against both rigids and still produced %u contacts",
                  FILTERED_TRI, sawFiltered);
    PXG_CHECK_MSG(sawPartial > 0u,
                  "triangle %u has only its corner VERTEX refs filtered and produced nothing: "
                  "the triangle path searches the TRIANGLE ref (S7, 5.12)", PARTIAL_TRI);

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
                "%u claimed; triangle %u filtered out, %u survived its vertex filters\n",
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
        q.wPoint = w2.point.addr; q.wNormalPen = w2.normalPen.addr;
        q.wBary = w2.barycentric.addr; q.wInfo = w2.info.addr; q.wCount = w2.count.addr;
        q.wByX = w2.byRigid.addr; q.wTempByX = w2.tempByRigid.addr;
        q.wIdxByX = w2.idxByRigid.addr; q.wRigidMat = w2.rigidMat.addr;
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
    cloth.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(cloth.mTriangleVertexIndices_ownedEdges)>(size_t(bCi.addr));
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
    p.stackSizeNeededOnDevice = bNeeded.addr;
    p.wPoint = w.point.addr; p.wNormalPen = w.normalPen.addr;
    p.wBary = w.barycentric.addr; p.wInfo = w.info.addr; p.wCount = w.count.addr;
    p.wByX = w.byRigid.addr; p.wTempByX = w.tempByRigid.addr;
    p.wIdxByX = w.idxByRigid.addr; p.wRigidMat = w.rigidMat.addr;
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
        const uint64_t ref1 = uint64_t(info[i * 4 + 2]) | (uint64_t(info[i * 4 + 3]) << 32);
        const uint32_t tri = uint32_t(ref1 & ((1u << 27) - 1u));
        PXG_CHECK_MSG(tri < NB_TRIS && ref1 == encClothTriangle(CLOTH_ID, tri),
                      "contact %u names ref1 0x%016llX", i, (unsigned long long)ref1);
        if (tri >= NB_TRIS) continue;
        ++seen[tri];
        const uint32_t mat = readRigidMat(dev, w, i);
        PXG_CHECK_MSG(info[i * 4 + 0] == uint32_t(NODE_CAPSULE & 0xffffffffull) &&
                      info[i * 4 + 1] == uint32_t(NODE_CAPSULE >> 32) &&
                      mat == (MAT_CAPSULE & 0xffffu),
                      "tri %c info record %08X %08X / mat %08X", 'A' + char(tri),
                      info[i * 4 + 1], info[i * 4 + 0], mat);
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

// cloth_psContactGenLaunch (5.12): twelve arguments, the 72-byte
// PxgParticleDbContactWriter by value -> 160 bytes (ABI row kClothPsCg).  5.12 dropped
// the restDistance argument; the writer has no material stream and no 32-bit sorted lane.
struct ClothPsCgPush
{
    float toleranceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, gpuShapes, particleSystems, femClothes;
    uint32_t stackSizeBytes, pad60;
    uint64_t stackPtr, midphasePairsNum, stackSizeNeeded;
    // PxgParticleDbContactWriter: point, normalPen, bary, info, count, byParticle,
    // tempByParticle, indexSortedByParticle, maxNumContacts
    uint64_t wPoint, wNormalPen, wBary, wInfo, wCount, wByX, wTempByX, wIdxByX;
    uint32_t wMaxContacts, pad156;
};
static_assert(sizeof(ClothPsCgPush) == 160, "cloth ps CG blob must be 160 bytes");
static_assert(offsetof(ClothPsCgPush, particleSystems) == 40, "particleSystems @40");
static_assert(offsetof(ClothPsCgPush, femClothes) == 48, "femClothes @48 -- AFTER particleSystems");
static_assert(offsetof(ClothPsCgPush, stackSizeBytes) == 56, "stackSizeBytes @56");
static_assert(offsetof(ClothPsCgPush, stackPtr) == 64, "stackPtr @64");
static_assert(offsetof(ClothPsCgPush, midphasePairsNum) == 72, "midphasePairsNum @72");
static_assert(offsetof(ClothPsCgPush, stackSizeNeeded) == 80, "stackSizeNeeded @80");
static_assert(offsetof(ClothPsCgPush, wPoint) == 88, "the writer starts @88 -- paramAlign(72) is 8");
static_assert(offsetof(ClothPsCgPush, wMaxContacts) == 152, "maxNumContacts @152");

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
    cloth.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(cloth.mTriangleVertexIndices_ownedEdges)>(size_t(bCi.addr));
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
    // either one.  (5.12: no rest distance in contact generation, C4.)
    const float CD_PARTICLE = 0.30f, CD_CLOTH = 0.10f;

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
    cloth.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(cloth.mTriangleVertexIndices_ownedEdges)>(size_t(bCi.addr));
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
    DevBuf bCd = makeDev(dev, cd);
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
    p.contactDistance = bCd.addr;
    p.gpuShapes = bShapes.addr; p.particleSystems = bPs.addr; p.femClothes = bDesc.addr;
    p.stackSizeBytes = NB_STACK * 16u;
    p.stackPtr = bStack.addr; p.midphasePairsNum = bNum.addr;
    p.stackSizeNeeded = bNeeded.addr;
    p.wPoint = w.point.addr; p.wNormalPen = w.normalPen.addr; p.wBary = w.barycentric.addr;
    p.wInfo = w.info.addr; p.wCount = w.count.addr;
    p.wByX = w.byRigid.addr; p.wTempByX = w.tempByRigid.addr; p.wIdxByX = w.idxByRigid.addr;
    p.wMaxContacts = MAXC;

    Kernel k("femClothPsContactGenBda.spv", "cloth_psContactGenLaunch",
             {}, sizeof(ClothPsCgPush));
    k.dispatch(512u, {}, p, 1u, 1u);
    dev->flush();

    // ---- the mirror ----------------------------------------------------------------
    struct Exp { F4 point, normalPen, bary; uint64_t ref0, ref1; };
    std::vector<Exp> expected;
    uint32_t faceHits = 0u;
    for (uint32_t i = 0; i < NB_STACK; ++i)          // C8: the CLAMPED walk
    {
        const uint32_t cm = stack[i * 4 + 0], pi = stack[i * 4 + 1], tri = stack[i * 4 + 2];
        const float cDistance = CD_PARTICLE + CD_CLOTH;
        (void)cm;   // both managers route to the same pair (C1); only the TYPE routing differs
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
        Exp e{};
        e.point = F4{ pp.x, pp.y, pp.z, 0.0f };                 // C4 (5.12): the particle itself
        e.normalPen = F4{ -n.x, -n.y, -n.z, m };                // C4: NEGATED, pen = |v|
        e.bary = computeBary(a, b, c, closest);
        e.ref0 = (uint64_t(PS_ID) << 34) | pi;                  // C3: PxgEncodeParticleRef
        e.ref1 = encClothTriangle(CLOTH_ID, tri);
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

    struct Rec { uint32_t key[16]; };
    auto less = [](const Rec& x, const Rec& y) {
        for (int i = 0; i < 16; ++i) if (x.key[i] != y.key[i]) return x.key[i] < y.key[i];
        return false;
    };
    std::vector<Rec> gpu, ref;
    for (uint32_t i = 0; i < got && i < MAXC; ++i)
    {
        Rec r{};
        r.key[0] = info[i * 4 + 0]; r.key[1] = info[i * 4 + 1];   // ref0 (the particle)
        r.key[2] = info[i * 4 + 2]; r.key[15] = info[i * 4 + 3];  // ref1 (the triangle)
        const float src[12] = { pt[i].x, pt[i].y, pt[i].z, pt[i].w,
                                np[i].x, np[i].y, np[i].z, np[i].w,
                                ba[i].x, ba[i].y, ba[i].z, ba[i].w };
        for (int j = 0; j < 12; ++j) r.key[3 + j] = f2u(src[j]);
        gpu.push_back(r);
    }
    for (const Exp& e : expected)
    {
        Rec r{};
        r.key[0] = uint32_t(e.ref0 & 0xffffffffull);
        r.key[1] = uint32_t(e.ref0 >> 32);
        r.key[2] = uint32_t(e.ref1 & 0xffffffffull); r.key[15] = uint32_t(e.ref1 >> 32);
        const float src[12] = { e.point.x, e.point.y, e.point.z, e.point.w,
                                e.normalPen.x, e.normalPen.y, e.normalPen.z, e.normalPen.w,
                                e.bary.x, e.bary.y, e.bary.z, e.bary.w };
        for (int j = 0; j < 12; ++j) r.key[3 + j] = f2u(src[j]);
        ref.push_back(r);
    }
    std::sort(gpu.begin(), gpu.end(), less);
    std::sort(ref.begin(), ref.end(), less);

    static const char* const KEY_NAME[16] = {
        "ref0.lo", "ref0.hi", "ref1.lo",
        "point.x", "point.y", "point.z", "point.w",
        "normal.x", "normal.y", "normal.z", "penetration",
        "bary.x", "bary.y", "bary.z", "bary.w", "ref1.hi"
    };
    uint32_t mismatches = 0u; int firstField = -1; size_t firstAt = 0;
    for (size_t i = 0; i < gpu.size() && i < ref.size(); ++i)
        for (int j = 0; j < 16; ++j)
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
                               bCd.h, bXf.h, bStack.h, bNum.h, bNeeded.h })
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
// (so the filter list is empty), ONE rigid (so the
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
// shared modules survive them.  This one synthesises the twelve (5.12) device-side
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
// the kCvxCoreCloth ABI row { 8,8,8,8,8,8,8,8,8,4,8,80 } (5.12: the filter pair list
// and its count are gone; the writer is the PxgRigidDbContactWriter).
struct CvxCoreClothPush
{
    uint64_t stackPtr, midphasePairsNum, cmInputs, shapes, femClothes;
    uint64_t shapeToRigidRemapTable, transformCache, contactDistance, restDistance;
    uint32_t stackSizeBytes, pad76;
    uint64_t stackSizeNeededOnDevice;
    uint64_t pad88;                   // paramAlign(80) = 16
    uint64_t w_outPoint, w_outNormalPen, w_outBarycentric, w_outContactInfo, w_outRigidMaterialIndex;
    uint64_t w_totalContactCount, w_contactByRigid, w_tempContactByRigid, w_contactIndexSortedByRigid;
    uint32_t w_maxNumContacts, pad172;
};
static_assert(sizeof(CvxCoreClothPush) == 176, "convex-core cloth push blob must be 176 bytes");
static_assert(offsetof(CvxCoreClothPush, stackSizeBytes) == 72, "stackSizeBytes @72");
static_assert(offsetof(CvxCoreClothPush, stackSizeNeededOnDevice) == 80, "stackSizeNeeded @80");
static_assert(offsetof(CvxCoreClothPush, w_outPoint) == 96, "writer @96");
static_assert(offsetof(CvxCoreClothPush, w_maxNumContacts) == 168, "maxNumContacts @168");

// Points every writer field of `p` at `w`.
static void fillCvxWriter(CvxCoreClothPush& p, const WriterBufs& w)
{
    p.w_outPoint = w.point.addr; p.w_outNormalPen = w.normalPen.addr;
    p.w_outBarycentric = w.barycentric.addr; p.w_outContactInfo = w.info.addr;
    p.w_outRigidMaterialIndex = w.rigidMat.addr; p.w_totalContactCount = w.count.addr;
    p.w_contactByRigid = w.byRigid.addr; p.w_tempContactByRigid = w.tempByRigid.addr;
    p.w_contactIndexSortedByRigid = w.idxByRigid.addr; p.w_maxNumContacts = w.maxContacts;
}

// PxComputeBarycentric, the TRIANGLE overload (PxMathUtils.h:390-409), transcribed
// operation for operation so the comparison can be BIT-EXACT.  This TU is built
// -ffp-contract=off, which is what the shaders' NoContraction stamp matches.
//
// NOT the corpus's `computeBarycentric` (pxgSlangClothTriContact.slang, from femClothPrimitives.cu):
// those are femClothPrimitives.cu's function, the same Cramer arithmetic plus a
// [-1e-4, 1+1e-4] acceptance window whose bool gates a contact.  The kernel under
// test has no such gate, and transcribing the windowed one would import a decision
// that does not belong on this path.
//
// 5.12: the barycentric is taken at the CLOTH'S OWN WITNESS, `clothPoint = points[i] -
// normal * dists[i] * 0.5f` (convexCoreCollision.cu:1248-1252), half a separation back
// from the written midpoint `points[i] + normal * dists[i] * 0.5f`.  The host only sees
// the written point, so it steps back the whole separation, (point - h) - h with
// h = normal * dist * 0.5 -- not bit-identical to the kernel's points[i] - h, so the
// comparisons below use a 2e-6 tolerance (the tilted case's witness shift moves the
// barycentric by ~1e-2).
static V3c cvxClothWitness(const F4& point, const F4& normalPen)
{
    // The written normal is -normal; the separation is written raw.
    const float hx = -normalPen.x * normalPen.w * 0.5f;
    const float hy = -normalPen.y * normalPen.w * 0.5f;
    const float hz = -normalPen.z * normalPen.w * 0.5f;
    return V3c{ (point.x - hx) - hx, (point.y - hy) - hy, (point.z - hz) - hz };
}
static bool baryNear(const F4& a, const F4& b)
{
    return std::fabs(a.x - b.x) <= 2e-6f && std::fabs(a.y - b.y) <= 2e-6f && std::fabs(a.z - b.z) <= 2e-6f;
}

// P9.94d: it is TrimeshBvh::getTriBarycentric (convexCoreCollision.cu:672-680) that the
// kernel mirrors, and that transforms the three vertices by the cloth's
// `scale.toMat33()` FIRST.  The cloth's PxMeshScale is default-constructed, i.e.
// identity, and an identity PxMat33 transform is exact on nonzero coordinates but
// maps -0 to +0 -- so the oracle goes through the real PxMeshScale().toMat33() too.
static F4 cpuTriBarycentricRaw(V3c a, V3c b, V3c c, V3c p);
static F4 cpuTriBarycentric(V3c a, V3c b, V3c c, V3c p)
{
    const physx::PxMat33 xform = physx::PxMeshScale().toMat33();
    auto xf = [&](const V3c& v) {
        const physx::PxVec3 r = xform.transform(physx::PxVec3(v.x, v.y, v.z));
        return V3c{ r.x, r.y, r.z };
    };
    return cpuTriBarycentricRaw(xf(a), xf(b), xf(c), p);
}
static F4 cpuTriBarycentricRaw(V3c a, V3c b, V3c c, V3c p)
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
    cloth.mTriangleVertexIndices_ownedEdges = reinterpret_cast<uint4*>(bTris.addr);
    // mTriMeshData is deliberately left NULL: this kernel never traverses, so
    // getNodes() and scaleBounds() are unreachable from it (convexCoreCollision.cu:1195-1279).
    // The kernel indexes femClothes[clothId] directly, so the array must REACH
    // index CLOTH_ID.  Uploading one descriptor and then asking for index 3 reads
    // off the end -- which is how the first run of this fixture segfaulted, and it
    // is precisely the mistake a scene with one cloth can never teach.
    // ---- the filter (5.12), which the scene has none of -------------------
    // The cloth's mRigidSurfaceFilterPairs, sorted (ref0, ref1), searched once per
    // pair for (rigid filter ref, TRIANGLE).  RIGID_NODE_IND's low word is not
    // 0xffffffff, so the rigid is dynamic and its filter ref is the node itself.
    // Two near misses, neither of which may suppress anything:
    //   - triangle 0's first VERTEX under the right rigid (5.6.1's per-vertex key);
    //   - triangle 0 under ANOTHER rigid.
    // CASE E below adds the real (rigid, triangle 1) entry.
    std::vector<FemPairInfoHost> filters = {
        FemPairInfoHost{ RIGID_NODE_IND, encClothVertex(CLOTH_ID, triInds[0]) },
        FemPairInfoHost{ RIGID_NODE_IND + 1ull, encClothTriangle(CLOTH_ID, 0u) },
    };
    DevBuf bFilters = makeDev(dev, filters);
    cloth.mRigidSurfaceFilterPairs = reinterpret_cast<PxgDeformablePairInfo*>(size_t(bFilters.addr));
    cloth.mNumRigidSurfaceFilterPairs = uint32_t(filters.size());

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
    push.stackSizeBytes = 4096u;           // far above 2 pairs: the clamp is a no-op here
    push.stackSizeNeededOnDevice = bStackNeeded.addr;
    fillCvxWriter(push, w);

    Kernel k("cvxCoreClothNpBda.spv", "convexCoreClothmeshNphase_Kernel32", {}, sizeof(CvxCoreClothPush));
    k.dispatch(/*gridX*/ 1u, {}, push, 1u, 1u);
    dev->flush();

    // ---- read back --------------------------------------------------------
    const uint32_t gCount = downloadBuffer<uint32_t>(dev, w.count.h, 1)[0];
    PXG_CHECK_MSG(gCount > 0u && gCount <= w.maxContacts,
                  "the kernel emitted %u contacts; a triangle 0.1 inside the box must produce "
                  "some, and the filter must NOT suppress it (its entries name a VERTEX ref and "
                  "another rigid, and the 5.12 test is one (rigid, triangle) search)", gCount);

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
    const std::vector<FemPairInfoHost> gInfo = downloadBuffer<FemPairInfoHost>(dev, w.info.h, gCount);
    const std::vector<uint64_t> gByRigid = downloadBuffer<uint64_t>(dev, w.byRigid.h, gCount);
    const std::vector<uint32_t> gTempByRigid = downloadBuffer<uint32_t>(dev, w.tempByRigid.h, gCount);
    const std::vector<uint32_t> gIdxByRigid = downloadBuffer<uint32_t>(dev, w.idxByRigid.h, gCount);
    const uint32_t gStackNeeded = downloadBuffer<uint32_t>(dev, bStackNeeded.h, 1)[0];

    // ---- the per-contact invariants ---------------------------------------
    size_t mm = 0;
    uint32_t sawTri0 = 0u, sawTri1 = 0u, onTriPlane = 0u, onBoxPlane = 0u;
    const uint64_t ref1Tri0 = encClothTriangle(CLOTH_ID, 0u);
    const uint64_t ref1Tri1 = encClothTriangle(CLOTH_ID, 1u);

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

        // 3. THE BARYCENTRIC IS OF THE CLOTH'S WITNESS (5.12).  Recomputed on the host
        //    from the kernel's OWN outPoint / outNormalPen, so this is a cross-check
        //    and not a re-derivation: a port that took the barycentric of the written
        //    midpoint (5.6.1), or permuted u/v/w, disagrees with its own output.
        const uint32_t tri = (gInfo[i].ref1 == ref1Tri1) ? 1u : 0u;
        const uint32_t b0 = triInds[tri * 4u + 0u], b1 = triInds[tri * 4u + 1u], b2 = triInds[tri * 4u + 2u];
        const V3c a{ verts[b0].x, verts[b0].y, verts[b0].z };
        const V3c b{ verts[b1].x, verts[b1].y, verts[b1].z };
        const V3c c{ verts[b2].x, verts[b2].y, verts[b2].z };
        const F4 refBary = cpuTriBarycentric(a, b, c, cvxClothWitness(gPoint[i], gNP[i]));
        if (!baryNear(refBary, gBary[i]))
        { ++mm; std::printf("  cvxCoreCloth[%u] outBarycentric (%g,%g,%g) != PxComputeBarycentric of "
                            "the cloth witness (%g,%g,%g)\n", i, gBary[i].x, gBary[i].y, gBary[i].z,
                            refBary.x, refBary.y, refBary.z); }

        // 4. THE MATERIAL INDEX (5.12): its own PxU16 stream, narrowed by the store.
        const uint32_t gMat = readRigidMat(dev, w, i);
        if (gMat != (MATERIAL_INDEX & 0xffffu))
        { ++mm; std::printf("  cvxCoreCloth[%u] rigidMaterialIndex = 0x%04x, expected 0x%04x\n",
                            i, gMat, MATERIAL_INDEX & 0xffffu); }

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
        if (gInfo[i].ref0 != RIGID_NODE_IND)
        { ++mm; std::printf("  cvxCoreCloth[%u] info.ref0 = 0x%016llx\n",
                            i, (unsigned long long)gInfo[i].ref0); }

        // 6. THE CLOTH TRIANGLE REF.  CLOTH_ID is 3, not 0, so both the id itself and
        //    the PxgDeformableRef shift are observable here.
        if (gInfo[i].ref1 != ref1Tri0 && gInfo[i].ref1 != ref1Tri1)
        { ++mm; std::printf("  cvxCoreCloth[%u] info.ref1 = 0x%016llx, expected 0x%016llx (tri 0) or "
                            "0x%016llx (tri 1) -- PxgEncodeDeformableTriangle(clothId %u, tri)\n",
                            i, (unsigned long long)gInfo[i].ref1, (unsigned long long)ref1Tri0,
                            (unsigned long long)ref1Tri1, CLOTH_ID); }
        if (gInfo[i].ref1 == ref1Tri0) ++sawTri0;
        if (gInfo[i].ref1 == ref1Tri1) ++sawTri1;

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
        fillCvxWriter(p2, w2);

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
        fillCvxWriter(p3, w3);

        k.dispatch(1u, {}, p3, 1u, 1u);
        dev->flush();

        const uint32_t c3 = downloadBuffer<uint32_t>(dev, w3.count.h, 1)[0];
        const uint32_t sn3v = downloadBuffer<uint32_t>(dev, bSn3.h, 1)[0];
        const std::vector<FemPairInfoHost> i3 = downloadBuffer<FemPairInfoHost>(dev, w3.info.h, c3 ? c3 : 1u);
        uint32_t tri1Leaked = 0u;
        for (uint32_t i = 0; i < c3; ++i)
            if (i3[i].ref1 == ref1Tri1) ++tri1Leaked;

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
    // CASE D -- A TILTED TRIANGLE, so the witness shift moves the barycentric
    // (5.12 takes it at the cloth's witness, not at the written midpoint).
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
        fillCvxWriter(pD, wD);

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
                const F4 refB = cpuTriBarycentric(ta, tb, tc, cvxClothWitness(dPoint[i], dNP[i]));
                if (!baryNear(refB, dBary[i]) || f2u(dBary[i].w) != f2u(0.0f))
                { ++dm; std::printf("  cvxCoreCloth tilted[%u] bary (%g,%g,%g,%g) != of the cloth witness "
                                    "(%g,%g,%g,0)\n", i, dBary[i].x, dBary[i].y, dBary[i].z,
                                    dBary[i].w, refB.x, refB.y, refB.z); }
                // The non-triviality guard: if the witness shift did NOT move the
                // barycentric -- which is what happens when the normal is perpendicular
                // to the triangle -- then this case cannot tell 5.12's witness from
                // 5.6.1's written midpoint and must say so rather than pass quietly.
                const F4 midB = cpuTriBarycentric(ta, tb, tc, V3c{ dPoint[i].x, dPoint[i].y, dPoint[i].z });
                if (!baryNear(midB, refB)) ++moved;
            }
            PXG_CHECK_MSG(dm == 0, "tilted-triangle barycentric mismatches: %zu of %u", dm, cD);
            PXG_CHECK_MSG(moved > 0u,
                          "the tilted case is DEGENERATE: for all %u contacts the barycentric of "
                          "the cloth witness equals that of the written midpoint, so this case "
                          "cannot tell the two apart and proves nothing", cD);
            std::printf("    [ info ] cvxCoreCloth tilted: %u contacts, %u of them with a "
                        "shift-sensitive barycentric\n", cD, moved);
        }
        destroyWriter(dev, wD);
        dev->destroyBuffer(bPairsD.h); dev->destroyBuffer(bPairNumD.h); dev->destroyBuffer(bSnD.h);
    }

    // =======================================================================
    // CASE E -- THE FILTER BITES (5.12).  A (rigid, triangle 1) entry is added to
    // the cloth's sorted filter list: triangle 1 must vanish and triangle 0 keep
    // exactly the contacts it made above.
    // =======================================================================
    {
        std::vector<FemPairInfoHost> filtersE = filters;
        filtersE.push_back(FemPairInfoHost{ RIGID_NODE_IND, encClothTriangle(CLOTH_ID, 1u) });
        std::sort(filtersE.begin(), filtersE.end(), [](const FemPairInfoHost& a, const FemPairInfoHost& b) {
            return a.ref0 != b.ref0 ? a.ref0 < b.ref0 : a.ref1 < b.ref1;
        });
        DevBuf bFiltersE = makeDev(dev, filtersE);
        std::vector<PxgFEMCloth> clothArrE = clothArr;
        clothArrE[CLOTH_ID].mRigidSurfaceFilterPairs = reinterpret_cast<PxgDeformablePairInfo*>(size_t(bFiltersE.addr));
        clothArrE[CLOTH_ID].mNumRigidSurfaceFilterPairs = uint32_t(filtersE.size());
        DevBuf bClothE = makeDev(dev, clothArrE);
        std::vector<uint32_t> snE = { 0u };
        DevBuf bSnE = makeDev(dev, snE);
        WriterBufs wE = makeWriter(dev, 64u);

        CvxCoreClothPush pE = push;
        pE.femClothes = bClothE.addr;
        pE.stackSizeNeededOnDevice = bSnE.addr;
        fillCvxWriter(pE, wE);
        k.dispatch(1u, {}, pE, 1u, 1u);
        dev->flush();

        const uint32_t cE = downloadBuffer<uint32_t>(dev, wE.count.h, 1)[0];
        const std::vector<FemPairInfoHost> iE = downloadBuffer<FemPairInfoHost>(dev, wE.info.h, cE ? cE : 1u);
        uint32_t tri1E = 0u;
        for (uint32_t i = 0; i < cE; ++i)
            if (iE[i].ref1 == ref1Tri1) ++tri1E;
        PXG_CHECK_MSG(cE == sawTri0 && tri1E == 0u,
                      "with (rigid, triangle 1) filtered the kernel produced %u contacts (%u from "
                      "triangle 1); expected %u, all from triangle 0", cE, tri1E, sawTri0);

        destroyWriter(dev, wE);
        for (PxgBufferHandle h : { bFiltersE.h, bClothE.h, bSnE.h }) dev->destroyBuffer(h);
    }

    destroyWriter(dev, w);
    for (PxgBufferHandle h : { bVerts.h, bTris.h, bCloth.h, bShapes.h, bTc.h, bCin.h,
                               bCd.h, bRd.h, bRemap.h, bFilters.h, bPairs.h, bPairNum.h,
                               bStackNeeded.h })
        dev->destroyBuffer(h);
}

} // namespace
