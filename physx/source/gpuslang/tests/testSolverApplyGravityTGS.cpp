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

// Verification of solverApplyGravityTGS.slang: applyTGSSubstepGravity and its
// applyAcceleration helper.  A synthetic set of dynamic bodies (mixed
// disableGravity / linear / angular acceleration states, all six lock-flag bits,
// invertible sqrt-inverse-inertia matrices) is run through the per-slab velocity
// update; the whole body-velocity buffer is compared component-wise (almostEqual)
// against a C++ reference replicating applyAcceleration, including the
// PxMat33::getInverse() angular transform.  Remap targets are spaced so slots
// never alias; skipped bodies must leave their slots untouched.
// PXG_TEST solverApplyGravityTGS_applySingularInertiaZeroSign (P9.86c) adds a directed
// bitwise case: singular sqrt-inverse-inertia (zero inverse-inertia axis, axis-aligned
// orientation) takes getInverse()'s PxMat33(PxIdentity) path, whose product keeps its
// computed-zero terms and turns a -0 acceleration lane into +0.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct GravityParams
{
    uint32_t numSlabs, allBodiesCount, bodyOffset, outputOffset, numThreadsPerBody;
    float gravityX, gravityY, gravityZ, stepDt;
};
struct F4 { float x, y, z, w; };
struct BodySimInput { uint32_t disableGravity, lockFlags; float accelScale; uint32_t pad0; F4 linAccel, angAccel; };
struct Mat33 { F4 col0, col1, col2; };

const uint32_t INVALID = 0xffffffffu;
const uint32_t LOCK_LIN_X = 1, LOCK_LIN_Y = 2, LOCK_LIN_Z = 4, LOCK_ANG_X = 8, LOCK_ANG_Y = 16, LOCK_ANG_Z = 32;

uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// Mutant twin of the singular branch: 1 = the old port shortcut "return v" (drops the
// computed-zero terms of PxMat33(PxIdentity) * v).  0 = the CUDA semantics.
int gIdentityMulVariant = 0;
uint32_t gSingularCalls = 0; // coverage: singular (identity) branch taken

uint32_t nextPow2(uint32_t v) { uint32_t p = 1; while (p < v) p <<= 1; return p; }

// PxMat33::getInverse() * v (column-major), identity if singular
void invInertiaMul(const Mat33& m, float vx, float vy, float vz, float& ox, float& oy, float& oz)
{
    const float c0x = m.col0.x, c0y = m.col0.y, c0z = m.col0.z;
    const float c1x = m.col1.x, c1y = m.col1.y, c1z = m.col1.z;
    const float c2x = m.col2.x, c2y = m.col2.y, c2z = m.col2.z;
    // det = c0 . (c1 x c2)
    const float crx = c1y * c2z - c1z * c2y, cry = c1z * c2x - c1x * c2z, crz = c1x * c2y - c1y * c2x;
    const float det = c0x * crx + c0y * cry + c0z * crz;
    if (det != 0.f)
    {
        const float invDet = 1.f / det;
        const float i0x = invDet * (c1y * c2z - c2y * c1z), i0y = invDet * -(c0y * c2z - c2y * c0z), i0z = invDet * (c0y * c1z - c0z * c1y);
        const float i1x = invDet * -(c1x * c2z - c1z * c2x), i1y = invDet * (c0x * c2z - c0z * c2x), i1z = invDet * -(c0x * c1z - c0z * c1x);
        const float i2x = invDet * (c1x * c2y - c1y * c2x), i2y = invDet * -(c0x * c2y - c0y * c2x), i2z = invDet * (c0x * c1y - c1x * c0y);
        ox = i0x * vx + i1x * vy + i2x * vz;
        oy = i0y * vx + i1y * vy + i2y * vz;
        oz = i0z * vx + i1z * vy + i2z * vz;
    }
    else
    {
        ++gSingularCalls;
        if (gIdentityMulVariant == 1) { ox = vx; oy = vy; oz = vz; }
        else
        {
            // PxMat33(PxIdentity) (PxMat33.h:240) * v = col0*v.x + col1*v.y + col2*v.z (transform,
            // PxMat33.h:346): the zero terms are kept, so (-0) + 0*(+x) = +0.
            ox = (1.f * vx + 0.f * vy) + 0.f * vz;
            oy = (0.f * vx + 1.f * vy) + 0.f * vz;
            oz = (0.f * vx + 0.f * vy) + 1.f * vz;
        }
    }
}

void applyAccelRef(std::vector<F4>& buf, uint32_t bodyIndex, float gx, float gy, float gz, float stepDt,
                   const F4& lin, const F4& ang, bool hasAng, float accelScale, uint32_t lockFlags,
                   const Mat33& inertia, uint32_t angOff)
{
    F4 v = buf[bodyIndex];
    v.x += accelScale * gx + lin.x * stepDt;
    v.y += accelScale * gy + lin.y * stepDt;
    v.z += accelScale * gz + lin.z * stepDt;
    if (lockFlags & LOCK_LIN_X) v.x = 0.f;
    if (lockFlags & LOCK_LIN_Y) v.y = 0.f;
    if (lockFlags & LOCK_LIN_Z) v.z = 0.f;
    if (hasAng)
    {
        F4 a = buf[bodyIndex + angOff];
        float ax, ay, az; invInertiaMul(inertia, ang.x, ang.y, ang.z, ax, ay, az);
        v.w += ax * stepDt; a.x += ay * stepDt; a.y += az * stepDt;
        if (lockFlags & LOCK_ANG_X) v.w = 0.f;
        if (lockFlags & LOCK_ANG_Y) a.x = 0.f;
        if (lockFlags & LOCK_ANG_Z) a.y = 0.f;
        buf[bodyIndex + angOff] = a;
    }
    buf[bodyIndex] = v;
}

PXG_TEST(solverApplyGravityTGS_apply)
{
    std::mt19937 rng(37010);
    const uint32_t numSlabs = 4;
    const uint32_t allBodiesCount = 200;
    const uint32_t bodyOffset = 40;
    const uint32_t numThreadsPerBody = std::min(nextPow2(numSlabs), 32u); // 4
    const uint32_t numBodiesPerWarp = 32u / numThreadsPerBody;            // 8
    const uint32_t outputOffset = 70000;
    const uint32_t bodyVelElems = 72000;
    const float gx = 0.f, gy = -9.81f, gz = 0.3f, stepDt = 0.016f;

    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 100000) / 100000.f); };

    // node-index indirection: a permutation of [0, allBodiesCount)
    std::vector<uint32_t> nodeIndex(allBodiesCount);
    for (uint32_t i = 0; i < allBodiesCount; ++i) nodeIndex[i] = i;
    std::shuffle(nodeIndex.begin(), nodeIndex.end(), rng);

    // body-sim inputs, indexed by node index
    std::vector<BodySimInput> sim(allBodiesCount);
    for (uint32_t n = 0; n < allBodiesCount; ++n)
    {
        BodySimInput s{};
        s.disableGravity = (rng() % 3 == 0) ? 1u : 0u;      // ~1/3 gravity-disabled
        s.lockFlags = rng() % 64u;                          // exercise all six lock bits
        s.accelScale = rf(0.5f, 1.5f);
        const bool wantLin = rng() % 2 == 0;
        const bool wantAng = rng() % 2 == 0;
        s.linAccel = wantLin ? F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), 0 } : F4{ 0, 0, 0, 0 };
        s.angAccel = wantAng ? F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 } : F4{ 0, 0, 0, 0 };
        sim[n] = s;
    }

    // sqrt-inverse-inertia per body: diagonally dominant -> invertible
    std::vector<Mat33> inertia(allBodiesCount);
    for (uint32_t i = 0; i < allBodiesCount; ++i)
    {
        Mat33 m{};
        m.col0 = { rf(3, 5), rf(-1, 1), rf(-1, 1), 0 };
        m.col1 = { rf(-1, 1), rf(3, 5), rf(-1, 1), 0 };
        m.col2 = { rf(-1, 1), rf(-1, 1), rf(3, 5), 0 };
        inertia[i] = m;
    }

    // per-(body,slab) remap references for dynamic bodies [bodyOffset, allBodiesCount)
    const uint32_t nPairs = numSlabs * (allBodiesCount - bodyOffset);
    std::vector<uint32_t> refs(nPairs);
    for (uint32_t i = 0; i < nPairs; ++i)
        refs[i] = (rng() % 100 < 15) ? INVALID : (96u * i);

    // initial body velocities: distinct per slot
    std::vector<F4> bodyVelInit(bodyVelElems);
    for (uint32_t e = 0; e < bodyVelElems; ++e)
        bodyVelInit[e] = { rf(-5, 5), rf(-5, 5), rf(-5, 5), rf(-5, 5) };

    // ---- C++ reference ----
    std::vector<F4> refBody = bodyVelInit;
    for (uint32_t bodyId = bodyOffset; bodyId < allBodiesCount; ++bodyId)
    {
        const BodySimInput& s = sim[nodeIndex[bodyId]];
        const bool hasLin = s.linAccel.x != 0.f || s.linAccel.y != 0.f || s.linAccel.z != 0.f;
        const bool hasAng = s.angAccel.x != 0.f || s.angAccel.y != 0.f || s.angAccel.z != 0.f;
        if (s.disableGravity == 0u || hasLin || hasAng)
        {
            const float agx = stepDt * gx, agy = stepDt * gy, agz = stepDt * gz;
            for (uint32_t b = 0; b < numSlabs; ++b)
            {
                const uint32_t r = refs[numSlabs * (bodyId - bodyOffset) + b];
                if (r != INVALID)
                    applyAccelRef(refBody, r, agx, agy, agz, stepDt, s.linAccel, s.angAccel, hasAng,
                                  s.accelScale, s.lockFlags, inertia[bodyId], 32);
            }
            applyAccelRef(refBody, outputOffset + bodyId, agx, agy, agz, stepDt, s.linAccel, s.angAccel, hasAng,
                          s.accelScale, s.lockFlags, inertia[bodyId], allBodiesCount);
        }
    }

    // ---- GPU ----
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t numWarps = (allBodiesCount + numBodiesPerWarp - 1) / numBodiesPerWarp; // 25

    PxgBufferHandle bRefs = createBufferFrom(dev, refs);
    PxgBufferHandle bBody = createBufferFrom(dev, bodyVelInit);
    PxgBufferHandle bNode = createBufferFrom(dev, nodeIndex);
    PxgBufferHandle bSim = createBufferFrom(dev, sim);
    PxgBufferHandle bInertia = createBufferFrom(dev, inertia);

    Kernel k("solverApplyGravityTGS.spv", "applyTGSSubstepGravity", { 4, 16, 4, 48, 48 }, sizeof(GravityParams));
    GravityParams p = { numSlabs, allBodiesCount, bodyOffset, outputOffset, numThreadsPerBody, gx, gy, gz, stepDt };
    k.dispatch(numWarps, { bRefs, bBody, bNode, bSim, bInertia }, p);
    dev->flush();

    const auto gBody = downloadBuffer<F4>(dev, bBody, bodyVelElems);
    size_t mm = 0;
    for (uint32_t e = 0; e < bodyVelElems; ++e)
    {
        const F4& a = gBody[e];
        const F4& r = refBody[e];
        if (!almostEqual(a.x, r.x, 1e-4f, 1e-4f) || !almostEqual(a.y, r.y, 1e-4f, 1e-4f) ||
            !almostEqual(a.z, r.z, 1e-4f, 1e-4f) || !almostEqual(a.w, r.w, 1e-4f, 1e-4f))
            ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "applyTGSSubstepGravity: %zu mismatches", mm);

    for (PxgBufferHandle h : { bRefs, bBody, bNode, bSim, bInertia }) dev->destroyBuffer(h);
}

// P9.86c: singular sqrt-inverse-inertia -> getInverse() returns PxMat33(PxIdentity)
// (solverMultiBlockTGS.cu:1592, PxMat33.h:240/:346).  Lane i of the identity product is
// (1*v_i + 0*v_j) + 0*v_k, so a -0 lane next to a +0/positive lane becomes +0.  Three bodies
// with one zero inverse-inertia axis at a 180-degree orientation and a direct-GPU-API torque
// (setRigidDynamicTorque, updateBodiesAndShapes.cu:1269-1276) carry a -0 angular-acceleration
// lane; their momocity lane is -0 (sqrtInertia * angVel of the same body).  The CUDA stores
// -0 + (+0*dt) = +0; the old port shortcut stores -0.  Compared BITWISE.
PXG_TEST(solverApplyGravityTGS_applySingularInertiaZeroSign)
{
    const uint32_t numSlabs = 2, allBodiesCount = 4, bodyOffset = 1, outputOffset = 64;
    const uint32_t numThreadsPerBody = std::min(nextPow2(numSlabs), 32u); // 2
    const uint32_t bodyVelElems = 80;
    const float gx = 0.f, gy = -9.81f, gz = 0.3f, stepDt = 0.016f;
    const float s5 = std::sqrt(5.f);

    const std::vector<uint32_t> nodeIndex = { 3, 2, 1, 0 };
    std::vector<BodySimInput> sim(allBodiesCount);
    std::vector<Mat33> inertia(allBodiesCount);
    // body 0: below bodyOffset, skipped
    inertia[0] = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 } };
    sim[nodeIndex[0]] = BodySimInput{ 0u, 0u, 1.f, 0u, { 1, 1, 1, 0 }, { 1, 1, 1, 0 } };
    const uint32_t lockLin = LOCK_LIN_X | LOCK_LIN_Y | LOCK_LIN_Z; // no angular lock
    // A: invInertia (0,0,5), q = (0,0,-1,0), torque (-1,0,-1), angVel (-1,0,-1)
    inertia[1] = { { 0.f, -0.f, 0.f, 0 }, { -0.f, 0.f, -0.f, 0 }, { 0.f, -0.f, s5, 0 } };
    sim[nodeIndex[1]] = BodySimInput{ 1u, lockLin, 1.f, 0u, { 0, 0, 0, 0 }, { -0.f, 0.f, -5.f, 0 } };
    // B: invInertia (5,0,0), q = (-1,0,0,0), torque (-1,-1,+0), angVel (-1,-1,0)
    inertia[2] = { { s5, 0.f, -0.f, 0 }, { 0.f, 0.f, -0.f, 0 }, { -0.f, -0.f, 0.f, 0 } };
    sim[nodeIndex[2]] = BodySimInput{ 1u, lockLin, 1.f, 0u, { 0, 0, 0, 0 }, { -5.f, -0.f, 0.f, 0 } };
    // C: invInertia (0,5,0), q = (0,-1,0,0), torque (1,-1,-1), angVel (1,-1,-1)
    inertia[3] = { { 0.f, -0.f, -0.f, 0 }, { -0.f, s5, 0.f, 0 }, { -0.f, 0.f, 0.f, 0 } };
    sim[nodeIndex[3]] = BodySimInput{ 1u, lockLin, 1.f, 0u, { 0, 0, 0, 0 }, { 0.f, -5.f, -0.f, 0 } };

    const std::vector<uint32_t> refs = { 0, 4, 8, 12, 16, 20 };
    std::vector<F4> init(bodyVelElems);
    for (uint32_t e = 0; e < bodyVelElems; ++e) init[e] = { 100.f + e, 200.f + e, 300.f + e, 400.f + e };
    // body 1 (A): linear 0,4,65 (w = -0); angular 32,36,69
    for (uint32_t e : { 0u, 4u, 65u }) init[e] = { 0.f, 0.f, 0.f, -0.f };
    for (uint32_t e : { 32u, 36u, 69u }) init[e] = { 0.f, 0.f, 0.25f, -0.5f };
    // body 2 (B): linear 8,12,66; angular 40,44,70 (x = -0)
    for (uint32_t e : { 8u, 12u, 66u }) init[e] = { 0.f, 0.f, 0.f, 0.f };
    for (uint32_t e : { 40u, 44u, 70u }) init[e] = { -0.f, 0.f, 0.25f, -0.5f };
    // body 3 (C): linear 16,20,67; angular 48,52,71 (y = -0)
    for (uint32_t e : { 16u, 20u, 67u }) init[e] = { 0.f, 0.f, 0.f, 0.f };
    for (uint32_t e : { 48u, 52u, 71u }) init[e] = { 0.f, -0.f, 0.25f, -0.5f };

    auto runRef = [&](int variant) {
        std::vector<F4> buf = init;
        gIdentityMulVariant = variant;
        gSingularCalls = 0;
        for (uint32_t bodyId = bodyOffset; bodyId < allBodiesCount; ++bodyId)
        {
            const BodySimInput& s = sim[nodeIndex[bodyId]];
            const bool hasLin = s.linAccel.x != 0.f || s.linAccel.y != 0.f || s.linAccel.z != 0.f;
            const bool hasAng = s.angAccel.x != 0.f || s.angAccel.y != 0.f || s.angAccel.z != 0.f;
            if (s.disableGravity == 0u || hasLin || hasAng)
            {
                const float agx = stepDt * gx, agy = stepDt * gy, agz = stepDt * gz;
                for (uint32_t b = 0; b < numSlabs; ++b)
                {
                    const uint32_t r = refs[numSlabs * (bodyId - bodyOffset) + b];
                    if (r != INVALID)
                        applyAccelRef(buf, r, agx, agy, agz, stepDt, s.linAccel, s.angAccel, hasAng,
                                      s.accelScale, s.lockFlags, inertia[bodyId], 32);
                }
                applyAccelRef(buf, outputOffset + bodyId, agx, agy, agz, stepDt, s.linAccel, s.angAccel, hasAng,
                              s.accelScale, s.lockFlags, inertia[bodyId], allBodiesCount);
            }
        }
        gIdentityMulVariant = 0;
        return buf;
    };
    const std::vector<F4> want = runRef(0);
    const uint32_t singular = gSingularCalls;
    const std::vector<F4> twin = runRef(1);
    gSingularCalls = 0;

    PXG_CHECK_MSG(singular == 9, "singular identity branch taken %u times, expected 9", singular);
    uint32_t zsA = 0, zsB = 0, zsC = 0, diffLanes = 0;
    for (uint32_t e : { 0u, 4u, 65u }) zsA += (f2u(want[e].w) == 0u && f2u(twin[e].w) == 0x80000000u);
    for (uint32_t e : { 40u, 44u, 70u }) zsB += (f2u(want[e].x) == 0u && f2u(twin[e].x) == 0x80000000u);
    for (uint32_t e : { 48u, 52u, 71u }) zsC += (f2u(want[e].y) == 0u && f2u(twin[e].y) == 0x80000000u);
    for (uint32_t e = 0; e < bodyVelElems; ++e)
    {
        const float* a = &want[e].x; const float* b = &twin[e].x;
        for (int l = 0; l < 4; ++l) diffLanes += f2u(a[l]) != f2u(b[l]);
    }
    PXG_CHECK_MSG(zsA == 3 && zsB == 3 && zsC == 3,
                  "mutant twin (identity shortcut) no longer changes lanes A.w/B.x/C.y (%u/%u/%u of 3) -- the case does not observe it",
                  zsA, zsB, zsC);
    PXG_CHECK_MSG(diffLanes == 9, "mutant twin differs in %u lanes, expected 9", diffLanes);

    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bRefs = createBufferFrom(dev, refs);
    PxgBufferHandle bBody = createBufferFrom(dev, init);
    PxgBufferHandle bNode = createBufferFrom(dev, nodeIndex);
    PxgBufferHandle bSim = createBufferFrom(dev, sim);
    PxgBufferHandle bInertia = createBufferFrom(dev, inertia);

    Kernel k("solverApplyGravityTGS.spv", "applyTGSSubstepGravity", { 4, 16, 4, 48, 48 }, sizeof(GravityParams));
    GravityParams p = { numSlabs, allBodiesCount, bodyOffset, outputOffset, numThreadsPerBody, gx, gy, gz, stepDt };
    k.dispatch(1, { bRefs, bBody, bNode, bSim, bInertia }, p);
    dev->flush();

    const auto got = downloadBuffer<F4>(dev, bBody, bodyVelElems);
    uint32_t mm = 0;
    for (uint32_t e = 0; e < bodyVelElems; ++e)
    {
        const float* a = &got[e].x; const float* r = &want[e].x;
        for (int l = 0; l < 4; ++l)
            if (f2u(a[l]) != f2u(r[l]))
            {
                if (mm == 0)
                    std::printf("  first mismatch: elem %u lane %d got 0x%08x want 0x%08x\n", e, l, f2u(a[l]), f2u(r[l]));
                ++mm;
            }
    }
    PXG_CHECK_MSG(mm == 0, "applyTGSSubstepGravity singular zero-sign: %u bitwise lane mismatches", mm);

    for (PxgBufferHandle h : { bRefs, bBody, bNode, bSim, bInertia }) dev->destroyBuffer(h);
}

} // namespace
