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

// P9.80 -- the two PBD buffer features no scene had ever created.
//
//   ps_update_volume_bound (particlesystem.cu:683)  -> psVolumeBoundBda.slang
//   ps_solveShapes         (particlesystem.cu:5121) -> psShapeMatchBda.slang
//
// WHAT THESE FIXTURES CHOOSE THAT PXG_SCENE=particlerigid CANNOT.  The scene
// proves both kernels run and that the shape matching does the work (y-extent
// 1.4999 with it, 0.0000 with the coefficient set to zero).  It cannot prove much
// more, because every indirection in it collapses to the identity:
//
//   - ONE particle system, so `activeParticleSystems[blockIdx.z]` is an identity;
//   - ONE particle buffer at index 0, so mParticleBufferRunsum[...] is 0 and the
//     buffer-local and system-global particle ids are the same number;
//   - ONE rigid buffer and ONE rigid, so startIndex is 0 and the per-rigid slice
//     is the whole array;
//   - a freshly built system whose mUnsortedToSortedMapping is very nearly the
//     identity, so a port that skipped the reverse lookup would read almost the
//     same particles;
//   - 64 particles == exactly ONE pass of the 64-thread block, so the `passes`
//     loop runs once and the accumulate-across-passes into shared never happens;
//   - and the grid is sized so that the upstream `rigidId > numRigids` off-by-one
//     block (particlesystem.cu:5150, '>' not '>=') is never launched.
//
// So these fixtures use TWO systems reached in REVERSE order, TWO rigid buffers
// pointing at DIFFERENT particle buffers, TWO rigids per buffer with a non-zero
// startIndex, a non-zero mParticleBufferRunsum, a real PERMUTATION for the
// reverse lookup, a particle count that is NOT a multiple of 64 and spans more
// than one pass, and a grid that DOES launch the off-by-one block.
//
// ---------------------------------------------------------------------------
// WHAT IS COMPARED BIT-EXACTLY, AND WHAT IS NOT -- AND WHY THE LINE IS THERE
// ---------------------------------------------------------------------------
// ps_solveShapes' output splits cleanly in two:
//
//   * mRigidTranslations is the reduced-and-drift-clamped centre of mass.  It is
//     pure float addition through blockReduction's XOR butterfly, with no
//     transcendental anywhere, so it is compared BIT-EXACTLY.  This is the half
//     that tests the association tree -- the thing a port is most likely to get
//     subtly wrong, and the thing psBoundsBda's min/max shape would have broken.
//   * mRigidRotations, mDelta and mNormalArray all pass through
//     extractRotation, whose PxQuat(angle, axis) constructor calls PxSinCos.
//     sin/cos have no cross-backend rounding guarantee, so these are compared to
//     a TOLERANCE and the fixture says so rather than pretending otherwise.
//
// ps_update_volume_bound is compared bit-exactly throughout, against vbOracle,
// which carries the CUDA's two rules: select per thread (PxMin/PxMax), then
// fminf/fmaxf across the block.  Min and max are association-free only under
// fminf/fmaxf's total order (-0 < +0, NaN ignored, two NaNs -> 0x7fffffff);
// psVolumeBoundZeroSign supplies the +/-0 and NaN inputs where the rules matter.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

// host_defines.h expands __device__/__align__ before the engine headers
// (nocuda/ is on the include path), matching testPsContactGen.cpp.
#include "host_defines.h"
#include "PxgParticleSystem.h"
#include "PxParticleBuffer.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct V3 { float x, y, z; };

static uint64_t addrOf(PxgSlangComputeDevice* dev, PxgBufferHandle h) { return dev->getBufferDeviceAddress(h); }
static void putU64(std::vector<uint8_t>& b, size_t off, uint64_t v) { std::memcpy(b.data() + off, &v, 8); }
static void putU32(std::vector<uint8_t>& b, size_t off, uint32_t v) { std::memcpy(b.data() + off, &v, 4); }

// ---- byte layouts, pinned against the REAL engine structs ------------------
static const uint32_t SZ_PSYS = 656u;
static const uint32_t PS_UNSORTED_POS = 0u;
static const uint32_t PS_SORTED_POS = 24u;
static const uint32_t PS_U2S = 48u;
static const uint32_t PS_NUM_BUFFERS = 132u;
static const uint32_t PS_DELTA = 368u;
static const uint32_t PS_NORMALS = 384u;
static const uint32_t PS_SIM_BUFFERS = 400u;
static const uint32_t PS_BUFFER_RUNSUM = 408u;
static const uint32_t PS_RIGID_SIM_BUFFERS = 440u;
static const uint32_t PS_NUM_RIGID_BUFFERS = 468u;
static_assert(sizeof(physx::PxgParticleSystem) == SZ_PSYS, "PxgParticleSystem stride");
static_assert(offsetof(physx::PxgParticleSystem, mDelta) == PS_DELTA, "PS_DELTA");
static_assert(offsetof(physx::PxgParticleSystem, mNormalArray) == PS_NORMALS, "PS_NORMALS");
static_assert(offsetof(physx::PxgParticleSystem, mParticleSimBuffers) == PS_SIM_BUFFERS, "PS_SIM_BUFFERS");
static_assert(offsetof(physx::PxgParticleSystem, mParticleBufferRunsum) == PS_BUFFER_RUNSUM, "PS_BUFFER_RUNSUM");
static_assert(offsetof(physx::PxgParticleSystem, mRigidSimBuffers) == PS_RIGID_SIM_BUFFERS, "PS_RIGID_SIM_BUFFERS");
static_assert(offsetof(physx::PxgParticleSystem, mNumRigidBuffers) == PS_NUM_RIGID_BUFFERS, "PS_NUM_RIGID_BUFFERS");

static const uint32_t SZ_SIMBUF = 88u;
static const uint32_t SB_VOLUMES = 32u;
static const uint32_t SB_NUM_VOLUMES = 60u;
static_assert(sizeof(physx::PxgParticleSimBuffer) == SZ_SIMBUF, "PxgParticleSimBuffer stride");
static_assert(offsetof(physx::PxgParticleSimBuffer, mVolumes) == SB_VOLUMES, "SB_VOLUMES");
static_assert(offsetof(physx::PxgParticleSimBuffer, mNumVolumes) == SB_NUM_VOLUMES, "SB_NUM_VOLUMES");

// PxgParticleRigidSimBuffer (PxgParticleSystem.h:183) -- note the DECLARATION
// ORDER: coefficients first, and mRigidOffsets FOURTH, between the rotations and
// the local positions.  All six are 8-byte pointers, so a swap would be silent.
static const uint32_t SZ_RIGIDBUF = 56u;
static const uint32_t RB_COEFFS = 0u, RB_TRANSLATIONS = 8u, RB_ROTATIONS = 16u,
                      RB_OFFSETS = 24u, RB_LOCALPOS = 32u, RB_LOCALNRM = 40u,
                      RB_NUMRIGIDS = 48u, RB_BUFINDEX = 52u;
static_assert(sizeof(physx::PxgParticleRigidSimBuffer) == SZ_RIGIDBUF, "PxgParticleRigidSimBuffer stride");
static_assert(offsetof(physx::PxgParticleRigidSimBuffer, mRigidCoefficients) == RB_COEFFS, "RB_COEFFS");
static_assert(offsetof(physx::PxgParticleRigidSimBuffer, mRigidTranslations) == RB_TRANSLATIONS, "RB_TRANSLATIONS");
static_assert(offsetof(physx::PxgParticleRigidSimBuffer, mRigidRotations) == RB_ROTATIONS, "RB_ROTATIONS");
static_assert(offsetof(physx::PxgParticleRigidSimBuffer, mRigidOffsets) == RB_OFFSETS, "RB_OFFSETS");
static_assert(offsetof(physx::PxgParticleRigidSimBuffer, mRigidLocalPositions) == RB_LOCALPOS, "RB_LOCALPOS");
static_assert(offsetof(physx::PxgParticleRigidSimBuffer, mRigidLocalNormals) == RB_LOCALNRM, "RB_LOCALNRM");
static_assert(offsetof(physx::PxgParticleRigidSimBuffer, mNumRigids) == RB_NUMRIGIDS, "RB_NUMRIGIDS");
static_assert(offsetof(physx::PxgParticleRigidSimBuffer, mParticleBufferIndex) == RB_BUFINDEX, "RB_BUFINDEX");

static const uint32_t SZ_VOLUME = 32u;
static_assert(sizeof(physx::PxParticleVolume) == SZ_VOLUME, "PxParticleVolume stride");
static_assert(offsetof(physx::PxParticleVolume, particleIndicesOffset) == 24u, "volume offset field");
static_assert(offsetof(physx::PxParticleVolume, numParticles) == 28u, "volume count field");

// ---- push blobs, byte for byte the HOST packs ------------------------------
struct PsVolumeBoundPush { uint64_t particleSystems; uint64_t activeParticleSystemIndices; };
static_assert(sizeof(PsVolumeBoundPush) == 16, "ps_update_volume_bound host pack is 16 bytes");

struct PsSolveShapesPush
{
    uint64_t particleSystems;
    uint64_t activeParticleSystems;
    float dt;                 // DEAD in the kernel body; present because the host pack has it
    float biasCoefficient;
};
static_assert(sizeof(PsSolveShapesPush) == 24, "ps_solveShapes host pack is 24 bytes");

// ===========================================================================
// blockReduction<AddOpPxReal, PxReal, 64> (reduction.cuh:226-247), transcribed.
//
// THE ASSOCIATION TREE IS THE POINT.  Float addition is not associative, so this
// has to reproduce the CUDA's XOR butterfly exactly -- radius 16, 8, 4, 2, 1,
// LARGEST STRIDE FIRST -- twice: once per 32-lane warp, then once over the two
// warp sums zero-padded out to 32 lanes.  Those thirty +0.0f additions are real.
// ===========================================================================
static const uint32_t PS_SOLVE_SHAPE = 64u;
static const uint32_t SM_WARPS = PS_SOLVE_SHAPE / 32u;

static void warpButterflyAdd(float* v)   // 32 lanes in place
{
    for (uint32_t radius = 16u; radius > 0u; radius >>= 1)
    {
        float next[32];
        for (uint32_t l = 0; l < 32u; ++l) next[l] = v[l] + v[l ^ radius];
        std::memcpy(v, next, sizeof(next));
    }
}

// Returns the value blockReduction hands to WARP 0 (every lane of it).  Threads
// outside warp 0 get initialValue back; no call site here reads those.
static float blockReduceAdd64(const float* perThread, float initialValue)
{
    float warpSum[SM_WARPS];
    for (uint32_t w = 0; w < SM_WARPS; ++w)
    {
        float lanes[32];
        for (uint32_t l = 0; l < 32u; ++l) lanes[l] = perThread[w * 32u + l];
        warpButterflyAdd(lanes);
        warpSum[w] = lanes[0];
    }
    float lvl2[32];
    for (uint32_t l = 0; l < 32u; ++l) lvl2[l] = (l < SM_WARPS) ? warpSum[l] : initialValue;
    warpButterflyAdd(lvl2);
    return lvl2[0];
}

// ---- the quaternion helpers, from the .cu ----------------------------------
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }

static F4 quatMul(F4 a, F4 b)
{
    return F4{ a.w * b.x + b.w * a.x + a.y * b.z - b.y * a.z,
               a.w * b.y + b.w * a.y + a.z * b.x - b.z * a.x,
               a.w * b.z + b.w * a.z + a.x * b.y - b.x * a.y,
               a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}

static void mat33FromQuat(F4 q, V3& c0, V3& c1, V3& c2)
{
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    c0 = V3{ 1.0f - yy - zz, xy + zw, xz - yw };
    c1 = V3{ xy - zw, 1.0f - xx - zz, yz + xw };
    c2 = V3{ xz + yw, yz - xw, 1.0f - xx - yy };
}

static V3 quatRotate(F4 q, V3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
               vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
               vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2 };
}

// extractRotation (deformableUtils.cuh:46).  THE sin/cos SEAM IS HERE.
static F4 extractRotation(V3 a0, V3 a1, V3 a2, F4 q, int maxIter)
{
    const float eps = 1.0e-6f;
    for (int iter = 0; iter < maxIter; ++iter)
    {
        V3 r0, r1, r2;
        mat33FromQuat(q, r0, r1, r2);

        const V3 ca = cross3(r0, a0), cb = cross3(r1, a1), cc = cross3(r2, a2);
        V3 omega{ ca.x + cb.x + cc.x, ca.y + cb.y + cc.y, ca.z + cb.z + cc.z };

        const float s = 1.0f / (std::fabs(dot3(r0, a0) + dot3(r1, a1) + dot3(r2, a2)) + eps);
        omega = V3{ omega.x * s, omega.y * s, omega.z * s };

        // PxVec3::normalize(): returns the OLD magnitude, scales by a reciprocal
        // formed ONCE, and only when m > 0.
        const float w = std::sqrt(dot3(omega, omega));
        if (w > 0.0f) { const float r = 1.0f / w; omega = V3{ omega.x * r, omega.y * r, omega.z * r }; }

        // PxQuat(angle, unitAxis): halves the angle and calls PxSinCos.
        const float a = w * 0.5f;
        const float sa = std::sin(a), cav = std::cos(a);
        const F4 tempQ{ omega.x * sa, omega.y * sa, omega.z * sa, cav };

        q = quatMul(tempQ, q);   // LEFT multiply

        const float qs = 1.0f / std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        q = F4{ q.x * qs, q.y * qs, q.z * qs, q.w * qs };

        if (w < eps) break;      // AFTER the update
    }
    return q;
}

// ===========================================================================
// P9.86c: ps_update_volume_bound's two min/max rules, transcribed, with twins.
//
// The kernel (particlesystem.cu:683-777) uses TWO different operations:
//
//   * per thread (:725-737), seeded with +/-PX_MAX_F32, PxMin/PxMax<float>.
//     Those are intrinsics::selectMin/selectMax (PxMath.h:82/:98,
//     PxUnixMathIntrinsics.h:116-125): `a < b ? a : b` / `a > b ? a : b`.  On a
//     +/-0 tie OR a NaN pos they return pos, the LATER operand, so a thread keeps
//     the sign of the last zero it met and a NaN in its last particle replaces
//     whatever it had.
//   * across the block (:740-746, :765-775), warpReduction<MinOpFloat/MaxOpFloat>
//     = fminf/fmaxf (reduction.cuh:53/:83).  On NVIDIA those are minNum/maxNum:
//     a NaN operand yields the other, -0 < +0 in either operand order, and two
//     NaNs give the canonical NaN 0x7fffffff.  C leaves the zero tie and the
//     NaN/NaN payload unspecified; the project takes NVIDIA's rule for both, as
//     psVolumeBoundBda.slang's vbFminf/vbFmaxf do with a CONSTANT
//     asfloat(0x7fffffffu) -- so pinning that payload pins the port's own
//     constant, not a NaN the driver produced (contrast testFemClothSim's 0/0
//     normal, where the payload is the driver's and is compared as "both NaN").
//
// GLSL.std.450 FMin/FMax (Slang min/max) on llvmpipe, which the pre-P9.86 port
// used at both levels: lp_screen sets lower_fminmax_signed_zero, so
// nir_lower_alu makes them bcsel(feq(a,b), imin/imax(bits), fmin/fmax_nsz(a,b))
// -- a +/-0 tie gives -0 / +0 in EITHER order (P9.86 F4, pinned both ways by
// p86SignedZero) -- and gallivm's fmin_nsz returns the non-NaN operand of a
// one-NaN pair and an INPUT NaN for two.  So the tree reverts differ from
// fminf/fmaxf only on a NaN/NaN pair, and the per-thread revert only on a
// mixed-sign zero tie (select keeps the later sign) and on a NaN pos.
// ===========================================================================
static const float VB_MAX_F32 = 3.4028235e38f;
static uint32_t vbBits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float vbFromBits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// NVIDIA fminf/fmaxf (PTX min.f32 / max.f32).
static float nvFminf(float a, float b)
{
    if (std::isnan(a)) return std::isnan(b) ? vbFromBits(0x7fffffffu) : b;
    if (std::isnan(b)) return a;
    if (a < b) return a;
    if (b < a) return b;
    return std::signbit(a) ? a : b;     // equal: -0 wins a +/-0 tie
}
static float nvFmaxf(float a, float b)
{
    if (std::isnan(a)) return std::isnan(b) ? vbFromBits(0x7fffffffu) : b;
    if (std::isnan(b)) return a;
    if (a > b) return a;
    if (b > a) return b;
    return std::signbit(a) ? b : a;     // equal: +0 wins a +/-0 tie
}
// llvmpipe FMin/FMax (see above).  Two NaNs give src0; every NaN in this fixture
// carries one payload, so which input NaN comes back is moot.
static float lpFMin(float a, float b)
{
    if (std::isnan(b)) return a;
    if (std::isnan(a)) return b;
    if (a == b) return std::signbit(a) ? a : b;
    return (a < b) ? a : b;
}
static float lpFMax(float a, float b)
{
    if (std::isnan(b)) return a;
    if (std::isnan(a)) return b;
    if (a == b) return std::signbit(a) ? b : a;
    return (a > b) ? a : b;
}

// Mutant twins of the port (0 = the CUDA):
//   1 tree min as FMin            (psVolumeBoundBda-r2, half of -r1-uses)
//   2 tree max as FMax            (psVolumeBoundBda-r3, half of -r1-uses)
//   3 per-thread PxMin/PxMax as FMin/FMax  (psVolumeBoundBda-r4)
//   4 tree fminf/fmaxf keeps the FIRST operand of a +/-0 tie  (helper-body guard)
//   5 tree fminf/fmaxf keeps the SECOND operand of a +/-0 tie (helper-body guard)
// Twins run in the PORT's tree order (vbBlockMin: s = 512..1, v[t] = op(v[t], v[t+s]))
// because that is the order a mutated port would run in.
static int gVbVariant = 0;
struct VbCounters
{
    uint32_t laterZeroWins;   // non-NaN per-thread steps where the select != FMin/FMax
    uint32_t nanDropped;      // per-thread steps where a NaN pos replaced a real particle value
    uint32_t nanPairs;        // tree fminf/fmaxf steps whose two operands are both NaN
    uint32_t treeZeroTies;    // tree steps on a mixed-sign +/-0 pair
    uint32_t orderMismatch;   // CUDA butterfly order and port tree order disagree (must be 0)
};
static VbCounters gVbCnt = {};

static float vbTreeMin(float a, float b)
{
    if (std::isnan(a) && std::isnan(b)) ++gVbCnt.nanPairs;
    const bool tie = !std::isnan(a) && a == b && vbBits(a) != vbBits(b);
    if (tie) ++gVbCnt.treeZeroTies;
    if (gVbVariant == 1) return lpFMin(a, b);
    if (tie && gVbVariant == 4) return a;
    if (tie && gVbVariant == 5) return b;
    return nvFminf(a, b);
}
static float vbTreeMax(float a, float b)
{
    if (std::isnan(a) && std::isnan(b)) ++gVbCnt.nanPairs;
    const bool tie = !std::isnan(a) && a == b && vbBits(a) != vbBits(b);
    if (tie) ++gVbCnt.treeZeroTies;
    if (gVbVariant == 2) return lpFMax(a, b);
    if (tie && gVbVariant == 4) return a;
    if (tie && gVbVariant == 5) return b;
    return nvFmaxf(a, b);
}

// The port's single 1024-entry tree (psVolumeBoundBda.slang vbBlockMin/Max).
static float vbPortTree(std::vector<float> v, bool isMax)
{
    for (uint32_t s = 512u; s > 0u; s >>= 1)
        for (uint32_t t = 0; t < s; ++t)
            v[t] = isMax ? vbTreeMax(v[t], v[t + s]) : vbTreeMin(v[t], v[t + s]);
    return v[0];
}

// The CUDA's two levels: a 32-lane xor butterfly per warp, radius 16..1
// (reduction.cuh:188-200), lane 31 of warp w stored to s[c][w] (:748-757), then
// warp c reduces those 32 entries with the same butterfly, lane 31's result
// (:765-775).  CUDA operators only -- this is what variant 0 returns.
static float vbCudaTree(const std::vector<float>& leaves, bool isMax)
{
    auto butterfly = [&](float* lanes)
    {
        for (uint32_t r = 16u; r > 0u; r >>= 1)
        {
            float next[32];
            for (uint32_t l = 0; l < 32u; ++l)
                next[l] = isMax ? nvFmaxf(lanes[l], lanes[l ^ r]) : nvFminf(lanes[l], lanes[l ^ r]);
            std::memcpy(lanes, next, sizeof(next));
        }
        return lanes[31];
    };
    float perWarp[32];
    for (uint32_t w = 0; w < 32u; ++w)
    {
        float lanes[32];
        for (uint32_t l = 0; l < 32u; ++l) lanes[l] = leaves[w * 32u + l];
        perWarp[w] = butterfly(lanes);
    }
    return butterfly(perWarp);
}

// ps_update_volume_bound over the slice [offset, offset + n) of a float4 array:
// thread t of 1024 folds workIndex t, t+1024, ... (:725-737), then the block
// reduction.  For variant 0 the result is the CUDA two-level butterfly, and the
// port's tree order is computed alongside and must give the same bits (fminf/
// fmaxf with NVIDIA's rules are a total order with NaN as identity, so they are
// associative and commutative -- counted, not assumed: orderMismatch).
static void vbOracle(const std::vector<float>& pos4, uint32_t offset, uint32_t n, float* mn, float* mx)
{
    for (int c = 0; c < 3; ++c)
    {
        std::vector<float> leafMin(1024u), leafMax(1024u);
        for (uint32_t t = 0; t < 1024u; ++t)
        {
            float m = VB_MAX_F32, M = -VB_MAX_F32;
            for (uint32_t w = t; w < n; w += 1024u)
            {
                const float x = pos4[(size_t(offset) + w) * 4u + size_t(c)];
                const float sMin = (m < x) ? m : x;      // PxMin = selectMin
                const float sMax = (M > x) ? M : x;      // PxMax = selectMax
                if (!std::isnan(m) && !std::isnan(x) && vbBits(sMin) != vbBits(lpFMin(m, x))) ++gVbCnt.laterZeroWins;
                if (!std::isnan(M) && !std::isnan(x) && vbBits(sMax) != vbBits(lpFMax(M, x))) ++gVbCnt.laterZeroWins;
                if (w != t && std::isnan(x) && !std::isnan(m)) ++gVbCnt.nanDropped;
                if (w != t && std::isnan(x) && !std::isnan(M)) ++gVbCnt.nanDropped;
                m = (gVbVariant == 3) ? lpFMin(m, x) : sMin;
                M = (gVbVariant == 3) ? lpFMax(M, x) : sMax;
            }
            leafMin[t] = m;
            leafMax[t] = M;
        }
        const float pMin = vbPortTree(leafMin, false);
        const float pMax = vbPortTree(leafMax, true);
        if (gVbVariant == 0)
        {
            mn[c] = vbCudaTree(leafMin, false);
            mx[c] = vbCudaTree(leafMax, true);
            if (vbBits(mn[c]) != vbBits(pMin)) ++gVbCnt.orderMismatch;
            if (vbBits(mx[c]) != vbBits(pMax)) ++gVbCnt.orderMismatch;
        }
        else
        {
            mn[c] = pMin;
            mx[c] = pMax;
        }
    }
}

} // namespace

// ===========================================================================
// ps_update_volume_bound
// ===========================================================================
PXG_TEST(psVolumeBound)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<PxgBufferHandle> all;
    auto mk = [&](auto const& vec) { PxgBufferHandle h = createBufferFrom(dev, vec); all.push_back(h); return h; };

    // TWO systems, the active list naming them in REVERSE order: blockIdx.z is an
    // indirection, not an index.
    const uint32_t NB_SYS = 2u;
    const uint32_t ACTIVE[2] = { 1u, 0u };

    // System s has (s + 2) buffers; buffer b of system s has (b + 1) volumes, and
    // the volumes carry NON-ZERO particleIndicesOffsets so the offset arithmetic
    // `volume.particleIndicesOffset + bufferOffset` is a real sum of two non-zero
    // numbers.  Total particles per system is large enough that the 1024-thread
    // block grid-strides more than once for the biggest volume.
    const uint32_t NB_PARTICLES = 3000u;

    // TWO SIGN-PURE REGIONS, and why they have to exist.  The reduction is seeded
    // with +/-PX_MAX_F32.  A mutant that seeds it with 0.0f instead is invisible
    // unless some volume's true minimum is POSITIVE (then a 0 seed wins and the
    // bound is wrong) or its true maximum is NEGATIVE.  With one pseudo-random
    // position field every slice straddles zero in every component, so both seeds
    // are unobservable -- which is exactly how `vol-seed-zero` survived the first
    // sweep.  Particles [2200, 2500) are strictly POSITIVE in all three components
    // and [2600, 2900) strictly NEGATIVE, and one volume per system is pointed at
    // one of them.
    const uint32_t POS_REGION = 2200u, NEG_REGION = 2600u, REGION_COUNT = 300u;

    std::vector<float> positions(size_t(NB_PARTICLES) * 4u);
    for (uint32_t i = 0; i < NB_PARTICLES; ++i)
    {
        // deliberately non-monotone so the AABB is not just the first/last element
        positions[i * 4 + 0] = std::sin(float(i) * 0.37f) * (1.0f + 0.01f * float(i % 13));
        positions[i * 4 + 1] = std::cos(float(i) * 0.11f) * 3.0f - 0.002f * float(i);
        positions[i * 4 + 2] = float((i * 7919u) % 251u) * 0.05f - 6.0f;
        positions[i * 4 + 3] = 1.0f;   // invMass, must not reach the bound
    }
    for (uint32_t i = 0; i < REGION_COUNT; ++i)
    {
        const float s0 = 1.0f + std::fabs(std::sin(float(i) * 0.41f)) * 2.0f;
        const float s1 = 1.0f + std::fabs(std::cos(float(i) * 0.23f)) * 2.0f;
        const float s2 = 1.0f + float(i % 37u) * 0.05f;
        positions[(POS_REGION + i) * 4 + 0] = s0;
        positions[(POS_REGION + i) * 4 + 1] = s1;
        positions[(POS_REGION + i) * 4 + 2] = s2;
        positions[(NEG_REGION + i) * 4 + 0] = -s1;
        positions[(NEG_REGION + i) * 4 + 1] = -s2;
        positions[(NEG_REGION + i) * 4 + 2] = -s0;
    }
    PxgBufferHandle bPos = mk(positions);
    if (!addrOf(dev, bPos)) { std::printf("  NOTE: no BDA -> skip\n"); for (PxgBufferHandle h : all) dev->destroyBuffer(h); return; }

    // A TRAP is a volume no block may legally touch: a live numParticles and a
    // poisoned bound, sitting exactly where an off-by-one guard would write.  Two
    // kinds are planted, one per guard:
    //
    //   * ONE PAST THE VOLUME COUNT.  Every volume array is allocated with nbVols+1
    //     entries while mNumVolumes says nbVols.  A `gid.x > mNumVolumes` guard runs
    //     the block with gid.x == nbVols, which reads that entry and writes a real
    //     bound into it.  Without the trap that block reads allocation padding --
    //     numParticles 0 -- loops zero times and writes nothing anyone checks,
    //     which is how `vol-guard-ge-to-gt` survived the first sweep looking
    //     correct.  It was a Class-D accident, not a correct guard.
    //   * ONE PAST THE BUFFER COUNT.  Every sim-buffer array is allocated with
    //     nbBuffers+1 entries, the extra one naming its own single-volume trap.
    //     grid.y is the MAX buffer count across systems, so system 0 really does
    //     get a block with bufferIndex == its buffer count; `vol-buffer-guard-off`
    //     survived for the same accidental reason.
    const uint32_t TRAP_COUNT = 500u;      // a live particle count
    const float POISON = -999.0f;
    auto makeTrapVolume = [&](uint32_t offset)
    {
        std::vector<uint8_t> v(SZ_VOLUME, 0u);
        for (uint32_t c = 0; c < 6u; ++c) std::memcpy(v.data() + c * 4u, &POISON, 4);
        putU32(v, 24u, offset);
        putU32(v, 28u, TRAP_COUNT);
        return v;
    };

    struct VolSpec { uint32_t sys, buf, vol, offset, count, rawOffset; };
    std::vector<VolSpec> specs;
    std::vector<uint8_t> systems(size_t(NB_SYS) * SZ_PSYS, 0u);
    std::vector<PxgBufferHandle> volHandles;
    std::vector<PxgBufferHandle> trapHandles;   // one per volume array + one per system
    uint32_t maxVolumes = 0u, maxBuffers = 0u;

    for (uint32_t s = 0; s < NB_SYS; ++s)
    {
        const uint32_t nbBuffers = s + 2u;
        maxBuffers = nbBuffers > maxBuffers ? nbBuffers : maxBuffers;

        // an EXCLUSIVE prefix sum, as the engine builds it
        std::vector<uint32_t> runsum(nbBuffers);
        uint32_t acc = 0u;
        for (uint32_t b = 0; b < nbBuffers; ++b) { runsum[b] = acc; acc += 300u + b * 137u; }

        // nbBuffers + 1: the last entry is the trap (see above)
        std::vector<uint8_t> simBufs(size_t(nbBuffers + 1u) * SZ_SIMBUF, 0u);
        for (uint32_t b = 0; b < nbBuffers; ++b)
        {
            const uint32_t nbVols = b + 1u;
            maxVolumes = nbVols > maxVolumes ? nbVols : maxVolumes;
            // nbVols + 1: the last entry is the per-array trap
            std::vector<uint8_t> vols(size_t(nbVols + 1u) * SZ_VOLUME, 0u);
            for (uint32_t v = 0; v < nbVols; ++v)
            {
                // buffer 0's volume 0 is aimed at a SIGN-PURE region so that the
                // +/-PX_MAX_F32 seeds are observable; the rest carry a small
                // non-zero offset so `volume.particleIndicesOffset + bufferOffset`
                // is a real sum of two non-zero numbers.
                const bool signPure = (b == 0u && v == 0u);
                const uint32_t off = signPure ? ((s == 0u) ? POS_REGION : NEG_REGION)
                                              : (11u + v * 29u);
                const uint32_t cnt = signPure ? REGION_COUNT
                                             : (40u + v * 570u + b * 90u); // spans >1 pass
                // poison the bound so "never written" is an impossible number
                for (uint32_t c = 0; c < 6u; ++c)
                    std::memcpy(vols.data() + size_t(v) * SZ_VOLUME + c * 4u, &POISON, 4);
                putU32(vols, size_t(v) * SZ_VOLUME + 24u, off);
                putU32(vols, size_t(v) * SZ_VOLUME + 28u, cnt);
                specs.push_back(VolSpec{ s, b, v, off + runsum[b], cnt, off });
            }
            {
                const std::vector<uint8_t> trap = makeTrapVolume(0u);
                std::memcpy(vols.data() + size_t(nbVols) * SZ_VOLUME, trap.data(), SZ_VOLUME);
            }
            PxgBufferHandle bVols = mk(vols);
            volHandles.push_back(bVols);
            putU64(simBufs, size_t(b) * SZ_SIMBUF + SB_VOLUMES, addrOf(dev, bVols));
            putU32(simBufs, size_t(b) * SZ_SIMBUF + SB_NUM_VOLUMES, nbVols);
        }
        {
            // the trap SIM BUFFER: one live volume, reachable only by a block whose
            // bufferIndex the guard should have culled
            PxgBufferHandle bTrapVol = mk(makeTrapVolume(0u));
            trapHandles.push_back(bTrapVol);
            putU64(simBufs, size_t(nbBuffers) * SZ_SIMBUF + SB_VOLUMES, addrOf(dev, bTrapVol));
            putU32(simBufs, size_t(nbBuffers) * SZ_SIMBUF + SB_NUM_VOLUMES, 1u);
        }
        PxgBufferHandle bSimBufs = mk(simBufs);
        PxgBufferHandle bRunsum = mk(runsum);

        const size_t o = size_t(s) * SZ_PSYS;
        putU64(systems, o + PS_UNSORTED_POS, addrOf(dev, bPos));
        putU32(systems, o + PS_NUM_BUFFERS, nbBuffers);
        putU64(systems, o + PS_SIM_BUFFERS, addrOf(dev, bSimBufs));
        putU64(systems, o + PS_BUFFER_RUNSUM, addrOf(dev, bRunsum));
    }
    PxgBufferHandle bSystems = mk(systems);
    PxgBufferHandle bActive = mk(std::vector<uint32_t>(ACTIVE, ACTIVE + NB_SYS));

    PsVolumeBoundPush p{};
    p.particleSystems = addrOf(dev, bSystems);
    p.activeParticleSystemIndices = addrOf(dev, bActive);

    {
        // the ENGINE grid: (mMaxVolumesPerBuffer, mMaxBuffersPerSystem, nbActive).
        // Both x and y overshoot for most blocks, which is what makes the two
        // early-outs load-bearing.
        Kernel k("psVolumeBoundBda.spv", "ps_update_volume_bound", {}, sizeof(PsVolumeBoundPush));
        k.dispatch(maxVolumes, {}, p, maxBuffers, NB_SYS);
    }
    dev->flush();

    // read every volume back and compare against a host AABB over the same slice
    gVbCnt = VbCounters{};
    size_t vh = 0;
    int checked = 0;
    for (uint32_t s = 0; s < NB_SYS; ++s)
    {
        const uint32_t nbBuffers = s + 2u;
        for (uint32_t b = 0; b < nbBuffers; ++b, ++vh)
        {
            const uint32_t nbVols = b + 1u;
            const std::vector<uint8_t> got =
                downloadBuffer<uint8_t>(dev, volHandles[vh], size_t(nbVols + 1u) * SZ_VOLUME);

            // the trap one past mNumVolumes must still hold its poison
            for (uint32_t c = 0; c < 6u; ++c)
            {
                float t = 0.0f;
                std::memcpy(&t, got.data() + size_t(nbVols) * SZ_VOLUME + c * 4u, 4);
                PXG_CHECK_MSG(std::memcmp(&t, &POISON, 4) == 0,
                              "sys %u buf %u: the volume ONE PAST mNumVolumes was written "
                              "(component %u is %.9g, not the -999 poison) -- the gid.x guard "
                              "is off by one",
                              s, b, c, double(t));
            }

            for (uint32_t v = 0; v < nbVols; ++v)
            {
                // find this volume's spec
                const VolSpec* sp = nullptr;
                for (const VolSpec& c : specs)
                    if (c.sys == s && c.buf == b && c.vol == v) { sp = &c; break; }
                PXG_CHECK_MSG(sp != nullptr, "missing spec for sys %u buf %u vol %u", s, b, v);
                if (!sp) continue;

                // P9.86c: the CUDA's rules (select per thread, then fminf/fmaxf), not
                // a serial `if (x < ref) ref = x`, which keeps the FIRST of equal
                // values and never takes a NaN.  This data has neither -0 nor NaN, so
                // the expected bits are the same; psVolumeBoundZeroSign has the inputs
                // where the rules matter.
                float refMin[3], refMax[3];
                gVbVariant = 0;
                vbOracle(positions, sp->offset, sp->count, refMin, refMax);

                float gotMin[3], gotMax[3];
                std::memcpy(gotMin, got.data() + size_t(v) * SZ_VOLUME, 12);
                std::memcpy(gotMax, got.data() + size_t(v) * SZ_VOLUME + 12u, 12);
                for (int c = 0; c < 3; ++c)
                {
                    PXG_CHECK_MSG(std::memcmp(&gotMin[c], &refMin[c], 4) == 0,
                                  "sys %u buf %u vol %u min[%d]: got %.9g, expected %.9g",
                                  s, b, v, c, double(gotMin[c]), double(refMin[c]));
                    PXG_CHECK_MSG(std::memcmp(&gotMax[c], &refMax[c], 4) == 0,
                                  "sys %u buf %u vol %u max[%d]: got %.9g, expected %.9g",
                                  s, b, v, c, double(gotMax[c]), double(refMax[c]));
                }
                // the offset/count fields must be untouched
                uint32_t offBack = 0, cntBack = 0;
                std::memcpy(&offBack, got.data() + size_t(v) * SZ_VOLUME + 24u, 4);
                std::memcpy(&cntBack, got.data() + size_t(v) * SZ_VOLUME + 28u, 4);
                PXG_CHECK_MSG(cntBack == sp->count, "volume numParticles was overwritten");
                PXG_CHECK_MSG(offBack == sp->rawOffset, "volume particleIndicesOffset was overwritten");
                ++checked;
            }
        }
    }
    // the volume behind the trap SIM BUFFER -- reachable only by a block whose
    // bufferIndex the guard should have culled -- must still hold its poison
    for (uint32_t s = 0; s < NB_SYS; ++s)
    {
        const std::vector<uint8_t> got = downloadBuffer<uint8_t>(dev, trapHandles[s], SZ_VOLUME);
        for (uint32_t c = 0; c < 6u; ++c)
        {
            float t = 0.0f;
            std::memcpy(&t, got.data() + c * 4u, 4);
            PXG_CHECK_MSG(std::memcmp(&t, &POISON, 4) == 0,
                          "sys %u: the sim buffer ONE PAST mCommonData_NumParticleBuffers was "
                          "reached (component %u is %.9g, not the -999 poison)",
                          s, c, double(t));
        }
    }

    for (PxgBufferHandle h : all) dev->destroyBuffer(h);
    // system 0 has 2 buffers carrying 1+2 volumes, system 1 has 3 carrying 1+2+3.
    PXG_CHECK_MSG(checked == 9, "expected 9 volumes across the two systems, checked %d", checked);
    PXG_CHECK_MSG(gVbCnt.orderMismatch == 0u,
                  "vbOracle: the CUDA butterfly and the port's tree order disagree on %u reductions",
                  gVbCnt.orderMismatch);
}

// ===========================================================================
// ps_update_volume_bound: the sign of zero and NaN (P9.86c)
//
// psVolumeBound's positions come from sin/cos/linear formulas: their only exact
// zeros are +0 and they contain no NaN, so neither of the kernel's two min/max
// rules is ever decisive there, and all four P9.86 audit-fix reverts in
// psVolumeBoundBda.slang survived it.  Five volumes of one buffer, each built for
// one rule:
//
//   V0 (1536 particles, invMass 0 -- never integrated, so the zero signs the user
//      gave stay): x > 0 and y < 0 everywhere except resting zeros.  Thread 5
//      reads x = -0 (slice 5) then +0 (slice 1029); thread 7 reads y = +0 (slice
//      7) then -0 (slice 1031).  PxMin keeps the LATER zero: min.x = +0, and
//      PxMax: max.y = -0.  FMin/FMax per thread (-r4) give -0 / +0.  Finite data.
//   V1 (1024 particles, one per thread): every x is the quiet NaN 0x7fc00000, so
//      every thread ends on it and the tree only ever combines two NaNs:
//      fminf/fmaxf give the canonical 0x7fffffff (the port's constant), tree
//      FMin/FMax (-r2 / -r3 / -r1-uses) hand back the input NaN, and per-thread
//      FMin/FMax (-r4) ignore it and keep the +/-PX_MAX_F32 seeds.
//   V2, V3 (1024 particles): +/-0 ties INSIDE the tree, in both operand orders of
//      the port's final step (slices 0 and 1).  fminf/fmaxf are sign-aware in
//      either order, so llvmpipe's FMin/FMax cannot be told from them here; these
//      pin the tie branch of vbFminf/vbFmaxf (twins 4/5).
//   V4 (2048 particles): thread 700 reads x = -50 (the unique minimum) then NaN,
//      thread 11 reads y = +50 (the unique maximum) then NaN.  The select takes the
//      NaN, and fminf/fmaxf drop it: -50 and +50 never reach the bound.  FMin/FMax
//      per thread (-r4) keep them.
// ===========================================================================
PXG_TEST(psVolumeBoundZeroSign)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<PxgBufferHandle> all;
    auto mk = [&](auto const& vec) { PxgBufferHandle h = createBufferFrom(dev, vec); all.push_back(h); return h; };

    const uint32_t NV = 5u;
    const uint32_t cnts[NV] = { 1536u, 1024u, 1024u, 1024u, 2048u };
    uint32_t offs[NV];
    uint32_t total = 0u;
    for (uint32_t v = 0; v < NV; ++v) { offs[v] = total; total += cnts[v]; }
    const uint32_t QNAN = 0x7fc00000u;
    const float qnan = vbFromBits(QNAN);

    std::vector<float> pos(size_t(total) * 4u);
    auto P = [&](uint32_t v, uint32_t i, int c) -> float& { return pos[(size_t(offs[v]) + i) * 4u + size_t(c)]; };
    for (uint32_t v = 0; v < NV; ++v)
        for (uint32_t i = 0; i < cnts[v]; ++i)
        {
            // x > 0, y < 0, z of mixed sign and never zero (0.5*k - 4.25)
            P(v, i, 0) = 0.25f + 0.001f * float(i);
            P(v, i, 1) = -0.25f - 0.001f * float(i);
            P(v, i, 2) = 0.5f * float(i % 17u) - 4.25f;
            P(v, i, 3) = (v == 0u) ? 0.0f : 1.0f;
        }
    // V0: resting zeros; the -0 is a negated zero coordinate
    P(0, 5u, 0) = -0.0f;  P(0, 5u + 1024u, 0) = 0.0f;
    P(0, 7u, 1) = 0.0f;   P(0, 7u + 1024u, 1) = -0.0f;
    // V1: every x the one quiet NaN
    for (uint32_t i = 0; i < cnts[1]; ++i) P(1, i, 0) = qnan;
    // V2: min.x tie (-0, +0), min.z tie (+0, -0), max.y tie (+0, -0) at the port's
    // final step v[0] = op(v[0], v[1]); z made positive so its zeros are the minimum
    for (uint32_t i = 0; i < cnts[2]; ++i) P(2, i, 2) = 0.5f + 0.002f * float(i);
    P(2, 0u, 0) = -0.0f;  P(2, 1u, 0) = 0.0f;
    P(2, 0u, 2) = 0.0f;   P(2, 1u, 2) = -0.0f;
    P(2, 0u, 1) = 0.0f;   P(2, 1u, 1) = -0.0f;
    // V3: max.y tie (-0, +0)
    P(3, 0u, 1) = -0.0f;  P(3, 1u, 1) = 0.0f;
    // V4: a NaN in the thread's LAST particle drops its extreme
    P(4, 700u, 0) = -50.0f;  P(4, 700u + 1024u, 0) = qnan;
    P(4, 11u, 1) = 50.0f;    P(4, 11u + 1024u, 1) = qnan;

    PxgBufferHandle bPos = mk(pos);
    if (!addrOf(dev, bPos)) { std::printf("  NOTE: no BDA -> skip\n"); for (PxgBufferHandle h : all) dev->destroyBuffer(h); return; }

    const float POISON = -999.0f;
    std::vector<uint8_t> vols(size_t(NV) * SZ_VOLUME, 0u);
    for (uint32_t v = 0; v < NV; ++v)
    {
        for (uint32_t c = 0; c < 6u; ++c) std::memcpy(vols.data() + size_t(v) * SZ_VOLUME + c * 4u, &POISON, 4);
        putU32(vols, size_t(v) * SZ_VOLUME + 24u, offs[v]);
        putU32(vols, size_t(v) * SZ_VOLUME + 28u, cnts[v]);
    }
    PxgBufferHandle bVols = mk(vols);
    std::vector<uint8_t> simBufs(SZ_SIMBUF, 0u);
    putU64(simBufs, SB_VOLUMES, addrOf(dev, bVols));
    putU32(simBufs, SB_NUM_VOLUMES, NV);
    PxgBufferHandle bSimBufs = mk(simBufs);
    PxgBufferHandle bRunsum = mk(std::vector<uint32_t>{ 0u });
    std::vector<uint8_t> systems(SZ_PSYS, 0u);
    putU64(systems, PS_UNSORTED_POS, addrOf(dev, bPos));
    putU32(systems, PS_NUM_BUFFERS, 1u);
    putU64(systems, PS_SIM_BUFFERS, addrOf(dev, bSimBufs));
    putU64(systems, PS_BUFFER_RUNSUM, addrOf(dev, bRunsum));
    PxgBufferHandle bSystems = mk(systems);
    PxgBufferHandle bActive = mk(std::vector<uint32_t>{ 0u });

    PsVolumeBoundPush p{};
    p.particleSystems = addrOf(dev, bSystems);
    p.activeParticleSystemIndices = addrOf(dev, bActive);
    {
        Kernel k("psVolumeBoundBda.spv", "ps_update_volume_bound", {}, sizeof(PsVolumeBoundPush));
        k.dispatch(NV, {}, p, 1u, 1u);
    }
    dev->flush();

    // img[variant][volume][0..2 = min xyz, 3..5 = max xyz]
    const int NVAR = 6;
    float img[NVAR][NV][6];
    VbCounters cnt = {};
    for (int variant = 0; variant < NVAR; ++variant)
    {
        gVbVariant = variant;
        gVbCnt = VbCounters{};
        for (uint32_t v = 0; v < NV; ++v) vbOracle(pos, offs[v], cnts[v], &img[variant][v][0], &img[variant][v][3]);
        if (variant == 0) cnt = gVbCnt;
    }
    gVbVariant = 0;

    // coverage: every directed input still reaches its rule
    PXG_CHECK_MSG(cnt.laterZeroWins >= 2u,
                  "the +/-0 per-thread tie where PxMin/PxMax keep the LATER zero was met %u times (need 2)", cnt.laterZeroWins);
    PXG_CHECK_MSG(cnt.nanPairs >= 2u, "fminf/fmaxf never combined two NaNs (%u)", cnt.nanPairs);
    PXG_CHECK_MSG(cnt.treeZeroTies >= 4u, "the tree met %u mixed-sign +/-0 pairs (need 4)", cnt.treeZeroTies);
    PXG_CHECK_MSG(cnt.nanDropped >= 2u, "a NaN pos replaced a real per-thread value %u times (need 2)", cnt.nanDropped);
    PXG_CHECK_MSG(cnt.orderMismatch == 0u, "the CUDA butterfly and the port tree order disagree (%u)", cnt.orderMismatch);

    // the CUDA's bits for the directed outputs, pinned
    auto B = [&](int var, uint32_t v, int f) { return vbBits(img[var][v][f]); };
    PXG_CHECK_MSG(B(0, 0, 0) == 0x00000000u, "V0 min.x must be thread 5's LATER +0, oracle %08x", B(0, 0, 0));
    PXG_CHECK_MSG(B(0, 0, 4) == 0x80000000u, "V0 max.y must be thread 7's LATER -0, oracle %08x", B(0, 0, 4));
    PXG_CHECK_MSG(B(0, 1, 0) == 0x7fffffffu && B(0, 1, 3) == 0x7fffffffu,
                  "V1 min.x/max.x must be the canonical NaN, oracle %08x/%08x", B(0, 1, 0), B(0, 1, 3));
    PXG_CHECK_MSG(B(0, 2, 0) == 0x80000000u && B(0, 2, 2) == 0x80000000u && B(0, 2, 4) == 0x00000000u,
                  "V2 min.x/min.z/max.y must be -0/-0/+0, oracle %08x/%08x/%08x", B(0, 2, 0), B(0, 2, 2), B(0, 2, 4));
    PXG_CHECK_MSG(B(0, 3, 4) == 0x00000000u, "V3 max.y must be +0, oracle %08x", B(0, 3, 4));
    PXG_CHECK_MSG(img[0][4][0] == P(4, 0u, 0) && img[0][4][4] == P(4, 0u, 1),
                  "V4: the NaN must drop -50/+50, oracle min.x %.9g max.y %.9g",
                  double(img[0][4][0]), double(img[0][4][4]));

    // each twin changes the image it is meant to change
    PXG_CHECK_MSG(B(1, 1, 0) != B(0, 1, 0), "twin 1 (tree FMin, -r2) leaves V1 min.x unchanged");
    PXG_CHECK_MSG(B(2, 1, 3) != B(0, 1, 3), "twin 2 (tree FMax, -r3) leaves V1 max.x unchanged");
    PXG_CHECK_MSG(B(3, 0, 0) != B(0, 0, 0) && B(3, 0, 4) != B(0, 0, 4),
                  "twin 3 (per-thread FMin/FMax, -r4) leaves V0 min.x/max.y unchanged");
    PXG_CHECK_MSG(B(3, 4, 0) != B(0, 4, 0) && B(3, 4, 4) != B(0, 4, 4),
                  "twin 3 (per-thread FMin/FMax, -r4) leaves V4 min.x/max.y unchanged");
    PXG_CHECK_MSG(B(4, 2, 2) != B(0, 2, 2) && B(4, 3, 4) != B(0, 3, 4),
                  "twin 4 (tree tie keeps the first operand) leaves V2 min.z / V3 max.y unchanged");
    PXG_CHECK_MSG(B(5, 2, 0) != B(0, 2, 0) && B(5, 2, 4) != B(0, 2, 4),
                  "twin 5 (tree tie keeps the second operand) leaves V2 min.x / max.y unchanged");

    // the device against the CUDA, bitwise, every bound float of every volume
    const std::vector<uint8_t> got = downloadBuffer<uint8_t>(dev, bVols, size_t(NV) * SZ_VOLUME);
    uint32_t bad = 0u;
    for (uint32_t v = 0; v < NV; ++v)
    {
        for (uint32_t f = 0; f < 6u; ++f)
        {
            float g; std::memcpy(&g, got.data() + size_t(v) * SZ_VOLUME + f * 4u, 4);
            if (vbBits(g) != B(0, v, int(f)))
            {
                ++bad;
                std::printf("    vol %u bound float %u: device %08x, the CUDA %08x\n", v, f, vbBits(g), B(0, v, int(f)));
            }
        }
        uint32_t offBack = 0, cntBack = 0;
        std::memcpy(&offBack, got.data() + size_t(v) * SZ_VOLUME + 24u, 4);
        std::memcpy(&cntBack, got.data() + size_t(v) * SZ_VOLUME + 28u, 4);
        PXG_CHECK_MSG(offBack == offs[v] && cntBack == cnts[v], "vol %u offset/count overwritten", v);
    }
    PXG_CHECK_MSG(bad == 0u, "%u bound floats differ bitwise from the CUDA", bad);
    std::printf("    [ note ] laterZeroWins=%u nanPairs=%u treeZeroTies=%u nanDropped=%u\n",
                cnt.laterZeroWins, cnt.nanPairs, cnt.treeZeroTies, cnt.nanDropped);
    for (PxgBufferHandle h : all) dev->destroyBuffer(h);
}

// ===========================================================================
// ps_solveShapes
// ===========================================================================
PXG_TEST(psShapeMatch)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<PxgBufferHandle> all;
    auto mk = [&](auto const& vec) { PxgBufferHandle h = createBufferFrom(dev, vec); all.push_back(h); return h; };

    // ONE system is enough here (the volume fixture already proves the blockIdx.z
    // indirection); what this one needs is the RIGID indirections.
    //
    //   - THREE rigid buffers, the second and third pointing at particle buffer 1,
    //     whose runsum entry is NON-ZERO;
    //   - TWO rigids in the first two, so the second rigid has a NON-ZERO startIndex;
    //   - n = 100 for one of them: not a multiple of 64, so `passes` is 2 and the
    //     tail lanes contribute 0.0f to the reduction on the second pass;
    //   - ONE rigid in the third, which is what makes the upstream `>` MEASURABLE.
    //
    // THE THIRD BUFFER, AND WHY IT IS PADDED.  The engine sizes the grid's x extent
    // from getMaxRigidsPerBuffer() -- the max ACROSS buffers -- so a buffer with
    // fewer rigids gets blocks it has no rigid for.  particlesystem.cu:5150 culls
    // them with `if (rigidId > numRigids) return;`, GREATER THAN, so the block with
    // rigidId == numRigids survives the guard and goes on to read rigidOffsets
    // [numRigids] and rigidOffsets[numRigids + 1] -- the second of which is one past
    // the end of an array the API documents as numRigids + 1 entries long.  With two
    // buffers of two rigids each that block is never launched and the guard is never
    // exercised; with this third buffer it IS, and a port that "corrects" the `>` to
    // `>=` stops writing everything that block writes.
    //
    // So the arrays of rigid buffer 2 are deliberately OVER-ALLOCATED -- three
    // offsets, two coefficients/translations/rotations, and local positions and
    // normals out to the padded end index.  That padding is NOT a claim that the
    // engine over-allocates: it is what makes the upstream over-read DEFINED here,
    // so the fixture measures the guard's SEMANTICS rather than whatever the
    // allocator happened to leave past the end.  (That upstream over-read is
    // recorded in RUNTIME_STATUS.md as a bug candidate; not reported.)
    const uint32_t NB_PARTICLES = 700u;
    const uint32_t RUNSUM[2] = { 0u, 250u };     // buffer 1 starts at 250
    const uint32_t NB_RIGID_BUFFERS = 3u;
    const uint32_t MAX_RIGIDS = 2u;              // getMaxRigidsPerBuffer() -> grid.x
    const uint32_t NUM_RIGIDS[3] = { 2u, 2u, 1u };
    // NOTE rb 2's entry [2] is PAST numRigids + 1: it is the value the surviving
    // block reads as its endIndex.
    const uint32_t OFFSETS[3][3] = { { 0u, 100u, 140u }, { 0u, 37u, 101u }, { 0u, 37u, 60u } };
    const uint32_t BUFIDX[3] = { 0u, 1u, 1u };
    // elements actually allocated per rigid buffer: the padded end index, so the
    // surviving block's local reads are in bounds of the ALLOCATION.
    const uint32_t LOCAL_COUNT[3] = { 140u, 101u, 60u };
    // coefficients / translations / rotations: MAX_RIGIDS entries everywhere, for
    // the same reason.
    const uint32_t RIGID_SLOTS = MAX_RIGIDS;

    std::vector<float> sortedPos(size_t(NB_PARTICLES) * 4u);
    for (uint32_t i = 0; i < NB_PARTICLES; ++i)
    {
        sortedPos[i * 4 + 0] = std::sin(float(i) * 0.21f) * 2.0f + 0.003f * float(i);
        sortedPos[i * 4 + 1] = std::cos(float(i) * 0.13f) * 1.5f;
        sortedPos[i * 4 + 2] = float((i * 104729u) % 197u) * 0.02f - 2.0f;
        sortedPos[i * 4 + 3] = 0.5f;
    }
    // a real PERMUTATION: a port that skipped the reverse lookup reads different
    // particles entirely.
    std::vector<uint32_t> u2s(NB_PARTICLES);
    for (uint32_t i = 0; i < NB_PARTICLES; ++i) u2s[i] = (i * 389u + 17u) % NB_PARTICLES;

    PxgBufferHandle bSorted = mk(sortedPos);
    if (!addrOf(dev, bSorted)) { std::printf("  NOTE: no BDA -> skip\n"); for (PxgBufferHandle h : all) dev->destroyBuffer(h); return; }
    PxgBufferHandle bU2S = mk(u2s);
    PxgBufferHandle bRunsum = mk(std::vector<uint32_t>(RUNSUM, RUNSUM + 2));

    // deltas and normals, zeroed; both are accumulated/written by the kernel
    std::vector<float> deltasInit(size_t(NB_PARTICLES) * 4u, 0.0f);
    std::vector<float> normalsInit(size_t(NB_PARTICLES) * 4u, 0.0f);
    PxgBufferHandle bDeltas = mk(deltasInit);
    PxgBufferHandle bNormals = mk(normalsInit);

    // per rigid buffer: coefficients, translations, rotations, offsets, localPos, localNrm
    std::vector<uint8_t> rigidBufs(size_t(NB_RIGID_BUFFERS) * SZ_RIGIDBUF, 0u);
    std::vector<std::vector<float>> localPosAll, localNrmAll, coeffAll, transAll, rotAll;
    std::vector<PxgBufferHandle> hTrans, hRot;

    for (uint32_t rb = 0; rb < NB_RIGID_BUFFERS; ++rb)
    {
        const uint32_t total = LOCAL_COUNT[rb];
        std::vector<float> localPos(size_t(total) * 4u), localNrm(size_t(total) * 4u);
        for (uint32_t i = 0; i < total; ++i)
        {
            localPos[i * 4 + 0] = std::sin(float(i) * 0.7f + float(rb)) * 0.9f;
            localPos[i * 4 + 1] = std::cos(float(i) * 0.5f) * 0.8f;
            localPos[i * 4 + 2] = float(int(i % 17) - 8) * 0.1f;
            localPos[i * 4 + 3] = 0.0f;
            localNrm[i * 4 + 0] = 0.0f; localNrm[i * 4 + 1] = 1.0f; localNrm[i * 4 + 2] = 0.0f;
            // a DISTINCT .w per element: the kernel must carry it through untouched
            localNrm[i * 4 + 3] = 100.0f + float(i);
        }
        std::vector<float> coeff(RIGID_SLOTS), trans(size_t(RIGID_SLOTS) * 4u), rot(size_t(RIGID_SLOTS) * 4u);
        for (uint32_t r = 0; r < RIGID_SLOTS; ++r)
        {
            coeff[r] = 0.25f + 0.5f * float(r);
            // a NON-ZERO old COM, so the drift clamp has something to compare to
            trans[r * 4 + 0] = 0.1f * float(r + 1); trans[r * 4 + 1] = -0.2f; trans[r * 4 + 2] = 0.3f; trans[r * 4 + 3] = 0.0f;
            // a NON-IDENTITY seed rotation: extractRotation refines from it
            rot[r * 4 + 0] = 0.1f; rot[r * 4 + 1] = 0.2f; rot[r * 4 + 2] = 0.3f;
            rot[r * 4 + 3] = std::sqrt(1.0f - 0.01f - 0.04f - 0.09f);
        }
        PxgBufferHandle bLP = mk(localPos), bLN = mk(localNrm), bC = mk(coeff), bT = mk(trans), bR = mk(rot);
        PxgBufferHandle bO = mk(std::vector<uint32_t>(OFFSETS[rb], OFFSETS[rb] + 3u));
        localPosAll.push_back(localPos); localNrmAll.push_back(localNrm);
        coeffAll.push_back(coeff); transAll.push_back(trans); rotAll.push_back(rot);
        hTrans.push_back(bT); hRot.push_back(bR);

        const size_t o = size_t(rb) * SZ_RIGIDBUF;
        putU64(rigidBufs, o + RB_COEFFS, addrOf(dev, bC));
        putU64(rigidBufs, o + RB_TRANSLATIONS, addrOf(dev, bT));
        putU64(rigidBufs, o + RB_ROTATIONS, addrOf(dev, bR));
        putU64(rigidBufs, o + RB_OFFSETS, addrOf(dev, bO));
        putU64(rigidBufs, o + RB_LOCALPOS, addrOf(dev, bLP));
        putU64(rigidBufs, o + RB_LOCALNRM, addrOf(dev, bLN));
        putU32(rigidBufs, o + RB_NUMRIGIDS, NUM_RIGIDS[rb]);
        putU32(rigidBufs, o + RB_BUFINDEX, BUFIDX[rb]);
    }
    PxgBufferHandle bRigidBufs = mk(rigidBufs);

    std::vector<uint8_t> systems(SZ_PSYS, 0u);
    putU64(systems, PS_SORTED_POS, addrOf(dev, bSorted));
    putU64(systems, PS_U2S, addrOf(dev, bU2S));
    putU64(systems, PS_BUFFER_RUNSUM, addrOf(dev, bRunsum));
    putU64(systems, PS_DELTA, addrOf(dev, bDeltas));
    putU64(systems, PS_NORMALS, addrOf(dev, bNormals));
    putU64(systems, PS_RIGID_SIM_BUFFERS, addrOf(dev, bRigidBufs));
    putU32(systems, PS_NUM_RIGID_BUFFERS, NB_RIGID_BUFFERS);
    PxgBufferHandle bSystems = mk(systems);
    PxgBufferHandle bActive = mk(std::vector<uint32_t>{ 0u });

    const float DT = 1.0f / 60.0f;      // DEAD in the kernel
    const float BIAS = 0.75f;

    PsSolveShapesPush p{};
    p.particleSystems = addrOf(dev, bSystems);
    p.activeParticleSystems = addrOf(dev, bActive);
    p.dt = DT;
    p.biasCoefficient = BIAS;

    {
        // grid x = MAX_RIGIDS, exactly as the engine sizes it from
        // getMaxRigidsPerBuffer().  Rigid buffer 2 holds ONE rigid, so its block
        // rigidId == 1 == numRigids is the one the upstream `>` lets through.
        Kernel k("psShapeMatchBda.spv", "ps_solveShapes", {}, sizeof(PsSolveShapesPush));
        k.dispatch(MAX_RIGIDS, {}, p, NB_RIGID_BUFFERS, 1u);
    }
    dev->flush();

    std::vector<std::vector<float>> gotTrans, gotRot;
    for (uint32_t rb = 0; rb < NB_RIGID_BUFFERS; ++rb)
    {
        gotTrans.push_back(downloadBuffer<float>(dev, hTrans[rb], size_t(RIGID_SLOTS) * 4u));
        gotRot.push_back(downloadBuffer<float>(dev, hRot[rb], size_t(RIGID_SLOTS) * 4u));
    }
    const std::vector<float> gotNormals = downloadBuffer<float>(dev, bNormals, normalsInit.size());
    const std::vector<float> gotDeltas = downloadBuffer<float>(dev, bDeltas, deltasInit.size());
    for (PxgBufferHandle h : all) dev->destroyBuffer(h);

    // ---- the reference, transcribed from particlesystem.cu:5121 -------------
    //
    // THE NORMAL ARRAY IS NOT A FUNCTION OF THE INPUT, AND THAT IS UPSTREAM.
    // Oddity (b) in psShapeMatchBda.slang's header -- `originalIndex = bufferIndex
    // + localOffset`, with bufferIndex the RIGID BUFFER index -- means rigid buffer
    // 0 writes slots [0, 140) and rigid buffer 1 writes slots [1, 102).  Those
    // ranges OVERLAP, the two writers are different BLOCKS, and nothing orders
    // them: the surviving value in a shared slot is whichever block ran last.  The
    // first draft of this fixture predicted the serial host order (rb = 1 last) and
    // duly failed on 11 runs out of 12, passing on the twelfth -- which is exactly
    // what a race looks like and exactly what a fixture must not paper over with a
    // wider tolerance.
    //
    // This is a property of the CUDA, not of the port: the .cu has the same two
    // blocks writing the same addresses.  (It has never bitten upstream because
    // every shipped scene has ONE rigid buffer, where `bufferIndex` is 0 and the
    // expression degenerates to `localOffset`.  Recorded as an upstream bug
    // candidate in RUNTIME_STATUS.md; not reported.)
    //
    // So the reference collects, per slot, EVERY value a block could legally have
    // left there, and the check below demands the hardware produced one of them.
    // That is still a real check -- a wrong rotation, a dropped .w or a write to
    // the wrong slot matches NO candidate -- and it is the strongest statement that
    // is true of the kernel.  A single-rigid-buffer fixture would have no race and
    // no coverage of `bufferIndex` either; this one keeps the coverage.
    struct NormalCand { float v[4]; };
    std::vector<std::vector<NormalCand>> normalCands(NB_PARTICLES);
    std::vector<float> refDeltas = deltasInit;
    for (uint32_t rb = 0; rb < NB_RIGID_BUFFERS; ++rb)
    {
        const uint32_t bufferOffset = RUNSUM[BUFIDX[rb]];
        for (uint32_t r = 0; r < MAX_RIGIDS; ++r)
        {
            // particlesystem.cu:5150, transcribed: GREATER THAN.  For rigid buffer 2
            // (numRigids == 1) this lets r == 1 through, and that block then reads
            // the padded offsets entry as its endIndex.
            if (r > NUM_RIGIDS[rb]) continue;

            const uint32_t startIndex = OFFSETS[rb][r];
            const uint32_t n = OFFSETS[rb][r + 1] - startIndex;
            const uint32_t passes = (n + PS_SOLVE_SHAPE - 1u) / PS_SOLVE_SHAPE;

            const V3 oldCom{ transAll[rb][r * 4 + 0], transAll[rb][r * 4 + 1], transAll[rb][r * 4 + 2] };
            V3 newCom{ 0.0f, 0.0f, 0.0f };

            for (uint32_t i = 0; i < passes; ++i)
            {
                float px[PS_SOLVE_SHAPE], py[PS_SOLVE_SHAPE], pz[PS_SOLVE_SHAPE];
                for (uint32_t t = 0; t < PS_SOLVE_SHAPE; ++t)
                {
                    const uint32_t offset = i * PS_SOLVE_SHAPE + t;
                    px[t] = py[t] = pz[t] = 0.0f;
                    if (offset < n)
                    {
                        const uint32_t si = u2s[bufferOffset + startIndex + offset];
                        px[t] = sortedPos[si * 4 + 0];
                        py[t] = sortedPos[si * 4 + 1];
                        pz[t] = sortedPos[si * 4 + 2];
                    }
                }
                newCom.x += blockReduceAdd64(px, 0.0f);
                newCom.y += blockReduceAdd64(py, 0.0f);
                newCom.z += blockReduceAdd64(pz, 0.0f);
            }

            const float inv = 1.0f / float(n);
            newCom = V3{ newCom.x * inv, newCom.y * inv, newCom.z * inv };
            const float eps = 0.00001f;
            if (std::fabs(newCom.x - oldCom.x) < eps) newCom.x = oldCom.x;
            if (std::fabs(newCom.y - oldCom.y) < eps) newCom.y = oldCom.y;
            if (std::fabs(newCom.z - oldCom.z) < eps) newCom.z = oldCom.z;

            V3 qr0{ 0, 0, 0 }, qr1{ 0, 0, 0 }, qr2{ 0, 0, 0 };
            for (uint32_t i = 0; i < passes; ++i)
            {
                float m[9][PS_SOLVE_SHAPE];
                for (uint32_t t = 0; t < PS_SOLVE_SHAPE; ++t)
                {
                    for (int c = 0; c < 9; ++c) m[c][t] = 0.0f;
                    const uint32_t offset = i * PS_SOLVE_SHAPE + t;
                    if (offset < n)
                    {
                        const uint32_t si = u2s[bufferOffset + startIndex + offset];
                        const V3 np{ sortedPos[si * 4 + 0], sortedPos[si * 4 + 1], sortedPos[si * 4 + 2] };
                        const V3 d{ np.x - newCom.x, np.y - newCom.y, np.z - newCom.z };
                        const float lx = localPosAll[rb][(startIndex + offset) * 4 + 0];
                        const float ly = localPosAll[rb][(startIndex + offset) * 4 + 1];
                        const float lz = localPosAll[rb][(startIndex + offset) * 4 + 2];
                        m[0][t] = d.x * lx; m[1][t] = d.y * lx; m[2][t] = d.z * lx;
                        m[3][t] = d.x * ly; m[4][t] = d.y * ly; m[5][t] = d.z * ly;
                        m[6][t] = d.x * lz; m[7][t] = d.y * lz; m[8][t] = d.z * lz;
                    }
                }
                qr0.x += blockReduceAdd64(m[0], 0.0f); qr0.y += blockReduceAdd64(m[1], 0.0f); qr0.z += blockReduceAdd64(m[2], 0.0f);
                qr1.x += blockReduceAdd64(m[3], 0.0f); qr1.y += blockReduceAdd64(m[4], 0.0f); qr1.z += blockReduceAdd64(m[5], 0.0f);
                qr2.x += blockReduceAdd64(m[6], 0.0f); qr2.y += blockReduceAdd64(m[7], 0.0f); qr2.z += blockReduceAdd64(m[8], 0.0f);
            }

            F4 q{ rotAll[rb][r * 4 + 0], rotAll[rb][r * 4 + 1], rotAll[rb][r * 4 + 2], rotAll[rb][r * 4 + 3] };
            q = extractRotation(qr0, qr1, qr2, q, 4);

            // ---- the BIT-EXACT half: the translation is the reduced COM -------
            const float* gt = gotTrans[rb].data() + size_t(r) * 4u;
            const float refT[3] = { newCom.x, newCom.y, newCom.z };
            for (int c = 0; c < 3; ++c)
                PXG_CHECK_MSG(std::memcmp(&gt[c], &refT[c], 4) == 0,
                              "rb %u rigid %u translation[%d]: got %.9g, expected %.9g "
                              "(this is the reduction tree, and it is compared BIT-EXACTLY)",
                              rb, r, c, double(gt[c]), double(refT[c]));

            // ---- the TOLERANCE half: everything downstream of sin/cos ---------
            const float* gr = gotRot[rb].data() + size_t(r) * 4u;
            const float refQ[4] = { q.x, q.y, q.z, q.w };
            for (int c = 0; c < 4; ++c)
                PXG_CHECK_MSG(std::fabs(gr[c] - refQ[c]) < 2.0e-5f,
                              "rb %u rigid %u rotation[%d]: got %.9g, expected %.9g "
                              "(tolerance: PxQuat(angle, axis) calls PxSinCos)",
                              rb, r, c, double(gr[c]), double(refQ[c]));

            const float stiffness = coeffAll[rb][r] * BIAS;
            for (uint32_t offset = 0; offset < n; ++offset)
            {
                const uint32_t localOffset = startIndex + offset;
                const uint32_t si = u2s[bufferOffset + localOffset];
                const V3 elastic{ sortedPos[si * 4 + 0], sortedPos[si * 4 + 1], sortedPos[si * 4 + 2] };
                const V3 lp{ localPosAll[rb][localOffset * 4 + 0], localPosAll[rb][localOffset * 4 + 1],
                             localPosAll[rb][localOffset * 4 + 2] };
                const V3 rot = quatRotate(q, lp);
                const V3 rigidPos{ rot.x + newCom.x, rot.y + newCom.y, rot.z + newCom.z };
                const V3 d{ elastic.x - rigidPos.x, elastic.y - rigidPos.y, elastic.z - rigidPos.z };

                // UPSTREAM: bufferIndex (the RIGID BUFFER index) + localOffset
                const uint32_t originalIndex = rb + localOffset;
                const V3 ln{ localNrmAll[rb][localOffset * 4 + 0], localNrmAll[rb][localOffset * 4 + 1],
                             localNrmAll[rb][localOffset * 4 + 2] };
                const V3 nr = quatRotate(q, ln);
                NormalCand cand;
                cand.v[0] = nr.x;
                cand.v[1] = nr.y;
                cand.v[2] = nr.z;
                cand.v[3] = localNrmAll[rb][localOffset * 4 + 3];
                normalCands[originalIndex].push_back(cand);

                refDeltas[si * 4 + 0] += -d.x * stiffness;
                refDeltas[si * 4 + 1] += -d.y * stiffness;
                refDeltas[si * 4 + 2] += -d.z * stiffness;
                refDeltas[si * 4 + 3] += 1.0f;
            }
        }
    }

    // Normals are downstream of the rotation, so tolerance -- and per the note
    // above, a slot two rigid buffers both target must match ONE of the two
    // candidates rather than a predicted winner.  A slot NO block targets must
    // still be exactly the zero it was uploaded with: that is what catches a write
    // that landed outside its range.
    uint32_t contested = 0u;
    for (uint32_t ptc = 0; ptc < NB_PARTICLES; ++ptc)
    {
        const std::vector<NormalCand>& cands = normalCands[ptc];
        if (cands.size() > 1u) ++contested;
        if (cands.empty())
        {
            for (uint32_t c = 0; c < 4u; ++c)
                PXG_CHECK_MSG(gotNormals[size_t(ptc) * 4u + c] == 0.0f,
                              "normal[%u] comp %u: got %.9g into a slot NO block writes",
                              ptc, c, double(gotNormals[size_t(ptc) * 4u + c]));
            continue;
        }
        bool matched = false;
        for (const NormalCand& cand : cands)
        {
            bool all = true;
            for (uint32_t c = 0; c < 4u && all; ++c)
                all = std::fabs(gotNormals[size_t(ptc) * 4u + c] - cand.v[c]) < 2.0e-4f;
            matched = matched || all;
        }
        PXG_CHECK_MSG(matched,
                      "normal[%u]: got (%.9g, %.9g, %.9g, %.9g), which matches none of "
                      "the %zu value(s) a block could have left there (first candidate: "
                      "%.9g, %.9g, %.9g, %.9g)",
                      ptc, double(gotNormals[size_t(ptc) * 4u + 0]), double(gotNormals[size_t(ptc) * 4u + 1]),
                      double(gotNormals[size_t(ptc) * 4u + 2]), double(gotNormals[size_t(ptc) * 4u + 3]),
                      cands.size(), double(cands[0].v[0]), double(cands[0].v[1]),
                      double(cands[0].v[2]), double(cands[0].v[3]));
    }
    // and the overlap the note describes must actually exist, or the candidate
    // machinery above is dead weight dressed up as rigour.
    PXG_CHECK_MSG(contested > 0u,
                  "no normal slot is written twice -- the two rigid buffers no longer "
                  "overlap, so this fixture has stopped covering the bufferIndex oddity");

    // The delta WEIGHT (.w) is a count of 1.0f additions and must be exact; the
    // three components are downstream of the rotation, so tolerance.
    for (size_t i = 0; i < refDeltas.size(); ++i)
    {
        if ((i % 4) == 3)
            PXG_CHECK_MSG(std::memcmp(&gotDeltas[i], &refDeltas[i], 4) == 0,
                          "delta weight[%zu]: got %.9g, expected %.9g (a count, exact)",
                          i / 4, double(gotDeltas[i]), double(refDeltas[i]));
        else
            PXG_CHECK_MSG(std::fabs(gotDeltas[i] - refDeltas[i]) < 2.0e-4f,
                          "delta[%zu] comp %zu: got %.9g, expected %.9g",
                          i / 4, i % 4, double(gotDeltas[i]), double(refDeltas[i]));
    }

    // ...and the normals must actually have been written somewhere, or every
    // comparison above is a comparison of zeros.
    bool anyNormal = false;
    for (size_t i = 0; i < gotNormals.size() && !anyNormal; ++i) anyNormal = (gotNormals[i] != 0.0f);
    PXG_CHECK_MSG(anyNormal, "the normal array is entirely zero -- nothing was written");
}
