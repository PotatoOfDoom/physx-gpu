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

// testPsSolveSprings.cpp -- P9.3b fixture for ps_solveSpringsLaunch, ported as
// the third entry point of npPsSprings.slang.
//
// ORACLE: a host transcription of the XPBD spring solve, spelled operation for
// operation (PxVec3::operator/ as a reciprocal multiply, PxMin as a ternary, the
// mul+add chains left un-contracted by -ffp-contract=off on this target).  The
// cloth record layout it walks is pinned by psSpringsLayout in testPsSprings.cpp;
// what carries the arithmetic is the bit-exact compare plus the sweep.
//
// THE SEQUENCE IS THE TEST.  The engine dispatches this kernel once per
// PARTITION (PxgPBDParticleSystemCore.cpp:999-1014), grid
// (ceil(maxSpringsPerPartition/512), maxClothBuffers, nbActive), block 512, and
// the fixture replays exactly that: one dispatch per partitionId with a flush
// between, over the same buffers, and the mirror walks the partitions in the
// same order.  One cloth buffer's mRemapOutput deliberately CHAINS -- partition
// 0's scatter lands on partition 1's read slots -- so the ordering is load
// bearing and a mutant that reorders or fuses the partitions dies.
//
// TWO ROUNDS.  Everything is built and run twice, once with isTGS false and once
// with isTGS true, because trap S5 is only visible as a difference between them:
// the PGS path reads AND writes mSpringLambda with an iMul warm start, the TGS
// path touches it not at all.  The lambda array is therefore seeded with
// DISTINCT NON-ZERO values (not the zeros ps_initializeSpringsLaunch would
// leave) so the warm-start read is observable, and on the TGS round every one of
// them must come back exactly as uploaded.
//
// GEOMETRY -- the standing four-system pool:
//   pool[0] = A   : 2 live cloth buffers (9 hand-built springs / 700 procedural)
//   pool[1] = DECOY, never active, 4 live cloth buffers -- all outputs poisoned
//   pool[2] = B   : 1 live cloth buffer, 6 springs in FOUR partitions (the most
//                   of any record, so partitionId 3 must find every other record
//                   past its own mNumPartitions), and the CHAINED mRemapOutput
//   pool[3] = C   : 2 live cloth buffers -- 5 springs in ONE partition, and a
//                   record with mNumSprings == 0 but real backing behind it
//   active = { 2, 0, 3 }   -- not a permutation, so indexing the pool by
//                             blockIdx.z directly hits the decoy and dies
// Every system's cloth ARRAY is padded to the across-system maximum (4) with
// REAL records whose outputs must stay poisoned: the P9.3a lesson that a bound
// is only tested when there is something poisoned on the wrong side of it.
//
// THE GRID IS NOT SQUARE, and the first sweep is why.  With three cloth-buffer
// slots and three active systems the Y and Z extents were equal, so swapping
// gridDim.Y for gridDim.Z covered exactly the same (system, buffer) set and the
// mutant SURVIVED as a true equivalence of the fixture, not of the kernel.  The
// decoy therefore carries FOUR cloth buffers (Y = 4) against three active
// systems (Z = 3), and the active list is allocated eight deep so the swapped
// mutant reads a live decoy index rather than running off the allocation.
//
// A1's first partition holds 600 springs, so the X grid is 2 blocks of 512 and
// every grid-decomposition mutant has somewhere to go wrong.
//
// THE POISON IS PLACED ON BOTH SIDES OF EVERY INDEX:
//   * mAccumulatedSpringsPerPartitions is handed to the kernel as an OFFSET
//     pointer with four lead-in entries holding 2 -- so a mutant that drops the
//     `partitionId > 0` guard and reads partitions[-1] gets a live, wrong start
//     index instead of faulting, and dies visibly;
//   * its four tail entries INCREASE past nbSprings, so a read one past the
//     partition index -- or a dropped `partitionId < nbPartitions` guard -- finds
//     a non-empty range to process instead of an empty one it could survive;
//   * mRemapOutput carries eight tail entries pointing into the last eight
//     remap slots, so an endpoint-stride mutant scatters somewhere legal and
//     observable rather than segfaulting;
//   * the remap arrays are 4*nbSprings + 8 slots: [0, 2N) is the gathered input,
//     [2N, 4N) is the accumulation region every scatter target lands in, and the
//     last eight must come back untouched.
// Because every scatter target is disjoint from every read slot of the same
// partition (asserted below, per record and per partition), the dispatch has no
// read/write race and the mirror may snapshot the reads.

#include "pxgTestFramework.h"

#include "host_defines.h"
#include "PxgParticleSystem.h"
#include "PxParticleBuffer.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct F4 { float x, y, z, w; };
struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };

template <typename T>
DevBuf makeDev(PxgSlangComputeDevice* dev, const std::vector<T>& data)
{
    DevBuf b;
    b.h = createBufferFrom(dev, data);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

// The push blob the bridge marshals for {8,8,4,4,1}: 25 bytes rounded to the
// 8-alignment.  The bool byte lands at 24 and the bridge zero-fills 25..31.
struct SsPush
{
    uint64_t particleSystems;
    uint64_t activeParticleSystems;
    float invDt;
    uint32_t partitionId;
    uint32_t isTGSWord;
    uint32_t pad28;
};
static_assert(sizeof(SsPush) == 32, "the ps_solveSpringsLaunch blob is 32 bytes");

static const uint32_t POISON_U32 = 0xABABABABu;
static const uint32_t BLOCK = 512u;   // PxgParticleSystemKernelBlockDim::ACCUMULATE_DELTA
static const uint32_t CLOTH_ARRAY_SLOTS = 4u;
static const uint32_t PARTITION_LEAD_IN = 4u;

// PxMin / PxAbs for float are intrinsics::selectMin and fabs -- plain ternaries,
// not fminf, so they are spelled the same way here and in the shader.
static float pxMinf(float a, float b) { return a < b ? a : b; }
static float pxAbsf(float a) { return std::fabs(a); }

static uint32_t hashu(uint32_t k, uint32_t salt)
{
    uint32_t h = k * 2654435761u + salt * 40503u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return h;
}

static float hashf(uint32_t k, uint32_t salt, float lo, float hi)
{
    const uint32_t h = hashu(k, salt) & 0xFFFFFFu;
    return lo + (hi - lo) * (float(h) / float(0x1000000u));
}

// ---------------------------------------------------------------------------
// One cloth record: the host image of everything the kernel reads and writes.
// ---------------------------------------------------------------------------
struct ClothRec
{
    uint32_t numSprings = 0;        // what mNumSprings reports (C1 lies: 0)
    uint32_t backingSprings = 0;    // how much storage actually exists
    uint32_t numPartitions = 0;
    bool live = false;              // active system AND inside mNumClothBuffers

    std::vector<PxParticleSpring> springs;   // backingSprings + 4 tail entries
    std::vector<uint32_t> partitions;        // lead-in + numPartitions + tail
    std::vector<uint32_t> writeIdx;          // 2*backingSprings + 8
    std::vector<F4> poses, vels;             // 4*backingSprings + 8
    std::vector<float> lambda;               // backingSprings + 8

    // the uploaded images, kept so the reference can start from them
    std::vector<F4> poses0, vels0;
    std::vector<float> lambda0;

    DevBuf bSprings, bPartitions, bWriteIdx, bPoses, bVels, bLambda;

    uint32_t remapSlots() const { return backingSprings * 4u + 8u; }
    uint32_t lambdaSlots() const { return backingSprings + 8u; }
    uint32_t partitionStart(uint32_t p) const
    {
        return p > 0u ? partitions[PARTITION_LEAD_IN + p - 1u] : 0u;
    }
    uint32_t partitionEnd(uint32_t p) const { return partitions[PARTITION_LEAD_IN + p]; }
};

struct SysRec
{
    uint32_t numClothBuffers = 0;
    std::vector<ClothRec> cloths;
    DevBuf bClothArr;
};

// ---------------------------------------------------------------------------
// Coverage: every path the kernel can take must be hit at least once, counted
// rather than assumed.
// ---------------------------------------------------------------------------
struct Coverage
{
    uint32_t solved = 0;          // reached the force block
    uint32_t wsumZero = 0;        // wsum <= 0, early out
    uint32_t degenerate = 0;      // dSq <= 1e-8, early out
    uint32_t nearThreshold = 0;   // 1e-8 < dSq < 1e-6, so the constant matters
    uint32_t justBelow = 0;       // 1e-10 < dSq <= 1e-8, likewise
    uint32_t tetherClamped = 0;   // k < 0 and e > 0 -> e := 0
    uint32_t tetherActive = 0;    // k < 0 and e < 0 -> the negated k acts
    uint32_t errorBiasHit = 0;    // |dlambda| > |errorBias|
    uint32_t errorBiasFlip = 0;   // ... and the two have opposite signs
    uint32_t oneFixed = 0;        // exactly one endpoint has invMass 0
    uint32_t zeroDamping = 0;
};

// ---------------------------------------------------------------------------
// The mirror: one partition of one cloth record, transcribed from
// particlesystem.cu:4137-4309.
// ---------------------------------------------------------------------------
static void refSolvePartition(ClothRec& c, float invDt, uint32_t partitionId, bool isTGS,
                              Coverage& cov)
{
    const uint32_t N = c.numSprings;
    if (N == 0u || partitionId >= c.numPartitions)
        return;

    const uint32_t startInd = c.partitionStart(partitionId);
    const uint32_t endInd = c.partitionEnd(partitionId);

    // Every scatter target is disjoint from every read slot of this partition
    // (asserted in the caller), so the whole partition reads the pre-dispatch
    // image no matter how the threads interleave.
    const std::vector<F4> poses = c.poses;
    const std::vector<F4> vels = c.vels;

    for (uint32_t workIndex = startInd; workIndex < endInd; ++workIndex)
    {
        const PxParticleSpring& sp = c.springs[workIndex];

        const F4 pos_im0 = poses[workIndex];
        const F4 pos_im1 = poses[workIndex + N];

        const float xi[3] = { pos_im0.x, pos_im0.y, pos_im0.z };
        const float xj[3] = { pos_im1.x, pos_im1.y, pos_im1.z };

        const F4 v0 = vels[workIndex];
        const F4 v1 = vels[workIndex + N];
        const float veli[3] = { v0.x, v0.y, v0.z };
        const float velj[3] = { v1.x, v1.y, v1.z };

        const float xij[3] = { xi[0] - xj[0], xi[1] - xj[1], xi[2] - xj[2] };
        const float wsum = pos_im0.w + pos_im1.w;

        float newPos0[3] = { xi[0], xi[1], xi[2] };
        float newPos1[3] = { xj[0], xj[1], xj[2] };
        float newVel0[3] = { veli[0], veli[1], veli[2] };
        float newVel1[3] = { velj[0], velj[1], velj[2] };

        const uint32_t writeIndex0 = c.writeIdx[workIndex];
        const uint32_t writeIndex1 = c.writeIdx[workIndex + N];

        if (wsum > 0.f)
        {
            const float dSq = xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2];

            if (dSq > 1e-8f)
            {
                ++cov.solved;
                if (dSq < 1e-6f) ++cov.nearThreshold;

                const float l = std::sqrt(dSq);

                const float len = sp.length;
                float e = len - l;
                float k = sp.stiffness;

                if (k < 0.0f)
                {
                    if (e > 0.0f) ++cov.tetherClamped; else ++cov.tetherActive;
                    e = pxMinf(e, 0.0f);
                    k *= -1.0f;
                }

                const float invL = 1.0f / l;                       // PxVec3::operator/
                const float cgrad[3] = { xij[0] * invL, xij[1] * invL, xij[2] * invL };
                const float dvel[3] = { veli[0] - velj[0], veli[1] - velj[1], veli[2] - velj[2] };
                const float dcdt = cgrad[0] * dvel[0] + cgrad[1] * dvel[1] + cgrad[2] * dvel[2];

                const float dt = 1.0f / invDt;
                const float b = dt * k;
                const float d = dt * sp.damping;
                const float a = dt * b + d;

                if (sp.damping == 0.0f) ++cov.zeroDamping;

                const float x = 1.0f / (1.0f + a * wsum);
                const float bias = x * b * e;
                const float velBias = -x * a * dcdt;

                float dlambda = 0.0f;

                if (!isTGS)
                {
                    const float iMul = 1.0f - x;
                    const float lambda = c.lambda[workIndex];
                    dlambda = ((bias + velBias) * dt + iMul * lambda) - lambda;
                    c.lambda[workIndex] = lambda + dlambda;
                }
                else
                {
                    dlambda = (bias + velBias) * dt;
                }

                const float errorBias = e / wsum;
                if (pxAbsf(dlambda) > pxAbsf(errorBias))
                {
                    ++cov.errorBiasHit;
                    if ((dlambda < 0.0f) != (errorBias < 0.0f) && errorBias != 0.0f)
                        ++cov.errorBiasFlip;
                    dlambda = errorBias;
                }

                const float deltaV[3] = { dlambda * cgrad[0], dlambda * cgrad[1],
                                          dlambda * cgrad[2] };
                const float deltaV0[3] = { deltaV[0] * pos_im0.w, deltaV[1] * pos_im0.w,
                                           deltaV[2] * pos_im0.w };
                const float deltaV1[3] = { deltaV[0] * pos_im1.w, deltaV[1] * pos_im1.w,
                                           deltaV[2] * pos_im1.w };

                if ((pos_im0.w == 0.0f) != (pos_im1.w == 0.0f)) ++cov.oneFixed;

                for (int i = 0; i < 3; ++i)
                {
                    newPos0[i] += deltaV0[i];
                    newPos1[i] -= deltaV1[i];
                    newVel0[i] += deltaV0[i] * invDt;
                    newVel1[i] -= deltaV1[i] * invDt;
                }
            }
            else
            {
                ++cov.degenerate;
                if (dSq > 1e-10f) ++cov.justBelow;
            }
        }
        else
        {
            ++cov.wsumZero;
        }

        c.poses[writeIndex0] = { newPos0[0], newPos0[1], newPos0[2], pos_im0.w };
        c.poses[writeIndex1] = { newPos1[0], newPos1[1], newPos1[2], pos_im1.w };
        c.vels[writeIndex0] = { newVel0[0], newVel0[1], newVel0[2], 0.0f };
        c.vels[writeIndex1] = { newVel1[0], newVel1[1], newVel1[2], 0.0f };
    }
}

// ---------------------------------------------------------------------------
// The nine hand-built springs of A0, one per decision in the kernel.
// ---------------------------------------------------------------------------
struct SpringCase
{
    float p0[3], im0;
    float p1[3], im1;
    float v0[3], v1[3];
    float length, stiffness, damping;
};

static const SpringCase kCasesA0[9] = {
    // s0 -- the ordinary case: both endpoints free, stretched, nothing clamped.
    { { 1.5f, 2.25f, -0.75f }, 0.8f, { 2.3f, 2.65f, -0.35f }, 0.5f,
      { 0.3f, -0.2f, 0.15f }, { -0.1f, 0.25f, -0.05f }, 0.9f, 120.0f, 3.0f },
    // s1 -- NEGATIVE stiffness, COMPRESSED: e := PxMin(e, 0) kills the whole
    //       correction (errorBias becomes 0 and the assignment zeroes dlambda).
    //       A port that only takes |k| pushes the endpoints apart instead.
    { { 0.4f, 1.1f, 0.6f }, 0.6f, { 0.9f, 1.4f, 0.95f }, 0.7f,
      { 0.5f, -0.4f, 0.2f }, { 0.2f, 0.1f, -0.3f }, 0.95f, -140.0f, 2.5f },
    // s2 -- NEGATIVE stiffness, EXTENDED: the clamp is a no-op and the k *= -1
    //       is what the result depends on (dropping it flips dlambda's sign).
    { { -1.2f, 0.5f, 2.0f }, 0.75f, { -0.4f, 0.9f, 1.55f }, 0.55f,
      { 0.1f, 0.2f, -0.15f }, { -0.2f, 0.05f, 0.25f }, 0.8f, -80.0f, 2.0f },
    // s3 -- BOTH endpoints fixed: wsum == 0.  The positions come back unchanged
    //       either way, so what makes the guard observable is springLambda: with
    //       the guard dropped the PGS branch writes (bias+velBias)*dt into it.
    { { 3.1f, -0.4f, 1.2f }, 0.0f, { 2.4f, -0.9f, 0.7f }, 0.0f,
      { 0.2f, 0.1f, 0.3f }, { -0.15f, 0.4f, -0.2f }, 0.7f, 90.0f, 1.5f },
    // s4 -- DEGENERATE, dSq in (1e-10, 1e-8]: skipped here, solved by a port
    //       that borrowed ps_solveDensityLaunch's 1e-10f.
    { { 0.5f, 0.25f, -0.125f }, 0.9f, { 0.50006f, 0.25004f, -0.125f }, 0.4f,
      { 0.12f, -0.08f, 0.05f }, { -0.03f, 0.07f, -0.02f }, 0.3f, 100.0f, 1.0f },
    // s5 -- just ABOVE the threshold, dSq in (1e-8, 1e-6): solved here, skipped
    //       by a port that rounded the constant up.
    { { -2.0f, 3.0f, 0.5f }, 0.5f, { -2.00018f, 3.0001f, 0.5f }, 0.5f,
      { 0.05f, -0.03f, 0.02f }, { -0.02f, 0.04f, -0.01f }, 0.0005f, 100.0f, 1.0f },
    // s6 -- the errorBias ASSIGNMENT with a SIGN FLIP: |dlambda| = 0.068 with
    //       dlambda < 0 while errorBias = +0.05, so the assignment REVERSES the
    //       correction where PxClamp(dlambda, -|eb|, |eb|) would not.
    { { 4.0f, 1.0f, 1.0f }, 1.2f, { 3.4f, 0.4f, 0.7f }, 0.8f,
      { 12.0f, 11.0f, 6.5f }, { -8.0f, -9.0f, -3.5f }, 1.0f, 100.0f, 10.0f },
    // s7 -- exactly ONE endpoint fixed: deltaV0 is zero and deltaV1 is not, so
    //       a port that weights both by wsum (or swaps the two masses) dies.
    { { 6.0f, 2.0f, -1.0f }, 0.0f, { 6.7f, 2.5f, -1.4f }, 1.25f,
      { 0.4f, 0.3f, -0.2f }, { 0.1f, -0.2f, 0.35f }, 1.2f, 150.0f, 4.0f },
    // s8 -- ZERO damping and two different non-zero masses: a = dt*b alone.
    { { -3.0f, -1.5f, 4.0f }, 1.6f, { -2.2f, -1.0f, 3.4f }, 0.35f,
      { 0.6f, -0.35f, 0.25f }, { 0.2f, 0.15f, -0.4f }, 1.0f, 200.0f, 0.0f },
};

// ---------------------------------------------------------------------------
// Record construction
// ---------------------------------------------------------------------------
static void fillPoison(ClothRec& c)
{
    F4 pf;
    const uint32_t p = POISON_U32;
    float pv; std::memcpy(&pv, &p, 4);
    pf = { pv, pv, pv, pv };
    c.poses.assign(c.remapSlots(), pf);
    c.vels.assign(c.remapSlots(), pf);
    c.lambda.assign(c.lambdaSlots(), pv);
}

// The gathered input occupies [0, 2N); everything past it starts poisoned.
static void seedEndpoint(ClothRec& c, uint32_t w, const SpringCase& sc)
{
    const uint32_t N = c.backingSprings;
    c.poses[w] = { sc.p0[0], sc.p0[1], sc.p0[2], sc.im0 };
    c.poses[w + N] = { sc.p1[0], sc.p1[1], sc.p1[2], sc.im1 };
    c.vels[w] = { sc.v0[0], sc.v0[1], sc.v0[2], 0.0f };
    c.vels[w + N] = { sc.v1[0], sc.v1[1], sc.v1[2], 0.0f };

    PxParticleSpring sp;
    std::memset(&sp, 0, sizeof(sp));
    sp.ind0 = 7u + w * 3u;          // unused by this kernel, but never zero
    sp.ind1 = 11u + w * 5u;
    sp.length = sc.length;
    sp.stiffness = sc.stiffness;
    sp.damping = sc.damping;
    c.springs[w] = sp;
}

// partitions: PARTITION_LEAD_IN entries of 2 (so a base-pointer mutant reads a
// live, wrong start index rather than faulting), the real accumulated counts,
// then four STRICTLY INCREASING tail entries N+1 .. N+4.  The tail is what gives
// a dropped `partitionId < nbPartitions` guard something to do: with a flat tail
// every past-the-end range is empty and the guard mutant is an identity.  The
// springs array carries four spare records and the remap arrays 4N+8 slots
// precisely so that extra range stays inside every allocation.
static void buildPartitions(ClothRec& c, const std::vector<uint32_t>& accumulated)
{
    c.numPartitions = uint32_t(accumulated.size());
    c.partitions.assign(PARTITION_LEAD_IN, 2u);
    for (uint32_t v : accumulated) c.partitions.push_back(v);
    for (uint32_t i = 0; i < 4u; ++i) c.partitions.push_back(c.backingSprings + 1u + i);
}

// Every target lands in the accumulation region [2N, 4N); the eight tail entries
// point at the last eight slots so an endpoint-stride mutant scatters somewhere
// legal and visible.
static void buildScatterTail(ClothRec& c, uint32_t salt)
{
    const uint32_t N = c.backingSprings;
    const uint32_t span = N * 2u;
    std::vector<uint32_t> perm(span);
    for (uint32_t i = 0; i < span; ++i) perm[i] = i;
    for (uint32_t i = span; i > 1u; --i)   // deterministic Fisher-Yates
    {
        const uint32_t j = hashu(i, salt) % i;
        const uint32_t t = perm[i - 1u]; perm[i - 1u] = perm[j]; perm[j] = t;
    }
    c.writeIdx.assign(span + 8u, 0u);
    for (uint32_t i = 0; i < span; ++i) c.writeIdx[i] = span + perm[i];
    for (uint32_t i = 0; i < 8u; ++i) c.writeIdx[span + i] = N * 4u + i;
}

PXG_TEST(psSolveSprings)
{
    PxgSlangComputeDevice* dev = testDevice();

    // Not 1, and not a power of two, so dt = 1/invDt is inexact and both the
    // reciprocal-of-a-reciprocal (S2) and the invDt velocity scale (S8) show.
    const float invDt = 60.0f;

    // -----------------------------------------------------------------------
    // Build the host image once; each round re-uploads it.
    // -----------------------------------------------------------------------
    SysRec sysA, sysD, sysB, sysC;
    SysRec* pool[4] = { &sysA, &sysD, &sysB, &sysC };

    auto makeRecord = [&](uint32_t backing, uint32_t reported,
                          const std::vector<uint32_t>& accumulated, uint32_t salt)
    {
        ClothRec c;
        c.backingSprings = backing;
        c.numSprings = reported;
        c.springs.assign(backing + 4u, PxParticleSpring());
        for (uint32_t i = 0; i < backing + 4u; ++i)
        {
            std::memset(&c.springs[i], 0, sizeof(PxParticleSpring));
            c.springs[i].length = 0.33f + float(i) * 0.017f;
            c.springs[i].stiffness = 55.0f + float(i);
            c.springs[i].damping = 0.5f + float(i) * 0.02f;
        }
        fillPoison(c);
        buildPartitions(c, accumulated);
        buildScatterTail(c, salt);
        return c;
    };

    // ---- A0: nine hand-built springs, three partitions -------------------
    {
        ClothRec c = makeRecord(9u, 9u, { 4u, 7u, 9u }, 4001u);
        for (uint32_t w = 0; w < 9u; ++w) seedEndpoint(c, w, kCasesA0[w]);
        sysA.cloths.push_back(c);
    }
    // ---- A1: 700 procedural springs, 600 of them in partition 0 ----------
    {
        const uint32_t N = 700u;
        ClothRec c = makeRecord(N, N, { 600u, 700u }, 4002u);
        for (uint32_t w = 0; w < N; ++w)
        {
            SpringCase sc;
            for (int i = 0; i < 3; ++i)
                sc.p0[i] = hashf(w, 10u + uint32_t(i), -4.0f, 4.0f);
            const bool degenerate = (w % 97u) == 5u;
            for (int i = 0; i < 3; ++i)
            {
                float d = hashf(w, 20u + uint32_t(i), -1.2f, 1.2f);
                d += (d < 0.0f ? -0.15f : 0.15f);
                sc.p1[i] = degenerate ? sc.p0[i] + d * 4e-5f : sc.p0[i] + d;
            }
            const bool bothFixed = (w % 53u) == 7u;
            const bool oneFixed = (w % 31u) == 3u;
            sc.im0 = bothFixed ? 0.0f : hashf(w, 30u, 0.2f, 1.5f);
            sc.im1 = (bothFixed || oneFixed) ? 0.0f : hashf(w, 31u, 0.2f, 1.5f);
            const float vs = ((w % 61u) == 9u) ? 22.0f : 3.0f;
            for (int i = 0; i < 3; ++i)
            {
                sc.v0[i] = hashf(w, 40u + uint32_t(i), -vs, vs);
                sc.v1[i] = hashf(w, 50u + uint32_t(i), -vs, vs);
            }
            sc.length = hashf(w, 60u, 0.2f, 2.0f);
            sc.stiffness = ((w % 7u) == 2u) ? -hashf(w, 70u, 30.0f, 220.0f)
                                            : hashf(w, 71u, 30.0f, 300.0f);
            sc.damping = ((w % 11u) == 4u) ? 0.0f : hashf(w, 80u, 0.5f, 8.0f);
            seedEndpoint(c, w, sc);
        }
        sysA.cloths.push_back(c);
    }
    // ---- A2: PADDING past mNumClothBuffers, real record, must stay poisoned
    {
        ClothRec c = makeRecord(5u, 5u, { 2u, 5u }, 4003u);
        for (uint32_t w = 0; w < 5u; ++w)
        {
            SpringCase sc;
            for (int i = 0; i < 3; ++i) sc.p0[i] = hashf(w, 90u + uint32_t(i), -2.0f, 2.0f);
            for (int i = 0; i < 3; ++i) sc.p1[i] = sc.p0[i] + hashf(w, 95u + uint32_t(i), 0.3f, 1.1f);
            sc.im0 = 0.7f; sc.im1 = 0.9f;
            for (int i = 0; i < 3; ++i) { sc.v0[i] = 0.4f + float(i); sc.v1[i] = -0.3f - float(i); }
            sc.length = 0.8f; sc.stiffness = 130.0f; sc.damping = 2.0f;
            seedEndpoint(c, w, sc);
        }
        sysA.cloths.push_back(c);
    }
    // ---- A3: a second PADDING record, so the array is 4 deep --------------
    {
        ClothRec c = makeRecord(8u, 8u, { 3u, 8u }, 4004u);
        for (uint32_t w = 0; w < 8u; ++w)
        {
            SpringCase sc;
            for (int i = 0; i < 3; ++i) sc.p0[i] = 12.0f + float(w) * 0.4f + float(i) * 0.3f;
            for (int i = 0; i < 3; ++i) sc.p1[i] = sc.p0[i] + 0.62f - float(i) * 0.11f;
            sc.im0 = 0.85f; sc.im1 = 0.45f;
            for (int i = 0; i < 3; ++i) { sc.v0[i] = 0.55f * float(i + 1); sc.v1[i] = -0.44f * float(i + 1); }
            sc.length = 0.95f; sc.stiffness = 165.0f; sc.damping = 2.6f;
            seedEndpoint(c, w, sc);
        }
        sysA.cloths.push_back(c);
    }
    sysA.numClothBuffers = 2u;

    // ---- the DECOY: three live records, never dispatched ------------------
    for (uint32_t b = 0; b < 4u; ++b)
    {
        ClothRec c = makeRecord(4u, 4u, { 2u, 4u }, 5000u + b);
        for (uint32_t w = 0; w < 4u; ++w)
        {
            SpringCase sc;
            for (int i = 0; i < 3; ++i) sc.p0[i] = 1.0f + float(b) + float(w) * 0.3f + float(i);
            for (int i = 0; i < 3; ++i) sc.p1[i] = sc.p0[i] + 0.4f + float(i) * 0.1f;
            sc.im0 = 0.6f; sc.im1 = 0.8f;
            for (int i = 0; i < 3; ++i) { sc.v0[i] = 0.2f * float(i + 1); sc.v1[i] = -0.5f * float(i + 1); }
            sc.length = 0.6f; sc.stiffness = 111.0f; sc.damping = 1.7f;
            seedEndpoint(c, w, sc);
        }
        sysD.cloths.push_back(c);
    }
    sysD.numClothBuffers = 4u;

    // ---- B0: six springs, FOUR partitions, CHAINED mRemapOutput -----------
    {
        ClothRec c = makeRecord(6u, 6u, { 1u, 3u, 4u, 6u }, 6001u);
        for (uint32_t w = 0; w < 6u; ++w)
        {
            SpringCase sc;
            sc.p0[0] = -1.0f + float(w) * 0.7f; sc.p0[1] = 2.0f - float(w) * 0.35f;
            sc.p0[2] = 0.6f + float(w) * 0.15f;
            sc.p1[0] = sc.p0[0] + 0.55f + float(w) * 0.05f;
            sc.p1[1] = sc.p0[1] - 0.4f - float(w) * 0.03f;
            sc.p1[2] = sc.p0[2] + 0.33f;
            sc.im0 = 0.45f + float(w) * 0.11f;
            sc.im1 = 1.3f - float(w) * 0.13f;
            for (int i = 0; i < 3; ++i)
            {
                sc.v0[i] = 0.9f - float(w) * 0.2f + float(i) * 0.3f;
                sc.v1[i] = -0.6f + float(w) * 0.17f - float(i) * 0.25f;
            }
            sc.length = 0.65f + float(w) * 0.09f;
            sc.stiffness = (w == 4u) ? -95.0f : (140.0f + float(w) * 7.0f);
            sc.damping = 1.1f + float(w) * 0.4f;
            seedEndpoint(c, w, sc);
        }
        // The scatter that makes the partition ORDER load bearing: partition 0's
        // spring writes onto partition 1's read slots, and partition 1's first
        // spring writes onto partition 2's.  Everything else goes to the tail.
        // (Slots: N = 6, reads [0,12), accumulation [12,24).)
        const uint32_t chain[12] = { 1u, 3u, 12u, 14u, 16u, 18u,
                                     7u, 9u, 13u, 15u, 17u, 19u };
        for (uint32_t i = 0; i < 12u; ++i) c.writeIdx[i] = chain[i];
        sysB.cloths.push_back(c);
    }
    for (uint32_t b = 1; b < 4u; ++b)   // padding past mNumClothBuffers = 1
    {
        ClothRec c = makeRecord(3u + b, 3u + b, { 1u, 3u + b }, 6100u + b);
        for (uint32_t w = 0; w < 3u + b; ++w)
        {
            SpringCase sc;
            for (int i = 0; i < 3; ++i) sc.p0[i] = -2.0f - float(w) - float(i) * 0.4f;
            for (int i = 0; i < 3; ++i) sc.p1[i] = sc.p0[i] + 0.5f + float(i) * 0.2f;
            sc.im0 = 0.55f; sc.im1 = 0.65f;
            for (int i = 0; i < 3; ++i) { sc.v0[i] = 0.7f - float(i); sc.v1[i] = -0.4f + float(i); }
            sc.length = 0.75f; sc.stiffness = 123.0f; sc.damping = 2.2f;
            seedEndpoint(c, w, sc);
        }
        sysB.cloths.push_back(c);
    }
    sysB.numClothBuffers = 1u;

    // ---- C0: five springs in ONE partition (idle for partitionId 1..3) ----
    {
        ClothRec c = makeRecord(5u, 5u, { 5u }, 7001u);
        for (uint32_t w = 0; w < 5u; ++w)
        {
            SpringCase sc;
            for (int i = 0; i < 3; ++i) sc.p0[i] = 3.0f - float(w) * 0.5f + float(i) * 0.25f;
            for (int i = 0; i < 3; ++i) sc.p1[i] = sc.p0[i] + 0.6f - float(i) * 0.15f;
            sc.im0 = 0.9f - float(w) * 0.12f;
            sc.im1 = 0.35f + float(w) * 0.16f;
            for (int i = 0; i < 3; ++i)
            {
                sc.v0[i] = -1.2f + float(w) * 0.4f + float(i) * 0.2f;
                sc.v1[i] = 0.8f - float(w) * 0.25f - float(i) * 0.3f;
            }
            sc.length = 1.05f - float(w) * 0.08f;
            sc.stiffness = 175.0f - float(w) * 12.0f;
            sc.damping = 3.3f - float(w) * 0.5f;
            seedEndpoint(c, w, sc);
        }
        sysC.cloths.push_back(c);
    }
    // ---- C1: mNumSprings == 0, but three springs' worth of real backing and a
    //          partition array claiming three, so `nbSprings > 0` has something
    //          to stop rather than an empty array that stops it by accident.
    {
        ClothRec c = makeRecord(3u, 0u, { 3u, 3u }, 7002u);
        for (uint32_t w = 0; w < 3u; ++w)
        {
            SpringCase sc;
            for (int i = 0; i < 3; ++i) sc.p0[i] = -5.0f + float(w) + float(i) * 0.3f;
            for (int i = 0; i < 3; ++i) sc.p1[i] = sc.p0[i] + 0.45f + float(i) * 0.1f;
            sc.im0 = 0.8f; sc.im1 = 0.6f;
            for (int i = 0; i < 3; ++i) { sc.v0[i] = 0.5f + float(i); sc.v1[i] = -0.9f - float(i); }
            sc.length = 0.5f; sc.stiffness = 160.0f; sc.damping = 1.9f;
            seedEndpoint(c, w, sc);
        }
        sysC.cloths.push_back(c);
    }
    {
        ClothRec c = makeRecord(2u, 2u, { 1u, 2u }, 7003u);   // padding
        for (uint32_t w = 0; w < 2u; ++w)
        {
            SpringCase sc;
            for (int i = 0; i < 3; ++i) sc.p0[i] = 8.0f + float(w) + float(i) * 0.2f;
            for (int i = 0; i < 3; ++i) sc.p1[i] = sc.p0[i] + 0.7f;
            sc.im0 = 0.5f; sc.im1 = 0.5f;
            for (int i = 0; i < 3; ++i) { sc.v0[i] = 0.25f * float(i + 1); sc.v1[i] = -0.35f * float(i + 1); }
            sc.length = 0.9f; sc.stiffness = 145.0f; sc.damping = 2.4f;
            seedEndpoint(c, w, sc);
        }
        sysC.cloths.push_back(c);
    }
    {
        ClothRec c = makeRecord(7u, 7u, { 4u, 7u }, 7004u);   // padding
        for (uint32_t w = 0; w < 7u; ++w)
        {
            SpringCase sc;
            for (int i = 0; i < 3; ++i) sc.p0[i] = -8.0f - float(w) * 0.3f + float(i) * 0.25f;
            for (int i = 0; i < 3; ++i) sc.p1[i] = sc.p0[i] + 0.48f + float(i) * 0.13f;
            sc.im0 = 0.72f; sc.im1 = 0.38f;
            for (int i = 0; i < 3; ++i) { sc.v0[i] = -0.65f * float(i + 1); sc.v1[i] = 0.31f * float(i + 1); }
            sc.length = 0.7f; sc.stiffness = 152.0f; sc.damping = 3.1f;
            seedEndpoint(c, w, sc);
        }
        sysC.cloths.push_back(c);
    }
    sysC.numClothBuffers = 2u;

    // -----------------------------------------------------------------------
    // The lambda seed: distinct and NON-zero, so the PGS warm start reads
    // something and the TGS round can prove it read nothing.
    // -----------------------------------------------------------------------
    uint32_t lamSalt = 0;
    for (SysRec* s : pool)
        for (ClothRec& c : s->cloths)
        {
            for (uint32_t i = 0; i < c.backingSprings; ++i)
                c.lambda[i] = ((i + lamSalt) % 3u == 1u ? -1.0f : 1.0f)
                            * (0.004f + 0.0013f * float(i) + 0.0007f * float(lamSalt));
            ++lamSalt;
        }

    // -----------------------------------------------------------------------
    // Liveness, the grid, and the safety properties the mirror relies on.
    // -----------------------------------------------------------------------
    // Only the first ACTIVE_COUNT entries are dispatched (gridZ); the rest exist
    // so that a grid mutant which indexes the active list by gridDim.Y (extent 4)
    // reads a live decoy index instead of running off the allocation.
    const uint32_t ACTIVE_COUNT = 3u;
    const std::vector<uint32_t> active = { 2u, 0u, 3u, 1u, 1u, 1u, 1u, 1u };
    for (uint32_t i = 0; i < 4u; ++i)
    {
        const bool isActive = (i == 2u || i == 0u || i == 3u);
        for (uint32_t b = 0; b < uint32_t(pool[i]->cloths.size()); ++b)
            pool[i]->cloths[b].live = isActive && (b < pool[i]->numClothBuffers);
    }

    uint32_t maxPartitions = 0, maxSpringsPerPartition = 0, maxClothBuffers = 0;
    for (SysRec* s : pool)
    {
        maxClothBuffers = PxMax(maxClothBuffers, uint32_t(s->cloths.size()));
        for (ClothRec& c : s->cloths)
        {
            maxPartitions = PxMax(maxPartitions, c.numPartitions);
            for (uint32_t p = 0; p < c.numPartitions; ++p)
                maxSpringsPerPartition =
                    PxMax(maxSpringsPerPartition, c.partitionEnd(p) - c.partitionStart(p));
        }
    }
    const uint32_t gridX = (maxSpringsPerPartition + BLOCK - 1u) / BLOCK;

    PXG_CHECK_MSG(maxClothBuffers == CLOTH_ARRAY_SLOTS, "the Y extent must be %u (got %u)",
                  CLOTH_ARRAY_SLOTS, maxClothBuffers);
    PXG_CHECK_MSG(gridX > 1u, "the X grid must span more than one block (got %u for %u springs)",
                  gridX, maxSpringsPerPartition);
    PXG_CHECK_MSG(maxPartitions == 4u, "B0 must own the deepest partition stack (got %u)",
                  maxPartitions);
    PXG_CHECK_MSG(sysB.numClothBuffers < CLOTH_ARRAY_SLOTS,
                  "B must carry real cloth records past its own bound");
    // The lesson of the first sweep, kept as an invariant rather than a comment:
    // a square grid makes swapping gridDim.Y for gridDim.Z an identity.
    PXG_CHECK_MSG(maxClothBuffers != ACTIVE_COUNT,
                  "the grid must NOT be square (Y = %u, Z = %u)", maxClothBuffers, ACTIVE_COUNT);
    PXG_CHECK_MSG(active.size() >= size_t(maxClothBuffers),
                  "the active list must survive being indexed by the Y extent (%zu vs %u)",
                  active.size(), maxClothBuffers);

    // Per record and per partition: the scatter targets are distinct, inside the
    // remap allocation, and disjoint from that partition's read slots.  Without
    // this the dispatch would race and the mirror would be a fiction.
    for (SysRec* s : pool)
        for (ClothRec& c : s->cloths)
        {
            if (c.numSprings == 0u) continue;
            const uint32_t N = c.numSprings;
            for (uint32_t p = 0; p < c.numPartitions; ++p)
            {
                std::set<uint32_t> reads, targets;
                for (uint32_t w = c.partitionStart(p); w < c.partitionEnd(p); ++w)
                {
                    reads.insert(w); reads.insert(w + N);
                    targets.insert(c.writeIdx[w]); targets.insert(c.writeIdx[w + N]);
                }
                const size_t want = 2u * size_t(c.partitionEnd(p) - c.partitionStart(p));
                PXG_CHECK_MSG(targets.size() == want,
                              "partition %u scatter targets must be distinct (%zu of %zu)",
                              p, targets.size(), want);
                for (uint32_t t : targets)
                {
                    PXG_CHECK_MSG(t < c.remapSlots(), "scatter target %u is past the allocation", t);
                    PXG_CHECK_MSG(reads.find(t) == reads.end(),
                                  "scatter target %u collides with partition %u's reads", t, p);
                }
            }
        }

    // -----------------------------------------------------------------------
    // One round = upload, dispatch every partition in order, compare.
    // -----------------------------------------------------------------------
    auto runRound = [&](bool isTGS, Coverage& cov) -> size_t
    {
        for (SysRec* s : pool)
        {
            std::vector<PxgParticleClothSimBuffer> arr(s->cloths.size());
            for (uint32_t b = 0; b < uint32_t(s->cloths.size()); ++b)
            {
                ClothRec& c = s->cloths[b];
                c.poses0 = c.poses; c.vels0 = c.vels; c.lambda0 = c.lambda;

                c.bSprings = makeDev(dev, c.springs);
                c.bPartitions = makeDev(dev, c.partitions);
                c.bWriteIdx = makeDev(dev, c.writeIdx);
                c.bPoses = makeDev(dev, c.poses);
                c.bVels = makeDev(dev, c.vels);
                c.bLambda = makeDev(dev, c.lambda);

                PxgParticleClothSimBuffer cb;
                std::memset(&cb, 0xEE, sizeof(cb));
                // the OFFSET pointer: the four lead-in entries sit BEFORE it
                cb.mAccumulatedSpringsPerPartitions = reinterpret_cast<PxU32*>(
                    size_t(c.bPartitions.addr) + size_t(PARTITION_LEAD_IN) * sizeof(PxU32));
                cb.mRemapOutput = reinterpret_cast<PxU32*>(size_t(c.bWriteIdx.addr));
                cb.mOrderedSprings = reinterpret_cast<PxParticleSpring*>(size_t(c.bSprings.addr));
                cb.mRemapPositions = reinterpret_cast<float4*>(size_t(c.bPoses.addr));
                cb.mRemapVelocities = reinterpret_cast<float4*>(size_t(c.bVels.addr));
                cb.mSpringLambda = reinterpret_cast<PxReal*>(size_t(c.bLambda.addr));
                cb.mNumSprings = c.numSprings;
                cb.mNumPartitions = c.numPartitions;
                cb.mParticleBufferIndex = 0u;
                arr[b] = cb;
            }
            s->bClothArr = makeDev(dev, arr);
        }

        DevBuf bActive = makeDev(dev, active);

        std::vector<PxgParticleSystem> descArr;
        for (SysRec* s : pool)
        {
            PxgParticleSystem ps;
            std::memset(&ps, 0xEE, sizeof(ps));
            ps.mClothSimBuffers =
                reinterpret_cast<PxgParticleClothSimBuffer*>(size_t(s->bClothArr.addr));
            ps.mNumClothBuffers = s->numClothBuffers;
            descArr.push_back(ps);
        }
        DevBuf bDesc = makeDev(dev, descArr);

        {
            Kernel k("npPsSprings.spv", "ps_solveSpringsLaunch", {}, sizeof(SsPush));
            for (uint32_t p = 0; p < maxPartitions; ++p)
            {
                SsPush push;
                std::memset(&push, 0, sizeof(push));
                push.particleSystems = bDesc.addr;
                push.activeParticleSystems = bActive.addr;
                push.invDt = invDt;
                push.partitionId = p;
                push.isTGSWord = isTGS ? 1u : 0u;
                k.dispatch(gridX, {}, push, maxClothBuffers, ACTIVE_COUNT);
                dev->flush();
            }
        }

        // The reference, replayed in the same partition order.
        for (SysRec* s : pool)
            for (ClothRec& c : s->cloths)
            {
                c.poses = c.poses0; c.vels = c.vels0; c.lambda = c.lambda0;
            }
        for (uint32_t p = 0; p < maxPartitions; ++p)
            for (SysRec* s : pool)
                for (ClothRec& c : s->cloths)
                    if (c.live)
                        refSolvePartition(c, invDt, p, isTGS, cov);

        size_t mm = 0;
        for (uint32_t si = 0; si < 4u; ++si)
        {
            SysRec* s = pool[si];
            for (uint32_t b = 0; b < uint32_t(s->cloths.size()); ++b)
            {
                ClothRec& c = s->cloths[b];
                const auto gPos = downloadBuffer<F4>(dev, c.bPoses.h, c.remapSlots());
                const auto gVel = downloadBuffer<F4>(dev, c.bVels.h, c.remapSlots());
                const auto gLam = downloadBuffer<float>(dev, c.bLambda.h, c.lambdaSlots());

                auto cmpF4 = [&](const std::vector<F4>& r, const std::vector<F4>& g,
                                 const char* what)
                {
                    for (size_t i = 0; i < r.size(); ++i)
                    {
                        const float* rp = &r[i].x; const float* gp = &g[i].x;
                        for (int k2 = 0; k2 < 4; ++k2)
                            if (f2u(rp[k2]) != f2u(gp[k2]))
                            {
                                if (mm < 12)
                                    std::printf("  %s sys %u buf %u %s slot %zu.%d: "
                                                "ref 0x%08x (%g) got 0x%08x (%g)\n",
                                                isTGS ? "TGS" : "PGS", si, b, what, i, k2,
                                                f2u(rp[k2]), double(rp[k2]),
                                                f2u(gp[k2]), double(gp[k2]));
                                ++mm;
                            }
                    }
                };
                cmpF4(c.poses, gPos, "remapPos");
                cmpF4(c.vels, gVel, "remapVel");
                for (size_t i = 0; i < c.lambda.size(); ++i)
                    if (f2u(c.lambda[i]) != f2u(gLam[i]))
                    {
                        if (mm < 12)
                            std::printf("  %s sys %u buf %u lambda %zu: ref 0x%08x (%g) "
                                        "got 0x%08x (%g)\n", isTGS ? "TGS" : "PGS", si, b, i,
                                        f2u(c.lambda[i]), double(c.lambda[i]),
                                        f2u(gLam[i]), double(gLam[i]));
                        ++mm;
                    }
            }
        }

        // On TGS the lambda array is never touched -- checked against the upload
        // rather than against the mirror, so a mirror that also wrote it
        // would not be able to hide the difference.
        if (isTGS)
            for (SysRec* s : pool)
                for (ClothRec& c : s->cloths)
                {
                    const auto gLam = downloadBuffer<float>(dev, c.bLambda.h, c.lambdaSlots());
                    for (size_t i = 0; i < c.lambda0.size(); ++i)
                        if (f2u(c.lambda0[i]) != f2u(gLam[i]))
                        {
                            if (mm < 12)
                                std::printf("  TGS wrote springLambda at %zu\n", i);
                            ++mm;
                        }
                }

        for (SysRec* s : pool)
        {
            for (ClothRec& c : s->cloths)
            {
                dev->destroyBuffer(c.bSprings.h); dev->destroyBuffer(c.bPartitions.h);
                dev->destroyBuffer(c.bWriteIdx.h); dev->destroyBuffer(c.bPoses.h);
                dev->destroyBuffer(c.bVels.h); dev->destroyBuffer(c.bLambda.h);
            }
            dev->destroyBuffer(s->bClothArr.h);
        }
        dev->destroyBuffer(bDesc.h);
        dev->destroyBuffer(bActive.h);
        return mm;
    };

    Coverage covPgs, covTgs;
    const size_t mmPgs = runRound(false, covPgs);
    const size_t mmTgs = runRound(true, covTgs);

    PXG_CHECK_MSG(mmPgs == 0, "the PGS spring solve is bit-exact (%zu mismatches)", mmPgs);
    PXG_CHECK_MSG(mmTgs == 0, "the TGS spring solve is bit-exact (%zu mismatches)", mmTgs);

    // Every decision in the kernel reached, counted rather than assumed.
    PXG_CHECK_MSG(covPgs.solved > 600u, "the force block must dominate (%u)", covPgs.solved);
    PXG_CHECK_MSG(covPgs.wsumZero >= 2u, "wsum == 0 must be hit (%u)", covPgs.wsumZero);
    PXG_CHECK_MSG(covPgs.degenerate >= 2u, "the dSq guard must fire (%u)", covPgs.degenerate);
    PXG_CHECK_MSG(covPgs.justBelow >= 1u,
                  "a pair must sit in (1e-10, 1e-8] so the constant matters (%u)", covPgs.justBelow);
    PXG_CHECK_MSG(covPgs.nearThreshold >= 1u,
                  "a solved pair must sit in (1e-8, 1e-6) (%u)", covPgs.nearThreshold);
    PXG_CHECK_MSG(covPgs.tetherClamped >= 1u, "a compressed tether must be clamped (%u)",
                  covPgs.tetherClamped);
    PXG_CHECK_MSG(covPgs.tetherActive >= 1u, "an extended tether must pull (%u)",
                  covPgs.tetherActive);
    PXG_CHECK_MSG(covPgs.errorBiasHit >= 2u, "the errorBias assignment must fire (%u)",
                  covPgs.errorBiasHit);
    PXG_CHECK_MSG(covPgs.errorBiasFlip >= 1u,
                  "the errorBias assignment must REVERSE dlambda at least once (%u)",
                  covPgs.errorBiasFlip);
    PXG_CHECK_MSG(covPgs.oneFixed >= 2u, "exactly one fixed endpoint must occur (%u)",
                  covPgs.oneFixed);
    PXG_CHECK_MSG(covPgs.zeroDamping >= 2u, "a zero-damping spring must be solved (%u)",
                  covPgs.zeroDamping);
    PXG_CHECK_MSG(covTgs.solved == covPgs.solved,
                  "both rounds must take the same control-flow path (%u vs %u)",
                  covTgs.solved, covPgs.solved);
}

} // namespace
