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

// Verification of softBodyPsPrepareBda.slang -- sb_particleContactPrepareLaunch, PhysX 5.12
// (softBody.cu).
//
// The mirror is transcribed from softBody.cu and the helpers it calls, not from the Slang: a
// mirror written from the port repeats the port's mistakes.
//
// 5.12 REWROTE THIS KERNEL.  The 5.6.1 PxgFEMParticleConstraintBlock (normal_pen, barycentric,
// velMultiplier from getSoftBodyInvMass's L2-scaled blended mass) became a
// PxgParticleDbContactBlock (normal_pen, barycentric, maxPenBiasClamp); the packed
// PxgFemOtherContactInfo became a PxgDeformablePairInfo (ref0 = PxgEncodeParticleRef,
// ref1 = the tetrahedron ref, an INVALID pair skipped after its applied forces are zeroed);
// and the rest distance is no longer the softbody's mRestDistance but the SUM
// particleSystem.mData.mRestOffset + softbody.mRestOffset, subtracted after the delta term:
//
//     pen = (normal_pen.w - dot(particleDelta - tetDelta, normal)) - restDistance
//     maxPenBiasClamp = PxMax(softbody.mPenBiasClamp, particleSystem.mData.mPenBiasClamp)
//
// ---------------------------------------------------------------------------
// WHAT THE SCENE IS BUILT TO SEPARATE
// ---------------------------------------------------------------------------
// S1  BOTH DELTAS ARE NON-ZERO AND DIFFERENT, so the sign of `- dot(particleDelta -
//     tetDelta, normal)` and the projection of the TET delta (all four corners, each with
//     its own value) are observable.
// S2  THE REST DISTANCE IS A SUM OF TWO PER-BODY TERMS.  Two softbodies (with a zero DECOY
//     between them) and two particle systems, all with different rest offsets, so reading
//     either term alone, or one body's term for every contact, fails.
// S3  THE CLAMP IS A SELECT-MAX OF TWO PER-BODY TERMS, with the larger one coming from the
//     softbody for some pairs and from the particle system for others.
// S4  THE WARP-BLOCKED LAYOUT.  45 contacts: two blocks of 32, the second partial; the lanes
//     past the count and the lanes of INVALID pairs keep their poison.
// S5  INVALID PAIRS are skipped -- but only AFTER both applied-force slots are zeroed.
// S6  THE TWO APPLIED-FORCE BUFFERS ARE DISTINCT float2 arrays, zeroed for every contact
//     (valid or not) and untouched past the count.
// S7  The `contacts` stream is NOT read: it is passed as a buffer of NaNs.

#include "pxgTestFramework.h"
#include "host_defines.h"
#include "vector_functions.h"
#include "PxgSoftBody.h"
#include "PxgParticleSystem.h"
#include "PxgSimulationCoreDesc.h"
#include "PxgDeformableConstraints.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif

namespace
{
using namespace pxgtest;
using namespace physx;

struct F4 { float x, y, z, w; };
struct F2 { float x, y; };
struct U4 { uint32_t x, y, z, w; };

// sb_particleContactPrepareLaunch (softBody.cu, 5.12), ten pointers.
struct SbPsPrepPush
{
    uint64_t softbodies, particleSystems, contacts, normalPens, barycentrics,
             contactInfos, numContacts, contactBlocks, softBodyAppliedForces,
             particleAppliedForces;
};
static_assert(sizeof(SbPsPrepPush) == 80, "sb prepare blob must be 80 bytes");
static_assert(offsetof(SbPsPrepPush, contacts) == 16, "contacts @16 -- THIRD, and dead");
static_assert(offsetof(SbPsPrepPush, contactBlocks) == 56, "contactBlocks @56");
static_assert(offsetof(SbPsPrepPush, particleAppliedForces) == 72, "particle forces LAST");

// PxgParticleDbContactBlock (PxgDeformableConstraints.h, 5.12): float4[32], float4[32], float[32].
static const uint32_t CB_STRIDE = 1152u;
static const uint32_t CB_NORMAL_PEN = 0u;
static const uint32_t CB_BARYCENTRIC = 512u;
static const uint32_t CB_MAXPENBIASCLAMP = 1024u;
static_assert(sizeof(PxgParticleDbContactBlock) == CB_STRIDE, "PxgParticleDbContactBlock stride");
static_assert(offsetof(PxgParticleDbContactBlock, barycentric) == CB_BARYCENTRIC, "barycentric");
static_assert(offsetof(PxgParticleDbContactBlock, maxPenBiasClamp) == CB_MAXPENBIASCLAMP, "maxPenBiasClamp");
static_assert(sizeof(PxgDeformablePairInfo) == 16, "PxgDeformablePairInfo");

// ---- the mirror, from softBody.cu -------------------------------------------------
// barycentricProjectTet (deformableUtils.cuh): a * b.x + b * b.y + c * b.z + d * b.w on
// whole float4s, left to right.
static F4 barycentricProjectTet(U4 tet, const std::vector<F4>& v, F4 b)
{
    const F4 a = v[tet.x], c = v[tet.y], d = v[tet.z], e = v[tet.w];
    F4 r;
    r.x = a.x * b.x + c.x * b.y + d.x * b.z + e.x * b.w;
    r.y = a.y * b.x + c.y * b.y + d.y * b.z + e.y * b.w;
    r.z = a.z * b.x + c.z * b.y + d.z * b.z + e.z * b.w;
    r.w = a.w * b.x + c.w * b.y + d.w * b.z + e.w * b.w;
    return r;
}
static float selectMax(float a, float b) { return (a < b) ? b : a; }   // PxMax

struct DevB { PxgBufferHandle h = 0; uint64_t addr = 0; };
template <class T> static DevB mk(PxgSlangComputeDevice* dev, const std::vector<T>& v)
{
    DevB b; b.h = createBufferFrom(dev, v); b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

struct Rng
{
    uint32_t s;
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    float uni(float lo, float hi) { return lo + (hi - lo) * float(next() & 0xffffffu) / float(0x1000000); }
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

PXG_TEST(softBodyPsPrepare)
{
    PxgSlangComputeDevice* dev = testDevice();
    Rng rng{ 0x5eed1234u };

    // ---- softbodies 0 and 2 (1 is a zero decoy); every tet has its own four vertices ----
    const uint32_t NT = 12u;
    const float SB_REST[3] = { 0.021f, 0.0f, 0.047f };
    const float SB_CLAMP[3] = { -0.6f, 0.0f, -2.5f };
    std::vector<U4> tets[3];
    std::vector<F4> deltaPos[3];
    std::vector<DevB> keep;
    std::vector<PxgSoftBody> sbArr(3);
    std::memset(sbArr.data(), 0, sbArr.size() * sizeof(PxgSoftBody));
    for (uint32_t s = 0; s < 3; s += 2)
    {
        for (uint32_t t = 0; t < NT; ++t)
            tets[s].push_back(U4{ 4 * t + 3, 4 * t + 1, 4 * t + 0, 4 * t + 2 });
        for (uint32_t v = 0; v < 4 * NT; ++v)
            deltaPos[s].push_back(F4{ rng.uni(-0.05f, 0.05f), rng.uni(-0.05f, 0.05f), rng.uni(-0.05f, 0.05f),
                                      rng.uni(0.5f, 2.0f) });
        const DevB bt = mk(dev, tets[s]), bd = mk(dev, deltaPos[s]);
        keep.push_back(bt); keep.push_back(bd);
        sbArr[s].mSimTetIndices = reinterpret_cast<uint4*>(size_t(bt.addr));
        sbArr[s].mSimDeltaPos = reinterpret_cast<float4*>(size_t(bd.addr));
        sbArr[s].mRestOffset = SB_REST[s];
        sbArr[s].mPenBiasClamp = SB_CLAMP[s];
    }
    const DevB bSb = mk(dev, sbArr);

    // ---- particle systems 0 and 1 ----
    const uint32_t NP = 40u;
    const float PS_REST[2] = { 0.013f, 0.032f };
    const float PS_CLAMP[2] = { -1.1f, -0.3f };   // S3: vs -0.6 / -2.5 the max comes from either side
    std::vector<F4> sortedDeltaP[2];
    std::vector<PxgParticleSystem> psArr(2);
    std::memset(psArr.data(), 0, psArr.size() * sizeof(PxgParticleSystem));
    for (uint32_t q = 0; q < 2; ++q)
    {
        for (uint32_t i = 0; i < NP; ++i)
            sortedDeltaP[q].push_back(F4{ rng.uni(-0.04f, 0.04f), rng.uni(-0.04f, 0.04f), rng.uni(-0.04f, 0.04f),
                                          rng.uni(0.5f, 2.0f) });
        const DevB b = mk(dev, sortedDeltaP[q]);
        keep.push_back(b);
        psArr[q].mSortedDeltaP = reinterpret_cast<float4*>(size_t(b.addr));
        psArr[q].mData.mRestOffset = PS_REST[q];
        psArr[q].mData.mPenBiasClamp = PS_CLAMP[q];
    }
    const DevB bPs = mk(dev, psArr);

    // ---- 45 contacts ----
    const uint32_t N = 45u, NB = (N + 31u) / 32u;
    std::vector<F4> normalPens, barys, deadContacts(N + 3u, F4{ NAN, NAN, NAN, NAN });
    std::vector<PxgDeformablePairInfo> infos;
    for (uint32_t c = 0; c < N; ++c)
    {
        const uint32_t s = (c % 3u == 1u) ? 2u : 0u;
        const uint32_t q = (c % 4u == 2u) ? 1u : 0u;
        PxgDeformablePairInfo pi;
        pi.ref0 = PxgEncodeParticleRef(q, (c * 7u) % NP);
        pi.ref1 = PxgEncodeDeformableTetrahedron(s, (c * 5u) % NT);
        if (c % 9u == 4u) pi.ref1 = PxgEncodeInvalidDeformableRef(s);   // S5
        infos.push_back(pi);
        float nx = rng.uni(-1.f, 1.f), ny = rng.uni(-1.f, 1.f), nz = rng.uni(-1.f, 1.f);
        const float l = std::sqrt(nx * nx + ny * ny + nz * nz);
        normalPens.push_back(F4{ nx / l, ny / l, nz / l, rng.uni(-0.05f, 0.05f) });
        float b[4] = { rng.uni(0.1f, 1.f), rng.uni(0.1f, 1.f), rng.uni(0.1f, 1.f), rng.uni(0.1f, 1.f) };
        const float sum = b[0] + b[1] + b[2] + b[3];
        barys.push_back(F4{ b[0] / sum, b[1] / sum, b[2] / sum, b[3] / sum });
    }
    std::vector<uint32_t> num = { N };
    const uint32_t POISON = 0xC5C5C5C5u;
    std::vector<uint32_t> blocksInit(NB * CB_STRIDE / 4u, POISON);
    std::vector<F2> sbForceInit(N + 6u, F2{ 3.0f, -3.0f }), psForceInit(N + 6u, F2{ -7.0f, 7.0f });

    const DevB bNp = mk(dev, normalPens), bBary = mk(dev, barys), bInfo = mk(dev, infos), bNum = mk(dev, num),
               bDead = mk(dev, deadContacts), bBlocks = mk(dev, blocksInit), bSbF = mk(dev, sbForceInit),
               bPsF = mk(dev, psForceInit);

    SbPsPrepPush p = {};
    p.softbodies = bSb.addr; p.particleSystems = bPs.addr; p.contacts = bDead.addr;
    p.normalPens = bNp.addr; p.barycentrics = bBary.addr; p.contactInfos = bInfo.addr;
    p.numContacts = bNum.addr; p.contactBlocks = bBlocks.addr;
    p.softBodyAppliedForces = bSbF.addr; p.particleAppliedForces = bPsF.addr;
    Kernel k("softBodyPsPrepareBda.spv", "sb_particleContactPrepareLaunch", {}, sizeof(SbPsPrepPush));
    k.dispatch(1024u, {}, p, 1u, 1u);
    dev->flush();

    // ---- the mirror ----
    std::vector<uint8_t> want(NB * CB_STRIDE);
    std::memcpy(want.data(), blocksInit.data(), want.size());
    uint32_t nValid = 0, nClampFromSb = 0, nClampFromPs = 0;
    for (uint32_t c = 0; c < N; ++c)
    {
        const PxgDeformablePairInfo& pi = infos[c];
        if (!PxgIsValidPairInfo(pi))
            continue;
        ++nValid;
        const uint32_t q = PxgGetParticleSystemId(pi.ref0), pidx = PxgGetParticleIndex(pi.ref0);
        const uint32_t s = PxgGetDeformableId(pi.ref1), t = PxgGetDeformableFeatureId(pi.ref1);
        const F4 tetDelta = barycentricProjectTet(tets[s][t], deltaPos[s], barys[c]);
        const F4 np = normalPens[c];
        const float nx = -np.x, ny = -np.y, nz = -np.z;
        const F4 pd = sortedDeltaP[q][pidx];
        const float rx = pd.x - tetDelta.x, ry = pd.y - tetDelta.y, rz = pd.z - tetDelta.z;
        const float restDistance = PS_REST[q] + SB_REST[s];
        const float pen = (np.w - (rx * nx + ry * ny + rz * nz)) - restDistance;
        const float clamp = selectMax(SB_CLAMP[s], PS_CLAMP[q]);
        if (SB_CLAMP[s] > PS_CLAMP[q]) ++nClampFromSb; else ++nClampFromPs;
        const size_t base = size_t(c / 32u) * CB_STRIDE, l = c & 31u;
        const F4 npOut = F4{ nx, ny, nz, pen };
        std::memcpy(&want[base + CB_NORMAL_PEN + l * 16], &npOut, 16);
        std::memcpy(&want[base + CB_BARYCENTRIC + l * 16], &barys[c], 16);
        std::memcpy(&want[base + CB_MAXPENBIASCLAMP + l * 4], &clamp, 4);
    }

    size_t bad = 0;
    const std::vector<uint8_t> got = downloadBuffer<uint8_t>(dev, bBlocks.h, want.size());
    for (size_t i = 0; i < want.size(); i += 4)
        if (std::memcmp(&got[i], &want[i], 4) != 0)
        {
            uint32_t g, w;
            std::memcpy(&g, &got[i], 4); std::memcpy(&w, &want[i], 4);
            if (++bad <= 8) std::printf("  block word @%zu got 0x%08x want 0x%08x\n", i, g, w);
        }
    const std::vector<F2> gSbF = downloadBuffer<F2>(dev, bSbF.h, N + 6u);
    const std::vector<F2> gPsF = downloadBuffer<F2>(dev, bPsF.h, N + 6u);
    for (uint32_t c = 0; c < N + 6u; ++c)
    {
        const F2 ws = (c < N) ? F2{ 0.f, 0.f } : sbForceInit[c];
        const F2 wp = (c < N) ? F2{ 0.f, 0.f } : psForceInit[c];
        if (f2u(gSbF[c].x) != f2u(ws.x) || f2u(gSbF[c].y) != f2u(ws.y) ||
            f2u(gPsF[c].x) != f2u(wp.x) || f2u(gPsF[c].y) != f2u(wp.y))
        { if (++bad <= 8) std::printf("  applied forces [%u] mismatch\n", c); }
    }
    PXG_CHECK_MSG(bad == 0, "sb_particleContactPrepareLaunch bit-exact vs the 5.12 CUDA mirror (%zu mismatches)", bad);
    PXG_CHECK_MSG(nValid > 32u && nValid < N, "coverage: %u valid of %u (both blocks, some invalid)", nValid, N);
    PXG_CHECK_MSG(nClampFromSb > 0u && nClampFromPs > 0u, "coverage: the clamp max must come from both sides (%u / %u)",
                  nClampFromSb, nClampFromPs);

    for (const DevB& b : keep) dev->destroyBuffer(b.h);
    for (const DevB* b : { &bSb, &bPs, &bNp, &bBary, &bInfo, &bNum, &bDead, &bBlocks, &bSbF, &bPsF })
        dev->destroyBuffer(b->h);
}

} // namespace
