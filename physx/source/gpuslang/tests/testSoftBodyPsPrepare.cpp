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

// Verification of softBodyPsPrepareBda.slang -- sb_particleContactPrepareLaunch (P9.39d).
//
// The mirror is transcribed from softBody.cu:844 and the helpers it calls, not from the
// Slang, for the reason P9.39c's header records: a mirror written from the port repeats
// the port's mistakes.
//
// ---------------------------------------------------------------------------
// WHAT THE SCENE IS BUILT TO SEPARATE
// ---------------------------------------------------------------------------
// H1  getSoftBodyInvMass's L2 SCALE.  The blended inverse mass is multiplied by
//     sqrt(bx^2+by^2+bz^2+bw^2), and the CLOTH twin has no such factor.  Every
//     barycentric here is chosen so that the scale differs measurably from 1 -- with a
//     barycentric like (1,0,0,0) the scale IS 1 and dropping the whole function is
//     invisible.  A coverage assertion enforces it.
//
// H2  THE .w OF THE BARYCENTRIC PROJECTION IS THE MASS INPUT.  mSimDeltaPos carries the
//     per-vertex inverse mass in .w, so a port that projected only .xyz would compile
//     and feed garbage to H1.  The four vertices of each tet carry DIFFERENT inverse
//     masses, so the blend is not a constant.
//
// H3  BOTH DELTAS ARE NON-ZERO AND DIFFERENT.  pen mixes the stored penetration, the
//     particle's accumulated delta and the tet's; with either delta zero, the sign
//     convention of `- dot(particleDelta - tetDelta, normal)` cannot be separated from
//     its opposite.
//
// H4  THE WARP-BLOCKED LAYOUT.  More than 32 contacts, so the constraint array is more
//     than one block and `constraints[workIndex/32]` indexed by `threadIdx.x & 31` has
//     to land in the right block AND the right lane.
//
// H5  THE TAIL (Q11).  The contact count is deliberately NOT a multiple of 32, and the
//     lanes past it are poisoned and asserted untouched -- the kernel returns before
//     writing them.
//
// H6  THE TWO APPLIED-FORCE BUFFERS ARE DISTINCT, and both are float2 arrays living in
//     float4-sized allocations: the kernel fills the first HALF OF EACH BUFFER densely
//     at stride 8, two contacts per 16-byte slot.  They are poisoned with different
//     patterns; everything past 8 * numContacts bytes must survive.
//
// H7  restDistance IS PER SOFTBODY.  Two softbodies with different rest distances, and
//     contacts against both, so reading softbody 0's value for every contact fails.

#include "pxgTestFramework.h"
#include "PxgSoftBody.h"
#include "PxgParticleSystem.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;

struct F4 { float x, y, z, w; };
struct F2 { float x, y; };
struct U4 { uint32_t x, y, z, w; };

// softBody.cu:845-854, ten pointers.
struct SbPsPrepPush
{
    uint64_t softbodies, particleSystems, contacts, normalPens, barycentrics,
             contactInfos, numContacts, constraints, softBodyAppliedForces,
             particleAppliedForces;
};
static_assert(sizeof(SbPsPrepPush) == 80, "sb prepare blob must be 80 bytes");
static_assert(offsetof(SbPsPrepPush, contacts) == 16, "contacts @16 -- THIRD, and dead");
static_assert(offsetof(SbPsPrepPush, constraints) == 56, "constraints @56");
static_assert(offsetof(SbPsPrepPush, particleAppliedForces) == 72, "particle forces LAST");

// PxgFEMParticleConstraintBlock (PxgFEMCore.h:184): float4[32], float4[32], float[32].
static const uint32_t CB_STRIDE = 1152u;
static const uint32_t CB_NORMAL_PEN = 0u;
static const uint32_t CB_BARYCENTRIC = 512u;
static const uint32_t CB_VELMULT = 1024u;

static const uint32_t BITSHIFT_TET = 20u;

// ---- the mirror, from softBody.cu -------------------------------------------------
static float dot3(F4 a, float bx, float by, float bz)
{ return a.x * bx + a.y * by + a.z * bz; }

// barycentricProject (softBody.cu:640): whole float4s, left to right.
static F4 barycentricProject(U4 tet, const std::vector<F4>& v, F4 b)
{
    const F4 a = v[tet.x], c = v[tet.y], d = v[tet.z], e = v[tet.w];
    F4 r;
    r.x = a.x * b.x + c.x * b.y + d.x * b.z + e.x * b.w;
    r.y = a.y * b.x + c.y * b.y + d.y * b.z + e.y * b.w;
    r.z = a.z * b.x + c.z * b.y + d.z * b.z + e.z * b.w;
    r.w = a.w * b.x + c.w * b.y + d.w * b.z + e.w * b.w;
    return r;
}

// getSoftBodyInvMass (softBody.cuh:38): baryMass * sqrt(sum of squares).  Written out
// rather than as a 4-vector length, which is the substitution this project has already
// measured diverging by one ULP (femClothDampingBda.slang:85-89).
static float getSoftBodyInvMass(float baryMass, F4 b)
{
    const float scale = std::sqrt(b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w);
    return baryMass * scale;
}

struct DevB { PxgBufferHandle h = 0; uint64_t addr = 0; };
template <class T> static DevB mk(PxgSlangComputeDevice* dev, const std::vector<T>& v)
{
    DevB b; b.h = createBufferFrom(dev, v); b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

PXG_TEST(softBodyPsPrepare)
{
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t NB_SB = 2u, NB_PS = 1u;
    // H7: different per softbody, neither zero, and not equal to each other.
    const float REST[NB_SB] = { 0.0625f, 0.1875f };

    // ---- two tet meshes, four sim vertices each, with DIFFERENT inverse masses in .w
    // (H2).  Positions are irrelevant here -- this kernel reads mSimDeltaPos, which is
    // an accumulated DELTA, not a position -- so the values are chosen to be dyadic and
    // distinguishable rather than geometric.
    std::vector<F4> simDelta0 = {
        { 0.25f,  -0.125f,  0.5f,    0.5f   },
        { -0.5f,   0.75f,  -0.25f,   0.25f  },
        { 0.125f,  0.25f,   0.125f,  1.0f   },
        { 0.75f,  -0.5f,    0.375f,  0.125f },
    };
    std::vector<F4> simDelta1 = {
        { -0.375f, 0.5f,    0.25f,   0.75f  },
        { 0.625f, -0.25f,   0.125f,  0.5f   },
        { 0.25f,   0.125f, -0.625f,  0.25f  },
        { -0.125f, 0.375f,  0.5f,    2.0f   },
    };
    DevB bDelta0 = mk(dev, simDelta0);
    if (!bDelta0.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }
    DevB bDelta1 = mk(dev, simDelta1);

    std::vector<U4> tets0 = { { 0u, 1u, 2u, 3u } };
    std::vector<U4> tets1 = { { 3u, 2u, 1u, 0u } };   // deliberately a different winding
    DevB bTets0 = mk(dev, tets0);
    DevB bTets1 = mk(dev, tets1);

    std::vector<PxgSoftBody> sbArr(NB_SB);
    std::memset(sbArr.data(), 0, sbArr.size() * sizeof(PxgSoftBody));
    sbArr[0].mSimTetIndices = reinterpret_cast<uint4*>(size_t(bTets0.addr));
    sbArr[0].mSimDeltaPos = reinterpret_cast<float4*>(size_t(bDelta0.addr));
    sbArr[0].mRestDistance = REST[0];
    sbArr[1].mSimTetIndices = reinterpret_cast<uint4*>(size_t(bTets1.addr));
    sbArr[1].mSimDeltaPos = reinterpret_cast<float4*>(size_t(bDelta1.addr));
    sbArr[1].mRestDistance = REST[1];
    DevB bSb = mk(dev, sbArr);

    // ---- the particle system: mSortedDeltaP carries the accumulated delta and the
    // particle's inverse mass in .w (H3).  Several entries have invMass 0, which the
    // prepare does NOT skip -- only the solve kernels do -- so they must still produce a
    // constraint, with velMultiplier falling back on the sbMass alone.
    std::vector<F4> sortedDeltaP;
    for (uint32_t i = 0; i < 64u; ++i)
    {
        const float t = float(int32_t(i) - 32) * 0.03125f;
        sortedDeltaP.push_back(F4{ t, -t * 0.5f, t * 0.25f,
                                   (i % 7u == 0u) ? 0.0f : (0.125f + float(i % 5u) * 0.25f) });
    }
    DevB bSdp = mk(dev, sortedDeltaP);

    std::vector<PxgParticleSystem> psArr(NB_PS);
    std::memset(psArr.data(), 0, psArr.size() * sizeof(PxgParticleSystem));
    psArr[0].mSortedDeltaP = reinterpret_cast<float4*>(size_t(bSdp.addr));
    DevB bPs = mk(dev, psArr);

    // ---- the contacts.  H4/H5: more than one warp of them, and NOT a multiple of 32.
    const uint32_t NB_CONTACTS = 77u;
    static_assert(true, "");
    std::vector<F4> contacts(NB_CONTACTS), normalPens(NB_CONTACTS), barys(NB_CONTACTS);
    std::vector<uint32_t> infos(NB_CONTACTS * 4u, 0u);
    uint32_t nonTrivialScale = 0u;
    for (uint32_t i = 0; i < NB_CONTACTS; ++i)
    {
        const uint32_t sbId = i % NB_SB;              // H7: alternate softbodies
        const uint32_t particleIdx = (i * 13u) % uint32_t(sortedDeltaP.size());

        // contacts[] is dead (Q7) but must still be a real buffer; poison it so a port
        // that USED it would produce nonsense rather than something plausible.
        contacts[i] = F4{ 1e30f, -1e30f, 1e30f, -1e30f };

        const float a = 0.125f + float(i % 9u) * 0.0625f;
        normalPens[i] = F4{ a, 1.0f - a, -0.5f + a, -0.25f + float(i % 5u) * 0.125f };

        // H1: barycentrics whose L2 norm is clearly not 1.  A vertex-like (1,0,0,0)
        // would make getSoftBodyInvMass the identity and hide it entirely.
        F4 b{ 0.25f + float(i % 3u) * 0.0625f, 0.25f, 0.25f, 0.25f };
        b.w = 1.0f - b.x - b.y - b.z;
        barys[i] = b;
        const float scale = std::sqrt(b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w);
        if (std::fabs(scale - 1.0f) > 0.05f) ++nonTrivialScale;

        // PxgFemOtherContactInfo: pairInd0 u64 @0 = (systemId << 32) | particleIndex,
        // pairInd1 u32 @8 = (softbodyId << 20) | tetId.  Word 3 is the rigid
        // material / inCollision pair this kernel must not touch.
        infos[i * 4 + 0] = particleIdx;
        infos[i * 4 + 1] = 0u;                        // particle system id
        infos[i * 4 + 2] = (sbId << BITSHIFT_TET) | 0u;
        infos[i * 4 + 3] = 0xDEADBEEFu;
    }
    PXG_CHECK_MSG(nonTrivialScale > 0u,
                  "every barycentric has L2 norm ~1, so getSoftBodyInvMass's scale (H1) "
                  "is the identity and dropping it would be invisible");
    PXG_CHECK_MSG(NB_CONTACTS % 32u != 0u,
                  "the contact count must NOT be a multiple of 32, or the partially "
                  "written tail block (H5) is never exercised");

    DevB bContacts = mk(dev, contacts), bNP = mk(dev, normalPens), bBary = mk(dev, barys);
    DevB bInfos = mk(dev, infos);
    std::vector<uint32_t> numContacts = { NB_CONTACTS };
    DevB bNum = mk(dev, numContacts);

    // ---- outputs.  The constraint array is sized for a whole extra block so the tail
    // beyond NB_CONTACTS can be poisoned and checked (H5).
    const uint32_t NB_BLOCKS = (NB_CONTACTS + 31u) / 32u + 1u;
    const uint32_t POISON_U = 0xCDCDCDCDu;
    std::vector<uint32_t> constraintsInit(NB_BLOCKS * CB_STRIDE / 4u, POISON_U);
    DevB bCons = mk(dev, constraintsInit);

    // H6: two DISTINCT buffers, poisoned with DIFFERENT patterns.  Allocated as float4
    // like the host does, while the kernel writes float2 DENSELY -- so exactly the first
    // 8 * NB_CONTACTS bytes are zeroed and the rest must survive.
    std::vector<F4> sbForcesInit(NB_CONTACTS + 8u, F4{ 11.0f, 11.0f, 11.0f, 11.0f });
    std::vector<F4> psForcesInit(NB_CONTACTS + 8u, F4{ 21.0f, 21.0f, 21.0f, 21.0f });
    DevB bSbF = mk(dev, sbForcesInit);
    DevB bPsF = mk(dev, psForcesInit);

    SbPsPrepPush p = {};
    p.softbodies = bSb.addr; p.particleSystems = bPs.addr;
    p.contacts = bContacts.addr; p.normalPens = bNP.addr; p.barycentrics = bBary.addr;
    p.contactInfos = bInfos.addr; p.numContacts = bNum.addr;
    p.constraints = bCons.addr;
    p.softBodyAppliedForces = bSbF.addr; p.particleAppliedForces = bPsF.addr;

    Kernel k("softBodyPsPrepareBda.spv", "sb_particleContactPrepareLaunch",
             {}, sizeof(SbPsPrepPush));
    k.dispatch(1024u, {}, p, 1u, 1u);
    dev->flush();

    // =====================================================================
    // the mirror
    // =====================================================================
    struct Want { F4 normalPen, bary; float velMult; };
    std::vector<Want> want(NB_CONTACTS);
    uint32_t zeroInvMass = 0u;
    for (uint32_t w = 0; w < NB_CONTACTS; ++w)
    {
        const uint32_t particleIdx = infos[w * 4 + 0];
        const uint32_t pairInd1 = infos[w * 4 + 2];
        const uint32_t sbId = pairInd1 >> BITSHIFT_TET;
        const uint32_t tetId = pairInd1 & 0x000fffffu;

        const std::vector<U4>& tets = (sbId == 0u) ? tets0 : tets1;
        const std::vector<F4>& sd = (sbId == 0u) ? simDelta0 : simDelta1;
        const U4 tet = tets[tetId];

        const F4 bary = barys[w];
        const F4 tetDelta_invMass = barycentricProject(tet, sd, bary);

        const F4 np = normalPens[w];
        const float nx = -np.x, ny = -np.y, nz = -np.z;

        const F4 dp = sortedDeltaP[particleIdx];
        const float invMass0 = dp.w;
        if (invMass0 == 0.0f) ++zeroInvMass;

        // pen = normal_pen.w - dot(particleDelta - tetDelta, normal) - restDistance,
        // left to right, exactly as softBody.cu:921 writes it.
        const float dx = dp.x - tetDelta_invMass.x;
        const float dy = dp.y - tetDelta_invMass.y;
        const float dz = dp.z - tetDelta_invMass.z;
        const float pen = np.w - (dx * nx + dy * ny + dz * nz) - REST[sbId];

        const float sbMass = getSoftBodyInvMass(tetDelta_invMass.w, bary);
        const float unitResponse = invMass0 + sbMass;
        const float velMult = (unitResponse > 0.0f) ? (1.0f / unitResponse) : 0.0f;

        want[w].normalPen = F4{ nx, ny, nz, pen };
        want[w].bary = bary;
        want[w].velMult = velMult;
    }
    PXG_CHECK_MSG(zeroInvMass > 0u,
                  "no contact has a zero-inverse-mass particle, so the velMultiplier "
                  "fallback path is never taken");

    // =====================================================================
    // compare
    // =====================================================================
    const std::vector<uint32_t> gCons =
        downloadBuffer<uint32_t>(dev, bCons.h, NB_BLOCKS * CB_STRIDE / 4u);
    auto blockWord = [&](uint32_t blk, uint32_t byteOff, uint32_t lane, uint32_t comp) {
        return gCons[(blk * CB_STRIDE + byteOff) / 4u + lane * 4u + comp];
    };
    auto blockFloat = [&](uint32_t blk, uint32_t byteOff, uint32_t lane, uint32_t comp) {
        const uint32_t u = blockWord(blk, byteOff, lane, comp);
        float f; std::memcpy(&f, &u, 4); return f;
    };
    auto sameBits = [](float a, float b) {
        uint32_t x, y; std::memcpy(&x, &a, 4); std::memcpy(&y, &b, 4); return x == y;
    };

    uint32_t bad = 0u;
    for (uint32_t w = 0; w < NB_CONTACTS; ++w)
    {
        const uint32_t blk = w / 32u, lane = w % 32u;
        const F4 gotNP{ blockFloat(blk, CB_NORMAL_PEN, lane, 0), blockFloat(blk, CB_NORMAL_PEN, lane, 1),
                        blockFloat(blk, CB_NORMAL_PEN, lane, 2), blockFloat(blk, CB_NORMAL_PEN, lane, 3) };
        const F4 gotB{ blockFloat(blk, CB_BARYCENTRIC, lane, 0), blockFloat(blk, CB_BARYCENTRIC, lane, 1),
                       blockFloat(blk, CB_BARYCENTRIC, lane, 2), blockFloat(blk, CB_BARYCENTRIC, lane, 3) };
        const uint32_t vmWord = gCons[(blk * CB_STRIDE + CB_VELMULT) / 4u + lane];
        float gotVm; std::memcpy(&gotVm, &vmWord, 4);

        const Want& e = want[w];
        const bool ok = sameBits(gotNP.x, e.normalPen.x) && sameBits(gotNP.y, e.normalPen.y) &&
                        sameBits(gotNP.z, e.normalPen.z) && sameBits(gotNP.w, e.normalPen.w) &&
                        sameBits(gotB.x, e.bary.x) && sameBits(gotB.y, e.bary.y) &&
                        sameBits(gotB.z, e.bary.z) && sameBits(gotB.w, e.bary.w) &&
                        sameBits(gotVm, e.velMult);
        if (!ok)
        {
            if (++bad <= 4u)
            {
                std::printf("  contact %u (block %u lane %u):\n", w, blk, lane);
                std::printf("    normalPen got (%a %a %a %a) want (%a %a %a %a)\n",
                            gotNP.x, gotNP.y, gotNP.z, gotNP.w,
                            e.normalPen.x, e.normalPen.y, e.normalPen.z, e.normalPen.w);
                std::printf("    bary      got (%a %a %a %a) want (%a %a %a %a)\n",
                            gotB.x, gotB.y, gotB.z, gotB.w,
                            e.bary.x, e.bary.y, e.bary.z, e.bary.w);
                std::printf("    velMult   got %a want %a\n", gotVm, e.velMult);
            }
        }
    }
    PXG_CHECK_MSG(bad == 0u, "%u of %u constraints disagree with the mirror", bad, NB_CONTACTS);

    // H5: the lanes past the contact count, and the whole spare block, are untouched.
    uint32_t tailTouched = 0u;
    for (uint32_t w = NB_CONTACTS; w < NB_BLOCKS * 32u; ++w)
    {
        const uint32_t blk = w / 32u, lane = w % 32u;
        for (uint32_t c = 0; c < 4u; ++c)
            if (blockWord(blk, CB_NORMAL_PEN, lane, c) != POISON_U ||
                blockWord(blk, CB_BARYCENTRIC, lane, c) != POISON_U)
                ++tailTouched;
        if (gCons[(blk * CB_STRIDE + CB_VELMULT) / 4u + lane] != POISON_U) ++tailTouched;
    }
    PXG_CHECK_MSG(tailTouched == 0u,
                  "%u words past contact %u were written; the kernel returns before the "
                  "three stores for those lanes (Q11)", tailTouched, NB_CONTACTS);

    // H6: both force buffers zeroed at stride 8, upper halves intact, and the two are
    // distinct allocations.
    const std::vector<float> gSbF = downloadBuffer<float>(dev, bSbF.h, (NB_CONTACTS + 8u) * 4u);
    const std::vector<float> gPsF = downloadBuffer<float>(dev, bPsF.h, (NB_CONTACTS + 8u) * 4u);
    const uint32_t zeroedFloats = NB_CONTACTS * 2u;      // float2 per contact, dense
    uint32_t forceBad = 0u;
    for (uint32_t i = 0; i < zeroedFloats; ++i)
    {
        if (gSbF[i] != 0.0f) ++forceBad;
        if (gPsF[i] != 0.0f) ++forceBad;
    }
    for (uint32_t i = zeroedFloats; i < (NB_CONTACTS + 8u) * 4u; ++i)
    {
        if (gSbF[i] != 11.0f) ++forceBad;
        if (gPsF[i] != 21.0f) ++forceBad;
    }
    PXG_CHECK_MSG(forceBad == 0u,
                  "%u applied-force checks failed: each buffer must be zeroed for exactly "
                  "its first %u floats (one float2 per contact, DENSE at stride 8) and "
                  "untouched after -- the float4-sized allocation is simply twice as "
                  "large, not half-used per slot (H6/Q10)",
                  forceBad, zeroedFloats);

    // the contact-info word this kernel must not touch
    const std::vector<uint32_t> gInfos = downloadBuffer<uint32_t>(dev, bInfos.h, NB_CONTACTS * 4u);
    uint32_t infoBad = 0u;
    for (uint32_t i = 0; i < NB_CONTACTS; ++i)
        if (gInfos[i * 4 + 3] != 0xDEADBEEFu) ++infoBad;
    PXG_CHECK_MSG(infoBad == 0u,
                  "%u contact-info slots had their fourth word changed; this kernel only "
                  "READS PxgFemOtherContactInfo", infoBad);

    std::printf("    [ note ] %u contacts over %u blocks (%u lanes of tail poison intact), "
                "2 softbodies with rest %g/%g, %u zero-inverse-mass particles, %u "
                "barycentrics with a non-unit L2 scale\n",
                NB_CONTACTS, NB_BLOCKS, NB_BLOCKS * 32u - NB_CONTACTS,
                double(REST[0]), double(REST[1]), zeroInvMass, nonTrivialScale);
}
} // namespace
