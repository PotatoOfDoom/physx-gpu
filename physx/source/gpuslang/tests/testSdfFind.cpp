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

// testSdfFind.cpp -- P9.12e, the fixture for pxgSlangSdfFind.
//
// ---------------------------------------------------------------------------
// WHAT IS UNDER TEST, AND WHAT IS NOT
// ---------------------------------------------------------------------------
// findInterestingVertices is a RESUMABLE STREAM, not a filter: it sweeps the
// particle array, keeps at most 1024 hits, and records where to continue.  The
// kernel calls it in a loop until it reports "done".  So the claims are
//
//   (1) the block-wide exclusive scan over 32 emulated warps is right,
//   (2) the buffer holds exactly the right indices, in the right ORDER,
//   (3) the count saturates at TargetCount,
//   (4) the resume index is one PAST the vertex that filled the buffer, and
//   (5) "offset unchanged" is what terminates the loop.
//
// Only (1) is about arithmetic; (2)-(5) are a protocol, and a fixture that
// called the function once would test none of them.  This one drives the whole
// loop and checks every round.
//
// Sample() still cannot run on the host, so the same arrangement as P9.12d: the
// shader reports the cull distance it computed per vertex, and the expected
// gather is built from THOSE numbers.  What is compared bit-for-bit against the
// engine is the part that is host-computable -- the vertex-space point, through
// PxTransform::transform and the PxQuat rotate/rotateInv pair that
// shape2Vertex is made of.  The distance itself is P9.12d's claim, measured
// there.
//
// ---------------------------------------------------------------------------
// FIVE CONFIGURATIONS, and each exists for a boundary
// ---------------------------------------------------------------------------
// The distances do not depend on the tolerance, so ONE dispatch with an
// infinite tolerance produces the whole table, and the other four tolerances
// are then chosen FROM that table to land exactly on the boundaries:
//
//   A  tolerance = PX_MAX_F32     every vertex qualifies -> ceil(N/1024) full
//                                 rounds, the last one short.  Also the run
//                                 that produces the distance table.
//   B  the 1st smallest distance  ZERO qualify (the test is strict `<`) -> one
//                                 round, count 0, and the loop must still
//                                 terminate.
//   C  the 1024th smallest        exactly 1023 qualify -> one round, the buffer
//                                 one short of full, no resume point written.
//   D  the 1025th smallest        exactly 1024 qualify -> the buffer fills
//                                 EXACTLY, so the resume point is written by
//                                 the thread at slot TargetCount-1 and the
//                                 second round finds nothing.  This is the only
//                                 configuration that separates "wrote the
//                                 resume point" from "did not need to".
//   E  the 2600th smallest        2599 qualify -> three rounds, two of them
//                                 saturated.
//
// Every count above is ASSERTED against the table rather than assumed, so a tie
// in the distances would be reported instead of quietly weakening a case.

#include "pxgTestFramework.h"

#include "PxgCudaShim.h"
#include "PxgCudaToolkitShim.h"
#include "pxgSdfSceneRef.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace physx { namespace gpuslang {
void pxgSetVulkanDriverShim(PxgCudaShim* shim);
} }

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using namespace physx;
using namespace pxgsdfref;

// npSdfFind.slang's push-constant block -- 176 bytes (128 until P9.34b/P9.35b), laid
// out to match.  P9.98 removed three stray shader members that had made it 188.
struct P_Find
{
    uint64_t meshPtr, positions, echoOut, roundsOut, bufOut, numRoundsOut;   //  0.. 47
    float aToBq[4];                                                          // 48
    float scaleRot[4];                                                       // 64
    float aToBp[3];  float tolerance;                                        // 80
    float scale[3];  uint32_t numVertices;                                   // 96
    uint32_t maxRounds, pad0, pad1, pad2;                                    // 112
    uint64_t meshInd;                                                        // 128  P9.34b
    uint64_t touchedOut;                                                     // 136  P9.35b
    float scale0Rot[4];                                                      // 144
    float scale0[3]; uint32_t pad172;                                        // 160 -> 176
};
static_assert(sizeof(P_Find) == 176, "the find blob must be 176 bytes");
static_assert(offsetof(P_Find, meshInd) == 128, "meshInd @128");
static_assert(offsetof(P_Find, scale0Rot) == 144, "scale0Rot @144");
static_assert(offsetof(P_Find, scale0) == 160, "scale0 @160");

static const uint32_t TARGET_COUNT = 1024u;    // SDFF_TARGET_COUNT
static const uint32_t BLOCK_DIM = 1024u;       // SDFF_BLOCK_DIM
static const uint32_t MAX_ROUNDS = 16u;
static const uint32_t NUM_VERTS = 4000u;

static bool bitEq(float a, float b) { return f2u(a) == f2u(b); }

// shape2Vertex, cudaNpCommon.h:284-296 -- rotate, per-component DIVIDE, rotateInv.
// PxQuat::rotate and rotateInv are the engine's own and are CALLED.
static PxVec3 hShape2Vertex(const PxVec3& v, const PxVec3& scale, const PxQuat& rotation)
{
    PxVec3 temp = rotation.rotate(v);        // ENGINE
    temp.x /= scale.x;
    temp.y /= scale.y;
    temp.z /= scale.z;
    return rotation.rotateInv(temp);         // ENGINE
}

// The inverse of the chain, used only to place the test points where they will
// produce a useful spread of distances.
static PxVec3 hVertex2Shape(const PxVec3& v, const PxVec3& scale, const PxQuat& rotation)
{
    PxVec3 temp = rotation.rotate(v);
    temp.x *= scale.x;
    temp.y *= scale.y;
    temp.z *= scale.z;
    return rotation.rotateInv(temp);
}

// ---------------------------------------------------------------------------
// The host mirror of one call to findInterestingVertices, and then of the
// caller's loop.  Control flow from sdfCollision.cuh:690-726 and
// particleSystemMeshMidphase.cu:478; nothing here needs the SDF, only the
// distances the shader reported.
// ---------------------------------------------------------------------------
struct Round
{
    uint32_t count = 0, before = 0, after = 0;
    std::vector<uint32_t> buf;
};

static std::vector<Round> mirrorRounds(const std::vector<float>& dist, float tol, uint32_t N)
{
    std::vector<Round> rounds;
    uint32_t startIndex = 0;
    while (startIndex < N && rounds.size() < MAX_ROUNDS)
    {
        Round r;
        r.before = startIndex;
        r.buf.assign(TARGET_COUNT, 0xFFFFFFFFu);

        uint32_t offset = startIndex;          // the shared resume slot, seeded
        uint32_t nbElementsInBuffer = 0;       // sharedIndicesCount

        for (uint32_t i = startIndex; i < N; i += BLOCK_DIM)
        {
            // The block's ballot, in thread order -- which is ascending `ind`.
            uint32_t total = 0;
            for (uint32_t k = 0; k < BLOCK_DIM; ++k)
            {
                const uint32_t ind = i + k;
                if (ind < N && dist[ind] < tol)
                    ++total;
            }
            uint32_t rank = 0;
            for (uint32_t k = 0; k < BLOCK_DIM; ++k)
            {
                const uint32_t ind = i + k;
                if (!(ind < N && dist[ind] < tol))
                    continue;
                const uint32_t writeOffset = nbElementsInBuffer + rank;
                ++rank;
                if (writeOffset < TARGET_COUNT)
                    r.buf[writeOffset] = ind;
                if (writeOffset == TARGET_COUNT - 1u)
                    offset = ind + 1u;         // S4
            }
            nbElementsInBuffer = std::min(total + nbElementsInBuffer, TARGET_COUNT);   // S5
        }

        r.count = nbElementsInBuffer;
        r.after = (startIndex != offset) ? offset : N;                                  // S8
        startIndex = r.after;
        rounds.push_back(r);
    }
    return rounds;
}

PXG_TEST(sdfFind)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }

    const PxgKernelDef manifest[] = { { "npSdfFind.spv", "sdfFindRounds", 0u, 0u } };
    static auto loader = [](const char* f) { return loadShaderBlob(f); };
    PxgCudaShim shim(dev, manifest, 1, +loader);
    physx::gpuslang::pxgSetVulkanDriverShim(&shim);

    std::vector<PxgBufferHandle> owned;
    auto keep = [&](PxgBufferHandle h) { owned.push_back(h); return h; };

    // ---- the SDF, and the P9.12c blob counts for the P9.12c reason ----
    SdfScene sp;
    buildSparse(sp);
    BlobLayout blob = layOutBlob(sp, 16u, 21u, 3u, 1u, 0u);
    const uint64_t hCoarse = sdfUploadTex(CU_AD_FORMAT_FLOAT, sp.nbx + 1, sp.nby + 1, sp.nbz + 1,
                                          sp.coarse.data(), sizeof(float));
    const uint64_t hSub = sdfUploadTex(CU_AD_FORMAT_UNSIGNED_INT8, sp.blkx * (sp.sg + 1),
                                       sp.blky * (sp.sg + 1), sp.blkz * (sp.sg + 1),
                                       sp.sub.data(), 1);
    PXG_CHECK_MSG(hCoarse != 0 && hSub != 0, "the SDF textures must upload");
    if (hCoarse == 0 || hSub == 0) return;
    std::memcpy(&blob.bytes[blob.texObjOffset], &hCoarse, 8);
    std::memcpy(&blob.bytes[blob.texObjSparseOffset], &hSub, 8);
    const uint64_t blobAddr = dev->getBufferDeviceAddress(keep(createBufferFrom(dev, blob.bytes)));

    // ---- the transforms.  Nothing here is identity or axis-aligned: a wrong
    // rotation order, a wrong inverse or a swapped scale component all move the
    // vertex-space point, which is compared bit-for-bit. ----
    const PxQuat aToBq = PxQuat(0.37f, PxVec3(0.4f, -0.8f, 0.44721f).getNormalized());
    const PxVec3 aToBp(0.21f, -0.63f, 1.07f);
    const PxTransform aToB(aToBp, aToBq);
    const PxVec3 scale(1.35f, 0.72f, 2.10f);                    // three distinct, none 1
    const PxQuat scaleRot = PxQuat(-0.58f, PxVec3(-0.6f, 0.48f, 0.64031f).getNormalized());

    // ---- the particles.  Generated in VERTEX space over an expanded box and
    // mapped back through the inverse chain, so the distances span the SDF's
    // interior, its faces and a margin outside it rather than clustering.
    //
    // THE LAST TAIL_FAR OF THEM ARE PLACED FAR OUTSIDE, and that is not
    // decoration.  Config D's whole job is the exactly-full buffer, and it only
    // separates "wrote the resume point" from "did not need to" if a SECOND
    // round actually runs -- which needs the highest qualifying index to be
    // below N-1.  With a uniform cloud the 1024 nearest vertices are scattered
    // over the whole array and the last of them lands on N-1 about a quarter of
    // the time; the first draft of this fixture drew exactly that and config D
    // collapsed to a single round, where an unwritten resume point would have
    // terminated the loop just as well.  A tail that can never qualify at any
    // of the four finite tolerances bounds the qualifying set away from the end
    // and makes the second round mandatory. ----
    const HostSdfBox box = hostSdfBox(sp);
    const PxVec3 ext = box.higher - box.lower;
    const uint32_t TAIL_FAR = 200u;
    std::vector<F4> positions(NUM_VERTS);
    {
        std::mt19937 rng(0x5DF9E12u);
        auto u01 = [&]() { return float(rng() % 1000000u) / 1000000.f; };
        const PxTransform bToA = aToB.getInverse();
        for (uint32_t i = 0; i < NUM_VERTS; ++i)
        {
            PxVec3 vtx;
            if (i < NUM_VERTS - TAIL_FAR)
                vtx = PxVec3(box.lower.x - 0.4f * ext.x + 1.8f * ext.x * u01(),
                             box.lower.y - 0.4f * ext.y + 1.8f * ext.y * u01(),
                             box.lower.z - 0.4f * ext.z + 1.8f * ext.z * u01());
            else
                vtx = PxVec3(box.lower.x - (8.0f + 3.0f * u01()) * ext.x,
                             box.lower.y - (8.0f + 3.0f * u01()) * ext.y,
                             box.lower.z - (8.0f + 3.0f * u01()) * ext.z);
            const PxVec3 shapeSpace = hVertex2Shape(vtx, scale, scaleRot);
            const PxVec3 world = bToA.transform(shapeSpace);
            positions[i] = F4{ world.x, world.y, world.z, 0.25f + 0.001f * float(i) };
        }
    }
    const uint64_t posAddr = dev->getBufferDeviceAddress(keep(createBufferFrom(dev, positions)));

    // ---- output buffers, poisoned ----
    std::vector<F4> echoInit(NUM_VERTS, F4{ -777.f, -777.f, -777.f, -777.f });
    std::vector<uint32_t> roundsInit(MAX_ROUNDS * 4u, 0xEEEEEEEEu);
    std::vector<uint32_t> bufInit(size_t(MAX_ROUNDS) * TARGET_COUNT, 0xEEEEEEEEu);
    std::vector<uint32_t> nrInit(1u, 0xEEEEEEEEu);

    P_Find base = {};
    base.meshPtr = blobAddr;
    base.positions = posAddr;
    base.aToBq[0] = aToBq.x; base.aToBq[1] = aToBq.y; base.aToBq[2] = aToBq.z; base.aToBq[3] = aToBq.w;
    base.scaleRot[0] = scaleRot.x; base.scaleRot[1] = scaleRot.y;
    base.scaleRot[2] = scaleRot.z; base.scaleRot[3] = scaleRot.w;
    base.aToBp[0] = aToBp.x; base.aToBp[1] = aToBp.y; base.aToBp[2] = aToBp.z;
    base.scale[0] = scale.x; base.scale[1] = scale.y; base.scale[2] = scale.z;
    base.numVertices = NUM_VERTS;
    base.maxRounds = MAX_ROUNDS;

    // One dispatch, returning everything the round-by-round check needs.
    struct Result
    {
        uint32_t numRounds = 0;
        std::vector<uint32_t> rounds;   // MAX_ROUNDS * 4
        std::vector<uint32_t> buf;      // MAX_ROUNDS * TARGET_COUNT
        std::vector<F4> echo;           // NUM_VERTS
    };
    auto run = [&](float tol) -> Result {
        Result out;
        const PxgBufferHandle bE = keep(createBufferFrom(dev, echoInit));
        const PxgBufferHandle bR = keep(createBufferFrom(dev, roundsInit));
        const PxgBufferHandle bB = keep(createBufferFrom(dev, bufInit));
        const PxgBufferHandle bN = keep(createBufferFrom(dev, nrInit));
        P_Find p = base;
        p.echoOut = dev->getBufferDeviceAddress(bE);
        p.roundsOut = dev->getBufferDeviceAddress(bR);
        p.bufOut = dev->getBufferDeviceAddress(bB);
        p.numRoundsOut = dev->getBufferDeviceAddress(bN);
        p.tolerance = tol;
        {
            Kernel k(Kernel::Bindless{}, "npSdfFind.spv", "sdfFindRounds", {}, sizeof(P_Find));
            PXG_CHECK_MSG(k.isValid(), "bindless pipeline creation succeeds for sdfFindRounds");
            if (!k.isValid()) return out;
            k.dispatch(1u, {}, p);            // ONE workgroup of 1024, as the kernel launches
        }
        dev->flush();
        out.numRounds = downloadBuffer<uint32_t>(dev, bN, 1)[0];
        out.rounds = downloadBuffer<uint32_t>(dev, bR, MAX_ROUNDS * 4u);
        out.buf = downloadBuffer<uint32_t>(dev, bB, size_t(MAX_ROUNDS) * TARGET_COUNT);
        out.echo = downloadBuffer<F4>(dev, bE, NUM_VERTS);
        return out;
    };

    // =====================================================================
    // Config A: an infinite tolerance.  Every vertex qualifies, and this is
    // also the run that produces the distance table the other four are built
    // from.
    // =====================================================================
    const Result A = run(PX_MAX_F32);
    if (A.numRounds == 0xEEEEEEEEu || A.numRounds == 0xFFFFFFFFu)
    {
        PXG_CHECK_MSG(false, "config A produced no round count (0x%08x) -- the mesh reported no SDF "
                      "or the dispatch never ran", A.numRounds);
        return;
    }

    // ---- the vertex-space point, bit-for-bit against the engine's own
    // PxTransform / PxQuat.  This is the only part of the per-vertex work that
    // is host-computable, and it is the part a wrong transform order lives in.
    {
        size_t badPt = 0, unseen = 0;
        for (uint32_t i = 0; i < NUM_VERTS; ++i)
        {
            const PxVec3 world(positions[i].x, positions[i].y, positions[i].z);
            const PxVec3 want = hShape2Vertex(aToB.transform(world), scale, scaleRot);   // ENGINE
            const F4& g = A.echo[i];
            if (g.w == -777.f && g.x == -777.f)
            {
                ++unseen;
                continue;
            }
            if (!bitEq(g.x, want.x) || !bitEq(g.y, want.y) || !bitEq(g.z, want.z))
            {
                if (badPt < 4)
                    std::printf("    vertex %u: point (%.9g,%.9g,%.9g) want (%.9g,%.9g,%.9g)\n",
                                i, g.x, g.y, g.z, want.x, want.y, want.z);
                ++badPt;
            }
        }
        PXG_CHECK_MSG(unseen == 0, "%zu of %u vertices were never visited by the sweep",
                      unseen, NUM_VERTS);
        PXG_CHECK_MSG(badPt == 0, "%zu vertex-space points differ from "
                      "shape2Vertex(aToB.transform(p)) computed with the engine's own PxQuat",
                      badPt);
    }

    std::vector<float> dist(NUM_VERTS);
    for (uint32_t i = 0; i < NUM_VERTS; ++i)
        dist[i] = A.echo[i].w;

    // ---- the round-by-round check, shared by all five configurations ----
    size_t configsChecked = 0;
    auto check = [&](const char* label, const Result& R, float tol) {
        const std::vector<Round> want = mirrorRounds(dist, tol, NUM_VERTS);
        PXG_CHECK_MSG(R.numRounds == uint32_t(want.size()),
                      "%s: %u rounds, expected %zu", label, R.numRounds, want.size());
        PXG_CHECK_MSG(R.numRounds < MAX_ROUNDS,
                      "%s: the round loop hit its safety bound (%u) -- startIndex is not advancing",
                      label, R.numRounds);
        if (R.numRounds != uint32_t(want.size()) || R.numRounds >= MAX_ROUNDS)
            return;

        size_t badHdr = 0, badBuf = 0, badCanary = 0;
        for (uint32_t r = 0; r < R.numRounds; ++r)
        {
            const uint32_t* h = &R.rounds[size_t(r) * 4];
            if (h[0] != want[r].count || h[1] != want[r].before || h[2] != want[r].after)
            {
                if (badHdr < 4)
                    std::printf("    %s round %u: count/before/after = %u/%u/%u, "
                                "expected %u/%u/%u\n", label, r, h[0], h[1], h[2],
                                want[r].count, want[r].before, want[r].after);
                ++badHdr;
            }
            if (h[3] != 0xC0FFEEu)
                ++badCanary;

            for (uint32_t k = 0; k < want[r].count; ++k)
            {
                const uint32_t got = R.buf[size_t(r) * TARGET_COUNT + k];
                if (got != want[r].buf[k])
                {
                    if (badBuf < 6)
                        std::printf("    %s round %u slot %u: %u, expected %u\n",
                                    label, r, k, got, want[r].buf[k]);
                    ++badBuf;
                }
            }
            // Nothing past the reported count may have been written -- an
            // overrun of the shared buffer would land here.
            for (uint32_t k = want[r].count; k < TARGET_COUNT; ++k)
                if (R.buf[size_t(r) * TARGET_COUNT + k] != 0xEEEEEEEEu)
                {
                    if (badBuf < 6)
                        std::printf("    %s round %u slot %u past count %u was written (%u)\n",
                                    label, r, k, want[r].count,
                                    R.buf[size_t(r) * TARGET_COUNT + k]);
                    ++badBuf;
                }
        }
        // Rounds that never ran must still be poison.
        for (uint32_t r = R.numRounds; r < MAX_ROUNDS; ++r)
            for (uint32_t k = 0; k < 4; ++k)
                if (R.rounds[size_t(r) * 4 + k] != 0xEEEEEEEEu)
                    ++badHdr;

        PXG_CHECK_MSG(badHdr == 0, "%s: %zu round headers wrong", label, badHdr);
        PXG_CHECK_MSG(badBuf == 0, "%s: %zu buffer slots wrong", label, badBuf);
        PXG_CHECK_MSG(badCanary == 0, "%s: %zu round headers lost their canary", label, badCanary);

        // The echoed distances must not depend on the tolerance.
        size_t badEcho = 0;
        for (uint32_t i = 0; i < NUM_VERTS; ++i)
        {
            if (R.echo[i].w == -777.f && R.echo[i].x == -777.f)
                continue;                     // not reached in this configuration
            if (!bitEq(R.echo[i].w, dist[i]))
                ++badEcho;
        }
        PXG_CHECK_MSG(badEcho == 0, "%s: %zu echoed distances differ from config A's -- the cull "
                      "distance must not depend on the tolerance", label, badEcho);
        ++configsChecked;
    };

    const uint32_t roundsA = uint32_t(mirrorRounds(dist, PX_MAX_F32, NUM_VERTS).size());
    PXG_CHECK_MSG(roundsA == (NUM_VERTS + TARGET_COUNT - 1u) / TARGET_COUNT,
                  "config A must need exactly ceil(N/TargetCount) rounds (%u)", roundsA);
    check("A(all)", A, PX_MAX_F32);

    // ---- the four boundary tolerances, taken from the table ----
    std::vector<float> sorted(dist);
    std::sort(sorted.begin(), sorted.end());

    auto emitCount = [&](float tol) {
        size_t n = 0;
        for (float d : dist) if (d < tol) ++n;
        return n;
    };

    struct Cfg { const char* label; float tol; size_t wantEmit; };
    const Cfg cfgs[4] = {
        { "B(none)",    sorted[0],    0u },
        { "C(1023)",    sorted[1023], 1023u },
        { "D(exact-1024)", sorted[1024], 1024u },
        { "E(2599)",    sorted[2599], 2599u },
    };
    for (const Cfg& c : cfgs)
    {
        const size_t n = emitCount(c.tol);
        PXG_CHECK_MSG(n == c.wantEmit,
                      "%s: the tolerance %.9g selects %zu vertices, not %zu -- a tie in the "
                      "distance table has weakened this case", c.label, c.tol, n, c.wantEmit);
        const Result R = run(c.tol);
        check(c.label, R, c.tol);
    }

    // The two boundaries this fixture exists for, asserted rather than hoped:
    // D must fill the buffer EXACTLY and therefore need a second, empty round;
    // C must stop one short and never write a resume point.
    {
        const std::vector<Round> wD = mirrorRounds(dist, sorted[1024], NUM_VERTS);
        const std::vector<Round> wC = mirrorRounds(dist, sorted[1023], NUM_VERTS);
        PXG_CHECK_MSG(wD.size() == 2 && wD[0].count == TARGET_COUNT && wD[1].count == 0,
                      "config D must be exactly two rounds of %u and 0 (got %zu rounds)",
                      TARGET_COUNT, wD.size());
        PXG_CHECK_MSG(wC.size() == 1 && wC[0].count == TARGET_COUNT - 1u,
                      "config C must be one round of %u (got %zu rounds, first count %u)",
                      TARGET_COUNT - 1u, wC.size(), wC.empty() ? 0u : wC[0].count);
        const std::vector<Round> wE = mirrorRounds(dist, sorted[2599], NUM_VERTS);
        PXG_CHECK_MSG(wE.size() == 3, "config E must need three rounds (got %zu)", wE.size());
    }

    PXG_CHECK_MSG(configsChecked == 5, "all five configurations must be checked (%zu were)",
                  configsChecked);
    std::printf("    [ note ] %u vertices, %u rounds at an infinite tolerance; boundaries at "
                "0 / %u / %u / 2599 qualifying vertices all checked round by round\n",
                NUM_VERTS, A.numRounds, TARGET_COUNT - 1u, TARGET_COUNT);

    for (PxgBufferHandle h : owned)
        dev->destroyBuffer(h);
    physx::gpuslang::pxgSetVulkanDriverShim(nullptr);
}


// ===========================================================================
// P9.34b: findInterestingTrianglesA<32, 1024>, the TRIANGLE sibling of the gather
// above and the outer loop of cloth_sdfMeshContactGenLaunch.
//
// Same resumable-stream protocol, checked the same way -- the expectation is built
// from the distances the function ITSELF reports through the echo, because Sample()
// is GPU-only, so what is under test is the gathering and the cull predicate rather
// than the distance (P9.12d's claim, measured there).
//
// Three things are new against the vertex form and each has its own check:
//   * the buffer element is a uint2, whose second half must be a literal 0;
//   * the cull test is a BOUNDING SPHERE -- centroid distance against
//     radius/min(scale) + tolerance -- and the radius is host-computable exactly,
//     so it is compared bit for bit rather than taken from the shader;
//   * the loop carries a count guard the vertex form lacks.  That guard cannot
//     change the gathered set (once the buffer saturates, no later write can
//     land and the resume handshake cannot fire), so the ONLY thing that can
//     observe it is which triangles were touched -- which the echo records.
// ===========================================================================
PXG_TEST(sdfFindTriangles)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }

    const PxgKernelDef manifest[] = { { "npSdfFind.spv", "sdfFindTriangleRounds", 0u, 0u } };
    static auto loaderT = [](const char* f) { return loadShaderBlob(f); };
    PxgCudaShim shim(dev, manifest, 1, +loaderT);
    physx::gpuslang::pxgSetVulkanDriverShim(&shim);

    std::vector<PxgBufferHandle> owned;
    auto keep = [&](PxgBufferHandle h) { owned.push_back(h); return h; };

    SdfScene sp;
    buildSparse(sp);
    BlobLayout blob = layOutBlob(sp, 16u, 21u, 3u, 1u, 0u);
    const uint64_t hCoarse = sdfUploadTex(CU_AD_FORMAT_FLOAT, sp.nbx + 1, sp.nby + 1, sp.nbz + 1,
                                          sp.coarse.data(), sizeof(float));
    const uint64_t hSub = sdfUploadTex(CU_AD_FORMAT_UNSIGNED_INT8, sp.blkx * (sp.sg + 1),
                                       sp.blky * (sp.sg + 1), sp.blkz * (sp.sg + 1),
                                       sp.sub.data(), 1);
    PXG_CHECK_MSG(hCoarse != 0 && hSub != 0, "the SDF textures must upload");
    if (hCoarse == 0 || hSub == 0) return;
    std::memcpy(&blob.bytes[blob.texObjOffset], &hCoarse, 8);
    std::memcpy(&blob.bytes[blob.texObjSparseOffset], &hSub, 8);
    const uint64_t blobAddr = dev->getBufferDeviceAddress(keep(createBufferFrom(dev, blob.bytes)));

    // Nothing identity or axis-aligned, on EITHER scale: the cloth's own scale0 and
    // the SDF mesh's scale1 are different non-uniform frames, so swapping them, or
    // using shape2Vertex where vertex2Shape belongs, moves the centroid.
    const PxQuat aToBq = PxQuat(0.37f, PxVec3(0.4f, -0.8f, 0.44721f).getNormalized());
    const PxVec3 aToBp(0.21f, -0.63f, 1.07f);
    const PxTransform aToB(aToBp, aToBq);
    const PxVec3 scale1(1.35f, 0.72f, 2.10f);
    const PxQuat scale1Rot = PxQuat(-0.58f, PxVec3(-0.6f, 0.48f, 0.64031f).getNormalized());
    const PxVec3 scale0(0.83f, 1.44f, 0.61f);
    const PxQuat scale0Rot = PxQuat(0.29f, PxVec3(0.31235f, 0.6247f, -0.71651f).getNormalized());

    const uint32_t NUM_TRIS = 4000u;
    const uint32_t NUM_V = NUM_TRIS * 3u;
    const HostSdfBox box = hostSdfBox(sp);
    const PxVec3 ext = box.higher - box.lower;
    const uint32_t TAIL_FAR = 200u;   // the same device as the vertex fixture, same reason

    std::vector<F4> verts(NUM_V);
    std::vector<uint32_t> inds(NUM_TRIS * 4u);
    {
        std::mt19937 rng(0x71A9C3u);
        auto u01 = [&]() { return float(rng() % 1000000u) / 1000000.f; };
        const PxTransform bToA = aToB.getInverse();
        for (uint32_t t = 0; t < NUM_TRIS; ++t)
        {
            // The triangles vary in SIZE as well as position, because the cull test
            // divides the bounding-sphere radius by the smallest scale component and
            // adds it to the tolerance -- a rig of uniform triangles cannot tell that
            // term from a constant.
            const float sz = (t % 7u == 0u) ? 0.9f * ext.x : 0.02f * ext.x * (1.0f + float(t % 5));
            PxVec3 c;
            if (t < NUM_TRIS - TAIL_FAR)
                c = PxVec3(box.lower.x - 0.4f * ext.x + 1.8f * ext.x * u01(),
                           box.lower.y - 0.4f * ext.y + 1.8f * ext.y * u01(),
                           box.lower.z - 0.4f * ext.z + 1.8f * ext.z * u01());
            else
                c = PxVec3(box.lower.x - (8.0f + 3.0f * u01()) * ext.x,
                           box.lower.y - (8.0f + 3.0f * u01()) * ext.y,
                           box.lower.z - (8.0f + 3.0f * u01()) * ext.z);
            const PxVec3 corner[3] = { c + PxVec3(sz, 0.0f, 0.0f),
                                       c + PxVec3(-0.3f * sz, 0.9f * sz, 0.2f * sz),
                                       c + PxVec3(0.1f * sz, -0.4f * sz, -0.8f * sz) };
            for (uint32_t k = 0; k < 3u; ++k)
            {
                // Generated in VERTEX space and mapped back through the inverse chain,
                // exactly as the vertex fixture does.
                const PxVec3 shapeSpace = hVertex2Shape(corner[k], scale0, scale0Rot);
                const PxVec3 world = bToA.transform(shapeSpace);
                verts[t * 3u + k] = F4{ world.x, world.y, world.z, 0.0f };
            }
            inds[t * 4u + 0] = t * 3u; inds[t * 4u + 1] = t * 3u + 1u;
            inds[t * 4u + 2] = t * 3u + 2u; inds[t * 4u + 3] = 0u;
        }
    }
    const uint64_t vAddr = dev->getBufferDeviceAddress(keep(createBufferFrom(dev, verts)));
    const uint64_t iAddr = dev->getBufferDeviceAddress(keep(createBufferFrom(dev, inds)));

    std::vector<F4> echoInit(NUM_TRIS, F4{ -777.f, -777.f, -777.f, -777.f });
    std::vector<uint32_t> roundsInit(MAX_ROUNDS * 4u, 0xEEEEEEEEu);
    std::vector<uint32_t> bufInit(size_t(MAX_ROUNDS) * TARGET_COUNT * 2u, 0xEEEEEEEEu);
    std::vector<uint32_t> nrInit(1u, 0xEEEEEEEEu);

    P_Find base = {};
    base.meshPtr = blobAddr;
    base.positions = vAddr;
    base.meshInd = iAddr;
    base.aToBq[0] = aToBq.x; base.aToBq[1] = aToBq.y; base.aToBq[2] = aToBq.z; base.aToBq[3] = aToBq.w;
    base.scaleRot[0] = scale1Rot.x; base.scaleRot[1] = scale1Rot.y;
    base.scaleRot[2] = scale1Rot.z; base.scaleRot[3] = scale1Rot.w;
    base.scale0Rot[0] = scale0Rot.x; base.scale0Rot[1] = scale0Rot.y;
    base.scale0Rot[2] = scale0Rot.z; base.scale0Rot[3] = scale0Rot.w;
    base.aToBp[0] = aToBp.x; base.aToBp[1] = aToBp.y; base.aToBp[2] = aToBp.z;
    base.scale[0] = scale1.x; base.scale[1] = scale1.y; base.scale[2] = scale1.z;
    base.scale0[0] = scale0.x; base.scale0[1] = scale0.y; base.scale0[2] = scale0.z;
    base.numVertices = NUM_TRIS;
    base.maxRounds = MAX_ROUNDS;

    struct Result
    {
        uint32_t numRounds = 0;
        std::vector<uint32_t> rounds, buf;
        std::vector<F4> echo;
    };
    auto run = [&](float tol, uint32_t maxRounds) -> Result {
        Result out;
        const PxgBufferHandle bE = keep(createBufferFrom(dev, echoInit));
        const PxgBufferHandle bR = keep(createBufferFrom(dev, roundsInit));
        const PxgBufferHandle bB = keep(createBufferFrom(dev, bufInit));
        const PxgBufferHandle bN = keep(createBufferFrom(dev, nrInit));
        P_Find p = base;
        p.echoOut = dev->getBufferDeviceAddress(bE);
        p.roundsOut = dev->getBufferDeviceAddress(bR);
        p.bufOut = dev->getBufferDeviceAddress(bB);
        p.numRoundsOut = dev->getBufferDeviceAddress(bN);
        p.tolerance = tol;
        p.maxRounds = maxRounds;
        Kernel k(Kernel::Bindless{}, "npSdfFind.spv", "sdfFindTriangleRounds", {}, sizeof(P_Find));
        PXG_CHECK_MSG(k.isValid(), "bindless pipeline creation succeeds for sdfFindTriangleRounds");
        if (!k.isValid()) return out;
        k.dispatch(1u, {}, p);
        dev->flush();
        out.numRounds = downloadBuffer<uint32_t>(dev, bN, 1)[0];
        out.rounds = downloadBuffer<uint32_t>(dev, bR, MAX_ROUNDS * 4u);
        out.buf = downloadBuffer<uint32_t>(dev, bB, size_t(MAX_ROUNDS) * TARGET_COUNT * 2u);
        out.echo = downloadBuffer<F4>(dev, bE, NUM_TRIS);
        return out;
    };

    // The host's own bounding-sphere radius, from the ENGINE's transforms.  Only the
    // SDF distance comes from the device.
    auto hostRadiusOverScale = [&](uint32_t t, PxVec3& centroidVertexSpace) {
        PxVec3 v[3];
        for (uint32_t k = 0; k < 3u; ++k)
        {
            const F4& w = verts[t * 3u + k];
            v[k] = aToB.transform(hVertex2Shape(PxVec3(w.x, w.y, w.z), scale0, scale0Rot));
        }
        const float third = 1.0f / 3.0f;
        const PxVec3 c = (v[0] + v[1] + v[2]) * third;
        centroidVertexSpace = hShape2Vertex(c, scale1, scale1Rot);
        float r = std::sqrt(PxMax((v[0] - c).magnitudeSquared(),
                             PxMax((v[1] - c).magnitudeSquared(), (v[2] - c).magnitudeSquared())));
        return r / PxMin(scale1.x, PxMin(scale1.y, scale1.z));
    };

    // ---- the main run ---------------------------------------------------------
    // A FINITE tolerance, and that is the whole point.  The first version of this
    // fixture used 1e30 to guarantee several rounds, which made
    // `distance < radius/min(scale) + tolerance` trivially true and erased the very
    // term under test: five mutants on the cull predicate -- a max of lengths instead
    // of a sqrt of a max, no scale division, the wrong scale extremum, the radius
    // dropped entirely, and a non-strict compare -- all survived against it.
    //
    // A finite tolerance also restores the second thing 1e30 destroyed.  With every
    // triangle qualifying, the first inner iteration fills the buffer exactly and the
    // guard exits, so no write offset ever exceeds the buffer and a port that CLAMPS
    // the write index instead of skipping it is indistinguishable.  With a partial
    // fill, a later iteration overflows and the difference lands in the last slot.
    const float TOL = 0.35f;
    const Result R = run(TOL, MAX_ROUNDS);
    if (R.numRounds == 0u || R.numRounds == 0xFFFFFFFFu)
    { PXG_CHECK_MSG(false, "the triangle gather produced no rounds (%u)", R.numRounds);
      for (auto h : owned) dev->destroyBuffer(h);
      physx::gpuslang::pxgSetVulkanDriverShim(nullptr); return; }

    size_t badCentroid = 0, badPred = 0, badSub = 0, badSet = 0;
    std::vector<uint8_t> want(NUM_TRIS, 0u);
    for (uint32_t t = 0; t < NUM_TRIS; ++t)
    {
        PxVec3 cv;
        const float r = hostRadiusOverScale(t, cv);
        const F4& e = R.echo[t];
        if (!bitEq(e.x, cv.x) || !bitEq(e.y, cv.y) || !bitEq(e.z, cv.z))
        {
            if (badCentroid < 4)
                std::printf("    tri %u centroid (%.9g %.9g %.9g) want (%.9g %.9g %.9g)\n",
                            t, e.x, e.y, e.z, cv.x, cv.y, cv.z);
            ++badCentroid;
        }
        // The predicate, from the distance the function itself reported.
        want[t] = (e.w < (r + TOL)) ? 1u : 0u;
    }
    PXG_CHECK_MSG(badCentroid == 0, "%zu centroids differ from the host's own transform chain "
                  "(vertex2Shape by scale0, aToB, then shape2Vertex by scale1)", badCentroid);

    // Round by round: the gathered indices must be the qualifying ones in order, and
    // every second half of the uint2 must be a literal zero.
    uint32_t cursor = 0u, gathered = 0u;
    for (uint32_t rr = 0; rr < R.numRounds; ++rr)
    {
        const uint32_t count = R.rounds[rr * 4u + 0];
        const uint32_t before = R.rounds[rr * 4u + 1];
        PXG_CHECK_MSG(R.rounds[rr * 4u + 3] == 0xC0FFEEu, "round %u wrote no canary", rr);
        if (rr == 0u) cursor = before;
        for (uint32_t sIdx = 0; sIdx < count; ++sIdx)
        {
            while (cursor < NUM_TRIS && !want[cursor]) ++cursor;
            const uint32_t gotIdx = R.buf[(size_t(rr) * TARGET_COUNT + sIdx) * 2u + 0];
            const uint32_t gotSub = R.buf[(size_t(rr) * TARGET_COUNT + sIdx) * 2u + 1];
            if (gotIdx != cursor)
            { if (badSet < 4)
                  std::printf("    round %u slot %u: index %u, expected %u\n", rr, sIdx, gotIdx, cursor);
              ++badSet; }
            if (gotSub != 0u) ++badSub;
            ++cursor; ++gathered;
        }
    }
    PXG_CHECK_MSG(badSet == 0, "%zu gathered indices are not the qualifying triangles in order",
                  badSet);
    PXG_CHECK_MSG(badSub == 0, "%zu buffer entries carry a non-zero subdivision level; with "
                  "SUPPORT_SUBDIVISION 0 upstream it is a literal 0", badSub);
    PXG_CHECK_MSG(R.numRounds > 1u,
                  "the gather must need more than one round, or the resume protocol is unchecked "
                  "(got %u)", R.numRounds);
    uint32_t nWant = 0u;
    for (uint32_t t = 0; t < NUM_TRIS; ++t) nWant += want[t];
    PXG_CHECK_MSG(nWant > TARGET_COUNT && nWant < NUM_TRIS - 200u,
                  "the tolerance must leave the cull predicate DECIDING something: %u of %u "
                  "triangles qualify, and the run needs more than one buffer's worth but not "
                  "all of them, or the bounding-sphere term is invisible", nWant, NUM_TRIS);
    // A partial fill is what makes an overflowing write offset possible at all.
    PXG_CHECK_MSG(R.rounds[0] == TARGET_COUNT,
                  "round 0 must fill the buffer (%u)", R.rounds[0]);

    // ---- the gate's strictness, found by SEARCH rather than by luck -----------
    // `distance < radius + tolerance` differs from `<=` only when the two sides are
    // exactly equal, which random geometry never produces.  The tolerance is a free
    // parameter, though, so the boundary can be MANUFACTURED: pick a triangle, take
    // the distance the device reported for it and the radius the host computes
    // exactly, and search the few float values of `tolerance` around `d - r` for one
    // where fl(r + tolerance) == d bitwise.  fl(r + t) is monotone in t, so if such a
    // value exists it is within an ULP or two of the difference.
    {
        uint32_t chosen = 0xFFFFFFFFu;
        float chosenTol = 0.0f;
        for (uint32_t t = 0; t < NUM_TRIS && chosen == 0xFFFFFFFFu; ++t)
        {
            PxVec3 cv;
            const float r = hostRadiusOverScale(t, cv);
            const float d = R.echo[t].w;
            if (!(d > 0.0f) || !std::isfinite(d) || !std::isfinite(r)) continue;
            const float base = d - r;
            for (int k = -3; k <= 3 && chosen == 0xFFFFFFFFu; ++k)
            {
                const float cand = (k == 0) ? base
                                            : u2f(uint32_t(int32_t(f2u(base)) + k));
                if (!(cand > 0.0f) || !std::isfinite(cand)) continue;
                if (f2u(r + cand) == f2u(d)) { chosen = t; chosenTol = cand; }
            }
        }
        PXG_CHECK_MSG(chosen != 0xFFFFFFFFu,
                      "no tolerance puts any triangle exactly on the gate's boundary, so `<` and "
                      "`<=` are indistinguishable and the strictness is unchecked");
        if (chosen != 0xFFFFFFFFu)
        {
            const Result RB = run(chosenTol, MAX_ROUNDS);
            // The boundary triangle must be EXCLUDED: the gate is strict.
            uint32_t seen = 0u, total = 0u;
            for (uint32_t rr = 0; rr < RB.numRounds; ++rr)
            {
                const uint32_t cnt = RB.rounds[rr * 4u + 0];
                total += cnt;
                for (uint32_t sIdx = 0; sIdx < cnt; ++sIdx)
                    if (RB.buf[(size_t(rr) * TARGET_COUNT + sIdx) * 2u + 0] == chosen) ++seen;
            }
            PxVec3 cvv;
            const float rr0 = hostRadiusOverScale(chosen, cvv);
            PXG_CHECK_MSG(f2u(rr0 + chosenTol) == f2u(RB.echo[chosen].w),
                          "the manufactured boundary did not hold at the chosen tolerance");
            PXG_CHECK_MSG(seen == 0u,
                          "triangle %u sits EXACTLY on the gate (distance == radius + tolerance) "
                          "and was gathered anyway; the comparison is strict", chosen);
            std::printf("    [ note ] gate strictness: triangle %u placed exactly on the boundary "
                        "at tolerance %.9g and correctly excluded (%u gathered)\n",
                        chosen, chosenTol, total);
        }
    }

    // ---- the count guard's ONLY observable ------------------------------------
    // One round, a tolerance that fills the buffer early.  With the guard the block
    // stops as soon as the buffer is full, so triangles in later blocks are never
    // sampled and their echo slots keep the poison.  Without it the sweep runs to the
    // end of the mesh and every slot is written.
    {
        const Result R1 = run(1.0e30f, 1u);
        const uint32_t count = R1.rounds[0];
        PXG_CHECK_MSG(count == TARGET_COUNT,
                      "the single-round configuration must fill the buffer exactly (%u)", count);
        size_t untouched = 0;
        for (uint32_t t = 0; t < NUM_TRIS; ++t)
            if (R1.echo[t].w == -777.f) ++untouched;
        PXG_CHECK_MSG(untouched > 0,
                      "every triangle was sampled in a single round that filled the buffer after "
                      "%u of %u; the loop's `sharedIndicesCount < TargetCount` guard -- which the "
                      "vertex sibling does not have -- is doing nothing", count, NUM_TRIS);
        std::printf("    [ note ] count guard: %u of %u triangles never sampled in the "
                    "buffer-filling round\n", uint32_t(untouched), NUM_TRIS);
    }

    std::printf("    [ note ] %u triangles, %u qualifying at tolerance %g, %u rounds, %u "
                "gathered; centroids bit-exact against the host transform chain, subdivision "
                "level 0 throughout\n", NUM_TRIS, nWant, TOL, R.numRounds, gathered);

    for (auto h : owned) dev->destroyBuffer(h);
    physx::gpuslang::pxgSetVulkanDriverShim(nullptr);
}

// ===========================================================================
// P9.35b -- findInterestingTets, the third member of the family.
//
// It is the VERTEX form's buffer (plain uints, not the triangle form's uint2)
// carrying the TRIANGLE form's cull test (a bounding sphere, not a point), and
// three things are its own:
//
//   U1  NO COUNT GUARD.  `i < nbTets` alone, where the triangle form carries
//       `&& sharedIndicesCount < TargetCount`.  P9.34b proved the guard is
//       output-neutral -- once the buffer saturates every later write offset is
//       past it and the resume handshake cannot fire again -- so its ABSENCE is
//       output-neutral too, and the only observable is which tets were sampled.
//       That test is the exact INVERSE of the triangle fixture's: there, some
//       triangles must go untouched in a buffer-filling round; here, NONE may.
//   U2  no vertex2Shape on the softbody side at all -- not a unit scale, no call.
//   U3  the radius divides by min(scale) with NO absolute value, while the KERNEL
//       that calls this divides its contact distance by scale.abs().minElement().
// ===========================================================================
PXG_TEST(sdfFindTets)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }

    const PxgKernelDef manifest[] = { { "npSdfFind.spv", "sdfFindTetRounds", 0u, 0u } };
    static auto loaderT2 = [](const char* f) { return loadShaderBlob(f); };
    PxgCudaShim shim(dev, manifest, 1, +loaderT2);
    physx::gpuslang::pxgSetVulkanDriverShim(&shim);

    std::vector<PxgBufferHandle> owned;
    auto keep = [&](PxgBufferHandle h) { owned.push_back(h); return h; };

    SdfScene sp;
    buildSparse(sp);
    BlobLayout blob = layOutBlob(sp, 16u, 21u, 3u, 1u, 0u);
    const uint64_t hCoarse = sdfUploadTex(CU_AD_FORMAT_FLOAT, sp.nbx + 1, sp.nby + 1, sp.nbz + 1,
                                          sp.coarse.data(), sizeof(float));
    const uint64_t hSub = sdfUploadTex(CU_AD_FORMAT_UNSIGNED_INT8, sp.blkx * (sp.sg + 1),
                                       sp.blky * (sp.sg + 1), sp.blkz * (sp.sg + 1),
                                       sp.sub.data(), 1);
    PXG_CHECK_MSG(hCoarse != 0 && hSub != 0, "the SDF textures must upload");
    if (hCoarse == 0 || hSub == 0) return;
    std::memcpy(&blob.bytes[blob.texObjOffset], &hCoarse, 8);
    std::memcpy(&blob.bytes[blob.texObjSparseOffset], &hSub, 8);
    const uint64_t blobAddr = dev->getBufferDeviceAddress(keep(createBufferFrom(dev, blob.bytes)));

    // Nothing identity or axis-aligned.  There is only ONE scale here -- U2 says
    // the softbody side has none -- so scale0 is not sent at all.
    const PxQuat aToBq = PxQuat(0.41f, PxVec3(-0.48f, 0.6f, 0.64031f).getNormalized());
    const PxVec3 aToBp(-0.33f, 0.74f, 0.52f);
    const PxTransform aToB(aToBp, aToBq);
    // U3 needs a NEGATIVE component to be observable at all, so the scale is a
    // variable here and the run below happens twice.  The negative one is not the
    // smallest in magnitude, so min(x,y,z) and min(|x|,|y|,|z|) differ in BOTH
    // value and sign -- the same device P9.34c used for the kernel's own cullScale,
    // pointed at the opposite spelling.
    PxVec3 scale1(1.35f, 0.72f, 2.10f);
    const PxQuat scale1Rot = PxQuat(-0.58f, PxVec3(-0.6f, 0.48f, 0.64031f).getNormalized());

    const uint32_t NUM_TETS = 4000u;
    const uint32_t NUM_V = NUM_TETS * 4u;
    const HostSdfBox box = hostSdfBox(sp);
    const PxVec3 ext = box.higher - box.lower;
    const uint32_t TAIL_FAR = 200u;

    std::vector<F4> verts(NUM_V);
    std::vector<uint32_t> inds(NUM_TETS * 4u);
    {
        std::mt19937 rng(0x5EC70A2u);
        auto u01 = [&]() { return float(rng() % 1000000u) / 1000000.f; };
        const PxTransform bToA = aToB.getInverse();
        for (uint32_t t = 0; t < NUM_TETS; ++t)
        {
            // Sizes vary, because the cull test divides the bounding-sphere radius by
            // the smallest scale component and adds it to the tolerance -- uniform
            // tets cannot tell that term from a constant.
            const float sz = (t % 7u == 0u) ? 0.9f * ext.x
                                            : 0.02f * ext.x * (1.0f + float(t % 5));
            PxVec3 c;
            if (t < NUM_TETS - TAIL_FAR)
                c = PxVec3(box.lower.x - 0.4f * ext.x + 1.8f * ext.x * u01(),
                           box.lower.y - 0.4f * ext.y + 1.8f * ext.y * u01(),
                           box.lower.z - 0.4f * ext.z + 1.8f * ext.z * u01());
            else
                c = PxVec3(box.lower.x - (8.0f + 3.0f * u01()) * ext.x,
                           box.lower.y - (8.0f + 3.0f * u01()) * ext.y,
                           box.lower.z - (8.0f + 3.0f * u01()) * ext.z);
            const PxVec3 corner[4] = { c + PxVec3(sz, 0.0f, 0.0f),
                                       c + PxVec3(-0.3f * sz, 0.9f * sz, 0.2f * sz),
                                       c + PxVec3(0.1f * sz, -0.4f * sz, -0.8f * sz),
                                       c + PxVec3(-0.2f * sz, -0.35f * sz, 0.75f * sz) };
            for (uint32_t k = 0; k < 4u; ++k)
            {
                // U2: generated in SHAPE space and mapped back through the inverse of
                // the a-to-b transform ALONE -- no scale on this side, which is what
                // makes the forward chain the kernel's.
                const PxVec3 world = bToA.transform(corner[k]);
                verts[t * 4u + k] = F4{ world.x, world.y, world.z, 0.0f };
            }
            // The four indices are ROTATED by t, so a port that reads .y where .x
            // belongs picks a different corner.
            const uint32_t r = t % 4u;
            for (uint32_t k = 0; k < 4u; ++k)
                inds[t * 4u + k] = t * 4u + (r + k) % 4u;
        }
    }
    const uint64_t vAddr = dev->getBufferDeviceAddress(keep(createBufferFrom(dev, verts)));
    const uint64_t iAddr = dev->getBufferDeviceAddress(keep(createBufferFrom(dev, inds)));

    const std::vector<F4> echoInit(NUM_TETS, F4{ -777.f, -777.f, -777.f, -777.f });
    const std::vector<uint32_t> touchInit(NUM_TETS, 0u);
    const std::vector<uint32_t> roundsInit(MAX_ROUNDS * 4u, 0xEEEEEEEEu);
    const std::vector<uint32_t> bufInit(size_t(MAX_ROUNDS) * TARGET_COUNT, 0xEEEEEEEEu);
    const std::vector<uint32_t> nrInit(1u, 0xEEEEEEEEu);

    P_Find base = {};
    base.meshPtr = blobAddr;
    base.positions = vAddr;
    base.meshInd = iAddr;
    base.aToBq[0] = aToBq.x; base.aToBq[1] = aToBq.y; base.aToBq[2] = aToBq.z; base.aToBq[3] = aToBq.w;
    base.scaleRot[0] = scale1Rot.x; base.scaleRot[1] = scale1Rot.y;
    base.scaleRot[2] = scale1Rot.z; base.scaleRot[3] = scale1Rot.w;
    base.aToBp[0] = aToBp.x; base.aToBp[1] = aToBp.y; base.aToBp[2] = aToBp.z;
    base.scale[0] = scale1.x; base.scale[1] = scale1.y; base.scale[2] = scale1.z;
    base.numVertices = NUM_TETS;
    base.maxRounds = MAX_ROUNDS;

    struct Result
    {
        uint32_t numRounds = 0;
        std::vector<uint32_t> rounds, buf, touched;
        std::vector<F4> echo;
    };
    auto run = [&](float tol, uint32_t maxRounds) -> Result {
        Result out;
        const PxgBufferHandle bE = keep(createBufferFrom(dev, echoInit));
        const PxgBufferHandle bT = keep(createBufferFrom(dev, touchInit));
        const PxgBufferHandle bR = keep(createBufferFrom(dev, roundsInit));
        const PxgBufferHandle bB = keep(createBufferFrom(dev, bufInit));
        const PxgBufferHandle bN = keep(createBufferFrom(dev, nrInit));
        P_Find p = base;
        p.scale[0] = scale1.x; p.scale[1] = scale1.y; p.scale[2] = scale1.z;
        p.echoOut = dev->getBufferDeviceAddress(bE);
        p.touchedOut = dev->getBufferDeviceAddress(bT);
        p.roundsOut = dev->getBufferDeviceAddress(bR);
        p.bufOut = dev->getBufferDeviceAddress(bB);
        p.numRoundsOut = dev->getBufferDeviceAddress(bN);
        p.tolerance = tol;
        p.maxRounds = maxRounds;
        Kernel k(Kernel::Bindless{}, "npSdfFind.spv", "sdfFindTetRounds", {}, sizeof(P_Find));
        PXG_CHECK_MSG(k.isValid(), "bindless pipeline creation succeeds for sdfFindTetRounds");
        if (!k.isValid()) return out;
        k.dispatch(1u, {}, p);
        dev->flush();
        out.numRounds = downloadBuffer<uint32_t>(dev, bN, 1)[0];
        out.rounds = downloadBuffer<uint32_t>(dev, bR, MAX_ROUNDS * 4u);
        out.buf = downloadBuffer<uint32_t>(dev, bB, size_t(MAX_ROUNDS) * TARGET_COUNT);
        out.echo = downloadBuffer<F4>(dev, bE, NUM_TETS);
        out.touched = downloadBuffer<uint32_t>(dev, bT, NUM_TETS);
        return out;
    };

    // The host's own centroid and radius, through the ENGINE's transform.  Only the
    // SDF distance comes from the device.
    auto hostRadiusOverScale = [&](uint32_t t, PxVec3& centroidVertexSpace) {
        PxVec3 v[4];
        for (uint32_t k = 0; k < 4u; ++k)
        {
            const F4& w = verts[inds[t * 4u + k]];
            v[k] = aToB.transform(PxVec3(w.x, w.y, w.z));            // U2: no scale
        }
        const PxVec3 c = (v[0] + v[1] + v[2] + v[3]) * 0.25f;
        centroidVertexSpace = hShape2Vertex(c, scale1, scale1Rot);
        const float r = std::sqrt(
            PxMax(PxMax((v[0] - c).magnitudeSquared(), (v[1] - c).magnitudeSquared()),
                  PxMax((v[2] - c).magnitudeSquared(), (v[3] - c).magnitudeSquared())));
        return r / PxMin(scale1.x, PxMin(scale1.y, scale1.z));       // U3: NO abs
    };

    const float TOL = 0.35f;
    const Result R = run(TOL, MAX_ROUNDS);
    if (R.numRounds == 0u || R.numRounds == 0xFFFFFFFFu)
    { PXG_CHECK_MSG(false, "the tet gather produced no rounds (%u)", R.numRounds);
      for (auto h : owned) dev->destroyBuffer(h);
      physx::gpuslang::pxgSetVulkanDriverShim(nullptr); return; }

    size_t badCentroid = 0, badSet = 0;
    std::vector<uint8_t> want(NUM_TETS, 0u);
    for (uint32_t t = 0; t < NUM_TETS; ++t)
    {
        PxVec3 cv;
        const float r = hostRadiusOverScale(t, cv);
        const F4& e = R.echo[t];
        if (!bitEq(e.x, cv.x) || !bitEq(e.y, cv.y) || !bitEq(e.z, cv.z))
        {
            if (badCentroid < 4)
                std::printf("    tet %u centroid (%.9g %.9g %.9g) want (%.9g %.9g %.9g)\n",
                            t, e.x, e.y, e.z, cv.x, cv.y, cv.z);
            ++badCentroid;
        }
        want[t] = (e.w < (r + TOL)) ? 1u : 0u;
    }
    PXG_CHECK_MSG(badCentroid == 0, "%zu centroids differ from the host's own chain (aToB alone, "
                  "then shape2Vertex by the MESH scale)", badCentroid);

    uint32_t cursor = 0u, gathered = 0u;
    for (uint32_t rr = 0; rr < R.numRounds; ++rr)
    {
        const uint32_t count = R.rounds[rr * 4u + 0];
        const uint32_t before = R.rounds[rr * 4u + 1];
        PXG_CHECK_MSG(R.rounds[rr * 4u + 3] == 0xC0FFEEu, "round %u wrote no canary", rr);
        if (rr == 0u) cursor = before;
        for (uint32_t sIdx = 0; sIdx < count; ++sIdx)
        {
            while (cursor < NUM_TETS && !want[cursor]) ++cursor;
            const uint32_t gotIdx = R.buf[size_t(rr) * TARGET_COUNT + sIdx];
            if (gotIdx != cursor)
            { if (badSet < 4)
                  std::printf("    round %u slot %u: index %u, expected %u\n",
                              rr, sIdx, gotIdx, cursor);
              ++badSet; }
            ++cursor; ++gathered;
        }
    }
    PXG_CHECK_MSG(badSet == 0, "%zu gathered indices are not the qualifying tets in order", badSet);
    PXG_CHECK_MSG(R.numRounds > 1u,
                  "the gather must need more than one round, or the resume protocol is unchecked "
                  "(got %u)", R.numRounds);
    uint32_t nWant = 0u;
    for (uint32_t t = 0; t < NUM_TETS; ++t) nWant += want[t];
    PXG_CHECK_MSG(nWant > TARGET_COUNT && nWant < NUM_TETS - 200u,
                  "the tolerance must leave the cull predicate DECIDING something: %u of %u tets "
                  "qualify, and the run needs more than one buffer's worth but not all of them, "
                  "or the bounding-sphere term is invisible", nWant, NUM_TETS);
    PXG_CHECK_MSG(R.rounds[0] == TARGET_COUNT,
                  "round 0 must fill the buffer (%u)", R.rounds[0]);

    // ---- U1: the MISSING count guard, and the assertion is INVERTED ------------
    // The triangle fixture requires that a buffer-filling round leave some triangles
    // untouched, because its loop stops when the buffer is full.  This loop does not,
    // so the same configuration must touch EVERY tet.  Same probe, opposite verdict:
    // that is what makes the two functions' asymmetry a checked fact rather than a
    // remark in a header.
    {
        const Result R1 = run(1.0e30f, 1u);
        const uint32_t count = R1.rounds[0];
        PXG_CHECK_MSG(count == TARGET_COUNT,
                      "the single-round configuration must fill the buffer exactly (%u)", count);
        size_t untouched = 0;
        for (uint32_t t = 0; t < NUM_TETS; ++t)
            if (R1.touched[t] == 0u) ++untouched;
        PXG_CHECK_MSG(untouched == 0,
                      "%zu of %u tets went unsampled in a round that filled the buffer after %u; "
                      "findInterestingTets has NO `sharedIndicesCount < TargetCount` guard, so it "
                      "must scan to the end of the mesh -- this is the exact inverse of the "
                      "triangle gather's contract", untouched, NUM_TETS, count);
        std::printf("    [ note ] no count guard: all %u tets sampled in the buffer-filling "
                    "round (the triangle form leaves 2976 of 4000 untouched)\n", NUM_TETS);
    }

    // ---- the gate's strictness, manufactured exactly as P9.34b did -------------
    {
        uint32_t chosen = 0xFFFFFFFFu;
        float chosenTol = 0.0f;
        for (uint32_t t = 0; t < NUM_TETS && chosen == 0xFFFFFFFFu; ++t)
        {
            PxVec3 cv;
            const float r = hostRadiusOverScale(t, cv);
            const float d = R.echo[t].w;
            if (!(d > 0.0f) || !std::isfinite(d) || !std::isfinite(r)) continue;
            const float baseT = d - r;
            for (int k = -3; k <= 3 && chosen == 0xFFFFFFFFu; ++k)
            {
                const float cand = (k == 0) ? baseT
                                            : u2f(uint32_t(int32_t(f2u(baseT)) + k));
                if (!(cand > 0.0f) || !std::isfinite(cand)) continue;
                if (f2u(r + cand) == f2u(d)) { chosen = t; chosenTol = cand; }
            }
        }
        PXG_CHECK_MSG(chosen != 0xFFFFFFFFu,
                      "no tolerance puts any tet exactly on the gate's boundary, so `<` and `<=` "
                      "are indistinguishable and the strictness is unchecked");
        if (chosen != 0xFFFFFFFFu)
        {
            const Result RB = run(chosenTol, MAX_ROUNDS);
            uint32_t seen = 0u, total = 0u;
            for (uint32_t rr = 0; rr < RB.numRounds; ++rr)
            {
                const uint32_t cnt = RB.rounds[rr * 4u + 0];
                total += cnt;
                for (uint32_t sIdx = 0; sIdx < cnt; ++sIdx)
                    if (RB.buf[size_t(rr) * TARGET_COUNT + sIdx] == chosen) ++seen;
            }
            PxVec3 cvv;
            const float rr0 = hostRadiusOverScale(chosen, cvv);
            PXG_CHECK_MSG(f2u(rr0 + chosenTol) == f2u(RB.echo[chosen].w),
                          "the manufactured boundary did not hold at the chosen tolerance");
            PXG_CHECK_MSG(seen == 0u,
                          "tet %u sits EXACTLY on the gate and was gathered anyway; the "
                          "comparison is strict", chosen);
            std::printf("    [ note ] gate strictness: tet %u placed exactly on the boundary at "
                        "tolerance %.9g and correctly excluded (%u gathered)\n",
                        chosen, chosenTol, total);
        }
    }

    // ---- U3: the radius divide takes NO absolute value ------------------------
    // With an all-positive scale `min(x,y,z)` and `min(|x|,|y|,|z|)` are the same
    // expression, so the first sweep left that mutant alive -- the rig could not
    // tell the two apart.  A NEGATIVE component that is not the smallest in
    // MAGNITUDE separates them in both value and sign, and the whole comparison
    // above is repeated against it: the host formula has no abs either, so if the
    // shader grew one the centroids would still match but the qualifying SET would
    // not.
    //
    // And this is the trap's other half, worth stating where it is checked: the
    // KERNEL that calls this function divides its own contact distance by
    // `scale.abs().minElement()` -- WITH the absolute value.  Two spellings of one
    // idea, forty lines apart, inside a single call chain.
    {
        scale1 = PxVec3(1.17f, -2.90f, 0.86f);
        const float minSigned = PxMin(scale1.x, PxMin(scale1.y, scale1.z));
        const float minAbs = PxMin(PxAbs(scale1.x), PxMin(PxAbs(scale1.y), PxAbs(scale1.z)));
        PXG_CHECK_MSG(minSigned < 0.0f && minAbs > 0.0f && f2u(minSigned) != f2u(-minAbs),
                      "the negative-scale run must make min(x,y,z) = %.9g and min of the "
                      "magnitudes = %.9g differ in sign AND in value", minSigned, minAbs);

        const float TOLN = 0.9f;
        const Result RN = run(TOLN, MAX_ROUNDS);
        PXG_CHECK_MSG(RN.numRounds > 0u && RN.numRounds != 0xFFFFFFFFu,
                      "the negative-scale gather produced no rounds (%u)", RN.numRounds);

        size_t badCN = 0, badSetN = 0;
        std::vector<uint8_t> wantN(NUM_TETS, 0u);
        uint32_t nWantN = 0u, nWantAbs = 0u;
        for (uint32_t t = 0; t < NUM_TETS; ++t)
        {
            PxVec3 cv;
            const float r = hostRadiusOverScale(t, cv);       // no abs, like the shader
            const F4& e = RN.echo[t];
            if (!bitEq(e.x, cv.x) || !bitEq(e.y, cv.y) || !bitEq(e.z, cv.z)) ++badCN;
            wantN[t] = (e.w < (r + TOLN)) ? 1u : 0u;
            nWantN += wantN[t];
            // What an abs()-taking port would have gathered, so the run can insist
            // the two sets genuinely differ rather than assuming they do.
            const float rAbs = (r * minSigned) / minAbs;
            if (e.w < (rAbs + TOLN)) ++nWantAbs;
        }
        PXG_CHECK_MSG(badCN == 0, "%zu centroids differ under the negative scale", badCN);

        uint32_t curN = 0u, gotN = 0u;
        for (uint32_t rr = 0; rr < RN.numRounds; ++rr)
        {
            const uint32_t count = RN.rounds[rr * 4u + 0];
            const uint32_t before = RN.rounds[rr * 4u + 1];
            if (rr == 0u) curN = before;
            for (uint32_t sIdx = 0; sIdx < count; ++sIdx)
            {
                while (curN < NUM_TETS && !wantN[curN]) ++curN;
                if (RN.buf[size_t(rr) * TARGET_COUNT + sIdx] != curN) ++badSetN;
                ++curN; ++gotN;
            }
        }
        PXG_CHECK_MSG(badSetN == 0,
                      "%zu gathered indices under the negative scale are not the qualifying tets "
                      "in order -- the radius divide must NOT take an absolute value", badSetN);
        PXG_CHECK_MSG(nWantN > 0u,
                      "the negative-scale run gathered nothing, so it discriminates nothing");
        const uint32_t diff = (nWantN > nWantAbs) ? (nWantN - nWantAbs) : (nWantAbs - nWantN);
        PXG_CHECK_MSG(diff > 100u,
                      "an abs()-taking port would gather %u tets against the engine's %u -- too "
                      "close for the missing absolute value to be observable", nWantAbs, nWantN);
        std::printf("    [ note ] no abs on the radius divide: min(x,y,z) %.9g against min of the "
                    "magnitudes %.9g; %u tets qualify, %u would under an abs() spelling\n",
                    minSigned, minAbs, nWantN, nWantAbs);
    }

    std::printf("    [ note ] %u tets, %u qualifying at tolerance %g, %u rounds, %u gathered; "
                "centroids bit-exact against the host chain\n",
                NUM_TETS, nWant, TOL, R.numRounds, gathered);

    for (auto h : owned) dev->destroyBuffer(h);
    physx::gpuslang::pxgSetVulkanDriverShim(nullptr);
}

} // namespace
