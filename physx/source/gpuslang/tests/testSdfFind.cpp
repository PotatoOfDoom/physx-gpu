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

// npSdfFind.slang's push-constant block -- 128 bytes, laid out to match.
struct P_Find
{
    uint64_t meshPtr, positions, echoOut, roundsOut, bufOut, numRoundsOut;   //  0.. 47
    float aToBq[4];                                                          // 48
    float scaleRot[4];                                                       // 64
    float aToBp[3];  float tolerance;                                        // 80
    float scale[3];  uint32_t numVertices;                                   // 96
    uint32_t maxRounds, pad0, pad1, pad2;                                    // 112
};

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

} // namespace
