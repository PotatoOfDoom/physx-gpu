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

// testSdfParticle.cpp -- P9.12f, the fixture for pxgSlangSdfParticle.
//
// The third fixture in this chain built the same way, and by now the
// arrangement is the point rather than a workaround:
//
//   * the shader reports the THREE doVertexSDFCollision results it obtained --
//     nothing else about them is predicted, because Sample() lives on the GPU;
//   * everything between and around them is recomputed on the host out of the
//     engine's own PxVec3 / PxQuat / PxTransform, CALLED, and compared
//     BIT-FOR-BIT.  There is not a tolerance in this file.
//
// COVERAGE IS ASSERTED, NOT DESIGNED.  The branch a pair takes depends on SDF
// values the host cannot predict, so hand-picking one pair per branch would be
// guesswork.  Instead a spread of 512 pairs is generated -- previous positions
// across and around the box, displacements from sub-voxel to several box
// extents -- every one of them is checked exactly, and the fixture then
// REQUIRES that the histogram contains each branch, each guard outcome, and the
// two paths that only exist because of the engine's own defects:
//
//   * a pair where the guard is entered but `m` is zero, which is the
//     UNINITIALISED-`dir` path (upstream candidate 27).  In this port `dir` is
//     zeroed, so `m == 0` and nothing is scaled; in the CUDA it is whatever the
//     stack held.  Asserting that the case OCCURS is what keeps the divergence
//     honest -- it is exercised, not merely described in a comment.
//   * a pair that reaches the CCD re-cast, which is where candidate 26's
//     double transform lives.  The host mirror performs the same double
//     transform, because the port transcribes it.
//
// The whole set is run twice, with enableCCD on and off, and the fixture
// requires branch 3 to appear in the first run and to be absent from the
// second -- which is the only thing that proves the flag is read at all.

#include "pxgTestFramework.h"

#include "PxgCudaShim.h"
#include "PxgCudaToolkitShim.h"
#include "pxgSdfSceneRef.h"

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

struct P_Part
{
    uint64_t meshPtr, positions, positionsPrev, outBuf;   //  0
    float aToBq[4];                                       // 32
    float scaleRot[4];                                    // 48
    float aToBp[3]; float cullScale;                      // 64
    float scale[3];  float contactDist;                   // 80
    float restDist; uint32_t numPairs, enableCCD, pad0;   // 96
};

static const uint32_t STRIDE = 64u;
static const uint32_t NUM_PAIRS = 512u;

// npSdfCollision.slang's push-constant block and the stride of its sdfCollVertex
// entry (P9.12d).  Used to re-evaluate doVertexSDFCollision INDEPENDENTLY at the
// points this shader says it sampled -- see the note above runAndCheck.
// P9.98: the whole SdfCollParams, 64 B since P9.34a appended triVerts/triOut (it was
// 48, the VUID-07987 the validation layer reported); sdfCollVertex reads neither.
struct P_Coll
{
    uint64_t meshPtr, points, gradIn, dirSeed, outBuf;
    uint32_t numPoints, pad;
    uint64_t triVerts, triOut;   // 48, 56 -- the triangle descent's (0 here)
};
static_assert(sizeof(P_Coll) == 64, "npSdfCollision push block (SdfCollParams)");
static const uint32_t STRIDE_VTX = 16u;

static bool bitEq(float a, float b) { return f2u(a) == f2u(b); }
static bool bitEq3(const float* a, const PxVec3& b)
{ return bitEq(a[0], b.x) && bitEq(a[1], b.y) && bitEq(a[2], b.z); }
static PxVec3 v3(const float* a) { return PxVec3(a[0], a[1], a[2]); }
static uint32_t asU(float f) { return f2u(f); }

// shape2Vertex, cudaNpCommon.h:284-296.  vertex2ShapeNormalVector (:299) is a
// call to this same function, so one mirror covers both uses.
static PxVec3 hShape2Vertex(const PxVec3& v, const PxVec3& scale, const PxQuat& rot)
{
    PxVec3 t = rot.rotate(v);          // ENGINE
    t.x /= scale.x; t.y /= scale.y; t.z /= scale.z;
    return rot.rotateInv(t);           // ENGINE
}
static PxVec3 hVertex2Shape(const PxVec3& v, const PxVec3& scale, const PxQuat& rot)
{
    PxVec3 t = rot.rotate(v);
    t.x *= scale.x; t.y *= scale.y; t.z *= scale.z;
    return rot.rotateInv(t);
}

PXG_TEST(sdfParticle)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }

    const PxgKernelDef manifest[] = { { "npSdfParticle.spv", "sdfParticleContactAt", 0u, 0u } };
    static auto loader = [](const char* f) { return loadShaderBlob(f); };
    PxgCudaShim shim(dev, manifest, 1, +loader);
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

    // Nothing identity, nothing axis-aligned, no scale component equal to
    // another or to 1 -- the vertex-space chain is compared bit-for-bit and a
    // wrong rotation order or a swapped scale lands there.
    const PxQuat aToBq = PxQuat(0.41f, PxVec3(-0.48f, 0.6f, 0.64031f).getNormalized());
    const PxVec3 aToBp(-0.17f, 0.44f, -0.92f);
    const PxTransform aToB(aToBp, aToBq);
    const PxVec3 scale(1.21f, 0.68f, 1.93f);
    const PxQuat scaleRot = PxQuat(0.63f, PxVec3(0.8f, 0.36f, 0.48f).getNormalized());

    // The kernel derives the cull tolerance this way (particleSystemMeshMidphase
    // .cu:470-471): the contact distance divided by the SMALLEST absolute scale
    // component, so a squashed mesh culls wider in vertex space.
    const float contactDist = 0.085f;
    const float cullScale = contactDist / PxMin(PxAbs(scale.x), PxMin(PxAbs(scale.y), PxAbs(scale.z)));
    // Two rest distances, and the second one is not a rounding of the first.
    // The CCD displacement clamps `startDist - restDist` at zero, and that clamp
    // is DEAD whenever restDist < contactDist: the branch is only reached when
    // startDist >= contactDist, so the difference is already positive.  Nothing
    // in the engine guarantees the ordering -- restDist comes from
    // restDistances[cmIdx] and contactDist from two contactDistance entries --
    // so one configuration turns it around and makes the clamp live.
    const float restDistSmall = 0.021f;
    const float restDistLarge = 0.140f;      // > contactDist

    const HostSdfBox box = hostSdfBox(sp);
    const PxVec3 ext = box.higher - box.lower;

    // The point cloud is generated in VERTEX space and mapped back through the
    // inverse of whichever transform the configuration uses, so every
    // configuration gets the same coverage of the SDF regardless of its pose.
    auto makeCloud = [&](const PxTransform& aToB, std::vector<F4>& outPos,
                         std::vector<F4>& outPrev, uint32_t seed) {
        outPos.resize(NUM_PAIRS);
        outPrev.resize(NUM_PAIRS);
        const PxTransform bToA = aToB.getInverse();
        std::mt19937 rng(seed);
        auto u01 = [&]() { return float(rng() % 1000000u) / 1000000.f; };
        auto sym = [&]() { return 2.0f * u01() - 1.0f; };
        for (uint32_t i = 0; i < NUM_PAIRS; ++i)
        {
            const PxVec3 vPrev(box.lower.x - 0.3f * ext.x + 1.6f * ext.x * u01(),
                               box.lower.y - 0.3f * ext.y + 1.6f * ext.y * u01(),
                               box.lower.z - 0.3f * ext.z + 1.6f * ext.z * u01());
            // Displacements from sub-voxel to several box extents, so pairs land
            // on every combination of "in range" / "out of range" for the two
            // samples -- which is what fills the branch histogram.
            const float mag = (i % 4u == 0u) ? 0.02f * sp.spacing
                            : (i % 4u == 1u) ? 0.9f * sp.spacing
                            : (i % 4u == 2u) ? 0.35f * ext.magnitude()
                            :                  2.5f * ext.magnitude();
            const PxVec3 v = vPrev + PxVec3(sym(), sym(), sym()) * mag;
            const PxVec3 wPrev = bToA.transform(hVertex2Shape(vPrev, scale, scaleRot));
            const PxVec3 w = bToA.transform(hVertex2Shape(v, scale, scaleRot));
            outPrev[i] = F4{ wPrev.x, wPrev.y, wPrev.z, 1.0f };
            outPos[i] = F4{ w.x, w.y, w.z, 1.0f };
        }
    };

    const std::vector<float> outInit(size_t(NUM_PAIRS) * STRIDE, -4242.f);

    struct Stats
    {
        size_t branch[4] = { 0, 0, 0, 0 };
        size_t enteredYes = 0, enteredNo = 0, mZero = 0, rejectedDistance = 0, ccdReached = 0;
        size_t clampFired = 0, crossChecked = 0;
        size_t bad = 0;
    };

    // ---- one configuration: a pose, the CCD flag, and its own cloud ----
    auto runAndCheck = [&](const char* label, const PxTransform& aToB, bool ccd,
                           float restDist, uint32_t seed) -> Stats {
        Stats st;
        std::vector<F4> pos, posPrev;
        makeCloud(aToB, pos, posPrev, seed);

        const PxgBufferHandle bOut = keep(createBufferFrom(dev, outInit));
        P_Part p = {};
        p.meshPtr = blobAddr;
        p.positions = dev->getBufferDeviceAddress(keep(createBufferFrom(dev, pos)));
        p.positionsPrev = dev->getBufferDeviceAddress(keep(createBufferFrom(dev, posPrev)));
        p.outBuf = dev->getBufferDeviceAddress(bOut);
        p.aToBq[0] = aToB.q.x; p.aToBq[1] = aToB.q.y; p.aToBq[2] = aToB.q.z; p.aToBq[3] = aToB.q.w;
        p.scaleRot[0] = scaleRot.x; p.scaleRot[1] = scaleRot.y;
        p.scaleRot[2] = scaleRot.z; p.scaleRot[3] = scaleRot.w;
        p.aToBp[0] = aToB.p.x; p.aToBp[1] = aToB.p.y; p.aToBp[2] = aToB.p.z;
        p.scale[0] = scale.x; p.scale[1] = scale.y; p.scale[2] = scale.z;
        p.cullScale = cullScale; p.contactDist = contactDist; p.restDist = restDist;
        p.numPairs = NUM_PAIRS;
        p.enableCCD = ccd ? 1u : 0u;
        {
            Kernel k(Kernel::Bindless{}, "npSdfParticle.spv", "sdfParticleContactAt", {}, sizeof(P_Part));
            PXG_CHECK_MSG(k.isValid(), "%s: bindless pipeline creation succeeds", label);
            if (!k.isValid()) return st;
            k.dispatch((NUM_PAIRS + 63u) / 64u, {}, p);
        }
        dev->flush();
        const std::vector<float> R = downloadBuffer<float>(dev, bOut, size_t(NUM_PAIRS) * STRIDE);

        auto fail = [&](uint32_t i, const char* what, float got, float want) {
            if (st.bad < 8)
                std::printf("    %s pair %u: %s = %.9g, expected %.9g\n", label, i, what, got, want);
            ++st.bad;
        };

        // Every point this shader claims to have sampled, re-evaluated
        // INDEPENDENTLY below through npSdfCollision's own sdfCollVertex entry.
        // Without it the tolerance handed to doVertexSDFCollision, and the value
        // its `dir` output is seeded with, are unobservable: both reach the host
        // only THROUGH the reported result, so the mirror follows whatever the
        // shader did.  The first sweep found exactly that -- two survivors,
        // b2-cullscale-is-contactdist and c27-dirprev-seed-nonzero.
        std::vector<F4> xPts;
        std::vector<F4> xSeed;           // what `dir` HELD when the module made the call
        std::vector<uint32_t> xSrc;      // pair index * 4 + kind (0 cur, 1 prev, 2 ccd)

        for (uint32_t i = 0; i < NUM_PAIRS; ++i)
        {
            const float* o = &R[size_t(i) * STRIDE];

            // ---- the parts that need no SDF at all ----
            const PxVec3 wPos(pos[i].x, pos[i].y, pos[i].z);
            const PxVec3 wPrev(posPrev[i].x, posPrev[i].y, posPrev[i].z);
            const PxVec3 ePos = aToB.transform(wPos);        // ENGINE
            const PxVec3 ePrev = aToB.transform(wPrev);      // ENGINE
            const PxVec3 eVtx = hShape2Vertex(ePos, scale, scaleRot);
            const PxVec3 eVtxPrev = hShape2Vertex(ePrev, scale, scaleRot);
            if (!bitEq3(o + 0, ePos)) fail(i, "pos.x", o[0], ePos.x);
            if (!bitEq3(o + 3, ePrev)) fail(i, "posPrev.x", o[3], ePrev.x);
            if (!bitEq3(o + 6, eVtx)) fail(i, "vtx.x", o[6], eVtx.x);
            if (!bitEq3(o + 9, eVtxPrev)) fail(i, "vtxPrev.x", o[9], eVtxPrev.x);

            // The first two calls are made with `dir` freshly zeroed.
            xPts.push_back(F4{ o[6], o[7], o[8], cullScale });
            xSeed.push_back(F4{ 0.f, 0.f, 0.f, 0.f });   xSrc.push_back(i * 4u + 0u);
            xPts.push_back(F4{ o[9], o[10], o[11], cullScale });
            xSeed.push_back(F4{ 0.f, 0.f, 0.f, 0.f });   xSrc.push_back(i * 4u + 1u);

            // ---- from here on, the three reported results are the input ----
            const float startDistRaw = o[12];
            const PxVec3 dirPrevRaw = v3(o + 13);
            const float distanceRaw = o[16];
            const PxVec3 dirRaw = v3(o + 17);

            const uint32_t entered = asU(o[20]);
            const bool wantEntered = (distanceRaw < cullScale) || (startDistRaw < cullScale);
            if (entered != (wantEntered ? 1u : 0u))
                fail(i, "entered", float(entered), float(wantEntered ? 1 : 0));
            (entered ? st.enteredYes : st.enteredNo)++;

            const uint32_t branch = asU(o[30]);
            const uint32_t candidate = asU(o[53]);
            if (branch > 3u) { fail(i, "branch", float(branch), 0.f); continue; }
            st.branch[branch]++;
            if ((branch != 0u) != (candidate != 0u))
                fail(i, "candidate", float(candidate), float(branch != 0u ? 1 : 0));

            // With the flag off the re-cast is never entered, so every field it
            // would write must still hold its reset value.  This is what makes a
            // mutant that ignores enableCCD visible even in a configuration
            // where the re-cast would not have emitted a contact anyway.
            if (!ccd)
            {
                for (uint32_t k = 31; k <= 48; ++k)
                    if (!bitEq(o[k], 0.f))
                    {
                        fail(i, "a CCD trace field with the flag off", o[k], 0.f);
                        break;
                    }
            }

            if (!entered)
            {
                if (branch != 0u) fail(i, "branch without guard", float(branch), 0.f);
                if (!bitEq(o[49], 0.f) || !bitEq(o[50], 0.f) || !bitEq(o[51], 0.f))
                    fail(i, "normal.x on the unentered path", o[49], 0.f);
                if (!bitEq(o[52], 0.f)) fail(i, "finalSep on the unentered path", o[52], 0.f);
                continue;
            }

            // vertex2ShapeNormalVector IS shape2Vertex.
            PxVec3 eDir = hShape2Vertex(dirRaw, scale, scaleRot);
            PxVec3 eDirPrev = hShape2Vertex(dirPrevRaw, scale, scaleRot);
            float eDistance = distanceRaw, eStartDist = startDistRaw;
            float m = eDir.magnitudeSquared();                    // ENGINE
            if (m > 0.0f)
            {
                m = 1.0f / PxSqrt(m);                             // ENGINE
                eDistance = eDistance * m;
                eStartDist = eStartDist * m;
                eDir = eDir * m;
                eDirPrev = eDirPrev * m;
            }
            else
            {
                ++st.mZero;                                       // candidate 27's path
            }
            if (!bitEq(o[21], m)) fail(i, "m", o[21], m);
            if (!bitEq(o[22], eStartDist)) fail(i, "startDist", o[22], eStartDist);
            if (!bitEq(o[23], eDistance)) fail(i, "distance", o[23], eDistance);
            if (!bitEq3(o + 24, eDir)) fail(i, "dir.x", o[24], eDir.x);
            if (!bitEq3(o + 27, eDirPrev)) fail(i, "dirPrev.x", o[27], eDirPrev.x);
            if (asU(distanceRaw) == f2u(PX_MAX_F32)) ++st.rejectedDistance;

            // ---- the three decisions, in the engine's order ----
            PxVec3 eNormal(0.f, 0.f, 0.f);
            float eSep = 0.f;
            uint32_t eBranch = 0u;
            if (eStartDist < contactDist)
            {
                eNormal = eDirPrev;
                eSep = eStartDist - restDist;
                eBranch = 1u;
            }
            if (eBranch == 0u && eDistance > 0.f && eDistance < contactDist)
            {
                eNormal = eDir;
                eSep = (eDistance - restDist) + eNormal.dot(ePrev - ePos);   // ENGINE
                eBranch = 2u;
            }
            if (ccd && eBranch == 0u && eDistance < 0.f)
            {
                ++st.ccdReached;
                const PxVec3 d = (ePos - ePrev).getNormalized();             // ENGINE
                if (eStartDist - restDist < 0.f) ++st.clampFired;
                const float displacement = -PxMax(0.f, eStartDist - restDist) / d.dot(eDir);
                const PxVec3 pt = ePrev + d * displacement;
                // Candidate 26: the port transcribes the double transform, so
                // the mirror must too.
                const PxVec3 ccdVtx = hShape2Vertex(aToB.transform(pt), scale, scaleRot);
                if (!bitEq3(o + 31, d)) fail(i, "ccdD.x", o[31], d.x);
                if (!bitEq(o[34], displacement)) fail(i, "ccdDisplacement", o[34], displacement);
                if (!bitEq3(o + 35, pt)) fail(i, "ccdPt.x", o[35], pt.x);
                if (!bitEq3(o + 38, ccdVtx)) fail(i, "ccdVtx.x", o[38], ccdVtx.x);
                // THE CCD CALL IS NOT MADE WITH A FRESH `dir`.  It reuses the one
                // the main block left behind, and PxSdfSampleWithGrad's reject
                // path (C9 in pxgSlangSdfCollision) does not touch it -- so when
                // the re-cast sample is rejected the SURVIVING value is the
                // earlier direction, not zero.  Seeding the reference with
                // anything else turns that behaviour into a false failure;
                // seeding it correctly turns it into a check, and it is the only
                // place in this fixture where C9 is observed from outside.
                xPts.push_back(F4{ o[38], o[39], o[40], cullScale });
                xSeed.push_back(F4{ o[24], o[25], o[26], 0.f });
                xSrc.push_back(i * 4u + 2u);

                float ccdDistance = o[41];                 // REPORTED
                PxVec3 ccdDir = hShape2Vertex(v3(o + 42), scale, scaleRot);
                if (m > 0.0f)
                {
                    ccdDistance = ccdDistance * m;
                    ccdDir = ccdDir * m;
                }
                if (!bitEq(o[45], ccdDistance)) fail(i, "ccdDistance", o[45], ccdDistance);
                if (!bitEq3(o + 46, ccdDir)) fail(i, "ccdDir.x", o[46], ccdDir.x);
                if (ccdDistance < contactDist)
                {
                    eNormal = ccdDir;
                    eSep = (ccdDistance - restDist) + eNormal.dot(ePrev - pt);
                    eBranch = 3u;
                }
            }

            if (branch != eBranch) fail(i, "branch", float(branch), float(eBranch));
            if (!bitEq3(o + 49, eNormal)) fail(i, "normal.x", o[49], eNormal.x);
            if (!bitEq(o[52], eSep)) fail(i, "finalSep", o[52], eSep);
        }
        // ---- the independent re-evaluation ----
        // npSdfCollision's sdfCollVertex is doVertexSDFCollision with the same
        // arguments this module hands it: the same point, the tolerance
        // cullScale, and a `dir` seeded with zero.  Its result must be the same
        // bits.  A mutant that passes a different tolerance, or seeds `dir`
        // differently, changes what this module reports but NOT what the
        // reference returns -- which is exactly what the first sweep's two
        // unobservable survivors needed.
        if (!xPts.empty())
        {
            const uint32_t M = uint32_t(xPts.size());
            const std::vector<float> xInit(size_t(M) * STRIDE_VTX, -999.f);
            const PxgBufferHandle bXP = keep(createBufferFrom(dev, xPts));
            const PxgBufferHandle bXS = keep(createBufferFrom(dev, xSeed));
            const PxgBufferHandle bXO = keep(createBufferFrom(dev, xInit));
            P_Coll xp = {};
            xp.meshPtr = blobAddr;
            xp.points = dev->getBufferDeviceAddress(bXP);
            xp.gradIn = dev->getBufferDeviceAddress(bXS);
            xp.dirSeed = dev->getBufferDeviceAddress(bXS);
            xp.outBuf = dev->getBufferDeviceAddress(bXO);
            xp.numPoints = M;
            {
                Kernel k(Kernel::Bindless{}, "npSdfCollision.spv", "sdfCollVertex", {}, sizeof(P_Coll));
                PXG_CHECK_MSG(k.isValid(), "%s: the cross-check pipeline must build", label);
                if (k.isValid())
                {
                    k.dispatch((M + 63u) / 64u, {}, xp);
                    dev->flush();
                    const std::vector<float> X = downloadBuffer<float>(dev, bXO, size_t(M) * STRIDE_VTX);
                    for (uint32_t j = 0; j < M; ++j)
                    {
                        const uint32_t i = xSrc[j] >> 2, kind = xSrc[j] & 3u;
                        const float* x = &X[size_t(j) * STRIDE_VTX];
                        const float* o = &R[size_t(i) * STRIDE];
                        const uint32_t distSlot = (kind == 0u) ? 16u : (kind == 1u) ? 12u : 41u;
                        const uint32_t dirSlot  = (kind == 0u) ? 17u : (kind == 1u) ? 13u : 42u;
                        ++st.crossChecked;
                        if (!bitEq(o[distSlot], x[11]))
                            fail(i, (kind == 0u) ? "distance vs the reference"
                                 : (kind == 1u) ? "startDist vs the reference"
                                                : "ccdDistance vs the reference",
                                 o[distSlot], x[11]);
                        if (!bitEq(o[dirSlot], x[8]) || !bitEq(o[dirSlot + 1], x[9]) ||
                            !bitEq(o[dirSlot + 2], x[10]))
                            fail(i, "a direction vs the reference", o[dirSlot], x[8]);
                    }
                }
            }
        }

        PXG_CHECK_MSG(st.bad == 0, "%s: %zu bit-exact comparisons failed", label, st.bad);
        std::printf("    [ note ] %s: guard %zu/%zu, branches %zu/%zu/%zu/%zu, "
                    "%zu rejected current samples, %zu with m == 0, %zu reaching the CCD re-cast, "
                    "%zu clamp hits, %zu samples cross-checked\n",
                    label, st.enteredYes, st.enteredNo, st.branch[0], st.branch[1], st.branch[2],
                    st.branch[3], st.rejectedDistance, st.mZero, st.ccdReached, st.clampFired,
                    st.crossChecked);
        return st;
    };

    // A: the real pose, CCD on.  B: the same, CCD off.  C: an IDENTITY pose,
    // CCD on -- see below for why the third one has to exist.
    const Stats A = runAndCheck("posed+ccd", aToB, true, restDistSmall, 0x5DF9A27u);
    const Stats B = runAndCheck("posed-noccd", aToB, false, restDistSmall, 0x5DF9A27u);
    const Stats C = runAndCheck("identity+ccd", PxTransform(PxIdentity), true,
                                restDistSmall, 0x71C0DEu);
    // D exists so that "the CCD flag is read" has a configuration in which
    // ignoring it would CHANGE something: under the posed pose the re-cast never
    // emits (candidate 26), so a mutant that drops the flag there is invisible.
    const Stats D = runAndCheck("identity-noccd", PxTransform(PxIdentity), false,
                                restDistSmall, 0x71C0DEu);
    // E turns the rest distance above the contact distance, which is the only
    // way the CCD displacement's PxMax(0, .) clamp is ever reached.
    const Stats E = runAndCheck("identity+ccd+restgt", PxTransform(PxIdentity), true,
                                restDistLarge, 0x71C0DEu);

    // ---- coverage, required rather than hoped for ----
    PXG_CHECK_MSG(A.enteredYes > 0 && A.enteredNo > 0,
                  "both outcomes of the cull guard must occur (%zu entered, %zu not)",
                  A.enteredYes, A.enteredNo);
    for (uint32_t b = 0; b < 3u; ++b)
        PXG_CHECK_MSG(A.branch[b] > 0, "contact branch %u never occurred", b);
    PXG_CHECK_MSG(A.rejectedDistance > 0,
                  "no pair had its current-position sample REJECTED while the guard was still "
                  "entered -- that is upstream candidate 27's path and it must be exercised");
    PXG_CHECK_MSG(A.mZero > 0,
                  "no pair reached the guard with m == 0 -- the zeroed-direction divergence is "
                  "then untested");
    PXG_CHECK_MSG(A.ccdReached > 0, "no pair reached the CCD re-cast at all");

    PXG_CHECK_MSG(A.crossChecked > 0,
                  "no sample was re-evaluated independently -- the tolerance and the direction "
                  "seed are then unobservable");

    // The CCD flag is read.  B alone does not prove it: under the posed pose the
    // re-cast never emits, so ignoring the flag there changes only trace fields.
    // D is the same check under an identity pose, where the re-cast DOES emit --
    // that is the configuration in which the flag is load-bearing.
    PXG_CHECK_MSG(B.ccdReached == 0 && B.branch[3] == 0,
                  "with enableCCD off the re-cast must be unreachable (%zu reached, %zu contacts)",
                  B.ccdReached, B.branch[3]);
    PXG_CHECK_MSG(D.ccdReached == 0 && D.branch[3] == 0,
                  "identity+noccd: the re-cast must be unreachable (%zu reached, %zu contacts)",
                  D.ccdReached, D.branch[3]);
    PXG_CHECK_MSG(C.branch[3] > D.branch[3],
                  "the identity pose must emit CCD contacts with the flag ON (%zu) and none with "
                  "it off (%zu) -- otherwise the flag is not proven to be read",
                  C.branch[3], D.branch[3]);

    // The displacement's PxMax(0, .) clamp is dead while restDist < contactDist.
    PXG_CHECK_MSG(E.clampFired > 0,
                  "the CCD displacement clamp never fired even with restDist (%g) above "
                  "contactDist (%g) -- it is then untested", restDistLarge, contactDist);
    PXG_CHECK_MSG(C.clampFired == 0,
                  "the clamp must be DEAD when restDist < contactDist (%zu hits) -- if it fires "
                  "there, this fixture's reasoning about the branch guard is wrong", C.clampFired);

    // WHY CONFIGURATION C EXISTS.  Under the posed configuration the CCD re-cast
    // is reached but almost never produces a contact, and that is not a fixture
    // weakness -- it is upstream candidate 26 showing itself.  The re-cast
    // samples at particleToTrimeshTransform.transform(pt) although `pt` is
    // ALREADY in that transform's output space, so for a non-identity pose it
    // looks up the SDF somewhere else entirely and the sample is rejected.  With
    // an identity pose the double transform is a no-op and the branch works, so
    // that is the only configuration in which its arithmetic can be exercised.
    // Measured here: posed reaches the re-cast %zu times and emits A.branch[3];
    // identity emits C.branch[3].
    PXG_CHECK_MSG(C.branch[3] > 0,
                  "the CCD re-cast must produce contacts under an IDENTITY pose (it reached the "
                  "re-cast %zu times and emitted %zu) -- without that its arithmetic is untested",
                  C.ccdReached, C.branch[3]);
    std::printf("    [ note ] candidate 26 measured: the posed configuration reaches the CCD "
                "re-cast %zu times and emits %zu contacts; the identity configuration reaches it "
                "%zu times and emits %zu\n",
                A.ccdReached, A.branch[3], C.ccdReached, C.branch[3]);

    for (PxgBufferHandle h : owned)
        dev->destroyBuffer(h);
    physx::gpuslang::pxgSetVulkanDriverShim(nullptr);
}

} // namespace
