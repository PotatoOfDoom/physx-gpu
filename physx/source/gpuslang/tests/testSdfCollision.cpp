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

// testSdfCollision.cpp -- P9.12d, the fixture for pxgSlangSdfCollision.
//
// ---------------------------------------------------------------------------
// HOW THIS IS CHECKED, given that Sample() cannot run on the host
// ---------------------------------------------------------------------------
// Every expectation below is BIT-EXACT.  There is not a tolerance in the whole
// composition check, and that is possible because of two arrangements:
//
//  (1) EVERY PRIMITIVE IS THE ENGINE'S OWN.  PxVec3::maximum / minimum /
//      magnitude / getNormalized / normalize / dot and PxAbs are all
//      PX_CUDA_CALLABLE, so the host expectation calls them rather than
//      re-deriving them.  What is transcribed from sdfCollision.cuh is the
//      ORDER, nothing else -- the P8.87 arrangement, but with a far larger
//      fraction of the content on the engine's side of the line than that
//      tranche had.
//
//  (2) THE SHADER REPORTS THE SAMPLES IT READ.  Sample() lives on the GPU, so
//      no host expectation predicts a sample value.  The *Ex entry points write
//      back every intermediate -- the clamped point, the texel coordinate, the
//      six gradient taps and their six values, the raw and normalised gradient
//      -- and the host checks the COMPOSITION against those.  `grad.x` must
//      equal `value[0] - value[1]` to the bit, whatever those values are.
//
// That leaves exactly one hole, and it is closed rather than argued away: a
// shader could report values it did not read.  So every tap value is
// CROSS-CHECKED against an independent dispatch -- the interior taps against
// npSdfTexture's sdfTexSampleAt (a DIFFERENT module, P9.12c's, over the same
// blob and the same textures), the fall-back taps against sdfCollDistance at
// those same world points.  A mutant that differences the wrong two taps, or
// samples at a coordinate other than the one it reports, dies on that.
//
// Sample() ITSELF is not this tranche's claim.  It is P9.12c's, and it was
// measured there against the engine's own Gu::SDF::decodeSparse.  What is
// re-checked here, cheaply, is that the values coming back are that function's
// and not noise: on the grid-aligned dense scene the reported sample at an
// integer texel coordinate must be the stored texel BIT-EXACTLY, and on the
// sparse scene it must agree with sparseOracle inside P9.12a's measured filter
// bounds.
//
// ---------------------------------------------------------------------------
// THE TWO SCENES, and why the second one exists
// ---------------------------------------------------------------------------
// SPARSE -- pxgSdfSceneRef.h's 12 x 8 x 20 / subgrid 4 scene, spacing 0.11 and
//   a meshLower of no particular niceness.  It is here because it is the only
//   one that drives Sample()'s RESIDENT and COARSE-FALL-BACK branches, and
//   because a helper layer that only ever ran over a dense SDF would not have
//   been tested against the shape the engine actually cooks.
//
// GRID-ALIGNED DENSE -- 9 x 6 x 13, spacing 0.25, meshLower on eighths.  New in
//   this tranche, and the numbers are chosen so that the texel coordinate f is
//   EXACTLY an integer for a point placed at a texel centre: 0.25 is a power of
//   two, so `(lower + n*0.25 - lower) * 4.0` is exact for every n in range.
//   Three things need that and cannot get it from the sparse scene:
//     - PxVolumeGrad's guard is `f >= 1 && f <= dims - 2`, INCLUSIVE.  Only an
//       exact f can sit ON either bound, and only then do `>` and `<` mutants
//       differ from `>=` and `<=`.
//     - the sample cross-check is bit-exact rather than filtered.
//     - the DISCONTINUITY path.  doVertexSDFCollision substitutes an arbitrary
//       direction when the gradient's magnitude is EXACTLY zero, and the only
//       way to produce that is for three pairs of taps to return identical
//       values -- which needs the taps to land on texel centres.  This scene's
//       texture has the six neighbours of texel (3,2,4) written in three equal
//       PAIRS for exactly that reason; it is the "exact center of a cube" the
//       CUDA's comment names.

#include "pxgTestFramework.h"

#include "PxgCudaShim.h"
#include "PxgCudaToolkitShim.h"
#include "pxgSdfSceneRef.h"
#include "GuDistancePointTriangle.h"   // the ENGINE'S OWN closestPtPointBaryTriangle (P9.34a)

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

// npSdfCollision.slang's single push-constant block.
struct P_Coll
{
    uint64_t meshPtr, points, gradIn, dirSeed, outBuf;
    uint32_t numPoints, pad;
    uint64_t triVerts, triOut;      // P9.34a
};

// npSdfTexture.slang's, used for the independent tap cross-check.
struct P_Tex
{
    uint64_t meshPtr, outMesh, outTex, coords, outValue;
    uint32_t numSamples, pad;
};

static const uint32_t STRIDE_DIST = 12u;
static const uint32_t STRIDE_GRAD = 32u;
static const uint32_t STRIDE_SG   = 12u;
static const uint32_t STRIDE_VTX  = 16u;

static bool bitEq(float a, float b) { return f2u(a) == f2u(b); }
static bool bitEq3(const float* a, const PxVec3& b)
{ return bitEq(a[0], b.x) && bitEq(a[1], b.y) && bitEq(a[2], b.z); }

static PxVec3 v3(const float* a) { return PxVec3(a[0], a[1], a[2]); }

// ---------------------------------------------------------------------------
// The grid-aligned dense scene.
// ---------------------------------------------------------------------------
static const uint32_t FLAT_X = 3u, FLAT_Y = 2u, FLAT_Z = 4u;

static size_t denseIdx(const SdfScene& s, uint32_t x, uint32_t y, uint32_t z)
{ return (size_t(z) * s.dy + y) * s.dx + x; }

static void buildDenseAligned(SdfScene& s)
{
    s.dx = 9; s.dy = 6; s.dz = 13;          // no two equal, and all distinct from
    s.sg = 0;                               // the sparse scene's 12 / 8 / 20
    s.nbx = s.nby = s.nbz = 0;
    s.blkx = s.blky = s.blkz = 0;
    s.smin = -0.19f; s.smax = 0.53f;        // unused on the dense path
    s.meshLower = PxVec3(-1.0f, 0.5f, 2.25f);
    s.spacing = 0.25f;                      // a power of two: see the file header
    std::mt19937 rng(0xA11 * 0x9E37u + 5u);
    s.coarse.resize(size_t(s.dx) * s.dy * s.dz);
    for (auto& v : s.coarse)
        v = -1.75f + 3.5f * float(rng() % 1000000u) / 1000000.f;
    s.starts.clear();
    s.sub.clear();

    // The discontinuity.  PxVolumeGrad differences the SIX FACE NEIGHBOURS of
    // the sample texel, so making those three PAIRS equal makes all three
    // components exactly zero -- and the centre texel is deliberately left
    // alone, because the gradient never reads it and a port that did would then
    // be visible.  The three pair values are different from each other so that
    // this is a flat GRADIENT and not a flat REGION.
    s.coarse[denseIdx(s, FLAT_X + 1, FLAT_Y, FLAT_Z)] = 0.4375f;
    s.coarse[denseIdx(s, FLAT_X - 1, FLAT_Y, FLAT_Z)] = 0.4375f;
    s.coarse[denseIdx(s, FLAT_X, FLAT_Y + 1, FLAT_Z)] = -0.8125f;
    s.coarse[denseIdx(s, FLAT_X, FLAT_Y - 1, FLAT_Z)] = -0.8125f;
    s.coarse[denseIdx(s, FLAT_X, FLAT_Y, FLAT_Z + 1)] = 1.1875f;
    s.coarse[denseIdx(s, FLAT_X, FLAT_Y, FLAT_Z - 1)] = 1.1875f;
}

// ---------------------------------------------------------------------------
// The host mirror.  Control flow from sdfCollision.cuh; every primitive is the
// engine's own PX_CUDA_CALLABLE original, CALLED.
// ---------------------------------------------------------------------------
static PxVec3 hClampToBox(const HostSdfBox& t, const PxVec3& p)
{
    return p.maximum(t.lower).minimum(t.higher);      // C1 + C2, ENGINE
}

static PxVec3 hTexCoord(const HostSdfBox& t, const PxVec3& clamped)
{
    return (clamped - t.lower) * t.invDx;             // C4, ENGINE
}

static bool hInterior(const HostSdfBox& t, const PxVec3& f)
{
    // C7: inclusive on both ends, ANDed, against the FINE dimensions.
    return f.x >= 1.0f && f.x <= t.dims[0] - 2.0f &&
           f.y >= 1.0f && f.y <= t.dims[1] - 2.0f &&
           f.z >= 1.0f && f.z <= t.dims[2] - 2.0f;
}

// The point at fine-grid coordinate `g`, exact whenever the scene is
// grid-aligned.
static PxVec3 posOfF(const HostSdfBox& t, float gx, float gy, float gz)
{
    return t.lower + PxVec3(gx * t.dx, gy * t.dx, gz * t.dx);
}

// A test point plus, when it sits on a texel centre, the integer coordinate the
// sample cross-check uses.
struct Pt
{
    PxVec3 p;
    int gx, gy, gz;      // -1 == not on a texel centre
    const char* name;
};

// ---------------------------------------------------------------------------
// One scene, all four entry points.
// ---------------------------------------------------------------------------
struct SceneRun
{
    const char* label;
    const SdfScene* s;
    HostSdfBox box;
    uint64_t blobAddr;
    bool expectDiscontinuity;      // exactly one point must produce a zero gradient
    std::vector<Pt> pts;
};

static void checkScene(PxgSlangComputeDevice* dev, SceneRun& r,
                       std::vector<PxgBufferHandle>& owned)
{
    const uint32_t N = uint32_t(r.pts.size());
    const HostSdfBox& box = r.box;

    auto keep = [&](PxgBufferHandle h) { owned.push_back(h); return h; };

    // ---- the host-side expectations that need no dispatch ----
    std::vector<PxVec3> wantClamped(N), wantF, wantDiff(N);
    std::vector<float> wantDiffMag(N);
    std::vector<int> wantInterior(N);
    wantF.resize(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        wantClamped[i] = hClampToBox(box, r.pts[i].p);
        wantDiff[i] = r.pts[i].p - wantClamped[i];
        wantDiffMag[i] = wantDiff[i].magnitude();                 // ENGINE
        wantF[i] = hTexCoord(box, wantClamped[i]);
        wantInterior[i] = hInterior(box, wantF[i]) ? 1 : 0;
    }

    // =====================================================================
    // Pass 1: PxSdfDistance.  Tolerances are still PX_MAX_F32 here -- the
    // sample values this pass reports are what the later passes' tolerance
    // cases are built from, which is the only way to place one EXACTLY on the
    // `dist < tolerance` boundary.
    // =====================================================================
    std::vector<F4> ptsBuf(N);
    for (uint32_t i = 0; i < N; ++i)
        ptsBuf[i] = F4{ r.pts[i].p.x, r.pts[i].p.y, r.pts[i].p.z, PX_MAX_F32 };
    std::vector<F4> gradBuf(N), dirBuf(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        // Five gradient shapes, cycled: a generic non-unit vector, the exact
        // zero (getNormalized returns zero, so dir falls back to diff alone), a
        // unit vector, a vector so short that its magnitudeSquared UNDERFLOWS to
        // zero -- non-zero input, zero output, and the engine does the same --
        // and a negative axis.
        switch (i % 5u)
        {
        case 0: gradBuf[i] = F4{  0.3f, -0.7f,  1.9f, 0.f }; break;
        case 1: gradBuf[i] = F4{  0.0f,  0.0f,  0.0f, 0.f }; break;
        case 2: gradBuf[i] = F4{  0.6f,  0.8f,  0.0f, 0.f }; break;
        case 3: gradBuf[i] = F4{ 1e-25f, 0.0f,  0.0f, 0.f }; break;
        default: gradBuf[i] = F4{ -2.5f, 0.0f,  0.0f, 0.f }; break;
        }
        // A per-point sentinel, so "dir was left alone" is unambiguous and a
        // cross-lane copy would be visible too.
        dirBuf[i] = F4{ -9.5f - float(i), 3.25f + float(i), 77.0f + float(i), 0.f };
    }

    const PxgBufferHandle bPts = keep(createBufferFrom(dev, ptsBuf));
    const PxgBufferHandle bGrad = keep(createBufferFrom(dev, gradBuf));
    const PxgBufferHandle bDir = keep(createBufferFrom(dev, dirBuf));

    std::vector<float> outInit(size_t(N) * STRIDE_GRAD, -12345.f);
    const PxgBufferHandle bOutD = keep(createBufferFrom(dev, outInit));
    const PxgBufferHandle bOutG = keep(createBufferFrom(dev, outInit));
    const PxgBufferHandle bOutS = keep(createBufferFrom(dev, outInit));
    const PxgBufferHandle bOutV = keep(createBufferFrom(dev, outInit));

    P_Coll prm = {};
    prm.meshPtr = r.blobAddr;
    prm.points = dev->getBufferDeviceAddress(bPts);
    prm.gradIn = dev->getBufferDeviceAddress(bGrad);
    prm.dirSeed = dev->getBufferDeviceAddress(bDir);
    prm.numPoints = N;

    auto run = [&](const char* entry, PxgBufferHandle out) {
        P_Coll p = prm;
        p.outBuf = dev->getBufferDeviceAddress(out);
        Kernel k(Kernel::Bindless{}, "npSdfCollision.spv", entry, {}, sizeof(P_Coll));
        PXG_CHECK_MSG(k.isValid(), "%s: bindless pipeline creation succeeds for %s", r.label, entry);
        if (!k.isValid()) return false;
        k.dispatch((N + 63) / 64, {}, p);
        return true;
    };

    if (!run("sdfCollDistance", bOutD)) return;
    dev->flush();
    const auto D = downloadBuffer<float>(dev, bOutD, size_t(N) * STRIDE_GRAD);

    size_t badClamp = 0, badF = 0, badMag = 0, badSum = 0, badSample = 0;
    size_t nOutside = 0, nInside = 0, nExactTexel = 0;
    double worstSample = 0.0;
    // P9.12a's measured filter bounds, reused: a FLOAT coarse texture is good to
    // ~3e-06 over a range of 8, an 8-bit UNORM subgrid to half a quantisation
    // step of its own range.
    const double sparseBound = (r.s->sg == 0u) ? 0.0
                             : 0.5 * double(r.s->smax - r.s->smin) / 255.0 + 1e-4;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* o = &D[size_t(i) * STRIDE_DIST];
        if (!bitEq3(o + 0, wantClamped[i]))
        {
            if (badClamp < 4)
                std::printf("    %s %s: clamped (%.9g,%.9g,%.9g) want (%.9g,%.9g,%.9g)\n",
                            r.label, r.pts[i].name, o[0], o[1], o[2],
                            wantClamped[i].x, wantClamped[i].y, wantClamped[i].z);
            ++badClamp;
        }
        if (!bitEq3(o + 3, wantF[i]))
        {
            if (badF < 4)
                std::printf("    %s %s: f (%.9g,%.9g,%.9g) want (%.9g,%.9g,%.9g)\n",
                            r.label, r.pts[i].name, o[3], o[4], o[5],
                            wantF[i].x, wantF[i].y, wantF[i].z);
            ++badF;
        }
        if (!bitEq(o[6], wantDiffMag[i]))
        {
            if (badMag < 4)
                std::printf("    %s %s: diffMag %.9g want %.9g\n",
                            r.label, r.pts[i].name, o[6], wantDiffMag[i]);
            ++badMag;
        }
        // C3: the return is Sample(f) PLUS diffMag, from the shader's own two
        // reported halves -- so this survives whatever Sample returns.
        if (!bitEq(o[8], o[7] + o[6]))
        {
            if (badSum < 4)
                std::printf("    %s %s: PxSdfDistance %.9g != sample %.9g + diffMag %.9g\n",
                            r.label, r.pts[i].name, o[8], o[7], o[6]);
            ++badSum;
        }
        if (wantDiffMag[i] > 0.0f) ++nOutside; else ++nInside;

        // Is the reported sample really the SDF's?
        if (r.pts[i].gx >= 0)
        {
            ++nExactTexel;
            if (r.s->sg == 0u)
            {
                const float want = r.s->coarse[denseIdx(*r.s, uint32_t(r.pts[i].gx),
                                                        uint32_t(r.pts[i].gy),
                                                        uint32_t(r.pts[i].gz))];
                if (!bitEq(o[7], want))
                {
                    if (badSample < 4)
                        std::printf("    %s %s: sample %.9g != stored texel %.9g\n",
                                    r.label, r.pts[i].name, o[7], want);
                    ++badSample;
                }
            }
            else
            {
                const float want = sparseOracle(*r.s, r.pts[i].gx, r.pts[i].gy, r.pts[i].gz);
                const double d = std::fabs(double(o[7]) - double(want));
                if (d > worstSample) worstSample = d;
                if (d > sparseBound)
                {
                    if (badSample < 4)
                        std::printf("    %s %s: sample %.9g vs oracle %.9g (diff %.3g)\n",
                                    r.label, r.pts[i].name, o[7], want, d);
                    ++badSample;
                }
            }
        }
    }
    PXG_CHECK_MSG(badClamp == 0, "%s: %zu clamped points differ from PxVec3::maximum().minimum()",
                  r.label, badClamp);
    PXG_CHECK_MSG(badF == 0, "%s: %zu texel coordinates differ from (clamped - lower) * invSdfDx",
                  r.label, badF);
    PXG_CHECK_MSG(badMag == 0, "%s: %zu diffMag values differ from PxVec3::magnitude()",
                  r.label, badMag);
    PXG_CHECK_MSG(badSum == 0, "%s: %zu PxSdfDistance returns are not Sample(f) + diffMag",
                  r.label, badSum);
    PXG_CHECK_MSG(badSample == 0, "%s: %zu reported samples are not the SDF's own values",
                  r.label, badSample);
    PXG_CHECK_MSG(nOutside > 0 && nInside > 0,
                  "%s: the point list must contain points BOTH inside and outside the box "
                  "(%zu outside, %zu inside) -- diffMag is dead weight otherwise",
                  r.label, nOutside, nInside);
    PXG_CHECK_MSG(nExactTexel > 0, "%s: no point lands on a texel centre, so no sample is "
                  "cross-checked against the SDF", r.label);

    // =====================================================================
    // Pass 2: PxVolumeGrad.
    // =====================================================================
    if (!run("sdfCollGrad", bOutG)) return;
    dev->flush();
    const auto G = downloadBuffer<float>(dev, bOutG, size_t(N) * STRIDE_GRAD);

    size_t badBranch = 0, badCoord = 0, badGrad = 0, badGf = 0;
    size_t nInterior = 0, nFallback = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* o = &G[size_t(i) * STRIDE_GRAD];
        uint32_t interior; std::memcpy(&interior, &o[0], 4);
        if (int(interior) != wantInterior[i])
        {
            if (badBranch < 4)
                std::printf("    %s %s: interior flag %u want %d (f = %.9g,%.9g,%.9g)\n",
                            r.label, r.pts[i].name, interior, wantInterior[i],
                            wantF[i].x, wantF[i].y, wantF[i].z);
            ++badBranch;
        }
        if (!bitEq3(o + 4, wantF[i])) ++badGf;
        (wantInterior[i] ? nInterior : nFallback)++;

        // C5: on the interior path the taps are TEXEL coordinates, on the
        // fall-back they are WORLD points.  The two spaces are wildly different
        // numbers, which is what makes a guard mutant visible here and not only
        // in the flag above.
        PxVec3 want[6];
        if (wantInterior[i])
        {
            const PxVec3 f = wantF[i];
            want[0] = PxVec3(f.x + 1.0f, f.y, f.z);
            want[1] = PxVec3(f.x - 1.0f, f.y, f.z);
            want[2] = PxVec3(f.x, f.y + 1.0f, f.z);
            want[3] = PxVec3(f.x, f.y - 1.0f, f.z);
            want[4] = PxVec3(f.x, f.y, f.z + 1.0f);
            want[5] = PxVec3(f.x, f.y, f.z - 1.0f);
        }
        else
        {
            const PxVec3 p = r.pts[i].p;
            want[0] = p + PxVec3(box.dx, 0.f, 0.f);
            want[1] = p - PxVec3(box.dx, 0.f, 0.f);
            want[2] = p + PxVec3(0.f, box.dx, 0.f);
            want[3] = p - PxVec3(0.f, box.dx, 0.f);
            want[4] = p + PxVec3(0.f, 0.f, box.dx);
            want[5] = p - PxVec3(0.f, 0.f, box.dx);
        }
        for (int k = 0; k < 6; ++k)
            if (!bitEq3(o + 7 + k * 3, want[k]))
            {
                if (badCoord < 6)
                    std::printf("    %s %s: tap %d (%.9g,%.9g,%.9g) want (%.9g,%.9g,%.9g)\n",
                                r.label, r.pts[i].name, k, o[7 + k * 3], o[8 + k * 3],
                                o[9 + k * 3], want[k].x, want[k].y, want[k].z);
                ++badCoord;
            }

        // C6: a plain difference of the two taps, no division by the step.
        const float wgx = o[25] - o[26];
        const float wgy = o[27] - o[28];
        const float wgz = o[29] - o[30];
        if (!bitEq(o[1], wgx) || !bitEq(o[2], wgy) || !bitEq(o[3], wgz))
        {
            if (badGrad < 4)
                std::printf("    %s %s: grad (%.9g,%.9g,%.9g) != tap differences "
                            "(%.9g,%.9g,%.9g)\n", r.label, r.pts[i].name,
                            o[1], o[2], o[3], wgx, wgy, wgz);
            ++badGrad;
        }
    }
    PXG_CHECK_MSG(badBranch == 0, "%s: %zu points take the wrong PxVolumeGrad branch", r.label, badBranch);
    PXG_CHECK_MSG(badGf == 0, "%s: %zu grad-entry texel coordinates disagree with the distance entry's",
                  r.label, badGf);
    PXG_CHECK_MSG(badCoord == 0, "%s: %zu gradient taps are at the wrong coordinate", r.label, badCoord);
    PXG_CHECK_MSG(badGrad == 0, "%s: %zu gradients are not the plain difference of their own taps "
                  "(a division by the step would show here)", r.label, badGrad);
    PXG_CHECK_MSG(nInterior > 0 && nFallback > 0,
                  "%s: BOTH PxVolumeGrad paths must be reached (%zu interior, %zu fall-back)",
                  r.label, nInterior, nFallback);

    // ---- the tap values, against an independent evaluation ----
    {
        std::vector<F4> texTaps;      // interior taps -> npSdfTexture (a DIFFERENT module)
        std::vector<uint32_t> texSrc;
        std::vector<F4> worldTaps;    // fall-back taps -> sdfCollDistance
        std::vector<uint32_t> worldSrc;
        for (uint32_t i = 0; i < N; ++i)
        {
            const float* o = &G[size_t(i) * STRIDE_GRAD];
            for (int k = 0; k < 6; ++k)
            {
                const F4 c = F4{ o[7 + k * 3], o[8 + k * 3], o[9 + k * 3], PX_MAX_F32 };
                if (wantInterior[i]) { texTaps.push_back(c); texSrc.push_back(i * 6u + uint32_t(k)); }
                else                 { worldTaps.push_back(c); worldSrc.push_back(i * 6u + uint32_t(k)); }
            }
        }
        size_t badTap = 0;
        if (!texTaps.empty())
        {
            const uint32_t M = uint32_t(texTaps.size());
            std::vector<float> init(M, -777.f);
            const PxgBufferHandle bC = keep(createBufferFrom(dev, texTaps));
            const PxgBufferHandle bO = keep(createBufferFrom(dev, init));
            P_Tex tp = {};
            tp.meshPtr = r.blobAddr;
            tp.coords = dev->getBufferDeviceAddress(bC);
            tp.outValue = dev->getBufferDeviceAddress(bO);
            tp.numSamples = M;
            {
                Kernel k(Kernel::Bindless{}, "npSdfTexture.spv", "sdfTexSampleAt", {}, sizeof(P_Tex));
                if (!k.isValid()) return;
                k.dispatch((M + 63) / 64, {}, tp);
            }
            dev->flush();
            const auto got = downloadBuffer<float>(dev, bO, M);
            for (uint32_t j = 0; j < M; ++j)
            {
                const uint32_t i = texSrc[j] / 6u, kk = texSrc[j] % 6u;
                const float reported = G[size_t(i) * STRIDE_GRAD + 25 + kk];
                if (!bitEq(reported, got[j]))
                {
                    if (badTap < 5)
                        std::printf("    %s %s tap %u: reported %.9g, npSdfTexture says %.9g\n",
                                    r.label, r.pts[i].name, kk, reported, got[j]);
                    ++badTap;
                }
            }
        }
        if (!worldTaps.empty())
        {
            const uint32_t M = uint32_t(worldTaps.size());
            std::vector<float> init(size_t(M) * STRIDE_DIST, -777.f);
            const PxgBufferHandle bC = keep(createBufferFrom(dev, worldTaps));
            const PxgBufferHandle bO = keep(createBufferFrom(dev, init));
            P_Coll p = prm;
            p.points = dev->getBufferDeviceAddress(bC);
            p.outBuf = dev->getBufferDeviceAddress(bO);
            p.numPoints = M;
            {
                Kernel k(Kernel::Bindless{}, "npSdfCollision.spv", "sdfCollDistance", {}, sizeof(P_Coll));
                if (!k.isValid()) return;
                k.dispatch((M + 63) / 64, {}, p);
            }
            dev->flush();
            const auto got = downloadBuffer<float>(dev, bO, size_t(M) * STRIDE_DIST);
            for (uint32_t j = 0; j < M; ++j)
            {
                const uint32_t i = worldSrc[j] / 6u, kk = worldSrc[j] % 6u;
                const float reported = G[size_t(i) * STRIDE_GRAD + 25 + kk];
                const float actual = got[size_t(j) * STRIDE_DIST + 8];
                if (!bitEq(reported, actual))
                {
                    if (badTap < 5)
                        std::printf("    %s %s tap %u: reported %.9g, PxSdfDistance says %.9g\n",
                                    r.label, r.pts[i].name, kk, reported, actual);
                    ++badTap;
                }
            }
        }
        PXG_CHECK_MSG(badTap == 0, "%s: %zu reported tap values are not what an independent "
                      "evaluation at the same coordinate returns", r.label, badTap);
    }

    // =====================================================================
    // Pass 3: PxSdfSampleWithGrad, now with real tolerances.
    // =====================================================================
    // Three of the points get a tolerance derived from the sample they just
    // reported.  One of them sits EXACTLY on it, which is the only way to tell
    // `dist < tolerance` from `dist <= tolerance`.
    const uint32_t iEq = 2u % N, iLo = 5u % N, iHi = 7u % N;
    ptsBuf[iEq].w = D[size_t(iEq) * STRIDE_DIST + 7];                  // == dist -> REJECT
    ptsBuf[iLo].w = D[size_t(iLo) * STRIDE_DIST + 7] - 0.0625f;        // <  dist -> REJECT
    ptsBuf[iHi].w = D[size_t(iHi) * STRIDE_DIST + 7] + 0.0625f;        // >  dist -> accept
    {
        const PxgBufferHandle b2 = keep(createBufferFrom(dev, ptsBuf));
        prm.points = dev->getBufferDeviceAddress(b2);
    }

    if (!run("sdfCollSampleGrad", bOutS)) return;
    dev->flush();
    const auto S = downloadBuffer<float>(dev, bOutS, size_t(N) * STRIDE_GRAD);

    size_t badDist = 0, badDiff = 0, badDir = 0, badRet = 0, badKeep = 0;
    size_t nAccept = 0, nReject = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* o = &S[size_t(i) * STRIDE_SG];
        // C8: `dist` is Sample(f) RAW.  The distance entry reported Sample(f)
        // and PxSdfDistance separately, and outside the box they differ -- so a
        // port that reused PxSdfDistance here fails on exactly those points.
        const float rawSample = D[size_t(i) * STRIDE_DIST + 7];
        if (!bitEq(o[0], rawSample))
        {
            if (badDist < 4)
                std::printf("    %s %s: dist %.9g != Sample(f) %.9g (PxSdfDistance was %.9g)\n",
                            r.label, r.pts[i].name, o[0], rawSample, D[size_t(i) * STRIDE_DIST + 8]);
            ++badDist;
        }
        if (!bitEq3(o + 5, wantDiff[i])) ++badDiff;

        const float tol = ptsBuf[i].w;
        const PxVec3 gin(gradBuf[i].x, gradBuf[i].y, gradBuf[i].z);
        const PxVec3 seed(dirBuf[i].x, dirBuf[i].y, dirBuf[i].z);
        if (o[0] < tol)
        {
            ++nAccept;
            // C10 + C11: the gradient is normalised AGAIN here, with the
            // engine's own getNormalized.
            const PxVec3 dir = (gin.getNormalized() * PxAbs(o[0]) + wantDiff[i]).getNormalized();
            const float ret = o[0] + dir.dot(wantDiff[i]);
            if (!bitEq3(o + 1, dir))
            {
                if (badDir < 4)
                    std::printf("    %s %s: dir (%.9g,%.9g,%.9g) want (%.9g,%.9g,%.9g)\n",
                                r.label, r.pts[i].name, o[1], o[2], o[3], dir.x, dir.y, dir.z);
                ++badDir;
            }
            if (!bitEq(o[4], ret))
            {
                if (badRet < 4)
                    std::printf("    %s %s: return %.9g want %.9g\n",
                                r.label, r.pts[i].name, o[4], ret);
                ++badRet;
            }
        }
        else
        {
            ++nReject;
            if (!bitEq(o[4], PX_MAX_F32))
            {
                if (badRet < 4)
                    std::printf("    %s %s: rejected return %.9g, want PX_MAX_F32\n",
                                r.label, r.pts[i].name, o[4]);
                ++badRet;
            }
            // C9: the reject path does NOT write dir.
            if (!bitEq3(o + 1, seed))
            {
                if (badKeep < 4)
                    std::printf("    %s %s: dir was overwritten on the reject path "
                                "((%.9g,%.9g,%.9g), seed (%.9g,%.9g,%.9g))\n",
                                r.label, r.pts[i].name, o[1], o[2], o[3], seed.x, seed.y, seed.z);
                ++badKeep;
            }
        }
    }
    PXG_CHECK_MSG(badDist == 0, "%s: %zu PxSdfSampleWithGrad dists are not the RAW Sample(f)",
                  r.label, badDist);
    PXG_CHECK_MSG(badDiff == 0, "%s: %zu diff vectors differ from localPos - clampedGridPt",
                  r.label, badDiff);
    PXG_CHECK_MSG(badDir == 0, "%s: %zu contact directions differ from the engine's own "
                  "(g.getNormalized() * |dist| + diff).getNormalized()", r.label, badDir);
    PXG_CHECK_MSG(badRet == 0, "%s: %zu returns differ from dist + dir.dot(diff)", r.label, badRet);
    PXG_CHECK_MSG(badKeep == 0, "%s: %zu rejected points had `dir` written anyway", r.label, badKeep);
    PXG_CHECK_MSG(nAccept > 0 && nReject > 0,
                  "%s: the tolerance must gate BOTH ways (%zu accepted, %zu rejected)",
                  r.label, nAccept, nReject);

    // =====================================================================
    // Pass 4: doVertexSDFCollision.
    // =====================================================================
    if (!run("sdfCollVertex", bOutV)) return;
    dev->flush();
    const auto V = downloadBuffer<float>(dev, bOutV, size_t(N) * STRIDE_GRAD);

    const PxVec3 discontinuity = PxVec3(0.5448762f, 0.672269f, 0.398825f).getNormalized();  // C13
    size_t badRaw = 0, badMagV = 0, badUsed = 0, badDirV = 0, badRetV = 0, badKeepV = 0, badDistV = 0;
    size_t nSubstituted = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* o = &V[size_t(i) * STRIDE_VTX];
        const float* g = &G[size_t(i) * STRIDE_GRAD];

        // It really calls PxVolumeGrad, and the same one: bit-for-bit against
        // the gradient entry's own output at the same point.
        if (!bitEq(o[0], g[1]) || !bitEq(o[1], g[2]) || !bitEq(o[2], g[3]))
        {
            if (badRaw < 4)
                std::printf("    %s %s: rawGrad (%.9g,%.9g,%.9g) != PxVolumeGrad "
                            "(%.9g,%.9g,%.9g)\n", r.label, r.pts[i].name,
                            o[0], o[1], o[2], g[1], g[2], g[3]);
            ++badRaw;
        }

        // C12: PxVec3::normalize() -- in place, RETURNING the magnitude.  Called
        // on the engine's own PxVec3 so the guard, the reciprocal and the
        // multiply are all the engine's.
        PxVec3 gn = v3(o);
        const float mag = gn.normalize();                                   // ENGINE
        if (!bitEq(o[3], mag))
        {
            if (badMagV < 4)
                std::printf("    %s %s: gradMag %.9g want %.9g\n", r.label, r.pts[i].name, o[3], mag);
            ++badMagV;
        }
        const bool substituted = (mag == 0.0f);
        if (substituted) { gn = discontinuity; ++nSubstituted; }
        if (!bitEq3(o + 4, gn))
        {
            if (badUsed < 4)
                std::printf("    %s %s: gradient handed on (%.9g,%.9g,%.9g) want (%.9g,%.9g,%.9g)"
                            "%s\n", r.label, r.pts[i].name, o[4], o[5], o[6], gn.x, gn.y, gn.z,
                            substituted ? "  [the discontinuity substitute]" : "");
            ++badUsed;
        }

        // The tail is PxSdfSampleWithGrad with that gradient -- same point, same
        // tolerance, same seed, so `dist` must be the same number pass 3 saw.
        if (!bitEq(o[7], S[size_t(i) * STRIDE_SG + 0])) ++badDistV;

        const float tol = ptsBuf[i].w;
        const PxVec3 seed(dirBuf[i].x, dirBuf[i].y, dirBuf[i].z);
        if (o[7] < tol)
        {
            const PxVec3 dir = (v3(o + 4).getNormalized() * PxAbs(o[7]) + wantDiff[i]).getNormalized();
            const float ret = o[7] + dir.dot(wantDiff[i]);
            if (!bitEq3(o + 8, dir))
            {
                if (badDirV < 4)
                    std::printf("    %s %s: vertex dir (%.9g,%.9g,%.9g) want (%.9g,%.9g,%.9g)\n",
                                r.label, r.pts[i].name, o[8], o[9], o[10], dir.x, dir.y, dir.z);
                ++badDirV;
            }
            if (!bitEq(o[11], ret))
            {
                if (badRetV < 4)
                    std::printf("    %s %s: vertex return %.9g want %.9g\n",
                                r.label, r.pts[i].name, o[11], ret);
                ++badRetV;
            }
        }
        else
        {
            if (!bitEq(o[11], PX_MAX_F32)) ++badRetV;
            if (!bitEq3(o + 8, seed)) ++badKeepV;
        }
    }
    PXG_CHECK_MSG(badRaw == 0, "%s: %zu doVertexSDFCollision gradients are not PxVolumeGrad's",
                  r.label, badRaw);
    PXG_CHECK_MSG(badMagV == 0, "%s: %zu gradient magnitudes differ from PxVec3::normalize()'s "
                  "return", r.label, badMagV);
    PXG_CHECK_MSG(badUsed == 0, "%s: %zu normalised/substituted gradients are wrong", r.label, badUsed);
    PXG_CHECK_MSG(badDistV == 0, "%s: %zu vertex dists differ from the sample-with-grad entry's",
                  r.label, badDistV);
    PXG_CHECK_MSG(badDirV == 0, "%s: %zu vertex directions are wrong", r.label, badDirV);
    PXG_CHECK_MSG(badRetV == 0, "%s: %zu vertex returns are wrong", r.label, badRetV);
    PXG_CHECK_MSG(badKeepV == 0, "%s: %zu rejected vertices had `dir` written anyway",
                  r.label, badKeepV);

    std::printf("    [ note ] %s: %u points -- %zu interior / %zu fall-back gradients, "
                "%zu outside the box, %zu accepted / %zu rejected, %zu substituted gradients; "
                "%zu samples cross-checked against the SDF (worst |diff| %.3g)\n",
                r.label, N, nInterior, nFallback, nOutside, nAccept, nReject, nSubstituted,
                nExactTexel, worstSample);
    // Only the grid-aligned scene can produce an EXACTLY zero gradient, so only
    // it may take the discontinuity substitute.
    if (r.expectDiscontinuity)
        PXG_CHECK_MSG(nSubstituted == 1,
                      "%s: exactly one point must hit the exactly-zero gradient and take the "
                      "discontinuity substitute (%zu did) -- without it C13's getNormalized on "
                      "the arbitrary direction is never executed", r.label, nSubstituted);
    else
        PXG_CHECK_MSG(nSubstituted == 0,
                      "%s: no point here should produce an exactly zero gradient (%zu did)",
                      r.label, nSubstituted);
}

PXG_TEST(sdfCollision)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }

    const PxgKernelDef manifest[] = { { "npSdfCollision.spv", "sdfCollDistance", 0u, 0u } };
    static auto loader = [](const char* f) { return loadShaderBlob(f); };
    PxgCudaShim shim(dev, manifest, 1, +loader);
    physx::gpuslang::pxgSetVulkanDriverShim(&shim);

    std::vector<PxgBufferHandle> owned;
    auto mkBuf = [&](const std::vector<uint8_t>& v) {
        PxgBufferHandle h = createBufferFrom(dev, v);
        owned.push_back(h);
        return h;
    };

    // =====================================================================
    // The sparse scene -- the only one that drives Sample()'s two branches.
    // =====================================================================
    SdfScene sp;
    buildSparse(sp);
    // The blob counts are P9.12c's, and for its reason: the pre-alignment size
    // 20 + 1168*3 + 20*16 + 48*21 = 4852 sits 4 bytes PAST a 16-boundary, so a
    // four-byte section-stride error can no longer be absorbed by the rounding.
    BlobLayout blobSp = layOutBlob(sp, 16u, 21u, 3u, 1u, 0u);
    const uint64_t hCoarse = sdfUploadTex(CU_AD_FORMAT_FLOAT, sp.nbx + 1, sp.nby + 1, sp.nbz + 1,
                                          sp.coarse.data(), sizeof(float));
    const uint64_t hSub = sdfUploadTex(CU_AD_FORMAT_UNSIGNED_INT8, sp.blkx * (sp.sg + 1),
                                       sp.blky * (sp.sg + 1), sp.blkz * (sp.sg + 1),
                                       sp.sub.data(), 1);
    PXG_CHECK_MSG(hCoarse != 0 && hSub != 0 && hCoarse != hSub,
                  "the two sparse textures must upload to two distinct slots (%llu, %llu)",
                  (unsigned long long)hCoarse, (unsigned long long)hSub);
    if (hCoarse == 0 || hSub == 0) return;
    std::memcpy(&blobSp.bytes[blobSp.texObjOffset], &hCoarse, 8);
    std::memcpy(&blobSp.bytes[blobSp.texObjSparseOffset], &hSub, 8);

    SceneRun rs;
    rs.label = "sparse";
    rs.s = &sp;
    rs.box = hostSdfBox(sp);
    rs.blobAddr = dev->getBufferDeviceAddress(mkBuf(blobSp.bytes));
    rs.expectDiscontinuity = false;
    {
        const HostSdfBox& b = rs.box;
        // Fine-grid coordinates chosen against the resident-cell map: cell index
        // z*3*2 + y*3 + x, resident at {1, 4, 12, 16, 22, 27}.
        rs.pts.push_back({ posOfF(b, 5, 2, 3),   5, 2, 3,   "resident-interior" });   // cell 1
        rs.pts.push_back({ posOfF(b, 9, 5, 15),  9, 5, 15,  "coarse-interior" });     // cell 23
        rs.pts.push_back({ posOfF(b, 1, 1, 1),   1, 1, 1,   "guard-lo" });
        rs.pts.push_back({ posOfF(b, 10, 6, 18), 10, 6, 18, "guard-hi" });
        rs.pts.push_back({ posOfF(b, 0, 4, 10),  0, 4, 10,  "face-x-lo" });
        rs.pts.push_back({ posOfF(b, 12, 4, 10), 12, 4, 10, "face-x-hi" });
        rs.pts.push_back({ posOfF(b, 6.5f, 3.5f, 9.5f), -1, -1, -1, "between-texels" });
        rs.pts.push_back({ posOfF(b, 8, 3, 7),   8, 3, 7,   "coarse-cell-8" });
        rs.pts.push_back({ posOfF(b, 4, 2, 2),   4, 2, 2,   "resident-cell-edge" });
        rs.pts.push_back({ b.higher + PxVec3(0.37f, 0.f, 0.f),  -1, -1, -1, "out-x" });
        rs.pts.push_back({ b.lower + PxVec3(0.f, -0.21f, 0.f) + PxVec3(0.f, 0.f, 1.3f),
                           -1, -1, -1, "out-y-in-z" });
        rs.pts.push_back({ b.higher + PxVec3(0.15f, 0.29f, 0.41f), -1, -1, -1, "out-xyz" });
    }
    checkScene(dev, rs, owned);

    // =====================================================================
    // The grid-aligned dense scene -- exact texel coordinates, and the
    // discontinuity.
    // =====================================================================
    SdfScene dn;
    buildDenseAligned(dn);
    BlobLayout blobDn = layOutBlob(dn, 5u, 9u, 1u, 0u, 0x0BADF00Du);
    const uint64_t hDense = sdfUploadTex(CU_AD_FORMAT_FLOAT, dn.dx, dn.dy, dn.dz,
                                         dn.coarse.data(), sizeof(float));
    PXG_CHECK_MSG(hDense != 0, "the grid-aligned dense texture upload failed");
    if (hDense == 0) return;
    std::memcpy(&blobDn.bytes[blobDn.texObjOffset], &hDense, 8);

    SceneRun rd;
    rd.label = "dense-aligned";
    rd.s = &dn;
    rd.box = hostSdfBox(dn);
    rd.blobAddr = dev->getBufferDeviceAddress(mkBuf(blobDn.bytes));
    rd.expectDiscontinuity = true;
    {
        const HostSdfBox& b = rd.box;
        // dims 9 x 6 x 13 -> the guard is f in [1,7] x [1,4] x [1,11], and the
        // box spans f in [0,8] x [0,5] x [0,12].
        rd.pts.push_back({ posOfF(b, float(FLAT_X), float(FLAT_Y), float(FLAT_Z)),
                           int(FLAT_X), int(FLAT_Y), int(FLAT_Z), "discontinuity" });
        rd.pts.push_back({ posOfF(b, 1, 1, 1),    1, 1, 1,    "guard-lo-exact" });
        rd.pts.push_back({ posOfF(b, 7, 4, 11),   7, 4, 11,   "guard-hi-exact" });
        rd.pts.push_back({ posOfF(b, 0.5f, 2, 4), -1, -1, -1, "below-guard-x" });
        rd.pts.push_back({ posOfF(b, 8, 2, 4),    8, 2, 4,    "face-x-hi" });
        rd.pts.push_back({ posOfF(b, 4, 2, 6),    4, 2, 6,    "interior" });
        rd.pts.push_back({ posOfF(b, 2, 1, 1),    2, 1, 1,    "interior-2" });
        rd.pts.push_back({ posOfF(b, 6, 4, 11),   6, 4, 11,   "interior-3" });
        rd.pts.push_back({ posOfF(b, 9, 2, 4),    -1, -1, -1, "out-x" });
        rd.pts.push_back({ posOfF(b, -2, 3, 14),  -1, -1, -1, "out-x-lo-z-hi" });
        rd.pts.push_back({ posOfF(b, -1.5f, 7.25f, -3), -1, -1, -1, "out-xyz" });
        rd.pts.push_back({ posOfF(b, 7.5f, 3.5f, 11.5f), -1, -1, -1, "half-texel-above-guard" });
        // C7's guard ANDs three axes.  These three fail on EXACTLY ONE of them
        // while passing the other two, which is what makes a mutant that drops
        // one conjunct -- or tests only x -- die rather than merely be lucky.
        rd.pts.push_back({ posOfF(b, 4, 0, 6),  4, 0, 6,  "guard-fails-y-lo-only" });
        rd.pts.push_back({ posOfF(b, 4, 5, 6),  4, 5, 6,  "guard-fails-y-hi-only" });
        rd.pts.push_back({ posOfF(b, 4, 2, 0),  4, 2, 0,  "guard-fails-z-lo-only" });
    }
    checkScene(dev, rd, owned);

    for (PxgBufferHandle h : owned)
        dev->destroyBuffer(h);
    physx::gpuslang::pxgSetVulkanDriverShim(nullptr);
}



// ===========================================================================
// P9.34a: Gu::closestPtPointBaryTriangle and doTriangleSDFCollision.
//
// The two pieces cloth_sdfMeshContactGenLaunch needs that P9.12d did not port.
// They are checked in two DIFFERENT ways, and the difference is the point:
//
//   * closestPtPointBaryTriangle is PX_CUDA_CALLABLE and texture-free, so this
//     file includes the engine's header and calls THE ENGINE'S OWN COPY.  That is
//     a real oracle -- there is no transcription on the host side to be wrong.
//
//   * doTriangleSDFCollision cannot be: it samples a texture from the GPU.  So
//     the shader TRACES its descent and the host REPLAYS it, forming
//     dfdu/dfdv/dfdw from the traced gradient, applying the traced step, and
//     calling the engine's clamp -- which must land on the traced newUvw bit for
//     bit.  What that checks is the composition; the gradient values themselves
//     are P9.12c/d's claim, measured there and not re-litigated here.
// ===========================================================================
PXG_TEST(sdfTriangleCollision)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }

    const PxgKernelDef manifest[] = { { "npSdfCollision.spv", "sdfCollBaryClamp", 0u, 0u } };
    static auto loader2 = [](const char* f) { return loadShaderBlob(f); };
    PxgCudaShim shim(dev, manifest, 1, +loader2);
    physx::gpuslang::pxgSetVulkanDriverShim(&shim);

    std::vector<PxgBufferHandle> owned;
    auto mkBuf = [&](const std::vector<uint8_t>& v) {
        PxgBufferHandle h = createBufferFrom(dev, v);
        owned.push_back(h);
        return h;
    };
    auto asBytes = [](const void* p, size_t n) {
        std::vector<uint8_t> v(n);
        std::memcpy(v.data(), p, n);
        return v;
    };

    // ---- part 1: the barycentric clamp, against the engine's own copy ---------
    // Every branch of the cascade, and both escapes inside each one-negative arm.
    std::vector<PxVec4> bary;
    auto addB = [&](float x, float y, float z) { bary.push_back(PxVec4(x, y, z, 0.0f)); };
    addB(1.0f/3, 1.0f/3, 1.0f/3);      // already interior
    addB(0.5f, 0.3f, 0.2f);
    addB(0.7f, -0.2f, 0.5f);           // one negative (y)
    addB(-0.2f, 0.7f, 0.5f);           // one negative (x)
    addB(0.5f, 0.7f, -0.2f);           // one negative (z)
    addB(-0.4f, -0.3f, 1.7f);          // two negative -> the Z vertex
    addB(-0.4f, 1.7f, -0.3f);          // two negative -> the Y vertex
    addB(1.7f, -0.4f, -0.3f);          // two negative -> the X vertex
    addB(-0.05f, 1.6f, -0.55f);        // one-negative arm whose escape fires
    addB(-0.55f, -0.05f, 1.6f);
    addB(0.0f, 0.0f, 1.0f);            // exactly on a vertex
    addB(0.0f, 0.5f, 0.5f);            // exactly on an edge
    addB(2.0f, 2.0f, 2.0f);            // far outside, sum 6 -> the projection matters
    addB(-1.0f, -1.0f, -1.0f);
    addB(0.0f, 0.0f, 0.0f);            // sum 0 -> projects to the centroid
    std::mt19937 rngB(9341u);
    for (uint32_t i = 0; i < 4000u; ++i)
    {
        auto r = [&]() { return (float(rngB() % 400001u) / 100000.0f) - 2.0f; };   // [-2, 2]
        addB(r(), r(), r());
    }
    const uint32_t NB = uint32_t(bary.size());

    std::vector<PxVec4> zero4(NB, PxVec4(0.0f));
    PxgBufferHandle bPts = mkBuf(asBytes(bary.data(), bary.size() * sizeof(PxVec4)));
    PxgBufferHandle bZero = mkBuf(asBytes(zero4.data(), zero4.size() * sizeof(PxVec4)));
    std::vector<float> outB(size_t(NB) * 4u, -1.0f);
    PxgBufferHandle bOutB = mkBuf(asBytes(outB.data(), outB.size() * sizeof(float)));

    P_Coll pb = {};
    pb.points = dev->getBufferDeviceAddress(bPts);
    pb.gradIn = dev->getBufferDeviceAddress(bZero);
    pb.dirSeed = dev->getBufferDeviceAddress(bZero);
    pb.outBuf = dev->getBufferDeviceAddress(bOutB);
    pb.numPoints = NB;
    {
        Kernel k(Kernel::Bindless{}, "npSdfCollision.spv", "sdfCollBaryClamp", {}, sizeof(P_Coll));
        PXG_CHECK_MSG(k.isValid(), "bindless pipeline creation succeeds for sdfCollBaryClamp");
        if (!k.isValid()) { for (auto h : owned) dev->destroyBuffer(h); return; }
        k.dispatch((NB + 63) / 64, {}, pb);
    }
    dev->flush();
    const auto B = downloadBuffer<float>(dev, bOutB, size_t(NB) * 4u);

    size_t badB = 0;
    uint32_t armCount[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (uint32_t i = 0; i < NB; ++i)
    {
        const PxVec3 c(bary[i].x, bary[i].y, bary[i].z);
        const PxVec3 want = physx::Gu::closestPtPointBaryTriangle(c);   // THE ENGINE
        const float* o = &B[size_t(i) * 4u];
        if (f2u(o[0]) != f2u(want.x) || f2u(o[1]) != f2u(want.y) || f2u(o[2]) != f2u(want.z))
        {
            if (badB < 4)
                std::printf("    bary[%u] (%.9g %.9g %.9g) -> (%.9g %.9g %.9g), engine says "
                            "(%.9g %.9g %.9g)\n", i, c.x, c.y, c.z, o[0], o[1], o[2],
                            want.x, want.y, want.z);
            ++badB;
        }
        // Which arm the engine took, classified the same way the shader branches, so
        // the run can assert every arm was actually reached.
        const float third = 1.0f / 3.0f;
        const PxVec3 q = c - PxVec3(third * (c.x + c.y + c.z - 1.0f));
        uint32_t arm;
        if (q.y < 0.0f && q.z < 0.0f) arm = 0;
        else if (q.x < 0.0f && q.z < 0.0f) arm = 1;
        else if (q.x < 0.0f && q.y < 0.0f) arm = 2;
        else if (q.x < 0.0f) arm = 3;
        else if (q.y < 0.0f) arm = 4;
        else if (q.z < 0.0f) arm = 5;
        else arm = 6;
        ++armCount[arm];
    }
    PXG_CHECK_MSG(badB == 0, "%zu of %u barycentric clamps differ from the engine's own "
                  "Gu::closestPtPointBaryTriangle", badB, NB);
    for (uint32_t a = 0; a < 7u; ++a)
        PXG_CHECK_MSG(armCount[a] > 0,
                      "arm %u of the clamp cascade was never reached (%u inputs); an unreached "
                      "branch is an unchecked branch", a, NB);

    // ---- part 2: the triangle descent, replayed from its own trace ------------
    SdfScene sp;
    buildSparse(sp);
    BlobLayout blobSp = layOutBlob(sp, 16u, 21u, 3u, 1u, 0u);
    const uint64_t hCoarse = sdfUploadTex(CU_AD_FORMAT_FLOAT, sp.nbx + 1, sp.nby + 1, sp.nbz + 1,
                                          sp.coarse.data(), sizeof(float));
    const uint64_t hSub = sdfUploadTex(CU_AD_FORMAT_UNSIGNED_INT8, sp.blkx * (sp.sg + 1),
                                       sp.blky * (sp.sg + 1), sp.blkz * (sp.sg + 1),
                                       sp.sub.data(), 1);
    if (hCoarse == 0 || hSub == 0)
    { std::printf("  NOTE: texture upload failed -> skip part 2\n");
      for (auto h : owned) dev->destroyBuffer(h); return; }
    std::memcpy(&blobSp.bytes[blobSp.texObjOffset], &hCoarse, 8);
    std::memcpy(&blobSp.bytes[blobSp.texObjSparseOffset], &hSub, 8);
    const uint64_t blobAddr = dev->getBufferDeviceAddress(mkBuf(blobSp.bytes));
    const HostSdfBox box = hostSdfBox(sp);

    // Triangles spanning the box in different ways: one straddling the surface, one
    // wholly inside, one wholly outside, one degenerate (all three vertices equal,
    // so `difference` is 0 and the initial step is an infinity), and one whose
    // centroid is nearer than every vertex so the cascade takes its else arm.
    struct Tri { PxVec3 a, b, c; float tol; const char* name; };
    std::vector<Tri> tris;
    tris.push_back({ posOfF(box, 3, 3, 3),  posOfF(box, 9, 3, 3),  posOfF(box, 3, 9, 3),
                     1.0f, "straddling" });
    tris.push_back({ posOfF(box, 5, 5, 5),  posOfF(box, 6, 5, 5),  posOfF(box, 5, 6, 5),
                     1.0f, "interior" });
    tris.push_back({ posOfF(box, -6, -6, -6), posOfF(box, -5, -6, -6), posOfF(box, -6, -5, -6),
                     1.0f, "outside" });
    tris.push_back({ posOfF(box, 4, 4, 4),  posOfF(box, 4, 4, 4),  posOfF(box, 4, 4, 4),
                     1.0f, "degenerate" });
    tris.push_back({ posOfF(box, 2, 8, 4),  posOfF(box, 12, 8, 4), posOfF(box, 7, 2, 12),
                     1.0f, "wide" });
    tris.push_back({ posOfF(box, 1, 1, 1),  posOfF(box, 14, 14, 1), posOfF(box, 1, 14, 14),
                     0.001f, "tight-tolerance" });
    // D1's cascade does NOT compare d1 against d0 in its second arm, so a port that
    // "fixes" it into a three-way minimum differs exactly when d0 and d1 tie.  These
    // two put v0 and v1 at mirror positions about the grid centre, which makes their
    // sampled distances equal, and the first sweep had nothing that did.
    tris.push_back({ posOfF(box, 4, 6, 6),  posOfF(box, 8, 6, 6),  posOfF(box, 6, 11, 6),
                     1.0f, "d0-eq-d1" });
    tris.push_back({ posOfF(box, 6, 4, 6),  posOfF(box, 6, 8, 6),  posOfF(box, 11, 6, 6),
                     1.0f, "d0-eq-d1-b" });
    const uint32_t NT = uint32_t(tris.size());

    std::vector<PxVec4> tv(size_t(NT) * 3u);
    std::vector<PxVec4> dirSeed(NT);
    for (uint32_t i = 0; i < NT; ++i)
    {
        tv[i * 3 + 0] = PxVec4(tris[i].a.x, tris[i].a.y, tris[i].a.z, tris[i].tol);
        tv[i * 3 + 1] = PxVec4(tris[i].b.x, tris[i].b.y, tris[i].b.z, 0.0f);
        tv[i * 3 + 2] = PxVec4(tris[i].c.x, tris[i].c.y, tris[i].c.z, 0.0f);
        dirSeed[i] = PxVec4(-3.0f, -4.0f, -5.0f, 0.0f);
    }
    const uint32_t TRI_STRIDE = 176u;
    std::vector<float> outT(size_t(NT) * TRI_STRIDE, -1.0f);
    PxgBufferHandle bTv = mkBuf(asBytes(tv.data(), tv.size() * sizeof(PxVec4)));
    PxgBufferHandle bDir = mkBuf(asBytes(dirSeed.data(), dirSeed.size() * sizeof(PxVec4)));
    PxgBufferHandle bOutT = mkBuf(asBytes(outT.data(), outT.size() * sizeof(float)));

    P_Coll pt = {};
    pt.meshPtr = blobAddr;
    pt.points = dev->getBufferDeviceAddress(bPts);
    pt.gradIn = dev->getBufferDeviceAddress(bZero);
    pt.dirSeed = dev->getBufferDeviceAddress(bDir);
    pt.outBuf = dev->getBufferDeviceAddress(bOutB);
    pt.numPoints = NT;
    pt.triVerts = dev->getBufferDeviceAddress(bTv);
    pt.triOut = dev->getBufferDeviceAddress(bOutT);
    {
        Kernel k(Kernel::Bindless{}, "npSdfCollision.spv", "sdfCollTriangle", {}, sizeof(P_Coll));
        PXG_CHECK_MSG(k.isValid(), "bindless pipeline creation succeeds for sdfCollTriangle");
        if (!k.isValid()) { for (auto h : owned) dev->destroyBuffer(h); return; }
        k.dispatch((NT + 63) / 64, {}, pt);
    }
    dev->flush();
    const auto T = downloadBuffer<float>(dev, bOutT, size_t(NT) * TRI_STRIDE);

    size_t badStart = 0, badDiff2 = 0, badStep = 0, badUvw = 0, badBreak = 0, badGradN = 0;
    size_t badIters = 0, badFinalG = 0;
    uint32_t armSeen[4] = { 0, 0, 0, 0 };
    uint32_t nConverged = 0, nRanOut = 0, nSubstituted = 0, nTied = 0, nFinalGChecked = 0;
    for (uint32_t i = 0; i < NT; ++i)
    {
        const float* o = &T[size_t(i) * TRI_STRIDE];
        const PxVec3 v0 = tris[i].a, v1 = tris[i].b, v2 = tris[i].c;
        const float d0 = o[0], d1 = o[1], d2 = o[2], dC = o[3];
        const uint32_t arm = uint32_t(o[4]);
        armSeen[arm < 4u ? arm : 3u]++;
        if (f2u(d0) == f2u(d1)) ++nTied;

        // D1/D2: the cascade, re-derived from the four traced distances.
        const float third = 1.0f / 3.0f;
        PxVec3 wantP = (v0 + v1 + v2) * third;
        PxVec3 wantUvw(0.0f);
        uint32_t wantArm;
        if (d0 < d1 && d0 < d2 && d0 < dC) { wantP = v0; wantUvw.x = 1.0f; wantArm = 0; }
        else if (d1 < d2 && d1 < dC)       { wantP = v1; wantUvw.y = 1.0f; wantArm = 1; }
        else if (d2 < dC)                  { wantP = v2; wantUvw.z = 1.0f; wantArm = 2; }
        else { wantUvw = PxVec3(third); wantArm = 3; }
        if (arm != wantArm || !bitEq3(o + 6, wantUvw) || !bitEq3(o + 9, wantP))
        {
            if (badStart < 4)
                std::printf("    tri %s: start arm %u want %u, uvw (%.9g %.9g %.9g)\n",
                            tris[i].name, arm, wantArm, o[6], o[7], o[8]);
            ++badStart;
        }

        // D3: nested max of SQUARED lengths, one sqrt.
        const float wantDifference = std::sqrt(PxMax((v0 - wantP).magnitudeSquared(),
                                      PxMax((v1 - wantP).magnitudeSquared(),
                                            (v2 - wantP).magnitudeSquared())));
        if (f2u(o[5]) != f2u(wantDifference))
        { if (badDiff2 < 4) std::printf("    tri %s: difference %.9g want %.9g\n",
                                        tris[i].name, o[5], wantDifference); ++badDiff2; }

        const uint32_t iters = uint32_t(o[12]);
        const uint32_t conv = uint32_t(o[13]);
        if (conv) ++nConverged; else ++nRanOut;

        // The CAP.  The replay below takes `iters` FROM the shader, so it is
        // self-consistent by construction and cannot see the loop bound at all --
        // which is why a 15-iteration mutant survived the first sweep.  A run that
        // did not converge must have used exactly 16.
        if (iters < 1u || iters > 16u || (conv == 0u && iters != 16u))
        { if (badIters < 4)
              std::printf("    tri %s: %u iterations, converged %u\n",
                          tris[i].name, iters, conv);
          ++badIters; }

        // D7: the gradient is RECOMPUTED at the final p.  If the port reused the
        // loop's last one instead, gradFinal would be that iteration's USED gradient
        // -- which is normalised, and a freshly sampled one is not.
        // The degenerate triangle is exempt, and not as a convenience: with all three
        // vertices equal, `difference` is 0, the initial step is +inf and every
        // gradient downstream is NaN, so the two values are identical bit patterns no
        // matter which one the port produced.  There is nothing to distinguish there,
        // so the check is applied only where the arithmetic is finite -- and the run
        // insists that some triangle qualified.
        const PxVec3 gFinal(o[17], o[18], o[19]);
        const bool finite = std::isfinite(gFinal.x) && std::isfinite(gFinal.y) &&
                            std::isfinite(gFinal.z);
        if (finite)
        {
            ++nFinalGChecked;
            if (bitEq3(o + 17, PxVec3(o[28 + (iters - 1u) * 9 + 5], o[28 + (iters - 1u) * 9 + 6],
                                      o[28 + (iters - 1u) * 9 + 7])))
            { if (badFinalG < 4)
                  std::printf("    tri %s: gradFinal is bit-identical to the loop's last USED "
                              "gradient (%.9g %.9g %.9g); it must be a fresh sample at the "
                              "final p\n", tris[i].name, gFinal.x, gFinal.y, gFinal.z);
              ++badFinalG; }
        }

        // The replay.  Everything below is derived on the host from the traced
        // gradient and step; only those two are taken from the shader.
        PxVec3 uvw = wantUvw;
        float step = 1.0f / (2.0f * wantDifference);
        for (uint32_t k = 0; k < iters; ++k)
        {
            const float* e = o + 28 + k * 9;
            const float stepIn = e[0], gradMag = e[1];
            const PxVec3 gradUsed(e[5], e[6], e[7]);
            if (gradMag == 0.0f) ++nSubstituted;
            else
            {
                // D8: gradUsed must be the raw gradient scaled by 1/gradMag, and the
                // result unit to within a float.
                const float m = gradUsed.magnitude();
                if (std::fabs(m - 1.0f) > 1e-5f)
                { if (badGradN < 4)
                      std::printf("    tri %s iter %u: |gradUsed| = %.9g\n", tris[i].name, k, m);
                  ++badGradN; }
            }
            if (f2u(stepIn) != f2u(step))
            { if (badStep < 4)
                  std::printf("    tri %s iter %u: step %.9g want %.9g\n",
                              tris[i].name, k, stepIn, step); ++badStep; }

            // D2: iteration 0 descends from the point the cascade chose, which is the
            // CENTROID in the else arm -- not from the barycentric blend of uvw, which
            // would differ there by a rounding.  Later iterations use the blend the
            // previous iteration produced.
            const PxVec3 p = (k == 0) ? wantP : (v0 * uvw.x + v1 * uvw.y + v2 * uvw.z);
            const float du = gradUsed.dot(v0 - p);
            const float dv = gradUsed.dot(v1 - p);
            const float dw = gradUsed.dot(v2 - p);
            PxVec3 newUVW = uvw;
            newUVW.x -= step * du;
            newUVW.y -= step * dv;
            newUVW.z -= step * dw;
            step = step * 0.8f;                                   // D4
            newUVW = physx::Gu::closestPtPointBaryTriangle(newUVW);   // THE ENGINE
            if (!bitEq3(e + 2, newUVW))
            {
                if (badUvw < 4)
                    std::printf("    tri %s iter %u: newUvw (%.9g %.9g %.9g) want "
                                "(%.9g %.9g %.9g)\n", tris[i].name, k, e[2], e[3], e[4],
                                newUVW.x, newUVW.y, newUVW.z);
                ++badUvw;
            }
            // D5: the loop stops exactly when the CLAMPED delta is under tolerance.
            const bool wantBreak = (uvw - newUVW).magnitudeSquared() < 1e-3f * 1e-3f;
            const bool didBreak = (k + 1u == iters) && conv != 0u;
            if (wantBreak != didBreak && !(k + 1u == iters && conv == 0u && !wantBreak))
            { if (badBreak < 4)
                  std::printf("    tri %s iter %u: break %d want %d (iters %u conv %u)\n",
                              tris[i].name, k, int(didBreak), int(wantBreak), iters, conv);
              ++badBreak; }
            if (wantBreak) break;
            uvw = newUVW;
        }
    }
    PXG_CHECK_MSG(badStart == 0, "%zu triangles chose a different starting iterate", badStart);
    PXG_CHECK_MSG(badDiff2 == 0, "%zu triangles computed a different `difference`", badDiff2);
    PXG_CHECK_MSG(badStep == 0, "%zu iterations used a step off the 1/(2d) * 0.8^k schedule",
                  badStep);
    PXG_CHECK_MSG(badUvw == 0, "%zu iterations landed on a different clamped barycentric than "
                  "the engine's own closestPtPointBaryTriangle applied to the replayed update",
                  badUvw);
    PXG_CHECK_MSG(badBreak == 0, "%zu iterations disagreed about where the loop stops", badBreak);
    PXG_CHECK_MSG(badGradN == 0, "%zu used gradients were not unit length", badGradN);
    PXG_CHECK_MSG(badIters == 0, "%zu triangles ran an iteration count the 16-step "
                  "bound does not allow", badIters);
    PXG_CHECK_MSG(badFinalG == 0, "%zu triangles reused the loop's last gradient "
                  "instead of resampling at the final p", badFinalG);
    PXG_CHECK_MSG(nFinalGChecked > 0,
                  "every triangle produced a non-finite final gradient, so the recompute of D7 "
                  "was never actually observed");
    PXG_CHECK_MSG(armSeen[3] > 0, "no triangle took the cascade's else arm (the centroid start), "
                  "so D1's fall-through is unchecked");
    PXG_CHECK_MSG(armSeen[0] + armSeen[1] + armSeen[2] > 0,
                  "no triangle started from a VERTEX, so the cascade's three arms are unchecked");
    // ---- part 3: the discontinuity substitution -------------------------------
    // gm == 0 is unreachable in the sparse scene above, so nSubstituted stayed 0 and
    // the whole substitution branch went unobserved -- which the first sweep caught by
    // leaving both of its mutants alive.  The DENSE scene has a point whose trilinear
    // gradient is exactly zero (FLAT_X/Y/Z, the same one the sdfCollision fixture uses
    // for its own substitution check), so a triangle collapsed onto it hits the branch
    // on its first iteration.
    {
        SdfScene dn;
        buildDenseAligned(dn);
        BlobLayout blobDn = layOutBlob(dn, 5u, 9u, 1u, 0u, 0x0BADF00Du);
        const uint64_t hDense = sdfUploadTex(CU_AD_FORMAT_FLOAT, dn.dx, dn.dy, dn.dz,
                                             dn.coarse.data(), sizeof(float));
        if (hDense == 0)
            std::printf("  NOTE: dense texture upload failed -> the discontinuity is unchecked\n");
        else
        {
            std::memcpy(&blobDn.bytes[blobDn.texObjOffset], &hDense, 8);
            const uint64_t dnAddr = dev->getBufferDeviceAddress(mkBuf(blobDn.bytes));
            const HostSdfBox dbox = hostSdfBox(dn);
            const PxVec3 flat = posOfF(dbox, float(FLAT_X), float(FLAT_Y), float(FLAT_Z));

            std::vector<PxVec4> dtv(3), ddir(1);
            dtv[0] = PxVec4(flat.x, flat.y, flat.z, 1.0f);
            dtv[1] = PxVec4(flat.x, flat.y, flat.z, 0.0f);
            dtv[2] = PxVec4(flat.x, flat.y, flat.z, 0.0f);
            ddir[0] = PxVec4(-3.0f, -4.0f, -5.0f, 0.0f);
            std::vector<float> outD(TRI_STRIDE, -1.0f);
            PxgBufferHandle bDtv = mkBuf(asBytes(dtv.data(), dtv.size() * sizeof(PxVec4)));
            PxgBufferHandle bDdir = mkBuf(asBytes(ddir.data(), ddir.size() * sizeof(PxVec4)));
            PxgBufferHandle bDout = mkBuf(asBytes(outD.data(), outD.size() * sizeof(float)));

            P_Coll pd = pt;
            pd.meshPtr = dnAddr;
            pd.numPoints = 1u;
            pd.dirSeed = dev->getBufferDeviceAddress(bDdir);
            pd.triVerts = dev->getBufferDeviceAddress(bDtv);
            pd.triOut = dev->getBufferDeviceAddress(bDout);
            Kernel k(Kernel::Bindless{}, "npSdfCollision.spv", "sdfCollTriangle", {}, sizeof(P_Coll));
            if (k.isValid())
            {
                k.dispatch(1u, {}, pd);
                dev->flush();
                const auto D2 = downloadBuffer<float>(dev, bDout, TRI_STRIDE);
                // Iteration 0's gradient must have had magnitude exactly zero, and the
                // substituted vector must be the RAW constant -- not normalised, and not
                // doVertexSDFCollision's different one.
                const float gm0 = D2[28 + 1];
                const PxVec3 used(D2[28 + 5], D2[28 + 6], D2[28 + 7]);
                const PxVec3 want(0.571846586f, 0.705545099f, 0.418566116f);
                PXG_CHECK_MSG(gm0 == 0.0f,
                              "the collapsed triangle on the dense scene's flat point must give a "
                              "ZERO gradient magnitude on iteration 0, got %.9g", gm0);
                if (gm0 == 0.0f)
                {
                    ++nSubstituted;
                    PXG_CHECK_MSG(f2u(used.x) == f2u(want.x) && f2u(used.y) == f2u(want.y) &&
                                  f2u(used.z) == f2u(want.z),
                                  "the discontinuity substitute is (%.9g %.9g %.9g); it must be "
                                  "sdfCollision.cuh:455's constant (%.9g %.9g %.9g) used RAW -- "
                                  "not normalised, and not doVertexSDFCollision's different one",
                                  used.x, used.y, used.z, want.x, want.y, want.z);
                }
            }
        }
    }

    PXG_CHECK_MSG(nSubstituted > 0,
                  "the discontinuity substitution was never reached, so both of its mutants -- a "
                  "normalised constant and the wrong constant -- are unchecked");
    PXG_CHECK_MSG(nConverged > 0, "no triangle converged, so the tolerance break is unchecked");
    PXG_CHECK_MSG(nRanOut > 0, "no triangle ran to the 16-iteration cap, so the bound is unchecked");
    PXG_CHECK_MSG(nTied > 0,
                  "no triangle produced d0 == d1, so D1's asymmetric cascade -- whose second arm "
                  "does not compare against d0 -- is indistinguishable from a three-way minimum");

    std::printf("    [ note ] bary clamp: %u inputs bit-exact against the engine's own copy, "
                "all 7 arms reached; triangle descent: %u triangles, %u converged / %u ran to 16, "
                "start arms %u/%u/%u/%u, %u with d0 == d1, %u discontinuity substitutions\n",
                NB, NT, nConverged, nRanOut, armSeen[0], armSeen[1], armSeen[2], armSeen[3],
                nTied, nSubstituted);

    for (auto h : owned) dev->destroyBuffer(h);
}

} // namespace
