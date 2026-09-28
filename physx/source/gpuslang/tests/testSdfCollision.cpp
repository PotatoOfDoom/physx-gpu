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
#include "GuDistancePointTetrahedron.h" // the ENGINE'S OWN closestPtPointTetrahedron (P9.35a)

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

// P9.86c: the pre-P9.86 port spelling of PxMax, `a < b ? b : a` -- PxMath.h:75's
// GENERIC template, which float does NOT use (PxMath.h:82 specialises it to
// intrinsics::selectMax, `a > b ? a : b`, PxUnixMathIntrinsics.h:122).  The two
// differ on a signed-zero tie: selectMax keeps the SECOND operand (the box
// bound).  A coverage twin only -- never an expectation.
static float hMaxGenericTwin(float a, float b) { return a < b ? b : a; }
static PxVec3 hClampToBoxGenericMax(const HostSdfBox& t, const PxVec3& p)
{
    return PxVec3(hMaxGenericTwin(p.x, t.lower.x), hMaxGenericTwin(p.y, t.lower.y),
                  hMaxGenericTwin(p.z, t.lower.z)).minimum(t.higher);
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
    // P9.86c coverage: points on which the generic-max twin changes the clamp,
    // the pass-3 dir, or the pass-4 vertex dir.
    size_t tieClamp = 0, tieDir = 0, tieDirV = 0;
    // Fall-back plus taps 0/2/4 whose IEEE add (sdfCollision.cuh:190-195)
    // differs bitwise from the llvmpipe-folded form that keeps p's zero lanes.
    size_t foldTap[3] = { 0, 0, 0 };
};

static bool bitEqV(const PxVec3& a, const PxVec3& b)
{ return bitEq(a.x, b.x) && bitEq(a.y, b.y) && bitEq(a.z, b.z); }

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
        if (!bitEqV(hClampToBoxGenericMax(box, r.pts[i].p), wantClamped[i])) ++r.tieClamp;
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
            // P9.86c coverage: the F3-folded plus taps leave the literal-zero
            // lanes as p's own (-0 stays -0).
            const PxVec3 folded[3] = { PxVec3(p.x + box.dx, p.y, p.z),
                                       PxVec3(p.x, p.y + box.dx, p.z),
                                       PxVec3(p.x, p.y, p.z + box.dx) };
            for (int k = 0; k < 3; ++k)
                if (!bitEqV(folded[k], want[2 * k])) ++r.foldTap[k];
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
            {
                const PxVec3 dT = r.pts[i].p - hClampToBoxGenericMax(box, r.pts[i].p);
                if (!bitEqV((gin.getNormalized() * PxAbs(o[0]) + dT).getNormalized(), dir)) ++r.tieDir;
            }
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
            {
                const PxVec3 dT = r.pts[i].p - hClampToBoxGenericMax(box, r.pts[i].p);
                if (!bitEqV((v3(o + 4).getNormalized() * PxAbs(o[7]) + dT).getNormalized(), dir)) ++r.tieDirV;
            }
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

// The grid-aligned dense scene's point list, relative to the run's own box.
static void addDenseAlignedPoints(SceneRun& r)
{
    const HostSdfBox& b = r.box;
    // dims 9 x 6 x 13 -> the guard is f in [1,7] x [1,4] x [1,11], and the
    // box spans f in [0,8] x [0,5] x [0,12].
    r.pts.push_back({ posOfF(b, float(FLAT_X), float(FLAT_Y), float(FLAT_Z)),
                       int(FLAT_X), int(FLAT_Y), int(FLAT_Z), "discontinuity" });
    r.pts.push_back({ posOfF(b, 1, 1, 1),    1, 1, 1,    "guard-lo-exact" });
    r.pts.push_back({ posOfF(b, 7, 4, 11),   7, 4, 11,   "guard-hi-exact" });
    r.pts.push_back({ posOfF(b, 0.5f, 2, 4), -1, -1, -1, "below-guard-x" });
    r.pts.push_back({ posOfF(b, 8, 2, 4),    8, 2, 4,    "face-x-hi" });
    r.pts.push_back({ posOfF(b, 4, 2, 6),    4, 2, 6,    "interior" });
    r.pts.push_back({ posOfF(b, 2, 1, 1),    2, 1, 1,    "interior-2" });
    r.pts.push_back({ posOfF(b, 6, 4, 11),   6, 4, 11,   "interior-3" });
    r.pts.push_back({ posOfF(b, 9, 2, 4),    -1, -1, -1, "out-x" });
    r.pts.push_back({ posOfF(b, -2, 3, 14),  -1, -1, -1, "out-x-lo-z-hi" });
    r.pts.push_back({ posOfF(b, -1.5f, 7.25f, -3), -1, -1, -1, "out-xyz" });
    r.pts.push_back({ posOfF(b, 7.5f, 3.5f, 11.5f), -1, -1, -1, "half-texel-above-guard" });
    // C7's guard ANDs three axes.  These three fail on EXACTLY ONE of them
    // while passing the other two, which is what makes a mutant that drops
    // one conjunct -- or tests only x -- die rather than merely be lucky.
    r.pts.push_back({ posOfF(b, 4, 0, 6),  4, 0, 6,  "guard-fails-y-lo-only" });
    r.pts.push_back({ posOfF(b, 4, 5, 6),  4, 5, 6,  "guard-fails-y-hi-only" });
    r.pts.push_back({ posOfF(b, 4, 2, 0),  4, 2, 0,  "guard-fails-z-lo-only" });
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
    addDenseAlignedPoints(rd);
    checkScene(dev, rd, owned);

    // =====================================================================
    // P9.86c: the dense scene shifted so that sdfBoxLower.x is EXACTLY +0
    // (meshLower.x + 0.5*spacing = -0.125 + 0.125, a computed sum, so never -0;
    // sdfCollision.cuh:71).  A localPos.x of -0 -- valid data: a coordinate
    // negated from zero, or rotated by a conjugated identity quaternion --
    // then TIES the lower face, and PxVec3::maximum's PxMax<float> is
    // selectMax (PxMath.h:82, PxUnixMathIntrinsics.h:122): the bound, +0,
    // survives.  clamped.x = +0, f.x = +0, diff.x = -0 in the CUDA; the generic
    // `a < b ? b : a` keeps -0 and flips f.x and diff.x.  At texel (0,5,4) = 0
    // the sample is +0, so with gradient case 4 (-2.5,0,0) the term
    // G.x*|dist| is -0 and dir.x = -0 + diff.x carries diff.x's sign into the
    // engine's contact normal (sdfCollision.cuh:215).
    // =====================================================================
    SdfScene dz;
    buildDenseAligned(dz);
    dz.meshLower.x = -0.125f;
    dz.coarse[denseIdx(dz, 0, 5, 4)] = 0.0f;       // dist == +0 at the dir point
    dz.coarse[denseIdx(dz, 1, 5, 4)] = -0.5f;      // its fall-back grad.x < 0
    BlobLayout blobDz = layOutBlob(dz, 5u, 9u, 1u, 0u, 0x0BADF00Du);
    const uint64_t hDz = sdfUploadTex(CU_AD_FORMAT_FLOAT, dz.dx, dz.dy, dz.dz,
                                      dz.coarse.data(), sizeof(float));
    PXG_CHECK_MSG(hDz != 0, "the zero-lower dense texture upload failed");
    if (hDz == 0) return;
    std::memcpy(&blobDz.bytes[blobDz.texObjOffset], &hDz, 8);

    SceneRun rz;
    rz.label = "dense-zero-lower";
    rz.s = &dz;
    rz.box = hostSdfBox(dz);
    rz.blobAddr = dev->getBufferDeviceAddress(mkBuf(blobDz.bytes));
    rz.expectDiscontinuity = true;
    PXG_CHECK_MSG(f2u(rz.box.lower.x) == 0u,
                  "dense-zero-lower: sdfBoxLower.x must be +0 (0x%08x)", f2u(rz.box.lower.x));
    addDenseAlignedPoints(rz);
    {
        const HostSdfBox& b = rz.box;
        auto nx = [](PxVec3 p) { p.x = -0.0f; return p; };
        // Indices 15..19: i % 5 picks gradient cases 0, 1, 2, 3, 4.
        rz.pts.push_back({ nx(posOfF(b, 0, 2, 4)),  0, 2, 4,  "neg0-x-tie" });
        // Not a tie: its -0 y and z lanes pin the fall-back tap
        // localPos + (dx,0,0) (sdfCollision.cuh:190), whose IEEE add makes them +0.
        rz.pts.push_back({ PxVec3(b.higher.x + 0.375f, -0.0f, -0.0f), 8, 0, 0, "neg0-yz-out" });
        rz.pts.push_back({ nx(posOfF(b, 0, 3, 7)),  0, 3, 7,  "neg0-x-tie-2" });
        rz.pts.push_back({ nx(posOfF(b, 0, 2, 13)), 0, 2, 12, "neg0-x-tie-out-z" });
        rz.pts.push_back({ nx(posOfF(b, 0, 6, 4)),  0, 5, 4,  "neg0-x-tie-zero-dist" });
    }
    checkScene(dev, rz, owned);
    PXG_CHECK_MSG(rz.tieClamp == 4,
                  "dense-zero-lower: %zu points tie -0 against the +0 lower face (want 4) -- "
                  "selectMax is indistinguishable from a<b?b:a otherwise", rz.tieClamp);
    PXG_CHECK_MSG(rz.tieDir >= 1,
                  "dense-zero-lower: no PxSdfSampleWithGrad dir depends on the tie's sign (%zu)",
                  rz.tieDir);
    PXG_CHECK_MSG(rz.tieDirV >= 1,
                  "dense-zero-lower: no doVertexSDFCollision dir depends on the tie's sign (%zu)",
                  rz.tieDirV);
    for (int k = 0; k < 3; ++k)
        PXG_CHECK_MSG(rz.foldTap[k] >= 1,
                      "dense-zero-lower: no fall-back tap %d has a -0 zero lane that its IEEE "
                      "add turns to +0 (%zu) -- the folded-lane revert would survive",
                      2 * k, rz.foldTap[k]);
    PXG_CHECK_MSG(rs.tieClamp == 0 && rd.tieClamp == 0,
                  "the sparse / dense-aligned scenes have no zero bound, so no tie (%zu, %zu)",
                  rs.tieClamp, rd.tieClamp);

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
    // D1's cascade is straddled only by a triple that satisfies THREE conditions at
    // once: d0 == d1 bitwise, and both strictly less than d2 AND than the centroid's
    // distance.  P9.34a added mirror-symmetric triangles which met the equality and
    // failed the centroid condition, so both cascade mutants survived.  Guessing
    // coordinates is the wrong tool -- the distances are GPU-only, so the fixture
    // SEARCHES for the triple instead, in two sampling passes.
    {
        // Pass A: a grid of candidates across and around the box.
        std::vector<PxVec4> cand;
        for (int gx = -2; gx <= 16; gx += 2)
            for (int gy = -2; gy <= 16; gy += 2)
                for (int gz = -2; gz <= 16; gz += 2)
                {
                    const PxVec3 q = posOfF(box, float(gx), float(gy), float(gz));
                    cand.push_back(PxVec4(q.x, q.y, q.z, 1.0f));
                }
        const uint32_t NC = uint32_t(cand.size());
        std::vector<float> outC(size_t(NC) * STRIDE_GRAD, 0.0f);
        PxgBufferHandle bCand = mkBuf(asBytes(cand.data(), cand.size() * sizeof(PxVec4)));
        PxgBufferHandle bOutC = mkBuf(asBytes(outC.data(), outC.size() * sizeof(float)));
        auto sampleDist = [&](PxgBufferHandle pts, PxgBufferHandle out, uint32_t n) {
            P_Coll pc = {};
            pc.meshPtr = blobAddr;
            pc.points = dev->getBufferDeviceAddress(pts);
            pc.gradIn = dev->getBufferDeviceAddress(bZero);
            pc.dirSeed = dev->getBufferDeviceAddress(bZero);
            pc.outBuf = dev->getBufferDeviceAddress(out);
            pc.numPoints = n;
            Kernel k(Kernel::Bindless{}, "npSdfCollision.spv", "sdfCollDistance", {}, sizeof(P_Coll));
            if (!k.isValid()) return false;
            k.dispatch((n + 63) / 64, {}, pc);
            dev->flush();
            return true;
        };
        bool found = false;
        if (sampleDist(bCand, bOutC, NC))
        {
            const auto C = downloadBuffer<float>(dev, bOutC, size_t(NC) * STRIDE_GRAD);
            auto dOf = [&](uint32_t i) { return C[size_t(i) * STRIDE_DIST + 8]; };

            // Pass B: for the equal-distance pairs, sample the centroids they would form.
            struct Cand { uint32_t i, j, k; };
            std::vector<Cand> triples;
            std::vector<PxVec4> cents;
            for (uint32_t i = 0; i < NC && cents.size() < 2000u; ++i)
                for (uint32_t j = i + 1; j < NC && cents.size() < 2000u; ++j)
                {
                    if (f2u(dOf(i)) != f2u(dOf(j))) continue;
                    for (uint32_t k = 0; k < NC && cents.size() < 2000u; ++k)
                    {
                        if (k == i || k == j || !(dOf(i) < dOf(k))) continue;
                        const PxVec3 pi(cand[i].x, cand[i].y, cand[i].z);
                        const PxVec3 pj(cand[j].x, cand[j].y, cand[j].z);
                        const PxVec3 pk(cand[k].x, cand[k].y, cand[k].z);
                        const PxVec3 ct = (pi + pj + pk) * (1.0f / 3.0f);
                        cents.push_back(PxVec4(ct.x, ct.y, ct.z, 1.0f));
                        triples.push_back({ i, j, k });
                    }
                }
            if (!cents.empty())
            {
                const uint32_t NCC = uint32_t(cents.size());
                std::vector<float> outCC(size_t(NCC) * STRIDE_GRAD, 0.0f);
                PxgBufferHandle bCent = mkBuf(asBytes(cents.data(), cents.size() * sizeof(PxVec4)));
                PxgBufferHandle bOutCC = mkBuf(asBytes(outCC.data(), outCC.size() * sizeof(float)));
                if (sampleDist(bCent, bOutCC, NCC))
                {
                    const auto CC = downloadBuffer<float>(dev, bOutCC, size_t(NCC) * STRIDE_GRAD);
                    for (uint32_t t = 0; t < NCC && !found; ++t)
                    {
                        const float dc = CC[size_t(t) * STRIDE_DIST + 8];
                        const Cand& c = triples[t];
                        if (!(dOf(c.i) < dc)) continue;     // the centroid must be SHALLOWER
                        tris.push_back({ PxVec3(cand[c.i].x, cand[c.i].y, cand[c.i].z),
                                         PxVec3(cand[c.j].x, cand[c.j].y, cand[c.j].z),
                                         PxVec3(cand[c.k].x, cand[c.k].y, cand[c.k].z),
                                         1.0f, "cascade-straddle" });
                        found = true;
                        std::printf("    [ note ] cascade straddle found: d0 = d1 = %.9g, "
                                    "d2 = %.9g, centroid %.9g\n", dOf(c.i), dOf(c.k), dc);
                    }
                }
            }
        }
        PXG_CHECK_MSG(found,
                      "no triple in the search grid satisfies d0 == d1 < min(d2, d_centroid), so "
                      "D1's asymmetric cascade -- whose second arm does not compare against d0 -- "
                      "is still indistinguishable from a three-way minimum");
    }

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

// ===========================================================================
// P9.35a -- the TETRAHEDRON pair, arranged exactly like the triangle pair above
// and for the same two reasons.
//
//   part 1  Gu::closestPtPointTetrahedron against THE ENGINE'S OWN COPY.  A second
//           oracle: PX_CUDA_CALLABLE, texture-free, so there is no mirror to be
//           wrong.  What it has to discriminate is the FACE SET -- the module
//           already contains a different four-face routine (closestPtTetrahedron,
//           hint-gated, normal-returning, faces (a,b,c)(a,b,d)(a,c,d)(b,c,d)), and
//           this one's faces are (a,b,c)(a,c,d)(a,d,b)(b,d,c).  Only the first
//           agrees.  Random tetrahedra are fed alongside the canonical simplex the
//           descent actually uses, because the engine copy handles both.
//
//   part 2  doTetrahedronSDFCollision, replayed from its own trace.  Same design as
//           the triangle descent: the shader logs the gradient it USED and the step
//           it went in with, the host rebuilds everything else and must land on the
//           logged newUvw bit for bit -- through the ENGINE's clamp, not a mirror.
//           One field is new, and it is the point of E3: `clamped`.  The projection
//           here is CONDITIONAL, and on an iteration that stays inside the simplex
//           it is the identity, so a port that clamps unconditionally is invisible
//           unless whether-it-ran is compared too.
// ===========================================================================
PXG_TEST(sdfTetrahedronCollision)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }

    const PxgKernelDef manifest[] = { { "npSdfCollision.spv", "sdfCollTetClamp", 0u, 0u } };
    static auto loader3 = [](const char* f) { return loadShaderBlob(f); };
    PxgCudaShim shim(dev, manifest, 1, +loader3);
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

    // ---- part 1: the tetrahedron clamp, against the engine's own copy ---------
    struct TetCase { PxVec4 p; PxVec4 a, b, c, d; };
    std::vector<TetCase> tc;
    const PxVec4 SA(1.0f, 0.0f, 0.0f, 0.0f), SB(0.0f, 1.0f, 0.0f, 0.0f);
    const PxVec4 SC(0.0f, 0.0f, 1.0f, 0.0f), SD(0.0f, 0.0f, 0.0f, 0.0f);
    auto addSimplex = [&](float x, float y, float z) {
        tc.push_back({ PxVec4(x, y, z, 0.0f), SA, SB, SC, SD });
    };
    // The canonical simplex, which is what the descent calls it with.  Interior,
    // each face's outside, each vertex's corner, each edge, and far away.
    addSimplex(0.25f, 0.25f, 0.25f);
    addSimplex(0.1f, 0.1f, 0.1f);
    addSimplex(2.0f, 0.0f, 0.0f);
    addSimplex(0.0f, 2.0f, 0.0f);
    addSimplex(0.0f, 0.0f, 2.0f);
    addSimplex(-1.0f, -1.0f, -1.0f);
    addSimplex(0.5f, 0.5f, 0.5f);
    addSimplex(-0.5f, 0.5f, 0.5f);
    addSimplex(0.5f, -0.5f, 0.5f);
    addSimplex(0.5f, 0.5f, -0.5f);
    addSimplex(1.0f, 1.0f, 1.0f);
    addSimplex(0.0f, 0.0f, 0.0f);
    addSimplex(1.0f / 3, 1.0f / 3, 1.0f / 3);
    // A point EQUIDISTANT from two faces of the canonical simplex, so G5's
    // tie-break (strict `<`, the earlier face wins) is exercised rather than
    // assumed.  On the canonical simplex the (a,c,d) and (a,d,b) faces are the
    // y = 0 and z = 0 planes; a point with y == z outside both is equidistant.
    addSimplex(0.3f, -0.4f, -0.4f);
    addSimplex(-0.4f, 0.3f, -0.4f);
    addSimplex(-0.4f, -0.4f, 0.3f);
    // General tetrahedra, so the face ORDER and WINDING are discriminated on
    // shapes where the four faces are genuinely different planes.  A degenerate
    // (all four corners equal) and a flat one (four coplanar corners) are in the
    // set on purpose: closestPtPointTriangle2's own denominators vanish there and
    // the engine's answer is whatever it is -- which is exactly what has to match.
    std::mt19937 rngT(0x7E7A1u);
    auto rr = [&]() { return (float(rngT() % 400001u) / 100000.0f) - 2.0f; };
    for (uint32_t i = 0; i < 3000u; ++i)
        tc.push_back({ PxVec4(rr(), rr(), rr(), 0.0f), PxVec4(rr(), rr(), rr(), 0.0f),
                       PxVec4(rr(), rr(), rr(), 0.0f), PxVec4(rr(), rr(), rr(), 0.0f),
                       PxVec4(rr(), rr(), rr(), 0.0f) });
    {
        const PxVec4 q(0.3f, 0.4f, 0.5f, 0.0f);
        tc.push_back({ q, q, q, q, q });                                   // degenerate
        tc.push_back({ PxVec4(0.5f, 0.5f, 1.0f, 0.0f), PxVec4(0.0f, 0.0f, 0.0f, 0.0f),
                       PxVec4(1.0f, 0.0f, 0.0f, 0.0f), PxVec4(0.0f, 1.0f, 0.0f, 0.0f),
                       PxVec4(1.0f, 1.0f, 0.0f, 0.0f) });                  // coplanar
    }
    const uint32_t NC = uint32_t(tc.size());

    std::vector<PxVec4> pts(NC), corners(size_t(NC) * 4u);
    for (uint32_t i = 0; i < NC; ++i)
    {
        pts[i] = tc[i].p;
        corners[size_t(i) * 4u + 0] = tc[i].a;
        corners[size_t(i) * 4u + 1] = tc[i].b;
        corners[size_t(i) * 4u + 2] = tc[i].c;
        corners[size_t(i) * 4u + 3] = tc[i].d;
    }
    const std::vector<PxVec4> zero4(NC, PxVec4(0.0f));
    const std::vector<float> outInit(size_t(NC) * 4u, -1234.0f);
    PxgBufferHandle bPts = mkBuf(asBytes(pts.data(), pts.size() * sizeof(PxVec4)));
    PxgBufferHandle bCorners = mkBuf(asBytes(corners.data(), corners.size() * sizeof(PxVec4)));
    PxgBufferHandle bZero = mkBuf(asBytes(zero4.data(), zero4.size() * sizeof(PxVec4)));
    PxgBufferHandle bOut = mkBuf(asBytes(outInit.data(), outInit.size() * sizeof(float)));

    P_Coll pc = {};
    pc.points = dev->getBufferDeviceAddress(bPts);
    pc.gradIn = dev->getBufferDeviceAddress(bZero);
    pc.dirSeed = dev->getBufferDeviceAddress(bZero);
    pc.outBuf = dev->getBufferDeviceAddress(bOut);
    pc.numPoints = NC;
    pc.triVerts = dev->getBufferDeviceAddress(bCorners);
    {
        Kernel k(Kernel::Bindless{}, "npSdfCollision.spv", "sdfCollTetClamp", {}, sizeof(P_Coll));
        PXG_CHECK_MSG(k.isValid(), "bindless pipeline creation succeeds for sdfCollTetClamp");
        if (!k.isValid()) { for (auto h : owned) dev->destroyBuffer(h); return; }
        k.dispatch((NC + 63) / 64, {}, pc);
    }
    dev->flush();
    const auto C = downloadBuffer<float>(dev, bOut, size_t(NC) * 4u);

    size_t badC = 0;
    uint32_t faceWin[4] = { 0, 0, 0, 0 };
    uint32_t nTiedFace = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const PxVec3 p(tc[i].p.x, tc[i].p.y, tc[i].p.z);
        const PxVec3 a(tc[i].a.x, tc[i].a.y, tc[i].a.z);
        const PxVec3 b(tc[i].b.x, tc[i].b.y, tc[i].b.z);
        const PxVec3 c(tc[i].c.x, tc[i].c.y, tc[i].c.z);
        const PxVec3 d(tc[i].d.x, tc[i].d.y, tc[i].d.z);
        const PxVec3 want = physx::Gu::closestPtPointTetrahedron(p, a, b, c, d);   // THE ENGINE
        const float* o = &C[size_t(i) * 4u];
        // Non-finite inputs produce non-finite answers on both sides; a bit compare
        // of a NaN never matches, so those are compared as bit patterns only when
        // both are finite and skipped otherwise -- which is a real skip, so the run
        // insists below that the finite cases dominate.
        const bool finite = std::isfinite(want.x) && std::isfinite(want.y) &&
                            std::isfinite(want.z);
        if (finite && (f2u(o[0]) != f2u(want.x) || f2u(o[1]) != f2u(want.y) ||
                       f2u(o[2]) != f2u(want.z)))
        {
            if (badC < 4)
                std::printf("    tet[%u] p (%.9g %.9g %.9g) -> (%.9g %.9g %.9g), engine says "
                            "(%.9g %.9g %.9g)\n", i, p.x, p.y, p.z, o[0], o[1], o[2],
                            want.x, want.y, want.z);
            ++badC;
        }

        // Which FACE won, re-derived the way the function branches, so the run can
        // insist all four were the winner somewhere.  Without this a port that drops
        // a face entirely passes whenever that face never wins.
        const PxVec3 ab = b - a, ac = c - a, ad = d - a, bc = c - b, bd = d - b;
        const PxVec3 f0 = physx::Gu::closestPtPointTriangle2(p, a, b, c, ab, ac);
        const PxVec3 f1 = physx::Gu::closestPtPointTriangle2(p, a, c, d, ac, ad);
        const PxVec3 f2 = physx::Gu::closestPtPointTriangle2(p, a, d, b, ad, ab);
        const PxVec3 f3 = physx::Gu::closestPtPointTriangle2(p, b, d, c, bd, bc);
        const PxVec3 fs[4] = { f0, f1, f2, f3 };
        float best = (f0 - p).magnitudeSquared();
        uint32_t win = 0;
        for (uint32_t k = 1; k < 4u; ++k)
        {
            const float sq = (fs[k] - p).magnitudeSquared();
            if (sq < best) { best = sq; win = k; }
        }
        if (finite) ++faceWin[win];
        // A tie between two faces at DIFFERENT points is what G5 decides; count them.
        for (uint32_t k = 0; k < 4u; ++k)
            if (k != win && (fs[k] - p).magnitudeSquared() == best &&
                (f2u(fs[k].x) != f2u(fs[win].x) || f2u(fs[k].y) != f2u(fs[win].y) ||
                 f2u(fs[k].z) != f2u(fs[win].z)))
            { ++nTiedFace; break; }
    }
    PXG_CHECK_MSG(badC == 0, "%zu of %u tetrahedron clamps differ from the engine's own "
                  "Gu::closestPtPointTetrahedron", badC, NC);
    for (uint32_t k = 0; k < 4u; ++k)
        PXG_CHECK_MSG(faceWin[k] > 0,
                      "face %u of the tetrahedron clamp never won (%u inputs); a face that never "
                      "wins is a face the comparison cannot see", k, NC);
    PXG_CHECK_MSG(nTiedFace > 0,
                  "no input made two faces tie at DIFFERENT points, so G5 -- the strict `<` that "
                  "keeps the EARLIER face -- is indistinguishable from `<=`");

    std::printf("    [ note ] tet clamp: %u inputs bit-exact against the engine's own copy; "
                "face wins %u/%u/%u/%u, %u cross-face ties at different points\n",
                NC, faceWin[0], faceWin[1], faceWin[2], faceWin[3], nTiedFace);

    // ---- part 2: the tetrahedron descent, replayed from its own trace ---------
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

    struct Tet { PxVec3 a, b, c, d; float tol; const char* name; };
    std::vector<Tet> tets;
    tets.push_back({ posOfF(box, 3, 3, 3), posOfF(box, 9, 3, 3), posOfF(box, 3, 9, 3),
                     posOfF(box, 3, 3, 9), 1.0f, "straddling" });
    tets.push_back({ posOfF(box, 5, 5, 5), posOfF(box, 6, 5, 5), posOfF(box, 5, 6, 5),
                     posOfF(box, 5, 5, 6), 1.0f, "interior" });
    tets.push_back({ posOfF(box, -6, -6, -6), posOfF(box, -5, -6, -6), posOfF(box, -6, -5, -6),
                     posOfF(box, -6, -6, -5), 1.0f, "outside" });
    // difference == 0, so the initial step is +inf and everything after it is NaN.
    tets.push_back({ posOfF(box, 4, 4, 4), posOfF(box, 4, 4, 4), posOfF(box, 4, 4, 4),
                     posOfF(box, 4, 4, 4), 1.0f, "degenerate" });
    // Wide enough that the gradient step leaves the simplex, so E3's projection runs.
    tets.push_back({ posOfF(box, 1, 8, 4), posOfF(box, 14, 8, 4), posOfF(box, 7, 1, 12),
                     posOfF(box, 7, 14, 12), 1.0f, "wide" });
    // Small enough that it does not, so the CONDITIONAL half of E3 is reached too --
    // and the run below insists both happened rather than hoping.
    tets.push_back({ posOfF(box, 7, 7, 7), posOfF(box, 7.2f, 7, 7), posOfF(box, 7, 7.2f, 7),
                     posOfF(box, 7, 7, 7.2f), 1.0f, "tiny" });
    tets.push_back({ posOfF(box, 1, 1, 1), posOfF(box, 14, 14, 1), posOfF(box, 1, 14, 14),
                     posOfF(box, 14, 1, 14), 0.001f, "tight-tolerance" });
    // E1's cascade never compares d1 against d0, so the arms are straddled only by a
    // tie.  Mirror-symmetric corners about the grid centre make d0 == d1 bitwise.
    tets.push_back({ posOfF(box, 4, 6, 6), posOfF(box, 8, 6, 6), posOfF(box, 6, 11, 6),
                     posOfF(box, 6, 6, 11), 1.0f, "d0-eq-d1" });
    tets.push_back({ posOfF(box, 6, 4, 6), posOfF(box, 6, 8, 6), posOfF(box, 11, 6, 6),
                     posOfF(box, 6, 6, 11), 1.0f, "d1-eq-d2" });
    // A tet collapsed onto the point where the dense scene's gradient vanishes, so
    // E7's discontinuity substitution is reached in iteration 0.
    {
        const PxVec3 ctr = posOfF(box, 6, 6, 6);
        tets.push_back({ ctr, ctr, ctr, ctr, 1.0f, "discontinuity" });
    }
    // ---- the eps-BAND candidates, constructed rather than hoped for ----------
    // The gate `x <= -eps || x >= 1+eps` is a four-way disjunction, so a port that
    // gets the SIGN of eps wrong clamps on strictly more iterations -- and every
    // extra clamp it makes is the IDENTITY (the point is already in the simplex),
    // so the only trace of it is the flag.  The first sweep found no iteration that
    // distinguishes them, because in every tet above at least one component escapes
    // far enough that both spellings clamp.
    //
    // What distinguishes them is an iteration where NOTHING escapes the real gate
    // but something sits inside the narrow band a negated eps would catch: a
    // component within 1e-4 of 0, or within 1e-4 of 1.  That is CONSTRUCTIBLE.
    // Take a tet whose last three vertices COINCIDE; then, starting from vertex 0,
    //     dfdu = grad . (v0 - p) = 0 exactly     (p IS v0)
    //     dfdv = dfdw = dfd4 = grad . (v1 - v0)
    // so x stays exactly 1.0 and the other three move by the same
    //     s = step * dfdv = cos(grad, v1 - v0) / 2
    // because step is 1/(2*|v1 - v0|) and |v1 - v0| IS the difference.  Choose
    // v1 - v0 PERPENDICULAR to the gradient and s collapses to a rounding, well
    // under 1e-4 -- so no component escapes, the real gate stays quiet, and the
    // negated one fires on x == 1.0.
    //
    // The gradient is GPU-only, so it is measured rather than assumed: sample it at
    // each base point, build the perpendicular from what comes back, and offer both
    // signs and several lengths because arm 0 also needs d0 to be the strict
    // minimum, which the SDF's curvature along the perpendicular decides.
    {
        std::vector<PxVec4> basePts;
        for (int gx = 2; gx <= 12; gx += 2)
            for (int gy = 2; gy <= 12; gy += 5)
                for (int gz = 2; gz <= 12; gz += 5)
                {
                    const PxVec3 q = posOfF(box, float(gx), float(gy), float(gz));
                    basePts.push_back(PxVec4(q.x, q.y, q.z, 0.0f));
                }
        const uint32_t NBP = uint32_t(basePts.size());
        std::vector<float> gOut(size_t(NBP) * STRIDE_GRAD, 0.0f);
        PxgBufferHandle bBase = mkBuf(asBytes(basePts.data(), basePts.size() * sizeof(PxVec4)));
        PxgBufferHandle bGOut = mkBuf(asBytes(gOut.data(), gOut.size() * sizeof(float)));
        P_Coll pg = {};
        pg.meshPtr = blobAddr;
        pg.points = dev->getBufferDeviceAddress(bBase);
        pg.gradIn = dev->getBufferDeviceAddress(bZero);
        pg.dirSeed = dev->getBufferDeviceAddress(bZero);
        pg.outBuf = dev->getBufferDeviceAddress(bGOut);
        pg.numPoints = NBP;
        Kernel kg(Kernel::Bindless{}, "npSdfCollision.spv", "sdfCollGrad", {}, sizeof(P_Coll));
        if (kg.isValid())
        {
            kg.dispatch((NBP + 63) / 64, {}, pg);
            dev->flush();
            const auto G = downloadBuffer<float>(dev, bGOut, size_t(NBP) * STRIDE_GRAD);
            const float ext = (box.higher - box.lower).magnitude();
            for (uint32_t i = 0; i < NBP; ++i)
            {
                const PxVec3 g(G[size_t(i) * STRIDE_GRAD + 1], G[size_t(i) * STRIDE_GRAD + 2],
                               G[size_t(i) * STRIDE_GRAD + 3]);
                if (g.magnitudeSquared() <= 0.0f) continue;
                const PxVec3 gn = g.getNormalized();
                const PxVec3 aux = (PxAbs(gn.x) < 0.9f) ? PxVec3(1.0f, 0.0f, 0.0f)
                                                        : PxVec3(0.0f, 1.0f, 0.0f);
                const PxVec3 perp = gn.cross(aux).getNormalized();
                const PxVec3 v0 = PxVec3(basePts[i].x, basePts[i].y, basePts[i].z);
                for (int sgn = -1; sgn <= 1; sgn += 2)
                    for (int lk = 0; lk < 3; ++lk)
                    {
                        const float L = (0.02f + 0.03f * float(lk)) * ext * float(sgn);
                        const PxVec3 v1 = v0 + perp * L;
                        tets.push_back({ v0, v1, v1, v1, 1.0f, "eps-band" });
                    }
            }
        }
    }

    // ---- the CASCADE straddle, searched for rather than guessed ---------------
    // E1's second arm never compares d1 against d0, so it parts company with a
    // four-way minimum on exactly one configuration: d0 == d1 BITWISE, and both
    // strictly below d2, d3 and the centroid's distance.  Then the first arm's
    // `d0 < d1` is false, the second arm fires, and a "fixed" cascade would have
    // started at v0 instead.  The distances are GPU-only, so guessing coordinates
    // is the wrong tool -- P9.34b established the right one, and this is it applied
    // to four corners instead of three.
    {
        std::vector<PxVec4> cand;
        for (int gx = -2; gx <= 16; gx += 2)
            for (int gy = -2; gy <= 16; gy += 2)
                for (int gz = -2; gz <= 16; gz += 2)
                {
                    const PxVec3 q = posOfF(box, float(gx), float(gy), float(gz));
                    cand.push_back(PxVec4(q.x, q.y, q.z, 1.0f));
                }
        const uint32_t NCa = uint32_t(cand.size());
        std::vector<float> outC(size_t(NCa) * STRIDE_GRAD, 0.0f);
        PxgBufferHandle bCand = mkBuf(asBytes(cand.data(), cand.size() * sizeof(PxVec4)));
        PxgBufferHandle bOutC = mkBuf(asBytes(outC.data(), outC.size() * sizeof(float)));
        auto sampleDist = [&](PxgBufferHandle pts, PxgBufferHandle out, uint32_t n) {
            P_Coll pcd = {};
            pcd.meshPtr = blobAddr;
            pcd.points = dev->getBufferDeviceAddress(pts);
            pcd.gradIn = dev->getBufferDeviceAddress(bZero);
            pcd.dirSeed = dev->getBufferDeviceAddress(bZero);
            pcd.outBuf = dev->getBufferDeviceAddress(out);
            pcd.numPoints = n;
            Kernel k(Kernel::Bindless{}, "npSdfCollision.spv", "sdfCollDistance", {},
                     sizeof(P_Coll));
            if (!k.isValid()) return false;
            k.dispatch((n + 63) / 64, {}, pcd);
            dev->flush();
            return true;
        };
        if (sampleDist(bCand, bOutC, NCa))
        {
            const auto C2 = downloadBuffer<float>(dev, bOutC, size_t(NCa) * STRIDE_GRAD);
            auto dOf = [&](uint32_t i) { return C2[size_t(i) * STRIDE_DIST + 8]; };
            auto pOf = [&](uint32_t i) { return PxVec3(cand[i].x, cand[i].y, cand[i].z); };

            struct Quad { uint32_t a, b, c, d; };
            std::vector<Quad> quads;
            for (uint32_t i = 0; i < NCa && quads.size() < 24u; ++i)
                for (uint32_t j = i + 1; j < NCa && quads.size() < 24u; ++j)
                {
                    if (f2u(dOf(i)) != f2u(dOf(j))) continue;
                    if (!std::isfinite(dOf(i))) continue;
                    const float V = dOf(i);
                    uint32_t c1 = NCa, c2 = NCa;
                    for (uint32_t k = 0; k < NCa; ++k)
                    {
                        if (k == i || k == j || !(dOf(k) > V)) continue;
                        if (c1 == NCa) c1 = k;
                        else if (c2 == NCa) { c2 = k; break; }
                    }
                    if (c1 != NCa && c2 != NCa) quads.push_back({ i, j, c1, c2 });
                }

            // Pass B: the centroid each quad would form, sampled, so the last
            // condition -- d0 below the CENTROID's distance -- is measured too.
            if (!quads.empty())
            {
                std::vector<PxVec4> ctr;
                for (const Quad& q : quads)
                {
                    const PxVec3 m = (pOf(q.a) + pOf(q.b) + pOf(q.c) + pOf(q.d)) * 0.25f;
                    ctr.push_back(PxVec4(m.x, m.y, m.z, 1.0f));
                }
                const uint32_t NQ = uint32_t(ctr.size());
                std::vector<float> outQ(size_t(NQ) * STRIDE_GRAD, 0.0f);
                PxgBufferHandle bCtr = mkBuf(asBytes(ctr.data(), ctr.size() * sizeof(PxVec4)));
                PxgBufferHandle bOutQ = mkBuf(asBytes(outQ.data(), outQ.size() * sizeof(float)));
                if (sampleDist(bCtr, bOutQ, NQ))
                {
                    const auto Q = downloadBuffer<float>(dev, bOutQ, size_t(NQ) * STRIDE_GRAD);
                    uint32_t added = 0;
                    for (uint32_t k = 0; k < NQ && added < 3u; ++k)
                    {
                        const float V = dOf(quads[k].a);
                        if (!(Q[size_t(k) * STRIDE_DIST + 8] > V)) continue;
                        tets.push_back({ pOf(quads[k].a), pOf(quads[k].b), pOf(quads[k].c),
                                         pOf(quads[k].d), 1.0f, "cascade-straddle" });
                        ++added;
                    }
                }
            }
        }
    }

    const uint32_t NT = uint32_t(tets.size());

    const uint32_t TET_STRIDE = 256u;
    std::vector<PxVec4> tv(size_t(NT) * 4u);
    for (uint32_t i = 0; i < NT; ++i)
    {
        tv[size_t(i) * 4u + 0] = PxVec4(tets[i].a.x, tets[i].a.y, tets[i].a.z, tets[i].tol);
        tv[size_t(i) * 4u + 1] = PxVec4(tets[i].b.x, tets[i].b.y, tets[i].b.z, 0.0f);
        tv[size_t(i) * 4u + 2] = PxVec4(tets[i].c.x, tets[i].c.y, tets[i].c.z, 0.0f);
        tv[size_t(i) * 4u + 3] = PxVec4(tets[i].d.x, tets[i].d.y, tets[i].d.z, 0.0f);
    }
    const std::vector<PxVec4> dirSeed(NT, PxVec4(0.0f));
    const std::vector<float> outTInit(size_t(NT) * TET_STRIDE, -1.0f);
    PxgBufferHandle bTv = mkBuf(asBytes(tv.data(), tv.size() * sizeof(PxVec4)));
    PxgBufferHandle bDir = mkBuf(asBytes(dirSeed.data(), dirSeed.size() * sizeof(PxVec4)));
    PxgBufferHandle bOutT = mkBuf(asBytes(outTInit.data(), outTInit.size() * sizeof(float)));

    P_Coll pt = {};
    pt.meshPtr = blobAddr;
    pt.points = dev->getBufferDeviceAddress(bPts);
    pt.gradIn = dev->getBufferDeviceAddress(bZero);
    pt.dirSeed = dev->getBufferDeviceAddress(bDir);
    pt.outBuf = dev->getBufferDeviceAddress(bOut);
    pt.numPoints = NT;
    pt.triVerts = dev->getBufferDeviceAddress(bTv);
    pt.triOut = dev->getBufferDeviceAddress(bOutT);
    {
        Kernel k(Kernel::Bindless{}, "npSdfCollision.spv", "sdfCollTetrahedron", {}, sizeof(P_Coll));
        PXG_CHECK_MSG(k.isValid(), "bindless pipeline creation succeeds for sdfCollTetrahedron");
        if (!k.isValid()) { for (auto h : owned) dev->destroyBuffer(h); return; }
        k.dispatch((NT + 63) / 64, {}, pt);
    }
    dev->flush();
    const auto T = downloadBuffer<float>(dev, bOutT, size_t(NT) * TET_STRIDE);

    size_t badStart = 0, badDiff = 0, badStep = 0, badUvw = 0, badBreak = 0, badGradN = 0;
    size_t badIters = 0, badFinalG = 0, badClamp = 0, badFinalP = 0;
    uint32_t armSeen[5] = { 0, 0, 0, 0, 0 };
    uint32_t nConverged = 0, nRanOut = 0, nSubstituted = 0, nTied = 0;
    uint32_t nClampRan = 0, nClampSkipped = 0, nFinalGChecked = 0, nNanMag = 0;
    uint32_t nFinalPChecked = 0, nEpsBand = 0, nStraddle = 0;
    for (uint32_t i = 0; i < NT; ++i)
    {
        const float* o = &T[size_t(i) * TET_STRIDE];
        const PxVec3 v0 = tets[i].a, v1 = tets[i].b, v2 = tets[i].c, v3 = tets[i].d;
        const float d0 = o[0], d1 = o[1], d2 = o[2], d3 = o[3], dC = o[4];
        const uint32_t arm = uint32_t(o[5]);
        armSeen[arm < 5u ? arm : 4u]++;
        if (f2u(d0) == f2u(d1) || f2u(d1) == f2u(d2) || f2u(d2) == f2u(d3)) ++nTied;
        // The one configuration that separates E1's asymmetric cascade from a
        // four-way minimum: d0 == d1 bitwise, and both strictly below everything
        // else, so the FIRST arm's `d0 < d1` fails and the SECOND fires.
        if (f2u(d0) == f2u(d1) && d0 < d2 && d0 < d3 && d0 < dC) ++nStraddle;

        // E1 / E2: the cascade, re-derived from the five traced distances.
        PxVec3 wantP = (v0 + v1 + v2 + v3) * 0.25f;
        PxVec4 wantUvw(0.0f);
        uint32_t wantArm;
        if (d0 < d1 && d0 < d2 && d0 < d3 && d0 < dC) { wantP = v0; wantUvw.x = 1.0f; wantArm = 0; }
        else if (d1 < d2 && d1 < d3 && d1 < dC)       { wantP = v1; wantUvw.y = 1.0f; wantArm = 1; }
        else if (d2 < d3 && d2 < dC)                  { wantP = v2; wantUvw.z = 1.0f; wantArm = 2; }
        else if (d3 < dC)                             { wantP = v3; wantUvw.w = 1.0f; wantArm = 3; }
        else { wantUvw = PxVec4(0.25f); wantArm = 4; }
        if (arm != wantArm || !bitEq(o[7], wantUvw.x) || !bitEq(o[8], wantUvw.y) ||
            !bitEq(o[9], wantUvw.z) || !bitEq(o[10], wantUvw.w) || !bitEq3(o + 11, wantP))
        {
            if (badStart < 4)
                std::printf("    tet %s: start arm %u want %u, uvw (%.9g %.9g %.9g %.9g)\n",
                            tets[i].name, arm, wantArm, o[7], o[8], o[9], o[10]);
            ++badStart;
        }

        // Nested max of SQUARED lengths, one sqrt, and the nesting is
        // max(max(e0,e1), max(e2,e3)) -- a left fold rounds the same here but the
        // shape is transcribed, so the mirror uses the shape too.
        const float wantDifference = std::sqrt(
            PxMax(PxMax((v0 - wantP).magnitudeSquared(), (v1 - wantP).magnitudeSquared()),
                  PxMax((v2 - wantP).magnitudeSquared(), (v3 - wantP).magnitudeSquared())));
        if (f2u(o[6]) != f2u(wantDifference))
        { if (badDiff < 4) std::printf("    tet %s: difference %.9g want %.9g\n",
                                       tets[i].name, o[6], wantDifference); ++badDiff; }

        const uint32_t iters = uint32_t(o[14]);
        const uint32_t conv = uint32_t(o[15]);
        if (conv) ++nConverged; else ++nRanOut;
        if (iters < 1u || iters > 16u || (conv == 0u && iters != 16u))
        { if (badIters < 4)
              std::printf("    tet %s: %u iterations, converged %u\n", tets[i].name, iters, conv);
          ++badIters; }

        // The gradient is RECOMPUTED at the final p.  Exempt where the arithmetic is
        // not finite, for the same reason as the triangle version: with a degenerate
        // tet every value downstream is NaN and the two are the same bit pattern
        // whichever the port produced.
        const PxVec3 gFinal(o[19], o[20], o[21]);
        const bool gfinite = std::isfinite(gFinal.x) && std::isfinite(gFinal.y) &&
                             std::isfinite(gFinal.z);
        if (gfinite)
        {
            ++nFinalGChecked;
            const float* le = o + 32 + (iters - 1u) * 14;
            if (bitEq3(o + 19, PxVec3(le[6], le[7], le[8])))
            { if (badFinalG < 4)
                  std::printf("    tet %s: gradFinal is bit-identical to the loop's last USED "
                              "gradient; it must be a fresh sample\n", tets[i].name);
              ++badFinalG; }
        }

        // E5: the returned point is the blend of the LAST newUvw, and it is rebuilt
        // on every iteration whether or not the projection ran.  Without comparing
        // it, an iteration that skips the clamp can leave p stale and nothing sees
        // it -- the replay derives p from uvw and never looks at what came back.
        {
            const float* le = o + 32 + (iters - 1u) * 14;
            const PxVec4 lastUvw(le[2], le[3], le[4], le[5]);
            const PxVec3 wantPFinal = v0 * lastUvw.x + v1 * lastUvw.y +
                                      v2 * lastUvw.z + v3 * lastUvw.w;
            const bool pfinite = std::isfinite(wantPFinal.x) && std::isfinite(wantPFinal.y) &&
                                 std::isfinite(wantPFinal.z);
            if (pfinite)
            {
                ++nFinalPChecked;
                if (!bitEq3(o + 16, wantPFinal) || !bitEq3(o + 22, wantPFinal))
                {
                    if (badFinalP < 4)
                        std::printf("    tet %s: pFinal (%.9g %.9g %.9g) / point (%.9g %.9g %.9g) "
                                    "want (%.9g %.9g %.9g)\n", tets[i].name, o[16], o[17], o[18],
                                    o[22], o[23], o[24], wantPFinal.x, wantPFinal.y, wantPFinal.z);
                    ++badFinalP;
                }
            }
        }

        // The replay.
        PxVec4 uvw = wantUvw;
        float step = 1.0f / (2.0f * wantDifference);
        for (uint32_t k = 0; k < iters; ++k)
        {
            const float* e = o + 32 + k * 14;
            const float stepIn = e[0], gradMag = e[1];
            const PxVec3 gradUsed(e[6], e[7], e[8]);
            const uint32_t clamped = uint32_t(e[9]);
            const PxVec3 gradRaw(e[10], e[11], e[12]);
            // THE RELATION, not just the magnitude.  PxVec3::normalize() is two
            // separate `if`s upstream -- `if (m > 0) divide` and then `if (m == 0)
            // substitute`, with no else -- so a NaN magnitude satisfies NEITHER and
            // the raw vector must survive untouched.  Turning the pair into an
            // if/else is invisible to a replay that takes gradUsed from the shader,
            // which is exactly what the first sweep caught.
            if (gradMag > 0.0f)
            {
                const PxVec3 want = gradRaw * (1.0f / gradMag);
                if (f2u(gradUsed.x) != f2u(want.x) || f2u(gradUsed.y) != f2u(want.y) ||
                    f2u(gradUsed.z) != f2u(want.z))
                { if (badGradN < 4)
                      std::printf("    tet %s iter %u: gradUsed (%.9g %.9g %.9g) is not the raw "
                                  "gradient over its magnitude (%.9g %.9g %.9g)\n", tets[i].name,
                                  k, gradUsed.x, gradUsed.y, gradUsed.z, want.x, want.y, want.z);
                  ++badGradN; }
            }
            else if (gradMag == 0.0f)
            {
                ++nSubstituted;
                const PxVec3 want =
                    PxVec3(0.5448762f, 0.672269f, 0.398825f).getNormalized();
                if (f2u(gradUsed.x) != f2u(want.x) || f2u(gradUsed.y) != f2u(want.y) ||
                    f2u(gradUsed.z) != f2u(want.z))
                { if (badGradN < 4)
                      std::printf("    tet %s iter %u: substitute (%.9g %.9g %.9g) is not the "
                                  "normalised constant\n", tets[i].name, k,
                                  gradUsed.x, gradUsed.y, gradUsed.z); ++badGradN; }
            }
            else
            {
                // A NaN magnitude: neither `if` fires, so gradUsed IS gradRaw, bit
                // for bit -- including the NaN payload.
                ++nNanMag;
                if (f2u(gradUsed.x) != f2u(gradRaw.x) || f2u(gradUsed.y) != f2u(gradRaw.y) ||
                    f2u(gradUsed.z) != f2u(gradRaw.z))
                { if (badGradN < 4)
                      std::printf("    tet %s iter %u: gradMag is NaN, so gradUsed must be the "
                                  "RAW gradient untouched -- got (%.9g %.9g %.9g) against "
                                  "(%.9g %.9g %.9g)\n", tets[i].name, k, gradUsed.x, gradUsed.y,
                                  gradUsed.z, gradRaw.x, gradRaw.y, gradRaw.z);
                  ++badGradN; }
            }
            if (f2u(stepIn) != f2u(step))
            { if (badStep < 4)
                  std::printf("    tet %s iter %u: step %.9g want %.9g\n",
                              tets[i].name, k, stepIn, step); ++badStep; }

            // Iteration 0 descends from the point the cascade chose -- the CENTROID in
            // the else arm, not the barycentric blend, which would differ by a rounding.
            const PxVec3 p = (k == 0) ? wantP
                                      : (v0 * uvw.x + v1 * uvw.y + v2 * uvw.z + v3 * uvw.w);
            const float du = gradUsed.dot(v0 - p);
            const float dv = gradUsed.dot(v1 - p);
            const float dw = gradUsed.dot(v2 - p);
            const float d4 = gradUsed.dot(v3 - p);
            PxVec4 newUVW = uvw;
            newUVW.x -= step * du;
            newUVW.y -= step * dv;
            newUVW.z -= step * dw;
            newUVW.w -= step * d4;
            step = step * 0.8f;

            // E3: the escape test, non-strict at both ends, four-way.
            const float eps = 1e-4f;
            const bool wantClamp =
                ((newUVW.x <= -eps || newUVW.x >= 1.0f + eps) ||
                 (newUVW.y <= -eps || newUVW.y >= 1.0f + eps) ||
                 (newUVW.z <= -eps || newUVW.z >= 1.0f + eps) ||
                 (newUVW.w <= -eps || newUVW.w >= 1.0f + eps));
            if (wantClamp)
            {
                // E8: THE ENGINE's clamp, on the CANONICAL simplex.
                const PxVec3 cp = physx::Gu::closestPtPointTetrahedron(
                    PxVec3(newUVW.x, newUVW.y, newUVW.z),
                    PxVec3(1.0f, 0.0f, 0.0f), PxVec3(0.0f, 1.0f, 0.0f),
                    PxVec3(0.0f, 0.0f, 1.0f), PxVec3(0.0f));
                newUVW.x = cp.x; newUVW.y = cp.y; newUVW.z = cp.z;
                newUVW.w = 1.0f - cp.x - cp.y - cp.z;      // E4
                ++nClampRan;
            }
            else
            {
                ++nClampSkipped;
            }
            // E3 is only observable if WHETHER the clamp ran is compared: on an
            // iteration that stays inside the simplex the projection is the identity,
            // so an unconditional port produces the same newUvw.
            // The iterations that tell a POSITIVE eps from a negative one: the real
            // gate stays quiet while a negated one would fire.
            if (!wantClamp)
            {
                const float band = 1e-4f;
                if (newUVW.x <= band || newUVW.x >= 1.0f - band ||
                    newUVW.y <= band || newUVW.y >= 1.0f - band ||
                    newUVW.z <= band || newUVW.z >= 1.0f - band ||
                    newUVW.w <= band || newUVW.w >= 1.0f - band)
                    ++nEpsBand;
            }
            if ((clamped != 0u) != wantClamp)
            { if (badClamp < 4)
                  std::printf("    tet %s iter %u: clamped %u want %d\n",
                              tets[i].name, k, clamped, int(wantClamp)); ++badClamp; }

            if (!bitEq(e[2], newUVW.x) || !bitEq(e[3], newUVW.y) ||
                !bitEq(e[4], newUVW.z) || !bitEq(e[5], newUVW.w))
            {
                if (badUvw < 4)
                    std::printf("    tet %s iter %u: newUvw (%.9g %.9g %.9g %.9g) want "
                                "(%.9g %.9g %.9g %.9g)\n", tets[i].name, k, e[2], e[3], e[4], e[5],
                                newUVW.x, newUVW.y, newUVW.z, newUVW.w);
                ++badUvw;
            }

            // E6: FOUR components, and the break precedes the assignment.
            const PxVec4 duvw = uvw - newUVW;
            const float dsq = duvw.x * duvw.x + duvw.y * duvw.y +
                              duvw.z * duvw.z + duvw.w * duvw.w;
            const bool wantBreak = dsq < 1e-3f * 1e-3f;
            const bool didBreak = (k + 1u == iters) && conv != 0u;
            if (wantBreak != didBreak && !(k + 1u == iters && conv == 0u && !wantBreak))
            { if (badBreak < 4)
                  std::printf("    tet %s iter %u: break %d want %d (iters %u conv %u)\n",
                              tets[i].name, k, int(didBreak), int(wantBreak), iters, conv);
              ++badBreak; }
            if (wantBreak) break;
            uvw = newUVW;
        }
    }
    PXG_CHECK_MSG(badStart == 0, "%zu tetrahedra chose a different starting iterate", badStart);
    PXG_CHECK_MSG(badFinalP == 0, "%zu tetrahedra returned a `point` that is not the barycentric "
                  "blend of their last newUvw -- E5 rebuilds p on EVERY iteration, clamped or "
                  "not, and without this check a port that rebuilds it only when the projection "
                  "ran is invisible", badFinalP);
    PXG_CHECK_MSG(badDiff == 0, "%zu tetrahedra computed a different `difference`", badDiff);
    PXG_CHECK_MSG(badStep == 0, "%zu step values differ from the 1/(2d) * 0.8^k schedule", badStep);
    PXG_CHECK_MSG(badClamp == 0, "%zu iterations disagreed about WHETHER the projection ran -- "
                  "E3's gate is what makes an unconditional clamp visible", badClamp);
    PXG_CHECK_MSG(badUvw == 0, "%zu iterations landed on a different newUvw than the engine's "
                  "own closestPtPointTetrahedron produces", badUvw);
    PXG_CHECK_MSG(badBreak == 0, "%zu iterations broke at the wrong point", badBreak);
    PXG_CHECK_MSG(badGradN == 0, "%zu used gradients were not unit length", badGradN);
    PXG_CHECK_MSG(badIters == 0, "%zu tetrahedra ran a number of iterations the loop bound "
                  "forbids", badIters);
    PXG_CHECK_MSG(badFinalG == 0, "%zu tetrahedra reused the loop's gradient instead of "
                  "resampling", badFinalG);

    for (uint32_t a = 0; a < 5u; ++a)
        if (a == 0u || a == 4u)
            PXG_CHECK_MSG(armSeen[a] > 0,
                          "cascade arm %u was never taken; an unreached arm is an unchecked arm",
                          a);
    PXG_CHECK_MSG(nConverged > 0, "no tetrahedron converged, so the tolerance break is unchecked");
    PXG_CHECK_MSG(nRanOut > 0, "no tetrahedron ran to the 16-iteration cap, so the bound is "
                  "unchecked");
    PXG_CHECK_MSG(nClampRan > 0 && nClampSkipped > 0,
                  "the projection ran on %u iterations and was skipped on %u -- both must be > 0, "
                  "or E3's condition is indistinguishable from an unconditional clamp",
                  nClampRan, nClampSkipped);
    // ---- part 3: the discontinuity substitution -------------------------------
    // gm == 0 is unreachable in the sparse scene, exactly as it was for the triangle
    // descent.  The DENSE scene has a point whose trilinear gradient is exactly zero,
    // so a tetrahedron collapsed onto it hits the branch on its first iteration.
    //
    // And this is where E7 earns its own check rather than borrowing the triangle's:
    // the two descents substitute DIFFERENT constants and treat them differently.
    // The triangle path uses sdfCollision.cuh:455's vector RAW, because that one is
    // exactly unit in float32; this one uses doVertexSDFCollision's vector through
    // getNormalized, and there the normalise MOVES the value -- P9.34a measured the
    // constant at 0.90790 squared.  So the expected bits here are the NORMALISED form
    // of the OTHER constant, and a port that copies either half from the triangle
    // sibling lands somewhere else.
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

            std::vector<PxVec4> dtv(4), ddir(1);
            dtv[0] = PxVec4(flat.x, flat.y, flat.z, 1.0f);
            dtv[1] = PxVec4(flat.x, flat.y, flat.z, 0.0f);
            dtv[2] = PxVec4(flat.x, flat.y, flat.z, 0.0f);
            dtv[3] = PxVec4(flat.x, flat.y, flat.z, 0.0f);
            ddir[0] = PxVec4(-3.0f, -4.0f, -5.0f, 0.0f);
            std::vector<float> outD(TET_STRIDE, -1.0f);
            PxgBufferHandle bDtv = mkBuf(asBytes(dtv.data(), dtv.size() * sizeof(PxVec4)));
            PxgBufferHandle bDdir = mkBuf(asBytes(ddir.data(), ddir.size() * sizeof(PxVec4)));
            PxgBufferHandle bDout = mkBuf(asBytes(outD.data(), outD.size() * sizeof(float)));

            P_Coll pd = pt;
            pd.meshPtr = dnAddr;
            pd.numPoints = 1u;
            pd.dirSeed = dev->getBufferDeviceAddress(bDdir);
            pd.triVerts = dev->getBufferDeviceAddress(bDtv);
            pd.triOut = dev->getBufferDeviceAddress(bDout);
            Kernel k(Kernel::Bindless{}, "npSdfCollision.spv", "sdfCollTetrahedron", {},
                     sizeof(P_Coll));
            if (k.isValid())
            {
                k.dispatch(1u, {}, pd);
                dev->flush();
                const auto D2 = downloadBuffer<float>(dev, bDout, TET_STRIDE);
                const float gm0 = D2[32 + 1];
                const PxVec3 used(D2[32 + 6], D2[32 + 7], D2[32 + 8]);
                // The engine's own getNormalized of the engine's own constant.
                const PxVec3 want =
                    PxVec3(0.5448762f, 0.672269f, 0.398825f).getNormalized();
                // The two vectors a wrong port would reach for, so the failure message
                // can say WHICH mistake was made rather than just "not equal".
                const PxVec3 rawSame(0.5448762f, 0.672269f, 0.398825f);
                const PxVec3 triConst(0.571846586f, 0.705545099f, 0.418566116f);
                PXG_CHECK_MSG(gm0 == 0.0f,
                              "the collapsed tetrahedron on the dense scene's flat point must give "
                              "a ZERO gradient magnitude on iteration 0, got %.9g", gm0);
                if (gm0 == 0.0f)
                {
                    ++nSubstituted;
                    const bool ok = f2u(used.x) == f2u(want.x) && f2u(used.y) == f2u(want.y) &&
                                    f2u(used.z) == f2u(want.z);
                    const bool isRaw = f2u(used.x) == f2u(rawSame.x);
                    const bool isTri = f2u(used.x) == f2u(triConst.x);
                    PXG_CHECK_MSG(ok,
                                  "the discontinuity substitute is (%.9g %.9g %.9g); it must be "
                                  "doVertexSDFCollision's constant NORMALISED (%.9g %.9g %.9g)%s",
                                  used.x, used.y, used.z, want.x, want.y, want.z,
                                  isRaw ? " -- this is that constant used RAW, which is the "
                                          "TRIANGLE path's treatment, not this one's"
                                        : (isTri ? " -- this is the TRIANGLE path's constant"
                                                 : ""));
                    // And the normalise is load-bearing here, which is asserted rather
                    // than recalled: the raw and normalised forms must differ.
                    PXG_CHECK_MSG(f2u(want.x) != f2u(rawSame.x),
                                  "this constant is already unit, so the getNormalized is a no-op "
                                  "and E7's normalise half is unobservable");
                }
            }
        }
    }

    PXG_CHECK_MSG(nSubstituted > 0,
                  "the discontinuity substitution was never reached, so E7 -- the VERTEX path's "
                  "constant, normalised -- is unchecked");
    PXG_CHECK_MSG(nFinalGChecked > 0,
                  "no tetrahedron had finite arithmetic at the final gradient, so the resample "
                  "check never applied to anything");
    PXG_CHECK_MSG(nFinalPChecked > 0,
                  "no tetrahedron had a finite final point, so the E5 check never applied");
    PXG_CHECK_MSG(nStraddle > 0,
                  "no tetrahedron produced d0 == d1 with both strictly below d2, d3 and the "
                  "centroid, so E1's asymmetric cascade -- whose second arm never compares "
                  "against d0 -- is indistinguishable from a four-way minimum");
    PXG_CHECK_MSG(nEpsBand > 0,
                  "no iteration left the real gate quiet while sitting inside the band a NEGATED "
                  "eps would catch, so the SIGN of eps is unobservable -- the constructed "
                  "perpendicular tets are meant to produce exactly that");
    PXG_CHECK_MSG(nNanMag > 0,
                  "no iteration produced a NaN gradient magnitude, so the two-if / if-else "
                  "difference in PxVec3::normalize -- where neither branch fires and the raw "
                  "vector must survive -- is unchecked");

    std::printf("    [ note ] tet descent: %u tetrahedra, %u converged / %u ran to 16, start "
                "arms %u/%u/%u/%u/%u, clamp ran %u / skipped %u, %u discontinuity "
                "substitutions, %u NaN magnitudes, %u eps-band iterations, %u tied pairs, "
                "%u cascade straddles\n",
                NT, nConverged, nRanOut, armSeen[0], armSeen[1], armSeen[2], armSeen[3],
                armSeen[4], nClampRan, nClampSkipped, nSubstituted, nNanMag, nEpsBand, nTied,
                nStraddle);

    for (auto h : owned) dev->destroyBuffer(h);
}

} // namespace
