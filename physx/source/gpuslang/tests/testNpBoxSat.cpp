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

// Verification of npBoxSat.slang: boxSat — the separating-axis test + feature
// selection half of doBoxBoxGenerateContacts.  Each group of 4 lanes carries a
// box pair (two extents + two rigid transforms + contactDist); the kernel emits
// (overlap, feature, sign, minOverlap).  The C++ reference transcribes the 15
// axes and the min-overlap feature pick.  overlap and feature are integers that
// turn on float comparisons, so groups where any separation predicate or the
// feature-decision gap sits within a small margin are skipped (a GPU-vs-CPU
// rounding flip there is measure-zero for the fixed seed); skips are asserted to
// stay a minority and coverage (overlap/separated + all 6 features) is checked.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct V3 { float x, y, z; };
struct SatIn { F4 b0e, b1e, c00, c01, c02, p0, c10, c11, c12, p1, misc; };

static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline float comp3(const V3& v, uint32_t i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }
static inline V3 absp(const V3& v, float e) { return V3{ std::fabs(v.x) + e, std::fabs(v.y) + e, std::fabs(v.z) + e }; }

const float MAXF = 3.4028235e38f;

// Reference SAT.  Returns marginal flag; fills overlap/feature/sign/minOverlap.
bool refSat(const V3& box0e, const V3& box1e, const V3 c0[3], const V3& p0,
            const V3 c1[3], const V3& p1, float cd,
            bool& overlap, uint32_t& feature, float& sign, float& minOverlap)
{
    const float MARG = 1e-3f, uEps = 1e-6f;
    // transform1To0
    V3 dP{ p1.x - p0.x, p1.y - p0.y, p1.z - p0.z };
    V3 r[3];
    for (int k = 0; k < 3; ++k)
        r[k] = V3{ dot3(c0[0], c1[k]), dot3(c0[1], c1[k]), dot3(c0[2], c1[k]) };
    V3 tp{ dot3(c0[0], dP), dot3(c0[1], dP), dot3(c0[2], dP) };
    // abs0To1 columns
    V3 ab[3] = {
        V3{ std::fabs(r[0].x) + uEps, std::fabs(r[1].x) + uEps, std::fabs(r[2].x) + uEps },
        V3{ std::fabs(r[0].y) + uEps, std::fabs(r[1].y) + uEps, std::fabs(r[2].y) + uEps },
        V3{ std::fabs(r[0].z) + uEps, std::fabs(r[1].z) + uEps, std::fabs(r[2].z) + uEps }
    };

    bool marginal = false;
    float laneMin[3]; uint32_t laneFeat[3]; float laneSign[3]; bool laneSep[3];

    for (uint32_t tid = 0; tid < 3; ++tid)
    {
        const V3 col = r[tid];
        const V3 abCol = absp(col, uEps);
        const V3 abT = ab[tid];
        const float tpc = comp3(tp, tid);
        const float ea = comp3(box0e, tid), eb = comp3(box1e, tid);

        const float rbA = dot3(abT, box1e);
        const float oA = (ea + rbA) - std::fabs(tpc) + cd;
        const float sA = tpc;
        const float sB = dot3(tp, col);
        const float raB = dot3(abCol, box0e);
        const float oB = (raB + eb) - std::fabs(sB) + cd;

        if (std::fabs(oA) < MARG || std::fabs(oB) < MARG) marginal = true; // face separation boundaries
        if (std::fabs(oA - oB) < MARG) marginal = true;                    // face-A vs face-B feature gate

        float mo = oA; uint32_t ft = tid; float sg = sA;
        if (mo > oB) { mo = oB; sg = sB; ft = tid + 3; }
        bool sepFace = (0.f > oA) || (0.f > oB);

        const uint32_t iHi = (tid == 2) ? 1 : 2, iLo = (tid == 0) ? 1 : 0;
        const float b1Lo = comp3(box1e, iLo), b1Hi = comp3(box1e, iHi);

        const float as1 = std::fabs(col.y * tp.z - col.z * tp.y);
        const float rs1 = abCol.z * box0e.y + abCol.y * box0e.z + comp3(ab[0], iHi) * b1Lo + comp3(ab[0], iLo) * b1Hi + cd;
        const float as2 = std::fabs(col.z * tp.x - col.x * tp.z);
        const float rs2 = abCol.z * box0e.x + abCol.x * box0e.z + comp3(ab[1], iHi) * b1Lo + comp3(ab[1], iLo) * b1Hi + cd;
        const float as3 = std::fabs(col.x * tp.y - col.y * tp.x);
        const float rs3 = abCol.y * box0e.x + abCol.x * box0e.y + comp3(ab[2], iHi) * b1Lo + comp3(ab[2], iLo) * b1Hi + cd;
        if (std::fabs(as1 - rs1) < MARG || std::fabs(as2 - rs2) < MARG || std::fabs(as3 - rs3) < MARG) marginal = true;
        bool sepEdge = (as1 > rs1) || (as2 > rs2) || (as3 > rs3);

        laneMin[tid] = mo; laneFeat[tid] = ft; laneSign[tid] = sg; laneSep[tid] = sepFace || sepEdge;
    }

    const bool separated = laneSep[0] || laneSep[1] || laneSep[2];
    // min pick + near-tie margins between the three lane minima
    if (std::fabs(laneMin[0] - laneMin[1]) < MARG || std::fabs(laneMin[0] - laneMin[2]) < MARG ||
        std::fabs(laneMin[1] - laneMin[2]) < MARG) marginal = true;
    uint32_t pick;
    if (laneMin[0] <= laneMin[1] && laneMin[0] <= laneMin[2]) pick = 0;
    else if (laneMin[1] <= laneMin[2]) pick = 1;
    else pick = 2;

    overlap = !separated;
    feature = laneFeat[pick];
    sign = laneSign[pick];
    minOverlap = laneMin[pick];
    return marginal;
}

// build a rotation matrix (columns) from Euler XYZ
void euler(float ax, float ay, float az, V3 col[3])
{
    const float cx = std::cos(ax), sx = std::sin(ax);
    const float cy = std::cos(ay), sy = std::sin(ay);
    const float cz = std::cos(az), sz = std::sin(az);
    col[0] = V3{ cy * cz, cy * sz, -sy };
    col[1] = V3{ sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy };
    col[2] = V3{ cx * sy * cz + sx * sz, cx * sy * sz - sx * cz, cx * cy };
}

PXG_TEST(npBoxSat_satFeature)
{
    std::mt19937 rng(50010);
    const uint32_t NG = 8000;
    const uint32_t N = NG * 4;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<SatIn> in(NG);
    std::vector<uint8_t> skip(NG, 0);
    std::vector<uint8_t> rOverlap(NG);
    std::vector<uint32_t> rFeat(NG);
    std::vector<float> rSign(NG), rMin(NG);

    size_t nSkip = 0, nOverlap = 0, nSep = 0, featHist[6] = { 0, 0, 0, 0, 0, 0 };
    for (uint32_t g = 0; g < NG; ++g)
    {
        V3 b0e{ rf(0.4f, 1.6f), rf(0.4f, 1.6f), rf(0.4f, 1.6f) };
        V3 b1e{ rf(0.4f, 1.6f), rf(0.4f, 1.6f), rf(0.4f, 1.6f) };
        V3 c0[3], c1[3];
        euler(rf(-3.1f, 3.1f), rf(-3.1f, 3.1f), rf(-3.1f, 3.1f), c0);
        euler(rf(-3.1f, 3.1f), rf(-3.1f, 3.1f), rf(-3.1f, 3.1f), c1);
        V3 p0{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f) };
        // offset chosen so a healthy mix separates / overlaps
        V3 p1{ p0.x + rf(-3.0f, 3.0f), p0.y + rf(-3.0f, 3.0f), p0.z + rf(-3.0f, 3.0f) };
        const float cd = rf(0.01f, 0.15f);

        SatIn s{};
        s.b0e = F4{ b0e.x, b0e.y, b0e.z, 0 }; s.b1e = F4{ b1e.x, b1e.y, b1e.z, 0 };
        s.c00 = F4{ c0[0].x, c0[0].y, c0[0].z, 0 }; s.c01 = F4{ c0[1].x, c0[1].y, c0[1].z, 0 }; s.c02 = F4{ c0[2].x, c0[2].y, c0[2].z, 0 };
        s.p0 = F4{ p0.x, p0.y, p0.z, 0 };
        s.c10 = F4{ c1[0].x, c1[0].y, c1[0].z, 0 }; s.c11 = F4{ c1[1].x, c1[1].y, c1[1].z, 0 }; s.c12 = F4{ c1[2].x, c1[2].y, c1[2].z, 0 };
        s.p1 = F4{ p1.x, p1.y, p1.z, 0 };
        s.misc = F4{ cd, 0, 0, 0 };
        in[g] = s;

        bool ov; uint32_t ft; float sg, mo;
        const bool marg = refSat(b0e, b1e, c0, p0, c1, p1, cd, ov, ft, sg, mo);
        skip[g] = marg ? 1u : 0u;
        rOverlap[g] = ov ? 1u : 0u; rFeat[g] = ft; rSign[g] = sg; rMin[g] = mo;
        if (marg) { ++nSkip; continue; }
        if (ov) { ++nOverlap; if (ft < 6) ++featHist[ft]; } else ++nSep;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<F4> outInit(NG, F4{ 9, 9, 9, 9 });
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct BoxSatParams { uint32_t numGroups; } p = { NG };
    Kernel k("npBoxSat.spv", "boxSat", { (uint32_t)sizeof(SatIn), 16 }, sizeof(BoxSatParams));
    k.dispatch(GROUPS, { bIn, bOut }, p);
    dev->flush();

    const auto g = downloadBuffer<F4>(dev, bOut, NG);
    size_t mmOverlap = 0, mmFeat = 0, mmSign = 0;
    for (uint32_t gi = 0; gi < NG; ++gi)
    {
        if (skip[gi]) continue;
        const uint32_t gov = (g[gi].x != 0.f) ? 1u : 0u;
        if (gov != rOverlap[gi]) { ++mmOverlap; continue; }
        if (gov == 0u) continue; // separated: feature/sign unused
        if ((uint32_t)(g[gi].y + 0.5f) != rFeat[gi]) ++mmFeat;
        if (!almostEqual(g[gi].z, rSign[gi], 1e-4f, 1e-5f)) ++mmSign;
    }
    PXG_CHECK_MSG(mmOverlap == 0, "boxSat: %zu overlap-flag mismatches", mmOverlap);
    PXG_CHECK_MSG(mmFeat == 0, "boxSat: %zu feature mismatches", mmFeat);
    PXG_CHECK_MSG(mmSign == 0, "boxSat: %zu sign mismatches", mmSign);
    PXG_CHECK_MSG(nSkip < NG / 4, "boxSat: too many marginal groups skipped (%zu/%u)", nSkip, NG);
    PXG_CHECK_MSG(nOverlap > NG / 20 && nSep > NG / 20, "boxSat: overlap/separated mix thin (%zu/%zu)", nOverlap, nSep);
    size_t featMin = featHist[0];
    for (int i = 1; i < 6; ++i) featMin = std::min(featMin, featHist[i]);
    PXG_CHECK_MSG(featMin > 0, "boxSat: not all 6 features exercised (%zu %zu %zu %zu %zu %zu)",
                  featHist[0], featHist[1], featHist[2], featHist[3], featHist[4], featHist[5]);

    dev->destroyBuffer(bIn); dev->destroyBuffer(bOut);
}

} // namespace
