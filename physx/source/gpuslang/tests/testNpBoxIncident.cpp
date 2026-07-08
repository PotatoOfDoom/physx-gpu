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

// Verification of npBoxIncident.slang: getIncidentPolygon4 (width-4 broadcast
// __shfl_sync incident-face selection + per-lane corner transform) and
// calculateBounds4 (width-4 __shfl_xor_sync butterfly AABB reduction).  Both are
// warp-cooperative over groups of 4 lanes; the references transcribe cudaBox.cu
// and are evaluated per group.  For incident, random rigid transforms exercise
// all three face-axis branches; for bounds, the min/max selection is bit-exact.

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
struct IncidentIn { F4 col0, col1, col2, col3, axis, ext; };

static inline V3 mulAdd(const F4& c0, const F4& c1, const F4& c2, const F4& c3, const V3& v)
{
    return V3{ c0.x * v.x + c1.x * v.y + c2.x * v.z + c3.x,
               c0.y * v.x + c1.y * v.y + c2.y * v.z + c3.y,
               c0.z * v.x + c1.z * v.y + c2.z * v.z + c3.z };
}
static inline float dot3(const F4& a, const F4& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// mirrors cudaBox.cu getIncidentPolygon4(): returns the 4 face-corner verts
// (one per lane) and the group-uniform incident-face normal.  Returns the
// selected face axis (0/1/2) so the test can confirm branch coverage.
uint32_t refIncident(const IncidentIn& in, V3 outVerts[4], V3& faceNormal, bool& nearTie)
{
    const float d0 = dot3(in.col0, in.axis);
    const float d1 = dot3(in.col1, in.axis);
    const float d2 = dot3(in.col2, in.axis);
    const float a0 = std::fabs(d0), a1 = std::fabs(d1), a2 = std::fabs(d2);

    uint32_t axisSel;
    if (a0 >= a1 && a0 >= a2) axisSel = 0;
    else if (a1 >= a2)        axisSel = 1;
    else                      axisSel = 2;

    // The face-axis branch turns on the ordering of |d0|,|d1|,|d2|.  If the top
    // two magnitudes are within FMA-contraction noise, the GPU dot product could
    // pick a different face than this CPU reference — a whole-vertex flip that no
    // tolerance can absorb.  Flag such (measure-zero) groups so the caller can
    // skip them; a genuine branch bug still shows on the ~all non-tie groups.
    const float top1 = std::max(a0, std::max(a1, a2));
    const float mid = (a0 + a1 + a2) - top1 - std::min(a0, std::min(a1, a2));
    nearTie = (top1 - mid) <= 1e-4f * (top1 + 1e-6f);

    for (uint32_t lane = 0; lane < 4; ++lane)
    {
        const float s0 = (lane == 0 || lane == 3) ? -1.f : 1.f;
        const float s1 = (lane & 2u) ? 1.f : -1.f;
        V3 sgn;
        if (axisSel == 0)
        {
            sgn.x = (d0 > 0.f) ? -1.f : 1.f; sgn.y = s0; sgn.z = s1;
            faceNormal = V3{ in.col0.x * sgn.x, in.col0.y * sgn.x, in.col0.z * sgn.x };
        }
        else if (axisSel == 1)
        {
            sgn.y = (d1 > 0.f) ? -1.f : 1.f; sgn.x = s0; sgn.z = s1;
            faceNormal = V3{ in.col1.x * sgn.y, in.col1.y * sgn.y, in.col1.z * sgn.y };
        }
        else
        {
            sgn.z = (d2 > 0.f) ? -1.f : 1.f; sgn.x = s0; sgn.y = s1;
            faceNormal = V3{ in.col2.x * sgn.z, in.col2.y * sgn.z, in.col2.z * sgn.z };
        }
        const V3 vert{ in.ext.x * sgn.x, in.ext.y * sgn.y, in.ext.z * sgn.z };
        outVerts[lane] = mulAdd(in.col0, in.col1, in.col2, in.col3, vert);
    }
    return axisSel;
}

PXG_TEST(npBoxIncident_incidentPolygon4)
{
    std::mt19937 rng(48010);
    const uint32_t NG = 3000;               // groups of 4
    const uint32_t N = NG * 4;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 100000) / 100000.f); };

    std::vector<IncidentIn> in(NG);
    size_t branchCount[3] = { 0, 0, 0 };
    std::vector<V3> refV(N);
    std::vector<V3> refN(NG);
    std::vector<uint8_t> tie(NG, 0);
    size_t tieCount = 0;
    for (uint32_t g = 0; g < NG; ++g)
    {
        // random rigid transform (rotation columns from Euler XYZ + translation)
        const float ax = rf(-3.1f, 3.1f), ay = rf(-3.1f, 3.1f), az = rf(-3.1f, 3.1f);
        const float cx = std::cos(ax), sx = std::sin(ax);
        const float cy = std::cos(ay), sy = std::sin(ay);
        const float cz = std::cos(az), sz = std::sin(az);
        IncidentIn s{};
        s.col0 = { cy * cz, cy * sz, -sy, 0 };
        s.col1 = { sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy, 0 };
        s.col2 = { cx * sy * cz + sx * sz, cx * sy * sz - sx * cz, cx * cy, 0 };
        s.col3 = { rf(-5, 5), rf(-5, 5), rf(-5, 5), 0 };
        // random axis (need not be unit; sign selection uses |col.dot(axis)|)
        s.axis = { rf(-1, 1), rf(-1, 1), rf(-1, 1), 0 };
        s.ext = { rf(0.3f, 2.5f), rf(0.3f, 2.5f), rf(0.3f, 2.5f), 0 };
        in[g] = s;

        V3 v[4]; V3 fn; bool nt = false;
        branchCount[refIncident(s, v, fn, nt)]++;
        for (uint32_t l = 0; l < 4; ++l) refV[g * 4 + l] = v[l];
        refN[g] = fn;
        tie[g] = nt ? 1u : 0u;
        if (nt) ++tieCount;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<F4> vInit(N, F4{ 9, 9, 9, 9 }), nInit(NG, F4{ 9, 9, 9, 9 });
    std::vector<F4> dummyPts(16, F4{ 0, 0, 0, 0 }), dummyMin(16), dummyMax(16);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bV = createBufferFrom(dev, vInit);
    PxgBufferHandle bN = createBufferFrom(dev, nInit);
    PxgBufferHandle bPts = createBufferFrom(dev, dummyPts);
    PxgBufferHandle bMin = createBufferFrom(dev, dummyMin);
    PxgBufferHandle bMax = createBufferFrom(dev, dummyMax);

    struct BoxIncParams { uint32_t numGroups; } p = { NG };
    Kernel k("npBoxIncident.spv", "boxIncidentPolygon4", { 96, 16, 16, 16, 16, 16 }, sizeof(BoxIncParams));
    k.dispatch(GROUPS, { bIn, bV, bN, bPts, bMin, bMax }, p);
    dev->flush();

    const auto gV = downloadBuffer<F4>(dev, bV, N);
    const auto gN = downloadBuffer<F4>(dev, bN, NG);
    size_t mmV = 0, mmN = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (tie[i >> 2]) continue; // measure-zero face-axis tie: branch may differ legitimately
        const V3& r = refV[i];
        if (!almostEqual(gV[i].x, r.x, 1e-4f, 1e-5f) || !almostEqual(gV[i].y, r.y, 1e-4f, 1e-5f) ||
            !almostEqual(gV[i].z, r.z, 1e-4f, 1e-5f)) ++mmV;
    }
    for (uint32_t g = 0; g < NG; ++g)
    {
        if (tie[g]) continue;
        const V3& r = refN[g];
        if (!almostEqual(gN[g].x, r.x, 1e-4f, 1e-5f) || !almostEqual(gN[g].y, r.y, 1e-4f, 1e-5f) ||
            !almostEqual(gN[g].z, r.z, 1e-4f, 1e-5f)) ++mmN;
    }
    PXG_CHECK_MSG(mmV == 0, "boxIncidentPolygon4: %zu vertex mismatches", mmV);
    PXG_CHECK_MSG(mmN == 0, "boxIncidentPolygon4: %zu faceNormal mismatches", mmN);
    // near-ties are measure-zero; if many appear the skip would gut the test
    PXG_CHECK_MSG(tieCount < NG / 100, "incident: too many face-axis near-ties (%zu)", tieCount);
    PXG_CHECK_MSG(branchCount[0] > NG / 8 && branchCount[1] > NG / 8 && branchCount[2] > NG / 8,
                  "incident: face-axis branches under-exercised (%zu/%zu/%zu)",
                  branchCount[0], branchCount[1], branchCount[2]);

    for (PxgBufferHandle h : { bIn, bV, bN, bPts, bMin, bMax }) dev->destroyBuffer(h);
}

PXG_TEST(npBoxIncident_calculateBounds4)
{
    std::mt19937 rng(48020);
    const uint32_t NG = 4000;
    const uint32_t N = NG * 4;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 100000) / 100000.f); };

    std::vector<F4> pts(N);
    for (uint32_t i = 0; i < N; ++i)
        pts[i] = F4{ rf(-10, 10), rf(-10, 10), rf(-10, 10), 0 };

    // reference: per group of 4, the AABB over the 4 points (broadcast to all lanes)
    std::vector<F4> refMin(N), refMax(N);
    for (uint32_t g = 0; g < NG; ++g)
    {
        F4 mn = pts[g * 4], mx = pts[g * 4];
        for (uint32_t l = 1; l < 4; ++l)
        {
            const F4& q = pts[g * 4 + l];
            mn.x = std::min(mn.x, q.x); mn.y = std::min(mn.y, q.y); mn.z = std::min(mn.z, q.z);
            mx.x = std::max(mx.x, q.x); mx.y = std::max(mx.y, q.y); mx.z = std::max(mx.z, q.z);
        }
        for (uint32_t l = 0; l < 4; ++l) { refMin[g * 4 + l] = mn; refMax[g * 4 + l] = mx; }
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<IncidentIn> dummyIn(16);
    std::vector<F4> dummyV(16), dummyN(16), minInit(N, F4{ 9, 9, 9, 9 }), maxInit(N, F4{ 9, 9, 9, 9 });
    PxgBufferHandle bIn = createBufferFrom(dev, dummyIn);
    PxgBufferHandle bV = createBufferFrom(dev, dummyV);
    PxgBufferHandle bN = createBufferFrom(dev, dummyN);
    PxgBufferHandle bPts = createBufferFrom(dev, pts);
    PxgBufferHandle bMin = createBufferFrom(dev, minInit);
    PxgBufferHandle bMax = createBufferFrom(dev, maxInit);

    struct BoxIncParams { uint32_t numGroups; } p = { NG };
    Kernel k("npBoxIncident.spv", "boxCalculateBounds4", { 96, 16, 16, 16, 16, 16 }, sizeof(BoxIncParams));
    k.dispatch(GROUPS, { bIn, bV, bN, bPts, bMin, bMax }, p);
    dev->flush();

    const auto gMin = downloadBuffer<F4>(dev, bMin, N);
    const auto gMax = downloadBuffer<F4>(dev, bMax, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        // min/max selection is exact — bit-exact comparison
        if (gMin[i].x != refMin[i].x || gMin[i].y != refMin[i].y || gMin[i].z != refMin[i].z ||
            gMax[i].x != refMax[i].x || gMax[i].y != refMax[i].y || gMax[i].z != refMax[i].z) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "boxCalculateBounds4: %zu mismatches", mm);

    for (PxgBufferHandle h : { bIn, bV, bN, bPts, bMin, bMax }) dev->destroyBuffer(h);
}

} // namespace
