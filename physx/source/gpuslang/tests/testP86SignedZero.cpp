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

// P9.86 -- the sign of zero on the device, with and without
// OpExecutionMode SignedZeroInfNanPreserve (see shaders/p86SignedZero.slang).

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace
{
using namespace pxgtest;

struct SzPushH { uint64_t inp, outp; };

const uint32_t P0 = 0x00000000u, N0 = 0x80000000u;
const uint32_t NOUT = 23;
// IEEE-754 results, which is what the CUDA computes.
const uint32_t kIeee[NOUT] = { P0, P0, P0, N0, P0, N0,  P0, P0, P0, N0, N0,  0, 0, N0, N0,  P0, P0, N0, P0,  P0, P0, P0, P0 };
const char* const kName[NOUT] = { "0 - (+0)", "(-0) + 0", "0 + (-0)", "0 * (-0)", "(0,0,0) - v", "(-0) - 0",
                                  "rz - (+0)", "(-0) + rz", "rz + (-0)", "rz * (-0)", "-1 * (+0)",
                                  "min(+0,-0)", "max(-0,+0)", "PxMin(+0,-0)", "PxMax(+0,-0)",
                                  "(-0) + pz", "pz - (+0)", "(0,0,0) * -1", "e=0; e -= +0",
                                  "(a,0,0)+(0,-0,..)", "unrolled 0 seed", "-0 + 0*pz", "select +0" };

// in[0] = pz, in[1] = nz: the probe's +0 / -0.  `swapZeros` loads them the other way
// round (pz = -0, nz = +0), which turns row 11 into min(-0,+0) and row 12 into
// max(+0,-0) without touching the shader; the other rows are then not meaningful.
std::vector<uint32_t> run(const char* spv, bool swapZeros = false)
{
    PxgSlangComputeDevice* dev = testDevice();
    const std::vector<float> in = { swapZeros ? -0.0f : 0.0f, swapZeros ? 0.0f : -0.0f, -1.0f, 0.0f };
    const std::vector<uint32_t> outInit(NOUT, 0xDEADBEEFu);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    SzPushH push{ dev->getBufferDeviceAddress(bIn), dev->getBufferDeviceAddress(bOut) };
    Kernel k(spv, "szProbe", {}, sizeof(SzPushH));
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();
    std::vector<uint32_t> got = downloadBuffer<uint32_t>(dev, bOut, NOUT);
    dev->destroyBuffer(bIn); dev->destroyBuffer(bOut);
    return got;
}
} // namespace

PXG_TEST(p86SignedZero)
{
    const std::vector<uint32_t> plain = run("p86SignedZero.spv");
    const std::vector<uint32_t> sz = run("p86SignedZeroSz.spv");
    std::printf("  %-14s %-10s %-10s %-10s\n", "expr", "IEEE", "plain", "SZINP");
    for (uint32_t k = 0; k < NOUT; ++k)
        std::printf("  %-14s %08x   %08x   %08x%s\n", kName[k], kIeee[k], plain[k], sz[k],
                    (k < 11 || k >= 13) && (plain[k] != kIeee[k] || sz[k] != kIeee[k]) ? "   <-- differs" : "");
    // Runtime-operand arithmetic and the PxMin/PxMax spellings are IEEE in both builds.
    for (uint32_t k = 6; k < 11; ++k)
        PXG_CHECK_MSG(plain[k] == kIeee[k] && sz[k] == kIeee[k], "%s: runtime-operand arithmetic lost the sign of zero", kName[k]);
    for (uint32_t k = 13; k < 15; ++k)
        PXG_CHECK_MSG(plain[k] == kIeee[k] && sz[k] == kIeee[k], "%s: select spelling lost the sign of zero", kName[k]);
    // What the engine relies on since P9.86 (every shader is stamped SignedZeroInfNanPreserve):
    // the mode stops the driver folding x * 0 and x +/- a PROVEN zero.  It does NOT stop
    // OpFSub(constant 0, x) -> -x (rows 0, 4, 18) -- the port spells those as selects -- and
    // rows 1-2 are slangc's own folds, never reaching the driver.
    for (uint32_t k : { 3u, 15u, 16u, 17u, 21u })
        PXG_CHECK_MSG(sz[k] == kIeee[k], "%s: SignedZeroInfNanPreserve no longer honoured", kName[k]);
    // The select spelling the port uses for the folds the mode does not stop (rows 0, 4,
    // 18, 19, 20): IEEE in both builds, whatever the driver does.
    PXG_CHECK_MSG(plain[22] == kIeee[22] && sz[22] == kIeee[22], "select spelling lost the sign of zero");

    // P9.86c F4: in a stamped shader llvmpipe orders the zeros of FMin/FMax -0 < +0 in
    // EITHER operand order -- NIR's fmin/fmax are commutative with fmin(+-0,-+0) = -0,
    // and llvmpipe (lower_fminmax_signed_zero, lp_screen.c) has nir_lower_alu rewrite
    // every signed-zero-preserving fmin/fmax to feq(a,b) ? imin/imax(bits) : fmin/fmax,
    // before gallivm picks an operand order.  That is NVIDIA's fminf/fmaxf zero rule,
    // and the EQUIVALENT verdict of npCvxCollide-r0 (epaCore's triangle-distance scan as
    // a plain min() chain, npCvxCollide.slang ~:744 vs epa.cuh:430-436) rests on it.
    // Pin both orders in the stamped build; the unstamped build (no lowering, x86 minps
    // /maxps keep the second operand on a tie) is only printed.
    const std::vector<uint32_t> plainSw = run("p86SignedZero.spv", true);
    const std::vector<uint32_t> szSw = run("p86SignedZeroSz.spv", true);
    std::printf("  %-18s %-10s %-10s %-10s\n", "F4 zero tie", "NVIDIA", "plain", "SZINP");
    std::printf("  %-18s %08x   %08x   %08x\n", "min(+0,-0)", N0, plain[11], sz[11]);
    std::printf("  %-18s %08x   %08x   %08x\n", "min(-0,+0)", N0, plainSw[11], szSw[11]);
    std::printf("  %-18s %08x   %08x   %08x\n", "max(-0,+0)", P0, plain[12], sz[12]);
    std::printf("  %-18s %08x   %08x   %08x\n", "max(+0,-0)", P0, plainSw[12], szSw[12]);
    PXG_CHECK_MSG(sz[11] == N0 && szSw[11] == N0,
                  "FMin zero tie no longer -0 in both orders (%08x / %08x) -- npCvxCollide-r0 needs a kill", sz[11], szSw[11]);
    PXG_CHECK_MSG(sz[12] == P0 && szSw[12] == P0,
                  "FMax zero tie no longer +0 in both orders (%08x / %08x)", sz[12], szSw[12]);
}

// ---------------------------------------------------------------------------------------------
// P9.86f: pxgSlangTriBox's AXISTEST_* / FINDMINMAX are intrinsics::selectMin/selectMax
// (GuIntersectionTriangleBoxRef.h:58-61, :108-156; PxUnixMathIntrinsics.h:116-125), which return the
// SECOND operand when the compare is false -- so a NaN second operand survives and the axis cannot
// reject.  GLSL min/max (llvmpipe FMin/FMax = minNum/maxNum) drop the NaN and return the first operand.
// The CPU mirror below is the CUDA's intrinsicsCheck; gTbVariant replays one reverted port line:
// 2s+1 = site s's `minimum` as FMin, 2s+2 = its `maximum` as FMax (sites 0..8 the nine AXISTESTs in
// the CUDA's order, 9..11 the X/Y/Z FINDMINMAX).
namespace
{
struct TbV { float x, y, z; };
int gTbVariant = 0;
int gTbSite = 0;
float tbFMinLp(float a, float b) { if (std::isnan(a)) return b; if (std::isnan(b)) return a; return (a == b) ? (std::signbit(a) ? a : b) : (a < b ? a : b); }
float tbFMaxLp(float a, float b) { if (std::isnan(a)) return b; if (std::isnan(b)) return a; return (a == b) ? (std::signbit(a) ? b : a) : (a > b ? a : b); }
float tbMin(float a, float b) { return (gTbVariant == 2 * gTbSite + 1) ? tbFMinLp(a, b) : ((a < b) ? a : b); }
float tbMax(float a, float b) { return (gTbVariant == 2 * gTbSite + 2) ? tbFMaxLp(a, b) : ((a > b) ? a : b); }
TbV tbSub(TbV a, TbV b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
float tbDot(TbV a, TbV b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
bool tbPlaneBox(TbV n, float d, TbV m)
{
    const TbV vmin = { n.x > 0.0f ? -m.x : m.x, n.y > 0.0f ? -m.y : m.y, n.z > 0.0f ? -m.z : m.z };
    const TbV vmax = { n.x > 0.0f ? m.x : -m.x, n.y > 0.0f ? m.y : -m.y, n.z > 0.0f ? m.z : -m.z };
    if (tbDot(n, vmin) + d > 0.0f) return false;
    if (tbDot(n, vmax) + d >= 0.0f) return true;
    return false;
}
bool tbRef(TbV c, TbV ex, TbV t0, TbV t1, TbV t2)
{
    const TbV v0 = tbSub(t0, c), v1 = tbSub(t1, c), v2 = tbSub(t2, c);
    auto in = [&](TbV v) { return std::fabs(v.x) <= ex.x && std::fabs(v.y) <= ex.y && std::fabs(v.z) <= ex.z; };
    if (in(v0) || in(v1) || in(v2)) return true;
    const TbV e0 = tbSub(v1, v0), e1 = tbSub(v2, v1), e2 = tbSub(v0, v2);
    float mn, mx, rad, pa, pb;
    auto axis = [&](int site, float a, float b, float r) {
        gTbSite = site; mn = tbMin(a, b); mx = tbMax(a, b); rad = r;
        return mn > rad || mx < -rad;
    };
    float fex = std::fabs(e0.x), fey = std::fabs(e0.y), fez = std::fabs(e0.z);
    pa = e0.z * v0.y - e0.y * v0.z; pb = e0.z * v2.y - e0.y * v2.z;    if (axis(0, pa, pb, fez * ex.y + fey * ex.z)) return false;   // X01
    pa = -e0.z * v0.x + e0.x * v0.z; pb = -e0.z * v2.x + e0.x * v2.z;  if (axis(1, pa, pb, fez * ex.x + fex * ex.z)) return false;   // Y02
    pa = e0.y * v1.x - e0.x * v1.y; pb = e0.y * v2.x - e0.x * v2.y;    if (axis(2, pa, pb, fey * ex.x + fex * ex.y)) return false;   // Z12
    fex = std::fabs(e1.x); fey = std::fabs(e1.y); fez = std::fabs(e1.z);
    pa = e1.z * v0.y - e1.y * v0.z; pb = e1.z * v2.y - e1.y * v2.z;    if (axis(3, pa, pb, fez * ex.y + fey * ex.z)) return false;   // X01
    pa = -e1.z * v0.x + e1.x * v0.z; pb = -e1.z * v2.x + e1.x * v2.z;  if (axis(4, pa, pb, fez * ex.x + fex * ex.z)) return false;   // Y02
    pa = e1.y * v0.x - e1.x * v0.y; pb = e1.y * v1.x - e1.x * v1.y;    if (axis(5, pa, pb, fey * ex.x + fex * ex.y)) return false;   // Z0
    fex = std::fabs(e2.x); fey = std::fabs(e2.y); fez = std::fabs(e2.z);
    pa = e2.z * v0.y - e2.y * v0.z; pb = e2.z * v1.y - e2.y * v1.z;    if (axis(6, pa, pb, fez * ex.y + fey * ex.z)) return false;   // X2
    pa = -e2.z * v0.x + e2.x * v0.z; pb = -e2.z * v1.x + e2.x * v1.z;  if (axis(7, pa, pb, fez * ex.x + fex * ex.z)) return false;   // Y1
    pa = e2.y * v1.x - e2.x * v1.y; pb = e2.y * v2.x - e2.x * v2.y;    if (axis(8, pa, pb, fey * ex.x + fex * ex.y)) return false;   // Z12
    // FINDMINMAX: selectMin(selectMin(x0, x1), x2), same for max
    const float vx[3] = { v0.x, v1.x, v2.x }, vy[3] = { v0.y, v1.y, v2.y }, vz[3] = { v0.z, v1.z, v2.z };
    const float* comp[3] = { vx, vy, vz };
    const float exc[3] = { ex.x, ex.y, ex.z };
    for (int k = 0; k < 3; ++k)
    {
        gTbSite = 9 + k;
        mn = tbMin(tbMin(comp[k][0], comp[k][1]), comp[k][2]);
        mx = tbMax(tbMax(comp[k][0], comp[k][1]), comp[k][2]);
        if (mn > exc[k] || mx < -exc[k]) return false;
    }
    const TbV n = { e0.y * e1.z - e0.z * e1.y, e0.z * e1.x - e0.x * e1.z, e0.x * e1.y - e0.y * e1.x };
    return tbPlaneBox(n, -tbDot(n, v0), ex);
}
} // namespace

// Directed inputs (found by a host search over the mirror): one triangle vertex coordinate pair per
// case overflows a*v.y / b*v.z to +-inf of the same sign, so the AXISTEST's second p is NaN while the
// first is +-inf or finite beyond +-rad.  The CUDA does not reject there and every case ends in the
// plane test with an overlap (true); each reverted AXISTEST line rejects (false) on at least one case.
// Box: centre 0, half extents 0.5.  The last three cases are controls (plain overlap, plain miss,
// an infinite vertex whose NaN plane is false either way).
PXG_TEST(p86SignedZero_triBoxNaN)
{
    struct Tri { TbV a, b, c; };
    const Tri tris[] = {
        { { -0x1.5af1dap+66, 0x1p-1, 0x1p-1 }, { 0x1.8p+1, 0x1p+0, 0x1p+0 }, { 0x1.158e48p+63, -0x1.5af1d8p+66, -0x1p+1 } },
        { { 0x1p-1, 0x1p-2, 0x1.158e46p+63 }, { 0x1p-1, 0x1p-1, -0x1p+1 }, { -0x1.a05568p+64, -0x1.5af1d6p+66, -0x1.a0556ap+64 } },
        { { 0x1p-1, 0x1p-2, -0x1p+1 }, { 0x1.158e48p+63, 0x0p+0, 0x1.158e46p+63 }, { -0x1.a0556cp+64, -0x1p+1, 0x1.a05568p+64 } },
        { { -0x1p+0, 0x1p-1, -0x1p-1 }, { 0x0p+0, -0x1.158e48p+63, -0x1.a05568p+64 }, { 0x1.5af1dap+66, 0x1.5af1d8p+66, 0x1.8p+1 } },
        { { 0x0p+0, -0x1.a0556cp+64, 0x1p-2 }, { -0x1.a05568p+64, -0x1.158e48p+63, -0x1p+1 }, { 0x1.158e48p+63, 0x1.5af1d8p+66, 0x1p+1 } },
        { { 0x0p+0, -0x1p+1, 0x1p-2 }, { -0x1.8p+1, 0x1.158e46p+63, -0x1.5af1d6p+66 }, { -0x1.a05568p+64, 0x1.158e46p+63, -0x1.5af1d6p+66 } },
        { { 0x1p-2, 0x0p+0, -0x1.a0556cp+64 }, { 0x1p-2, -0x1.a05568p+64, 0x1p+0 }, { 0x1.8p+1, 0x1.5af1dap+66, -0x1.158e46p+63 } },
        { { 0x0p+0, 0x1p+0, -0x1p+0 }, { -0x1p-1, 0x1p-1, 0x1p+1 }, { -0x1.158e48p+63, -0x1.a0556ap+64, -0x1.a05568p+64 } },
        { { -0x1p-1, 0x1p-2, -0x1.a05568p+64 }, { 0x1.158e46p+63, 0x1p+1, -0x1.a05568p+64 }, { -0x1.a0556cp+64, 0x0p+0, 0x1.5af1d8p+66 } },
        { { 0x1p-1, -0x1.5af1d6p+66, -0x1p-1 }, { 0x1.5af1dap+66, -0x1.8p+1, -0x1p-1 }, { 0x1.5af1d8p+66, 0x1.5af1d6p+66, -0x1.8p+1 } },
        { { 0x1.a0556cp+64, -0x1p-1, -0x1.8p+1 }, { 0x1.158e48p+63, -0x1p+1, -0x1.158e46p+63 }, { -0x1.158e46p+63, -0x1p-1, 0x1.5af1d6p+66 } },
        { { 0x1.5af1d6p+66, 0x1.8p+1, 0x1p-2 }, { 0x1p+0, -0x1.5af1d6p+66, -0x1p+1 }, { -0x1.5af1dap+66, -0x1.158e46p+63, 0x1p+1 } },
        { { -0x1.8p+1, 0x1p-1, -0x1p-1 }, { 0x1.a05568p+64, 0x1p-2, -0x1.5af1d8p+66 }, { 0x1.a05568p+64, -0x1p+0, -0x1.a0556ap+64 } },
        { { -0x1p-1, -0x1.8p+1, -0x1p-1 }, { 0x1p-1, 0x1.a0556cp+64, -0x1.5af1d6p+66 }, { -0x1.158e48p+63, 0x1.5af1d8p+66, -0x1.5af1d6p+66 } },
        { { -0x1.8p+1, -0x1.5af1dap+66, -0x1p-1 }, { 0x1.5af1d8p+66, -0x1.158e46p+63, -0x1.8p+1 }, { -0x1.5af1d8p+66, 0x1.158e48p+63, 0x1p+1 } },
        { { 0x1p-1, 0x1p+1, -0x1.158e48p+63 }, { -0x1p+1, -0x1.a0556cp+64, 0x1.a0556cp+64 }, { -0x1p+1, -0x1.a0556ap+64, 0x1.5af1d6p+66 } },
        { { -0x1p-1, -0x1p-1, 0x1p+0 }, { 0x1.5af1d8p+66, 0x1p-1, -0x1.158e48p+63 }, { 0x1.5af1d8p+66, -0x1.158e48p+63, -0x1.a0556ap+64 } },
        { { -0x1p-1, -0x1p-1, 0x1.a05568p+64 }, { -0x1p+0, 0x1.a05568p+64, -0x1.5af1dap+66 }, { 0x1p-1, 0x1.5af1d6p+66, -0x1.5af1d8p+66 } },
        { { 0.25f, 0.0f, 0.0f }, { 2.0f, 0.0f, 0.0f }, { 0.0f, 2.0f, 0.0f } },
        { { 3.0f, 3.0f, 3.0f }, { 4.0f, 3.0f, 3.0f }, { 3.0f, 4.0f, 3.0f } },
        { { 1.0f, 0.0f, 0.0f }, { INFINITY, 1.0f, 0.0f }, { 0.0f, 2.0f, 0.0f } },
    };
    const uint32_t N = uint32_t(sizeof(tris) / sizeof(tris[0])), NDIR = N - 3;
    const TbV c = { 0.0f, 0.0f, 0.0f }, ex = { 0.5f, 0.5f, 0.5f };
    std::vector<float> in;
    std::vector<uint32_t> ref(N);
    uint32_t twinKills[25] = {};
    for (uint32_t k = 0; k < N; ++k)
    {
        for (const TbV& v : { c, ex, tris[k].a, tris[k].b, tris[k].c })
            in.insert(in.end(), { v.x, v.y, v.z, 0.0f });
        gTbVariant = 0;
        ref[k] = tbRef(c, ex, tris[k].a, tris[k].b, tris[k].c) ? 1u : 0u;
        PXG_CHECK_MSG(ref[k] == (k <= NDIR ? 1u : 0u), "triBox case %u: CUDA mirror %u", k, ref[k]);
        for (int v = 1; v <= 24; ++v)
        {
            gTbVariant = v;
            if ((tbRef(c, ex, tris[k].a, tris[k].b, tris[k].c) ? 1u : 0u) != ref[k]) ++twinKills[v];
        }
        gTbVariant = 0;
    }
    // Every AXISTEST twin (min and max of all nine sites) must change at least one directed result.
    // The FINDMINMAX twins (19..24) change none: a NaN there needs a NaN vertex coordinate, whose
    // edges make the plane normal's other two lanes and d NaN, so the CUDA's plane test is false too
    // -- the FMin/FMax revert can only add a reject to a case that already ends false (equivalent).
    for (int v = 1; v <= 18; ++v)
        PXG_CHECK_MSG(twinKills[v] > 0, "triBox: twin %d (site %d %s) changes no directed case", v, (v - 1) / 2, (v & 1) ? "min" : "max");
    for (int v = 19; v <= 24; ++v)
        PXG_CHECK_MSG(twinKills[v] == 0, "triBox: FINDMINMAX twin %d changed %u cases", v, twinKills[v]);

    PxgSlangComputeDevice* dev = testDevice();
    const std::vector<uint32_t> outInit(N, 0xDEADBEEFu);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    SzPushH push{ dev->getBufferDeviceAddress(bIn), dev->getBufferDeviceAddress(bOut) };
    Kernel k("p86SignedZeroSz.spv", "tbProbe", {}, sizeof(SzPushH));
    k.dispatch(N, {}, push, 1u, 1u);
    dev->flush();
    const std::vector<uint32_t> got = downloadBuffer<uint32_t>(dev, bOut, N);
    dev->destroyBuffer(bIn); dev->destroyBuffer(bOut);
    for (uint32_t i = 0; i < N; ++i)
        PXG_CHECK_MSG(got[i] == ref[i], "tbIntersectTriangleBox case %u: GPU %u, CUDA %u", i, got[i], ref[i]);
}
