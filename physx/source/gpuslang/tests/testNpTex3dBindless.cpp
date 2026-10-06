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

// testNpTex3dBindless.cpp -- P9.12a, the fixture for the CUtexObject replacement;
// P9.96, made bitwise against the emulated CUDA filter.
//
// npTex3d (P2) already established that a 3D volume can be sampled through a bound
// descriptor.  What it CANNOT establish is the property the SDF path actually
// depends on: that a texture handle can be STORED IN DEVICE MEMORY, loaded by a
// kernel, and used to sample -- which is what `tex3D<float>(mesh.mTexObject, ...)`
// does, with mTexObject sitting inside a struct the kernel reads.
//
// This fixture is that property, and it is deliberately hostile to the ways it
// could pass by accident:
//
//   * THREE volumes with THREE DIFFERENT EXTENTS and THREE DIFFERENT FORMATS
//     (R32F, R8_UNORM, R16_UNORM -- the three the engine's SDF upload asks for),
//     and the queried extent is read back and compared per sample;
//   * the handles are INTERLEAVED per sample, so adjacent lanes carry different
//     handles and the non-uniform path is what runs;
//   * no extent is a power of two and no two are equal;
//   * slot 0 is never allocated, so the reserved null handle stays unwritten --
//     the state PxgGeometryManager leaves mTexObject in for a mesh with no SDF.
//
// THE VALUES, since P9.96, are compared BIT FOR BIT against tests/pxgTex3dRef.h --
// every sample, centres and interior and clamp shell alike, for all three formats.
// Until P9.96 the module handed a Vulkan sampler a normalized coordinate and this
// fixture held interior points to a filtering tolerance (T3 of the old module
// header); the module now evaluates the Programming Guide's linear filter itself
// (pxgSlangTex3d.slang, T1-T3, K0-K5), so there is no implementation-defined
// precision left to tolerate.  The distance from an exact double trilinear is still
// printed, as a statistic about CUDA's 1/256 weights and nothing more.
//
// THE DIRECTED SETS, per volume (P9.96 plan C6.2), so that every non-equivalent
// spelling of the filter changes at least one expected value -- asserted on the host
// before anything is launched (the bite asserts below):
//   (a) every texel centre;
//   (b) per axis and k = 0..255, x = i + 0.5 + (k + 0.5)/256, an exact weight TIE,
//       plus its two neighbouring floats -- ties-down dies on every k, round-to-even
//       on the even ones, truncation and the bit-width mutants everywhere;
//   (c) a weight that rounds UP TO 1.0 (255.75/256) at i = N-2 and i = N-1;
//   (d) the clamp shell {-0.7, 0.2, 0.49999997, N-0.5, nextafter(N-0.5, +inf),
//       N+0.3, N+5} on each axis, and the eight corners;
//   (e) 700 random points over [-1, N+1];
//   (f) +-inf and +-1e30 coordinates, bitwise like everything else: GLSL.std.450
//       FMin/FMax are defined for them, so they clamp to the edge and return the edge
//       sample (K5).  Only a NaN coordinate is checked just for having been WRITTEN:
//       the module promises in-range addresses for it, not a value;
//   (g) SIGNED ZEROS (R32F volume only, P9.96 check round), the one special value
//       K5 lets reach a result through finite texels: a -0 texel on the far x face
//       with negative zero-weight taps, sampled at x = N-0.5 and past it, must return
//       -0 -- and is the only data on which the missing upper index clamp
//       (idx-clamp-hi-missing, whose out-of-bounds tap llvmpipe reads as +0) is
//       visible; a -0 far-face texel with one positive tap returns +0; a -0 low-face
//       texel with negative taps returns -0; a +0 texel with negative taps returns +0.

#include "pxgTestFramework.h"
#include "pxgTex3dRef.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

// Three distinct, non-power-of-two, mutually different extents.
struct VolDesc
{
    uint32_t w, h, d;
    PxgSlangComputeDevice::TextureFormat fmt;
    const char* name;
};

static const VolDesc kVols[3] = {
    { 12u, 9u, 7u, PxgSlangComputeDevice::TextureFormat::R32Float, "R32F 12x9x7" },
    { 6u, 11u, 5u, PxgSlangComputeDevice::TextureFormat::R8Unorm, "R8 6x11x5" },
    { 10u, 3u, 13u, PxgSlangComputeDevice::TextureFormat::R16Unorm, "R16 10x3x13" },
};

static uint32_t f2u(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// An exact (double) trilinear with the coordinate clamp, over the values the texture
// holds.  A STATISTIC only since P9.96: how far CUDA's 1/256 weights sit from exact.
static double trilinear(const pxgtex::Vol& V, double x, double y, double z)
{
    auto cc = [](double c, uint32_t n) { return std::min(std::max(c, 0.5), double(n) - 0.5); };
    const double ux = cc(x, V.w) - 0.5, uy = cc(y, V.h) - 0.5, uz = cc(z, V.d) - 0.5;
    const double fx = std::floor(ux), fy = std::floor(uy), fz = std::floor(uz);
    const double ax = ux - fx, ay = uy - fy, az = uz - fz;
    double acc = 0.0;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
            {
                const double w = (dx ? ax : 1.0 - ax) * (dy ? ay : 1.0 - ay) * (dz ? az : 1.0 - az);
                acc += w * double(V.at(pxgtex::clampIdx(int(fx) + dx, V.w), pxgtex::clampIdx(int(fy) + dy, V.h),
                                       pxgtex::clampIdx(int(fz) + dz, V.d)));
            }
    return acc;
}

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };

struct TxbParams
{
    uint64_t handles, coords, outValue, outSize;
    uint32_t numSamples;
    uint32_t pad;
};

// One dispatch of an npTex3dBindless entry over (handle, coordinate) pairs.
static bool runEntry(PxgSlangComputeDevice* dev, const char* entry, const std::vector<uint32_t>& hs,
                     const std::vector<F4>& coords, std::vector<float>& got, std::vector<U4>& size)
{
    const uint32_t N = uint32_t(hs.size());
    const PxgBufferHandle hH = createBufferFrom(dev, hs);
    const PxgBufferHandle hC = createBufferFrom(dev, coords);
    const PxgBufferHandle hO = createBufferFrom(dev, std::vector<float>(N, -12345.0f));
    const PxgBufferHandle hS = createBufferFrom(dev, std::vector<U4>(N, U4{ 0xEEu, 0xEEu, 0xEEu, 0xEEu }));
    TxbParams prm = { dev->getBufferDeviceAddress(hH), dev->getBufferDeviceAddress(hC),
                      dev->getBufferDeviceAddress(hO), dev->getBufferDeviceAddress(hS), N, 0u };
    bool ok;
    {
        Kernel k(Kernel::Bindless{}, "npTex3dBindless.spv", entry, {}, sizeof(TxbParams));
        ok = k.isValid();
        if (ok)
            k.dispatch((N + 63) / 64, {}, prm);
    }
    dev->flush();
    if (ok)
    {
        got = downloadBuffer<float>(dev, hO, N);
        size = downloadBuffer<U4>(dev, hS, N);
    }
    dev->destroyBuffer(hH); dev->destroyBuffer(hC); dev->destroyBuffer(hO); dev->destroyBuffer(hS);
    return ok;
}

enum Kind : int { K_CENTRE = 0, K_TIE, K_ALPHA1, K_SHELL, K_CORNER, K_RANDOM, K_HUGE, K_SZERO, K_NAN, K_COUNT };
static const char* const kKindName[K_COUNT] = { "centre", "tie", "alpha->1", "shell", "corner", "random",
                                                "inf/huge", "signed-zero", "NaN" };

PXG_TEST(npTex3dBindless)
{
    PxgSlangComputeDevice* dev = testDevice();

    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }

    std::mt19937 rng(0xB1D1E55u);
    auto u01 = [&rng]() { return double(rng() % 1000000u) / 1000000.0; };

    // The three volumes.  `vol` carries the values the texture holds AFTER the format
    // read: the float itself, or the engine's UNORM spelling (pxgTex3dRef.h, pinned
    // against the driver by npTex3dBindlessUnormExhaustive).
    pxgtex::Vol vol[3];
    std::vector<PxgTextureHandle> handles(3, 0);
    for (uint32_t v = 0; v < 3; ++v)
    {
        const VolDesc& vd = kVols[v];
        const size_t n = size_t(vd.w) * vd.h * vd.d;
        vol[v].w = vd.w; vol[v].h = vd.h; vol[v].d = vd.d;
        vol[v].t.resize(n);
        switch (vd.fmt)
        {
        case PxgSlangComputeDevice::TextureFormat::R32Float:
        {
            std::vector<float> raw(n);
            for (size_t i = 0; i < n; ++i)
                raw[i] = float(-4.0 + 8.0 * u01());
            // (g): the signed-zero texels and their taps (see the header).  Set after the
            // random fill, so the rest of the volume and every rng draw are unchanged.
            auto set = [&](uint32_t x, uint32_t y, uint32_t z, float f) { raw[(size_t(z) * vd.h + y) * vd.w + x] = f; };
            const uint32_t X = vd.w - 1u;
            set(X, 4, 3, -0.0f); set(X, 5, 3, -1.5f); set(X, 4, 4, -0.75f); set(X, 5, 4, -2.25f);
            set(X, 1, 1, -0.0f); set(X, 2, 1, 1.25f); set(X, 1, 2, -3.0f); set(X, 2, 2, -1.0f);
            set(0, 6, 5, -0.0f);
            for (uint32_t t = 1; t < 8; ++t)
                set(t & 1u, 6u + ((t >> 1) & 1u), 5u + (t >> 2), -0.5f - 0.25f * float(t));
            set(5, 2, 2, 0.0f);
            for (uint32_t t = 1; t < 8; ++t)
                set(5u + (t & 1u), 2u + ((t >> 1) & 1u), 2u + (t >> 2), -1.0f - 0.125f * float(t));
            vol[v].t = raw;
            handles[v] = dev->createTexture3DFmt(vd.w, vd.h, vd.d, vd.fmt, raw.data());
            break;
        }
        case PxgSlangComputeDevice::TextureFormat::R8Unorm:
        {
            std::vector<uint8_t> raw(n);
            for (size_t i = 0; i < n; ++i)
            {
                raw[i] = uint8_t(rng() % 256u);
                vol[v].t[i] = pxgtex::unorm8(raw[i]);
            }
            handles[v] = dev->createTexture3DFmt(vd.w, vd.h, vd.d, vd.fmt, raw.data());
            break;
        }
        default:
        {
            std::vector<uint16_t> raw(n);
            for (size_t i = 0; i < n; ++i)
            {
                raw[i] = uint16_t(rng() % 65536u);
                vol[v].t[i] = pxgtex::unorm16(raw[i]);
            }
            handles[v] = dev->createTexture3DFmt(vd.w, vd.h, vd.d, vd.fmt, raw.data());
            break;
        }
        }
        PXG_CHECK_MSG(handles[v] != 0, "%s: createTexture3DFmt returned the reserved null handle", vd.name);
    }
    PXG_CHECK_MSG(handles[0] != handles[1] && handles[1] != handles[2] && handles[0] != handles[2],
                  "the three volumes must occupy three distinct bindless slots (%u, %u, %u)",
                  unsigned(handles[0]), unsigned(handles[1]), unsigned(handles[2]));

    // ---- the directed sets, per volume ----
    // zsign: the K5 sign a (g) sample must come out with (0 = not a (g) sample,
    // 1 = +0, 2 = -0); checked against the oracle before the launch.
    struct S { F4 c; int kind; int zsign; };
    std::vector<std::vector<S>> per(3);
    const float QNAN = std::numeric_limits<float>::quiet_NaN(), INF = std::numeric_limits<float>::infinity();
    for (uint32_t v = 0; v < 3; ++v)
    {
        const VolDesc& vd = kVols[v];
        const uint32_t dims[3] = { vd.w, vd.h, vd.d };
        auto inner = [&](uint32_t ax) { return float(0.5 + u01() * (dims[ax] - 1.0)); };
        auto put = [&](float x, float y, float z, int kind, int zsign = 0) {
            per[v].push_back({ F4{ x, y, z, 0.f }, kind, zsign });
        };
        auto putAxis = [&](uint32_t ax, float c, int kind) {
            float p[3] = { inner(0), inner(1), inner(2) };
            p[ax] = c;
            put(p[0], p[1], p[2], kind);
        };
        // (a)
        for (uint32_t z = 0; z < vd.d; ++z)
            for (uint32_t y = 0; y < vd.h; ++y)
                for (uint32_t x = 0; x < vd.w; ++x)
                    put(float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f, K_CENTRE);
        for (uint32_t ax = 0; ax < 3; ++ax)
        {
            const uint32_t N = dims[ax];
            // (b) exact ties and their neighbours
            for (uint32_t k = 0; k < 256; ++k)
            {
                const float i = float(k % (N - 1u));
                const float x = i + 0.5f + (float(k) + 0.5f) / 256.0f;   // exact: (2k+1)/512
                putAxis(ax, x, K_TIE);
                putAxis(ax, std::nextafter(x, -INF), K_TIE);
                putAxis(ax, std::nextafter(x, INF), K_TIE);
            }
            // (c) a weight that rounds up to 1.0
            putAxis(ax, float(N - 2u) + 0.5f + 255.75f / 256.0f, K_ALPHA1);
            putAxis(ax, float(N - 1u) + 0.5f + 255.75f / 256.0f, K_ALPHA1);
            // (d) the clamp shell
            const float Nf = float(N);
            const float shell[7] = { -0.7f, 0.2f, 0.49999997f, Nf - 0.5f, std::nextafter(Nf - 0.5f, INF),
                                     Nf + 0.3f, Nf + 5.0f };
            for (float c : shell)
                putAxis(ax, c, K_SHELL);
            // (f) non-finite and huge
            for (float c : { QNAN, INF, -INF, 1e30f, -1e30f })
                putAxis(ax, c, std::isnan(c) ? K_NAN : K_HUGE);
        }
        for (float c : { QNAN, INF, -INF, 1e30f, -1e30f })
            put(c, c, c, std::isnan(c) ? K_NAN : K_HUGE);
        for (int c = 0; c < 8; ++c)
            put((c & 1) ? float(vd.w) + 0.3f : -0.7f, (c & 2) ? float(vd.h) + 0.3f : -0.7f,
                (c & 4) ? float(vd.d) + 0.3f : -0.7f, K_CORNER);
        // (e)
        for (uint32_t i = 0; i < 700; ++i)
            put(float(-1.0 + u01() * (vd.w + 2)), float(-1.0 + u01() * (vd.h + 2)), float(-1.0 + u01() * (vd.d + 2)),
                K_RANDOM);
        // (g), after every rng draw.  y and z sit at texel centres, so every tap but
        // the first has weight 0.
        if (vd.fmt == PxgSlangComputeDevice::TextureFormat::R32Float)
        {
            const float W = float(vd.w);
            for (float x : { W - 0.5f, std::nextafter(W - 0.5f, INF), W + 0.3f, W + 5.0f })
            {
                put(x, 4.5f, 3.5f, K_SZERO, 2);          // -0, negative taps: -0
                put(x, 1.5f, 1.5f, K_SZERO, 1);          // -0, one positive tap: +0
            }
            for (float x : { 0.5f, 0.2f, -0.7f })
                put(x, 6.5f, 5.5f, K_SZERO, 2);          // low face, -0, negative taps: -0
            put(5.5f, 2.5f, 2.5f, K_SZERO, 1);           // +0, negative taps: +0
        }
    }
    // Round-robin, so lane i and lane i+1 sample different textures.
    std::vector<uint32_t> sHandle, sVol;
    std::vector<F4> sCoord;
    std::vector<int> sKind, sZsign;
    {
        size_t cursor[3] = { 0, 0, 0 };
        bool any = true;
        while (any)
        {
            any = false;
            for (uint32_t v = 0; v < 3; ++v)
            {
                if (cursor[v] >= per[v].size())
                    continue;
                any = true;
                const S& e = per[v][cursor[v]++];
                sHandle.push_back(uint32_t(handles[v]));
                sCoord.push_back(e.c);
                sVol.push_back(v);
                sKind.push_back(e.kind);
                sZsign.push_back(e.zsign);
            }
        }
    }
    const uint32_t N = uint32_t(sHandle.size());

    // ---- the expectation, and the bite asserts on it, before anything runs ----
    std::vector<float> want(N, 0.0f);
    for (uint32_t i = 0; i < N; ++i)
        if (sKind[i] != K_NAN)
            want[i] = pxgtex::tex3D(vol[sVol[i]], sCoord[i].x, sCoord[i].y, sCoord[i].z);
    // (g) must really expect the K5 signs, or idx-clamp-hi-missing has nothing to die on.
    {
        size_t zBad = 0, zNeg = 0, zPos = 0;
        for (uint32_t i = 0; i < N; ++i)
        {
            if (sZsign[i] == 0) continue;
            const uint32_t expect = sZsign[i] == 2 ? 0x80000000u : 0x00000000u;
            (sZsign[i] == 2 ? zNeg : zPos)++;
            if (f2u(want[i]) != expect)
            {
                std::printf("    signed-zero sample (%.9g, %.9g, %.9g): oracle 0x%08x, K5 says 0x%08x\n", sCoord[i].x,
                            sCoord[i].y, sCoord[i].z, f2u(want[i]), expect);
                ++zBad;
            }
        }
        PXG_CHECK_MSG(zBad == 0 && zNeg >= 1 && zPos >= 1,
                      "the signed-zero set (g) does not carry the K5 signs: %zu wrong, %zu -0, %zu +0", zBad, zNeg, zPos);
    }
    {
        size_t weakest = ~size_t(0);
        std::printf("    [ note ] bite asserts (samples whose expectation the variant changes, of %u):\n     ", N);
        for (int var = 1; var < pxgtex::V_COUNT; ++var)
        {
            size_t differ = 0;
            for (uint32_t i = 0; i < N; ++i)
            {
                if (sKind[i] == K_NAN || (sKind[i] == K_HUGE && !pxgtex::variantClampsAllCoords(var)))
                    continue;       // the variant is undefined there (pxgTex3dRef.h)
                if (f2u(pxgtex::tex3D(vol[sVol[i]], sCoord[i].x, sCoord[i].y, sCoord[i].z, var)) != f2u(want[i]))
                    ++differ;
            }
            std::printf(" %s %zu%s", pxgtex::variantName(var), differ, var % 6 == 0 ? "\n     " : ";");
            if (pxgtex::variantIsEquivalent(var))
                PXG_CHECK_MSG(differ == 0, "%s is declared EQUIVALENT but changes %zu expectations",
                              pxgtex::variantName(var), differ);
            else
            {
                PXG_CHECK_MSG(differ >= 1, "%s changes NO expectation: its shader mutant would survive",
                              pxgtex::variantName(var));
                if (differ < weakest) weakest = differ;
            }
        }
        std::printf("\n");
    }

    std::vector<float> got;
    std::vector<U4> gotSize;
    const bool ran = runEntry(dev, "tex3dBindlessSample", sHandle, sCoord, got, gotSize);
    PXG_CHECK_MSG(ran, "bindless pipeline creation succeeds");
    if (!ran)
        return;

    // 1. The extent the shader queried must be the extent the volume was created with:
    //    the handle resolved to the RIGHT slot and not merely to A slot.
    size_t sizeBad = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const VolDesc& vd = kVols[sVol[i]];
        if (gotSize[i].x != vd.w || gotSize[i].y != vd.h || gotSize[i].z != vd.d || gotSize[i].w != sHandle[i])
        {
            if (sizeBad < 5)
                std::printf("    sample %u (%s): queried %ux%ux%u handle %u, expected %ux%ux%u handle %u\n", i, vd.name,
                            gotSize[i].x, gotSize[i].y, gotSize[i].z, gotSize[i].w, vd.w, vd.h, vd.d, sHandle[i]);
            ++sizeBad;
        }
    }
    PXG_CHECK_MSG(sizeBad == 0, "%zu of %u samples resolved to the wrong bindless slot", sizeBad, N);

    // 2. Every non-NaN sample, bit for bit against the oracle.  At a centre that is the
    //    stored texel itself (weights 1 and 0); it is checked again against the volume
    //    directly so a wrong oracle cannot hide a wrong centre -- up to the SIGN of a
    //    zero texel, which K5 gives to the zero-weight taps and the bitwise check pins.
    size_t bad[K_COUNT] = {}, count[K_COUNT] = {}, centreBad = 0, unwritten = 0;
    double worst[3] = { 0.0, 0.0, 0.0 };
    for (uint32_t i = 0; i < N; ++i)
    {
        const int k = sKind[i];
        ++count[k];
        if (k == K_NAN)
        {
            if (f2u(got[i]) == f2u(-12345.0f)) ++unwritten;
            continue;
        }
        const uint32_t v = sVol[i];
        if (k == K_CENTRE)
        {
            const float t = vol[v].at(int(sCoord[i].x), int(sCoord[i].y), int(sCoord[i].z));
            if (f2u(got[i]) != f2u(t) && !(t == 0.0f && got[i] == 0.0f)) ++centreBad;
        }
        if (f2u(got[i]) != f2u(want[i]))
        {
            if (bad[k] < 4)
                std::printf("    %s %u (%s) at (%.9g, %.9g, %.9g): got %.9g (0x%08x), want %.9g (0x%08x)\n",
                            kKindName[k], i, kVols[v].name, sCoord[i].x, sCoord[i].y, sCoord[i].z, got[i],
                            f2u(got[i]), want[i], f2u(want[i]));
            ++bad[k];
        }
        const double e = std::fabs(double(got[i]) - trilinear(vol[v], sCoord[i].x, sCoord[i].y, sCoord[i].z));
        if (e > worst[v]) worst[v] = e;
    }
    size_t totalBad = 0;
    for (int k = 0; k < K_COUNT; ++k)
    {
        PXG_CHECK_MSG(count[k] > 0, "no %s samples were generated", kKindName[k]);
        totalBad += bad[k];
    }
    PXG_CHECK_MSG(centreBad == 0, "%zu texel-centre samples are not the stored texel bit for bit", centreBad);
    PXG_CHECK_MSG(totalBad == 0,
                  "%zu non-NaN samples differ from pxgtex::tex3D (centre %zu, tie %zu, alpha->1 %zu, shell %zu, "
                  "corner %zu, random %zu, inf/huge %zu, signed-zero %zu)", totalBad, bad[K_CENTRE], bad[K_TIE],
                  bad[K_ALPHA1], bad[K_SHELL], bad[K_CORNER], bad[K_RANDOM], bad[K_HUGE], bad[K_SZERO]);
    PXG_CHECK_MSG(unwritten == 0, "%zu NaN-coordinate lanes were not written", unwritten);
    std::printf("    [ note ] %u samples bitwise (centre %zu, tie %zu, alpha->1 %zu, shell %zu, corner %zu, random %zu, "
                "inf/huge %zu, signed-zero %zu; %zu NaN written)\n", N - uint32_t(count[K_NAN]), count[K_CENTRE],
                count[K_TIE], count[K_ALPHA1], count[K_SHELL], count[K_CORNER], count[K_RANDOM], count[K_HUGE],
                count[K_SZERO], count[K_NAN] - unwritten);
    std::printf("    [ note ] statistic: worst |tex3D - exact trilinear| %s %.3g, %s %.3g, %s %.3g\n",
                kVols[0].name, worst[0], kVols[1].name, worst[1], kVols[2].name, worst[2]);

    for (uint32_t v = 0; v < 3; ++v)
        dev->destroyTexture3D(handles[v]);
}

// ===========================================================================
// npTex3dBindlessUnormExhaustive -- P9.96's UNORM gate (plan C6.1), the MEASUREMENT
// behind T2/K3 of pxgSlangTex3d.slang and behind pxgtex::unorm8/unorm16.
//
// Every 8-bit code (an R8 volume 23 x 11 x 1 holding 0..252 and a 3 x 1 x 1 holding
// 253..255) and every 16-bit code (an R16 volume 41 x 40 x 40, code = index mod 65536)
// is read back twice: through tex3dBindlessLoad (one raw texel Load) and through
// pxgTex3D at the texel centre.  Both must be bit-equal to the engine's spelling
// float(v) * (1.0f / (2^b - 1)) -- Gu::SDF::decodeSample's, and the one Mesa's
// lp_build_unsigned_norm_to_float is read to compute.  The correctly rounded quotient
// v / (2^b - 1) differs from it on 126 of the 256 and 512 of the 65536 codes, so the
// gate tells the two apart; how many codes the driver returns each for is printed.
//
// Outcome A (the plan's name) is "bit-equal everywhere": the oracle's spelling stands.
// If this fails, the printed counts say which closed form the driver uses (outcome B1:
// re-pin the oracle to it) or that it is none (B2: the _UINT fallback).
// ===========================================================================
PXG_TEST(npTex3dBindlessUnormExhaustive)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }
    struct G { uint32_t w, h, d; PxgSlangComputeDevice::TextureFormat fmt; uint32_t first, bits; };
    const G vols[3] = { { 23u, 11u, 1u, PxgSlangComputeDevice::TextureFormat::R8Unorm, 0u, 8u },
                        { 3u, 1u, 1u, PxgSlangComputeDevice::TextureFormat::R8Unorm, 253u, 8u },
                        { 41u, 40u, 40u, PxgSlangComputeDevice::TextureFormat::R16Unorm, 0u, 16u } };
    PxgTextureHandle h[3];
    std::vector<uint32_t> hs, codeOf, bitsOf;
    std::vector<F4> coordLoad, coordCentre;
    for (int v = 0; v < 3; ++v)
    {
        const G& g = vols[v];
        const size_t n = size_t(g.w) * g.h * g.d;
        std::vector<uint8_t> r8(n);
        std::vector<uint16_t> r16(n);
        for (size_t i = 0; i < n; ++i)
        {
            const uint32_t code = (g.first + uint32_t(i)) % (1u << g.bits);
            r8[i] = uint8_t(code);
            r16[i] = uint16_t(code);
        }
        h[v] = dev->createTexture3DFmt(g.w, g.h, g.d, g.fmt, g.bits == 8u ? (const void*)r8.data() : (const void*)r16.data());
        PXG_CHECK_MSG(h[v] != 0, "volume %d: createTexture3DFmt failed", v);
        for (uint32_t z = 0; z < g.d; ++z)
            for (uint32_t y = 0; y < g.h; ++y)
                for (uint32_t x = 0; x < g.w; ++x)
                {
                    const size_t i = (size_t(z) * g.h + y) * g.w + x;
                    hs.push_back(uint32_t(h[v]));
                    codeOf.push_back((g.first + uint32_t(i)) % (1u << g.bits));
                    bitsOf.push_back(g.bits);
                    coordLoad.push_back(F4{ float(x), float(y), float(z), 0.f });
                    coordCentre.push_back(F4{ float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f, 0.f });
                }
    }
    std::vector<float> gotLoad, gotCentre;
    std::vector<U4> sz;
    PXG_CHECK_MSG(runEntry(dev, "tex3dBindlessLoad", hs, coordLoad, gotLoad, sz), "load pipeline");
    PXG_CHECK_MSG(runEntry(dev, "tex3dBindlessSample", hs, coordCentre, gotCentre, sz), "sample pipeline");
    if (gotLoad.size() != hs.size() || gotCentre.size() != hs.size())
        return;

    // Per width: codes seen, Load == engine spelling, Load == correctly rounded
    // quotient, centre == Load, and how many codes the two spellings tell apart.
    for (uint32_t bits : { 8u, 16u })
    {
        const uint32_t ncodes = 1u << bits;
        const float maxv = float(ncodes - 1u);
        std::vector<int> seen(ncodes, 0);
        size_t eqEngine = 0, eqQuot = 0, centreEqLoad = 0, spellDiffer = 0, n = 0, shown = 0;
        for (uint32_t c = 0; c < ncodes; ++c)
        {
            const float eng = bits == 8u ? pxgtex::unorm8(c) : pxgtex::unorm16(c);
            if (f2u(eng) != f2u(float(c) / maxv)) ++spellDiffer;
        }
        for (size_t i = 0; i < hs.size(); ++i)
        {
            if (bitsOf[i] != bits) continue;
            const uint32_t c = codeOf[i];
            if (seen[c]++) continue;            // each code once (the R16 volume wraps)
            ++n;
            const float eng = bits == 8u ? pxgtex::unorm8(c) : pxgtex::unorm16(c);
            const float quot = float(c) / maxv;   // IEEE division: the correctly rounded quotient
            if (f2u(gotLoad[i]) == f2u(eng)) ++eqEngine;
            else if (shown++ < 4)
                std::printf("    UNORM%u code %u: Load 0x%08x, engine spelling 0x%08x, quotient 0x%08x\n", bits, c,
                            f2u(gotLoad[i]), f2u(eng), f2u(quot));
            if (f2u(gotLoad[i]) == f2u(quot)) ++eqQuot;
            if (f2u(gotCentre[i]) == f2u(gotLoad[i])) ++centreEqLoad;
        }
        PXG_CHECK_MSG(n == ncodes, "UNORM%u: only %zu of %u codes were read", bits, n, ncodes);
        std::printf("    [ note ] UNORM%u: %zu codes; Load == v*(1/%u) on %zu, == correctly rounded v/%u on %zu "
                    "(the two spellings differ on %zu); centre sample == Load on %zu\n", bits, n, ncodes - 1u, eqEngine,
                    ncodes - 1u, eqQuot, spellDiffer, centreEqLoad);
        PXG_CHECK_MSG(eqEngine == ncodes,
                      "UNORM%u: the driver's Load differs from the engine spelling on %zu codes -- outcome B of the "
                      "P9.96 plan (B1: re-pin pxgtex::unorm to the measured closed form; B2: the _UINT fallback)",
                      bits, ncodes - eqEngine);
        PXG_CHECK_MSG(centreEqLoad == ncodes, "UNORM%u: %zu centre samples differ from the raw Load", bits,
                      ncodes - centreEqLoad);
        PXG_CHECK_MSG(spellDiffer == (bits == 8u ? 126u : 512u),
                      "UNORM%u: the host spellings differ on %zu codes (expected %u) -- the gate would not discriminate",
                      bits, spellDiffer, bits == 8u ? 126u : 512u);
    }
    for (int v = 0; v < 3; ++v)
        dev->destroyTexture3D(h[v]);
}

// ===========================================================================
// npTex3dBindlessBench -- P9.96's microbenchmark, OFF unless PXG_TEX3D_BENCH is set
// (ctest runs it as a no-op).  ns per pxgTex3D sample through npTex3dBindless.spv,
// 2^20 samples per dispatch, ten timed dispatches after one warm-up, measured
// separately with ONE handle on every lane (uniform) and with the three handles
// interleaved lane by lane (the NonUniform path).  The kernel also writes the
// queried extent, so this is "one fixture sample", not the bare filter.
// ===========================================================================
PXG_TEST(npTex3dBindlessBench)
{
    if (!std::getenv("PXG_TEX3D_BENCH"))
    {
        std::printf("  NOTE: PXG_TEX3D_BENCH not set -> benchmark skipped\n");
        return;
    }
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
        return;
    std::mt19937 rng(0xBE4C4u);
    PxgTextureHandle handles[3];
    for (uint32_t v = 0; v < 3; ++v)
    {
        const VolDesc& vd = kVols[v];
        const size_t n = size_t(vd.w) * vd.h * vd.d;
        std::vector<float> f(n);
        std::vector<uint8_t> b8(n);
        std::vector<uint16_t> b16(n);
        for (size_t i = 0; i < n; ++i) { f[i] = float(rng() % 1000u) / 125.0f - 4.0f; b8[i] = uint8_t(rng()); b16[i] = uint16_t(rng()); }
        const void* src = vd.fmt == PxgSlangComputeDevice::TextureFormat::R32Float ? (const void*)f.data()
                        : vd.fmt == PxgSlangComputeDevice::TextureFormat::R8Unorm ? (const void*)b8.data() : (const void*)b16.data();
        handles[v] = dev->createTexture3DFmt(vd.w, vd.h, vd.d, vd.fmt, src);
    }
    const uint32_t N = 1u << 20;
    std::vector<F4> coords(N);
    for (uint32_t i = 0; i < N; ++i)
        coords[i] = F4{ float(rng() % 12000u) / 1000.0f, float(rng() % 3000u) / 1000.0f, float(rng() % 5000u) / 1000.0f, 0.f };
    const PxgBufferHandle hCoords = createBufferFrom(dev, coords);
    const PxgBufferHandle hOut = createBufferFrom(dev, std::vector<float>(N, 0.0f));
    const PxgBufferHandle hSize = createBufferFrom(dev, std::vector<U4>(N, U4{ 0u, 0u, 0u, 0u }));
    for (int mode = 0; mode < 2; ++mode)
    {
        std::vector<uint32_t> hs(N);
        for (uint32_t i = 0; i < N; ++i) hs[i] = uint32_t(handles[mode == 0 ? 0u : i % 3u]);
        const PxgBufferHandle hHandles = createBufferFrom(dev, hs);
        TxbParams prm = {
            dev->getBufferDeviceAddress(hHandles), dev->getBufferDeviceAddress(hCoords),
            dev->getBufferDeviceAddress(hOut), dev->getBufferDeviceAddress(hSize), N, 0u };
        Kernel k(Kernel::Bindless{}, "npTex3dBindless.spv", "tex3dBindlessSample", {}, sizeof(TxbParams));
        PXG_CHECK_MSG(k.isValid(), "bench pipeline");
        if (!k.isValid()) return;
        k.dispatch(N / 64, {}, prm);
        dev->flush();
        const int reps = 10;
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r)
        {
            k.dispatch(N / 64, {}, prm);
            dev->flush();
        }
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("    [ bench ] %s handles: %.2f ns/sample (%d x %u samples, %.3f s)\n",
                    mode == 0 ? "uniform    " : "interleaved", sec * 1e9 / (double(reps) * N), reps, N, sec);
        dev->destroyBuffer(hHandles);
    }
    for (uint32_t v = 0; v < 3; ++v) dev->destroyTexture3D(handles[v]);
    dev->destroyBuffer(hCoords); dev->destroyBuffer(hOut); dev->destroyBuffer(hSize);
}

} // namespace
