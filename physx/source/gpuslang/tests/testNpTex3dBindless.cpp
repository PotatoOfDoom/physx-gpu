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

// testNpTex3dBindless.cpp -- P9.12a, the fixture for the CUtexObject replacement.
//
// npTex3d (P2) already established that a 3D volume can be sampled with CUDA's
// filtering and addressing.  What it CANNOT establish is the property the SDF
// path actually depends on: that a texture handle can be STORED IN DEVICE
// MEMORY, loaded by a kernel, and used to sample -- which is what
// `tex3D<float>(mesh.mTexObject, ...)` does, with mTexObject sitting inside a
// struct the kernel reads.  Its texture arrives as a bound descriptor chosen by
// the host at dispatch time, so the handle is never a value at all.
//
// This fixture is that property, and it is deliberately hostile to the ways it
// could pass by accident:
//
//   * THREE volumes with THREE DIFFERENT EXTENTS and THREE DIFFERENT FORMATS
//     (R32F, R8_UNORM, R16_UNORM -- the three the engine's SDF upload asks for),
//     so a shader that sampled the wrong slot would divide by the wrong extent
//     and land somewhere else entirely, and the queried extent is read back and
//     compared per sample;
//   * the handles are INTERLEAVED per sample, so adjacent lanes carry different
//     handles and the non-uniform path is what runs.  A driver that resolved the
//     descriptor per subgroup instead of per lane fails here and nowhere else;
//   * no extent is a power of two and no two are equal, per the standing rule
//     that every bound and size in a fixture must be distinct;
//   * slot 0 is never allocated, so the reserved null handle stays unwritten --
//     which is the state PxgGeometryManager leaves mTexObject in for a mesh
//     with no SDF.
//
// The value comparison follows the P2 arrangement exactly: texel CENTRES must
// be bit-exact (the filter weights are 0 and 1 there), interior points carry
// the implementation's filtering precision and are compared against a host
// double-precision trilinear reference with a documented tolerance.  For the
// UNORM formats the reference is built from the same quantised values the
// upload carries, so quantisation is not folded into the tolerance.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

// The reference filter, in double, over the values the texture actually holds.
static double trilinear(const std::vector<float>& vol, uint32_t W, uint32_t H, uint32_t D,
                        double x, double y, double z)
{
    const double ux = x - 0.5, uy = y - 0.5, uz = z - 0.5;
    const double fx = std::floor(ux), fy = std::floor(uy), fz = std::floor(uz);
    const double ax = ux - fx, ay = uy - fy, az = uz - fz;
    auto clampi = [](double v, uint32_t dim) {
        long i = long(v);
        if (i < 0) i = 0;
        if (i >= long(dim)) i = long(dim) - 1;
        return uint32_t(i);
    };
    double acc = 0.0;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
            {
                const double w = (dx ? ax : 1.0 - ax) * (dy ? ay : 1.0 - ay) * (dz ? az : 1.0 - az);
                const uint32_t ix = clampi(fx + dx, W), iy = clampi(fy + dy, H), iz = clampi(fz + dz, D);
                acc += w * double(vol[(size_t(iz) * H + iy) * W + ix]);
            }
    return acc;
}

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };

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

    // Build the three volumes.  `expect` carries the values the texture holds
    // AFTER quantisation, so the UNORM cases are compared against what was
    // actually stored rather than against what was asked for.
    std::vector<std::vector<float>> expect(3);
    std::vector<std::vector<uint32_t>> rawInt(3);   // the stored integer, UNORM formats only
    std::vector<PxgTextureHandle> handles(3, 0);
    for (uint32_t v = 0; v < 3; ++v)
    {
        const VolDesc& vd = kVols[v];
        const size_t n = size_t(vd.w) * vd.h * vd.d;
        expect[v].resize(n);
        switch (vd.fmt)
        {
        case PxgSlangComputeDevice::TextureFormat::R32Float:
        {
            std::vector<float> raw(n);
            for (size_t i = 0; i < n; ++i)
                raw[i] = float(-4.0 + 8.0 * u01());
            expect[v] = raw;
            handles[v] = dev->createTexture3DFmt(vd.w, vd.h, vd.d, vd.fmt, raw.data());
            break;
        }
        case PxgSlangComputeDevice::TextureFormat::R8Unorm:
        {
            std::vector<uint8_t> raw(n);
            for (size_t i = 0; i < n; ++i)
            {
                raw[i] = uint8_t(rng() % 256u);
                expect[v][i] = float(raw[i]) / 255.0f;   // the UNORM promotion
                rawInt[v].push_back(raw[i]);
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
                expect[v][i] = float(raw[i]) / 65535.0f;
                rawInt[v].push_back(raw[i]);
            }
            handles[v] = dev->createTexture3DFmt(vd.w, vd.h, vd.d, vd.fmt, raw.data());
            break;
        }
        }
        PXG_CHECK_MSG(handles[v] != 0, "%s: createTexture3DFmt returned the reserved null handle",
                      vd.name);
    }
    PXG_CHECK_MSG(handles[0] != handles[1] && handles[1] != handles[2] && handles[0] != handles[2],
                  "the three volumes must occupy three distinct bindless slots (%u, %u, %u)",
                  unsigned(handles[0]), unsigned(handles[1]), unsigned(handles[2]));

    // Sample set, INTERLEAVED across the three volumes so consecutive lanes
    // carry different handles.  Per volume: every texel centre, then random
    // points spilling one texel past every face (the clamp path).
    const uint32_t NR = 700;
    std::vector<uint32_t> sHandle;
    std::vector<F4> sCoord;
    std::vector<uint32_t> sVol;      // which volume this sample belongs to
    std::vector<int> sIsCentre;
    std::vector<size_t> sCentreIdx;
    {
        std::vector<std::vector<std::pair<F4, long>>> per(3);
        for (uint32_t v = 0; v < 3; ++v)
        {
            const VolDesc& vd = kVols[v];
            for (uint32_t z = 0; z < vd.d; ++z)
                for (uint32_t y = 0; y < vd.h; ++y)
                    for (uint32_t x = 0; x < vd.w; ++x)
                        per[v].push_back({ F4{ float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f, 0.f },
                                           long((size_t(z) * vd.h + y) * vd.w + x) });
            for (uint32_t i = 0; i < NR; ++i)
                per[v].push_back({ F4{ float(-1.0 + u01() * (vd.w + 2)),
                                       float(-1.0 + u01() * (vd.h + 2)),
                                       float(-1.0 + u01() * (vd.d + 2)), 0.f },
                                   -1 });
        }
        // Round-robin, so lane i and lane i+1 sample different textures.
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
                const auto& e = per[v][cursor[v]++];
                sHandle.push_back(uint32_t(handles[v]));
                sCoord.push_back(e.first);
                sVol.push_back(v);
                sIsCentre.push_back(e.second >= 0 ? 1 : 0);
                sCentreIdx.push_back(e.second >= 0 ? size_t(e.second) : 0);
            }
        }
    }
    const uint32_t N = uint32_t(sHandle.size());

    std::vector<float> outInit(N, -12345.0f);
    std::vector<U4> sizeInit(N, U4{ 0xEEu, 0xEEu, 0xEEu, 0xEEu });
    const PxgBufferHandle hHandles = createBufferFrom(dev, sHandle);
    const PxgBufferHandle hCoords = createBufferFrom(dev, sCoord);
    const PxgBufferHandle hOut = createBufferFrom(dev, outInit);
    const PxgBufferHandle hSize = createBufferFrom(dev, sizeInit);

    struct P
    {
        uint64_t handles, coords, outValue, outSize;
        uint32_t numSamples;
        uint32_t pad;
    } prm = { dev->getBufferDeviceAddress(hHandles), dev->getBufferDeviceAddress(hCoords),
              dev->getBufferDeviceAddress(hOut), dev->getBufferDeviceAddress(hSize), N, 0u };

    {
        Kernel k(Kernel::Bindless{}, "npTex3dBindless.spv", "tex3dBindlessSample", {}, sizeof(P));
        PXG_CHECK_MSG(k.isValid(), "bindless pipeline creation succeeds");
        if (!k.isValid())
            return;
        k.dispatch((N + 63) / 64, {}, prm);
    }
    dev->flush();

    const auto got = downloadBuffer<float>(dev, hOut, N);
    const auto gotSize = downloadBuffer<U4>(dev, hSize, N);

    // 1. The extent the shader queried must be the extent the volume was
    //    created with.  This is what proves the handle resolved to the RIGHT
    //    slot and not merely to A slot.
    size_t sizeBad = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const VolDesc& vd = kVols[sVol[i]];
        if (gotSize[i].x != vd.w || gotSize[i].y != vd.h || gotSize[i].z != vd.d ||
            gotSize[i].w != sHandle[i])
        {
            if (sizeBad < 5)
                std::printf("    sample %u (%s): queried %ux%ux%u handle %u, expected %ux%ux%u handle %u\n",
                            i, vd.name, gotSize[i].x, gotSize[i].y, gotSize[i].z, gotSize[i].w,
                            vd.w, vd.h, vd.d, sHandle[i]);
            ++sizeBad;
        }
    }
    PXG_CHECK_MSG(sizeBad == 0, "%zu of %u samples resolved to the wrong bindless slot", sizeBad, N);

    // 2. Texel centres.  The trilinear weights there are 0 and 1, so NO
    //    filtering precision enters -- but that only makes the FLOAT volume
    //    bit-exact.  For a UNORM volume the sampler still has to convert the
    //    stored integer to a float, and MEASURED on llvmpipe that conversion
    //    differs from a host `raw / max` (230/255 comes back 0x3f66e6e8 where
    //    the host computes 0x3f66e6e7 -- the signature of a reciprocal multiply
    //    rather than a divide).  Vulkan requires neither spelling.
    //
    //    ULP COUNT IS THE WRONG METRIC HERE and saying so is the point: the
    //    same absolute error is 1 ULP at 0.9 and 29 ULP at 0.07, so an
    //    ULP bound would be a statement about the sample's magnitude rather
    //    than about the conversion.  What IS exactly true, and is what the SDF
    //    path actually depends on, is that the sample IDENTIFIES ITS TEXEL:
    //    round(got * max) must be the integer that was stored.  That has no
    //    fudge factor in it at all.
    size_t centres = 0, centreBad = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (!sIsCentre[i])
            continue;
        ++centres;
        const uint32_t v = sVol[i];
        const VolDesc& vd = kVols[v];
        const float want = expect[v][sCentreIdx[i]];
        bool bad;
        if (vd.fmt == PxgSlangComputeDevice::TextureFormat::R32Float)
        {
            bad = f2u(got[i]) != f2u(want);
        }
        else
        {
            const double maxv = (vd.fmt == PxgSlangComputeDevice::TextureFormat::R8Unorm)
                                    ? 255.0 : 65535.0;
            const long back = std::lround(double(got[i]) * maxv);
            bad = back < 0 || uint32_t(back) != rawInt[v][sCentreIdx[i]];
        }
        if (bad)
        {
            if (centreBad < 5)
                std::printf("    centre %u (%s): got %.9g (0x%08x), want %.9g (0x%08x)\n",
                            i, vd.name, got[i], f2u(got[i]), want, f2u(want));
            ++centreBad;
        }
    }
    PXG_CHECK_MSG(centres > 0, "no texel-centre samples were generated");
    PXG_CHECK_MSG(centreBad == 0,
                  "%zu of %zu texel-centre samples do not identify their texel (bit-exact for "
                  "R32F, round-trips to the stored integer for the UNORM formats)",
                  centreBad, centres);

    // 3. Interior and clamped points against the host reference.  The tolerance
    //    is the filter's, documented in pxgSlangTex3d.slang (T3): CUDA uses
    //    9-bit fixed-point weights, Vulkan's precision is implementation
    //    defined, and neither is the double the reference computes in.  It is
    //    scaled by the volume's value RANGE rather than being an absolute
    //    number, so the UNORM volumes (range 1) and the float volume (range 8)
    //    are held to the same relative standard.
    // The bound is PRINCIPLED per format rather than fitted to what passed:
    //
    //   * for a UNORM volume the filter's own working precision is at best the
    //     quantisation step (1/255 or 1/65535), because the values it is
    //     blending are themselves multiples of that step -- llvmpipe filters
    //     these in fixed point.  Two steps is the bound; MEASURED worst on this
    //     fixture is well inside it and is printed, so a regression that
    //     doubles the error is visible even though it would still pass;
    //   * for a FLOAT volume there is no quantisation, so what remains is the
    //     weight precision alone.  CUDA specifies 9-bit fixed-point weights and
    //     Vulkan leaves it implementation-defined; 1e-4 of the value range
    //     covers both and is far tighter than the UNORM bound.
    //
    // This is the P2 tolerance discussion, now separated by format because the
    // three formats do NOT share a precision.
    size_t interior = 0, interiorBad = 0;
    double worst[3] = { 0.0, 0.0, 0.0 };
    for (uint32_t i = 0; i < N; ++i)
    {
        if (sIsCentre[i])
            continue;
        ++interior;
        const uint32_t v = sVol[i];
        const VolDesc& vd = kVols[v];
        const double want = trilinear(expect[v], vd.w, vd.h, vd.d,
                                      sCoord[i].x, sCoord[i].y, sCoord[i].z);
        double bound;
        switch (vd.fmt)
        {
        case PxgSlangComputeDevice::TextureFormat::R8Unorm:  bound = 2.0 / 255.0; break;
        case PxgSlangComputeDevice::TextureFormat::R16Unorm: bound = 2.0 / 65535.0; break;
        default:                                             bound = 1e-4 * 8.0; break;
        }
        const double err = std::fabs(double(got[i]) - want);
        if (err > worst[v])
            worst[v] = err;
        if (err > bound)
        {
            if (interiorBad < 5)
                std::printf("    interior %u (%s) at (%.4f, %.4f, %.4f): got %.9g, want %.9g, "
                            "err %.3g > %.3g\n",
                            i, vd.name, sCoord[i].x, sCoord[i].y, sCoord[i].z, got[i], want,
                            err, bound);
            ++interiorBad;
        }
    }
    PXG_CHECK_MSG(interior > 0, "no interior samples were generated");
    PXG_CHECK_MSG(interiorBad == 0,
                  "%zu of %zu interior samples exceed their format's filtering bound",
                  interiorBad, interior);
    std::printf("    [ note ] %zu centres, %zu interior; worst filter error %s %.3g, %s %.3g, "
                "%s %.3g\n", centres, interior,
                kVols[0].name, worst[0], kVols[1].name, worst[1], kVols[2].name, worst[2]);

    for (uint32_t v = 0; v < 3; ++v)
        dev->destroyTexture3D(handles[v]);
    dev->destroyBuffer(hHandles);
    dev->destroyBuffer(hCoords);
    dev->destroyBuffer(hOut);
    dev->destroyBuffer(hSize);
}

} // namespace
