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

// RUNTIME_PLAN.md P2 battery: the tex3D replacement (3D sampled image,
// trilinear, clamp-to-edge).  Two obligations:
//   1. voxel-center samples are EXACT (frac == 0 degenerates the filter to
//      a plain texel read on any conformant implementation; the coordinate
//      normalization round-trips exactly for power-of-two extents);
//   2. arbitrary points (including far outside the volume, exercising
//      clamp-to-edge on all six faces) match a double-precision trilinear
//      reference within a filtering tolerance.  Vulkan does NOT pin the
//      filter's weight precision, so this path is a documented REPLACEMENT
//      of CUDA tex3D, not a bit-exact port; fixed-point hardware filters
//      get a per-device tolerance in P5 (PXG_TEX3D_TOL overrides).

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static const uint32_t W = 32, H = 16, D = 8;

static double trilerpRef(const std::vector<float>& vol, double x, double y, double z)
{
    // GL/CUDA linear filtering with clamp-to-edge, in double precision.
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

PXG_TEST(npTex3d)
{
    PxgSlangComputeDevice* dev = testDevice();

    std::mt19937 rng(31337);
    std::vector<float> vol(size_t(W) * H * D);
    for (auto& v : vol)
        v = -4.f + 8.f * float(rng() % 1000000) / 1000000.f;

    const PxgTextureHandle tex = dev->createTexture3D(W, H, D, vol.data());
    if (tex == 0)
    {
        std::printf("  NOTE: backend has no 3D texture support -> skip\n");
        return;
    }

    // Sample set: all voxel centers first, then random points spilling
    // one voxel beyond every face.
    const uint32_t NC = W * H * D;
    const uint32_t NR = 2048;
    const uint32_t N = NC + NR;
    std::vector<float> pts(4 * N, 0.f);
    uint32_t n = 0;
    for (uint32_t z = 0; z < D; ++z)
        for (uint32_t y = 0; y < H; ++y)
            for (uint32_t x = 0; x < W; ++x, ++n)
            {
                pts[4 * n + 0] = float(x) + 0.5f;
                pts[4 * n + 1] = float(y) + 0.5f;
                pts[4 * n + 2] = float(z) + 0.5f;
            }
    for (uint32_t i = 0; i < NR; ++i, ++n)
    {
        pts[4 * n + 0] = -1.f + float(rng() % 100000) / 100000.f * (W + 2);
        pts[4 * n + 1] = -1.f + float(rng() % 100000) / 100000.f * (H + 2);
        pts[4 * n + 2] = -1.f + float(rng() % 100000) / 100000.f * (D + 2);
    }

    PxgBufferHandle bPts = createBufferFrom(dev, pts);
    std::vector<float> outInit(N, 0.f);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numSamples; float invW, invH, invD; } prm = { N, 1.f / W, 1.f / H, 1.f / D };
    const std::vector<uint8_t> blob = loadShaderBlob("npTex3d.spv");
    const uint32_t strides[2] = { 16, 4 };
    const PxgPipelineHandle pipe = dev->createPipelineTex(blob.data(), blob.size(), "tex3dSample",
                                                          2, strides, sizeof(P), 1);
    PXG_CHECK_MSG(pipe != 0, "texture pipeline creation succeeds");

    const PxgBufferHandle bufs[2] = { bPts, bOut };
    dev->dispatchTex(pipe, (N + 63) / 64, 1, 1, bufs, 2, &tex, 1, &prm, sizeof(P));
    dev->flush();
    const auto g = downloadBuffer<float>(dev, bOut, N);

    // 1. centers: exact
    size_t mm = 0;
    for (uint32_t i = 0; i < NC; ++i)
        if (f2u(g[i]) != f2u(vol[i]))
        {
            if (mm < 5)
                std::printf("  CENTER MM %u: got %g want %g\n", i, g[i], vol[i]);
            ++mm;
        }
    PXG_CHECK_MSG(mm == 0, "voxel-center samples are bitwise texel reads (%zu bad)", mm);

    // 2. arbitrary points: filtering tolerance vs double reference
    const char* tolEnv = std::getenv("PXG_TEX3D_TOL");
    const double tol = tolEnv ? std::atof(tolEnv) : 1e-4;
    size_t bad = 0;
    double worst = 0.0;
    for (uint32_t i = NC; i < N; ++i)
    {
        const double ref = trilerpRef(vol, pts[4 * i], pts[4 * i + 1], pts[4 * i + 2]);
        const double err = std::fabs(double(g[i]) - ref);
        worst = err > worst ? err : worst;
        if (err > tol * (1.0 + std::fabs(ref)))
        {
            if (bad < 5)
                std::printf("  LERP MM %u at (%g,%g,%g): got %.9g want %.9g\n", i,
                            pts[4 * i], pts[4 * i + 1], pts[4 * i + 2], g[i], ref);
            ++bad;
        }
    }
    std::printf("  filtered samples: worst abs error %.3g (tol %.3g)\n", worst, tol);
    PXG_CHECK_MSG(bad == 0, "filtered samples within tolerance of the trilinear reference (%zu bad)", bad);

    dev->destroyPipeline(pipe);
    dev->destroyBuffer(bPts);
    dev->destroyBuffer(bOut);
    dev->destroyTexture3D(tex);
}

}  // namespace
