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

// RUNTIME_PLAN.md P3 conformance test: drive the PxgCudaShim exactly the way
// ScScene drives PxCudaContext — memAlloc device pointers, memcpyHtoD inputs,
// memsetD32 the output, build a parameter blob of raw device addresses
// (nested in a struct), launchKernel by manifest id, streamSynchronize,
// memcpyDtoH the result — and check the alloc/copy/launch semantics: bitwise
// AXPY output, interior-pointer resolution, memsetD32, and DtoD copy.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgStreamId;
using physx::gpuslang::PxgResult;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static std::vector<uint8_t> shimLoader(const char* file)
{
    return loadShaderBlob(file);
}

PXG_TEST(p3Shim)
{
    PxgSlangComputeDevice* dev = testDevice();

    // Manifest: one kernel, id 0.  Blob = AxpyBuffers(3x u64) + float + uint.
    struct AxpyBuffers { uint64_t x, y, out; };
    struct AxpyParams { AxpyBuffers bufs; float alpha; uint32_t count; };
    const PxgKernelDef manifest[] = {
        { "npShimAxpy.spv", "shimAxpy", uint32_t(sizeof(AxpyParams)), 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &shimLoader);

    const uint32_t N = 8192;
    std::mt19937 rng(0xC0FFEE);
    std::vector<float> x(N), y(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        x[i] = float(int(rng() % 4001) - 2000) * 0.25f;   // exact quarter-integers
        y[i] = float(int(rng() % 4001) - 2000) * 0.5f;    // exact half-integers
    }
    const float alpha = 3.0f;                              // exact => a*x + y exact

    PxgDevicePtr dx = 0, dy = 0, dout = 0;
    PXG_CHECK_MSG(shim.memAlloc(&dx, N * 4) == PxgResult::Success, "memAlloc x");
    PXG_CHECK_MSG(shim.memAlloc(&dy, N * 4) == PxgResult::Success, "memAlloc y");
    PXG_CHECK_MSG(shim.memAlloc(&dout, N * 4) == PxgResult::Success, "memAlloc out");
    if (dx == 0 || dy == 0 || dout == 0)
    {
        std::printf("  NOTE: shim has no device-address allocations (no BDA) -> skip\n");
        return;
    }

    PXG_CHECK_MSG(shim.memcpyHtoD(dx, x.data(), N * 4) == PxgResult::Success, "HtoD x");
    PXG_CHECK_MSG(shim.memcpyHtoD(dy, y.data(), N * 4) == PxgResult::Success, "HtoD y");
    PXG_CHECK_MSG(shim.memsetD32(dout, 0xdeadbeefu, N) == PxgResult::Success, "memsetD32 out");

    // Interior-pointer resolution: address of element 100 of x must land in
    // x's buffer at the right offset (CUDA flat-address-space parity).
    {
        physx::gpuslang::PxgBufferHandle b0, b1;
        size_t o0, o1;
        const bool r0 = shim.resolve(dx, &b0, &o0);
        const bool r1 = shim.resolve(dx + 100 * 4, &b1, &o1);
        PXG_CHECK_MSG(r0 && r1 && b0 == b1 && o0 == 0 && o1 == 400,
                      "interior device pointer resolves to (same buffer, +400)");
        physx::gpuslang::PxgBufferHandle bx;
        size_t ox;
        PXG_CHECK_MSG(!shim.resolve(dx + N * 4, &bx, &ox), "one-past-the-end pointer does not resolve");
    }

    // Build the parameter blob of raw device addresses (nested) and launch.
    AxpyParams prm;
    prm.bufs.x = dx;
    prm.bufs.y = dy;
    prm.bufs.out = dout;
    prm.alpha = alpha;
    prm.count = N;
    PXG_CHECK_MSG(shim.launchKernel(0, (N + 63) / 64, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launchKernel by manifest id");
    shim.streamSynchronize();

    std::vector<float> got(N, 0.f);
    PXG_CHECK_MSG(shim.memcpyDtoH(got.data(), dout, N * 4) == PxgResult::Success, "DtoH out");

    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float ref = alpha * x[i] + y[i];
        if (f2u(ref) != f2u(got[i]))
        {
            if (mm < 5)
                std::printf("  AXPY MM %u: ref %g got %g\n", i, ref, got[i]);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "shim AXPY launch bitwise-correct through BDA marshalling (%zu bad)", mm);

    // Interior-offset HtoD/DtoH (PhysX copies into sub-ranges of pooled
    // buffers): overwrite a middle slice of dout via an interior pointer,
    // read it back through the same interior pointer, and confirm the slice
    // changed while its neighbors did not.
    {
        const uint32_t K = 1000, M = 256;
        std::vector<float> patch(M);
        for (uint32_t i = 0; i < M; ++i)
            patch[i] = -777.0f - float(i);
        PXG_CHECK_MSG(shim.memcpyHtoD(dout + K * 4, patch.data(), M * 4) == PxgResult::Success,
                      "interior-offset HtoD");
        shim.streamSynchronize();

        std::vector<float> back(M, 0.f);
        PXG_CHECK_MSG(shim.memcpyDtoH(back.data(), dout + K * 4, M * 4) == PxgResult::Success,
                      "interior-offset DtoH");
        PXG_CHECK_MSG(std::memcmp(back.data(), patch.data(), M * 4) == 0,
                      "interior-offset HtoD/DtoH round-trips exactly");

        std::vector<float> whole(N, 0.f);
        shim.memcpyDtoH(whole.data(), dout, N * 4);
        size_t nbad = 0;
        for (uint32_t i = 0; i < N; ++i)
        {
            const float want = (i >= K && i < K + M) ? patch[i - K] : (alpha * x[i] + y[i]);
            if (f2u(want) != f2u(whole[i]))
                ++nbad;
        }
        PXG_CHECK_MSG(nbad == 0, "interior HtoD touched exactly the slice [K, K+M) (%zu bad)", nbad);
    }

    // DtoD: copy out -> a fresh allocation, read both back, must be equal.
    PxgDevicePtr dcopy = 0;
    PXG_CHECK_MSG(shim.memAlloc(&dcopy, N * 4) == PxgResult::Success, "memAlloc copy");
    PXG_CHECK_MSG(shim.memcpyDtoD(dcopy, dout, N * 4) == PxgResult::Success, "DtoD copy");
    std::vector<float> outNow(N, 0.f), copied(N, 0.f);
    shim.memcpyDtoH(outNow.data(), dout, N * 4);
    shim.memcpyDtoH(copied.data(), dcopy, N * 4);
    PXG_CHECK_MSG(std::memcmp(copied.data(), outNow.data(), N * 4) == 0, "DtoD copy is bitwise identical");

    // Stream/event surface: run a launch and synchronize on it through the
    // event API instead of streamSynchronize; the result must be visible
    // after eventSynchronize, and eventQuery must then report completion.
    {
        PxgStreamId s1 = 0, s2 = 0;
        PXG_CHECK_MSG(shim.streamCreate(&s1) == PxgResult::Success && s1 != 0, "streamCreate s1");
        PXG_CHECK_MSG(shim.streamCreate(&s2) == PxgResult::Success && s2 != 0 && s2 != s1, "streamCreate s2");

        uint32_t ev = 0;
        PXG_CHECK_MSG(shim.eventCreate(&ev) == PxgResult::Success && ev != 0, "eventCreate");

        PxgDevicePtr d2 = 0;
        PXG_CHECK_MSG(shim.memAlloc(&d2, N * 4) == PxgResult::Success, "memAlloc stream out");
        shim.memsetD32(d2, 0, N);
        AxpyParams p2;      // dx/dy are still alive here (freed at the very end)
        p2.bufs.x = dx;
        p2.bufs.y = dy;
        p2.bufs.out = d2;
        p2.alpha = alpha;
        p2.count = N;
        PXG_CHECK_MSG(shim.launchKernel(0, (N + 63) / 64, 1, 1, &p2, sizeof(p2)) == PxgResult::Success,
                      "launchKernel on stream");
        PXG_CHECK_MSG(shim.eventRecord(ev, s1) == PxgResult::Success, "eventRecord");
        // A second stream waiting on the event then a dependent sync: on the
        // single serialized queue the dependency is already honored.
        PXG_CHECK_MSG(shim.streamWaitEvent(s2, ev) == PxgResult::Success, "streamWaitEvent");
        PXG_CHECK_MSG(shim.eventSynchronize(ev) == PxgResult::Success, "eventSynchronize");
        PXG_CHECK_MSG(shim.eventQuery(ev) == PxgResult::Success, "eventQuery reports complete after sync");

        std::vector<float> s(N, 0.f);
        shim.memcpyDtoH(s.data(), d2, N * 4);
        size_t sb = 0;
        for (uint32_t i = 0; i < N; ++i)
            if (f2u(alpha * x[i] + y[i]) != f2u(s[i]))
                ++sb;
        PXG_CHECK_MSG(sb == 0, "event-synchronized launch result is visible and correct (%zu bad)", sb);

        // Negative handles.
        PXG_CHECK_MSG(shim.eventSynchronize(9999) == PxgResult::NotFound, "eventSynchronize rejects bad handle");
        PXG_CHECK_MSG(shim.streamDestroy(9999) == PxgResult::NotFound, "streamDestroy rejects bad handle");

        shim.eventDestroy(ev);
        shim.streamDestroy(s1);
        shim.streamDestroy(s2);
        shim.memFree(d2);
    }

    // memFree parity: interior pointer must NOT free; base must.
    PXG_CHECK_MSG(shim.memFree(dx + 4) == PxgResult::NotFound, "memFree rejects an interior pointer");
    PXG_CHECK_MSG(shim.memFree(dx) == PxgResult::Success, "memFree accepts the base pointer");
    shim.memFree(dy);
    shim.memFree(dout);
    shim.memFree(dcopy);
}

}  // namespace
