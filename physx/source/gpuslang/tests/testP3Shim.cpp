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

// P5 Step 4b: the pinned-host-memory seam (cuMemHostAlloc + mapped device
// pointers).  The engine writes pinned memory on the CPU and launches kernels
// that read it by device address (copy-descriptor queues, pinned mirrors,
// PxPinnedArray) — and reads kernel output back through the same mapping with
// no explicit DtoH.  Both directions are verified bitwise through shimAxpy.
// An ATOMIC read-modify-write through a buffer device address that points into
// PINNED HOST memory, and the numeric fact that makes the ABI's hostPtrMask
// necessary in the first place.
//
// p3ShimPinned proves an ordinary store into pinned memory is visible through the
// CPU mapping.  The convex-vs-trimesh core kernel's prologue needs more than that:
// it does an atomicMax on `maxTempMemRequirement` (convexMesh.cu:674), a word
// PhysX allocates pinned and reads on the CPU with no copy at all
// (PxgNarrowphaseCore.cpp:223, :1644).  If an atomic on host-visible memory were a
// fault or a silent no-op here, the collisionStackSize overflow warning would stop
// firing while contacts were being dropped -- exactly the kind of silence this
// port has been bitten by before.
PXG_TEST(p3ShimPinnedAtomic)
{
    PxgSlangComputeDevice* dev = testDevice();

    struct PinnedAtomicParams { uint64_t target, counter; uint32_t count, pad; };
    const PxgKernelDef manifest[] = {
        { "pinnedAtomicBda.spv", "pinnedAtomicLaunch", uint32_t(sizeof(PinnedAtomicParams)), 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &shimLoader);

    void* hw = nullptr;
    if (shim.memHostAlloc(&hw, 2 * sizeof(uint32_t)) != PxgResult::Success)
    {
        std::printf("  NOTE: no pinned-mapped support on this backend -> skip\n");
        return;
    }
    uint32_t* words = static_cast<uint32_t*>(hw);
    words[0] = 0;
    words[1] = 0;

    const PxgDevicePtr dw = shim.hostGetDevicePointer(hw);
    PXG_CHECK_MSG(dw != 0, "the pinned word has a device address");

    // THE ABI's hostPtrMask IS A CORRECTNESS REQUIREMENT THIS PLATFORM CANNOT
    // DEMONSTRATE.  Under CUDA's unified addressing a pinned host pointer IS a
    // device pointer, which is why PhysX hands such pointers straight to kernels;
    // Vulkan makes no such promise, so the bridge must translate the argument.
    //
    // On llvmpipe it does not have to: the software rasteriser lives in the
    // calling process's address space and its buffer device addresses ARE the CPU
    // mappings, so the two numbers coincide and an untranslated pointer works by
    // accident.  This is reported rather than asserted in either direction --
    // demanding they differ fails here, and demanding they match would bake a
    // backend accident into the suite.  The mask is exercised for real only on a
    // device where they diverge.
    if (uint64_t(dw) == uint64_t(uintptr_t(hw)))
        std::printf("  NOTE: this backend maps pinned memory at its device address "
                    "(host == dev == %llu); the ABI hostPtrMask is inert here\n",
                    (unsigned long long)dw);
    PXG_CHECK_MSG(shim.hostGetDevicePointer(static_cast<uint8_t*>(hw) + 4) == dw + 4,
                  "the interior pinned pointer the mask would translate resolves correctly");

    const uint32_t N = 4096;
    uint32_t expectedMax = 0;
    for (uint32_t i = 0; i < N; ++i)
        expectedMax = std::max(expectedMax, ((i * 2654435761u) >> 8) & 0xffffu);

    PinnedAtomicParams prm;
    prm.target = dw;
    prm.counter = dw + 4;
    prm.count = N;
    prm.pad = 0;
    PXG_CHECK_MSG(shim.launchKernel(0, (N + 63) / 64, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launchKernel doing atomics on pinned memory");
    shim.streamSynchronize();

    // Read THROUGH THE MAPPING, no DtoH -- the engine's model for this word.
    PXG_CHECK_MSG(words[0] == expectedMax, "atomicMax on pinned memory: got=%u exp=%u",
                  words[0], expectedMax);
    PXG_CHECK_MSG(words[1] == N, "atomicAdd on pinned memory lost updates: got=%u exp=%u",
                  words[1], N);

    shim.memFreeHost(hw);
}

PXG_TEST(p3ShimPinned)
{
    PxgSlangComputeDevice* dev = testDevice();

    struct AxpyBuffers { uint64_t x, y, out; };
    struct AxpyParams { AxpyBuffers bufs; float alpha; uint32_t count; };
    const PxgKernelDef manifest[] = {
        { "npShimAxpy.spv", "shimAxpy", uint32_t(sizeof(AxpyParams)), 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &shimLoader);

    const uint32_t N = 4096;

    // Pinned input + output; ordinary device buffer for y.
    void* hx = nullptr;
    void* hout = nullptr;
    if (shim.memHostAlloc(&hx, N * 4) != PxgResult::Success)
    {
        std::printf("  NOTE: no pinned-mapped support on this backend -> skip\n");
        return;
    }
    PXG_CHECK_MSG(shim.memHostAlloc(&hout, N * 4) == PxgResult::Success, "memHostAlloc out");
    PxgDevicePtr dy = 0;
    PXG_CHECK_MSG(shim.memAlloc(&dy, N * 4) == PxgResult::Success, "memAlloc y");

    // CPU writes the pinned input directly (the engine's model: no HtoD).
    float* x = static_cast<float*>(hx);
    std::vector<float> y(N);
    std::mt19937 rng(0xBEEF);
    for (uint32_t i = 0; i < N; ++i)
    {
        x[i] = float(int(rng() % 4001) - 2000) * 0.25f;
        y[i] = float(int(rng() % 4001) - 2000) * 0.5f;
    }
    shim.memcpyHtoD(dy, y.data(), N * 4);
    std::memset(hout, 0xcd, N * 4);

    // Mapped device pointers: base + interior resolution, unknown rejection.
    const PxgDevicePtr dxm = shim.hostGetDevicePointer(hx);
    const PxgDevicePtr doutm = shim.hostGetDevicePointer(hout);
    PXG_CHECK_MSG(dxm != 0 && doutm != 0, "pinned allocations have device addresses");
    PXG_CHECK_MSG(shim.hostGetDevicePointer(static_cast<uint8_t*>(hx) + 400) == dxm + 400,
                  "interior pinned pointer resolves at the right offset");
    PXG_CHECK_MSG(shim.hostGetDevicePointer(&rng) == 0, "unregistered host memory yields the null device pointer");
    {
        // The pinned device address participates in the flat address space
        // (memcpys/resolve), like any cuMemHostAlloc'd pointer.
        physx::gpuslang::PxgBufferHandle b;
        size_t off;
        PXG_CHECK_MSG(shim.resolve(dxm + 100 * 4, &b, &off) && off == 400,
                      "pinned device address resolves in the flat address space");
    }

    const float alpha = 3.0f;
    AxpyParams prm;
    prm.bufs.x = dxm;
    prm.bufs.y = dy;
    prm.bufs.out = doutm;
    prm.alpha = alpha;
    prm.count = N;
    PXG_CHECK_MSG(shim.launchKernel(0, (N + 63) / 64, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launchKernel reading+writing pinned memory");
    shim.streamSynchronize();

    // CPU reads the kernel's output THROUGH THE MAPPING (no DtoH).
    const float* out = static_cast<const float*>(hout);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        uint32_t ru, gu;
        const float ref = alpha * x[i] + y[i];
        std::memcpy(&ru, &ref, 4);
        std::memcpy(&gu, &out[i], 4);
        if (ru != gu)
            ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "pinned round-trip bitwise-correct (%zu bad)", mm);

    // DtoH from a pinned device address must also work (staged-copy parity).
    std::vector<float> staged(N, 0.f);
    PXG_CHECK_MSG(shim.memcpyDtoH(staged.data(), doutm, N * 4) == PxgResult::Success, "DtoH from pinned");
    PXG_CHECK_MSG(std::memcmp(staged.data(), out, N * 4) == 0, "DtoH from pinned matches the mapping");

    // Free semantics: interior rejected, base accepted, unknown rejected.
    PXG_CHECK_MSG(shim.memFreeHost(static_cast<uint8_t*>(hx) + 4) == PxgResult::NotFound,
                  "memFreeHost rejects an interior pointer");
    PXG_CHECK_MSG(shim.memFreeHost(hx) == PxgResult::Success, "memFreeHost accepts the base pointer");
    PXG_CHECK_MSG(shim.hostGetDevicePointer(hx) == 0, "freed pinned memory no longer resolves");
    shim.memFreeHost(hout);
    shim.memFree(dy);
}


// The async transfer model (stream-ordered, no drain).  memcpyDtoHAsync into pinned
// memory and memcpyHtoDAsync out of it are recorded as device copies that touch the
// pinned bytes WHEN THEY EXECUTE, as CUDA's DMA does:
//   * a DtoHAsync into pinned memory, then an HtoDAsync of the same bytes, must carry
//     the kernel's result -- a call-time snapshot of the pinned source (the v2 staged
//     upload before this model) reads the bytes from before the DtoH ran;
//   * an HtoDAsync from PAGEABLE memory is a snapshot (CUDA stages it before
//     returning): overwriting the source right after the call must not leak through;
//   * a synchronous HtoD from pinned memory a kernel is still writing must see the
//     write (it waits, as cuMemcpyHtoD does);
//   * an event recorded behind pending work makes progress under eventQuery polling
//     and eventSynchronize (event records no longer submit; the queries must).
PXG_TEST(p3ShimAsyncPinned)
{
    PxgSlangComputeDevice* dev = testDevice();

    struct AxpyBuffers { uint64_t x, y, out; };
    struct AxpyParams { AxpyBuffers bufs; float alpha; uint32_t count; };
    const PxgKernelDef manifest[] = {
        { "npShimAxpy.spv", "shimAxpy", uint32_t(sizeof(AxpyParams)), 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &shimLoader);

    const uint32_t N = 4096;
    void* hp = nullptr;
    if (shim.memHostAlloc(&hp, N * 4) != PxgResult::Success)
    {
        std::printf("  NOTE: no pinned-mapped support on this backend -> skip\n");
        return;
    }
    void* hq = nullptr;
    PXG_CHECK_MSG(shim.memHostAlloc(&hq, N * 4) == PxgResult::Success, "memHostAlloc q");
    PxgDevicePtr dx = 0, dy = 0, dout = 0, dcopy = 0, dsnap = 0, dsync = 0;
    PXG_CHECK_MSG(shim.memAlloc(&dx, N * 4) == PxgResult::Success, "memAlloc x");
    PXG_CHECK_MSG(shim.memAlloc(&dy, N * 4) == PxgResult::Success, "memAlloc y");
    PXG_CHECK_MSG(shim.memAlloc(&dout, N * 4) == PxgResult::Success, "memAlloc out");
    PXG_CHECK_MSG(shim.memAlloc(&dcopy, N * 4) == PxgResult::Success, "memAlloc copy");
    PXG_CHECK_MSG(shim.memAlloc(&dsnap, N * 4) == PxgResult::Success, "memAlloc snap");
    PXG_CHECK_MSG(shim.memAlloc(&dsync, N * 4) == PxgResult::Success, "memAlloc sync");

    std::vector<float> x(N), y(N), ref(N);
    std::mt19937 rng(0xA5A5);
    for (uint32_t i = 0; i < N; ++i)
    {
        x[i] = float(int(rng() % 4001) - 2000) * 0.25f;
        y[i] = float(int(rng() % 4001) - 2000) * 0.5f;
        ref[i] = 3.0f * x[i] + y[i];
    }
    PXG_CHECK(shim.memcpyHtoD(dx, x.data(), N * 4) == PxgResult::Success);
    PXG_CHECK(shim.memcpyHtoD(dy, y.data(), N * 4) == PxgResult::Success);
    shim.streamSynchronize();
    std::memset(hp, 0xcd, N * 4);   // stale pattern the copies must not deliver

    AxpyParams prm;
    prm.bufs.x = dx;
    prm.bufs.y = dy;
    prm.bufs.out = dout;
    prm.alpha = 3.0f;
    prm.count = N;
    PXG_CHECK(shim.launchKernel(0, (N + 63) / 64, 1, 1, &prm, sizeof(prm)) == PxgResult::Success);
    PXG_CHECK_MSG(shim.memcpyDtoHAsync(hp, dout, N * 4) == PxgResult::Success, "DtoHAsync into pinned");
    PXG_CHECK_MSG(shim.memcpyHtoDAsync(dcopy, hp, N * 4) == PxgResult::Success, "HtoDAsync from pinned");

    // Pageable async source: a snapshot.  Overwrite it right after the call.
    std::vector<float> pageable(x);
    PXG_CHECK_MSG(shim.memcpyHtoDAsync(dsnap, pageable.data(), N * 4) == PxgResult::Success, "HtoDAsync pageable");
    std::fill(pageable.begin(), pageable.end(), -1.0f);

    // Event behind the pending work: polling must make progress (bounded).
    uint32_t ev = 0;
    PXG_CHECK(shim.eventCreate(&ev) == PxgResult::Success);
    PXG_CHECK(shim.eventRecord(ev, 0) == PxgResult::Success);
    bool done = false;
    for (uint32_t i = 0; i < 2000000 && !done; ++i)
        done = shim.eventQuery(ev) == PxgResult::Success;
    PXG_CHECK_MSG(done, "eventQuery on a recorded event reports completion (the batch is submitted)");
    PXG_CHECK(shim.eventSynchronize(ev) == PxgResult::Success);

    const float* pinned = static_cast<const float*>(hp);
    PXG_CHECK_MSG(std::memcmp(pinned, ref.data(), N * 4) == 0, "DtoHAsync delivered the kernel result into pinned memory");
    std::vector<float> got(N, 0.f);
    PXG_CHECK(shim.memcpyDtoH(got.data(), dcopy, N * 4) == PxgResult::Success);
    PXG_CHECK_MSG(std::memcmp(got.data(), ref.data(), N * 4) == 0,
                  "HtoDAsync from pinned read the bytes the earlier DtoHAsync wrote (execution-time read)");
    PXG_CHECK(shim.memcpyDtoH(got.data(), dsnap, N * 4) == PxgResult::Success);
    PXG_CHECK_MSG(std::memcmp(got.data(), x.data(), N * 4) == 0,
                  "HtoDAsync from pageable memory is a call-time snapshot");

    // Synchronous HtoD from pinned memory a recorded kernel writes.
    std::memset(hq, 0xcd, N * 4);
    prm.bufs.out = shim.hostGetDevicePointer(hq);
    PXG_CHECK(shim.launchKernel(0, (N + 63) / 64, 1, 1, &prm, sizeof(prm)) == PxgResult::Success);
    PXG_CHECK(shim.memcpyHtoD(dsync, hq, N * 4) == PxgResult::Success);
    shim.streamSynchronize();
    PXG_CHECK(shim.memcpyDtoH(got.data(), dsync, N * 4) == PxgResult::Success);
    PXG_CHECK_MSG(std::memcmp(got.data(), ref.data(), N * 4) == 0,
                  "synchronous HtoD from pinned memory waits for the kernel writing it");

    shim.eventDestroy(ev);
    shim.memFree(dx);
    shim.memFree(dy);
    shim.memFree(dout);
    shim.memFree(dcopy);
    shim.memFree(dsnap);
    shim.memFree(dsync);
    shim.memFreeHost(hp);
    shim.memFreeHost(hq);
}

}  // namespace
