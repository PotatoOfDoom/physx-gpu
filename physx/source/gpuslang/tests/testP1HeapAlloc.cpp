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

// RUNTIME_PLAN.md P1.2 test: the buddy sub-allocator (PxgSlangHeapAllocator).
// Buddy invariants (alignment, non-overlap, coalescing), root growth,
// deferred frees, exceptional oversize allocations, per-group statistics,
// plus a randomized stress run checked against a shadow interval model and a
// real device round-trip through a sub-allocated range.

#include "pxgTestFramework.h"
#include "PxgSlangHeapAllocator.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgSlangHeapAllocator;
using physx::gpuslang::PxgSlangHeapAlloc;

static const uint64_t HEAP = 1u << 20; // 1 MiB roots keep the test heap small

PXG_TEST(p1HeapAlloc)
{
    PxgSlangComputeDevice* dev = testDevice();

    // --- alignment + rounding + basic reuse ---
    {
        PxgSlangHeapAllocator heap(dev, HEAP);
        const PxgSlangHeapAlloc a = heap.allocate(1, 0);
        PXG_CHECK_MSG(a.valid() && a.size == 128 && a.offset % 128 == 0,
                      "1-byte request rounds to the 128 B minimum class");
        const PxgSlangHeapAlloc b = heap.allocate(129, 0);
        PXG_CHECK_MSG(b.valid() && b.size == 256 && b.offset % 128 == 0,
                      "129-byte request rounds to 256 B");
        PXG_CHECK_MSG(heap.totalBytesInUse() == 128 + 256, "stats track rounded sizes");
        heap.deallocate(a);
        heap.deallocate(b);
        PXG_CHECK_MSG(heap.totalBytesInUse() == 0, "stats drain to zero");

        // Fully drained: the whole root must be allocatable again (coalescing).
        const PxgSlangHeapAlloc whole = heap.allocate(HEAP, 0);
        PXG_CHECK_MSG(whole.valid() && heap.numRoots() == 1,
                      "buddy coalescing recovers the full root after drain");
        heap.deallocate(whole);
    }

    // --- root growth + fragmentation behavior ---
    {
        PxgSlangHeapAllocator heap(dev, HEAP);
        PxgSlangHeapAlloc quarters[4];
        for (int i = 0; i < 4; ++i)
        {
            quarters[i] = heap.allocate(HEAP / 4, 1);
            PXG_CHECK_MSG(quarters[i].valid(), "quarter-root allocation %d succeeds", i);
        }
        PXG_CHECK_MSG(heap.numRoots() == 1, "four quarters fill exactly one root");

        const PxgSlangHeapAlloc extra = heap.allocate(HEAP / 4, 1);
        PXG_CHECK_MSG(extra.valid() && heap.numRoots() == 2, "full root triggers growth by one root");

        // Free two NON-buddy quarters (0 and 2): a half-root request must not
        // fit in root 0 (no adjacent buddy pair), so it lands elsewhere.
        heap.deallocate(quarters[0]);
        heap.deallocate(quarters[2]);
        const PxgSlangHeapAlloc half = heap.allocate(HEAP / 2, 1);
        PXG_CHECK_MSG(half.valid(), "half-root allocation succeeds somewhere");
        PXG_CHECK_MSG(!(half.buffer == quarters[0].buffer && half.offset < HEAP * 3 / 4),
                      "fragmented non-buddy quarters cannot host a half-root block");

        // Free the remaining pieces; both roots must drain to empty.
        heap.deallocate(quarters[1]);
        heap.deallocate(quarters[3]);
        heap.deallocate(extra);
        heap.deallocate(half);
        PXG_CHECK_MSG(heap.totalBytesInUse() == 0, "all freed after fragmentation dance");
    }

    // --- deferred frees ---
    {
        PxgSlangHeapAllocator heap(dev, HEAP);
        const PxgSlangHeapAlloc a = heap.allocate(HEAP, 2); // occupies the whole root
        PXG_CHECK_MSG(a.valid(), "whole-root allocation succeeds");
        heap.deallocateDeferred(a);
        PXG_CHECK_MSG(heap.totalBytesInUse() == HEAP,
                      "deferred free keeps the allocation live until flush");
        const PxgSlangHeapAlloc b = heap.allocate(HEAP, 2);
        PXG_CHECK_MSG(b.valid() && heap.numRoots() == 2,
                      "pre-flush allocation cannot reuse the deferred space (grows instead)");
        heap.flushDeferredDeallocs();
        PXG_CHECK_MSG(heap.totalBytesInUse() == HEAP, "flush releases the deferred allocation");
        const PxgSlangHeapAlloc c = heap.allocate(HEAP, 2);
        PXG_CHECK_MSG(c.valid() && heap.numRoots() == 2,
                      "post-flush allocation reuses the released root");
        heap.deallocate(b);
        heap.deallocate(c);
    }

    // --- exceptional oversize allocations ---
    {
        PxgSlangHeapAllocator heap(dev, HEAP);
        const PxgSlangHeapAlloc big = heap.allocate(3 * HEAP, 3);
        PXG_CHECK_MSG(big.valid() && big.offset == 0 && big.size >= 3 * HEAP,
                      "oversize request gets a dedicated buffer");
        PXG_CHECK_MSG(heap.bytesInUse(3) == big.size, "exceptional alloc tracked in stats");

        // The exceptional buffer must be real, writable device memory.
        std::vector<uint32_t> pattern(1024);
        for (size_t i = 0; i < pattern.size(); ++i)
            pattern[i] = 0xbeef0000u + uint32_t(i);
        dev->upload(big.buffer, pattern.data(), pattern.size() * 4, big.size - pattern.size() * 4);
        std::vector<uint32_t> got(pattern.size(), 0);
        dev->download(big.buffer, got.data(), got.size() * 4, big.size - pattern.size() * 4);
        PXG_CHECK_MSG(got == pattern, "exceptional buffer round-trips at its far end");

        heap.deallocate(big);
        PXG_CHECK_MSG(heap.bytesInUse(3) == 0, "exceptional dealloc updates stats");
    }

    // --- device round-trip through a sub-allocated range ---
    {
        PxgSlangHeapAllocator heap(dev, HEAP);
        const PxgSlangHeapAlloc pad = heap.allocate(4096, 4);
        const PxgSlangHeapAlloc a = heap.allocate(8192, 4);
        std::vector<uint32_t> pattern(8192 / 4);
        for (size_t i = 0; i < pattern.size(); ++i)
            pattern[i] = uint32_t(i * 2246822519u);
        dev->upload(a.buffer, pattern.data(), a.size, a.offset);
        std::vector<uint32_t> got(pattern.size(), 0);
        dev->download(a.buffer, got.data(), a.size, a.offset);
        PXG_CHECK_MSG(got == pattern, "sub-allocated range round-trips bitwise");
        heap.deallocate(pad);
        heap.deallocate(a);
    }

    // --- randomized stress vs shadow interval model ---
    {
        PxgSlangHeapAllocator heap(dev, HEAP);
        std::mt19937 rng(1337);
        struct Shadow { PxgSlangHeapAlloc alloc; int group; };
        std::vector<Shadow> live;
        // per-buffer map of [offset, offset+size) intervals for overlap checks
        std::map<PxgBufferHandle, std::map<uint64_t, uint64_t>> intervals;

        size_t overlapErrors = 0, alignErrors = 0, failedAllocs = 0;
        uint64_t shadowBytes = 0;

        for (int op = 0; op < 4000; ++op)
        {
            const bool doAlloc = live.empty() || (rng() % 100) < 60;
            if (doAlloc)
            {
                const uint64_t bytes = 1 + rng() % (256 * 1024);
                const int group = int(rng() % 4);
                const PxgSlangHeapAlloc a = heap.allocate(bytes, group);
                if (!a.valid())
                {
                    ++failedAllocs;
                    continue;
                }
                if (a.offset % 128 != 0)
                    ++alignErrors;
                auto& perBuf = intervals[a.buffer];
                // overlap iff a prior interval starts before our end and ends after our start
                auto next = perBuf.lower_bound(a.offset);
                if (next != perBuf.end() && next->first < a.offset + a.size)
                    ++overlapErrors;
                if (next != perBuf.begin())
                {
                    auto prev = std::prev(next);
                    if (prev->first + prev->second > a.offset)
                        ++overlapErrors;
                }
                perBuf[a.offset] = a.size;
                live.push_back(Shadow{ a, group });
                shadowBytes += a.size;
            }
            else
            {
                const size_t pick = rng() % live.size();
                const Shadow s = live[pick];
                live[pick] = live.back();
                live.pop_back();
                intervals[s.alloc.buffer].erase(s.alloc.offset);
                heap.deallocate(s.alloc);
                shadowBytes -= s.alloc.size;
            }
            if (heap.totalBytesInUse() < shadowBytes)
                break; // stats must never undercount actual usable bytes
        }

        PXG_CHECK_MSG(overlapErrors == 0, "stress: no overlapping live allocations (%zu)", overlapErrors);
        PXG_CHECK_MSG(alignErrors == 0, "stress: all offsets 128 B aligned (%zu)", alignErrors);
        PXG_CHECK_MSG(failedAllocs == 0, "stress: no allocation failures (%zu)", failedAllocs);
        PXG_CHECK_MSG(heap.totalBytesInUse() >= shadowBytes,
                      "stress: stats cover at least the usable bytes");
        std::printf("  stress: %zu live at end, %u roots, %llu KiB in use, %llu KiB reserved\n",
                    live.size(), heap.numRoots(),
                    (unsigned long long)(heap.totalBytesInUse() / 1024),
                    (unsigned long long)(heap.deviceBytesReserved() / 1024));

        for (const Shadow& s : live)
            heap.deallocate(s.alloc);
        PXG_CHECK_MSG(heap.totalBytesInUse() == 0, "stress: full drain reaches zero");
        const PxgSlangHeapAlloc whole = heap.allocate(HEAP, 0);
        PXG_CHECK_MSG(whole.valid(), "stress: coalescing recovers a whole root after drain");
        heap.deallocate(whole);
    }
}

}  // namespace
