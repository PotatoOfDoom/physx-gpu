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

#ifndef PXG_SLANG_HEAP_ALLOCATOR_H
#define PXG_SLANG_HEAP_ALLOCATOR_H

// PxgSlangHeapAllocator (RUNTIME_PLAN.md P1.2)
//
// Buddy sub-allocator over PxgSlangComputeDevice buffers, reproducing the
// semantics of the CUDA runtime's PxgHeapMemoryAllocator:
//   - power-of-two size classes with a 128 B minimum (all returned offsets
//     are 128 B aligned, matching the CUDA heap's alignment contract);
//   - buddy coalescing on free, so drained heaps recover their largest class;
//   - "roots": the heap grows by whole device buffers of heapBytes each when
//     the existing roots cannot satisfy a request;
//   - exceptional allocations: requests larger than heapBytes get a dedicated
//     device buffer and bypass the buddy structure (freed on deallocate);
//   - deferred frees: deallocateDeferred() parks a free until
//     flushDeferredDeallocs() runs at a step boundary, mirroring how the CUDA
//     runtime keeps memory alive until in-flight work has retired;
//   - per-group statistics (the PxsHeapStats category tags).
//
// Host-side bookkeeping only: the device sees plain buffers; every allocation
// is addressed as (PxgBufferHandle, byte offset).

#include "PxgSlangComputeDevice.h"

#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace physx
{
namespace gpuslang
{

struct PxgSlangHeapAlloc
{
    PxgBufferHandle buffer = 0; // 0 = allocation failed
    uint64_t offset = 0;
    uint64_t size = 0;          // usable (rounded-up) size in bytes

    bool valid() const { return buffer != 0; }
};

class PxgSlangHeapAllocator
{
public:
    static const uint64_t kMinBlockBytes = 128; // CUDA heap alignment contract

    PxgSlangHeapAllocator(PxgSlangComputeDevice* device, uint64_t heapBytes)
        : mDevice(device), mHeapBytes(roundUpPow2(heapBytes < kMinBlockBytes ? kMinBlockBytes : heapBytes))
    {
        mNumLevels = log2u(mHeapBytes / kMinBlockBytes) + 1;
    }

    ~PxgSlangHeapAllocator()
    {
        flushDeferredDeallocs();
        for (Root& root : mRoots)
            mDevice->destroyBuffer(root.buffer);
        for (const auto& kv : mExceptional)
            mDevice->destroyBuffer(kv.first);
    }

    PxgSlangHeapAlloc allocate(uint64_t bytes, int group = 0)
    {
        if (bytes == 0)
            bytes = kMinBlockBytes;

        PxgSlangHeapAlloc out;

        // Exceptional path: bigger than a whole root.
        if (bytes > mHeapBytes)
        {
            const uint64_t size = roundUpPow2(bytes);
            const PxgBufferHandle buffer = mDevice->createBuffer(size_t(size));
            if (buffer == 0)
                return out;
            mExceptional[buffer] = ExceptionalAlloc{ size, group };
            addStats(group, size);
            out.buffer = buffer;
            out.offset = 0;
            out.size = size;
            return out;
        }

        const uint64_t size = bytes < kMinBlockBytes ? kMinBlockBytes : roundUpPow2(bytes);
        const uint32_t level = levelOf(size);

        for (size_t r = 0; r < mRoots.size(); ++r)
        {
            const uint64_t offset = allocateFromRoot(mRoots[r], level);
            if (offset != kNoSpace)
                return finishAlloc(uint32_t(r), offset, size, level, group);
        }

        // Grow by one root.
        Root root;
        root.buffer = mDevice->createBuffer(size_t(mHeapBytes));
        if (root.buffer == 0)
            return out;
        root.freePerLevel.resize(mNumLevels);
        root.freePerLevel[0].insert(0); // one block spanning the whole root
        mRoots.push_back(root);

        const uint64_t offset = allocateFromRoot(mRoots.back(), level);
        assert(offset != kNoSpace);
        return finishAlloc(uint32_t(mRoots.size() - 1), offset, size, level, group);
    }

    void deallocate(const PxgSlangHeapAlloc& alloc)
    {
        if (!alloc.valid())
            return;

        auto ex = mExceptional.find(alloc.buffer);
        if (ex != mExceptional.end())
        {
            subStats(ex->second.group, ex->second.size);
            mDevice->destroyBuffer(alloc.buffer);
            mExceptional.erase(ex);
            return;
        }

        const uint32_t rootIndex = rootOf(alloc.buffer);
        assert(rootIndex != uint32_t(-1) && "deallocate: unknown buffer");
        if (rootIndex == uint32_t(-1))
            return;

        const uint64_t key = allocKey(rootIndex, alloc.offset);
        auto it = mLive.find(key);
        assert(it != mLive.end() && "deallocate: unknown offset (double free?)");
        if (it == mLive.end())
            return;

        subStats(it->second.group, levelSize(it->second.level));
        freeIntoRoot(mRoots[rootIndex], alloc.offset, it->second.level);
        mLive.erase(it);
    }

    // Parks the free until flushDeferredDeallocs(); the allocation stays
    // occupied (and its device bytes untouched) until then.
    void deallocateDeferred(const PxgSlangHeapAlloc& alloc)
    {
        if (alloc.valid())
            mDeferred.push_back(alloc);
    }

    void flushDeferredDeallocs()
    {
        for (const PxgSlangHeapAlloc& alloc : mDeferred)
            deallocate(alloc);
        mDeferred.clear();
    }

    // ---- statistics ----
    uint64_t bytesInUse(int group) const
    {
        auto it = mGroupBytes.find(group);
        return it == mGroupBytes.end() ? 0 : it->second;
    }
    uint64_t totalBytesInUse() const { return mTotalBytes; }
    uint64_t peakBytesInUse() const { return mPeakBytes; }
    uint64_t deviceBytesReserved() const
    {
        uint64_t reserved = mRoots.size() * mHeapBytes;
        for (const auto& kv : mExceptional)
            reserved += kv.second.size;
        return reserved;
    }
    uint32_t numRoots() const { return uint32_t(mRoots.size()); }

private:
    static const uint64_t kNoSpace = ~uint64_t(0);

    struct Root
    {
        PxgBufferHandle buffer = 0;
        // freePerLevel[L] holds byte offsets of free blocks of size
        // mHeapBytes >> L (level 0 = the whole root).
        std::vector<std::unordered_set<uint64_t>> freePerLevel;
    };

    struct LiveAlloc
    {
        uint32_t level;
        int group;
    };

    struct ExceptionalAlloc
    {
        uint64_t size;
        int group;
    };

    static uint64_t roundUpPow2(uint64_t v)
    {
        if (v <= 1)
            return 1;
        --v;
        v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16; v |= v >> 32;
        return v + 1;
    }

    static uint32_t log2u(uint64_t v)
    {
        uint32_t r = 0;
        while (v > 1)
        {
            v >>= 1;
            ++r;
        }
        return r;
    }

    uint64_t levelSize(uint32_t level) const { return mHeapBytes >> level; }
    uint32_t levelOf(uint64_t sizePow2) const { return log2u(mHeapBytes / sizePow2); }
    static uint64_t allocKey(uint32_t rootIndex, uint64_t offset)
    {
        return (uint64_t(rootIndex) << 40) | offset; // offsets < heapBytes < 2^40
    }

    uint32_t rootOf(PxgBufferHandle buffer) const
    {
        for (size_t r = 0; r < mRoots.size(); ++r)
            if (mRoots[r].buffer == buffer)
                return uint32_t(r);
        return uint32_t(-1);
    }

    // Find (splitting as needed) a free block of exactly `level`; kNoSpace if
    // the root cannot satisfy it.
    uint64_t allocateFromRoot(Root& root, uint32_t level)
    {
        // Find the deepest available level at or above the request.
        uint32_t from = level + 1;
        while (from > 0 && root.freePerLevel[from - 1].empty())
            --from;
        if (from == 0)
            return kNoSpace;
        --from; // level with a free block, from <= level

        uint64_t offset = *root.freePerLevel[from].begin();
        root.freePerLevel[from].erase(root.freePerLevel[from].begin());

        // Split down to the requested level, freeing the upper buddy halves.
        for (uint32_t l = from; l < level; ++l)
        {
            const uint64_t half = levelSize(l + 1);
            root.freePerLevel[l + 1].insert(offset + half);
        }
        return offset;
    }

    void freeIntoRoot(Root& root, uint64_t offset, uint32_t level)
    {
        // Coalesce with the buddy while possible.
        while (level > 0)
        {
            const uint64_t size = levelSize(level);
            const uint64_t buddy = offset ^ size;
            auto& freeSet = root.freePerLevel[level];
            auto it = freeSet.find(buddy);
            if (it == freeSet.end())
                break;
            freeSet.erase(it);
            offset = offset < buddy ? offset : buddy;
            --level;
        }
        root.freePerLevel[level].insert(offset);
    }

    PxgSlangHeapAlloc finishAlloc(uint32_t rootIndex, uint64_t offset, uint64_t size, uint32_t level, int group)
    {
        mLive[allocKey(rootIndex, offset)] = LiveAlloc{ level, group };
        addStats(group, levelSize(level));

        PxgSlangHeapAlloc out;
        out.buffer = mRoots[rootIndex].buffer;
        out.offset = offset;
        out.size = size;
        return out;
    }

    void addStats(int group, uint64_t bytes)
    {
        mGroupBytes[group] += bytes;
        mTotalBytes += bytes;
        if (mTotalBytes > mPeakBytes)
            mPeakBytes = mTotalBytes;
    }

    void subStats(int group, uint64_t bytes)
    {
        assert(mGroupBytes[group] >= bytes && mTotalBytes >= bytes);
        mGroupBytes[group] -= bytes;
        mTotalBytes -= bytes;
    }

    PxgSlangComputeDevice* mDevice;
    uint64_t mHeapBytes;
    uint32_t mNumLevels = 0;

    std::vector<Root> mRoots;
    std::unordered_map<uint64_t, LiveAlloc> mLive;
    std::unordered_map<PxgBufferHandle, ExceptionalAlloc> mExceptional;
    std::vector<PxgSlangHeapAlloc> mDeferred;

    std::unordered_map<int, uint64_t> mGroupBytes;
    uint64_t mTotalBytes = 0;
    uint64_t mPeakBytes = 0;
};

} // namespace gpuslang
} // namespace physx

#endif // PXG_SLANG_HEAP_ALLOCATOR_H
