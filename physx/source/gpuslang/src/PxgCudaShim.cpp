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

#include "PxgCudaShim.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace physx
{
namespace gpuslang
{

PxgCudaShim::PxgCudaShim(PxgSlangComputeDevice* device, const PxgKernelDef* manifest,
                         uint32_t manifestCount, PxgSpirvLoader loader)
    : mDevice(device), mManifest(manifest), mManifestCount(manifestCount), mLoader(loader)
{
}

PxgCudaShim::~PxgCudaShim()
{
    for (auto& kv : mPipelines)
        if (kv.second)
            mDevice->destroyPipeline(kv.second);
    for (auto& kv : mAllocs)
        mDevice->destroyBuffer(kv.second.buffer);
}

// ---- memory ----

PxgResult PxgCudaShim::memAlloc(PxgDevicePtr* out, size_t bytes)
{
    if (!out || bytes == 0)
        return PxgResult::InvalidValue;
    const PxgBufferHandle buf = mDevice->createBuffer(bytes);
    if (buf == 0)
        return PxgResult::OutOfMemory;
    const PxgDevicePtr base = mDevice->getBufferDeviceAddress(buf);
    if (base == 0)
    {
        // No BDA on this device: the shim's flat-address model is unavailable.
        // The binding-table fallback (RUNTIME_PLAN.md P3.2) would take over
        // here; not exercised on BDA-capable devices/CI.
        mDevice->destroyBuffer(buf);
        return PxgResult::InvalidValue;
    }
    mAllocs[base] = Alloc{ buf, base, bytes };
    *out = base;
    return PxgResult::Success;
}

PxgResult PxgCudaShim::memFree(PxgDevicePtr ptr)
{
    auto it = mAllocs.find(ptr);
    if (it == mAllocs.end())
        return PxgResult::NotFound; // CUDA requires the base pointer, not an interior one
    mDevice->destroyBuffer(it->second.buffer);
    mAllocs.erase(it);
    return PxgResult::Success;
}

// ---- pinned host memory (P5 Step 4b) ----

PxgResult PxgCudaShim::memHostAlloc(void** out, size_t bytes)
{
    if (!out)
        return PxgResult::InvalidValue;
    if (bytes == 0)
        bytes = 1;
    void* mapped = nullptr;
    const PxgBufferHandle buf = mDevice->createHostVisibleBuffer(bytes, &mapped);
    if (buf == 0 || mapped == nullptr)
        return PxgResult::OutOfMemory;
    const PxgDevicePtr base = mDevice->getBufferDeviceAddress(buf);
    if (base == 0)
    {
        mDevice->destroyBuffer(buf);
        return PxgResult::InvalidValue; // no BDA: pinned-mapped model unavailable
    }
    mAllocs[base] = Alloc{ buf, base, bytes };
    mHostAllocs[uintptr_t(mapped)] = HostAlloc{ base, bytes };
    *out = mapped;
    return PxgResult::Success;
}

PxgResult PxgCudaShim::memFreeHost(void* p)
{
    auto it = mHostAllocs.find(uintptr_t(p));
    if (it == mHostAllocs.end())
        return PxgResult::NotFound; // base pointer required, like memFree
    const PxgDevicePtr devBase = it->second.devBase;
    mHostAllocs.erase(it);
    auto ait = mAllocs.find(devBase);
    if (ait != mAllocs.end())
    {
        mDevice->destroyBuffer(ait->second.buffer);
        mAllocs.erase(ait);
    }
    return PxgResult::Success;
}

PxgDevicePtr PxgCudaShim::hostGetDevicePointer(const void* p) const
{
    const uintptr_t addr = uintptr_t(p);
    if (addr == 0 || mHostAllocs.empty())
        return 0;
    auto it = mHostAllocs.upper_bound(addr);
    if (it == mHostAllocs.begin())
        return 0;
    --it;
    const uintptr_t base = it->first;
    const HostAlloc& a = it->second;
    if (addr < base || addr >= base + a.size)
        return 0;
    return a.devBase + (addr - base);
}

bool PxgCudaShim::resolve(PxgDevicePtr ptr, PxgBufferHandle* buffer, size_t* offset) const
{
    if (ptr == 0 || mAllocs.empty())
        return false;
    // First allocation whose base is > ptr; the candidate is its predecessor.
    auto it = mAllocs.upper_bound(ptr);
    if (it == mAllocs.begin())
        return false;
    --it;
    const Alloc& a = it->second;
    if (ptr < a.base || ptr >= a.base + a.size)
        return false;
    if (buffer)
        *buffer = a.buffer;
    if (offset)
        *offset = size_t(ptr - a.base);
    return true;
}

PxgResult PxgCudaShim::memcpyHtoD(PxgDevicePtr dst, const void* src, size_t bytes)
{
    PxgBufferHandle buf;
    size_t off;
    if (!resolve(dst, &buf, &off))
        return PxgResult::InvalidValue;
    mDevice->upload(buf, src, bytes, off);
    return PxgResult::Success;
}

PxgResult PxgCudaShim::memcpyDtoH(void* dst, PxgDevicePtr src, size_t bytes)
{
    PxgBufferHandle buf;
    size_t off;
    if (!resolve(src, &buf, &off))
        return PxgResult::InvalidValue;
    mDevice->download(buf, dst, bytes, off);
    return PxgResult::Success;
}

PxgResult PxgCudaShim::memcpyDtoD(PxgDevicePtr dst, PxgDevicePtr src, size_t bytes)
{
    PxgBufferHandle sBuf, dBuf;
    size_t sOff, dOff;
    if (!resolve(src, &sBuf, &sOff) || !resolve(dst, &dBuf, &dOff))
        return PxgResult::InvalidValue;
    // Device-side copy (vkCmdCopyBuffer), stream-ordered; the base class
    // stages through host on backends without a device path.
    mDevice->copyBuffer(dBuf, dOff, sBuf, sOff, bytes);
    return PxgResult::Success;
}

PxgResult PxgCudaShim::memsetD32(PxgDevicePtr dst, uint32_t value, size_t numWords)
{
    PxgBufferHandle buf;
    size_t off;
    if (!resolve(dst, &buf, &off))
        return PxgResult::InvalidValue;
    // Device-side fill (vkCmdFillBuffer), stream-ordered.
    mDevice->fillBuffer(buf, off, value, numWords * sizeof(uint32_t));
    return PxgResult::Success;
}

// ---- launch ----

PxgPipelineHandle PxgCudaShim::pipelineFor(PxgKernelId id)
{
    auto it = mPipelines.find(id);
    if (it != mPipelines.end())
        return it->second;

    const PxgKernelDef& def = mManifest[id];
    const std::vector<uint8_t> blob = mLoader(def.spirvFile);
    PxgPipelineHandle pipe = 0;
    if (def.numTextures > 0)
        pipe = mDevice->createPipelineTex(blob.data(), blob.size(), def.entry, 0, nullptr,
                                          def.pushConstantSize, def.numTextures);
    else
        // Zero storage-buffer bindings: every buffer argument travels as a
        // device address inside the parameter blob (BDA model).
        pipe = mDevice->createPipeline(blob.data(), blob.size(), def.entry, 0, nullptr,
                                       def.pushConstantSize);
    mPipelines[id] = pipe;
    return pipe;
}

PxgResult PxgCudaShim::launchKernel(PxgKernelId id, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                                    const void* paramBlob, uint32_t paramBlobSize,
                                    const PxgTextureHandle* textures, uint32_t numTextures)
{
    if (id >= mManifestCount)
        return PxgResult::NotFound;
    if (paramBlobSize != mManifest[id].pushConstantSize || numTextures != mManifest[id].numTextures)
        return PxgResult::InvalidValue;

    const PxgPipelineHandle pipe = pipelineFor(id);
    if (pipe == 0)
        return PxgResult::NotFound;

    // PXG_LAUNCH_TRACE=1: name every dispatch on stderr BEFORE it is recorded.  With
    // PXG_SYNC_EACH=1 alongside, the last line printed is the kernel that faulted --
    // which is the only way to attribute a shader-side fault at all, because a
    // SIGSEGV inside llvmpipe's JIT arrives with no symbols and a garbage backtrace.
    // P9.52b-6 needed exactly this and had to add it; keeping it is cheaper than
    // rediscovering the need.
    static const bool trace = std::getenv("PXG_LAUNCH_TRACE") != nullptr;
    if (trace)
    {
        std::fprintf(stderr, "[pxg-launch] %-40s grid=(%u,%u,%u) blob=%u\n",
                     mManifest[id].cudaName ? mManifest[id].cudaName : (mManifest[id].entry ? mManifest[id].entry : "?"),
                     gridX, gridY, gridZ, paramBlobSize);
        std::fflush(stderr);
    }

    if (numTextures > 0)
        mDevice->dispatchTex(pipe, gridX, gridY, gridZ, nullptr, 0, textures, numTextures,
                             paramBlob, paramBlobSize);
    else
        mDevice->dispatch(pipe, gridX, gridY, gridZ, nullptr, 0, paramBlob, paramBlobSize);

    // PXG_SYNC_EACH=1: drain the queue after every dispatch so a shader-side
    // fault (e.g. a stray BDA dereference) is attributed to the exact kernel
    // that produced it — the launch trace's last line becomes the culprit
    // instead of a whole flushed batch.  Debug-only; off by default.
    static const bool syncEach = std::getenv("PXG_SYNC_EACH") != nullptr;
    if (syncEach)
        mDevice->flush();
    return PxgResult::Success;
}

// ---- streams & events ----

PxgResult PxgCudaShim::streamCreate(PxgStreamId* out)
{
    if (!out)
        return PxgResult::InvalidValue;
    const PxgStreamId id = mNextStream++;
    mStreams[id] = true;
    *out = id;
    return PxgResult::Success;
}

PxgResult PxgCudaShim::streamDestroy(PxgStreamId stream)
{
    auto it = mStreams.find(stream);
    if (it == mStreams.end())
        return PxgResult::NotFound;
    mStreams.erase(it);
    return PxgResult::Success;
}

PxgResult PxgCudaShim::streamSynchronize(PxgStreamId /*stream*/)
{
    // One queue: syncing any stream drains all recorded + submitted work.
    mDevice->flush();
    return PxgResult::Success;
}

PxgResult PxgCudaShim::eventCreate(uint32_t* out)
{
    if (!out)
        return PxgResult::InvalidValue;
    const uint32_t id = mNextEvent++;
    mEvents[id] = 0;
    *out = id;
    return PxgResult::Success;
}

PxgResult PxgCudaShim::eventDestroy(uint32_t event)
{
    auto it = mEvents.find(event);
    if (it == mEvents.end())
        return PxgResult::NotFound;
    mEvents.erase(it);
    return PxgResult::Success;
}

PxgResult PxgCudaShim::eventRecord(uint32_t event, PxgStreamId /*stream*/)
{
    auto it = mEvents.find(event);
    if (it == mEvents.end())
        return PxgResult::NotFound;
    // Enqueue a marker after all prior work: submit the recorded batch
    // (non-blocking) and capture its ticket.  submit() with nothing recorded
    // returns the most recent ticket, so back-to-back records are harmless.
    const uint64_t ticket = mDevice->submit();
    it->second = ticket;
    mLastLaunchTicket = ticket;
    return PxgResult::Success;
}

PxgResult PxgCudaShim::eventSynchronize(uint32_t event)
{
    auto it = mEvents.find(event);
    if (it == mEvents.end())
        return PxgResult::NotFound;
    mDevice->wait(it->second);
    return PxgResult::Success;
}

PxgResult PxgCudaShim::eventQuery(uint32_t event)
{
    auto it = mEvents.find(event);
    if (it == mEvents.end())
        return PxgResult::NotFound;
    return mDevice->poll(it->second) ? PxgResult::Success : PxgResult::InvalidValue;
}

PxgResult PxgCudaShim::streamWaitEvent(PxgStreamId stream, uint32_t event)
{
    if (stream != 0 && mStreams.find(stream) == mStreams.end())
        return PxgResult::NotFound;
    auto it = mEvents.find(event);
    if (it == mEvents.end())
        return PxgResult::NotFound;
    // The event's captured work was already submitted at record time; on the
    // single serialized queue every later submission waits on the previous
    // ticket (submit()'s stream ordering), so the cross-stream dependency is
    // already satisfied — this validates the handles and is otherwise a
    // no-op.  A real cross-queue wait lands with P6 async compute.
    return PxgResult::Success;
}

// ---- 3D textures (P9.12b) ----
//
// The split between "the array" and "the texture object" is CUDA's, and it is
// kept rather than collapsed because the ENGINE's call sequence has that shape:
// PxgGeometryManager.cpp:733-769 creates the array, copies into it, and only
// then builds the object.  Collapsing the three into one entry point would
// force the driver shim to guess which of them the engine is in the middle of.

PxgResult PxgCudaShim::arrayCreate3D(uint32_t* outArray, uint32_t width, uint32_t height,
                                     uint32_t depth, PxgSlangComputeDevice::TextureFormat format)
{
    if (!outArray || width == 0 || height == 0 || depth == 0)
        return PxgResult::InvalidValue;
    if (!mDevice->supportsBindlessTextures())
        return PxgResult::Unsupported;

    size_t texel = 4;
    switch (format)
    {
    case PxgSlangComputeDevice::TextureFormat::R8Unorm:  texel = 1; break;
    case PxgSlangComputeDevice::TextureFormat::R16Unorm: texel = 2; break;
    default: break;
    }

    const uint32_t id = mNextArrayId++;
    CudaArray a;
    a.width = width;
    a.height = height;
    a.depth = depth;
    a.format = format;
    // Zero-filled: CUDA leaves a fresh array undefined, but a texture object
    // built over an array nobody copied into would otherwise sample whatever
    // the heap held.  Zero is a value the caller can recognise; garbage is not.
    a.host.assign(size_t(width) * height * depth * texel, 0u);
    mArrays[id] = std::move(a);
    *outArray = id;
    return PxgResult::Success;
}

PxgResult PxgCudaShim::arrayUpload(uint32_t arrayId, const void* src, size_t bytes)
{
    auto it = mArrays.find(arrayId);
    if (it == mArrays.end())
        return PxgResult::NotFound;
    if (!src || bytes != it->second.host.size())
        return PxgResult::InvalidValue;
    // A copy after the image exists would never reach the GPU.  Refusing is the
    // only honest answer: the alternative is a texture that silently holds the
    // previous contents.
    if (it->second.consumed)
        return PxgResult::InvalidValue;
    std::memcpy(it->second.host.data(), src, bytes);
    it->second.uploaded = true;
    return PxgResult::Success;
}

PxgResult PxgCudaShim::textureObjectCreate(PxgTextureHandle* out, uint32_t arrayId)
{
    if (!out)
        return PxgResult::InvalidValue;
    auto it = mArrays.find(arrayId);
    if (it == mArrays.end())
        return PxgResult::NotFound;
    CudaArray& a = it->second;
    const PxgTextureHandle h = mDevice->createTexture3DFmt(a.width, a.height, a.depth, a.format,
                                                           a.host.data());
    if (h == 0)
        return PxgResult::Unsupported;
    a.consumed = true;
    // The bytes are on the GPU now; the staging copy is dead weight.  The record
    // itself stays so a later cuArrayDestroy still resolves.
    a.host.clear();
    a.host.shrink_to_fit();
    *out = h;
    return PxgResult::Success;
}

PxgResult PxgCudaShim::arrayDestroy(uint32_t arrayId)
{
    auto it = mArrays.find(arrayId);
    if (it == mArrays.end())
        return PxgResult::NotFound;
    // Destroying the array does NOT destroy the texture: in CUDA the texture
    // object keeps the array alive, and the engine relies on that -- it drops
    // the CUarray and goes on sampling.  Here the image is owned by the device
    // and outlives this record by construction.
    mArrays.erase(it);
    return PxgResult::Success;
}

PxgResult PxgCudaShim::textureObjectDestroy(PxgTextureHandle handle)
{
    if (handle == 0)
        return PxgResult::Success;   // the null texture, as CUDA treats it
    mDevice->destroyTexture3D(handle);
    return PxgResult::Success;
}

} // namespace gpuslang
} // namespace physx
