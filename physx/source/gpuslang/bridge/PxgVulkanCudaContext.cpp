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

#include "PxgVulkanCudaContext.h"
#include "PxgVulkanKernelAbi.h"

#if PX_SUPPORT_GPU_PHYSX

#include <cstdlib>
#include <cstring>
#include <vector>

namespace physx
{
namespace gpuslang
{

// Defined in bridge/PxgVulkanDriverShim.cpp: registers the live context so the raw
// (context-less) CUDA-driver shim entry points can route memcpys through the bridge.
void pxgSetVulkanDriverContext(physx::PxCudaContext* ctx);

namespace
{
// ---- handle <-> shim id encodings (see header) ----
inline CUstream toStream(PxgStreamId id) { return reinterpret_cast<CUstream>(uintptr_t(id)); }
inline PxgStreamId fromStream(CUstream s) { return PxgStreamId(reinterpret_cast<uintptr_t>(s)); }
inline CUevent toEvent(uint32_t id) { return reinterpret_cast<CUevent>(uintptr_t(id)); }
inline uint32_t fromEvent(CUevent e) { return uint32_t(reinterpret_cast<uintptr_t>(e)); }
inline CUfunction toFunc(PxgKernelId id) { return reinterpret_cast<CUfunction>(uintptr_t(id) + 1u); }
inline bool fromFunc(CUfunction f, PxgKernelId* id)
{
    const uintptr_t v = reinterpret_cast<uintptr_t>(f);
    if (v == 0)
        return false;
    *id = PxgKernelId(v - 1u);
    return true;
}
// A single non-null sentinel: kernels are precompiled, so "loading a module"
// is a no-op that just needs a valid-looking handle.
inline CUmodule sentinelModule() { return reinterpret_cast<CUmodule>(uintptr_t(1)); }

// CUDA result codes we surface (values from cuda.h's CUresult).
enum { PXG_CUDA_SUCCESS = 0, PXG_CUDA_ERROR_INVALID_VALUE = 1, PXG_CUDA_ERROR_OUT_OF_MEMORY = 2 };

inline PxCUresult toResult(PxgResult r)
{
    PxU32 v = PXG_CUDA_SUCCESS;
    if (r == PxgResult::OutOfMemory)
        v = PXG_CUDA_ERROR_OUT_OF_MEMORY;
    else if (r != PxgResult::Success)
        v = PXG_CUDA_ERROR_INVALID_VALUE;
    return PxCUresult(v);
}
inline PxCUresult okResult() { return PxCUresult(PxU32(PXG_CUDA_SUCCESS)); }
inline PxCUresult invalidResult() { return PxCUresult(PxU32(PXG_CUDA_ERROR_INVALID_VALUE)); }
inline PxCUresult oomResult() { return PxCUresult(PxU32(PXG_CUDA_ERROR_OUT_OF_MEMORY)); }

// CUDA lays each kernel parameter out in the .param bank at its type's natural
// alignment, which the driver reads from the compiled function signature
// (cuLaunchKernel; see CudaContextManager.cpp:1165-1179).  The bridge only has
// the argument size, from which the alignment is exact for scalars, pointers,
// and power-of-two vector types — the argument kinds PhysX GPU launch sites
// use.  (A by-value struct whose size is not a power of two — e.g. a 12-byte
// float3 — would need its Slang push-constant field authored to match; PhysX
// passes such aggregates by device pointer, not by value.)
inline size_t paramAlign(size_t size)
{
    if (size >= 16) return 16;
    if (size >= 8)  return 8;
    if (size >= 4)  return 4;
    if (size >= 2)  return 2;
    return 1;
}

class PxgVulkanCudaContext : public physx::PxCudaContext
{
public:
    PxgVulkanCudaContext(PxgCudaShim* shim, const PxgKernelDef* manifest, uint32_t manifestCount)
        : mShim(shim), mManifest(manifest), mManifestCount(manifestCount)
    {
        mAllocatorCallback = nullptr;
        // Register as the live context so the raw (context-less) CUDA-driver shim
        // entry points (bridge/PxgVulkanDriverShim.cpp) route memcpys through us.
        pxgSetVulkanDriverContext(this);
    }

    void release() override { pxgSetVulkanDriverContext(nullptr); delete this; }

    // The backing shim (bridge-internal): the ccm's getMappedDevicePtr resolves
    // pinned host pointers through it.
    PxgCudaShim* shim() const { return mShim; }

    // ---- memory ----
    PxCUresult memAlloc(CUdeviceptr* dptr, size_t bytesize) override
    {
        if (mAbort) return mLastResult;
        PxgDevicePtr p = 0;
        const PxgResult r = mShim->memAlloc(&p, bytesize);
        if (dptr)
            *dptr = CUdeviceptr(p);
        return record(r);
    }
    PxCUresult memFree(CUdeviceptr dptr) override
    {
        if (mAbort) return mLastResult;
        return record(mShim->memFree(PxgDevicePtr(dptr)));
    }

    PxCUresult memHostAlloc(void** pp, size_t bytesize, unsigned int /*Flags*/) override
    {
        // P5 Step 4b: real pinned-mapped memory — a persistently mapped host-
        // visible buffer with a device address, so the engine's mapped-pointer
        // model (PxPinnedArray + getMappedDevicePtr, the copy-descriptor queue,
        // bpDesc mirrors, ...) works as it does on CUDA.
        if (mAbort) return mLastResult;
        return record(mShim->memHostAlloc(pp, bytesize));
    }
    PxCUresult memFreeHost(void* p) override
    {
        if (mAbort) return mLastResult;
        return record(mShim->memFreeHost(p));
    }
    PxCUresult memHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int /*Flags*/) override
    {
        // Interior CPU pointers of pinned allocations resolve to their device
        // address (CUDA parity).  Unknown host memory yields the null result.
        // Does NOT touch mLastResult (an unsupported query — even mid-abort —
        // never clobbers the sticky abort/OOM state).
        const PxgDevicePtr d = mShim->hostGetDevicePointer(p);
        if (pdptr)
            *pdptr = CUdeviceptr(d);
        return d != 0 ? okResult() : (p ? invalidResult() : okResult());
    }

    // ---- modules / functions ----
    PxCUresult moduleLoadDataEx(CUmodule* module, const void* /*image*/, unsigned int /*numOptions*/,
                                PxCUjit_option* /*options*/, void** /*optionValues*/) override
    {
        if (mAbort) return mLastResult;
        if (module)
            *module = sentinelModule();
        return record(PxgResult::Success);
    }
    PxCUresult moduleGetFunction(CUfunction* hfunc, CUmodule /*hmod*/, const char* name) override
    {
        if (mAbort) return mLastResult;
        for (uint32_t i = 0; i < mManifestCount; ++i)
        {
            // Require a compiled module (spirvFile), not just a reserved entry
            // name, so name resolution agrees with the wrangler's launchability
            // gate (PxgVulkanKernelWrangler::isMapped): a name is only handed a
            // launchable handle if the kernel is actually mapped to SPIR-V.
            if (mManifest[i].spirvFile && mManifest[i].entry && name &&
                std::strcmp(mManifest[i].entry, name) == 0)
            {
                if (hfunc)
                    *hfunc = toFunc(i);
                return record(PxgResult::Success);
            }
        }
        return record(PxgResult::NotFound);
    }
    PxCUresult moduleUnload(CUmodule /*hmod*/) override { return record(PxgResult::Success); }

    // ---- streams ----
    PxCUresult streamCreate(CUstream* phStream, unsigned int /*Flags*/) override
    {
        if (mAbort) return mLastResult;
        PxgStreamId id = 0;
        const PxgResult r = mShim->streamCreate(&id);
        if (phStream)
            *phStream = toStream(id);
        return record(r);
    }
    PxCUresult streamCreateWithPriority(CUstream* phStream, unsigned int flags, int /*priority*/) override
    {
        return streamCreate(phStream, flags);
    }
    PxCUresult streamFlush(CUstream hStream) override
    {
        if (mAbort) return mLastResult;
        return record(mShim->streamSynchronize(fromStream(hStream)));
    }
    PxCUresult streamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int /*Flags*/) override
    {
        if (mAbort) return mLastResult;
        return record(mShim->streamWaitEvent(fromStream(hStream), fromEvent(hEvent)));
    }
    PxCUresult streamWaitEvent(CUstream hStream, CUevent hEvent) override
    {
        if (mAbort) return mLastResult;
        return record(mShim->streamWaitEvent(fromStream(hStream), fromEvent(hEvent)));
    }
    PxCUresult streamDestroy(CUstream hStream) override
    {
        if (mAbort) return mLastResult;
        return record(mShim->streamDestroy(fromStream(hStream)));
    }
    PxCUresult streamSynchronize(CUstream hStream) override
    {
        if (mAbort) return mLastResult;
        return record(mShim->streamSynchronize(fromStream(hStream)));
    }

    // ---- events ----
    PxCUresult eventCreate(CUevent* phEvent, unsigned int /*Flags*/) override
    {
        if (mAbort) return mLastResult;
        uint32_t id = 0;
        const PxgResult r = mShim->eventCreate(&id);
        if (phEvent)
            *phEvent = toEvent(id);
        return record(r);
    }
    PxCUresult eventRecord(CUevent hEvent, CUstream hStream) override
    {
        if (mAbort) return mLastResult;
        return record(mShim->eventRecord(fromEvent(hEvent), fromStream(hStream)));
    }
    PxCUresult eventQuery(CUevent hEvent) override
    {
        if (mAbort) return mLastResult;
        return record(mShim->eventQuery(fromEvent(hEvent)));
    }
    PxCUresult eventSynchronize(CUevent hEvent) override
    {
        if (mAbort) return mLastResult;
        return record(mShim->eventSynchronize(fromEvent(hEvent)));
    }
    PxCUresult eventDestroy(CUevent hEvent) override
    {
        if (mAbort) return mLastResult;
        return record(mShim->eventDestroy(fromEvent(hEvent)));
    }

    // ---- launch ----
    PxCUresult launchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                            unsigned int /*blockDimX*/, unsigned int /*blockDimY*/, unsigned int /*blockDimZ*/,
                            unsigned int /*sharedMemBytes*/, CUstream /*hStream*/, PxCudaKernelParam* kernelParams,
                            size_t kernelParamsSizeInBytes, void** /*extra*/, const char* /*file*/,
                            int /*line*/) override
    {
        if (mAbort) return mLastResult;
        PxgKernelId id = 0;
        if (!fromFunc(f, &id))
            return record(PxgResult::InvalidValue);

        // CUDA hands each argument to the driver as its own pointer; the driver
        // lays each out in the .param bank at the NATURAL alignment declared by
        // the compiled function signature (CudaContextManager.cpp:1165-1179).
        // The Slang kernel takes ONE push-constant blob whose struct fields use
        // that same natural alignment, so reproduce CUDA's layout here: place
        // each argument at its type alignment (paramAlign, reconstructed from
        // the PxCudaKernelParam.size field).  Tight concatenation would misplace
        // any argument that follows a smaller one (e.g. a device pointer after a
        // PxU32), handing the kernel a shifted blob and a garbage buffer-device
        // address.  For the single-descriptor-struct kernel (count == 1) the
        // first argument starts at offset 0, so the blob is an identity copy.
        const uint32_t count = uint32_t(kernelParamsSizeInBytes / sizeof(PxCudaKernelParam));
        std::vector<uint8_t> blob;
        size_t maxAlign = 1;
        for (uint32_t i = 0; i < count; ++i)
        {
            const size_t a = paramAlign(kernelParams[i].size);
            if (a > maxAlign) maxAlign = a;
            blob.resize((blob.size() + (a - 1)) & ~(a - 1));   // pad to natural alignment
            const uint8_t* d = static_cast<const uint8_t*>(kernelParams[i].data);
            blob.insert(blob.end(), d, d + kernelParams[i].size);
        }
        // Trailing struct padding: a compiled push-constant struct's size is rounded
        // up to its largest member's alignment (e.g. { uint64 ptr; uint count; } is 16,
        // not 12).  The Slang side sees exactly that struct, so round the blob to match,
        // otherwise the shim's blobSize == pushConstantSize gate rejects the launch.
        blob.resize((blob.size() + (maxAlign - 1)) & ~(maxAlign - 1));
        return record(mShim->launchKernel(id, gridDimX, gridDimY, gridDimZ, blob.data(),
                                          uint32_t(blob.size())));
    }
    PxCUresult launchKernel(CUfunction f, PxU32 gridDimX, PxU32 gridDimY, PxU32 gridDimZ,
                            PxU32 /*blockDimX*/, PxU32 /*blockDimY*/, PxU32 /*blockDimZ*/, PxU32 /*sharedMemBytes*/,
                            CUstream /*hStream*/, void** kernelParams, void** /*extra*/, const char* /*file*/,
                            int /*line*/) override
    {
        // The void** launch form (PX_CUDA_KERNEL_PARAM2; USE_NEW_LAUNCH_FUNCTION
        // in broadphase/AABB-manager/articulation) carries no per-argument sizes
        // — CUDA reads them from the compiled kernel's parameter metadata.  The
        // bridge reads them from the per-kernel launch-ABI table instead
        // (PxgVulkanKernelAbi.cpp) and assembles the same natural-alignment blob
        // as the sized form.  A kernel without an ABI row is refused cleanly.
        if (mAbort) return mLastResult;
        PxgKernelId id = 0;
        if (!fromFunc(f, &id))
            return record(PxgResult::InvalidValue);
        const PxgKernelAbiRow* abi = id < mManifestCount ? pxgFindKernelAbi(mManifest[id].cudaName) : nullptr;
        if (!abi || !kernelParams)
            return record(PxgResult::InvalidValue);

        std::vector<uint8_t> blob;
        size_t maxAlign = 1;
        for (uint32_t i = 0; i < abi->argCount; ++i)
        {
            const size_t a = paramAlign(abi->argSizes[i]);
            if (a > maxAlign) maxAlign = a;
            blob.resize((blob.size() + (a - 1)) & ~(a - 1));
            const uint8_t* d = static_cast<const uint8_t*>(kernelParams[i]);
            blob.insert(blob.end(), d, d + abi->argSizes[i]);
        }
        blob.resize((blob.size() + (maxAlign - 1)) & ~(maxAlign - 1));
        return record(mShim->launchKernel(id, gridDimX, gridDimY, gridDimZ, blob.data(),
                                          uint32_t(blob.size())));
    }

    // ---- copies (async == sync: P3 serializes on one queue) ----
    PxCUresult memcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) override
    {
        if (mAbort) return mLastResult;
        return record(mShim->memcpyDtoH(dstHost, PxgDevicePtr(srcDevice), ByteCount));
    }
    PxCUresult memcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream /*hStream*/) override
    {
        if (mAbort) return mLastResult;
        return record(mShim->memcpyDtoH(dstHost, PxgDevicePtr(srcDevice), ByteCount));
    }
    PxCUresult memcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) override
    {
        if (mAbort) return mLastResult;
        return record(mShim->memcpyHtoD(PxgDevicePtr(dstDevice), srcHost, ByteCount));
    }
    PxCUresult memcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream /*hStream*/) override
    {
        if (mAbort) return mLastResult;
        return record(mShim->memcpyHtoD(PxgDevicePtr(dstDevice), srcHost, ByteCount));
    }
    PxCUresult memcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream /*hStream*/) override
    {
        if (mAbort) return mLastResult;
        return record(mShim->memcpyDtoD(PxgDevicePtr(dstDevice), PxgDevicePtr(srcDevice), ByteCount));
    }
    PxCUresult memcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) override
    {
        if (mAbort) return mLastResult;
        return record(mShim->memcpyDtoD(PxgDevicePtr(dstDevice), PxgDevicePtr(srcDevice), ByteCount));
    }
    PxCUresult memcpyPeerAsync(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream) override
    {
        if (mAbort) return mLastResult;
        return record(PxgResult::InvalidValue); // single device: no peer copies
    }

    // ---- memset ----
    PxCUresult memsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream /*hStream*/) override
    {
        if (mAbort) return mLastResult;
        return record(mShim->memsetD32(PxgDevicePtr(dstDevice), ui, N));
    }
    PxCUresult memsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) override
    {
        if (mAbort) return mLastResult;
        return record(mShim->memsetD32(PxgDevicePtr(dstDevice), ui, N));
    }
    PxCUresult memsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream /*hStream*/) override
    {
        return memsetD8(dstDevice, uc, N);
    }
    PxCUresult memsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) override
    {
        if (mAbort) return mLastResult;
        // vkCmdFillBuffer is 32-bit; a byte fill maps to it only when the run
        // is 4-byte aligned (replicate the byte across the word).  Sub-word
        // byte fills (unaligned address or count not a multiple of 4) are
        // reported as InvalidValue rather than silently mis-filled — see
        // RUNTIME_PLAN.md for the P4 sub-word fill path.
        if ((N % 4u) != 0u || (uint64_t(dstDevice) % 4u) != 0u)
            return record(PxgResult::InvalidValue);
        const uint32_t word = uint32_t(uc) * 0x01010101u;
        return record(mShim->memsetD32(PxgDevicePtr(dstDevice), word, N / 4u));
    }
    PxCUresult memsetD16(CUdeviceptr dstDevice, unsigned short uh, size_t N) override
    {
        if (mAbort) return mLastResult;
        if ((N % 2u) != 0u || (uint64_t(dstDevice) % 4u) != 0u)
            return record(PxgResult::InvalidValue);
        const uint32_t word = (uint32_t(uh) << 16) | uint32_t(uh);
        return record(mShim->memsetD32(PxgDevicePtr(dstDevice), word, N / 2u));
    }

    // ---- misc ----
    // Matches CudaCtx::getLastError: in abort mode report OUT_OF_MEMORY (the
    // sticky terminal state the engine polls for) rather than the last recorded
    // op result, which a benign forwarded call could otherwise have cleared.
    PxCUresult getLastError() override { return mAbort ? oomResult() : mLastResult; }
    void setAbortMode(bool abort) override
    {
        mAbort = abort;
        // When abort clears, recover from the sticky OUT_OF_MEMORY so the next
        // getLastError() reads healthy again (matches CudaCtx::setAbortMode).
        if (!abort && mLastResult.value == uint32_t(PXG_CUDA_ERROR_OUT_OF_MEMORY))
            mLastResult = okResult();
    }
    bool isInAbortMode() override { return mAbort; }

private:
    PxCUresult record(PxgResult r)
    {
        mLastResult = toResult(r);
        return mLastResult;
    }

    PxgCudaShim* mShim;
    const PxgKernelDef* mManifest;
    uint32_t mManifestCount;
    PxCUresult mLastResult = okResult();
    bool mAbort = false;
};

// The PxCudaContextManager the scene receives.  It owns a PxgVulkanCudaContext
// and reports device properties; the per-thread acquire/release is a no-op
// because a single Vulkan device is not a per-thread-current CUDA context.
class PxgVulkanCudaContextManager : public physx::PxCudaContextManager
{
public:
    PxgVulkanCudaContextManager(PxgSlangComputeDevice* device, PxgCudaShim* shim,
                                const PxgKernelDef* manifest, uint32_t manifestCount)
        : mDevice(device)
    {
        mContext = new PxgVulkanCudaContext(shim, manifest, manifestCount);
    }

    // ---- context access ----
    // P5 Step 4b: pinned host memory is a persistently mapped host-visible
    // buffer with a device address; any interior CPU pointer of one resolves.
    CUdeviceptr getMappedDevicePtr(void* pinnedHostBuffer) override
    {
        return CUdeviceptr(mContext->shim()->hostGetDevicePointer(pinnedHostBuffer));
    }
    void acquireContext() override { ++mAcquireCount; }
    bool tryAcquireContext() override { ++mAcquireCount; return true; }
    void releaseContext() override { if (mAcquireCount > 0) --mAcquireCount; }
    CUcontext getContext() override { return reinterpret_cast<CUcontext>(uintptr_t(1)); }
    PxCudaContext* getCudaContext() override { return mContext; }
    bool contextIsValid() const override { return mContext != nullptr; }

    // ---- capability queries: the Vulkan port implements the modern kernels,
    // so it reports at least the SM levels the GPU code paths gate on ----
    bool supportsArchSM10() const override { return true; }
    bool supportsArchSM11() const override { return true; }
    bool supportsArchSM12() const override { return true; }
    bool supportsArchSM13() const override { return true; }
    bool supportsArchSM20() const override { return true; }
    bool supportsArchSM30() const override { return true; }
    bool supportsArchSM35() const override { return true; }
    bool supportsArchSM50() const override { return true; }
    bool supportsArchSM52() const override { return true; }
    bool supportsArchSM60() const override { return true; }
    bool isIntegrated() const override { return false; }     // staged device-local model
    bool canMapHostMemory() const override { return false; } // no zero-copy mapped host
    int getDriverVersion() const override { return 12000; }
    size_t getDeviceTotalMemBytes() const override { return size_t(4) << 30; }
    int getMultiprocessorCount() const override { return 16; }
    unsigned int getClockRate() const override { return 1000000; }
    int getSharedMemPerBlock() const override { return 32 * 1024; }
    int getSharedMemPerMultiprocessor() const override { return 64 * 1024; }
    unsigned int getMaxThreadsPerBlock() const override { return 1024; }
    const char* getDeviceName() const override { return mDevice ? mDevice->getDeviceName() : "PhysXGpuVulkan"; }
    CUdevice getDevice() const override { return 0; }
    void setUsingConcurrentStreams(bool v) override { mConcurrentStreams = v; }
    bool getUsingConcurrentStreams() const override { return mConcurrentStreams; }
    void getDeviceMemoryInfo(size_t& free, size_t& total) const override
    {
        total = getDeviceTotalMemBytes();
        free = total;
    }
    CUmodule* getCuModules() override
    {
        // Kernels are precompiled SPIR-V, so there is no CUmodule table — but the
        // engine's KernelWrangler ctor indexes getCuModules()[moduleIndex] (all
        // moduleIndex==0 in our manifest-driven function table) and hands the
        // result to moduleGetFunction, which ignores it.  Return a one-element
        // array holding the non-null module sentinel so that read is valid.
        static CUmodule sModule = reinterpret_cast<CUmodule>(uintptr_t(1));
        return &sModule;
    }
    void release() override { delete this; }

protected:
    ~PxgVulkanCudaContextManager() override
    {
        if (mContext)
            mContext->release();
    }

    // ---- deprecated Internal buffer helpers: forward onto the context ----
    void* allocDeviceBufferInternal(PxU64 numBytes, const char* /*f*/, PxI32 /*l*/) override
    {
        CUdeviceptr p = 0;
        return mContext->memAlloc(&p, size_t(numBytes)) == 0 ? reinterpret_cast<void*>(uintptr_t(p)) : nullptr;
    }
    void* allocPinnedHostBufferInternal(PxU64 numBytes, const char* /*f*/, PxI32 /*l*/) override
    {
        void* p = nullptr;
        mContext->memHostAlloc(&p, size_t(numBytes), 0);
        return p;
    }
    void freeDeviceBufferInternal(void* deviceBuffer) override
    {
        mContext->memFree(CUdeviceptr(uintptr_t(deviceBuffer)));
    }
    void freePinnedHostBufferInternal(void* pinnedHostBuffer) override
    {
        mContext->memFreeHost(pinnedHostBuffer);
    }
    void clearDeviceBufferAsyncInternal(void* deviceBuffer, PxU32 numBytes, CUstream stream, PxI32 value) override
    {
        mContext->memsetD32Async(CUdeviceptr(uintptr_t(deviceBuffer)), PxU32(value), numBytes / 4, stream);
    }
    void copyDToHAsyncInternal(void* hostBuffer, const void* deviceBuffer, PxU32 numBytes, CUstream stream) override
    {
        mContext->memcpyDtoHAsync(hostBuffer, CUdeviceptr(uintptr_t(deviceBuffer)), numBytes, stream);
    }
    void copyHToDAsyncInternal(void* deviceBuffer, const void* hostBuffer, PxU32 numBytes, CUstream stream) override
    {
        mContext->memcpyHtoDAsync(CUdeviceptr(uintptr_t(deviceBuffer)), hostBuffer, numBytes, stream);
    }
    void copyDToDAsyncInternal(void* dst, const void* src, PxU32 numBytes, CUstream stream) override
    {
        mContext->memcpyDtoDAsync(CUdeviceptr(uintptr_t(dst)), CUdeviceptr(uintptr_t(src)), numBytes, stream);
    }
    void copyDToHInternal(void* hostBuffer, const void* deviceBuffer, PxU32 numBytes) override
    {
        mContext->memcpyDtoH(hostBuffer, CUdeviceptr(uintptr_t(deviceBuffer)), numBytes);
    }
    void copyHToDInternal(void* deviceBuffer, const void* hostBuffer, PxU32 numBytes) override
    {
        mContext->memcpyHtoD(CUdeviceptr(uintptr_t(deviceBuffer)), hostBuffer, numBytes);
    }
    void memsetD8AsyncInternal(void* dst, const PxU8& value, PxU32 numBytes, CUstream stream) override
    {
        mContext->memsetD8Async(CUdeviceptr(uintptr_t(dst)), value, numBytes, stream);
    }
    void memsetD32AsyncInternal(void* dst, const PxU32& value, PxU32 numIntegers, CUstream stream) override
    {
        mContext->memsetD32Async(CUdeviceptr(uintptr_t(dst)), value, numIntegers, stream);
    }

private:
    PxgSlangComputeDevice* mDevice;
    PxgVulkanCudaContext* mContext;
    unsigned mAcquireCount = 0;
    bool mConcurrentStreams = false;
};

} // anonymous namespace

physx::PxCudaContext* PxgCreateVulkanCudaContext(PxgCudaShim* shim, const PxgKernelDef* manifest,
                                                 uint32_t manifestCount)
{
    return new PxgVulkanCudaContext(shim, manifest, manifestCount);
}

physx::PxCudaContextManager* PxgCreateVulkanCudaContextManager(PxgSlangComputeDevice* device,
                                                               PxgCudaShim* shim,
                                                               const PxgKernelDef* manifest,
                                                               uint32_t manifestCount)
{
    return new PxgVulkanCudaContextManager(device, shim, manifest, manifestCount);
}

// ---- PxgVulkanKernelWrangler ----
// getFunction uses the SAME id->CUfunction encoding as the anonymous-namespace
// toFunc above (kernel id + 1 in the pointer bits), so a handle it returns
// decodes back to manifest[id] inside PxgVulkanCudaContext::launchKernel.

CUfunction PxgVulkanKernelWrangler::getFunction(uint32_t kernelId) const
{
    if (kernelId >= mCount || !mManifest[kernelId].spirvFile)
        return nullptr;
    return toFunc(kernelId);
}

const char* PxgVulkanKernelWrangler::getFunctionName(uint32_t kernelId) const
{
    return kernelId < mCount ? mManifest[kernelId].entry : nullptr;
}

bool PxgVulkanKernelWrangler::isMapped(uint32_t kernelId) const
{
    return kernelId < mCount && mManifest[kernelId].spirvFile != nullptr;
}

uint32_t PxgVulkanKernelWrangler::mappedCount() const
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < mCount; ++i)
        if (mManifest[i].spirvFile)
            ++n;
    return n;
}

} // namespace gpuslang
} // namespace physx

#endif // PX_SUPPORT_GPU_PHYSX
