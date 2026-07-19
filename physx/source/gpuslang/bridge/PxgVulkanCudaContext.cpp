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

#if PX_SUPPORT_GPU_PHYSX

#include <cstdlib>
#include <cstring>
#include <vector>

namespace physx
{
namespace gpuslang
{

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

class PxgVulkanCudaContext : public physx::PxCudaContext
{
public:
    PxgVulkanCudaContext(PxgCudaShim* shim, const PxgKernelDef* manifest, uint32_t manifestCount)
        : mShim(shim), mManifest(manifest), mManifestCount(manifestCount)
    {
        mAllocatorCallback = nullptr;
    }

    void release() override { delete this; }

    // ---- memory ----
    PxCUresult memAlloc(CUdeviceptr* dptr, size_t bytesize) override
    {
        PxgDevicePtr p = 0;
        const PxgResult r = mShim->memAlloc(&p, bytesize);
        if (dptr)
            *dptr = CUdeviceptr(p);
        return record(r);
    }
    PxCUresult memFree(CUdeviceptr dptr) override { return record(mShim->memFree(PxgDevicePtr(dptr))); }

    PxCUresult memHostAlloc(void** pp, size_t bytesize, unsigned int /*Flags*/) override
    {
        void* p = std::malloc(bytesize ? bytesize : 1);
        if (pp)
            *pp = p;
        return p ? record(PxgResult::Success) : record(PxgResult::OutOfMemory);
    }
    PxCUresult memFreeHost(void* p) override { std::free(p); return record(PxgResult::Success); }
    PxCUresult memHostGetDevicePointer(CUdeviceptr* /*pdptr*/, void* /*p*/, unsigned int /*Flags*/) override
    {
        // Zero-copy mapped-host device pointers are not part of the P3 single-
        // device model; callers use explicit staged copies instead.
        return record(PxgResult::InvalidValue);
    }

    // ---- modules / functions ----
    PxCUresult moduleLoadDataEx(CUmodule* module, const void* /*image*/, unsigned int /*numOptions*/,
                                PxCUjit_option* /*options*/, void** /*optionValues*/) override
    {
        if (module)
            *module = sentinelModule();
        return record(PxgResult::Success);
    }
    PxCUresult moduleGetFunction(CUfunction* hfunc, CUmodule /*hmod*/, const char* name) override
    {
        for (uint32_t i = 0; i < mManifestCount; ++i)
        {
            if (mManifest[i].entry && name && std::strcmp(mManifest[i].entry, name) == 0)
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
        return record(mShim->streamSynchronize(fromStream(hStream)));
    }
    PxCUresult streamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int /*Flags*/) override
    {
        return record(mShim->streamWaitEvent(fromStream(hStream), fromEvent(hEvent)));
    }
    PxCUresult streamWaitEvent(CUstream hStream, CUevent hEvent) override
    {
        return record(mShim->streamWaitEvent(fromStream(hStream), fromEvent(hEvent)));
    }
    PxCUresult streamDestroy(CUstream hStream) override
    {
        return record(mShim->streamDestroy(fromStream(hStream)));
    }
    PxCUresult streamSynchronize(CUstream hStream) override
    {
        return record(mShim->streamSynchronize(fromStream(hStream)));
    }

    // ---- events ----
    PxCUresult eventCreate(CUevent* phEvent, unsigned int /*Flags*/) override
    {
        uint32_t id = 0;
        const PxgResult r = mShim->eventCreate(&id);
        if (phEvent)
            *phEvent = toEvent(id);
        return record(r);
    }
    PxCUresult eventRecord(CUevent hEvent, CUstream hStream) override
    {
        return record(mShim->eventRecord(fromEvent(hEvent), fromStream(hStream)));
    }
    PxCUresult eventQuery(CUevent hEvent) override { return record(mShim->eventQuery(fromEvent(hEvent))); }
    PxCUresult eventSynchronize(CUevent hEvent) override
    {
        return record(mShim->eventSynchronize(fromEvent(hEvent)));
    }
    PxCUresult eventDestroy(CUevent hEvent) override { return record(mShim->eventDestroy(fromEvent(hEvent))); }

    // ---- launch ----
    PxCUresult launchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                            unsigned int /*blockDimX*/, unsigned int /*blockDimY*/, unsigned int /*blockDimZ*/,
                            unsigned int /*sharedMemBytes*/, CUstream /*hStream*/, PxCudaKernelParam* kernelParams,
                            size_t kernelParamsSizeInBytes, void** /*extra*/, const char* /*file*/,
                            int /*line*/) override
    {
        PxgKernelId id = 0;
        if (!fromFunc(f, &id))
            return record(PxgResult::InvalidValue);

        // CUDA passes each argument as its own pointer and reads sizeof(arg)
        // from the function signature; PhysX carries those sizes in the
        // PxCudaKernelParam.size fields.  The Slang kernel receives one push-
        // constant blob, so concatenate the arguments in declaration order —
        // the identity copy for the common single-descriptor-struct kernel.
        const uint32_t count = uint32_t(kernelParamsSizeInBytes / sizeof(PxCudaKernelParam));
        std::vector<uint8_t> blob;
        for (uint32_t i = 0; i < count; ++i)
        {
            const uint8_t* d = static_cast<const uint8_t*>(kernelParams[i].data);
            blob.insert(blob.end(), d, d + kernelParams[i].size);
        }
        return record(mShim->launchKernel(id, gridDimX, gridDimY, gridDimZ, blob.data(),
                                          uint32_t(blob.size())));
    }
    PxCUresult launchKernel(CUfunction /*f*/, PxU32 /*gridDimX*/, PxU32 /*gridDimY*/, PxU32 /*gridDimZ*/,
                            PxU32 /*blockDimX*/, PxU32 /*blockDimY*/, PxU32 /*blockDimZ*/, PxU32 /*sharedMemBytes*/,
                            CUstream /*hStream*/, void** /*kernelParams*/, void** /*extra*/, const char* /*file*/,
                            int /*line*/) override
    {
        // The void** form carries no per-argument sizes, so the push-constant
        // blob cannot be assembled generically; PhysX GPU launches use the
        // PxCudaKernelParam form above.  (P4 revisits any void** call sites.)
        return record(PxgResult::InvalidValue);
    }

    // ---- copies (async == sync: P3 serializes on one queue) ----
    PxCUresult memcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) override
    {
        return record(mShim->memcpyDtoH(dstHost, PxgDevicePtr(srcDevice), ByteCount));
    }
    PxCUresult memcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream /*hStream*/) override
    {
        return record(mShim->memcpyDtoH(dstHost, PxgDevicePtr(srcDevice), ByteCount));
    }
    PxCUresult memcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) override
    {
        return record(mShim->memcpyHtoD(PxgDevicePtr(dstDevice), srcHost, ByteCount));
    }
    PxCUresult memcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream /*hStream*/) override
    {
        return record(mShim->memcpyHtoD(PxgDevicePtr(dstDevice), srcHost, ByteCount));
    }
    PxCUresult memcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream /*hStream*/) override
    {
        return record(mShim->memcpyDtoD(PxgDevicePtr(dstDevice), PxgDevicePtr(srcDevice), ByteCount));
    }
    PxCUresult memcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) override
    {
        return record(mShim->memcpyDtoD(PxgDevicePtr(dstDevice), PxgDevicePtr(srcDevice), ByteCount));
    }
    PxCUresult memcpyPeerAsync(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream) override
    {
        return record(PxgResult::InvalidValue); // single device: no peer copies
    }

    // ---- memset ----
    PxCUresult memsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream /*hStream*/) override
    {
        return record(mShim->memsetD32(PxgDevicePtr(dstDevice), ui, N));
    }
    PxCUresult memsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) override
    {
        return record(mShim->memsetD32(PxgDevicePtr(dstDevice), ui, N));
    }
    PxCUresult memsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream /*hStream*/) override
    {
        return memsetD8(dstDevice, uc, N);
    }
    PxCUresult memsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) override
    {
        // vkCmdFillBuffer is 32-bit; a byte fill maps to it only when the run
        // is 4-byte aligned (replicate the byte across the word).
        if ((N % 4u) != 0u || (uint64_t(dstDevice) % 4u) != 0u)
            return record(PxgResult::InvalidValue);
        const uint32_t word = uint32_t(uc) * 0x01010101u;
        return record(mShim->memsetD32(PxgDevicePtr(dstDevice), word, N / 4u));
    }
    PxCUresult memsetD16(CUdeviceptr dstDevice, unsigned short uh, size_t N) override
    {
        if ((N % 2u) != 0u || (uint64_t(dstDevice) % 4u) != 0u)
            return record(PxgResult::InvalidValue);
        const uint32_t word = (uint32_t(uh) << 16) | uint32_t(uh);
        return record(mShim->memsetD32(PxgDevicePtr(dstDevice), word, N / 2u));
    }

    // ---- misc ----
    PxCUresult getLastError() override { return mLastResult; }
    void setAbortMode(bool abort) override { mAbort = abort; }
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
    CUdeviceptr getMappedDevicePtr(void* /*pinnedHostBuffer*/) override { return 0; }
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
    CUmodule* getCuModules() override { return nullptr; } // kernels are precompiled SPIR-V
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

} // namespace gpuslang
} // namespace physx

#endif // PX_SUPPORT_GPU_PHYSX
