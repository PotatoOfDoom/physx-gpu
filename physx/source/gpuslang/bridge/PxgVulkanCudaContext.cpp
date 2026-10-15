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

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include "foundation/PxFoundation.h"
#include "foundation/PxErrorCallback.h"

namespace physx
{
namespace gpuslang
{

// Defined in bridge/PxgVulkanDriverShim.cpp: registers the live context so the raw
// (context-less) CUDA-driver shim entry points can route memcpys through the bridge.
void pxgSetVulkanDriverContext(physx::PxCudaContext* ctx);
// P9.12b: the 3D-texture driver entry points need the SHIM, not the context --
// cuArray3DCreate and friends have no PxCudaContext counterpart at all.
void pxgSetVulkanDriverShim(PxgCudaShim* shim);

namespace
{
// The live PxgVulkanCudaContexts.  pxgUnmappedLaunchCount takes any PxCudaContext
// and must tell ours apart without RTTI: PhysX builds with -fno-rtti / /GR-, so a
// dynamic_cast does not compile there.
// Both are leaked on purpose (like the bridge runtime): a context released from an
// application's static destructor must not find them already destroyed.
std::mutex& liveContextMutex()
{
    static std::mutex* m = new std::mutex;
    return *m;
}

std::vector<const physx::PxCudaContext*>& liveContexts()
{
    static std::vector<const physx::PxCudaContext*>* v = new std::vector<const physx::PxCudaContext*>;
    return *v;
}

void registerLiveContext(const physx::PxCudaContext* ctx, bool live)
{
    std::lock_guard<std::mutex> lock(liveContextMutex());
    std::vector<const physx::PxCudaContext*>& v = liveContexts();
    if (live)
        v.push_back(ctx);
    else
        v.erase(std::remove(v.begin(), v.end(), ctx), v.end());
}

bool isLiveContext(const physx::PxCudaContext* ctx)
{
    std::lock_guard<std::mutex> lock(liveContextMutex());
    const std::vector<const physx::PxCudaContext*>& v = liveContexts();
    return std::find(v.begin(), v.end(), ctx) != v.end();
}

// ---- handle <-> shim id encodings (see header) ----
inline CUstream toStream(PxgStreamId id) { return reinterpret_cast<CUstream>(uintptr_t(id)); }
inline PxgStreamId fromStream(CUstream s) { return PxgStreamId(reinterpret_cast<uintptr_t>(s)); }
inline CUevent toEvent(uint32_t id) { return reinterpret_cast<CUevent>(uintptr_t(id)); }
inline uint32_t fromEvent(CUevent e) { return uint32_t(reinterpret_cast<uintptr_t>(e)); }
inline CUfunction toFunc(PxgKernelId id) { return reinterpret_cast<CUfunction>(uintptr_t(id) + 1u); }

// P9.40 -- the UNMAPPED sentinel.  A kernel that is in the registry but has no
// SPIR-V gets a NON-NULL handle with this bit set, so that a launch through it can be
// named and counted.  fromFunc rejects it exactly as it rejects null, so nothing that
// used to refuse a null slot behaves differently; only the early return in
// launchKernel looks at it, through fromUnmappedFunc.
static const uintptr_t kUnmappedFuncBit = uintptr_t(1) << 40;
inline CUfunction toUnmappedFunc(PxgKernelId id)
{ return reinterpret_cast<CUfunction>((uintptr_t(id) + 1u) | kUnmappedFuncBit); }
inline bool fromUnmappedFunc(CUfunction f, PxgKernelId* id)
{
    const uintptr_t v = reinterpret_cast<uintptr_t>(f);
    if ((v & kUnmappedFuncBit) == 0 || (v & ~kUnmappedFuncBit) == 0)
        return false;
    *id = PxgKernelId((v & ~kUnmappedFuncBit) - 1u);
    return true;
}
inline bool fromFunc(CUfunction f, PxgKernelId* id)
{
    const uintptr_t v = reinterpret_cast<uintptr_t>(f);
    if (v == 0 || (v & kUnmappedFuncBit) != 0)
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
// the argument size, from which the alignment is the largest power of two that
// divides the size (capped at 16) — exact for scalars, pointers, power-of-two
// vector types, AND tightly-packed scalar aggregates such as a 12-byte PxVec3
// (alignment 4) passed by value (e.g. gravity in preIntegrationLaunchTGS).  The
// matching Slang push-constant field must be authored to the same layout.
//
// NOT exact for a by-value struct whose size has a larger power-of-two factor than
// its alignment: the 80-byte PxgFEMContactWriter (nine pointers + PxU32, alignof 8)
// gets paramAlign(80) = 16, where CUDA's .param bank would place it at 8.  Where the
// argument before it ends on an 8-but-not-16 boundary that inserts 8 pad bytes (P9.97,
// measured over every row): kClothBoxTriCg (four cloth kernels) and kSbPsContactGen
// put the writer at @112, not @104, and kSbSdfMidphase at @96, not @88.  The blob is
// built from this function and the push size from abiAlign (PxgVulkanKernelAbi.cpp),
// which agree, so the shaders must follow THIS placement -- the first five always did
// (`_pad104`); sb_sdfMeshMidphaseGeneratePairsLaunch's did not until P9.97b, and on the
// engine path it read its writer from the padding.  check_push_block.py's layout join
// now holds every shader to this rule.  Changing paramAlign would re-lay-out those six
// rows and their shaders; the divergence from CUDA is harmless as long as both ends
// agree, because nothing else ever reads the blob.
inline size_t paramAlign(size_t size)
{
    if (size == 0) return 1;
    if (size % 16 == 0) return 16;
    if (size % 8 == 0)  return 8;
    if (size % 4 == 0)  return 4;
    if (size % 2 == 0)  return 2;
    return 1;
}

class PxgVulkanCudaContext : public physx::PxCudaContext
{
public:
    PxgVulkanCudaContext(PxgCudaShim* shim, const PxgKernelDef* manifest, uint32_t manifestCount)
        : mShim(shim), mManifest(manifest), mManifestCount(manifestCount),
          mUnmappedLaunches(manifestCount, 0u)
    {
        mAllocatorCallback = nullptr;
        // Register as the live context so the raw (context-less) CUDA-driver shim
        // entry points (bridge/PxgVulkanDriverShim.cpp) route memcpys through us.
        pxgSetVulkanDriverContext(this);
        pxgSetVulkanDriverShim(shim);
        registerLiveContext(this, true);
    }

    void release() override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        reportUnmappedLaunches();
        registerLiveContext(this, false);
        pxgSetVulkanDriverContext(nullptr); pxgSetVulkanDriverShim(nullptr); delete this;
    }

    // P9.40.  How often kernel `id` was launched while unmapped (0 for mapped ones).
    uint32_t unmappedLaunchCount(PxgKernelId id) const
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        return id < mManifestCount ? mUnmappedLaunches[id] : 0u;
    }

    // The backing shim (bridge-internal): the ccm's getMappedDevicePtr resolves
    // pinned host pointers through it.
    PxgCudaShim* shim() const { return mShim; }

    // ---- memory ----
    PxCUresult memAlloc(CUdeviceptr* dptr, size_t bytesize) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        PxgDevicePtr p = 0;
        const PxgResult r = mShim->memAlloc(&p, bytesize);
        if (dptr)
            *dptr = CUdeviceptr(p);
        return record(r);
    }
    PxCUresult memFree(CUdeviceptr dptr) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        return record(mShim->memFree(PxgDevicePtr(dptr)));
    }

    PxCUresult memHostAlloc(void** pp, size_t bytesize, unsigned int /*Flags*/) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        // P5 Step 4b: real pinned-mapped memory — a persistently mapped host-
        // visible buffer with a device address, so the engine's mapped-pointer
        // model (PxPinnedArray + getMappedDevicePtr, the copy-descriptor queue,
        // bpDesc mirrors, ...) works as it does on CUDA.
        if (mAbort) return mLastResult;
        return record(mShim->memHostAlloc(pp, bytesize));
    }
    PxCUresult memFreeHost(void* p) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        return record(mShim->memFreeHost(p));
    }
    PxCUresult memHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int /*Flags*/) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
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
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        if (module)
            *module = sentinelModule();
        return record(PxgResult::Success);
    }
    PxCUresult moduleGetFunction(CUfunction* hfunc, CUmodule /*hmod*/, const char* name) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
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
        // P9.40: a registry name WITHOUT SPIR-V.  Still NotFound -- the engine
        // wrangler keeps reporting it and keeps mError -- but the slot receives the
        // unmapped sentinel instead of staying null, so a later launch through it can
        // say which kernel was skipped.  fromFunc refuses the sentinel, so the launch
        // itself still fails with InvalidValue exactly as before.
        for (uint32_t i = 0; i < mManifestCount; ++i)
        {
            if (!mManifest[i].spirvFile && mManifest[i].cudaName && name &&
                std::strcmp(mManifest[i].cudaName, name) == 0)
            {
                if (hfunc)
                    *hfunc = toUnmappedFunc(i);
                return record(PxgResult::NotFound);
            }
        }
        return record(PxgResult::NotFound);
    }
    PxCUresult moduleUnload(CUmodule /*hmod*/) override { std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex()); return record(PxgResult::Success); }

    // ---- streams ----
    PxCUresult streamCreate(CUstream* phStream, unsigned int /*Flags*/) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        PxgStreamId id = 0;
        const PxgResult r = mShim->streamCreate(&id);
        if (phStream)
            *phStream = toStream(id);
        return record(r);
    }
    PxCUresult streamCreateWithPriority(CUstream* phStream, unsigned int flags, int /*priority*/) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        return streamCreate(phStream, flags);
    }
    PxCUresult streamFlush(CUstream hStream) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        return record(mShim->streamSynchronize(fromStream(hStream)));
    }
    PxCUresult streamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int /*Flags*/) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        return record(mShim->streamWaitEvent(fromStream(hStream), fromEvent(hEvent)));
    }
    PxCUresult streamWaitEvent(CUstream hStream, CUevent hEvent) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        return record(mShim->streamWaitEvent(fromStream(hStream), fromEvent(hEvent)));
    }
    PxCUresult streamDestroy(CUstream hStream) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        return record(mShim->streamDestroy(fromStream(hStream)));
    }
    PxCUresult streamSynchronize(CUstream hStream) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        return record(mShim->streamSynchronize(fromStream(hStream)));
    }

    // ---- events ----
    PxCUresult eventCreate(CUevent* phEvent, unsigned int /*Flags*/) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        uint32_t id = 0;
        const PxgResult r = mShim->eventCreate(&id);
        if (phEvent)
            *phEvent = toEvent(id);
        return record(r);
    }
    PxCUresult eventRecord(CUevent hEvent, CUstream hStream) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        return record(mShim->eventRecord(fromEvent(hEvent), fromStream(hStream)));
    }
    PxCUresult eventQuery(CUevent hEvent) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        return record(mShim->eventQuery(fromEvent(hEvent)));
    }
    PxCUresult eventSynchronize(CUevent hEvent) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        return record(mShim->eventSynchronize(fromEvent(hEvent)));
    }
    PxCUresult eventDestroy(CUevent hEvent) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        return record(mShim->eventDestroy(fromEvent(hEvent)));
    }

    // ---- launch ----
    // PXG_LAUNCH_TRACE=1: append one line per launch attempt to stderr —
    // "form kernelName result" — the ground truth of what the engine actually
    // launches (and in what order) during a stepped scene.  Drives the P5
    // Step 4 bring-up work list; costs nothing when the env var is unset.
    // P9.48d: a REFUSED line said WHICH kernel was refused and nothing about WHY,
    // and the shim's blob gate is the usual reason -- its check is exact, so a
    // launch site that passes one argument too many (or a struct whose trailing
    // padding rounds differently) is rejected silently and the kernel simply
    // never runs.  That is indistinguishable, in a trace, from a kernel that ran
    // and found nothing.  The blob size the bridge actually built goes in the
    // line so the next reader can compare it against the shader's blob at once.
    void traceRefusal(const char* form, PxgKernelId id, uint32_t blobBytes, PxgResult res)
    {
        static const bool on = std::getenv("PXG_LAUNCH_TRACE") != nullptr;
        if (!on)
            return;
        const char* name = id < mManifestCount && mManifest[id].cudaName
            ? mManifest[id].cudaName
            : (id < mManifestCount && mManifest[id].entry ? mManifest[id].entry : "?");
        if (res == PxgResult::NotFound)
        {
            // P9.98: the device refused the pipeline (or the dispatch) and said why on
            // stderr -- a VkResult, or a device limit the entry point exceeds.  The shim
            // caches a refused build, so the device's line is printed ONCE, at the first
            // launch: it can be far above this one, hence "the first ... naming <entry>".
            const char* entry = id < mManifestCount && mManifest[id].entry ? mManifest[id].entry : "?";
            std::fprintf(stderr, "[pxg-refused] %s %s: no pipeline could be built or bound for it -- the device "
                                 "refused the build (the first [PxgVulkan] line naming %s says why) or this "
                                 "dispatch (the [PxgVulkan] dispatch line just above)\n", form, name, entry);
        }
        else
            std::fprintf(stderr, "[pxg-refused] %s %s built a %u-byte push blob -- the shim "
                                 "requires an EXACT match with the shader's\n", form, name, blobBytes);
    }

    void traceLaunch(const char* form, PxgKernelId id, bool ok,
                     PxU32 gx = 0xffffffffu, PxU32 gy = 0, PxU32 gz = 0)
    {
        static const bool on = std::getenv("PXG_LAUNCH_TRACE") != nullptr;
        if (!on)
            return;
        const char* name = id < mManifestCount && mManifest[id].cudaName
            ? mManifest[id].cudaName
            : (id < mManifestCount && mManifest[id].entry ? mManifest[id].entry : "?");
        // P9.48d: the GRID goes in the line.  "launched" and "launched over
        // nothing" are different facts, and the trace could not tell them apart:
        // a dispatch whose grid has a ZERO dimension is a legal no-op, so a
        // kernel the engine drives with numTests == 0 traced exactly like one
        // doing work.  A whole scene's worth of "the chain runs" rested on that
        // distinction.  When any dimension is 0 the line says so in words, so it
        // cannot be skimmed past.
        if (gx == 0xffffffffu)
            std::fprintf(stderr, "[pxg-launch] %s %s %s\n", form, name, ok ? "OK" : "REFUSED");
        else if (gx == 0u || gy == 0u || gz == 0u)
            std::fprintf(stderr, "[pxg-launch] %s %s %s grid(%u,%u,%u) EMPTY-GRID\n",
                         form, name, ok ? "OK" : "REFUSED", gx, gy, gz);
        else
            std::fprintf(stderr, "[pxg-launch] %s %s %s grid(%u,%u,%u)\n",
                         form, name, ok ? "OK" : "REFUSED", gx, gy, gz);
    }

    // PXG_LAUNCH_TRACE=1: name the kernel BEFORE handing it to the shim, so the
    // trace brackets every launch with a "...\" line and an "OK"/"REFUSED" line.
    //
    // NOTE what this does NOT buy on its own: launchKernel only RECORDS a
    // vkCmdDispatch, and submission signals a timeline semaphore rather than
    // waiting on a fence, so a batch of dispatches is still in flight when the
    // "OK" lines print.  A shader that faults therefore kills the process
    // several trace lines after the launch that caused it.  To attribute a GPU
    // fault to its exact kernel, set PXG_SYNC_EACH=1 as well: PxgCudaShim::
    // launchKernel drains the queue after every dispatch, so the kernel whose
    // "[pxg-dispatch] ..." line has no matching "OK" line is the one that died.
    void traceDispatch(const char* form, PxgKernelId id)
    {
        static const bool on = std::getenv("PXG_LAUNCH_TRACE") != nullptr;
        if (!on)
            return;
        const char* name = id < mManifestCount && mManifest[id].cudaName
            ? mManifest[id].cudaName
            : (id < mManifestCount && mManifest[id].entry ? mManifest[id].entry : "?");
        std::fprintf(stderr, "[pxg-dispatch] %s %s ...\n", form, name);
        std::fflush(stderr);
    }

    // Pinned-host pointer arguments: substitute the buffer device address.
    //
    // Under CUDA's unified addressing a cuMemHostAlloc pointer IS a device
    // pointer, so PhysX passes such pointers to kernels unchanged.  Here pinned
    // memory is a host-visible Vulkan buffer whose CPU mapping and device address
    // are unrelated numbers, and the blob builders copy argument bytes verbatim --
    // so such arguments must be translated on the way in.
    //
    // Which arguments: ONLY the ones the ABI row's hostPtrMask marks.  Translating
    // every 8-byte argument that happens to lie in a pinned allocation (tried once)
    // is wrong: some pinned pointers are ADDRESS TAGS that a kernel only subtracts
    // from another pinned pointer -- the narrowphase contact-stream bases
    // (startContactPatches/Points/Forces, cpuCompressed*Base) -- and translating one
    // side of that subtraction yields a garbage patch index (an AMD RX 9070 XT read
    // fault after constraintContactBlockPrePrepLaunch).  llvmpipe cannot reveal a
    // missing mark: there a pinned allocation's device address equals its CPU
    // address, so an untranslated pointer works by accident.  PXG_VERBOSE=1 lists
    // unmarked arguments that carry a pinned address (see noteUnmarkedHostPtr) so a
    // missing mark can be found on llvmpipe, too.
    //
    // A marked argument: null stays null (CUDA parity: kernels null-check these), and
    // a non-null value that resolves to nothing is a REFUSED LAUNCH rather than a
    // shader writing to a CPU address.
    bool translateHostPtrs(const PxgKernelAbiRow* abi, std::vector<uint8_t>& blob,
                           const std::vector<size_t>& offsets, PxgKernelId id) const
    {
        if (!abi)
            return true;
        static const bool verbose = [] {
            const char* v = std::getenv("PXG_VERBOSE");
            return v && std::strcmp(v, "0") != 0;
        }();
        for (uint32_t i = 0; i < abi->argCount && i < offsets.size(); ++i)
        {
            const bool marked = ((abi->hostPtrMask >> i) & 1u) != 0u;
            if (!marked && !verbose)
                continue;
            if (abi->argSizes[i] != 8 || offsets[i] + 8 > blob.size())
            {
                if (marked)
                    return false;
                continue;
            }
            uint64_t value = 0;
            std::memcpy(&value, blob.data() + offsets[i], 8);
            if (value == 0)
                continue;
            const PxgDevicePtr dev = mShim->hostGetDevicePointer(reinterpret_cast<const void*>(uintptr_t(value)));
            if (!marked)
            {
                if (dev != 0)
                    noteUnmarkedHostPtr(id, i);
                continue;
            }
            if (dev == 0)
                return false;
            const uint64_t devAddr = uint64_t(dev);
            std::memcpy(blob.data() + offsets[i], &devAddr, 8);
        }
        return true;
    }

    // PXG_VERBOSE=1: name each (kernel, argument) that carried a pinned host pointer
    // without a hostPtrMask bit, once.  Either an address tag (correctly left alone)
    // or a missing mark -- the kernel source decides which.
    void noteUnmarkedHostPtr(PxgKernelId id, uint32_t arg) const
    {
        static std::vector<uint64_t> seen;
        const uint64_t key = (uint64_t(id) << 8) | arg;
        if (std::find(seen.begin(), seen.end(), key) != seen.end())
            return;
        seen.push_back(key);
        const char* name = id < mManifestCount && mManifest[id].cudaName ? mManifest[id].cudaName : "?";
        std::fprintf(stderr, "[pxg-hostptr] %s argument %u: pinned host pointer NOT translated (no hostPtrMask bit; "
                             "mark it if the kernel dereferences it)\n", name, arg);
    }

    PxCUresult launchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                            unsigned int /*blockDimX*/, unsigned int /*blockDimY*/, unsigned int /*blockDimZ*/,
                            unsigned int /*sharedMemBytes*/, CUstream /*hStream*/, PxCudaKernelParam* kernelParams,
                            size_t kernelParamsSizeInBytes, void** /*extra*/, const char* /*file*/,
                            int /*line*/) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        PxgKernelId id = 0;
        if (!fromFunc(f, &id))
        {
            noteUnmappedLaunch(f, "sized");
            return record(PxgResult::InvalidValue);
        }

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
        std::vector<size_t> offsets;
        offsets.reserve(count);
        size_t maxAlign = 1;
        for (uint32_t i = 0; i < count; ++i)
        {
            const size_t a = paramAlign(kernelParams[i].size);
            if (a > maxAlign) maxAlign = a;
            blob.resize((blob.size() + (a - 1)) & ~(a - 1));   // pad to natural alignment
            offsets.push_back(blob.size());
            const uint8_t* d = static_cast<const uint8_t*>(kernelParams[i].data);
            blob.insert(blob.end(), d, d + kernelParams[i].size);
        }
        // Trailing struct padding: a compiled push-constant struct's size is rounded
        // up to its largest member's alignment (e.g. { uint64 ptr; uint count; } is 16,
        // not 12).  The Slang side sees exactly that struct, so round the blob to match,
        // otherwise the shim's blobSize == pushConstantSize gate rejects the launch.
        blob.resize((blob.size() + (maxAlign - 1)) & ~(maxAlign - 1));
        {
            const PxgKernelAbiRow* abi = id < mManifestCount ? pxgFindKernelAbi(mManifest[id].cudaName) : nullptr;
            if (!translateHostPtrs(abi, blob, offsets, id))
            {
                traceLaunch("sized", id, false);
                return record(PxgResult::InvalidValue);
            }
        }
        traceDispatch("sized", id);
        const PxgResult res = mShim->launchKernel(id, gridDimX, gridDimY, gridDimZ, blob.data(),
                                                  uint32_t(blob.size()));
        if (res != PxgResult::Success)
            traceRefusal("sized", id, uint32_t(blob.size()), res);
        traceLaunch("sized", id, res == PxgResult::Success, gridDimX, gridDimY, gridDimZ);
        if (res == PxgResult::Success)
            dumpArgs(id, blob);
        return record(res);
    }

    // PXG_DUMP_ARGS=<kernel>:<blobOffset>[>inner]:<bytes>[:<blobOffset>[>inner]:<bytes>...]
    // After every launch of <kernel>, reads <bytes> from the device pointer stored at
    // <blobOffset> of its push blob and prints them as 32-bit words on stderr; with
    // `>inner` it first follows the 64-bit pointer stored `inner` bytes into that
    // buffer (one level, e.g. a shape record's mesh pointer).  A
    // bring-up probe for the case the launch trace cannot separate: a kernel that
    // RUNS and whose effect does not arrive.  memcpyDtoH drains the queue first
    // (P3 serialises on one queue), so the words are what the dispatch wrote.
    // P9.83b-8 T7 added it for triangleTriangleCollision in PXG_SCENE=sdfmesh.
    void dumpArgs(PxgKernelId id, const std::vector<uint8_t>& blob)
    {
        static const char* spec = std::getenv("PXG_DUMP_ARGS");
        if (!spec || id >= mManifestCount || !mManifest[id].cudaName)
            return;
        const char* name = mManifest[id].cudaName;
        const char* colon = std::strchr(spec, ':');
        if (!colon || std::strlen(name) != size_t(colon - spec) || std::strncmp(spec, name, size_t(colon - spec)) != 0)
            return;
        const char* p = colon + 1;
        while (*p)
        {
            char* end = nullptr;
            const unsigned long off = std::strtoul(p, &end, 10);
            if (end == p)
                return;
            long inner = -1;
            if (*end == '>')
            {
                const char* q = end + 1;
                inner = long(std::strtoul(q, &end, 10));
                if (end == q)
                    return;
            }
            if (*end != ':')
                return;
            p = end + 1;
            const unsigned long bytes = std::strtoul(p, &end, 10);
            if (end == p)
                return;
            p = (*end == ':') ? end + 1 : end;
            if (off + 8u > blob.size() || bytes == 0u || bytes > 4096u)
                continue;
            uint64_t addr = 0;
            std::memcpy(&addr, blob.data() + off, 8);
            if (inner >= 0 && addr)
            {
                uint64_t next = 0;
                if (mShim->memcpyDtoH(&next, PxgDevicePtr(addr + uint64_t(inner)), 8) != PxgResult::Success)
                    next = 0;
                addr = next;
            }
            std::vector<uint32_t> w((bytes + 3u) / 4u, 0u);
            const PxgResult r = addr ? mShim->memcpyDtoH(w.data(), PxgDevicePtr(addr), bytes) : PxgResult::InvalidValue;
            std::fprintf(stderr, "[pxg-dump] %s @%lu addr=0x%llx%s:", name, off, (unsigned long long)addr,
                         r == PxgResult::Success ? "" : " (unreadable)");
            if (r == PxgResult::Success)
                for (size_t i = 0; i < w.size(); ++i)
                    std::fprintf(stderr, " %08x", w[i]);
            std::fprintf(stderr, "\n");
        }
    }
    PxCUresult launchKernel(CUfunction f, PxU32 gridDimX, PxU32 gridDimY, PxU32 gridDimZ,
                            PxU32 /*blockDimX*/, PxU32 /*blockDimY*/, PxU32 /*blockDimZ*/, PxU32 /*sharedMemBytes*/,
                            CUstream /*hStream*/, void** kernelParams, void** /*extra*/, const char* /*file*/,
                            int /*line*/) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        // The void** launch form (PX_CUDA_KERNEL_PARAM2; USE_NEW_LAUNCH_FUNCTION
        // in broadphase/AABB-manager/articulation) carries no per-argument sizes
        // — CUDA reads them from the compiled kernel's parameter metadata.  The
        // bridge reads them from the per-kernel launch-ABI table instead
        // (PxgVulkanKernelAbi.cpp) and assembles the same natural-alignment blob
        // as the sized form.  A kernel without an ABI row is refused cleanly.
        if (mAbort) return mLastResult;
        PxgKernelId id = 0;
        if (!fromFunc(f, &id))
        {
            noteUnmappedLaunch(f, "void**");
            return record(PxgResult::InvalidValue);
        }
        const PxgKernelAbiRow* abi = id < mManifestCount ? pxgFindKernelAbi(mManifest[id].cudaName) : nullptr;
        if (!abi || !kernelParams)
        {
            traceLaunch("void**", id, false);
            return record(PxgResult::InvalidValue);
        }

        std::vector<uint8_t> blob;
        std::vector<size_t> offsets;
        offsets.reserve(abi->argCount);
        size_t maxAlign = 1;
        for (uint32_t i = 0; i < abi->argCount; ++i)
        {
            const size_t a = paramAlign(abi->argSizes[i]);
            if (a > maxAlign) maxAlign = a;
            blob.resize((blob.size() + (a - 1)) & ~(a - 1));
            offsets.push_back(blob.size());
            const uint8_t* d = static_cast<const uint8_t*>(kernelParams[i]);
            blob.insert(blob.end(), d, d + abi->argSizes[i]);
        }
        blob.resize((blob.size() + (maxAlign - 1)) & ~(maxAlign - 1));
        if (!translateHostPtrs(abi, blob, offsets, id))
        {
            traceLaunch("void**", id, false);
            return record(PxgResult::InvalidValue);
        }
        traceDispatch("void**", id);
        const PxgResult res = mShim->launchKernel(id, gridDimX, gridDimY, gridDimZ, blob.data(),
                                                  uint32_t(blob.size()));
        if (res != PxgResult::Success)
            traceRefusal("void**", id, uint32_t(blob.size()), res);
        traceLaunch("void**", id, res == PxgResult::Success);
        return record(res);
    }

    // ---- copies ----
    // One queue serializes every stream.  An async copy whose host side is pinned is
    // recorded into it (no drain); the synchronous forms, and async copies to or from
    // pageable memory, complete before returning (PxgCudaShim::memcpy*).
    PxCUresult memcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        return record(mShim->memcpyDtoH(dstHost, PxgDevicePtr(srcDevice), ByteCount));
    }
    PxCUresult memcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream /*hStream*/) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        return record(mShim->memcpyDtoHAsync(dstHost, PxgDevicePtr(srcDevice), ByteCount));
    }
    PxCUresult memcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        return record(mShim->memcpyHtoD(PxgDevicePtr(dstDevice), srcHost, ByteCount));
    }
    PxCUresult memcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream /*hStream*/) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        return record(mShim->memcpyHtoDAsync(PxgDevicePtr(dstDevice), srcHost, ByteCount));
    }
    PxCUresult memcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream /*hStream*/) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        return record(mShim->memcpyDtoD(PxgDevicePtr(dstDevice), PxgDevicePtr(srcDevice), ByteCount));
    }
    PxCUresult memcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        return record(mShim->memcpyDtoD(PxgDevicePtr(dstDevice), PxgDevicePtr(srcDevice), ByteCount));
    }
    PxCUresult memcpyPeerAsync(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        return record(PxgResult::InvalidValue); // single device: no peer copies
    }

    // ---- memset ----
    PxCUresult memsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream /*hStream*/) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        return record(mShim->memsetD32(PxgDevicePtr(dstDevice), ui, N));
    }
    PxCUresult memsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        if (mAbort) return mLastResult;
        return record(mShim->memsetD32(PxgDevicePtr(dstDevice), ui, N));
    }
    PxCUresult memsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream /*hStream*/) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        return memsetD8(dstDevice, uc, N);
    }
    PxCUresult memsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
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
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
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
    PxCUresult getLastError() override { std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex()); return mAbort ? oomResult() : mLastResult; }
    void setAbortMode(bool abort) override
    {
        std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex());
        mAbort = abort;
        // When abort clears, recover from the sticky OUT_OF_MEMORY so the next
        // getLastError() reads healthy again (matches CudaCtx::setAbortMode).
        if (!abort && mLastResult.value == uint32_t(PXG_CUDA_ERROR_OUT_OF_MEMORY))
            mLastResult = okResult();
    }
    bool isInAbortMode() override { std::lock_guard<std::recursive_mutex> apiLock(pxgGpuApiMutex()); return mAbort; }

private:
    PxCUresult record(PxgResult r)
    {
        mLastResult = toResult(r);
        return mLastResult;
    }

    // P9.40.  Called on the refuse path of both launch forms.  A null handle stays
    // anonymous (that is what the sentinel exists to avoid); a sentinel handle names
    // the kernel, bumps its counter, and under PXG_LAUNCH_TRACE prints ONE line the
    // first time -- not one per frame, which for a per-step kernel would be hundreds.
    void noteUnmappedLaunch(CUfunction f, const char* form)
    {
        PxgKernelId id = 0;
        if (!fromUnmappedFunc(f, &id) || id >= mManifestCount)
            return;
        const uint32_t n = ++mUnmappedLaunches[id];
        static const bool on = std::getenv("PXG_LAUNCH_TRACE") != nullptr;
        if (on && n == 1u)
            std::fprintf(stderr, "[pxg-launch] %s %s UNMAPPED -- skipped (further launches not traced)\n",
                         form, mManifest[id].cudaName ? mManifest[id].cudaName : "?");
    }

    // P9.40.  At release, every kernel that was launched while unmapped is reported
    // through the foundation error callback, one line each, so the host that owns the
    // scene sees it in the same stream as the wrangler's startup list -- and so a
    // harness can count "[pxg-unmapped]" lines and fail.  Silence here would be the
    // exact defect this exists to end, so the report is unconditional whenever a
    // foundation exists; only the trace line above is opt-in.
    void reportUnmappedLaunches()
    {
        uint32_t distinct = 0;
        for (uint32_t i = 0; i < mManifestCount; ++i)
            if (mUnmappedLaunches[i]) ++distinct;
        if (distinct == 0)
            return;
        physx::PxFoundation* fnd = PxIsFoundationValid();
        for (uint32_t i = 0; i < mManifestCount; ++i)
        {
            if (!mUnmappedLaunches[i])
                continue;
            char buf[320];
            std::snprintf(buf, sizeof(buf),
                          "[pxg-unmapped] %s was launched %u time(s) but has no SPIR-V mapping: "
                          "every launch was SKIPPED, the physics it computes did not happen",
                          mManifest[i].cudaName ? mManifest[i].cudaName : "?", mUnmappedLaunches[i]);
            if (fnd)
                fnd->getErrorCallback().reportError(physx::PxErrorCode::eINTERNAL_ERROR, buf, PX_FL);
            else
                std::fprintf(stderr, "%s\n", buf);
        }
    }

    PxgCudaShim* mShim;
    const PxgKernelDef* mManifest;
    uint32_t mManifestCount;
    std::vector<uint32_t> mUnmappedLaunches;   // P9.40: per kernel id
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
    void acquireContext() override { mAcquireCount.fetch_add(1u); }
    bool tryAcquireContext() override { mAcquireCount.fetch_add(1u); return true; }
    void releaseContext() override
    {
        unsigned n = mAcquireCount.load();
        while (n > 0 && !mAcquireCount.compare_exchange_weak(n, n - 1u)) {}
    }
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
    std::atomic<unsigned> mAcquireCount{0};
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

// P9.40 -- see the header.
uint32_t pxgUnmappedLaunchCount(const physx::PxCudaContext* ctx, uint32_t kernelId)
{
    if (!ctx || !isLiveContext(ctx))
        return 0u;
    return static_cast<const PxgVulkanCudaContext*>(ctx)->unmappedLaunchCount(kernelId);
}

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
