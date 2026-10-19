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

// testPipelineRefusal.cpp -- P9.98: what the device does when a pipeline cannot be
// built, and what it builds when it can.  Suite gpuslang.pipelineRefusal.
//
// F4 (the validation layer's one real finding).  Until P9.98 createPipelineInternal
// checked its VkResults with PXG_VK_CHECK, which only asserts -- and both build trees
// are -DNDEBUG -- then registered the pipeline anyway.  The shim's `pipe == 0 ->
// NotFound` therefore never fired, and the first dispatch bound VK_NULL_HANDLE:
// MEASURED, scan with vkCreateComputePipelines forced to fail (an LD_PRELOAD shim)
// exits 139, and so does a wrong entry name (VK -13).  dispatch() looked the handle up
// with operator[], so an unknown handle default-inserted a null pipeline too.
//
// Device limits.  Nothing compared a kernel with maxPushConstantsSize,
// maxComputeSharedMemorySize or the workgroup-size limits; llvmpipe's are generous
// (256 / 32768 / 1024), so a kernel over a smaller device's limit was a driver error
// there -- which then ran into F4.  The PxgSlangDeviceDesc test hooks lower a limit
// on a second device, so the refusal is shown here on llvmpipe.
//
// F2.  The layout's push range is max(caller size, the entry's SPIR-V block), and
// dispatch zero-fills the surplus (VUID-07987).  p98PushPad.slang reads its tail,
// which no engine kernel does, so the zero fill is observable here.
//
// The numbers pinned below are the measured ones from the compiled modules; the C++
// walker (pxgSpirvReflectEntry) agrees with tools/check_push_block.py and
// tools/check_groupshared.py on all 911 entry points of the build (P9.98 report).
//
// Forced Vulkan failures.  llvmpipe never fails a create call for a well-formed request,
// and the device refuses a missing entry or an over-limit kernel BEFORE it calls Vulkan,
// so nothing would reach the per-call VkResult checks.  This file therefore interposes
// the four create calls createPipelineInternal makes (and their destroys): the backend
// is a static library linked into this executable, so its calls bind to the definitions
// below, which forward to libvulkan (dlsym RTLD_NEXT) unless pipelineRefusal_vkFailures
// has armed one of them.  Unarmed they only forward -- every other suite in the binary
// runs through them unchanged.
//
// Check round 0 (P9.98).  The same interposer now also covers the buffer, texture,
// descriptor and init calls (pipelineRefusal_outOfMemory): a failed allocation must be a
// 0 handle with nothing leaked -- the shim's OutOfMemory -- where it used to segfault
// inside createBuffer; a failed descriptor-set allocation must be a refused dispatch;
// and a failed init step must be no device.  pipelineRefusal_bindings pins what a
// dispatch binds: the counts and every buffer / texture / arg-buffer handle are checked
// before anything is recorded (they were assert-only, and operator[] lookups).
//
// Check round 1.  The interposer writes JUNK into a failed create's output (the spec leaves
// it undefined) and counts any destroy of it; it also forces the sampler and the remaining
// init steps (command pool, command buffers, and the now-optional pipeline cache).  Added:
// sizes past 4 GiB (hand-assembled modules) must be refused, not wrapped and built; a grid
// over maxComputeWorkGroupCount is a refused dispatch; every PXG_VK_LIMIT_* name is pinned;
// device and shim transfers past their buffer are refused; and the widened push range is
// reported under PXG_VK_DEBUG_NAMES.
//
// P9.99 (pipelineRefusal_subgroupSize*).  createPipelineEx asks for an EXACT subgroup size
// with FULL subgroups.  Until P9.99 it checked only the device's window: an in-window size
// on an entry whose workgroup x is not a multiple of it (VUID-02757) or whose x*y*z exceeds
// size x maxComputeWorkgroupSubgroups (VUID-02756) was BUILT -- measured on HEAD 2a97aa5
// with p99SubgroupShape.spv: 5 rows, 5 layer errors, both memory modes -- and llvmpipe,
// which refuses no size itself (4, 6, 12, 16, 256: VkResult 0), ran x = 4 as 8-lane
// subgroups with 4 active lanes.  The rule (pxgRequiredSubgroupSizeFits) is pinned by a
// literal host oracle; the device rows by what the driver was asked for (the interposer
// records every vkCreateComputePipelines stage) and by what the subgroups did; the
// desc hooks take a capability away so each refusal shows on llvmpipe.  No engine path
// calls createPipelineEx (the shim builds through createPipeline / -Tex / -Bindless).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace
{
// Which create call fails next (nullptr = none), and the objects the creates made that
// their destroys have not released yet, counted only while gP98Track is set.
const char* gP98Fail = nullptr;
uint32_t gP98Skip = 0;   // armed calls to let through before the first failure (p98Fire)
uint32_t gP98NullPool = 0;   // vkAllocateDescriptorSets calls handed a pool that failed to create
bool gP98Track = false;
uint32_t gP98Created = 0;
std::set<uint64_t> gP98Live;

// Check round 1: a failing create call leaves its output UNDEFINED (Vulkan spec,
// Fundamentals > Return Codes) unless the command says otherwise, so the interposer writes
// this junk value there -- not the VK_NULL_HANDLE that hid a device which destroyed
// whatever the output held.  Every destroy / free below counts a junk handle and does NOT
// forward it (the driver would fault).  vkAllocateCommandBuffers, vkAllocateDescriptorSets
// and vkCreateComputePipelines specify VK_NULL_HANDLE, so they keep writing it.
const uint64_t kP98Junk = 0xBAD0BAD0BAD0ull;
uint32_t gP98JunkDestroyed = 0;

template <class H>
void p98WriteJunk(H* out)
{
    *out = reinterpret_cast<H>(kP98Junk);
}
// True (and counted) when `h` is the junk a failed create left behind.
bool p98IsJunk(uint64_t h)
{
    if (h != kP98Junk)
        return false;
    ++gP98JunkDestroyed;
    std::fprintf(stderr, "[pipelineRefusal] !! a failed create's undefined output was destroyed\n");
    return true;
}

template <class F>
F p98Next(const char* name)
{
    return reinterpret_cast<F>(dlsym(RTLD_NEXT, name));
}
bool p98Armed(const char* call)
{
    return gP98Fail != nullptr && std::strcmp(gP98Fail, call) == 0;
}
// p98Armed with a countdown: the first gP98Skip matching calls go through.
bool p98Fire(const char* call)
{
    if (!p98Armed(call))
        return false;
    if (gP98Skip > 0)
    {
        --gP98Skip;
        return false;
    }
    return true;
}
void p98Made(uint64_t h)
{
    if (gP98Track && h)
    {
        gP98Live.insert(h);
        ++gP98Created;
    }
}
void p98Gone(uint64_t h)
{
    if (gP98Track)
        gP98Live.erase(h);
}

// P9.99: what the LAST vkCreateComputePipelines call asked the driver for, recorded armed or
// not.  llvmpipe runs full 8-lane subgroups at x = 64 with or without the request, so only
// this record shows a dropped VkPipelineShaderStageRequiredSubgroupSizeCreateInfo, a wrong
// size in it, or a missing REQUIRE_FULL_SUBGROUPS flag.
struct P99Stage
{
    uint32_t calls = 0;
    VkPipelineShaderStageCreateFlags flags = 0;
    bool chained = false;   // a required-subgroup-size struct on the stage's pNext chain
    uint32_t size = 0;      // its requiredSubgroupSize
};
P99Stage gP99Stage;
} // namespace

// One create / destroy pair: a failure writes junk and returns `err`; a success is
// tracked; a destroy of the junk is counted and swallowed.
#define P98_CREATE(Name, Handle, err, ...)                                                     \
    {                                                                                          \
        static const auto real = p98Next<PFN_##Name>(#Name);                                   \
        if (p98Fire(#Name))                                                                    \
        {                                                                                      \
            p98WriteJunk(out);                                                                 \
            return err;                                                                        \
        }                                                                                      \
        const VkResult r = real(__VA_ARGS__);                                                  \
        if (r == VK_SUCCESS)                                                                   \
            p98Made(uint64_t(*out));                                                           \
        return r;                                                                              \
    }
#define P98_DESTROY(Name, h, ...)                                                              \
    {                                                                                          \
        static const auto real = p98Next<PFN_##Name>(#Name);                                   \
        if (p98IsJunk(uint64_t(h)))                                                            \
            return;                                                                            \
        p98Gone(uint64_t(h));                                                                  \
        real(__VA_ARGS__);                                                                     \
    }

extern "C" {
VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(VkDevice d, const VkShaderModuleCreateInfo* ci,
                                                    const VkAllocationCallbacks* a, VkShaderModule* out)
P98_CREATE(vkCreateShaderModule, VkShaderModule, VK_ERROR_OUT_OF_HOST_MEMORY, d, ci, a, out)
VKAPI_ATTR void VKAPI_CALL vkDestroyShaderModule(VkDevice d, VkShaderModule m, const VkAllocationCallbacks* a)
P98_DESTROY(vkDestroyShaderModule, m, d, m, a)
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorSetLayout(VkDevice d, const VkDescriptorSetLayoutCreateInfo* ci,
                                                           const VkAllocationCallbacks* a,
                                                           VkDescriptorSetLayout* out)
P98_CREATE(vkCreateDescriptorSetLayout, VkDescriptorSetLayout, VK_ERROR_OUT_OF_HOST_MEMORY, d, ci, a, out)
VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorSetLayout(VkDevice d, VkDescriptorSetLayout l,
                                                        const VkAllocationCallbacks* a)
P98_DESTROY(vkDestroyDescriptorSetLayout, l, d, l, a)
VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineLayout(VkDevice d, const VkPipelineLayoutCreateInfo* ci,
                                                      const VkAllocationCallbacks* a, VkPipelineLayout* out)
P98_CREATE(vkCreatePipelineLayout, VkPipelineLayout, VK_ERROR_OUT_OF_HOST_MEMORY, d, ci, a, out)
VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineLayout(VkDevice d, VkPipelineLayout l, const VkAllocationCallbacks* a)
P98_DESTROY(vkDestroyPipelineLayout, l, d, l, a)
// Three ways to fail: an error with no handle (what the spec asks of a driver), an
// error with a handle written anyway (it must still be destroyed), and success with no
// handle (it must still be refused).
VKAPI_ATTR VkResult VKAPI_CALL vkCreateComputePipelines(VkDevice d, VkPipelineCache c, uint32_t n,
                                                        const VkComputePipelineCreateInfo* ci,
                                                        const VkAllocationCallbacks* a, VkPipeline* out)
{
    static const auto real = p98Next<PFN_vkCreateComputePipelines>("vkCreateComputePipelines");
    ++gP99Stage.calls;
    gP99Stage.flags = n > 0 ? ci[0].stage.flags : 0;
    gP99Stage.chained = false;
    gP99Stage.size = 0;
    for (const VkBaseInStructure* s = n > 0 ? static_cast<const VkBaseInStructure*>(ci[0].stage.pNext) : nullptr; s;
         s = s->pNext)
        if (s->sType == VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO)
        {
            gP99Stage.chained = true;
            gP99Stage.size =
                reinterpret_cast<const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo*>(s)->requiredSubgroupSize;
        }
    if (p98Armed("vkCreateComputePipelines") || p98Armed("vkCreateComputePipelines+null"))
    {
        for (uint32_t i = 0; i < n; ++i)
            out[i] = VK_NULL_HANDLE;
        return p98Armed("vkCreateComputePipelines") ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS;
    }
    const VkResult r = real(d, c, n, ci, a, out);
    if (r == VK_SUCCESS)
        for (uint32_t i = 0; i < n; ++i)
            p98Made(uint64_t(out[i]));
    return p98Armed("vkCreateComputePipelines+handle") && r == VK_SUCCESS ? VK_ERROR_UNKNOWN : r;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyPipeline(VkDevice d, VkPipeline p, const VkAllocationCallbacks* a)
P98_DESTROY(vkDestroyPipeline, p, d, p, a)

// ---- check round 0: buffers, memory, images, descriptors, init ----
VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(VkDevice d, const VkBufferCreateInfo* ci,
                                              const VkAllocationCallbacks* a, VkBuffer* out)
P98_CREATE(vkCreateBuffer, VkBuffer, VK_ERROR_OUT_OF_DEVICE_MEMORY, d, ci, a, out)
VKAPI_ATTR void VKAPI_CALL vkDestroyBuffer(VkDevice d, VkBuffer b, const VkAllocationCallbacks* a)
P98_DESTROY(vkDestroyBuffer, b, d, b, a)
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(VkDevice d, const VkMemoryAllocateInfo* ai,
                                                const VkAllocationCallbacks* a, VkDeviceMemory* out)
P98_CREATE(vkAllocateMemory, VkDeviceMemory, VK_ERROR_OUT_OF_DEVICE_MEMORY, d, ai, a, out)
VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks* a)
P98_DESTROY(vkFreeMemory, m, d, m, a)
VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory(VkDevice d, VkBuffer b, VkDeviceMemory m, VkDeviceSize o)
{
    static const auto real = p98Next<PFN_vkBindBufferMemory>("vkBindBufferMemory");
    if (p98Fire("vkBindBufferMemory"))
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    return real(d, b, m, o);
}
VKAPI_ATTR VkResult VKAPI_CALL vkMapMemory(VkDevice d, VkDeviceMemory m, VkDeviceSize o, VkDeviceSize n,
                                           VkMemoryMapFlags f, void** out)
{
    static const auto real = p98Next<PFN_vkMapMemory>("vkMapMemory");
    if (p98Fire("vkMapMemory"))
    {
        p98WriteJunk(out);   // ppData is undefined on failure too
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    return real(d, m, o, n, f, out);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateImage(VkDevice d, const VkImageCreateInfo* ci,
                                             const VkAllocationCallbacks* a, VkImage* out)
P98_CREATE(vkCreateImage, VkImage, VK_ERROR_OUT_OF_DEVICE_MEMORY, d, ci, a, out)
VKAPI_ATTR void VKAPI_CALL vkDestroyImage(VkDevice d, VkImage i, const VkAllocationCallbacks* a)
P98_DESTROY(vkDestroyImage, i, d, i, a)
VKAPI_ATTR VkResult VKAPI_CALL vkBindImageMemory(VkDevice d, VkImage i, VkDeviceMemory m, VkDeviceSize o)
{
    static const auto real = p98Next<PFN_vkBindImageMemory>("vkBindImageMemory");
    if (p98Fire("vkBindImageMemory"))
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    return real(d, i, m, o);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(VkDevice d, const VkImageViewCreateInfo* ci,
                                                 const VkAllocationCallbacks* a, VkImageView* out)
P98_CREATE(vkCreateImageView, VkImageView, VK_ERROR_OUT_OF_HOST_MEMORY, d, ci, a, out)
VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(VkDevice d, VkImageView v, const VkAllocationCallbacks* a)
P98_DESTROY(vkDestroyImageView, v, d, v, a)
// Check round 1: the bound texture's sampler (createSamplerCounted).
VKAPI_ATTR VkResult VKAPI_CALL vkCreateSampler(VkDevice d, const VkSamplerCreateInfo* ci,
                                               const VkAllocationCallbacks* a, VkSampler* out)
P98_CREATE(vkCreateSampler, VkSampler, VK_ERROR_OUT_OF_HOST_MEMORY, d, ci, a, out)
VKAPI_ATTR void VKAPI_CALL vkDestroySampler(VkDevice d, VkSampler smp, const VkAllocationCallbacks* a)
P98_DESTROY(vkDestroySampler, smp, d, smp, a)
// "vkAllocateDescriptorSets": a hard failure.  "pools-exhausted": every pool reports
// OUT_OF_POOL_MEMORY, a fresh one included (the device used to loop forever creating
// pools).  "pools-exhausted+create": that, and the new pool cannot be created either.
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorPool(VkDevice d, const VkDescriptorPoolCreateInfo* ci,
                                                      const VkAllocationCallbacks* a, VkDescriptorPool* out)
{
    static const auto real = p98Next<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool");
    if (p98Armed("pools-exhausted+create"))
    {
        p98WriteJunk(out);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    return real(d, ci, a, out);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorPool(VkDevice d, VkDescriptorPool pool, const VkAllocationCallbacks* a)
P98_DESTROY(vkDestroyDescriptorPool, pool, d, pool, a)
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateDescriptorSets(VkDevice d, const VkDescriptorSetAllocateInfo* ai,
                                                        VkDescriptorSet* out)
{
    static const auto real = p98Next<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets");
    // A pool the device failed to create must never be used (VUID-VkDescriptorSetAllocateInfo-
    // descriptorPool-parameter); llvmpipe tolerates a null one, so it is counted here, armed
    // or not -- and the junk one is never forwarded.
    if (ai->descriptorPool == VK_NULL_HANDLE || uint64_t(ai->descriptorPool) == kP98Junk)
    {
        ++gP98NullPool;
        if (uint64_t(ai->descriptorPool) == kP98Junk)
            return VK_ERROR_OUT_OF_POOL_MEMORY;
    }
    if (p98Armed("vkAllocateDescriptorSets") || p98Armed("pools-exhausted") || p98Armed("pools-exhausted+create"))
    {
        for (uint32_t i = 0; i < ai->descriptorSetCount; ++i)
            out[i] = VK_NULL_HANDLE;
        return p98Armed("vkAllocateDescriptorSets") ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_ERROR_OUT_OF_POOL_MEMORY;
    }
    return real(d, ai, out);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateFence(VkDevice d, const VkFenceCreateInfo* ci, const VkAllocationCallbacks* a,
                                             VkFence* out)
P98_CREATE(vkCreateFence, VkFence, VK_ERROR_OUT_OF_HOST_MEMORY, d, ci, a, out)
VKAPI_ATTR void VKAPI_CALL vkDestroyFence(VkDevice d, VkFence f, const VkAllocationCallbacks* a)
P98_DESTROY(vkDestroyFence, f, d, f, a)
// Check round 1: the other init steps -- command pool, command buffers, pipeline cache.
VKAPI_ATTR VkResult VKAPI_CALL vkCreateCommandPool(VkDevice d, const VkCommandPoolCreateInfo* ci,
                                                   const VkAllocationCallbacks* a, VkCommandPool* out)
P98_CREATE(vkCreateCommandPool, VkCommandPool, VK_ERROR_OUT_OF_HOST_MEMORY, d, ci, a, out)
VKAPI_ATTR void VKAPI_CALL vkDestroyCommandPool(VkDevice d, VkCommandPool cp, const VkAllocationCallbacks* a)
P98_DESTROY(vkDestroyCommandPool, cp, d, cp, a)
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateCommandBuffers(VkDevice d, const VkCommandBufferAllocateInfo* ai,
                                                        VkCommandBuffer* out)
{
    static const auto real = p98Next<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers");
    if (p98Fire("vkAllocateCommandBuffers"))
    {
        for (uint32_t i = 0; i < ai->commandBufferCount; ++i)
            out[i] = VK_NULL_HANDLE;   // specified: every entry NULL on failure
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    return real(d, ai, out);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineCache(VkDevice d, const VkPipelineCacheCreateInfo* ci,
                                                     const VkAllocationCallbacks* a, VkPipelineCache* out)
P98_CREATE(vkCreatePipelineCache, VkPipelineCache, VK_ERROR_OUT_OF_HOST_MEMORY, d, ci, a, out)
VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineCache(VkDevice d, VkPipelineCache pc, const VkAllocationCallbacks* a)
P98_DESTROY(vkDestroyPipelineCache, pc, d, pc, a)
} // extern "C"

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgResult;
using physx::gpuslang::PxgSpirvEntryInfo;
using physx::gpuslang::pxgSpirvReflectEntry;
using physx::gpuslang::pxgRequiredSubgroupSizeFits;

std::vector<uint8_t> refusalLoader(const char* f)
{
    return loadShaderBlob(f);
}

PxgSpirvEntryInfo reflect(const char* file, const char* entry, bool* ok)
{
    const std::vector<uint8_t> blob = loadShaderBlob(file);
    PxgSpirvEntryInfo info;
    *ok = pxgSpirvReflectEntry(blob.data(), blob.size(), entry, &info);
    return info;
}

PxgPipelineHandle build(PxgSlangComputeDevice* dev, const char* file, const char* entry, uint32_t push)
{
    const std::vector<uint8_t> blob = loadShaderBlob(file);
    return dev->createPipeline(blob.data(), blob.size(), entry, 0, nullptr, push);
}

PxgSlangComputeDevice* limitedDevice(uint32_t push, uint32_t shared, uint32_t invocations, uint32_t size = 0,
                                     uint32_t count = 0)
{
    PxgSlangDeviceDesc desc;
    const char* preferSw = std::getenv("PXG_PREFER_SOFTWARE_DEVICE");
    desc.preferSoftwareDevice = preferSw == nullptr || std::strcmp(preferSw, "0") != 0;
    desc.useDeviceLocalMemory = physx::gpuslang::pxgDeviceV2FromEnv();
    desc.limitMaxPushConstantsSize = push;
    desc.limitMaxComputeSharedMemorySize = shared;
    desc.limitMaxComputeWorkGroupInvocations = invocations;
    desc.limitMaxComputeWorkGroupSize = size;
    desc.limitMaxComputeWorkGroupCount = count;
    return physx::gpuslang::PxgCreateVulkanComputeDevice(desc);
}

// Check round 1: the physical device's own VkPhysicalDeviceLimits, read independently of the
// device under test (its own instance), for the device named `name`; false if not found.
bool physicalLimits(const char* name, VkPhysicalDeviceLimits* out)
{
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ci = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ci.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS)
        return false;
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    if (n)
        vkEnumeratePhysicalDevices(inst, &n, pds.data());
    bool found = false;
    for (VkPhysicalDevice pd : pds)
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);
        if (!found && std::strcmp(props.deviceName, name) == 0)
        {
            *out = props.limits;
            found = true;
        }
    }
    vkDestroyInstance(inst, nullptr);
    return found;
}

// P9.99: a device whose SubgroupCaps the desc hooks have lowered, in the suite's memory mode
// (the hooks only take a capability away; the features stay enabled).
PxgSlangComputeDevice* subgroupDevice(bool noSizeControl, bool noFullSubgroups, uint32_t maxWorkgroupSubgroups)
{
    PxgSlangDeviceDesc desc;
    const char* preferSw = std::getenv("PXG_PREFER_SOFTWARE_DEVICE");
    desc.preferSoftwareDevice = preferSw == nullptr || std::strcmp(preferSw, "0") != 0;
    desc.useDeviceLocalMemory = physx::gpuslang::pxgDeviceV2FromEnv();
    desc.noSubgroupSizeControl = noSizeControl;
    desc.noFullSubgroups = noFullSubgroups;
    desc.limitMaxComputeWorkgroupSubgroups = maxWorkgroupSubgroups;
    return physx::gpuslang::PxgCreateVulkanComputeDevice(desc);
}

// P9.99: the physical device's own subgroup properties and subgroup-size-control
// properties / features, read through a separate Vulkan 1.3 instance (as physicalLimits),
// for the device named `name`; false if not found.
struct PhysicalSubgroup
{
    uint32_t subgroupSize = 0;
    VkPhysicalDeviceSubgroupSizeControlProperties props = {};
    VkPhysicalDeviceSubgroupSizeControlFeatures feats = {};
    // What SubgroupCaps must say (the device's init reads exactly these).
    bool canRequire() const
    {
        return feats.subgroupSizeControl == VK_TRUE && (props.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT);
    }
    bool full() const { return feats.computeFullSubgroups == VK_TRUE; }
    // The numbers the device rows below are written for (llvmpipe, 256-bit).
    bool isLlvmpipeShape() const
    {
        return props.minSubgroupSize == 8 && props.maxSubgroupSize == 8 && canRequire() && full() &&
               props.maxComputeWorkgroupSubgroups == 32;
    }
};
bool physicalSubgroupControl(const char* name, PhysicalSubgroup* out)
{
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ci = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ci.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS)
        return false;
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    if (n)
        vkEnumeratePhysicalDevices(inst, &n, pds.data());
    bool found = false;
    for (VkPhysicalDevice pd : pds)
    {
        VkPhysicalDeviceSubgroupSizeControlProperties sizeProps = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES
        };
        VkPhysicalDeviceSubgroupProperties sgProps = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
        sgProps.pNext = &sizeProps;
        VkPhysicalDeviceProperties2 props2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        props2.pNext = &sgProps;
        vkGetPhysicalDeviceProperties2(pd, &props2);
        if (found || std::strcmp(props2.properties.deviceName, name) != 0)
            continue;
        VkPhysicalDeviceSubgroupSizeControlFeatures sizeFeats = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES
        };
        VkPhysicalDeviceFeatures2 feats2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        feats2.pNext = &sizeFeats;
        vkGetPhysicalDeviceFeatures2(pd, &feats2);
        out->subgroupSize = sgProps.subgroupSize;
        out->props = sizeProps;
        out->props.pNext = nullptr;
        out->feats = sizeFeats;
        out->feats.pNext = nullptr;
        found = true;
    }
    vkDestroyInstance(inst, nullptr);
    return found;
}

// Check round 1: sets an environment variable for its lifetime and then RESTORES what the
// caller had (unsetenv only when it was unset), so a fixture never clobbers a user's hook.
struct ScopedEnv
{
    std::string name, old;
    bool had;
    ScopedEnv(const char* n, const char* v) : name(n)
    {
        const char* o = std::getenv(n);
        had = o != nullptr;
        if (had)
            old = o;
        setenv(n, v, 1);
    }
    ~ScopedEnv()
    {
        if (had)
            setenv(name.c_str(), old.c_str(), 1);
        else
            unsetenv(name.c_str());
    }
};

// Check round 1: what `fn` writes to stderr (through a temporary file), so a fixture can
// pin a diagnostic line.
template <class F>
std::string captureStderr(F fn)
{
    std::fflush(stderr);
    FILE* tmp = std::tmpfile();
    if (!tmp)
    {
        fn();
        return "<no tmpfile>";
    }
    const int saved = dup(fileno(stderr));
    dup2(fileno(tmp), fileno(stderr));
    fn();
    std::fflush(stderr);
    dup2(saved, fileno(stderr));
    close(saved);
    std::string out;
    std::rewind(tmp);
    char buf[512];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), tmp)) > 0)
        out.append(buf, n);
    std::fclose(tmp);
    std::fputs(out.c_str(), stderr);   // still in the test log
    return out;
}

// P9.99: createPipelineEx on p99SubgroupShape.spv (one 16-byte storage buffer, a 4-byte push
// block), with the stderr it wrote.
PxgPipelineHandle p99BuildEx(PxgSlangComputeDevice* dev, const std::vector<uint8_t>& blob, const char* entry,
                             uint32_t size, std::string* message)
{
    const uint32_t stride = 16;
    PxgPipelineHandle h = 0;
    const std::string m =
        captureStderr([&] { h = dev->createPipelineEx(blob.data(), blob.size(), entry, 1, &stride, 4, size); });
    if (message)
        *message = m;
    return h;
}

// P9.99: runs `pipe` (`invocations` per workgroup) over 3 workgroups and counts the threads
// NOT in a full 8-lane subgroup: lane count != 8, WavePrefixSum(1) != lane index, or
// ballot / WaveActiveSum(1) != the lane count -- or not written at all.  `ballot` (if
// non-null) gets the first thread's ballot count.
uint32_t p99NotFullThreads(PxgSlangComputeDevice* dev, PxgPipelineHandle pipe, uint32_t invocations,
                           uint32_t* ballot = nullptr)
{
    struct SgOut
    {
        uint32_t x, y, z, w;
    };
    const uint32_t groups = 3, n = groups * invocations;
    const std::vector<SgOut> init(n, SgOut{ 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu });
    PxgBufferHandle b = createBufferFrom(dev, init);
    const uint32_t numThreads = n;
    dev->dispatch(pipe, groups, 1, 1, &b, 1, &numThreads, sizeof(numThreads));
    dev->flush();
    const std::vector<SgOut> g = downloadBuffer<SgOut>(dev, b, n);
    dev->destroyBuffer(b);
    uint32_t notFull = 0;
    for (const SgOut& o : g)
    {
        const uint32_t lane = o.x & 0xffffu, count = o.x >> 16;
        if (o.x == 0xffffffffu || count != 8u || o.y != lane || o.z != count || o.w != count)
            ++notFull;
    }
    if (ballot)
        *ballot = g.empty() ? 0u : g[0].z;
    return notFull;
}

// Check round 1: a hand-assembled SPIR-V 1.5 module whose one GLCompute entry "main"
// (64 x 1 x 1) lists one variable per entry of `lens`: a float[len] in Workgroup storage,
// or -- `push` -- a struct { float[len] } push block.  Only reflection and the limit
// checks ever see it: they refuse it before any Vulkan call (it has no function body).
// The walker summed sizes in uint32, so 4 * 2^30 wrapped to 0 and was BUILT (then the
// device crashed at the flush, measured rc 139).
std::vector<uint32_t> hugeArrayModule(const std::vector<uint32_t>& lens, bool push)
{
    const uint32_t n = uint32_t(lens.size());
    // ids: 1 fn, 2 float, 3 uint, then per variable: const, array, [struct], pointer, var
    const uint32_t per = push ? 5u : 4u;
    const uint32_t bound = 4u + per * n;
    std::vector<uint32_t> w = { 0x07230203u, 0x00010500u, 0u, bound, 0u };
    w.push_back(((5u + n) << 16) | 15u);   // OpEntryPoint GLCompute %1 "main" %vars...
    w.push_back(5u);
    w.push_back(1u);
    w.push_back(0x6e69616du);   // "main"
    w.push_back(0u);
    for (uint32_t i = 0; i < n; ++i)
        w.push_back(4u + per * i + per - 1u);
    for (uint32_t x : { (6u << 16) | 16u, 1u, 17u, 64u, 1u, 1u })   // OpExecutionMode LocalSize
        w.push_back(x);
    for (uint32_t x : { (3u << 16) | 22u, 2u, 32u, (4u << 16) | 21u, 3u, 32u, 0u })   // float, uint
        w.push_back(x);
    for (uint32_t i = 0; i < n; ++i)
    {
        const uint32_t c = 4u + per * i, arr = c + 1u, st = c + 2u, ptr = c + per - 2u, var = c + per - 1u;
        const uint32_t storage = push ? 9u : 4u;   // PushConstant / Workgroup
        for (uint32_t x : { (4u << 16) | 43u, 3u, c, lens[i], (4u << 16) | 28u, arr, 2u, c })
            w.push_back(x);
        if (push)
            for (uint32_t x : { (3u << 16) | 30u, st, arr, (5u << 16) | 72u, st, 0u, 35u, 0u })
                w.push_back(x);   // OpTypeStruct { arr }, OpMemberDecorate Offset 0
        for (uint32_t x : { (4u << 16) | 32u, ptr, storage, push ? st : arr, (4u << 16) | 59u, ptr, var, storage })
            w.push_back(x);
    }
    return w;
}

struct PushPadParams
{
    uint64_t out;
    uint32_t head0, head1;
    uint32_t tail[12];
};
static_assert(sizeof(PushPadParams) == 64, "p98PushPad.slang's block");

// p98PushPad.spv with two stray bytes: a WELL-FORMED module followed by a ragged tail, so
// only the codeSize % 4 check can refuse it (a short or junk blob is refused earlier).
std::vector<uint8_t> raggedPushPad()
{
    std::vector<uint8_t> blob = loadShaderBlob("p98PushPad.spv");
    blob.push_back(0);
    blob.push_back(0);
    return blob;
}

// A 5-word header, OpEntryPoint GLCompute %1 "main", then an OpTypeBool whose word count
// is 1: the walker used to read past the instruction and assign a reversed range
// (std::length_error, i.e. abort) instead of returning false.
std::vector<uint32_t> truncatedTypeModule()
{
    return { 0x07230203u, 0x00010500u, 0u, 16u, 0u,
             (5u << 16) | 15u, 5u, 1u, 0x6e69616du /* "main" */, 0u,
             (1u << 16) | 20u, (1u << 16) | 20u };
}

} // namespace

// The walker's numbers for the entries the validation layer flagged.
PXG_TEST(pipelineRefusal_reflect)
{
    struct Row
    {
        const char* file;
        const char* entry;
        uint32_t push, workgroup;
    };
    const Row rows[] = {
        // F1: 188 before P9.98 deleted npSdfFind's three stray pads; every fixture pushes 176.
        { "npSdfFind.spv", "sdfFindRounds", 176, 8324 },
        { "npSdfFind.spv", "sdfFindTriangleRounds", 176, 12420 },
        { "npSdfFind.spv", "sdfFindTetRounds", 176, 8324 },
        { "npSdfCollision.spv", "sdfCollVertex", 64, 0 },   // testSdfParticle pushed 48 until P9.98, 64 since
        // F2: the engine rows push 104 and 16.
        // 5.12: the module's block grew to 192 (PxgRigidParticleContactWriter is 80 B @112; 5.6.1: 176).
        { "psContactGenBda.spv", "ps_primitivesDiffuseCollisionLaunch", 192, 0 },
        { "psContactGenBda.spv", "ps_convexDiffuseCollisionLaunch", 192, 0 },
        { "psDiffuseBuffers.spv", "ps_updateUnsortedDiffuseArrayLaunch", 24, 4096 },
        // F3: per ENTRY, not the module's 40448 the layer reports for every entry.
        { "solverTGS.spv", "solveWholeIslandTGS", 32, 30720 },   // 5.12: the 1024 B residual-reduction scratch went with residual reporting (5.6.1: 31744)
        { "solverTGS.spv", "writeBackBodiesTGS", 24, 0 },
        { "p98PushPad.spv", "pushPadEcho", 64, 0 },
    };
    for (const Row& r : rows)
    {
        bool ok = false;
        const PxgSpirvEntryInfo info = reflect(r.file, r.entry, &ok);
        PXG_CHECK_MSG(ok && info.perEntry && info.pushDecided && info.workgroupDecided,
                      "%s:%s must reflect per entry (ok %d)", r.file, r.entry, int(ok));
        PXG_CHECK_MSG(info.pushBlockBytes == r.push, "%s:%s push block %u B, expected %u", r.file, r.entry,
                      info.pushBlockBytes, r.push);
        PXG_CHECK_MSG(info.workgroupBytes == r.workgroup, "%s:%s Workgroup %u B, expected %u", r.file, r.entry,
                      info.workgroupBytes, r.workgroup);
    }
    bool ok = false;
    const PxgSpirvEntryInfo ls = reflect("solverTGS.spv", "solveWholeIslandTGS", &ok);
    PXG_CHECK_MSG(ls.localSize[0] == 256 && ls.localSize[1] == 1 && ls.localSize[2] == 1,
                  "solveWholeIslandTGS local size (%u, %u, %u)", ls.localSize[0], ls.localSize[1], ls.localSize[2]);
    const PxgSpirvEntryInfo ly = reflect("p98PushPad.spv", "pushPadDimY", &ok);
    PXG_CHECK_MSG(ok && ly.localSize[0] == 1 && ly.localSize[1] == 2 && ly.localSize[2] == 1,
                  "pushPadDimY local size (%u, %u, %u)", ly.localSize[0], ly.localSize[1], ly.localSize[2]);
    const PxgSpirvEntryInfo lz = reflect("p98PushPad.spv", "pushPadDimZ", &ok);
    PXG_CHECK_MSG(ok && lz.localSize[0] == 1 && lz.localSize[1] == 1 && lz.localSize[2] == 2,
                  "pushPadDimZ local size (%u, %u, %u)", lz.localSize[0], lz.localSize[1], lz.localSize[2]);

    reflect("solverTGS.spv", "noSuchEntryP98", &ok);
    PXG_CHECK_MSG(!ok, "a missing entry point must not reflect");
    const uint32_t junk[8] = { 0xdeadbeefu, 1, 2, 3, 4, 5, 6, 7 };
    PxgSpirvEntryInfo info;
    PXG_CHECK_MSG(!pxgSpirvReflectEntry(junk, sizeof(junk), "x", &info), "a non-SPIR-V blob must not reflect");
    PXG_CHECK_MSG(!pxgSpirvReflectEntry(junk, 6, "x", &info), "a short ragged blob must not reflect");
    const std::vector<uint8_t> ragged = raggedPushPad();
    PXG_CHECK_MSG(pxgSpirvReflectEntry(ragged.data(), ragged.size() - 2, "pushPadEcho", &info),
                  "the module without its ragged tail reflects (control)");
    PXG_CHECK_MSG(!pxgSpirvReflectEntry(ragged.data(), ragged.size(), "pushPadEcho", &info),
                  "a well-formed module with a 2-byte ragged tail must not reflect");
    const std::vector<uint32_t> trunc = truncatedTypeModule();
    PXG_CHECK_MSG(!pxgSpirvReflectEntry(trunc.data(), trunc.size() * 4, "main", &info),
                  "a type instruction of word count 1 is malformed: false, not an abort");

    // Check round 1: sizes past 4 GiB saturate at UINT32_MAX (decided, so over every
    // limit) instead of wrapping to a small number.  The control is an honest 4096 B.
    struct Huge
    {
        const char* what;
        std::vector<uint32_t> lens;
        bool push;
        uint32_t expect;
    };
    const Huge huge[] = {
        { "float[1024] Workgroup (control)", { 1024u }, false, 4096u },
        { "float[0x40000000] Workgroup (4 GiB, wrapped to 0)", { 0x40000000u }, false, 0xFFFFFFFFu },
        { "two float[0x20000000] Workgroup (2 x 2 GiB, the SUM wrapped to 0)", { 0x20000000u, 0x20000000u }, false,
          0xFFFFFFFFu },
        { "float[0x40000001] Workgroup (wrapped to 4)", { 0x40000001u }, false, 0xFFFFFFFFu },
        { "struct { float[16] } push block (control)", { 16u }, true, 64u },
        { "struct { float[0x40000000] } push block (wrapped to 0)", { 0x40000000u }, true, 0xFFFFFFFFu },
    };
    for (const Huge& h : huge)
    {
        const std::vector<uint32_t> m = hugeArrayModule(h.lens, h.push);
        PxgSpirvEntryInfo hi;
        const bool hok = pxgSpirvReflectEntry(m.data(), m.size() * 4, "main", &hi);
        const uint32_t got = h.push ? hi.pushBlockBytes : hi.workgroupBytes;
        const bool decided = h.push ? hi.pushDecided : hi.workgroupDecided;
        PXG_CHECK_MSG(hok && decided && got == h.expect, "%s: reflect %d, decided %d, %u B (expected %u)", h.what,
                      int(hok), int(decided), got, h.expect);
    }
}

// F4: a pipeline that cannot be built is a 0 handle, and a 0 / unknown handle is a
// refused dispatch -- no crash, nothing recorded, and the device keeps working.
PXG_TEST(pipelineRefusal_badEntryAndModule)
{
    PxgSlangComputeDevice* dev = testDevice();

    std::fprintf(stderr, "[pipelineRefusal] the [PxgVulkan] refusals below are EXPECTED\n");
    const PxgPipelineHandle missing = build(dev, "solverTGS.spv", "noSuchEntryP98", 24);
    PXG_CHECK_MSG(missing == 0, "a missing entry point must be refused, got handle %u", missing);

    const uint32_t junk[16] = { 0xdeadbeefu };
    PXG_CHECK_MSG(dev->createPipeline(junk, sizeof(junk), "x", 0, nullptr, 0) == 0, "a non-SPIR-V module");
    PXG_CHECK_MSG(dev->createPipeline(junk, 6, "x", 0, nullptr, 0) == 0, "a short ragged module");
    PXG_CHECK_MSG(dev->createPipeline(nullptr, 0, "x", 0, nullptr, 0) == 0, "no module at all");
    // A well-formed module + 2 bytes would otherwise reach vkCreateShaderModule with a
    // codeSize that is not a multiple of 4 (VUID-VkShaderModuleCreateInfo-codeSize-08735).
    const std::vector<uint8_t> ragged = raggedPushPad();
    PXG_CHECK_MSG(dev->createPipeline(ragged.data(), ragged.size(), "pushPadEcho", 0, nullptr, 64) == 0,
                  "a well-formed module with a ragged tail must be refused");
    const std::vector<uint32_t> trunc = truncatedTypeModule();
    PXG_CHECK_MSG(dev->createPipeline(trunc.data(), trunc.size() * 4, "main", 0, nullptr, 0) == 0,
                  "a module with a truncated type instruction must be refused");
    if (dev->supportsBindlessTextures())
    {
        const std::vector<uint8_t> sdf = loadShaderBlob("npSdfFind.spv");
        PXG_CHECK_MSG(dev->createPipelineBindless(sdf.data(), sdf.size(), "noSuchEntryP98", 0, nullptr, 176) == 0,
                      "the bindless door refuses a missing entry too");
    }

    uint32_t blob[16] = {};
    PXG_CHECK_MSG(!dev->dispatch(0, 1, 1, 1, nullptr, 0, blob, 16), "dispatch of handle 0 must be refused");
    PXG_CHECK_MSG(!dev->dispatch(0x7ffffff0u, 1, 1, 1, nullptr, 0, blob, 16), "an unknown handle must be refused");
    // ~0u = "not held"; a lookup that default-inserted would now hold a pipeline with a
    // 0-byte range.  Stop here if it did: the no-push dispatches below would then bind
    // a null pipeline, and the fixture should report that, not depend on a driver fault.
    const bool lookupClean = dev->pipelinePushRangeBytes(0x7ffffff0u) == ~0u;
    PXG_CHECK_MSG(lookupClean, "an unknown handle must not have been inserted by the lookup (push range %u)",
                  dev->pipelinePushRangeBytes(0x7ffffff0u));
    if (!lookupClean)
        return;
    // With no push bytes, the size check cannot be what refuses: only the lookup can.
    PXG_CHECK_MSG(!dev->dispatch(0, 1, 1, 1, nullptr, 0, nullptr, 0), "handle 0, no push: refused");
    PXG_CHECK_MSG(!dev->dispatch(0x7ffffff1u, 1, 1, 1, nullptr, 0, nullptr, 0), "unknown handle, no push: refused");
    dev->flush();   // a null pipeline recorded by a broken lookup would fault here at the latest

    // A refused push: more bytes than the pipeline's range.
    const PxgPipelineHandle wb = build(dev, "solverTGS.spv", "writeBackBodiesTGS", 24);
    PXG_CHECK_MSG(wb != 0, "writeBackBodiesTGS must build");
    uint8_t big[256] = {};
    PXG_CHECK_MSG(!dev->dispatch(wb, 1, 1, 1, nullptr, 0, big, 128), "a blob past the push range must be refused");
    if (wb)
        dev->destroyPipeline(wb);
    PXG_CHECK_MSG(!dev->dispatch(wb, 1, 1, 1, nullptr, 0, big, 24), "a destroyed handle must be refused");

    // The ENGINE path: the shim refuses the launch (NotFound), every time, and a good
    // row in the same manifest still runs.
    struct AxpyParams
    {
        uint64_t x, y, out;
        float alpha;
        uint32_t count;
    };
    const PxgKernelDef manifest[] = {
        { "solverTGS.spv", "noSuchEntryP98", 24u, 0u },
        { "npShimAxpy.spv", "shimAxpy", uint32_t(sizeof(AxpyParams)), 0u },
    };
    PxgCudaShim shim(dev, manifest, 2, &refusalLoader);
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, blob, 24) == PxgResult::NotFound, "the shim must refuse (1st)");
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, blob, 24) == PxgResult::NotFound, "the shim must refuse (2nd)");

    const uint32_t N = 100;
    PxgDevicePtr dx = 0, dy = 0, dout = 0;
    shim.memAlloc(&dx, N * 4);
    shim.memAlloc(&dy, N * 4);
    shim.memAlloc(&dout, N * 4);
    std::vector<float> x(N), y(N), out(N, -1.0f);
    for (uint32_t i = 0; i < N; ++i)
    {
        x[i] = float(i);
        y[i] = 0.5f * float(i) + 1.0f;
    }
    shim.memcpyHtoD(dx, x.data(), N * 4);
    shim.memcpyHtoD(dy, y.data(), N * 4);
    AxpyParams prm = { dx, dy, dout, 2.0f, N };
    // The device hands out pipeline handles in sequence, so the shim's axpy pipeline
    // (built lazily by the launch below) is the one after this probe's.
    const PxgPipelineHandle probe = build(dev, "p98PushPad.spv", "pushPadEcho", 64);
    PXG_CHECK_MSG(probe != 0, "the probe pipeline must build");
    dev->destroyPipeline(probe);
    PXG_CHECK_MSG(shim.launchKernel(1, (N + 63) / 64, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "a good row must still launch after the refusals");
    shim.streamSynchronize();
    shim.memcpyDtoH(out.data(), dout, N * 4);
    uint32_t bad = 0;
    for (uint32_t i = 0; i < N; ++i)
        bad += out[i] != 2.0f * x[i] + y[i] ? 1u : 0u;
    PXG_CHECK_MSG(bad == 0, "axpy after the refusals: %u of %u wrong", bad, N);

    // A pipeline the shim holds but the device no longer does: the device refuses the
    // DISPATCH (records nothing), and the shim must report that as a refused launch
    // rather than Success.
    const PxgPipelineHandle axpyPipe = probe + 1;
    PXG_CHECK_MSG(dev->pipelinePushRangeBytes(axpyPipe) == uint32_t(sizeof(AxpyParams)),
                  "handle %u should be the shim's axpy pipeline (push range %u, expected %u)", axpyPipe,
                  dev->pipelinePushRangeBytes(axpyPipe), uint32_t(sizeof(AxpyParams)));
    dev->destroyPipeline(axpyPipe);
    PXG_CHECK_MSG(shim.launchKernel(1, (N + 63) / 64, 1, 1, &prm, sizeof(prm)) == PxgResult::NotFound,
                  "a dispatch the device refuses must be a refused launch (NotFound)");

    // Check round 1: the shim's transfers check the whole RANGE against the allocation
    // (resolve() checked only the start pointer): past the end is InvalidValue, CUDA's
    // answer, and changes nothing; exactly to the end is Success.
    {
        std::vector<float> keep(N, 7.0f);
        PXG_CHECK(shim.memcpyHtoD(dout, keep.data(), N * 4) == PxgResult::Success);
        std::vector<float> junk(2 * N, -9.0f), sink(2 * N, 5.0f);
        PXG_CHECK_MSG(shim.memcpyHtoD(dout + (N - 1) * 4, junk.data(), 8) == PxgResult::InvalidValue,
                      "memcpyHtoD 4 bytes past the end");
        PXG_CHECK_MSG(shim.memcpyDtoH(sink.data(), dout + 4, N * 4) == PxgResult::InvalidValue,
                      "memcpyDtoH 4 bytes past the end");
        PXG_CHECK_MSG(sink[0] == 5.0f, "a refused memcpyDtoH writes nothing");
        PXG_CHECK_MSG(shim.memcpyDtoD(dout, dx + 8, N * 4) == PxgResult::InvalidValue, "memcpyDtoD, source past the end");
        PXG_CHECK_MSG(shim.memcpyDtoD(dout + 8, dx, N * 4) == PxgResult::InvalidValue,
                      "memcpyDtoD, destination past the end");
        PXG_CHECK_MSG(shim.memsetD32(dout + 4, 0u, N) == PxgResult::InvalidValue, "memsetD32 one word past the end");
        // (SIZE_MAX / 4 + 2) words: the byte count wraps to 4 if it is formed before the check.
        PXG_CHECK_MSG(shim.memsetD32(dout, 0u, ~size_t(0) / 4 + 2) == PxgResult::InvalidValue,
                      "memsetD32 whose byte count overflows");
        shim.streamSynchronize();
        std::vector<float> back(N, 0.0f);
        PXG_CHECK(shim.memcpyDtoH(back.data(), dout, N * 4) == PxgResult::Success);   // exactly to the end
        uint32_t changed = 0;
        for (float v : back)
            changed += v != 7.0f ? 1u : 0u;
        PXG_CHECK_MSG(changed == 0, "out-of-range shim transfers must change nothing: %u word(s) changed", changed);
        PXG_CHECK_MSG(shim.memsetD32(dout + 4, 0u, N - 1) == PxgResult::Success, "memsetD32 exactly to the end");
        void* host = nullptr;
        PXG_CHECK(shim.memHostAlloc(&host, 64) == PxgResult::Success);
        const PxgDevicePtr hdev = shim.hostGetDevicePointer(host);
        PXG_CHECK_MSG(shim.memcpyHtoD(hdev + 60, junk.data(), 8) == PxgResult::InvalidValue,
                      "memcpyHtoD past the end of a pinned allocation");
        shim.memFreeHost(host);
    }
    shim.memFree(dx);
    shim.memFree(dy);
    shim.memFree(dout);
}

// Device limits: an entry over a (lowered) limit is refused with a reason, per ENTRY.
PXG_TEST(pipelineRefusal_deviceLimits)
{
    PxgSlangComputeDevice* real = testDevice();
    const PxgSlangComputeDevice::ComputeLimits rl = real->computeLimits();
    std::printf("  device limits: push %u B, shared %u B, invocations %u, size (%u, %u, %u)\n",
                rl.maxPushConstantsSize, rl.maxComputeSharedMemorySize, rl.maxComputeWorkGroupInvocations,
                rl.maxComputeWorkGroupSize[0], rl.maxComputeWorkGroupSize[1], rl.maxComputeWorkGroupSize[2]);
    PXG_CHECK_MSG(rl.maxPushConstantsSize >= 128 && rl.maxComputeSharedMemorySize >= 16384 &&
                      rl.maxComputeWorkGroupInvocations >= 128,
                  "the spec minimums");
    // Check round 1: every stored limit IS the physical device's (no hook is set for
    // testDevice) -- read through a separate instance, so a limit the device never reads
    // (left at "unlimited") cannot pass as a generous one.
    {
        VkPhysicalDeviceLimits pl;
        const bool havePl = physicalLimits(real->getDeviceName(), &pl);
        PXG_CHECK_MSG(havePl, "the physical device %s must be found again", real->getDeviceName());
        if (havePl)
        {
            bool same = rl.maxPushConstantsSize == pl.maxPushConstantsSize &&
                        rl.maxComputeSharedMemorySize == pl.maxComputeSharedMemorySize &&
                        rl.maxComputeWorkGroupInvocations == pl.maxComputeWorkGroupInvocations;
            for (int d = 0; d < 3; ++d)
                same = same && rl.maxComputeWorkGroupSize[d] == pl.maxComputeWorkGroupSize[d] &&
                       rl.maxComputeWorkGroupCount[d] == pl.maxComputeWorkGroupCount[d];
            PXG_CHECK_MSG(same, "computeLimits() must be the physical device's (count %u vs %u, size %u vs %u)",
                          rl.maxComputeWorkGroupCount[0], pl.maxComputeWorkGroupCount[0],
                          rl.maxComputeWorkGroupSize[0], pl.maxComputeWorkGroupSize[0]);
        }
    }
    if (rl.maxPushConstantsSize < 176 || rl.maxComputeSharedMemorySize < 30720 ||
        rl.maxComputeWorkGroupInvocations < 1024 || rl.maxComputeWorkGroupSize[0] < 1024)
    {
        std::printf("  NOTE: this device's limits are below the probe kernels' -> skip the lowered-limit half\n");
        return;
    }
    // On the real device every probe builds.
    const char* probes[][2] = { { "solverTGS.spv", "solveWholeIslandTGS" },
                                { "psContactGenBda.spv", "ps_primitivesDiffuseCollisionLaunch" },
                                { "aggregateBda.spv", "addAggPairsStage1" } };
    const uint32_t probePush[] = { 32, 104, 16 };
    for (int i = 0; i < 3; ++i)
    {
        const PxgPipelineHandle h = build(real, probes[i][0], probes[i][1], probePush[i]);
        PXG_CHECK_MSG(h != 0, "%s must build on the real device", probes[i][1]);
        if (h)
            real->destroyPipeline(h);
    }

    std::fprintf(stderr, "[pipelineRefusal] the [PxgVulkan] limit refusals below are EXPECTED\n");
    PxgSlangComputeDevice* dev = limitedDevice(128, 16384, 256);
    PXG_CHECK_MSG(dev != nullptr, "the lowered-limit device must create");
    if (!dev)
        return;
    const PxgSlangComputeDevice::ComputeLimits l = dev->computeLimits();
    PXG_CHECK_MSG(l.maxPushConstantsSize == 128 && l.maxComputeSharedMemorySize == 16384 &&
                      l.maxComputeWorkGroupInvocations == 256,
                  "the hooks must lower the limits (push %u, shared %u, invocations %u)", l.maxPushConstantsSize,
                  l.maxComputeSharedMemorySize, l.maxComputeWorkGroupInvocations);

    // Shared memory: 30720 B > 16384 refused; its module sibling with 0 B builds.
    PXG_CHECK_MSG(build(dev, "solverTGS.spv", "solveWholeIslandTGS", 32) == 0,
                  "30720 B of groupshared must be refused under a 16384 B limit");
    const PxgPipelineHandle wb = build(dev, "solverTGS.spv", "writeBackBodiesTGS", 24);
    PXG_CHECK_MSG(wb != 0, "writeBackBodiesTGS (0 B, same module) must build: the limit is per entry");
    // Push: the engine row pushes 104 of a 176-byte block -- over 128, refused.
    PXG_CHECK_MSG(build(dev, "psContactGenBda.spv", "ps_primitivesDiffuseCollisionLaunch", 104) == 0,
                  "a 176 B push block must be refused under a 128 B limit, whatever the caller pushes");
    // ... and the other way round: a 160-byte CALLER over the limit, its entry's block (24 B)
    // under it.  The limit is checked against the layout's range, max(caller, block).
    PXG_CHECK_MSG(build(dev, "solverTGS.spv", "writeBackBodiesTGS", 160) == 0,
                  "a 160 B caller must be refused under a 128 B limit although the entry's block is 24 B");
    // Invocations: 1024 threads > 256 refused.
    PXG_CHECK_MSG(build(dev, "aggregateBda.spv", "addAggPairsStage1", 16) == 0,
                  "a 1024-thread workgroup must be refused under a 256-invocation limit");

    // Through the shim: refused launch, no crash.
    const PxgKernelDef manifest[] = { { "solverTGS.spv", "solveWholeIslandTGS", 32u, 0u } };
    {
        PxgCudaShim shim(dev, manifest, 1, &refusalLoader);
        uint32_t blob[8] = {};
        PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, blob, 32) == PxgResult::NotFound,
                      "the shim must refuse an over-limit kernel");
    }
    if (wb)
        dev->destroyPipeline(wb);
    delete dev;

    // A second device: a hook ABOVE the device's limit changes nothing (they only
    // lower), the environment hook lowers too (the smallest source wins), and the
    // workgroup SIZE is checked per dimension -- addAggPairsStage1's 1024 x 1 x 1 is
    // refused under a 512 size limit although its 1024 invocations fit.  The variable
    // is set with no GPU work in flight and removed before anything else runs.
    real->flush();
    PxgSlangComputeDevice* devB = nullptr;
    {
        ScopedEnv env("PXG_VK_LIMIT_SHARED_MEMORY", "20000");
        devB = limitedDevice(1u << 20, 1u << 20, 0, 512);
    }
    PXG_CHECK_MSG(devB != nullptr, "the second lowered-limit device must create");
    if (!devB)
        return;
    const PxgSlangComputeDevice::ComputeLimits lb = devB->computeLimits();
    PXG_CHECK_MSG(lb.maxPushConstantsSize == rl.maxPushConstantsSize, "a hook above the limit must not raise it (push %u, "
                  "device %u)", lb.maxPushConstantsSize, rl.maxPushConstantsSize);
    PXG_CHECK_MSG(lb.maxComputeSharedMemorySize == 20000u, "PXG_VK_LIMIT_SHARED_MEMORY=20000 must lower the limit "
                  "below the desc's (ignored) 1 MiB: got %u", lb.maxComputeSharedMemorySize);
    PXG_CHECK_MSG(lb.maxComputeWorkGroupInvocations == rl.maxComputeWorkGroupInvocations, "no invocation hook: %u vs %u",
                  lb.maxComputeWorkGroupInvocations, rl.maxComputeWorkGroupInvocations);
    for (int d = 0; d < 3; ++d)
    {
        const uint32_t want = rl.maxComputeWorkGroupSize[d] < 512u ? rl.maxComputeWorkGroupSize[d] : 512u;
        PXG_CHECK_MSG(lb.maxComputeWorkGroupSize[d] == want, "workgroup size[%d] %u, expected %u", d,
                      lb.maxComputeWorkGroupSize[d], want);
    }
    PXG_CHECK_MSG(build(devB, "aggregateBda.spv", "addAggPairsStage1", 16) == 0,
                  "a 1024 x 1 x 1 workgroup must be refused under a 512 per-dimension limit");
    PXG_CHECK_MSG(build(devB, "solverTGS.spv", "solveWholeIslandTGS", 32) == 0,
                  "30720 B of groupshared must be refused under the environment's 20000 B");
    const PxgPipelineHandle diffuse = build(devB, "psContactGenBda.spv", "ps_primitivesDiffuseCollisionLaunch", 104);
    PXG_CHECK_MSG(diffuse != 0, "a 192 B push block and a 256 x 1 x 1 workgroup fit this device");
    if (diffuse)
        devB->destroyPipeline(diffuse);
    delete devB;

    // Check round 1: Workgroup / push sizes past 4 GiB are refused on the REAL device
    // (they wrapped to 0 B and were built, then crashed at the flush).  Refused before any
    // Vulkan object exists, so the hand-assembled modules never reach the driver.
    {
        const std::vector<uint32_t> one = hugeArrayModule({ 0x40000000u }, false);
        const std::vector<uint32_t> two = hugeArrayModule({ 0x20000000u, 0x20000000u }, false);
        const std::vector<uint32_t> pushBig = hugeArrayModule({ 0x40000000u }, true);
        PXG_CHECK_MSG(real->createPipeline(one.data(), one.size() * 4, "main", 0, nullptr, 0) == 0,
                      "4 GiB of groupshared must be refused, not wrapped to 0 B and built");
        PXG_CHECK_MSG(real->createPipeline(two.data(), two.size() * 4, "main", 0, nullptr, 0) == 0,
                      "2 x 2 GiB of groupshared must be refused, not summed to 0 B and built");
        PXG_CHECK_MSG(real->createPipeline(pushBig.data(), pushBig.size() * 4, "main", 0, nullptr, 16) == 0,
                      "a 4 GiB push block must be refused, not wrapped to 0 B and built");
    }

    // Check round 1: every documented environment hook, by its exact name (a typo in the
    // device's string would otherwise pass every test), each restored afterwards.
    struct Hook
    {
        const char* env;
        const char* value;
    };
    const Hook hooks[] = { { "PXG_VK_LIMIT_PUSH_CONSTANTS", "128" },
                           { "PXG_VK_LIMIT_SHARED_MEMORY", "16384" },
                           { "PXG_VK_LIMIT_WORKGROUP_INVOCATIONS", "128" },
                           { "PXG_VK_LIMIT_WORKGROUP_SIZE", "128" },
                           { "PXG_VK_LIMIT_WORKGROUP_COUNT", "1000" } };
    real->flush();
    for (const Hook& hk : hooks)
    {
        PxgSlangComputeDevice* hd = nullptr;
        {
            ScopedEnv env(hk.env, hk.value);
            hd = limitedDevice(0, 0, 0);
        }
        PXG_CHECK_MSG(hd != nullptr, "%s=%s: the device must create", hk.env, hk.value);
        if (!hd)
            continue;
        const PxgSlangComputeDevice::ComputeLimits h = hd->computeLimits();
        const uint32_t v = uint32_t(std::strtoul(hk.value, nullptr, 0));
        bool lowered = false;
        if (std::strcmp(hk.env, "PXG_VK_LIMIT_PUSH_CONSTANTS") == 0)
            lowered = h.maxPushConstantsSize == v && h.maxComputeSharedMemorySize == rl.maxComputeSharedMemorySize;
        else if (std::strcmp(hk.env, "PXG_VK_LIMIT_SHARED_MEMORY") == 0)
            lowered = h.maxComputeSharedMemorySize == v && h.maxPushConstantsSize == rl.maxPushConstantsSize;
        else if (std::strcmp(hk.env, "PXG_VK_LIMIT_WORKGROUP_INVOCATIONS") == 0)
            lowered = h.maxComputeWorkGroupInvocations == v &&
                      h.maxComputeWorkGroupSize[0] == rl.maxComputeWorkGroupSize[0];
        else if (std::strcmp(hk.env, "PXG_VK_LIMIT_WORKGROUP_SIZE") == 0)
        {
            lowered = h.maxComputeWorkGroupInvocations == rl.maxComputeWorkGroupInvocations;
            for (int d = 0; d < 3; ++d)
                lowered = lowered && h.maxComputeWorkGroupSize[d] ==
                                         (rl.maxComputeWorkGroupSize[d] < v ? rl.maxComputeWorkGroupSize[d] : v);
        }
        else
            lowered = h.maxComputeWorkGroupCount[0] == v && h.maxComputeWorkGroupCount[1] == v &&
                      h.maxComputeWorkGroupCount[2] == v &&
                      h.maxComputeWorkGroupSize[0] == rl.maxComputeWorkGroupSize[0];
        PXG_CHECK_MSG(lowered, "%s=%s must lower exactly its limit (push %u, shared %u, invocations %u, size %u, "
                      "count %u)", hk.env, hk.value, h.maxPushConstantsSize, h.maxComputeSharedMemorySize,
                      h.maxComputeWorkGroupInvocations, h.maxComputeWorkGroupSize[0], h.maxComputeWorkGroupCount[0]);
        delete hd;
    }

    // Check round 1: the GRID against maxComputeWorkGroupCount -- a dimension over it is a
    // refused dispatch (it was recorded: VUID-vkCmdDispatch-groupCountX-00386), and the
    // shim reports a refused launch.  On the real device just past its limit, and on one
    // whose count hook is 4: (4, 4, 4) runs, a 5 in any dimension is refused.
    {
        const PxgPipelineHandle echo = build(real, "p98PushPad.spv", "pushPadEcho", 64);
        PXG_CHECK_MSG(echo != 0, "pushPadEcho must build");
        PushPadParams gp = {};
        const PxgBufferHandle gout = real->createBuffer(64);
        gp.out = real->getBufferDeviceAddress(gout);
        for (int d = 0; d < 3; ++d)
        {
            if (rl.maxComputeWorkGroupCount[d] == 0xFFFFFFFFu)
                continue;
            uint32_t g[3] = { 1, 1, 1 };
            g[d] = rl.maxComputeWorkGroupCount[d] + 1u;
            PXG_CHECK_MSG(!real->dispatch(echo, g[0], g[1], g[2], nullptr, 0, &gp, sizeof(gp)),
                          "grid dimension %d = %u (limit %u) must be refused", d, g[d], rl.maxComputeWorkGroupCount[d]);
        }
        real->flush();
        if (echo)
            real->destroyPipeline(echo);
        real->destroyBuffer(gout);

        PxgSlangComputeDevice* gd = limitedDevice(0, 0, 0, 0, 4);
        PXG_CHECK_MSG(gd != nullptr, "the count-4 device must create");
        if (gd)
        {
            const PxgSlangComputeDevice::ComputeLimits lg = gd->computeLimits();
            PXG_CHECK_MSG(lg.maxComputeWorkGroupCount[0] == 4 && lg.maxComputeWorkGroupCount[1] == 4 &&
                              lg.maxComputeWorkGroupCount[2] == 4,
                          "the count hook lowers all three dimensions (%u, %u, %u)", lg.maxComputeWorkGroupCount[0],
                          lg.maxComputeWorkGroupCount[1], lg.maxComputeWorkGroupCount[2]);
            const PxgPipelineHandle e4 = build(gd, "p98PushPad.spv", "pushPadEcho", 64);
            const PxgBufferHandle o4 = gd->createBuffer(64);
            const std::vector<uint32_t> minus(16, 0xffffffffu);
            gd->upload(o4, minus.data(), 64);
            PushPadParams p4 = {};
            p4.out = gd->getBufferDeviceAddress(o4);
            p4.head0 = 0x4444u;
            PXG_CHECK_MSG(!gd->dispatch(e4, 5, 1, 1, nullptr, 0, &p4, sizeof(p4)), "grid (5, 1, 1) over a count of 4");
            PXG_CHECK_MSG(!gd->dispatch(e4, 1, 5, 1, nullptr, 0, &p4, sizeof(p4)), "grid (1, 5, 1) over a count of 4");
            PXG_CHECK_MSG(!gd->dispatch(e4, 1, 1, 5, nullptr, 0, &p4, sizeof(p4)), "grid (1, 1, 5) over a count of 4");
            gd->flush();
            PXG_CHECK_MSG(downloadBuffer<uint32_t>(gd, o4, 1)[0] == 0xffffffffu, "a refused grid records nothing");
            PXG_CHECK_MSG(gd->dispatch(e4, 4, 4, 4, nullptr, 0, &p4, sizeof(p4)), "grid (4, 4, 4) fits a count of 4");
            gd->flush();
            PXG_CHECK_MSG(downloadBuffer<uint32_t>(gd, o4, 1)[0] == 0x4444u, "the (4, 4, 4) grid ran");
            const PxgKernelDef echoDef[] = { { "p98PushPad.spv", "pushPadEcho", 64u, 0u } };
            {
                PxgCudaShim shim(gd, echoDef, 1, &refusalLoader);
                PXG_CHECK_MSG(shim.launchKernel(0, 5, 1, 1, &p4, sizeof(p4)) == PxgResult::NotFound,
                              "the shim must report a refused grid as a refused launch");
                PXG_CHECK_MSG(shim.launchKernel(0, 4, 1, 1, &p4, sizeof(p4)) == PxgResult::Success,
                              "and launch a grid that fits");
                shim.streamSynchronize();
            }
            gd->destroyBuffer(o4);
            if (e4)
                gd->destroyPipeline(e4);
            delete gd;
        }
    }

    // A third device: every dimension lowered to 1, so dimensions 1 and 2 are each the
    // ONLY reason a workgroup is refused (two invocations fit every invocation limit).
    PxgSlangComputeDevice* devC = limitedDevice(0, 0, 0, 1);
    PXG_CHECK_MSG(devC != nullptr, "the size-1 device must create");
    if (!devC)
        return;
    const PxgSlangComputeDevice::ComputeLimits lc = devC->computeLimits();
    PXG_CHECK_MSG(lc.maxComputeWorkGroupSize[0] == 1 && lc.maxComputeWorkGroupSize[1] == 1 &&
                      lc.maxComputeWorkGroupSize[2] == 1,
                  "the size hook lowers all three dimensions (%u, %u, %u)", lc.maxComputeWorkGroupSize[0],
                  lc.maxComputeWorkGroupSize[1], lc.maxComputeWorkGroupSize[2]);
    PXG_CHECK_MSG(build(devC, "p98PushPad.spv", "pushPadDimY", 64) == 0,
                  "a (1, 2, 1) workgroup must be refused under a per-dimension limit of 1");
    PXG_CHECK_MSG(build(devC, "p98PushPad.spv", "pushPadDimZ", 64) == 0,
                  "a (1, 1, 2) workgroup must be refused under a per-dimension limit of 1");
    const PxgPipelineHandle one = build(devC, "p98PushPad.spv", "pushPadEcho", 64);
    PXG_CHECK_MSG(one != 0, "a (1, 1, 1) workgroup fits a per-dimension limit of 1");
    if (one)
        devC->destroyPipeline(one);
    delete devC;
}

// F4, the per-call VkResult checks: each of the four create calls is made to fail in
// turn (the interposer at the top of this file).  The build must be refused, every
// object the attempt had made must have been destroyed, and the device must go on
// building and running the same entry point.
PXG_TEST(pipelineRefusal_vkFailures)
{
    PxgSlangComputeDevice* dev = testDevice();
    const std::vector<uint8_t> blob = loadShaderBlob("p98PushPad.spv");
    struct Case
    {
        const char* call;
        uint32_t madeBeforeFailing;   // module, set layout, pipeline layout, pipeline
    };
    const Case cases[] = {
        { "vkCreateShaderModule", 0 },        { "vkCreateDescriptorSetLayout", 1 },
        { "vkCreatePipelineLayout", 2 },      { "vkCreateComputePipelines", 3 },
        { "vkCreateComputePipelines+handle", 4 }, { "vkCreateComputePipelines+null", 3 },
    };
    std::fprintf(stderr, "[pipelineRefusal] the [PxgVulkan] refusals below are EXPECTED (forced failures)\n");
    gP98JunkDestroyed = 0;
    for (const Case& c : cases)
    {
        gP98Live.clear();
        gP98Created = 0;
        gP98Track = true;
        gP98Fail = c.call;
        const PxgPipelineHandle h = dev->createPipeline(blob.data(), blob.size(), "pushPadEcho", 0, nullptr, 64);
        gP98Fail = nullptr;
        gP98Track = false;
        PXG_CHECK_MSG(h == 0, "%s failing: the build must be refused, got handle %u", c.call, h);
        PXG_CHECK_MSG(gP98Created == c.madeBeforeFailing, "%s failing: %u object(s) made before it, expected %u",
                      c.call, gP98Created, c.madeBeforeFailing);
        PXG_CHECK_MSG(gP98Live.empty(), "%s failing: %zu object(s) of the refused build were never destroyed", c.call,
                      gP98Live.size());
        // Check round 1: the failed call's output is undefined (the interposer writes junk);
        // the refusal must not have destroyed it.
        PXG_CHECK_MSG(gP98JunkDestroyed == 0, "%s failing: the undefined output was handed to a destroy call",
                      c.call);
        gP98JunkDestroyed = 0;
        if (h)
            dev->destroyPipeline(h);
    }

    // The device still builds and runs the entry.
    const PxgPipelineHandle ok = dev->createPipeline(blob.data(), blob.size(), "pushPadEcho", 0, nullptr, 64);
    PXG_CHECK_MSG(ok != 0, "pushPadEcho must build once nothing is forced to fail");
    if (!ok)
        return;
    const PxgBufferHandle out = dev->createBuffer(64);
    PushPadParams p = {};
    p.out = dev->getBufferDeviceAddress(out);
    p.head0 = 0x5eed0001u;
    p.head1 = 0x5eed0002u;
    PXG_CHECK(dev->dispatch(ok, 1, 1, 1, nullptr, 0, &p, sizeof(p)));
    dev->flush();
    const std::vector<uint32_t> w = downloadBuffer<uint32_t>(dev, out, 16);
    PXG_CHECK_MSG(w[0] == 0x5eed0001u && w[1] == 0x5eed0002u, "the device runs after the refusals (%08x %08x)", w[0],
                  w[1]);
    dev->destroyBuffer(out);
    dev->destroyPipeline(ok);
}

// F2: the layout covers the declared block, and the surplus is pushed as zeros.
PXG_TEST(pipelineRefusal_pushPadding)
{
    PxgSlangComputeDevice* dev = testDevice();
    struct Case
    {
        const char* file;
        const char* entry;
        uint32_t caller, range;
    };
    const Case cases[] = {
        { "psContactGenBda.spv", "ps_primitivesDiffuseCollisionLaunch", 104, 192 },   // 5.12 block (5.6.1: 176)
        { "psContactGenBda.spv", "ps_convexDiffuseCollisionLaunch", 104, 192 },
        { "psDiffuseBuffers.spv", "ps_updateUnsortedDiffuseArrayLaunch", 16, 24 },
        { "solverTGS.spv", "writeBackBodiesTGS", 24, 24 },     // exact: unchanged
        { "solverTGS.spv", "writeBackBodiesTGS", 40, 40 },     // a caller over the block keeps its size
        { "p98PushPad.spv", "pushPadEcho", 16, 64 },
    };
    for (const Case& c : cases)
    {
        const PxgPipelineHandle h = build(dev, c.file, c.entry, c.caller);
        PXG_CHECK_MSG(h != 0, "%s must build", c.entry);
        PXG_CHECK_MSG(dev->pipelinePushRangeBytes(h) == c.range, "%s caller %u: layout range %u, expected %u",
                      c.entry, c.caller, dev->pipelinePushRangeBytes(h), c.range);
        if (h)
            dev->destroyPipeline(h);
    }

    // Check round 1: the widening is REPORTED under PXG_VK_DEBUG_NAMES (what
    // tools/run_validation.sh sets), because it makes VUID-07987 unreachable: a shader that
    // reads past the caller's bytes now reads zeros with no layer message, and
    // tools/vvl_triage.py fails on this line unless validation_waivers.txt lists the entry.
    {
        const std::vector<uint8_t> pad = loadShaderBlob("p98PushPad.spv");
        std::string quiet, narrowLine, exactLine;
        dev->flush();
        // The control is a device made with PXG_VK_DEBUG_NAMES=0: the suite may run under
        // tools/run_validation.sh, which sets it to 1 for testDevice() too.
        PxgSlangComputeDevice* plain = nullptr;
        {
            ScopedEnv env("PXG_VK_DEBUG_NAMES", "0");
            plain = limitedDevice(0, 0, 0);
        }
        PXG_CHECK_MSG(plain != nullptr, "a PXG_VK_DEBUG_NAMES=0 device must create");
        if (plain)
        {
            quiet = captureStderr([&] {
                const PxgPipelineHandle h = plain->createPipeline(pad.data(), pad.size(), "pushPadEcho", 0, nullptr, 16);
                if (h)
                    plain->destroyPipeline(h);
            });
            delete plain;
        }
        PXG_CHECK_MSG(quiet.find("zero-filled") == std::string::npos,
                      "without PXG_VK_DEBUG_NAMES the widening is silent (got: %s)", quiet.c_str());
        PxgSlangComputeDevice* named = nullptr;
        {
            ScopedEnv env("PXG_VK_DEBUG_NAMES", "1");
            named = limitedDevice(0, 0, 0);
        }
        PXG_CHECK_MSG(named != nullptr, "a PXG_VK_DEBUG_NAMES device must create");
        if (named)
        {
            narrowLine = captureStderr([&] {
                const PxgPipelineHandle h = named->createPipeline(pad.data(), pad.size(), "pushPadEcho", 0, nullptr, 16);
                if (h)
                    named->destroyPipeline(h);
            });
            exactLine = captureStderr([&] {
                const PxgPipelineHandle h = named->createPipeline(pad.data(), pad.size(), "pushPadEcho", 0, nullptr, 64);
                if (h)
                    named->destroyPipeline(h);
            });
            delete named;
        }
        PXG_CHECK_MSG(narrowLine.find("[PxgVulkan] createPipeline(pushPadEcho): push block 64 B wider than caller "
                                      "16 B; zero-filled") != std::string::npos,
                      "a widened range must be reported under PXG_VK_DEBUG_NAMES (got: %s)", narrowLine.c_str());
        PXG_CHECK_MSG(exactLine.find("zero-filled") == std::string::npos,
                      "a caller that fills the block is not reported (got: %s)", exactLine.c_str());
    }

    // Observable zero fill.  A FULL push of a 0xA5 pattern first, through a pipeline whose
    // layout is identical (so the push-constant state carries over on a driver that keeps
    // it); then a 48-byte caller, whose last 16 bytes must read back as zero; then the
    // 16-byte caller, whose 48 must -- after the 48-byte caller left 0xA5 in the padded
    // copy's bytes 16..47, so a copy that is not re-zeroed for each push shows too.
    const PxgPipelineHandle wide = build(dev, "p98PushPad.spv", "pushPadEcho", 64);
    const PxgPipelineHandle mid = build(dev, "p98PushPad.spv", "pushPadEcho", 48);
    const PxgPipelineHandle narrow = build(dev, "p98PushPad.spv", "pushPadEcho", 16);
    PXG_CHECK_MSG(wide != 0 && mid != 0 && narrow != 0, "the three p98PushPad pipelines must build");
    if (!wide || !mid || !narrow)
        return;
    const PxgBufferHandle outWide = dev->createBuffer(64);
    const PxgBufferHandle outMid = dev->createBuffer(64);
    const PxgBufferHandle outNarrow = dev->createBuffer(64);
    const std::vector<uint32_t> minus(16, 0xffffffffu);
    dev->upload(outWide, minus.data(), 64);
    dev->upload(outMid, minus.data(), 64);
    dev->upload(outNarrow, minus.data(), 64);

    PushPadParams full;
    full.out = dev->getBufferDeviceAddress(outWide);
    full.head0 = 0x11111111u;
    full.head1 = 0x22222222u;
    for (uint32_t& t : full.tail)
        t = 0xa5a5a5a5u;
    PXG_CHECK(dev->dispatch(wide, 1, 1, 1, nullptr, 0, &full, sizeof(full)));
    dev->barrier();
    PushPadParams part = full;
    part.out = dev->getBufferDeviceAddress(outMid);
    part.head0 = 0x55555555u;
    part.head1 = 0x66666666u;
    PXG_CHECK(dev->dispatch(mid, 1, 1, 1, nullptr, 0, &part, 48));      // out, heads, tail0, tail1
    dev->barrier();
    PushPadParams head = full;
    head.out = dev->getBufferDeviceAddress(outNarrow);
    head.head0 = 0x33333333u;
    head.head1 = 0x44444444u;
    PXG_CHECK(dev->dispatch(narrow, 1, 1, 1, nullptr, 0, &head, 16));   // only out/head0/head1
    dev->flush();

    const std::vector<uint32_t> w = downloadBuffer<uint32_t>(dev, outWide, 16);
    const std::vector<uint32_t> m = downloadBuffer<uint32_t>(dev, outMid, 16);
    const std::vector<uint32_t> n = downloadBuffer<uint32_t>(dev, outNarrow, 16);
    PXG_CHECK_MSG(w[0] == 0x11111111u && w[1] == 0x22222222u && w[2] == 0xa5a5a5a5u && w[13] == 0xa5a5a5a5u,
                  "the full push round-trips (%08x %08x %08x %08x)", w[0], w[1], w[2], w[13]);
    uint32_t midBad = 0;
    for (int i = 2; i < 10; ++i)
        midBad += m[i] != 0xa5a5a5a5u ? 1u : 0u;
    for (int i = 10; i < 14; ++i)
        midBad += m[i] != 0u ? 1u : 0u;
    PXG_CHECK_MSG(m[0] == 0x55555555u && m[1] == 0x66666666u && midBad == 0,
                  "the 48-byte caller: its 48 bytes, then 16 zeros (%u of 12 tail words wrong; [10] = %08x)", midBad,
                  m[10]);
    PXG_CHECK_MSG(n[0] == 0x33333333u && n[1] == 0x44444444u, "the narrow head (%08x %08x)", n[0], n[1]);
    uint32_t nonZero = 0;
    for (int i = 2; i < 14; ++i)
        nonZero += n[i] != 0u ? 1u : 0u;
    PXG_CHECK_MSG(nonZero == 0, "the 48 bytes past the narrow caller's 16 must be pushed as zeros: %u of 12 "
                                "words are not (first %08x)", nonZero, n[2]);
    dev->destroyBuffer(outWide);
    dev->destroyBuffer(outMid);
    dev->destroyBuffer(outNarrow);
    dev->destroyPipeline(wide);
    dev->destroyPipeline(mid);
    dev->destroyPipeline(narrow);
}

// Check round 0: what a dispatch BINDS is validated before anything is recorded -- the
// buffer and texture counts must be the pipeline's, and every buffer, texture and
// indirect-argument handle must be live (they were assert-only counts and operator[]
// lookups: a stale or 0 buffer recorded a null descriptor, and dispatchIndirect with a
// null arg buffer segfaulted at the flush).  The refusals are followed by a flush (a
// null object recorded by a broken check faults there at the latest) and by the same
// dispatches done right, which must still work.
PXG_TEST(pipelineRefusal_bindings)
{
    PxgSlangComputeDevice* dev = testDevice();
    const std::vector<uint8_t> ind = loadShaderBlob("npIndirect.spv");
    struct IndParams
    {
        uint32_t count, pad0, pad1, pad2;
    } prm = { 64, 0, 0, 0 };
    const uint32_t stride = 4;
    const PxgPipelineHandle pipeWrite =
        dev->createPipeline(ind.data(), ind.size(), "writeArgs", 1, &stride, sizeof(IndParams));
    const PxgPipelineHandle pipeFill =
        dev->createPipeline(ind.data(), ind.size(), "fillGrid", 1, &stride, sizeof(IndParams));
    PXG_CHECK_MSG(pipeWrite != 0 && pipeFill != 0, "the npIndirect pipelines must build");
    if (!pipeWrite || !pipeFill)
        return;
    const PxgBufferHandle bArgs = dev->createBuffer(16);
    const PxgBufferHandle bOut = dev->createBuffer(64 * 4);
    const std::vector<uint32_t> zero(64, 0u);
    dev->upload(bArgs, zero.data(), 16);
    dev->upload(bOut, zero.data(), 64 * 4);
    PxgBufferHandle stale = dev->createBuffer(256);
    dev->destroyBuffer(stale);
    const PxgBufferHandle none = 0;

    std::fprintf(stderr, "[pipelineRefusal] the [PxgVulkan] dispatch refusals below are EXPECTED\n");
    const size_t P = sizeof(IndParams);
    PXG_CHECK_MSG(!dev->dispatch(pipeFill, 1, 1, 1, &stale, 1, &prm, P), "a destroyed buffer must be refused");
    PXG_CHECK_MSG(!dev->dispatch(pipeFill, 1, 1, 1, &none, 1, &prm, P), "buffer handle 0 must be refused");
    PXG_CHECK_MSG(!dev->dispatch(pipeFill, 1, 1, 1, nullptr, 0, &prm, P), "0 buffers to a 1-buffer pipeline");
    const PxgBufferHandle two[2] = { bOut, bArgs };
    PXG_CHECK_MSG(!dev->dispatch(pipeFill, 1, 1, 1, two, 2, &prm, P), "2 buffers to a 1-buffer pipeline");
    PXG_CHECK_MSG(!dev->dispatch(pipeFill, 1, 1, 1, nullptr, 1, &prm, P), "a null buffer array");
    const PxgTextureHandle someTex = 1;
    PXG_CHECK_MSG(!dev->dispatchTex(pipeFill, 1, 1, 1, &bOut, 1, &someTex, 1, &prm, P),
                  "a texture to a pipeline that binds none");

    PXG_CHECK_MSG(!dev->dispatchIndirect(pipeFill, 0, 0, &bOut, 1, &prm, P), "arg buffer 0 must be refused");
    PXG_CHECK_MSG(!dev->dispatchIndirect(pipeFill, stale, 0, &bOut, 1, &prm, P), "a destroyed arg buffer");
    PXG_CHECK_MSG(!dev->dispatchIndirect(pipeFill, bArgs, 2, &bOut, 1, &prm, P), "a misaligned arg offset");
    PXG_CHECK_MSG(!dev->dispatchIndirect(pipeFill, bArgs, 8, &bOut, 1, &prm, P),
                  "an arg offset whose 12-byte command runs past the 16-byte buffer");
    PXG_CHECK_MSG(!dev->dispatchIndirect(pipeFill, bArgs, 0, &stale, 1, &prm, P), "a destroyed bound buffer");
    PXG_CHECK_MSG(!dev->dispatchIndirect(pipeFill, bArgs, 0, nullptr, 0, &prm, P), "0 buffers, indirect");
    // Transfers on a handle that is not live are skipped (a message, nothing recorded or
    // mapped): they were operator[] lookups too, and mapped or recorded VK_NULL_HANDLE.
    std::vector<uint32_t> probe(4, 0xc0ffee00u);
    dev->upload(stale, zero.data(), 16);
    dev->download(stale, probe.data(), 16);
    dev->copyBuffer(bOut, 0, stale, 0, 16);
    dev->copyBuffer(stale, 0, bOut, 0, 16);
    dev->fillBuffer(stale, 0, 7u, 16);
    PXG_CHECK_MSG(probe[0] == 0xc0ffee00u && probe[3] == 0xc0ffee00u,
                  "a download from a destroyed buffer must leave the destination alone (%08x)", probe[0]);

    // Check round 1: a transfer whose range runs past its (live) buffer is refused the same
    // way -- it was a heap overrun of the mapping, or an out-of-range vkCmdCopyBuffer /
    // vkMapMemory, then rc 139.  bArgs is 16 bytes of zeros; nothing below may change it.
    const std::vector<uint32_t> ones(64, 0xffffffffu);
    const size_t huge = ~size_t(0);
    dev->upload(bArgs, ones.data(), 16, 8);         // 8 bytes past the end
    dev->upload(bArgs, ones.data(), 4, 20);         // starts past the end
    dev->upload(bArgs, ones.data(), huge - 7, 8);   // offset + size overflows
    dev->fillBuffer(bArgs, 4, 0xffffffffu, 16);
    dev->fillBuffer(bArgs, 20, 0xffffffffu, 4);
    // The copies move ONES, so an overrun that was recorded anyway shows in the zeros.
    const PxgBufferHandle bOnes = dev->createBuffer(16);
    dev->upload(bOnes, ones.data(), 16);
    dev->copyBuffer(bArgs, 8, bOnes, 0, 16);        // destination overrun
    dev->copyBuffer(bOut, 0, bOnes, 4, 16);         // source overrun (bOut must stay zero too)
    std::vector<uint32_t> past(8, 0xc0ffee00u);
    dev->download(bArgs, past.data(), 32, 0);
    dev->download(bArgs, past.data(), 4, 16);
    PXG_CHECK_MSG(past[0] == 0xc0ffee00u && past[7] == 0xc0ffee00u,
                  "a download past the end must leave the destination alone (%08x)", past[0]);
    dev->flush();
    {
        const std::vector<uint32_t> a = downloadBuffer<uint32_t>(dev, bArgs, 4);
        const std::vector<uint32_t> o = downloadBuffer<uint32_t>(dev, bOut, 64);
        uint32_t touched = 0;
        for (uint32_t v : a)
            touched += v != 0u ? 1u : 0u;
        for (uint32_t v : o)
            touched += v != 0u ? 1u : 0u;
        PXG_CHECK_MSG(touched == 0, "out-of-range transfers must change nothing: %u word(s) changed", touched);
        dev->destroyBuffer(bOnes);
        // The pinned (persistently mapped) form, whose overrun was a raw heap write.
        void* mapped = nullptr;
        const PxgBufferHandle pinned = dev->createHostVisibleBuffer(16, &mapped);
        PXG_CHECK_MSG(pinned != 0 && mapped != nullptr, "a 16-byte pinned buffer must create");
        if (pinned)
        {
            std::memset(mapped, 0, 16);
            dev->upload(pinned, ones.data(), 16, 4);
            dev->download(pinned, past.data(), 32, 0);
            PXG_CHECK_MSG(static_cast<const uint32_t*>(mapped)[3] == 0u && past[0] == 0xc0ffee00u,
                          "an out-of-range pinned transfer must be refused");
            dev->upload(pinned, ones.data(), 16, 0);    // exactly in range: done
            PXG_CHECK_MSG(static_cast<const uint32_t*>(mapped)[3] == 0xffffffffu, "an in-range pinned upload lands");
            dev->destroyBuffer(pinned);
        }
    }
    dev->flush();

    // The same dispatches done right: writeArgs computes the grid, fillGrid runs on it.
    PXG_CHECK(dev->dispatch(pipeWrite, 1, 1, 1, &bArgs, 1, &prm, P));
    dev->barrier();
    PXG_CHECK(dev->dispatchIndirect(pipeFill, bArgs, 0, &bOut, 1, &prm, P));
    dev->flush();
    const std::vector<uint32_t> got = downloadBuffer<uint32_t>(dev, bOut, 64);
    uint32_t bad = 0;
    for (uint32_t i = 0; i < 64; ++i)
        bad += got[i] != i + 1u ? 1u : 0u;
    PXG_CHECK_MSG(bad == 0, "fillGrid after the refusals: %u of 64 wrong", bad);

    // Textures (the set-0 path, createPipelineTex / dispatchTex): 0, destroyed and
    // bindless (no sampler) handles are refused; the bound texture runs.
    const std::vector<float> texels(8, 0.25f);
    const PxgTextureHandle tex = dev->createTexture3D(2, 2, 2, texels.data());
    if (tex == 0)
        std::printf("  NOTE: backend has no 3D texture support -> skip the texture half\n");
    else
    {
        const std::vector<uint8_t> tb = loadShaderBlob("npTex3d.spv");
        struct TexParams
        {
            uint32_t numSamples;
            float invW, invH, invD;
        } tp = { 4, 0.5f, 0.5f, 0.5f };
        const uint32_t strides[2] = { 16, 4 };
        const PxgPipelineHandle pipeTex =
            dev->createPipelineTex(tb.data(), tb.size(), "tex3dSample", 2, strides, sizeof(TexParams), 1);
        PXG_CHECK_MSG(pipeTex != 0, "the npTex3d pipeline must build");
        const std::vector<float> pts(16, 1.0f);
        const PxgBufferHandle bPts = createBufferFrom(dev, pts);
        const PxgBufferHandle bTexOut = createBufferFrom(dev, std::vector<float>(4, -1.0f));
        const PxgBufferHandle bufs[2] = { bPts, bTexOut };
        PxgTextureHandle gone = dev->createTexture3D(2, 2, 2, texels.data());
        dev->destroyTexture3D(gone);
        const PxgTextureHandle noTex = 0;
        const size_t T = sizeof(TexParams);
        PXG_CHECK_MSG(!dev->dispatchTex(pipeTex, 1, 1, 1, bufs, 2, &noTex, 1, &tp, T), "texture handle 0");
        PXG_CHECK_MSG(!dev->dispatchTex(pipeTex, 1, 1, 1, bufs, 2, &gone, 1, &tp, T), "a destroyed texture");
        PXG_CHECK_MSG(!dev->dispatchTex(pipeTex, 1, 1, 1, bufs, 2, nullptr, 0, &tp, T), "0 textures to a 1-texture "
                      "pipeline");
        if (dev->supportsBindlessTextures())
        {
            const PxgTextureHandle bindless = dev->createTexture3DFmt(
                2, 2, 2, PxgSlangComputeDevice::TextureFormat::R32Float, texels.data());
            PXG_CHECK_MSG(bindless != 0, "a bindless texture must create");
            PXG_CHECK_MSG(!dev->dispatchTex(pipeTex, 1, 1, 1, bufs, 2, &bindless, 1, &tp, T),
                          "a bindless texture (no sampler) on the bound path");
            if (bindless)
                dev->destroyTexture3D(bindless);
        }
        dev->flush();
        PXG_CHECK(dev->dispatchTex(pipeTex, 1, 1, 1, bufs, 2, &tex, 1, &tp, T));
        dev->flush();
        const std::vector<float> s = downloadBuffer<float>(dev, bTexOut, 4);
        PXG_CHECK_MSG(s[0] == 0.25f && s[3] == 0.25f, "the bound texture samples after the refusals (%g %g)", s[0],
                      s[3]);
        dev->destroyBuffer(bPts);
        dev->destroyBuffer(bTexOut);
        if (pipeTex)
            dev->destroyPipeline(pipeTex);
        dev->destroyTexture3D(tex);
    }
    dev->destroyBuffer(bArgs);
    dev->destroyBuffer(bOut);
    dev->destroyPipeline(pipeWrite);
    dev->destroyPipeline(pipeFill);
}

// Check round 0: an allocation that fails is a 0 handle with NOTHING leaked -- the shim's
// OutOfMemory / Unsupported -- where createBuffer used to bind and address a null
// VkDeviceMemory and segfault (measured, rc 139, both memory modes).  Each create call is
// failed in turn through the interposer at the top of this file; the objects the attempt
// had made are counted and must all be destroyed.  A failed descriptor-set allocation is
// a refused dispatch (it used to be an uninitialised pool, an endless pool loop, or a
// null set bound), and a failed init step is no device.
PXG_TEST(pipelineRefusal_outOfMemory)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::fprintf(stderr, "[pipelineRefusal] the [PxgVulkan] allocation failures below are EXPECTED (forced)\n");
    struct Case
    {
        const char* call;
        uint32_t skip;
        int kind;          // 0 createBuffer, 1 createHostVisibleBuffer, 2 createTexture3D
        uint32_t made;     // objects made before the failure (buffer, memory, image, view)
    };
    const Case cases[] = {
        { "vkCreateBuffer", 0, 0, 0 },     { "vkAllocateMemory", 0, 0, 1 },  { "vkBindBufferMemory", 0, 0, 2 },
        { "vkCreateBuffer", 0, 1, 0 },     { "vkAllocateMemory", 0, 1, 1 },  { "vkBindBufferMemory", 0, 1, 2 },
        { "vkMapMemory", 0, 1, 2 },
        // the texture: image, its memory, then the staging buffer and ITS memory, then the view
        { "vkCreateImage", 0, 2, 0 },      { "vkAllocateMemory", 0, 2, 1 },  { "vkBindImageMemory", 0, 2, 2 },
        { "vkCreateBuffer", 0, 2, 2 },     { "vkAllocateMemory", 1, 2, 3 },  { "vkBindBufferMemory", 0, 2, 4 },
        { "vkMapMemory", 0, 2, 4 },        { "vkCreateImageView", 0, 2, 4 },
        // check round 1: the bound texture's sampler, after the view exists
        { "vkCreateSampler", 0, 2, 5 },
    };
    const std::vector<float> texels(8, 0.5f);
    const uint32_t samplersBefore = dev->liveSamplerCount();
    gP98JunkDestroyed = 0;
    for (const Case& c : cases)
    {
        static const char* kinds[] = { "createBuffer", "createHostVisibleBuffer", "createTexture3D" };
        dev->flush();   // nothing of an earlier batch may land inside the tracked window
        gP98Live.clear();
        gP98Created = 0;
        gP98Track = true;
        gP98Skip = c.skip;
        gP98Fail = c.call;
        void* mapped = &gP98Created;   // any non-null value: a refusal must clear it
        uint32_t h = 0;
        if (c.kind == 0)
            h = dev->createBuffer(4096);
        else if (c.kind == 1)
            h = dev->createHostVisibleBuffer(4096, &mapped);
        else
            h = dev->createTexture3D(2, 2, 2, texels.data());
        gP98Fail = nullptr;
        gP98Skip = 0;
        gP98Track = false;
        PXG_CHECK_MSG(h == 0, "%s with %s (skip %u) failing: must return 0, got %u", kinds[c.kind], c.call, c.skip,
                      h);
        if (c.kind == 1)
            PXG_CHECK_MSG(mapped == nullptr, "%s with %s failing: *mappedPtr must be NULL", kinds[c.kind], c.call);
        PXG_CHECK_MSG(gP98Created == c.made, "%s with %s (skip %u) failing: %u object(s) made before it, expected %u",
                      kinds[c.kind], c.call, c.skip, gP98Created, c.made);
        PXG_CHECK_MSG(gP98Live.empty(), "%s with %s (skip %u) failing: %zu object(s) never released", kinds[c.kind],
                      c.call, c.skip, gP98Live.size());
        PXG_CHECK_MSG(gP98JunkDestroyed == 0, "%s with %s (skip %u) failing: the call's undefined output was "
                      "handed to a destroy / free", kinds[c.kind], c.call, c.skip);
        gP98JunkDestroyed = 0;
        if (h && c.kind == 2)
            dev->destroyTexture3D(h);
        else if (h)
            dev->destroyBuffer(h);
    }
    PXG_CHECK_MSG(dev->liveSamplerCount() == samplersBefore, "no sampler may leak (%u, was %u)",
                  dev->liveSamplerCount(), samplersBefore);

    // The engine's door: the shim reports OutOfMemory and hands out nothing.
    struct AxpyParams
    {
        uint64_t x, y, out;
        float alpha;
        uint32_t count;
    };
    const PxgKernelDef manifest[] = { { "npShimAxpy.spv", "shimAxpy", uint32_t(sizeof(AxpyParams)), 0u } };
    PxgCudaShim shim(dev, manifest, 1, &refusalLoader);
    PxgDevicePtr p = 0x5eedu;
    gP98Fail = "vkAllocateMemory";
    PXG_CHECK_MSG(shim.memAlloc(&p, 1u << 20) == PxgResult::OutOfMemory, "memAlloc: OutOfMemory");
    gP98Fail = "vkMapMemory";
    void* hp = &p;
    PXG_CHECK_MSG(shim.memHostAlloc(&hp, 4096) == PxgResult::OutOfMemory, "memHostAlloc: OutOfMemory");
    gP98Fail = nullptr;
    PXG_CHECK_MSG(p == 0x5eedu && hp == &p, "a refused allocation hands out no pointer");

    // Descriptor sets: a hard failure, every pool exhausted (a fresh one included -- this
    // looped forever), and no new pool either.  Each is a refused dispatch, and through
    // the shim a refused launch.
    const uint32_t N = 64;
    PxgDevicePtr dx = 0, dy = 0, dout = 0;
    PXG_CHECK(shim.memAlloc(&dx, N * 4) == PxgResult::Success);
    PXG_CHECK(shim.memAlloc(&dy, N * 4) == PxgResult::Success);
    PXG_CHECK(shim.memAlloc(&dout, N * 4) == PxgResult::Success);
    std::vector<float> x(N), y(N), out(N, -1.0f);
    for (uint32_t i = 0; i < N; ++i)
    {
        x[i] = float(i);
        y[i] = 3.0f;
    }
    shim.memcpyHtoD(dx, x.data(), N * 4);
    shim.memcpyHtoD(dy, y.data(), N * 4);
    shim.memcpyHtoD(dout, out.data(), N * 4);
    AxpyParams prm = { dx, dy, dout, 2.0f, N };
    const std::vector<uint8_t> ind = loadShaderBlob("npIndirect.spv");
    const uint32_t stride = 4;
    const PxgPipelineHandle pipeFill = dev->createPipeline(ind.data(), ind.size(), "fillGrid", 1, &stride, 16);
    const PxgBufferHandle bOut = dev->createBuffer(64 * 4);
    const PxgBufferHandle bArgs = createBufferFrom(dev, std::vector<uint32_t>{ 1u, 1u, 1u, 0u });
    const uint32_t ip[4] = { 64, 0, 0, 0 };
    const char* descFailures[] = { "vkAllocateDescriptorSets", "pools-exhausted", "pools-exhausted+create" };
    gP98NullPool = 0;
    for (const char* f : descFailures)
    {
        gP98Fail = f;
        const bool d = dev->dispatch(pipeFill, 1, 1, 1, &bOut, 1, ip, 16);
        const bool di = dev->dispatchIndirect(pipeFill, bArgs, 0, &bOut, 1, ip, 16);
        const PxgResult lr = shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm));
        gP98Fail = nullptr;
        PXG_CHECK_MSG(!d && !di, "%s: dispatch / dispatchIndirect must be refused (%d %d)", f, int(d), int(di));
        PXG_CHECK_MSG(lr == PxgResult::NotFound, "%s: the shim must refuse the launch (%d)", f, int(lr));
    }
    dev->flush();
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "the shim launches once descriptor sets can be had again");
    shim.streamSynchronize();
    shim.memcpyDtoH(out.data(), dout, N * 4);
    uint32_t bad = 0;
    for (uint32_t i = 0; i < N; ++i)
        bad += out[i] != 2.0f * x[i] + y[i] ? 1u : 0u;
    PXG_CHECK_MSG(bad == 0, "axpy after the forced failures: %u of %u wrong", bad, N);
    // A few more dispatches, so every frame slot's pools are used again: a pool that
    // failed to create must not have been kept and handed to the driver.
    for (int i = 0; i < 8; ++i)
    {
        PXG_CHECK(dev->dispatch(pipeFill, 1, 1, 1, &bOut, 1, ip, 16));
        dev->flush();
    }
    PXG_CHECK_MSG(gP98NullPool == 0, "a descriptor pool that failed to create was used %u time(s)", gP98NullPool);
    shim.memFree(dx);
    shim.memFree(dy);
    shim.memFree(dout);
    dev->destroyBuffer(bOut);
    dev->destroyBuffer(bArgs);
    if (pipeFill)
        dev->destroyPipeline(pipeFill);

    PXG_CHECK_MSG(gP98JunkDestroyed == 0, "a failed descriptor pool's undefined output was destroyed");

    // A failed init step: no device, and no crash tearing down the partial one -- and the
    // failed call's undefined output (junk, here) is not destroyed with it.  Check round 1
    // adds the command pool and the command buffers (round 0 forced only the fence).
    const char* fatal[] = { "vkCreateCommandPool", "vkAllocateCommandBuffers", "vkCreateFence" };
    for (const char* f : fatal)
    {
        gP98JunkDestroyed = 0;
        gP98Fail = f;
        PxgSlangComputeDevice* broken = limitedDevice(0, 0, 0);
        gP98Fail = nullptr;
        PXG_CHECK_MSG(broken == nullptr, "a device whose %s fails must not be created", f);
        PXG_CHECK_MSG(gP98JunkDestroyed == 0, "%s failing at init: its undefined output was destroyed", f);
        delete broken;
    }

    // Check round 1: the pipeline cache is OPTIONAL -- a failed vkCreatePipelineCache is
    // reported and the device works without one (round 0 had made it fatal).
    gP98JunkDestroyed = 0;
    gP98Fail = "vkCreatePipelineCache";
    PxgSlangComputeDevice* noCache = limitedDevice(0, 0, 0);
    gP98Fail = nullptr;
    PXG_CHECK_MSG(noCache != nullptr, "a failed vkCreatePipelineCache must not cost the device");
    if (noCache)
    {
        const PxgPipelineHandle e = build(noCache, "p98PushPad.spv", "pushPadEcho", 64);
        PXG_CHECK_MSG(e != 0, "pipelines build without a cache");
        const PxgBufferHandle o = noCache->createBuffer(64);
        PushPadParams pp = {};
        pp.out = noCache->getBufferDeviceAddress(o);
        pp.head0 = 0xcace0001u;
        PXG_CHECK(e != 0 && noCache->dispatch(e, 1, 1, 1, nullptr, 0, &pp, sizeof(pp)));
        noCache->flush();
        PXG_CHECK_MSG(downloadBuffer<uint32_t>(noCache, o, 1)[0] == 0xcace0001u, "and run");
        noCache->destroyBuffer(o);
        if (e)
            noCache->destroyPipeline(e);
        delete noCache;
    }
    PXG_CHECK_MSG(gP98JunkDestroyed == 0, "the failed cache's undefined output was destroyed");
}

// ---- P9.99: createPipelineEx's required subgroup size ----

namespace
{
// Reaches the BASE default createPipelineEx -- the one the D3D12 backend inherits -- with
// the eleven pure virtuals stubbed; createPipeline answers 77.
struct P99BaseDevice : PxgSlangComputeDevice
{
    const char* getDeviceName() const override { return "p99-base"; }
    const char* getBackendName() const override { return "P99Base"; }
    PxgBufferHandle createBuffer(size_t) override { return 0; }
    void destroyBuffer(PxgBufferHandle) override {}
    void upload(PxgBufferHandle, const void*, size_t, size_t) override {}
    void download(PxgBufferHandle, void*, size_t, size_t) override {}
    PxgPipelineHandle createPipeline(const void*, size_t, const char*, uint32_t, const uint32_t*, uint32_t) override
    {
        return PxgPipelineHandle(77);
    }
    void destroyPipeline(PxgPipelineHandle) override {}
    bool dispatch(PxgPipelineHandle, uint32_t, uint32_t, uint32_t, const PxgBufferHandle*, uint32_t, const void*,
                  uint32_t) override
    {
        return true;
    }
    void barrier() override {}
    void flush() override {}
};
} // namespace

// The rule as a LITERAL host oracle, no device.  L is llvmpipe (measured: window [8, 8],
// maxComputeWorkgroupSubgroups 32); W a wide window [8, 64] with 16 subgroups and default 32,
// so a rule written against minSize or defaultSize instead of the requested size is caught
// (a 32-wide x is a multiple of 8 but not of 64); `big` puts the limit product past 32 bits,
// and a (2^31, 2, 2) workgroup the invocation product.  A refusing row must name its rule,
// an accepting row must leave `why` empty, and a null or zero-length `why` is never written.
PXG_TEST(pipelineRefusal_subgroupSizeRule)
{
    typedef PxgSlangComputeDevice::SubgroupCaps Caps;
    Caps L;
    L.defaultSize = 8;
    L.minSize = 8;
    L.maxSize = 8;
    L.canRequireSize = true;
    L.fullSubgroups = true;
    L.maxComputeWorkgroupSubgroups = 32;
    Caps W;
    W.defaultSize = 32;
    W.minSize = 8;
    W.maxSize = 64;
    W.canRequireSize = true;
    W.fullSubgroups = true;
    W.maxComputeWorkgroupSubgroups = 16;
    Caps noCtl = L;
    noCtl.canRequireSize = false;
    Caps noFull = L;
    noFull.fullSubgroups = false;
    Caps noMws = L;
    noMws.maxComputeWorkgroupSubgroups = 0;
    Caps big;
    big.defaultSize = 2;
    big.minSize = 2;
    big.maxSize = 2;
    big.canRequireSize = true;
    big.fullSubgroups = true;
    big.maxComputeWorkgroupSubgroups = 0x80000000u;
    struct Row
    {
        const char* name;
        const Caps* caps;
        uint32_t size;
        uint32_t localSize[3];
        bool fits;
        const char* says;   // what a refusal must name
    };
    const Row rows[] = {
        { "no request, any shape", &L, 0, { 1, 1, 1 }, true, "" },
        { "no request, no size control", &noCtl, 0, { 4, 1, 1 }, true, "" },
        { "64 at 8", &L, 8, { 64, 1, 1 }, true, "" },
        { "(8, 32, 1) = 8 x 32, the boundary", &L, 8, { 8, 32, 1 }, true, "" },
        { "(8, 33, 1) > 256", &L, 8, { 8, 33, 1 }, false, "VUID-02756" },
        { "(8, 1, 33) > 256 through z", &L, 8, { 8, 1, 33 }, false, "VUID-02756" },
        { "512 > 256", &L, 8, { 512, 1, 1 }, false, "VUID-02756" },
        { "x 4 not a multiple", &L, 8, { 4, 1, 1 }, false, "VUID-02757" },
        { "x 12 >= 8, not a multiple", &L, 8, { 12, 1, 1 }, false, "VUID-02757" },
        { "16 above the window", &L, 16, { 64, 1, 1 }, false, "VUID-02762" },
        { "4 below the window", &L, 4, { 64, 1, 1 }, false, "VUID-02761" },
        { "256 above the window", &L, 256, { 256, 1, 1 }, false, "VUID-02762" },
        { "12 not a power of two (narrow)", &L, 12, { 48, 1, 1 }, false, "VUID-02760" },
        { "24 inside a wide window, not a power of two", &W, 24, { 48, 1, 1 }, false, "VUID-02760" },
        { "wide: 8, the min boundary", &W, 8, { 64, 1, 1 }, true, "" },
        { "wide: 32", &W, 32, { 64, 1, 1 }, true, "" },
        { "wide: 64, the max boundary", &W, 64, { 64, 1, 1 }, true, "" },
        { "wide: 64 x 16 = 1024, the boundary", &W, 64, { 64, 16, 1 }, true, "" },
        { "wide: 64 x 17 > 1024", &W, 64, { 64, 17, 1 }, false, "VUID-02756" },
        { "no size control", &noCtl, 8, { 64, 1, 1 }, false, "VUID-02755" },
        { "no full subgroups", &noFull, 8, { 64, 1, 1 }, false, "VUID-02785" },
        { "unknown x", &L, 8, { 0, 1, 1 }, false, "not readable" },
        { "unknown y", &L, 8, { 8, 0, 1 }, false, "not readable" },
        { "unknown z", &L, 8, { 8, 1, 0 }, false, "not readable" },
        { "maxComputeWorkgroupSubgroups unknown refuses", &noMws, 8, { 8, 1, 1 }, false, "VUID-02756" },
        { "64-bit limit product", &big, 2, { 2, 1, 1 }, true, "" },
        { "64-bit invocation product", &L, 8, { 0x80000000u, 2, 2 }, false, "VUID-02756" },
        { "wide: 64 on x 32 (min and default divide x)", &W, 64, { 32, 1, 1 }, false, "VUID-02757" },
        { "wide: 16 on (8, 2, 1)", &W, 16, { 8, 2, 1 }, false, "VUID-02757" },
        { "wide: 8 on x 8 (default 32)", &W, 8, { 8, 1, 1 }, true, "" },
    };
    for (const Row& r : rows)
    {
        char why[320];
        std::memset(why, 'J', sizeof(why));
        why[sizeof(why) - 1] = '\0';
        const bool got = pxgRequiredSubgroupSizeFits(*r.caps, r.size, r.localSize, why, sizeof(why));
        const bool ok = got == r.fits && (r.fits ? why[0] == '\0' : std::strstr(why, r.says) != nullptr);
        PXG_CHECK_MSG(ok, "%s: size %u on (%u, %u, %u) got %d (expected %d, naming '%s'), why '%s'", r.name, r.size,
                      r.localSize[0], r.localSize[1], r.localSize[2], int(got), int(r.fits), r.says, why);
    }
    const uint32_t x4[3] = { 4, 1, 1 };
    PXG_CHECK_MSG(!pxgRequiredSubgroupSizeFits(L, 8, x4, nullptr, 0), "a null why still refuses");
    char one[1] = { 'J' };
    const bool zeroLen = pxgRequiredSubgroupSizeFits(L, 8, x4, one, 0);
    PXG_CHECK_MSG(!zeroLen && one[0] == 'J', "a zero-length why refuses and is not written (got %d, '%c')",
                  int(zeroLen), one[0]);
    std::printf("  %zu oracle rows + 2 why rows\n", sizeof(rows) / sizeof(rows[0]));
}

// createPipelineEx on the real device.  Its SubgroupCaps must be the physical device's own
// (read through a separate instance).  On llvmpipe's numbers -- gated on the PHYSICAL values,
// so a device that never reads them cannot skip the rows -- every row of
// p99SubgroupShape.spv is either BUILT (exactly one pipeline create, asking the driver for
// the size WITH full subgroups, and every thread then in a full 8-lane subgroup) or
// REFUSED with its VUID before any Vulkan object exists.  sgShape4 without a request is
// the detector's positive control: a 4-thread workgroup cannot fill an 8-lane subgroup, so
// llvmpipe runs it with 4 active lanes and the counter must see all 12 threads.
PXG_TEST(pipelineRefusal_subgroupSizeDevice)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgSlangComputeDevice::SubgroupCaps caps = dev->getSubgroupCaps();
    std::printf("  subgroup caps: default %u, window [%u, %u], can require %d, full %d, "
                "maxComputeWorkgroupSubgroups %u\n",
                caps.defaultSize, caps.minSize, caps.maxSize, int(caps.canRequireSize), int(caps.fullSubgroups),
                caps.maxComputeWorkgroupSubgroups);
    PhysicalSubgroup ps;
    const bool havePs = physicalSubgroupControl(dev->getDeviceName(), &ps);
    PXG_CHECK_MSG(havePs, "the physical device %s must be found again", dev->getDeviceName());
    if (!havePs)
        return;
    std::printf("  physical: subgroupSize %u, window [%u, %u], requiredSubgroupSizeStages 0x%x, subgroupSizeControl "
                "%u, computeFullSubgroups %u, maxComputeWorkgroupSubgroups %u\n",
                ps.subgroupSize, ps.props.minSubgroupSize, ps.props.maxSubgroupSize,
                unsigned(ps.props.requiredSubgroupSizeStages), unsigned(ps.feats.subgroupSizeControl),
                unsigned(ps.feats.computeFullSubgroups), ps.props.maxComputeWorkgroupSubgroups);
    PXG_CHECK_MSG(caps.defaultSize == ps.subgroupSize && caps.minSize == ps.props.minSubgroupSize &&
                      caps.maxSize == ps.props.maxSubgroupSize,
                  "the window must be the physical device's");
    PXG_CHECK_MSG(caps.maxComputeWorkgroupSubgroups == ps.props.maxComputeWorkgroupSubgroups,
                  "maxComputeWorkgroupSubgroups must be the physical device's (%u vs %u)",
                  caps.maxComputeWorkgroupSubgroups, ps.props.maxComputeWorkgroupSubgroups);
    PXG_CHECK_MSG(caps.canRequireSize == ps.canRequire() && caps.fullSubgroups == ps.full(),
                  "canRequireSize / fullSubgroups must be the physical device's (%d %d vs %d %d)",
                  int(caps.canRequireSize), int(caps.fullSubgroups), int(ps.canRequire()), int(ps.full()));
    if (!ps.isLlvmpipeShape())
    {
        std::printf("  NOTE: not llvmpipe's subgroup numbers -> skip the device rows\n");
        return;
    }

    const std::vector<uint8_t> blob = loadShaderBlob("p99SubgroupShape.spv");
    struct Case
    {
        const char* entry;
        uint32_t invocations;
        uint32_t size;
        const char* says;    // nullptr: must build
        uint32_t notFull;    // built: threads (of 3 workgroups) not in a full 8-lane subgroup
        uint32_t ballot;     // built: the first thread's ballot count
    };
    const Case cases[] = {
        { "sgShape64", 64, 8, nullptr, 0, 8 },
        { "sgShape8x32", 256, 8, nullptr, 0, 8 },
        { "sgShape4", 4, 0, nullptr, 12, 4 },   // the detector's positive control
        { "sgShape8x33", 264, 8, "VUID-02756", 0, 0 },
        { "sgShape8x1x33", 264, 8, "VUID-02756", 0, 0 },
        { "sgShape512", 512, 8, "VUID-02756", 0, 0 },
        { "sgShape4", 4, 8, "VUID-02757", 0, 0 },
        { "sgShape12", 12, 8, "VUID-02757", 0, 0 },
        { "sgShape64", 64, 16, "VUID-02762", 0, 0 },
        { "sgShape64", 64, 4, "VUID-02761", 0, 0 },
        { "sgShape64", 64, 12, "VUID-02760", 0, 0 },
        { "sgShape64", 64, 256, "VUID-02762", 0, 0 },
    };
    std::fprintf(stderr, "[pipelineRefusal] the [PxgVulkan] subgroup-size refusals below are EXPECTED\n");
    for (const Case& c : cases)
    {
        const uint32_t callsBefore = gP99Stage.calls;
        gP98Live.clear();
        gP98Created = 0;
        gP98Track = true;
        std::string m;
        const PxgPipelineHandle h = p99BuildEx(dev, blob, c.entry, c.size, &m);
        gP98Track = false;
        const bool full = (gP99Stage.flags & VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT) != 0;
        std::printf("  %-14s @%-3u -> %s\n", c.entry, c.size, h ? "built" : "refused");
        if (c.says)
        {
            PXG_CHECK_MSG(h == 0, "%s @%u must be refused (%s)", c.entry, c.size, c.says);
            PXG_CHECK_MSG(gP98Created == 0, "%s @%u: refused before ANY Vulkan object, but %u were made", c.entry,
                          c.size, gP98Created);
            PXG_CHECK_MSG(gP99Stage.calls == callsBefore, "%s @%u: no vkCreateComputePipelines call (%u)", c.entry,
                          c.size, gP99Stage.calls - callsBefore);
            PXG_CHECK_MSG(m.find(c.entry) != std::string::npos && m.find(c.says) != std::string::npos &&
                              m.find("; refused") != std::string::npos,
                          "%s @%u: the line must name the entry, %s and '; refused': '%s'", c.entry, c.size, c.says,
                          m.c_str());
            if (h)
            {
                // Only on a device that wrongly built it: what the subgroups then did.
                const uint32_t nf = p99NotFullThreads(dev, h, c.invocations);
                std::printf("    built anyway: threads in a non-full subgroup %u of %u\n", nf, 3 * c.invocations);
            }
        }
        else
        {
            PXG_CHECK_MSG(h != 0 && m.empty(), "%s @%u must build without a line: '%s'", c.entry, c.size, m.c_str());
            PXG_CHECK_MSG(gP99Stage.calls == callsBefore + 1, "%s @%u: exactly one vkCreateComputePipelines call (%u)",
                          c.entry, c.size, gP99Stage.calls - callsBefore);
            if (c.size)
                PXG_CHECK_MSG(gP99Stage.chained && gP99Stage.size == c.size && full,
                              "%s @%u: the driver must be asked for size %u with full subgroups (struct %d, size %u, "
                              "full %d)",
                              c.entry, c.size, c.size, int(gP99Stage.chained), gP99Stage.size, int(full));
            else
                PXG_CHECK_MSG(!gP99Stage.chained && !full,
                              "%s @0: no size struct and no full-subgroups flag (struct %d, full %d)", c.entry,
                              int(gP99Stage.chained), int(full));
            if (h)
            {
                uint32_t ballot = 0;
                const uint32_t nf = p99NotFullThreads(dev, h, c.invocations, &ballot);
                std::printf("    threads in a non-full subgroup %u of %u, first ballot %u\n", nf, 3 * c.invocations,
                            ballot);
                PXG_CHECK_MSG(nf == c.notFull && ballot == c.ballot,
                              "%s @%u: %u threads not in a full subgroup (expected %u), ballot %u (expected %u)",
                              c.entry, c.size, nf, c.notFull, ballot, c.ballot);
            }
        }
        if (h)
            dev->destroyPipeline(h);
    }
}

// The desc hooks take one capability away each -- so each capability refusal shows on
// llvmpipe -- and never raise one.  And the base default (D3D12's) refuses WITH a line.
PXG_TEST(pipelineRefusal_subgroupSizeHooks)
{
    {
        P99BaseDevice base;
        const uint32_t stride = 16;
        PxgPipelineHandle h = 1;
        const std::string m =
            captureStderr([&] { h = base.createPipelineEx(nullptr, 0, "p99probe", 1, &stride, 4, 32); });
        PXG_CHECK_MSG(h == 0 && m.find("p99probe") != std::string::npos && m.find("size 32") != std::string::npos &&
                          m.find("; refused") != std::string::npos,
                      "the base default refuses a size with a line: handle %u, '%s'", h, m.c_str());
        PXG_CHECK_MSG(base.createPipelineEx(nullptr, 0, "p99probe", 1, &stride, 4, 0) == PxgPipelineHandle(77),
                      "the base default forwards size 0 to createPipeline");
    }

    PxgSlangComputeDevice* real = testDevice();
    PhysicalSubgroup ps;
    const bool havePs = physicalSubgroupControl(real->getDeviceName(), &ps);
    PXG_CHECK_MSG(havePs, "the physical device %s must be found again", real->getDeviceName());
    if (!havePs)
        return;
    if (!ps.isLlvmpipeShape())
    {
        std::printf("  NOTE: not llvmpipe's subgroup numbers -> skip the hook rows\n");
        return;
    }
    const std::vector<uint8_t> blob = loadShaderBlob("p99SubgroupShape.spv");
    const uint32_t mws = ps.props.maxComputeWorkgroupSubgroups;
    std::fprintf(stderr, "[pipelineRefusal] the [PxgVulkan] subgroup-size refusals below are EXPECTED (hooks)\n");
    struct Hook
    {
        const char* what;
        bool noCtl, noFull;
        uint32_t limit;
        bool canRequire, fullSubgroups;
        uint32_t mws;           // what the caps must say
        const char* says;       // sgShape64 @8: nullptr = builds, else the refusal's rule
    };
    const Hook hooks[] = {
        { "noSubgroupSizeControl", true, false, 0, false, true, mws, "VUID-02755" },
        { "noFullSubgroups", false, true, 0, true, false, mws, "VUID-02785" },
        { "limitMaxComputeWorkgroupSubgroups 4", false, false, 4, true, true, 4, "VUID-02756" },
        { "limitMaxComputeWorkgroupSubgroups 1000 (above: no effect)", false, false, 1000, true, true, mws, nullptr },
    };
    for (const Hook& k : hooks)
    {
        PxgSlangComputeDevice* d = subgroupDevice(k.noCtl, k.noFull, k.limit);
        PXG_CHECK_MSG(d != nullptr, "%s: the device must be created", k.what);
        if (!d)
            continue;
        const PxgSlangComputeDevice::SubgroupCaps c = d->getSubgroupCaps();
        PXG_CHECK_MSG(c.canRequireSize == k.canRequire && c.fullSubgroups == k.fullSubgroups &&
                          c.maxComputeWorkgroupSubgroups == k.mws && c.minSize == ps.props.minSubgroupSize &&
                          c.maxSize == ps.props.maxSubgroupSize,
                      "%s: caps can require %d, full %d, maxComputeWorkgroupSubgroups %u (expected %d %d %u)", k.what,
                      int(c.canRequireSize), int(c.fullSubgroups), c.maxComputeWorkgroupSubgroups, int(k.canRequire),
                      int(k.fullSubgroups), k.mws);
        std::string m;
        PxgPipelineHandle h = p99BuildEx(d, blob, "sgShape64", 8, &m);
        if (k.says)
            PXG_CHECK_MSG(h == 0 && m.find(k.says) != std::string::npos && m.find("; refused") != std::string::npos,
                          "%s: sgShape64 @8 must be refused naming %s: '%s'", k.what, k.says, m.c_str());
        else
            PXG_CHECK_MSG(h != 0 && m.empty(), "%s: sgShape64 @8 must build: '%s'", k.what, m.c_str());
        if (h)
            d->destroyPipeline(h);
        h = p99BuildEx(d, blob, "sgShape64", 0, &m);
        PXG_CHECK_MSG(h != 0 && m.empty(), "%s: without a request sgShape64 still builds: '%s'", k.what, m.c_str());
        if (h)
            d->destroyPipeline(h);
        delete d;
    }
}
