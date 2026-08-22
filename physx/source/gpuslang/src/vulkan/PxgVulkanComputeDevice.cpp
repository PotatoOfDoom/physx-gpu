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

// Vulkan compute backend for the Slang port of the PhysX GPU kernels.
//
// Implementation notes:
//   - Submission (RUNTIME_PLAN.md P1.1): command buffers are N-buffered and
//     retired through ONE timeline semaphore per queue.  submit() hands work
//     to the device without waiting and returns a monotonic ticket;
//     wait()/poll() synchronize on tickets; flush() == wait(submit()).  A
//     blocking-fence fallback covers devices without timeline semaphores.
//   - Memory (RUNTIME_PLAN.md P1.2): PxgSlangDeviceDesc.useDeviceLocalMemory
//     selects DEVICE_LOCAL storage buffers with all host<->device transfers
//     staged through a persistently mapped per-frame ring (the discrete-GPU
//     production path; transfer<->compute edges use per-buffer barriers).
//     Otherwise buffers are HOST_VISIBLE|HOST_COHERENT and transfers are
//     direct maps (fast on UMA/software devices such as llvmpipe).
//   - One descriptor set per dispatch, allocated from per-frame pools that
//     are reset when the frame slot is recycled.
//   - All shader buffer parameters are storage buffers (the Slang kernels
//     declare RWStructuredBuffer uniformly), so every descriptor is
//     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER.

#include "PxgSlangComputeDevice.h"

#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace physx
{
namespace gpuslang
{

#define PXG_VK_CHECK(expr)                                                                 \
    do                                                                                     \
    {                                                                                      \
        VkResult pxgVkResult_ = (expr);                                                    \
        if (pxgVkResult_ != VK_SUCCESS)                                                    \
        {                                                                                  \
            fprintf(stderr, "[gpuslang/vulkan] %s failed: %d (%s:%d)\n", #expr,            \
                    int(pxgVkResult_), __FILE__, __LINE__);                                \
            assert(false);                                                                 \
        }                                                                                  \
    } while (0)

class PxgVulkanComputeDevice : public PxgSlangComputeDevice
{
public:
    PxgVulkanComputeDevice() {}

    ~PxgVulkanComputeDevice() override
    {
        if (mDevice)
        {
            vkDeviceWaitIdle(mDevice);

            for (auto& kv : mBuffers)
            {
                vkDestroyBuffer(mDevice, kv.second.buffer, nullptr);
                vkFreeMemory(mDevice, kv.second.memory, nullptr);
            }
            for (auto& kv : mTextures)
                releaseTexture(kv.second);
            for (auto& kv : mPipelines)
            {
                vkDestroyPipeline(mDevice, kv.second.pipeline, nullptr);
                vkDestroyPipelineLayout(mDevice, kv.second.layout, nullptr);
                vkDestroyDescriptorSetLayout(mDevice, kv.second.setLayout, nullptr);
                vkDestroyShaderModule(mDevice, kv.second.module, nullptr);
            }
            for (Frame& frame : mFrames)
            {
                for (VkDescriptorPool pool : frame.pools)
                    vkDestroyDescriptorPool(mDevice, pool, nullptr);
                if (frame.staging)
                {
                    vkDestroyBuffer(mDevice, frame.staging, nullptr);
                    vkFreeMemory(mDevice, frame.stagingMemory, nullptr);
                }
            }
            if (mBindlessSampler)
                vkDestroySampler(mDevice, mBindlessSampler, nullptr);
            if (mBindlessPool)
                vkDestroyDescriptorPool(mDevice, mBindlessPool, nullptr);
            if (mBindlessSetLayout)
                vkDestroyDescriptorSetLayout(mDevice, mBindlessSetLayout, nullptr);
            if (mQueryPool)
                vkDestroyQueryPool(mDevice, mQueryPool, nullptr);
            if (mPipelineCache)
            {
                savePipelineCache();
                vkDestroyPipelineCache(mDevice, mPipelineCache, nullptr);
            }
            if (mTimeline)
                vkDestroySemaphore(mDevice, mTimeline, nullptr);
            if (mCommandPool)
                vkDestroyCommandPool(mDevice, mCommandPool, nullptr);
            if (mFence)
                vkDestroyFence(mDevice, mFence, nullptr);
            vkDestroyDevice(mDevice, nullptr);
        }
        if (mInstance)
            vkDestroyInstance(mInstance, nullptr);
    }

    bool init(const PxgSlangDeviceDesc& desc)
    {
        // ---- instance ----
        // Vulkan 1.2+ is required: slangc emits SPIR-V 1.4/1.5 modules.
        uint32_t instanceVersion = VK_API_VERSION_1_0;
        vkEnumerateInstanceVersion(&instanceVersion);
        if (instanceVersion < VK_API_VERSION_1_2)
            return false;

        VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
        appInfo.pApplicationName = "PhysX gpuslang";
        appInfo.apiVersion = instanceVersion >= VK_API_VERSION_1_3 ? VK_API_VERSION_1_3 : VK_API_VERSION_1_2;

        std::vector<const char*> layers;
        if (desc.enableValidation)
            layers.push_back("VK_LAYER_KHRONOS_validation");

        VkInstanceCreateInfo instInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        instInfo.pApplicationInfo = &appInfo;
        instInfo.enabledLayerCount = uint32_t(layers.size());
        instInfo.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();

        if (vkCreateInstance(&instInfo, nullptr, &mInstance) != VK_SUCCESS)
            return false;

        // ---- physical device ----
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(mInstance, &count, nullptr);
        if (count == 0)
            return false;
        std::vector<VkPhysicalDevice> physicalDevices(count);
        vkEnumeratePhysicalDevices(mInstance, &count, physicalDevices.data());

        VkPhysicalDevice chosen = VK_NULL_HANDLE;
        for (VkPhysicalDevice pd : physicalDevices)
        {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(pd, &props);
            if (props.apiVersion < VK_API_VERSION_1_2)
                continue; // the kernels are SPIR-V 1.4/1.5 modules
            const bool isSoftware = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
            const bool isDiscrete = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;

            if (chosen == VK_NULL_HANDLE)
                chosen = pd;
            else if (desc.preferSoftwareDevice && isSoftware)
                chosen = pd;
            else if (!desc.preferSoftwareDevice && isDiscrete)
                chosen = pd;
        }
        if (chosen == VK_NULL_HANDLE)
            return false;
        mPhysicalDevice = chosen;

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(mPhysicalDevice, &props);
        mDeviceName = props.deviceName;
        mTimestampPeriod = props.limits.timestampPeriod;   // ns per tick, 0 => timestamps unusable

        // ---- queue family ----
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(mPhysicalDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(mPhysicalDevice, &queueFamilyCount, families.data());

        mQueueFamily = uint32_t(-1);
        for (uint32_t i = 0; i < queueFamilyCount; ++i)
        {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            {
                mQueueFamily = i;
                mTimestampValidBits = families[i].timestampValidBits;
                break;
            }
        }
        if (mQueueFamily == uint32_t(-1))
            return false;

        // ---- logical device ----
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        queueInfo.queueFamilyIndex = mQueueFamily;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &priority;

        // Query core + extended features so the Phase-2 runtime profile
        // (RUNTIME_PLAN.md section 1) can be enabled where offered; everything
        // is guarded so reduced devices still create successfully.
        // VK_EXT_shader_atomic_float: OpAtomicFAddEXT on a buffer-device-address
        // float.  This is what CUDA's atomicAdd(float*) lowers to, and several
        // deformable kernels genuinely need it -- their contact solves accumulate
        // per-vertex position deltas from contacts that DO share vertices, so a
        // plain read-modify-write silently loses updates under contention.
        // Enabling the extension is gated on the device advertising it; shaders
        // that emit OpAtomicFAddEXT will fail pipeline creation without it, so the
        // capability is recorded and reported rather than silently assumed.
        bool atomicFloatExtPresent = false;
        {
            uint32_t extCount = 0;
            vkEnumerateDeviceExtensionProperties(mPhysicalDevice, nullptr, &extCount, nullptr);
            std::vector<VkExtensionProperties> exts(extCount);
            if (extCount)
                vkEnumerateDeviceExtensionProperties(mPhysicalDevice, nullptr, &extCount, exts.data());
            for (const VkExtensionProperties& e : exts)
                if (std::strcmp(e.extensionName, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) == 0)
                    atomicFloatExtPresent = true;
        }

        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatSupported = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT
        };
        VkPhysicalDeviceSubgroupSizeControlFeatures sgSizeSupported = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES
        };
        if (atomicFloatExtPresent)
            sgSizeSupported.pNext = &atomicFloatSupported;
        VkPhysicalDeviceVulkan12Features supported12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
        supported12.pNext = &sgSizeSupported;
        VkPhysicalDeviceFeatures2 supported2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        supported2.pNext = &supported12;
        vkGetPhysicalDeviceFeatures2(mPhysicalDevice, &supported2);
        const VkPhysicalDeviceFeatures& supported = supported2.features;

        // Core features: shaderInt64 (PxU64 accumulators, e.g. the broad-phase
        // overlap-check region scan) and shaderFloat64 (kernels transcribing
        // CUDA code that mixes double literals into float expressions, e.g.
        // the sb_extractRotationAPD Rodrigues update '(1.0 - l2)/(1.0f + l2)',
        // must perform those spots in f64 to match the C++ reference rounding).
        VkPhysicalDeviceFeatures2 enabled2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        enabled2.features.shaderInt64 = supported.shaderInt64;
        mHasShaderInt64 = supported.shaderInt64 == VK_TRUE;
        enabled2.features.shaderFloat64 = supported.shaderFloat64;

        // Vulkan 1.2 features: buffer device address carries the CUDA
        // device-pointer argument model; timeline semaphores back the v2
        // submission path.
        VkPhysicalDeviceVulkan12Features enabled12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
        enabled12.bufferDeviceAddress = supported12.bufferDeviceAddress;
        enabled12.timelineSemaphore = supported12.timelineSemaphore;
        mHasBufferDeviceAddress = supported12.bufferDeviceAddress == VK_TRUE;

        // Descriptor indexing (P9.12a): what makes a texture handle stored in a
        // device buffer sampleable, which is the CUtexObject model the SDF path
        // is written against.  All five are Vulkan 1.2 CORE features, so there
        // is no extension to add -- only features to turn on, and only if the
        // device has them.  updateAfterBind is required because textures are
        // created while earlier dispatches may still reference the set.
        mHasBindlessTextures =
            supported12.descriptorIndexing == VK_TRUE &&
            supported12.runtimeDescriptorArray == VK_TRUE &&
            supported12.shaderSampledImageArrayNonUniformIndexing == VK_TRUE &&
            supported12.descriptorBindingPartiallyBound == VK_TRUE &&
            supported12.descriptorBindingSampledImageUpdateAfterBind == VK_TRUE;
        if (mHasBindlessTextures)
        {
            enabled12.descriptorIndexing = VK_TRUE;
            enabled12.runtimeDescriptorArray = VK_TRUE;
            enabled12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
            enabled12.descriptorBindingPartiallyBound = VK_TRUE;
            enabled12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
        }

        // Subgroup size control: the warp-32 primitive library requires an
        // exact subgroup size on hardware targets (llvmpipe emulates instead).
        VkPhysicalDeviceSubgroupSizeControlFeatures enabledSgSize = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES
        };
        enabledSgSize.subgroupSizeControl = sgSizeSupported.subgroupSizeControl;
        enabledSgSize.computeFullSubgroups = sgSizeSupported.computeFullSubgroups;

        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT enabledAtomicFloat = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT
        };
        enabledAtomicFloat.shaderBufferFloat32AtomicAdd = atomicFloatSupported.shaderBufferFloat32AtomicAdd;
        enabledAtomicFloat.shaderBufferFloat32Atomics = atomicFloatSupported.shaderBufferFloat32Atomics;
        mHasShaderBufferFloat32AtomicAdd =
            atomicFloatExtPresent && atomicFloatSupported.shaderBufferFloat32AtomicAdd == VK_TRUE;

        enabled2.pNext = &enabled12;
        enabled12.pNext = &enabledSgSize;
        if (mHasShaderBufferFloat32AtomicAdd)
            enabledSgSize.pNext = &enabledAtomicFloat;

        VkPhysicalDeviceSubgroupSizeControlProperties sgSizeProps = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES
        };
        VkPhysicalDeviceSubgroupProperties sgProps = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
        sgProps.pNext = &sgSizeProps;
        VkPhysicalDeviceProperties2 props2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        props2.pNext = &sgProps;
        vkGetPhysicalDeviceProperties2(mPhysicalDevice, &props2);
        mSubgroupCaps.defaultSize = sgProps.subgroupSize;
        mSubgroupCaps.minSize = sgSizeProps.minSubgroupSize;
        mSubgroupCaps.maxSize = sgSizeProps.maxSubgroupSize;
        mSubgroupCaps.canRequireSize = enabledSgSize.subgroupSizeControl == VK_TRUE &&
                                       (sgSizeProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
        mSubgroupCaps.fullSubgroups = enabledSgSize.computeFullSubgroups == VK_TRUE;

        std::vector<const char*> deviceExts;
        if (mHasShaderBufferFloat32AtomicAdd)
            deviceExts.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);

        VkDeviceCreateInfo deviceInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        deviceInfo.pNext = &enabled2;
        deviceInfo.enabledExtensionCount = uint32_t(deviceExts.size());
        deviceInfo.ppEnabledExtensionNames = deviceExts.empty() ? nullptr : deviceExts.data();

        if (vkCreateDevice(mPhysicalDevice, &deviceInfo, nullptr, &mDevice) != VK_SUCCESS)
            return false;

        vkGetDeviceQueue(mDevice, mQueueFamily, 0, &mQueue);

        if (mHasBindlessTextures && !initBindlessTextures())
        {
            // Losing the bindless path is not fatal -- everything that does not
            // sample keeps working -- but it must be REPORTED rather than
            // silently degraded, because the only symptom downstream would be
            // an SDF kernel reading zeros.
            std::fprintf(stderr, "[PxgVulkan] bindless texture set could not be created; "
                                 "SDF sampling is unavailable on this device\n");
            mHasBindlessTextures = false;
        }

        // ---- command pool / buffer / fence ----
        VkCommandPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = mQueueFamily;
        PXG_VK_CHECK(vkCreateCommandPool(mDevice, &poolInfo, nullptr, &mCommandPool));

        VkCommandBufferAllocateInfo cmdInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cmdInfo.commandPool = mCommandPool;
        cmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdInfo.commandBufferCount = kFramesInFlight;
        VkCommandBuffer cmds[kFramesInFlight] = {};
        PXG_VK_CHECK(vkAllocateCommandBuffers(mDevice, &cmdInfo, cmds));
        for (uint32_t i = 0; i < kFramesInFlight; ++i)
            mFrames[i].cmd = cmds[i];

        VkFenceCreateInfo fenceInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        PXG_VK_CHECK(vkCreateFence(mDevice, &fenceInfo, nullptr, &mFence));

        // One timeline semaphore per queue: every submit signals the next
        // monotonic value, so any past submission can be waited on or polled
        // by value without fences (RUNTIME_PLAN.md P1.1).
        if (enabled12.timelineSemaphore)
        {
            VkSemaphoreTypeCreateInfo typeInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
            typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            typeInfo.initialValue = 0;
            VkSemaphoreCreateInfo semInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
            semInfo.pNext = &typeInfo;
            PXG_VK_CHECK(vkCreateSemaphore(mDevice, &semInfo, nullptr, &mTimeline));
            mHasTimeline = mTimeline != VK_NULL_HANDLE;
        }

        mDesc = desc;
        if (desc.pipelineCachePath)
            mPipelineCachePath = desc.pipelineCachePath;

        // Timestamp query pool for per-stage GPU timing.  Only usable when the
        // compute queue reports valid timestamp bits and a nonzero period.
        if (mTimestampValidBits > 0 && mTimestampPeriod > 0.0f)
        {
            VkQueryPoolCreateInfo qpInfo = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
            qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qpInfo.queryCount = kMaxTimestamps;
            if (vkCreateQueryPool(mDevice, &qpInfo, nullptr, &mQueryPool) == VK_SUCCESS)
                mHasTimestamps = true;
        }

        // Pipeline cache: seed from the on-disk blob when present.  An invalid
        // or foreign blob is rejected by the driver's header check, which
        // simply starts an empty cache — safe by design.
        std::vector<uint8_t> cacheSeed;
        if (!mPipelineCachePath.empty())
        {
            if (FILE* f = fopen(mPipelineCachePath.c_str(), "rb"))
            {
                fseek(f, 0, SEEK_END);
                const long sz = ftell(f);
                fseek(f, 0, SEEK_SET);
                if (sz > 0)
                {
                    cacheSeed.resize(size_t(sz));
                    if (fread(cacheSeed.data(), 1, size_t(sz), f) != size_t(sz))
                        cacheSeed.clear();
                }
                fclose(f);
            }
        }
        VkPipelineCacheCreateInfo cacheInfo = { VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
        cacheInfo.initialDataSize = cacheSeed.size();
        cacheInfo.pInitialData = cacheSeed.empty() ? nullptr : cacheSeed.data();
        PXG_VK_CHECK(vkCreatePipelineCache(mDevice, &cacheInfo, nullptr, &mPipelineCache));

        vkGetPhysicalDeviceMemoryProperties(mPhysicalDevice, &mMemoryProperties);

        return true;
    }

    const char* getDeviceName() const override { return mDeviceName.c_str(); }
    const char* getBackendName() const override { return "Vulkan"; }

    // ---- resources ----

    PxgBufferHandle createBuffer(size_t sizeBytes) override
    {
        if (sizeBytes == 0)
            sizeBytes = 4;

        VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufferInfo.size = sizeBytes;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        if (mHasBufferDeviceAddress)
            bufferInfo.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        Buffer buf = {};
        buf.size = sizeBytes;
        PXG_VK_CHECK(vkCreateBuffer(mDevice, &bufferInfo, nullptr, &buf.buffer));

        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(mDevice, buf.buffer, &memReq);

        // v2 memory mode: DEVICE_LOCAL storage with staged transfers.  The
        // staged path is taken even when the winning memory type happens to
        // also be host-visible (UMA/software devices) so the production
        // transfer path is exercised everywhere, CI included.
        buf.hostVisible = !mDesc.useDeviceLocalMemory;
        const uint32_t memoryType = buf.hostVisible
            ? findMemoryType(memReq.memoryTypeBits,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
            : findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        VkMemoryAllocateFlagsInfo allocFlags = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
        allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

        VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocInfo.pNext = mHasBufferDeviceAddress ? &allocFlags : nullptr;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = memoryType;
        PXG_VK_CHECK(vkAllocateMemory(mDevice, &allocInfo, nullptr, &buf.memory));
        PXG_VK_CHECK(vkBindBufferMemory(mDevice, buf.buffer, buf.memory, 0));

        if (mHasBufferDeviceAddress)
        {
            VkBufferDeviceAddressInfo addrInfo = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
            addrInfo.buffer = buf.buffer;
            buf.deviceAddress = vkGetBufferDeviceAddress(mDevice, &addrInfo);
        }

        const PxgBufferHandle handle = mNextBufferHandle++;
        mBuffers[handle] = buf;
        return handle;
    }

    // Pinned-host buffer (RUNTIME_PLAN.md P5 Step 4b): the CUDA cuMemHostAlloc
    // model — memory the CPU writes/reads through a persistent mapping while
    // shaders access it by buffer device address.  Forces HOST_VISIBLE|HOST_
    // COHERENT in BOTH memory modes (v2's device-local policy applies to bulk
    // simulation buffers, not the pinned mirrors), maps it for the buffer's
    // lifetime, and hands the mapping back.  Host writes made before a queue
    // submission are visible to the device (implicit domain acquire at submit);
    // device writes become host-visible at the flush/wait sync points.
    PxgBufferHandle createHostVisibleBuffer(size_t sizeBytes, void** mappedPtr) override
    {
        if (sizeBytes == 0)
            sizeBytes = 4;

        VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufferInfo.size = sizeBytes;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        if (mHasBufferDeviceAddress)
            bufferInfo.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        Buffer buf = {};
        buf.size = sizeBytes;
        buf.hostVisible = true;
        PXG_VK_CHECK(vkCreateBuffer(mDevice, &bufferInfo, nullptr, &buf.buffer));

        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(mDevice, buf.buffer, &memReq);

        VkMemoryAllocateFlagsInfo allocFlags = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
        allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

        VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocInfo.pNext = mHasBufferDeviceAddress ? &allocFlags : nullptr;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        PXG_VK_CHECK(vkAllocateMemory(mDevice, &allocInfo, nullptr, &buf.memory));
        PXG_VK_CHECK(vkBindBufferMemory(mDevice, buf.buffer, buf.memory, 0));

        if (mHasBufferDeviceAddress)
        {
            VkBufferDeviceAddressInfo addrInfo = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
            addrInfo.buffer = buf.buffer;
            buf.deviceAddress = vkGetBufferDeviceAddress(mDevice, &addrInfo);
        }

        PXG_VK_CHECK(vkMapMemory(mDevice, buf.memory, 0, VK_WHOLE_SIZE, 0, &buf.mappedPtr));
        if (mappedPtr)
            *mappedPtr = buf.mappedPtr;

        const PxgBufferHandle handle = mNextBufferHandle++;
        mBuffers[handle] = buf;
        return handle;
    }

    void destroyBuffer(PxgBufferHandle handle) override
    {
        auto it = mBuffers.find(handle);
        if (it == mBuffers.end())
            return;
        // An open recording may reference this buffer (staged uploads record
        // copies); submit it so the wait below actually covers that work.
        flush();
        vkDeviceWaitIdle(mDevice);
        if (it->second.mappedPtr)
            vkUnmapMemory(mDevice, it->second.memory);
        vkDestroyBuffer(mDevice, it->second.buffer, nullptr);
        vkFreeMemory(mDevice, it->second.memory, nullptr);
        mBuffers.erase(it);
    }

    void upload(PxgBufferHandle handle, const void* data, size_t sizeBytes, size_t dstOffset) override
    {
        if (sizeBytes == 0)
            return;
        Buffer& buf = mBuffers[handle];
        if (buf.mappedPtr)
        {
            // Persistently mapped pinned buffer: same stream-ordering drain,
            // then write through the standing mapping (re-mapping is illegal).
            flush();
            memcpy(static_cast<uint8_t*>(buf.mappedPtr) + dstOffset, data, sizeBytes);
            return;
        }
        if (buf.hostVisible)
        {
            // cuMemcpyHtoD is stream-ordered: drain prior recorded/in-flight
            // GPU work before the direct host write, so a fillBuffer/copyBuffer
            // or dispatch cannot clobber this upload out of order.  Cheap when
            // idle (submit returns the last ticket, already retired).
            flush();
            void* mapped = nullptr;
            PXG_VK_CHECK(vkMapMemory(mDevice, buf.memory, dstOffset, sizeBytes, 0, &mapped));
            memcpy(mapped, data, sizeBytes);
            vkUnmapMemory(mDevice, buf.memory);
            return;
        }

        // Staged path: copy through the current frame's persistently mapped
        // staging ring, recorded into the open command buffer so the copy
        // orders exactly like a stream-ordered H2D copy — after previously
        // recorded dispatches, before subsequent ones.
        const uint8_t* src = static_cast<const uint8_t*>(data);
        size_t remaining = sizeBytes;
        size_t off = dstOffset;
        while (remaining)
        {
            beginRecording();
            Frame& f = mFrames[mFrameIndex];
            ensureStaging(f);
            const size_t avail = kStagingBytes - f.stagingOffset;
            if (avail == 0)
            {
                flush(); // ring exhausted: retire this batch, continue on a fresh frame
                continue;
            }
            const size_t chunk = remaining < avail ? remaining : avail;
            memcpy(f.stagingMapped + f.stagingOffset, src, chunk);

            bufferBarrier(buf.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            VkBufferCopy region = { f.stagingOffset, off, chunk };
            vkCmdCopyBuffer(mCommandBuffer, f.staging, buf.buffer, 1, &region);
            bufferBarrier(buf.buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

            f.stagingOffset += chunk;
            src += chunk;
            off += chunk;
            remaining -= chunk;
        }
    }

    void download(PxgBufferHandle handle, void* data, size_t sizeBytes, size_t srcOffset) override
    {
        if (sizeBytes == 0)
            return;
        Buffer& buf = mBuffers[handle];
        if (buf.mappedPtr)
        {
            // Persistently mapped pinned buffer: same device->host visibility
            // barrier + drain, then read through the standing mapping.
            beginRecording();
            bufferBarrier(buf.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
            flush();
            memcpy(data, static_cast<const uint8_t*>(buf.mappedPtr) + srcOffset, sizeBytes);
            return;
        }
        if (buf.hostVisible)
        {
            // cuMemcpyDtoH is stream-ordered: drain any recorded GPU work
            // (dispatches, copyBuffer/fillBuffer) that may write this buffer
            // and, with a device->host domain barrier, make those writes
            // visible to the host read.  On same-queue submission order the
            // barrier's first scope covers prior submitted batches, so this
            // closes the device->host visibility edge that HOST_COHERENT alone
            // does not guarantee.
            beginRecording();
            bufferBarrier(buf.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
            flush();
            void* mapped = nullptr;
            PXG_VK_CHECK(vkMapMemory(mDevice, buf.memory, srcOffset, sizeBytes, 0, &mapped));
            memcpy(data, mapped, sizeBytes);
            vkUnmapMemory(mDevice, buf.memory);
            return;
        }

        // Staged path: everything recorded so far must have retired before
        // the readback copy of a device-local buffer is recorded.
        flush();
        uint8_t* dst = static_cast<uint8_t*>(data);
        size_t remaining = sizeBytes;
        size_t off = srcOffset;
        while (remaining)
        {
            beginRecording();
            Frame& f = mFrames[mFrameIndex];
            ensureStaging(f);
            const size_t chunk = remaining < kStagingBytes ? remaining : kStagingBytes;

            bufferBarrier(buf.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            VkBufferCopy region = { off, 0, chunk };
            vkCmdCopyBuffer(mCommandBuffer, buf.buffer, f.staging, 1, &region);
            bufferBarrier(f.staging, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);

            Frame* recorded = &f;
            flush();
            memcpy(dst, recorded->stagingMapped, chunk);

            dst += chunk;
            off += chunk;
            remaining -= chunk;
        }
    }

    // Device-to-device copy recorded into the current stream (cuMemcpyDtoD
    // parity): compute<->transfer barriers bracket the copy so it orders
    // after prior writes and before subsequent reads on either buffer.
    void copyBuffer(PxgBufferHandle dstHandle, size_t dstOffset, PxgBufferHandle srcHandle,
                    size_t srcOffset, size_t sizeBytes) override
    {
        if (sizeBytes == 0)
            return;
        const Buffer& src = mBuffers[srcHandle];
        const Buffer& dst = mBuffers[dstHandle];
        beginRecording();
        bufferBarrier(src.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        bufferBarrier(dst.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        VkBufferCopy region = { srcOffset, dstOffset, sizeBytes };
        vkCmdCopyBuffer(mCommandBuffer, src.buffer, dst.buffer, 1, &region);
        bufferBarrier(dst.buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }

    // 32-bit fill (cuMemsetD32 parity) via vkCmdFillBuffer; offset/size are
    // 4-byte aligned by contract.
    void fillBuffer(PxgBufferHandle dstHandle, size_t dstOffset, uint32_t value, size_t sizeBytes) override
    {
        if (sizeBytes == 0)
            return;
        // vkCmdFillBuffer requires 4-byte aligned offset and size (the
        // cuMemsetD32 contract); a misaligned device pointer is caller error.
        assert((dstOffset % 4) == 0 && (sizeBytes % 4) == 0);
        const Buffer& dst = mBuffers[dstHandle];
        beginRecording();
        bufferBarrier(dst.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        vkCmdFillBuffer(mCommandBuffer, dst.buffer, dstOffset, sizeBytes, value);
        bufferBarrier(dst.buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }

    // ---- textures ----

    bool supportsBindlessTextures() const override { return mHasBindlessTextures; }

    PxgTextureHandle createTexture3D(uint32_t width, uint32_t height, uint32_t depth,
                                     const float* texels) override
    {
        return createTexture3DInternal(width, height, depth, TextureFormat::R32Float, texels, false);
    }

    PxgTextureHandle createTexture3DFmt(uint32_t width, uint32_t height, uint32_t depth,
                                        TextureFormat format, const void* texels) override
    {
        if (!mHasBindlessTextures)
            return 0;
        return createTexture3DInternal(width, height, depth, format, texels, true);
    }

    static VkFormat vkFormatOf(TextureFormat f)
    {
        switch (f)
        {
        case TextureFormat::R8Unorm:  return VK_FORMAT_R8_UNORM;
        case TextureFormat::R16Unorm: return VK_FORMAT_R16_UNORM;
        default:                      return VK_FORMAT_R32_SFLOAT;
        }
    }

    static size_t texelSizeOf(TextureFormat f)
    {
        switch (f)
        {
        case TextureFormat::R8Unorm:  return 1;
        case TextureFormat::R16Unorm: return 2;
        default:                      return 4;
        }
    }

    PxgTextureHandle createTexture3DInternal(uint32_t width, uint32_t height, uint32_t depth,
                                             TextureFormat format, const void* texels, bool bindless)
    {
        Texture tex;
        tex.width = width;
        tex.height = height;
        tex.depth = depth;
        tex.bindless = bindless;
        const VkFormat vkFormat = vkFormatOf(format);

        VkImageCreateInfo imageInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.imageType = VK_IMAGE_TYPE_3D;
        imageInfo.format = vkFormat;
        imageInfo.extent = { width, height, depth };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        PXG_VK_CHECK(vkCreateImage(mDevice, &imageInfo, nullptr, &tex.image));

        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(mDevice, tex.image, &req);
        VkMemoryAllocateInfo alloc = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        PXG_VK_CHECK(vkAllocateMemory(mDevice, &alloc, nullptr, &tex.memory));
        PXG_VK_CHECK(vkBindImageMemory(mDevice, tex.image, tex.memory, 0));

        // One-shot staging upload with the required layout transitions.
        const size_t bytes = size_t(width) * height * depth * texelSizeOf(format);
        VkBuffer staging = VK_NULL_HANDLE;
        VkDeviceMemory stagingMem = VK_NULL_HANDLE;
        {
            VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            bi.size = bytes;
            bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            PXG_VK_CHECK(vkCreateBuffer(mDevice, &bi, nullptr, &staging));
            VkMemoryRequirements sreq;
            vkGetBufferMemoryRequirements(mDevice, staging, &sreq);
            VkMemoryAllocateInfo sa = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
            sa.allocationSize = sreq.size;
            sa.memoryTypeIndex = findMemoryType(
                sreq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            PXG_VK_CHECK(vkAllocateMemory(mDevice, &sa, nullptr, &stagingMem));
            PXG_VK_CHECK(vkBindBufferMemory(mDevice, staging, stagingMem, 0));
            void* mapped = nullptr;
            PXG_VK_CHECK(vkMapMemory(mDevice, stagingMem, 0, VK_WHOLE_SIZE, 0, &mapped));
            memcpy(mapped, texels, bytes);
            vkUnmapMemory(mDevice, stagingMem);
        }

        beginRecording();
        imageBarrier(tex.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     0, VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy region = {};
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageExtent = { width, height, depth };
        vkCmdCopyBufferToImage(mCommandBuffer, staging, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        imageBarrier(tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        flush();

        vkDestroyBuffer(mDevice, staging, nullptr);
        vkFreeMemory(mDevice, stagingMem, nullptr);

        VkImageViewCreateInfo viewInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        viewInfo.image = tex.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
        viewInfo.format = vkFormat;
        viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        PXG_VK_CHECK(vkCreateImageView(mDevice, &viewInfo, nullptr, &tex.view));

        // The CUDA SDF sampler configuration: trilinear, clamp-to-edge,
        // element-type reads, no mips.
        VkSamplerCreateInfo samplerInfo = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (!bindless)
            PXG_VK_CHECK(vkCreateSampler(mDevice, &samplerInfo, nullptr, &tex.sampler));

        const PxgTextureHandle handle = mNextTextureHandle++;
        if (bindless)
        {
            // The handle IS the descriptor-array slot, which is what makes it
            // storable in a device buffer.  Overflowing the array would alias
            // one texture onto another silently, so it is refused instead.
            if (handle >= kBindlessTextureCapacity)
            {
                std::fprintf(stderr, "[PxgVulkan] bindless texture capacity (%u) exhausted\n",
                             kBindlessTextureCapacity);
                vkDestroyImageView(mDevice, tex.view, nullptr);
                vkDestroyImage(mDevice, tex.image, nullptr);
                vkFreeMemory(mDevice, tex.memory, nullptr);
                return 0;
            }
            writeBindlessSlot(uint32_t(handle), tex.view);
        }
        mTextures[handle] = tex;
        return handle;
    }

    // ---- the bindless set ----

    bool initBindlessTextures()
    {
        VkDescriptorSetLayoutBinding bindings[2] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        bindings[0].descriptorCount = kBindlessTextureCapacity;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // PARTIALLY_BOUND: slots for textures that do not exist yet (and slot 0,
        // the reserved null handle) are never written and never read.
        // UPDATE_AFTER_BIND: createTexture3DFmt may run while a previously
        // recorded dispatch still references the set.
        const VkDescriptorBindingFlags flags[2] = {
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            0
        };
        VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO
        };
        flagsInfo.bindingCount = 2;
        flagsInfo.pBindingFlags = flags;

        VkDescriptorSetLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = 2;
        layoutInfo.pBindings = bindings;
        layoutInfo.pNext = &flagsInfo;
        layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        if (vkCreateDescriptorSetLayout(mDevice, &layoutInfo, nullptr, &mBindlessSetLayout) != VK_SUCCESS)
            return false;

        VkDescriptorPoolSize poolSizes[2] = {};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        poolSizes[0].descriptorCount = kBindlessTextureCapacity;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLER;
        poolSizes[1].descriptorCount = 1;
        VkDescriptorPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        if (vkCreateDescriptorPool(mDevice, &poolInfo, nullptr, &mBindlessPool) != VK_SUCCESS)
            return false;

        VkDescriptorSetAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.descriptorPool = mBindlessPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &mBindlessSetLayout;
        if (vkAllocateDescriptorSets(mDevice, &allocInfo, &mBindlessSet) != VK_SUCCESS)
            return false;

        // The engine's SDF sampler, verbatim: clamp on all three axes, linear
        // filter, no mips (PxgGeometryManager.cpp:761-769).
        VkSamplerCreateInfo samplerInfo = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(mDevice, &samplerInfo, nullptr, &mBindlessSampler) != VK_SUCCESS)
            return false;

        VkDescriptorImageInfo smpInfo = {};
        smpInfo.sampler = mBindlessSampler;
        VkWriteDescriptorSet write = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        write.dstSet = mBindlessSet;
        write.dstBinding = 1;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        write.pImageInfo = &smpInfo;
        vkUpdateDescriptorSets(mDevice, 1, &write, 0, nullptr);
        return true;
    }

    void writeBindlessSlot(uint32_t slot, VkImageView view)
    {
        VkDescriptorImageInfo imgInfo = {};
        imgInfo.imageView = view;
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet write = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        write.dstSet = mBindlessSet;
        write.dstBinding = 0;
        write.dstArrayElement = slot;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        write.pImageInfo = &imgInfo;
        vkUpdateDescriptorSets(mDevice, 1, &write, 0, nullptr);
    }

    void destroyTexture3D(PxgTextureHandle handle) override
    {
        auto it = mTextures.find(handle);
        if (it == mTextures.end())
            return;
        flush();
        vkDeviceWaitIdle(mDevice);
        releaseTexture(it->second);
        mTextures.erase(it);
    }

    // ---- pipelines ----

    PxgPipelineHandle createPipeline(const void* code, size_t codeSize, const char* entryName,
                                     uint32_t numBuffers, const uint32_t* bufferStrides,
                                     uint32_t pushConstantSize) override
    {
        return createPipelineInternal(code, codeSize, entryName, numBuffers, bufferStrides, pushConstantSize,
                                      0, 0, false);
    }

    PxgPipelineHandle createPipelineEx(const void* code, size_t codeSize, const char* entryName,
                                       uint32_t numBuffers, const uint32_t* bufferStrides,
                                       uint32_t pushConstantSize, uint32_t requiredSubgroupSize) override
    {
        if (requiredSubgroupSize != 0 &&
            (!mSubgroupCaps.canRequireSize || requiredSubgroupSize < mSubgroupCaps.minSize ||
             requiredSubgroupSize > mSubgroupCaps.maxSize))
            return 0;
        return createPipelineInternal(code, codeSize, entryName, numBuffers, bufferStrides, pushConstantSize,
                                      requiredSubgroupSize, 0, false);
    }

    PxgPipelineHandle createPipelineTex(const void* code, size_t codeSize, const char* entryName,
                                        uint32_t numBuffers, const uint32_t* bufferStrides,
                                        uint32_t pushConstantSize, uint32_t numTextures) override
    {
        return createPipelineInternal(code, codeSize, entryName, numBuffers, bufferStrides, pushConstantSize,
                                      0, numTextures, false);
    }

    PxgPipelineHandle createPipelineBindless(const void* code, size_t codeSize, const char* entryName,
                                             uint32_t numBuffers, const uint32_t* bufferStrides,
                                             uint32_t pushConstantSize) override
    {
        if (!mHasBindlessTextures)
            return 0;
        return createPipelineInternal(code, codeSize, entryName, numBuffers, bufferStrides, pushConstantSize,
                                      0, 0, true);
    }

    SubgroupCaps getSubgroupCaps() const override { return mSubgroupCaps; }

    uint64_t getBufferDeviceAddress(PxgBufferHandle handle) const override
    {
        auto it = mBuffers.find(handle);
        return it == mBuffers.end() ? 0 : it->second.deviceAddress;
    }

    PxgPipelineHandle createPipelineInternal(const void* code, size_t codeSize, const char* entryName,
                                             uint32_t numBuffers, const uint32_t* bufferStrides,
                                             uint32_t pushConstantSize, uint32_t requiredSubgroupSize,
                                             uint32_t numTextures, bool bindless)
    {
        (void)bufferStrides; // SPIR-V carries the layout; strides are for D3D12

        Pipeline pipe = {};
        pipe.numBuffers = numBuffers;
        pipe.numTextures = numTextures;
        pipe.pushConstantSize = pushConstantSize;
        pipe.bindless = bindless;

        VkShaderModuleCreateInfo moduleInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        moduleInfo.codeSize = codeSize;
        moduleInfo.pCode = reinterpret_cast<const uint32_t*>(code);
        PXG_VK_CHECK(vkCreateShaderModule(mDevice, &moduleInfo, nullptr, &pipe.module));

        // Storage buffers at 0..numBuffers-1; each texture is a sampled
        // image + sampler pair right after (Slang emits Texture3D and
        // SamplerState as separate descriptors in declaration order).
        const uint32_t numBindings = numBuffers + 2 * numTextures;
        std::vector<VkDescriptorSetLayoutBinding> bindings(numBindings);
        for (uint32_t i = 0; i < numBuffers; ++i)
        {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        for (uint32_t t = 0; t < numTextures; ++t)
        {
            VkDescriptorSetLayoutBinding& img = bindings[numBuffers + 2 * t];
            img.binding = numBuffers + 2 * t;
            img.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            img.descriptorCount = 1;
            img.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            VkDescriptorSetLayoutBinding& smp = bindings[numBuffers + 2 * t + 1];
            smp.binding = numBuffers + 2 * t + 1;
            smp.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            smp.descriptorCount = 1;
            smp.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo setLayoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        setLayoutInfo.bindingCount = numBindings;
        setLayoutInfo.pBindings = bindings.empty() ? nullptr : bindings.data();
        PXG_VK_CHECK(vkCreateDescriptorSetLayout(mDevice, &setLayoutInfo, nullptr, &pipe.setLayout));

        VkPushConstantRange pushRange = {};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = pushConstantSize;

        // Set 0 is the pipeline's own bindings; set 1, when declared, is the
        // device-wide bindless texture set.
        const VkDescriptorSetLayout setLayouts[2] = { pipe.setLayout, mBindlessSetLayout };
        VkPipelineLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = bindless ? 2u : 1u;
        layoutInfo.pSetLayouts = setLayouts;
        layoutInfo.pushConstantRangeCount = pushConstantSize > 0 ? 1 : 0;
        layoutInfo.pPushConstantRanges = pushConstantSize > 0 ? &pushRange : nullptr;
        PXG_VK_CHECK(vkCreatePipelineLayout(mDevice, &layoutInfo, nullptr, &pipe.layout));

        VkPipelineShaderStageCreateInfo stageInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = pipe.module;
        stageInfo.pName = entryName;

        // Exact-subgroup-size request (warp-32 primitive library on hardware
        // targets; validated against SubgroupCaps by createPipelineEx).
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sgSizeInfo = {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO
        };
        if (requiredSubgroupSize != 0)
        {
            sgSizeInfo.requiredSubgroupSize = requiredSubgroupSize;
            stageInfo.pNext = &sgSizeInfo;
            if (mSubgroupCaps.fullSubgroups)
                stageInfo.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        }

        VkComputePipelineCreateInfo pipelineInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = pipe.layout;
        PXG_VK_CHECK(vkCreateComputePipelines(mDevice, mPipelineCache, 1, &pipelineInfo, nullptr, &pipe.pipeline));

        const PxgPipelineHandle handle = mNextPipelineHandle++;
        mPipelines[handle] = pipe;
        return handle;
    }

    void destroyPipeline(PxgPipelineHandle handle) override
    {
        auto it = mPipelines.find(handle);
        if (it == mPipelines.end())
            return;
        // An open recording may reference this pipeline; submit it first.
        flush();
        vkDeviceWaitIdle(mDevice);
        vkDestroyPipeline(mDevice, it->second.pipeline, nullptr);
        vkDestroyPipelineLayout(mDevice, it->second.layout, nullptr);
        vkDestroyDescriptorSetLayout(mDevice, it->second.setLayout, nullptr);
        vkDestroyShaderModule(mDevice, it->second.module, nullptr);
        mPipelines.erase(it);
    }

    // ---- execution ----

    void dispatch(PxgPipelineHandle pipelineHandle, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                  const PxgBufferHandle* buffers, uint32_t numBuffers,
                  const void* pushConstants, uint32_t pushConstantSize) override
    {
        dispatchInternal(pipelineHandle, gridX, gridY, gridZ, buffers, numBuffers, nullptr, 0,
                         pushConstants, pushConstantSize);
    }

    void dispatchTex(PxgPipelineHandle pipelineHandle, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                     const PxgBufferHandle* buffers, uint32_t numBuffers,
                     const PxgTextureHandle* textures, uint32_t numTextures,
                     const void* pushConstants, uint32_t pushConstantSize) override
    {
        dispatchInternal(pipelineHandle, gridX, gridY, gridZ, buffers, numBuffers, textures, numTextures,
                         pushConstants, pushConstantSize);
    }

    void dispatchInternal(PxgPipelineHandle pipelineHandle, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                          const PxgBufferHandle* buffers, uint32_t numBuffers,
                          const PxgTextureHandle* textures, uint32_t numTextures,
                          const void* pushConstants, uint32_t pushConstantSize)
    {
        Pipeline& pipe = mPipelines[pipelineHandle];
        assert(numBuffers == pipe.numBuffers);
        assert(numTextures == pipe.numTextures);
        assert(pushConstantSize == pipe.pushConstantSize);

        beginRecording();

        const uint32_t numBindings = numBuffers + 2 * numTextures;
        VkDescriptorSet set = allocateDescriptorSet(pipe.setLayout, numBindings);

        std::vector<VkDescriptorBufferInfo> bufferInfos(numBuffers);
        std::vector<VkDescriptorImageInfo> imageInfos(2 * numTextures);
        std::vector<VkWriteDescriptorSet> writes(numBindings);
        for (uint32_t i = 0; i < numBuffers; ++i)
        {
            const Buffer& buf = mBuffers[buffers[i]];
            bufferInfos[i].buffer = buf.buffer;
            bufferInfos[i].offset = 0;
            bufferInfos[i].range = VK_WHOLE_SIZE;

            writes[i] = VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bufferInfos[i];
        }
        for (uint32_t t = 0; t < numTextures; ++t)
        {
            const Texture& tex = mTextures[textures[t]];
            imageInfos[2 * t].imageView = tex.view;
            imageInfos[2 * t].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfos[2 * t + 1].sampler = tex.sampler;

            VkWriteDescriptorSet& wImg = writes[numBuffers + 2 * t];
            wImg = VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            wImg.dstSet = set;
            wImg.dstBinding = numBuffers + 2 * t;
            wImg.descriptorCount = 1;
            wImg.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            wImg.pImageInfo = &imageInfos[2 * t];

            VkWriteDescriptorSet& wSmp = writes[numBuffers + 2 * t + 1];
            wSmp = VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            wSmp.dstSet = set;
            wSmp.dstBinding = numBuffers + 2 * t + 1;
            wSmp.descriptorCount = 1;
            wSmp.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            wSmp.pImageInfo = &imageInfos[2 * t + 1];
        }
        if (numBindings > 0)
            vkUpdateDescriptorSets(mDevice, numBindings, writes.data(), 0, nullptr);

        vkCmdBindPipeline(mCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
        if (numBindings > 0)
            vkCmdBindDescriptorSets(mCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout, 0, 1, &set, 0, nullptr);
        if (pipe.bindless)
            vkCmdBindDescriptorSets(mCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout, 1, 1,
                                    &mBindlessSet, 0, nullptr);
        if (pushConstantSize > 0)
            vkCmdPushConstants(mCommandBuffer, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantSize, pushConstants);

        vkCmdDispatch(mCommandBuffer, gridX, gridY, gridZ);
    }

    bool supportsIndirectDispatch() const override { return true; }

    bool supportsTimestamps() const override { return mHasTimestamps; }

    uint32_t writeTimestamp() override
    {
        if (!mHasTimestamps || mTimestampCount >= kMaxTimestamps)
            return 0xffffffffu;
        beginRecording();
        const uint32_t idx = mTimestampCount++;
        // Bottom-of-pipe: the timestamp is written once all prior commands
        // have completed, so a begin/end pair brackets the work between them.
        vkCmdWriteTimestamp(mCommandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, mQueryPool, idx);
        return idx;
    }

    double getTimestampNanos(uint32_t begin, uint32_t end) override
    {
        if (!mHasTimestamps || begin == 0xffffffffu || end == 0xffffffffu ||
            begin >= kMaxTimestamps || end >= kMaxTimestamps)
            return -1.0;
        // The lower query index was written earlier in the stream, so its
        // tick is the earlier time; masked modular subtraction (late - early)
        // yields the elapsed ticks and is correct across a counter wrap.  The
        // result is therefore independent of the argument order.
        const uint32_t lo = begin < end ? begin : end;
        const uint32_t hi = begin < end ? end : begin;
        const uint32_t count = hi - lo + 1u;
        std::vector<uint64_t> results(count, 0);
        if (vkGetQueryPoolResults(mDevice, mQueryPool, lo, count, count * sizeof(uint64_t),
                                  results.data(), sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS)
            return -1.0;
        const uint64_t mask = mTimestampValidBits >= 64 ? ~0ull : ((1ull << mTimestampValidBits) - 1ull);
        const uint64_t early = results[0] & mask;
        const uint64_t late = results[count - 1] & mask;
        const uint64_t diff = (late - early) & mask;
        return double(diff) * double(mTimestampPeriod);
    }

    void dispatchIndirect(PxgPipelineHandle pipelineHandle, PxgBufferHandle argBuffer, size_t argOffset,
                          const PxgBufferHandle* buffers, uint32_t numBuffers,
                          const void* pushConstants, uint32_t pushConstantSize) override
    {
        Pipeline& pipe = mPipelines[pipelineHandle];
        assert(numBuffers == pipe.numBuffers);
        assert(pipe.numTextures == 0);
        assert(pushConstantSize == pipe.pushConstantSize);
        // vkCmdDispatchIndirect requires a 4-byte aligned arg offset holding
        // a VkDispatchIndirectCommand (3 x uint32).
        assert((argOffset % 4) == 0);

        beginRecording();

        VkDescriptorSet set = allocateDescriptorSet(pipe.setLayout, numBuffers);
        std::vector<VkDescriptorBufferInfo> bufferInfos(numBuffers);
        std::vector<VkWriteDescriptorSet> writes(numBuffers);
        for (uint32_t i = 0; i < numBuffers; ++i)
        {
            const Buffer& buf = mBuffers[buffers[i]];
            bufferInfos[i].buffer = buf.buffer;
            bufferInfos[i].offset = 0;
            bufferInfos[i].range = VK_WHOLE_SIZE;
            writes[i] = VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bufferInfos[i];
        }
        if (numBuffers > 0)
            vkUpdateDescriptorSets(mDevice, numBuffers, writes.data(), 0, nullptr);

        // The arg buffer was produced by a prior compute dispatch; make its
        // workgroup counts visible to the indirect-command read stage.
        const Buffer& arg = mBuffers[argBuffer];
        bufferBarrier(arg.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT);

        vkCmdBindPipeline(mCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
        if (numBuffers > 0)
            vkCmdBindDescriptorSets(mCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout, 0, 1, &set, 0, nullptr);
        if (pipe.bindless)
            vkCmdBindDescriptorSets(mCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout, 1, 1,
                                    &mBindlessSet, 0, nullptr);
        if (pushConstantSize > 0)
            vkCmdPushConstants(mCommandBuffer, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantSize, pushConstants);

        vkCmdDispatchIndirect(mCommandBuffer, arg.buffer, argOffset);
    }

    void barrier() override
    {
        if (!mRecording)
            return;

        VkMemoryBarrier memBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(mCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);
    }

    uint64_t submit() override
    {
        if (!mRecording)
            return mLastTicket;

        PXG_VK_CHECK(vkEndCommandBuffer(mCommandBuffer));
        Frame& f = mFrames[mFrameIndex];

        VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &mCommandBuffer;

        if (mHasTimeline)
        {
            const uint64_t ticket = ++mTimelineValue;
            // Stream ordering: every batch waits on the previous ticket, so
            // consecutive submissions execute (and become visible) in order
            // exactly like work on one CUDA stream.  Same-queue submissions
            // are NOT execution-ordered by submission order alone, and a
            // wait on value 0 (first batch) is trivially satisfied.
            const uint64_t waitValue = ticket - 1;
            const VkPipelineStageFlags waitStage =
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkTimelineSemaphoreSubmitInfo timelineInfo = { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
            timelineInfo.waitSemaphoreValueCount = 1;
            timelineInfo.pWaitSemaphoreValues = &waitValue;
            timelineInfo.signalSemaphoreValueCount = 1;
            timelineInfo.pSignalSemaphoreValues = &ticket;
            submitInfo.pNext = &timelineInfo;
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = &mTimeline;
            submitInfo.pWaitDstStageMask = &waitStage;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &mTimeline;
            PXG_VK_CHECK(vkQueueSubmit(mQueue, 1, &submitInfo, VK_NULL_HANDLE));
            f.retireValue = ticket;
            mLastTicket = ticket;
        }
        else
        {
            // No timeline semaphores: blocking submit, so every returned
            // ticket is already retired (0 == "complete" by convention).
            PXG_VK_CHECK(vkQueueSubmit(mQueue, 1, &submitInfo, mFence));
            PXG_VK_CHECK(vkWaitForFences(mDevice, 1, &mFence, VK_TRUE, UINT64_MAX));
            PXG_VK_CHECK(vkResetFences(mDevice, 1, &mFence));
            f.retireValue = 0;
        }

        ++mSubmitCount;
        mRecording = false;
        return mLastTicket;
    }

    void wait(uint64_t ticket) override
    {
        if (ticket == 0 || !mHasTimeline)
            return;
        VkSemaphoreWaitInfo waitInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores = &mTimeline;
        waitInfo.pValues = &ticket;
        PXG_VK_CHECK(vkWaitSemaphores(mDevice, &waitInfo, UINT64_MAX));
    }

    bool poll(uint64_t ticket) override
    {
        if (ticket == 0 || !mHasTimeline)
            return true;
        uint64_t value = 0;
        PXG_VK_CHECK(vkGetSemaphoreCounterValue(mDevice, mTimeline, &value));
        return value >= ticket;
    }

    void flush() override
    {
        wait(submit());
    }

private:
    struct Buffer
    {
        VkBuffer buffer;
        VkDeviceMemory memory;
        size_t size;
        uint64_t deviceAddress = 0;
        bool hostVisible = true;
        // Persistently mapped CPU pointer for pinned-host buffers
        // (createHostVisibleBuffer); null for ordinary buffers.  A persistently
        // mapped memory cannot be vkMapMemory'd again, so upload/download go
        // through this pointer directly when set.
        void* mappedPtr = nullptr;
    };

    struct Texture
    {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        // Owned only by the bound path; the bindless path shares
        // mBindlessSampler and leaves this null.
        VkSampler sampler = VK_NULL_HANDLE;
        uint32_t width = 0, height = 0, depth = 0;
        bool bindless = false;
    };

    static const uint32_t kFramesInFlight = 4;
    static const size_t kStagingBytes = 8u * 1024u * 1024u;

    // One in-flight submission: command buffer, its descriptor pools, and its
    // slice of the staging ring, all recycled once retireValue has signaled.
    struct Frame
    {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        std::vector<VkDescriptorPool> pools;
        size_t currentPool = 0;
        uint64_t retireValue = 0;
        VkBuffer staging = VK_NULL_HANDLE;
        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        uint8_t* stagingMapped = nullptr;
        size_t stagingOffset = 0;
    };

    struct Pipeline
    {
        VkShaderModule module;
        VkDescriptorSetLayout setLayout;
        VkPipelineLayout layout;
        VkPipeline pipeline;
        uint32_t numBuffers;
        uint32_t numTextures;
        uint32_t pushConstantSize;
        bool bindless = false;   // declares the device-wide texture set at index 1
    };

    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties) const
    {
        for (uint32_t i = 0; i < mMemoryProperties.memoryTypeCount; ++i)
        {
            if ((typeBits & (1u << i)) &&
                (mMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        }
        assert(false && "no suitable memory type");
        return 0;
    }

    // Acquire the next frame slot and open its command buffer.  Waiting on
    // the slot's retire ticket makes N-buffering safe: descriptor sets and
    // staging bytes of a frame are only recycled once the device is done
    // with that frame's submission.
    void beginRecording()
    {
        if (mRecording)
            return;
        mFrameIndex = uint32_t(mSubmitCount % kFramesInFlight);
        Frame& f = mFrames[mFrameIndex];
        wait(f.retireValue);
        for (VkDescriptorPool pool : f.pools)
            PXG_VK_CHECK(vkResetDescriptorPool(mDevice, pool, 0));
        f.currentPool = 0;
        f.stagingOffset = 0;
        mCommandBuffer = f.cmd;

        VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        PXG_VK_CHECK(vkBeginCommandBuffer(mCommandBuffer, &beginInfo));
        // Timestamps are recorded per batch: reset the whole pool up front so
        // query indices restart at 0 for this recording.
        if (mHasTimestamps)
        {
            vkCmdResetQueryPool(mCommandBuffer, mQueryPool, 0, kMaxTimestamps);
            mTimestampCount = 0;
        }
        mRecording = true;
    }

    void ensureStaging(Frame& f)
    {
        if (f.staging)
            return;
        VkBufferCreateInfo info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        info.size = kStagingBytes;
        info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        PXG_VK_CHECK(vkCreateBuffer(mDevice, &info, nullptr, &f.staging));

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(mDevice, f.staging, &req);
        VkMemoryAllocateInfo alloc = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = findMemoryType(
            req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        PXG_VK_CHECK(vkAllocateMemory(mDevice, &alloc, nullptr, &f.stagingMemory));
        PXG_VK_CHECK(vkBindBufferMemory(mDevice, f.staging, f.stagingMemory, 0));

        void* mapped = nullptr;
        PXG_VK_CHECK(vkMapMemory(mDevice, f.stagingMemory, 0, VK_WHOLE_SIZE, 0, &mapped));
        f.stagingMapped = static_cast<uint8_t*>(mapped);
    }

    void imageBarrier(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                      VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                      VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage)
    {
        VkImageMemoryBarrier b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.srcAccessMask = srcAccess;
        b.dstAccessMask = dstAccess;
        b.oldLayout = oldLayout;
        b.newLayout = newLayout;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdPipelineBarrier(mCommandBuffer, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
    }

    void savePipelineCache()
    {
        if (mPipelineCachePath.empty())
            return;
        size_t sz = 0;
        if (vkGetPipelineCacheData(mDevice, mPipelineCache, &sz, nullptr) != VK_SUCCESS || sz == 0)
            return;
        std::vector<uint8_t> data(sz);
        if (vkGetPipelineCacheData(mDevice, mPipelineCache, &sz, data.data()) != VK_SUCCESS)
            return;
        if (FILE* f = fopen(mPipelineCachePath.c_str(), "wb"))
        {
            const size_t wrote = fwrite(data.data(), 1, sz, f);
            fclose(f);
            // Don't leave a truncated blob behind (disk full / short write):
            // a partial file would be rejected on next load anyway, but
            // removing it keeps the cache path clean.
            if (wrote != sz)
                remove(mPipelineCachePath.c_str());
        }
    }

    void releaseTexture(Texture& tex)
    {
        if (tex.sampler)
            vkDestroySampler(mDevice, tex.sampler, nullptr);
        if (tex.view)
            vkDestroyImageView(mDevice, tex.view, nullptr);
        if (tex.image)
            vkDestroyImage(mDevice, tex.image, nullptr);
        if (tex.memory)
            vkFreeMemory(mDevice, tex.memory, nullptr);
    }

    // Per-buffer memory dependency on the readback/upload edges
    // (RUNTIME_PLAN.md P1.3); compute<->compute ordering stays on the global
    // barrier() to match stream-ordered CUDA semantics.
    void bufferBarrier(VkBuffer buffer, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                       VkPipelineStageFlags dstStage, VkAccessFlags dstAccess)
    {
        VkBufferMemoryBarrier b = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
        b.srcAccessMask = srcAccess;
        b.dstAccessMask = dstAccess;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer = buffer;
        b.offset = 0;
        b.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(mCommandBuffer, srcStage, dstStage, 0, 0, nullptr, 1, &b, 0, nullptr);
    }

    VkDescriptorSet allocateDescriptorSet(VkDescriptorSetLayout layout, uint32_t numBuffers)
    {
        Frame& f = mFrames[mFrameIndex];
        for (;;)
        {
            if (f.currentPool == f.pools.size())
            {
                VkDescriptorPoolSize poolSizes[3] = {};
                poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                poolSizes[0].descriptorCount = 1024;
                poolSizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                poolSizes[1].descriptorCount = 128;
                poolSizes[2].type = VK_DESCRIPTOR_TYPE_SAMPLER;
                poolSizes[2].descriptorCount = 128;

                VkDescriptorPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
                poolInfo.maxSets = 256;
                poolInfo.poolSizeCount = 3;
                poolInfo.pPoolSizes = poolSizes;

                VkDescriptorPool pool;
                PXG_VK_CHECK(vkCreateDescriptorPool(mDevice, &poolInfo, nullptr, &pool));
                f.pools.push_back(pool);
            }

            VkDescriptorSetAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            allocInfo.descriptorPool = f.pools[f.currentPool];
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &layout;

            VkDescriptorSet set = VK_NULL_HANDLE;
            const VkResult res = vkAllocateDescriptorSets(mDevice, &allocInfo, &set);
            if (res == VK_SUCCESS)
                return set;
            if (res == VK_ERROR_OUT_OF_POOL_MEMORY || res == VK_ERROR_FRAGMENTED_POOL)
            {
                ++f.currentPool;
                continue;
            }
            PXG_VK_CHECK(res);
            return VK_NULL_HANDLE;
        }
    }

    VkInstance mInstance = VK_NULL_HANDLE;
    VkPhysicalDevice mPhysicalDevice = VK_NULL_HANDLE;
    VkDevice mDevice = VK_NULL_HANDLE;
    VkQueue mQueue = VK_NULL_HANDLE;
    uint32_t mQueueFamily = uint32_t(-1);
    VkCommandPool mCommandPool = VK_NULL_HANDLE;
    VkCommandBuffer mCommandBuffer = VK_NULL_HANDLE;
    VkFence mFence = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mMemoryProperties = {};

    std::unordered_map<PxgBufferHandle, Buffer> mBuffers;
    std::unordered_map<PxgPipelineHandle, Pipeline> mPipelines;
    std::unordered_map<PxgTextureHandle, Texture> mTextures;
    PxgBufferHandle mNextBufferHandle = 1;
    PxgPipelineHandle mNextPipelineHandle = 1;
    PxgTextureHandle mNextTextureHandle = 1;

    Frame mFrames[kFramesInFlight];
    uint32_t mFrameIndex = 0;
    uint64_t mSubmitCount = 0;
    VkSemaphore mTimeline = VK_NULL_HANDLE;
    bool mHasTimeline = false;
    uint64_t mTimelineValue = 0;
    uint64_t mLastTicket = 0;
    PxgSlangDeviceDesc mDesc;
    VkPipelineCache mPipelineCache = VK_NULL_HANDLE;
    std::string mPipelineCachePath;

    static const uint32_t kMaxTimestamps = 256;
    VkQueryPool mQueryPool = VK_NULL_HANDLE;
    bool mHasTimestamps = false;
    uint32_t mTimestampCount = 0;
    uint32_t mTimestampValidBits = 0;
    float mTimestampPeriod = 0.0f;

    bool mRecording = false;
    bool mHasShaderInt64 = false;
    bool mHasBufferDeviceAddress = false;
    // VK_EXT_shader_atomic_float / shaderBufferFloat32AtomicAdd.  Kernels that
    // transcribe CUDA's atomicAdd(float*) emit OpAtomicFAddEXT and require it.
    bool mHasShaderBufferFloat32AtomicAdd = false;
    SubgroupCaps mSubgroupCaps;

    // ---- the bindless texture set (P9.12a) ----
    // One device-wide descriptor set at index 1: an unbounded array of sampled
    // 3D images at binding 0 and ONE shared sampler at binding 1.  One sampler
    // is enough because the engine builds every SDF texture with the identical
    // descriptor -- clamp on all three axes, linear filter, unnormalized
    // element reads (PxgGeometryManager.cpp:761-769).  A future texture that
    // wants different addressing turns binding 1 into an array too; until then
    // sharing it is a statement about the engine, not a shortcut.
    static const uint32_t kBindlessTextureCapacity = 4096;
    bool mHasBindlessTextures = false;
    VkDescriptorSetLayout mBindlessSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool mBindlessPool = VK_NULL_HANDLE;
    VkDescriptorSet mBindlessSet = VK_NULL_HANDLE;
    VkSampler mBindlessSampler = VK_NULL_HANDLE;

    std::string mDeviceName;
};

PxgSlangComputeDevice* PxgCreateVulkanComputeDevice(const PxgSlangDeviceDesc& desc)
{
    PxgVulkanComputeDevice* device = new PxgVulkanComputeDevice();
    if (!device->init(desc))
    {
        delete device;
        return nullptr;
    }
    return device;
}

} // namespace gpuslang
} // namespace physx
