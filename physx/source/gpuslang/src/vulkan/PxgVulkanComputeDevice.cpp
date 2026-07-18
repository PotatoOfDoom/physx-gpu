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
        VkPhysicalDeviceSubgroupSizeControlFeatures sgSizeSupported = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES
        };
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

        // Subgroup size control: the warp-32 primitive library requires an
        // exact subgroup size on hardware targets (llvmpipe emulates instead).
        VkPhysicalDeviceSubgroupSizeControlFeatures enabledSgSize = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES
        };
        enabledSgSize.subgroupSizeControl = sgSizeSupported.subgroupSizeControl;
        enabledSgSize.computeFullSubgroups = sgSizeSupported.computeFullSubgroups;

        enabled2.pNext = &enabled12;
        enabled12.pNext = &enabledSgSize;

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

        VkDeviceCreateInfo deviceInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        deviceInfo.pNext = &enabled2;

        if (vkCreateDevice(mPhysicalDevice, &deviceInfo, nullptr, &mDevice) != VK_SUCCESS)
            return false;

        vkGetDeviceQueue(mDevice, mQueueFamily, 0, &mQueue);

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
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
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

    void destroyBuffer(PxgBufferHandle handle) override
    {
        auto it = mBuffers.find(handle);
        if (it == mBuffers.end())
            return;
        // An open recording may reference this buffer (staged uploads record
        // copies); submit it so the wait below actually covers that work.
        flush();
        vkDeviceWaitIdle(mDevice);
        vkDestroyBuffer(mDevice, it->second.buffer, nullptr);
        vkFreeMemory(mDevice, it->second.memory, nullptr);
        mBuffers.erase(it);
    }

    void upload(PxgBufferHandle handle, const void* data, size_t sizeBytes, size_t dstOffset) override
    {
        if (sizeBytes == 0)
            return;
        Buffer& buf = mBuffers[handle];
        if (buf.hostVisible)
        {
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
        if (buf.hostVisible)
        {
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

    // ---- pipelines ----

    PxgPipelineHandle createPipeline(const void* code, size_t codeSize, const char* entryName,
                                     uint32_t numBuffers, const uint32_t* bufferStrides,
                                     uint32_t pushConstantSize) override
    {
        return createPipelineInternal(code, codeSize, entryName, numBuffers, bufferStrides, pushConstantSize, 0);
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
                                      requiredSubgroupSize);
    }

    SubgroupCaps getSubgroupCaps() const override { return mSubgroupCaps; }

    uint64_t getBufferDeviceAddress(PxgBufferHandle handle) const override
    {
        auto it = mBuffers.find(handle);
        return it == mBuffers.end() ? 0 : it->second.deviceAddress;
    }

    PxgPipelineHandle createPipelineInternal(const void* code, size_t codeSize, const char* entryName,
                                             uint32_t numBuffers, const uint32_t* bufferStrides,
                                             uint32_t pushConstantSize, uint32_t requiredSubgroupSize)
    {
        (void)bufferStrides; // SPIR-V carries the layout; strides are for D3D12

        Pipeline pipe = {};
        pipe.numBuffers = numBuffers;
        pipe.pushConstantSize = pushConstantSize;

        VkShaderModuleCreateInfo moduleInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        moduleInfo.codeSize = codeSize;
        moduleInfo.pCode = reinterpret_cast<const uint32_t*>(code);
        PXG_VK_CHECK(vkCreateShaderModule(mDevice, &moduleInfo, nullptr, &pipe.module));

        std::vector<VkDescriptorSetLayoutBinding> bindings(numBuffers);
        for (uint32_t i = 0; i < numBuffers; ++i)
        {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo setLayoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        setLayoutInfo.bindingCount = numBuffers;
        setLayoutInfo.pBindings = bindings.empty() ? nullptr : bindings.data();
        PXG_VK_CHECK(vkCreateDescriptorSetLayout(mDevice, &setLayoutInfo, nullptr, &pipe.setLayout));

        VkPushConstantRange pushRange = {};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = pushConstantSize;

        VkPipelineLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &pipe.setLayout;
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
        PXG_VK_CHECK(vkCreateComputePipelines(mDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipe.pipeline));

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
        Pipeline& pipe = mPipelines[pipelineHandle];
        assert(numBuffers == pipe.numBuffers);
        assert(pushConstantSize == pipe.pushConstantSize);

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

        vkCmdBindPipeline(mCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
        if (numBuffers > 0)
            vkCmdBindDescriptorSets(mCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout, 0, 1, &set, 0, nullptr);
        if (pushConstantSize > 0)
            vkCmdPushConstants(mCommandBuffer, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantSize, pushConstants);

        vkCmdDispatch(mCommandBuffer, gridX, gridY, gridZ);
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
        uint32_t pushConstantSize;
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
                VkDescriptorPoolSize poolSize = {};
                poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                poolSize.descriptorCount = 1024;

                VkDescriptorPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
                poolInfo.maxSets = 256;
                poolInfo.poolSizeCount = 1;
                poolInfo.pPoolSizes = &poolSize;

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
    PxgBufferHandle mNextBufferHandle = 1;
    PxgPipelineHandle mNextPipelineHandle = 1;

    Frame mFrames[kFramesInFlight];
    uint32_t mFrameIndex = 0;
    uint64_t mSubmitCount = 0;
    VkSemaphore mTimeline = VK_NULL_HANDLE;
    bool mHasTimeline = false;
    uint64_t mTimelineValue = 0;
    uint64_t mLastTicket = 0;
    PxgSlangDeviceDesc mDesc;

    bool mRecording = false;
    bool mHasShaderInt64 = false;
    bool mHasBufferDeviceAddress = false;
    SubgroupCaps mSubgroupCaps;

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
