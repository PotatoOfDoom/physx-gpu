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
// Simplicity-first implementation notes:
//   - Buffers are HOST_VISIBLE|HOST_COHERENT with DEVICE_LOCAL preferred when
//     available in a combined heap.  A production integration would use
//     device-local memory plus staging rings; for kernel verification and
//     software devices (llvmpipe) host-visible memory is both simpler and
//     faster.
//   - One descriptor set per dispatch, allocated from a growing pool that is
//     reset on flush().
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
            for (VkDescriptorPool pool : mDescriptorPools)
                vkDestroyDescriptorPool(mDevice, pool, nullptr);
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

        // Enable 64-bit integer support when the device offers it (needed by
        // kernels that carry PxU64 accumulators, e.g. the broad-phase
        // overlap-check region scan).  Guarded so devices without shaderInt64
        // still create successfully.
        VkPhysicalDeviceFeatures supported = {};
        vkGetPhysicalDeviceFeatures(mPhysicalDevice, &supported);
        VkPhysicalDeviceFeatures enabled = {};
        enabled.shaderInt64 = supported.shaderInt64;
        mHasShaderInt64 = supported.shaderInt64 == VK_TRUE;
        // Enable 64-bit float support when offered: kernels transcribing CUDA
        // code that mixes double literals into float expressions (e.g. the
        // sb_extractRotationAPD Rodrigues update '(1.0 - l2) / (1.0f + l2)')
        // must perform those spots in f64 to match the C++ reference rounding.
        enabled.shaderFloat64 = supported.shaderFloat64;

        VkDeviceCreateInfo deviceInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        deviceInfo.pEnabledFeatures = &enabled;

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
        cmdInfo.commandBufferCount = 1;
        PXG_VK_CHECK(vkAllocateCommandBuffers(mDevice, &cmdInfo, &mCommandBuffer));

        VkFenceCreateInfo fenceInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        PXG_VK_CHECK(vkCreateFence(mDevice, &fenceInfo, nullptr, &mFence));

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
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        Buffer buf = {};
        buf.size = sizeBytes;
        PXG_VK_CHECK(vkCreateBuffer(mDevice, &bufferInfo, nullptr, &buf.buffer));

        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(mDevice, buf.buffer, &memReq);

        const uint32_t memoryType = findMemoryType(
            memReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = memoryType;
        PXG_VK_CHECK(vkAllocateMemory(mDevice, &allocInfo, nullptr, &buf.memory));
        PXG_VK_CHECK(vkBindBufferMemory(mDevice, buf.buffer, buf.memory, 0));

        const PxgBufferHandle handle = mNextBufferHandle++;
        mBuffers[handle] = buf;
        return handle;
    }

    void destroyBuffer(PxgBufferHandle handle) override
    {
        auto it = mBuffers.find(handle);
        if (it == mBuffers.end())
            return;
        vkDeviceWaitIdle(mDevice);
        vkDestroyBuffer(mDevice, it->second.buffer, nullptr);
        vkFreeMemory(mDevice, it->second.memory, nullptr);
        mBuffers.erase(it);
    }

    void upload(PxgBufferHandle handle, const void* data, size_t sizeBytes, size_t dstOffset) override
    {
        Buffer& buf = mBuffers[handle];
        void* mapped = nullptr;
        PXG_VK_CHECK(vkMapMemory(mDevice, buf.memory, dstOffset, sizeBytes, 0, &mapped));
        memcpy(mapped, data, sizeBytes);
        vkUnmapMemory(mDevice, buf.memory);
    }

    void download(PxgBufferHandle handle, void* data, size_t sizeBytes, size_t srcOffset) override
    {
        Buffer& buf = mBuffers[handle];
        void* mapped = nullptr;
        PXG_VK_CHECK(vkMapMemory(mDevice, buf.memory, srcOffset, sizeBytes, 0, &mapped));
        memcpy(data, mapped, sizeBytes);
        vkUnmapMemory(mDevice, buf.memory);
    }

    // ---- pipelines ----

    PxgPipelineHandle createPipeline(const void* code, size_t codeSize, const char* entryName,
                                     uint32_t numBuffers, const uint32_t* bufferStrides,
                                     uint32_t pushConstantSize) override
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

    void flush() override
    {
        if (!mRecording)
            return;

        PXG_VK_CHECK(vkEndCommandBuffer(mCommandBuffer));

        VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &mCommandBuffer;
        PXG_VK_CHECK(vkQueueSubmit(mQueue, 1, &submitInfo, mFence));
        PXG_VK_CHECK(vkWaitForFences(mDevice, 1, &mFence, VK_TRUE, UINT64_MAX));
        PXG_VK_CHECK(vkResetFences(mDevice, 1, &mFence));

        // Recycle descriptor pools for the next batch.
        for (VkDescriptorPool pool : mDescriptorPools)
            PXG_VK_CHECK(vkResetDescriptorPool(mDevice, pool, 0));
        mCurrentPool = 0;

        mRecording = false;
    }

private:
    struct Buffer
    {
        VkBuffer buffer;
        VkDeviceMemory memory;
        size_t size;
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

    void beginRecording()
    {
        if (mRecording)
            return;
        VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        PXG_VK_CHECK(vkBeginCommandBuffer(mCommandBuffer, &beginInfo));
        mRecording = true;
    }

    VkDescriptorSet allocateDescriptorSet(VkDescriptorSetLayout layout, uint32_t numBuffers)
    {
        for (;;)
        {
            if (mCurrentPool == mDescriptorPools.size())
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
                mDescriptorPools.push_back(pool);
            }

            VkDescriptorSetAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            allocInfo.descriptorPool = mDescriptorPools[mCurrentPool];
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &layout;

            VkDescriptorSet set = VK_NULL_HANDLE;
            const VkResult res = vkAllocateDescriptorSets(mDevice, &allocInfo, &set);
            if (res == VK_SUCCESS)
                return set;
            if (res == VK_ERROR_OUT_OF_POOL_MEMORY || res == VK_ERROR_FRAGMENTED_POOL)
            {
                ++mCurrentPool;
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

    std::vector<VkDescriptorPool> mDescriptorPools;
    size_t mCurrentPool = 0;
    bool mRecording = false;
    bool mHasShaderInt64 = false;

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
