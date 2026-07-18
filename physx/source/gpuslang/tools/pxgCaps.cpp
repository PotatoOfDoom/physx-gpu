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

// pxg-caps: probes every Vulkan physical device against the PhysX-GPU-on-
// Vulkan runtime profile (RUNTIME_PLAN.md section 1) and prints a
// per-device capability matrix with a PASS/FAIL verdict for the required
// set and the state of every optional feature gate.

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstring>
#include <vector>

static const char* deviceTypeName(VkPhysicalDeviceType t)
{
    switch (t)
    {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu (software)";
    default: return "other";
    }
}

static void printFlag(const char* name, bool present, bool required)
{
    std::printf("  %-46s %s%s\n", name, present ? "YES" : "no ",
                present ? "" : (required ? "   <-- REQUIRED, missing" : "   (optional)"));
}

int main()
{
    VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    appInfo.pApplicationName = "pxg-caps";
    uint32_t instanceVersion = VK_API_VERSION_1_0;
    vkEnumerateInstanceVersion(&instanceVersion);
    appInfo.apiVersion = instanceVersion >= VK_API_VERSION_1_3 ? VK_API_VERSION_1_3 : VK_API_VERSION_1_2;

    VkInstanceCreateInfo instInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    instInfo.pApplicationInfo = &appInfo;

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instInfo, nullptr, &instance) != VK_SUCCESS)
    {
        std::printf("pxg-caps: vkCreateInstance failed (no Vulkan loader/ICD?)\n");
        return 1;
    }
    std::printf("Vulkan instance version: %u.%u.%u\n", VK_API_VERSION_MAJOR(instanceVersion),
                VK_API_VERSION_MINOR(instanceVersion), VK_API_VERSION_PATCH(instanceVersion));

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    std::printf("Physical devices: %u\n", count);

    int worstVerdict = 0;
    for (uint32_t d = 0; d < count; ++d)
    {
        VkPhysicalDevice pd = devices[d];

        // ---- properties chain ----
        VkPhysicalDeviceSubgroupSizeControlProperties sgSizeProps = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES
        };
        VkPhysicalDeviceSubgroupProperties sgProps = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES
        };
        sgProps.pNext = &sgSizeProps;
        VkPhysicalDeviceProperties2 props2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        props2.pNext = &sgProps;
        vkGetPhysicalDeviceProperties2(pd, &props2);
        const VkPhysicalDeviceProperties& props = props2.properties;

        // ---- features chain ----
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloat = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT
        };
        VkPhysicalDeviceSubgroupSizeControlFeatures sgSizeFeat = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES
        };
        sgSizeFeat.pNext = &atomicFloat;
        VkPhysicalDeviceVulkan12Features feat12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
        feat12.pNext = &sgSizeFeat;
        VkPhysicalDeviceFeatures2 feat2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        feat2.pNext = &feat12;
        vkGetPhysicalDeviceFeatures2(pd, &feat2);

        // extension list (for EXT presence independent of the feature bools)
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, exts.data());
        auto hasExt = [&](const char* name) {
            for (const auto& e : exts)
                if (std::strcmp(e.extensionName, name) == 0)
                    return true;
            return false;
        };

        std::printf("\n=== Device %u: %s [%s] api %u.%u.%u driver 0x%x ===\n", d, props.deviceName,
                    deviceTypeName(props.deviceType), VK_API_VERSION_MAJOR(props.apiVersion),
                    VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion),
                    props.driverVersion);

        // ---- subgroup facts ----
        const VkSubgroupFeatureFlags neededOps = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT |
                                                 VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
                                                 VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
                                                 VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
        const bool subgroupOps = (sgProps.supportedOperations & neededOps) == neededOps &&
                                 (sgProps.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
        const bool sizeControl = sgSizeFeat.subgroupSizeControl == VK_TRUE;
        const bool fullSubgroups = sgSizeFeat.computeFullSubgroups == VK_TRUE;
        const bool canRequire32 = sizeControl && sgSizeProps.minSubgroupSize <= 32 &&
                                  sgSizeProps.maxSubgroupSize >= 32 &&
                                  (sgSizeProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;

        std::printf("  subgroup: default=%u  min=%u  max=%u  requiredSizeStages(compute)=%s\n",
                    sgProps.subgroupSize, sgSizeProps.minSubgroupSize, sgSizeProps.maxSubgroupSize,
                    (sgSizeProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) ? "yes" : "no");
        std::printf("  maxComputeSharedMemorySize: %u KiB   maxComputeWorkGroupInvocations: %u\n",
                    props.limits.maxComputeSharedMemorySize / 1024, props.limits.maxComputeWorkGroupInvocations);

        // ---- required set ----
        std::printf("  --- required profile ---\n");
        const bool api12 = props.apiVersion >= VK_API_VERSION_1_2;
        printFlag("Vulkan >= 1.2", api12, true);
        printFlag("subgroup ops (basic/vote/arith/ballot/shuffle)", subgroupOps, true);
        printFlag("subgroupSizeControl", sizeControl, true);
        printFlag("computeFullSubgroups", fullSubgroups, true);
        printFlag("required subgroup size 32 (compute)", canRequire32, true);
        printFlag("shaderInt64", feat2.features.shaderInt64 == VK_TRUE, true);
        printFlag("timelineSemaphore", feat12.timelineSemaphore == VK_TRUE, true);
        printFlag("bufferDeviceAddress", feat12.bufferDeviceAddress == VK_TRUE, true);

        const bool requiredOk = api12 && subgroupOps && sizeControl && fullSubgroups && canRequire32 &&
                                feat2.features.shaderInt64 && feat12.timelineSemaphore &&
                                feat12.bufferDeviceAddress;

        // ---- optional gates ----
        std::printf("  --- optional gates ---\n");
        printFlag("VK_EXT_shader_atomic_float (ext present)", hasExt(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME), false);
        printFlag("  shaderBufferFloat32AtomicAdd", atomicFloat.shaderBufferFloat32AtomicAdd == VK_TRUE, false);
        printFlag("  shaderSharedFloat32AtomicAdd", atomicFloat.shaderSharedFloat32AtomicAdd == VK_TRUE, false);
        printFlag("shaderFloat64", feat2.features.shaderFloat64 == VK_TRUE, false);
        printFlag("shaderBufferInt64Atomics", feat12.shaderBufferInt64Atomics == VK_TRUE, false);
        printFlag("vulkanMemoryModel", feat12.vulkanMemoryModel == VK_TRUE, false);

        std::printf("  VERDICT: %s\n", requiredOk ? "PROFILE-OK" : "PROFILE-FAIL");
        if (!requiredOk && worstVerdict == 0)
            worstVerdict = 2;
    }

    vkDestroyInstance(instance, nullptr);
    return worstVerdict;
}
