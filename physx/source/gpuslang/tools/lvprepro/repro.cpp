// Standalone Vulkan reproducer — no PhysX, no external deps beyond the Vulkan loader.
//
//   slangc -O0 -fvk-use-entrypoint-name -target spirv -entry reproMain repro.slang -o repro.spv
//   g++ -O0 -g repro.cpp -lvulkan -o repro && ./repro [delta]
//
// Exits 0 and prints the result vector when the dispatch survives.  On Mesa
// llvmpipe 25.2.8 (LLVM 20.1.2) it instead dies with SIGSEGV inside a driver worker
// thread as soon as `delta` is non-zero; delta == 0 runs clean.
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

#define VK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    std::fprintf(stderr, "%s failed: %d (line %d)\n", #x, int(r_), __LINE__); std::exit(2); } } while (0)

static std::vector<char> readFile(const char* path)
{
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> buf((size_t(n)));
    if (std::fread(buf.data(), 1, size_t(n), f) != size_t(n)) { std::exit(2); }
    std::fclose(f);
    return buf;
}

struct Ctx
{
    VkInstance instance{};
    VkPhysicalDevice phys{};
    VkDevice device{};
    VkQueue queue{};
    uint32_t queueFamily = 0;
    VkCommandPool pool{};
};

struct Buf
{
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    VkDeviceAddress address = 0;
    void* mapped = nullptr;
};

static uint32_t findMemType(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags want)
{
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    std::fprintf(stderr, "no suitable memory type\n"); std::exit(2);
}

static Buf makeBuffer(const Ctx& c, VkDeviceSize size)
{
    Buf b;
    VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK(vkCreateBuffer(c.device, &bi, nullptr, &b.buffer));

    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(c.device, b.buffer, &mr);

    VkMemoryAllocateFlagsInfo fi{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
    fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.pNext = &fi;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = findMemType(c.phys, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK(vkAllocateMemory(c.device, &ai, nullptr, &b.memory));
    VK(vkBindBufferMemory(c.device, b.buffer, b.memory, 0));
    VK(vkMapMemory(c.device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped));

    VkBufferDeviceAddressInfo dai{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
    dai.buffer = b.buffer;
    b.address = vkGetBufferDeviceAddress(c.device, &dai);
    return b;
}

struct Push
{
    uint64_t posA, posB, pullIdx, writeIdx;
    uint32_t count;
    float delta;
};

int main(int argc, char** argv)
{
    const float delta = (argc > 1) ? float(std::atof(argv[1])) : 1.0f;
    std::printf("repro: delta = %g\n", delta);

    Ctx c;
    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "lvprepro";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ii{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ii.pApplicationInfo = &app;
    VK(vkCreateInstance(&ii, nullptr, &c.instance));

    uint32_t n = 0;
    VK(vkEnumeratePhysicalDevices(c.instance, &n, nullptr));
    std::vector<VkPhysicalDevice> devs(n);
    VK(vkEnumeratePhysicalDevices(c.instance, &n, devs.data()));
    if (!n) { std::fprintf(stderr, "no vulkan device\n"); return 2; }
    c.phys = devs[0];
    VkPhysicalDeviceProperties pp{};
    vkGetPhysicalDeviceProperties(c.phys, &pp);
    std::printf("device: %s (api %u.%u.%u)\n", pp.deviceName,
                VK_VERSION_MAJOR(pp.apiVersion), VK_VERSION_MINOR(pp.apiVersion), VK_VERSION_PATCH(pp.apiVersion));

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &qn, qs.data());
    for (uint32_t i = 0; i < qn; ++i) if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c.queueFamily = i; break; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qi.queueFamilyIndex = c.queueFamily; qi.queueCount = 1; qi.pQueuePriorities = &prio;

    VkPhysicalDeviceVulkan12Features f12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    f12.bufferDeviceAddress = VK_TRUE;
    VkDeviceCreateInfo di{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    di.pNext = &f12;
    di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    VK(vkCreateDevice(c.phys, &di, nullptr, &c.device));
    vkGetDeviceQueue(c.device, c.queueFamily, 0, &c.queue);

    // --- data ---------------------------------------------------------------
    const uint32_t count = 3;              // "tets"
    const uint32_t nA = 8, nB = count * 4; // source / destination element counts
    Buf posA = makeBuffer(c, nA * 16);
    Buf posB = makeBuffer(c, nB * 16);
    Buf pull = makeBuffer(c, count * 16);
    Buf writ = makeBuffer(c, count * 16);

    float* a = static_cast<float*>(posA.mapped);
    for (uint32_t i = 0; i < nA; ++i)
    { a[i*4+0] = 0.1f*float(i); a[i*4+1] = 0.2f*float(i); a[i*4+2] = 0.3f*float(i); a[i*4+3] = 1.0f + 0.25f*float(i); }
    float* b = static_cast<float*>(posB.mapped);
    for (uint32_t i = 0; i < nB; ++i)
    { b[i*4+0] = 1.0f+0.1f*float(i); b[i*4+1] = 2.0f+0.1f*float(i); b[i*4+2] = 3.0f+0.1f*float(i); b[i*4+3] = 0.5f; }

    const uint32_t CP = 0xFFFFFFFFu;
    uint32_t* pu = static_cast<uint32_t*>(pull.mapped);
    const uint32_t pullData[3][4] = { { 1u, CP, 3u, CP }, { CP, CP, CP, CP }, { 0u, 2u, 4u, 5u } };
    std::memcpy(pu, pullData, sizeof(pullData));
    uint32_t* wr = static_cast<uint32_t*>(writ.mapped);
    for (uint32_t t = 0; t < count; ++t) for (uint32_t k = 0; k < 4; ++k) wr[t*4+k] = t*4+k;

    // --- pipeline -----------------------------------------------------------
    std::vector<char> spv = readFile("repro.spv");
    VkShaderModuleCreateInfo smi{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smi.codeSize = spv.size();
    smi.pCode = reinterpret_cast<const uint32_t*>(spv.data());
    VkShaderModule sm{};
    VK(vkCreateShaderModule(c.device, &smi, nullptr, &sm));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0; pcr.size = sizeof(Push);
    VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout layout{};
    VK(vkCreatePipelineLayout(c.device, &pli, nullptr, &layout));

    VkComputePipelineCreateInfo cpi{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = sm;
    cpi.stage.pName = "reproMain";
    cpi.layout = layout;
    VkPipeline pipe{};
    VK(vkCreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));

    VkCommandPoolCreateInfo cpci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpci.queueFamilyIndex = c.queueFamily;
    VK(vkCreateCommandPool(c.device, &cpci, nullptr, &c.pool));
    VkCommandBufferAllocateInfo cbi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbi.commandPool = c.pool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
    VkCommandBuffer cb{};
    VK(vkAllocateCommandBuffers(c.device, &cbi, &cb));

    Push push{};
    push.posA = posA.address; push.posB = posB.address;
    push.pullIdx = pull.address; push.writeIdx = writ.address;
    push.count = count; push.delta = delta;

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK(vkBeginCommandBuffer(cb, &bi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
    vkCmdDispatch(cb, 1, 1, 1);
    VK(vkEndCommandBuffer(cb));

    VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence{};
    VK(vkCreateFence(c.device, &fci, nullptr, &fence));
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    std::printf("submitting...\n"); std::fflush(stdout);
    VK(vkQueueSubmit(c.queue, 1, &si, fence));
    VK(vkWaitForFences(c.device, 1, &fence, VK_TRUE, ~0ull));
    std::printf("dispatch survived\n");

    for (uint32_t i = 0; i < nB; ++i)
        std::printf("  posB[%2u] = (%8.4f, %8.4f, %8.4f, %6.3f)\n", i, b[i*4+0], b[i*4+1], b[i*4+2], b[i*4+3]);
    return 0;
}
