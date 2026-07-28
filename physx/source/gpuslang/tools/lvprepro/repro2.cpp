// Standalone driver for the REAL ported kernel (softBodySolveBda.spv,
// sb_gm_cp_solveTetrahedronsPartitionLaunch).  Builds a synthetic PxgSoftBody descriptor
// by byte offset — no PhysX headers — and dispatches through plain Vulkan.
//
//   g++ -O0 -g repro2.cpp -lvulkan -o repro2 && ./repro2 <path-to-softBodySolveBda.spv>
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define VK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    std::fprintf(stderr, "%s failed: %d (line %d)\n", #x, int(r_), __LINE__); std::exit(2); } } while (0)

static std::vector<char> readFile(const char* path)
{
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> buf((size_t(n)));
    if (std::fread(buf.data(), 1, size_t(n), f) != size_t(n)) std::exit(2);
    std::fclose(f);
    return buf;
}

struct Ctx { VkInstance inst{}; VkPhysicalDevice phys{}; VkDevice dev{}; VkQueue q{}; uint32_t qf = 0; };
struct Buf { VkBuffer buffer{}; VkDeviceMemory mem{}; VkDeviceAddress addr = 0; void* mapped = nullptr; };

static uint32_t memType(VkPhysicalDevice p, uint32_t bits, VkMemoryPropertyFlags want)
{
    VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(p, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u<<i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    std::exit(2);
}

static Buf mkBuf(const Ctx& c, VkDeviceSize size)
{
    Buf b;
    VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size ? size : 16;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VK(vkCreateBuffer(c.dev, &bi, nullptr, &b.buffer));
    VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(c.dev, b.buffer, &mr);
    VkMemoryAllocateFlagsInfo fi{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
    fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.pNext = &fi; ai.allocationSize = mr.size;
    ai.memoryTypeIndex = memType(c.phys, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK(vkAllocateMemory(c.dev, &ai, nullptr, &b.mem));
    VK(vkBindBufferMemory(c.dev, b.buffer, b.mem, 0));
    VK(vkMapMemory(c.dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.mapped));
    std::memset(b.mapped, 0, size_t(bi.size));
    VkBufferDeviceAddressInfo dai{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
    dai.buffer = b.buffer; b.addr = vkGetBufferDeviceAddress(c.dev, &dai);
    return b;
}

// PxgSoftBody field offsets (from pxgSlangSoftBodyDesc.slang)
enum : size_t {
    O_mSimVelocity_InvMass = 88, O_mSimPosition_InvMass = 96,
    O_mSimTetraRestPoses = 136, O_mSimTetraRotations = 168,
    O_mSimVelocity_invMassCP = 176, O_mSimPosition_InvMassCP = 184,
    O_mSimRemapOutputCP = 192, O_mSimAccumulatedPartitionsCP = 208,
    O_mSimTetraMultipliers = 264, O_mOrderedMaterialIndices = 288,
    O_mInitialRotation = 348, O_mNumTetsGM = 376, O_mNumPartitionsGM = 380,
    O_mSimPullIndices = 400, O_mNumTetsPerElement = 412, O_mFlagsWord = 392,
    SIZEOF_PxgSoftBody = 432
};

struct Push
{
    uint64_t gSoftbodies, activeSoftbodies;
    uint32_t numActive; float invDt; uint32_t partitionId, flagsWord;
    uint64_t materials;
};

int main(int argc, char** argv)
{
    const char* spvPath = (argc > 1) ? argv[1] : "softBodySolveBda.spv";

    Ctx c;
    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ii{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO }; ii.pApplicationInfo = &app;
    VK(vkCreateInstance(&ii, nullptr, &c.inst));
    uint32_t n = 0; VK(vkEnumeratePhysicalDevices(c.inst, &n, nullptr));
    std::vector<VkPhysicalDevice> d(n); VK(vkEnumeratePhysicalDevices(c.inst, &n, d.data()));
    c.phys = d[0];
    VkPhysicalDeviceProperties pp{}; vkGetPhysicalDeviceProperties(c.phys, &pp);
    std::printf("device: %s\n", pp.deviceName);
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qs(qn); vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &qn, qs.data());
    for (uint32_t i = 0; i < qn; ++i) if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c.qf = i; break; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qi.queueFamilyIndex = c.qf; qi.queueCount = 1; qi.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan12Features f12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    f12.bufferDeviceAddress = VK_TRUE;
    VkDeviceCreateInfo di{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    di.pNext = &f12; di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    VK(vkCreateDevice(c.phys, &di, nullptr, &c.dev));
    vkGetDeviceQueue(c.dev, c.qf, 0, &c.q);

    const uint32_t nbTets = 3;
    Buf cpPos = mkBuf(c, nbTets*4*16 + 1024), cpVel = mkBuf(c, nbTets*4*16 + 1024);
    Buf unPos = mkBuf(c, 64*16), unVel = mkBuf(c, 64*16);
    Buf pull = mkBuf(c, 64*16), remap = mkBuf(c, 64*16);
    Buf rest = mkBuf(c, 1152 + 4096), rot = mkBuf(c, 64*16);
    Buf mult = mkBuf(c, 768 + 4096), mat = mkBuf(c, 256);
    Buf matIdx = mkBuf(c, 256), parts = mkBuf(c, 256);
    Buf desc = mkBuf(c, SIZEOF_PxgSoftBody + 256), active = mkBuf(c, 256);

    float* p = static_cast<float*>(cpPos.mapped);
    for (uint32_t i = 0; i < nbTets*4; ++i)
    { p[i*4+0]=0.11f*float(i)+0.03f; p[i*4+1]=1.0f-0.07f*float(i); p[i*4+2]=0.19f*float(i); p[i*4+3]=1.0f+0.25f*float(i); }
    float* up = static_cast<float*>(unPos.mapped);
    for (uint32_t i = 0; i < 64; ++i)
    { up[i*4+0]=-0.31f*float(i); up[i*4+1]=0.9f+0.05f*float(i); up[i*4+2]=0.23f*float(i); up[i*4+3]=2.0f; }
    const uint32_t CP = 0xFFFFFFFFu;
    uint32_t* pu = static_cast<uint32_t*>(pull.mapped);
    const uint32_t pd[3][4] = { {1u,CP,3u,CP}, {CP,CP,CP,CP}, {0u,2u,4u,5u} };
    std::memcpy(pu, pd, sizeof(pd));
    uint32_t* rm = static_cast<uint32_t*>(remap.mapped);
    for (uint32_t t = 0; t < nbTets; ++t) for (uint32_t k = 0; k < 4; ++k) rm[t*4+k] = t*4+k;
    // rest poses: PxgMat33Block, lane t
    float* rp = static_cast<float*>(rest.mapped);
    for (uint32_t t = 0; t < nbTets; ++t)
    {
        rp[t*4+0]=1.13f; rp[t*4+1]=0.21f; rp[t*4+2]=-0.09f; rp[t*4+3]=-0.19f;
        rp[128+t*4+0]=0.97f; rp[128+t*4+1]=0.13f; rp[128+t*4+2]=0.07f; rp[128+t*4+3]=-0.27f;
        rp[256+t]=1.07f;
    }
    float* rq = static_cast<float*>(rot.mapped);
    for (uint32_t i = 0; i < 64; ++i) { rq[i*4+3] = 1.0f; }
    float* mp = static_cast<float*>(mat.mapped);
    mp[0]=1.0e5f; mp[1]=0.31f; mp[3]=0.0f;
    static_cast<uint32_t*>(mat.mapped)[4] = 32768u;   // dampingScale, materialModel = 0
    static_cast<uint32_t*>(parts.mapped)[0] = nbTets;

    uint8_t* dp = static_cast<uint8_t*>(desc.mapped);
    auto setPtr = [&](size_t off, uint64_t v) { std::memcpy(dp + off, &v, 8); };
    auto setU32 = [&](size_t off, uint32_t v) { std::memcpy(dp + off, &v, 4); };
    setPtr(O_mSimVelocity_InvMass, unVel.addr);
    setPtr(O_mSimPosition_InvMass, unPos.addr);
    setPtr(O_mSimTetraRestPoses, rest.addr);
    setPtr(O_mSimTetraRotations, rot.addr);
    setPtr(O_mSimVelocity_invMassCP, cpVel.addr);
    setPtr(O_mSimPosition_InvMassCP, cpPos.addr);
    setPtr(O_mSimRemapOutputCP, remap.addr);
    setPtr(O_mSimAccumulatedPartitionsCP, parts.addr);
    setPtr(O_mSimTetraMultipliers, mult.addr);
    setPtr(O_mOrderedMaterialIndices, matIdx.addr);
    setPtr(O_mSimPullIndices, pull.addr);
    float one = 1.0f; std::memcpy(dp + O_mInitialRotation + 12, &one, 4);   // quat w = 1
    setU32(O_mNumTetsGM, nbTets);
    setU32(O_mNumPartitionsGM, 1u);
    setU32(O_mNumTetsPerElement, 1u);
    setU32(O_mFlagsWord, 0u);

    std::vector<char> spv = readFile(spvPath);
    VkShaderModuleCreateInfo smi{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smi.codeSize = spv.size(); smi.pCode = reinterpret_cast<const uint32_t*>(spv.data());
    VkShaderModule sm{}; VK(vkCreateShaderModule(c.dev, &smi, nullptr, &sm));
    VkPushConstantRange pcr{}; pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pcr.size = sizeof(Push);
    VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout layout{}; VK(vkCreatePipelineLayout(c.dev, &pli, nullptr, &layout));
    VkComputePipelineCreateInfo cpi{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = sm;
    cpi.stage.pName = "sb_gm_cp_solveTetrahedronsPartitionLaunch"; cpi.layout = layout;
    VkPipeline pipe{}; VK(vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));

    VkCommandPoolCreateInfo cpci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO }; cpci.queueFamilyIndex = c.qf;
    VkCommandPool pool{}; VK(vkCreateCommandPool(c.dev, &cpci, nullptr, &pool));
    VkCommandBufferAllocateInfo cbi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbi.commandPool = pool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
    VkCommandBuffer cb{}; VK(vkAllocateCommandBuffers(c.dev, &cbi, &cb));

    Push push{};
    push.gSoftbodies = desc.addr; push.activeSoftbodies = active.addr;
    push.numActive = 1u; push.invDt = 60.0f; push.partitionId = 0u; push.flagsWord = 0u;
    push.materials = mat.addr;

    VkCommandBufferBeginInfo bbi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK(vkBeginCommandBuffer(cb, &bbi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
    vkCmdDispatch(cb, 1, 1, 1);
    VK(vkEndCommandBuffer(cb));
    VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence{}; VK(vkCreateFence(c.dev, &fci, nullptr, &fence));
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO }; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    std::printf("submitting the REAL kernel...\n"); std::fflush(stdout);
    VK(vkQueueSubmit(c.q, 1, &si, fence));
    VK(vkWaitForFences(c.dev, 1, &fence, VK_TRUE, ~0ull));
    std::printf("dispatch survived\n");
    for (uint32_t i = 0; i < nbTets*4; ++i)
        std::printf("  cpPos[%2u] = (%9.5f, %9.5f, %9.5f, %6.3f)\n", i, p[i*4+0], p[i*4+1], p[i*4+2], p[i*4+3]);
    return 0;
}
