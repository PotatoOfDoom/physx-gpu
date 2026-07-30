// Standalone driver for softBodyContactRemapBda.spv's
// sb_other_contact_remap_to_simLaunch, which SIGSEGVs on llvmpipe inside the gpuslang test
// harness.  No PhysX headers: the PxgSoftBody descriptor is built by raw byte offset and
// the kernel is dispatched through plain Vulkan, exactly as repro2 did for the tranche-6
// FEM solve.
//
// The point of this file is to answer ONE question first: does the fault survive outside
// the harness?  If it does, the shader is at fault and can be bisected in a fast loop.  If
// it does not, the harness or the PhysX-side buffer setup is, and the shader is exonerated.
//
// Every buffer is deliberately OVER-ALLOCATED by kilobytes, so a tight-fit overrun (the
// llvmpipe load-widening failure mode this project has hit repeatedly) is excluded by
// construction rather than by argument.
//
//   g++ -O0 -g repro3.cpp -lvulkan -o repro3 && ./repro3 <path-to-softBodyContactRemapBda.spv>
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

// PxgSoftBody field offsets (pxgSlangSoftBodyDesc.slang) — only the four the remap reads.
enum : size_t {
    O_mSimTetIndices = 80,
    O_mSimPosition_InvMass = 96,
    O_mTetsRemapColToSim = 232,
    O_mTetsAccumulatedRemapColToSim = 240,
    SIZEOF_PxgSoftBody = 432
};

// sb_other_contact_remap_to_simLaunch push blob (56 bytes, ABI row {8,4,8,8,8,8,8}).
struct Push
{
    uint64_t softbodies;
    uint32_t maxNumContacts, pad12;
    uint64_t point, outBarycentric, outContactInfo, totalContactCount, prevContactCount;
};
static_assert(sizeof(Push) == 56, "remap push blob must be 56 bytes");

int main(int argc, char** argv)
{
    const char* spvPath = (argc > 1) ? argv[1] : "softBodyContactRemapBda.spv";
    // argv[2], if given, overrides the number of candidate sim tets behind the contact.
    const uint32_t nbCandidates = (argc > 2) ? uint32_t(std::atoi(argv[2])) : 3u;
    // argv[3]/argv[4] override the declared contact count and maxNumContacts, so the
    // clamp path can be driven standalone.
    const uint32_t argCount = (argc > 3) ? uint32_t(std::atoi(argv[3])) : 0u;
    const uint32_t argMax   = (argc > 4) ? uint32_t(std::atoi(argv[4])) : 0u;
    // argv[5]: sliver thickness.  Flattens tet 0's apex to (0.3, 0.3, thickness) so the
    // barycentric determinants become ill-conditioned, and moves contact 1 far outside, to
    // see whether the fallback's per-component clamp is observable at all.
    const float argSliver = (argc > 5) ? float(std::atof(argv[5])) : 0.0f;
    const float argPx = (argc > 6) ? float(std::atof(argv[6])) : -5.0f;

    Ctx c;
    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ii{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO }; ii.pApplicationInfo = &app;
    VK(vkCreateInstance(&ii, nullptr, &c.inst));
    uint32_t n = 0; VK(vkEnumeratePhysicalDevices(c.inst, &n, nullptr));
    std::vector<VkPhysicalDevice> d(n); VK(vkEnumeratePhysicalDevices(c.inst, &n, d.data()));
    c.phys = d[0];
    VkPhysicalDeviceProperties pp{}; vkGetPhysicalDeviceProperties(c.phys, &pp);
    std::printf("device: %s   candidates: %u\n", pp.deviceName, nbCandidates);
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qs(qn); vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &qn, qs.data());
    for (uint32_t i = 0; i < qn; ++i) if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c.qf = i; break; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qi.queueFamilyIndex = c.qf; qi.queueCount = 1; qi.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan12Features f12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    f12.bufferDeviceAddress = VK_TRUE;
    // shaderInt64 is NOT optional for these kernels: every device address is computed in
    // 64-bit integer arithmetic.  The gpuslang bridge enables it
    // (PxgVulkanComputeDevice.cpp), so a repro that does not is not reproducing the same
    // device.  repro/repro2 got away without it; this kernel does not.
    VkPhysicalDeviceFeatures2 f2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    f2.features.shaderInt64 = VK_TRUE;
    f2.pNext = &f12;
    VkDeviceCreateInfo di{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    di.pNext = &f2; di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    VK(vkCreateDevice(c.phys, &di, nullptr, &c.dev));
    vkGetDeviceQueue(c.dev, c.qf, 0, &c.q);

    // --- buffers, all wildly over-allocated -------------------------------------
    const uint32_t NC = 2u;                     // contacts
    Buf simVerts = mkBuf(c, 64 * 1024);
    Buf simTets  = mkBuf(c, 64 * 1024);
    Buf remapCol = mkBuf(c, 64 * 1024);
    Buf accum    = mkBuf(c, 64 * 1024);
    Buf desc     = mkBuf(c, SIZEOF_PxgSoftBody + 4096);
    Buf point    = mkBuf(c, 64 * 1024);
    Buf bary     = mkBuf(c, 64 * 1024);
    Buf info     = mkBuf(c, 64 * 1024);
    Buf count    = mkBuf(c, 4096);

    // nbCandidates sim tets, each with its own four vertices.  Tet 0 is the unit tet at the
    // origin and CONTAINS the first contact point; the rest march away along +x.
    float* sv = static_cast<float*>(simVerts.mapped);
    uint32_t* st = static_cast<uint32_t*>(simTets.mapped);
    for (uint32_t t = 0; t < nbCandidates; ++t)
    {
        const float ox = float(t) * 4.0f;
        const float corner[4][3] = { { ox, 0, 0 }, { ox + 1, 0, 0 }, { ox, 1, 0 }, { ox, 0, 1 } };
        for (uint32_t k = 0; k < 4u; ++k)
        {
            sv[(t * 4 + k) * 4 + 0] = corner[k][0];
            sv[(t * 4 + k) * 4 + 1] = corner[k][1];
            sv[(t * 4 + k) * 4 + 2] = corner[k][2];
            sv[(t * 4 + k) * 4 + 3] = 1.0f;
            st[t * 4 + k] = t * 4 + k;
        }
    }
    uint32_t* rc = static_cast<uint32_t*>(remapCol.mapped);
    for (uint32_t t = 0; t < nbCandidates; ++t) rc[t] = t;
    static_cast<uint32_t*>(accum.mapped)[0] = nbCandidates;   // inclusive prefix sum, tet 0

    uint8_t* dp = static_cast<uint8_t*>(desc.mapped);
    auto setPtr = [&](size_t off, uint64_t v) { std::memcpy(dp + off, &v, 8); };
    setPtr(O_mSimTetIndices, simTets.addr);
    setPtr(O_mSimPosition_InvMass, simVerts.addr);
    setPtr(O_mTetsRemapColToSim, remapCol.addr);
    setPtr(O_mTetsAccumulatedRemapColToSim, accum.addr);

    // contact 0 inside tet 0; contact 1 outside every candidate.
    float* pt = static_cast<float*>(point.mapped);
    pt[0] = 0.2f; pt[1] = 0.2f; pt[2] = 0.2f; pt[3] = 0.0f;
    pt[4] = argPx; pt[5] = 0.2f; pt[6] = 0.2f; pt[7] = 0.0f;
    if (argSliver != 0.0f)
    {
        sv[3 * 4 + 0] = 0.3f; sv[3 * 4 + 1] = 0.3f; sv[3 * 4 + 2] = argSliver;
    }
    // PxgFemOtherContactInfo (16 B): pairInd0 u64 @0, pairInd1 u32 @8, rigidMatInd @12.
    // Both contacts name softbody 0, collision tet 0 -> pairInd1 == 0.
    const uint32_t nContacts = argCount ? argCount : NC;
    const uint32_t nMax = argMax ? argMax : 64u;
    static_cast<uint32_t*>(count.mapped)[0] = nContacts;
    // sentinel the whole barycentric buffer so an out-of-bounds write is visible
    float* bpi = static_cast<float*>(bary.mapped);
    for (uint32_t i = 0; i < 4u * 256u; ++i) bpi[i] = -1.0f;

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
    cpi.stage.pName = "sb_other_contact_remap_to_simLaunch"; cpi.layout = layout;
    VkPipeline pipe{};
    std::printf("creating the pipeline...\n"); std::fflush(stdout);
    VK(vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));
    std::printf("pipeline created\n"); std::fflush(stdout);

    VkCommandPoolCreateInfo cpci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO }; cpci.queueFamilyIndex = c.qf;
    VkCommandPool pool{}; VK(vkCreateCommandPool(c.dev, &cpci, nullptr, &pool));
    VkCommandBufferAllocateInfo cbi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbi.commandPool = pool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
    VkCommandBuffer cb{}; VK(vkAllocateCommandBuffers(c.dev, &cbi, &cb));

    Push push{};
    push.softbodies = desc.addr;
    push.maxNumContacts = nMax;
    push.point = point.addr;
    push.outBarycentric = bary.addr;
    push.outContactInfo = info.addr;
    push.totalContactCount = count.addr;
    push.prevContactCount = 0ull;   // OPTIONAL in CUDA — must not be dereferenced

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
    std::printf("submitting the REAL remap kernel...\n"); std::fflush(stdout);
    VK(vkQueueSubmit(c.q, 1, &si, fence));
    VK(vkWaitForFences(c.dev, 1, &fence, VK_TRUE, ~0ull));
    std::printf("dispatch survived\n");

    const uint32_t* ip = static_cast<const uint32_t*>(info.mapped);
    const float* bp = static_cast<const float*>(bary.mapped);
    std::printf("  count read back = %u\n", static_cast<const uint32_t*>(count.mapped)[0]);
    for (uint32_t i = 0; i < nContacts; ++i)
        std::printf("  contact %u: pairInd1 = 0x%08x (tet %u)  bary = (%.4f, %.4f, %.4f, %.4f)\n",
                    i, ip[i * 4 + 2], ip[i * 4 + 2] & 0xfffffu,
                    bp[i * 4 + 0], bp[i * 4 + 1], bp[i * 4 + 2], bp[i * 4 + 3]);
    return 0;
}
