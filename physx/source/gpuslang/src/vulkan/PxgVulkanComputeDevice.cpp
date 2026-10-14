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
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace physx
{
namespace gpuslang
{

// A Vulkan call whose failure this device cannot recover from: recording, submission,
// waiting, mapping for a transfer, the staging ring.  P9.98 check round 0: it ABORTS in
// every build -- it used to assert only, and both build trees are -DNDEBUG, so a failure
// ran on with a null or stale object and crashed later, somewhere unrelated.  Every
// failure a caller CAN act on is checked by hand instead and refused with a reason:
// pipelines (createPipelineInternal), buffers and textures (createBuffer /
// createHostVisibleBuffer / createTexture3D* return 0, which the shim reports as
// OutOfMemory), descriptor sets (the dispatch is refused) and init (no device).
// On VK_ERROR_DEVICE_LOST the device's fault report (VK_EXT_device_fault, when the
// driver offers it) is printed before aborting: the faulting GPU address and the
// allocation it falls in or next to.  Defined after the device class.
static void pxgVkReportFatal(VkResult result);

#define PXG_VK_CHECK(expr)                                                                 \
    do                                                                                     \
    {                                                                                      \
        VkResult pxgVkResult_ = (expr);                                                    \
        if (pxgVkResult_ != VK_SUCCESS)                                                    \
        {                                                                                  \
            fprintf(stderr, "[gpuslang/vulkan] %s failed: %d (%s:%d); aborting\n", #expr,  \
                    int(pxgVkResult_), __FILE__, __LINE__);                                \
            pxgVkReportFatal(pxgVkResult_);                                                \
            fflush(stderr);                                                                \
            std::abort();                                                                  \
        }                                                                                  \
    } while (0)

// P9.97 -- see the declaration in PxgSlangComputeDevice.h.  Module-wide on purpose:
// stricter than per entry, and no module in the build declares anything it refuses
// (P9.97 scan of all 471 .spv).  ONLY the bindless door applies it: createPipeline
// must not, because sdfCollBaryClamp and sdfTexProbe share a module with set-1
// entries and are built without set 1 (P9.84 classifies per entry).
bool pxgSpirvFitsBindlessLayout(const void* code, size_t codeSize, uint32_t* badSet, uint32_t* badBinding)
{
    if (badSet)
        *badSet = ~0u;
    if (badBinding)
        *badBinding = ~0u;
    if (!code || codeSize < 20 || (codeSize & 3u) != 0)
        return false;
    const size_t n = codeSize / 4;
    std::vector<uint32_t> w(n);
    std::memcpy(w.data(), code, n * 4);
    if (w[0] != 0x07230203u)
        return false;
    std::unordered_map<uint32_t, uint32_t> setOf, bindingOf;
    for (size_t i = 5; i < n;)
    {
        const uint32_t wc = w[i] >> 16, op = w[i] & 0xFFFFu;
        if (wc == 0 || i + wc > n)
            return false;
        if (op == 71u && wc >= 4u)               // OpDecorate <id> <decoration> <literal>
        {
            if (w[i + 2] == 34u)                 // DescriptorSet
                setOf[w[i + 1]] = w[i + 3];
            else if (w[i + 2] == 33u)            // Binding
                bindingOf[w[i + 1]] = w[i + 3];
        }
        i += wc;
    }
    for (const auto& v : setOf)
    {
        const uint32_t set = v.second;
        const auto b = bindingOf.find(v.first);
        const uint32_t binding = b == bindingOf.end() ? ~0u : b->second;
        if (set > 1u || (set == 1u && binding != 0u))
        {
            if (badSet)
                *badSet = set;
            if (badBinding)
                *badBinding = binding;
            return false;
        }
    }
    return true;
}

bool pxgSpirvDeclaresSampler(const void* code, size_t codeSize)
{
    if (!code || codeSize < 20 || (codeSize & 3u) != 0)
        return false;
    const size_t n = codeSize / 4;
    std::vector<uint32_t> w(n);
    std::memcpy(w.data(), code, n * 4);
    if (w[0] != 0x07230203u)
        return false;
    for (size_t i = 5; i < n;)
    {
        const uint32_t wc = w[i] >> 16, op = w[i] & 0xFFFFu;
        if (wc == 0 || i + wc > n)
            return false;
        if (op == 26u /* OpTypeSampler */ || op == 27u /* OpTypeSampledImage */)
            return true;
        i += wc;
    }
    return false;
}

// P9.98 -- see the declaration in PxgSlangComputeDevice.h.  The size rules are the two
// static checkers' (tools/check_push_block.py for the push block, rounded to 4;
// tools/check_groupshared.py for Workgroup storage, including its struct rounding),
// so the device refuses exactly what those tools measure; the P9.98 fixture
// gpuslang.pipelineRefusal pins the numbers for the modules the validation layer
// flagged.
namespace
{
struct SpirvTypeTable
{
    struct Type
    {
        uint32_t op = 0;
        std::vector<uint32_t> a;
    };
    std::unordered_map<uint32_t, Type> types;
    std::unordered_map<uint32_t, uint32_t> consts, arrayStride;
    std::unordered_map<uint64_t, uint32_t> memberOffset;   // (struct << 32 | member) -> Offset

    // P9.98 check round 1: sizes are 64-bit and SATURATE at kHuge.  They were uint32 and
    // wrapped: `groupshared float gA[0x40000000]` (4 GiB) reflected as 0 B, passed the
    // limit check and was built -- and the device crashed at the flush (measured, rc 139),
    // the crash class the limit check exists to turn into a refusal.  kHuge is far above
    // any uint32, so a saturated size can never come back under a limit; the public
    // PxgSpirvEntryInfo then reports it as UINT32_MAX (over every limit), not undecided.
    static constexpr uint64_t kUndecided = ~0ull;
    static constexpr uint64_t kHuge = 1ull << 40;
    static uint64_t sat(uint64_t v) { return v > kHuge ? kHuge : v; }
    static uint64_t mul(uint64_t a, uint64_t b) { return (b != 0 && a > kHuge / b) ? kHuge : sat(a * b); }

    // Byte size of `tid`, or kUndecided.  `workgroup` selects check_groupshared's struct
    // rounding (to the largest member alignment) over the push block's (to 4, applied by
    // the caller).
    uint64_t size(uint32_t tid, bool workgroup, int depth = 0) const
    {
        const auto it = types.find(tid);
        if (depth > 24 || it == types.end())
            return kUndecided;
        const Type& t = it->second;
        switch (t.op)
        {
        case 20u: return 4u;                                               // OpTypeBool
        case 21u: case 22u: return t.a.empty() ? kUndecided : (t.a[0] / 8u ? t.a[0] / 8u : 1u); // Int/Float
        case 23u: case 24u:                                                // Vector/Matrix
        {
            if (t.a.size() < 2)
                return kUndecided;
            const uint64_t e = size(t.a[0], workgroup, depth + 1);
            return e == kUndecided ? kUndecided : mul(e, t.a[1]);
        }
        case 28u:                                                          // OpTypeArray
        {
            if (t.a.size() < 2)
                return kUndecided;
            const auto n = consts.find(t.a[1]);
            if (n == consts.end())
                return kUndecided;
            const auto st = arrayStride.find(tid);
            const uint64_t stride = st != arrayStride.end() ? st->second : size(t.a[0], workgroup, depth + 1);
            return stride == kUndecided ? kUndecided : mul(stride, n->second);
        }
        case 29u: return 0u;                                               // RuntimeArray
        case 32u: return 8u;                                               // pointer (BDA)
        case 30u:                                                          // OpTypeStruct
        {
            if (t.a.empty())
                return 0u;
            uint64_t total = 0, align = 1;
            bool decorated = true;
            for (size_t m = 0; m < t.a.size(); ++m)
                if (memberOffset.find((uint64_t(tid) << 32) | m) == memberOffset.end())
                    decorated = false;
            for (size_t m = 0; m < t.a.size(); ++m)
            {
                const uint64_t ms = size(t.a[m], workgroup, depth + 1);
                if (ms == kUndecided)
                    return kUndecided;
                const uint64_t a = (ms == 1u || ms == 2u || ms == 4u || ms == 8u || ms == 16u) ? ms : 16u;
                align = a > align ? a : align;
                if (decorated)
                {
                    const uint64_t end = sat(memberOffset.at((uint64_t(tid) << 32) | m) + ms);
                    total = end > total ? end : total;
                }
                else
                    total = sat((total + a - 1u) / a * a + ms);
            }
            if (workgroup || !decorated)
                total = (total + align - 1u) / align * align;
            else
                total = (total + 3u) & ~uint64_t(3);   // check_push_block.struct_size
            return sat(total);
        }
        default: return kUndecided;
        }
    }
};

// A 64-bit size as PxgSpirvEntryInfo reports it: UINT32_MAX for anything that does not
// fit (over every device limit, so it is refused -- never wrapped to a small number).
static uint32_t reportedBytes(uint64_t v)
{
    return v > 0xFFFFFFFFull ? 0xFFFFFFFFu : uint32_t(v);
}
} // namespace

// *malformed tells the two false results apart for createPipelineInternal's reason: the
// blob is not a well-formed word stream (true), or it is and has no such entry (false).
static bool reflectEntryImpl(const void* code, size_t codeSize, const char* entry, PxgSpirvEntryInfo* out,
                             bool* malformed)
{
    *malformed = true;
    if (!code || !entry || codeSize < 20 || (codeSize & 3u) != 0)
        return false;
    const size_t n = codeSize / 4;
    std::vector<uint32_t> w(n);
    std::memcpy(w.data(), code, n * 4);
    if (w[0] != 0x07230203u)
        return false;
    const bool perEntry = w[1] >= 0x00010400u;   // interface lists every global since 1.4

    SpirvTypeTable tt;
    struct Var
    {
        uint32_t id, pointee, storage;
    };
    std::vector<Var> vars;
    std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> pointers;   // id -> (storage, pointee)
    size_t epIface = 0, epEnd = 0;
    uint32_t epFn = 0;
    bool found = false;
    struct ExecMode
    {
        uint32_t fn, mode, x, y, z;
        bool ids;
    };
    std::vector<ExecMode> modes;
    const size_t entryLen = std::strlen(entry);
    for (size_t i = 5; i < n;)
    {
        const uint32_t wc = w[i] >> 16, op = w[i] & 0xFFFFu;
        if (wc == 0 || i + wc > n)
            return false;
        if (op == 15u && wc >= 4u)                                  // OpEntryPoint
        {
            const char* name = reinterpret_cast<const char*>(&w[i + 3]);
            const size_t maxBytes = size_t(wc - 3) * 4;
            const size_t len = strnlen(name, maxBytes);
            if (!found && len < maxBytes && len == entryLen && std::memcmp(name, entry, len) == 0 &&
                w[i + 1] == 5u /* GLCompute */)
            {
                found = true;
                epFn = w[i + 2];
                epIface = i + 3 + (len + 4) / 4;
                epEnd = i + wc;
            }
        }
        else if ((op == 16u || op == 331u) && wc >= 6u &&           // OpExecutionMode(Id)
                 (w[i + 2] == 17u || w[i + 2] == 38u))              // LocalSize / LocalSizeId
            modes.push_back({ w[i + 1], w[i + 2], w[i + 3], w[i + 4], w[i + 5], op == 331u || w[i + 2] == 38u });
        else if (op == 71u && wc >= 4u && w[i + 2] == 6u)           // OpDecorate ArrayStride
            tt.arrayStride[w[i + 1]] = w[i + 3];
        else if (op == 72u && wc >= 5u && w[i + 3] == 35u)          // OpMemberDecorate Offset
            tt.memberOffset[(uint64_t(w[i + 1]) << 32) | w[i + 2]] = w[i + 4];
        else if (op == 43u && wc >= 4u)                             // OpConstant (low word)
            tt.consts[w[i + 2]] = w[i + 3];
        else if (op == 32u && wc >= 4u)                             // OpTypePointer
        {
            pointers[w[i + 1]] = { w[i + 2], w[i + 3] };
            tt.types[w[i + 1]].op = op;
        }
        else if ((op >= 20u && op <= 24u) || op == 28u || op == 29u || op == 30u)
        {
            // Every type instruction has a result id (word 1); a word count of 1 is not
            // a well-formed instruction, and the assign below would be a reversed range.
            if (wc < 2u)
                return false;
            SpirvTypeTable::Type& t = tt.types[w[i + 1]];
            t.op = op;
            t.a.assign(w.begin() + i + 2, w.begin() + i + wc);
        }
        else if (op == 59u && wc >= 4u)                             // OpVariable
        {
            const auto p = pointers.find(w[i + 1]);
            if (p != pointers.end())
                vars.push_back({ w[i + 2], p->second.second, w[i + 3] });
        }
        i += wc;
    }
    *malformed = false;
    if (!found)
        return false;

    PxgSpirvEntryInfo info;
    info.perEntry = perEntry;
    uint64_t pushBytes = 0, workgroupBytes = 0;   // 64-bit, saturating: see SpirvTypeTable
    for (const Var& v : vars)
    {
        if (v.storage != 9u && v.storage != 4u)                    // PushConstant / Workgroup
            continue;
        if (perEntry)
        {
            bool listed = false;
            for (size_t k = epIface; k < epEnd && !listed; ++k)
                listed = w[k] == v.id;
            if (!listed)
                continue;
        }
        if (v.storage == 9u)
        {
            const uint64_t s = tt.size(v.pointee, false);
            if (s == SpirvTypeTable::kUndecided)
                info.pushDecided = false;
            else
            {
                const uint64_t r = (s + 3u) & ~uint64_t(3);
                pushBytes = r > pushBytes ? r : pushBytes;
            }
        }
        else
        {
            const uint64_t s = tt.size(v.pointee, true);
            if (s == SpirvTypeTable::kUndecided)
                info.workgroupDecided = false;
            else
                workgroupBytes = SpirvTypeTable::sat(workgroupBytes + s);
        }
    }
    info.pushBlockBytes = reportedBytes(pushBytes);
    info.workgroupBytes = reportedBytes(workgroupBytes);
    for (const ExecMode& m : modes)
    {
        if (m.fn != epFn)
            continue;
        const uint32_t xyz[3] = { m.x, m.y, m.z };
        for (int d = 0; d < 3; ++d)
        {
            if (!m.ids)
                info.localSize[d] = xyz[d];
            else
            {
                const auto c = tt.consts.find(xyz[d]);
                info.localSize[d] = c == tt.consts.end() ? 0u : c->second;
            }
        }
    }
    if (out)
        *out = info;
    return true;
}

bool pxgSpirvReflectEntry(const void* code, size_t codeSize, const char* entry, PxgSpirvEntryInfo* out)
{
    bool malformed = false;
    return reflectEntryImpl(code, codeSize, entry, out, &malformed);
}

// P9.99: createPipelineEx's contract -- an EXACT subgroup size with FULL subgroups -- checked
// against the device and the entry's workgroup shape, in the order a reader would ask.
// Before P9.99 only the window was checked: llvmpipe builds every request it is handed
// (measured: 4, 6, 12, 16 and 256 at a window of [8, 8], VkResult 0), and an in-window
// size on an X-not-a-multiple entry ran PARTIAL subgroups under REQUIRE_FULL_SUBGROUPS.
bool pxgRequiredSubgroupSizeFits(const PxgSlangComputeDevice::SubgroupCaps& caps, uint32_t size,
                                 const uint32_t localSize[3], char* why, size_t whyLen)
{
    char scratch[4];
    if (!why || whyLen == 0)
    {
        why = scratch;
        whyLen = sizeof(scratch);
    }
    why[0] = '\0';
    if (size == 0)
        return true;
    if (!caps.canRequireSize)
    {
        std::snprintf(why, whyLen, "required subgroup size %u: the device cannot require a subgroup size in compute "
                                   "(subgroupSizeControl, requiredSubgroupSizeStages; VUID-02755)", size);
        return false;
    }
    if (!caps.fullSubgroups)
    {
        std::snprintf(why, whyLen, "required subgroup size %u: the device has no computeFullSubgroups, and "
                                   "createPipelineEx promises full subgroups (VUID-02785; PXG_WARP_NATIVE)", size);
        return false;
    }
    if ((size & (size - 1u)) != 0u)
    {
        std::snprintf(why, whyLen, "required subgroup size %u is not a power of two (VUID-02760)", size);
        return false;
    }
    if (size < caps.minSize || size > caps.maxSize)
    {
        // Name the bound actually crossed: 02761 is the minimum, 02762 the maximum.
        std::snprintf(why, whyLen, "required subgroup size %u is %s the device's window [%u, %u] (%s)", size,
                      size < caps.minSize ? "below" : "above", caps.minSize, caps.maxSize,
                      size < caps.minSize ? "VUID-02761" : "VUID-02762");
        return false;
    }
    if (localSize[0] == 0u || localSize[1] == 0u || localSize[2] == 0u)
    {
        std::snprintf(why, whyLen, "required subgroup size %u: the entry's workgroup size is not readable from "
                                   "the SPIR-V, so full subgroups cannot be shown", size);
        return false;
    }
    if (localSize[0] % size != 0u)
    {
        std::snprintf(why, whyLen, "required subgroup size %u: workgroup size x %u is not a multiple of it, so "
                                   "not every subgroup can be full (VUID-02757)", size, localSize[0]);
        return false;
    }
    const uint64_t invocations = uint64_t(localSize[0]) * localSize[1] * localSize[2];
    const uint64_t most = uint64_t(size) * caps.maxComputeWorkgroupSubgroups;
    if (invocations > most)
    {
        std::snprintf(why, whyLen, "required subgroup size %u: workgroup (%u, %u, %u) = %llu invocations exceeds "
                                   "%u x maxComputeWorkgroupSubgroups %u = %llu (VUID-02756)", size, localSize[0],
                      localSize[1], localSize[2], (unsigned long long)invocations, size,
                      caps.maxComputeWorkgroupSubgroups, (unsigned long long)most);
        return false;
    }
    return true;
}

class PxgVulkanComputeDevice;
// The live device for pxgVkReportFatal (a process has one).
static PxgVulkanComputeDevice* gFaultReportDevice = nullptr;

class PxgVulkanComputeDevice : public PxgSlangComputeDevice
{
private:
    struct Pipeline;   // defined with the other private state below
    struct Buffer;
    struct Texture;

public:
    PxgVulkanComputeDevice() {}

    ~PxgVulkanComputeDevice() override
    {
        if (gFaultReportDevice == this)
            gFaultReportDevice = nullptr;
        if (mDevice)
        {
            vkDeviceWaitIdle(mDevice);

            for (auto& kv : mBuffers)
            {
                vkDestroyBuffer(mDevice, kv.second.buffer, nullptr);
                freeMemoryCounted(kv.second.memory);
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
                    freeMemoryCounted(frame.stagingMemory);
                }
            }
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

    // P9.98 check round 0: an init step that fails makes NO device (the factory returns
    // null, the destructor releases what exists) instead of a device with null objects.
    // P9.98 check round 1: the Vulkan spec leaves a create call's output UNDEFINED when it
    // fails (Fundamentals > Return Codes: only vkAllocateCommandBuffers,
    // vkAllocateDescriptorSets and vkCreate*Pipelines promise VK_NULL_HANDLE), and every
    // refusal path -- refuseBuffer, refuseTexture, refusePipeline, the destructor after a
    // failed init -- destroys the non-null objects it finds.  So each such call's output is
    // reset here before anything can read it: a driver that leaves junk behind must not
    // have that junk handed to vkDestroy* / vkFree*.
    template <class H>
    static VkResult madeOrNull(VkResult r, H* out)
    {
        if (r != VK_SUCCESS)
            *out = VK_NULL_HANDLE;
        return r;
    }

    static bool initStep(VkResult r, const char* what)
    {
        if (r == VK_SUCCESS)
            return true;
        std::fprintf(stderr, "[PxgVulkan] init: %s failed with VkResult %d; no device\n", what, int(r));
        return false;
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

        // P9.98: PXG_VK_DEBUG_NAMES=1 (set by tools/run_validation.sh) or enableValidation
        // names every shader module, pipeline layout and pipeline after its ENTRY POINT
        // through VK_EXT_debug_utils, so each validation-layer message carries
        // `name = <entry>` and can be attributed -- and a waiver scoped to one module --
        // without an LD_PRELOAD probe.  Off otherwise; no effect on what runs.
        std::vector<const char*> instExts;
        {
            const char* dn = std::getenv("PXG_VK_DEBUG_NAMES");
            const bool wantNames = desc.enableValidation || (dn && std::strcmp(dn, "0") != 0);
            mDebugNames = wantNames;
            uint32_t extCount = 0;
            if (wantNames && vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr) == VK_SUCCESS)
            {
                std::vector<VkExtensionProperties> exts(extCount);
                vkEnumerateInstanceExtensionProperties(nullptr, &extCount, exts.data());
                for (const VkExtensionProperties& e : exts)
                    if (std::strcmp(e.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0)
                        instExts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            }
        }

        VkInstanceCreateInfo instInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        instInfo.pApplicationInfo = &appInfo;
        instInfo.enabledLayerCount = uint32_t(layers.size());
        instInfo.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
        instInfo.enabledExtensionCount = uint32_t(instExts.size());
        instInfo.ppEnabledExtensionNames = instExts.empty() ? nullptr : instExts.data();

        if (madeOrNull(vkCreateInstance(&instInfo, nullptr, &mInstance), &mInstance) != VK_SUCCESS)
            return false;
        if (!instExts.empty())
            mSetObjectName = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
                vkGetInstanceProcAddr(mInstance, "vkSetDebugUtilsObjectNameEXT"));

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
        allocStats().maxCount = props.limits.maxMemoryAllocationCount;

        // P9.98: the limits every pipeline is checked against at creation.  Nothing
        // compared them before, and llvmpipe's are generous (push 256, shared 32768),
        // so a kernel over a smaller device's limit failed vkCreateComputePipelines
        // there -- and that failure was not handled either (F4).  The test hooks may
        // only LOWER a limit.
        mLimits.maxPushConstantsSize = props.limits.maxPushConstantsSize;
        mLimits.maxComputeSharedMemorySize = props.limits.maxComputeSharedMemorySize;
        mLimits.maxComputeWorkGroupInvocations = props.limits.maxComputeWorkGroupInvocations;
        for (int d = 0; d < 3; ++d)
        {
            mLimits.maxComputeWorkGroupSize[d] = props.limits.maxComputeWorkGroupSize[d];
            mLimits.maxComputeWorkGroupCount[d] = props.limits.maxComputeWorkGroupCount[d];
        }
        {
            auto lower = [](uint32_t& limit, uint32_t hook, const char* env) {
                if (hook != 0 && hook < limit)
                    limit = hook;
                if (const char* e = std::getenv(env))
                {
                    const unsigned long v = std::strtoul(e, nullptr, 0);
                    if (v != 0 && v < limit)
                        limit = uint32_t(v);
                }
            };
            lower(mLimits.maxPushConstantsSize, desc.limitMaxPushConstantsSize, "PXG_VK_LIMIT_PUSH_CONSTANTS");
            lower(mLimits.maxComputeSharedMemorySize, desc.limitMaxComputeSharedMemorySize,
                  "PXG_VK_LIMIT_SHARED_MEMORY");
            lower(mLimits.maxComputeWorkGroupInvocations, desc.limitMaxComputeWorkGroupInvocations,
                  "PXG_VK_LIMIT_WORKGROUP_INVOCATIONS");
            for (int d = 0; d < 3; ++d)
            {
                lower(mLimits.maxComputeWorkGroupSize[d], desc.limitMaxComputeWorkGroupSize,
                      "PXG_VK_LIMIT_WORKGROUP_SIZE");
                lower(mLimits.maxComputeWorkGroupCount[d], desc.limitMaxComputeWorkGroupCount,
                      "PXG_VK_LIMIT_WORKGROUP_COUNT");
            }
        }

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
        bool deviceFaultExtPresent = false;
        {
            uint32_t extCount = 0;
            vkEnumerateDeviceExtensionProperties(mPhysicalDevice, nullptr, &extCount, nullptr);
            std::vector<VkExtensionProperties> exts(extCount);
            if (extCount)
                vkEnumerateDeviceExtensionProperties(mPhysicalDevice, nullptr, &extCount, exts.data());
            for (const VkExtensionProperties& e : exts)
            {
                if (std::strcmp(e.extensionName, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) == 0)
                    atomicFloatExtPresent = true;
#ifdef VK_EXT_device_fault
                if (std::strcmp(e.extensionName, VK_EXT_DEVICE_FAULT_EXTENSION_NAME) == 0)
                    deviceFaultExtPresent = true;
#endif
            }
        }
        (void)deviceFaultExtPresent;

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
#ifdef VK_EXT_device_fault
        // VK_EXT_device_fault: diagnostics only.  After VK_ERROR_DEVICE_LOST it tells
        // which GPU address faulted; nothing depends on it being present.
        VkPhysicalDeviceFaultFeaturesEXT faultSupported = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT };
        if (deviceFaultExtPresent)
        {
            faultSupported.pNext = supported2.pNext;
            supported2.pNext = &faultSupported;
        }
#endif
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
        mSubgroupCaps.maxComputeWorkgroupSubgroups = sgSizeProps.maxComputeWorkgroupSubgroups;
        // P9.99 test hooks (PxgSlangDeviceDesc): they only take a capability away, so a
        // fixture can show createPipelineEx's refusals on llvmpipe; the features stay enabled.
        if (desc.noSubgroupSizeControl)
            mSubgroupCaps.canRequireSize = false;
        if (desc.noFullSubgroups)
            mSubgroupCaps.fullSubgroups = false;
        if (desc.limitMaxComputeWorkgroupSubgroups != 0 &&
            desc.limitMaxComputeWorkgroupSubgroups < mSubgroupCaps.maxComputeWorkgroupSubgroups)
            mSubgroupCaps.maxComputeWorkgroupSubgroups = desc.limitMaxComputeWorkgroupSubgroups;

        std::vector<const char*> deviceExts;
        if (mHasShaderBufferFloat32AtomicAdd)
            deviceExts.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
#ifdef VK_EXT_device_fault
        VkPhysicalDeviceFaultFeaturesEXT enabledFault = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT };
        if (deviceFaultExtPresent && faultSupported.deviceFault == VK_TRUE)
        {
            enabledFault.deviceFault = VK_TRUE;
            enabledFault.pNext = enabled2.pNext;
            enabled2.pNext = &enabledFault;
            deviceExts.push_back(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
            mHasDeviceFault = true;
        }
#endif

        VkDeviceCreateInfo deviceInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        deviceInfo.pNext = &enabled2;
        deviceInfo.enabledExtensionCount = uint32_t(deviceExts.size());
        deviceInfo.ppEnabledExtensionNames = deviceExts.empty() ? nullptr : deviceExts.data();

        if (madeOrNull(vkCreateDevice(mPhysicalDevice, &deviceInfo, nullptr, &mDevice), &mDevice) != VK_SUCCESS)
            return false;

        vkGetDeviceQueue(mDevice, mQueueFamily, 0, &mQueue);
#ifdef VK_EXT_device_fault
        if (mHasDeviceFault)
            mGetDeviceFaultInfo = reinterpret_cast<PFN_vkGetDeviceFaultInfoEXT>(
                vkGetDeviceProcAddr(mDevice, "vkGetDeviceFaultInfoEXT"));
#endif
        gFaultReportDevice = this;

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
        if (!initStep(madeOrNull(vkCreateCommandPool(mDevice, &poolInfo, nullptr, &mCommandPool), &mCommandPool),
                      "vkCreateCommandPool"))
            return false;

        VkCommandBufferAllocateInfo cmdInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cmdInfo.commandPool = mCommandPool;
        cmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdInfo.commandBufferCount = kFramesInFlight;
        VkCommandBuffer cmds[kFramesInFlight] = {};
        if (!initStep(vkAllocateCommandBuffers(mDevice, &cmdInfo, cmds), "vkAllocateCommandBuffers"))
            return false;
        for (uint32_t i = 0; i < kFramesInFlight; ++i)
            mFrames[i].cmd = cmds[i];

        VkFenceCreateInfo fenceInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        if (!initStep(madeOrNull(vkCreateFence(mDevice, &fenceInfo, nullptr, &mFence), &mFence), "vkCreateFence"))
            return false;

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
            // A failure here is not fatal: the blocking-fence path below covers devices
            // without timeline semaphores, so the device falls back to it -- and says so.
            const VkResult r = vkCreateSemaphore(mDevice, &semInfo, nullptr, &mTimeline);
            if (r != VK_SUCCESS)
            {
                std::fprintf(stderr, "[PxgVulkan] init: vkCreateSemaphore (timeline) failed with VkResult %d; "
                                     "using blocking fence submission\n", int(r));
                mTimeline = VK_NULL_HANDLE;
            }
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
            if (madeOrNull(vkCreateQueryPool(mDevice, &qpInfo, nullptr, &mQueryPool), &mQueryPool) == VK_SUCCESS)
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
        // P9.98 check round 1: the cache is OPTIONAL -- vkCreateComputePipelines takes
        // VK_NULL_HANDLE, the destructor null-checks it and savePipelineCache is skipped
        // without one -- so a failure is reported and the device carries on without it, as
        // it does for the timeline semaphore and the query pool.  (Round 0 had made it a
        // fatal init step; before that it was an assert-only check.)
        {
            const VkResult r =
                madeOrNull(vkCreatePipelineCache(mDevice, &cacheInfo, nullptr, &mPipelineCache), &mPipelineCache);
            if (r != VK_SUCCESS)
                std::fprintf(stderr, "[PxgVulkan] init: vkCreatePipelineCache failed with VkResult %d; continuing "
                                     "without a pipeline cache\n", int(r));
        }

        vkGetPhysicalDeviceMemoryProperties(mPhysicalDevice, &mMemoryProperties);

        return true;
    }

    const char* getDeviceName() const override { return mDeviceName.c_str(); }
    const char* getBackendName() const override { return "Vulkan"; }

    // ---- resources ----

    PxgBufferHandle createBuffer(size_t sizeBytes) override
    {
        // v2 memory mode: DEVICE_LOCAL storage with staged transfers.  The
        // staged path is taken even when the winning memory type happens to
        // also be host-visible (UMA/software devices) so the production
        // transfer path is exercised everywhere, CI included.
        return createBufferInternal(sizeBytes, !mDesc.useDeviceLocalMemory, nullptr, "createBuffer");
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
        if (mappedPtr)
            *mappedPtr = nullptr;
        void* mapped = nullptr;
        const PxgBufferHandle handle = createBufferInternal(sizeBytes, true, &mapped, "createHostVisibleBuffer");
        if (handle && mappedPtr)
            *mappedPtr = mapped;
        return handle;
    }

    // P9.98 check round 0: every step is checked IN RELEASE BUILDS.  A failure releases
    // whatever the attempt had made, says which call failed, and returns 0 -- which the
    // shim's memAlloc / memHostAlloc report as OutOfMemory (the bridge's
    // CUDA_ERROR_OUT_OF_MEMORY).  These were PXG_VK_CHECKs, assert-only in Release: a
    // failed vkAllocateMemory went on to bind and address a null VkDeviceMemory and
    // segfaulted inside createBuffer (measured, rc 139, both memory modes).
    // `persistentMap` non-null = the pinned-host form: HOST_VISIBLE|HOST_COHERENT in both
    // memory modes, mapped for the buffer's lifetime.
    PxgBufferHandle createBufferInternal(size_t sizeBytes, bool hostVisible, void** persistentMap,
                                         const char* caller)
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
        buf.hostVisible = hostVisible;
        VkResult r = madeOrNull(vkCreateBuffer(mDevice, &bufferInfo, nullptr, &buf.buffer), &buf.buffer);
        if (r != VK_SUCCESS)
            return refuseBuffer(buf, caller, sizeBytes, "vkCreateBuffer", r);

        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(mDevice, buf.buffer, &memReq);
        const uint32_t memoryType = hostVisible
            ? findMemoryType(memReq.memoryTypeBits,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
            : findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (memoryType == kNoMemoryType)
            return refuseBuffer(buf, caller, sizeBytes, "no memory type with the required properties", VK_SUCCESS);

        VkMemoryAllocateFlagsInfo allocFlags = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
        allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

        VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocInfo.pNext = mHasBufferDeviceAddress ? &allocFlags : nullptr;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = memoryType;
        r = madeOrNull(allocateMemoryCounted(&allocInfo, &buf.memory), &buf.memory);
        if (r != VK_SUCCESS)
            return refuseBuffer(buf, caller, sizeBytes, "vkAllocateMemory", r);
        r = vkBindBufferMemory(mDevice, buf.buffer, buf.memory, 0);
        if (r != VK_SUCCESS)
            return refuseBuffer(buf, caller, sizeBytes, "vkBindBufferMemory", r);

        if (mHasBufferDeviceAddress)
        {
            VkBufferDeviceAddressInfo addrInfo = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
            addrInfo.buffer = buf.buffer;
            buf.deviceAddress = vkGetBufferDeviceAddress(mDevice, &addrInfo);
        }

        if (persistentMap)
        {
            r = vkMapMemory(mDevice, buf.memory, 0, VK_WHOLE_SIZE, 0, &buf.mappedPtr);
            if (r != VK_SUCCESS || !buf.mappedPtr)
            {
                buf.mappedPtr = nullptr;   // nothing to unmap
                return refuseBuffer(buf, caller, sizeBytes, "vkMapMemory", r == VK_SUCCESS ? VK_ERROR_MEMORY_MAP_FAILED : r);
            }
            *persistentMap = buf.mappedPtr;
        }

        const PxgBufferHandle handle = mNextBufferHandle++;
        mBuffers[handle] = buf;
        return handle;
    }

    PxgBufferHandle refuseBuffer(Buffer& buf, const char* caller, size_t sizeBytes, const char* what, VkResult r)
    {
        if (r != VK_SUCCESS)
            std::fprintf(stderr, "[PxgVulkan] %s(%zu B): %s failed with VkResult %d; returning 0\n", caller,
                         sizeBytes, what, int(r));
        else
            std::fprintf(stderr, "[PxgVulkan] %s(%zu B): %s; returning 0\n", caller, sizeBytes, what);
        if (buf.buffer)
            vkDestroyBuffer(mDevice, buf.buffer, nullptr);
        if (buf.memory)
            freeMemoryCounted(buf.memory);
        buf = Buffer{};
        return 0;
    }

    // P9.98 check round 0: buffers are looked up with find(), never operator[], which
    // default-inserted a Buffer of null handles for an unknown id (a 0 from a refused
    // createBuffer, a destroyed buffer) and then recorded or mapped VK_NULL_HANDLE.  A
    // transfer on such a handle is skipped with a message; a dispatch is refused.
    Buffer* liveBuffer(PxgBufferHandle handle, const char* caller)
    {
        const auto it = handle == 0 ? mBuffers.end() : mBuffers.find(handle);
        if (it == mBuffers.end())
        {
            std::fprintf(stderr, "[PxgVulkan] %s: buffer handle %u is not a live buffer; refused\n", caller,
                         unsigned(handle));
            return nullptr;
        }
        return &it->second;
    }

    // P9.98 check round 1: a transfer must lie inside its buffer.  upload / download memcpy'd
    // through the mapping and copyBuffer / fillBuffer recorded the command with no bounds
    // check: measured, 4096 B at offset 1020 of a 1024-byte buffer was a heap overrun of the
    // pinned mapping or a VUID-vkCmdCopyBuffer-size-00116 / vkMapMemory-size-00681, then
    // rc 139.  Refused like a stale handle: a message, nothing mapped or recorded.
    // Overflow-safe: offset + bytes is never formed.
    static bool inRange(const Buffer& buf, PxgBufferHandle handle, size_t offset, size_t bytes, const char* caller)
    {
        if (offset <= buf.size && bytes <= buf.size - offset)
            return true;
        std::fprintf(stderr, "[PxgVulkan] %s: %zu B at offset %zu runs past buffer %u's %zu bytes; refused\n", caller,
                     bytes, offset, unsigned(handle), buf.size);
        return false;
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
        freeMemoryCounted(it->second.memory);
        mBuffers.erase(it);
    }

    void upload(PxgBufferHandle handle, const void* data, size_t sizeBytes, size_t dstOffset) override
    {
        if (sizeBytes == 0)
            return;
        Buffer* live = liveBuffer(handle, "upload");
        if (!live || !inRange(*live, handle, dstOffset, sizeBytes, "upload"))
            return;
        Buffer& buf = *live;
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
        Buffer* live = liveBuffer(handle, "download");
        if (!live || !inRange(*live, handle, srcOffset, sizeBytes, "download"))
            return;
        Buffer& buf = *live;
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
        const Buffer* liveSrc = liveBuffer(srcHandle, "copyBuffer");
        const Buffer* liveDst = liveBuffer(dstHandle, "copyBuffer");
        if (!liveSrc || !liveDst || !inRange(*liveSrc, srcHandle, srcOffset, sizeBytes, "copyBuffer (source)") ||
            !inRange(*liveDst, dstHandle, dstOffset, sizeBytes, "copyBuffer (destination)"))
            return;
        const Buffer& src = *liveSrc;
        const Buffer& dst = *liveDst;
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
        const Buffer* liveDst = liveBuffer(dstHandle, "fillBuffer");
        if (!liveDst || !inRange(*liveDst, dstHandle, dstOffset, sizeBytes, "fillBuffer"))
            return;
        const Buffer& dst = *liveDst;
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

    // P9.98 check round 0: like createBufferInternal, every step is checked in Release;
    // a failure releases what the attempt made (image, memory, view, staging) and
    // returns 0, which the shim reports as Unsupported (cuTexObjectCreate fails).
    PxgTextureHandle createTexture3DInternal(uint32_t width, uint32_t height, uint32_t depth,
                                             TextureFormat format, const void* texels, bool bindless)
    {
        Texture tex;
        tex.width = width;
        tex.height = height;
        tex.depth = depth;
        tex.bindless = bindless;
        const VkFormat vkFormat = vkFormatOf(format);
        const char* caller = bindless ? "createTexture3DFmt" : "createTexture3D";

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
        VkResult r = madeOrNull(vkCreateImage(mDevice, &imageInfo, nullptr, &tex.image), &tex.image);
        if (r != VK_SUCCESS)
            return refuseTexture(tex, VK_NULL_HANDLE, VK_NULL_HANDLE, caller, "vkCreateImage", r);

        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(mDevice, tex.image, &req);
        VkMemoryAllocateInfo alloc = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (alloc.memoryTypeIndex == kNoMemoryType)
            return refuseTexture(tex, VK_NULL_HANDLE, VK_NULL_HANDLE, caller,
                                 "no device-local memory type for the image", VK_SUCCESS);
        r = madeOrNull(allocateMemoryCounted(&alloc, &tex.memory), &tex.memory);
        if (r != VK_SUCCESS)
            return refuseTexture(tex, VK_NULL_HANDLE, VK_NULL_HANDLE, caller, "vkAllocateMemory", r);
        r = vkBindImageMemory(mDevice, tex.image, tex.memory, 0);
        if (r != VK_SUCCESS)
            return refuseTexture(tex, VK_NULL_HANDLE, VK_NULL_HANDLE, caller, "vkBindImageMemory", r);

        // One-shot staging upload with the required layout transitions.
        const size_t bytes = size_t(width) * height * depth * texelSizeOf(format);
        VkBuffer staging = VK_NULL_HANDLE;
        VkDeviceMemory stagingMem = VK_NULL_HANDLE;
        {
            VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            bi.size = bytes;
            bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            r = madeOrNull(vkCreateBuffer(mDevice, &bi, nullptr, &staging), &staging);
            if (r != VK_SUCCESS)
                return refuseTexture(tex, VK_NULL_HANDLE, VK_NULL_HANDLE, caller, "vkCreateBuffer (staging)", r);
            VkMemoryRequirements sreq;
            vkGetBufferMemoryRequirements(mDevice, staging, &sreq);
            VkMemoryAllocateInfo sa = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
            sa.allocationSize = sreq.size;
            sa.memoryTypeIndex = findMemoryType(
                sreq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (sa.memoryTypeIndex == kNoMemoryType)
                return refuseTexture(tex, staging, VK_NULL_HANDLE, caller,
                                     "no host-visible memory type for the staging buffer", VK_SUCCESS);
            r = madeOrNull(allocateMemoryCounted(&sa, &stagingMem), &stagingMem);
            if (r != VK_SUCCESS)
                return refuseTexture(tex, staging, VK_NULL_HANDLE, caller, "vkAllocateMemory (staging)", r);
            r = vkBindBufferMemory(mDevice, staging, stagingMem, 0);
            if (r != VK_SUCCESS)
                return refuseTexture(tex, staging, stagingMem, caller, "vkBindBufferMemory (staging)", r);
            void* mapped = nullptr;
            r = vkMapMemory(mDevice, stagingMem, 0, VK_WHOLE_SIZE, 0, &mapped);
            if (r != VK_SUCCESS || !mapped)
                return refuseTexture(tex, staging, stagingMem, caller, "vkMapMemory (staging)",
                                     r == VK_SUCCESS ? VK_ERROR_MEMORY_MAP_FAILED : r);
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
        freeMemoryCounted(stagingMem);

        VkImageViewCreateInfo viewInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        viewInfo.image = tex.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
        viewInfo.format = vkFormat;
        viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        r = madeOrNull(vkCreateImageView(mDevice, &viewInfo, nullptr, &tex.view), &tex.view);
        if (r != VK_SUCCESS)
            return refuseTexture(tex, VK_NULL_HANDLE, VK_NULL_HANDLE, caller, "vkCreateImageView", r);

        // Only the bound path samples in hardware: a createTexture3D texture (set 0,
        // createPipelineTex/dispatchTex -- the P2 probe npTex3d, kept as the record
        // of what the Vulkan sampler does) carries the CUDA SDF sampler
        // configuration: trilinear, clamp-to-edge, element-type reads, no mips.
        // A bindless texture owns NO sampler (P9.97): pxgTex3D fetches texels.
        if (!bindless)
        {
            VkSamplerCreateInfo samplerInfo = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
            samplerInfo.magFilter = VK_FILTER_LINEAR;
            samplerInfo.minFilter = VK_FILTER_LINEAR;
            samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            r = madeOrNull(createSamplerCounted(samplerInfo, &tex.sampler), &tex.sampler);   // not counted
            if (r != VK_SUCCESS)
                return refuseTexture(tex, VK_NULL_HANDLE, VK_NULL_HANDLE, caller, "vkCreateSampler", r);
        }

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
                releaseTexture(tex);
                return 0;
            }
            writeBindlessSlot(uint32_t(handle), tex.view);
        }
        mTextures[handle] = tex;
        return handle;
    }

    PxgTextureHandle refuseTexture(Texture& tex, VkBuffer staging, VkDeviceMemory stagingMem, const char* caller,
                                   const char* what, VkResult r)
    {
        if (r != VK_SUCCESS)
            std::fprintf(stderr, "[PxgVulkan] %s(%u x %u x %u): %s failed with VkResult %d; returning 0\n", caller,
                         tex.width, tex.height, tex.depth, what, int(r));
        else
            std::fprintf(stderr, "[PxgVulkan] %s(%u x %u x %u): %s; returning 0\n", caller, tex.width, tex.height,
                         tex.depth, what);
        if (staging)
            vkDestroyBuffer(mDevice, staging, nullptr);
        if (stagingMem)
            freeMemoryCounted(stagingMem);
        releaseTexture(tex);
        tex = Texture{};
        return 0;
    }

    // ---- the bindless set ----

    bool initBindlessTextures()
    {
        // P9.97: ONE binding.  Binding 1 -- a shared LINEAR/CLAMP sampler for
        // pxgTex3D's SampleLevel since P9.12a -- lost its last reader when P9.96
        // moved pxgTex3D to texel loads, and is gone from the layout, the pool and
        // the device: no VkSampler is created for this set.  createPipelineBindless
        // refuses a module that declares anything but binding 0 here.
        const VkDescriptorSetLayoutBinding bindings[] = {
            { 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kBindlessTextureCapacity, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        };
        const uint32_t numBindings = uint32_t(sizeof(bindings) / sizeof(bindings[0]));

        // PARTIALLY_BOUND: slots for textures that do not exist yet (and slot 0,
        // the reserved null handle) are never written and never read.
        // UPDATE_AFTER_BIND: createTexture3DFmt may run while a previously
        // recorded dispatch still references the set.  One flag word per binding.
        VkDescriptorBindingFlags flags[sizeof(bindings) / sizeof(bindings[0])] = {
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
        };
        VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO
        };
        flagsInfo.bindingCount = numBindings;
        flagsInfo.pBindingFlags = flags;

        VkDescriptorSetLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = numBindings;
        layoutInfo.pBindings = bindings;
        layoutInfo.pNext = &flagsInfo;
        layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        recordBindlessLayout(layoutInfo);
        if (madeOrNull(vkCreateDescriptorSetLayout(mDevice, &layoutInfo, nullptr, &mBindlessSetLayout),
                       &mBindlessSetLayout) != VK_SUCCESS)
            return false;

        const VkDescriptorPoolSize poolSizes[] = {
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kBindlessTextureCapacity },
        };
        VkDescriptorPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = uint32_t(sizeof(poolSizes) / sizeof(poolSizes[0]));
        poolInfo.pPoolSizes = poolSizes;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        recordBindlessPool(poolInfo);
        if (madeOrNull(vkCreateDescriptorPool(mDevice, &poolInfo, nullptr, &mBindlessPool), &mBindlessPool) !=
            VK_SUCCESS)
            return false;

        VkDescriptorSetAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.descriptorPool = mBindlessPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &mBindlessSetLayout;
        return vkAllocateDescriptorSets(mDevice, &allocInfo, &mBindlessSet) == VK_SUCCESS;
    }

    // P9.97: what the driver was handed for the bindless set, for bindlessSetInfo().
    void recordBindlessLayout(const VkDescriptorSetLayoutCreateInfo& li)
    {
        const VkDescriptorSetLayoutBindingFlagsCreateInfo* fi = nullptr;
        for (const VkBaseInStructure* p = static_cast<const VkBaseInStructure*>(li.pNext); p; p = p->pNext)
            if (p->sType == VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO)
                fi = reinterpret_cast<const VkDescriptorSetLayoutBindingFlagsCreateInfo*>(p);
        BindlessSetInfo& s = mBindlessInfo;
        s.layoutFlags = li.flags;
        s.bindingCount = li.bindingCount;
        s.bindingFlagsCount = fi ? fi->bindingCount : 0u;
        for (uint32_t i = 0; i < li.bindingCount && i < BindlessSetInfo::kMax; ++i)
        {
            s.binding[i] = li.pBindings[i].binding;
            s.descriptorType[i] = uint32_t(li.pBindings[i].descriptorType);
            s.descriptorCount[i] = li.pBindings[i].descriptorCount;
            s.stageFlags[i] = li.pBindings[i].stageFlags;
            s.bindingFlags[i] = (fi && i < fi->bindingCount) ? fi->pBindingFlags[i] : 0u;
        }
    }

    void recordBindlessPool(const VkDescriptorPoolCreateInfo& pi)
    {
        BindlessSetInfo& s = mBindlessInfo;
        s.poolFlags = pi.flags;
        s.poolMaxSets = pi.maxSets;
        s.poolSizeCount = pi.poolSizeCount;
        for (uint32_t i = 0; i < pi.poolSizeCount && i < BindlessSetInfo::kMax; ++i)
        {
            s.poolType[i] = uint32_t(pi.pPoolSizes[i].type);
            s.poolCount[i] = pi.pPoolSizes[i].descriptorCount;
        }
    }

    BindlessSetInfo bindlessSetInfo() const override
    {
        return mHasBindlessTextures ? mBindlessInfo : BindlessSetInfo();
    }

    // P9.97: every VkSampler goes through these two, which is what makes
    // liveSamplerCount() an account of the device's samplers.
    VkResult createSamplerCounted(const VkSamplerCreateInfo& info, VkSampler* out)
    {
        const VkResult r = vkCreateSampler(mDevice, &info, nullptr, out);
        if (r == VK_SUCCESS)
            ++mLiveSamplers;
        return r;
    }

    // P9.98 check round 1: every VkDeviceMemory goes through these two, so the device can
    // say how many allocations it holds at once.  The Best Practices waiver for
    // small-allocation (F5) rests on the engine's PEAK staying far below
    // maxMemoryAllocationCount; with PXG_VK_ALLOC_STATS=1 (tools/run_validation.sh sets
    // it) the process prints its peak at exit and tools/vvl_triage.py checks the bound
    // the waiver states.  Process-wide, like SyncStatsReport: the engine's device is not
    // necessarily destroyed before the process ends.
    struct AllocStats
    {
        uint64_t live = 0, peak = 0, maxCount = 0;
        bool report = std::getenv("PXG_VK_ALLOC_STATS") != nullptr;
        ~AllocStats()
        {
            if (report)
                std::fprintf(stderr, "[pxg-alloc] peak %llu live VkDeviceMemory allocation(s) "
                                     "(maxMemoryAllocationCount %llu)\n",
                             (unsigned long long)peak, (unsigned long long)maxCount);
        }
    };
    static AllocStats& allocStats()
    {
        static AllocStats stats;   // destroyed -- and printed -- at exit
        return stats;
    }
    VkResult allocateMemoryCounted(const VkMemoryAllocateInfo* info, VkDeviceMemory* out)
    {
        const VkResult r = vkAllocateMemory(mDevice, info, nullptr, out);
        if (r == VK_SUCCESS)
        {
            AllocStats& a = allocStats();
            a.peak = ++a.live > a.peak ? a.live : a.peak;
        }
        return r;
    }
    void freeMemoryCounted(VkDeviceMemory memory)
    {
        if (memory == VK_NULL_HANDLE)
            return;
        vkFreeMemory(mDevice, memory, nullptr);
        --allocStats().live;
    }

    void destroySamplerCounted(VkSampler sampler)
    {
        if (sampler == VK_NULL_HANDLE)
            return;
        vkDestroySampler(mDevice, sampler, nullptr);
        --mLiveSamplers;
    }

    uint32_t liveSamplerCount() const override { return mLiveSamplers; }

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
        // P9.99: the request is checked in createPipelineInternal, once the entry's workgroup
        // shape is read (VUID-02756 / 02757 need it), and refused through refusePipeline like
        // every P9.98 refusal.  Before P9.99 only the window was checked here.
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
        {
            std::fprintf(stderr, "[PxgVulkan] createPipelineBindless(%s): this device has no bindless texture set "
                                 "(descriptor indexing); refused\n",
                         entryName ? entryName : "?");
            return 0;
        }
        // P9.97: a module the one-binding set cannot hold is refused HERE, because
        // llvmpipe does not refuse it: measured, a SamplerState at (1, 1) builds,
        // then samples through whichever VkSampler the process happens to own --
        // or segfaults at dispatch when it owns none.
        uint32_t badSet = 0, badBinding = 0;
        if (!pxgSpirvFitsBindlessLayout(code, codeSize, &badSet, &badBinding))
        {
            if (badSet == ~0u)
                std::fprintf(stderr, "[PxgVulkan] createPipelineBindless(%s): the module is not a well-formed "
                                     "SPIR-V word stream; refused\n",
                             entryName ? entryName : "?");
            else if (badBinding == ~0u)
                std::fprintf(stderr, "[PxgVulkan] createPipelineBindless(%s): the module declares a set-%u "
                                     "variable without a Binding; refused\n",
                             entryName ? entryName : "?", badSet);
            else
                std::fprintf(stderr,
                             "[PxgVulkan] createPipelineBindless(%s): the module declares set %u binding %u, "
                             "which the bindless layout (set 1 = binding 0 only, P9.97) does not have; refused\n",
                             entryName ? entryName : "?", badSet, badBinding);
            return 0;
        }
        if (pxgSpirvDeclaresSampler(code, codeSize))
        {
            std::fprintf(stderr, "[PxgVulkan] createPipelineBindless(%s): the module declares a sampler; the "
                                 "bindless path fetches texels and never samples in hardware (P9.96); refused\n",
                         entryName ? entryName : "?");
            return 0;
        }
        return createPipelineInternal(code, codeSize, entryName, numBuffers, bufferStrides, pushConstantSize,
                                      0, 0, true);
    }

    SubgroupCaps getSubgroupCaps() const override { return mSubgroupCaps; }

    ComputeLimits computeLimits() const override { return mLimits; }

    uint32_t pipelinePushRangeBytes(PxgPipelineHandle handle) const override
    {
        const auto it = mPipelines.find(handle);
        return it == mPipelines.end() ? ~0u : it->second.layoutPushSize;
    }

    uint64_t getBufferDeviceAddress(PxgBufferHandle handle) const override
    {
        auto it = mBuffers.find(handle);
        return it == mBuffers.end() ? 0 : it->second.deviceAddress;
    }

    // P9.98 (F4): every step's VkResult is checked IN RELEASE BUILDS -- PXG_VK_CHECK only
    // asserts, and both build trees are -DNDEBUG -- and a failure destroys whatever was
    // created and returns 0.  Before P9.98 a failed vkCreateComputePipelines still got a
    // handle, the shim's `pipe == 0 -> NotFound` never fired, and the first dispatch bound
    // VK_NULL_HANDLE and segfaulted (measured: scan with vkCreateComputePipelines forced
    // to fail, rc 139).  The checks against the device limits run BEFORE any Vulkan
    // object exists, so an over-limit kernel is a refusal with a reason rather than a
    // driver error (or, on a driver that does not check, undefined behaviour).
    void nameObject(VkObjectType type, uint64_t handle, const char* name)
    {
        if (!mSetObjectName || !handle || !name)
            return;
        VkDebugUtilsObjectNameInfoEXT info = { VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
        info.objectType = type;
        info.objectHandle = handle;
        info.pObjectName = name;
        mSetObjectName(mDevice, &info);
    }

    PxgPipelineHandle refusePipeline(Pipeline& pipe, const char* entryName, const char* what, int vkResult)
    {
        if (vkResult != 0)
            std::fprintf(stderr, "[PxgVulkan] createPipeline(%s): %s failed with VkResult %d; refused\n",
                         entryName ? entryName : "?", what, vkResult);
        else
            std::fprintf(stderr, "[PxgVulkan] createPipeline(%s): %s; refused\n", entryName ? entryName : "?", what);
        if (pipe.pipeline)
            vkDestroyPipeline(mDevice, pipe.pipeline, nullptr);
        if (pipe.layout)
            vkDestroyPipelineLayout(mDevice, pipe.layout, nullptr);
        if (pipe.setLayout)
            vkDestroyDescriptorSetLayout(mDevice, pipe.setLayout, nullptr);
        if (pipe.module)
            vkDestroyShaderModule(mDevice, pipe.module, nullptr);
        pipe = Pipeline{};
        return 0;
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

        // ---- what the entry point asks of the device, from the SPIR-V (P9.98) ----
        PxgSpirvEntryInfo info;
        bool malformed = false;
        if (!reflectEntryImpl(code, codeSize, entryName, &info, &malformed))
            return refusePipeline(pipe, entryName,
                                  malformed ? "the module is not a well-formed SPIR-V word stream"
                                            : "the module has no GLCompute entry point of that name",
                                  0);
        char why[320];
        // F2: the layout's push range covers the entry's DECLARED block, not just the
        // bytes the caller pushes -- VUID-VkComputePipelineCreateInfo-layout-07987 -- and
        // dispatch zero-fills the surplus.  Shared push structs (one Slang struct sized
        // for the largest of several entries) stay legal and the shaders stay unchanged.
        pipe.layoutPushSize = (pushConstantSize + 3u) & ~3u;
        if (info.pushDecided && info.pushBlockBytes > pipe.layoutPushSize)
        {
            pipe.layoutPushSize = info.pushBlockBytes;
        }
        if (pipe.layoutPushSize > mLimits.maxPushConstantsSize)
        {
            std::snprintf(why, sizeof(why),
                          "push-constant range %u B%s (caller %u B, SPIR-V block %u B) exceeds the device's "
                          "maxPushConstantsSize %u B",
                          pipe.layoutPushSize, pipe.layoutPushSize == 0xFFFFFFFFu ? " (or more)" : "",
                          pushConstantSize, info.pushBlockBytes, mLimits.maxPushConstantsSize);
            return refusePipeline(pipe, entryName, why, 0);
        }
        // Per ENTRY POINT, not per module: Vulkan applies the limit per pipeline
        // (tools/check_groupshared.py's header; F3 is a layer that sums the module).
        if (info.workgroupDecided && info.workgroupBytes > mLimits.maxComputeSharedMemorySize)
        {
            std::snprintf(why, sizeof(why),
                          "Workgroup (groupshared) storage %u B%s exceeds the device's "
                          "maxComputeSharedMemorySize %u B",
                          info.workgroupBytes, info.workgroupBytes == 0xFFFFFFFFu ? " (or more)" : "",
                          mLimits.maxComputeSharedMemorySize);
            return refusePipeline(pipe, entryName, why, 0);
        }
        if (info.localSize[0] && info.localSize[1] && info.localSize[2])
        {
            const uint64_t invocations = uint64_t(info.localSize[0]) * info.localSize[1] * info.localSize[2];
            for (int d = 0; d < 3; ++d)
                if (info.localSize[d] > mLimits.maxComputeWorkGroupSize[d])
                {
                    std::snprintf(why, sizeof(why),
                                  "workgroup size (%u, %u, %u) exceeds the device's maxComputeWorkGroupSize[%d] %u",
                                  info.localSize[0], info.localSize[1], info.localSize[2], d,
                                  mLimits.maxComputeWorkGroupSize[d]);
                    return refusePipeline(pipe, entryName, why, 0);
                }
            if (invocations > mLimits.maxComputeWorkGroupInvocations)
            {
                std::snprintf(why, sizeof(why),
                              "workgroup size (%u, %u, %u) = %llu invocations exceeds the device's "
                              "maxComputeWorkGroupInvocations %u",
                              info.localSize[0], info.localSize[1], info.localSize[2],
                              (unsigned long long)invocations, mLimits.maxComputeWorkGroupInvocations);
                return refusePipeline(pipe, entryName, why, 0);
            }
        }
        // P9.99: an exact subgroup size (createPipelineEx) only when the device can require it
        // AND this entry's shape lets every subgroup be full.  Measured on HEAD 2a97aa5: an
        // in-window 8 on x = 4 / 12 (VUID-02757) or on 264 / 512 invocations (VUID-02756) was
        // BUILT, and llvmpipe ran x = 4 as 8-lane subgroups with 4 active lanes.
        if (requiredSubgroupSize != 0 &&
            !pxgRequiredSubgroupSizeFits(mSubgroupCaps, requiredSubgroupSize, info.localSize, why, sizeof(why)))
            return refusePipeline(pipe, entryName, why, 0);

        VkShaderModuleCreateInfo moduleInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        moduleInfo.codeSize = codeSize;
        moduleInfo.pCode = reinterpret_cast<const uint32_t*>(code);
        VkResult r = madeOrNull(vkCreateShaderModule(mDevice, &moduleInfo, nullptr, &pipe.module), &pipe.module);
        if (r != VK_SUCCESS)
            return refusePipeline(pipe, entryName, "vkCreateShaderModule", int(r));
        nameObject(VK_OBJECT_TYPE_SHADER_MODULE, uint64_t(pipe.module), entryName);

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
        r = madeOrNull(vkCreateDescriptorSetLayout(mDevice, &setLayoutInfo, nullptr, &pipe.setLayout), &pipe.setLayout);
        if (r != VK_SUCCESS)
            return refusePipeline(pipe, entryName, "vkCreateDescriptorSetLayout", int(r));

        VkPushConstantRange pushRange = {};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = pipe.layoutPushSize;

        // Set 0 is the pipeline's own bindings; set 1, when declared, is the
        // device-wide bindless texture set.
        const VkDescriptorSetLayout setLayouts[2] = { pipe.setLayout, mBindlessSetLayout };
        VkPipelineLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = bindless ? 2u : 1u;
        layoutInfo.pSetLayouts = setLayouts;
        layoutInfo.pushConstantRangeCount = pipe.layoutPushSize > 0 ? 1 : 0;
        layoutInfo.pPushConstantRanges = pipe.layoutPushSize > 0 ? &pushRange : nullptr;
        r = madeOrNull(vkCreatePipelineLayout(mDevice, &layoutInfo, nullptr, &pipe.layout), &pipe.layout);
        if (r != VK_SUCCESS)
            return refusePipeline(pipe, entryName, "vkCreatePipelineLayout", int(r));
        nameObject(VK_OBJECT_TYPE_PIPELINE_LAYOUT, uint64_t(pipe.layout), entryName);

        VkPipelineShaderStageCreateInfo stageInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = pipe.module;
        stageInfo.pName = entryName;

        // Exact-subgroup-size request (warp-32 primitive library on hardware
        // targets).  Checked above (pxgRequiredSubgroupSizeFits), which also guarantees
        // computeFullSubgroups, so full subgroups are always requested with it (P9.99).
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sgSizeInfo = {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO
        };
        if (requiredSubgroupSize != 0)
        {
            sgSizeInfo.requiredSubgroupSize = requiredSubgroupSize;
            stageInfo.pNext = &sgSizeInfo;
            stageInfo.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        }

        VkComputePipelineCreateInfo pipelineInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = pipe.layout;
        r = vkCreateComputePipelines(mDevice, mPipelineCache, 1, &pipelineInfo, nullptr, &pipe.pipeline);
        // A driver may return a non-success code and still write a handle (refusePipeline
        // destroys it), or return success and none: both mean "no pipeline" here.
        if (r != VK_SUCCESS)
            return refusePipeline(pipe, entryName, "vkCreateComputePipelines", int(r));
        if (pipe.pipeline == VK_NULL_HANDLE)
            return refusePipeline(pipe, entryName, "vkCreateComputePipelines returned VK_SUCCESS but no pipeline", 0);
        nameObject(VK_OBJECT_TYPE_PIPELINE, uint64_t(pipe.pipeline), entryName);

        // Check round 1: widening the range past the caller's blob (F2, above) makes
        // VUID-07987 unreachable, so a shader that READS past the caller's bytes now reads
        // zeros with no layer message.  Under PXG_VK_DEBUG_NAMES (tools/run_validation.sh)
        // a BUILT pipeline whose range was widened is reported instead, and
        // tools/vvl_triage.py fails on the line unless validation_waivers.txt lists the entry.
        if (mDebugNames && pipe.layoutPushSize > ((pushConstantSize + 3u) & ~3u))
            std::fprintf(stderr, "[PxgVulkan] createPipeline(%s): push block %u B wider than caller %u B; "
                                 "zero-filled\n", entryName ? entryName : "?", info.pushBlockBytes, pushConstantSize);

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

    bool dispatch(PxgPipelineHandle pipelineHandle, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                  const PxgBufferHandle* buffers, uint32_t numBuffers,
                  const void* pushConstants, uint32_t pushConstantSize) override
    {
        return dispatchInternal(pipelineHandle, gridX, gridY, gridZ, buffers, numBuffers, nullptr, 0,
                         pushConstants, pushConstantSize);
    }

    bool dispatchTex(PxgPipelineHandle pipelineHandle, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                     const PxgBufferHandle* buffers, uint32_t numBuffers,
                     const PxgTextureHandle* textures, uint32_t numTextures,
                     const void* pushConstants, uint32_t pushConstantSize) override
    {
        return dispatchInternal(pipelineHandle, gridX, gridY, gridZ, buffers, numBuffers, textures, numTextures,
                         pushConstants, pushConstantSize);
    }

    // P9.98 (F4): the pipeline is looked up with find(), never operator[] -- which used
    // to default-insert a Pipeline of null handles for an unknown id and then bind
    // VK_NULL_HANDLE.  A refusal records nothing and says why; the shim reports it as a
    // refused launch (NotFound).
    const Pipeline* livePipeline(PxgPipelineHandle handle, uint32_t pushConstantSize, const char* caller) const
    {
        const auto it = handle == 0 ? mPipelines.end() : mPipelines.find(handle);
        if (it == mPipelines.end())
        {
            std::fprintf(stderr, "[PxgVulkan] %s: pipeline handle %u is not a live pipeline (refused or never "
                                 "created); nothing recorded\n", caller, unsigned(handle));
            return nullptr;
        }
        if (pushConstantSize > it->second.layoutPushSize)
        {
            std::fprintf(stderr, "[PxgVulkan] %s: a %u-byte push blob does not fit pipeline %u's %u-byte push "
                                 "range; nothing recorded\n", caller, pushConstantSize, unsigned(handle),
                         it->second.layoutPushSize);
            return nullptr;
        }
        return &it->second;
    }

    // F2: push the WHOLE layout range, the caller's bytes followed by zeros, so the
    // block the shader declares is fully defined.  When the caller fills the range
    // exactly (every engine row but three, P9.98 scope) its own pointer is pushed.
    void pushConstantsPadded(const Pipeline& pipe, const void* pushConstants, uint32_t pushConstantSize)
    {
        if (pipe.layoutPushSize == 0)
            return;
        const void* src = pushConstants;
        if (pushConstantSize != pipe.layoutPushSize || !pushConstants)
        {
            mPushScratch.assign(pipe.layoutPushSize, 0u);
            if (pushConstants && pushConstantSize)
                std::memcpy(mPushScratch.data(), pushConstants, pushConstantSize);
            src = mPushScratch.data();
        }
        vkCmdPushConstants(mCommandBuffer, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pipe.layoutPushSize, src);
    }

    // P9.98 check round 0: what the dispatch BINDS is validated as strictly as the
    // pipeline -- the counts must be the pipeline's (they were assert-only in Release)
    // and every buffer / texture handle must be live (they were operator[] lookups,
    // which default-inserted a null object and recorded VK_NULL_HANDLE descriptors: VVL
    // VUID-VkDescriptorBufferInfo-buffer-02998 / vkCmdDispatch-None-08114, and a SIGSEGV
    // for dispatchIndirect's arg buffer).  All of it is resolved before anything is
    // recorded, so a refusal records nothing.
    bool bindingsMatch(const Pipeline& pipe, PxgPipelineHandle handle, uint32_t numBuffers,
                       const PxgBufferHandle* buffers, uint32_t numTextures, const PxgTextureHandle* textures,
                       const char* caller)
    {
        if (numBuffers == pipe.numBuffers && numTextures == pipe.numTextures && (numBuffers == 0 || buffers) &&
            (numTextures == 0 || textures))
            return true;
        std::fprintf(stderr, "[PxgVulkan] %s: pipeline %u binds %u buffer(s) and %u texture(s); the dispatch "
                             "passed %u%s and %u%s; nothing recorded\n", caller, unsigned(handle), pipe.numBuffers,
                     pipe.numTextures, numBuffers, numBuffers && !buffers ? " (null array)" : "", numTextures,
                     numTextures && !textures ? " (null array)" : "");
        return false;
    }

    bool resolveBuffers(const PxgBufferHandle* buffers, uint32_t numBuffers,
                        std::vector<VkDescriptorBufferInfo>& infos, const char* caller)
    {
        infos.assign(numBuffers, VkDescriptorBufferInfo{});
        for (uint32_t i = 0; i < numBuffers; ++i)
        {
            const Buffer* buf = liveBuffer(buffers[i], caller);
            if (!buf)
                return false;
            infos[i].buffer = buf->buffer;
            infos[i].offset = 0;
            infos[i].range = VK_WHOLE_SIZE;
        }
        return true;
    }

    VkDescriptorSet descriptorSetOrRefuse(const Pipeline& pipe, uint32_t numBindings, const char* caller)
    {
        const VkDescriptorSet set = allocateDescriptorSet(pipe.setLayout, numBindings);
        if (set == VK_NULL_HANDLE)
            std::fprintf(stderr, "[PxgVulkan] %s: no descriptor set could be allocated (see the line above); "
                                 "nothing recorded\n", caller);
        return set;
    }

    bool dispatchInternal(PxgPipelineHandle pipelineHandle, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                          const PxgBufferHandle* buffers, uint32_t numBuffers,
                          const PxgTextureHandle* textures, uint32_t numTextures,
                          const void* pushConstants, uint32_t pushConstantSize)
    {
        const Pipeline* live = livePipeline(pipelineHandle, pushConstantSize, "dispatch");
        if (!live)
            return false;
        const Pipeline& pipe = *live;
        assert(pushConstantSize == pipe.pushConstantSize);
        if (!bindingsMatch(pipe, pipelineHandle, numBuffers, buffers, numTextures, textures, "dispatch"))
            return false;
        // P9.98 check round 1: the grid against maxComputeWorkGroupCount
        // (VUID-vkCmdDispatch-groupCountX-00386 / Y-00387 / Z-00388).  CUDA accepts a
        // gridDim.x up to 2^31-1; Vulkan's minimum -- and llvmpipe's value -- is 65535, and
        // a larger grid was recorded as if valid.
        const uint32_t grid[3] = { gridX, gridY, gridZ };
        for (int d = 0; d < 3; ++d)
            if (grid[d] > mLimits.maxComputeWorkGroupCount[d])
            {
                std::fprintf(stderr, "[PxgVulkan] dispatch: grid (%u, %u, %u) exceeds the device's "
                                     "maxComputeWorkGroupCount[%d] %u (pipeline %u); nothing recorded\n",
                             gridX, gridY, gridZ, d, mLimits.maxComputeWorkGroupCount[d], unsigned(pipelineHandle));
                return false;
            }

        std::vector<VkDescriptorBufferInfo> bufferInfos;
        if (!resolveBuffers(buffers, numBuffers, bufferInfos, "dispatch"))
            return false;
        std::vector<VkDescriptorImageInfo> imageInfos(2 * numTextures);
        for (uint32_t t = 0; t < numTextures; ++t)
        {
            const auto it = textures[t] == 0 ? mTextures.end() : mTextures.find(textures[t]);
            // A bindless texture (createTexture3DFmt) owns no sampler, so it cannot fill
            // this path's image + sampler pair.
            if (it == mTextures.end() || it->second.sampler == VK_NULL_HANDLE)
            {
                std::fprintf(stderr, "[PxgVulkan] dispatch: texture handle %u is not a live bound texture "
                                     "(createTexture3D); nothing recorded\n", unsigned(textures[t]));
                return false;
            }
            imageInfos[2 * t].imageView = it->second.view;
            imageInfos[2 * t].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfos[2 * t + 1].sampler = it->second.sampler;
        }

        beginRecording();

        const uint32_t numBindings = numBuffers + 2 * numTextures;
        const VkDescriptorSet set = descriptorSetOrRefuse(pipe, numBindings, "dispatch");
        if (set == VK_NULL_HANDLE)
            return false;

        std::vector<VkWriteDescriptorSet> writes(numBindings);
        for (uint32_t i = 0; i < numBuffers; ++i)
        {
            writes[i] = VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bufferInfos[i];
        }
        for (uint32_t t = 0; t < numTextures; ++t)
        {
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
        pushConstantsPadded(pipe, pushConstants, pushConstantSize);

        noteDispatch();
        vkCmdDispatch(mCommandBuffer, gridX, gridY, gridZ);
        return true;
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

    bool dispatchIndirect(PxgPipelineHandle pipelineHandle, PxgBufferHandle argBuffer, size_t argOffset,
                          const PxgBufferHandle* buffers, uint32_t numBuffers,
                          const void* pushConstants, uint32_t pushConstantSize) override
    {
        const Pipeline* live = livePipeline(pipelineHandle, pushConstantSize, "dispatchIndirect");
        if (!live)
            return false;
        const Pipeline& pipe = *live;
        assert(pushConstantSize == pipe.pushConstantSize);
        // The indirect path binds no textures, so a textured pipeline cannot run here.
        if (!bindingsMatch(pipe, pipelineHandle, numBuffers, buffers, 0, nullptr, "dispatchIndirect"))
            return false;
        std::vector<VkDescriptorBufferInfo> bufferInfos;
        if (!resolveBuffers(buffers, numBuffers, bufferInfos, "dispatchIndirect"))
            return false;
        // vkCmdDispatchIndirect reads a VkDispatchIndirectCommand (3 x uint32) at a
        // 4-byte aligned offset that lies inside the arg buffer.
        const Buffer* arg = liveBuffer(argBuffer, "dispatchIndirect (arg buffer)");
        if (!arg)
            return false;
        if ((argOffset % 4) != 0 || argOffset > arg->size || arg->size - argOffset < 3 * sizeof(uint32_t))
        {
            std::fprintf(stderr, "[PxgVulkan] dispatchIndirect: arg offset %zu is not a 4-byte aligned "
                                 "VkDispatchIndirectCommand inside the %zu-byte arg buffer %u; nothing recorded\n",
                         argOffset, arg->size, unsigned(argBuffer));
            return false;
        }

        beginRecording();

        const VkDescriptorSet set = descriptorSetOrRefuse(pipe, numBuffers, "dispatchIndirect");
        if (set == VK_NULL_HANDLE)
            return false;
        std::vector<VkWriteDescriptorSet> writes(numBuffers);
        for (uint32_t i = 0; i < numBuffers; ++i)
        {
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
        bufferBarrier(arg->buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT);

        vkCmdBindPipeline(mCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
        if (numBuffers > 0)
            vkCmdBindDescriptorSets(mCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout, 0, 1, &set, 0, nullptr);
        if (pipe.bindless)
            vkCmdBindDescriptorSets(mCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout, 1, 1,
                                    &mBindlessSet, 0, nullptr);
        pushConstantsPadded(pipe, pushConstants, pushConstantSize);

        noteDispatch();
        vkCmdDispatchIndirect(mCommandBuffer, arg->buffer, argOffset);
        return true;
    }

    // P9.85 bookkeeping for unorderedDispatchCount().  With PXG_SYNC_STATS=1 the
    // process-wide total is printed at exit -- from a static object, because the
    // engine's device is not necessarily destroyed before the process ends.
    struct SyncStatsReport
    {
        uint64_t unordered = 0, dispatches = 0;
        ~SyncStatsReport()
        {
            std::fprintf(stderr, "[pxg-sync] %llu of %llu dispatch(es) recorded with no compute->compute "
                                 "barrier after the previous dispatch of their batch\n",
                         (unsigned long long)unordered, (unsigned long long)dispatches);
        }
    };
    void noteDispatch()
    {
        static const bool stats = std::getenv("PXG_SYNC_STATS") != nullptr;
        if (stats)
        {
            static SyncStatsReport report;   // destroyed -- and printed -- at exit
            report.dispatches += 1;
            report.unordered += mDispatchSinceBarrier ? 1u : 0u;
        }
        if (mDispatchSinceBarrier)
            ++mUnorderedDispatches;
        mDispatchSinceBarrier = true;
    }

    uint64_t unorderedDispatchCount() const override { return mUnorderedDispatches; }

    void barrier() override
    {
        if (!mRecording)
            return;

        VkMemoryBarrier memBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(mCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);
        mDispatchSinceBarrier = false;
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
        mDispatchSinceBarrier = false;   // the next batch waits on this one
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

    // After VK_ERROR_DEVICE_LOST: print what the driver knows about the fault and map
    // each faulting address onto the allocations (handle, size, and how far inside or
    // past it the address lies).  Diagnostics only; the caller aborts afterwards.
    void reportDeviceFault()
    {
#ifdef VK_EXT_device_fault
        if (!mHasDeviceFault || !mGetDeviceFaultInfo)
        {
            std::fprintf(stderr, "[gpuslang/vulkan] device lost; this driver offers no VK_EXT_device_fault report. "
                                 "Run with PXG_LAUNCH_TRACE=1 PXG_SYNC_EACH=1 to find the last kernel.\n");
            return;
        }
        VkDeviceFaultCountsEXT counts = { VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT };
        if (mGetDeviceFaultInfo(mDevice, &counts, nullptr) != VK_SUCCESS)
        {
            std::fprintf(stderr, "[gpuslang/vulkan] device lost; vkGetDeviceFaultInfoEXT returned no report\n");
            return;
        }
        std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
        std::vector<VkDeviceFaultVendorInfoEXT> vendor(counts.vendorInfoCount);
        VkDeviceFaultInfoEXT info = { VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT };
        info.pAddressInfos = addresses.empty() ? nullptr : addresses.data();
        info.pVendorInfos = vendor.empty() ? nullptr : vendor.data();
        counts.vendorBinarySize = 0;   // the binary dump is not requested
        if (mGetDeviceFaultInfo(mDevice, &counts, &info) != VK_SUCCESS && counts.addressInfoCount == 0)
        {
            std::fprintf(stderr, "[gpuslang/vulkan] device lost; vkGetDeviceFaultInfoEXT failed\n");
            return;
        }
        std::fprintf(stderr, "[gpuslang/vulkan] device fault: %s\n", info.description);
        for (uint32_t i = 0; i < counts.addressInfoCount; ++i)
        {
            const VkDeviceFaultAddressInfoEXT& a = addresses[i];
            const char* kind = "?";
            switch (a.addressType)
            {
            case VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_EXT: kind = "none"; break;
            case VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT: kind = "invalid read"; break;
            case VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT: kind = "invalid write"; break;
            case VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT: kind = "invalid execute"; break;
            case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_EXT: kind = "instruction pointer (unknown)"; break;
            case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_INVALID_EXT: kind = "instruction pointer (invalid)"; break;
            case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT: kind = "instruction pointer (fault)"; break;
            default: break;
            }
            const uint64_t addr = a.reportedAddress;
            std::fprintf(stderr, "[gpuslang/vulkan]   %s at 0x%llx (precision 0x%llx)\n", kind,
                         static_cast<unsigned long long>(addr), static_cast<unsigned long long>(a.addressPrecision));
            describeAddress(addr);
        }
        for (uint32_t i = 0; i < counts.vendorInfoCount; ++i)
            std::fprintf(stderr, "[gpuslang/vulkan]   vendor: %s (code 0x%llx, data 0x%llx)\n", vendor[i].description,
                         static_cast<unsigned long long>(vendor[i].vendorFaultCode),
                         static_cast<unsigned long long>(vendor[i].vendorFaultData));
#else
        std::fprintf(stderr, "[gpuslang/vulkan] device lost (built without VK_EXT_device_fault headers)\n");
#endif
    }

    // Which allocation an address belongs to, or the nearest ones around it.
    void describeAddress(uint64_t addr) const
    {
        const Buffer* below = nullptr;
        const Buffer* above = nullptr;
        PxgBufferHandle belowHandle = 0, aboveHandle = 0;
        for (const auto& kv : mBuffers)
        {
            const Buffer& b = kv.second;
            if (!b.deviceAddress)
                continue;
            if (addr >= b.deviceAddress && addr < b.deviceAddress + b.size)
            {
                std::fprintf(stderr, "[gpuslang/vulkan]     inside buffer %u (base 0x%llx, %zu bytes) at offset %llu\n",
                             unsigned(kv.first), static_cast<unsigned long long>(b.deviceAddress), b.size,
                             static_cast<unsigned long long>(addr - b.deviceAddress));
                return;
            }
            if (b.deviceAddress <= addr && (!below || b.deviceAddress > below->deviceAddress))
            {
                below = &b;
                belowHandle = kv.first;
            }
            if (b.deviceAddress > addr && (!above || b.deviceAddress < above->deviceAddress))
            {
                above = &b;
                aboveHandle = kv.first;
            }
        }
        if (below)
            std::fprintf(stderr, "[gpuslang/vulkan]     %llu bytes past the end of buffer %u (base 0x%llx, %zu bytes)\n",
                         static_cast<unsigned long long>(addr - (below->deviceAddress + below->size)), unsigned(belowHandle),
                         static_cast<unsigned long long>(below->deviceAddress), below->size);
        if (above)
            std::fprintf(stderr, "[gpuslang/vulkan]     %llu bytes before buffer %u (base 0x%llx, %zu bytes)\n",
                         static_cast<unsigned long long>(above->deviceAddress - addr), unsigned(aboveHandle),
                         static_cast<unsigned long long>(above->deviceAddress), above->size);
        if (!below && !above)
            std::fprintf(stderr, "[gpuslang/vulkan]     not near any live buffer\n");
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
        // Owned only by the bound path (createTexture3D); a bindless texture has
        // none (P9.97 -- there is no bindless sampler any more).
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
        // P9.98 F2: the layout's push range, max(pushConstantSize, the entry's SPIR-V
        // block) rounded to 4.  dispatch pushes exactly this many bytes, zero-padded.
        uint32_t layoutPushSize = 0;
    };

    // kNoMemoryType when no type matches (P9.98 check round 0: it returned type 0 with an
    // assert-only complaint, i.e. allocated from whatever type 0 is in Release).
    static const uint32_t kNoMemoryType = ~0u;
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties) const
    {
        for (uint32_t i = 0; i < mMemoryProperties.memoryTypeCount; ++i)
        {
            if ((typeBits & (1u << i)) &&
                (mMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        }
        return kNoMemoryType;
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
        // The staging ring carries every v2 transfer; without it the device cannot work.
        PXG_VK_CHECK(alloc.memoryTypeIndex == kNoMemoryType ? VK_ERROR_FEATURE_NOT_PRESENT : VK_SUCCESS);
        PXG_VK_CHECK(allocateMemoryCounted(&alloc, &f.stagingMemory));
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
        destroySamplerCounted(tex.sampler);
        if (tex.view)
            vkDestroyImageView(mDevice, tex.view, nullptr);
        if (tex.image)
            vkDestroyImage(mDevice, tex.image, nullptr);
        if (tex.memory)
            freeMemoryCounted(tex.memory);
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

    // VK_NULL_HANDLE when no set can be had -- the pool cannot be created, a FRESH pool
    // cannot hold the layout, or any other failure (P9.98 check round 0: `pool` was
    // uninitialised after a failed vkCreateDescriptorPool and pushed anyway, a fresh
    // pool that still reported OUT_OF_POOL_MEMORY looped forever creating pools, and
    // other failures were assert-only).  The dispatch is then refused.
    VkDescriptorSet allocateDescriptorSet(VkDescriptorSetLayout layout, uint32_t numBuffers)
    {
        (void)numBuffers;
        Frame& f = mFrames[mFrameIndex];
        for (;;)
        {
            bool fresh = false;
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

                VkDescriptorPool pool = VK_NULL_HANDLE;
                const VkResult pr = vkCreateDescriptorPool(mDevice, &poolInfo, nullptr, &pool);
                if (pr != VK_SUCCESS || pool == VK_NULL_HANDLE)
                {
                    std::fprintf(stderr, "[PxgVulkan] vkCreateDescriptorPool failed with VkResult %d\n", int(pr));
                    return VK_NULL_HANDLE;
                }
                f.pools.push_back(pool);
                fresh = true;
            }

            VkDescriptorSetAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            allocInfo.descriptorPool = f.pools[f.currentPool];
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &layout;

            VkDescriptorSet set = VK_NULL_HANDLE;
            const VkResult res = vkAllocateDescriptorSets(mDevice, &allocInfo, &set);
            if (res == VK_SUCCESS && set != VK_NULL_HANDLE)
                return set;
            if ((res == VK_ERROR_OUT_OF_POOL_MEMORY || res == VK_ERROR_FRAGMENTED_POOL) && !fresh)
            {
                ++f.currentPool;
                continue;
            }
            std::fprintf(stderr, "[PxgVulkan] vkAllocateDescriptorSets failed with VkResult %d%s\n", int(res),
                         fresh ? " on a fresh pool" : "");
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
    ComputeLimits mLimits;                 // P9.98: checked by createPipelineInternal
    PFN_vkSetDebugUtilsObjectNameEXT mSetObjectName = nullptr;   // P9.98: PXG_VK_DEBUG_NAMES
    bool mDebugNames = false;              // P9.98 check round 1: also reports widened push ranges
    std::vector<uint8_t> mPushScratch;     // P9.98 F2: the zero-padded push copy
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
    bool mDispatchSinceBarrier = false;   // P9.85
    uint64_t mUnorderedDispatches = 0;    // P9.85
    bool mHasShaderInt64 = false;
    bool mHasBufferDeviceAddress = false;
    // VK_EXT_shader_atomic_float / shaderBufferFloat32AtomicAdd.  Kernels that
    // transcribe CUDA's atomicAdd(float*) emit OpAtomicFAddEXT and require it.
    bool mHasShaderBufferFloat32AtomicAdd = false;
    bool mHasDeviceFault = false;
#ifdef VK_EXT_device_fault
    PFN_vkGetDeviceFaultInfoEXT mGetDeviceFaultInfo = nullptr;
#endif
    SubgroupCaps mSubgroupCaps;

    // ---- the bindless texture set (P9.12a; one binding since P9.97) ----
    // One device-wide descriptor set at index 1: an array of 4096 sampled 3D
    // images at binding 0 and nothing else.  Until P9.97 binding 1 held ONE
    // shared sampler; P9.96 made pxgTex3D fetch texels (OpImageFetch, which needs
    // no sampler) and evaluate CUDA's clamp/linear filter itself, so the engine's
    // one texture descriptor -- clamp on all three axes, linear filter,
    // normalized element reads, unnormalized coordinates
    // (PxgGeometryManager.cpp:762-770) -- lives in shader code, and P9.97 removed
    // the binding, its pool size and the sampler.  A texture that wants different
    // addressing means extending pxgTex3D, not this set.
    static const uint32_t kBindlessTextureCapacity = 4096;
    bool mHasBindlessTextures = false;
    VkDescriptorSetLayout mBindlessSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool mBindlessPool = VK_NULL_HANDLE;
    VkDescriptorSet mBindlessSet = VK_NULL_HANDLE;
    BindlessSetInfo mBindlessInfo;   // P9.97: as created, see recordBindlessLayout/Pool
    uint32_t mLiveSamplers = 0;      // P9.97: see createSamplerCounted

    std::string mDeviceName;
};

static void pxgVkReportFatal(VkResult result)
{
    if (result == VK_ERROR_DEVICE_LOST && gFaultReportDevice)
        gFaultReportDevice->reportDeviceFault();
}

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
