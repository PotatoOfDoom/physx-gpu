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

// testShimBindless.cpp -- P9.84: does the ENGINE path sample a bindless texture?
//
// P9.12a built the device-wide bindless texture set (set 1) and certified it with
// gpuslang.npTex3dBindless.  That fixture creates its pipeline with
// `Kernel(Kernel::Bindless{}, ...)`, i.e. through createPipelineBindless -- a door
// the ENGINE never walks through.  PxgCudaShim::pipelineFor, which builds every
// engine pipeline, calls createPipeline (a layout with set 0 only) or
// createPipelineTex, and nothing else.
//
// Meanwhile three MAPPED engine kernels declare set 1 in their SPIR-V --
// cloth_sdfMeshContactGenLaunch, ps_sdfMeshCollisonLaunch and
// sb_sdfMeshMidphaseGeneratePairsLaunch -- and the engine uploads its SDFs INTO
// that set (bridge cuTexObjectCreate -> createTexture3DFmt, bindless).  Until
// P9.97b no scene reached any of the three with a real SDF, so nothing had run
// the combination: an engine-built pipeline sampling an engine-uploaded texture.
// PXG_SCENE=softbody PXG_SB_FLOOR=sdfmesh now runs it through
// sb_sdfMeshMidphaseGeneratePairsLaunch; this fixture still pins it directly.
//
// This is that combination, and nothing else.  The entry is npTex3dBindless's
// own (fully BDA, handle read from memory), so the shader side is already
// certified; the manifest row carries numTextures = 0 exactly as every engine row
// does; the texture goes through the shim's arrayCreate3D -> arrayUpload ->
// textureObjectCreate, which is the engine's upload sequence
// (PxgGeometryManager.cpp:732-770).  Texel CENTRES only, so the expected value is
// the stored value bit-for-bit with no filtering tolerance to hide behind.
//
// MEASURED BEFORE THE FIX: the process SEGFAULTS inside vkCreateComputePipelines
// (gdb: PxgCudaShim::launchKernel -> pipelineFor -> createPipeline -> libvulkan_lvp).
// A set-1 shader against a set-0-only layout is not "wrong samples" on llvmpipe; it
// is a dead process on the first launch of any of the three kernels.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"
#include "PxgSlangKernelManifest.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgResult;
using physx::gpuslang::pxgSpirvEntryUsesDescriptorSet;

static std::vector<uint8_t> shimBindlessLoader(const char* file) { return loadShaderBlob(file); }

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// A minimal SPIR-V word stream -- header, ONE OpEntryPoint, and OpDecorate
// instructions {target, decoration, literal} -- for the classifier only; it is
// never handed to a driver.
static std::vector<uint8_t> synthSpv(uint32_t version, const char* name, std::vector<uint32_t> iface,
                                     std::vector<std::vector<uint32_t>> decorations)
{
    std::vector<uint32_t> w = { 0x07230203u, version, 0u, 100u, 0u };
    const size_t len = std::strlen(name);
    std::vector<uint32_t> nameWords((len + 4) / 4, 0u);
    std::memcpy(nameWords.data(), name, len);
    const uint32_t wc = uint32_t(3 + nameWords.size() + iface.size());
    w.push_back((wc << 16) | 15u);   // OpEntryPoint
    w.push_back(5u);                 // GLCompute
    w.push_back(1u);                 // the function id
    w.insert(w.end(), nameWords.begin(), nameWords.end());
    w.insert(w.end(), iface.begin(), iface.end());
    for (const auto& d : decorations)
    {
        w.push_back((4u << 16) | 71u);   // OpDecorate
        w.insert(w.end(), d.begin(), d.end());
    }
    std::vector<uint8_t> b(w.size() * 4);
    std::memcpy(b.data(), w.data(), b.size());
    return b;
}

struct ShimTxbParams
{
    uint64_t handles, coords, outValue, outSize;
    uint32_t numSamples, pad;
};
static_assert(sizeof(ShimTxbParams) == 40, "push blob size");

} // namespace

PXG_TEST(shimBindless_enginePath)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }

    // An ENGINE-shaped manifest row: numTextures = 0, the way every row the
    // manifest generator emits looks.  Nothing here tells the shim the kernel
    // samples -- which is exactly the engine's situation.
    const PxgKernelDef manifest[] = {
        { "npTex3dBindless.spv", "tex3dBindlessSample", uint32_t(sizeof(ShimTxbParams)), 0u,
          "tex3dBindlessSample" },
    };
    PxgCudaShim shim(dev, manifest, 1, &shimBindlessLoader);

    // The texture, through the engine's three-call upload.  Extents distinct and
    // not powers of two; values distinct per texel.
    const uint32_t W = 7, H = 5, D = 3;
    std::vector<float> texels(size_t(W) * H * D);
    for (size_t i = 0; i < texels.size(); ++i)
        texels[i] = 1.5f + 0.25f * float(i) - (i % 3 == 0 ? 7.0f : 0.0f);

    uint32_t arr = 0;
    PXG_CHECK_MSG(shim.arrayCreate3D(&arr, W, H, D, PxgSlangComputeDevice::TextureFormat::R32Float)
                      == PxgResult::Success, "arrayCreate3D failed");
    PXG_CHECK_MSG(shim.arrayUpload(arr, texels.data(), texels.size() * sizeof(float)) == PxgResult::Success,
                  "arrayUpload failed");
    PxgTextureHandle tex = 0;
    PXG_CHECK_MSG(shim.textureObjectCreate(&tex, arr) == PxgResult::Success && tex != 0,
                  "textureObjectCreate failed or returned the reserved null handle");

    // One sample per texel centre.
    const uint32_t N = W * H * D;
    std::vector<uint32_t> handles(N, uint32_t(tex));
    std::vector<float> coords(size_t(N) * 4);
    for (uint32_t z = 0; z < D; ++z)
        for (uint32_t y = 0; y < H; ++y)
            for (uint32_t x = 0; x < W; ++x)
            {
                const size_t i = (size_t(z) * H + y) * W + x;
                coords[i * 4 + 0] = float(x) + 0.5f;
                coords[i * 4 + 1] = float(y) + 0.5f;
                coords[i * 4 + 2] = float(z) + 0.5f;
                coords[i * 4 + 3] = 0.0f;
            }

    PxgDevicePtr dHandles = 0, dCoords = 0, dOut = 0, dSize = 0;
    shim.memAlloc(&dHandles, handles.size() * 4);
    shim.memAlloc(&dCoords, coords.size() * 4);
    shim.memAlloc(&dOut, size_t(N) * 4);
    shim.memAlloc(&dSize, size_t(N) * 16);
    shim.memcpyHtoD(dHandles, handles.data(), handles.size() * 4);
    shim.memcpyHtoD(dCoords, coords.data(), coords.size() * 4);
    // Poison, so "the kernel never ran" and "it ran and read zeros" are told apart.
    shim.memsetD32(dOut, 0xDEADBEEFu, N);
    shim.memsetD32(dSize, 0xDEADBEEFu, size_t(N) * 4);

    ShimTxbParams prm{ dHandles, dCoords, dOut, dSize, N, 0u };
    const PxgResult lr = shim.launchKernel(0, (N + 63) / 64, 1, 1, &prm, sizeof(prm));
    PXG_CHECK_MSG(lr == PxgResult::Success,
                  "shimBindless: launchKernel returned %d -- the engine path could not even build "
                  "a pipeline for a kernel that samples set 1", int(lr));
    shim.streamSynchronize();

    std::vector<float> got(N);
    std::vector<uint32_t> size(size_t(N) * 4);
    shim.memcpyDtoH(got.data(), dOut, size_t(N) * 4);
    shim.memcpyDtoH(size.data(), dSize, size_t(N) * 16);

    size_t mmValue = 0, mmSize = 0, poisoned = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (f2u(got[i]) == 0xDEADBEEFu) ++poisoned;
        if (f2u(got[i]) != f2u(texels[i]))
        {
            if (mmValue < 5)
                std::printf("  sample %u: got %.9g (0x%08x) want %.9g\n", i, got[i], f2u(got[i]), texels[i]);
            ++mmValue;
        }
        const uint32_t* s = &size[size_t(i) * 4];
        if (s[0] != W || s[1] != H || s[2] != D || s[3] != uint32_t(tex))
        {
            if (mmSize < 3)
                std::printf("  sample %u: extent got (%u,%u,%u,h=%u) want (%u,%u,%u,h=%u)\n",
                            i, s[0], s[1], s[2], s[3], W, H, D, unsigned(tex));
            ++mmSize;
        }
    }

    PXG_CHECK_MSG(mmValue == 0 && mmSize == 0,
                  "shimBindless: through the ENGINE path, %zu of %u texel-centre samples and %zu of "
                  "%u extents are wrong (%zu still hold the poison, i.e. were never written)",
                  mmValue, N, mmSize, N, poisoned);

    std::printf("  shimBindless_enginePath: %u texel centres sampled through PxgCudaShim; "
                "%zu value / %zu extent mismatches, %zu untouched\n",
                N, mmValue, mmSize, poisoned);

    for (PxgDevicePtr p : { dHandles, dCoords, dOut, dSize }) shim.memFree(p);
    shim.textureObjectDestroy(tex);
    shim.arrayDestroy(arr);
}

// ---------------------------------------------------------------------------
// The classification pipelineFor now relies on, checked on REAL blobs.
// ---------------------------------------------------------------------------
// shimBindless_enginePath shows the door works for a kernel that samples.  This
// one shows the door is chosen for EXACTLY the kernels that sample: over every
// mapped row of the engine manifest, the entries that use set 1 must be the five
// SDF kernels and nothing else.  P9.83b-8 mapped triangleTriangleCollision (it
// samples both meshes' SDFs) and the list grew by one, on purpose; P9.95 mapped
// evaluatePointDistancesSDFBatch (PxDirectGPUAPI::evaluateSDFDistances samples the
// queried mesh's SDF) and it grew by one again.
PXG_TEST(shimBindless_classify)
{
    using physx::gpuslang::gPxgSlangKernelManifest;
    using physx::gpuslang::PXG_SLANG_KERNEL_COUNT;

    std::map<std::string, std::vector<uint8_t>> blobs;
    auto blobOf = [&](const char* file) -> const std::vector<uint8_t>& {
        auto it = blobs.find(file);
        if (it == blobs.end())
            it = blobs.emplace(file, loadShaderBlob(file)).first;
        return it->second;
    };

    std::set<std::string> sampling;
    size_t mapped = 0, missing = 0;
    for (unsigned i = 0; i < PXG_SLANG_KERNEL_COUNT; ++i)
    {
        const auto& row = gPxgSlangKernelManifest[i];
        if (!row.spirvFile || !row.entry)
            continue;
        ++mapped;
        const std::vector<uint8_t>& b = blobOf(row.spirvFile);
        if (b.empty())
        {
            ++missing;
            continue;
        }
        if (pxgSpirvEntryUsesDescriptorSet(b, row.entry, 1u))
            sampling.insert(row.cudaName);
    }
    const std::set<std::string> expected = { "cloth_sdfMeshContactGenLaunch", "ps_sdfMeshCollisonLaunch",
                                             "sb_sdfMeshMidphaseGeneratePairsLaunch", "triangleTriangleCollision",
                                             "evaluatePointDistancesSDFBatch" };
    PXG_CHECK_MSG(missing == 0, "shimBindless_classify: %zu mapped rows have no loadable .spv", missing);
    for (const std::string& n : sampling)
        if (!expected.count(n))
            std::printf("  classified as sampling, not expected: %s\n", n.c_str());
    for (const std::string& n : expected)
        if (!sampling.count(n))
            std::printf("  expected to sample, not classified: %s\n", n.c_str());
    PXG_CHECK_MSG(sampling == expected,
                  "shimBindless_classify: %zu engine entries classified as using set 1, expected the 5 "
                  "SDF kernels exactly", sampling.size());

    // PER ENTRY, not per module: two harness modules mix sampling and
    // non-sampling entries.  Measured with a SPIR-V scan when P9.84 was written.
    struct Case { const char* file; const char* entry; uint32_t set; bool uses; };
    const Case cases[] = {
        { "npTex3dBindless.spv", "tex3dBindlessSample", 1u, true },
        { "npSdfTexture.spv", "sdfTexSampleAt", 1u, true },
        { "npSdfTexture.spv", "sdfTexProbe", 1u, false },
        { "npSdfCollision.spv", "sdfCollDistance", 1u, true },
        { "npSdfCollision.spv", "sdfCollTetrahedron", 1u, true },
        { "npSdfCollision.spv", "sdfCollBaryClamp", 1u, false },
        { "npSdfCollision.spv", "sdfCollTetClamp", 1u, false },
        { "npTriTriBlock.spv", "ttbProbe", 1u, false },         // a module with no descriptor set at all
        // The SET number is what is asked, not "any descriptor": npTex3d's
        // createPipelineTex-style texture lives in set 0, and its gOut buffer carries
        // Binding 1 (the sampler is at binding 3), so a test that read the wrong
        // decoration word would say yes.
        { "npTex3d.spv", "tex3dSample", 1u, false },
        { "npTex3d.spv", "tex3dSample", 0u, true },
        { "npTex3dBindless.spv", "tex3dBindlessSample", 0u, false },
        // An entry the module does not have: nothing to read per entry, so the
        // answer is the conservative module-wide one.
        { "npSdfTexture.spv", "noSuchEntry", 1u, true },
        { "npTriTriBlock.spv", "noSuchEntry", 1u, false },
        // ...and a stored name that is a PREFIX of the query is not a match:
        // "sdfTexProbe" must not answer for "sdfTexProbeX".
        { "npSdfTexture.spv", "sdfTexProbeX", 1u, true },
    };
    size_t bad = 0;
    for (const Case& c : cases)
    {
        const bool got = pxgSpirvEntryUsesDescriptorSet(blobOf(c.file), c.entry, c.set);
        if (got != c.uses)
        {
            std::printf("  %s:%s set %u classified %d, want %d\n", c.file, c.entry, c.set, int(got), int(c.uses));
            ++bad;
        }
    }
    PXG_CHECK_MSG(bad == 0, "shimBindless_classify: %zu per-entry cases wrong", bad);

    // Below SPIR-V 1.4 an interface list names only Input/Output variables, so
    // the per-entry answer is unavailable and the module-wide one must be used:
    // the same blob, with only its version word lowered to 1.3, must classify the
    // non-sampling sdfTexProbe as USING the set.
    // 1.4 itself is the first version with the complete list, so the same blob
    // at exactly 1.4 still gets the per-entry answer.
    std::vector<uint8_t> v13 = blobOf("npSdfTexture.spv");
    PXG_CHECK_MSG(v13.size() >= 20, "npSdfTexture.spv did not load");
    if (v13.size() >= 20)
    {
        std::vector<uint8_t> v14 = v13;
        const uint32_t ver13 = 0x00010300u, ver14 = 0x00010400u;
        std::memcpy(&v13[4], &ver13, 4);
        std::memcpy(&v14[4], &ver14, 4);
        PXG_CHECK_MSG(pxgSpirvEntryUsesDescriptorSet(v13, "sdfTexProbe", 1u),
                      "shimBindless_classify: a SPIR-V 1.3 blob must fall back to the module-wide answer");
        PXG_CHECK_MSG(!pxgSpirvEntryUsesDescriptorSet(v14, "sdfTexProbe", 1u),
                      "shimBindless_classify: a SPIR-V 1.4 blob already lists every global; per entry");
    }

    // Hand-built blobs for the two ends of the interface list, which no compiled
    // module pins down: every one here has the set-1 variable as the ONLY or the
    // LAST interface id, behind a name whose length is 3 mod 4 (the terminator
    // shares the last name word, so starting one word late skips an id).
    {
        const auto t1 = synthSpv(0x00010500u, "abc", { 7u }, { { 7u, 34u, 1u } });
        const auto t2 = synthSpv(0x00010500u, "abcdefg", { 3u, 5u, 7u }, { { 7u, 34u, 1u } });
        const auto t3 = synthSpv(0x00010500u, "abc", { 3u }, { { 7u, 34u, 1u } });
        // A Binding (33) of 1 and a DescriptorSet of 0: only the second is the set.
        const auto t4 = synthSpv(0x00010500u, "abc", { 7u }, { { 7u, 33u, 1u }, { 7u, 34u, 0u } });
        PXG_CHECK_MSG(pxgSpirvEntryUsesDescriptorSet(t1, "abc", 1u),
                      "shimBindless_classify: synthetic -- the only interface id is the set-1 variable");
        PXG_CHECK_MSG(pxgSpirvEntryUsesDescriptorSet(t2, "abcdefg", 1u),
                      "shimBindless_classify: synthetic -- the set-1 variable is the LAST interface id");
        PXG_CHECK_MSG(!pxgSpirvEntryUsesDescriptorSet(t3, "abc", 1u),
                      "shimBindless_classify: synthetic -- a set-1 variable NOT in the entry's interface");
        PXG_CHECK_MSG(!pxgSpirvEntryUsesDescriptorSet(t4, "abc", 1u),
                      "shimBindless_classify: synthetic -- Binding 1 is not DescriptorSet 1");
    }

    // Malformed input is "no", never a read past the end.
    std::vector<uint8_t> trunc(blobOf("npTex3dBindless.spv").begin(), blobOf("npTex3dBindless.spv").begin() + 12);
    std::vector<uint8_t> badMagic = blobOf("npTex3dBindless.spv");
    badMagic[0] ^= 0xFFu;
    std::vector<uint8_t> odd = blobOf("npTex3dBindless.spv");
    odd.push_back(0);
    PXG_CHECK_MSG(!pxgSpirvEntryUsesDescriptorSet(trunc, "tex3dBindlessSample", 1u) &&
                  !pxgSpirvEntryUsesDescriptorSet(badMagic, "tex3dBindlessSample", 1u) &&
                  !pxgSpirvEntryUsesDescriptorSet(odd, "tex3dBindlessSample", 1u) &&
                  !pxgSpirvEntryUsesDescriptorSet(blobOf("npTex3dBindless.spv"), nullptr, 1u),
                  "shimBindless_classify: a truncated, wrong-magic or misaligned blob (or a null entry) "
                  "must classify as not sampling");

    std::printf("  shimBindless_classify: %zu mapped engine rows over %zu modules; %zu use set 1 "
                "(the 5 SDF kernels); %zu per-entry cases\n",
                mapped, blobs.size(), sampling.size(), sizeof(cases) / sizeof(cases[0]));
}

// ---------------------------------------------------------------------------
// P9.97: the bindless set has ONE binding, and no VkSampler stands behind it.
// ---------------------------------------------------------------------------
// P9.12a's set carried a shared LINEAR/CLAMP sampler at binding 1.  P9.96 moved
// pxgTex3D to texel loads and no module has declared that sampler since (scan of
// all 471 .spv); P9.97 removed it from the layout, the pool and the device.  Three
// things are pinned here, each against LITERAL values (raw Vulkan enum numbers, so
// an edit to a shared header cannot move both sides):
//   1. the set as handed to the driver: ONE binding {0, SAMPLED_IMAGE, 4096,
//      COMPUTE, PARTIALLY_BOUND|UPDATE_AFTER_BIND}, layout flag
//      UPDATE_AFTER_BIND_POOL, ONE pool size {SAMPLED_IMAGE, 4096}, maxSets 1, pool
//      flag UPDATE_AFTER_BIND -- and no SAMPLER anywhere;
//   2. sampler ownership, on a PRIVATE device so the count is absolute: none after
//      init, none for a bindless texture, exactly one for a set-0 createTexture3D
//      texture (the P2 probe's path, which still samples in hardware), none after
//      it is destroyed;
//   3. the door refuses a module the layout cannot hold.  MEASURED (P9.97 scratch
//      probe) why that is not left to the driver: llvmpipe builds a pipeline whose
//      module declares a SamplerState at (1, 1) against the one-binding layout, then
//      samples through whichever VkSampler the process happens to own -- or
//      segfaults at dispatch when it owns none.  The refused modules here are
//      npTex3dBindless.spv with its image array's Binding rewritten 0 -> 1 (the old
//      sampler slot) or its DescriptorSet 1 -> 2.  They are never dispatched.

// npTex3dBindless.spv with ONE decoration of the variable decorated DescriptorSet 1
// rewritten: its `decoration` (33 Binding / 34 DescriptorSet) gets the literal
// `value`, and -- when newKind != decoration -- becomes decoration `newKind` (35
// Offset: the variable then has no Binding at all).  Empty if the blob does not
// have exactly one such variable and one such decoration on it.
static std::vector<uint8_t> repatchSetOneVar(std::vector<uint8_t> b, uint32_t decoration, uint32_t value,
                                             uint32_t newKind = ~0u)
{
    if (b.size() < 20 || (b.size() & 3u) != 0)
        return {};
    std::vector<uint32_t> w(b.size() / 4);
    std::memcpy(w.data(), b.data(), b.size());
    uint32_t target = 0, targets = 0;
    for (size_t i = 5; i < w.size();)
    {
        const uint32_t wc = w[i] >> 16, op = w[i] & 0xFFFFu;
        if (wc == 0 || i + wc > w.size())
            return {};
        if (op == 71u && wc >= 4u && w[i + 2] == 34u && w[i + 3] == 1u)
        {
            target = w[i + 1];
            ++targets;
        }
        i += wc;
    }
    size_t hits = 0;
    for (size_t i = 5; i < w.size(); i += w[i] >> 16)
        if ((w[i] & 0xFFFFu) == 71u && (w[i] >> 16) >= 4u && w[i + 1] == target && w[i + 2] == decoration)
        {
            w[i + 3] = value;
            if (newKind != ~0u)
                w[i + 2] = newKind;
            ++hits;
        }
    if (targets != 1 || hits != 1)
        return {};
    std::memcpy(b.data(), w.data(), b.size());
    return b;
}

PXG_TEST(shimBindless_setLayout)
{
    using physx::gpuslang::pxgSpirvFitsBindlessLayout;
    using physx::gpuslang::pxgSpirvDeclaresSampler;
    PxgSlangComputeDevice* shared = testDevice();
    const bool llvmpipe = std::strstr(shared->getDeviceName(), "llvmpipe") != nullptr;
    if (!shared->supportsBindlessTextures())
    {
        // llvmpipe has all five descriptor-indexing features (measured since P9.12a):
        // there a missing set is a FAILURE -- e.g. a layout its pool cannot allocate
        // -- never a skip that would silence every bindless fixture.
        PXG_CHECK_MSG(!llvmpipe, "shimBindless_setLayout: llvmpipe has no bindless set");
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }

    // ---- 1. the set as created ----
    const PxgSlangComputeDevice::BindlessSetInfo s = shared->bindlessSetInfo();
    PXG_CHECK_MSG(s.bindingCount == 1u && s.bindingFlagsCount == 1u,
                  "shimBindless_setLayout: %u binding(s), %u flag word(s); want exactly 1 (the image array)",
                  s.bindingCount, s.bindingFlagsCount);
    PXG_CHECK_MSG(s.binding[0] == 0u && s.descriptorType[0] == 2u /* SAMPLED_IMAGE */ &&
                  s.descriptorCount[0] == 4096u && s.stageFlags[0] == 0x20u /* COMPUTE */ &&
                  s.bindingFlags[0] == 0x5u /* PARTIALLY_BOUND | UPDATE_AFTER_BIND */,
                  "shimBindless_setLayout: binding {%u, type %u, n %u, stages 0x%x, flags 0x%x}; want "
                  "{0, 2, 4096, 0x20, 0x5}", s.binding[0], s.descriptorType[0], s.descriptorCount[0],
                  s.stageFlags[0], s.bindingFlags[0]);
    PXG_CHECK_MSG(s.layoutFlags == 0x2u /* UPDATE_AFTER_BIND_POOL */,
                  "shimBindless_setLayout: layout flags 0x%x, want 0x2", s.layoutFlags);
    PXG_CHECK_MSG(s.poolSizeCount == 1u && s.poolType[0] == 2u && s.poolCount[0] == 4096u &&
                  s.poolMaxSets == 1u && s.poolFlags == 0x2u /* UPDATE_AFTER_BIND */,
                  "shimBindless_setLayout: pool {%u size(s), first {type %u, n %u}, maxSets %u, flags 0x%x}; "
                  "want {1, {2, 4096}, 1, 0x2}", s.poolSizeCount, s.poolType[0], s.poolCount[0],
                  s.poolMaxSets, s.poolFlags);
    size_t samplerSlots = 0;
    for (uint32_t i = 0; i < s.bindingCount && i < PxgSlangComputeDevice::BindlessSetInfo::kMax; ++i)
        samplerSlots += s.descriptorType[i] == 0u /* SAMPLER */ ? 1 : 0;
    for (uint32_t i = 0; i < s.poolSizeCount && i < PxgSlangComputeDevice::BindlessSetInfo::kMax; ++i)
        samplerSlots += s.poolType[i] == 0u ? 1 : 0;
    PXG_CHECK_MSG(samplerSlots == 0, "shimBindless_setLayout: %zu SAMPLER entries in the bindless layout/pool",
                  samplerSlots);

    // ---- 2. sampler ownership, on a private device ----
    {
        PxgSlangDeviceDesc desc;
        const char* preferSw = std::getenv("PXG_PREFER_SOFTWARE_DEVICE");
        desc.preferSoftwareDevice = preferSw == nullptr || std::strcmp(preferSw, "0") != 0;
        desc.useDeviceLocalMemory = physx::gpuslang::pxgDeviceV2FromEnv();
        std::unique_ptr<PxgSlangComputeDevice> dev(physx::gpuslang::PxgCreateVulkanComputeDevice(desc));
        PXG_CHECK_MSG(dev && dev->supportsBindlessTextures(),
                      "shimBindless_setLayout: a second device with a bindless set could not be created");
        if (dev && dev->supportsBindlessTextures())
        {
            const float texels[8] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };
            const uint32_t c0 = dev->liveSamplerCount();
            const PxgTextureHandle tb = dev->createTexture3DFmt(
                2, 2, 2, PxgSlangComputeDevice::TextureFormat::R32Float, texels);
            const uint32_t c1 = dev->liveSamplerCount();
            const PxgTextureHandle tp = dev->createTexture3D(2, 2, 2, texels);
            const uint32_t c2 = dev->liveSamplerCount();
            dev->destroyTexture3D(tp);
            const uint32_t c3 = dev->liveSamplerCount();
            dev->destroyTexture3D(tb);
            const uint32_t c4 = dev->liveSamplerCount();
            PXG_CHECK_MSG(tb != 0 && tp != 0, "shimBindless_setLayout: texture creation failed (%u, %u)",
                          unsigned(tb), unsigned(tp));
            PXG_CHECK_MSG(c0 == 0u && c1 == 0u && c2 == 1u && c3 == 0u && c4 == 0u,
                          "shimBindless_setLayout: live samplers init/+bindless/+bound/-bound/-bindless = "
                          "%u/%u/%u/%u/%u; want 0/0/1/0/0", c0, c1, c2, c3, c4);
        }
    }

    // ---- 3. the door ----
    // Every module the build has fits the LAYOUT check: the manifest's .spv plus the
    // fixture modules that declare set 1, and npTex3d (its sampler lives in set 0, which
    // the layout check does not look at -- the sampler check below refuses it).
    using physx::gpuslang::gPxgSlangKernelManifest;
    using physx::gpuslang::PXG_SLANG_KERNEL_COUNT;
    std::set<std::string> files = { "npTex3dBindless.spv", "npSdfTexture.spv", "npSdfCollision.spv",
                                    "npSdfFind.spv", "npSdfParticle.spv", "npTex3d.spv" };
    for (unsigned i = 0; i < PXG_SLANG_KERNEL_COUNT; ++i)
        if (gPxgSlangKernelManifest[i].spirvFile)
            files.insert(gPxgSlangKernelManifest[i].spirvFile);
    size_t refusedReal = 0, unloadable = 0;
    for (const std::string& f : files)
    {
        const std::vector<uint8_t> b = loadShaderBlob(f.c_str());
        if (b.empty())
        {
            ++unloadable;
            continue;
        }
        uint32_t bs = 0, bb = 0;
        if (!pxgSpirvFitsBindlessLayout(b.data(), b.size(), &bs, &bb))
        {
            std::printf("  refused a real module: %s (set %u binding %u)\n", f.c_str(), bs, bb);
            ++refusedReal;
        }
    }
    PXG_CHECK_MSG(unloadable == 0 && refusedReal == 0,
                  "shimBindless_setLayout: %zu of %zu modules refused, %zu not loadable", refusedReal,
                  files.size(), unloadable);

    const std::vector<uint8_t> real = loadShaderBlob("npTex3dBindless.spv");
    const std::vector<uint8_t> atB1 = repatchSetOneVar(real, 33u, 1u);   // Binding 0 -> 1
    const std::vector<uint8_t> atS2 = repatchSetOneVar(real, 34u, 2u);   // DescriptorSet 1 -> 2
    const std::vector<uint8_t> noBnd = repatchSetOneVar(real, 33u, 0u, 35u); // no Binding decoration
    PXG_CHECK_MSG(!atB1.empty() && !atS2.empty() && !noBnd.empty(),
                  "shimBindless_setLayout: could not build the patched blobs");
    uint32_t badSetB1 = 0, badBindingB1 = 0, badSetS2 = 0, badBindingS2 = 0;
    const bool fitsB1 = pxgSpirvFitsBindlessLayout(atB1.data(), atB1.size(), &badSetB1, &badBindingB1);
    const bool fitsS2 = pxgSpirvFitsBindlessLayout(atS2.data(), atS2.size(), &badSetS2, &badBindingS2);
    PXG_CHECK_MSG(!fitsB1 && badSetB1 == 1u && badBindingB1 == 1u,
                  "shimBindless_setLayout: image array at (1, 1) -> fits %d, reported (%u, %u); want refused "
                  "at (1, 1)", int(fitsB1), badSetB1, badBindingB1);
    PXG_CHECK_MSG(!fitsS2 && badSetS2 == 2u && badBindingS2 == 0u,
                  "shimBindless_setLayout: image array at (2, 0) -> fits %d, reported (%u, %u); want refused "
                  "at (2, 0)", int(fitsS2), badSetS2, badBindingS2);

    // A set-1 variable with no Binding is not "binding 0" (Vulkan requires the
    // decoration; the guard does not guess).
    uint32_t badSetNB = 0, badBindingNB = 0;
    const bool fitsNB = pxgSpirvFitsBindlessLayout(noBnd.data(), noBnd.size(), &badSetNB, &badBindingNB);
    PXG_CHECK_MSG(!fitsNB && badSetNB == 1u && badBindingNB == ~0u,
                  "shimBindless_setLayout: set-1 variable without Binding -> fits %d, reported (%u, 0x%x)",
                  int(fitsNB), badSetNB, badBindingNB);

    // Malformed word streams do not fit: an instruction running past the end, a wrong magic.
    const uint32_t overrun[6] = { 0x07230203u, 0x00010500u, 0u, 100u, 0u, (10u << 16) | 71u };
    std::vector<uint8_t> badMagic = real;
    if (!badMagic.empty())
        badMagic[0] ^= 0xFFu;
    PXG_CHECK_MSG(!pxgSpirvFitsBindlessLayout(overrun, sizeof(overrun), nullptr, nullptr) &&
                  !pxgSpirvFitsBindlessLayout(badMagic.data(), badMagic.size(), nullptr, nullptr),
                  "shimBindless_setLayout: a malformed blob must not fit");
    // ...nor does a blob whose size is not a whole number of words (copying it word-wise
    // would read past its end), nor one shorter than the 5-word header even with a valid
    // magic.  Both report (~0u, ~0u): "not a word stream", not a set/binding.
    std::vector<uint8_t> plusOne = real;
    plusOne.push_back(0u);
    const std::vector<uint8_t> head16(real.begin(), real.begin() + (real.size() >= 16 ? 16 : real.size()));
    uint32_t bsP1 = 0, bbP1 = 0, bs16 = 0, bb16 = 0;
    const bool fitsP1 = pxgSpirvFitsBindlessLayout(plusOne.data(), plusOne.size(), &bsP1, &bbP1);
    const bool fits16 = pxgSpirvFitsBindlessLayout(head16.data(), head16.size(), &bs16, &bb16);
    PXG_CHECK_MSG(!fitsP1 && bsP1 == ~0u && bbP1 == ~0u && !fits16 && bs16 == ~0u && bb16 == ~0u,
                  "shimBindless_setLayout: real+1 byte -> fits %d (%u, %u); first 16 bytes -> fits %d (%u, %u); "
                  "want both refused as (~0u, ~0u)", int(fitsP1), bsP1, bbP1, int(fits16), bs16, bb16);

    // ---- 3c. no hardware sampling through this door (the P9.96 policy) ----
    // Among every module the build has, exactly npTex3d.spv declares a sampler type --
    // the set-0 P2 probe, which goes through createPipelineTex, never through here.
    size_t declSampler = 0;
    bool tex3dDeclares = false;
    for (const std::string& f : files)
    {
        const std::vector<uint8_t> b = loadShaderBlob(f.c_str());
        if (pxgSpirvDeclaresSampler(b.data(), b.size()))
        {
            ++declSampler;
            if (f == "npTex3d.spv")
                tex3dDeclares = true;
            else
                std::printf("  declares a sampler type: %s\n", f.c_str());
        }
    }
    PXG_CHECK_MSG(declSampler == 1 && tex3dDeclares,
                  "shimBindless_setLayout: %zu module(s) declare a sampler type (npTex3d.spv among them: %d); "
                  "want exactly npTex3d.spv", declSampler, int(tex3dDeclares));
    // Synthetic words after a valid 5-word header: OpTypeSampler %9 / OpTypeSampledImage
    // %9 %8 declare one; an OpDecorate does not; an instruction overrunning the end is
    // malformed and reads as "no".
    auto synth = [](std::initializer_list<uint32_t> tail) {
        std::vector<uint32_t> w = { 0x07230203u, 0x00010500u, 0u, 100u, 0u };
        w.insert(w.end(), tail);
        return w;
    };
    const std::vector<uint32_t> sy26 = synth({ (2u << 16) | 26u, 9u });
    const std::vector<uint32_t> sy27 = synth({ (3u << 16) | 27u, 9u, 8u });
    const std::vector<uint32_t> sy71 = synth({ (4u << 16) | 71u, 9u, 33u, 0u });
    const std::vector<uint32_t> syOv = synth({ (10u << 16) | 26u });
    const bool d26 = pxgSpirvDeclaresSampler(sy26.data(), sy26.size() * 4);
    const bool d27 = pxgSpirvDeclaresSampler(sy27.data(), sy27.size() * 4);
    const bool d71 = pxgSpirvDeclaresSampler(sy71.data(), sy71.size() * 4);
    const bool dOv = pxgSpirvDeclaresSampler(syOv.data(), syOv.size() * 4);
    PXG_CHECK_MSG(d26 && d27 && !d71 && !dOv,
                  "shimBindless_setLayout: declaresSampler(OpTypeSampler, OpTypeSampledImage, OpDecorate, overrun) "
                  "= %d %d %d %d; want 1 1 0 0", int(d26), int(d27), int(d71), int(dOv));
    // ...and the door refuses the real sampling module (never dispatched).
    {
        const std::vector<uint8_t> t3 = loadShaderBlob("npTex3d.spv");
        const uint32_t t3Strides[2] = { 16, 4 };
        const PxgPipelineHandle pT3 = shared->createPipelineBindless(t3.data(), t3.size(), "tex3dSample", 2,
                                                                     t3Strides, 16u);
        PXG_CHECK_MSG(pT3 == 0, "shimBindless_setLayout: createPipelineBindless(npTex3d.spv) gave %u; want 0 "
                      "(it declares a sampler)", unsigned(pT3));
        if (pT3)
            shared->destroyPipeline(pT3);
    }

    // ...and the DOOR applies it: the patched module gets no pipeline, the real one does.
    // (Never dispatched: a pipeline built from atB1 is exactly the measured hazard.)
    const PxgPipelineHandle pBad = shared->createPipelineBindless(atB1.data(), atB1.size(), "tex3dBindlessLoad",
                                                                  0, nullptr, uint32_t(sizeof(ShimTxbParams)));
    const PxgPipelineHandle pOk = shared->createPipelineBindless(real.data(), real.size(), "tex3dBindlessLoad",
                                                                 0, nullptr, uint32_t(sizeof(ShimTxbParams)));
    PXG_CHECK_MSG(pBad == 0 && pOk != 0,
                  "shimBindless_setLayout: createPipelineBindless gave %u for the (1, 1) module (want 0) and "
                  "%u for the real one (want != 0)", unsigned(pBad), unsigned(pOk));
    if (pBad)
        shared->destroyPipeline(pBad);
    if (pOk)
        shared->destroyPipeline(pOk);

    std::printf("  shimBindless_setLayout: 1 binding {0, SAMPLED_IMAGE, 4096}, 0 sampler slots; samplers "
                "0/0/1/0/0; %zu modules fit, (1,1) and (2,0) refused; only npTex3d declares a sampler\n",
                files.size());
}
