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
// that set (bridge cuTexObjectCreate -> createTexture3DFmt, bindless).  No scene
// reaches any of the three with a real SDF, so nothing has ever run the
// combination: an engine-built pipeline sampling an engine-uploaded texture.
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
#include <map>
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
// mapped row of the engine manifest, the entries that use set 1 must be the four
// SDF kernels and nothing else.  P9.83b-8 mapped triangleTriangleCollision (it
// samples both meshes' SDFs) and the list grew by one, on purpose.
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
                                             "sb_sdfMeshMidphaseGeneratePairsLaunch", "triangleTriangleCollision" };
    PXG_CHECK_MSG(missing == 0, "shimBindless_classify: %zu mapped rows have no loadable .spv", missing);
    for (const std::string& n : sampling)
        if (!expected.count(n))
            std::printf("  classified as sampling, not expected: %s\n", n.c_str());
    for (const std::string& n : expected)
        if (!sampling.count(n))
            std::printf("  expected to sample, not classified: %s\n", n.c_str());
    PXG_CHECK_MSG(sampling == expected,
                  "shimBindless_classify: %zu engine entries classified as using set 1, expected the 4 "
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
        // createPipelineTex-style texture lives in set 0 (with a Binding 1 on its
        // sampler, so a test that read the wrong decoration word would say yes).
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
                "(the 4 SDF kernels); %zu per-entry cases\n",
                mapped, blobs.size(), sampling.size(), sizeof(cases) / sizeof(cases[0]));
}
