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

// RUNTIME_PLAN.md P3/P4 test: the generated Slang kernel manifest.  Proves the
// manifest (tools/gen_kernel_manifest.py -> PxgSlangKernelManifest.h) is
// consistent with the LIVE registry header and that every mapped row points at
// a real, valid SPIR-V module:
//   1. row count == the registry's KERNEL_DEF count;
//   2. row i names registry kernel i (exact order + name — catches drift if a
//      KERNEL_DEF is added/reordered without regenerating);
//   3. every mapped row's .spv exists and has the SPIR-V magic word;
//   4. reports the launch-kernel port coverage (mapped / 501).

#include "pxgTestFramework.h"

#include "PxgSlangKernelManifest.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx::gpuslang;

// The live registry, expanded straight from the engine header via the X-macro,
// so the test cross-checks the generated manifest against ground truth.
static const char* kRegistryNames[] = {
#define KERNEL_DEF(id, name) name,
#include "PxgKernelNames.h"
#undef KERNEL_DEF
};

PXG_TEST(p4Manifest)
{
    const size_t nReg = sizeof(kRegistryNames) / sizeof(kRegistryNames[0]);
    PXG_CHECK_MSG(nReg == PXG_SLANG_KERNEL_COUNT,
                  "manifest row count matches the live registry (%zu vs %u)", nReg,
                  PXG_SLANG_KERNEL_COUNT);

    // Exact order + name.
    size_t bad = 0;
    const size_t n = nReg < PXG_SLANG_KERNEL_COUNT ? nReg : PXG_SLANG_KERNEL_COUNT;
    for (size_t i = 0; i < n; ++i)
    {
        if (std::strcmp(kRegistryNames[i], gPxgSlangKernelManifest[i].cudaName) != 0)
        {
            if (bad < 5)
                std::printf("  ORDER MM row %zu: registry '%s' vs manifest '%s'\n", i,
                            kRegistryNames[i], gPxgSlangKernelManifest[i].cudaName);
            ++bad;
        }
    }
    PXG_CHECK_MSG(bad == 0, "manifest is in exact registry order (%zu mismatches)", bad);

    // Every mapped row resolves to a valid SPIR-V module; the entry name equals
    // the CUDA name (the launch-kernel port contract).
    size_t mapped = 0, badSpirv = 0, badEntry = 0;
    for (size_t i = 0; i < PXG_SLANG_KERNEL_COUNT; ++i)
    {
        const PxgSlangKernelRow& r = gPxgSlangKernelManifest[i];
        if (!r.spirvFile)
            continue;
        ++mapped;
        if (!r.entry || std::strcmp(r.entry, r.cudaName) != 0)
            ++badEntry;
        const std::vector<uint8_t> blob = loadShaderBlob(r.spirvFile);
        uint32_t magic = 0;
        if (blob.size() >= 4)
            std::memcpy(&magic, blob.data(), 4);
        if (magic != 0x07230203u)
        {
            if (badSpirv < 5)
                std::printf("  BAD SPIR-V for '%s' (%s): magic %08x\n", r.cudaName, r.spirvFile, magic);
            ++badSpirv;
        }
    }
    PXG_CHECK_MSG(badSpirv == 0, "every mapped module is valid SPIR-V (%zu bad)", badSpirv);
    PXG_CHECK_MSG(badEntry == 0, "every mapped entry name equals its CUDA name (%zu bad)", badEntry);
    PXG_CHECK_MSG(mapped == PXG_SLANG_KERNEL_MAPPED,
                  "mapped count matches the manifest header (%zu vs %u)", mapped,
                  PXG_SLANG_KERNEL_MAPPED);

    std::printf("  manifest coverage: %zu/%u registry launch kernels mapped to a ported Slang entry "
                "(%.1f%%)\n", mapped, PXG_SLANG_KERNEL_COUNT,
                100.0 * double(mapped) / double(PXG_SLANG_KERNEL_COUNT));
}

}  // namespace
