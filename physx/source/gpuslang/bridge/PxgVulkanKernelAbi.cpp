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

// See PxgVulkanKernelAbi.h.  One row per kernel brought up on the engine
// launch path (P5 Step 4); argument sizes are transcribed from the CUDA
// __global__ signature verbatim.  Pointer = 8, PxU32/float/enum = 4,
// PxU16 = 2, PxU8/bool = 1; aggregates passed BY VALUE get their sizeof
// (PhysX passes large aggregates by device pointer, so by-value structs
// stay small).

#include "PxgVulkanKernelAbi.h"

#include <cstring>

namespace physx
{
namespace gpuslang
{
namespace
{

// gpucommon/src/CUDA/MemCopyBalanced.cu:
//   MemCopyBalanced(CopyDesc* descs, PxU32 count)
static const uint8_t kMemCopyBalanced[] = { 8, 4 };

// gpubroadphase/src/CUDA/broadphase.cu:
//   translateAABBsLaunch(const PxBounds3* in, PxgIntegerAABB* out,
//                        const PxReal* contactDistances, const PxU32* envIds,
//                        PxU32 boxesCapacity, PxU8 x6 shift/envID bits)
static const uint8_t kTranslateAABBs[] = { 8, 8, 8, 8, 4, 1, 1, 1, 1, 1, 1 };

// broadphase.cu: clearNewFlagLaunch(const PxgBroadPhaseDesc*) / bpSignalComplete(PxU32*)
static const uint8_t kOnePtr[] = { 8 };

// broadphase.cu: initializeSapBox1DLaunch(const PxgBroadPhaseDesc*, PxU32 numHandles, bool isNew)
static const uint8_t kSapBoxInit[] = { 8, 4, 1 };

static const PxgKernelAbiRow kRows[] = {
    { "MemCopyBalanced", kMemCopyBalanced, uint32_t(sizeof(kMemCopyBalanced)) },
    { "translateAABBsLaunch", kTranslateAABBs, uint32_t(sizeof(kTranslateAABBs)) },
    { "clearNewFlagLaunch", kOnePtr, 1u },
    { "bpSignalComplete", kOnePtr, 1u },
    { "markCreatedPairsLaunch", kOnePtr, 1u },
    { "initializeSapBox1DLaunch", kSapBoxInit, uint32_t(sizeof(kSapBoxInit)) },
    { "initializeRadixRanks", kOnePtr, 1u },
    { "updateHandles", kOnePtr, 1u },
};

inline size_t abiAlign(size_t size)
{
    if (size >= 16) return 16;
    if (size >= 8)  return 8;
    if (size >= 4)  return 4;
    if (size >= 2)  return 2;
    return 1;
}

} // anonymous namespace

const PxgKernelAbiRow* pxgFindKernelAbi(const char* cudaName)
{
    if (!cudaName)
        return nullptr;
    for (const PxgKernelAbiRow& row : kRows)
        if (std::strcmp(row.cudaName, cudaName) == 0)
            return &row;
    return nullptr;
}

uint32_t pxgKernelAbiPushSize(const PxgKernelAbiRow& row)
{
    size_t offset = 0;
    size_t maxAlign = 1;
    for (uint32_t i = 0; i < row.argCount; ++i)
    {
        const size_t a = abiAlign(row.argSizes[i]);
        if (a > maxAlign)
            maxAlign = a;
        offset = (offset + (a - 1)) & ~(a - 1);
        offset += row.argSizes[i];
    }
    offset = (offset + (maxAlign - 1)) & ~(maxAlign - 1);
    return uint32_t(offset);
}

} // namespace gpuslang
} // namespace physx
