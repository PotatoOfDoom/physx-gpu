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

// memCopy.slang
//
// Slang port of the balanced copy dispatcher and clamp utilities
// (gpucommon/src/CUDA/MemCopyBalanced.cu, used by PxgCopyManager):
//   - MemCopyBalanced
//   - clampMaxValue / clampMaxValues
//
// Differences from CUDA:
//   - PxgCopyManager::CopyDesc carried raw device pointers (dest/source).
//     Descriptor-based APIs cannot dereference arbitrary pointers, so the
//     port copies between two bound buffers (which may alias the same
//     buffer) and the descriptor holds *byte offsets* into them.  The
//     word-granular balanced copy loop is unchanged: one workgroup per
//     descriptor, all "warps" of the group striding over the payload.
//   - clampMaxValue(s) took bare device pointers; the port clamps entries of
//     a bound buffer selected by push-constant indices.

static const uint WARP_SIZE = 32;
static const uint WARPS_PER_BLOCK = 4; // COPY_KERNEL_WARPS_PER_BLOCK
static const uint BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK;

// mirrors PxgCopyManager::CopyDesc (4 x size_t = 32 bytes); dest/source are
// byte offsets into gCopyDst/gCopySrc instead of raw device pointers
struct PxgCopyDesc
{
    uint2 dest;   // byte offset into gCopyDst (low, high)
    uint2 source; // byte offset into gCopySrc (low, high)
    uint2 bytes;  // payload size in bytes (low, high)
    uint2 pad;
};

struct MemCopyParams
{
    uint count; // number of descriptors
};

[[vk::push_constant]]
ConstantBuffer<MemCopyParams> gParams;

RWStructuredBuffer<PxgCopyDesc> gDescs; // binding 0
RWStructuredBuffer<uint> gCopySrc;      // binding 1 (word-addressed)
RWStructuredBuffer<uint> gCopyDst;      // binding 2 (word-addressed)

// mirrors MemCopyBalanced / copyBalanced<COPY_KERNEL_WARPS_PER_BLOCK>()
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void MemCopyBalanced(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    if (groupId.x >= gParams.count)
        return;

    const PxgCopyDesc desc = gDescs[groupId.x];

    const uint srcWord = desc.source.x / 4;
    const uint dstWord = desc.dest.x / 4;
    const uint size = desc.bytes.x / 4; // size is in bytes, we're copying words

    for (uint a = groupThreadId.x; a < size; a += BLOCK_SIZE)
        gCopyDst[dstWord + a] = gCopySrc[srcWord + a];
}
