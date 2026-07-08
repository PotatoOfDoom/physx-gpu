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

// clampValues.slang
//
// Slang port of the clamp utilities from gpucommon/src/CUDA/MemCopyBalanced.cu:
//   - clampMaxValue
//   - clampMaxValues
//
// The CUDA kernels take bare device pointers to the counters; the port
// clamps entries of a bound buffer selected by push-constant indices.
// Single-thread launches, like the originals.

struct ClampParams
{
    uint index0;
    uint index1;
    uint index2;
    uint maxValue;
};

[[vk::push_constant]]
ConstantBuffer<ClampParams> gParams;

RWStructuredBuffer<uint> gValues; // binding 0

// mirrors clampMaxValue()
[shader("compute")]
[numthreads(1, 1, 1)]
void clampMaxValue()
{
    if (gValues[gParams.index0] > gParams.maxValue)
        gValues[gParams.index0] = gParams.maxValue;
}

// mirrors clampMaxValues()
[shader("compute")]
[numthreads(1, 1, 1)]
void clampMaxValues()
{
    if (gValues[gParams.index0] > gParams.maxValue)
        gValues[gParams.index0] = gParams.maxValue;
    if (gValues[gParams.index1] > gParams.maxValue)
        gValues[gParams.index1] = gParams.maxValue;
    if (gValues[gParams.index2] > gParams.maxValue)
        gValues[gParams.index2] = gParams.maxValue;
}
