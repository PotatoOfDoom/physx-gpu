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

// npCvxTriFind.slang
//
// Slang port of the triangle-index lookup used by the convex-mesh/heightfield
// post-process cores — tranche 113.  Two pieces:
//
//   * binarySearch (gpucommon/src/CUDA/reduction.cuh :729): returns the LARGEST
//     index into a sorted array with data[index] <= value; if data[0] > value
//     (or the array is empty) it returns 0.  Textbook left/right bisection with
//     a `<=` step and a `left ? left-1 : 0` clamp.
//   * findTriangle / findFaceContactTriangle / findEdgeContactTriangle
//     (convexMeshPostProcess.cu :154-179): OR a TriangleSearchFlags bit into the
//     target key, binary-search the sorted orderedTriangleIndices array, and
//     confirm exact equality (index < count && data[index] == target).
//
// findTriangle ORs the raw literal 0x80000000, which is exactly
// TriangleSearchFlags::eFACE_CONTACT, so findTriangle and findFaceContactTriangle
// are behaviourally IDENTICAL — both are transcribed verbatim (the two names are
// distinct call sites: findTriangle <- convexHeightfieldPostProcessCore,
// findFaceContactTriangle <- convexTrimeshPostProcessCore).  eEDGE_CONTACT =
// 0xc0000000.
//
// Everything is exact unsigned-integer arithmetic — no floating point, no warp
// communication, no shared-memory hull scratch — so the CUDA, the CPU reference
// and this shader agree BIT-EXACTLY by construction.  One thread handles one
// independent lookup; the test drives the full branch matrix of the search and
// the equality confirm (present, absent-interior, below data[0], above the tail,
// duplicate keys, and count 0/1).

static const uint FACE_CONTACT_FLAG = 0x80000000u; // TriangleSearchFlags::eFACE_CONTACT
static const uint EDGE_CONTACT_FLAG = 0xc0000000u; // TriangleSearchFlags::eEDGE_CONTACT

struct FindParams
{
    uint numTests;
    uint maxLen; // stride (in uints) of each test's key block
    uint pad0, pad1;
}

struct FindIn
{
    uint count;  // number of valid sorted keys for this test
    uint target; // raw target index (WITHOUT the flag bit)
    uint which;  // 0 = findTriangle, 1 = findFaceContactTriangle, 2 = findEdgeContactTriangle
    uint pad;
}

struct FindOut
{
    uint result;      // 1 if found (index<count && data[index]==flaggedTarget), else 0
    uint searchIndex; // the raw binarySearch return (for direct verification)
}

[[vk::push_constant]]
ConstantBuffer<FindParams> gParams;

RWStructuredBuffer<uint> gData;    // binding 0: sorted keys, maxLen per test
RWStructuredBuffer<FindIn> gIn;    // binding 1
RWStructuredBuffer<FindOut> gOut;  // binding 2

// reduction.cuh :729 — largest index with gData[base+index] <= value, else 0
uint binarySearch(uint base, uint numElements, uint value)
{
    uint left = 0u;
    uint right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (gData[base + pos] <= value)
            left = pos + 1u;
        else
            right = pos;
    }
    return left != 0u ? left - 1u : 0u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ctCvxTriFind(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;

    const FindIn e = gIn[t];
    const uint base = t * gParams.maxLen;
    const uint flag = (e.which == 2u) ? EDGE_CONTACT_FLAG : FACE_CONTACT_FLAG;
    const uint target = e.target | flag;

    const uint index = binarySearch(base, e.count, target);
    const uint result = (index < e.count && gData[base + index] == target) ? 1u : 0u;

    FindOut o;
    o.result = result;
    o.searchIndex = index;
    gOut[t] = o;
}
