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

// npArtiBitfield.slang
//
// Slang port of the 64-bit ArticulationBitField scan helpers
// articulationLowestSetBit / articulationHighestSetBit
// (articulationImpulseResponse.cuh L47-57), used to iterate the articulation
// link bitfields.
//
//   lowestSetBit(val)  = val==0 ? 0 : __ffsll(val)-1   (index of lowest set bit)
//   highestSetBit(val) = 63 - __clzll(val)             (index of highest set bit)
//
// The 64-bit value is passed as a uint2 (lo, hi); __ffsll/__clzll are realised
// per-32-bit-word via firstbitlow/firstbithigh (SPIR-V FindILsb/FindUMsb,
// which return 0xFFFFFFFF for a zero word).  Pure unsigned-integer arithmetic
// (no float exception classes).  highestSetBit(0) is UB in CUDA (__clzll(0));
// here it returns firstbithigh(0)=0xFFFFFFFF and the reference matches, but the
// fixture does not rely on it.

struct BsIn
{
    uint4 v;   // v.x = low 32 bits, v.y = high 32 bits
}

struct BsOut
{
    uint4 r;   // r.x = lowest set bit, r.y = highest set bit
}

struct BsParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<BsParams> gParams;

RWStructuredBuffer<BsIn> gIn;    // binding 0
RWStructuredBuffer<BsOut> gOut;  // binding 1

// __ffsll(val)-1 for val!=0, 0 for val==0
uint lowestSetBit(uint lo, uint hi)
{
    if (lo != 0u)
        return firstbitlow(lo);
    else if (hi != 0u)
        return 32u + firstbitlow(hi);
    else
        return 0u;   // val == 0
}

// 63 - __clzll(val) = index of highest set bit (val==0 -> 0xFFFFFFFF)
uint highestSetBit(uint lo, uint hi)
{
    if (hi != 0u)
        return 32u + firstbithigh(hi);
    else
        return firstbithigh(lo);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bitScan(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const BsIn s = gIn[tid];

    const uint lo = s.v.x;
    const uint hi = s.v.y;

    gOut[tid].r = uint4(lowestSetBit(lo, hi), highestSetBit(lo, hi), 0u, 0u);
}
