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

// npCrsWrapper2.slang -- P9.83b-2.
//
//   The fixture entry for pxgSlangContactReduceSerial's TWO wrappers over one
//   body.  It runs BOTH on the same lane arrays and writes both masks, because
//   the property under test is not "wrapper 2 is right" on its own -- it is that
//   wrapper 2 differs from wrapper 1 in the way contactReduction.cuh says it does.
//
// contactReduce  (:171-177)  adj = 0,          initial = separation
// contactReduce2 (:180-188)  adj = -separation, initial = |projected point|
//
// A fixture that only checked wrapper 2 against its own mirror could pass with
// both wrappers implemented identically, which is exactly the mistake the four
// pre-existing transcriptions had baked in (they all collapse f and g onto one
// array, correct only while adj is zero).  So the test also counts how often the
// two masks DIFFER and fails if that count is small: an observable that is true
// under the null hypothesis measures nothing.
//
// No NoContraction stamp is needed and none is applied: the fixture compares the
// shader against a CPU mirror built with -ffp-contract=off, and the module this
// drives is already stamped through its engine callers.  If this entry is ever
// added to PXG_NOCONTRACT_SHADERS, the module's other callers must go with it --
// see the fcConvexTri note in tests/CMakeLists.txt.

import pxgSlangContactReduceSerial;

struct CrsParams
{
    uint numCases;
    uint pad0, pad1, pad2;
}

// One case: 32 lanes of point + separation, a normal, the live-lane mask and the
// cluster bias.  Laid out as float4 so the host mirror and the shader agree on
// stride without either of them choosing it.
struct CrsCase
{
    float4 pt[32];        // xyz = pointRaw, w unused
    float4 sep8[8];       // 32 separations, four per float4
    float4 normal;        // xyz = normal, w unused
    uint4 misc;           // x = allMask, y..w unused
    float4 bias;          // x = clusterBias
}

[[vk::push_constant]] ConstantBuffer<CrsParams> gP;

RWStructuredBuffer<CrsCase> gIn;    // binding 0
RWStructuredBuffer<uint4> gOut;     // binding 1: x = wrapper1 mask, y = wrapper2 mask

[shader("compute")]
[numthreads(64, 1, 1)]
void crsWrapperPair(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gP.numCases) return;

    const CrsCase c = gIn[t];

    float3 pt[32];
    float sep[32];
    for (uint j = 0; j < 32u; ++j)
    {
        pt[j] = c.pt[j].xyz;
        const float4 s = c.sep8[j >> 2];
        const uint k = j & 3u;
        sep[j] = (k == 0u) ? s.x : (k == 1u) ? s.y : (k == 2u) ? s.z : s.w;
    }

    const CrsConfig cfg = crsConfig(true, true, 6);
    const uint m1 = crsContactReduce(pt, sep, c.normal.xyz, c.misc.x, c.bias.x, cfg);
    const uint m2 = crsContactReduce2(pt, sep, c.normal.xyz, c.misc.x, c.bias.x, cfg);

    gOut[t] = uint4(m1, m2, 0u, 0u);
}
