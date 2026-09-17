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

// npTriTriReduce.slang -- P9.83b-5.
//
//   Fixture entry for ttpReduceContactsSerial, the serial replacement for
//   reduceContactsSingleWarp (convexMeshMidphase.cu:440-503).
//
// The property under test is the STREAMING loop, not the reduction it calls --
// crsContactReduce2 is already certified by gpuslang.npCrsWrapper2, which
// asserts it differs from wrapper 1.  What nothing covers yet is the refill
// discipline around it: which staging slot each free lane takes, that `c`
// advances by the number of lanes OFFERED rather than taken, that survivors
// compact in ascending lane order, and that the 100-pass bound is a real bound.
//
// So the host mirror is written from the CUDA loop and the case set is sized to
// make the loop run MANY passes: with totalContacts up to 198 and at most 6
// survivors per pass, a full case streams 198 points through 32 lanes in about
// seven refills.  A case set that only ever fit in one pass would certify the
// reduction a second time and the loop not at all.
//
// No NoContraction stamp: the module's other callers are not stamped either, and
// stamping only the fixture certifies a build of the module the engine never
// runs.  See the note in testNpCrsWrapper2.cpp, where that mistake was nearly
// made for the same module.

import pxgSlangTriTriPatch;

struct TtrParams
{
    uint numCases;
    uint pad0, pad1, pad2;
}

struct TtrCase
{
    float4 pts[198];     // xyz = point, w = separation (the staging buffer)
    float4 normal;       // xyz = patch normal, w = clusterBias
    uint4 misc;          // x = totalContacts, y..w unused
}

[[vk::push_constant]] ConstantBuffer<TtrParams> gP;

RWStructuredBuffer<TtrCase> gIn;     // binding 0
RWStructuredBuffer<float4> gOut;     // binding 1: 8 float4 per case
                                     //   [0..5] keepPoints
                                     //   [6]    (pointCount, deepest, mask, 0)
                                     //   [7]    reserved

[shader("compute")]
[numthreads(64, 1, 1)]
void ttrReduce(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gP.numCases) return;

    float4 pts[198];
    for (uint j = 0; j < 198u; ++j)
        pts[j] = gIn[t].pts[j];

    const float4 n = gIn[t].normal;
    const TtpReduceResult r = ttpReduceContactsSerial(pts, gIn[t].misc.x, n.xyz, n.w);

    for (uint k = 0; k < 6u; ++k)
        gOut[t * 8u + k] = r.keepPoints[k];
    gOut[t * 8u + 6u] = float4(float(r.pointCount), r.deepest, asfloat(r.survivorMask), 0.0f);
    gOut[t * 8u + 7u] = float4(0, 0, 0, 0);
}
