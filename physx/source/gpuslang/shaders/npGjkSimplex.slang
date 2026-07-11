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

// npGjkSimplex.slang
//
// Slang port of the GJK simplex-layout normal accessors setNormal0 / setNormal123
// (gjk.cuh :40-50) — tranche 121.  The simplex is stored as 8 vec4s in a 32-lane
// warp: the 4 vertices occupy lanes 0-15, the 4 EPA start-normals occupy lanes
// 16-31 (normal slot k = lanes 16+4k .. 19+4k).  These two writes place the
// closest-feature normals into the simplex so it can be shot straight into EPA.
//
//   setNormal0(z, nV, index):  tmpV = splatV(index, nV);
//                              lane in [16,19] -> tmpV (normal slot 0 = nV's
//                              vector `index`); every other lane keeps z.
//   setNormal123(z, nV):       tmpV = shfl_up(nV, 20);
//                              lane >= 20 -> nV[L-20] (normal slots 1,2,3 receive
//                              the 3 face normals held in the vertex slots); lanes
//                              < 20 keep z.
//
// splatV(index,nV) at lane L = nV[(index<<2)|(L&3)] (tranche 120); __shfl_up_sync
// by 20 returns the caller's own value when L-20 < 0 (irrelevant here, those lanes
// keep z).  Serialised one thread per warp.  These are pure lane-select copies —
// no arithmetic — so the port is bit-exact; the test pins the exact lane partition
// with distinct per-lane payloads.
//
// Note: in setNormal123 the shfl_up "own value" branch (tmpV for lanes < 20)
// mirrors the CUDA (__shfl_up_sync yields a value for every lane) but is dead —
// the output keeps z for L < 20 — so a mutation of that gate is unobservable.
// The branch is retained verbatim (faithful to the CUDA shfl_up).

struct SxParams { uint numTests; uint pad0, pad1, pad2; }

struct SxIn { uint4 s; } // x = index (setNormal0)

[[vk::push_constant]]
ConstantBuffer<SxParams> gParams;

RWStructuredBuffer<float> gZ;    // binding 0: simplex z (32/test)
RWStructuredBuffer<float> gNV;   // binding 1: normal source nV (32/test)
RWStructuredBuffer<SxIn> gIn;    // binding 2
RWStructuredBuffer<float> gOut;  // binding 3: new simplex (32/test)

// setNormal0: write nV's vector `index` into normal slot 0 (lanes 16-19).
[shader("compute")]
[numthreads(64, 1, 1)]
void ctGjkSetNormal0(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint base = t * 32u;
    const uint index = gIn[t].s.x;
    for (uint L = 0u; L < 32u; ++L)
    {
        const uint src = ((index << 2) | (L & 3u)) & 31u; // splatV(index, nV) at lane L
        const float tmpV = gNV[base + src];
        gOut[base + L] = (L < 16u || L > 19u) ? gZ[base + L] : tmpV;
    }
}

// setNormal123: copy the 3 vertex-slot vectors of nV into normal slots 1,2,3.
[shader("compute")]
[numthreads(64, 1, 1)]
void ctGjkSetNormal123(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint base = t * 32u;
    for (uint L = 0u; L < 32u; ++L)
    {
        // shfl_up(nV, 20): lane L reads L-20 when L >= 20, else its own value
        const float tmpV = (L >= 20u) ? gNV[base + (L - 20u)] : gNV[base + L];
        gOut[base + L] = (L < 20u) ? gZ[base + L] : tmpV;
    }
}
