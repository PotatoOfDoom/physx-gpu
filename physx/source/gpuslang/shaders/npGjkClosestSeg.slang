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

// npGjkClosestSeg.slang
//
// Slang port of the GJK 1-simplex (line-segment) closest-point — the `case 1`
// inline of the gjk() driver (gjk.cuh :323-329), the missing sibling of
// closestPointTri (tranche 122) and closestPointTet (tranche 123).  Given the
// existing base vertex a (vector slot 0 of the simplex) and the freshly added
// support vertex c, it returns the closest point of the origin on segment [a,c]
// plus a feature code:
//
//   * DEGENERATE (0xffff): c coincident with a (segment length 0), point = 0
//   * 0x2102 (line region): origin projects onto the interior -> a + l*x/m
//   * 0x2111 (new-vertex region): origin past c -> point = c
//
// where l = c - a, m = dot(l,l), x = -dot(a,l).  The simplex itself is NOT
// modified here (unlike the Tri/Tet cores which plant an EPA start normal); the
// caller performs the simplex permute separately.
//
// Warp layout.  Both a (= splatV(0, simplexV)) and the support vertex c
// (loadV3Unsafe -> reinterpret_cast<float*>(v)[lane&3]) are a single vec3
// replicated across all 8 vector-slots: lane L holds component L&3.  Hence
// dot(l,l) and dot(a,l) are uniform across the whole warp after splatX, the
// `m==0` and `x<m` predicates are warp-uniform (divergence-free, as in the
// CUDA), and the result a + l*x/m is likewise replicated.  With integer simplex
// coordinates every product / dot is exact and the single division x/m is
// correctly-rounded IEEE (note the CUDA grouping `l * x / m` == `(l*x)/m`), so
// the shader and the CPU reference agree bit-for-bit.

static const uint DEGENERATE = 0xffffu;

struct CtParams { uint numTests; uint pad0, pad1, pad2; }

[[vk::push_constant]]
ConstantBuffer<CtParams> gParams;

RWStructuredBuffer<float> gZ;    // binding 0: simplex zV (32/test) — in (only slot 0 = a used)
RWStructuredBuffer<float> gCV;   // binding 1: support vertex cV (32/test) — in
RWStructuredBuffer<float> gCP;   // binding 2: closest point (32/test) — out
RWStructuredBuffer<uint> gRet;   // binding 3: return feature code (per test)

// ---- serialized 8-vec4-in-warp layout ops (tranche 120) --------------------
void dotW(float a[32], float b[32], out float o[32])
{
    float x[32];
    for (uint L = 0u; L < 32u; ++L) x[L] = a[L] * b[L];
    for (uint L = 0u; L < 32u; ++L)
    {
        const float s1 = (L + 1u < 32u) ? x[L + 1u] : x[L];
        const float s2 = (L + 2u < 32u) ? x[L + 2u] : x[L];
        o[L] = x[L] + s1 + s2;
    }
}
void splatXW(float a[32], out float o[32]) { for (uint L = 0u; L < 32u; ++L) o[L] = a[L & 28u]; }
void splatVW(uint index, float a[32], out float o[32])
{
    for (uint L = 0u; L < 32u; ++L) o[L] = a[((index << 2) | (L & 3u)) & 31u];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ctClosestPointSeg(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint base = t * 32u;

    float zV[32], cV[32];
    for (uint L = 0u; L < 32u; ++L) { zV[L] = gZ[base + L]; cV[L] = gCV[base + L]; }

    // aV = splatV(0, simplexV)
    float aV[32]; splatVW(0u, zV, aV);
    // lV = supportVertexV - aV
    float lV[32]; for (uint L = 0u; L < 32u; ++L) lV[L] = cV[L] - aV[L];
    // m = splatX(dot(lV,lV))
    float dll[32]; dotW(lV, lV, dll);
    float m[32]; splatXW(dll, m);
    // x = -splatX(dot(aV,lV))
    float dal[32]; dotW(aV, lV, dal);
    float x[32]; splatXW(dal, x);
    for (uint L = 0u; L < 32u; ++L) x[L] = -x[L];

    // Note (equivalent boundary): the m==0 and x<m predicates are warp-uniform.
    // Because a and the support vertex are each replicated across all 8 vector-
    // slots (splatV(0,...) / loadV3Unsafe), dot(l,l) and dot(a,l) are identical
    // in every slot's comp-0 lane, so m and x take the same value at lanes
    // 0,4,8,...,28.  Reading the representative from lane 4 (or any comp-0 lane)
    // instead of lane 0 is therefore unobservable — a documented equivalent
    // mutant.  Lane 0 is used verbatim, faithful to the CUDA warp semantics.
    float pointV[32];
    uint f;
    if (m[0] == 0.0f)               // FAIL: coincident vertex
    {
        f = DEGENERATE;
        for (uint L = 0u; L < 32u; ++L) pointV[L] = 0.0f;
    }
    else if (x[0] < m[0])           // REGION: line segment
    {
        f = 0x2102u;
        for (uint L = 0u; L < 32u; ++L) pointV[L] = aV[L] + lV[L] * x[L] / m[L];
    }
    else                            // REGION: new vertex
    {
        f = 0x2111u;
        for (uint L = 0u; L < 32u; ++L) pointV[L] = cV[L];
    }

    for (uint L = 0u; L < 32u; ++L) gCP[base + L] = pointV[L];
    gRet[t] = f;
}
