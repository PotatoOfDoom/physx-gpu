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

// npGjkClosestTri.slang
//
// Slang port of closestPointTri (gjk.cuh :75-110) — tranche 122, the GJK
// warp-SIMD closest-point of the origin on the triangle formed by a line simplex
// (2 base verts in vector slots 0,1) and the new apex cV.  Returns a feature code
// and writes the closest point; on the face region it also plants an EPA
// start-normal in the simplex.  Built on the tranche-120/121 layout primitives
// (dot / cross / splatX / splatV / anyX / packXFlags, setNormal0) and next3/prev3.
//
// The four Voronoi regions (verbatim from the CUDA):
//   * DEGENERATE  (0xffff): apex coincident with a base vert (edge length 0)
//   * face   (0x0123 / 0x1023): origin inside both edge planes -> project onto
//            the face plane, set normal slot 0, winding by projD sign
//   * edge   (eid<<8 | 0x22): outside an edge plane but projecting onto that edge
//   * apex   (0x0021): the apex itself
//
// Serialisation: one thread simulates the whole 32-lane warp with per-lane
// arrays.  The layout is 8 vec4s (lane L = component L&3 of vector L>>2); the
// cross-product lane indices are per-lane cp0I/cp1I = (L&28) | next3/prev3(L&3),
// __shfl_xor(·,4) swaps vector-slot pairs, __shfl(·,0) broadcasts lane 0, and
// __ffs(mask) = mask ? firstbitlow(mask)+1 : 0.  With integer simplex coordinates
// every product / dot / cross is exact and the two divisions (projD/dot(mV,mV) and
// pS/lD) are correctly-rounded IEEE, so the shader and the CPU reference agree
// bit-for-bit; the meaningful results live in the comp-0 lanes and the residue in
// the others is reproduced identically.

static const uint DEGENERATE = 0xffffu;

struct CtParams { uint numTests; uint pad0, pad1, pad2; }

[[vk::push_constant]]
ConstantBuffer<CtParams> gParams;

RWStructuredBuffer<float> gZ;    // binding 0: simplex zV (32/test) — in
RWStructuredBuffer<float> gCV;   // binding 1: apex cV (32/test)
RWStructuredBuffer<float> gCP;   // binding 2: closest point (32/test) — out
RWStructuredBuffer<float> gZOut; // binding 3: simplex zV (32/test) — out (normal set)
RWStructuredBuffer<uint> gRet;   // binding 4: return feature code (per test)

uint next3(uint x) { return x == 2u ? 0u : x + 1u; }
uint prev3(uint x) { return x == 0u ? 2u : x - 1u; }

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
void crossW(float a[32], float b[32], uint cp0[32], uint cp1[32], out float o[32])
{
    for (uint L = 0u; L < 32u; ++L)
        o[L] = a[cp0[L]] * b[cp1[L]] - a[cp1[L]] * b[cp0[L]];
}
void splatXW(float a[32], out float o[32]) { for (uint L = 0u; L < 32u; ++L) o[L] = a[L & 28u]; }
void splatVW(uint index, float a[32], out float o[32])
{
    for (uint L = 0u; L < 32u; ++L) o[L] = a[((index << 2) | (L & 3u)) & 31u];
}
uint ffs(uint m) { return m == 0u ? 0u : firstbitlow(m) + 1u; }

[shader("compute")]
[numthreads(64, 1, 1)]
void ctClosestPointTri(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint base = t * 32u;

    float zV[32], cV[32];
    for (uint L = 0u; L < 32u; ++L) { zV[L] = gZ[base + L]; cV[L] = gCV[base + L]; }

    uint cp0I[32], cp1I[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        cp0I[L] = ((L & 28u) | next3(L & 3u)) & 31u;
        cp1I[L] = ((L & 28u) | prev3(L & 3u)) & 31u;
    }

    float closestPoint[32];
    for (uint L = 0u; L < 32u; ++L) closestPoint[L] = 0.0f;

    // eV = cV - zV;  lD = splatX(dot(eV,eV))
    float eV[32]; for (uint L = 0u; L < 32u; ++L) eV[L] = cV[L] - zV[L];
    float dee[32]; dotW(eV, eV, dee);
    float lD[32]; splatXW(dee, lD);

    // anyX<2>(lD == 0) -> DEGENERATE
    if ((lD[0] == 0.0f) || (lD[4] == 0.0f))
    {
        for (uint L = 0u; L < 32u; ++L) { gCP[base + L] = 0.0f; gZOut[base + L] = zV[L]; }
        gRet[t] = DEGENERATE;
        return;
    }

    // shortEdgeV = (shfl_xor(lD,4) < lD) ? shfl_xor(eV,4) : eV
    float shortEdgeV[32];
    for (uint L = 0u; L < 32u; ++L)
        shortEdgeV[L] = (lD[L ^ 4u] < lD[L]) ? eV[L ^ 4u] : eV[L];

    // mV = cross(shfl_xor(zV,4) - zV, shortEdgeV, cp0I, cp1I)
    float d1[32]; for (uint L = 0u; L < 32u; ++L) d1[L] = zV[L ^ 4u] - zV[L];
    float mV[32]; crossW(d1, shortEdgeV, cp0I, cp1I, mV);

    // pS = dot(eV, cV)
    float pS[32]; dotW(eV, cV, pS);

    // uS = dot(cross(mV, eV, cp0I, cp1I), cV) <= 0
    float cme[32]; crossW(mV, eV, cp0I, cp1I, cme);
    float dcme[32]; dotW(cme, cV, dcme);
    bool uS[32]; for (uint L = 0u; L < 32u; ++L) uS[L] = dcme[L] <= 0.0f;

    // ---- face region: !anyX<2>(uS) ----
    if (!(uS[0] || uS[4]))
    {
        float dmc[32]; dotW(mV, cV, dmc);
        const float projD = dmc[0];               // shfl(dot(mV,cV), 0)
        float dmm[32]; dotW(mV, mV, dmm);
        float sxdmm[32]; splatXW(dmm, sxdmm);
        float tmp[32]; for (uint L = 0u; L < 32u; ++L) tmp[L] = mV[L] * projD / sxdmm[L];
        splatVW(0u, tmp, closestPoint);

        const bool flip = projD < 0.0f;
        for (uint L = 0u; L < 32u; ++L) if ((L & 3u) == 3u) mV[L] = projD; // mV.w = projD

        // setNormal0(zV, mV, flip): normal slot 0 (lanes 16-19) = splatV(flip, mV)
        const uint index = flip ? 1u : 0u;
        for (uint L = 0u; L < 32u; ++L)
        {
            const uint src = ((index << 2) | (L & 3u)) & 31u;
            const float tmpV = mV[src];
            zV[L] = (L < 16u || L > 19u) ? zV[L] : tmpV;
        }

        for (uint L = 0u; L < 32u; ++L) { gCP[base + L] = closestPoint[L]; gZOut[base + L] = zV[L]; }
        gRet[t] = flip ? 0x0123u : 0x1023u;
        return;
    }

    // ---- edge region: packXFlags<2>(uS & pS>0) ----
    uint emask = 0u;
    if (uS[0] && (pS[0] > 0.0f)) emask |= 1u;
    if (uS[4] && (pS[4] > 0.0f)) emask |= 2u;
    uint eid = ffs(emask);
    if (eid != 0u)
    {
        eid = eid - 1u;
        float sxpS[32]; splatXW(pS, sxpS);
        float tmp[32]; for (uint L = 0u; L < 32u; ++L) tmp[L] = cV[L] - eV[L] * sxpS[L] / lD[L];
        splatVW(eid, tmp, closestPoint);
        for (uint L = 0u; L < 32u; ++L) { gCP[base + L] = closestPoint[L]; gZOut[base + L] = zV[L]; }
        gRet[t] = (eid << 8) | 0x22u;
        return;
    }

    // ---- apex region ----
    for (uint L = 0u; L < 32u; ++L) { gCP[base + L] = cV[L]; gZOut[base + L] = zV[L]; }
    gRet[t] = 0x0021u;
}
