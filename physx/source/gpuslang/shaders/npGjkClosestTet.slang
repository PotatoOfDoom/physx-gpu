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

// npGjkClosestTet.slang
//
// Slang port of closestPointTet (gjk.cuh :113-169) — tranche 123, the GJK
// warp-SIMD closest-point of the origin on the tetrahedron spanned by a triangle
// simplex (3 base verts in vector slots 0,1,2) and the new apex dV, in the 8-vec4
// layout.  It returns a feature code and the closest point; on the inside region
// it sets the 3 face normals (setNormal123) and on the face region an outward
// face normal (setNormal0).  Built on the tranche-120/121/122 layout primitives.
//
// Regions (verbatim from the CUDA):
//   * DEGENERATE (0xffff): apex beyond the old face (n0V check), an edge length 0,
//     a simplex inversion (aS<=gS), or no Voronoi region found
//   * inside (0x2104):    allX<3>(aS>0)  -> plant the 3 face normals for EPA
//   * apex   (0x2131):    allX<3>(pS<=0)
//   * edge   (eid<<8|0x32): outside two edge planes of a triangle, projects onto edge
//   * face   (next3(fid)<<12 | fid<<8 | 0x33): outside a face plane, inside its edges
//
// Serialisation: one thread simulates the whole 32-lane warp.  Block-rotate
// shuffles vNextI = tI>7?tI-8:tI+4 and vPrevI = tI<4?tI+8:tI-4 cycle the 3 face
// triangles; __shfl_up(aS,3) plants the face distance in the normal's w lane;
// per-lane cross indices and the comp-0-lane convention are as in closestPointTri
// (tranche 122).  With integer simplex coordinates every dot/cross is exact and
// the divisions (pS/lD, aS/dot(nV,nV)) are correctly-rounded IEEE, so the shader
// and the CPU reference agree bit-for-bit.
//
// Note (equivalent boundary): weakening any single face's inversion test from
// `aS<=gS` to `aS<gS` is unobservable.  aS[k]==gS[k] means the k-th opposite
// vertex is coplanar with the k-th face's triangle, i.e. all four points share a
// plane — a global condition that holds for ALL three faces at once, so the two
// faces still using `<=` keep returning DEGENERATE at the boundary (verified by
// an ~8,000,000-tet brute force: zero divergence, and never a face at equality
// with another face not).  The `<=` is retained verbatim, faithful to the CUDA.

static const uint DEGENERATE = 0xffffu;

struct CtParams { uint numTests; uint pad0, pad1, pad2; }

[[vk::push_constant]]
ConstantBuffer<CtParams> gParams;

RWStructuredBuffer<float> gZ;    // binding 0: simplex zV (32/test) — in
RWStructuredBuffer<float> gDV;   // binding 1: apex dV (32/test)
RWStructuredBuffer<float> gCP;   // binding 2: closest point (32/test) — out
RWStructuredBuffer<float> gZOut; // binding 3: simplex zV (32/test) — out
RWStructuredBuffer<uint> gRet;   // binding 4: return feature code (per test)

uint next3(uint x) { return x == 2u ? 0u : x + 1u; }
uint prev3(uint x) { return x == 0u ? 2u : x - 1u; }
uint ffs(uint m) { return m == 0u ? 0u : firstbitlow(m) + 1u; }

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
    for (uint L = 0u; L < 32u; ++L) o[L] = a[cp0[L]] * b[cp1[L]] - a[cp1[L]] * b[cp0[L]];
}
void splatXW(float a[32], out float o[32]) { for (uint L = 0u; L < 32u; ++L) o[L] = a[L & 28u]; }
void splatVW(uint index, float a[32], out float o[32])
{
    for (uint L = 0u; L < 32u; ++L) o[L] = a[((index << 2) | (L & 3u)) & 31u];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ctClosestPointTet(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint base = t * 32u;

    float zV[32], dV[32];
    for (uint L = 0u; L < 32u; ++L) { zV[L] = gZ[base + L]; dV[L] = gDV[base + L]; }

    float closestPoint[32];
    for (uint L = 0u; L < 32u; ++L) closestPoint[L] = 0.0f;

    // n0V = splatV(4, zV); DEGENERATE if new vertex beyond old face
    float n0V[32]; splatVW(4u, zV, n0V);
    float dn0d[32]; dotW(n0V, dV, dn0d);
    float dn0z[32]; dotW(n0V, zV, dn0z);
    if (dn0d[0] > dn0z[0])
    {
        for (uint L = 0u; L < 32u; ++L) { gCP[base + L] = 0.0f; gZOut[base + L] = zV[L]; }
        gRet[t] = DEGENERATE;
        return;
    }

    uint cp0I[32], cp1I[32], vNextI[32], vPrevI[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        cp0I[L] = ((L & 28u) | next3(L & 3u)) & 31u;
        cp1I[L] = ((L & 28u) | prev3(L & 3u)) & 31u;
        vNextI[L] = (L > 7u) ? (L - 8u) : (L + 4u);
        vPrevI[L] = (L < 4u) ? (L + 8u) : (L - 4u);
    }

    // eV = dV - zV;  lD = splatX(dot(eV,eV))
    float eV[32]; for (uint L = 0u; L < 32u; ++L) eV[L] = dV[L] - zV[L];
    float dee[32]; dotW(eV, eV, dee);
    float lD[32]; splatXW(dee, lD);
    if ((lD[0] == 0.0f) || (lD[4] == 0.0f) || (lD[8] == 0.0f))
    {
        for (uint L = 0u; L < 32u; ++L) { gCP[base + L] = 0.0f; gZOut[base + L] = zV[L]; }
        gRet[t] = DEGENERATE;
        return;
    }

    // shortEdgeV = (shfl(lD,vNext) < lD) ? shfl(eV,vNext) : eV
    float shortEdgeV[32];
    for (uint L = 0u; L < 32u; ++L)
        shortEdgeV[L] = (lD[vNextI[L]] < lD[L]) ? eV[vNextI[L]] : eV[L];

    // nV = cross(shortEdgeV, shfl(zV,vNext) - zV, cp0I, cp1I)
    float d2[32]; for (uint L = 0u; L < 32u; ++L) d2[L] = zV[vNextI[L]] - zV[L];
    float nV[32]; crossW(shortEdgeV, d2, cp0I, cp1I, nV);

    float aS[32]; dotW(nV, dV, aS);
    float zvp[32]; for (uint L = 0u; L < 32u; ++L) zvp[L] = zV[vPrevI[L]];
    float gS[32]; dotW(zvp, nV, gS);

    // anyX<3>(aS <= gS) -> DEGENERATE (inversion/collapse)
    if ((aS[0] <= gS[0]) || (aS[4] <= gS[4]) || (aS[8] <= gS[8]))
    {
        for (uint L = 0u; L < 32u; ++L) { gCP[base + L] = 0.0f; gZOut[base + L] = zV[L]; }
        gRet[t] = DEGENERATE;
        return;
    }

    // nV.w (comp-3 lanes) = shfl_up(aS,3)
    for (uint L = 0u; L < 32u; ++L)
        if ((L & 3u) == 3u) nV[L] = aS[L - 3u]; // L>=3 always for comp-3 lanes

    // ---- inside region: allX<3>(aS>0) ----
    if ((aS[0] > 0.0f) && (aS[4] > 0.0f) && (aS[8] > 0.0f))
    {
        // setNormal123(zV, nV): lanes 20-31 = nV[L-20]
        for (uint L = 0u; L < 32u; ++L) if (L >= 20u) zV[L] = nV[L - 20u];
        for (uint L = 0u; L < 32u; ++L) { gCP[base + L] = closestPoint[L]; gZOut[base + L] = zV[L]; }
        gRet[t] = 0x2104u;
        return;
    }

    float pS[32]; dotW(eV, dV, pS);

    // ---- apex region: allX<3>(pS<=0) ----
    if ((pS[0] <= 0.0f) && (pS[4] <= 0.0f) && (pS[8] <= 0.0f))
    {
        for (uint L = 0u; L < 32u; ++L) { gCP[base + L] = dV[L]; gZOut[base + L] = zV[L]; }
        gRet[t] = 0x2131u;
        return;
    }

    // uiS = dot(cross(eV, nV), dV) <= 0;  ujS = dot(cross(shfl(eV,vNext), nV), dV) > 0
    float cen[32]; crossW(eV, nV, cp0I, cp1I, cen);
    float dcen[32]; dotW(cen, dV, dcen);
    bool uiS[32]; for (uint L = 0u; L < 32u; ++L) uiS[L] = dcen[L] <= 0.0f;

    float evn[32]; for (uint L = 0u; L < 32u; ++L) evn[L] = eV[vNextI[L]];
    float cevn[32]; crossW(evn, nV, cp0I, cp1I, cevn);
    float dcevn[32]; dotW(cevn, dV, dcevn);
    bool ujS[32]; for (uint L = 0u; L < 32u; ++L) ujS[L] = dcevn[L] > 0.0f;

    // ---- edge region: packXFlags<3>(uiS & shfl(ujS,vPrev) & pS>=0) ----
    uint emask = 0u;
    if (uiS[0] && ujS[vPrevI[0]] && (pS[0] >= 0.0f)) emask |= 1u;
    if (uiS[4] && ujS[vPrevI[4]] && (pS[4] >= 0.0f)) emask |= 2u;
    if (uiS[8] && ujS[vPrevI[8]] && (pS[8] >= 0.0f)) emask |= 4u;
    uint eid = ffs(emask);
    if (eid != 0u)
    {
        eid = eid - 1u;
        float pl[32]; for (uint L = 0u; L < 32u; ++L) pl[L] = pS[L] / lD[L];
        float spl[32]; splatXW(pl, spl);
        float tmp[32]; for (uint L = 0u; L < 32u; ++L) tmp[L] = dV[L] - eV[L] * spl[L];
        splatVW(eid, tmp, closestPoint);
        for (uint L = 0u; L < 32u; ++L) { gCP[base + L] = closestPoint[L]; gZOut[base + L] = zV[L]; }
        gRet[t] = (eid << 8) | 0x32u;
        return;
    }

    // ---- face region: packXFlags<3>(!uiS & !ujS & aS<=0) ----
    uint fmask = 0u;
    if (!uiS[0] && !ujS[0] && (aS[0] <= 0.0f)) fmask |= 1u;
    if (!uiS[4] && !ujS[4] && (aS[4] <= 0.0f)) fmask |= 2u;
    if (!uiS[8] && !ujS[8] && (aS[8] <= 0.0f)) fmask |= 4u;
    uint fid = ffs(fmask);
    if (fid != 0u)
    {
        fid = fid - 1u;
        float dnn[32]; dotW(nV, nV, dnn);
        float ad[32]; for (uint L = 0u; L < 32u; ++L) ad[L] = aS[L] / dnn[L];
        float sad[32]; splatXW(ad, sad);
        float tmp[32]; for (uint L = 0u; L < 32u; ++L) tmp[L] = nV[L] * sad[L];
        splatVW(fid, tmp, closestPoint);
        // setNormal0(zV, -nV, fid): lanes 16-19 = splatV(fid, -nV)
        for (uint L = 0u; L < 32u; ++L)
        {
            const uint src = ((fid << 2) | (L & 3u)) & 31u;
            const float tmpV = -nV[src];
            zV[L] = (L < 16u || L > 19u) ? zV[L] : tmpV;
        }
        for (uint L = 0u; L < 32u; ++L) { gCP[base + L] = closestPoint[L]; gZOut[base + L] = zV[L]; }
        gRet[t] = (next3(fid) << 12) | (fid << 8) | 0x33u;
        return;
    }

    for (uint L = 0u; L < 32u; ++L) { gCP[base + L] = 0.0f; gZOut[base + L] = zV[L]; }
    gRet[t] = DEGENERATE;
}
