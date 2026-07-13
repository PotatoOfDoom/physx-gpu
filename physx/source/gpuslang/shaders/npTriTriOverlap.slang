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

// npTriTriOverlap.slang
//
// Slang port of triangleTriangleOverlaps (convexMeshMidphase.cu L1386-1528):
// the SDF-free mesh-mesh overlap marking kernel.  One warp per pair,
// serialized.  The prep stage is ported verbatim:
//   - root BV32 node bounds of both meshes, reduced with exact fminf/fmaxf
//     over the (1 << mNbNodes) - 1 lane mask (values only; the minIndex /
//     maxIndex winner lanes are unused);
//   - 8 AABB corners per mesh with the CUDA bit pattern
//     pos.c = (lane & bit) ? MIN.c : MAX.c (bit set selects the MIN);
//   - vertex2Shape (multiply transform) with each mesh's scale, then the A
//     side is moved into B space via aToB = t1.transformInv(t0);
//   - initialDir = aToB.p normalized when |p|^2 > 0, else (0, 1, 0);
//     convergenceRatio = 1 - 0.000225f, minSep = contactDistance.
// The GJK over the two 8-point clouds is the tranche-129 cold-start driver,
// copied verbatim (the prep writes the clouds into the same gVA/gVB scratch
// buffers the driver reads).  overlaps = (result != GjkResult::eDISTANT) is
// derived from GjkOut.ret by the test.
// (The sibling kernel triangleTriangleCollision is NOT ported: its core calls
// doTriangleSDFCollision over tex3D SDF textures -- hardware filtering, not
// bit-exact-portable, same category as sdfCollision.cuh.)

static const uint DEGENERATE = 0xffffu;
static const float PX_MAX_F32 = 3.4028234663852886e38f;

// GjkResult::Enum
static const uint GJK_DISTANT = 0u;
static const uint GJK_CLOSE   = 1u;
static const uint GJK_OVERLAP = 2u;

struct GjkParams { uint numTests; uint maxV; uint pad0, pad1; }
struct TtoIn
{
    float4 t0q, t0p, t1q, t1p;        // PxsCachedTransform 0/1 (q, p)
    float4 scaleS0, scaleR0;          // mesh0 PxMeshScale (scale xyz, rotation quat)
    float4 scaleS1, scaleR1;
    uint4 counts;                     // x = mNbNodes0 (1..32), y = mNbNodes1
    float4 misc;                      // x = contactDistance
    float4 nMin0[32], nMax0[32];      // root BV32 node bounds, mesh 0
    float4 nMin1[32], nMax1[32];      // root BV32 node bounds, mesh 1
}
struct GjkOut
{
    uint ret; uint degenerate; uint size; uint pad;
    float lowerBound; float upperBound; float p1; float p2;
    float4 direction; float4 closestPointA; float4 closestPointB; float4 closestPointDir;
}

[[vk::push_constant]]
ConstantBuffer<GjkParams> gParams;

RWStructuredBuffer<float4> gVA;   // binding 0: scratch, written by the prep stage (8/test)
RWStructuredBuffer<float4> gVB;   // binding 1: convex B cloud (maxV/test)
RWStructuredBuffer<TtoIn> gIn;    // binding 2
RWStructuredBuffer<GjkOut> gOut;  // binding 3

// ---- layout primitives (per-lane serialised warp) --------------------------
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
void magW(float v[32], out float cm[32])
{
    float dv[32]; dotW(v, v, dv);
    for (uint L = 0u; L < 32u; ++L) cm[L] = sqrt(dv[L & 28u]);
}
uint encodeIndices(uint a, uint b) { return 0x3f800000u | ((b & 0xffu) << 8) | (a & 0xffu); }

// scalar helpers for the output stage
float3 crossX(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
float3 absX(float3 a) { return float3(abs(a.x), abs(a.y), abs(a.z)); }
float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 getNormalized(float3 v)
{
    const float m = dot3(v, v);
    if (m > 0.0f) { const float r = 1.0f / sqrt(m); return float3(v.x * r, v.y * r, v.z * r); }
    return float3(0.0f, 0.0f, 0.0f);
}

// ---- closestPointTri (tranche 122) : line simplex (slots 0,1) + apex cV ------
uint closestPointTriW(inout float zV[32], float cV[32], out float closestPoint[32])
{
    uint cp0I[32], cp1I[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        cp0I[L] = ((L & 28u) | next3(L & 3u)) & 31u;
        cp1I[L] = ((L & 28u) | prev3(L & 3u)) & 31u;
    }
    for (uint L = 0u; L < 32u; ++L) closestPoint[L] = 0.0f;

    float eV[32]; for (uint L = 0u; L < 32u; ++L) eV[L] = cV[L] - zV[L];
    float dee[32]; dotW(eV, eV, dee);
    float lD[32]; splatXW(dee, lD);
    if ((lD[0] == 0.0f) || (lD[4] == 0.0f)) return DEGENERATE;

    float shortEdgeV[32];
    for (uint L = 0u; L < 32u; ++L) shortEdgeV[L] = (lD[L ^ 4u] < lD[L]) ? eV[L ^ 4u] : eV[L];
    float d1[32]; for (uint L = 0u; L < 32u; ++L) d1[L] = zV[L ^ 4u] - zV[L];
    float mV[32]; crossW(d1, shortEdgeV, cp0I, cp1I, mV);
    float pS[32]; dotW(eV, cV, pS);
    float cme[32]; crossW(mV, eV, cp0I, cp1I, cme);
    float dcme[32]; dotW(cme, cV, dcme);
    bool uS[32]; for (uint L = 0u; L < 32u; ++L) uS[L] = dcme[L] <= 0.0f;

    if (!(uS[0] || uS[4]))
    {
        float dmc[32]; dotW(mV, cV, dmc);
        const float projD = dmc[0];
        float dmm[32]; dotW(mV, mV, dmm);
        float sxdmm[32]; splatXW(dmm, sxdmm);
        float tmp[32]; for (uint L = 0u; L < 32u; ++L) tmp[L] = mV[L] * projD / sxdmm[L];
        splatVW(0u, tmp, closestPoint);
        const bool flip = projD < 0.0f;
        for (uint L = 0u; L < 32u; ++L) if ((L & 3u) == 3u) mV[L] = projD;
        const uint index = flip ? 1u : 0u;
        for (uint L = 0u; L < 32u; ++L)
        {
            const uint src = ((index << 2) | (L & 3u)) & 31u;
            const float tmpV = mV[src];
            zV[L] = (L < 16u || L > 19u) ? zV[L] : tmpV;
        }
        return flip ? 0x0123u : 0x1023u;
    }

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
        return (eid << 8) | 0x22u;
    }

    for (uint L = 0u; L < 32u; ++L) closestPoint[L] = cV[L];
    return 0x0021u;
}

// ---- closestPointTet (tranche 123) : triangle simplex (slots 0,1,2) + apex dV -
uint closestPointTetW(inout float zV[32], float dV[32], out float closestPoint[32])
{
    for (uint L = 0u; L < 32u; ++L) closestPoint[L] = 0.0f;

    float n0V[32]; splatVW(4u, zV, n0V);
    float dn0d[32]; dotW(n0V, dV, dn0d);
    float dn0z[32]; dotW(n0V, zV, dn0z);
    if (dn0d[0] > dn0z[0]) return DEGENERATE;

    uint cp0I[32], cp1I[32], vNextI[32], vPrevI[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        cp0I[L] = ((L & 28u) | next3(L & 3u)) & 31u;
        cp1I[L] = ((L & 28u) | prev3(L & 3u)) & 31u;
        vNextI[L] = (L > 7u) ? (L - 8u) : (L + 4u);
        vPrevI[L] = (L < 4u) ? (L + 8u) : (L - 4u);
    }

    float eV[32]; for (uint L = 0u; L < 32u; ++L) eV[L] = dV[L] - zV[L];
    float dee[32]; dotW(eV, eV, dee);
    float lD[32]; splatXW(dee, lD);
    if ((lD[0] == 0.0f) || (lD[4] == 0.0f) || (lD[8] == 0.0f)) return DEGENERATE;

    float shortEdgeV[32];
    for (uint L = 0u; L < 32u; ++L) shortEdgeV[L] = (lD[vNextI[L]] < lD[L]) ? eV[vNextI[L]] : eV[L];
    float d2[32]; for (uint L = 0u; L < 32u; ++L) d2[L] = zV[vNextI[L]] - zV[L];
    float nV[32]; crossW(shortEdgeV, d2, cp0I, cp1I, nV);

    float aS[32]; dotW(nV, dV, aS);
    float zvp[32]; for (uint L = 0u; L < 32u; ++L) zvp[L] = zV[vPrevI[L]];
    float gS[32]; dotW(zvp, nV, gS);
    if ((aS[0] <= gS[0]) || (aS[4] <= gS[4]) || (aS[8] <= gS[8])) return DEGENERATE;

    for (uint L = 0u; L < 32u; ++L) if ((L & 3u) == 3u) nV[L] = aS[L - 3u];

    if ((aS[0] > 0.0f) && (aS[4] > 0.0f) && (aS[8] > 0.0f))
    {
        for (uint L = 0u; L < 32u; ++L) if (L >= 20u) zV[L] = nV[L - 20u];
        return 0x2104u;
    }

    float pS[32]; dotW(eV, dV, pS);
    if ((pS[0] <= 0.0f) && (pS[4] <= 0.0f) && (pS[8] <= 0.0f))
    {
        for (uint L = 0u; L < 32u; ++L) closestPoint[L] = dV[L];
        return 0x2131u;
    }

    float cen[32]; crossW(eV, nV, cp0I, cp1I, cen);
    float dcen[32]; dotW(cen, dV, dcen);
    bool uiS[32]; for (uint L = 0u; L < 32u; ++L) uiS[L] = dcen[L] <= 0.0f;
    float evn[32]; for (uint L = 0u; L < 32u; ++L) evn[L] = eV[vNextI[L]];
    float cevn[32]; crossW(evn, nV, cp0I, cp1I, cevn);
    float dcevn[32]; dotW(cevn, dV, dcevn);
    bool ujS[32]; for (uint L = 0u; L < 32u; ++L) ujS[L] = dcevn[L] > 0.0f;

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
        return (eid << 8) | 0x32u;
    }

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
        for (uint L = 0u; L < 32u; ++L)
        {
            const uint src = ((fid << 2) | (L & 3u)) & 31u;
            const float tmpV = -nV[src];
            zV[L] = (L < 16u || L > 19u) ? zV[L] : tmpV;
        }
        return (next3(fid) << 12) | (fid << 8) | 0x33u;
    }

    return DEGENERATE;
}

// ---- support mapping over the two clouds (tranche 116/127) ------------------
uint supportIdxA(float3 dir, uint vbase, uint nb)
{
    uint bestN = 0xFFFFFFFFu;
    float bestDist = -PX_MAX_F32;
    for (uint i = 0u; i < nb; i += 32u)
    {
        float d[32];
        for (uint lane = 0u; lane < 32u; ++lane)
        {
            const uint n = i + lane;
            d[lane] = (n < nb) ? dot3(dir, gVA[vbase + n].xyz) : -PX_MAX_F32;
        }
        float chunkMax = -PX_MAX_F32;
        for (uint lane = 0u; lane < 32u; ++lane) chunkMax = max(chunkMax, d[lane]);
        if (chunkMax > bestDist)
        {
            uint ll = 0u;
            for (uint lane = 0u; lane < 32u; ++lane) if (d[lane] == chunkMax) { ll = lane; break; }
            bestDist = chunkMax; bestN = i + ll;
        }
    }
    return bestN;
}
uint supportIdxB(float3 dir, uint vbase, uint nb)
{
    uint bestN = 0xFFFFFFFFu;
    float bestDist = -PX_MAX_F32;
    for (uint i = 0u; i < nb; i += 32u)
    {
        float d[32];
        for (uint lane = 0u; lane < 32u; ++lane)
        {
            const uint n = i + lane;
            d[lane] = (n < nb) ? dot3(dir, gVB[vbase + n].xyz) : -PX_MAX_F32;
        }
        float chunkMax = -PX_MAX_F32;
        for (uint lane = 0u; lane < 32u; ++lane) chunkMax = max(chunkMax, d[lane]);
        if (chunkMax > bestDist)
        {
            uint ll = 0u;
            for (uint lane = 0u; lane < 32u; ++lane) if (d[lane] == chunkMax) { ll = lane; break; }
            bestDist = chunkMax; bestN = i + ll;
        }
    }
    return bestN;
}

// ---- prep helpers (established exact forms) ---------------------------------
float4 quatConj(float4 q) { return float4(-q.x, -q.y, -q.z, q.w); }
float4 quatMulQ(float4 a, float4 b) // PxQuat::operator* (this = a)
{
    return float4(a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y,
                  a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z,
                  a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x,
                  a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z); // sequential like PxQuat.h:390 -- NOT regrouped
}
float3 quatRotQ(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
float3 quatRotInvQ(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
float3 vertex2ShapeQ(float3 v, float3 sc, float4 rot) // multiply transform
{
    const float3 r = quatRotQ(rot, v);
    return quatRotInvQ(rot, float3(r.x * sc.x, r.y * sc.y, r.z * sc.z));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void triTriOverlap(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint vbase = t * gParams.maxV;

    const TtoIn e = gIn[t];
    const uint nbA = 8u, nbB = 8u;
    const float minSep = e.misc.x;                       // contactDistance
    const float convergenceRatio = 1.0f - 0.000225f;

    // ---- prep (cu L1447-1514) ----
    // aToB = transform1.transformInv(transform0)
    const float4 aToBq = quatMulQ(quatConj(e.t1q), e.t0q);
    const float3 aToBp = quatRotInvQ(e.t1q, e.t0p.xyz - e.t1p.xyz);

    // root-node bounds over the (1 << mNbNodes) - 1 masked lanes, exact min/max
    float3 min0 = e.nMin0[0].xyz, max0 = e.nMax0[0].xyz;
    for (uint L = 1u; L < e.counts.x; ++L)
    {
        min0 = min(min0, e.nMin0[L].xyz); max0 = max(max0, e.nMax0[L].xyz);
    }
    float3 min1 = e.nMin1[0].xyz, max1 = e.nMax1[0].xyz;
    for (uint L = 1u; L < e.counts.y; ++L)
    {
        min1 = min(min1, e.nMin1[L].xyz); max1 = max(max1, e.nMax1[L].xyz);
    }

    // 8 corners each: a set bit selects the MIN component (cu L1477-1486),
    // then vertex2Shape scaling; the A side moves into B space
    for (uint i = 0u; i < 8u; ++i)
    {
        float3 pos0;
        pos0.x = ((i & 1u) != 0u) ? min0.x : max0.x;
        pos0.y = ((i & 2u) != 0u) ? min0.y : max0.y;
        pos0.z = ((i & 4u) != 0u) ? min0.z : max0.z;
        float3 v0 = vertex2ShapeQ(pos0, e.scaleS0.xyz, e.scaleR0);
        v0 = quatRotQ(aToBq, v0) + aToBp;
        gVA[vbase + i] = float4(v0, 0.0f);

        float3 pos1;
        pos1.x = ((i & 1u) != 0u) ? min1.x : max1.x;
        pos1.y = ((i & 2u) != 0u) ? min1.y : max1.y;
        pos1.z = ((i & 4u) != 0u) ? min1.z : max1.z;
        gVB[vbase + i] = float4(vertex2ShapeQ(pos1, e.scaleS1.xyz, e.scaleR1), 0.0f);
    }

    const float sqDist = dot3(aToBp, aToBp);
    const float3 initialUnitDir = sqDist > 0.0f ? getNormalized(aToBp) : float3(0.0f, 1.0f, 0.0f);

    // ---- cold-start init (cacheMask = 0) ----
    float simplexV[32]; for (uint L = 0u; L < 32u; ++L) simplexV[L] = 0.0f;
    float closestPointV[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        const uint c = L & 3u;
        closestPointV[L] = (c == 0u) ? initialUnitDir.x : (c == 1u) ? initialUnitDir.y : (c == 2u) ? initialUnitDir.z : PX_MAX_F32;
    }
    float upperBound = PX_MAX_F32, lowerBound = -PX_MAX_F32;
    float cmA[32]; magW(closestPointV, cmA);
    // Note (deliberate hardening of a CUDA UB): the CUDA declares `bool degenerate;`
    // uninitialised (gjk.cuh:268) and only assigns it at :337, inside the loop and
    // after the support-step convergence break.  On the first iteration, a distant
    // pair whose first support already gives lowerBound>minSep breaks at :307-308
    // BEFORE :337, so `output.degenerate = degenerate` (:375) reads an indeterminate
    // register.  Since PhysX documents that `degenerate` can only accompany a result
    // of CLOSE (schlockShared.h:202-203) — the caller ignores it for DISTANT/OVERLAP
    // — we make it well-defined here as false.  This is the sole point where the port
    // is intentionally not bit-identical to (nondeterministic) CUDA; it is unobservable
    // to any correct caller.
    bool degenerate = false;

    int size = 0;
    int backupSize = 0;
    float backupSimplexV[32]; for (uint L = 0u; L < 32u; ++L) backupSimplexV[L] = 0.0f;
    float backupClosestV[32]; for (uint L = 0u; L < 32u; ++L) backupClosestV[L] = 0.0f;
    float3 directionOut = float3(0.0f, 0.0f, 0.0f);

    // ---- main do-while ----
    bool run = true;
    [loop] while (run)
    {
        float dV[32]; for (uint L = 0u; L < 32u; ++L) dV[L] = closestPointV[L] / cmA[L];

        // cold path: support vertices
        const float3 dvVec = float3(dV[0], dV[1], dV[2]);
        const uint indexA = supportIdxA(float3(-dvVec.x, -dvVec.y, -dvVec.z), vbase, nbA);
        const uint indexB = supportIdxB(dvVec, vbase, nbB);
        const float3 vA = gVA[vbase + indexA].xyz;
        const float3 vB = gVB[vbase + indexB].xyz;
        const float3 sv = vA - vB;
        float supportVertexV[32];
        for (uint L = 0u; L < 32u; ++L)
        {
            const uint c = L & 3u;
            supportVertexV[L] = (c == 0u) ? sv.x : (c == 1u) ? sv.y : (c == 2u) ? sv.z : 0.0f;
        }
        const float encoded = asfloat(encodeIndices(indexA, indexB));
        for (uint L = 0u; L < 32u; ++L)
            if ((L >> 2) == (uint)size) simplexV[L] = ((L & 3u) < 3u) ? supportVertexV[L] : encoded;

        float dds[32]; dotW(dV, supportVertexV, dds);
        const float lb = dds[0];
        if (lb > lowerBound)
        {
            directionOut = float3(-dV[0], -dV[1], -dV[2]);
            lowerBound = lb;
            if (lowerBound > minSep || lowerBound >= upperBound * convergenceRatio) { run = false; break; }
        }

        // switch(size) dispatch
        uint f = 0u;
        float pointV[32];
        if (size == 0)
        {
            f = 0x2101u;
            for (uint L = 0u; L < 32u; ++L) pointV[L] = supportVertexV[L];
        }
        else if (size == 1)
        {
            float aV[32]; splatVW(0u, simplexV, aV);
            float lV[32]; for (uint L = 0u; L < 32u; ++L) lV[L] = supportVertexV[L] - aV[L];
            float dll[32]; dotW(lV, lV, dll);
            float mm[32]; splatXW(dll, mm);
            float dal[32]; dotW(aV, lV, dal);
            float xx[32]; splatXW(dal, xx); for (uint L = 0u; L < 32u; ++L) xx[L] = -xx[L];
            if (mm[0] == 0.0f) { f = DEGENERATE; for (uint L = 0u; L < 32u; ++L) pointV[L] = 0.0f; }
            else if (xx[0] < mm[0]) { f = 0x2102u; for (uint L = 0u; L < 32u; ++L) pointV[L] = aV[L] + lV[L] * xx[L] / mm[L]; }
            else { f = 0x2111u; for (uint L = 0u; L < 32u; ++L) pointV[L] = supportVertexV[L]; }
        }
        else if (size == 2)
        {
            f = closestPointTriW(simplexV, supportVertexV, pointV);
        }
        else
        {
            f = closestPointTetW(simplexV, supportVertexV, pointV);
        }

        magW(pointV, cmA);
        const int newSize = (int)(f & 0xfu);
        degenerate = (f == DEGENERATE) || (cmA[0] >= upperBound && newSize <= size);
        if (degenerate) { run = false; break; } // cold start: warmStarted is always false -> break

        backupSize = (cmA[0] >= upperBound) ? size : 0;
        if (backupSize != 0)
        {
            for (uint L = 0u; L < 32u; ++L) { backupSimplexV[L] = simplexV[L]; backupClosestV[L] = closestPointV[L]; }
        }
        else
        {
            upperBound = cmA[0];
        }

        for (uint L = 0u; L < 32u; ++L) closestPointV[L] = pointV[L];
        size = newSize;

        // simplex permute
        float tmpP[32];
        for (uint L = 0u; L < 32u; ++L)
        {
            const uint sourceIndex = (f >> ((L & 28u) + 4u)) & 0xfu;
            tmpP[L] = simplexV[((sourceIndex << 2) | (L & 3u)) & 31u];
        }
        for (uint L = 0u; L < 32u; ++L) if (L < 12u) simplexV[L] = tmpP[L];

        if (!(upperBound > 0.0f && lowerBound < upperBound * convergenceRatio)) run = false;
    }

    if (backupSize > 0)
    {
        for (uint L = 0u; L < 32u; ++L) { simplexV[L] = backupSimplexV[L]; closestPointV[L] = backupClosestV[L]; }
        size = backupSize;
    }

    // ---- output stage ----
    GjkOut o;
    o.degenerate = degenerate ? 1u : 0u;
    o.lowerBound = lowerBound;
    o.upperBound = upperBound;
    o.size = (uint)size;
    o.pad = 0u; o.p1 = 0.0f; o.p2 = 0.0f;
    o.direction = float4(directionOut, 0.0f);
    o.closestPointA = float4(0.0f, 0.0f, 0.0f, 0.0f);
    o.closestPointB = float4(0.0f, 0.0f, 0.0f, 0.0f);
    o.closestPointDir = float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (upperBound == 0.0f) { o.ret = GJK_OVERLAP; gOut[t] = o; return; }
    if (lowerBound > minSep) { o.ret = GJK_DISTANT; gOut[t] = o; return; }

    const float3 cp = float3(closestPointV[0], closestPointV[1], closestPointV[2]);
    const float3 dir = -getNormalized(cp);
    o.closestPointDir = float4(dir, 0.0f);

    const uint a0 = asuint(simplexV[3]) & 0xffu;
    const uint a1 = asuint(simplexV[7]) & 0xffu;
    const uint a2 = asuint(simplexV[11]) & 0xffu;

    float3 cpA;
    if (size == 3)
    {
        const float3 A = float3(simplexV[0], simplexV[1], simplexV[2]);
        const float3 B = float3(simplexV[4], simplexV[5], simplexV[6]);
        const float3 C = float3(simplexV[8], simplexV[9], simplexV[10]);
        const float3 m = crossX(B - A, C - A);
        const float3 ma = absX(m);
        const float3 s1 = crossX(B - cp, C - cp);
        const float3 s2 = crossX(C - cp, A - cp);
        float d, u, v;
        if (ma.x >= ma.y && ma.x >= ma.z) { d = 1.0f / m.x; u = s1.x * d; v = s2.x * d; }
        else if (ma.y >= ma.z)            { d = 1.0f / m.y; u = s1.y * d; v = s2.y * d; }
        else                              { d = 1.0f / m.z; u = s1.z * d; v = s2.z * d; }
        const float w = 1.0f - u - v;
        cpA = u * gVA[vbase + a0].xyz + v * gVA[vbase + a1].xyz + w * gVA[vbase + a2].xyz;
    }
    else if (size == 2)
    {
        const float3 V0 = float3(simplexV[0], simplexV[1], simplexV[2]);
        const float3 V1 = float3(simplexV[4], simplexV[5], simplexV[6]);
        const float3 dv = V1 - V0;
        const float q = -dot3(V0, dv) / dot3(dv, dv);
        cpA = (1.0f - q) * gVA[vbase + a0].xyz + q * gVA[vbase + a1].xyz;
    }
    else
    {
        cpA = gVA[vbase + a0].xyz;
    }

    o.closestPointA = float4(cpA, 0.0f);
    o.closestPointB = float4(cpA + upperBound * dir, 0.0f);
    o.ret = GJK_CLOSE;
    gOut[t] = o;
}
