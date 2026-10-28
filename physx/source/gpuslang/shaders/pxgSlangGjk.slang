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

// pxgSlangGjk — the cold-start GJK driver (gjk.cuh :235-418), extracted from npGjk.slang so
// that BOTH the standalone leaf test (ctGjk) and the engine-path contact-gen kernels run the
// SAME implementation.
//
// WHY EXTRACT RATHER THAN COPY.  `squawk::gjk` is ONE CUDA function called by every
// deformable-vs-convex path in the narrowphase.  That is the case the port's sharing rule
// says to share (pxgSlangFemRigidAttach's header states it: one CUDA function used by both
// -> share it, or the copies rot; two CUDA functions that happen to agree -> keep two).
// Five hundred lines of simplex bookkeeping is precisely what must not exist twice.
//
// WHAT CHANGED IN THE MOVE.  Nothing but where the vertex clouds come from.  npGjk.slang
// read them out of two RWStructuredBuffers at `vbase + i`; the core now takes them as
// GjkCloud values, so a caller with device pointers, a caller with descriptor-set buffers
// and a caller with hand-built local arrays can all use it.  Every arithmetic line, every
// branch and the deliberate `degenerate = false` hardening are byte-for-byte the ones that
// were already verified bit-exact against the CPU reference.
//
// One thread simulates the 32-lane warp with per-lane arrays (lane L = component L&3 of
// vector L>>2), exactly as the leaf shader did.
module pxgSlangGjk;

import pxgSlangGjkShared;   // next3/prev3/ffs/encodeIndices, crossX/absX/dot3/getNormalized

// The two convex point clouds.  64 is CONVEX_MAX_VERTICES_POLYGONS, the largest hull the
// narrowphase admits; a tetrahedron uses four of them.
public static const uint GJK_MAX_CLOUD = 64u;
public struct GjkCloud
{
    public float3 v[GJK_MAX_CLOUD];
    public uint n;
}

public struct GjkOut
{
    public uint ret;
    public uint degenerate;
    public uint size;
    public uint pad;
    public float lowerBound;
    public float upperBound;
    public float p1;
    public float p2;
    public float4 direction;
    public float4 closestPointA;
    public float4 closestPointB;
    public float4 closestPointDir;
}


// npGjk.slang
//
// Slang port of the full cold-start gjk() driver (gjk.cuh :235-418) — tranche
// 129.  This wires together every previously-ported (and individually bit-exact
// verified) sub-piece into the complete GJK do-while: initialisation from the
// initial search direction, the support-vertex expansion + lower-bound update
// (tranche 127), the switch(size) dispatch to the 1/2/3-simplex closest-point
// cores (tranches 124/122/123), the per-iteration simplex permute + state update
// (tranche 126), the keep-best backup restore, and the result/output stage
// (tranche 125).  The warm-start cache path (tranche 128) is not exercised here
// (cold start: cacheMask=0), but is retained in the standalone shader.
//
// One thread simulates the 32-lane warp with per-lane arrays (lane L = component
// L&3 of vector L>>2).  With integer convex-hull vertices every dot/cross is
// exact, magnitude is a correctly-rounded sqrt and every division is a
// correctly-rounded OpFDiv, so the entire iteration — its branch decisions, its
// iteration count, and its witness points — is bit-for-bit reproducible against
// the CPU reference.

static const uint DEGENERATE = 0xffffu;
static const float PX_MAX_F32 = 3.4028234663852886e38f;

// GjkResult::Enum
public static const uint GJK_DISTANT = 0u;
public static const uint GJK_CLOSE   = 1u;
public static const uint GJK_OVERLAP = 2u;






// ---- layout primitives (per-lane serialised warp) --------------------------
// next3 / prev3 / ffs / encodeIndices and the scalar helpers crossX / absX / dot3 /
// getNormalized live in pxgSlangGjkShared (moved there verbatim, so the cooperative
// driver pxgSlangGjkWarp compiles the same bodies).

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
uint supportIdxA(float3 dir, GjkCloud a)
{
    const uint nbLocal = a.n;
    uint bestN = 0xFFFFFFFFu;
    float bestDist = -PX_MAX_F32;
    for (uint i = 0u; i < nbLocal; i += 32u)
    {
        float d[32];
        for (uint lane = 0u; lane < 32u; ++lane)
        {
            const uint n = i + lane;
            d[lane] = (n < nbLocal) ? dot3(dir, a.v[n]) : -PX_MAX_F32;
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
uint supportIdxB(float3 dir, GjkCloud b)
{
    const uint nbLocal = b.n;
    uint bestN = 0xFFFFFFFFu;
    float bestDist = -PX_MAX_F32;
    for (uint i = 0u; i < nbLocal; i += 32u)
    {
        float d[32];
        for (uint lane = 0u; lane < 32u; ++lane)
        {
            const uint n = i + lane;
            d[lane] = (n < nbLocal) ? dot3(dir, b.v[n]) : -PX_MAX_F32;
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


// The driver.  Returns through `o`; the return value is o.ret (GJK_DISTANT/CLOSE/OVERLAP).
// `cachedSimplex` is CUDA's GjkCachedData: the final simplex, which EPA starts from.  The
// leaf test never looked at it, so npGjk.slang never produced it; tetPrimitivesCollide2 must.
public uint gjkRun(GjkCloud a, GjkCloud b, float3 initialUnitDir, float minSep,
                   float convergenceRatio, out GjkOut o, out float cachedSimplex[32])
{
    for (uint ci = 0u; ci < 32u; ++ci) cachedSimplex[ci] = 0.0f;


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
        const uint indexA = supportIdxA(float3(-dvVec.x, -dvVec.y, -dvVec.z), a);
        const uint indexB = supportIdxB(dvVec, b);
        const float3 vA = a.v[indexA];
        const float3 vB = b.v[indexB];
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
    o.pad = 0u;
    for (uint ci = 0u; ci < 32u; ++ci) cachedSimplex[ci] = simplexV[ci];
    o.degenerate = degenerate ? 1u : 0u;
    o.lowerBound = lowerBound;
    o.upperBound = upperBound;
    o.size = (uint)size;
    o.pad = 0u; o.p1 = 0.0f; o.p2 = 0.0f;
    o.direction = float4(directionOut, 0.0f);
    o.closestPointA = float4(0.0f, 0.0f, 0.0f, 0.0f);
    o.closestPointB = float4(0.0f, 0.0f, 0.0f, 0.0f);
    o.closestPointDir = float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (upperBound == 0.0f) { o.ret = GJK_OVERLAP; return o.ret; }
    if (lowerBound > minSep) { o.ret = GJK_DISTANT; return o.ret; }

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
        cpA = u * a.v[a0] + v * a.v[a1] + w * a.v[a2];
    }
    else if (size == 2)
    {
        const float3 V0 = float3(simplexV[0], simplexV[1], simplexV[2]);
        const float3 V1 = float3(simplexV[4], simplexV[5], simplexV[6]);
        const float3 dv = V1 - V0;
        const float q = -dot3(V0, dv) / dot3(dv, dv);
        cpA = (1.0f - q) * a.v[a0] + q * a.v[a1];
    }
    else
    {
        cpA = a.v[a0];
    }

    o.closestPointA = float4(cpA, 0.0f);
    o.closestPointB = float4(cpA + upperBound * dir, 0.0f);
    o.ret = GJK_CLOSE;

    return o.ret;
}
