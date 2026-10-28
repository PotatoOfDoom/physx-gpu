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

// pxgSlangGjkWarp -- tetPrimitivesCollide2 (convexNpCommon.h:212) with squawk::gjk
// (gjk.cuh) and squawk::epa (epa.cuh) in their WARP-COOPERATIVE form: a 32-thread
// group, thread L holding what CUDA lane L holds.
//
// ---------------------------------------------------------------------------
// WHY A SECOND DRIVER
// ---------------------------------------------------------------------------
// pxgSlangGjk / pxgSlangEpa / pxgSlangTetCollide run the whole query on ONE thread
// that simulates the warp with 32-element per-lane arrays.  That is the verified
// reference and it stays (five kernels call it).  But on a GPU it is the slowest
// possible shape: every warp step is a 32-iteration loop, the per-lane arrays are
// dynamically indexed (crossW's a[cp0[L]], splatVW, the permute) and so live in
// scratch memory, and 31 of 32 threads of the group wait.  The CUDA does one step
// per lane in a register and exchanges with shuffles.
//
// This module is the CUDA's shape: per-lane state in registers, the clouds and the
// EPA polytope in groupshared (the CUDA's TetCollideScratch / EpaScratch), and every
// __shfl_sync / __ballot_sync an EXCHANGE through groupshared.
//
// ---------------------------------------------------------------------------
// BIT-EXACT AGAINST THE SERIAL DRIVER, BY CONSTRUCTION AND BY TEST
// ---------------------------------------------------------------------------
// It is a lane-for-lane transcription of the SERIAL module, not of the .cu: every
// `for (L) o[L] = expr(a[L], a[g(L)])` there is `o = expr(a, exchange(a, g(lane)))`
// here, with the same operands in the same association.  Values the serial code
// reads at a FIXED lane (lD[0], aS[4], dmc[0], ...) are read back from the exchange
// buffer by every thread, and a dotW result at a fixed lane is recomputed from the
// buffered products with dotW's own expression (gwDotAt) -- the same three
// operands, the same two additions.  So each thread holds the bits the serial
// arrays hold at its index, and every uniform decision sees the same bits.
// Ordered side effects keep the serial order: the silhouette's linked-list writes,
// which can collide on a degenerate polytope, are applied by thread 0 in lane order
// (the serial loop's last-writer-wins), and the EPA minimum keeps the serial select
// chain (P9.86) rather than a butterfly.  npTetCollide_warpDifferential compares the
// two drivers bit for bit over 4 600 random queries, EPA starts from both
// seeds, the anomaly gate and the degenerate-EPA retry included.
//
// ---------------------------------------------------------------------------
// THE EXCHANGE AND ITS ONE BARRIER
// ---------------------------------------------------------------------------
// An exchange is: every thread gwPut()s its lane value(s) into the CURRENT buffer,
// gwSync() (one barrier, then the parity flips), then gwGet() reads any lane from
// the buffer just published.  The buffers alternate, so the next exchange writes
// the OTHER one and needs no second barrier: a thread writing buffer p at exchange
// k+2 has passed exchange k+1's barrier, which every thread reached only after its
// exchange-k reads of buffer p.  This requires every thread to run the same
// sequence of exchanges -- which is the barrier rule anyway (every branch around an
// exchange is on a value read back from a buffer at a FIXED lane, hence
// group-uniform).  `par` carries the parity; it is never derived from the lane.
//
// THE CALLER'S CONTRACT: the group is exactly one warp, [numthreads(32,1,1)]; the
// two clouds are written with gwCloudPut BEHIND a barrier that follows any earlier
// read of them (the caller's per-pair barrier), and gwTetPrimitivesCollide2's first
// barrier publishes them.  The results are group-uniform.
module pxgSlangGjkWarp;

import pxgSlangGjk;          // GjkOut, GJK_DISTANT/CLOSE/OVERLAP, GJK_MAX_CLOUD
import pxgSlangEpa;          // EpaOut
import pxgSlangGjkShared;    // the helpers both serial drivers use, the same bodies
import pxgSlangTetCollide;   // TETC_CONVERGENCE_RATIO

static const uint GW_DEGENERATE = 0xffffu;             // gjk.cuh DEGENERATE
static const float GW_MAX_F32 = 3.4028234663852886e38f;
static const uint GW_EPA_MAX_VERTS = 18u;              // schlock::EPA_MAX_VERTS

// ---- groupshared state ---------------------------------------------------------
groupshared float sGwF[2][4][32];   // exchange buffers [parity][channel][lane]
groupshared uint sGwU[2][2][32];
groupshared float4 sGwCloud[2][GJK_MAX_CLOUD];   // [0] = A (the rigid), [1] = B (the tet)
// EpaScratch: the polytope's vertices, their support indices, and the silhouette's
// linked list (nxt), cone fix-up table (edg) and flattened loop (lp).
groupshared float sGwEpaVx[GW_EPA_MAX_VERTS];
groupshared float sGwEpaVy[GW_EPA_MAX_VERTS];
groupshared float sGwEpaVz[GW_EPA_MAX_VERTS];
groupshared uint sGwEpaIdx[GW_EPA_MAX_VERTS];
groupshared uint sGwEpaNxt[GW_EPA_MAX_VERTS];
groupshared uint sGwEpaEdg[GW_EPA_MAX_VERTS];
groupshared uint sGwEpaLp[GW_EPA_MAX_VERTS];
groupshared int sGwEpaNbLoop;

// ---- the exchange ----------------------------------------------------------------
void gwPut(uint par, uint ch, uint lane, float v) { sGwF[par][ch][lane] = v; }
void gwPutU(uint par, uint ch, uint lane, uint v) { sGwU[par][ch][lane] = v; }
void gwSync(inout uint par)
{
    GroupMemoryBarrierWithGroupSync();
    par ^= 1u;
}
// `src & 31u` keeps the address in range for the lanes a per-lane guard discards
// (P9.22: llvmpipe issues guarded groupshared loads unpredicated as well).
float gwGet(uint par, uint ch, uint src) { return sGwF[par ^ 1u][ch][src & 31u]; }
uint gwGetU(uint par, uint ch, uint src) { return sGwU[par ^ 1u][ch][src & 31u]; }

// dotW's result at lane k, from the products x = a * b the lanes put into `ch`:
// x[k] + s1 + s2 with dotW's boundary rule -- the serial expression exactly.
float gwDotAt(uint par, uint ch, uint k)
{
    const float x = gwGet(par, ch, k);
    const float s1 = (k + 1u < 32u) ? gwGet(par, ch, k + 1u) : x;
    const float s2 = (k + 2u < 32u) ? gwGet(par, ch, k + 2u) : x;
    return x + s1 + s2;
}

// ---- the clouds --------------------------------------------------------------------
public void gwCloudPut(uint cloud, uint i, float3 v)
{
    sGwCloud[cloud][min(i, GJK_MAX_CLOUD - 1u)] = float4(v, 0.0f);
}
float3 gwCloudVert(uint cloud, uint i) { return sGwCloud[cloud][min(i, GJK_MAX_CLOUD - 1u)].xyz; }
float3 gwEpaVert(uint i)
{
    const uint k = min(i, GW_EPA_MAX_VERTS - 1u);
    return float3(sGwEpaVx[k], sGwEpaVy[k], sGwEpaVz[k]);
}

// supportIdxA(dirA) over cloud A and supportIdxB(dirB) over cloud B in lockstep:
// lane l takes vertex i + l of each 32-vertex chunk, and every thread then folds the
// chunk exactly as the serial loop does (running max from -PX_MAX_F32, lowest lane
// on a tie).  A chunk past one cloud's count is skipped for that cloud, so each
// result is the serial function's.
void gwSupportPair(inout uint par, uint lane, float3 dirA, uint nbA, float3 dirB, uint nbB,
                   out uint indexA, out uint indexB)
{
    uint bestNA = 0xFFFFFFFFu, bestNB = 0xFFFFFFFFu;
    float bestDistA = -GW_MAX_F32, bestDistB = -GW_MAX_F32;
    const uint nb = max(nbA, nbB);
    for (uint i = 0u; i < nb; i += 32u)
    {
        const uint n = i + lane;
        const float dA = (n < nbA) ? dot3(dirA, gwCloudVert(0u, n)) : -GW_MAX_F32;
        const float dB = (n < nbB) ? dot3(dirB, gwCloudVert(1u, n)) : -GW_MAX_F32;
        gwPut(par, 0u, lane, dA);
        gwPut(par, 1u, lane, dB);
        gwSync(par);
        if (i < nbA)
        {
            float chunkMax = -GW_MAX_F32;
            for (uint l = 0u; l < 32u; ++l) chunkMax = max(chunkMax, gwGet(par, 0u, l));
            if (chunkMax > bestDistA)
            {
                uint ll = 0u;
                for (uint l = 0u; l < 32u; ++l) if (gwGet(par, 0u, l) == chunkMax) { ll = l; break; }
                bestDistA = chunkMax; bestNA = i + ll;
            }
        }
        if (i < nbB)
        {
            float chunkMax = -GW_MAX_F32;
            for (uint l = 0u; l < 32u; ++l) chunkMax = max(chunkMax, gwGet(par, 1u, l));
            if (chunkMax > bestDistB)
            {
                uint ll = 0u;
                for (uint l = 0u; l < 32u; ++l) if (gwGet(par, 1u, l) == chunkMax) { ll = l; break; }
                bestDistB = chunkMax; bestNB = i + ll;
            }
        }
    }
    indexA = bestNA;
    indexB = bestNB;
}

// ---- closestPointTri (pxgSlangGjk closestPointTriW, gjk.cuh) ------------------------
// z is this lane's simplexV, cv its supportVertexV.
uint gwClosestPointTri(inout uint par, uint lane, inout float z, float cv, out float closestPoint)
{
    const uint L = lane;
    const uint cp0 = ((L & 28u) | next3(L & 3u)) & 31u;
    const uint cp1 = ((L & 28u) | prev3(L & 3u)) & 31u;
    closestPoint = 0.0f;

    const float eV = cv - z;
    gwPut(par, 0u, L, eV * eV);
    gwPut(par, 1u, L, eV);
    gwPut(par, 2u, L, z);
    gwSync(par);
    const float lD = gwDotAt(par, 0u, L & 28u);                  // splatXW(dotW(eV, eV))
    if ((gwDotAt(par, 0u, 0u) == 0.0f) || (gwDotAt(par, 0u, 4u) == 0.0f)) return GW_DEGENERATE;
    const float lDx = gwDotAt(par, 0u, (L ^ 4u) & 28u);          // lD[L ^ 4]
    const float shortEdgeV = (lDx < lD) ? gwGet(par, 1u, L ^ 4u) : eV;
    const float d1 = gwGet(par, 2u, L ^ 4u) - z;

    gwPut(par, 0u, L, d1);
    gwPut(par, 1u, L, shortEdgeV);
    gwPut(par, 2u, L, eV * cv);
    gwSync(par);
    const float mV = gwGet(par, 0u, cp0) * gwGet(par, 1u, cp1) - gwGet(par, 0u, cp1) * gwGet(par, 1u, cp0);
    const float pS0 = gwDotAt(par, 2u, 0u);                      // dotW(eV, cV)[0]
    const float pS4 = gwDotAt(par, 2u, 4u);
    const float sxpS = gwDotAt(par, 2u, L & 28u);                // splatXW(pS)

    gwPut(par, 0u, L, mV);
    gwPut(par, 1u, L, eV);
    gwSync(par);
    const float cme = gwGet(par, 0u, cp0) * gwGet(par, 1u, cp1) - gwGet(par, 0u, cp1) * gwGet(par, 1u, cp0);

    gwPut(par, 0u, L, cme * cv);
    gwSync(par);
    const bool uS0 = gwDotAt(par, 0u, 0u) <= 0.0f;
    const bool uS4 = gwDotAt(par, 0u, 4u) <= 0.0f;

    if (!(uS0 || uS4))
    {
        gwPut(par, 0u, L, mV * cv);
        gwPut(par, 1u, L, mV * mV);
        gwSync(par);
        const float projD = gwDotAt(par, 0u, 0u);
        const float sxdmm = gwDotAt(par, 1u, L & 28u);
        const float tmp = mV * projD / sxdmm;
        const bool flip = projD < 0.0f;
        const float mVn = ((L & 3u) == 3u) ? projD : mV;
        const uint index = flip ? 1u : 0u;
        gwPut(par, 0u, L, tmp);
        gwPut(par, 1u, L, mVn);
        gwSync(par);
        closestPoint = gwGet(par, 0u, L & 3u);                    // splatVW(0, tmp)
        const float tmpV = gwGet(par, 1u, ((index << 2) | (L & 3u)) & 31u);
        z = (L < 16u || L > 19u) ? z : tmpV;
        return flip ? 0x0123u : 0x1023u;
    }

    uint emask = 0u;
    if (uS0 && (pS0 > 0.0f)) emask |= 1u;
    if (uS4 && (pS4 > 0.0f)) emask |= 2u;
    uint eid = ffs(emask);
    if (eid != 0u)
    {
        eid = eid - 1u;
        const float tmp = cv - eV * sxpS / lD;
        gwPut(par, 0u, L, tmp);
        gwSync(par);
        closestPoint = gwGet(par, 0u, ((eid << 2) | (L & 3u)) & 31u);
        return (eid << 8) | 0x22u;
    }

    closestPoint = cv;
    return 0x0021u;
}

// ---- closestPointTet (pxgSlangGjk closestPointTetW, gjk.cuh) ------------------------
uint gwClosestPointTet(inout uint par, uint lane, inout float z, float dv, out float closestPoint)
{
    const uint L = lane;
    closestPoint = 0.0f;

    gwPut(par, 0u, L, z);
    gwSync(par);
    const float n0V = gwGet(par, 0u, (16u | (L & 3u)) & 31u);     // splatVW(4, zV)
    gwPut(par, 0u, L, n0V * dv);
    gwPut(par, 1u, L, n0V * z);
    gwSync(par);
    if (gwDotAt(par, 0u, 0u) > gwDotAt(par, 1u, 0u)) return GW_DEGENERATE;

    const uint cp0 = ((L & 28u) | next3(L & 3u)) & 31u;
    const uint cp1 = ((L & 28u) | prev3(L & 3u)) & 31u;
    const uint vNext = (L > 7u) ? (L - 8u) : (L + 4u);
    const uint vPrev = (L < 4u) ? (L + 8u) : (L - 4u);

    const float eV = dv - z;
    gwPut(par, 0u, L, eV * eV);
    gwPut(par, 1u, L, eV);
    gwPut(par, 2u, L, z);
    gwSync(par);
    const float lD = gwDotAt(par, 0u, L & 28u);
    if ((gwDotAt(par, 0u, 0u) == 0.0f) || (gwDotAt(par, 0u, 4u) == 0.0f) || (gwDotAt(par, 0u, 8u) == 0.0f))
        return GW_DEGENERATE;
    const float lDn = gwDotAt(par, 0u, vNext & 28u);            // lD[vNextI[L]]
    const float shortEdgeV = (lDn < lD) ? gwGet(par, 1u, vNext) : eV;
    const float d2 = gwGet(par, 2u, vNext) - z;
    const float zvp = gwGet(par, 2u, vPrev);                     // zV[vPrevI[L]]

    gwPut(par, 0u, L, shortEdgeV);
    gwPut(par, 1u, L, d2);
    gwSync(par);
    float nV = gwGet(par, 0u, cp0) * gwGet(par, 1u, cp1) - gwGet(par, 0u, cp1) * gwGet(par, 1u, cp0);

    gwPut(par, 0u, L, nV * dv);                                  // aS = dotW(nV, dV)
    gwPut(par, 1u, L, zvp * nV);                                 // gS = dotW(zvp, nV)
    gwSync(par);
    const float aS0 = gwDotAt(par, 0u, 0u), aS4 = gwDotAt(par, 0u, 4u), aS8 = gwDotAt(par, 0u, 8u);
    if ((aS0 <= gwDotAt(par, 1u, 0u)) || (aS4 <= gwDotAt(par, 1u, 4u)) || (aS8 <= gwDotAt(par, 1u, 8u)))
        return GW_DEGENERATE;
    const float aSL28 = gwDotAt(par, 0u, L & 28u);
    const float aSm3 = gwDotAt(par, 0u, L - 3u);                // aS[L - 3], used where (L & 3) == 3
    if ((L & 3u) == 3u) nV = aSm3;

    if ((aS0 > 0.0f) && (aS4 > 0.0f) && (aS8 > 0.0f))
    {
        gwPut(par, 0u, L, nV);
        gwSync(par);
        const float nvv = gwGet(par, 0u, L - 20u);
        if (L >= 20u) z = nvv;
        return 0x2104u;
    }

    gwPut(par, 0u, L, eV * dv);                                  // pS = dotW(eV, dV)
    gwPut(par, 1u, L, eV);
    gwPut(par, 2u, L, nV);
    gwSync(par);
    const float pS0 = gwDotAt(par, 0u, 0u), pS4 = gwDotAt(par, 0u, 4u), pS8 = gwDotAt(par, 0u, 8u);
    const float pSL28 = gwDotAt(par, 0u, L & 28u);
    if ((pS0 <= 0.0f) && (pS4 <= 0.0f) && (pS8 <= 0.0f))
    {
        closestPoint = dv;
        return 0x2131u;
    }
    const float cen = gwGet(par, 1u, cp0) * gwGet(par, 2u, cp1) - gwGet(par, 1u, cp1) * gwGet(par, 2u, cp0);
    const float evn = gwGet(par, 1u, vNext);

    gwPut(par, 0u, L, cen * dv);                                 // dcen = dotW(cen, dV)
    gwPut(par, 1u, L, evn);
    gwPut(par, 2u, L, nV);
    gwSync(par);
    const bool uiS0 = gwDotAt(par, 0u, 0u) <= 0.0f;
    const bool uiS4 = gwDotAt(par, 0u, 4u) <= 0.0f;
    const bool uiS8 = gwDotAt(par, 0u, 8u) <= 0.0f;
    const float cevn = gwGet(par, 1u, cp0) * gwGet(par, 2u, cp1) - gwGet(par, 1u, cp1) * gwGet(par, 2u, cp0);

    gwPut(par, 0u, L, cevn * dv);                                // dcevn = dotW(cevn, dV)
    gwSync(par);
    const bool ujS0 = gwDotAt(par, 0u, 0u) > 0.0f;
    const bool ujS4 = gwDotAt(par, 0u, 4u) > 0.0f;
    const bool ujS8 = gwDotAt(par, 0u, 8u) > 0.0f;

    // vPrevI[0] = 8, vPrevI[4] = 0, vPrevI[8] = 4
    uint emask = 0u;
    if (uiS0 && ujS8 && (pS0 >= 0.0f)) emask |= 1u;
    if (uiS4 && ujS0 && (pS4 >= 0.0f)) emask |= 2u;
    if (uiS8 && ujS4 && (pS8 >= 0.0f)) emask |= 4u;
    uint eid = ffs(emask);
    if (eid != 0u)
    {
        eid = eid - 1u;
        const float spl = pSL28 / lD;                            // splatXW(pS / lD)
        const float tmp = dv - eV * spl;
        gwPut(par, 0u, L, tmp);
        gwSync(par);
        closestPoint = gwGet(par, 0u, ((eid << 2) | (L & 3u)) & 31u);
        return (eid << 8) | 0x32u;
    }

    uint fmask = 0u;
    if (!uiS0 && !ujS0 && (aS0 <= 0.0f)) fmask |= 1u;
    if (!uiS4 && !ujS4 && (aS4 <= 0.0f)) fmask |= 2u;
    if (!uiS8 && !ujS8 && (aS8 <= 0.0f)) fmask |= 4u;
    uint fid = ffs(fmask);
    if (fid != 0u)
    {
        fid = fid - 1u;
        gwPut(par, 0u, L, nV * nV);                              // dnn = dotW(nV, nV)
        gwSync(par);
        const float sad = aSL28 / gwDotAt(par, 0u, L & 28u);    // splatXW(aS / dnn)
        const float tmp = nV * sad;
        gwPut(par, 0u, L, tmp);
        gwPut(par, 1u, L, nV);
        gwSync(par);
        closestPoint = gwGet(par, 0u, ((fid << 2) | (L & 3u)) & 31u);
        const float tmpV = -gwGet(par, 1u, ((fid << 2) | (L & 3u)) & 31u);
        z = (L < 16u || L > 19u) ? z : tmpV;
        return (next3(fid) << 12) | (fid << 8) | 0x33u;
    }

    return GW_DEGENERATE;
}

// ---- gjk (pxgSlangGjk gjkRun, gjk.cuh:235-418), cold start ---------------------------
// `cachedLane` is this lane's word of the final simplex (GjkCachedData), EPA's input.
public uint gwGjkRun(inout uint par, uint lane, uint nbA, uint nbB, float3 initialUnitDir,
                     float minSep, float convergenceRatio, out GjkOut o, out float cachedLane)
{
    const uint L = lane;
    const uint c = L & 3u;
    cachedLane = 0.0f;

    float simplexV = 0.0f;
    float closestPointV = (c == 0u) ? initialUnitDir.x : (c == 1u) ? initialUnitDir.y
                        : (c == 2u) ? initialUnitDir.z : GW_MAX_F32;
    float upperBound = GW_MAX_F32, lowerBound = -GW_MAX_F32;
    gwPut(par, 0u, L, closestPointV * closestPointV);
    gwSync(par);
    float cmA = sqrt(gwDotAt(par, 0u, L & 28u));                // magW
    // the serial driver's deliberate `degenerate = false` hardening (pxgSlangGjk)
    bool degenerate = false;

    int size = 0;
    int backupSize = 0;
    float backupSimplexV = 0.0f;
    float backupClosestV = 0.0f;
    float3 directionOut = float3(0.0f, 0.0f, 0.0f);

    bool run = true;
    [loop] while (run)
    {
        const float dV = closestPointV / cmA;

        gwPut(par, 0u, L, dV);
        gwSync(par);
        const float3 dvVec = float3(gwGet(par, 0u, 0u), gwGet(par, 0u, 1u), gwGet(par, 0u, 2u));
        uint indexA, indexB;
        gwSupportPair(par, L, float3(-dvVec.x, -dvVec.y, -dvVec.z), nbA, dvVec, nbB, indexA, indexB);
        const float3 vA = gwCloudVert(0u, indexA);
        const float3 vB = gwCloudVert(1u, indexB);
        const float3 sv = vA - vB;
        const float supportVertexV = (c == 0u) ? sv.x : (c == 1u) ? sv.y : (c == 2u) ? sv.z : 0.0f;
        const float encoded = asfloat(encodeIndices(indexA, indexB));
        if ((L >> 2) == (uint)size) simplexV = (c < 3u) ? supportVertexV : encoded;

        // dotW(dV, supportVertexV)[0] = x[0] + x[1] + x[2], x = dV * supportVertexV
        const float lb = dvVec.x * sv.x + dvVec.y * sv.y + dvVec.z * sv.z;
        if (lb > lowerBound)
        {
            directionOut = float3(-dvVec.x, -dvVec.y, -dvVec.z);
            lowerBound = lb;
            if (lowerBound > minSep || lowerBound >= upperBound * convergenceRatio) { run = false; break; }
        }

        uint f = 0u;
        float pointV;
        if (size == 0)
        {
            f = 0x2101u;
            pointV = supportVertexV;
        }
        else if (size == 1)
        {
            gwPut(par, 0u, L, simplexV);
            gwSync(par);
            const float aV = gwGet(par, 0u, L & 3u);             // splatVW(0, simplexV)
            const float lV = supportVertexV - aV;
            gwPut(par, 0u, L, lV * lV);
            gwPut(par, 1u, L, aV * lV);
            gwSync(par);
            const float mm = gwDotAt(par, 0u, L & 28u);
            const float xx = -gwDotAt(par, 1u, L & 28u);
            const float mm0 = gwDotAt(par, 0u, 0u);
            const float xx0 = -gwDotAt(par, 1u, 0u);
            if (mm0 == 0.0f) { f = GW_DEGENERATE; pointV = 0.0f; }
            else if (xx0 < mm0) { f = 0x2102u; pointV = aV + lV * xx / mm; }
            else { f = 0x2111u; pointV = supportVertexV; }
        }
        else if (size == 2)
        {
            f = gwClosestPointTri(par, L, simplexV, supportVertexV, pointV);
        }
        else
        {
            f = gwClosestPointTet(par, L, simplexV, supportVertexV, pointV);
        }

        gwPut(par, 0u, L, pointV * pointV);
        gwSync(par);
        cmA = sqrt(gwDotAt(par, 0u, L & 28u));                  // magW(pointV)
        const float cm0 = sqrt(gwDotAt(par, 0u, 0u));
        const int newSize = (int)(f & 0xfu);
        degenerate = (f == GW_DEGENERATE) || (cm0 >= upperBound && newSize <= size);
        if (degenerate) { run = false; break; }   // cold start: warmStarted is always false

        backupSize = (cm0 >= upperBound) ? size : 0;
        if (backupSize != 0)
        {
            backupSimplexV = simplexV;
            backupClosestV = closestPointV;
        }
        else
        {
            upperBound = cm0;
        }

        closestPointV = pointV;
        size = newSize;

        // simplex permute: lanes < 12 take lane (sourceIndex << 2 | L & 3).  The shift is
        // only formed for those lanes; the serial loop's shift by 32 on lanes 28..31 is a
        // dead value there.
        gwPut(par, 0u, L, simplexV);
        gwSync(par);
        const uint sourceIndex = (L < 12u) ? ((f >> ((L & 28u) + 4u)) & 0xfu) : 0u;
        const float tmpP = gwGet(par, 0u, ((sourceIndex << 2) | (L & 3u)) & 31u);
        if (L < 12u) simplexV = tmpP;

        if (!(upperBound > 0.0f && lowerBound < upperBound * convergenceRatio)) run = false;
    }

    if (backupSize > 0)
    {
        simplexV = backupSimplexV;
        closestPointV = backupClosestV;
        size = backupSize;
    }

    // ---- output stage ----
    cachedLane = simplexV;
    o.pad = 0u;
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

    gwPut(par, 0u, L, closestPointV);
    gwPut(par, 1u, L, simplexV);
    gwSync(par);
    const float3 cp = float3(gwGet(par, 0u, 0u), gwGet(par, 0u, 1u), gwGet(par, 0u, 2u));
    const float3 dir = -getNormalized(cp);
    o.closestPointDir = float4(dir, 0.0f);

    const uint a0 = asuint(gwGet(par, 1u, 3u)) & 0xffu;
    const uint a1 = asuint(gwGet(par, 1u, 7u)) & 0xffu;
    const uint a2 = asuint(gwGet(par, 1u, 11u)) & 0xffu;

    float3 cpA;
    if (size == 3)
    {
        const float3 A = float3(gwGet(par, 1u, 0u), gwGet(par, 1u, 1u), gwGet(par, 1u, 2u));
        const float3 B = float3(gwGet(par, 1u, 4u), gwGet(par, 1u, 5u), gwGet(par, 1u, 6u));
        const float3 C = float3(gwGet(par, 1u, 8u), gwGet(par, 1u, 9u), gwGet(par, 1u, 10u));
        const float3 m = crossX(B - A, C - A);
        const float3 ma = absX(m);
        const float3 s1 = crossX(B - cp, C - cp);
        const float3 s2 = crossX(C - cp, A - cp);
        float d, u, v;
        if (ma.x >= ma.y && ma.x >= ma.z) { d = 1.0f / m.x; u = s1.x * d; v = s2.x * d; }
        else if (ma.y >= ma.z)            { d = 1.0f / m.y; u = s1.y * d; v = s2.y * d; }
        else                              { d = 1.0f / m.z; u = s1.z * d; v = s2.z * d; }
        const float w = 1.0f - u - v;
        cpA = u * gwCloudVert(0u, a0) + v * gwCloudVert(0u, a1) + w * gwCloudVert(0u, a2);
    }
    else if (size == 2)
    {
        const float3 V0 = float3(gwGet(par, 1u, 0u), gwGet(par, 1u, 1u), gwGet(par, 1u, 2u));
        const float3 V1 = float3(gwGet(par, 1u, 4u), gwGet(par, 1u, 5u), gwGet(par, 1u, 6u));
        const float3 dv = V1 - V0;
        const float q = -dot3(V0, dv) / dot3(dv, dv);
        cpA = (1.0f - q) * gwCloudVert(0u, a0) + q * gwCloudVert(0u, a1);
    }
    else
    {
        cpA = gwCloudVert(0u, a0);
    }

    o.closestPointA = float4(cpA, 0.0f);
    o.closestPointB = float4(cpA + upperBound * dir, 0.0f);
    o.ret = GJK_CLOSE;
    return o.ret;
}

// ---- epa (pxgSlangEpa epaRun, epa.cuh:388-526) ----------------------------------------
// Per-lane triangle state: normal, dist, vv (Vec3I8 vertices), aa (Vec3I8 adjacency).

// newPoint (epa.cuh): the support in +dir and -dir, keep the one farther from ref.
void gwNewPoint(inout uint par, uint lane, float3 dir, float3 ref, uint nbA, uint nbB,
                out float3 vertex, out uint indices)
{
    uint iAp, iBp, iAn, iBn;
    gwSupportPair(par, lane, dir, nbA, float3(-dir.x, -dir.y, -dir.z), nbB, iAp, iBp);
    gwSupportPair(par, lane, float3(-dir.x, -dir.y, -dir.z), nbA, dir, nbB, iAn, iBn);
    const float3 vp = gwCloudVert(0u, iAp) - gwCloudVert(1u, iBp);
    const float3 vn = gwCloudVert(0u, iAn) - gwCloudVert(1u, iBn);
    if (abs(dot3(vp - ref, dir)) > abs(dot3(vn - ref, dir))) { vertex = vp; indices = ((iBp << 8) | (iAp & 0xffu)) & 0xffffu; }
    else { vertex = vn; indices = ((iBn << 8) | (iAn & 0xffu)) & 0xffffu; }
}

void gwEpaStart4(inout uint par, uint lane, float input, inout float3 normal, inout float dist,
                 inout uint vv, inout uint aa)
{
    const uint L = lane;
    // S.verts[L >> 2] component (L & 3) for (L & 3) < 3, S.idx[L >> 2] for (L & 3) == 3
    if (L < 16u)
    {
        const uint k = L >> 2;
        const uint cc = L & 3u;
        if (cc == 0u) sGwEpaVx[k] = input;
        else if (cc == 1u) sGwEpaVy[k] = input;
        else if (cc == 2u) sGwEpaVz[k] = input;
        else sGwEpaIdx[k] = asuint(input) & 0xffffu;
    }
    gwPut(par, 0u, L, input * input);
    gwPut(par, 1u, L, input);
    gwSync(par);                                                 // also publishes the seed polytope
    if (L < 4u)
    {
        const float m = 1.0f / sqrt(gwDotAt(par, 0u, L * 4u + 16u));
        normal = float3(gwGet(par, 1u, L * 4u + 16u) * m, gwGet(par, 1u, L * 4u + 17u) * m,
                        gwGet(par, 1u, L * 4u + 18u) * m);
        dist = gwGet(par, 1u, L * 4u + 19u) * m;
    }
    if (L == 0u) { aa = SA4_0; vv = SV4_0; }
    else if (L == 1u) { aa = SA4_1; vv = SV4_1; }
    else if (L == 2u) { aa = SA4_2; vv = SV4_2; }
    else if (L == 3u) { aa = SA4_3; vv = SV4_3; }
}

void gwEpaStart123(inout uint par, uint lane, float input, uint size, uint nbA, uint nbB,
                   inout float3 normalOut, inout float distOut, inout uint vv, inout uint aa)
{
    const uint L = lane;
    gwPut(par, 0u, L, input);
    gwSync(par);
    // verts[tI - tI / 4] = input[tI] for (tI & 3) < 3; indices[tI >> 2] otherwise
    float3 vertex0 = float3(gwGet(par, 0u, 0u), gwGet(par, 0u, 1u), gwGet(par, 0u, 2u));
    float3 vertex1 = float3(gwGet(par, 0u, 4u), gwGet(par, 0u, 5u), gwGet(par, 0u, 6u));
    float3 vertex2 = float3(gwGet(par, 0u, 8u), gwGet(par, 0u, 9u), gwGet(par, 0u, 10u));
    const uint index0 = asuint(gwGet(par, 0u, 3u)) & 0xffffu;
    uint index1 = asuint(gwGet(par, 0u, 7u)) & 0xffffu;
    uint index2 = asuint(gwGet(par, 0u, 11u)) & 0xffffu;
    if (size < 2u)
    {
        float3 nv; uint ni;
        gwNewPoint(par, L, float3(1.0f, 0.0f, 0.0f), vertex0, nbA, nbB, nv, ni);
        vertex1 = nv; index1 = ni;
    }
    if (size < 3u)
    {
        const float3 dir = vertex1 - vertex0;
        const float3 axis = (abs(dir.x) < abs(dir.y)) ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
        float3 nv; uint ni;
        gwNewPoint(par, L, cross3(axis, dir), vertex0, nbA, nbB, nv, ni);
        vertex2 = nv; index2 = ni;
    }
    if (L == 0u)
    {
        sGwEpaVx[0] = vertex0.x; sGwEpaVy[0] = vertex0.y; sGwEpaVz[0] = vertex0.z;
        sGwEpaVx[1] = vertex1.x; sGwEpaVy[1] = vertex1.y; sGwEpaVz[1] = vertex1.z;
        sGwEpaVx[2] = vertex2.x; sGwEpaVy[2] = vertex2.y; sGwEpaVz[2] = vertex2.z;
        sGwEpaIdx[0] = index0; sGwEpaIdx[1] = index1; sGwEpaIdx[2] = index2;
    }
    const float3 e01 = vertex1 - vertex0, e12 = vertex2 - vertex1, e02 = vertex2 - vertex0;
    float3 normal;
    if (mag3(e01) <= mag3(e02)) normal = getNormalized(cross3(e01, e12));
    else                        normal = getNormalized(cross3(e02, e12));
    const float distance = dot3(normal, vertex0);
    if (L == 0u) { normalOut = normal; distOut = distance; vv = 0x020100u; aa = 0x010101u; }
    if (L == 1u) { normalOut = -normal; distOut = -distance; vv = 0x020001u; aa = 0x000000u; }
    GroupMemoryBarrierWithGroupSync();                           // the seed polytope is visible
}

// replaceTriangle (pxgSlangEpa, epa.cuh:64-251).  The apex is S.verts[nbVertices - 1],
// written and published by the caller.
bool gwReplaceTriangle(inout uint par, uint lane, int nbVertices, int triIndex, float lowerBound,
                       inout uint liveB, float releps, inout float3 normal, inout float dist,
                       inout uint vv, inout uint aa)
{
    const uint L = lane;
    const uint laneB = 1u << L;
    const float3 apex = gwEpaVert((uint)(nbVertices - 1));
    const bool threatened = ((liveB & laneB) != 0u) && (dot3(normal, apex) - dist >= releps);
    gwPutU(par, 0u, L, threatened ? 1u : 0u);
    gwSync(par);
    uint threatenedB = 0u;
    for (uint l = 0u; l < 32u; ++l) if (gwGetU(par, 0u, l) != 0u) threatenedB |= (1u << l);
    if (threatenedB == 0u || threatenedB == liveB || ((threatenedB & (1u << (uint)triIndex)) == 0u)) return false;

    const uint b0B = 1u << g3(aa, 0u), b1B = 1u << g3(aa, 1u), b2B = 1u << g3(aa, 2u);
    const uint nbr = (b0B | b1B | b2B) & (threatened ? threatenedB : 0u);
    uint killB = 1u << (uint)triIndex;
    [loop] for (uint pB = 0u; pB != killB;)
    {
        pB = killB;
        gwPutU(par, 0u, L, ((nbr & killB) != 0u) ? 1u : 0u);
        gwSync(par);
        uint bal = 0u;
        for (uint l = 0u; l < 32u; ++l) if (gwGetU(par, 0u, l) != 0u) bal |= (1u << l);
        killB |= bal;
    }
    const uint deadB = (~liveB) | killB;

    const bool alive = (laneB & ~deadB) != 0u;
    const bool d0 = alive && ((b0B & deadB) != 0u);
    const bool d1 = alive && ((b1B & deadB) != 0u);
    const bool d2 = alive && ((b2B & deadB) != 0u);

    // The silhouette edges: (flags, vertices) per lane, applied below in lane order.
    gwPutU(par, 0u, L, (d0 ? 1u : 0u) | (d1 ? 2u : 0u) | (d2 ? 4u : 0u));
    gwPutU(par, 1u, L, vv);
    gwSync(par);
    uint d0d1d2B = 0u;
    uint nbEdges = 0u;
    for (uint l = 0u; l < 32u; ++l)
    {
        const uint fl = gwGetU(par, 0u, l);
        if (fl != 0u) d0d1d2B |= (1u << l);
        nbEdges += (fl & 1u) + ((fl >> 1) & 1u) + ((fl >> 2) & 1u);
    }
    if (d0d1d2B == 0u) return false;

    // z of the lowest silhouette lane (the serial `zz`: d2 over d1 over d0)
    const uint first = firstbitlow(d0d1d2B);
    const uint fFl = gwGetU(par, 0u, first);
    const uint fVv = gwGetU(par, 1u, first);
    uint loopStart = 0u;
    if ((fFl & 1u) != 0u) loopStart = g3(fVv, 2u);
    if ((fFl & 2u) != 0u) loopStart = g3(fVv, 0u);
    if ((fFl & 4u) != 0u) loopStart = g3(fVv, 1u);

    // Thread 0 writes the linked list in the serial order (lane 0..31, d0 then d1 then
    // d2), so a vertex two silhouette edges start from keeps the serial last writer;
    // then it flattens the loop as the CUDA's lane 0 does.  The clamps only keep a
    // corrupt index inside the arrays (the serial arrays have no bound check either).
    if (L == 0u)
    {
        for (uint t = 0u; t < 32u; ++t)
        {
            const uint fl = gwGetU(par, 0u, t);
            const uint tv = gwGetU(par, 1u, t);
            const uint v0a = g3(tv, 0u), v1a = g3(tv, 1u), v2a = g3(tv, 2u);
            if ((fl & 1u) != 0u) { sGwEpaNxt[min(v2a, GW_EPA_MAX_VERTS - 1u)] = v1a; sGwEpaEdg[min(v2a, GW_EPA_MAX_VERTS - 1u)] = t; }
            if ((fl & 2u) != 0u) { sGwEpaNxt[min(v0a, GW_EPA_MAX_VERTS - 1u)] = v2a; sGwEpaEdg[min(v0a, GW_EPA_MAX_VERTS - 1u)] = t; }
            if ((fl & 4u) != 0u) { sGwEpaNxt[min(v1a, GW_EPA_MAX_VERTS - 1u)] = v0a; sGwEpaEdg[min(v1a, GW_EPA_MAX_VERTS - 1u)] = t; }
        }
        int nbLoop = 0;
        sGwEpaLp[nbLoop++] = loopStart;
        uint vertex = sGwEpaNxt[min(loopStart, GW_EPA_MAX_VERTS - 1u)];
        [loop] while (vertex != loopStart)
        {
            if (vertex >= GW_EPA_MAX_VERTS) { nbLoop = -1; break; }
            if (nbLoop >= (int)(GW_EPA_MAX_VERTS - 1u)) { nbLoop = -1; break; }
            sGwEpaLp[nbLoop++] = vertex;
            vertex = sGwEpaNxt[vertex];
        }
        sGwEpaNbLoop = nbLoop;
    }
    GroupMemoryBarrierWithGroupSync();                           // the loop and its length
    const int nbLoop = sGwEpaNbLoop;
    if (nbLoop != (int)nbEdges) return false;

    // the cone: the nbLoop lowest dead lanes, in lane order
    uint coneB = 0u;
    for (uint t = 0u; t < 32u; ++t)
    {
        const uint tB = 1u << t;
        if (((deadB & tB) != 0u) && ((int)countbits(deadB & (tB - 1u)) < nbLoop)) coneB |= tB;
    }
    const uint coneIndex = countbits(deadB & (laneB - 1u));
    bool fail = false;
    if ((coneB & laneB) != 0u)
    {
        // The loop vertices are distinct (the walk reached loopStart again), so every
        // cone lane reads and then writes its OWN edg entry: no lane sees another's.
        const uint v0c = (uint)(nbVertices - 1);
        const uint v1c = sGwEpaLp[min(coneIndex, GW_EPA_MAX_VERTS - 1u)];
        const uint v2c = sGwEpaLp[min((coneIndex + 1u == (uint)nbLoop) ? 0u : coneIndex + 1u, GW_EPA_MAX_VERTS - 1u)];
        vv = mk3(v0c, v1c, v2c);
        aa = s3(aa, 0u, sGwEpaEdg[min(v1c, GW_EPA_MAX_VERTS - 1u)]);
        const uint mask1 = (coneIndex + 1u == (uint)nbLoop) ? 0u : (2u * laneB - 1u);
        aa = s3(aa, 1u, firstbitlow(coneB & ~mask1));
        const uint mask2 = ((coneIndex == 0u) ? 0u : laneB) - 1u;
        aa = s3(aa, 2u, firstbithigh(coneB & mask2));
        sGwEpaEdg[min(v1c, GW_EPA_MAX_VERTS - 1u)] = L;
        const float3 p0 = gwEpaVert(v0c), p1 = gwEpaVert(v1c), p2 = gwEpaVert(v2c);
        const float3 e01 = p1 - p0, e02 = p2 - p0;
        float3 n;
        if (magSq3(e01) < magSq3(e02)) n = cross3(e01, p2 - p1);
        else                           n = cross3(e02, p2 - p1);
        const float nm = mag3(n);
        const float inv = 1.0f / nm;
        normal = float3(n.x * inv, n.y * inv, n.z * inv);
        dist = dot3(normal, p0);
        fail = (nm == 0.0f) || ((lowerBound > 0.0f) && ((dist - lowerBound) < -1e-4f));
    }
    GroupMemoryBarrierWithGroupSync();                           // the cone's edg write-backs

    if (d0) aa = s3(aa, 0u, sGwEpaEdg[min(g3(vv, 2u), GW_EPA_MAX_VERTS - 1u)]);
    if (d1) aa = s3(aa, 1u, sGwEpaEdg[min(g3(vv, 0u), GW_EPA_MAX_VERTS - 1u)]);
    if (d2) aa = s3(aa, 2u, sGwEpaEdg[min(g3(vv, 1u), GW_EPA_MAX_VERTS - 1u)]);
    liveB = (liveB & ~deadB) | coneB;

    gwPutU(par, 0u, L, fail ? 1u : 0u);
    gwSync(par);
    uint failB = 0u;
    for (uint l = 0u; l < 32u; ++l) if (gwGetU(par, 0u, l) != 0u) failB |= (1u << l);
    return failB == 0u;
}

// `input` is this lane's word of the GjkCachedData simplex GJK left behind.
public uint gwEpaRun(inout uint par, uint lane, float input, uint size, float releps,
                     uint nbA, uint nbB, out EpaOut o)
{
    const uint L = lane;
    if (L < GW_EPA_MAX_VERTS)
    {
        sGwEpaVx[L] = 0.0f; sGwEpaVy[L] = 0.0f; sGwEpaVz[L] = 0.0f;
        sGwEpaIdx[L] = 0u; sGwEpaNxt[L] = 0u; sGwEpaEdg[L] = 0u; sGwEpaLp[L] = 0u;
    }
    float3 normal = float3(0.0f, 0.0f, 0.0f);
    float dist = 0.0f;
    uint vv = 0u, aa = 0u;
    GroupMemoryBarrierWithGroupSync();                           // zeroed before the seed writes

    float lowerBound = -GW_MAX_F32, upperBound = GW_MAX_F32;
    uint bestVerts = 0u;
    float3 bestDir = float3(0.0f, 0.0f, 0.0f);
    int nbVertices;
    uint liveTriangles;

    if (size == 4u) { gwEpaStart4(par, L, input, normal, dist, vv, aa); nbVertices = 4; liveTriangles = 0xfu; }
    else { gwEpaStart123(par, L, input, size, nbA, nbB, normal, dist, vv, aa); nbVertices = 3; liveTriangles = 0x3u; }

    bool nondegenerate = true;
    [loop] while (nondegenerate && (uint)nbVertices < GW_EPA_MAX_VERTS)
    {
        // the shortest-distance live triangle, by the serial select chain (P9.86)
        const float tvLane = ((liveTriangles & (1u << L)) != 0u) ? dist : GW_MAX_F32;
        gwPut(par, 0u, L, tvLane);
        gwPut(par, 1u, L, normal.x);
        gwPut(par, 2u, L, normal.y);
        gwPut(par, 3u, L, normal.z);
        gwPutU(par, 0u, L, vv);
        gwSync(par);
        float e = GW_MAX_F32;
        for (uint l = 0u; l < 32u; ++l)
        {
            const float tv = gwGet(par, 0u, l);
            e = ((tv < e) || ((tv == 0.0f) && (e == 0.0f) && (asuint(tv) != 0u))) ? tv : e;
        }
        uint f = 0u;
        for (uint l = 0u; l < 32u; ++l) if (e == gwGet(par, 0u, l)) f |= (1u << l);
        const uint triIndex = firstbitlow(f);
        const float3 d = float3(gwGet(par, 1u, triIndex), gwGet(par, 2u, triIndex), gwGet(par, 3u, triIndex));
        const uint triVerts = gwGetU(par, 0u, triIndex);

        uint indexA, indexB;
        gwSupportPair(par, L, d, nbA, float3(-d.x, -d.y, -d.z), nbB, indexA, indexB);
        const float3 supportVertex = gwCloudVert(0u, indexA) - gwCloudVert(1u, indexB);
        const float thisFrameUb = dot3(supportVertex, d);

        if (abs(thisFrameUb - e) <= releps)
        {
            upperBound = thisFrameUb; lowerBound = e;
            bestVerts = triVerts;
            bestDir = d;
            break;
        }
        if (e > upperBound) break;

        const float ub = min(upperBound, thisFrameUb);
        if (ub < upperBound)
        {
            upperBound = ub; lowerBound = e;
            bestVerts = triVerts;
            bestDir = d;
        }

        if (L == 0u)
        {
            sGwEpaVx[nbVertices] = supportVertex.x;
            sGwEpaVy[nbVertices] = supportVertex.y;
            sGwEpaVz[nbVertices] = supportVertex.z;
            sGwEpaIdx[nbVertices] = ((indexB << 8) | (indexA & 0xffu)) & 0xffffu;
        }
        nbVertices = nbVertices + 1;
        GroupMemoryBarrierWithGroupSync();                       // the apex is visible
        nondegenerate = gwReplaceTriangle(par, L, nbVertices, (int)triIndex, e, liveTriangles, releps,
                                          normal, dist, vv, aa);
    }

    o.result = 0u; o.p0 = 0u; o.p1 = 0u; o.q0 = 0.0f; o.q1 = 0.0f;
    o.degenerate = nondegenerate ? 0u : 1u;
    o.lowerBound = lowerBound; o.upperBound = upperBound;
    const float3 closestPoint = float3(bestDir.x * lowerBound, bestDir.y * lowerBound, bestDir.z * lowerBound);
    o.direction = float4(bestDir, 0.0f);

    const uint v0 = g3(bestVerts, 0u), v1 = g3(bestVerts, 1u), v2 = g3(bestVerts, 2u);
    float u, vv2, ww;
    barycentrics(gwEpaVert(v0), gwEpaVert(v1), gwEpaVert(v2), closestPoint, u, vv2, ww);
    const float3 cpA = gwCloudVert(0u, sGwEpaIdx[min(v0, GW_EPA_MAX_VERTS - 1u)] & 0xffu) * u
                     + gwCloudVert(0u, sGwEpaIdx[min(v1, GW_EPA_MAX_VERTS - 1u)] & 0xffu) * vv2
                     + gwCloudVert(0u, sGwEpaIdx[min(v2, GW_EPA_MAX_VERTS - 1u)] & 0xffu) * ww;
    o.closestPointA = float4(cpA, 0.0f);
    o.closestPointB = float4(cpA - closestPoint, 0.0f);
    o.result = (lowerBound < 0.0f) ? GJK_CLOSE : GJK_OVERLAP;
    return o.result;
}

// ---- tetPrimitivesCollide2 (pxgSlangTetCollide, convexNpCommon.h:212) ----------------
public static const uint GW_PATH_EPA = 1u;       // EPA ran (OVERLAP, or the anomaly gate)
public static const uint GW_PATH_ANOMALY = 2u;   // ... because of the anomaly gate
public static const uint GW_PATH_RETRY = 4u;     // EPA was degenerate: GJK + EPA again from +y
public static const uint GW_PATH_START4 = 8u;    // the first EPA started from a tetrahedron

void gwFoldEpaIntoGjk(EpaOut e, inout GjkOut o)
{
    o.degenerate = e.degenerate;
    o.lowerBound = e.lowerBound;
    o.upperBound = e.upperBound;
    o.direction = e.direction;
    o.closestPointA = e.closestPointA;
    o.closestPointB = e.closestPointB;
}

// The clouds are sGwCloud[0] (nbA vertices) and sGwCloud[1] (nbB), written with
// gwCloudPut behind a barrier; the first barrier here publishes them.  `path` says
// which branches ran (GW_PATH_*), for the fixture's coverage count.
public uint gwTetPrimitivesCollide2(uint lane, uint nbA, uint nbB, float3 searchDir, float contactDistance,
                                    float inSphereRadius0, float inSphereRadius1, out GjkOut o, out uint path)
{
    uint par = 0u;
    path = 0u;
    GroupMemoryBarrierWithGroupSync();                           // the caller's cloud writes

    float3 initialDir = float3(1.0f, 0.0f, 0.0f);
    if (dot(searchDir, searchDir) > 0.0f)
        initialDir = searchDir;

    const float relEpsBase = 0.5f * (inSphereRadius0 + inSphereRadius1);
    const float releps = relEpsBase * (1.0f - TETC_CONVERGENCE_RATIO);

    float cached;
    const uint gjkResult = gwGjkRun(par, lane, nbA, nbB, initialDir, contactDistance,
                                    TETC_CONVERGENCE_RATIO, o, cached);
    if (gjkResult == GJK_DISTANT)
        return GJK_DISTANT;

    const bool separated = (gjkResult == GJK_CLOSE);
    // PxVec3::dot, (x + y) + z from the NoContraction stamp -- the serial driver's P9.89 note
    const bool anomaly = separated && dot(o.closestPointDir.xyz, o.direction.xyz) < 0.999f;

    uint result = GJK_CLOSE;
    if (!separated || anomaly)
    {
        path |= GW_PATH_EPA | (anomaly ? GW_PATH_ANOMALY : 0u) | ((o.size == 4u) ? GW_PATH_START4 : 0u);
        EpaOut e;
        result = gwEpaRun(par, lane, cached, o.size, releps, nbA, nbB, e);
        gwFoldEpaIntoGjk(e, o);

        if (o.degenerate != 0u)
        {
            path |= GW_PATH_RETRY;
            float cached2;
            gwGjkRun(par, lane, nbA, nbB, float3(0.0f, 1.0f, 0.0f), contactDistance,
                     TETC_CONVERGENCE_RATIO, o, cached2);
            EpaOut e2;
            result = gwEpaRun(par, lane, cached2, o.size, releps, nbA, nbB, e2);
            gwFoldEpaIntoGjk(e2, o);
        }
    }
    return result;
}
