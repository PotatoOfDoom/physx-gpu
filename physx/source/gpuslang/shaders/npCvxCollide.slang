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

// npCvxCollide.slang — the assembled collide() contact generator
// (cudaGJKEPA.cu L1275-1465) for the convex-convex stage2 narrowphase, in the
// single-thread warp-SIMULATION form of the verified standalone tranches:
//
//   prep (npCvxCollidePrep):        prepareVertices + warm-start seed +
//                                   runGJK re-pack tail + initialDir
//   gjkWarm (npGjkWarm, T133):      full warm-start-capable gjk() driver
//   epaCore (npEpa, T132):          full epa() driver (Scratch/Lanes renamed
//                                   EpaScratch/EpaLanes)
//   decide (npCvxCollideDecide):    post-GJK decision + sphere witness store
//                                   (here GATED on type0 == eSPHERE, unlike
//                                   the standalone which runs it always)
//   generatedFaceContacts (T136):   capsule witness-ray face contacts
//   generatedContactsEEContacts     capsule edge-edge fallback (T143), blob
//     (npCvxEEDriver):              readers switched from gBlob to Ptr<> over
//                                   convexPtr1
//
// Only the SPHERE and CAPSULE type0 dispatch arms are assembled; the
// convex-hull arm (generateConvexContacts, cu L1458-1462) is DEFERRED — the
// stage2 caller gates the collide() call on type0 so convex-type0 pairs keep
// the elision contract.
//
// Contact points are written INSIDE collide() straight through Ptr<float4>
// views of the manifold (localNormalPen @ manifoldBase+0, localContactA @
// +64, localContactB @ +128), mirroring the .cu ss_scratch pointers.  The
// warm-start re-pack words 13-15 that runGJK produces (cu L813-817) are
// computed and DISCARDED: the .cu stage2 epilogue stores only moreData words
// 9..12 (constantDataSize = 36 B, cu L2635-2640), so nothing here may write
// manifold words 13-15.
//
// No push constants, no [shader] entries, no global buffers — pure module,
// imported by npCvxStage2Core (same split rationale as npCvxStage2Setup /
// npCvxStage2SetupDbg: one push-constant block per compiled entry).

module npCvxCollide;

import pxgSlangNpAbi;

// ---------------------------------------------------------------------------
// Internal constants (non-public: npCvxStage2Core exports public EGT_* and
// imports this module, so these must not be exported).
// ---------------------------------------------------------------------------
static const uint DEGENERATE = 0xffffu;
static const float PXG_MAX_F32 = 3.4028234663852886e38f;
static const uint GJK_DISTANT = 0u; // GjkResult::eDISTANT
static const uint GJK_CLOSE = 1u;   // GjkResult::eCLOSE
static const uint GJK_OVERLAP = 2u; // GjkResult::eOVERLAP
static const uint WARP_SIZE = 32u;
static const uint SZ_HEADER = 32u;  // cooked-hull header uint4 + extents float4
static const float PCM_LOWER_EPS = 1e-2f;
static const float PCM_UPPER_EPS = 5e-2f;
static const uint EPA_MAX_VERTS = 18u;
static const uint EGT_SPHERE = 0u;
static const uint EGT_CAPSULE = 2u;

// ---------------------------------------------------------------------------
// Public scratch structs (cu CollideScratch L701-755; GjkOutput/GjkCachedData
// are the schlock structs; the cache is the exact 32-float image the GJK warp
// registers hold — verts + packed idx at slots 3/7/11/15, tetra planes 16-31).
// ---------------------------------------------------------------------------
public struct GjkOutput
{
    public float3 direction;
    public float3 closestPointA;
    public float3 closestPointB;
    public float3 closestPointDir;
    public float lowerBound;
    public float upperBound;
    public bool degenerate;
}

public struct GjkCachedData
{
    public float v[32];
    public uint size;
}

public struct CollideScratch
{
    public float4 vA[64];                     // prepared cloud A (aToB-transformed)
    public float4 vB[64];                     // prepared cloud B (shape space)
    public GjkCachedData cachedData;
    public GjkOutput gjkOutput;
    public float3 scale0; public float4 rot0;
    public float3 scale1; public float4 rot1;
    public float4 aToBq; public float3 aToBp;
    public uint64_t manifoldBase;             // localNormalPen=+0, A=+64, B=+128
    public uint64_t convexPtr0, convexPtr1;   // cooked-hull blob bases (hullPtr+16)
    public uint nbVertices0, nbPolygons0, nbEdges0;
    public uint nbVertices1, nbPolygons1, nbEdges1;
    public float contactDistance, toleranceLength, replaceBreakingThreshold;
    public float inSphereRadius0, inSphereRadius1;
    public uint initialNbContacts;
    public uint type0, type1;
}

// ---------------------------------------------------------------------------
// Internal math helpers (single copies of the standalone files' duplicates;
// quat rotate/rotateInv come from pxgSlangNpAbi).
// ---------------------------------------------------------------------------
float3 vertex2Shape(float3 v, float3 scale, float4 rot)
{
    return npAbiQuatRotateInv(rot, npAbiQuatRotate(rot, v) * scale);
}
float3 shape2Vertex(float3 v, float3 scale, float4 rot)
{
    return npAbiQuatRotateInv(rot, npAbiQuatRotate(rot, v) / scale);
}
// PxTransform::transformInv(PxVec3) = q.rotateInv(input - p)
float3 transformInvPoint(float4 q, float3 p, float3 v)
{
    return npAbiQuatRotateInv(q, v - p);
}

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 crossX(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
float3 absX(float3 a) { return float3(abs(a.x), abs(a.y), abs(a.z)); }
float magSq3(float3 v) { return dot3(v, v); }
float mag3(float3 v) { return sqrt(dot3(v, v)); }
float3 getNormalized(float3 v) { const float m = dot3(v, v); if (m > 0.0f) { const float r = 1.0f / sqrt(m); return float3(v.x * r, v.y * r, v.z * r); } return float3(0.0f, 0.0f, 0.0f); }

// warp-simulation primitives (npGjkWarm.slang L68-87)
uint next3(uint x) { return x == 2u ? 0u : x + 1u; }
uint prev3(uint x) { return x == 0u ? 2u : x - 1u; }
uint ffs(uint m) { return m == 0u ? 0u : firstbitlow(m) + 1u; }

void dotW(float a[32], float b[32], out float o[32])
{
    float x[32];
    for (uint L = 0u; L < 32u; ++L) x[L] = a[L] * b[L];
    for (uint L = 0u; L < 32u; ++L) { const float s1 = (L + 1u < 32u) ? x[L + 1u] : x[L]; const float s2 = (L + 2u < 32u) ? x[L + 2u] : x[L]; o[L] = x[L] + s1 + s2; }
}
void crossW(float a[32], float b[32], uint cp0[32], uint cp1[32], out float o[32]) { for (uint L = 0u; L < 32u; ++L) o[L] = a[cp0[L]] * b[cp1[L]] - a[cp1[L]] * b[cp0[L]]; }
void splatXW(float a[32], out float o[32]) { for (uint L = 0u; L < 32u; ++L) o[L] = a[L & 28u]; }
void splatVW(uint index, float a[32], out float o[32]) { for (uint L = 0u; L < 32u; ++L) o[L] = a[((index << 2) | (L & 3u)) & 31u]; }
void magW(float v[32], out float cm[32]) { float dv[32]; dotW(v, v, dv); for (uint L = 0u; L < 32u; ++L) cm[L] = sqrt(dv[L & 28u]); }
uint encodeIndices(uint a, uint b) { return 0x3f800000u | ((b & 0xffu) << 8) | (a & 0xffu); }

// Vec3I8 (npEpa.slang L100-102)
uint mk3(uint a, uint b, uint c) { return (a & 0xffu) | ((b & 0xffu) << 8) | ((c & 0xffu) << 16); }
uint g3(uint p, uint i) { return (p >> (i * 8u)) & 0xffu; }
uint s3(uint p, uint i, uint val) { return (p & ~(0xffu << (i * 8u))) | ((val & 0xffu) << (i * 8u)); }

// T141 accessors / blob byte offsets (npCvxEEDriver.slang L51-58)
uint u32HighW(uint v) { return (v >> 16) & 0xffffu; }
uint u32LowW(uint v)  { return v & 0xffffu; }
uint u16HighW(uint v) { return (v >> 8) & 0xffu; }
uint getVRef8W(uint val)   { return u32HighW(val); }
uint getNbVertsW(uint val) { return u16HighW(u32LowW(val)); }
uint planesByteOff(uint nbV) { return SZ_HEADER + 16u * nbV; }
uint polyDescByteOff(uint nbV, uint nbP) { return SZ_HEADER + 16u * nbV + 16u * nbP; }
uint vertexDataByteOff(uint nbV, uint nbP, uint nbE) { return SZ_HEADER + 16u * nbV + 20u * nbP + 6u * nbE + 3u * nbV; }

// blob readers (npCvxEEDriver.slang L127-133, gBlob -> Ptr<uint> over the
// cooked-hull base; the base is 16-byte aligned so word indexing is exact)
float4 blobLoadFloat4(uint64_t base, uint byteOff)
{
    Ptr<uint> w = Ptr<uint>(base);
    const uint i = byteOff >> 2;
    return float4(asfloat(w[i]), asfloat(w[i + 1]), asfloat(w[i + 2]), asfloat(w[i + 3]));
}
uint blobLoadU32(uint64_t base, uint byteOff) { return (Ptr<uint>(base))[byteOff >> 2]; }
uint blobLoadU8(uint64_t base, uint byteOff) { const uint word = (Ptr<uint>(base))[byteOff >> 2]; return (word >> (8u * (byteOff & 3u))) & 0xffu; }

// ---------------------------------------------------------------------------
// support argmax over the prepared clouds (npGjkWarm.slang L153-164 /
// npEpa.slang L105-128 — identical bodies; single copies over s.vA/s.vB)
// ---------------------------------------------------------------------------
uint supportIdxA(float3 dir, uint nb, inout CollideScratch s)
{
    uint bestN = 0xFFFFFFFFu; float bestDist = -PXG_MAX_F32;
    for (uint i = 0u; i < nb; i += 32u) { float d[32]; for (uint l = 0u; l < 32u; ++l) { const uint n = i + l; d[l] = (n < nb) ? dot3(dir, s.vA[n].xyz) : -PXG_MAX_F32; } float cmax = -PXG_MAX_F32; for (uint l = 0u; l < 32u; ++l) cmax = max(cmax, d[l]); if (cmax > bestDist) { uint ll = 0u; for (uint l = 0u; l < 32u; ++l) if (d[l] == cmax) { ll = l; break; } bestDist = cmax; bestN = i + ll; } }
    return bestN;
}
uint supportIdxB(float3 dir, uint nb, inout CollideScratch s)
{
    uint bestN = 0xFFFFFFFFu; float bestDist = -PXG_MAX_F32;
    for (uint i = 0u; i < nb; i += 32u) { float d[32]; for (uint l = 0u; l < 32u; ++l) { const uint n = i + l; d[l] = (n < nb) ? dot3(dir, s.vB[n].xyz) : -PXG_MAX_F32; } float cmax = -PXG_MAX_F32; for (uint l = 0u; l < 32u; ++l) cmax = max(cmax, d[l]); if (cmax > bestDist) { uint ll = 0u; for (uint l = 0u; l < 32u; ++l) if (d[l] == cmax) { ll = l; break; } bestDist = cmax; bestN = i + ll; } }
    return bestN;
}

// ---------------------------------------------------------------------------
// prep: prepareVertices (npCvxCollidePrep.slang L120-125; cudaNpCommon.h
// L315-343).  A gets the aToB transform applied, B is shape-space only; all
// 64 slots zero-initialized first.  verts0Ptr == 0 selects the synthetic
// sphere/capsule vertices (cu L2465-2476), else the blob cloud at
// convexPtr+32 (cu L2486/L2497).
// ---------------------------------------------------------------------------
void cvxPrepareVertices(inout CollideScratch s, float4 verts0Synth[2], uint64_t verts0Ptr, uint64_t verts1Ptr)
{
    for (uint i = 0u; i < 64u; ++i) { s.vA[i] = float4(0, 0, 0, 0); s.vB[i] = float4(0, 0, 0, 0); }
    for (uint i = 0u; i < s.nbVertices0; ++i)
    {
        const float3 raw = (verts0Ptr != 0) ? (Ptr<float4>(verts0Ptr))[i].xyz : verts0Synth[i].xyz;
        s.vA[i] = float4(npAbiQuatRotate(s.aToBq, vertex2Shape(raw, s.scale0, s.rot0)) + s.aToBp, 0.0f);
    }
    for (uint i = 0u; i < s.nbVertices1; ++i)
        s.vB[i] = float4(vertex2Shape((Ptr<float4>(verts1Ptr))[i].xyz, s.scale1, s.rot1), 0.0f);
}

// ---------------------------------------------------------------------------
// gjkWarm — the tranche-133 warm-start-capable gjk() driver (npGjkWarm.slang
// ctGjkWarm body L170-320), with the cache image round-tripping through
// s.cachedData (the standalone never stores back to gCache; the .cu
// storeSimplex at gjk.cuh L370-380 happens after the backup restore, before
// the early returns — mirrored here so EPA consumes the exact image).
// Fills s.gjkOutput; returns the GjkResult code.
// ---------------------------------------------------------------------------

uint closestPointTriW(inout float zV[32], float cV[32], out float closestPoint[32])
{
    uint cp0I[32], cp1I[32];
    for (uint L = 0u; L < 32u; ++L) { cp0I[L] = ((L & 28u) | next3(L & 3u)) & 31u; cp1I[L] = ((L & 28u) | prev3(L & 3u)) & 31u; }
    for (uint L = 0u; L < 32u; ++L) closestPoint[L] = 0.0f;
    float eV[32]; for (uint L = 0u; L < 32u; ++L) eV[L] = cV[L] - zV[L];
    float dee[32]; dotW(eV, eV, dee); float lD[32]; splatXW(dee, lD);
    if ((lD[0] == 0.0f) || (lD[4] == 0.0f)) return DEGENERATE;
    float shortEdgeV[32]; for (uint L = 0u; L < 32u; ++L) shortEdgeV[L] = (lD[L ^ 4u] < lD[L]) ? eV[L ^ 4u] : eV[L];
    float d1[32]; for (uint L = 0u; L < 32u; ++L) d1[L] = zV[L ^ 4u] - zV[L];
    float mV[32]; crossW(d1, shortEdgeV, cp0I, cp1I, mV);
    float pS[32]; dotW(eV, cV, pS);
    float cme[32]; crossW(mV, eV, cp0I, cp1I, cme); float dcme[32]; dotW(cme, cV, dcme);
    bool uS[32]; for (uint L = 0u; L < 32u; ++L) uS[L] = dcme[L] <= 0.0f;
    if (!(uS[0] || uS[4]))
    {
        float dmc[32]; dotW(mV, cV, dmc); const float projD = dmc[0];
        float dmm[32]; dotW(mV, mV, dmm); float sxdmm[32]; splatXW(dmm, sxdmm);
        float tmp[32]; for (uint L = 0u; L < 32u; ++L) tmp[L] = mV[L] * projD / sxdmm[L]; splatVW(0u, tmp, closestPoint);
        const bool flip = projD < 0.0f; for (uint L = 0u; L < 32u; ++L) if ((L & 3u) == 3u) mV[L] = projD;
        const uint index = flip ? 1u : 0u;
        for (uint L = 0u; L < 32u; ++L) { const uint src = ((index << 2) | (L & 3u)) & 31u; zV[L] = (L < 16u || L > 19u) ? zV[L] : mV[src]; }
        return flip ? 0x0123u : 0x1023u;
    }
    uint emask = 0u; if (uS[0] && (pS[0] > 0.0f)) emask |= 1u; if (uS[4] && (pS[4] > 0.0f)) emask |= 2u;
    uint eid = ffs(emask);
    if (eid != 0u) { eid = eid - 1u; float sxpS[32]; splatXW(pS, sxpS); float tmp[32]; for (uint L = 0u; L < 32u; ++L) tmp[L] = cV[L] - eV[L] * sxpS[L] / lD[L]; splatVW(eid, tmp, closestPoint); return (eid << 8) | 0x22u; }
    for (uint L = 0u; L < 32u; ++L) closestPoint[L] = cV[L]; return 0x0021u;
}
uint closestPointTetW(inout float zV[32], float dV[32], out float closestPoint[32])
{
    for (uint L = 0u; L < 32u; ++L) closestPoint[L] = 0.0f;
    float n0V[32]; splatVW(4u, zV, n0V); float dn0d[32]; dotW(n0V, dV, dn0d); float dn0z[32]; dotW(n0V, zV, dn0z);
    if (dn0d[0] > dn0z[0]) return DEGENERATE;
    uint cp0I[32], cp1I[32], vNextI[32], vPrevI[32];
    for (uint L = 0u; L < 32u; ++L) { cp0I[L] = ((L & 28u) | next3(L & 3u)) & 31u; cp1I[L] = ((L & 28u) | prev3(L & 3u)) & 31u; vNextI[L] = (L > 7u) ? (L - 8u) : (L + 4u); vPrevI[L] = (L < 4u) ? (L + 8u) : (L - 4u); }
    float eV[32]; for (uint L = 0u; L < 32u; ++L) eV[L] = dV[L] - zV[L];
    float dee[32]; dotW(eV, eV, dee); float lD[32]; splatXW(dee, lD);
    if ((lD[0] == 0.0f) || (lD[4] == 0.0f) || (lD[8] == 0.0f)) return DEGENERATE;
    float shortEdgeV[32]; for (uint L = 0u; L < 32u; ++L) shortEdgeV[L] = (lD[vNextI[L]] < lD[L]) ? eV[vNextI[L]] : eV[L];
    float d2[32]; for (uint L = 0u; L < 32u; ++L) d2[L] = zV[vNextI[L]] - zV[L];
    float nV[32]; crossW(shortEdgeV, d2, cp0I, cp1I, nV);
    float aS[32]; dotW(nV, dV, aS); float zvp[32]; for (uint L = 0u; L < 32u; ++L) zvp[L] = zV[vPrevI[L]]; float gS[32]; dotW(zvp, nV, gS);
    if ((aS[0] <= gS[0]) || (aS[4] <= gS[4]) || (aS[8] <= gS[8])) return DEGENERATE;
    for (uint L = 0u; L < 32u; ++L) if ((L & 3u) == 3u) nV[L] = aS[L - 3u];
    if ((aS[0] > 0.0f) && (aS[4] > 0.0f) && (aS[8] > 0.0f)) { for (uint L = 0u; L < 32u; ++L) if (L >= 20u) zV[L] = nV[L - 20u]; return 0x2104u; }
    float pS[32]; dotW(eV, dV, pS);
    if ((pS[0] <= 0.0f) && (pS[4] <= 0.0f) && (pS[8] <= 0.0f)) { for (uint L = 0u; L < 32u; ++L) closestPoint[L] = dV[L]; return 0x2131u; }
    float cen[32]; crossW(eV, nV, cp0I, cp1I, cen); float dcen[32]; dotW(cen, dV, dcen); bool uiS[32]; for (uint L = 0u; L < 32u; ++L) uiS[L] = dcen[L] <= 0.0f;
    float evn[32]; for (uint L = 0u; L < 32u; ++L) evn[L] = eV[vNextI[L]]; float cevn[32]; crossW(evn, nV, cp0I, cp1I, cevn); float dcevn[32]; dotW(cevn, dV, dcevn); bool ujS[32]; for (uint L = 0u; L < 32u; ++L) ujS[L] = dcevn[L] > 0.0f;
    uint emask = 0u;
    if (uiS[0] && ujS[vPrevI[0]] && (pS[0] >= 0.0f)) emask |= 1u;
    if (uiS[4] && ujS[vPrevI[4]] && (pS[4] >= 0.0f)) emask |= 2u;
    if (uiS[8] && ujS[vPrevI[8]] && (pS[8] >= 0.0f)) emask |= 4u;
    uint eid = ffs(emask);
    if (eid != 0u) { eid = eid - 1u; float pl[32]; for (uint L = 0u; L < 32u; ++L) pl[L] = pS[L] / lD[L]; float spl[32]; splatXW(pl, spl); float tmp[32]; for (uint L = 0u; L < 32u; ++L) tmp[L] = dV[L] - eV[L] * spl[L]; splatVW(eid, tmp, closestPoint); return (eid << 8) | 0x32u; }
    uint fmask = 0u;
    if (!uiS[0] && !ujS[0] && (aS[0] <= 0.0f)) fmask |= 1u;
    if (!uiS[4] && !ujS[4] && (aS[4] <= 0.0f)) fmask |= 2u;
    if (!uiS[8] && !ujS[8] && (aS[8] <= 0.0f)) fmask |= 4u;
    uint fid = ffs(fmask);
    if (fid != 0u) { fid = fid - 1u; float dnn[32]; dotW(nV, nV, dnn); float ad[32]; for (uint L = 0u; L < 32u; ++L) ad[L] = aS[L] / dnn[L]; float sad[32]; splatXW(ad, sad); float tmp[32]; for (uint L = 0u; L < 32u; ++L) tmp[L] = nV[L] * sad[L]; splatVW(fid, tmp, closestPoint); for (uint L = 0u; L < 32u; ++L) { const uint src = ((fid << 2) | (L & 3u)) & 31u; zV[L] = (L < 16u || L > 19u) ? zV[L] : -nV[src]; } return (next3(fid) << 12) | (fid << 8) | 0x33u; }
    return DEGENERATE;
}

uint gjkWarm(inout CollideScratch s, float3 initialUnitDir, float minSep, float convergenceRatio)
{
    const uint nbA = s.nbVertices0;
    const uint nbB = s.nbVertices1;
    const uint cacheSize = s.cachedData.size;

    uint cacheMask = (1u << cacheSize) - 1u;
    float simplexV[32]; for (uint L = 0u; L < 32u; ++L) simplexV[L] = s.cachedData.v[L]; // loadSimplex

    // state that must survive to the output stage (reset per restart)
    float closestPointV[32]; float cmA[32]; float upperBound = PXG_MAX_F32, lowerBound = -PXG_MAX_F32;
    int size = 0; int backupSize = 0; bool degenerate = false;
    float backupSimplexV[32]; float backupClosestV[32];
    float3 directionOut = float3(0.0f, 0.0f, 0.0f);

    bool restart = true;
    [loop] while (restart)
    {
        restart = false;
        bool warmStarted = false;
        size = 0; backupSize = 0;
        for (uint L = 0u; L < 32u; ++L) { backupSimplexV[L] = 0.0f; backupClosestV[L] = 0.0f; }
        if (cacheMask != 0u) splatVW(0u, simplexV, closestPointV);
        else for (uint L = 0u; L < 32u; ++L) { const uint c = L & 3u; closestPointV[L] = (c == 0u) ? initialUnitDir.x : (c == 1u) ? initialUnitDir.y : (c == 2u) ? initialUnitDir.z : PXG_MAX_F32; }
        upperBound = PXG_MAX_F32; lowerBound = -PXG_MAX_F32;
        magW(closestPointV, cmA);

        bool run = true;
        [loop] while (run)
        {
            float dV[32]; for (uint L = 0u; L < 32u; ++L) dV[L] = closestPointV[L] / cmA[L];

            uint candidates = 0u;
            if (cacheMask != 0u)
            {
                float dd[32]; dotW(dV, simplexV, dd);
                for (uint L = 0u; L < 32u; ++L) if (dd[(L << 2) & 31u] < upperBound) candidates |= (1u << L);
                candidates &= cacheMask;
            }

            float supportVertexV[32];
            if (candidates != 0u)
            {
                warmStarted = true;
                const uint lowBit = candidates & (0u - candidates);
                cacheMask &= ~(lowBit * 2u - 1u);
                const uint lowIdx = firstbitlow(candidates);
                splatVW(lowIdx, simplexV, supportVertexV);
                for (uint L = 0u; L < 32u; ++L) if ((L >> 2) == (uint)size) simplexV[L] = supportVertexV[L];
            }
            else
            {
                cacheMask = 0u;
                const float3 dvVec = float3(dV[0], dV[1], dV[2]);
                const uint indexA = supportIdxA(float3(-dvVec.x, -dvVec.y, -dvVec.z), nbA, s);
                const uint indexB = supportIdxB(dvVec, nbB, s);
                const float3 sv = s.vA[indexA].xyz - s.vB[indexB].xyz;
                for (uint L = 0u; L < 32u; ++L) { const uint c = L & 3u; supportVertexV[L] = (c == 0u) ? sv.x : (c == 1u) ? sv.y : (c == 2u) ? sv.z : 0.0f; }
                const float encoded = asfloat(encodeIndices(indexA, indexB));
                for (uint L = 0u; L < 32u; ++L) if ((L >> 2) == (uint)size) simplexV[L] = ((L & 3u) < 3u) ? supportVertexV[L] : encoded;
                float dds[32]; dotW(dV, supportVertexV, dds); const float lb = dds[0];
                if (lb > lowerBound)
                {
                    directionOut = float3(-dV[0], -dV[1], -dV[2]);
                    lowerBound = lb;
                    if (lowerBound > minSep || lowerBound >= upperBound * convergenceRatio) { run = false; break; }
                }
            }

            uint f = 0u; float pointV[32];
            if (size == 0) { f = 0x2101u; for (uint L = 0u; L < 32u; ++L) pointV[L] = supportVertexV[L]; }
            else if (size == 1)
            {
                float aV[32]; splatVW(0u, simplexV, aV);
                float lV[32]; for (uint L = 0u; L < 32u; ++L) lV[L] = supportVertexV[L] - aV[L];
                float dll[32]; dotW(lV, lV, dll); float mm[32]; splatXW(dll, mm);
                float dal[32]; dotW(aV, lV, dal); float xx[32]; splatXW(dal, xx); for (uint L = 0u; L < 32u; ++L) xx[L] = -xx[L];
                if (mm[0] == 0.0f) { f = DEGENERATE; for (uint L = 0u; L < 32u; ++L) pointV[L] = 0.0f; }
                else if (xx[0] < mm[0]) { f = 0x2102u; for (uint L = 0u; L < 32u; ++L) pointV[L] = aV[L] + lV[L] * xx[L] / mm[L]; }
                else { f = 0x2111u; for (uint L = 0u; L < 32u; ++L) pointV[L] = supportVertexV[L]; }
            }
            else if (size == 2) { f = closestPointTriW(simplexV, supportVertexV, pointV); }
            else { f = closestPointTetW(simplexV, supportVertexV, pointV); }

            magW(pointV, cmA);
            const int newSize = (int)(f & 0xfu);
            degenerate = (f == DEGENERATE) || (cmA[0] >= upperBound && newSize <= size);
            if (degenerate)
            {
                if (!warmStarted) { run = false; break; }
                else { cacheMask = 0u; restart = true; run = false; break; }
            }

            backupSize = (cmA[0] >= upperBound) ? size : 0;
            if (backupSize != 0) { for (uint L = 0u; L < 32u; ++L) { backupSimplexV[L] = simplexV[L]; backupClosestV[L] = closestPointV[L]; } }
            else upperBound = cmA[0];

            for (uint L = 0u; L < 32u; ++L) closestPointV[L] = pointV[L];
            size = newSize;

            float tmpP[32];
            for (uint L = 0u; L < 32u; ++L) { const uint si = (f >> ((L & 28u) + 4u)) & 0xfu; tmpP[L] = simplexV[((si << 2) | (L & 3u)) & 31u]; }
            for (uint L = 0u; L < 32u; ++L) if (L < 12u) simplexV[L] = tmpP[L];

            if (!(upperBound > 0.0f && lowerBound < upperBound * convergenceRatio)) run = false;
        }
    }

    if (backupSize > 0) { for (uint L = 0u; L < 32u; ++L) { simplexV[L] = backupSimplexV[L]; closestPointV[L] = backupClosestV[L]; } size = backupSize; }

    // storeSimplex + size (gjk.cuh L370-380): EPA and the re-pack tail consume
    // this exact image.
    for (uint L = 0u; L < 32u; ++L) s.cachedData.v[L] = simplexV[L];
    s.cachedData.size = (uint)size;

    s.gjkOutput.degenerate = degenerate;
    s.gjkOutput.lowerBound = lowerBound;
    s.gjkOutput.upperBound = upperBound;
    s.gjkOutput.direction = directionOut;
    s.gjkOutput.closestPointA = float3(0.0f, 0.0f, 0.0f);
    s.gjkOutput.closestPointB = float3(0.0f, 0.0f, 0.0f);
    s.gjkOutput.closestPointDir = float3(0.0f, 0.0f, 0.0f);

    if (upperBound == 0.0f) return GJK_OVERLAP;
    if (lowerBound > minSep) return GJK_DISTANT;

    const float3 cp = float3(closestPointV[0], closestPointV[1], closestPointV[2]);
    const float3 dir = -getNormalized(cp);
    s.gjkOutput.closestPointDir = dir;
    const uint a0 = asuint(simplexV[3]) & 0xffu, a1 = asuint(simplexV[7]) & 0xffu, a2 = asuint(simplexV[11]) & 0xffu;
    float3 cpA;
    if (size == 3)
    {
        const float3 A = float3(simplexV[0], simplexV[1], simplexV[2]), B = float3(simplexV[4], simplexV[5], simplexV[6]), C = float3(simplexV[8], simplexV[9], simplexV[10]);
        const float3 m = crossX(B - A, C - A); const float3 ma = absX(m);
        const float3 s1 = crossX(B - cp, C - cp); const float3 s2 = crossX(C - cp, A - cp);
        float d, u, v;
        if (ma.x >= ma.y && ma.x >= ma.z) { d = 1.0f / m.x; u = s1.x * d; v = s2.x * d; }
        else if (ma.y >= ma.z)            { d = 1.0f / m.y; u = s1.y * d; v = s2.y * d; }
        else                              { d = 1.0f / m.z; u = s1.z * d; v = s2.z * d; }
        const float w = 1.0f - u - v;
        cpA = u * s.vA[a0].xyz + v * s.vA[a1].xyz + w * s.vA[a2].xyz;
    }
    else if (size == 2)
    {
        const float3 V0 = float3(simplexV[0], simplexV[1], simplexV[2]), V1 = float3(simplexV[4], simplexV[5], simplexV[6]);
        const float3 dv = V1 - V0; const float q = -dot3(V0, dv) / dot3(dv, dv);
        cpA = (1.0f - q) * s.vA[a0].xyz + q * s.vA[a1].xyz;
    }
    else cpA = s.vA[a0].xyz;

    s.gjkOutput.closestPointA = cpA;
    s.gjkOutput.closestPointB = cpA + upperBound * dir;
    return GJK_CLOSE;
}

// ---------------------------------------------------------------------------
// runGJK (cu L781-820): gjkWarm + the warm-start re-pack tail.  The re-pack
// words are returned to the caller which DISCARDS them — stage2 never stores
// manifold words 13-15 (see the header note).
// ---------------------------------------------------------------------------
uint runGJK(float3 initialDir, inout CollideScratch s, float convergenceRatio,
            out uint rePack13, out uint rePack14, out uint rePack15)
{
    const uint gjkResult = gjkWarm(s, initialDir, s.contactDistance, convergenceRatio);

    uint warmStartAIndices = 0u;
    uint warmStartBIndices = 0u;
    for (uint tI = 0u; tI < s.cachedData.size; ++tI)
    {
        const uint idx = asuint(s.cachedData.v[(tI << 2) + 3u]);
        const uint a = idx & 0xFFu;         // CachedVertex::a()
        const uint b = (idx >> 8u) & 0xFFu; // CachedVertex::b()
        warmStartAIndices |= (a << (tI << 3u));
        warmStartBIndices |= (b << (tI << 3u));
    }
    rePack13 = s.cachedData.size;
    rePack14 = warmStartAIndices;
    rePack15 = warmStartBIndices;
    return gjkResult;
}

// ---------------------------------------------------------------------------
// epaCore — the tranche-132 epa() driver (npEpa.slang ctEpa body L316-393),
// Scratch/Lanes renamed EpaScratch/EpaLanes, vertex source = s.vA/s.vB,
// cache image = s.cachedData (read-only, like the .cu const GjkCachedData*).
// Overwrites s.gjkOutput.{degenerate,lowerBound,upperBound,direction,
// closestPointA,closestPointB} — NOT closestPointDir (epa.cuh never writes
// it).  Returns the GjkResult code.
// ---------------------------------------------------------------------------
struct EpaScratch
{
    float3 verts[18];
    uint idx[18];   // PxU16 index words
    uint nxt[18];   // linked-list next
    uint edg[18];   // silhouette-edge -> cone-tri fixup
    uint lp[18];    // flattened silhouette loop
    float3 bestDir;
}
struct EpaLanes
{
    float3 normal[32];
    float dist[32];
    uint vv[32]; // Vec3I8 vertex indices
    uint aa[32]; // Vec3I8 adjacencies
}

void newPoint(float3 dir, float3 ref, uint nbA, uint nbB, inout CollideScratch s, out float3 vertex, out uint indices)
{
    const uint iAp = supportIdxA(dir, nbA, s), iBp = supportIdxB(float3(-dir.x, -dir.y, -dir.z), nbB, s);
    const uint iAn = supportIdxA(float3(-dir.x, -dir.y, -dir.z), nbA, s), iBn = supportIdxB(dir, nbB, s);
    const float3 vp = s.vA[iAp].xyz - s.vB[iBp].xyz;
    const float3 vn = s.vA[iAn].xyz - s.vB[iBn].xyz;
    if (abs(dot3(vp - ref, dir)) > abs(dot3(vn - ref, dir))) { vertex = vp; indices = ((iBp << 8) | (iAp & 0xffu)) & 0xffffu; }
    else { vertex = vn; indices = ((iBn << 8) | (iAn & 0xffu)) & 0xffffu; }
}

// schlock::barycentrics
void barycentrics(float3 a, float3 b, float3 c, float3 p, out float u, out float v, out float w)
{
    const float3 m = crossX(b - a, c - a);
    const float3 ma = absX(m);
    const float3 a1 = crossX(b - p, c - p);
    const float3 a2 = crossX(c - p, a - p);
    float d;
    if (ma.x >= ma.y && ma.x >= ma.z) { d = 1.0f / m.x; u = a1.x * d; v = a2.x * d; }
    else if (ma.y >= ma.z)            { d = 1.0f / m.y; u = a1.y * d; v = a2.y * d; }
    else                              { d = 1.0f / m.z; u = a1.z * d; v = a2.z * d; }
    w = 1.0f - u - v;
}

// seed constants: epaStart4 (tranche 130)
static const uint SV4_0 = 0x020100u, SV4_1 = 0x000103u, SV4_2 = 0x010203u, SV4_3 = 0x020003u;
static const uint SA4_0 = 0x010302u, SA4_1 = 0x020300u, SA4_2 = 0x030100u, SA4_3 = 0x010200u;
void epaStart4(float input[32], inout EpaScratch S, inout EpaLanes L)
{
    for (uint tI = 0u; tI < 16u; ++tI)
    {
        if ((tI & 3u) < 3u) { const uint fi = tI - tI / 4u; const uint vk = fi / 3u; const uint cc = fi % 3u; if (cc == 0u) S.verts[vk].x = input[tI]; else if (cc == 1u) S.verts[vk].y = input[tI]; else S.verts[vk].z = input[tI]; }
        else S.idx[tI >> 2] = asuint(input[tI]) & 0xffffu;
    }
    float d[32];
    { float x[32]; for (uint l = 0u; l < 32u; ++l) x[l] = input[l] * input[l];
      for (uint l = 0u; l < 32u; ++l) { const float s1 = (l + 1u < 32u) ? x[l + 1u] : x[l]; const float s2 = (l + 2u < 32u) ? x[l + 2u] : x[l]; d[l] = x[l] + s1 + s2; } }
    for (uint tI = 0u; tI < 4u; ++tI)
    {
        const float m = 1.0f / sqrt(d[tI * 4u + 16u]);
        L.normal[tI] = float3(input[tI * 4u + 16u] * m, input[tI * 4u + 17u] * m, input[tI * 4u + 18u] * m);
        L.dist[tI] = input[tI * 4u + 19u] * m;
    }
    L.aa[0] = SA4_0; L.aa[1] = SA4_1; L.aa[2] = SA4_2; L.aa[3] = SA4_3;
    L.vv[0] = SV4_0; L.vv[1] = SV4_1; L.vv[2] = SV4_2; L.vv[3] = SV4_3;
}

// epaStart123 (tranche 131)
void epaStart123(float input[32], uint size, uint nbA, uint nbB, inout CollideScratch cs, inout EpaScratch S, inout EpaLanes L)
{
    float verts[12]; uint indices[4];
    for (uint tI = 0u; tI < 16u; ++tI) { if ((tI & 3u) < 3u) verts[tI - tI / 4u] = input[tI]; else indices[tI >> 2] = asuint(input[tI]) & 0xffffu; }
    float3 vertex0 = float3(verts[0], verts[1], verts[2]);
    float3 vertex1 = float3(verts[3], verts[4], verts[5]);
    if (size < 2u) { float3 nv; uint ni; newPoint(float3(1.0f, 0.0f, 0.0f), vertex0, nbA, nbB, cs, nv, ni); vertex1 = nv; verts[3] = nv.x; verts[4] = nv.y; verts[5] = nv.z; indices[1] = ni; }
    float3 vertex2 = float3(verts[6], verts[7], verts[8]);
    if (size < 3u)
    {
        const float3 dir = vertex1 - vertex0;
        const float3 axis = (abs(dir.x) < abs(dir.y)) ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
        float3 nv; uint ni; newPoint(crossX(axis, dir), vertex0, nbA, nbB, cs, nv, ni); vertex2 = nv; verts[6] = nv.x; verts[7] = nv.y; verts[8] = nv.z; indices[2] = ni;
    }
    S.verts[0] = float3(verts[0], verts[1], verts[2]); S.verts[1] = float3(verts[3], verts[4], verts[5]); S.verts[2] = float3(verts[6], verts[7], verts[8]);
    S.idx[0] = indices[0]; S.idx[1] = indices[1]; S.idx[2] = indices[2];
    const float3 e01 = vertex1 - vertex0, e12 = vertex2 - vertex1, e02 = vertex2 - vertex0;
    float3 normal;
    if (mag3(e01) <= mag3(e02)) normal = getNormalized(crossX(e01, e12));
    else                        normal = getNormalized(crossX(e02, e12));
    const float distance = dot3(normal, vertex0);
    L.normal[0] = normal; L.dist[0] = distance; L.vv[0] = 0x020100u; L.aa[0] = 0x010101u;
    L.normal[1] = -normal; L.dist[1] = -distance; L.vv[1] = 0x020001u; L.aa[1] = 0x000000u;
}

// replaceTriangle (epa.cuh :64-251)
bool replaceTriangle(inout EpaScratch S, inout EpaLanes L, int nbVertices, int triIndex, float lowerBound, inout uint liveB, float releps)
{
    const float3 apex = S.verts[nbVertices - 1];
    bool threatened[32]; uint threatenedB = 0u;
    for (uint tI = 0u; tI < 32u; ++tI)
    {
        const bool th = ((liveB & (1u << tI)) != 0u) && (dot3(L.normal[tI], apex) - L.dist[tI] >= releps);
        threatened[tI] = th; if (th) threatenedB |= (1u << tI);
    }
    if (threatenedB == 0u || threatenedB == liveB || ((threatenedB & (1u << (uint)triIndex)) == 0u)) return false;

    uint b0B[32], b1B[32], b2B[32], nbr[32];
    for (uint tI = 0u; tI < 32u; ++tI)
    {
        b0B[tI] = 1u << g3(L.aa[tI], 0u); b1B[tI] = 1u << g3(L.aa[tI], 1u); b2B[tI] = 1u << g3(L.aa[tI], 2u);
        nbr[tI] = (b0B[tI] | b1B[tI] | b2B[tI]) & (threatened[tI] ? threatenedB : 0u);
    }
    uint killB = 1u << (uint)triIndex;
    [loop] for (uint pB = 0u; pB != killB;)
    {
        pB = killB;
        uint bal = 0u;
        for (uint tI = 0u; tI < 32u; ++tI) if ((nbr[tI] & killB) != 0u) bal |= (1u << tI);
        killB |= bal;
    }
    const uint deadB = (~liveB) | killB;

    bool d0[32], d1[32], d2[32]; uint v0a[32], v1a[32], v2a[32], zz[32];
    for (uint tI = 0u; tI < 32u; ++tI)
    {
        const uint laneB = 1u << tI;
        const bool alive = (laneB & ~deadB) != 0u;
        d0[tI] = alive && ((b0B[tI] & deadB) != 0u);
        d1[tI] = alive && ((b1B[tI] & deadB) != 0u);
        d2[tI] = alive && ((b2B[tI] & deadB) != 0u);
        v0a[tI] = g3(L.vv[tI], 0u); v1a[tI] = g3(L.vv[tI], 1u); v2a[tI] = g3(L.vv[tI], 2u); zz[tI] = 0u;
    }
    for (uint tI = 0u; tI < 32u; ++tI)
    {
        if (d0[tI]) { zz[tI] = v2a[tI]; S.nxt[v2a[tI]] = v1a[tI]; S.edg[v2a[tI]] = tI; }
        if (d1[tI]) { zz[tI] = v0a[tI]; S.nxt[v0a[tI]] = v2a[tI]; S.edg[v0a[tI]] = tI; }
        if (d2[tI]) { zz[tI] = v1a[tI]; S.nxt[v1a[tI]] = v0a[tI]; S.edg[v1a[tI]] = tI; }
    }
    uint d0d1d2B = 0u;
    for (uint tI = 0u; tI < 32u; ++tI) if (d0[tI] || d1[tI] || d2[tI]) d0d1d2B |= (1u << tI);
    if (d0d1d2B == 0u) return false;

    const uint loopStart = zz[firstbitlow(d0d1d2B)];
    int nbLoop = 0;
    S.lp[nbLoop++] = loopStart;
    { uint vertex = S.nxt[loopStart];
      [loop] while (vertex != loopStart)
      {
          if (vertex >= EPA_MAX_VERTS) { nbLoop = -1; break; }
          if (nbLoop >= (int)(EPA_MAX_VERTS - 1u)) { nbLoop = -1; break; }
          S.lp[nbLoop++] = vertex;
          vertex = S.nxt[vertex];
      } }

    uint bd0 = 0u, bd1 = 0u, bd2 = 0u;
    for (uint tI = 0u; tI < 32u; ++tI) { if (d0[tI]) bd0 |= (1u << tI); if (d1[tI]) bd1 |= (1u << tI); if (d2[tI]) bd2 |= (1u << tI); }
    if (nbLoop != (int)(countbits(bd0) + countbits(bd1) + countbits(bd2))) return false;

    uint coneIndex[32]; uint coneB = 0u;
    for (uint tI = 0u; tI < 32u; ++tI)
    {
        const uint laneB = 1u << tI;
        coneIndex[tI] = countbits(deadB & (laneB - 1u));
        if (((deadB & laneB) != 0u) && ((int)coneIndex[tI] < nbLoop)) coneB |= (1u << tI);
    }
    bool fail[32]; for (uint tI = 0u; tI < 32u; ++tI) fail[tI] = false;
    for (uint tI = 0u; tI < 32u; ++tI)
    {
        const uint laneB = 1u << tI;
        if ((coneB & laneB) != 0u)
        {
            const uint v0c = (uint)(nbVertices - 1);
            const uint v1c = S.lp[coneIndex[tI]];
            const uint v2c = S.lp[(coneIndex[tI] + 1u == (uint)nbLoop) ? 0u : coneIndex[tI] + 1u];
            L.vv[tI] = mk3(v0c, v1c, v2c);
            L.aa[tI] = s3(L.aa[tI], 0u, S.edg[v1c]);
            const uint mask1 = (coneIndex[tI] + 1u == (uint)nbLoop) ? 0u : (2u * laneB - 1u);
            L.aa[tI] = s3(L.aa[tI], 1u, firstbitlow(coneB & ~mask1));
            const uint mask2 = ((coneIndex[tI] == 0u) ? 0u : laneB) - 1u;
            L.aa[tI] = s3(L.aa[tI], 2u, firstbithigh(coneB & mask2));
            S.edg[v1c] = tI;
            const float3 p0 = S.verts[v0c], p1 = S.verts[v1c], p2 = S.verts[v2c];
            const float3 e01 = p1 - p0, e02 = p2 - p0;
            float3 n;
            if (magSq3(e01) < magSq3(e02)) n = crossX(e01, p2 - p1);
            else                           n = crossX(e02, p2 - p1);
            const float nm = mag3(n);
            const float inv = 1.0f / nm;
            L.normal[tI] = float3(n.x * inv, n.y * inv, n.z * inv);
            L.dist[tI] = dot3(L.normal[tI], p0);
            fail[tI] = (nm == 0.0f) || ((lowerBound > 0.0f) && ((L.dist[tI] - lowerBound) < -1e-4f));
        }
    }
    for (uint tI = 0u; tI < 32u; ++tI)
    {
        if (d0[tI]) L.aa[tI] = s3(L.aa[tI], 0u, S.edg[g3(L.vv[tI], 2u)]);
        if (d1[tI]) L.aa[tI] = s3(L.aa[tI], 1u, S.edg[g3(L.vv[tI], 0u)]);
        if (d2[tI]) L.aa[tI] = s3(L.aa[tI], 2u, S.edg[g3(L.vv[tI], 1u)]);
    }
    liveB = (liveB & ~deadB) | coneB;
    uint failB = 0u; for (uint tI = 0u; tI < 32u; ++tI) if (fail[tI]) failB |= (1u << tI);
    return failB == 0u;
}

uint epaCore(inout CollideScratch cs, float releps)
{
    const uint nbA = cs.nbVertices0;
    const uint nbB = cs.nbVertices1;
    const uint cacheSize = cs.cachedData.size;

    float input[32];
    for (uint i = 0u; i < 32u; ++i) input[i] = cs.cachedData.v[i];

    EpaScratch S;
    for (uint i = 0u; i < 18u; ++i) { S.verts[i] = float3(0.0f, 0.0f, 0.0f); S.idx[i] = 0u; S.nxt[i] = 0u; S.edg[i] = 0u; S.lp[i] = 0u; }
    S.bestDir = float3(0.0f, 0.0f, 0.0f);
    EpaLanes L;
    for (uint i = 0u; i < 32u; ++i) { L.normal[i] = float3(0.0f, 0.0f, 0.0f); L.dist[i] = 0.0f; L.vv[i] = 0u; L.aa[i] = 0u; }

    float lowerBound = -PXG_MAX_F32, upperBound = PXG_MAX_F32;
    uint bestVerts = 0u;
    int nbVertices; uint liveTriangles;

    if (cacheSize == 4u) { epaStart4(input, S, L); nbVertices = 4; liveTriangles = 0xfu; }
    else { epaStart123(input, cacheSize, nbA, nbB, cs, S, L); nbVertices = 3; liveTriangles = 0x3u; }

    bool nondegenerate = true;
    [loop] while (nondegenerate && (uint)nbVertices < EPA_MAX_VERTS)
    {
        // shortest-distance live triangle
        float e = PXG_MAX_F32;
        for (uint tI = 0u; tI < 32u; ++tI) { const float tv = ((liveTriangles & (1u << tI)) != 0u) ? L.dist[tI] : PXG_MAX_F32; e = min(e, tv); }
        uint f = 0u;
        for (uint tI = 0u; tI < 32u; ++tI) { const float tv = ((liveTriangles & (1u << tI)) != 0u) ? L.dist[tI] : PXG_MAX_F32; if (e == tv) f |= (1u << tI); }
        const uint triIndex = firstbitlow(f);
        const float3 d = L.normal[triIndex];

        const uint indexA = supportIdxA(d, nbA, cs), indexB = supportIdxB(float3(-d.x, -d.y, -d.z), nbB, cs);
        const float3 supportVertex = cs.vA[indexA].xyz - cs.vB[indexB].xyz;
        const float thisFrameUb = dot3(supportVertex, d);

        if (abs(thisFrameUb - e) <= releps)
        {
            upperBound = thisFrameUb; lowerBound = e;
            bestVerts = L.vv[triIndex];
            S.bestDir = d;
            break;
        }
        if (e > upperBound) break;

        const float ub = min(upperBound, thisFrameUb);
        if (ub < upperBound)
        {
            upperBound = ub; lowerBound = e;
            bestVerts = L.vv[triIndex];
            S.bestDir = d;
        }

        S.verts[nbVertices] = supportVertex;
        S.idx[nbVertices] = ((indexB << 8) | (indexA & 0xffu)) & 0xffffu;
        nbVertices = nbVertices + 1;
        nondegenerate = replaceTriangle(S, L, nbVertices, (int)triIndex, e, liveTriangles, releps);
    }

    cs.gjkOutput.degenerate = !nondegenerate;
    cs.gjkOutput.lowerBound = lowerBound;
    cs.gjkOutput.upperBound = upperBound;
    const float3 bestDir = S.bestDir;
    const float3 closestPoint = float3(bestDir.x * lowerBound, bestDir.y * lowerBound, bestDir.z * lowerBound);
    cs.gjkOutput.direction = bestDir;

    const uint v0 = g3(bestVerts, 0u), v1 = g3(bestVerts, 1u), v2 = g3(bestVerts, 2u);
    float u, vv, ww;
    barycentrics(S.verts[v0], S.verts[v1], S.verts[v2], closestPoint, u, vv, ww);
    const float3 cpA = cs.vA[S.idx[v0] & 0xffu].xyz * u + cs.vA[S.idx[v1] & 0xffu].xyz * vv + cs.vA[S.idx[v2] & 0xffu].xyz * ww;
    cs.gjkOutput.closestPointA = cpA;
    cs.gjkOutput.closestPointB = cpA - closestPoint;
    return (lowerBound < 0.0f) ? GJK_CLOSE : GJK_OVERLAP;
}

// ---------------------------------------------------------------------------
// intersectRayPolyhedron (npCvxFaceContacts.slang L115-176 / cu L1025-1097),
// planes read from the cooked-hull blob via Ptr<float4>.  The 32-lane block
// structure is kept verbatim: per block, tTemp compares against the PRE-block
// tFirst/tLast; the max/min commit happens after the block — the exact CUDA
// warp-reduction ordering.
// ---------------------------------------------------------------------------
bool intersectRayPolyhedron(float3 a, float3 dir, uint64_t planesBase, uint nbPolygons,
                            out float tEnter, out float tExit)
{
    const float eps = 1e-7f;
    float tFirst = 0.0f;
    float tLast = PXG_MAX_F32;
    tEnter = 0.0f;
    tExit = 0.0f;

    for (uint i = 0u; i < nbPolygons; i += WARP_SIZE)
    {
        bool blockSep = false;
        float newFirst = tFirst;
        float newLast = tLast;

        for (uint lane = 0u; lane < WARP_SIZE; ++lane)
        {
            const uint k = i + lane;
            if (k >= nbPolygons)
                continue;

            const float4 plane = (Ptr<float4>(planesBase))[k];
            const float3 n = float3(plane.x, plane.y, plane.z);
            const float d = plane.w;

            const float denominator = dot(n, dir);
            const float distToPlane = dot(n, a) + d;

            if (abs(denominator) < eps)
            {
                if (distToPlane > 0.0f)
                    blockSep = true;
            }
            else
            {
                const float tTemp = -distToPlane / denominator;
                const bool con = 0.0f > denominator;
                const bool con0 = tTemp > tFirst;
                const bool con1 = tLast > tTemp;
                if (con && con0)
                    newFirst = max(newFirst, tTemp);
                if (con1 && !con)
                    newLast = min(newLast, tTemp);
            }
        }

        if (blockSep)
            return false;

        tFirst = newFirst;
        tLast = newLast;

        if (tFirst > tLast)
            return false;
    }

    tEnter = tFirst;
    tExit = tLast;
    return true;
}

// ---------------------------------------------------------------------------
// generatedFaceContacts (npCvxFaceContacts.slang L196-235 / cu L1099-1141):
// two witness-vertex rays cast into hull B, contacts written straight to the
// manifold at slot = prior count (the ray0-miss -> ray1-in-slot-0 quirk is
// inherent to that indexing, exactly as in the .cu).
// ---------------------------------------------------------------------------
uint generatedFaceContacts(inout CollideScratch s)
{
    float tEnter = 0.0f, tExit = 0.0f;
    const float inflatedRadius = s.contactDistance;
    const float3 dir = -s.gjkOutput.direction;

    const float3 vertexSpaceDir = shape2Vertex(-dir, s.scale1, s.rot1);
    const float3 p0 = shape2Vertex(s.vA[0].xyz, s.scale1, s.rot1);
    const float3 p1 = shape2Vertex(s.vA[1].xyz, s.scale1, s.rot1);

    const uint64_t planesBase = s.convexPtr1 + uint64_t(planesByteOff(s.nbVertices1));

    Ptr<float4> mNp = Ptr<float4>(s.manifoldBase);
    Ptr<float4> mA = Ptr<float4>(s.manifoldBase + 64u);
    Ptr<float4> mB = Ptr<float4>(s.manifoldBase + 128u);

    uint numContacts = 0u;

    if (intersectRayPolyhedron(p0, vertexSpaceDir, planesBase, s.nbPolygons1, tEnter, tExit) && inflatedRadius >= tEnter)
    {
        const float3 va = transformInvPoint(s.aToBq, s.aToBp, s.vA[0].xyz);
        const float3 vb = s.vA[0].xyz - dir * tEnter;
        mA[numContacts] = float4(va, 0.0f);
        mB[numContacts] = float4(vb, 0.0f);
        mNp[numContacts] = float4(dir.x, dir.y, dir.z, tEnter);
        numContacts++;
    }

    if (intersectRayPolyhedron(p1, vertexSpaceDir, planesBase, s.nbPolygons1, tEnter, tExit) && inflatedRadius >= tEnter)
    {
        const float3 va = transformInvPoint(s.aToBq, s.aToBp, s.vA[1].xyz);
        const float3 vb = s.vA[1].xyz - dir * tEnter;
        mA[numContacts] = float4(va, 0.0f);
        mB[numContacts] = float4(vb, 0.0f);
        mNp[numContacts] = float4(dir.x, dir.y, dir.z, tEnter);
        numContacts++;
    }

    return numContacts;
}

// ---------------------------------------------------------------------------
// getPolygonIndexFromLocalWitness (npCvxEEDriver.slang L136-201, T138),
// planes from the blob over convexPtr1.
// ---------------------------------------------------------------------------
uint getPolygonIndexBlob(inout CollideScratch s, uint pOff,
                         float3 witnessPt, float3 n, float radius, float toleranceLength)
{
    const uint64_t baseBlob = s.convexPtr1;
    const uint nbPolygons = s.nbPolygons1;
    const float3 scale = s.scale1;
    const float4 rot = s.rot1;

    const float margin = radius * 0.01f;
    const float lowerEps = toleranceLength * PCM_LOWER_EPS;
    const float upperEps = toleranceLength * PCM_UPPER_EPS;
    const float tolerance = clamp(margin, lowerEps, upperEps);
    const float3 localWitness = shape2Vertex(witnessPt, scale, rot);
    const float eps = -tolerance;

    float laneMin[32];
    float laneMax[32];
    uint laneIdx[32];
    for (uint L = 0u; L < WARP_SIZE; ++L)
    {
        float minDist = PXG_MAX_F32;
        float maxDist = -PXG_MAX_F32;
        uint index = 0u;
        for (uint i = L; i < nbPolygons; i += WARP_SIZE)
        {
            const float4 pl = blobLoadFloat4(baseBlob, pOff + 16u * i);
            const float pd = dot(float3(pl.x, pl.y, pl.z), localWitness) + pl.w;
            const float toleranceDist = pd >= eps ? pd : PXG_MAX_F32;
            minDist = min(toleranceDist, minDist);
            if (toleranceDist > maxDist) { maxDist = toleranceDist; index = i; }
        }
        laneMin[L] = minDist; laneMax[L] = maxDist; laneIdx[L] = index;
    }

    float minDist = PXG_MAX_F32;
    for (uint L = 0u; L < WARP_SIZE; ++L) minDist = min(minDist, laneMin[L]);

    if (minDist == PXG_MAX_F32)
    {
        float maxDist = -PXG_MAX_F32;
        for (uint L = 0u; L < WARP_SIZE; ++L) maxDist = max(maxDist, laneMax[L]);
        for (uint L = 0u; L < WARP_SIZE; ++L) if (laneMax[L] == maxDist) return laneIdx[L];
        return 0u;
    }

    float laneProj[32];
    uint laneProjIdx[32];
    for (uint L = 0u; L < WARP_SIZE; ++L)
    {
        float bestProjection = PXG_MAX_F32;
        uint index = laneIdx[L];
        for (uint i = L; i < nbPolygons; i += WARP_SIZE)
        {
            const float4 pl = blobLoadFloat4(baseBlob, pOff + 16u * i);
            const float3 pn = float3(pl.x, pl.y, pl.z);
            const float pd = dot(pn, localWitness) + pl.w;
            const float toleranceDist = pd >= eps ? pd : PXG_MAX_F32;
            if (tolerance > (toleranceDist - minDist))
            {
                const float3 planeN = getNormalized(shape2Vertex(pn, scale, rot));
                const float proj = dot(planeN, n);
                if (bestProjection > proj) { index = i; bestProjection = proj; }
            }
        }
        laneProj[L] = bestProjection; laneProjIdx[L] = index;
    }
    float bestProjection = PXG_MAX_F32;
    for (uint L = 0u; L < WARP_SIZE; ++L) bestProjection = min(bestProjection, laneProj[L]);
    for (uint L = 0u; L < WARP_SIZE; ++L) if (laneProj[L] == bestProjection) return laneProjIdx[L];
    return 0u;
}

// convex-convex generateEE (npCvxEEDriver.slang L204-242, T134)
bool generateEE(float3 p, float3 q, float3 normal, float3 a, float3 b,
                float4 aToBq, float3 aToBp, float inflatedRadius,
                out float3 glocalA, out float3 glocalB, out float4 gnormalPen)
{
    glocalA = float3(0, 0, 0); glocalB = float3(0, 0, 0); gnormalPen = float4(0, 0, 0, 0);
    const float expandedRatio = 0.005f;
    const float3 ab = b - a;
    const float3 n = cross(ab, normal);
    const float d = dot(n, a);
    const float np = dot(n, p);
    const float nq = dot(n, q);
    const float signP = np - d;
    const float signQ = nq - d;
    const float temp = signP * signQ;
    if (temp > 0.0f) return false;
    const float3 pq = q - p;
    const float npq = dot(n, pq);
    if (npq == 0.0f) return false;
    const float segTValue = (d - np) / npq;
    const float3 localPointA = pq * segTValue + p;
    const float3 perNormal = cross(normal, pq);
    const float3 ap = localPointA - a;
    const float nom = dot(perNormal, ap);
    const float denom = dot(perNormal, ab);
    const float tValue = nom / denom;
    const float maxT = 1.0f + expandedRatio;
    const float minT = -expandedRatio;
    if (tValue > maxT || minT > tValue) return false;
    const float3 v = ap - ab * tValue;
    const float signedDist = dot(v, normal);
    if (inflatedRadius >= signedDist)
    {
        glocalB = localPointA - v;
        glocalA = transformInvPoint(aToBq, aToBp, localPointA);
        gnormalPen = float4(normal.x, normal.y, normal.z, signedDist);
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// generatedContactsEEContacts (npCvxEEDriver.slang L250-320 / cu L1198-1271):
// A-edge (vA[0], vA[1]) vs the edges of hull B's best-matching polygon,
// appended to the manifold at index >= numContactsIn with the normal/pen
// NEGATED on store, capped at 4.
// ---------------------------------------------------------------------------
uint generatedContactsEEContacts(inout CollideScratch s, uint numContactsIn)
{
    uint numContacts = numContactsIn;
    const float3 normal = s.gjkOutput.direction;

    const uint pOff = planesByteOff(s.nbVertices1);
    const uint dOff = polyDescByteOff(s.nbVertices1, s.nbPolygons1);
    const uint vdOff = vertexDataByteOff(s.nbVertices1, s.nbPolygons1, s.nbEdges1);

    const uint bestPlaneBIdx = getPolygonIndexBlob(s, pOff, s.gjkOutput.closestPointB, normal,
                                                   s.inSphereRadius1, s.toleranceLength);
    const uint polyDesc = blobLoadU32(s.convexPtr1, dOff + 4u * bestPlaneBIdx);
    const uint nbVerts = getNbVertsW(polyDesc);
    const uint vref8 = getVRef8W(polyDesc);

    const float inflatedRadius = s.contactDistance;

    Ptr<float4> mNp = Ptr<float4>(s.manifoldBase);
    Ptr<float4> mA = Ptr<float4>(s.manifoldBase + 64u);
    Ptr<float4> mB = Ptr<float4>(s.manifoldBase + 128u);

    // warp-strided over polygon edges, block size WARP_SIZE
    for (uint i = 0u; i < nbVerts; i += WARP_SIZE)
    {
        uint offset = 0u; // running warpScanExclusive within this block
        const uint blockNC = numContacts; // numContacts at block start (write base)
        for (uint lane = 0u; lane < WARP_SIZE; ++lane)
        {
            const uint start = i + lane;
            if (start >= nbVerts) continue;
            const uint end = (start + 1u) % nbVerts;
            const uint i0 = blobLoadU8(s.convexPtr1, vdOff + vref8 + start);
            const uint i1 = blobLoadU8(s.convexPtr1, vdOff + vref8 + end);
            const float3 v0 = s.vB[i0].xyz;
            const float3 v1 = s.vB[i1].xyz;
            float3 lA, lB; float4 lNp;
            const bool hit = generateEE(s.vA[0].xyz, s.vA[1].xyz, normal, v0, v1,
                                        s.aToBq, s.aToBp, inflatedRadius, lA, lB, lNp);
            if (hit)
            {
                if (offset < 4u)
                {
                    const uint idx = blockNC + offset;
                    if (idx < 4u)
                    {
                        mA[idx] = float4(lA, 0.0f);
                        mB[idx] = float4(lB, 0.0f);
                        mNp[idx] = float4(-lNp.x, -lNp.y, -lNp.z, -lNp.w);
                    }
                }
                offset += 1u;
            }
        }
        numContacts = min(numContacts + offset, 4u);
    }

    return numContacts;
}

// ---------------------------------------------------------------------------
// collide (cu L1275-1465), SPHERE + CAPSULE type0 arms only (the convex-hull
// arm is deferred; the stage2 caller gates on type0).  verts0Ptr == 0 selects
// the synthetic verts0Synth vertices; ws13/14/15 are manifold moreData words
// 13-15 (nbWarmStartPoints / warmStartA / warmStartB) read by the caller.
// ---------------------------------------------------------------------------
public bool collide(inout CollideScratch s,
                    float4 verts0Synth[2], uint64_t verts0Ptr, uint64_t verts1Ptr,
                    uint ws13, uint ws14, uint ws15,
                    inout uint numManifoldContacts)
{
    const float convergenceRatio = 1.0f - 0.000225f;

    // 1. prepareVertices (cu L1294-1295)
    cvxPrepareVertices(s, verts0Synth, verts0Ptr, verts1Ptr);

    // 2. warm-start unpack/seed (cu L1300-1340): shift = tI*8 bits; encode
    //    0x3f800000 | b<<8 | a into the cache image index slots.
    const uint numManifoldWarmStartPoints = ws13;
    for (uint tI = 0u; tI < numManifoldWarmStartPoints; ++tI)
    {
        const uint aIndex = (ws14 >> (tI << 3u)) & 0xFFu;
        const uint bIndex = (ws15 >> (tI << 3u)) & 0xFFu;
        const float3 cv = s.vA[aIndex].xyz - s.vB[bIndex].xyz;
        s.cachedData.v[(tI << 2) + 0u] = cv.x;
        s.cachedData.v[(tI << 2) + 1u] = cv.y;
        s.cachedData.v[(tI << 2) + 2u] = cv.z;
        s.cachedData.v[(tI << 2) + 3u] = asfloat(encodeIndices(aIndex, bIndex));
    }
    s.cachedData.size = numManifoldWarmStartPoints;

    // 3. initialDir (cu L1345-1350): aToB.p if nonzero, else (1,0,0),
    //    UNNORMALIZED (GJK lazy-normalizes).
    float3 initialDir = float3(1.0f, 0.0f, 0.0f);
    if (dot(s.aToBp, s.aToBp) > 0.0f)
        initialDir = s.aToBp;

    // 4. runGJK (cu L1352); the re-pack words are DISCARDED (never stored).
    uint rp13, rp14, rp15;
    const uint gjkResult = runGJK(initialDir, s, convergenceRatio, rp13, rp14, rp15);

    // 6. eDISTANT early-out (cu L1354-1357)
    if (gjkResult == GJK_DISTANT)
        return false;

    // 7. anomaly gate on the FIRST GJK direction (cu L1359-1360; EPA
    //    overwrites gjkOutput, so snapshot first; PxVec3::dot associates
    //    (x+y)+z).
    const float3 cpDir = s.gjkOutput.closestPointDir;
    const float3 dirFirst = s.gjkOutput.direction;
    const float dp = (cpDir.x * dirFirst.x + cpDir.y * dirFirst.y) + cpDir.z * dirFirst.z;
    const bool anomaly = (gjkResult == GJK_CLOSE) && (dp < 0.999f);
    bool separated = (gjkResult == GJK_CLOSE);

    // 8/9. EPA gate + degenerate re-run (cu L1362-1396)
    if (!separated || anomaly)
    {
        const float releps = 0.5f * (s.inSphereRadius0 + s.inSphereRadius1) * (1.0f - convergenceRatio);
        const uint epaResult = epaCore(s, releps);
        separated = (epaResult == GJK_CLOSE);

        if (s.gjkOutput.degenerate)
        {
            // re-run gjk/epa cold with the SAME initialDir
            s.cachedData.size = 0u;
            runGJK(initialDir, s, convergenceRatio, rp13, rp14, rp15);
            const uint epaResult2 = epaCore(s, releps);
            separated = (epaResult2 == GJK_CLOSE);
        }
    }
    // (separated is not consumed by the sphere/capsule arms — same as the .cu)

    // 10. type0 dispatch
    if (s.type0 == EGT_SPHERE)
    {
        // sphere witness store (cu L1398-1445; npCvxCollideDecide L131-144):
        // uses the FINAL post-EPA direction; shfl-order sep = (x+y)+z.
        const float3 dirF = s.gjkOutput.direction;
        const float3 wA = s.gjkOutput.closestPointA;
        const float3 wB = s.gjkOutput.closestPointB;
        const float sepX = (wB.x - wA.x) * dirF.x;
        const float sepY = (wB.y - wA.y) * dirF.y;
        const float sepZ = (wB.z - wA.z) * dirF.z;
        const float sep = (sepX + sepY) + sepZ;

        Ptr<float4> mNp = Ptr<float4>(s.manifoldBase);
        Ptr<float4> mA = Ptr<float4>(s.manifoldBase + 64u);
        Ptr<float4> mB = Ptr<float4>(s.manifoldBase + 128u);
        mA[0] = float4(0.0f, 0.0f, 0.0f, 0.0f); // sphere centre
        mB[0] = float4(wB, 0.0f);
        mNp[0] = float4(-dirF, sep);

        numManifoldContacts = 1u;
    }
    else if (s.type0 == EGT_CAPSULE)
    {
        // capsule arm (cu L1446-1457) — face contacts, then EE fallback
        uint numContacts = generatedFaceContacts(s);
        if (numContacts < 2u)
            numContacts = generatedContactsEEContacts(s, numContacts);
        numManifoldContacts = numContacts;
    }
    // else: convex-hull generateConvexContacts (cu L1458-1462) — DEFERRED;
    // unreachable behind the stage2 type gate.

    return true;
}
