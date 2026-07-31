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

// pxgSlangEpa — the EPA polytope expansion (epa.cuh), extracted from npEpa.slang for the
// same reason pxgSlangGjk was extracted from npGjk.slang: `squawk::epa` is ONE CUDA routine
// that the leaf test and every engine-path contact-gen kernel must run identically.  See
// pxgSlangGjk's header for the sharing rule.
//
// Nothing but the vertex-cloud plumbing changed: `gVA[vbase + i]` becomes `a.v[i]` on the
// GjkCloud the caller supplies, and the three helpers that carried `vbase, nbA, nbB` take
// the clouds instead.  EPA_MAX_VERTS is 18, so the polytope lives in per-thread arrays and
// needs no shared storage.
module pxgSlangEpa;

import pxgSlangGjk;   // GjkCloud, GJK_MAX_CLOUD, and the GJK_* result enum

public struct EpaOut
{
    uint result; uint degenerate; uint p0, p1;
    float lowerBound; float upperBound; float q0, q1;
    float4 direction; float4 closestPointA; float4 closestPointB;
}


// npEpa.slang
//
// Slang port of the full epa() driver (epa.cuh :388-526) — tranche 132.  EPA
// (Expanding Polytope Algorithm) runs after GJK reports overlap: it grows the
// simplex into a polytope of the Minkowski difference, expanding toward the
// origin until it finds the closest face (the minimum translational distance).
// This wires together the previously-ported seed functions epaStart4 (130) /
// epaStart123 (131) with the intricate replaceTriangle topology update
// (epa.cuh :64-251 — a ballot flood-fill of the triangles "silhouetted" by the
// new vertex, a single-thread linked-list flatten of the silhouette loop, and a
// cone of new triangles) and the main loop that repeatedly picks the closest
// live triangle, adds its support vertex, and calls replaceTriangle until the
// support distance meets the face distance within releps.
//
// One thread simulates the 32-lane warp (up to EPA_MAX_FACES = 32 triangles) with
// per-lane arrays; the EpaScratch polytope (up to EPA_MAX_VERTS = 18 vertices, its
// linked-list/adjacency buffers) lives in per-thread arrays.  With integer convex
// hulls every dot/cross is exact, magnitude/getNormalized use the correctly-rounded
// sqrt + 1/sqrt forms, the support argmax and all bitmap/topology ops are exact,
// so the whole EPA run — its branch decisions, iteration count and witness points —
// is bit-for-bit reproducible against the CPU reference.

static const uint EPA_MAX_VERTS = 18u;
static const uint GJK_CLOSE = 1u;
static const uint GJK_OVERLAP = 2u;
static const float EPA_MAX_F32 = 3.4028234663852886e38f;


// EpaScratch polytope (per-thread)
struct Scratch
{
    float3 verts[18];
    uint idx[18];   // PxU16 index words
    uint nxt[18];   // linked-list next
    uint edg[18];   // silhouette-edge -> cone-tri fixup
    uint lp[18];    // flattened silhouette loop
    float3 bestDir;
}
// per-triangle (lane) state
struct Lanes
{
    float3 normal[32];
    float dist[32];
    uint vv[32]; // Vec3I8 vertex indices
    uint aa[32]; // Vec3I8 adjacencies
}



// ---- scalar helpers --------------------------------------------------------
float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 cross3(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
float3 absX(float3 a) { return float3(abs(a.x), abs(a.y), abs(a.z)); }
float magSq3(float3 v) { return dot3(v, v); }
float mag3(float3 v) { return sqrt(dot3(v, v)); }
float3 getNormalized(float3 v) { const float m = dot3(v, v); if (m > 0.0f) { const float r = 1.0f / sqrt(m); return float3(v.x * r, v.y * r, v.z * r); } return float3(0.0f, 0.0f, 0.0f); }

// Vec3I8
uint mk3(uint a, uint b, uint c) { return (a & 0xffu) | ((b & 0xffu) << 8) | ((c & 0xffu) << 16); }
uint g3(uint p, uint i) { return (p >> (i * 8u)) & 0xffu; }
uint s3(uint p, uint i, uint val) { return (p & ~(0xffu << (i * 8u))) | ((val & 0xffu) << (i * 8u)); }

// support argmax over a cloud (vec3 dir, lowest-index tie-break)
uint supportIdxA(float3 dir, GjkCloud a)
{
    const uint nb = a.n;
    uint bestN = 0xFFFFFFFFu; float bestDist = -EPA_MAX_F32;
    for (uint i = 0u; i < nb; i += 32u)
    {
        float d[32];
        for (uint lane = 0u; lane < 32u; ++lane) { const uint n = i + lane; d[lane] = (n < nb) ? dot3(dir, a.v[n]) : -EPA_MAX_F32; }
        float cmax = -EPA_MAX_F32; for (uint lane = 0u; lane < 32u; ++lane) cmax = max(cmax, d[lane]);
        if (cmax > bestDist) { uint ll = 0u; for (uint lane = 0u; lane < 32u; ++lane) if (d[lane] == cmax) { ll = lane; break; } bestDist = cmax; bestN = i + ll; }
    }
    return bestN;
}
uint supportIdxB(float3 dir, GjkCloud b)
{
    const uint nb = b.n;
    uint bestN = 0xFFFFFFFFu; float bestDist = -EPA_MAX_F32;
    for (uint i = 0u; i < nb; i += 32u)
    {
        float d[32];
        for (uint lane = 0u; lane < 32u; ++lane) { const uint n = i + lane; d[lane] = (n < nb) ? dot3(dir, b.v[n]) : -EPA_MAX_F32; }
        float cmax = -EPA_MAX_F32; for (uint lane = 0u; lane < 32u; ++lane) cmax = max(cmax, d[lane]);
        if (cmax > bestDist) { uint ll = 0u; for (uint lane = 0u; lane < 32u; ++lane) if (d[lane] == cmax) { ll = lane; break; } bestDist = cmax; bestN = i + ll; }
    }
    return bestN;
}
void newPoint(float3 dir, float3 ref, GjkCloud a, GjkCloud b, out float3 vertex, out uint indices)
{
    const uint iAp = supportIdxA(dir, a), iBp = supportIdxB(float3(-dir.x, -dir.y, -dir.z), b);
    const uint iAn = supportIdxA(float3(-dir.x, -dir.y, -dir.z), a), iBn = supportIdxB(dir, b);
    const float3 vp = a.v[iAp] - b.v[iBp];
    const float3 vn = a.v[iAn] - b.v[iBn];
    if (abs(dot3(vp - ref, dir)) > abs(dot3(vn - ref, dir))) { vertex = vp; indices = ((iBp << 8) | (iAp & 0xffu)) & 0xffffu; }
    else { vertex = vn; indices = ((iBn << 8) | (iAn & 0xffu)) & 0xffffu; }
}

// schlock::barycentrics
void barycentrics(float3 a, float3 b, float3 c, float3 p, out float u, out float v, out float w)
{
    const float3 m = cross3(b - a, c - a);
    const float3 ma = absX(m);
    const float3 a1 = cross3(b - p, c - p);
    const float3 a2 = cross3(c - p, a - p);
    float d;
    if (ma.x >= ma.y && ma.x >= ma.z) { d = 1.0f / m.x; u = a1.x * d; v = a2.x * d; }
    else if (ma.y >= ma.z)            { d = 1.0f / m.y; u = a1.y * d; v = a2.y * d; }
    else                              { d = 1.0f / m.z; u = a1.z * d; v = a2.z * d; }
    w = 1.0f - u - v;
}

// ---- seed: epaStart4 (tranche 130) -----------------------------------------
static const uint SV4_0 = 0x020100u, SV4_1 = 0x000103u, SV4_2 = 0x010203u, SV4_3 = 0x020003u;
static const uint SA4_0 = 0x010302u, SA4_1 = 0x020300u, SA4_2 = 0x030100u, SA4_3 = 0x010200u;
void epaStart4(float input[32], inout Scratch S, inout Lanes L)
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

// ---- seed: epaStart123 (tranche 131) ---------------------------------------
void epaStart123(float input[32], uint size, GjkCloud a, GjkCloud b, inout Scratch S, inout Lanes L)
{
    float verts[12]; uint indices[4];
    for (uint tI = 0u; tI < 16u; ++tI) { if ((tI & 3u) < 3u) verts[tI - tI / 4u] = input[tI]; else indices[tI >> 2] = asuint(input[tI]) & 0xffffu; }
    float3 vertex0 = float3(verts[0], verts[1], verts[2]);
    float3 vertex1 = float3(verts[3], verts[4], verts[5]);
    if (size < 2u) { float3 nv; uint ni; newPoint(float3(1.0f, 0.0f, 0.0f), vertex0, a, b, nv, ni); vertex1 = nv; verts[3] = nv.x; verts[4] = nv.y; verts[5] = nv.z; indices[1] = ni; }
    float3 vertex2 = float3(verts[6], verts[7], verts[8]);
    if (size < 3u)
    {
        const float3 dir = vertex1 - vertex0;
        const float3 axis = (abs(dir.x) < abs(dir.y)) ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
        float3 nv; uint ni; newPoint(cross3(axis, dir), vertex0, a, b, nv, ni); vertex2 = nv; verts[6] = nv.x; verts[7] = nv.y; verts[8] = nv.z; indices[2] = ni;
    }
    S.verts[0] = float3(verts[0], verts[1], verts[2]); S.verts[1] = float3(verts[3], verts[4], verts[5]); S.verts[2] = float3(verts[6], verts[7], verts[8]);
    S.idx[0] = indices[0]; S.idx[1] = indices[1]; S.idx[2] = indices[2];
    const float3 e01 = vertex1 - vertex0, e12 = vertex2 - vertex1, e02 = vertex2 - vertex0;
    float3 normal;
    if (mag3(e01) <= mag3(e02)) normal = getNormalized(cross3(e01, e12));
    else                        normal = getNormalized(cross3(e02, e12));
    const float distance = dot3(normal, vertex0);
    L.normal[0] = normal; L.dist[0] = distance; L.vv[0] = 0x020100u; L.aa[0] = 0x010101u;
    L.normal[1] = -normal; L.dist[1] = -distance; L.vv[1] = 0x020001u; L.aa[1] = 0x000000u;
}

// ---- replaceTriangle (epa.cuh :64-251) -------------------------------------
bool replaceTriangle(inout Scratch S, inout Lanes L, int nbVertices, int triIndex, float lowerBound, inout uint liveB, float releps)
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
            if (magSq3(e01) < magSq3(e02)) n = cross3(e01, p2 - p1);
            else                           n = cross3(e02, p2 - p1);
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


// The driver.  `input` is the 32-float GjkCachedData simplex GJK left behind, `size` its
// vertex count, `releps` the convergence epsilon the caller derives from the ratio.
public uint epaRun(float input[32], uint size, float releps, GjkCloud a, GjkCloud b, out EpaOut o)
{


    Scratch S;
    for (uint i = 0u; i < 18u; ++i) { S.verts[i] = float3(0.0f, 0.0f, 0.0f); S.idx[i] = 0u; S.nxt[i] = 0u; S.edg[i] = 0u; S.lp[i] = 0u; }
    S.bestDir = float3(0.0f, 0.0f, 0.0f);
    Lanes L;
    for (uint i = 0u; i < 32u; ++i) { L.normal[i] = float3(0.0f, 0.0f, 0.0f); L.dist[i] = 0.0f; L.vv[i] = 0u; L.aa[i] = 0u; }

    float lowerBound = -EPA_MAX_F32, upperBound = EPA_MAX_F32;
    uint bestVerts = 0u;
    int nbVertices; uint liveTriangles;

    if (size == 4u) { epaStart4(input, S, L); nbVertices = 4; liveTriangles = 0xfu; }
    else { epaStart123(input, size, a, b, S, L); nbVertices = 3; liveTriangles = 0x3u; }

    bool nondegenerate = true;
    [loop] while (nondegenerate && (uint)nbVertices < EPA_MAX_VERTS)
    {
        // shortest-distance live triangle
        float e = EPA_MAX_F32;
        for (uint tI = 0u; tI < 32u; ++tI) { const float tv = ((liveTriangles & (1u << tI)) != 0u) ? L.dist[tI] : EPA_MAX_F32; e = min(e, tv); }
        uint f = 0u;
        for (uint tI = 0u; tI < 32u; ++tI) { const float tv = ((liveTriangles & (1u << tI)) != 0u) ? L.dist[tI] : EPA_MAX_F32; if (e == tv) f |= (1u << tI); }
        const uint triIndex = firstbitlow(f);
        const float3 d = L.normal[triIndex];

        const uint indexA = supportIdxA(d, a), indexB = supportIdxB(float3(-d.x, -d.y, -d.z), b);
        const float3 supportVertex = a.v[indexA] - b.v[indexB];
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

    EpaOut o;
    o.result = 0u; o.p0 = 0u; o.p1 = 0u; o.q0 = 0.0f; o.q1 = 0.0f;
    o.degenerate = nondegenerate ? 0u : 1u;
    o.lowerBound = lowerBound; o.upperBound = upperBound;
    const float3 bestDir = S.bestDir;
    const float3 closestPoint = float3(bestDir.x * lowerBound, bestDir.y * lowerBound, bestDir.z * lowerBound);
    o.direction = float4(bestDir, 0.0f);

    const uint v0 = g3(bestVerts, 0u), v1 = g3(bestVerts, 1u), v2 = g3(bestVerts, 2u);
    float u, vv, ww;
    barycentrics(S.verts[v0], S.verts[v1], S.verts[v2], closestPoint, u, vv, ww);
    const float3 cpA = a.v[S.idx[v0] & 0xffu] * u + a.v[S.idx[v1] & 0xffu] * vv + a.v[S.idx[v2] & 0xffu] * ww;
    o.closestPointA = float4(cpA, 0.0f);
    o.closestPointB = float4(cpA - closestPoint, 0.0f);
    o.result = (lowerBound < 0.0f) ? GJK_CLOSE : GJK_OVERLAP;

    return o.result;
}
