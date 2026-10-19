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

// pxgSlangPolyClip.slang
//
// `polyClip` + `addContacts` (convexTriangle.cuh:918-1058) — the Sutherland-
// Hodgman-style clip of a convex face against a triangle that IS the per-triangle
// core of rigid convex-vs-mesh contact generation.  P8.31 corrected the earlier
// claim that this core was GJK/EPA; P8.34 established that it cannot be lifted out
// of the Phase-1 shader by the P8.24 extract-and-prove-by-opcode pattern, because
// there the emission is fused with the control flow.  So this is a PORT from the
// `.cu`, with its own fixture and its own oracle.
//
// THE SHAPE.  One warp per pair.  Lane i holds vertex i of poly0 (the convex
// face, up to 32 verts) and, for i < 3, vertex i of poly1 (the triangle).  Points
// are generated where a vertex of one polygon lies inside the other (including on
// an edge) and where two edges cross.  The early-outs are NOT an optimisation:
// with two polygons exactly on top of one another they are what stops the same
// point being generated twice (the comment at convexTriangle.cuh:911 says so).
//
// ONE WARP PER GROUP.  Every consumer MUST declare `[numthreads(32,1,1)]`.
// `polyClip` returns early on `!__any_sync(out1)` and again on `!__any_sync(out0)`;
// those are warp-uniform but NOT group-uniform, so with two warps in a group one
// could leave while the other runs on into a barrier — and llvmpipe does not
// error on a divergent barrier, it HANGS.  The CUDA kernel packs
// NP_TRIMESH_WARPS_PER_BLOCK = 2 warps per block; a consumer here dispatches twice
// as many groups instead, which is legal because pairs are independent.
//
// THE 32-LANE VIRTUAL WARP.  llvmpipe's subgroup is 8, so every `__shfl_sync` is a
// groupshared read and every `__ballot_sync` an `InterlockedOr` on a groupshared
// word between two barriers.  Values that are shuffled are staged ONCE at the
// point the CUDA computes them, so the shuffles themselves cost nothing extra:
// v0/v1 at entry, e1planeN when it is formed, in0/out0 when they are.
//
// TWO DELIBERATE OMISSIONS, both dead in the CUDA:
//   * `triEdgeMask` is a parameter of the CUDA `polyClip` and its only use,
//     `e1cross &= triEdgeMask`, is commented out at convexTriangle.cuh:995.  The
//     parameter is kept here so the signature still mirrors the original, and it
//     is just as unused.
//   * `s0d = shfl(s0, next) - s0` (:1023) feeds only the commented-out `lambda`
//     formula at :1036.  Computing it would cost a staging barrier for nothing, so
//     it is not computed.  Neither omission can move a single output bit.

static const uint PC_WARP = 32u;
public static const uint PC_NUM_TMP_CONTACTS_PER_PAIR = 32u;   // convexNpCommon.h:33
static const float PC_MAX_F32 = 3.4028234663852886e+38f;       // PX_MAX_F32

groupshared uint gsPcWord;                 // ballot / and-reduce / or-reduce
groupshared float gsPcV0[3 * PC_WARP];     // poly0 verts, staged once
groupshared float gsPcV1[3 * PC_WARP];     // poly1 (triangle) verts, staged once
groupshared float gsPcE1N[3 * PC_WARP];    // triangle edge-plane normals
groupshared uint gsPcIn0[PC_WARP];
groupshared uint gsPcOut0[PC_WARP];

// The contact staging buffer.  `s_contactsTransposed` in the CUDA — the name says
// transposed but the layout written here is interleaved, 4 floats (x, y, z, sep)
// per contact, exactly as convexTriangle.cuh:926-929 writes it.
groupshared float gsPcContacts[4u * PC_NUM_TMP_CONTACTS_PER_PAIR];

// --------------------------------------------------------------------------
// warp primitives
// --------------------------------------------------------------------------
// THE LEADING BARRIER IS NOT REDUNDANT.  Each of these returns by reading
// gsPcWord after the trailing barrier, and the NEXT reduction opens by resetting
// that same word — so without a barrier first, lane 0 can reset the word while
// another lane has not yet performed its read of the previous result.  That is a
// real race and it bit this port: `addContacts` runs two ballots back to back, and
// with the reset unguarded the second one clobbered the first, every lane read an
// exclusive-scan base of 0, and two contacts were written on top of each other at
// the same index — while `nbContacts` still counted two, so the COUNT looked
// right and only the second slot came back empty.
uint pcReduceOr(uint lane, uint v)
{
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) gsPcWord = 0u;
    GroupMemoryBarrierWithGroupSync();
    InterlockedOr(gsPcWord, v);
    GroupMemoryBarrierWithGroupSync();
    return gsPcWord;
}

uint pcReduceAnd(uint lane, uint v)
{
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) gsPcWord = 0xffffffffu;
    GroupMemoryBarrierWithGroupSync();
    InterlockedAnd(gsPcWord, v);
    GroupMemoryBarrierWithGroupSync();
    return gsPcWord;
}

uint pcBallot(uint lane, bool pred)
{
    return pcReduceOr(lane, pred ? (1u << lane) : 0u);
}

// warpScanExclusive(mask, lane) = popc(mask & ((1<<lane)-1))   (reduction.cuh:592)
uint pcScanExclusive(uint mask, uint lane)
{
    return countbits(mask & ((1u << lane) - 1u));
}

// THE MASK MIRRORS `__shfl_sync`.  CUDA specifies that a source lane outside
// [0, width-1] is taken modulo width, and polyClip relies on it: the "next vertex"
// index is `tI + 1 == nbVerts0 ? 0 : tI + 1`, which for lane 31 with nbVerts0 < 32
// is 32 — a legal shuffle that lands back on lane 0.  Without the mask the same
// expression indexes one past the end of the x-component run and silently reads
// lane 0's Y instead.  (This was live for exactly one lane and the fixture could
// not see it while the padding lanes all held the same vertex.)
float3 pcReadV0(uint s) { const uint src = s & (PC_WARP - 1u); return float3(gsPcV0[src], gsPcV0[PC_WARP + src], gsPcV0[2u * PC_WARP + src]); }
float3 pcReadV1(uint s) { const uint src = s & (PC_WARP - 1u); return float3(gsPcV1[src], gsPcV1[PC_WARP + src], gsPcV1[2u * PC_WARP + src]); }
float3 pcReadE1N(uint s) { const uint src = s & (PC_WARP - 1u); return float3(gsPcE1N[src], gsPcE1N[PC_WARP + src], gsPcE1N[2u * PC_WARP + src]); }

public float4 pcContact(uint index)
{
    return float4(gsPcContacts[4u * index + 0u], gsPcContacts[4u * index + 1u],
                  gsPcContacts[4u * index + 2u], gsPcContacts[4u * index + 3u]);
}

// --------------------------------------------------------------------------
// addContacts (convexTriangle.cuh:918)
//
// `nbContacts` is a per-lane variable in the CUDA too, but every lane adds the
// same popcount, so it stays warp-uniform — which is what lets the caller use it
// as the running base of the exclusive scan.
// --------------------------------------------------------------------------
void pcAddContacts(uint lane, uint flags, float3 pos, float sep, float maxSep, inout uint nbContacts)
{
    // Inverting the test covers us for NaNs (the CUDA comment at :920).
    flags &= pcBallot(lane, !(sep > maxSep));

    const uint index = pcScanExclusive(flags, lane) + nbContacts;

    const bool add = ((flags & (1u << lane)) != 0u) && index < PC_NUM_TMP_CONTACTS_PER_PAIR;

    if (add)
    {
        gsPcContacts[4u * index + 0u] = pos.x;
        gsPcContacts[4u * index + 1u] = pos.y;
        gsPcContacts[4u * index + 2u] = pos.z;
        gsPcContacts[4u * index + 3u] = sep;
    }

    nbContacts += countbits(pcBallot(lane, add));
}

// --------------------------------------------------------------------------
// polyClip (convexTriangle.cuh:941).  plane0/poly0 is the convex face, plane1/
// poly1 the triangle; planes are (n.xyz, d) with distance(p) = dot(p,n) + d.
// --------------------------------------------------------------------------
public uint pcPolyClip(uint lane, float4 plane0, float3 v0, uint nbVerts0,
                       float4 plane1, float3 v1, float3 axis, float maxSep,
                       uint triEdgeMask, uint initContacts)
{
    const uint tI = lane;

    const float dnom0 = dot(plane1.xyz, axis);
    const float dnom1 = dot(plane0.xyz, axis);
    const float t0 = dot(v0, plane1.xyz) + plane1.w;
    const float t1 = dot(v1, plane0.xyz) + plane0.w;

    // If the point is beyond the other polygon's plane by more than the contact
    // distance it has to go, or it becomes a phantom contact (:956).
    const float s0 = ((t0 > maxSep) || (dnom0 == 0.0f)) ? PC_MAX_F32 : -t0 / dnom0;
    const float s1 = ((t1 > maxSep) || (dnom1 == 0.0f)) ? PC_MAX_F32 : t1 / dnom1;

    uint in0 = 0u, out0 = 0u;   // per poly0 POINT: status against the 3 triangle planes
    uint in1 = 0u, out1 = 0u;   // per poly0 PLANE: status of the 3 triangle points

    uint nbContacts = initContacts;

    // v0 and v1 are shuffled from here on; stage them once.
    gsPcV0[tI] = v0.x; gsPcV0[PC_WARP + tI] = v0.y; gsPcV0[2u * PC_WARP + tI] = v0.z;
    gsPcV1[tI] = v1.x; gsPcV1[PC_WARP + tI] = v1.y; gsPcV1[2u * PC_WARP + tI] = v1.z;
    GroupMemoryBarrierWithGroupSync();

    // ---- triangle points against poly0's planes --------------------------
    const float3 e0 = pcReadV0((tI + 1u == nbVerts0) ? 0u : tI + 1u) - v0;
    const float3 e0planeN = cross(e0, axis);
    const float e0planeD = dot(e0planeN, v0);

    for (uint j = 0u, bj = 1u; j < 3u; j++, bj <<= 1)
    {
        const float t = dot(pcReadV1(j), e0planeN) - e0planeD;
        in1 |= (t < 0.0f ? bj : 0u);
        out1 |= (t > 0.0f ? bj : 0u);
    }

    if (tI >= nbVerts0)
    {
        in1 = 0u;
        out1 = 0u;
    }

    // b = AND over the warp of ~out1: the triangle points inside EVERY poly0 plane.
    const uint b = pcReduceAnd(tI, ~out1);

    pcAddContacts(tI, b & 7u, v1 - s1 * axis, s1, maxSep, nbContacts);

    if (pcReduceOr(tI, out1) == 0u)   // all poly1 points inside poly0, so done
        return nbContacts;

    in1 |= in1 << 3; out1 |= out1 << 3;            // save poly1's edge crosses
    const uint e1cross = ((in1 & (out1 >> 1)) | ((in1 >> 1) & out1)) & 7u;
    // e1cross &= triEdgeMask;   // dead in the CUDA too (:995)

    // ---- poly0 points against the triangle's planes ----------------------
    const float3 e1 = pcReadV1((tI + 1u == 3u) ? 0u : tI + 1u) - v1;
    const float3 e1planeN = cross(axis, e1);
    const float e1planeD = dot(e1planeN, v1);

    gsPcE1N[tI] = e1planeN.x; gsPcE1N[PC_WARP + tI] = e1planeN.y; gsPcE1N[2u * PC_WARP + tI] = e1planeN.z;
    GroupMemoryBarrierWithGroupSync();

    for (uint j = 0u; j < nbVerts0; j++)
    {
        const float t = dot(pcReadV0(j), e1planeN) - e1planeD;
        const uint inB = pcBallot(tI, t < 0.0f);
        const uint outB = pcBallot(tI, t > 0.0f);
        if (j == tI)
        {
            in0 = inB;
            out0 = outB;
        }
    }

    in0 &= (1u << 3) - 1u;
    out0 &= (1u << 3) - 1u;

    pcAddContacts(tI, pcBallot(tI, out0 == 0u) & ((1u << nbVerts0) - 1u), v0, s0, maxSep, nbContacts);

    if (pcReduceOr(tI, out0) == 0u)   // all poly0 points inside poly1, so done
        return nbContacts;

    gsPcIn0[tI] = in0;
    gsPcOut0[tI] = out0;
    GroupMemoryBarrierWithGroupSync();

    const uint nextLane = ((tI + 1u == nbVerts0) ? 0u : tI + 1u) & (PC_WARP - 1u);
    const uint in0p = gsPcIn0[nextLane];
    const uint out0p = gsPcOut0[nextLane];
    const uint e0cross = (in0 & out0p) | (in0p & out0);

    // ---- edge-edge crossings ---------------------------------------------
    const uint c = e0cross & e1cross;

    uint a = pcReduceOr(tI, c);

    for (; a != 0u; a &= (a - 1u))
    {
        const uint index = firstbitlow(a);              // lowestSetIndex
        const float3 m = pcReadE1N(index);

        // Ray-trace from a poly0 edge to the triangle plane.
        const float denom = dot(m, e0);
        const float3 triVert0 = pcReadV1(index);
        const float lambda = dot(triVert0 - v0, m) / denom;
        const float3 p = v0 + lambda * e0;

        // 5.10+ (convexTriangle.cuh, 5.12 :1024-1103): the separation along `axis`
        // of d = p - triVert0.  The plane-normal formula t / dnom0 distorts as
        // dnom0 -> 0, so below |dnom0| = 1e-4 the 2x2 Gram/Cramer solve in the
        // basis {e1, axis} is used instead, and the contact is discarded when
        // |m|^2 / |e1|^2 (sin^2 of the e1/axis angle) is below 1e-12.  dnom0 is
        // warp-uniform, so the branch is too.
        const float3 d = p - triVert0;

        float sep;
        if (abs(dnom0) > 1.0e-4f)
        {
            const float t = -dot(d, plane1.xyz);
            sep = (denom == 0.0f) ? PC_MAX_F32 : t / dnom0;
        }
        else
        {
            // shuffle(FULL_MASK, e1, index): lane `index`'s e1 is
            // v1[next(index)] - v1[index], rebuilt from the staged v1 with the
            // same subtraction (index < 3, so next() is the CUDA's tI + 1 == 3 ? 0).
            const float3 e1FromShuffle = pcReadV1((index + 1u == 3u) ? 0u : index + 1u) - pcReadV1(index);

            const float e1MagnSqr = dot(e1FromShuffle, e1FromShuffle);
            const float mMagnSqr = dot(m, m);
            const float sepTmp = -((e1MagnSqr * dot(d, axis)) - (dot(axis, e1FromShuffle) * dot(d, e1FromShuffle))) / mMagnSqr;

            // discard the contact for ill-defined configurations
            sep = ((denom == 0.0f) || ((mMagnSqr / e1MagnSqr) < 1.0e-12f)) ? PC_MAX_F32 : sepTmp;
        }

        pcAddContacts(tI, pcBallot(tI, (c & (a & (0u - a))) != 0u), p, sep, maxSep, nbContacts);
    }

    return nbContacts;
}
