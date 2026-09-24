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

// npTriTriFind.slang -- P9.83b-8 (3/n).
//
//   ttfRounds     ONE WORKGROUP OF 1024 THREADS PER CASE.  Drives the mesh-mesh
//                 kernel's inner loop exactly as cu:1062-1209 does -- gather with
//                 the register copy of the refinement count, reset it, read
//                 (ind, subInd) per thread, decide refinement, addToRefinementBuffer,
//                 thread 0 publishes the count, barrier, copy it back -- and records
//                 every round: count, start before and after, the whole index buffer,
//                 the refinement buffer.  The contact body between the gather and the
//                 refinement decision is 2/n's and is not here; the decision is a hash.
//   ttfCullProbe  serial lanes: ttfTriangleCullSphere's radius and sample point,
//                 bit for bit.
//
// The SDF sample is the TTF_SDF_DISTANCE hook (see pxgSlangTriTriFind), defined
// below to one of two fixture distances per case:
//   mode 0  a hash of (ind, sub): a primitive passes with probability passPermille,
//           a child (sub != 0) with childPermille -- 1000 makes every child pass,
//           which is what upstream candidate 60 needs.
//   mode 1  an analytic plane, dot(n, p) - d spelled out: the decision then depends
//           on the sample point, the radius and the tolerance, all mirrored on the
//           host bit for bit.
//
// Stamped NoContraction, like its siblings.

#define PXG_WARP_BLOCK 1024
#include "pxgSlangWarpInc.slang"

import pxgSlangCommon;
import pxgSlangSdfTexture;
import pxgSlangSdfCollision;
import pxgSlangContactReduceSerial;
import pxgSlangTriTriPatch;
import pxgSlangTriTriGeom;

#include "pxgSlangSdfFind.slang"
#define TTB_PATCH_GUARD 1
#include "pxgSlangTriTriBlock.slang"

struct TtfPush
{
    uint64_t cases;      // @0
    uint64_t out;        // @8
    uint64_t meshInd;    // @16
    uint64_t meshVerts;  // @24
    uint numCases;       // @32
    uint pad0, pad1, pad2;
}

[[vk::push_constant]] ConstantBuffer<TtfPush> gF;

// Case record: 2240 bytes.
//   float4 scale0, rot0, scale1, rot1, aToBp, aToBq     @0    (96)
//   float4 plane (xyz normal, w = d)                    @96
//   uint4  cfg0 (nbTriangles, startIndex, initialN, maxRounds)          @112
//   uint4  cfg1 (mode, passPermille, childPermille, refinePermille)     @128
//   uint4  cfg2 (seed, maxRefinementLevel, asuint(tolerance), refineAll) @144
//   uint4  pad                                          @160
//   uint2  seedSub[256]                                 @176  (2048) -> 2224, padded to 2240
struct TtfCase
{
    float4 scale0, rot0, scale1, rot1, aToBp, aToBq;
    float4 plane;
    uint4 cfg0, cfg1, cfg2, cfgPad;
}
static const uint TTF_CASE_STRIDE = 2240u;
static const uint TTF_CASE_SEED_OFFSET = 176u;

// Output record per case, in uints:
//   [0, 64)                       16 round headers: (count, startBefore, startAfter, numSubAfter)
//   [64, 64 + 16*2048)            16 index-buffer images, 1024 uint2 each
//   [.., + 16*512)                16 refinement-buffer images, 256 uint2 each
//   last 4                        (roundsRun, finalStart, finalNumSub, 0x51DE)
static const uint TTF_MAX_ROUNDS = 16u;
static const uint TTF_OUT_IDX = 64u;
static const uint TTF_OUT_SUB = TTF_OUT_IDX + TTF_MAX_ROUNDS * 2048u;
static const uint TTF_OUT_TAIL = TTF_OUT_SUB + TTF_MAX_ROUNDS * 512u;
static const uint TTF_OUT_STRIDE = TTF_OUT_TAIL + 4u;

static TtfCase sCase;   // the current workgroup's case, read by the distance hook

uint ttfHash(uint a, uint b, uint seed)
{
    uint x = a * 0x9E3779B1u ^ (b * 0x85EBCA77u) ^ seed;
    x ^= x >> 15;
    x *= 0x2C1B3C6Du;
    x ^= x >> 12;
    x *= 0x297A2D39u;
    x ^= x >> 15;
    return x;
}

float ttfFixtureDistance(uint ind, uint sub, float3 p)
{
    if (sCase.cfg1.x == 0u)
    {
        const uint permille = sub != 0u ? sCase.cfg1.z : sCase.cfg1.y;
        return (ttfHash(ind, sub, sCase.cfg2.x) % 1000u) < permille ? -1.0f : 1.0e30f;
    }
    return p.x * sCase.plane.x + p.y * sCase.plane.y + p.z * sCase.plane.z - sCase.plane.w;
}

#define TTF_SDF_DISTANCE(t, ind, sub, p) ttfFixtureDistance(ind, sub, p)
#include "pxgSlangTriTriFind.slang"

bool ttfRefineDecision(uint ind, uint sub)
{
    if (sCase.cfg2.w != 0u)
        return true;
    return (ttfHash(ind, sub, sCase.cfg2.x ^ 0x5EEDu) % 1000u) < sCase.cfg1.w;
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void ttfRounds(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    const uint c = gid.x;
    if (c >= gF.numCases) return;
    const uint tid = gtid.x;

    const uint64_t base = gF.cases + uint64_t(c) * uint64_t(TTF_CASE_STRIDE);
    sCase = Ptr<TtfCase>(base)[0];
    Ptr<uint> o = Ptr<uint>(gF.out) + uint64_t(c) * uint64_t(TTF_OUT_STRIDE);

    const uint nbTriangles = sCase.cfg0.x;
    const uint maxRounds = min(sCase.cfg0.w, TTF_MAX_ROUNDS);
    const float tolerance = asfloat(sCase.cfg2.z);
    SparseSdfTexture unusedTexture = {};   // the hook ignores it

    // The refinement state a previous round would have left.
    if (tid < TTF_SUBDIVIDE_COUNT)
        gTtfSubdivide[tid] = Ptr<uint2>(base + uint64_t(TTF_CASE_SEED_OFFSET))[tid];
    if (tid == 0u)
        gTtfNumSubdivide = sCase.cfg0.z;
    GroupMemoryBarrierWithGroupSync();

    uint numTrianglesToSubdividePerThread = gTtfNumSubdivide;   // the register copy (cu:1209)
    uint startIndex = sCase.cfg0.y;
    uint round = 0u;

    // cu:1064, on the register copy -- block-uniform, see the P9.83b-8 refine map.
    while ((startIndex < nbTriangles || numTrianglesToSubdividePerThread > 0u) && round < maxRounds)
    {
        const uint before = startIndex;
        const uint nbFound = ttfFindInterestingTriangles(tid, nbTriangles, gF.meshInd, gF.meshVerts,
                                                         sCase.scale0.xyz, sCase.rot0, sCase.scale1.xyz, sCase.rot1,
                                                         tolerance, unusedTexture,
                                                         sCase.aToBp.xyz, sCase.aToBq, startIndex,
                                                         numTrianglesToSubdividePerThread);
        numTrianglesToSubdividePerThread = 0u;   // cu:1080

        // cu:1082-1083
        const uint2 entry = ttfIndexGet(tid);
        const uint ind = tid < nbFound ? entry.x : nbTriangles;
        const uint subInd = tid < nbFound ? entry.y : 0u;

        Ptr<uint> img = o + TTF_OUT_IDX + round * 2048u;
        img[2u * tid] = entry.x;
        img[2u * tid + 1u] = entry.y;

        const bool needsRefinement = ind < nbTriangles && ttfRefineDecision(ind, subInd);
        const uint newBufferCount = ttfAddToRefinementBuffer(tid, needsRefinement, ind, subInd, 0u,
                                                             sCase.cfg2.y);
        if (tid == 0u)
            gTtfNumSubdivide = newBufferCount;   // cu:1199-1200

        GroupMemoryBarrierWithGroupSync();       // stands for cu:1202's scan and :1204

        numTrianglesToSubdividePerThread = gTtfNumSubdivide;   // cu:1209

        if (tid < TTF_SUBDIVIDE_COUNT)
        {
            Ptr<uint> simg = o + TTF_OUT_SUB + round * 512u;
            const uint2 s = gTtfSubdivide[tid];
            simg[2u * tid] = s.x;
            simg[2u * tid + 1u] = s.y;
        }
        if (tid == 0u)
        {
            o[round * 4u + 0u] = nbFound;
            o[round * 4u + 1u] = before;
            o[round * 4u + 2u] = startIndex;
            o[round * 4u + 3u] = numTrianglesToSubdividePerThread;
        }
        ++round;
        GroupMemoryBarrierWithGroupSync();   // the images are read before the next round writes
    }

    if (tid == 0u)
    {
        o[TTF_OUT_TAIL + 0u] = round;
        o[TTF_OUT_TAIL + 1u] = startIndex;
        o[TTF_OUT_TAIL + 2u] = numTrianglesToSubdividePerThread;
        o[TTF_OUT_TAIL + 3u] = 0x51DEu;
    }
}

// ---- the cull sphere, serial lanes -------------------------------------------
// case: six float4 transforms (scale0, rot0, scale1, rot1, aToBp, aToBq) @0..95,
//       then uint4 misc (x = ind, y = subInd) @96: 112 B
// out: float4 (centroidVtx, radius)
struct TtfCullCase
{
    float4 scale0, rot0, scale1, rot1, aToBp, aToBq;
    uint4 misc;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ttfCullProbe(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gF.numCases) return;
    const TtfCullCase cs = Ptr<TtfCullCase>(gF.cases)[i];
    float3 centroidVtx;
    const float r = ttfTriangleCullSphere(gF.meshInd, gF.meshVerts, cs.scale0.xyz, cs.rot0, cs.aToBp.xyz, cs.aToBq,
                                          cs.misc.x, cs.misc.y, cs.scale1.xyz, cs.rot1, centroidVtx);
    Ptr<uint4>(gF.out)[i] = uint4(asuint(centroidVtx), asuint(r));
}
