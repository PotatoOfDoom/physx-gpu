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
// CORRECTION (P9.83).  This file used to close with the note "the sibling
// kernel triangleTriangleCollision is NOT ported: its core calls
// doTriangleSDFCollision over tex3D SDF textures -- hardware filtering, not
// bit-exact-portable, same category as sdfCollision.cuh."  That is false, and
// it has been false since P9.12d/P9.34c.  The SDF texture path IS ported:
// pxgSlangSdfTexture.slang reimplements the tex3D fetch as an explicit
// trilinear blend over the raw sparse/dense grid (no hardware filtering is
// involved), and pxgSlangSdfCollision.slang carries doTriangleSDFCollision
// itself.  Two MAPPED engine kernels already import both --
// sb_sdfMeshMidphaseGeneratePairsLaunch (manifest -> softBodySdfMidphaseBda.spv)
// and cloth_sdfMeshContactGenLaunch (manifest -> femClothSdfMeshCgBda.spv).
// The note is kept, struck through, rather than deleted: a stale "cannot be
// done" is the one kind of comment that stops a tranche before it starts, and
// the record of having believed it is worth more than a clean file.
//
// ---------------------------------------------------------------------------
// P9.83: THE BODY MOVED OUT, AND WHY
// ---------------------------------------------------------------------------
// This file used to carry its own copy of the prep AND its own five-hundred-line
// copy of the cold-start GJK driver.  Both now live in modules:
//
//   pxgSlangTriTriOverlap   the decision -- root-bounds reduce, the corner bit
//                           pattern, vertex2Shape, aToB, the initial direction
//                           and the DISTANT test
//   pxgSlangGjk             squawk::gjk itself, which every deformable-vs-convex
//                           contact gen in this tree already ran
//
// The engine-ABI kernel (triTriOverlapBda.slang) calls the SAME two, so this
// fixture now certifies what the engine runs rather than a sibling of it.  What
// is left here is only where the bytes are: a host-filled TtoIn instead of a
// cooked blob, and the scratch buffers the test reads the prep points back from.

import pxgSlangGjk;
import pxgSlangTriTriOverlap;

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

[[vk::push_constant]]
ConstantBuffer<GjkParams> gParams;

RWStructuredBuffer<float4> gVA;   // binding 0: the A cloud the prep produced (8/test)
RWStructuredBuffer<float4> gVB;   // binding 1: the B cloud (8/test)
RWStructuredBuffer<TtoIn> gIn;    // binding 2
RWStructuredBuffer<GjkOut> gOut;  // binding 3

[shader("compute")]
[numthreads(64, 1, 1)]
void triTriOverlap(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const uint vbase = t * gParams.maxV;

    const TtoIn e = gIn[t];

    const TtoBox b0 = ttoRootBox(e.nMin0, e.nMax0, e.counts.x);
    const TtoBox b1 = ttoRootBox(e.nMin1, e.nMax1, e.counts.y);

    GjkOut o;
    float3 cloudA[8], cloudB[8];
    const uint ret = ttoOverlap(b0, b1,
                                e.t0q, e.t0p.xyz, e.t1q, e.t1p.xyz,
                                e.scaleS0.xyz, e.scaleR0,
                                e.scaleS1.xyz, e.scaleR1,
                                e.misc.x,               // contactDistance == minSep
                                o, cloudA, cloudB);

    // The test reads the prep points back slot by slot; the w lanes are ignored
    // there (`if ((j & 3) == 3) continue`) and written as zero for determinism.
    for (uint i = 0u; i < 8u; ++i)
    {
        gVA[vbase + i] = float4(cloudA[i], 0.0f);
        gVB[vbase + i] = float4(cloudB[i], 0.0f);
    }
    gOut[t] = o;
    // ttoOverlapFlag(ret) is what the ENGINE kernel stores; here the test derives
    // it from o.ret itself, so the call is not made and `ret` is only the echo.
    gOut[t].ret = ret;
}
