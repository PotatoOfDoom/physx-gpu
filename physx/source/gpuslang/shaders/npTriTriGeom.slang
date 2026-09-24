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

// npTriTriGeom.slang -- P9.83b-8 (1/n).
//
//   Fixture entries for pxgSlangTriTriGeom: one thread per case, no barriers, no
//   shared memory.  Every helper in that module is pure arithmetic over its
//   arguments (plus two BDA loads), so a serial lane is the whole story -- the
//   difficulty is the ORDER of operations, and that is what the host oracle in
//   testNpTriTriGeom.cpp pins, bit for bit, against the engine's own headers.
//
//   ttgSubdivProbe   getSubTriangleEncoded, decode, encode(decode(x)), and the
//                    four elevations of one id
//   ttgVertsProbe    getTriangleVertices (with and without a sub-index) over a
//                    BDA mesh, with a non-uniform rotated scale and a rigid
//                    transform
//   ttgMathProbe     triangleRadiusSquared, sdfRadiusSquared, PxMax on its own,
//                    transformInv and getInverse
//
// Stamped NoContraction (PXG_NOCONTRACT_SHADERS): the host oracle is compiled
// with -ffp-contract=off, and quatRotate's mul+add chains would otherwise be
// free to fuse on one side only.  The engine kernel that will call this module
// is stamped too (P9.83b-8 wires it into the same list), so the fixture does
// not certify a build of the module the engine never runs.

import pxgSlangTriTriGeom;

struct TtgParams
{
    uint numCases;
    uint pad0, pad1, pad2;
    uint64_t cases;       // per-entry case records
    uint64_t outBuf;      // per-entry output records
    uint64_t meshInd;     // uint4 per triangle   (ttgVertsProbe only)
    uint64_t meshVerts;   // float4 per vertex    (ttgVertsProbe only)
}

[[vk::push_constant]] ConstantBuffer<TtgParams> gP;

// ---- subdivision -----------------------------------------------------------
// case: float4 a, b, c (xyz); uint4 (encodedSubIndex, idId, 0, 0)    = 64 B
// out:  float4 a', b', c' (w = 0)   from getSubTriangleEncoded(encodedSubIndex)
//       uint4 (level, index, encode(level, index), 0)   from decode(idId)
//       uint4 elevate(idId, 0..3)                                       = 80 B
// Two ids because the domains differ: getSubTriangle is undefined past level 16
// (a shift of 32), so encodedSubIndex stays within the levels whose index fits;
// decode/encode/elevate are plain integer arithmetic, defined for EVERY uint,
// and idId covers all 64 values of the six-bit level field.
struct TtgSubdivCase
{
    float4 a, b, c;
    uint4 misc;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ttgSubdivProbe(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gP.numCases) return;

    const TtgSubdivCase cs = Ptr<TtgSubdivCase>(gP.cases)[i];
    float3 a = cs.a.xyz, b = cs.b.xyz, c = cs.c.xyz;
    const uint enc = cs.misc.x;
    ttgGetSubTriangleEncoded(a, b, c, enc);

    const uint idId = cs.misc.y;
    uint level, index;
    ttgDecodeSubdivisionId(idId, level, index);

    Ptr<uint4> o = Ptr<uint4>(gP.outBuf) + i * 5u;
    o[0] = uint4(asuint(a), 0u);
    o[1] = uint4(asuint(b), 0u);
    o[2] = uint4(asuint(c), 0u);
    o[3] = uint4(level, index, ttgEncodeSubdivisionId(level, index), 0u);
    o[4] = uint4(ttgElevateSubdivisionId(idId, 0u), ttgElevateSubdivisionId(idId, 1u),
                 ttgElevateSubdivisionId(idId, 2u), ttgElevateSubdivisionId(idId, 3u));
}

// ---- mesh vertices ---------------------------------------------------------
// case: float4 scale (xyz), scaleRot, tp (xyz), tq;
//       uint4 (triangleIndex, encodedSubIndex, 0, 0)                     = 80 B
// out:  float4 a, b, c (w = 0)                                           = 48 B
struct TtgVertsCase
{
    float4 scale, scaleRot, tp, tq;
    uint4 misc;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ttgVertsProbe(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gP.numCases) return;

    const TtgVertsCase cs = Ptr<TtgVertsCase>(gP.cases)[i];
    float3 a, b, c;
    ttgGetTriangleVertices(gP.meshInd, gP.meshVerts, cs.scale.xyz, cs.scaleRot, cs.tp.xyz, cs.tq,
                           cs.misc.x, cs.misc.y, a, b, c);
    Ptr<uint4> o = Ptr<uint4>(gP.outBuf) + i * 3u;
    o[0] = uint4(asuint(a), 0u);
    o[1] = uint4(asuint(b), 0u);
    o[2] = uint4(asuint(c), 0u);
}

// ---- radii, PxMax, the two inverses -----------------------------------------
// case: float4 v0, v1, v2 (xyz; v0.w, v1.w = the PxMax operands);
//       float4 boxLo, boxHi (xyz); float4 tp (xyz), tq; float4 v (xyz)   = 128 B
// out:  float4 (triRadSq, sdfRadSq, PxMax(v0.w, v1.w), 0);
//       float4 transformInv(v); float4 invP; float4 invQ                 = 64 B
struct TtgMathCase
{
    float4 v0, v1, v2;
    float4 boxLo, boxHi;
    float4 tp, tq;
    float4 v;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ttgMathProbe(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gP.numCases) return;

    const TtgMathCase cs = Ptr<TtgMathCase>(gP.cases)[i];
    float3 invP;
    float4 invQ;
    ttgGetInverse(cs.tp.xyz, cs.tq, invP, invQ);

    Ptr<uint4> o = Ptr<uint4>(gP.outBuf) + i * 4u;
    o[0] = uint4(asuint(ttgTriangleRadiusSquared(cs.v0.xyz, cs.v1.xyz, cs.v2.xyz)),
                 asuint(ttgSdfRadiusSquared(cs.boxLo.xyz, cs.boxHi.xyz)),
                 asuint(ttgPxMax(cs.v0.w, cs.v1.w)), 0u);
    o[1] = uint4(asuint(ttgTransformInv(cs.tp.xyz, cs.tq, cs.v.xyz)), 0u);
    o[2] = uint4(asuint(invP), 0u);
    o[3] = asuint(invQ);
}
