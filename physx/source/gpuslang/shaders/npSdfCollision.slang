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

// npSdfCollision.slang -- P9.12d, one entry point per function of
// pxgSlangSdfCollision, each writing back everything it used.
//
// ONE push-constant block for ALL FOUR entry points, and that is not a tidiness
// choice.  Two [[vk::push_constant]] blocks in one module are laid out in one
// shared range, so the second one sits at the first one's SIZE -- and a host that
// pushes each struct at offset 0 hands the later entry points their parameters
// shifted.  The symptom is not a validation error (each entry point references
// only its own block, so the SPIR-V is legal); it is a garbage device address and
// a segfault with no output at all.  Measured in P9.12c on npSdfTexture.
//
// WHY EACH ENTRY WRITES SO MUCH.  Sample() cannot be evaluated on the host, so
// the fixture never predicts a sample value; it checks each function's
// COMPOSITION against the values that function reports having read.  The traces
// are therefore the test, not diagnostics -- see the oracle note at the head of
// pxgSlangSdfCollision.slang.

import pxgSlangSdfCollision;
import pxgSlangSdfTexture;
import pxgSlangTetClosest;   // closestPtPointBaryTriangle (P9.34a)

struct SdfCollParams
{
    uint64_t meshPtr;      // the cooked-mesh blob
    uint64_t points;       // float4 per point: xyz = localPos, w = tolerance
    uint64_t gradIn;       // float4 per point: xyz = the gradient handed in
    uint64_t dirSeed;      // float4 per point: xyz = the incoming value of `dir`
    uint64_t outBuf;       // per-entry stride, documented at each entry
    uint numPoints;
    uint pad0;
    // P9.34a: the triangle descent needs three vertices per entry rather than one
    // point, so it reads them from here instead of `points`.
    uint64_t triVerts;     // 3 float4 per entry: v0, v1, v2
    uint64_t triOut;       // the trace, stride documented at the entry
}

[[vk::push_constant]]
ConstantBuffer<SdfCollParams> gP;

// Every entry repeats the blob walk and initialize() per thread, exactly as the
// CUDA repeats them per block.  Pure address arithmetic over a read-only blob.
bool loadTexture(out SparseSdfTexture t)
{
    PxgSdfMesh m;
    if (!sdfReadMesh(gP.meshPtr, m))
    {
        t = sdfTexInitialize(m);
        return false;
    }
    t = sdfTexInitialize(m);
    return true;
}

// ---- PxSdfDistance.  Stride 12 words. -------------------------------------
//   0..2 clamped   3..5 f   6 diffMag   7 Sample(f)   8 return
[shader("compute")]
[numthreads(64, 1, 1)]
void sdfCollDistance(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gP.numPoints)
        return;

    SparseSdfTexture t;
    if (!loadTexture(t))
        return;

    const float3 p = Ptr<float4>(gP.points)[i].xyz;

    float3 clamped, f;
    float diffMag, sampleValue;
    const float d = sdfDistanceEx(t, p, clamped, f, diffMag, sampleValue);

    Ptr<float> o = Ptr<float>(gP.outBuf) + i * 12;
    o[0] = clamped.x; o[1] = clamped.y; o[2] = clamped.z;
    o[3] = f.x;       o[4] = f.y;       o[5] = f.z;
    o[6] = diffMag;
    o[7] = sampleValue;
    o[8] = d;
}

// ---- PxVolumeGrad.  Stride 32 words. --------------------------------------
//   0 interior   1..3 grad   4..6 f   7..24 coord[0..5]   25..30 value[0..5]
[shader("compute")]
[numthreads(64, 1, 1)]
void sdfCollGrad(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gP.numPoints)
        return;

    SparseSdfTexture t;
    if (!loadTexture(t))
        return;

    const float3 p = Ptr<float4>(gP.points)[i].xyz;

    SdfGradTrace tr;
    const float3 g = sdfVolumeGradEx(t, p, tr);

    Ptr<float> o = Ptr<float>(gP.outBuf) + i * 32;
    o[0] = asfloat(tr.interior);
    o[1] = g.x; o[2] = g.y; o[3] = g.z;
    o[4] = tr.f.x; o[5] = tr.f.y; o[6] = tr.f.z;
    [ForceUnroll]
    for (uint k = 0; k < 6; ++k)
    {
        o[7 + k * 3 + 0] = tr.coord[k].x;
        o[7 + k * 3 + 1] = tr.coord[k].y;
        o[7 + k * 3 + 2] = tr.coord[k].z;
        o[25 + k] = tr.value[k];
    }
}

// ---- PxSdfSampleWithGrad.  Stride 12 words. -------------------------------
//   0 dist   1..3 dir   4 return   5..7 diff
[shader("compute")]
[numthreads(64, 1, 1)]
void sdfCollSampleGrad(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gP.numPoints)
        return;

    SparseSdfTexture t;
    if (!loadTexture(t))
        return;

    const float4 pt = Ptr<float4>(gP.points)[i];
    const float3 g = Ptr<float4>(gP.gradIn)[i].xyz;
    // C9: seeded from the host so the reject path's "dir is left alone" is an
    // observable rather than a claim about undefined memory.
    float3 dir = Ptr<float4>(gP.dirSeed)[i].xyz;

    float dist;
    float3 diff;
    const float r = sdfSampleWithGradEx(t, pt.xyz, g, dir, pt.w, dist, diff);

    Ptr<float> o = Ptr<float>(gP.outBuf) + i * 12;
    o[0] = dist;
    o[1] = dir.x; o[2] = dir.y; o[3] = dir.z;
    o[4] = r;
    o[5] = diff.x; o[6] = diff.y; o[7] = diff.z;
}

// ---- doVertexSDFCollision.  Stride 16 words. ------------------------------
//   0..2 rawGrad   3 gradMag   4..6 usedGrad   7 dist   8..10 dir   11 return
//   12 interior
[shader("compute")]
[numthreads(64, 1, 1)]
void sdfCollVertex(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gP.numPoints)
        return;

    SparseSdfTexture t;
    if (!loadTexture(t))
        return;

    const float4 pt = Ptr<float4>(gP.points)[i];
    float3 dir = Ptr<float4>(gP.dirSeed)[i].xyz;

    SdfGradTrace tr;
    float3 rawGrad, usedGrad, diff;
    float gradMag, dist;
    const float r = sdfDoVertexCollisionEx(t, pt.xyz, dir, pt.w, tr, rawGrad, gradMag,
                                           usedGrad, dist, diff);

    Ptr<float> o = Ptr<float>(gP.outBuf) + i * 16;
    o[0] = rawGrad.x; o[1] = rawGrad.y; o[2] = rawGrad.z;
    o[3] = gradMag;
    o[4] = usedGrad.x; o[5] = usedGrad.y; o[6] = usedGrad.z;
    o[7] = dist;
    o[8] = dir.x; o[9] = dir.y; o[10] = dir.z;
    o[11] = r;
    o[12] = asfloat(tr.interior);
}


// ---- Gu::closestPtPointBaryTriangle.  Stride 4 words (P9.34a). -------------
//   0..2 the returned barycentric   3 unused
//
// The one function on this page with a REAL oracle: it is PX_CUDA_CALLABLE and
// texture-free, so the fixture calls the engine's own copy on the host and
// compares bit for bit rather than against a transcription.
[shader("compute")]
[numthreads(64, 1, 1)]
void sdfCollBaryClamp(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gP.numPoints)
        return;

    const float3 c = Ptr<float4>(gP.points)[i].xyz;
    const float3 r = closestPtPointBaryTriangle(c);

    Ptr<float> o = Ptr<float>(gP.outBuf) + i * 4;
    o[0] = r.x; o[1] = r.y; o[2] = r.z; o[3] = 0.0f;
}

// ---- doTriangleSDFCollision.  Stride 176 words (P9.34a). -------------------
//    0..3   d0 d1 d2 dCenter
//    4      startArm            5      difference
//    6..8   uvw0                9..11  p0
//   12      iters              13      converged
//   14..16  pFinal             17..19  gradFinal
//   20..22  point (out)        23..25  dir (inout, on the way out)
//   26      the returned distance
//   27      unused
//   28 + 9k .. : per-iteration, nine words each for k = 0..15:
//              +0 stepIn   +1 gradMag   +2..4 newUvw   +5..7 gradUsed   +8 unused
//
// The USED gradient is in the trace and the RAW one is not, and that split is the
// whole design.  The fixture REPLAYS each iteration on the host from (uvw, stepIn,
// gradUsed, v0, v1, v2): it forms dfdu/dfdv/dfdw, applies the step, calls THE
// ENGINE'S OWN closestPtPointBaryTriangle, and must land on `newUvw` bit for bit.
// That checks the composition -- the cascade, the step schedule, the clamp, the
// convergence test and where the loop stops.  It does not check the gradient
// values, and must not: those are GPU-only samples and are P9.12c/d's claim,
// measured there.  `gradMag` is carried too, so the replay can also verify that
// gradUsed is the raw one normalised or the discontinuity substitute.
[shader("compute")]
[numthreads(64, 1, 1)]
void sdfCollTriangle(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gP.numPoints)
        return;

    SparseSdfTexture t;
    if (!loadTexture(t))
        return;

    Ptr<float4> tv = Ptr<float4>(gP.triVerts) + i * 3;
    const float3 v0 = tv[0].xyz, v1 = tv[1].xyz, v2 = tv[2].xyz;
    const float tolerance = tv[0].w;

    float3 point;
    float3 dir = Ptr<float4>(gP.dirSeed)[i].xyz;
    SdfTriTrace tr;
    const float d = sdfDoTriangleCollisionEx(t, v0, v1, v2, point, dir, tolerance, tr);

    Ptr<float> o = Ptr<float>(gP.triOut) + i * 176;
    o[0] = tr.d0; o[1] = tr.d1; o[2] = tr.d2; o[3] = tr.dCenter;
    o[4] = float(tr.startArm);
    o[5] = tr.difference;
    o[6] = tr.uvw0.x; o[7] = tr.uvw0.y; o[8] = tr.uvw0.z;
    o[9] = tr.p0.x;  o[10] = tr.p0.y;  o[11] = tr.p0.z;
    o[12] = float(tr.iters);
    o[13] = float(tr.converged);
    o[14] = tr.pFinal.x;   o[15] = tr.pFinal.y;   o[16] = tr.pFinal.z;
    o[17] = tr.gradFinal.x; o[18] = tr.gradFinal.y; o[19] = tr.gradFinal.z;
    o[20] = point.x; o[21] = point.y; o[22] = point.z;
    o[23] = dir.x;   o[24] = dir.y;   o[25] = dir.z;
    o[26] = d;
    o[27] = 0.0f;
    for (uint k = 0u; k < SDF_TRI_MAX_ITERS; ++k)
    {
        Ptr<float> e = o + 28 + k * 9;
        e[0] = tr.it[k].stepIn;
        e[1] = tr.it[k].gradMag;
        e[2] = tr.it[k].newUvw.x;   e[3] = tr.it[k].newUvw.y;   e[4] = tr.it[k].newUvw.z;
        e[5] = tr.it[k].gradUsed.x; e[6] = tr.it[k].gradUsed.y; e[7] = tr.it[k].gradUsed.z;
        e[8] = 0.0f;
    }
}

// ---- Gu::closestPtPointTetrahedron, the five-argument overload.  P9.35a. ----
// Stride 4 words: the returned point, then a pad.  `points` carries p, and the
// four tetrahedron corners come from `triVerts`, four float4 per entry -- so this
// entry reuses that buffer with a stride of FOUR rather than the triangle
// descent's three.
//
// This is an ORACLE test: the function is PX_CUDA_CALLABLE and texture-free, so
// the fixture calls the ENGINE's own copy on the host and compares bit for bit.
// There is no hand-written mirror that could be wrong in the same way.
[shader("compute")]
[numthreads(64, 1, 1)]
void sdfCollTetClamp(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gP.numPoints)
        return;

    const float3 p = Ptr<float4>(gP.points)[i].xyz;
    Ptr<float4> tv = Ptr<float4>(gP.triVerts) + i * 4;
    const float3 r = guClosestPtPointTetrahedron(p, tv[0].xyz, tv[1].xyz, tv[2].xyz, tv[3].xyz);

    Ptr<float> o = Ptr<float>(gP.outBuf) + i * 4;
    o[0] = r.x; o[1] = r.y; o[2] = r.z; o[3] = 0.0f;
}

// ---- doTetrahedronSDFCollision.  Stride 208 words (P9.35a). ----------------
//    0..4   d0 d1 d2 d3 dCenter
//    5      startArm            6      difference
//    7..10  uvw0               11..13  p0
//   14      iters              15      converged
//   16..18  pFinal             19..21  gradFinal
//   22..24  point (out)        25..27  dir (inout, on the way out)
//   28      the returned distance
//   29..31  unused
//   32 + 11k .. : per-iteration, eleven words each for k = 0..15:
//              +0 stepIn  +1 gradMag  +2..5 newUvw  +6..8 gradUsed
//              +9 clamped  +10 unused
//
// Same arrangement as the triangle trace, plus ONE extra field that the triangle
// version has no use for: `clamped`.  E3 makes the projection conditional here, so
// whether it ran on a given iteration is part of what the replay has to reproduce
// -- and a fixture that only replayed the arithmetic would be blind to a port that
// clamps unconditionally, because on an iteration that stays inside the simplex
// the projection is the identity.
[shader("compute")]
[numthreads(64, 1, 1)]
void sdfCollTetrahedron(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gP.numPoints)
        return;

    SparseSdfTexture t;
    if (!loadTexture(t))
        return;

    Ptr<float4> tv = Ptr<float4>(gP.triVerts) + i * 4;
    const float3 v0 = tv[0].xyz, v1 = tv[1].xyz, v2 = tv[2].xyz, v3 = tv[3].xyz;
    const float tolerance = tv[0].w;

    float3 point;
    float3 dir = Ptr<float4>(gP.dirSeed)[i].xyz;
    SdfTetTrace tr;
    const float d = sdfDoTetrahedronCollisionEx(t, v0, v1, v2, v3, point, dir, tolerance, tr);

    Ptr<float> o = Ptr<float>(gP.triOut) + i * 208;
    o[0] = tr.d0; o[1] = tr.d1; o[2] = tr.d2; o[3] = tr.d3; o[4] = tr.dCenter;
    o[5] = float(tr.startArm);
    o[6] = tr.difference;
    o[7] = tr.uvw0.x; o[8] = tr.uvw0.y; o[9] = tr.uvw0.z; o[10] = tr.uvw0.w;
    o[11] = tr.p0.x; o[12] = tr.p0.y; o[13] = tr.p0.z;
    o[14] = float(tr.iters);
    o[15] = float(tr.converged);
    o[16] = tr.pFinal.x;    o[17] = tr.pFinal.y;    o[18] = tr.pFinal.z;
    o[19] = tr.gradFinal.x; o[20] = tr.gradFinal.y; o[21] = tr.gradFinal.z;
    o[22] = point.x; o[23] = point.y; o[24] = point.z;
    o[25] = dir.x;   o[26] = dir.y;   o[27] = dir.z;
    o[28] = d;
    o[29] = 0.0f; o[30] = 0.0f; o[31] = 0.0f;
    for (uint k = 0u; k < SDF_TET_MAX_ITERS; ++k)
    {
        Ptr<float> e = o + 32 + k * 11;
        e[0] = tr.it[k].stepIn;
        e[1] = tr.it[k].gradMag;
        e[2] = tr.it[k].newUvw.x; e[3] = tr.it[k].newUvw.y;
        e[4] = tr.it[k].newUvw.z; e[5] = tr.it[k].newUvw.w;
        e[6] = tr.it[k].gradUsed.x; e[7] = tr.it[k].gradUsed.y; e[8] = tr.it[k].gradUsed.z;
        e[9] = float(tr.it[k].clamped);
        e[10] = 0.0f;
    }
}
