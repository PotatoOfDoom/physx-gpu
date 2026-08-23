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

struct SdfCollParams
{
    uint64_t meshPtr;      // the cooked-mesh blob
    uint64_t points;       // float4 per point: xyz = localPos, w = tolerance
    uint64_t gradIn;       // float4 per point: xyz = the gradient handed in
    uint64_t dirSeed;      // float4 per point: xyz = the incoming value of `dir`
    uint64_t outBuf;       // per-entry stride, documented at each entry
    uint numPoints;
    uint pad0;
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
