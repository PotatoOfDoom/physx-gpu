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

// skinning.slang
//
// Slang port of the deformable-skinning kernels from
// gpucommon/src/CUDA/utility.cu (backing PxDeformableSkinning):
//   - interleaveBuffers
//   - zeroNormals
//   - normalVectorsAreaWeighted
//   - normalizeNormals
//   - interpolateSkinnedClothVertices    (Phong interpolation)
//   - interpolateSkinnedSoftBodyVertices (barycentric interpolation)
//
// Differences from CUDA:
//   - PxTrimeshSkinningGpuData / PxTetmeshSkinningGpuData embed raw device
//     pointers (PxTypedBoundedData<PxVec3>::data etc.).  The port keeps the
//     stride/count semantics but replaces every pointer with a byte offset
//     into a single bound data buffer (gSkinData); descriptors live in their
//     own buffer, indexed by the Y workgroup coordinate exactly like the
//     CUDA blockIdx.y.
//   - CUDA float atomicAdd becomes a compare-exchange loop on the raw bits,
//     which every Vulkan/D3D12 device supports (VK_EXT_shader_atomic_float
//     is optional and llvmpipe lacks it).  Accumulation order was already
//     nondeterministic under CUDA; this changes nothing semantically.

static const uint BLOCK_SIZE = 256;

// PxTypedBoundedData<PxVec3> with the pointer replaced by a byte offset
struct PxgStridedVec3View
{
    uint offset; // byte offset into gSkinData
    uint stride; // byte stride between elements
    uint count;
    uint pad;
};

// mirrors PxTrimeshSkinningGpuData (utility.cu consumers), pointer-free
struct PxgTrimeshSkinningDesc
{
    PxgStridedVec3View guideVertices;   // 0
    PxgStridedVec3View guideNormals;    // 16
    uint guideTrianglesOffset;          // 32: byte offset, 3 uints per triangle
    uint nbGuideTriangles;              // 36
    uint skinningInfoOffset;            // 40: byte offset, PxTriangleMeshEmbeddingInfo (16B each)
    float halfSurfaceThickness;         // 44
    PxgStridedVec3View skinnedVertices; // 48
};                                      // = 64

// mirrors PxTetmeshSkinningGpuData, pointer-free
struct PxgTetmeshSkinningDesc
{
    PxgStridedVec3View guideVertices;   // 0
    uint guideTetrahedraOffset;         // 16: byte offset, 4 uints per tetrahedron
    uint skinningInfoOffset;            // 20: byte offset, PxTetrahedronMeshEmbeddingInfo (16B each)
    uint pad0;                          // 24
    uint pad1;                          // 28
    PxgStridedVec3View skinnedVertices; // 32
};                                      // = 48

struct SkinningParams
{
    uint length; // interleaveBuffers only
};

[[vk::push_constant]]
ConstantBuffer<SkinningParams> gParams;

RWStructuredBuffer<uint> gDescs;    // binding 0: descriptor array (raw words; tri or tet layout)
RWStructuredBuffer<uint> gSkinData; // binding 1: all mesh data (word-addressed)
RWStructuredBuffer<float4> gIn0;    // binding 2: interleaveBuffers vertices
RWStructuredBuffer<float4> gIn1;    // binding 3: interleaveBuffers normals
RWStructuredBuffer<float> gOut;     // binding 4: interleaveBuffers result (PxVec3 pairs)

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

float3 loadVec3(uint byteOffset)
{
    const uint w = byteOffset / 4;
    return float3(asfloat(gSkinData[w]), asfloat(gSkinData[w + 1]), asfloat(gSkinData[w + 2]));
}

void storeVec3(uint byteOffset, float3 v)
{
    const uint w = byteOffset / 4;
    gSkinData[w] = asuint(v.x);
    gSkinData[w + 1] = asuint(v.y);
    gSkinData[w + 2] = asuint(v.z);
}

float3 viewAt(PxgStridedVec3View view, uint i)
{
    return loadVec3(view.offset + i * view.stride);
}

void viewStore(PxgStridedVec3View view, uint i, float3 v)
{
    storeVec3(view.offset + i * view.stride, v);
}

uint loadIndex(uint byteOffset, uint i)
{
    return gSkinData[byteOffset / 4 + i];
}

// float atomic add via compare-exchange on the raw bits (word index)
void atomicAddFloat(uint wordIndex, float value)
{
    uint expected = gSkinData[wordIndex];
    for (;;)
    {
        const uint desired = asuint(asfloat(expected) + value);
        uint original;
        InterlockedCompareExchange(gSkinData[wordIndex], expected, desired, original);
        if (original == expected)
            break;
        expected = original;
    }
}

// mirrors AtomicAdd3() in atomic.cuh
void atomicAdd3(PxgStridedVec3View view, uint i, float3 v)
{
    const uint w = (view.offset + i * view.stride) / 4;
    atomicAddFloat(w + 0, v.x);
    atomicAddFloat(w + 1, v.y);
    atomicAddFloat(w + 2, v.z);
}

// mirrors PxVec3::normalizeSafe(): returns the magnitude, leaves the vector
// untouched when it is degenerate
float normalizeSafe(inout float3 v)
{
    // P9.47: normalizeSafe's magnitude is PxSqrt(magnitudeSquared()), not GLSL Length.
    const float mag = sqrt(dot(v, v));
    if (mag < 1e-20f) // PX_NORMALIZATION_EPSILON
        return 0.0f;
    v *= 1.0f / mag;
    return mag;
}

PxgTrimeshSkinningDesc loadTrimeshDesc(uint descIndex)
{
    const uint base = descIndex * 16; // 64 bytes = 16 words
    PxgTrimeshSkinningDesc d;
    d.guideVertices.offset = gDescs[base + 0];
    d.guideVertices.stride = gDescs[base + 1];
    d.guideVertices.count = gDescs[base + 2];
    d.guideNormals.offset = gDescs[base + 4];
    d.guideNormals.stride = gDescs[base + 5];
    d.guideNormals.count = gDescs[base + 6];
    d.guideTrianglesOffset = gDescs[base + 8];
    d.nbGuideTriangles = gDescs[base + 9];
    d.skinningInfoOffset = gDescs[base + 10];
    d.halfSurfaceThickness = asfloat(gDescs[base + 11]);
    d.skinnedVertices.offset = gDescs[base + 12];
    d.skinnedVertices.stride = gDescs[base + 13];
    d.skinnedVertices.count = gDescs[base + 14];
    return d;
}

PxgTetmeshSkinningDesc loadTetmeshDesc(uint descIndex)
{
    const uint base = descIndex * 12; // 48 bytes = 12 words
    PxgTetmeshSkinningDesc d;
    d.guideVertices.offset = gDescs[base + 0];
    d.guideVertices.stride = gDescs[base + 1];
    d.guideVertices.count = gDescs[base + 2];
    d.guideTetrahedraOffset = gDescs[base + 4];
    d.skinningInfoOffset = gDescs[base + 5];
    d.skinnedVertices.offset = gDescs[base + 8];
    d.skinnedVertices.stride = gDescs[base + 9];
    d.skinnedVertices.count = gDescs[base + 10];
    return d;
}

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

// mirrors interleaveBuffers() in utility.cu
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void interleaveBuffers(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint threadIndex = dispatchThreadId.x;
    if (threadIndex >= gParams.length)
        return;

    const float4 v = gIn0[threadIndex];
    const float4 n = gIn1[threadIndex];
    gOut[6 * threadIndex + 0] = v.x;
    gOut[6 * threadIndex + 1] = v.y;
    gOut[6 * threadIndex + 2] = v.z;
    gOut[6 * threadIndex + 3] = n.x;
    gOut[6 * threadIndex + 4] = n.y;
    gOut[6 * threadIndex + 5] = n.z;
}

// mirrors zeroNormals() in utility.cu
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void zeroNormals(uint3 dispatchThreadId : SV_DispatchThreadID, uint3 groupId : SV_GroupID)
{
    const PxgTrimeshSkinningDesc d = loadTrimeshDesc(groupId.y);

    const uint threadIndex = dispatchThreadId.x;
    if (threadIndex >= d.guideVertices.count)
        return;

    viewStore(d.guideNormals, threadIndex, float3(0.0f));
}

// mirrors normalVectorsAreaWeighted() in utility.cu
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void normalVectorsAreaWeighted(uint3 dispatchThreadId : SV_DispatchThreadID, uint3 groupId : SV_GroupID)
{
    const PxgTrimeshSkinningDesc d = loadTrimeshDesc(groupId.y);

    const uint threadIndex = dispatchThreadId.x;
    if (threadIndex >= d.nbGuideTriangles)
        return;

    const uint i0 = loadIndex(d.guideTrianglesOffset, 3 * threadIndex + 0);
    const uint i1 = loadIndex(d.guideTrianglesOffset, 3 * threadIndex + 1);
    const uint i2 = loadIndex(d.guideTrianglesOffset, 3 * threadIndex + 2);

    const float3 p0 = viewAt(d.guideVertices, i0);
    const float3 p1 = viewAt(d.guideVertices, i1);
    const float3 p2 = viewAt(d.guideVertices, i2);
    const float3 n = cross(p1 - p0, p2 - p0);

    atomicAdd3(d.guideNormals, i0, n);
    atomicAdd3(d.guideNormals, i1, n);
    atomicAdd3(d.guideNormals, i2, n);
}

// mirrors normalizeNormals() in utility.cu
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void normalizeNormals(uint3 dispatchThreadId : SV_DispatchThreadID, uint3 groupId : SV_GroupID)
{
    const PxgTrimeshSkinningDesc d = loadTrimeshDesc(groupId.y);

    const uint threadIndex = dispatchThreadId.x;
    if (threadIndex >= d.guideVertices.count)
        return;

    float3 n = viewAt(d.guideNormals, threadIndex);
    normalizeSafe(n);
    viewStore(d.guideNormals, threadIndex, n);
}

// mirrors evaluatePointPhongInterpolation<true>() in utility.cu
float3 evaluatePointPhongInterpolation(float3 a, float3 b, float3 c, float3 uvw,
                                       float3 nA, float3 nB, float3 nC,
                                       float halfSurfaceThickness, float alpha = 0.625f)
{
    const float3 q = uvw.x * a + uvw.y * b + uvw.z * c;

    const float scale1 = dot(q - a, nA);
    const float3 projA = q - scale1 * nA;

    const float scale2 = dot(q - b, nB);
    const float3 projB = q - scale2 * nB;

    const float scale3 = dot(q - c, nC);
    const float3 projC = q - scale3 * nC;

    const float3 qStar = uvw.x * projA + uvw.y * projB + uvw.z * projC;

    float3 dir = qStar - q;
    float offset = normalizeSafe(dir) * alpha;

    // Asymptotic function applied to offset such that the magnitude of offset
    // cannot exceed halfSurfaceThickness
    float ratio = 0.0f;
    if (halfSurfaceThickness > 0.0f)
    {
        ratio = offset / halfSurfaceThickness;
        ratio = tanh(ratio); // softMin(val, 1)
    }
    offset = ratio * halfSurfaceThickness;

    return q + offset * dir;
}

// mirrors interpolateSkinnedClothVertices() in utility.cu
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void interpolateSkinnedClothVertices(uint3 dispatchThreadId : SV_DispatchThreadID, uint3 groupId : SV_GroupID)
{
    const PxgTrimeshSkinningDesc d = loadTrimeshDesc(groupId.y);

    const uint threadIndex = dispatchThreadId.x;
    if (threadIndex >= d.skinnedVertices.count)
        return;

    // PxTriangleMeshEmbeddingInfo: { PxVec2 uv; PxReal offsetAlongInterpolatedNormal; PxU32 guideTriangleId; }
    const uint infoBase = d.skinningInfoOffset / 4 + threadIndex * 4;
    const float2 uv = float2(asfloat(gSkinData[infoBase]), asfloat(gSkinData[infoBase + 1]));
    const float offsetAlongInterpolatedNormal = asfloat(gSkinData[infoBase + 2]);
    const uint guideTriangleId = gSkinData[infoBase + 3];

    const uint i0 = loadIndex(d.guideTrianglesOffset, 3 * guideTriangleId + 0);
    const uint i1 = loadIndex(d.guideTrianglesOffset, 3 * guideTriangleId + 1);
    const uint i2 = loadIndex(d.guideTrianglesOffset, 3 * guideTriangleId + 2);

    const float w = 1.0f - uv.x - uv.y;

    const float3 uvw = float3(uv.x, uv.y, w);
    float3 uvwProj = max(uvw, float3(0.0f));
    const float sumProj = uvwProj.x + uvwProj.y + uvwProj.z;
    if (sumProj > 0.0f)
        uvwProj *= 1.0f / sumProj;

    const float3 nA = viewAt(d.guideNormals, i0);
    const float3 nB = viewAt(d.guideNormals, i1);
    const float3 nC = viewAt(d.guideNormals, i2);

    float3 normal = uvwProj.x * nA + uvwProj.y * nB + uvwProj.z * nC;
    normalizeSafe(normal);

    const float3 a = viewAt(d.guideVertices, i0);
    const float3 b = viewAt(d.guideVertices, i1);
    const float3 c = viewAt(d.guideVertices, i2);

    const float3 pointPhong = evaluatePointPhongInterpolation(a, b, c, uvwProj, nA, nB, nC,
                                                              d.halfSurfaceThickness);

    const float3 pointUVW = uvw.x * a + uvw.y * b + uvw.z * c;
    const float3 pointUVWProj = uvwProj.x * a + uvwProj.y * b + uvwProj.z * c;

    const float3 offsetPoint = pointPhong + offsetAlongInterpolatedNormal * normal + pointUVW - pointUVWProj;

    viewStore(d.skinnedVertices, threadIndex, offsetPoint);
}

// mirrors interpolateSkinnedSoftBodyVertices() in utility.cu
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void interpolateSkinnedSoftBodyVertices(uint3 dispatchThreadId : SV_DispatchThreadID, uint3 groupId : SV_GroupID)
{
    const PxgTetmeshSkinningDesc d = loadTetmeshDesc(groupId.y);

    const uint threadIndex = dispatchThreadId.x;
    if (threadIndex >= d.skinnedVertices.count)
        return;

    // Uses linear barycentric interpolation - plenty of room for improvements
    // PxTetrahedronMeshEmbeddingInfo: { PxVec3 uvw; PxU32 guideTetrahedronId; }
    const uint infoBase = d.skinningInfoOffset / 4 + threadIndex * 4;
    const float3 uvw = float3(asfloat(gSkinData[infoBase]), asfloat(gSkinData[infoBase + 1]),
                              asfloat(gSkinData[infoBase + 2]));
    const uint guideTetrahedronId = gSkinData[infoBase + 3];

    const uint t0 = loadIndex(d.guideTetrahedraOffset, 4 * guideTetrahedronId + 0);
    const uint t1 = loadIndex(d.guideTetrahedraOffset, 4 * guideTetrahedronId + 1);
    const uint t2 = loadIndex(d.guideTetrahedraOffset, 4 * guideTetrahedronId + 2);
    const uint t3 = loadIndex(d.guideTetrahedraOffset, 4 * guideTetrahedronId + 3);

    const float s = 1.0f - uvw.x - uvw.y - uvw.z;

    const float3 point = uvw.x * viewAt(d.guideVertices, t0) + uvw.y * viewAt(d.guideVertices, t1) +
                         uvw.z * viewAt(d.guideVertices, t2) + s * viewAt(d.guideVertices, t3);

    viewStore(d.skinnedVertices, threadIndex, point);
}
