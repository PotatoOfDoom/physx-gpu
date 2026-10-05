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

// sdfQueryBda.slang -- P9.95, evaluatePointDistancesSDFBatch on the engine ABI.
//
// convexMeshMidphase.cu:1233-1287, the kernel behind PxDirectGPUAPI::
// evaluateSDFDistances (NpDirectGPUAPI.cpp:151-163 -> PxgSimulationController.cpp:
// 631-636 -> PxgGpuNarrowphaseCore::evaluateSDFDistances, PxgNarrowphaseCore.cpp:
// 8899-8942).  For every sample point of every queried shape it writes the
// normalised SDF gradient and the signed distance, in mesh-local space.
//
// Composition, not new numerics: the mesh read and the texture set-up are P9.12c's
// (pxgSlangSdfTexture), PxVolumeGrad and PxSdfDistance are P9.12d's
// (pxgSlangSdfCollision), both swept modules.  The only new arithmetic is
// PxVec3::normalizeSafe, transcribed below.
//
// LAUNCH (PxgNarrowphaseCore.cpp:8922-8924): grid (ceil(maxPointCount / 1024),
// nbElements, 1), block (1024, 1, 1), no dynamic shared memory.  blockIdx.y is the
// queried shape, blockIdx.x * 1024 + threadIdx.x the point.
//
// ---------------------------------------------------------------------------
// THE TRAPS
// ---------------------------------------------------------------------------
//  Q1. numthreads is the host's block, 1024.  The bridge drops blockDim, so the
//      shader's group size IS the block size.
//  Q2. The shape is blockIdx.Y, not X.
//  Q3. Both slices start at batchId * maxPointCount -- the SLICE STRIDE, not the
//      shape's own point count -- and the product is a PxU32 (it wraps in 32 bits
//      before it is widened), added to the base pointer as its own step:
//      `&localPositionsConcatenated[batchId * maxPointCount]` then `[i]`.  Here the
//      slice base is a 64-bit byte offset from the 32-bit product, and the point
//      index is added by the pointer index.
//  Q4. blockDim is the literal 1024 (Q1), not a push-constant field.
//  Q5. The guard is `i >= pointCountPerShape[batchId]` -- the per-shape count, not
//      maxPointCount.  Slots between the two are never written.
//  Q6. Only .xyz of the sample point is read (`getXYZ()`); .w is unused by the API.
//  Q7. The store is PxVec4(gradient, signedDistance): the NORMALISED gradient in
//      .xyz and the distance in .w.
//  Q8. No PxMeshScale and no pose: "local space" is the mesh's raw vertex space
//      (PxDirectGPUAPI.h:404), and the kernel reads neither.
//  Q9. The CUDA stages the shape, the mesh and the texture through shared memory
//      (warp 0, thread 0, one __syncthreads).  Here every thread re-reads them from
//      the immutable blob, as psSdfMidphaseBda does: same values, no barrier.
// Q10. normalizeSafe (PxVec3.h:307-314) leaves the vector UNCHANGED below
//      PX_NORMALIZATION_EPSILON -- it neither zeroes it (that is getNormalized) nor
//      divides -- and otherwise MULTIPLIES by the reciprocal `1/mag`.  The compare
//      is `<`.  pxgSlangClothTriContact's copy is the `>=` complement written for a
//      different caller and is deliberately not imported.
// Q11. The sample-point load sits below a per-invocation guard, and llvmpipe does
//      not predicate a load in a masked lane (P9.22, check_speculated_loads.py).  In
//      the last batch the lanes past the count would read up to 1023 float4 past the
//      user's allocation.  The load index is clamped to the last live point; for
//      every lane that reaches the store (i < count) it is i itself.
//
// DIVERGENCES, recorded:
//  D1. A shape whose mesh carries no SDF (sdfDims.x*y*z == 0) returns WITHOUT
//      writing.  The CUDA samples texture handle 0 with an uninitialised shared
//      texture and writes whatever comes back (the precondition "the shapes must be
//      triangle mesh shapes with SDFs", PxDirectGPUAPI.h:406, is not checked
//      anywhere -- upstream candidate).  psSdfMidphaseBda made the same choice.
//  D2. The shape and mesh reads sit below the point guard (Q9 makes them per
//      thread).  For valid input the values are identical.

import pxgSlangNpAbi;        // PxgShapeGpu (48 B; hullOrMeshPtr @32)
import pxgSlangSdfTexture;   // sdfReadMesh, sdfTexInitialize
import pxgSlangSdfCollision; // sdfDistance, sdfVolumeGrad

static const float PX_NORMALIZATION_EPSILON = 1e-20f;   // PxSimpleTypes.h:103

// The six arguments in PX_CUDA_KERNEL_PARAM order (PxgNarrowphaseCore.cpp:8912-8920).
// The PxU32 between two pointers pads the last pointer to 40 (kSdfPointDist).
struct SdfQueryPush
{
    uint64_t gpuShapes;          // @0   const PxgShape*
    uint64_t shapeRefsSDF;       // @8   const PxShapeGPUIndex*
    uint64_t localPositions;     // @16  const PxVec4*, nbElements * maxPointCount
    uint64_t pointCountPerShape; // @24  const PxU32*
    uint     maxPointCount;      // @32
    uint     pad0;               // @36  (std430 aligns the next field to 40 anyway)
    uint64_t localGradAndSdf;    // @40  PxVec4*, nbElements * maxPointCount
};                               // 48 bytes

[[vk::push_constant]]
ConstantBuffer<SdfQueryPush> gSq;

// PxVec3::normalizeSafe (PxVec3.h:307-314), literal: magnitude() is
// sqrtf(x*x + y*y + z*z) (PxVec3.h:179), then `*this *= 1.0f / mag`.
float3 sqNormalizeSafe(float3 v)
{
    const float mag = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (mag < PX_NORMALIZATION_EPSILON)
        return v;
    const float inv = 1.0f / mag;
    return float3(v.x * inv, v.y * inv, v.z * inv);
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void evaluatePointDistancesSDFBatch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint batchId = gid.y;                                                   // .cu:1243
    const uint shapeRef0 = Ptr<uint>(gSq.shapeRefsSDF)[batchId];                  // .cu:1252
    const uint numSamplePositions = Ptr<uint>(gSq.pointCountPerShape)[batchId];   // .cu:1254
    // .cu:1253/:1255: the PxU32 product, widened to a byte offset afterwards (Q3).
    const uint64_t sliceBytes = uint64_t(batchId * gSq.maxPointCount) * uint64_t(16);

    const uint i = gtid.x + gid.x * 1024u;                                        // .cu:1277
    if (i >= numSamplePositions)                                                  // .cu:1279
        return;

    const PxgShapeGpu shape0 = Ptr<PxgShapeGpu>(gSq.gpuShapes)[shapeRef0];        // .cu:1260
    PxgSdfMesh mesh0;
    if (!sdfReadMesh(shape0.hullOrMeshPtr, mesh0))                                // D1
        return;
    const SparseSdfTexture tex = sdfTexInitialize(mesh0);                         // .cu:1271

    // Q11: value-identical for every lane that reaches the store (i < count).
    const uint li = min(i, max(numSamplePositions, 1u) - 1u);
    const float3 localPos = Ptr<float4>(gSq.localPositions + sliceBytes)[li].xyz; // .cu:1282, Q6
    float3 gradient = sdfVolumeGrad(tex, localPos);                               // .cu:1283
    gradient = sqNormalizeSafe(gradient);                                         // .cu:1284
    const float signedDistance = sdfDistance(tex, localPos);                      // .cu:1285
    Ptr<float4>(gSq.localGradAndSdf + sliceBytes)[i] =
        float4(gradient, signedDistance);                                         // .cu:1286, Q7
}
