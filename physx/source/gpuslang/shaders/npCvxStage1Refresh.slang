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

// npCvxStage1Refresh.slang
//
// P5 Step 4 (BDA, engine launch path): convexConvexNphase_stage1Kernel
// (cudaGJKEPA.cu L2170-2388) on the exact engine ABI —
//   (PxU32 numTests, PxReal toleranceLength, const PxgContactManagerInput*,
//    const PxsCachedTransform*, PxgShape*, PxgPersistentContactManifold*,
//    PxU32* tempKeepPairArray)   argSizes {4,4,8,8,8,8,8} -> 48-byte blob.
// Engine grid (PxgNarrowphaseCore.cpp ~L616): blockDim (32,4),
// numBlocks = ceil(numTests*4 / 128) — 4 CUDA lanes per pair,
// globalElementIndex = flatTid >> 2, tidInGroup = threadIdx.x % 4.
//
// The per-pair arithmetic is the tranche-153 bound-buffer port VERBATIM
// (bit-verified there); only the parameter plumbing and data access changed:
//   - shapes/transforms/manifold/keep-array are read and written through
//     Ptr<> from the raw device addresses in the push blob;
//   - the convex blob extents are read from shape.hullOrMeshPtr + 32
//     (sizeof(float4) + sizeof(uint4)), the real cooked-hull layout, instead
//     of arriving pre-extracted;
//   - the CUDA's 4 cooperating lanes are serialized on lane 0 of each group
//     (the intra-group shuffles/ballot become straight-line code; lanes 1-3
//     return after the numTests guard).  Memory effects are identical: the
//     kept slots are compacted in place (reads complete before writes, as
//     the warp-synchronous CUDA guarantees), unkept trailing slots keep
//     their previous contents, all four moreData uint4s are stored back
//     (lane 3's y/z/w warm-start words pass through), and
//     tempKeepPairArray[pair] = bRunNarrowphase.
//
// Per pair: flip = type1 < type0 swaps types/scales/blob ptrs/cache refs;
// PCM margins (sphere/capsule margin0 = scale0.y, extents0 = 0); aToB =
// t1.transformInv(t0); refresh keeps slot j iff the reprojected 2D distance
// <= (0.8 * minMargin)^2; bRunNarrowphase = count changed, else the
// invalidate_BoxConvex ratio/quat-threshold test; on rerun lanes 0..2
// rewrite moreData with (aToB.p, 0), t0.q, t1.q; the counts word packs
// newNb | (bRun ? initialNb << 8 : 0).

import pxgSlangNpAbi;

static const float thresholdsConvex[5] = { 0.5f, 0.125f, 0.25f, 0.375f, 0.375f };
static const float quatThresholdsConvex[5] = { 0.9998f, 0.9999f, 0.9999f, 0.9999f, 0.9999f };
static const float thresholdsSphere[3] = { 0.5f, 0.1f, 0.75f };
static const float quatThresholdsSphere[3] = { 0.9995f, 0.9999f, 0.9997f };

static const uint EGT_SPHERE = 0u;
static const uint EGT_CAPSULE = 2u;

float quatDot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

// Engine ABI blob: {4,4,8,8,8,8,8} at natural alignment = 48 bytes.
struct Stage1Params
{
    uint numTests;               // @0
    float toleranceLength;       // @4
    uint64_t cmInputs;           // @8   const PxgContactManagerInput*
    uint64_t transformCache;     // @16  const PxsCachedTransform*
    uint64_t shapes;             // @24  PxgShape*
    uint64_t cmManifolds;        // @32  PxgPersistentContactManifold*
    uint64_t tempKeepPairArray;  // @40  PxU32*
}

[[vk::push_constant]]
ConstantBuffer<Stage1Params> gS1;

[shader("compute")]
[numthreads(32, 4, 1)]
void convexConvexNphase_stage1Kernel(uint3 groupID : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    // flatTid = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x
    const uint flatTid = groupID.x * 128u + tid.y * 32u + tid.x;
    const uint globalElementIndex = flatTid >> 2u;

    if (globalElementIndex >= gS1.numTests)
        return;

    const uint tidInGroup = tid.x % 4u;
    if (tidInGroup != 0u)
        return; // serialized: lane 0 of each group performs all 4 lanes' work

    const float toleranceLength = gS1.toleranceLength;

    const PxgContactManagerInputGpu input =
        (Ptr<PxgContactManagerInputGpu>(gS1.cmInputs))[globalElementIndex];

    Ptr<PxgShapeGpu> shapes = Ptr<PxgShapeGpu>(gS1.shapes);
    const PxgShapeGpu shape0 = shapes[input.shapeRef0];
    const PxgShapeGpu shape1 = shapes[input.shapeRef1];

    uint64_t convexPtrA = shape0.hullOrMeshPtr;
    uint64_t convexPtrB = shape1.hullOrMeshPtr;
    float3 scale0 = float3(shape0.scaleX, shape0.scaleY, shape0.scaleZ);
    float3 scale1 = float3(shape1.scaleX, shape1.scaleY, shape1.scaleZ);
    uint type0 = shape0.type;
    uint type1 = shape1.type;
    uint transformCacheRef0 = input.transformCacheRef0;
    uint transformCacheRef1 = input.transformCacheRef1;

    const bool flip = type1 < type0;
    if (flip)
    {
        uint tu = type0; type0 = type1; type1 = tu;
        const float3 ts = scale0; scale0 = scale1; scale1 = ts;
        const uint64_t tp = convexPtrA; convexPtrA = convexPtrB; convexPtrB = tp;
        tu = transformCacheRef0; transformCacheRef0 = transformCacheRef1; transformCacheRef1 = tu;
    }

    float convexMargin0;
    float3 extents0;
    if (type0 == EGT_SPHERE || type0 == EGT_CAPSULE)
    {
        convexMargin0 = scale0.y; // sphere radius
        extents0 = float3(0, 0, 0);
    }
    else
    {
        // extents float4 at blob + sizeof(float4) + sizeof(uint4)
        const float4 extents0_f4 = (Ptr<float4>(convexPtrA + 32u))[0];
        extents0 = float3(extents0_f4.x * scale0.x, extents0_f4.y * scale0.y, extents0_f4.z * scale0.z);
        convexMargin0 = npAbiPcmConvexMargin(extents0, toleranceLength);
    }
    const float4 extents1_f4 = (Ptr<float4>(convexPtrB + 32u))[0];
    const float3 extents1 = float3(extents1_f4.x * scale1.x, extents1_f4.y * scale1.y, extents1_f4.z * scale1.z);
    const float convexMargin1 = npAbiPcmConvexMargin(extents1, toleranceLength);
    const float minMargin = min(convexMargin0, convexMargin1);

    Ptr<PxsCachedTransformGpu> cache = Ptr<PxsCachedTransformGpu>(gS1.transformCache);
    const PxsCachedTransformGpu tc0 = cache[transformCacheRef0];
    const PxsCachedTransformGpu tc1 = cache[transformCacheRef1];
    const float4 t0q = float4(tc0.qx, tc0.qy, tc0.qz, tc0.qw);
    const float3 t0p = float3(tc0.px, tc0.py, tc0.pz);
    const float4 t1q = float4(tc1.qx, tc1.qy, tc1.qz, tc1.qw);
    const float3 t1p = float3(tc1.px, tc1.py, tc1.pz);

    // aToB = t1.transformInv(t0)
    const float4 aToBq = npAbiQuatMul(float4(-t1q.x, -t1q.y, -t1q.z, t1q.w), t0q);
    // P9.86: cudaGJKEPA.cu:2238 t1.transformInv(t0) is PxTransform.h:162-168 q.getConjugate().rotate(src.p - p),
    // NOT q.rotateInv(src.p - p): same bits for every nonzero value, but a zero result can take the
    // other sign (t1q = PxQuat(PxHalfPi, x), t0p - t1p = (0,-3,0): CUDA x = -0, rotateInv +0).
    const float3 aToBp = npAbiQuatRotate(float4(-t1q.x, -t1q.y, -t1q.z, t1q.w), t0p - t1p);

    Ptr<PxgPersistentContactManifoldGpu> manifold =
        Ptr<PxgPersistentContactManifoldGpu>(gS1.cmManifolds + uint64_t(globalElementIndex) * 256u);

    // the 4 lanes' moreData uint4 reads (words 0-15 of the manifold tail)
    const float4 moreRelPos = manifold[0].relativePos;
    const float4 moreQuatA = manifold[0].quatA;
    const float4 moreQuatB = manifold[0].quatB;
    const uint initialNbContacts = manifold[0].nbContacts;
    const uint moreW13 = manifold[0].nbWarmStartPoints;
    const uint moreW14 = manifold[0].warmStartA;
    const uint moreW15 = manifold[0].warmStartB;

    // refresh + keep + in-place compaction.  All 4 lanes' reads complete
    // before any write (the CUDA ballot separates them), so read first.
    float4 srcNp[4], srcA[4], srcB[4];
    for (uint j = 0u; j < 4u; ++j)
    {
        if (j < initialNbContacts)
        {
            srcNp[j] = manifold[0].localNormalPen[j];
            srcA[j] = manifold[0].localContactA[j];
            srcB[j] = manifold[0].localContactB[j];
        }
        else
        {
            srcNp[j] = float4(0, 0, 0, 0); srcA[j] = float4(0, 0, 0, 0); srcB[j] = float4(0, 0, 0, 0);
        }
    }

    uint newNbContacts = 0u;
    for (uint j = 0u; j < 4u; ++j)
    {
        if (j >= initialNbContacts)
            break;
        const float4 pointA = srcA[j];
        const float4 pointB = srcB[j];
        float4 pointNormPen = srcNp[j];

        const float3 localAInB = npAbiQuatRotate(aToBq, pointA.xyz) + aToBp;
        const float3 localBInB = pointB.xyz;
        const float3 v = localAInB - localBInB;
        const float3 localNormal = pointNormPen.xyz;
        const float dist = dot(v, localNormal);
        pointNormPen.w = dist;

        const float3 projectedPoint = localAInB - localNormal * dist;
        const float3 projectedDifference = localBInB - projectedPoint;
        const float distance2d = dot(projectedDifference, projectedDifference);
        const float projectBreakingThreshold = minMargin * 0.8f;
        const float sqThreshold = projectBreakingThreshold * projectBreakingThreshold;

        if (distance2d <= sqThreshold)
        {
            manifold[0].localContactB[newNbContacts] = pointB;
            manifold[0].localContactA[newNbContacts] = pointA;
            manifold[0].localNormalPen[newNbContacts] = pointNormPen;
            ++newNbContacts;
        }
    }

    bool bRunNarrowphase = newNbContacts != initialNbContacts;

    if (!bRunNarrowphase)
    {
        const float3 relativePos = moreRelPos.xyz;
        const float4 quatA = moreQuatA;
        const float4 quatB = moreQuatB;

        const float radiusB = sqrt(extents1.x * extents1.x + extents1.y * extents1.y + extents1.z * extents1.z);

        float ratio, thresholdQ, radiusA;
        if (type0 == EGT_SPHERE || type0 == EGT_CAPSULE)
        {
            radiusA = extents0.y; // 0 (extents0 zeroed on the sphere/capsule path)
            ratio = thresholdsSphere[newNbContacts];
            thresholdQ = quatThresholdsSphere[newNbContacts];
        }
        else
        {
            radiusA = sqrt(extents0.x * extents0.x + extents0.y * extents0.y + extents0.z * extents0.z);
            ratio = thresholdsConvex[newNbContacts];
            thresholdQ = quatThresholdsConvex[newNbContacts];
        }

        const float thresholdP = minMargin * ratio;
        const float3 deltaPv = aToBp - relativePos;
        const float deltaP = max(abs(deltaPv.x), max(abs(deltaPv.y), abs(deltaPv.z)));
        const float deltaQA = quatDot(t0q, quatA);
        const float deltaQB = quatDot(t1q, quatB);

        bool generateContacts = (deltaP > thresholdP) || (thresholdQ > deltaQA) || (thresholdQ > deltaQB);
        if (!generateContacts)
        {
            const float aRadian = deltaQA < 1.0f ? acos(deltaQA) : 0.0f;
            const float bRadian = deltaQB < 1.0f ? acos(deltaQB) : 0.0f;
            const float travelDistA = aRadian * radiusA;
            const float travelDistB = bRadian * radiusB;
            generateContacts = (travelDistA > thresholdP) || (travelDistB > thresholdP);
        }
        bRunNarrowphase = generateContacts;
    }

    // lane 3: counts word + keep-pair flag; lanes 0..2 on rerun: moreData rewrite.
    // All four moreData uint4s are stored back (lane 3's y/z/w pass through).
    (Ptr<uint>(gS1.tempKeepPairArray))[globalElementIndex] = bRunNarrowphase ? 1u : 0u;

    if (bRunNarrowphase)
    {
        manifold[0].relativePos = float4(aToBp.x, aToBp.y, aToBp.z, asfloat(0u));
        manifold[0].quatA = t0q;
        manifold[0].quatB = t1q;
    }
    else
    {
        manifold[0].relativePos = moreRelPos;
        manifold[0].quatA = moreQuatA;
        manifold[0].quatB = moreQuatB;
    }
    manifold[0].nbContacts = newNbContacts | (bRunNarrowphase ? (initialNbContacts << 8u) : 0u);
    manifold[0].nbWarmStartPoints = moreW13;
    manifold[0].warmStartA = moreW14;
    manifold[0].warmStartB = moreW15;
}
