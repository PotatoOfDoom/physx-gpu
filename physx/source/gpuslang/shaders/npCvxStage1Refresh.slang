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
// Slang port of convexConvexNphase_stage1Kernel (cudaGJKEPA.cu L2170-2388):
// the per-pair persistent-manifold refresh + narrowphase-rerun decision that
// runs before collide().  Per pair (4 CUDA lanes serialized):
//   - shape/flip setup (flip = type1 < type0 swaps types/scales/blob ptrs/cache
//     refs); PCM margins: sphere/capsule margin0 = scale0.y with extents0 = 0,
//     else extents = blobExtents * scale and margin = min(minExtent * 0.25,
//     toleranceLength * 0.05) (calculatePCMConvexMargin); minMargin =
//     fminf(margin0, margin1);
//   - aToB = t1.transformInv(t0) (q = conj(t1.q) * t0.q, p =
//     t1.q.rotateInv(t0.p - t1.p));
//   - manifold refresh per slot j < initialNbContacts: localAInB =
//     aToB.transform(pointA), dist = (localAInB - pointB).dot(localNormal) ->
//     new pen; keep iff |pointB - (localAInB - localNormal*dist)|^2 <=
//     (0.8 * minMargin)^2; kept slots compacted in place (unkept trailing slots
//     keep their previous contents);
//   - bRunNarrowphase = (newNb != initialNb); when the count is unchanged, the
//     invalidate_BoxConvex test runs instead: ratio/quat thresholds indexed by
//     newNb from the sphere (3-entry) or convex (5-entry) tables; deltaP =
//     max component |curRelPos - preRelPos|; rerun when deltaP > minMargin *
//     ratio or thresholdQ > quatA.dot(preQuatA) or thresholdQ > quatB.dot(
//     preQuatB), else when acos(deltaQ)*radius (radiusA = extents0.y for
//     sphere/capsule -- 0 for a sphere -- else |extents0|, radiusB =
//     |extents1|) exceeds the position threshold;
//   - lane-3 counts word: newNbContacts | (bRunNarrowphase ? initialNb << 8 :
//     0); tempKeepPairArray = bRunNarrowphase; on rerun lanes 0..2 rewrite
//     moreData with (aToB.p, 0), t0.q, t1.q; lane 3's y/z/w (warm-start data)
//     pass through unchanged.
//
// Blob extents arrive as direct inputs (the blob byte reads are tranche 142).

static const float FLT_MAXV = 3.4028235e38f;
static const uint EGT_SPHERE = 0u;
static const uint EGT_CAPSULE = 2u;
static const float MESH_MANIFOLD_EPSILON = 0.05f;
static const float BOX_MARGIN_RATIO = 0.25f;

static const float thresholdsConvex[5] = { 0.5f, 0.125f, 0.25f, 0.375f, 0.375f };
static const float quatThresholdsConvex[5] = { 0.9998f, 0.9999f, 0.9999f, 0.9999f, 0.9999f };
static const float thresholdsSphere[3] = { 0.5f, 0.1f, 0.75f };
static const float quatThresholdsSphere[3] = { 0.9995f, 0.9999f, 0.9997f };

float3 quatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
float3 quatRotateInv(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
// PxQuat operator* (this = a, arg = b)
float4 quatMul(float4 a, float4 b)
{
    return float4(a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y,
                  a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z,
                  a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x,
                  a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}
float quatDot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

float calculatePCMConvexMargin(float3 extents, float toleranceLength)
{
    const float mn = min(min(extents.x, extents.y), extents.z);
    const float toleranceMargin = toleranceLength * MESH_MANIFOLD_EPSILON;
    return min(mn * BOX_MARGIN_RATIO, toleranceMargin);
}

struct Stage1In
{
    uint4 meta;        // x=type0 y=type1 z,w pad
    float4 scale0;     // .xyz; .w = toleranceLength
    float4 scale1;     // .xyz
    float4 extents0;   // raw blob extents A (.xyz)
    float4 extents1;   // raw blob extents B (.xyz)
    float4 t0q, t0p;   // transformCache[ref0]
    float4 t1q, t1p;   // transformCache[ref1]
    float4 mNp[4];     // manifold normal+pen
    float4 mA[4];      // manifold A points
    float4 mB[4];      // manifold B points
    uint4 moreData0;   // mRelativePos bits (x,y,z), w
    uint4 moreData1;   // mQuatA bits
    uint4 moreData2;   // mQuatB bits
    uint4 moreData3;   // x=mNbContacts y=mNbWarmStartPoints z=mWarmStartA w=mWarmStartB
}

struct Stage1Out
{
    float4 mNp[4];     // manifold after refresh/compaction
    float4 mA[4];
    float4 mB[4];
    uint4 moreData0;   // after
    uint4 moreData1;
    uint4 moreData2;
    uint4 moreData3;
    uint4 result;      // x=tempKeepPair (bRunNarrowphase) y=newNbContacts z,w pad
}

struct Stage1Params { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<Stage1Params> gParams;

RWStructuredBuffer<Stage1In> gIn;    // binding 0
RWStructuredBuffer<Stage1Out> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxStage1Refresh(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const Stage1In s = gIn[t];

    uint type0 = s.meta.x;
    uint type1 = s.meta.y;
    float3 scale0 = s.scale0.xyz;
    float3 scale1 = s.scale1.xyz;
    float3 ext0raw = s.extents0.xyz;
    float3 ext1raw = s.extents1.xyz;
    const float toleranceLength = s.scale0.w;
    float4 t0q = s.t0q; float3 t0p = s.t0p.xyz;
    float4 t1q = s.t1q; float3 t1p = s.t1p.xyz;

    const bool flip = type1 < type0;
    if (flip)
    {
        const uint tt = type0; type0 = type1; type1 = tt;
        const float3 ts = scale0; scale0 = scale1; scale1 = ts;
        const float3 te = ext0raw; ext0raw = ext1raw; ext1raw = te;
        const float4 tq = t0q; t0q = t1q; t1q = tq;
        const float3 tp = t0p; t0p = t1p; t1p = tp;
    }

    float convexMargin0;
    float3 extents0;
    if (type0 == EGT_SPHERE || type0 == EGT_CAPSULE)
    {
        convexMargin0 = scale0.y;
        extents0 = float3(0, 0, 0);
    }
    else
    {
        extents0 = float3(ext0raw.x * scale0.x, ext0raw.y * scale0.y, ext0raw.z * scale0.z);
        convexMargin0 = calculatePCMConvexMargin(extents0, toleranceLength);
    }
    const float3 extents1 = float3(ext1raw.x * scale1.x, ext1raw.y * scale1.y, ext1raw.z * scale1.z);
    const float convexMargin1 = calculatePCMConvexMargin(extents1, toleranceLength);
    const float minMargin = min(convexMargin0, convexMargin1);

    // aToB = t1.transformInv(t0)
    const float4 aToBq = quatMul(float4(-t1q.x, -t1q.y, -t1q.z, t1q.w), t0q);
    const float3 aToBp = quatRotateInv(t1q, t0p - t1p);

    const uint initialNb = s.moreData3.x;

    // refresh + keep + in-place compaction (unkept trailing slots unchanged)
    Stage1Out r;
    for (uint j = 0u; j < 4u; ++j) { r.mNp[j] = s.mNp[j]; r.mA[j] = s.mA[j]; r.mB[j] = s.mB[j]; }

    float4 keptNp[4], keptA[4], keptB[4];
    uint newNb = 0u;
    for (uint j = 0u; j < 4u; ++j)
    {
        if (j >= initialNb) break;
        const float4 pointA = s.mA[j];
        const float4 pointB = s.mB[j];
        float4 pointNormPen = s.mNp[j];

        const float3 localAInB = quatRotate(aToBq, pointA.xyz) + aToBp;
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
            keptNp[newNb] = pointNormPen; keptA[newNb] = pointA; keptB[newNb] = pointB;
            ++newNb;
        }
    }
    for (uint j = 0u; j < newNb; ++j)
    {
        r.mNp[j] = keptNp[j]; r.mA[j] = keptA[j]; r.mB[j] = keptB[j];
    }

    bool bRun = newNb != initialNb;

    if (!bRun)
    {
        const float3 preRelPos = float3(asfloat(s.moreData0.x), asfloat(s.moreData0.y), asfloat(s.moreData0.z));
        const float4 preQuatA = float4(asfloat(s.moreData1.x), asfloat(s.moreData1.y), asfloat(s.moreData1.z), asfloat(s.moreData1.w));
        const float4 preQuatB = float4(asfloat(s.moreData2.x), asfloat(s.moreData2.y), asfloat(s.moreData2.z), asfloat(s.moreData2.w));

        const float radiusB = sqrt(extents1.x * extents1.x + extents1.y * extents1.y + extents1.z * extents1.z);

        float ratio, thresholdQ, radiusA;
        if (type0 == EGT_SPHERE || type0 == EGT_CAPSULE)
        {
            radiusA = extents0.y; // 0 for a sphere/capsule (extents0 = 0)
            ratio = thresholdsSphere[newNb];
            thresholdQ = quatThresholdsSphere[newNb];
        }
        else
        {
            radiusA = sqrt(extents0.x * extents0.x + extents0.y * extents0.y + extents0.z * extents0.z);
            ratio = thresholdsConvex[newNb];
            thresholdQ = quatThresholdsConvex[newNb];
        }

        const float thresholdP = minMargin * ratio;
        const float3 deltaPv = aToBp - preRelPos;
        const float deltaP = max(abs(deltaPv.x), max(abs(deltaPv.y), abs(deltaPv.z)));
        const float deltaQA = quatDot(t0q, preQuatA);
        const float deltaQB = quatDot(t1q, preQuatB);

        bool generateContacts = (deltaP > thresholdP) || (thresholdQ > deltaQA) || (thresholdQ > deltaQB);
        if (!generateContacts)
        {
            const float aRadian = deltaQA < 1.0f ? acos(deltaQA) : 0.0f;
            const float bRadian = deltaQB < 1.0f ? acos(deltaQB) : 0.0f;
            const float travelDistA = aRadian * radiusA;
            const float travelDistB = bRadian * radiusB;
            generateContacts = (travelDistA > thresholdP) || (travelDistB > thresholdP);
        }
        bRun = generateContacts;
    }

    // moreData writeback
    r.moreData0 = s.moreData0;
    r.moreData1 = s.moreData1;
    r.moreData2 = s.moreData2;
    r.moreData3 = s.moreData3;
    r.moreData3.x = newNb | (bRun ? (initialNb << 8u) : 0u);
    if (bRun)
    {
        r.moreData0 = uint4(asuint(aToBp.x), asuint(aToBp.y), asuint(aToBp.z), 0u);
        r.moreData1 = uint4(asuint(t0q.x), asuint(t0q.y), asuint(t0q.z), asuint(t0q.w));
        r.moreData2 = uint4(asuint(t1q.x), asuint(t1q.y), asuint(t1q.z), asuint(t1q.w));
    }
    r.result = uint4(bRun ? 1u : 0u, newNb, 0u, 0u);
    gOut[t] = r;
}
