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

// boxBoxNphase.slang
//
// Slang port of boxBoxNphase_Kernel (gpunarrowphase/src/CUDA/cudaBox.cu) on the
// ENGINE launch ABI (BDA).  This is the full box-vs-box narrow-phase contact
// generation entry point: it mirrors the CUDA kernel EXACTLY — thread mapping
// (4 threads per pair, blockDim 64 = 16 pairs / block), the collaborative
// separating-axis test + feature selection, the incident-face polygon, the
// box-box clipping core (calculateContacts), the 4-point manifold reduction,
// and the contact-stream / patch-stream output tail with the atomic offset
// reservation on PxgPatchAndContactCounters.
//
// The per-stage math is transcribed from the bit-verified box-box primitive
// shaders (npBoxSat / npBoxFeature / npBoxIncident / npBoxContacts / npBoxReduce)
// and the contact-output glue (npContactPatch / npInsertPatch).  The CUDA runs a
// 4-lane group with __shfl_sync / __any_sync / __syncwarp(groupMask); here the
// group lives inside a 64-thread block (16 groups) and every warp primitive is a
// groupshared exchange guarded by a block-wide barrier.  Because llvmpipe's
// subgroup width is not 32, and to keep every block-wide barrier in uniform
// control flow, the CUDA's early `return false` (SAT separation) and the
// out-of-range guard are converted to predication: every thread reaches every
// barrier; only the data writes are gated.  Separated / out-of-range pairs
// produce zero contacts (matching the CUDA return-false path), and — exactly
// like the CUDA — the output tail still runs for every in-range pair so its
// cmOutput touch/patch flags are updated even when nbContacts == 0.
//
// Element structs read/written through device pointers (PxgContactManagerInput,
// PxgShape, PxsCachedTransform, PxsContactManagerOutput, PxgPatchAndContact-
// Counters, PxsMaterialData, PxContact stream, PxContactPatch stream) are
// INDEXED, so their strides are part of the ABI; the mirrors below match the
// engine C++ layouts byte-for-byte.  Device pointers stored back into
// PxsContactManagerOutput (contactPatches / contactPoints / contactForces) are
// real BDA addresses (startContact* + byte offset), matching the CUDA which
// writes raw device pointers there.

import pxgSlangCommon; // PxgShape, PxsCachedTransform, PxgMat33, mat33FromQuat, PXG_MAX_F32

static const uint  INVALID = 0xffffffffu;
static const uint  MAXP = 6u;                 // PXG_MAX_NUM_POINTS_PER_CONTACT_PATCH
static const float BIG = PXG_MAX_F32;
static const uint  CONTACT_SIZE = 16u;        // sizeof(PxContact)
static const uint  FORCE_SIZE = 4u;           // sizeof(PxU32)
static const uint  PATCH_SIZE = 64u;          // sizeof(PxContactPatch)

// PxgPatchAndContactCounters::OverflowError
static const uint OVERFLOW_CONTACT = 1u;
static const uint OVERFLOW_FORCE = 2u;
static const uint OVERFLOW_PATCH = 4u;

// PxsContactManagerStatusFlag
static const uint FLAG_HAS_NO_TOUCH = 1u;
static const uint FLAG_HAS_TOUCH = 2u;
static const uint FLAG_TOUCH_KNOWN = 3u; // eHAS_NO_TOUCH | eHAS_TOUCH

// PxMaterialFlag / PxCombineMode
static const uint MODE_AVERAGE = 0u, MODE_MIN = 1u, MODE_MULTIPLY = 2u, MODE_MAX = 3u;
static const uint MAT_DISABLE_FRICTION = 1u, MAT_DISABLE_STRONG_FRICTION = 2u, MAT_COMPLIANT_ACC_SPRING = 16u;

// ---------------------------------------------------------------------------
// Element-struct mirrors (byte-exact vs the engine C++ layouts).
// ---------------------------------------------------------------------------

// PxgContactManagerInput (PxgContactManager.h, PX_ALIGN(16), 16 bytes)
struct PxgContactManagerInput
{
    uint shapeRef0;
    uint shapeRef1;
    uint transformCacheRef0;
    uint transformCacheRef1;
}

// PxgPatchAndContactCounters (cudaNpCommon.h, PX_ALIGN(16), 16 bytes)
struct PxgPatchAndContactCounters
{
    uint patchesBytes;
    uint contactsBytes;
    uint forceAndIndiceBytes;
    uint overflowError;
}

// PxsContactManagerOutput (PxsContactManagerState.h, PX_ALIGN(16), 48 bytes).
// The four device pointers are real BDA addresses; the packed status byte-word
// and nbContacts/flags half-words mirror the C++ bit layout.
struct PxsContactManagerOutput
{
    uint64_t contactPatches;    // @0
    uint64_t contactPoints;     // @8
    uint64_t contactForces;     // @16
    uint64_t frictionPatches;   // @24
    uint allflagsWord;          // @32 : allflagsStart(0:8)|nbPatches(8:16)|statusFlag(16:24)|prevPatches(24:32)
    uint nbContacts_flags;      // @36 : nbContacts(0:16)|flags(16:32)
    uint2 pad;                  // @40 : PxU8 pad[8]
}

// PxsMaterialData (PxsMaterialCore.h, 24 bytes, align 4).
struct PxsMaterialData
{
    float dynamicFriction;      // @0
    float staticFriction;       // @4
    float restitution;          // @8
    float damping;              // @12
    uint packedFlags;           // @16 : flags(0:16 PxU16) | fricCombineMode(16:24) | restCombineMode(24:32)
    uint packedDamp;            // @20 : dampingCombineMode(0:8) | pad
}

// ---------------------------------------------------------------------------
// Push constant: mirrors the CUDA __global__ parameter list EXACTLY.
// Natural alignment (matches the engine void** blob / abiAlign): numTests(4)
// then 14 device pointers (8-aligned, so the pointer bank starts at @8), then
// the three PxU32 byte-limits and the PxReal toleranceLength.  = 136 bytes.
// ---------------------------------------------------------------------------
struct BoxBoxParams
{
    uint numTests;                    // @0
    uint64_t cmInputs;                // @8   const PxgContactManagerInput*
    uint64_t cmOutputs;               // @16  PxsContactManagerOutput*
    uint64_t shapes;                  // @24  PxgShape*
    uint64_t transformCache;          // @32  const PxsCachedTransform*
    uint64_t contactDistance;         // @40  const PxReal*
    uint64_t materials;               // @48  const PxsMaterialData*
    uint64_t contactStream;           // @56  PxU8*
    uint64_t patchStream;             // @64  PxU8*
    uint64_t patchAndContactCounters; // @72  PxgPatchAndContactCounters*
    uint64_t touchChangeFlags;        // @80  PxU32*
    uint64_t patchChangeFlags;        // @88  PxU32*
    uint64_t startContactPatches;     // @96  PxU8*
    uint64_t startContactPoints;      // @104 PxU8*
    uint64_t startContactForces;      // @112 PxU8*
    uint patchBytesLimit;             // @120
    uint contactBytesLimit;           // @124
    uint forceBytesLimit;             // @128
    float toleranceLength;            // @132 (unused by this kernel, present for ABI parity)
}                                     // = 136

[[vk::push_constant]]
ConstantBuffer<BoxBoxParams> gParams;

// ---------------------------------------------------------------------------
// Groupshared scratch (64-thread block = 16 groups of 4).  Well under llvmpipe's
// 32 KiB budget (~9 KiB used).
// ---------------------------------------------------------------------------
groupshared float4 gTempBuff[192];  // TempBoxBoxBuffer: 3 float4 / thread (localContacts)
groupshared float3 gPts[64];        // shVerts: incident-polygon vertex / lane
groupshared float3 gExch[64];       // float3 butterfly / broadcast exchange
groupshared float  gsA[64];         // float exchange (SAT min, argmin values)
groupshared float  gsB[64];         // float exchange (SAT sign, second argmin)
groupshared uint   gsFeat[64];      // SAT feature
groupshared uint   gsSep[64];       // SAT separated flag
groupshared uint   gsPen[64];       // calculateContacts: penetration flag
groupshared uint   gsArea[64];      // calculateContacts: area flag
groupshared uint   gsArea2[64];     // calculateContacts: corner-inside flag
groupshared uint   gsCount[64];     // per-lane contact count
groupshared float4 gsLC[64];        // reduction: chosen local-space contacts
groupshared uint   gsCByte[64];     // output: contactByteOffset broadcast (lane 3)

float comp3(float3 v, uint i) { return i == 0u ? v.x : (i == 1u ? v.y : v.z); }

// newTransformV.transform(v) : PxMat34 transform (rotation cols + translation)
float3 xform(float3 c0, float3 c1, float3 c2, float3 p, float3 v)
{
    return c0 * v.x + c1 * v.y + c2 * v.z + p;
}

// PxMat33::transformTranspose(v) = (col0.v, col1.v, col2.v)
float3 mat33TT(float3 c0, float3 c1, float3 c2, float3 v)
{
    return float3(dot(c0, v), dot(c1, v), dot(c2, v));
}

// contains(): 2D point-in-quad crossing test (mirrors cudaBox.cu contains()).
bool contains4(float2 v0, float2 v1, float2 v2, float2 v3, float2 p, float2 bmin, float2 bmax)
{
    if (p.x < bmin.x || p.x > bmax.x || p.y < bmin.y || p.y > bmax.y)
        return false;

    float2 V[4] = { v0, v1, v2, v3 };
    const float eps = 1e-6f;
    const float zero = 0.0f;
    uint intersectionPoints = 0u;
    bool inside = false;
    bool done = false;

    uint j = 3u;
    for (uint i = 0u; i < 4u && !done; ++i)
    {
        const float jx = V[j].x, jy = V[j].y;
        const float ix = V[i].x, iy = V[i].y;
        if ((p.x == jx && p.y == jy) || (p.x == ix && p.y == iy))
        {
            inside = true; done = true;
        }
        else
        {
            const bool yflag0 = jy > p.y;
            const bool yflag1 = iy > p.y;
            if (yflag0 != yflag1)
            {
                const float jix = ix - jx;
                const float jiy = iy - jy;
                const float jty = p.y - jy;
                const float part1 = jty * jix;
                const float part2 = (jx + eps) * jiy;
                const float part3 = p.x * jiy;
                const bool comp = jiy > zero;
                const float tmp = part1 + part2;
                const float comp1 = comp ? tmp : part3;
                const float comp2 = comp ? part3 : tmp;
                if (comp1 >= comp2)
                {
                    if (intersectionPoints == 1u) { inside = false; done = true; }
                    else intersectionPoints++;
                }
            }
            j = i;
        }
    }
    if (!done) inside = intersectionPoints > 0u;
    return inside;
}

// intersectSegmentAABB(): parametric segment-vs-AABB slab test.
bool segAABB(float3 p0, float3 d, float3 mx, float3 mn, out float tmin, out float tmax)
{
    const float eps = 1e-6f;
    const float3 absV = abs(d);
    float tminf = 0.0f;
    float tmaxf = 1.0f;
    tmin = 0.0f; tmax = 1.0f;

    const bool pX = eps > absV.x;
    const bool pY = eps > absV.y;
    const bool pZ = eps > absV.z;

    if ((pX && (p0.x > mx.x || p0.x < mn.x)) ||
        (pY && (p0.y > mx.y || p0.y < mn.y)) ||
        (pZ && (p0.z > mx.z || p0.z < mn.z)))
        return false;

    const float3 odd = float3(1.0f / d.x, 1.0f / d.y, 1.0f / d.z);
    const float3 t1 = float3(pX ? 0.0f : (mn.x - p0.x) * odd.x,
                             pY ? 0.0f : (mn.y - p0.y) * odd.y,
                             pZ ? 0.0f : (mn.z - p0.z) * odd.z);
    const float3 t2 = float3(pX ? BIG : (mx.x - p0.x) * odd.x,
                             pY ? BIG : (mx.y - p0.y) * odd.y,
                             pZ ? BIG : (mx.z - p0.z) * odd.z);
    const float3 tt1 = min(t1, t2);
    const float3 tt2 = max(t1, t2);
    const float ft1 = max(tt1.x, max(tt1.y, tt1.z));
    const float ft2 = min(tt2.x, min(tt2.y, tt2.z));
    tminf = max(ft1, tminf);
    tmaxf = min(tmaxf, ft2);
    tmin = tminf; tmax = tmaxf;
    return !((tminf > tmaxf) || (tminf > 1.0f));
}

// PxsCombineMaterials scalar combine.
float combineScalars(float a, float b, uint mode)
{
    if (mode == MODE_AVERAGE) return 0.5f * (a + b);
    if (mode == MODE_MIN) return min(a, b);
    if (mode == MODE_MULTIPLY) return a * b;
    if (mode == MODE_MAX) return max(a, b);
    return 0.0f;
}

// ---------------------------------------------------------------------------
// boxBoxNphase_Kernel : full box-box narrow phase on the engine ABI.
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void boxBoxNphase_Kernel(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    const uint t = dtid.x;
    const uint lt = gtid.x;
    const uint lane = lt & 3u;         // threadIndexInGroup
    const uint gbase = lt & ~3u;       // group start (local block thread index)
    const uint lbase = 3u * gbase;     // group base into gTempBuff (localContacts)
    const uint workIndex = t >> 2u;    // one pair per 4 threads

    const uint numTests = gParams.numTests;
    const bool valid = workIndex < numTests;
    const uint safeWi = valid ? workIndex : (numTests > 0u ? numTests - 1u : 0u);

    // ---- device pointers ----
    Ptr<PxgContactManagerInput> cmInputs = Ptr<PxgContactManagerInput>(gParams.cmInputs);
    Ptr<PxsContactManagerOutput> cmOutputs = Ptr<PxsContactManagerOutput>(gParams.cmOutputs);
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gParams.shapes);
    Ptr<PxsCachedTransform> transformCache = Ptr<PxsCachedTransform>(gParams.transformCache);
    Ptr<float> contactDistance = Ptr<float>(gParams.contactDistance);
    Ptr<PxsMaterialData> materials = Ptr<PxsMaterialData>(gParams.materials);
    Ptr<PxgPatchAndContactCounters> counters = Ptr<PxgPatchAndContactCounters>(gParams.patchAndContactCounters);
    Ptr<uint> touchChangeFlags = Ptr<uint>(gParams.touchChangeFlags);
    Ptr<uint> patchChangeFlags = Ptr<uint>(gParams.patchChangeFlags);

    // ---- per-pair inputs (all 4 lanes read the same pair) ----
    const PxgContactManagerInput cin = cmInputs[safeWi];
    const uint tcRef0 = cin.transformCacheRef0;
    const uint tcRef1 = cin.transformCacheRef1;
    const PxsCachedTransform tc0 = transformCache[tcRef0];
    const PxsCachedTransform tc1 = transformCache[tcRef1];
    const float cDist = contactDistance[tcRef0] + contactDistance[tcRef1];

    const PxgShape shape0 = shapes[cin.shapeRef0];
    const PxgShape shape1 = shapes[cin.shapeRef1];
    const float3 box0e = shape0.scale();
    const float3 box1e = shape1.scale();

    // transform0 / transform1 : PxMat34 from the cached PxTransform.
    const PxgMat33 rot0 = mat33FromQuat(tc0.transform.q());
    const PxgMat33 rot1 = mat33FromQuat(tc1.transform.q());
    const float3 c00 = rot0.col0, c01 = rot0.col1, c02 = rot0.col2, tp0 = tc0.transform.p();
    const float3 c10 = rot1.col0, c11 = rot1.col1, c12 = rot1.col2, tp1 = tc1.transform.p();

    // =======================================================================
    // Stage 1 : separating-axis test + feature/sign selection (npBoxSat).
    // =======================================================================
    const float uEps = 1e-6f;
    const float3 dP = tp1 - tp0;
    // transform1To0 = transform0.transformTranspose(transform1)
    const float3 r0 = float3(dot(c00, c10), dot(c01, c10), dot(c02, c10));
    const float3 r1 = float3(dot(c00, c11), dot(c01, c11), dot(c02, c11));
    const float3 r2 = float3(dot(c00, c12), dot(c01, c12), dot(c02, c12));
    const float3 tp = float3(dot(c00, dP), dot(c01, dP), dot(c02, dP));
    const float3 ab0 = float3(abs(r0.x), abs(r1.x), abs(r2.x)) + uEps;
    const float3 ab1 = float3(abs(r0.y), abs(r1.y), abs(r2.y)) + uEps;
    const float3 ab2 = float3(abs(r0.z), abs(r1.z), abs(r2.z)) + uEps;

    float minO = BIG;
    uint feat = 0u;
    float sgn = 0.0f;
    bool sep = false;

    if (lane < 3u)
    {
        const uint tid = lane;
        const float3 col = (tid == 0u) ? r0 : ((tid == 1u) ? r1 : r2);
        const float3 abCol = abs(col) + uEps;
        const float3 abT = (tid == 0u) ? ab0 : ((tid == 1u) ? ab1 : ab2);
        const float tpc = comp3(tp, tid);
        const float ea = comp3(box0e, tid);
        const float eb = comp3(box1e, tid);

        const float rbA = dot(abT, box1e);
        const float oA = (ea + rbA) - abs(tpc) + cDist;
        const float sA = tpc;

        const float sB = dot(tp, col);
        const float raB = dot(abCol, box0e);
        const float oB = (raB + eb) - abs(sB) + cDist;

        minO = oA; feat = tid; sgn = sA;
        if (minO > oB) { minO = oB; sgn = sB; feat = tid + 3u; }

        bool sepFace = (0.0f > oA) || (0.0f > oB);

        const uint iHi = (tid == 2u) ? 1u : 2u;
        const uint iLo = (tid == 0u) ? 1u : 0u;
        const float b1Lo = comp3(box1e, iLo);
        const float b1Hi = comp3(box1e, iHi);

        const float as1 = abs(col.y * tp.z - col.z * tp.y);
        const float ra1 = abCol.z * box0e.y + abCol.y * box0e.z;
        const float rb1 = comp3(ab0, iHi) * b1Lo + comp3(ab0, iLo) * b1Hi;
        const bool sep1 = as1 > (ra1 + rb1 + cDist);

        const float as2 = abs(col.z * tp.x - col.x * tp.z);
        const float ra2 = abCol.z * box0e.x + abCol.x * box0e.z;
        const float rb2 = comp3(ab1, iHi) * b1Lo + comp3(ab1, iLo) * b1Hi;
        const bool sep2 = as2 > (ra2 + rb2 + cDist);

        const float as3 = abs(col.x * tp.y - col.y * tp.x);
        const float ra3 = abCol.y * box0e.x + abCol.x * box0e.y;
        const float rb3 = comp3(ab2, iHi) * b1Lo + comp3(ab2, iLo) * b1Hi;
        const bool sep3 = as3 > (ra3 + rb3 + cDist);

        sep = sepFace || sep1 || sep2 || sep3;
    }

    gsA[lt] = minO;
    gsFeat[lt] = feat;
    gsB[lt] = sgn;
    gsSep[lt] = sep ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();

    const bool separated = (gsSep[gbase] | gsSep[gbase + 1u] | gsSep[gbase + 2u]) != 0u;
    const float smin0 = gsA[gbase], smin1 = gsA[gbase + 1u], smin2 = gsA[gbase + 2u];
    uint pick;
    if (smin0 <= smin1 && smin0 <= smin2) pick = gbase;
    else if (smin1 <= smin2)              pick = gbase + 1u;
    else                                  pick = gbase + 2u;
    const uint feature = gsFeat[pick];
    const float sign = gsB[pick];
    const bool active = valid && !separated;
    GroupMemoryBarrierWithGroupSync();

    // =======================================================================
    // Stage 2 : feature -> newTransformV / transform1ToNew / localNormal
    //           (npBoxFeature).  Per-pair uniform, no barriers.
    // =======================================================================
    const bool neg = 0.0f >= sign;
    float3 mtd, nc0, nc1, nc2, np;
    float e0, e1;
    if (feature == 0u)
    {
        if (neg) { mtd = c00;  nc0 = -c02; nc1 = c01;  nc2 = c00; }
        else     { mtd = -c00; nc0 = c02;  nc1 = c01;  nc2 = -c00; }
        np = tp0 - mtd * box0e.x; e0 = box0e.z; e1 = box0e.y;
    }
    else if (feature == 1u)
    {
        if (neg) { mtd = c01;  nc0 = c00; nc1 = -c02; nc2 = c01; }
        else     { mtd = -c01; nc0 = c00; nc1 = c02;  nc2 = -c01; }
        np = tp0 - mtd * box0e.y; e0 = box0e.x; e1 = box0e.z;
    }
    else if (feature == 2u)
    {
        if (neg) { mtd = c02;  nc0 = c00; nc1 = c01;  nc2 = c02; }
        else     { mtd = -c02; nc0 = c00; nc1 = -c01; nc2 = -c02; }
        np = tp0 - mtd * box0e.z; e0 = box0e.x; e1 = box0e.y;
    }
    else if (feature == 3u)
    {
        if (neg) { mtd = c10;  nc0 = c12;  nc1 = c11; nc2 = -c10; }
        else     { mtd = -c10; nc0 = -c12; nc1 = c11; nc2 = c10; }
        np = tp1 + mtd * box1e.x; e0 = box1e.z; e1 = box1e.y;
    }
    else if (feature == 4u)
    {
        if (neg) { mtd = c11;  nc0 = c10; nc1 = c12;  nc2 = -c11; }
        else     { mtd = -c11; nc0 = c10; nc1 = -c12; nc2 = c11; }
        np = tp1 + mtd * box1e.y; e0 = box1e.x; e1 = box1e.z;
    }
    else
    {
        if (neg) { mtd = c12;  nc0 = c10; nc1 = -c11; nc2 = -c12; }
        else     { mtd = -c12; nc0 = c10; nc1 = c11;  nc2 = c12; }
        np = tp1 + mtd * box1e.z; e0 = box1e.x; e1 = box1e.y;
    }

    const bool useT1 = feature < 3u;
    const float3 oc0 = useT1 ? c10 : c00;
    const float3 oc1 = useT1 ? c11 : c01;
    const float3 oc2 = useT1 ? c12 : c02;
    const float3 oP  = useT1 ? tp1 : tp0;

    // transform1ToNew = newTransformV.transformTranspose(other)
    const float3 t1n0 = mat33TT(nc0, nc1, nc2, oc0);
    const float3 t1n1 = mat33TT(nc0, nc1, nc2, oc1);
    const float3 t1n2 = mat33TT(nc0, nc1, nc2, oc2);
    const float3 t1nP = mat33TT(nc0, nc1, nc2, oP - np);
    const float3 localNormal = mat33TT(nc0, nc1, nc2, mtd);

    // =======================================================================
    // Stage 3 : incident-face polygon vertex + incident normal (npBoxIncident /
    //           getIncidentPolygon4).
    // =======================================================================
    const float3 axis = useT1 ? -localNormal : localNormal;
    const float3 iext = useT1 ? box1e : box0e;

    float dd = 0.0f;
    if (lane < 3u)
    {
        const float3 col = (lane == 0u) ? t1n0 : ((lane == 1u) ? t1n1 : t1n2);
        dd = dot(col, axis);
    }
    gsA[lt] = abs(dd);
    gsB[lt] = dd;
    GroupMemoryBarrierWithGroupSync();

    const float absd0 = gsA[gbase], absd1 = gsA[gbase + 1u], absd2 = gsA[gbase + 2u];
    const float dv0 = gsB[gbase], dv1 = gsB[gbase + 1u], dv2 = gsB[gbase + 2u];
    const float s0 = (lane == 0u || lane == 3u) ? -1.0f : 1.0f;
    const float s1 = (lane & 2u) != 0u ? 1.0f : -1.0f;

    float3 isgn;
    float3 faceNormal;
    if (absd0 >= absd1 && absd0 >= absd2)
    {
        isgn.x = (dv0 > 0.0f) ? -1.0f : 1.0f; isgn.y = s0; isgn.z = s1;
        faceNormal = t1n0 * isgn.x;
    }
    else if (absd1 >= absd2)
    {
        isgn.y = (dv1 > 0.0f) ? -1.0f : 1.0f; isgn.x = s0; isgn.z = s1;
        faceNormal = t1n1 * isgn.y;
    }
    else
    {
        isgn.z = (dv2 > 0.0f) ? -1.0f : 1.0f; isgn.x = s0; isgn.y = s1;
        faceNormal = t1n2 * isgn.z;
    }
    const float3 ivert = iext * isgn;
    const float3 point = xform(t1n0, t1n1, t1n2, t1nP, ivert); // incident vertex, box-A local
    const float3 normal = mtd; // patch normal (group-uniform)

    // publish the incident polygon (shVerts)
    GroupMemoryBarrierWithGroupSync();
    gPts[lt] = point;
    GroupMemoryBarrierWithGroupSync();

    // =======================================================================
    // Stage 4 : calculateContacts (npBoxContacts) -> up to 3 contacts / lane,
    //           written at gTempBuff[lbase + 4*count + lane].
    // =======================================================================
    // calculateBounds4 : width-4 butterfly AABB of the 4 incident vertices.
    float3 bmin = point, bmax = point;
    gExch[lt] = point;
    GroupMemoryBarrierWithGroupSync();
    const float3 e1v = gExch[lt ^ 1u];
    GroupMemoryBarrierWithGroupSync();
    bmin = min(bmin, e1v); bmax = max(bmax, e1v);
    gExch[lt] = bmin;
    GroupMemoryBarrierWithGroupSync();
    const float3 e2n = gExch[lt ^ 2u];
    GroupMemoryBarrierWithGroupSync();
    bmin = min(bmin, e2n);
    gExch[lt] = bmax;
    GroupMemoryBarrierWithGroupSync();
    const float3 e2x = gExch[lt ^ 2u];
    GroupMemoryBarrierWithGroupSync();
    bmax = max(bmax, e2x);

    const float extentX = 1.00001f * e0;
    const float extentY = 1.00001f * e1;
    const float3 ext = float3(extentX, extentY, BIG);
    const float3 negExt = float3(-extentX, -extentY, -cDist - 1e-7f);

    uint count = 0u;

    // 1. projection / area contact
    const float z = -point.z;
    const bool pPen = cDist > z;
    bool pArea = false;
    if (pPen)
    {
        if (extentX >= abs(point.x) && extentY >= abs(point.y))
        {
            pArea = true;
            if (active) gTempBuff[lbase + lane] = float4(point, z);
            count = 1u;
        }
    }
    gsPen[lt] = pPen ? 1u : 0u;
    gsArea[lt] = pArea ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();

    const bool allArea = (gsArea[gbase] & gsArea[gbase + 1u] & gsArea[gbase + 2u] & gsArea[gbase + 3u]) != 0u;

    // 2. reference-face corner inside incident polygon
    bool pArea2 = false;
    if (!allArea)
    {
        const float denom = faceNormal.z;
        const float qx = (lane == 0u || lane == 3u) ? -extentX : extentX;
        const float qy = (lane & 2u) != 0u ? -extentY : extentY;
        const float2 q0 = float2(qx, qy);
        const float3 P0 = gPts[gbase], P1 = gPts[gbase + 1u], P2 = gPts[gbase + 2u], P3 = gPts[gbase + 3u];
        if (contains4(P0.xy, P1.xy, P2.xy, P3.xy, q0, bmin.xy, bmax.xy))
        {
            const float3 diff = float3(P0.x - qx, P0.y - qy, P0.z);
            const float nom = dot(faceNormal, diff);
            const float tval = nom / denom;
            if (cDist > (-tval))
            {
                pArea2 = true;
                if (active && (4u * count + lane) < 12u)
                    gTempBuff[lbase + 4u * count + lane] = float4(qx, qy, tval, -tval);
                count++;
            }
        }
    }
    gsArea2[lt] = pArea2 ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();

    const bool allArea2 = (gsArea2[gbase] & gsArea2[gbase + 1u] & gsArea2[gbase + 2u] & gsArea2[gbase + 3u]) != 0u;
    const bool done2 = allArea || allArea2;

    // 3. edge clip against the reference-face slab
    if (!done2)
    {
        const uint rStart = lane;
        const uint rEnd = (lane + 1u) & 3u;
        const float3 p0 = gPts[gbase + rStart];
        const float3 p1 = gPts[gbase + rEnd];
        const bool pPenStart = gsPen[gbase + rStart] != 0u;
        const bool pPenEnd = gsPen[gbase + rEnd] != 0u;
        const bool pAreaStart = gsArea[gbase + rStart] != 0u;
        const bool pAreaEnd = gsArea[gbase + rEnd] != 0u;

        if (pPenStart || pPenEnd)
        {
            const bool con0 = pPenStart && pAreaStart;
            const bool con1 = pPenEnd && pAreaEnd;
            if (!(con0 && con1))
            {
                const float3 p0p1 = p1 - p0;
                float tmn, tmx;
                if (segAABB(p0, p0p1, ext, negExt, tmn, tmx))
                {
                    if (!con0)
                    {
                        const float3 ip = p0 + p0p1 * tmn;
                        if (active && (4u * count + lane) < 12u)
                            gTempBuff[lbase + 4u * count + lane] = float4(ip, -ip.z);
                        count++;
                    }
                    if (!con1)
                    {
                        const float3 ip = p0 + p0p1 * tmx;
                        if (active && (4u * count + lane) < 12u)
                            gTempBuff[lbase + 4u * count + lane] = float4(ip, -ip.z);
                        count++;
                    }
                }
            }
        }
    }

    const uint effCount = active ? min(count, 3u) : 0u;

    // origContacts : this lane's up-to-3 local-space contacts.
    float4 origC[3];
    origC[0] = gTempBuff[lbase + lane];
    origC[1] = gTempBuff[lbase + lane + 4u];
    origC[2] = gTempBuff[lbase + lane + 8u];
    GroupMemoryBarrierWithGroupSync();

    // =======================================================================
    // Stage 5 : contact reduction (npBoxReduce).  Barriers unconditional;
    //           only data writes predicated (keeps the block barrier uniform).
    // =======================================================================
    gsCount[lt] = effCount;
    GroupMemoryBarrierWithGroupSync();
    const uint n0 = gsCount[gbase], n1 = gsCount[gbase + 1u], n2 = gsCount[gbase + 2u], n3 = gsCount[gbase + 3u];
    const uint total = n0 + n1 + n2 + n3;
    uint rc = n0;
    if (lane >= 1u) rc += n1;
    if (lane >= 2u) rc += n2;
    if (lane >= 3u) rc += n3;
    const uint exclStart = rc - effCount;
    const bool caseB = total > MAXP;

    // total <= 6 : compacted store, transformed to global.
    if (active && !caseB)
    {
        for (uint i = 0u; i < effCount; ++i)
        {
            const float3 gp = xform(nc0, nc1, nc2, np, origC[i].xyz);
            gTempBuff[lbase + exclStart + i] = float4(gp, origC[i].w);
        }
    }

    // total > 6 : 4-point manifold reduction.
    float dW = BIG; uint dI = 0u;
    for (uint i = 0u; i < effCount; ++i) { if (origC[i].w < dW) { dW = origC[i].w; dI = i; } }
    gsA[lt] = caseB ? dW : BIG;
    GroupMemoryBarrierWithGroupSync();
    {
        const float m = min(gsA[gbase], min(gsA[gbase + 1u], min(gsA[gbase + 2u], gsA[gbase + 3u])));
        const uint win = (gsA[gbase] == m) ? 0u : ((gsA[gbase + 1u] == m) ? 1u : ((gsA[gbase + 2u] == m) ? 2u : 3u));
        if (caseB && lane == win) gsLC[gbase + 0u] = origC[dI];
    }
    GroupMemoryBarrierWithGroupSync();
    const float3 deepestPt = gsLC[gbase + 0u].xyz;

    const float3 dpp = deepestPt - faceNormal * dot(deepestPt, faceNormal);
    float fW = -BIG; uint fI = 0u;
    for (uint i = 0u; i < effCount; ++i)
    {
        const float3 dvec = dpp - origC[i].xyz;
        const float ds = dot(dvec, dvec);
        if (ds > fW) { fW = ds; fI = i; }
    }
    gsA[lt] = caseB ? -fW : BIG;
    GroupMemoryBarrierWithGroupSync();
    {
        const float m = min(gsA[gbase], min(gsA[gbase + 1u], min(gsA[gbase + 2u], gsA[gbase + 3u])));
        const uint win = (gsA[gbase] == m) ? 0u : ((gsA[gbase + 1u] == m) ? 1u : ((gsA[gbase + 2u] == m) ? 2u : 3u));
        if (caseB && lane == win) gsLC[gbase + 1u] = origC[fI];
    }
    GroupMemoryBarrierWithGroupSync();
    const float3 furthestPt = gsLC[gbase + 1u].xyz;

    const float3 dir = cross(faceNormal, furthestPt - deepestPt);
    float mxV = -BIG, mnV = BIG; uint mxI = 0u, mnI = 0u;
    for (uint i = 0u; i < effCount; ++i)
    {
        const float dp = dot(dir, origC[i].xyz - deepestPt);
        if (dp > mxV) { mxV = dp; mxI = i; }
        if (dp < mnV) { mnV = dp; mnI = i; }
    }
    gsA[lt] = caseB ? mnV : BIG;
    gsB[lt] = caseB ? -mxV : BIG;
    GroupMemoryBarrierWithGroupSync();
    {
        const float ma = min(gsA[gbase], min(gsA[gbase + 1u], min(gsA[gbase + 2u], gsA[gbase + 3u])));
        const uint winA = (gsA[gbase] == ma) ? 0u : ((gsA[gbase + 1u] == ma) ? 1u : ((gsA[gbase + 2u] == ma) ? 2u : 3u));
        const float mb = min(gsB[gbase], min(gsB[gbase + 1u], min(gsB[gbase + 2u], gsB[gbase + 3u])));
        const uint winB = (gsB[gbase] == mb) ? 0u : ((gsB[gbase + 1u] == mb) ? 1u : ((gsB[gbase + 2u] == mb) ? 2u : 3u));
        if (caseB && lane == winA) gsLC[gbase + 2u] = origC[mnI];
        if (caseB && lane == winB) gsLC[gbase + 3u] = origC[mxI];
    }
    GroupMemoryBarrierWithGroupSync();

    // caseB : transform the 4 chosen contacts to global, one per lane.
    if (active && caseB)
    {
        const float4 lc = gsLC[gbase + lane];
        gTempBuff[lbase + lane] = float4(xform(nc0, nc1, nc2, np, lc.xyz), lc.w);
    }

    uint nbContacts = active ? (caseB ? 4u : total) : 0u;
    GroupMemoryBarrierWithGroupSync();

    // =======================================================================
    // Stage 6 : output tail.  Lane 3 reserves stream space and writes the patch;
    //           runs for EVERY in-range pair (nbContacts may be 0), matching the
    //           CUDA which always calls set/registerContactPatch.
    // =======================================================================
    uint contactByteOffset = INVALID;
    if (valid && lane == 3u)
    {
        // --- setContactPointAndForcePointers ---
        if (nbContacts != 0u)
        {
            uint cOff, fOff;
            InterlockedAdd(counters[0].contactsBytes, CONTACT_SIZE * nbContacts, cOff);
            InterlockedAdd(counters[0].forceAndIndiceBytes, FORCE_SIZE * nbContacts, fOff);
            contactByteOffset = cOff;
            uint forceOffset = fOff;
            if ((cOff + CONTACT_SIZE) > gParams.contactBytesLimit)
            {
                InterlockedOr(counters[0].overflowError, OVERFLOW_CONTACT);
                contactByteOffset = INVALID;
            }
            if ((fOff + FORCE_SIZE) > gParams.forceBytesLimit)
            {
                InterlockedOr(counters[0].overflowError, OVERFLOW_FORCE);
                forceOffset = INVALID;
            }
            if (contactByteOffset != INVALID)
            {
                cmOutputs[workIndex].contactForces = gParams.startContactForces + uint64_t(forceOffset);
                cmOutputs[workIndex].contactPoints = gParams.startContactPoints + uint64_t(contactByteOffset);
            }
            else
            {
                cmOutputs[workIndex].contactForces = 0uL;
                cmOutputs[workIndex].contactPoints = 0uL;
            }
        }

        // --- registerContactPatch ---
        const uint allflags = cmOutputs[workIndex].allflagsWord;
        const uint oldStatusFlags = (allflags >> 16u) & 0xffu;    // statusFlag byte
        uint statusFlags = oldStatusFlags & (~FLAG_TOUCH_KNOWN);
        if (nbContacts != 0u) statusFlags |= FLAG_HAS_TOUCH;
        else                  statusFlags |= FLAG_HAS_NO_TOUCH;

        const uint prevPatches = (allflags >> 8u) & 0xffu;        // current nbPatches byte
        const bool previouslyHadTouch = (oldStatusFlags & FLAG_HAS_TOUCH) != 0u;
        const bool prevTouchKnown = (oldStatusFlags & FLAG_TOUCH_KNOWN) != 0u;
        const uint numPatches = (nbContacts > 0u) ? 1u : 0u;
        const bool currentlyHasTouch = nbContacts > 0u;
        const bool change = (previouslyHadTouch != currentlyHasTouch) || (!prevTouchKnown);

        touchChangeFlags[workIndex] = change ? 1u : 0u;
        patchChangeFlags[workIndex] = (prevPatches != numPatches) ? 1u : 0u;

        // allflagsStart=0, nbPatches=numPatches, statusFlag=statusFlags, prevPatches=prevPatches
        cmOutputs[workIndex].allflagsWord =
            (prevPatches << 24u) | (statusFlags << 16u) | (numPatches << 8u);
        // set nbContacts (low 16), preserve flags (high 16)
        {
            const uint nf = cmOutputs[workIndex].nbContacts_flags;
            cmOutputs[workIndex].nbContacts_flags = (nf & 0xffff0000u) | (nbContacts & 0xffffu);
        }

        uint patchIndex = INVALID;
        if (nbContacts != 0u)
        {
            InterlockedAdd(counters[0].patchesBytes, PATCH_SIZE, patchIndex);
            if ((patchIndex + PATCH_SIZE) > gParams.patchBytesLimit)
            {
                InterlockedOr(counters[0].overflowError, OVERFLOW_PATCH);
                patchIndex = INVALID;

                statusFlags = (statusFlags & (~FLAG_TOUCH_KNOWN)) | FLAG_HAS_NO_TOUCH;
                cmOutputs[workIndex].allflagsWord = (prevPatches << 24u) | (statusFlags << 16u);
                const uint nf = cmOutputs[workIndex].nbContacts_flags;
                cmOutputs[workIndex].nbContacts_flags = (nf & 0xffff0000u);
                touchChangeFlags[workIndex] = previouslyHadTouch ? 1u : 0u;
                patchChangeFlags[workIndex] = (prevPatches != 0u) ? 1u : 0u;
            }
            else
            {
                cmOutputs[workIndex].contactPatches = gParams.startContactPatches + uint64_t(patchIndex);
            }
        }

        // --- insertIntoPatchStream ---
        if (patchIndex != INVALID)
        {
            const uint mi0 = shape0.materialIndex;
            const uint mi1 = shape1.materialIndex;
            const PxsMaterialData m0 = materials[mi0];
            const PxsMaterialData m1 = materials[mi1];

            const float r0f = m0.restitution, r1f = m1.restitution;
            const bool compliant0 = r0f < 0.0f, compliant1 = r1f < 0.0f;
            const bool exactlyOneCompliant = compliant0 != compliant1;
            const bool bothCompliant = compliant0 && compliant1;
            const uint m0flags = m0.packedFlags & 0xffffu;
            const uint m1flags = m1.packedFlags & 0xffffu;
            const uint m0fric = (m0.packedFlags >> 16u) & 0xffu;
            const uint m1fric = (m1.packedFlags >> 16u) & 0xffu;
            const uint m0rest = (m0.packedFlags >> 24u) & 0xffu;
            const uint m1rest = (m1.packedFlags >> 24u) & 0xffu;
            const uint m0damp = m0.packedDamp & 0xffu;
            const uint m1damp = m1.packedDamp & 0xffu;
            const bool cAcc0 = (m0flags & MAT_COMPLIANT_ACC_SPRING) != 0u;
            const bool cAcc1 = (m1flags & MAT_COMPLIANT_ACC_SPRING) != 0u;
            const bool exactlyOneAcc = cAcc0 != cAcc1;

            float rest, damping;
            if (bothCompliant && exactlyOneAcc)
                rest = cAcc0 ? r0f : r1f;
            else
            {
                const uint mode = exactlyOneCompliant ? MODE_MIN : max(m0rest, m1rest);
                const float flip = (bothCompliant && (mode == MODE_MULTIPLY)) ? -1.0f : 1.0f;
                rest = flip * combineScalars(r0f, r1f, mode);
            }
            if (bothCompliant && exactlyOneAcc)
                damping = cAcc0 ? m0.damping : m1.damping;
            else
            {
                const uint mode = exactlyOneCompliant ? MODE_MAX : max(m0damp, m1damp);
                damping = combineScalars(m0.damping, m1.damping, mode);
            }

            const uint combineFlags = m0flags | m1flags;
            float sta, dyn;
            uint matFlags;
            if ((combineFlags & MAT_DISABLE_FRICTION) == 0u)
            {
                const uint fm = max(m0fric, m1fric);
                const float dynF = combineScalars(m0.dynamicFriction, m1.dynamicFriction, fm);
                const float staF = combineScalars(m0.staticFriction, m1.staticFriction, fm);
                const float fDyn = max(dynF, 0.0f);
                dyn = fDyn;
                sta = (staF - fDyn) >= 0.0f ? staF : fDyn;
                matFlags = combineFlags;
            }
            else
            {
                matFlags = combineFlags | MAT_DISABLE_STRONG_FRICTION;
                dyn = 0.0f; sta = 0.0f;
            }

            Ptr<uint> patchBase = Ptr<uint>(gParams.patchStream + uint64_t(patchIndex));
            const uint one = asuint(1.0f);
            patchBase[0] = one; patchBase[1] = one; patchBase[2] = one; patchBase[3] = one;
            patchBase[4] = asuint(normal.x);
            patchBase[5] = asuint(normal.y);
            patchBase[6] = asuint(normal.z);
            patchBase[7] = asuint(rest);
            patchBase[8] = asuint(dyn);
            patchBase[9] = asuint(sta);
            patchBase[10] = asuint(damping);
            patchBase[11] = ((nbContacts & 0xffu) << 16u) | ((matFlags & 0xffu) << 24u);
            patchBase[12] = (mi0 & 0xffffu) << 16u;
            patchBase[13] = (mi1 & 0xffffu);
        }
    }

    // broadcast contactByteOffset from lane 3 to the group.
    gsCByte[lt] = contactByteOffset;
    GroupMemoryBarrierWithGroupSync();
    contactByteOffset = gsCByte[gbase + 3u];

    // write the contact points/penetrations to the contact stream.
    if (contactByteOffset != INVALID)
    {
        Ptr<float4> baseStream = Ptr<float4>(gParams.contactStream + uint64_t(contactByteOffset));
        for (uint i = lane; i < nbContacts; i += 4u)
            baseStream[i] = gTempBuff[lbase + i];
    }
}
