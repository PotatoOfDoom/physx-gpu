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
//
// ===========================================================================
// femClothMeshContactGenBda — cloth_meshContactGenLaunch
// (gpunarrowphase/src/CUDA/femClothPrimitives.cu:3749, core
// staticTriangleVsClothCollision at :3605, launch PxgNarrowphaseCore.cpp:6669 inside
// testSDKFemClothTrimesh, CLOTH_MESH_CG), on the engine BDA ABI.
//
// P8.17 step 2 of TWO — the consumer step 1's midphase had none.  One THREAD per midphase pair
// clips the cloth triangle against the mesh triangle, takes the deepest surviving point, and
// writes a rigid-vs-deformable contact.  Everything downstream is already mapped: the contacts
// go into the RIGID contact stream through PxgFEMContactWriter, so
// cloth_rigidContactPrepare -> query -> cloth_solveRigidClothCollision carry them.
//
// THE CORE IS A LIFT, NOT A NEW TRANSCRIPTION.  `fcStaticTriCloth.slang` already ports exactly
// this routine for the descriptor-set engine (its MODE 0), with its own fixture
// (testFcStaticTriCloth.cpp) covering the trim pipeline's sign-change bookkeeping, the
// deepest-point averaging band, the filter skip and the 32-bit write lane against a CPU
// transcription.  The clipping code below is that port, verbatim; what is NEW here is the BDA
// plumbing around it — the 192-byte push blob, the cooked-blob walk, the PxgFEMCloth descriptor
// reads and the pair-stack iteration, all of which `fcStaticTriCloth`'s header explicitly lists
// as "engine glue not ported".  Two copies of the clipping pipeline is a cost this tranche pays
// knowingly: the descriptor-set shader is a unit-test harness with a case-array interface and
// cannot be imported by a push-constant kernel without dragging its bindings along.
//
// THE ARG PACK IS **NOT** sb_meshContactGenLaunch's, even though both are 192 bytes and both end
// in an 80-byte writer.  The middle differs:
//
//   softbody: ..., shapeToRigidRemapTable*, stackPtr*, midphasePairsNum*, PxU32 stackSizeBytes,
//             filterPairs*, PxU32 nbFilterPairs, stackSizeNeeded*, writer
//   cloth:    ..., shapeToRigidRemapTable*, PxU32 stackSizeBytes, stackPtr*, midphasePairsNum*,
//             filterPairs*, PxU32 nbFilterPairs, stackSizeNeeded*, writer
//
// `stackSizeBytes` moves from @80 to @64 and the two stack pointers shift with it.  Reading one
// pack with the other's layout hands the kernel a byte count where it expects a device pointer.
//
// WHAT DIFFERS FROM THE SOFTBODY SIBLING'S CONTACT GEN, beyond the pack:
//   1. no GJK/EPA.  The cloth path is a POLYGON CLIP: the cloth triangle is trimmed by the mesh
//      triangle's three edge planes and its two +-cDistance slabs, and the contact is the deepest
//      surviving corner (or the average of the corners within 0.01*cDistance of it);
//   2. the transform runs the OTHER WAY.  Step 1's midphase brings the cloth into the mesh's
//      vertex space; this kernel brings the MESH TRIANGLE into the CLOTH's local frame --
//      `vertex2Shape` (which MULTIPLIES by the scale) then
//      `clothTransform.transformInv(trimeshToWorld)`.  The cloth positions are used untransformed;
//   3. the filter skip needs ALL THREE cloth vertices to be filtered before the triangle is
//      dropped, where the softbody path filters on a single element;
//   4. the write is `writeRigidVsDeformableContact32` — barycentrics ARE written, the 32-bit
//      `contactSortedByX` lane carries the low word of the rigid node index, and the u64
//      `contactByX`/`tempContactByX` keys are deliberately NOT touched.
//
// *** THE GRID.  READ THIS BEFORE CHANGING numthreads. ***
// The engine launches grid (2048, 1, 1) with block (128, 1, 1) and grid-strides:
//
//     globalThreadIdx = threadIdx.x + blockIdx.x * blockDim.x;   // [0, 262144)
//     for (i = globalThreadIdx; i < l; i += gridDim.x * blockDim.x)   // stride 262144
//
// with `l = ceil(numPairs / 32) * 32` — rounded UP to a warp multiple so every lane of every
// partially-filled warp still reaches the ballot.  The bridge drops blockDim, so a
// [numthreads(32,1,1)] group is ONE warp and the block's FOUR warps are walked serially in a
// group-uniform `sub` loop.  The loop is driven on the warp BASE rather than on the per-lane
// index: `base` and the stride are both multiples of 32 and `l` is too, so `base + lane < l`
// is the same for every lane and the ballot's barriers stay uniform.  Deriving the stride from
// the group count instead of the CUDA 262144 would double-visit or skip pairs.
//
// DEVIATIONS from CUDA (recipe policy, each preserving the emitted contact SET and the atomic
// total): globalScanExclusiveSingleWarp -> a groupshared InterlockedOr ballot plus
// countbits(mask & lanemask_lt) and a single lane-0 reservation; every descriptor read goes
// through Ptr<> instead of a cached __shared__ block.
//
// NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the plane distances, the trim pipeline's
// lerps and the barycentric solve are mul-feeding-add chains compared bit-exactly against the
// -ffp-contract=off CPU reference.
// ===========================================================================
module femClothMeshContactGenBda;

import pxgSlangCommon;         // quatRotate / quatRotateInv / PxgTransform28
import pxgSlangFEMClothDesc;   // PxgFEMClothGpu

// PxgSimulationCoreDesc.h:41 — PxEncodeClothIndex(clothId, elem) = (clothId << 20) | elem.
static const uint PXG_BITSHIFT_ELEMENT_ID = 20u;
static const uint COLLISION_FLAG_MASK = 1u << 31;
static const float PX_NORMALIZATION_EPSILON = 1e-20f;

// Gu::BV32DataPacked stride, for the cooked-blob walk.
static const uint64_t BV32_STRIDE = 1168UL;

// The engine's launch shape (PxgNarrowphaseCore.cpp:6694).
static const uint CLOTH_CG_WARPS_PER_BLOCK = 4u;    // blockDim.x 128 / WARP_SIZE 32
static const uint CLOTH_CG_STRIDE = 2048u * 128u;   // gridDim.x * blockDim.x

// ---------------------------------------------------------------------------
// Push-constant blob — BYTE-IDENTICAL to the CUDA pack (PxgNarrowphaseCore.cpp:6671):
// { PxReal toleranceLength, cmInputs*, transformCache*, contactDistance*, restDistance*,
//   gpuShapes*, femClothes*, shapeToRigidRemapTable*, PxU32 stackSizeBytes, stackPtr*,
//   midphasePairsNum*, filterPairs*, PxU32 nbFilterPairs, stackSizeNeededOnDevice*,
//   PxgFEMContactWriter writer }.
// 4 @0, seven pointers @8..@56, PxU32 @64, a pointer realigned to @72, @80, @88, PxU32 @96, a
// pointer at @104, then the 80-byte writer at @112 -> 192-byte blob.
// ABI row {4,8,8,8,8,8,8,8,4,8,8,8,4,8,80}.
// ---------------------------------------------------------------------------
struct ClothMeshCgParams
{
    float toleranceLength;          // @0   (ABI parity only — never read)
    uint _pad4;                     // @4
    uint64_t cmInputs;              // @8
    uint64_t transformCache;        // @16
    uint64_t contactDistance;       // @24
    uint64_t restDistance;          // @32
    uint64_t gpuShapes;             // @40
    uint64_t femClothes;            // @48
    uint64_t shapeToRigidRemapTable;// @56  PxNodeIndex* (u64 per transform-cache slot)
    uint stackSizeBytes;            // @64
    uint _pad68;                    // @68
    uint64_t stackPtr;              // @72  uint4 pairs from step 1
    uint64_t midphasePairsNum;      // @80
    uint64_t filterPairs;           // @88  PxgRigidFilterPair*
    uint nbFilterPairs;             // @96
    uint _pad100;                   // @100
    uint64_t stackSizeNeededOnDevice; // @104
    // --- PxgFEMContactWriter, by value (PxgFEMCore.h:361) ---
    uint64_t w_outPoint;            // @112 float4*
    uint64_t w_outNormalPen;        // @120 float4*
    uint64_t w_outBarycentric;      // @128 float4*
    uint64_t w_outContactInfo;      // @136 PxgFemOtherContactInfo* (16 B stride)
    uint64_t w_totalContactCount;   // @144 PxU32*
    uint64_t w_contactByX;          // @152 PxU64*  (NOT written by this path)
    uint64_t w_tempContactByX;      // @160 PxU32*  (NOT written by this path)
    uint64_t w_contactIndexSortedByX; // @168 PxU32*
    uint64_t w_contactSortedByX;    // @176 PxU32*
    uint w_maxNumContacts;          // @184 PxU32
    uint _pad188;                   // @188 -> 192
}

[[vk::push_constant]] ConstantBuffer<ClothMeshCgParams> gCg;

// PxgContactManagerInput (PxgContactManager.h): four PxU32.
struct PxgContactManagerInput { uint shapeRef0, shapeRef1, transformCacheRef0, transformCacheRef1; }

// PxgShape (PxgConvexConvexShape.h, 48 B) comes from pxgSlangCommon rather than a local copy —
// this shader already imports that module for quatRotate / PxgTransform28, and a second
// declaration is an ambiguous reference rather than a shadow.

// PxgRigidFilterPair (PxgSimulationCoreDesc.h:167, 32 B): index0 u64 @0, index1 u64 @8,
// index2 u32 @16, tail padding.
struct PairR { uint2 index0; uint2 index1; uint index2; uint pad0, pad1, pad2; }

// ---------------------------------------------------------------------------
// Small geometry helpers (fcStaticTriCloth.slang, verbatim).
// ---------------------------------------------------------------------------
float3 cgGetNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}
// PxPlane(point, normal): n, d = -p.n ; distance(p) = n.p + d
float4 makePlane(float3 p, float3 n) { return float4(n, -dot(p, n)); }
float planeDistance(float4 pl, float3 p) { return dot(pl.xyz, p) + pl.w; }

// vertex2Shape(v, scale, rot) = rot.rotateInv(rot.rotate(v) * scale)  (cudaNpCommon.h:270).
// MULTIPLIES by the scale — the inverse of step 1's shape2Vertex, which divides.
float3 vertex2Shape(float3 v, float3 scale, float4 rot)
{
    return quatRotateInv(rot, quatRotate(rot, v) * scale);
}
float3 transformPt(float4 q, float3 p, float3 v) { return quatRotate(q, v) + p; }

// femClothPrimitives.cu computeBarycentric (the bool result is unused at this call site).
void computeBarycentric(float3 a, float3 b, float3 c, float3 p, out float4 barycentric)
{
    const float3 v0 = b - a;
    const float3 v1 = c - a;
    const float3 v2 = p - a;
    const float d00 = dot(v0, v0);
    const float d01 = dot(v0, v1);
    const float d11 = dot(v1, v1);
    const float d20 = dot(v2, v0);
    const float d21 = dot(v2, v1);
    const float denom = d00 * d11 - d01 * d01;
    const float v = (d11 * d20 - d01 * d21) / denom;
    const float w = (d00 * d21 - d01 * d20) / denom;
    const float u = 1.0f - v - w;
    barycentric = float4(u, v, w, 0.0f);
}

// ---------------------------------------------------------------------------
// find(PxgRigidFilterPair*) (deformableElementFilter.cuh:67) — the binary search with the
// ignoreIndex2 early accept, as softBodyMeshContactGenBda carries it.
// ---------------------------------------------------------------------------
bool u64Lt(uint2 a, uint2 b) { return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x); }
bool u64Eq(uint2 a, uint2 b) { return a.x == b.x && a.y == b.y; }

bool findRigidPair(uint2 q0, uint2 q1)
{
    // A null filter list with nbFilterPairs == 0 is the common case, and a zero-trip
    // `while (l < r)` is NOT enough on llvmpipe: it speculates the first iteration's load and
    // faults at address 0.  SELECT THE POINTER, not the loaded value — gCg.femClothes is
    // dereferenced by every invocation anyway, so a speculated read there is harmless, and
    // `r = 0` keeps the RESULT exactly CUDA's.  Diagnosed at length in
    // softBodySelfCollisionBda.slang.
    const bool havePairs = (gCg.filterPairs != 0UL) && (gCg.nbFilterPairs != 0u);
    Ptr<PairR> pairs = Ptr<PairR>(havePairs ? gCg.filterPairs : gCg.femClothes);
    uint l = 0u, r = havePairs ? gCg.nbFilterPairs : 0u;
    while (l < r)
    {
        const uint mid = (l + r) / 2u;
        const PairR p = pairs[mid];
        int cmp = 0;
        if (u64Lt(q0, p.index0)) cmp = -1;
        else if (u64Lt(p.index0, q0)) cmp = 1;
        else if (u64Lt(q1, p.index1)) cmp = -1;
        else if (u64Lt(p.index1, q1)) cmp = 1;
        else if (0u < p.index2) cmp = -1;   // the query's index2 is 0
        if (cmp == 0 || (u64Eq(q0, p.index0) && u64Eq(q1, p.index1)))
            return true;
        if (cmp < 0) r = mid; else l = mid + 1u;
    }
    return false;
}

// ---------------------------------------------------------------------------
// The polygon trim pipeline (femClothPrimitives.cu:3436-3641), lifted from
// fcStaticTriCloth.slang where it is already fixture-verified against a CPU transcription.
// ---------------------------------------------------------------------------

// Only correct when a and b straddle the plane (one may lie exactly on it).
float3 intersectionPoint(float4 plane, float3 a, float3 b)
{
    const float distA = abs(planeDistance(plane, a));
    const float distB = abs(planeDistance(plane, b));
    const float f = distA / (distA + distB);
    return (1.0f - f) * a + f * b;
}

void insertPt(inout float3 arr[8], int arrCount, int index, float3 element)
{
    for (int i = arrCount; i > index; --i)
        arr[i] = arr[i - 1];
    arr[index] = element;
}

// In-place convex-loop clip: points with distance > 0 (STRICT) are removed.
int trimLoop(float4 plane, inout float3 loop[8], int loopCount)
{
    if (loopCount < 3)
        return loopCount;

    float3 intersectionA = float3(0, 0, 0);
    int changeA = -1;
    float3 intersectionB = float3(0, 0, 0);
    int changeB = -1;

    bool keep = false;

    bool prevOutside = planeDistance(plane, loop[0]) > 0.0f;
    for (int i = 0; i < loopCount; ++i)
    {
        const bool outside = planeDistance(plane, loop[(i + 1) % loopCount]) > 0.0f;
        if (outside != prevOutside)
        {
            const float3 inter = intersectionPoint(plane, loop[i], loop[(i + 1) % loopCount]);
            if (changeA < 0)
            {
                changeA = i;
                keep = !prevOutside;
                intersectionA = inter;
            }
            else
            {
                changeB = i;
                intersectionB = inter;
            }
        }
        prevOutside = outside;
    }

    if (changeA >= 0 && changeB >= 0)
    {
        int loopIndexer = -1;
        for (int i = 0; i < loopCount; ++i)
        {
            if (keep)
            {
                ++loopIndexer;
                loop[loopIndexer] = loop[i];
            }
            if (i == changeA || i == changeB)
            {
                const float3 pt = (i == changeA) ? intersectionA : intersectionB;
                if (loopIndexer == i && !keep)
                {
                    ++loopIndexer;
                    insertPt(loop, loopCount, loopIndexer, pt);
                    ++loopCount;
                    ++i;
                    ++changeB;
                }
                else
                {
                    ++loopIndexer;
                    loop[loopIndexer] = pt;
                }
                keep = !keep;
            }
        }
        loopCount = loopIndexer + 1;
    }
    else if (prevOutside)
        loopCount = 0;   // no crossing and every point outside

    return loopCount;
}

float4 createEdgePlane(float3 triNormal, float3 edgeStart, float3 edgeEnd)
{
    const float3 n = cgGetNormalized(cross(edgeEnd - edgeStart, triNormal));
    return makePlane(edgeStart, n);
}

// The 5-plane, 8-element variant: three edge planes plus the two +-distance slabs.
int trimTriangleByTriangle(inout float3 poly[8], int polyLength, float3 triA, float3 triB, float3 triC,
                           float maxDistanceLimitPosNormalDir, float maxDistanceLimitNegNormalDir)
{
    float3 n = cross(triB - triA, triC - triA);
    // PxVec3::normalizeSafe: n is left UNCHANGED below the epsilon, unlike getNormalized which
    // returns zero.  Only distinguishable for 0 < |cross| < 1e-20, a subnormal window that
    // cannot survive the vertex transform chain — kept because it is what the .cu writes.
    const float mag = sqrt(dot(n, n));
    if (mag >= PX_NORMALIZATION_EPSILON)
        n = n * (1.0f / mag);

    polyLength = trimLoop(createEdgePlane(n, triA, triB), poly, polyLength);
    polyLength = trimLoop(createEdgePlane(n, triB, triC), poly, polyLength);
    polyLength = trimLoop(createEdgePlane(n, triC, triA), poly, polyLength);

    polyLength = trimLoop(makePlane(triA + maxDistanceLimitPosNormalDir * n, n), poly, polyLength);
    polyLength = trimLoop(makePlane(triA - maxDistanceLimitNegNormalDir * n, -n), poly, polyLength);

    return polyLength;
}

// Running min with a +-tolerance averaging band: a strictly deeper point RESTARTS the average,
// a near-tie ACCUMULATES into it without moving minDist.  A fully clipped-away polygon leaves
// minDist at 1e9, which fails the `sep < cDistance` gate (the 0/0 in the average is unused).
float3 getDeepestPointSimplified(float3 poly[8], int polyLength, float4 plane, float tolerance,
                                 out float minDist)
{
    minDist = 1000000000.0f;
    float3 deepestPoint = float3(0, 0, 0);
    int count = 0;
    for (int i = 0; i < polyLength; ++i)
    {
        const float d = planeDistance(plane, poly[i]);
        if (d < minDist - tolerance)
        {
            minDist = d;
            deepestPoint = poly[i];
            count = 1;
        }
        else if (d < minDist + tolerance)
        {
            deepestPoint += poly[i];
            count++;
        }
    }
    deepestPoint /= float(count);
    return deepestPoint;
}

struct FemContact
{
    float3 position;
    float3 normal;
    float separation;
    float4 barycentric;
}

// staticTriangleVsClothCollision (femClothPrimitives.cu:3605).
bool staticTriangleVsClothCollision(float3 localA0, float3 localB0, float3 localC0,
                                    float3 a1, float3 b1, float3 c1,
                                    float4 clothQ, float3 clothP, float cDistance,
                                    out FemContact contact)
{
    float3 convexPoly[8];
    convexPoly[0] = a1;
    convexPoly[1] = b1;
    convexPoly[2] = c1;
    for (int k = 3; k < 8; ++k)
        convexPoly[k] = float3(0, 0, 0);

    const float3 fixNormal = cgGetNormalized(cross(localB0 - localA0, localC0 - localA0));
    const float4 staticTrianglePlane = makePlane(localA0, fixNormal);

    const int numPolyCorners =
        trimTriangleByTriangle(convexPoly, 3, localA0, localB0, localC0, cDistance, cDistance);

    float sep;
    const float3 deepestPoint = getDeepestPointSimplified(convexPoly, numPolyCorners,
                                                          staticTrianglePlane, 0.01f * cDistance, sep);

    const bool hasContact = sep < cDistance;

    contact.position = float3(0, 0, 0);
    contact.normal = float3(0, 0, 0);
    contact.separation = 0.0f;
    contact.barycentric = float4(0, 0, 0, 0);
    if (hasContact)
    {
        computeBarycentric(a1, b1, c1, deepestPoint, contact.barycentric);
        contact.position = transformPt(clothQ, clothP, deepestPoint);
        contact.normal = quatRotate(clothQ, fixNormal);
        contact.separation = sep;
    }
    return hasContact;
}

// PxgFEMContactWriter::writeRigidVsDeformableContact32 (PxgFEMCore.h:501).  The 32-bit sort
// lane only: `contactSortedByX` takes the LOW WORD of the rigid node index and the u64
// `contactByX` / `tempContactByX` keys are deliberately left alone — this path's radix sort
// does not read them.  setRigidMaterialIndex(mat) followed by markInCollision(false) collapses
// to a single store of (mat & 0x7fffffff).
void writeRigidVsDeformableContact32(uint index, float4 contact, float4 normalPen, float4 bary,
                                     uint2 pairInd0, uint pairInd1, uint rigidMaterialIndex,
                                     uint contactSortedByRigid)
{
    if (index >= gCg.w_maxNumContacts)
        return;
    Ptr<float4>(gCg.w_outPoint)[index] = contact;
    Ptr<float4>(gCg.w_outNormalPen)[index] = normalPen;
    Ptr<float4>(gCg.w_outBarycentric)[index] = bary;
    // PxgFemOtherContactInfo (16 B): pairInd0 u64 @0, pairInd1 u32 @8, rigidMatInd @12.
    const uint64_t ci = gCg.w_outContactInfo + uint64_t(index) * 16UL;
    Ptr<uint2>(ci)[0] = pairInd0;
    Ptr<uint>(ci + 8UL)[0] = pairInd1;
    Ptr<uint>(ci + 12UL)[0] = rigidMaterialIndex & ~COLLISION_FLAG_MASK;
    Ptr<uint>(gCg.w_contactSortedByX)[index] = contactSortedByRigid;
    Ptr<uint>(gCg.w_contactIndexSortedByX)[index] = index;
}

// ---------------------------------------------------------------------------
// Warp emulation scratch: one group == one warp.
// ---------------------------------------------------------------------------
groupshared uint sCgBallot;      // __ballot_sync mask
groupshared uint sCgScanStart;   // the single lane-0 reservation, broadcast

uint cgBallot(bool pred, uint lane)
{
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) sCgBallot = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sCgBallot, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sCgBallot;
}

// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void cloth_meshContactGenLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    const uint stackSize = gCg.stackSizeBytes / 16u;
    const uint numMidphasePairsFound = Ptr<uint>(gCg.midphasePairsNum)[0];
    const uint numPairs = min(numMidphasePairsFound, stackSize);
    // l = ceil(numPairs / WARP_SIZE) * WARP_SIZE — every lane of a partial warp still ballots.
    const uint l = ((numPairs + 31u) / 32u) * 32u;

    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gCg.femClothes);
    Ptr<float> contactDistance = Ptr<float>(gCg.contactDistance);
    Ptr<float> restDistance = Ptr<float>(gCg.restDistance);
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gCg.gpuShapes);

    // The block's four warps, walked serially (see the header on the grid).
    for (uint sub = 0u; sub < CLOTH_CG_WARPS_PER_BLOCK; ++sub)
    {
    const uint warpBase = gid.x * (CLOTH_CG_WARPS_PER_BLOCK * 32u) + sub * 32u;

    // `base`, the stride and `l` are all multiples of 32, so this condition is the same for
    // every lane and the ballot's barriers below stay group-uniform.
    for (uint base = warpBase; base < l; base += CLOTH_CG_STRIDE)
    {
        const uint i = base + lane;

        bool hasContact = false;
        uint cmIdx = 0u;
        uint clothId = 0u;
        uint clothTriangleIdx = 0u;
        uint2 rigidId = uint2(0u, 0u);
        uint rigidMaterialIndex = 0xFFFFFFFFu;
        FemContact c;
        c.position = float3(0, 0, 0);
        c.normal = float3(0, 0, 0);
        c.separation = 0.0f;
        c.barycentric = float4(0, 0, 0, 0);

        if (i < numPairs)
        {
            const uint4 curPair = Ptr<uint4>(gCg.stackPtr)[i];
            cmIdx = curPair.x;
            clothTriangleIdx = curPair.y;              // the CLOTH triangle
            const uint meshTriangleIdx = curPair.z;    // the MESH triangle

            const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(gCg.cmInputs)[cmIdx];
            const PxgShape shape0 = shapes[cin.shapeRef0];
            const PxgShape shape1 = shapes[cin.shapeRef1];

            const float cDistance = contactDistance[cin.transformCacheRef0] +
                                    contactDistance[cin.transformCacheRef1];

            // Both halves are eTRIANGLEMESH; the CLOTH is the one carrying a body id.
            PxgShape clothShape = shape0, trimeshShape = shape1;
            uint clothCacheRef = cin.transformCacheRef0, trimeshCacheRef = cin.transformCacheRef1;
            if (shape0.particleOrSoftbodyId == 0xffffffffu)
            {
                clothShape = shape1;   trimeshShape = shape0;
                clothCacheRef = cin.transformCacheRef1; trimeshCacheRef = cin.transformCacheRef0;
            }
            rigidMaterialIndex = trimeshShape.materialIndex;
            rigidId = Ptr<uint2>(gCg.shapeToRigidRemapTable)[trimeshCacheRef];   // PxNodeIndex
            clothId = clothShape.particleOrSoftbodyId;

            const uint4 clothVertInd = Ptr<uint4>(clothes[clothId].mTriangleVertexIndices)[clothTriangleIdx];

            // The triangle is dropped only when ALL THREE of its cloth vertices are filtered.
            const uint clothMask0 = (clothId << PXG_BITSHIFT_ELEMENT_ID) | clothVertInd.x;
            const uint clothMask1 = (clothId << PXG_BITSHIFT_ELEMENT_ID) | clothVertInd.y;
            const uint clothMask2 = (clothId << PXG_BITSHIFT_ELEMENT_ID) | clothVertInd.z;
            const bool skipTriangle = findRigidPair(rigidId, uint2(clothMask0, 0u)) &&
                                      findRigidPair(rigidId, uint2(clothMask1, 0u)) &&
                                      findRigidPair(rigidId, uint2(clothMask2, 0u));

            if (!skipTriangle)
            {
                // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.
                const uint64_t xfTri = gCg.transformCache + uint64_t(trimeshCacheRef) * 32UL;
                const float4 triQ = Ptr<float4>(xfTri)[0];
                const float3 triP = float3(Ptr<float>(xfTri + 16UL)[0],
                                           Ptr<float>(xfTri + 20UL)[0],
                                           Ptr<float>(xfTri + 24UL)[0]);
                const uint64_t xfCloth = gCg.transformCache + uint64_t(clothCacheRef) * 32UL;
                const float4 clothQ = Ptr<float4>(xfCloth)[0];
                const float3 clothP = float3(Ptr<float>(xfCloth + 16UL)[0],
                                             Ptr<float>(xfCloth + 20UL)[0],
                                             Ptr<float>(xfCloth + 24UL)[0]);

                // The cooked triangle-mesh blob (readTriangleMesh, triangleMesh.cuh:120).
                // `.w` is nbPackedNodes and sizes the node ARRAY — `.z` is a different count.
                const uint64_t meshPtr = uint64_t(trimeshShape.hullOrMeshOffset.x) |
                                         (uint64_t(trimeshShape.hullOrMeshOffset.y) << 32);
                const uint4 meshHeader = Ptr<uint4>(meshPtr)[0];
                const uint64_t vertsAddr = meshPtr + 16UL + uint64_t(meshHeader.w) * BV32_STRIDE;
                const uint64_t indicesAddr = vertsAddr + uint64_t(meshHeader.x) * 16UL;

                const uint4 triVertInd = Ptr<uint4>(indicesAddr)[meshTriangleIdx];
                const float3 triV0 = Ptr<float4>(vertsAddr)[triVertInd.x].xyz;
                const float3 triV1 = Ptr<float4>(vertsAddr)[triVertInd.y].xyz;
                const float3 triV2 = Ptr<float4>(vertsAddr)[triVertInd.z].xyz;

                const float3 triScale = trimeshShape.scale();
                const float4 triScaleRot = trimeshShape.scaleRotation();

                // trimeshToClothTransform = clothTransform.transformInv(trimeshToWorld)
                const PxgTransform28 clothT = makeTransform28(clothQ, clothP);
                const PxgTransform28 meshT = makeTransform28(triQ, triP);
                const PxgTransform28 meshToCloth = transformInvTransform(clothT, meshT);

                const float3 localA0 = transformPt(meshToCloth.q(), meshToCloth.p(),
                                                   vertex2Shape(triV0, triScale, triScaleRot));
                const float3 localB0 = transformPt(meshToCloth.q(), meshToCloth.p(),
                                                   vertex2Shape(triV1, triScale, triScaleRot));
                const float3 localC0 = transformPt(meshToCloth.q(), meshToCloth.p(),
                                                   vertex2Shape(triV2, triScale, triScaleRot));

                Ptr<float4> clothPos = Ptr<float4>(clothes[clothId].mPosition_InvMass);
                const float3 a1 = clothPos[clothVertInd.x].xyz;
                const float3 b1 = clothPos[clothVertInd.y].xyz;
                const float3 c1 = clothPos[clothVertInd.z].xyz;

                hasContact = staticTriangleVsClothCollision(localA0, localB0, localC0, a1, b1, c1,
                                                            clothQ, clothP, cDistance, c);
            }
        }

        // globalScanExclusiveSingleWarp(hasContact, writer.totalContactCount).
        const uint mask = cgBallot(hasContact, lane);
        const uint offset = countbits(mask & ((1u << lane) - 1u));
        const uint validCount = countbits(mask);
        if (lane == 0u)
        {
            uint start = 0u;
            if (validCount > 0u)
                InterlockedAdd(Ptr<uint>(gCg.w_totalContactCount)[0], validCount, start);
            sCgScanStart = start;
        }
        GroupMemoryBarrierWithGroupSync();
        const uint index = sCgScanStart + offset;

        if (hasContact)
        {
            writeRigidVsDeformableContact32(index,
                                            float4(c.position, restDistance[cmIdx]),
                                            float4(c.normal, c.separation),
                                            c.barycentric,
                                            rigidId,
                                            (clothId << PXG_BITSHIFT_ELEMENT_ID) | clothTriangleIdx,
                                            rigidMaterialIndex,
                                            rigidId.x);   // PxU32(rigidId.getInd())
        }
    }
    }

    // The stack high-water mark, so the engine can grow the shared midphase stack.
    if (gid.x == 0u && lane == 0u)
        InterlockedMax(Ptr<uint>(gCg.stackSizeNeededOnDevice)[0], numMidphasePairsFound * 16u);
}
