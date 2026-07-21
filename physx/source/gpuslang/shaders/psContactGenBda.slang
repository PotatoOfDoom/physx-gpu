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

// psContactGenBda.slang
//
// P6.4a-2 — ps_primitivesCollisionLaunch (the ONE heavy particle-vs-primitive
// contact GENERATOR) on the Vulkan engine ABI (BDA).  Ported from
// gpunarrowphase/src/CUDA/cudaParticleSystem.cu:1189 (__global__) whose device
// core is particlePrimitiveCollision (:268) + psPrimitivesCollision (:377);
// launch site PxgNarrowphaseCore.cpp:3225 (PS_PRIMITIVES_COLLISION), the 15-arg
// PxCudaKernelParam pack { isTGS(bool), numTests(u32), toleranceLength(f32),
// cmInputs*, gpuShapes*, transformCache*, boundsd*, contactDistance*,
// restDistanced*, materials*, tempCellsHistogramd* (the per-pair cell run-sum
// = "startIndices"), totalPairsd*, particleSystemsd*, shapeToRigidRemapTabled*,
// writer (PxgParticleContactWriter BY VALUE) }.
//
// Reused cores / mirrors (nothing re-derived):
//   - the particlePrimitiveCollision contact core (plane / sphere / capsule /
//     box; eCONVEXCORE deferred) and the PxgParticleContactWriter::writeContact
//     staging fan-out are TRANSCRIBED from the tranche-89 harness
//     psPrimCollide.slang (which itself ports particleCollision.cuh).  The ONLY
//     change: the switch keys on the REAL PxGeometryType (eSPHERE 0 / ePLANE 1 /
//     eCAPSULE 2 / eBOX 3) read straight from PxgShape.type, not the harness's
//     pre-remapped 0=plane encoding.
//   - PxgParticleSystemGpu descriptor mirror: pxgSlangParticleDesc (P6.0) — all
//     fields walked here (mSortedOriginPos_InvMass @392, mSortedPositions_InvMass
//     @24, mCellStart @168, mCellEnd @176, mSortedToUnsortedMapping @56,
//     mParticleBufferRunsum @408, mParticleSimBuffers @400,
//     mCommonData_NumParticleBuffers @132, mData_Flags @312, grid fields @88..)
//     already exist; and PxgParticleSimBufferGpu (mFilterPairs @40,
//     mNumFilterPairs @64) for the rigid filter list.
//   - PxgContactManagerInput (16B) / PxgShape (48B) / PxsCachedTransform (32B) /
//     PxgRigidFilterPair (32B) mirrors transcribed locally (byte layouts pinned
//     by static_asserts against the REAL engine structs in testPsContactGen.cpp),
//     matching the psBoundsBda.slang precedent (keeps the rigid-common module out
//     of the particle-descriptor TU).
//   - PxgCellData::update (workIndex -> pair via binarySearch of the cell
//     run-sum, LoadShapePair<ePARTICLESYSTEM>, combine+fattenFast +
//     calcGridHashInBounds) and the gridCal helpers are transcribed from
//     cudaParticleSystem.cu:1138 / gridCal.slang.
//   - the (particleSystem, rigidId, compressedParticleId) filter find() is the
//     deformableElementFilter.cuh:118 localize-into-sim-buffer + binary search
//     (same shape as psPrimCollide.slang's findPs/findRange/findRigid).
//
// Deviations from CUDA (recipe policy) + llvmpipe guards:
//   - NO 32-lane-warp assumption.  The CUDA caller's grid-stride ballot/warpScan
//     compaction of non-empty work items into shared batches is a pure SCHEDULING
//     optimisation — the SET of emitted contacts and the running numTotalContacts
//     total are identical whether or not work is compacted (contact-buffer
//     indices are already atomicAdd-order-nondeterministic, and the six radix
//     staging arrays are sorted downstream).  It is DROPPED: every work item is
//     processed once, in place, by the thread that owns it (same as the harness
//     core).
//   - the per-warp `range = warpReduction<Max>` that makes the inner ballot
//     group-uniform is lowered to a full-BLOCK groupshared max reduction
//     (blockMaxRange).  Iterating to the block max (>= the warp max) only adds
//     idle rounds — extra `ind` past a lane's own cell population fail the
//     `particleIndex < endIndex` guard, intersect stays false, so the contact set
//     is unchanged.  A block-uniform loop bound is what lets the ballot barriers
//     below be safe without a hardware-subgroup width.
//   - the per-warp contact aggregation (ballot + popc prefix + ONE lane-0
//     atomicAdd) is kept but done per 32-lane LOGICAL tile through groupshared
//     (sMask/sStart, WARPS_PER_BLOCK of them) with full-block barriers; output-
//     equivalent to the CUDA __ballot_sync/__shfl_sync warp path.
//   - descriptor / cmInput / shape / transform / bound / filter reads are direct
//     Ptr<> loads (BDA nested-pointer walk) instead of the CUDA blockCopy-to-
//     shared + LoadShapePair warp-broadcast; semantically identical read-fanout.
//   - every speculative BDA Ptr<> load sits behind a uniform-scalar guard
//     (workIndex < totalComparison, startIndex != EMPTY_CELL, particleIndex <
//     endIndex) so nothing hoists out of a zero-trip cell loop.
//   - materials* and toleranceLength are unused by the contact GENERATOR (they
//     feed the later solve); present for ABI parity only.  eCONVEXCORE is deferred
//     (no particle cloud vs convex-core scene), matching the harness.

import pxgSlangParticleDesc;

// PxgParticleSystemKernelBlockDim::PS_COLLISION (256) /
// PxgParticleSystemKernelGridDim::PS_COLLISION (1024) — the fixed CUDA launch
// geometry (PxgParticleSystemCoreKernelIndices.h:42/:59).  The Vulkan dispatch
// grid must match PS_GRID so the grid-stride below covers all work items.
static const uint PS_BLOCK = 256u;
static const uint PS_GRID = 1024u;
static const uint WARP = 32u;
static const uint WARPS_PER_BLOCK = PS_BLOCK / WARP; // 8

static const uint EMPTY_CELL = 0xffffffffu;
static const float PX_MAX_F32 = 3.4028235e38f;
static const float PX_EPS_REAL = 1.192092896e-07f;

// PxGeometryType (PxGeometry.h): eSPHERE 0, ePLANE 1, eCAPSULE 2, eBOX 3,
// eCONVEXCORE 4, ... ePARTICLESYSTEM 6.
static const uint GEOM_SPHERE = 0u;
static const uint GEOM_PLANE = 1u;
static const uint GEOM_CAPSULE = 2u;
static const uint GEOM_BOX = 3u;
static const uint GEOM_PARTICLESYSTEM = 6u;

// PxParticleFlag::eENABLE_SPECULATIVE_CCD (PxPBDParticleSystem.h:185)
static const uint PS_FLAG_ENABLE_CCD = 8u;

// gridCal.cuh forward-projection step scales.
static const float FWD_PGS = 1.0f;
static const float FWD_TGS = 0.5f;

// ---------------------------------------------------------------------------
// Local element-struct mirrors (byte-exact vs the engine C++ layouts; pinned by
// static_asserts in testPsContactGen.cpp).
// ---------------------------------------------------------------------------
// PxgContactManagerInput (PxgContactManager.h, 16 bytes)
struct PxgContactManagerInput
{
    uint shapeRef0;          // @0
    uint shapeRef1;          // @4
    uint transformCacheRef0; // @8
    uint transformCacheRef1; // @12
}

// PxgShape (PxgConvexConvexShape.h, 48 bytes)
struct PxgShape
{
    float scaleX, scaleY, scaleZ;                     // @0
    float scaleRotX, scaleRotY, scaleRotZ, scaleRotW; // @12
    uint materialIndex;                               // @28
    uint2 hullOrMeshOffset;                           // @32
    uint type;                                        // @40
    uint particleOrSoftbodyId;                        // @44
}                                                     // = 48

// PxsCachedTransform (PxsTransformCache.h, 32 bytes): PxTransform (q@0, p@16) + flags.
struct PxsCachedTransform
{
    float qx, qy, qz, qw; // @0  PxQuat
    float px, py, pz;     // @16 PxVec3
    uint flags;           // @28 -> 32
}

// PxgRigidFilterPair (PxgSimulationCoreDesc.h, 32 bytes): index0 u64, index1 u64,
// index2 u32, padding[3].
struct PairR { uint2 index0; uint2 index1; uint index2; uint pad0, pad1, pad2; }

// ---------------------------------------------------------------------------
// Push constant blob — BYTE-IDENTICAL to the CUDA PxCudaKernelParam pack.  Each
// arg is placed at its natural alignment (scalar -> its size, pointer/struct ->
// 8); the trailing 72-byte writer struct is inlined field-by-field @104.
//   isTGS(bool)@0 numTests@4 toleranceLength@8 [pad@12]
//   cmInputs@16 shapes@24 transformCache@32 bounds@40 contactDistance@48
//   restDistances@56 materials@64 startIndices@72 totalNumPairs@80
//   particleSystems@88 shapeToRigidRemap@96
//   writer @104 : particleContacts@104 numTotalContacts@112
//     contactSortedByParticle@120 tempContactByParticle@128
//     contactIndexSortedByParticle@136 contactByRigid@144 tempContactByRigid@152
//     contactIndexSortedByRigid@160 maxContacts@168 [pad@172]
//   total = 176 bytes (sizeof(writer) = 72).
// ---------------------------------------------------------------------------
struct PsCollideParams
{
    uint isTGSWord;                        // @0  (byte0 = bool isTGS)
    uint numTests;                         // @4
    float toleranceLength;                 // @8  (unused; ABI parity)
    uint _pad12;                           // @12
    uint64_t cmInputs;                     // @16
    uint64_t shapes;                       // @24
    uint64_t transformCache;               // @32
    uint64_t bounds;                       // @40
    uint64_t contactDistance;              // @48
    uint64_t restDistances;                // @56
    uint64_t materials;                    // @64 (unused; ABI parity)
    uint64_t startIndices;                 // @72 (per-pair cell run-sum)
    uint64_t totalNumPairs;                // @80
    uint64_t particleSystems;              // @88
    uint64_t shapeToRigidRemap;            // @96
    // ---- PxgParticleContactWriter (by value, 72B) @104 ----
    uint64_t w_particleContacts;           // @104 PxgParticlePrimitiveContact*
    uint64_t w_numTotalContacts;           // @112 PxU32*
    uint64_t w_contactSortedByParticle;    // @120 PxU64*
    uint64_t w_tempContactByParticle;      // @128 PxU32*
    uint64_t w_contactIndexSortedByParticle; // @136 PxU32*
    uint64_t w_contactByRigid;             // @144 PxNodeIndex*
    uint64_t w_tempContactByRigid;         // @152 PxU32*
    uint64_t w_contactIndexSortedByRigid;  // @160 PxU32*
    uint w_maxContacts;                    // @168 PxU32
    uint _pad172;                          // @172 -> 176
}
[[vk::push_constant]]
ConstantBuffer<PsCollideParams> gP;

// PxgParticlePrimitiveContact (PxgParticleSystem.h, PX_ALIGN(16), 32 bytes)
struct PsContact
{
    float4 normal_pen; // @0
    uint2 rigidId;     // @16 PxU64
    uint2 particleId;  // @24 PxU64 -> 32
}

// ---------------------------------------------------------------------------
// quat helpers (pxgSlangCommon.slang:104/118, transcribed to keep the rigid-
// common module out of this TU).
// ---------------------------------------------------------------------------
float3 quatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3((vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2),
                  (vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2),
                  (vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2));
}
float3 quatRotateInv(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3((vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * dot2),
                  (vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * dot2),
                  (vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * dot2));
}
float3 getBasisVector0(float4 q)
{
    const float x2 = q.x * 2.0f;
    const float w2 = q.w * 2.0f;
    return float3((q.w * w2) - 1.0f + q.x * x2,
                  (q.z * w2) + q.y * x2,
                  (-q.y * w2) + q.z * x2);
}

// ---------------------------------------------------------------------------
// gridCal helpers (gridCal.slang) for PxgCellData::update.
// ---------------------------------------------------------------------------
int3 calcGridPos(float3 p, float cw)
{
    return int3(int(floor(p.x / cw)), int(floor(p.y / cw)), int(floor(p.z / cw)));
}
uint calcGridHash(int3 g, uint3 s)
{
    g.x = g.x & int(s.x - 1u);
    g.y = g.y & int(s.y - 1u);
    g.z = g.z & int(s.z - 1u);
    return uint((g.z * int(s.y)) * int(s.x) + g.y * int(s.x) + g.x);
}
uint3 calcWrappedGridRangeSize(int3 mn, int3 mx, uint3 s)
{
    return uint3(uint(clamp(mx.x - mn.x + 1, 0, int(s.x))),
                 uint(clamp(mx.y - mn.y + 1, 0, int(s.y))),
                 uint(clamp(mx.z - mn.z + 1, 0, int(s.z))));
}
uint3 calcGridOffsetInRange(uint3 rs, uint off)
{
    return uint3(off % rs.x, (off / rs.x) % rs.y, off / (rs.x * rs.y));
}
uint calcGridHashInBounds(float3 bmn, float3 bmx, float cw, uint cellOffset, uint3 gridSize)
{
    const int3 gpMin = calcGridPos(bmn, cw);
    const int3 gpMax = calcGridPos(bmx, cw);
    const uint3 rs = calcWrappedGridRangeSize(gpMin, gpMax, gridSize);
    const uint3 off = calcGridOffsetInRange(rs, cellOffset);
    const int3 gp = int3(gpMin.x + int(off.x), gpMin.y + int(off.y), gpMin.z + int(off.z));
    return calcGridHash(gp, gridSize);
}

float getSpecVolume(out float3 cVolumePos, float3 currentPos, float3 predictedPos,
                    float contactDist, bool isTGS)
{
    const float scaleInv = isTGS ? (1.0f / FWD_TGS) : (1.0f / FWD_PGS);
    const float3 cVolumeOffset = (predictedPos - currentPos) * scaleInv * 0.5f;
    cVolumePos = currentPos + cVolumeOffset;
    return length(cVolumeOffset) + contactDist;
}

// binarySearch (reduction.cuh:729): largest index s.t. data[index] <= value,
// else 0.
uint binarySearchRunSum(Ptr<uint> data, uint n, uint value)
{
    uint l = 0u, r = n;
    while (l < r)
    {
        const uint m = (l + r) / 2u;
        if (data[m] <= value) l = m + 1u; else r = m;
    }
    return l != 0u ? l - 1u : 0u;
}

// ---------------------------------------------------------------------------
// (particleSystem, rigidId, compressedParticleId) rigid filter lookup
// (deformableElementFilter.cuh:118).
// ---------------------------------------------------------------------------
bool u64Lt(uint2 a, uint2 b) { return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x); }
bool u64Eq(uint2 a, uint2 b) { return a.x == b.x && a.y == b.y; }

uint findRange(Ptr<uint> values, uint length, uint value)
{
    if (length == 0u) return 0u;
    uint l = 0u, r = length;
    while (l < r)
    {
        const uint m = (l + r) / 2u;
        if (values[m] > value) r = m; else l = m + 1u;
    }
    return r - 1u;
}

// PxgRigidFilterPair::compare (index0 u64, index1 u64, index2 u32) with the
// ignoreIndex2 early accept.
bool findRigid(uint64_t pairsAddr, uint nbPairs, uint2 q0, uint2 q1)
{
    Ptr<PairR> pairs = Ptr<PairR>(pairsAddr);
    uint l = 0u, r = nbPairs;
    while (l < r)
    {
        const uint mid = (l + r) / 2u;
        const PairR p = pairs[mid];
        int cmp = 0;
        if (u64Lt(q0, p.index0)) cmp = -1;
        else if (u64Lt(p.index0, q0)) cmp = 1;
        else if (u64Lt(q1, p.index1)) cmp = -1;
        else if (u64Lt(p.index1, q1)) cmp = 1;
        else if (0u < p.index2) cmp = -1;
        if (cmp == 0 || (u64Eq(q0, p.index0) && u64Eq(q1, p.index1)))
            return true;
        if (cmp < 0) r = mid; else l = mid + 1u;
    }
    return false;
}

bool findPs(Ptr<PxgParticleSystemGpu> systems, uint psId, uint2 rigidId, uint2 compressedParticleId)
{
    const uint particleId = compressedParticleId.x; // PxGetParticleIndex = low 32
    const uint numBuffers = systems[psId].mCommonData_NumParticleBuffers;
    Ptr<uint> runsum = Ptr<uint>(systems[psId].mParticleBufferRunsum);
    const uint id = findRange(runsum, numBuffers, particleId);
    Ptr<PxgParticleSimBufferGpu> simBufs = Ptr<PxgParticleSimBufferGpu>(systems[psId].mParticleSimBuffers);
    const uint64_t pairsAddr = simBufs[id].mFilterPairs;
    const uint nbPairs = simBufs[id].mNumFilterPairs;
    const uint localId = particleId - runsum[id];
    return findRigid(pairsAddr, nbPairs, rigidId, uint2(localId, 0u));
}

// ---------------------------------------------------------------------------
// particlePrimitiveCollision (particleCollision.cuh / psPrimCollide.slang) — the
// contact core.  Keys on the REAL PxGeometryType.  eCONVEXCORE deferred.
// ---------------------------------------------------------------------------
float distancePointSegmentSquared(float3 a, float3 b, float3 p, out float param)
{
    const float3 ap = p - a;
    const float3 ab = b - a;
    const float nom = dot(ap, ab);
    const float denom = dot(ab, ab);
    const float tValue = clamp(nom / denom, 0.0f, 1.0f);
    const float t = denom > 0.0f ? tValue : 0.0f;
    const float3 v = ap - (ab * t);
    param = t;
    return dot(v, v);
}

bool contactPointBox(inout float3 normal, inout float dist,
                     float3 pointPos, float3 boxP, float4 boxQ, float3 halfBoxExtents, float maxDist)
{
    const float3 relPointPos = quatRotateInv(boxQ, pointPos - boxP);
    const float3 closestPos = clamp(relPointPos, -halfBoxExtents, halfBoxExtents);
    if (maxDist < PX_MAX_F32)
    {
        const float maxDistSq = maxDist * maxDist;
        const float3 dv = relPointPos - closestPos;
        if (dot(dv, dv) > maxDistSq)
            return false;
    }
    const bool bX = halfBoxExtents.x >= abs(relPointPos.x);
    const bool bY = halfBoxExtents.y >= abs(relPointPos.y);
    const bool bZ = halfBoxExtents.z >= abs(relPointPos.z);
    const bool isInside = bX && bY && bZ;
    if (isInside)
    {
        const float3 distToSurface = halfBoxExtents - abs(closestPos);
        normal = closestPos.z >= 0.0f ? float3(0, 0, 1) : float3(0, 0, -1);
        dist = -distToSurface.z;
        if (distToSurface.x <= distToSurface.y && distToSurface.x <= distToSurface.z)
        {
            normal = closestPos.x >= 0.0f ? float3(1, 0, 0) : float3(-1, 0, 0);
            dist = -distToSurface.x;
        }
        else if (distToSurface.y <= distToSurface.z)
        {
            normal = closestPos.y >= 0.0f ? float3(0, 1, 0) : float3(0, -1, 0);
            dist = -distToSurface.y;
        }
    }
    else
    {
        normal = relPointPos - closestPos;
        const float mag = sqrt(dot(normal, normal));
        normal = normal * (1.0f / mag);
        dist = mag;
    }
    normal = quatRotate(boxQ, normal);
    return true;
}

bool contactPointSphere(inout float3 normal, inout float dist,
                        float3 pointPos, float3 spherePos, float sphereRadius, float maxDist)
{
    const float3 relPointPos = pointPos - spherePos;
    const float distCenterSq = dot(relPointPos, relPointPos);
    if (maxDist < PX_MAX_F32)
    {
        const float maxDistCenter = maxDist + sphereRadius;
        if (distCenterSq > maxDistCenter * maxDistCenter)
            return false;
    }
    const float distCenter = sqrt(distCenterSq);
    normal = (distCenter > PX_EPS_REAL) ? relPointPos * (1.0f / distCenter) : float3(1, 0, 0);
    dist = distCenter - sphereRadius;
    return true;
}

bool contactPointCapsule(inout float3 normal, inout float dist,
                         float3 pointPos, float3 capsulePos, float3 capsuleDir,
                         float capsuleRadius, float capsuleHalfHeight, float maxDist)
{
    const float3 capsuleAxisVec = capsuleDir * capsuleHalfHeight;
    const float3 relPointPos = pointPos - capsulePos;
    float t;
    const float distSegmentSq = distancePointSegmentSquared(-capsuleAxisVec, capsuleAxisVec, relPointPos, t);
    if (maxDist < PX_MAX_F32)
    {
        const float maxDistSegment = maxDist + capsuleRadius;
        if (distSegmentSq > maxDistSegment * maxDistSegment)
            return false;
    }
    const float3 distSegmentVec = relPointPos - capsuleAxisVec * (2.0f * t - 1.0f);
    const float distSegment = sqrt(distSegmentSq);
    normal = (distSegment > PX_EPS_REAL) ? (distSegmentVec * (1.0f / distSegment)) : float3(1, 0, 0);
    dist = distSegment - capsuleRadius;
    return true;
}

// shapeType keys the REAL PxGeometryType.  shapeQ / shapeP = the rigid transform
// (from transformCache); scale = PxgShape.scale.scale.
bool particlePrimitiveCollision(float3 particlePos, float3 cVolumePos, float cVolumeRadius,
                                float4 shapeQ, float3 shapeP, uint shapeType, float3 scale,
                                bool enableCCD, inout float3 normal, inout float dist)
{
    bool intersect = false;
    switch (shapeType)
    {
    case GEOM_PLANE:
    {
        const float3 pn = quatRotate(shapeQ, float3(1, 0, 0));
        const float pdd = -dot(pn, shapeP);
        dist = dot(pn, cVolumePos) + pdd;
        if (dist <= cVolumeRadius)
        {
            if (enableCCD)
                dist = dot(pn, particlePos) + pdd;
            normal = pn;
            intersect = true;
        }
        break;
    }
    case GEOM_SPHERE:
    {
        intersect = contactPointSphere(normal, dist, cVolumePos, shapeP, scale.x, cVolumeRadius);
        if (intersect && enableCCD)
            contactPointSphere(normal, dist, particlePos, shapeP, scale.x, PX_MAX_F32);
        break;
    }
    case GEOM_CAPSULE:
    {
        const float3 capsuleDir = getBasisVector0(shapeQ);
        intersect = contactPointCapsule(normal, dist, cVolumePos, shapeP, capsuleDir,
                                        scale.y, scale.x, cVolumeRadius);
        if (intersect && enableCCD)
            contactPointCapsule(normal, dist, particlePos, shapeP, capsuleDir,
                                scale.y, scale.x, PX_MAX_F32);
        break;
    }
    case GEOM_BOX:
    {
        intersect = contactPointBox(normal, dist, cVolumePos, shapeP, shapeQ, scale, cVolumeRadius);
        if (intersect && enableCCD)
            contactPointBox(normal, dist, particlePos, shapeP, shapeQ, scale, PX_MAX_F32);
        break;
    }
    default: // eCONVEXCORE and beyond deferred (CUDA default: no intersect)
        break;
    }
    return intersect;
}

// PxgParticleContactWriter::writeContact — capacity-gated 7-array staging fan-out
// through the by-value writer's BDA pointers.
void writeContact(uint index, float4 normalPen, uint2 compressedParticleIndex, uint2 rigidId)
{
    if (index < gP.w_maxContacts)
    {
        Ptr<uint2>(gP.w_contactByRigid)[index] = rigidId;
        Ptr<uint>(gP.w_tempContactByRigid)[index] = rigidId.x; // rigidId.getInd() & 0xffffffff
        Ptr<uint>(gP.w_contactIndexSortedByRigid)[index] = index;

        Ptr<uint2>(gP.w_contactSortedByParticle)[index] = compressedParticleIndex;
        Ptr<uint>(gP.w_tempContactByParticle)[index] = compressedParticleIndex.x; // PxGetParticleIndex
        Ptr<uint>(gP.w_contactIndexSortedByParticle)[index] = index;

        PsContact ct;
        ct.normal_pen = normalPen;
        ct.rigidId = rigidId;
        ct.particleId = compressedParticleIndex;
        Ptr<PsContact>(gP.w_particleContacts)[index] = ct;
    }
}

// ---------------------------------------------------------------------------
groupshared uint sRange[PS_BLOCK];    // per-thread cell population (block max)
groupshared uint sMask[WARPS_PER_BLOCK];  // per-tile intersect ballot
groupshared uint sStart[WARPS_PER_BLOCK]; // per-tile contact start index

[shader("compute")]
[numthreads(256, 1, 1)]
void ps_primitivesCollisionLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint warpId = tid / WARP;
    const uint lane = tid & (WARP - 1u);
    const bool isTGS = (gP.isTGSWord & 0xffu) != 0u;

    const uint totalComparison = Ptr<uint>(gP.totalNumPairs)[0];

    Ptr<PxgContactManagerInput> cmInputs = Ptr<PxgContactManagerInput>(gP.cmInputs);
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gP.shapes);
    Ptr<PxsCachedTransform> transformCache = Ptr<PxsCachedTransform>(gP.transformCache);
    Ptr<float> boundsArr = Ptr<float>(gP.bounds);
    Ptr<float> contactDistance = Ptr<float>(gP.contactDistance);
    Ptr<float> restDistances = Ptr<float>(gP.restDistances);
    Ptr<uint> startIndices = Ptr<uint>(gP.startIndices);
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gP.particleSystems);
    Ptr<uint2> shapeToRigidRemap = Ptr<uint2>(gP.shapeToRigidRemap);

    const uint stride = PS_BLOCK * PS_GRID;
    const uint nbIterations = (totalComparison + stride - 1u) / stride;

    for (uint it = 0u; it < nbIterations; ++it)
    {
        const uint workIndex = it * stride + gid.x * PS_BLOCK + tid;
        const bool valid = workIndex < totalComparison;

        // ---- PxgCellData::update ----
        uint psId = 0u;
        uint gridHash = 0u;
        float cDistance = 0.0f;
        float restDistance = 0.0f;
        uint rigidCacheRef = 0u;
        // rigid shape (tested against the particles)
        float4 rigidQ = float4(0, 0, 0, 1);
        float3 rigidP = float3(0, 0, 0);
        uint rigidType = 0xffffffffu;
        float3 rigidScale = float3(0, 0, 0);
        uint2 rigidId = uint2(EMPTY_CELL, 0u);
        uint startIndex = EMPTY_CELL;
        uint endIndex = 0u;
        uint range = 0u;
        bool enableCCD = false;
        uint64_t curPosAddr = 0ull, predPosAddr = 0ull, sortedToUnsortedAddr = 0ull;

        if (valid)
        {
            const uint pos = binarySearchRunSum(startIndices, gP.numTests, workIndex);
            const uint startPairIndex = startIndices[pos];

            // LoadShapePair<ePARTICLESYSTEM>: particle shape -> slot0, other -> rigid.
            const PxgContactManagerInput cin = cmInputs[pos];
            uint particleShapeRef = cin.shapeRef0;
            uint particleCacheRef = cin.transformCacheRef0;
            uint rigidShapeRef = cin.shapeRef1;
            rigidCacheRef = cin.transformCacheRef1;
            if (shapes[cin.shapeRef0].type != GEOM_PARTICLESYSTEM)
            {
                particleShapeRef = cin.shapeRef1;
                particleCacheRef = cin.transformCacheRef1;
                rigidShapeRef = cin.shapeRef0;
                rigidCacheRef = cin.transformCacheRef0;
            }

            const PxgShape rigidShape = shapes[rigidShapeRef];
            rigidType = rigidShape.type;
            rigidScale = float3(rigidShape.scaleX, rigidShape.scaleY, rigidShape.scaleZ);
            const PxsCachedTransform tc = transformCache[rigidCacheRef];
            rigidQ = float4(tc.qx, tc.qy, tc.qz, tc.qw);
            rigidP = float3(tc.px, tc.py, tc.pz);
            rigidId = shapeToRigidRemap[rigidCacheRef];

            psId = shapes[particleShapeRef].particleOrSoftbodyId;

            cDistance = contactDistance[particleCacheRef] + contactDistance[rigidCacheRef];
            restDistance = restDistances[pos];

            // combine(bounds[particleCacheRef], bounds[rigidCacheRef]).fattenFast(cDistance)
            const uint pb = particleCacheRef * 6u;
            const uint rb = rigidCacheRef * 6u;
            float3 omn = max(float3(boundsArr[pb + 0u], boundsArr[pb + 1u], boundsArr[pb + 2u]),
                             float3(boundsArr[rb + 0u], boundsArr[rb + 1u], boundsArr[rb + 2u]));
            float3 omx = min(float3(boundsArr[pb + 3u], boundsArr[pb + 4u], boundsArr[pb + 5u]),
                             float3(boundsArr[rb + 3u], boundsArr[rb + 4u], boundsArr[rb + 5u]));
            omn -= float3(cDistance, cDistance, cDistance);
            omx += float3(cDistance, cDistance, cDistance);

            const float cellWidth = systems[psId].mCommonData_GridCellWidth;
            const uint3 gridSize = uint3(systems[psId].mCommonData_GridSizeX,
                                         systems[psId].mCommonData_GridSizeY,
                                         systems[psId].mCommonData_GridSizeZ);
            const uint cellOffset = workIndex - startPairIndex;
            gridHash = calcGridHashInBounds(omn, omx, cellWidth, cellOffset, gridSize);

            // ---- psPrimitivesCollision setup ----
            enableCCD = (systems[psId].mData_Flags & PS_FLAG_ENABLE_CCD) > 0u;
            curPosAddr = systems[psId].mSortedOriginPos_InvMass;
            predPosAddr = systems[psId].mSortedPositions_InvMass;
            sortedToUnsortedAddr = systems[psId].mSortedToUnsortedMapping;

            startIndex = Ptr<uint>(systems[psId].mCellStart)[gridHash];
            if (startIndex != EMPTY_CELL)
            {
                endIndex = Ptr<uint>(systems[psId].mCellEnd)[gridHash];
                range = endIndex - startIndex;
            }
        }

        // ---- block-uniform max cell population (llvmpipe guard: no warp shuffle) ----
        sRange[tid] = range;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = PS_BLOCK / 2u; s > 0u; s >>= 1)
        {
            if (tid < s)
                sRange[tid] = max(sRange[tid], sRange[tid + s]);
            GroupMemoryBarrierWithGroupSync();
        }
        const uint blockMaxRange = sRange[0];
        GroupMemoryBarrierWithGroupSync();

        for (uint ind = 0u; ind < blockMaxRange; ++ind)
        {
            bool intersect = false;
            float3 normal = float3(0, 0, 0);
            float distance = 0.0f;
            uint2 compressed = uint2(0, 0);

            if (valid && startIndex != EMPTY_CELL)
            {
                const uint particleIndex = startIndex + ind;
                if (particleIndex < endIndex)
                {
                    const float3 currentPos = Ptr<float4>(curPosAddr)[particleIndex].xyz;

                    float3 cVolumePos;
                    float cVolumeRadius;
                    if (enableCCD)
                    {
                        const float3 predictedPos = Ptr<float4>(predPosAddr)[particleIndex].xyz;
                        cVolumeRadius = getSpecVolume(cVolumePos, currentPos, predictedPos, cDistance, isTGS);
                    }
                    else
                    {
                        cVolumeRadius = cDistance;
                        cVolumePos = currentPos;
                    }

                    compressed = uint2(particleIndex, psId); // PxEncodeParticleIndex
                    const uint2 particleMask = uint2(Ptr<uint>(sortedToUnsortedAddr)[particleIndex], psId);

                    if (!findPs(systems, psId, rigidId, particleMask))
                        intersect = particlePrimitiveCollision(currentPos, cVolumePos, cVolumeRadius,
                                                               rigidQ, rigidP, rigidType, rigidScale,
                                                               enableCCD, normal, distance);
                }
            }

            // ---- per-32-lane-tile contact aggregation (one atomicAdd per tile) ----
            const uint threadMask = (1u << lane) - 1u;
            if (lane == 0u) sMask[warpId] = 0u;
            GroupMemoryBarrierWithGroupSync();
            if (intersect) InterlockedOr(sMask[warpId], 1u << lane);
            GroupMemoryBarrierWithGroupSync();
            const uint mask = sMask[warpId];
            const uint nbToLock = countbits(mask);
            const uint contactIndex = countbits(mask & threadMask);
            if (lane == 0u)
            {
                uint s = 0u;
                if (nbToLock != 0u)
                    InterlockedAdd(Ptr<uint>(gP.w_numTotalContacts)[0], nbToLock, s);
                sStart[warpId] = s;
            }
            GroupMemoryBarrierWithGroupSync();
            const uint contactStartIndex = sStart[warpId];

            if (intersect)
                writeContact(contactStartIndex + contactIndex,
                             float4(normal, distance - restDistance), compressed, rigidId);
            GroupMemoryBarrierWithGroupSync();
        }
    }
}
