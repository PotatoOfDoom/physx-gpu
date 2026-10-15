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
//   - NO 32-lane-warp assumption.  The CUDA caller's ballot/warpScan compaction is
//     a block prefix sum through groupshared here (see the entry).
//   - the per-warp `range = warpReduction<Max>` round loop and the per-warp contact
//     aggregation (ballot + popc prefix + ONE lane-0 atomicAdd per round) need a
//     32-lane group that can be synchronised, which takes subgroup operations
//     (forbidden in engine modules, check_subgroup_ops.py).  They were first lowered
//     to a BLOCK-max round loop with block barriers; that made every thread run the
//     fullest cell's rounds (~50% of SnippetPBFMultiMat's GPU time on an RX 9070 XT
//     once the fluid settled).  Now each thread walks its own cell: count, one
//     atomicAdd per cell, write (psPrimProcessCell).  Same contact set and count;
//     the stream order is per cell instead of per warp round -- it was atomic-order
//     dependent anyway, and every consumer sorts.
//   - the CUDA wrapper's compaction of non-empty work items into full blocks IS kept
//     (the ps_primitivesCollisionLaunch entry), in the CUDA's queue order.
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
import pxgSlangDefFilter;   // P9.38
// P9.14b: the convex particle path.  All three are already-swept modules -- the
// cooked-hull blob reader, the GJK driver and the GJK-then-EPA policy layer that
// the softbody-vs-rigid path goes through.  particleConvexCollision
// (cudaParticleSystem.cu:666) is that same policy layer with a point cloud on one
// side, so it is COMPOSED here rather than transcribed a second time.
import pxgSlangConvexHull;
import pxgSlangGjk;
import pxgSlangTetCollide;

// P9.89: EVERY dot() IN THIS FILE IS ORDERED BY THE NoContraction STAMP.
// Each float dot here transcribes one of two CUDA shapes:
//   * a PxVec3 dot or magnitudeSquared, `x*v.x + y*v.y + z*v.z`
//     (PxVec3.h:173/:274), i.e. (x + y) + z with every product rounded; or
//   * the convex kernels' three-lane penetration, shfl0 + shfl1 + shfl2
//     (cudaParticleSystem.cu:964/:1117), i.e. the same (p0 + p1) + p2.
// SPIR-V does not fix OpDot's summation order.  llvmpipe (Mesa lower_fdot)
// evaluates an UNDECORATED OpDot as x + (y + z) and a NoContraction-decorated
// one as (x + y) + z.  Both were measured on 65536 random cases with 0 misses.
// This .spv was unstamped until P9.89, so every site below came out
// re-associated.  The fix is psContactGenBda's entry in PXG_NOCONTRACT_SHADERS
// (tests/CMakeLists.txt).  It stamps the importers of pxgSlangGjk / Epa /
// TetCollide here the way the softbody path already stamped them, and that
// includes the anomaly dot at pxgSlangTetCollide.slang:93.  Taking this shader
// off that list re-associates every site marked "P9.89" below.
// testPsContactGen (p89DotOrder), testPsConvexContactGen (p89PenLaneOrder) and
// its diffuse twin compare against -ffp-contract=off mirrors in the CUDA order
// and fail without the stamp.

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
static const float FWD_DIFFUSE = 1.0f;   // PARTICLE_FORWARD_PROJECTION_STEP_SCALE_DIFFUSE

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

// isDiffuse is NOT a synonym for "PGS".  getParticleSpeculativeContactVolume
// (gridCal.cuh:122) branches on it FIRST and only then on isTGS, and
// PARTICLE_FORWARD_PROJECTION_STEP_SCALE_DIFFUSE is 1.0 -- so a diffuse particle
// keeps scaleInv 1.0 even in a TGS step, where a fluid particle gets 2.0.  The
// fluid kernel below has always passed false and could not see this; the diffuse
// kernel is what makes the parameter load-bearing.
//
// One mutant survives here and its proof is worth keeping, because it depends on
// two engine constants rather than on this code: passing `isDiffuse = isTGS` is
// EQUIVALENT.  With isTGS true it takes the diffuse branch (1/FWD_DIFFUSE); with
// isTGS false it takes the PGS branch (1/FWD_PGS) -- and FWD_DIFFUSE and FWD_PGS
// are BOTH 1.0 in gridCal.cuh, so both give 1.0, which is what a diffuse caller
// must get either way.  The spelling is still wrong and the sweep says so: if
// NVIDIA ever gives the diffuse scale its own value, that mutant starts dying.
float getSpecVolume(out float3 cVolumePos, float3 currentPos, float3 predictedPos,
                    float contactDist, bool isDiffuse, bool isTGS)
{
    const float scaleInv = isDiffuse ? (1.0f / FWD_DIFFUSE)
                                     : (isTGS ? (1.0f / FWD_TGS) : (1.0f / FWD_PGS));
    const float3 cVolumeOffset = (predictedPos - currentPos) * scaleInv * 0.5f;
    cVolumePos = currentPos + cVolumeOffset;
    // P9.47: gridCal.cuh:139 is `cVolumeOffset.magnitude() + contactDist`, and
    // PxVec3::magnitude() is PxSqrt(magnitudeSquared()) -- not GLSL Length, which
    // the Vulkan environment spec leaves several ULP of room.
    // P9.89: magnitudeSquared is (x*x + y*y) + z*z (gridCal.cuh:139, PxVec3.h:173).
    // That holds only because this .spv is stamped (see the P9.89 note at the
    // imports).  The radius only GATES, so testPsContactGen plants a CCD particle
    // whose plane distance falls between the two possible radii.
    return sqrt(dot(cVolumeOffset, cVolumeOffset)) + contactDist;
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
    return findRigid(pairsAddr, nbPairs, rigidId, localId);
}

// ---------------------------------------------------------------------------
// particlePrimitiveCollision (particleCollision.cuh / psPrimCollide.slang) — the
// contact core.  Keys on the REAL PxGeometryType.  eCONVEXCORE deferred.
// ---------------------------------------------------------------------------
float distancePointSegmentSquared(float3 a, float3 b, float3 p, out float param)
{
    const float3 ap = p - a;
    const float3 ab = b - a;
    // P9.89: sphereCollision.cuh:97-98 ap.dot(ab) / ab.dot(ab), and :103 v.dot(v)
    // below, are all (x + y) + z.  The stamp keeps OpDot in that order.
    const float nom = dot(ap, ab);
    const float denom = dot(ab, ab);
    // P9.86f: sphereCollision.cuh:98 PxClamp(nom / denom, 0, 1) = PxMin(1, PxMax(0, .)) with the float
    // selects of PxUnixMathIntrinsics.h:116-125.  A NaN quotient (a particle at infinity against a
    // capsule axis lane of 0: inf * 0) stays NaN, so the CUDA's distSegmentSq is NaN, passes the maxDist
    // gate (NaN > x is false) and reports a contact with normal (1,0,0) and a NaN distance; GLSL FClamp
    // gave t = 0 and an infinite distance that the gate rejects.  A -0 quotient only reaches 2t - 1 and
    // a squared lane, so its sign never shows.
    const float tQ = nom / denom;
    const float tLo = (0.0f > tQ) ? 0.0f : tQ;
    const float tValue = (1.0f < tLo) ? 1.0f : tLo;
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
        // P9.89: particleCollision.cuh:56 magnitudeSquared(), (x + y) + z by the stamp.
        // A gate only: testPsContactGen plants a box whose maxDist^2 falls between the
        // two possible corner values.
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
        // P9.89: normalizeFast's magnitude (particleCollision.cuh:92), (x + y) + z by the stamp.
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
    // P9.89: particleCollision.cuh:107 magnitudeSquared(), (x + y) + z by the stamp.
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
        // P9.86: PxPlane(1,0,0,0).transform (PxPlane.h:137, cudaParticleSystem.cu:285) is
        // `d - p.dot(n)` with d a literal 0, i.e. IEEE `0 - x`: +0 for EITHER zero.  Unary
        // minus turns the +0 dot of a plane through the origin into -0, and dist = dot + pdd
        // then keeps a -0 dot (particle exactly on the plane) that the CUDA's +0 pdd turns
        // into +0 -- stored as normal_pen.w when restDistance == +0.
        // P9.89: p.dot(n) here and both plane.distance() dots below (PxPlane.h,
        // cudaParticleSystem.cu:285-294) are (x + y) + z by the stamp.
        const float pdot = dot(pn, shapeP);
        const float pdd = (pdot == 0.0f) ? 0.0f : -pdot;
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

// P9.38: the binary search over PxgRigidFilterPair now lives in pxgSlangDefFilter.
// It was written out by hand fifteen times across this corpus; every copy was checked
// against deformableElementFilter.cuh:67 and agreed, and that is exactly why they were
// worth collapsing -- fifteen identical transcriptions are fifteen future chances to
// re-derive one of them wrong.
bool findRigid(uint64_t pairsAddr, uint nbPairs, uint2 q0, uint mask)
{
    return dfFindRigidPair(pairsAddr, nbPairs, q0, mask);
}

struct PsPrimCell
{
    uint psId;
    uint gridHash;
    float cDistance;
    float restDistance;
    uint rigidCacheRef;
    float4 rigidQ;
    float3 rigidP;
    uint rigidType;
    float3 rigidScale;
    uint2 rigidId;
    uint startIndex;
    uint endIndex;
    uint range;
    bool enableCCD;
    uint64_t curPosAddr;
    uint64_t predPosAddr;
    uint64_t sortedToUnsortedAddr;
};

// PxgCellData::update + psPrimitivesCollision setup for one work item: the pair,
// the rigid shape, the cell (gridHash) and its particle range.  No barriers -- the
// compaction pass calls it per thread to find the non-empty cells.
PsPrimCell psPrimSetup(uint workIndex, bool valid)
{
    Ptr<PxgContactManagerInput> cmInputs = Ptr<PxgContactManagerInput>(gP.cmInputs);
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gP.shapes);
    Ptr<PxsCachedTransform> transformCache = Ptr<PxsCachedTransform>(gP.transformCache);
    Ptr<float> boundsArr = Ptr<float>(gP.bounds);
    Ptr<float> contactDistance = Ptr<float>(gP.contactDistance);
    Ptr<float> restDistances = Ptr<float>(gP.restDistances);
    Ptr<uint> startIndices = Ptr<uint>(gP.startIndices);
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gP.particleSystems);
    Ptr<uint2> shapeToRigidRemap = Ptr<uint2>(gP.shapeToRigidRemap);
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

    PsPrimCell c;
    c.psId = psId;
    c.gridHash = gridHash;
    c.cDistance = cDistance;
    c.restDistance = restDistance;
    c.rigidCacheRef = rigidCacheRef;
    c.rigidQ = rigidQ;
    c.rigidP = rigidP;
    c.rigidType = rigidType;
    c.rigidScale = rigidScale;
    c.rigidId = rigidId;
    c.startIndex = startIndex;
    c.endIndex = endIndex;
    c.range = range;
    c.enableCCD = enableCCD;
    c.curPosAddr = curPosAddr;
    c.predPosAddr = predPosAddr;
    c.sortedToUnsortedAddr = sortedToUnsortedAddr;
    return c;
}

// One round of the block on one cell per thread: the cell's particles against the
// rigid shape, contacts aggregated per 32-lane tile.  Contains block barriers: the
// whole block calls it together (block-uniform), `valid` false for idle threads.
void psPrimProcessCell(uint workIndex, bool valid, uint tid, uint warpId, uint lane, bool isTGS)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gP.particleSystems);
    const PsPrimCell c = psPrimSetup(workIndex, valid);
    const uint psId = c.psId;
    const uint gridHash = c.gridHash;
    const float cDistance = c.cDistance;
    const float restDistance = c.restDistance;
    const uint rigidCacheRef = c.rigidCacheRef;
    const float4 rigidQ = c.rigidQ;
    const float3 rigidP = c.rigidP;
    const uint rigidType = c.rigidType;
    const float3 rigidScale = c.rigidScale;
    const uint2 rigidId = c.rigidId;
    const uint startIndex = c.startIndex;
    const uint endIndex = c.endIndex;
    const uint range = c.range;
    const bool enableCCD = c.enableCCD;
    const uint64_t curPosAddr = c.curPosAddr;
    const uint64_t predPosAddr = c.predPosAddr;
    const uint64_t sortedToUnsortedAddr = c.sortedToUnsortedAddr;
    // Each thread walks ITS OWN cell's particles -- no block-max rounds, no barriers.
    // Pass 1 counts the cell's contacts, one atomicAdd reserves them as a contiguous
    // run, pass 2 recomputes and writes them (the test is deterministic, so the two
    // passes agree).  The CUDA aggregates per warp instead (ballot + one atomic per
    // warp per particle round) and loops to the WARP's largest cell; the port had
    // lowered that to the BLOCK's largest cell with three block barriers per round,
    // because a 32-lane tile cannot be synchronised without subgroup operations
    // (check_subgroup_ops.py).  Once a fluid settles in a container, cell
    // populations are uneven and every one of the 256 threads ran the fullest cell's
    // rounds: SnippetPBFMultiMat spent ~50% of its GPU time here on an RX 9070 XT.
    // The contact SET and count are the CUDA's; the order within the stream differs
    // (per cell instead of per warp round), and it was atomic-order dependent before.
    uint nbContacts = 0u;
    if (valid && startIndex != EMPTY_CELL)
    {
        for (uint particleIndex = startIndex; particleIndex < endIndex; ++particleIndex)
        {
            float3 normal;
            float distance;
            if (psPrimTestParticle(systems, psId, rigidId, particleIndex, curPosAddr, predPosAddr,
                                   sortedToUnsortedAddr, enableCCD, isTGS, cDistance, rigidQ, rigidP,
                                   rigidType, rigidScale, normal, distance))
                ++nbContacts;
        }
    }
    if (nbContacts == 0u)
        return;
    uint contactIndex = 0u;
    InterlockedAdd(Ptr<uint>(gP.w_numTotalContacts)[0], nbContacts, contactIndex);
    for (uint particleIndex = startIndex; particleIndex < endIndex; ++particleIndex)
    {
        float3 normal;
        float distance;
        if (psPrimTestParticle(systems, psId, rigidId, particleIndex, curPosAddr, predPosAddr,
                               sortedToUnsortedAddr, enableCCD, isTGS, cDistance, rigidQ, rigidP,
                               rigidType, rigidScale, normal, distance))
        {
            writeContact(contactIndex, float4(normal, distance - restDistance),
                         uint2(particleIndex, psId) /* PxEncodeParticleIndex */, rigidId);
            ++contactIndex;
        }
    }
}

// One particle of a cell against the rigid shape (the CUDA loop body,
// cudaParticleSystem.cu psPrimitivesCollision): speculative volume, the
// particle-rigid filter, then the primitive test.
bool psPrimTestParticle(Ptr<PxgParticleSystemGpu> systems, uint psId, uint2 rigidId, uint particleIndex,
                        uint64_t curPosAddr, uint64_t predPosAddr, uint64_t sortedToUnsortedAddr,
                        bool enableCCD, bool isTGS, float cDistance, float4 rigidQ, float3 rigidP,
                        uint rigidType, float3 rigidScale, out float3 normal, out float distance)
{
    normal = float3(0, 0, 0);
    distance = 0.0f;
    const float3 currentPos = Ptr<float4>(curPosAddr)[particleIndex].xyz;

    float3 cVolumePos;
    float cVolumeRadius;
    if (enableCCD)
    {
        const float3 predictedPos = Ptr<float4>(predPosAddr)[particleIndex].xyz;
        cVolumeRadius = getSpecVolume(cVolumePos, currentPos, predictedPos, cDistance, false, isTGS);
    }
    else
    {
        cVolumeRadius = cDistance;
        cVolumePos = currentPos;
    }

    const uint2 particleMask = uint2(Ptr<uint>(sortedToUnsortedAddr)[particleIndex], psId);
    if (findPs(systems, psId, rigidId, particleMask))
        return false;
    return particlePrimitiveCollision(currentPos, cVolumePos, cVolumeRadius, rigidQ, rigidP, rigidType, rigidScale,
                                      enableCCD, normal, distance);
}


groupshared uint sWork[PS_BLOCK];     // compacted non-empty work items (cudaParticleSystem.cu shWorkIndices)
groupshared uint sScan[PS_BLOCK];     // block prefix sum of "this item's cell is non-empty"

[shader("compute")]
[numthreads(256, 1, 1)]
void ps_primitivesCollisionLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint warpId = tid / WARP;
    const uint lane = tid & (WARP - 1u);
    const bool isTGS = (gP.isTGSWord & 0xffu) != 0u;

    const uint totalComparison = Ptr<uint>(gP.totalNumPairs)[0];
    const uint stride = PS_BLOCK * PS_GRID;
    const uint nbIterations = (totalComparison + stride - 1u) / stride;

    // The CUDA wrapper's compaction (cudaParticleSystem.cu:1189-1320): each round, the
    // work items whose cell holds particles are appended, in thread order, to a
    // shared queue; a full queue (PS_BLOCK items) is processed by the whole block,
    // the remainder after the last round.  Most cells of a pair's overlap box are
    // empty -- far more once a fluid splashes and its bounds grow -- and processing
    // every block of raw work items with the particle loop made this kernel ~50x
    // slower in that phase (SnippetPBFMultiMat on an RX 9070 XT).  The ballot +
    // warpScan of the CUDA becomes a block prefix sum through groupshared; the queue
    // order (and so which cells share a 32-lane tile) is the CUDA's.
    uint workCount = 0u;   // block-uniform: derived from groupshared totals only
    for (uint it = 0u; it < nbIterations; ++it)
    {
        const uint workIndex = it * stride + gid.x * PS_BLOCK + tid;
        const bool valid = workIndex < totalComparison;
        uint flag = 0u;
        if (valid)
        {
            const PsPrimCell c = psPrimSetup(workIndex, true);
            flag = c.range != 0u ? 1u : 0u;
        }

        // inclusive scan of `flag` over the block (Hillis-Steele)
        sScan[tid] = flag;
        GroupMemoryBarrierWithGroupSync();
        for (uint off = 1u; off < PS_BLOCK; off <<= 1)
        {
            const uint v = tid >= off ? sScan[tid - off] : 0u;
            GroupMemoryBarrierWithGroupSync();
            sScan[tid] += v;
            GroupMemoryBarrierWithGroupSync();
        }
        const uint total = sScan[PS_BLOCK - 1u];
        const uint outIndex = workCount + sScan[tid] - flag;
        uint overflowIndex = 0xffffffffu;
        if (flag != 0u)
        {
            if (outIndex < PS_BLOCK)
                sWork[outIndex] = workIndex;
            else
                overflowIndex = outIndex - PS_BLOCK;
        }
        GroupMemoryBarrierWithGroupSync();

        workCount += total;
        if (workCount >= PS_BLOCK)
        {
            const uint queued = sWork[tid];
            psPrimProcessCell(queued, true, tid, warpId, lane, isTGS);
            workCount -= PS_BLOCK;
            GroupMemoryBarrierWithGroupSync();
            if (overflowIndex != 0xffffffffu)
                sWork[overflowIndex] = workIndex;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (workCount != 0u)
    {
        const bool has = tid < workCount;
        const uint queued = has ? sWork[tid] : 0u;
        psPrimProcessCell(queued, has, tid, warpId, lane, isTGS);
    }
}

// ===========================================================================
// P9.13g -- ps_primitivesDiffuseCollisionLaunch
// gpunarrowphase/src/CUDA/cudaParticleSystem.cu:1324 (wrapper) and :505-635
// (psDiffusePrimitivesCollision), both read in full.
//
// The DIFFUSE twin of the kernel above.  It shares this translation unit
// deliberately: the work-index binary search, calcGridHashInBounds,
// getSpecVolume and the whole particlePrimitiveCollision family are the SAME
// code, already swept, and duplicating them into a second module is exactly the
// duplication this port exists to avoid.  What differs is listed below, and the
// list is short.
//
// ---------------------------------------------------------------------------
// D1 -- the GRID is the diffuse one.  mDiffuseCellStart / mDiffuseCellEnd, not
//     mCellStart / mCellEnd.  The HASH is not: gridHash still comes from the
//     system's common cell width and grid size, so the two grids share an index
//     space and only their bucket arrays differ.  A port that also swaps the
//     hash source finds nothing at all.
//
// D2 -- the positions are the diffuse SORTED pair:
//     mDiffuseSortedOriginPos_LifeTime is `current` and
//     mDiffuseSortedPos_LifeTime is `predicted`.  Note P9.13e overloads the
//     FORMER with velocity averages between UpdatePBF and Compact -- but this
//     kernel runs in narrowphase, before either, so it sees real origins.
//
// D3 -- isDiffuse = TRUE in the speculative contact volume, which is NOT the
//     same as isTGS = false: see getSpecVolume above.
//
// D4 -- there is NO FILTER.  The fluid kernel calls findPs to honour
//     particle-rigid filter pairs and rigid attachments; the diffuse path has
//     neither and calls particlePrimitiveCollision unconditionally.
//
// D5 -- the whole kernel early-outs on mCommonData.mMaxDiffuseParticles == 0,
//     BEFORE it reads the cell arrays.  For a particle system with no diffuse
//     side those pointers are null.
//
// D6 -- the OUTPUT is per-particle, not a global contact list.  There is no
//     warp aggregation and no numTotalContacts: each intersection does its own
//     atomicAdd on counts[particleIndex] and, if the returned slot is below
//     MaxStaticContactsPerParticle, writes the normal/penetration, a ZEROED
//     impulse and the rigid node index at particleIndex + slot * numParticles.
//     P9.13f's consumer reads exactly that layout.
//
// D7 -- the atomicAdd runs on EVERY intersection and only the write is capped,
//     so the stored count overshoots and the consumer's clamp is what keeps the
//     read inside what was written.  The two halves of that contract are in two
//     different kernels and neither is complete alone.
//
// D8 -- numParticles is `*mNumDiffuseParticles`, slot [0] of the two-slot pair.
//
// DEVIATION, deliberate and observable-equivalent: the CUDA wrapper runs a
// block-wide WORK QUEUE -- ballot the work items whose cell is non-empty,
// warpScanExclusive them into a 256-entry __shared__ array, and flush the body
// only once a full block has accumulated, carrying an overflow index across
// iterations.  That is a scheduling device: every work item with a non-empty
// cell reaches psDiffusePrimitivesCollision exactly once, items with an empty
// cell reach it never (and the body would do nothing for them anyway), and the
// body has no cross-lane traffic at all.  So this port runs one thread per work
// item over the same grid-stride and calls the body directly.  What that costs
// is occupancy on divergent blocks, not correctness -- and the atomic ORDER was
// never determined in the first place (the P9.11 rule: the SET of contacts is
// asserted, never which slot each landed in).
// ===========================================================================

// PxgParticleContactInfo::MaxStaticContactsPerParticle (PxgParticleSystem.h:133)
static const uint PS_MAX_STATIC_CONTACTS = 12u;

[shader("compute")]
[numthreads(256, 1, 1)]
void ps_primitivesDiffuseCollisionLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
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
        if (workIndex >= totalComparison)
            continue;

        // ---- PxgCellData::update, identical to the fluid kernel's ----
        const uint pos = binarySearchRunSum(startIndices, gP.numTests, workIndex);
        const uint startPairIndex = startIndices[pos];

        const PxgContactManagerInput cin = cmInputs[pos];
        uint particleShapeRef = cin.shapeRef0;
        uint particleCacheRef = cin.transformCacheRef0;
        uint rigidShapeRef = cin.shapeRef1;
        uint rigidCacheRef = cin.transformCacheRef1;
        if (shapes[cin.shapeRef0].type != GEOM_PARTICLESYSTEM)
        {
            particleShapeRef = cin.shapeRef1;
            particleCacheRef = cin.transformCacheRef1;
            rigidShapeRef = cin.shapeRef0;
            rigidCacheRef = cin.transformCacheRef0;
        }

        const PxgShape rigidShape = shapes[rigidShapeRef];
        const uint rigidType = rigidShape.type;
        const float3 rigidScale = float3(rigidShape.scaleX, rigidShape.scaleY, rigidShape.scaleZ);
        const PxsCachedTransform tc = transformCache[rigidCacheRef];
        const float4 rigidQ = float4(tc.qx, tc.qy, tc.qz, tc.qw);
        const float3 rigidP = float3(tc.px, tc.py, tc.pz);
        const uint2 rigidId = shapeToRigidRemap[rigidCacheRef];

        const uint psId = shapes[particleShapeRef].particleOrSoftbodyId;

        const float cDistance = contactDistance[particleCacheRef] + contactDistance[rigidCacheRef];
        const float restDistance = restDistances[pos];

        const uint pb = particleCacheRef * 6u;
        const uint rb = rigidCacheRef * 6u;
        float3 omn = max(float3(boundsArr[pb + 0u], boundsArr[pb + 1u], boundsArr[pb + 2u]),
                         float3(boundsArr[rb + 0u], boundsArr[rb + 1u], boundsArr[rb + 2u]));
        float3 omx = min(float3(boundsArr[pb + 3u], boundsArr[pb + 4u], boundsArr[pb + 5u]),
                         float3(boundsArr[rb + 3u], boundsArr[rb + 4u], boundsArr[rb + 5u]));
        omn -= float3(cDistance, cDistance, cDistance);
        omx += float3(cDistance, cDistance, cDistance);

        // D1: the HASH still comes from the system's common grid.
        const float cellWidth = systems[psId].mCommonData_GridCellWidth;
        const uint3 gridSize = uint3(systems[psId].mCommonData_GridSizeX,
                                     systems[psId].mCommonData_GridSizeY,
                                     systems[psId].mCommonData_GridSizeZ);
        const uint cellOffset = workIndex - startPairIndex;
        const uint gridHash = calcGridHashInBounds(omn, omx, cellWidth, cellOffset, gridSize);

        // D5: before any cell-array read.
        if (systems[psId].mCommonData_MaxDiffuseParticles == 0u)
            continue;

        // D1: the DIFFUSE bucket arrays.
        const uint startIndex = Ptr<uint>(systems[psId].mDiffuseCellStart)[gridHash];
        if (startIndex == EMPTY_CELL)
            continue;
        const uint endIndex = Ptr<uint>(systems[psId].mDiffuseCellEnd)[gridHash];
        const uint range = endIndex - startIndex;
        if (range == 0u)
            continue;

        const bool enableCCD = (systems[psId].mData_Flags & PS_FLAG_ENABLE_CCD) > 0u;
        // D2: origin = current, sorted-pos = predicted.
        Ptr<float4> curPos = Ptr<float4>(systems[psId].mDiffuseSortedOriginPos_LifeTime);
        Ptr<float4> predPos = Ptr<float4>(systems[psId].mDiffuseSortedPos_LifeTime);

        // D6/D8: the per-particle output arrays and the published count.
        Ptr<float4> infos = Ptr<float4>(systems[psId].mDiffuseOneWayContactInfos);
        Ptr<float> impulses = Ptr<float>(systems[psId].mDiffuseOneWayForces);
        Ptr<uint2> nodeIndices = Ptr<uint2>(systems[psId].mDiffuseOneWayNodeIndex);
        Ptr<uint> counts = Ptr<uint>(systems[psId].mDiffuseOneWayContactCount);
        const uint numParticles = uint(Ptr<int>(systems[psId].mNumDiffuseParticles)[0]);

        for (uint ind = 0u; ind < range; ++ind)
        {
            const uint particleIndex = startIndex + ind;
            if (particleIndex >= endIndex)
                continue;

            const float3 currentPos = curPos[particleIndex].xyz;

            float3 cVolumePos;
            float cVolumeRadius;
            if (enableCCD)
            {
                const float3 predictedPos = predPos[particleIndex].xyz;
                // D3: isDiffuse = true.
                cVolumeRadius = getSpecVolume(cVolumePos, currentPos, predictedPos,
                                              cDistance, true, isTGS);
            }
            else
            {
                cVolumeRadius = cDistance;
                cVolumePos = currentPos;
            }

            float3 normal = float3(0, 0, 0);
            float distance = 0.0f;
            // D4: no filter lookup on this path.
            const bool intersect = particlePrimitiveCollision(currentPos, cVolumePos, cVolumeRadius,
                                                             rigidQ, rigidP, rigidType, rigidScale,
                                                             enableCCD, normal, distance);
            if (!intersect)
                continue;

            // D7: the counter always advances; only the write is capped.
            uint slot = 0u;
            InterlockedAdd(counts[particleIndex], 1u, slot);
            if (slot < PS_MAX_STATIC_CONTACTS)
            {
                const uint outputIndex = particleIndex + slot * numParticles;
                infos[outputIndex] = float4(normal, distance - restDistance);
                impulses[outputIndex] = 0.0f;
                nodeIndices[outputIndex] = rigidId;
            }
        }
    }
}

// ===========================================================================
// P9.14b -- ps_convexCollisionLaunch
// gpunarrowphase/src/CUDA/cudaParticleSystem.cu:1470 (wrapper), :807
// (psConvexCollision) and :666 (particleConvexCollision), all read in full.
//
// The particle-vs-CONVEX-HULL contact generator: the fourth and last of this
// file's family, and the only one whose contact test is GJK/EPA rather than a
// closed-form primitive query.  It shares the work-index search, the cell hash
// and the writer with the primitives kernel above, and it shares its GJK/EPA
// policy with the SOFTBODY path -- particleConvexCollision is
// tetPrimitivesCollide2 with a one-point cloud on side A, so it is composed
// from that module rather than transcribed a second time.
//
// ---------------------------------------------------------------------------
// X1 -- THE ENGINE PASSES isDiffuse = TRUE HERE, IN THE FLUID KERNEL.
//     getParticleSpeculativeContactVolume's fifth argument is `isDiffuse`, and
//     the four particle collision bodies pass:
//         psPrimitivesCollision        (:463)   false
//         psDiffusePrimitivesCollision (:610)   true
//         psConvexCollision            (:912)   TRUE     <-- the fluid path
//         psConvexDiffuseCollision     (:1078)  true
//     Since PARTICLE_FORWARD_PROJECTION_STEP_SCALE_DIFFUSE and _PGS are both
//     1.0 and _TGS is 0.5, this is observable only in a TGS step: the SAME
//     fluid particle, with the SAME CCD flag, gets a contact volume scaled by
//     1.0 against a hull and by 2.0 against a box.  That is an ENGINE
//     INCONSISTENCY -- upstream candidate, DOCUMENTED AND NOT REPORTED -- and
//     it is transcribed exactly, because the job of this port is to match the
//     engine and not to correct it.
//
// X2 -- `combine()` IS AN INTERSECTION.  cudaParticleSystem.cu:83 defines a
//     FILE-LOCAL combine that takes max of the minima and min of the maxima --
//     the overlap region.  PxBounds3::combine, whose name it borrows, is the
//     UNION.  The primitives kernel above already does the right thing; the
//     name is recorded here because it actively misleads.
//
// X3 -- ONE WARP PER WORK ITEM, AND THE WORK ASSIGNMENT IS NOT A GRID-STRIDE.
//         workIndex = i + (warpIndex + numWarpPerBlock * blockIdx.x) * nbIter
//     Each warp owns a CONTIGUOUS RUN of nbIter indices, where
//     nbIter = ceil(nbWarpsRequired / (1024 * 8)).  The primitives kernel's
//     `i * stride + blockIdx.x * PS_BLOCK + tid` covers the same SET but hands
//     different items to different threads; the two are interchangeable only
//     because the items are independent.  Transcribed as the engine spells it.
//
// X4 -- OUR GJK/EPA DRIVER IS PER-THREAD, so the CUDA's 32-lane cooperation
//     collapses: lane 0 of each warp does the whole work item and lanes 1..31
//     return immediately.  The block shape stays (32, 8, 1) because that is
//     what the engine dispatches, and warpIndex is threadIdx.y exactly as in
//     the .cu.  The per-warp shared EpaScratch, ParticleConvexCollideScratch
//     and the warpCopy'd PxgParticleSystem all disappear with it -- they are
//     CUDA staging for a cooperative query this port does not run
//     cooperatively.
//
// X5 -- inSphereRadius1 = extents.w * minScale, and minScale is 1.0 UNLESS the
//     shape is eBOX, where it is the smallest of the three scale components.
//     A convex mesh bakes its scale into the blob and upstream leaves the
//     factor at 1; a box's PxMeshScale IS its half-extent.  Same asymmetry the
//     softbody path documents.
//
// X6 -- CCD RUNS GJK TWICE, AND ONLY THE SECOND ANSWER SURVIVES.  The first
//     pass is a pure distance test at the contact-volume centre with
//     minSep = cVolumeRadius; if it is not DISTANT the cache is thrown away and
//     the whole query is re-run from the ACTUAL particle position with
//     minSep = contactDistance + 2*cVolumeRadius -- "make sure to cover the
//     larger volume that includes both the current and the whole cVolume".
//     The anomaly test and EPA run on the SECOND result only.  Composing this
//     as two tetPrimitivesCollide2 calls would run EPA on the first pass too.
//
// X7 -- THE STORED NORMAL IS NEGATED.  `outCP = threadIdx.x < 3 ? (-normal)
//     : pen`, i.e. normal_pen.xyz = -gjkOutput.direction.  The primitives
//     kernel stores its normal unflipped.  Two kernels, one contact stream,
//     opposite conventions -- and both are what the solver expects.
//
// X8 -- the penetration is assembled from THREE LANES:
//     `pen = (witnessB - witnessA) * normal` per component, then the three
//     components are shuffled together and restDistance subtracted ONCE.  That
//     is `dot(closestPointB - closestPointA, direction) - restDistance`, and
//     the summation order is lane 0 + lane 1 + lane 2, i.e. (x + y) + z, with
//     each lane product rounded before it crosses lanes.
//     P9.89 corrected this note.  It used to say "which is what `dot` lowers
//     to".  That holds only for a NoContraction-DECORATED OpDot.  On an
//     undecorated one llvmpipe evaluates x + (y + z).  That is a different
//     association, not a contraction into fmas: the association model missed 0
//     of 65536 random cases, and the port differed from the CUDA on 19346 of
//     them.  The old measurement of (x + y) + z came from a stamped fixture.
//     This .spv was unstamped until P9.89.  Now it is stamped, and the stamp is
//     what keeps `dot` in the lane order (see the P9.89 note at the imports).
//
// X9 -- the filter IS applied here, unlike the diffuse twin, and its particle
//     key is the UNSORTED index (mSortedToUnsortedMapping), not the sorted one
//     the contact record carries.
//
// X10 -- the atomicAdd on numTotalContacts ALWAYS runs and only the write is
//     capped by maxContacts -- the same overshoot contract as the one-way
//     count in P9.13f/g.
// ===========================================================================

// PxMeshScale::transform (PxMeshScale.h): a mesh scale is a scale along a ROTATED
// frame, so a vertex is rotated INTO that frame with quatRotate, scaled
// componentwise, and rotated BACK with quatRotateInv.  It is NOT `v * scale`
// unless the scale rotation is identity -- and writing the two rotations the
// other way round is a silent transpose that only shows on a non-identity
// scaleRot.  Same spelling as the softbody path's cgVertex2Shape.
float3 psVertex2Shape(float3 v, float3 scale, float4 rot)
{
    float3 t = quatRotate(rot, v);
    t.x *= scale.x;
    t.y *= scale.y;
    t.z *= scale.z;
    return quatRotateInv(rot, t);
}

// The cooked-hull decode and world-space staging, shared by BOTH convex entry
// points (psConvexCollision :807 and psConvexDiffuseCollision :1000 spell it out
// identically, down to the eBOX-only scale factor).  One spelling, two callers.
GjkCloud psStageHull(PxgShape rigidShape, float4 rigidQ, float3 rigidP,
                     out float inSphereRadius1)
{
    const uint64_t hullPtr = uint64_t(rigidShape.hullOrMeshOffset.x) |
                             (uint64_t(rigidShape.hullOrMeshOffset.y) << 32);
    const uint hullDesc = chHullDesc(hullPtr);
    const float4 extents = chExtents(hullPtr);
    // Upstream asserts nbVertices1 <= CONVEX_MAX_VERTICES_POLYGONS (64), which is
    // exactly GJK_MAX_CLOUD; clamped so a corrupt header cannot walk off the
    // cloud, as the softbody path does.
    const uint nbVerts = min(chNbVerts(hullDesc), GJK_MAX_CLOUD);

    const float3 scale = float3(rigidShape.scaleX, rigidShape.scaleY, rigidShape.scaleZ);
    const float4 scaleRot = float4(rigidShape.scaleRotX, rigidShape.scaleRotY,
                                   rigidShape.scaleRotZ, rigidShape.scaleRotW);
    // X5: 1.0 unless the shape is a BOX, whose PxMeshScale IS its half-extent.
    float minScale1 = 1.0f;
    if (rigidShape.type == GEOM_BOX)
        minScale1 = min(scale.z, min(scale.x, scale.y));
    inSphereRadius1 = extents.w * minScale1;

    GjkCloud hull;
    for (uint v = 0u; v < GJK_MAX_CLOUD; ++v) hull.v[v] = float3(0.0f, 0.0f, 0.0f);
    hull.n = nbVerts;
    // prepareVertices (cudaNpCommon.h:315): vertex space -> shape space -> world.
    for (uint v = 0u; v < nbVerts; ++v)
    {
        const float3 vv = chVertex(hullPtr, v).xyz;
        hull.v[v] = quatRotate(rigidQ, psVertex2Shape(vv, scale, scaleRot)) + rigidP;
    }
    return hull;
}

// particleConvexCollision (cudaParticleSystem.cu:666), composed.  Returns false
// only for DISTANT; everything else leaves its answer in `o`.
bool particleConvexCollision(float3 particlePos, float3 cVolumePos, float cVolumeRadius,
                             bool enableCCD, float contactDistance, float3 hullOrigin,
                             GjkCloud b, float inSphereRadius1, out GjkOut o)
{
    GjkCloud a;
    for (uint i = 0u; i < GJK_MAX_CLOUD; ++i) a.v[i] = float3(0.0f, 0.0f, 0.0f);
    a.n = 1u;
    a.v[0] = cVolumePos;

    float3 dir0 = float3(1.0f, 0.0f, 0.0f);
    const float3 aToB0 = cVolumePos - hullOrigin;
    // P9.89: cu:698 magnitudeSquared() > 0.  A sum of three squares is positive
    // exactly when one square is (NaN and inf behave the same either way), so this
    // test and its second-pass twin (cu:733) do not depend on the order.
    if (dot(aToB0, aToB0) > 0.0f) dir0 = aToB0;

    if (!enableCCD)
    {
        // X6: without CCD the first pass IS the query -- cVolumeRadius equals
        // the contact distance here, and the anomaly/EPA policy runs on it.
        const uint r = tetPrimitivesCollide2(a, b, dir0, cVolumeRadius,
                                             0.0f, inSphereRadius1, o);
        return r != GJK_DISTANT;
    }

    // X6, first pass: a DISTANCE TEST ONLY.  No anomaly test, no EPA -- the
    // CUDA discards this simplex before the second pass.
    float cached0[32];
    const uint r0 = gjkRun(a, b, dir0, cVolumeRadius, TETC_CONVERGENCE_RATIO, o, cached0);
    if (r0 == GJK_DISTANT)
        return false;

    a.v[0] = particlePos;
    float3 dir1 = float3(1.0f, 0.0f, 0.0f);
    const float3 aToB1 = particlePos - hullOrigin;
    if (dot(aToB1, aToB1) > 0.0f) dir1 = aToB1;

    const uint r1 = tetPrimitivesCollide2(a, b, dir1,
                                          contactDistance + cVolumeRadius * 2.0f,
                                          0.0f, inSphereRadius1, o);
    return r1 != GJK_DISTANT;
}

[shader("compute")]
[numthreads(32, 8, 1)]
void ps_convexCollisionLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    // X4: lane 0 owns the work item; the other 31 have nothing to do.
    if (gtid.x != 0u)
        return;
    const uint warpIndex = gtid.y;
    const bool isTGS = (gP.isTGSWord & 0xffu) != 0u;

    Ptr<PxgContactManagerInput> cmInputs = Ptr<PxgContactManagerInput>(gP.cmInputs);
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gP.shapes);
    Ptr<PxsCachedTransform> transformCache = Ptr<PxsCachedTransform>(gP.transformCache);
    Ptr<float> boundsArr = Ptr<float>(gP.bounds);
    Ptr<float> contactDistance = Ptr<float>(gP.contactDistance);
    Ptr<float> restDistances = Ptr<float>(gP.restDistances);
    Ptr<uint> startIndices = Ptr<uint>(gP.startIndices);
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gP.particleSystems);
    Ptr<uint2> shapeToRigidRemap = Ptr<uint2>(gP.shapeToRigidRemap);

    const uint nbWarpsRequired = Ptr<uint>(gP.totalNumPairs)[0];
    const uint totalNumWarps = PS_GRID * WARPS_PER_BLOCK;
    const uint nbIter = (nbWarpsRequired + totalNumWarps - 1u) / totalNumWarps;

    for (uint i = 0u; i < nbIter; ++i)
    {
        // X3: a contiguous run per warp, not a grid-stride.
        const uint workIndex = i + (warpIndex + WARPS_PER_BLOCK * gid.x) * nbIter;
        if (workIndex >= nbWarpsRequired)
            continue;

        const uint pos = binarySearchRunSum(startIndices, gP.numTests, workIndex);
        const uint startPairIndex = startIndices[pos];

        // LoadShapePair<ePARTICLESYSTEM>: the particle shape goes to slot 0.
        const PxgContactManagerInput cin = cmInputs[pos];
        uint particleShapeRef = cin.shapeRef0;
        uint particleCacheRef = cin.transformCacheRef0;
        uint rigidShapeRef = cin.shapeRef1;
        uint rigidCacheRef = cin.transformCacheRef1;
        if (shapes[cin.shapeRef0].type != GEOM_PARTICLESYSTEM)
        {
            particleShapeRef = cin.shapeRef1;
            particleCacheRef = cin.transformCacheRef1;
            rigidShapeRef = cin.shapeRef0;
            rigidCacheRef = cin.transformCacheRef0;
        }

        const PxgShape rigidShape = shapes[rigidShapeRef];
        const uint psId = shapes[particleShapeRef].particleOrSoftbodyId;

        const PxsCachedTransform tc = transformCache[rigidCacheRef];
        const float4 rigidQ = float4(tc.qx, tc.qy, tc.qz, tc.qw);
        const float3 rigidP = float3(tc.px, tc.py, tc.pz);
        const uint2 rigidId = shapeToRigidRemap[rigidCacheRef];

        const float cDistance = contactDistance[particleCacheRef] + contactDistance[rigidCacheRef];
        const float restDistance = restDistances[pos];

        // X2: `combine` is the OVERLAP region, then fattenFast(cDistance).
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
        const uint gridHash = calcGridHashInBounds(omn, omx, cellWidth,
                                                   workIndex - startPairIndex, gridSize);

        // ---- the hull, decoded and staged in world space once per work item ----
        float inSphereRadius1;
        const GjkCloud hull = psStageHull(rigidShape, rigidQ, rigidP, inSphereRadius1);

        // ---- the cell ----
        const uint startIndex = Ptr<uint>(systems[psId].mCellStart)[gridHash];
        if (startIndex == EMPTY_CELL)
            continue;
        const uint endIndex = Ptr<uint>(systems[psId].mCellEnd)[gridHash];
        const uint range = endIndex - startIndex;
        if (range == 0u)
            continue;

        const bool enableCCD = (systems[psId].mData_Flags & PS_FLAG_ENABLE_CCD) > 0u;
        Ptr<float4> curPos = Ptr<float4>(systems[psId].mSortedOriginPos_InvMass);
        Ptr<float4> predPos = Ptr<float4>(systems[psId].mSortedPositions_InvMass);
        Ptr<uint> sortedToUnsorted = Ptr<uint>(systems[psId].mSortedToUnsortedMapping);

        for (uint ind = 0u; ind < range; ++ind)
        {
            const uint particleIndex = startIndex + ind;

            // X9: the filter key is the UNSORTED index.
            if (findPs(systems, psId, rigidId, uint2(sortedToUnsorted[particleIndex], psId)))
                continue;

            const float3 currentPos = curPos[particleIndex].xyz;

            float3 cVolumePos;
            float cVolumeRadius;
            if (enableCCD)
            {
                const float3 predictedPos = predPos[particleIndex].xyz;
                // X1: isDiffuse = TRUE, in the FLUID kernel.  The engine's own
                // spelling; see the header.
                cVolumeRadius = getSpecVolume(cVolumePos, currentPos, predictedPos,
                                              cDistance, true, isTGS);
            }
            else
            {
                cVolumePos = currentPos;
                cVolumeRadius = cDistance;
            }

            GjkOut o;
            if (!particleConvexCollision(currentPos, cVolumePos, cVolumeRadius, enableCCD,
                                         cDistance, rigidP, hull, inSphereRadius1, o))
                continue;

            // X8 / X7.  P9.89: cudaParticleSystem.cu:963-964, shfl0 + shfl1 + shfl2
            // - restDistance = ((p0 + p1) + p2) - restDistance.  The stamp keeps
            // OpDot in that order (testPsConvexContactGen p89PenLaneOrder).
            const float pen = dot(o.closestPointB.xyz - o.closestPointA.xyz,
                                  o.direction.xyz) - restDistance;

            // X10: the counter always advances; only the write is capped.
            uint index;
            InterlockedAdd(Ptr<uint>(gP.w_numTotalContacts)[0], 1u, index);
            writeContact(index, float4(-o.direction.xyz, pen),
                         uint2(particleIndex, psId), rigidId);
        }
    }
}

// ===========================================================================
// P9.14c -- ps_convexDiffuseCollisionLaunch
// gpunarrowphase/src/CUDA/cudaParticleSystem.cu:1586 (wrapper) and :986-1130
// (psConvexDiffuseCollision), both read in full.
//
// The DIFFUSE twin of the kernel above, and the LAST kernel of the diffuse
// chain -- the last diffuse row in the manifest that still carried nullptr.
// It stands to ps_convexCollisionLaunch exactly as
// ps_primitivesDiffuseCollisionLaunch stands to ps_primitivesCollisionLaunch:
// same work-index run, same hull staging, same GJK/EPA policy, different grid,
// different positions, no filter, and a per-particle output instead of a
// global contact stream.  So it composes psStageHull and
// particleConvexCollision rather than transcribing either a third time, and
// the delta list below is the whole tranche.
//
// ---------------------------------------------------------------------------
// XD1 -- the GRID is the diffuse one: mDiffuseCellStart / mDiffuseCellEnd.  The
//     HASH is not -- it still comes from the system's common cell width and
//     grid size, so both grids share one index space.  Identical to D1.
//
// XD2 -- the positions are mDiffuseSortedOriginPos_LifeTime (current) and
//     mDiffuseSortedPos_LifeTime (predicted).  P9.13e overloads the FORMER with
//     velocity averages between UpdatePBF and Compact, but this kernel runs in
//     narrowphase, before either.  Identical to D2.
//
// XD3 -- there is NO FILTER.  The fluid twin calls findPs on the UNSORTED index
//     (X9); this path calls particleConvexCollision unconditionally.  Identical
//     to D4, and it is the difference that makes the two bodies diverge right
//     where they look most alike.
//
// XD4 -- the OUTPUT is the per-particle one-way list, not the global contact
//     stream: atomicAdd on counts[particleIndex], and if the returned slot is
//     below MaxStaticContactsPerParticle write the normal/penetration, a ZEROED
//     impulse and the rigid node index at particleIndex + slot * numParticles.
//     Column-major with a stride of numParticles -- P9.13f's consumer reads
//     exactly that, and P9.13g's producer writes exactly that.  The counter
//     always advances and only the write is capped (D7 / X10, the same
//     overshoot contract in a third kernel).
//
// XD5 -- numParticles is `*mNumDiffuseParticles`, slot [0] of the two-slot pair
//     P9.13c documented: the PUBLISHED count, not the staging count.
//
// XD6 -- THE EARLY-OUT IS A `return`, NOT A `continue`, AND IT IS IN THE
//     WRAPPER.  cudaParticleSystem.cu:1653, inside
//     `for (i = 0; i < nbIterationsPerWarps; ++i)`:
//
//         // If there is no diffuse particles or is not using PBD, skip this function
//         if (ss_particleSystem.mCommonData.mMaxDiffuseParticles == 0)
//             return;
//
//     The comment says "skip this function"; the statement abandons the WARP'S
//     WHOLE REMAINING RUN.  Because each warp owns a CONTIGUOUS run of
//     nbIterationsPerWarps work indices (X3) and consecutive indices belong to
//     whichever pair the run-sum puts them in, a warp that meets one
//     diffuse-less particle system drops every later work item in its run --
//     including items belonging to particle systems that DO have diffuse
//     particles.  It is unreachable while one iteration is enough for the whole
//     grid (nbWarpsRequired <= 1024*8 = 8192 cells), which is why no scene has
//     ever hit it; past that it silently loses contacts.
//
//     The diffuse PRIMITIVES twin does not have this shape: its guard sits in
//     the BODY (D5) and its wrapper's loop simply moves on.
//
//     ENGINE BUG CANDIDATE -- DOCUMENTED, NOT REPORTED.  Transcribed exactly:
//     this port matches the engine, including where the engine is wrong, and a
//     mutant that "fixes" it to `continue` must die.
//
// XD7 -- isDiffuse = TRUE in the speculative contact volume.  Here that is the
//     CONSISTENT choice; it is the FLUID twin that is odd (X1).
//
// XD8 -- the wrapper's early-out runs BEFORE the bounds and the grid hash are
//     computed, unlike D5 which sits after them in the body.  Both orders reach
//     the same set of cells; the placement is kept because the `return` above
//     makes it observable -- moving it later would change which work items in
//     the run are abandoned.
//
// X4 applies unchanged: our GJK/EPA driver is per-thread, so lane 0 of each
// warp does the whole work item and the per-warp shared EpaScratch,
// ParticleConvexCollideScratch and warpCopy'd PxgParticleSystem all disappear.
// The block shape stays (32, 8, 1) because that is what the engine dispatches.
// X2 (combine is the OVERLAP), X5 (the eBOX-only scale factor), X6 (CCD runs
// GJK twice and only the second answer survives), X7 (the stored normal is
// NEGATED) and X8 (the three-lane penetration is a dot product) are all
// identical here and live in psStageHull / particleConvexCollision.
// ===========================================================================

[shader("compute")]
[numthreads(32, 8, 1)]
void ps_convexDiffuseCollisionLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    // X4: lane 0 owns the work item; the other 31 have nothing to do.
    if (gtid.x != 0u)
        return;
    const uint warpIndex = gtid.y;
    const bool isTGS = (gP.isTGSWord & 0xffu) != 0u;

    Ptr<PxgContactManagerInput> cmInputs = Ptr<PxgContactManagerInput>(gP.cmInputs);
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gP.shapes);
    Ptr<PxsCachedTransform> transformCache = Ptr<PxsCachedTransform>(gP.transformCache);
    Ptr<float> boundsArr = Ptr<float>(gP.bounds);
    Ptr<float> contactDistance = Ptr<float>(gP.contactDistance);
    Ptr<float> restDistances = Ptr<float>(gP.restDistances);
    Ptr<uint> startIndices = Ptr<uint>(gP.startIndices);
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gP.particleSystems);
    Ptr<uint2> shapeToRigidRemap = Ptr<uint2>(gP.shapeToRigidRemap);

    const uint nbWarpsRequired = Ptr<uint>(gP.totalNumPairs)[0];
    const uint totalNumWarps = PS_GRID * WARPS_PER_BLOCK;
    const uint nbIter = (nbWarpsRequired + totalNumWarps - 1u) / totalNumWarps;

    for (uint i = 0u; i < nbIter; ++i)
    {
        // X3 again -- and here the CONTIGUITY is load-bearing, not merely the
        // engine's spelling: XD6's `return` throws away the rest of THIS warp's
        // run, so which items share a warp decides which ones are lost.
        const uint workIndex = i + (warpIndex + WARPS_PER_BLOCK * gid.x) * nbIter;
        if (workIndex >= nbWarpsRequired)
            continue;

        const uint pos = binarySearchRunSum(startIndices, gP.numTests, workIndex);
        const uint startPairIndex = startIndices[pos];

        // LoadShapePair<ePARTICLESYSTEM>: the particle shape goes to slot 0.
        const PxgContactManagerInput cin = cmInputs[pos];
        uint particleShapeRef = cin.shapeRef0;
        uint particleCacheRef = cin.transformCacheRef0;
        uint rigidShapeRef = cin.shapeRef1;
        uint rigidCacheRef = cin.transformCacheRef1;
        if (shapes[cin.shapeRef0].type != GEOM_PARTICLESYSTEM)
        {
            particleShapeRef = cin.shapeRef1;
            particleCacheRef = cin.transformCacheRef1;
            rigidShapeRef = cin.shapeRef0;
            rigidCacheRef = cin.transformCacheRef0;
        }

        const uint psId = shapes[particleShapeRef].particleOrSoftbodyId;

        // XD6/XD8: the wrapper's early-out, before the bounds and the hash, and
        // spelled `return` exactly as the engine spells it.  NOT a `continue`.
        if (systems[psId].mCommonData_MaxDiffuseParticles == 0u)
            return;

        const PxgShape rigidShape = shapes[rigidShapeRef];

        const PxsCachedTransform tc = transformCache[rigidCacheRef];
        const float4 rigidQ = float4(tc.qx, tc.qy, tc.qz, tc.qw);
        const float3 rigidP = float3(tc.px, tc.py, tc.pz);
        const uint2 rigidId = shapeToRigidRemap[rigidCacheRef];

        const float cDistance = contactDistance[particleCacheRef] + contactDistance[rigidCacheRef];
        const float restDistance = restDistances[pos];

        // X2 again: `combine` is the OVERLAP region, then fattenFast(cDistance).
        const uint pb = particleCacheRef * 6u;
        const uint rb = rigidCacheRef * 6u;
        float3 omn = max(float3(boundsArr[pb + 0u], boundsArr[pb + 1u], boundsArr[pb + 2u]),
                         float3(boundsArr[rb + 0u], boundsArr[rb + 1u], boundsArr[rb + 2u]));
        float3 omx = min(float3(boundsArr[pb + 3u], boundsArr[pb + 4u], boundsArr[pb + 5u]),
                         float3(boundsArr[rb + 3u], boundsArr[rb + 4u], boundsArr[rb + 5u]));
        omn -= float3(cDistance, cDistance, cDistance);
        omx += float3(cDistance, cDistance, cDistance);

        // XD1, first half: the HASH still comes from the system's common grid.
        const float cellWidth = systems[psId].mCommonData_GridCellWidth;
        const uint3 gridSize = uint3(systems[psId].mCommonData_GridSizeX,
                                     systems[psId].mCommonData_GridSizeY,
                                     systems[psId].mCommonData_GridSizeZ);
        const uint gridHash = calcGridHashInBounds(omn, omx, cellWidth,
                                                   workIndex - startPairIndex, gridSize);

        // ---- the hull, decoded and staged in world space once per work item ----
        float inSphereRadius1;
        const GjkCloud hull = psStageHull(rigidShape, rigidQ, rigidP, inSphereRadius1);

        // ---- XD1: the DIFFUSE bucket arrays ----
        const uint startIndex = Ptr<uint>(systems[psId].mDiffuseCellStart)[gridHash];
        if (startIndex == EMPTY_CELL)
            continue;
        const uint endIndex = Ptr<uint>(systems[psId].mDiffuseCellEnd)[gridHash];
        const uint range = endIndex - startIndex;
        if (range == 0u)
            continue;

        const bool enableCCD = (systems[psId].mData_Flags & PS_FLAG_ENABLE_CCD) > 0u;
        // XD2: origin = current, sorted-pos = predicted.
        Ptr<float4> curPos = Ptr<float4>(systems[psId].mDiffuseSortedOriginPos_LifeTime);
        Ptr<float4> predPos = Ptr<float4>(systems[psId].mDiffuseSortedPos_LifeTime);

        // XD4/XD5: the per-particle output arrays and the published count.
        Ptr<float4> infos = Ptr<float4>(systems[psId].mDiffuseOneWayContactInfos);
        Ptr<float> impulses = Ptr<float>(systems[psId].mDiffuseOneWayForces);
        Ptr<uint2> nodeIndices = Ptr<uint2>(systems[psId].mDiffuseOneWayNodeIndex);
        Ptr<uint> counts = Ptr<uint>(systems[psId].mDiffuseOneWayContactCount);
        const uint numParticles = uint(Ptr<int>(systems[psId].mNumDiffuseParticles)[0]);

        for (uint ind = 0u; ind < range; ++ind)
        {
            const uint particleIndex = startIndex + ind;

            // XD3: no filter lookup on this path.
            const float3 currentPos = curPos[particleIndex].xyz;

            float3 cVolumePos;
            float cVolumeRadius;
            if (enableCCD)
            {
                const float3 predictedPos = predPos[particleIndex].xyz;
                // XD7: isDiffuse = true.
                cVolumeRadius = getSpecVolume(cVolumePos, currentPos, predictedPos,
                                              cDistance, true, isTGS);
            }
            else
            {
                cVolumePos = currentPos;
                cVolumeRadius = cDistance;
            }

            GjkOut o;
            if (!particleConvexCollision(currentPos, cVolumePos, cVolumeRadius, enableCCD,
                                         cDistance, rigidP, hull, inSphereRadius1, o))
                continue;

            // X8 / X7: the dot product, and the NEGATED normal.  P9.89:
            // cudaParticleSystem.cu:1116-1117, ((p0 + p1) + p2) - restDistance.  The
            // stamp keeps OpDot in that order (psConvexDiffuseContactGen_p89PenLaneOrder).
            const float pen = dot(o.closestPointB.xyz - o.closestPointA.xyz,
                                  o.direction.xyz) - restDistance;

            // XD4: the counter always advances; only the write is capped.
            uint slot = 0u;
            InterlockedAdd(counts[particleIndex], 1u, slot);
            if (slot < PS_MAX_STATIC_CONTACTS)
            {
                const uint outputIndex = particleIndex + slot * numParticles;
                infos[outputIndex] = float4(-o.direction.xyz, pen);
                impulses[outputIndex] = 0.0f;
                nodeIndices[outputIndex] = rigidId;
            }
        }
    }
}
