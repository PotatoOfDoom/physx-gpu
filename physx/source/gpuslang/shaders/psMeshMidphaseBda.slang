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

// psMeshMidphaseBda.slang
//
// P9.10 -- ps_meshCollisonLaunch (particleSystemMeshMidphase.cu:775, body
// psMeshMidphaseCollision at :613-741) on the engine BDA ABI: the particle-vs-
// TRIANGLE-MESH midphase and contact generation.  This is the kernel that fills
// mOneWayContactInfos / mOneWayContactCount -- the arrays P9.9 proved
// ps_accumulateStaticDensity reads and, until this lands, finds empty in every
// scene we can build.
//
// PROVENANCE, and it matters for how this file should be read.  The traverser
// body is NOT a second transcription of the CUDA.  shaders/psMeshMid.slang is a
// Phase-1 port of PsTreeContactGenTraverser with its own fixture
// (psMeshMid_traverser), and its header already lists exactly what it lacks: the
// engine glue.  This module lifts that body onto BDA and adds only the glue.
// Where the two files differ at all it is because BDA needs Ptr<> where the
// Phase-1 module has bound buffers -- the arithmetic is meant to be identical,
// and the fixture is required to prove it GPU-DIFFERENTIALLY (one arena, both
// entry points, byte compare), the P8.97d arrangement, rather than by a third
// reading of the CUDA.
//
// LAUNCH, measured at PxgNarrowphaseCore.cpp:3948-3968:
//   grid  (PxgParticleSystemKernelGridDim::PS_MESH_COLLISION = 16384, numTests, 2)
//   block (WARP_SIZE = 32, PS_MIDPHASE_COLLISION_WAPRS_PER_BLOCK = 1, 1)
// One warp per block, so every groupshared array below is a single warp's worth
// and needs no per-warp indexing.
//
// Args -> ABI {1,4,8,8,8,8,8,8,8,72}, 136-byte blob:
//   (bool isTGS, PxReal toleranceLength, cmInputs*, transformCache*,
//    contactDistance*, restDistances*, gpuShapes*, particleSystems*,
//    shapeToRigidRemapTable*, PxgParticleContactWriter BY VALUE)
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
//  M1. gridDim.Z == 2 IS THE DIFFUSE SPLIT.  blockIdx.z == 1 is the diffuse pass:
//      it reads mDiffuseSortedOriginPos_LifeTime / mDiffuseSortedPos_LifeTime,
//      writes the mDiffuse* one-way arrays, takes its particle count from
//      *mNumDiffuseParticles, and is SUPPRESSED at the writer.  It also returns
//      early when mMaxDiffuseParticles == 0.  A port that ignores gridDim.Z runs
//      the solid pass twice and doubles every contact.
//      NOTE that "suppressed at the writer" means the writeContact LOOP only --
//      see CANDIDATE 19 below, which is where the suppression stops.
//  M2. `toleranceLength` is an ARGUMENT AND IS NEVER USED by this kernel body.
//      It is in the signature (and therefore in the ABI blob) and read by the SDF
//      twin only.  Transcribed as an unused field rather than dropped, because
//      dropping it would shift every later offset in the blob.
//  M3. contactDist is NOT divided by the smallest scale component here.  The
//      softbody twin (softBodyMeshMidphaseBda) DOES divide, because it compares a
//      world distance against vertex-space separations.  This kernel instead
//      converts the box delta BACK to shape space (vertex2Shape) before comparing,
//      so the division would be wrong twice over.  Copying the softbody line is
//      the obvious mistake and it gets a mutant.
//  M4. The cooked-mesh header is uint4{nbVerts, nbTris, nbBv32TreeNodes,
//      nbPackedNodes} and only .w sizes the NODE ARRAY.  .z is a different count.
//      Sizing the vertex pointer with .z walks past the nodes and reads garbage
//      triangles.
//  M5. The active-particle mask is taken ONCE per work index and NEVER refined,
//      faithfully to the CUDA: intersectBoxFullWarp computes a refined mask but
//      returns only "any" (the scratch's sBv32ActiveParticles stack is unused
//      legacy upstream).
//  M6. The per-particle contact list caps at MaxStaticContactsPerMesh = 6 with a
//      SERIALIZED WORST-CONTACT REPLACEMENT for the overflow, and the owning
//      lane's count SATURATES at 6 after any overflow round -- it does not keep
//      counting.  The one-way append then caps again at
//      MaxStaticContactsPerParticle = 12.
//  M7. The one-way output index is `globalParticleIndex + i * numParticles` --
//      COLUMN-major with a numParticles stride.  That is exactly the layout
//      ps_accumulateStaticDensity reads (P9.8 trap B2); the two tranches were
//      written independently and agree.
//
// ---------------------------------------------------------------------------
// TWO DELIBERATE DIVERGENCES FROM THE CUDA -- upstream candidates 17 and 18.
// Both are documented here rather than silently "fixed", per the porting rule.
// ---------------------------------------------------------------------------
//  D1 (CANDIDATE 17).  The CUDA reads
//          gridParticleIndex[particleIndex]
//      with NO `particleIndex < numParticles` bound (:715); the traverser only
//      bounds it afterwards.  grid_particle_index is allocated at
//      maxParticles * sizeof(PxU32) (PxgPBDParticleSystemCore.cpp:1830) and the
//      last warp reads up to ceil(numParticles/32)*32 - 1, so the read runs past
//      the allocation exactly when
//          ceil(numParticles / 32) * 32 > maxParticles
//      i.e. when the system is FULL and its capacity is not a multiple of 32.
//      There is a SECOND way the same line goes out of bounds and it is worse:
//      in the diffuse pass numParticles is *mNumDiffuseParticles but the array is
//      still the SOLID mSortedToUnsortedMapping, so the read is bounded by the
//      diffuse count against a solid-sized allocation.
//      This cannot be transcribed literally: P8.96c's fifth platform rule is
//      that an unclamped BDA load really does fault on llvmpipe.  The port
//      therefore BRANCH-GUARDS the load with `!isDiffuse && idx < numParticles`,
//      which is P8.96c's safe form twice over (branch-guarded, and a divergent
//      gather rather than the uniform scalar load that speculates).
//      THE GUARD CANNOT CHANGE ANY OUTPUT, which is why it is a guard and not a
//      clamp-and-diverge: numParticles <= maxParticles in the solid pass, so
//      every out-of-bounds index also satisfies idx >= numParticles, and such a
//      lane is outside the constructor's active ballot and outside finalize's
//      `globalParticleIndex < numParticles` guard -- it contributes nothing
//      whatever the filter says.  In the diffuse pass the value is never
//      consumed at all (`find` runs only when !isDiffuse).  So the guarded and
//      unguarded programs agree on every byte written, under D2's model of what
//      a returned lane does to the collective ops.
//  D2 (CANDIDATE 18).  The CUDA's filter does
//          if (!isDiffuse && find(...)) return;
//      a DIVERGENT return standing in front of warp-cooperative code that runs
//      __ballot_sync(FULL_MASK, ...).  CUDA requires every thread named in the
//      mask to participate, so a filtered lane can corrupt the ballots for the
//      OTHER particles of the same warp.  Here the warp is emulated over
//      GROUP-WIDE BARRIERS, where an early return would deadlock rather than
//      merely mis-mask.  The port therefore carries a STICKY per-lane `laneDone`
//      flag: a filtered lane stays in every barrier and ballot but is excluded
//      from the active mask and contributes no contacts, and -- like the CUDA's
//      return -- it stays out for ALL REMAINING iterations, not just the current
//      one.  That reproduces the return's effect on the lane's own coverage while
//      leaving the collective operations well-formed.
//
// ---------------------------------------------------------------------------
// A THIRD UPSTREAM CANDIDATE, TRANSCRIBED AND NOT FIXED.
// ---------------------------------------------------------------------------
//  CANDIDATE 19.  finalizeFullWarp's writer tail reserves before it suppresses:
//          if (idxInWarp == 31)
//              startIndex = atomicAdd(writer.numTotalContacts, count);   // :345
//          ...
//          if (remaining && !isDiffuseParticlesThread)                   // :356
//              ... writer.writeContact(...)
//      The RESERVATION has no isDiffuseParticlesThread guard; only the write does.
//      So the diffuse pass advances the global primitive-contact counter by
//      everything it found and then writes none of it, leaving HOLES in
//      particleContacts (and in all six sibling arrays) beneath a count that
//      claims they are contacts.  Whatever those slots held is what downstream
//      sorts and solves.
//      Reachable whenever a scene has diffuse particles AND a triangle mesh on a
//      non-static actor -- for a STATIC mesh most contacts are consumed into the
//      one-way arrays first, so `remaining` is usually zero and the reservation is
//      small; for a dynamic one `id` stays 0 and the whole contact count is
//      reserved.  Measured in psMeshMidphaseDiffuse (c): with the diffuse
//      positions aliased onto the solid ones the counter lands on exactly twice the
//      solid-only figure and exactly half the reserved slots are never written.
//      TRANSCRIBED AS IS.  The port reproduces the reservation bug-for-bug and the
//      fixture asserts the faithful numbers, so a port that "fixed" it by guarding
//      the atomicAdd fails.
//
// The module is in PXG_NOCONTRACT_SHADERS: the triangle tests are dense float
// work and FMA contraction moves the results off the Phase-1 module's bit
// pattern, which would defeat the differential fixture.

import pxgSlangCommon;
import pxgSlangParticleDesc;
import pxgSlangBv32;   // P8.25: the Gu::BV32DataPacked reader, shared
// P9.12g: the contact writer, the rigid element filter, LoadShapePair and the
// two records they move -- extracted here when ps_sdfMeshCollisonLaunch became
// the fourth caller.  What used to sit in this file is now one import.
import pxgSlangParticleContact;

// ---------------------------------------------------------------------------
// Constants, all pinned by the fixture's layout test.
// ---------------------------------------------------------------------------
static const uint WARP = 32u;
static const uint MIDPHASE_WARPS_PER_BLOCK = 1u;   // PS_MIDPHASE_COLLISION_WAPRS_PER_BLOCK
static const uint PS_MESH_COLLISION_GRID = 16384u; // PxgParticleSystemKernelGridDim
static const uint TOTAL_NUM_WARPS = PS_MESH_COLLISION_GRID * MIDPHASE_WARPS_PER_BLOCK;

static const uint STACK_SIZE = 192u;

static const uint MAX_MESH_CONTACTS = 6u;          // MaxStaticContactsPerMesh
static const uint MAX_PARTICLE_CONTACTS = 12u;     // MaxStaticContactsPerParticle

static const uint GEOM_PARTICLESYSTEM = 6u;        // PxGeometryType::ePARTICLESYSTEM
static const uint GEOM_TRIANGLEMESH = 8u;          // PxGeometryType::eTRIANGLEMESH

static const float PX_MAX_F32 = 3.4028235e38f;
static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_PGS = 1.0f;
static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_TGS = 0.5f;
static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_DIFFUSE = 1.0f;

// PxParticleFlag::eENABLE_SPECULATIVE_CCD (PxPBDParticleSystem.h:185).  It is
// 1 << 3, NOT 1 << 2 -- 1 << 2 is eFULL_DIFFUSE_ADVECTION.  The draft of this
// file had 1 << 2 and psMeshMidphaseLayout caught it; psBoundsBda and
// psContactGenBda both already spell 8.
static const uint PX_PARTICLE_FLAG_SPECULATIVE_CCD = 8u;

// ---------------------------------------------------------------------------
// Push constants -- the ten CUDA arguments, writer inlined by value.
// ---------------------------------------------------------------------------
struct PsMeshMidPush
{
    uint isTGSByte;                 // @0   CUDA bool, one byte; read masked
    float toleranceLength;          // @4   M2: present in the ABI, unused here
    uint64_t cmInputs;              // @8
    uint64_t transformCache;        // @16
    uint64_t contactDistance;       // @24
    uint64_t restDistances;         // @32
    uint64_t gpuShapes;             // @40
    uint64_t particleSystems;       // @48
    uint64_t shapeToRigidRemapTable;// @56
    // PxgParticleContactWriter, BY VALUE (PxgParticleSystem.h:604), 72 bytes @64
    uint64_t w_particleContacts;            // @64
    uint64_t w_numTotalContacts;            // @72
    uint64_t w_contactSortedByParticle;     // @80
    uint64_t w_tempContactByParticle;       // @88
    uint64_t w_contactIndexSortedByParticle;// @96
    uint64_t w_contactByRigid;              // @104
    uint64_t w_tempContactByRigid;          // @112
    uint64_t w_contactIndexSortedByRigid;   // @120
    uint w_maxContacts;                     // @128 -> 136
}
[[vk::push_constant]]
ConstantBuffer<PsMeshMidPush> gMm;

// PxgShape (PxgConvexConvexShape.h, 48 B) comes from pxgSlangCommon.  Three of
// the older BDA midphases carry private copies; this one does not need to.

// ---------------------------------------------------------------------------
// Groupshared: ONE warp per block, so one slot each.
// ---------------------------------------------------------------------------
groupshared int sStack[STACK_SIZE];
groupshared uint sMask;
groupshared uint sStart;
groupshared float3 sBcastP;    // broadcast particle pos (shape space)
groupshared float3 sBcastCvp;  // broadcast volume pos (shape space)
groupshared float3 sBcastCvv;  // broadcast volume pos (vertex space)
groupshared float sBcastR;     // broadcast volume radius / depth
groupshared uint sBcastCnt;    // broadcast contact count
groupshared float3 shLocalContacts[WARP][MAX_MESH_CONTACTS];
groupshared float shLocalContactDepths[WARP][MAX_MESH_CONTACTS];
groupshared float sRed[WARP];
groupshared uint sScan[WARP];

uint ballot32(bool pred, uint lane)
{
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) sMask = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sMask, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sMask;
}

// ---------------------------------------------------------------------------
// Geometry, lifted verbatim from psMeshMid.slang (Phase 1, psMeshMid_traverser).
// ---------------------------------------------------------------------------
float3 vertex2Shape(float3 v, float3 scale, float4 rotation)
{
    return quatRotateInv(rotation, quatRotate(rotation, v) * scale);
}

float3 shape2Vertex(float3 v, float3 scale, float4 rotation)
{
    return quatRotateInv(rotation, quatRotate(rotation, v) / scale);
}

float getParticleSpeculativeContactVolume(out float3 cVolumePos,
    float3 currentPos, float3 predictedPos, float contactDist, bool isDiffuse, bool isTGS)
{
    const float sScaleInvPGS = 1.0f / PARTICLE_FORWARD_PROJECTION_STEP_SCALE_PGS;
    const float sScaleInvTGS = 1.0f / PARTICLE_FORWARD_PROJECTION_STEP_SCALE_TGS;
    const float sScaleInvDiffuse = 1.0f / PARTICLE_FORWARD_PROJECTION_STEP_SCALE_DIFFUSE;
    float scaleInv;
    if (isDiffuse)
        scaleInv = sScaleInvDiffuse;
    else
        scaleInv = isTGS ? sScaleInvTGS : sScaleInvPGS;
    const float3 cVolumeOffset = (predictedPos - currentPos) * scaleInv * 0.5f;
    cVolumePos = currentPos + cVolumeOffset;
    return length(cVolumeOffset) + contactDist;
}

float3 closestPtPointTriangle2(float3 p, float3 a, float3 b, float3 c, float3 ab, float3 ac)
{
    const float3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) { const float v = d1 / (d1 - d3); return a + v * ab; }
    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) { const float w = d2 / (d2 - d6); return a + w * ac; }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    { const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return b + w * (c - b); }
    const float denom = 1.0f / (va + vb + vc);
    return a + ab * (vb * denom) + ac * (vc * denom);
}

float distancePointTriangleSquared(float3 p, float3 a, float3 b, float3 c, out float3 closestP)
{
    const float3 ab = b - a;
    const float3 ac = c - a;
    closestP = closestPtPointTriangle2(p, a, b, c, ab, ac);
    const float3 v = p - closestP;
    return dot(v, v);
}

bool particleTriangleTest(inout float3 normal, inout float distance,
                          float3 particlePos, float3 cVolumePos, float cVolumeRadius,
                          float3 triV0, float3 triV1, float3 triV2, bool enableCCD)
{
    const float3 cn = cross(triV1 - triV0, triV2 - triV0);
    const float m = dot(cn, cn);
    const float3 triNormal = m > 0.0f ? cn * rsqrt(m) : float3(0, 0, 0);
    const float cVolumeRadiusSq = cVolumeRadius * cVolumeRadius;

    const float cVolumePosPlaneDist = dot(cVolumePos - triV0, triNormal);
    float3 closestPos;
    if (cVolumePosPlaneDist < cVolumeRadius)
    {
        float distSq = distancePointTriangleSquared(cVolumePos, triV0, triV1, triV2, closestPos);
        if (distSq <= cVolumeRadiusSq)
        {
            const float particlePosPlaneDist = dot(particlePos - triV0, triNormal);
            if (particlePosPlaneDist > 0.0f)
            {
                if (enableCCD)
                    distSq = distancePointTriangleSquared(particlePos, triV0, triV1, triV2, closestPos);
                const float3 distVec = particlePos - closestPos;
                normal = (distSq > 0.0f) ? distVec * rsqrt(distSq) : triNormal;
                distance = sqrt(distSq);
                return true;
            }
        }
    }
    return false;
}

// The packed BV32 node reader is pxgSlangBv32 (bvMin / bvMax / bvData /
// bvNbNodes / GU_BV4_CHILD_OFFSET_SHIFT_COUNT / BV32_STRIDE) -- the same one the
// four other midphases use.  Nothing about the node format is repeated here.

// The eight writer pointers and the capacity, out of this shader's push block
// and into the shared record pxgSlangParticleContact defines.
PsContactWriter mmWriter()
{
    PsContactWriter w;
    w.particleContacts = gMm.w_particleContacts;
    w.numTotalContacts = gMm.w_numTotalContacts;
    w.contactSortedByParticle = gMm.w_contactSortedByParticle;
    w.tempContactByParticle = gMm.w_tempContactByParticle;
    w.contactIndexSortedByParticle = gMm.w_contactIndexSortedByParticle;
    w.contactByRigid = gMm.w_contactByRigid;
    w.tempContactByRigid = gMm.w_tempContactByRigid;
    w.contactIndexSortedByRigid = gMm.w_contactIndexSortedByRigid;
    w.maxContacts = gMm.w_maxContacts;
    return w;
}

void writeContact(uint index, float4 normalPen, uint2 compressedParticleIndex, uint2 rigidId)
{
    psWriteContact(mmWriter(), index, normalPen, compressedParticleIndex, rigidId);
}

// LoadShapePair<eTRIANGLEMESH, ePARTICLESYSTEM>: the TRIANGLE MESH is the tested
// type here, so the trimesh half comes back first.  The heightfield twin tests
// the PARTICLE SYSTEM instead -- see pxgSlangParticleContact's header.
void loadTrimeshParticlePair(uint cmIdx,
                             out PxgShape triShape, out uint triCacheRef,
                             out PxgShape psShape, out uint psCacheRef)
{
    psLoadShapePair(gMm.cmInputs, gMm.gpuShapes, cmIdx, GEOM_TRIANGLEMESH,
                    triShape, triCacheRef, psShape, psCacheRef);
}

// ---------------------------------------------------------------------------
// The work-loop index arithmetic, as FUNCTIONS so the kernel and the coverage
// entry point below share one copy.  P9.9's review found a fixture that could
// not pin the very constant it existed to pin, because the constant only became
// observable at a size the fixture never reached; here the arithmetic is at
// least guaranteed to be the same arithmetic in both places, so the coverage
// test's kills are kills on the kernel.
// ---------------------------------------------------------------------------
uint psMeshNbWarpsRequired(uint numParticles) { return (numParticles + WARP - 1u) / WARP; }

uint psMeshNbIterations(uint numParticles)
{
    return (psMeshNbWarpsRequired(numParticles) + TOTAL_NUM_WARPS - 1u) / TOTAL_NUM_WARPS;
}

uint psMeshWorkIndex(uint it, uint warpIdx, uint blockIdxX, uint nbIterationsPerWarps)
{
    return it + (warpIdx + MIDPHASE_WARPS_PER_BLOCK * blockIdxX) * nbIterationsPerWarps;
}

// ===========================================================================
// ps_meshCollisonLaunch -- the kernel.  grid (16384, numTests, 2), block (32,1,1).
//
// Reading order below is the CUDA's: psMeshMidphaseCollision's glue (:613-712),
// then the work loop (:714-741) whose body is the traverser's constructor,
// bv32TreeTraversal and finalizeFullWarp -- all three lifted from
// psMeshMid.slang, which the differential fixture holds this file against.
// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void ps_meshCollisonLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const uint warpIdx = gtid.y;          // MIDPHASE_WARPS_PER_BLOCK == 1, so always 0
    const uint cmIdx = gid.y;             // blockIdx.y -- one contact pair per y-block
    const bool isDiffuse = (gid.z == 1u); // M1: gridDim.z == 2 is the diffuse split
    // The CUDA bool sits in one byte at blob offset 0 with three bytes of padding
    // behind it; read as a masked word (the P8.97d arrangement).
    const bool isTGS = (gMm.isTGSByte & 0xFFu) != 0u;

    // ---- LoadShapePair + the transform cache ----
    PxgShape triShape, psShape;
    uint triCacheRef, psCacheRef;
    loadTrimeshParticlePair(cmIdx, triShape, triCacheRef, psShape, psCacheRef);

    // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.
    const uint64_t xf = gMm.transformCache + uint64_t(triCacheRef) * 32UL;
    const float4 meshQ = Ptr<float4>(xf)[0];
    const float3 meshP = float3(Ptr<float>(xf + 16UL)[0],
                                Ptr<float>(xf + 20UL)[0],
                                Ptr<float>(xf + 24UL)[0]);

    // M3: NO division by the smallest scale component.  The box test converts its
    // delta back to SHAPE space before comparing, so this stays a world distance.
    const float contactDist = Ptr<float>(gMm.contactDistance)[triCacheRef] +
                              Ptr<float>(gMm.contactDistance)[psCacheRef];
    const float restDist = Ptr<float>(gMm.restDistances)[cmIdx];
    const uint2 rigidId = Ptr<uint2>(gMm.shapeToRigidRemapTable)[triCacheRef];

    const uint psId = psShape.particleOrSoftbodyId;
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gMm.particleSystems);

    // Uniform across the whole group (isDiffuse and psId are both block-uniform),
    // so this early-out cannot strand a lane at a barrier.
    if (isDiffuse && systems[psId].mCommonData_MaxDiffuseParticles == 0u)
        return;

    const float3 triScale = float3(triShape.scaleX, triShape.scaleY, triShape.scaleZ);
    const float4 triScaleRot = float4(triShape.scaleRotX, triShape.scaleRotY,
                                      triShape.scaleRotZ, triShape.scaleRotW);
    // MeshScaling::flipNormal -- the scale determinant's sign (convexMesh.cu:183).
    const bool flipNormal = (triScale.x * triScale.y * triScale.z) < 0.0f;

    // ---- readTriangleMesh (triangleMesh.cuh:120) ----
    const uint64_t meshPtr = uint64_t(triShape.hullOrMeshOffset.x) |
                             (uint64_t(triShape.hullOrMeshOffset.y) << 32);
    const uint4 meshHeader = Ptr<uint4>(meshPtr)[0];   // nbVerts, nbTris, nbBv32Nodes, nbPacked
    const uint64_t treeAddr = meshPtr + 16UL;
    // M4: .w is nbPackedNodes and sizes the node ARRAY.  .z is a different count.
    const uint64_t vertsAddr = treeAddr + uint64_t(meshHeader.w) * BV32_STRIDE;
    const uint64_t indicesAddr = vertsAddr + uint64_t(meshHeader.x) * 16UL;
    Ptr<float4> triVerts = Ptr<float4>(vertsAddr);
    Ptr<uint4> triIndices = Ptr<uint4>(indicesAddr);

    const bool enableCCD = (systems[psId].mData_Flags & PX_PARTICLE_FLAG_SPECULATIVE_CCD) > 0u;

    // ---- the diffuse / solid array selection (M1) ----
    uint64_t curPosAddr, predPosAddr;
    uint64_t oneWayInfosAddr, oneWayNodeIdxAddr, oneWayCountsAddr;
    uint numParticles;
    if (isDiffuse)
    {
        curPosAddr = systems[psId].mDiffuseSortedOriginPos_LifeTime;
        predPosAddr = systems[psId].mDiffuseSortedPos_LifeTime;
        numParticles = Ptr<uint>(systems[psId].mNumDiffuseParticles)[0];
        oneWayInfosAddr = systems[psId].mDiffuseOneWayContactInfos;
        oneWayNodeIdxAddr = systems[psId].mDiffuseOneWayNodeIndex;
        oneWayCountsAddr = systems[psId].mDiffuseOneWayContactCount;
    }
    else
    {
        curPosAddr = systems[psId].mSortedOriginPos_InvMass;
        predPosAddr = systems[psId].mSortedPositions_InvMass;
        numParticles = systems[psId].mCommonData_NumParticles;
        oneWayInfosAddr = systems[psId].mOneWayContactInfos;
        oneWayNodeIdxAddr = systems[psId].mOneWayNodeIndex;
        oneWayCountsAddr = systems[psId].mOneWayContactCount;
    }

    const uint nbWarpsRequired = psMeshNbWarpsRequired(numParticles);
    const uint nbIterationsPerWarps = psMeshNbIterations(numParticles);
    const uint64_t gridParticleIndexAddr = systems[psId].mSortedToUnsortedMapping;

    // D2: sticky, per lane, for ALL remaining iterations -- the CUDA's `return`.
    bool laneDone = false;

    for (uint it = 0u; it < nbIterationsPerWarps; ++it)
    {
        const uint workIndex = psMeshWorkIndex(it, warpIdx, gid.x, nbIterationsPerWarps);

        // Block-uniform: every lane of the group agrees, so the barriers below are
        // reached by all 32 lanes or by none.
        if (workIndex < nbWarpsRequired)
        {
            GroupMemoryBarrierWithGroupSync();   // the groupshared contact lists are reused

            const uint globalParticleIndex = lane + workIndex * WARP;

            // ---- the rigid filter, D1's guard and D2's sticky flag ----
            if (!isDiffuse && globalParticleIndex < numParticles && !laneDone)
            {
                const uint2 particleMask =
                    uint2(Ptr<uint>(gridParticleIndexAddr)[globalParticleIndex], psId);
                if (psFindFilter(systems, psId, rigidId, particleMask))
                    laneDone = true;
            }

            const bool laneLive = (globalParticleIndex < numParticles) && !laneDone;

            // ---- PsTreeContactGenTraverser's constructor ----
            float3 particlePosShapeSpace = float3(0, 0, 0);
            float3 cVolumePosShapeSpace = float3(0, 0, 0);
            float3 cVolumePosVertexSpace = float3(0, 0, 0);
            float cVolumeRadius = 0.0f;
            uint contactCount = 0u;

            if (laneLive)
            {
                const float3 currentPos = Ptr<float4>(curPosAddr)[globalParticleIndex].xyz;
                particlePosShapeSpace = quatRotateInv(meshQ, currentPos - meshP);
                if (enableCCD)
                {
                    const float3 predictedPos = Ptr<float4>(predPosAddr)[globalParticleIndex].xyz;
                    float3 cVolumePos;
                    cVolumeRadius = getParticleSpeculativeContactVolume(cVolumePos, currentPos,
                                        predictedPos, contactDist, isDiffuse, isTGS);
                    cVolumePosShapeSpace = quatRotateInv(meshQ, cVolumePos - meshP);
                }
                else
                {
                    cVolumePosShapeSpace = particlePosShapeSpace;
                    cVolumeRadius = contactDist;
                }
                cVolumePosVertexSpace = shape2Vertex(cVolumePosShapeSpace, triScale, triScaleRot);
            }

            // M5: set ONCE, never refined.
            const uint activeParticles = ballot32(laneLive, lane);

            // ---- bv32TreeTraversal ----
            uint currentNodeIndex = 0u;
            bool hasNode = true;
            uint nbStack = 0u;

            while (hasNode)
            {
                const uint nodeIdx = bvNodeIdxOf(currentNodeIndex);

                if (bvIsLeaf(currentNodeIndex))
                {
                    // ---- leaf: intersectPrimitiveFullWarp ----
                    const uint leafOffset = bvLeafSlotOf(currentNodeIndex);
                    const uint data = bvData(treeAddr, nodeIdx, leafOffset);
                    const uint nbPrimitives = bvDataNbPrimitives(data);
                    const uint primStartIndex = bvDataPrimStart(data);
                    uint primitiveIndex = 0xffffffffu;
                    // One lane per primitive: the six-bit field can say 63 and the
                    // traversal silently drops the rest.  `lane <` IS that clamp.
                    if (lane < nbPrimitives)
                        primitiveIndex = lane + primStartIndex;

                    float3 tv0 = float3(0, 0, 0), tv1 = float3(0, 0, 0), tv2 = float3(0, 0, 0);
                    if (primitiveIndex != 0xffffffffu)
                    {
                        const uint4 triIdx = triIndices[primitiveIndex];
                        tv0 = vertex2Shape(triVerts[triIdx.x].xyz, triScale, triScaleRot);
                        tv1 = vertex2Shape(triVerts[triIdx.y].xyz, triScale, triScaleRot);
                        tv2 = vertex2Shape(triVerts[triIdx.z].xyz, triScale, triScaleRot);
                        if (flipNormal)
                        {
                            const float3 t = tv1;
                            tv1 = tv2;
                            tv2 = t;
                        }
                    }

                    for (uint i = activeParticles; i != 0u; i &= (i - 1u))
                    {
                        const uint p = firstbitlow(i);
                        GroupMemoryBarrierWithGroupSync();
                        if (lane == p)
                        {
                            sBcastP = particlePosShapeSpace;
                            sBcastCvp = cVolumePosShapeSpace;
                            sBcastR = cVolumeRadius;
                            sBcastCnt = contactCount;
                        }
                        GroupMemoryBarrierWithGroupSync();
                        const float3 pPos = sBcastP;
                        const float3 pCvp = sBcastCvp;
                        const float pR = sBcastR;
                        const uint contactStartIndex = sBcastCnt;

                        bool intersect = false;
                        float3 normal = float3(0, 0, 0);
                        float distance = PX_MAX_F32;
                        if (primitiveIndex != 0xffffffffu)
                        {
                            float3 normalShapeSpace = float3(0, 0, 0);
                            intersect = particleTriangleTest(normalShapeSpace, distance, pPos, pCvp, pR,
                                                             tv0, tv1, tv2, enableCCD);
                            if (intersect)
                                normal = quatRotate(meshQ, normalShapeSpace);
                        }

                        const uint resultWarp = ballot32(intersect, lane);
                        const uint offset = countbits(resultWarp & ((1u << lane) - 1u));
                        const uint validCount = countbits(resultWarp);

                        if (lane == p)
                            contactCount += validCount;

                        bool failedWrite = false;
                        if (intersect)
                        {
                            const uint index = contactStartIndex + offset;
                            if (index < MAX_MESH_CONTACTS)
                            {
                                shLocalContacts[p][index] = normal;
                                shLocalContactDepths[p][index] = distance;
                            }
                            else
                                failedWrite = true;   // M6
                        }

                        const uint failMask = ballot32(failedWrite, lane);
                        if (failMask != 0u)
                        {
                            // M6: serialized worst-contact replacement.
                            float outSep = -PX_MAX_F32;
                            if (lane < MAX_MESH_CONTACTS)
                                outSep = shLocalContactDepths[p][lane];

                            for (uint f = failMask; f != 0u; f &= (f - 1u))
                            {
                                const uint failLane = firstbitlow(f);
                                GroupMemoryBarrierWithGroupSync();
                                if (lane == failLane)
                                    sBcastR = distance;   // reused as the depth broadcast
                                GroupMemoryBarrierWithGroupSync();
                                const float depth = sBcastR;

                                const uint candidateSwapMask = ballot32(depth < outSep, lane);
                                if (candidateSwapMask != 0u)
                                {
                                    // maxIndex over outSep; lowest lane wins ties.
                                    sRed[lane] = outSep;
                                    GroupMemoryBarrierWithGroupSync();
                                    uint bestIndex = 0u;
                                    {
                                        float best = -PX_MAX_F32;
                                        for (uint j = 0u; j < WARP; ++j)
                                            if (sRed[j] > best) { best = sRed[j]; bestIndex = j; }
                                    }
                                    GroupMemoryBarrierWithGroupSync();
                                    if (lane == failLane)
                                    {
                                        shLocalContacts[p][bestIndex] = normal;
                                        shLocalContactDepths[p][bestIndex] = distance;
                                    }
                                    GroupMemoryBarrierWithGroupSync();
                                    if (lane == bestIndex)
                                        outSep = depth;
                                }
                            }
                            // The owning lane SATURATES -- it does not keep counting.
                            if (lane == p)
                                contactCount = MAX_MESH_CONTACTS;
                        }
                    }
                }
                else
                {
                    // ---- internal node: intersectBoxFullWarp ----
                    const uint nbChildren = bvNbNodes(treeAddr, nodeIdx);

                    float3 mn = float3(0, 0, 0), mx = float3(0, 0, 0);
                    const bool hasBox = lane < nbChildren;
                    if (hasBox)
                    {
                        mn = bvMin(treeAddr, nodeIdx, lane);
                        mx = bvMax(treeAddr, nodeIdx, lane);
                    }

                    bool intersect = false;
                    for (uint i = activeParticles; i != 0u; i &= (i - 1u))
                    {
                        const uint p = firstbitlow(i);
                        GroupMemoryBarrierWithGroupSync();
                        if (lane == p)
                        {
                            sBcastCvv = cVolumePosVertexSpace;
                            sBcastR = cVolumeRadius;
                        }
                        GroupMemoryBarrierWithGroupSync();
                        const float3 sCvv = sBcastCvv;
                        const float sR = sBcastR;
                        if (hasBox)
                        {
                            const float3 closestVertexSpace = max(mn, min(mx, sCvv));
                            const float3 distVecVertexSpace = closestVertexSpace - sCvv;
                            // M3 again: the delta goes back to SHAPE space to be measured.
                            const float3 distVec = vertex2Shape(distVecVertexSpace, triScale, triScaleRot);
                            const float distSq = dot(distVec, distVec);
                            if (distSq < sR * sR)
                                intersect = true;
                        }
                    }

                    uint currentIndex = bvChildIndex(currentNodeIndex, lane);

                    bool isValidNode = false;
                    if (lane < nbChildren)
                    {
                        const uint data = bvData(treeAddr, nodeIdx, lane);
                        isValidNode = intersect;
                        if (isValidNode && bvDataIsInternal(data))
                            currentIndex = bvDataChildNodeIndex(data);
                    }

                    const uint resultWarp = ballot32(isValidNode, lane);
                    const uint offset = countbits(resultWarp & ((1u << lane) - 1u));
                    const uint validCount = countbits(resultWarp);

                    if (isValidNode)
                        sStack[nbStack + offset] = int(currentIndex);

                    nbStack += validCount;
                }

                GroupMemoryBarrierWithGroupSync();

                if (nbStack > 0u)
                {
                    nbStack -= 1u;
                    currentNodeIndex = uint(sStack[nbStack]);
                }
                else
                    hasNode = false;
            }

            // ---- finalizeFullWarp ----
            uint id = 0u;
            if (laneLive)
            {
                if (nodeIndexIsStaticBody(rigidId))
                {
                    if (contactCount != 0u)
                    {
                        uint startIndex;
                        InterlockedAdd(Ptr<uint>(oneWayCountsAddr)[globalParticleIndex],
                                       contactCount, startIndex);
                        // M6 again: the append caps at MaxStaticContactsPerParticle.
                        const uint endIndex = min(MAX_PARTICLE_CONTACTS, startIndex + contactCount);
                        for (uint k = startIndex; k < endIndex; ++k, ++id)
                        {
                            const float3 n = shLocalContacts[lane][id];
                            const float pen = shLocalContactDepths[lane][id] - restDist;
                            // M7: COLUMN-major, stride numParticles.
                            const uint outputIndex = globalParticleIndex + k * numParticles;
                            Ptr<float4>(oneWayInfosAddr)[outputIndex] = float4(n, pen);
                            Ptr<uint2>(oneWayNodeIdxAddr)[outputIndex] = rigidId;
                        }
                    }
                }
            }

            // Whatever did not fit (and everything, for a dynamic rigid) goes to the
            // shared writer: inclusive warp scan, ONE lane-31 atomicAdd, per-lane writes.
            const uint remaining = contactCount - id;
            const uint anyMask = ballot32(remaining != 0u, lane);
            if (anyMask != 0u)
            {
                sScan[lane] = remaining;
                GroupMemoryBarrierWithGroupSync();
                uint count = 0u;
                for (uint j = 0u; j <= lane; ++j)
                    count += sScan[j];
                uint total = count;
                for (uint j = lane + 1u; j < WARP; ++j)
                    total += sScan[j];
                GroupMemoryBarrierWithGroupSync();

                if (lane == WARP - 1u)
                {
                    uint s;
                    InterlockedAdd(Ptr<uint>(gMm.w_numTotalContacts)[0], total, s);
                    sStart = s;
                }
                GroupMemoryBarrierWithGroupSync();
                uint startIndex = sStart;

                if (laneLive)
                {
                    startIndex = startIndex + count - remaining;

                    // M1: the diffuse dispatch is SUPPRESSED at the writer.
                    if (remaining != 0u && !isDiffuse)
                    {
                        const uint2 compressedParticleIndex = uint2(globalParticleIndex, psId);
                        for (uint k = id, index = startIndex; k < contactCount; ++k, ++index)
                        {
                            const float3 n = shLocalContacts[lane][k];
                            const float pen = shLocalContactDepths[lane][k] - restDist;
                            writeContact(index, float4(n, pen), compressedParticleIndex, rigidId);
                        }
                    }
                }
            }
        }
    }
}

// ===========================================================================
// psMeshMidphaseWorkIndex -- TEST-ONLY, absent from the manifest.
//
// The kernel's work loop can only be exercised at nbIterationsPerWarps > 1 by a
// particle count above TOTAL_NUM_WARPS * 32 = 524288, and a fixture cannot afford
// the positions, the mesh and the twelve one-way columns for a million particles.
// So this entry point drives the SAME three functions with nothing else attached
// and records which work indices the loop covers, and how many times.
//
// At nbIterationsPerWarps == 1 the `* nbIterationsPerWarps` factor is an identity
// and pins nothing -- exactly the hole P9.9's review found in the boundary-density
// fixture -- which is why the test drives it at 1, 2 AND 3.
// ===========================================================================
struct PsMeshWorkPush
{
    uint64_t coverage;    // @0  uint* , one slot per work index
    uint numParticles;    // @8
    uint _pad;            // @12 -> 16
}
[[vk::push_constant]]
ConstantBuffer<PsMeshWorkPush> gMw;

[shader("compute")]
[numthreads(32, 1, 1)]
void psMeshMidphaseWorkIndex(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint numParticles = gMw.numParticles;
    const uint nbWarpsRequired = psMeshNbWarpsRequired(numParticles);
    const uint nbIterationsPerWarps = psMeshNbIterations(numParticles);

    for (uint it = 0u; it < nbIterationsPerWarps; ++it)
    {
        const uint workIndex = psMeshWorkIndex(it, gtid.y, gid.x, nbIterationsPerWarps);
        if (workIndex < nbWarpsRequired)
        {
            if (gtid.x == 0u)
            {
                uint prev;
                InterlockedAdd(Ptr<uint>(gMw.coverage)[workIndex], 1u, prev);
            }
        }
    }
}
