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

// psSdfMidphaseBda.slang -- P9.12h, ps_sdfMeshCollisonLaunch on the engine ABI.
//
// particleSystemMeshMidphase.cu:746 (the wrapper) over :375-600
// (psSdfMeshMidphaseCollision).  Composition, not transcription: the two halves
// that carry the work are already swept modules --
//
//   pxgSlangSdfFind      P9.12e  the resumable 1024-thread candidate gather
//   pxgSlangSdfParticle  P9.12f  the per-candidate contact decision
//
// -- and the plumbing under them is pxgSlangParticleContact (P9.12g).  What is
// new here is the glue: the shape-pair resolve, the SDF handed to both halves,
// the cull tolerance, the resume loop, the element filter, the static one-way
// write and the block-wide contact reservation.
//
// LAUNCH (PxgNarrowphaseCore.cpp:3901-3903): grid (numTests, 2, 1), block
// (1024, 1, 1).  blockIdx.X is the contact pair and blockIdx.Y selects the
// diffuse pass -- which is NOT the shape the two sibling particle midphases use
// (they are (grid, numTests, 2), pair on Y and diffuse on Z).  A port that
// copies their spelling reads the pair index out of the wrong dimension.
//
// ---------------------------------------------------------------------------
// THE TRAPS
// ---------------------------------------------------------------------------
//  W1. `tolerenceLength` is the FIRST parameter and is NEVER USED.  It stays in
//      the ABI row because the host still pushes it.
//  W2. `positions` is mSortedPositions_InvMass -- the PREDICTED buffer -- and
//      `positionsPrev` is mSortedOriginPos_InvMass.  The mesh midphase names the
//      same two buffers the other way round (:433-439 vs :663-671).  P9.12f's B1.
//  W3. The cull tolerance is `contactDist / scale.abs().minElement()` -- divided
//      by the SMALLEST absolute scale component, so a squashed mesh culls WIDER
//      in vertex space.  It is not contactDist, and it is not a per-axis value.
//  W4. The particle-to-mesh transform is the INVERSE of the cached mesh pose.
//      PxTransform::getInverse() is `PxTransform(q.rotateInv(-p), q.getConjugate())`
//      -- conjugate the quaternion AND rotate the negated translation by the
//      INVERSE, not by the original.
//  W5. THE FILTER AND THE CONTACT USE DIFFERENT PARTICLE INDICES.  The element
//      filter is queried with the UNSORTED index
//      (`gridParticleIndex[ind]`, :572) while the contact is written with the
//      SORTED one (`ind`, :601).  Both are packed the same way, which is what
//      makes swapping them silent.
//  W6. The filter is skipped for the diffuse pass, and so is the writer contact
//      -- but the static one-way write in between is NOT.  Diffuse particles get
//      one-way contacts and never reach the contact stream.
//  W7. The one-way counter is incremented UNCONDITIONALLY and only then tested
//      against the capacity, so it can end ABOVE it.  And unlike the heightfield
//      twin (which stores the clamped count back over the atomic), THIS kernel
//      leaves the over-count standing.  Transcribed; see the candidate note.
//  W8. `outputIndex = ind + startIndex * numParticles` -- COLUMN-major, stride
//      numParticles.
//  W9. `candidateContact` is cleared only when the one-way store ACTUALLY
//      happened.  Over capacity the contact falls through to the writer instead
//      of being dropped.
// W10. The writer normal is rotated to WORLD by the mesh pose; the one-way
//      normal above it is NOT -- it stays in mesh-local space.
// W11. globalScanExclusive<32> (reduction.cuh:673) is BLOCK-wide, not per-warp:
//      one threadBlockScanExclusive over all 1024 threads and ONE atomicAdd by
//      thread 0, with a __syncthreads() on each side.  Every thread must reach
//      it, including the ones whose slot is past nbFoundVertices -- on the
//      emulation path those barriers are workgroup barriers and stranding a
//      thread deadlocks the block.
//
// A DELIBERATE DIVERGENCE, the same one every fixture in this chain has made:
// the CUDA stages the cooked mesh and the initialised SDF texture through shared
// memory (thread 0 / threads 0..31 plus two __syncthreads()).  Here both are
// recomputed per thread from the read-only blob.  It is pure address arithmetic
// over immutable data, so the cost is the only difference -- and it removes two
// barriers and a shared-memory hazard from a kernel that already carries plenty.

#define PXG_WARP_BLOCK 1024
#include "pxgSlangWarpInc.slang"

import pxgSlangCommon;
import pxgSlangParticleDesc;
import pxgSlangParticleContact;
import pxgSlangSdfTexture;
import pxgSlangSdfCollision;
import pxgSlangSdfParticle;

#include "pxgSlangSdfFind.slang"

static const uint GEOM_TRIANGLEMESH = 8u;          // PxGeometryType::eTRIANGLEMESH
static const uint MAX_PARTICLE_CONTACTS = 12u;     // MaxStaticContactsPerParticle
// PxParticleFlag::eENABLE_SPECULATIVE_CCD is 1 << 3, not 1 << 2 (that is
// eFULL_DIFFUSE_ADVECTION) -- the same constant psHfMidphaseBda pins.
static const uint PX_PARTICLE_FLAG_SPECULATIVE_CCD = 8u;

// The kernel's nine arguments, in order.  The writer is passed BY VALUE and
// lands at 64 because the seven pointers before it stop at 60 and it is
// 8-aligned.  136 bytes total -- over Vulkan's 128-byte guarantee but inside
// llvmpipe's 256, exactly as the two sibling particle midphases already are.
struct PsSdfMidPush
{
    float tolerenceLength;                  // @0   W1: DEAD
    uint pad0;                              // @4
    uint64_t cmInputs;                      // @8
    uint64_t transformCache;                // @16
    uint64_t contactDistance;               // @24
    uint64_t restDistances;                 // @32
    uint64_t gpuShapes;                     // @40
    uint64_t particleSystems;               // @48
    uint64_t shapeToRigidRemapTable;        // @56
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
ConstantBuffer<PsSdfMidPush> gSd;

PsContactWriter sdWriter()
{
    PsContactWriter w;
    w.particleContacts = gSd.w_particleContacts;
    w.numTotalContacts = gSd.w_numTotalContacts;
    w.contactSortedByParticle = gSd.w_contactSortedByParticle;
    w.tempContactByParticle = gSd.w_tempContactByParticle;
    w.contactIndexSortedByParticle = gSd.w_contactIndexSortedByParticle;
    w.contactByRigid = gSd.w_contactByRigid;
    w.tempContactByRigid = gSd.w_tempContactByRigid;
    w.contactIndexSortedByRigid = gSd.w_contactIndexSortedByRigid;
    w.maxContacts = gSd.w_maxContacts;
    return w;
}

// W11: globalScanExclusive<NbWarps>, reduction.cuh:669-687.  This lived here as a
// private copy from P9.12h until P9.34c gave it a second consumer; it is now
// `sdfGlobalScanExclusive` in pxgSlangSdfFind, beside the block scan it is built on.
uint sdGlobalScanExclusive(uint tid, bool threadContributesElement)
{
    return sdfGlobalScanExclusive(tid, threadContributesElement, gSd.w_numTotalContacts);
}

// ===========================================================================
[shader("compute")]
[numthreads(1024, 1, 1)]
void ps_sdfMeshCollisonLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint cmIdx = gid.x;                 // blockIdx.x -- NOT .y (see the header)
    const bool isDiffuse = (gid.y == 1u);     // blockIdx.y

    PxgShape triShape, psShape;
    uint triCacheRef, psCacheRef;
    psLoadShapePair(gSd.cmInputs, gSd.gpuShapes, cmIdx, GEOM_TRIANGLEMESH,
                    triShape, triCacheRef, psShape, psCacheRef);

    // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.
    const uint64_t xf = gSd.transformCache + uint64_t(triCacheRef) * 32UL;
    const float4 meshQ = Ptr<float4>(xf)[0];
    const float3 meshP = float3(Ptr<float>(xf + 16UL)[0],
                                Ptr<float>(xf + 20UL)[0],
                                Ptr<float>(xf + 24UL)[0]);

    const float contactDist = Ptr<float>(gSd.contactDistance)[triCacheRef] +
                              Ptr<float>(gSd.contactDistance)[psCacheRef];
    const float restDist = Ptr<float>(gSd.restDistances)[cmIdx];
    const uint2 rigidId = Ptr<uint2>(gSd.shapeToRigidRemapTable)[triCacheRef];

    const uint psId = psShape.particleOrSoftbodyId;
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gSd.particleSystems);

    // Block-uniform, so it cannot strand a thread at a barrier below.
    if (isDiffuse && systems[psId].mCommonData_MaxDiffuseParticles == 0u)
        return;

    const uint64_t meshPtr = uint64_t(triShape.hullOrMeshOffset.x) |
                             (uint64_t(triShape.hullOrMeshOffset.y) << 32);
    PxgSdfMesh mesh;
    // The CUDA does not test this -- the host never routes a mesh without an SDF
    // to this kernel.  Returning is block-uniform and strictly safer than walking
    // into an uninitialised texture handle.
    if (!sdfReadMesh(meshPtr, mesh))
        return;
    const SparseSdfTexture tex = sdfTexInitialize(mesh);

    // W2: `positions` is the PREDICTED buffer here.
    uint64_t posAddr, posPrevAddr;
    uint64_t oneWayInfos, oneWayNodeIdx, oneWayCounts;
    uint numParticles;
    if (isDiffuse)
    {
        posAddr = systems[psId].mDiffuseSortedPos_LifeTime;
        posPrevAddr = systems[psId].mDiffuseSortedOriginPos_LifeTime;
        numParticles = Ptr<uint>(systems[psId].mNumDiffuseParticles)[0];
        oneWayInfos = systems[psId].mDiffuseOneWayContactInfos;
        oneWayNodeIdx = systems[psId].mDiffuseOneWayNodeIndex;
        oneWayCounts = systems[psId].mDiffuseOneWayContactCount;
    }
    else
    {
        posAddr = systems[psId].mSortedPositions_InvMass;
        posPrevAddr = systems[psId].mSortedOriginPos_InvMass;
        numParticles = systems[psId].mCommonData_NumParticles;
        oneWayInfos = systems[psId].mOneWayContactInfos;
        oneWayNodeIdx = systems[psId].mOneWayNodeIndex;
        oneWayCounts = systems[psId].mOneWayContactCount;
    }

    const uint64_t gridParticleIndex = systems[psId].mSortedToUnsortedMapping;
    const bool enableCCD = (systems[psId].mData_Flags & PX_PARTICLE_FLAG_SPECULATIVE_CCD) > 0u;

    const float3 scale = triShape.scale();
    const float4 scaleRot = triShape.scaleRotation();

    // W3: divided by the SMALLEST absolute scale component.
    float cullScale = contactDist;
    cullScale /= min(abs(scale.x), min(abs(scale.y), abs(scale.z)));

    // W4: PxTransform::getInverse() -- the conjugate, and the negated
    // translation rotated by the INVERSE.
    const float4 invQ = float4(-meshQ.x, -meshQ.y, -meshQ.z, meshQ.w);
    const float3 invP = quatRotateInv(meshQ, -meshP);

    // The resume loop, particleSystemMeshMidphase.cu:478.  Workgroup-uniform:
    // startIndex is the same value in every thread.
    uint startIndex = 0u;
    while (startIndex < numParticles)
    {
        const uint nbFoundVertices = sdfFindInterestingVertices(tid, numParticles, posAddr,
                                                                scale, scaleRot, cullScale,
                                                                tex, invP, invQ, startIndex);

        const uint ind = (tid < nbFoundVertices) ? gSdfIndices[tid] : numParticles;

        bool candidateContact = false;
        float3 normal = float3(0.0f, 0.0f, 0.0f);
        float finalSep = 0.0f;

        if (ind < numParticles)
        {
            // The P8.96c clamp rule: the guard is applied to the INDEX, because a
            // guarded uniform BDA load really does execute on llvmpipe.
            const float3 posWorld = Ptr<float4>(posAddr)[ind].xyz;
            const float3 posPrevWorld = Ptr<float4>(posPrevAddr)[ind].xyz;

            candidateContact = sdfParticleContact(tex, posWorld, posPrevWorld, invP, invQ,
                                                  scale, scaleRot, cullScale, contactDist,
                                                  restDist, enableCCD, normal, finalSep);

            // W5: the filter takes the UNSORTED index.  W6: never for diffuse.
            const uint2 particleMask = uint2(Ptr<uint>(gridParticleIndex)[ind], psId);
            if (!isDiffuse && psFindFilter(systems, psId, rigidId, particleMask))
                candidateContact = false;
        }

        if (nodeIndexIsStaticBody(rigidId))
        {
            if (candidateContact)
            {
                // W7: incremented first, tested after -- the count can exceed the
                // capacity, and nothing clamps it back.
                uint slot;
                InterlockedAdd(Ptr<uint>(oneWayCounts)[ind], 1u, slot);
                const uint outputIndex = ind + slot * numParticles;   // W8: column-major
                if (slot < MAX_PARTICLE_CONTACTS)
                {
                    // W10: mesh-local, NOT rotated to world.
                    Ptr<float4>(oneWayInfos)[outputIndex] = float4(normal, finalSep);
                    Ptr<uint2>(oneWayNodeIdx)[outputIndex] = rigidId;
                    candidateContact = false;    // W9: only when it was stored
                }
            }
        }

        // W11: EVERY thread, including the ones past nbFoundVertices.
        const uint contactIndex = sdGlobalScanExclusive(tid, candidateContact);

        if (candidateContact && !isDiffuse)
        {
            const float3 worldNormal = quatRotate(meshQ, normal);
            // W5 again: the contact carries the SORTED index.
            psWriteContact(sdWriter(), contactIndex, float4(worldNormal, finalSep),
                           uint2(ind, psId), rigidId);
        }
    }
}
