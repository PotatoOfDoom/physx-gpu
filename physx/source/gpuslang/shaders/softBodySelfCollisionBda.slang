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

// softBodySelfCollisionBda.slang
//
// P8.4 tranche a — the softbody SELF-collision midphase on the engine BDA ABI:
//
//   sb_selfCollisionMidphaseGeneratePairsLaunch (gpunarrowphase/src/CUDA/
//   softbodySoftbodyMidPhase.cu:959, launch PxgSoftBodyCore.cpp:411,
//   SB_SELFCOLLISION_MIDPHASE) — one warp per SURFACE vertex of one softbody walks that
//   same softbody's BV32 tetrahedron BVH, keeps the single deepest surviving
//   vertex-vs-tetrahedron contact, and appends it through PxgSoftBodyContactWriter.
//
// This is the first kernel of the softbody SELF-collision chain, the counterpart of the
// cloth path's cloth_selfCollisionMidphaseVTLaunch.  Without it a softbody passes through
// itself: fold a cube and the two halves interpenetrate freely.
//
// WHAT IS REUSED RATHER THAN REINVENTED:
//   - the TRAVERSER body is a LIFT of npSbSelf.slang (ctSbSelfQuery), the unit-tested
//     descriptor-set port of SbMeshTreeTraverser: the surface-hint gate, the shared-vertex
//     skip, the PxgNonRigidFilterPair lookup, isValidTet, isPointOutsideOfPlane4,
//     closestPtTetrahedron with the surface hint, the inside/outside contact decision, the
//     rest-pose re-check and the stress gate all come across verbatim, with the
//     RWStructuredBuffer reads replaced by Ptr<>.  Same relationship softBodyContactRemapBda
//     has to npSbRemap;
//   - the BV32 walk, the groupshared stack, the ballot/scan lowering and the barrier
//     placement come from softBodyContactGenBda / femClothSelfCollisionBda unchanged;
//   - PxgSoftBodyGpu is the byte-exact mirror the whole softbody port already shares.
//
// *** THE GRID.  READ THIS BEFORE CHANGING numthreads. ***
// The engine launches grid (SB_SBMIDPHASE = 1024, nbActiveSelfCollisionSoftbodies, 1) with
// block (WARP_SIZE, MIDPHASE_WARPS_PER_BLOCK = 2, 1) and then GRID-STRIDES over the
// softbody's vertices:
//
//     const PxU32 globalWarpIdx = threadIdx.y + blockIdx.x * blockDim.y;   // [0, 2048)
//     const PxU32 NbWarps       = blockDim.y * gridDim.x;                  // 2048
//     for (vertIdx = globalWarpIdx; vertIdx < numVerts; vertIdx += NbWarps)
//
// The bridge forwards gridDimX/Y/Z but DROPS blockDim (PxgVulkanCudaContext.cpp:292-331),
// so a [numthreads(32,1,1)] group is ONE warp.  Two things follow, and BOTH are needed:
//   - blockIdx.y is a real dimension here, so cmIdx = gid.y.  (femClothSelfCollisionBda is
//     the precedent -- this is not the first 2D grid in the port);
//   - the block's two warps are walked SERIALLY in a group-uniform `sub` loop, exactly as
//     in softBodyContactGenBda, and the grid stride stays the CUDA value 2048 rather than
//     the group count.  Deriving the stride from gid alone would either double-visit or
//     skip vertices; hard-coding SB_SBMIDPHASE * MIDPHASE_WARPS_PER_BLOCK reproduces the
//     CUDA partition exactly.  Missing that on the cloth side covered only half the
//     vertices and cost a long investigation.
//
// DEVIATIONS from CUDA (recipe policy, each preserving the emitted contact SET and the
// atomic total):
//   - NO 32-lane warp intrinsics.  __ballot_sync -> groupshared InterlockedOr;
//     warpScanExclusive -> countbits(mask & lanemask_lt); and finalizeFullWarp's
//     `minIndex(bestDist, FULL_MASK, sep)` -> a groupshared float[32] plus a serial scan
//     with strict `<`.  minIndex returns lowestSetIndex(ballot(minV == v)), i.e. the LOWEST
//     lane holding the minimum; a strict-`<` scan from lane 0 upward picks the same lane.
//     Ties matter: two tetrahedra at the same distance must not both write;
//   - the per-warp femMidphaseScratch (meshVerts / meshVertsIndices / bv32PackedNodes
//     cached by lane 0 behind a __syncthreads) is dropped; every lane reads those addresses
//     directly through Ptr<>.  Same values, one barrier fewer;
//   - mSurfaceVertsHint and mTetMeshSurfaceHint are PxU8 arrays.  llvmpipe has no 8-bit
//     storage in these shaders, so the containing 32-bit word is read and the byte
//     extracted.  NOTE the word address is formed with `& ~uint64_t(3)`: writing `& ~3UL`
//     MISCOMPILES -- Slang evaluates the complement in 32 bits and clears the top half of
//     the device address (documented at length in softBodySolveBda.slang);
//   - PxgSoftBodyContactWriter is passed BY VALUE as the last CUDA argument, so it is
//     INLINED into the push blob (@24..@75), giving an 80-byte blob.  Same treatment
//     PxgFEMContactWriter gets in softBodyContactGenBda.
//
// The tetmesh blob hanging off PxgSoftBody::mTetMeshData opens with a uint4 header
// (nbVerts, nbTets, maxDepth, nbBv32TreeNodes), so the BV32 node array starts at
// mTetMeshData + 16.
//
// NoContraction-stamped (PXG_NOCONTRACT_SHADERS): isValidTet's signed volume,
// isPointOutsideOfPlane4's four plane products and closestPtPointTriangle2's barycentric
// determinants are all dot-of-cross / mul-feeding-add chains that would otherwise fuse to
// FMAs and break parity with the -ffp-contract=off CPU reference.  getNormalized's rsqrt is
// the one non-portable operation, and it only picks a NORMAL direction -- the accept/reject
// decisions above are all comparisons of contracted-free products.

import pxgSlangSoftBodyDesc;   // PxgSoftBodyGpu (mTetMeshData@0, mNumVerts@364, ...)
import pxgSlangTetClosest;     // getNormalized / isValidTet / isPointOutsideOfPlane4 /
import pxgSlangBv32;   // P9.17b: the BV32 reader AND its index rules, replacing a
                      // private copy of both that was verbatim identical to it.
                               // closestPtTetrahedron — extracted at P8.8 step 4, when the
                               // cloth-vertex contact generator needed the same four.

// PxgNpKernelIndices.h:32 — the engine's warps-per-block for every midphase launch.
static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;
// PxgSoftBodyCoreKernelIndices.h:55 — SB_SBMIDPHASE, the fixed x-grid of this launch.
static const uint SB_SBMIDPHASE = 1024u;
// blockDim.y * gridDim.x, the CUDA grid stride over vertices.
static const uint SC_NB_WARPS = SB_SBMIDPHASE * MIDPHASE_WARPS_PER_BLOCK;

static const uint WARP = 32u;
static const float PX_MAX_F32 = 3.4028235e38f;

// PxEncodeSoftBodyIndex (PxgSimulationCoreDesc.h): (softBodyId << 20) | tetId.
static const uint PXG_BITSHIFT_TET_ID = 20u;

// @1024, mNbNodes @1152.

// femMidphaseScratch.sBv32Nodes[192] — a 6-deep BV32 tree's worst-case pending set.
static const uint STACK_SIZE = 192u;

// PxgNonRigidFilterPair (PxgSimulationCoreDesc.h:235), 16 B: index0 PxU64 @0, index1 @8,
// index2 @12.
static const uint64_t NRPAIR_STRIDE = 16UL;

// ---------------------------------------------------------------------------
// Push-constant blob — BYTE-IDENTICAL to the CUDA scMidphaseKernelParams pack
// (PxgSoftBodyCore.cpp:400): { const PxU32* activeSoftbodies, const PxReal* contactDistance,
// const PxgSoftBody* softbodies, PxgSoftBodyContactWriter writer } -> 80 bytes.
// sizeof(PxgSoftBodyContactWriter) = 56 (six pointers + PxU32 maxContacts + 4 tail pad), so
// paramAlign gives it 8-byte alignment and it lands at @24.  ABI row {8,8,8,56}.
// ---------------------------------------------------------------------------
struct SelfCollisionParams
{
    uint64_t activeSoftbodies;    // @0
    uint64_t contactDistance;     // @8
    uint64_t softbodies;          // @16
    // --- PxgSoftBodyContactWriter, inlined ---
    uint64_t outPoint;            // @24
    uint64_t outNormalPen;        // @32
    uint64_t outBarycentric0;     // @40
    uint64_t outBarycentric1;     // @48
    uint64_t outContactInfo;      // @56
    uint64_t totalContactCount;   // @64
    uint maxContacts;             // @72
    uint _pad76;                  // @76 -> 80
}

[[vk::push_constant]] ConstantBuffer<SelfCollisionParams> gSc;

// A PxU8 array read through the containing 32-bit word.  See the header on `& ~uint64_t(3)`.
uint byteAt(uint64_t base, uint idx)
{
    const uint64_t addr = base + uint64_t(idx);
    const uint word = Ptr<uint>(addr & ~uint64_t(3))[0];
    const uint shift = uint(addr & 3UL) * 8u;
    return (word >> shift) & 0xffu;
}

uint pxEncodeSoftBodyIndex(uint softBodyId, uint tetId)
{
    return (softBodyId << PXG_BITSHIFT_TET_ID) | tetId;
}

// ---------------------------------------------------------------------------
// PxgNonRigidFilterPair binary search — find(pairs, nbPairs, id0, id1) with
// ignoreIndex2 = true (deformableElementFilter.cuh:130).  Both query keys are PxU32
// encodings, so index0's high word is always 0.
// ---------------------------------------------------------------------------
bool u64Lt(uint2 a, uint2 b) { return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x); }
bool u64Eq(uint2 a, uint2 b) { return a.x == b.x && a.y == b.y; }

bool findFilter(uint64_t pairsAddr, uint nbPairs, uint id0, uint id1)
{
    // *** mFilteringPairs IS NULL WHEN THE BODY HAS NO FILTER PAIRS. ***
    // PxgSimulationCore only assigns it inside the dirty-filter branch
    // (PxgSimulationCore.cpp:1446), so a body that never registered a filter reaches this
    // kernel with a null pointer and nbPairs == 0 -- which is the common case, and the case
    // the first live self-collision scene hit immediately.
    //
    // A zero-trip `while (l < r)` is NOT enough on llvmpipe.  It speculates the first
    // iteration's load, faults at address 0, and the garbage it substitutes DIFFERS BETWEEN
    // LANES -- so `findFilter` stops being warp-uniform, the caller's `continue` stops being
    // group-uniform, and the workgroup barrier after it is reached by only part of the
    // group.  That is why the symptom was a SIGSEGV whose location moved with the
    // groupshared layout (STACK_SIZE 192 vs 193 changed it) and vanished under
    // PXG_SYNC_EACH: the fault is a barrier divergence, not a bad store.
    //
    // The fix is the one softBodyContactRemapBda spells out for prevContactCount: SELECT
    // THE POINTER, NOT THE LOADED VALUE.  Guarding with `if` cannot work -- the branch is
    // already there and llvmpipe if-converts it -- so the faulting ADDRESS has to go.
    // gSc.softbodies is dereferenced unconditionally by every invocation of this kernel, so
    // it is never null and a speculated 16-byte read at its offset 0 is harmless; forcing
    // `r` to 0 keeps the RESULT exactly what CUDA computes for an empty list.
    const bool havePairs = (pairsAddr != 0UL) && (nbPairs != 0u);
    const uint64_t base = havePairs ? pairsAddr : gSc.softbodies;
    const uint2 q0 = uint2(id0, 0u);
    uint l = 0u, r = havePairs ? nbPairs : 0u;
    while (l < r)
    {
        const uint mid = (l + r) / 2u;
        const uint64_t pAddr = base + uint64_t(mid) * NRPAIR_STRIDE;
        const uint2 pIndex0 = uint2(Ptr<uint>(pAddr)[0], Ptr<uint>(pAddr + 4UL)[0]);
        const uint pIndex1 = Ptr<uint>(pAddr + 8UL)[0];
        const uint pIndex2 = Ptr<uint>(pAddr + 12UL)[0];

        // PxgNonRigidFilterPair::compare, with the query's index2 fixed at 0.
        int cmp = 0;
        if (u64Lt(q0, pIndex0)) cmp = -1;
        else if (u64Lt(pIndex0, q0)) cmp = 1;
        else if (id1 < pIndex1) cmp = -1;
        else if (id1 > pIndex1) cmp = 1;
        else if (0u < pIndex2) cmp = -1;

        if (cmp == 0 || (u64Eq(q0, pIndex0) && id1 == pIndex1))
            return true;
        if (cmp < 0) r = mid; else l = mid + 1u;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Geometry helpers — verbatim from npSbSelf.slang / npSbTet.slang.
// ---------------------------------------------------------------------------
// PxgSoftBodyContactWriter::writeContact (PxgSoftBodyCore.h:276).  auxInd is NOT touched by
// the CUDA writer, so it is not written here either.
void writeSelfContact(uint index, float4 contact, float4 normalPen,
                      float4 barycentric0, float4 barycentric1, uint pairId0, uint pairId1)
{
    if (index >= gSc.maxContacts)
        return;

    Ptr<float4>(gSc.outPoint)[index] = contact;
    Ptr<float4>(gSc.outNormalPen)[index] = normalPen;
    Ptr<float4>(gSc.outBarycentric0)[index] = barycentric0;
    Ptr<float4>(gSc.outBarycentric1)[index] = barycentric1;

    // PxgFemFemContactInfo (PxgFEMCore.h:51), 16 B: pairInd0 PxU64 @0, pairInd1 @8,
    // auxInd @12.  pairId0 is a PxU32 assigned into a PxU64 field -> zero-extended.
    const uint64_t infoAddr = gSc.outContactInfo + uint64_t(index) * 16UL;
    Ptr<uint>(infoAddr)[0] = pairId0;
    Ptr<uint>(infoAddr + 4UL)[0] = 0u;
    Ptr<uint>(infoAddr + 8UL)[0] = pairId1;
}

// ---------------------------------------------------------------------------
// Warp emulation scratch: one group == one warp, so a single slot each.
// ---------------------------------------------------------------------------
groupshared int sScStack[STACK_SIZE];   // bv32 to-be-expanded node stack
groupshared uint sScBallot;             // __ballot_sync mask
groupshared float sScBest[WARP];        // minIndex reduction over the lane bests

uint scBallot(bool pred, uint lane)
{
    if (lane == 0u) sScBallot = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sScBallot, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sScBallot;
}

// ===========================================================================
// sb_selfCollisionMidphaseGeneratePairsLaunch
// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void sb_selfCollisionMidphaseGeneratePairsLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const uint cmIdx = gid.y;                                   // blockIdx.y
    const uint softbodyId = Ptr<uint>(gSc.activeSoftbodies)[cmIdx];

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gSc.softbodies);

    // mBodyFlags is byte 1 of mFlagsWord.  Self-collision disabled or kinematic -> nothing.
    const uint bodyFlags = (softbodies[softbodyId].mFlagsWord >> 8u) & 0xFFu;
    if ((bodyFlags & PXG_SOFTBODY_BODYFLAG_DISABLE_SELF_COLLISION) != 0u ||
        (bodyFlags & PXG_SOFTBODY_BODYFLAG_KINEMATIC) != 0u)
        return;

    const uint numVerts = softbodies[softbodyId].mNumVerts;
    const float selfCollisionDistance = softbodies[softbodyId].mSelfCollisionFilterDistance;
    const float filterDistanceSq = selfCollisionDistance * selfCollisionDistance;
    const float selfCollisionStressTolerance = softbodies[softbodyId].mSelfCollisionStressTolerance;

    const uint elementInd = softbodies[softbodyId].mElementIndex;
    const float distance = Ptr<float>(gSc.contactDistance)[elementInd] * 2.0f;

    const uint64_t treeAddr = softbodies[softbodyId].mTetMeshData + 16UL;   // + sizeof(uint4)
    Ptr<float4> tetVerts = Ptr<float4>(softbodies[softbodyId].mPosition_InvMass);
    Ptr<uint4> tetIndices = Ptr<uint4>(softbodies[softbodyId].mTetIndices);
    Ptr<float4> restTetVerts = Ptr<float4>(softbodies[softbodyId].mRestPosition);
    Ptr<float> tetraStresses = Ptr<float>(softbodies[softbodyId].mTetraStressCoefficient);
    Ptr<uint> vertToSurfaceTetRemap = Ptr<uint>(softbodies[softbodyId].mSurfaceVertToTetRemap);
    const uint64_t surfaceVertsHint = softbodies[softbodyId].mSurfaceVertsHint;
    const uint64_t surfaceHintBase = softbodies[softbodyId].mTetMeshSurfaceHint;
    const uint64_t pairsAddr = softbodies[softbodyId].mFilteringPairs;
    const uint nbFilterPairs = softbodies[softbodyId].mNumFilterPairs;

    // See the header on the grid: the block's two warps are walked serially, and the stride
    // stays the CUDA blockDim.y * gridDim.x.
    for (uint sub = 0u; sub < MIDPHASE_WARPS_PER_BLOCK; ++sub)
    {
    const uint globalWarpIdx = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub;

    for (uint vertIdx = globalWarpIdx; vertIdx < numVerts; vertIdx += SC_NB_WARPS)
    {
        GroupMemoryBarrierWithGroupSync();   // groupshared scratch is reused per query

        // Only SURFACE vertices generate self contacts.
        if (byteAt(surfaceVertsHint, vertIdx) == 0u)
            continue;

        // The vertex-level filter: (softbodyId, vertIdx) against the whole-body wildcard.
        if (findFilter(pairsAddr, nbFilterPairs,
                       pxEncodeSoftBodyIndex(softbodyId, vertIdx),
                       pxEncodeSoftBodyIndex(softbodyId, 0xfffffu)))
            continue;

        const float3 point = tetVerts[vertIdx].xyz;
        const float3 restP = restTetVerts[vertIdx].xyz;

        // --- SbMeshTreeTraverser per-lane state ---
        float bestDist = PX_MAX_F32;
        float3 bestClosestP = float3(0, 0, 0);
        float3 bestNormal = float3(0, 0, 0);
        uint bestPrimIndex = 0u;

        uint currentNodeIndex = 0u;   // root
        bool hasNode = true;
        uint nbStack = 0u;

        while (hasNode)
        {
            const uint nodeIdx = bvNodeIdxOf(currentNodeIndex);

            if (bvIsLeaf(currentNodeIndex))
            {
                // Leaf: the primitive range lives in the PARENT node's mData slot.
                const uint leafOffset = bvLeafSlotOf(currentNodeIndex);
                const uint data = bvData(treeAddr, nodeIdx, leafOffset);
                const uint nbPrimitives = bvDataNbPrimitives(data);
                const uint primStartIndex = bvDataPrimStart(data);
                uint primitiveIndex = 0xffffffffu;
                if (lane < nbPrimitives)
                    primitiveIndex = lane + primStartIndex;
                // P9.28: the `!= 0xffffffff` guard below is PER-LANE, so this platform turns it
                // into a mask and issues the reads under it for the EXCLUDED lanes too -- at
                // primStartIndex + 0xffffffff, sixty-four gigabytes out for a uint4 array.
                // `safePrim` is the ADDRESS those reads use; the sentinel still decides the
                // VALUE, so nothing observable changes.  Traced from a reproducible sporadic
                // SIGSEGV, not from reading.
                const uint safePrim = min(primitiveIndex,
                                          primStartIndex + max(nbPrimitives, 1u) - 1u);

                // SbMeshTreeTraverser::intersectPrimitiveFullWarp
                // (softbodySoftbodyMidPhase.cu:359).
                if (primitiveIndex != 0xffffffffu)
                {
                    bool bHasCollision = false;
                    bool inside = false;
                    float dist = PX_MAX_F32;
                    float3 closestP = float3(0, 0, 0);
                    float3 normal = float3(0, 0, 0);

                    const uint4 tetIdx1 = tetIndices[safePrim];
                    const uint hint = byteAt(surfaceHintBase, primitiveIndex);

                    // Don't generate contacts with a tet the query vertex belongs to.
                    if (hint != 0u && !(tetIdx1.x == vertIdx || tetIdx1.y == vertIdx ||
                                        tetIdx1.z == vertIdx || tetIdx1.w == vertIdx))
                    {
                        const uint softBodyMask0 = pxEncodeSoftBodyIndex(softbodyId, vertIdx);
                        const uint softBodyMask1 = pxEncodeSoftBodyIndex(softbodyId, primitiveIndex);
                        if (!findFilter(pairsAddr, nbFilterPairs, softBodyMask0, softBodyMask1))
                        {
                            float3 a = tetVerts[tetIdx1.x].xyz;
                            float3 b = tetVerts[tetIdx1.y].xyz;
                            float3 c = tetVerts[tetIdx1.z].xyz;
                            float3 d = tetVerts[tetIdx1.w].xyz;

                            if (isValidTet(a, b, c, d))
                            {
                                inside = isPointOutsideOfPlane4(point, a, b, c, d);

                                float sqDist2 = PX_MAX_F32;
                                closestPtTetrahedron(point, hint, a, b, c, d, normal, closestP, sqDist2);

                                float3 v = point - closestP;
                                normal = -normal;

                                if (inside)
                                {
                                    bHasCollision = true;
                                    dist = dot(v, normal);
                                }
                                else if (dot(v, normal) > 0.0f)
                                {
                                    // PxVec3::normalize returns the LENGTH and leaves v unit.
                                    // P9.36: through operator/= -- ONE reciprocal, three multiplies
                                    // (PxVec3.h:260) -- and with the m > 0 guard, which is reachable
                                    // for denormal-scale v (dot(v, normal) > 0 while dot(v, v)
                                    // underflows to 0).
                                    dist = length(v);
                                    if (dist > 0.0f)
                                        v = v * (1.0f / dist);
                                    if (dot(normal, v) > 0.707f)
                                        normal = v;
                                    bHasCollision = dist < distance;
                                }

                                if (bHasCollision)
                                {
                                    // Re-check in the REST pose: a contact that already
                                    // existed at rest is not a self collision.
                                    a = restTetVerts[tetIdx1.x].xyz;
                                    b = restTetVerts[tetIdx1.y].xyz;
                                    c = restTetVerts[tetIdx1.z].xyz;
                                    d = restTetVerts[tetIdx1.w].xyz;

                                    float3 newNorm = float3(0, 0, 0);
                                    float3 newClosestP = float3(0, 0, 0);
                                    float newSqDist = PX_MAX_F32;

                                    const bool inside2 = isPointOutsideOfPlane4(restP, a, b, c, d);
                                    closestPtTetrahedron(restP, 0xfu, a, b, c, d, newNorm, newClosestP, newSqDist);

                                    bHasCollision = !inside2 && newSqDist > filterDistanceSq;

                                    if (bHasCollision)
                                    {
                                        const float stress = tetraStresses[safePrim];
                                        if (stress > selfCollisionStressTolerance)
                                            bHasCollision = false;
                                    }
                                }
                            }
                        }
                    }

                    if (bHasCollision && dist < bestDist)
                    {
                        bestDist = dist;
                        bestNormal = normal;
                        bestClosestP = closestP;
                        bestPrimIndex = primitiveIndex;
                    }
                }
            }
            else
            {
                // Internal node: SbMeshTreeTraverser::intersectBoxFullWarp — the query
                // point's clamped distance to the child box against the contact distance.
                const uint nbChildren = bvNbNodes(treeAddr, nodeIdx);
                float3 mn = float3(0, 0, 0), mx = float3(0, 0, 0);
                const bool hasBox = lane < nbChildren;
                if (hasBox)
                {
                    mn = bvMin(treeAddr, nodeIdx, lane);
                    mx = bvMax(treeAddr, nodeIdx, lane);
                }
                bool intersect = false;
                if (hasBox)
                {
                    float3 closest;
                    closest.x = max(mn.x, min(mx.x, point.x));
                    closest.y = max(mn.y, min(mx.y, point.y));
                    closest.z = max(mn.z, min(mx.z, point.z));
                    const float3 dv = closest - point;
                    intersect = dot(dv, dv) < distance * distance;
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

                const uint mask = scBallot(isValidNode, lane);
                const uint offset = countbits(mask & ((1u << lane) - 1u));
                const uint validCount = countbits(mask);
                if (isValidNode)
                    sScStack[nbStack + offset] = int(currentIndex);
                nbStack += validCount;
            }

            GroupMemoryBarrierWithGroupSync();
            if (nbStack > 0u)
            {
                nbStack -= 1u;
                currentNodeIndex = uint(sScStack[nbStack]);
            }
            else
                hasNode = false;
        }

        // finalizeFullWarp: minIndex over the lane bests — the LOWEST lane holding the
        // minimum wins, which a strict-`<` scan from lane 0 reproduces exactly.
        sScBest[lane] = bestDist;
        GroupMemoryBarrierWithGroupSync();
        uint winnerLane = 0u;
        {
            float best = PX_MAX_F32;
            for (uint j = 0u; j < WARP; ++j)
                if (sScBest[j] < best) { best = sScBest[j]; winnerLane = j; }
        }

        if (winnerLane == lane && bestDist < distance)
        {
            uint index;
            InterlockedAdd(Ptr<uint>(gSc.totalContactCount)[0], 1u, index);

            const float pen = bestDist;
            const uint tetrahedronIdx0 = vertToSurfaceTetRemap[vertIdx];
            // NOTE the two "barycentric" slots carry the query point and the closest point,
            // not barycentrics — that is what the CUDA writes (softbodySoftbodyMidPhase.cu
            // :503) and sb_fem_contact_remap_to_sim turns them into real barycentrics.
            writeSelfContact(index,
                             float4(bestClosestP, 0.0f),
                             float4(bestNormal, pen),
                             float4(point, 0.0f),
                             float4(bestClosestP, 0.0f),
                             pxEncodeSoftBodyIndex(softbodyId, tetrahedronIdx0),
                             pxEncodeSoftBodyIndex(softbodyId, bestPrimIndex));
        }
    }
    }
}
