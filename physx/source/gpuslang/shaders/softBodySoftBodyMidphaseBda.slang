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
// softBodySoftBodyMidphaseBda — sb_sbMidphaseGeneratePairsLaunch
// (gpunarrowphase/src/CUDA/softbodySoftbodyMidPhase.cu:811, launch
// PxgNarrowphaseCore.cpp:4441, SB_SB_MIDPHASE), on the engine BDA ABI.
//
// *** THIS IS THE WHOLE SOFTBODY-VS-SOFTBODY NARROWPHASE. ***  Unlike every other collision
// pair in the port, it is ONE kernel, not a chain.  `testSDKSoftbodies`
// (PxgNarrowphaseCore.cpp:4399) dispatches this and nothing else, then hands the result to
// `softbodyFemContactApplyCollisionToSimMeshMapping` -- the already-mapped
// `sb_fem_contact_remap_to_simLaunch`.  Constraint prepare and solve are
// `sb_softbodyContactPrepareLaunch` and `sb_solveOutputSSDeltaVLaunch`, both already mapped by
// the SELF-collision tranche (P8.4), which shares the same buffers and the same two kernels.
// So this one file closes softbody-vs-softbody.
//
// WHAT IT DOES: one warp per SURFACE vertex of softbody0 walks softbody1's BV32 tetrahedron
// tree, keeps the single deepest surviving vertex-vs-tetrahedron contact, and appends it
// through PxgSoftBodyContactWriter.
//
// RELATIONSHIP TO softBodySelfCollisionBda.  The traverser body is the SAME traverser, and the
// two were transcribed from adjacent structs in the same .cu (SbMeshTreeTraverser at :359 and
// SbSbMeshTreeTraverser at :504).  Four things differ, and all four are consequences of the
// query vertex and the tree belonging to DIFFERENT bodies:
//
//   1. no shared-vertex skip.  Self-collision must not test a vertex against a tet it is a
//      corner of; here the two bodies share no vertices, so the test is absent upstream;
//   2. no rest-pose re-check and no stress gate.  Both exist to tell "the mesh is folded onto
//      itself" from "the mesh was always shaped like that", a question that only arises within
//      one body;
//   3. the contact distance is the SUM of the two shapes' contact distances taken from the
//      contact-manager input, not one body's distance doubled;
//   4. the emitted pair ids name two different softbodies.
//
// The inside/outside decision is byte-for-byte the same arithmetic despite looking different in
// the two .cu structs.  Upstream's self-collision negates the normal BEFORE the branch and
// tests `v.dot(normal) > 0`; upstream's SS leaves it un-negated and tests `v.dot(normal) < 0`,
// then picks `v.dot(normal) < -0.707f ? v : -normal`.  Substituting normal -> -normal turns
// each into the other, including the 0.707 tie-break.  This port keeps the self-collision
// FORM, because that form is the one already covered by fixtures.
//
// *** THE GRID IS THREE-DIMENSIONAL, AND blockIdx.z IS NOT A SPARE. ***
// The engine launches grid (SB_SBMIDPHASE = 1024, numTests, 2) with block
// (WARP_SIZE, MIDPHASE_WARPS_PER_BLOCK = 2, 1).  `blockIdx.z` is the `flip` flag: z = 0 tests
// shape0's vertices against shape1's tree, z = 1 tests shape1's vertices against shape0's.
// Upstream calls this out in a comment -- "we do bidirectional collision (A to B and B to A).
// This is to ensure we don't miss contacts when we have a soft body with an insufficiently
// tessellated tetrahedron mesh."  DROPPING THE Z DIMENSION HALVES THE CONTACT SET and the
// coarser of two meshes silently sinks into the finer one.  The bridge forwards gridDimX/Y/Z
// verbatim, so `flip = gid.z` needs nothing special -- but it does need to be there.
//
// The bridge DROPS blockDim (PxgVulkanCudaContext.cpp:292-331), so a [numthreads(32,1,1)] group
// is ONE warp and the block's two warps are walked serially in a group-uniform `sub` loop,
// with the grid stride kept at the CUDA `blockDim.y * gridDim.x` = 2048.  Same treatment, same
// reasoning as softBodySelfCollisionBda -- see its header for why deriving the stride from the
// group count instead is wrong.
//
// DEVIATIONS from CUDA (recipe policy, each preserving the emitted contact SET and the atomic
// total):
//   - NO 32-lane warp intrinsics.  __ballot_sync -> groupshared InterlockedOr;
//     warpScanExclusive -> countbits(mask & lanemask_lt); finalizeFullWarp's
//     `minIndex(bestDist, FULL_MASK, sep)` -> a groupshared float[32] plus a serial strict-`<`
//     scan from lane 0, which picks the same LOWEST-lane minimum minIndex does.  Ties matter:
//     two tetrahedra at the same distance must not both write;
//   - the per-warp femMidphaseScratch (meshVerts / meshVertsIndices / bv32PackedNodes cached by
//     lane 0 behind a __syncthreads) is dropped; every lane reads those addresses directly
//     through Ptr<>.  Same values, one barrier fewer;
//   - PxgShape_ReadWarp / PxgContactManagerInput_ReadWarp are cooperative 32-lane struct loads;
//     here every lane simply reads the struct.  Same bytes;
//   - mSurfaceVertsHint and mTetMeshSurfaceHint are PxU8 arrays.  llvmpipe has no 8-bit storage
//     in these shaders, so the containing 32-bit word is read and the byte extracted.  NOTE the
//     word address is formed with `& ~uint64_t(3)`: writing `& ~3UL` MISCOMPILES -- Slang
//     evaluates the complement in 32 bits and clears the top half of the device address
//     (documented at length in softBodySolveBda.slang);
//   - PxgSoftBodyContactWriter is passed BY VALUE as the last CUDA argument, so it is INLINED
//     into the push blob (@56..@111), giving a 112-byte blob.
//
// The tetmesh blob hanging off PxgSoftBody::mTetMeshData opens with a uint4 header
// (nbVerts, nbTets, maxDepth, nbBv32TreeNodes), so the BV32 node array starts at
// mTetMeshData + 16.
//
// UNUSED PARAMETERS ARE KEPT IN THE BLOB.  `toleranceLength`, `transformCache` and `bounds` are
// in the CUDA signature and never read by the core (upstream passes them for symmetry with the
// softbody-vs-rigid midphase).  They stay in the struct because the blob is a byte image of the
// CUDA arg pack: dropping one shifts every field after it.
//
// NoContraction-stamped (PXG_NOCONTRACT_SHADERS): isValidTet's signed volume,
// isPointOutsideOfPlane4's four plane products and closestPtPointTriangle2's barycentric
// determinants are dot-of-cross / mul-feeding-add chains that would otherwise fuse to FMAs and
// break parity with the -ffp-contract=off CPU reference.
// ===========================================================================

import pxgSlangSoftBodyDesc;   // PxgSoftBodyGpu
import pxgSlangTetClosest;     // getNormalized / isValidTet / isPointOutsideOfPlane4 /
                               // closestPtTetrahedron

// PxgNpKernelIndices.h:32 — the engine's warps-per-block for every midphase launch.
static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;
// PxgSoftBodyCoreKernelIndices.h:55 — SB_SBMIDPHASE, the fixed x-grid of this launch.
static const uint SB_SBMIDPHASE = 1024u;
// blockDim.y * gridDim.x, the CUDA grid stride over vertices.
static const uint SS_NB_WARPS = SB_SBMIDPHASE * MIDPHASE_WARPS_PER_BLOCK;

static const uint WARP = 32u;
static const float PX_MAX_F32 = 3.4028235e38f;

// PxEncodeSoftBodyIndex (PxgSimulationCoreDesc.h): (softBodyId << 20) | tetId.
static const uint PXG_BITSHIFT_TET_ID = 20u;
// PxDeformableVolume.h:57 — the "any tet" wildcard the vertex-level filter query uses.
static const uint PX_MAX_NB_DEFORMABLE_VOLUME_TET = 0x000fffffu;

// PxGeometryType::eTETRAHEDRONMESH.
static const uint GEOM_TETRAHEDRONMESH = 7u;

// Gu::BV32DataPacked: 1168 B — mMin[32] float4 @0, mMax[32] float4 @512, mData[32] uint
// @1024, mNbNodes @1152.
import pxgSlangBv32;   // P8.25: the BV32 reader, shared

// femMidphaseScratch.sBv32Nodes[192] — a 6-deep BV32 tree's worst-case pending set.
static const uint STACK_SIZE = 192u;

// PxgNonRigidFilterPair (PxgSimulationCoreDesc.h:235), 16 B: index0 PxU64 @0, index1 @8,
// index2 @12.
static const uint64_t NRPAIR_STRIDE = 16UL;

// ---------------------------------------------------------------------------
// Byte-exact narrowphase ABI mirrors (identical to softBodyContactGenBda's).
// ---------------------------------------------------------------------------
struct PxgContactManagerInput   // PxgContactManager.h, 16 B
{
    uint shapeRef0;          // @0
    uint shapeRef1;          // @4
    uint transformCacheRef0; // @8
    uint transformCacheRef1; // @12
}

struct PxgShape                 // PxgConvexConvexShape.h, 48 B
{
    float scaleX, scaleY, scaleZ;                     // @0
    float scaleRotX, scaleRotY, scaleRotZ, scaleRotW; // @12
    uint materialIndex;                               // @28
    uint2 hullOrMeshOffset;                           // @32
    uint type;                                        // @40
    uint particleOrSoftbodyId;                        // @44
}

// ---------------------------------------------------------------------------
// Push-constant blob — BYTE-IDENTICAL to the CUDA arg pack (PxgNarrowphaseCore.cpp:4443):
// { PxReal toleranceLength, const PxgContactManagerInput* cmInputs,
//   const PxsCachedTransform* transformCache, const PxBounds3* bounds,
//   const PxReal* contactDistance, const PxgShape* gpuShapes, const PxgSoftBody* softbodies,
//   PxgSoftBodyContactWriter writer }
// -> PxReal @0, then six pointers realigned to @8..@48, then the 56-byte writer (paramAlign 8)
// at @56 -> a 112-byte blob.  ABI row {4,8,8,8,8,8,8,56}.
// ---------------------------------------------------------------------------
struct SbSbMidphaseParams
{
    float toleranceLength;        // @0   (unused by the core — see the header)
    uint _pad4;                   // @4
    uint64_t cmInputs;            // @8
    uint64_t transformCache;      // @16  (unused by the core)
    uint64_t bounds;              // @24  (unused by the core)
    uint64_t contactDistance;     // @32
    uint64_t gpuShapes;           // @40
    uint64_t softbodies;          // @48
    // --- PxgSoftBodyContactWriter, inlined ---
    uint64_t outPoint;            // @56
    uint64_t outNormalPen;        // @64
    uint64_t outBarycentric0;     // @72
    uint64_t outBarycentric1;     // @80
    uint64_t outContactInfo;      // @88
    uint64_t totalContactCount;   // @96
    uint maxContacts;             // @104
    uint _pad108;                 // @108 -> 112
}

[[vk::push_constant]] ConstantBuffer<SbSbMidphaseParams> gSs;

// ---------------------------------------------------------------------------
// BV32DataPacked field readers (explicit byte offset through Ptr<> — no whole-node copy).
// ---------------------------------------------------------------------------

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
// ignoreIndex2 = true (deformableElementFilter.cuh:130).
// ---------------------------------------------------------------------------
bool u64Lt(uint2 a, uint2 b) { return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x); }
bool u64Eq(uint2 a, uint2 b) { return a.x == b.x && a.y == b.y; }

bool findFilter(uint64_t pairsAddr, uint nbPairs, uint id0, uint id1)
{
    // *** mFilteringPairs IS NULL WHEN THE BODY HAS NO FILTER PAIRS, WHICH IS THE COMMON CASE. ***
    // A zero-trip `while (l < r)` is NOT enough on llvmpipe: it speculates the first
    // iteration's load, faults at address 0, and the garbage it substitutes DIFFERS BETWEEN
    // LANES -- so findFilter stops being warp-uniform, the caller's `continue` stops being
    // group-uniform, and the workgroup barrier after it is reached by only part of the group.
    // The fix is SELECT THE POINTER, NOT THE LOADED VALUE: gSs.softbodies is dereferenced
    // unconditionally by every invocation, so it is never null and a speculated read at its
    // offset 0 is harmless, while `r = 0` keeps the RESULT exactly CUDA's for an empty list.
    // Diagnosed in full in softBodySelfCollisionBda.slang.
    const bool havePairs = (pairsAddr != 0UL) && (nbPairs != 0u);
    const uint64_t base = havePairs ? pairsAddr : gSs.softbodies;
    const uint2 q0 = uint2(id0, 0u);
    uint l = 0u, r = havePairs ? nbPairs : 0u;
    while (l < r)
    {
        const uint mid = (l + r) / 2u;
        const uint64_t pAddr = base + uint64_t(mid) * NRPAIR_STRIDE;
        const uint2 pIndex0 = uint2(Ptr<uint>(pAddr)[0], Ptr<uint>(pAddr + 4UL)[0]);
        const uint pIndex1 = Ptr<uint>(pAddr + 8UL)[0];
        const uint pIndex2 = Ptr<uint>(pAddr + 12UL)[0];

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

// PxgSoftBodyContactWriter::writeContact (PxgSoftBodyCore.h:276).  auxInd is NOT touched by the
// CUDA writer, so it is not written here either.
void writeSsContact(uint index, float4 contact, float4 normalPen,
                    float4 barycentric0, float4 barycentric1, uint pairId0, uint pairId1)
{
    if (index >= gSs.maxContacts)
        return;

    Ptr<float4>(gSs.outPoint)[index] = contact;
    Ptr<float4>(gSs.outNormalPen)[index] = normalPen;
    Ptr<float4>(gSs.outBarycentric0)[index] = barycentric0;
    Ptr<float4>(gSs.outBarycentric1)[index] = barycentric1;

    // PxgFemFemContactInfo (PxgFEMCore.h:51), 16 B: pairInd0 PxU64 @0, pairInd1 @8, auxInd @12.
    // pairId0 is a PxU32 assigned into a PxU64 field -> zero-extended.
    const uint64_t infoAddr = gSs.outContactInfo + uint64_t(index) * 16UL;
    Ptr<uint>(infoAddr)[0] = pairId0;
    Ptr<uint>(infoAddr + 4UL)[0] = 0u;
    Ptr<uint>(infoAddr + 8UL)[0] = pairId1;
}

// ---------------------------------------------------------------------------
// Warp emulation scratch: one group == one warp, so a single slot each.
// ---------------------------------------------------------------------------
groupshared int sSsStack[STACK_SIZE];   // bv32 to-be-expanded node stack
groupshared uint sSsBallot;             // __ballot_sync mask
groupshared float sSsBest[WARP];        // minIndex reduction over the lane bests

uint ssBallot(bool pred, uint lane)
{
    if (lane == 0u) sSsBallot = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sSsBallot, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sSsBallot;
}

// ===========================================================================
// sb_sbMidphaseGeneratePairsLaunch
// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void sb_sbMidphaseGeneratePairsLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const uint cmIdx = gid.y;              // blockIdx.y — one contact manager per y
    const bool flip = (gid.z != 0u);       // blockIdx.z — the bidirectional pass, see the header

    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(gSs.cmInputs)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gSs.gpuShapes);

    const PxgShape shape0 = shapes[flip ? cin.shapeRef1 : cin.shapeRef0];
    const PxgShape shape1 = shapes[flip ? cin.shapeRef0 : cin.shapeRef1];

    // Upstream asserts both are tetmeshes; a non-tetmesh pair here is a bucket-routing bug and
    // walking a null mTetMeshData would fault rather than report it.
    if (shape0.type != GEOM_TETRAHEDRONMESH || shape1.type != GEOM_TETRAHEDRONMESH)
        return;

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gSs.softbodies);

    const uint softbodyId0 = shape0.particleOrSoftbodyId;   // the QUERY body
    const uint softbodyId1 = shape1.particleOrSoftbodyId;   // the body whose TREE is walked

    // NOTE: the two contact distances are indexed by the UNFLIPPED transform cache refs, i.e.
    // their sum does not depend on `flip` -- which is what upstream does, and is right, since a
    // sum is symmetric.
    const float contactDist = Ptr<float>(gSs.contactDistance)[cin.transformCacheRef0] +
                              Ptr<float>(gSs.contactDistance)[cin.transformCacheRef1];

    const uint numVerts = softbodies[softbodyId0].mNumVerts;

    Ptr<float4> queryVerts = Ptr<float4>(softbodies[softbodyId0].mPosition_InvMass);
    Ptr<uint> vertToSurfaceTetRemap0 = Ptr<uint>(softbodies[softbodyId0].mSurfaceVertToTetRemap);
    const uint64_t surfaceVertsHint0 = softbodies[softbodyId0].mSurfaceVertsHint;
    const uint64_t pairsAddr = softbodies[softbodyId0].mFilteringPairs;
    const uint nbFilterPairs = softbodies[softbodyId0].mNumFilterPairs;

    const uint64_t treeAddr = softbodies[softbodyId1].mTetMeshData + 16UL;   // + sizeof(uint4)
    Ptr<float4> treeVerts = Ptr<float4>(softbodies[softbodyId1].mPosition_InvMass);
    Ptr<uint4> treeTetIndices = Ptr<uint4>(softbodies[softbodyId1].mTetIndices);
    const uint64_t surfaceTetHint1 = softbodies[softbodyId1].mTetMeshSurfaceHint;

    // See the header on the grid: the block's two warps are walked serially, and the stride
    // stays the CUDA blockDim.y * gridDim.x.
    for (uint sub = 0u; sub < MIDPHASE_WARPS_PER_BLOCK; ++sub)
    {
    const uint globalWarpIdx = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub;

    for (uint vertIdx = globalWarpIdx; vertIdx < numVerts; vertIdx += SS_NB_WARPS)
    {
        GroupMemoryBarrierWithGroupSync();   // groupshared scratch is reused per query

        // Only SURFACE vertices of the query body generate contacts.
        if (byteAt(surfaceVertsHint0, vertIdx) == 0u)
            continue;

        // The vertex-level filter: this vertex against the OTHER body's "any tet" wildcard.
        if (findFilter(pairsAddr, nbFilterPairs,
                       pxEncodeSoftBodyIndex(softbodyId0, vertIdx),
                       pxEncodeSoftBodyIndex(softbodyId1, PX_MAX_NB_DEFORMABLE_VOLUME_TET)))
            continue;

        const float3 point = queryVerts[vertIdx].xyz;

        // --- SbSbMeshTreeTraverser per-lane state ---
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

                // SbSbMeshTreeTraverser::intersectPrimitiveFullWarp
                // (softbodySoftbodyMidPhase.cu:540).
                if (primitiveIndex != 0xffffffffu)
                {
                    bool bHasCollision = false;
                    bool inside = false;
                    float dist = PX_MAX_F32;
                    float3 closestP = float3(0, 0, 0);
                    float3 normal = float3(0, 0, 0);

                    const uint4 tetIdx1 = treeTetIndices[safePrim];
                    const uint hint = byteAt(surfaceTetHint1, primitiveIndex);

                    // Only SURFACE tetrahedra of the target body receive contacts.  There is no
                    // shared-vertex skip here: the two bodies share no vertices.
                    if (hint != 0u)
                    {
                        const uint softBodyMask0 = pxEncodeSoftBodyIndex(softbodyId0, vertIdx);
                        const uint softBodyMask1 = pxEncodeSoftBodyIndex(softbodyId1, primitiveIndex);
                        if (!findFilter(pairsAddr, nbFilterPairs, softBodyMask0, softBodyMask1))
                        {
                            const float3 a = treeVerts[tetIdx1.x].xyz;
                            const float3 b = treeVerts[tetIdx1.y].xyz;
                            const float3 c = treeVerts[tetIdx1.z].xyz;
                            const float3 d = treeVerts[tetIdx1.w].xyz;

                            if (isValidTet(a, b, c, d))
                            {
                                inside = isPointOutsideOfPlane4(point, a, b, c, d);

                                float sqDist2 = PX_MAX_F32;
                                closestPtTetrahedron(point, hint, a, b, c, d, normal, closestP, sqDist2);

                                float3 v = point - closestP;
                                // See the header: negating here and testing `> 0` is upstream's
                                // SS `< 0` on the un-negated normal, term for term.
                                normal = -normal;

                                if (inside)
                                {
                                    bHasCollision = true;
                                    dist = dot(v, normal);
                                }
                                else if (dot(v, normal) > 0.0f)
                                {
                                    // PxVec3::normalize returns the LENGTH and leaves v unit.
                                    dist = length(v);
                                    v = v / dist;
                                    if (dot(normal, v) > 0.707f)
                                        normal = v;
                                    bHasCollision = dist < contactDist;
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
                // Internal node: intersectBoxFullWarp — the query point's clamped distance to
                // the child box against the contact distance.
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
                    intersect = dot(dv, dv) < contactDist * contactDist;
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

                const uint mask = ssBallot(isValidNode, lane);
                const uint offset = countbits(mask & ((1u << lane) - 1u));
                const uint validCount = countbits(mask);
                if (isValidNode)
                    sSsStack[nbStack + offset] = int(currentIndex);
                nbStack += validCount;
            }

            GroupMemoryBarrierWithGroupSync();
            if (nbStack > 0u)
            {
                nbStack -= 1u;
                currentNodeIndex = uint(sSsStack[nbStack]);
            }
            else
                hasNode = false;
        }

        // finalizeFullWarp: minIndex over the lane bests — the LOWEST lane holding the minimum
        // wins, which a strict-`<` scan from lane 0 reproduces exactly.
        sSsBest[lane] = bestDist;
        GroupMemoryBarrierWithGroupSync();
        uint winnerLane = 0u;
        {
            float best = PX_MAX_F32;
            for (uint j = 0u; j < WARP; ++j)
                if (sSsBest[j] < best) { best = sSsBest[j]; winnerLane = j; }
        }

        if (winnerLane == lane && bestDist < contactDist)
        {
            uint index;
            InterlockedAdd(Ptr<uint>(gSs.totalContactCount)[0], 1u, index);

            const float pen = bestDist;
            const uint tetrahedronIdx0 = vertToSurfaceTetRemap0[vertIdx];
            // NOTE the two "barycentric" slots carry the query point and the closest point, not
            // barycentrics — that is what the CUDA writes (softbodySoftbodyMidPhase.cu:644) and
            // sb_fem_contact_remap_to_sim turns them into real barycentrics.
            writeSsContact(index,
                           float4(bestClosestP, 0.0f),
                           float4(bestNormal, pen),
                           float4(point, 0.0f),
                           float4(bestClosestP, 0.0f),
                           pxEncodeSoftBodyIndex(softbodyId0, tetrahedronIdx0),
                           pxEncodeSoftBodyIndex(softbodyId1, bestPrimIndex));
        }
    }
    }
}
