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
// softBodyPsMidphaseBda.slang
//
// P9.39a -- sb_psMidphaseGeneratePairsLaunch
// gpunarrowphase/src/CUDA/softbodyMidPhase.cu:306 (kernel, body :323-407),
// SoftbodyBoxTraverser :253-304, bv32Traversal.cuh (the walk and
// pushOntoStackFullWarp) -- all read in full.
//
// The first of the six kernels of the softbody-vs-PARTICLE chain, the last
// softbody chain in the engine with no port.  It is the TET twin of
// femClothPsMidphaseBda (P9.30): same query -- a particle position inflated into a
// cube -- same BV32 walk, same pair stack, and the SAME 88-byte push blob with the
// same ABI row {4,4,8,8,8,8,8,8,8,4,8,8}.  What differs is the tree it walks and
// the primitive it tests, and those differences are the traps.
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
// T1 -- THE PAIR IS (cmIdx, particleIndex, tetIndex, 0), the PARTICLE in .y and
//     the tetrahedron in .z, exactly as the cloth sibling puts the particle in .y
//     and the triangle in .z.  Every softbody-vs-softbody midphase in this port
//     writes a softbody element in .y instead; copying one of those swaps the two
//     and the contact gen then indexes tetrahedra by particle id.
//
// T2 -- THE `isValidTet` GATE HAS NO COUNTERPART IN THE CLOTH SIBLING, and this
//     kernel is otherwise close enough to it to invite a copy.  softbodyMidPhase.cu
//     :287 tests `isValidTet(v0,v1,v2,v3)` -- signed volume > 1e-9 -- BEFORE the
//     bound test, so a degenerate or inverted tetrahedron produces no pair at all.
//     A port that clones the cloth traverser drops the gate silently and emits
//     pairs for collapsed tets, which the contact gen then divides by.
//
// T3 -- THE COLLISION MESH, NOT THE SIMULATION MESH.  The CUDA reads
//     `softbody.mPosition_InvMass` (@64) and `softbody.mTetIndices` (@16).
//     PxgSoftBody also carries mSimPosition_InvMass and mSimTetIndices (@80), a
//     DIFFERENT and usually coarser tetrahedralisation, and the two are the same
//     type so nothing catches the confusion but the result.
//
// T4 -- eTETRAHEDRONMESH IS 7.  PxGeometryType (PxGeometry.h) counts eSPHERE 0,
//     ePLANE 1, eCAPSULE 2, eBOX 3, eCONVEXCORE 4, eCONVEXMESH 5,
//     ePARTICLESYSTEM 6, eTETRAHEDRONMESH 7, eTRIANGLEMESH 8, eHEIGHTFIELD 9.
//     P9.35c wrote 9 here in its sibling and the sort therefore always swapped,
//     handing the walk a null mesh pointer and faulting the device.  The routing
//     below tests for ePARTICLESYSTEM (6) rather than for the mesh type, which is
//     LoadShapePairWarp's own rule and is immune to that mistake.
//
// T5 -- THE QUERY BOX IS A CUBE of half-side cDistance, not a bound inflated by
//     it: `PxBounds3(p - PxVec3(r), p + PxVec3(r))` with
//     r = contactDistance[particleCacheRef] + contactDistance[softbodyCacheRef],
//     the SUM of both shapes' distances.
//
// T6 -- SENTINEL-GUARDED BDA READ, the P9.28 class.  `primitiveIndex` is
//     0xffffffff for lanes past the leaf's primitive count and the index read sits
//     under `!= 0xffffffff`.  On this platform that guard is a MASK, so the read
//     issues for the excluded lanes too, at base + 0xffffffff * 16.  The ADDRESS is
//     clamped; the sentinel still decides the VALUE, and it is the sentinel that
//     goes into the pair.
//
//     WRITING `safePrim` INTO THE PAIR INSTEAD IS AN EQUIVALENT MUTANT, and the
//     sweep left it standing.  It is a proof rather than a gap: a pair is pushed only
//     when `intersect` is true, `intersect` can only be true under
//     `primitiveIndex != 0xffffffff`, and there primitiveIndex == lane + primStartIndex
//     with lane < nbPrimitives, so primitiveIndex <= primStartIndex + nbPrimitives - 1
//     and the min that forms safePrim is a no-op.  The two are equal on every pair that
//     is ever written, so no fixture can separate them.  The sentinel form is kept
//     because it is what the CUDA says, and because the equality stops holding the
//     moment a lane past the count is allowed to push.
//
// T7 -- NO TRANSFORM AND NO SCALE.  Both the softbody's collision vertices and the
//     particles are already world-space; transformCache and bounds are in the blob
//     for ABI parity and are never read.  The softbody-vs-MESH midphase next door
//     does transform, and copying it here would move the body.
//
// T8 -- the CUDA warp-copies the WHOLE PxgParticleSystem (656 B) into shared memory
//     before reading two fields of it.  That is CUDA memory-traffic engineering,
//     not semantics: this port reads mSortedPositions_InvMass and
//     mCommonData_NumParticles through BDA where they are used.
//
// T9 -- the particle loop strides by blockDim.y * gridDim.x -- WARPS, not threads,
//     and NOT including blockDim.x.  The launch grid is a literal 1024 written
//     inline at PxgNarrowphaseCore.cpp:3388 (not the SB_SBMIDPHASE constant every
//     other softbody midphase in that file uses, though it has the same value), and
//     the comment directly above it -- "each warp deal with one test" -- is
//     copy-pasted from the softbody-vs-softbody launch and does not describe this
//     kernel, which gives each warp a STRIDE of particles.
// ===========================================================================

import pxgSlangSoftBodyDesc;   // PxgSoftBodyGpu (mTetMeshData@0, mTetIndices@16,
                               //                 mPosition_InvMass@64)
import pxgSlangParticleDesc;   // PxgParticleSystemGpu (mSortedPositions_InvMass)
import pxgSlangBv32;           // P8.25: the BV32 reader, shared
import pxgSlangTetClosest;     // isValidTet -- T2

static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;
static const uint SBPS_MIDPHASE_GRID = 1024u;                                   // T9
static const uint SBPS_NB_WARPS = SBPS_MIDPHASE_GRID * MIDPHASE_WARPS_PER_BLOCK;
static const uint STACK_SIZE = 192u;    // femMidphaseScratch.sBv32Nodes[192]

static const uint GEOM_PARTICLESYSTEM = 6u;   // T4

// ---------------------------------------------------------------------------
// Push-constant blob -- BYTE-IDENTICAL to the CUDA pack
// (PxgNarrowphaseCore.cpp params list :3369-3383): { PxU32 numWorkItems,
// PxReal toleranceLength, cmInputs*, gpuShapes*, transformCache*, bounds*,
// contactDistance*, softbodies*, particleSystems*, PxU32 stackSizeBytes,
// stackPtr*, midphasePairsNum* } -> 88 B, ABI row {4,4,8,8,8,8,8,8,8,4,8,8}.
//
// The cloth sibling's blob is the same twelve slots in the same order with
// femClothes where softbodies is; that is why the row is reused rather than
// re-derived, and why the two must not be allowed to drift apart silently.
// ---------------------------------------------------------------------------
struct SbPsMidphaseParams
{
    uint numWorkItems;        // @0
    float toleranceLength;    // @4   (ABI parity only -- T7)
    uint64_t cmInputs;        // @8
    uint64_t gpuShapes;       // @16
    uint64_t transformCache;  // @24  (ABI parity only -- T7)
    uint64_t bounds;          // @32  (ABI parity only -- T7)
    uint64_t contactDistance; // @40
    uint64_t softbodies;      // @48
    uint64_t particleSystems; // @56
    uint stackSizeBytes;      // @64
    uint64_t stackPtr;        // @72
    uint64_t midphasePairsNum;// @80  -> 88
}

[[vk::push_constant]]
ConstantBuffer<SbPsMidphaseParams> gSp;

struct PxgContactManagerInput
{
    uint shapeRef0;
    uint shapeRef1;
    uint transformCacheRef0;
    uint transformCacheRef1;
}

struct PxgShape
{
    float4 scale_rot;
    float3 scale_scale;
    uint materialIndex;
    uint2 hullOrMeshOffset;
    uint type;
    uint particleOrSoftbodyId;
}

// T4: LoadShapePairWarp<ePARTICLESYSTEM, eTETRAHEDRONMESH> -- shape0 is the one
// whose TYPE matches ePARTICLESYSTEM, and the pair is swapped if slot 0 is not it.
void loadParticleSoftbodyPair(uint64_t cmInputsAddr, uint64_t shapesAddr, uint cmIdx,
                              out PxgShape particleShape, out uint particleCacheRef,
                              out PxgShape softbodyShape, out uint softbodyCacheRef)
{
    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(cmInputsAddr)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(shapesAddr);
    uint sRefP = cin.shapeRef0, cRefP = cin.transformCacheRef0;
    uint sRefS = cin.shapeRef1, cRefS = cin.transformCacheRef1;
    if (shapes[cin.shapeRef0].type != GEOM_PARTICLESYSTEM)
    {
        sRefP = cin.shapeRef1; cRefP = cin.transformCacheRef1;
        sRefS = cin.shapeRef0; cRefS = cin.transformCacheRef0;
    }
    particleShape = shapes[sRefP]; particleCacheRef = cRefP;
    softbodyShape = shapes[sRefS]; softbodyCacheRef = cRefS;
}

// ---------------------------------------------------------------------------
// Warp emulation scratch: one group == one warp, so a single slot each.
// ---------------------------------------------------------------------------
groupshared int sSpStack[STACK_SIZE];
groupshared uint sSpBallot;
groupshared uint sSpBase;

uint spBallot(bool pred, uint lane)
{
    if (lane == 0u) sSpBallot = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sSpBallot, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sSpBallot;
}

// pushOntoStackFullWarp (bv32Traversal.cuh:197): ONE atomicAdd of the popcount by
// lane 0, broadcast, and each pushing lane writes at base + its exclusive prefix.
void spPushFullWarp(bool pushes, uint lane, uint stackCapacity, uint4 item)
{
    const uint mask = spBallot(pushes, lane);
    const uint total = countbits(mask);
    if (lane == 0u)
    {
        uint base = 0u;
        if (total != 0u)
            InterlockedAdd(Ptr<uint>(gSp.midphasePairsNum)[0], total, base);
        sSpBase = base;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint stackIndex = sSpBase + countbits(mask & ((1u << lane) - 1u));
    // "If we run out of stacksize, drop contacts" -- the counter still advanced,
    // which is what the contact gen's atomicMax high-water mark then reports.
    if (pushes && stackIndex < stackCapacity)
        Ptr<uint4>(gSp.stackPtr)[stackIndex] = item;
    GroupMemoryBarrierWithGroupSync();
}

// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void sb_psMidphaseGeneratePairsLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const uint cmIdx = gid.y;                      // one contact manager per y-block

    if (cmIdx >= gSp.numWorkItems)
        return;

    PxgShape particleShape, softbodyShape;
    uint particleCacheRef, softbodyCacheRef;
    loadParticleSoftbodyPair(gSp.cmInputs, gSp.gpuShapes, cmIdx,
                             particleShape, particleCacheRef, softbodyShape, softbodyCacheRef);

    // T8: the two fields are read where they are used, not staged into a 656-byte
    // groupshared copy of the whole PxgParticleSystem.
    Ptr<PxgParticleSystemGpu> particleSystems = Ptr<PxgParticleSystemGpu>(gSp.particleSystems);
    const uint particleSystemId = particleShape.particleOrSoftbodyId;
    Ptr<float4> sortedPose = Ptr<float4>(particleSystems[particleSystemId].mSortedPositions_InvMass);
    const uint nbParticles = particleSystems[particleSystemId].mCommonData_NumParticles;

    // The SOFTBODY's tree: mTetMeshData, then a uint4 header, then the packed nodes.
    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gSp.softbodies);
    const uint softbodyId = softbodyShape.particleOrSoftbodyId;
    const uint64_t treeAddr = softbodies[softbodyId].mTetMeshData + 16UL;
    // T3: the COLLISION mesh.
    Ptr<float4> tetVerts = Ptr<float4>(softbodies[softbodyId].mPosition_InvMass);
    Ptr<uint4> tetIndices = Ptr<uint4>(softbodies[softbodyId].mTetIndices);

    // T5: a CUBE of half-side cDistance around the particle.
    const float cDistance = Ptr<float>(gSp.contactDistance)[particleCacheRef] +
                            Ptr<float>(gSp.contactDistance)[softbodyCacheRef];

    const uint stackCapacity = gSp.stackSizeBytes / 16u;

    // The block's two warps, walked serially: the CUDA block is (32, 2, 1) and this
    // group is one warp, so the group covers both of the block's warps in turn.
    for (uint sub = 0u; sub < MIDPHASE_WARPS_PER_BLOCK; ++sub)
    {
    const uint globalWarpIdx = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub;

    // T9: strided by WARPS.
    for (uint particleIndex = globalWarpIdx; particleIndex < nbParticles;
         particleIndex += SBPS_NB_WARPS)
    {
        GroupMemoryBarrierWithGroupSync();   // groupshared scratch is reused per query

        // T7: world space on both sides -- no transform, no scale.
        // P9.49: safe ADDRESS for the read this platform issues even when the loop
        // condition admits no lane.  This is the softbody twin of the cloth site
        // that faulted (femClothPsMidphaseBda), line for line.
        const float3 center = sortedPose[min(particleIndex, max(nbParticles, 1u) - 1u)].xyz;
        const float3 qMin = center - cDistance;
        const float3 qMax = center + cDistance;

        uint currentNodeIndex = 0u;   // root
        bool hasNode = true;
        uint nbStack = 0u;

        while (hasNode)
        {
            const uint nodeIdx = bvNodeIdxOf(currentNodeIndex);

            if (bvIsLeaf(currentNodeIndex))
            {
                const uint leafOffset = bvLeafSlotOf(currentNodeIndex);
                const uint data = bvData(treeAddr, nodeIdx, leafOffset);
                const uint nbPrimitives = bvDataNbPrimitives(data);
                const uint primStartIndex = bvDataPrimStart(data);
                // T6: clamp the ADDRESS, leave the VALUE to the guard.
                uint primitiveIndex = 0xffffffffu;
                if (lane < nbPrimitives)
                    primitiveIndex = lane + primStartIndex;
                const uint safePrim = min(primitiveIndex,
                                          primStartIndex + max(nbPrimitives, 1u) - 1u);

                bool intersect = false;
                if (primitiveIndex != 0xffffffffu)
                {
                    const uint4 tet = tetIndices[safePrim];
                    const float3 w0 = tetVerts[tet.x].xyz;
                    const float3 w1 = tetVerts[tet.y].xyz;
                    const float3 w2 = tetVerts[tet.z].xyz;
                    const float3 w3 = tetVerts[tet.w].xyz;

                    // T2: the validity gate the cloth sibling does not have.
                    if (isValidTet(w0, w1, w2, w3))
                    {
                        // tetBoundingBox(v0..v3).intersects(mBox)
                        const float3 tMin = min(min(w0, w1), min(w2, w3));
                        const float3 tMax = max(max(w0, w1), max(w2, w3));
                        intersect = !(tMin.x > qMax.x || qMin.x > tMax.x ||
                                      tMin.y > qMax.y || qMin.y > tMax.y ||
                                      tMin.z > qMax.z || qMin.z > tMax.z);
                    }
                }

                // T1: (cmIdx, PARTICLE, TET, 0).
                spPushFullWarp(intersect, lane, stackCapacity,
                               uint4(cmIdx, particleIndex, primitiveIndex, 0u));
            }
            else
            {
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
                    intersect = !(mn.x > qMax.x || qMin.x > mx.x ||
                                  mn.y > qMax.y || qMin.y > mx.y ||
                                  mn.z > qMax.z || qMin.z > mx.z);

                uint currentIndex = bvChildIndex(currentNodeIndex, lane);
                bool isValidNode = false;
                if (lane < nbChildren)
                {
                    const uint d = bvData(treeAddr, nodeIdx, lane);
                    isValidNode = intersect;
                    if (isValidNode && bvDataIsInternal(d))
                        currentIndex = bvDataChildNodeIndex(d);
                }

                const uint mask = spBallot(isValidNode, lane);
                const uint offset = countbits(mask & ((1u << lane) - 1u));
                const uint validCount = countbits(mask);
                if (isValidNode)
                    sSpStack[nbStack + offset] = int(currentIndex);
                nbStack += validCount;
            }

            GroupMemoryBarrierWithGroupSync();
            if (nbStack > 0u)
            {
                nbStack -= 1u;
                currentNodeIndex = uint(sSpStack[nbStack]);
            }
            else
                hasNode = false;
        }
    }
    }
}
