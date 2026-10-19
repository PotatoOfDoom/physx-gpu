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

// softBodyHfContactGenBda.slang -- P9.81.
//
//   sb_heightfieldContactGenLaunch (softbodyPrimitives.cu:1139) on the engine
//   launch ABI -- step 2 of the softbody-vs-HEIGHTFIELD pair, whose step 1
//   (sb_heightfieldMidphaseGeneratePairsLaunch) has been mapped since P8.11.
//
// Per midphase pair (cmIdx, tetIdx, triIdx) the kernel runs GJK/EPA between the
// world-space TETRAHEDRON and one world-space HEIGHTFIELD TRIANGLE and appends to
// the rigid-vs-deformable contact stream.
//
// ---------------------------------------------------------------------------
// WHY THIS WAS NEVER LAUNCHED
// ---------------------------------------------------------------------------
// Nothing in the kernel or its launcher gates it: PxgNarrowphaseCore.cpp:5461 fires
// it unconditionally, and the only guard is `if (numTests == 0) return;` on the
// GPU_BUCKET_ID::eSoftbodyHeightfield bucket
// (PxgNphaseImplementationContext.cpp:85-86).  A pair enters that bucket through
// caseTetraHeightfield (:1392, wired at :1522) with NO compatibility test at all --
// an overlapping AABB between a PxDeformableVolume and a PxHeightFieldGeometry is
// the whole requirement.  No scene in this harness had ever put those two together:
// addHeightFieldFloor had five call sites and PXG_SB_FLOOR offered box, plane,
// sphere, mesh, convexcore and none.  PXG_SB_FLOOR=heightfield (P9.81a) is the
// sixth, and its first run measured the cube falling straight through.
//
// ---------------------------------------------------------------------------
// THREE DIFFERENCES FROM sb_meshContactGenLaunch, AND ALL THREE MATTER
// ---------------------------------------------------------------------------
// This kernel and softBodyMeshContactGenBda.slang's are the same shape -- same
// pair stack, same LoadShapePairWarp, same computeInnerSphere / tetPrimitivesCollide2,
// same writeVolumeContactFullWarp (5.12) -- and copying the sibling gets two things
// wrong.  (5.6.1 had a third: this kernel had no filter arm.  5.12 added the same
// rigid-volume filter as the mesh path, findRigidVolumeFilter(rigid, heightfield cache
// ref, tetrahedron), ahead of GJK, and both packs are now the same 176 bytes.)
//
//   (1) THE FILTER IS SHARED NOW (5.12) -- kept as a numbered point so (2) and (3)
//       keep their names.
//
//   (2) NO FRONT-FACE GATE.  The mesh path does
//           if (intersect) intersect &&= gjkOutput.direction.dot(n) >= 0
//       (softbodyPrimitives.cu:878-881).  This one does not: :1130 goes straight from
//       the GjkResult to the write.  A contact generated against the BACK of a
//       heightfield triangle is emitted here and dropped there.
//
//   (3) THE NORMAL WRITTEN IS THE TRIANGLE NORMAL, NOT THE GJK DIRECTION.  The mesh
//       path passes `gjkOutput.direction.getNormalized()` (:886); this one passes
//       `&n.x` (:1136), the triangle's own geometric normal.  That changes the stored
//       normal AND the penetration, because writeVolumeContactFullWarp
//       computes pen as (witnessB - witnessA) . normalPtr -- the witness separation
//       projected on whatever normal it was handed.  Using the GJK direction here
//       would still produce contacts, just at a different depth, which is exactly the
//       kind of wrong a scene-level settle number hides and a fixture does not.
//
// Both call sites pass flipOutputNormal = true, so the stored xyz is -n.  The
// penetration is computed from the UNFLIPPED normal and is not negated with it.
//
// ---------------------------------------------------------------------------
// THE TRIANGLE COMES OUT OF THE SAMPLE GRID ALREADY SCALED
// ---------------------------------------------------------------------------
// getTriangle(..., NULL, triangleIdx, heightfieldShape.scale, nbRows, nbCols, samples)
// returns the triangle in the heightfield's SHAPE space with the scale already
// applied, so there is NO vertex2Shape on this path -- the same shape
// femClothHfContactGenBda.slang documents as its C2.  The port calls the module's
// hfGetTriangle, which is that same decoder including the `wrongHanded` winding swap.
// The three vertices are then rotated by the heightfield's cached transform and
// translated, which is all the CUDA's rotateR(v, q, threadIdx.x) + p does once the
// lane-parallel spelling is unwound.
//
// nbRows is read into the scratch and then DROPPED, which is checked rather than
// assumed: getTriangle takes `rows` but uses it only inside `if (adjacencyIndices)`
// (heightfieldUtil.cuh:289-300), and this call site passes NULL.  Every vertex lookup
// underneath indexes by COLUMNS alone.  So the port reads nbCols and skips nbRows, the
// same shape femClothHfContactGenBda has.
//
// ---------------------------------------------------------------------------
// THE CROSS PRODUCT IS THE OTHER WAY ROUND
// ---------------------------------------------------------------------------
// Upstream (:1129-1131) builds v20 = vB[2] - vB[0], v21 = vB[1] - vB[0] and
// n = normalize(cross(v20, v21)) -- the NEGATIVE of the familiar (v1-v0) x (v2-v0)
// winding normal.  Its sibling does the same and says so; it is transcribed, not
// tidied, and here it reaches the solver directly rather than only gating a branch.
//
// ---------------------------------------------------------------------------
// LANE 0 RUNS THE QUERY
// ---------------------------------------------------------------------------
// As in every softbody contact-gen port before this one: the CUDA spreads one pair
// across 32 lanes (cooperative vertex loads, warp-wide GJK), the Slang GJK is scalar,
// so lane 0 runs the whole query and lanes 1..31 return.  The work PARTITION is
// unchanged -- the same (block, warp) pair owns the same stack entries -- so the
// emitted contact set and the atomic total are identical.
//
// THE GRID: SB_HFCG = 1024 blocks of (WARP_SIZE, MIDPHASE_WARPS_PER_BLOCK = 2).  The
// bridge drops blockDim, so the block's two warps are walked serially and the stride
// stays the CUDA `gridDim.x * blockDim.y` = 2048.
//
// THE TAIL: atomicMax(stackSizeNeededOnDevice, numMidphasePairsFound * sizeof(uint4))
// on the UNCLAMPED count, so the host can grow the stack even on a truncated frame.
//
// ---------------------------------------------------------------------------
// THE ONE MUTANT THAT SURVIVES, AND WHY THAT IS A RESULT
// ---------------------------------------------------------------------------
// Reversing the GJK SEARCH SEED -- `triCenter - tetCenter` where the CUDA writes
// `tetCenter - triCenter` -- survives the sweep, and it is not an untested corner.
// The seed is an initial direction, not an input: GJK converges to the unique
// closest-point pair of two convex sets whatever direction it starts from, and only
// the iteration count differs.  The first sweep could have been accused of only
// having SEPARATED pairs, where that is easiest to believe, so the fixture gained a
// penetrating tetrahedron -- the one pair that reaches GJK's OVERLAP arm and
// therefore EPA -- and the mutant survived that too.  Both arms, same result.
// The seed is transcribed as upstream writes it; it is recorded here as measured-
// equivalent rather than left looking like a gap in the sweep.
//
// NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the GJK/EPA support and witness
// arithmetic, computeInnerSphere's plane distances and the triangle normal's
// cross-then-normalize are mul-feeding-add chains that would otherwise fuse to FMAs
// and break parity with the -ffp-contract=off CPU reference.
// ===========================================================================
module softBodyHfContactGenBda;

import pxgSlangSoftBodyDesc;   // PxgSoftBodyGpu
import pxgSlangHeightfield;    // hfGetTriangle
import pxgSlangGjk;            // GjkCloud / GjkOut / GJK_OVERLAP / GJK_CLOSE / GJK_MAX_CLOUD
import pxgSlangTetCollide;     // tetPrimitivesCollide2 / computeInnerSphere
import pxgSlangDefFilter;      // 5.12: findRigidVolumeFilter
import pxgSlangDeformableRef;     // PxgEncodeDeformableTetrahedron
import pxgSlangDeformableWriters; // PxgRigidDbContactWriterGpu

// PxgNpKernelIndices.h:32 / PxgSoftBodyCoreKernelIndices.h:58 — SB_HFCG.
static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;
static const uint SB_HFCG = 1024u;
static const uint SB_HFCG_STRIDE = SB_HFCG * MIDPHASE_WARPS_PER_BLOCK;

static const uint GEOM_TETRAHEDRONMESH = 7u;

// The triangle's inner-sphere radius, upstream's literal (softbodyPrimitives.cu:1113).
static const float TRI_INNER_SPHERE = 0.000001f;

// ---------------------------------------------------------------------------
// Push-constant blob — BYTE-IDENTICAL to the CUDA pack (PxgNarrowphaseCore.cpp:5441).
// { PxReal tolerenceLength, cmInputs*, transformCache*, contactDistance*, restDistance*,
//   gpuShapes*, softbodies*, shapeToRigidRemapTable*, stackPtr*, midphasePairsNum*,
//   PxU32 stackSizeBytes, stackSizeNeededOnDevice*, PxgRigidDbContactWriter writer }
// -> 4 @0, nine pointers @8..@72, PxU32 @80, a pointer @88, and the 80-byte writer at
// @96 -> 176-byte blob.  ABI row { 4,8,8,8,8,8,8,8,8,8,4,8,80 }.
//
// In 5.6.1 this was the sibling's row with filterPairs (8) and nbFilterPairs (4)
// removed; 5.12 removed them from the sibling too, so the two rows coincide.  They stay
// separate rows, so a later divergence shows up in check_launch_abi.py.
// ---------------------------------------------------------------------------
struct SbHfCgParams
{
    float tolerenceLength;          // @0    (ABI parity only — never read)
    uint _pad4;                     // @4
    uint64_t cmInputs;              // @8
    uint64_t transformCache;        // @16
    uint64_t contactDistance;       // @24
    uint64_t restDistance;          // @32
    uint64_t gpuShapes;             // @40
    uint64_t softbodies;            // @48
    uint64_t shapeToRigidRemapTable;// @56
    uint64_t stackPtr;              // @64
    uint64_t midphasePairsNum;      // @72
    uint stackSizeBytes;            // @80
    uint _pad84;                    // @84
    uint64_t stackSizeNeededOnDevice; // @88
    // --- PxgRigidDbContactWriter, by value (PxgDeformableContactWriters.h, 5.12) ---
    uint64_t w_outPoint;                  // @96  float4*
    uint64_t w_outNormalPen;              // @104 float4*
    uint64_t w_outBarycentric;            // @112 float4* (unused: writeVolumeContact)
    uint64_t w_outContactInfo;            // @120 PxgDeformablePairInfo*
    uint64_t w_outRigidMaterialIndex;     // @128 PxU16*
    uint64_t w_totalContactCount;         // @136 PxU32*
    uint64_t w_contactByRigid;            // @144 PxU64*
    uint64_t w_tempContactByRigid;        // @152 PxU32*
    uint64_t w_contactIndexSortedByRigid; // @160 PxU32*
    uint w_maxNumContacts;                // @168 PxU32
    uint _pad172;                         // @172 -> 176
}

[[vk::push_constant]] ConstantBuffer<SbHfCgParams> gHc;

// PxgContactManagerInput (PxgContactManager.h): four PxU32.
struct PxgContactManagerInput { uint shapeRef0, shapeRef1, transformCacheRef0, transformCacheRef1; }

struct PxgShape                 // PxgConvexConvexShape.h, 48 B
{
    float scaleX, scaleY, scaleZ;                     // @0
    float scaleRotX, scaleRotY, scaleRotZ, scaleRotW; // @12
    uint materialIndex;                               // @28
    uint2 hullOrMeshOffset;                           // @32
    uint type;                                        // @40
    uint particleOrSoftbodyId;                        // @44
}

PxgRigidDbContactWriterGpu hcWriter()
{
    PxgRigidDbContactWriterGpu w;
    w.outPoint = gHc.w_outPoint;                 w.outNormalPen = gHc.w_outNormalPen;
    w.outBarycentric = gHc.w_outBarycentric;     w.outContactInfo = gHc.w_outContactInfo;
    w.outRigidMaterialIndex = gHc.w_outRigidMaterialIndex;
    w.totalContactCount = gHc.w_totalContactCount;
    w.contactByRigid = gHc.w_contactByRigid;     w.tempContactByRigid = gHc.w_tempContactByRigid;
    w.contactIndexSortedByRigid = gHc.w_contactIndexSortedByRigid;
    w.maxNumContacts = gHc.w_maxNumContacts;
    return w;
}


// PxQuat::rotate (PxQuat.h:286), literal.
float3 hcQuatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}

// PxVec3::getNormalized() -- no zero guard upstream beyond the magnitude test the
// sibling uses; kept identical to it so check_helper_uniformity.py can see the pair.
float3 hcGetNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}

// LoadShapePairWarp<eTETRAHEDRONMESH, eHEIGHTFIELD>: the softbody is whichever half of
// the pair IS the tetrahedron mesh; the heightfield is the other.  NOT positional --
// unlike femClothHfContactGenBda's C1, this call site routes by geometry type, exactly
// as the mesh sibling does.
void loadSoftbodyHfPair(uint cmIdx, out PxgShape sbShape, out uint sbCacheRef,
                        out PxgShape hfShape, out uint hfCacheRef)
{
    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(gHc.cmInputs)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gHc.gpuShapes);
    uint sRefSb = cin.shapeRef0, cRefSb = cin.transformCacheRef0;
    uint sRefHf = cin.shapeRef1, cRefHf = cin.transformCacheRef1;
    if (shapes[cin.shapeRef0].type != GEOM_TETRAHEDRONMESH)
    {
        sRefSb = cin.shapeRef1; cRefSb = cin.transformCacheRef1;
        sRefHf = cin.shapeRef0; cRefHf = cin.transformCacheRef0;
    }
    sbShape = shapes[sRefSb]; sbCacheRef = cRefSb;
    hfShape = shapes[sRefHf]; hfCacheRef = cRefHf;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void sb_heightfieldContactGenLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    const uint stackSize = gHc.stackSizeBytes / 16u;
    const uint numFound = Ptr<uint>(gHc.midphasePairsNum)[0];
    const uint numPairs = min(numFound, stackSize);

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gHc.softbodies);
    Ptr<float> contactDistance = Ptr<float>(gHc.contactDistance);
    Ptr<float> restDistance = Ptr<float>(gHc.restDistance);

    // Lane 0 runs the whole query (see the header); the other 31 have nothing to do.
    if (lane == 0u)
    {
    for (uint sub = 0u; sub < MIDPHASE_WARPS_PER_BLOCK; ++sub)
    {
    const uint globalWarpIdx = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub;

    for (uint i = globalWarpIdx; i < numPairs; i += SB_HFCG_STRIDE)
    {
        const uint4 curPair = Ptr<uint4>(gHc.stackPtr)[i];
        const uint cmIdx = curPair.x;
        const uint tetrahedronInd = curPair.y;
        const uint triangleIdx = curPair.z;

        PxgShape sbShape, hfShape;
        uint sbCacheRef, hfCacheRef;
        loadSoftbodyHfPair(cmIdx, sbShape, sbCacheRef, hfShape, hfCacheRef);

        const float cDistance = contactDistance[sbCacheRef] + contactDistance[hfCacheRef];
        const uint softbodyId = sbShape.particleOrSoftbodyId;

        // PxNodeIndex is a PxU64; the remap table is indexed by the HEIGHTFIELD's cache ref.
        const uint2 rigidId = Ptr<uint2>(gHc.shapeToRigidRemapTable)[hfCacheRef];

        // (1) 5.12: a filter pair for this rigid and tetrahedron suppresses contact
        // generation between them (a null array is replaced by the softbody descriptor:
        // llvmpipe speculates the first load, see pxgSlangDefFilter).
        if (findRigidVolumeFilter(
                softBodyRigidVolumeFilters(gHc.softbodies + uint64_t(softbodyId) * PXG_SOFTBODY_STRIDE),
                dfNodeU64(rigidId), hfCacheRef, softbodyId, tetrahedronInd))
            continue;

        // ---- cloud A: the tetrahedron, already in WORLD space ---------------------
        const uint4 tetIdx = Ptr<uint4>(softbodies[softbodyId].mTetIndices)[tetrahedronInd];
        Ptr<float4> tetVerts = Ptr<float4>(softbodies[softbodyId].mPosition_InvMass);

        GjkCloud a;
        for (uint k = 0u; k < GJK_MAX_CLOUD; ++k) a.v[k] = float3(0.0f, 0.0f, 0.0f);
        a.n = 4u;
        a.v[0] = tetVerts[tetIdx.x].xyz;
        a.v[1] = tetVerts[tetIdx.y].xyz;
        a.v[2] = tetVerts[tetIdx.z].xyz;
        a.v[3] = tetVerts[tetIdx.w].xyz;

        float3 tetCenter;
        const float innerSphere = computeInnerSphere(a.v[0], a.v[1], a.v[2], a.v[3], tetCenter);

        // ---- cloud B: the heightfield triangle, out of the sample grid ------------
        // The heightfield blob: nbRows @0, nbCols @4, then the samples.  Upstream reads
        // nbRows into its scratch and never uses it (getTriangle indexes by COLUMNS).
        const uint64_t hfPtr = uint64_t(hfShape.hullOrMeshOffset.x) |
                               (uint64_t(hfShape.hullOrMeshOffset.y) << 32);
        const uint nbCols = Ptr<uint>(hfPtr + 4UL)[0];
        const uint64_t samples = hfPtr + 8UL;

        // ALREADY IN SHAPE SPACE -- no vertex2Shape.  See the header.
        float3 t0, t1, t2;
        hfGetTriangle(samples, triangleIdx,
                      float3(hfShape.scaleX, hfShape.scaleY, hfShape.scaleZ), nbCols,
                      t0, t1, t2);

        // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.
        const uint64_t xf = gHc.transformCache + uint64_t(hfCacheRef) * 32UL;
        const float4 hfQ = Ptr<float4>(xf)[0];
        const float3 hfP = float3(Ptr<float>(xf + 16UL)[0],
                                  Ptr<float>(xf + 20UL)[0],
                                  Ptr<float>(xf + 24UL)[0]);

        GjkCloud b;
        for (uint k = 0u; k < GJK_MAX_CLOUD; ++k) b.v[k] = float3(0.0f, 0.0f, 0.0f);
        b.n = 3u;
        b.v[0] = hcQuatRotate(hfQ, t0) + hfP;
        b.v[1] = hcQuatRotate(hfQ, t1) + hfP;
        b.v[2] = hcQuatRotate(hfQ, t2) + hfP;

        const float3 triCenter = (b.v[0] + b.v[1] + b.v[2]) * (1.0f / 3.0f);

        // *** THE CROSS PRODUCT IS THE OTHER WAY ROUND. ***  See the header.
        const float3 v20 = b.v[2] - b.v[0];
        const float3 v21 = b.v[1] - b.v[0];
        const float3 n = hcGetNormalized(cross(v20, v21));

        GjkOut o;
        const uint result = tetPrimitivesCollide2(a, b, tetCenter - triCenter, cDistance,
                                                  innerSphere, TRI_INNER_SPHERE, o);

        // DIFFERENCE (2): no front-face gate.  The mesh sibling adds
        // `intersect &&= dot(o.direction.xyz, n) >= 0` here; this kernel does not.
        const bool intersect = (result == GJK_OVERLAP || result == GJK_CLOSE);
        if (!intersect)
            continue;

        // DIFFERENCE (3): the normal handed to the writer is the TRIANGLE normal, so
        // the penetration is the witness separation projected on IT and not on the GJK
        // direction.  The stored xyz is -n (flipOutputNormal = true); the penetration is
        // computed from the unflipped n and is NOT negated with it.
        const float3 wA = o.closestPointA.xyz;
        const float3 wB = o.closestPointB.xyz;
        // P9.86: the CUDA sums the three lanes and then adds penOffset, a literal 0 here
        // (softbodyPrimitives.cu:339, :1136).  That IEEE `x + 0` turns a -0 into +0, and
        // slangc/llvmpipe would drop it, so it is spelled as a select.
        const float penSum = dot(wB - wA, n);
        const float pen = (penSum == 0.0f) ? 0.0f : penSum;

        uint index;
        InterlockedAdd(Ptr<uint>(gHc.w_totalContactCount)[0], 1u, index);
        if (index >= gHc.w_maxNumContacts)
            continue;

        // 5.12: writeSortKey(index, rigidRef) on every path (5.6.1's alternativeWriteMode is gone).
        hcWriter().writeVolumeContact(index,
                                      float4(wB.x, wB.y, wB.z, restDistance[cmIdx]),
                                      float4(-n.x, -n.y, -n.z, pen),
                                      dfNodeU64(rigidId), hfShape.materialIndex,
                                      PxgEncodeDeformableTetrahedron(softbodyId, tetrahedronInd));
    }
    }

    // The tail reports the UNCLAMPED demand, so the host can grow the stack for the next
    // frame even when this one was truncated.  One thread does it, as CUDA does.
    if (gid.x == 0u)
        InterlockedMax(Ptr<uint>(gHc.stackSizeNeededOnDevice)[0], numFound * 16u);
    }
}
