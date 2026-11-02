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

// ===========================================================================
// softBodyClothMidphaseBda — sb_clothMidphaseGeneratePairsLaunch
// (softbodySoftbodyMidPhase.cu:1444, core at :1299), on the engine BDA ABI.
//
// The first kernel of the softbody-vs-cloth chain.  One 32-lane warp per query
// TETRAHEDRON of the softbody; the tree walked is the CLOTH's BV32 triangle BVH.  Each warp
// keeps the FOUR closest triangles it saw and appends them to the shared midphase stack.
//
// THE TRAVERSER IS TriangleLeafBoundMinMaxTraverser<false>, already transcribed once as the
// descriptor-set kernel `ctTetTriMid` (npTetTriMid.slang) — its header carries the detailed
// derivation of satIntersect, the evolving tolerance and the top-4 extraction, and this port
// is the same logic reached through Ptr<> instead of through bound buffers.  The one piece
// npTetTriMid does NOT have is the driver around it, which is what this file adds:
//
//   * LoadShapePairWarp<eTETRAHEDRONMESH, eTRIANGLEMESH> — the softbody is whichever half of
//     the pair IS the tet mesh, and the cloth is the other.  Getting it backwards reads a
//     cloth id as a softbody index;
//   * the cloth's tree: mTriMeshData skips one uint4 (nbVerts_nbPrimitives_maxDepth_
//     nbBv32TreeNodes) and the BV32 node array starts there;
//   * the SURFACE-HINT gate — a tetrahedron with hint == 0 is interior and generates nothing;
//   * the transform into CLOTH space.  The .cu does `clothToWorld.transformInv(worldV)` on
//     all four tet corners before building the bound, because the BVH is in the cloth's local
//     frame.  For a deformable that transform is usually identity, which is exactly why it is
//     transcribed rather than dropped: "usually identity" is not "identity".
//
// GRID.  The engine launches gridDim = (SB_SBMIDPHASE = 1024, numTests, 1) with blockDim =
// (32, MIDPHASE_WARPS_PER_BLOCK = 2, 1) (PxgNarrowphaseCore.cpp).  The bridge forwards the grid
// and ignores blockDim, so blockIdx.y is the contact-manager index.  The block's two warps used
// to be walked SERIALLY here; the launch site now (PXG_NO_CUDA_TOOLKIT) launches one group per
// CUDA warp, (SB_SBMIDPHASE * 2, numTests, 1), so the warp index is gid.x and the stride the
// group count WorkgroupCount().x -- the CUDA's blockDim.y * gridDim.x = 2048 at that grid, and
// at any other grid still a stride that visits every tetrahedron exactly once.
//
// NoContraction-stamped: satIntersect is dot-of-cross throughout and the plane distances are
// mul-feeding-add chains; getNormalized's rsqrt only picks a direction and never a decision.
// ===========================================================================
module softBodyClothMidphaseBda;

import pxgSlangSoftBodyDesc;   // PxgSoftBodyGpu
import pxgSlangFEMClothDesc;   // PxgFEMClothGpu
import pxgSlangBv32;   // P9.17b: the BV32 reader AND its index rules, replacing a
                      // private copy of both that was verbatim identical to it.

// PxgNpKernelIndices.h:32 / PxgSoftBodyCoreKernelIndices.h:55.
static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;
static const uint SB_SBMIDPHASE = 1024u;

static const uint WARP = 32u;
static const float PX_MAX_F32 = 3.4028235e38f;

// femMidphaseScratch.sBv32Nodes[192] — a 6-deep BV32 tree's worst-case pending set.
static const uint STACK_SIZE = 192u;

// The traverser keeps at most four triangles per query tetrahedron
// (softbodySoftbodyMidPhase.cu:1405, `const PxU32 MaxContacts = 4`).
static const uint MAX_CONTACTS = 4u;

static const uint GEOM_TETRAHEDRONMESH = 7u;

// ---------------------------------------------------------------------------
// Push-constant blob — BYTE-IDENTICAL to the CUDA pack (PxgNarrowphaseCore.cpp:4597):
// { PxReal toleranceLength, cmInputs*, transformCache*, bounds*, contactDistance*,
//   gpuShapes*, softbodies*, clothes*, PxU32 stackSizeBytes, stackPtr*,
//   midphasePairsNum* }.
// Natural alignment: 4 @0, then seven pointers @8..@56, PxU32 @64, then a pointer realigned
// to @72 and @80 -> 88-byte blob.  ABI row {4,8,8,8,8,8,8,8,4,8,8}.
// ---------------------------------------------------------------------------
struct SbClothMidphaseParams
{
    float toleranceLength;          // @0   (ABI parity only — the traverser never reads it)
    uint _pad4;                     // @4
    uint64_t cmInputs;              // @8
    uint64_t transformCache;        // @16
    uint64_t bounds;                // @24  (ABI parity only)
    uint64_t contactDistance;       // @32
    uint64_t gpuShapes;             // @40
    uint64_t softbodies;            // @48
    uint64_t clothes;               // @56
    uint stackSizeBytes;            // @64
    uint _pad68;                    // @68
    uint64_t stackPtr;              // @72
    uint64_t midphasePairsNum;      // @80 -> 88
}

[[vk::push_constant]] ConstantBuffer<SbClothMidphaseParams> gCm;

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

// A PxU8 array read through the containing 32-bit word.  The `& ~uint64_t(3)` is load-bearing
// on llvmpipe, which faults on a sub-word BDA load that straddles nothing at all.
uint byteAt(uint64_t base, uint idx)
{
    const uint64_t addr = base + uint64_t(idx);
    const uint word = Ptr<uint>(addr & ~uint64_t(3))[0];
    const uint shift = uint(addr & 3UL) * 8u;
    return (word >> shift) & 0xffu;
}

// ---------------------------------------------------------------------------
// Geometry, transcribed from npTetTriMid.slang (which took it from the .cu).
// ---------------------------------------------------------------------------
float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}

bool isValidTet(float3 v0, float3 v1, float3 v2, float3 v3)
{
    return dot(v1 - v0, cross(v2 - v0, v3 - v0)) > 1e-9f;
}

// PxPlane(p0, p1, p2): n = normalize(cross(p1-p0, p2-p0)), d = -p0.dot(n).
void makePlane(float3 p0, float3 p1, float3 p2, out float3 n, out float d)
{
    n = getNormalized(cross(p1 - p0, p2 - p0));
    d = -dot(p0, n);
}

// Gu::createPlane: flip so pd lies on the non-positive side.
void createPlane(float3 pa, float3 pb, float3 pc, float3 pd, out float3 n, out float d)
{
    makePlane(pa, pb, pc, n, d);
    if (dot(n, pd) + d > 0.0f) { n = -n; d = -d; }
}

// P9.89: EVERY FOLD OF THE SAT AND OF THE TET BOUND IS A SELECT, NOT min()/max().
// softbodySoftbodyMidPhase.cu folds with PxMin/PxMax<float>, and so does tetBoundingBox
// (deformableCollision.cuh:64-94); both are selectMin(a, b) = a < b ? a : b and
// selectMax(a, b) = a > b ? a : b (PxMath.h:82-100, PxUnixMathIntrinsics.h:116-125).  An equal
// compare returns b, so a +-0 tie keeps b's zero; a NaN in either operand fails the compare, so b
// is returned -- a NaN b sticks and a NaN a is dropped.  The port spelled them min()/max(), i.e.
// GLSL.std.450 FMin/FMax, which llvmpipe evaluates as "the non-NaN operand" with -0/+0 on a zero
// tie.  Nesting and operand order were already the CUDA's; only the operator differed.
// Kernel-visible through a NaN projection: a triangle whose THIRD vertex has an infinite
// coordinate (and whose first edge has a zero component off that axis) passes the front-face
// gate -- its edge cross holds 0 * inf = NaN, getNormalized's `m > 0` fails, the normal is the
// zero vector and the gate reads +-0 >= 0 -- and every tet plane whose normal has an exact zero
// in that coordinate projects the vertex as 0 * inf = NaN.  minProject then returns NaN (the
// CUDA's PxMin(d1, NaN) = NaN sticks), satIntersect's sep = PxMax(sep, NaN) WIPES the running
// separation, and the next plane RESETS it -- a NaN last plane drops the triangle.  All-finite
// input gets there too: a tet with two ~2e19 edges overflows the cross products of faces 0 and 3
// and their normals come out (0, 0, NaN).  FMin/FMax skipped every NaN instead.
// testNpTetTriMid softBodySimClothMidphase_p89NaNSelect drives that through this kernel, and
// softBodySimClothMidphase_p89SatProbe reads every fold, tetBoundingBox included, out bit for bit
// through sb_clothMidphaseSatProbe.
float sbcSelMin(float a, float b) { return (a < b) ? a : b; }
float sbcSelMax(float a, float b) { return (a > b) ? a : b; }
float3 sbcSelMin3(float3 a, float3 b) { return float3(sbcSelMin(a.x, b.x), sbcSelMin(a.y, b.y), sbcSelMin(a.z, b.z)); }
float3 sbcSelMax3(float3 a, float3 b) { return float3(sbcSelMax(a.x, b.x), sbcSelMax(a.y, b.y), sbcSelMax(a.z, b.z)); }

// minProject(plane, v0, v1, v2) (softbodySoftbodyMidPhase.cu:72-76).
// P9.89: PxMin(plane.distance(v0), PxMin(plane.distance(v1), plane.distance(v2))), both selects;
// the `+ d` is inside each distance.
float minProject3(float3 n, float d, float3 v0, float3 v1, float3 v2)
{
    return sbcSelMin(dot(n, v0) + d, sbcSelMin(dot(n, v1) + d, dot(n, v2) + d));
}

// satIntersect<false>: the triangle plane term, then the four tet planes, with an early-out
// whenever sep exceeds the tolerance -- and the tolerance EVOLVES with the running best, so
// the traversal gets cheaper as it finds closer triangles.
// NOTE WHAT IS *NOT* HERE.  satIntersect<BackFaceCull> culls internally only when the
// template argument is true, and this call site is the <false> instantiation -- the cull
// lives at the CALLER, as the outer gate below.  npTetTriMid.slang keeps the internal copy
// because it stands in for both instantiations; carrying it here too made the outer gate
// provably redundant, and a mutant that deleted the gate survived because the duplicate still
// culled.  One gate, in the place the .cu puts it.
// P9.89: each sep = PxMax(sep, minProject(tet0.planes[k], v0, v1, v2))
// (softbodySoftbodyMidPhase.cu:93, :96, :99, :102) is a select, running value first: a NaN
// projection wipes sep, and the next plane's value replaces it.
float satIntersectTetTri(float3 pn[4], float pd[4], float3 centroid,
                         float3 v0, float3 v1, float3 v2, float tolerance)
{
    float3 n; float d;
    makePlane(v0, v1, v2, n, d);
    float sep = minProject3(n, d, v0, v1, v2);
    if (sep > tolerance) return sep;
    sep = sbcSelMax(sep, minProject3(pn[0], pd[0], v0, v1, v2));
    if (sep > tolerance) return sep;
    sep = sbcSelMax(sep, minProject3(pn[1], pd[1], v0, v1, v2));
    if (sep > tolerance) return sep;
    sep = sbcSelMax(sep, minProject3(pn[2], pd[2], v0, v1, v2));
    if (sep > tolerance) return sep;
    sep = sbcSelMax(sep, minProject3(pn[3], pd[3], v0, v1, v2));
    return sep;
}

// tetBoundingBox (deformableCollision.cuh:64-94), called by computeTetBoundWithTetrahedron
// (softbodySoftbodyMidPhase.cu:655-670).  P9.89: per axis tX0 = PxMin(v0, v1), tX1 = PxMin(v2, v3),
// minimum = PxMin(tX0, tX1) (:70-80), and the PxMax twin for the maximum (:83-93) -- the pairwise
// tree, as selects.  Output-equivalent in the kernel (a NaN corner already failed isValidTet, and
// intersectBoxFullWarp cannot see a zero's sign), bit-exact as a function: the probe reads it.
void tetBoundingBox(float3 t0, float3 t1, float3 t2, float3 t3, out float3 tMin, out float3 tMax)
{
    tMin = sbcSelMin3(sbcSelMin3(t0, t1), sbcSelMin3(t2, t3));
    tMax = sbcSelMax3(sbcSelMax3(t0, t1), sbcSelMax3(t2, t3));
}

// PxTransform::transformInv(p) = q.rotateInv(p - t).  PxQuat::rotateInv (PxQuat.h:324).
float3 cmQuatRotateInv(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}

// LoadShapePairWarp<eTETRAHEDRONMESH, eTRIANGLEMESH>: the softbody is whichever half of the
// pair IS the tetrahedron mesh; the cloth is the other.
void loadSoftbodyClothPair(uint64_t cmInputsAddr, uint64_t shapesAddr, uint cmIdx,
                           out PxgShape sbShape, out uint sbCacheRef,
                           out PxgShape clothShape, out uint clothCacheRef)
{
    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(cmInputsAddr)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(shapesAddr);
    uint sRefSb = cin.shapeRef0, cRefSb = cin.transformCacheRef0;
    uint sRefCl = cin.shapeRef1, cRefCl = cin.transformCacheRef1;
    if (shapes[cin.shapeRef0].type != GEOM_TETRAHEDRONMESH)
    {
        sRefSb = cin.shapeRef1; cRefSb = cin.transformCacheRef1;
        sRefCl = cin.shapeRef0; cRefCl = cin.transformCacheRef0;
    }
    sbShape = shapes[sRefSb]; sbCacheRef = cRefSb;
    clothShape = shapes[sRefCl]; clothCacheRef = cRefCl;
}

// ---------------------------------------------------------------------------
// Warp emulation scratch: one group == one warp, so a single slot each.
// ---------------------------------------------------------------------------
groupshared int sCmStack[STACK_SIZE];   // bv32 to-be-expanded node stack
groupshared uint sCmBallot;             // __ballot_sync mask
groupshared float sCmBest[WARP];        // minIndex reduction over the lane bests
groupshared uint sCmIndex0;             // atomicAdd result broadcast

uint cmBallot(bool pred, uint lane)
{
    if (lane == 0u) sCmBallot = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sCmBallot, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sCmBallot;
}

// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void sb_clothMidphaseGeneratePairsLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const uint cmIdx = gid.y;                      // blockIdx.y — one pair per y-block

    PxgShape sbShape, clothShape;
    uint sbCacheRef, clothCacheRef;
    loadSoftbodyClothPair(gCm.cmInputs, gCm.gpuShapes, cmIdx,
                          sbShape, sbCacheRef, clothShape, clothCacheRef);

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gCm.softbodies);
    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gCm.clothes);
    const uint softbodyId = sbShape.particleOrSoftbodyId;
    const uint clothId = clothShape.particleOrSoftbodyId;

    // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.
    const uint64_t xf = gCm.transformCache + uint64_t(clothCacheRef) * 32UL;
    const float4 clothQ = Ptr<float4>(xf)[0];
    const float3 clothP = float3(Ptr<float>(xf + 16UL)[0],
                                 Ptr<float>(xf + 20UL)[0],
                                 Ptr<float>(xf + 24UL)[0]);

    // mTriMeshData starts with uint4(nbVerts, nbPrimitives, maxDepth, nbBv32TreeNodes); the
    // BV32 node array begins immediately after it.
    const uint64_t treeAddr = clothes[clothId].mTriMeshData + 16UL;
    Ptr<float4> triVerts = Ptr<float4>(clothes[clothId].mPosition_InvMass);
    Ptr<uint4> triIndices = Ptr<uint4>(clothes[clothId].mTriangleVertexIndices_ownedEdges);

    const float contactDist = Ptr<float>(gCm.contactDistance)[sbCacheRef] +
                              Ptr<float>(gCm.contactDistance)[clothCacheRef];

    const uint nbTets = softbodies[softbodyId].mNumTets;
    Ptr<uint4> tetIndices = Ptr<uint4>(softbodies[softbodyId].mTetIndices);
    Ptr<float4> tetVerts = Ptr<float4>(softbodies[softbodyId].mPosition_InvMass);
    const uint64_t surfaceHintBase = softbodies[softbodyId].mTetMeshSurfaceHint;

    const uint stackCapacity = gCm.stackSizeBytes / 16u;

    // One group per CUDA warp (see the header on the grid): globalWarpIdx is the group
    // index and the stride the group count, i.e. blockDim.y * gridDim.x.
    {
    const uint globalWarpIdx = gid.x;

    for (uint tetrahedronIdx = globalWarpIdx; tetrahedronIdx < nbTets; tetrahedronIdx += WorkgroupCount().x)
    {
        GroupMemoryBarrierWithGroupSync();   // groupshared scratch is reused per query

        // P9.49: safe ADDRESS for the reads this platform issues even when the loop
        // condition admits no lane (see femClothPsMidphaseBda, where it faulted).
        // BOTH reads below need it -- the surface hint is read before the `continue`,
        // so it is the first one out of range, not the index fetch.
        const uint safeTet = min(tetrahedronIdx, max(nbTets, 1u) - 1u);

        // Interior tetrahedra never generate contacts.
        if (byteAt(surfaceHintBase, safeTet) == 0u)
            continue;

        const uint4 tetIdx = tetIndices[safeTet];

        // World -> cloth local, all four corners (see the header on why this stays).
        const float3 t0 = cmQuatRotateInv(clothQ, tetVerts[tetIdx.x].xyz - clothP);
        const float3 t1 = cmQuatRotateInv(clothQ, tetVerts[tetIdx.y].xyz - clothP);
        const float3 t2 = cmQuatRotateInv(clothQ, tetVerts[tetIdx.z].xyz - clothP);
        const float3 t3 = cmQuatRotateInv(clothQ, tetVerts[tetIdx.w].xyz - clothP);

        // computeTetBoundWithTetrahedron: the isValidTet gate, the AABB, and the four
        // outward planes plus the centroid the SAT term needs.
        if (!isValidTet(t0, t1, t2, t3))
            continue;

        float3 tMin, tMax;
        tetBoundingBox(t0, t1, t2, t3, tMin, tMax);

        float3 pn[4]; float pd[4];
        createPlane(t0, t1, t2, t3, pn[0], pd[0]);
        createPlane(t0, t1, t3, t2, pn[1], pd[1]);
        createPlane(t0, t2, t3, t1, pn[2], pd[2]);
        createPlane(t1, t2, t3, t0, pn[3], pd[3]);
        // constructTetrahedron's centroid is (a + b + c) / 3 — the first THREE corners, not
        // all four.  Transcribed, not corrected.
        const float3 centroid = (t0 + t1 + t2) * (1.0f / 3.0f);

        // --- TriangleLeafBoundMinMaxTraverser<false> per-lane state ---
        float bestDist = PX_MAX_F32;
        uint bestPrimIndex = 0xffffffffu;

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

                if (primitiveIndex != 0xffffffffu)
                {
                    const uint4 triIdx = triIndices[safePrim];
                    const float3 v0 = triVerts[triIdx.x].xyz;
                    const float3 v1 = triVerts[triIdx.y].xyz;
                    const float3 v2 = triVerts[triIdx.z].xyz;

                    float3 tn; float td;
                    makePlane(v0, v1, v2, tn, td);

                    // The OUTER front-face gate.  Both engine instantiations sit behind it,
                    // which is what makes satIntersect's own BackFaceCull dead code.
                    if (dot(tn, centroid) + td >= 0.0f)
                    {
                        // P9.89: PxMin(mBestDistance, mContactDistance)
                        // (softbodySoftbodyMidPhase.cu:153), a select.  Output-equivalent:
                        // bestDist is never NaN, a NaN contactDist visits no leaf, and
                        // `sep > tolerance` cannot see a +-0 tie.
                        const float separation =
                            satIntersectTetTri(pn, pd, centroid, v0, v1, v2,
                                               sbcSelMin(bestDist, contactDist));
                        if (separation < bestDist)
                        {
                            bestDist = separation;
                            bestPrimIndex = primitiveIndex;
                        }
                    }
                }
            }
            else
            {
                // intersectBoxFullWarp: the per-axis AABB-vs-AABB displacement.
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
                    float3 dsp;
                    dsp.x = (mn.x > tMax.x) ? (mn.x - tMax.x) : ((mx.x < tMin.x) ? (mx.x - tMin.x) : 0.0f);
                    dsp.y = (mn.y > tMax.y) ? (mn.y - tMax.y) : ((mx.y < tMin.y) ? (mx.y - tMin.y) : 0.0f);
                    dsp.z = (mn.z > tMax.z) ? (mn.z - tMax.z) : ((mx.z < tMin.z) ? (mx.z - tMin.z) : 0.0f);
                    intersect = dot(dsp, dsp) < contactDist * contactDist;
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

                const uint mask = cmBallot(isValidNode, lane);
                const uint offset = countbits(mask & ((1u << lane) - 1u));
                const uint validCount = countbits(mask);
                if (isValidNode)
                    sCmStack[nbStack + offset] = int(currentIndex);
                nbStack += validCount;
            }

            GroupMemoryBarrierWithGroupSync();
            if (nbStack > 0u)
            {
                nbStack -= 1u;
                currentNodeIndex = uint(sCmStack[nbStack]);
            }
            else
                hasNode = false;
        }

        // --- the top-4 extraction (softbodySoftbodyMidPhase.cu:1405) ---------------
        GroupMemoryBarrierWithGroupSync();
        sCmBest[lane] = bestDist;
        const uint mask = cmBallot(bestDist < contactDist, lane);
        uint count = countbits(mask);
        if (count > MAX_CONTACTS) count = MAX_CONTACTS;

        if (count == 0u)
            continue;

        // ONE atomicAdd of the whole count, broadcast to the warp.
        if (lane == 0u)
        {
            uint idx;
            InterlockedAdd(Ptr<uint>(gCm.midphasePairsNum)[0], count, idx);
            sCmIndex0 = idx;
        }
        GroupMemoryBarrierWithGroupSync();
        const uint index = sCmIndex0;

        float bestSep = -1.0f;
        for (uint i = 0u; i < count && bestSep < contactDist; ++i)
        {
            // minIndex: the warp argmin, lowest lane winning ties.
            uint winnerLane = 0u;
            float best = PX_MAX_F32;
            for (uint j = 0u; j < WARP; ++j)
                if (sCmBest[j] < best) { best = sCmBest[j]; winnerLane = j; }
            bestSep = best;

            GroupMemoryBarrierWithGroupSync();
            if (winnerLane == lane && sCmBest[lane] < contactDist)
            {
                if ((index + i) < stackCapacity)
                    Ptr<uint4>(gCm.stackPtr)[index + i] =
                        uint4(cmIdx, tetrahedronIdx, bestPrimIndex, 0u);
                sCmBest[lane] = PX_MAX_F32;
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    }
}

// ===========================================================================
// sb_clothMidphaseSatProbe -- P9.89: the SAT folds and tetBoundingBox read out bit for bit.
// Test-only entry (never in the manifest); the engine kernel above is the only engine consumer
// of these functions.
//
// WHY A PROBE.  The kernel's only outputs are the (cmIdx, tetIdx, primIndex) triples, so a
// separation reaches them only through `sep > tolerance`, `separation < bestDist`,
// `bestDist < contactDist` and the lane argmin, and the tet bound only through the box test's
// `>`/`<` and a difference taken when it is non-zero -- none of which sees the sign of a zero.
// A NaN projection does reach the pairs (softBodySimClothMidphase_p89NaNSelect), but not at
// every fold: the kernel's route to a NaN projection is an infinite triangle vertex, which also
// makes the tri-plane term NaN, so the planes[0] fold never sees a finite running value there;
// and a NaN tet corner never reaches the bound, because isValidTet rejects it first.  Every
// fold is a transcription of a CUDA select and has to be bit-exact as a function, so this entry
// evaluates them on caller-supplied operands, one record per thread.
//
// Record, SBC_PROBE_IN_FLOAT4S float4s:
//   [0..2]  triangle v0, v1, v2 (xyz)
//   [3..6]  tet planes 0..3 (n.xyz, d)
//   [7]     centroid (xyz, unused by satIntersect<false>), satIntersect tolerance in .w
//   [8]     a lone plane (n.xyz, d)
//   [9..12] tet corners t0..t3 (xyz) for tetBoundingBox
// Output, SBC_PROBE_OUT_FLOAT4S float4s per record:
//   [0] = (minProject3(lone plane, v0, v1, v2),
//          satIntersectTetTri(planes, v0, v1, v2, tolerance),
//          satIntersectTetTri(planes, v0, v1, v2, PX_MAX_F32)   -- the whole chain,
//          minProject3(makePlane(v0, v1, v2), v0, v1, v2))      -- the tri-plane term
//   [1] = (tMin.xyz, tMax.x), [2] = (tMax.yz, 0, 0)
// ===========================================================================
struct SbClothSatProbeParams
{
    uint64_t input;       // @0
    uint64_t output;      // @8
    uint numRecords;      // @16
    uint _pad20, _pad24, _pad28;
}

[[vk::push_constant]] ConstantBuffer<SbClothSatProbeParams> gSbcSatProbe;

static const uint SBC_PROBE_IN_FLOAT4S = 13u;
static const uint SBC_PROBE_OUT_FLOAT4S = 3u;

[shader("compute")]
[numthreads(32, 1, 1)]
void sb_clothMidphaseSatProbe(uint3 dtid : SV_DispatchThreadID)
{
    const uint r = dtid.x;
    if (r >= gSbcSatProbe.numRecords)
        return;

    Ptr<float4> rec = Ptr<float4>(gSbcSatProbe.input + uint64_t(r) * uint64_t(SBC_PROBE_IN_FLOAT4S * 16u));
    const float3 v0 = rec[0].xyz, v1 = rec[1].xyz, v2 = rec[2].xyz;
    float3 pn[4];
    float pd[4];
    for (uint k = 0u; k < 4u; ++k)
    {
        const float4 p = rec[3u + k];
        pn[k] = p.xyz;
        pd[k] = p.w;
    }
    const float4 ct = rec[7];
    const float4 lp = rec[8];

    float3 tn;
    float td;
    makePlane(v0, v1, v2, tn, td);

    float3 tMin, tMax;
    tetBoundingBox(rec[9].xyz, rec[10].xyz, rec[11].xyz, rec[12].xyz, tMin, tMax);

    Ptr<float4> o = Ptr<float4>(gSbcSatProbe.output + uint64_t(r) * uint64_t(SBC_PROBE_OUT_FLOAT4S * 16u));
    o[0] = float4(minProject3(lp.xyz, lp.w, v0, v1, v2),
                  satIntersectTetTri(pn, pd, ct.xyz, v0, v1, v2, ct.w),
                  satIntersectTetTri(pn, pd, ct.xyz, v0, v1, v2, PX_MAX_F32),
                  minProject3(tn, td, v0, v1, v2));
    o[1] = float4(tMin, tMax.x);
    o[2] = float4(tMax.y, tMax.z, 0.0f, 0.0f);
}
