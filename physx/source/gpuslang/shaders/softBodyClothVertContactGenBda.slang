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
// softBodyClothVertContactGenBda — sb_clothVertContactGenLaunch
// (softbodyPrimitives.cu:2577, core sbClothVertCollision at :2449), on the engine BDA ABI.
//
// The consumer of step 3's pairs, and the simplest kernel of the chain: ONE THREAD per
// (cloth vertex, softbody tetrahedron) pair, no warp cooperation of any kind, no GJK.  A
// point-vs-tetrahedron closest-point query answers it outright.
//
// THE INSIDE AND OUTSIDE CASES ARE DIFFERENT CONTACTS, not two spellings of one.
// closestPtTetrahedron gives the closest point on the tet's SURFACE faces (the 4-bit hint
// says which are surface) together with that face's outward normal n2, and `v = closestP - p`
// normalises to `m` = the distance.  Then:
//
//   * the vertex is INSIDE the tet (all four PointOutsideOfPlane4 products >= 0):
//     normal = n2 and pen = -m.  NEGATIVE penetration, and NO contact-distance test -- a
//     vertex already inside is a contact however deep it is;
//   * the vertex is OUTSIDE: reject when sqDist > cDistance^2, then pen = +m and the normal
//     is `v` when it is well-conditioned AND agrees with the face normal past 0.707
//     (45 degrees), otherwise n2.  The 0.707 is upstream's, and the `m > 1e-10f` guard is
//     what keeps a zero-length v from being normalised into garbage.
//
// So the SIGN of pen carries the inside/outside distinction downstream, and both branches
// write the same point -- closestP, the witness on the tet's surface.
//
// THE FILTER IS A THREE-WAY OR (5.12: three exact searches of the softbody's
// mVolumeSurfaceFilterPairs, findVolumeSurfaceVertexFilter): any of (ANY tet, this vertex),
// (this tet, ANY vertex of the cloth) or (this tet, this vertex) kills it.
//
// NO BARYCENTRIC IS WRITTEN (5.12): the vertex overload of
// PxgDbDbContactWriter::writeSurfaceVolumeContact leaves both slots untouched -- the vertex
// needs none, and the tet's is computed by sb_fem_contact_remap_to_simLaunch from the stored
// point.  (5.6.1 wrote the (0,0,0,1) vertex marker and computeTetBarycentric of closestP.)
//
// GRID.  A plain 1D grid-stride loop over the midphase pairs (gridDim.x * blockDim.x).  The
// bridge ignores blockDim, so the stride here is the CUDA blockDim.x * gridDim.x with
// blockDim.x = 32 folded in.
// ===========================================================================
module softBodyClothVertContactGenBda;

import pxgSlangSoftBodyDesc;
import pxgSlangDefFilter;   // 5.12: findVolumeSurfaceVertexFilter
import pxgSlangDeformableRef;     // PxgEncodeDeformable*, PXG_DREF_ANY_FEATURE
import pxgSlangDeformableWriters; // PxgDbDbContactWriterGpu
import pxgSlangFEMClothDesc;
import pxgSlangTetClosest;   // getNormalized / isPointOutsideOfPlane4 / closestPtTetrahedron

static const float PX_MAX_F32 = 3.4028235e38f;
static const uint GEOM_TETRAHEDRONMESH = 7u;



// The engine launches 512 blocks of 256 threads (PxgNarrowphaseCore.cpp:4805).  The bridge
// ignores blockDim and dispatches groups of 32, so the thread-to-pair MAPPING differs from
// CUDA's -- which is harmless here and only here: this is a grid-stride loop over independent
// pairs, so all that matters is that every index below numPairs is visited exactly once, and
// any (base, stride) with base < stride does that.  The kernels that ballot within a warp do
// NOT have that freedom, which is why they walk the block's warps serially instead.
static const uint SB_VERT_CG_GRID = 512u;
static const uint SB_VERT_CG_BLOCK = 32u;
static const uint SB_VERT_CG_STRIDE = SB_VERT_CG_GRID * SB_VERT_CG_BLOCK;

// ---------------------------------------------------------------------------
// Push-constant blob — BYTE-IDENTICAL to the 5.12 CUDA pack:
// { PxReal tolerenceLength, cmInputs*, transformCache*, contactDistance*, gpuShapes*,
//   softbodies*, clothes*, PxU32 stackSizeBytes, stackPtr*, midphasePairsNum*,
//   stackSizeNeededOnDevice*, PxgDbDbContactWriter writer }.
// 4 @0 (padded to 8), six pointers @8..@48, PxU32 @56, then pointers realigned to @64, @72
// and @80, the 56-byte writer at @88 -> 144-byte blob.  ABI row {4,8,8,8,8,8,8,4,8,8,8,56}.
//
// NOTE the stackSizeBytes position: step 2's pack has it AFTER midphasePairsNum, this one
// has it BEFORE stackPtr.  Same twelve arguments, same 144 bytes, different order.
// ---------------------------------------------------------------------------
struct SbClothVertCgParams
{
    float tolerenceLength;          // @0   (ABI parity only)
    uint _pad4;                     // @4
    uint64_t cmInputs;              // @8
    uint64_t transformCache;        // @16  (ABI parity only — nothing is transformed here)
    uint64_t contactDistance;       // @24
    uint64_t gpuShapes;             // @32
    uint64_t softbodies;            // @40
    uint64_t clothes;               // @48
    uint stackSizeBytes;            // @56
    uint _pad60;                    // @60
    uint64_t stackPtr;              // @64
    uint64_t midphasePairsNum;      // @72
    uint64_t stackSizeNeededOnDevice; // @80
    // --- PxgDbDbContactWriter, inlined (PxgDeformableContactWriters.h, 5.12) ---
    uint64_t w_outPoint;            // @88
    uint64_t w_outNormalPen;        // @96
    uint64_t w_outBarycentric0;     // @104 (not written: vertex overload)
    uint64_t w_outBarycentric1;     // @112 (not written: the remap's)
    uint64_t w_outContactInfo;      // @120 PxgDeformablePairInfo*
    uint64_t w_totalContactCount;   // @128
    uint w_maxContacts;             // @136
    uint _pad140;                   // @140 -> 144
}

[[vk::push_constant]] ConstantBuffer<SbClothVertCgParams> gVc;

struct PxgContactManagerInput { uint shapeRef0, shapeRef1, transformCacheRef0, transformCacheRef1; }

struct PxgShape
{
    float scaleX, scaleY, scaleZ;
    float scaleRotX, scaleRotY, scaleRotZ, scaleRotW;
    uint materialIndex;
    uint2 hullOrMeshOffset;
    uint type;
    uint particleOrSoftbodyId;
}

PxgDbDbContactWriterGpu vcWriter()
{
    PxgDbDbContactWriterGpu w;
    w.outPoint = gVc.w_outPoint;               w.outNormalPen = gVc.w_outNormalPen;
    w.outBarycentric0 = gVc.w_outBarycentric0; w.outBarycentric1 = gVc.w_outBarycentric1;
    w.outContactInfo = gVc.w_outContactInfo;   w.totalContactCount = gVc.w_totalContactCount;
    w.maxContacts = gVc.w_maxContacts;
    return w;
}

uint byteAt(uint64_t base, uint idx)
{
    const uint64_t addr = base + uint64_t(idx);
    const uint word = Ptr<uint>(addr & ~uint64_t(3))[0];
    const uint shift = uint(addr & 3UL) * 8u;
    return (word >> shift) & 0xffu;
}

void loadSoftbodyClothPair(uint cmIdx, out PxgShape sbShape, out uint sbCacheRef,
                           out PxgShape clothShape, out uint clothCacheRef)
{
    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(gVc.cmInputs)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gVc.gpuShapes);
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

[shader("compute")]
[numthreads(32, 1, 1)]
void sb_clothVertContactGenLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint globalThreadIdx = gid.x * SB_VERT_CG_BLOCK + gtid.x;

    const uint stackSize = gVc.stackSizeBytes / 16u;
    const uint numFound = Ptr<uint>(gVc.midphasePairsNum)[0];
    const uint numPairs = min(numFound, stackSize);

    // atomicMax(stackSizeNeededOnDevice, numMidphasePairsFound * sizeof(uint4)) -- the engine
    // reads this back to resize the stack, so it reports what was FOUND, not what fitted.
    if (globalThreadIdx == 0u)
        InterlockedMax(Ptr<uint>(gVc.stackSizeNeededOnDevice)[0], numFound * 16u);

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gVc.softbodies);
    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gVc.clothes);
    Ptr<float> contactDistance = Ptr<float>(gVc.contactDistance);

    for (uint i = globalThreadIdx; i < numPairs; i += SB_VERT_CG_STRIDE)
    {
        const uint4 curPair = Ptr<uint4>(gVc.stackPtr)[i];
        const uint cmIdx = curPair.x;
        const uint particleIndex = curPair.y;     // the CLOTH vertex
        const uint tetrahedronIdx = curPair.z;

        PxgShape sbShape, clothShape;
        uint sbCacheRef, clothCacheRef;
        loadSoftbodyClothPair(cmIdx, sbShape, sbCacheRef, clothShape, clothCacheRef);

        const float cDistance = contactDistance[sbCacheRef] + contactDistance[clothCacheRef];

        const uint softbodyId = sbShape.particleOrSoftbodyId;
        const uint clothId = clothShape.particleOrSoftbodyId;

        // A three-way OR (5.12, see the header).  The array goes through dfFilterArray with
        // the softbody descriptor as the safe address (llvmpipe speculates the first load
        // of a zero-trip search, see pxgSlangDefFilter).
        const DfFilterArray filters =
            softBodyVolumeSurfaceFilters(gVc.softbodies + uint64_t(softbodyId) * PXG_SOFTBODY_STRIDE);
        if (findVolumeSurfaceVertexFilter(filters, softbodyId, PXG_DREF_ANY_FEATURE, clothId, particleIndex) ||
            findVolumeSurfaceVertexFilter(filters, softbodyId, tetrahedronIdx, clothId, PXG_DREF_ANY_FEATURE) ||
            findVolumeSurfaceVertexFilter(filters, softbodyId, tetrahedronIdx, clothId, particleIndex))
            continue;

        const uint surfaceMask = byteAt(softbodies[softbodyId].mTetMeshSurfaceHint, tetrahedronIdx);
        if (surfaceMask == 0u)
            continue;

        const uint4 tetIdx = Ptr<uint4>(softbodies[softbodyId].mTetIndices)[tetrahedronIdx];
        Ptr<float4> tetVerts = Ptr<float4>(softbodies[softbodyId].mPosition_InvMass);
        const float3 a = tetVerts[tetIdx.x].xyz;
        const float3 b = tetVerts[tetIdx.y].xyz;
        const float3 c = tetVerts[tetIdx.z].xyz;
        const float3 d = tetVerts[tetIdx.w].xyz;

        const float3 p = Ptr<float4>(clothes[clothId].mPosition_InvMass)[particleIndex].xyz;

        const bool inside = isPointOutsideOfPlane4(p, a, b, c, d);

        float3 n2 = float3(0, 0, 0), closestP = float3(0, 0, 0);
        float sqDist = PX_MAX_F32;
        closestPtTetrahedron(p, surfaceMask, a, b, c, d, n2, closestP, sqDist);

        // PxVec3::normalize returns the LENGTH and leaves the vector unit.
        float3 v = closestP - p;
        // P9.36: PxVec3::normalize scales through operator/= (PxVec3.h:260) -- ONE
        // reciprocal, three multiplies.
        const float m = sqrt(dot(v, v));   // P9.47: magnitude(), not Length
        if (m > 0.0f)
            v = v * (1.0f / m);

        float3 n;
        float pen;
        if (inside)
        {
            // No contact-distance test: a vertex already inside is a contact at any depth.
            n = n2;
            pen = -m;
        }
        else
        {
            if (sqDist > cDistance * cDistance)
                continue;
            pen = m;
            n = (m > 1e-10f && dot(v, n2) > 0.707f) ? v : n2;
        }

        uint index;
        InterlockedAdd(Ptr<uint>(gVc.w_totalContactCount)[0], 1u, index);
        if (index >= gVc.w_maxContacts)
            continue;

        // writeSurfaceVolumeContact, the vertex overload: ref0 = the CLOTH vertex, ref1 = the
        // SOFTBODY tetrahedron; no barycentric.
        vcWriter().writeSurfaceVolumeContactVertex(index,
                                                   float4(closestP.x, closestP.y, closestP.z, 0.0f),
                                                   float4(n.x, n.y, n.z, pen),
                                                   PxgEncodeDeformableVertex(clothId, particleIndex),
                                                   PxgEncodeDeformableTetrahedron(softbodyId, tetrahedronIdx));
    }
}
