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

// fcStaticTriCloth.slang
//
// Fixture harness for the static-triangle vs cloth-triangle contact path of
// femClothPrimitives.cu (PhysX 5.12) -- the shared core of
// cloth_meshContactGenLaunch and cloth_heightfieldContactGenLaunch -- plus the
// per-vertex box path of cloth_boxVertexContactGenLaunch.  One case per lane,
// driven by testFcStaticTriCloth.cpp's random battery against a CPU
// transcription.
//
// WHAT RUNS HERE IS THE ENGINE'S CODE.  The clipping core is
// pxgSlangClothTriContact (staticTriangleVsClothCollision, the module
// femClothMeshContactGenBda and femClothHfContactGenBda import), the box query
// is pxgSlangSphereBox, the filter is pxgSlangDefFilter and the writer is
// pxgSlangDeformableWriters' PxgRigidDbContactWriter.  This file only carries
// the per-case glue the engine kernels wrap around them, so the battery's
// hundreds of trim / averaging-band / degenerate cases land on the modules the
// engine ships, not on a copy.  (Until the 5.12 port this shader carried its
// own transcription of the core plus the 5.6.1 PxgRigidFilterPair filter and
// PxgFEMContactWriter.)
//
// MODE 0 (cloth_meshContactGenLaunch per-pair core):
// - the 5.12 filter: ONE exact search of the cloth's mRigidSurfaceFilterPairs
//   for (PxgEncodeRigidFilterRef(rigidId, rigidCacheRef),
//   PxgEncodeDeformableTriangle(clothId, triangleIdx)) --
//   findRigidSurfaceTriangleFilter (5.6.1 needed all three cloth VERTICES
//   filtered);
// - the static mesh triangle brought into cloth space:
//   vertex2Shape(triV, trimeshScale) then trimeshToClothTransform =
//   clothTransform.transformInv(trimeshToWorld) applied to each vertex;
// - staticTriangleVsClothCollision (the convexPoly seeded with the cloth
//   triangle, the static triangle's plane, trimTriangleByTriangle's 5-plane
//   clip with the in-place insert branch, getDeepestPointSimplified's
//   +-0.01*cDistance averaging band, hasContact = sep < cDistance STRICT);
//   see pxgSlangClothTriContact for the per-step notes;
// - the write: globalScanExclusiveSingleWarp -- one ballot over the warp,
//   exclusive-popc lane offsets, a SINGLE lane-0 atomicAdd reservation
//   (skipped when the warp has no contacts) -- then
//   PxgRigidDbContactWriter::writeSurfaceContact: point (w = restDistance),
//   normalPen (w = separation), barycentric, pair info (rigidId.getInd(),
//   PxgEncodeDeformableTriangle(clothId, triangleIdx)), the PxU16 rigid
//   material, and the rigid sort key (contactByRigid = the node index,
//   tempContactByRigid = its low word, contactIndexSortedByRigid = index).
//
// MODE 1 (cloth_boxVertexContactGenLaunch per-vertex core):
// - the 5.12 filter: findRigidSurfaceVertexFilter, (rigid filter ref,
//   PxgEncodeDeformableVertex(clothId, vertexIdx));
// - spherebox with sphereRadius = 0 (pxgSlangSphereBox);
// - contact point = clothP - pen * normal, w = restDistance; the same ballot +
//   single-reservation slot packing, then the VERTEX overload of
//   writeSurfaceContact: no barycentric is written (5.6.1 wrote the
//   (0, 0, 0, 1) vertex marker), pair info (rigid, vertex ref), material,
//   sort key.
//
// KNOWN LIMITS (verified textually, not pinnable by tests): the
// +cDistance slab plane in trimTriangleByTriangle is OUTPUT-NEUTRAL
// redundancy -- any surviving deepest point already satisfies
// sep <= cDistance, and everything the slab would remove is rejected by
// the sep < cDistance gate (or never enters the +-0.01*cDistance
// averaging band of a deeper point); only cases with sep inside that
// band of cDistance could differ, which is exactly the tainted gate
// boundary.  The -cDistance slab IS live (it clamps deep penetration)
// and is pinned by the deep-penetration fixtures.  The normalizeSafe
// (trimTriangleByTriangle) vs getNormalized (fixNormal) distinction is
// pinnable only for 0 < |cross| < 1e-20 -- a subnormal window that cannot
// survive the O(1) vertex transform chain -- so it is verified textually;
// the exact-zero degenerate path (zero normal, zero planes, centroid
// contact) IS pinned by fixtures.
//
// WARP SERIALIZATION: both launches are data-parallel one-CASE-per-thread
// kernels whose only cross-lane effect is the slot-packing ballot; a
// batch of 32 consecutive cases is serialized into one thread (lane
// order = ascending case index within the batch, one reservation per
// batch), which is output-identical per the established doctrine.
//
// Engine glue not reproduced (documented): LoadShapePair/transform caches,
// the shape0/shape1 cloth-identification branch (cases carry the resolved
// roles), PxgFEMCloth loads (the filter list is passed directly), the trimesh
// blob walk, the pair-stack iteration and the stackSizeNeededOnDevice
// atomicMax -- femClothSimMeshContactGen and femClothSimBoxVertContactGen
// drive those on the engine kernels.  The heightfield flavour
// cloth_heightfieldContactGenLaunch differs from mode 0 only upstream of
// this core (getTriangle, no filter, no vertex2Shape).

import pxgSlangCommon;
import pxgSlangSphereBox;          // sphereboxNormalPen (the engine's box query)
import pxgSlangClothTriContact;    // staticTriangleVsClothCollision (the engine's clipping core)
import pxgSlangDeformableRef;      // PxgEncodeDeformableTriangle / Vertex
import pxgSlangDefFilter;          // findRigidSurfaceTriangleFilter / VertexFilter
import pxgSlangDeformableWriters;  // PxgRigidDbContactWriterGpu

// The filter list and the writer's streams are device addresses (BDA), the way
// the engine kernels read them from the PxgFEMCloth and the writer.
struct FstParams
{
    uint numCases; uint maxContacts; uint nbFilterPairs; uint mode;
    uint64_t filterPairs;       // PxgDeformablePairInfo*, sorted
    uint64_t filterSafeAddr;    // any mapped address, read in place of a NULL list
    // PxgRigidDbContactWriter
    uint64_t outPoint, outNormalPen, outBarycentric, outContactInfo, outRigidMaterialIndex;
    uint64_t totalContactCount, contactByRigid, tempContactByRigid, contactIndexSortedByRigid;
}

struct FstCase
{
    float4 triA;     // mode 0: static tri vert A (trimesh VERTEX space); mode 1: cloth vertex P (world)
    float4 triB;     // mode 0: static tri vert B; mode 1: box half extents (xyz)
    float4 triC;     // mode 0: static tri vert C
    float4 clothQ;   // mode 0: cloth transform quat;   mode 1: rigid (box) transform quat
    float4 clothP;   // mode 0: cloth transform p, w = contactDist; mode 1: rigid p, w = contactDist
    float4 meshQ;    // mode 0: trimesh transform quat
    float4 meshP;    // mode 0: trimesh transform p, w = restDist; mode 1: w = restDist
    float4 scale;    // mode 0: trimesh mesh scale (xyz)
    float4 scaleRot; // mode 0: trimesh scale rotation quat
    uint4 ids;       // x = rigidId lo, y = rigidId hi, z = clothId, w = clothTriangleIdx (mode 0) / vertexIdx (mode 1)
    uint4 ids2;      // x = rigidMaterialIndex, y/z/w = cloth triangle vertex indices (mode 0)
    uint4 ids3;      // x = the rigid's transform cache ref (keys a STATIC rigid's filter ref)
}

[[vk::push_constant]]
ConstantBuffer<FstParams> gParams;

RWStructuredBuffer<FstCase> gCase;        // binding 0
RWStructuredBuffer<float4> gClothPos;     // binding 1: cloth mPosition_InvMass
RWStructuredBuffer<float4> gDebug;        // binding 2: (skipped, hasContact, sep|pen, polyCorners)

PxgRigidDbContactWriterGpu fstWriter()
{
    PxgRigidDbContactWriterGpu w;
    w.outPoint = gParams.outPoint;
    w.outNormalPen = gParams.outNormalPen;
    w.outBarycentric = gParams.outBarycentric;
    w.outContactInfo = gParams.outContactInfo;
    w.outRigidMaterialIndex = gParams.outRigidMaterialIndex;
    w.totalContactCount = gParams.totalContactCount;
    w.contactByRigid = gParams.contactByRigid;
    w.tempContactByRigid = gParams.tempContactByRigid;
    w.contactIndexSortedByRigid = gParams.contactIndexSortedByRigid;
    w.maxNumContacts = gParams.maxContacts;
    return w;
}

// The clip's corner count, for the debug lane only: the same engine
// trimTriangleByTriangle call staticTriangleVsClothCollision makes, repeated
// because the module does not export its intermediate.
int debugPolyCorners(float3 localA0, float3 localB0, float3 localC0, float3 a1, float3 b1, float3 c1,
                     float cDistance)
{
    float3 convexPoly[8];
    convexPoly[0] = a1;
    convexPoly[1] = b1;
    convexPoly[2] = c1;
    for (int k = 3; k < 8; ++k)
        convexPoly[k] = float3(0, 0, 0);
    return trimTriangleByTriangle(convexPoly, 3, localA0, localB0, localC0, cDistance, cDistance);
}

// ---------------------------------------------------------------------------
// One thread serializes one 32-case warp batch (lane order = case order);
// the batch does ONE reservation for all its contacts, like the CUDA
// warp's globalScanExclusiveSingleWarp / lane-0 atomicAdd.
[shader("compute")]
[numthreads(64, 1, 1)]
void fcStaticTriClothKernel(uint3 dtid : SV_DispatchThreadID)
{
    const uint batch = dtid.x;
    const uint base = batch * 32;
    if (base >= gParams.numCases) return;

    bool laneHas[32];
    float4 lanePoint[32];
    float4 laneNormalPen[32];
    float4 laneBary[32];
    uint64_t laneRigid[32];
    uint64_t laneRef1[32];
    uint laneMat[32];

    const DfFilterArray filters = dfFilterArray(gParams.filterPairs, gParams.nbFilterPairs,
                                                gParams.filterSafeAddr);

    uint totalCount = 0;
    for (uint lane = 0; lane < 32; ++lane)
    {
        const uint i = base + lane;
        laneHas[lane] = false;
        lanePoint[lane] = float4(0, 0, 0, 0);
        laneNormalPen[lane] = float4(0, 0, 0, 0);
        laneBary[lane] = float4(0, 0, 0, 0);
        laneRigid[lane] = 0;
        laneRef1[lane] = 0;
        laneMat[lane] = 0;
        if (i >= gParams.numCases)
            continue;

        const FstCase cs = gCase[i];
        const uint64_t rigidId = uint64_t(cs.ids.x) | (uint64_t(cs.ids.y) << 32);
        const uint rigidCacheRef = cs.ids3.x;
        const uint clothId = cs.ids.z;
        const float cDistance = cs.clothP.w;
        const float restDist = cs.meshP.w;

        if (gParams.mode == 0)
        {
            const uint clothTriangleIdx = cs.ids.w;
            const uint3 vi = cs.ids2.yzw;

            // A filter pair for this rigid and triangle suppresses contact generation.
            if (findRigidSurfaceTriangleFilter(filters, rigidId, rigidCacheRef, clothId, clothTriangleIdx))
            {
                gDebug[i] = float4(1, 0, 0, 0);
                continue;
            }

            // trimeshToClothTransform = clothT.transformInv(trimeshT)
            const PxgTransform28 clothT = makeTransform28(cs.clothQ, cs.clothP.xyz);
            const PxgTransform28 meshT = makeTransform28(cs.meshQ, cs.meshP.xyz);
            const PxgTransform28 meshToCloth = transformInvTransform(clothT, meshT);

            const float3 localA0 = transformPt(meshToCloth.q(), meshToCloth.p(),
                                               vertex2Shape(cs.triA.xyz, cs.scale.xyz, cs.scaleRot));
            const float3 localB0 = transformPt(meshToCloth.q(), meshToCloth.p(),
                                               vertex2Shape(cs.triB.xyz, cs.scale.xyz, cs.scaleRot));
            const float3 localC0 = transformPt(meshToCloth.q(), meshToCloth.p(),
                                               vertex2Shape(cs.triC.xyz, cs.scale.xyz, cs.scaleRot));

            const float3 a1 = gClothPos[vi.x].xyz;
            const float3 b1 = gClothPos[vi.y].xyz;
            const float3 c1 = gClothPos[vi.z].xyz;

            FemContact c;
            const bool hasContact = staticTriangleVsClothCollision(localA0, localB0, localC0, a1, b1, c1,
                                                                   cs.clothQ, cs.clothP.xyz, cDistance, c);
            const int dbgCorners = debugPolyCorners(localA0, localB0, localC0, a1, b1, c1, cDistance);
            gDebug[i] = float4(0, hasContact ? 1 : 0, c.separation, float(dbgCorners));
            if (hasContact)
            {
                laneHas[lane] = true;
                lanePoint[lane] = float4(c.position, restDist);
                laneNormalPen[lane] = float4(c.normal, c.separation);
                laneBary[lane] = c.barycentric;
                laneRigid[lane] = rigidId;
                laneRef1[lane] = PxgEncodeDeformableTriangle(clothId, clothTriangleIdx);
                laneMat[lane] = cs.ids2.x;
                ++totalCount;
            }
        }
        else
        {
            const uint vertexIdx = cs.ids.w;
            if (findRigidSurfaceVertexFilter(filters, rigidId, rigidCacheRef, clothId, vertexIdx))
            {
                gDebug[i] = float4(1, 0, 0, 0);
                continue;
            }

            float4 normalPen;
            const uint numContacts = sphereboxNormalPen(cs.triA.xyz, cs.clothQ, cs.clothP.xyz, 0.0f,
                                                        cs.triB.xyz, cDistance, normalPen);
            gDebug[i] = float4(0, float(numContacts), normalPen.w, 0);
            if (numContacts != 0)
            {
                const float3 rigidP = cs.triA.xyz - normalPen.w * normalPen.xyz;
                laneHas[lane] = true;
                lanePoint[lane] = float4(rigidP, restDist);
                laneNormalPen[lane] = normalPen;
                laneRigid[lane] = rigidId;
                laneRef1[lane] = PxgEncodeDeformableVertex(clothId, vertexIdx);
                laneMat[lane] = cs.ids2.x;
                ++totalCount;
            }
        }
    }

    // one reservation per batch, skipped when the ballot is empty
    uint startIndex = 0;
    if (totalCount > 0)
        InterlockedAdd(Ptr<uint>(gParams.totalContactCount)[0], totalCount, startIndex);

    PxgRigidDbContactWriterGpu writer = fstWriter();
    uint offset = 0;
    for (uint lane = 0; lane < 32; ++lane)
    {
        if (!laneHas[lane])
            continue;
        const uint index = startIndex + offset;
        ++offset;
        if (gParams.mode == 0)
            writer.writeSurfaceContact(index, lanePoint[lane], laneNormalPen[lane], laneRigid[lane],
                                       laneMat[lane], laneRef1[lane], laneBary[lane]);
        else
            writer.writeSurfaceContactVertex(index, lanePoint[lane], laneNormalPen[lane], laneRigid[lane],
                                             laneMat[lane], laneRef1[lane]);
    }
}
