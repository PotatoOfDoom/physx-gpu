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
// softBodySdfMidphaseBda — sb_sdfMeshMidphaseGeneratePairsLaunch
// (softbodySoftbodyMidPhase.cu:1121), on the engine BDA ABI.
// Launch PxgNarrowphaseCore.cpp:4886, grid numTests, block 1024.
//
// The direct sibling of P9.34c's cloth kernel, and reading the two side by side
// is the fastest way to see what each of them actually says.  Both halves it
// stands on are ported and swept on their own:
//
//   pxgSlangSdfFind        P9.35b  findInterestingTets, the resumable gather
//   pxgSlangSdfCollision   P9.35a  doTetrahedronSDFCollision, the descent
//
// ---------------------------------------------------------------------------
// THE TRAPS
// ---------------------------------------------------------------------------
//  T1. TWO PARAMETERS ARE DEAD AND THERE IS NO COUNT.  `tolerenceLength` and
//      `bounds` are never read in the body, and -- unlike the cloth sibling --
//      there is no `numWorkItems` argument at all, so there is NO `cmIdx >=
//      count` guard.  The grid is exactly numTests, which is what makes that
//      safe; a port that invents a guard needs a parameter it does not have.
//
//  T2. cullScale TAKES THE ABSOLUTE VALUE: `worldContactDist /
//      trimeshScale.scale.abs().minElement()`.  This is the OPPOSITE of the
//      cloth sibling (P9.34c, S3), which divides by the signed minimum from the
//      same expression shape.  And inside ONE call chain it also disagrees with
//      the gather it calls: findInterestingTets divides its bounding-sphere
//      radius by the SIGNED minimum (P9.35b, U3).  Two spellings, forty lines
//      apart, both live.
//
//  T3. THE SOFTBODY'S OWN TRANSFORM CACHE IS NEVER READ.  The a-to-b transform
//      is `trimeshToWorld.getInverse()` -- the BARE inverse of the mesh pose,
//      not `transformInv(softbodyPose)`.  The cloth sibling composes both poses
//      because cloth vertices live in the cloth's frame; softbody vertices are
//      already in world space, so there is nothing to compose.  A port that
//      copies the cloth line reads a cache slot the engine never touches.
//
//  T4. THE SOFTBODY SIDE HAS NO SCALE, not even a unit one.  The tet vertices go
//      through the transform and then straight into shape2Vertex by the TRIMESH's
//      scale.  (The cloth sibling passes an explicit `PxMeshScale()`, which is
//      the identity; here there is no call at all.)
//
//  T5. THE FILTER IS CONSULTED LAST, and it queries ONE index.  It runs after
//      the separation, the normal and the world-space point are all computed,
//      INSIDE `if (candidateContact)`, and it asks about the TET -- one
//      PxEncodeSoftBodyIndex query, not the cloth kernel's conjunction over
//      three vertices.  Moving it earlier changes nothing observable but moving
//      it OUT of the candidate test does: a tet that failed either gate is never
//      asked about.
//
//  T6. THE WRITER IS writeContactNoBarycentric.  No barycentric array, and NO
//      MATERIAL INDEX -- the fourth info dword is left exactly as it was, where
//      the cloth sibling writes the rigid material with bit 31 cleared.
//
//  T7. THE WORLD TRANSFORM HAPPENS BEFORE THE FILTER, inside the candidate
//      branch, so by the time the write runs `contactPos` and `contactDir` are
//      already world-space.  Transcribed in that order.
//
//  T8. THE SCAN IS BLOCK-WIDE AND EVERY THREAD CALLS IT, exactly as in the cloth
//      sibling: globalScanExclusive<32> over all 1024 threads with one atomicAdd
//      by thread 0, outside the candidate test.
// ===========================================================================

#define PXG_WARP_BLOCK 1024
#include "pxgSlangWarpInc.slang"

import pxgSlangCommon;
import pxgSlangSoftBodyDesc;
import pxgSlangSdfTexture;
import pxgSlangSdfCollision;
import pxgSlangDefFilter;      // P9.35c: the filter, shared rather than copied

#include "pxgSlangSdfFind.slang"

// sb_sdfMeshMidphaseGeneratePairsLaunch(PxReal tolerenceLength,
//   PxgContactManagerInput*, PxsCachedTransform*, PxBounds3*, PxReal* contactDistance,
//   PxReal* restDistance, PxgShape*, PxgSoftBody*, PxNodeIndex*,
//   PxgRigidFilterPair*, PxU32 nbFilterPairs, PxgFEMContactWriter by value)
// -> ABI row {4,8,8,8,8,8,8,8,8,8,4,80}, 168-byte blob.
struct SbSdfMidParams
{
    float tolerenceLength;         // @0    T1: never read
    uint _pad4;                    // @4
    uint64_t cmInputs;             // @8
    uint64_t transformCache;       // @16
    uint64_t bounds;               // @24   T1: never read
    uint64_t contactDistance;      // @32
    uint64_t restDistance;         // @40
    uint64_t gpuShapes;            // @48
    uint64_t softbodies;           // @56
    uint64_t shapeToRigidRemap;    // @64
    uint64_t filterPairs;          // @72
    uint nbFilterPairs;            // @80
    uint _pad84;                   // @84
    // ---- PxgFEMContactWriter (by value, 80B) @88 ----
    uint64_t w_outPoint;             // @88
    uint64_t w_outNormalPen;         // @96
    uint64_t w_outBarycentric;       // @104
    uint64_t w_outContactInfo;       // @112
    uint64_t w_totalContactCount;    // @120
    uint64_t w_contactByX;           // @128
    uint64_t w_tempContactByX;       // @136
    uint64_t w_contactIndexSortedByX; // @144
    uint64_t w_contactSortedByX;     // @152
    uint w_maxNumContacts;           // @160
    uint _pad164;                    // @164 -> 168
}

[[vk::push_constant]] ConstantBuffer<SbSdfMidParams> gSb;

static const uint GEOM_TETRAHEDRONMESH = 7u;     // PxGeometryType::eTETRAHEDRONMESH
static const uint PXG_BITSHIFT_TET_ID = 20u;     // PxgSimulationCoreDesc.h:40

// PxgFEMContactWriter::writeContactNoBarycentric (PxgFEMCore.h:428).
// T6: the barycentric array is NOT written and the info's fourth dword is left
// alone -- this overload sets only the pair indices.
void sbWriteContactNoBarycentric(uint index, float4 contact, float4 normalPen,
                                 uint2 pairInd0, uint pairInd1, uint2 rigidId)
{
    if (index >= gSb.w_maxNumContacts) return;

    Ptr<uint2>(gSb.w_contactByX)[index] = rigidId;
    Ptr<uint>(gSb.w_tempContactByX)[index] = rigidId.x;
    Ptr<uint>(gSb.w_contactIndexSortedByX)[index] = index;

    Ptr<float4>(gSb.w_outPoint)[index] = contact;
    Ptr<float4>(gSb.w_outNormalPen)[index] = normalPen;

    const uint64_t infoBase = gSb.w_outContactInfo + uint64_t(index) * 16UL;
    Ptr<uint>(infoBase)[0] = pairInd0.x;
    Ptr<uint>(infoBase + 4UL)[0] = pairInd0.y;
    Ptr<uint>(infoBase + 8UL)[0] = pairInd1;
    // The fourth dword is deliberately untouched.
}

// LoadShapePairWarp<eTETRAHEDRONMESH, eTRIANGLEMESH>: the softbody is whichever
// half of the pair IS the tetrahedron mesh.
void sbLoadPair(uint cmIdx, out PxgShape sbShape, out uint sbCacheRef,
                out PxgShape triShape, out uint triCacheRef)
{
    // PxgContactManagerInput (16 B): shapeRef0, shapeRef1, cacheRef0, cacheRef1.
    const uint4 cm = Ptr<uint4>(gSb.cmInputs)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gSb.gpuShapes);

    uint sRefSb = cm.x, cRefSb = cm.z;
    uint sRefTri = cm.y, cRefTri = cm.w;
    if (shapes[cm.x].type != GEOM_TETRAHEDRONMESH)
    {
        sRefSb = cm.y; cRefSb = cm.w;
        sRefTri = cm.x; cRefTri = cm.z;
    }
    sbShape = shapes[sRefSb]; sbCacheRef = cRefSb;
    triShape = shapes[sRefTri]; triCacheRef = cRefTri;
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void sb_sdfMeshMidphaseGeneratePairsLaunch(uint3 gtid : SV_GroupThreadID,
                                           uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint cmIdx = gid.x;          // T1: no count, so no guard

    PxgShape softbodyShape, trimeshShape;
    uint softbodyCacheRef, trimeshCacheRef;
    sbLoadPair(cmIdx, softbodyShape, softbodyCacheRef, trimeshShape, trimeshCacheRef);

    // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.
    const uint64_t xfT = gSb.transformCache + uint64_t(trimeshCacheRef) * 32UL;
    const float4 meshQ = Ptr<float4>(xfT)[0];
    const float3 meshP = float3(Ptr<float>(xfT + 16UL)[0], Ptr<float>(xfT + 20UL)[0],
                                Ptr<float>(xfT + 24UL)[0]);

    const uint2 rigidId = Ptr<uint2>(gSb.shapeToRigidRemap + uint64_t(trimeshCacheRef) * 8UL)[0];

    const float3 mScale = trimeshShape.scale();
    const float4 mScaleRot = trimeshShape.scaleRotation();

    const uint64_t meshPtr = uint64_t(trimeshShape.hullOrMeshOffset.x) |
                             (uint64_t(trimeshShape.hullOrMeshOffset.y) << 32);
    PxgSdfMesh mesh;
    if (!sdfReadMesh(meshPtr, mesh))
        return;                        // block-uniform
    const SparseSdfTexture tex = sdfTexInitialize(mesh);

    const float worldContactDist = Ptr<float>(gSb.contactDistance)[softbodyCacheRef] +
                                   Ptr<float>(gSb.contactDistance)[trimeshCacheRef];
    // T2: the smallest MAGNITUDE, unlike the cloth sibling and unlike the gather
    // this very function calls.
    const float cullScale = worldContactDist /
                            min(abs(mScale.x), min(abs(mScale.y), abs(mScale.z)));

    const float restDist = Ptr<float>(gSb.restDistance)[cmIdx];

    const uint sbId = softbodyShape.particleOrSoftbodyId;
    Ptr<PxgSoftBodyGpu> bodies = Ptr<PxgSoftBodyGpu>(gSb.softbodies);
    const uint nbTets = bodies[sbId].mNumTets;
    const uint64_t tetIndices = bodies[sbId].mTetIndices;
    const uint64_t tetVerts = bodies[sbId].mPosition_InvMass;

    // T3: the BARE inverse of the mesh pose.  PxTransform::getInverse() is
    // (q.getConjugate(), q.rotateInv(-p)), and rotateInv by q is rotate by the
    // conjugate, so this is the conjugate applied to the negated translation.
    const float4 invQ = float4(-meshQ.x, -meshQ.y, -meshQ.z, meshQ.w);
    const float4 aToBq = invQ;
    const float3 aToBp = quatRotate(invQ, -meshP);

    uint startIndex = 0u;
    while (startIndex < nbTets)
    {
        const uint nbFound = sdfFindInterestingTets(tid, nbTets, tetIndices, tetVerts,
                                                    mScale, mScaleRot, cullScale,
                                                    tex, aToBp, aToBq, startIndex);

        const uint ind = (tid < nbFound) ? gSdfIndices[tid] : nbTets;

        bool candidateContact = false;
        float sep = 0.0f;
        float3 contactDir = float3(0.0f, 0.0f, 0.0f);
        float3 contactPos = float3(0.0f, 0.0f, 0.0f);
        uint pairId1 = 0u;

        if (ind < nbTets)
        {
            const uint4 tet = Ptr<uint4>(tetIndices)[ind];
            Ptr<float4> vp = Ptr<float4>(tetVerts);
            // T4: no scale on the softbody side -- transform, then shape2Vertex by
            // the TRIMESH's scale.
            const float3 v0 = sdfShape2Vertex(sdfTransformPoint(aToBp, aToBq, vp[tet.x].xyz),
                                              mScale, mScaleRot);
            const float3 v1 = sdfShape2Vertex(sdfTransformPoint(aToBp, aToBq, vp[tet.y].xyz),
                                              mScale, mScaleRot);
            const float3 v2 = sdfShape2Vertex(sdfTransformPoint(aToBp, aToBq, vp[tet.z].xyz),
                                              mScale, mScaleRot);
            const float3 v3 = sdfShape2Vertex(sdfTransformPoint(aToBp, aToBq, vp[tet.w].xyz),
                                              mScale, mScaleRot);

            sep = sdfDoTetrahedronCollision(tex, v0, v1, v2, v3, contactPos, contactDir,
                                            cullScale);

            if (sep < cullScale)
            {
                // vertex2ShapeNormalVector IS shape2Vertex.
                contactDir = sdfShape2Vertex(contactDir, mScale, mScaleRot);

                float m = dot(contactDir, contactDir);
                if (m > 0.0f)
                {
                    m = 1.0f / sqrt(m);
                    sep = sep * m;
                    contactDir = contactDir * m;
                }

                candidateContact = sep < worldContactDist;

                if (candidateContact)
                {
                    // T7: to world space HERE, before the filter.
                    contactPos = sdfTransformPoint(meshP, meshQ,
                                                   sdfVertex2Shape(contactPos, mScale, mScaleRot));
                    contactDir = quatRotate(meshQ, contactDir);

                    pairId1 = (sbId << PXG_BITSHIFT_TET_ID) | ind;
                    // T5: LAST, one query, and only for a surviving candidate.
                    if (dfFindRigidPair(gSb.filterPairs, gSb.nbFilterPairs, rigidId, pairId1))
                        candidateContact = false;
                }
            }
        }

        // T8: EVERY thread, including those past nbFound.
        const uint contactIndex = sdfGlobalScanExclusive(tid, candidateContact,
                                                         gSb.w_totalContactCount);

        if (candidateContact)
        {
            sbWriteContactNoBarycentric(contactIndex,
                                        float4(contactPos, restDist),
                                        float4(contactDir, sep),
                                        rigidId, pairId1, rigidId);
        }
    }
}
