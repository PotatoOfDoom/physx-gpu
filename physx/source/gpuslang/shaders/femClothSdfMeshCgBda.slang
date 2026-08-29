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
// femClothSdfMeshCgBda — cloth_sdfMeshContactGenLaunch (femClothPrimitives.cu:2310),
// on the engine BDA ABI.  Launch PxgNarrowphaseCore.cpp:6507, grid numTests, block 1024.
//
// The last unmapped cloth contact gen, deferred as "large" since P9.27.  It is
// COMPOSITION rather than transcription: the two halves that carry the work are
// already ported and swept --
//
//   pxgSlangSdfFind        P9.34b  findInterestingTrianglesA, the resumable gather
//   pxgSlangSdfCollision   P9.34a  doTriangleSDFCollision, the descent
//
// -- and Gu::closestPtPointBaryTriangle came with them.  What is new here is the
// glue, and the glue is where the traps are.
//
// ---------------------------------------------------------------------------
// THE TRAPS
// ---------------------------------------------------------------------------
//  S1. ONE BLOCK PER TEST, and the launch site says otherwise.  The kernel takes
//      `pairIdx = blockIdx.x` with 1024 threads, while the comment above the launch
//      reads "each warp deal with one test" (PxgNarrowphaseCore.cpp:6526).  It is
//      wrong: numBlocks is numTests and the whole block cooperates on one pair,
//      which is what the shared mesh, the shared SDF texture and the block-wide
//      gather all require.  Upstream candidate 42, documented and not reported.
//
//  S2. `rest` IS INDEXED BY THE PAIR, not by a cache slot (:2402):
//          contactDistance = contactDistances[transformCacheRef0]
//                          + contactDistances[transformCacheRef1]
//          rest            = restDistances[pairIdx]
//      restDistances is a per-CONTACT-MANAGER array; contactDistances is per
//      transform-cache slot.  Reading either with the other's index is a live
//      defect, and the fixture gives every slot of both a distinct value so it
//      shows.
//      And the honest half of the same observation: the first line uses the RAW
//      cache refs rather than the cloth/trimesh-sorted ones the lines just above
//      computed, and that difference is NOT observable.  The sorted pair is the
//      raw pair swapped, and float addition is commutative, so the two spellings
//      produce the same bits.  Transcribed as written anyway; a mutant aimed at
//      it would be a provable equivalent, so none is written.
//
//  S3. cullScale DIVIDES BY THE SMALLEST SCALE COMPONENT, and unlike the particle
//      sibling (psSdfMidphaseBda's W3) it does NOT take absolute values.  A
//      negative scale component therefore produces a negative cullScale here and a
//      positive one there, from the same expression shape.
//
//  S4. THE CLOTH SIDE IS UNSCALED.  `PxMeshScale unitScale = PxMeshScale()` is
//      passed for the cloth in both the gather and the vertex fetch, so the cloth's
//      own shape scale is never applied -- its vertices are already in the right
//      frame.  The TRIMESH's scale is applied to everything.
//
//  S5. THE FILTER NEEDS ALL THREE VERTICES.  `skipTriangle` is a conjunction of
//      three finds, so a triangle survives if even one of its vertices is not
//      filtered.  An `||` would drop triangles the engine keeps.
//
//  S6. `sep` IS RESCALED BEFORE THE SECOND GATE.  The order is: gate on
//      `sep < cullScale`, then transform `dir` out of vertex space, then normalise
//      it AND SCALE `sep` BY THE SAME 1/|dir|, and only then test
//      `sep < contactDistance`.  Testing the unscaled sep against contactDistance
//      is a different kernel.
//
//  S7. `sep = sep;` (:2516) is a real self-assignment upstream.  It is not
//      transcribed, because there is nothing to transcribe; it is named here so the
//      next reader does not go looking for what it meant.
//
//  S8. THE SCAN IS BLOCK-WIDE AND EVERY THREAD CALLS IT.  globalScanExclusive<32>
//      over all 1024 threads, one atomicAdd by thread 0 -- so the call sits outside
//      the `candidateContact` test, and threads past the gathered count take part
//      with `false`.  Skipping it for them strands the block at a barrier.
//
//  S9. THE CONTACT IS WRITTEN IN WORLD SPACE, but the point comes back from the
//      descent in the trimesh's VERTEX space: it is pushed through vertex2Shape
//      first, then the trimesh-to-world transform.  The normal takes the rotation
//      only.
// ===========================================================================

#define PXG_WARP_BLOCK 1024
#include "pxgSlangWarpInc.slang"

import pxgSlangCommon;
import pxgSlangFEMClothDesc;
import pxgSlangSdfTexture;
import pxgSlangSdfCollision;
import pxgSlangClothTriContact;   // computeBarycentric
import pxgSlangConvexTriCloth;    // FcCtWriter + the contact write

#include "pxgSlangSdfFind.slang"

// cloth_sdfMeshContactGenLaunch(PxU32 numWorkItems, PxgContactManagerInput*,
//   PxsCachedTransform*, PxReal* contactDistances, PxReal* restDistances, PxgShape*,
//   PxgFEMCloth*, PxNodeIndex*, PxgRigidFilterPair*, PxU32 nbFilterPairs,
//   PxgFEMContactWriter by value) -> ABI row {4,8,8,8,8,8,8,8,8,4,80}, 160-byte blob.
struct ClothSdfCgParams
{
    uint numWorkItems;             // @0
    uint _pad4;                    // @4
    uint64_t cmInputs;             // @8
    uint64_t transformCache;       // @16
    uint64_t contactDistances;     // @24
    uint64_t restDistances;        // @32
    uint64_t gpuShapes;            // @40
    uint64_t femClothes;           // @48
    uint64_t shapeToRigidRemap;    // @56
    uint64_t filterPairs;          // @64
    uint nbFilterPairs;            // @72
    uint _pad76;                   // @76
    // ---- PxgFEMContactWriter (by value, 80B) @80 ----
    uint64_t w_outPoint;             // @80
    uint64_t w_outNormalPen;         // @88
    uint64_t w_outBarycentric;       // @96
    uint64_t w_outContactInfo;       // @104
    uint64_t w_totalContactCount;    // @112
    uint64_t w_contactByX;           // @120
    uint64_t w_tempContactByX;       // @128
    uint64_t w_contactIndexSortedByX; // @136
    uint64_t w_contactSortedByX;     // @144
    uint w_maxNumContacts;           // @152
    uint _pad156;                    // @156 -> 160
}

[[vk::push_constant]] ConstantBuffer<ClothSdfCgParams> gCs;

// PXG_BITSHIFT_ELEMENT_ID comes from pxgSlangConvexTriCloth, which already exports it.
static const uint GEOM_TRIANGLEMESH = 8u;          // PxGeometryType::eTRIANGLEMESH

FcCtWriter csWriter()
{
    FcCtWriter w;
    w.point = gCs.w_outPoint;          w.normalPen = gCs.w_outNormalPen;
    w.bary = gCs.w_outBarycentric;     w.info = gCs.w_outContactInfo;
    w.count = gCs.w_totalContactCount; w.byX = gCs.w_contactByX;
    w.tempByX = gCs.w_tempContactByX;  w.idxByX = gCs.w_contactIndexSortedByX;
    w.maxContacts = gCs.w_maxNumContacts;
    return w;
}

// PxgShape comes from pxgSlangCommon rather than being redeclared -- the other cloth
// contact gens predate that module and carry their own copies.

// PxgRigidFilterPair (PxgSimulationCoreDesc.h, 32 B).
struct PairR { uint2 index0; uint2 index1; uint index2; uint pad0, pad1, pad2; }

// `find` (PxgSimulationCoreDesc.h) -- a BINARY search, not a linear one, with the
// ignoreIndex2 early accept.  A cloth vertex's compressed id folds into
// q1 = uint2(clothMask, 0).
//
// THE NINTH COPY, and that is a scope decision rather than a defence: this corpus
// already carries eight spellings of `find` (femClothContactGenBda,
// femClothMeshContactGenBda, npDefFilter, npSbPrim, psContactGenBda, psPrimCollide,
// softBodyContactGenBda, softBodyMeshContactGenBda), which is exactly the drift the
// P9.32 module rule exists to stop.  Consolidating eight live call sites is its own
// change with its own sweep; doing it inside this tranche would put an unrelated
// refactor under a kernel port's verification.  Filed separately.
bool u64Lt(uint2 a, uint2 b) { return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x); }
bool u64Eq(uint2 a, uint2 b) { return a.x == b.x && a.y == b.y; }

bool csFindRigid(uint nbPairs, uint2 q0, uint2 q1)
{
    Ptr<PairR> pairs = Ptr<PairR>(gCs.filterPairs);
    uint l = 0u, r = nbPairs;
    while (l < r)
    {
        const uint mid = (l + r) / 2u;
        const PairR p = pairs[mid];
        int cmp = 0;
        if (u64Lt(q0, p.index0)) cmp = -1;
        else if (u64Lt(p.index0, q0)) cmp = 1;
        else if (u64Lt(q1, p.index1)) cmp = -1;
        else if (u64Lt(p.index1, q1)) cmp = 1;
        else if (0u < p.index2) cmp = -1;
        if (cmp == 0 || (u64Eq(q0, p.index0) && u64Eq(q1, p.index1)))
            return true;
        if (cmp < 0) r = mid; else l = mid + 1u;
    }
    return false;
}

// S8: globalScanExclusive<32> against the writer's counter -- P9.34c moved it into
// pxgSlangSdfFind so this and psSdfMidphaseBda share one copy.
uint csGlobalScan(uint tid, bool contributes)
{
    return sdfGlobalScanExclusive(tid, contributes, gCs.w_totalContactCount);
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void cloth_sdfMeshContactGenLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint pairIdx = gid.x;              // S1: ONE BLOCK per test

    if (pairIdx >= gCs.numWorkItems)
        return;

    // PxgContactManagerInput (16 B): shapeRef0, shapeRef1, cacheRef0, cacheRef1.
    const uint4 cm = Ptr<uint4>(gCs.cmInputs)[pairIdx];

    Ptr<PxgShape> shapes = Ptr<PxgShape>(gCs.gpuShapes);
    const PxgShape shape0 = shapes[cm.x];
    const PxgShape shape1 = shapes[cm.y];

    // The cloth is whichever side carries a deformable id.
    const bool clothIsZero = (shape0.particleOrSoftbodyId != 0xffffffffu);
    const PxgShape clothShape = clothIsZero ? shape0 : shape1;
    const PxgShape trimeshShape = clothIsZero ? shape1 : shape0;
    const uint clothCacheRef = clothIsZero ? cm.z : cm.w;
    const uint trimeshCacheRef = clothIsZero ? cm.w : cm.z;

    // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.
    const uint64_t xfT = gCs.transformCache + uint64_t(trimeshCacheRef) * 32UL;
    const float4 meshQ = Ptr<float4>(xfT)[0];
    const float3 meshP = float3(Ptr<float>(xfT + 16UL)[0], Ptr<float>(xfT + 20UL)[0],
                                Ptr<float>(xfT + 24UL)[0]);
    const uint64_t xfC = gCs.transformCache + uint64_t(clothCacheRef) * 32UL;
    const float4 clothQ = Ptr<float4>(xfC)[0];
    const float3 clothP = float3(Ptr<float>(xfC + 16UL)[0], Ptr<float>(xfC + 20UL)[0],
                                 Ptr<float>(xfC + 24UL)[0]);

    const uint64_t meshPtr = uint64_t(trimeshShape.hullOrMeshOffset.x) |
                             (uint64_t(trimeshShape.hullOrMeshOffset.y) << 32);
    PxgSdfMesh mesh;
    if (!sdfReadMesh(meshPtr, mesh))
        return;                               // block-uniform
    const SparseSdfTexture tex = sdfTexInitialize(mesh);

    // S2: the RAW cache refs, and the PAIR index for `rest`.
    const float contactDistance = Ptr<float>(gCs.contactDistances)[cm.z] +
                                  Ptr<float>(gCs.contactDistances)[cm.w];
    const float rest = Ptr<float>(gCs.restDistances)[pairIdx];

    const float3 mScale = trimeshShape.scale();
    const float4 mScaleRot = trimeshShape.scaleRotation();

    // S3: the SMALLEST component, and NOT its absolute value.
    float cullScale = contactDistance;
    cullScale = cullScale / min(mScale.x, min(mScale.y, mScale.z));

    // PxTransform::transformInv(other) = getInverse() * other, i.e. the cloth's
    // transform expressed in the trimesh's frame.
    const float4 invQ = float4(-meshQ.x, -meshQ.y, -meshQ.z, meshQ.w);
    const float4 aToBq = quatMul(invQ, clothQ);
    const float3 aToBp = quatRotate(invQ, clothP - meshP);

    // S4: the cloth side is UNSCALED.
    const float3 unitScale = float3(1.0f, 1.0f, 1.0f);
    const float4 unitRot = float4(0.0f, 0.0f, 0.0f, 1.0f);

    const uint clothId = clothShape.particleOrSoftbodyId;
    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gCs.femClothes);
    const uint64_t triIdxAddr = clothes[clothId].mTriangleVertexIndices;
    const uint64_t posAddr = clothes[clothId].mPosition_InvMass;
    const uint nbClothTriangles = clothes[clothId].mNbTriangles;

    const uint2 rigidId = Ptr<uint2>(gCs.shapeToRigidRemap + uint64_t(trimeshCacheRef) * 8UL)[0];

    uint startIndex = 0u;
    while (startIndex < nbClothTriangles)
    {
        const uint nbFound = sdfFindInterestingTriangles(tid, nbClothTriangles, triIdxAddr,
                                                         posAddr, unitScale, unitRot,
                                                         mScale, mScaleRot, cullScale,
                                                         tex, aToBp, aToBq, startIndex);

        const uint ind = (tid < nbFound) ? gSdfIndices2[tid].x : nbClothTriangles;

        bool candidateContact = false;
        float3 normal = float3(0.0f, 0.0f, 0.0f);
        float3 pos = float3(0.0f, 0.0f, 0.0f);
        float sep = 0.0f;
        float4 tBarycentric = float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (ind < nbClothTriangles)
        {
            const uint4 triangle = Ptr<uint4>(triIdxAddr)[ind];

            // S5: ALL THREE must be filtered for the triangle to be skipped.
            const uint m0 = (clothId << PXG_BITSHIFT_ELEMENT_ID) | triangle.x;
            const uint m1 = (clothId << PXG_BITSHIFT_ELEMENT_ID) | triangle.y;
            const uint m2 = (clothId << PXG_BITSHIFT_ELEMENT_ID) | triangle.z;
            const bool skipTriangle =
                csFindRigid(gCs.nbFilterPairs, rigidId, uint2(m0, 0u)) &&
                csFindRigid(gCs.nbFilterPairs, rigidId, uint2(m1, 0u)) &&
                csFindRigid(gCs.nbFilterPairs, rigidId, uint2(m2, 0u));

            if (!skipTriangle)
            {
                Ptr<float4> vp = Ptr<float4>(posAddr);
                // getTriangleVertices with the cloth's UNIT scale (S4), then into the
                // trimesh's VERTEX space, which is where the descent works.
                float3 v0 = sdfTransformPoint(aToBp, aToBq,
                                              sdfVertex2Shape(vp[triangle.x].xyz, unitScale, unitRot));
                float3 v1 = sdfTransformPoint(aToBp, aToBq,
                                              sdfVertex2Shape(vp[triangle.y].xyz, unitScale, unitRot));
                float3 v2 = sdfTransformPoint(aToBp, aToBq,
                                              sdfVertex2Shape(vp[triangle.z].xyz, unitScale, unitRot));
                v0 = sdfShape2Vertex(v0, mScale, mScaleRot);
                v1 = sdfShape2Vertex(v1, mScale, mScaleRot);
                v2 = sdfShape2Vertex(v2, mScale, mScaleRot);

                float3 dir = float3(0.0f, 0.0f, 0.0f);
                sep = sdfDoTriangleCollision(tex, v0, v1, v2, pos, dir, cullScale);

                computeBarycentric(v0, v1, v2, pos, tBarycentric);

                // S9: back to shape space here; to world only at the write.
                pos = sdfVertex2Shape(pos, mScale, mScaleRot);

                if (sep < cullScale)
                {
                    // vertex2ShapeNormalVector IS shape2Vertex -- a covariant vector
                    // takes the inverse scale, and the engine defines it as that call.
                    dir = sdfShape2Vertex(dir, mScale, mScaleRot);

                    float m = dot(dir, dir);
                    if (m > 0.0f)
                    {
                        m = 1.0f / sqrt(m);
                        sep = sep * m;         // S6: sep is rescaled BEFORE the gate
                        dir = dir * m;
                    }

                    candidateContact = sep < contactDistance;
                    if (candidateContact)
                        normal = dir;          // S7: upstream's `sep = sep;` has nothing to port
                }
            }
        }

        // S8: EVERY thread, including those past nbFound.
        const uint contactIndex = csGlobalScan(tid, candidateContact);

        if (candidateContact)
        {
            const float3 worldNormal = quatRotate(meshQ, normal);
            const float3 worldPos = quatRotate(meshQ, pos) + meshP;
            writeRigidVsDeformableContact(csWriter(), contactIndex,
                                          float4(worldPos, rest),
                                          float4(worldNormal, sep), tBarycentric,
                                          rigidId, (clothId << PXG_BITSHIFT_ELEMENT_ID) | ind,
                                          trimeshShape.materialIndex, rigidId);
        }
    }
}
