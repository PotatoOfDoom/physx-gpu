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

// femClothPsContactGenBda.slang
//
// P9.30 -- cloth_psContactGenLaunch
// gpunarrowphase/src/CUDA/femClothPrimitives.cu:2866 (wrapper), :2759
// (clothParticleCollision), :114 (computeBarycentric),
// GuDistancePointTriangle.h:44 (closestPtPointTriangle2), PxgFEMCore.h:467
// (PxgFEMContactWriter::writeContact) -- all read in full.
//
// The SECOND half of the cloth-vs-PARTICLE chain.  The simplest contact
// generator in the cloth family: one particle against one triangle, at most one
// contact, and no warp aggregation at all -- every contact takes its own
// atomicAdd of 1.
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
// C1 -- THE PAIR IS (cmIdx, particleIndex, triangleIndex), the midphase's M1.
//
// C2 -- A DIFFERENT WRITER METHOD.  This calls writeContact, not
//     writeRigidVsDeformableContact: it writes pairInd0 as a **u64** and
//     pairInd1 as a u32 and leaves the material/inCollision word ALONE, where
//     the rigid path sets it.  It also never touches contactSortedByX, which is
//     NULL for the particle writer (PxgFEMCore.h:393) -- the buffer does not
//     exist, so a port that writes it stores through a null pointer.
//
// C3 -- THE SORT KEY IS THE PARTICLE.  `writeContact(..., pairInd0, pairInd1,
//     pairInd0)` passes the ENCODED PARTICLE INDEX as the id the contact is
//     sorted by, where every rigid caller passes a rigid node index.  The host
//     built this writer as PxgFEMContactWriter(clothCore, **true**), which is
//     what points the buffers at the particle streams in the first place.
//
// C4 -- THE STORED NORMAL IS NEGATED.  `v = p - closestPt` points from the
//     triangle to the particle; what is written is -n.  The penetration is
//     `m - restDistance` and NOT `m - cDistance`, and the contact point is
//     `p - n * restDistance` -- so restDistance appears twice and cDistance only
//     in the gate.
//
// C5 -- THE GATE IS `<=`, and it is against `cDistance * cDistance` with
//     cDistance the SUM of the two contact distances.  Not the rest distance,
//     and not a strict inequality.
//
// C6 -- `n = v / m` IS UNGUARDED.  A particle exactly on the triangle gives
//     m == 0 and a NaN normal, which then reaches the solver.  This is upstream
//     behaviour and is ported as written; it is upstream candidate 39.
//
// C7 -- `computeBarycentric` returns a bool that says whether the point is
//     inside the +-1e-4 window, and the caller only asserts it.  The value is
//     stored either way, so the bool is not ported.
//
// C8 -- the walk is PER THREAD over `min(*midphasePairsNum, stackSize)`, and the
//     high-water mark takes the UNCLAMPED count times 16, from one thread of the
//     grid, outside the loop.
// ===========================================================================

import pxgSlangFEMClothDesc;   // PxgFEMClothGpu (mPosition_InvMass@16, mTriangleVertexIndices@72)
import pxgSlangParticleDesc;   // PxgParticleSystemGpu (mSortedPositions_InvMass@24)
import pxgSlangTetClosest;     // closestPtPointTriangle2 (P9.30 made it public)

static const uint PSCG_GRID = 512u;
static const uint PSCG_BLOCK = 256u;
static const uint PSCG_STRIDE = PSCG_GRID * PSCG_BLOCK;   // 131072

static const uint GEOM_PARTICLESYSTEM = 6u;
static const uint PXG_CLOTH_ID_SHIFT = 20u;

// ---------------------------------------------------------------------------
// Push-constant blob -- BYTE-IDENTICAL to the CUDA pack
// (PxgNarrowphaseCore.cpp:3631): thirteen arguments, the last of them
// PxgFEMContactWriter by value (80 B) -> 176 bytes.
// ABI row {4,8,8,8,8,8,8,8,4,8,8,8,80}.
//
// The writer lands at @96 and NOT at @88: paramAlign(80) is 16, not 8
// (P9.15e), and the arguments before it end at 88.  The cloth-vs-heightfield
// contact gen's blob is ALSO 176 bytes with its writer ALSO at @96 -- and its
// middle is arranged differently, so sizeof and the writer offset both agree
// while five fields sit one slot apart.  P9.29's fixture lost a day to exactly
// that coincidence in the other direction; every offset below is asserted in
// the fixture for that reason.
// ---------------------------------------------------------------------------
struct ClothPsCgParams
{
    float toleranceLength;         // @0  (ABI parity only)
    uint _pad4;                    // @4
    uint64_t cmInputs;             // @8
    uint64_t transformCache;       // @16 (ABI parity only -- nothing is transformed)
    uint64_t contactDistance;      // @24
    uint64_t restDistance;         // @32
    uint64_t gpuShapes;            // @40
    uint64_t particleSystems;      // @48
    uint64_t femClothes;           // @56
    uint stackSizeBytes;           // @64
    uint _pad68;                   // @68
    uint64_t stackPtr;             // @72
    uint64_t midphasePairsNum;     // @80
    uint64_t stackSizeNeeded;      // @88  atomicMax output
    // ---- PxgFEMContactWriter (by value, 80B) @96 ----
    uint64_t w_outPoint;            // @96
    uint64_t w_outNormalPen;        // @104
    uint64_t w_outBarycentric;      // @112
    uint64_t w_outContactInfo;      // @120  PxgFemOtherContactInfo* (16 B stride)
    uint64_t w_totalContactCount;   // @128
    uint64_t w_contactByX;          // @136  PxU64* -- the PARTICLE sort key (C3)
    uint64_t w_tempContactByX;      // @144
    uint64_t w_contactIndexSortedByX; // @152
    uint64_t w_contactSortedByX;    // @160  NULL for particles (C2); never written
    uint w_maxNumContacts;          // @168
    uint _pad172;                   // @172 -> 176
}

[[vk::push_constant]] ConstantBuffer<ClothPsCgParams> gCg;

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

// LoadShapePair<ePARTICLESYSTEM, eTRIANGLEMESH> -- routed by TYPE, as in the
// midphase.
void loadParticleClothPair(uint cmIdx,
                           out PxgShape particleShape, out uint particleCacheRef,
                           out PxgShape clothShape, out uint clothCacheRef)
{
    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(gCg.cmInputs)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gCg.gpuShapes);
    uint sRefP = cin.shapeRef0, cRefP = cin.transformCacheRef0;
    uint sRefC = cin.shapeRef1, cRefC = cin.transformCacheRef1;
    if (shapes[cin.shapeRef0].type != GEOM_PARTICLESYSTEM)
    {
        sRefP = cin.shapeRef1; cRefP = cin.transformCacheRef1;
        sRefC = cin.shapeRef0; cRefC = cin.transformCacheRef0;
    }
    particleShape = shapes[sRefP]; particleCacheRef = cRefP;
    clothShape = shapes[sRefC]; clothCacheRef = cRefC;
}

// computeBarycentric (femClothPrimitives.cu:114).  C7: the bool is not ported.
float4 computeBarycentric(float3 a, float3 b, float3 c, float3 p)
{
    const float3 v0 = b - a;
    const float3 v1 = c - a;
    const float3 v2 = p - a;

    const float d00 = dot(v0, v0);
    const float d01 = dot(v0, v1);
    const float d11 = dot(v1, v1);
    const float d20 = dot(v2, v0);
    const float d21 = dot(v2, v1);

    const float denom = d00 * d11 - d01 * d01;
    const float v = (d11 * d20 - d01 * d21) / denom;
    const float w = (d00 * d21 - d01 * d20) / denom;
    const float u = 1.0f - v - w;

    return float4(u, v, w, 0.0f);
}

// PxgFEMContactWriter::writeContact (PxgFEMCore.h:467) -> writeContactNoBarycentric
// plus the barycentric store.  C2: no material word, no contactSortedByX.
void writeContact(uint index, float4 contact, float4 normalPen, float4 barycentric,
                  uint2 pairInd0, uint pairInd1, uint2 id)
{
    if (index >= gCg.w_maxNumContacts)
        return;

    Ptr<uint2>(gCg.w_contactByX)[index] = id;
    Ptr<uint>(gCg.w_tempContactByX)[index] = id.x;      // PxU32(id & 0xffffffff)
    Ptr<uint>(gCg.w_contactIndexSortedByX)[index] = index;

    Ptr<float4>(gCg.w_outPoint)[index] = contact;
    Ptr<float4>(gCg.w_outNormalPen)[index] = normalPen;

    // PxgFemOtherContactInfo (16 B): pairInd0 u64 @0, pairInd1 u32 @8.  The word
    // at @12 is the rigid material / inCollision pair and is NOT touched here.
    const uint64_t infoBase = gCg.w_outContactInfo + uint64_t(index) * 16UL;
    Ptr<uint2>(infoBase + 0UL)[0] = pairInd0;
    Ptr<uint>(infoBase + 8UL)[0] = pairInd1;

    Ptr<float4>(gCg.w_outBarycentric)[index] = barycentric;
}

// ===========================================================================
[shader("compute")]
[numthreads(256, 1, 1)]
void cloth_psContactGenLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint globalThreadIdx = gtid.x + gid.x * PSCG_BLOCK;

    const uint stackSize = gCg.stackSizeBytes / 16u;
    const uint numFound = Ptr<uint>(gCg.midphasePairsNum)[0];
    const uint numPairs = min(numFound, stackSize);      // C8

    Ptr<uint4> pairs = Ptr<uint4>(gCg.stackPtr);
    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gCg.femClothes);
    Ptr<PxgParticleSystemGpu> particleSystems = Ptr<PxgParticleSystemGpu>(gCg.particleSystems);

    for (uint i = globalThreadIdx; i < numPairs; i += PSCG_STRIDE)
    {
        const uint4 curPair = pairs[i];
        const uint cmIdx = curPair.x;
        const uint particleIndex = curPair.y;            // C1
        const uint triangleIdx = curPair.z;

        PxgShape particleShape, clothShape;
        uint particleCacheRef, clothCacheRef;
        loadParticleClothPair(cmIdx, particleShape, particleCacheRef, clothShape, clothCacheRef);

        Ptr<float> contactDistance = Ptr<float>(gCg.contactDistance);
        const float cDistance = contactDistance[particleCacheRef] + contactDistance[clothCacheRef];
        const float restDistance = Ptr<float>(gCg.restDistance)[cmIdx];

        const uint clothId = clothShape.particleOrSoftbodyId;
        const uint particleSystemId = particleShape.particleOrSoftbodyId;

        Ptr<float4> sortedPose = Ptr<float4>(particleSystems[particleSystemId].mSortedPositions_InvMass);
        Ptr<float4> meshVerts = Ptr<float4>(clothes[clothId].mPosition_InvMass);
        Ptr<uint4> triIndices = Ptr<uint4>(clothes[clothId].mTriangleVertexIndices);

        const uint4 triVertIdx = triIndices[triangleIdx];
        const float3 a = meshVerts[triVertIdx.x].xyz;
        const float3 b = meshVerts[triVertIdx.y].xyz;
        const float3 c = meshVerts[triVertIdx.z].xyz;
        const float3 p = sortedPose[particleIndex].xyz;

        const float3 ab = b - a;
        const float3 ac = c - a;

        const float3 closestPt = closestPtPointTriangle2(p, a, b, c, ab, ac);
        const float3 v = p - closestPt;
        const float sqDist = dot(v, v);

        const float sqContactDistance = cDistance * cDistance;

        if (sqDist <= sqContactDistance)                 // C5
        {
            const float m = sqrt(sqDist);
            // C6: unguarded, and a RECIPROCAL-MULTIPLY -- femClothPrimitives.cu:2837
            // writes `PxVec3 n = v / m`, which is PxVec3::operator/(Type)
            // (PxVec3.h:219-223): 1.0/m formed once, then three multiplies.  P9.36.
            const float3 n = v * (1.0f / m);
            const float pen = m - restDistance;          // C4
            const float3 contact = p - n * restDistance; // C4

            const float4 tBarycentric = computeBarycentric(a, b, c, closestPt);

            uint index = 0u;
            InterlockedAdd(Ptr<uint>(gCg.w_totalContactCount)[0], 1u, index);

            // PxEncodeParticleIndex (PxgSimulationCoreDesc.h:118):
            // (u64(systemId) << 32) | particleId.
            const uint2 pairInd0 = uint2(particleIndex, particleSystemId);
            const uint pairInd1 = (clothId << PXG_CLOTH_ID_SHIFT) | triangleIdx;

            writeContact(index, float4(contact, 0.0f),
                         float4(-n.x, -n.y, -n.z, pen),  // C4: the NEGATED normal
                         tBarycentric, pairInd0, pairInd1, pairInd0);   // C3
        }
    }

    // C8: the UNCLAMPED count, in bytes, by one thread of the grid.
    if (globalThreadIdx == 0u)
        InterlockedMax(Ptr<uint>(gCg.stackSizeNeeded)[0], numFound * 16u);
}
