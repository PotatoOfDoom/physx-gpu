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

// softBodyPsContactGenBda.slang
//
// P9.39c -- sb_psContactGenLaunch
// gpunarrowphase/src/CUDA/softbodyPrimitives.cu:2086 (wrapper), :1961
// (sbParticleCollision), :1730 (PointOutsideOfPlane4), :1864
// (closestPtTetrahedron) and, in PhysX 5.12, PxgDeformableContactWriters.h
// (PxgParticleDbContactWriter::writeVolumeContact) -- all read in full.
//
// The SECOND half of the softbody-vs-particle chain, on top of P9.39a's midphase.
// One particle against one tetrahedron, at most one contact, no warp aggregation:
// every contact takes its own atomicAdd of 1.
//
// PhysX 5.12 REMOVED this kernel's filter (the particle-buffer filter chain P9.39b was
// written for is gone from the engine), its rest distance, its barycentric and the
// sorted-to-unsorted lookup that keyed the filter.  What is left is the surface-hint
// gate, the closest point and the volume write.
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
// S1 -- THE PAIR IS (cmIdx, particleIndex, tetrahedronIdx), which is exactly what
//     P9.39a's midphase pushes.  The y and z words are NOT interchangeable and the
//     midphase fixture pins their order.
//
// S2 -- ONE GATE (5.12): surfaceMask == 0 RETURNS.  5.6.1 also skipped the pair when
//     the (unsorted particle, whole softbody) or (unsorted particle, tet) pair was in the
//     particle-buffer filter list; 5.12 dropped that filter, and with it 5.6.1's
//     sorted/unsorted index split -- the particle is named by its SORTED index throughout.
//
// S4 -- PointOutsideOfPlane4 IS USED ONLY AS A CONJUNCTION HERE.  The engine returns
//     a float4 of four sign products and this kernel asks only
//     `x >= 0 && y >= 0 && z >= 0 && w >= 0`, which is exactly what the module's
//     `isPointOutsideOfPlane4` returns.  That substitution is legitimate HERE and
//     NOT in general: getClosestPtPointTetrahedron (softbodyPrimitives.cu:1762) reads
//     the components individually (`result.x < 0.f`), so the bool form cannot serve
//     it.  Recorded so the next reader does not "simplify" that one too.
//
// S5 -- TWO PENETRATION BRANCHES THAT COMPUTE DIFFERENT THINGS.
//     Inside the tetrahedron:  pen = dot(n, n2), and the normal becomes n2 -- the
//       SURFACE normal from closestPtTetrahedron, not the point-to-point direction.
//     Outside:  the squared-distance early-out fires FIRST (sqDist > cDistance^2 is
//       a return, and cDistance is squared, not compared against sqrt), then
//       pen = |n| (via normalize), and n is replaced by n2 only when the two disagree
//       (dot < 0.8) or the distance is degenerate (sqDist < 1e-8).
//     PhysX 5.12 moved the rest distance out of contact generation: 5.6.1 subtracted
//     restDistances[cmIdx] in both branches.
//     The contact distance gate exists ONLY in the outside branch: a particle inside
//     the tetrahedron always produces a contact regardless of cDistance.
//
// S6 -- `pen = n.normalize()` IS A RECIPROCAL-MULTIPLY, NOT A DIVIDE.  PxVec3::normalize
//     (PxVec3.h:295) takes the magnitude, and `*this /= m` (PxVec3.h:260) forms the
//     reciprocal ONCE and multiplies three times.  Written here as `n * (1.0f/m)`.
//     Spelling it `n / m` is three componentwise divides, which differ from the engine
//     in at least one component for about half of all vectors -- the P9.36 defect
//     class, and gpuslang.vectorDivide fails the build on it.
//
// S7 -- THE COUNTER STILL INCREMENTS WHEN THE BUFFER IS FULL.  atomicAdd runs before
//     the maxNumContacts test, so totalContactCount overshoots and the host reads the
//     NEEDED count, not the written count.  Transcribed, not "fixed".
//
// S8 -- mTetMeshSurfaceHint IS A PxU8 ARRAY.  llvmpipe has no 8-bit storage in these
//     shaders, so the containing 32-bit word is read and the byte extracted.  The word
//     address is formed with `& ~uint64_t(3)`: writing `& ~3UL` MISCOMPILES, because
//     Slang evaluates the complement in 32 bits and clears the top half of the device
//     address.
//
// S9 -- THE COLLISION MESH, NOT THE SIMULATION MESH.  mTetIndices (@16) and
//     mPosition_InvMass (@64), never mSimTetIndices (@80) / mSimPosition_InvMass (@96).
//
// S10 -- stackSizeNeededOnDevice IS WRITTEN BY THREAD 0 OF THE WHOLE GRID with the
//     UNCLAMPED pair count times 16, so the host learns how much stack it should have
//     had.  min(numFound, stackSize) is used for the loop but NOT for this atomicMax.

module softBodyPsContactGenBda;

import pxgSlangSoftBodyDesc;    // PxgSoftBodyGpu (mTetMeshSurfaceHint@8, mTetIndices@16,
                                //                 mPosition_InvMass@64)
import pxgSlangParticleDesc;    // PxgParticleSystemGpu (mSortedPositions_InvMass@24)
import pxgSlangDeformableRef;   // PxgEncodeParticleRef, PxgEncodeDeformableTetrahedron
import pxgSlangDeformableWriters; // PxgParticleDbContactWriterGpu
import pxgSlangTetClosest;      // isPointOutsideOfPlane4 / closestPtTetrahedron

static const uint GEOM_PARTICLESYSTEM = 6u;


static const uint SBPSCG_GRID  = 512u;
static const uint SBPSCG_BLOCK = 256u;
static const uint SBPSCG_STRIDE = SBPSCG_GRID * SBPSCG_BLOCK;

// sb_psContactGenLaunch (5.12): twelve arguments, the last of them
// PxgParticleDbContactWriter by value (72 B, paramAlign 8) -> 160-byte blob.
// ABI row {4,8,8,8,8,8,8,4,8,8,8,72}.
struct SbPsCgParams
{
    float toleranceLength;            // @0   (ABI parity only)
    uint _pad4;                       // @4
    uint64_t cmInputs;                // @8
    uint64_t transformCache;          // @16  (ABI parity only -- nothing is transformed)
    uint64_t contactDistance;         // @24
    uint64_t gpuShapes;               // @32
    uint64_t particleSystems;         // @40
    uint64_t softbodies;              // @48
    uint stackSizeBytes;              // @56
    uint _pad60;                      // @60
    uint64_t stackPtr;                // @64
    uint64_t midphasePairsNum;        // @72
    uint64_t stackSizeNeeded;         // @80  atomicMax output
    // ---- PxgParticleDbContactWriter (by value, 72 B) @88 ----
    uint64_t w_outPoint;                     // @88
    uint64_t w_outNormalPen;                 // @96
    uint64_t w_outBarycentric;               // @104 (not written: writeVolumeContact)
    uint64_t w_outContactInfo;               // @112 PxgDeformablePairInfo*
    uint64_t w_totalContactCount;            // @120
    uint64_t w_contactByParticle;            // @128 PxU64* -- the PARTICLE sort key
    uint64_t w_tempContactByParticle;        // @136
    uint64_t w_contactIndexSortedByParticle; // @144
    uint w_maxNumContacts;                   // @152
    uint _pad156;                            // @156 -> 160
}

[[vk::push_constant]] ConstantBuffer<SbPsCgParams> gSc;

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

// LoadShapePair<ePARTICLESYSTEM, eTETRAHEDRONMESH> -- routed by TYPE, exactly as the
// midphase routes it.  Assuming shapeRef0 is the particle silently reads the softbody's
// id as a particle-system index.
void loadParticleSoftbodyPair(uint cmIdx,
                              out PxgShape particleShape, out uint particleCacheRef,
                              out PxgShape softbodyShape, out uint softbodyCacheRef)
{
    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(gSc.cmInputs)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gSc.gpuShapes);
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

// S8: mTetMeshSurfaceHint is a PxU8 array; read the containing 32-bit word and extract
// the byte.  `& ~3UL` would truncate the device address -- see the header.
uint surfaceHint(uint64_t hintBase, uint primIdx)
{
    const uint64_t addr = hintBase + uint64_t(primIdx);
    const uint word = Ptr<uint>(addr & ~uint64_t(3))[0];
    const uint shift = uint(addr & 3UL) * 8u;
    return (word >> shift) & 0xffu;
}

PxgParticleDbContactWriterGpu scWriter()
{
    PxgParticleDbContactWriterGpu w;
    w.outPoint = gSc.w_outPoint;                 w.outNormalPen = gSc.w_outNormalPen;
    w.outBarycentric = gSc.w_outBarycentric;     w.outContactInfo = gSc.w_outContactInfo;
    w.totalContactCount = gSc.w_totalContactCount;
    w.contactByParticle = gSc.w_contactByParticle;
    w.tempContactByParticle = gSc.w_tempContactByParticle;
    w.contactIndexSortedByParticle = gSc.w_contactIndexSortedByParticle;
    w.maxNumContacts = gSc.w_maxNumContacts;
    return w;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sb_psContactGenLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint globalThreadIdx = gtid.x + gid.x * SBPSCG_BLOCK;

    const uint stackSize = gSc.stackSizeBytes / 16u;
    const uint numFound = Ptr<uint>(gSc.midphasePairsNum)[0];
    const uint numPairs = min(numFound, stackSize);

    Ptr<uint4> pairs = Ptr<uint4>(gSc.stackPtr);
    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gSc.softbodies);
    Ptr<PxgParticleSystemGpu> particleSystems = Ptr<PxgParticleSystemGpu>(gSc.particleSystems);

    for (uint i = globalThreadIdx; i < numPairs; i += SBPSCG_STRIDE)
    {
        // P9.49: safe ADDRESS -- this loop starts at a grid-wide id under a
        // PERSISTENT grid, so its first index can already be past numPairs and
        // the guard is only a mask here.  Identity for every admitted lane.
        const uint4 curPair = pairs[min(i, max(numPairs, 1u) - 1u)];
        const uint cmIdx = curPair.x;                    // S1
        const uint particleIndex = curPair.y;
        const uint tetrahedronIdx = curPair.z;

        PxgShape particleShape, softbodyShape;
        uint particleCacheRef, softbodyCacheRef;
        loadParticleSoftbodyPair(cmIdx, particleShape, particleCacheRef,
                                 softbodyShape, softbodyCacheRef);

        Ptr<float> contactDistance = Ptr<float>(gSc.contactDistance);
        const float cDistance = contactDistance[particleCacheRef] +
                                contactDistance[softbodyCacheRef];

        const uint softbodyId = softbodyShape.particleOrSoftbodyId;
        const uint particleSystemId = particleShape.particleOrSoftbodyId;

        Ptr<float4> sortedPose = Ptr<float4>(particleSystems[particleSystemId].mSortedPositions_InvMass);
        Ptr<float4> tetmeshVerts = Ptr<float4>(softbodies[softbodyId].mPosition_InvMass);   // S9
        Ptr<uint4> tetIndices = Ptr<uint4>(softbodies[softbodyId].mTetIndices);             // S9

        const uint4 tetIdx = tetIndices[tetrahedronIdx];

        const uint surfaceMask = surfaceHint(softbodies[softbodyId].mTetMeshSurfaceHint,
                                             tetrahedronIdx);

        // S2 (5.12): the only gate.
        if (surfaceMask == 0u)
            continue;

        const float3 a = tetmeshVerts[tetIdx.x].xyz;
        const float3 b = tetmeshVerts[tetIdx.y].xyz;
        const float3 c = tetmeshVerts[tetIdx.z].xyz;
        const float3 d = tetmeshVerts[tetIdx.w].xyz;

        const float3 p = sortedPose[particleIndex].xyz;

        // S4: the engine's float4 is consumed only as this conjunction.
        const bool inside = isPointOutsideOfPlane4(p, a, b, c, d);

        float3 closestP = float3(0.0f, 0.0f, 0.0f);
        float3 n2 = float3(0.0f, 0.0f, 0.0f);
        float sqDist = 3.4028234663852886e+38f;          // PX_MAX_F32
        closestPtTetrahedron(p, surfaceMask, a, b, c, d, n2, closestP, sqDist);

        float3 n = -(p - closestP);
        float pen;

        if (inside)                                       // S5
        {
            pen = dot(n, n2);
            n = n2;
        }
        else
        {
            const float sqContactDistance = cDistance * cDistance;
            if (sqDist > sqContactDistance)
                continue;

            // S6: PxVec3::normalize -- magnitude, then ONE reciprocal times three.
            //
            // sqrt(dot(n, n)) AND NOT length(n).  PxVec3::magnitude (PxVec3.h:270) is
            // PxSqrt(magnitudeSquared()) -- an OpDot followed by a correctly-rounded
            // OpSqrt, both exactly specified.  Slang's `length` lowers to
            // GLSL.std.450 Length, which the SPIR-V environment spec allows 2 ULP of
            // error on: a DIFFERENT instruction with a slacker contract, whose
            // agreement would have to be measured per driver the way P9.33a had to
            // measure InverseSqrt.  Spelling it out costs nothing and needs no
            // measurement.
            const float m = sqrt(dot(n, n));
            if (m > 0.0f)
                n = n * (1.0f / m);
            pen = m;

            if (sqDist < 1e-8f || dot(n, n2) < 0.8f)
                n = n2;
        }

        // S7: the counter increments even when the write is dropped.
        uint index;
        InterlockedAdd(Ptr<uint>(gSc.w_totalContactCount)[0], 1u, index);
        if (index >= gSc.w_maxNumContacts)
            continue;

        const uint64_t particleRef = PxgEncodeParticleRef(particleSystemId, particleIndex);
        const uint64_t tetRef = PxgEncodeDeformableTetrahedron(softbodyId, tetrahedronIdx);
        scWriter().writeVolumeContact(index, float4(p, 0.0f), float4(n, pen), particleRef, tetRef);
    }

    if (globalThreadIdx == 0u)                            // S10
    {
        InterlockedMax(Ptr<uint>(gSc.stackSizeNeeded)[0], numFound * 16u);
    }
}
