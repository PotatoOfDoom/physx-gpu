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
// (closestPtTetrahedron), deformableCollision.cuh:96 (computeTetBarycentric),
// deformableElementFilter.cuh:167 (the particle filter chain),
// PxgFEMCore.h:467/:428 (PxgFEMContactWriter::writeContact) -- all read in full.
//
// The SECOND half of the softbody-vs-particle chain, on top of P9.39a's midphase.
// One particle against one tetrahedron, at most one contact, no warp aggregation:
// every contact takes its own atomicAdd of 1.
//
// This is the kernel P9.39b existed for.  Its filter is the ONLY caller in the
// corpus that needs `ignoreIndex2 = false` and a real index2, and it reaches it
// through dfFindParticleNonRigid -- the particle-buffer chain, not the rigid one.
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
// S1 -- THE PAIR IS (cmIdx, particleIndex, tetrahedronIdx), which is exactly what
//     P9.39a's midphase pushes.  The y and z words are NOT interchangeable and the
//     midphase fixture pins their order.
//
// S2 -- THREE GATES, SHORT-CIRCUITED IN THIS ORDER:
//         surfaceMask == 0  ||  find(..., particleMask, tetFullMask)
//                           ||  find(..., particleMask, tetMask)
//     and any of them RETURNS -- this is a skip, not a contact.  The two finds use
//     the SAME particle key and differ only in the second: tetFullMask names the
//     WHOLE softbody (PX_MAX_NB_DEFORMABLE_VOLUME_TET as the tet id), tetMask names
//     this one tetrahedron.  Collapsing them to one lookup loses whole-body
//     filtering; swapping their order is unobservable but written as the engine
//     writes it.
//
// S3 -- TWO DIFFERENT PARTICLE INDICES, AND THEY ARE NOT INTERCHANGEABLE.
//     The FILTER key is the UNSORTED index (mSortedToUnsortedMapping[particleIndex]);
//     the POSITION and the contact's pairInd0 use the SORTED index unchanged.  The
//     engine spells this out by building `particleMask` from unsortedParticleIndex
//     and `compressedParticleIndex` from particleIndex, thirty lines apart.  A port
//     that uses one index for both compiles, runs, and filters the wrong particle.
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
//     Inside the tetrahedron:  pen = dot(n, n2) - restDistance, and the normal
//       becomes n2 -- the SURFACE normal from closestPtTetrahedron, not the
//       point-to-point direction.
//     Outside:  the squared-distance early-out fires FIRST (sqDist > cDistance^2 is
//       a return, and cDistance is squared, not compared against sqrt), then
//       pen = |n| (via normalize) - restDistance, and n is replaced by n2 only when
//       the two disagree (dot < 0.8) or the distance is degenerate (sqDist < 1e-8).
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
import pxgSlangParticleDesc;    // PxgParticleSystemGpu (mSortedPositions_InvMass@24,
                                //                       mSortedToUnsortedMapping@56)
import pxgSlangDefFilter;       // P9.39b: dfFindParticleNonRigid
import pxgSlangTetClosest;      // isPointOutsideOfPlane4 / closestPtTetrahedron

static const uint GEOM_PARTICLESYSTEM = 6u;

// PxgSimulationCoreDesc.h:104 -- the sentinel tet id that names the WHOLE softbody.
static const uint PX_MAX_NB_DEFORMABLE_VOLUME_TET = 0xfffffu;

static const uint SBPSCG_GRID  = 512u;
static const uint SBPSCG_BLOCK = 256u;
static const uint SBPSCG_STRIDE = SBPSCG_GRID * SBPSCG_BLOCK;

struct SbPsCgParams
{
    float toleranceLength;            // @0   (ABI parity only)
    uint _pad4;                       // @4
    uint64_t cmInputs;                // @8
    uint64_t transformCache;          // @16  (ABI parity only -- nothing is transformed)
    uint64_t contactDistance;         // @24
    uint64_t restDistance;            // @32
    uint64_t gpuShapes;               // @40
    uint64_t particleSystems;         // @48
    uint64_t softbodies;              // @56
    uint64_t filterPairs;             // @64
    uint numFilterPairs;              // @72
    uint stackSizeBytes;              // @76
    uint64_t stackPtr;                // @80
    uint64_t midphasePairsNum;        // @88
    uint64_t stackSizeNeeded;         // @96  atomicMax output
    uint2 _pad104;                    // @104 -> the by-value writer is 16-ALIGNED
    // ---- PxgFEMContactWriter (by value, 80 B) @112 ----
    uint64_t w_outPoint;              // @112
    uint64_t w_outNormalPen;          // @120
    uint64_t w_outBarycentric;        // @128
    uint64_t w_outContactInfo;        // @136  PxgFemOtherContactInfo* (16 B stride)
    uint64_t w_totalContactCount;     // @144
    uint64_t w_contactByX;            // @152  PxU64* -- the PARTICLE sort key
    uint64_t w_tempContactByX;        // @160
    uint64_t w_contactIndexSortedByX; // @168
    uint64_t w_contactSortedByX;      // @176  NULL for particles; never written
    uint w_maxNumContacts;            // @184
    uint _pad188;                     // @188 -> 192
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

// P9.86: PxClamp(t, 0, 1) = PxMin(1, PxMax(0, t)) (PxMath.h:142-146), and the float
// PxMax/PxMin are the selects `a > b ? a : b` / `a < b ? a : b`
// (PxUnixMathIntrinsics.h:116-125).  Slang's clamp is GLSL.std.450 FClamp, which
// llvmpipe lowers to maxps(t, +0): +0 for t = -0 (and 0 for NaN), where the CUDA
// keeps -0 (and NaN).  Identical to FClamp for every other input.
float sbPsPxClamp01(float t)
{
    const float lo = (0.0f > t) ? 0.0f : t;   // PxMax(0, t)
    return (1.0f < lo) ? 1.0f : lo;           // PxMin(1, lo)
}

// deformableCollision.cuh:96.  Four SEPARATE divisions by detBcd -- the engine writes
// `detPcd / detBcd` three times over and never forms a reciprocal, so this is one of
// the few places a divide is faithful.  The sum is `((1 - v) - w) - x`, left to right.
bool computeTetBarycentric(float3 a, float3 b, float3 c, float3 d, float3 point,
                           out float4 barycentric)
{
    const float eps = 1e-5f;
    const float minEps = -eps, maxEps = 1.0f + eps;
    const float3 ba = b - a, ca = c - a, da = d - a, pa = point - a;

    const float detBcd = dot(ba, cross(ca, da));
    const float v = dot(pa, cross(ca, da)) / detBcd;
    const float w = dot(ba, cross(pa, da)) / detBcd;
    const float x = dot(ba, cross(ca, pa)) / detBcd;
    const float u = 1.0f - v - w - x;
    barycentric = float4(sbPsPxClamp01(u), sbPsPxClamp01(v),
                         sbPsPxClamp01(w), sbPsPxClamp01(x));   // P9.86: PxClamp (deformableCollision.cuh:133-136)
    return (u >= minEps && u <= maxEps) && (v >= minEps && v <= maxEps) &&
           (w >= minEps && w <= maxEps) && (x >= minEps && x <= maxEps);
}

// PxgFEMContactWriter::writeContact (PxgFEMCore.h:467) -> writeContactNoBarycentric
// (:428) plus outBarycentric.  Seven arrays; the word at info+12 (rigid material /
// inCollision) is NOT touched by this overload, and contactSortedByX is null for the
// particle writer and never written.
void writeContact(uint index, float4 contact, float4 normalPen, float4 barycentric,
                  uint2 pairInd0, uint pairInd1, uint2 id)
{
    if (index >= gSc.w_maxNumContacts)
        return;

    Ptr<uint2>(gSc.w_contactByX)[index] = id;
    Ptr<uint>(gSc.w_tempContactByX)[index] = id.x;      // PxU32(id & 0xffffffff)
    Ptr<uint>(gSc.w_contactIndexSortedByX)[index] = index;

    Ptr<float4>(gSc.w_outPoint)[index] = contact;
    Ptr<float4>(gSc.w_outNormalPen)[index] = normalPen;

    const uint64_t infoBase = gSc.w_outContactInfo + uint64_t(index) * 16UL;
    Ptr<uint2>(infoBase + 0UL)[0] = pairInd0;
    Ptr<uint>(infoBase + 8UL)[0] = pairInd1;

    Ptr<float4>(gSc.w_outBarycentric)[index] = barycentric;
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
        const float restDistance = Ptr<float>(gSc.restDistance)[cmIdx];

        const uint softbodyId = softbodyShape.particleOrSoftbodyId;
        const uint particleSystemId = particleShape.particleOrSoftbodyId;

        Ptr<float4> sortedPose = Ptr<float4>(particleSystems[particleSystemId].mSortedPositions_InvMass);
        Ptr<float4> tetmeshVerts = Ptr<float4>(softbodies[softbodyId].mPosition_InvMass);   // S9
        Ptr<uint4> tetIndices = Ptr<uint4>(softbodies[softbodyId].mTetIndices);             // S9

        const uint4 tetIdx = tetIndices[tetrahedronIdx];

        // S3: the FILTER key is the UNSORTED index; the position below uses the sorted one.
        const uint unsortedParticleIndex =
            Ptr<uint>(particleSystems[particleSystemId].mSortedToUnsortedMapping)[particleIndex];

        // PxEncodeParticleIndex (PxgSimulationCoreDesc.h:118): (systemId << 32) | index.
        // The filter overload takes a PxU32 and the engine hands it this PxU64, so the
        // system id is TRUNCATED AWAY at the call -- only the index survives.  Written
        // as the surviving half rather than as a u64 that is immediately discarded.
        const uint particleMask = unsortedParticleIndex;

        // PxEncodeSoftBodyIndex (PxgSimulationCoreDesc.h:110): (id << 20) | tetId.
        const uint tetMask = (softbodyId << 20u) | tetrahedronIdx;
        const uint tetFullMask = (softbodyId << 20u) | PX_MAX_NB_DEFORMABLE_VOLUME_TET;

        const uint surfaceMask = surfaceHint(softbodies[softbodyId].mTetMeshSurfaceHint,
                                             tetrahedronIdx);

        // S2.  The null-pair bound stays at the CALL SITE, as pxgSlangDefFilter's header
        // requires: with no filter pairs the module is handed a known-safe address and
        // r = 0, so the speculated load in its loop cannot fault on llvmpipe.
        const bool havePairs = (gSc.filterPairs != 0UL) && (gSc.numFilterPairs != 0u);
        const uint64_t pairsAddr = havePairs ? gSc.filterPairs : gSc.softbodies;
        const uint nbPairs = havePairs ? gSc.numFilterPairs : 0u;

        if (surfaceMask == 0u ||
            dfFindParticleNonRigid(gSc.particleSystems +
                                       uint64_t(particleSystemId) * PXG_PARTICLE_SYSTEM_STRIDE,
                                   pairsAddr, nbPairs, particleMask, tetFullMask) ||
            dfFindParticleNonRigid(gSc.particleSystems +
                                       uint64_t(particleSystemId) * PXG_PARTICLE_SYSTEM_STRIDE,
                                   pairsAddr, nbPairs, particleMask, tetMask))
            continue;

        const float3 a = tetmeshVerts[tetIdx.x].xyz;
        const float3 b = tetmeshVerts[tetIdx.y].xyz;
        const float3 c = tetmeshVerts[tetIdx.z].xyz;
        const float3 d = tetmeshVerts[tetIdx.w].xyz;

        const float3 p = sortedPose[particleIndex].xyz;

        // S4: the engine's float4 is consumed only as this conjunction.
        const bool inside = isPointOutsideOfPlane4(p, a, b, c, d);

        float4 tBarycentric;
        float3 closestP = float3(0.0f, 0.0f, 0.0f);
        float3 n2 = float3(0.0f, 0.0f, 0.0f);
        float sqDist = 3.4028234663852886e+38f;          // PX_MAX_F32
        closestPtTetrahedron(p, surfaceMask, a, b, c, d, n2, closestP, sqDist);

        computeTetBarycentric(a, b, c, d, closestP, tBarycentric);

        float3 n = -(p - closestP);
        float pen;

        if (inside)                                       // S5
        {
            pen = dot(n, n2) - restDistance;
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

            pen -= restDistance;
        }

        // S7: the counter increments even when the write is dropped.
        uint index;
        InterlockedAdd(Ptr<uint>(gSc.w_totalContactCount)[0], 1u, index);
        if (index >= gSc.w_maxNumContacts)
            continue;

        const uint2 compressedParticleIndex = uint2(particleIndex, particleSystemId);
        writeContact(index, float4(p, 0.0f), float4(n, pen), tBarycentric,
                     compressedParticleIndex, tetMask, compressedParticleIndex);
    }

    if (globalThreadIdx == 0u)                            // S10
    {
        InterlockedMax(Ptr<uint>(gSc.stackSizeNeeded)[0], numFound * 16u);
    }
}
