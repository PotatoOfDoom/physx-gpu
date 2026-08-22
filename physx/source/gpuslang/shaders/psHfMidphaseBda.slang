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

// psHfMidphaseBda.slang
//
// P9.11 -- ps_heightfieldCollisonLaunch (particleSystemHFMidPhaseCG.cu:262, body
// psHeightfieldMidphaseCollision at :81-250) on the engine BDA ABI: the
// particle-vs-HEIGHTFIELD midphase and contact generation.  The SECOND of the
// three producers of mOneWayContactInfos, after P9.10's triangle-mesh twin.
//
// PROVENANCE.  As in P9.10, this is a LIFT and not a second transcription.
// shaders/psHfCollide.slang is a Phase-1 port of the same body with its own
// fixture (psHfCollide_core, which carries a full host reference), and its header
// lists exactly what it lacks: the engine glue.  This module lifts that body onto
// BDA and adds only glue.  The fixture proves the lift GPU-DIFFERENTIALLY -- one
// arena, both entry points, byte compare -- rather than by a third reading.
//
// It is a MUCH smaller lift than the mesh twin, and the three reasons are worth
// stating because they remove three whole classes of hazard:
//   * one particle per thread, no warp cooperation anywhere in the body, so no
//     ballot emulation and no barriers;
//   * NO rigid filter -- this kernel never calls find(particleSystem, ...) and
//     never reads gridParticleIndex, so upstream candidates 17 and 18 (the
//     unbounded index read and the divergent return in front of a ballot) simply
//     do not arise here;
//   * the per-particle contact list is a genuine per-thread local array, so
//     there is no shared-memory staging to reproduce.
//
// LAUNCH, measured at PxgNarrowphaseCore.cpp:4065:
//   grid  (PxgParticleSystemKernelGridDim::PS_HEIGHTFIELD_COLLISION = 4096,
//          numTests, 2)
//   block (PxgParticleSystemKernelBlockDim::PS_HEIGHTFIELD_COLLISION = 64, 1, 1)
// Note the block is 64 THREADS, not a warp shape -- there is no warp structure in
// the body to respect.
//
// Args -> ABI {1,4,8,8,8,8,8,8,8,72}, 136-byte blob -- BYTE-IDENTICAL to
// ps_meshCollisonLaunch's (PxgNarrowphaseCore.cpp:4049-4060 against :3952-3963).
//   (bool isTGS, PxReal toleranceLength, cmInputs*, transformCache*,
//    contactDistance*, restDistances*, gpuShapes*, particleSystems*,
//    shapeToRigidRemapTable*, PxgParticleContactWriter BY VALUE)
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
//  H1. gridDim.Z == 2 IS THE DIFFUSE SPLIT and gridDim.Y is the pair index, the
//      same shape as the mesh twin.  A port that ignores Z runs the solid pass
//      twice and doubles every contact.
//  H2. THE DIFFUSE EARLY-OUT IS A DIFFERENT SHAPE FROM THE MESH TWIN'S, and the
//      difference is observable.  The mesh kernel returns on
//      `isDiffuse && mMaxDiffuseParticles == 0`.  This one instead sets
//          numParticles = (mMaxDiffuseParticles > 0) ? *mNumDiffuseParticles : 0
//      and then returns on `isDiffuse && numParticles == 0` (:340-345).  Those
//      differ exactly when maxDiffuse > 0 and the live count is 0 -- the mesh
//      kernel proceeds (and its work loop finds nothing), this one returns.  Same
//      output, different path; but a port that copies the mesh kernel's guard and
//      then dereferences mNumDiffuseParticles unconditionally reads a null
//      pointer on every non-diffuse system.  Transcribe THIS one.
//  H3. `toleranceLength` is an ARGUMENT AND IS NEVER READ by this body, exactly as
//      in the mesh twin.  It stays in the ABI blob because dropping it would shift
//      every later offset by four bytes.
//  H4. THE SHAPE PAIR IS LOADED IN THE OPPOSITE ORDER from the mesh twin:
//      LoadShapePairWarp<ePARTICLESYSTEM, eHEIGHTFIELD> (:293) against the mesh
//      kernel's LoadShapePair<eTRIANGLEMESH, ePARTICLESYSTEM>.  Copying P9.10's
//      loadTrimeshParticlePair and only swapping the type constant gets the
//      transform-cache refs crossed.
//  H5. `oneOverHeightScale` is 1/scale.y WITHOUT the absolute value, while row and
//      column both take PxAbs (:117-119).  Faithful, and asymmetric on purpose.
//  H6. The negative-scale min/max swap (:131-146) and getTriangle's handedness
//      vertex swap (heightfieldUtil.cuh:271-280) are transcribed but are DEAD
//      upstream: PxHeightFieldGeometry::isValid() rejects scales below
//      PX_MIN_HEIGHTFIELD_XZ_SCALE / _Y_SCALE, and for a hypothetical negative
//      scale the |scale| division followed by a bare swap leaves min > max so
//      maxRow - minRow underflows.  Ported because the CUDA has it, not because
//      anything reaches it -- and psHfCollide's fixture already says so.
//  H7. PxBitAndByte's conversion MASKS OFF THE HIGH BIT.
//      `operator storageType() const { return storageType(mData & ~bitMask); }`
//      (PxBitAndData.h:54-57, bitMask 0x80), so `PxU32 material =
//      sample.materialIndex0` is `byte & 0x7F`, NOT the raw byte.  The high bit of
//      materialIndex0 is the TESSELLATION FLAG.  A port that compares the raw byte
//      against eHOLE = 127 misses every hole on a tess-flagged cell and quietly
//      generates contacts through it.
//  H8. The sample's height is a SIGNED 16-bit value in the low half of the 4-byte
//      sample word (PxHeightFieldSample.h:73), so it needs a sign-extending read,
//      not a mask.
//  H9. getMinRow clamps to rows-2 and getMaxRow to rows-1 (heightfieldUtil.cuh:37-58)
//      -- DIFFERENT bounds, and the same asymmetry for columns.
// H10. The cell loop runs (maxRow - minRow) * columnSpan * 2 iterations with
//      `index = i / 2` and the material selected by `i & 1`, so consecutive i
//      visit the SAME vertex with its two triangles.
// H11. THERE IS NO WORST-CONTACT REPLACEMENT PASS.  The mesh twin has one; this
//      kernel simply DROPS everything past MaxStaticContactsPerMesh = 6, and the
//      CUDA says so in a comment at :215-216 ("this code still misses a logic from
//      the triangle mesh near phase").  Transcribed as-is.
// H12. The finalize QUIRK -- see CANDIDATE 20 below.  A PLAIN STORE of the count
//      follows the atomicAdd and overwrites it.
// H13. Consequently `remaining = contactCounts - contactWritten` is ALWAYS ZERO
//      for a static rigid (contactCounts <= 6 <= 12 = the cap), so the shared
//      writer path only ever fires for a DYNAMIC heightfield, where contactWritten
//      stays 0 and the whole temp list is emitted from src = 0.
// H14. A CONTRAST WORTH RECORDING: this kernel's writer tail guards its
//      atomicAdd with `!isDiffuse` (:243), so it does NOT have the mesh twin's
//      reserve-then-suppress defect (candidate 19).  The two kernels were written
//      by the same hand and only one of them got this right.
// H15. The one-way output index is `particleIndex + i * numParticles` (:236) --
//      COLUMN-major with a numParticles stride, the same layout
//      ps_accumulateStaticDensity reads (P9.8 trap B2).
//
// ---------------------------------------------------------------------------
// FIVE UPSTREAM CANDIDATES (20-24), TRANSCRIBED AND NOT FIXED.  Reachability is
// stated for each, because two of the five cannot be reached through the public
// API and saying otherwise would inflate the count.
// ---------------------------------------------------------------------------
//  CANDIDATE 20 -- LIVE, and the serious one.  The static finalize (:238-243) is
//      if (rigidId.isStaticBody())
//      {
//          if (contactCounts) { atomicAdd(&oneWayContactCounts[p], contactCounts); ... }
//          contactWritten = PxMin(contactCounts, MaxStaticContactsPerParticle);
//          oneWayContactCounts[p] = contactWritten;          // <-- PLAIN STORE
//      }
//      Note where the store SITS: inside isStaticBody() but OUTSIDE
//      `if (contactCounts)`.  So a pair that finds NOTHING still stores ZERO over
//      the counter.  Four failures follow:
//       (a) IT ERASES OTHER PRODUCERS.  The launch is grid (4096, numTests, 2), so
//           every heightfield pair in the scene runs concurrently and each one
//           plain-stores over the same particle's counter.  Last writer wins.  With
//           two heightfields the second one's count is read against the FIRST one's
//           slots, so contacts are substituted, not merely lost.
//       (b) IT ERASES THE OTHER KERNELS.  ps_meshCollisonLaunch and
//           ps_sdfMeshCollisonLaunch only ACCUMULATE into the same arrays, and all
//           three dispatch on the particle stream.  A heightfield pair that finds
//           nothing zeroes the count that the triangle-mesh kernel just built.
//       (c) IT IS A RACE, not just a lost update: an atomic read-modify-write and a
//           plain store to one address from different blocks, unordered.
//       (d) The stored value is derived from contactCounts, not from what was
//           actually written, so when the atomicAdd returns a base at or past the
//           twelve-contact cap the loop writes nothing while the count still claims
//           contacts -- and `remaining` is 0, so they are not forwarded either.
//  CANDIDATE 21 -- LATENT, unreachable with the current constants.  The writer loop
//      restarts at `src = 0` (:250) where the mesh twin resumes at the number
//      already consumed (particleSystemMeshMidphase.cu:361).  Leftovers would be
//      re-emitted from slot 0.  Unreachable because `remaining` is non-zero only
//      when contactWritten is 0 (H13), which holds as long as
//      MaxStaticContactsPerMesh (6) <= MaxStaticContactsPerParticle (12).  Raise
//      the former above the latter and it fires at once.
//  CANDIDATE 22 -- DEAD PATH, and a HAZARD FOR THE PORT (see D4).  If the H6 swaps
//      ever fired with both bounds positive and inverted -- lmin.x 3.5, lmax.x 1.2
//      -- then minRow 3, maxRow 2, and `maxRow - minRow` underflows to 0xFFFFFFFF.
//      The zero-span guard multiplies rather than compares, so 2 * span * 0xFFFFFFFF
//      is not zero and does NOT early out; the cell loop then runs ~4.29e9 times
//      reading far past nbRows * nbCols.  Unreachable through the public API
//      (PxHeightFieldGeometry.h:147 rejects non-positive scales), so the kernel is
//      relying on host validation for memory safety with no in-kernel guard.
//  CANDIDATE 23 -- DEAD PATH.  The y axis is divided by the SIGNED height scale
//      (:117) while x and z both take PxAbs (:118-119), and only x and z get the
//      min/max swap (:131-146).  A negative height scale therefore leaves
//      miny > maxy and inverts the meaning of the con0/con1 cull.  Unreachable for
//      the same reason as 22.  Either all three axes should take |scale| and swap,
//      or none should.
//  CANDIDATE 24 -- LIVE.  PxParticleRigidFilterPair filtering DOES NOT APPLY TO
//      HEIGHTFIELDS.  The mesh path filters at particleSystemMeshMidphase.cu:717;
//      this file has no equivalent and does not even include
//      deformableElementFilter.cuh.  A user who filters a particle/heightfield pair
//      through the public API gets contacts anyway.  A feature gap rather than an
//      arithmetic error -- and it is precisely why P9.10's laneDone machinery must
//      NOT be carried over here.
//
//  NOT A NEW CANDIDATE, because the vendor documents it in source: the missing
//  best-contact selection (H11).  With more than six candidates the six FIRST IN
//  CELL ORDER win rather than the six deepest.  The comment at :216-217 says so.
//  The fixture pins the first-six-in-order behaviour, which is also what makes the
//  cell iteration order byte-observable.
//
// ---------------------------------------------------------------------------
// FOUR DELIBERATE DIVERGENCES, all provably output-identical on every input the
// public API can produce.
// ---------------------------------------------------------------------------
//  D1. THE DIFFUSE COUNT IS READ BEHIND A BRANCH, NOT A TERNARY.  The CUDA writes
//          numParticles = isDiffuseParticlesActive ? *mNumDiffuseParticles : 0
//      (:319) -- a guarded dereference of a pointer that is NULL whenever the
//      system has no diffuse particles.  That is precisely P8.96c's fifth platform
//      rule: llvmpipe compiles a guarded UNIFORM load to an unconditional load
//      plus a select, and it faults.  The port uses an `if`.  Same value, and the
//      branch is block-uniform so it costs nothing.
//  D2. NO GROUPSHARED SCRATCH AND NO BARRIER.  The CUDA fills a block-uniform
//      PsHeightfieldScratch from warp 0 lane 0 and then __syncthreads() (:286-338).
//      Every field of it is a pure function of read-only kernel inputs, so having
//      every thread compute it is equal by construction -- and it removes a
//      barrier, which on the emulated-warp llvmpipe path is a liability rather
//      than an optimisation.  The only cost is that the descriptor reads happen
//      per thread instead of once per block.
//  D3. THE PER-PARTICLE CONTACT LIST IS A LOCAL ARRAY.  The CUDA declares
//      sTempContacts[blockDim][6] in shared memory but indexes it only by the
//      owning thread's own threadIdx.x (:278, :294), so it is per-thread private
//      storage spelled as shared.  A local float4[6] is the same object.
//  D4. THE CELL SPAN IS GUARDED AGAINST UNSIGNED UNDERFLOW.  Candidate 22 shows
//      that an inverted bound turns the cell loop into a ~4.29e9-iteration walk off
//      the end of the sample array.  On a GPU that is a very long kernel; on
//      llvmpipe it is an unbounded out-of-bounds read, and it would also convert
//      any mutant that inverts a bound from a clean kill into a hung sweep.  So the
//      port checks `maxRow >= minRow && maxColumn >= minColumn` before computing
//      the span.
//      IT CANNOT CHANGE AN OUTPUT.  With every scale positive -- which
//      PxHeightFieldGeometry::isValid() enforces -- the H6 swaps never fire, so
//      lmax >= lmin on both axes; getMinRow floors and clamps to rows-2 while
//      getMaxRow ceils and clamps to rows-1, so maxRow >= minRow follows, and the
//      same for columns.  The guard is therefore never taken on a legal scene, and
//      on an illegal one it replaces undefined behaviour with an early out.
//
// The module is in PXG_NOCONTRACT_SHADERS, and so is psHfCollide as of this
// tranche: the triangle tests are dense float work, FMA contraction moves the
// results, and the differential fixture compares the two BIT FOR BIT.  P9.10
// learned this the hard way -- psMeshMid was not in the list and the differential
// disagreed at the ULP on 14 of 97 particles with the counts matching everywhere.
// THE RULE THAT COST TWO TRANCHES TO LEARN: when a tranche lifts a Phase-1 module
// onto BDA for a differential fixture, the PHASE-1 MODULE MUST JOIN
// PXG_NOCONTRACT_SHADERS IN THE SAME TRANCHE.

import pxgSlangCommon;
import pxgSlangParticleDesc;

// ---------------------------------------------------------------------------
// Constants, all pinned by the fixture's layout test.
// ---------------------------------------------------------------------------
static const uint HF_COLLISION_BLOCK = 64u;    // PxgParticleSystemKernelBlockDim
static const uint HF_COLLISION_GRID = 4096u;   // PxgParticleSystemKernelGridDim

static const uint MAX_MESH_CONTACTS = 6u;      // MaxStaticContactsPerMesh
static const uint MAX_PARTICLE_CONTACTS = 12u; // MaxStaticContactsPerParticle
static const uint HOLE = 127u;                 // PxHeightFieldMaterial::eHOLE

// PxGeometryType (PxGeometry.h:48-62) is a plain 0-based enum:
//   eSPHERE 0, ePLANE 1, eCAPSULE 2, eBOX 3, eCONVEXCORE 4, eCONVEXMESH 5,
//   ePARTICLESYSTEM 6, eTETRAHEDRONMESH 7, eTRIANGLEMESH 8, eHEIGHTFIELD 9.
// The draft of this file had eHEIGHTFIELD at 5, which is eCONVEXMESH; the layout
// test pins both of these so that miscount cannot survive.
static const uint GEOM_PARTICLESYSTEM = 6u;    // PxGeometryType::ePARTICLESYSTEM
static const uint GEOM_HEIGHTFIELD = 9u;       // PxGeometryType::eHEIGHTFIELD

static const float PX_MAX_F32 = 3.4028235e38f;
static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_PGS = 1.0f;
static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_TGS = 0.5f;
static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_DIFFUSE = 1.0f;

// PxParticleFlag::eENABLE_SPECULATIVE_CCD (PxPBDParticleSystem.h:185).  It is
// 1 << 3; 1 << 2 is eFULL_DIFFUSE_ADVECTION.  P9.10's first draft had 1 << 2 and
// its layout test caught it -- the same test pins it here.
static const uint PX_PARTICLE_FLAG_SPECULATIVE_CCD = 8u;

// ---------------------------------------------------------------------------
// Push constants -- the ten CUDA arguments, writer inlined by value.  Byte for
// byte the same blob as psMeshMidphaseBda's PsMeshMidPush.
// ---------------------------------------------------------------------------
struct PsHfMidPush
{
    uint isTGSByte;                 // @0   CUDA bool, one byte; read masked
    float toleranceLength;          // @4   H3: present in the ABI, unused here
    uint64_t cmInputs;              // @8
    uint64_t transformCache;        // @16
    uint64_t contactDistance;       // @24
    uint64_t restDistances;         // @32
    uint64_t gpuShapes;             // @40
    uint64_t particleSystems;       // @48
    uint64_t shapeToRigidRemapTable;// @56
    // PxgParticleContactWriter, BY VALUE (PxgParticleSystem.h:604), 72 bytes @64
    uint64_t w_particleContacts;            // @64
    uint64_t w_numTotalContacts;            // @72
    uint64_t w_contactSortedByParticle;     // @80
    uint64_t w_tempContactByParticle;       // @88
    uint64_t w_contactIndexSortedByParticle;// @96
    uint64_t w_contactByRigid;              // @104
    uint64_t w_tempContactByRigid;          // @112
    uint64_t w_contactIndexSortedByRigid;   // @120
    uint w_maxContacts;                     // @128 -> 136
}
[[vk::push_constant]]
ConstantBuffer<PsHfMidPush> gHf;

struct PxgContactManagerInput { uint shapeRef0, shapeRef1, transformCacheRef0, transformCacheRef1; }

// PxgParticlePrimitiveContact (32 bytes) -- the record P9.8 and P9.10 also use.
struct PsContact
{
    float4 normal_pen;
    uint2 rigidId;
    uint2 particleId;
}

// ---------------------------------------------------------------------------
// The heightfield sample block, read over BDA.
//
// heightfieldShape.hullOrMeshPtr points at a PxU32 array whose [0] is nbRows,
// [1] is nbCols, and whose samples begin at [2] (particleSystemHFMidPhaseCG.cu:
// :148-150).  A PxHeightFieldSample is FOUR bytes -- PxI16 height, then the two
// PxBitAndByte material bytes (PxHeightFieldSample.h:64-105) -- so one sample is
// one uint and the whole array is addressed as uints.
//
//   bits  0..15 : height, SIGNED (H8)
//   bits 16..22 : materialIndex0, seven bits (H7)
//   bit  23     : the TESSELLATION flag, which is materialIndex0's high bit
//   bits 24..30 : materialIndex1, seven bits
//   bit  31     : materialIndex1's own high bit, unused by the format
// ---------------------------------------------------------------------------
uint hfSample(uint64_t samples, uint vertexIndex)
{
    return Ptr<uint>(samples)[vertexIndex];
}

float hfGetHeight(uint64_t samples, uint vertexIndex)
{
    // Sign-extend the low half: shift up then arithmetic-shift down.
    const int h = int(hfSample(samples, vertexIndex) << 16) >> 16;
    return float(h);
}

bool hfZerothVertexShared(uint64_t samples, uint vertexIndex)
{
    return (hfSample(samples, vertexIndex) & 0x800000u) != 0u;
}

// Both material bytes are PxBitAndByte, whose conversion strips the high bit, so
// both masks are seven bits wide (H7).
uint hfMaterialIndex0(uint64_t samples, uint vertexIndex)
{
    return (hfSample(samples, vertexIndex) >> 16) & 0x7Fu;
}
uint hfMaterialIndex1(uint64_t samples, uint vertexIndex)
{
    return (hfSample(samples, vertexIndex) >> 24) & 0x7Fu;
}

float3 hfGetVertex(uint64_t samples, uint vertexIndex, uint columns)
{
    const uint row = vertexIndex / columns;
    const uint col = vertexIndex % columns;
    return float3(float(row), hfGetHeight(samples, vertexIndex), float(col));
}

float3 hf2shapep(float3 v, float rowScale, float heightScale, float columnScale)
{
    return float3(v.x * rowScale, v.y * heightScale, v.z * columnScale);
}

// getTriangleVertexIndices (heightfieldUtil.cuh:115-165): the tessellation flag
// picks which diagonal splits the quad.
void hfTriVertexIndices(uint64_t samples, uint triangleIndex, uint columns,
                        out uint i0, out uint i1, out uint i2)
{
    const uint cell = triangleIndex >> 1;
    const bool firstTri = (triangleIndex & 0x1u) == 0u;
    if (hfZerothVertexShared(samples, cell))
    {
        if (firstTri) { i0 = cell + columns; i1 = cell; i2 = cell + columns + 1u; }
        else          { i0 = cell + 1u; i1 = cell + columns + 1u; i2 = cell; }
    }
    else
    {
        if (firstTri) { i0 = cell; i1 = cell + 1u; i2 = cell + columns; }
        else          { i0 = cell + columns + 1u; i1 = cell + columns; i2 = cell + 1u; }
    }
}

// getTriangle (heightfieldUtil.cuh:262-310).  The CUDA also builds a `handedness`
// vector there; it is assigned and never read (the swap uses `wrongHanded`), so
// it is dead code and the port drops it -- as psHfCollide already does.
void hfGetTriangle(uint64_t samples, uint triangleIndex, float3 scale, uint columns,
                   out float3 v0, out float3 v1, out float3 v2)
{
    const float rowScale = scale.x;
    const float heightScale = scale.y;
    const float columnScale = scale.z;
    bool wrongHanded = false;
    if (columnScale < 0.0f) wrongHanded = !wrongHanded;   // H6: dead upstream
    if (rowScale < 0.0f) wrongHanded = !wrongHanded;

    uint t0, t1, t2;
    hfTriVertexIndices(samples, triangleIndex, columns, t0, t1, t2);

    const float3 vertex0 = hfGetVertex(samples, t0, columns);
    const float3 vertex1 = hfGetVertex(samples, wrongHanded ? t2 : t1, columns);
    const float3 vertex2 = hfGetVertex(samples, wrongHanded ? t1 : t2, columns);

    v0 = hf2shapep(vertex0, rowScale, heightScale, columnScale);
    v1 = hf2shapep(vertex1, rowScale, heightScale, columnScale);
    v2 = hf2shapep(vertex2, rowScale, heightScale, columnScale);
}

// H9: the min helpers clamp to n-2, the max helpers to n-1.
uint hfGetMinRow(float x, uint rows)    { return uint(clamp(int(floor(x)), 0, int(rows) - 2)); }
uint hfGetMaxRow(float x, uint rows)    { return uint(clamp(int(ceil(x)), 0, int(rows) - 1)); }
uint hfGetMinColumn(float z, uint cols) { return uint(clamp(int(floor(z)), 0, int(cols) - 2)); }
uint hfGetMaxColumn(float z, uint cols) { return uint(clamp(int(ceil(z)), 0, int(cols) - 1)); }

// ---------------------------------------------------------------------------
// Geometry, lifted from psHfCollide.slang (Phase 1, psHfCollide_core).  These are
// the same bodies psMeshMidphaseBda carries; the corpus duplicates them because
// each BDA module needs Ptr<> where the Phase-1 module has bound buffers, and
// neither of the two existing copies lives in an importable module.
// ---------------------------------------------------------------------------
float getParticleSpeculativeContactVolume(out float3 cVolumePos,
    float3 currentPos, float3 predictedPos, float contactDist, bool isDiffuse, bool isTGS)
{
    const float sScaleInvPGS = 1.0f / PARTICLE_FORWARD_PROJECTION_STEP_SCALE_PGS;
    const float sScaleInvTGS = 1.0f / PARTICLE_FORWARD_PROJECTION_STEP_SCALE_TGS;
    const float sScaleInvDiffuse = 1.0f / PARTICLE_FORWARD_PROJECTION_STEP_SCALE_DIFFUSE;
    float scaleInv;
    if (isDiffuse)
        scaleInv = sScaleInvDiffuse;
    else
        scaleInv = isTGS ? sScaleInvTGS : sScaleInvPGS;
    const float3 cVolumeOffset = (predictedPos - currentPos) * scaleInv * 0.5f;
    cVolumePos = currentPos + cVolumeOffset;
    return length(cVolumeOffset) + contactDist;
}

float3 closestPtPointTriangle2(float3 p, float3 a, float3 b, float3 c, float3 ab, float3 ac)
{
    const float3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) { const float v = d1 / (d1 - d3); return a + v * ab; }
    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) { const float w = d2 / (d2 - d6); return a + w * ac; }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    { const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return b + w * (c - b); }
    const float denom = 1.0f / (va + vb + vc);
    return a + ab * (vb * denom) + ac * (vc * denom);
}

float distancePointTriangleSquared(float3 p, float3 a, float3 b, float3 c, out float3 closestP)
{
    const float3 ab = b - a;
    const float3 ac = c - a;
    closestP = closestPtPointTriangle2(p, a, b, c, ab, ac);
    const float3 v = p - closestP;
    return dot(v, v);
}

bool particleTriangleTest(inout float3 normal, inout float distance,
                          float3 particlePos, float3 cVolumePos, float cVolumeRadius,
                          float3 triV0, float3 triV1, float3 triV2, bool enableCCD)
{
    const float3 cn = cross(triV1 - triV0, triV2 - triV0);
    const float m = dot(cn, cn);
    const float3 triNormal = m > 0.0f ? cn * rsqrt(m) : float3(0, 0, 0);
    const float cVolumeRadiusSq = cVolumeRadius * cVolumeRadius;

    const float cVolumePosPlaneDist = dot(cVolumePos - triV0, triNormal);
    float3 closestPos;
    if (cVolumePosPlaneDist < cVolumeRadius)
    {
        float distSq = distancePointTriangleSquared(cVolumePos, triV0, triV1, triV2, closestPos);
        if (distSq <= cVolumeRadiusSq)
        {
            const float particlePosPlaneDist = dot(particlePos - triV0, triNormal);
            if (particlePosPlaneDist > 0.0f)
            {
                if (enableCCD)
                    distSq = distancePointTriangleSquared(particlePos, triV0, triV1, triV2, closestPos);
                const float3 distVec = particlePos - closestPos;
                normal = (distSq > 0.0f) ? distVec * rsqrt(distSq) : triNormal;
                distance = sqrt(distSq);
                return true;
            }
        }
    }
    return false;
}

// PxgParticleContactWriter::writeContact (PxgParticleSystem.h:615)
void writeContact(uint index, float4 normalPen, uint2 compressedParticleIndex, uint2 rigidId)
{
    if (index < gHf.w_maxContacts)
    {
        (Ptr<uint2>(gHf.w_contactByRigid))[index] = rigidId;
        (Ptr<uint>(gHf.w_tempContactByRigid))[index] = rigidId.x;
        (Ptr<uint>(gHf.w_contactIndexSortedByRigid))[index] = index;
        (Ptr<uint2>(gHf.w_contactSortedByParticle))[index] = compressedParticleIndex;
        (Ptr<uint>(gHf.w_tempContactByParticle))[index] = compressedParticleIndex.x;
        (Ptr<uint>(gHf.w_contactIndexSortedByParticle))[index] = index;
        PsContact ct;
        ct.normal_pen = normalPen;
        ct.rigidId = rigidId;
        ct.particleId = compressedParticleIndex;
        (Ptr<PsContact>(gHf.w_particleContacts))[index] = ct;
    }
}

// LoadShapePairWarp<ePARTICLESYSTEM, eHEIGHTFIELD> (particleSystemHFMidPhaseCG.cu:293).
// H4: the FIRST template argument is the PARTICLE SYSTEM here, the opposite of the
// mesh twin, so the "does slot 0 already hold what I want" test is on the PARTICLE
// type and not on the geometry type.
void loadParticleHeightfieldPair(uint cmIdx,
                                 out PxgShape psShape, out uint psCacheRef,
                                 out PxgShape hfShape, out uint hfCacheRef)
{
    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(gHf.cmInputs)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gHf.gpuShapes);
    uint sRefPs = cin.shapeRef0, cRefPs = cin.transformCacheRef0;
    uint sRefHf = cin.shapeRef1, cRefHf = cin.transformCacheRef1;
    if (shapes[cin.shapeRef0].type != GEOM_PARTICLESYSTEM)
    {
        sRefPs = cin.shapeRef1; cRefPs = cin.transformCacheRef1;
        sRefHf = cin.shapeRef0; cRefHf = cin.transformCacheRef0;
    }
    psShape = shapes[sRefPs]; psCacheRef = cRefPs;
    hfShape = shapes[sRefHf]; hfCacheRef = cRefHf;
}

// ===========================================================================
// ps_heightfieldCollisonLaunch -- the kernel.
// grid (4096, numTests, 2), block (64, 1, 1).
// ===========================================================================
[shader("compute")]
[numthreads(64, 1, 1)]
void ps_heightfieldCollisonLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint cmIdx = gid.y;              // blockIdx.y -- one contact pair per y-block
    const bool isDiffuse = (gid.z == 1u);  // H1
    // The CUDA bool occupies one byte at blob offset 0 with three bytes of padding
    // behind it; read as a masked word (the P8.97d arrangement).
    const bool isTGS = (gHf.isTGSByte & 0xFFu) != 0u;

    // ---- D2: the block-uniform scratch, computed per thread ----
    PxgShape psShape, hfShape;
    uint psCacheRef, hfCacheRef;
    loadParticleHeightfieldPair(cmIdx, psShape, psCacheRef, hfShape, hfCacheRef);

    // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.
    const uint64_t xf = gHf.transformCache + uint64_t(hfCacheRef) * 32UL;
    const float4 hfQ = Ptr<float4>(xf)[0];
    const float3 hfP = float3(Ptr<float>(xf + 16UL)[0],
                              Ptr<float>(xf + 20UL)[0],
                              Ptr<float>(xf + 24UL)[0]);

    const float contactDist = Ptr<float>(gHf.contactDistance)[psCacheRef] +
                              Ptr<float>(gHf.contactDistance)[hfCacheRef];
    const float restDist = Ptr<float>(gHf.restDistances)[cmIdx];
    const uint2 rigidId = Ptr<uint2>(gHf.shapeToRigidRemapTable)[hfCacheRef];

    const uint psId = psShape.particleOrSoftbodyId;
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gHf.particleSystems);

    const bool enableCCD = (systems[psId].mData_Flags & PX_PARTICLE_FLAG_SPECULATIVE_CCD) > 0u;

    // The heightfield scale lives in PxgShape's PxMeshScale::scale -- x is the row
    // scale, y the height scale, z the column scale.
    const float3 hfScale = float3(hfShape.scaleX, hfShape.scaleY, hfShape.scaleZ);

    // ---- the heightfield sample block ----
    const uint64_t hfPtr = uint64_t(hfShape.hullOrMeshOffset.x) |
                           (uint64_t(hfShape.hullOrMeshOffset.y) << 32);
    const uint nbRows = Ptr<uint>(hfPtr)[0];
    const uint nbCols = Ptr<uint>(hfPtr)[1];
    const uint64_t samples = hfPtr + 8UL;

    // ---- H2 / D1: the diffuse-or-solid selection ----
    uint64_t curPosAddr, predPosAddr;
    uint64_t oneWayInfosAddr, oneWayNodeIdxAddr, oneWayCountsAddr;
    uint numParticles;
    if (isDiffuse)
    {
        curPosAddr = systems[psId].mDiffuseSortedOriginPos_LifeTime;
        predPosAddr = systems[psId].mDiffuseSortedPos_LifeTime;
        // D1: an `if`, not the CUDA's ternary -- mNumDiffuseParticles is NULL
        // whenever the system carries no diffuse particles, and a guarded uniform
        // BDA load really does execute on llvmpipe (P8.96c).
        if (systems[psId].mCommonData_MaxDiffuseParticles > 0u)
            numParticles = Ptr<uint>(systems[psId].mNumDiffuseParticles)[0];
        else
            numParticles = 0u;
        oneWayInfosAddr = systems[psId].mDiffuseOneWayContactInfos;
        oneWayNodeIdxAddr = systems[psId].mDiffuseOneWayNodeIndex;
        oneWayCountsAddr = systems[psId].mDiffuseOneWayContactCount;
    }
    else
    {
        curPosAddr = systems[psId].mSortedOriginPos_InvMass;
        predPosAddr = systems[psId].mSortedPositions_InvMass;
        numParticles = systems[psId].mCommonData_NumParticles;
        oneWayInfosAddr = systems[psId].mOneWayContactInfos;
        oneWayNodeIdxAddr = systems[psId].mOneWayNodeIndex;
        oneWayCountsAddr = systems[psId].mOneWayContactCount;
    }

    // H2: the guard is on the COUNT, not on mMaxDiffuseParticles.  Block-uniform,
    // so it cannot strand a thread anywhere.
    if (isDiffuse && numParticles == 0u)
        return;

    // ---- the grid-stride loop, one particle per thread ----
    // The CUDA's stride is `gridDim.x * blockDim.x` (:346), computed at runtime.
    // WorkgroupCount() gives the real dispatch width, so the port keeps that
    // property instead of baking in PS_HEIGHTFIELD_COLLISION -- which matters
    // because the fixture dispatches narrower grids on purpose, and because a
    // baked constant would silently skip particles there.  This is why the module
    // is absent from PXG_HLSL_ENTRIES: WorkgroupCount() has no cs_6_0 lowering,
    // the same reason particleContactSortBda is excluded.
    const uint stride = WorkgroupCount().x * HF_COLLISION_BLOCK;
    for (uint particleIndex = gid.x * HF_COLLISION_BLOCK + gtid.x;
         particleIndex < numParticles;
         particleIndex += stride)
    {
        const float3 currentPos = Ptr<float4>(curPosAddr)[particleIndex].xyz;
        float3 cVolumePos;
        float cVolumeRadius;
        if (enableCCD)
        {
            const float3 predictedPos = Ptr<float4>(predPosAddr)[particleIndex].xyz;
            cVolumeRadius = getParticleSpeculativeContactVolume(cVolumePos, currentPos,
                                predictedPos, contactDist, isDiffuse, isTGS);
        }
        else
        {
            cVolumePos = currentPos;
            cVolumeRadius = contactDist;
        }

        // PxBounds3::transformFast(inverse(pose), worldBound), in centre/extent
        // form: the rotated extent is |R^T| applied to the axis extents.
        const float3 wmin = cVolumePos - float3(cVolumeRadius, cVolumeRadius, cVolumeRadius);
        const float3 wmax = cVolumePos + float3(cVolumeRadius, cVolumeRadius, cVolumeRadius);
        const float3 wc = (wmin + wmax) * 0.5f;
        const float3 we = (wmax - wmin) * 0.5f;
        const float3 lc = quatRotateInv(hfQ, wc - hfP);
        const float3 col0 = quatRotateInv(hfQ, float3(1, 0, 0));
        const float3 col1 = quatRotateInv(hfQ, float3(0, 1, 0));
        const float3 col2 = quatRotateInv(hfQ, float3(0, 0, 1));
        const float3 le = abs(col0) * we.x + abs(col1) * we.y + abs(col2) * we.z;
        float3 lmin = lc - le;
        float3 lmax = lc + le;

        // H5: the height scale is NOT wrapped in abs, the other two are.
        const float oneOverHeightScale = 1.0f / hfScale.y;
        const float oneOverRowScale = 1.0f / abs(hfScale.x);
        const float oneOverColScale = 1.0f / abs(hfScale.z);

        lmin = float3(lmin.x * oneOverRowScale, lmin.y * oneOverHeightScale, lmin.z * oneOverColScale);
        lmax = float3(lmax.x * oneOverRowScale, lmax.y * oneOverHeightScale, lmax.z * oneOverColScale);

        // H6: dead upstream, transcribed anyway.
        if (hfScale.x < 0.0f) { const float t = lmin.x; lmin.x = lmax.x; lmax.x = t; }
        if (hfScale.z < 0.0f) { const float t = lmin.z; lmin.z = lmax.z; lmax.z = t; }

        if ((lmin.x > float(nbRows - 1u)) || (lmin.z > float(nbCols - 1u))
            || (lmax.x < 0.0f) || (lmax.z < 0.0f))
            continue;   // the CUDA `return`s; the grid-stride loop makes it a continue

        const uint minRow = hfGetMinRow(lmin.x, nbRows);
        const uint maxRow = hfGetMaxRow(lmax.x, nbRows);
        const uint minColumn = hfGetMinColumn(lmin.z, nbCols);
        const uint maxColumn = hfGetMaxColumn(lmax.z, nbCols);

        // D4: the underflow guard.  The CUDA's own zero-span test MULTIPLIES, so
        // an underflowed span sails through it -- see candidate 22.
        if (maxRow < minRow || maxColumn < minColumn)
            continue;

        if ((2u * (maxColumn - minColumn) * (maxRow - minRow)) == 0u)
            continue;

        const float3 cVolumePosShapeSpace = quatRotateInv(hfQ, cVolumePos - hfP);
        const float3 particlePosShapeSpace = quatRotateInv(hfQ, currentPos - hfP);

        const float miny = lmin.y;
        const float maxy = lmax.y;
        const uint columnSpan = maxColumn - minColumn;

        // D3: the CUDA spells this shared, indexed by the owning thread alone.
        float4 tempContacts[MAX_MESH_CONTACTS];
        uint contactCounts = 0u;

        const uint totalNumProcessed = (maxRow - minRow) * columnSpan * 2u;   // H10
        for (uint i = 0u; i < totalNumProcessed; ++i)
        {
            const uint index = i / 2u;
            const uint vertexIndex = (minRow + index / columnSpan) * nbCols
                                   + (minColumn + index % columnSpan);
            const float h0 = hfGetHeight(samples, vertexIndex);
            const float h1 = hfGetHeight(samples, vertexIndex + 1u);
            const float h2 = hfGetHeight(samples, vertexIndex + nbCols);
            const float h3 = hfGetHeight(samples, vertexIndex + nbCols + 1u);
            const bool con0 = maxy < h0 && maxy < h1 && maxy < h2 && maxy < h3;
            const bool con1 = miny > h0 && miny > h1 && miny > h2 && miny > h3;

            if (!(con0 || con1))
            {
                const bool isMaterial1 = (i & 1u) != 0u;
                const uint material = isMaterial1 ? hfMaterialIndex1(samples, vertexIndex)
                                                  : hfMaterialIndex0(samples, vertexIndex);
                if (material != HOLE)   // H7
                {
                    const uint triangleIdx = isMaterial1 ? ((vertexIndex << 1) + 1u)
                                                         : (vertexIndex << 1);

                    float3 triV0, triV1, triV2;
                    hfGetTriangle(samples, triangleIdx, hfScale, nbCols, triV0, triV1, triV2);

                    float3 normalShapeSpace = float3(0, 0, 0);
                    float distance = 0.0f;
                    const bool intersect = particleTriangleTest(normalShapeSpace, distance,
                        particlePosShapeSpace, cVolumePosShapeSpace, cVolumeRadius,
                        triV0, triV1, triV2, enableCCD);

                    // H11: NO worst-contact replacement.  Anything past six is
                    // dropped, and the CUDA says so in a comment of its own.
                    if (intersect && contactCounts < MAX_MESH_CONTACTS)
                    {
                        const float3 normal = quatRotate(hfQ, normalShapeSpace);
                        tempContacts[contactCounts++] = float4(normal, distance - restDist);
                    }
                }
            }
        }

        // Already capped by the guard above; the CUDA re-clamps here and so does
        // this port, faithfully.  A mutant that deletes this line is an EQUIVALENCE
        // and is predicted as one rather than discovered as a survivor.
        contactCounts = min(contactCounts, MAX_MESH_CONTACTS);

        uint contactWritten = 0u;
        if (nodeIndexIsStaticBody(rigidId))
        {
            if (contactCounts != 0u)
            {
                uint contactIndex;
                InterlockedAdd(Ptr<uint>(oneWayCountsAddr)[particleIndex], contactCounts, contactIndex);
                const uint endIndex = min(contactIndex + contactCounts, MAX_PARTICLE_CONTACTS);
                for (uint k = contactIndex, src = 0u; k < endIndex; ++k, ++src)
                {
                    // H15: COLUMN-major, stride numParticles.
                    const uint outputIndex = particleIndex + k * numParticles;
                    Ptr<float4>(oneWayInfosAddr)[outputIndex] = tempContacts[src];
                    Ptr<uint2>(oneWayNodeIdxAddr)[outputIndex] = rigidId;
                }
            }
            contactWritten = min(contactCounts, MAX_PARTICLE_CONTACTS);
            // CANDIDATE 20: a PLAIN STORE over the atomicAdd.  Transcribed.
            Ptr<uint>(oneWayCountsAddr)[particleIndex] = contactWritten;
        }

        // H13: zero for a static rigid, the whole list for a dynamic one.
        const uint remaining = contactCounts - contactWritten;
        // H14: unlike the mesh twin, the RESERVATION is inside the !isDiffuse
        // guard, so the diffuse pass burns no writer slots here.
        if (remaining > 0u && !isDiffuse)
        {
            const uint2 compressedParticleIndex = uint2(particleIndex, psId);
            uint contactStartIndex;
            InterlockedAdd(Ptr<uint>(gHf.w_numTotalContacts)[0], remaining, contactStartIndex);
            const uint contactEndIndex = contactStartIndex + remaining;
            for (uint k = contactStartIndex, src = 0u; k < contactEndIndex; ++k, ++src)
                writeContact(k, tempContacts[src], compressedParticleIndex, rigidId);
        }
    }
}
