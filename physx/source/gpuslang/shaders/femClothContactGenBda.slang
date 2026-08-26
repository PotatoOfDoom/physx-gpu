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

// femClothContactGenBda.slang
//
// P7.1 tranche 4a — cloth-vertex-vs-PRIMITIVE contact GENERATION (the first
// tranche of the cloth-drape-on-a-floor arc) on the Vulkan engine ABI (BDA).
// Ported line-for-line from gpunarrowphase/src/CUDA:
//   - cloth_midphaseGenerateVertexPairsLaunch (femClothMidPhase.cu, launch
//     PxgNarrowphaseCore.cpp:6317, CLOTH_MIDPHASE_VERTEX_PRIMS): the broadphase->
//     midphase pass that emits cloth-vertex <-> primitive-shape candidate pairs.
//     Each cloth vertex is tested against the rigid shape's world AABB
//     (bounds[rigidCacheRef]); a hit appends uint4(cmIdx, vertId, 0, 0) to the
//     intermediate stack and bumps midphasePairsNum.
//   - cloth_planeVertContactGenLaunch (femClothPrimitives.cu:853, launch :5853,
//     CLOTH_PLANE_VERTEX_CG): cloth-vertex-vs-PLANE contact generation (a cloth
//     falling onto a floor) — vertPlaneCollide + the PxgFEMContactWriter append.
//   - cloth_SphereVertexContactGenLaunch (femClothPrimitives.cu:700, launch :5776,
//     CLOTH_VERT_SPHERE_CG): cloth-vertex-vs-SPHERE contact generation (a drape
//     over a sphere) — spheresphere + the same writer append.  (The kernel's
//     eCAPSULE branch is DEFERRED — see DEVIATIONS.)
//
// Doctrine (identical to the prior cloth BDA tranches femClothSolveBda /
// femClothRigidAttachBda and the particle contact-gen precedent psContactGenBda):
//   - the PxgFEMCloth descriptor arrives as a device pointer read through Ptr<>
//     (the byte-exact pxgSlangFEMClothDesc mirror); mPosition_InvMass @16 and
//     mNbVerts @316 are the only fields the vertex kernels touch;
//   - the per-pair narrowphase inputs (PxgContactManagerInput / PxgShape /
//     PxsCachedTransform / PxgRigidFilterPair / PxNodeIndex remap) are transcribed
//     locally as byte-exact mirrors (pinned by static_asserts against the real
//     engine structs in testFemClothSim.cpp), matching the psContactGenBda
//     precedent (keeps the rigid-common module out of the cloth-descriptor TU);
//   - the PxBounds3 world AABB (24B: min @0, max @12) is read straight from the
//     bounds array by float offset (rigidCacheRef*6);
//   - the by-value PxgFEMContactWriter (PxgFEMCore.h:361) is inlined field-by-field
//     into the contact-gen push blob and its device pointers are written directly
//     (writeRigidVsDeformableContact -> writeContactCore + the FEM-rigid contact
//     info tail), exactly as psContactGenBda inlined PxgParticleContactWriter.
//
// DEVIATIONS from CUDA (recipe policy; each preserves the emitted contact SET and
// the running contact/pair total, and is exercised by testFemClothSim.cpp):
//   - NO 32-lane warp intrinsics.  The CUDA __ballot_sync / __shfl_sync / lane-0
//     atomicAdd contact-compaction is a pure SCHEDULING optimisation — the SET of
//     emitted contacts and the running total are identical whether or not work is
//     compacted, and the six contact-buffer arrays are radix-sorted downstream, so
//     the atomicAdd-order-dependent buffer INDEX carries no observable meaning.
//     It is lowered to a per-32-lane-LOGICAL-tile groupshared aggregation
//     (InterlockedOr ballot + countbits prefix + ONE lane-0 InterlockedAdd) with
//     full-block barriers at block-UNIFORM points (the loop bound numRequiredThreads
//     / nbVerts is block-uniform, so every barrier is reached uniformly) — the
//     exact llvmpipe-safe pattern of psContactGenBda;
//   - LoadShapePair(Warp)'s warp-cooperative shape/transform/cmInput reads are done
//     as direct per-thread Ptr<> loads (semantically identical read-fanout; same as
//     psContactGenBda);
//   - the midphase grid.x extent (numBlocks = 32, PxgNarrowphaseCore.cpp:6335) is a
//     compile-time constant MIDPHASE_GRID_X so the grid-stride matches the launch
//     (psContactGenBda hardcodes PS_GRID the same way — there is no gridDim built-in
//     in a compute shader); the dispatch MUST use grid (32, numTests, 1);
//   - cloth_SphereVertexContactGenLaunch's eCAPSULE branch (vertCapsuleCollide,
//     femClothPrimitives.cu:462) is DEFERRED — this tranche's scope is the sphere
//     drape.  For a non-sphere rigid the else-path produces no contact (the CUDA
//     asserts eCAPSULE there; a capsule input is out of scope and never launched in
//     the sphere test).  Same deferral shape as psContactGenBda's eCONVEXCORE;
//   - PxgFemOtherContactInfo::setRigidMaterialIndex(mat) followed by
//     markInCollision(false) (PxgFEMCore.h:148/:152) is emitted as a single store of
//     (mat & 0x7fffffff): setRigidMaterialIndex writes (old & 0x80000000) | (mat &
//     0x7fffffff) and the subsequent markInCollision(false) clears bit 31, so the
//     net 32-bit word is provably (mat & 0x7fffffff) regardless of the prior buffer
//     content — bit-identical, no read-modify-write needed;
//   - the mesh-vertex-vs-mesh / capsule / box sibling kernels in the same .cu files
//     are out of scope (this tranche is plane + sphere, the floor/sphere drape).
//
// All float arithmetic is NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the
// midphase (closest-v).magnitudeSquared() feeds the compared `<= tolerance`, the
// plane transformInv feeds the compared `contactDist >= x` AND the stored
// separation, and spheresphere's delta.dot feeds the compared `inflatedSum^2 >
// distanceSq` AND the stored penetration — any of which would fuse to an FMA
// otherwise, breaking bitwise parity with the -ffp-contract=off CPU reference.

import pxgSlangFEMClothDesc;   // PxgFEMClothGpu (mPosition_InvMass @16, mNbVerts @316)
import pxgSlangSphereBox;      // P9.15a: spherebox, lifted out of fcStaticTriCloth
import pxgSlangBv32;           // P9.15b: the BV32 node reader (P8.25), shared
import pxgSlangConvexTriCloth; // P9.15c: the lifted convex-vs-triangle core

// PxGeometryType (PxGeometry.h): eSPHERE 0, ePLANE 1, eCAPSULE 2, ... eTRIANGLEMESH 8.
static const uint GEOM_SPHERE = 0u;
static const uint GEOM_PLANE = 1u;
static const uint GEOM_CAPSULE = 2u;
static const uint GEOM_TRIANGLEMESH = 8u;

// PXG_BITSHIFT_ELEMENT_ID (PxgSimulationCoreDesc.h:41) — the cloth-id / element
// packing used by PxEncodeClothIndex(clothId, elem) = (clothId << 20) | elem.
static const uint PXG_CLOTH_ID_SHIFT = 20u;

// Launch geometry.
//   midphase: grid (32, numTests, 1), block (WARP_SIZE=32, MIDPHASE_WARPS=2, 1).
//   plane:    grid (numTests, 1, 1),  block (32, 16, 1) = 512.
//   sphere:   grid (numTests, 1, 1),  block (32,  1, 1) = 32.
static const uint WARP = 32u;
static const uint MIDPHASE_WARPS = 2u;
static const uint MIDPHASE_BLOCK = WARP * MIDPHASE_WARPS;      // 64
static const uint MIDPHASE_GRID_X = 32u;
static const uint PLANE_WARPS = 16u;
static const uint PLANE_BLOCK = WARP * PLANE_WARPS;            // 512
static const uint SPHERE_WARPS = 1u;
static const uint SPHERE_BLOCK = WARP * SPHERE_WARPS;          // 32
static const uint BOX_WARPS = 16u;                             // (32, 16, 1)
static const uint BOX_BLOCK = WARP * BOX_WARPS;                // 512

// ---------------------------------------------------------------------------
// Local element-struct mirrors (byte-exact vs the engine C++ layouts; pinned by
// static_asserts in testFemClothSim.cpp).  Same transcription as psContactGenBda.
// ---------------------------------------------------------------------------
// PxgContactManagerInput (PxgContactManager.h, 16 bytes)
struct PxgContactManagerInput
{
    uint shapeRef0;          // @0
    uint shapeRef1;          // @4
    uint transformCacheRef0; // @8
    uint transformCacheRef1; // @12
}

// PxgShape (PxgConvexConvexShape.h, 48 bytes)
struct PxgShape
{
    float scaleX, scaleY, scaleZ;                     // @0
    float scaleRotX, scaleRotY, scaleRotZ, scaleRotW; // @12
    uint materialIndex;                               // @28
    uint2 hullOrMeshOffset;                           // @32
    uint type;                                        // @40
    uint particleOrSoftbodyId;                        // @44
}                                                     // = 48

// PxsCachedTransform (PxsTransformCache.h, 32 bytes): PxTransform (q@0, p@16) + flags.
struct PxsCachedTransform
{
    float qx, qy, qz, qw; // @0  PxQuat
    float px, py, pz;     // @16 PxVec3
    uint flags;           // @28 -> 32
}

// PxgRigidFilterPair (PxgSimulationCoreDesc.h, 32 bytes): index0 u64, index1 u64,
// index2 u32, padding.
struct PairR { uint2 index0; uint2 index1; uint index2; uint pad0, pad1, pad2; }

// ---------------------------------------------------------------------------
// Push-constant blobs — BYTE-IDENTICAL to each CUDA PxCudaKernelParam pack.
// ---------------------------------------------------------------------------
// cloth_midphaseGenerateVertexPairsLaunch (femClothMidPhase.cu; launch pack
// PxgNarrowphaseCore.cpp:6318): { PxU32 numTests, PxReal toleranceLength,
// cmInputs*, transformCache*, bounds*, contactDistance*, gpuShapes*, femClothes*,
// stackPtr(uint4*), midphasePairsNum(PxU32*) } -> 72-byte blob.
// (ABI row {4,4,8,8,8,8,8,8,8,8}.)
struct MidphaseParams
{
    uint numTests;              // @0
    float toleranceLength;      // @4  (unused; ABI parity)
    uint64_t cmInputs;          // @8
    uint64_t transformCache;    // @16 (unused; ABI parity — the vertex midphase
                                //      reads the rigid AABB straight from bounds)
    uint64_t bounds;            // @24
    uint64_t contactDistance;   // @32
    uint64_t gpuShapes;         // @40
    uint64_t femClothes;        // @48
    uint64_t stackPtr;          // @56 output (uint4*)
    uint64_t midphasePairsNum;  // @64 output (PxU32*) -> 72
}

// cloth_planeVertContactGenLaunch / cloth_SphereVertexContactGenLaunch share the
// arg pack (femClothPrimitives.cu; launch PxgNarrowphaseCore.cpp:5853 / :5776):
// { PxU32 numTests, cmInputs*, transformCache*, contactDistance*, restDistance*,
// gpuShapes*, femClothes*, shapeToRigidRemapTable*, filterPairs*, PxU32 nbFilterPairs,
// PxgFEMContactWriter writer(BY VALUE, 80B) } -> 160-byte blob.
// (ABI row {4,8,8,8,8,8,8,8,8,4,80}.)
struct ContactGenParams
{
    uint numTests;                 // @0
    uint _pad4;                    // @4
    uint64_t cmInputs;             // @8
    uint64_t transformCache;       // @16
    uint64_t contactDistance;      // @24
    uint64_t restDistance;         // @32
    uint64_t gpuShapes;            // @40
    uint64_t femClothes;           // @48
    uint64_t shapeToRigidRemap;    // @56  PxNodeIndex* (PxU64 per shape cache ref)
    uint64_t filterPairs;          // @64  PxgRigidFilterPair*
    uint nbFilterPairs;            // @72
    uint _pad76;                   // @76
    // ---- PxgFEMContactWriter (by value, 80B) @80 ----
    uint64_t w_outPoint;            // @80  float4*
    uint64_t w_outNormalPen;        // @88  float4*
    uint64_t w_outBarycentric;      // @96  float4*
    uint64_t w_outContactInfo;      // @104 PxgFemOtherContactInfo* (16B stride)
    uint64_t w_totalContactCount;   // @112 PxU32*
    uint64_t w_contactByX;          // @120 PxU64* (contactByRigid)
    uint64_t w_tempContactByX;      // @128 PxU32*
    uint64_t w_contactIndexSortedByX; // @136 PxU32*
    uint64_t w_contactSortedByX;    // @144 PxU32* (unused by the vertex writer; ABI parity)
    uint w_maxNumContacts;          // @152 PxU32
    uint _pad156;                   // @156 -> 160
}

// cloth_midphaseGeneratePairsLaunch (femClothMidPhase.cu:191; launch pack
// PxgNarrowphaseCore.cpp:1125): { PxU32 numTests, PxReal toleranceLength, cmInputs*,
// transformCache*, bounds*, contactDistance*, gpuShapes*, femClothes*, stackPtr(PxU8*),
// midphasePairsNum(PxU32*), PxU32 stackSizeBytes } -> 80-byte blob.
// (ABI row {4,4,8,8,8,8,8,8,8,8,4}.)
//
// NOTE the difference from the SOFTBODY twin's row, kSbMidphasePrimitives
// { 4,4,8,8,8,8,8,8,4,8,8 }: there stackSizeBytes sits BEFORE the two output
// pointers, here it comes LAST.  Same eleven arguments, same types, different
// order -- and reusing the softbody row would put every pointer past it at the
// wrong offset.
struct ClothTriMidphaseParams
{
    uint numTests;              // @0
    float toleranceLength;      // @4  (unused; ABI parity)
    uint64_t cmInputs;          // @8
    uint64_t transformCache;    // @16 (unused; the AABB comes from bounds)
    uint64_t bounds;            // @24
    uint64_t contactDistance;   // @32
    uint64_t gpuShapes;         // @40
    uint64_t femClothes;        // @48
    uint64_t stackPtr;          // @56 output (uint4*)
    uint64_t midphasePairsNum;  // @64 output (PxU32*)
    uint stackSizeBytes;        // @72
    uint _pad76;                // @76 -> 80
}

[[vk::push_constant]] ConstantBuffer<MidphaseParams> gMid;
[[vk::push_constant]] ConstantBuffer<ContactGenParams> gCG;
// cloth_boxTriangleContactGenLaunch (femClothPrimitives.cu:3357; launch pack
// PxgNarrowphaseCore.cpp:5995): fifteen arguments -> 184-byte blob.
// (ABI row {4,8,8,8,8,8,8,8,8,8,8,4,4,8,80}.)
struct ClothBoxTriParams
{
    float toleranceLength;         // @0  (unused; ABI parity)
    uint _pad4;                    // @4
    uint64_t cmInputs;             // @8
    uint64_t transformCache;       // @16
    uint64_t contactDistance;      // @24
    uint64_t restDistance;         // @32
    uint64_t gpuShapes;            // @40
    uint64_t femClothes;           // @48
    uint64_t shapeToRigidRemap;    // @56
    uint64_t stackPtr;             // @64  the midphase's (cmIdx, triIdx) pairs
    uint64_t midphasePairsNum;     // @72
    uint64_t filterPairs;          // @80  clothAttachmentPairs
    uint nbFilterPairs;            // @88
    uint stackSizeBytes;           // @92
    uint64_t stackSizeNeeded;      // @96  atomicMax output
    uint64_t _pad104;              // @104
    // ---- PxgFEMContactWriter (by value, 80B) @112 ----
    //
    // @112, NOT @104, and that cost this tranche a segfault.  The bridge places
    // each argument at paramAlign(size) (PxgVulkanCudaContext.cpp:93), and
    // paramAlign(80) is **16**, not 8 -- 80 is a multiple of 16.  The arguments
    // before the writer end at 104, which is 8-aligned but not 16-aligned, so the
    // bridge pads to 112.  A struct that puts the writer at 104 reads the padding
    // as its first pointer and stores through it: `segfault at 0`.
    //
    // The plane/sphere siblings above hide the rule by luck: their writer lands at
    // 80, which is already 16-aligned, so 8- and 16-alignment agree there.  Any
    // future kernel whose by-value tail is a multiple of 16 must check this.
    uint64_t b_outPoint;           // @112
    uint64_t b_outNormalPen;       // @120
    uint64_t b_outBarycentric;     // @128
    uint64_t b_outContactInfo;     // @136
    uint64_t b_totalContactCount;  // @144
    uint64_t b_contactByX;         // @152
    uint64_t b_tempContactByX;     // @160
    uint64_t b_contactIndexSortedByX; // @168
    uint64_t b_contactSortedByX;   // @176 (unused by this writer; ABI parity)
    uint b_maxNumContacts;         // @184
    uint _pad188;                  // @188 -> 192
}

[[vk::push_constant]] ConstantBuffer<ClothTriMidphaseParams> gTMid;
[[vk::push_constant]] ConstantBuffer<ClothBoxTriParams> gBTri;

// ---------------------------------------------------------------------------
// PxQuat helpers (PxQuat.h, transcribed to keep the rigid-common module out of
// this TU; identical to psContactGenBda / femClothRigidAttachBda).
// ---------------------------------------------------------------------------
float3 quatRotateInv(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3((vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * dot2),
                  (vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * dot2),
                  (vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * dot2));
}
float3 getBasisVector0(float4 q)
{
    const float x2 = q.x * 2.0f;
    const float w2 = q.w * 2.0f;
    return float3((q.w * w2) - 1.0f + q.x * x2,
                  (q.z * w2) + q.y * x2,
                  (-q.y * w2) + q.z * x2);
}

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// PxQuat multiply / conjugate / rotate (PxQuat.h), transcribed here for the same
// reason quatRotateInv above is: importing pxgSlangCommon would put a second
// quatRotate into this translation unit.  Spelling copied operation for operation.
float4 ctQuatMul(float4 a, float4 b)
{
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
        a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}
float4 ctQuatConj(float4 q) { return float4(-q.x, -q.y, -q.z, q.w); }
float3 ctQuatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3((vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2),
                  (vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2),
                  (vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2));
}

// ---------------------------------------------------------------------------
// PxgRigidFilterPair binary-search find (deformableElementFilter.cuh:67/:89) with
// the ignoreIndex2 early accept — transcribed from psContactGenBda::findRigid.  A
// cloth vertex's compressed id folds into q1 = uint2(clothMask, 0).
// ---------------------------------------------------------------------------
bool u64Lt(uint2 a, uint2 b) { return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x); }
bool u64Eq(uint2 a, uint2 b) { return a.x == b.x && a.y == b.y; }

bool findRigid(uint64_t pairsAddr, uint nbPairs, uint2 q0, uint2 q1)
{
    Ptr<PairR> pairs = Ptr<PairR>(pairsAddr);
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
        else if (0u < p.index2) cmp = -1;   // q.index2 == 0
        if (cmp == 0 || (u64Eq(q0, p.index0) && u64Eq(q1, p.index1)))
            return true;
        if (cmp < 0) r = mid; else l = mid + 1u;
    }
    return false;
}

// ---------------------------------------------------------------------------
// LoadShapePair<eTRIANGLEMESH> (dataReadWriteHelper.cuh:292): the cloth is the
// eTRIANGLEMESH shape; the other shape is the rigid.  Returns via out-params.
// ---------------------------------------------------------------------------
void loadClothRigidPair(uint64_t cmInputsAddr, uint64_t shapesAddr, uint cmIdx,
                        out PxgShape clothShape, out uint clothCacheRef,
                        out PxgShape rigidShape, out uint rigidCacheRef)
{
    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(cmInputsAddr)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(shapesAddr);
    uint sRefCloth = cin.shapeRef0, cRefCloth = cin.transformCacheRef0;
    uint sRefRigid = cin.shapeRef1, cRefRigid = cin.transformCacheRef1;
    if (shapes[cin.shapeRef0].type != GEOM_TRIANGLEMESH)
    {
        sRefCloth = cin.shapeRef1; cRefCloth = cin.transformCacheRef1;
        sRefRigid = cin.shapeRef0; cRefRigid = cin.transformCacheRef0;
    }
    clothShape = shapes[sRefCloth]; clothCacheRef = cRefCloth;
    rigidShape = shapes[sRefRigid]; rigidCacheRef = cRefRigid;
}

// ===========================================================================
// cloth_midphaseGenerateVertexPairsLaunch (femClothMidPhase.cu)
// ===========================================================================
groupshared uint sMidMask[MIDPHASE_WARPS];
groupshared uint sMidStart[MIDPHASE_WARPS];

[shader("compute")]
[numthreads(64, 1, 1)] // (WARP_SIZE, MIDPHASE_WARPS, 1) flattened
void cloth_midphaseGenerateVertexPairsLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint warpId = tid >> 5;       // threadIdx.y
    const uint lane = tid & (WARP - 1u); // threadIdx.x
    const uint cmIdx = gid.y;           // blockIdx.y

    if (cmIdx >= gMid.numTests)
        return;

    PxgShape clothShape, rigidShape;
    uint clothCacheRef, rigidCacheRef;
    loadClothRigidPair(gMid.cmInputs, gMid.gpuShapes, cmIdx,
                       clothShape, clothCacheRef, rigidShape, rigidCacheRef);

    const uint femClothId = clothShape.particleOrSoftbodyId;
    Ptr<PxgFEMClothGpu> femClothes = Ptr<PxgFEMClothGpu>(gMid.femClothes);
    const uint64_t meshVertsAddr = femClothes[femClothId].mPosition_InvMass;
    const uint nbVerts = femClothes[femClothId].mNbVerts;

    // rigid world AABB = bounds[rigidCacheRef] (24B: min @0, max @12); NOT fattened
    // (the vertex midphase leaves aabb.fattenFast commented out).
    Ptr<float> bounds = Ptr<float>(gMid.bounds);
    const uint rb = rigidCacheRef * 6u;
    const float3 aabbMin = float3(bounds[rb + 0u], bounds[rb + 1u], bounds[rb + 2u]);
    const float3 aabbMax = float3(bounds[rb + 3u], bounds[rb + 4u], bounds[rb + 5u]);

    Ptr<float> contactDistance = Ptr<float>(gMid.contactDistance);
    const float contactDist = contactDistance[clothCacheRef] + contactDistance[rigidCacheRef];
    const float tolerance = contactDist * contactDist;

    Ptr<float4> meshVerts = Ptr<float4>(meshVertsAddr);

    const uint idx = tid + gid.x * MIDPHASE_BLOCK;                 // threadIdx.* + blockIdx.x*blockDim
    const uint stride = MIDPHASE_BLOCK * MIDPHASE_GRID_X;          // blockDim.x*blockDim.y*gridDim.x

    for (uint i = 0u; i < nbVerts; i += stride)
    {
        const uint id = idx + i;
        bool hasHit = false;
        if (id < nbVerts)
        {
            const float4 vf = meshVerts[id];
            const float3 v = float3(vf.x, vf.y, vf.z);
            // aabb.closestPoint(v) = clamp(v, min, max) = min.maximum(max.minimum(v))
            const float3 closest = max(aabbMin, min(aabbMax, v));
            const float3 d = closest - v;
            hasHit = (d.x * d.x + d.y * d.y + d.z * d.z) <= tolerance;
        }

        // per-32-lane-tile ballot -> lane-0 atomicAdd -> exclusive-prefix scatter.
        if (lane == 0u) sMidMask[warpId] = 0u;
        GroupMemoryBarrierWithGroupSync();
        if (hasHit) InterlockedOr(sMidMask[warpId], 1u << lane);
        GroupMemoryBarrierWithGroupSync();
        const uint hitMask = sMidMask[warpId];
        const uint validCount = countbits(hitMask);
        if (lane == 0u)
        {
            uint s = 0u;
            if (validCount != 0u)
                InterlockedAdd(Ptr<uint>(gMid.midphasePairsNum)[0], validCount, s);
            sMidStart[warpId] = s;
        }
        GroupMemoryBarrierWithGroupSync();
        const uint prevStackShift = sMidStart[warpId];
        if (hasHit)
        {
            const uint offset = countbits(hitMask & ((1u << lane) - 1u));
            Ptr<uint4>(gMid.stackPtr)[prevStackShift + offset] = uint4(cmIdx, id, 0u, 0u);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// ---------------------------------------------------------------------------
// PxgFEMContactWriter::writeRigidVsDeformableContact (PxgFEMCore.h:485) -> the
// capacity-gated writeContactCore staging + the FEM-rigid contact-info tail.
// pairInd0 == rigidId.getInd() (the plane/sphere callers pass them equal).
// ---------------------------------------------------------------------------
void writeRigidVsDeformableContact(uint index, float4 contact, float4 normalPen, float4 barycentric,
                                   uint2 rigidId, uint pairInd1, uint rigidMaterialIndex)
{
    if (index >= gCG.w_maxNumContacts)
        return;

    // writeContactCore
    Ptr<uint2>(gCG.w_contactByX)[index] = rigidId;                 // contactByX[index] = rigidId (PxU64)
    Ptr<uint>(gCG.w_tempContactByX)[index] = rigidId.x;            // PxU32(rigidId & 0xffffffff)
    Ptr<uint>(gCG.w_contactIndexSortedByX)[index] = index;
    Ptr<float4>(gCG.w_outPoint)[index] = contact;
    Ptr<float4>(gCG.w_outNormalPen)[index] = normalPen;

    // result == true branch
    Ptr<float4>(gCG.w_outBarycentric)[index] = barycentric;
    // PxgFemOtherContactInfo (16B): pairInd0 u64 @0, pairInd1 u32 @8, rigidMatInd @12
    const uint64_t infoBase = gCG.w_outContactInfo + uint64_t(index) * 16UL;
    Ptr<uint2>(infoBase + 0UL)[0] = rigidId;                      // pairInd0 == rigidId.getInd()
    Ptr<uint>(infoBase + 8UL)[0] = pairInd1;
    // setRigidMaterialIndex(mat) + markInCollision(false) == (mat & 0x7fffffff)
    Ptr<uint>(infoBase + 12UL)[0] = rigidMaterialIndex & 0x7fffffffu;
}

// ===========================================================================
// cloth_planeVertContactGenLaunch (femClothPrimitives.cu:853)
// ===========================================================================
groupshared uint sPlaneMask[PLANE_WARPS];
groupshared uint sPlaneStart[PLANE_WARPS];

[shader("compute")]
[numthreads(512, 1, 1)] // (32, 16, 1) flattened
void cloth_planeVertContactGenLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint warpId = tid >> 5;
    const uint lane = tid & (WARP - 1u);
    const uint cmIdx = gid.x;

    if (cmIdx >= gCG.numTests)
        return;

    PxgShape clothShape, rigidShape;
    uint clothCacheRef, rigidCacheRef;
    loadClothRigidPair(gCG.cmInputs, gCG.gpuShapes, cmIdx,
                       clothShape, clothCacheRef, rigidShape, rigidCacheRef);

    const uint2 rigidId = Ptr<uint2>(gCG.shapeToRigidRemap)[rigidCacheRef];
    const uint femClothId = clothShape.particleOrSoftbodyId;
    Ptr<PxgFEMClothGpu> femClothes = Ptr<PxgFEMClothGpu>(gCG.femClothes);
    Ptr<float4> verts = Ptr<float4>(femClothes[femClothId].mPosition_InvMass);
    const uint numVerts = femClothes[femClothId].mNbVerts;

    const PxsCachedTransform rtc = Ptr<PxsCachedTransform>(gCG.transformCache)[rigidCacheRef];
    const float4 rigidQ = float4(rtc.qx, rtc.qy, rtc.qz, rtc.qw);
    const float3 rigidP = float3(rtc.px, rtc.py, rtc.pz);

    Ptr<float> contactDistance = Ptr<float>(gCG.contactDistance);
    const float contactDist = contactDistance[clothCacheRef] + contactDistance[rigidCacheRef];
    const uint rigidMaterialIndex = rigidShape.materialIndex;
    const float restDistance = Ptr<float>(gCG.restDistance)[cmIdx];

    // THE BOUND IS ROUNDED TO THE **BLOCK**, NOT TO THE WARP -- see BARRIER
    // UNIFORMITY in the header.  The CUDA rounds to 32 because its warp
    // intrinsics only need warp-uniform control flow; our emulation is
    // groupshared + a BLOCK barrier, which needs block-uniform control flow, and
    // a 32-rounded bound gives threads in the same block different trip counts.
    // Rounding to the block only adds all-false iterations: a tile with no
    // contact does no atomicAdd and no store, so the emitted set is unchanged.
    const uint numRequiredThreads = (numVerts + (PLANE_BLOCK - 1u)) & (~(PLANE_BLOCK - 1u));

    for (uint i = tid; i < numRequiredThreads; i += PLANE_BLOCK)
    {
        bool hasContact = false;
        float4 contact = float4(0, 0, 0, 0);
        float4 normalPen = float4(0, 0, 0, 0);
        uint pairInd1 = 0u;

        if (i < numVerts)
        {
            const float4 pf = verts[i];
            const float3 clothP = float3(pf.x, pf.y, pf.z);

            // vertPlaneCollide (femClothPrimitives.cu:141): p in plane space, plane
            // normal = q.getBasisVector0(); contact if contactDist >= pInPlane.x.
            const float3 pInPlaneSpace = quatRotateInv(rigidQ, clothP - rigidP);
            const float3 normal = getBasisVector0(rigidQ);
            if (contactDist >= pInPlaneSpace.x)
            {
                normalPen = float4(normal.x, normal.y, normal.z, pInPlaneSpace.x);

                const uint clothMask = (femClothId << PXG_CLOTH_ID_SHIFT) | i;
                // filter: drop the contact if (rigidId, clothMask) is filtered.
                const bool filtered = findRigid(gCG.filterPairs, gCG.nbFilterPairs, rigidId, uint2(clothMask, 0u));
                if (!filtered)
                {
                    hasContact = true;
                    pairInd1 = clothMask;
                    const float3 rP = clothP - normalPen.w * normal;
                    contact = float4(rP.x, rP.y, rP.z, restDistance);
                }
            }
        }

        // per-tile ballot -> lane-0 atomicAdd -> exclusive-prefix write.
        if (lane == 0u) sPlaneMask[warpId] = 0u;
        GroupMemoryBarrierWithGroupSync();
        if (hasContact) InterlockedOr(sPlaneMask[warpId], 1u << lane);
        GroupMemoryBarrierWithGroupSync();
        const uint mask = sPlaneMask[warpId];
        const uint totalCount = countbits(mask);
        const uint threadOffset = countbits(mask & ((1u << lane) - 1u));
        if (lane == 0u)
        {
            uint s = 0u;
            if (totalCount != 0u)
                InterlockedAdd(Ptr<uint>(gCG.w_totalContactCount)[0], totalCount, s);
            sPlaneStart[warpId] = s;
        }
        GroupMemoryBarrierWithGroupSync();
        const uint startIndex = sPlaneStart[warpId];
        if (hasContact)
            writeRigidVsDeformableContact(startIndex + threadOffset, contact, normalPen,
                                          float4(0, 0, 0, 1), rigidId, pairInd1, rigidMaterialIndex);
        GroupMemoryBarrierWithGroupSync();
    }
}

// ===========================================================================
// cloth_SphereVertexContactGenLaunch (femClothPrimitives.cu:700).  eCAPSULE
// branch DEFERRED (see header).
// ===========================================================================
groupshared uint sSphMask[SPHERE_WARPS];
groupshared uint sSphStart[SPHERE_WARPS];

[shader("compute")]
[numthreads(32, 1, 1)] // (32, 1, 1)
void cloth_SphereVertexContactGenLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint warpId = tid >> 5;      // == 0 (single warp)
    const uint lane = tid & (WARP - 1u);
    const uint cmIdx = gid.x;

    if (cmIdx >= gCG.numTests)
        return;

    PxgShape clothShape, rigidShape;
    uint clothCacheRef, rigidCacheRef;
    loadClothRigidPair(gCG.cmInputs, gCG.gpuShapes, cmIdx,
                       clothShape, clothCacheRef, rigidShape, rigidCacheRef);

    const uint rigidType = rigidShape.type;
    const uint2 rigidId = Ptr<uint2>(gCG.shapeToRigidRemap)[rigidCacheRef];
    const uint femClothId = clothShape.particleOrSoftbodyId;
    Ptr<PxgFEMClothGpu> femClothes = Ptr<PxgFEMClothGpu>(gCG.femClothes);
    Ptr<float4> verts = Ptr<float4>(femClothes[femClothId].mPosition_InvMass);
    const uint numVerts = femClothes[femClothId].mNbVerts;

    const PxsCachedTransform rtc = Ptr<PxsCachedTransform>(gCG.transformCache)[rigidCacheRef];
    const float3 rigidCenter = float3(rtc.px, rtc.py, rtc.pz);

    Ptr<float> contactDistance = Ptr<float>(gCG.contactDistance);
    const float contactDist = contactDistance[clothCacheRef] + contactDistance[rigidCacheRef];
    const uint rigidMaterialIndex = rigidShape.materialIndex;
    const float restDistance = Ptr<float>(gCG.restDistance)[cmIdx];
    const float sphereRadius = rigidShape.scaleY;   // scale.scale.y

    // Block-rounded -- see BARRIER UNIFORMITY above.  SPHERE_BLOCK is 32 here, so
    // this happens to equal the CUDA's `& ~31`; spelled against the BLOCK anyway,
    // because that is the rule.  A later SPHERE_WARPS > 1 would make `31` wrong
    // silently, and the sweep cannot see a barrier-uniformity fault.
    const uint numRequiredThreads = (numVerts + (SPHERE_BLOCK - 1u)) & (~(SPHERE_BLOCK - 1u));

    for (uint i = tid; i < numRequiredThreads; i += SPHERE_BLOCK)
    {
        bool hasContact = false;
        float4 contact = float4(0, 0, 0, 0);
        float4 normalPen = float4(0, 0, 0, 0);
        uint pairInd1 = 0u;

        if (i < numVerts)
        {
            const uint clothMask = (femClothId << PXG_CLOTH_ID_SHIFT) | i;
            const bool skipVertex = findRigid(gCG.filterPairs, gCG.nbFilterPairs, rigidId, uint2(clothMask, 0u));
            if (!skipVertex)
            {
                const float4 cf = verts[i];
                const float3 p = float3(cf.x, cf.y, cf.z);

                if (rigidType == GEOM_SPHERE)
                {
                    // spheresphere(p, center, r0=0, r1=radius, contactDist, normalPen)
                    // (sphereCollision.cuh:67).
                    const float3 delta = p - rigidCenter;
                    const float distanceSq = dot3(delta, delta);
                    const float radiusSum = sphereRadius;         // 0 + radius
                    const float inflatedSum = radiusSum + contactDist;
                    if ((inflatedSum * inflatedSum) > distanceSq)
                    {
                        const float eps = 1e-4f;
                        const float dist = sqrt(distanceSq);
                        const float3 n = (eps >= dist) ? float3(1.0f, 0.0f, 0.0f) : (delta * (1.0f / dist));
                        const float pen = dist - radiusSum;
                        normalPen = float4(n.x, n.y, n.z, pen);
                        hasContact = true;
                    }
                    // else: eCAPSULE branch DEFERRED (see header).
                }

                if (hasContact)
                {
                    pairInd1 = clothMask;
                    contact = float4(cf.x, cf.y, cf.z, restDistance); // verts[i], w=restDistance
                }
            }
        }

        if (lane == 0u) sSphMask[warpId] = 0u;
        GroupMemoryBarrierWithGroupSync();
        if (hasContact) InterlockedOr(sSphMask[warpId], 1u << lane);
        GroupMemoryBarrierWithGroupSync();
        const uint mask = sSphMask[warpId];
        const uint totalCount = countbits(mask);
        const uint threadOffset = countbits(mask & ((1u << lane) - 1u));
        if (lane == 0u)
        {
            uint s = 0u;
            if (totalCount != 0u)
                InterlockedAdd(Ptr<uint>(gCG.w_totalContactCount)[0], totalCount, s);
            sSphStart[warpId] = s;
        }
        GroupMemoryBarrierWithGroupSync();
        const uint startIndex = sSphStart[warpId];
        if (hasContact)
            writeRigidVsDeformableContact(startIndex + threadOffset, contact, normalPen,
                                          float4(0, 0, 0, 1), rigidId, pairInd1, rigidMaterialIndex);
        GroupMemoryBarrierWithGroupSync();
    }
}

// ===========================================================================
// P9.15a -- cloth_boxVertexContactGenLaunch
// gpunarrowphase/src/CUDA/femClothPrimitives.cu:3078, read in full, with the
// query itself at sphereCollision.cuh:371 (lifted into pxgSlangSphereBox).
//
// The third vertex-vs-primitive contact generator of this file and the closest
// sibling of the two above: same 11-argument pack, same ContactGenParams blob,
// same filter, same writer, same warp-aggregation tail.  It is the ONLY kernel
// of the three that an existing SCENE already asks for -- `cloth` emits
// "GPU cloth_boxVertexContactGenLaunch fail to launch" every frame -- so this
// is the one place in the port where the work list came from the engine rather
// than from us.
//
// ---------------------------------------------------------------------------
// WHAT DIFFERS FROM THE PLANE / SPHERE SIBLINGS
// ---------------------------------------------------------------------------
// V1 -- THE CONTACT POINT IS THE POINT ON THE **RIGID**, NOT THE CLOTH VERTEX.
//     `rigidP = clothP - normalPen.w * normalPen.xyz`, then
//     `contact = float4(rigidP, restDistance)`.  The SPHERE sibling stores the
//     cloth vertex itself (`verts[i]`); the PLANE sibling stores the projected
//     point, like this one.  Three neighbouring kernels, two conventions, and
//     the difference is invisible until the vertex is actually penetrating --
//     at pen == 0 the two points coincide.  The fixture therefore needs a
//     vertex strictly INSIDE the box, not merely within the contact distance.
//
// V2 -- the block is (32, **16**, 1) = 512 threads, one BLOCK per contact-manager
//     pair (`cmIdx = blockIdx.x`), so it shares the plane sibling's shape and NOT
//     the sphere's single warp.
//
// V3 -- the filter test runs BEFORE the query, unlike the plane sibling which
//     queries first and filters after.  Both suppress the same contacts, so this
//     is transcription rather than semantics -- recorded because a reader
//     comparing the two bodies will notice and wonder.
//
// V4 -- the box half-extents are `rigidShape.scale.scale`, read as
//     (scaleX, scaleY, scaleZ).  There is no halving anywhere in the chain: a
//     PxBoxGeometry's half-extents ARE the shape scale (SB7).
//
// V5 -- the sphere radius handed to the query is 0.  A cloth VERTEX is a point;
//     the radius exists only because the query is shared with the real
//     sphere-vs-box path.
//
// V6 -- `restDistance` is indexed by **cmIdx**, the contact-manager pair, while
//     `contactDistance` is indexed by the two CACHE REFS and summed.  Two
//     different index spaces, one line apart.
//
// ---------------------------------------------------------------------------
// BARRIER UNIFORMITY -- a platform rule this tranche had to learn
// ---------------------------------------------------------------------------
// The CUDA loops to `numRequiredThreads = (numVerts + 31) & ~31` -- rounded up to
// a WARP -- and aggregates with `__ballot_sync` / `warpScanExclusive`, which
// require only WARP-uniform control flow.  Every thread of a warp shares a trip
// count there, so the CUDA is correct.
//
// Our port has no warp intrinsics: it emulates them with groupshared slots and
// `GroupMemoryBarrierWithGroupSync()`, which is `OpControlBarrier` and requires
// control flow uniform over the **whole block**.  With a 512-thread block and a
// 32-rounded bound, `numVerts = 3` gives threads 0..31 one iteration and threads
// 32..511 none -- so the barrier sits in non-uniform control flow, which is
// undefined in SPIR-V and Vulkan.  llvmpipe tolerates it; that is not a licence.
//
// So the bound is rounded to the BLOCK here, and the plane sibling above was
// corrected to match in the same tranche -- it had the hazard from P7.1 and its
// own fixture (3 verts, 512 threads) was reaching it on every run.  The sphere
// sibling needs no change: its block IS one warp, so a 32-rounded bound is
// already block-uniform.
//
// The fix cannot change what is emitted.  The extra iterations are all-false for
// every lane, and an all-false tile takes neither the atomicAdd nor the store.
// ===========================================================================
groupshared uint sBoxMask[BOX_WARPS];
groupshared uint sBoxStart[BOX_WARPS];

[shader("compute")]
[numthreads(512, 1, 1)] // (32, 16, 1) flattened -- V2
void cloth_boxVertexContactGenLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint warpId = tid >> 5;
    const uint lane = tid & (WARP - 1u);
    const uint cmIdx = gid.x;

    if (cmIdx >= gCG.numTests)
        return;

    PxgShape clothShape, rigidShape;
    uint clothCacheRef, rigidCacheRef;
    loadClothRigidPair(gCG.cmInputs, gCG.gpuShapes, cmIdx,
                       clothShape, clothCacheRef, rigidShape, rigidCacheRef);

    // V4: the shape scale IS the box's half-extents.
    const float3 boxHalfExtents = float3(rigidShape.scaleX, rigidShape.scaleY, rigidShape.scaleZ);

    const uint2 rigidId = Ptr<uint2>(gCG.shapeToRigidRemap)[rigidCacheRef];
    const uint femClothId = clothShape.particleOrSoftbodyId;
    Ptr<PxgFEMClothGpu> femClothes = Ptr<PxgFEMClothGpu>(gCG.femClothes);
    Ptr<float4> verts = Ptr<float4>(femClothes[femClothId].mPosition_InvMass);
    const uint numVerts = femClothes[femClothId].mNbVerts;

    const PxsCachedTransform rtc = Ptr<PxsCachedTransform>(gCG.transformCache)[rigidCacheRef];
    const float4 boxQ = float4(rtc.qx, rtc.qy, rtc.qz, rtc.qw);
    const float3 boxP = float3(rtc.px, rtc.py, rtc.pz);

    Ptr<float> contactDistance = Ptr<float>(gCG.contactDistance);
    const float contactDist = contactDistance[clothCacheRef] + contactDistance[rigidCacheRef];
    const uint rigidMaterialIndex = rigidShape.materialIndex;
    const float restDistance = Ptr<float>(gCG.restDistance)[cmIdx];   // V6
    const float sphereRadius = 0.0f;                                  // V5

    // Block-rounded -- see BARRIER UNIFORMITY above.
    const uint numRequiredThreads = (numVerts + (BOX_BLOCK - 1u)) & (~(BOX_BLOCK - 1u));

    for (uint i = tid; i < numRequiredThreads; i += BOX_BLOCK)
    {
        bool hasContact = false;
        float4 contact = float4(0, 0, 0, 0);
        float4 normalPen = float4(0, 0, 0, 0);
        uint pairInd1 = 0u;

        if (i < numVerts)
        {
            const uint clothMask = (femClothId << PXG_CLOTH_ID_SHIFT) | i;
            // V3: the filter runs FIRST on this path.
            const bool skipVertex = findRigid(gCG.filterPairs, gCG.nbFilterPairs, rigidId, uint2(clothMask, 0u));
            if (!skipVertex)
            {
                const float4 pf = verts[i];
                const float3 clothP = float3(pf.x, pf.y, pf.z);

                if (sphereboxNormalPen(clothP, boxQ, boxP, sphereRadius,
                                       boxHalfExtents, contactDist, normalPen) != 0u)
                {
                    hasContact = true;
                    pairInd1 = clothMask;
                    // V1: the point on the RIGID, not the cloth vertex.
                    const float3 rP = clothP - normalPen.w * float3(normalPen.x, normalPen.y, normalPen.z);
                    contact = float4(rP.x, rP.y, rP.z, restDistance);
                }
            }
        }

        // per-tile ballot -> lane-0 atomicAdd -> exclusive-prefix write.
        if (lane == 0u) sBoxMask[warpId] = 0u;
        GroupMemoryBarrierWithGroupSync();
        if (hasContact) InterlockedOr(sBoxMask[warpId], 1u << lane);
        GroupMemoryBarrierWithGroupSync();
        const uint mask = sBoxMask[warpId];
        const uint totalCount = countbits(mask);
        const uint threadOffset = countbits(mask & ((1u << lane) - 1u));
        if (lane == 0u)
        {
            uint s = 0u;
            if (totalCount != 0u)
                InterlockedAdd(Ptr<uint>(gCG.w_totalContactCount)[0], totalCount, s);
            sBoxStart[warpId] = s;
        }
        GroupMemoryBarrierWithGroupSync();
        const uint startIndex = sBoxStart[warpId];
        if (hasContact)
            writeRigidVsDeformableContact(startIndex + threadOffset, contact, normalPen,
                                          float4(0, 0, 0, 1), rigidId, pairInd1, rigidMaterialIndex);
        GroupMemoryBarrierWithGroupSync();
    }
}

// ===========================================================================
// P9.15b -- cloth_midphaseGeneratePairsLaunch
// gpunarrowphase/src/CUDA/femClothMidPhase.cu:191 (wrapper), :110-186
// (femClothMidphaseCore) and :64-110 (ClothTreeTraverser), all read in full.
//
// Walks the CLOTH's own BV32 tree against ONE rigid's world AABB and pushes the
// (cmIdx, triangleIndex) pairs whose triangle bound overlaps it.  It is the
// producer cloth_boxTriangleContactGenLaunch consumes, and the second of the
// three kernels the `cloth` scene has been naming.
//
// ITS TWIN IS ALREADY PORTED.  sb_midphaseGeneratePairsLaunch
// (softBodyContactGenBda.slang, P8.2) is the same traversal against the same
// kind of AABB, and this body is composed from that shape rather than derived
// again from bv32Traversal.cuh.  The BV32 node reader itself is the shared
// pxgSlangBv32 module (P8.25).  What is NOT yet shared is the traversal LOOP,
// which now exists in seven transcriptions across this corpus -- filed as its
// own refactor rather than attempted here.
//
// ---------------------------------------------------------------------------
// WHAT DIFFERS FROM THE SOFTBODY TWIN
// ---------------------------------------------------------------------------
// M1 -- TRIANGLES, NOT TETS.  The primitive is `mTriangleVertexIndices[i]`, a
//     uint4 whose .xyz are the three corners and whose .w is not part of the
//     triangle.  Three vertices bound it, not four, and reading .w into the
//     bound is a silent enlargement that only shows when .w indexes a vertex
//     outside the triangle's own extent.
//
// M2 -- NO SURFACE HINT.  The softbody twin skips interior tetrahedra through
//     mTetMeshSurfaceHint; a cloth has no interior, so every triangle is a
//     candidate and there is no hint array to read.
//
// M3 -- NO VALIDITY TEST.  The softbody twin drops degenerate tets via
//     isValidTet; the cloth traverser has no equivalent and bounds a degenerate
//     triangle like any other.
//
// M4 -- the tree lives at `mTriMeshData + sizeof(uint4)`.  The 16-byte header
//     (nbVerts, nbTets, maxDepth, nbBv32TreeNodes) is skipped exactly as the
//     softbody twin skips its own -- and, like it, none of those four fields is
//     read by the traversal.
//
// M5 -- THE BOX IS THE RIGID'S, FATTENED BY THE PAIR'S SUMMED CONTACT DISTANCE.
//     `contactDistance[clothCacheRef] + contactDistance[rigidCacheRef]`, then
//     `aabb.fattenFast(contactDist)` -- min minus, max plus, on all three axes.
//     The CLOTH's own bound is never read; the cloth side of the test is the
//     per-triangle bound computed from live vertices.
//
// M6 -- `stackSizeBytes` IS THE LAST ARGUMENT, and the stack bound is
//     `stackSizeBytes / sizeof(uint4)`.  The softbody twin takes the same eleven
//     arguments in a DIFFERENT order (see ClothTriMidphaseParams).
//
// M7 -- THE COUNTER IS NOT CAPPED, THE WRITE IS.  `pushOntoStackFullWarp`
//     (bv32Traversal.cuh:197) reserves with one atomicAdd per tile and then
//     drops any slot at or past stackSize.  So midphasePairsNum can exceed the
//     stack -- that overshoot is how the engine sizes the next frame's stack
//     (`stackSizeNeededOnDevice`).  Capping the counter would silently shrink it.
//
// DEVIATIONS, deliberate, and the same two the softbody twin documents:
//   - the CUDA caches meshVerts / meshVertsIndices / bv32PackedNodes into
//     __shared__ behind `if (threadIdx.x == 0)` + __syncwarp.  Every one of them
//     is read straight through BDA here, so the cache and its barrier are
//     dropped; they have no observable effect;
//   - the engine packs MIDPHASE_WARPS_PER_BLOCK warps per block and the bridge
//     forwards only gridDim, so the block's warps are walked SERIALLY in a
//     group-uniform loop.  `sub` and its break are group-uniform, so every
//     barrier below is reached uniformly -- the P9.15a rule, honoured by
//     construction rather than by luck.
// ===========================================================================

static const uint TMID_WARPS_PER_BLOCK = 2u;   // MIDPHASE_WARPS_PER_BLOCK
static const uint TMID_STACK_SIZE = 192u;      // femMidphaseScratch.sBv32Nodes[192]

bool tmidBoundsIntersect(float3 aMin, float3 aMax, float3 bMin, float3 bMax)
{
    return !(bMin.x > aMax.x || aMin.x > bMax.x ||
             bMin.y > aMax.y || aMin.y > bMax.y ||
             bMin.z > aMax.z || aMin.z > bMax.z);
}

groupshared int sTmidStack[TMID_STACK_SIZE];   // the to-be-expanded node stack
groupshared uint sTmidBallot;                  // __ballot_sync mask
groupshared uint sTmidStart;                   // atomicAdd result broadcast

uint tmidBallot(bool pred, uint lane)
{
    if (lane == 0u) sTmidBallot = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sTmidBallot, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sTmidBallot;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void cloth_midphaseGeneratePairsLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    for (uint sub = 0u; sub < TMID_WARPS_PER_BLOCK; ++sub)
    {
    GroupMemoryBarrierWithGroupSync();   // the groupshared scratch is reused per sub-warp
    const uint cmIdx = gid.x * TMID_WARPS_PER_BLOCK + sub;
    if (cmIdx >= gTMid.numTests)
        break;

    PxgShape clothShape, rigidShape;
    uint clothCacheRef, rigidCacheRef;
    loadClothRigidPair(gTMid.cmInputs, gTMid.gpuShapes, cmIdx,
                       clothShape, clothCacheRef, rigidShape, rigidCacheRef);

    const uint femClothId = clothShape.particleOrSoftbodyId;
    Ptr<PxgFEMClothGpu> femClothes = Ptr<PxgFEMClothGpu>(gTMid.femClothes);

    const uint64_t triMeshData = femClothes[femClothId].mTriMeshData;
    const uint64_t treeAddr = triMeshData + 16UL;   // M4: skip the uint4 header

    Ptr<float4> meshVerts = Ptr<float4>(femClothes[femClothId].mPosition_InvMass);
    Ptr<uint4> meshVertsIndices = Ptr<uint4>(femClothes[femClothId].mTriangleVertexIndices);

    // M5: the RIGID's world AABB, fattened by the pair's summed contact distance.
    Ptr<float> bounds = Ptr<float>(gTMid.bounds);
    const uint rb = rigidCacheRef * 6u;
    Ptr<float> contactDistance = Ptr<float>(gTMid.contactDistance);
    const float contactDist = contactDistance[clothCacheRef] + contactDistance[rigidCacheRef];
    const float3 boxMin = float3(bounds[rb + 0u], bounds[rb + 1u], bounds[rb + 2u]) - contactDist;
    const float3 boxMax = float3(bounds[rb + 3u], bounds[rb + 4u], bounds[rb + 5u]) + contactDist;

    const uint stackSize = gTMid.stackSizeBytes / 16u;   // M6: / sizeof(uint4)

    uint currentNodeIndex = 0u;   // root
    bool hasNode = true;
    uint nbStack = 0u;

    while (hasNode)
    {
        const uint nodeIdx = bvNodeIdxOf(currentNodeIndex);

        if (bvIsLeaf(currentNodeIndex))
        {
            // Leaf: the primitive range lives in the PARENT node's mData slot.
            const uint leafOffset = bvLeafSlotOf(currentNodeIndex);
            const uint data = bvData(treeAddr, nodeIdx, leafOffset);
            const uint nbPrimitives = bvDataNbPrimitives(data);
            const uint primStartIndex = bvDataPrimStart(data);
            uint primitiveIndex = 0xffffffffu;
            if (lane < nbPrimitives)
                primitiveIndex = lane + primStartIndex;

            // ClothTreeTraverser::intersectPrimitiveFullWarp (femClothMidPhase.cu:87).
            bool intersect = false;
            if (primitiveIndex != 0xffffffffu)
            {
                // M1: three corners, from .xyz.  M2/M3: no hint, no validity test.
                const uint4 triIdx = meshVertsIndices[primitiveIndex];
                const float3 w0 = meshVerts[triIdx.x].xyz;
                const float3 w1 = meshVerts[triIdx.y].xyz;
                const float3 w2 = meshVerts[triIdx.z].xyz;
                // triBoundingBox (deformableCollision.cuh) -- pairwise min/max in the
                // same association order as the .cu.
                const float3 tMin = min(w0, min(w1, w2));
                const float3 tMax = max(w0, max(w1, w2));
                intersect = tmidBoundsIntersect(tMin, tMax, boxMin, boxMax);
            }

            // M7: pushOntoStackFullWarp -- one atomicAdd per tile, scatter by the
            // exclusive prefix, DROP anything at or past stackSize.  The counter is
            // NOT capped; its overshoot is what sizes the next frame's stack.
            const uint mask = tmidBallot(intersect, lane);
            const uint offset = countbits(mask & ((1u << lane) - 1u));
            const uint validCount = countbits(mask);
            if (lane == 0u)
            {
                uint start = 0u;
                if (validCount > 0u)
                    InterlockedAdd(Ptr<uint>(gTMid.midphasePairsNum)[0], validCount, start);
                sTmidStart = start;
            }
            GroupMemoryBarrierWithGroupSync();
            const uint stackIndex = sTmidStart + offset;
            if (intersect && stackIndex < stackSize)
                Ptr<uint4>(gTMid.stackPtr)[stackIndex] = uint4(cmIdx, primitiveIndex, 0u, 0u);
        }
        else
        {
            // Internal node: test the rigid box against each child's bounds.
            const uint nbChildren = bvNbNodes(treeAddr, nodeIdx);
            float3 mn = float3(0, 0, 0), mx = float3(0, 0, 0);
            const bool hasBox = lane < nbChildren;
            if (hasBox)
            {
                mn = bvMin(treeAddr, nodeIdx, lane);
                mx = bvMax(treeAddr, nodeIdx, lane);
            }
            // ClothTreeTraverser::intersectBoxFullWarp: false when the lane has no box.
            bool intersect = false;
            if (hasBox)
                intersect = tmidBoundsIntersect(boxMin, boxMax, mn, mx);

            uint currentIndex = bvChildIndex(currentNodeIndex, lane);
            bool isValidNode = false;
            if (lane < nbChildren)
            {
                const uint data = bvData(treeAddr, nodeIdx, lane);
                isValidNode = intersect;
                if (isValidNode && bvDataIsInternal(data))
                    currentIndex = bvDataChildNodeIndex(data);
            }

            const uint mask = tmidBallot(isValidNode, lane);
            const uint offset = countbits(mask & ((1u << lane) - 1u));
            const uint validCount = countbits(mask);
            if (isValidNode)
                sTmidStack[nbStack + offset] = int(currentIndex);
            nbStack += validCount;
        }

        GroupMemoryBarrierWithGroupSync();
        if (nbStack > 0u)
        {
            nbStack -= 1u;
            currentNodeIndex = uint(sTmidStack[nbStack]);
        }
        else
            hasNode = false;
    }
    }
}

// ===========================================================================
// P9.15d -- cloth_boxTriangleContactGenLaunch
// gpunarrowphase/src/CUDA/femClothPrimitives.cu:3357 (wrapper) and :3216-3355
// (clothBoxTriangleCollision), both read in full.
//
// The LAST kernel the `cloth` scene refuses, and the consumer of the pairs
// P9.15b produces.  Its contact machine -- convexTriangleContactGen: SAT,
// feature selection, polygon clipping, contact reduction and the writer -- is
// NOT here: P9.15c lifted it into pxgSlangConvexTriCloth, where a Phase-1
// battery verifies it.  What is here is exactly what that module's header has
// always called "caller glue not ported": load the pair, build the two
// transforms, apply the filter, move the triangle into convex space, fill an
// FctCase, call.
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
// B1 -- THE FILTER IS AN **AND** OVER ALL THREE VERTICES.  A triangle is skipped
//     only when EVERY one of its corners is filtered against this rigid:
//         skip = find(v0) && find(v1) && find(v2)
//     The vertex kernel next door skips on a SINGLE match.  Two kernels, one
//     filter list, opposite quantifiers -- and `||` here silently drops
//     triangles that share one attached corner, which is the common case for a
//     cloth pinned at a few points.
//
// B2 -- THE CLOTH IS NOT SCALED.  The CUDA has the three vertex2Shape calls
//     COMMENTED OUT and uses the raw world vertices; only the RIGID carries a
//     shape scale.  A port that "restores" the symmetry scales the cloth twice.
//
// B3 -- the pair count is `min(*midphasePairsNum, stackSize)`.  The midphase's
//     counter overshoots on purpose (P9.15b's M7), so consuming it unclamped
//     walks off the stack.
//
// B4 -- `stackSizeNeededOnDevice` is an atomicMax of the UNCLAMPED count times
//     sizeof(uint4), written by ONE thread of the whole grid, and it is how the
//     engine sizes next frame's stack.  Feeding it the clamped count makes the
//     stack never grow.
//
// B5 -- trimeshToConvexTransform is `rigidTransform.transformInv(clothTransform)`:
//         q = conj(rigidQ) * clothQ,   p = quatRotateInv(rigidQ, clothP - rigidP)
//     and the triangle is then moved by THAT, while `convexTransform` -- the one
//     the module uses to put the contact back into world space -- is the RIGID's
//     own transform.  Two transforms, one of them a composition; swapping them
//     produces contacts in the wrong frame.
//
// B6 -- restDistance is indexed by **cmIdx** (from the pair's .x), contactDistance
//     by the two cache refs and summed.  Same split as the vertex kernel's V6.
//
// B7 -- the grid-stride is `gridDim.x * blockDim.y` = 4096 * 2, reproduced as a
//     constant because the bridge forwards only gridDim.  Same precedent as the
//     softbody contact-gen's SB_CG_GRID.
//
// DEVIATION: the CUDA runs one WARP per pair over a shared ConvexTriangleScratch;
// the lifted module is per-thread (its own header records why that is
// output-identical -- every reduction is order-fixed and the only atomic is the
// single output reservation).  So lane 0 owns the pair and lanes 1..31 return.
// ===========================================================================

static const uint BTRI_GRID = 4096u;
static const uint BTRI_WARPS_PER_BLOCK = 2u;

[shader("compute")]
[numthreads(32, 2, 1)]
void cloth_boxTriangleContactGenLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    // The module's driver is per-thread; lane 0 owns the pair.
    if (gtid.x != 0u)
        return;
    const uint globalWarpIdx = gtid.y + gid.x * BTRI_WARPS_PER_BLOCK;

    const uint stackSize = gBTri.stackSizeBytes / 16u;
    const uint numFound = Ptr<uint>(gBTri.midphasePairsNum)[0];
    const uint numPairs = min(numFound, stackSize);          // B3

    Ptr<uint4> pairs = Ptr<uint4>(gBTri.stackPtr);
    Ptr<PxgFEMClothGpu> femClothes = Ptr<PxgFEMClothGpu>(gBTri.femClothes);
    Ptr<PxsCachedTransform> transformCache = Ptr<PxsCachedTransform>(gBTri.transformCache);
    Ptr<float> contactDistance = Ptr<float>(gBTri.contactDistance);

    FcCtWriter w;
    w.point = gBTri.b_outPoint;  w.normalPen = gBTri.b_outNormalPen;
    w.bary = gBTri.b_outBarycentric; w.info = gBTri.b_outContactInfo;
    w.count = gBTri.b_totalContactCount; w.byX = gBTri.b_contactByX;
    w.tempByX = gBTri.b_tempContactByX;  w.idxByX = gBTri.b_contactIndexSortedByX;
    w.maxContacts = gBTri.b_maxNumContacts;

    for (uint i = globalWarpIdx; i < numPairs; i += BTRI_GRID * BTRI_WARPS_PER_BLOCK)  // B7
    {
        const uint4 curPair = pairs[i];
        const uint cmIdx = curPair.x;
        const uint triangleIdx = curPair.y;

        PxgShape clothShape, rigidShape;
        uint clothCacheRef, rigidCacheRef;
        loadClothRigidPair(gBTri.cmInputs, gBTri.gpuShapes, cmIdx,
                           clothShape, clothCacheRef, rigidShape, rigidCacheRef);

        const uint2 rigidId = Ptr<uint2>(gBTri.shapeToRigidRemap)[rigidCacheRef];
        const uint clothId = clothShape.particleOrSoftbodyId;

        Ptr<uint4> vertIndices = Ptr<uint4>(femClothes[clothId].mTriangleVertexIndices);
        const uint4 triIdx = vertIndices[triangleIdx];

        // B1: skip ONLY when all three corners are filtered.
        const uint m0 = (clothId << PXG_CLOTH_ID_SHIFT) | triIdx.x;
        const uint m1 = (clothId << PXG_CLOTH_ID_SHIFT) | triIdx.y;
        const uint m2 = (clothId << PXG_CLOTH_ID_SHIFT) | triIdx.z;
        const bool skipTriangle =
            findRigid(gBTri.filterPairs, gBTri.nbFilterPairs, rigidId, uint2(m0, 0u)) &&
            findRigid(gBTri.filterPairs, gBTri.nbFilterPairs, rigidId, uint2(m1, 0u)) &&
            findRigid(gBTri.filterPairs, gBTri.nbFilterPairs, rigidId, uint2(m2, 0u));
        if (skipTriangle)
            continue;

        const PxsCachedTransform ctc = transformCache[clothCacheRef];
        const PxsCachedTransform rtc = transformCache[rigidCacheRef];
        const float4 clothQ = float4(ctc.qx, ctc.qy, ctc.qz, ctc.qw);
        const float3 clothP = float3(ctc.px, ctc.py, ctc.pz);
        const float4 rigidQ = float4(rtc.qx, rtc.qy, rtc.qz, rtc.qw);
        const float3 rigidP = float3(rtc.px, rtc.py, rtc.pz);

        // B5: PxTransform::transformInv -- q = conj(rigidQ) * clothQ,
        //     p = conj(rigidQ).rotate(clothP - rigidP).
        const float4 toConvQ = ctQuatMul(ctQuatConj(rigidQ), clothQ);
        const float3 toConvP = quatRotateInv(rigidQ, clothP - rigidP);

        Ptr<float4> verts = Ptr<float4>(femClothes[clothId].mPosition_InvMass);
        // B2: the cloth is NOT scaled -- the raw world vertices go straight in.
        const float3 v0 = verts[triIdx.x].xyz;
        const float3 v1 = verts[triIdx.y].xyz;
        const float3 v2 = verts[triIdx.z].xyz;
        const float3 t0 = ctQuatRotate(toConvQ, v0) + toConvP;
        const float3 t1 = ctQuatRotate(toConvQ, v1) + toConvP;
        const float3 t2 = ctQuatRotate(toConvQ, v2) + toConvP;

        const float3 triN = normalize(cross(t1 - t0, t2 - t0));

        FctCase cs;
        cs.convexScale = float4(rigidShape.scaleX, rigidShape.scaleY, rigidShape.scaleZ,
                                contactDistance[clothCacheRef] + contactDistance[rigidCacheRef]);
        cs.convexScaleRot = float4(rigidShape.scaleRotX, rigidShape.scaleRotY,
                                   rigidShape.scaleRotZ, rigidShape.scaleRotW);
        cs.t0 = float4(t0, Ptr<float>(gBTri.restDistance)[cmIdx]);   // B6
        cs.t1 = float4(t1, 0.0f);
        cs.t2 = float4(t2, 0.0f);
        cs.triNormal = float4(triN, 0.0f);
        cs.convexP = float4(rigidP, 0.0f);      // B5: the RIGID's own transform
        cs.convexQ = rigidQ;
        cs.ids = uint4(rigidId.x, rigidId.y, clothId, triangleIdx);
        cs.ids2 = uint4(rigidShape.materialIndex, 0u, 0u, 0u);

        const uint64_t hullAddr = uint64_t(rigidShape.hullOrMeshOffset.x) |
                                  (uint64_t(rigidShape.hullOrMeshOffset.y) << 32);

        fcClothConvexTriContactGen(cs, hullAddr, w);
    }

    // B4: the UNCLAMPED count, in bytes, by one thread of the grid.
    if (globalWarpIdx == 0u)
        InterlockedMax(Ptr<uint>(gBTri.stackSizeNeeded)[0], numFound * 16u);
}
