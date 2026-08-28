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

// femClothHfContactGenBda.slang -- P9.27b
//
// cloth_heightfieldContactGenLaunch (femClothPrimitives.cu:3643) and
// cloth_heightfieldVertexContactGenLaunch (:2581), both read in full.  The
// consumers of the midphase pair P9.27 landed.
//
// ONE of them writes contacts.  The other does not, and that is upstream's doing
// -- see the second half of this header.
//
// ===========================================================================
// THE LIVE ONE: cloth_heightfieldContactGenLaunch
// ===========================================================================
// Structurally cloth_meshContactGenLaunch's sibling: walk the pair stack, put the
// static triangle into the CLOTH's space, run staticTriangleVsClothCollision,
// ballot, reserve, write.  The core is shared (pxgSlangClothTriContact, P9.27b),
// so what is left here is the four things that differ -- and every one of them is
// a plausible copy-paste from the file next door:
//
// C1 -- THE SHAPES ARE READ POSITIONALLY, NOT ROUTED BY BODY ID.  The mesh sibling
//     has eTRIANGLEMESH on both sides and can only tell cloth from mesh by which
//     one carries a body id (`particleOrSoftbodyId != 0xffffffff`).  This kernel
//     does NOT do that: it takes `gpuShapes[shapeRef0]` as the cloth and
//     `gpuShapes[shapeRef1]` as the heightfield, full stop (:3690-3691), and the
//     transform cache ref it uses for the rigid side is transformCacheRef1.
//     Transcribed as it stands.  A port that "helpfully" added the id swap would
//     silently exchange the two shapes for any pair the broadphase happened to
//     emit the other way round.
//
// C2 -- THE TRIANGLE COMES OUT OF THE SAMPLES ALREADY SCALED.  getTriangle
//     multiplies the row/height/column units by the shape scale itself
//     (hfGetTriangle in pxgSlangHeightfield), so there is NO vertex2Shape here.
//     The mesh sibling needs one because its cooked vertices are in vertex space.
//     Applying the scale twice is the obvious error and it is silent: it moves
//     the terrain rather than failing.
//
// C3 -- NO FILTER PAIRS.  The mesh sibling carries filterPairs/nbFilterPairs and
//     drops a cloth triangle only when ALL THREE of its vertices are filtered
//     against the rigid.  This pack has neither argument (which is also why the
//     two ABI rows differ), so every pair the midphase produced is tested.
//
// C4 -- THE TRANSFORM IS hfToCloth, built as clothTransform.transformInv(hfTransform),
//     and applied to a triangle that is already in the heightfield's SHAPE space.
//
// THE GRID: the engine launches (2048, 1, 1) x (128, 1, 1)
// (PxgNarrowphaseCore.cpp:7024) and grid-strides by gridDim.x * blockDim.x =
// 262144 over `l = ceil(numPairs / 32) * 32`.  The bridge drops blockDim, so a
// [numthreads(32,1,1)] group is ONE warp and the block's FOUR warps are walked
// serially.  `base`, the stride and `l` are all multiples of 32, so the loop
// condition is identical in every lane and the ballot's barriers stay
// group-uniform.
//
// ===========================================================================
// THE DEAD ONE: cloth_heightfieldVertexContactGenLaunch
// ===========================================================================
// The same story P9.26 found in the cloth-vs-cooked-mesh chain, in the same file,
// one screen up.  The kernel computes a closest point, a normal, a penetration and
// a barycentric for every (cloth vertex, heightfield triangle) pair -- and throws
// all of it away, because the only assignment to its `hasContacts` flag is
// commented out:
//
//     bool hasContacts = false;                      // :2616
//     ...
//         if ((pen <= cDistance) && normal.dot(n) > 0.f)
//         {
//             pairInd0 = rigidId.getInd();
//             pairInd1 = PxEncodeClothIndex(clothId, clothVertIndex);
//             //hasContacts = true;                   // :2737  <-- COMMENTED OUT
//             contact = make_float4(...);
//         }
//
// PxgOutputIndex (:79) gates its atomicAdd on a non-zero popcount, so the contact
// COUNTER does not move either, and the kernel's whole observable output is one
// atomicMax sizing next frame's stack.
//
// This port therefore implements the observable behaviour and NOT the dead
// geometry, for the reason P9.26 wrote down at length: the only observable is "no
// contacts", which every implementation of dead code satisfies equally, so a
// transcription would be two hundred lines no fixture here could verify.
//
// tools/check_disabled_kernels.py has carried this kernel as a NOTE since P9.26,
// with the instruction "gate it when a Slang port of it lands".  P9.27b is that
// landing, and it is now GATED: the build fails the day upstream re-enables the
// line, deletes it, or stops gating the write on the flag.
//
// Upstream candidate 33's sibling.  Documented, NOT reported.
//
// One more, found while reading and worth a line because it is the kind of thing
// that hides a real bug: the shape-type assertion at :2640 is
//     assert(shape0.type = PxGeometryType::eTRIANGLEMESH && shape1.type == ...)
// -- a single `=`, so it asserts an ASSIGNMENT and can never fail.  Harmless here
// (the target is a local copy and asserts are compiled out in release), but it
// means the type invariant this kernel relies on has never actually been checked.
// Upstream candidate 38; documented, NOT reported.
// ===========================================================================
module femClothHfContactGenBda;

import pxgSlangCommon;           // quatRotate / quatRotateInv / PxgTransform28
import pxgSlangFEMClothDesc;     // PxgFEMClothGpu
import pxgSlangHeightfield;      // sample format + hfGetTriangle
import pxgSlangClothTriContact;  // staticTriangleVsClothCollision + its clipping

static const uint WARP = 32u;
static const uint HF_CG_WARPS_PER_BLOCK = 4u;                    // block (128,1,1)
static const uint HF_CG_GRID = 2048u;
static const uint HF_CG_STRIDE = HF_CG_GRID * HF_CG_WARPS_PER_BLOCK * WARP;   // 262144

// PxgFEMCloth.h: the element id occupies the low bits of the encoded index.
static const uint PXG_BITSHIFT_ELEMENT_ID = 20u;

// ---------------------------------------------------------------------------
// The LIVE kernel's push-constant blob -- BYTE-IDENTICAL to the CUDA pack
// (PxgNarrowphaseCore.cpp:7006): { PxReal tolerenceLength, cmInputs*,
// transformCache*, contactDistance*, restDistance*, gpuShapes*, clothes*,
// shapeToRigidRemapTable*, stackPtr*, midphasePairsNum*, PxU32 stackSizeBytes,
// stackSizeNeededOnDevice*, PxgFEMContactWriter writer }.
// 4 @0, nine pointers @8..@72, PxU32 @80, a pointer @88, then the 80-byte writer
// at @96 (already 16-aligned, so paramAlign moves nothing) -> 176 bytes.
// ABI row { 4,8,8,8,8,8,8,8,8,8,4,8,80 }.
// ---------------------------------------------------------------------------
struct ClothHfCgParams
{
    float toleranceLength;         // @0   (ABI parity only)
    uint _pad4;                    // @4
    uint64_t cmInputs;             // @8
    uint64_t transformCache;       // @16
    uint64_t contactDistance;      // @24
    uint64_t restDistance;         // @32
    uint64_t gpuShapes;            // @40
    uint64_t femClothes;           // @48
    uint64_t shapeToRigidRemap;    // @56
    uint64_t stackPtr;             // @64
    uint64_t midphasePairsNum;     // @72
    uint stackSizeBytes;           // @80
    uint _pad84;                   // @84
    uint64_t stackSizeNeededOnDevice; // @88
    // ---- PxgFEMContactWriter (by value, 80 B) @96 ----
    uint64_t w_outPoint;           // @96
    uint64_t w_outNormalPen;       // @104
    uint64_t w_outBarycentric;     // @112
    uint64_t w_outContactInfo;     // @120
    uint64_t w_totalContactCount;  // @128
    uint64_t w_contactByX;         // @136
    uint64_t w_tempContactByX;     // @144
    uint64_t w_contactIndexSortedByX; // @152
    uint64_t w_contactSortedByX;   // @160
    uint w_maxNumContacts;         // @168
    uint _pad172;                  // @172 -> 176
}

[[vk::push_constant]] ConstantBuffer<ClothHfCgParams> gCg;

struct PxgContactManagerInput { uint shapeRef0, shapeRef1, transformCacheRef0, transformCacheRef1; }

// PxgShape comes from pxgSlangCommon -- a local copy here would be a fourth, and
// the compiler says so: "ambiguous reference to 'PxgShape'".  A good error.

float3 transformPt(float4 q, float3 p, float3 v) { return quatRotate(q, v) + p; }

// ---------------------------------------------------------------------------
// Warp emulation scratch.
// ---------------------------------------------------------------------------
groupshared uint sCgBallot;
groupshared uint sCgScanStart;

uint cgBallot(bool pred, uint lane)
{
    if (lane == 0u) sCgBallot = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sCgBallot, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sCgBallot;
}

// PxgFEMContactWriter::writeRigidVsDeformableContact32 -- the same five stores the
// mesh sibling makes, kept here rather than in the shared module because the
// writer is the caller's glue and its pack differs between the two kernels.
void writeRigidVsDeformableContact32(uint index, float4 contact, float4 normalPen, float4 bary,
                                     uint2 pairInd0, uint pairInd1, uint rigidMatIndex,
                                     uint rigidX)
{
    if (index >= gCg.w_maxNumContacts)
        return;
    Ptr<float4>(gCg.w_outPoint)[index] = contact;
    Ptr<float4>(gCg.w_outNormalPen)[index] = normalPen;
    Ptr<float4>(gCg.w_outBarycentric)[index] = bary;
    // PxgFemOtherContactInfo: uint4 { pairInd0.lo, pairInd0.hi, pairInd1, materialIndex }
    Ptr<uint4>(gCg.w_outContactInfo)[index] =
        uint4(pairInd0.x, pairInd0.y, pairInd1, rigidMatIndex);
    Ptr<uint>(gCg.w_contactByX)[index] = rigidX;
}

// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void cloth_heightfieldContactGenLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    const uint stackSize = gCg.stackSizeBytes / 16u;
    const uint numMidphasePairsFound = Ptr<uint>(gCg.midphasePairsNum)[0];
    const uint numPairs = min(numMidphasePairsFound, stackSize);
    const uint l = ((numPairs + WARP - 1u) / WARP) * WARP;

    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gCg.femClothes);
    Ptr<float> contactDistance = Ptr<float>(gCg.contactDistance);
    Ptr<float> restDistance = Ptr<float>(gCg.restDistance);
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gCg.gpuShapes);

    for (uint sub = 0u; sub < HF_CG_WARPS_PER_BLOCK; ++sub)
    {
    const uint warpBase = gid.x * (HF_CG_WARPS_PER_BLOCK * WARP) + sub * WARP;

    for (uint base = warpBase; base < l; base += HF_CG_STRIDE)
    {
        const uint i = base + lane;

        bool hasContact = false;
        uint cmIdx = 0u;
        uint clothId = 0u;
        uint clothTriangleIdx = 0u;
        uint2 rigidId = uint2(0u, 0u);
        uint rigidMaterialIndex = 0xFFFFFFFFu;
        FemContact c;
        c.position = float3(0, 0, 0);
        c.normal = float3(0, 0, 0);
        c.separation = 0.0f;
        c.barycentric = float4(0, 0, 0, 0);

        if (i < numPairs)
        {
            const uint4 curPair = Ptr<uint4>(gCg.stackPtr)[i];
            cmIdx = curPair.x;
            clothTriangleIdx = curPair.y;             // the CLOTH triangle
            const uint hfTriangleIdx = curPair.z;     // the HEIGHTFIELD triangle

            const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(gCg.cmInputs)[cmIdx];
            // C1: POSITIONAL, not routed by body id.  See the header.
            const PxgShape clothShape = shapes[cin.shapeRef0];
            const PxgShape hfShape = shapes[cin.shapeRef1];
            const uint clothCacheRef = cin.transformCacheRef0;
            const uint hfCacheRef = cin.transformCacheRef1;

            rigidMaterialIndex = hfShape.materialIndex;
            rigidId = Ptr<uint2>(gCg.shapeToRigidRemap)[hfCacheRef];   // PxNodeIndex
            clothId = clothShape.particleOrSoftbodyId;

            const float cDistance = contactDistance[clothCacheRef] + contactDistance[hfCacheRef];

            // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.
            const uint64_t xfHf = gCg.transformCache + uint64_t(hfCacheRef) * 32UL;
            const float4 hfQ = Ptr<float4>(xfHf)[0];
            const float3 hfP = float3(Ptr<float>(xfHf + 16UL)[0],
                                      Ptr<float>(xfHf + 20UL)[0],
                                      Ptr<float>(xfHf + 24UL)[0]);
            const uint64_t xfCloth = gCg.transformCache + uint64_t(clothCacheRef) * 32UL;
            const float4 clothQ = Ptr<float4>(xfCloth)[0];
            const float3 clothP = float3(Ptr<float>(xfCloth + 16UL)[0],
                                         Ptr<float>(xfCloth + 20UL)[0],
                                         Ptr<float>(xfCloth + 24UL)[0]);

            // The heightfield blob: nbRows @0, nbCols @4, then the samples.
            const uint64_t hfPtr = uint64_t(hfShape.hullOrMeshOffset.x) |
                                   (uint64_t(hfShape.hullOrMeshOffset.y) << 32);
            const uint nbCols = Ptr<uint>(hfPtr + 4UL)[0];
            const uint64_t samples = hfPtr + 8UL;

            // C2: already in the heightfield's SHAPE space -- no vertex2Shape.
            float3 a0, b0, c0;
            hfGetTriangle(samples, hfTriangleIdx, hfShape.scale(), nbCols, a0, b0, c0);

            // C4: hfToCloth = clothTransform.transformInv(hfTransform)
            const PxgTransform28 clothT = makeTransform28(clothQ, clothP);
            const PxgTransform28 hfT = makeTransform28(hfQ, hfP);
            const PxgTransform28 hfToCloth = transformInvTransform(clothT, hfT);

            const float3 localA0 = transformPt(hfToCloth.q(), hfToCloth.p(), a0);
            const float3 localB0 = transformPt(hfToCloth.q(), hfToCloth.p(), b0);
            const float3 localC0 = transformPt(hfToCloth.q(), hfToCloth.p(), c0);

            // C3: no filter -- every pair the midphase produced is tested.
            const uint4 clothVertInd =
                Ptr<uint4>(clothes[clothId].mTriangleVertexIndices)[clothTriangleIdx];
            Ptr<float4> clothPos = Ptr<float4>(clothes[clothId].mPosition_InvMass);
            const float3 a1 = clothPos[clothVertInd.x].xyz;
            const float3 b1 = clothPos[clothVertInd.y].xyz;
            const float3 c1 = clothPos[clothVertInd.z].xyz;

            hasContact = staticTriangleVsClothCollision(localA0, localB0, localC0, a1, b1, c1,
                                                        clothQ, clothP, cDistance, c);
        }

        // globalScanExclusiveSingleWarp(hasContact, writer.totalContactCount).
        const uint mask = cgBallot(hasContact, lane);
        const uint offset = countbits(mask & ((1u << lane) - 1u));
        const uint validCount = countbits(mask);
        if (lane == 0u)
        {
            uint start = 0u;
            if (validCount > 0u)
                InterlockedAdd(Ptr<uint>(gCg.w_totalContactCount)[0], validCount, start);
            sCgScanStart = start;
        }
        GroupMemoryBarrierWithGroupSync();
        const uint index = sCgScanStart + offset;

        if (hasContact)
        {
            writeRigidVsDeformableContact32(index,
                                            float4(c.position, restDistance[cmIdx]),
                                            float4(c.normal, c.separation),
                                            c.barycentric,
                                            rigidId,
                                            (clothId << PXG_BITSHIFT_ELEMENT_ID) | clothTriangleIdx,
                                            rigidMaterialIndex,
                                            rigidId.x);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    }

    // The stack high-water mark, so the engine can grow the shared midphase stack.
    // UNCLAMPED, deliberately: a value that cannot exceed the stack it is meant to
    // grow could never ask for a bigger one.
    if (gid.x == 0u && lane == 0u)
        InterlockedMax(Ptr<uint>(gCg.stackSizeNeededOnDevice)[0], numMidphasePairsFound * 16u);
}

// ---------------------------------------------------------------------------
// The DEAD kernel's push-constant blob (PxgNarrowphaseCore.cpp:7157).  The SAME
// thirteen arguments as the live one, in a DIFFERENT ORDER: the writer comes
// BEFORE stackSizeBytes and stackSizeNeededOnDevice rather than after them.
//
// Both packs are 176 bytes.  Reading one with the other's layout hands the kernel
// a byte count where a device pointer belongs and vice versa -- which is exactly
// why the two get separate ABI rows and why this comment names the difference
// rather than leaving the reader to diff two structs.
// ABI row { 4,8,8,8,8,8,8,8,8,8,80,4,8 }.
// ---------------------------------------------------------------------------
struct ClothHfVertCgParams
{
    float toleranceLength;         // @0   (ABI parity only)
    uint _pad4;                    // @4
    uint64_t cmInputs;             // @8   (ABI parity only)
    uint64_t transformCache;       // @16  (ABI parity only)
    uint64_t contactDistance;      // @24  (ABI parity only)
    uint64_t restDistance;         // @32  (ABI parity only)
    uint64_t gpuShapes;            // @40  (ABI parity only)
    uint64_t femClothes;           // @48  (ABI parity only)
    uint64_t shapeToRigidRemap;    // @56  (ABI parity only)
    uint64_t stackPtr;             // @64  (ABI parity only)
    uint64_t midphasePairsNum;     // @72
    // ---- PxgFEMContactWriter (by value, 80 B) @80 ----
    uint64_t w_outPoint;           // @80   (ABI parity only)
    uint64_t w_outNormalPen;       // @88   (ABI parity only)
    uint64_t w_outBarycentric;     // @96   (ABI parity only)
    uint64_t w_outContactInfo;     // @104  (ABI parity only)
    uint64_t w_totalContactCount;  // @112  (never incremented -- see the header)
    uint64_t w_contactByX;         // @120  (ABI parity only)
    uint64_t w_tempContactByX;     // @128  (ABI parity only)
    uint64_t w_contactIndexSortedByX; // @136 (ABI parity only)
    uint64_t w_contactSortedByX;   // @144  (ABI parity only)
    uint w_maxNumContacts;         // @152
    uint _pad156;                  // @156 -> 160
    uint stackSizeBytes;           // @160  (ABI parity only -- see below)
    uint _pad164;                  // @164
    uint64_t stackSizeNeededOnDevice; // @168 -> 176
}

[[vk::push_constant]] ConstantBuffer<ClothHfVertCgParams> gVc;

// ===========================================================================
// The kernel's ONE observable output.  Everything else it computes is discarded
// upstream, and the header says why that is transcribed as absence rather than as
// two hundred unverifiable lines.  The grid-stride loop is described there too,
// and deliberately not written: P9.26's sweep found `stackSizeBytes / 16 -> / 4`
// SURVIVING against an empty walk, because the clamp fed only the bound of a loop
// with no body.  Code that cannot be tested does not get to sit here looking
// tested.
// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void cloth_heightfieldVertexContactGenLaunch(uint3 gtid : SV_GroupThreadID,
                                             uint3 gid : SV_GroupID)
{
    const uint numFound = Ptr<uint>(gVc.midphasePairsNum)[0];
    if (gid.x == 0u && gtid.x == 0u)
        InterlockedMax(Ptr<uint>(gVc.stackSizeNeededOnDevice)[0], numFound * 16u);
}
