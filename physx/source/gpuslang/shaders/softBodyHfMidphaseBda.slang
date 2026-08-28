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
// softBodyHfMidphaseBda — sb_heightfieldMidphaseGeneratePairsLaunch
// (gpunarrowphase/src/CUDA/softbodyHFMidPhase.cu:281, core `sbHeightfieldMidphaseCore` at
// :173, launch PxgNarrowphaseCore.cpp:5341, SB_HF_MIDPHASE), on the engine BDA ABI.
//
// Step 1 of TWO for softbody-vs-HEIGHTFIELD.  Step 2 (`sb_heightfieldContactGenLaunch`) runs
// GJK/EPA per pair and writes into the RIGID contact stream, exactly as the trimesh path's step
// 2 does -- same helper, same `flipOutputNormal = true` / `alternativeWriteMode = true`
// (softbodyPrimitives.cu:1136).  Everything after that is already mapped.
//
// *** THERE IS NO BVH HERE. ***  Every other midphase in this port walks a BV32 tree; a
// heightfield has none.  Its geometry IS its index: the tetrahedron's world AABB is pushed into
// the heightfield's local, unscaled ROW/COLUMN space, and the integer rectangle it lands on
// names the candidate quads directly.  So the whole traversal is four clamps and a double loop,
// and none of the BV32 machinery appears.
//
// THE COOKED BLOB is likewise flat (`heightfieldUtil.cuh`, and the core at :236):
//     PxU32 nbRows @0 | PxU32 nbCols @4 | PxHeightFieldSample samples[nbRows * nbCols] @8
// and a sample is FOUR BYTES: `PxI16 height` @0, `PxBitAndByte materialIndex0` @2,
// `materialIndex1` @3.  The material bytes carry a FLAG IN BIT 7 (`tessFlag`, which way the
// quad is split), so a material comparison has to mask it off -- `PxBitAndDataT::operator T()`
// returns `mData & ~bitMask`.  Comparing the raw byte against eHOLE = 127 silently treats a
// tessellated material-127 quad as solid and an untessellated one as a hole.
//
// WHAT THE CORE DOES, in order:
//   1. the tetrahedron's world AABB, FATTENED by the contact distance (`fattenFast`);
//   2. that AABB through the heightfield's INVERSE transform -- `PxBounds3::transformFast`,
//      i.e. `basisExtent(inv.transform(centre), PxMat33(inv.q), extents)`, the rotated-basis
//      AABB rather than a corner sweep;
//   3. divided by |rowScale| / heightScale / |colScale|, WITH A MIN/MAX SWAP on each axis whose
//      scale is NEGATIVE.  A mirrored heightfield is a real configuration and the swap is the
//      only thing that keeps the row range the right way round;
//   4. clamped to a row/column rectangle (`getMinRow` floors to [0, rows-2], `getMaxRow` ceils
//      to [0, rows-1] -- note the DIFFERENT upper clamps, one of which leaves room for the
//      quad's far corner);
//   5. every quad in that rectangle contributes TWO candidate triangles, one per material slot.
//      A quad is skipped when the tet's y-range lies entirely above or entirely below all four
//      of its corner heights, and a triangle is skipped when its material is eHOLE;
//   6. the surviving triangle index is `(vertexIndex << 1) | isMaterial1`.
//
// THE Y COMPARISON IS DONE IN INTEGERS UPSTREAM, and that is transcribed rather than corrected:
// `heightfieldComputePairs` takes `miny`/`maxy` as `PxU32` (heightfieldUtil.cuh:334) and the
// caller passes the local bound's float y through that conversion, which the predicates then
// compare back as floats.
//
// THE CONVERSION SATURATES.  CUDA compiles it to `cvt.rzi.u32.f32`, which clamps out-of-range
// results, so a bound at y = -0.4 arrives as **0** -- NOT as a huge unsigned number, which is
// what the C++ HOST would produce and what an earlier version of this comment claimed.  The
// difference is not academic: minY = 0 makes `con1 = minY > every corner` false for any quad
// with a non-negative corner and the quad is KEPT, while a huge minY makes it true and the
// quad is DROPPED.  Anything resting on terrain has a bound reaching below local y = 0.
//
// This was measured, not read: the fixture's T_NEGY tetrahedron reaches lMin.y = -0.1 and
// emits two pairs, which only the saturating reading produces.  Before that case existed every
// tetrahedron in the fixture sat at lMin.y = +0.8 and the branch was unreachable, so the
// fixture passed identically with and without saturation -- also measured.
//
// The conversion is spelled explicitly (`hfF2uSat`) rather than as a bare `uint(x)`.  A bare
// cast is `OpConvertFToU`, which SPIR-V leaves UNDEFINED for an out-of-range operand: llvmpipe
// happens to give 0 and therefore happens to agree, but agreement by accident is not the same
// as agreement, and the next driver need not repeat it.  fcMidphase.slang made this call
// correctly in Phase 1 (`f2uSat`, with a CPU mirror in its fixture); this file did not, and
// the two disagreeing about the same upstream conversion is what found it.
//
// *** THE WORK PARTITION IS NOT THE USUAL ONE. ***  Sixty-four warps are assigned per contact
// manager (`nbWarpsPerCm = 64`, i.e. 2048 threads), and inside one, EACH LANE owns a different
// tetrahedron:
//     workIndex          = i + (warpIndex + 2 * blockIdx.x) * nbIterationsPerWarps
//     wrappedThreadIndex = (workIndex * 32 + threadIdx.x) % 2048
//     tetrahedronIdx     = wrappedThreadIndex + 2048 * j
// The lanes then warp-scan their pair counts so the warp does ONE atomicAdd and each lane
// writes at its own offset.  This port keeps all of it, with the scan emulated in groupshared.
//
// `numTets` COMES OUT OF THE TETMESH BLOB HEADER (`mTetMeshData`'s uint4 `.y`), not from
// `PxgSoftBody::mNumTets`.  They agree in practice; the blob is what upstream reads.
//
// DEVIATIONS from CUDA: no 32-lane intrinsics (the warp scan becomes a groupshared prefix sum,
// the broadcast a groupshared slot); the per-warp scratch struct is dropped and every lane
// reads the addresses directly.  PxU8/PxI16 fields are read through the containing 32-bit word,
// with the word address formed as `& ~uint64_t(3)` -- writing `& ~3UL` MISCOMPILES, because
// Slang evaluates the complement in 32 bits and clears the top half of the device address.
//
// NoContraction-stamped (PXG_NOCONTRACT_SHADERS): basisExtent's three scaled basis columns and
// the quaternion rotations feeding them are mul-then-add chains that would otherwise fuse to
// FMAs and move a bound by an ULP -- which is enough to change a floor()/ceil() and therefore
// the emitted pair SET, not merely its arithmetic.
// ===========================================================================
module softBodyHfMidphaseBda;

import pxgSlangSoftBodyDesc;   // PxgSoftBodyGpu
import pxgSlangHeightfield;   // P9.27: the sample format, the rectangle, the quad walk

// PxgNpKernelIndices.h:32 / PxgSoftBodyCoreKernelIndices.h:55.
static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;
static const uint SB_SBMIDPHASE = 1024u;
static const uint TOTAL_NUM_WARPS = SB_SBMIDPHASE * MIDPHASE_WARPS_PER_BLOCK;   // 2048

static const uint WARP = 32u;
static const uint NB_WARPS_PER_CM = 64u;                 // softbodyHFMidPhase.cu:301
static const uint NB_THREADS_PER_CM = NB_WARPS_PER_CM * WARP;   // 2048

static const uint GEOM_TETRAHEDRONMESH = 7u;

// ---------------------------------------------------------------------------
// Push-constant blob — BYTE-IDENTICAL to the CUDA pack (PxgNarrowphaseCore.cpp:5343):
// { PxU32 numNPWorkItems, PxReal toleranceLength, cmInputs*, transformCache*,
//   contactDistance*, gpuShapes*, softbodies*, PxU32 stackSizeBytes, stackPtr*,
//   midphasePairsNum* }
// -> 4 @0, 4 @4, five pointers @8..@40, PxU32 @48, then a pointer realigned to @56 and @64
// -> 72-byte blob.  ABI row { 4,4,8,8,8,8,8,4,8,8 }.
// ---------------------------------------------------------------------------
struct SbHfMidphaseParams
{
    uint numNPWorkItems;            // @0
    float toleranceLength;          // @4   (ABI parity only — the core never reads it)
    uint64_t cmInputs;              // @8
    uint64_t transformCache;        // @16
    uint64_t contactDistance;       // @24
    uint64_t gpuShapes;             // @32
    uint64_t softbodies;            // @40
    uint stackSizeBytes;            // @48
    uint _pad52;                    // @52
    uint64_t stackPtr;              // @56
    uint64_t midphasePairsNum;      // @64 -> 72
}

[[vk::push_constant]] ConstantBuffer<SbHfMidphaseParams> gHf;

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

// A PxU8 array read through the containing 32-bit word.  See the header on `& ~uint64_t(3)`.
uint byteAt(uint64_t base, uint idx)
{
    const uint64_t addr = base + uint64_t(idx);
    const uint word = Ptr<uint>(addr & ~uint64_t(3))[0];
    const uint shift = uint(addr & 3UL) * 8u;
    return (word >> shift) & 0xffu;
}

// The sample format, the row/column clamps, the saturating y conversion, the admit
// gate and the quad walk all live in pxgSlangHeightfield now (P9.27).  They used to
// live here AND in fcMidphase.slang, and the two copies disagreed about the
// conversion for several tranches -- see P9.27a.

// PxQuat::rotate / rotateInv (PxQuat.h:286, :324), literal.
float3 hfQuatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}
float3 hfQuatRotateInv(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}

// LoadShapePairWarp<eTETRAHEDRONMESH, eHEIGHTFIELD>.
void loadSoftbodyHfPair(uint cmIdx, out PxgShape sbShape, out uint sbCacheRef,
                        out PxgShape hfShape, out uint hfCacheRef)
{
    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(gHf.cmInputs)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gHf.gpuShapes);
    uint sRefSb = cin.shapeRef0, cRefSb = cin.transformCacheRef0;
    uint sRefHf = cin.shapeRef1, cRefHf = cin.transformCacheRef1;
    if (shapes[cin.shapeRef0].type != GEOM_TETRAHEDRONMESH)
    {
        sRefSb = cin.shapeRef1; cRefSb = cin.transformCacheRef1;
        sRefHf = cin.shapeRef0; cRefHf = cin.transformCacheRef0;
    }
    sbShape = shapes[sRefSb]; sbCacheRef = cRefSb;
    hfShape = shapes[sRefHf]; hfCacheRef = cRefHf;
}

// ---------------------------------------------------------------------------
// Warp emulation scratch: one group == one warp.
// ---------------------------------------------------------------------------
groupshared uint sHfCount[WARP];   // per-lane pair counts, for the exclusive scan
groupshared uint sHfStart;         // the warp's single atomicAdd result

// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void sb_heightfieldMidphaseGeneratePairsLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    const uint nbWarpsRequired = NB_WARPS_PER_CM * gHf.numNPWorkItems;
    const uint nbIterationsPerWarps = (nbWarpsRequired + TOTAL_NUM_WARPS - 1u) / TOTAL_NUM_WARPS;
    const uint stackSize = gHf.stackSizeBytes / 16u;

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gHf.softbodies);

    // The bridge drops blockDim, so the block's two warps are walked serially and `warpIndex`
    // is the loop variable rather than threadIdx.y.
    for (uint warpIndex = 0u; warpIndex < MIDPHASE_WARPS_PER_BLOCK; ++warpIndex)
    {
    for (uint i = 0u; i < nbIterationsPerWarps; ++i)
    {
        const uint workIndex = i + (warpIndex + MIDPHASE_WARPS_PER_BLOCK * gid.x) * nbIterationsPerWarps;
        if (workIndex >= nbWarpsRequired)
            continue;

        const uint cmIdx = workIndex / NB_WARPS_PER_CM;

        PxgShape sbShape, hfShape;
        uint sbCacheRef, hfCacheRef;
        loadSoftbodyHfPair(cmIdx, sbShape, sbCacheRef, hfShape, hfCacheRef);

        const uint softbodyId = sbShape.particleOrSoftbodyId;

        // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.
        const uint64_t xf = gHf.transformCache + uint64_t(hfCacheRef) * 32UL;
        const float4 hfQ = Ptr<float4>(xf)[0];
        const float3 hfP = float3(Ptr<float>(xf + 16UL)[0],
                                  Ptr<float>(xf + 20UL)[0],
                                  Ptr<float>(xf + 24UL)[0]);

        // numTets comes out of the TETMESH BLOB HEADER, not mNumTets (see the header).
        const uint64_t tetMeshData = softbodies[softbodyId].mTetMeshData;
        const uint nbTets = Ptr<uint4>(tetMeshData)[0].y;

        Ptr<uint4> tetIndices = Ptr<uint4>(softbodies[softbodyId].mTetIndices);
        Ptr<float4> tetVerts = Ptr<float4>(softbodies[softbodyId].mPosition_InvMass);
        const uint64_t surfaceHintBase = softbodies[softbodyId].mTetMeshSurfaceHint;

        const float contactDist = Ptr<float>(gHf.contactDistance)[sbCacheRef] +
                                  Ptr<float>(gHf.contactDistance)[hfCacheRef];

        // The flat heightfield blob: nbRows @0, nbCols @4, samples @8.
        const uint64_t hfPtr = uint64_t(hfShape.hullOrMeshOffset.x) |
                               (uint64_t(hfShape.hullOrMeshOffset.y) << 32);
        const uint nbRows = Ptr<uint>(hfPtr)[0];
        const uint nbCols = Ptr<uint>(hfPtr + 4UL)[0];
        const uint64_t samples = hfPtr + 8UL;

        const float oneOverHeightScale = 1.0f / hfShape.scaleY;
        const float oneOverRowScale = 1.0f / abs(hfShape.scaleX);
        const float oneOverColScale = 1.0f / abs(hfShape.scaleZ);

        const uint nbIter = (nbTets + NB_THREADS_PER_CM - 1u) / NB_THREADS_PER_CM;

        for (uint j = 0u; j < nbIter; ++j)
        {
            const uint wrappedThreadIndex = (workIndex * WARP + lane) % NB_THREADS_PER_CM;
            const uint tetrahedronIdx = wrappedThreadIndex + NB_THREADS_PER_CM * j;

            // --- the per-lane rectangle -----------------------------------------
            uint nbPairs = 0u;
            HfRect rect;
            rect.live = false;
            rect.minRow = 0u; rect.maxRow = 0u; rect.minColumn = 0u; rect.maxColumn = 0u;
            rect.nbCols = nbCols; rect.minyF = 0.0f; rect.maxyF = 0.0f;

            if (tetrahedronIdx < nbTets && byteAt(surfaceHintBase, tetrahedronIdx) != 0u)
            {
                const uint4 tetIdx = tetIndices[tetrahedronIdx];
                const float3 w0 = tetVerts[tetIdx.x].xyz;
                const float3 w1 = tetVerts[tetIdx.y].xyz;
                const float3 w2 = tetVerts[tetIdx.z].xyz;
                const float3 w3 = tetVerts[tetIdx.w].xyz;

                // tetBoundingBox, then fattenFast(contactDist).
                float3 bMin = min(min(w0, w1), min(w2, w3)) - contactDist;
                float3 bMax = max(max(w0, w1), max(w2, w3)) + contactDist;

                // transformFast + the per-axis divisions + the negative-scale swaps:
                // pxgSlangHeightfield::hfLocalBound.  Then the admit gate, the clamps
                // and the saturating y conversion: hfMakeRect.
                float3 lMin, lMax;
                hfLocalBound(hfQ, hfP, float3(hfShape.scaleX, hfShape.scaleY, hfShape.scaleZ),
                             bMin, bMax, lMin, lMax);
                rect = hfMakeRect(lMin, lMax, nbRows, nbCols);
            }

            // --- heightfieldComputePairs (heightfieldUtil.cuh:334) ----------------
            {
                const uint total = hfQuadSpan(rect);
                for (uint t = 0u; t < total; ++t)
                    if (hfQuadAt(samples, rect, t).survives)
                        ++nbPairs;
            }

            // --- the warp scan: one atomicAdd, each lane writes at its own offset ---
            GroupMemoryBarrierWithGroupSync();
            sHfCount[lane] = nbPairs;
            GroupMemoryBarrierWithGroupSync();
            uint pairOffset = 0u;
            for (uint k = 0u; k < lane; ++k) pairOffset += sHfCount[k];
            uint validCount = 0u;
            for (uint k = 0u; k < WARP; ++k) validCount += sHfCount[k];

            if (validCount == 0u)
                continue;

            if (lane == 0u)
            {
                uint idx;
                InterlockedAdd(Ptr<uint>(gHf.midphasePairsNum)[0], validCount, idx);
                sHfStart = idx;
            }
            GroupMemoryBarrierWithGroupSync();
            uint writeIndex = sHfStart + pairOffset;

            // --- heightfieldOutputPairs (heightfieldUtil.cuh:382) -----------------
            // The SAME walk again, emitting this time.  The `writeIndex < size` guard is in
            // the loop condition upstream, so a full stack stops the WRITES but the count
            // above was already committed -- which is how the host learns it needs more.
            {
                const uint total = hfQuadSpan(rect);
                for (uint t = 0u; t < total && writeIndex < stackSize; ++t)
                {
                    const HfQuad q = hfQuadAt(samples, rect, t);
                    if (q.survives)
                    {
                        Ptr<uint4>(gHf.stackPtr)[writeIndex] =
                            uint4(cmIdx, tetrahedronIdx, q.triangleIdx, 0u);
                        ++writeIndex;
                    }
                }
            }
        }
    }
    }
}
