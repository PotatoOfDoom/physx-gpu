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

// femClothHfMidphaseBda.slang -- P9.27
//
// cloth_heightfieldMidphaseGeneratePairsLaunch (femClothHFMidPhase.cu:185) and
// cloth_midphaseVertexHeightfieldLaunch (:296), both read in full, together with
// the clothHeightfieldMidphaseCore (:76) they share.
//
// ONE BODY, TWO ENTRY POINTS -- AND HERE THAT IS UPSTREAM'S OWN STRUCTURE.
//
// P9.20 gave two engine names one Slang body on the strength of a measured
// equivalence, and had to add tools/check_cuda_twin_kernels.py to keep that
// equivalence true.  This case needs no such check: the CUDA itself declares one
// `__device__` function and calls it from both kernels.  The two differ in
// exactly two things, and this file parameterises exactly those two:
//
//   * the ELEMENT bound -- a cloth TRIANGLE's bound, fattened by the contact
//     distance, against a SPHERE bound of that same distance around a cloth
//     VERTEX;
//   * the COUNT the element index is compared against -- mNbTriangles against
//     mNbVerts.
//
// Everything downstream (the transform into heightfield units, the admit gate,
// the row/column clamps, the saturating y conversion, the quad walk, the warp
// reservation and the payload) is one code path, as it is upstream.
//
// ---------------------------------------------------------------------------
// THREE TRAPS, one of them a defect in the engine
// ---------------------------------------------------------------------------
// H1 -- THE VERTEX KERNEL READS positions[vertIndex] WITHOUT A BOUNDS CHECK.
//     The triangle sibling guards its element load with
//     `if (triangleIdx < nbTriangles)` (:264); the vertex one does not (:376),
//     and the guard inside the shared core comes AFTER the load has already
//     happened.  `nbIter` rounds the vertex count up to a multiple of 2048, so
//     every cloth whose vertex count is not a multiple of 2048 reads past the
//     end of mPosition_InvMass -- which is most of them.
//
//     On CUDA that is a benign over-read into whatever follows.  On THIS
//     platform it is the P9.22 fault class: llvmpipe turns a per-lane guard into
//     a mask and issues the load anyway, and the address is a raw process
//     address, so an out-of-range index is a segfault and not a wasted load.
//     The port therefore CLAMPS THE INDEX and leaves the value's effect
//     unchanged -- the bound built from it is discarded by the same
//     `elementIdx < numElements` gate that discards it upstream, so clamping
//     changes no output and removes the fault.  That is the fix pattern P9.22
//     established.
//
//     NO CHECK COVERS IT, and the reason is more interesting than a gap.
//     Removing the clamp and running tools/check_speculated_loads.py reports
//     NOTHING, and the first reading of that was mine being wrong: I took it for
//     a blind spot, on the grounds that the index is `wrappedThreadIndex +
//     2048 * j` and so overshoots by a full STRIDE -- 2047 float4s, 32 KiB --
//     rather than by the workgroup the LOW-launch tier assumes.
//
//     Then I built the rule that would catch it and ran it over the corpus.  It
//     fires on THIRTY-SIX sites, and they are the ordinary grid-stride loop that
//     this port is made of: `workIndex = it * BLOCK + tid + ...` overshoots by
//     one BLOCK in exactly the same way.  This site is not a special hazard; it
//     is the same shape as every strided loop here, and the checker already
//     names that shape LOW-loop with the reason "overshoot <= one stride".
//
//     So the claim was withdrawn rather than shipped.  Whether a one-stride
//     overshoot is acceptable AT ALL is a real question, but it is a question
//     about the whole corpus and about what llvmpipe actually does with a fully
//     masked vector -- not about this shader.  P9.22's crash was a different
//     tier: an index bounded only by a PARAMETER, with nothing in the function
//     relating it to the buffer, which the checker still reports HIGH.
//
//     The clamp stays regardless.  It costs one min(), it removes an over-read
//     the CUDA performs and this platform need not, and no fixture or mutant can
//     see it either way -- which is why it is written down here.
//
//     Upstream candidate 36; documented, NOT reported.
//
// H2 -- THE GRID IS LAUNCHED WITH gridDim.y = numTests AND NEITHER KERNEL READS
//     blockIdx.y.  Verified by reading every blockIdx in the file: the work
//     partition is
//         workIndex = i + (warpIndex + 2 * blockIdx.x) * nbIterationsPerWarps
//     and .y appears nowhere in either kernel or in the shared core.  Every
//     y-block therefore repeats the WHOLE partition, so with numTests contact
//     managers every pair is emitted numTests times and midphasePairsNum
//     advances numTests times as far.
//
//     The softbody twin, with the identical nbWarpsPerCm = 64 scheme, is
//     launched with gridDim.y = 1 (PxgNarrowphaseCore.cpp:5360).  That is the
//     evidence this is a mistake and not a design -- and it is invisible with a
//     single cloth-vs-heightfield pair, which is why nobody has hit it.
//
//     PORTED BUG-FOR-BUG.  The consumer contact-gen reads the duplicated pairs
//     and the counter, so "fixing" it here would make this port disagree with
//     the engine it is replacing.  The fixture drives numTests = 2 and MEASURES
//     the duplication rather than describing it.
//
//     Upstream candidate 37; documented, NOT reported.
//
// H3 -- THE WORK PARTITION IS NOT THE USUAL ONE, and it is the softbody
//     heightfield midphase's, not the mesh midphase's.  Sixty-four warps are
//     assigned per contact manager (2048 threads), and within one warp EACH LANE
//     owns a different element:
//         wrappedThreadIndex = (workIndex * 32 + threadIdx.x) % 2048
//         elementIdx         = wrappedThreadIndex + 2048 * j
//     The lanes then warp-scan their pair counts so the warp performs ONE
//     atomicAdd and each lane writes at its own offset.  A port that assumed the
//     familiar one-element-per-warp scheme would emit the right pairs in the
//     wrong order and the wrong count.
//
// ---------------------------------------------------------------------------
// WARP EMULATION: the bridge forwards gridDim and drops blockDim, so a
// [numthreads(32,1,1)] group is ONE warp and the block's two warps are walked
// serially in a group-uniform `sub` loop.  The warp scan becomes a groupshared
// prefix sum and the lane-0 broadcast a groupshared slot.  Both loops are over
// group-uniform quantities (nbIterationsPerWarps, nbIter), so nothing here puts
// a barrier in divergent control flow.
// ===========================================================================
module femClothHfMidphaseBda;

import pxgSlangFEMClothDesc;   // PxgFEMClothGpu
import pxgSlangHeightfield;    // P9.27: format, bound transform, rectangle, quad walk

// PxgNpKernelIndices.h:32 / PxgSoftBodyCoreKernelIndices.h:55.
static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;
static const uint SB_SBMIDPHASE = 1024u;
static const uint TOTAL_NUM_WARPS = SB_SBMIDPHASE * MIDPHASE_WARPS_PER_BLOCK;   // 2048

static const uint WARP = 32u;
static const uint NB_WARPS_PER_CM = 64u;                        // femClothHFMidPhase.cu:206
static const uint NB_THREADS_PER_CM = NB_WARPS_PER_CM * WARP;   // 2048

// ---------------------------------------------------------------------------
// Push-constant blob -- BYTE-IDENTICAL to the CUDA pack, and the SAME pack for
// both kernels (PxgNarrowphaseCore.cpp:6909 and :7060 list the same ten
// arguments in the same order):
// { PxU32 numNPWorkItems, PxReal toleranceLength, cmInputs*, transformCache*,
//   contactDistance*, gpuShapes*, clothes*, stackPtr*, midphasePairsNum*,
//   PxU32 stackSizeBytes }
// -> 4 @0, 4 @4, seven pointers @8..@56, PxU32 @64 -> 72-byte blob.
// ABI row { 4,4,8,8,8,8,8,8,8,4 }.
//
// NOT kSbHfMidphase { 4,4,8,8,8,8,8,4,8,8 }, even though both are 72 bytes and
// carry the same ten types: there stackSizeBytes sits at @48 BEFORE the two
// output pointers, here it sits at @64 AFTER them.  Reading one with the other's
// layout hands the kernel a byte count where a device pointer belongs.
// ---------------------------------------------------------------------------
struct ClothHfMidphaseParams
{
    uint numNPWorkItems;            // @0
    float toleranceLength;          // @4   (ABI parity only -- the core never reads it)
    uint64_t cmInputs;              // @8
    uint64_t transformCache;        // @16
    uint64_t contactDistance;       // @24
    uint64_t gpuShapes;             // @32
    uint64_t femClothes;            // @40
    uint64_t stackPtr;              // @48  output
    uint64_t midphasePairsNum;      // @56  output
    uint stackSizeBytes;            // @64
    uint _pad68;                    // @68 -> 72
}

[[vk::push_constant]] ConstantBuffer<ClothHfMidphaseParams> gHf;

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

static const uint GEOM_TRIANGLEMESH = 8u;   // PxGeometryType::eTRIANGLEMESH

// LoadShapePair<eTRIANGLEMESH, eHEIGHTFIELD> (dataReadWriteHelper.cuh): the pair
// is ordered by TYPE, not by body id -- unlike the cloth-vs-cooked-mesh case,
// where BOTH shapes are eTRIANGLEMESH and only the id can tell them apart.
void loadClothHfPair(uint64_t cmInputsAddr, uint64_t shapesAddr, uint cmIdx,
                     out PxgShape clothShape, out uint clothCacheRef,
                     out PxgShape hfShape, out uint hfCacheRef)
{
    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(cmInputsAddr)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(shapesAddr);
    const PxgShape s0 = shapes[cin.shapeRef0];
    const PxgShape s1 = shapes[cin.shapeRef1];
    if (s0.type == GEOM_TRIANGLEMESH)
    {
        clothShape = s0; clothCacheRef = cin.transformCacheRef0;
        hfShape = s1;    hfCacheRef = cin.transformCacheRef1;
    }
    else
    {
        clothShape = s1; clothCacheRef = cin.transformCacheRef1;
        hfShape = s0;    hfCacheRef = cin.transformCacheRef0;
    }
}

// ---------------------------------------------------------------------------
// Warp emulation scratch: one group == one warp, so a single row each.
// ---------------------------------------------------------------------------
groupshared uint sHfCount[WARP];
groupshared uint sHfStart;

// ===========================================================================
// The shared body.  `vertexQuery` selects the two things the CUDA's two callers
// differ in and nothing else -- see the header.
// ===========================================================================
void clothHfMidphaseBody(uint3 gtid, uint3 gid, bool vertexQuery)
{
    const uint lane = gtid.x;

    const uint stackSize = gHf.stackSizeBytes / 16u;
    const uint nbWarpsRequired = NB_WARPS_PER_CM * gHf.numNPWorkItems;
    const uint nbIterationsPerWarps =
        (nbWarpsRequired + TOTAL_NUM_WARPS - 1u) / TOTAL_NUM_WARPS;

    // H2: gid.y is NOT read.  The engine launches gridDim.y = numTests and the
    // CUDA ignores it, so every y-block repeats this whole partition and every
    // pair is emitted numTests times.  Reproduced deliberately.
    for (uint sub = 0u; sub < MIDPHASE_WARPS_PER_BLOCK; ++sub)
    {
    const uint warpIndex = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub;

    for (uint i = 0u; i < nbIterationsPerWarps; ++i)
    {
        const uint workIndex = i + warpIndex * nbIterationsPerWarps;
        if (workIndex >= nbWarpsRequired)
            continue;

        const uint cmIdx = workIndex / NB_WARPS_PER_CM;

        PxgShape clothShape, hfShape;
        uint clothCacheRef, hfCacheRef;
        loadClothHfPair(gHf.cmInputs, gHf.gpuShapes, cmIdx,
                        clothShape, clothCacheRef, hfShape, hfCacheRef);

        // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.
        const uint64_t xf = gHf.transformCache + uint64_t(hfCacheRef) * 32UL;
        const float4 hfQ = Ptr<float4>(xf)[0];
        const float3 hfP = float3(Ptr<float>(xf + 16UL)[0],
                                  Ptr<float>(xf + 20UL)[0],
                                  Ptr<float>(xf + 24UL)[0]);
        const float3 hfScale = float3(hfShape.scaleX, hfShape.scaleY, hfShape.scaleZ);

        // The heightfield blob: nbRows @0, nbCols @4, then the samples.
        const uint64_t hfPtr = uint64_t(hfShape.hullOrMeshOffset.x) |
                               (uint64_t(hfShape.hullOrMeshOffset.y) << 32);
        const uint nbRows = Ptr<uint>(hfPtr)[0];
        const uint nbCols = Ptr<uint>(hfPtr + 4UL)[0];
        const uint64_t samples = hfPtr + 8UL;

        const float contactDist = Ptr<float>(gHf.contactDistance)[clothCacheRef] +
                                  Ptr<float>(gHf.contactDistance)[hfCacheRef];

        Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gHf.femClothes);
        const uint clothId = clothShape.particleOrSoftbodyId;
        Ptr<float4> positions = Ptr<float4>(clothes[clothId].mPosition_InvMass);
        Ptr<uint4> triVertIndices = Ptr<uint4>(clothes[clothId].mTriangleVertexIndices_ownedEdges);

        const uint numElements = vertexQuery ? clothes[clothId].mNbVerts
                                             : clothes[clothId].mNbTriangles;
        const uint nbIter = (numElements + NB_THREADS_PER_CM - 1u) / NB_THREADS_PER_CM;

        for (uint j = 0u; j < nbIter; ++j)
        {
            const uint wrappedThreadIndex = (workIndex * WARP + lane) % NB_THREADS_PER_CM;
            const uint elementIdx = wrappedThreadIndex + NB_THREADS_PER_CM * j;

            // --- the element's WORLD bound ---------------------------------
            HfRect rect;
            rect.live = false;
            rect.minRow = 0u; rect.maxRow = 0u; rect.minColumn = 0u; rect.maxColumn = 0u;
            rect.nbCols = nbCols; rect.minyF = 0.0f; rect.maxyF = 0.0f;

            if (elementIdx < numElements)
            {
                float3 bMin, bMax;
                if (vertexQuery)
                {
                    // H1: the CUDA loads positions[vertIndex] BEFORE this guard.
                    // The index is clamped instead; the value it yields for an
                    // out-of-range lane is discarded by this very `if`, so the
                    // output is unchanged and the fault is gone.
                    const float3 p = positions[min(elementIdx, numElements - 1u)].xyz;
                    bMin = p - contactDist;
                    bMax = p + contactDist;
                }
                else
                {
                    const uint4 vi = triVertIndices[elementIdx];
                    const float3 w0 = positions[vi.x].xyz;
                    const float3 w1 = positions[vi.y].xyz;
                    const float3 w2 = positions[vi.z].xyz;
                    // triBoundingBox, then fattenFast(contactDist).
                    bMin = min(min(w0, w1), w2) - contactDist;
                    bMax = max(max(w0, w1), w2) + contactDist;
                }

                float3 lMin, lMax;
                hfLocalBound(hfQ, hfP, hfScale, bMin, bMax, lMin, lMax);
                rect = hfMakeRect(lMin, lMax, nbRows, nbCols);
            }

            // --- heightfieldComputePairs -----------------------------------
            uint nbPairs = 0u;
            {
                const uint total = hfQuadSpan(rect);
                for (uint t = 0u; t < total; ++t)
                    if (hfQuadAt(samples, rect, t).survives)
                        ++nbPairs;
            }

            // --- the warp scan: ONE atomicAdd, each lane at its own offset ---
            GroupMemoryBarrierWithGroupSync();
            sHfCount[lane] = nbPairs;
            GroupMemoryBarrierWithGroupSync();
            uint pairOffset = 0u;
            for (uint k = 0u; k < lane; ++k) pairOffset += sHfCount[k];
            uint validCount = 0u;
            for (uint k = 0u; k < WARP; ++k) validCount += sHfCount[k];

            if (lane == 0u)
            {
                uint idx = 0u;
                // The CUDA reserves only when the warp has something; with an
                // all-zero warp `startIndex` stays 0xffffffff and no lane writes.
                if (validCount > 0u)
                    InterlockedAdd(Ptr<uint>(gHf.midphasePairsNum)[0], validCount, idx);
                sHfStart = idx;
            }
            GroupMemoryBarrierWithGroupSync();

            // --- heightfieldOutputPairs ------------------------------------
            // The SAME walk again, emitting this time.  `writeIndex < stackSize`
            // is upstream's loop condition, so a full stack stops the WRITES
            // while the count above is already committed -- which is how the
            // host learns it needs a bigger one.
            uint writeIndex = sHfStart + pairOffset;
            {
                const uint total = hfQuadSpan(rect);
                for (uint t = 0u; t < total && writeIndex < stackSize; ++t)
                {
                    const HfQuad q = hfQuadAt(samples, rect, t);
                    if (q.survives)
                    {
                        Ptr<uint4>(gHf.stackPtr)[writeIndex] =
                            uint4(cmIdx, elementIdx, q.triangleIdx, 0u);
                        ++writeIndex;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    }
}

// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void cloth_heightfieldMidphaseGeneratePairsLaunch(uint3 gtid : SV_GroupThreadID,
                                                  uint3 gid : SV_GroupID)
{
    clothHfMidphaseBody(gtid, gid, /*vertexQuery*/ false);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void cloth_midphaseVertexHeightfieldLaunch(uint3 gtid : SV_GroupThreadID,
                                           uint3 gid : SV_GroupID)
{
    clothHfMidphaseBody(gtid, gid, /*vertexQuery*/ true);
}
