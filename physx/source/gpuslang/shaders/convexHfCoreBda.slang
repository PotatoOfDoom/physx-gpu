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

// convexHfCoreBda.slang
//
// `convexHeightfieldNarrowphase` (convexHeightfield.cu:271-357, whose per-pair
// body is convexHeightfieldNarrowphaseCore at :160-269) on the engine BDA ABI --
// the convex-vs-heightfield-triangle narrowphase core.  P9.51b.
//
// ===========================================================================
// WHY THIS IS THE TRIMESH CORE WITH ONE PARAGRAPH CHANGED
// ===========================================================================
//
// Read side by side, convexHeightfieldNarrowphaseCore and the trimesh
// convexTrimeshNarrowphase's loop body differ in exactly two places:
//
//   1. WHERE THE TRIANGLE COMES FROM.  The trimesh reads three indices out of a
//      cooked index buffer and three vertices out of a cooked vertex buffer; the
//      heightfield derives both from arithmetic on the sample grid
//      (convexHeightfield.cu:83-116 getTriangle).
//   2. THE REMAP.  A cooked mesh carries a faceRemap that turns the GPU triangle
//      index back into the index the CPU cooker used; a heightfield has none, and
//      the .cu passes `triangleIdx` TWICE into convexTriangleContactGen (:266,
//      with the comment "height field don't have cpu remap triangle index").
//
// Everything else -- the stack carve, the pointer publication, the pinned-host
// atomicMax, the pair prologue, the SAT, the polygon selection, the clipper and
// the emit tail -- is the same code on the same data.  P9.51b split
// `pxgSlangCvxTriSetup`'s prologue into `ctCommon` for that reason rather than
// copying it here; see that module's header for the measurement showing the split
// changed no arithmetic.
//
// Both differences live in `pxgSlangCvxHfSetup`, not here: this kernel has no unit
// fixture of its own -- like its trimesh sibling it is a composition of modules
// verified one at a time -- so anything it computed inline would be unverified
// until a scene ran.  `chSetup` is driven against a host mirror by
// `gpuslang.cvxHfSetup`, and the slot swap's deliberate absence is argued in that
// module's header.
//
// ===========================================================================
// THE DISPATCH SHAPE, AND THE COVERAGE IT GIVES UP -- INHERITED, NOT NEW
// ===========================================================================
//
// This carries `convexTrimeshCoreBda`'s limitation verbatim, and it is restated
// here rather than cross-referenced because a reader of THIS file needs it:
//
// The CUDA is a GRID-STRIDE loop over pairs (convexHeightfield.cu:331) whose body
// performs a CHAINED buffer-device-address load -- `hullOrMeshPtr` read out of the
// shape array and then dereferenced (:123, :239).  P8.40 proved that pattern
// segfaults llvmpipe inside the JIT-compiled shader.  One workgroup per pair is
// not available: the pair count is a DEVICE value (`*nbPairsGlobal`, :307) while
// the grid is computed on the HOST, and the CUDA shim exposes only a direct
// dispatch.
//
// So this took the shape the midphase and the trimesh core then shipped green:
// one workgroup per CUDA BLOCK, and a loop of CONSTANT trip count over that
// block's two warps.  (All three now launch one workgroup per CUDA WARP instead, see
// the loop; the grid-stride loop below is what covers the pairs.)
//
// **THE GRID-STRIDE LOOP IS BACK, AND A SCENE HAD TO PROVE IT WAS NEEDED.**
//
// An earlier version of this header said the missing loop was a known, accepted
// coverage gap, and that closing it "needs either indirect dispatch in the shim or a
// host-side grid derived from the same device counter, and neither is in this
// tranche".  Both halves were wrong.
//
// It is not merely a coverage gap.  The host launches this kernel with
// `numBlocks = (numTests * coreGridMultiplier + warps - 1) / warps`, where
// `coreGridMultiplier` is 32 (PxgNarrowphaseCore.cpp:2898) -- so for ONE contact
// manager the single pass covers 32 pairs.  A half-0.5 box on a heightfield needs
// about eight, which is why every scene passed.  A half-3 box needs about seventy,
// and PXG_SCENE=hfbigbox measured what happens past the ceiling: the box fell
// straight THROUGH the terrain, final y = -70.8 and still accelerating at 39 m/s.
// Records for the uncovered pairs are never written, and the P9.52b-6 measurement on
// the sphere kernel showed the same condition there ending in a SIGSEGV inside
// convexTrimeshCorrelate.
//
// And it needs neither indirect dispatch nor a host-side grid: `WorkgroupCount()` is
// available in Slang, the trip count `ceil(nbPairs / WorkgroupCount().x)` is
// WORKGROUP-UNIFORM, and the loop is three lines.
//
// ON THE P8.40 HAZARD, precisely.  This file previously recorded that a chained
// buffer-device-address load inside a grid-stride loop segfaults llvmpipe in the JIT.
// The shape restored here -- an OUTER grid-stride loop wrapping the CONSTANT-trip
// sub-warp loop, chained BDA loads inside -- does not reproduce it: hfbigbox,
// hffloor, meshfloor, spherehf and convex all step clean, and the sphere kernel's own
// grid-stride loop (P9.52b-6) has the same shape and the same chained loads.  That is
// a measurement of THIS shape, not a refutation of whatever P8.40 tested; the earlier
// finding may well hold for a per-thread grid-stride written differently.  What can
// be said is that the hazard as stated was too broad to justify dropping the loop.
// Since the launch site went to one group per CUDA warp the sub-warp loop is gone and
// this is a plain group-stride loop; its index is clamped for the loads (the P9.49
// empty-mask pass, see the loop), which keeps every chained address a real one.
//
// ONE WARP PER WORKGROUP is not a choice either: `pxgSlangPolyClip` and
// `pxgSlangContactReduce` both require `[numthreads(32,1,1)]` because their
// early-outs are warp-uniform but not group-uniform, and llvmpipe HANGS on a
// divergent barrier rather than erroring.  The launch site launches one such group
// per CUDA warp (see the loop).
//
// ===========================================================================
// THE ABI ROW IS SHARED WITH THE TRIMESH CORE, AND THAT IS CHECKED
// ===========================================================================
//
// convexHeightfieldNarrowphase (:273-297) and convexTrimeshNarrowphase
// (convexMesh.cu:623-647) declare the SAME 21 parameters in the SAME order with
// the same types, so the push-constant blob is byte-identical and both rows in
// `PxgVulkanKernelAbi.cpp` point at `kCvxTriCore` with `kCvxTriCoreHostPtrMask`.
// P9.48d is why that has to be exact rather than merely close: `PxgCudaShim.cpp`
// requires an EXACT blob match, and a kernel whose ABI row is missing or wrong
// gets `pushConstantSizeFor() == 0` and every launch REFUSED IN SILENCE.
//
// Argument 18 (`maxTempMemRequirement`) is PINNED HOST memory, which is why the
// mask exists; see the trimesh core's header.

module convexHfCoreBda;

import pxgSlangMidphaseAlloc;
import pxgSlangCvxTriSetup;
import pxgSlangCvxHfSetup;
import pxgSlangConvexHull;
import pxgSlangConvexSat;
import pxgSlangPolyClip;
import pxgSlangCvxTriEmit;

// PxgNpKernelIndices.h:33
static const uint NP_TRIMESH_WARPS_PER_BLOCK = 2u;
// calculateConvexMeshPairMemRequirement() (midphaseAllocate.cuh:93).  NOT the
// ConvexMeshPair stride, which is also 56 -- two unrelated numbers that coincide.
static const uint CTC_MEM_PER_PAIR = 56u;
static const float CTC_ORTHO_EPS = 1e-6f;

// The 21-argument pack of convexHeightfield.cu:273-297 at CUDA's natural-alignment
// .param layout -- the same 168-byte blob as CvxTriCoreParams, field for field.
struct CvxHfCoreParams
{
    uint64_t cmInputs;                 // @0
    uint64_t transformCache;           // @8
    uint64_t contactDistance;          // @16
    uint64_t gpuShapes;                // @24
    uint64_t cvxTrimeshPair;           // @32
    uint64_t cvxTriNIPtr;              // @40
    uint64_t cvxTriContactsPtr;        // @48
    uint64_t cvxTriMaxDepthPtr;        // @56
    uint64_t cvxTriIntermPtr;          // @64
    uint64_t orderedCvxTriIntermPtr;   // @72
    uint64_t cvxTriSecondPassPairsPtr; // @80
    uint64_t stackPtr;                 // @88
    uint64_t nbPairsGlobal;            // @96
    uint64_t nbPaddedPairsGlobal;      // @104
    uint64_t nbSecondPassPairs;        // @112
    uint     stackSizeBytes;           // @120
    uint     _pad124;                  // @124 (the next argument is 8-aligned)
    uint64_t tempConvexTriContacts;    // @128
    uint64_t pTempContactIndex;        // @136
    uint64_t maxTempMemRequirement;    // @144  PINNED HOST memory
    uint64_t midphasePairsNeeded;      // @152
    uint     nbContactManagers;        // @160
    uint     _pad164;                  // @164 (blob rounded to maxAlign 8)
}                                      // = 168

[[vk::push_constant]]
ConstantBuffer<CvxHfCoreParams> gCh;

[shader("compute")]
[numthreads(32, 1, 1)]
void convexHeightfieldNarrowphase(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    const uint nbPairs = Ptr<uint>(gCh.nbPairsGlobal)[0];
    const uint nbPaddedPairs = Ptr<uint>(gCh.nbPaddedPairsGlobal)[0];

    // Every CUDA block computes its own private copy (convexHeightfield.cu:313).
    const MidphaseAlloc a = maCompute(gCh.stackPtr, nbPairs, nbPaddedPairs);

    if (gid.x == 0u && lane == 0u)
    {
        // BY NAME, never by position: maCompute returns its seven addresses in
        // ALLOCATION order, which is not the order of the CUDA function's
        // out-parameters, and every one of them points inside the same stack
        // allocation -- so a misplaced store silently overwrites another
        // section's records instead of faulting.
        Ptr<uint64_t>(gCh.cvxTriNIPtr)[0] = a.ni;
        Ptr<uint64_t>(gCh.cvxTriContactsPtr)[0] = a.contacts;
        Ptr<uint64_t>(gCh.cvxTriMaxDepthPtr)[0] = a.maxDepth;
        Ptr<uint64_t>(gCh.cvxTriIntermPtr)[0] = a.interm;
        Ptr<uint64_t>(gCh.orderedCvxTriIntermPtr)[0] = a.ordered;
        Ptr<uint64_t>(gCh.cvxTriSecondPassPairsPtr)[0] = a.secondPass;

        // The atomic lands in PINNED HOST memory -- see the header.
        const uint afterPairPadding = 3u * (CTC_MEM_PER_PAIR - 2u * 4u);
        const uint afterTriangleIndexPadding = gCh.nbContactManagers * 6u * 4u;
        const uint required = CTC_MEM_PER_PAIR * Ptr<uint>(gCh.midphasePairsNeeded)[0]
                            + afterPairPadding + afterTriangleIndexPadding;
        uint old;
        InterlockedMax(Ptr<uint>(gCh.maxTempMemRequirement)[0], required, old);
    }

    const uint tempContactSize = gCh.stackSizeBytes / 16u;   // sizeof(ConvexTriContact)

    // A trip count uniform across the group, so the barriers inside are reached by
    // the whole group on every iteration.  The one `return` below is group-uniform and
    // comes before the first barrier.
    // convexHeightfield.cu:330 / convexMesh.cu -- the CUDA's loop is
    //   `for (globalWarpIndex = blockIdx.x*blockDim.y + threadIdx.y;
    //         globalWarpIndex < nbPairs;
    //         globalWarpIndex += gridDim.x * blockDim.y)`
    // with blockDim.y == NP_TRIMESH_WARPS_PER_BLOCK.  The port's launch site
    // (PxgNarrowphaseCore.cpp, PXG_NO_CUDA_TOOLKIT) launches one 32-thread group per
    // CUDA warp, gridDim.x * blockDim.y of them, so the group index IS the CUDA's
    // globalWarpIndex and the group count its stride: each group takes exactly the pairs
    // of its CUDA warp (the block's two warps used to be walked serially in one group).
    // THIS loop covers the stride, and it is not optional -- see the header.  Its trip
    // count, `ceil(nbPairs / WorkgroupCount().x)`, is WORKGROUP-UNIFORM, so the barriers
    // inside stay in uniform control flow.
    const uint gridStride = WorkgroupCount().x;

    // A group with no pair at all leaves before the loop.  On hardware that only skips a
    // loop test, but on llvmpipe the barrier-bearing body below still runs once with an
    // EMPTY mask (P9.49), on the clamped -- i.e. a real -- pair, and the engine launches
    // 32 groups per contact manager whatever the pair count: without this exit every
    // idle one did that masked pass (the trimesh sibling, with its 4096-group floor,
    // went from ~1 ms to 16.5 ms per launch in meshfloor, median).
    if (gid.x >= nbPairs)
        return;

    for (uint warpLoop = gid.x; warpLoop < nbPairs; warpLoop += gridStride)
    {
    {
        GroupMemoryBarrierWithGroupSync();

        // llvmpipe can run a barrier-bearing loop body once more with an EMPTY mask and
        // still issue its loads (P9.22, P9.49); here they are chained (the pair names a
        // contact manager, whose shapes name a hull or a mesh), so the clamp keeps the
        // first of them inside the pair array.  Every live iteration has
        // globalWarpIndex == warpLoop.
        const uint globalWarpIndex = min(warpLoop, max(nbPairs, 1u) - 1u);

        const uint4 curPair = Ptr<uint4>(a.pairs)[globalWarpIndex];

        // The prologue AND the heightfield triangle, from the module the
        // `cvxHfSetupBda` fixture drives against a host mirror.  Written as a call
        // rather than inline for that reason: this kernel has no unit fixture of
        // its own, so anything it computes itself is unverified until a scene runs.
        const ChSetup s = chSetup(gCh.cmInputs, gCh.transformCache, gCh.contactDistance,
                                  gCh.gpuShapes, gCh.cvxTrimeshPair, curPair);

        const uint hullDesc = chHullDesc(s.hull);

        csScaleVerts(lane, s.hull, hullDesc, s.convexScale, s.convexScaleRot);
        GroupMemoryBarrierWithGroupSync();

        uint featureIndex;
        float separation;
        float3 minNormal;
        const bool isSATPassed =
            csConvexMeshSAT(lane, s.hull, hullDesc, s.convexScale, s.convexScaleRot,
                            s.contactDist, s.triLocVerts0, s.triLocVerts1, s.triLocVerts2,
                            s.triangleLocNormal, s.triAdjTrisIdx.x, s.triAdjTrisIdx.y,
                            s.triAdjTrisIdx.z, featureIndex, separation, minNormal);

        // convexMeshSAT's barrier sits BEFORE the caller's first early return, so
        // every lane reaches it whatever the verdict.
        GroupMemoryBarrierWithGroupSync();

        bool delayContacts = false;
        uint numContactsGenerated = 0u;

        if (isSATPassed)
        {
            delayContacts = csDelayGenerateContact(minNormal, s.triangleLocNormal, featureIndex);

            uint faceIndex2 = 0xffffffffu;
            const uint faceIndex1 = csSelectPolygonFeatureIndex(lane, s.hull, hullDesc,
                                                                minNormal, featureIndex, faceIndex2);

            // PxPlane(triLocVerts[0], triangleLocNormal)
            const float4 triPlane = float4(s.triangleLocNormal,
                                           -dot(s.triangleLocNormal, s.triLocVerts0));

            const float3 verts1 = (lane == 0u) ? s.triLocVerts0
                                : ((lane == 1u) ? s.triLocVerts1
                                : ((lane == 2u) ? s.triLocVerts2 : float3(0.0f, 0.0f, 0.0f)));

            float4 plane0 = float4(csPlaneN(faceIndex1), csPlaneD(faceIndex1));
            float4 plane1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (faceIndex2 != 0xffffffffu)
                plane1 = float4(csPlaneN(faceIndex2), csPlaneD(faceIndex2));

            GroupMemoryBarrierWithGroupSync();

            // The CUDA's `for (faceIndex = faceIndex1; faceIndex != 0xFFFFFFFF;)`
            // runs at most twice and its trip count is warp-uniform.  Written as a
            // two-iteration loop with a uniform guard so the barriers inside
            // `pcPolyClip` stay in uniform control flow.
            uint faceIndex = faceIndex1;
            for (uint pass = 0u; pass < 2u; ++pass)
            {
                if (faceIndex != 0xffffffffu)
                {
                    const uint polyDesc = chPolyDesc(s.hull, hullDesc, faceIndex);
                    const uint nbVerts0 = chPolyNbVerts(polyDesc);

                    const bool predicate = abs(dot(minNormal, plane0.xyz)) >= CTC_ORTHO_EPS
                                        && abs(dot(minNormal, triPlane.xyz)) >= CTC_ORTHO_EPS;

                    if (predicate)
                    {
                        float3 verts0 = float3(0.0f, 0.0f, 0.0f);
                        if (lane < nbVerts0)
                            verts0 = csScaledVert(chPolyVertex(s.hull, hullDesc,
                                                               chPolyVRef8(polyDesc), lane));

                        numContactsGenerated =
                            pcPolyClip(lane, plane0, verts0, nbVerts0, triPlane, verts1,
                                       minNormal, s.contactDist, 0xffffffffu, numContactsGenerated);
                    }
                }

                faceIndex = faceIndex2;
                faceIndex2 = 0xffffffffu;
                plane0 = plane1;
            }

            GroupMemoryBarrierWithGroupSync();
        }

        const float4 staged = pcContact(lane);
        const float3 rawContact0 = pcContact(0u).xyz;

        // THE REMAP ARGUMENT IS `triangleIdx`, NOT A REMAP.  A heightfield has no
        // cooked faceRemap, so the .cu passes the triangle index for both
        // parameters (:266) -- the same value twice, deliberately.
        cteEmitTail(lane, isSATPassed, delayContacts, numContactsGenerated,
                    staged, rawContact0, minNormal,
                    s.triLocVerts0, s.triLocVerts1, s.triLocVerts2,
                    s.convexTriPairOffset, s.convexTriPairOffsetPadded,
                    s.triangleIdx, curPair.y, globalWarpIndex,
                    a.ni, a.contacts, a.maxDepth, a.interm, a.ordered,
                    a.secondPass, gCh.nbSecondPassPairs,
                    gCh.tempConvexTriContacts, tempContactSize, gCh.pTempContactIndex);
    }
    }
}
