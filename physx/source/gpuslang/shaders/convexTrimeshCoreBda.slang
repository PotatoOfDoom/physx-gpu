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

// convexTrimeshCoreBda.slang
//
// `convexTrimeshNarrowphase` (convexMesh.cu:621-759) on the engine BDA ABI: the
// convex-vs-triangle narrowphase core, composed from the six modules the
// preceding tranches built and verified one at a time.
//
//   pxgSlangMidphaseAlloc   the stack carve
//   pxgSlangCvxTriSetup     the per-pair prologue                       (P8.46b)
//   pxgSlangConvexHull      the cooked hull reader                      (P8.39)
//   pxgSlangConvexSat       the three SAT stages, driver, face selection (P8.40-42)
//   pxgSlangPolyClip        the clipper                                 (P8.38)
//   pxgSlangCvxTriEmit      the tail, which pulls in the reduction      (P8.44-45)
//
// ===========================================================================
// THE DISPATCH SHAPE, AND THE COVERAGE IT GIVES UP
// ===========================================================================
//
// The CUDA is a GRID-STRIDE loop over pairs (convexMesh.cu:691) whose body
// performs a CHAINED buffer-device-address load — `hullOrMeshPtr` is read out of
// the shape array and then dereferenced (convexMesh.cu:147, :163).  That is
// exactly the pattern P8.40 proved segfaults llvmpipe inside the JIT-compiled
// shader.
//
// The obvious remedy — one workgroup per pair — is NOT AVAILABLE.  The pair count
// is a DEVICE value (`*nbPairsGlobal`, convexMesh.cu:664) while the grid is
// computed on the HOST, and the CUDA shim exposes only a direct dispatch: the
// Vulkan device has `dispatchIndirect` but nothing reaches it from this path.
//
// So this takes the shape `convexTrimeshMidphaseBda` already ships green: one
// workgroup per CUDA BLOCK, and a loop of CONSTANT trip count over that block's
// two warps.  That shader performs a chained BDA load inside exactly this kind of
// loop and does not crash, which is the evidence that the hazard is the
// grid-stride form and not loops as such.
//
// **THE GRID-STRIDE LOOP IS BACK, AND A SCENE HAD TO PROVE IT WAS NEEDED.**
//
// An earlier version of this header said the missing loop was a known, accepted
// coverage gap, and that closing it "needs either indirect dispatch in the shim or a
// host-side grid derived from the same device counter, and neither is in this
// tranche".  Both halves were wrong.
//
// It is not merely a coverage gap.  The host launches this kernel with
// `max(4096, 32 * numTests)` blocks (PxgNarrowphaseCore.cpp:2596), so this
// kernel's ceiling is 8192 pairs and no scene here comes near it -- which is why
// the omission never showed on THIS path.  Its heightfield sibling is launched with
// `numTests * 32` warps instead (PxgNarrowphaseCore.cpp:2898), a ceiling of 32 pairs
// for one contact manager, and PXG_SCENE=hfbigbox drove a half-3 box past it: the box
// fell straight THROUGH the terrain, final y = -70.8 and still accelerating at
// 39 m/s.  The loop is restored here for the same reason, before a denser scene
// finds the 8192 as well.
// Records for the uncovered pairs are never written, and the P9.52b-6 measurement on
// the sphere kernel showed the same condition there ending in a SIGSEGV inside
// convexTrimeshCorrelate.
//
// And it needs neither indirect dispatch nor a host-side grid: `WorkgroupCount()` is
// available in Slang, the trip count `ceil(nbPairs / (WorkgroupCount().x * 2))` is
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
//
// ONE WARP PER WORKGROUP is not a choice either: `pxgSlangPolyClip` and
// `pxgSlangContactReduce` both require `[numthreads(32,1,1)]`, because their
// early-outs are warp-uniform but not group-uniform and llvmpipe HANGS on a
// divergent barrier rather than erroring.  The CUDA's two warps per block become
// two sequential iterations here.
//
// ===========================================================================
// THE ALLOCATOR POINTERS, WHICH CORRUPT RATHER THAN FAULT
// ===========================================================================
//
// `maCompute` returns its seven addresses in ALLOCATION order, which is NOT the
// order of the CUDA function's out-parameters.  Every one of them points inside
// the same stack allocation, so writing `a.maxDepth` where `a.interm` belongs
// still lands in valid memory — it silently overwrites another section's records.
// The six publishing stores below are therefore written by NAME against the
// struct, never by position.
//
// `a.pairs` is deliberately NOT published: downstream consumers know that region
// as the stack base.

import pxgSlangMidphaseAlloc;
import pxgSlangCvxTriSetup;
import pxgSlangConvexHull;
import pxgSlangConvexSat;
import pxgSlangPolyClip;
import pxgSlangCvxTriEmit;

// PxgNpKernelIndices.h:33
static const uint NP_TRIMESH_WARPS_PER_BLOCK = 2u;
// calculateConvexMeshPairMemRequirement() (midphaseAllocate.cuh:93): uint4 16 +
// ConvexTriNormalAndIndex 16 + ConvexTriContacts 4 + PxReal 4 +
// ConvexTriIntermediateData 4 + PxU32 4 + PxU32 * 2 = 56.  NOT the ConvexMeshPair
// stride, which is also 56 -- two unrelated numbers that happen to coincide, so
// changing one must never be taken as licence to change the other.  Both are
// pinned to the engine headers by static_assert in testNpAbiLayout.cpp.
static const uint CTC_MEM_PER_PAIR = 56u;
static const float CTC_ORTHO_EPS = 1e-6f;

// The 21-argument pack of convexMesh.cu:622-647, at CUDA's natural-alignment
// .param layout.  The two pads are the layout's own, never read.
struct CvxTriCoreParams
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
    uint64_t maxTempMemRequirement;    // @144  PINNED HOST memory -- see below
    uint64_t midPhasePairsNeeded;      // @152
    uint     nbContactManagers;        // @160
    uint     _pad164;                  // @164 (blob rounded to maxAlign 8)
}                                      // = 168

[[vk::push_constant]]
ConstantBuffer<CvxTriCoreParams> gCt;

[shader("compute")]
[numthreads(32, 1, 1)]
void convexTrimeshNarrowphase(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    const uint nbPairs = Ptr<uint>(gCt.nbPairsGlobal)[0];
    const uint nbPaddedPairs = Ptr<uint>(gCt.nbPaddedPairsGlobal)[0];

    // Every CUDA block computes its own private copy (convexMesh.cu:668-670).
    const MidphaseAlloc a = maCompute(gCt.stackPtr, nbPairs, nbPaddedPairs);

    if (gid.x == 0u && lane == 0u)
    {
        // By NAME, never by position -- see the header.
        Ptr<uint64_t>(gCt.cvxTriNIPtr)[0] = a.ni;
        Ptr<uint64_t>(gCt.cvxTriContactsPtr)[0] = a.contacts;
        Ptr<uint64_t>(gCt.cvxTriMaxDepthPtr)[0] = a.maxDepth;
        Ptr<uint64_t>(gCt.cvxTriIntermPtr)[0] = a.interm;
        Ptr<uint64_t>(gCt.orderedCvxTriIntermPtr)[0] = a.ordered;
        Ptr<uint64_t>(gCt.cvxTriSecondPassPairsPtr)[0] = a.secondPass;

        // THE ATOMIC LANDS IN PINNED HOST MEMORY.  `maxTempMemRequirement` is a
        // cuMemHostAlloc'd word the CPU reads with no copy, where it drives the
        // collisionStackSize overflow warning.  P8.46a established two things this
        // depends on: that an atomic read-modify-write on such memory works at all
        // here, and that the ABI must TRANSLATE the argument, since the runtime
        // backs pinned memory with a host-visible buffer whose device address is
        // its own number.  The kernel's ABI row carries the hostPtrMask bit for
        // argument 19 for exactly that reason.
        const uint afterPairPadding = 3u * (CTC_MEM_PER_PAIR - 2u * 4u);
        const uint afterTriangleIndexPadding = gCt.nbContactManagers * 6u * 4u;
        const uint required = CTC_MEM_PER_PAIR * Ptr<uint>(gCt.midPhasePairsNeeded)[0]
                            + afterPairPadding + afterTriangleIndexPadding;
        uint old;
        InterlockedMax(Ptr<uint>(gCt.maxTempMemRequirement)[0], required, old);
    }

    const uint tempContactSize = gCt.stackSizeBytes / 16u;   // sizeof(ConvexTriContact)

    // A CONSTANT trip count, so the barriers inside are reached by the whole
    // group on every iteration.  The guard uses `continue`, not `return`, for the
    // same reason.
    // convexHeightfield.cu:330 / convexMesh.cu -- the CUDA's loop is
    //   `for (globalWarpIndex = blockIdx.x*blockDim.y + threadIdx.y;
    //         globalWarpIndex < nbPairs;
    //         globalWarpIndex += gridDim.x * blockDim.y)`
    // with blockDim.y == NP_TRIMESH_WARPS_PER_BLOCK.  The `sub` loop below covers
    // `threadIdx.y`; THIS loop covers the stride, and it is not optional -- see the
    // header.  Its trip count, `ceil(nbPairs / (WorkgroupCount().x * 2))`, is
    // WORKGROUP-UNIFORM, so the barriers inside stay in uniform control flow.
    const uint gridStride = WorkgroupCount().x * NP_TRIMESH_WARPS_PER_BLOCK;

    for (uint base = 0u; base < nbPairs; base += gridStride)
    {
    for (uint sub = 0u; sub < NP_TRIMESH_WARPS_PER_BLOCK; ++sub)
    {
        GroupMemoryBarrierWithGroupSync();

        const uint globalWarpIndex = base + gid.x * NP_TRIMESH_WARPS_PER_BLOCK + sub;
        if (globalWarpIndex >= nbPairs)
            continue;

        const uint4 curPair = Ptr<uint4>(a.pairs)[globalWarpIndex];

        const CtSetup s = ctSetup(gCt.cmInputs, gCt.transformCache, gCt.contactDistance,
                                  gCt.gpuShapes, gCt.cvxTrimeshPair, curPair);

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
            // runs at most twice, and its trip count is warp-uniform.  Written as
            // a two-iteration loop with a uniform guard so the barriers inside
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

        // The clipper's staging slot for this lane, and the RAW first contact the
        // second-pass classifier needs.  Both reads are in bounds for every lane;
        // the emit tail's own `lane < count` guard is what makes the value
        // irrelevant where the lane holds no contact.
        const float4 staged = pcContact(lane);
        const float3 rawContact0 = pcContact(0u).xyz;

        cteEmitTail(lane, isSATPassed, delayContacts, numContactsGenerated,
                    staged, rawContact0, minNormal,
                    s.triLocVerts0, s.triLocVerts1, s.triLocVerts2,
                    s.convexTriPairOffset, s.convexTriPairOffsetPadded,
                    s.remapCpuTriangleIdx, curPair.y, globalWarpIndex,
                    a.ni, a.contacts, a.maxDepth, a.interm, a.ordered,
                    a.secondPass, gCt.nbSecondPassPairs,
                    gCt.tempConvexTriContacts, tempContactSize, gCt.pTempContactIndex);
    }
    }
}
