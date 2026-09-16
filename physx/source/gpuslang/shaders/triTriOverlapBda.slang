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

// triTriOverlapBda.slang -- P9.83.
//
//   triangleTriangleOverlaps (convexMeshMidphase.cu:1386-1528) on the engine
//   launch ABI.  The first of the two kernels testSDKTriMeshTriMeshGpu runs: it
//   marks, per contact manager, whether the two meshes' scaled root bounding
//   boxes come within contactDistance of each other.  The SECOND kernel
//   (triangleTriangleCollision) reads that byte as its `generateContacts` flag,
//   so a mesh-vs-mesh pair whose byte is wrong is either skipped outright or
//   pushed through a full SDF collision for nothing.
//
// ---------------------------------------------------------------------------
// ASSEMBLY, NOT A NEW PORT
// ---------------------------------------------------------------------------
// Everything that decides is in pxgSlangTriTriOverlap (P9.83) and pxgSlangGjk
// (P9.12d), and npTriTriOverlap.slang's fixture certifies those two directly.
// What is new in this file is only the glue Phase 1 could not have: the
// contact-manager load, the cooked-blob walk to each mesh's BV32 root, and the
// byte-lane store into the interim overlap array.
//
// ---------------------------------------------------------------------------
// ONE WARP PER PAIR -- AND THE GRID IS SIZED FROM THE BLOCK
// ---------------------------------------------------------------------------
// This launcher is NOT the one-block-per-pair shape triPlaneNpBda has.  It is
//
//     numBlocks = (numTests + NP_TRIMESH_WARPS_PER_BLOCK - 1) / NP_TRIMESH_WARPS_PER_BLOCK
//     launchKernel(..., numBlocks,1,1,  32, NP_TRIMESH_WARPS_PER_BLOCK, 1, ...)
//                                                        (PxgNarrowphaseCore.cpp:1405-1409)
//
// -- a 2-D block of 32 x warpsPerBlock, one WARP per contact manager, and the
// grid derived from the block size.  That is exactly the shape P9.68's checker
// exists for: declaring a smaller [numthreads] than the host's block would leave
// a fixed fraction of the pairs untouched forever.  So the group size here is
// (32, NP_TRIMESH_WARPS_PER_BLOCK, 1) -- the host's block, verbatim -- and the
// pair index is `gid.x * NP_TRIMESH_WARPS_PER_BLOCK + gtid.y`, which is the
// CUDA's `warpIndex + blockIdx.x * blockDim.y`.
//
// NP_TRIMESH_WARPS_PER_BLOCK IS 2.  Not 4, not 32: PxgNpKernelIndices.h:33
// spells it as a bare `#define NP_TRIMESH_WARPS_PER_BLOCK 2`, so there is no
// arithmetic to re-derive it from and nothing in the build would object to a
// different number here.  A shader compiled for the wrong value still RUNS -- it
// simply addresses the wrong pairs, which reads as a physics bug rather than a
// launch bug.  It appears in two places in this file (the [numthreads] y and the
// pair index) and they must agree with the header and with each other.
//
// Lane 0 of each warp serializes its pair.  The CUDA spreads the 32 root-node
// slots and the 16 corner points across the warp and syncs twice; the module's
// core is the serial transcription of that and is the version the fixture
// certified, so the other 31 lanes have nothing to do.
//
// ---------------------------------------------------------------------------
// THE OVERLAP ARRAY IS A PxU8 PER PAIR, AND FOUR PAIRS SHARE A WORD
// ---------------------------------------------------------------------------
// `gpuIntermOverlap` is `mIntermStackAlloc.allocateAligned(256, numTests * 1)`
// (PxgNarrowphaseCore.cpp:1388) -- one BYTE per contact manager, and NOT zeroed.
// Vulkan here has no 8-bit storage (the device never requests
// VK_KHR_8bit_storage / storageBuffer8BitAccess), so the store has to go through
// the containing uint, and four pairs -- four different warps, possibly four
// different blocks -- share each one.  A read-modify-write would lose three of
// every four flags.
//
// The fix is the one abStoreU8 (pxgSlangArticulationBlocks) already uses and
// documents: an InterlockedAnd that clears only this pair's byte lane followed by
// an InterlockedOr that sets it.  Each warp touches disjoint bits of the word, so
// every interleaving of the two atomics across four writers lands the same four
// bytes.  The AND is what makes the uninitialised allocation safe; dropping it
// (because "the value is 0 or 1 anyway") would leave stale garbage in the lane
// whenever the flag is false, and a non-zero byte is `true`.
//
// NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the GJK's lower/upper bound
// comparison is what decides DISTANT, and it is fed by mul-add chains.

module triTriOverlapBda;

import pxgSlangCommon;
import pxgSlangNpAbi;
import pxgSlangBv32;
import pxgSlangGjk;
import pxgSlangTriTriOverlap;

// PxgNpKernelIndices.h:33 -- `#define NP_TRIMESH_WARPS_PER_BLOCK 2`.
static const uint NP_TRIMESH_WARPS_PER_BLOCK = 2u;

// ---------------------------------------------------------------------------
// Push-constant blob -- the CUDA parameter list EXACTLY
// (PxgNarrowphaseCore.cpp:1393-1401): PxU32 @0, then six pointers @8..@56.
// ABI row { 4, 8, 8, 8, 8, 8, 8 } -> 56 bytes (4 + 4 of padding + six 8-byte
// pointers; the pad is the alignment the first pointer forces, not a field).
// ---------------------------------------------------------------------------
struct TriTriOverlapParams
{
    uint numWorkItems;          // @0
    uint _pad4;                 // @4
    uint64_t cmInputs;          // @8
    uint64_t transformCache;    // @16
    uint64_t contactDistance;   // @24
    uint64_t shapes;            // @32
    uint64_t cmOutputs;         // @40   (see below -- pushed, never read)
    uint64_t overlaps;          // @48   PxU8[numWorkItems]
}

[[vk::push_constant]] ConstantBuffer<TriTriOverlapParams> gTT;

// `cmOutputs` IS DEAD IN THE CUDA TOO.  triangleTriangleOverlaps takes
// `PxsContactManagerOutput* cmOutputs` at :1393 and never names it again in the
// body; only its sibling triangleTriangleCollision writes outputs.  It is kept in
// the struct because the ABI row must match the host's pack byte for byte -- the
// argument occupies @40 whether or not anything reads it, and dropping it would
// slide `overlaps` up to @40 and make every flag land on the output array.  This
// is recorded in check_push_block.py's partial-reader ratchet rather than fixed:
// the port does not get to change the engine's parameter list.

// ---------------------------------------------------------------------------
// the cooked mesh's BV32 root (readTriangleMesh, triangleMesh.cuh:120-150)
//   @0  uint4 (numVerts, numTris, _, nbPackedNodes)
//   @16 BV32DataPacked[nbPackedNodes]     <- node 0 is the root
// Only the root's child bounds are needed here; the verts, indices, adjacency
// and remap sections past the tree are never touched by this kernel.
// ---------------------------------------------------------------------------
TtoBox meshRootBox(uint64_t meshPtr)
{
    const uint64_t treeAddr = meshPtr + 16UL;
    const uint nbRootSlots = bvNbNodes(treeAddr, 0u);

    float4 mins[32], maxs[32];
    for (uint s = 0u; s < 32u; ++s)
    {
        mins[s] = float4(bvMin(treeAddr, 0u, s), 0.0f);
        maxs[s] = float4(bvMax(treeAddr, 0u, s), 0.0f);
    }
    // Every slot is READ but only the first nbRootSlots are FOLDED -- the module
    // applies the mask.  Reading all 32 unconditionally is what the CUDA does
    // (one lane per slot, no predicate on the load) and the node is a fixed
    // 1168-byte record, so the reads are always in bounds.
    return ttoRootBox(mins, maxs, nbRootSlots);
}

// One byte per pair into a word four pairs share.  See the header.
void storeOverlapByte(uint64_t base, uint pairIdx, bool flag)
{
    Ptr<uint> w = Ptr<uint>(base + uint64_t(pairIdx >> 2u) * 4UL);
    const uint shift = (pairIdx & 3u) * 8u;
    uint prev;
    InterlockedAnd(w[0], ~(0xFFu << shift), prev);
    if (flag)
        InterlockedOr(w[0], 1u << shift, prev);
}

[shader("compute")]
[numthreads(32, 2, 1)]
void triangleTriangleOverlaps(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    // warpIndex + blockIdx.x * blockDim.y (cu:1415)
    const uint pairIdx = gtid.y + gid.x * NP_TRIMESH_WARPS_PER_BLOCK;
    if (pairIdx >= gTT.numWorkItems)
        return;
    if (gtid.x != 0u)
        return;                   // lane 0 serializes its warp (see the header)

    const PxgContactManagerInputGpu cin = Ptr<PxgContactManagerInputGpu>(gTT.cmInputs)[pairIdx];

    // NO SWAP.  Unlike the convex-vs-mesh paths, this kernel reads slot 0 and
    // slot 1 as they arrive (cu:1429-1443): both shapes ARE triangle meshes, so
    // there is nothing to sort, and which one is "mesh0" only decides which
    // cloud moves into whose space -- an OBB-vs-OBB distance that is symmetric.
    Ptr<PxgShapeGpu> shapes = Ptr<PxgShapeGpu>(gTT.shapes);
    const PxgShapeGpu shape0 = shapes[cin.shapeRef0];
    const PxgShapeGpu shape1 = shapes[cin.shapeRef1];

    Ptr<PxsCachedTransformGpu> xforms = Ptr<PxsCachedTransformGpu>(gTT.transformCache);
    const PxsCachedTransformGpu xf0 = xforms[cin.transformCacheRef0];
    const PxsCachedTransformGpu xf1 = xforms[cin.transformCacheRef1];

    Ptr<float> cd = Ptr<float>(gTT.contactDistance);
    const float contactDistance = cd[cin.transformCacheRef0] + cd[cin.transformCacheRef1];

    const TtoBox b0 = meshRootBox(shape0.hullOrMeshPtr);
    const TtoBox b1 = meshRootBox(shape1.hullOrMeshPtr);

    GjkOut o;
    float3 cloudA[8], cloudB[8];
    const uint ret = ttoOverlap(b0, b1,
                                float4(xf0.qx, xf0.qy, xf0.qz, xf0.qw), float3(xf0.px, xf0.py, xf0.pz),
                                float4(xf1.qx, xf1.qy, xf1.qz, xf1.qw), float3(xf1.px, xf1.py, xf1.pz),
                                float3(shape0.scaleX, shape0.scaleY, shape0.scaleZ),
                                float4(shape0.scaleRotX, shape0.scaleRotY, shape0.scaleRotZ, shape0.scaleRotW),
                                float3(shape1.scaleX, shape1.scaleY, shape1.scaleZ),
                                float4(shape1.scaleRotX, shape1.scaleRotY, shape1.scaleRotZ, shape1.scaleRotW),
                                contactDistance,
                                o, cloudA, cloudB);

    storeOverlapByte(gTT.overlaps, pairIdx, ttoOverlapFlag(ret));
}
