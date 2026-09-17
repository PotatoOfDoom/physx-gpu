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

// npTriTriSharedProbe.slang -- P9.83b-4.
//
// THIS SHADER COMPUTES NOTHING.  It exists to MEASURE the Workgroup-storage
// footprint of doTriangleTriangleCollision (convexMeshMidphase.cu:972-1231)
// before that kernel is written, because the answer decides whether it can be
// one entry point at all: llvmpipe's maxComputeSharedMemorySize is 32768 and
// Vulkan applies it PER PIPELINE.
//
// P9.83b-pre projected 34816 bytes and called the kernel blocked on it.  P9.83b-2
// then argued the blocker dissolves once the contact reduction is serial, because
// the warp-emulated form's gPxgWarpScratch (4096 B) disappears: 34816 - 4096 =
// 30720.  That is an ARGUMENT, and this tranche has just spent three corrections
// learning what arguments are worth here (see the survivor list in
// pxgSlangContactReduceSerial.slang).  So the declarations below are the CUDA's,
// one for one, and gpuslang.groupshared reads the number out of the compiled
// SPIR-V.  A projection that agrees with a measurement is a measurement; a
// projection that stands alone is a guess with a number in it.
//
// The CUDA declarations being mirrored, in order (convexMeshMidphase.cu:993-1013):
//
//   Patch    patchesS[MAX_MESH_MESH_PATCHES2]      128 patches
//   PxVec4   tempPointsS[33 * MESH_MESH_CONTACT_LIMIT]   33 * 6 = 198
//   PxU32    countersS[32]
//   PxReal   shMinSepS[32]
//   uint2    triangleIndicesS[1024]
//   uint2    trianglesToSubdivideS[256]
//   PxU32    numTrianglesToSubdivideS
//   char     sSdfTexture[sizeof(SparseSDFTexture)]
//   bool     anyKeepS
//
// plus two that doTriangleTriangleCollision does NOT declare, because the
// helpers it calls SELF-ALLOCATE them -- and reading only the kernel's own
// declarations therefore undercounts:
//
//   PxU32    offset            findInterestingTrianglesA, sdfCollision.cuh:624
//   PxU32    perWarpCountS[32] threadBlockScanExclusive<32>, reduction.cuh:628,
//                              which is the overload addIndexToBuffer calls
//
// NOT declared here, deliberately: gPxgWarpScratch.  Its absence is the whole
// claim under test -- it is what the warp-emulated reduction needs and the serial
// one does not.

import pxgSlangSdfTexture;

static const uint MAX_MESH_MESH_PATCHES2 = 128u;
static const uint MESH_MESH_CONTACT_LIMIT = 6u;

// convexMeshMidphase.cu:432-438.  116 bytes in CUDA; SPIR-V Workgroup layout
// rounds the float3 to a 16-byte slot, so the measurement below is the number
// that matters, not the CUDA sizeof.
struct Patch
{
    float4 keepPoints[MESH_MESH_CONTACT_LIMIT];   // w = separation
    float3 normal;
    float deepest;
    uint pointCount;
};

groupshared Patch gPatches[MAX_MESH_MESH_PATCHES2];
groupshared float4 gTempPoints[33 * MESH_MESH_CONTACT_LIMIT];
groupshared uint gCounters[32];
groupshared float gMinSep[32];
groupshared uint2 gTriangleIndices[1024];
groupshared uint2 gTrianglesToSubdivide[256];
groupshared uint gNumTrianglesToSubdivide;
groupshared SparseSdfTexture gSdfTexture;
groupshared uint gAnyKeep;

// TWO MORE BUFFERS THAT ARE NOT DECLARED IN doTriangleTriangleCollision, and
// that the first version of this probe therefore missed.  They are SELF-ALLOCATED
// inside the helpers it calls, which is exactly why reading only the kernel's own
// declarations undercounts:
//
//   findInterestingTrianglesA (sdfCollision.cuh:624)  `__shared__ PxU32 offset;`
//   threadBlockScanExclusive<32> (reduction.cuh:628)  `__shared__ PxU32
//       perWarpCountS[NbWarps];` -- the two-argument overload allocates its own,
//       and addIndexToBuffer calls exactly that one (sdfCollision.cuh:502).
//
// A probe that measures nine of eleven buffers is the kind of partial measurement
// that reads like a complete one.
groupshared uint gFindOffset;
groupshared uint gScanPerWarpCount[32];

struct ProbeParams
{
    uint numItems;
    uint pad0, pad1, pad2;
}

[[vk::push_constant]] ConstantBuffer<ProbeParams> gP;
RWStructuredBuffer<uint> gOut;   // binding 0

// Every declaration above must be READ and WRITTEN, or SPIR-V's OpEntryPoint
// interface list will not name it and the measurement will read low.  P9.83b-1
// established that an UNUSED Workgroup variable costs nothing; this probe would
// be worthless if it accidentally relied on that.
[shader("compute")]
[numthreads(1024, 1, 1)]
void triTriSharedProbe(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    const uint t = gtid.x;

    if (t == 0u)
    {
        gNumTrianglesToSubdivide = 0u;
        gAnyKeep = 0u;
        gSdfTexture.sdfBoxLower = float3(0, 0, 0);
        gSdfTexture.sdfDx = 1.0f;
        gSdfTexture.sdfBoxHigher = float3(1, 1, 1);
        gSdfTexture.invSdfDx = 1.0f;
        gSdfTexture.texture = 0u;
        gSdfTexture.textureSubgrids = 0u;
        gSdfTexture.coarseSize = uint3(1, 1, 1);
        gSdfTexture.subgridSize = 0u;
        gSdfTexture.subgridStartSlots = 0UL;
        gSdfTexture.fineToCoarse = 1.0f;
        gSdfTexture.sdfDims = uint4(1, 1, 1, 0);
        gSdfTexture.subgridsMinSdfValue = 0.0f;
        gSdfTexture.subgridsSdfValueRange = 1.0f;
    }

    if (t == 0u) gFindOffset = 0u;
    if (t < 32u) { gCounters[t] = t; gMinSep[t] = float(t); gScanPerWarpCount[t] = t; }
    for (uint p = t; p < MAX_MESH_MESH_PATCHES2; p += 1024u)
    {
        for (uint k = 0; k < MESH_MESH_CONTACT_LIMIT; ++k)
            gPatches[p].keepPoints[k] = float4(float(k), 0, 0, 0);
        gPatches[p].normal = float3(0, 1, 0);
        gPatches[p].deepest = 0.0f;
        gPatches[p].pointCount = 0u;
    }
    for (uint q = t; q < 33u * MESH_MESH_CONTACT_LIMIT; q += 1024u)
        gTempPoints[q] = float4(float(q), 0, 0, 0);
    gTriangleIndices[t] = uint2(t, 0u);
    for (uint r = t; r < 256u; r += 1024u)
        gTrianglesToSubdivide[r] = uint2(r, 0u);

    GroupMemoryBarrierWithGroupSync();

    if (dtid.x < gP.numItems)
    {
        uint acc = gCounters[t & 31u] + uint(gMinSep[t & 31u])
                 + gTriangleIndices[t].x + gNumTrianglesToSubdivide + gAnyKeep
                 + gFindOffset + gScanPerWarpCount[t & 31u]
                 + uint(gPatches[t & 127u].keepPoints[0].x)
                 + uint(gPatches[t & 127u].normal.y) + uint(gPatches[t & 127u].deepest)
                 + gPatches[t & 127u].pointCount
                 + uint(gTempPoints[t % 198u].x)
                 + gTrianglesToSubdivide[t & 255u].x
                 + uint(gSdfTexture.sdfBoxLower.x) + uint(gSdfTexture.sdfDx)
                 + uint(gSdfTexture.sdfBoxHigher.x) + uint(gSdfTexture.invSdfDx)
                 + gSdfTexture.texture + gSdfTexture.textureSubgrids
                 + gSdfTexture.coarseSize.x + gSdfTexture.subgridSize
                 + uint(gSdfTexture.subgridStartSlots) + uint(gSdfTexture.fineToCoarse)
                 + gSdfTexture.sdfDims.x + uint(gSdfTexture.subgridsMinSdfValue)
                 + uint(gSdfTexture.subgridsSdfValueRange);
        gOut[dtid.x] = acc;
    }
}
