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

// P5 Step 4: empirical layout proof for the narrowphase element-struct
// mirrors (pxgSlangNpAbi.slang) that the convex-convex engine-ABI kernels
// index through raw device pointers.  Fills TWO-element arrays of the REAL
// engine structs with distinct sentinels, uploads the raw bytes, and the
// npAbiProbe kernel reads every field of both elements through the mirrors
// (element [1] proves the stride).  Bitwise compare + a mutation pass
// (flip one field per struct, re-run, assert exactly the matching probe
// word moved) proves the probe reads live memory at the right offsets.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h"
#include "PxContact.h"
#include "PxgContactManager.h"
#include "PxgConvexConvexShape.h"
#include "PxgPersistentContactManifold.h"
#include "PxsContactManagerState.h"
#include "PxsMaterialCore.h"
#include "cudaNpCommon.h"
#include "PxsTransformCache.h"

#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> npAbiLoader(const char* file) { return loadShaderBlob(file); }

static uint64_t sentP(uint32_t idx) { return 0xB000000000000200ull + uint64_t(idx) * 0x5000Bull; }
static uint32_t sentU(uint32_t idx) { return 0x60000031u + idx * 0x10007u; }
static float sentF(uint32_t idx) { return 1.5f + 0.25f * float(idx); }
static uint64_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

PXG_TEST(npAbiLayout_probe)
{
    using namespace physx;

    // compile-time size pins (the strides the mirrors bake in)
    static_assert(sizeof(PxgContactManagerInput) == 16, "PxgContactManagerInput size");
    static_assert(sizeof(PxgShape) == 48, "PxgShape size");
    static_assert(sizeof(PxsCachedTransform) == 32, "PxsCachedTransform size");
    static_assert(sizeof(PxgPersistentContactManifold) == 256, "PxgPersistentContactManifold size");
    static_assert(sizeof(PxsContactManagerOutput) == 48, "PxsContactManagerOutput size");
    static_assert(sizeof(PxsMaterialData) == 24, "PxsMaterialData size");
    static_assert(sizeof(PxgPatchAndContactCounters) == 16, "PxgPatchAndContactCounters size");
    static_assert(sizeof(PxGpuContactPair) == 80, "PxGpuContactPair size");

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npNpAbiProbe.spv", "npAbiProbe", 72u, 0 }, // 9 x u64
    };
    PxgCudaShim shim(dev, manifest, 1, &npAbiLoader);

    uint32_t p = 0, u = 0, f = 0;

    // ---- REAL engine structs, sentinel-filled (0xEE background) ----
    PxgContactManagerInput cmIn[2];
    std::memset(cmIn, 0xEE, sizeof(cmIn));
    for (int e = 0; e < 2; ++e)
    {
        cmIn[e].shapeRef0 = sentU(u++);
        cmIn[e].shapeRef1 = sentU(u++);
        cmIn[e].transformCacheRef0 = sentU(u++);
        cmIn[e].transformCacheRef1 = sentU(u++);
    }

    PxgShape sh[2];
    std::memset(sh, 0xEE, sizeof(sh));
    for (int e = 0; e < 2; ++e)
    {
        sh[e].scale.scale = PxVec3(sentF(f), sentF(f + 1), sentF(f + 2));
        sh[e].scale.rotation = PxQuat(sentF(f + 3), sentF(f + 4), sentF(f + 5), sentF(f + 6));
        f += 7;
        sh[e].materialIndex = sentU(u++);
        sh[e].hullOrMeshPtr = size_t(sentP(p++));
        sh[e].type = sentU(u++);
        sh[e].particleOrSoftbodyId = sentU(u++);
    }

    PxsCachedTransform xf[2];
    std::memset(xf, 0xEE, sizeof(xf));
    for (int e = 0; e < 2; ++e)
    {
        xf[e].transform.q = PxQuat(sentF(f), sentF(f + 1), sentF(f + 2), sentF(f + 3));
        xf[e].transform.p = PxVec3(sentF(f + 4), sentF(f + 5), sentF(f + 6));
        f += 7;
        xf[e].flags = sentU(u++);
    }

    PxgPersistentContactManifold mf[2];
    std::memset(mf, 0xEE, sizeof(mf));
    for (int e = 0; e < 2; ++e)
    {
        for (int j = 0; j < 4; ++j)
        {
            mf[e].mLocalNormal_PenW[j] = make_float4(sentF(f), sentF(f + 1), sentF(f + 2), sentF(f + 3));
            mf[e].mLocalContactA[j] = make_float4(sentF(f + 4), sentF(f + 5), sentF(f + 6), sentF(f + 7));
            mf[e].mLocalContactB[j] = make_float4(sentF(f + 8), sentF(f + 9), sentF(f + 10), sentF(f + 11));
            f += 12;
        }
        mf[e].mRelativePos = make_float4(sentF(f), sentF(f + 1), sentF(f + 2), sentF(f + 3));
        f += 4;
        mf[e].mQuatA = PxAlignedQuat(sentF(f), sentF(f + 1), sentF(f + 2), sentF(f + 3));
        f += 4;
        mf[e].mQuatB = PxAlignedQuat(sentF(f), sentF(f + 1), sentF(f + 2), sentF(f + 3));
        f += 4;
        mf[e].mNbContacts = sentU(u++);
        mf[e].mNbWarmStartPoints = sentU(u++);
        mf[e].mWarmStartA = sentU(u++);
        mf[e].mWarmStartB = sentU(u++);
    }

    PxsContactManagerOutput cmOut[2];
    std::memset(cmOut, 0xEE, sizeof(cmOut));
    for (int e = 0; e < 2; ++e)
    {
        cmOut[e].contactPatches = reinterpret_cast<PxU8*>(sentP(p++));
        cmOut[e].contactPoints = reinterpret_cast<PxU8*>(sentP(p++));
        cmOut[e].contactForces = reinterpret_cast<PxReal*>(sentP(p++));
        cmOut[e].frictionPatches = reinterpret_cast<PxU8*>(sentP(p++));
        cmOut[e].allflagsStart = PxU8(0x11 + e);
        cmOut[e].nbPatches = PxU8(0x22 + e);
        cmOut[e].statusFlag = PxU8(0x33 + e);
        cmOut[e].prevPatches = PxU8(0x44 + e);
        cmOut[e].nbContacts = PxU16(0x1234 + e);
        cmOut[e].flags = PxU16(0x5678 + e);
    }

    PxsMaterialData mat[2];
    std::memset(mat, 0xEE, sizeof(mat));
    for (int e = 0; e < 2; ++e)
    {
        mat[e].dynamicFriction = sentF(f++);
        mat[e].staticFriction = sentF(f++);
        mat[e].restitution = sentF(f++);
        mat[e].damping = sentF(f++);
        mat[e].flags = PxMaterialFlags(PxU16(0x0A0B + e));
        mat[e].fricCombineMode = PxU8(0x51 + e);
        mat[e].restCombineMode = PxU8(0x62 + e);
        mat[e].dampingCombineMode = PxU8(0x73 + e);
    }

    PxgPatchAndContactCounters pc[2];
    std::memset(pc, 0xEE, sizeof(pc));
    for (int e = 0; e < 2; ++e)
    {
        pc[e].patchesBytes = sentU(u++);
        pc[e].contactsBytes = sentU(u++);
        pc[e].forceAndIndiceBytes = sentU(u++);
        pc[e].overflowError = sentU(u++);
    }

    PxGpuContactPair gp[2];
    std::memset(gp, 0xEE, sizeof(gp));
    for (int e = 0; e < 2; ++e)
    {
        gp[e].contactPatches = reinterpret_cast<PxU8*>(sentP(p++));
        gp[e].contactPoints = reinterpret_cast<PxU8*>(sentP(p++));
        gp[e].contactForces = reinterpret_cast<PxReal*>(sentP(p++));
        gp[e].frictionPatches = reinterpret_cast<PxU8*>(sentP(p++));
        gp[e].transformCacheRef0 = sentU(u++);
        gp[e].transformCacheRef1 = sentU(u++);
        gp[e].nodeIndex0 = PxNodeIndex(PxU32(0x123456 + e), PxU32(e));
        gp[e].nodeIndex1 = PxNodeIndex(PxU32(0x654321 + e), PxU32(2 + e));
        gp[e].actor0 = reinterpret_cast<PxActor*>(sentP(p++));
        gp[e].actor1 = reinterpret_cast<PxActor*>(sentP(p++));
        gp[e].nbContacts = PxU16(0x2345 + e);
        gp[e].nbPatches = PxU16(0x6789 + e);
    }

    // ---- expected words in EXACTLY the probe's order ----
    auto u64of = [](const void* q) { return uint64_t(reinterpret_cast<uintptr_t>(q)); };
    std::vector<uint64_t> expect;
    for (int e = 0; e < 2; ++e)
    {
        expect.push_back(cmIn[e].shapeRef0);
        expect.push_back(cmIn[e].shapeRef1);
        expect.push_back(cmIn[e].transformCacheRef0);
        expect.push_back(cmIn[e].transformCacheRef1);
    }
    for (int e = 0; e < 2; ++e)
    {
        expect.push_back(fbits(sh[e].scale.scale.x));
        expect.push_back(fbits(sh[e].scale.scale.y));
        expect.push_back(fbits(sh[e].scale.scale.z));
        expect.push_back(fbits(sh[e].scale.rotation.x));
        expect.push_back(fbits(sh[e].scale.rotation.y));
        expect.push_back(fbits(sh[e].scale.rotation.z));
        expect.push_back(fbits(sh[e].scale.rotation.w));
        expect.push_back(sh[e].materialIndex);
        expect.push_back(uint64_t(sh[e].hullOrMeshPtr));
        expect.push_back(sh[e].type);
        expect.push_back(sh[e].particleOrSoftbodyId);
    }
    for (int e = 0; e < 2; ++e)
    {
        expect.push_back(fbits(xf[e].transform.q.x));
        expect.push_back(fbits(xf[e].transform.q.y));
        expect.push_back(fbits(xf[e].transform.q.z));
        expect.push_back(fbits(xf[e].transform.q.w));
        expect.push_back(fbits(xf[e].transform.p.x));
        expect.push_back(fbits(xf[e].transform.p.y));
        expect.push_back(fbits(xf[e].transform.p.z));
        expect.push_back(xf[e].flags);
    }
    for (int e = 0; e < 2; ++e)
    {
        expect.push_back(fbits(mf[e].mLocalNormal_PenW[0].x));
        expect.push_back(fbits(mf[e].mLocalNormal_PenW[3].w));
        expect.push_back(fbits(mf[e].mLocalContactA[0].x));
        expect.push_back(fbits(mf[e].mLocalContactA[3].w));
        expect.push_back(fbits(mf[e].mLocalContactB[0].x));
        expect.push_back(fbits(mf[e].mLocalContactB[3].w));
        expect.push_back(fbits(mf[e].mRelativePos.x));
        expect.push_back(fbits(mf[e].mRelativePos.w));
        expect.push_back(fbits(mf[e].mQuatA.q.x));
        expect.push_back(fbits(mf[e].mQuatB.q.w));
        expect.push_back(mf[e].mNbContacts);
        expect.push_back(mf[e].mNbWarmStartPoints);
        expect.push_back(mf[e].mWarmStartA);
        expect.push_back(mf[e].mWarmStartB);
    }
    for (int e = 0; e < 2; ++e)
    {
        expect.push_back(u64of(cmOut[e].contactPatches));
        expect.push_back(u64of(cmOut[e].contactPoints));
        expect.push_back(u64of(cmOut[e].contactForces));
        expect.push_back(u64of(cmOut[e].frictionPatches));
        expect.push_back(uint64_t(cmOut[e].allflagsStart) | (uint64_t(cmOut[e].nbPatches) << 8)
                         | (uint64_t(cmOut[e].statusFlag) << 16) | (uint64_t(cmOut[e].prevPatches) << 24));
        expect.push_back(uint64_t(cmOut[e].nbContacts) | (uint64_t(cmOut[e].flags) << 16));
    }
    for (int e = 0; e < 2; ++e)
    {
        expect.push_back(fbits(mat[e].dynamicFriction));
        expect.push_back(fbits(mat[e].staticFriction));
        expect.push_back(fbits(mat[e].restitution));
        expect.push_back(fbits(mat[e].damping));
        expect.push_back(uint64_t(PxU16(mat[e].flags)) | (uint64_t(mat[e].fricCombineMode) << 16)
                         | (uint64_t(mat[e].restCombineMode) << 24));
        expect.push_back(uint64_t(mat[e].dampingCombineMode));
    }
    for (int e = 0; e < 2; ++e)
    {
        expect.push_back(pc[e].patchesBytes);
        expect.push_back(pc[e].contactsBytes);
        expect.push_back(pc[e].forceAndIndiceBytes);
        expect.push_back(pc[e].overflowError);
    }
    for (int e = 0; e < 2; ++e)
    {
        expect.push_back(u64of(gp[e].contactPatches));
        expect.push_back(u64of(gp[e].contactPoints));
        expect.push_back(u64of(gp[e].contactForces));
        expect.push_back(u64of(gp[e].frictionPatches));
        expect.push_back(gp[e].transformCacheRef0);
        expect.push_back(gp[e].transformCacheRef1);
        expect.push_back(gp[e].nodeIndex0.getInd());
        expect.push_back(gp[e].nodeIndex1.getInd());
        expect.push_back(u64of(gp[e].actor0));
        expect.push_back(u64of(gp[e].actor1));
        expect.push_back(uint64_t(gp[e].nbContacts) | (uint64_t(gp[e].nbPatches) << 16));
    }
    PXG_CHECK_MSG(expect.size() == 128, "probe word count is 128 (got %zu)", expect.size());

    // ---- upload + probe ----
    struct Alloc { PxgDevicePtr d; const void* h; size_t n; };
    PxgDevicePtr dCmIn = 0, dSh = 0, dXf = 0, dMf = 0, dCmOut = 0, dMat = 0, dPc = 0, dGp = 0, dOut = 0;
    Alloc allocs[] = {
        { 0, cmIn, sizeof(cmIn) }, { 0, sh, sizeof(sh) }, { 0, xf, sizeof(xf) },
        { 0, mf, sizeof(mf) }, { 0, cmOut, sizeof(cmOut) }, { 0, mat, sizeof(mat) },
        { 0, pc, sizeof(pc) }, { 0, gp, sizeof(gp) },
    };
    for (Alloc& a : allocs)
    {
        PXG_CHECK_MSG(shim.memAlloc(&a.d, a.n) == PxgResult::Success, "memAlloc struct array");
        if (!a.d) { std::printf("  NOTE: no BDA -> skip\n"); return; }
        shim.memcpyHtoD(a.d, a.h, a.n);
    }
    dCmIn = allocs[0].d; dSh = allocs[1].d; dXf = allocs[2].d; dMf = allocs[3].d;
    dCmOut = allocs[4].d; dMat = allocs[5].d; dPc = allocs[6].d; dGp = allocs[7].d;
    PXG_CHECK_MSG(shim.memAlloc(&dOut, expect.size() * 8) == PxgResult::Success, "memAlloc out");
    shim.memsetD32(dOut, 0, expect.size() * 2);

    struct ProbeParams
    {
        uint64_t cmInput, shape, xf, manifold, cmOutput, material, counters, gpuPair, outWords;
    } prm = { dCmIn, dSh, dXf, dMf, dCmOut, dMat, dPc, dGp, dOut };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success, "launch probe");
    shim.streamSynchronize();

    std::vector<uint64_t> got(expect.size(), 0);
    shim.memcpyDtoH(got.data(), dOut, got.size() * 8);

    size_t bad = 0;
    for (size_t i = 0; i < expect.size(); ++i)
        if (got[i] != expect[i])
        {
            if (bad < 10)
                PXG_CHECK_MSG(false, "probe[%zu]: gpu=%016llx expected=%016llx", i,
                              (unsigned long long)got[i], (unsigned long long)expect[i]);
            ++bad;
        }
    PXG_CHECK_MSG(bad == 0, "np element-struct mirrors are byte-exact (%zu bad of %zu)", bad, expect.size());

    // ---- mutation pass: flip one field per struct (element [1]), re-run,
    // assert exactly the matching probe words moved ----
    cmIn[1].transformCacheRef1 ^= 0xA5A5u;
    sh[1].hullOrMeshPtr ^= size_t(0xFF00FF00ull);
    xf[1].transform.p.z = -xf[1].transform.p.z;
    mf[1].mNbContacts ^= 0x77u;
    cmOut[1].statusFlag = PxU8(cmOut[1].statusFlag ^ 0x5A);
    mat[1].fricCombineMode = PxU8(mat[1].fricCombineMode ^ 0x0F);
    pc[1].overflowError ^= 0x101u;
    gp[1].nodeIndex1 = PxNodeIndex(PxU32(0x00BEEF), PxU32(7));

    shim.memcpyHtoD(dCmIn, cmIn, sizeof(cmIn));
    shim.memcpyHtoD(dSh, sh, sizeof(sh));
    shim.memcpyHtoD(dXf, xf, sizeof(xf));
    shim.memcpyHtoD(dMf, mf, sizeof(mf));
    shim.memcpyHtoD(dCmOut, cmOut, sizeof(cmOut));
    shim.memcpyHtoD(dMat, mat, sizeof(mat));
    shim.memcpyHtoD(dPc, pc, sizeof(pc));
    shim.memcpyHtoD(dGp, gp, sizeof(gp));
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success, "launch probe (mutants)");
    shim.streamSynchronize();

    std::vector<uint64_t> got2(expect.size(), 0);
    shim.memcpyDtoH(got2.data(), dOut, got2.size() * 8);

    // indices of the probe words the mutations must move (element [1] rows)
    const size_t idxCmIn = 4 + 3;                       // cmIn[1].transformCacheRef1
    const size_t idxShape = 8 + 11 + 8;                 // sh[1].hullOrMeshPtr
    const size_t idxXf = 8 + 22 + 8 + 6;                // xf[1].transform.p.z
    const size_t idxMf = 8 + 22 + 16 + 14 + 10;         // mf[1].mNbContacts
    const size_t idxCmOut = 8 + 22 + 16 + 28 + 6 + 4;   // cmOut[1].allflagsWord
    const size_t idxMat = 8 + 22 + 16 + 28 + 12 + 6 + 4;      // mat[1].flagsModes0
    const size_t idxPc = 8 + 22 + 16 + 28 + 12 + 12 + 4 + 3;  // pc[1].overflowError
    const size_t idxGp = 8 + 22 + 16 + 28 + 12 + 12 + 8 + 11 + 7; // gp[1].nodeIndex1
    const size_t mutated[] = { idxCmIn, idxShape, idxXf, idxMf, idxCmOut, idxMat, idxPc, idxGp };

    size_t moved = 0, unexpected = 0;
    for (size_t i = 0; i < expect.size(); ++i)
    {
        bool shouldMove = false;
        for (size_t m : mutated)
            if (i == m) shouldMove = true;
        const bool didMove = got2[i] != got[i];
        if (shouldMove && didMove) ++moved;
        if (shouldMove && !didMove)
            PXG_CHECK_MSG(false, "mutation at probe[%zu] did not move", i);
        if (!shouldMove && didMove)
        {
            ++unexpected;
            if (unexpected <= 5)
                PXG_CHECK_MSG(false, "probe[%zu] moved without a mutation (%016llx -> %016llx)", i,
                              (unsigned long long)got[i], (unsigned long long)got2[i]);
        }
    }
    PXG_CHECK_MSG(moved == 8 && unexpected == 0, "mutation pass: %zu/8 moved, %zu unexpected", moved, unexpected);

    for (Alloc& a : allocs)
        shim.memFree(a.d);
    shim.memFree(dOut);
}

} // namespace
