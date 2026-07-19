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

// Verification of memCopy.slang, clampValues.slang and skinning.slang
// against C++ transcriptions of MemCopyBalanced.cu and utility.cu.

#include "pxgTestFramework.h"
#include "pxgGpuStructs.h"
#include "pxgRefMath.h"
#include "PxgCudaShim.h"

#include <cstring>
#include <numeric>
#include <random>

namespace
{

using namespace pxgtest;

// ---------------------------------------------------------------------------
// MemCopyBalanced
// ---------------------------------------------------------------------------

// RUNTIME_PLAN.md P5 Step 4: MemCopyBalanced is now driven the way the engine drives
// it — through PxgCudaShim (memAlloc device addresses, launch with a param blob of raw
// pointers), NOT the descriptor-binding Kernel harness.  So CopyDesc.dest/source are the
// REAL device addresses the kernel dereferences via BDA (mirroring PxgCopyManager::CopyDesc
// = 4 x size_t), matching the production ABI exactly.
struct CopyDesc
{
    uint64_t dest;   // device address of the destination words
    uint64_t source; // device address of the source words
    uint64_t bytes;
    uint64_t pad;
};
static_assert(sizeof(CopyDesc) == 32, "CopyDesc mirror must be 32 bytes");

struct MemCopyParams
{
    uint64_t descs; // device address of CopyDesc[count]
    uint32_t count;
};

static std::vector<uint8_t> memCopyLoader(const char* file) { return loadShaderBlob(file); }

PXG_TEST(memCopyBalanced_randomDescriptors)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(900);

    const uint32_t numWords = 200000;
    std::vector<uint32_t> src(numWords), dst(numWords, 0xDEADBEEFu);
    for (uint32_t& v : src)
        v = rng();
    const std::vector<uint32_t> dstOriginal = dst;

    // carve non-overlapping destination ranges with random sizes (including
    // empty and single-word copies), recorded as word offsets for the reference
    struct Range { uint32_t destWord, srcWord, words; };
    std::vector<Range> ranges;
    uint32_t cursor = 0;
    while (cursor + 70 < numWords && ranges.size() < 40)
    {
        const uint32_t words = rng() % 60; // may be 0
        Range r;
        r.destWord = cursor;
        r.srcWord = rng() % (numWords - words);
        r.words = words;
        ranges.push_back(r);
        cursor += words + rng() % 10;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "memCopy.spv", "MemCopyBalanced", uint32_t(sizeof(MemCopyParams)), 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &memCopyLoader);

    PxgDevicePtr dsrc = 0, ddst = 0, ddescs = 0;
    PXG_CHECK_MSG(shim.memAlloc(&dsrc, numWords * 4) == PxgResult::Success, "memAlloc src");
    PXG_CHECK_MSG(shim.memAlloc(&ddst, numWords * 4) == PxgResult::Success, "memAlloc dst");
    if (dsrc == 0 || ddst == 0)
    {
        std::printf("  NOTE: shim has no device-address allocations (no BDA) -> skip\n");
        return;
    }
    PXG_CHECK_MSG(shim.memcpyHtoD(dsrc, src.data(), numWords * 4) == PxgResult::Success, "HtoD src");
    PXG_CHECK_MSG(shim.memcpyHtoD(ddst, dst.data(), numWords * 4) == PxgResult::Success, "HtoD dst");

    // descriptors carry absolute device addresses into the src/dst allocations
    std::vector<CopyDesc> descs;
    for (const Range& r : ranges)
    {
        CopyDesc d = {};
        d.dest = ddst + uint64_t(r.destWord) * 4;
        d.source = dsrc + uint64_t(r.srcWord) * 4;
        d.bytes = uint64_t(r.words) * 4;
        descs.push_back(d);
    }
    PXG_CHECK_MSG(shim.memAlloc(&ddescs, descs.size() * sizeof(CopyDesc)) == PxgResult::Success, "memAlloc descs");
    PXG_CHECK_MSG(shim.memcpyHtoD(ddescs, descs.data(), descs.size() * sizeof(CopyDesc)) == PxgResult::Success, "HtoD descs");

    const MemCopyParams params = { ddescs, uint32_t(descs.size()) };
    PXG_CHECK_MSG(shim.launchKernel(0, uint32_t(descs.size()), 1, 1, &params, sizeof(params)) == PxgResult::Success,
                  "launchKernel MemCopyBalanced through BDA marshalling");
    shim.streamSynchronize();

    std::vector<uint32_t> gpuDst(numWords, 0);
    PXG_CHECK_MSG(shim.memcpyDtoH(gpuDst.data(), ddst, numWords * 4) == PxgResult::Success, "DtoH dst");

    // reference
    std::vector<uint32_t> refDst = dstOriginal;
    for (const Range& r : ranges)
        std::memcpy(refDst.data() + r.destWord, src.data() + r.srcWord, size_t(r.words) * 4);

    size_t mismatches = 0;
    for (uint32_t i = 0; i < numWords; ++i)
    {
        if (gpuDst[i] != refDst[i])
        {
            if (mismatches < 5)
                PXG_CHECK_MSG(false, "dst[%u]: gpu=%08x ref=%08x", i, gpuDst[i], refDst[i]);
            ++mismatches;
        }
    }
    PXG_CHECK_MSG(mismatches == 0, "%zu mismatched words", mismatches);

    shim.memFree(dsrc);
    shim.memFree(ddst);
    shim.memFree(ddescs);
}

// ---------------------------------------------------------------------------
// clampMaxValue / clampMaxValues
// ---------------------------------------------------------------------------

struct ClampParams
{
    uint32_t index0, index1, index2, maxValue;
};

PXG_TEST(clampMaxValues)
{
    PxgSlangComputeDevice* dev = testDevice();

    std::vector<uint32_t> values = { 5, 500, 42, 1000, 7, 99 };
    PxgBufferHandle buf = createBufferFrom(dev, values);

    Kernel clamp1("clampValues.spv", "clampMaxValue", { 4 }, sizeof(ClampParams));
    Kernel clamp3("clampValues.spv", "clampMaxValues", { 4 }, sizeof(ClampParams));

    clamp1.dispatch(1, { buf }, ClampParams{ 1, 0, 0, 100 }); // values[1]: 500 -> 100
    dev->barrier();
    clamp3.dispatch(1, { buf }, ClampParams{ 2, 3, 5, 50 }); // 42 stays, 1000 -> 50, 99 -> 50
    dev->flush();

    const std::vector<uint32_t> out = downloadBuffer<uint32_t>(dev, buf, values.size());
    const uint32_t expected[6] = { 5, 100, 42, 50, 7, 50 };
    for (int i = 0; i < 6; ++i)
        PXG_CHECK_MSG(out[i] == expected[i], "values[%d]: gpu=%u ref=%u", i, out[i], expected[i]);

    dev->destroyBuffer(buf);
}

// ---------------------------------------------------------------------------
// skinning (utility.cu)
// ---------------------------------------------------------------------------

struct StridedView
{
    uint32_t offset, stride, count, pad;
};

struct TrimeshDesc
{
    StridedView guideVertices;
    StridedView guideNormals;
    uint32_t guideTrianglesOffset;
    uint32_t nbGuideTriangles;
    uint32_t skinningInfoOffset;
    float halfSurfaceThickness;
    StridedView skinnedVertices;
};
static_assert(sizeof(TrimeshDesc) == 64, "TrimeshDesc must be 64 bytes");

struct TetmeshDesc
{
    StridedView guideVertices;
    uint32_t guideTetrahedraOffset;
    uint32_t skinningInfoOffset;
    uint32_t pad0, pad1;
    StridedView skinnedVertices;
};
static_assert(sizeof(TetmeshDesc) == 48, "TetmeshDesc must be 48 bytes");

struct SkinningParams
{
    uint32_t length;
};

float refNormalizeSafe(Vec3& v)
{
    const float mag = v.magnitude();
    if (mag < 1e-20f)
        return 0.0f;
    v *= 1.0f / mag;
    return mag;
}

// mirrors evaluatePointPhongInterpolation<true>() in utility.cu
Vec3 refPhong(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& uvw, const Vec3& nA,
              const Vec3& nB, const Vec3& nC, float halfSurfaceThickness, float alpha = 0.625f)
{
    const Vec3 q = a * uvw.x + b * uvw.y + c * uvw.z;

    const float scale1 = (q - a).dot(nA);
    const Vec3 projA = q - nA * scale1;
    const float scale2 = (q - b).dot(nB);
    const Vec3 projB = q - nB * scale2;
    const float scale3 = (q - c).dot(nC);
    const Vec3 projC = q - nC * scale3;

    const Vec3 qStar = projA * uvw.x + projB * uvw.y + projC * uvw.z;

    Vec3 dir = qStar - q;
    float offset = refNormalizeSafe(dir) * alpha;

    float ratio = 0.0f;
    if (halfSurfaceThickness > 0.0f)
    {
        ratio = offset / halfSurfaceThickness;
        ratio = std::tanh(ratio);
    }
    offset = ratio * halfSurfaceThickness;

    return q + dir * offset;
}

PXG_TEST(skinning_clothPipeline)
{
    std::mt19937 rng(901);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-2.f, 2.f);

    const uint32_t numDescs = 3;
    const uint32_t vertStrideWords = 4; // strided views: 16-byte stride exercises stride handling

    // data buffer assembled per desc: guide verts, guide normals, triangles,
    // skinning infos, skinned verts
    std::vector<uint32_t> data;
    std::vector<TrimeshDesc> descs(numDescs);
    struct MeshInfo
    {
        uint32_t nbVerts, nbTris, nbSkinned;
        std::vector<Vec3> verts;
        std::vector<uint32_t> tris;
        std::vector<float> uv;      // 2 per skinned vertex
        std::vector<float> offsets; // 1 per skinned vertex
        std::vector<uint32_t> triId;
    };
    std::vector<MeshInfo> meshes(numDescs);

    auto pushWord = [&](uint32_t w) { data.push_back(w); };
    auto pushFloat = [&](float f) { uint32_t w; std::memcpy(&w, &f, 4); data.push_back(w); };

    for (uint32_t m = 0; m < numDescs; ++m)
    {
        MeshInfo& mesh = meshes[m];
        mesh.nbVerts = 20 + rng() % 30;
        mesh.nbTris = 30 + rng() % 40;
        mesh.nbSkinned = 50 + rng() % 50;

        TrimeshDesc& d = descs[m];

        // guide vertices (strided)
        d.guideVertices = { uint32_t(data.size() * 4), vertStrideWords * 4, mesh.nbVerts, 0 };
        for (uint32_t v = 0; v < mesh.nbVerts; ++v)
        {
            const Vec3 p(sym(rng), sym(rng), sym(rng));
            mesh.verts.push_back(p);
            pushFloat(p.x); pushFloat(p.y); pushFloat(p.z); pushFloat(0.f);
        }

        // guide normals (strided, zero-initialized on the CPU side; the GPU
        // pipeline re-zeros them anyway)
        d.guideNormals = { uint32_t(data.size() * 4), vertStrideWords * 4, mesh.nbVerts, 0 };
        for (uint32_t v = 0; v < mesh.nbVerts * vertStrideWords; ++v)
            pushFloat(123.0f); // poison: zeroNormals must clear it

        // triangles
        d.guideTrianglesOffset = uint32_t(data.size() * 4);
        d.nbGuideTriangles = mesh.nbTris;
        for (uint32_t t = 0; t < mesh.nbTris; ++t)
        {
            uint32_t i0 = rng() % mesh.nbVerts, i1 = rng() % mesh.nbVerts, i2 = rng() % mesh.nbVerts;
            mesh.tris.insert(mesh.tris.end(), { i0, i1, i2 });
            pushWord(i0); pushWord(i1); pushWord(i2);
        }

        // skinning infos (PxTriangleMeshEmbeddingInfo: uv, offset, triId)
        d.skinningInfoOffset = uint32_t(data.size() * 4);
        for (uint32_t s = 0; s < mesh.nbSkinned; ++s)
        {
            const float u = unit(rng) * 1.2f - 0.1f; // slightly outside [0,1] to hit the projection path
            const float v = unit(rng) * 1.2f - 0.1f;
            const float off = sym(rng) * 0.1f;
            const uint32_t tri = rng() % mesh.nbTris;
            mesh.uv.insert(mesh.uv.end(), { u, v });
            mesh.offsets.push_back(off);
            mesh.triId.push_back(tri);
            pushFloat(u); pushFloat(v); pushFloat(off); pushWord(tri);
        }

        // skinned vertices (tightly packed vec3)
        d.skinnedVertices = { uint32_t(data.size() * 4), 12, mesh.nbSkinned, 0 };
        for (uint32_t s = 0; s < mesh.nbSkinned * 3; ++s)
            pushFloat(0.f);

        d.halfSurfaceThickness = unit(rng) < 0.5f ? 0.f : 0.05f + unit(rng) * 0.2f;
    }

    // ---- GPU: zeroNormals -> normalVectorsAreaWeighted -> normalizeNormals
    //          -> interpolateSkinnedClothVertices
    PxgSlangComputeDevice* dev = testDevice();

    const std::vector<uint32_t> strides = { 4, 4, 16, 16, 4 };
    Kernel zero("skinning.spv", "zeroNormals", strides, sizeof(SkinningParams));
    Kernel areaWeighted("skinning.spv", "normalVectorsAreaWeighted", strides, sizeof(SkinningParams));
    Kernel normalizeK("skinning.spv", "normalizeNormals", strides, sizeof(SkinningParams));
    Kernel interpolate("skinning.spv", "interpolateSkinnedClothVertices", strides, sizeof(SkinningParams));

    PxgBufferHandle descBuf = createBufferFrom(dev, descs);
    PxgBufferHandle dataBuf = createBufferFrom(dev, data);
    PxgBufferHandle dummyF4a = dev->createBuffer(16);
    PxgBufferHandle dummyF4b = dev->createBuffer(16);
    PxgBufferHandle dummyOut = dev->createBuffer(16);

    uint32_t maxVerts = 0, maxTris = 0, maxSkinned = 0;
    for (const MeshInfo& mesh : meshes)
    {
        maxVerts = std::max(maxVerts, mesh.nbVerts);
        maxTris = std::max(maxTris, mesh.nbTris);
        maxSkinned = std::max(maxSkinned, mesh.nbSkinned);
    }

    const std::vector<PxgBufferHandle> bufs = { descBuf, dataBuf, dummyF4a, dummyF4b, dummyOut };
    const SkinningParams params = { 0 };
    zero.dispatch((maxVerts + 255) / 256, bufs, params, numDescs);
    dev->barrier();
    areaWeighted.dispatch((maxTris + 255) / 256, bufs, params, numDescs);
    dev->barrier();
    normalizeK.dispatch((maxVerts + 255) / 256, bufs, params, numDescs);
    dev->barrier();
    interpolate.dispatch((maxSkinned + 255) / 256, bufs, params, numDescs);
    dev->flush();

    const std::vector<uint32_t> gpuData = downloadBuffer<uint32_t>(dev, dataBuf, data.size());

    // ---- reference (transcription of utility.cu) ----
    for (uint32_t m = 0; m < numDescs; ++m)
    {
        const MeshInfo& mesh = meshes[m];
        const TrimeshDesc& d = descs[m];

        std::vector<Vec3> normals(mesh.nbVerts, Vec3(0.f));
        for (uint32_t t = 0; t < mesh.nbTris; ++t)
        {
            const uint32_t* tri = &mesh.tris[3 * t];
            const Vec3 n = (mesh.verts[tri[1]] - mesh.verts[tri[0]]).cross(mesh.verts[tri[2]] - mesh.verts[tri[0]]);
            normals[tri[0]] += n;
            normals[tri[1]] += n;
            normals[tri[2]] += n;
        }
        for (Vec3& n : normals)
            refNormalizeSafe(n);

        // compare normals (atomic accumulation order differs -> tolerance)
        for (uint32_t v = 0; v < mesh.nbVerts; ++v)
        {
            const uint32_t w = (d.guideNormals.offset + v * d.guideNormals.stride) / 4;
            float gx, gy, gz;
            std::memcpy(&gx, &gpuData[w], 4);
            std::memcpy(&gy, &gpuData[w + 1], 4);
            std::memcpy(&gz, &gpuData[w + 2], 4);
            PXG_CHECK_MSG(almostEqual(gx, normals[v].x, 2e-3f, 2e-3f) &&
                              almostEqual(gy, normals[v].y, 2e-3f, 2e-3f) &&
                              almostEqual(gz, normals[v].z, 2e-3f, 2e-3f),
                          "mesh %u normal[%u]: gpu=(%g,%g,%g) ref=(%g,%g,%g)", m, v, gx, gy, gz,
                          normals[v].x, normals[v].y, normals[v].z);
        }

        // compare skinned vertices
        for (uint32_t s = 0; s < mesh.nbSkinned; ++s)
        {
            const uint32_t* tri = &mesh.tris[3 * mesh.triId[s]];
            const float u = mesh.uv[2 * s], v = mesh.uv[2 * s + 1];
            const float w0 = 1.0f - u - v;

            const Vec3 uvw(u, v, w0);
            Vec3 uvwProj = uvw.maximum(Vec3(0.f));
            const float sumProj = uvwProj.x + uvwProj.y + uvwProj.z;
            if (sumProj > 0.0f)
                uvwProj *= 1.0f / sumProj;

            const Vec3 nA = normals[tri[0]], nB = normals[tri[1]], nC = normals[tri[2]];
            Vec3 normal = nA * uvwProj.x + nB * uvwProj.y + nC * uvwProj.z;
            refNormalizeSafe(normal);

            const Vec3 a = mesh.verts[tri[0]], b = mesh.verts[tri[1]], c = mesh.verts[tri[2]];
            const Vec3 pointPhong = refPhong(a, b, c, uvwProj, nA, nB, nC, d.halfSurfaceThickness);
            const Vec3 pointUVW = a * uvw.x + b * uvw.y + c * uvw.z;
            const Vec3 pointUVWProj = a * uvwProj.x + b * uvwProj.y + c * uvwProj.z;

            const Vec3 expected = pointPhong + normal * mesh.offsets[s] + pointUVW - pointUVWProj;

            const uint32_t w = (d.skinnedVertices.offset + s * d.skinnedVertices.stride) / 4;
            float gx, gy, gz;
            std::memcpy(&gx, &gpuData[w], 4);
            std::memcpy(&gy, &gpuData[w + 1], 4);
            std::memcpy(&gz, &gpuData[w + 2], 4);
            PXG_CHECK_MSG(almostEqual(gx, expected.x, 2e-3f, 2e-3f) &&
                              almostEqual(gy, expected.y, 2e-3f, 2e-3f) &&
                              almostEqual(gz, expected.z, 2e-3f, 2e-3f),
                          "mesh %u skinned[%u]: gpu=(%g,%g,%g) ref=(%g,%g,%g)", m, s, gx, gy, gz,
                          expected.x, expected.y, expected.z);
        }
    }

    dev->destroyBuffer(descBuf);
    dev->destroyBuffer(dataBuf);
    dev->destroyBuffer(dummyF4a);
    dev->destroyBuffer(dummyF4b);
    dev->destroyBuffer(dummyOut);
}

PXG_TEST(skinning_softBodyAndInterleave)
{
    std::mt19937 rng(902);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-2.f, 2.f);

    PxgSlangComputeDevice* dev = testDevice();

    // ---- interpolateSkinnedSoftBodyVertices ----
    const uint32_t nbVerts = 40, nbTets = 25, nbSkinned = 80;

    std::vector<uint32_t> data;
    auto pushWord = [&](uint32_t w) { data.push_back(w); };
    auto pushFloat = [&](float f) { uint32_t w; std::memcpy(&w, &f, 4); data.push_back(w); };

    TetmeshDesc desc = {};
    std::vector<Vec3> verts;
    std::vector<uint32_t> tets;
    std::vector<Vec3> uvws;
    std::vector<uint32_t> tetIds;

    desc.guideVertices = { uint32_t(data.size() * 4), 12, nbVerts, 0 };
    for (uint32_t v = 0; v < nbVerts; ++v)
    {
        const Vec3 p(sym(rng), sym(rng), sym(rng));
        verts.push_back(p);
        pushFloat(p.x); pushFloat(p.y); pushFloat(p.z);
    }

    desc.guideTetrahedraOffset = uint32_t(data.size() * 4);
    for (uint32_t t = 0; t < nbTets * 4; ++t)
    {
        const uint32_t idx = rng() % nbVerts;
        tets.push_back(idx);
        pushWord(idx);
    }

    desc.skinningInfoOffset = uint32_t(data.size() * 4);
    for (uint32_t s = 0; s < nbSkinned; ++s)
    {
        const Vec3 uvw(unit(rng), unit(rng) * 0.5f, unit(rng) * 0.3f);
        const uint32_t tet = rng() % nbTets;
        uvws.push_back(uvw);
        tetIds.push_back(tet);
        pushFloat(uvw.x); pushFloat(uvw.y); pushFloat(uvw.z); pushWord(tet);
    }

    desc.skinnedVertices = { uint32_t(data.size() * 4), 12, nbSkinned, 0 };
    for (uint32_t s = 0; s < nbSkinned * 3; ++s)
        pushFloat(0.f);

    const std::vector<uint32_t> strides = { 4, 4, 16, 16, 4 };
    Kernel softBody("skinning.spv", "interpolateSkinnedSoftBodyVertices", strides, sizeof(SkinningParams));

    std::vector<TetmeshDesc> descs = { desc };
    PxgBufferHandle descBuf = createBufferFrom(dev, descs);
    PxgBufferHandle dataBuf = createBufferFrom(dev, data);
    PxgBufferHandle dummyF4a = dev->createBuffer(16);
    PxgBufferHandle dummyF4b = dev->createBuffer(16);
    PxgBufferHandle dummyOut = dev->createBuffer(16);

    const SkinningParams params = { 0 };
    softBody.dispatch((nbSkinned + 255) / 256, { descBuf, dataBuf, dummyF4a, dummyF4b, dummyOut },
                      params, 1);
    dev->flush();

    const std::vector<uint32_t> gpuData = downloadBuffer<uint32_t>(dev, dataBuf, data.size());

    for (uint32_t s = 0; s < nbSkinned; ++s)
    {
        const uint32_t* tet = &tets[4 * tetIds[s]];
        const Vec3 uvw = uvws[s];
        const float sw = 1.0f - uvw.x - uvw.y - uvw.z;
        const Vec3 expected = verts[tet[0]] * uvw.x + verts[tet[1]] * uvw.y + verts[tet[2]] * uvw.z +
                              verts[tet[3]] * sw;

        const uint32_t w = (desc.skinnedVertices.offset + s * desc.skinnedVertices.stride) / 4;
        float gx, gy, gz;
        std::memcpy(&gx, &gpuData[w], 4);
        std::memcpy(&gy, &gpuData[w + 1], 4);
        std::memcpy(&gz, &gpuData[w + 2], 4);
        PXG_CHECK_MSG(almostEqual(gx, expected.x) && almostEqual(gy, expected.y) && almostEqual(gz, expected.z),
                      "softbody skinned[%u]: gpu=(%g,%g,%g) ref=(%g,%g,%g)", s, gx, gy, gz, expected.x,
                      expected.y, expected.z);
    }

    dev->destroyBuffer(descBuf);
    dev->destroyBuffer(dataBuf);

    // ---- interleaveBuffers ----
    const uint32_t length = 1000;
    std::vector<Float4> vertsIn(length), normalsIn(length);
    for (uint32_t i = 0; i < length; ++i)
    {
        vertsIn[i] = Float4{ sym(rng), sym(rng), sym(rng), unit(rng) };
        normalsIn[i] = Float4{ sym(rng), sym(rng), sym(rng), unit(rng) };
    }

    Kernel interleave("skinning.spv", "interleaveBuffers", strides, sizeof(SkinningParams));

    PxgBufferHandle dummyDesc = dev->createBuffer(64);
    PxgBufferHandle dummyData = dev->createBuffer(16);
    PxgBufferHandle vBuf = createBufferFrom(dev, vertsIn);
    PxgBufferHandle nBuf = createBufferFrom(dev, normalsIn);
    PxgBufferHandle outBuf = dev->createBuffer(length * 6 * sizeof(float));

    const SkinningParams interleaveParams = { length };
    interleave.dispatch((length + 255) / 256, { dummyDesc, dummyData, vBuf, nBuf, outBuf },
                        interleaveParams);
    dev->flush();

    const std::vector<float> out = downloadBuffer<float>(dev, outBuf, length * 6);
    for (uint32_t i = 0; i < length; ++i)
    {
        PXG_CHECK_MSG(out[6 * i] == vertsIn[i].x && out[6 * i + 1] == vertsIn[i].y &&
                          out[6 * i + 2] == vertsIn[i].z && out[6 * i + 3] == normalsIn[i].x &&
                          out[6 * i + 4] == normalsIn[i].y && out[6 * i + 5] == normalsIn[i].z,
                      "interleave[%u]", i);
    }

    dev->destroyBuffer(dummyDesc);
    dev->destroyBuffer(dummyData);
    dev->destroyBuffer(dummyF4a);
    dev->destroyBuffer(dummyF4b);
    dev->destroyBuffer(dummyOut);
    dev->destroyBuffer(vBuf);
    dev->destroyBuffer(nBuf);
    dev->destroyBuffer(outBuf);
}

} // namespace
