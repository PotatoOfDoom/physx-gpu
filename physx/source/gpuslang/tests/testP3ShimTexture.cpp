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

// testP3ShimTexture.cpp -- P9.12b, the SDF upload path through the CUDA-driver
// entry points the engine actually calls.
//
// The tested surface is deliberately the OUTER one.  P9.12a's fixture proves the
// bindless layer samples correctly; this one proves that the exact call sequence
// PxgGeometryManager performs --
//
//     cuArray3DCreate(&arr, {W, H, D, format, 1 channel, 0})     :739
//     cuMemcpy3DAsync({HOST -> ARRAY, tightly packed}, stream)   :755
//     cuTexObjectCreate(&tex, {ARRAY, arr}, {clamp x3, linear})  :770
//
// -- produces a CUtexObject the shader can sample, and that the translation layer
// refuses everything it cannot faithfully represent instead of approximating it.
// That second half matters more than the first: every refusal below is a case
// where a silent approximation would have produced wrong SAMPLED VALUES rather
// than an error, and wrong sampled values in an SDF are wrong contacts.

#include "pxgTestFramework.h"

#include "PxgCudaShim.h"
#include "PxgCudaToolkitShim.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace physx { namespace gpuslang {
void pxgSetVulkanDriverShim(PxgCudaShim* shim);
} }

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;

static std::vector<uint8_t> texLoader(const char* file)
{
    return loadShaderBlob(file);
}

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };

// The engine's own upload, transcribed from PxgGeometryManager.cpp:723-771 so the
// call SHAPE is the engine's and not one invented here.
static CUresult engineUpload(CUtexObject* outTex, CUarray* outArr, CUarray_format format,
                             uint32_t w, uint32_t h, uint32_t d, const void* data, size_t texelBytes)
{
    CUDA_ARRAY3D_DESCRIPTOR arrDesc;
    std::memset(&arrDesc, 0, sizeof(arrDesc));
    arrDesc.Format = format;
    arrDesc.NumChannels = 1;
    arrDesc.Width = w;
    arrDesc.Height = h;
    arrDesc.Depth = d;
    arrDesc.Flags = 0;
    CUresult r = cuArray3DCreate(outArr, &arrDesc);
    if (r != CUDA_SUCCESS)
        return r;

    CUDA_MEMCPY3D copyParam;
    std::memset(&copyParam, 0, sizeof(copyParam));
    copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyParam.dstArray = *outArr;
    copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
    copyParam.srcHost = data;
    copyParam.srcPitch = w * texelBytes;
    copyParam.srcHeight = h;
    copyParam.WidthInBytes = copyParam.srcPitch;
    copyParam.Height = h;
    copyParam.Depth = d;
    r = cuMemcpy3DAsync(&copyParam, nullptr);
    if (r != CUDA_SUCCESS)
        return r;

    CUDA_RESOURCE_DESC resDesc;
    std::memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
    resDesc.res.array.hArray = *outArr;

    CUDA_TEXTURE_DESC texDesc;
    std::memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.filterMode = CU_TR_FILTER_MODE_LINEAR;
    return cuTexObjectCreate(outTex, &resDesc, &texDesc, nullptr);
}

PXG_TEST(p3ShimTexture3d)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }

    const PxgKernelDef manifest[] = { { "npTex3dBindless.spv", "tex3dBindlessSample", 0u, 0u } };
    PxgCudaShim shim(dev, manifest, 1, &texLoader);
    physx::gpuslang::pxgSetVulkanDriverShim(&shim);

    std::mt19937 rng(0x5DF0000u);

    // A DENSE float volume and a QUANTISED 8-bit one, which is exactly the pair a
    // sparse SDF carries: the coarse background grid is CU_AD_FORMAT_FLOAT and the
    // narrow-band subgrid texture is CU_AD_FORMAT_UNSIGNED_INT8
    // (PxgGeometryManager.cpp:657/663).  Different extents so a crossed handle is
    // visible in the queried size and not only in the values.
    const uint32_t W0 = 9, H0 = 5, D0 = 6;
    const uint32_t W1 = 4, H1 = 7, D1 = 3;
    std::vector<float> dense(size_t(W0) * H0 * D0);
    for (auto& v : dense)
        v = -2.5f + 5.0f * float(rng() % 1000000u) / 1000000.f;
    std::vector<uint8_t> fine(size_t(W1) * H1 * D1);
    for (auto& v : fine)
        v = uint8_t(rng() % 256u);

    CUtexObject texDense = 0, texFine = 0;
    CUarray arrDense = nullptr, arrFine = nullptr;
    PXG_CHECK_MSG(engineUpload(&texDense, &arrDense, CU_AD_FORMAT_FLOAT, W0, H0, D0,
                               dense.data(), sizeof(float)) == CUDA_SUCCESS,
                  "the engine's float SDF upload sequence failed");
    PXG_CHECK_MSG(engineUpload(&texFine, &arrFine, CU_AD_FORMAT_UNSIGNED_INT8, W1, H1, D1,
                               fine.data(), 1) == CUDA_SUCCESS,
                  "the engine's 8-bit subgrid upload sequence failed");
    PXG_CHECK_MSG(texDense != 0 && texFine != 0 && texDense != texFine,
                  "the two uploads must yield two distinct non-null texture objects (%llu, %llu)",
                  (unsigned long long)texDense, (unsigned long long)texFine);

    // The engine DROPS the CUarray and keeps sampling the texture
    // (PxgGeometryManager keeps only the CUtexObject in PxgTriangleMesh).  If
    // destroying the array took the image with it, everything below would read
    // zeros -- so this is asserted by doing it, here, before the sampling.
    PXG_CHECK_MSG(cuArrayDestroy(arrDense) == CUDA_SUCCESS, "cuArrayDestroy(dense) failed");
    PXG_CHECK_MSG(cuArrayDestroy(arrFine) == CUDA_SUCCESS, "cuArrayDestroy(fine) failed");

    // Sample every texel centre of both volumes through the handles the engine
    // would have stored, interleaved.
    std::vector<uint32_t> sHandle;
    std::vector<F4> sCoord;
    std::vector<uint32_t> sVol;
    std::vector<size_t> sIdx;
    {
        std::vector<std::pair<F4, size_t>> per[2];
        for (uint32_t z = 0; z < D0; ++z)
            for (uint32_t y = 0; y < H0; ++y)
                for (uint32_t x = 0; x < W0; ++x)
                    per[0].push_back({ F4{ x + 0.5f, y + 0.5f, z + 0.5f, 0.f },
                                       (size_t(z) * H0 + y) * W0 + x });
        for (uint32_t z = 0; z < D1; ++z)
            for (uint32_t y = 0; y < H1; ++y)
                for (uint32_t x = 0; x < W1; ++x)
                    per[1].push_back({ F4{ x + 0.5f, y + 0.5f, z + 0.5f, 0.f },
                                       (size_t(z) * H1 + y) * W1 + x });
        size_t c[2] = { 0, 0 };
        while (c[0] < per[0].size() || c[1] < per[1].size())
            for (uint32_t v = 0; v < 2; ++v)
            {
                if (c[v] >= per[v].size())
                    continue;
                sHandle.push_back(uint32_t(v == 0 ? texDense : texFine));
                sCoord.push_back(per[v][c[v]].first);
                sVol.push_back(v);
                sIdx.push_back(per[v][c[v]].second);
                ++c[v];
            }
    }
    const uint32_t N = uint32_t(sHandle.size());

    std::vector<float> outInit(N, -999.f);
    std::vector<U4> sizeInit(N, U4{ 0u, 0u, 0u, 0u });
    const PxgBufferHandle hHandles = createBufferFrom(dev, sHandle);
    const PxgBufferHandle hCoords = createBufferFrom(dev, sCoord);
    const PxgBufferHandle hOut = createBufferFrom(dev, outInit);
    const PxgBufferHandle hSize = createBufferFrom(dev, sizeInit);
    struct P
    {
        uint64_t handles, coords, outValue, outSize;
        uint32_t numSamples, pad;
    } prm = { dev->getBufferDeviceAddress(hHandles), dev->getBufferDeviceAddress(hCoords),
              dev->getBufferDeviceAddress(hOut), dev->getBufferDeviceAddress(hSize), N, 0u };
    {
        Kernel k(Kernel::Bindless{}, "npTex3dBindless.spv", "tex3dBindlessSample", {}, sizeof(P));
        PXG_CHECK_MSG(k.isValid(), "bindless pipeline creation succeeds");
        if (!k.isValid())
            return;
        k.dispatch((N + 63) / 64, {}, prm);
    }
    dev->flush();
    const auto got = downloadBuffer<float>(dev, hOut, N);
    const auto gotSize = downloadBuffer<U4>(dev, hSize, N);

    size_t bad = 0, sizeBad = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t v = sVol[i];
        const uint32_t w = v == 0 ? W0 : W1, h = v == 0 ? H0 : H1, d = v == 0 ? D0 : D1;
        if (gotSize[i].x != w || gotSize[i].y != h || gotSize[i].z != d)
        {
            if (sizeBad < 4)
                std::printf("    sample %u (vol %u): queried %ux%ux%u, uploaded %ux%ux%u\n",
                            i, v, gotSize[i].x, gotSize[i].y, gotSize[i].z, w, h, d);
            ++sizeBad;
            continue;
        }
        bool ok;
        if (v == 0)
        {
            uint32_t a, b;
            const float want = dense[sIdx[i]];
            std::memcpy(&a, &got[i], 4);
            std::memcpy(&b, &want, 4);
            ok = a == b;
        }
        else
        {
            // The 8-bit texture is read as a NORMALIZED float, which is what
            // SparseSDFTexture::applySubgridSdfScale then rescales -- so what has
            // to hold is that the sample identifies its stored byte.
            ok = std::lround(double(got[i]) * 255.0) == long(fine[sIdx[i]]);
        }
        if (!ok)
        {
            if (bad < 4)
                std::printf("    sample %u (vol %u, texel %zu): got %.9g\n", i, v, sIdx[i], got[i]);
            ++bad;
        }
    }
    PXG_CHECK_MSG(sizeBad == 0, "%zu samples resolved to the wrong texture object", sizeBad);
    PXG_CHECK_MSG(bad == 0, "%zu of %u texel-centre samples do not match the uploaded volume",
                  bad, N);

    // ---- the refusals ----
    //
    // Each of these would, if accepted, change SAMPLED VALUES rather than fail:
    // a two-channel array would halve the extent, a wrap sampler would fold
    // out-of-range SDF lookups back inside the mesh, a point filter would drop the
    // interpolation the SDF depends on, an integer read would skip the [0,1]
    // promotion applySubgridSdfScale assumes, and a sub-region copy would place
    // the data somewhere other than where the sample looks.
    {
        CUDA_ARRAY3D_DESCRIPTOR ad;
        std::memset(&ad, 0, sizeof(ad));
        ad.Format = CU_AD_FORMAT_FLOAT;
        ad.NumChannels = 2;
        ad.Width = 4; ad.Height = 4; ad.Depth = 4;
        CUarray a = reinterpret_cast<CUarray>(uintptr_t(0xABCD));
        PXG_CHECK_MSG(cuArray3DCreate(&a, &ad) != CUDA_SUCCESS,
                      "a two-channel array must be refused, not silently narrowed");
        PXG_CHECK_MSG(a == nullptr, "a refused cuArray3DCreate must null the out-handle");

        ad.NumChannels = 1;
        ad.Format = CU_AD_FORMAT_SIGNED_INT32;
        PXG_CHECK_MSG(cuArray3DCreate(&a, &ad) != CUDA_SUCCESS,
                      "a format outside {FLOAT, UINT8, UINT16} must be refused");
    }
    {
        // A real array, then descriptors that ask for something the shared
        // sampler cannot give.
        CUDA_ARRAY3D_DESCRIPTOR ad;
        std::memset(&ad, 0, sizeof(ad));
        ad.Format = CU_AD_FORMAT_FLOAT;
        ad.NumChannels = 1;
        ad.Width = 4; ad.Height = 4; ad.Depth = 4;
        CUarray a = nullptr;
        PXG_CHECK_MSG(cuArray3DCreate(&a, &ad) == CUDA_SUCCESS, "control array creation failed");

        std::vector<float> tiny(64, 0.25f);
        CUDA_MEMCPY3D cp;
        std::memset(&cp, 0, sizeof(cp));
        cp.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        cp.dstArray = a;
        cp.srcMemoryType = CU_MEMORYTYPE_HOST;
        cp.srcHost = tiny.data();
        cp.srcPitch = 4 * sizeof(float);
        cp.srcHeight = 4;
        cp.WidthInBytes = cp.srcPitch;
        cp.Height = 4;
        cp.Depth = 4;

        CUDA_MEMCPY3D off = cp;
        off.dstXInBytes = 4;
        PXG_CHECK_MSG(cuMemcpy3D(&off) != CUDA_SUCCESS,
                      "a sub-region copy must be refused; this shim only does whole arrays");
        CUDA_MEMCPY3D pitched = cp;
        pitched.srcPitch = cp.WidthInBytes + 4;
        PXG_CHECK_MSG(cuMemcpy3D(&pitched) != CUDA_SUCCESS,
                      "a padded source pitch must be refused, not read as tightly packed");
        PXG_CHECK_MSG(cuMemcpy3D(&cp) == CUDA_SUCCESS, "the well-formed control copy failed");

        CUDA_RESOURCE_DESC rd;
        std::memset(&rd, 0, sizeof(rd));
        rd.resType = CU_RESOURCE_TYPE_ARRAY;
        rd.res.array.hArray = a;
        CUDA_TEXTURE_DESC td;
        std::memset(&td, 0, sizeof(td));
        td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
        td.filterMode = CU_TR_FILTER_MODE_LINEAR;

        CUtexObject t = 0;
        CUDA_TEXTURE_DESC wrapZ = td;
        wrapZ.addressMode[2] = CU_TR_ADDRESS_MODE_WRAP;
        PXG_CHECK_MSG(cuTexObjectCreate(&t, &rd, &wrapZ, nullptr) != CUDA_SUCCESS,
                      "a wrap address mode must be refused; the shared sampler clamps");
        CUDA_TEXTURE_DESC pointFilter = td;
        pointFilter.filterMode = CU_TR_FILTER_MODE_POINT;
        PXG_CHECK_MSG(cuTexObjectCreate(&t, &rd, &pointFilter, nullptr) != CUDA_SUCCESS,
                      "a point filter must be refused; the shared sampler is linear");
        CUDA_TEXTURE_DESC asInt = td;
        asInt.flags = CU_TRSF_READ_AS_INTEGER;
        PXG_CHECK_MSG(cuTexObjectCreate(&t, &rd, &asInt, nullptr) != CUDA_SUCCESS,
                      "CU_TRSF_READ_AS_INTEGER must be refused; the UNORM formats promote");

        PXG_CHECK_MSG(cuTexObjectCreate(&t, &rd, &td, nullptr) == CUDA_SUCCESS,
                      "the well-formed control descriptor was refused");
        PXG_CHECK_MSG(t != 0, "the control texture object is null");

        // A copy that arrives after the image exists can never reach the GPU, so
        // it is refused rather than dropped.
        PXG_CHECK_MSG(cuMemcpy3D(&cp) != CUDA_SUCCESS,
                      "a copy into an array a texture was already built over must be refused");

        cuTexObjectDestroy(t);
        cuArrayDestroy(a);
    }

    cuTexObjectDestroy(texDense);
    cuTexObjectDestroy(texFine);
    dev->destroyBuffer(hHandles);
    dev->destroyBuffer(hCoords);
    dev->destroyBuffer(hOut);
    dev->destroyBuffer(hSize);
    physx::gpuslang::pxgSetVulkanDriverShim(nullptr);
}

} // namespace
