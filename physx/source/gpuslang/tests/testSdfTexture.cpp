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

// testSdfTexture.cpp -- P9.12c, the fixture for pxgSlangSdfTexture.
//
// THE ORACLE, and it is a real one for once.  Gu::SDF::decodeSparse
// (GuSDF.cpp:1340-1412) evaluates the same sparse SDF on the host at an integer
// fine-grid coordinate and makes every decision SparseSDFTexture::Sample makes:
// the coarse cell by integer division, the far-edge fix-up, decodeTriple's
// 10-bit unpack, the (subgridSize+1) block stride, and the coarse fall-back over
// a (nbX+1, nbY+1, nbZ+1) grid.  At a texel centre the trilinear weights are 0
// and 1, so the GPU sample must equal it.
//
// It is not linked into this binary -- it lives in PhysXCommon, and gpuslang-tests
// deliberately links only the backend and foundation.  So `sparseOracle` below is
// the P8.73-P8.80 arrangement: CONTROL FLOW TRANSCRIBED, with every primitive it
// rests on taken from the engine's own header -- Gu::SDF::decodeTriple,
// Gu::SDF::decodeSample, Gu::idx3D and Gu::DenseSDF::sampleSDFDirect are all
// PX_FORCE_INLINE there and are CALLED, not re-derived.
//
// AND IT HAS BEEN CHECKED AGAINST THE REAL FUNCTION.  A one-off program linked
// against libphysx-sdk.a ran this exact mirror against Gu::SDF::decodeSparse over
// the identical hand-built SDF: 2457 fine-grid samples, 6 resident cells and 24
// coarse-fall-back cells, 537 of them on a far edge -- ZERO mismatches, worst
// |diff| exactly 0.  So the expectation below is the engine's behaviour and not
// this file's reading of it; what is transcribed has been measured against the
// original once, offline, rather than asserted.
//
// THE SCENE, and every number in it distinct per the standing rule:
//   fine dims      12 x 8 x 20     (no two equal)
//   subgrid size   4               -> coarse cells 3 x 2 x 5 (no two equal)
//   coarse grid    4 x 3 x 6       (= cells + 1, which is what the engine uploads)
//   block grid     3 x 2 x 1       -> subgrid texture 15 x 10 x 5
//   resident       6 of 30 cells, NOT the first six and NOT contiguous, mapped
//                  to block coordinates in a DIFFERENT order again, so a port
//                  that ignores the start slot, or that reads it with the fine
//                  dimensions, lands on another cell's data rather than on zeros.
//                  THREE OF THEM HAVE y == 1, which is not decoration: the start
//                  slot index is z*w*h + y*w + x, so swapping w and h leaves the
//                  z term alone and changes only the y term -- with every
//                  resident cell at y == 0 (as the first draft had it) that
//                  mutant is invisible, and `t12c-idx-wh-swapped` survived.
// The dense mesh beside it is 7 x 5 x 9 with subgridSize 0 -- the branch whose
// margin is 0.5 instead of 0.0.

#include "pxgTestFramework.h"

#include "PxgCudaShim.h"
#include "PxgCudaToolkitShim.h"
#include "GuSDF.h"

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
using namespace physx;

struct F4 { float x, y, z, w; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// sizeof(Gu::BV32DataPacked), GuBV32.h -- the stride the blob walk must use.
static const uint32_t BV32_STRIDE = 1168u;

// ---------------------------------------------------------------------------
// The hand-built SDF.
// ---------------------------------------------------------------------------
struct SdfScene
{
    uint32_t dx, dy, dz;            // fine dims
    uint32_t sg;                    // subgrid size (0 == dense)
    uint32_t nbx, nby, nbz;         // coarse cells
    uint32_t blkx, blky, blkz;      // subgrid block grid
    float smin, smax;
    PxVec3 meshLower;
    float spacing;
    std::vector<float> coarse;      // the FLOAT texture
    std::vector<uint32_t> starts;   // per coarse cell
    std::vector<uint8_t> sub;       // the 8-bit subgrid texture
};

static void buildSparse(SdfScene& s)
{
    s.dx = 12; s.dy = 8; s.dz = 20;
    s.sg = 4;
    s.nbx = s.dx / s.sg; s.nby = s.dy / s.sg; s.nbz = s.dz / s.sg;   // 3, 2, 5
    s.blkx = 3; s.blky = 2; s.blkz = 1;
    s.smin = -0.37f; s.smax = 0.91f;
    s.meshLower = PxVec3(-0.3f, 0.17f, 0.42f);
    s.spacing = 0.11f;

    std::mt19937 rng(0x5DF12Cu);
    s.coarse.resize(size_t(s.nbx + 1) * (s.nby + 1) * (s.nbz + 1));
    for (auto& v : s.coarse)
        v = -1.5f + 3.0f * float(rng() % 1000000u) / 1000000.f;
    s.sub.resize(size_t(s.blkx * (s.sg + 1)) * (s.blky * (s.sg + 1)) * (s.blkz * (s.sg + 1)));
    for (auto& v : s.sub)
        v = uint8_t(rng() % 256u);

    s.starts.assign(size_t(s.nbx) * s.nby * s.nbz, 0xFFFFFFFFu);
    const uint32_t cells[6] = { 1u, 4u, 12u, 16u, 22u, 27u };
    const uint32_t blocks[6][3] = { {0,0,0}, {2,1,0}, {1,0,0}, {0,1,0}, {2,0,0}, {1,1,0} };
    for (int i = 0; i < 6; ++i)
        s.starts[cells[i]] = blocks[i][0] | (blocks[i][1] << 10) | (blocks[i][2] << 20);
}

static void buildDense(SdfScene& s)
{
    s.dx = 7; s.dy = 5; s.dz = 9;
    s.sg = 0;
    s.nbx = s.nby = s.nbz = 0;
    s.blkx = s.blky = s.blkz = 0;
    s.smin = -0.23f; s.smax = 0.64f;
    s.meshLower = PxVec3(1.25f, -0.4f, 3.1f);
    s.spacing = 0.07f;
    std::mt19937 rng(0xDE45Eu);
    s.coarse.resize(size_t(s.dx) * s.dy * s.dz);
    for (auto& v : s.coarse)
        v = -2.0f + 4.0f * float(rng() % 1000000u) / 1000000.f;
    s.starts.clear();
    s.sub.clear();
}

// GuSDF.cpp:1340-1412 (decodeSparse2), CONTROL FLOW TRANSCRIBED with every
// primitive it rests on CALLED from the engine's own header -- decodeTriple,
// decodeSample, idx3D and DenseSDF::sampleSDFDirect are all PX_FORCE_INLINE
// there.  Verified bit-identical to Gu::SDF::decodeSparse itself offline over
// this exact scene; see the file header.
//
// It takes the SCENE rather than a Gu::SDF because instantiating one would drag
// in that class's out-of-line destructor, and with it the whole GuSDF.cpp
// translation unit (AABB trees, winding numbers, mesh analysis) into a binary
// that deliberately links only the backend and foundation.  Nothing is lost:
// none of the four primitives above needs an instance.
static float sparseOracle(const SdfScene& s, int32_t xx, int32_t yy, int32_t zz)
{
    if (xx < 0 || yy < 0 || zz < 0 || xx > int32_t(s.dx) || yy > int32_t(s.dy) ||
        zz > int32_t(s.dz))
        return 1.0f;

    const uint32_t nbX = s.dx / s.sg;
    const uint32_t nbY = s.dy / s.sg;
    const uint32_t nbZ = s.dz / s.sg;

    uint32_t xBase = uint32_t(xx) / s.sg;
    uint32_t yBase = uint32_t(yy) / s.sg;
    uint32_t zBase = uint32_t(zz) / s.sg;
    uint32_t x = uint32_t(xx) % s.sg;
    uint32_t y = uint32_t(yy) % s.sg;
    uint32_t z = uint32_t(zz) % s.sg;

    if (xBase == nbX) { --xBase; x = s.sg; }
    if (yBase == nbY) { --yBase; y = s.sg; }
    if (zBase == nbZ) { --zBase; z = s.sg; }

    const uint32_t startId = s.starts[zBase * nbX * nbY + yBase * nbX + xBase];
    if (startId != 0xFFFFFFFFu)
    {
        uint32_t bx, by, bz;
        Gu::SDF::decodeTriple(startId, bx, by, bz);                      // ENGINE
        bx *= (s.sg + 1); by *= (s.sg + 1); bz *= (s.sg + 1);
        const uint32_t w = s.blkx * (s.sg + 1);
        const uint32_t h = s.blky * (s.sg + 1);
        const uint32_t index = Gu::idx3D(bx + x, by + y, bz + z, w, h);  // ENGINE
        return Gu::SDF::decodeSample(s.sub.data(), index, 1, s.smin, s.smax);  // ENGINE
    }
    Gu::DenseSDF coarse(nbX + 1, nbY + 1, nbZ + 1, const_cast<float*>(s.coarse.data()));
    const float sc = 1.0f / s.sg;
    return coarse.sampleSDFDirect(PxVec3(xBase + x * sc, yBase + y * sc, zBase + z * sc));  // ENGINE
}

// ---------------------------------------------------------------------------
// The cooked-mesh blob, laid out exactly as PxgGeometryManager does
// (PxgGeometryManager.cpp:520-700).  The sections the SDF walk must SKIP are
// filled with a recognisable pattern rather than zeros, so a wrong stride lands
// on junk that cannot be mistaken for a plausible field.
// ---------------------------------------------------------------------------
struct BlobLayout
{
    std::vector<uint8_t> bytes;
    size_t texObjOffset = 0;        // where the coarse handle goes
    size_t texObjSparseOffset = 0;  // where the subgrid handle goes
    size_t sdfDimsOffset = 0;
};

static BlobLayout layOutBlob(const SdfScene& s, uint32_t numVerts, uint32_t numTris,
                             uint32_t nbPackedNodes, uint32_t projectionFlag, uint32_t junkSparseHandle)
{
    BlobLayout b;
    auto put32 = [&b](uint32_t v) { const size_t o = b.bytes.size(); b.bytes.resize(o + 4); std::memcpy(&b.bytes[o], &v, 4); };
    auto putF = [&](float v) { uint32_t u; std::memcpy(&u, &v, 4); put32(u); };
    auto junk = [&b](size_t n, uint8_t seed) { const size_t o = b.bytes.size(); b.bytes.resize(o + n);
                                               for (size_t i = 0; i < n; ++i) b.bytes[o + i] = uint8_t(seed + (i * 37u)); };

    put32(numVerts); put32(numTris); put32(0xABCDEF01u); put32(nbPackedNodes);  // .z is NOT the node count
    junk(size_t(nbPackedNodes) * BV32_STRIDE, 0x11);   // BV32 tree
    junk(size_t(numVerts) * 16, 0x22);                 // float4 verts
    junk(size_t(numTris) * 16, 0x33);                  // uint4 indices
    junk(size_t(numTris) * 16, 0x44);                  // uint4 adjacencies
    junk(size_t(numTris) * 4, 0x55);                   // uint32 faceRemap
    junk(size_t(numVerts + 1) * 4, 0x66);              // accumulatedTrianglesRef -- numVerts+1
    junk(size_t(numTris) * 12, 0x77);                  // trianglesReferences[3*numTris]
    while (b.bytes.size() % 16 != 0)                   // the one alignment step
        b.bytes.push_back(0x99);

    b.sdfDimsOffset = b.bytes.size();
    put32(s.dx); put32(s.dy); put32(s.dz); put32(projectionFlag);
    putF(s.meshLower.x); putF(s.meshLower.y); putF(s.meshLower.z); putF(s.spacing);
    put32(s.sg); put32(f2u(s.smin)); put32(f2u(s.smax)); put32(uint32_t(s.starts.size()));

    b.texObjOffset = b.bytes.size();
    put32(0u); put32(0u);                              // mTexObject, filled after upload
    b.texObjSparseOffset = b.bytes.size();
    // On the DENSE path the engine never writes this word -- it stays whatever the
    // blob held.  The fixture puts a recognisable value there and requires the
    // port to read exactly it, which proves the field is read at the right offset
    // without pretending the engine zeroes it.
    put32(junkSparseHandle); put32(0u);
    for (uint32_t v : s.starts)
        put32(v);
    return b;
}

// One struct for both entries -- see the note in npSdfTexture.slang about why two
// push-constant blocks in one module is a trap rather than a style question.
struct P_Sdf
{
    uint64_t meshPtr, outMesh, outTex, coords, outValue;
    uint32_t numSamples, pad;
};

PXG_TEST(sdfTexture)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsBindlessTextures())
    {
        std::printf("  NOTE: backend has no descriptor indexing -> skip\n");
        return;
    }

    const PxgKernelDef manifest[] = { { "npSdfTexture.spv", "sdfTexProbe", 0u, 0u } };
    static auto loader = [](const char* f) { return loadShaderBlob(f); };
    PxgCudaShim shim(dev, manifest, 1, +loader);
    physx::gpuslang::pxgSetVulkanDriverShim(&shim);

    std::vector<PxgBufferHandle> owned;
    auto mkBuf = [&](const std::vector<uint8_t>& v) {
        PxgBufferHandle h = createBufferFrom(dev, v);
        owned.push_back(h);
        return h;
    };

    // Uploads a volume through the ENGINE's own call sequence, so this fixture
    // exercises the P9.12b path a second time as a side effect.
    auto uploadTex = [&](CUarray_format fmt, uint32_t w, uint32_t h, uint32_t d,
                         const void* data, size_t texelBytes) -> uint64_t {
        CUDA_ARRAY3D_DESCRIPTOR ad; std::memset(&ad, 0, sizeof(ad));
        ad.Format = fmt; ad.NumChannels = 1; ad.Width = w; ad.Height = h; ad.Depth = d;
        CUarray arr = nullptr;
        if (cuArray3DCreate(&arr, &ad) != CUDA_SUCCESS) return 0;
        CUDA_MEMCPY3D cp; std::memset(&cp, 0, sizeof(cp));
        cp.dstMemoryType = CU_MEMORYTYPE_ARRAY; cp.dstArray = arr;
        cp.srcMemoryType = CU_MEMORYTYPE_HOST; cp.srcHost = data;
        cp.srcPitch = w * texelBytes; cp.srcHeight = h;
        cp.WidthInBytes = cp.srcPitch; cp.Height = h; cp.Depth = d;
        if (cuMemcpy3DAsync(&cp, nullptr) != CUDA_SUCCESS) return 0;
        CUDA_RESOURCE_DESC rd; std::memset(&rd, 0, sizeof(rd));
        rd.resType = CU_RESOURCE_TYPE_ARRAY; rd.res.array.hArray = arr;
        CUDA_TEXTURE_DESC td; std::memset(&td, 0, sizeof(td));
        td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
        td.filterMode = CU_TR_FILTER_MODE_LINEAR;
        CUtexObject t = 0;
        if (cuTexObjectCreate(&t, &rd, &td, nullptr) != CUDA_SUCCESS) return 0;
        cuArrayDestroy(arr);
        return uint64_t(t);
    };

    // =====================================================================
    // Case 1: the SPARSE mesh.
    // =====================================================================
    SdfScene sp;
    buildSparse(sp);

    const uint32_t JUNK_SPARSE = 0u;   // overwritten below for the sparse case
    // numVerts/numTris/nbPackedNodes are not arbitrary.  The blob's pre-alignment
    // size is 20 + 1168*N + 20*V + 48*T, and rounding UP to 16 absorbs any stride
    // error that does not cross a 16-boundary downwards -- which is how
    // `b4-accum-no-plus-1` (a FOUR-byte error) survived the first sweep.  16/21/3
    // puts the true total at 4852, i.e. 4 past a boundary, so a four-byte
    // under-count lands on 4848 and the alignment can no longer hide it, while
    // 12 bytes of padding still exercise the alignment step itself.
    BlobLayout blobSp = layOutBlob(sp, 16u, 21u, 3u, 1u, JUNK_SPARSE);

    const uint64_t hCoarse = uploadTex(CU_AD_FORMAT_FLOAT, sp.nbx + 1, sp.nby + 1, sp.nbz + 1,
                                       sp.coarse.data(), sizeof(float));
    const uint64_t hSub = uploadTex(CU_AD_FORMAT_UNSIGNED_INT8, sp.blkx * (sp.sg + 1),
                                    sp.blky * (sp.sg + 1), sp.blkz * (sp.sg + 1),
                                    sp.sub.data(), 1);
    PXG_CHECK_MSG(hCoarse != 0 && hSub != 0 && hCoarse != hSub,
                  "the two sparse textures must upload to two distinct slots (%llu, %llu)",
                  (unsigned long long)hCoarse, (unsigned long long)hSub);
    std::memcpy(&blobSp.bytes[blobSp.texObjOffset], &hCoarse, 8);
    std::memcpy(&blobSp.bytes[blobSp.texObjSparseOffset], &hSub, 8);

    const PxgBufferHandle bBlobSp = mkBuf(blobSp.bytes);
    const uint64_t aBlobSp = dev->getBufferDeviceAddress(bBlobSp);

    // ---- the probe: did the walk land on the right bytes? ----
    {
        std::vector<uint32_t> outMesh(16, 0xEEEEEEEEu), outTex(24, 0xEEEEEEEEu);
        const PxgBufferHandle bM = createBufferFrom(dev, outMesh);
        const PxgBufferHandle bT = createBufferFrom(dev, outTex);
        owned.push_back(bM); owned.push_back(bT);
        P_Sdf prm = {}; prm.meshPtr = aBlobSp; prm.outMesh = dev->getBufferDeviceAddress(bM);
        prm.outTex = dev->getBufferDeviceAddress(bT);
        {
            Kernel k(Kernel::Bindless{}, "npSdfTexture.spv", "sdfTexProbe", {}, sizeof(P_Sdf));
            PXG_CHECK_MSG(k.isValid(), "bindless pipeline creation succeeds");
            if (!k.isValid()) return;
            k.dispatch(1u, {}, prm);
        }
        dev->flush();
        const auto m = downloadBuffer<uint32_t>(dev, bM, 16);
        const auto t = downloadBuffer<uint32_t>(dev, bT, 24);

        PXG_CHECK_MSG(m[0] == 1u, "the sparse mesh must report an SDF");
        PXG_CHECK_MSG(m[1] == sp.dx && m[2] == sp.dy && m[3] == sp.dz,
                      "sdfDims %u,%u,%u != %u,%u,%u -- the blob walk landed on the wrong bytes",
                      m[1], m[2], m[3], sp.dx, sp.dy, sp.dz);
        PXG_CHECK_MSG(m[4] == 1u, "sdfDims.w is the projection FLAG and must be read through (%u)", m[4]);
        PXG_CHECK_MSG(m[5] == f2u(sp.meshLower.x) && m[6] == f2u(sp.meshLower.y) &&
                      m[7] == f2u(sp.meshLower.z), "meshLower mismatch");
        PXG_CHECK_MSG(m[8] == f2u(sp.spacing), "spacing mismatch");
        PXG_CHECK_MSG(m[9] == sp.sg, "subgridSize %u != %u", m[9], sp.sg);
        PXG_CHECK_MSG(m[10] == f2u(sp.smin) && m[11] == f2u(sp.smax), "subgrid value bounds mismatch");
        PXG_CHECK_MSG(m[12] == uint32_t(sp.starts.size()), "numStartSlots %u != %zu", m[12], sp.starts.size());
        PXG_CHECK_MSG(m[13] == uint32_t(hCoarse), "texObject %u != %llu", m[13], (unsigned long long)hCoarse);
        PXG_CHECK_MSG(m[14] == uint32_t(hSub), "texObjectSparse %u != %llu", m[14], (unsigned long long)hSub);
        // The strongest single check in the probe: the FIRST START SLOT, read
        // through the decoded pointer.  It is only right if every one of the
        // eight preceding section strides and the 16-byte alignment were right.
        PXG_CHECK_MSG(m[15] == sp.starts[0],
                      "subgridStarts[0] = 0x%08x, expected 0x%08x -- the blob walk is off",
                      m[15], sp.starts[0]);

        // initialize(): the sparse margin is 0.0, so the box is meshLower..
        // meshLower + dims*spacing exactly.
        const float lo[3] = { sp.meshLower.x, sp.meshLower.y, sp.meshLower.z };
        const float hi[3] = { sp.meshLower.x + sp.dx * sp.spacing,
                              sp.meshLower.y + sp.dy * sp.spacing,
                              sp.meshLower.z + sp.dz * sp.spacing };
        for (int i = 0; i < 3; ++i)
        {
            PXG_CHECK_MSG(t[i] == f2u(lo[i]), "sdfBoxLower[%d] mismatch (sparse margin must be 0)", i);
            PXG_CHECK_MSG(t[4 + i] == f2u(hi[i]), "sdfBoxHigher[%d] mismatch", i);
        }
        PXG_CHECK_MSG(t[3] == f2u(sp.spacing), "sdfDx mismatch");
        PXG_CHECK_MSG(t[7] == f2u(1.0f / sp.spacing), "invSdfDx mismatch");
        PXG_CHECK_MSG(t[10] == sp.nbx && t[11] == sp.nby && t[12] == sp.nbz,
                      "coarseSize %u,%u,%u != %u,%u,%u", t[10], t[11], t[12], sp.nbx, sp.nby, sp.nbz);
        PXG_CHECK_MSG(t[14] == f2u(1.0f / float(sp.sg)), "fineToCoarse mismatch");
        PXG_CHECK_MSG(t[20] == f2u(sp.smax - sp.smin),
                      "subgridsSdfValueRange must be max - min, computed the same way");
    }

    // ---- the samples, against the oracle ----
    {
        std::vector<F4> coords;
        std::vector<float> want;
        std::vector<int> resident;
        for (int32_t z = 0; z <= int32_t(sp.dz); ++z)
            for (int32_t y = 0; y <= int32_t(sp.dy); ++y)
                for (int32_t x = 0; x <= int32_t(sp.dx); ++x)
                {
                    coords.push_back(F4{ float(x), float(y), float(z), 0.f });
                    want.push_back(sparseOracle(sp, x, y, z));
                    const uint32_t cx = std::min(uint32_t(x) / sp.sg, sp.nbx - 1);
                    const uint32_t cy = std::min(uint32_t(y) / sp.sg, sp.nby - 1);
                    const uint32_t cz = std::min(uint32_t(z) / sp.sg, sp.nbz - 1);
                    resident.push_back(sp.starts[cz * sp.nbx * sp.nby + cy * sp.nbx + cx] != 0xFFFFFFFFu);
                }
        const uint32_t N = uint32_t(coords.size());
        std::vector<float> outInit(N, -777.f);
        const PxgBufferHandle bC = createBufferFrom(dev, coords);
        const PxgBufferHandle bO = createBufferFrom(dev, outInit);
        owned.push_back(bC); owned.push_back(bO);
        P_Sdf prm = {}; prm.meshPtr = aBlobSp; prm.coords = dev->getBufferDeviceAddress(bC);
        prm.outValue = dev->getBufferDeviceAddress(bO); prm.numSamples = N;
        {
            Kernel k(Kernel::Bindless{}, "npSdfTexture.spv", "sdfTexSampleAt", {}, sizeof(P_Sdf));
            PXG_CHECK_MSG(k.isValid(), "bindless sample pipeline creation succeeds");
            if (!k.isValid()) return;
            k.dispatch((N + 63) / 64, {}, prm);
        }
        dev->flush();
        const auto got = downloadBuffer<float>(dev, bO, N);

        // Two different bounds, because the two paths carry different precision
        // and lumping them together would hide a regression in the tighter one:
        //   RESIDENT  -- an 8-bit UNORM texel read at its centre.  P9.12a measured
        //                the conversion as faithful to the stored byte, so what
        //                must hold is that the value round-trips to the same byte,
        //                i.e. |diff| below half a quantisation step of the range.
        //   COARSE    -- a FLOAT texture, interpolated.  The host reference is the
        //                engine's own PxTriLerp, the GPU is the hardware filter;
        //                P9.12a measured that at 3.25e-06 over a range of 8, so
        //                1e-5 of this scene's range is the bound.
        const double residentBound = 0.5 * double(sp.smax - sp.smin) / 255.0;
        const double coarseBound = 1e-5 * 3.0;
        size_t badR = 0, badC = 0, nR = 0, nC = 0;
        double worstR = 0.0, worstC = 0.0;
        for (uint32_t i = 0; i < N; ++i)
        {
            const double d = std::fabs(double(got[i]) - double(want[i]));
            if (resident[i])
            {
                ++nR;
                if (d > worstR) worstR = d;
                if (d > residentBound)
                {
                    if (badR < 5)
                        std::printf("    resident (%g,%g,%g): got %.9g want %.9g diff %.3g\n",
                                    coords[i].x, coords[i].y, coords[i].z, got[i], want[i], d);
                    ++badR;
                }
            }
            else
            {
                ++nC;
                if (d > worstC) worstC = d;
                if (d > coarseBound)
                {
                    if (badC < 5)
                        std::printf("    coarse (%g,%g,%g): got %.9g want %.9g diff %.3g\n",
                                    coords[i].x, coords[i].y, coords[i].z, got[i], want[i], d);
                    ++badC;
                }
            }
        }
        PXG_CHECK_MSG(nR > 0 && nC > 0,
                      "both the resident and the coarse-fall-back paths must be reached "
                      "(%zu resident, %zu coarse)", nR, nC);
        PXG_CHECK_MSG(badR == 0, "%zu of %zu RESIDENT samples disagree with the engine oracle "
                      "(worst %.3g, bound %.3g)", badR, nR, worstR, residentBound);
        PXG_CHECK_MSG(badC == 0, "%zu of %zu COARSE-FALL-BACK samples disagree with the engine "
                      "oracle (worst %.3g, bound %.3g)", badC, nC, worstC, coarseBound);
        std::printf("    [ note ] sparse: %zu resident (worst %.3g), %zu coarse (worst %.3g)\n",
                    nR, worstR, nC, worstC);
    }

    // =====================================================================
    // Case 2: the DENSE mesh -- the branch whose margin is 0.5, whose
    // fineToCoarse is +inf, and whose subgrid handle the engine never writes.
    // =====================================================================
    {
        SdfScene dn;
        buildDense(dn);
        const uint32_t JUNK = 0x0BADF00Du;
        BlobLayout blobDn = layOutBlob(dn, 5u, 9u, 1u, 0u, JUNK);
        const uint64_t hDense = uploadTex(CU_AD_FORMAT_FLOAT, dn.dx, dn.dy, dn.dz,
                                          dn.coarse.data(), sizeof(float));
        PXG_CHECK_MSG(hDense != 0, "the dense texture upload failed");
        std::memcpy(&blobDn.bytes[blobDn.texObjOffset], &hDense, 8);

        const PxgBufferHandle bBlobDn = mkBuf(blobDn.bytes);
        const uint64_t aBlobDn = dev->getBufferDeviceAddress(bBlobDn);

        std::vector<uint32_t> outMesh(16, 0xEEEEEEEEu), outTex(24, 0xEEEEEEEEu);
        const PxgBufferHandle bM = createBufferFrom(dev, outMesh);
        const PxgBufferHandle bT = createBufferFrom(dev, outTex);
        owned.push_back(bM); owned.push_back(bT);
        P_Sdf prm = {}; prm.meshPtr = aBlobDn; prm.outMesh = dev->getBufferDeviceAddress(bM);
        prm.outTex = dev->getBufferDeviceAddress(bT);
        {
            Kernel k(Kernel::Bindless{}, "npSdfTexture.spv", "sdfTexProbe", {}, sizeof(P_Sdf));
            if (!k.isValid()) return;
            k.dispatch(1u, {}, prm);
        }
        dev->flush();
        const auto m = downloadBuffer<uint32_t>(dev, bM, 16);
        const auto t = downloadBuffer<uint32_t>(dev, bT, 24);

        PXG_CHECK_MSG(m[0] == 1u, "the dense mesh must report an SDF");
        PXG_CHECK_MSG(m[9] == 0u, "the dense mesh's subgridSize must be 0 (got %u)", m[9]);
        PXG_CHECK_MSG(m[12] == 0u, "the dense mesh has no start slots (got %u)", m[12]);
        PXG_CHECK_MSG(m[14] == JUNK,
                      "the subgrid handle word must be read at its offset even on the dense path, "
                      "where the engine never writes it (got 0x%08x, wrote 0x%08x)", m[14], JUNK);

        // T1: the dense margin is 0.5, so BOTH ends move by half a voxel.
        const float lo[3] = { dn.meshLower.x + 0.5f * dn.spacing,
                              dn.meshLower.y + 0.5f * dn.spacing,
                              dn.meshLower.z + 0.5f * dn.spacing };
        const float hi[3] = { dn.meshLower.x + (dn.dx - 0.5f) * dn.spacing,
                              dn.meshLower.y + (dn.dy - 0.5f) * dn.spacing,
                              dn.meshLower.z + (dn.dz - 0.5f) * dn.spacing };
        for (int i = 0; i < 3; ++i)
        {
            PXG_CHECK_MSG(t[i] == f2u(lo[i]), "dense sdfBoxLower[%d]: the margin must be 0.5", i);
            PXG_CHECK_MSG(t[4 + i] == f2u(hi[i]), "dense sdfBoxHigher[%d]: the margin must be 0.5", i);
        }
        // T2: PxMax(1u, subgridSize) is what keeps this from dividing by zero.
        PXG_CHECK_MSG(t[10] == dn.dx && t[11] == dn.dy && t[12] == dn.dz,
                      "dense coarseSize must be sdfDims / max(1, 0) = sdfDims (got %u,%u,%u)",
                      t[10], t[11], t[12]);
        // T3: fineToCoarse is 1/0 on this path.  Transcribed, and pinned, because
        // a port that "fixes" it to 1 or to 0 has diverged from the CUDA even
        // though nothing on this path reads it.
        float f2c; std::memcpy(&f2c, &t[14], 4);
        PXG_CHECK_MSG(std::isinf(f2c) && f2c > 0.0f,
                      "dense fineToCoarse is 1.0f/0 = +inf in the CUDA and must stay so (got %g)", f2c);

        // Samples: every texel centre against the dense grid itself.
        std::vector<F4> coords;
        std::vector<float> want;
        for (uint32_t z = 0; z < dn.dz; ++z)
            for (uint32_t y = 0; y < dn.dy; ++y)
                for (uint32_t x = 0; x < dn.dx; ++x)
                {
                    coords.push_back(F4{ float(x), float(y), float(z), 0.f });
                    want.push_back(dn.coarse[(size_t(z) * dn.dy + y) * dn.dx + x]);
                }
        const uint32_t N = uint32_t(coords.size());
        std::vector<float> outInit(N, -777.f);
        const PxgBufferHandle bC = createBufferFrom(dev, coords);
        const PxgBufferHandle bO = createBufferFrom(dev, outInit);
        owned.push_back(bC); owned.push_back(bO);
        P_Sdf sprm = {}; sprm.meshPtr = aBlobDn; sprm.coords = dev->getBufferDeviceAddress(bC);
        sprm.outValue = dev->getBufferDeviceAddress(bO); sprm.numSamples = N;
        {
            Kernel k(Kernel::Bindless{}, "npSdfTexture.spv", "sdfTexSampleAt", {}, sizeof(P_Sdf));
            if (!k.isValid()) return;
            k.dispatch((N + 63) / 64, {}, sprm);
        }
        dev->flush();
        const auto got = downloadBuffer<float>(dev, bO, N);
        size_t bad = 0;
        for (uint32_t i = 0; i < N; ++i)
            if (f2u(got[i]) != f2u(want[i]))
            {
                if (bad < 5)
                    std::printf("    dense (%g,%g,%g): got %.9g want %.9g\n",
                                coords[i].x, coords[i].y, coords[i].z, got[i], want[i]);
                ++bad;
            }
        // A FLOAT texture at a texel centre has weights 0 and 1, so this is exact.
        PXG_CHECK_MSG(bad == 0, "%zu of %u dense texel-centre samples are not bit-exact", bad, N);
        std::printf("    [ note ] dense: %u texel centres bit-exact\n", N);
    }

    for (PxgBufferHandle h : owned)
        dev->destroyBuffer(h);
    physx::gpuslang::pxgSetVulkanDriverShim(nullptr);
}

} // namespace
