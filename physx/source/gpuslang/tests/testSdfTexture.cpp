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
// deliberately links only the backend and foundation.  So `sparseOracle` is the
// P8.73-P8.80 arrangement: CONTROL FLOW TRANSCRIBED, with every primitive it
// rests on taken from the engine's own header -- Gu::SDF::decodeTriple,
// Gu::SDF::decodeSample, Gu::idx3D and Gu::DenseSDF::sampleSDFDirect are all
// PX_FORCE_INLINE there and are CALLED, not re-derived.  It, the scene it runs
// over and the cooked-blob writer now live in pxgSdfSceneRef.h, shared with the
// P9.12d fixture; see that header for the scene's numbers and for the offline
// measurement against the real Gu::SDF::decodeSparse.
//
// P9.96: NO TOLERANCE IS LEFT.  pxgTex3D evaluates CUDA's linear filter itself, so
// every sample here is bitwise:
//   * RESIDENT points against sparseOracle -- the engine's decodeSample, because the
//     driver's UNORM read is the engine's spelling (npTex3dBindlessUnormExhaustive);
//   * every point against sdfSampleRef (pxgSdfSceneRef.h), the host transcription of
//     Sample() over pxgtex::tex3D, which is itself anchored to sparseOracle bit for
//     bit on the resident points and on the coarse points with zero weights;
//   * COARSE points keep their bound against sparseOracle as a secondary check: its
//     coarse branch is the engine's nested PxTriLerp at quarter weights, which rounds
//     differently from the guide's eight-term sum (pxgSlangTex3d.slang K2);
//   * a FRACTIONAL pass -- the first check of Sample() between texels -- over both
//     meshes: random f, block borders, subgrid faces, and exact weight TIES
//     f = k + (j + 0.5)/256 on the dense mesh, which is what lets this fixture tell
//     round-half-up from round-to-even and ties-down.
// Bite asserts over the whole sample set run on the host at the END, after the three
// launches (only the sdfSampleRef anchor to decodeSparse runs before its launch).

#include "pxgTestFramework.h"

#include "PxgCudaShim.h"
#include "PxgCudaToolkitShim.h"
#include "pxgSdfSceneRef.h"

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
using namespace pxgsdfref;

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

    // pxgSdfSceneRef.h's sdfUploadTex goes through the ENGINE's own call
    // sequence, so this fixture exercises the P9.12b path a second time as a
    // side effect.
    auto uploadTex = sdfUploadTex;

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

    const SdfRefVols spVols = sdfRefVols(sp);
    // Everything compared bitwise against sdfSampleRef, for the bite asserts below.
    struct RefSample { const SdfScene* s; const SdfRefVols* v; PxVec3 f; };
    std::vector<RefSample> allRef;

    auto runAt = [&](uint64_t blobAddr, const std::vector<F4>& coords, std::vector<float>& got) {
        const uint32_t N = uint32_t(coords.size());
        const PxgBufferHandle bC = createBufferFrom(dev, coords);
        const PxgBufferHandle bO = createBufferFrom(dev, std::vector<float>(N, -777.f));
        owned.push_back(bC); owned.push_back(bO);
        P_Sdf prm = {}; prm.meshPtr = blobAddr; prm.coords = dev->getBufferDeviceAddress(bC);
        prm.outValue = dev->getBufferDeviceAddress(bO); prm.numSamples = N;
        {
            Kernel k(Kernel::Bindless{}, "npSdfTexture.spv", "sdfTexSampleAt", {}, sizeof(P_Sdf));
            PXG_CHECK_MSG(k.isValid(), "bindless sample pipeline creation succeeds");
            if (!k.isValid()) return false;
            k.dispatch((N + 63) / 64, {}, prm);
        }
        dev->flush();
        got = downloadBuffer<float>(dev, bO, N);
        return true;
    };

    // ---- the integer fine grid, against the engine oracle and sdfSampleRef ----
    {
        std::vector<F4> coords;
        std::vector<float> want, ref;
        std::vector<int> resident, alpha0;
        for (int32_t z = 0; z <= int32_t(sp.dz); ++z)
            for (int32_t y = 0; y <= int32_t(sp.dy); ++y)
                for (int32_t x = 0; x <= int32_t(sp.dx); ++x)
                {
                    const PxVec3 f{ float(x), float(y), float(z) };
                    coords.push_back(F4{ f.x, f.y, f.z, 0.f });
                    want.push_back(sparseOracle(sp, x, y, z));
                    ref.push_back(sdfSampleRef(sp, spVols, f));
                    allRef.push_back({ &sp, &spVols, f });
                    const uint32_t cx = std::min(uint32_t(x) / sp.sg, sp.nbx - 1);
                    const uint32_t cy = std::min(uint32_t(y) / sp.sg, sp.nby - 1);
                    const uint32_t cz = std::min(uint32_t(z) / sp.sg, sp.nbz - 1);
                    resident.push_back(sp.starts[cz * sp.nbx * sp.nby + cy * sp.nbx + cx] != 0xFFFFFFFFu);
                    // Zero weights in BOTH evaluations: a multiple of sg on every axis
                    // (the guide sum's weights are 0), and NOT on a far face, where
                    // sampleSDFDirect clamps its base to N - 2 and lerps with t = 1.0,
                    // which PxTriLerp does not evaluate exactly (measured: 5 of the 65
                    // multiple-of-sg coarse points sit on a far face and miss by an ulp).
                    alpha0.push_back(x % int32_t(sp.sg) == 0 && y % int32_t(sp.sg) == 0 && z % int32_t(sp.sg) == 0 &&
                                     x < int32_t(sp.dx) && y < int32_t(sp.dy) && z < int32_t(sp.dz));
                }
        const uint32_t N = uint32_t(coords.size());

        // The transcription's anchor to the engine (pxgSdfSceneRef.h): bit-equal on
        // every resident point and every zero-weight coarse point, BEFORE the launch.
        size_t anchorN = 0, anchorBad = 0, alpha0Coarse = 0;
        for (uint32_t i = 0; i < N; ++i)
        {
            if (!resident[i] && !alpha0[i]) continue;
            ++anchorN;
            if (!resident[i]) ++alpha0Coarse;
            if (f2u(ref[i]) != f2u(want[i]))
            {
                if (anchorBad < 5)
                    std::printf("    anchor (%g,%g,%g) %s: sdfSampleRef %.9g decodeSparse %.9g\n", coords[i].x,
                                coords[i].y, coords[i].z, resident[i] ? "resident" : "coarse", ref[i], want[i]);
                ++anchorBad;
            }
        }
        PXG_CHECK_MSG(anchorBad == 0 && alpha0Coarse > 0,
                      "sdfSampleRef differs from the engine's decodeSparse on %zu of %zu anchor points "
                      "(%zu of them zero-weight coarse)", anchorBad, anchorN, alpha0Coarse);

        std::vector<float> got;
        if (!runAt(aBlobSp, coords, got)) return;

        // COARSE keeps sparseOracle as a secondary BOUND: the engine's nested
        // PxTriLerp at quarter weights against the guide's sum (K2 of the module).
        const double coarseBound = 1e-5 * 3.0;
        size_t badR = 0, badRef = 0, badC = 0, nR = 0, nC = 0, nCexact = 0;
        double worstC = 0.0;
        for (uint32_t i = 0; i < N; ++i)
        {
            if (f2u(got[i]) != f2u(ref[i]))
            {
                if (badRef < 5)
                    std::printf("    sparse (%g,%g,%g): got %.9g (0x%08x) want sdfSampleRef %.9g (0x%08x)\n",
                                coords[i].x, coords[i].y, coords[i].z, got[i], f2u(got[i]), ref[i], f2u(ref[i]));
                ++badRef;
            }
            if (resident[i])
            {
                ++nR;
                if (f2u(got[i]) != f2u(want[i]))
                {
                    if (badR < 5)
                        std::printf("    resident (%g,%g,%g): got %.9g want decodeSparse %.9g\n",
                                    coords[i].x, coords[i].y, coords[i].z, got[i], want[i]);
                    ++badR;
                }
            }
            else
            {
                ++nC;
                const double d = std::fabs(double(got[i]) - double(want[i]));
                if (d == 0.0) ++nCexact;
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
        PXG_CHECK_MSG(badRef == 0, "%zu of %u sparse samples differ from sdfSampleRef", badRef, N);
        PXG_CHECK_MSG(badR == 0, "%zu of %zu RESIDENT samples are not the engine's decodeSparse bit for bit",
                      badR, nR);
        PXG_CHECK_MSG(badC == 0, "%zu of %zu COARSE-FALL-BACK samples exceed the PxTriLerp bound "
                      "(worst %.3g, bound %.3g)", badC, nC, worstC, coarseBound);
        std::printf("    [ note ] sparse grid: %u bitwise vs sdfSampleRef; %zu resident bitwise vs decodeSparse; "
                    "%zu coarse (%zu bit-equal to PxTriLerp, worst %.3g)\n", N, nR, nC, nCexact, worstC);
    }

    // ---- the sparse mesh BETWEEN texels, against sdfSampleRef ----
    {
        std::mt19937 rng(0x5D96Fu);
        auto uf = [&rng](float lo, float hi) { return lo + (hi - lo) * float(rng() % 1000000u) / 1000000.0f; };
        const float D[3] = { float(sp.dx), float(sp.dy), float(sp.dz) };
        std::vector<F4> coords;
        auto put = [&](float x, float y, float z) { coords.push_back(F4{ x, y, z, 0.f }); };
        for (int i = 0; i < 600; ++i)                               // random
            put(uf(0.0f, D[0]), uf(0.0f, D[1]), uf(0.0f, D[2]));
        for (int ax = 0; ax < 3; ++ax)                              // block borders
        {
            const uint32_t nb = ax == 0 ? sp.nbx : ax == 1 ? sp.nby : sp.nbz;
            for (uint32_t m = 1; m < nb; ++m)
                for (float off : { -0.3f, -1.0f / 256.0f, 0.0f, 1.0f / 256.0f, 0.3f })
                {
                    float p[3] = { uf(0.0f, D[0]), uf(0.0f, D[1]), uf(0.0f, D[2]) };
                    p[ax] = float(m * sp.sg) + off;
                    put(p[0], p[1], p[2]);
                }
        }
        // Subgrid faces: f = base*sg + {0, sg, sg + 1} on one axis of every resident
        // cell, the other axes fractional.  What Sample() makes of them (it picks the
        // cell from f, not from `base`): +0 is local 0 of this cell; +sg and +sg+1 are
        // local 0 and 1 of the NEXT cell (often non-resident, so the coarse path), except
        // on the grid's far face, where the cell index is clamped and min(., D) lands
        // both on local sg.  Local sg + 1, the bound of Sample()'s subgrid clamp, needs
        // f >= dims + 1, which neither this fixture nor the engine's box clamp produces.
        for (uint32_t c = 0; c < sp.starts.size(); ++c)
        {
            if (sp.starts[c] == 0xFFFFFFFFu) continue;
            const uint32_t cx = c % sp.nbx, cy = (c / sp.nbx) % sp.nby, cz = c / (sp.nbx * sp.nby);
            for (int ax = 0; ax < 3; ++ax)
                for (uint32_t loc : { 0u, sp.sg, sp.sg + 1u })
                {
                    float p[3] = { float(cx * sp.sg) + uf(0.1f, float(sp.sg) - 0.1f),
                                   float(cy * sp.sg) + uf(0.1f, float(sp.sg) - 0.1f),
                                   float(cz * sp.sg) + uf(0.1f, float(sp.sg) - 0.1f) };
                    const uint32_t base = ax == 0 ? cx : ax == 1 ? cy : cz;
                    p[ax] = std::min(float(base * sp.sg + loc), D[ax]);
                    put(p[0], p[1], p[2]);
                }
        }
        std::vector<float> got;
        if (!runAt(aBlobSp, coords, got)) return;
        size_t bad = 0, nRes = 0;
        for (uint32_t i = 0; i < coords.size(); ++i)
        {
            const PxVec3 f(coords[i].x, coords[i].y, coords[i].z);
            allRef.push_back({ &sp, &spVols, f });
            if (sdfRefResident(sp, f)) ++nRes;
            const float want = sdfSampleRef(sp, spVols, f);
            if (f2u(got[i]) != f2u(want))
            {
                if (bad < 5)
                    std::printf("    sparse fractional (%.9g,%.9g,%.9g): got %.9g want %.9g\n", f.x, f.y, f.z, got[i], want);
                ++bad;
            }
        }
        PXG_CHECK_MSG(nRes > 0 && nRes < coords.size(), "the fractional pass must reach both branches (%zu of %zu resident)",
                      nRes, coords.size());
        PXG_CHECK_MSG(bad == 0, "%zu of %zu fractional sparse samples differ from sdfSampleRef", bad, coords.size());
        std::printf("    [ note ] sparse fractional: %zu bitwise (%zu resident)\n", coords.size(), nRes);
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

        // The dense mesh between texels: random f over the engine's domain [0, N-1]
        // (the box shrunk by the 0.5 margin) and exact weight TIES f = k + (j + 0.5)/256
        // on each axis, j in {0, 1, 2, 3, 128, 129} -- both parities, at both ends of
        // the weight range -- so ties-up, ties-down and ties-to-even give three answers.
        static SdfRefVols dnVols;
        dnVols = sdfRefVols(dn);
        std::mt19937 rng(0xDE96Fu);
        auto uf = [&rng](float lo, float hi) { return lo + (hi - lo) * float(rng() % 1000000u) / 1000000.0f; };
        const float D[3] = { float(dn.dx), float(dn.dy), float(dn.dz) };
        std::vector<F4> fc;
        for (int i = 0; i < 600; ++i)
            fc.push_back(F4{ uf(0.0f, D[0] - 1.0f), uf(0.0f, D[1] - 1.0f), uf(0.0f, D[2] - 1.0f), 0.f });
        for (int ax = 0; ax < 3; ++ax)
            for (uint32_t k = 0; k + 1 < uint32_t(D[ax]); ++k)
                for (uint32_t j : { 0u, 1u, 2u, 3u, 128u, 129u })
                {
                    float p[3] = { uf(0.0f, D[0] - 1.0f), uf(0.0f, D[1] - 1.0f), uf(0.0f, D[2] - 1.0f) };
                    p[ax] = float(k) + (float(j) + 0.5f) / 256.0f;
                    fc.push_back(F4{ p[0], p[1], p[2], 0.f });
                }
        const uint32_t NF = uint32_t(fc.size());
        const PxgBufferHandle bFC = createBufferFrom(dev, fc);
        const PxgBufferHandle bFO = createBufferFrom(dev, std::vector<float>(NF, -777.f));
        owned.push_back(bFC); owned.push_back(bFO);
        P_Sdf fprm = {}; fprm.meshPtr = aBlobDn; fprm.coords = dev->getBufferDeviceAddress(bFC);
        fprm.outValue = dev->getBufferDeviceAddress(bFO); fprm.numSamples = NF;
        {
            Kernel k(Kernel::Bindless{}, "npSdfTexture.spv", "sdfTexSampleAt", {}, sizeof(P_Sdf));
            if (!k.isValid()) return;
            k.dispatch((NF + 63) / 64, {}, fprm);
        }
        dev->flush();
        const auto fgot = downloadBuffer<float>(dev, bFO, NF);
        size_t fbad = 0;
        for (uint32_t i = 0; i < NF; ++i)
        {
            const PxVec3 f(fc[i].x, fc[i].y, fc[i].z);
            allRef.push_back({ &dn, &dnVols, f });
            const float want = sdfSampleRef(dn, dnVols, f);
            if (f2u(fgot[i]) != f2u(want))
            {
                if (fbad < 5)
                    std::printf("    dense fractional (%.9g,%.9g,%.9g): got %.9g want %.9g\n", f.x, f.y, f.z, fgot[i], want);
                ++fbad;
            }
        }
        PXG_CHECK_MSG(fbad == 0, "%zu of %u fractional dense samples differ from sdfSampleRef", fbad, NF);
        std::printf("    [ note ] dense fractional: %u bitwise (600 random, %u ties)\n", NF, NF - 600u);
    }

    // Bite asserts over everything compared bitwise above.  The engine never leaves
    // the texture's [0.5, N - 0.5] coordinate range (dense: f is clamped to the box
    // shrunk by the 0.5 margin; sparse: the coarse coordinate stays inside the
    // (nb+1) grid and the local one inside [0, sg+1] + 0.5), so the coordinate-clamp
    // variants have NOTHING to bite on here -- they are npTex3dBindless's (its clamp
    // shell) -- except cc-guide, whose [0, N-1] already differs on the far face
    // x = N - 0.5 (pxgSlangTex3d K4).  The dimx index clamp needs a volume with w < h,
    // and every volume here has w >= h.  The missing x index clamp changes only the
    // sign of a zero result over a -0 far-face texel, which no volume here holds
    // (npTex3dBindless's signed-zero set has one).  The P9.96 sweep lists the matching
    // mutants as this fixture's expected survivors; every other non-equivalent
    // variant must change an expectation.
    {
        auto outOfDomain = [](int v) {
            return v == pxgtex::V_CC_NONE || v == pxgtex::V_CC_LO_X || v == pxgtex::V_CC_HI_X ||
                   v == pxgtex::V_CC_HI_N_Y || v == pxgtex::V_CC_LO_0_Z || v == pxgtex::V_Y_DIMX ||
                   v == pxgtex::V_IDX_HI_X;
        };
        std::printf("    [ note ] bite asserts over %zu samples:\n     ", allRef.size());
        for (int var = 1; var < pxgtex::V_COUNT; ++var)
        {
            size_t differ = 0;
            for (const RefSample& r : allRef)
                if (f2u(sdfSampleRef(*r.s, *r.v, r.f, var)) != f2u(sdfSampleRef(*r.s, *r.v, r.f))) ++differ;
            std::printf(" %s %zu%s", pxgtex::variantName(var), differ, var % 6 == 0 ? "\n     " : ";");
            if (pxgtex::variantIsEquivalent(var))
                PXG_CHECK_MSG(differ == 0, "%s is declared equivalent but changes %zu expectations",
                              pxgtex::variantName(var), differ);
            else if (!outOfDomain(var))
                PXG_CHECK_MSG(differ >= 1, "%s changes no expectation of sdfTexture", pxgtex::variantName(var));
        }
        std::printf("\n");
    }

    for (PxgBufferHandle h : owned)
        dev->destroyBuffer(h);
    physx::gpuslang::pxgSetVulkanDriverShim(nullptr);
}

} // namespace
