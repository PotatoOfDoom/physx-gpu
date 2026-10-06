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

// pxgSdfSceneRef.h -- the hand-built SDF scenes, the cooked-mesh blob layout and
// the decodeSparse mirror, shared by every fixture over the SDF layer.
//
// Extracted from testSdfTexture.cpp in P9.12d, when the second fixture over the
// same layer arrived.  ONE scene, ONE blob writer, ONE oracle: a second copy of
// any of the three would be a second thing to keep in step with the engine, and
// the whole point of the mirror below is that it was measured against the
// engine ONCE (see the note on sparseOracle) rather than re-derived per fixture.
//
// THE SPARSE SCENE, every number in it distinct per the standing rule:
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

#ifndef PXG_SDF_SCENE_REF_H
#define PXG_SDF_SCENE_REF_H

#include "GuSDF.h"
#include "PxgCudaToolkitShim.h"
#include "pxgTex3dRef.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace pxgsdfref
{
using namespace physx;

struct F4 { float x, y, z, w; };

inline uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

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

inline void buildSparse(SdfScene& s)
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

inline void buildDense(SdfScene& s)
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
// there.
//
// AND IT HAS BEEN CHECKED AGAINST THE REAL FUNCTION.  A one-off program linked
// against libphysx-sdk.a ran this exact mirror against Gu::SDF::decodeSparse over
// the identical hand-built SDF: 2457 fine-grid samples, 6 resident cells and 24
// coarse-fall-back cells, 537 of them on a far edge -- ZERO mismatches, worst
// |diff| exactly 0.  So the expectation is the engine's behaviour and not this
// file's reading of it; what is transcribed has been measured against the
// original once, offline, rather than asserted.
//
// It takes the SCENE rather than a Gu::SDF because instantiating one would drag
// in that class's out-of-line destructor, and with it the whole GuSDF.cpp
// translation unit (AABB trees, winding numbers, mesh analysis) into a binary
// that deliberately links only the backend and foundation.  Nothing is lost:
// none of the four primitives above needs an instance.
inline float sparseOracle(const SdfScene& s, int32_t xx, int32_t yy, int32_t zz)
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
// sdfSampleRef (P9.96): SparseSDFTexture::Sample (pxgSlangSdfTexture.slang
// sdfTexSample, sdfCollision.cuh:113-151) on the host, at ANY fine coordinate f,
// over pxgtex::tex3D -- the dense, coarse-fall-back and resident branches, statement
// for statement and in the shader's expression order: the float-to-uint truncation
// and unsigned clamp for the cell (T7), the rescale into coarse units only on the
// fall-back (T8), the local clamp to [0, sg + 1] (T9), applySubgridStart's ADD
// (T10), and `raw * range + min` with range = max - min (T4) only on the resident
// path (T11).  The subgrid volume holds pxgtex::unorm8 of the stored bytes.
//
// WHAT ANCHORS IT TO THE ENGINE.  sparseOracle above is the engine's decodeSparse.
// At an integer f the two must agree BIT FOR BIT on every RESIDENT point (the subgrid
// coordinate is integer + 0.5, so the weights are 1 and 0, and the value is
// unorm8(v) * range + min exactly as decodeSample spells it) and on every COARSE point
// whose fine coordinate is a multiple of sg on all three axes AND below the far face
// (coarse weights 0 in both: on a far face sampleSDFDirect clamps its base to N - 2
// and lerps with t = 1.0, which PxTriLerp does not evaluate exactly -- measured, 5 of
// the 65 multiple-of-sg coarse points of buildSparse miss there by an ulp).
// At every OTHER integer coarse point they legitimately differ by roundings:
// sparseOracle's coarse branch is DenseSDF::sampleSDFDirect, the engine's nested
// PxTriLerp at quarter weights, which is not the guide's eight-term sum.  Those
// points are held to a bound against sparseOracle and bitwise only against this.
// Domain: f >= 0 on the sparse path (the shader's float-to-uint is undefined below
// 0, and every caller clamps to the box first); any finite f on the dense path.
// `variant` threads a pxgtex::Variant through for bite asserts.
// ---------------------------------------------------------------------------
struct SdfRefVols
{
    pxgtex::Vol primary;    // dense: the field; sparse: the (nbx+1)(nby+1)(nbz+1) coarse grid
    pxgtex::Vol sub;        // sparse only: the subgrid texture, after the UNORM read
};

inline SdfRefVols sdfRefVols(const SdfScene& s)
{
    SdfRefVols v;
    if (s.sg == 0u)
    {
        v.primary.w = s.dx; v.primary.h = s.dy; v.primary.d = s.dz;
    }
    else
    {
        v.primary.w = s.nbx + 1u; v.primary.h = s.nby + 1u; v.primary.d = s.nbz + 1u;
        v.sub.w = s.blkx * (s.sg + 1u); v.sub.h = s.blky * (s.sg + 1u); v.sub.d = s.blkz * (s.sg + 1u);
        v.sub.t.resize(s.sub.size());
        for (size_t i = 0; i < s.sub.size(); ++i)
            v.sub.t[i] = pxgtex::unorm8(s.sub[i]);
    }
    v.primary.t = s.coarse;
    return v;
}

inline float sdfSampleRef(const SdfScene& s, const SdfRefVols& v, PxVec3 f, int variant = pxgtex::V_GUIDE)
{
    if (s.sg == 0u)
        return pxgtex::tex3D(v.primary, f.x + 0.5f, f.y + 0.5f, f.z + 0.5f, variant);

    const float fineToCoarse = 1.0f / float(s.sg);
    const uint32_t csx = s.dx / s.sg, csy = s.dy / s.sg, csz = s.dz / s.sg;
    const uint32_t xBase = std::min(std::max(uint32_t(f.x * fineToCoarse), 0u), csx - 1u);
    const uint32_t yBase = std::min(std::max(uint32_t(f.y * fineToCoarse), 0u), csy - 1u);
    const uint32_t zBase = std::min(std::max(uint32_t(f.z * fineToCoarse), 0u), csz - 1u);
    const uint32_t startSlot = s.starts[zBase * csx * csy + yBase * csx + xBase];
    if (startSlot == 0xFFFFFFFFu)
    {
        f = PxVec3(f.x * fineToCoarse, f.y * fineToCoarse, f.z * fineToCoarse);
        return pxgtex::tex3D(v.primary, f.x + 0.5f, f.y + 0.5f, f.z + 0.5f, variant);
    }
    const float hi = float(s.sg + 1u);
    f.x = std::min(std::max(f.x - float(xBase * s.sg), 0.0f), hi);
    f.y = std::min(std::max(f.y - float(yBase * s.sg), 0.0f), hi);
    f.z = std::min(std::max(f.z - float(zBase * s.sg), 0.0f), hi);
    uint32_t triple = startSlot;
    f.x += float((triple & 1023u) * (s.sg + 1u));
    triple >>= 10;
    f.y += float((triple & 1023u) * (s.sg + 1u));
    triple >>= 10;
    f.z += float((triple & 1023u) * (s.sg + 1u));
    const float raw = pxgtex::tex3D(v.sub, f.x + 0.5f, f.y + 0.5f, f.z + 0.5f, variant);
    const float range = s.smax - s.smin;
    return raw * range + s.smin;
}

// Which branch sdfSampleRef takes at f (sparse scenes only): true == resident.
inline bool sdfRefResident(const SdfScene& s, const PxVec3& f)
{
    const float fineToCoarse = 1.0f / float(s.sg);
    const uint32_t csx = s.dx / s.sg, csy = s.dy / s.sg, csz = s.dz / s.sg;
    const uint32_t xBase = std::min(uint32_t(f.x * fineToCoarse), csx - 1u);
    const uint32_t yBase = std::min(uint32_t(f.y * fineToCoarse), csy - 1u);
    const uint32_t zBase = std::min(uint32_t(f.z * fineToCoarse), csz - 1u);
    return s.starts[zBase * csx * csy + yBase * csx + xBase] != 0xFFFFFFFFu;
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

// P9.83b-8 (4/n): the same writer with REAL vertices and triangles -- the mesh-mesh
// kernel reads the geometry half too -- and optionally NO SDF.  `s == nullptr`
// writes sdfDims = (0, 0, 0, projectionFlag) and ends the blob there, which is what
// the engine does for a mesh without one; the projection flag is written either
// way because doFirst reads it (convexMeshMidphase.cu:1358-1360).  A null
// vertBytes / triBytes keeps the junk fill, so layOutBlob below is unchanged.
inline BlobLayout layOutBlobEx(const SdfScene* s, uint32_t numVerts, uint32_t numTris,
                               uint32_t nbPackedNodes, uint32_t projectionFlag, uint32_t junkSparseHandle,
                               const void* vertBytes, const void* triBytes)
{
    BlobLayout b;
    auto put32 = [&b](uint32_t v) { const size_t o = b.bytes.size(); b.bytes.resize(o + 4); std::memcpy(&b.bytes[o], &v, 4); };
    auto putF = [&](float v) { uint32_t u; std::memcpy(&u, &v, 4); put32(u); };
    auto junk = [&b](size_t n, uint8_t seed) { const size_t o = b.bytes.size(); b.bytes.resize(o + n);
                                               for (size_t i = 0; i < n; ++i) b.bytes[o + i] = uint8_t(seed + (i * 37u)); };

    put32(numVerts); put32(numTris); put32(0xABCDEF01u); put32(nbPackedNodes);  // .z is NOT the node count
    junk(size_t(nbPackedNodes) * BV32_STRIDE, 0x11);   // BV32 tree
    auto raw = [&b](const void* src, size_t n) { const size_t o = b.bytes.size(); b.bytes.resize(o + n);
                                                 std::memcpy(&b.bytes[o], src, n); };
    if (vertBytes) raw(vertBytes, size_t(numVerts) * 16); else junk(size_t(numVerts) * 16, 0x22);   // float4 verts
    if (triBytes) raw(triBytes, size_t(numTris) * 16); else junk(size_t(numTris) * 16, 0x33);       // uint4 indices
    junk(size_t(numTris) * 16, 0x44);                  // uint4 adjacencies
    junk(size_t(numTris) * 4, 0x55);                   // uint32 faceRemap
    junk(size_t(numVerts + 1) * 4, 0x66);              // accumulatedTrianglesRef -- numVerts+1
    junk(size_t(numTris) * 12, 0x77);                  // trianglesReferences[3*numTris]
    while (b.bytes.size() % 16 != 0)                   // the one alignment step
        b.bytes.push_back(0x99);

    b.sdfDimsOffset = b.bytes.size();
    if (!s)
    {
        put32(0u); put32(0u); put32(0u); put32(projectionFlag);
        return b;
    }
    put32(s->dx); put32(s->dy); put32(s->dz); put32(projectionFlag);
    putF(s->meshLower.x); putF(s->meshLower.y); putF(s->meshLower.z); putF(s->spacing);
    put32(s->sg); put32(f2u(s->smin)); put32(f2u(s->smax)); put32(uint32_t(s->starts.size()));

    b.texObjOffset = b.bytes.size();
    put32(0u); put32(0u);                              // mTexObject, filled after upload
    b.texObjSparseOffset = b.bytes.size();
    // On the DENSE path the engine never writes this word -- it stays whatever the
    // blob held.  The fixture puts a recognisable value there and requires the
    // port to read exactly it, which proves the field is read at the right offset
    // without pretending the engine zeroes it.
    put32(junkSparseHandle); put32(0u);
    for (uint32_t v : s->starts)
        put32(v);
    return b;
}

inline BlobLayout layOutBlob(const SdfScene& s, uint32_t numVerts, uint32_t numTris,
                             uint32_t nbPackedNodes, uint32_t projectionFlag, uint32_t junkSparseHandle)
{
    return layOutBlobEx(&s, numVerts, numTris, nbPackedNodes, projectionFlag, junkSparseHandle, nullptr, nullptr);
}

// Uploads a volume through the ENGINE's own call sequence, so every fixture over
// this layer exercises the P9.12b driver-shim path as a side effect rather than
// reaching past it into the Vulkan backend.  Returns 0 on failure, and the shim
// must already be installed via pxgSetVulkanDriverShim.
inline uint64_t sdfUploadTex(CUarray_format fmt, uint32_t w, uint32_t h, uint32_t d,
                             const void* data, size_t texelBytes)
{
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
}

// SparseSDFTexture::initialize (sdfCollision.cuh:66-91) on the host, for the
// fixtures that need the box and the texel transform without a dispatch.  Every
// field of it is compared bit-for-bit against the GPU's own by the P9.12c probe,
// so this is a mirror of something already measured, not a second source.
struct HostSdfBox
{
    PxVec3 lower, higher;
    float dx, invDx;
    uint32_t dims[3];
};

inline HostSdfBox hostSdfBox(const SdfScene& s)
{
    HostSdfBox t;
    const float margin = (s.sg == 0u) ? 0.5f : 0.0f;
    t.lower = s.meshLower + PxVec3(s.spacing) * margin;
    t.higher = s.meshLower + PxVec3((s.dx - margin) * s.spacing,
                                    (s.dy - margin) * s.spacing,
                                    (s.dz - margin) * s.spacing);
    t.invDx = 1.0f / s.spacing;
    t.dx = s.spacing;
    t.dims[0] = s.dx; t.dims[1] = s.dy; t.dims[2] = s.dz;
    return t;
}

} // namespace pxgsdfref

#endif // PXG_SDF_SCENE_REF_H
