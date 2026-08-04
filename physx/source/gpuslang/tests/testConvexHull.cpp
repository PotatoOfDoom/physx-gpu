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

// P8.39 — the cooked convex-hull blob reader (pxgSlangConvexHull).
//
// THE ORACLE IS THE CONSTRUCTION, NOT A FORMULA.  The blob is built by APPENDING
// its sections in order and recording where each one landed; the expected offsets
// are those recorded positions.  Restating the shader's closed-form arithmetic in
// the test would be circular — the same slip in both would cancel — whereas an
// append-and-record builder derives the layout from the section sizes alone.
//
// Every section is also filled with values that encode (which section, which
// index), so a wrong address is caught twice over: once by the reported offset and
// once by the value read back through it.
//
// Suite (ctest filter "convexHull"):
//   convexHullLayout — seven hull shapes, including counts that leave every run
//                      after the planes unaligned, and the 8-bit field maxima.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;

struct HullProbePush
{
    uint64_t hulls, output;
    uint32_t numHulls, polyIndex, edgeIndex, vertIndex;
};

const uint32_t OUT_WORDS = 128 / 4;

// cudaNpCommon.h:137-172 over cutil_math.h:765-788 — the packings the shader
// decodes.  Spelled out here as construction, so the test builds what it expects.
uint32_t packHullDesc(uint32_t nbEdges, uint32_t nbVerts, uint32_t nbPolygons)
{
    return (nbEdges << 16) | (nbVerts << 8) | nbPolygons;
}
uint32_t packPolyDesc(uint32_t vRef8, uint32_t nbVerts, uint32_t minIndex)
{
    return (vRef8 << 16) | (nbVerts << 8) | minIndex;
}

struct Blob
{
    std::vector<uint8_t> bytes;
    uint32_t offVerts = 0, offPlanes = 0, offPolyDescs = 0;
    uint32_t offVertsByEdges = 0, offFacesByEdges = 0, offFacesByVerts = 0, offVertexData = 0;
    uint32_t nbVerts = 0, nbPolygons = 0, nbEdges = 0;

    void putU8(uint8_t v) { bytes.push_back(v); }
    void putU16(uint16_t v) { bytes.push_back(uint8_t(v)); bytes.push_back(uint8_t(v >> 8)); }
    void putU32(uint32_t v) { for (int i = 0; i < 4; ++i) bytes.push_back(uint8_t(v >> (8 * i))); }
    void putF32(float v) { uint32_t u; std::memcpy(&u, &v, 4); putU32(u); }
};

// A byte whose high nibble names the section and whose low nibble tracks the
// index, so both a cross-section slip and a few-bytes slip are visible.
uint8_t tag(uint32_t section, uint32_t index) { return uint8_t((section << 4) | (index & 0x0fu)); }

Blob buildHull(uint32_t nbVerts, uint32_t nbPolygons, uint32_t nbEdges)
{
    Blob b;
    b.nbVerts = nbVerts;
    b.nbPolygons = nbPolygons;
    b.nbEdges = nbEdges;

    // @0 float4 centre of mass
    b.putF32(-1.5f); b.putF32(-2.5f); b.putF32(-3.5f); b.putF32(0.0f);
    // @16 uint4, .x = the packed hull descriptor
    b.putU32(packHullDesc(nbEdges, nbVerts, nbPolygons));
    b.putU32(0xAAAA0001u); b.putU32(0xAAAA0002u); b.putU32(0xAAAA0003u);
    // @32 float4 the chain does not read
    b.putF32(9.5f); b.putF32(9.25f); b.putF32(9.125f); b.putF32(9.0f);

    b.offVerts = uint32_t(b.bytes.size());
    for (uint32_t i = 0; i < nbVerts; ++i)
    { b.putF32(1000.0f + float(i)); b.putF32(1000.5f + float(i)); b.putF32(1000.25f + float(i)); b.putF32(0.0f); }

    b.offPlanes = uint32_t(b.bytes.size());
    for (uint32_t i = 0; i < nbPolygons; ++i)
    { b.putF32(2000.0f + float(i)); b.putF32(2000.5f + float(i)); b.putF32(2000.25f + float(i)); b.putF32(-float(i)); }

    b.offPolyDescs = uint32_t(b.bytes.size());
    for (uint32_t i = 0; i < nbPolygons; ++i)
        b.putU32(packPolyDesc(3u * i, 4u + (i & 3u), i & 0xffu));

    b.offVertsByEdges = uint32_t(b.bytes.size());
    for (uint32_t k = 0; k < 2u * nbEdges; ++k)
        b.putU16(uint16_t(0x4000u | (k & 0x0fffu)));

    b.offFacesByEdges = uint32_t(b.bytes.size());
    for (uint32_t k = 0; k < 2u * nbEdges; ++k)
        b.putU8(tag(5u, k));

    b.offFacesByVerts = uint32_t(b.bytes.size());
    for (uint32_t k = 0; k < 3u * nbVerts; ++k)
        b.putU8(tag(6u, k));

    b.offVertexData = uint32_t(b.bytes.size());
    for (uint32_t k = 0; k < 3u * nbPolygons + 16u; ++k)
        b.putU8(tag(7u, k));

    // The shader assembles the byte and ushort runs out of whole uints, so it can
    // read up to three bytes past the last element of a run.  Pad so that read is
    // inside the allocation rather than inside the next buffer.
    for (int i = 0; i < 16; ++i) b.putU8(0xFFu);
    return b;
}

PXG_TEST(convexHullLayout)
{
    PxgSlangComputeDevice* dev = testDevice();

    struct Shape { uint32_t v, p, e; const char* name; };
    const Shape shapes[] = {
        { 4, 4, 6, "tetrahedron" },
        { 8, 6, 12, "box" },
        { 3, 3, 3, "degenerate triple" },
        { 7, 5, 11, "odd counts (every run unaligned)" },
        { 5, 2, 2, "two polygons, two edges" },
        { 255, 255, 1000, "the 8-bit field maxima" },
        { 6, 2, 65535, "the 16-bit edge maximum" },
    };
    const uint32_t n = uint32_t(sizeof(shapes) / sizeof(shapes[0]));

    const uint32_t polyIndex = 1u, edgeIndex = 1u, vertIndex = 2u;

    std::vector<Blob> blobs;
    std::vector<PxgBufferHandle> hullBufs;
    std::vector<uint64_t> hullAddrs;
    for (const Shape& s : shapes)
    {
        blobs.push_back(buildHull(s.v, s.p, s.e));
        PxgBufferHandle hb = createBufferFrom(dev, blobs.back().bytes);
        hullBufs.push_back(hb);
        hullAddrs.push_back(dev->getBufferDeviceAddress(hb));
    }

    PxgBufferHandle bAddrs = createBufferFrom(dev, hullAddrs);
    std::vector<uint32_t> outInit(size_t(n) * OUT_WORDS, 0xDEADBEEFu);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    HullProbePush push{};
    push.hulls = dev->getBufferDeviceAddress(bAddrs);
    push.output = dev->getBufferDeviceAddress(bOut);
    push.numHulls = n;
    push.polyIndex = polyIndex;
    push.edgeIndex = edgeIndex;
    push.vertIndex = vertIndex;

    {
        Kernel k("convexHullProbeBda.spv", "convexHullProbeLaunch", {}, sizeof(HullProbePush));
        k.dispatch(1u, {}, push, 1u, 1u);
        dev->flush();
    }

    const auto got = downloadBuffer<uint32_t>(dev, bOut, outInit.size());

    size_t bad = 0;
    auto chk = [&](const char* what, const char* shape, uint32_t g, uint32_t e) {
        if (g != e)
        {
            ++bad;
            if (bad <= 12) std::printf("  [%s] %s: got=%u (0x%x) expected=%u (0x%x)\n",
                                       shape, what, g, g, e, e);
        }
    };

    for (uint32_t i = 0; i < n; ++i)
    {
        const Blob& b = blobs[i];
        const uint32_t* r = got.data() + size_t(i) * OUT_WORDS;
        const char* nm = shapes[i].name;

        chk("offset vertices", nm, r[0], b.offVerts);
        chk("offset planes", nm, r[1], b.offPlanes);
        chk("offset polyDescs", nm, r[2], b.offPolyDescs);
        chk("offset verticesByEdges16", nm, r[3], b.offVertsByEdges);
        chk("offset facesByEdges8", nm, r[4], b.offFacesByEdges);
        chk("offset facesByVertices8", nm, r[5], b.offFacesByVerts);
        chk("offset vertexData8", nm, r[6], b.offVertexData);

        chk("nbEdges", nm, r[7], b.nbEdges);
        chk("nbVerts", nm, r[8], b.nbVerts);
        chk("nbPolygons", nm, r[9], b.nbPolygons);

        const uint32_t expPd = packPolyDesc(3u * polyIndex, 4u + (polyIndex & 3u), polyIndex & 0xffu);
        chk("polyDesc", nm, r[10], expPd);
        chk("polyDesc vRef8", nm, r[11], 3u * polyIndex);
        chk("polyDesc nbVerts", nm, r[12], 4u + (polyIndex & 3u));
        chk("polyDesc minIndex", nm, r[13], polyIndex & 0xffu);

        chk("edge vertex 0", nm, r[14], 0x4000u | ((2u * edgeIndex + 0u) & 0x0fffu));
        chk("edge vertex 1", nm, r[15], 0x4000u | ((2u * edgeIndex + 1u) & 0x0fffu));
        chk("edge face 0", nm, r[16], tag(5u, 2u * edgeIndex + 0u));
        chk("edge face 1", nm, r[17], tag(5u, 2u * edgeIndex + 1u));

        chk("vertex face 0", nm, r[18], tag(6u, 3u * vertIndex + 0u));
        chk("vertex face 1", nm, r[19], tag(6u, 3u * vertIndex + 1u));
        chk("vertex face 2", nm, r[20], tag(6u, 3u * vertIndex + 2u));
        chk("polygon vertex 0", nm, r[21], tag(7u, 3u * polyIndex));

        float com[4], vert[4];
        std::memcpy(com, r + 24, 16);
        std::memcpy(vert, r + 28, 16);
        if (com[0] != -1.5f || com[1] != -2.5f || com[2] != -3.5f)
        { ++bad; std::printf("  [%s] centre of mass: (%g %g %g)\n", nm, com[0], com[1], com[2]); }
        if (vert[0] != 1000.0f + float(vertIndex) || vert[1] != 1000.5f + float(vertIndex))
        { ++bad; std::printf("  [%s] vertex[%u]: (%g %g)\n", nm, vertIndex, vert[0], vert[1]); }
    }

    for (PxgBufferHandle h : hullBufs) dev->destroyBuffer(h);
    dev->destroyBuffer(bAddrs);
    dev->destroyBuffer(bOut);

    PXG_CHECK_MSG(bad == 0, "the hull blob reader matches the built layout (%zu mismatches)", bad);
}

} // namespace
