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

// P5 Step 5 tranches 8+11: the ASSEMBLED collide() (GJK + EPA + all three
// type0 contact-gen arms, npCvxCollide.slang) inside
// convexConvexNphase_stage2Kernel, driven through PxgCudaShim on the ENGINE
// ABI (56-byte Stage2Params blob, same launch pattern as
// testNpCvxStage2Setup.cpp) against REAL cooked box convex-hull blobs
// (fixture shared with testConvexPlaneNphase.cpp).
//
// VERIFICATION LEVEL: geometric integration.  The sub-pieces (gjkWarm /
// epaCore / face / EE / localNormal / addManifold / refFrame / polyClip /
// reduce / clipEmit) are bit-verified by their standalone tranches; here
// the ASSEMBLY is checked against CPU expectations derived from
// cudaGJKEPA.cu for axis-aligned capsule/sphere-on-box configurations:
//   1. capsule lying on the box top, surface touching -> 2 face contacts,
//      manifold normal (0,+1,0) in B-local (np.xyz = -gjkOutput.direction;
//      the GJK direction is (0,-1,0), i.e. B->closest-feature-of-A points up
//      so -direction points from B to A), np.w = tEnter = segment-to-surface
//      distance = radius (+ separation);
//   2. deep overlap (segment inside the hull) -> EPA path; per the .cu the
//      face-ray tEnter clamps at 0 (intersectRayPolyhedron starts tFirst=0),
//      so np.w == 0 and localContactB == the segment endpoints;
//   3. separated but within contactDistance -> GJK-close face contacts with
//      np.w = gap;
//   4. distant -> collide() returns false, word 12 = refreshedNb|0x80000000,
//      contact slots untouched;
//   5. capsule overhanging a narrow box -> face rays miss (plane
//      separation), EE fallback emits the two edge contacts (negated
//      normalPen);
//   6. sphere type0 -> exactly 1 contact, contactA = 0 (sphere centre),
//      normal = -gjkOutput.direction;
//   7. mutation canary: flipping the hull's top-face plane must break the
//      touch expectations (proves the kernel reads the blob planes);
// and (tranche 11) for convex-hull-vs-box pairs through the
// generateConvexContacts arm (cu L823-1022) — shape0 = a cooked cube hull
// with type eCONVEXMESH (5, PxGeometry.h L55), shape1 = the big box hull
// with type eBOX (3); type1 < type0 flips the pair (stage2Body), so
// post-flip hull A = the box at the origin, hull B = the cube at pos0,
// aToBp = -pos0, and the manifold normal is -Y in B-local:
//   8. near/touching face-face (gap +0.05 < contactDist, GJK eCLOSE) and
//      barely/deep overlap (EPA) -> polyClip clips B's bottom face against
//      A's top face; B's 4 verts land inside A's face -> the first
//      addContacts emits exactly 4 (cu L153), no reduce; contacts at the
//      cube's bottom corners, np = (0,-1,0, gap);
//   9. distant -> eDISTANT invalidate writeback, slots untouched;
//  10. cube rotated 45 deg about Y over an equal box -> both faces clip to
//      8 edge-edge crossings (cu L197-211) -> reduce<false,false,4>
//      (cu L971/L1005) down to 4 octagon vertices;
//  11. preloaded 4-contact manifold + aligned localNor -> fullContactGen
//      false (cu L887): witness REPLACE (within replaceBreakingThreshold,
//      cu L574-586) and full+matchless 5-candidate reduce-drop (cu L588-601)
//      leave the preloaded manifold intact — no clip regen;
//  12. mutation canary on hull A's top plane.
// Also asserted every run: quatB words 9-11 rewritten byte-identically,
// relativePos/quatA untouched, and warm-start words 13-15 READ but NEVER
// stored back (sentinel patterns survive).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

// ---------------------------------------------------------------------------
// Host mirrors of the engine element structs (byte-exact vs the C++ headers).
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct HInput { uint32_t shapeRef0, shapeRef1, transformCacheRef0, transformCacheRef1; };  // 16
struct HCachedTransform { float qx, qy, qz, qw, px, py, pz; uint32_t flags; };              // 32
struct HShape { float sx, sy, sz, srx, sry, srz, srw; uint32_t materialIndex;
                uint64_t hullPtr; uint32_t type, particleId; };                             // 48
struct HManifold                                                                            // 256
{
    float lNp[4][4]; float lA[4][4]; float lB[4][4];       // 0 / 64 / 128
    float relPos[4]; float quatA[4]; float quatB[4];       // 192 / 208 / 224
    uint32_t nbContacts, nbWarmStart, warmA, warmB;        // 240..252
};
#pragma pack(pop)

static_assert(sizeof(HInput) == 16, "HInput");
static_assert(sizeof(HCachedTransform) == 32, "HCachedTransform");
static_assert(sizeof(HShape) == 48, "HShape");
static_assert(sizeof(HManifold) == 256, "HManifold");

struct V3 { float x, y, z; };

// ---------------------------------------------------------------------------
// The cooked box convex-hull blob (fixture shared with
// testConvexPlaneNphase.cpp): 8 verts, 6 faces, 12 edges at the GRB byte
// offsets relative to convexPtrB = hullPtr + 16.  Mesh scale identity so
// vertex2Shape(v) == v.
// ---------------------------------------------------------------------------
// Face windings are COUNTER-clockwise seen from outside (Newell normal ==
// the outward plane normal), matching real cooked hulls (the cooking
// unify-normals pass, GuCookingConvexPolygonsBuilder.cpp L1109-1160, flips
// polygon windings until computeNewellPlane agrees with the outward
// triangle normals).  polyClip's edge planes (e0planeN = e0 x axis,
// cudaGJKEPA.cu L133) assume this orientation, so the winding is
// load-bearing for the tranche-11 convex-hull fixtures.  (Tranche 8's
// windings were reversed; only the winding-insensitive EE path consumed
// them.)
static const int kBoxFaceVtx[6][4] = {
    { 6, 7, 5, 4 }, // +x
    { 1, 3, 2, 0 }, // -x
    { 3, 7, 6, 2 }, // +y
    { 4, 5, 1, 0 }, // -y
    { 5, 7, 3, 1 }, // +z
    { 2, 6, 4, 0 }, // -z
};
static const float kBoxFaceN[6][3] = {
    { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 },
};
static const int kBoxEdgeFaces[12][2] = {
    { 0, 2 }, { 0, 3 }, { 0, 4 }, { 0, 5 }, { 1, 2 }, { 1, 3 },
    { 1, 4 }, { 1, 5 }, { 2, 4 }, { 2, 5 }, { 3, 4 }, { 3, 5 },
};

static V3 boxCorner(int i, V3 h)
{
    return V3{ (i & 4) ? h.x : -h.x, (i & 2) ? h.y : -h.y, (i & 1) ? h.z : -h.z };
}

static std::vector<uint8_t> buildBoxHull(V3 h)
{
    std::vector<uint8_t> blob(512, 0);
    auto setU32 = [&](size_t off, uint32_t v) { std::memcpy(&blob[off], &v, 4); };
    auto setF = [&](size_t off, float v) { std::memcpy(&blob[off], &v, 4); };

    const uint32_t V = 8, P = 6, E = 12;
    const size_t convexPtrB = 16;               // hullPtr + sizeof(float4)
    // header uint4 (only .x used): nbPolygons | nbVerts<<8 | nbEdges<<16
    setU32(convexPtrB + 0, P | (V << 8) | (E << 16));
    // extents float4 (.xyz internal AABB, .w inSphereRadius)
    setF(convexPtrB + 16, h.x); setF(convexPtrB + 20, h.y); setF(convexPtrB + 24, h.z);
    setF(convexPtrB + 28, std::fmin(std::fmin(h.x, h.y), h.z));

    const size_t vertsBase = convexPtrB + 32;
    for (uint32_t i = 0; i < V; ++i)
    {
        const V3 c = boxCorner(int(i), h);
        setF(vertsBase + 16 * i + 0, c.x); setF(vertsBase + 16 * i + 4, c.y); setF(vertsBase + 16 * i + 8, c.z);
    }
    const size_t planesBase = vertsBase + 16 * V;
    for (uint32_t f = 0; f < P; ++f)
    {
        setF(planesBase + 16 * f + 0, kBoxFaceN[f][0]);
        setF(planesBase + 16 * f + 4, kBoxFaceN[f][1]);
        setF(planesBase + 16 * f + 8, kBoxFaceN[f][2]);
        const V3 fv = boxCorner(kBoxFaceVtx[f][0], h);
        setF(planesBase + 16 * f + 12, -(kBoxFaceN[f][0] * fv.x + kBoxFaceN[f][1] * fv.y + kBoxFaceN[f][2] * fv.z));
    }
    const size_t polyDescBase = planesBase + 16 * P;
    for (uint32_t f = 0; f < P; ++f)
        setU32(polyDescBase + 4 * f, (4u * f) << 16 | (4u << 8) | 0u); // vRef8<<16 | nbVerts<<8 | minIndex

    // verticesByEdges16 [polyDescBase + 4P, +4E) unused by collide() (left 0).
    const size_t fbeBase = polyDescBase + 4 * P + 4 * E;
    for (uint32_t e = 0; e < E; ++e)
    {
        blob[fbeBase + 2 * e + 0] = uint8_t(kBoxEdgeFaces[e][0]);
        blob[fbeBase + 2 * e + 1] = uint8_t(kBoxEdgeFaces[e][1]);
    }
    // facesByVertices8 [fbeBase + 2E, +3V) unused (left 0).
    const size_t vtxDataBase = fbeBase + 2 * E + 3 * V;
    for (uint32_t f = 0; f < P; ++f)
        for (uint32_t k = 0; k < 4; ++k)
            blob[vtxDataBase + 4 * f + k] = uint8_t(kBoxFaceVtx[f][k]);
    return blob;
}

// The 56-byte engine parameter blob.
struct Stage2Blob
{
    uint32_t numTests;
    float toleranceLength;
    uint64_t tempKeepPairsArray, cmInputs, cmManifolds, shapes, transformCache, contactDistance;
};
static_assert(sizeof(Stage2Blob) == 56, "stage2 engine blob is 56 bytes");

struct ExpContact { V3 a, b, n; float pen; };

// One shape0-on-box scene.  Shape 1 is always the cooked box hull (type
// eBOX = 3) at the origin; shape 0 is a sphere/capsule (tranche 8) or a
// second cooked hull with type eCONVEXMESH = 5 (tranche 11; the flip in
// stage2Body then makes the BOX hull A and the cube hull B).
struct Scene
{
    uint32_t type0;      // 0 = sphere, 2 = capsule, 5 = convex-mesh hull
    V3 scale0;           // capsule: (halfHeight, radius, radius); sphere: (r, r, r); hull: mesh scale
    V3 pos0;             // shape 0 world position (box is at the origin)
    V3 boxHalf;          // shape1 hull half-extents
    float cdPerShape;    // per-shape contact distance (cd0 == cd1)
    uint32_t countsWord; // stage-1 counts word preloaded in the manifold
    bool flipTopPlane;   // mutation canary: negate the +y plane of shape1's hull
    // tranche-11 extensions (defaulted so tranche-8 braced inits stand)
    V3 h0 = {};                              // type0 == 5: shape0 hull half-extents
    float q0x = 0, q0y = 0, q0z = 0, q0w = 1; // shape0 world orientation
    uint32_t preloadCount = 0;               // manifold slots preloaded before launch
    ExpContact pre[4] = {};                  // (countsWord low byte must match)
    // P9.90: shape1 blob override (a generally rotated box hull); boxHalf unused then
    const std::vector<uint8_t>* hull1 = nullptr;
};

struct RunResult
{
    HManifold mf;    // manifold after the launch
    size_t tailBad;  // moreData words 0-11 + 13-15 preservation violations
};

static std::vector<uint8_t> ccLoader(const char* file) { return loadShaderBlob(file); }

static bool runScene(const Scene& sc, RunResult& out)
{
    PxgSlangComputeDevice* dev = testDevice();
    static const PxgKernelDef manifest[] = {
        { "npCvxStage2Setup.spv", "convexConvexNphase_stage2Kernel", 56u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &ccLoader);

    std::vector<uint8_t> blob = sc.hull1 ? *sc.hull1 : buildBoxHull(sc.boxHalf);
    if (sc.flipTopPlane)
    {
        // negate the +y plane (polygon 2): n = (0,-1,0), d = -d
        const size_t planesBase = 16 + 32 + 16 * 8 + 16 * 2;
        float n[4]; std::memcpy(n, &blob[planesBase], 16);
        n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2]; n[3] = -n[3];
        std::memcpy(&blob[planesBase], n, 16);
    }
    const bool cvx0 = (sc.type0 == 5u); // shape0 is a second cooked hull
    std::vector<uint8_t> blob0;
    if (cvx0)
        blob0 = buildBoxHull(sc.h0);

    HInput in = { 0, 1, 0, 1 };
    std::vector<HShape> shapes(2);
    shapes[0] = { sc.scale0.x, sc.scale0.y, sc.scale0.z, 0, 0, 0, 1, 3u, 0u, sc.type0, 0u };
    shapes[1] = { 1, 1, 1, 0, 0, 0, 1, 5u, 0u, 3u /*eBOX*/, 0u }; // hullPtr filled below
    std::vector<HCachedTransform> xf(2);
    xf[0] = { sc.q0x, sc.q0y, sc.q0z, sc.q0w, sc.pos0.x, sc.pos0.y, sc.pos0.z, 0u };
    xf[1] = { 0, 0, 0, 1, 0, 0, 0, 0u };
    std::vector<float> cd = { sc.cdPerShape, sc.cdPerShape };
    uint32_t keep = 1u;

    HManifold mf;
    std::memset(&mf, 0xCD, sizeof(mf)); // contact slots: 0xCD sentinel
    mf.relPos[0] = 1; mf.relPos[1] = 2; mf.relPos[2] = 3; mf.relPos[3] = 4;
    mf.quatA[0] = 5; mf.quatA[1] = 6; mf.quatA[2] = 7; mf.quatA[3] = 8;
    mf.quatB[0] = 9; mf.quatB[1] = 10; mf.quatB[2] = 11; mf.quatB[3] = 12;
    mf.nbContacts = sc.countsWord;
    mf.nbWarmStart = 0u;          // ws13 = 0: seeds nothing, but words 14/15
    mf.warmA = 0xAABBCCDDu;       // must be READ-only (never stored back)
    mf.warmB = 0x11223344u;
    for (uint32_t j = 0; j < sc.preloadCount && j < 4; ++j)
    {
        // preloaded persistent-manifold points (the stage-1 refreshed state)
        mf.lNp[j][0] = sc.pre[j].n.x; mf.lNp[j][1] = sc.pre[j].n.y;
        mf.lNp[j][2] = sc.pre[j].n.z; mf.lNp[j][3] = sc.pre[j].pen;
        mf.lA[j][0] = sc.pre[j].a.x; mf.lA[j][1] = sc.pre[j].a.y;
        mf.lA[j][2] = sc.pre[j].a.z; mf.lA[j][3] = 0.0f;
        mf.lB[j][0] = sc.pre[j].b.x; mf.lB[j][1] = sc.pre[j].b.y;
        mf.lB[j][2] = sc.pre[j].b.z; mf.lB[j][3] = 0.0f;
    }

    PxgDevicePtr dBlob = 0, dBlob0 = 0, dIn = 0, dShapes = 0, dXf = 0, dCd = 0, dKeep = 0, dMf = 0;
    shim.memAlloc(&dBlob, blob.size());
    if (!dBlob) { std::printf("  NOTE: no BDA -> skip\n"); return false; }
    shapes[1].hullPtr = uint64_t(dBlob);
    if (cvx0)
    {
        shim.memAlloc(&dBlob0, blob0.size());
        shapes[0].hullPtr = uint64_t(dBlob0);
        shim.memcpyHtoD(dBlob0, blob0.data(), blob0.size());
    }
    shim.memAlloc(&dIn, sizeof(HInput));
    shim.memAlloc(&dShapes, shapes.size() * sizeof(HShape));
    shim.memAlloc(&dXf, xf.size() * sizeof(HCachedTransform));
    shim.memAlloc(&dCd, cd.size() * 4);
    shim.memAlloc(&dKeep, 4);
    shim.memAlloc(&dMf, sizeof(HManifold));

    shim.memcpyHtoD(dBlob, blob.data(), blob.size());
    shim.memcpyHtoD(dIn, &in, sizeof(HInput));
    shim.memcpyHtoD(dShapes, shapes.data(), shapes.size() * sizeof(HShape));
    shim.memcpyHtoD(dXf, xf.data(), xf.size() * sizeof(HCachedTransform));
    shim.memcpyHtoD(dCd, cd.data(), cd.size() * 4);
    shim.memcpyHtoD(dKeep, &keep, 4);
    shim.memcpyHtoD(dMf, &mf, sizeof(HManifold));

    Stage2Blob params = { 1u, 1.0f, dKeep, dIn, dMf, dShapes, dXf, dCd };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &params, sizeof(params)) == PxgResult::Success,
                  "launch stage2 (collide assembled)");
    shim.streamSynchronize();

    shim.memcpyDtoH(&out.mf, dMf, sizeof(HManifold));

    // moreData tail preservation: relativePos (0-3) / quatA (4-7) untouched,
    // quatB (8-11) rewritten byte-identically, warm words 13-15 NEVER stored.
    out.tailBad = 0;
    if (std::memcmp(out.mf.relPos, mf.relPos, 16) != 0) ++out.tailBad;
    if (std::memcmp(out.mf.quatA, mf.quatA, 16) != 0) ++out.tailBad;
    if (std::memcmp(out.mf.quatB, mf.quatB, 16) != 0) ++out.tailBad;
    if (out.mf.nbWarmStart != mf.nbWarmStart || out.mf.warmA != mf.warmA || out.mf.warmB != mf.warmB)
        ++out.tailBad;

    for (PxgDevicePtr p : { dBlob, dIn, dShapes, dXf, dCd, dKeep, dMf })
        shim.memFree(p);
    if (dBlob0)
        shim.memFree(dBlob0);
    return true;
}

static bool eq4(const float* g, float x, float y, float z, float w)
{
    return almostEqual(g[0], x, 2e-4f, 2e-5f) && almostEqual(g[1], y, 2e-4f, 2e-5f) &&
           almostEqual(g[2], z, 2e-4f, 2e-5f) && almostEqual(g[3], w, 2e-4f, 2e-5f);
}

static bool slotUntouched(const HManifold& m, uint32_t j)
{
    const uint8_t* np = reinterpret_cast<const uint8_t*>(m.lNp[j]);
    const uint8_t* a = reinterpret_cast<const uint8_t*>(m.lA[j]);
    const uint8_t* b = reinterpret_cast<const uint8_t*>(m.lB[j]);
    for (int k = 0; k < 16; ++k)
        if (np[k] != 0xCD || a[k] != 0xCD || b[k] != 0xCD) return false;
    return true;
}

// Order-independent slot match of the first `count` manifold slots.
static size_t matchContacts(const HManifold& m, const std::vector<ExpContact>& exp, bool quiet, const char* label)
{
    size_t bad = 0;
    const uint32_t nb = uint32_t(exp.size());
    std::vector<bool> used(nb, false);
    for (uint32_t k = 0; k < nb; ++k)
    {
        bool matched = false;
        for (uint32_t j = 0; j < nb && !matched; ++j)
        {
            if (used[j]) continue;
            if (eq4(m.lA[k], exp[j].a.x, exp[j].a.y, exp[j].a.z, 0.0f) &&
                eq4(m.lB[k], exp[j].b.x, exp[j].b.y, exp[j].b.z, 0.0f) &&
                eq4(m.lNp[k], exp[j].n.x, exp[j].n.y, exp[j].n.z, exp[j].pen))
            { used[j] = true; matched = true; }
        }
        if (!matched)
        {
            if (!quiet)
                std::printf("  [%s] slot %u unmatched: A(%g %g %g) B(%g %g %g) np(%g %g %g | %g)\n", label, k,
                            m.lA[k][0], m.lA[k][1], m.lA[k][2], m.lB[k][0], m.lB[k][1], m.lB[k][2],
                            m.lNp[k][0], m.lNp[k][1], m.lNp[k][2], m.lNp[k][3]);
            ++bad;
        }
    }
    return bad;
}

// Full check of one touching-scene expectation; returns the mismatch count.
static size_t checkScene(const Scene& sc, uint32_t expWord12, const std::vector<ExpContact>& exp,
                         bool quiet, const char* label)
{
    RunResult r;
    if (!runScene(sc, r)) return 0; // no BDA -> skip (counts as pass, like siblings)
    size_t bad = r.tailBad;
    if (r.tailBad && !quiet) std::printf("  [%s] moreData tail violated (%zu)\n", label, r.tailBad);
    if (r.mf.nbContacts != expWord12)
    {
        if (!quiet) std::printf("  [%s] word12 gpu=%08x ref=%08x\n", label, r.mf.nbContacts, expWord12);
        ++bad;
    }
    bad += matchContacts(r.mf, exp, quiet, label);
    for (uint32_t j = uint32_t(exp.size()); j < 4; ++j)
        if (!slotUntouched(r.mf, j))
        {
            if (!quiet) std::printf("  [%s] slot %u sentinel violated\n", label, j);
            ++bad;
        }
    return bad;
}

// ---------------------------------------------------------------------------
// 1+3. capsule face path: touching and separated-within-contactDistance.
// capsule (hh=0.5, r=0.5, axis X) over box (2, 0.5, 2); cd = 0.05+0.05;
// contactDist = 0.1 + r (expandedMargin = radius).  gap g = y0 - 0.5 from
// segment to box top; contacts iff g <= 0.6; np = ((0,1,0), g),
// A-local (+-hh, 0, 0), B-local (+-hh, 0.5, 0).
// ---------------------------------------------------------------------------
PXG_TEST(cvxCollide_capsuleFaceTouch)
{
    const Scene touch = { 2u, V3{ 0.5f, 0.5f, 0.5f }, V3{ 0, 1.0f, 0 }, V3{ 2, 0.5f, 2 }, 0.05f, 0u, false };
    std::vector<ExpContact> exp = {
        { V3{ 0.5f, 0, 0 }, V3{ 0.5f, 0.5f, 0 }, V3{ 0, 1, 0 }, 0.5f },
        { V3{ -0.5f, 0, 0 }, V3{ -0.5f, 0.5f, 0 }, V3{ 0, 1, 0 }, 0.5f },
    };
    size_t bad = checkScene(touch, 2u, exp, false, "touch");
    PXG_CHECK_MSG(bad == 0, "capsule touch: %zu mismatches", bad);
    std::printf("  [touch] 2 face contacts, np=(0,1,0,0.5) ok\n");

    const Scene near_ = { 2u, V3{ 0.5f, 0.5f, 0.5f }, V3{ 0, 1.05f, 0 }, V3{ 2, 0.5f, 2 }, 0.05f, 0u, false };
    std::vector<ExpContact> expN = {
        { V3{ 0.5f, 0, 0 }, V3{ 0.5f, 0.5f, 0 }, V3{ 0, 1, 0 }, 0.55f },
        { V3{ -0.5f, 0, 0 }, V3{ -0.5f, 0.5f, 0 }, V3{ 0, 1, 0 }, 0.55f },
    };
    bad = checkScene(near_, 2u, expN, false, "near");
    PXG_CHECK_MSG(bad == 0, "capsule near: %zu mismatches", bad);
    std::printf("  [near] 2 face contacts within contactDistance, np.w=0.55 ok\n");
}

// ---------------------------------------------------------------------------
// 2. deep overlap: segment inside the hull -> GJK overlap -> EPA; face rays
// start inside so tEnter clamps at 0 (the .cu tFirst=0 semantics): np.w = 0
// and localContactB = the segment endpoints.
// ---------------------------------------------------------------------------
PXG_TEST(cvxCollide_capsuleDeepEpa)
{
    const Scene deep = { 2u, V3{ 0.5f, 0.5f, 0.5f }, V3{ 0, 0.3f, 0 }, V3{ 2, 0.5f, 2 }, 0.05f, 0u, false };
    std::vector<ExpContact> exp = {
        { V3{ 0.5f, 0, 0 }, V3{ 0.5f, 0.3f, 0 }, V3{ 0, 1, 0 }, 0.0f },
        { V3{ -0.5f, 0, 0 }, V3{ -0.5f, 0.3f, 0 }, V3{ 0, 1, 0 }, 0.0f },
    };
    const size_t bad = checkScene(deep, 2u, exp, false, "deep");
    PXG_CHECK_MSG(bad == 0, "capsule deep/EPA: %zu mismatches", bad);
    std::printf("  [deep] EPA path, 2 face contacts at the segment (tEnter clamp) ok\n");
}

// ---------------------------------------------------------------------------
// 4. distant: gap beyond contactDistance -> collide() returns false; word 12
// keeps the refreshed count with the no-touch bit; slots untouched.
// ---------------------------------------------------------------------------
PXG_TEST(cvxCollide_capsuleDistant)
{
    const Scene far_ = { 2u, V3{ 0.5f, 0.5f, 0.5f }, V3{ 0, 2.5f, 0 }, V3{ 2, 0.5f, 2 }, 0.05f,
                         3u | (2u << 8), false };
    const size_t bad = checkScene(far_, 3u | 0x80000000u, {}, false, "distant");
    PXG_CHECK_MSG(bad == 0, "capsule distant: %zu mismatches", bad);
    std::printf("  [distant] eDISTANT -> word12=refreshedNb|0x80000000, slots untouched ok\n");
}

// ---------------------------------------------------------------------------
// 5. EE fallback: capsule (hh=0.5) overhanging a narrow box (hx=0.3) — both
// face rays hit the +-x plane separation, so generatedContactsEEContacts
// emits the two top-face edge crossings with the NEGATED normalPen:
// B-local (+-0.3, 0.5, 0), A-local (+-0.3, 0, 0), np = ((0,1,0), 0.4).
// ---------------------------------------------------------------------------
PXG_TEST(cvxCollide_capsuleEdgeEE)
{
    const Scene ee = { 2u, V3{ 0.5f, 0.5f, 0.5f }, V3{ 0, 0.9f, 0 }, V3{ 0.3f, 0.5f, 0.5f }, 0.05f, 0u, false };
    std::vector<ExpContact> exp = {
        { V3{ 0.3f, 0, 0 }, V3{ 0.3f, 0.5f, 0 }, V3{ 0, 1, 0 }, 0.4f },
        { V3{ -0.3f, 0, 0 }, V3{ -0.3f, 0.5f, 0 }, V3{ 0, 1, 0 }, 0.4f },
    };
    const size_t bad = checkScene(ee, 2u, exp, false, "ee");
    PXG_CHECK_MSG(bad == 0, "capsule EE: %zu mismatches", bad);
    std::printf("  [ee] face rays miss, 2 edge-edge contacts ok\n");
}

// ---------------------------------------------------------------------------
// 6. sphere type0: 1 contact, contactA = sphere centre (0), contactB = the
// GJK witness on the box, normal = -gjkOutput.direction = (0,1,0), sep = gap.
// ---------------------------------------------------------------------------
PXG_TEST(cvxCollide_sphereWitness)
{
    const Scene sph = { 0u, V3{ 0.5f, 0.5f, 0.5f }, V3{ 0, 1.0f, 0 }, V3{ 2, 0.5f, 2 }, 0.05f, 0u, false };
    std::vector<ExpContact> exp = {
        { V3{ 0, 0, 0 }, V3{ 0, 0.5f, 0 }, V3{ 0, 1, 0 }, 0.5f },
    };
    const size_t bad = checkScene(sph, 1u, exp, false, "sphere");
    PXG_CHECK_MSG(bad == 0, "sphere witness: %zu mismatches", bad);
    std::printf("  [sphere] 1 witness contact, A=centre, np=(0,1,0,0.5) ok\n");
}

// ---------------------------------------------------------------------------
// 7. mutation canary: flip the hull's +y plane; the touch expectations must
// FAIL — proves the test discriminates and the kernel reads the blob planes.
// ---------------------------------------------------------------------------
PXG_TEST(cvxCollide_mutation)
{
    const Scene mut = { 2u, V3{ 0.5f, 0.5f, 0.5f }, V3{ 0, 1.0f, 0 }, V3{ 2, 0.5f, 2 }, 0.05f, 0u, true };
    std::vector<ExpContact> exp = {
        { V3{ 0.5f, 0, 0 }, V3{ 0.5f, 0.5f, 0 }, V3{ 0, 1, 0 }, 0.5f },
        { V3{ -0.5f, 0, 0 }, V3{ -0.5f, 0.5f, 0 }, V3{ 0, 1, 0 }, 0.5f },
    };
    RunResult probe;
    if (!runScene(mut, probe)) return; // no BDA -> skip
    const size_t bad = checkScene(mut, 2u, exp, true, "mutation");
    PXG_CHECK_MSG(bad != 0, "mutation must FAIL the touch expectations");
    std::printf("  [mutation] flipped top plane breaks the expectations (%zu mismatches) ok\n", bad);
}

// ===========================================================================
// Tranche 11: the convex-hull arm (generateConvexContacts, cu L823-1022).
//
// Geometry shared by fixtures 8-12: shape0 = cube hull (half h0 = 0.5,
// type eCONVEXMESH = 5) at (0, y0, 0); shape1 = box hull (half (2, .5, 2)
// unless noted, type eBOX = 3) at the origin.  type1 (3) < type0 (5) flips
// the pair, so post-flip hull A = the BOX at the origin, hull B = the cube,
// aToBp = (0, -y0, 0).  gap = (y0 - 0.5) - boxHalf.y.  contactDistance =
// 0.05 + 0.05 + expandedMargin(0 for convex type0) = 0.1.
//
// Derivation of the face-face expectations (all axis-aligned fixtures):
// GJK/EPA return direction = (0,1,0) (the min-separation/penetration axis;
// side faces are >= 2 away).  bestPlaneAIdx = A's +y face and bestPlaneBIdx
// = B's -y face (getPolygonIndexFromLocalWitness: min plane distance 0 at
// the witness, projection tie-break onto -normal/normal picks the
// anti-parallel face, cu L677-689).  referenceProject == incidentProject ==
// |normal.y| exactly (rotation preserves y), so the strict > (cu L951) takes
// the ELSE branch: referencePlane = A's +y face transformed to B space
// (y = 0.5 - y0), axis = (0,1,0).  polyClip: B's 4 bottom verts all inside
// A's face -> the FIRST addContacts (cu L153) emits pos = v1 - s1*axis with
// s1 = plane0.distance(v1) = gap, then the !any(out1) early-out (cu L155).
// 4 contacts -> no reduce; the emit tail (cu L994-1015) writes
//   pa = clip pt (y_B = 0.5 - y0),  pb = pa + n*sep (y_B = -0.5),
//   mA = aToB.transformInv(pa) = pa + (0, y0, 0) (y_A = +0.5),
//   np = (-referencePlane.n, sep) = (0,-1,0, gap):
// the cube's bottom corners, A-local on the box top, B-local on the cube
// bottom.  word12 = 4 (touch).
// ===========================================================================

static Scene makeCvxScene(float y0, V3 boxHalf)
{
    Scene sc = { 5u, V3{ 1, 1, 1 }, V3{ 0, y0, 0 }, boxHalf, 0.05f, 0u, false };
    sc.h0 = V3{ 0.5f, 0.5f, 0.5f };
    return sc;
}

static std::vector<ExpContact> cvxCornerContacts(float sep)
{
    std::vector<ExpContact> v;
    const float xs[4] = { -0.5f, -0.5f, 0.5f, 0.5f };
    const float zs[4] = { -0.5f, 0.5f, -0.5f, 0.5f };
    for (int k = 0; k < 4; ++k)
        v.push_back({ V3{ xs[k], 0.5f, zs[k] }, V3{ xs[k], -0.5f, zs[k] }, V3{ 0, -1, 0 }, sep });
    return v;
}

// ---------------------------------------------------------------------------
// 8. face-face: barely-overlapping (y0 = 0.98, pen -0.02 -> GJK eOVERLAP ->
// EPA finds the +y face at depth 0.02) and separated-within-contactDistance
// (y0 = 1.05, gap +0.05 < 0.1 -> GJK eCLOSE, EPA skipped).  Both regenerate
// through polyClip (numManifoldContacts 1 < 4 after the witness add,
// cu L887) -> 4 corner contacts.
// ---------------------------------------------------------------------------
PXG_TEST(cvxCollide_hullFaceTouch)
{
    const Scene touch = makeCvxScene(0.98f, V3{ 2, 0.5f, 2 });
    size_t bad = checkScene(touch, 4u, cvxCornerContacts(-0.02f), false, "hullTouch");
    PXG_CHECK_MSG(bad == 0, "hull face touch (EPA): %zu mismatches", bad);
    std::printf("  [hullTouch] EPA, 4 corner contacts, np=(0,-1,0,-0.02) ok\n");

    const Scene near_ = makeCvxScene(1.05f, V3{ 2, 0.5f, 2 });
    bad = checkScene(near_, 4u, cvxCornerContacts(0.05f), false, "hullNear");
    PXG_CHECK_MSG(bad == 0, "hull near (GJK close): %zu mismatches", bad);
    std::printf("  [hullNear] GJK close, 4 corner contacts, np.w=+0.05 ok\n");
}

// ---------------------------------------------------------------------------
// 9. deep overlap (y0 = 0.7, pen -0.3): EPA path (top face at 0.3 vs sides
// at ~2.5), same clip -> 4 corner contacts with np.w = -0.3.
// ---------------------------------------------------------------------------
PXG_TEST(cvxCollide_hullDeepEpa)
{
    const Scene deep = makeCvxScene(0.7f, V3{ 2, 0.5f, 2 });
    const size_t bad = checkScene(deep, 4u, cvxCornerContacts(-0.3f), false, "hullDeep");
    PXG_CHECK_MSG(bad == 0, "hull deep/EPA: %zu mismatches", bad);
    std::printf("  [hullDeep] EPA path, 4 corner contacts, np.w=-0.3 ok\n");
}

// ---------------------------------------------------------------------------
// 10. distant (y0 = 1.5, gap 0.5 > 0.1): GJK eDISTANT -> collide() returns
// false (cu L1354-1357); word 12 = refreshedNb | 0x80000000, slots untouched.
// ---------------------------------------------------------------------------
PXG_TEST(cvxCollide_hullDistant)
{
    Scene far_ = makeCvxScene(1.5f, V3{ 2, 0.5f, 2 });
    far_.countsWord = 3u | (2u << 8);
    const size_t bad = checkScene(far_, 3u | 0x80000000u, {}, false, "hullDistant");
    PXG_CHECK_MSG(bad == 0, "hull distant: %zu mismatches", bad);
    std::printf("  [hullDistant] eDISTANT -> word12=refreshedNb|0x80000000, slots untouched ok\n");
}

// ---------------------------------------------------------------------------
// 11. rotated-45-degree edge-dominant clip: the cube rotated 45 deg about Y
// (q = (0, sin22.5, 0, cos22.5)) over an EQUAL box (half 0.5), gap +0.05.
// In B space, A's top face is B's bottom face rotated -45 deg: neither
// polygon's vertices are inside the other (corner radius 0.7071 > 0.5), so
// polyClip's two containment passes emit nothing and the edge-edge loop
// (cu L197-211) emits the 8 octagon crossings, all with sep = gap (s0d = 0
// on flat parallel faces) -> nbContacts 8 > 4 -> reduce<false,false,4>
// (cu L1005) keeps 4.  Every kept contact must be one of the 8 octagon
// vertices: B-local (+-1/2, +-t) / (+-t, +-1/2) with t = sqrt(2)/2 - 1/2, at
// y_B = -0.5; A-local = the +45-deg Y-rotation of the B point at y_A = +0.5;
// np = (0,-1,0, 0.05) exactly (pure-Y rotation preserves the y axis).
// The 4-subset reduce chooses is tie-break-dependent (all seps equal), so
// the assertion is set-membership + distinctness + count == 4.
// ---------------------------------------------------------------------------
PXG_TEST(cvxCollide_hullRot45)
{
    Scene rot = makeCvxScene(1.05f, V3{ 0.5f, 0.5f, 0.5f });
    const float s8 = std::sin(3.14159265358979323846 / 8.0);
    const float c8 = std::cos(3.14159265358979323846 / 8.0);
    rot.q0y = s8; rot.q0w = c8;

    RunResult r;
    if (!runScene(rot, r)) return; // no BDA -> skip
    size_t bad = r.tailBad;
    if (r.mf.nbContacts != 4u)
    {
        std::printf("  [rot45] word12 gpu=%08x ref=4\n", r.mf.nbContacts);
        ++bad;
    }

    // the 8 octagon candidates (B-local xz), A = rotY(+45)(B)
    const double t = std::sqrt(2.0) / 2.0 - 0.5;
    const double bx[8] = { 0.5, 0.5, -0.5, -0.5, t, -t, t, -t };
    const double bz[8] = { t, -t, t, -t, 0.5, 0.5, -0.5, -0.5 };
    const double ch = std::sqrt(2.0) / 2.0; // cos45 == sin45
    std::vector<ExpContact> cand;
    for (int k = 0; k < 8; ++k)
    {
        const double ax = bx[k] * ch + bz[k] * ch;   // x' = x c + z s
        const double az = bz[k] * ch - bx[k] * ch;   // z' = z c - x s
        cand.push_back({ V3{ float(ax), 0.5f, float(az) },
                         V3{ float(bx[k]), -0.5f, float(bz[k]) },
                         V3{ 0, -1, 0 }, 0.05f });
    }
    std::vector<bool> used(8, false);
    const uint32_t nb = r.mf.nbContacts & 0xFFu;
    for (uint32_t k = 0; k < nb && k < 4; ++k)
    {
        bool matched = false;
        for (int j = 0; j < 8 && !matched; ++j)
        {
            if (used[j]) continue;
            if (eq4(r.mf.lA[k], cand[j].a.x, cand[j].a.y, cand[j].a.z, 0.0f) &&
                eq4(r.mf.lB[k], cand[j].b.x, cand[j].b.y, cand[j].b.z, 0.0f) &&
                eq4(r.mf.lNp[k], cand[j].n.x, cand[j].n.y, cand[j].n.z, cand[j].pen))
            { used[j] = true; matched = true; }
        }
        if (!matched)
        {
            std::printf("  [rot45] slot %u not an octagon vertex: A(%g %g %g) B(%g %g %g) np(%g %g %g | %g)\n",
                        k, r.mf.lA[k][0], r.mf.lA[k][1], r.mf.lA[k][2],
                        r.mf.lB[k][0], r.mf.lB[k][1], r.mf.lB[k][2],
                        r.mf.lNp[k][0], r.mf.lNp[k][1], r.mf.lNp[k][2], r.mf.lNp[k][3]);
            ++bad;
        }
    }
    PXG_CHECK_MSG(bad == 0, "hull rot45: %zu mismatches", bad);
    std::printf("  [rot45] 8 EE crossings reduced to 4 distinct octagon vertices ok\n");
}

// ---------------------------------------------------------------------------
// 12. preloaded steady-state manifold (y0 = 1.05): 4 preloaded contacts with
// normal (0,-1,0) -> localNor = (0,-1,0), dot(localNor, -normal) = 1 >= 0.707
// and post-add count = 4.  5.12 puts `initialNbContacts > 1` in front of the
// predicate (cudaGJKEPA.cu:864, the PCM jitter fix), so the no-regen path needs
// the counts word's initial-count byte <= 1: (a)/(b) use countsWord = 4 (byte
// 1 zero) and fullContactGen is FALSE -> NO polyClip regen; the manifold is
// exactly the addManifoldPoint result.  (c) is (a) with the stage-1 packing of
// a 4-contact manifold, 4 | 4<<8: 5.6.1 skipped the regen there, 5.12 forces
// it, and the clip rebuilds the four corner contacts.
//   (a) REPLACE: slot 0 preloaded at the GJK witness (A (-0.5, .5, -.5) /
//       B (-0.5,-.5,-.5); the witness is the corner-most segment point —
//       GJK's tie-breaks pick support verts vA[2]/vA[7]/vB[0], the segment
//       simplex gives q = 12/32, cpA = (-0.5,-0.55,-0.5)) with a STALE
//       sep = 0.2; the new point is within replaceBreakingThreshold
//       (identical A/B) -> replaces slot 0, refreshing sep to +0.05
//       (cu L574-586).  A clip regen would instead move slots 1-3 to the
//       +-0.5 corners — discriminated.
//   (b) REDUCE-DROP: all 4 slots at +-0.45 corners with sep 0.04 (DEEPER
//       than the witness sep ~0.04999995 — fp: 0.5 - 1.05f + 0.5 rounds
//       BELOW 0.05f, so equal-sep preloads would make the witness the
//       deepest candidate i0 and the reduce would keep it, dropping slot 0
//       — verified on-device).  No match (nearest B-distance^2 = 0.005 >>
//       rbt^2 = 6.25e-6, rbt = min(pcmMargin) * 0.05 = 0.0025) -> full +
//       matchless -> the 5-candidate reduce (cu L588-601): i0 = slot0
//       (lowest-lane min sep 0.04), i1 = slot2 (farthest, 1.62), f = slot3
//       (max d = +0.81), g = slot1 (min d = -0.81, opposite sign kept) ->
//       the new witness (lane 4, d = 0 by symmetric cancellation) is
//       DROPPED and the compaction preserves slots 0-3 verbatim.
// ---------------------------------------------------------------------------
PXG_TEST(cvxCollide_hullManifoldSteady)
{
    // (a) replace
    Scene rep = makeCvxScene(1.05f, V3{ 2, 0.5f, 2 });
    rep.countsWord = 4u;   // initialNbContacts 0: the 5.12 jitter term stays off
    rep.preloadCount = 4;
    rep.pre[0] = { V3{ -0.5f, 0.5f, -0.5f }, V3{ -0.5f, -0.5f, -0.5f }, V3{ 0, -1, 0 }, 0.2f };
    rep.pre[1] = { V3{ 0.45f, 0.5f, -0.45f }, V3{ 0.45f, -0.5f, -0.45f }, V3{ 0, -1, 0 }, 0.05f };
    rep.pre[2] = { V3{ 0.45f, 0.5f, 0.45f }, V3{ 0.45f, -0.5f, 0.45f }, V3{ 0, -1, 0 }, 0.05f };
    rep.pre[3] = { V3{ -0.45f, 0.5f, 0.45f }, V3{ -0.45f, -0.5f, 0.45f }, V3{ 0, -1, 0 }, 0.05f };
    std::vector<ExpContact> expR;
    expR.push_back({ rep.pre[0].a, rep.pre[0].b, V3{ 0, -1, 0 }, 0.05f }); // sep refreshed
    expR.push_back(rep.pre[1]);
    expR.push_back(rep.pre[2]);
    expR.push_back(rep.pre[3]);
    size_t bad = checkScene(rep, 4u, expR, false, "steadyReplace");
    PXG_CHECK_MSG(bad == 0, "hull steady replace: %zu mismatches", bad);
    std::printf("  [steadyReplace] witness replaced slot 0 (sep 0.2 -> 0.05), no regen ok\n");

    // (b) reduce-drop
    Scene drop = makeCvxScene(1.05f, V3{ 2, 0.5f, 2 });
    drop.countsWord = 4u;
    drop.preloadCount = 4;
    drop.pre[0] = { V3{ -0.45f, 0.5f, -0.45f }, V3{ -0.45f, -0.5f, -0.45f }, V3{ 0, -1, 0 }, 0.04f };
    drop.pre[1] = { V3{ 0.45f, 0.5f, -0.45f }, V3{ 0.45f, -0.5f, -0.45f }, V3{ 0, -1, 0 }, 0.04f };
    drop.pre[2] = { V3{ 0.45f, 0.5f, 0.45f }, V3{ 0.45f, -0.5f, 0.45f }, V3{ 0, -1, 0 }, 0.04f };
    drop.pre[3] = { V3{ -0.45f, 0.5f, 0.45f }, V3{ -0.45f, -0.5f, 0.45f }, V3{ 0, -1, 0 }, 0.04f };
    std::vector<ExpContact> expD(drop.pre, drop.pre + 4);
    bad = checkScene(drop, 4u, expD, false, "steadyDrop");
    PXG_CHECK_MSG(bad == 0, "hull steady reduce-drop: %zu mismatches", bad);
    std::printf("  [steadyDrop] 5-candidate reduce dropped the witness, manifold intact ok\n");

    // (c) the 5.12 jitter fix: same preload as (a), initialNbContacts = 4
    Scene jit = rep;
    jit.countsWord = 4u | (4u << 8);
    bad = checkScene(jit, 4u, cvxCornerContacts(0.05f), false, "steadyJitter512");
    PXG_CHECK_MSG(bad == 0, "hull steady, initialNbContacts 4 (5.12 forces the regen): %zu mismatches", bad);
    std::printf("  [steadyJitter512] initialNbContacts > 1 forced the clip regen ok\n");
}

// ---------------------------------------------------------------------------
// 13. mutation canary: negate hull A's +y plane (shape1's blob).  The
// incident plane flips orientation, so the else-branch clip axis becomes
// (0,-1,0), inverting polyClip's edge-plane in/out tests — no clip contacts
// are emitted and the manifold keeps only the addManifoldPoint witness
// (word12 = 1): the hullNear expectations must FAIL (proves the kernel
// reads hull A's blob planes).
// ---------------------------------------------------------------------------
PXG_TEST(cvxCollide_hullMutation)
{
    Scene mut = makeCvxScene(1.05f, V3{ 2, 0.5f, 2 });
    mut.flipTopPlane = true;
    RunResult probe;
    if (!runScene(mut, probe)) return; // no BDA -> skip
    const size_t bad = checkScene(mut, 4u, cvxCornerContacts(0.05f), true, "hullMutation");
    PXG_CHECK_MSG(bad != 0, "hull mutation must FAIL the near expectations");
    std::printf("  [hullMutation] flipped A top plane breaks the expectations (%zu mismatches) ok\n", bad);
}


// ---------------------------------------------------------------------------
// 14. P9.90: the capsule face-ray dot order.  intersectRayPolyhedron's
// denominator = n.dot(dir) and distToPlane = n.dot(a) + d (cudaGJKEPA.cu
// :1051-1052) are PxVec3 dots, (x + y) + z (PxVec3.h:173); the port spells
// them dot(), an OpDot, which llvmpipe sums x + (y + z) unless the .spv is
// stamped NoContraction.  The scene makes every term non-zero: hull B is a
// box whose vertices AND plane normals are rotated by a general quaternion
// R, and a capsule lies over its +y face (rotated by R, then yawed), so the
// GJK direction, the ray origins and the plane normals all have three
// non-zero, non-round components.  The GJK/EPA path has no OpDot, so the
// test takes the GJK direction from the kernel's own output (np.xyz = -dir)
// and transcribes everything downstream of it in the CUDA order:
// prepareVertices (cu L2465-2476 + cudaNpCommon.h:315-343), the vertex-space
// ray (cu L1103-1107), intersectRayPolyhedron and the slot store
// (cu L1112-1136).  np.w = tEnter and localContactB = vA - dir * tEnter are
// compared BITWISE; a twin with the old x + (y + z) dot must disagree (the
// coverage guard: the scene is order-sensitive).
// ---------------------------------------------------------------------------
static V3 p90QRot(const float q[4], V3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q[3] * q[3] - 0.5f;
    const float d2 = q[0] * vx + q[1] * vy + q[2] * vz;
    return V3{ vx * w2 + (q[1] * vz - q[2] * vy) * q[3] + q[0] * d2,
               vy * w2 + (q[2] * vx - q[0] * vz) * q[3] + q[1] * d2,
               vz * w2 + (q[0] * vy - q[1] * vx) * q[3] + q[2] * d2 };
}
static V3 p90QRotInv(const float q[4], V3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q[3] * q[3] - 0.5f;
    const float d2 = q[0] * vx + q[1] * vy + q[2] * vz;
    return V3{ vx * w2 - (q[1] * vz - q[2] * vy) * q[3] + q[0] * d2,
               vy * w2 - (q[2] * vx - q[0] * vz) * q[3] + q[1] * d2,
               vz * w2 - (q[0] * vy - q[1] * vx) * q[3] + q[2] * d2 };
}
static void p90QMul(const float a[4], const float b[4], float o[4])
{
    o[0] = a[3] * b[0] + b[3] * a[0] + a[1] * b[2] - a[2] * b[1];
    o[1] = a[3] * b[1] + b[3] * a[1] + a[2] * b[0] - a[0] * b[2];
    o[2] = a[3] * b[2] + b[3] * a[2] + a[0] * b[1] - a[1] * b[0];
    o[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2];
}
static V3 p90Sub(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 p90Add(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 p90MulV(V3 a, V3 b) { return V3{ a.x * b.x, a.y * b.y, a.z * b.z }; }
static V3 p90DivV(V3 a, V3 b) { return V3{ a.x / b.x, a.y / b.y, a.z / b.z }; }
static float p90DotCuda(V3 a, V3 b) { return (a.x * b.x + a.y * b.y) + a.z * b.z; } // PxVec3::dot
static float p90DotOld(V3 a, V3 b) { return a.x * b.x + (a.y * b.y + a.z * b.z); } // unstamped OpDot
static uint32_t p90Bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// the box hull of buildBoxHull with every vertex and plane normal rotated by R;
// d = -n.dot(first face vertex) (the winding stays outward under a rotation)
static std::vector<uint8_t> buildRotBoxHull(V3 h, const float R[4])
{
    std::vector<uint8_t> blob = buildBoxHull(h);
    const size_t vertsBase = 16 + 32, planesBase = vertsBase + 16 * 8;
    V3 rv[8];
    for (int i = 0; i < 8; ++i)
    {
        rv[i] = p90QRot(R, boxCorner(i, h));
        std::memcpy(&blob[vertsBase + 16 * i], &rv[i], 12);
    }
    for (int f = 0; f < 6; ++f)
    {
        const V3 n = p90QRot(R, V3{ kBoxFaceN[f][0], kBoxFaceN[f][1], kBoxFaceN[f][2] });
        const float d = -p90DotCuda(n, rv[kBoxFaceVtx[f][0]]);
        const float pl[4] = { n.x, n.y, n.z, d };
        std::memcpy(&blob[planesBase + 16 * f], pl, 16);
    }
    return blob;
}

// intersectRayPolyhedron (cu L1025-1097; one warp block: nbPolygons = 6 < 32)
static bool p90Ray(const std::vector<uint8_t>& blob, V3 a, V3 dir, float (*dotFn)(V3, V3), float& tEnter)
{
    const size_t planesBase = 16 + 32 + 16 * 8;
    const float eps = 1e-7f;
    float tFirst = 0.0f, tLast = 3.4028234663852886e38f;
    tEnter = 0.0f;
    bool blockSep = false;
    float newFirst = tFirst, newLast = tLast;
    for (int k = 0; k < 6; ++k)
    {
        float pl[4];
        std::memcpy(pl, &blob[planesBase + 16 * k], 16);
        const V3 n{ pl[0], pl[1], pl[2] };
        const float denominator = dotFn(n, dir);
        const float distToPlane = dotFn(n, a) + pl[3];
        if (std::fabs(denominator) < eps)
        {
            if (distToPlane > 0.0f) blockSep = true;
        }
        else
        {
            const float tTemp = -distToPlane / denominator;
            const bool con = 0.0f > denominator;
            if (con && tTemp > tFirst) newFirst = std::fmax(newFirst, tTemp);
            if (tLast > tTemp && !con) newLast = std::fmin(newLast, tTemp);
        }
    }
    if (blockSep) return false;
    tFirst = newFirst; tLast = newLast;
    if (tFirst > tLast) return false;
    tEnter = tFirst;
    return true;
}

struct P90Case { float R[4]; float yaw; V3 off; float gap; };

// Runs one case; returns the bitwise mirror mismatches, reports the old-order
// twin's differing words and whether the scene produced two face contacts.
static size_t p90RunCase(const P90Case& pc, size_t& twinDiff, bool& twoFace, bool quiet)
{
    twinDiff = 0; twoFace = false;
    float R[4];
    {
        const float m = std::sqrt(((pc.R[0] * pc.R[0] + pc.R[1] * pc.R[1]) + pc.R[2] * pc.R[2]) + pc.R[3] * pc.R[3]);
        for (int i = 0; i < 4; ++i) R[i] = pc.R[i] / m;
    }
    const V3 h{ 1.7f, 0.43f, 1.3f };
    const std::vector<uint8_t> hull = buildRotBoxHull(h, R);
    const float yawQ[4] = { 0.0f, std::sin(0.5f * pc.yaw), 0.0f, std::cos(0.5f * pc.yaw) };
    float q0[4];
    p90QMul(R, yawQ, q0);
    const float hh = 0.37f, rad = 0.29f, cdPer = 0.061f;
    const V3 pos0 = p90QRot(R, V3{ pc.off.x, h.y + rad + pc.gap, pc.off.z });

    Scene sc = { 2u, V3{ hh, rad, rad }, pos0, h, cdPer, 0u, false };
    sc.q0x = q0[0]; sc.q0y = q0[1]; sc.q0z = q0[2]; sc.q0w = q0[3];
    sc.hull1 = &hull;
    RunResult r;
    if (!runScene(sc, r)) return 0; // no BDA -> skip
    size_t bad = r.tailBad;
    if (r.mf.nbContacts != 2u)
    {
        if (!quiet) std::printf("  [p90] word12 gpu=%08x, want 2 face contacts\n", r.mf.nbContacts);
        return bad + 1;
    }

    // prepareVertices: aToB = transf1.transformInv(transf0), transf1 = identity
    const float t1q[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    const float c1[4] = { -t1q[0], -t1q[1], -t1q[2], t1q[3] };
    float aq[4];
    p90QMul(c1, q0, aq);
    // P9.94d: transformInv(PxTransform).p = q.getConjugate().rotate(src.p - p)
    // (PxTransform.h:162-169), not q.rotateInv -- they differ in the sign of an exact zero.
    const V3 ap = p90QRot(c1, p90Sub(pos0, V3{ 0.0f, 0.0f, 0.0f }));
    const float idq[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    const V3 sc0{ hh, rad, rad }, one{ 1.0f, 1.0f, 1.0f };
    auto vertex2Shape = [&](V3 v, V3 s) { return p90QRotInv(idq, p90MulV(p90QRot(idq, v), s)); };
    auto shape2Vertex = [&](V3 v, V3 s) { return p90QRotInv(idq, p90DivV(p90QRot(idq, v), s)); };
    const V3 vA[2] = { p90Add(p90QRot(aq, vertex2Shape(V3{ 1.0f, 0.0f, 0.0f }, sc0)), ap),
                       p90Add(p90QRot(aq, vertex2Shape(V3{ -1.0f, 0.0f, 0.0f }, sc0)), ap) };
    const float inflatedRadius = (cdPer + cdPer) + rad;

    const V3 dir{ r.mf.lNp[0][0], r.mf.lNp[0][1], r.mf.lNp[0][2] }; // = -gjkOutput.direction
    const V3 vsd = shape2Vertex(V3{ -dir.x, -dir.y, -dir.z }, one);
    twoFace = true;
    for (int k = 0; k < 2; ++k)
    {
        const V3 p = shape2Vertex(vA[k], one);
        float tNew = 0.0f, tOld = 0.0f;
        const bool hitNew = p90Ray(hull, p, vsd, &p90DotCuda, tNew) && inflatedRadius >= tNew;
        const bool hitOld = p90Ray(hull, p, vsd, &p90DotOld, tOld) && inflatedRadius >= tOld;
        if (!hitNew) { twoFace = false; ++bad; continue; }
        const V3 bNew = p90Sub(vA[k], V3{ dir.x * tNew, dir.y * tNew, dir.z * tNew });
        const V3 bOld = p90Sub(vA[k], V3{ dir.x * tOld, dir.y * tOld, dir.z * tOld });
        const V3 aExp = p90QRotInv(aq, p90Sub(vA[k], ap));
        const float expNp[4] = { dir.x, dir.y, dir.z, tNew };
        const float expA[4] = { aExp.x, aExp.y, aExp.z, 0.0f };
        const float expB[4] = { bNew.x, bNew.y, bNew.z, 0.0f };
        const float oldW[4] = { tOld, bOld.x, bOld.y, bOld.z };
        const float newW[4] = { tNew, bNew.x, bNew.y, bNew.z };
        size_t slotBad = 0;
        for (int c = 0; c < 4; ++c)
        {
            if (p90Bits(r.mf.lNp[k][c]) != p90Bits(expNp[c])) ++slotBad;
            if (p90Bits(r.mf.lA[k][c]) != p90Bits(expA[c])) ++slotBad;
            if (p90Bits(r.mf.lB[k][c]) != p90Bits(expB[c])) ++slotBad;
            if (!hitOld || p90Bits(oldW[c]) != p90Bits(newW[c])) ++twinDiff;
        }
        if (slotBad && !quiet)
            std::printf("  [p90] slot %d: gpu np.w=%08x B=(%08x %08x %08x) | CUDA-order %08x (%08x %08x %08x) | old-order %08x (%08x %08x %08x)\n",
                        k, p90Bits(r.mf.lNp[k][3]), p90Bits(r.mf.lB[k][0]), p90Bits(r.mf.lB[k][1]), p90Bits(r.mf.lB[k][2]),
                        p90Bits(tNew), p90Bits(bNew.x), p90Bits(bNew.y), p90Bits(bNew.z),
                        p90Bits(tOld), p90Bits(bOld.x), p90Bits(bOld.y), p90Bits(bOld.z));
        bad += slotBad;
    }
    // slots 2/3 untouched
    for (uint32_t j = 2; j < 4; ++j)
        if (!slotUntouched(r.mf, j)) ++bad;
    return bad;
}

PXG_TEST(cvxCollide_p90RayDotOrder)
{
    // three drawn scenes (R unnormalized, normalized in p90RunCase; yaw about
    // the face normal; offset along the face; capsule-surface gap), each one
    // chosen so the old-order twin changes 7 of its 8 compared words
    static const P90Case kCases[3] = {
        { { -0.843457997f, -0.200796008f, -0.844009995f, 0.976701915f }, -0.59839797f,
          V3{ 0.483326435f, 0.0f, 0.358987182f }, 0.0648716539f },
        { { 0.60563004f, -0.535809994f, 0.018075943f, 0.951876402f }, -0.119771957f,
          V3{ -0.0635100007f, 0.0f, 0.0359064043f }, 0.0551702827f },
        { { 0.343778014f, -0.484485984f, -0.0675820112f, 0.450248003f }, 1.49427295f,
          V3{ -0.0520223975f, 0.0f, -0.108169585f }, 0.0664741024f },
    };
    for (int c = 0; c < 3; ++c)
    {
        size_t twin = 0;
        bool two = false;
        const size_t bad = p90RunCase(kCases[c], twin, two, false);
        PXG_CHECK_MSG(bad == 0, "p90 case %d: %zu words differ from the CUDA-order (x + y) + z transcription", c, bad);
        PXG_CHECK_MSG(two, "p90 case %d: the scene no longer yields two face-ray contacts", c);
        PXG_CHECK_MSG(twin != 0, "p90 case %d: the old-order x + (y + z) twin equals the CUDA order -- scene not order-sensitive", c);
        std::printf("  [p90] case %d: 2 face contacts bitwise in the CUDA dot order; old-order twin differs in %zu/8 words\n", c, twin);
    }
}

} // namespace
