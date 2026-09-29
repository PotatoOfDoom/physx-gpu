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

// P5 Step 5 tranche C: convexPlaneNphase_Kernel (npCvxPlaneNphase.slang) on the
// ENGINE launch ABI, driven through PxgCudaShim exactly as
// PxgNarrowphaseCore.cpp launches it — 20-argument CUDA parameter pack packed
// to a 144-byte natural-alignment blob, grid ceil(numTests/4) x (32,4) blocks,
// REAL PxgShape / PxsCachedTransform / PxsContactManagerOutput /
// PxgPatchAndContactCounters / PxsMaterialData byte layouts and a REAL cooked
// box convex-hull blob (header/extents/verts/planes/polyDesc/facesByEdges8/
// vertexData8 at the GRB byte offsets the kernel dereferences off
// shape.hullOrMeshPtr) plus a live 256-byte PxgPersistentContactManifold.
//
// VERIFICATION LEVEL: geometric integration.  A unit box rests on an infinite
// PxPlane; the CPU reference derives the expected contacts directly from the
// box-on-plane geometry (the bottom face's corners projected to the plane).
// It covers:
//   1. the REGEN path (empty manifold, box resting -> 4 contacts, world points
//      / separations / patch normal / counters / touch+patch flags / cmOutput
//      stream pointers, and the populated manifold's mNbContacts);
//   2. the REFRESH/RELOAD path across TWO invocations on the same device
//      manifold (second launch reuses the stored PCM instead of regenerating —
//      proven by perturbing the stored contact and observing the emitted
//      points follow the perturbed reload, not a fresh regen);
//   3. the SEPARATED no-touch path (box lifted beyond contactDistance -> zero
//      contacts, eHAS_NO_TOUCH, counters untouched, NULL cmOutput pointers);
//   4. a MUTATION negative control (swapped startContactPoints/Forces must make
//      the geometric check fail).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"
#include "pxgTriTriRef.h"

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

static const uint32_t INVALID = 0xffffffffu;

// P9.87: THE PATCH STREAM IS POISONED, AND NOT UNIFORMLY.  insertIntoPatchStream
// (contactPatchUtils.cuh:161-178) assigns the header field by field: words 0-12
// whole, word 13 only its low half (:171, a PxU16 store), words 14-15 never.  A
// zero-filled stream cannot tell "left alone" from "stored 0", which is how the
// shared tail zeroed pad[0] unnoticed.  The poison differs per word position so an
// RMW that reads the WRONG word (14 or 15 instead of 13) is visible too.
static const uint32_t POISON = 0xCDCDCDCDu;
static uint32_t poisonWord(uint64_t k) { return POISON ^ (uint32_t(k) * 0x00010001u); }

// ---------------------------------------------------------------------------
// Host mirrors of the engine element structs (byte-exact vs the C++ headers).
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct HInput { uint32_t shapeRef0, shapeRef1, transformCacheRef0, transformCacheRef1; };  // 16
struct HCachedTransform { float qx, qy, qz, qw, px, py, pz; uint32_t flags; };              // 32
struct HShape { float sx, sy, sz, srx, sry, srz, srw; uint32_t materialIndex;
                uint64_t hullPtr; uint32_t type, particleId; };                             // 48
struct HOutput { uint64_t contactPatches, contactPoints, contactForces, frictionPatches;
                 uint32_t allflagsWord, nbContacts_flags; uint32_t pad0, pad1; };           // 48
struct HCounters { uint32_t patchesBytes, contactsBytes, forceAndIndiceBytes, overflowError; }; // 16
struct HMaterial { float dyn, sta, rest, damp; uint32_t packedFlags, packedDamp; };         // 24
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
static_assert(sizeof(HOutput) == 48, "HOutput");
static_assert(sizeof(HCounters) == 16, "HCounters");
static_assert(sizeof(HMaterial) == 24, "HMaterial");
static_assert(sizeof(HManifold) == 256, "HManifold");

// ---------------------------------------------------------------------------
// Minimal vector / quaternion math matching the shader's PxQuat convention.
// ---------------------------------------------------------------------------
struct V3 { float x, y, z; };
struct Q4 { float x, y, z, w; };

static V3 add(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 sub(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 qRot(Q4 q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z, w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz;
    return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
               vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
               vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2 };
}
// world = xform.transform(local)
static V3 xform(Q4 q, V3 p, V3 local) { return add(qRot(q, local), p); }

// ---------------------------------------------------------------------------
// A geometrically valid cooked box convex-hull blob (8 verts, 6 faces, 12
// edges) laid out at the GRB byte offsets the kernel reads (relative to
// convexPtrB = hullPtr + 16).  Vertices carry the box half-extents directly
// (mesh scale = identity), so vertex2Shape(v) == v.
// ---------------------------------------------------------------------------
static const int kBoxFaceVtx[6][4] = {
    { 4, 5, 7, 6 }, // +x
    { 0, 2, 3, 1 }, // -x
    { 2, 6, 7, 3 }, // +y
    { 0, 1, 5, 4 }, // -y
    { 1, 3, 7, 5 }, // +z
    { 0, 4, 6, 2 }, // -z
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
    // extents float4 (.xyz internal AABB, .w inSphereRadius — unused by plane path)
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
        // plane d = -(n . faceVertex); irrelevant to the plane kernel but keeps the hull sane
        const V3 fv = boxCorner(kBoxFaceVtx[f][0], h);
        setF(planesBase + 16 * f + 12, -(kBoxFaceN[f][0] * fv.x + kBoxFaceN[f][1] * fv.y + kBoxFaceN[f][2] * fv.z));
    }
    const size_t polyDescBase = planesBase + 16 * P;
    for (uint32_t f = 0; f < P; ++f)
        setU32(polyDescBase + 4 * f, (4u * f) << 16 | (4u << 8) | 0u); // vRef8<<16 | nbVerts<<8 | minIndex

    // verticesByEdges16 [polyDescBase + 4P, +4E) is unused by the plane kernel (left 0).
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

// ---------------------------------------------------------------------------
// CPU reference: the contacts the kernel must emit for one box-on-plane pair.
// A contact is a bottom-face corner within contactDistance of the plane; its
// world point is the corner and its separation is the signed plane distance.
// ---------------------------------------------------------------------------
struct RefContact { V3 p; float sep; };
struct PlaneScene
{
    V3 halfExt;
    Q4 planeQ; V3 planeP;
    Q4 convexQ; V3 convexP;
    float contactDist;
    uint32_t matPlane, matConvex;
};

static V3 worldPlaneNormal(const PlaneScene& s) { return qRot(s.planeQ, V3{ 1, 0, 0 }); }

static std::vector<RefContact> refContacts(const PlaneScene& s)
{
    const V3 wn = worldPlaneNormal(s);
    std::vector<RefContact> out;
    for (int i = 0; i < 8; ++i)
    {
        const V3 wc = xform(s.convexQ, s.convexP, boxCorner(i, s.halfExt));
        const float sd = dot3(sub(wc, s.planeP), wn);
        if (sd < s.contactDist) out.push_back({ wc, sd });
    }
    return out;
}

// ---------------------------------------------------------------------------
// Scene -> engine arrays, one pair.  Convex is shapeRef1 (order-independent:
// the kernel identifies the ePLANE shape).
// ---------------------------------------------------------------------------
static HCachedTransform makeXf(Q4 q, V3 p) { return { q.x, q.y, q.z, q.w, p.x, p.y, p.z, 0u }; }

struct DeviceScene
{
    PxgCudaShim* shim;
    PxgDevicePtr dIn = 0, dOut = 0, dShapes = 0, dXf = 0, dCd = 0, dMat = 0, dCStream = 0, dPStream = 0,
                 dCounters = 0, dTouch = 0, dPatch = 0, dForce = 0, dBlob = 0, dMf = 0;
    uint32_t contactLimit = 1u << 20, patchLimit = 1u << 20, forceLimit = 1u << 20;
    std::vector<uint8_t> hostBlob;

    void free()
    {
        for (PxgDevicePtr p : { dIn, dOut, dShapes, dXf, dCd, dMat, dCStream, dPStream, dCounters, dTouch, dPatch, dForce, dBlob, dMf })
            if (p) shim->memFree(p);
    }
};

// The 144-byte engine parameter blob (argument order per PxgNarrowphaseCore.cpp).
struct EngineBlob
{
    uint32_t numTests;
    uint64_t cmInputs, cmOutputs, shapes, transformCache, contactDistance, materials, contactStream,
             patchStream, cmManifolds, counters, touch, patch, startPatch, startPoint, startForce;
    uint32_t patchLimit, contactLimit, forceLimit; float tolerance;
};
static_assert(sizeof(EngineBlob) == 144, "convexPlaneNphase engine blob is 144 bytes");

static void poisonPatchStream(PxgCudaShim& shim, const DeviceScene& d)
{
    std::vector<uint32_t> w(d.patchLimit / 4);
    for (size_t k = 0; k < w.size(); ++k) w[k] = poisonWord(k);
    shim.memcpyHtoD(d.dPStream, w.data(), w.size() * 4);
}

// Materials palette (indexed by shape.materialIndex).
// P9.88: a second bank at [0x10000 + i] holds DIFFERENT materials, so a scene with index
// k | 0x10000 tells a lookup that narrows to PxU16 (contactPatchUtils.cuh:149-150) from one
// that does not, and both stay in bounds.
static std::vector<HMaterial> makeMaterials()
{
    std::vector<HMaterial> m(0x10000u + 8u);
    for (uint32_t i = 0; i < 8u; ++i)
    {
        const float f = 0.2f + 0.1f * i;
        m[i] = { f, f + 0.3f, 0.4f, 0.05f * i,
                 uint32_t(uint32_t(i % 4) << 16) | (uint32_t((i + 1) % 4) << 24), uint32_t((i + 2) % 4) };
        const float g = 1.1f - 0.07f * i;
        m[0x10000u + i] = { g, g + 0.15f, 0.1f + 0.05f * i, 0.9f - 0.1f * i,
                            uint32_t(uint32_t((i + 3) % 4) << 16) | (uint32_t((i + 2) % 4) << 24), uint32_t((i + 1) % 4) };
    }
    // P9.86c: +-0 material ties (every value legal: NpMaterial.cpp:110/125/140/161 accept -0).
    // [8]/[9] -0/+0 with every combine mode eMIN (1), [10]/[11] +0/-0 with eMAX (3).
    // PxMin/PxMax<float> = selectMin/selectMax keep the SECOND operand on a tie
    // (PxUnixMathIntrinsics.h:116/122); GLSL FMin/FMax (the reverted port) are sign-aware.
    m[8]  = { -0.f, -0.f, -0.f, -0.f, (1u << 16) | (1u << 24), 1u };
    m[9]  = {  0.f,  0.f,  0.f,  0.f, (1u << 16) | (1u << 24), 1u };
    m[10] = {  0.f,  0.f,  0.f,  0.f, (3u << 16) | (3u << 24), 3u };
    m[11] = { -0.f, -0.f, -0.f, -0.f, (3u << 16) | (3u << 24), 3u };
    return m;
}

static bool inRange(uint64_t a, uint64_t base, uint64_t bytes, uint64_t span) { return a >= base && a + bytes <= base + span; }

// Build device buffers for one pair; the manifold is provided by the caller.
static void buildDevice(PxgCudaShim& shim, const PlaneScene& s, const HManifold& mf, uint32_t priorAllflags, DeviceScene& d)
{
    d.shim = &shim;
    d.hostBlob = buildBoxHull(s.halfExt);

    std::vector<HMaterial> materials = makeMaterials();
    HInput in = { 0, 1, 0, 1 }; // shape/cache 0 = plane, 1 = convex
    std::vector<HCachedTransform> xf = { makeXf(s.planeQ, s.planeP), makeXf(s.convexQ, s.convexP) };
    std::vector<float> cd = { 0.5f * s.contactDist, 0.5f * s.contactDist };
    std::vector<HShape> shapes(2);
    shapes[0] = { 1, 1, 1, 0, 0, 0, 1, s.matPlane, 0u, 1u /*ePLANE*/, 0u };            // plane
    shapes[1] = { 1, 1, 1, 0, 0, 0, 1, s.matConvex, 0u, 3u /*eBOX*/, 0u };             // convex (hullPtr filled below)
    HOutput out; std::memset(&out, 0, sizeof(out));
    out.allflagsWord = priorAllflags; out.nbContacts_flags = 0xAAAA0000u;
    HCounters counters = { 0, 0, 0, 0 };
    uint32_t touch = 7u, patch = 7u;

    shim.memAlloc(&d.dBlob, d.hostBlob.size());
    shapes[1].hullPtr = uint64_t(d.dBlob);

    shim.memAlloc(&d.dIn, sizeof(HInput));
    shim.memAlloc(&d.dOut, sizeof(HOutput));
    shim.memAlloc(&d.dShapes, shapes.size() * sizeof(HShape));
    shim.memAlloc(&d.dXf, xf.size() * sizeof(HCachedTransform));
    shim.memAlloc(&d.dCd, cd.size() * sizeof(float));
    shim.memAlloc(&d.dMat, materials.size() * sizeof(HMaterial));
    shim.memAlloc(&d.dCStream, d.contactLimit);
    shim.memAlloc(&d.dPStream, d.patchLimit);
    shim.memAlloc(&d.dCounters, sizeof(HCounters));
    shim.memAlloc(&d.dTouch, 4);
    shim.memAlloc(&d.dPatch, 4);
    shim.memAlloc(&d.dForce, d.forceLimit);
    shim.memAlloc(&d.dMf, sizeof(HManifold));

    shim.memcpyHtoD(d.dBlob, d.hostBlob.data(), d.hostBlob.size());
    shim.memcpyHtoD(d.dIn, &in, sizeof(HInput));
    shim.memcpyHtoD(d.dOut, &out, sizeof(HOutput));
    shim.memcpyHtoD(d.dShapes, shapes.data(), shapes.size() * sizeof(HShape));
    shim.memcpyHtoD(d.dXf, xf.data(), xf.size() * sizeof(HCachedTransform));
    shim.memcpyHtoD(d.dCd, cd.data(), cd.size() * sizeof(float));
    shim.memcpyHtoD(d.dMat, materials.data(), materials.size() * sizeof(HMaterial));
    shim.memsetD32(d.dCStream, 0, d.contactLimit / 4);
    poisonPatchStream(shim, d);
    shim.memcpyHtoD(d.dCounters, &counters, sizeof(HCounters));
    shim.memcpyHtoD(d.dTouch, &touch, 4);
    shim.memcpyHtoD(d.dPatch, &patch, 4);
    shim.memcpyHtoD(d.dMf, &mf, sizeof(HManifold));
}

static EngineBlob makeBlob(const DeviceScene& d, bool mutate)
{
    EngineBlob b = { 1u, d.dIn, d.dOut, d.dShapes, d.dXf, d.dCd, d.dMat, d.dCStream, d.dPStream, d.dMf, d.dCounters,
                     d.dTouch, d.dPatch, d.dPStream, d.dCStream, d.dForce, d.patchLimit, d.contactLimit, d.forceLimit, 1.0f };
    if (mutate) { const uint64_t s = b.startPoint; b.startPoint = b.startForce; b.startForce = s; }
    return b;
}

static const Q4 kIdentityQ = { 0, 0, 0, 1 };
static const Q4 kPlaneUpQ = { 0, 0, 0.70710678f, 0.70710678f }; // maps local +x -> world +y

// Count / value comparison of the emitted contact stream against the reference
// contact set (order-independent).  Returns the mismatch count.
static size_t compareContacts(PxgCudaShim& shim, const DeviceScene& d, const HOutput& gOut,
                              const std::vector<RefContact>& expect, const V3& worldN, uint32_t matPlane,
                              uint32_t matConvex, const std::vector<HMaterial>& materials, bool quiet, const char* label)
{
    size_t bad = 0;
    const uint32_t nb = uint32_t(expect.size());
    const uint32_t gNb = gOut.nbContacts_flags & 0xffffu;
    if (gNb != nb) { if (!quiet) std::printf("  [%s] nbContacts gpu=%u ref=%u\n", label, gNb, nb); return bad + 1; }
    if ((gOut.nbContacts_flags >> 16) != 0xAAAAu) ++bad; // high flags preserved

    if (nb == 0)
        return bad;

    const uint64_t cpAddr = gOut.contactPoints, patchAddr = gOut.contactPatches, cfAddr = gOut.contactForces;
    if (!inRange(cpAddr, d.dCStream, nb * 16u, d.contactLimit) ||
        !inRange(patchAddr, d.dPStream, 64u, d.patchLimit) ||
        !inRange(cfAddr, d.dForce, nb * 4u, d.forceLimit))
    {
        if (!quiet) std::printf("  [%s] cmOutput pointer out of range\n", label);
        return bad + 1;
    }
    const uint64_t cOff = cpAddr - d.dCStream, pOff = patchAddr - d.dPStream;

    std::vector<uint8_t> cs(nb * 16), ps(64);
    shim.memcpyDtoH(cs.data(), d.dCStream + cOff, nb * 16);
    shim.memcpyDtoH(ps.data(), d.dPStream + pOff, 64);

    // set-match each gpu contact to a reference contact
    std::vector<bool> used(nb, false);
    for (uint32_t k = 0; k < nb; ++k)
    {
        float f4[4]; std::memcpy(f4, &cs[k * 16], 16);
        bool matched = false;
        for (uint32_t j = 0; j < nb && !matched; ++j)
        {
            if (used[j]) continue;
            if (almostEqual(f4[0], expect[j].p.x, 2e-4f, 2e-5f) && almostEqual(f4[1], expect[j].p.y, 2e-4f, 2e-5f) &&
                almostEqual(f4[2], expect[j].p.z, 2e-4f, 2e-5f) && almostEqual(f4[3], expect[j].sep, 2e-4f, 2e-5f))
            { used[j] = true; matched = true; }
        }
        if (!matched) { if (!quiet) std::printf("  [%s] contact %u (%g %g %g | %g) unmatched\n", label, k, f4[0], f4[1], f4[2], f4[3]); ++bad; }
    }

    // patch normal = worldNormal
    float pf[16]; std::memcpy(pf, ps.data(), 64);
    if (!(almostEqual(pf[4], worldN.x, 2e-4f, 2e-5f) && almostEqual(pf[5], worldN.y, 2e-4f, 2e-5f) &&
          almostEqual(pf[6], worldN.z, 2e-4f, 2e-5f)))
    { if (!quiet) std::printf("  [%s] patch normal gpu=(%g %g %g) ref=(%g %g %g)\n", label, pf[4], pf[5], pf[6], worldN.x, worldN.y, worldN.z); ++bad; }
    // ALL SIXTEEN WORDS (P9.87).  insertIntoPatchStream (contactPatchUtils.cuh:161-178):
    // words 0-12 whole, word 13 its LOW half only (:171 PxU16), words 14-15 never --
    // so bytes 54-63 must come back as the per-word poison this slot was uploaded with.
    const uint32_t* pu = reinterpret_cast<const uint32_t*>(pf);
    auto toH = [](const HMaterial& h) {
        return tritriref::MaterialH{ h.dyn, h.sta, h.rest, h.damp, h.packedFlags, h.packedDamp };
    };
    const tritriref::CombinedH cm = tritriref::combine(toH(materials[matPlane & 0xffffu]),
                                                       toH(materials[matConvex & 0xffffu]));
    const uint64_t sw = pOff / 4u;   // first stream word of this slot
    uint32_t exp[16] = {};
    exp[0] = exp[1] = exp[2] = exp[3] = tritriref::bitsOfF(1.0f);
    exp[7] = tritriref::bitsOfF(cm.rest);  exp[8] = tritriref::bitsOfF(cm.dyn);
    exp[9] = tritriref::bitsOfF(cm.sta);   exp[10] = tritriref::bitsOfF(cm.damping);
    exp[11] = ((nb & 0xffu) << 16) | ((cm.flags & 0xffu) << 24);               // startContactIndex = 0
    exp[12] = (matPlane & 0xffffu) << 16;                                      // internalFlags = 0
    exp[13] = (poisonWord(sw + 13) & 0xffff0000u) | (matConvex & 0xffffu);   // pad[0] untouched
    exp[14] = poisonWord(sw + 14); exp[15] = poisonWord(sw + 15);              // pad[1..4] untouched
    for (uint32_t w = 0; w < 16u; ++w)
    {
        if (w >= 4u && w <= 6u) continue;   // the normal: tolerance-checked above
        if (pu[w] != exp[w])
        {
            if (!quiet) std::printf("  [%s] patch word %u gpu=%08x ref=%08x\n", label, w, pu[w], exp[w]);
            ++bad;
        }
    }
    return bad;
}

static std::vector<uint8_t> shimLoader(const char* file) { return loadShaderBlob(file); }

static PxgCudaShim makeShim(PxgSlangComputeDevice* dev)
{
    static const PxgKernelDef manifest[] = { { "npCvxPlaneNphase.spv", "convexPlaneNphase_Kernel", 144u, 0 } };
    return PxgCudaShim(dev, manifest, 1, &shimLoader);
}

// ---------------------------------------------------------------------------
// 1. REGEN + SEPARATED + MUTATION
// ---------------------------------------------------------------------------
static void checkResting(const PlaneScene& s, uint32_t priorAllflags, bool mutate, bool separated, const char* label)
{
    PxgSlangComputeDevice* dev = testDevice();
    PxgCudaShim shim = makeShim(dev);

    HManifold mf; std::memset(&mf, 0, sizeof(mf)); // empty PCM: mNbContacts = 0 -> regen
    DeviceScene d;
    buildDevice(shim, s, mf, priorAllflags, d);
    if (!d.dBlob || !d.dMf) { std::printf("  NOTE: no BDA -> skip\n"); d.free(); return; }

    EngineBlob blob = makeBlob(d, mutate);
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &blob, sizeof(blob)) == PxgResult::Success, "launch (%s)", label);
    shim.streamSynchronize();

    HOutput gOut; HCounters gCnt; uint32_t gTouch, gPatch;
    shim.memcpyDtoH(&gOut, d.dOut, sizeof(HOutput));
    shim.memcpyDtoH(&gCnt, d.dCounters, sizeof(HCounters));
    shim.memcpyDtoH(&gTouch, d.dTouch, 4);
    shim.memcpyDtoH(&gPatch, d.dPatch, 4);

    const std::vector<RefContact> expect = refContacts(s);
    const V3 wn = worldPlaneNormal(s);
    const uint32_t nb = uint32_t(expect.size());
    if (!separated) PXG_CHECK_MSG(nb == 4, "[%s] reference expects 4 resting contacts (got %u)", label, nb);
    else            PXG_CHECK_MSG(nb == 0, "[%s] reference expects 0 contacts (got %u)", label, nb);

    size_t bad = compareContacts(shim, d, gOut, expect, wn, s.matPlane, s.matConvex, makeMaterials(), mutate, label);

    // status word + change flags
    const uint32_t oldStatus = (priorAllflags >> 16) & 0xffu, oldPrevPatches = (priorAllflags >> 8) & 0xffu;
    const bool prevHadTouch = (oldStatus & 2u) != 0u, prevKnown = (oldStatus & 3u) != 0u;
    // registerContactPatch (contactPatchUtils.cuh:81-89): the old status byte survives with
    // eTOUCH_KNOWN cleared, then eHAS_TOUCH or eHAS_NO_TOUCH is set.
    const uint32_t status = (oldStatus & ~3u) | (nb ? 2u : 1u), numPatches = nb ? 1u : 0u;
    const uint32_t expAllflags = (oldPrevPatches << 24) | (status << 16) | (numPatches << 8);
    if (gOut.allflagsWord != expAllflags) { if (!mutate) std::printf("  [%s] allflags gpu=%08x ref=%08x\n", label, gOut.allflagsWord, expAllflags); ++bad; }
    const bool curTouch = nb > 0;
    if (gTouch != (((prevHadTouch != curTouch) || !prevKnown) ? 1u : 0u)) ++bad;
    if (gPatch != ((oldPrevPatches != numPatches) ? 1u : 0u)) ++bad;

    // counters
    if (gCnt.contactsBytes != nb * 16u) ++bad;
    if (gCnt.patchesBytes != (nb ? 64u : 0u)) ++bad;
    if (gCnt.forceAndIndiceBytes != nb * 4u) ++bad;
    if (gCnt.overflowError != 0u) ++bad;

    // separated: NULL cmOutput pointers; resting: manifold populated with mNbContacts
    if (separated)
    {
        if (gOut.contactPoints != 0u || gOut.contactForces != 0u) ++bad;
    }
    else if (!mutate)
    {
        HManifold gMf; shim.memcpyDtoH(&gMf, d.dMf, sizeof(HManifold));
        if (gMf.nbContacts != nb) { std::printf("  [%s] manifold mNbContacts gpu=%u ref=%u\n", label, gMf.nbContacts, nb); ++bad; }
    }

    if (mutate) PXG_CHECK_MSG(bad != 0, "[%s] mutation must FAIL", label);
    else        PXG_CHECK_MSG(bad == 0, "[%s] %zu mismatches", label, bad);

    std::printf("  [%s] contacts=%u overflow=%u %s\n", label, nb, gCnt.overflowError, mutate ? "(mutation ok)" : "");
    d.free();
}

PXG_TEST(convexPlaneNphase_restingBox)
{
    // plane up (+y), unit box resting with 2 cm penetration -> 4 face contacts
    PlaneScene a = { V3{ 1, 1, 1 }, kPlaneUpQ, V3{ 0, 0, 0 }, kIdentityQ, V3{ 0, 0.98f, 0 }, 0.1f, 2, 5 };
    // The prior allflags word is allflagsStart(byte 0) | nbPatches(byte 1) | statusFlag(byte 2)
    // | prevPatches(byte 3) (PxsContactManagerState.h:74-77; the CUDA reads the status with
    // u16Low(u32High(allflags)), contactPatchUtils.cuh:81).  0x00000203 and 0x00000001 put
    // 3 / 1 in byte 0, NOT in the status: the status-byte priors are the three below.
    checkResting(a, 0x00000000u, false, false, "up/prior0");
    checkResting(a, 0x00000203u, false, false, "up/prior203");
    checkResting(a, 0x00020103u, false, false, "up/priorKnownTouch");
    checkResting(a, 0x00010000u, false, false, "up/priorKnownNoTouch");
    checkResting(a, 0x7C4A0200u, false, false, "up/priorHighBits");

    // plane along +x (identity plane quat), box resting along -x
    PlaneScene b = { V3{ 1.2f, 0.8f, 1.0f }, kIdentityQ, V3{ 0, 0, 0 }, kIdentityQ, V3{ 1.18f, 0.3f, -0.2f }, 0.08f, 1, 4 };
    checkResting(b, 0x00000000u, false, false, "x/prior0");

    // slightly translated resting box (offset in the plane) still 4 contacts
    PlaneScene c = { V3{ 1, 1, 1 }, kPlaneUpQ, V3{ 0, 0, 0 }, kIdentityQ, V3{ 0.7f, 0.99f, -0.4f }, 0.1f, 3, 6 };
    checkResting(c, 0x00000001u, false, false, "up/translated");

    // P9.88: material indices with bit 16 set -- the plane's, the convex's, and both.
    PlaneScene h0 = a; h0.matPlane = 0x10000u | a.matPlane;
    checkResting(h0, 0x00000000u, false, false, "up/matPlaneHi");
    PlaneScene h1 = a; h1.matConvex = 0x10000u | a.matConvex;
    checkResting(h1, 0x00000000u, false, false, "up/matConvexHi");
    PlaneScene h2 = c; h2.matPlane |= 0x10000u; h2.matConvex |= 0x10000u;
    checkResting(h2, 0x00000000u, false, false, "up/matBothHi");

    // P9.86c: +-0 material ties.  mi0 is the plane's material (cudaGJKEPA.cu:3180,
    // npCvxPlaneNphase.slang:525), so combine(M[matPlane], M[matConvex]) is the CUDA order.
    // eMIN (-0, +0): CUDA words 7/9/10 = +0, FMin gives -0.  eMAX (+0, -0): CUDA -0, FMax +0.
    PlaneScene zMin = a; zMin.matPlane = 8; zMin.matConvex = 9;
    checkResting(zMin, 0x00000000u, false, false, "up/zeroTieMin");
    PlaneScene zMax = a; zMax.matPlane = 10; zMax.matConvex = 11;
    checkResting(zMax, 0x00000000u, false, false, "up/zeroTieMax");
    {
        const std::vector<HMaterial> M = makeMaterials();
        auto toH = [](const HMaterial& h) {
            return tritriref::MaterialH{ h.dyn, h.sta, h.rest, h.damp, h.packedFlags, h.packedDamp };
        };
        auto sep = [](const tritriref::CombinedH& c, const tritriref::CombinedH& t) {
            using tritriref::bitsOfF;
            return (bitsOfF(c.rest) != bitsOfF(t.rest)) + (bitsOfF(c.sta) != bitsOfF(t.sta)) +
                   (bitsOfF(c.damping) != bitsOfF(t.damping));
        };
        const int dMin = sep(tritriref::combine(toH(M[8]), toH(M[9])), tritriref::combine(toH(M[8]), toH(M[9]), 1));
        const int dMax = sep(tritriref::combine(toH(M[10]), toH(M[11])), tritriref::combine(toH(M[10]), toH(M[11]), 2));
        PXG_CHECK_MSG(dMin == 3, "zeroTieMin: FMin twin differs in %d of rest/sta/damping, expected 3", dMin);
        PXG_CHECK_MSG(dMax == 3, "zeroTieMax: FMax twin differs in %d of rest/sta/damping, expected 3", dMax);
    }
}

PXG_TEST(convexPlaneNphase_separated)
{
    // box lifted clear of contactDistance -> zero contacts, eHAS_NO_TOUCH
    PlaneScene s = { V3{ 1, 1, 1 }, kPlaneUpQ, V3{ 0, 0, 0 }, kIdentityQ, V3{ 0, 1.4f, 0 }, 0.1f, 2, 5 };
    checkResting(s, 0x00000203u, false, true, "separated/prior203");
    checkResting(s, 0x00020103u, false, true, "separated/priorKnownTouch");     // touch lost: change
    checkResting(s, 0x00010000u, false, true, "separated/priorKnownNoTouch");   // still no touch: no change
    checkResting(s, 0x7C4A0200u, false, true, "separated/priorHighBits");
}

PXG_TEST(convexPlaneNphase_mutation)
{
    PlaneScene a = { V3{ 1, 1, 1 }, kPlaneUpQ, V3{ 0, 0, 0 }, kIdentityQ, V3{ 0, 0.98f, 0 }, 0.1f, 2, 5 };
    checkResting(a, 0x00000000u, true, false, "mutation");
}

// ---------------------------------------------------------------------------
// 2. REFRESH / RELOAD across two invocations on the SAME device manifold.
//    Invocation 1 (empty PCM) regenerates + populates the manifold; before
//    invocation 2 the stored contact points mB are PERTURBED — if the kernel
//    took the reload path, the emitted world points follow the perturbed PCM
//    (world = convexTransform.transform(mB)); a fresh regen would ignore mB and
//    re-emit the exact box corners.  We assert the perturbed reload.
// ---------------------------------------------------------------------------
PXG_TEST(convexPlaneNphase_refreshReload)
{
    PxgSlangComputeDevice* dev = testDevice();
    PxgCudaShim shim = makeShim(dev);

    PlaneScene s = { V3{ 1, 1, 1 }, kPlaneUpQ, V3{ 0, 0, 0 }, kIdentityQ, V3{ 0, 0.98f, 0 }, 0.1f, 2, 5 };

    HManifold mf; std::memset(&mf, 0, sizeof(mf)); // empty -> regen on invocation 1
    DeviceScene d;
    buildDevice(shim, s, mf, 0x00000000u, d);
    if (!d.dBlob || !d.dMf) { std::printf("  NOTE: no BDA -> skip\n"); d.free(); return; }

    // ---- invocation 1: regen ----
    EngineBlob blob = makeBlob(d, false);
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &blob, sizeof(blob)) == PxgResult::Success, "reload launch 1");
    shim.streamSynchronize();

    HOutput gOut1; shim.memcpyDtoH(&gOut1, d.dOut, sizeof(HOutput));
    HManifold gMf; shim.memcpyDtoH(&gMf, d.dMf, sizeof(HManifold));
    PXG_CHECK_MSG((gOut1.nbContacts_flags & 0xffffu) == 4u, "reload: invocation 1 must produce 4 contacts (%u)", gOut1.nbContacts_flags & 0xffffu);
    PXG_CHECK_MSG(gMf.nbContacts == 4u, "reload: invocation 1 must populate manifold (mNbContacts=%u)", gMf.nbContacts);

    // ---- perturb the stored convex-space contacts mB (small, keeps refresh) ----
    const float kD = 0.003f;
    std::vector<RefContact> perturbed;
    const V3 wn = worldPlaneNormal(s);
    for (uint32_t j = 0; j < 4; ++j)
    {
        V3 mB = V3{ gMf.lB[j][0] + kD, gMf.lB[j][1] - kD, gMf.lB[j][2] + kD };
        gMf.lB[j][0] = mB.x; gMf.lB[j][1] = mB.y; gMf.lB[j][2] = mB.z;
        // reload world point = convexTransform.transform(mB); sep = refreshed dist
        const V3 wc = xform(s.convexQ, s.convexP, mB);
        const float sep = dot3(sub(wc, s.planeP), wn); // plane-space x of the reloaded point
        perturbed.push_back({ wc, sep });
    }
    shim.memcpyHtoD(d.dMf, &gMf, sizeof(HManifold));

    // reset the output-side buffers (counters/stream/cmOutput/flags) but keep the manifold
    HOutput out; std::memset(&out, 0, sizeof(out)); out.nbContacts_flags = 0xAAAA0000u;
    HCounters counters = { 0, 0, 0, 0 };
    uint32_t seven = 7u;
    shim.memcpyHtoD(d.dOut, &out, sizeof(HOutput));
    shim.memcpyHtoD(d.dCounters, &counters, sizeof(HCounters));
    shim.memsetD32(d.dCStream, 0, d.contactLimit / 4);
    poisonPatchStream(shim, d);   // invocation 2's pad expectation must not depend on invocation 1
    shim.memcpyHtoD(d.dTouch, &seven, 4);
    shim.memcpyHtoD(d.dPatch, &seven, 4);

    // ---- invocation 2: refresh -> reload (must reuse the perturbed PCM) ----
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &blob, sizeof(blob)) == PxgResult::Success, "reload launch 2");
    shim.streamSynchronize();

    HOutput gOut2; shim.memcpyDtoH(&gOut2, d.dOut, sizeof(HOutput));
    size_t bad = compareContacts(shim, d, gOut2, perturbed, wn, s.matPlane, s.matConvex, makeMaterials(), false, "reload");
    PXG_CHECK_MSG(bad == 0, "reload: invocation 2 must reuse the perturbed PCM (%zu mismatches)", bad);

    std::printf("  [reload] two invocations: regen then refresh/reload of perturbed PCM ok\n");
    d.free();
}

// ---------------------------------------------------------------------------
// 3. P9.90: DOT ORDER, BITWISE.  The kernel's float OpDots transcribe PxVec3
//    dots, x*v.x + y*v.y + z*v.z = (x + y) + z (PxVec3.h:173/:274).  Two of
//    them reach an output directly:
//      - the working normal n = bToA.rotateInv(localNormal).getNormalized()
//        (cudaGJKEPA.cu:2863): magnitudeSquared feeds 1/sqrt, and n is stored
//        in every manifold localNormalPen.xyz on a regen;
//      - the refresh distance dist = v.dot(n) (:2905), emitted as the contact
//        separation on the reload path.
//    llvmpipe evaluates an UNDECORATED OpDot as x + (y + z) (Mesa lower_fdot);
//    the NoContraction stamp (PXG_NOCONTRACT_SHADERS) gives (x + y) + z.  The
//    existing cases above are axis-aligned and tolerance-compared, so they
//    cannot see the difference.  Here every pose is general (random plane
//    rotation, tilted box, non-round extents and offsets), and the manifold
//    and contact stream are compared word for word against a
//    -ffp-contract=off mirror of the kernel in the CUDA order.  A twin in the
//    old order must differ on both sites (coverage guard).
// ---------------------------------------------------------------------------
static Q4 qMulH(Q4 a, Q4 b)   // PxQuat operator* == npAbiQuatMul
{
    return Q4{ a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y,
               a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z,
               a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x,
               a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}
static V3 qRotInvH(Q4 q, V3 v) // npAbiQuatRotateInv
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z, w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz;
    return V3{ vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2,
               vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2,
               vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2 };
}
// cudaOrder: (x + y) + z, the CUDA's PxVec3::dot.  !cudaOrder: x + (y + z), the
// unstamped llvmpipe OpDot.
static float dotO(V3 a, V3 b, bool cudaOrder)
{
    const float x = a.x * b.x, y = a.y * b.y, z = a.z * b.z;
    return cudaOrder ? (x + y) + z : x + (y + z);
}
static V3 normO(V3 v, bool cudaOrder)   // getNormalized (npCvxPlaneNphase.slang:139)
{
    const float m = dotO(v, v, cudaOrder);
    if (!(m > 0.0f)) return V3{ 0, 0, 0 };
    const float r = 1.0f / std::sqrt(m);
    return V3{ v.x * r, v.y * r, v.z * r };
}
static uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

PXG_TEST(convexPlaneNphase_p90DotOrder)
{
    PxgSlangComputeDevice* dev = testDevice();
    PxgCudaShim shim = makeShim(dev);

    uint32_t rng = 0x9E3779B9u;
    auto rnd = [&]() { rng = rng * 1664525u + 1013904223u; return float(rng >> 8) * (1.0f / 16777216.0f); }; // [0,1)
    auto rs = [&](float lo, float hi) { return lo + (hi - lo) * rnd(); };

    const int kScenes = 16;
    int badRegen = 0, badReload = 0, twinN = 0, twinDist = 0, ran = 0;
    for (int sc = 0; sc < kScenes; ++sc)
    {
        // ---- a general pose ----
        Q4 pq = { rs(-1, 1), rs(-1, 1), rs(-1, 1), rs(-1, 1) };
        {
            const float l = std::sqrt(pq.x * pq.x + pq.y * pq.y + pq.z * pq.z + pq.w * pq.w);
            pq = Q4{ pq.x / l, pq.y / l, pq.z / l, pq.w / l };
        }
        V3 axis = { rs(-1, 1), rs(-1, 1), rs(-1, 1) };
        {
            const float l = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
            axis = V3{ axis.x / l, axis.y / l, axis.z / l };
        }
        const float ang = rs(0.013f, 0.057f), sh = std::sin(0.5f * ang);
        const Q4 tilt = { axis.x * sh, axis.y * sh, axis.z * sh, std::cos(0.5f * ang) };
        const V3 h = { rs(0.31f, 0.77f), rs(0.31f, 0.77f), rs(0.31f, 0.77f) };
        // plane-space placement: the tilted box's lowest corner 1.3 cm below the plane
        float minX = 1e30f;
        for (int i = 0; i < 8; ++i) minX = std::fmin(minX, qRot(tilt, boxCorner(i, h)).x);
        const V3 pDesign = { -minX - rs(0.011f, 0.017f), rs(-0.9f, 0.9f), rs(-0.9f, 0.9f) };
        PlaneScene s;
        s.halfExt = h;
        s.planeQ = pq;
        s.planeP = V3{ rs(-2, 2), rs(-2, 2), rs(-2, 2) };
        s.convexQ = qMulH(pq, tilt);
        s.convexP = add(s.planeP, qRot(pq, pDesign));
        s.contactDist = 0.2f;
        s.matPlane = 2; s.matConvex = 5;

        HManifold mf; std::memset(&mf, 0, sizeof(mf));
        DeviceScene d;
        buildDevice(shim, s, mf, 0u, d);
        if (!d.dBlob || !d.dMf) { std::printf("  NOTE: no BDA -> skip\n"); d.free(); return; }
        ++ran;

        // ---- mirror: refresh inputs (cudaGJKEPA.cu:2859-2863) ----
        const Q4 bToAq = qMulH(Q4{ -pq.x, -pq.y, -pq.z, pq.w }, s.convexQ);
        const V3 bToAp = qRotInvH(pq, sub(s.convexP, s.planeP));
        const V3 nRaw = qRotInvH(bToAq, V3{ 1, 0, 0 });
        const V3 n = normO(nRaw, true), nOld = normO(nRaw, false);
        const float contactDist = 0.5f * s.contactDist + 0.5f * s.contactDist;

        // ---- invocation 1: regen.  Face 1 (-x) is the support face; its four
        //      corners in vertexData8 order (kBoxFaceVtx[1]) are the four lanes. ----
        EngineBlob blob = makeBlob(d, false);
        PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &blob, sizeof(blob)) == PxgResult::Success, "p90 launch 1 (scene %d)", sc);
        shim.streamSynchronize();
        HOutput gOut1; shim.memcpyDtoH(&gOut1, d.dOut, sizeof(HOutput));
        HManifold gMf; shim.memcpyDtoH(&gMf, d.dMf, sizeof(HManifold));

        HManifold ref; std::memcpy(&ref, &mf, sizeof(mf));
        V3 worldRef[4]; float sepRef[4];
        uint32_t nbRef = 0;
        for (int L = 0; L < 4; ++L)
        {
            const V3 vtx = boxCorner(kBoxFaceVtx[1][L], h);
            const Q4 id = { 0, 0, 0, 1 };
            const V3 r = qRot(id, vtx);
            const V3 pS = qRotInvH(id, V3{ r.x * 1.0f, r.y * 1.0f, r.z * 1.0f });   // vertex2Shape, identity scale
            const V3 p = add(qRot(bToAq, pS), bToAp);
            if (!(p.x < contactDist)) continue;
            const V3 pInB = qRotInvH(bToAq, sub(p, bToAp));
            ref.lB[nbRef][0] = pInB.x; ref.lB[nbRef][1] = pInB.y; ref.lB[nbRef][2] = pInB.z; ref.lB[nbRef][3] = 0.0f;
            ref.lA[nbRef][0] = p.x - p.x; ref.lA[nbRef][1] = p.y; ref.lA[nbRef][2] = p.z; ref.lA[nbRef][3] = 0.0f;
            ref.lNp[nbRef][0] = n.x; ref.lNp[nbRef][1] = n.y; ref.lNp[nbRef][2] = n.z; ref.lNp[nbRef][3] = p.x;
            worldRef[nbRef] = add(qRot(pq, p), s.planeP); sepRef[nbRef] = p.x;
            ++nbRef;
        }
        PXG_CHECK_MSG(nbRef == 4u, "p90 scene %d: the mirror expects 4 face contacts (got %u)", sc, nbRef);
        ref.relPos[0] = -bToAp.x; ref.relPos[1] = -bToAp.y; ref.relPos[2] = -bToAp.z; ref.relPos[3] = 0.0f;
        std::memcpy(ref.quatA, &pq, 16); std::memcpy(ref.quatB, &s.convexQ, 16);
        ref.nbContacts = nbRef;

        if (n.x != nOld.x || n.y != nOld.y || n.z != nOld.z) ++twinN;

        PXG_CHECK_MSG((gOut1.nbContacts_flags & 0xffffu) == nbRef, "p90 scene %d: regen nb gpu=%u ref=%u",
                      sc, gOut1.nbContacts_flags & 0xffffu, nbRef);
        {
            const uint32_t* gw = reinterpret_cast<const uint32_t*>(&gMf);
            const uint32_t* rw = reinterpret_cast<const uint32_t*>(&ref);
            int bad = 0;
            for (int w = 0; w < 61; ++w)   // through nbContacts; the warm-start words are untouched zeros
                if (gw[w] != rw[w])
                {
                    if (bad < 4) std::printf("  [p90 %d] manifold word %d gpu=%08x ref=%08x\n", sc, w, gw[w], rw[w]);
                    ++bad;
                }
            std::vector<float> cs(4u * nbRef);
            shim.memcpyDtoH(cs.data(), d.dCStream + (gOut1.contactPoints - d.dCStream), cs.size() * 4);
            for (uint32_t k = 0; k < nbRef; ++k)
            {
                const float e[4] = { worldRef[k].x, worldRef[k].y, worldRef[k].z, sepRef[k] };
                for (int c = 0; c < 4; ++c)
                    if (fbits(cs[4 * k + c]) != fbits(e[c]))
                    {
                        if (bad < 4) std::printf("  [p90 %d] regen contact %u.%d gpu=%08x ref=%08x\n", sc, k, c, fbits(cs[4 * k + c]), fbits(e[c]));
                        ++bad;
                    }
            }
            if (bad) ++badRegen;
        }

        // ---- invocation 2: same poses, stored mB perturbed off the corners so v is
        //      a general vector -> refresh keeps all four, no rerun, reload path. ----
        const float kP[4][3] = { { 0.0031f, -0.0027f, 0.0019f }, { -0.0023f, 0.0037f, -0.0011f },
                                 { 0.0013f, 0.0029f, -0.0033f }, { -0.0041f, -0.0017f, 0.0026f } };
        for (int j = 0; j < 4; ++j)
            for (int c = 0; c < 3; ++c) gMf.lB[j][c] += kP[j][c];
        shim.memcpyHtoD(d.dMf, &gMf, sizeof(HManifold));
        HOutput out; std::memset(&out, 0, sizeof(out)); out.nbContacts_flags = 0xAAAA0000u;
        HCounters counters = { 0, 0, 0, 0 };
        shim.memcpyHtoD(d.dOut, &out, sizeof(HOutput));
        shim.memcpyHtoD(d.dCounters, &counters, sizeof(HCounters));
        shim.memsetD32(d.dCStream, 0, d.contactLimit / 4);
        PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &blob, sizeof(blob)) == PxgResult::Success, "p90 launch 2 (scene %d)", sc);
        shim.streamSynchronize();
        HOutput gOut2; shim.memcpyDtoH(&gOut2, d.dOut, sizeof(HOutput));
        PXG_CHECK_MSG((gOut2.nbContacts_flags & 0xffffu) == 4u, "p90 scene %d: reload nb gpu=%u, expected 4",
                      sc, gOut2.nbContacts_flags & 0xffffu);
        {
            std::vector<float> cs(16);
            shim.memcpyDtoH(cs.data(), d.dCStream + (gOut2.contactPoints - d.dCStream), 64);
            int bad = 0;
            bool twinDiff = false;
            for (int j = 0; j < 4; ++j)
            {
                const V3 pA = { gMf.lA[j][0], gMf.lA[j][1], gMf.lA[j][2] };
                const V3 pB = { gMf.lB[j][0], gMf.lB[j][1], gMf.lB[j][2] };
                const V3 v = sub(pB, qRotInvH(bToAq, sub(pA, bToAp)));   // :2902-2904
                const float dist = dotO(v, n, true);                      // :2905
                const float distOld = dotO(v, nOld, false);
                if (fbits(dist) != fbits(distOld)) twinDiff = true;
                const V3 w = add(qRot(pq, add(qRot(bToAq, pB), bToAp)), s.planeP);
                const float e[4] = { w.x, w.y, w.z, dist };
                for (int c = 0; c < 4; ++c)
                    if (fbits(cs[4 * j + c]) != fbits(e[c]))
                    {
                        if (bad < 4) std::printf("  [p90 %d] reload contact %d.%d gpu=%08x (%.9g) ref=%08x (%.9g) old-order=%08x\n",
                                                 sc, j, c, fbits(cs[4 * j + c]), cs[4 * j + c], fbits(e[c]), e[c], c == 3 ? fbits(distOld) : 0u);
                        ++bad;
                    }
            }
            if (twinDiff) ++twinDist;
            if (bad) ++badReload;
        }
        d.free();
    }
    if (ran == 0) return;
    std::printf("  [p90] %d scenes: regen mismatches in %d, reload mismatches in %d; old-order twin moves n in %d, dist in %d\n",
                ran, badRegen, badReload, twinN, twinDist);
    PXG_CHECK_MSG(badRegen == 0, "p90: %d of %d regen manifolds/streams differ from the CUDA-order mirror", badRegen, ran);
    PXG_CHECK_MSG(badReload == 0, "p90: %d of %d reload streams differ from the CUDA-order mirror", badReload, ran);
    PXG_CHECK_MSG(twinN > 0, "p90: the x+(y+z) twin must change the working normal n in some scene (coverage)");
    PXG_CHECK_MSG(twinDist > 0, "p90: the x+(y+z) twin must change a reloaded separation in some scene (coverage)");
}

} // namespace
