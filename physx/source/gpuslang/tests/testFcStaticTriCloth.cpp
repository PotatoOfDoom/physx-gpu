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

// Verification of fcStaticTriCloth.slang -- a harness around the ENGINE's
// pxgSlangClothTriContact / pxgSlangSphereBox / pxgSlangDefFilter /
// pxgSlangDeformableWriters modules -- against a CPU transcription of
// femClothPrimitives.cu (PhysX 5.12).  Mode 0 (cloth_meshContactGenLaunch
// core): the (rigid filter ref, cloth TRIANGLE) filter search, vertex2Shape +
// trimeshToClothTransform, staticTriangleVsClothCollision (trim pipeline +
// tolerance-averaged deepest point) and PxgRigidDbContactWriter::
// writeSurfaceContact (pair info, PxU16 material, barycentric, the u64 rigid
// sort key).  Mode 1 (cloth_boxVertexContactGenLaunch core): the (rigid filter
// ref, cloth VERTEX) filter search, spherebox with radius 0, contact = clothP -
// pen*n, and the vertex overload of writeSurfaceContact, which writes NO
// barycentric.  The filter references come from the engine's host helpers
// (PxgEncodeRigidFilterRef keys a STATIC rigid on its transform cache id), and
// the near misses include the 5.6.1 keys -- all three triangle VERTICES, the raw
// static node index -- which must no longer filter anything.  Both modes serialize a
// 32-case warp batch per thread with ONE reservation per batch: batches
// whose cases are all untainted must occupy consecutive ascending slots.
// A small-capacity re-dispatch checks the capacity gate (reservation
// ungated, stores dropped).  Slack taints: trim() strict outside tests
// near zero, intersectionPoint conditioning, the deepest-point +-tol
// averaging band edges, the sep < cDistance gate (note: for a NON-EMPTY
// clipped loop the +cDistance slab already enforces sep <= cDistance, so
// the gate only rejects exact equality — rejects happen via the
// clipped-empty 1e9 path), spherebox's inflation gate, inside-box
// membership boundaries and the min-face tie chain.

#include "pxgTestFramework.h"

// no-CUDA-toolkit stand-in for the __device__/__align__ qualifiers; must precede the
// engine headers.
#include "host_defines.h"
#include "PxNodeIndex.h"
#include "PxgSimulationCoreDesc.h"   // PxgDeformablePairInfo, PxgEncode*, PxgComparePairInfo

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct FstCase { F4 triA, triB, triC, clothQ, clothP, meshQ, meshP, scale, scaleRot; U4 ids, ids2, ids3; };
// Byte-identical to fcStaticTriCloth.slang's FstParams: the filter list and the
// PxgRigidDbContactWriter streams are device addresses.
struct FstParams
{
    uint32_t numCases, maxContacts, nbFilterPairs, mode;
    uint64_t filterPairs, filterSafeAddr;
    uint64_t outPoint, outNormalPen, outBarycentric, outContactInfo, outRigidMaterialIndex;
    uint64_t totalContactCount, contactByRigid, tempContactByRigid, contactIndexSortedByRigid;
};
static_assert(sizeof(FstParams) == 104, "fcStaticTriCloth push block");
static_assert(sizeof(PxgDeformablePairInfo) == 16, "PxgDeformablePairInfo");

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static V3 mulv(V3 a, V3 b) { return { a.x * b.x, a.y * b.y, a.z * b.z }; }
static V3 divv(V3 a, V3 b) { return { a.x / b.x, a.y / b.y, a.z / b.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 crossp(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 v3f(F4 f) { return { f.x, f.y, f.z }; }
static V3 getNorm0(V3 v) { const float m = dot3(v, v); return m > 0 ? mul(v, 1.0f / std::sqrt(m)) : V3{ 0, 0, 0 }; }

static V3 quatRotate(F4 q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d = q.x * vx + q.y * vy + q.z * vz;
    return { vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d,
             vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d,
             vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d };
}
static V3 quatRotateInv(F4 q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d = q.x * vx + q.y * vy + q.z * vz;
    return { vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d,
             vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d,
             vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d };
}
static F4 quatMul(F4 a, F4 b)
{
    return { a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
             a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
             a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
             a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}
static F4 quatConj(F4 q) { return { -q.x, -q.y, -q.z, q.w }; }
static V3 vertex2Shape(V3 v, V3 scale, F4 rot) { return quatRotateInv(rot, mulv(quatRotate(rot, v), scale)); }
static V3 shape2Vertex(V3 v, V3 scale, F4 rot) { return quatRotateInv(rot, divv(quatRotate(rot, v), scale)); }

struct Slack
{
    bool near = false;
    // exact zero is exempt: it only arises from exactly-degenerate inputs
    // (zero plane normals from duplicated vertices), where both sides
    // compute bit-identical zeros — a genuinely near-boundary value being
    // exactly 0.0f is measure-zero
    void note(float margin, float eps) { if (margin != 0.0f && std::fabs(margin) < eps) near = true; }
};

// ---- trim pipeline reference (femClothPrimitives.cu 3436-3641) --------------
static F4 makePlane(V3 p, V3 n) { return { n.x, n.y, n.z, -dot3(p, n) }; }
static float planeDist(F4 pl, V3 p) { return pl.x * p.x + pl.y * p.y + pl.z * p.z + pl.w; }

static const float TRIM_EPS = 1e-4f;
static const float BAND_EPS = 1e-4f;

static V3 intersectionPointRef(F4 plane, V3 a, V3 b, Slack& sl)
{
    const float distA = std::fabs(planeDist(plane, a));
    const float distB = std::fabs(planeDist(plane, b));
    sl.note(distA + distB, 1e-6f); // both-on-plane conditioning
    const float f = distA / (distA + distB);
    return add(mul(a, 1.0f - f), mul(b, f));
}

static void insertPtRef(V3* arr, int arrCount, int index, V3 element)
{
    for (int i = arrCount; i > index; --i)
        arr[i] = arr[i - 1];
    arr[index] = element;
}

static int trimRef(F4 plane, V3* loop, int loopCount, Slack& sl)
{
    if (loopCount < 3)
        return loopCount;

    for (int i = 0; i < loopCount; ++i)
        sl.note(planeDist(plane, loop[i]), TRIM_EPS); // strict > 0 membership

    V3 intersectionA = { 0, 0, 0 };
    int changeA = -1;
    V3 intersectionB = { 0, 0, 0 };
    int changeB = -1;

    bool keep = false;
    bool prevOutside = planeDist(plane, loop[0]) > 0.0f;
    for (int i = 0; i < loopCount; ++i)
    {
        const bool outside = planeDist(plane, loop[(i + 1) % loopCount]) > 0.0f;
        if (outside != prevOutside)
        {
            const V3 inter = intersectionPointRef(plane, loop[i], loop[(i + 1) % loopCount], sl);
            if (changeA < 0) { changeA = i; keep = !prevOutside; intersectionA = inter; }
            else { changeB = i; intersectionB = inter; }
        }
        prevOutside = outside;
    }

    if (changeA >= 0 && changeB >= 0)
    {
        int loopIndexer = -1;
        for (int i = 0; i < loopCount; ++i)
        {
            if (keep)
                loop[++loopIndexer] = loop[i];
            if (i == changeA || i == changeB)
            {
                const V3 pt = (i == changeA) ? intersectionA : intersectionB;
                if (loopIndexer == i && !keep)
                {
                    insertPtRef(loop, loopCount, ++loopIndexer, pt);
                    ++loopCount;
                    ++i;
                    ++changeB;
                }
                else
                    loop[++loopIndexer] = pt;
                keep = !keep;
            }
        }
        loopCount = loopIndexer + 1;
    }
    else if (prevOutside)
        loopCount = 0;

    return loopCount;
}

static F4 createEdgePlaneRef(V3 triNormal, V3 e0, V3 e1)
{
    const V3 n = getNorm0(crossp(sub(e1, e0), triNormal));
    return makePlane(e0, n);
}

static int trimTriangleByTriangleRef(V3* poly, int polyLength, V3 triA, V3 triB, V3 triC,
                                     float maxPos, float maxNeg, Slack& sl)
{
    V3 n = crossp(sub(triB, triA), sub(triC, triA));
    const float mag = std::sqrt(dot3(n, n));
    if (mag >= 1e-20f) n = mul(n, 1.0f / mag); // normalizeSafe: unchanged below eps

    polyLength = trimRef(createEdgePlaneRef(n, triA, triB), poly, polyLength, sl);
    polyLength = trimRef(createEdgePlaneRef(n, triB, triC), poly, polyLength, sl);
    polyLength = trimRef(createEdgePlaneRef(n, triC, triA), poly, polyLength, sl);
    polyLength = trimRef(makePlane(add(triA, mul(n, maxPos)), n), poly, polyLength, sl);
    polyLength = trimRef(makePlane(sub(triA, mul(n, maxNeg)), { -n.x, -n.y, -n.z }), poly, polyLength, sl);
    return polyLength;
}

static V3 deepestRef(const V3* poly, int polyLength, F4 plane, float tolerance, float& minDist, Slack& sl)
{
    minDist = 1000000000.0f;
    V3 deepestPoint = { 0, 0, 0 };
    int count = 0;
    for (int i = 0; i < polyLength; ++i)
    {
        const float d = planeDist(plane, poly[i]);
        if (i > 0) // first point always enters the strict branch (1e9 margins)
        {
            sl.note(d - (minDist - tolerance), BAND_EPS);
            sl.note(d - (minDist + tolerance), BAND_EPS);
        }
        if (d < minDist - tolerance)
        {
            minDist = d;
            deepestPoint = poly[i];
            count = 1;
        }
        else if (d < minDist + tolerance)
        {
            deepestPoint = add(deepestPoint, poly[i]);
            count++;
        }
    }
    if (count > 0)
        deepestPoint = mul(deepestPoint, 1.0f / float(count));
    return deepestPoint;
}

static void baryRef(V3 a, V3 b, V3 c, V3 p, F4& bary, Slack& sl)
{
    const V3 v0 = sub(b, a), v1 = sub(c, a), v2 = sub(p, a);
    const float d00 = dot3(v0, v0), d01 = dot3(v0, v1), d11 = dot3(v1, v1);
    const float d20 = dot3(v2, v0), d21 = dot3(v2, v1);
    const float denom = d00 * d11 - d01 * d01;
    sl.note(denom, 1e-8f);
    const float v = (d11 * d20 - d01 * d21) / denom;
    const float w = (d00 * d21 - d01 * d20) / denom;
    bary = { 1.0f - v - w, v, w, 0.0f };
}

// ---- filter table (5.12: one sorted PxgDeformablePairInfo list per cloth) ------
static uint64_t rigid64(uint32_t lo, uint32_t hi) { return (uint64_t(hi) << 32) | lo; }
// PxgEncodeRigidFilterRef: a body by its node, a STATIC (low word PX_INVALID_NODE) by
// its shape's transform cache id.
static uint64_t filterKey(uint64_t rigid, uint32_t cacheRef)
{
    return PxgEncodeRigidFilterRef(PxNodeIndex(PxU64(rigid)), cacheRef);
}
static bool pairLess(const PxgDeformablePairInfo& a, const PxgDeformablePairInfo& b)
{
    return PxgComparePairInfo(a, b) < 0;
}
static bool findRef(const std::vector<PxgDeformablePairInfo>& t, uint64_t ref0, uint64_t ref1)
{
    const PxgDeformablePairInfo key = { ref0, ref1 };
    return std::binary_search(t.begin(), t.end(), key, pairLess);
}

static F4 randQuat(std::mt19937& rng)
{
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    F4 q = { u(rng), u(rng), u(rng), u(rng) + 1.5f };
    const float m = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return { q.x / m, q.y / m, q.z / m, q.w / m };
}

struct RefOut
{
    bool filtered = false, wrote = false;
    V3 p = {}, n = {};
    float sep = 0;
    F4 bary = {};
};

// P9.94a mutant twin: true selects the port's PRE-FIX trimeshToClothTransform.p spelling
// clothQ.rotateInv(v) instead of the CUDA's transformInv(PxTransform) = q.getConjugate().rotate(v).
// Used only to prove that fcStaticTriCloth_p94ZeroSign separates the two forms.
static bool gP94TinvRotInv = false;

// P9.94a: mode-0 static-triangle chain (femClothPrimitives.cu:3876 transform, then
// staticTriangleVsClothCollision up to the sep gate), shared by the random fixture and the
// directed zero-sign test.  poly is in/out (cloth triangle in, clipped loop out).
struct Mode0Chain { V3 fixNormal; F4 triPlane; int corners; V3 dp; float sep; };
static Mode0Chain refMode0Chain(const FstCase& cs, V3* poly, Slack& sl)
{
    // trimeshToClothTransform = clothT.transformInv(meshT) (PxTransform.h:162-169 LITERALLY:
    // qinv = q.getConjugate(); qinv.rotate(src.p - p), not q.rotateInv(src.p - p))
    const F4 clothQInv = quatConj(cs.clothQ);
    const F4 mtcQ = quatMul(clothQInv, cs.meshQ);
    const V3 mtcV = sub(v3f(cs.meshP), v3f(cs.clothP));
    const V3 mtcP = gP94TinvRotInv ? quatRotateInv(cs.clothQ, mtcV) : quatRotate(clothQInv, mtcV);
    const V3 lA = add(quatRotate(mtcQ, vertex2Shape(v3f(cs.triA), v3f(cs.scale), cs.scaleRot)), mtcP);
    const V3 lB = add(quatRotate(mtcQ, vertex2Shape(v3f(cs.triB), v3f(cs.scale), cs.scaleRot)), mtcP);
    const V3 lC = add(quatRotate(mtcQ, vertex2Shape(v3f(cs.triC), v3f(cs.scale), cs.scaleRot)), mtcP);
    const float cd = cs.clothP.w;

    Mode0Chain m;
    m.fixNormal = getNorm0(crossp(sub(lB, lA), sub(lC, lA)));
    m.triPlane = makePlane(lA, m.fixNormal);
    m.corners = trimTriangleByTriangleRef(poly, 3, lA, lB, lC, cd, cd, sl);
    m.dp = deepestRef(poly, m.corners, m.triPlane, 0.01f * cd, m.sep, sl);
    return m;
}

static const uint32_t POISON = 0xCDCDCDCDu;
static const uint64_t INFOPOISON = 0xDEADBEEFDEADBEEFull;
static const uint32_t MATPOISON = 0xFFFFu;
static const uint32_t CLOTH0 = 3, CLOTH1 = 5;

// Everything one dispatch wrote through the PxgRigidDbContactWriter.
struct Run
{
    std::vector<F4> pt, np, bc, dbg;
    std::vector<PxgDeformablePairInfo> in;
    std::vector<uint64_t> by;
    std::vector<uint32_t> tb, ix, mat;
    uint32_t wc;
};
static Run runFst(PxgSlangComputeDevice* dev, const Kernel& k, const std::vector<FstCase>& cc,
                  const std::vector<F4>& cp, const std::vector<PxgDeformablePairInfo>& filters,
                  uint32_t mode, uint32_t cap, uint32_t bufCap)
{
    const uint32_t n = (uint32_t)cc.size();
    std::vector<F4> femF4Init(bufCap, { -9, -9, -9, -9 });
    std::vector<PxgDeformablePairInfo> infoInit(bufCap, PxgDeformablePairInfo{ INFOPOISON, INFOPOISON });
    std::vector<uint32_t> u1Init(bufCap, POISON), matInit((bufCap + 1) / 2, 0xFFFFFFFFu), wc0(1, 0);
    std::vector<uint64_t> u64Init(bufCap, (uint64_t(POISON) << 32) | POISON);
    std::vector<F4> dbgInit(n, { -7, -7, -7, -7 });
    // An empty list still needs a mapped address: one dummy entry, nbFilterPairs = 0.
    const std::vector<PxgDeformablePairInfo> fl =
        filters.empty() ? std::vector<PxgDeformablePairInfo>(1, PxgDeformablePairInfo{ 0, 0 }) : filters;
    PxgBufferHandle bC = createBufferFrom(dev, cc);
    PxgBufferHandle bP = createBufferFrom(dev, cp);
    PxgBufferHandle bDb = createBufferFrom(dev, dbgInit);
    PxgBufferHandle bF = createBufferFrom(dev, fl);
    PxgBufferHandle bPt = createBufferFrom(dev, femF4Init);
    PxgBufferHandle bNp = createBufferFrom(dev, femF4Init);
    PxgBufferHandle bBc = createBufferFrom(dev, femF4Init);
    PxgBufferHandle bIn = createBufferFrom(dev, infoInit);
    PxgBufferHandle bMat = createBufferFrom(dev, matInit);
    PxgBufferHandle bWc = createBufferFrom(dev, wc0);
    PxgBufferHandle bBy = createBufferFrom(dev, u64Init);
    PxgBufferHandle bTb = createBufferFrom(dev, u1Init);
    PxgBufferHandle bIx = createBufferFrom(dev, u1Init);
    FstParams prm = {};
    prm.numCases = n;
    prm.maxContacts = cap;
    prm.nbFilterPairs = (uint32_t)filters.size();
    prm.mode = mode;
    prm.filterPairs = dev->getBufferDeviceAddress(bF);
    prm.filterSafeAddr = prm.filterPairs;
    prm.outPoint = dev->getBufferDeviceAddress(bPt);
    prm.outNormalPen = dev->getBufferDeviceAddress(bNp);
    prm.outBarycentric = dev->getBufferDeviceAddress(bBc);
    prm.outContactInfo = dev->getBufferDeviceAddress(bIn);
    prm.outRigidMaterialIndex = dev->getBufferDeviceAddress(bMat);
    prm.totalContactCount = dev->getBufferDeviceAddress(bWc);
    prm.contactByRigid = dev->getBufferDeviceAddress(bBy);
    prm.tempContactByRigid = dev->getBufferDeviceAddress(bTb);
    prm.contactIndexSortedByRigid = dev->getBufferDeviceAddress(bIx);
    const uint32_t batches = (n + 31) / 32;
    k.dispatch((batches + 63) / 64, { bC, bP, bDb }, prm);
    dev->flush();
    Run r;
    r.pt = downloadBuffer<F4>(dev, bPt, bufCap);
    r.np = downloadBuffer<F4>(dev, bNp, bufCap);
    r.bc = downloadBuffer<F4>(dev, bBc, bufCap);
    r.in = downloadBuffer<PxgDeformablePairInfo>(dev, bIn, bufCap);
    r.by = downloadBuffer<uint64_t>(dev, bBy, bufCap);
    r.tb = downloadBuffer<uint32_t>(dev, bTb, bufCap);
    r.ix = downloadBuffer<uint32_t>(dev, bIx, bufCap);
    const std::vector<uint32_t> matWords = downloadBuffer<uint32_t>(dev, bMat, (bufCap + 1) / 2);
    for (uint32_t s = 0; s < bufCap; ++s)
        r.mat.push_back((matWords[s / 2] >> (16u * (s & 1u))) & 0xFFFFu);
    r.dbg = downloadBuffer<F4>(dev, bDb, n);
    r.wc = downloadBuffer<uint32_t>(dev, bWc, 1)[0];
    for (PxgBufferHandle h : { bC, bP, bDb, bF, bPt, bNp, bBc, bIn, bMat, bWc, bBy, bTb, bIx })
        dev->destroyBuffer(h);
    return r;
}

PXG_TEST(fcStaticTriCloth_meshBoxVertex)
{
    const uint32_t NC = 1024, CAP = 1400, TINY = 48;
    std::mt19937 rng(20260710);
    std::uniform_real_distribution<float> uf(0.0f, 1.0f);

    // ================= MODE 0: static mesh triangle vs cloth triangle =========
    std::vector<FstCase> cases(NC);
    std::vector<F4> clothPos(NC * 3);
    std::vector<PxgDeformablePairInfo> filters;

    for (uint32_t i = 0; i < NC; ++i)
    {
        FstCase& cs = cases[i];
        const float cd = 0.15f + 0.25f * uf(rng);

        // static triangle in a canonical cloth-space frame (rotated+shifted below)
        const V3 SA = { -1.1f - 0.3f * uf(rng), -0.9f - 0.3f * uf(rng), 0.0f };
        V3 SB = { 1.2f + 0.3f * uf(rng), -1.0f - 0.3f * uf(rng), 0.0f };
        const V3 SC = { 0.05f + 0.2f * uf(rng), 1.3f + 0.3f * uf(rng), 0.0f };
        // degenerate family: duplicated static vertex => cross products are
        // exactly zero end-to-end (identical inputs stay bit-identical through
        // the transform chain), all trim planes are zero planes, sep = 0, and
        // the kernel emits a contact with a ZERO normal at the cloth-triangle
        // centroid — real upstream behaviour that must be reproduced
        const bool degenerate = (i % 32 == 17);
        if (degenerate) SB = SA;
        const V3 cen = mul(add(add(SA, SB), SC), 1.0f / 3.0f);

        // cloth triangle in the same frame: lateral (x,y) + height z per family
        V3 CA, CB, CC;
        const uint32_t fam = i % 8;
        float rad = 0.32f + 0.1f * uf(rng);
        float baseH = 0.3f * cd;
        float tiltB = 0.0f, tiltC = 0.0f;
        V3 latOfs = { 0.25f * (uf(rng) - 0.5f), 0.25f * (uf(rng) - 0.5f), 0 };
        switch (fam)
        {
        case 0: break; // parallel hover inside: tolerance-averaging over all corners
        case 1: baseH = 0.2f * cd; tiltB = 0.4f * cd; tiltC = 0.7f * cd; break; // single deepest
        case 2: rad = 3.5f + 1.5f * uf(rng); baseH = 0.25f * cd; tiltB = 0.35f * cd; tiltC = 0.15f * cd; break; // big: full clip
        case 3: latOfs = { 4.5f + uf(rng), 3.8f + uf(rng), 0 }; break; // laterally outside: clipped empty
        case 4: baseH = (1.5f + 0.8f * uf(rng)) * cd; tiltB = 0.2f * cd; tiltC = 0.4f * cd; break; // above +cd slab: clipped empty
        case 5: baseH = 0.75f * cd; tiltB = 0.8f * cd; tiltC = 1.6f * cd; break; // straddles the +cd slab
        case 6: // penetrating; every second case beyond the -cd slab (pins the slab clip)
            if (i % 16 == 6) { baseH = -0.4f * cd; tiltB = 0.6f * cd; tiltC = 0.9f * cd; }
            else { baseH = (-1.6f - 0.5f * uf(rng)) * cd; tiltB = 1.1f * cd; tiltC = 2.0f * cd; }
            break;
        default: baseH = cd * (0.97f + 0.06f * uf(rng)); break; // gate/slab boundary probes (mostly tainted)
        }
        const float a0 = 6.2831853f * uf(rng);
        CA = add(add(cen, latOfs), V3{ rad * std::cos(a0), rad * std::sin(a0), baseH });
        CB = add(add(cen, latOfs), V3{ rad * std::cos(a0 + 2.2f), rad * std::sin(a0 + 2.2f), baseH + tiltB });
        CC = add(add(cen, latOfs), V3{ rad * std::cos(a0 + 4.3f), rad * std::sin(a0 + 4.3f), baseH + tiltC });

        // random cloth-space placement of the whole scene
        const F4 qF = randQuat(rng);
        const V3 tF = { 2.0f * (uf(rng) - 0.5f), 2.0f * (uf(rng) - 0.5f), 2.0f * (uf(rng) - 0.5f) };
        const V3 sA = add(quatRotate(qF, SA), tF), sB = add(quatRotate(qF, SB), tF), sC = add(quatRotate(qF, SC), tF);
        const V3 cA = add(quatRotate(qF, CA), tF), cB = add(quatRotate(qF, CB), tF), cC = add(quatRotate(qF, CC), tF);
        clothPos[i * 3 + 0] = { cA.x, cA.y, cA.z, 1.0f };
        clothPos[i * 3 + 1] = { cB.x, cB.y, cB.z, 1.0f };
        clothPos[i * 3 + 2] = { cC.x, cC.y, cC.z, 1.0f };

        // transforms + mesh scale; static tri verts derived by the inverse chain
        cs.clothQ = randQuat(rng);
        const V3 clothP = { 3.0f * (uf(rng) - 0.5f), 3.0f * (uf(rng) - 0.5f), 3.0f * (uf(rng) - 0.5f) };
        cs.clothP = { clothP.x, clothP.y, clothP.z, cd };
        cs.meshQ = randQuat(rng);
        const V3 meshP = { 3.0f * (uf(rng) - 0.5f), 3.0f * (uf(rng) - 0.5f), 3.0f * (uf(rng) - 0.5f) };
        cs.meshP = { meshP.x, meshP.y, meshP.z, 0.05f + 0.2f * uf(rng) };
        const V3 scl = { 0.8f + 0.6f * uf(rng), 0.8f + 0.6f * uf(rng), 0.8f + 0.6f * uf(rng) };
        cs.scale = { scl.x, scl.y, scl.z, 0 };
        cs.scaleRot = randQuat(rng);
        // world = clothT(clothLocal); meshShape = meshT^-1(world); vertex = shape2Vertex
        const V3 wA = add(quatRotate(cs.clothQ, sA), clothP);
        const V3 wB = add(quatRotate(cs.clothQ, sB), clothP);
        const V3 wC = add(quatRotate(cs.clothQ, sC), clothP);
        const V3 mA = shape2Vertex(quatRotateInv(cs.meshQ, sub(wA, meshP)), scl, cs.scaleRot);
        const V3 mB = shape2Vertex(quatRotateInv(cs.meshQ, sub(wB, meshP)), scl, cs.scaleRot);
        const V3 mC = shape2Vertex(quatRotateInv(cs.meshQ, sub(wC, meshP)), scl, cs.scaleRot);
        cs.triA = { mA.x, mA.y, mA.z, 0 };
        cs.triB = { mB.x, mB.y, mB.z, 0 };
        cs.triC = { mC.x, mC.y, mC.z, 0 };

        // Every ninth case pair is a STATIC rigid (PxNodeIndex low word PX_INVALID_NODE),
        // whose filter reference is its transform cache id, not its node.
        const bool isStatic = (i % 9 == 2 || i % 9 == 7) && (i % 2 == 1);
        const uint32_t rlo = isStatic ? 0xFFFFFFFFu : 0x40u + i % 11, rhi = 2 + i % 5;
        const uint32_t cacheRef = 100u + i % 7;
        cs.ids = { rlo, rhi, CLOTH0, i };
        cs.ids2 = { (0x90u + i % 13) | ((i % 6 == 1) ? 0x80000000u : 0u), i * 3 + 0, i * 3 + 1, i * 3 + 2 };
        cs.ids3 = { cacheRef, 0, 0, 0 };

        const uint64_t r64 = rigid64(rlo, rhi);
        const uint64_t key = filterKey(r64, cacheRef);
        const uint64_t triRef = PxgEncodeDeformableTriangle(CLOTH0, i);
        if (i % 9 == 2) // real skip: the (rigid, triangle) pair
            filters.push_back({ key, triRef });
        else if (i % 9 == 5) // near miss: all three VERTICES (the 5.6.1 key) -- must NOT skip
            for (uint32_t k = 0; k < 3; ++k)
                filters.push_back({ key, PxgEncodeDeformableVertex(CLOTH0, i * 3 + k) });
        else if (i % 9 == 7) // decoys: another rigid's key, and a static's RAW node index
            filters.push_back({ isStatic ? r64 : filterKey(r64 ^ 0x1000u, cacheRef), triRef });
    }

    // mode 1 cases share the filter table (different clothId => disjoint masks)
    const uint32_t NB = 640;
    std::vector<FstCase> bcases(NB);
    for (uint32_t i = 0; i < NB; ++i)
    {
        FstCase& cs = bcases[i];
        std::memset(&cs, 0, sizeof(cs));
        const float cd = 0.1f + 0.2f * uf(rng);
        const V3 he = { 0.4f + 1.0f * uf(rng), 0.4f + 1.0f * uf(rng), 0.4f + 1.0f * uf(rng) };
        cs.clothQ = randQuat(rng);
        const V3 bp = { 2.0f * (uf(rng) - 0.5f), 2.0f * (uf(rng) - 0.5f), 2.0f * (uf(rng) - 0.5f) };
        cs.clothP = { bp.x, bp.y, bp.z, cd };
        cs.meshP = { 0, 0, 0, 0.02f + 0.1f * uf(rng) };
        cs.triB = { he.x, he.y, he.z, 0 };

        // local point per family, then to world
        V3 lp;
        const uint32_t fam = i % 6;
        const float sx = uf(rng) > 0.5f ? 1.0f : -1.0f, sy = uf(rng) > 0.5f ? 1.0f : -1.0f, sz = uf(rng) > 0.5f ? 1.0f : -1.0f;
        switch (fam)
        {
        case 0: lp = { sx * he.x * 0.35f * uf(rng), sy * he.y * 0.4f * uf(rng), sz * (he.z - 0.35f * uf(rng) * he.z) }; break; // inside, z-face-ish
        case 1: lp = { sx * (he.x - 0.05f * he.x * uf(rng)), sy * he.y * 0.3f, sz * he.z * 0.3f }; break;                      // inside near x face
        case 2: lp = { sx * (he.x + 0.6f * cd * uf(rng)), sy * he.y * 0.5f, sz * he.z * 0.5f }; break;                          // outside near face
        case 3: lp = { sx * (he.x + 0.7f * cd), sy * (he.y + 0.7f * cd), sz * (he.z + 0.7f * cd) }; break;                      // outside near corner
        case 4: lp = { sx * (he.x + 2.5f * cd + 0.5f), sy * he.y * 0.4f, sz * he.z * 0.4f }; break;                             // clean reject
        default: lp = { sx * (he.x * 0.5f), sy * (he.y * 0.5f), sz * he.z * 0.5f }; break;                                      // inside, tie-prone
        }
        const V3 wp = add(quatRotate(cs.clothQ, lp), bp);
        cs.triA = { wp.x, wp.y, wp.z, 0 };

        const bool isStatic = (i % 20 == 13 || i % 20 == 19);
        const uint32_t rlo = isStatic ? 0xFFFFFFFFu : 0xA00u + i % 9, rhi = 1 + i % 4;
        const uint32_t cacheRef = 200u + i % 5;
        cs.ids = { rlo, rhi, CLOTH1, i };
        cs.ids2 = { (0x11u + i % 7) | ((i % 5 == 2) ? 0x80000000u : 0u), 0, 0, 0 };
        cs.ids3 = { cacheRef, 0, 0, 0 };
        const uint64_t key = filterKey(rigid64(rlo, rhi), cacheRef);
        if (i % 10 == 3)
            filters.push_back({ key, PxgEncodeDeformableVertex(CLOTH1, i) });
        else if (i % 10 == 6) // near miss: adjacent vertex id
            filters.push_back({ key, PxgEncodeDeformableVertex(CLOTH1, i + 1) });
        else if (i % 10 == 9) // near misses: the TRIANGLE of the same id, and a static's raw node
            filters.push_back({ isStatic ? rigid64(rlo, rhi) : key, isStatic ? PxgEncodeDeformableVertex(CLOTH1, i)
                                                                             : PxgEncodeDeformableTriangle(CLOTH1, i) });
    }
    std::sort(filters.begin(), filters.end(), pairLess);

    // ---- CPU reference, mode 0 ----
    std::vector<RefOut> ref(NC);
    std::vector<int> taint(NC, 0);
    size_t skipN = 0, contactN = 0, emptyN = 0, avgN = 0, deepN = 0, degenN = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const FstCase& cs = cases[i];
        Slack sl;
        RefOut& ro = ref[i];
        ro.filtered = findRef(filters, filterKey(rigid64(cs.ids.x, cs.ids.y), cs.ids3.x),
                              PxgEncodeDeformableTriangle(CLOTH0, cs.ids.w));
        if (ro.filtered) { ++skipN; continue; }

        const V3 a1 = v3f(clothPos[cs.ids2.y]);
        const V3 b1 = v3f(clothPos[cs.ids2.z]);
        const V3 c1 = v3f(clothPos[cs.ids2.w]);
        const float cd = cs.clothP.w;

        // P9.94a: transform + trim + deepest point via the shared literal chain
        V3 poly[8] = { a1, b1, c1, {}, {}, {}, {}, {} };
        const Mode0Chain m0 = refMode0Chain(cs, poly, sl);
        const V3 fixNormal = m0.fixNormal;
        const F4 triPlane = m0.triPlane;
        const int corners = m0.corners;
        const float sep = m0.sep;
        const V3 dp = m0.dp;
        int cnt = 0;
        for (int k = 0; k < corners; ++k) // count the averaging band population for coverage
            if (planeDist(triPlane, poly[k]) < sep + 0.011f * cd) ++cnt;
        sl.note(sep - cd, BAND_EPS); // strict gate
        ro.wrote = sep < cd;
        ro.sep = sep;
        if (ro.wrote)
        {
            baryRef(a1, b1, c1, dp, ro.bary, sl);
            ro.p = add(quatRotate(cs.clothQ, dp), v3f(cs.clothP));
            ro.n = quatRotate(cs.clothQ, fixNormal);
        }
        taint[i] = sl.near ? 1 : 0;
        if (!taint[i])
        {
            if (ro.wrote)
            {
                ++contactN;
                if (cnt > 1) ++avgN;
                if (sep < -0.9f * cd) ++deepN;
                if (i % 32 == 17) ++degenN; // zero-normal degenerate contact
            }
            else if (corners == 0) ++emptyN;
        }
    }

    // ---- CPU reference, mode 1 ----
    std::vector<RefOut> bref(NB);
    std::vector<int> btaint(NB, 0);
    size_t bSkipN = 0, bInsideN = 0, bOutsideN = 0, bRejN = 0;
    for (uint32_t i = 0; i < NB; ++i)
    {
        const FstCase& cs = bcases[i];
        Slack sl;
        RefOut& ro = bref[i];
        ro.filtered = findRef(filters, filterKey(rigid64(cs.ids.x, cs.ids.y), cs.ids3.x),
                              PxgEncodeDeformableVertex(CLOTH1, i));
        if (ro.filtered) { ++bSkipN; continue; }

        const V3 he = v3f(cs.triB);
        const V3 sc = quatRotateInv(cs.clothQ, sub(v3f(cs.triA), v3f(cs.clothP)));
        const float cd = cs.clothP.w;
        const V3 p = { std::min(std::max(sc.x, -he.x), he.x),
                       std::min(std::max(sc.y, -he.y), he.y),
                       std::min(std::max(sc.z, -he.z), he.z) };
        const V3 v = sub(sc, p);
        const float lengthSq = dot3(v, v);
        const float infl = cd * cd;
        sl.note(infl - lengthSq, 2e-5f * (1.0f + lengthSq));
        bool inside = false;
        if (infl > lengthSq)
        {
            const bool bX = he.x >= std::fabs(sc.x), bY = he.y >= std::fabs(sc.y), bZ = he.z >= std::fabs(sc.z);
            sl.note(he.x - std::fabs(sc.x), 2e-5f);
            sl.note(he.y - std::fabs(sc.y), 2e-5f);
            sl.note(he.z - std::fabs(sc.z), 2e-5f);
            V3 locN;
            float pen;
            if (bX && bY && bZ)
            {
                inside = true;
                const V3 d2s = { he.x - std::fabs(p.x), he.y - std::fabs(p.y), he.z - std::fabs(p.z) };
                sl.note(d2s.x - d2s.y, 2e-5f);
                sl.note(d2s.x - d2s.z, 2e-5f);
                sl.note(d2s.y - d2s.z, 2e-5f);
                locN = p.z >= 0 ? V3{ 0, 0, 1 } : V3{ 0, 0, -1 };
                pen = -d2s.z;
                float chosen = p.z;
                if (d2s.x <= d2s.y && d2s.x <= d2s.z) { locN = p.x >= 0 ? V3{ 1, 0, 0 } : V3{ -1, 0, 0 }; pen = -d2s.x; chosen = p.x; }
                else if (d2s.y <= d2s.z) { locN = p.y >= 0 ? V3{ 0, 1, 0 } : V3{ 0, -1, 0 }; pen = -d2s.y; chosen = p.y; }
                sl.note(chosen, 2e-5f); // >= 0 sign pick of the winning axis
            }
            else
            {
                const float len = std::sqrt(lengthSq);
                sl.note(len, 5e-4f); // v/|v| direction conditioning for grazing exits
                locN = mul(v, 1.0f / len);
                pen = len;
            }
            ro.wrote = true;
            ro.n = quatRotate(cs.clothQ, locN);
            ro.sep = pen;
            ro.p = sub(v3f(cs.triA), mul(ro.n, pen));
        }
        btaint[i] = sl.near ? 1 : 0;
        if (!btaint[i])
        {
            if (ro.wrote) { if (inside) ++bInsideN; else ++bOutsideN; }
            else ++bRejN;
        }
    }

    // ---- GPU ----
    PxgSlangComputeDevice* dev = testDevice();
    Kernel k("fcStaticTriCloth.spv", "fcStaticTriClothKernel",
             { (uint32_t)sizeof(FstCase), 16, 16 }, sizeof(FstParams));

    // buffers always sized CAP; `cap` only sets the writer's maxContacts gate,
    // so slots in [cap, CAP) must stay poison
    auto runMode = [&](const std::vector<FstCase>& cc, const std::vector<F4>& cp, uint32_t mode, uint32_t cap) -> Run
    {
        return runFst(dev, k, cc, cp, filters, mode, cap, CAP);
    };

    const Run r0 = runMode(cases, clothPos, 0, CAP);
    const std::vector<F4> dummyPos(4, { 0, 0, 0, 0 });
    const Run r1 = runMode(bcases, dummyPos, 1, CAP);
    const Run r0t = runMode(cases, clothPos, 0, TINY); // capacity gate

    // ---- verify mode 0 ----
    size_t mm = 0, mmW = 0;
    std::vector<int> slotOf(NC, -1);
    for (uint32_t s = 0; s < std::min(r0.wc, CAP); ++s)
    {
        const uint64_t ref1 = r0.in[s].ref1;
        const uint32_t caseIdx = PxgGetDeformableFeatureId(ref1);
        if (PxgGetDeformableId(ref1) != CLOTH0 || PxgGetDeformableFeatureType(ref1) != PxgDeformableRef::eTRIANGLE ||
            ref1 != PxgEncodeDeformableTriangle(CLOTH0, caseIdx) || caseIdx >= NC || slotOf[caseIdx] != -1)
        { ++mmW; continue; }
        slotOf[caseIdx] = (int)s;
        const uint64_t r64 = rigid64(cases[caseIdx].ids.x, cases[caseIdx].ids.y);
        // ref0 is the rigid's NODE (rigidId.getInd()), static or not -- the filter key is
        // not what the contact carries.
        if (r0.in[s].ref0 != r64) ++mmW;
        if (r0.mat[s] != (cases[caseIdx].ids2.x & 0xFFFFu)) ++mmW;
        if (r0.by[s] != r64 || r0.tb[s] != PxgGetContactSortKey(r64) || r0.ix[s] != s) ++mmW;
    }
    uint32_t sumChecked = 0, numTainted = 0;
    size_t checked = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const RefOut& ro = ref[i];
        if (ro.filtered)
        {
            if (r0.dbg[i].x != 1.0f || slotOf[i] != -1) ++mm;
            continue;
        }
        if (taint[i]) { ++numTainted; continue; }
        ++checked;
        if ((r0.dbg[i].y != 0.0f) != ro.wrote) { ++mm; continue; }
        if (!ro.wrote) { if (slotOf[i] != -1) ++mm; continue; }
        ++sumChecked;
        if (slotOf[i] == -1) { ++mm; continue; }
        const uint32_t s = (uint32_t)slotOf[i];
        if (!almostEqual(r0.pt[s].x, ro.p.x, 2e-3f, 2e-3f) ||
            !almostEqual(r0.pt[s].y, ro.p.y, 2e-3f, 2e-3f) ||
            !almostEqual(r0.pt[s].z, ro.p.z, 2e-3f, 2e-3f) ||
            r0.pt[s].w != cases[i].meshP.w ||
            !almostEqual(r0.np[s].x, ro.n.x, 2e-3f, 2e-3f) ||
            !almostEqual(r0.np[s].y, ro.n.y, 2e-3f, 2e-3f) ||
            !almostEqual(r0.np[s].z, ro.n.z, 2e-3f, 2e-3f) ||
            !almostEqual(r0.np[s].w, ro.sep, 5e-4f, 5e-4f) ||
            !almostEqual(r0.bc[s].x, ro.bary.x, 6e-3f, 6e-3f) ||
            !almostEqual(r0.bc[s].y, ro.bary.y, 6e-3f, 6e-3f) ||
            !almostEqual(r0.bc[s].z, ro.bary.z, 6e-3f, 6e-3f) ||
            r0.bc[s].w != 0.0f)
            ++mm;
    }
    PXG_CHECK_MSG(mm == 0 && mmW == 0, "fcStaticTriCloth m0: %zu content + %zu slot mismatches", mm, mmW);
    PXG_CHECK_MSG(r0.wc >= sumChecked && r0.wc <= sumChecked + numTainted + 0,
                  "fcStaticTriCloth m0: counter %u outside [%u, %u]", r0.wc, sumChecked, sumChecked + numTainted);

    // per-batch contiguity (batches with no tainted case)
    size_t mmB = 0;
    for (uint32_t b = 0; b * 32 < NC; ++b)
    {
        bool clean = true;
        for (uint32_t l = 0; l < 32 && b * 32 + l < NC; ++l)
            if (taint[b * 32 + l]) clean = false;
        if (!clean) continue;
        int prev = -1;
        for (uint32_t l = 0; l < 32 && b * 32 + l < NC; ++l)
        {
            const uint32_t i = b * 32 + l;
            if (ref[i].filtered || !ref[i].wrote) continue;
            if (slotOf[i] == -1) continue; // already counted as mismatch
            if (prev != -1 && slotOf[i] != prev + 1) ++mmB;
            prev = slotOf[i];
        }
    }
    PXG_CHECK_MSG(mmB == 0, "fcStaticTriCloth m0: %zu batch-contiguity breaks", mmB);

    // ---- capacity run: same counter (reservation ungated), stores gated ----
    PXG_CHECK_MSG(r0t.wc == r0.wc, "fcStaticTriCloth cap: counter %u != %u", r0t.wc, r0.wc);
    size_t mmC = 0;
    for (uint32_t s = 0; s < TINY; ++s)
        if (r0t.ix[s] != s) ++mmC; // all slots below TINY are claimed and written
    for (uint32_t s = TINY; s < CAP; ++s) // beyond the gate: EVERY writer lane poison
        if (r0t.ix[s] != POISON || r0t.tb[s] != POISON || r0t.by[s] != ((uint64_t(POISON) << 32) | POISON) ||
            r0t.in[s].ref0 != INFOPOISON || r0t.in[s].ref1 != INFOPOISON || r0t.mat[s] != MATPOISON ||
            r0t.pt[s].x != -9.0f || r0t.np[s].x != -9.0f || r0t.bc[s].x != -9.0f)
            ++mmC;
    PXG_CHECK_MSG(mmC == 0, "fcStaticTriCloth cap: %zu capacity-gate leaks", mmC);

    // ---- verify mode 1 ----
    size_t bmm = 0, bmmW = 0;
    std::vector<int> bslotOf(NB, -1);
    for (uint32_t s = 0; s < std::min(r1.wc, CAP); ++s)
    {
        const uint64_t ref1 = r1.in[s].ref1;
        const uint32_t caseIdx = PxgGetDeformableFeatureId(ref1);
        if (ref1 != PxgEncodeDeformableVertex(CLOTH1, caseIdx) || caseIdx >= NB || bslotOf[caseIdx] != -1)
        { ++bmmW; continue; }
        bslotOf[caseIdx] = (int)s;
        const uint64_t r64 = rigid64(bcases[caseIdx].ids.x, bcases[caseIdx].ids.y);
        if (r1.in[s].ref0 != r64 || r1.by[s] != r64 || r1.tb[s] != PxgGetContactSortKey(r64) || r1.ix[s] != s)
            ++bmmW;
        if (r1.mat[s] != (bcases[caseIdx].ids2.x & 0xFFFFu)) ++bmmW;
    }
    uint32_t bSumChecked = 0, bNumTainted = 0;
    size_t bchecked = 0;
    for (uint32_t i = 0; i < NB; ++i)
    {
        const RefOut& ro = bref[i];
        if (ro.filtered)
        {
            if (r1.dbg[i].x != 1.0f || bslotOf[i] != -1) ++bmm;
            continue;
        }
        if (btaint[i]) { ++bNumTainted; continue; }
        ++bchecked;
        if ((r1.dbg[i].y != 0.0f) != ro.wrote) { ++bmm; continue; }
        if (!ro.wrote) { if (bslotOf[i] != -1) ++bmm; continue; }
        ++bSumChecked;
        if (bslotOf[i] == -1) { ++bmm; continue; }
        const uint32_t s = (uint32_t)bslotOf[i];
        if (!almostEqual(r1.pt[s].x, ro.p.x, 2e-3f, 2e-3f) ||
            !almostEqual(r1.pt[s].y, ro.p.y, 2e-3f, 2e-3f) ||
            !almostEqual(r1.pt[s].z, ro.p.z, 2e-3f, 2e-3f) ||
            r1.pt[s].w != bcases[i].meshP.w ||
            !almostEqual(r1.np[s].x, ro.n.x, 2e-3f, 2e-3f) ||
            !almostEqual(r1.np[s].y, ro.n.y, 2e-3f, 2e-3f) ||
            !almostEqual(r1.np[s].z, ro.n.z, 2e-3f, 2e-3f) ||
            !almostEqual(r1.np[s].w, ro.sep, 5e-4f, 5e-4f) ||
            // 5.12: the vertex overload writes NO barycentric (5.6.1 wrote (0,0,0,1))
            r1.bc[s].x != -9.0f || r1.bc[s].y != -9.0f || r1.bc[s].z != -9.0f || r1.bc[s].w != -9.0f)
            ++bmm;
    }
    PXG_CHECK_MSG(bmm == 0 && bmmW == 0, "fcStaticTriCloth m1: %zu content + %zu slot mismatches", bmm, bmmW);
    PXG_CHECK_MSG(r1.wc >= bSumChecked && r1.wc <= bSumChecked + bNumTainted,
                  "fcStaticTriCloth m1: counter %u outside [%u, %u]", r1.wc, bSumChecked, bSumChecked + bNumTainted);
    size_t bmmB = 0;
    for (uint32_t b = 0; b * 32 < NB; ++b)
    {
        bool clean = true;
        for (uint32_t l = 0; l < 32 && b * 32 + l < NB; ++l)
            if (btaint[b * 32 + l]) clean = false;
        if (!clean) continue;
        int prev = -1;
        for (uint32_t l = 0; l < 32 && b * 32 + l < NB; ++l)
        {
            const uint32_t i = b * 32 + l;
            if (bref[i].filtered || !bref[i].wrote || bslotOf[i] == -1) continue;
            if (prev != -1 && bslotOf[i] != prev + 1) ++bmmB;
            prev = bslotOf[i];
        }
    }
    PXG_CHECK_MSG(bmmB == 0, "fcStaticTriCloth m1: %zu batch-contiguity breaks", bmmB);

    PXG_CHECK_MSG(checked > NC * 3 / 5 && bchecked > NB * 3 / 5,
                  "fcStaticTriCloth: too many tainted (m0=%u m1=%u)", numTainted, bNumTainted);
    PXG_CHECK_MSG(skipN > 80 && contactN > 300 && emptyN > 100 && avgN > 60 && deepN > 25 && degenN > 20,
                  "fcStaticTriCloth m0 coverage thin (s=%zu c=%zu e=%zu a=%zu d=%zu g=%zu)",
                  skipN, contactN, emptyN, avgN, deepN, degenN);
    PXG_CHECK_MSG(bSkipN > 40 && bInsideN > 120 && bOutsideN > 100 && bRejN > 60,
                  "fcStaticTriCloth m1 coverage thin (s=%zu i=%zu o=%zu r=%zu)", bSkipN, bInsideN, bOutsideN, bRejN);
}

// P9.94a: directed zero-sign cases for trimeshToClothTransform = clothT.transformInv(meshT).
// Cloth identity, meshQ = (0,0,1,0), static triangle on the cloth-space x = 0 plane: the
// normal is (1,+0,+0) and every planeDist term is a signed zero, so sep (a strictly-inside,
// unclipped cloth vertex) is exactly the sign of mtcP.x.  The CUDA conj().rotate and the old
// rotateInv spelling give opposite zero signs in that lane; A and B cover both directions.
static uint32_t p94Bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

PXG_TEST(fcStaticTriCloth_p94ZeroSign)
{
    struct Variant { const char* name; F4 meshP; V3 cloth[3]; uint32_t cudaSep; };
    const Variant variants[2] = {
        { "A", { -0.0f, -0.0f, 0.5f, 0.05f },
          { { -0.0f, -1.25f, -0.75f }, { -0.0f, -1.5f, -0.75f }, { -0.0f, -1.25f, -1.0f } }, 0x00000000u },
        { "B", { -0.0f, -0.0f, -0.0f, 0.05f },
          { { -0.0f, -1.25f, -1.25f }, { -0.0f, -1.5f, -1.25f }, { -0.0f, -1.25f, -1.5f } }, 0x80000000u },
    };

    PxgSlangComputeDevice* dev = testDevice();
    Kernel k("fcStaticTriCloth.spv", "fcStaticTriClothKernel",
             { (uint32_t)sizeof(FstCase), 16, 16 }, sizeof(FstParams));
    const uint32_t CAPD = 4;

    for (const Variant& v : variants)
    {
        FstCase cs;
        std::memset(&cs, 0, sizeof(cs));
        cs.triA = { 0.0f, 1.0f, -1.0f, 0 };
        cs.triB = { 0.0f, 2.0f, -1.0f, 0 };
        cs.triC = { 0.0f, 1.0f, -2.0f, 0 };
        cs.clothQ = { 0, 0, 0, 1 };
        cs.clothP = { 0, 0, 0, 0.1f }; // w = cDistance
        cs.meshQ = { 0, 0, 1, 0 };
        cs.meshP = v.meshP;
        cs.scale = { 1, 1, 1, 0 };
        cs.scaleRot = { 0, 0, 0, 1 };
        cs.ids = { 0x40u, 2u, CLOTH0, 0u };
        cs.ids2 = { 0x90u, 0u, 1u, 2u };
        const std::vector<FstCase> cc(1, cs);
        std::vector<F4> cp(3);
        for (int j = 0; j < 3; ++j)
            cp[j] = { v.cloth[j].x, v.cloth[j].y, v.cloth[j].z, 1.0f };

        // literal CUDA oracle, then the rotateInv twin
        Slack sl;
        V3 poly[8] = { v.cloth[0], v.cloth[1], v.cloth[2], {}, {}, {}, {}, {} };
        const Mode0Chain lit = refMode0Chain(cs, poly, sl);
        gP94TinvRotInv = true;
        Slack slT;
        V3 polyT[8] = { v.cloth[0], v.cloth[1], v.cloth[2], {}, {}, {}, {}, {} };
        const Mode0Chain twin = refMode0Chain(cs, polyT, slT);
        gP94TinvRotInv = false;
        const uint32_t litBits = p94Bits(lit.sep), twinBits = p94Bits(twin.sep);
        PXG_CHECK_MSG(lit.corners == 3 && lit.sep < cs.clothP.w && litBits == v.cudaSep,
                      "fcStaticTriCloth p94 %s: oracle corners %d sep 0x%08x (want 3, 0x%08x)",
                      v.name, lit.corners, litBits, v.cudaSep);
        PXG_CHECK_MSG(twinBits != litBits,
                      "fcStaticTriCloth p94 %s: twin sep 0x%08x == literal 0x%08x (case does not discriminate)",
                      v.name, twinBits, litBits);

        // GPU: one case, empty filter table
        const Run r = runFst(dev, k, cc, cp, std::vector<PxgDeformablePairInfo>(), 0, CAPD, CAPD);
        const std::vector<F4>& np = r.np;
        const std::vector<F4>& dbg = r.dbg;
        const uint32_t wc = r.wc;

        PXG_CHECK_MSG(wc == 1 && dbg[0].y == 1.0f && dbg[0].w == 3.0f,
                      "fcStaticTriCloth p94 %s: wc %u hasContact %g corners %g", v.name, wc, dbg[0].y, dbg[0].w);
        PXG_CHECK_MSG(p94Bits(np[0].w) == litBits && p94Bits(dbg[0].z) == litBits,
                      "fcStaticTriCloth p94 %s: gpu sep np.w 0x%08x dbg.z 0x%08x != literal 0x%08x (twin 0x%08x)",
                      v.name, p94Bits(np[0].w), p94Bits(dbg[0].z), litBits, twinBits);
    }
}

} // namespace
