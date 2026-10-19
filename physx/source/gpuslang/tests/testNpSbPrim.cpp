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

// Verification of npSbPrim.slang (softbodyPrimitives.cu non-GJK cores, PhysX
// 5.12).  sbTetPlane and sbEmit compare EXACTLY (1/64-grid vertices,
// signed-permutation plane quaternions, integer emitter fixtures) with
// deliberate on-threshold cases pinning the inclusive gates; sbParticle
// and sbClothVert use transcribed CPU references with Slack taints on the
// sqrt/normalize-driven gates (sqDist vs cDistance^2, dot(n,n2) vs 0.8 /
// 0.707, sqDist vs 1e-8, m vs 1e-10, cross-face running-min ties with the
// benign-same-point exclusion).  The filters and writers the shader calls are
// the ENGINE's modules: filter tables pin the rigid-volume search (a STATIC
// rigid is keyed on its transform cache id, PxgEncodeRigidFilterRef) and the
// three volume-surface searches ((ANY tet, vertex), (tet, ANY vertex), (tet,
// vertex)) with near misses that include the reversed pair order and a
// triangle ref; writer checks cover PxgRigidDbContactWriter::writeVolumeContact
// (pair info, PxU16 material, the u64 sort key), PxgParticleDbContactWriter
// (particle ref sort key) and the vertex overload of PxgDbDbContactWriter::
// writeSurfaceVolumeContact -- none of them writes a barycentric in 5.12 --
// plus capacity runs (ungated counters overshoot, stores stop).  References
// and pair order come from the real engine host helpers.
//
// 5.12 removed what the 5.6.1 transcription here used to pin: the particle
// filter chain (findRange / mUniqueId / the u64->u32 mask truncation) and the
// rest distance of sbParticleCollision, the barycentrics of the particle and
// cloth-vertex contacts, the emitter's alternativeWriteMode, and the
// PxgFEMContactWriter / PxgSoftBodyContactWriter packed ids.

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
struct SbCase { F4 a, b, c, d, p, q; U4 ids, ids2; };
// Byte-identical to npSbPrim.slang's SbParams: the filter list and the writer streams
// are device addresses.
struct SbParams
{
    uint32_t numCases, maxContacts, nbFilter, pad0;
    uint64_t filterPairs, filterSafeAddr;
    uint64_t outPoint, outNormalPen, outBarycentric0, outBarycentric1, outContactInfo, outMaterial;
    uint64_t totalContactCount, sortKey, sortKeyTemp, sortIndex;
};
static_assert(sizeof(SbParams) == 112, "npSbPrim push block");
static_assert(sizeof(PxgDeformablePairInfo) == 16, "PxgDeformablePairInfo");

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static V3 neg(V3 a) { return { -a.x, -a.y, -a.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 crossp(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 v3(F4 f) { return { f.x, f.y, f.z }; }
static V3 getNorm0(V3 v) { const float m = dot3(v, v); return m > 0 ? mul(v, 1.0f / std::sqrt(m)) : V3{ 0, 0, 0 }; }
static float pxNormalizeRef(V3& v)
{
    const float m = std::sqrt(dot3(v, v));
    if (m > 0.0f) v = mul(v, 1.0f / m);
    return m;
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
static V3 basisVector0(F4 q)
{
    const float x2 = q.x * 2.0f, w2 = q.w * 2.0f;
    return { (q.w * w2) - 1.0f + q.x * x2, (q.z * w2) + q.y * x2, (-q.y * w2) + q.z * x2 };
}
static F4 exactQuat(std::mt19937& rng)
{
    if (rng() % 4 == 0) return { 0, 0, 0, 1 };
    const uint32_t s = rng() % 16;
    return { (s & 1) ? -0.5f : 0.5f, (s & 2) ? -0.5f : 0.5f, (s & 4) ? -0.5f : 0.5f, (s & 8) ? -0.5f : 0.5f };
}
static float grid64(std::mt19937& rng, float lo, float hi)
{
    const int n = int((hi - lo) * 64.0f);
    return lo + float(int(rng() % uint32_t(n + 1))) / 64.0f;
}

struct Slack
{
    bool near = false;
    void note(float margin, float eps) { if (margin != 0.0f && std::fabs(margin) < eps) near = true; }
};

// ---- filter references (5.12: one sorted PxgDeformablePairInfo list) -----------
static uint64_t rigid64(uint32_t lo, uint32_t hi) { return (uint64_t(hi) << 32) | lo; }
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
static const uint32_t ANY = PxgDeformableRef::ANY_FEATURE;

// ---- geometry references (transcribed) -----------------------------------------
static F4 pointOutsideOfPlane4Ref(V3 p, V3 a, V3 b, V3 c, V3 d)
{
    const V3 ap = sub(p, a), ab = sub(b, a), ac = sub(c, a), ad = sub(d, a);
    const V3 v0 = crossp(ab, ac);
    const float sa0 = dot3(v0, ap), sd0 = dot3(v0, ad);
    const V3 v1 = crossp(ac, ad);
    const float sa1 = dot3(v1, ap), sd1 = dot3(v1, ab);
    const V3 v2 = crossp(ad, ab);
    const float sa2 = dot3(v2, ap), sd2 = dot3(v2, ac);
    const V3 bd = sub(d, b), bc = sub(c, b);
    const V3 v3_ = crossp(bd, bc);
    const float sd3 = dot3(v3_, sub(p, b)), sa3 = dot3(v3_, sub(a, b));
    return { sa0 * sd0, sa1 * sd1, sa2 * sd2, sa3 * sd3 };
}

static V3 closestPtPointTriangle2Ref(V3 p, V3 a, V3 b, V3 c, V3 ab, V3 ac)
{
    const V3 ap = sub(p, a);
    const float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    const V3 bp = sub(p, b);
    const float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) return add(a, mul(ab, d1 / (d1 - d3)));
    const V3 cp = sub(p, c);
    const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) return add(a, mul(ac, d2 / (d2 - d6)));
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, mul(sub(c, b), w));
    }
    const float denom = 1.0f / (va + vb + vc);
    return add(a, add(mul(ab, vb * denom), mul(ac, vc * denom)));
}

static void closestPtTetrahedronRef(V3 point, uint32_t hint, V3 a, V3 b, V3 c, V3 d,
                                    V3& normal, V3& closestP, float& sqDist, Slack& sl)
{
    const V3 ac = sub(c, a), ab = sub(b, a), ad = sub(d, a);
    struct Face { uint32_t bit; V3 p0, p1, p2, e0, e1, nc0, nc1, flipRef; bool flipGT; };
    // face list transcribed in order; flip test operand + comparison direction
    const Face faces[4] = {
        { 1, a, b, c, ab, ac, ac, ab, ad, false },
        { 2, a, b, d, ab, ad, ab, ad, ac, false },
        { 4, a, c, d, ac, ad, ac, ad, ab, false },
        { 8, b, c, d, sub(c, b), sub(d, b), sub(c, b), sub(d, b), ab, true },
    };
    for (int f = 0; f < 4; ++f)
    {
        if (!(hint & faces[f].bit)) continue;
        const V3 tPoint = closestPtPointTriangle2Ref(point, faces[f].p0, faces[f].p1, faces[f].p2, faces[f].e0, faces[f].e1);
        const V3 v = sub(tPoint, point);
        const float tSqDist = dot3(v, v);
        if (tSqDist < sqDist)
        {
            V3 n = getNorm0(crossp(faces[f].nc0, faces[f].nc1));
            // faces 1/2/4 build cross(ac,ab)/cross(ab,ad)/cross(ac,ad); face 1 order is (ac,ab)
            if (f == 0) n = getNorm0(crossp(ac, ab));
            const float flipDot = dot3(faces[f].flipRef, n);
            if (faces[f].flipGT ? (flipDot > 0.0f) : (flipDot < 0.0f)) n = neg(n);
            normal = n;
            closestP = tPoint;
            sqDist = tSqDist;
        }
        else
        {
            // cross-face running-min tie: output-relevant when the tied
            // candidate differs in closest point OR face normal (n2 feeds the
            // output even when the points coincide)
            V3 nc = getNorm0(crossp(faces[f].nc0, faces[f].nc1));
            if (f == 0) nc = getNorm0(crossp(ac, ab));
            const float fd = dot3(faces[f].flipRef, nc);
            if (faces[f].flipGT ? (fd > 0.0f) : (fd < 0.0f)) nc = neg(nc);
            const V3 dcp = sub(tPoint, closestP);
            const V3 dcn = sub(nc, normal);
            if (dot3(dcp, dcp) > 1e-6f || dot3(dcn, dcn) > 1e-6f)
                sl.note(tSqDist - sqDist, 1e-4f);
        }
    }
}

static const uint32_t POISON = 0xCDCDCDCDu;
static const uint64_t INFOPOISON = 0xDEADBEEFDEADBEEFull;
static const uint64_t KEYPOISON = (uint64_t(POISON) << 32) | POISON;
static const uint32_t MATPOISON = 0xFFFFu;

// shared GPU runner ------------------------------------------------------------
struct SbRun
{
    std::vector<F4> pt, np, bc0, bc1, dbg;
    std::vector<PxgDeformablePairInfo> in;
    std::vector<uint64_t> by;
    std::vector<uint32_t> tb, ix, mat;
    uint32_t wc;
};
static SbRun runSb(const char* entry, const std::vector<SbCase>& cases,
                   const std::vector<PxgDeformablePairInfo>& filters, uint32_t maxContacts, uint32_t CAP)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t n = (uint32_t)cases.size();
    std::vector<F4> f4Init(CAP, { -9, -9, -9, -9 });
    std::vector<PxgDeformablePairInfo> infoInit(CAP, PxgDeformablePairInfo{ INFOPOISON, INFOPOISON });
    std::vector<uint32_t> u1Init(CAP, POISON), matInit((CAP + 1) / 2, 0xFFFFFFFFu), wc0(1, 0);
    std::vector<uint64_t> u64Init(CAP, KEYPOISON);
    std::vector<F4> dbgInit(n, { -7, -7, -7, -7 });
    // An empty list still needs a mapped address: one dummy entry, nbFilter = 0.
    const std::vector<PxgDeformablePairInfo> fl =
        filters.empty() ? std::vector<PxgDeformablePairInfo>(1, PxgDeformablePairInfo{ 0, 0 }) : filters;

    PxgBufferHandle bC = createBufferFrom(dev, cases);
    PxgBufferHandle bDb = createBufferFrom(dev, dbgInit);
    PxgBufferHandle bF = createBufferFrom(dev, fl);
    PxgBufferHandle bPt = createBufferFrom(dev, f4Init);
    PxgBufferHandle bNp = createBufferFrom(dev, f4Init);
    PxgBufferHandle bB0 = createBufferFrom(dev, f4Init);
    PxgBufferHandle bB1 = createBufferFrom(dev, f4Init);
    PxgBufferHandle bIn = createBufferFrom(dev, infoInit);
    PxgBufferHandle bMat = createBufferFrom(dev, matInit);
    PxgBufferHandle bWc = createBufferFrom(dev, wc0);
    PxgBufferHandle bBy = createBufferFrom(dev, u64Init);
    PxgBufferHandle bTb = createBufferFrom(dev, u1Init);
    PxgBufferHandle bIx = createBufferFrom(dev, u1Init);

    SbParams prm = {};
    prm.numCases = n;
    prm.maxContacts = maxContacts;
    prm.nbFilter = (uint32_t)filters.size();
    prm.filterPairs = dev->getBufferDeviceAddress(bF);
    prm.filterSafeAddr = prm.filterPairs;
    prm.outPoint = dev->getBufferDeviceAddress(bPt);
    prm.outNormalPen = dev->getBufferDeviceAddress(bNp);
    prm.outBarycentric0 = dev->getBufferDeviceAddress(bB0);
    prm.outBarycentric1 = dev->getBufferDeviceAddress(bB1);
    prm.outContactInfo = dev->getBufferDeviceAddress(bIn);
    prm.outMaterial = dev->getBufferDeviceAddress(bMat);
    prm.totalContactCount = dev->getBufferDeviceAddress(bWc);
    prm.sortKey = dev->getBufferDeviceAddress(bBy);
    prm.sortKeyTemp = dev->getBufferDeviceAddress(bTb);
    prm.sortIndex = dev->getBufferDeviceAddress(bIx);
    Kernel k("npSbPrim.spv", entry, { (uint32_t)sizeof(SbCase), 16 }, sizeof(SbParams));
    k.dispatch((n + 63) / 64, { bC, bDb }, prm);
    dev->flush();
    SbRun r;
    r.pt = downloadBuffer<F4>(dev, bPt, CAP);
    r.np = downloadBuffer<F4>(dev, bNp, CAP);
    r.bc0 = downloadBuffer<F4>(dev, bB0, CAP);
    r.bc1 = downloadBuffer<F4>(dev, bB1, CAP);
    r.in = downloadBuffer<PxgDeformablePairInfo>(dev, bIn, CAP);
    r.by = downloadBuffer<uint64_t>(dev, bBy, CAP);
    r.tb = downloadBuffer<uint32_t>(dev, bTb, CAP);
    r.ix = downloadBuffer<uint32_t>(dev, bIx, CAP);
    const std::vector<uint32_t> matWords = downloadBuffer<uint32_t>(dev, bMat, (CAP + 1) / 2);
    for (uint32_t s = 0; s < CAP; ++s)
        r.mat.push_back((matWords[s / 2] >> (16u * (s & 1u))) & 0xFFFFu);
    r.dbg = downloadBuffer<F4>(dev, bDb, n);
    r.wc = downloadBuffer<uint32_t>(dev, bWc, 1)[0];
    for (PxgBufferHandle h : { bC, bDb, bF, bPt, bNp, bB0, bB1, bIn, bMat, bWc, bBy, bTb, bIx })
        dev->destroyBuffer(h);
    return r;
}
// No 5.12 writer on these paths writes a barycentric: both slots stay poisoned.
static bool baryUntouched(const SbRun& r, uint32_t s)
{
    return r.bc0[s].x == -9.0f && r.bc0[s].w == -9.0f && r.bc1[s].x == -9.0f && r.bc1[s].w == -9.0f;
}

// ============================================================================
// sbTetPlane — EXACT
// ============================================================================
PXG_TEST(npSbPrim_tetPlane)
{
    const uint32_t NC = 1024, CAP = 3200, SBID = 4;
    std::mt19937 rng(1000701);
    std::vector<SbCase> cases(NC);
    std::vector<PxgDeformablePairInfo> rfilter;
    size_t boundaryN = 0, filteredN = 0, staticN = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        SbCase& cs = cases[i];
        std::memset(&cs, 0, sizeof(cs));
        const F4 q = exactQuat(rng);
        const V3 tp = { float(int(rng() % 9)) - 4.0f, float(int(rng() % 9)) - 4.0f, float(int(rng() % 9)) - 4.0f };
        cs.q = q;
        cs.p = { tp.x, tp.y, tp.z, 0 };
        const float cd = 0.25f + float(i % 4) * 0.25f;
        const float rd = 0.0625f * float(1 + i % 5);

        // choose plane-space x values per vert, then map back to world exactly:
        // v = q.rotate(planeSpace) + p (signed permutation => exact)
        float px[4];
        for (int k = 0; k < 4; ++k)
        {
            switch ((i + k) % 5)
            {
            case 0: px[k] = cd; ++boundaryN; break;                 // EXACTLY on the inclusive gate
            case 1: px[k] = cd - grid64(rng, 0.25f, 1.5f); break;   // accepted (may be negative = penetrating)
            case 2: px[k] = cd + grid64(rng, 0.25f, 2.0f); break;   // rejected
            case 3: px[k] = -grid64(rng, 0.5f, 2.0f); break;        // deep accept
            default: px[k] = cd + 0.015625f; break;                 // just rejected (exact)
            }
        }
        V3 verts[4];
        for (int k = 0; k < 4; ++k)
        {
            const V3 ps = { px[k], grid64(rng, -2, 2), grid64(rng, -2, 2) };
            // rotate by q (forward): use rotateInv of conjugate = rotate
            const F4 qc = { -q.x, -q.y, -q.z, q.w };
            verts[k] = add(quatRotateInv(qc, ps), tp);
        }
        cs.a = { verts[0].x, verts[0].y, verts[0].z, cd };
        cs.b = { verts[1].x, verts[1].y, verts[1].z, rd };
        cs.c = { verts[2].x, verts[2].y, verts[2].z, 0 };
        cs.d = { verts[3].x, verts[3].y, verts[3].z, 0 };
        // Every other filtered / decoy case is a STATIC rigid, keyed on its cache ref.
        const bool isStatic = (i % 9 == 2 || i % 9 == 5) && (i % 2 == 1);
        const uint32_t rlo = isStatic ? 0xFFFFFFFFu : 0x300u + i % 13, rhi = 1 + i % 3;
        const uint32_t cacheRef = 40u + i % 6;
        cs.ids = { rlo, rhi, SBID, i };
        cs.ids2 = { (0x21u + i % 9) | ((i % 5 == 1) ? 0x80000000u : 0u), cacheRef, 0, 0 };

        const uint64_t r64 = rigid64(rlo, rhi);
        const uint64_t key = filterKey(r64, cacheRef);
        if (i % 9 == 2) { rfilter.push_back({ key, PxgEncodeDeformableTetrahedron(SBID, i) }); ++filteredN; if (isStatic) ++staticN; }
        else if (i % 9 == 5) // near misses: the next tet, and a static's RAW node index
            rfilter.push_back({ isStatic ? r64 : key, PxgEncodeDeformableTetrahedron(SBID, isStatic ? i : i + 1) });
    }
    std::sort(rfilter.begin(), rfilter.end(), pairLess);

    // exact reference
    struct Ref { bool filtered; int hits[4]; float pen[4]; V3 n; };
    std::vector<Ref> ref(NC);
    uint32_t total = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const SbCase& cs = cases[i];
        Ref& ro = ref[i];
        ro.filtered = findRef(rfilter, filterKey(rigid64(cs.ids.x, cs.ids.y), cs.ids2.y),
                              PxgEncodeDeformableTetrahedron(SBID, i));
        if (ro.filtered) { ro.hits[0] = ro.hits[1] = ro.hits[2] = ro.hits[3] = 0; continue; }
        const V3 verts[4] = { v3(cs.a), v3(cs.b), v3(cs.c), v3(cs.d) };
        for (int k = 0; k < 4; ++k)
        {
            const V3 ps = quatRotateInv(cs.q, sub(verts[k], v3(cs.p)));
            ro.hits[k] = cs.a.w >= ps.x ? 1 : 0;
            ro.pen[k] = ps.x;
            if (ro.hits[k]) ++total;
        }
        ro.n = basisVector0(cs.q);
    }

    const SbRun r = runSb("sbTetPlane", cases, rfilter, CAP, CAP);
    PXG_CHECK_MSG(r.wc == total, "tetPlane: counter %u != %u", r.wc, total);

    // slot map: the tet ref is unique per case; per-case runs contiguous in vert order
    std::vector<std::vector<uint32_t>> slots(NC);
    size_t mm = 0;
    for (uint32_t s = 0; s < std::min(r.wc, CAP); ++s)
    {
        const uint32_t caseIdx = PxgGetDeformableFeatureId(r.in[s].ref1);
        if (caseIdx >= NC || r.in[s].ref1 != PxgEncodeDeformableTetrahedron(SBID, caseIdx)) { ++mm; continue; }
        slots[caseIdx].push_back(s);
    }
    for (uint32_t i = 0; i < NC; ++i)
    {
        const Ref& ro = ref[i];
        const uint32_t want = ro.filtered ? 0 : uint32_t(ro.hits[0] + ro.hits[1] + ro.hits[2] + ro.hits[3]);
        if ((r.dbg[i].x != 0.0f) != ro.filtered) ++mm;
        if (slots[i].size() != want) { ++mm; continue; }
        int vi = 0;
        for (uint32_t j = 0; j < want; ++j)
        {
            while (!ro.hits[vi]) ++vi;
            const uint32_t s = slots[i][j];
            if (j > 0 && s != slots[i][j - 1] + 1) ++mm; // contiguous run
            const V3 vert = vi == 0 ? v3(cases[i].a) : vi == 1 ? v3(cases[i].b) : vi == 2 ? v3(cases[i].c) : v3(cases[i].d);
            if (r.pt[s].x != vert.x || r.pt[s].y != vert.y || r.pt[s].z != vert.z || r.pt[s].w != cases[i].b.w) ++mm;
            if (r.np[s].x != ro.n.x || r.np[s].y != ro.n.y || r.np[s].z != ro.n.z || r.np[s].w != ro.pen[vi]) ++mm;
            // ref0 and the sort key are the rigid's NODE, static or not
            const uint64_t r64 = rigid64(cases[i].ids.x, cases[i].ids.y);
            if (r.in[s].ref0 != r64) ++mm;
            if (r.mat[s] != (cases[i].ids2.x & 0xFFFFu)) ++mm;
            if (r.by[s] != r64 || r.tb[s] != PxgGetContactSortKey(r64) || r.ix[s] != s) ++mm;
            if (!baryUntouched(r, s)) ++mm; // writeVolumeContact writes no barycentric
            ++vi;
        }
    }
    PXG_CHECK_MSG(mm == 0, "tetPlane: %zu mismatches", mm);
    PXG_CHECK_MSG(boundaryN > 500 && filteredN > 80 && staticN > 40,
                  "tetPlane coverage thin (b=%zu f=%zu s=%zu)", boundaryN, filteredN, staticN);

    // capacity: counter unchanged, stores gated
    const uint32_t TINY = total / 3;
    const SbRun rc = runSb("sbTetPlane", cases, rfilter, TINY, CAP);
    size_t bad = rc.wc != total ? 1 : 0;
    for (uint32_t s = TINY; s < CAP; ++s)
        if (rc.ix[s] != POISON || rc.pt[s].x != -9.0f || rc.in[s].ref0 != INFOPOISON || rc.mat[s] != MATPOISON ||
            rc.by[s] != KEYPOISON) ++bad;
    PXG_CHECK_MSG(bad == 0, "tetPlane cap: %zu leaks", bad);
}

// ============================================================================
// sbEmit — EXACT
// ============================================================================
PXG_TEST(npSbPrim_emit)
{
    const uint32_t NC = 512, CAP = 700;
    std::mt19937 rng(1000702);
    std::vector<SbCase> cases(NC);
    size_t flipN = 0, offN = 0, dropN = 0;
    const uint32_t SBID = 11;
    for (uint32_t i = 0; i < NC; ++i)
    {
        SbCase& cs = cases[i];
        std::memset(&cs, 0, sizeof(cs));
        const V3 A = { float(int(rng() % 9)) - 4, float(int(rng() % 9)) - 4, float(int(rng() % 9)) - 4 };
        const V3 B = add(A, V3{ float(int(rng() % 5)) - 2, float(int(rng() % 5)) - 2, float(int(rng() % 5)) - 2 });
        const V3 n = { float(int(rng() % 5)) - 2, float(int(rng() % 5)) - 2, float(int(rng() % 5)) - 2 };
        const float penOffset = (i % 3 == 1) ? -float(1 + i % 4) * 0.25f : 0.0f; // -radius style
        if (penOffset != 0) ++offN;
        cs.a = { A.x, A.y, A.z, 0 };
        cs.b = { B.x, B.y, B.z, 0 };
        cs.c = { n.x, n.y, n.z, 0 };
        cs.d = { penOffset, 0.125f * float(1 + i % 4), 0, 0 };
        const bool intersect = (i % 6) != 4;
        if (!intersect) ++dropN;
        const bool flip = (i % 4) == 2;
        if (flip) ++flipN;
        cs.ids = { 0x1000u + i, 5 + i % 7, 9000 + i, (0x31u + i % 5) | ((i % 7 == 3) ? 0x80000000u : 0u) };
        cs.ids2 = { intersect ? 1u : 0u, flip ? 1u : 0u, SBID, 0 };
    }

    uint32_t total = 0;
    for (uint32_t i = 0; i < NC; ++i) if (cases[i].ids2.x) ++total;

    const SbRun r = runSb("sbEmit", cases, {}, CAP, CAP);
    PXG_CHECK_MSG(r.wc == total, "emit: counter %u != %u", r.wc, total);

    std::vector<int> slotOf(NC, -1);
    size_t mm = 0;
    for (uint32_t s = 0; s < std::min(r.wc, CAP); ++s)
    {
        const uint32_t caseIdx = PxgGetDeformableFeatureId(r.in[s].ref1) - 9000;
        if (r.in[s].ref1 != PxgEncodeDeformableTetrahedron(SBID, caseIdx + 9000)) { ++mm; continue; }
        if (caseIdx >= NC || slotOf[caseIdx] != -1) { ++mm; continue; }
        slotOf[caseIdx] = (int)s;
    }
    for (uint32_t i = 0; i < NC; ++i)
    {
        const SbCase& cs = cases[i];
        if (!cs.ids2.x) { if (slotOf[i] != -1) ++mm; continue; }
        if (slotOf[i] == -1) { ++mm; continue; }
        const uint32_t s = (uint32_t)slotOf[i];
        const V3 A = v3(cs.a), B = v3(cs.b), n = v3(cs.c);
        const float pen = dot3(sub(B, A), n) + cs.d.x;
        const bool flip = cs.ids2.y != 0;
        if (r.pt[s].x != B.x || r.pt[s].y != B.y || r.pt[s].z != B.z || r.pt[s].w != cs.d.y) ++mm;
        const V3 en = flip ? neg(n) : n;
        if (r.np[s].x != en.x || r.np[s].y != en.y || r.np[s].z != en.z || r.np[s].w != pen) ++mm;
        const uint64_t r64 = rigid64(cs.ids.x, cs.ids.y);
        if (r.in[s].ref0 != r64 || r.mat[s] != (cs.ids.w & 0xFFFFu)) ++mm;
        if (r.by[s] != r64 || r.tb[s] != PxgGetContactSortKey(r64) || r.ix[s] != s) ++mm;
        if (!baryUntouched(r, s)) ++mm; // no barycentric
    }
    PXG_CHECK_MSG(mm == 0, "emit: %zu mismatches", mm);

    // capacity: UNGATED counter overshoots; sentinel drops all writes >= max
    const uint32_t TINY = total / 2;
    const SbRun rc = runSb("sbEmit", cases, {}, TINY, CAP);
    size_t bad = rc.wc != total ? 1 : 0; // counter intentionally overshoots the cap
    for (uint32_t s = TINY; s < CAP; ++s)
        if (rc.ix[s] != POISON || rc.pt[s].x != -9.0f || rc.in[s].ref0 != INFOPOISON ||
            rc.tb[s] != POISON || rc.by[s] != KEYPOISON || rc.mat[s] != MATPOISON) ++bad;
    PXG_CHECK_MSG(bad == 0, "emit cap: %zu leaks", bad);
    PXG_CHECK_MSG(flipN > 100 && offN > 140 && dropN > 60,
                  "emit coverage thin (f=%zu o=%zu d=%zu)", flipN, offN, dropN);
}

// ============================================================================
// shared tet fixture builder for sbParticle / sbClothVert
// ============================================================================
struct TetFix
{
    V3 a, b, c, d, p;
    uint32_t surfaceMask;
    bool wantInside, bigCd;
};
static TetFix makeTetFix(std::mt19937& rng, uint32_t i)
{
    TetFix f;
    std::uniform_real_distribution<float> uf(0.0f, 1.0f);
    const V3 base = { grid64(rng, -3, 3), grid64(rng, -3, 3), grid64(rng, -3, 3) };
    // well-conditioned tet on the 1/64 grid
    f.a = base;
    f.b = add(base, V3{ 1.0f + grid64(rng, 0, 0.5f), grid64(rng, 0, 0.25f), grid64(rng, 0, 0.25f) });
    f.c = add(base, V3{ grid64(rng, 0, 0.25f), 1.0f + grid64(rng, 0, 0.5f), grid64(rng, 0, 0.25f) });
    f.d = add(base, V3{ grid64(rng, 0, 0.25f), grid64(rng, 0, 0.25f), 1.0f + grid64(rng, 0, 0.5f) });
    f.surfaceMask = 1 + rng() % 15;
    f.wantInside = false;
    f.bigCd = false;
    const V3 cen = mul(add(add(f.a, f.b), add(f.c, f.d)), 0.25f);
    switch (i % 10)
    {
    case 0:
    case 5: // inside: blend toward center
        f.p = add(mul(cen, 0.75f), mul(f.a, 0.25f));
        f.wantInside = true;
        f.surfaceMask = 0xf;
        break;
    case 1: // outside near a face, within contact distance
        if ((i / 10) % 2 == 1)
        {
            // GRAZING: p a few 1e-5 below the interior of face abc — sqDist
            // lands under the 1e-8 rescue clause, which must replace the
            // normalized-noise direction by n2
            f.p = add(mul(add(add(f.a, f.b), f.c), 1.0f / 3.0f), V3{ 0, 0, -5e-5f - 3e-5f * uf(rng) });
            f.surfaceMask = 1;
            break;
        }
        f.p = add(cen, V3{ -0.4f - 0.2f * uf(rng), -0.4f, -0.4f });
        f.surfaceMask = 0xf;
        break;
    case 6: // FACE-OPPOSED: p straight below face abc (projection inside the
            // triangle) — v points up into the tet, and closestPtTetrahedron's
            // normals are flipped INWARD (toward the remaining vertex), so
            // dot(v, n2) ~ +1 feeds the v-direction / keep-n paths
        f.p = { cen.x, cen.y, f.a.z - 0.5f - 0.3f * uf(rng) };
        f.surfaceMask = 1;
        f.bigCd = true;
        break;
    case 2:
    case 7: // outside far (reject)
        f.p = add(cen, V3{ 3.0f + uf(rng), 3.0f, 3.0f });
        break;
    case 3: // ON-BOUNDARY inside: p exactly on edge ab (two products exactly 0,
            // inclusive >= keeps it INSIDE) while the hint mask excludes the
            // adjacent faces — the inside/outside paths then produce DIFFERENT
            // pen formulas, pinning the inclusive inside test
        if ((i / 10) % 2 == 0)
        {
            f.p = mul(add(f.a, f.b), 0.5f);
            f.surfaceMask = 8;
            f.bigCd = true;
            f.wantInside = true;
            break;
        }
        f.p = add(f.b, V3{ 0.3f + 0.2f * uf(rng), -0.3f, -0.3f });
        break;
    case 8: // outside near an edge/corner
        f.p = add(f.b, V3{ 0.3f + 0.2f * uf(rng), -0.3f, -0.3f });
        break;
    default: // exactly on vertex a: sqDist == 0 path (n -> n2)
        f.p = f.a;
        f.surfaceMask = 0xf;
        f.wantInside = true; // on the corner: all four products are exactly 0 => inside
        break;
    }
    return f;
}

// ============================================================================
// sbParticle — taint-based
// ============================================================================
PXG_TEST(npSbPrim_particle)
{
    const uint32_t NC = 1024, CAP = 1400, SBID = 6, SYSID = 3;
    std::mt19937 rng(1000703);
    std::vector<SbCase> cases(NC);
    // 5.12: no particle filter chain on this path (and no rest distance) -- the
    // surface hint is the only gate in front of the geometry.
    size_t maskZeroN = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        SbCase& cs = cases[i];
        std::memset(&cs, 0, sizeof(cs));
        const TetFix f = makeTetFix(rng, i);
        // on-edge-inside family: SMALL cd pins that the inside path has NO
        // cDistance gate (closest hint face lies far beyond cd, contact must
        // still be emitted); face-opposed family needs the big cd
        cs.a = { f.a.x, f.a.y, f.a.z, f.bigCd ? (f.wantInside ? 0.125f : 2.5f) : 0.5f + 0.25f * float(i % 3) };
        cs.b = { f.b.x, f.b.y, f.b.z, 0.0625f * float(1 + i % 4) };
        cs.c = { f.c.x, f.c.y, f.c.z, 0 };
        cs.d = { f.d.x, f.d.y, f.d.z, 0 };
        cs.p = { f.p.x, f.p.y, f.p.z, 0 };
        cs.ids = { SBID, i, SYSID, (i * 5 + 29) % NC /*sorted, decoupled from tetIdx*/ };
        cs.ids2 = { (i % 13 == 7) ? 0u : f.surfaceMask, 0, 0, 0 };
        if (i % 13 == 7) ++maskZeroN;
    }

    struct Ref { bool skipped = false, wrote = false, inside = false; V3 p, n; float pen; };
    std::vector<Ref> ref(NC);
    std::vector<int> taint(NC, 0);
    size_t insideN = 0, outsideN = 0, rejN = 0, keepN = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const SbCase& cs = cases[i];
        Slack sl;
        Ref& ro = ref[i];
        const uint32_t surfaceMask = cs.ids2.x;
        ro.skipped = surfaceMask == 0;
        if (ro.skipped) continue;

        const V3 a = v3(cs.a), b = v3(cs.b), c = v3(cs.c), d = v3(cs.d), p = v3(cs.p);
        const F4 res = pointOutsideOfPlane4Ref(p, a, b, c, d);
        V3 closestP = {}, n2 = {};
        float sqDist = 3.402823466e38f;
        closestPtTetrahedronRef(p, surfaceMask, a, b, c, d, n2, closestP, sqDist, sl);

        V3 n = neg(sub(p, closestP));
        float pen;
        const bool inside = res.x >= 0 && res.y >= 0 && res.z >= 0 && res.w >= 0;
        // exact sign products on grid inputs: no taint on the inside test itself
        if (inside)
        {
            pen = dot3(n, n2);
            n = n2;
        }
        else
        {
            const float cd2 = cs.a.w * cs.a.w;
            sl.note(sqDist - cd2, 2e-4f);
            if (sqDist > cd2) { ro.wrote = false; taint[i] = sl.near; if (!taint[i]) ++rejN; continue; }
            pen = pxNormalizeRef(n);
            sl.note(sqDist - 1e-8f, 1e-9f);
            sl.note(dot3(n, n2) - 0.8f, 2e-4f);
            if (sqDist < 1e-8f || dot3(n, n2) < 0.8f)
                n = n2;
            else
                ++keepN; // normalized contact direction kept (not replaced by n2)
        }
        ro.wrote = true;
        ro.inside = inside;
        ro.p = p;
        ro.n = n;
        ro.pen = pen;
        taint[i] = sl.near;
        if (!taint[i]) { if (inside) ++insideN; else ++outsideN; }
    }

    uint32_t sumClean = 0, numTainted = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        if (ref[i].skipped) continue;
        if (taint[i]) { ++numTainted; continue; }
        if (ref[i].wrote) ++sumClean;
    }

    const SbRun r = runSb("sbParticle", cases, {}, CAP, CAP);
    std::vector<int> slotOf(NC, -1);
    size_t mm = 0;
    for (uint32_t s = 0; s < std::min(r.wc, CAP); ++s)
    {
        const uint32_t caseIdx = PxgGetDeformableFeatureId(r.in[s].ref1);
        if (caseIdx >= NC || r.in[s].ref1 != PxgEncodeDeformableTetrahedron(SBID, caseIdx) || slotOf[caseIdx] != -1)
        { ++mm; continue; }
        slotOf[caseIdx] = (int)s;
        // ref0 and the sort key are the particle ref of the SORTED index
        const uint64_t pref = PxgEncodeParticleRef(SYSID, (caseIdx * 5 + 29) % NC);
        if (r.in[s].ref0 != pref || r.by[s] != pref || r.tb[s] != PxgGetContactSortKey(pref) || r.ix[s] != s) ++mm;
        if (r.mat[s] != MATPOISON || !baryUntouched(r, s)) ++mm;   // no material stream, no barycentric
    }
    for (uint32_t i = 0; i < NC; ++i)
    {
        const Ref& ro = ref[i];
        if (ro.skipped)
        {
            if (r.dbg[i].x != 1.0f || slotOf[i] != -1) ++mm;
            continue;
        }
        if (taint[i]) continue;
        if ((r.dbg[i].y != 0.0f) != ro.wrote) { ++mm; continue; }
        if (!ro.wrote) { if (slotOf[i] != -1) ++mm; continue; }
        if ((r.dbg[i].w != 0.0f) != ro.inside) ++mm; // inclusive inside test
        if (slotOf[i] == -1) { ++mm; continue; }
        const uint32_t s = (uint32_t)slotOf[i];
        if (r.pt[s].x != ro.p.x || r.pt[s].y != ro.p.y || r.pt[s].z != ro.p.z || r.pt[s].w != 0.0f) ++mm;
        if (!almostEqual(r.np[s].x, ro.n.x, 2e-3f, 2e-3f) || !almostEqual(r.np[s].y, ro.n.y, 2e-3f, 2e-3f) ||
            !almostEqual(r.np[s].z, ro.n.z, 2e-3f, 2e-3f) || !almostEqual(r.np[s].w, ro.pen, 2e-3f, 2e-3f)) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "particle: %zu mismatches", mm);
    PXG_CHECK_MSG(r.wc >= sumClean && r.wc <= sumClean + numTainted,
                  "particle: counter %u outside [%u, %u]", r.wc, sumClean, sumClean + numTainted);
    PXG_CHECK_MSG(insideN > 200 && outsideN > 150 && rejN > 100 && maskZeroN > 50 && keepN > 30,
                  "particle coverage thin (i=%zu o=%zu r=%zu z=%zu k=%zu)",
                  insideN, outsideN, rejN, maskZeroN, keepN);
}

// ============================================================================
// sbClothVert — taint-based
// ============================================================================
PXG_TEST(npSbPrim_clothVert)
{
    const uint32_t NC = 1024, CAP = 1400, SBID = 9, CLOTHID = 5;
    std::mt19937 rng(1000704);
    std::vector<SbCase> cases(NC);
    std::vector<PxgDeformablePairInfo> nrf;   // the softbody's mVolumeSurfaceFilterPairs
    size_t f1N = 0, f2N = 0, f3N = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        SbCase& cs = cases[i];
        std::memset(&cs, 0, sizeof(cs));
        const TetFix f = makeTetFix(rng, i);
        cs.a = { f.a.x, f.a.y, f.a.z, f.bigCd ? (f.wantInside ? 0.125f : 2.5f) : 0.5f + 0.25f * float(i % 3) };
        cs.b = { f.b.x, f.b.y, f.b.z, 0 };
        cs.c = { f.c.x, f.c.y, f.c.z, 0 };
        cs.d = { f.d.x, f.d.y, f.d.z, 0 };
        cs.p = { f.p.x, f.p.y, f.p.z, 0 };
        cs.ids = { SBID, i, CLOTHID, (i * 7 + 13) % NC /*vertIndex, decoupled from tetIdx*/ };
        cs.ids2 = { (i % 17 == 11) ? 0u : f.surfaceMask, 0, 0, 0 };

        const uint32_t vIdx = (i * 7 + 13) % NC;
        const uint64_t tetRef = PxgEncodeDeformableTetrahedron(SBID, i);
        const uint64_t vertRef = PxgEncodeDeformableVertex(CLOTHID, vIdx);
        if (i % 9 == 2) { nrf.push_back({ PxgEncodeDeformableTetrahedron(SBID, ANY), vertRef }); ++f1N; }
        else if (i % 9 == 4) { nrf.push_back({ tetRef, PxgEncodeDeformableVertex(CLOTHID, ANY) }); ++f2N; }
        else if (i % 9 == 6) { nrf.push_back({ tetRef, vertRef }); ++f3N; }
        // near misses: the next vertex, the TRIANGLE of the same id, the pair reversed
        else if (i % 9 == 8) nrf.push_back({ tetRef, PxgEncodeDeformableVertex(CLOTHID, vIdx + 1) });
        else if (i % 9 == 1) nrf.push_back({ tetRef, PxgEncodeDeformableTriangle(CLOTHID, vIdx) });
        else if (i % 9 == 3) nrf.push_back({ vertRef, tetRef });
    }
    std::sort(nrf.begin(), nrf.end(), pairLess);

    struct Ref { bool skipped = false, wrote = false, inside = false; V3 cp, n; float pen; };
    std::vector<Ref> ref(NC);
    std::vector<int> taint(NC, 0);
    size_t insideN = 0, outsideN = 0, rejN = 0, vDirN = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const SbCase& cs = cases[i];
        Slack sl;
        Ref& ro = ref[i];
        const uint32_t vIdx = (i * 7 + 13) % NC;
        // findVolumeSurfaceVertexFilter: (ANY tet, vertex), (tet, ANY vertex), (tet, vertex)
        ro.skipped = findRef(nrf, PxgEncodeDeformableTetrahedron(SBID, ANY), PxgEncodeDeformableVertex(CLOTHID, vIdx)) ||
                     findRef(nrf, PxgEncodeDeformableTetrahedron(SBID, i), PxgEncodeDeformableVertex(CLOTHID, ANY)) ||
                     findRef(nrf, PxgEncodeDeformableTetrahedron(SBID, i), PxgEncodeDeformableVertex(CLOTHID, vIdx));
        if (!ro.skipped && cs.ids2.x == 0) ro.skipped = true; // surfaceMask gate (dbg code 2)
        if (ro.skipped) continue;

        const V3 a = v3(cs.a), b = v3(cs.b), c = v3(cs.c), d = v3(cs.d), p = v3(cs.p);
        const F4 res = pointOutsideOfPlane4Ref(p, a, b, c, d);
        V3 closestP = {}, n2 = {};
        float sqDist = 3.402823466e38f;
        closestPtTetrahedronRef(p, cs.ids2.x, a, b, c, d, n2, closestP, sqDist, sl);

        V3 v = sub(closestP, p);
        const float m = pxNormalizeRef(v);
        V3 n;
        float pen;
        const bool inside = res.x >= 0 && res.y >= 0 && res.z >= 0 && res.w >= 0;
        if (inside)
        {
            n = n2;
            pen = -m;
        }
        else
        {
            const float cd2 = cs.a.w * cs.a.w;
            sl.note(sqDist - cd2, 2e-4f);
            if (sqDist > cd2) { taint[i] = sl.near; if (!taint[i]) ++rejN; continue; }
            pen = m;
            sl.note(m - 1e-10f, 1e-11f);
            sl.note(dot3(v, n2) - 0.707f, 2e-4f);
            // unlike sbParticleCollision there is NO sqDist < 1e-8 rescue here:
            // a grazing contact keeps the normalized micro-vector as its
            // direction, which is ill-conditioned -- taint it
            sl.note(m, 2e-4f);
            const bool useV = m > 1e-10f && dot3(v, n2) > 0.707f;
            n = useV ? v : n2;
            if (useV) ++vDirN;
        }
        ro.wrote = true;
        ro.inside = inside;
        ro.cp = closestP;
        ro.n = n;
        ro.pen = pen;
        taint[i] = sl.near;
        if (!taint[i]) { if (inside) ++insideN; else ++outsideN; }
    }

    uint32_t sumClean = 0, numTainted = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        if (ref[i].skipped) continue;
        if (taint[i]) { ++numTainted; continue; }
        if (ref[i].wrote) ++sumClean;
    }

    const SbRun r = runSb("sbClothVert", cases, nrf, CAP, CAP);
    std::vector<int> slotOf(NC, -1);
    size_t mm = 0;
    for (uint32_t s = 0; s < std::min(r.wc, CAP); ++s)
    {
        const uint32_t caseIdx = PxgGetDeformableFeatureId(r.in[s].ref1);
        if (caseIdx >= NC || r.in[s].ref1 != PxgEncodeDeformableTetrahedron(SBID, caseIdx) || slotOf[caseIdx] != -1)
        { ++mm; continue; }
        slotOf[caseIdx] = (int)s;
        // ref0 = the cloth VERTEX (decoupled from the tet index); the DbDb writer has no
        // sort key, no material and -- for a vertex -- no barycentric
        if (r.in[s].ref0 != PxgEncodeDeformableVertex(CLOTHID, (caseIdx * 7 + 13) % NC)) ++mm;
        if (r.by[s] != KEYPOISON || r.tb[s] != POISON || r.ix[s] != POISON || r.mat[s] != MATPOISON) ++mm;
        if (!baryUntouched(r, s)) ++mm;
    }
    for (uint32_t i = 0; i < NC; ++i)
    {
        const Ref& ro = ref[i];
        if (ro.skipped)
        {
            if (r.dbg[i].x == 0.0f || slotOf[i] != -1) ++mm;
            continue;
        }
        if (taint[i]) continue;
        if ((r.dbg[i].y != 0.0f) != ro.wrote) { ++mm; continue; }
        if (!ro.wrote) { if (slotOf[i] != -1) ++mm; continue; }
        if ((r.dbg[i].w != 0.0f) != ro.inside) ++mm; // inclusive inside test
        if (slotOf[i] == -1) { ++mm; continue; }
        const uint32_t s = (uint32_t)slotOf[i];
        if (!almostEqual(r.pt[s].x, ro.cp.x, 2e-3f, 2e-3f) || !almostEqual(r.pt[s].y, ro.cp.y, 2e-3f, 2e-3f) ||
            !almostEqual(r.pt[s].z, ro.cp.z, 2e-3f, 2e-3f) || r.pt[s].w != 0.0f) ++mm;
        if (!almostEqual(r.np[s].x, ro.n.x, 2e-3f, 2e-3f) || !almostEqual(r.np[s].y, ro.n.y, 2e-3f, 2e-3f) ||
            !almostEqual(r.np[s].z, ro.n.z, 2e-3f, 2e-3f) || !almostEqual(r.np[s].w, ro.pen, 2e-3f, 2e-3f)) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "clothVert: %zu mismatches", mm);
    PXG_CHECK_MSG(r.wc >= sumClean && r.wc <= sumClean + numTainted,
                  "clothVert: counter %u outside [%u, %u]", r.wc, sumClean, sumClean + numTainted);
    PXG_CHECK_MSG(insideN > 200 && outsideN > 120 && rejN > 100 && vDirN > 40 &&
                  f1N > 80 && f2N > 80 && f3N > 80,
                  "clothVert coverage thin (i=%zu o=%zu r=%zu v=%zu f=%zu/%zu/%zu)",
                  insideN, outsideN, rejN, vDirN, f1N, f2N, f3N);
}

} // namespace
