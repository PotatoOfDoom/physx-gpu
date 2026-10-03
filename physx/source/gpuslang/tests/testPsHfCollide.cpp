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

// Verification of psHfCollide.slang (psHeightfieldMidphaseCollision — the
// per-particle particle-vs-heightfield contact core of
// particleSystemHFMidPhaseCG.cu).  Per non-tainted particle the temp
// contact list is DETERMINISTIC (cell-order accumulation, cap 6, no
// replacement), so the one-way slots (static dispatch) and the writer
// contacts (dynamic dispatch) compare directly.  Slack taints a particle
// when any of: the local-bound row/column floors sit near integers, the
// per-cell height culls (con0/con1) are within rounding, or any
// particleTriangleTest gate is ambiguous.  Faithful quirks pinned: the
// static path's PLAIN count store after the atomicAdd, the writer's
// src = 0 restart, hole materials and the tessellation-flag diagonal.
// The negative-scale min/max swap and the getTriangle handedness swap are
// NOT dispatched: PxHeightFieldGeometry::isValid() rejects scales below
// PX_MIN_HEIGHTFIELD_XZ_SCALE, and for a hypothetical negative scale the
// kernel's |scale| division + bare swap leaves min > max so maxRow - minRow
// underflows (defensive dead code upstream, transcribed but untestable).
// Dispatch matrix: static TGS-CCD (one-way slots + node indices), dynamic
// TGS-CCD (writer + all six radix staging arrays), triple static (the
// plain-store quirk), non-CCD, PGS-CCD static (forward-projection constant
// 1 vs 2), diffuse dynamic (writer suppression leaves everything pristine),
// diffuse static (diffuse constant with the TGS flag set), and a
// small-capacity dynamic run (the writeContact capacity guard: counter
// overshoots, slots beyond maxContacts stay untouched).  The grid is
// RECTANGULAR (rows != cols) and 10% of particles straddle or lie beyond
// the four grid edges so row/column transpositions, the getMin*/getMax*
// clamps and the early-outs are all output-relevant.  materialIndex1
// samples include reserved-bit values (0xFF must read as eHOLE via the
// PxBitAndByte 7-bit mask).

#include "pxgTestFramework.h"
#include "pxgPsHfBoundCases.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct PhMesh { F4 hfP, hfQ, scale, misc; U4 dims, rigid; };
struct PsC { F4 normal_pen; uint32_t rLo, rHi, pLo, pHi; };
struct PhParams { uint32_t numParticles, maxContacts, flags, particleSystemId; };

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 muls(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 v3(F4 f) { return { f.x, f.y, f.z }; }

static const float MAXF = 3.4028235e38f;
static const uint32_t MAXMESH = 6, MAXPART = 12, HOLE = 127;

struct Slack { bool near = false; void noteSc(float l, float r, float scale) { if (std::fabs(l - r) < 1e-5f * scale) near = true; } };
static float l1n(V3 v) { return std::fabs(v.x) + std::fabs(v.y) + std::fabs(v.z); }

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

// mirrors PxMat33(const PxQuat&) (PxMat33.h:136-163)
static void refMat33FromQuat(F4 q, V3& col0, V3& col1, V3& col2)
{
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    col0 = { 1.0f - yy - zz, xy + zw, xz - yw };
    col1 = { xy - zw, 1.0f - xx - zz, yz + xw };
    col2 = { xz + yw, yz - xw, 1.0f - xx - yy };
}

// Which spelling of PxBounds3::transformFast(pose.getInverse(), worldBound)
// (particleSystemHFMidPhaseCG.cu:122) the reference uses.  Literal is the CUDA and
// the only one the GPU is compared against; the others are the pre-P9.94c port's
// pieces and the dropped-conjugate slip, kept ONLY so psHfCollide_boundLiteral can
// assert that its directed inputs tell each of them apart from the literal one.
enum class HfBoundSpelling
{
    Literal,          // (inv.q.rotate(c) + inv.p, PxMat33(inv.q)), inv = (q.rotateInv(-p), conj q)
    CentreRotateInv,  // centre q.rotateInv(c - p), basis literal
    BasisRotateInv,   // centre literal, basis columns q.rotateInv(e_i)
    OldPort,          // both of the above (the port before P9.94c)
    BasisNoConj       // literal, but PxMat33(q) instead of PxMat33(conj q)
};

static void refLocalBound(const PhMesh& mesh, V3 wmin, V3 wmax, HfBoundSpelling s, V3& lmin, V3& lmax)
{
    const F4 q = mesh.hfQ;
    const V3 wc = muls(add(wmin, wmax), 0.5f), we = muls(sub(wmax, wmin), 0.5f);
    const bool oldCentre = s == HfBoundSpelling::CentreRotateInv || s == HfBoundSpelling::OldPort;
    const bool oldBasis = s == HfBoundSpelling::BasisRotateInv || s == HfBoundSpelling::OldPort;
    V3 lc, le;
    if (oldCentre)
        lc = quatRotateInv(q, sub(wc, v3(mesh.hfP)));
    else
    {
        // PxTransform::getInverse() (PxTransform.h:114-118), then inv.transform(c)
        // (PxTransform.h:128-132)
        const F4 invQ = { -q.x, -q.y, -q.z, q.w };
        lc = add(quatRotate(invQ, wc), quatRotateInv(q, V3{ -mesh.hfP.x, -mesh.hfP.y, -mesh.hfP.z }));
    }
    if (oldBasis)
    {
        const V3 c0 = quatRotateInv(q, V3{ 1, 0, 0 });
        const V3 c1 = quatRotateInv(q, V3{ 0, 1, 0 });
        const V3 c2 = quatRotateInv(q, V3{ 0, 0, 1 });
        le = { std::fabs(c0.x) * we.x + std::fabs(c1.x) * we.y + std::fabs(c2.x) * we.z,
               std::fabs(c0.y) * we.x + std::fabs(c1.y) * we.y + std::fabs(c2.y) * we.z,
               std::fabs(c0.z) * we.x + std::fabs(c1.z) * we.y + std::fabs(c2.z) * we.z };
    }
    else
    {
        // PxBounds3::basisExtent(center, PxMat33(inv.q), extent) (PxBounds3.h:301-314)
        const F4 invQ = { -q.x, -q.y, -q.z, q.w };
        V3 b0, b1, b2;
        refMat33FromQuat(s == HfBoundSpelling::BasisNoConj ? q : invQ, b0, b1, b2);
        const V3 c0 = muls(b0, we.x), c1 = muls(b1, we.y), c2 = muls(b2, we.z);
        le = { std::fabs(c0.x) + std::fabs(c1.x) + std::fabs(c2.x),
               std::fabs(c0.y) + std::fabs(c1.y) + std::fabs(c2.y),
               std::fabs(c0.z) + std::fabs(c1.z) + std::fabs(c2.z) };
    }
    lmin = sub(lc, le);
    lmax = add(lc, le);
}

static V3 refTriClosest(V3 p, V3 a, V3 b, V3 c)
{
    const V3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    const float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0 && d2 <= 0) return a;
    const V3 bp = sub(p, b);
    const float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0 && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) { const float v = d1 / (d1 - d3); return add(a, muls(ab, v)); }
    const V3 cp = sub(p, c);
    const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0 && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) { const float w = d2 / (d2 - d6); return add(a, muls(ac, w)); }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
    { const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return add(b, muls(sub(c, b), w)); }
    const float denom = 1.0f / (va + vb + vc);
    return add(add(a, muls(ab, vb * denom)), muls(ac, vc * denom));
}

static bool refPTT(V3& normal, float& distance, V3 pp, V3 cvp, float cvr,
                   V3 a, V3 b, V3 c, bool ccd, Slack& sl)
{
    const V3 cn = cross3(sub(b, a), sub(c, a));
    const float m = dot3(cn, cn);
    const float blow = l1n(sub(b, a)) * l1n(sub(c, a)) / (std::sqrt(m) + 1e-30f);
    const V3 tn = m > 0 ? muls(cn, 1.0f / std::sqrt(m)) : V3{ 0, 0, 0 };
    const float pd = dot3(sub(cvp, a), tn);
    sl.noteSc(pd, cvr, (l1n(cvp) + l1n(a)) * (1.0f + blow) + 1.0f);
    if (pd < cvr)
    {
        V3 closest = refTriClosest(cvp, a, b, c);
        V3 dv0 = sub(cvp, closest);
        float distSq = dot3(dv0, dv0);
        sl.noteSc(distSq, cvr * cvr, l1n(dv0) * (l1n(cvp) + l1n(closest)) + distSq + cvr * cvr + 1e-3f);
        if (distSq <= cvr * cvr)
        {
            const float ppd = dot3(sub(pp, a), tn);
            sl.noteSc(ppd, 0.0f, (l1n(pp) + l1n(a)) * (1.0f + blow) + 1e-3f);
            if (ppd > 0.0f)
            {
                if (ccd)
                {
                    closest = refTriClosest(pp, a, b, c);
                    const V3 dv = sub(pp, closest);
                    distSq = dot3(dv, dv);
                }
                const V3 dv = sub(pp, closest);
                if (distSq != 0.0f) sl.noteSc(std::sqrt(distSq), 0.0f, 1000.0f);
                normal = (distSq > 0.0f) ? muls(dv, 1.0f / std::sqrt(distSq)) : tn;
                distance = std::sqrt(distSq);
                return true;
            }
        }
    }
    return false;
}

// ---- heightfield helpers (mirroring the shader) ---------------------------
struct Hf
{
    std::vector<uint32_t> samples;
    uint32_t rows, cols;
    float getHeight(uint32_t v) const { return float(int32_t(samples[v] << 16) >> 16); }
    bool shared0(uint32_t v) const { return (samples[v] & 0x800000u) != 0; }
    // PxBitAndByte reads strip the high bit (tess flag resp. reserved flag)
    uint32_t mat0(uint32_t v) const { return (samples[v] >> 16) & 0x7fu; }
    uint32_t mat1(uint32_t v) const { return (samples[v] >> 24) & 0x7fu; }
    V3 getVertex(uint32_t v) const { return { float(v / cols), getHeight(v), float(v % cols) }; }
};

static void hfTriVerts(const Hf& hf, uint32_t triIdx, V3 scale, V3& v0, V3& v1, V3& v2)
{
    const uint32_t cell = triIdx >> 1;
    const bool first = (triIdx & 1) == 0;
    uint32_t i0, i1, i2;
    if (hf.shared0(cell))
    {
        if (first) { i0 = cell + hf.cols; i1 = cell; i2 = cell + hf.cols + 1; }
        else       { i0 = cell + 1; i1 = cell + hf.cols + 1; i2 = cell; }
    }
    else
    {
        if (first) { i0 = cell; i1 = cell + 1; i2 = cell + hf.cols; }
        else       { i0 = cell + hf.cols + 1; i1 = cell + hf.cols; i2 = cell + 1; }
    }
    bool wrongHanded = false;
    if (scale.z < 0) wrongHanded = !wrongHanded;
    if (scale.x < 0) wrongHanded = !wrongHanded;
    const V3 A = hf.getVertex(i0);
    const V3 B = hf.getVertex(wrongHanded ? i2 : i1);
    const V3 C = hf.getVertex(wrongHanded ? i1 : i2);
    v0 = { A.x * scale.x, A.y * scale.y, A.z * scale.z };
    v1 = { B.x * scale.x, B.y * scale.y, B.z * scale.z };
    v2 = { C.x * scale.x, C.y * scale.y, C.z * scale.z };
}

static uint32_t clampI(int v, int lo, int hi) { return (uint32_t)std::min(std::max(v, lo), hi); }

// full reference for one particle against one heightfield mesh; returns the
// deterministic temp list
static uint32_t refParticle(const Hf& hf, const PhMesh& mesh, V3 cur, V3 pred,
                            bool ccd, bool tgs, bool diffuse, F4* temp, Slack& sl,
                            HfBoundSpelling spelling = HfBoundSpelling::Literal)
{
    const float contactDist = mesh.scale.w, restDist = mesh.misc.x;
    V3 cvp = cur;
    float cvr = contactDist;
    if (ccd)
    {
        float scaleInv = diffuse ? 1.0f : (tgs ? 2.0f : 1.0f);
        const V3 off = muls(sub(pred, cur), scaleInv * 0.5f);
        cvp = add(cur, off);
        cvr = std::sqrt(dot3(off, off)) + contactDist;
    }
    const V3 wmin = sub(cvp, V3{ cvr, cvr, cvr }), wmax = add(cvp, V3{ cvr, cvr, cvr });
    V3 lmin, lmax;
    refLocalBound(mesh, wmin, wmax, spelling, lmin, lmax);
    const float oh = 1.0f / mesh.scale.y, orow = 1.0f / std::fabs(mesh.scale.x), ocol = 1.0f / std::fabs(mesh.scale.z);
    lmin = { lmin.x * orow, lmin.y * oh, lmin.z * ocol };
    lmax = { lmax.x * orow, lmax.y * oh, lmax.z * ocol };
    if (mesh.scale.x < 0) std::swap(lmin.x, lmax.x);
    if (mesh.scale.z < 0) std::swap(lmin.z, lmax.z);

    // row/col boundary floors: taint near-integer coordinates
    sl.noteSc(lmin.x, std::round(lmin.x), 20.0f);
    sl.noteSc(lmax.x, std::round(lmax.x), 20.0f);
    sl.noteSc(lmin.z, std::round(lmin.z), 20.0f);
    sl.noteSc(lmax.z, std::round(lmax.z), 20.0f);

    if (lmin.x > float(hf.rows - 1) || lmin.z > float(hf.cols - 1) || lmax.x < 0 || lmax.z < 0)
        return 0;
    const uint32_t minRow = clampI((int)std::floor(lmin.x), 0, (int)hf.rows - 2);
    const uint32_t maxRow = clampI((int)std::ceil(lmax.x), 0, (int)hf.rows - 1);
    const uint32_t minCol = clampI((int)std::floor(lmin.z), 0, (int)hf.cols - 2);
    const uint32_t maxCol = clampI((int)std::ceil(lmax.z), 0, (int)hf.cols - 1);
    if (2 * (maxCol - minCol) * (maxRow - minRow) == 0)
        return 0;

    const V3 cvpS = quatRotateInv(mesh.hfQ, sub(cvp, v3(mesh.hfP)));
    const V3 ppS = quatRotateInv(mesh.hfQ, sub(cur, v3(mesh.hfP)));
    const float miny = lmin.y, maxy = lmax.y;
    const uint32_t span = maxCol - minCol;
    uint32_t count = 0;

    const uint32_t total = (maxRow - minRow) * span * 2;
    for (uint32_t i = 0; i < total; ++i)
    {
        const uint32_t idx = i / 2;
        const uint32_t vertexIndex = (minRow + idx / span) * hf.cols + (minCol + idx % span);
        const float h0 = hf.getHeight(vertexIndex), h1 = hf.getHeight(vertexIndex + 1);
        const float h2 = hf.getHeight(vertexIndex + hf.cols), h3 = hf.getHeight(vertexIndex + hf.cols + 1);
        // height culls: taint near ties (heights are exact integers)
        sl.noteSc(maxy, h0, 30.0f); sl.noteSc(maxy, h1, 30.0f); sl.noteSc(maxy, h2, 30.0f); sl.noteSc(maxy, h3, 30.0f);
        sl.noteSc(miny, h0, 30.0f); sl.noteSc(miny, h1, 30.0f); sl.noteSc(miny, h2, 30.0f); sl.noteSc(miny, h3, 30.0f);
        const bool con0 = maxy < h0 && maxy < h1 && maxy < h2 && maxy < h3;
        const bool con1 = miny > h0 && miny > h1 && miny > h2 && miny > h3;
        if (con0 || con1) continue;
        const bool isMat1 = (i & 1) != 0;
        const uint32_t material = isMat1 ? hf.mat1(vertexIndex) : hf.mat0(vertexIndex);
        if (material == HOLE) continue;
        const uint32_t triIdx = isMat1 ? ((vertexIndex << 1) + 1) : (vertexIndex << 1);
        V3 a, b, c;
        hfTriVerts(hf, triIdx, v3(mesh.scale), a, b, c);
        V3 n;
        float d;
        if (refPTT(n, d, ppS, cvpS, cvr, a, b, c, ccd, sl) && count < MAXMESH)
        {
            const V3 wn = quatRotate(mesh.hfQ, n);
            temp[count++] = { wn.x, wn.y, wn.z, d - restDist };
        }
    }
    return count;
}

PXG_TEST(psHfCollide_core)
{
    std::mt19937 rng(93011);
    std::uniform_real_distribution<float> u01(0, 1);
    std::uniform_real_distribution<float> uv(-1, 1);

    // RECTANGULAR grid: rows != cols so any rows/cols transposition (vertex
    // stride, dims swap, clamp bounds, early-out bounds) changes the output
    const uint32_t ROWS = 24, COLS = 17, NP = 640, SYS = 6;
    Hf hf;
    hf.rows = ROWS;
    hf.cols = COLS;
    hf.samples.resize(ROWS * COLS);
    for (uint32_t r = 0; r < ROWS; ++r)
        for (uint32_t c = 0; c < COLS; ++c)
        {
            const int16_t h = (int16_t)(40.0f * std::sin(r * 0.7f) * std::cos(c * 0.5f) + 25.0f * uv(rng));
            uint32_t m0 = (rng() % 11 == 3) ? HOLE : (rng() % 5);
            // materialIndex1 is PxBitAndByte with a RESERVED high bit: store
            // 0xFF (hole + flag) and flagged non-holes so the 7-bit mask is
            // load-bearing (0xFF must read as eHOLE = 127)
            uint32_t m1 = (rng() % 11 == 7) ? (HOLE | 0x80u)
                        : (rng() % 13 == 5) ? (0x80u | (rng() % 5))
                                            : (rng() % 5);
            const uint32_t tess = (rng() % 2) ? 0x80u : 0u;
            hf.samples[r * COLS + c] = (uint32_t)(uint16_t)h | ((m0 | tess) << 16) | (m1 << 24);
        }

    PhMesh mesh;
    mesh.hfP = { 0.4f, -0.3f, 0.2f, 0 };
    {
        F4 q = { 0.15f, -0.2f, 0.1f, 1.0f };
        const float m = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        mesh.hfQ = { q.x / m, q.y / m, q.z / m, q.w / m };
    }
    mesh.scale = { 0.5f, 0.02f, 0.55f, 0.35f }; // w = contactDist
    mesh.misc = { 0.04f, 0, 0, 0 };
    mesh.dims = { ROWS, COLS, 0, 0 };
    mesh.rigid = { 0x99u, 4u, 0, 0 };

    // particles: mostly near the surface; the last 64 target the grid edges —
    // straddling row/column 0 and rows-1/cols-1 (the getMin*/getMax* clamps
    // must bind) and fully outside each side (the early-outs must reject).
    // KNOWN LIMIT: the getMinRow/getMinColumn HI clamp (rows-2/cols-2) binds
    // only when the lower bound sits exactly ON rows-1/cols-1 (anything above
    // early-outs, anything below floors lower) — a measure-zero input this
    // test cannot generate; the clamp is defensive OOB protection.
    const uint32_t EDGE0 = NP - 64;
    std::vector<F4> curPos(NP), predPos(NP);
    for (uint32_t p = 0; p < NP; ++p)
    {
        float lr = 1.0f + (ROWS - 3.0f) * u01(rng);
        float lc2 = 1.0f + (COLS - 3.0f) * u01(rng);
        if (p >= EDGE0)
        {
            switch (p % 8)
            {
            case 0: lr = -0.4f + 0.3f * u01(rng); break;                    // straddle row 0
            case 1: lr = (ROWS - 1.0f) + 0.1f + 0.3f * u01(rng); break;     // straddle row rows-1
            case 2: lc2 = -0.4f + 0.3f * u01(rng); break;                   // straddle col 0
            case 3: lc2 = (COLS - 1.0f) + 0.1f + 0.3f * u01(rng); break;    // straddle col cols-1
            case 4: lr = ROWS + 2.0f; break;                                // out +row (early-out)
            case 5: lc2 = -3.0f; break;                                     // out -col (early-out)
            case 6: lr = -0.4f + 0.3f * u01(rng);
                    lc2 = (COLS - 1.0f) + 0.1f + 0.3f * u01(rng); break;    // corner straddle
            default: lr = ROWS + 3.0f; lc2 = COLS + 3.0f; break;            // out both
            }
        }
        const uint32_t vr = (uint32_t)std::min(std::max(lr, 0.0f), ROWS - 1.0f);
        const uint32_t vc = (uint32_t)std::min(std::max(lc2, 0.0f), COLS - 1.0f);
        const float lh = hf.getHeight(vr * COLS + vc);
        const V3 lp = { lr * mesh.scale.x, (lh + 60.0f * uv(rng)) * mesh.scale.y, lc2 * mesh.scale.z };
        const V3 wp = add(quatRotate(mesh.hfQ, lp), v3(mesh.hfP));
        curPos[p] = { wp.x, wp.y, wp.z, 1 };
        predPos[p] = { wp.x + 0.15f * uv(rng), wp.y + 0.15f * uv(rng), wp.z + 0.15f * uv(rng), 1 };
    }

    const bool CCD = true, TGS = true;

    // reference
    std::vector<std::vector<F4>> refT(NP);
    std::vector<uint8_t> taint(NP, 0);
    size_t contactsP = 0, capped = 0;
    for (uint32_t p = 0; p < NP; ++p)
    {
        F4 tmp[MAXMESH];
        Slack sl;
        const uint32_t n = refParticle(hf, mesh, v3(curPos[p]), v3(predPos[p]), CCD, TGS, false, tmp, sl);
        refT[p].assign(tmp, tmp + n);
        taint[p] = sl.near ? 1 : 0;
        if (!taint[p] && n) ++contactsP;
        if (!taint[p] && n == MAXMESH) ++capped;
    }

    // ---- GPU: static + dynamic dispatches ----
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t CAP = 4096;
    std::vector<PhMesh> meshV(1, mesh);
    std::vector<F4> infoInit(NP * MAXPART, { -9, -9, -9, -9 });
    std::vector<uint32_t> nodeInit(NP * MAXPART * 2, 0xcccccccc), cnt0(NP, 0), wc0(1, 0);
    std::vector<PsC> ctInit(CAP);
    std::vector<uint32_t> z2(CAP * 2, 0), z1(CAP, 0);
    const std::vector<uint32_t> strides = { (uint32_t)sizeof(PhMesh), 4, 16, 16, 16, 8, 4,
                                            (uint32_t)sizeof(PsC), 8, 4, 4, 8, 4, 4, 4 };
    struct RunOut
    {
        std::vector<F4> info;                                // one-way contact infos
        std::vector<uint32_t> cnt, node;                     // one-way counts + node indices
        std::vector<PsC> ct;                                 // writer contacts
        std::vector<uint32_t> byR, tByR, idxR, byP, tByP, idxP; // writer radix staging
        uint32_t wc;                                         // writer counter
    };
    auto run = [&](uint32_t flags, RunOut& o, uint32_t reps = 1, uint32_t cap = 0) {
        if (cap == 0) cap = CAP;
        PxgBufferHandle bM = createBufferFrom(dev, meshV);
        PxgBufferHandle bS = createBufferFrom(dev, hf.samples);
        PxgBufferHandle bCur = createBufferFrom(dev, curPos);
        PxgBufferHandle bPred = createBufferFrom(dev, predPos);
        PxgBufferHandle bI = createBufferFrom(dev, infoInit);
        PxgBufferHandle bNi = createBufferFrom(dev, nodeInit);
        PxgBufferHandle bC = createBufferFrom(dev, cnt0);
        PxgBufferHandle bW = createBufferFrom(dev, ctInit);
        PxgBufferHandle bBR = createBufferFrom(dev, z2);
        PxgBufferHandle bTBR = createBufferFrom(dev, z1);
        PxgBufferHandle bIBR = createBufferFrom(dev, z1);
        PxgBufferHandle bSP = createBufferFrom(dev, z2);
        PxgBufferHandle bTBP = createBufferFrom(dev, z1);
        PxgBufferHandle bIBP = createBufferFrom(dev, z1);
        PxgBufferHandle bWc = createBufferFrom(dev, wc0);
        PhParams prm = { NP, cap, flags, SYS };
        Kernel k("psHfCollide.spv", "ctPsHfCollide", strides, sizeof(PhParams));
        for (uint32_t r = 0; r < reps; ++r)
            k.dispatch((NP + 255) / 256, { bM, bS, bCur, bPred, bI, bNi, bC, bW, bBR, bTBR, bIBR, bSP, bTBP, bIBP, bWc }, prm);
        dev->flush();
        o.info = downloadBuffer<F4>(dev, bI, NP * MAXPART);
        o.cnt = downloadBuffer<uint32_t>(dev, bC, NP);
        o.node = downloadBuffer<uint32_t>(dev, bNi, NP * MAXPART * 2);
        o.ct = downloadBuffer<PsC>(dev, bW, CAP);
        o.byR = downloadBuffer<uint32_t>(dev, bBR, CAP * 2);
        o.tByR = downloadBuffer<uint32_t>(dev, bTBR, CAP);
        o.idxR = downloadBuffer<uint32_t>(dev, bIBR, CAP);
        o.byP = downloadBuffer<uint32_t>(dev, bSP, CAP * 2);
        o.tByP = downloadBuffer<uint32_t>(dev, bTBP, CAP);
        o.idxP = downloadBuffer<uint32_t>(dev, bIBP, CAP);
        o.wc = downloadBuffer<uint32_t>(dev, bWc, 1)[0];
        for (PxgBufferHandle h : { bM, bS, bCur, bPred, bI, bNi, bC, bW, bBR, bTBR, bIBR, bSP, bTBP, bIBP, bWc })
            dev->destroyBuffer(h);
    };

    RunOut A;
    run(1u | 2u | 8u, A); // CCD, TGS, static

    size_t mm = 0, checked = 0, skipped = 0, edgeHits = 0;
    for (uint32_t p = 0; p < NP; ++p)
    {
        if (taint[p]) { ++skipped; continue; }
        ++checked;
        const auto& rt = refT[p];
        if (A.cnt[p] != rt.size()) { ++mm; continue; }
        bool ok = true;
        for (uint32_t s = 0; s < rt.size() && ok; ++s)
        {
            const uint32_t oi = p + s * NP;
            ok = almostEqual(A.info[oi].w, rt[s].w, 1e-4f, 1e-4f) &&
                 almostEqual(A.info[oi].x, rt[s].x, 1e-4f, 1e-4f) &&
                 almostEqual(A.info[oi].y, rt[s].y, 1e-4f, 1e-4f) &&
                 almostEqual(A.info[oi].z, rt[s].z, 1e-4f, 1e-4f) &&
                 A.node[oi * 2] == 0x99u && A.node[oi * 2 + 1] == 4u;
        }
        for (uint32_t s = (uint32_t)rt.size(); s < MAXPART && ok; ++s)
        {
            const uint32_t oi = p + s * NP;
            ok = A.info[oi].x == -9.0f && A.node[oi * 2] == 0xccccccccu;
        }
        if (!ok) ++mm;
        if (p >= EDGE0 && !rt.empty()) ++edgeHits;
    }
    PXG_CHECK_MSG(mm == 0, "psHfCollide: %zu static mismatches", mm);
    PXG_CHECK_MSG(A.wc == 0, "psHfCollide: static writer unexpectedly used (%u)", A.wc);
    PXG_CHECK_MSG(checked > NP * 3 / 4, "psHfCollide: too many tainted (%zu)", skipped);
    PXG_CHECK_MSG(contactsP > 120 && capped >= 3,
                  "psHfCollide: coverage thin (cp=%zu cap=%zu)", contactsP, capped);
    PXG_CHECK_MSG(edgeHits >= 8, "psHfCollide: edge coverage thin (%zu)", edgeHits);

    // dynamic rigid: everything through the writer
    RunOut B;
    run(1u | 2u, B);
    std::vector<std::vector<uint32_t>> slots(NP);
    size_t mm2 = 0;
    for (uint32_t i = 0; i < std::min(B.wc, CAP); ++i)
    {
        const PsC& c = B.ct[i];
        if (c.pLo < NP && c.pHi == SYS && c.rLo == 0x99u && c.rHi == 4u)
            slots[c.pLo].push_back(i);
        else { ++mm2; continue; }
        // all six radix staging arrays must be consistent with the contact
        if (B.byR[i * 2] != 0x99u || B.byR[i * 2 + 1] != 4u || B.tByR[i] != 0x99u ||
            B.idxR[i] != i || B.byP[i * 2] != c.pLo || B.byP[i * 2 + 1] != c.pHi ||
            B.tByP[i] != c.pLo || B.idxP[i] != i)
            ++mm2;
    }
    uint32_t sumChecked = 0, numTainted = 0;
    for (uint32_t p = 0; p < NP; ++p)
    {
        if (taint[p]) { ++numTainted; continue; }
        const auto& rt = refT[p];
        sumChecked += (uint32_t)rt.size();
        if (slots[p].size() != rt.size()) { ++mm2; continue; }
        std::sort(slots[p].begin(), slots[p].end());
        for (uint32_t s = 0; s < rt.size(); ++s)
        {
            const PsC& g = B.ct[slots[p][s]];
            if (!almostEqual(g.normal_pen.w, rt[s].w, 1e-4f, 1e-4f) ||
                !almostEqual(g.normal_pen.x, rt[s].x, 1e-4f, 1e-4f) ||
                !almostEqual(g.normal_pen.y, rt[s].y, 1e-4f, 1e-4f) ||
                !almostEqual(g.normal_pen.z, rt[s].z, 1e-4f, 1e-4f)) { ++mm2; break; }
        }
    }
    // a tainted particle's GPU count can legitimately exceed its reference
    // count, so the upper bound allows every tainted particle its full cap
    const uint32_t sumMax = sumChecked + (uint32_t)MAXMESH * numTainted;
    PXG_CHECK_MSG(mm2 == 0, "psHfCollide: %zu writer mismatches", mm2);
    PXG_CHECK_MSG(B.wc >= sumChecked && B.wc <= sumMax,
                  "psHfCollide: writer counter %u outside [%u, %u]", B.wc, sumChecked, sumMax);

    // triple static dispatch: real usage runs one invocation per heightfield
    // mesh against the SAME one-way buffers.  The faithful plain-store quirk
    // makes the final count n (the last mesh's contactWritten), NOT the
    // accumulated atomicAdd result 3n: d1 writes slots [0, n) and stores n;
    // d2 atomicAdds from n, writes [n, 2n) and stores n again; d3 atomicAdds
    // from the RESET count n, overwrites [n, 2n) with the same values and
    // stores n.  Slots >= 2n stay sentinel and the writer never fires
    // (static remainder is always 0).
    {
        RunOut C;
        run(1u | 2u | 8u, C, 3);
        size_t mm4 = 0;
        for (uint32_t p = 0; p < NP; ++p)
        {
            if (taint[p]) continue;
            const auto& rt = refT[p];
            const uint32_t n = (uint32_t)rt.size();
            if (C.cnt[p] != n) { ++mm4; continue; }
            bool ok = true;
            for (uint32_t s = 0; s < 2 * n && ok; ++s)
            {
                const uint32_t oi = p + s * NP;
                ok = almostEqual(C.info[oi].w, rt[s % n].w, 1e-4f, 1e-4f) &&
                     almostEqual(C.info[oi].x, rt[s % n].x, 1e-4f, 1e-4f);
            }
            for (uint32_t s = 2 * n; s < MAXPART && ok; ++s)
                ok = C.info[p + s * NP].x == -9.0f;
            if (!ok) ++mm4;
        }
        PXG_CHECK_MSG(mm4 == 0, "psHfCollide: %zu triple-dispatch mismatches", mm4);
        PXG_CHECK_MSG(C.wc == 0, "psHfCollide: triple-dispatch writer fired (%u)", C.wc);
    }

    // no CCD: plain contactDist volume, cVolumePos == currentPos, no
    // closest-point recompute in particleTriangleTest (static dispatch)
    {
        std::vector<std::vector<F4>> refT2(NP);
        std::vector<uint8_t> taint2(NP, 0);
        for (uint32_t p = 0; p < NP; ++p)
        {
            F4 tmp[MAXMESH];
            Slack sl;
            const uint32_t n = refParticle(hf, mesh, v3(curPos[p]), v3(predPos[p]), false, false, false, tmp, sl);
            refT2[p].assign(tmp, tmp + n);
            taint2[p] = sl.near ? 1 : 0;
        }
        RunOut D;
        run(8u, D);
        size_t mm3 = 0, hits3 = 0;
        for (uint32_t p = 0; p < NP; ++p)
        {
            if (taint2[p]) continue;
            const auto& rt = refT2[p];
            if (D.cnt[p] != rt.size()) { ++mm3; continue; }
            if (!rt.empty()) ++hits3;
            for (uint32_t s = 0; s < rt.size(); ++s)
            {
                const uint32_t oi = p + s * NP;
                if (!almostEqual(D.info[oi].w, rt[s].w, 1e-4f, 1e-4f) ||
                    !almostEqual(D.info[oi].x, rt[s].x, 1e-4f, 1e-4f)) { ++mm3; break; }
            }
        }
        PXG_CHECK_MSG(mm3 == 0, "psHfCollide: %zu non-CCD mismatches", mm3);
        PXG_CHECK_MSG(hits3 > 40, "psHfCollide: non-CCD coverage thin (%zu)", hits3);
    }

    // PGS CCD (static): scaleInv = 1 instead of the TGS 2 — the contact
    // volume differs, so the PGS branch of the forward-projection constant
    // is pinned against the TGS reference used above
    std::vector<std::vector<F4>> refPgs(NP);
    std::vector<uint8_t> taintPgs(NP, 0);
    {
        size_t diffPgs = 0;
        for (uint32_t p = 0; p < NP; ++p)
        {
            F4 tmp[MAXMESH];
            Slack sl;
            const uint32_t n = refParticle(hf, mesh, v3(curPos[p]), v3(predPos[p]), true, false, false, tmp, sl);
            refPgs[p].assign(tmp, tmp + n);
            taintPgs[p] = sl.near ? 1 : 0;
            if (!taintPgs[p] && !taint[p] && refPgs[p].size() != refT[p].size()) ++diffPgs;
        }
        RunOut E;
        run(1u | 8u, E);
        size_t mmE = 0, hitsE = 0;
        for (uint32_t p = 0; p < NP; ++p)
        {
            if (taintPgs[p]) continue;
            const auto& rt = refPgs[p];
            if (E.cnt[p] != rt.size()) { ++mmE; continue; }
            if (!rt.empty()) ++hitsE;
            for (uint32_t s = 0; s < rt.size(); ++s)
            {
                const uint32_t oi = p + s * NP;
                if (!almostEqual(E.info[oi].w, rt[s].w, 1e-4f, 1e-4f) ||
                    !almostEqual(E.info[oi].x, rt[s].x, 1e-4f, 1e-4f)) { ++mmE; break; }
            }
        }
        PXG_CHECK_MSG(mmE == 0, "psHfCollide: %zu PGS mismatches", mmE);
        PXG_CHECK_MSG(hitsE > 40, "psHfCollide: PGS coverage thin (%zu)", hitsE);
        PXG_CHECK_MSG(diffPgs >= 5, "psHfCollide: PGS/TGS volumes indistinguishable (%zu)", diffPgs);
    }

    // diffuse + dynamic rigid: the writer suppression (!isDiffuse) must leave
    // EVERYTHING untouched — no one-way writes (not static), no reservation
    {
        RunOut F;
        run(1u | 2u | 4u, F);
        size_t mmF = 0;
        for (uint32_t p = 0; p < NP && mmF == 0; ++p)
            if (F.cnt[p] != 0) ++mmF;
        for (uint32_t i = 0; i < NP * MAXPART && mmF == 0; ++i)
            if (F.info[i].x != -9.0f || F.node[i * 2] != 0xccccccccu) ++mmF;
        for (uint32_t i = 0; i < CAP && mmF == 0; ++i)
            if (F.ct[i].rLo != 0 || F.idxR[i] != 0 || F.idxP[i] != 0 || F.byP[i * 2] != 0) ++mmF;
        PXG_CHECK_MSG(mmF == 0 && F.wc == 0,
                      "psHfCollide: diffuse dynamic touched output (mm=%zu wc=%u)", mmF, F.wc);
    }

    // diffuse + static: the one-way path still runs; the CCD volume uses the
    // DIFFUSE forward-projection constant (1, same as PGS) even though the
    // TGS flag is set, so the diffuse branch of the constant is pinned
    {
        RunOut G;
        run(1u | 2u | 4u | 8u, G);
        size_t mmG = 0;
        for (uint32_t p = 0; p < NP; ++p)
        {
            if (taintPgs[p]) continue;
            const auto& rt = refPgs[p];
            if (G.cnt[p] != rt.size()) { ++mmG; continue; }
            for (uint32_t s = 0; s < rt.size(); ++s)
            {
                const uint32_t oi = p + s * NP;
                if (!almostEqual(G.info[oi].w, rt[s].w, 1e-4f, 1e-4f) ||
                    !almostEqual(G.info[oi].x, rt[s].x, 1e-4f, 1e-4f)) { ++mmG; break; }
            }
        }
        PXG_CHECK_MSG(mmG == 0, "psHfCollide: %zu diffuse-static mismatches", mmG);
        PXG_CHECK_MSG(G.wc == 0, "psHfCollide: diffuse-static writer fired (%u)", G.wc);
    }

    // small writer capacity: the atomicAdd reservation is NOT gated (the
    // counter overshoots to the same total as the full-capacity run) but only
    // slots below maxContacts may be written
    {
        const uint32_t TINY = 48;
        RunOut H;
        run(1u | 2u, H, 1, TINY);
        size_t mmH = 0;
        for (uint32_t i = 0; i < TINY; ++i)
            if (H.ct[i].rLo != 0x99u || H.ct[i].rHi != 4u || H.ct[i].pLo >= NP ||
                H.ct[i].pHi != SYS || H.idxR[i] != i || H.idxP[i] != i)
                ++mmH;
        for (uint32_t i = TINY; i < CAP; ++i)
            if (H.ct[i].rLo != 0 || H.idxR[i] != 0 || H.byP[i * 2] != 0 || H.tByP[i] != 0)
                ++mmH;
        PXG_CHECK_MSG(mmH == 0, "psHfCollide: %zu capacity-guard violations", mmH);
        PXG_CHECK_MSG(H.wc == B.wc && H.wc > TINY,
                      "psHfCollide: overflow counter %u (expected %u > %u)", H.wc, B.wc, TINY);
    }
}

// P9.94c: the local bound must be the CUDA's literal
// PxBounds3::transformFast(pose.getInverse(), worldBound).  Under psHfCollide_core's
// general rotation the old rotateInv spelling moves bounds by an ULP but never a
// contact, so this test uses the directed axis-aligned poses of pxgPsHfBoundCases.h,
// where the one-way contact COUNT itself depends on the spelling.  No Slack taint:
// these particles sit on an integer row boundary by construction, and that is the
// point.  Counts, node indices and sentinels compare exactly; contact values with
// almostEqual only (every non-count decision has margin, see the header).
PXG_TEST(psHfCollide_boundLiteral)
{
    using namespace pxgpshfbound;
    auto fb = [](uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; };
    auto bitsOf = [](float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; };
    const uint32_t NP = 2, SYS = 6, CAP = 64;

    Hf hf;
    hf.rows = kRows;
    hf.cols = kCols;
    hf.samples.resize(kRows * kCols);
    for (uint32_t r = 0; r < kRows; ++r)
        for (uint32_t c = 0; c < kCols; ++c)
        {
            const int16_t h = r < 3 ? kRowHeight[r] : int16_t(0);
            hf.samples[r * kCols + c] = uint32_t(uint16_t(h));   // materials 0, tess flag 0
        }

    PxgSlangComputeDevice* dev = testDevice();
    const std::vector<uint32_t> strides = { (uint32_t)sizeof(PhMesh), 4, 16, 16, 16, 8, 4,
                                            (uint32_t)sizeof(PsC), 8, 4, 4, 8, 4, 4, 4 };
    const HfBoundSpelling spellings[5] = { HfBoundSpelling::Literal, HfBoundSpelling::CentreRotateInv,
                                           HfBoundSpelling::BasisRotateInv, HfBoundSpelling::OldPort,
                                           HfBoundSpelling::BasisNoConj };
    const char* spellingName[5] = { "literal", "centre-rotateInv", "basis-rotateInv", "old-port",
                                    "basis-noconj" };

    for (const Pose& pose : kPoses)
    {
        PhMesh mesh;
        mesh.hfP = { fb(pose.p[0]), fb(pose.p[1]), fb(pose.p[2]), 0 };
        mesh.hfQ = { fb(pose.q[0]), fb(pose.q[1]), fb(pose.q[2]), fb(pose.q[3]) };
        const float cd = fb(pose.cdPs) + fb(pose.cdHf);
        mesh.scale = { 1.0f, 1.0f, 1.0f, cd };
        mesh.misc = { kRestDist, 0, 0, 0 };
        mesh.dims = { kRows, kCols, 0, 0 };
        mesh.rigid = { 0x99u, 4u, 0, 0 };
        if (std::strcmp(pose.name, "DIAG035") == 0)
            PXG_CHECK_MSG(bitsOf(cd) == 0x3eb33333u, "%s: contactDist bits 0x%08x", pose.name, bitsOf(cd));

        std::vector<F4> pos(NP);
        std::vector<F4> ref[2];
        uint32_t nLit[2], sumLit = 0;
        for (uint32_t i = 0; i < NP; ++i)
        {
            const Part& part = pose.part[i];
            pos[i] = { fb(part.cur[0]), fb(part.cur[1]), fb(part.cur[2]), 1.0f };
            const uint32_t expect[5] = { part.nLit, part.nCentre, part.nBasis, part.nOld, part.nNoConj };
            for (uint32_t k = 0; k < 5; ++k)
            {
                F4 tmp[MAXMESH];
                Slack sl;   // deliberately unused: these inputs are knife-edged by design
                const uint32_t n = refParticle(hf, mesh, v3(pos[i]), v3(pos[i]), false, false, false,
                                               tmp, sl, spellings[k]);
                // the DESIGN check: each spelling yields the tabulated count, so the
                // inputs really separate the literal spelling from every other one
                PXG_CHECK_MSG(n == expect[k], "%s part %u: %s reference count %u, table %u",
                              pose.name, i, spellingName[k], n, expect[k]);
                if (k == 0) ref[i].assign(tmp, tmp + n);
            }
            nLit[i] = (uint32_t)ref[i].size();
            sumLit += nLit[i];
        }

        std::vector<PhMesh> meshV(1, mesh);
        std::vector<F4> infoInit(NP * MAXPART, { -9, -9, -9, -9 });
        std::vector<uint32_t> nodeInit(NP * MAXPART * 2, 0xcccccccc), cnt0(NP, 0), wc0(1, 0);
        std::vector<PsC> ctInit(CAP);
        std::memset(ctInit.data(), 0xAB, ctInit.size() * sizeof(PsC));
        std::vector<uint32_t> z2(CAP * 2, 0), z1(CAP, 0);
        for (uint32_t flags : { 8u, 0u })   // static rigid, then dynamic rigid (non-CCD, PGS)
        {
            PxgBufferHandle bM = createBufferFrom(dev, meshV);
            PxgBufferHandle bS = createBufferFrom(dev, hf.samples);
            PxgBufferHandle bCur = createBufferFrom(dev, pos);
            PxgBufferHandle bPred = createBufferFrom(dev, pos);
            PxgBufferHandle bI = createBufferFrom(dev, infoInit);
            PxgBufferHandle bNi = createBufferFrom(dev, nodeInit);
            PxgBufferHandle bC = createBufferFrom(dev, cnt0);
            PxgBufferHandle bW = createBufferFrom(dev, ctInit);
            PxgBufferHandle bBR = createBufferFrom(dev, z2);
            PxgBufferHandle bTBR = createBufferFrom(dev, z1);
            PxgBufferHandle bIBR = createBufferFrom(dev, z1);
            PxgBufferHandle bSP = createBufferFrom(dev, z2);
            PxgBufferHandle bTBP = createBufferFrom(dev, z1);
            PxgBufferHandle bIBP = createBufferFrom(dev, z1);
            PxgBufferHandle bWc = createBufferFrom(dev, wc0);
            PhParams prm = { NP, CAP, flags, SYS };
            {
                Kernel k("psHfCollide.spv", "ctPsHfCollide", strides, sizeof(PhParams));
                k.dispatch(1, { bM, bS, bCur, bPred, bI, bNi, bC, bW, bBR, bTBR, bIBR, bSP, bTBP, bIBP, bWc }, prm);
            }
            dev->flush();
            const auto info = downloadBuffer<F4>(dev, bI, NP * MAXPART);
            const auto cnt = downloadBuffer<uint32_t>(dev, bC, NP);
            const auto node = downloadBuffer<uint32_t>(dev, bNi, NP * MAXPART * 2);
            const auto ct = downloadBuffer<PsC>(dev, bW, CAP);
            const uint32_t wc = downloadBuffer<uint32_t>(dev, bWc, 1)[0];
            for (PxgBufferHandle h : { bM, bS, bCur, bPred, bI, bNi, bC, bW, bBR, bTBR, bIBR, bSP, bTBP, bIBP, bWc })
                dev->destroyBuffer(h);

            const bool isStatic = (flags & 8u) != 0;
            size_t bad = 0;
            for (uint32_t i = 0; i < NP; ++i)
            {
                const uint32_t nOneWay = isStatic ? nLit[i] : 0u;
                PXG_CHECK_MSG(cnt[i] == nOneWay, "%s part %u flags %u: one-way count %u, literal %u",
                              pose.name, i, flags, cnt[i], nOneWay);
                for (uint32_t s = 0; s < MAXPART; ++s)
                {
                    const uint32_t oi = i + s * NP;
                    if (s < nOneWay && cnt[i] == nOneWay)
                    {
                        const F4& r = ref[i][s];
                        if (!almostEqual(info[oi].x, r.x, 1e-4f, 1e-4f) || !almostEqual(info[oi].y, r.y, 1e-4f, 1e-4f) ||
                            !almostEqual(info[oi].z, r.z, 1e-4f, 1e-4f) || !almostEqual(info[oi].w, r.w, 1e-4f, 1e-4f) ||
                            node[oi * 2] != 0x99u || node[oi * 2 + 1] != 4u)
                            ++bad;
                    }
                    else if (s >= nOneWay)
                    {
                        if (info[oi].x != -9.0f || info[oi].y != -9.0f || info[oi].z != -9.0f ||
                            info[oi].w != -9.0f || node[oi * 2] != 0xccccccccu || node[oi * 2 + 1] != 0xccccccccu)
                            ++bad;
                    }
                }
            }
            if (isStatic)
                PXG_CHECK_MSG(wc == 0, "%s: static writer used (%u)", pose.name, wc);
            else
            {
                PXG_CHECK_MSG(wc == sumLit, "%s: writer counter %u, literal total %u", pose.name, wc, sumLit);
                std::vector<uint32_t> slots[2];
                for (uint32_t j = 0; j < std::min(wc, CAP); ++j)
                {
                    if (ct[j].pLo < NP && ct[j].pHi == SYS && ct[j].rLo == 0x99u && ct[j].rHi == 4u)
                        slots[ct[j].pLo].push_back(j);
                    else
                        ++bad;
                }
                for (uint32_t i = 0; i < NP; ++i)
                {
                    if (slots[i].size() != nLit[i]) { ++bad; continue; }
                    std::sort(slots[i].begin(), slots[i].end());
                    for (uint32_t s = 0; s < nLit[i]; ++s)
                    {
                        const F4& g = ct[slots[i][s]].normal_pen;
                        const F4& r = ref[i][s];
                        if (!almostEqual(g.x, r.x, 1e-4f, 1e-4f) || !almostEqual(g.y, r.y, 1e-4f, 1e-4f) ||
                            !almostEqual(g.z, r.z, 1e-4f, 1e-4f) || !almostEqual(g.w, r.w, 1e-4f, 1e-4f))
                            ++bad;
                    }
                }
            }
            PXG_CHECK_MSG(bad == 0, "%s flags %u: %zu slot/value/sentinel mismatches", pose.name, flags, bad);
        }
    }
}

} // namespace
