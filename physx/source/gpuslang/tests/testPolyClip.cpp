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

// P8.38 — polyClip + addContacts (convexTriangle.cuh:918-1058) on a 32-lane
// virtual warp.
//
// The oracle is an INDEPENDENT transcription: a scalar C++ simulation of the same
// warp algorithm, with the shuffles as array reads and the ballots as explicit
// loops over 32 lanes.  It shares the maths with the shader but not the shape, so
// a mistake in the virtual-warp machinery — a missing barrier, a ballot over the
// wrong predicate, a shuffle off by one lane — shows up as a disagreement.
//
// Suite (ctest filter "polyClip"):
//   polyClipCases — the three exit paths, plus rejection, degeneracy and the
//                   coincident-polygon case the early-outs exist for.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t LANES = 32u;
const uint32_t MAX_CONTACTS = 32u;
const float MAXF = 3.4028234663852886e+38f;

struct V3
{
    float x, y, z;
};
V3 mk(float x, float y, float z) { return V3{ x, y, z }; }
V3 sub(V3 a, V3 b) { return mk(a.x - b.x, a.y - b.y, a.z - b.z); }
V3 add3(V3 a, V3 b) { return mk(a.x + b.x, a.y + b.y, a.z + b.z); }
V3 scl(V3 a, float s) { return mk(a.x * s, a.y * s, a.z * s); }
float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 cross3(V3 a, V3 b)
{
    return mk(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

uint32_t popc(uint32_t v)
{
    uint32_t n = 0;
    while (v) { n += v & 1u; v >>= 1; }
    return n;
}
uint32_t lowestIndex(uint32_t v)
{
    uint32_t i = 0;
    while (!((v >> i) & 1u)) ++i;
    return i;
}

struct ClipInput
{
    float plane0[4];
    float plane1[4];
    V3 axis;
    float maxSep;
    uint32_t nbVerts0;
    uint32_t initContacts;
    V3 v0[LANES];
    V3 v1[LANES];
};

// ---------------------------------------------------------------------------
// the oracle: addContacts (convexTriangle.cuh:918) over 32 explicit lanes
// ---------------------------------------------------------------------------
void refAddContacts(uint32_t flags, const V3* pos, const float* sep, float maxSep,
                    float* contacts, uint32_t& nbContacts)
{
    uint32_t ballot = 0;
    for (uint32_t l = 0; l < LANES; ++l)
        if (!(sep[l] > maxSep))          // inverted test, so NaN counts as inside
            ballot |= 1u << l;
    flags &= ballot;

    uint32_t addMask = 0;
    for (uint32_t l = 0; l < LANES; ++l)
    {
        const uint32_t index = popc(flags & ((1u << l) - 1u)) + nbContacts;
        const bool add = ((flags >> l) & 1u) && index < MAX_CONTACTS;
        if (add)
        {
            contacts[4 * index + 0] = pos[l].x;
            contacts[4 * index + 1] = pos[l].y;
            contacts[4 * index + 2] = pos[l].z;
            contacts[4 * index + 3] = sep[l];
            addMask |= 1u << l;
        }
    }
    nbContacts += popc(addMask);
}

// exitPath: 0 = returned at !any(out1), 1 = at !any(out0), 2 = ran the edge loop
uint32_t refPolyClip(const ClipInput& in, float* contacts, uint32_t& exitPath)
{
    exitPath = 2;
    const V3 n0 = mk(in.plane0[0], in.plane0[1], in.plane0[2]);
    const V3 n1 = mk(in.plane1[0], in.plane1[1], in.plane1[2]);
    const uint32_t nbVerts0 = in.nbVerts0;

    const float dnom0 = dot3(n1, in.axis);
    const float dnom1 = dot3(n0, in.axis);

    float s0[LANES], s1[LANES];
    for (uint32_t l = 0; l < LANES; ++l)
    {
        const float t0 = dot3(in.v0[l], n1) + in.plane1[3];
        const float t1 = dot3(in.v1[l], n0) + in.plane0[3];
        s0[l] = ((t0 > in.maxSep) || (dnom0 == 0.0f)) ? MAXF : -t0 / dnom0;
        s1[l] = ((t1 > in.maxSep) || (dnom1 == 0.0f)) ? MAXF : t1 / dnom1;
    }

    uint32_t nbContacts = in.initContacts;

    V3 e0[LANES], e0planeN[LANES];
    float e0planeD[LANES];
    for (uint32_t l = 0; l < LANES; ++l)
    {
        e0[l] = sub(in.v0[(l + 1 == nbVerts0) ? 0 : (l + 1) % LANES], in.v0[l]);
        e0planeN[l] = cross3(e0[l], in.axis);
        e0planeD[l] = dot3(e0planeN[l], in.v0[l]);
    }

    uint32_t in1[LANES] = {}, out1[LANES] = {};
    for (uint32_t l = 0; l < LANES; ++l)
    {
        for (uint32_t j = 0, bj = 1; j < 3; ++j, bj <<= 1)
        {
            const float t = dot3(in.v1[j], e0planeN[l]) - e0planeD[l];
            in1[l] |= (t < 0.0f ? bj : 0u);
            out1[l] |= (t > 0.0f ? bj : 0u);
        }
        if (l >= nbVerts0) { in1[l] = 0; out1[l] = 0; }
    }

    uint32_t b = 0xffffffffu, anyOut1 = 0;
    for (uint32_t l = 0; l < LANES; ++l) { b &= ~out1[l]; anyOut1 |= out1[l]; }

    {
        V3 pos[LANES];
        for (uint32_t l = 0; l < LANES; ++l) pos[l] = sub(in.v1[l], scl(in.axis, s1[l]));
        refAddContacts(b & 7u, pos, s1, in.maxSep, contacts, nbContacts);
    }

    if (anyOut1 == 0)
    {
        exitPath = 0;
        return nbContacts;
    }

    uint32_t e1cross[LANES];
    for (uint32_t l = 0; l < LANES; ++l)
    {
        const uint32_t i1 = in1[l] | (in1[l] << 3);
        const uint32_t o1 = out1[l] | (out1[l] << 3);
        e1cross[l] = ((i1 & (o1 >> 1)) | ((i1 >> 1) & o1)) & 7u;
    }

    V3 e1planeN[LANES];
    float e1planeD[LANES];
    for (uint32_t l = 0; l < LANES; ++l)
    {
        const V3 e1 = sub(in.v1[(l + 1 == 3) ? 0 : (l + 1) % LANES], in.v1[l]);
        e1planeN[l] = cross3(in.axis, e1);
        e1planeD[l] = dot3(e1planeN[l], in.v1[l]);
    }

    uint32_t in0[LANES] = {}, out0[LANES] = {};
    for (uint32_t j = 0; j < nbVerts0; ++j)
    {
        uint32_t inB = 0, outB = 0;
        for (uint32_t l = 0; l < LANES; ++l)
        {
            const float t = dot3(in.v0[j], e1planeN[l]) - e1planeD[l];
            if (t < 0.0f) inB |= 1u << l;
            if (t > 0.0f) outB |= 1u << l;
        }
        in0[j] = inB;
        out0[j] = outB;
    }

    uint32_t anyOut0 = 0, notOut0Ballot = 0;
    for (uint32_t l = 0; l < LANES; ++l)
    {
        in0[l] &= 7u;
        out0[l] &= 7u;
        anyOut0 |= out0[l];
        if (out0[l] == 0) notOut0Ballot |= 1u << l;
    }

    refAddContacts(notOut0Ballot & ((1u << nbVerts0) - 1u), in.v0, s0, in.maxSep, contacts, nbContacts);

    if (anyOut0 == 0)
    {
        exitPath = 1;
        return nbContacts;
    }

    uint32_t c[LANES], a = 0;
    for (uint32_t l = 0; l < LANES; ++l)
    {
        const uint32_t nxt = (l + 1 == nbVerts0) ? 0 : (l + 1) % LANES;
        const uint32_t e0cross = (in0[l] & out0[nxt]) | (in0[nxt] & out0[l]);
        c[l] = e0cross & e1cross[l];
        a |= c[l];
    }

    for (; a != 0; a &= (a - 1))
    {
        const uint32_t index = lowestIndex(a);
        const uint32_t lowBit = a & (0u - a);
        const V3 m = e1planeN[index];
        const V3 triVert0 = in.v1[index];

        V3 pos[LANES];
        float sep[LANES];
        uint32_t flags = 0;
        for (uint32_t l = 0; l < LANES; ++l)
        {
            const float denom = dot3(m, e0[l]);
            const float lambda = dot3(sub(triVert0, in.v0[l]), m) / denom;
            pos[l] = add3(in.v0[l], scl(e0[l], lambda));
            const float t = -dot3(sub(pos[l], triVert0), n1);
            sep[l] = ((denom == 0.0f) || (dnom0 == 0.0f)) ? MAXF : t / dnom0;
            if (c[l] & lowBit) flags |= 1u << l;
        }
        refAddContacts(flags, pos, sep, in.maxSep, contacts, nbContacts);
    }

    return nbContacts;
}

// ---------------------------------------------------------------------------
// scene builders
// ---------------------------------------------------------------------------
struct PolyClipPush
{
    uint64_t input, output;
    uint32_t numPairs, pad;
};

// THE TWO WINDINGS ARE OPPOSITE, and getting one wrong turns the whole clipper
// inside out silently — the first attempt here had both backwards and every case
// came out with zero contacts through the edge-edge path instead of three through
// the early-out.  polyClip classifies with `t < 0 -> inside`, and it builds the
// two edge-plane normals with the operands SWAPPED: poly0 uses cross(e0, axis) and
// the triangle uses cross(axis, e1).  So with axis = +z, poly0 must be wound
// counter-clockwise and the triangle CLOCKWISE for both normals to point outward.
//
// PADDING LANES GET DISTINCT VALUES, and that is not cosmetic.  The first version
// of this fixture filled every lane past the end with vertex 0, which is also the
// wrap target of `tI + 1 == nbVerts0 ? 0 : tI + 1` — so a mutant that dropped the
// wrap entirely read the padding and got the right answer anyway, and so did a
// shader that indexed the staging array past the end.  Distinct padding makes both
// visible.  It cannot change the true result: lanes past nbVerts0 have in1/out1
// forced to zero, are never assigned in0/out0, and are masked out of the second
// addContacts by (1 << nbVerts0) - 1.
void ngon(ClipInput& ci, uint32_t n, float radius, float cx, float cy)
{
    ci.nbVerts0 = n;
    for (uint32_t i = 0; i < LANES; ++i)
    {
        if (i >= n)
        {
            ci.v0[i] = mk(1000.0f + float(i), -1000.0f - float(i), 3.0f + float(i));
            continue;
        }
        const float ang = 6.28318530718f * float(i) / float(n);   // CCW
        ci.v0[i] = mk(cx + radius * std::cos(ang), cy + radius * std::sin(ang), 0.0f);
    }
}

// Takes the triangle counter-clockwise and stores it clockwise, so the call sites
// read naturally.
void triangle(ClipInput& ci, V3 a, V3 b, V3 c)
{
    for (uint32_t i = 0; i < LANES; ++i)
        ci.v1[i] = (i == 0) ? a : (i == 1) ? c : (i == 2) ? b
                                             : mk(-2000.0f - float(i), 2000.0f + float(i), -4.0f - float(i));
}

// poly0 in z = 0 with normal +z, the triangle at z = -depth with normal -z, and
// the separating axis +z.  Then s0 and s1 both come out as -depth.
void frame(ClipInput& ci, float depth, float maxSep)
{
    ci.plane0[0] = 0.0f; ci.plane0[1] = 0.0f; ci.plane0[2] = 1.0f; ci.plane0[3] = 0.0f;
    ci.plane1[0] = 0.0f; ci.plane1[1] = 0.0f; ci.plane1[2] = -1.0f; ci.plane1[3] = -depth;
    ci.axis = mk(0.0f, 0.0f, 1.0f);
    ci.maxSep = maxSep;
    ci.initContacts = 0;
}

void packInput(const ClipInput& ci, std::vector<float>& dst)
{
    const size_t base = dst.size();
    dst.resize(base + 1088 / 4, 0.0f);
    float* p = dst.data() + base;
    std::memcpy(p + 0, ci.plane0, 16);
    std::memcpy(p + 4, ci.plane1, 16);
    p[8] = ci.axis.x; p[9] = ci.axis.y; p[10] = ci.axis.z; p[11] = ci.maxSep;
    uint32_t counts[4] = { ci.nbVerts0, ci.initContacts, 0u, 0u };
    std::memcpy(p + 12, counts, 16);
    for (uint32_t i = 0; i < LANES; ++i)
    {
        p[16 + 4 * i + 0] = ci.v0[i].x; p[16 + 4 * i + 1] = ci.v0[i].y; p[16 + 4 * i + 2] = ci.v0[i].z;
        p[144 + 4 * i + 0] = ci.v1[i].x; p[144 + 4 * i + 1] = ci.v1[i].y; p[144 + 4 * i + 2] = ci.v1[i].z;
    }
}

PXG_TEST(polyClipCases)
{
    PxgSlangComputeDevice* dev = testDevice();

    std::vector<ClipInput> cases;
    std::vector<const char*> names;

    {   // 1: triangle wholly inside a big quad -> 3 contacts, first early-out
        ClipInput ci{}; frame(ci, 0.02f, 0.05f);
        ngon(ci, 4, 2.0f, 0.0f, 0.0f);
        triangle(ci, mk(-0.3f, -0.3f, -0.02f), mk(0.4f, -0.2f, -0.02f), mk(0.0f, 0.5f, -0.02f));
        cases.push_back(ci); names.push_back("triangle inside quad");
    }
    {   // 2: quad wholly inside a big triangle -> 4 contacts, second early-out
        ClipInput ci{}; frame(ci, 0.01f, 0.05f);
        ngon(ci, 4, 0.2f, 0.0f, 0.0f);
        triangle(ci, mk(-3.0f, -2.0f, -0.01f), mk(3.0f, -2.0f, -0.01f), mk(0.0f, 3.0f, -0.01f));
        cases.push_back(ci); names.push_back("quad inside triangle");
    }
    {   // 3: partial overlap -> the edge-edge loop runs
        ClipInput ci{}; frame(ci, 0.02f, 0.05f);
        ngon(ci, 4, 1.0f, 0.0f, 0.0f);
        triangle(ci, mk(0.5f, -0.4f, -0.02f), mk(2.5f, 0.0f, -0.02f), mk(0.5f, 0.6f, -0.02f));
        cases.push_back(ci); names.push_back("partial overlap, quad");
    }
    {   // 4: partial overlap against an octagon (nbVerts0 = 8)
        ClipInput ci{}; frame(ci, 0.03f, 0.05f);
        ngon(ci, 8, 1.0f, 0.0f, 0.0f);
        triangle(ci, mk(0.3f, -0.7f, -0.03f), mk(2.0f, 0.2f, -0.03f), mk(0.2f, 0.9f, -0.03f));
        cases.push_back(ci); names.push_back("partial overlap, octagon");
    }
    {   // 5: the same overlap, but SEPARATED rather than penetrating and by more
        //    than maxSep, so every contact is rejected by the addContacts ballot
        //    and the count stays 0.  A negative depth puts the triangle above the
        //    face: penetration makes sep more negative and is always accepted, so
        //    only a positive separation can exercise the rejection.
        ClipInput ci{}; frame(ci, -0.5f, 0.05f);
        ngon(ci, 4, 1.0f, 0.0f, 0.0f);
        triangle(ci, mk(0.5f, -0.4f, 0.5f), mk(2.5f, 0.0f, 0.5f), mk(0.5f, 0.6f, 0.5f));
        cases.push_back(ci); names.push_back("separated beyond maxSep");
    }
    {   // 6: axis in the triangle's plane -> dnom0 == 0, the PX_MAX_F32 branch
        ClipInput ci{}; frame(ci, 0.02f, 0.05f);
        ci.axis = mk(1.0f, 0.0f, 0.0f);
        ngon(ci, 4, 1.0f, 0.0f, 0.0f);
        triangle(ci, mk(-0.3f, -0.3f, -0.02f), mk(0.4f, -0.2f, -0.02f), mk(0.0f, 0.5f, -0.02f));
        cases.push_back(ci); names.push_back("axis in plane1 (dnom0 == 0)");
    }
    {   // 7: coincident polygons — poly0 IS the triangle.  This is the case the
        //    early-outs exist for (convexTriangle.cuh:911): without them the same
        //    points come out more than once.
        ClipInput ci{}; frame(ci, 0.0f, 0.05f);
        const V3 a = mk(-0.5f, -0.5f, 0.0f), bb = mk(0.5f, -0.5f, 0.0f), cc = mk(0.0f, 0.5f, 0.0f);
        ci.nbVerts0 = 3;
        for (uint32_t i = 0; i < LANES; ++i)   // poly0 CCW, padding distinct
            ci.v0[i] = (i == 0) ? a : (i == 1) ? bb : (i == 2) ? cc
                                    : mk(1000.0f + float(i), -1000.0f - float(i), 3.0f + float(i));
        triangle(ci, a, bb, cc);               // the same triangle, stored CW
        cases.push_back(ci); names.push_back("coincident polygons");
    }
    {   // 8: a non-zero initContacts, so the exclusive scan starts off-zero
        ClipInput ci{}; frame(ci, 0.02f, 0.05f);
        ngon(ci, 5, 1.0f, 0.1f, -0.1f);
        triangle(ci, mk(0.4f, -0.5f, -0.02f), mk(2.0f, 0.1f, -0.02f), mk(0.3f, 0.8f, -0.02f));
        ci.initContacts = 7;
        cases.push_back(ci); names.push_back("initContacts = 7");
    }

    {   // 9: a 31-gon wholly inside a big triangle with a non-zero base, so the
        //    second addContacts runs past NUM_TMP_CONTACTS_PER_PAIR.  5 + 31 = 36
        //    candidates for 32 slots: four are dropped, and `nbContacts` must count
        //    only the ones that were actually written.  (31 and not 32: with
        //    nbVerts0 == 32 the CUDA's own `(1 << nbVerts0) - 1` shifts by the full
        //    word width, which is undefined in C++ and in SPIR-V alike.  That is an
        //    upstream landmine, not something this port can fix, and deliberately
        //    not pinned by a test whose expected value would be a guess.)
        ClipInput ci{}; frame(ci, 0.02f, 0.05f);
        ngon(ci, 31, 0.2f, 0.0f, 0.0f);
        triangle(ci, mk(-3.0f, -2.0f, -0.02f), mk(3.0f, -2.0f, -0.02f), mk(0.0f, 3.0f, -0.02f));
        ci.initContacts = 5;
        cases.push_back(ci); names.push_back("31-gon, capacity overflow");
    }
    {   // 10: a triangle vertex EXACTLY on a poly0 edge.  The header comment at
        //     convexTriangle.cuh:910 calls this out ("including on an edge"), and it
        //     is the only case that separates `t < 0` from `t <= 0`.  The quad's
        //     right edge is the segment x = 1, and the triangle's first vertex sits
        //     on it at (1, 0) — exact in binary floating point, so the dot product
        //     lands on zero rather than near it.
        ClipInput ci{}; frame(ci, 0.02f, 0.05f);
        ci.nbVerts0 = 4;
        const V3 q[4] = { mk(1.0f, -1.0f, 0.0f), mk(1.0f, 1.0f, 0.0f),
                          mk(-1.0f, 1.0f, 0.0f), mk(-1.0f, -1.0f, 0.0f) };
        for (uint32_t i = 0; i < LANES; ++i)
            ci.v0[i] = (i < 4) ? q[i] : mk(1000.0f + float(i), -1000.0f - float(i), 3.0f + float(i));
        //     One more vertex has to stick OUT, or the pair leaves at the first
        //     early-out and the in/out classification of the boundary vertex never
        //     reaches e1cross, where it is what actually matters.
        triangle(ci, mk(1.0f, 0.0f, -0.02f), mk(2.0f, 0.5f, -0.02f), mk(0.5f, -0.5f, -0.02f));
        cases.push_back(ci); names.push_back("triangle vertex on a poly0 edge");
    }
    {   // 11: a NaN vertex.  addContacts tests `!(sep > maxSep)` rather than
        //     `sep <= maxSep` and the CUDA comment at :920 says why — "Inverting
        //     test covers us for nans".  The two spellings differ ONLY when sep is
        //     NaN, and with finite geometry every path that could produce one is
        //     already guarded to PX_MAX_F32, so a NaN has to come in through the
        //     vertices.  Contacts are compared bitwise, so a NaN that both sides
        //     produce identically counts as agreement.
        ClipInput ci{}; frame(ci, 0.02f, 0.05f);
        ngon(ci, 4, 1.0f, 0.0f, 0.0f);
        ci.v0[2] = mk(ci.v0[2].x, ci.v0[2].y, std::nanf(""));
        triangle(ci, mk(-3.0f, -2.0f, -0.02f), mk(3.0f, -2.0f, -0.02f), mk(0.0f, 3.0f, -0.02f));
        cases.push_back(ci); names.push_back("NaN vertex");
    }

    const uint32_t numPairs = uint32_t(cases.size());
    std::vector<float> input;
    for (const ClipInput& ci : cases) packInput(ci, input);
    std::vector<float> output(size_t(numPairs) * (528 / 4), -1.0f);

    PxgBufferHandle bIn = createBufferFrom(dev, input);
    PxgBufferHandle bOut = createBufferFrom(dev, output);

    PolyClipPush push{};
    push.input = dev->getBufferDeviceAddress(bIn);
    push.output = dev->getBufferDeviceAddress(bOut);
    push.numPairs = numPairs;

    {
        Kernel k("polyClipBda.spv", "polyClipLaunch", {}, sizeof(PolyClipPush));
        k.dispatch(numPairs, {}, push, 1u, 1u);
        dev->flush();
    }

    const auto got = downloadBuffer<float>(dev, bOut, output.size());

    size_t bad = 0, pathsHit = 0;
    bool sawEarly1 = false, sawEarly2 = false, sawEdgeLoop = false, sawEmpty = false;
    for (uint32_t p = 0; p < numPairs; ++p)
    {
        float refContacts[4 * MAX_CONTACTS];
        for (float& f : refContacts) f = 0.0f;
        uint32_t exitPath = 0;
        const uint32_t refN = refPolyClip(cases[p], refContacts, exitPath);
        if (std::getenv("PXG_POLYCLIP_DEBUG"))
            std::printf("  [dbg] %-28s exitPath=%u refN=%u (init=%u)\n",
                        names[p], exitPath, refN, cases[p].initContacts);
        if (exitPath == 0) sawEarly1 = true;
        else if (exitPath == 1) sawEarly2 = true;
        else sawEdgeLoop = true;

        const float* rec = got.data() + size_t(p) * (528 / 4);
        uint32_t gotN = 0;
        std::memcpy(&gotN, rec, 4);

        if (gotN != refN)
        {
            ++bad;
            std::printf("  [%s] nbContacts got=%u ref=%u\n", names[p], gotN, refN);
            continue;
        }
        if (refN == cases[p].initContacts) sawEmpty = true;

        for (uint32_t i = cases[p].initContacts; i < refN; ++i)
            for (uint32_t k = 0; k < 4; ++k)
            {
                const float g = rec[4 + 4 * i + k];
                const float e = refContacts[4 * i + k];
                // Bitwise, not ==: a NaN has to compare equal to the same NaN, and
                // the claim being made is bit-exactness anyway.
                uint32_t gb = 0, eb = 0;
                std::memcpy(&gb, &g, 4);
                std::memcpy(&eb, &e, 4);
                if (gb != eb)
                {
                    ++bad;
                    if (bad <= 8)
                        std::printf("  [%s] contact %u comp %u: got=%.9g (0x%08x) ref=%.9g (0x%08x)\n",
                                    names[p], i, k, g, gb, e, eb);
                }
            }
        ++pathsHit;
    }

    // Coverage: the three exits are what the tranche is about, so the oracle
    // reports which one it took and the fixture asserts all three are reached.
    // Without this a geometry mistake could quietly leave two paths untested.
    PXG_CHECK_MSG(sawEarly1, "some pair returns at !any(out1)");
    PXG_CHECK_MSG(sawEarly2, "some pair returns at !any(out0)");
    PXG_CHECK_MSG(sawEdgeLoop, "some pair runs the edge-edge loop");
    PXG_CHECK_MSG(sawEmpty, "some pair produces no contacts at all");
    PXG_CHECK_MSG(bad == 0, "polyClip matches the scalar oracle over %u pairs (%zu mismatches)",
                  numPairs, bad);
    PXG_CHECK_MSG(pathsHit == numPairs, "every pair compared (%zu/%u)", pathsHit, numPairs);
}

} // namespace
