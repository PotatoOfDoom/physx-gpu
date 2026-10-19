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

// Verification of softBodyPsContactGenBda.slang -- sb_psContactGenLaunch (P9.39c).
//
// The mirror below is transcribed from the PhysX 5.12 softbodyPrimitives.cu
// (sbParticleCollision) and the helpers it calls, NOT from the Slang.  That ordering is the whole point: a
// mirror written from the port reproduces the port's mistakes, which is how four
// bit-exact fixtures in this corpus stayed green over a defect (P9.36).
//
// A SECOND host transcription of closestPtTetrahedron / closestPtPointTriangle2 already
// exists in testNpSbPrim.cpp, for the Phase-1 twin of
// this collision.  This one is written independently rather than shared, so that two
// hands had to agree; the two should be consolidated the next time either is touched,
// which is noted rather than done inside a tranche about the contact gen.
//
// ---------------------------------------------------------------------------
// WHAT THE SCENE IS BUILT TO SEPARATE
// ---------------------------------------------------------------------------
// PHYSX 5.12 REMOVED THREE THINGS THIS SCENE USED TO SEPARATE, and the scene now pins
// their ABSENCE instead: the particle-buffer filter chain (G1-G3: the sorted-to-unsorted
// lookup, the whole-body / per-tet masks and the mUniqueId key -- mSortedToUnsortedMapping
// is NULL here, so a port that still reads it faults), the rest distance (the two
// penetration branches no longer subtract restDistances[cmIdx]; the argument is gone) and
// the barycentric (writeVolumeContact writes none: the slot is poisoned and must come back
// untouched; G7's -0 clamp therefore has nothing left to check here).
//
// G4  BOTH PENETRATION BRANCHES, AND THE GATE THAT ONLY ONE OF THEM HAS.  A particle
//     strictly inside a tetrahedron takes the dot(n, n2) branch with NO contact
//     distance test; particles outside take the squared-distance early-out.  One
//     particle sits just inside the cDistance shell and one just outside it.
//
// G5  surfaceMask == 0 SKIPS.  One tet has hint 0 while still being geometrically
//     reachable, so the skip is not confounded with a miss.
//
// G6  THE COUNTER OVERSHOOTS.  maxNumContacts is smaller than the number of contacts
//     the scene generates, so the dropped writes and the still-incrementing counter
//     are both measured.
//
// G8  THE 5.12 RECORD.  The pair info is (PxgEncodeParticleRef(system, sorted index),
//     PxgEncodeDeformableTetrahedron(softbody, tet)), and the particle sort key is ref0 in
//     contactByParticle with its low 32 bits (PxgGetContactSortKey) in
//     tempContactByParticle -- compared against the real 5.12 host encoders.

#include "pxgTestFramework.h"
#include "PxgSoftBody.h"
#include "PxgParticleSystem.h"
#include "PxgSimulationCoreDesc.h"   // PxgEncodeParticleRef / PxgEncodeDeformableTetrahedron
#include "PxgDeformableContactWriters.h"   // sizeof(PxgParticleDbContactWriter)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;

// ---------------------------------------------------------------------------
// push constants -- softbodyPrimitives.cu (5.12) sb_psContactGenLaunch, in order.  The
// by-value PxgParticleDbContactWriter tail is 72 bytes, paramAlign 8, so it lands at @88.
// ---------------------------------------------------------------------------
struct SbPsCgPush
{
    float toleranceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, gpuShapes, particleSystems, softbodies;
    uint32_t stackSizeBytes, pad60;
    uint64_t stackPtr, midphasePairsNum, stackSizeNeeded;
    uint64_t w_outPoint, w_outNormalPen, w_outBarycentric, w_outContactInfo,
             w_totalContactCount, w_contactByParticle, w_tempContactByParticle,
             w_contactIndexSortedByParticle;
    uint32_t w_maxNumContacts, pad156;
};
static_assert(sizeof(SbPsCgPush) == 160, "sb_psContactGen blob must be 160 bytes (5.12)");
static_assert(offsetof(SbPsCgPush, stackSizeBytes) == 56, "stackSizeBytes @56");
static_assert(offsetof(SbPsCgPush, stackPtr) == 64, "stackPtr @64");
static_assert(offsetof(SbPsCgPush, stackSizeNeeded) == 80, "stackSizeNeeded @80");
static_assert(offsetof(SbPsCgPush, w_outPoint) == 88, "the 72-byte writer is 8-aligned at 88");
static_assert(offsetof(SbPsCgPush, w_maxNumContacts) == 152, "maxNumContacts @152");
static_assert(sizeof(PxgParticleDbContactWriter) == 72, "PxgParticleDbContactWriter is 72 bytes");

struct CmIn { uint32_t shapeRef0, shapeRef1, cacheRef0, cacheRef1; };
struct ShapeH
{
    float sx, sy, sz, srx, sry, srz, srw;
    uint32_t materialIndex, hullLo, hullHi, type, particleOrSoftbodyId;
};
static_assert(sizeof(ShapeH) == 48, "PxgShape mirror must be 48 bytes");

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };

static const uint32_t GEOM_PARTICLESYSTEM_G = 6u;
static const uint32_t GEOM_TETRAHEDRONMESH_G = 7u;
// ---------------------------------------------------------------------------
// host vector math -- plain float, evaluated in the engine's written order
// ---------------------------------------------------------------------------
struct V3 { float x, y, z; };
static V3 vsub(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 vneg(V3 a) { return V3{ -a.x, -a.y, -a.z }; }
static V3 vmul(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static float vdot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 vcross(V3 a, V3 b)
{ return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static float vmagSq(V3 a) { return vdot(a, a); }

// PxVec3::getNormalized (PxVec3.h:286): m = magnitudeSquared(); m > 0 ? v * PxRecipSqrt(m) : 0
static V3 getNormalized(V3 v)
{
    const float m = vmagSq(v);
    return m > 0.0f ? vmul(v, 1.0f / std::sqrt(m)) : V3{ 0.0f, 0.0f, 0.0f };
}

// PxVec3::normalize (PxVec3.h:295): m = magnitude(); if (m > 0) *this /= m; return m.
// operator/= (PxVec3.h:260) forms the reciprocal ONCE -- NOT three divides.
static float normalizeInPlace(V3& v)
{
    const float m = std::sqrt(vmagSq(v));
    if (m > 0.0f)
    {
        const float r = 1.0f / m;
        v = V3{ v.x * r, v.y * r, v.z * r };
    }
    return m;
}

// Gu::closestPtPointTriangle2 (GuDistancePointTriangle.h:44), ab/ac form, including
// the final association a + ab*v + ac*w with the SCALARS scaled first.
static V3 closestPtPointTriangle2(V3 p, V3 a, V3 b, V3 c, V3 ab, V3 ac)
{
    // THE BRANCH ORDER IS PART OF THE FUNCTION.  Edge AB is tested BEFORE vertex C,
    // and cp/d5/d6 are not computed until after it.  My first transcription here put
    // the three vertex tests together and the three edge tests after them; when two
    // regions both qualify -- which happens on their shared boundary -- the first test
    // wins, so that reordering is observable.  Caught by reading
    // GuDistancePointTriangle.h:44 rather than by trusting the shape of the code.
    const V3 ap = vsub(p, a);
    const float d1 = vdot(ab, ap), d2 = vdot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    const V3 bp = vsub(p, b);
    const float d3 = vdot(ab, bp), d4 = vdot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        const float v = d1 / (d1 - d3);
        return V3{ a.x + v * ab.x, a.y + v * ab.y, a.z + v * ab.z };
    }

    const V3 cp = vsub(p, c);
    const float d5 = vdot(ab, cp), d6 = vdot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        const float w = d2 / (d2 - d6);
        return V3{ a.x + w * ac.x, a.y + w * ac.y, a.z + w * ac.z };
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        const V3 bc = vsub(c, b);
        return V3{ b.x + w * bc.x, b.y + w * bc.y, b.z + w * bc.z };
    }

    // a + ab*v + ac*w -- scalars scaled FIRST, sum left to right (the P9.37 class).
    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom, w = vc * denom;
    const V3 t = V3{ a.x + ab.x * v, a.y + ab.y * v, a.z + ab.z * v };
    return V3{ t.x + ac.x * w, t.y + ac.y * w, t.z + ac.z * w };
}

// softbodyPrimitives.cu:1864.  NOTE face 8: it flips on `> 0`, where the other three
// flip on `< 0`, and it tests against `ab` rather than the remaining edge.
static void closestPtTetrahedron(V3 point, uint32_t hint, V3 a, V3 b, V3 c, V3 d,
                                 V3& normal, V3& closestP, float& sqDist)
{
    const V3 ac = vsub(c, a), ab = vsub(b, a), ad = vsub(d, a);

    if (hint & 1u)
    {
        const V3 tPoint = closestPtPointTriangle2(point, a, b, c, ab, ac);
        const float tSqDist = vmagSq(vsub(tPoint, point));
        if (tSqDist < sqDist)
        {
            V3 n = getNormalized(vcross(ac, ab));
            if (vdot(ad, n) < 0.0f) n = vneg(n);
            normal = n; closestP = tPoint; sqDist = tSqDist;
        }
    }
    if (hint & 2u)
    {
        const V3 tPoint = closestPtPointTriangle2(point, a, b, d, ab, ad);
        const float tSqDist = vmagSq(vsub(tPoint, point));
        if (tSqDist < sqDist)
        {
            V3 n = getNormalized(vcross(ab, ad));
            if (vdot(ac, n) < 0.0f) n = vneg(n);
            normal = n; closestP = tPoint; sqDist = tSqDist;
        }
    }
    if (hint & 4u)
    {
        const V3 tPoint = closestPtPointTriangle2(point, a, c, d, ac, ad);
        const float tSqDist = vmagSq(vsub(tPoint, point));
        if (tSqDist < sqDist)
        {
            V3 n = getNormalized(vcross(ac, ad));
            if (vdot(ab, n) < 0.0f) n = vneg(n);
            normal = n; closestP = tPoint; sqDist = tSqDist;
        }
    }
    if (hint & 8u)
    {
        const V3 bd = vsub(d, b), bc = vsub(c, b);
        const V3 tPoint = closestPtPointTriangle2(point, b, c, d, bc, bd);
        const float tSqDist = vmagSq(vsub(tPoint, point));
        if (tSqDist < sqDist)
        {
            V3 n = getNormalized(vcross(bc, bd));
            if (vdot(ab, n) > 0.0f) n = vneg(n);     // > 0, not < 0
            normal = n; closestP = tPoint; sqDist = tSqDist;
        }
    }
}

// softbodyPrimitives.cu:1730.  The kernel consumes the four products ONLY as
// "all >= 0", which is what this returns.
static bool allPlanesNonNegative(V3 p, V3 a, V3 b, V3 c, V3 d)
{
    const V3 ap = vsub(p, a), ab = vsub(b, a), ac = vsub(c, a), ad = vsub(d, a);
    const V3 v0 = vcross(ab, ac);
    const float s0 = vdot(v0, ap) * vdot(v0, ad);
    const V3 v1 = vcross(ac, ad);
    const float s1 = vdot(v1, ap) * vdot(v1, ab);
    const V3 v2 = vcross(ad, ab);
    const float s2 = vdot(v2, ap) * vdot(v2, ac);
    const V3 bd = vsub(d, b), bc = vsub(c, b);
    const V3 v3 = vcross(bd, bc);
    const float s3 = vdot(v3, vsub(p, b)) * vdot(v3, vsub(a, b));
    return s0 >= 0.0f && s1 >= 0.0f && s2 >= 0.0f && s3 >= 0.0f;
}

struct DevB { PxgBufferHandle h = 0; uint64_t addr = 0; };
template <class T> static DevB mk(PxgSlangComputeDevice* dev, const std::vector<T>& v)
{
    DevB b; b.h = createBufferFrom(dev, v); b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

PXG_TEST(softBodyPsContactGen)
{
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t SB_ID = 1u, PS_ID = 2u;
    // Asymmetric AND dyadic: 0.25 + 0.0625 = 0.3125 exactly, so a particle can be put
    // exactly on the contact shell and the >-vs->= choice becomes reachable, while
    // reading one contact distance instead of the sum stays separable.
    const float CD_PARTICLE = 0.25f, CD_SOFTBODY = 0.0625f;
    const float CD_SUM = CD_PARTICLE + CD_SOFTBODY;

    // ---- the tetrahedra.  Unit corner tets on dyadic coordinates, each owning its
    // four vertices, so one can be moved or given a different surface hint without
    // disturbing the others.
    std::vector<F4> verts;
    auto pushTet = [&](float ox, float oy, float oz) {
        verts.push_back(F4{ ox,        oy,        oz,        1.0f });
        verts.push_back(F4{ ox + 1.0f, oy,        oz,        1.0f });
        verts.push_back(F4{ ox,        oy + 1.0f, oz,        1.0f });
        verts.push_back(F4{ ox,        oy,        oz + 1.0f, 1.0f });
    };
    pushTet(0.0f,  0.0f, 0.0f);   // 0
    pushTet(4.0f,  0.0f, 0.0f);   // 1
    pushTet(8.0f,  0.0f, 0.0f);   // 2
    pushTet(12.0f, 0.0f, 0.0f);   // 3  -- surface hint 0 (G5)
    pushTet(16.0f, 0.0f, 0.0f);   // 4
    pushTet(20.0f, 0.0f, 0.0f);   // 5
    // 6 is FOUR UNITS on a side, not one.  The unit corner tets have an inradius well
    // under cDistance, so no particle can be strictly inside one AND farther than
    // cDistance from every face -- which made "add the contact-distance gate to the
    // inside branch too" a mutation with no observable effect.  A side-4 tet has an
    // inradius of 4/(3+sqrt(3)) ~ 0.845, comfortably past the 0.3125 shell.
    verts.push_back(F4{ 30.0f, 0.0f, 0.0f, 1.0f });
    verts.push_back(F4{ 34.0f, 0.0f, 0.0f, 1.0f });
    verts.push_back(F4{ 30.0f, 4.0f, 0.0f, 1.0f });
    verts.push_back(F4{ 30.0f, 0.0f, 4.0f, 1.0f });
    // 7 is INVERTED (b and c swapped below): a=(40,0,0) b=(40,1,0) c=(41,0,0) d=(40,0,1),
    // detBcd = -1.  Cooking keeps inverted tets (GuCookingTetrahedronMesh.cpp:383-386)
    // and deformation produces them.  See G7.
    pushTet(40.0f, 0.0f, 0.0f);   // 7: verts 28..31
    const uint32_t NB_TETS = 8u;

    std::vector<U4> tets;
    for (uint32_t t = 0; t < NB_TETS; ++t)
        tets.push_back(U4{ 4u * t + 0u, 4u * t + 1u, 4u * t + 2u, 4u * t + 3u });
    tets[7] = U4{ 28u, 30u, 29u, 31u };   // G7: b <-> c, so the tet is inverted

    DevB bVerts = mk(dev, verts);
    if (!bVerts.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }
    DevB bTets = mk(dev, tets);

    // ---- surface hints, one byte per tet.  Tet 3 is 0 (G5); the rest name different
    // face subsets so closestPtTetrahedron's four branches are all entered and the
    // running minimum has to choose between faces.
    std::vector<uint8_t> hints = { 0xFu, 0xFu, 0x9u, 0x0u, 0xFu, 0x6u, 0xFu, 0xFu };
    hints.resize(((hints.size() + 3u) / 4u) * 4u, 0u);
    std::vector<uint32_t> hintWords(hints.size() / 4u, 0u);
    std::memcpy(hintWords.data(), hints.data(), hints.size());
    DevB bHints = mk(dev, hintWords);

    // ---- the particles.  Placed relative to tet 0 and then translated, so each one's
    // role is readable.  See the per-line notes.
    std::vector<F4> parts = {
        { 0.25f,  0.25f,   0.125f,  1.0f },  // 0: strictly INSIDE tet 0 (G4 inside branch)
        { 4.5f,  -0.125f,  0.5f,    1.0f },  // 1: outside tet 1, within the shell
        { 8.25f,  0.25f,   0.0625f, 1.0f },  // 2: inside tet 2 (hint 0x9: faces 1 and 8)
        { 12.25f, 0.25f,   0.125f,  1.0f },  // 3: inside tet 3, but hint 0 -> SKIP (G5)
        { 16.0f, -2.0f,    0.0f,    1.0f },  // 4: far outside -> sqDist > cd^2, dropped
        { 20.0f,  0.0f,   -CD_SUM,  1.0f },  // 5: EXACTLY on the shell below tet 5's a
        { 0.75f,  0.75f,   0.75f,   1.0f },  // 6: outside tet 0 past the slanted face
        // 7..9 exist so the contact count OVERSHOOTS the overflow dispatch's cap.  The
        // first version of this scene produced exactly three contacts per manager and
        // the G6 assertion refused to run against a cap it could not exceed.
        { 16.25f, 0.25f,   0.125f,  1.0f },  // 7: inside tet 4
        { 16.5f, -0.125f,  0.25f,   1.0f },  // 8: outside tet 4, within the shell
        { 8.5f,  -0.125f,  0.25f,   1.0f },  // 9: outside tet 2 (hint 0x9: two faces only)
        // 10..13 exist because the P9.39c mutation sweep left four mutants green.  Each
        // one lands in a window the first scene had no sample in.
        //
        // 10: sqDist == 0.25, which is GREATER than cDistance^2 (0.09765625) and LESS
        //     than cDistance (0.3125).  Correct code drops it; a port that forgot to
        //     square the contact distance keeps it.  Placed under the interior of tet
        //     1's y == 0 face so exactly one face is nearest.
        { 4.3f,  -0.5f,     0.2f,    1.0f },  // 10
        // 11: strictly inside the BIG tet and 0.8 from its nearest face, so sqDist
        //     (0.64) is far past cDistance^2.  The inside branch has no distance gate;
        //     adding one would drop this contact.
        { 30.8f,  0.8f,     0.8f,    1.0f },  // 11
        // 12: just outside tet 1's VERTEX a along (-1,-1,-1)/sqrt(3), 0.25 away.  The
        //     closest point is the vertex, so n points along (1,1,1)/sqrt(3) while the
        //     winning face's normal n2 is (0,0,1): dot(n, n2) ~ 0.577, which sits
        //     BETWEEN 0.5 and 0.8.  That is the only kind of sample that pins the 0.8
        //     threshold -- a face-interior closest point gives dot = +-1 and cannot.
        { 4.0f - 0.14433757f, -0.14433757f, -0.14433757f, 1.0f },  // 12
        // 13: a second contact against the big tet, from outside, so tet 6 is not only
        //     ever reached by the inside branch.
        { 30.0f, -0.125f,   1.0f,    1.0f },  // 13
        // 14 (G7, P9.86): a + (-1/8,-1/8,-1/8) against the INVERTED tet 7.  Faces abc,
        //     abd, acd all return vertex a exactly, so pa = (+0,+0,+0), every cross
        //     product in computeTetBarycentric is +0, and v = w = x = +0 / -1 = -0.
        //     PxClamp(-0, 0, 1) keeps -0 (float selects); FClamp gives +0.
        { 39.875f, -0.125f, -0.125f, 1.0f },  // 14
    };
    const uint32_t NB_PARTICLES = uint32_t(parts.size());
    DevB bParts = mk(dev, parts);

    std::vector<PxgParticleSystem> psArr(PS_ID + 1u);
    std::memset(psArr.data(), 0, psArr.size() * sizeof(PxgParticleSystem));
    psArr[PS_ID].mSortedPositions_InvMass = reinterpret_cast<float4*>(size_t(bParts.addr));
    // 5.12 reads neither the sorted-to-unsorted map nor the particle buffers here: NULL, so
    // a port that still runs the 5.6.1 filter chain faults rather than passing.
    psArr[PS_ID].mSortedToUnsortedMapping = nullptr;
    psArr[PS_ID].mCommonData.mNumParticles = NB_PARTICLES;
    DevB bPs = mk(dev, psArr);
    PXG_CHECK_MSG(sizeof(PxgParticleSystem) == 608,
                  "PXG_PARTICLE_SYSTEM_STRIDE in pxgSlangParticleDesc.slang is 608 and must "
                  "equal sizeof(PxgParticleSystem), which is %zu", sizeof(PxgParticleSystem));

    // ---- softbody descriptor
    PxgSoftBody sb;
    std::memset(&sb, 0, sizeof(sb));
    sb.mTetIndices = reinterpret_cast<uint4*>(size_t(bTets.addr));
    sb.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bVerts.addr));
    sb.mTetMeshSurfaceHint = reinterpret_cast<PxU8*>(size_t(bHints.addr));
    sb.mNumTets = NB_TETS;
    sb.mNumVerts = uint32_t(verts.size());
    std::vector<PxgSoftBody> sbArr(SB_ID + 1u);
    std::memset(sbArr.data(), 0, sbArr.size() * sizeof(PxgSoftBody));
    sbArr[SB_ID] = sb;
    DevB bSb = mk(dev, sbArr);

    // ---- shapes: two managers with the particle system in OPPOSITE slots, so routing
    // by TYPE is separable from slot order.
    std::vector<ShapeH> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(ShapeH));
    shapes[0].type = GEOM_PARTICLESYSTEM_G;  shapes[0].particleOrSoftbodyId = PS_ID;
    shapes[0].sx = shapes[0].sy = shapes[0].sz = 1.0f; shapes[0].srw = 1.0f;
    shapes[1].type = GEOM_TETRAHEDRONMESH_G; shapes[1].particleOrSoftbodyId = SB_ID;
    shapes[1].sx = shapes[1].sy = shapes[1].sz = 1.0f; shapes[1].srw = 1.0f;
    DevB bShapes = mk(dev, shapes);

    std::vector<CmIn> cms = { { 0u, 1u, 0u, 1u }, { 1u, 0u, 1u, 0u } };
    DevB bCms = mk(dev, cms);
    std::vector<float> cd = { CD_PARTICLE, CD_SOFTBODY };
    DevB bCd = mk(dev, cd);
    std::vector<float> xf(2 * 8, 0.0f); xf[3] = 1.0f; xf[8 + 3] = 1.0f;   // ABI parity only
    DevB bXf = mk(dev, xf);

    // ---- the midphase pairs.  ONE MANAGER PER DISPATCH.
    //
    // The identity a contact carries is (particle ref, tet ref) and that does NOT mention the
    // contact manager, so with both managers in one dispatch the two contacts for the same
    // particle-tet pair would be indistinguishable in the output.  One manager per dispatch
    // keeps the key unique.
    struct Expect
    {
        uint64_t ref0, ref1;
        F4 point, normalPen;
    };

    uint32_t skippedHint = 0u;
    uint32_t insideBranch = 0u, outsideBranch = 0u, droppedByDistance = 0u, n2Substituted = 0u;

    // `stackLimit` is min(numFound, stackSize) -- the kernel processes only that many
    // stack entries, in order.  The mirror therefore walks the stack rather than the
    // (particle, tet) product, because with a clamp the two are not the same set.
    auto mirrorFor = [&](uint32_t cmIdx, uint32_t stackLimit) {
        std::vector<Expect> out;
        uint32_t seen = 0u;
        const CmIn& cin = cms[cmIdx];
        uint32_t sP = cin.shapeRef0, cP = cin.cacheRef0, sS = cin.shapeRef1, cS = cin.cacheRef1;
        if (shapes[cin.shapeRef0].type != GEOM_PARTICLESYSTEM_G)
        { sP = cin.shapeRef1; cP = cin.cacheRef1; sS = cin.shapeRef0; cS = cin.cacheRef0; }
        const float cDistance = cd[cP] + cd[cS];
        const uint32_t particleSystemId = shapes[sP].particleOrSoftbodyId;
        const uint32_t softbodyId = shapes[sS].particleOrSoftbodyId;

        for (uint32_t particleIndex = 0; particleIndex < NB_PARTICLES; ++particleIndex)
            for (uint32_t tetrahedronIdx = 0; tetrahedronIdx < NB_TETS; ++tetrahedronIdx)
            {
                if (seen++ >= stackLimit) continue;      // past the clamp: never read
                const U4& ti = tets[tetrahedronIdx];
                const uint32_t surfaceMask = hints[tetrahedronIdx];

                // 5.12: the surface hint is the only gate (the filter chain is gone).
                if (surfaceMask == 0u) { ++skippedHint; continue; }

                const F4& av = verts[ti.x]; const F4& bv = verts[ti.y];
                const F4& cv = verts[ti.z]; const F4& dv = verts[ti.w];
                const V3 a{ av.x, av.y, av.z }, b{ bv.x, bv.y, bv.z };
                const V3 c{ cv.x, cv.y, cv.z }, d{ dv.x, dv.y, dv.z };
                const V3 pp{ parts[particleIndex].x, parts[particleIndex].y,
                             parts[particleIndex].z };

                const bool inside = allPlanesNonNegative(pp, a, b, c, d);

                V3 closestP{ 0, 0, 0 }, n2{ 0, 0, 0 };
                float sqDist = 3.4028234663852886e+38f;
                closestPtTetrahedron(pp, surfaceMask, a, b, c, d, n2, closestP, sqDist);

                V3 n = vneg(vsub(pp, closestP));
                float pen;
                if (inside)
                {
                    ++insideBranch;
                    pen = vdot(n, n2);
                    n = n2;
                }
                else
                {
                    const float sqCd = cDistance * cDistance;
                    if (sqDist > sqCd) { ++droppedByDistance; continue; }
                    ++outsideBranch;
                    pen = normalizeInPlace(n);
                    if (sqDist < 1e-8f || vdot(n, n2) < 0.8f) { n = n2; ++n2Substituted; }
                }

                Expect e;
                e.ref0 = PxgEncodeParticleRef(particleSystemId, particleIndex);
                e.ref1 = PxgEncodeDeformableTetrahedron(softbodyId, tetrahedronIdx);
                e.point = F4{ pp.x, pp.y, pp.z, 0.0f };
                e.normalPen = F4{ n.x, n.y, n.z, pen };
                out.push_back(e);
            }
        return out;
    };

    const uint32_t POISON = 0xCDCDCDCDu;

    // One dispatch: build the stack for a single manager, run, and compare.  maxContacts
    // is a parameter so the same code covers "everything fits" and "the writer overflows".
    auto runOne = [&](uint32_t cmIdx, uint32_t maxContacts, uint32_t stackPairs,
                      const std::vector<Expect>& exp, const char* what)
    {
        std::vector<U4> stack;
        for (uint32_t pi = 0; pi < NB_PARTICLES; ++pi)
            for (uint32_t t = 0; t < NB_TETS; ++t)
                stack.push_back(U4{ cmIdx, pi, t, 0u });
        const uint32_t nbStack = uint32_t(stack.size());
        DevB bStack = mk(dev, stack);
        std::vector<uint32_t> numFound = { nbStack };
        DevB bNum = mk(dev, numFound);

        const uint32_t slots = maxContacts + 4u;
        std::vector<F4> zero4(slots, F4{ 0, 0, 0, 0 });
        std::vector<uint32_t> poisonW(slots * 4u, POISON);
        std::vector<uint32_t> poisonBary(slots * 4u, POISON);
        std::vector<uint32_t> poison2(slots * 2u, POISON);
        std::vector<uint32_t> poison1(slots, POISON);
        std::vector<uint32_t> one = { 0u };
        DevB bPoint = mk(dev, zero4), bNP = mk(dev, zero4), bBary = mk(dev, poisonBary);
        DevB bInfo = mk(dev, poisonW), bByX = mk(dev, poison2), bTemp = mk(dev, poison1);
        DevB bIdx = mk(dev, poison1), bTotal = mk(dev, one), bNeeded = mk(dev, one);

        SbPsCgPush p = {};
        p.toleranceLength = 1.0f;
        p.cmInputs = bCms.addr; p.transformCache = bXf.addr; p.contactDistance = bCd.addr;
        p.gpuShapes = bShapes.addr;
        p.particleSystems = bPs.addr; p.softbodies = bSb.addr;
        // stackPairs may be SMALLER than nbStack: numPairs = min(numFound, stackSize)
        // clamps the loop, while stackSizeNeededOnDevice must report the UNCLAMPED
        // count.  With the two equal, using the clamped one is unobservable, which is
        // how S10-needed-uses-clamped survived the first sweep.
        p.stackSizeBytes = stackPairs * 16u;
        p.stackPtr = bStack.addr; p.midphasePairsNum = bNum.addr;
        p.stackSizeNeeded = bNeeded.addr;
        p.w_outPoint = bPoint.addr; p.w_outNormalPen = bNP.addr;
        p.w_outBarycentric = bBary.addr; p.w_outContactInfo = bInfo.addr;
        p.w_totalContactCount = bTotal.addr; p.w_contactByParticle = bByX.addr;
        p.w_tempContactByParticle = bTemp.addr; p.w_contactIndexSortedByParticle = bIdx.addr;
        p.w_maxNumContacts = maxContacts;

        Kernel k("softBodyPsContactGenBda.spv", "sb_psContactGenLaunch", {}, sizeof(SbPsCgPush));
        k.dispatch(64u, {}, p, 1u, 1u);
        dev->flush();

        const std::vector<uint32_t> gotTotal = downloadBuffer<uint32_t>(dev, bTotal.h, 1);
        const std::vector<uint32_t> gotNeeded = downloadBuffer<uint32_t>(dev, bNeeded.h, 1);
        PXG_CHECK_MSG(gotTotal[0] == uint32_t(exp.size()),
                      "%s: totalContactCount %u, mirror says %zu -- the counter increments "
                      "for every contact the kernel REACHES, including dropped writes",
                      what, gotTotal[0], exp.size());
        PXG_CHECK_MSG(gotNeeded[0] == nbStack * 16u,
                      "%s: stackSizeNeededOnDevice %u, expected %u (the UNCLAMPED pair count "
                      "times 16, written by grid thread 0)", what, gotNeeded[0], nbStack * 16u);

        const uint32_t written = std::min<uint32_t>(maxContacts, uint32_t(exp.size()));
        const std::vector<F4> gPoint = downloadBuffer<F4>(dev, bPoint.h, slots);
        const std::vector<F4> gNP = downloadBuffer<F4>(dev, bNP.h, slots);
        const std::vector<uint32_t> gBary = downloadBuffer<uint32_t>(dev, bBary.h, slots * 4u);
        const std::vector<uint32_t> gInfo = downloadBuffer<uint32_t>(dev, bInfo.h, slots * 4u);
        const std::vector<uint32_t> gByX = downloadBuffer<uint32_t>(dev, bByX.h, slots * 2u);
        const std::vector<uint32_t> gTemp = downloadBuffer<uint32_t>(dev, bTemp.h, slots);
        const std::vector<uint32_t> gIdx = downloadBuffer<uint32_t>(dev, bIdx.h, slots);

        auto sameF4 = [](const F4& x, const F4& y)
        { return std::memcmp(&x, &y, sizeof(F4)) == 0; };

        uint32_t matched = 0u, mismatched = 0u;
        for (uint32_t i = 0; i < written; ++i)
        {
            const uint64_t r0 = uint64_t(gInfo[i * 4 + 0]) | (uint64_t(gInfo[i * 4 + 1]) << 32);
            const uint64_t r1 = uint64_t(gInfo[i * 4 + 2]) | (uint64_t(gInfo[i * 4 + 3]) << 32);
            const Expect* found = nullptr;
            for (const Expect& e : exp)
                if (e.ref0 == r0 && e.ref1 == r1) { found = &e; break; }
            if (!found)
            {
                ++mismatched;
                std::printf("  %s slot %u: refs (%016llx, %016llx) are not in the mirror\n",
                            what, i, (unsigned long long)r0, (unsigned long long)r1);
                continue;
            }
            const uint64_t byX = uint64_t(gByX[i * 2 + 0]) | (uint64_t(gByX[i * 2 + 1]) << 32);
            // G8 / writeVolumeContact: no barycentric -- the poisoned slot is untouched.
            const bool baryUntouched = gBary[i * 4 + 0] == POISON && gBary[i * 4 + 1] == POISON &&
                                       gBary[i * 4 + 2] == POISON && gBary[i * 4 + 3] == POISON;
            const bool ok = sameF4(gPoint[i], found->point) && sameF4(gNP[i], found->normalPen) &&
                            baryUntouched && byX == found->ref0 &&
                            gTemp[i] == PxgGetContactSortKey(found->ref0) && gIdx[i] == i;
            if (!ok)
            {
                ++mismatched;
                std::printf("  %s slot %u (refs %016llx, %016llx):\n", what, i,
                            (unsigned long long)r0, (unsigned long long)r1);
                std::printf("    point  got (%a %a %a %a) want (%a %a %a %a)\n",
                            gPoint[i].x, gPoint[i].y, gPoint[i].z, gPoint[i].w,
                            found->point.x, found->point.y, found->point.z, found->point.w);
                std::printf("    nrmPen got (%a %a %a %a) want (%a %a %a %a)\n",
                            gNP[i].x, gNP[i].y, gNP[i].z, gNP[i].w, found->normalPen.x,
                            found->normalPen.y, found->normalPen.z, found->normalPen.w);
                std::printf("    bary   %s\n", baryUntouched ? "untouched" : "WRITTEN (5.12 writes none)");
                std::printf("    byX (%u,%u) temp %u idx %u\n",
                            gByX[i * 2 + 0], gByX[i * 2 + 1], gTemp[i], gIdx[i]);
                continue;
            }
            ++matched;
        }
        PXG_CHECK_MSG(mismatched == 0u, "%s: %u of %u written contacts disagree with the mirror",
                      what, mismatched, written);
        PXG_CHECK_MSG(matched == written, "%s: matched %u of %u", what, matched, written);

        for (uint32_t i = maxContacts; i < slots; ++i)
            PXG_CHECK_MSG(gTemp[i] == POISON && gIdx[i] == POISON,
                          "%s: slot %u is past maxNumContacts %u but was written "
                          "(temp %u, idx %u)", what, i, maxContacts, gTemp[i], gIdx[i]);
        return written;
    };

    const uint32_t NB_STACK = NB_PARTICLES * NB_TETS;
    const uint32_t CLAMPED_B = NB_STACK - 7u;    // manager 1 runs with a SHORT stack
    const std::vector<Expect> expA = mirrorFor(0u, NB_STACK);
    const std::vector<Expect> expB = mirrorFor(1u, CLAMPED_B);
    PXG_CHECK_MSG(CLAMPED_B < NB_STACK,
                  "the overflow dispatch's stack must hold FEWER pairs than the midphase "
                  "reports, or numPairs == numFound and reading the clamped count instead "
                  "of the unclamped one is invisible");

    // Coverage FIRST: every mechanism the scene claims to separate must have fired.
    // These run before the comparison because a scene that never enters a branch makes
    // a green comparison meaningless -- which is exactly what happened on the first run,
    // where a (5.6.1) whole-body filter entry had silently deleted the inside branch.
    PXG_CHECK_MSG(skippedHint > 0u, "no pair was skipped by surfaceMask == 0 (G5)");
    PXG_CHECK_MSG(insideBranch > 0u, "the inside-the-tetrahedron branch never ran (G4)");
    PXG_CHECK_MSG(outsideBranch > 0u, "the outside branch never ran (G4)");
    PXG_CHECK_MSG(droppedByDistance > 0u,
                  "no pair was dropped by the squared contact-distance test, so the gate "
                  "that exists only in the outside branch (G4) is not exercised");
    PXG_CHECK_MSG(mirrorFor(1u, NB_STACK).size() > expB.size(),
                  "the clamped stack tail must contain a contact");

    // A: everything fits, so every contact is checked bit for bit.
    const uint32_t wroteA = runOne(0u, 64u, NB_STACK, expA, "cm0/all-fit");
    // B: the writer deliberately overflows (G6).
    const uint32_t MAX_B = 3u;
    PXG_CHECK_MSG(expB.size() > MAX_B,
                  "manager 1 generates %zu contacts but the overflow dispatch caps at %u; "
                  "it must OVERSHOOT for the dropped-write path (G6) to be measured",
                  expB.size(), MAX_B);
    const uint32_t wroteB = runOne(1u, MAX_B, CLAMPED_B, expB, "cm1/overflow+short-stack");

    std::printf("    [ note ] cm0: %zu contacts (%u written); cm1: %zu contacts (%u written, "
                "%u dropped by the writer); skipped %u by hint; branches %u inside, %u outside "
                "(%u took n2), %u dropped by distance\n",
                expA.size(), wroteA, expB.size(), wroteB, uint32_t(expB.size()) - wroteB,
                skippedHint, insideBranch, outsideBranch, n2Substituted, droppedByDistance);
}
} // namespace
