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

// testFemClothSolve512.cpp -- the PhysX 5.12 FEM cloth (deformable surface) constraint
// kernels of FEMClothConstraintPrep.cu / FEMClothExternalSolve.cu, checked word for word
// against the host transcription of the 5.12 CUDA (pxgFemClothSolveRef.h on top of
// pxgSoftBodySolveRef.h and pxgParticleSolveRef.h):
//
//   cloth-rigid contacts      cloth_rigidContactPrepareLaunch, cloth_queryRigidClothContact-
//                             ReferenceCountLaunch, cloth_solveRigidClothCollisionLaunch
//   cloth-rigid attachments   cloth_rigidAttachmentPrepareLaunch, cloth_queryRigidCloth-
//                             AttachmentReferenceCountLaunch, cloth_solveRigidClothAttachmentLaunch
//   cloth-particle contacts   cloth_particleContactPrepareLaunch, cloth_queryCPContact-
//                             ReferenceCountLaunch, cloth_solveCPOutputClothDeltaVLaunch,
//                             cloth_solveCPOutputParticleDeltaVLaunch
//   cloth-cloth attachments   cloth_queryClothClothAttachmentReferenceCountLaunch,
//                             cloth_solveOutputAttachmentClothClothDeltaVLaunch
//   cloth-cloth contacts      cloth_clothContactPrepareLaunch, cloth_queryClothClothContact-
//                             {VT,EE}CountLaunch, cloth_solveClothClothDelta{VT,EE}Launch
//   finalize of an iteration  cloth_applyExternalDeltasLaunch (position / velocity iteration)
//
// THE SCENE.  A cloth pool with a zero DECOY in the middle (cloths 0 and 2 are real), so a
// kernel that confuses ids reads zeros.  Every triangle has its own three vertices and there
// are lone vertices for the vertex refs; the contacts / attachments are spread so that no
// vertex receives more than two scatters per dispatch, where the float atomics commute
// (pxgSoftBodySolveRef.h).  Inverse mass 0 vertices, barycentric weights within 1e-3 of
// zero (the reference-count gate), INVALID and pre-ACTIVE pairs, two blocks of 32 with a
// partial second, PGS / TGS and position / velocity iterations are covered.  The rigid
// and particle sides are pxgParticleSolveWorld's.

#include "pxgTestFramework.h"
#include "pxgParticleSolveWorld.h"
#include "pxgFemClothSolveRef.h"
#include "host_defines.h"
#include "vector_functions.h"
#include "PxgFEMCloth.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace pxgpsref;
using namespace pxgpsworld;
using namespace pxgsbref;
using namespace pxgfcref;

static const float kDt = 1.0f / 60.0f;
static const uint32_t GRID = 1024u;   // PxgSoftBodyKernelGridDim::SB_UPDATEROTATION

// Block strides / offsets (PxgDeformableConstraints.h, 5.12; asserted against the headers
// in testDeformableRef512).
static const uint32_t RDBC_STRIDE = 2688u, RDBC_raXn_resp = 0u, RDBC_raXnF0_resp = 512u, RDBC_raXnF1_resp = 1024u,
                      RDBC_normal_errorW = 1536u, RDBC_fricTan0_invMass0 = 2048u, RDBC_maxPenBiasClamp = 2560u;
static const uint32_t PDBC_STRIDE = 1152u, PDBC_normal_pen = 0u, PDBC_barycentric = 512u, PDBC_maxPenBiasClamp = 1024u;
static const uint32_t RDBA_STRIDE = 3200u, RDBA_barycentric = 0u, RDBA_raXn0_biasW = 512u, RDBA_raXn1_biasW = 1024u,
                      RDBA_raXn2_biasW = 1536u, RDBA_velMultiplierXYZ_invMassW = 2048u, RDBA_deformableRef = 2560u,
                      RDBA_rigidId = 2816u, RDBA_rigidBodyRefCount = 3072u;
static const uint32_t DBDBA_STRIDE = 1536u, DBDBA_barycentric0 = 0u, DBDBA_barycentric1 = 512u,
                      DBDBA_deformableRef0 = 1024u, DBDBA_deformableRef1 = 1280u;
static const uint32_t DATT_STRIDE = 48u;

static const uint32_t POISON = 0xC5C5C5C5u;

template <typename T> static void putAt(std::vector<uint8_t>& b, size_t off, const T& v) { std::memcpy(b.data() + off, &v, sizeof(T)); }
template <typename T> static T getAt(const std::vector<uint8_t>& b, size_t off) { T v; std::memcpy(&v, b.data() + off, sizeof(T)); return v; }

static float f32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

static bool cmpF4(const char* what, uint32_t i, F4 got, F4 want, uint32_t& bad)
{
    if (bitsEqual(got, want))
        return true;
    if (++bad <= 8)
        std::printf("    %s[%u] got (%a %a %a %a) want (%a %a %a %a)\n", what, i, got.x, got.y, got.z, got.w, want.x,
                    want.y, want.z, want.w);
    return false;
}

static bool cmpF(const char* what, uint32_t i, float got, float want, uint32_t& bad)
{
    if (bitsEqual(got, want))
        return true;
    if (++bad <= 8)
        std::printf("    %s[%u] got %a want %a\n", what, i, got, want);
    return false;
}

static void cmpVec(const char* what, const std::vector<F4>& got, const std::vector<F4>& want, uint32_t& bad)
{
    for (uint32_t i = 0; i < want.size(); ++i)
        cmpF4(what, i, got[i], want[i], bad);
}

static F4 randomBary(Rng& rng, bool tiny)
{
    float b[3] = { rng.uni(0.1f, 1.0f), rng.uni(0.1f, 1.0f), rng.uni(0.1f, 1.0f) };
    if (tiny)
        b[rng.next() % 3u] = 5.0e-4f;   // within the 1e-3 reference-count gate
    const float s = b[0] + b[1] + b[2];
    return F4{ b[0] / s, b[1] / s, b[2] / s, 0.0f };
}

// ===========================================================================
// The cloth world
// ===========================================================================
// NTRI triangles per cloth, each with its own three vertices (3t .. 3t+2), then NLOOSE lone
// vertices for the vertex refs.
static const uint32_t NTRI = 48u, NLOOSE = 40u, NV = 3u * NTRI + NLOOSE;

struct ClothWorld
{
    PxgSlangComputeDevice* dev = nullptr;
    std::vector<PxgBufferHandle> owned;

    HostCloth cloth[3];   // 0 and 2 real, 1 the decoy
    std::vector<float> surfaceFriction;   // PxsDeformableSurfaceMaterialData dynamicFriction per material

    PxgBufferHandle bDelta[3] = { 0, 0, 0 }, bPos[3] = { 0, 0, 0 }, bVel[3] = { 0, 0, 0 }, bAcc[3] = { 0, 0, 0 };
    uint64_t pool = 0, surfMaterials = 0;

    ~ClothWorld() { for (PxgBufferHandle h : owned) dev->destroyBuffer(h); }

    template <typename T> PxgBufferHandle make(const std::vector<T>& v)
    {
        PxgBufferHandle h = createBufferFrom(dev, v);
        owned.push_back(h);
        return h;
    }
    uint64_t addr(PxgBufferHandle h) const { return dev->getBufferDeviceAddress(h); }

    std::vector<F4> readDelta(uint32_t c) const { return downloadBuffer<F4>(dev, bDelta[c], cloth[c].delta.size()); }
    std::vector<F4> readPos(uint32_t c) const { return downloadBuffer<F4>(dev, bPos[c], cloth[c].pos.size()); }
    std::vector<F4> readVel(uint32_t c) const { return downloadBuffer<F4>(dev, bVel[c], cloth[c].vel.size()); }
    std::vector<F4> readAcc(uint32_t c) const { return downloadBuffer<F4>(dev, bAcc[c], cloth[c].accDelta.size()); }
};

// Host side only; uploadClothWorld puts it on the device (tests that shape the geometry edit
// the host cloths in between).
static void buildClothHost(ClothWorld& C, uint32_t seed)
{
    Rng rng{ seed | 1u };
    C.surfaceFriction = { 0.45f, 0.2f, 0.7f };
    for (uint32_t ci = 0; ci < 3; ci += 2)
    {
        HostCloth& c = C.cloth[ci];
        for (uint32_t t = 0; t < NTRI; ++t)
        {
            c.tris.push_back(U4{ 3 * t + 1, 3 * t + 2, 3 * t + 0, 0x5u });   // .w: owned-edge bits, not an index
            c.matIdx.push_back(uint16_t(rng.next() % 3u));
        }
        for (uint32_t v = 0; v < NV; ++v)
        {
            const float invMass = (v % 7u == 3u) ? 0.0f : rng.uni(0.5f, 3.0f);
            c.pos.push_back(F4{ rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), invMass });
            c.vel.push_back(F4{ rng.uni(-2.f, 2.f), rng.uni(-2.f, 2.f), rng.uni(-2.f, 2.f), invMass });
            c.accDelta.push_back(F4{ rng.uni(-0.03f, 0.03f), rng.uni(-0.03f, 0.03f), rng.uni(-0.03f, 0.03f), invMass });
            c.delta.push_back(F4{ 0.0f, 0.0f, 0.0f, 0.0f });
            c.dynFriction.push_back(rng.uni(0.0f, 0.9f));
            c.restPos.push_back(F4{ rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), 1.0f });
            c.prevPos.push_back(F4{ rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), 1.0f });
        }
        c.restOffset = (ci == 0) ? 0.015f : 0.02f;
        c.penBiasClamp = (ci == 0) ? -0.8f : -1.6f;
        c.selfCollisionFilterDistance = (ci == 0) ? 0.05f : 0.08f;
    }
}

static void uploadClothWorld(ClothWorld& C, PxgSlangComputeDevice* dev)
{
    C.dev = dev;
    std::vector<physx::PxgFEMCloth> desc(3);
    std::memset(desc.data(), 0, desc.size() * sizeof(physx::PxgFEMCloth));
    for (uint32_t ci = 0; ci < 3; ci += 2)
    {
        HostCloth& c = C.cloth[ci];
        std::vector<uint32_t> mi((c.matIdx.size() + 1) / 2 + 2, 0u);
        std::memcpy(mi.data(), c.matIdx.data(), c.matIdx.size() * 2);
        C.bDelta[ci] = C.make(c.delta);
        C.bPos[ci] = C.make(c.pos);
        C.bVel[ci] = C.make(c.vel);
        C.bAcc[ci] = C.make(c.accDelta);
        physx::PxgFEMCloth& d = desc[ci];
        d.mTriangleVertexIndices_ownedEdges = reinterpret_cast<uint4*>(size_t(C.addr(C.make(c.tris))));
        d.mPosition_InvMass = reinterpret_cast<float4*>(size_t(C.addr(C.bPos[ci])));
        d.mVelocity_InvMass = reinterpret_cast<float4*>(size_t(C.addr(C.bVel[ci])));
        d.mAccumulatedDeltaPos = reinterpret_cast<float4*>(size_t(C.addr(C.bAcc[ci])));
        d.mDeltaPos = reinterpret_cast<float4*>(size_t(C.addr(C.bDelta[ci])));
        d.mRestPosition = reinterpret_cast<float4*>(size_t(C.addr(C.make(c.restPos))));
        d.mPrevPositionInContactOffset = reinterpret_cast<float4*>(size_t(C.addr(C.make(c.prevPos))));
        d.mDynamicFrictions = reinterpret_cast<float*>(size_t(C.addr(C.make(c.dynFriction))));
        d.mMaterialIndices = reinterpret_cast<physx::PxU16*>(size_t(C.addr(C.make(mi))));
        d.mRestOffset = c.restOffset;
        d.mPenBiasClamp = c.penBiasClamp;
        d.mSelfCollisionFilterDistance = c.selfCollisionFilterDistance;
        d.mNbVerts = uint32_t(c.pos.size());
        d.mNbTriangles = uint32_t(c.tris.size());
    }
    C.pool = C.addr(C.make(desc));
    // PxsDeformableSurfaceMaterialData: 32 B, dynamicFriction @8.
    std::vector<float> sm(8 * C.surfaceFriction.size(), 7.0f);
    for (size_t m = 0; m < C.surfaceFriction.size(); ++m) sm[8 * m + 2] = C.surfaceFriction[m];
    C.surfMaterials = C.addr(C.make(sm));
}

static void buildClothWorld(ClothWorld& C, PxgSlangComputeDevice* dev, uint32_t seed)
{
    buildClothHost(C, seed);
    uploadClothWorld(C, dev);
}

// A per-(cloth, vertex) scatter budget: every vertex a ref can write is charged, and no vertex
// takes more than two, so the float atomics of one dispatch commute.
struct ScatterBudget
{
    uint8_t n[3][NV] = {};
    bool charge(uint64_t r)
    {
        uint8_t* c = n[refId(r) % 3u];
        const uint32_t e = refFeature(r);
        if (refType(r) == REF_eVERTEX)
            return ++c[e] <= 2u;
        bool ok = true;
        for (uint32_t k = 0; k < 3; ++k) ok = (++c[3 * e + k] <= 2u) && ok;
        return ok;
    }
    bool fits(uint64_t r0, bool r0IsCloth, uint64_t r1)
    {
        ScatterBudget trial = *this;
        const bool ok0 = r0IsCloth ? trial.charge(r0) : true;
        const bool ok1 = trial.charge(r1);
        if (!ok0 || !ok1) return false;
        *this = trial;
        return true;
    }
};

// A random cloth feature ref: a triangle of cloth 0 / 2 or one of its lone vertices.
static uint64_t randomClothRef(Rng& rng, uint32_t ci, bool allowVertex)
{
    if (allowVertex && rng.next() % 3u == 0u)
        return encodeRef(ci, REF_eVERTEX, 3u * NTRI + rng.next() % NLOOSE);
    return encodeRef(ci, REF_eTRIANGLE, rng.next() % NTRI);
}

// ===========================================================================
// Cloth-rigid contacts
// ===========================================================================
struct RcContact
{
    PairInfo pair;     // ref0 = rigid node, ref1 = triangle or vertex ref of cloth 0 or 2
    F4 point_rest;     // contacts_restW
    F4 normalPen;
    F4 bary;
    uint16_t rigidMat;
};

struct RdbLane { F4 raXn_resp, raXnF0_resp, raXnF1_resp, normal_errorW, fricTan0_invMass0; float maxPenBiasClamp; };

// prepareRigidDbContact (deformableUtils.cuh), on the World's rigid data (the same
// transcription as testSoftBodySolve512's).
static RdbLane prepareRigidDbContactRef(const World& W, V3 normal, V3 p, float pen, V3 delta, uint64_t rigidId,
                                        float penBiasClampFEM)
{
    RdbLane b;
    V3 t0, t1;
    computeBasisVectors(normal, t0, t1);
    float penBiasClampRigid;
    float invMass0;
    if (nodeIsArticulation(rigidId))
    {
        const World::Link& L = W.links[nodeLinkId(rigidId)];
        penBiasClampRigid = L.penBiasClamp;
        const V3 ra = sub(p, L.p);
        const V3 raXn = cross(ra, normal), raXF0 = cross(ra, t0), raXF1 = cross(ra, t1);
        V3 top, bottom;
        spatialMul(L.response, normal, raXn, top, bottom);
        const float resp0 = dot(top, raXn) + dot(bottom, normal);
        spatialMul(L.response, t0, raXF0, top, bottom);
        const float respF0 = dot(top, raXF0) + dot(bottom, t0);
        spatialMul(L.response, t1, raXF1, top, bottom);
        const float respF1 = dot(top, raXF1) + dot(bottom, t1);
        b.raXn_resp = F4{ raXn.x, raXn.y, raXn.z, resp0 };
        b.raXnF0_resp = F4{ raXF0.x, raXF0.y, raXF0.z, respF0 };
        b.raXnF1_resp = F4{ raXF1.x, raXF1.y, raXF1.z, respF1 };
        invMass0 = 1.0f;
    }
    else
    {
        const uint32_t idx = nodeIsStatic(rigidId) ? 0u : W.solverBodyIndices[nodeIndex(rigidId)];
        const World::Body& B = W.bodies[idx];
        penBiasClampRigid = B.angVel_penBiasClamp.w;
        invMass0 = B.linVel_invMass.w;
        const bool isKinematic = (invMass0 == 0.0f) && !nodeIsStatic(rigidId);
        const V3 ra = sub(p, xyz(B.p));
        const V3 raXn = cross(ra, normal), raXF0 = cross(ra, t0), raXF1 = cross(ra, t1);
        const V3 a = mat33Mul(B.sqrtInvInertia, raXn);
        const float resp0 = (dot(a, a)) + invMass0;
        const V3 f0 = mat33Mul(B.sqrtInvInertia, raXF0), f1 = mat33Mul(B.sqrtInvInertia, raXF1);
        const float respF0 = (dot(f0, f0)) + invMass0;
        const float respF1 = (dot(f1, f1)) + invMass0;
        if (isKinematic)
        {
            b.raXn_resp = F4{ raXn.x, raXn.y, raXn.z, resp0 };
            b.raXnF0_resp = F4{ raXF0.x, raXF0.y, raXF0.z, respF0 };
            b.raXnF1_resp = F4{ raXF1.x, raXF1.y, raXF1.z, respF1 };
        }
        else
        {
            b.raXn_resp = F4{ a.x, a.y, a.z, resp0 };
            b.raXnF0_resp = F4{ f0.x, f0.y, f0.z, respF0 };
            b.raXnF1_resp = F4{ f1.x, f1.y, f1.z, respF1 };
        }
    }
    const float error = pen + dot(delta, normal);
    b.normal_errorW = F4{ normal.x, normal.y, normal.z, error };
    b.fricTan0_invMass0 = F4{ t0.x, t0.y, t0.z, invMass0 };
    b.maxPenBiasClamp = selectMax(penBiasClampRigid, penBiasClampFEM);
    return b;
}

static void writeRdbLane(std::vector<uint8_t>& blocks, uint32_t w, const RdbLane& b)
{
    const size_t base = size_t(w / 32u) * RDBC_STRIDE;
    const uint32_t l = w & 31u;
    putAt(blocks, base + RDBC_raXn_resp + l * 16, b.raXn_resp);
    putAt(blocks, base + RDBC_raXnF0_resp + l * 16, b.raXnF0_resp);
    putAt(blocks, base + RDBC_raXnF1_resp + l * 16, b.raXnF1_resp);
    putAt(blocks, base + RDBC_normal_errorW + l * 16, b.normal_errorW);
    putAt(blocks, base + RDBC_fricTan0_invMass0 + l * 16, b.fricTan0_invMass0);
    putAt(blocks, base + RDBC_maxPenBiasClamp + l * 4, b.maxPenBiasClamp);
}

static RdbLane readRdbLane(const std::vector<uint8_t>& blocks, uint32_t w)
{
    const size_t base = size_t(w / 32u) * RDBC_STRIDE;
    const uint32_t l = w & 31u;
    RdbLane b;
    b.raXn_resp = getAt<F4>(blocks, base + RDBC_raXn_resp + l * 16);
    b.raXnF0_resp = getAt<F4>(blocks, base + RDBC_raXnF0_resp + l * 16);
    b.raXnF1_resp = getAt<F4>(blocks, base + RDBC_raXnF1_resp + l * 16);
    b.normal_errorW = getAt<F4>(blocks, base + RDBC_normal_errorW + l * 16);
    b.fricTan0_invMass0 = getAt<F4>(blocks, base + RDBC_fricTan0_invMass0 + l * 16);
    b.maxPenBiasClamp = getAt<float>(blocks, base + RDBC_maxPenBiasClamp + l * 4);
    return b;
}

// The cloth delta a contact prepare blends: the vertex's accumulated delta, or the triangle's
// barycentric blend of it.
static F4 clothContactDelta(const HostCloth& c, uint64_t ref, F4 bary)
{
    const uint32_t e = refFeature(ref);
    if (refType(ref) == REF_eVERTEX)
        return c.accDelta[e];
    return projectTri(c.accDelta, c.tris[e], bary);
}

// 45 contacts over the two real cloths, a few INVALID pairs, at most two scatters a vertex.
static std::vector<RcContact> makeRcContacts(const ClothWorld& C, uint32_t seed)
{
    Rng rng{ seed * 7u + 3u };
    const uint64_t rigids[6] = { staticNode(), makeRigidNode(4), makeRigidNode(6), makeRigidNode(8),
                                 makeArticulationNode(10, 2), makeArticulationNode(10, 1) };
    ScatterBudget budget;
    std::vector<RcContact> cs;
    const uint32_t NC = 45u;
    for (uint32_t tries = 0; cs.size() < NC && tries < 10000u; ++tries)
    {
        RcContact rc;
        const uint32_t c = uint32_t(cs.size());
        const uint32_t ci = (c % 3u == 1u) ? 2u : 0u;
        const uint64_t feature = randomClothRef(rng, ci, true);
        if (c % 11u == 4u)
            rc.pair.ref1 = encodeRef(ci, 0, 0) | REF_INVALID;
        else
        {
            if (!budget.fits(0, false, feature)) continue;
            rc.pair.ref1 = feature;
        }
        rc.pair.ref0 = rigids[rng.next() % 6u];
        V3 n = v3(rng.uni(-0.5f, 0.5f), 1.0f, rng.uni(-0.5f, 0.5f));
        if (c % 9u == 0u) n = v3(0.0f, -1.0f, 0.0f);
        normalize(n);
        rc.normalPen = F4{ n.x, n.y, n.z, rng.uni(-0.06f, 0.04f) };
        rc.point_rest = F4{ rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), C.cloth[ci].restOffset };
        // A vertex ref's barycentric slot holds garbage the kernels must not read.
        rc.bary = (refType(rc.pair.ref1) == REF_eVERTEX) ? F4{ 1e30f, -1e30f, 7.0f, 3.0f } : randomBary(rng, c % 5u == 2u);
        rc.rigidMat = uint16_t(rng.next() % 3u);
        cs.push_back(rc);
    }
    return cs;
}

// PxsMaterialData (PxsMaterialCore.h), 24 B: dynamicFriction @0, fricCombineMode PxU8 @18.
struct RigidMat { float friction; uint8_t combine; };
static const RigidMat kRigidMats[3] = { { 0.5f, 0 }, { 0.2f, 3 }, { 0.9f, 1 } };

static std::vector<uint8_t> rigidMaterialTable()
{
    std::vector<uint8_t> t(3 * 24, 0);
    for (uint32_t m = 0; m < 3; ++m)
    {
        putAt(t, m * 24 + 0, kRigidMats[m].friction);
        putAt(t, m * 24 + 4, 0.33f);   // staticFriction (unread)
        t[m * 24 + 18] = kRigidMats[m].combine;
        t[m * 24 + 16] = 0xA5;         // flags (unread)
    }
    return t;
}

struct RcPrepPush
{
    uint64_t femClothes, contacts_restW, normalPens, barycentrics, contactInfos, numContacts, contactBlocks, preDesc,
        prepareDesc, rigidLambdaNs;
    float invDt; uint32_t pad84;
    uint64_t sharedDesc;
    uint32_t isTGS, pad100;
};
static_assert(sizeof(RcPrepPush) == 104, "cloth_rigidContactPrepareLaunch blob");

PXG_TEST(femClothSolve512RigidContactPrepare)
{
    PxgSlangComputeDevice* dev = testDevice();
    World W;
    buildWorld(W, dev, 0x5150u);
    ClothWorld C;
    buildClothWorld(C, dev, 0x77u);
    const std::vector<RcContact> cs = makeRcContacts(C, 1u);
    const uint32_t NC = uint32_t(cs.size());

    std::vector<F4> pr, np, bc;
    std::vector<PairInfo> pairs;
    for (const RcContact& c : cs) { pr.push_back(c.point_rest); np.push_back(c.normalPen); bc.push_back(c.bary); pairs.push_back(c.pair); }
    std::vector<uint32_t> num = { NC };
    const uint32_t NB = (NC + 31u) / 32u + 1u;
    std::vector<uint32_t> blocksInit(NB * RDBC_STRIDE / 4u, POISON);
    std::vector<F2> lambdaInit(NC + 8u, F2{ 9.0f, -9.0f });

    PxgBufferHandle bBlocks = W.make(blocksInit), bLambda = W.make(lambdaInit);
    RcPrepPush p = {};
    p.femClothes = C.pool;
    p.contacts_restW = W.addr(W.make(pr));
    p.normalPens = W.addr(W.make(np));
    p.barycentrics = W.addr(W.make(bc));
    p.contactInfos = W.addr(W.make(pairs));
    p.numContacts = W.addr(W.make(num));
    p.contactBlocks = W.addr(bBlocks);
    p.preDesc = W.prePrepDesc;
    p.prepareDesc = W.prepDesc;
    p.rigidLambdaNs = W.addr(bLambda);
    p.invDt = 1.0f / kDt;
    p.sharedDesc = W.sharedDesc;
    p.isTGS = 1u;

    Kernel k("femClothRigidContactBda.spv", "cloth_rigidContactPrepareLaunch", {}, sizeof(RcPrepPush));
    k.dispatch(GRID, {}, p);
    dev->flush();

    const std::vector<uint8_t> gotBlocks = downloadBuffer<uint8_t>(dev, bBlocks, NB * RDBC_STRIDE);
    const std::vector<F2> gotLambda = downloadBuffer<F2>(dev, bLambda, NC + 8u);

    uint32_t bad = 0, kinds[4] = { 0, 0, 0, 0 }, invalid = 0, vertexRefs = 0;
    for (uint32_t w = 0; w < NC; ++w)
    {
        const RcContact& c = cs[w];
        cmpF4("rigidLambdaNs", w, F4{ gotLambda[w].x, gotLambda[w].y, 0, 0 }, F4{ 0, 0, 0, 0 }, bad);
        if (!pairValid(c.pair.ref1))
        {
            ++invalid;
            const RdbLane g = readRdbLane(gotBlocks, w);
            if (std::memcmp(&g.normal_errorW, "\xC5\xC5\xC5\xC5", 4) != 0) { ++bad; std::printf("    invalid lane %u written\n", w); }
            continue;
        }
        const HostCloth& h = C.cloth[refId(c.pair.ref1)];
        vertexRefs += refType(c.pair.ref1) == REF_eVERTEX;
        const F4 d = clothContactDelta(h, c.pair.ref1, c.bary);
        const V3 normal = v3(-c.normalPen.x, -c.normalPen.y, -c.normalPen.z);
        const RdbLane want = prepareRigidDbContactRef(W, normal, xyz(c.point_rest), c.normalPen.w - c.point_rest.w,
                                                      xyz(d), c.pair.ref0, h.penBiasClamp);
        const RdbLane got = readRdbLane(gotBlocks, w);
        cmpF4("raXn_resp", w, got.raXn_resp, want.raXn_resp, bad);
        cmpF4("raXnF0_resp", w, got.raXnF0_resp, want.raXnF0_resp, bad);
        cmpF4("raXnF1_resp", w, got.raXnF1_resp, want.raXnF1_resp, bad);
        cmpF4("normal_errorW", w, got.normal_errorW, want.normal_errorW, bad);
        cmpF4("fricTan0_invMass0", w, got.fricTan0_invMass0, want.fricTan0_invMass0, bad);
        cmpF("maxPenBiasClamp", w, got.maxPenBiasClamp, want.maxPenBiasClamp, bad);
        const uint64_t r = c.pair.ref0;
        ++kinds[nodeIsStatic(r) ? 0 : nodeIsArticulation(r) ? 3 : (W.bodies[W.solverBodyIndices[nodeIndex(r)]].linVel_invMass.w == 0.f ? 1 : 2)];
    }
    for (uint32_t w = NC; w < NC + 8u; ++w)
        cmpF4("rigidLambdaNs tail", w, F4{ gotLambda[w].x, gotLambda[w].y, 0, 0 }, F4{ 9.0f, -9.0f, 0, 0 }, bad);
    PXG_CHECK_MSG(bad == 0, "cloth_rigidContactPrepareLaunch: %u mismatches", bad);
    PXG_CHECK_MSG(kinds[0] && kinds[1] && kinds[2] && kinds[3] && invalid && vertexRefs,
                  "coverage: static %u kinematic %u dynamic %u link %u invalid %u vertex %u", kinds[0], kinds[1],
                  kinds[2], kinds[3], invalid, vertexRefs);
}

struct RcQueryPush
{
    uint64_t femClothes, contactInfos, barycentrics, contactBlocks, numContacts, prePrepDesc, solverCoreDesc,
        artiCoreDesc, solverBodyVelPool;
    float dt; uint32_t pad76;
    uint64_t appliedForces, rigidBodyRefCounts;
    uint32_t flags, pad100;   // byte0 isTGS, byte1 isVelocityIteration
};
static_assert(sizeof(RcQueryPush) == 104, "cloth_queryRigidClothContactReferenceCountLaunch blob");

struct RcSolvePush
{
    uint64_t femClothes, contactInfos, contactRigidMaterials, barycentrics, contactBlocks, numContacts, prePrepDesc,
        solverCoreDesc, artiCoreDesc, solverBodyVelPool, rigidDeltaVel, appliedForces, rigidBodyRefCounts;
    float dt; uint32_t pad108;
    uint64_t materials, rigidBodyMaterials;
    uint32_t flags, pad132;   // byte0 isTGS, byte1 isVelocityIteration
};
static_assert(sizeof(RcSolvePush) == 136, "cloth_solveRigidClothCollisionLaunch blob");

static BlockLane toBlockLane(const RdbLane& l)
{
    BlockLane bl;
    bl.raXn_resp = l.raXn_resp; bl.raXnF0_resp = l.raXnF0_resp; bl.raXnF1_resp = l.raXnF1_resp;
    bl.normal_errorW = l.normal_errorW; bl.fricTan0_invMass0 = l.fricTan0_invMass0;
    return bl;
}

static ContactPair toContactPair(const RdbLane& l)
{
    ContactPair cp;
    cp.normal = xyz(l.normal_errorW); cp.initPen = l.normal_errorW.w;
    cp.tangent0 = xyz(l.fricTan0_invMass0); cp.maxPenBiasClamp = l.maxPenBiasClamp;
    return cp;
}

// Query then solve, each checked against the oracle, in one of the four modes.
static void runRcQuerySolve(bool isTGS, bool isVelocityIteration, uint32_t seed)
{
    PxgSlangComputeDevice* dev = testDevice();
    World W;
    buildWorld(W, dev, seed);
    ClothWorld C;
    buildClothWorld(C, dev, seed + 1u);
    std::vector<RcContact> cs = makeRcContacts(C, seed);
    const uint32_t NC = uint32_t(cs.size());
    const VelocityReader reader = W.reader();

    // Some pairs arrive ACTIVE already (latched by an earlier iteration).
    for (uint32_t w = 0; w < NC; ++w)
        if (w % 4u == 3u && pairValid(cs[w].pair.ref1))
            cs[w].pair.ref1 = setActive(cs[w].pair.ref1, true);

    std::vector<RdbLane> lanes(NC);
    std::vector<uint8_t> blocks(((NC + 31u) / 32u) * RDBC_STRIDE, 0);
    std::vector<F4> bc;
    std::vector<PairInfo> pairs;
    std::vector<uint16_t> rmat;
    for (uint32_t w = 0; w < NC; ++w)
    {
        const RcContact& c = cs[w];
        bc.push_back(c.bary);
        pairs.push_back(c.pair);
        rmat.push_back(c.rigidMat);
        if (!pairValid(c.pair.ref1)) continue;
        const HostCloth& h = C.cloth[refId(c.pair.ref1)];
        const F4 d = clothContactDelta(h, c.pair.ref1, c.bary);
        lanes[w] = prepareRigidDbContactRef(W, v3(-c.normalPen.x, -c.normalPen.y, -c.normalPen.z), xyz(c.point_rest),
                                            c.normalPen.w - c.point_rest.w, xyz(d), c.pair.ref0, h.penBiasClamp);
        writeRdbLane(blocks, w, lanes[w]);
    }
    Rng rng{ seed ^ 0x1234u };
    std::vector<F2> applied(NC);
    for (uint32_t w = 0; w < NC; ++w)
        applied[w] = (w % 6u == 0u) ? F2{ 0.f, 0.f } : F2{ rng.uni(0.0f, 0.05f), rng.uni(-0.03f, 0.03f) };
    std::vector<uint32_t> num = { NC };

    // ---------------- query ----------------
    std::vector<uint32_t> refCounts(W.refCountSlots() + 4u, 0u);
    PxgBufferHandle bPairs = W.make(pairs), bRef = W.make(refCounts), bApplied = W.make(applied);
    const uint64_t aBc = W.addr(W.make(bc)), aBlocks = W.addr(W.make(blocks)), aNum = W.addr(W.make(num));
    {
        RcQueryPush q = {};
        q.femClothes = C.pool; q.contactInfos = W.addr(bPairs); q.barycentrics = aBc; q.contactBlocks = aBlocks;
        q.numContacts = aNum; q.prePrepDesc = W.prePrepDesc; q.solverCoreDesc = W.coreDesc; q.artiCoreDesc = W.artiCoreDesc;
        q.solverBodyVelPool = W.velPoolAddr; q.dt = kDt; q.appliedForces = W.addr(bApplied);
        q.rigidBodyRefCounts = W.addr(bRef);
        q.flags = (isTGS ? 1u : 0u) | (isVelocityIteration ? 0x100u : 0u);
        Kernel k("femClothRigidContactBda.spv", "cloth_queryRigidClothContactReferenceCountLaunch", {}, sizeof(RcQueryPush));
        k.dispatch(GRID, {}, q);
        dev->flush();
    }

    std::vector<uint32_t> wantRef = refCounts;
    std::vector<PairInfo> wantPairs = pairs;
    uint32_t activated = 0, latched = 0, stayedInactive = 0;
    for (uint32_t w = 0; w < NC; ++w)
    {
        const PairInfo& pi = pairs[w];
        if (!pairValid(pi.ref1)) continue;
        const bool wasActive = pairActive(pi.ref1);
        HostCloth& h = C.cloth[refId(pi.ref1)];
        const int g = W.gid(pi.ref0);
        const float rigidInvMass = lanes[w].fricTan0_invMass0.w;
        DbPart db;
        db.readCloth(h, pi.ref1, cs[w].bary, nullptr, true, kDt, isVelocityIteration);
        bool isActive = false;
        if (wasActive) { isActive = true; ++latched; }
        else
        {
            RigidPart rigid;
            rigid.readBodyProperties(g, rigidInvMass, nullptr);
            rigid.readContactPrep(toBlockLane(lanes[w]));
            rigid.readVelocity(reader, pi.ref0, isTGS);
            const ContactPair cp = toContactPair(lanes[w]);
            SolveOut so;
            isActive = solveRigidDbContactDb(rigid, db, cp, so, applied[w].x, applied[w].y, kDt, false, true,
                                             isVelocityIteration);
            if (isActive) { wantPairs[w].ref1 = setActive(pi.ref1, true); ++activated; }
            else ++stayedInactive;
        }
        if (isActive)
        {
            bumpCloth(h, pi.ref1, db.bc);
            if (rigidIsRefCounted(g, rigidInvMass)) wantRef[g] += 1u;
        }
    }

    uint32_t bad = 0;
    {
        const std::vector<PairInfo> gotPairs = downloadBuffer<PairInfo>(dev, bPairs, NC);
        for (uint32_t w = 0; w < NC; ++w)
            if (gotPairs[w].ref0 != wantPairs[w].ref0 || gotPairs[w].ref1 != wantPairs[w].ref1)
            { if (++bad <= 8) std::printf("    pair %u got %llx want %llx\n", w, (unsigned long long)gotPairs[w].ref1, (unsigned long long)wantPairs[w].ref1); }
        const std::vector<uint32_t> gotRef = downloadBuffer<uint32_t>(dev, bRef, wantRef.size());
        for (size_t i = 0; i < wantRef.size(); ++i)
            if (gotRef[i] != wantRef[i]) { if (++bad <= 8) std::printf("    refCount[%zu] got %u want %u\n", i, gotRef[i], wantRef[i]); }
        for (uint32_t ci = 0; ci < 3; ci += 2)
            cmpVec("query mDeltaPos", C.readDelta(ci), C.cloth[ci].delta, bad);
    }
    PXG_CHECK_MSG(bad == 0, "cloth_queryRigidClothContactReferenceCountLaunch (%s %s): %u mismatches",
                  isTGS ? "TGS" : "PGS", isVelocityIteration ? "vel" : "pos", bad);
    PXG_CHECK_MSG(activated > 0 && latched > 0, "coverage: activated %u latched %u inactive %u", activated, latched,
                  stayedInactive);

    // ---------------- solve (on the query's output) ----------------
    std::vector<F4> deltaVelInit(2 * NC, F4{ f32(POISON), f32(POISON), f32(POISON), f32(POISON) });
    std::vector<uint32_t> rmatWords((rmat.size() + 1) / 2 + 2, 0u);
    std::memcpy(rmatWords.data(), rmat.data(), rmat.size() * 2);
    PxgBufferHandle bDeltaVel = W.make(deltaVelInit);
    {
        RcSolvePush sp = {};
        sp.femClothes = C.pool; sp.contactInfos = W.addr(bPairs); sp.contactRigidMaterials = W.addr(W.make(rmatWords));
        sp.barycentrics = aBc; sp.contactBlocks = aBlocks; sp.numContacts = aNum; sp.prePrepDesc = W.prePrepDesc;
        sp.solverCoreDesc = W.coreDesc; sp.artiCoreDesc = W.artiCoreDesc; sp.solverBodyVelPool = W.velPoolAddr;
        sp.rigidDeltaVel = W.addr(bDeltaVel); sp.appliedForces = W.addr(bApplied); sp.rigidBodyRefCounts = W.addr(bRef);
        sp.dt = kDt; sp.materials = C.surfMaterials; sp.rigidBodyMaterials = W.addr(W.make(rigidMaterialTable()));
        sp.flags = (isTGS ? 1u : 0u) | (isVelocityIteration ? 0x100u : 0u);
        Kernel k("femClothRigidContactBda.spv", "cloth_solveRigidClothCollisionLaunch", {}, sizeof(RcSolvePush));
        k.dispatch(GRID, {}, sp);
        dev->flush();
    }

    // The solve reads the post-query .w and scatters .xyz only, so one snapshot serves all.
    const HostCloth snap[3] = { C.cloth[0], C.cloth[1], C.cloth[2] };
    std::vector<F4> wantDeltaVel = deltaVelInit;
    std::vector<F2> wantApplied = applied;
    uint32_t appliedCount = 0;
    for (uint32_t w = 0; w < NC; ++w)
    {
        const PairInfo& pi = wantPairs[w];
        if (!pairValid(pi.ref1)) continue;
        const bool isActive = pairActive(pi.ref1);
        const int g = W.gid(pi.ref0);
        const float rigidInvMass = lanes[w].fricTan0_invMass0.w;
        RigidPart rigid;
        ContactPair cp;
        SolveOut so;
        if (isActive)
        {
            rigid.readBodyProperties(g, rigidInvMass, wantRef.data());
            rigid.friction = kRigidMats[cs[w].rigidMat].friction;
            rigid.fricCombineMode = kRigidMats[cs[w].rigidMat].combine;
            DbPart db;
            db.readCloth(snap[refId(pi.ref1)], pi.ref1, cs[w].bary, C.surfaceFriction.data(), false, kDt, isVelocityIteration);
            rigid.readContactPrep(toBlockLane(lanes[w]));
            rigid.readVelocity(reader, pi.ref0, isTGS);
            cp = toContactPair(lanes[w]);
            const bool hasApplied = solveRigidDbContactDb(rigid, db, cp, so, applied[w].x, applied[w].y, kDt, true, false,
                                                          isVelocityIteration);
            if (hasApplied)
            {
                ++appliedCount;
                V3 delta;
                so.computeDelta(cp, delta, kDt);
                wantApplied[w] = F2{ so.accumulatedDeltaLambdaN, applied[w].y + so.deltaLambdaT };
                db.writeCloth(C.cloth[refId(pi.ref1)], delta);
            }
        }
        std::vector<F4> two(2);
        rigid.writeContactDeltas(two.data(), pi.ref0, cp, so, 0, 1);
        wantDeltaVel[w] = two[0];
        wantDeltaVel[w + NC] = two[1];
    }

    bad = 0;
    cmpVec("rigidDeltaVel", downloadBuffer<F4>(dev, bDeltaVel, 2 * NC), wantDeltaVel, bad);
    {
        const std::vector<F2> gotApplied = downloadBuffer<F2>(dev, bApplied, NC);
        for (uint32_t w = 0; w < NC; ++w)
            cmpF4("appliedForces", w, F4{ gotApplied[w].x, gotApplied[w].y, 0, 0 },
                  F4{ wantApplied[w].x, wantApplied[w].y, 0, 0 }, bad);
    }
    for (uint32_t ci = 0; ci < 3; ci += 2)
        cmpVec("solve mDeltaPos", C.readDelta(ci), C.cloth[ci].delta, bad);
    PXG_CHECK_MSG(bad == 0, "cloth_solveRigidClothCollisionLaunch (%s %s): %u mismatches", isTGS ? "TGS" : "PGS",
                  isVelocityIteration ? "vel" : "pos", bad);
    PXG_CHECK_MSG(appliedCount > 0, "coverage: no contact applied an impulse");
}

PXG_TEST(femClothSolve512RigidContactPGS) { runRcQuerySolve(false, false, 11u); }
PXG_TEST(femClothSolve512RigidContactTGS) { runRcQuerySolve(true, false, 12u); }
PXG_TEST(femClothSolve512RigidContactPGSVel) { runRcQuerySolve(false, true, 13u); }
PXG_TEST(femClothSolve512RigidContactTGSVel) { runRcQuerySolve(true, true, 14u); }

// ===========================================================================
// Cloth-rigid attachments
// ===========================================================================
struct DefAttachment { PairInfo pair; F4 bOrP0, bary1; };

// prepareRigidDbAttachment (deformableUtils.cuh), on the World's rigid data (the same
// transcription as testSoftBodySolve512's).
static RdbAttachLane prepareRigidDbAttachmentRef(const World& W, V3 point, float defInvMass, V3 localPose,
                                                 uint64_t rigidId, uint64_t elemRef, F4 bary, uint32_t refCount,
                                                 F4 slotBaryPrior, bool& zeroDeltaVel)
{
    RdbAttachLane b;
    const V3 n0 = v3(1, 0, 0), n1 = v3(0, 1, 0), n2 = v3(0, 0, 1);
    zeroDeltaVel = false;
    if (nodeIsArticulation(rigidId))
    {
        const World::Link& L = W.links[nodeLinkId(rigidId)];
        const V3 ra = quatRotate(L.q, localPose);
        const V3 error = sub(add(ra, L.p), point);
        const V3 r0 = cross(ra, n0), r1 = cross(ra, n1), r2 = cross(ra, n2);
        V3 t0, b0, t1, b1, t2, b2;
        spatialMul(L.response, n0, r0, t0, b0);
        spatialMul(L.response, n1, r1, t1, b1);
        spatialMul(L.response, n2, r2, t2, b2);
        const float resp0 = dot(t0, r0) + dot(b0, n0) + defInvMass;
        const float resp1 = dot(t1, r1) + dot(b1, n1) + defInvMass;
        const float resp2 = dot(t2, r2) + dot(b2, n2) + defInvMass;
        b.raXn0_biasW = F4{ r0.x, r0.y, r0.z, dot(error, n0) };
        b.raXn1_biasW = F4{ r1.x, r1.y, r1.z, dot(error, n1) };
        b.raXn2_biasW = F4{ r2.x, r2.y, r2.z, dot(error, n2) };
        b.velMultiplierXYZ_invMassW = F4{ resp0 > 0.f ? 1.f / resp0 : 0.f, resp1 > 0.f ? 1.f / resp1 : 0.f,
                                          resp2 > 0.f ? 1.f / resp2 : 0.f, 1.0f };
    }
    else
    {
        const uint32_t idx = nodeIsStatic(rigidId) ? 0u : W.solverBodyIndices[nodeIndex(rigidId)];
        const World::Body& B = W.bodies[idx];
        const float invMass0 = B.linVel_invMass.w;
        float inv[9];
        float inertiaScale = 1.0f;
        if (invMass0 == 0.0f && !nodeIsStatic(rigidId))
        {
            const float id[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
            std::memcpy(inv, id, sizeof(inv));
            inertiaScale = 0.0f;
        }
        else
            std::memcpy(inv, B.sqrtInvInertia, sizeof(inv));
        const V3 ra = quatRotate(B.q, localPose);
        const V3 error = sub(add(ra, xyz(B.p)), point);
        const V3 r0 = mat33Mul(inv, cross(ra, n0)), r1 = mat33Mul(inv, cross(ra, n1)), r2 = mat33Mul(inv, cross(ra, n2));
        const float resp0 = (dot(r0, r0)) * inertiaScale + invMass0 + defInvMass;
        const float resp1 = (dot(r1, r1)) * inertiaScale + invMass0 + defInvMass;
        const float resp2 = (dot(r2, r2)) * inertiaScale + invMass0 + defInvMass;
        b.raXn0_biasW = F4{ r0.x, r0.y, r0.z, dot(error, n0) };
        b.raXn1_biasW = F4{ r1.x, r1.y, r1.z, dot(error, n1) };
        b.raXn2_biasW = F4{ r2.x, r2.y, r2.z, dot(error, n2) };
        b.velMultiplierXYZ_invMassW = F4{ resp0 > 0.f ? 1.f / resp0 : 0.f, resp1 > 0.f ? 1.f / resp1 : 0.f,
                                          resp2 > 0.f ? 1.f / resp2 : 0.f, invMass0 };
        zeroDeltaVel = true;
    }
    b.deformableRef = elemRef;
    b.rigidId = rigidId;
    b.barycentric = (refType(elemRef) == REF_eVERTEX) ? slotBaryPrior : bary;
    b.rigidBodyRefCount = refCount;
    return b;
}

static void putRdba(std::vector<uint8_t>& blk, uint32_t w, const RdbAttachLane& b)
{
    const size_t base = size_t(w / 32u) * RDBA_STRIDE;
    const uint32_t l = w & 31u;
    putAt(blk, base + RDBA_barycentric + l * 16, b.barycentric);
    putAt(blk, base + RDBA_raXn0_biasW + l * 16, b.raXn0_biasW);
    putAt(blk, base + RDBA_raXn1_biasW + l * 16, b.raXn1_biasW);
    putAt(blk, base + RDBA_raXn2_biasW + l * 16, b.raXn2_biasW);
    putAt(blk, base + RDBA_velMultiplierXYZ_invMassW + l * 16, b.velMultiplierXYZ_invMassW);
    putAt(blk, base + RDBA_deformableRef + l * 8, b.deformableRef);
    putAt(blk, base + RDBA_rigidId + l * 8, b.rigidId);
    putAt(blk, base + RDBA_rigidBodyRefCount + l * 4, b.rigidBodyRefCount);
}

static RdbAttachLane getRdba(const std::vector<uint8_t>& blk, uint32_t w)
{
    const size_t base = size_t(w / 32u) * RDBA_STRIDE;
    const uint32_t l = w & 31u;
    RdbAttachLane b;
    b.barycentric = getAt<F4>(blk, base + RDBA_barycentric + l * 16);
    b.raXn0_biasW = getAt<F4>(blk, base + RDBA_raXn0_biasW + l * 16);
    b.raXn1_biasW = getAt<F4>(blk, base + RDBA_raXn1_biasW + l * 16);
    b.raXn2_biasW = getAt<F4>(blk, base + RDBA_raXn2_biasW + l * 16);
    b.velMultiplierXYZ_invMassW = getAt<F4>(blk, base + RDBA_velMultiplierXYZ_invMassW + l * 16);
    b.deformableRef = getAt<uint64_t>(blk, base + RDBA_deformableRef + l * 8);
    b.rigidId = getAt<uint64_t>(blk, base + RDBA_rigidId + l * 8);
    b.rigidBodyRefCount = getAt<uint32_t>(blk, base + RDBA_rigidBodyRefCount + l * 4);
    return b;
}

// NA attachments: triangles and lone vertices of cloths 0 / 2, at most two scatters a vertex.
static std::vector<DefAttachment> makeRigidAttachments(uint32_t NA, uint32_t seed)
{
    Rng rng{ seed * 3u + 11u };
    const uint64_t rigids[6] = { staticNode(), makeRigidNode(4), makeRigidNode(6), makeRigidNode(8),
                                 makeArticulationNode(10, 2), makeArticulationNode(10, 1) };
    ScatterBudget budget;
    std::vector<DefAttachment> as;
    for (uint32_t tries = 0; as.size() < NA && tries < 10000u; ++tries)
    {
        DefAttachment d;
        const uint32_t a = uint32_t(as.size());
        const uint32_t ci = (a % 2u) ? 2u : 0u;
        d.pair.ref1 = randomClothRef(rng, ci, true);
        if (!budget.fits(0, false, d.pair.ref1)) continue;
        d.pair.ref0 = rigids[(a + rng.next()) % 6u];
        d.bOrP0 = F4{ rng.uni(-0.5f, 0.5f), rng.uni(-0.5f, 0.5f), rng.uni(-0.5f, 0.5f), 0.0f };
        d.bary1 = randomBary(rng, a % 5u == 2u);
        as.push_back(d);
    }
    return as;
}

static RdbAttachLane rigidAttachLaneRef(const World& W, const ClothWorld& C, const DefAttachment& d, uint32_t refCount,
                                        F4 slotPrior, bool& zeroDeltaVel)
{
    const HostCloth& h = C.cloth[refId(d.pair.ref1)];
    const uint32_t e = refFeature(d.pair.ref1);
    F4 pose;
    float invMass1;
    if (refType(d.pair.ref1) == REF_eVERTEX)
    {
        pose = h.pos[e];
        invMass1 = pose.w;
    }
    else
    {
        const U4 t = h.tris[e];
        const F4 b = d.bary1;
        const F4 p0 = h.pos[t.x], p1 = h.pos[t.y], p2 = h.pos[t.z];
        pose = add4(add4(mul4(p0, b.x), mul4(p1, b.y)), mul4(p2, b.z));
        invMass1 = b.x * b.x * p0.w + b.y * b.y * p1.w + b.z * b.z * p2.w;
    }
    return prepareRigidDbAttachmentRef(W, xyz(pose), invMass1, xyz(d.bOrP0), d.pair.ref0, d.pair.ref1, d.bary1,
                                       refCount, slotPrior, zeroDeltaVel);
}

struct RaPrepPush
{
    uint64_t clothes, rigidAttachments, activeRigidAttachments, rigidAttachmentRigidRefCounts, rigidAttachmentIds;
    uint32_t numRigidAttachments, pad44;
    uint64_t attachmentBlocks, preDesc, prepareDesc, sharedDesc, rigidDeltaVel;
};
static_assert(sizeof(RaPrepPush) == 88, "cloth_rigidAttachmentPrepareLaunch blob");

PXG_TEST(femClothSolve512RigidAttachPrepare)
{
    PxgSlangComputeDevice* dev = testDevice();
    World W;
    buildWorld(W, dev, 0x777u, /*rotatedFrames*/ true);
    ClothWorld C;
    buildClothWorld(C, dev, 0x88u);
    const uint32_t NREC = 50u;
    const std::vector<DefAttachment> recs = makeRigidAttachments(NREC, 5u);
    PXG_CHECK_MSG(recs.size() == NREC, "attachment generator fell short (%zu)", recs.size());
    // the active subset, in a permuted order
    std::vector<uint32_t> active;
    for (uint32_t i = 0; i < NREC; ++i)
        if (i % 6u != 4u) active.push_back((i * 31u) % NREC);
    const uint32_t NA = uint32_t(active.size());
    std::vector<uint8_t> recBytes(NREC * DATT_STRIDE, 0);
    for (uint32_t i = 0; i < NREC; ++i)
    {
        putAt(recBytes, i * DATT_STRIDE + 0, recs[i].pair);
        putAt(recBytes, i * DATT_STRIDE + 16, recs[i].bOrP0);
        putAt(recBytes, i * DATT_STRIDE + 32, recs[i].bary1);
    }
    std::vector<uint32_t> refCounts(NA);
    for (uint32_t w = 0; w < NA; ++w) refCounts[w] = 1u + (w % 3u);
    const uint32_t NB = (NA + 31u) / 32u;
    std::vector<uint32_t> blocksInit(NB * RDBA_STRIDE / 4u, POISON);
    std::vector<uint64_t> idsInit(NA + 4u, 0xABABABABABABABABull);
    std::vector<F4> dvInit(2 * NA, F4{ 3.f, 3.f, 3.f, 3.f });
    PxgBufferHandle bBlocks = W.make(blocksInit), bIds = W.make(idsInit), bDv = W.make(dvInit);

    RaPrepPush p = {};
    p.clothes = C.pool; p.rigidAttachments = W.addr(W.make(recBytes)); p.activeRigidAttachments = W.addr(W.make(active));
    p.rigidAttachmentRigidRefCounts = W.addr(W.make(refCounts)); p.rigidAttachmentIds = W.addr(bIds);
    p.numRigidAttachments = NA; p.attachmentBlocks = W.addr(bBlocks); p.preDesc = W.prePrepDesc; p.prepareDesc = W.prepDesc;
    p.sharedDesc = W.sharedDesc; p.rigidDeltaVel = W.addr(bDv);
    Kernel k("femClothRigidAttachBda.spv", "cloth_rigidAttachmentPrepareLaunch", {}, sizeof(RaPrepPush));
    k.dispatch(GRID, {}, p);
    dev->flush();

    const std::vector<uint8_t> got = downloadBuffer<uint8_t>(dev, bBlocks, NB * RDBA_STRIDE);
    const std::vector<uint64_t> gotIds = downloadBuffer<uint64_t>(dev, bIds, NA + 4u);
    const std::vector<F4> gotDv = downloadBuffer<F4>(dev, bDv, 2 * NA);
    std::vector<F4> wantDv = dvInit;
    uint32_t bad = 0, nVertex = 0, nArti = 0;
    const F4 poisonF4 = F4{ f32(POISON), f32(POISON), f32(POISON), f32(POISON) };
    for (uint32_t w = 0; w < NA; ++w)
    {
        const DefAttachment& d = recs[active[w]];
        bool zeroDv = false;
        const RdbAttachLane want = rigidAttachLaneRef(W, C, d, refCounts[w], poisonF4, zeroDv);
        if (zeroDv) { wantDv[w] = F4{ 0, 0, 0, 0 }; wantDv[w + NA] = F4{ 0, 0, 0, 0 }; }
        if (refType(d.pair.ref1) == REF_eVERTEX) ++nVertex;
        if (nodeIsArticulation(d.pair.ref0)) ++nArti;
        const RdbAttachLane g = getRdba(got, w);
        cmpF4("barycentric", w, g.barycentric, want.barycentric, bad);
        cmpF4("raXn0_biasW", w, g.raXn0_biasW, want.raXn0_biasW, bad);
        cmpF4("raXn1_biasW", w, g.raXn1_biasW, want.raXn1_biasW, bad);
        cmpF4("raXn2_biasW", w, g.raXn2_biasW, want.raXn2_biasW, bad);
        cmpF4("velMultiplierXYZ_invMassW", w, g.velMultiplierXYZ_invMassW, want.velMultiplierXYZ_invMassW, bad);
        if (g.deformableRef != want.deformableRef || g.rigidId != want.rigidId || g.rigidBodyRefCount != want.rigidBodyRefCount)
        { if (++bad <= 8) std::printf("    lane %u ids/refCount mismatch\n", w); }
        if (gotIds[w] != d.pair.ref0) { if (++bad <= 8) std::printf("    rigidAttachmentIds[%u]\n", w); }
    }
    for (uint32_t w = NA; w < NA + 4u; ++w)
        if (gotIds[w] != 0xABABABABABABABABull) ++bad;
    cmpVec("rigidDeltaVel", gotDv, wantDv, bad);
    PXG_CHECK_MSG(bad == 0, "cloth_rigidAttachmentPrepareLaunch: %u mismatches", bad);
    PXG_CHECK_MSG(nVertex > 0 && nArti > 0, "coverage: vertex %u articulation %u", nVertex, nArti);
}

struct RaQueryPush { uint64_t clothes, attachmentBlocks; uint32_t numAttachments, pad20; };
static_assert(sizeof(RaQueryPush) == 24, "cloth_queryRigidClothAttachmentReferenceCountLaunch blob");

struct RaSolvePush
{
    uint64_t clothes, attachmentBlocks;
    uint32_t numAttachments, pad20;
    uint64_t prePrepDesc, solverCoreDesc, artiCoreDesc, solverBodyVelPool;
    float dt, biasCoefficient;
    uint64_t rigidDeltaVel;
    uint32_t flags, pad76;   // byte0 isVelocityIteration, byte1 isTGS
};
static_assert(sizeof(RaSolvePush) == 80, "cloth_solveRigidClothAttachmentLaunch blob");

static void runRigidAttachQuerySolve(bool isTGS, bool isVelocityIteration, uint32_t seed)
{
    PxgSlangComputeDevice* dev = testDevice();
    World W;
    buildWorld(W, dev, seed, /*rotatedFrames*/ true);
    ClothWorld C;
    buildClothWorld(C, dev, seed + 3u);
    const uint32_t NA = 45u;
    const std::vector<DefAttachment> as = makeRigidAttachments(NA, seed);
    PXG_CHECK_MSG(as.size() == NA, "attachment generator fell short (%zu)", as.size());
    const VelocityReader reader = W.reader();

    std::vector<uint8_t> blocks(((NA + 31u) / 32u) * RDBA_STRIDE, 0);
    std::vector<RdbAttachLane> lanes(NA);
    for (uint32_t w = 0; w < NA; ++w)
    {
        bool zdv;
        lanes[w] = rigidAttachLaneRef(W, C, as[w], 1u + (w % 3u), F4{ 1e30f, -1e30f, 1e30f, -1e30f }, zdv);
        putRdba(blocks, w, lanes[w]);
    }
    const uint64_t aBlocks = W.addr(W.make(blocks));

    // ---- query ----
    {
        RaQueryPush q = { C.pool, aBlocks, NA, 0u };
        Kernel k("femClothRigidAttachBda.spv", "cloth_queryRigidClothAttachmentReferenceCountLaunch", {}, sizeof(RaQueryPush));
        k.dispatch(GRID, {}, q);
        dev->flush();
    }
    for (uint32_t w = 0; w < NA; ++w)
        bumpCloth(C.cloth[refId(lanes[w].deformableRef)], lanes[w].deformableRef, lanes[w].barycentric);
    uint32_t bad = 0;
    for (uint32_t ci = 0; ci < 3; ci += 2)
        cmpVec("query mDeltaPos", C.readDelta(ci), C.cloth[ci].delta, bad);
    PXG_CHECK_MSG(bad == 0, "cloth_queryRigidClothAttachmentReferenceCountLaunch: %u mismatches", bad);

    // ---- solve ----
    const float bias = isVelocityIteration ? 0.0f : 0.7f;
    std::vector<F4> dvInit(2 * NA, F4{ 5.f, 5.f, 5.f, 5.f });
    PxgBufferHandle bDv = W.make(dvInit);
    {
        RaSolvePush sp = {};
        sp.clothes = C.pool; sp.attachmentBlocks = aBlocks; sp.numAttachments = NA; sp.prePrepDesc = W.prePrepDesc;
        sp.solverCoreDesc = W.coreDesc; sp.artiCoreDesc = W.artiCoreDesc; sp.solverBodyVelPool = W.velPoolAddr;
        sp.dt = kDt; sp.biasCoefficient = bias; sp.rigidDeltaVel = W.addr(bDv);
        sp.flags = (isVelocityIteration ? 1u : 0u) | (isTGS ? 0x100u : 0u);
        Kernel k("femClothRigidAttachBda.spv", "cloth_solveRigidClothAttachmentLaunch", {}, sizeof(RaSolvePush));
        k.dispatch(GRID, {}, sp);
        dev->flush();
    }
    const HostCloth snap[3] = { C.cloth[0], C.cloth[1], C.cloth[2] };
    std::vector<F4> wantDv = dvInit;
    uint32_t nSkipped = 0, nWritten = 0;
    for (uint32_t w = 0; w < NA; ++w)
    {
        const RdbAttachLane& L = lanes[w];
        DbPart db;
        db.readClothAttachment(snap[refId(L.deformableRef)], L.deformableRef, L.barycentric, isTGS, kDt, isVelocityIteration);
        if (nodeIsStatic(L.rigidId) && db.attachPointInvMass == 0.0f) { ++nSkipped; continue; }
        AttachRigid rigid;
        rigid.readAttachmentPrep(L);
        rigid.readVelocity(reader, L.rigidId, isTGS);
        V3 dLin, dAng;
        const V3 imp = solveRigidDbAttachment(rigid, db, kDt, bias, isTGS, isVelocityIteration, dLin, dAng);
        if (rigid.writeAttachmentDeltas(dLin, dAng, wantDv[w], wantDv[w + NA], L.rigidId)) ++nWritten;
        if (!(imp.x == 0.0f && imp.y == 0.0f && imp.z == 0.0f))
            db.writeCloth(C.cloth[refId(L.deformableRef)], mul(neg(imp), kDt));
    }
    bad = 0;
    cmpVec("rigidDeltaVel", downloadBuffer<F4>(dev, bDv, 2 * NA), wantDv, bad);
    for (uint32_t ci = 0; ci < 3; ci += 2)
        cmpVec("solve mDeltaPos", C.readDelta(ci), C.cloth[ci].delta, bad);
    PXG_CHECK_MSG(bad == 0, "cloth_solveRigidClothAttachmentLaunch (%s %s): %u mismatches", isTGS ? "TGS" : "PGS",
                  isVelocityIteration ? "vel" : "pos", bad);
    PXG_CHECK_MSG(nWritten > 0, "coverage: written %u skipped %u", nWritten, nSkipped);
}

PXG_TEST(femClothSolve512RigidAttachPGS) { runRigidAttachQuerySolve(false, false, 51u); }
PXG_TEST(femClothSolve512RigidAttachTGS) { runRigidAttachQuerySolve(true, false, 52u); }
PXG_TEST(femClothSolve512RigidAttachPGSVel) { runRigidAttachQuerySolve(false, true, 53u); }
PXG_TEST(femClothSolve512RigidAttachTGSVel) { runRigidAttachQuerySolve(true, true, 54u); }

// ===========================================================================
// Cloth-particle contacts (prepare, query and the two solve halves)
// ===========================================================================
struct CpPrepPush
{
    uint64_t clothes, particlesystems, contacts, normalPens, barycentrics, contactInfos, numContacts, contactBlocks,
        clothAppliedForces, particleAppliedForces;
};
static_assert(sizeof(CpPrepPush) == 80, "cloth_particleContactPrepareLaunch blob");

struct CpQueryPush
{
    uint64_t clothes, particlesystems, contactInfos, contactBlocks, numContacts;
    float dt; uint32_t isVelocityIteration;
};
static_assert(sizeof(CpQueryPush) == 48, "cloth_queryCPContactReferenceCountLaunch blob");

struct CpSolveClothPush
{
    uint64_t clothes, particlesystems, contactInfos, contactBlocks, numContacts, appliedForces, materials;
    float dt; uint32_t isVelocityIteration;
};
static_assert(sizeof(CpSolveClothPush) == 64, "cloth_solveCPOutputClothDeltaVLaunch blob");

struct CpSolvePsPush
{
    uint64_t clothes, particlesystems, contactInfos, contactBlocks, numContacts, deltaP, appliedForces, materials;
    float dt; uint32_t isVelocityIteration;
};
static_assert(sizeof(CpSolvePsPush) == 72, "cloth_solveCPOutputParticleDeltaVLaunch blob");

struct CpContact { PairInfo pair; F4 normalPen, bary; };

// 45 particle-vs-triangle contacts, triangles visited at most twice, a few INVALID.
static std::vector<CpContact> makeCpContacts(uint32_t seed)
{
    Rng rng{ seed * 31u + 1u };
    ScatterBudget budget;
    std::vector<CpContact> cs;
    const uint32_t NC = 45u;
    for (uint32_t tries = 0; cs.size() < NC && tries < 10000u; ++tries)
    {
        const uint32_t c = uint32_t(cs.size());
        CpContact cp;
        const uint32_t sys = (c % 3u == 2u) ? 2u : 0u;
        cp.pair.ref0 = particleRef(sys, rng.next() % 10u);
        if (c % 12u == 5u)
            cp.pair.ref1 = encodeRef(0, 0, 0) | REF_INVALID;
        else
        {
            const uint64_t tri = encodeRef(c % 2u ? 0u : 2u, REF_eTRIANGLE, rng.next() % NTRI);
            if (!budget.fits(0, false, tri)) continue;
            cp.pair.ref1 = tri;
        }
        V3 n = v3(rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f));
        normalize(n);
        cp.normalPen = F4{ n.x, n.y, n.z, rng.uni(-0.06f, 0.03f) };
        cp.bary = randomBary(rng, c % 6u == 1u);
        cs.push_back(cp);
    }
    return cs;
}

PXG_TEST(femClothSolve512ParticleContactPrepare)
{
    PxgSlangComputeDevice* dev = testDevice();
    World W;
    buildWorld(W, dev, 0x9191u);
    ClothWorld C;
    buildClothWorld(C, dev, 0x31u);
    const std::vector<CpContact> cs = makeCpContacts(3u);
    const uint32_t NC = uint32_t(cs.size());

    std::vector<F4> np, bc, contacts(NC, F4{ 1e30f, 1e30f, 1e30f, 1e30f });
    std::vector<PairInfo> pairs;
    for (const CpContact& c : cs) { np.push_back(c.normalPen); bc.push_back(c.bary); pairs.push_back(c.pair); }
    std::vector<uint32_t> num = { NC };
    const uint32_t NB = (NC + 31u) / 32u + 1u;
    std::vector<uint32_t> blocksInit(NB * PDBC_STRIDE / 4u, POISON);
    std::vector<F2> appliedInit(NC + 8u, F2{ 9.0f, -9.0f });
    PxgBufferHandle bBlocks = W.make(blocksInit), bAppC = W.make(appliedInit), bAppP = W.make(appliedInit);

    CpPrepPush p = { C.pool, W.psysPool, W.addr(W.make(contacts)), W.addr(W.make(np)), W.addr(W.make(bc)),
                     W.addr(W.make(pairs)), W.addr(W.make(num)), W.addr(bBlocks), W.addr(bAppC), W.addr(bAppP) };
    Kernel k("femClothPsPrepareBda.spv", "cloth_particleContactPrepareLaunch", {}, sizeof(CpPrepPush));
    k.dispatch(GRID, {}, p);
    dev->flush();

    const std::vector<uint8_t> got = downloadBuffer<uint8_t>(dev, bBlocks, NB * PDBC_STRIDE);
    const std::vector<F2> gotC = downloadBuffer<F2>(dev, bAppC, NC + 8u), gotP = downloadBuffer<F2>(dev, bAppP, NC + 8u);
    uint32_t bad = 0, invalid = 0;
    for (uint32_t w = 0; w < NC + 8u; ++w)
    {
        const F2 wantA = (w < NC) ? F2{ 0, 0 } : F2{ 9.0f, -9.0f };
        cmpF4("clothAppliedForces", w, F4{ gotC[w].x, gotC[w].y, 0, 0 }, F4{ wantA.x, wantA.y, 0, 0 }, bad);
        cmpF4("particleAppliedForces", w, F4{ gotP[w].x, gotP[w].y, 0, 0 }, F4{ wantA.x, wantA.y, 0, 0 }, bad);
    }
    for (uint32_t w = 0; w < NC; ++w)
    {
        const CpContact& c = cs[w];
        const size_t base = size_t(w / 32u) * PDBC_STRIDE;
        const uint32_t l = w & 31u;
        const F4 gNp = getAt<F4>(got, base + PDBC_normal_pen + l * 16), gBc = getAt<F4>(got, base + PDBC_barycentric + l * 16);
        const float gClamp = getAt<float>(got, base + PDBC_maxPenBiasClamp + l * 4);
        if (!pairValid(c.pair.ref1))
        {
            ++invalid;
            if (getAt<uint32_t>(got, base + PDBC_normal_pen + l * 16) != POISON) { ++bad; std::printf("    invalid lane %u written\n", w); }
            continue;
        }
        const HostCloth& h = C.cloth[refId(c.pair.ref1)];
        const HostSystem& hs = W.sys[particleSystemIdOf(c.pair.ref0)];
        const F4 delta1 = projectTri(h.accDelta, h.tris[refFeature(c.pair.ref1)], c.bary);
        const F4 dp = hs.deltaP[particleIndexOf(c.pair.ref0)];
        const V3 normal = v3(-c.normalPen.x, -c.normalPen.y, -c.normalPen.z);
        const V3 delta = v3(delta1.x - dp.x, delta1.y - dp.y, delta1.z - dp.z);
        const float restDistance = hs.restOffset + h.restOffset;
        const float pen = c.normalPen.w + dot(normal, delta) - restDistance;
        cmpF4("normal_pen", w, gNp, F4{ normal.x, normal.y, normal.z, pen }, bad);
        cmpF4("barycentric", w, gBc, c.bary, bad);
        cmpF("maxPenBiasClamp", w, gClamp, selectMax(h.penBiasClamp, hs.penBiasClamp), bad);
    }
    PXG_CHECK_MSG(bad == 0, "cloth_particleContactPrepareLaunch: %u mismatches", bad);
    PXG_CHECK_MSG(invalid > 0, "coverage: no INVALID pair");
}

static void runCpQuerySolve(bool isVelocityIteration, uint32_t seed)
{
    PxgSlangComputeDevice* dev = testDevice();
    World W;
    buildWorld(W, dev, seed);
    ClothWorld C;
    buildClothWorld(C, dev, seed + 7u);
    Rng rng{ seed * 17u + 5u };
    std::vector<CpContact> cs = makeCpContacts(seed);
    const uint32_t NC = uint32_t(cs.size());

    std::vector<PairInfo> pairs;
    std::vector<uint8_t> blocks(((NC + 31u) / 32u) * PDBC_STRIDE, 0);
    std::vector<float> clamps(NC);
    for (uint32_t c = 0; c < NC; ++c)
    {
        if (c % 7u == 2u && pairValid(cs[c].pair.ref1)) cs[c].pair.ref1 = setActive(cs[c].pair.ref1, true);
        pairs.push_back(cs[c].pair);
        clamps[c] = -rng.uni(0.1f, 2.0f);
        const size_t base = size_t(c / 32u) * PDBC_STRIDE;
        putAt(blocks, base + PDBC_normal_pen + (c & 31u) * 16, cs[c].normalPen);
        putAt(blocks, base + PDBC_barycentric + (c & 31u) * 16, cs[c].bary);
        putAt(blocks, base + PDBC_maxPenBiasClamp + (c & 31u) * 4, clamps[c]);
    }
    std::vector<F2> appliedC(NC), appliedP(NC);
    for (uint32_t w = 0; w < NC; ++w)
    {
        appliedC[w] = (w % 5u == 0u) ? F2{ 0.f, 0.f } : F2{ rng.uni(0.0f, 0.04f), rng.uni(-0.03f, 0.03f) };
        appliedP[w] = appliedC[w];
    }
    std::vector<uint32_t> num = { NC };
    PxgBufferHandle bPairs = W.make(pairs), bAppC = W.make(appliedC), bAppP = W.make(appliedP);
    const uint64_t aBlocks = W.addr(W.make(blocks)), aNum = W.addr(W.make(num));

    // ---- query ----
    {
        CpQueryPush q = { C.pool, W.psysPool, W.addr(bPairs), aBlocks, aNum, kDt, isVelocityIteration ? 1u : 0u };
        Kernel k("femClothPsSolveBda.spv", "cloth_queryCPContactReferenceCountLaunch", {}, sizeof(CpQueryPush));
        k.dispatch(GRID, {}, q);
        dev->flush();
    }
    std::vector<PairInfo> wantPairs = pairs;
    uint32_t nActive = 0, nKinematic = 0;
    for (uint32_t w = 0; w < NC; ++w)
    {
        const PairInfo& pi = pairs[w];
        if (!pairValid(pi.ref1)) { wantPairs[w].ref1 = setActive(pi.ref1, false); continue; }
        const uint32_t sys = particleSystemIdOf(pi.ref0), pidx = particleIndexOf(pi.ref0);
        const ParticleView pv = W.particleView(sys, pidx);
        if (pv.sortedDeltaP.w == 0.0f) { wantPairs[w].ref1 = setActive(pi.ref1, false); ++nKinematic; continue; }
        DbPart clothPart, psPart;
        clothPart.readCloth(C.cloth[refId(pi.ref1)], pi.ref1, cs[w].bary, nullptr, true, kDt, isVelocityIteration);
        psPart.readParticle(pv, false, false, kDt, isVelocityIteration);
        const ContactPair cp = contactPairDbDb(cs[w].normalPen, clamps[w]);
        SolveOut so;
        const bool isActive = solveDbDbContact(clothPart, psPart, cp, so, 0.f, 0.f, 0.f, kDt, true, isVelocityIteration);
        wantPairs[w].ref1 = setActive(pi.ref1, isActive);
        if (isActive)
        {
            ++nActive;
            bumpCloth(C.cloth[refId(pi.ref1)], pi.ref1, clothPart.bc);
            W.sys[sys].accum[pidx].w += 1.0f;
        }
    }
    uint32_t bad = 0;
    {
        const std::vector<PairInfo> got = downloadBuffer<PairInfo>(dev, bPairs, NC);
        for (uint32_t w = 0; w < NC; ++w)
            if (got[w].ref1 != wantPairs[w].ref1) { if (++bad <= 8) std::printf("    pair %u got %llx want %llx\n", w, (unsigned long long)got[w].ref1, (unsigned long long)wantPairs[w].ref1); }
        for (uint32_t s = 0; s < 3; s += 2)
        {
            cmpVec("query mDeltaPos", C.readDelta(s), C.cloth[s].delta, bad);
            cmpVec("query mAccumDeltaP", downloadBuffer<F4>(dev, W.bAccum[s], W.sys[s].accum.size()), W.sys[s].accum, bad);
        }
    }
    PXG_CHECK_MSG(bad == 0, "cloth_queryCPContactReferenceCountLaunch (%s): %u mismatches", isVelocityIteration ? "vel" : "pos", bad);
    PXG_CHECK_MSG(nActive > 0 && nKinematic > 0, "coverage: active %u kinematic %u", nActive, nKinematic);

    // ---- the two solve halves on the query's output ----
    std::vector<F4> deltaInit(NC, F4{ 7.f, 7.f, 7.f, 7.f });
    PxgBufferHandle bDeltaP = W.make(deltaInit);
    {
        CpSolveClothPush sp = { C.pool, W.psysPool, W.addr(bPairs), aBlocks, aNum, W.addr(bAppC), C.surfMaterials, kDt,
                                isVelocityIteration ? 1u : 0u };
        Kernel k("femClothPsSolveBda.spv", "cloth_solveCPOutputClothDeltaVLaunch", {}, sizeof(CpSolveClothPush));
        k.dispatch(GRID, {}, sp);
        CpSolvePsPush pp = { C.pool, W.psysPool, W.addr(bPairs), aBlocks, aNum, W.addr(bDeltaP), W.addr(bAppP),
                             C.surfMaterials, kDt, isVelocityIteration ? 1u : 0u };
        Kernel k2("femClothPsSolveBda.spv", "cloth_solveCPOutputParticleDeltaVLaunch", {}, sizeof(CpSolvePsPush));
        k2.dispatch(GRID, {}, pp);
        dev->flush();
    }
    // Both halves read the SAME pre-solve mDeltaPos .w (the cloth half only touches .xyz).
    const HostCloth snap[3] = { C.cloth[0], C.cloth[1], C.cloth[2] };
    std::vector<F2> wantC = appliedC, wantP = appliedP;
    std::vector<F4> wantDeltaP(NC);
    uint32_t nApplied = 0;
    for (uint32_t w = 0; w < NC; ++w)
    {
        const PairInfo& pi = wantPairs[w];
        const uint32_t sys = particleSystemIdOf(pi.ref0), pidx = particleIndexOf(pi.ref0);
        const ParticleView pv = W.particleView(sys, pidx);
        if (pv.sortedDeltaP.w == 0.0f) { wantDeltaP[w] = F4{ 0, 0, 0, 0 }; continue; }
        if (!pairActive(pi.ref1)) { wantDeltaP[w] = F4{ 0, 0, 0, 0 }; continue; }
        DbPart clothPart, psPart;
        clothPart.readCloth(snap[refId(pi.ref1)], pi.ref1, cs[w].bary, C.surfaceFriction.data(), false, kDt, isVelocityIteration);
        psPart.readParticle(pv, true, true, kDt, isVelocityIteration);
        const ContactPair cp = contactPairDbDb(cs[w].normalPen, clamps[w]);
        const float mu = (clothPart.friction + psPart.friction) * 0.5f;
        SolveOut so;
        if (solveDbDbContact(clothPart, psPart, cp, so, appliedC[w].x, appliedC[w].y, mu, kDt, false, isVelocityIteration))
        {
            ++nApplied;
            V3 delta;
            so.computeDelta(cp, delta, kDt);
            clothPart.writeCloth(C.cloth[refId(pi.ref1)], delta);
            wantDeltaP[w] = psPart.writeParticle(neg(delta), so);
            wantC[w] = F2{ so.accumulatedDeltaLambdaN, appliedC[w].y + so.deltaLambdaT };
            wantP[w] = F2{ so.accumulatedDeltaLambdaN, appliedP[w].y + so.deltaLambdaT };
        }
        else
            wantDeltaP[w] = F4{ 0, 0, 0, 0 };
    }
    bad = 0;
    {
        cmpVec("deltaP", downloadBuffer<F4>(dev, bDeltaP, NC), wantDeltaP, bad);
        const std::vector<F2> gc = downloadBuffer<F2>(dev, bAppC, NC), gp = downloadBuffer<F2>(dev, bAppP, NC);
        for (uint32_t w = 0; w < NC; ++w)
        {
            cmpF4("appliedCloth", w, F4{ gc[w].x, gc[w].y, 0, 0 }, F4{ wantC[w].x, wantC[w].y, 0, 0 }, bad);
            cmpF4("appliedParticle", w, F4{ gp[w].x, gp[w].y, 0, 0 }, F4{ wantP[w].x, wantP[w].y, 0, 0 }, bad);
        }
        for (uint32_t s = 0; s < 3; s += 2)
            cmpVec("solve mDeltaPos", C.readDelta(s), C.cloth[s].delta, bad);
    }
    PXG_CHECK_MSG(bad == 0, "cloth_solveCPOutput{Cloth,Particle}DeltaVLaunch (%s): %u mismatches", isVelocityIteration ? "vel" : "pos", bad);
    PXG_CHECK_MSG(nApplied > 0, "coverage: no CP contact applied");
}

PXG_TEST(femClothSolve512ParticleContactPos) { runCpQuerySolve(false, 41u); }
PXG_TEST(femClothSolve512ParticleContactVel) { runCpQuerySolve(true, 42u); }

// ===========================================================================
// Cloth-cloth attachments
// ===========================================================================
// NA attachments between cloths 0 and 2 (every fifth a self attachment).  Attachment 0 joins
// two pinned vertices (inverse mass 0 on both sides), the solve's wTot == 0 skip.
static std::vector<DefAttachment> makeCcAttachments(uint32_t NA, uint32_t seed)
{
    Rng rng{ seed * 29u + 7u };
    ScatterBudget budget;
    std::vector<DefAttachment> as;
    {
        DefAttachment d;
        d.pair.ref0 = encodeRef(0, REF_eVERTEX, 3u * NTRI + 6u);   // 150 % 7 == 3: pinned
        d.pair.ref1 = encodeRef(2, REF_eVERTEX, 3u * NTRI + 13u);  // 157 % 7 == 3: pinned
        d.bOrP0 = F4{ 1e30f, -1e30f, 1e30f, -1e30f };
        d.bary1 = F4{ -1e30f, 1e30f, -1e30f, 1e30f };
        budget.fits(d.pair.ref0, true, d.pair.ref1);
        as.push_back(d);
    }
    for (uint32_t tries = 0; as.size() < NA && tries < 10000u; ++tries)
    {
        DefAttachment d;
        const uint32_t a = uint32_t(as.size());
        const uint32_t c0 = (rng.next() & 1u) ? 2u : 0u;
        const uint32_t c1 = (a % 5u == 4u) ? c0 : (2u - c0);
        d.pair.ref0 = randomClothRef(rng, c0, true);
        d.pair.ref1 = randomClothRef(rng, c1, true);
        d.bOrP0 = randomBary(rng, a % 6u == 2u);
        d.bary1 = randomBary(rng, a % 7u == 3u);
        if (budget.fits(d.pair.ref0, true, d.pair.ref1))
            as.push_back(d);
    }
    return as;
}

static void putDbdba(std::vector<uint8_t>& blk, uint32_t w, const DefAttachment& d, F4 prior0, F4 prior1)
{
    const size_t base = size_t(w / 32u) * DBDBA_STRIDE;
    const uint32_t l = w & 31u;
    putAt(blk, base + DBDBA_barycentric0 + l * 16, refType(d.pair.ref0) == REF_eVERTEX ? prior0 : d.bOrP0);
    putAt(blk, base + DBDBA_barycentric1 + l * 16, refType(d.pair.ref1) == REF_eVERTEX ? prior1 : d.bary1);
    putAt(blk, base + DBDBA_deformableRef0 + l * 8, d.pair.ref0);
    putAt(blk, base + DBDBA_deformableRef1 + l * 8, d.pair.ref1);
}

struct CaQueryPush { uint64_t clothes, attachmentBlocks; uint32_t numAttachments, pad20; };
static_assert(sizeof(CaQueryPush) == 24, "cloth_queryClothClothAttachmentReferenceCountLaunch blob");
struct CaSolvePush { uint64_t clothes, attachmentBlocks; uint32_t numAttachments; float dt; uint32_t isVelocityIteration, pad28; };
static_assert(sizeof(CaSolvePush) == 32, "cloth_solveOutputAttachmentClothClothDeltaVLaunch blob");

static void runCcAttachQuerySolve(bool isVelocityIteration, uint32_t seed)
{
    PxgSlangComputeDevice* dev = testDevice();
    ClothWorld C;
    buildClothWorld(C, dev, seed);
    const uint32_t NA = 45u;
    const std::vector<DefAttachment> as = makeCcAttachments(NA, seed);
    PXG_CHECK_MSG(as.size() == NA, "attachment generator fell short (%zu)", as.size());

    std::vector<uint8_t> blocks(((NA + 31u) / 32u) * DBDBA_STRIDE, 0);
    for (uint32_t w = 0; w < NA; ++w)
        putDbdba(blocks, w, as[w], F4{ 7.0f, -3.0f, 11.0f, 0.5f }, F4{ -9.0f, 4.0f, 0.25f, 6.0f });
    const uint64_t aBlocks = C.addr(C.make(blocks));

    // ---- query ----
    {
        CaQueryPush q = { C.pool, aBlocks, NA, 0u };
        Kernel k("femClothAttachBda.spv", "cloth_queryClothClothAttachmentReferenceCountLaunch", {}, sizeof(CaQueryPush));
        k.dispatch(GRID, {}, q);
        dev->flush();
    }
    for (const DefAttachment& d : as)
    {
        bumpCloth(C.cloth[refId(d.pair.ref0)], d.pair.ref0, d.bOrP0);
        bumpCloth(C.cloth[refId(d.pair.ref1)], d.pair.ref1, d.bary1);
    }
    uint32_t bad = 0;
    for (uint32_t ci = 0; ci < 3; ci += 2)
        cmpVec("query mDeltaPos", C.readDelta(ci), C.cloth[ci].delta, bad);
    PXG_CHECK_MSG(bad == 0, "cloth_queryClothClothAttachmentReferenceCountLaunch: %u mismatches", bad);

    // ---- solve ----
    {
        CaSolvePush sp = { C.pool, aBlocks, NA, kDt, isVelocityIteration ? 0xABCDEF01u : 0xABCDEF00u, 0u };
        Kernel k("femClothAttachBda.spv", "cloth_solveOutputAttachmentClothClothDeltaVLaunch", {}, sizeof(CaSolvePush));
        k.dispatch(GRID, {}, sp);
        dev->flush();
    }
    // The solve reads the post-query .w and positions / velocities, which no scatter touches.
    const HostCloth snap[3] = { C.cloth[0], C.cloth[1], C.cloth[2] };
    uint32_t nSkipped = 0, nSolved = 0, nSelf = 0;
    for (const DefAttachment& d : as)
    {
        DbPart db0, db1;
        const V3 pos0 = db0.readClothAttachmentDbDb(snap[refId(d.pair.ref0)], d.pair.ref0, d.bOrP0, kDt, isVelocityIteration);
        const V3 pos1 = db1.readClothAttachmentDbDb(snap[refId(d.pair.ref1)], d.pair.ref1, d.bary1, kDt, isVelocityIteration);
        V3 delta;
        if (solveDbDbAttachment(db0, db1, pos0, pos1, delta))
        {
            ++nSolved;
            db0.writeCloth(C.cloth[refId(d.pair.ref0)], delta);
            db1.writeCloth(C.cloth[refId(d.pair.ref1)], neg(delta));
        }
        else
            ++nSkipped;
        if (refId(d.pair.ref0) == refId(d.pair.ref1)) ++nSelf;
    }
    bad = 0;
    for (uint32_t ci = 0; ci < 3; ci += 2)
        cmpVec("solve mDeltaPos", C.readDelta(ci), C.cloth[ci].delta, bad);
    PXG_CHECK_MSG(bad == 0, "cloth_solveOutputAttachmentClothClothDeltaVLaunch (%s): %u mismatches",
                  isVelocityIteration ? "vel" : "pos", bad);
    PXG_CHECK_MSG(nSolved > 0 && nSkipped > 0 && nSelf > 0, "coverage: solved %u skipped %u self %u", nSolved, nSkipped,
                  nSelf);
}

PXG_TEST(femClothSolve512ClothAttachPos) { runCcAttachQuerySolve(false, 71u); }
PXG_TEST(femClothSolve512ClothAttachVel) { runCcAttachQuerySolve(true, 72u); }

// ===========================================================================
// Cloth-cloth contacts
// ===========================================================================
struct CcPrepPush
{
    uint64_t clothes, contactInfos, numContacts;
    uint32_t maxContacts, pad28;
    uint64_t clothMaterials, updateContactPairs;
};
static_assert(sizeof(CcPrepPush) == 48, "cloth_clothContactPrepareLaunch blob");
struct CcQueryPush { uint64_t cloths, contactInfos, numContacts; };
static_assert(sizeof(CcQueryPush) == 24, "cloth_queryClothClothContact{VT,EE}CountLaunch blob");
struct CcSolvePush { uint64_t cloths, contactInfos, numContacts; float dt; uint32_t isVelocityIteration; };
static_assert(sizeof(CcSolvePush) == 32, "cloth_solveClothClothDelta{VT,EE}Launch blob");

static V3 triPoint(const std::vector<F4>& p, U4 t, float a, float b, float c)
{
    return add(add(mul(xyz(p[t.x]), a), mul(xyz(p[t.y]), b)), mul(xyz(p[t.z]), c));
}

static V3 unitNormal(V3 a, V3 b, V3 c)
{
    V3 n = cross(sub(b, a), sub(c, a));
    normalize(n);
    return n;
}

static void setXYZ(F4& f, V3 v) { f.x = v.x; f.y = v.y; f.z = v.z; }

// VT contacts: a lone vertex of cloth c0 placed near a triangle of cloth c1 (every fourth a
// self contact), at a height in [-1.4, 1.4] x the rest distance (beyond 1 is separated), one
// exactly in the plane (the triangle-normal fallback).  Each vertex and triangle is used once,
// so no vertex takes more than one scatter.  The rest positions and the prev positions are
// shaped alongside, for the prepare's filter and the solve's initPen.
struct CcContact { PairInfo pair; };

static std::vector<CcContact> shapeVtContacts(ClothWorld& C, uint32_t n, uint32_t seed)
{
    Rng rng{ seed * 13u + 9u };
    std::vector<CcContact> cs;
    for (uint32_t k = 0; k < n && k < NLOOSE && k < NTRI; ++k)
    {
        const uint32_t c0 = (k & 1u) ? 2u : 0u;
        const uint32_t c1 = (k % 4u == 3u) ? c0 : 2u - c0;
        HostCloth& h0 = C.cloth[c0];
        HostCloth& h1 = C.cloth[c1];
        const uint32_t v = 3u * NTRI + k;
        const uint32_t tri = (k * 5u + 1u) % NTRI;
        const U4 t = h1.tris[tri];
        const F4 b = randomBary(rng, false);
        const V3 onTri = triPoint(h1.pos, t, b.x, b.y, b.z);
        const V3 n = unitNormal(xyz(h1.pos[t.x]), xyz(h1.pos[t.y]), xyz(h1.pos[t.z]));
        const float restDist = h0.restOffset + h1.restOffset;
        const float hgt = (k == 5u) ? 0.0f : rng.uni(-1.4f, 1.4f) * restDist;
        setXYZ(h0.pos[v], add(onTri, mul(n, hgt)));
        // prev position: a small motion since the last pair refresh.
        setXYZ(h0.prevPos[v], add(xyz(h0.pos[v]), v3(rng.uni(-0.004f, 0.004f), rng.uni(-0.004f, 0.004f), rng.uni(-0.004f, 0.004f))));
        for (uint32_t j = 0; j < 3; ++j)
        {
            const uint32_t tv = (j == 0) ? t.x : (j == 1) ? t.y : t.z;
            setXYZ(h1.prevPos[tv], add(xyz(h1.pos[tv]), v3(rng.uni(-0.004f, 0.004f), rng.uni(-0.004f, 0.004f), rng.uni(-0.004f, 0.004f))));
        }
        // rest geometry for the prepare's self filter: near (filtered) or far.
        const V3 restOnTri = triPoint(h1.restPos, t, b.x, b.y, b.z);
        const V3 rn = unitNormal(xyz(h1.restPos[t.x]), xyz(h1.restPos[t.y]), xyz(h1.restPos[t.z]));
        setXYZ(h0.restPos[v], add(restOnTri, mul(rn, (k % 3u == 0u) ? 0.5f * h0.selfCollisionFilterDistance
                                                                     : 3.0f * h0.selfCollisionFilterDistance)));
        CcContact c;
        c.pair.ref0 = encodeRef(c0, REF_eVERTEX, v);
        c.pair.ref1 = encodeRef(c1, REF_eTRIANGLE, tri);
        cs.push_back(c);
    }
    return cs;
}

// EE contacts: edge (localEdge) of a triangle of cloth c0 and an edge of a triangle of cloth
// c1, the second edge rebuilt to cross the first's interior at a gap in [-1.4, 1.4] x the rest
// distance; one exactly crossing (the edge-cross normal fallback) and one parallel (no contact).
static std::vector<CcContact> shapeEeContacts(ClothWorld& C, uint32_t n, uint32_t seed)
{
    Rng rng{ seed * 19u + 3u };
    std::vector<CcContact> cs;
    for (uint32_t k = 0; k < n && 2u * k + 1u < NTRI; ++k)
    {
        const uint32_t c0 = (k & 1u) ? 2u : 0u;
        const uint32_t c1 = (k % 4u == 2u) ? c0 : 2u - c0;
        HostCloth& h0 = C.cloth[c0];
        HostCloth& h1 = C.cloth[c1];
        const uint32_t tri0 = 2u * k, tri1 = 2u * k + 1u;
        const uint32_t e0 = rng.next() % 3u, e1 = rng.next() % 3u;
        const U4 t0 = h0.tris[tri0], t1 = h1.tris[tri1];
        const uint32_t i0[3] = { t0.x, t0.y, t0.z }, i1[3] = { t1.x, t1.y, t1.z };
        const uint32_t a0 = i0[e0], a1 = i0[(e0 + 1u) % 3u];
        const uint32_t b0 = i1[e1], b1 = i1[(e1 + 1u) % 3u];
        const V3 p0 = xyz(h0.pos[a0]), p1 = xyz(h0.pos[a1]);
        const V3 dir0 = sub(p1, p0);
        V3 u0 = dir0;
        const float len0 = normalize(u0);
        V3 side, up;
        computeBasisVectors(u0, side, up);
        const float s = rng.uni(0.25f, 0.75f);
        const V3 mid = add(p0, mul(dir0, s));
        const float restDist = h0.restOffset + h1.restOffset;
        const float gap = (k == 3u) ? 0.0f : rng.uni(-1.4f, 1.4f) * restDist;
        // Edge 1 is as long as edge 0: closestPtLineLine mixes the two edge lengths in its
        // s / t terms (deformableCollision.cuh: b1 * a01 is scaled by 1 / r1Len where the
        // minimizer has 1 / r0Len, and a01 * b0 by 1 / r0Len where it has 1 / r1Len), so it
        // finds the true closest points only for equal lengths.  The oracle transcribes the
        // CUDA as written; equal lengths just keep the shaped gap the actual distance.
        V3 dir1 = (k == 6u) ? u0 : add(mul(side, 0.8f), mul(u0, 0.3f));   // k == 6: parallel
        normalize(dir1);
        const float t = rng.uni(0.3f, 0.7f);
        const V3 c = add(mid, mul(up, gap));
        setXYZ(h1.pos[b0], sub(c, mul(dir1, t * len0)));
        setXYZ(h1.pos[b1], add(c, mul(dir1, (1.0f - t) * len0)));
        const uint32_t touched[4] = { a0, a1, b0, b1 };
        for (uint32_t j = 0; j < 4; ++j)
        {
            HostCloth& h = (j < 2) ? h0 : h1;
            setXYZ(h.prevPos[touched[j]], add(xyz(h.pos[touched[j]]),
                                              v3(rng.uni(-0.004f, 0.004f), rng.uni(-0.004f, 0.004f), rng.uni(-0.004f, 0.004f))));
        }
        // rest geometry: the self pairs get edges at a filtered or unfiltered rest distance.
        const V3 r0 = xyz(h0.restPos[a0]), r1 = xyz(h0.restPos[a1]);
        V3 ru = sub(r1, r0);
        normalize(ru);
        V3 rside, rup;
        computeBasisVectors(ru, rside, rup);
        const V3 rmid = add(r0, mul(sub(r1, r0), 0.5f));
        const float rgap = (k % 3u == 0u) ? 0.5f * h0.selfCollisionFilterDistance : 3.0f * h0.selfCollisionFilterDistance;
        setXYZ(h1.restPos[b0], add(add(rmid, mul(rup, rgap)), mul(rside, -0.3f)));
        setXYZ(h1.restPos[b1], add(add(rmid, mul(rup, rgap)), mul(rside, 0.3f)));
        CcContact cc;
        cc.pair.ref0 = encodeEdgeRef(c0, tri0, e0);
        cc.pair.ref1 = encodeEdgeRef(c1, tri1, e1);
        cs.push_back(cc);
    }
    return cs;
}

static uint32_t pick3(U4 v, uint32_t i) { return (i == 0u) ? v.x : (i == 1u) ? v.y : v.z; }

static void eeVertices(const ClothWorld& C, const PairInfo& p, uint32_t vi[4])
{
    const HostCloth& h0 = C.cloth[refId(p.ref0)];
    const HostCloth& h1 = C.cloth[refId(p.ref1)];
    const uint32_t e0 = refLocalEdge(p.ref0), e1 = refLocalEdge(p.ref1);
    const U4 t0 = h0.tris[refFeature(p.ref0)], t1 = h1.tris[refFeature(p.ref1)];
    vi[0] = pick3(t0, e0);
    vi[1] = pick3(t0, (e0 + 1u) % 3u);
    vi[2] = pick3(t1, e1);
    vi[3] = pick3(t1, (e1 + 1u) % 3u);
}

PXG_TEST(femClothSolve512ClothContactPrepare)
{
    PxgSlangComputeDevice* dev = testDevice();
    for (int isEE = 0; isEE < 2; ++isEE)
    {
        ClothWorld C;
        buildClothHost(C, 0x515u + uint32_t(isEE));
        std::vector<CcContact> cs = isEE ? shapeEeContacts(C, 20u, 7u) : shapeVtContacts(C, 32u, 7u);
        uploadClothWorld(C, dev);
        // An INVALID pair passes through untouched; the clamp drops the last three.
        cs[2].pair.ref1 |= REF_INVALID;
        const uint32_t NCraw = uint32_t(cs.size()), maxContacts = NCraw - 3u;
        std::vector<PairInfo> pairs;
        for (const CcContact& c : cs) pairs.push_back(c.pair);
        std::vector<uint32_t> num = { NCraw };
        std::vector<uint32_t> gate = { 0xA5A5A501u };   // PxU8 1 in byte0, garbage above
        PxgBufferHandle bPairs = C.make(pairs), bNum = C.make(num);
        CcPrepPush p = { C.pool, C.addr(bPairs), C.addr(bNum), maxContacts, 0u, 0xDEADBEEFull, C.addr(C.make(gate)) };
        Kernel k("femClothClothContactBda.spv", "cloth_clothContactPrepareLaunch", {}, sizeof(CcPrepPush));
        k.dispatch(GRID, {}, p);
        dev->flush();

        std::vector<PairInfo> want = pairs;
        uint32_t nFiltered = 0, nCross = 0;
        for (uint32_t w = 0; w < maxContacts; ++w)
        {
            const PairInfo& pi = pairs[w];
            if (refId(pi.ref0) != refId(pi.ref1)) { ++nCross; continue; }
            const HostCloth& h0 = C.cloth[refId(pi.ref0)];
            const HostCloth& h1 = C.cloth[refId(pi.ref1)];
            float restDistSq;
            if (refType(pi.ref0) == REF_eTRIANGLE_EDGE)
            {
                uint32_t vi[4];
                eeVertices(C, pi, vi);
                float s, t;
                closestPtEdgeEdge(xyz(h0.restPos[vi[0]]), xyz(h0.restPos[vi[1]]), xyz(h1.restPos[vi[2]]),
                                  xyz(h1.restPos[vi[3]]), s, t, restDistSq);
            }
            else
            {
                const U4 t = h1.tris[refFeature(pi.ref1)];
                const V3 r0 = xyz(h0.restPos[refFeature(pi.ref0)]), r1 = xyz(h1.restPos[t.x]), r2 = xyz(h1.restPos[t.y]),
                         r3 = xyz(h1.restPos[t.z]);
                const V3 closest = closestPtPointTriangle2(r0, r1, r2, r3, sub(r2, r1), sub(r3, r1));
                restDistSq = dot(sub(r0, closest), sub(r0, closest));
            }
            if (restDistSq <= h0.selfCollisionFilterDistance * h0.selfCollisionFilterDistance)
            {
                want[w].ref1 |= REF_INVALID;
                ++nFiltered;
            }
        }
        uint32_t bad = 0;
        const std::vector<PairInfo> got = downloadBuffer<PairInfo>(dev, bPairs, NCraw);
        for (uint32_t w = 0; w < NCraw; ++w)
            if (got[w].ref0 != want[w].ref0 || got[w].ref1 != want[w].ref1)
            { if (++bad <= 8) std::printf("    %s pair %u got %llx want %llx\n", isEE ? "EE" : "VT", w, (unsigned long long)got[w].ref1, (unsigned long long)want[w].ref1); }
        const uint32_t gotNum = downloadBuffer<uint32_t>(dev, bNum, 1)[0];
        if (gotNum != maxContacts) { ++bad; std::printf("    numContacts %u want %u\n", gotNum, maxContacts); }
        PXG_CHECK_MSG(bad == 0, "cloth_clothContactPrepareLaunch (%s): %u mismatches", isEE ? "EE" : "VT", bad);
        PXG_CHECK_MSG(nFiltered > 0 && nCross > 0, "coverage (%s): filtered %u cross %u", isEE ? "EE" : "VT", nFiltered, nCross);

        // A zero gate leaves everything as it was, the count included.
        std::vector<PairInfo> fresh = pairs;
        PxgBufferHandle bPairs2 = C.make(fresh), bNum2 = C.make(num);
        std::vector<uint32_t> gate0 = { 0xA5A5A500u };
        CcPrepPush p0 = { C.pool, C.addr(bPairs2), C.addr(bNum2), maxContacts, 0u, 0ull, C.addr(C.make(gate0)) };
        k.dispatch(GRID, {}, p0);
        dev->flush();
        const std::vector<PairInfo> got0 = downloadBuffer<PairInfo>(dev, bPairs2, NCraw);
        bad = 0;
        for (uint32_t w = 0; w < NCraw; ++w)
            if (got0[w].ref1 != pairs[w].ref1) ++bad;
        if (downloadBuffer<uint32_t>(dev, bNum2, 1)[0] != NCraw) ++bad;
        PXG_CHECK_MSG(bad == 0, "cloth_clothContactPrepareLaunch (%s) with the gate off: %u changes", isEE ? "EE" : "VT", bad);
    }
}

// The count + solve pair of one stream (VT or EE) in a position or velocity iteration.
static void runCcQuerySolve(bool isEE, bool isVelocityIteration, uint32_t seed)
{
    PxgSlangComputeDevice* dev = testDevice();
    ClothWorld C;
    buildClothHost(C, seed);
    std::vector<CcContact> cs = isEE ? shapeEeContacts(C, 20u, seed) : shapeVtContacts(C, 34u, seed);
    uploadClothWorld(C, dev);
    // Pair 1 is INVALID (skipped by the count), pair 4 arrives ACTIVE (recomputed anyway).
    cs[1].pair.ref1 |= REF_INVALID;
    cs[4].pair.ref1 = setActive(cs[4].pair.ref1, true);
    const uint32_t NC = uint32_t(cs.size());
    std::vector<PairInfo> pairs;
    for (const CcContact& c : cs) pairs.push_back(c.pair);
    std::vector<uint32_t> num = { NC };
    PxgBufferHandle bPairs = C.make(pairs);
    const uint64_t aNum = C.addr(C.make(num));
    const char* spv = "femClothClothContactBda.spv";

    // ---- count ----
    {
        CcQueryPush q = { C.pool, C.addr(bPairs), aNum };
        Kernel k(spv, isEE ? "cloth_queryClothClothContactEECountLaunch" : "cloth_queryClothClothContactVTCountLaunch", {},
                 sizeof(CcQueryPush));
        k.dispatch(GRID, {}, q);
        dev->flush();
    }
    struct Geo { bool active; uint32_t vi[4]; uint32_t ci[4]; F4 w; V3 n, disp; };
    std::vector<Geo> geo(NC);
    std::vector<PairInfo> wantPairs = pairs;
    uint32_t nActive = 0, nInactive = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const PairInfo& pi = pairs[i];
        Geo& g = geo[i];
        g.active = false;
        if (!pairValid(pi.ref1)) continue;
        const uint32_t c0 = refId(pi.ref0), c1 = refId(pi.ref1);
        if (isEE)
        {
            eeVertices(C, pi, g.vi);
            g.ci[0] = g.ci[1] = c0; g.ci[2] = g.ci[3] = c1;
        }
        else
        {
            const U4 t = C.cloth[c1].tris[refFeature(pi.ref1)];
            g.vi[0] = refFeature(pi.ref0); g.vi[1] = t.x; g.vi[2] = t.y; g.vi[3] = t.z;
            g.ci[0] = c0; g.ci[1] = g.ci[2] = g.ci[3] = c1;
        }
        const F4 x0 = C.cloth[g.ci[0]].pos[g.vi[0]], x1 = C.cloth[g.ci[1]].pos[g.vi[1]],
                 x2 = C.cloth[g.ci[2]].pos[g.vi[2]], x3 = C.cloth[g.ci[3]].pos[g.vi[3]];
        g.active = isEE ? eeContact(C.cloth[c0], C.cloth[c1], x0, x1, x2, x3, g.w, g.n, g.disp)
                        : vtContact(C.cloth[c0], C.cloth[c1], x0, x1, x2, x3, g.w, g.n, g.disp);
        wantPairs[i].ref1 = setActive(pi.ref1, g.active);
        if (!g.active) { ++nInactive; continue; }
        ++nActive;
        const float ws[4] = { g.w.x, g.w.y, g.w.z, g.w.w };
        for (uint32_t j = 0; j < 4; ++j)
            if (std::fabs(ws[j]) > 1e-3f) C.cloth[g.ci[j]].delta[g.vi[j]].w += std::fabs(ws[j]);
    }
    uint32_t bad = 0;
    {
        const std::vector<PairInfo> got = downloadBuffer<PairInfo>(dev, bPairs, NC);
        for (uint32_t w = 0; w < NC; ++w)
            if (got[w].ref0 != wantPairs[w].ref0 || got[w].ref1 != wantPairs[w].ref1)
            { if (++bad <= 8) std::printf("    pair %u got %llx want %llx\n", w, (unsigned long long)got[w].ref1, (unsigned long long)wantPairs[w].ref1); }
        for (uint32_t ci = 0; ci < 3; ci += 2)
            cmpVec("count mDeltaPos", C.readDelta(ci), C.cloth[ci].delta, bad);
    }
    PXG_CHECK_MSG(bad == 0, "cloth_queryClothClothContact%sCountLaunch: %u mismatches", isEE ? "EE" : "VT", bad);
    PXG_CHECK_MSG(nActive > 2 && nInactive > 0, "coverage: active %u inactive %u", nActive, nInactive);

    // ---- solve ----
    {
        CcSolvePush sp = { C.pool, C.addr(bPairs), aNum, kDt, isVelocityIteration ? 0x5A5A5A01u : 0x5A5A5A00u };
        Kernel k(spv, isEE ? "cloth_solveClothClothDeltaEELaunch" : "cloth_solveClothClothDeltaVTLaunch", {},
                 sizeof(CcSolvePush));
        k.dispatch(GRID, {}, sp);
        dev->flush();
    }
    const HostCloth snap[3] = { C.cloth[0], C.cloth[1], C.cloth[2] };
    uint32_t nApplied = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const Geo& g = geo[i];
        if (!g.active) continue;
        const PairInfo& pi = wantPairs[i];
        const HostCloth& h0 = snap[refId(pi.ref0)];
        const HostCloth& h1 = snap[refId(pi.ref1)];
        const F4 x[4] = { snap[g.ci[0]].pos[g.vi[0]], snap[g.ci[1]].pos[g.vi[1]], snap[g.ci[2]].pos[g.vi[2]],
                          snap[g.ci[3]].pos[g.vi[3]] };
        const float mu = 0.25f * (snap[g.ci[0]].dynFriction[g.vi[0]] + snap[g.ci[1]].dynFriction[g.vi[1]] +
                                  snap[g.ci[2]].dynFriction[g.vi[2]] + snap[g.ci[3]].dynFriction[g.vi[3]]);
        float cnt[4];
        for (uint32_t j = 0; j < 4; ++j) cnt[j] = snap[g.ci[j]].delta[g.vi[j]].w;
        DbPart db0, db1;
        ContactPair cp;
        cp.maxPenBiasClamp = selectMax(h0.penBiasClamp, h1.penBiasClamp);
        db0.ref = pi.ref0;
        db1.ref = pi.ref1;
        if (isEE)
        {
            db0.bc = f4(g.w.x, g.w.y, 0, 0);
            db1.bc = f4(-g.w.z, -g.w.w, 0, 0);
            db0.vim = f4(cnt[0] * x[0].w, cnt[1] * x[1].w, 0, 0);
            db1.vim = f4(cnt[2] * x[2].w, cnt[3] * x[3].w, 0, 0);
        }
        else
        {
            db0.bc = f4(1, 0, 0, 0);
            db1.bc = f4(-g.w.y, -g.w.z, -g.w.w, 0);
            db0.vim = f4(cnt[0] * x[0].w, 0, 0, 0);
            db1.vim = f4(cnt[1] * x[1].w, cnt[2] * x[2].w, cnt[3] * x[3].w, 0);
        }
        cp.normal = neg(g.n);
        if (!isVelocityIteration)
        {
            const F4 pv[4] = { snap[g.ci[0]].prevPos[g.vi[0]], snap[g.ci[1]].prevPos[g.vi[1]], snap[g.ci[2]].prevPos[g.vi[2]],
                               snap[g.ci[3]].prevPos[g.vi[3]] };
            if (isEE)
            {
                const F4 d0 = add4(smul4(db0.bc.x, sub4(x[0], pv[0])), smul4(db0.bc.y, sub4(x[1], pv[1])));
                const F4 d1 = add4(smul4(db1.bc.x, sub4(x[2], pv[2])), smul4(db1.bc.y, sub4(x[3], pv[3])));
                db0.linDelta = xyz(d0);
                db1.linDelta = xyz(d1);
            }
            else
            {
                db0.linDelta = v3(x[0].x - pv[0].x, x[0].y - pv[0].y, x[0].z - pv[0].z);
                const F4 d1 = add4(add4(smul4(db1.bc.x, sub4(x[1], pv[1])), smul4(db1.bc.y, sub4(x[2], pv[2]))),
                                   smul4(db1.bc.z, sub4(x[3], pv[3])));
                db1.linDelta = xyz(d1);
            }
            cp.initPen = initPen(h0, h1, x[0], x[1], x[2], x[3], pv[0], pv[1], pv[2], pv[3], g.w, g.n, g.disp);
            db0.posDelta = db0.linDelta;
            db1.posDelta = db1.linDelta;
        }
        else
        {
            cp.initPen = dot(g.disp, g.n) - (h0.restOffset + h1.restOffset);
            const F4 v[4] = { snap[g.ci[0]].vel[g.vi[0]], snap[g.ci[1]].vel[g.vi[1]], snap[g.ci[2]].vel[g.vi[2]],
                              snap[g.ci[3]].vel[g.vi[3]] };
            if (isEE)
            {
                db0.linDelta = mul(xyz(add4(smul4(db0.bc.x, v[0]), smul4(db0.bc.y, v[1]))), kDt);
                db1.linDelta = mul(xyz(add4(smul4(db1.bc.x, v[2]), smul4(db1.bc.y, v[3]))), kDt);
            }
            else
            {
                db0.linDelta = mul(xyz(v[0]), kDt);
                db1.linDelta = mul(xyz(add4(add4(smul4(db1.bc.x, v[1]), smul4(db1.bc.y, v[2])), smul4(db1.bc.z, v[3]))), kDt);
            }
        }
        SolveOut so;
        if (solveDbDbContact(db0, db1, cp, so, 0.0f, 0.0f, mu, kDt, false, isVelocityIteration))
        {
            ++nApplied;
            V3 delta;
            so.computeDelta(cp, delta, kDt);
            if (isEE)
            {
                db0.writeClothEdge(C.cloth[g.ci[0]], g.vi[0], g.vi[1], delta);
                db1.writeClothEdge(C.cloth[g.ci[2]], g.vi[2], g.vi[3], neg(delta));
            }
            else
            {
                db0.writeCloth(C.cloth[refId(pi.ref0)], delta);
                db1.writeCloth(C.cloth[refId(pi.ref1)], neg(delta));
            }
        }
    }
    bad = 0;
    for (uint32_t ci = 0; ci < 3; ci += 2)
        cmpVec("solve mDeltaPos", C.readDelta(ci), C.cloth[ci].delta, bad);
    PXG_CHECK_MSG(bad == 0, "cloth_solveClothClothDelta%sLaunch (%s): %u mismatches", isEE ? "EE" : "VT",
                  isVelocityIteration ? "vel" : "pos", bad);
    PXG_CHECK_MSG(nApplied > 0, "coverage: no %s contact applied", isEE ? "EE" : "VT");
}

PXG_TEST(femClothSolve512ClothContactVTPos) { runCcQuerySolve(false, false, 81u); }
PXG_TEST(femClothSolve512ClothContactVTVel) { runCcQuerySolve(false, true, 82u); }
PXG_TEST(femClothSolve512ClothContactEEPos) { runCcQuerySolve(true, false, 83u); }
PXG_TEST(femClothSolve512ClothContactEEVel) { runCcQuerySolve(true, true, 84u); }

// ===========================================================================
// cloth_applyExternalDeltasLaunch
// ===========================================================================
struct ApplyPush { uint64_t clothes, activeClothes; float dt; uint32_t isVelocityIteration; };
static_assert(sizeof(ApplyPush) == 24, "cloth_applyExternalDeltasLaunch blob");

static void runApplyExternalDeltas(bool isVelocityIteration, uint32_t seed)
{
    PxgSlangComputeDevice* dev = testDevice();
    ClothWorld C;
    buildClothHost(C, seed);
    Rng rng{ seed * 3u + 1u };
    for (uint32_t ci = 0; ci < 3; ci += 2)
        for (uint32_t v = 0; v < NV; ++v)
        {
            const float w = (v % 4u == 0u) ? 0.0f : float(1u + rng.next() % 3u) * ((v % 9u == 2u) ? 0.37f : 1.0f);
            C.cloth[ci].delta[v] = F4{ rng.uni(-0.02f, 0.02f), rng.uni(-0.02f, 0.02f), rng.uni(-0.02f, 0.02f), w };
            if (v % 11u == 6u) C.cloth[ci].pos[v].w = -0.0f;   // a -0 inverse mass: x + 0 turns it +0
        }
    uploadClothWorld(C, dev);
    std::vector<uint32_t> active = { 2u, 0u };
    ApplyPush p = { C.pool, C.addr(C.make(active)), kDt, isVelocityIteration ? 0x77777701u : 0x77777700u };
    Kernel k("femClothRigidAttachBda.spv", "cloth_applyExternalDeltasLaunch", {}, sizeof(ApplyPush));
    k.dispatch((NV + 63u) / 64u, {}, p, 2u);
    dev->flush();

    uint32_t bad = 0, nApplied = 0;
    const float invDt = 1.0f / kDt;
    for (uint32_t ci = 0; ci < 3; ci += 2)
    {
        HostCloth& h = C.cloth[ci];
        for (uint32_t v = 0; v < NV; ++v)
        {
            F4 d = h.delta[v];
            if (d.w == 0.0f) continue;
            ++nApplied;
            const float scale = 1.0f / d.w;
            d.x *= scale; d.y *= scale; d.z *= scale; d.w = 0.0f;
            F4& vel = h.vel[v];
            vel = F4{ vel.x + d.x * invDt, vel.y + d.y * invDt, vel.z + d.z * invDt, vel.w + d.w * invDt };
            if (!isVelocityIteration)
            {
                F4& pos = h.pos[v];
                pos = F4{ pos.x + d.x, pos.y + d.y, pos.z + d.z, pos.w + d.w };
                F4& acc = h.accDelta[v];
                acc = F4{ acc.x + d.x, acc.y + d.y, acc.z + d.z, acc.w + d.w };
            }
            h.delta[v] = F4{ 0, 0, 0, 0 };
        }
        cmpVec("mVelocity_InvMass", C.readVel(ci), h.vel, bad);
        cmpVec("mPosition_InvMass", C.readPos(ci), h.pos, bad);
        cmpVec("mAccumulatedDeltaPos", C.readAcc(ci), h.accDelta, bad);
        cmpVec("mDeltaPos", C.readDelta(ci), h.delta, bad);
    }
    PXG_CHECK_MSG(bad == 0, "cloth_applyExternalDeltasLaunch (%s): %u mismatches", isVelocityIteration ? "vel" : "pos", bad);
    PXG_CHECK_MSG(nApplied > 0, "coverage: nothing applied");
}

PXG_TEST(femClothSolve512ApplyExternalDeltasPos) { runApplyExternalDeltas(false, 91u); }
PXG_TEST(femClothSolve512ApplyExternalDeltasVel) { runApplyExternalDeltas(true, 92u); }

} // namespace
