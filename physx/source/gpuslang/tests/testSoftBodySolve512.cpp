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

// testSoftBodySolve512.cpp -- the PhysX 5.12 softbody (deformable volume) constraint
// kernels of softBody.cu / softBodyGM.cu, checked word for word against the host
// transcription of the 5.12 CUDA (pxgSoftBodySolveRef.h on top of pxgParticleSolveRef.h):
//
//   softbody-rigid contacts   sb_rigidContactPrepareLaunch, sb_queryRigidSoftContact-
//                             ReferenceCountLaunch, sb_solveRigidSoftCollisionLaunch
//   softbody-softbody         sb_softbodyContactPrepareLaunch, sb_querySSContact-
//                             ReferenceCountLaunch, sb_solveOutputSSDeltaVLaunch
//   softbody-cloth            sb_clothContactPrepareLaunch, sb_querySCContact-
//                             ReferenceCountLaunch, sb_solveOutputSCDeltaVLaunch
//   softbody-particle         sb_querySBParticleContactReferenceCountLaunch,
//                             sb_solveOutputSPDeltaVLaunch, sb_solveOutputParticleDeltaVLaunch
//                             (the prepare is testSoftBodyPsPrepare)
//   softbody-rigid attach     sb_rigidAttachmentPrepareLaunch, sb_queryRigidSoftAttachment-
//                             ReferenceCountLaunch, sb_solveRigidSoftAttachmentLaunch
//   deformable attach         femAttachmentPrepareLaunch, sb_querySoftBodyAttachment-
//                             ReferenceCountLaunch, sb_solveOutputSoftBodyAttachmentDeltaVLaunch,
//                             sb_queryClothAttachmentReferenceCountLaunch,
//                             sb_solveOutputClothAttachmentDeltaVLaunch
//   finalize of an iteration  sb_gm_applyExternalDeltasLaunch (position / velocity iteration)
//
// THE SCENE.  Descriptor pools with a zero DECOY in the middle (softbodies 0 and 2 are
// real, cloth 1 is real), so a kernel that confuses ids reads zeros.  Every tetrahedron and
// triangle has its own vertices, and the contacts / attachments are spread so that no vertex
// receives more than two scatters per dispatch: the float atomics then commute (see the
// oracle header).  Vertices with inverse mass 0, barycentric weights within 1e-3 of zero
// (the reference-count gate), INVALID pairs, pre-ACTIVE pairs, two blocks of 32 with a
// partial second, both PGS and TGS and both position and velocity iterations are covered.
// The rigid side is pxgParticleSolveWorld's: static, dynamic, kinematic and articulation-
// link participants (with rotated frames for the attachment anchors).

#include "pxgTestFramework.h"
#include "pxgParticleSolveWorld.h"
#include "pxgSoftBodySolveRef.h"
#include "host_defines.h"
#include "vector_functions.h"
#include "PxgSoftBody.h"
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

static const float kDt = 1.0f / 60.0f;
static const uint32_t GRID = 1024u;   // PxgSoftBodyKernelGridDim::SB_UPDATEROTATION

// Block strides / offsets (PxgDeformableConstraints.h, 5.12; asserted against the headers
// in testDeformableRef512).
static const uint32_t RDBC_STRIDE = 2688u, RDBC_raXn_resp = 0u, RDBC_raXnF0_resp = 512u, RDBC_raXnF1_resp = 1024u,
                      RDBC_normal_errorW = 1536u, RDBC_fricTan0_invMass0 = 2048u, RDBC_maxPenBiasClamp = 2560u;
static const uint32_t DBDBC_STRIDE = 640u, DBDBC_normal_pen = 0u, DBDBC_maxPenBiasClamp = 512u;
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

// ===========================================================================
// The deformable world
// ===========================================================================
struct SbWorld
{
    PxgSlangComputeDevice* dev = nullptr;
    std::vector<PxgBufferHandle> owned;

    HostSoftBody sb[3];     // 0 and 2 real, 1 the decoy
    HostCloth cloth[2];     // 1 real, 0 the decoy
    std::vector<float> volumeFriction;    // PxsDeformableVolumeMaterialData dynamicFriction per material
    std::vector<float> surfaceFriction;   // PxsDeformableSurfaceMaterialData dynamicFriction per material

    // device
    PxgBufferHandle bSbDelta[3] = { 0, 0, 0 }, bSbPos[3] = { 0, 0, 0 }, bSbVel[3] = { 0, 0, 0 },
                    bSbDeltaPos[3] = { 0, 0, 0 };
    PxgBufferHandle bClothDelta[2] = { 0, 0 };
    uint64_t sbPool = 0, clothPool = 0, volMaterials = 0, surfMaterials = 0;

    ~SbWorld() { for (PxgBufferHandle h : owned) dev->destroyBuffer(h); }

    template <typename T> PxgBufferHandle make(const std::vector<T>& v)
    {
        PxgBufferHandle h = createBufferFrom(dev, v);
        owned.push_back(h);
        return h;
    }
    uint64_t addr(PxgBufferHandle h) const { return dev->getBufferDeviceAddress(h); }

    void uploadSoftBodyDelta(uint32_t s) { dev->upload(bSbDelta[s], sb[s].delta.data(), sb[s].delta.size() * sizeof(F4)); }
    void uploadClothDelta() { dev->upload(bClothDelta[1], cloth[1].delta.data(), cloth[1].delta.size() * sizeof(F4)); }
    std::vector<F4> readSoftBodyDelta(uint32_t s) const { return downloadBuffer<F4>(dev, bSbDelta[s], sb[s].delta.size()); }
    std::vector<F4> readClothDelta() const { return downloadBuffer<F4>(dev, bClothDelta[1], cloth[1].delta.size()); }
};

static F4 randomBary(Rng& rng, bool tiny, uint32_t n)
{
    float b[4] = { rng.uni(0.1f, 1.0f), rng.uni(0.1f, 1.0f), rng.uni(0.1f, 1.0f), (n == 4) ? rng.uni(0.1f, 1.0f) : 0.0f };
    if (tiny)
        b[rng.next() % n] = 5.0e-4f;   // within the 1e-3 reference-count gate
    const float s = b[0] + b[1] + b[2] + b[3];
    return F4{ b[0] / s, b[1] / s, b[2] / s, (n == 4) ? b[3] / s : 0.0f };
}

// NT tetrahedra per softbody, each with its own four vertices; NTRI triangles with their
// own three vertices plus NLOOSE lone cloth vertices.
static const uint32_t NT = 40u, NTRI = 24u, NLOOSE = 16u;

static void buildSbWorld(SbWorld& S, PxgSlangComputeDevice* dev, uint32_t seed)
{
    S.dev = dev;
    Rng rng{ seed | 1u };
    S.volumeFriction = { 0.3f, 0.65f, 0.1f };
    S.surfaceFriction = { 0.45f, 0.2f };

    for (uint32_t s = 0; s < 3; s += 2)
    {
        HostSoftBody& h = S.sb[s];
        for (uint32_t t = 0; t < NT; ++t)
        {
            h.tets.push_back(U4{ 4 * t + 2, 4 * t + 0, 4 * t + 3, 4 * t + 1 });   // a permuted corner order
            h.matIdx.push_back(uint16_t(rng.next() % 3u));
        }
        for (uint32_t v = 0; v < 4 * NT; ++v)
        {
            const float invMass = (v % 7u == 5u) ? 0.0f : rng.uni(0.4f, 2.0f);
            h.pos.push_back(F4{ rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), invMass });
            h.vel.push_back(F4{ rng.uni(-2.f, 2.f), rng.uni(-2.f, 2.f), rng.uni(-2.f, 2.f), invMass });
            h.deltaPos.push_back(F4{ rng.uni(-0.03f, 0.03f), rng.uni(-0.03f, 0.03f), rng.uni(-0.03f, 0.03f), invMass });
            h.delta.push_back(F4{ 0.0f, 0.0f, 0.0f, 0.0f });
        }
        h.restOffset = (s == 0) ? 0.02f : 0.035f;
        h.penBiasClamp = (s == 0) ? -1.5f : -0.4f;
    }
    {
        HostCloth& c = S.cloth[1];
        for (uint32_t t = 0; t < NTRI; ++t)
        {
            c.tris.push_back(U4{ 3 * t + 1, 3 * t + 2, 3 * t + 0, 0x5u });   // .w: owned-edge bits, not an index
            c.matIdx.push_back(uint16_t(rng.next() % 2u));
        }
        for (uint32_t v = 0; v < 3 * NTRI + NLOOSE; ++v)
        {
            const float invMass = (v % 5u == 3u) ? 0.0f : rng.uni(0.5f, 3.0f);
            c.pos.push_back(F4{ rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), invMass });
            c.vel.push_back(F4{ rng.uni(-2.f, 2.f), rng.uni(-2.f, 2.f), rng.uni(-2.f, 2.f), invMass });
            c.accDelta.push_back(F4{ rng.uni(-0.03f, 0.03f), rng.uni(-0.03f, 0.03f), rng.uni(-0.03f, 0.03f), invMass });
            c.delta.push_back(F4{ 0.0f, 0.0f, 0.0f, 0.0f });
            c.dynFriction.push_back(rng.uni(0.0f, 0.9f));
        }
        c.restOffset = 0.015f;
        c.penBiasClamp = -0.8f;
    }

    // ---- device ----
    std::vector<physx::PxgSoftBody> sbDesc(3);
    std::memset(sbDesc.data(), 0, sbDesc.size() * sizeof(physx::PxgSoftBody));
    for (uint32_t s = 0; s < 3; s += 2)
    {
        HostSoftBody& h = S.sb[s];
        std::vector<uint32_t> mi((h.matIdx.size() + 1) / 2 + 2, 0u);
        std::memcpy(mi.data(), h.matIdx.data(), h.matIdx.size() * 2);
        S.bSbPos[s] = S.make(h.pos);
        S.bSbVel[s] = S.make(h.vel);
        S.bSbDeltaPos[s] = S.make(h.deltaPos);
        S.bSbDelta[s] = S.make(h.delta);
        physx::PxgSoftBody& d = sbDesc[s];
        d.mSimTetIndices = reinterpret_cast<uint4*>(size_t(S.addr(S.make(h.tets))));
        d.mSimPosition_InvMass = reinterpret_cast<float4*>(size_t(S.addr(S.bSbPos[s])));
        d.mSimVelocity_InvMass = reinterpret_cast<float4*>(size_t(S.addr(S.bSbVel[s])));
        d.mSimDeltaPos = reinterpret_cast<float4*>(size_t(S.addr(S.bSbDeltaPos[s])));
        d.mSimDelta = reinterpret_cast<float4*>(size_t(S.addr(S.bSbDelta[s])));
        d.mMaterialIndices = reinterpret_cast<physx::PxU16*>(size_t(S.addr(S.make(mi))));
        d.mRestOffset = h.restOffset;
        d.mPenBiasClamp = h.penBiasClamp;
        d.mNumVertsGM = uint32_t(h.pos.size());
        d.mNumTetsGM = uint32_t(h.tets.size());
    }
    S.sbPool = S.addr(S.make(sbDesc));

    std::vector<physx::PxgFEMCloth> clothDesc(2);
    std::memset(clothDesc.data(), 0, clothDesc.size() * sizeof(physx::PxgFEMCloth));
    {
        HostCloth& c = S.cloth[1];
        std::vector<uint32_t> mi((c.matIdx.size() + 1) / 2 + 2, 0u);
        std::memcpy(mi.data(), c.matIdx.data(), c.matIdx.size() * 2);
        S.bClothDelta[1] = S.make(c.delta);
        physx::PxgFEMCloth& d = clothDesc[1];
        d.mTriangleVertexIndices_ownedEdges = reinterpret_cast<uint4*>(size_t(S.addr(S.make(c.tris))));
        d.mPosition_InvMass = reinterpret_cast<float4*>(size_t(S.addr(S.make(c.pos))));
        d.mVelocity_InvMass = reinterpret_cast<float4*>(size_t(S.addr(S.make(c.vel))));
        d.mAccumulatedDeltaPos = reinterpret_cast<float4*>(size_t(S.addr(S.make(c.accDelta))));
        d.mDeltaPos = reinterpret_cast<float4*>(size_t(S.addr(S.bClothDelta[1])));
        d.mDynamicFrictions = reinterpret_cast<float*>(size_t(S.addr(S.make(c.dynFriction))));
        d.mMaterialIndices = reinterpret_cast<physx::PxU16*>(size_t(S.addr(S.make(mi))));
        d.mRestOffset = c.restOffset;
        d.mPenBiasClamp = c.penBiasClamp;
        d.mNbVerts = uint32_t(c.pos.size());
        d.mNbTriangles = uint32_t(c.tris.size());
    }
    S.clothPool = S.addr(S.make(clothDesc));

    // PxsDeformableVolumeMaterialData / PxsDeformableSurfaceMaterialData: 32 B, dynamicFriction @8.
    std::vector<float> vm(8 * S.volumeFriction.size(), 7.0f), sm(8 * S.surfaceFriction.size(), 7.0f);
    for (size_t m = 0; m < S.volumeFriction.size(); ++m) vm[8 * m + 2] = S.volumeFriction[m];
    for (size_t m = 0; m < S.surfaceFriction.size(); ++m) sm[8 * m + 2] = S.surfaceFriction[m];
    S.volMaterials = S.addr(S.make(vm));
    S.surfMaterials = S.addr(S.make(sm));
}

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

// ===========================================================================
// Softbody-rigid contacts
// ===========================================================================
struct RsContact
{
    PairInfo pair;     // ref0 = rigid node, ref1 = tet ref of softbody 0 or 2
    F4 point_rest;     // contacts_restW
    F4 normalPen;
    F4 bary;
    uint16_t rigidMat;
};

struct RdbLane { F4 raXn_resp, raXnF0_resp, raXnF1_resp, normal_errorW, fricTan0_invMass0; float maxPenBiasClamp; };

// prepareRigidDbContact (deformableUtils.cuh), on the World's rigid data.
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

// 45 contacts, tets visited at most twice, a few INVALID pairs.
static std::vector<RsContact> makeRsContacts(const SbWorld& S, uint32_t seed)
{
    Rng rng{ seed * 7u + 3u };
    const uint64_t rigids[6] = { staticNode(), makeRigidNode(4), makeRigidNode(6), makeRigidNode(8),
                                 makeArticulationNode(10, 2), makeArticulationNode(10, 1) };
    std::vector<RsContact> cs;
    const uint32_t NC = 45u;
    for (uint32_t c = 0; c < NC; ++c)
    {
        RsContact rc;
        const uint32_t s = (c % 3u == 1u) ? 2u : 0u;
        const uint32_t tet = (c * 7u + s) % NT;
        rc.pair.ref0 = rigids[rng.next() % 6u];
        rc.pair.ref1 = (c % 11u == 4u) ? (encodeRef(s, 0, 0) | REF_INVALID) : encodeRef(s, REF_eTETRAHEDRON, tet);
        V3 n = v3(rng.uni(-0.5f, 0.5f), 1.0f, rng.uni(-0.5f, 0.5f));
        if (c % 9u == 0u) n = v3(0.0f, -1.0f, 0.0f);
        normalize(n);
        rc.normalPen = F4{ n.x, n.y, n.z, rng.uni(-0.06f, 0.04f) };
        rc.point_rest = F4{ rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), S.sb[s].restOffset };
        rc.bary = randomBary(rng, c % 5u == 2u, 4);
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

struct RsPrepPush
{
    uint64_t softbodies, contacts_restW, normalPens, barycentrics, contactInfos, numContacts, contactBlocks, preDesc,
        prepareDesc, rigidAppliedForces;
    float invDt; uint32_t pad84;
    uint64_t sharedDesc;
    uint32_t isTGS, pad100;
};
static_assert(sizeof(RsPrepPush) == 104, "sb_rigidContactPrepareLaunch blob");

PXG_TEST(softBodySolve512RigidContactPrepare)
{
    PxgSlangComputeDevice* dev = testDevice();
    World W;
    buildWorld(W, dev, 0x5150u);
    SbWorld S;
    buildSbWorld(S, dev, 0x77u);
    const std::vector<RsContact> cs = makeRsContacts(S, 1u);
    const uint32_t NC = uint32_t(cs.size());

    std::vector<F4> pr, np, bc;
    std::vector<PairInfo> pairs;
    for (const RsContact& c : cs) { pr.push_back(c.point_rest); np.push_back(c.normalPen); bc.push_back(c.bary); pairs.push_back(c.pair); }
    std::vector<uint32_t> num = { NC };
    const uint32_t NB = (NC + 31u) / 32u + 1u;
    std::vector<uint32_t> blocksInit(NB * RDBC_STRIDE / 4u, POISON);
    std::vector<F2> appliedInit(NC + 8u, F2{ 9.0f, -9.0f });

    PxgBufferHandle bBlocks = W.make(blocksInit), bApplied = W.make(appliedInit);
    RsPrepPush p = {};
    p.softbodies = S.sbPool;
    p.contacts_restW = W.addr(W.make(pr));
    p.normalPens = W.addr(W.make(np));
    p.barycentrics = W.addr(W.make(bc));
    p.contactInfos = W.addr(W.make(pairs));
    p.numContacts = W.addr(W.make(num));
    p.contactBlocks = W.addr(bBlocks);
    p.preDesc = W.prePrepDesc;
    p.prepareDesc = W.prepDesc;
    p.rigidAppliedForces = W.addr(bApplied);
    p.invDt = 1.0f / kDt;
    p.sharedDesc = W.sharedDesc;
    p.isTGS = 1u;

    Kernel k("softBodyRigidContactBda.spv", "sb_rigidContactPrepareLaunch", {}, sizeof(RsPrepPush));
    k.dispatch(GRID, {}, p);
    dev->flush();

    const std::vector<uint8_t> gotBlocks = downloadBuffer<uint8_t>(dev, bBlocks, NB * RDBC_STRIDE);
    const std::vector<F2> gotApplied = downloadBuffer<F2>(dev, bApplied, NC + 8u);

    uint32_t bad = 0, kinds[4] = { 0, 0, 0, 0 }, invalid = 0;
    for (uint32_t w = 0; w < NC; ++w)
    {
        const RsContact& c = cs[w];
        cmpF4("applied", w, F4{ gotApplied[w].x, gotApplied[w].y, 0, 0 }, F4{ 0, 0, 0, 0 }, bad);
        if (!pairValid(c.pair.ref1))
        {
            ++invalid;
            const RdbLane g = readRdbLane(gotBlocks, w);
            if (std::memcmp(&g.normal_errorW, "\xC5\xC5\xC5\xC5", 4) != 0) { ++bad; std::printf("    invalid lane %u written\n", w); }
            continue;
        }
        const HostSoftBody& h = S.sb[refId(c.pair.ref1)];
        const F4 d = projectTet(h.deltaPos, h.tets[refFeature(c.pair.ref1)], c.bary);
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
        cmpF4("applied tail", w, F4{ gotApplied[w].x, gotApplied[w].y, 0, 0 }, F4{ 9.0f, -9.0f, 0, 0 }, bad);
    PXG_CHECK_MSG(bad == 0, "sb_rigidContactPrepareLaunch: %u mismatches", bad);
    PXG_CHECK_MSG(kinds[0] && kinds[1] && kinds[2] && kinds[3] && invalid,
                  "coverage: static %u kinematic %u dynamic %u link %u invalid %u", kinds[0], kinds[1], kinds[2],
                  kinds[3], invalid);
}

struct RsQueryPush
{
    uint64_t softbodies, contactInfos, barycentrics, contactBlocks, numContacts, prePrepDesc, solverCoreDesc,
        artiCoreDesc, solverBodyVelPool;
    float dt; uint32_t pad76;
    uint64_t appliedForces, rigidBodyRefCounts;
    uint32_t flags, pad100;   // byte0 isTGS, byte1 isVelocityIteration
};
static_assert(sizeof(RsQueryPush) == 104, "sb_queryRigidSoftContactReferenceCountLaunch blob");

struct RsSolvePush
{
    uint64_t softbodies, contactInfos, contactRigidMaterials, barycentrics, contactBlocks, numContacts, prePrepDesc,
        solverCoreDesc, artiCoreDesc, solverBodyVelPool, rigidDeltaVel, appliedForces, rigidBodyRefCounts;
    float dt; uint32_t pad108;
    uint64_t materials, rigidBodyMaterials;
    uint32_t flags, pad132;   // byte0 isTGS, byte1 isVelocityIteration
};
static_assert(sizeof(RsSolvePush) == 136, "sb_solveRigidSoftCollisionLaunch blob");

// Query then solve, each checked against the oracle, in one of the four modes.
static void runRsQuerySolve(bool isTGS, bool isVelocityIteration, uint32_t seed)
{
    PxgSlangComputeDevice* dev = testDevice();
    World W;
    buildWorld(W, dev, seed);
    SbWorld S;
    buildSbWorld(S, dev, seed + 1u);
    std::vector<RsContact> cs = makeRsContacts(S, seed);
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
        const RsContact& c = cs[w];
        bc.push_back(c.bary);
        pairs.push_back(c.pair);
        rmat.push_back(c.rigidMat);
        if (!pairValid(c.pair.ref1)) continue;
        const HostSoftBody& h = S.sb[refId(c.pair.ref1)];
        const F4 d = projectTet(h.deltaPos, h.tets[refFeature(c.pair.ref1)], c.bary);
        lanes[w] = prepareRigidDbContactRef(W, v3(-c.normalPen.x, -c.normalPen.y, -c.normalPen.z), xyz(c.point_rest),
                                            c.normalPen.w - c.point_rest.w, xyz(d), c.pair.ref0,
                                            h.penBiasClamp);
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
        RsQueryPush q = {};
        q.softbodies = S.sbPool; q.contactInfos = W.addr(bPairs); q.barycentrics = aBc; q.contactBlocks = aBlocks;
        q.numContacts = aNum; q.prePrepDesc = W.prePrepDesc; q.solverCoreDesc = W.coreDesc; q.artiCoreDesc = W.artiCoreDesc;
        q.solverBodyVelPool = W.velPoolAddr; q.dt = kDt; q.appliedForces = W.addr(bApplied);
        q.rigidBodyRefCounts = W.addr(bRef);
        q.flags = (isTGS ? 1u : 0u) | (isVelocityIteration ? 0x100u : 0u);
        Kernel k("softBodyRigidContactBda.spv", "sb_queryRigidSoftContactReferenceCountLaunch", {}, sizeof(RsQueryPush));
        k.dispatch(GRID, {}, q);
        dev->flush();
    }

    // oracle query
    std::vector<uint32_t> wantRef = refCounts;
    std::vector<PairInfo> wantPairs = pairs;
    uint32_t activated = 0, latched = 0, stayedInactive = 0;
    for (uint32_t w = 0; w < NC; ++w)
    {
        const PairInfo& pi = pairs[w];
        if (!pairValid(pi.ref1)) continue;
        const bool wasActive = pairActive(pi.ref1);
        HostSoftBody& h = S.sb[refId(pi.ref1)];
        const int g = W.gid(pi.ref0);
        const float rigidInvMass = lanes[w].fricTan0_invMass0.w;
        DbPart db;
        db.readSoftBody(h, pi.ref1, cs[w].bary, nullptr, true, kDt, isVelocityIteration);
        bool isActive = false;
        if (wasActive) { isActive = true; ++latched; }
        else
        {
            RigidPart rigid;
            rigid.readBodyProperties(g, rigidInvMass, nullptr);
            BlockLane bl;
            bl.raXn_resp = lanes[w].raXn_resp; bl.raXnF0_resp = lanes[w].raXnF0_resp; bl.raXnF1_resp = lanes[w].raXnF1_resp;
            bl.normal_errorW = lanes[w].normal_errorW; bl.fricTan0_invMass0 = lanes[w].fricTan0_invMass0;
            rigid.readContactPrep(bl);
            rigid.readVelocity(reader, pi.ref0, isTGS);
            ContactPair cp;
            cp.normal = xyz(lanes[w].normal_errorW); cp.initPen = lanes[w].normal_errorW.w;
            cp.tangent0 = xyz(lanes[w].fricTan0_invMass0); cp.maxPenBiasClamp = lanes[w].maxPenBiasClamp;
            SolveOut so;
            isActive = solveRigidDbContactDb(rigid, db, cp, so, applied[w].x, applied[w].y, kDt, false, true,
                                             isVelocityIteration);
            if (isActive) { wantPairs[w].ref1 = setActive(pi.ref1, true); ++activated; }
            else ++stayedInactive;
        }
        if (isActive)
        {
            bumpSoftBody(h, pi.ref1, db.bc);
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
        for (uint32_t s = 0; s < 3; s += 2)
            cmpVec("query mSimDelta", S.readSoftBodyDelta(s), S.sb[s].delta, bad);
    }
    PXG_CHECK_MSG(bad == 0, "sb_queryRigidSoftContactReferenceCountLaunch (%s %s): %u mismatches",
                  isTGS ? "TGS" : "PGS", isVelocityIteration ? "vel" : "pos", bad);
    PXG_CHECK_MSG(activated > 0 && latched > 0, "coverage: activated %u latched %u inactive %u", activated, latched,
                  stayedInactive);

    // ---------------- solve (on the query's output) ----------------
    std::vector<F4> deltaVelInit(2 * NC, F4{ f32(POISON), f32(POISON), f32(POISON), f32(POISON) });
    std::vector<uint32_t> rmatWords((rmat.size() + 1) / 2 + 2, 0u);
    std::memcpy(rmatWords.data(), rmat.data(), rmat.size() * 2);
    PxgBufferHandle bDeltaVel = W.make(deltaVelInit);
    {
        RsSolvePush sp = {};
        sp.softbodies = S.sbPool; sp.contactInfos = W.addr(bPairs); sp.contactRigidMaterials = W.addr(W.make(rmatWords));
        sp.barycentrics = aBc; sp.contactBlocks = aBlocks; sp.numContacts = aNum; sp.prePrepDesc = W.prePrepDesc;
        sp.solverCoreDesc = W.coreDesc; sp.artiCoreDesc = W.artiCoreDesc; sp.solverBodyVelPool = W.velPoolAddr;
        sp.rigidDeltaVel = W.addr(bDeltaVel); sp.appliedForces = W.addr(bApplied); sp.rigidBodyRefCounts = W.addr(bRef);
        sp.dt = kDt; sp.materials = S.volMaterials; sp.rigidBodyMaterials = W.addr(W.make(rigidMaterialTable()));
        sp.flags = (isTGS ? 1u : 0u) | (isVelocityIteration ? 0x100u : 0u);
        Kernel k("softBodyRigidContactBda.spv", "sb_solveRigidSoftCollisionLaunch", {}, sizeof(RsSolvePush));
        k.dispatch(GRID, {}, sp);
        dev->flush();
    }

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
            HostSoftBody& h = S.sb[refId(pi.ref1)];
            rigid.readBodyProperties(g, rigidInvMass, wantRef.data());
            rigid.friction = kRigidMats[cs[w].rigidMat].friction;
            rigid.fricCombineMode = kRigidMats[cs[w].rigidMat].combine;
            DbPart db;
            db.readSoftBody(h, pi.ref1, cs[w].bary, S.volumeFriction.data(), false, kDt, isVelocityIteration);
            BlockLane bl;
            bl.raXn_resp = lanes[w].raXn_resp; bl.raXnF0_resp = lanes[w].raXnF0_resp; bl.raXnF1_resp = lanes[w].raXnF1_resp;
            bl.normal_errorW = lanes[w].normal_errorW; bl.fricTan0_invMass0 = lanes[w].fricTan0_invMass0;
            rigid.readContactPrep(bl);
            rigid.readVelocity(reader, pi.ref0, isTGS);
            cp.normal = xyz(lanes[w].normal_errorW); cp.initPen = lanes[w].normal_errorW.w;
            cp.tangent0 = xyz(lanes[w].fricTan0_invMass0); cp.maxPenBiasClamp = lanes[w].maxPenBiasClamp;
            const bool hasApplied = solveRigidDbContactDb(rigid, db, cp, so, applied[w].x, applied[w].y, kDt, true, false,
                                                          isVelocityIteration);
            if (hasApplied)
            {
                ++appliedCount;
                V3 delta;
                so.computeDelta(cp, delta, kDt);
                wantApplied[w] = F2{ so.accumulatedDeltaLambdaN, applied[w].y + so.deltaLambdaT };
                db.writeSoftBody(h, delta);
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
    for (uint32_t s = 0; s < 3; s += 2)
        cmpVec("solve mSimDelta", S.readSoftBodyDelta(s), S.sb[s].delta, bad);
    PXG_CHECK_MSG(bad == 0, "sb_solveRigidSoftCollisionLaunch (%s %s): %u mismatches", isTGS ? "TGS" : "PGS",
                  isVelocityIteration ? "vel" : "pos", bad);
    PXG_CHECK_MSG(appliedCount > 0, "coverage: no contact applied an impulse");
}

PXG_TEST(softBodySolve512RigidContactPGS) { runRsQuerySolve(false, false, 11u); }
PXG_TEST(softBodySolve512RigidContactTGS) { runRsQuerySolve(true, false, 12u); }
PXG_TEST(softBodySolve512RigidContactPGSVel) { runRsQuerySolve(false, true, 13u); }
PXG_TEST(softBodySolve512RigidContactTGSVel) { runRsQuerySolve(true, true, 14u); }

// ===========================================================================
// Softbody-softbody contacts
// ===========================================================================
struct DbContact
{
    PairInfo pair;
    F4 normalPen, bary0, bary1;
};

static std::vector<DbContact> makeSsContacts(uint32_t n, uint32_t seed)
{
    Rng rng{ seed * 13u + 5u };
    std::vector<DbContact> cs;
    for (uint32_t c = 0; c < n; ++c)
    {
        DbContact d;
        d.pair.ref0 = encodeRef(0, REF_eTETRAHEDRON, (c * 3u) % NT);
        d.pair.ref1 = (c % 13u == 6u) ? (encodeRef(2, 0, 0) | REF_INVALID) : encodeRef(2, REF_eTETRAHEDRON, (c * 11u) % NT);
        if (c % 5u == 1u) d.pair.ref1 = setActive(d.pair.ref1, true);   // stale ACTIVE bits the query rewrites
        V3 nrm = v3(rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f));
        normalize(nrm);
        // Every fourth contact separated well beyond the rest distance so the query sees inactive pairs.
        const float pen = rng.uni(-0.05f, 0.05f);
        d.normalPen = F4{ nrm.x, nrm.y, nrm.z, (c % 4u == 3u) ? pen + 0.5f : pen };
        d.bary0 = randomBary(rng, c % 6u == 2u, 4);
        d.bary1 = randomBary(rng, c % 7u == 3u, 4);
        cs.push_back(d);
    }
    return cs;
}

struct SsPrepPush
{
    uint64_t softbodies, contacts, normalPens, barycentrics0, barycentrics1, contactInfos, numContacts, contactBlocks,
        softBodyAppliedForces;
    uint32_t maxContacts, pad76;
};
static_assert(sizeof(SsPrepPush) == 80, "sb_softbodyContactPrepareLaunch blob");

static void dbdbLaneSet(std::vector<uint8_t>& blocks, uint32_t w, F4 np, float clamp)
{
    const size_t base = size_t(w / 32u) * DBDBC_STRIDE;
    putAt(blocks, base + DBDBC_normal_pen + (w & 31u) * 16, np);
    putAt(blocks, base + DBDBC_maxPenBiasClamp + (w & 31u) * 4, clamp);
}

// sb_softbodyContactPrepareLaunch's lane.
static void ssPrepareRef(const SbWorld& S, const DbContact& c, F4& np, float& clamp)
{
    const HostSoftBody& a = S.sb[refId(c.pair.ref0)];
    const HostSoftBody& b = S.sb[refId(c.pair.ref1)];
    const float restDistance = a.restOffset + b.restOffset;
    const F4 d0 = projectTet(a.deltaPos, a.tets[refFeature(c.pair.ref0)], c.bary0);
    const F4 d1 = projectTet(b.deltaPos, b.tets[refFeature(c.pair.ref1)], c.bary1);
    const V3 n = v3(-c.normalPen.x, -c.normalPen.y, -c.normalPen.z);
    const V3 rel = v3(d0.x - d1.x, d0.y - d1.y, d0.z - d1.z);
    const float pen = (c.normalPen.w - restDistance) + dot(rel, n);
    np = F4{ n.x, n.y, n.z, pen };
    clamp = selectMax(a.penBiasClamp, b.penBiasClamp);
}

PXG_TEST(softBodySolve512SoftContactPrepare)
{
    PxgSlangComputeDevice* dev = testDevice();
    SbWorld S;
    buildSbWorld(S, dev, 0x31u);
    const uint32_t N = 50u, MAXC = 45u;   // the count is clamped to maxContacts and written back
    const std::vector<DbContact> cs = makeSsContacts(N, 2u);
    std::vector<F4> np, b0, b1, dead(N, F4{ 1e30f, 1e30f, 1e30f, 1e30f });
    std::vector<PairInfo> pairs;
    for (const DbContact& c : cs) { np.push_back(c.normalPen); b0.push_back(c.bary0); b1.push_back(c.bary1); pairs.push_back(c.pair); }
    std::vector<uint32_t> num = { N };
    const uint32_t NB = (N + 31u) / 32u;
    std::vector<uint32_t> blocksInit(NB * DBDBC_STRIDE / 4u, POISON);
    std::vector<F2> appliedInit(N, F2{ 5.0f, 5.0f });

    PxgBufferHandle bBlocks = S.make(blocksInit), bApplied = S.make(appliedInit), bNum = S.make(num);
    SsPrepPush p = {};
    p.softbodies = S.sbPool; p.contacts = S.addr(S.make(dead)); p.normalPens = S.addr(S.make(np));
    p.barycentrics0 = S.addr(S.make(b0)); p.barycentrics1 = S.addr(S.make(b1)); p.contactInfos = S.addr(S.make(pairs));
    p.numContacts = S.addr(bNum); p.contactBlocks = S.addr(bBlocks); p.softBodyAppliedForces = S.addr(bApplied);
    p.maxContacts = MAXC;
    Kernel k("softBodySelfSolveBda.spv", "sb_softbodyContactPrepareLaunch", {}, sizeof(SsPrepPush));
    k.dispatch(GRID, {}, p);
    dev->flush();

    std::vector<uint8_t> want(NB * DBDBC_STRIDE);
    std::memcpy(want.data(), blocksInit.data(), want.size());
    for (uint32_t w = 0; w < MAXC; ++w)
    {
        if (!pairValid(cs[w].pair.ref1)) continue;
        F4 n; float clamp;
        ssPrepareRef(S, cs[w], n, clamp);
        dbdbLaneSet(want, w, n, clamp);
    }
    uint32_t bad = 0;
    const std::vector<uint8_t> got = downloadBuffer<uint8_t>(dev, bBlocks, want.size());
    for (size_t i = 0; i < want.size(); i += 4)
        if (std::memcmp(&got[i], &want[i], 4) != 0)
        { if (++bad <= 8) std::printf("    block word @%zu got %08x want %08x\n", i, getAt<uint32_t>(got, i), getAt<uint32_t>(want, i)); }
    const std::vector<F2> gotApplied = downloadBuffer<F2>(dev, bApplied, N);
    for (uint32_t w = 0; w < N; ++w)
    {
        const F2 e = (w < MAXC) ? F2{ 0.f, 0.f } : F2{ 5.0f, 5.0f };
        cmpF4("applied", w, F4{ gotApplied[w].x, gotApplied[w].y, 0, 0 }, F4{ e.x, e.y, 0, 0 }, bad);
    }
    PXG_CHECK_MSG(downloadBuffer<uint32_t>(dev, bNum, 1)[0] == MAXC, "numContacts must be clamped to maxContacts and written back");
    PXG_CHECK_MSG(bad == 0, "sb_softbodyContactPrepareLaunch: %u mismatches", bad);
}

struct SsQueryPush
{
    uint64_t softbodies, contactInfos, barycentrics0, barycentrics1, contactBlocks, numContacts;
    float dt; uint32_t isVelocityIteration;
};
static_assert(sizeof(SsQueryPush) == 56, "sb_querySSContactReferenceCountLaunch blob");

struct SsSolvePush
{
    uint64_t softbodies, contactInfos, barycentrics0, barycentrics1, contactBlocks, numContacts;
    float dt; uint32_t pad52;
    uint64_t appliedForces, materials;
    uint32_t isVelocityIteration, pad76;
};
static_assert(sizeof(SsSolvePush) == 80, "sb_solveOutputSSDeltaVLaunch blob");

static void runSsQuerySolve(bool isVelocityIteration, uint32_t seed)
{
    PxgSlangComputeDevice* dev = testDevice();
    SbWorld S;
    buildSbWorld(S, dev, seed);
    const uint32_t NC = 45u;
    const std::vector<DbContact> cs = makeSsContacts(NC, seed);

    std::vector<uint8_t> blocks(((NC + 31u) / 32u) * DBDBC_STRIDE, 0);
    std::vector<F4> b0, b1;
    std::vector<PairInfo> pairs;
    for (uint32_t w = 0; w < NC; ++w)
    {
        b0.push_back(cs[w].bary0); b1.push_back(cs[w].bary1); pairs.push_back(cs[w].pair);
        if (!pairValid(cs[w].pair.ref1)) continue;
        F4 n; float clamp;
        ssPrepareRef(S, cs[w], n, clamp);
        dbdbLaneSet(blocks, w, n, clamp);
    }
    Rng rng{ seed ^ 0x9e37u };
    std::vector<F2> applied(NC);
    for (uint32_t w = 0; w < NC; ++w)
        applied[w] = (w % 4u == 0u) ? F2{ 0.f, 0.f } : F2{ rng.uni(0.0f, 0.04f), rng.uni(-0.03f, 0.03f) };
    std::vector<uint32_t> num = { NC };
    PxgBufferHandle bPairs = S.make(pairs), bApplied = S.make(applied);
    const uint64_t aB0 = S.addr(S.make(b0)), aB1 = S.addr(S.make(b1)), aBlocks = S.addr(S.make(blocks)),
                   aNum = S.addr(S.make(num));

    // ---- query ----
    {
        SsQueryPush q = { S.sbPool, S.addr(bPairs), aB0, aB1, aBlocks, aNum, kDt, isVelocityIteration ? 1u : 0u };
        Kernel k("softBodySelfSolveBda.spv", "sb_querySSContactReferenceCountLaunch", {}, sizeof(SsQueryPush));
        k.dispatch(GRID, {}, q);
        dev->flush();
    }
    std::vector<PairInfo> wantPairs = pairs;
    uint32_t nActive = 0, nInactive = 0;
    for (uint32_t w = 0; w < NC; ++w)
    {
        const PairInfo& pi = pairs[w];
        if (!pairValid(pi.ref1)) { wantPairs[w].ref1 = setActive(pi.ref1, false); continue; }
        DbPart d0, d1;
        d0.readSoftBody(S.sb[refId(pi.ref0)], pi.ref0, cs[w].bary0, nullptr, true, kDt, isVelocityIteration);
        d1.readSoftBody(S.sb[refId(pi.ref1)], pi.ref1, cs[w].bary1, nullptr, true, kDt, isVelocityIteration);
        const ContactPair cp = contactPairDbDb(getAt<F4>(blocks, (w / 32u) * DBDBC_STRIDE + (w & 31u) * 16),
                                               getAt<float>(blocks, (w / 32u) * DBDBC_STRIDE + DBDBC_maxPenBiasClamp + (w & 31u) * 4));
        SolveOut so;
        const bool isActive = solveDbDbContact(d0, d1, cp, so, 0.f, 0.f, 0.f, kDt, true, isVelocityIteration);
        wantPairs[w].ref1 = setActive(pi.ref1, isActive);
        if (isActive)
        {
            ++nActive;
            bumpSoftBody(S.sb[refId(pi.ref0)], pi.ref0, d0.bc);
            bumpSoftBody(S.sb[refId(pi.ref1)], pi.ref1, d1.bc);
        }
        else ++nInactive;
    }
    uint32_t bad = 0;
    {
        const std::vector<PairInfo> got = downloadBuffer<PairInfo>(dev, bPairs, NC);
        for (uint32_t w = 0; w < NC; ++w)
            if (got[w].ref0 != wantPairs[w].ref0 || got[w].ref1 != wantPairs[w].ref1)
            { if (++bad <= 8) std::printf("    pair %u got %llx want %llx\n", w, (unsigned long long)got[w].ref1, (unsigned long long)wantPairs[w].ref1); }
        for (uint32_t s = 0; s < 3; s += 2)
            cmpVec("query mSimDelta", S.readSoftBodyDelta(s), S.sb[s].delta, bad);
    }
    PXG_CHECK_MSG(bad == 0, "sb_querySSContactReferenceCountLaunch (%s): %u mismatches", isVelocityIteration ? "vel" : "pos", bad);
    PXG_CHECK_MSG(nActive > 0 && nInactive > 0, "coverage: active %u inactive %u", nActive, nInactive);

    // ---- solve ----
    {
        SsSolvePush sp = {};
        sp.softbodies = S.sbPool; sp.contactInfos = S.addr(bPairs); sp.barycentrics0 = aB0; sp.barycentrics1 = aB1;
        sp.contactBlocks = aBlocks; sp.numContacts = aNum; sp.dt = kDt; sp.appliedForces = S.addr(bApplied);
        sp.materials = S.volMaterials; sp.isVelocityIteration = isVelocityIteration ? 1u : 0u;
        Kernel k("softBodySelfSolveBda.spv", "sb_solveOutputSSDeltaVLaunch", {}, sizeof(SsSolvePush));
        k.dispatch(GRID, {}, sp);
        dev->flush();
    }
    std::vector<F2> wantApplied = applied;
    uint32_t nApplied = 0;
    for (uint32_t w = 0; w < NC; ++w)
    {
        const PairInfo& pi = wantPairs[w];
        if (!pairActive(pi.ref1)) continue;
        HostSoftBody& h0 = S.sb[refId(pi.ref0)];
        HostSoftBody& h1 = S.sb[refId(pi.ref1)];
        DbPart d0, d1;
        d0.readSoftBody(h0, pi.ref0, cs[w].bary0, S.volumeFriction.data(), false, kDt, isVelocityIteration);
        d1.readSoftBody(h1, pi.ref1, cs[w].bary1, S.volumeFriction.data(), false, kDt, isVelocityIteration);
        const float mu = (d0.friction + d1.friction) * 0.5f;
        const ContactPair cp = contactPairDbDb(getAt<F4>(blocks, (w / 32u) * DBDBC_STRIDE + (w & 31u) * 16),
                                               getAt<float>(blocks, (w / 32u) * DBDBC_STRIDE + DBDBC_maxPenBiasClamp + (w & 31u) * 4));
        SolveOut so;
        if (solveDbDbContact(d0, d1, cp, so, applied[w].x, applied[w].y, mu, kDt, false, isVelocityIteration))
        {
            ++nApplied;
            V3 delta;
            so.computeDelta(cp, delta, kDt);
            d0.writeSoftBody(h0, delta);
            d1.writeSoftBody(h1, neg(delta));
            wantApplied[w] = F2{ so.accumulatedDeltaLambdaN, applied[w].y + so.deltaLambdaT };
        }
    }
    bad = 0;
    {
        const std::vector<F2> got = downloadBuffer<F2>(dev, bApplied, NC);
        for (uint32_t w = 0; w < NC; ++w)
            cmpF4("appliedForces", w, F4{ got[w].x, got[w].y, 0, 0 }, F4{ wantApplied[w].x, wantApplied[w].y, 0, 0 }, bad);
        for (uint32_t s = 0; s < 3; s += 2)
            cmpVec("solve mSimDelta", S.readSoftBodyDelta(s), S.sb[s].delta, bad);
    }
    PXG_CHECK_MSG(bad == 0, "sb_solveOutputSSDeltaVLaunch (%s): %u mismatches", isVelocityIteration ? "vel" : "pos", bad);
    PXG_CHECK_MSG(nApplied > 0, "coverage: no SS contact applied");
}

PXG_TEST(softBodySolve512SoftContactPos) { runSsQuerySolve(false, 21u); }
PXG_TEST(softBodySolve512SoftContactVel) { runSsQuerySolve(true, 22u); }

// ===========================================================================
// Softbody-cloth contacts
// ===========================================================================
// ref0 = cloth (vertex, or for the query / solve also a triangle), ref1 = softbody tet.
static std::vector<DbContact> makeScContacts(uint32_t n, uint32_t seed, bool withTriangles)
{
    Rng rng{ seed * 17u + 9u };
    std::vector<DbContact> cs;
    const uint32_t NVC = 3u * NTRI + NLOOSE;
    for (uint32_t c = 0; c < n; ++c)
    {
        DbContact d;
        if (withTriangles && c % 4u == 0u)
            d.pair.ref0 = encodeRef(1, REF_eTRIANGLE, c / 4u);
        else
            d.pair.ref0 = encodeRef(1, REF_eVERTEX, (c * 5u) % NVC);
        d.pair.ref1 = (c % 10u == 7u) ? (encodeRef(0, 0, 0) | REF_INVALID) : encodeRef(c % 2u ? 2u : 0u, REF_eTETRAHEDRON, (c * 9u) % NT);
        if (c % 6u == 5u) d.pair.ref1 = setActive(d.pair.ref1, true);
        V3 nrm = v3(rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f));
        normalize(nrm);
        d.normalPen = F4{ nrm.x, nrm.y, nrm.z, rng.uni(-0.05f, 0.05f) };
        d.bary0 = randomBary(rng, c % 8u == 4u, 3);
        d.bary1 = randomBary(rng, c % 5u == 3u, 4);
        cs.push_back(d);
    }
    return cs;
}

static void scPrepareRef(const SbWorld& S, const DbContact& c, F4& np, float& clamp)
{
    const HostCloth& cl = S.cloth[refId(c.pair.ref0)];
    const HostSoftBody& sb = S.sb[refId(c.pair.ref1)];
    const F4 clothDelta = cl.accDelta[refFeature(c.pair.ref0)];
    const F4 sbDelta = projectTet(sb.deltaPos, sb.tets[refFeature(c.pair.ref1)], c.bary1);
    const float restDistance = cl.restOffset + sb.restOffset;
    const V3 n = xyz(c.normalPen);
    const V3 rel = v3(clothDelta.x - sbDelta.x, clothDelta.y - sbDelta.y, clothDelta.z - sbDelta.z);
    np = F4{ n.x, n.y, n.z, c.normalPen.w - restDistance + dot(rel, n) };
    clamp = selectMax(cl.penBiasClamp, sb.penBiasClamp);
}

PXG_TEST(softBodySolve512ClothContactPrepare)
{
    PxgSlangComputeDevice* dev = testDevice();
    SbWorld S;
    buildSbWorld(S, dev, 0x41u);
    const uint32_t N = 40u, MAXC = 37u;
    const std::vector<DbContact> cs = makeScContacts(N, 3u, false);
    std::vector<F4> np, b0, b1;
    std::vector<PairInfo> pairs;
    for (const DbContact& c : cs) { np.push_back(c.normalPen); b0.push_back(F4{ 1e30f, 1e30f, 1e30f, 1e30f }); b1.push_back(c.bary1); pairs.push_back(c.pair); }
    std::vector<uint32_t> num = { N };
    const uint32_t NB = (N + 31u) / 32u;
    std::vector<uint32_t> blocksInit(NB * DBDBC_STRIDE / 4u, POISON);
    std::vector<F2> appliedInit(N, F2{ 3.0f, 3.0f });
    PxgBufferHandle bBlocks = S.make(blocksInit), bApplied = S.make(appliedInit), bNum = S.make(num);
    SsPrepPush p = {};   // the same ten-argument shape: (softbodies, clothes, normalPens, ...)
    p.softbodies = S.sbPool; p.contacts = S.clothPool; p.normalPens = S.addr(S.make(np));
    p.barycentrics0 = S.addr(S.make(b0)); p.barycentrics1 = S.addr(S.make(b1)); p.contactInfos = S.addr(S.make(pairs));
    p.numContacts = S.addr(bNum); p.contactBlocks = S.addr(bBlocks); p.softBodyAppliedForces = S.addr(bApplied);
    p.maxContacts = MAXC;
    Kernel k("softBodyClothPrepareBda.spv", "sb_clothContactPrepareLaunch", {}, sizeof(SsPrepPush));
    k.dispatch(GRID, {}, p);
    dev->flush();

    std::vector<uint8_t> want(NB * DBDBC_STRIDE);
    std::memcpy(want.data(), blocksInit.data(), want.size());
    for (uint32_t w = 0; w < MAXC; ++w)
    {
        if (!pairValid(cs[w].pair.ref1)) continue;
        F4 n; float clamp;
        scPrepareRef(S, cs[w], n, clamp);
        dbdbLaneSet(want, w, n, clamp);
    }
    uint32_t bad = 0;
    const std::vector<uint8_t> got = downloadBuffer<uint8_t>(dev, bBlocks, want.size());
    for (size_t i = 0; i < want.size(); i += 4)
        if (std::memcmp(&got[i], &want[i], 4) != 0)
        { if (++bad <= 8) std::printf("    block word @%zu got %08x want %08x\n", i, getAt<uint32_t>(got, i), getAt<uint32_t>(want, i)); }
    const std::vector<F2> gotApplied = downloadBuffer<F2>(dev, bApplied, N);
    for (uint32_t w = 0; w < N; ++w)
    {
        const F2 e = (w < MAXC) ? F2{ 0.f, 0.f } : F2{ 3.0f, 3.0f };
        cmpF4("applied", w, F4{ gotApplied[w].x, gotApplied[w].y, 0, 0 }, F4{ e.x, e.y, 0, 0 }, bad);
    }
    PXG_CHECK_MSG(downloadBuffer<uint32_t>(dev, bNum, 1)[0] == MAXC, "numContacts must be clamped and written back");
    PXG_CHECK_MSG(bad == 0, "sb_clothContactPrepareLaunch: %u mismatches", bad);
}

struct ScQueryPush
{
    uint64_t softbodies, clothes, contactInfos, barycentrics0, barycentrics1, contactBlocks, numContacts;
    float dt; uint32_t isVelocityIteration;
};
static_assert(sizeof(ScQueryPush) == 64, "sb_querySCContactReferenceCountLaunch blob");

struct ScSolvePush
{
    uint64_t softbodies, clothes, contactInfos, barycentrics0, barycentrics1, contactBlocks, numContacts;
    float dt; uint32_t pad60;
    uint64_t appliedForces, sbMaterials, clothMaterials;
    uint32_t isVelocityIteration, pad92;
};
static_assert(sizeof(ScSolvePush) == 96, "sb_solveOutputSCDeltaVLaunch blob");

static void runScQuerySolve(bool isVelocityIteration, uint32_t seed)
{
    PxgSlangComputeDevice* dev = testDevice();
    SbWorld S;
    buildSbWorld(S, dev, seed);
    const uint32_t NC = 45u;
    const std::vector<DbContact> cs = makeScContacts(NC, seed, true);

    std::vector<uint8_t> blocks(((NC + 31u) / 32u) * DBDBC_STRIDE, 0);
    std::vector<F4> b0, b1;
    std::vector<PairInfo> pairs;
    Rng rng{ seed ^ 0x51u };
    for (uint32_t w = 0; w < NC; ++w)
    {
        b0.push_back(cs[w].bary0); b1.push_back(cs[w].bary1); pairs.push_back(cs[w].pair);
        if (!pairValid(cs[w].pair.ref1)) continue;
        // The blocks are the input here; random values exercise every branch of the solve.
        dbdbLaneSet(blocks, w, F4{ cs[w].normalPen.x, cs[w].normalPen.y, cs[w].normalPen.z, rng.uni(-0.06f, 0.03f) },
                    -rng.uni(0.2f, 2.0f));
    }
    std::vector<F2> applied(NC);
    for (uint32_t w = 0; w < NC; ++w)
        applied[w] = (w % 4u == 1u) ? F2{ 0.f, 0.f } : F2{ rng.uni(0.0f, 0.04f), rng.uni(-0.03f, 0.03f) };
    std::vector<uint32_t> num = { NC };
    PxgBufferHandle bPairs = S.make(pairs), bApplied = S.make(applied);
    const uint64_t aB0 = S.addr(S.make(b0)), aB1 = S.addr(S.make(b1)), aBlocks = S.addr(S.make(blocks)),
                   aNum = S.addr(S.make(num));

    {
        ScQueryPush q = { S.sbPool, S.clothPool, S.addr(bPairs), aB0, aB1, aBlocks, aNum, kDt, isVelocityIteration ? 1u : 0u };
        Kernel k("softBodyClothSolveBda.spv", "sb_querySCContactReferenceCountLaunch", {}, sizeof(ScQueryPush));
        k.dispatch(GRID, {}, q);
        dev->flush();
    }
    std::vector<PairInfo> wantPairs = pairs;
    uint32_t nActive = 0, nTri = 0;
    for (uint32_t w = 0; w < NC; ++w)
    {
        const PairInfo& pi = pairs[w];
        if (!pairValid(pi.ref1)) { wantPairs[w].ref1 = setActive(pi.ref1, false); continue; }
        DbPart dc, ds;
        dc.readCloth(S.cloth[refId(pi.ref0)], pi.ref0, cs[w].bary0, nullptr, true, kDt, isVelocityIteration);
        ds.readSoftBody(S.sb[refId(pi.ref1)], pi.ref1, cs[w].bary1, nullptr, true, kDt, isVelocityIteration);
        const ContactPair cp = contactPairDbDb(getAt<F4>(blocks, (w / 32u) * DBDBC_STRIDE + (w & 31u) * 16),
                                               getAt<float>(blocks, (w / 32u) * DBDBC_STRIDE + DBDBC_maxPenBiasClamp + (w & 31u) * 4));
        SolveOut so;
        const bool isActive = solveDbDbContact(dc, ds, cp, so, 0.f, 0.f, 0.f, kDt, true, isVelocityIteration);
        wantPairs[w].ref1 = setActive(pi.ref1, isActive);
        if (isActive)
        {
            ++nActive;
            if (refType(pi.ref0) == REF_eTRIANGLE) ++nTri;
            bumpCloth(S.cloth[refId(pi.ref0)], pi.ref0, dc.bc);
            bumpSoftBody(S.sb[refId(pi.ref1)], pi.ref1, ds.bc);
        }
    }
    uint32_t bad = 0;
    {
        const std::vector<PairInfo> got = downloadBuffer<PairInfo>(dev, bPairs, NC);
        for (uint32_t w = 0; w < NC; ++w)
            if (got[w].ref1 != wantPairs[w].ref1) { if (++bad <= 8) std::printf("    pair %u got %llx want %llx\n", w, (unsigned long long)got[w].ref1, (unsigned long long)wantPairs[w].ref1); }
        cmpVec("query cloth mDeltaPos", S.readClothDelta(), S.cloth[1].delta, bad);
        for (uint32_t s = 0; s < 3; s += 2)
            cmpVec("query mSimDelta", S.readSoftBodyDelta(s), S.sb[s].delta, bad);
    }
    PXG_CHECK_MSG(bad == 0, "sb_querySCContactReferenceCountLaunch (%s): %u mismatches", isVelocityIteration ? "vel" : "pos", bad);
    PXG_CHECK_MSG(nActive > 0 && nTri > 0, "coverage: active %u (triangles %u)", nActive, nTri);

    {
        ScSolvePush sp = {};
        sp.softbodies = S.sbPool; sp.clothes = S.clothPool; sp.contactInfos = S.addr(bPairs); sp.barycentrics0 = aB0;
        sp.barycentrics1 = aB1; sp.contactBlocks = aBlocks; sp.numContacts = aNum; sp.dt = kDt;
        sp.appliedForces = S.addr(bApplied); sp.sbMaterials = S.volMaterials; sp.clothMaterials = S.surfMaterials;
        sp.isVelocityIteration = isVelocityIteration ? 1u : 0u;
        Kernel k("softBodyClothSolveBda.spv", "sb_solveOutputSCDeltaVLaunch", {}, sizeof(ScSolvePush));
        k.dispatch(GRID, {}, sp);
        dev->flush();
    }
    std::vector<F2> wantApplied = applied;
    uint32_t nApplied = 0;
    for (uint32_t w = 0; w < NC; ++w)
    {
        const PairInfo& pi = wantPairs[w];
        if (!pairActive(pi.ref1)) continue;
        HostCloth& hc = S.cloth[refId(pi.ref0)];
        HostSoftBody& hs = S.sb[refId(pi.ref1)];
        DbPart dc, ds;
        dc.readCloth(hc, pi.ref0, cs[w].bary0, S.surfaceFriction.data(), false, kDt, isVelocityIteration);
        ds.readSoftBody(hs, pi.ref1, cs[w].bary1, S.volumeFriction.data(), false, kDt, isVelocityIteration);
        const float mu = (dc.friction + ds.friction) * 0.5f;
        const ContactPair cp = contactPairDbDb(getAt<F4>(blocks, (w / 32u) * DBDBC_STRIDE + (w & 31u) * 16),
                                               getAt<float>(blocks, (w / 32u) * DBDBC_STRIDE + DBDBC_maxPenBiasClamp + (w & 31u) * 4));
        SolveOut so;
        if (solveDbDbContact(dc, ds, cp, so, applied[w].x, applied[w].y, mu, kDt, false, isVelocityIteration))
        {
            ++nApplied;
            V3 delta;
            so.computeDelta(cp, delta, kDt);
            dc.writeCloth(hc, delta);
            ds.writeSoftBody(hs, neg(delta));
            wantApplied[w] = F2{ so.accumulatedDeltaLambdaN, applied[w].y + so.deltaLambdaT };
        }
    }
    bad = 0;
    {
        const std::vector<F2> got = downloadBuffer<F2>(dev, bApplied, NC);
        for (uint32_t w = 0; w < NC; ++w)
            cmpF4("appliedForces", w, F4{ got[w].x, got[w].y, 0, 0 }, F4{ wantApplied[w].x, wantApplied[w].y, 0, 0 }, bad);
        cmpVec("solve cloth mDeltaPos", S.readClothDelta(), S.cloth[1].delta, bad);
        for (uint32_t s = 0; s < 3; s += 2)
            cmpVec("solve mSimDelta", S.readSoftBodyDelta(s), S.sb[s].delta, bad);
    }
    PXG_CHECK_MSG(bad == 0, "sb_solveOutputSCDeltaVLaunch (%s): %u mismatches", isVelocityIteration ? "vel" : "pos", bad);
    PXG_CHECK_MSG(nApplied > 0, "coverage: no SC contact applied");
}

PXG_TEST(softBodySolve512ClothContactPos) { runScQuerySolve(false, 31u); }
PXG_TEST(softBodySolve512ClothContactVel) { runScQuerySolve(true, 32u); }

// ===========================================================================
// Softbody-particle contacts (query and the two solve halves)
// ===========================================================================
struct SpQueryPush
{
    uint64_t softbodies, particlesystems, contactInfos, contactBlocks, numContacts;
    float dt; uint32_t isVelocityIteration;
};
static_assert(sizeof(SpQueryPush) == 48, "sb_querySBParticleContactReferenceCountLaunch blob");

struct SpSolveSbPush
{
    uint64_t softbodies, particlesystems, contactInfos, contactBlocks, numContacts, appliedForces;
    float dt; uint32_t pad52;
    uint64_t materials;
    uint32_t isVelocityIteration, pad68;
};
static_assert(sizeof(SpSolveSbPush) == 72, "sb_solveOutputSPDeltaVLaunch blob");

struct SpSolvePsPush
{
    uint64_t softbodies, particlesystems, contactInfos, contactBlocks, numContacts, deltaP, appliedForces;
    float dt; uint32_t pad60;
    uint64_t materials;
    uint32_t isVelocityIteration, pad76;
};
static_assert(sizeof(SpSolvePsPush) == 80, "sb_solveOutputParticleDeltaVLaunch blob");

static void runSpQuerySolve(bool isVelocityIteration, uint32_t seed)
{
    PxgSlangComputeDevice* dev = testDevice();
    World W;
    buildWorld(W, dev, seed);
    SbWorld S;
    buildSbWorld(S, dev, seed + 7u);
    const uint32_t NC = 45u;
    Rng rng{ seed * 31u + 1u };

    std::vector<PairInfo> pairs;
    std::vector<uint8_t> blocks(((NC + 31u) / 32u) * PDBC_STRIDE, 0);
    struct Lane { F4 np, bary; float clamp; };
    std::vector<Lane> lanes(NC);
    for (uint32_t c = 0; c < NC; ++c)
    {
        const uint32_t sys = (c % 3u == 2u) ? 2u : 0u;
        PairInfo pi;
        pi.ref0 = particleRef(sys, rng.next() % 10u);
        pi.ref1 = (c % 12u == 5u) ? (encodeRef(0, 0, 0) | REF_INVALID) : encodeRef(c % 2u ? 0u : 2u, REF_eTETRAHEDRON, (c * 13u) % NT);
        if (c % 7u == 2u) pi.ref1 = setActive(pi.ref1, true);
        pairs.push_back(pi);
        V3 n = v3(rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f), rng.uni(-1.f, 1.f));
        normalize(n);
        lanes[c].np = F4{ n.x, n.y, n.z, rng.uni(-0.06f, 0.03f) };
        lanes[c].bary = randomBary(rng, c % 6u == 1u, 4);
        lanes[c].clamp = -rng.uni(0.1f, 2.0f);
        const size_t base = size_t(c / 32u) * PDBC_STRIDE;
        putAt(blocks, base + PDBC_normal_pen + (c & 31u) * 16, lanes[c].np);
        putAt(blocks, base + PDBC_barycentric + (c & 31u) * 16, lanes[c].bary);
        putAt(blocks, base + PDBC_maxPenBiasClamp + (c & 31u) * 4, lanes[c].clamp);
    }
    std::vector<F2> appliedSb(NC), appliedPs(NC);
    for (uint32_t w = 0; w < NC; ++w)
    {
        appliedSb[w] = (w % 5u == 0u) ? F2{ 0.f, 0.f } : F2{ rng.uni(0.0f, 0.04f), rng.uni(-0.03f, 0.03f) };
        appliedPs[w] = appliedSb[w];
    }
    std::vector<uint32_t> num = { NC };
    PxgBufferHandle bPairs = W.make(pairs), bAppSb = W.make(appliedSb), bAppPs = W.make(appliedPs);
    const uint64_t aBlocks = W.addr(W.make(blocks)), aNum = W.addr(W.make(num));

    // ---- query ----
    {
        SpQueryPush q = { S.sbPool, W.psysPool, W.addr(bPairs), aBlocks, aNum, kDt, isVelocityIteration ? 1u : 0u };
        Kernel k("softBodyPsSolveBda.spv", "sb_querySBParticleContactReferenceCountLaunch", {}, sizeof(SpQueryPush));
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
        DbPart sbPart, psPart;
        sbPart.readSoftBody(S.sb[refId(pi.ref1)], pi.ref1, lanes[w].bary, nullptr, true, kDt, isVelocityIteration);
        psPart.readParticle(pv, false, false, kDt, isVelocityIteration);
        const ContactPair cp = contactPairDbDb(lanes[w].np, lanes[w].clamp);
        SolveOut so;
        const bool isActive = solveDbDbContact(sbPart, psPart, cp, so, 0.f, 0.f, 0.f, kDt, true, isVelocityIteration);
        wantPairs[w].ref1 = setActive(pi.ref1, isActive);
        if (isActive)
        {
            ++nActive;
            bumpSoftBody(S.sb[refId(pi.ref1)], pi.ref1, sbPart.bc);
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
            cmpVec("query mSimDelta", S.readSoftBodyDelta(s), S.sb[s].delta, bad);
            cmpVec("query mAccumDeltaP", downloadBuffer<F4>(dev, W.bAccum[s], W.sys[s].accum.size()), W.sys[s].accum, bad);
        }
    }
    PXG_CHECK_MSG(bad == 0, "sb_querySBParticleContactReferenceCountLaunch (%s): %u mismatches", isVelocityIteration ? "vel" : "pos", bad);
    PXG_CHECK_MSG(nActive > 0 && nKinematic > 0, "coverage: active %u kinematic %u", nActive, nKinematic);

    // ---- the two solve halves on the query's output ----
    std::vector<F4> deltaInit(NC, F4{ 7.f, 7.f, 7.f, 7.f });
    PxgBufferHandle bDeltaP = W.make(deltaInit);
    {
        SpSolveSbPush sp = {};
        sp.softbodies = S.sbPool; sp.particlesystems = W.psysPool; sp.contactInfos = W.addr(bPairs); sp.contactBlocks = aBlocks;
        sp.numContacts = aNum; sp.appliedForces = W.addr(bAppSb); sp.dt = kDt; sp.materials = S.volMaterials;
        sp.isVelocityIteration = isVelocityIteration ? 1u : 0u;
        Kernel k("softBodyPsSolveBda.spv", "sb_solveOutputSPDeltaVLaunch", {}, sizeof(SpSolveSbPush));
        k.dispatch(GRID, {}, sp);
        SpSolvePsPush pp = {};
        pp.softbodies = S.sbPool; pp.particlesystems = W.psysPool; pp.contactInfos = W.addr(bPairs); pp.contactBlocks = aBlocks;
        pp.numContacts = aNum; pp.deltaP = W.addr(bDeltaP); pp.appliedForces = W.addr(bAppPs); pp.dt = kDt;
        pp.materials = S.volMaterials; pp.isVelocityIteration = isVelocityIteration ? 1u : 0u;
        Kernel k2("softBodyPsSolveBda.spv", "sb_solveOutputParticleDeltaVLaunch", {}, sizeof(SpSolvePsPush));
        k2.dispatch(GRID, {}, pp);
        dev->flush();
    }
    // Both halves read the SAME pre-solve mSimDelta (the particle half never writes it, and the
    // SB half only touches .xyz, which neither reader consumes), so the oracle evaluates the
    // two from one snapshot.
    const HostSoftBody snap[3] = { S.sb[0], S.sb[1], S.sb[2] };
    std::vector<F2> wantSb = appliedSb, wantPs = appliedPs;
    std::vector<F4> wantDeltaP(NC);
    uint32_t nApplied = 0;
    for (uint32_t w = 0; w < NC; ++w)
    {
        const PairInfo& pi = wantPairs[w];
        const uint32_t sys = particleSystemIdOf(pi.ref0), pidx = particleIndexOf(pi.ref0);
        // particleView reads the host accum, which carries the query's bumps.
        const ParticleView pv = W.particleView(sys, pidx);
        if (pv.sortedDeltaP.w == 0.0f) { wantDeltaP[w] = F4{ 0, 0, 0, 0 }; continue; }
        if (!pairActive(pi.ref1)) { wantDeltaP[w] = F4{ 0, 0, 0, 0 }; continue; }
        const HostSoftBody& hs = snap[refId(pi.ref1)];
        DbPart sbPart, psPart;
        sbPart.readSoftBody(hs, pi.ref1, lanes[w].bary, S.volumeFriction.data(), false, kDt, isVelocityIteration);
        psPart.readParticle(pv, true, true, kDt, isVelocityIteration);
        const ContactPair cp = contactPairDbDb(lanes[w].np, lanes[w].clamp);
        const float mu = (sbPart.friction + psPart.friction) * 0.5f;
        SolveOut so;
        if (solveDbDbContact(sbPart, psPart, cp, so, appliedSb[w].x, appliedSb[w].y, mu, kDt, false, isVelocityIteration))
        {
            ++nApplied;
            V3 delta;
            so.computeDelta(cp, delta, kDt);
            sbPart.writeSoftBody(S.sb[refId(pi.ref1)], delta);
            wantDeltaP[w] = psPart.writeParticle(neg(delta), so);
            wantSb[w] = F2{ so.accumulatedDeltaLambdaN, appliedSb[w].y + so.deltaLambdaT };
            wantPs[w] = F2{ so.accumulatedDeltaLambdaN, appliedPs[w].y + so.deltaLambdaT };
        }
        else
            wantDeltaP[w] = F4{ 0, 0, 0, 0 };
    }
    bad = 0;
    {
        cmpVec("deltaP", downloadBuffer<F4>(dev, bDeltaP, NC), wantDeltaP, bad);
        const std::vector<F2> gs = downloadBuffer<F2>(dev, bAppSb, NC), gp = downloadBuffer<F2>(dev, bAppPs, NC);
        for (uint32_t w = 0; w < NC; ++w)
        {
            cmpF4("appliedSb", w, F4{ gs[w].x, gs[w].y, 0, 0 }, F4{ wantSb[w].x, wantSb[w].y, 0, 0 }, bad);
            cmpF4("appliedPs", w, F4{ gp[w].x, gp[w].y, 0, 0 }, F4{ wantPs[w].x, wantPs[w].y, 0, 0 }, bad);
        }
        for (uint32_t s = 0; s < 3; s += 2)
            cmpVec("solve mSimDelta", S.readSoftBodyDelta(s), S.sb[s].delta, bad);
    }
    PXG_CHECK_MSG(bad == 0, "sb_solveOutput{SP,Particle}DeltaVLaunch (%s): %u mismatches", isVelocityIteration ? "vel" : "pos", bad);
    PXG_CHECK_MSG(nApplied > 0, "coverage: no SP contact applied");
}

PXG_TEST(softBodySolve512ParticleContactPos) { runSpQuerySolve(false, 41u); }
PXG_TEST(softBodySolve512ParticleContactVel) { runSpQuerySolve(true, 42u); }

// ===========================================================================
// Softbody-rigid attachments
// ===========================================================================
struct DefAttachment { PairInfo pair; F4 bOrP0, bary1; };

// prepareRigidDbAttachment (deformableUtils.cuh), on the World's rigid data.
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

// NA attachments: tets (each at most twice) and lone vertices (each once) of softbodies 0 / 2.
static std::vector<DefAttachment> makeRigidAttachments(uint32_t NA, uint32_t seed)
{
    Rng rng{ seed * 3u + 11u };
    const uint64_t rigids[6] = { staticNode(), makeRigidNode(4), makeRigidNode(6), makeRigidNode(8),
                                 makeArticulationNode(10, 2), makeArticulationNode(10, 1) };
    std::vector<DefAttachment> as;
    for (uint32_t a = 0; a < NA; ++a)
    {
        DefAttachment d;
        const uint32_t s = (a % 2u) ? 2u : 0u;
        d.pair.ref0 = rigids[(a + rng.next()) % 6u];
        d.pair.ref1 = (a % 3u == 0u) ? encodeRef(s, REF_eVERTEX, (a * 17u) % (4u * NT))
                                     : encodeRef(s, REF_eTETRAHEDRON, (a * 7u + 3u) % NT);
        d.bOrP0 = F4{ rng.uni(-0.5f, 0.5f), rng.uni(-0.5f, 0.5f), rng.uni(-0.5f, 0.5f), 0.0f };
        d.bary1 = randomBary(rng, a % 5u == 2u, 4);
        as.push_back(d);
    }
    return as;
}

static RdbAttachLane rigidAttachLaneRef(const World& W, const SbWorld& S, const DefAttachment& d, uint32_t refCount,
                                        F4 slotPrior, bool& zeroDeltaVel)
{
    const HostSoftBody& h = S.sb[refId(d.pair.ref1)];
    const uint32_t e = refFeature(d.pair.ref1);
    F4 gridPos;
    float invMass1;
    if (refType(d.pair.ref1) == REF_eVERTEX)
    {
        gridPos = h.pos[e];
        invMass1 = gridPos.w;
    }
    else
    {
        const U4 t = h.tets[e];
        const F4 b = d.bary1;
        const F4 p0 = h.pos[t.x], p1 = h.pos[t.y], p2 = h.pos[t.z], p3 = h.pos[t.w];
        gridPos = add4(add4(add4(mul4(p0, b.x), mul4(p1, b.y)), mul4(p2, b.z)), mul4(p3, b.w));
        invMass1 = b.x * b.x * p0.w + b.y * b.y * p1.w + b.z * b.z * p2.w + b.w * b.w * p3.w;
    }
    return prepareRigidDbAttachmentRef(W, xyz(gridPos), invMass1, xyz(d.bOrP0), d.pair.ref0, d.pair.ref1, d.bary1,
                                       refCount, slotPrior, zeroDeltaVel);
}

struct RaPrepPush
{
    uint64_t softbodies, rigidAttachments, activeRigidAttachments, rigidAttachmentRigidRefCounts, rigidAttachmentIds;
    uint32_t numRigidAttachments, pad44;
    uint64_t attachmentBlocks, preDesc, prepareDesc, sharedDesc, rigidDeltaVel;
};
static_assert(sizeof(RaPrepPush) == 88, "sb_rigidAttachmentPrepareLaunch blob");

PXG_TEST(softBodySolve512RigidAttachPrepare)
{
    PxgSlangComputeDevice* dev = testDevice();
    World W;
    buildWorld(W, dev, 0x777u, /*rotatedFrames*/ true);
    SbWorld S;
    buildSbWorld(S, dev, 0x88u);
    const uint32_t NREC = 50u;
    const std::vector<DefAttachment> recs = makeRigidAttachments(NREC, 5u);
    // the active subset, in a permuted order: 41 of the 50 records
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
    p.softbodies = S.sbPool; p.rigidAttachments = W.addr(W.make(recBytes)); p.activeRigidAttachments = W.addr(W.make(active));
    p.rigidAttachmentRigidRefCounts = W.addr(W.make(refCounts)); p.rigidAttachmentIds = W.addr(bIds);
    p.numRigidAttachments = NA; p.attachmentBlocks = W.addr(bBlocks); p.preDesc = W.prePrepDesc; p.prepareDesc = W.prepDesc;
    p.sharedDesc = W.sharedDesc; p.rigidDeltaVel = W.addr(bDv);
    Kernel k("softBodyRigidAttachBda.spv", "sb_rigidAttachmentPrepareLaunch", {}, sizeof(RaPrepPush));
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
        const RdbAttachLane want = rigidAttachLaneRef(W, S, d, refCounts[w], poisonF4, zeroDv);
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
    PXG_CHECK_MSG(bad == 0, "sb_rigidAttachmentPrepareLaunch: %u mismatches", bad);
    PXG_CHECK_MSG(nVertex > 0 && nArti > 0, "coverage: vertex %u articulation %u", nVertex, nArti);
}

struct RaQueryPush { uint64_t softbodies, attachmentBlocks; uint32_t numAttachments, pad20; };
static_assert(sizeof(RaQueryPush) == 24, "attachment query blob");

struct RaSolvePush
{
    uint64_t softbodies, attachmentBlocks;
    uint32_t numAttachments, pad20;
    uint64_t prePrepDesc, solverCoreDesc, artiCoreDesc, solverBodyVelPool;
    float dt, biasCoefficient;
    uint64_t rigidDeltaVel;
    uint32_t flags, pad76;   // byte0 isVelocityIteration, byte1 isTGS
};
static_assert(sizeof(RaSolvePush) == 80, "sb_solveRigidSoftAttachmentLaunch blob");

static void runRigidAttachQuerySolve(bool isTGS, bool isVelocityIteration, uint32_t seed)
{
    PxgSlangComputeDevice* dev = testDevice();
    World W;
    buildWorld(W, dev, seed, /*rotatedFrames*/ true);
    SbWorld S;
    buildSbWorld(S, dev, seed + 3u);
    const uint32_t NA = 45u;
    const std::vector<DefAttachment> as = makeRigidAttachments(NA, seed);
    const VelocityReader reader = W.reader();

    std::vector<uint8_t> blocks(((NA + 31u) / 32u) * RDBA_STRIDE, 0);
    std::vector<RdbAttachLane> lanes(NA);
    for (uint32_t w = 0; w < NA; ++w)
    {
        bool zdv;
        lanes[w] = rigidAttachLaneRef(W, S, as[w], 1u + (w % 3u), F4{ 1e30f, -1e30f, 1e30f, -1e30f }, zdv);
        putRdba(blocks, w, lanes[w]);
    }
    const uint64_t aBlocks = W.addr(W.make(blocks));

    // ---- query ----
    {
        RaQueryPush q = { S.sbPool, aBlocks, NA, 0u };
        Kernel k("softBodyRigidAttachBda.spv", "sb_queryRigidSoftAttachmentReferenceCountLaunch", {}, sizeof(RaQueryPush));
        k.dispatch(GRID, {}, q);
        dev->flush();
    }
    for (uint32_t w = 0; w < NA; ++w)
        bumpSoftBody(S.sb[refId(lanes[w].deformableRef)], lanes[w].deformableRef, lanes[w].barycentric);
    uint32_t bad = 0;
    for (uint32_t s = 0; s < 3; s += 2)
        cmpVec("query mSimDelta", S.readSoftBodyDelta(s), S.sb[s].delta, bad);
    PXG_CHECK_MSG(bad == 0, "sb_queryRigidSoftAttachmentReferenceCountLaunch: %u mismatches", bad);

    // ---- solve ----
    const float bias = isVelocityIteration ? 0.0f : 0.7f;
    std::vector<F4> dvInit(2 * NA, F4{ 5.f, 5.f, 5.f, 5.f });
    PxgBufferHandle bDv = W.make(dvInit);
    {
        RaSolvePush sp = {};
        sp.softbodies = S.sbPool; sp.attachmentBlocks = aBlocks; sp.numAttachments = NA; sp.prePrepDesc = W.prePrepDesc;
        sp.solverCoreDesc = W.coreDesc; sp.artiCoreDesc = W.artiCoreDesc; sp.solverBodyVelPool = W.velPoolAddr;
        sp.dt = kDt; sp.biasCoefficient = bias; sp.rigidDeltaVel = W.addr(bDv);
        sp.flags = (isVelocityIteration ? 1u : 0u) | (isTGS ? 0x100u : 0u);
        Kernel k("softBodyRigidAttachBda.spv", "sb_solveRigidSoftAttachmentLaunch", {}, sizeof(RaSolvePush));
        k.dispatch(GRID, {}, sp);
        dev->flush();
    }
    const HostSoftBody snap[3] = { S.sb[0], S.sb[1], S.sb[2] };
    std::vector<F4> wantDv = dvInit;
    uint32_t nSkipped = 0, nWritten = 0;
    for (uint32_t w = 0; w < NA; ++w)
    {
        const RdbAttachLane& L = lanes[w];
        DbPart db;
        db.readSoftBodyAttachment(snap[refId(L.deformableRef)], L.deformableRef, L.barycentric, isTGS, kDt, isVelocityIteration);
        if (nodeIsStatic(L.rigidId) && db.attachPointInvMass == 0.0f) { ++nSkipped; continue; }
        AttachRigid rigid;
        rigid.readAttachmentPrep(L);
        rigid.readVelocity(reader, L.rigidId, isTGS);
        V3 dLin, dAng;
        const V3 imp = solveRigidDbAttachment(rigid, db, kDt, bias, isTGS, isVelocityIteration, dLin, dAng);
        if (rigid.writeAttachmentDeltas(dLin, dAng, wantDv[w], wantDv[w + NA], L.rigidId)) ++nWritten;
        if (!(imp.x == 0.0f && imp.y == 0.0f && imp.z == 0.0f))
            db.writeSoftBody(S.sb[refId(L.deformableRef)], mul(neg(imp), kDt));
    }
    bad = 0;
    cmpVec("rigidDeltaVel", downloadBuffer<F4>(dev, bDv, 2 * NA), wantDv, bad);
    for (uint32_t s = 0; s < 3; s += 2)
        cmpVec("solve mSimDelta", S.readSoftBodyDelta(s), S.sb[s].delta, bad);
    PXG_CHECK_MSG(bad == 0, "sb_solveRigidSoftAttachmentLaunch (%s %s): %u mismatches", isTGS ? "TGS" : "PGS",
                  isVelocityIteration ? "vel" : "pos", bad);
    PXG_CHECK_MSG(nWritten > 0, "coverage: written %u skipped %u", nWritten, nSkipped);
}

PXG_TEST(softBodySolve512RigidAttachPGS) { runRigidAttachQuerySolve(false, false, 51u); }
PXG_TEST(softBodySolve512RigidAttachTGS) { runRigidAttachQuerySolve(true, false, 52u); }
PXG_TEST(softBodySolve512RigidAttachPGSVel) { runRigidAttachQuerySolve(false, true, 53u); }
PXG_TEST(softBodySolve512RigidAttachTGSVel) { runRigidAttachQuerySolve(true, true, 54u); }


// ===========================================================================
// Deformable-deformable attachments (softbody-softbody, softbody-cloth)
// ===========================================================================
// A per-(deformable, vertex) scatter budget: every vertex a side can write to (all corners of
// a tet / triangle, or the one vertex) is charged, and no vertex takes more than two, so the
// float atomics of one dispatch commute.
struct ScatterBudget
{
    uint8_t sb[3][4 * NT] = {};
    uint8_t cloth[3 * NTRI + NLOOSE] = {};

    // Charges every vertex side `r` can write; false when one goes over two.
    bool charge(uint64_t r, bool isCloth)
    {
        const uint32_t e = refFeature(r);
        uint8_t* c = isCloth ? cloth : sb[refId(r) % 3u];
        uint32_t v[4] = { e, 0, 0, 0 }, n = 1;
        if (refType(r) != REF_eVERTEX)
        {
            n = isCloth ? 3u : 4u;
            for (uint32_t k = 0; k < n; ++k) v[k] = n * e + k;
        }
        bool ok = true;
        for (uint32_t k = 0; k < n; ++k) ok = (++c[v[k]] <= 2u) && ok;
        return ok;
    }
    // Self attachments charge the same counters twice.
    bool fits(uint64_t r0, uint64_t r1, bool cloth1)
    {
        ScatterBudget trial = *this;
        const bool ok0 = trial.charge(r0, false);
        const bool ok1 = trial.charge(r1, cloth1);
        if (!ok0 || !ok1) return false;
        *this = trial;
        return true;
    }
};

// NA attachments.  Side 0 is a softbody (0 or 2), side 1 a softbody (possibly the same one)
// or the cloth.  Attachment 0 joins two pinned vertices (inverse mass 0 on both sides), the
// solve's wTot == 0 skip.
static std::vector<DefAttachment> makeDbAttachments(uint32_t NA, uint32_t seed, bool toCloth)
{
    Rng rng{ seed * 29u + 7u };
    ScatterBudget budget;
    std::vector<DefAttachment> as;
    {
        DefAttachment d;
        d.pair.ref0 = encodeRef(0, REF_eVERTEX, 5u);                                         // 5 % 7 == 5: pinned
        d.pair.ref1 = toCloth ? encodeRef(1, REF_eVERTEX, 3u) : encodeRef(2, REF_eVERTEX, 12u);   // pinned
        d.bOrP0 = F4{ 1e30f, -1e30f, 1e30f, -1e30f };
        d.bary1 = F4{ -1e30f, 1e30f, -1e30f, 1e30f };
        budget.fits(d.pair.ref0, d.pair.ref1, toCloth);
        as.push_back(d);
    }
    for (uint32_t tries = 0; as.size() < NA && tries < 10000u; ++tries)
    {
        DefAttachment d;
        const uint32_t a = uint32_t(as.size());
        const uint32_t s0 = (rng.next() & 1u) ? 2u : 0u;
        d.pair.ref0 = (rng.next() % 3u == 0u) ? encodeRef(s0, REF_eVERTEX, rng.next() % (4u * NT))
                                              : encodeRef(s0, REF_eTETRAHEDRON, rng.next() % NT);
        if (toCloth)
        {
            const uint32_t nv = 3u * NTRI + NLOOSE;
            d.pair.ref1 = (rng.next() % 3u == 0u) ? encodeRef(1, REF_eVERTEX, rng.next() % nv)
                                                  : encodeRef(1, REF_eTRIANGLE, rng.next() % NTRI);
        }
        else
        {
            const uint32_t s1 = (a % 5u == 4u) ? s0 : (2u - s0);   // every fifth a self attachment
            d.pair.ref1 = (rng.next() % 3u == 0u) ? encodeRef(s1, REF_eVERTEX, rng.next() % (4u * NT))
                                                  : encodeRef(s1, REF_eTETRAHEDRON, rng.next() % NT);
        }
        d.bOrP0 = randomBary(rng, a % 6u == 2u, 4);
        d.bary1 = randomBary(rng, a % 7u == 3u, toCloth ? 3u : 4u);
        if (budget.fits(d.pair.ref0, d.pair.ref1, toCloth))
            as.push_back(d);
    }
    return as;
}

// The block lane femAttachmentPrepareLaunch writes; a vertex side keeps whatever its
// barycentric slot held (`prior0` / `prior1`).
static void putDbdba(std::vector<uint8_t>& blk, uint32_t w, const DefAttachment& d, F4 prior0, F4 prior1)
{
    const size_t base = size_t(w / 32u) * DBDBA_STRIDE;
    const uint32_t l = w & 31u;
    putAt(blk, base + DBDBA_barycentric0 + l * 16, refType(d.pair.ref0) == REF_eVERTEX ? prior0 : d.bOrP0);
    putAt(blk, base + DBDBA_barycentric1 + l * 16, refType(d.pair.ref1) == REF_eVERTEX ? prior1 : d.bary1);
    putAt(blk, base + DBDBA_deformableRef0 + l * 8, d.pair.ref0);
    putAt(blk, base + DBDBA_deformableRef1 + l * 8, d.pair.ref1);
}

struct FaPrepPush
{
    uint64_t femAttachments, activeFemAttachments;
    uint32_t numFemAttachments, pad20;
    uint64_t attachmentBlocks;
};
static_assert(sizeof(FaPrepPush) == 32, "femAttachmentPrepareLaunch blob");

PXG_TEST(softBodySolve512FemAttachPrepare)
{
    PxgSlangComputeDevice* dev = testDevice();
    SbWorld S;
    buildSbWorld(S, dev, 0x99u);
    const uint32_t NREC = 60u;
    std::vector<DefAttachment> recs = makeDbAttachments(NREC / 2u, 61u, false);
    const std::vector<DefAttachment> rc = makeDbAttachments(NREC - uint32_t(recs.size()), 62u, true);
    recs.insert(recs.end(), rc.begin(), rc.end());
    PXG_CHECK_MSG(recs.size() == NREC, "attachment generator fell short (%zu)", recs.size());
    std::vector<uint32_t> active;
    for (uint32_t i = 0; i < NREC; ++i)
        if (i % 7u != 2u) active.push_back((i * 37u) % NREC);
    const uint32_t NA = uint32_t(active.size());
    std::vector<uint8_t> recBytes(NREC * DATT_STRIDE, 0);
    for (uint32_t i = 0; i < NREC; ++i)
    {
        putAt(recBytes, i * DATT_STRIDE + 0, recs[i].pair);
        putAt(recBytes, i * DATT_STRIDE + 16, recs[i].bOrP0);
        putAt(recBytes, i * DATT_STRIDE + 32, recs[i].bary1);
    }
    const uint32_t NB = (NA + 31u) / 32u + 1u;   // one spare block past the work must stay untouched
    std::vector<uint32_t> blocksInit(NB * DBDBA_STRIDE / 4u, POISON);
    PxgBufferHandle bBlocks = S.make(blocksInit);

    FaPrepPush p = {};
    p.femAttachments = S.addr(S.make(recBytes)); p.activeFemAttachments = S.addr(S.make(active));
    p.numFemAttachments = NA; p.attachmentBlocks = S.addr(bBlocks);
    Kernel k("softBodyFemAttachBda.spv", "femAttachmentPrepareLaunch", {}, sizeof(FaPrepPush));
    k.dispatch(GRID, {}, p);
    dev->flush();

    std::vector<uint8_t> want(NB * DBDBA_STRIDE);
    std::memcpy(want.data(), blocksInit.data(), want.size());
    const F4 poisonF4 = F4{ f32(POISON), f32(POISON), f32(POISON), f32(POISON) };
    uint32_t nVertex0 = 0, nVertex1 = 0;
    for (uint32_t w = 0; w < NA; ++w)
    {
        const DefAttachment& d = recs[active[w]];
        putDbdba(want, w, d, poisonF4, poisonF4);
        nVertex0 += refType(d.pair.ref0) == REF_eVERTEX;
        nVertex1 += refType(d.pair.ref1) == REF_eVERTEX;
    }
    const std::vector<uint8_t> got = downloadBuffer<uint8_t>(dev, bBlocks, want.size());
    uint32_t bad = 0;
    for (size_t i = 0; i < want.size(); i += 4)
        if (std::memcmp(&got[i], &want[i], 4) != 0)
        { if (++bad <= 8) std::printf("    block word @%zu got %08x want %08x\n", i, getAt<uint32_t>(got, i), getAt<uint32_t>(want, i)); }
    PXG_CHECK_MSG(bad == 0, "femAttachmentPrepareLaunch: %u mismatches", bad);
    PXG_CHECK_MSG(nVertex0 > 0 && nVertex1 > 0 && nVertex0 < NA && nVertex1 < NA, "coverage: vertex sides %u / %u of %u",
                  nVertex0, nVertex1, NA);
}

struct SaQueryPush { uint64_t softbodies, attachmentBlocks; uint32_t numAttachments, pad20; };
static_assert(sizeof(SaQueryPush) == 24, "sb_querySoftBodyAttachmentReferenceCountLaunch blob");
struct SaSolvePush { uint64_t softbodies, attachmentBlocks; uint32_t numAttachments; float dt; uint32_t isVelocityIteration, pad28; };
static_assert(sizeof(SaSolvePush) == 32, "sb_solveOutputSoftBodyAttachmentDeltaVLaunch blob");
struct CaQueryPush { uint64_t softbodies, clothes, attachmentBlocks; uint32_t numAttachments, pad28; };
static_assert(sizeof(CaQueryPush) == 32, "sb_queryClothAttachmentReferenceCountLaunch blob");
struct CaSolvePush { uint64_t softbodies, clothes, attachmentBlocks; uint32_t numAttachments; float dt; uint32_t isVelocityIteration, pad36; };
static_assert(sizeof(CaSolvePush) == 40, "sb_solveOutputClothAttachmentDeltaVLaunch blob");

static void runDbAttachQuerySolve(bool toCloth, bool isVelocityIteration, uint32_t seed)
{
    PxgSlangComputeDevice* dev = testDevice();
    SbWorld S;
    buildSbWorld(S, dev, seed);
    const uint32_t NA = 45u;
    const std::vector<DefAttachment> as = makeDbAttachments(NA, seed, toCloth);
    PXG_CHECK_MSG(as.size() == NA, "attachment generator fell short (%zu)", as.size());

    // A vertex side's barycentric slot holds NaN-free garbage the kernels must not read.
    std::vector<uint8_t> blocks(((NA + 31u) / 32u) * DBDBA_STRIDE, 0);
    for (uint32_t w = 0; w < NA; ++w)
        putDbdba(blocks, w, as[w], F4{ 7.0f, -3.0f, 11.0f, 0.5f }, F4{ -9.0f, 4.0f, 0.25f, 6.0f });
    const uint64_t aBlocks = S.addr(S.make(blocks));
    const char* spv = toCloth ? "softBodyClothAttachBda.spv" : "softBodyFemAttachBda.spv";

    // ---- query ----
    if (toCloth)
    {
        CaQueryPush q = { S.sbPool, S.clothPool, aBlocks, NA, 0u };
        Kernel k(spv, "sb_queryClothAttachmentReferenceCountLaunch", {}, sizeof(CaQueryPush));
        k.dispatch(GRID, {}, q);
    }
    else
    {
        SaQueryPush q = { S.sbPool, aBlocks, NA, 0u };
        Kernel k(spv, "sb_querySoftBodyAttachmentReferenceCountLaunch", {}, sizeof(SaQueryPush));
        k.dispatch(GRID, {}, q);
    }
    dev->flush();
    for (const DefAttachment& d : as)
    {
        bumpSoftBody(S.sb[refId(d.pair.ref0)], d.pair.ref0, d.bOrP0);
        if (toCloth) bumpCloth(S.cloth[1], d.pair.ref1, d.bary1);
        else bumpSoftBody(S.sb[refId(d.pair.ref1)], d.pair.ref1, d.bary1);
    }
    uint32_t bad = 0;
    for (uint32_t s = 0; s < 3; s += 2)
        cmpVec("query mSimDelta", S.readSoftBodyDelta(s), S.sb[s].delta, bad);
    if (toCloth)
        cmpVec("query cloth mDeltaPos", S.readClothDelta(), S.cloth[1].delta, bad);
    PXG_CHECK_MSG(bad == 0, "%s query: %u mismatches", toCloth ? "cloth attachment" : "softbody attachment", bad);

    // ---- solve ----
    if (toCloth)
    {
        CaSolvePush sp = { S.sbPool, S.clothPool, aBlocks, NA, kDt, isVelocityIteration ? 0xABCDEF01u : 0xABCDEF00u, 0u };
        Kernel k(spv, "sb_solveOutputClothAttachmentDeltaVLaunch", {}, sizeof(CaSolvePush));
        k.dispatch(GRID, {}, sp);
    }
    else
    {
        SaSolvePush sp = { S.sbPool, aBlocks, NA, kDt, isVelocityIteration ? 0xABCDEF01u : 0xABCDEF00u, 0u };
        Kernel k(spv, "sb_solveOutputSoftBodyAttachmentDeltaVLaunch", {}, sizeof(SaSolvePush));
        k.dispatch(GRID, {}, sp);
    }
    dev->flush();
    uint32_t nSkipped = 0, nSolved = 0, nSelf = 0;
    for (const DefAttachment& d : as)
    {
        DbPart db0, db1;
        const V3 pos0 = db0.readSoftBodyAttachmentDbDb(S.sb[refId(d.pair.ref0)], d.pair.ref0, d.bOrP0, kDt, isVelocityIteration);
        const V3 pos1 = toCloth ? db1.readClothAttachmentDbDb(S.cloth[1], d.pair.ref1, d.bary1, kDt, isVelocityIteration)
                                : db1.readSoftBodyAttachmentDbDb(S.sb[refId(d.pair.ref1)], d.pair.ref1, d.bary1, kDt,
                                                                 isVelocityIteration);
        V3 delta;
        if (solveDbDbAttachment(db0, db1, pos0, pos1, delta))
        {
            ++nSolved;
            if (toCloth) db1.writeCloth(S.cloth[1], neg(delta));
            else db1.writeSoftBody(S.sb[refId(d.pair.ref1)], neg(delta));
            db0.writeSoftBody(S.sb[refId(d.pair.ref0)], delta);
        }
        else
            ++nSkipped;
        if (!toCloth && refId(d.pair.ref0) == refId(d.pair.ref1)) ++nSelf;
    }
    bad = 0;
    for (uint32_t s = 0; s < 3; s += 2)
        cmpVec("solve mSimDelta", S.readSoftBodyDelta(s), S.sb[s].delta, bad);
    if (toCloth)
        cmpVec("solve cloth mDeltaPos", S.readClothDelta(), S.cloth[1].delta, bad);
    PXG_CHECK_MSG(bad == 0, "%s solve (%s): %u mismatches", toCloth ? "cloth attachment" : "softbody attachment",
                  isVelocityIteration ? "vel" : "pos", bad);
    PXG_CHECK_MSG(nSolved > 0 && nSkipped > 0 && (toCloth || nSelf > 0), "coverage: solved %u skipped %u self %u",
                  nSolved, nSkipped, nSelf);
}

PXG_TEST(softBodySolve512SoftAttachPos) { runDbAttachQuerySolve(false, false, 71u); }
PXG_TEST(softBodySolve512SoftAttachVel) { runDbAttachQuerySolve(false, true, 72u); }
PXG_TEST(softBodySolve512ClothAttachPos) { runDbAttachQuerySolve(true, false, 73u); }
PXG_TEST(softBodySolve512ClothAttachVel) { runDbAttachQuerySolve(true, true, 74u); }

} // namespace
