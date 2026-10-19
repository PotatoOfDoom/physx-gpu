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

// pxgSoftBodySolveRef.h -- the host oracle of the PhysX 5.12 softbody (deformable volume)
// constraint kernels, transcribed from the CUDA, NOT from the Slang port:
//
//   gpusimulationcontroller/src/CUDA/deformableAndParticleUtils.cuh
//       bumpDbRefCountVtx/Tri/Tet, bumpClothRefCount, bumpSoftBodyRefCount,
//       updateDeltaVtx/Tri/Tet, PxgDeformablePart<PxVec3/PxVec4>::readCloth / readSoftBody /
//       readSoftBodyAttachment / readClothAttachment / readSoftBodyAttachmentDbDb /
//       readClothAttachmentDbDb / writeCloth / writeSoftBody, solveDbDbContact,
//       solveRigidDbAttachment, solveDbDbAttachment, PxgRigidPart::readAttachmentPrep /
//       writeAttachmentDeltas
//   gpusimulationcontroller/src/CUDA/deformableUtils.cuh
//       barycentricProjectTet, prepareRigidDbContact, prepareRigidDbAttachment
//
// It builds on pxgParticleSolveRef.h (V3/F4, the PxVec3 spellings, solveRigidDbContact,
// PxgRigidPart, the velocity reader).  PxgDeformablePart<Vec> is one struct here with `dim`
// (3 = PxVec3, 4 = PxVec4); every reduction over bc / vertexInvMasses runs over exactly `dim`
// components in the PxVec3 / PxVec4 dot order.
//
// THE SCATTERS.  The kernels add into mSimDelta / mDeltaPos with float atomics, whose order
// across threads is not fixed.  The scenes keep every vertex at <= 2 contributions per
// dispatch onto a +0-seeded .xyz, where IEEE addition is order independent; the oracle adds
// them in contact order.  The .w reference counts are integer counts in floats, exact in any
// order.

#ifndef PXG_SOFTBODY_SOLVE_REF_H
#define PXG_SOFTBODY_SOLVE_REF_H

#include "pxgParticleSolveRef.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace pxgsbref
{
using namespace pxgpsref;

struct U4 { uint32_t x, y, z, w; };

// ---------------------------------------------------------------------------
// PxgDeformableRef (PxgSimulationCoreDesc.h, 5.12)
// ---------------------------------------------------------------------------
static const uint32_t REF_eVERTEX = 0u, REF_eTRIANGLE = 1u, REF_eTETRAHEDRON = 2u;
static const uint64_t REF_INVALID = uint64_t(1) << 32;
static const uint64_t REF_ACTIVE = uint64_t(1) << 33;
inline uint64_t encodeRef(uint32_t deformableId, uint32_t type, uint32_t featureId)
{
    return (uint64_t(deformableId) << 34) | (uint64_t(type) << 29) | uint64_t(featureId);
}
inline uint32_t refId(uint64_t r) { return uint32_t(r >> 34); }
inline uint32_t refType(uint64_t r) { return (uint32_t(r) >> 29) & 7u; }
inline uint32_t refFeature(uint64_t r) { return uint32_t(r) & 0x07ffffffu; }
inline bool pairValid(uint64_t ref1) { return (ref1 & REF_INVALID) == 0; }
inline bool pairActive(uint64_t ref1) { return (ref1 & REF_ACTIVE) != 0; }
inline uint64_t setActive(uint64_t ref1, bool a) { return a ? (ref1 | REF_ACTIVE) : (ref1 & ~REF_ACTIVE); }

inline F4 f4(float x, float y, float z, float w) { F4 r = { x, y, z, w }; return r; }
inline F4 add4(F4 a, F4 b) { return f4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
inline F4 mul4(F4 a, float s) { return f4(a.x * s, a.y * s, a.z * s, a.w * s); }

// barycentricProjectTet / Tri: ((a * b.x + b * b.y) + c * b.z) [+ d * b.w].
inline F4 projectTet(const std::vector<F4>& buf, U4 t, F4 b)
{
    return add4(add4(add4(mul4(buf[t.x], b.x), mul4(buf[t.y], b.y)), mul4(buf[t.z], b.z)), mul4(buf[t.w], b.w));
}
inline F4 projectTri(const std::vector<F4>& buf, U4 t, F4 b)
{
    return add4(add4(mul4(buf[t.x], b.x), mul4(buf[t.y], b.y)), mul4(buf[t.z], b.z));
}

// ---------------------------------------------------------------------------
// Host deformables.  Each field is a whole per-vertex / per-element array of the descriptor.
// ---------------------------------------------------------------------------
struct HostSoftBody
{
    std::vector<U4> tets;                 // mSimTetIndices
    std::vector<F4> pos, vel, deltaPos;   // mSimPosition_InvMass, mSimVelocity_InvMass, mSimDeltaPos
    std::vector<F4> delta;                // mSimDelta (.w = refCount)
    std::vector<uint16_t> matIdx;         // mMaterialIndices (per sim tet)
    float restOffset = 0.0f, penBiasClamp = 0.0f;
};

struct HostCloth
{
    std::vector<U4> tris;                       // mTriangleVertexIndices_ownedEdges
    std::vector<F4> pos, vel, accDelta;         // mPosition_InvMass, mVelocity_InvMass, mAccumulatedDeltaPos
    std::vector<F4> delta;                      // mDeltaPos (.w = refCount)
    std::vector<float> dynFriction;             // mDynamicFrictions (per vertex)
    std::vector<uint16_t> matIdx;               // mMaterialIndices (per triangle)
    float restOffset = 0.0f, penBiasClamp = 0.0f;
    // Read by the cloth-cloth kernels only (testFemClothSolve512).
    std::vector<F4> restPos, prevPos;           // mRestPosition, mPrevPositionInContactOffset
    float selfCollisionFilterDistance = 0.0f;   // mSelfCollisionFilterDistance
};

// ---------------------------------------------------------------------------
// Reference counts and scatters
// ---------------------------------------------------------------------------
inline void bumpTet(std::vector<F4>& d, U4 t, float w0, float w1, float w2, float w3)
{
    if (std::fabs(w0) > 1e-3f) d[t.x].w += 1.0f;
    if (std::fabs(w1) > 1e-3f) d[t.y].w += 1.0f;
    if (std::fabs(w2) > 1e-3f) d[t.z].w += 1.0f;
    if (std::fabs(w3) > 1e-3f) d[t.w].w += 1.0f;
}
inline void bumpTri(std::vector<F4>& d, U4 t, float w0, float w1, float w2)
{
    if (std::fabs(w0) > 1e-3f) d[t.x].w += 1.0f;
    if (std::fabs(w1) > 1e-3f) d[t.y].w += 1.0f;
    if (std::fabs(w2) > 1e-3f) d[t.z].w += 1.0f;
}
inline void bumpSoftBody(HostSoftBody& sb, uint64_t ref, F4 w)
{
    if (refType(ref) == REF_eVERTEX)
        sb.delta[refFeature(ref)].w += 1.0f;
    else
        bumpTet(sb.delta, sb.tets[refFeature(ref)], w.x, w.y, w.z, w.w);
}
inline void bumpCloth(HostCloth& c, uint64_t ref, F4 w)
{
    if (refType(ref) == REF_eVERTEX)
        c.delta[refFeature(ref)].w += 1.0f;
    else
        bumpTri(c.delta, c.tris[refFeature(ref)], w.x, w.y, w.z);
}

// AtomicAdd(float4&, PxVec3): x, y, z.
inline void addXYZ(F4& d, V3 v) { d.x += v.x; d.y += v.y; d.z += v.z; }
inline void updateVtx(std::vector<F4>& out, uint32_t i, V3 delta, float invMass)
{
    if (invMass > 0.0f) addXYZ(out[i], mul(delta, invMass));
}
inline void updateTet(std::vector<F4>& out, U4 t, V3 delta, F4 m)
{
    if (m.x > 0.0f) addXYZ(out[t.x], mul(delta, m.x));
    if (m.y > 0.0f) addXYZ(out[t.y], mul(delta, m.y));
    if (m.z > 0.0f) addXYZ(out[t.z], mul(delta, m.z));
    if (m.w > 0.0f) addXYZ(out[t.w], mul(delta, m.w));
}
inline void updateTri(std::vector<F4>& out, U4 t, V3 delta, F4 m)
{
    if (m.x > 0.0f) addXYZ(out[t.x], mul(delta, m.x));
    if (m.y > 0.0f) addXYZ(out[t.y], mul(delta, m.y));
    if (m.z > 0.0f) addXYZ(out[t.z], mul(delta, m.z));
}

// ---------------------------------------------------------------------------
// PxgDeformablePart<Vec>
// ---------------------------------------------------------------------------
struct DbPart
{
    int dim = 3;
    float friction = 0.0f;
    V3 linDelta = { 0, 0, 0 }, posDelta = { 0, 0, 0 };
    F4 vim = { 0, 0, 0, 0 };   // vertexInvMasses
    F4 bc = { 0, 0, 0, 0 };
    uint64_t ref = 0;
    float attachPointInvMass = 0.0f, attachPointInvMassSplit = 0.0f;

    // bc.multiply(bc).dot(vertexInvMasses)
    float bcResponse() const
    {
        if (dim == 4)
            return (bc.x * bc.x) * vim.x + (bc.y * bc.y) * vim.y + (bc.z * bc.z) * vim.z + (bc.w * bc.w) * vim.w;
        return (bc.x * bc.x) * vim.x + (bc.y * bc.y) * vim.y + (bc.z * bc.z) * vim.z;
    }

    // materialsFriction: the per-material dynamicFriction table (nullptr = NULL materials).
    void readSoftBody(const HostSoftBody& sb, uint64_t r, F4 bary, const float* materialsFriction,
                      bool checkOnlyActivity, float dt, bool isVelocityIteration)
    {
        dim = 4;
        ref = r;
        const uint32_t tetId = refFeature(r);
        bc = bary;
        const std::vector<F4>& pdvBuf = isVelocityIteration ? sb.vel : sb.deltaPos;
        const U4 t = sb.tets[tetId];
        const F4 p0 = pdvBuf[t.x], p1 = pdvBuf[t.y], p2 = pdvBuf[t.z], p3 = pdvBuf[t.w];
        const F4 pdv = add4(add4(add4(mul4(p0, bc.x), mul4(p1, bc.y)), mul4(p2, bc.z)), mul4(p3, bc.w));
        linDelta = v3(pdv.x, pdv.y, pdv.z);
        if (isVelocityIteration)
        {
            linDelta = mul(linDelta, dt);
            const F4 acc = projectTet(sb.deltaPos, t, bc);
            posDelta = v3(acc.x, acc.y, acc.z);
        }
        else
            posDelta = linDelta;
        vim = f4(p0.w, p1.w, p2.w, p3.w);
        if (!checkOnlyActivity)
        {
            const uint16_t gm = sb.matIdx[tetId];
            friction = materialsFriction ? materialsFriction[gm] : 0.0f;
            vim = f4(vim.x * sb.delta[t.x].w, vim.y * sb.delta[t.y].w, vim.z * sb.delta[t.z].w, vim.w * sb.delta[t.w].w);
        }
    }

    void readCloth(const HostCloth& c, uint64_t r, F4 bary, const float* materialsFriction,
                   bool countReferenceOnly, float dt, bool isVelocityIteration)
    {
        dim = 3;
        ref = r;
        const uint32_t e = refFeature(r);
        const std::vector<F4>& pdvBuf = isVelocityIteration ? c.vel : c.accDelta;
        F4 pdv;
        if (refType(r) != REF_eVERTEX)
        {
            bc = f4(bary.x, bary.y, bary.z, 0.0f);
            const U4 t = c.tris[e];
            const F4 p0 = pdvBuf[t.x], p1 = pdvBuf[t.y], p2 = pdvBuf[t.z];
            pdv = add4(add4(mul4(p0, bc.x), mul4(p1, bc.y)), mul4(p2, bc.z));
            vim = f4(p0.w, p1.w, p2.w, 0.0f);
            if (isVelocityIteration)
            {
                const F4 acc = add4(add4(mul4(c.accDelta[t.x], bc.x), mul4(c.accDelta[t.y], bc.y)), mul4(c.accDelta[t.z], bc.z));
                posDelta = v3(acc.x, acc.y, acc.z);
            }
            if (!countReferenceOnly)
            {
                friction = materialsFriction ? materialsFriction[c.matIdx[e]] : 0.0f;
                vim = f4(vim.x * c.delta[t.x].w, vim.y * c.delta[t.y].w, vim.z * c.delta[t.z].w, 0.0f);
            }
        }
        else
        {
            bc = f4(1.0f, 0.0f, 0.0f, 0.0f);
            pdv = pdvBuf[e];
            vim = f4(pdv.w, 0.0f, 0.0f, 0.0f);
            if (isVelocityIteration)
                posDelta = xyz(c.accDelta[e]);
            if (!countReferenceOnly)
            {
                friction = materialsFriction ? c.dynFriction[e] : 0.0f;
                vim.x *= c.delta[e].w;
            }
        }
        linDelta = v3(pdv.x, pdv.y, pdv.z);
        if (isVelocityIteration)
            linDelta = mul(linDelta, dt);
        else
            posDelta = linDelta;
    }

    void readParticle(const ParticleView& p, bool hasMaterial, bool inflateByRefCount, float dt, bool isVelocityIteration)
    {
        DeformablePart d;
        d.readParticle(p, hasMaterial, inflateByRefCount, dt, isVelocityIteration);
        dim = 3;
        friction = d.friction;
        linDelta = d.linDelta;
        posDelta = d.posDelta;
        vim = f4(d.vertexInvMasses.x, d.vertexInvMasses.y, d.vertexInvMasses.z, 0.0f);
        bc = f4(d.bc.x, d.bc.y, d.bc.z, 0.0f);
    }

    void readSoftBodyAttachment(const HostSoftBody& sb, uint64_t r, F4 bary, bool isTGS, float dt, bool isVelocityIteration)
    {
        dim = 4;
        ref = r;
        const uint32_t e = refFeature(r);
        const std::vector<F4>& src = (isTGS && !isVelocityIteration) ? sb.deltaPos : sb.vel;
        if (refType(r) == REF_eVERTEX)
        {
            const F4 p = src[e];
            const float rc = sb.delta[e].w;
            linDelta = xyz(p);
            vim = f4(p.w * rc, 0.0f, 0.0f, 0.0f);
            attachPointInvMass = p.w;
            attachPointInvMassSplit = p.w * rc;
            bc = f4(0, 0, 0, 0);
        }
        else
        {
            const U4 t = sb.tets[e];
            const F4 p0 = src[t.x], p1 = src[t.y], p2 = src[t.z], p3 = src[t.w];
            const F4 p = add4(add4(add4(mul4(p0, bary.x), mul4(p1, bary.y)), mul4(p2, bary.z)), mul4(p3, bary.w));
            linDelta = xyz(p);
            const float r0 = sb.delta[t.x].w, r1 = sb.delta[t.y].w, r2 = sb.delta[t.z].w, r3 = sb.delta[t.w].w;
            vim = f4(p0.w * r0, p1.w * r1, p2.w * r2, p3.w * r3);
            const float bx2 = bary.x * bary.x, by2 = bary.y * bary.y, bz2 = bary.z * bary.z, bw2 = bary.w * bary.w;
            attachPointInvMass = bx2 * p0.w + by2 * p1.w + bz2 * p2.w + bw2 * p3.w;
            attachPointInvMassSplit = bx2 * p0.w * r0 + by2 * p1.w * r1 + bz2 * p2.w * r2 + bw2 * p3.w * r3;
            bc = bary;
        }
        if (isTGS && isVelocityIteration)
            linDelta = mul(linDelta, dt);
    }

    V3 readSoftBodyAttachmentDbDb(const HostSoftBody& sb, uint64_t r, F4 bary, float dt, bool isVelocityIteration)
    {
        dim = 4;
        ref = r;
        const uint32_t e = refFeature(r);
        const std::vector<F4>& pv = isVelocityIteration ? sb.vel : sb.pos;
        if (refType(r) == REF_eVERTEX)
        {
            const F4 p = pv[e];
            const float rc = sb.delta[e].w;
            vim = f4(p.w * rc, 0, 0, 0);
            bc = f4(0, 0, 0, 0);
            attachPointInvMassSplit = p.w * rc;
            V3 v = xyz(p);
            if (isVelocityIteration) v = mul(v, dt);
            return v;
        }
        const U4 t = sb.tets[e];
        const F4 p0 = pv[t.x], p1 = pv[t.y], p2 = pv[t.z], p3 = pv[t.w];
        const F4 p = add4(add4(add4(mul4(p0, bary.x), mul4(p1, bary.y)), mul4(p2, bary.z)), mul4(p3, bary.w));
        const float r0 = sb.delta[t.x].w, r1 = sb.delta[t.y].w, r2 = sb.delta[t.z].w, r3 = sb.delta[t.w].w;
        vim = f4(p0.w * r0, p1.w * r1, p2.w * r2, p3.w * r3);
        bc = bary;
        const float bx2 = bary.x * bary.x, by2 = bary.y * bary.y, bz2 = bary.z * bary.z, bw2 = bary.w * bary.w;
        attachPointInvMassSplit = bx2 * p0.w * r0 + by2 * p1.w * r1 + bz2 * p2.w * r2 + bw2 * p3.w * r3;
        V3 v = xyz(p);
        if (isVelocityIteration) v = mul(v, dt);
        return v;
    }

    V3 readClothAttachmentDbDb(const HostCloth& c, uint64_t r, F4 bary, float dt, bool isVelocityIteration)
    {
        dim = 3;
        ref = r;
        const uint32_t e = refFeature(r);
        const std::vector<F4>& pv = isVelocityIteration ? c.vel : c.pos;
        if (refType(r) == REF_eVERTEX)
        {
            const F4 p = pv[e];
            const float rc = c.delta[e].w;
            vim = f4(p.w * rc, 0, 0, 0);
            bc = f4(0, 0, 0, 0);
            attachPointInvMassSplit = p.w * rc;
            V3 v = xyz(p);
            if (isVelocityIteration) v = mul(v, dt);
            return v;
        }
        const U4 t = c.tris[e];
        const F4 p0 = pv[t.x], p1 = pv[t.y], p2 = pv[t.z];
        const F4 p = add4(add4(mul4(p0, bary.x), mul4(p1, bary.y)), mul4(p2, bary.z));
        const float r0 = c.delta[t.x].w, r1 = c.delta[t.y].w, r2 = c.delta[t.z].w;
        vim = f4(p0.w * r0, p1.w * r1, p2.w * r2, 0.0f);
        bc = f4(bary.x, bary.y, bary.z, 0.0f);
        const float bx2 = bary.x * bary.x, by2 = bary.y * bary.y, bz2 = bary.z * bary.z;
        attachPointInvMassSplit = bx2 * p0.w * r0 + by2 * p1.w * r1 + bz2 * p2.w * r2;
        V3 v = xyz(p);
        if (isVelocityIteration) v = mul(v, dt);
        return v;
    }

    // readClothAttachment (RB-DB attachments): PGS reads the velocity, TGS the accumulated
    // delta (the velocity in TGS velocity iterations, scaled by dt).
    void readClothAttachment(const HostCloth& c, uint64_t r, F4 bary, bool isTGS, float dt, bool isVelocityIteration)
    {
        dim = 3;
        ref = r;
        const uint32_t e = refFeature(r);
        const std::vector<F4>& src = (isTGS && !isVelocityIteration) ? c.accDelta : c.vel;
        if (refType(r) == REF_eVERTEX)
        {
            const F4 p = src[e];
            const float rc = c.delta[e].w;
            linDelta = xyz(p);
            vim = f4(p.w * rc, 0.0f, 0.0f, 0.0f);
            attachPointInvMass = p.w;
            attachPointInvMassSplit = p.w * rc;
            bc = f4(0, 0, 0, 0);
        }
        else
        {
            const U4 t = c.tris[e];
            const F4 p0 = src[t.x], p1 = src[t.y], p2 = src[t.z];
            const F4 p = add4(add4(mul4(p0, bary.x), mul4(p1, bary.y)), mul4(p2, bary.z));
            linDelta = xyz(p);
            const float r0 = c.delta[t.x].w, r1 = c.delta[t.y].w, r2 = c.delta[t.z].w;
            vim = f4(p0.w * r0, p1.w * r1, p2.w * r2, 0.0f);
            const float bx2 = bary.x * bary.x, by2 = bary.y * bary.y, bz2 = bary.z * bary.z;
            attachPointInvMass = bx2 * p0.w + by2 * p1.w + bz2 * p2.w;
            attachPointInvMassSplit = bx2 * p0.w * r0 + by2 * p1.w * r1 + bz2 * p2.w * r2;
            bc = f4(bary.x, bary.y, bary.z, 0.0f);
        }
        if (isTGS && isVelocityIteration)
            linDelta = mul(linDelta, dt);
    }

    // writeClothEdge: the cloth-cloth EE scatter, updateDeltaEdge with vim.x * bc.x / vim.y * bc.y.
    void writeClothEdge(HostCloth& c, uint32_t v0, uint32_t v1, V3 delta) const
    {
        updateVtx(c.delta, v0, delta, vim.x * bc.x);
        updateVtx(c.delta, v1, delta, vim.y * bc.y);
    }

    void writeSoftBody(HostSoftBody& sb, V3 delta) const
    {
        const uint32_t e = refFeature(ref);
        if (refType(ref) == REF_eVERTEX)
            updateVtx(sb.delta, e, delta, vim.x);
        else
            updateTet(sb.delta, sb.tets[e], delta, f4(vim.x * bc.x, vim.y * bc.y, vim.z * bc.z, vim.w * bc.w));
    }

    void writeCloth(HostCloth& c, V3 delta) const
    {
        const uint32_t e = refFeature(ref);
        if (refType(ref) == REF_eVERTEX)
            updateVtx(c.delta, e, delta, vim.x);
        else
            updateTri(c.delta, c.tris[e], delta, f4(vim.x * bc.x, vim.y * bc.y, vim.z * bc.z, 0.0f));
    }

    F4 writeParticle(V3 delta, const SolveOut& so) const
    {
        const V3 s = mul(delta, vim.x);
        const float applied = (so.deltaLambdaN != 0.0f || so.deltaLambdaT != 0.0f) ? 1.0f : 0.0f;
        return f4(s.x, s.y, s.z, applied);
    }

    // The pxgParticleSolveRef view, for solveRigidDbContact (dim is irrelevant there except
    // for the bc response, which the 4-wide tet form must keep).
};

// solveRigidDbContact for a tet participant: the particle oracle's form with the 4-wide bc
// response (pxgParticleSolveRef's DeformablePart is PxVec3).
inline bool solveRigidDbContactDb(const RigidPart& rigid, const DbPart& db, const ContactPair& cp, SolveOut& so,
                                  float appliedForceRef, float appliedTanLambdaRef, float dt, bool wasActive,
                                  bool checkOnlyActivity, bool isVelocityIteration)
{
    so.accumulatedDeltaLambdaN = appliedForceRef;
    const float appliedTanRef = appliedTanLambdaRef;
    const float threshold = 1.0e-14f;
    const float invDt = 1.0f / dt;

    const float separation = cp.initPen + dot(rigid.linDelta, cp.normal) + dot(rigid.angDelta, rigid.raXn)
                             - dot(db.posDelta, cp.normal);

    const V3 rigidLinDelta = isVelocityIteration ? v3(0.0f, 0.0f, 0.0f) : rigid.linDelta;
    const V3 rigidAngDelta = isVelocityIteration ? v3(0.0f, 0.0f, 0.0f) : rigid.angDelta;
    const V3 relLinDelta = sub(rigidLinDelta, db.linDelta);
    const float normalVel = dot(rigid.linVel, cp.normal) + dot(rigid.angVel, rigid.raXn);

    float CN;
    if (isVelocityIteration)
        CN = normalVel + (selectMax(0.0f, separation) + dot(relLinDelta, cp.normal)) * invDt;
    else
        CN = normalVel + selectMax(cp.maxPenBiasClamp, separation * invDt);

    const bool isActive = wasActive || CN < 0.0f;
    if (checkOnlyActivity || !isActive)
        return isActive;

    const float deformableInvMass_massSplitting = db.bcResponse();
    const float rigidRefCount = float(rigid.referenceCount);
    const float unitResponse = rigidRefCount * rigid.raXnResp + deformableInvMass_massSplitting;
    const float invDenom = (unitResponse > 0.0f) ? (1.0f / unitResponse) : 0.0f;

    so.deltaLambdaN = selectMax(-CN * invDenom, -so.accumulatedDeltaLambdaN);
    so.accumulatedDeltaLambdaN += so.deltaLambdaN;

    const V3 fric0 = cp.tangent0;
    const V3 fric1 = cross(cp.normal, fric0);
    const float tanVel0 = dot(rigid.linVel, fric0) + dot(rigid.angVel, rigid.raXnF0);
    const float tanVel1 = dot(rigid.linVel, fric1) + dot(rigid.angVel, rigid.raXnF1);
    const float CT0 = (dot(fric0, relLinDelta) + dot(rigidAngDelta, rigid.raXnF0)) * invDt + tanVel0;
    const float CT1 = (dot(fric1, relLinDelta) + dot(rigidAngDelta, rigid.raXnF1)) * invDt + tanVel1;
    const V3 relTanDelta = add(mul(fric0, CT0), mul(fric1, CT1));
    const float tanMagSq = dot(relTanDelta, relTanDelta);

    if (tanMagSq > threshold)
    {
        const float CT = std::sqrt(tanMagSq);
        const float invTanMag = 1.0f / CT;
        so.tangent = mul(relTanDelta, invTanMag);
        const float frac0 = dot(so.tangent, fric0);
        const float frac1 = dot(so.tangent, fric1);
        so.raXt = add(mul(rigid.raXnF0, frac0), mul(rigid.raXnF1, frac1));
        const float unitResponseT0 = rigidRefCount * rigid.raXnF0Resp + deformableInvMass_massSplitting;
        const float invTanDenom0 = (unitResponseT0 > 0.0f) ? (1.0f / unitResponseT0) : 0.0f;
        const float unitResponseT1 = rigidRefCount * rigid.raXnF1Resp + deformableInvMass_massSplitting;
        const float invTanDenom1 = (unitResponseT1 > 0.0f) ? (1.0f / unitResponseT1) : 0.0f;
        const float deltaLambdaT0 = CT0 * invTanDenom0;
        const float deltaLambdaT1 = CT1 * invTanDenom1;
        const float combinedFriction = combineScalars(rigid.friction, db.friction, rigid.fricCombineMode);
        const float requiredLambdaTNeg = -std::sqrt(deltaLambdaT0 * deltaLambdaT0 + deltaLambdaT1 * deltaLambdaT1);
        const float frictionLimit = combinedFriction * so.accumulatedDeltaLambdaN;
        const float targetAccumLambdaT = selectMax(appliedTanRef + requiredLambdaTNeg, -frictionLimit);
        so.deltaLambdaT = targetAccumLambdaT - appliedTanRef;
    }

    return so.deltaLambdaN != 0.0f || so.deltaLambdaT != 0.0f;
}

// ContactPair::readContactPrepDbDb (normal_pen + the block's clamp).
inline ContactPair contactPairDbDb(F4 normalPen, float maxPenBiasClamp)
{
    ContactPair cp;
    cp.normal = xyz(normalPen);
    cp.initPen = normalPen.w;
    cp.maxPenBiasClamp = maxPenBiasClamp;
    return cp;
}

inline bool solveDbDbContact(const DbPart& db0, const DbPart& db1, const ContactPair& cp, SolveOut& so,
                             float appliedNormalLambdaRef, float appliedTanLambdaRef, float frictionCoefficient,
                             float dt, bool checkOnlyActivity, bool isVelocityIteration)
{
    so.accumulatedDeltaLambdaN = appliedNormalLambdaRef;
    const float appliedTanRef = appliedTanLambdaRef;
    const float invDt = 1.0f / dt;

    const V3 relLinDelta = sub(db1.linDelta, db0.linDelta);
    const float separation = cp.initPen + dot(sub(db1.posDelta, db0.posDelta), cp.normal);

    float CN;
    if (isVelocityIteration)
        CN = (selectMax(0.0f, separation) + dot(relLinDelta, cp.normal)) * invDt;
    else
        CN = selectMax(cp.maxPenBiasClamp, separation * invDt);

    const bool isActive = CN < 0.0f;
    if (checkOnlyActivity || !isActive)
        return isActive;

    const float db0Resp = db0.bcResponse();
    const float db1Resp = db1.bcResponse();
    const float unitResponse = db0Resp + db1Resp;
    const float invDenom = (unitResponse > 1e-16f) ? (1.0f / unitResponse) : 0.0f;

    so.deltaLambdaN = selectMax(-CN * invDenom, -so.accumulatedDeltaLambdaN);
    so.accumulatedDeltaLambdaN += so.deltaLambdaN;

    const float normalProj = dot(relLinDelta, cp.normal);
    V3 tanDir = sub(relLinDelta, mul(cp.normal, normalProj));
    const float tanDelta = normalize(tanDir);
    so.tangent = tanDir;

    const float requiredLambdaTNeg = -tanDelta * invDt * invDenom;
    const float frictionLimit = so.accumulatedDeltaLambdaN * frictionCoefficient;
    const float targetAccumLambdaT = selectMax(appliedTanRef + requiredLambdaTNeg, -frictionLimit);
    so.deltaLambdaT = targetAccumLambdaT - appliedTanRef;

    return so.deltaLambdaN != 0.0f || so.deltaLambdaT != 0.0f;
}

// ---------------------------------------------------------------------------
// Attachments
// ---------------------------------------------------------------------------
struct RdbAttachLane
{
    F4 barycentric, raXn0_biasW, raXn1_biasW, raXn2_biasW, velMultiplierXYZ_invMassW;
    uint64_t deformableRef, rigidId;
    uint32_t rigidBodyRefCount;
};

struct AttachRigid
{
    V3 raXn0, raXn1, raXn2, positionErrorXYZ, velMultiplierXYZ;
    float attachInvMass = 0.0f;
    uint32_t attachRefCount = 1;
    V3 linVel = { 0, 0, 0 }, angVel = { 0, 0, 0 }, linDelta = { 0, 0, 0 }, angDelta = { 0, 0, 0 };

    void readAttachmentPrep(const RdbAttachLane& b)
    {
        raXn0 = xyz(b.raXn0_biasW);
        raXn1 = xyz(b.raXn1_biasW);
        raXn2 = xyz(b.raXn2_biasW);
        positionErrorXYZ = v3(b.raXn0_biasW.w, b.raXn1_biasW.w, b.raXn2_biasW.w);
        velMultiplierXYZ = xyz(b.velMultiplierXYZ_invMassW);
        attachInvMass = b.velMultiplierXYZ_invMassW.w;
        attachRefCount = b.rigidBodyRefCount;
    }

    void readVelocity(const VelocityReader& r, uint64_t rigidId, bool isTGS)
    {
        if (isTGS) r.readTGS(rigidId, linVel, angVel, linDelta, angDelta);
        else r.readPGS(rigidId, linVel, angVel, linDelta, angDelta);
    }

    // writeAttachmentDeltas: returns false for a static rigid (nothing written).
    bool writeAttachmentDeltas(V3 dLin, V3 dAng, F4& linOut, F4& angOut, uint64_t rigidId) const
    {
        if (nodeIsStatic(rigidId))
            return false;
        if (attachInvMass == 0.0f)
        {
            linOut = f4(0, 0, 0, 0);
            angOut = f4(0, 0, 0, 0);
        }
        else
        {
            const float refN = float(attachRefCount);
            const V3 l = mul(dLin, refN), a = mul(dAng, refN);
            linOut = f4(l.x, l.y, l.z, 1.0f);
            angOut = f4(a.x, a.y, a.z, 0.0f);
        }
        return true;
    }
};

inline V3 solveRigidDbAttachment(const AttachRigid& rigid, const DbPart& db, float dt, float biasCoefficient,
                                 bool isTGS, bool isVelocityIteration, V3& outDeltaLinVel, V3& outDeltaAngVel)
{
    const float invDt = 1.0f / dt;
    const float vX = rigid.linVel.x + dot(rigid.angVel, rigid.raXn0);
    const float vY = rigid.linVel.y + dot(rigid.angVel, rigid.raXn1);
    const float vZ = rigid.linVel.z + dot(rigid.angVel, rigid.raXn2);
    const float refN = float(rigid.attachRefCount);
    const float denomBias = refN * db.attachPointInvMass - db.attachPointInvMassSplit;
    const float adj0 = 1.0f / (refN / rigid.velMultiplierXYZ.x - denomBias);
    const float adj1 = 1.0f / (refN / rigid.velMultiplierXYZ.y - denomBias);
    const float adj2 = 1.0f / (refN / rigid.velMultiplierXYZ.z - denomBias);
    float d0, d1, d2;
    if (isTGS)
    {
        const V3 rl = isVelocityIteration ? v3(0, 0, 0) : rigid.linDelta;
        const V3 ra = isVelocityIteration ? v3(0, 0, 0) : rigid.angDelta;
        const V3 ld = sub(db.linDelta, rl);
        const float ex = (ld.x - dot(ra, rigid.raXn0) - rigid.positionErrorXYZ.x * biasCoefficient) * invDt;
        const float ey = (ld.y - dot(ra, rigid.raXn1) - rigid.positionErrorXYZ.y * biasCoefficient) * invDt;
        const float ez = (ld.z - dot(ra, rigid.raXn2) - rigid.positionErrorXYZ.z * biasCoefficient) * invDt;
        d0 = (ex - vX) * adj0;
        d1 = (ey - vY) * adj1;
        d2 = (ez - vZ) * adj2;
    }
    else
    {
        const float ex = db.linDelta.x - rigid.positionErrorXYZ.x * biasCoefficient * invDt;
        const float ey = db.linDelta.y - rigid.positionErrorXYZ.y * biasCoefficient * invDt;
        const float ez = db.linDelta.z - rigid.positionErrorXYZ.z * biasCoefficient * invDt;
        d0 = (ex - vX) * adj0;
        d1 = (ey - vY) * adj1;
        d2 = (ez - vZ) * adj2;
    }
    const V3 imp = v3(d0, d1, d2);
    outDeltaLinVel = mul(imp, rigid.attachInvMass);
    outDeltaAngVel = add(add(mul(rigid.raXn0, d0), mul(rigid.raXn1, d1)), mul(rigid.raXn2, d2));
    return imp;
}

inline bool solveDbDbAttachment(const DbPart& db0, const DbPart& db1, V3 pos0, V3 pos1, V3& outDelta)
{
    const V3 error = sub(pos1, pos0);
    const float wTot = db0.attachPointInvMassSplit + db1.attachPointInvMassSplit;
    if (wTot > 0.0f)
    {
        outDelta = mul(error, 1.0f / wTot);
        return true;
    }
    outDelta = v3(0, 0, 0);
    return false;
}

// PxQuat::rotate.
inline V3 quatRotate(F4 q, V3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return v3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
              vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
              vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}

inline bool bitsEqualV(const std::vector<F4>& a, const std::vector<F4>& b, size_t& bad, const char* what)
{
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        if (!bitsEqual(a[i], b[i]))
        {
            if (++bad <= 6)
                std::printf("    %s[%zu] got (%a %a %a %a) want (%a %a %a %a)\n", what, i, a[i].x, a[i].y, a[i].z,
                            a[i].w, b[i].x, b[i].y, b[i].z, b[i].w);
        }
    return bad == 0;
}

} // namespace pxgsbref

#endif // PXG_SOFTBODY_SOLVE_REF_H
